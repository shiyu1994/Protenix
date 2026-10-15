# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import hashlib
import json
import logging
import math
import os
import random
import socket
import subprocess
import sys
import time
import traceback
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Sequence, Tuple


def configure_rank_local_triton_cache() -> str | None:
    """Use one local Triton cache per distributed rank when requested."""
    cache_root = os.environ.get("PROTENIX_TRITON_CACHE_ROOT", "").strip()
    if not cache_root:
        return None
    rank_text = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        rank = int(rank_text)
    except ValueError as exc:
        raise ValueError(f"Invalid distributed rank for Triton cache: {rank_text!r}") from exc
    cache_dir = os.path.join(os.path.abspath(cache_root), f"rank_{rank}")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    return cache_dir


# This must run before importing torch/Protenix modules that may initialize
# Triton. Shared-network cache directories can fail under concurrent writers.
configure_rank_local_triton_cache()

import torch
import torch.distributed as dist
import wandb

from runner.wandb_logging import define_training_metrics, log_training_metrics
import numpy as np

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_model_type import model_configs
from protenix.config.config import parse_configs, parse_sys_args, save_config
from protenix.data.msa.msa_featurizer import (
    MULTICHAIN_MSA_SHUFFLE_VARIANT_POLICY_VERSION,
    PAIRING_NON_PAIRING_BUDGET_POLICY_VERSION,
    PARTIAL_QUERY_ONLY_MSA_POLICY_VERSION,
    VIRTUAL_SEGMENT_MSA_SHUFFLE_POLICY_VERSION,
)
from protenix.data.pipeline.dataloader import get_dataloaders
from protenix.data.pipeline.sample_context import (
    MSA_MODE_POLICY_VERSION,
    MSA_MODE_CROP_AWARE_QUERY_ONLY_POLICY_VERSION,
    MSA_MODE_MULTICHAIN_SHUFFLE_VARIANT_GLOBAL_QUERY_POLICY_VERSION,
    MSA_MODE_MULTICHAIN_SHUFFLE_VARIANT_POLICY_VERSION,
    MSA_MODE_QUERY_ONLY,
    MSA_MODE_SPECIES_PAIRED,
    MSA_MODE_UNPAIRED_SHUFFLE,
    MSA_MODE_VIRTUAL_SEGMENT_POLICY_VERSION,
    MULTICHAIN_MSA_SHUFFLE_VARIANT_STREAM,
    QUERY_ONLY_CHAIN_SELECTION_STREAM,
    VIRTUAL_SEGMENT_MSA_SHUFFLE_STREAM,
)
from protenix.metrics.lddt_metrics import LDDTMetrics
from protenix.model.loss import ProtenixLoss
from protenix.model.protenix import Protenix
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.fsdp import (
    apply_fsdp2,
    build_fsdp_device_mesh,
    get_fsdp2_full_model_state_dict,
    get_fsdp2_full_state_dict,
    get_fsdp2_shard_modules,
    optimizer_state_uses_canonical_fqns,
    set_fsdp2_full_model_state_dict,
    set_fsdp2_full_optimizer_state_dict,
    sync_module_states,
    validate_distributed_strategy,
)
from protenix.utils.lr_scheduler import FinetuneLRScheduler, get_lr_scheduler
from protenix.utils.metrics import SimpleMetricAggregator
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import autocasting_disable_decorator, to_device
from protenix.utils.training import get_optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from runner.ema import EMAWrapper

# Disable WANDB's console output capture to reduce unnecessary logging
os.environ["WANDB_CONSOLE"] = "off"

torch.serialization.add_safe_globals([Namespace])


def should_empty_cache(step: int, interval: int) -> bool:
    """Return whether cached CUDA memory should be released at this step."""
    return interval > 0 and step > 0 and step % interval == 0


def should_run_evaluation(
    step_need_eval: bool,
    is_last_step: bool,
    eval_at_end: bool,
) -> bool:
    """Return whether evaluation should run for the current training step."""
    return step_need_eval or (is_last_step and eval_at_end)


def validate_benchmark_window(
    warmup_steps: int,
    measure_steps: int,
    max_steps: int,
) -> bool:
    """Validate the optional synchronized steady-state benchmark window."""
    if warmup_steps < 0 and measure_steps < 0:
        return False
    if warmup_steps < 0:
        raise ValueError("benchmark_warmup_steps must be >= 0 when benchmarking")
    if measure_steps <= 0:
        raise ValueError("benchmark_measure_steps must be > 0 when benchmarking")
    if warmup_steps + measure_steps > max_steps:
        raise ValueError(
            "benchmark warmup + measurement steps must not exceed max_steps"
        )
    return True


def build_benchmark_window_result(
    rank_records: list[Mapping[str, Any]],
    *,
    warmup_steps: int,
    measure_steps: int,
) -> Dict[str, Any]:
    """Aggregate synchronized per-rank timing, memory, and health records."""
    if not rank_records:
        raise ValueError("rank_records must not be empty")

    records = sorted(rank_records, key=lambda record: int(record["rank"]))
    elapsed_values = [float(record["elapsed_seconds"]) for record in records]
    elapsed_seconds = max(elapsed_values)
    min_elapsed_seconds = min(elapsed_values)
    if elapsed_seconds <= 0:
        raise ValueError("rank elapsed_seconds must be positive")
    slowest_record = max(
        records,
        key=lambda record: float(record["elapsed_seconds"]),
    )
    total_samples = sum(int(record["sample_count"]) for record in records)
    total_tokens = sum(int(record["token_count"]) for record in records)
    total_atoms = sum(int(record["atom_count"]) for record in records)

    return {
        "schema_version": 1,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "world_size": len(records),
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_step": elapsed_seconds / measure_steps,
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "total_atoms": total_atoms,
        "samples_per_sec": total_samples / elapsed_seconds,
        "tokens_per_sec": total_tokens / elapsed_seconds,
        "atoms_per_sec": total_atoms / elapsed_seconds,
        "slowest_rank": int(slowest_record["rank"]),
        "rank_elapsed_imbalance": (
            elapsed_seconds / min_elapsed_seconds
            if min_elapsed_seconds > 0
            else None
        ),
        "max_allocated_bytes": max(
            int(record.get("max_allocated_bytes", 0)) for record in records
        ),
        "max_reserved_bytes": max(
            int(record.get("max_reserved_bytes", 0)) for record in records
        ),
        "max_loss_nonfinite_count_per_rank": max(
            int(record.get("loss_nonfinite_count", 0)) for record in records
        ),
        "max_gradient_nonfinite_count_per_rank": max(
            int(record.get("gradient_nonfinite_count", 0)) for record in records
        ),
        "max_optimizer_skipped_count_per_rank": max(
            int(record.get("optimizer_skipped_count", 0)) for record in records
        ),
        "rank_records": records,
    }


def get_progress_bar_steps(eval_interval: int, iters_to_accumulate: int) -> int:
    """Return a non-zero progress bar window size."""
    return max(1, eval_interval * iters_to_accumulate)


def gradient_accumulation_sync_context(
    model: torch.nn.Module,
    *,
    use_ddp: bool,
    global_step: int,
    iters_to_accumulate: int,
    force_sync: bool = False,
):
    """Skip DDP reduction until an accumulation boundary unless sync is required.

    PyTorch 2.7.1 cannot run the first backward of a ``static_graph=True`` DDP
    reducer inside ``no_sync()``. The trainer therefore forces that one
    lifecycle-local backward to synchronize before enabling normal accumulation.
    """
    if iters_to_accumulate <= 0:
        raise ValueError("iters_to_accumulate must be positive")
    is_update_step = (global_step + 1) % iters_to_accumulate == 0
    if not use_ddp or is_update_step or force_sync:
        return nullcontext()
    no_sync = getattr(model, "no_sync", None)
    if not callable(no_sync):
        raise TypeError("DDP gradient accumulation requires model.no_sync()")
    return no_sync()


def validate_expected_distributed_topology(
    *,
    world_size: int,
    local_world_size: int,
    num_nodes: int,
) -> None:
    """Fail if torchrun child coordinates differ from a locked launcher."""
    observed = {
        "PROTENIX_EXPECTED_WORLD_SIZE": int(world_size),
        "PROTENIX_EXPECTED_LOCAL_WORLD_SIZE": int(local_world_size),
        "PROTENIX_EXPECTED_NUM_NODES": int(num_nodes),
    }
    for environment_name, observed_value in observed.items():
        expected_text = os.environ.get(environment_name)
        if expected_text is None:
            continue
        if not expected_text.isdigit() or int(expected_text) <= 0:
            raise ValueError(
                f"{environment_name} must be a positive integer"
            )
        expected_value = int(expected_text)
        if observed_value != expected_value:
            raise RuntimeError(
                f"{environment_name} mismatch: expected={expected_value}, "
                f"observed={observed_value}"
            )


def get_eval_aggregator_names() -> list[str]:
    """Return evaluation aggregations used for persisted metrics."""
    return ["avg", "median", "pct90", "min", "max"]


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def validate_static_graph_template_usage(configs: Any) -> Dict[str, bool]:
    """Require invariant TemplateEmbedder participation across train sources."""
    data_cfg = _get_value(configs, "data", {})
    raw_train_sets = _get_value(data_cfg, "train_sets", []) or []
    if isinstance(raw_train_sets, str):
        train_sets = [
            item.strip()
            for item in raw_train_sets.split(",")
            if item.strip()
        ]
    else:
        train_sets = [str(item) for item in raw_train_sets]
    if not train_sets:
        return {}

    global_template_cfg = _get_value(data_cfg, "template", {})
    global_enabled = _get_value(
        global_template_cfg,
        "enable_prot_template",
        True,
    )
    global_dummy_when_disabled = _get_value(
        global_template_cfg,
        "emit_dummy_template_when_disabled",
        False,
    )
    for option_name, option_value in (
        ("data.template.enable_prot_template", global_enabled),
        (
            "data.template.emit_dummy_template_when_disabled",
            global_dummy_when_disabled,
        ),
    ):
        if not isinstance(option_value, (bool, np.bool_)):
            raise TypeError(
                f"{option_name} must be a boolean, got "
                f"{type(option_value).__name__}: {option_value!r}"
            )

    template_embedder_used = {}
    for dataset_name in train_sets:
        dataset_cfg = _get_value(data_cfg, dataset_name, {})
        dataset_template_cfg = _get_value(dataset_cfg, "template", {})
        enabled = _get_value(
            dataset_template_cfg,
            "enable_prot_template",
            global_enabled,
        )
        dummy_when_disabled = _get_value(
            dataset_template_cfg,
            "emit_dummy_template_when_disabled",
            global_dummy_when_disabled,
        )
        for option_name, option_value in (
            ("enable_prot_template", enabled),
            (
                "emit_dummy_template_when_disabled",
                dummy_when_disabled,
            ),
        ):
            if not isinstance(option_value, (bool, np.bool_)):
                raise TypeError(
                    f"data.{dataset_name}.template.{option_name} must be a "
                    f"boolean, got {type(option_value).__name__}: "
                    f"{option_value!r}"
                )
        template_embedder_used[dataset_name] = bool(
            enabled or dummy_when_disabled
        )

    if len(set(template_embedder_used.values())) > 1:
        used = [
            name
            for name, is_used in template_embedder_used.items()
            if is_used
        ]
        skipped = [
            name
            for name, is_used in template_embedder_used.items()
            if not is_used
        ]
        raise RuntimeError(
            "DDP static_graph requires invariant TemplateEmbedder parameter "
            "usage across training datasets. TemplateEmbedder is active for "
            f"{used} but skipped for {skipped}. Set "
            "emit_dummy_template_when_disabled=true on template-disabled "
            "training sources, or use a correctly configured dynamic DDP "
            "profile."
        )
    return template_embedder_used


def build_ddp_kwargs(configs: Any, local_rank: int) -> Dict[str, Any]:
    """Build the audited DDP options without changing historical defaults."""
    ddp_cfg = _get_value(configs, "ddp", {})
    gradient_as_bucket_view = _get_value(
        ddp_cfg, "gradient_as_bucket_view", False
    )
    if not isinstance(gradient_as_bucket_view, bool):
        raise TypeError(
            "ddp.gradient_as_bucket_view must be a bool, got "
            f"{gradient_as_bucket_view!r}"
        )
    validate_static_graph_template_usage(configs)
    return {
        "find_unused_parameters": _get_value(
            configs, "find_unused_parameters", False
        ),
        "device_ids": [local_rank],
        "output_device": local_rank,
        "static_graph": True,
        "gradient_as_bucket_view": gradient_as_bucket_view,
    }


def to_jsonable(value: Any) -> Any:
    """Convert common training values to JSON-serializable objects."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return to_jsonable(value.item())
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    # ml_collections.ConfigDict deliberately does not register as a standard
    # Mapping.  Falling through to str(value) produces YAML-looking strings in
    # benchmark manifests and makes strict result validation impossible.
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return to_jsonable(value.to_dict())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return to_jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return to_jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    return str(value)


def append_jsonl(path: os.PathLike[str] | str, record: Mapping[str, Any]) -> None:
    """Append one JSON record, creating the parent directory when needed."""
    path_str = os.fspath(path)
    parent = os.path.dirname(path_str)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path_str, "a", encoding="utf-8") as f:
        json.dump(to_jsonable(record), f, ensure_ascii=True, sort_keys=True)
        f.write("\n")


def write_json(path: os.PathLike[str] | str, data: Mapping[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    path_str = os.fspath(path)
    parent = os.path.dirname(path_str)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path_str, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def _now_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _exception_record(exception: BaseException | None) -> Dict[str, Any] | None:
    if exception is None:
        return None
    return {
        "type": type(exception).__name__,
        "message": str(exception),
        "traceback": "".join(
            traceback.format_exception(
                type(exception),
                exception,
                exception.__traceback__,
            )
        ),
    }


def build_run_status_record(
    *,
    status: str,
    run_name: str,
    run_dir: str,
    step: int,
    global_step: int,
    max_steps: int,
    latest_checkpoint: str | None = None,
    latest_ema_checkpoint: str | None = None,
    exit_reason: str | None = None,
    exception: BaseException | None = None,
    started_at: str | None = None,
    updated_at: str | None = None,
) -> Dict[str, Any]:
    """Build the latest training lifecycle status record."""
    update_time = updated_at or _now_timestamp()
    return {
        "event": "run_status",
        "status": status,
        "run_name": run_name,
        "run_dir": run_dir,
        "started_at": started_at,
        "start_time": started_at,
        "updated_at": update_time,
        "last_update_time": update_time,
        "last_step": step,
        "last_global_step": global_step,
        "max_steps": max_steps,
        "rank": DIST_WRAPPER.rank,
        "world_size": DIST_WRAPPER.world_size,
        "local_rank": DIST_WRAPPER.local_rank,
        "local_world_size": DIST_WRAPPER.local_world_size,
        "num_nodes": DIST_WRAPPER.num_nodes,
        "node_rank": DIST_WRAPPER.node_rank,
        "latest_checkpoint": latest_checkpoint,
        "latest_ema_checkpoint": latest_ema_checkpoint,
        "exit_reason": exit_reason,
        "exception": _exception_record(exception),
    }


def write_run_status(
    path: os.PathLike[str] | str,
    record: Mapping[str, Any],
) -> None:
    """Persist the latest lifecycle status as a JSON snapshot."""
    write_json(path, record)


def append_run_heartbeat(
    path: os.PathLike[str] | str,
    record: Mapping[str, Any],
) -> None:
    """Append one lifecycle heartbeat record."""
    append_jsonl(path, record)


def _git_summary(cwd: str) -> Dict[str, Any]:
    def run_git(args: list[str]) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=cwd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    status = run_git(["status", "--short"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "dirty": bool(status),
        "status_short": status.splitlines()[:200],
    }


def build_training_data_contract(configs: Any) -> Dict[str, Any]:
    """Build a canonical identity for dataset order, masses, and sampler policy."""
    data_cfg = _get_value(configs, "data", {})
    sampler_cfg = _get_value(data_cfg, "train_sampler", {})
    template_cfg = _get_value(data_cfg, "template", {})
    msa_cfg = _get_value(data_cfg, "msa", {})

    raw_train_sets = _get_value(data_cfg, "train_sets", []) or []
    if isinstance(raw_train_sets, str):
        train_sets = [
            item.strip() for item in raw_train_sets.split(",") if item.strip()
        ]
    else:
        train_sets = [str(item) for item in raw_train_sets]

    raw_weights = _get_value(sampler_cfg, "train_sample_weights", []) or []
    if isinstance(raw_weights, str):
        train_sample_weights = [
            float(item.strip())
            for item in raw_weights.split(",")
            if item.strip()
        ]
    else:
        train_sample_weights = [float(item) for item in raw_weights]

    def optional_float(config: Any, key: str) -> float | None:
        value = _get_value(config, key)
        return None if value is None else float(value)

    global_query_only_probability = optional_float(
        data_cfg,
        "query_only_msa_probability",
    )
    global_pair_as_unpaired_probability = optional_float(
        data_cfg,
        "pair_msa_as_unpaired_probability",
    )
    global_template_use_probability = optional_float(
        data_cfg,
        "template_use_probability",
    )
    global_template_dropout_rate = optional_float(
        template_cfg,
        "template_dropout_rate",
    )

    def strict_bool(config: Any, key: str, default: bool = False) -> bool:
        value = _get_value(config, key, default)
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(
                f"data option {key} must be a boolean, got "
                f"{type(value).__name__}: {value!r}"
            )
        return bool(value)

    def strict_nonnegative_int(
        config: Any,
        key: str,
        default: int = 0,
    ) -> int:
        value = _get_value(config, key, default)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise TypeError(
                f"data option {key} must be a non-negative integer, got "
                f"{type(value).__name__}: {value!r}"
            )
        value = int(value)
        if value < 0:
            raise ValueError(f"data option {key} must be >= 0, got {value}")
        return value

    def strict_probability(
        config: Any,
        key: str,
        default: float,
    ) -> float:
        value = _get_value(config, key, default)
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(
                f"data option {key} must be a probability, got {value!r}"
            )
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"data option {key} must be a probability, got {value!r}"
            ) from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"data option {key} must be finite and in [0, 1], got "
                f"{value!r}"
            )
        return value

    raw_query_inclusion_config = _get_value(
        msa_cfg,
        "query_inclusion_max_sample_size",
    )
    if raw_query_inclusion_config is None:
        query_inclusion_max_sample_size = {"train": 0, "test": 0}
    elif isinstance(
        raw_query_inclusion_config,
        (int, np.integer),
    ) and not isinstance(raw_query_inclusion_config, (bool, np.bool_)):
        scalar_threshold = strict_nonnegative_int(
            {"value": raw_query_inclusion_config},
            "value",
        )
        query_inclusion_max_sample_size = {
            "train": scalar_threshold,
            "test": scalar_threshold,
        }
    elif (
        isinstance(raw_query_inclusion_config, Mapping)
        or isinstance(raw_query_inclusion_config, SimpleNamespace)
        or hasattr(raw_query_inclusion_config, "get")
    ):
        query_inclusion_max_sample_size = {
            stage: strict_nonnegative_int(
                raw_query_inclusion_config,
                stage,
            )
            for stage in ("train", "test")
        }
    else:
        raise TypeError(
            "data option query_inclusion_max_sample_size must be a "
            "non-negative integer or train/test mapping, got "
            f"{type(raw_query_inclusion_config).__name__}: "
            f"{raw_query_inclusion_config!r}"
        )

    virtual_segment_shuffle_enabled = strict_bool(
        msa_cfg,
        "enable_virtual_segment_msa_shuffle",
    )
    multichain_shuffle_variant_sampling_enabled = strict_bool(
        msa_cfg,
        "enable_multichain_msa_shuffle_variant_sampling",
    )
    multichain_shuffle_segment_probability = strict_probability(
        msa_cfg,
        "multichain_msa_shuffle_segment_probability",
        0.5,
    )
    if (
        multichain_shuffle_variant_sampling_enabled
        and not virtual_segment_shuffle_enabled
    ):
        raise ValueError(
            "enable_multichain_msa_shuffle_variant_sampling requires "
            "enable_virtual_segment_msa_shuffle=true"
        )
    partial_query_only_enabled = strict_bool(
        msa_cfg,
        "enable_crop_aware_partial_query_only",
    )
    partial_query_only_max_chains = None
    if partial_query_only_enabled:
        partial_query_only_max_chains = strict_nonnegative_int(
            msa_cfg,
            "partial_query_only_max_chains",
            2,
        )
        if partial_query_only_max_chains < 1:
            raise ValueError(
                "data option partial_query_only_max_chains must be >= 1"
            )
    strict_data_semantics_enabled = bool(
        virtual_segment_shuffle_enabled
        or multichain_shuffle_variant_sampling_enabled
        or partial_query_only_enabled
    )
    virtual_segment_residue_gap = None
    virtual_segment_min_length = None
    if virtual_segment_shuffle_enabled:
        virtual_segment_residue_gap = strict_nonnegative_int(
            msa_cfg,
            "virtual_segment_residue_gap",
            24,
        )
        virtual_segment_min_length = strict_nonnegative_int(
            msa_cfg,
            "virtual_segment_min_length",
            9,
        )
        if virtual_segment_residue_gap < 1:
            raise ValueError(
                "data option virtual_segment_residue_gap must be >= 1"
            )
        if virtual_segment_min_length < 1:
            raise ValueError(
                "data option virtual_segment_min_length must be >= 1"
            )

    def canonicalize_config_value(value: Any) -> Any:
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        elif isinstance(value, SimpleNamespace):
            value = vars(value)
        if isinstance(value, dict):
            return {
                str(key): canonicalize_config_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [canonicalize_config_value(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        return value

    def merged_effective_config(
        global_config: Any,
        dataset_config: Any,
        *,
        dataset_name: str,
        stage: str | None = None,
    ) -> Dict[str, Any]:
        """Mirror the shallow global-to-dataset override used at runtime."""
        global_mapping = canonicalize_config_value(global_config)
        dataset_mapping = canonicalize_config_value(dataset_config)
        if not isinstance(global_mapping, dict):
            raise TypeError(
                "global MSA/template configuration must be a mapping, got "
                f"{type(global_mapping).__name__}"
            )
        if not isinstance(dataset_mapping, dict):
            raise TypeError(
                "dataset MSA/template configuration must be a mapping, got "
                f"{type(dataset_mapping).__name__}"
            )
        effective = dict(global_mapping)
        effective["dataset_name"] = dataset_name
        effective.update(dataset_mapping)
        if stage is not None:
            effective["stage"] = stage
        return effective

    def canonical_effective_template_config(
        global_config: Any,
        dataset_config: Any,
        *,
        dataset_name: str,
        stage: str,
    ) -> Dict[str, Any]:
        """Canonicalize only template options that can affect this stage."""
        effective = merged_effective_config(
            global_config,
            dataset_config,
            dataset_name=dataset_name,
            stage=stage,
        )
        if stage == "train":
            # These options are consumed only by evaluation/inference.  They
            # were added after the v10 training checkpoint was written and
            # must not create a false training-data contract delta.
            for key in (
                "evaluation_sequence_template_cache_root",
                "evaluation_sequence_template_cache_mode",
                "evaluation_sequence_template_cache_filename",
                "evaluation_sequence_template_cache_policy",
            ):
                effective.pop(key, None)
        return effective

    def independent_pairing_non_pairing_budget_applies(
        effective: Mapping[str, Any],
    ) -> bool:
        """Return whether the new source-budget gate changes this dataset."""
        if not strict_bool(
            effective,
            "independent_pairing_non_pairing_budget",
        ):
            return False
        paired_sources = canonicalize_config_value(
            _get_value(effective, "prot_pairing_dbs", []) or []
        )
        unpaired_sources = canonicalize_config_value(
            _get_value(effective, "prot_non_pairing_dbs", []) or []
        )
        if not isinstance(paired_sources, list) or not isinstance(
            unpaired_sources,
            list,
        ):
            raise TypeError(
                "protein pairing/non-pairing database settings must be lists"
            )
        for paired_source, unpaired_source in zip(
            paired_sources,
            unpaired_sources,
        ):
            paired_source = str(paired_source)
            sources = [
                source
                for source in str(unpaired_source).split("-")
                if source
            ]
            if (
                paired_source
                and paired_source in sources
                and any(source != paired_source for source in sources)
            ):
                return True
        return False

    def canonical_effective_msa_config(
        global_config: Any,
        dataset_config: Any,
        *,
        dataset_name: str,
    ) -> Dict[str, Any]:
        """Canonicalize the runtime MSA merge without inactive new defaults."""
        effective = merged_effective_config(
            global_config,
            dataset_config,
            dataset_name=dataset_name,
        )
        effective_partial_enabled = strict_bool(
            effective,
            "enable_crop_aware_partial_query_only",
        )
        effective_variant_sampling_enabled = strict_bool(
            effective,
            "enable_multichain_msa_shuffle_variant_sampling",
        )
        effective_variant_segment_probability = strict_probability(
            effective,
            "multichain_msa_shuffle_segment_probability",
            0.5,
        )
        if (
            effective_variant_sampling_enabled
            and not strict_bool(
                effective,
                "enable_virtual_segment_msa_shuffle",
            )
        ):
            raise ValueError(
                "effective MSA option "
                "enable_multichain_msa_shuffle_variant_sampling requires "
                "enable_virtual_segment_msa_shuffle=true"
            )
        effective_partial_max_chains = strict_nonnegative_int(
            effective,
            "partial_query_only_max_chains",
            2,
        )
        if effective_partial_max_chains < 1:
            raise ValueError(
                "data option partial_query_only_max_chains must be >= 1"
            )
        if not effective_partial_enabled:
            # These keys did not exist in the historical virtual-segment v2
            # contract. Their disabled defaults have no runtime effect, so
            # omit them to preserve strict-resume identity for that policy.
            effective.pop(
                "enable_crop_aware_partial_query_only",
                None,
            )
            effective.pop("partial_query_only_max_chains", None)
        if not effective_variant_sampling_enabled:
            # These defaults were introduced after the historical v2
            # contract and have no runtime effect while the gate is disabled.
            effective.pop(
                "enable_multichain_msa_shuffle_variant_sampling",
                None,
            )
            effective.pop(
                "multichain_msa_shuffle_segment_probability",
                None,
            )
        else:
            effective[
                "multichain_msa_shuffle_segment_probability"
            ] = effective_variant_segment_probability
        if not independent_pairing_non_pairing_budget_applies(effective):
            # The option is contract-relevant only when it changes runtime
            # source accounting.  Its disabled default, and enabling it for a
            # dataset without both a reused paired source and a genuine
            # non-pairing source, are identical to the historical behavior.
            effective.pop("independent_pairing_non_pairing_budget", None)
        return effective

    independent_pairing_non_pairing_budget_by_dataset = {
        dataset_name: independent_pairing_non_pairing_budget_applies(
            merged_effective_config(
                msa_cfg,
                _get_value(
                    _get_value(data_cfg, dataset_name, {}),
                    "msa",
                    {},
                ),
                dataset_name=dataset_name,
            )
        )
        for dataset_name in train_sets
    }
    strict_data_semantics_enabled = bool(
        strict_data_semantics_enabled
        or any(independent_pairing_non_pairing_budget_by_dataset.values())
    )

    def resolve_effective_sampler_seed() -> int:
        """Resolve the seed exactly as the training DataLoader does."""
        configured_seed = _get_value(sampler_cfg, "seed", -1)
        try:
            inherit_global_seed = (
                configured_seed is None or int(configured_seed) < 0
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "data.train_sampler.seed must be integer-compatible, got "
                f"{configured_seed!r}"
            ) from exc
        effective_seed = (
            _get_value(configs, "seed")
            if inherit_global_seed
            else configured_seed
        )
        if effective_seed is None:
            raise ValueError(
                "schema-7 training data contract cannot resolve the sampler "
                "seed: data.train_sampler.seed is negative/None and the "
                "global training seed is missing"
            )
        try:
            return int(effective_seed)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "effective training sampler seed must be "
                f"integer-compatible, got {effective_seed!r}"
            ) from exc

    def read_file_identity(path: str, *, artifact: str) -> Dict[str, Any]:
        resolved_path = os.path.realpath(path)
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(
                f"{artifact} does not exist or is not a file: "
                f"{resolved_path}"
            )
        digest = hashlib.sha256()
        size_bytes = 0
        with open(resolved_path, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)
        return {
            "path": resolved_path,
            "size_bytes": size_bytes,
            "sha256": digest.hexdigest(),
        }

    def read_runtime_partition_contract(
        *,
        dataset_name: str,
        runtime_index_dir: str,
        manifest: Mapping[str, Any],
    ) -> Dict[str, Any] | None:
        partition = manifest.get("partition")
        if partition is None:
            return None
        if not isinstance(partition, Mapping):
            raise ValueError(
                f"{dataset_name} runtime manifest partition must be an object"
            )
        artifacts = manifest.get("artifacts")
        pair_record = (
            artifacts.get("partition_pair")
            if isinstance(artifacts, Mapping)
            else None
        )
        if not isinstance(pair_record, Mapping):
            raise ValueError(
                f"{dataset_name} partitioned runtime manifest lacks the "
                "partition_pair artifact"
            )
        if pair_record.get("path") != "partition_pair.json":
            raise ValueError(
                f"{dataset_name} partition_pair artifact must use "
                "partition_pair.json"
            )
        pair_identity = read_file_identity(
            os.path.join(runtime_index_dir, "partition_pair.json"),
            artifact=f"{dataset_name} partition-pair receipt",
        )
        try:
            expected_size = int(pair_record["size_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{dataset_name} partition_pair artifact has invalid size"
            ) from exc
        if (
            pair_identity["size_bytes"] != expected_size
            or pair_identity["sha256"] != pair_record.get("sha256")
            or pair_identity["sha256"]
            != partition.get("pair_receipt_sha256")
        ):
            raise ValueError(
                f"{dataset_name} partition-pair receipt identity does not "
                "match its runtime manifest"
            )
        with open(pair_identity["path"], "rb") as handle:
            pair_receipt = json.load(handle)
        if not isinstance(pair_receipt, Mapping):
            raise ValueError(
                f"{dataset_name} partition-pair receipt must be an object"
            )
        return {
            "manifest_metadata": canonicalize_config_value(partition),
            "receipt_artifact": pair_identity,
            "receipt": canonicalize_config_value(pair_receipt),
        }

    effective_augmentation_by_dataset = {}
    effective_loss_and_crop_policy_by_dataset = {}
    effective_runtime_semantics_by_dataset = {}
    strict_effective_runtime_semantics_by_dataset = {}
    runtime_distillation_indices = {}
    immutable_dataset_artifacts = {}
    for dataset_name in train_sets:
        dataset_cfg = _get_value(data_cfg, dataset_name, {})
        dataset_augmentation_cfg = _get_value(dataset_cfg, "augmentation", {})
        dataset_template_cfg = _get_value(dataset_cfg, "template", {})
        dataset_cropping_cfg = _get_value(
            dataset_cfg,
            "cropping_configs",
            {},
        )
        dataset_sampler_cfg = _get_value(
            dataset_cfg,
            "sampler_configs",
            {},
        )
        dataset_msa_cfg = _get_value(dataset_cfg, "msa", {})
        dataset_base_info = _get_value(dataset_cfg, "base_info", {})
        disable_non_protein_diffusion_weights = strict_bool(
            dataset_base_info,
            "disable_non_protein_diffusion_weights",
        )
        require_resolved_protein_crop = strict_bool(
            dataset_base_info,
            "require_resolved_protein_crop",
        )
        is_distillation = strict_bool(
            dataset_base_info,
            "is_distillation",
        )
        min_diffusion_supervised_tokens = strict_nonnegative_int(
            dataset_base_info,
            "min_diffusion_supervised_tokens",
        )
        effective_loss_and_crop_policy_by_dataset[dataset_name] = {
            "disable_non_protein_diffusion_weights": (
                disable_non_protein_diffusion_weights
            ),
            "is_distillation": is_distillation,
            "diffusion_supervision_policy": (
                "protein_mse_residual_and_smooth_lddt_endpoints_v1"
                if disable_non_protein_diffusion_weights
                else "all_resolved_atoms_v1"
            ),
            "require_resolved_protein_crop": require_resolved_protein_crop,
            "min_diffusion_supervised_tokens": (
                min_diffusion_supervised_tokens
            ),
            "diffusion_supervised_token_definition": (
                "token_has_resolved_protein_atom_v1"
                if disable_non_protein_diffusion_weights
                else "token_has_resolved_atom_v1"
            ),
            "minimum_validation_policy": (
                "pre_msa_crop_post_crop_final_feature_and_post_permutation_v2"
                if min_diffusion_supervised_tokens > 0
                else None
            ),
            "required_token_crop_policy": (
                "nearest_eligible_min_k_remove_farthest_complete_ref_space_uid_v2"
                if require_resolved_protein_crop
                else None
            ),
        }
        effective_runtime_semantics_by_dataset[dataset_name] = {
            "base_info": canonicalize_config_value(dataset_base_info),
            "cropping_configs": canonicalize_config_value(
                dataset_cropping_cfg
            ),
            "sampler_configs": canonicalize_config_value(
                dataset_sampler_cfg
            ),
            "msa": canonicalize_config_value(dataset_msa_cfg),
            "template": canonicalize_config_value(dataset_template_cfg),
        }
        if strict_data_semantics_enabled:
            strict_effective_runtime_semantics_by_dataset[dataset_name] = {
                "stage": "train",
                "dataset_config": canonicalize_config_value(dataset_cfg),
                "effective_msa": canonical_effective_msa_config(
                    msa_cfg,
                    dataset_msa_cfg,
                    dataset_name=dataset_name,
                ),
                "effective_template": canonical_effective_template_config(
                    template_cfg,
                    dataset_template_cfg,
                    dataset_name=dataset_name,
                    stage="train",
                ),
                "train_ref_pos_augment": canonicalize_config_value(
                    _get_value(data_cfg, "train_ref_pos_augment", True)
                ),
                "limits": canonicalize_config_value(
                    _get_value(data_cfg, "limits", -1)
                ),
            }

        dataset_success_fpath = str(
            _get_value(
                dataset_base_info,
                "dataset_success_fpath",
                "",
            )
            or ""
        ).strip()
        dataset_manifest_fpath = str(
            _get_value(
                dataset_base_info,
                "dataset_manifest_fpath",
                "",
            )
            or ""
        ).strip()
        if dataset_success_fpath or dataset_manifest_fpath:
            if not dataset_success_fpath or not dataset_manifest_fpath:
                raise ValueError(
                    f"{dataset_name} must configure both "
                    "dataset_success_fpath and dataset_manifest_fpath"
                )
            success_identity = read_file_identity(
                dataset_success_fpath,
                artifact=f"{dataset_name} success marker",
            )
            manifest_identity = read_file_identity(
                dataset_manifest_fpath,
                artifact=f"{dataset_name} manifest",
            )
            indices_fpath = str(
                _get_value(dataset_base_info, "indices_fpath", "") or ""
            ).strip()
            if not indices_fpath:
                raise ValueError(
                    f"{dataset_name} immutable artifact contract requires "
                    "indices_fpath"
                )
            indices_identity = read_file_identity(
                indices_fpath,
                artifact=f"{dataset_name} indices",
            )
            with open(success_identity["path"], "rb") as handle:
                success_payload = json.load(handle)
            declared_manifest_sha256 = success_payload.get(
                "manifest_sha256"
            )
            if not isinstance(declared_manifest_sha256, str) or len(
                declared_manifest_sha256
            ) != 64:
                raise ValueError(
                    f"{dataset_name} success marker must declare a "
                    "64-character manifest_sha256"
                )
            if declared_manifest_sha256 != manifest_identity["sha256"]:
                raise ValueError(
                    f"{dataset_name} success marker declares manifest "
                    f"{declared_manifest_sha256}, but actual sha256 is "
                    f"{manifest_identity['sha256']}"
                )
            with open(manifest_identity["path"], "rb") as handle:
                dataset_manifest = json.load(handle)
            if dataset_manifest.get("status") != "complete":
                raise ValueError(
                    f"{dataset_name} manifest status must be complete"
                )
            if dataset_manifest.get("success_eligible") is not True:
                raise ValueError(
                    f"{dataset_name} manifest must set "
                    "success_eligible=true"
                )
            manifest_indices_sha256 = _get_value(
                _get_value(
                    dataset_manifest,
                    "artifacts",
                    {},
                ),
                "indices",
                {},
            )
            manifest_indices_sha256 = _get_value(
                manifest_indices_sha256,
                "sha256",
            )
            if manifest_indices_sha256 != indices_identity["sha256"]:
                raise ValueError(
                    f"{dataset_name} manifest indices sha256 "
                    f"{manifest_indices_sha256!r} does not match actual "
                    f"{indices_identity['sha256']}"
                )
            immutable_dataset_artifacts[dataset_name] = {
                "success": success_identity,
                "manifest": manifest_identity,
                "indices": indices_identity,
                "declared_manifest_sha256": declared_manifest_sha256,
                "cache_sha256": success_payload.get("cache_sha256"),
                "entry_count": success_payload.get("entry_count"),
                "index_row_count": success_payload.get("index_row_count"),
                "pickle_count": success_payload.get("pickle_count"),
            }
        runtime_index_dir = str(
            _get_value(dataset_base_info, "runtime_index_dir", "") or ""
        ).strip()
        if runtime_index_dir:
            manifest_path = os.path.join(runtime_index_dir, "manifest.json")
            with open(manifest_path, "rb") as handle:
                manifest_bytes = handle.read()
            manifest = json.loads(manifest_bytes)
            runtime_contract = {
                "runtime_index_dir": os.path.realpath(runtime_index_dir),
                "manifest_sha256": hashlib.sha256(
                    manifest_bytes
                ).hexdigest(),
                "schema_version": manifest.get("schema_version"),
                "format": manifest.get("format"),
                "row_count": manifest.get("row_count"),
                "cluster_count": manifest.get("cluster_count"),
                "source_index_sha256": _get_value(
                    manifest.get(
                        "source",
                        manifest.get("source_index", {}),
                    ),
                    "sha256",
                ),
            }
            partition_contract = read_runtime_partition_contract(
                dataset_name=dataset_name,
                runtime_index_dir=runtime_index_dir,
                manifest=manifest,
            )
            if partition_contract is not None:
                runtime_contract["partition"] = partition_contract
            runtime_distillation_indices[dataset_name] = runtime_contract
        query_only_probability = optional_float(
            dataset_augmentation_cfg,
            "query_only_msa_probability",
        )
        if query_only_probability is None:
            query_only_probability = global_query_only_probability
        pair_as_unpaired_probability = optional_float(
            dataset_augmentation_cfg,
            "pair_msa_as_unpaired_probability",
        )
        if pair_as_unpaired_probability is None:
            pair_as_unpaired_probability = global_pair_as_unpaired_probability
        species_paired_probability = None
        if (
            query_only_probability is not None
            and pair_as_unpaired_probability is not None
        ):
            species_paired_probability = (
                1.0 - query_only_probability - pair_as_unpaired_probability
            )
        effective_augmentation_by_dataset[dataset_name] = {
            "query_only_msa_probability": query_only_probability,
            "pair_msa_as_unpaired_probability": pair_as_unpaired_probability,
            "species_paired_msa_probability": species_paired_probability,
            "template_use_probability": optional_float(
                dataset_augmentation_cfg,
                "template_use_probability",
            )
            if _get_value(dataset_augmentation_cfg, "template_use_probability")
            is not None
            else global_template_use_probability,
            "legacy_template_dropout_rate": optional_float(
                dataset_template_cfg,
                "template_dropout_rate",
            )
            if _get_value(dataset_template_cfg, "template_dropout_rate")
            is not None
            else global_template_dropout_rate,
        }

    payload = {
        "schema_version": 6,
        "profile_id": os.environ.get(
            "PROTENIX_TRAINING_DATA_PROFILE", ""
        ).strip(),
        "train_sets": train_sets,
        "train_sample_weights": train_sample_weights,
        "epoch_size": _get_value(data_cfg, "epoch_size"),
        "sampler_seed": _get_value(sampler_cfg, "seed", -1),
        "sampler_replacement": bool(
            _get_value(sampler_cfg, "replacement", True)
        ),
        "augmentation": {
            "decision_version": str(
                _get_value(data_cfg, "augmentation_decision_version", "") or ""
            ),
            "msa_mode_policy": (
                MSA_MODE_MULTICHAIN_SHUFFLE_VARIANT_POLICY_VERSION
                if (
                    multichain_shuffle_variant_sampling_enabled
                    and partial_query_only_enabled
                )
                else (
                    MSA_MODE_MULTICHAIN_SHUFFLE_VARIANT_GLOBAL_QUERY_POLICY_VERSION
                )
                if multichain_shuffle_variant_sampling_enabled
                else MSA_MODE_CROP_AWARE_QUERY_ONLY_POLICY_VERSION
                if partial_query_only_enabled
                else MSA_MODE_VIRTUAL_SEGMENT_POLICY_VERSION
                if virtual_segment_shuffle_enabled
                else MSA_MODE_POLICY_VERSION
            ),
            "msa_mode_stream": "msa_mode",
            "msa_mode_order": [
                MSA_MODE_QUERY_ONLY,
                MSA_MODE_UNPAIRED_SHUFFLE,
                MSA_MODE_SPECIES_PAIRED,
            ],
            "query_only_msa_probability": global_query_only_probability,
            "pair_msa_as_unpaired_probability": (
                global_pair_as_unpaired_probability
            ),
            "species_paired_msa_probability": (
                None
                if global_query_only_probability is None
                or global_pair_as_unpaired_probability is None
                else 1.0
                - global_query_only_probability
                - global_pair_as_unpaired_probability
            ),
            "unpaired_shuffle_flags": {
                "pair_msa_as_unpaired_only": True,
                "shuffle_pair_msa_when_unpaired_only": True,
            },
            "species_paired_flags": {
                "pair_msa_as_unpaired_only": False,
                "shuffle_pair_msa_when_unpaired_only": False,
            },
            "template_use_probability": global_template_use_probability,
            "legacy_template_dropout_rate": global_template_dropout_rate,
            "effective_by_dataset": effective_augmentation_by_dataset,
        },
    }
    if any(query_inclusion_max_sample_size.values()):
        payload["augmentation"]["model_msa_query_inclusion"] = {
            "policy_version": MODEL_MSA_QUERY_INCLUSION_POLICY_VERSION,
            "query_row_index": 0,
            "max_sample_size": query_inclusion_max_sample_size,
            "depth_accounting": (
                MODEL_MSA_QUERY_INCLUSION_DEPTH_ACCOUNTING
            ),
            "above_threshold": (
                MODEL_MSA_QUERY_INCLUSION_ABOVE_THRESHOLD
            ),
        }
    if virtual_segment_shuffle_enabled:
        payload["augmentation"]["virtual_segment_msa_shuffle"] = {
            "enabled": True,
            "policy_version": (
                VIRTUAL_SEGMENT_MSA_SHUFFLE_POLICY_VERSION
            ),
            "seed_stream": VIRTUAL_SEGMENT_MSA_SHUFFLE_STREAM,
            "residue_gap": virtual_segment_residue_gap,
            "min_segment_length": virtual_segment_min_length,
            "grouping_axis": (
                "unique_physical_asym_id_canonical_res_id"
            ),
            "short_segment_policy": (
                "nearest_eligible_same_chain_tie_left_v1"
            ),
            "protein_msa_sources": (
                "all_merged_paired_and_unpaired_rows"
            ),
            "query_row_fixed": True,
            "existing_paired_query_placeholder_fixed": True,
        }
    if multichain_shuffle_variant_sampling_enabled:
        payload["augmentation"]["multichain_msa_shuffle_variant"] = {
            "enabled": True,
            "policy_version": (
                MULTICHAIN_MSA_SHUFFLE_VARIANT_POLICY_VERSION
            ),
            "seed_stream": MULTICHAIN_MSA_SHUFFLE_VARIANT_STREAM,
            "decision_scope": (
                "unpaired_shuffle_crop_visible_physical_protein_chains"
            ),
            "multichain_threshold": 2,
            "homomer_copies_are_distinct_chains": True,
            "non_protein_chains_are_candidates": False,
            "segment_probability": (
                multichain_shuffle_segment_probability
            ),
            "whole_physical_chain_probability": (
                1.0 - multichain_shuffle_segment_probability
            ),
            "single_protein_chain_policy": "segment",
            "segment_grouping_policy": (
                VIRTUAL_SEGMENT_MSA_SHUFFLE_POLICY_VERSION
            ),
            "whole_chain_grouping_axis": (
                "unique_physical_protein_asym_id"
            ),
            "protein_msa_sources": (
                "all_merged_paired_and_unpaired_rows"
            ),
            "query_row_fixed": True,
            "deterministic_scope": "occurrence_seeded_given_crop",
        }
    if partial_query_only_enabled:
        payload["augmentation"]["crop_aware_partial_query_only"] = {
            "enabled": True,
            "policy_version": PARTIAL_QUERY_ONLY_MSA_POLICY_VERSION,
            "seed_stream": QUERY_ONLY_CHAIN_SELECTION_STREAM,
            "selection_stage": "post_crop_pre_homolog_io",
            "candidate_axis": "crop_selected_physical_protein_asym_id",
            "candidate_order": "sorted_unique_asym_id",
            "homomer_copies_are_distinct_candidates": True,
            "selected_count_policy": (
                "uniform_integer_1_to_min_max_chains_n_minus_1"
            ),
            "max_query_only_chains": partial_query_only_max_chains,
            "minimum_retained_protein_chains": 1,
            "zero_or_one_protein_policy": (
                "legacy_global_query_only_v1"
            ),
            "selected_chain_homolog_io": "skipped",
            "selected_chain_non_query_rows": (
                "gap_with_zero_deletion_query_profile_v1"
            ),
            "retained_chain_policy": "normal_protenix_msa_flow",
            "sole_retained_chain_paired_hits": (
                "merge_into_unpaired_to_preserve_all_sources_v1"
            ),
            "multi_protein_non_protein_policy": (
                "not_candidates_and_normal_msa_flow"
            ),
            "query_row_fixed": True,
            "deterministic_scope": "occurrence_seeded_given_crop",
        }
    profile_id = payload["profile_id"]
    current_disordered_profile = (
        "weighted_pdb_afdb_mgnify_disordered_v2"
    )
    current_disordered_train_sets = [
        "weightedPDB_before210930_v20260101",
        "afdb_mgnify_distillation_v20260725",
        "openfold3_disordered_v20260725",
    ]
    if (
        profile_id == current_disordered_profile
        and train_sets != current_disordered_train_sets
    ):
        raise ValueError(
            f"{current_disordered_profile} requires train_sets="
            f"{current_disordered_train_sets}, got {train_sets}"
        )
    if profile_id == current_disordered_profile:
        expected_distillation = {
            "weightedPDB_before210930_v20260101": False,
            "afdb_mgnify_distillation_v20260725": True,
            "openfold3_disordered_v20260725": True,
        }
        for dataset_name, expected_value in expected_distillation.items():
            observed_value = (
                effective_loss_and_crop_policy_by_dataset[dataset_name][
                    "is_distillation"
                ]
            )
            if observed_value is not expected_value:
                raise ValueError(
                    f"{current_disordered_profile} requires "
                    f"{dataset_name}.is_distillation="
                    f"{str(expected_value).lower()}"
                )
        for dataset_name in (
            "weightedPDB_before210930_v20260101",
            "afdb_mgnify_distillation_v20260725",
        ):
            if effective_loss_and_crop_policy_by_dataset[dataset_name][
                "disable_non_protein_diffusion_weights"
            ]:
                raise ValueError(
                    f"{current_disordered_profile} requires "
                    f"{dataset_name}."
                    "disable_non_protein_diffusion_weights=false"
                )
        if not effective_loss_and_crop_policy_by_dataset[
            "openfold3_disordered_v20260725"
        ]["disable_non_protein_diffusion_weights"]:
            raise ValueError(
                f"{current_disordered_profile} requires "
                "openfold3_disordered_v20260725."
                "disable_non_protein_diffusion_weights=true"
            )
        if (
            "openfold3_disordered_v20260725"
            not in immutable_dataset_artifacts
        ):
            raise ValueError(
                f"{current_disordered_profile} requires immutable "
                "disordered indices/_SUCCESS/manifest artifacts"
            )
        if (
            "afdb_mgnify_distillation_v20260725"
            not in runtime_distillation_indices
        ):
            raise ValueError(
                f"{current_disordered_profile} requires the AFDB/MGnify "
                "runtime-index manifest"
            )
        if not virtual_segment_shuffle_enabled:
            raise ValueError(
                f"{current_disordered_profile} requires "
                "data.msa.enable_virtual_segment_msa_shuffle=true"
            )
        if not partial_query_only_enabled:
            raise ValueError(
                f"{current_disordered_profile} requires "
                "data.msa.enable_crop_aware_partial_query_only=true"
            )
        if partial_query_only_max_chains != 2:
            raise ValueError(
                f"{current_disordered_profile} requires "
                "partial_query_only_max_chains=2"
            )
        if (
            virtual_segment_residue_gap != 24
            or virtual_segment_min_length != 9
        ):
            raise ValueError(
                f"{current_disordered_profile} requires virtual segment "
                "residue_gap=24 and min_segment_length=9"
            )
        for dataset_name in current_disordered_train_sets:
            effective_msa = strict_effective_runtime_semantics_by_dataset[
                dataset_name
            ]["effective_msa"]
            effective_virtual_segment_enabled = strict_bool(
                effective_msa,
                "enable_virtual_segment_msa_shuffle",
            )
            effective_virtual_segment_gap = strict_nonnegative_int(
                effective_msa,
                "virtual_segment_residue_gap",
                24,
            )
            effective_virtual_segment_min_length = strict_nonnegative_int(
                effective_msa,
                "virtual_segment_min_length",
                9,
            )
            effective_partial_enabled = strict_bool(
                effective_msa,
                "enable_crop_aware_partial_query_only",
            )
            effective_partial_max_chains = strict_nonnegative_int(
                effective_msa,
                "partial_query_only_max_chains",
                2,
            )
            if (
                not effective_virtual_segment_enabled
                or effective_virtual_segment_gap != 24
                or effective_virtual_segment_min_length != 9
                or not effective_partial_enabled
                or effective_partial_max_chains != 2
            ):
                raise ValueError(
                    f"{current_disordered_profile} requires {dataset_name} "
                    "effective MSA settings "
                    "virtual_segment=true/gap24/min9 and "
                    "partial_query_only=true/max2"
                )
            augmentation = effective_augmentation_by_dataset[dataset_name]
            if (
                augmentation["query_only_msa_probability"] != 0.2
                or augmentation[
                    "pair_msa_as_unpaired_probability"
                ]
                != 0.3
                or augmentation["species_paired_msa_probability"] != 0.5
            ):
                raise ValueError(
                    f"{current_disordered_profile} requires {dataset_name} "
                    "MSA mode probabilities query/unpaired/species="
                    "0.2/0.3/0.5"
                )
    if (
        len(train_sets) == len(current_disordered_train_sets)
        and set(train_sets) == set(current_disordered_train_sets)
    ):
        for dataset_name in current_disordered_train_sets:
            policy = effective_loss_and_crop_policy_by_dataset[dataset_name]
            if policy["min_diffusion_supervised_tokens"] != 4:
                raise ValueError(
                    "the current three-source disordered training contract "
                    f"requires {dataset_name}.min_diffusion_supervised_tokens"
                    "=4"
                )
        for dataset_name in (
            "afdb_mgnify_distillation_v20260725",
            "openfold3_disordered_v20260725",
        ):
            policy = effective_loss_and_crop_policy_by_dataset[dataset_name]
            if not policy["require_resolved_protein_crop"]:
                raise ValueError(
                    "the current three-source disordered training contract "
                    f"requires {dataset_name}."
                    "require_resolved_protein_crop=true"
                )
        if effective_loss_and_crop_policy_by_dataset[
            "weightedPDB_before210930_v20260101"
        ]["require_resolved_protein_crop"]:
            raise ValueError(
                "the current three-source disordered training contract "
                "requires weightedPDB_before210930_v20260101."
                "require_resolved_protein_crop=false"
            )
        disordered_policy = effective_loss_and_crop_policy_by_dataset[
            "openfold3_disordered_v20260725"
        ]
        if not disordered_policy["is_distillation"]:
            raise ValueError(
                "openfold3_disordered_v20260725 must use "
                "is_distillation=true"
            )

    short_long_disordered_profile_weights = {
        "weighted_pdb_afdb_mgnify_short_long_disordered_v3": [
            0.5,
            0.005,
            0.495,
            0.02,
        ],
        "weighted_pdb_afdb_mgnify_short_long_disordered_v4": [
            0.49,
            0.005,
            0.505,
            0.02,
        ],
    }
    is_short_long_disordered_profile = (
        profile_id in short_long_disordered_profile_weights
    )
    short_long_disordered_profile = (
        profile_id
        if is_short_long_disordered_profile
        else "weighted_pdb_afdb_mgnify_short_long_disordered_v3"
    )
    short_distillation_dataset = (
        "afdb_mgnify_short_distillation_v20260725"
    )
    long_distillation_dataset = (
        "afdb_mgnify_long_distillation_v20260725"
    )
    short_long_disordered_train_sets = [
        "weightedPDB_before210930_v20260101",
        short_distillation_dataset,
        long_distillation_dataset,
        "openfold3_disordered_v20260725",
    ]
    if (
        is_short_long_disordered_profile
        and train_sets != short_long_disordered_train_sets
    ):
        raise ValueError(
            f"{short_long_disordered_profile} requires train_sets="
            f"{short_long_disordered_train_sets}, got {train_sets}"
        )
    if is_short_long_disordered_profile:
        expected_weights = short_long_disordered_profile_weights[profile_id]
        if train_sample_weights != expected_weights:
            raise ValueError(
                f"{short_long_disordered_profile} requires raw outer "
                f"weights={expected_weights}, got {train_sample_weights}"
            )
        expected_distillation = {
            "weightedPDB_before210930_v20260101": False,
            short_distillation_dataset: True,
            long_distillation_dataset: True,
            "openfold3_disordered_v20260725": True,
        }
        for dataset_name, expected_value in expected_distillation.items():
            policy = effective_loss_and_crop_policy_by_dataset[
                dataset_name
            ]
            if policy["is_distillation"] is not expected_value:
                raise ValueError(
                    f"{short_long_disordered_profile} requires "
                    f"{dataset_name}.is_distillation="
                    f"{str(expected_value).lower()}"
                )
            if policy["min_diffusion_supervised_tokens"] != 4:
                raise ValueError(
                    f"{short_long_disordered_profile} requires "
                    f"{dataset_name}.min_diffusion_supervised_tokens=4"
                )
        for dataset_name in (
            short_distillation_dataset,
            long_distillation_dataset,
            "openfold3_disordered_v20260725",
        ):
            if not effective_loss_and_crop_policy_by_dataset[
                dataset_name
            ]["require_resolved_protein_crop"]:
                raise ValueError(
                    f"{short_long_disordered_profile} requires "
                    f"{dataset_name}.require_resolved_protein_crop=true"
                )
        pdb_policy = effective_loss_and_crop_policy_by_dataset[
            "weightedPDB_before210930_v20260101"
        ]
        if pdb_policy["require_resolved_protein_crop"]:
            raise ValueError(
                f"{short_long_disordered_profile} requires "
                "weightedPDB_before210930_v20260101."
                "require_resolved_protein_crop=false"
            )
        for dataset_name in (
            "weightedPDB_before210930_v20260101",
            short_distillation_dataset,
            long_distillation_dataset,
        ):
            if effective_loss_and_crop_policy_by_dataset[dataset_name][
                "disable_non_protein_diffusion_weights"
            ]:
                raise ValueError(
                    f"{short_long_disordered_profile} requires "
                    f"{dataset_name}."
                    "disable_non_protein_diffusion_weights=false"
                )
        if not effective_loss_and_crop_policy_by_dataset[
            "openfold3_disordered_v20260725"
        ]["disable_non_protein_diffusion_weights"]:
            raise ValueError(
                f"{short_long_disordered_profile} requires "
                "openfold3_disordered_v20260725."
                "disable_non_protein_diffusion_weights=true"
            )
        if (
            "openfold3_disordered_v20260725"
            not in immutable_dataset_artifacts
        ):
            raise ValueError(
                f"{short_long_disordered_profile} requires immutable "
                "disordered indices/_SUCCESS/manifest artifacts"
            )
        for dataset_name in (
            short_distillation_dataset,
            long_distillation_dataset,
        ):
            if dataset_name not in runtime_distillation_indices:
                raise ValueError(
                    f"{short_long_disordered_profile} requires "
                    f"{dataset_name} runtime-index manifest"
                )
        if (
            runtime_distillation_indices[short_distillation_dataset][
                "runtime_index_dir"
            ]
            == runtime_distillation_indices[long_distillation_dataset][
                "runtime_index_dir"
            ]
        ):
            raise ValueError(
                f"{short_long_disordered_profile} requires distinct "
                "Short and Long runtime-index directories"
            )
        expected_partition_counts = {
            short_distillation_dataset: {
                "row_count": 15_425_587,
                "cluster_count": 14_327_216,
            },
            long_distillation_dataset: {
                "row_count": 18_073_064,
                "cluster_count": 16_163_340,
            },
        }
        for dataset_name, expected_counts in (
            expected_partition_counts.items()
        ):
            for count_name, expected_count in expected_counts.items():
                actual_count = runtime_distillation_indices[dataset_name][
                    count_name
                ]
                if actual_count != expected_count:
                    raise ValueError(
                        f"{short_long_disordered_profile} requires "
                        f"{dataset_name}.{count_name}={expected_count}, got "
                        f"{actual_count!r}"
                    )
        partition_row_count = sum(
            int(
                runtime_distillation_indices[dataset_name]["row_count"]
            )
            for dataset_name in (
                short_distillation_dataset,
                long_distillation_dataset,
            )
        )
        if partition_row_count != 33_498_651:
            raise ValueError(
                f"{short_long_disordered_profile} requires Short+Long "
                "row_count=33498651, got "
                f"{partition_row_count}"
            )
        expected_partition_roles = {
            short_distillation_dataset: "short_le200",
            long_distillation_dataset: "long_gt200",
        }
        expected_partition_criterion = {
            "representative": (
                "final representative_row_id from clusters.csv"
            ),
            "short": "canonical_length <= 200",
            "long": "canonical_length > 200",
            "max_representative_length": 200,
        }
        expected_partition_pair_id = (
            "9c87d52dd446f622be689b7a09611f302d3791ac2448ecd86f8aee2488fb8734"
        )
        partition_receipts = {}
        partition_artifact_hashes = set()
        partition_pair_ids = set()
        for dataset_name, expected_role in expected_partition_roles.items():
            runtime_contract = runtime_distillation_indices[dataset_name]
            partition_contract = runtime_contract.get("partition")
            if not isinstance(partition_contract, Mapping):
                raise ValueError(
                    f"{short_long_disordered_profile} requires "
                    f"{dataset_name} partition metadata and receipt"
                )
            metadata = partition_contract.get("manifest_metadata")
            receipt = partition_contract.get("receipt")
            artifact = partition_contract.get("receipt_artifact")
            if (
                not isinstance(metadata, Mapping)
                or not isinstance(receipt, Mapping)
                or not isinstance(artifact, Mapping)
            ):
                raise ValueError(
                    f"{dataset_name} has an incomplete partition contract"
                )
            pair_id = metadata.get("pair_id")
            criterion = metadata.get("criterion")
            if (
                metadata.get("schema_version") != 1
                or metadata.get("role") != expected_role
                or metadata.get("member_policy")
                != "retain_whole_source_cluster"
                or metadata.get("source_cluster_id_policy")
                != "preserve_stable_cluster_id"
                or not isinstance(criterion, Mapping)
                or dict(criterion) != expected_partition_criterion
                or not isinstance(pair_id, str)
                or len(pair_id) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in pair_id
                )
            ):
                raise ValueError(
                    f"{dataset_name} has invalid length-partition metadata "
                    f"for role {expected_role}"
                )
            source_row_count = metadata.get("source_row_count")
            source_cluster_count = metadata.get("source_cluster_count")
            if (
                source_row_count != 33_498_651
                or source_cluster_count != 30_490_556
            ):
                raise ValueError(
                    f"{dataset_name} partition metadata has incorrect "
                    "source coverage counts"
                )
            receipt_source = receipt.get("source")
            receipt_coverage = receipt.get("coverage")
            receipt_partitions = receipt.get("partitions")
            if (
                receipt.get("schema_version") != 1
                or receipt.get("format")
                != "protenix_distillation_length_partition_pair"
                or receipt.get("pair_id") != pair_id
                or receipt.get("criterion") != criterion
                or receipt.get("member_policy")
                != "retain_whole_source_cluster"
                or not isinstance(receipt_source, Mapping)
                or receipt_source.get("row_count") != 33_498_651
                or receipt_source.get("cluster_count") != 30_490_556
                or not isinstance(receipt_coverage, Mapping)
                or receipt_coverage.get("row_count_sum") != 33_498_651
                or receipt_coverage.get("cluster_count_sum") != 30_490_556
                or receipt_coverage.get("overlap") != 0
                or not isinstance(receipt_partitions, Mapping)
            ):
                raise ValueError(
                    f"{dataset_name} has an invalid partition-pair receipt"
                )
            source_hashes = {}
            for source_name in (
                "clusters",
                "short_paths",
                "short_paths_stats",
                "canonical_audit",
                "canonical_audit_success",
            ):
                source_record = receipt_source.get(source_name)
                source_sha256 = (
                    source_record.get("sha256")
                    if isinstance(source_record, Mapping)
                    else None
                )
                if (
                    not isinstance(source_sha256, str)
                    or len(source_sha256) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in source_sha256
                    )
                ):
                    raise ValueError(
                        f"{dataset_name} partition receipt has invalid "
                        f"{source_name} SHA-256"
                    )
                source_hashes[source_name] = source_sha256
            full_manifest_sha256 = receipt_source.get(
                "full_runtime_manifest_sha256"
            )
            if (
                not isinstance(full_manifest_sha256, str)
                or len(full_manifest_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in full_manifest_sha256
                )
            ):
                raise ValueError(
                    f"{dataset_name} partition receipt has invalid full "
                    "runtime manifest SHA-256"
                )
            pair_identity_material = {
                "partition_schema_version": 1,
                "full_runtime_manifest_sha256": full_manifest_sha256,
                "clusters_sha256": source_hashes["clusters"],
                "short_paths_sha256": source_hashes["short_paths"],
                "short_stats_sha256": source_hashes[
                    "short_paths_stats"
                ],
                "canonical_audit_sha256": source_hashes[
                    "canonical_audit"
                ],
                "canonical_audit_success_sha256": source_hashes[
                    "canonical_audit_success"
                ],
                "max_representative_length": 200,
            }
            computed_pair_id = hashlib.sha256(
                json.dumps(
                    pair_identity_material,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if computed_pair_id != pair_id:
                raise ValueError(
                    f"{dataset_name} partition pair_id does not match its "
                    "recorded source hashes"
                )
            if pair_id != expected_partition_pair_id:
                raise ValueError(
                    f"{short_long_disordered_profile} requires production "
                    f"partition pair_id={expected_partition_pair_id}, got "
                    f"{pair_id}"
                )
            for member_dataset, member_role in (
                expected_partition_roles.items()
            ):
                member = receipt_partitions.get(member_role)
                member_runtime = runtime_distillation_indices[member_dataset]
                if (
                    not isinstance(member, Mapping)
                    or member.get("row_count")
                    != member_runtime["row_count"]
                    or member.get("cluster_count")
                    != member_runtime["cluster_count"]
                ):
                    raise ValueError(
                        f"{dataset_name} partition-pair receipt disagrees "
                        f"with {member_role} runtime counts"
                    )
            partition_receipts[dataset_name] = receipt
            partition_artifact_hashes.add(artifact.get("sha256"))
            partition_pair_ids.add(pair_id)
        if (
            len(partition_pair_ids) != 1
            or len(partition_artifact_hashes) != 1
            or partition_receipts[short_distillation_dataset]
            != partition_receipts[long_distillation_dataset]
        ):
            raise ValueError(
                f"{short_long_disordered_profile} requires one byte-identical "
                "Short/Long partition-pair receipt"
            )
        if not virtual_segment_shuffle_enabled:
            raise ValueError(
                f"{short_long_disordered_profile} requires "
                "data.msa.enable_virtual_segment_msa_shuffle=true"
            )
        if (
            virtual_segment_residue_gap != 24
            or virtual_segment_min_length != 9
        ):
            raise ValueError(
                f"{short_long_disordered_profile} requires virtual segment "
                "residue_gap=24 and min_segment_length=9"
            )
        if not multichain_shuffle_variant_sampling_enabled:
            raise ValueError(
                f"{short_long_disordered_profile} requires "
                "enable_multichain_msa_shuffle_variant_sampling=true"
            )
        if multichain_shuffle_segment_probability != 0.5:
            raise ValueError(
                f"{short_long_disordered_profile} requires "
                "multichain_msa_shuffle_segment_probability=0.5"
            )
        if not partial_query_only_enabled:
            raise ValueError(
                f"{short_long_disordered_profile} requires "
                "data.msa.enable_crop_aware_partial_query_only=true"
            )
        if partial_query_only_max_chains != 2:
            raise ValueError(
                f"{short_long_disordered_profile} requires "
                "partial_query_only_max_chains=2"
            )
        for dataset_name in short_long_disordered_train_sets:
            effective_msa = strict_effective_runtime_semantics_by_dataset[
                dataset_name
            ]["effective_msa"]
            if not strict_bool(
                effective_msa,
                "enable_virtual_segment_msa_shuffle",
            ):
                raise ValueError(
                    f"{short_long_disordered_profile} requires "
                    f"{dataset_name} effective virtual-segment shuffle"
                )
            if (
                strict_nonnegative_int(
                    effective_msa,
                    "virtual_segment_residue_gap",
                    24,
                )
                != 24
                or strict_nonnegative_int(
                    effective_msa,
                    "virtual_segment_min_length",
                    9,
                )
                != 9
            ):
                raise ValueError(
                    f"{short_long_disordered_profile} requires "
                    f"{dataset_name} effective gap24/min9"
                )
            if not strict_bool(
                effective_msa,
                "enable_multichain_msa_shuffle_variant_sampling",
            ):
                raise ValueError(
                    f"{short_long_disordered_profile} requires "
                    f"{dataset_name} effective multichain variant sampling"
                )
            if (
                strict_probability(
                    effective_msa,
                    "multichain_msa_shuffle_segment_probability",
                    0.5,
                )
                != 0.5
            ):
                raise ValueError(
                    f"{short_long_disordered_profile} requires "
                    f"{dataset_name} effective segment probability 0.5"
                )
            if not strict_bool(
                effective_msa,
                "enable_crop_aware_partial_query_only",
            ) or (
                strict_nonnegative_int(
                    effective_msa,
                    "partial_query_only_max_chains",
                    2,
                )
                != 2
            ):
                raise ValueError(
                    f"{short_long_disordered_profile} requires "
                    f"{dataset_name} effective partial query-only max2"
                )
            augmentation = effective_augmentation_by_dataset[dataset_name]
            if (
                not math.isclose(
                    augmentation["query_only_msa_probability"],
                    0.1,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    augmentation[
                        "pair_msa_as_unpaired_probability"
                    ],
                    0.3,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    augmentation["species_paired_msa_probability"],
                    0.6,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"{short_long_disordered_profile} requires "
                    f"{dataset_name} MSA mode probabilities "
                    "query/unpaired/species=0.1/0.3/0.6"
                )

    uses_extended_loss_or_crop_policy = any(
        policy["disable_non_protein_diffusion_weights"]
        or policy["require_resolved_protein_crop"]
        or policy["min_diffusion_supervised_tokens"] > 0
        for policy in effective_loss_and_crop_policy_by_dataset.values()
    )
    if strict_data_semantics_enabled:
        payload["schema_version"] = 7
        payload["effective_sampler_seed"] = resolve_effective_sampler_seed()
        payload["sampler_seed_resolution"] = (
            "negative_or_none_inherits_global_seed_v1"
        )
        payload["dataset_runtime_semantics"] = {
            "effective_by_dataset": (
                strict_effective_runtime_semantics_by_dataset
            ),
        }
        if uses_extended_loss_or_crop_policy:
            payload["loss_and_crop_policy"] = {
                "effective_by_dataset": (
                    effective_loss_and_crop_policy_by_dataset
                ),
            }
            if immutable_dataset_artifacts:
                payload["immutable_dataset_artifacts"] = (
                    immutable_dataset_artifacts
                )
    elif uses_extended_loss_or_crop_policy:
        payload["schema_version"] = 6
        payload["loss_and_crop_policy"] = {
            "effective_by_dataset": (
                effective_loss_and_crop_policy_by_dataset
            ),
        }
        payload["dataset_runtime_semantics"] = {
            "effective_by_dataset": effective_runtime_semantics_by_dataset,
        }
        if immutable_dataset_artifacts:
            payload["immutable_dataset_artifacts"] = (
                immutable_dataset_artifacts
            )
    elif runtime_distillation_indices:
        # Runtime-index-backed distillation predates the disordered-v2
        # supervision contract and retains its own schema identity.
        payload["schema_version"] = 4
    else:
        # Preserve byte-for-byte schema-3 identities for existing PDB-only
        # checkpoints when all newly introduced policies are inactive. Active
        # model-side query inclusion is recorded in the common augmentation
        # payload above and intentionally changes the strict-resume data hash.
        payload["schema_version"] = 3

    if runtime_distillation_indices:
        payload["runtime_distillation_indices"] = (
            runtime_distillation_indices
        )

    soft_cluster_schedule_path = str(
        _get_value(
            sampler_cfg,
            "soft_cluster_schedule_manifest",
            "",
        )
        or ""
    ).strip()
    if soft_cluster_schedule_path:
        from protenix.data.pipeline.soft_cluster_schedule import (
            build_soft_cluster_schedule_contract,
            load_soft_cluster_schedule_manifest,
        )

        schedule_info = load_soft_cluster_schedule_manifest(
            soft_cluster_schedule_path
        )
        payload["soft_cluster_schedule"] = (
            build_soft_cluster_schedule_contract(schedule_info)
        )
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **payload,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def build_train_dataloader_runtime_contract(configs: Any) -> Dict[str, Any]:
    """Resolve the operational DataLoader settings that strict resume locks."""
    data_cfg = _get_value(configs, "data", {})
    num_workers = int(_get_value(data_cfg, "num_dl_workers", 0))
    if num_workers < 0:
        raise ValueError(f"data.num_dl_workers must be >= 0, got {num_workers}")
    if num_workers == 0:
        return {
            "num_workers": 0,
            "persistent_workers": False,
            "prefetch_factor": None,
            "pin_memory": False,
        }
    prefetch_factor = int(
        _get_value(data_cfg, "train_dl_prefetch_factor", 2)
    )
    if prefetch_factor <= 0:
        raise ValueError(
            "data.train_dl_prefetch_factor must be > 0 when "
            f"data.num_dl_workers > 0, got {prefetch_factor}"
        )
    return {
        "num_workers": num_workers,
        "persistent_workers": bool(
            _get_value(data_cfg, "train_dl_persistent_workers", True)
        ),
        "prefetch_factor": prefetch_factor,
        "pin_memory": False,
    }


def build_model_architecture_contract(configs: Any) -> Dict[str, Any]:
    """Build a canonical identity for parameter-shaping model configuration."""

    def optional_int(config: Any, key: str, default: Any = None) -> int | None:
        value = _get_value(config, key, default)
        return None if value is None else int(value)

    def optional_bool(
        config: Any,
        key: str,
        default: Any = None,
    ) -> bool | None:
        value = _get_value(config, key, default)
        return None if value is None else bool(value)

    model_cfg = _get_value(configs, "model", {})
    input_embedder_cfg = _get_value(model_cfg, "input_embedder", {})
    relative_position_cfg = _get_value(
        model_cfg,
        "relative_position_encoding",
        {},
    )
    template_cfg = _get_value(model_cfg, "template_embedder", {})
    msa_cfg = _get_value(model_cfg, "msa_module", {})
    pairformer_cfg = _get_value(model_cfg, "pairformer", {})
    diffusion_cfg = _get_value(model_cfg, "diffusion_module", {})
    diffusion_atom_encoder_cfg = _get_value(diffusion_cfg, "atom_encoder", {})
    diffusion_transformer_cfg = _get_value(diffusion_cfg, "transformer", {})
    diffusion_atom_decoder_cfg = _get_value(diffusion_cfg, "atom_decoder", {})
    confidence_cfg = _get_value(model_cfg, "confidence_head", {})
    distogram_cfg = _get_value(model_cfg, "distogram_head", {})
    constraint_cfg = _get_value(model_cfg, "constraint_embedder", {})
    pocket_cfg = _get_value(constraint_cfg, "pocket_embedder", {})
    contact_cfg = _get_value(constraint_cfg, "contact_embedder", {})
    substructure_cfg = _get_value(constraint_cfg, "substructure_embedder", {})
    contact_atom_cfg = _get_value(constraint_cfg, "contact_atom_embedder", {})
    esm_cfg = _get_value(configs, "esm", {})

    payload = {
        "schema_version": 1,
        "model_name": str(_get_value(configs, "model_name", "") or ""),
        "global": {
            "c_s": optional_int(configs, "c_s"),
            "c_z": optional_int(configs, "c_z"),
            "c_s_inputs": optional_int(configs, "c_s_inputs"),
            "c_atom": optional_int(configs, "c_atom"),
            "c_atompair": optional_int(configs, "c_atompair"),
            "c_token": optional_int(configs, "c_token"),
            "max_atoms_per_token": optional_int(configs, "max_atoms_per_token"),
            "no_bins": optional_int(configs, "no_bins"),
        },
        "input_embedder": {
            "c_atom": optional_int(input_embedder_cfg, "c_atom"),
            "c_atompair": optional_int(input_embedder_cfg, "c_atompair"),
            "c_token": optional_int(input_embedder_cfg, "c_token"),
            "esm_enable": optional_bool(esm_cfg, "enable", False),
            "esm_embedding_dim": optional_int(esm_cfg, "embedding_dim"),
        },
        "relative_position_encoding": {
            "r_max": optional_int(relative_position_cfg, "r_max", 32),
            "s_max": optional_int(relative_position_cfg, "s_max", 2),
            "c_z": optional_int(relative_position_cfg, "c_z"),
        },
        "template_embedder": {
            "c": optional_int(template_cfg, "c", 64),
            "c_z": optional_int(template_cfg, "c_z"),
            "n_blocks": optional_int(template_cfg, "n_blocks", 2),
            "num_intermediate_factor": optional_int(
                template_cfg,
                "num_intermediate_factor",
                2,
            ),
            "hidden_scale_up": optional_bool(
                template_cfg,
                "hidden_scale_up",
                False,
            ),
        },
        "msa_module": {
            "c_m": optional_int(msa_cfg, "c_m", 64),
            "c_z": optional_int(msa_cfg, "c_z"),
            "c_s_inputs": optional_int(msa_cfg, "c_s_inputs"),
            "n_blocks": optional_int(msa_cfg, "n_blocks", 4),
            "hidden_scale_up": optional_bool(
                msa_cfg,
                "hidden_scale_up",
                False,
            ),
        },
        "pairformer": {
            "n_blocks": optional_int(pairformer_cfg, "n_blocks", 48),
            "n_heads": optional_int(pairformer_cfg, "n_heads", 16),
            "c_z": optional_int(pairformer_cfg, "c_z"),
            "c_s": optional_int(pairformer_cfg, "c_s"),
            "num_intermediate_factor": optional_int(
                pairformer_cfg,
                "num_intermediate_factor",
                4,
            ),
            "hidden_scale_up": optional_bool(
                pairformer_cfg,
                "hidden_scale_up",
                False,
            ),
        },
        "diffusion_module": {
            "c_token": optional_int(diffusion_cfg, "c_token", 768),
            "c_atom": optional_int(diffusion_cfg, "c_atom"),
            "c_atompair": optional_int(diffusion_cfg, "c_atompair"),
            "c_z": optional_int(diffusion_cfg, "c_z"),
            "c_s": optional_int(diffusion_cfg, "c_s"),
            "c_s_inputs": optional_int(diffusion_cfg, "c_s_inputs"),
            "c_noise_embedding": optional_int(
                diffusion_cfg,
                "c_noise_embedding",
                256,
            ),
            "atom_encoder": {
                "n_blocks": optional_int(
                    diffusion_atom_encoder_cfg,
                    "n_blocks",
                    3,
                ),
                "n_heads": optional_int(
                    diffusion_atom_encoder_cfg,
                    "n_heads",
                    4,
                ),
                "n_queries": optional_int(
                    diffusion_atom_encoder_cfg,
                    "n_queries",
                    32,
                ),
                "n_keys": optional_int(
                    diffusion_atom_encoder_cfg,
                    "n_keys",
                    128,
                ),
            },
            "transformer": {
                "n_blocks": optional_int(
                    diffusion_transformer_cfg,
                    "n_blocks",
                    24,
                ),
                "n_heads": optional_int(
                    diffusion_transformer_cfg,
                    "n_heads",
                    16,
                ),
            },
            "atom_decoder": {
                "n_blocks": optional_int(
                    diffusion_atom_decoder_cfg,
                    "n_blocks",
                    3,
                ),
                "n_heads": optional_int(
                    diffusion_atom_decoder_cfg,
                    "n_heads",
                    4,
                ),
            },
        },
        "confidence_head": {
            "n_blocks": optional_int(confidence_cfg, "n_blocks", 4),
            "c_z": optional_int(confidence_cfg, "c_z"),
            "c_s": optional_int(confidence_cfg, "c_s"),
            "c_s_inputs": optional_int(confidence_cfg, "c_s_inputs"),
            "b_pae": optional_int(confidence_cfg, "b_pae", 64),
            "b_pde": optional_int(confidence_cfg, "b_pde", 64),
            "b_plddt": optional_int(confidence_cfg, "b_plddt", 50),
            "b_resolved": optional_int(confidence_cfg, "b_resolved", 2),
            "max_atoms_per_token": optional_int(
                confidence_cfg,
                "max_atoms_per_token",
            ),
            "hidden_scale_up": optional_bool(
                confidence_cfg,
                "hidden_scale_up",
                False,
            ),
        },
        "distogram_head": {
            "c_z": optional_int(distogram_cfg, "c_z"),
            "no_bins": optional_int(distogram_cfg, "no_bins"),
        },
        "constraint_embedder": {
            "c_constraint_z": optional_int(constraint_cfg, "c_constraint_z"),
            "c_constraint_s": optional_int(constraint_cfg, "c_constraint_s"),
            "c_constraint_atom_pair": optional_int(
                constraint_cfg,
                "c_constraint_atom_pair",
            ),
            "pocket": {
                "enable": optional_bool(pocket_cfg, "enable", False),
                "c_s_input": optional_int(pocket_cfg, "c_s_input"),
                "c_z_input": optional_int(pocket_cfg, "c_z_input"),
            },
            "contact": {
                "enable": optional_bool(contact_cfg, "enable", False),
                "c_z_input": optional_int(contact_cfg, "c_z_input"),
            },
            "substructure": {
                "enable": optional_bool(substructure_cfg, "enable", False),
                "hidden_dim": optional_int(substructure_cfg, "hidden_dim"),
                "n_layers": optional_int(substructure_cfg, "n_layers"),
                "n_classes": optional_int(substructure_cfg, "n_classes"),
            },
            "contact_atom": {
                "enable": optional_bool(contact_atom_cfg, "enable", False),
                "c_z_input": optional_int(contact_atom_cfg, "c_z_input"),
            },
        },
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **payload,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def build_run_manifest(configs: Any, run_name: str, run_dir: str) -> Dict[str, Any]:
    """Build a compact manifest for reproducing and auditing a training run."""
    data_cfg = _get_value(configs, "data", {})
    train_sampler_cfg = _get_value(data_cfg, "train_sampler", {})
    training_data_contract = build_training_data_contract(configs)
    train_dataloader_runtime = build_train_dataloader_runtime_contract(configs)
    model_architecture_contract = build_model_architecture_contract(configs)
    msa_cfg = _get_value(data_cfg, "msa", {})
    model_cfg = _get_value(configs, "model", {})
    pairformer_cfg = _get_value(model_cfg, "pairformer", {})
    msa_module_cfg = _get_value(model_cfg, "msa_module", {})
    diffusion_module_cfg = _get_value(model_cfg, "diffusion_module", {})
    diffusion_cfg = _get_value(configs, "sample_diffusion", {})
    adam_cfg = _get_value(configs, "adam", {})
    ddp_cfg = _get_value(configs, "ddp", {})
    benchmark_dataset_name = os.environ.get("BENCHMARK_DATASET_NAME", "").strip()
    allowed_benchmark_datasets = {
        "benchmark_long_protein_context_384_800_msa2048",
        "benchmark_long_protein_context_768",
        "benchmark_long_protein_context_1024",
        "benchmark_long_protein_context_1024_800",
        "benchmark_long_protein_context_1024_800_msa2048",
        "benchmark_long_protein_context_960_800_msa2048",
    }
    if benchmark_dataset_name not in allowed_benchmark_datasets:
        configured_train_sets = _get_value(data_cfg, "train_sets", []) or []
        benchmark_dataset_name = next(
            (
                str(name)
                for name in configured_train_sets
                if str(name) in allowed_benchmark_datasets
            ),
            "benchmark_long_protein_context_768",
        )
    benchmark_dataset_cfg = _get_value(data_cfg, benchmark_dataset_name, {})
    benchmark_base_info = _get_value(benchmark_dataset_cfg, "base_info", {})
    benchmark_index_path = str(
        _get_value(benchmark_base_info, "indices_fpath", "") or ""
    )
    benchmark_artifacts: Dict[str, Any] = {"index_path": benchmark_index_path}
    if benchmark_index_path:
        benchmark_summary_path = os.path.join(
            os.path.dirname(os.path.abspath(benchmark_index_path)), "summary.json"
        )
        benchmark_artifacts["summary_path"] = benchmark_summary_path
        try:
            with open(benchmark_index_path, "rb") as handle:
                digest = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            benchmark_artifacts["index_sha256"] = digest.hexdigest()
            with open(benchmark_summary_path, "r", encoding="utf-8") as handle:
                benchmark_artifacts["selection_summary"] = json.load(handle)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            benchmark_artifacts["metadata_error"] = str(exc)
    benchmark_code_files = (
        "benchmark.sh",
        "finetune_submit.sh",
        "finetune_demo.sh",
        "run_crop384_db48_ddp_scratch_50k.sh",
        "run_crop384_db48_ddp_scratch_200k.sh",
        "run_crop384_db48_ddp_disordered_v4_from_v8_step999.sh",
        "resume_crop384_db48_ddp_scratch_50k.sh",
        "resume_crop384_db48_ddp_scratch_200k.sh",
        "resume_crop384_db48_ddp_disordered_v4.sh",
        "configs/crop384_db48_data_profiles.sh",
        "configs/configs_base.py",
        "configs/configs_data.py",
        "runner/train.py",
        "runner/ema.py",
        "protenix/data/pipeline/sample_context.py",
        "protenix/data/pipeline/dataset.py",
        "protenix/data/pipeline/data_pipeline.py",
        "protenix/data/msa/msa_featurizer.py",
        "protenix/data/msa/msa_utils.py",
        "protenix/data/template/template_featurizer.py",
        "protenix/data/template/template_utils.py",
        "protenix/data/template/sequence_template_cache.py",
        "protenix/utils/fsdp.py",
        "protenix/model/protenix.py",
        "protenix/model/modules/primitives.py",
        "protenix/model/modules/pairformer.py",
        "protenix/model/modules/diffusion.py",
        "protenix/model/layer_norm/cache_provenance.py",
        "protenix/model/layer_norm/layer_norm.py",
        "protenix/model/layer_norm/torch_ext_compile.py",
        "protenix/model/layer_norm/kernel/layer_norm_cuda.cpp",
        "protenix/model/layer_norm/kernel/layer_norm_cuda_kernel.cu",
        "scripts/benchmark/build_long_protein_benchmark.py",
        "scripts/benchmark/build_msa_aware_long_protein_benchmark.py",
        "scripts/benchmark/audit_msa_depth.py",
        "scripts/benchmark/compare_aistudio_benchmarks.py",
        "scripts/benchmark/compare_implementation_benchmarks.py",
        "scripts/benchmark/run_aistudio_benchmark.sh",
        "scripts/benchmark/summarize_aistudio_benchmark.py",
        "scripts/runtime/cuda126_env.sh",
        "scripts/runtime/sequence_template_cache_contract.sh",
        "scripts/runtime/validate_v8_step999_warmstart_pair.py",
    )
    code_sha256: Dict[str, str] = {}
    for relative_path in benchmark_code_files:
        full_path = os.path.join(os.getcwd(), relative_path)
        try:
            digest = hashlib.sha256()
            with open(full_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            code_sha256[relative_path] = digest.hexdigest()
        except OSError:
            code_sha256[relative_path] = ""
    layer_norm_extension_path = getattr(
        sys.modules.get("fast_layer_norm_cuda_v2"),
        "__file__",
        None,
    )
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "created_at": _now_timestamp(),
        "command": sys.argv,
        "cwd": os.getcwd(),
        "project": _get_value(configs, "project"),
        "model_name": _get_value(configs, "model_name"),
        "seed": _get_value(configs, "seed"),
        "training": {
            "load_params_only": _get_value(configs, "load_params_only", True),
            "dtype": _get_value(configs, "dtype"),
            "train_crop_size": _get_value(configs, "train_crop_size"),
            "diffusion_batch_size": _get_value(configs, "diffusion_batch_size"),
            "diffusion_chunk_size": _get_value(configs, "diffusion_chunk_size"),
            "iters_to_accumulate": _get_value(configs, "iters_to_accumulate"),
            "ema_decay": _get_value(configs, "ema_decay"),
            "ema_shadow_device": _get_value(configs, "ema_shadow_device"),
            "lr": _get_value(configs, "lr"),
            "lr_warmup_steps": _get_value(configs, "warmup_steps"),
            "triangle_attention": _get_value(configs, "triangle_attention"),
            "triangle_multiplicative": _get_value(
                configs, "triangle_multiplicative"
            ),
            "n_cycle": _get_value(model_cfg, "N_cycle"),
            "diffusion_n_step": _get_value(diffusion_cfg, "N_step"),
            "adam": {
                "use_adamw": _get_value(adam_cfg, "use_adamw", False),
                "fused": _get_value(adam_cfg, "fused", False),
            },
        },
        "benchmark": {
            "variant": os.environ.get("BENCHMARK_VARIANT", ""),
            "workload_profile": os.environ.get(
                "BENCHMARK_WORKLOAD_PROFILE", ""
            ),
            "dataset_name": benchmark_dataset_name,
            "crop_size": _get_value(configs, "train_crop_size"),
            "dataset_rows": _int_or_zero(os.environ.get("BENCHMARK_DATASET_ROWS")),
            "warmup_steps": _get_value(configs, "benchmark_warmup_steps", -1),
            "measure_steps": _get_value(configs, "benchmark_measure_steps", -1),
            "skip_final_checkpoint": _get_value(
                configs, "skip_final_checkpoint", False
            ),
            "exact_crop": os.environ.get("BENCHMARK_EXACT_CROP", "false"),
            "expected_torch_cuda": os.environ.get("EXPECTED_TORCH_CUDA", ""),
            "official_checkpoint_sha256": os.environ.get(
                "EXPECTED_CHECKPOINT_SHA256", ""
            ),
            "initial_checkpoint_sha256": os.environ.get(
                "EXPECTED_CHECKPOINT_SHA256", ""
            ),
            "phase_cuda_memory": os.environ.get(
                "PROTENIX_BENCHMARK_PHASE_MEMORY", "false"
            ),
        },
        "model": {
            "architecture_contract": model_architecture_contract,
            "c_s": _get_value(configs, "c_s"),
            "c_z": _get_value(configs, "c_z"),
            "msa_c_m": _get_value(msa_module_cfg, "c_m"),
            "msa_blocks_per_ckpt": _get_value(
                msa_module_cfg, "blocks_per_ckpt"
            ),
            "msa_chunk_size": _get_value(msa_module_cfg, "msa_chunk_size"),
            "msa_checkpoint_cpu_offload_every_n": _get_value(
                msa_module_cfg, "checkpoint_cpu_offload_every_n"
            ),
            "pairformer_n_blocks": _get_value(pairformer_cfg, "n_blocks"),
            "pairformer_blocks_per_ckpt": _get_value(
                pairformer_cfg, "blocks_per_ckpt"
            ),
            "pairformer_checkpoint_cpu_offload_every_n": _get_value(
                pairformer_cfg, "checkpoint_cpu_offload_every_n"
            ),
            "pairformer_transition_chunk_size": _get_value(
                pairformer_cfg, "train_pair_transition_chunk_size"
            ),
            "pairformer_transition_checkpoint_chunks": _get_value(
                pairformer_cfg, "train_pair_transition_checkpoint_chunks", False
            ),
            "diffusion_transition_chunk_size": _get_value(
                diffusion_module_cfg, "train_pair_transition_chunk_size"
            ),
            "diffusion_transition_checkpoint_chunks": _get_value(
                diffusion_module_cfg,
                "train_pair_transition_checkpoint_chunks",
                False,
            ),
            "diffusion_c_s": _get_value(diffusion_module_cfg, "c_s"),
            "diffusion_c_token": _get_value(
                diffusion_module_cfg,
                "c_token",
            ),
        },
        "initial_checkpoint": {
            "model": _get_value(configs, "load_checkpoint_path", ""),
            "ema": _get_value(configs, "load_ema_checkpoint_path", ""),
            **(
                {
                    "mode": "params_only_source_pair",
                    "source_run_id": os.environ.get(
                        "PROTENIX_WARMSTART_SOURCE_RUN_ID",
                        "",
                    ),
                    "source_step": int(
                        os.environ["PROTENIX_WARMSTART_CHECKPOINT_STEP"]
                    ),
                    "checkpoint_pair_id": os.environ.get(
                        "PROTENIX_WARMSTART_SOURCE_CHECKPOINT_PAIR_ID",
                        "",
                    ),
                    "raw_sha256": os.environ.get(
                        "PROTENIX_WARMSTART_RAW_SHA256",
                        "",
                    ),
                    "ema_sha256": os.environ.get(
                        "PROTENIX_WARMSTART_EMA_SHA256",
                        "",
                    ),
                    "source_training_contract_id": os.environ.get(
                        "PROTENIX_WARMSTART_SOURCE_TRAINING_CONTRACT_ID",
                        "",
                    ),
                    "source_data_contract_sha256": os.environ.get(
                        "PROTENIX_WARMSTART_SOURCE_DATA_CONTRACT_SHA256",
                        "",
                    ),
                    "source_model_architecture_sha256": os.environ.get(
                        "PROTENIX_WARMSTART_SOURCE_MODEL_ARCHITECTURE_SHA256",
                        "",
                    ),
                    "imported": {
                        "live_model": "raw",
                        "ema_shadow": "ema",
                        "optimizer": False,
                        "scheduler": False,
                        "step": False,
                        "grad_scaler": False,
                        "rank_rng": False,
                        "sampler_cursor": False,
                    },
                    "target_start_step": 0,
                    "target_start_global_step": 0,
                }
                if os.environ.get("PROTENIX_INITIALIZATION_MODE")
                == "params_only_source_pair"
                else {}
            ),
        },
        "resume": {
            "raw_checkpoint": _get_value(configs, "load_checkpoint_path", ""),
            "ema_checkpoint": _get_value(
                configs, "load_ema_checkpoint_path", ""
            ),
            "expected_checkpoint_step": os.environ.get(
                "PROTENIX_RESUME_CHECKPOINT_STEP", ""
            ),
            "loaded_checkpoint_step": None,
            "next_training_step": None,
            "checkpoint_format_version": 2,
            "rank_runtime_state_schema_version": 1,
            "train_dataloader_state_schema_version": 1,
            "strict_full_resume_requested": (
                os.environ.get("PROTENIX_STRICT_FULL_RESUME", "0").strip()
                == "1"
            ),
            "training_contract_id": os.environ.get(
                "PROTENIX_TRAINING_CONTRACT_ID", ""
            ),
            "data_profile_id": training_data_contract["profile_id"],
            "data_contract_sha256": training_data_contract["sha256"],
            "checkpoint_pair_id": None,
            "data_resume_scope": "sampler_cursor_and_rank_main_process_rng",
            "data_worker_rng_state_checkpointed": (
                train_dataloader_runtime["num_workers"] == 0
            ),
            "train_dataloader_runtime": train_dataloader_runtime,
        },
        "data": {
            "profile_id": training_data_contract["profile_id"],
            "contract": training_data_contract,
            "train_sets": _get_value(data_cfg, "train_sets"),
            "test_sets": _get_value(data_cfg, "test_sets"),
            "num_dl_workers": _get_value(data_cfg, "num_dl_workers"),
            "train_dataloader_runtime": train_dataloader_runtime,
            "epoch_size": _get_value(data_cfg, "epoch_size"),
            "train_sample_weights": _get_value(
                train_sampler_cfg, "train_sample_weights"
            ),
            "train_sampler_seed": _get_value(train_sampler_cfg, "seed", -1),
            "train_sampler_replacement": _get_value(
                train_sampler_cfg, "replacement", True
            ),
            "msa": to_jsonable(msa_cfg),
            "benchmark_dataset": {
                "name": benchmark_dataset_name,
                "base_info": to_jsonable(
                    _get_value(benchmark_dataset_cfg, "base_info", {})
                ),
                "cropping_configs": to_jsonable(
                    _get_value(benchmark_dataset_cfg, "cropping_configs", {})
                ),
                "artifacts": benchmark_artifacts,
            },
        },
        "distributed": {
            "world_size": DIST_WRAPPER.world_size,
            "rank": DIST_WRAPPER.rank,
            "local_rank": DIST_WRAPPER.local_rank,
            "local_world_size": DIST_WRAPPER.local_world_size,
            "num_nodes": DIST_WRAPPER.num_nodes,
            "node_rank": DIST_WRAPPER.node_rank,
            "strategy": _get_value(configs, "distributed_strategy", "ddp"),
            "optimizer_sharding": _get_value(
                configs, "optimizer_sharding", "none"
            ),
            "ddp": {
                "gradient_as_bucket_view": _get_value(
                    ddp_cfg, "gradient_as_bucket_view", False
                )
            },
            "fsdp": to_jsonable(_get_value(configs, "fsdp", {})),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "cuda_home": os.environ.get("CUDA_HOME", ""),
            "cudacxx": os.environ.get("CUDACXX", ""),
            "cuda_toolchain_fingerprint": os.environ.get(
                "PROTENIX_CUDA_TOOLCHAIN_FINGERPRINT", ""
            ),
            "layer_norm_cache_fingerprint": os.environ.get(
                "PROTENIX_LAYER_NORM_CACHE_FINGERPRINT", ""
            ),
            "layer_norm_source_sha256": os.environ.get(
                "PROTENIX_LAYER_NORM_SOURCE_SHA256", ""
            ),
            "nvcc_sha256": os.environ.get(
                "PROTENIX_CUDA_NVCC_SHA256", ""
            ),
            "host_cc_sha256": os.environ.get(
                "PROTENIX_CUDA_HOST_CC_SHA256", ""
            ),
            "host_cxx_sha256": os.environ.get(
                "PROTENIX_CUDA_HOST_CXX_SHA256", ""
            ),
            "protenix_root_dir": os.environ.get("PROTENIX_ROOT_DIR", ""),
            "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR", ""),
            "torch_lib_dir": os.environ.get("PROTENIX_TORCH_LIB_DIR", ""),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
            "tmpdir": os.environ.get("TMPDIR", ""),
            "torch_multiprocessing_sharing_strategy": (
                torch.multiprocessing.get_sharing_strategy()
            ),
            "cc": os.environ.get("CC", ""),
            "cxx": os.environ.get("CXX", ""),
            "fused_layer_norm_extension": (
                os.path.realpath(layer_norm_extension_path)
                if layer_norm_extension_path
                else None
            ),
            "module_locations": {
                name: os.path.realpath(
                    getattr(sys.modules.get(name), "__file__", "") or ""
                )
                for name in ("configs", "protenix")
            },
        },
        "git": _git_summary(os.getcwd()),
        "code_sha256": code_sha256,
    }


def build_checkpoint_record(
    path: str,
    step: int,
    ema_suffix: str = "",
) -> Dict[str, Any]:
    """Build one checkpoint index record."""
    return {
        "event": "checkpoint",
        "path": path,
        "step": step,
        "ema_suffix": ema_suffix,
        "is_ema": bool(ema_suffix),
        "created_at": _now_timestamp(),
    }


def validate_checkpoint_header(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: str,
    load_params_only: bool,
) -> str:
    """Validate format/kind before any live training state is modified."""
    format_version = checkpoint.get("format_version")
    if format_version is not None and format_version != 2:
        raise ValueError(f"unsupported checkpoint format_version: {format_version!r}")
    if format_version == 2 and "checkpoint_kind" not in checkpoint:
        raise ValueError("format_version=2 checkpoint must define checkpoint_kind")

    checkpoint_kind = checkpoint.get("checkpoint_kind", "legacy")
    if checkpoint_kind not in {"legacy", "raw", "ema"}:
        raise ValueError(f"unsupported checkpoint_kind: {checkpoint_kind!r}")

    if not load_params_only:
        if checkpoint_kind == "ema":
            raise ValueError(
                "EMA checkpoints are params-only snapshots and cannot be used "
                "for full optimizer resume"
            )
        # Historical checkpoints had no kind tag. Their conventional filename
        # is the only available guard against mixing EMA parameters with raw
        # optimizer moments during a full resume.
        if checkpoint_kind == "legacy" and "_ema" in os.path.basename(
            checkpoint_path
        ).lower():
            raise ValueError(
                "legacy EMA-named checkpoints cannot be used for full optimizer "
                "resume; use params-only loading"
            )
    return checkpoint_kind


def validate_expected_resume_checkpoint_step(
    checkpoint_step: Any,
    *,
    checkpoint_path: str,
) -> None:
    """Verify a full-resume checkpoint against the launcher-declared step."""
    expected_text = os.environ.get("PROTENIX_RESUME_CHECKPOINT_STEP", "").strip()
    if not expected_text:
        return
    try:
        expected_step = int(expected_text)
    except ValueError as exc:
        raise ValueError(
            "PROTENIX_RESUME_CHECKPOINT_STEP must be an integer, got "
            f"{expected_text!r}"
        ) from exc
    if checkpoint_step is None:
        raise ValueError(
            "full resume checkpoint has no step, but "
            f"PROTENIX_RESUME_CHECKPOINT_STEP={expected_step} was requested"
        )
    actual_step = int(checkpoint_step)
    if actual_step != expected_step:
        raise ValueError(
            "full resume checkpoint step mismatch: "
            f"checkpoint {checkpoint_path} contains step={actual_step}, "
            f"PROTENIX_RESUME_CHECKPOINT_STEP={expected_step}"
        )


STRICT_FULL_RESUME_SCHEMA_VERSION = 3
_MIGRATION_EXPECTATIONS_UNSET = object()
STRICT_RESUME_MSA_QUERY_PIN_MIGRATION_V1 = (
    "msa_query_pin_v10_to_v11_full_state_v1"
)
STRICT_RESUME_SOFT_CLUSTER_SCHEDULE_MIGRATION_V1 = (
    "soft_cluster_schedule_manifest_full_state_v1"
)
STRICT_RESUME_SOFT_CLUSTER_CATALOG_REPLACEMENT_MIGRATION_V2 = (
    "soft_cluster_catalog_replacement_full_state_v2"
)
STRICT_RESUME_WEIGHTEDPDB_MSA_INDEPENDENT_MIGRATION_V1 = (
    "weightedpdb_independent_pairing_nonpairing_full_state_v1"
)
SOFT_CLUSTER_SCHEDULE_DIR_ENV = "PROTENIX_SOFT_CLUSTER_SCHEDULE_DIR"
SOFT_CLUSTER_MIGRATION_MODES = frozenset(
    {
        STRICT_RESUME_SOFT_CLUSTER_SCHEDULE_MIGRATION_V1,
        STRICT_RESUME_SOFT_CLUSTER_CATALOG_REPLACEMENT_MIGRATION_V2,
    }
)
CHECKPOINT_BOUND_MIGRATION_MODES = frozenset(
    set(SOFT_CLUSTER_MIGRATION_MODES)
    | {STRICT_RESUME_WEIGHTEDPDB_MSA_INDEPENDENT_MIGRATION_V1}
)
WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_RUN_ID = (
    "protenix_v4_step21999_residual_ge200_softcluster_g02_"
    "v9_straggler_20260804_140441"
)
WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_TRAINING_CONTRACT_ID = (
    "crop384_db48_ddp_disordered_v5_ws128_acc4_"
    "catalogreplacement_residual_ge200_20260804_softcluster_g02_"
    "checkpointbound_step21999_wave_schedule_v2"
)
WEIGHTEDPDB_MSA_INDEPENDENT_TARGET_TRAINING_CONTRACT_ID = (
    WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_TRAINING_CONTRACT_ID
    + "_weightedpdb_pairing_nonpairing_independent10k_v1"
)
WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_DATA_CONTRACT_SHA256 = (
    "bbb8d9b0a987e2aed17743df24024daa832d7b66d05ab45783a11a0f03a3f42c"
)
WEIGHTEDPDB_MSA_INDEPENDENT_TARGET_DATA_CONTRACT_SHA256 = (
    "ccf4fcfd799521fbc7cd0be1c6cc0eea744cf18fe87ab906c24bc485486dce00"
)
WEIGHTEDPDB_MSA_INDEPENDENT_MODEL_ARCHITECTURE_SHA256 = (
    "cd1cbb658e352f0d56a2dcd65c5e0e95ae2e29c321f4fd0da9f79c6145edc34c"
)
WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_STEP = 27799
WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_COMPLETED_UPDATES = 27800
WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_SCHEDULER_STEP = 27800
WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_NEXT_GLOBAL_STEP = 111200
WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_PAIR_ID = (
    "d40d64151cc8a8ff17458c310ca0a53f40047d3b5d4adf5577db3d653ae1c8f2"
)
WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_RAW_SHA256 = (
    "4cefb9af5088a4671b4287149c95efcd72cac37449136fe37b4e4d060754f94a"
)
WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_EMA_SHA256 = (
    "b3ad325bde97f2b249a03c4004a2a00d704e92a3aa0be925b0705e1a23ee67c8"
)
WEIGHTEDPDB_MSA_INDEPENDENT_DATA_PROFILE = (
    "weighted_pdb_legacy_residual_ge200_disordered_v5"
)
WEIGHTEDPDB_MSA_INDEPENDENT_DATASET = "weightedPDB_before210930_v20260101"
WEIGHTEDPDB_MSA_INDEPENDENT_PAIRED_RAW_DEPTH = 10000
WEIGHTEDPDB_MSA_INDEPENDENT_UNPAIRED_RAW_DEPTH = 10000
WEIGHTEDPDB_MSA_INDEPENDENT_FINAL_DEPTH = 16384
WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_SCHEDULER_STATE = {
    "warmup_steps": 1000,
    "decay_steps": 25000,
    "lr": 0.001,
    "decay_factor": 0.95,
    "base_lrs": [0.001],
    "last_epoch": WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_SCHEDULER_STEP,
    "_step_count": WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_SCHEDULER_STEP + 1,
    "_last_lr": [0.00095],
}
CATALOG_REPLACEMENT_FORMAT_V1 = "protenix.catalog_replacement.v1"
CATALOG_REPLACEMENT_ALLOWED_DELTA_V1 = (
    "legacy_short_long_to_legacy_residual_ge200_v1"
)
CATALOG_REPLACEMENT_SOURCE_TRAIN_SETS_V1 = (
    "weightedPDB_before210930_v20260101",
    "afdb_mgnify_short_distillation_v20260725",
    "afdb_mgnify_long_distillation_v20260725",
    "openfold3_disordered_v20260725",
)
CATALOG_REPLACEMENT_TARGET_TRAIN_SETS_V1 = (
    "weightedPDB_before210930_v20260101",
    "afdb_mgnify_legacy_residual_distillation_v20260804",
    "all_sources_ge200_structural_distillation_v20260804",
    "openfold3_disordered_v20260725",
)
CATALOG_REPLACEMENT_MASSES_V1 = (0.49, 0.005, 0.505, 0.02)
CATALOG_REPLACEMENT_SOURCE_SOFT_DATASETS_V1 = (
    CATALOG_REPLACEMENT_SOURCE_TRAIN_SETS_V1[1],
    CATALOG_REPLACEMENT_SOURCE_TRAIN_SETS_V1[2],
)
CATALOG_REPLACEMENT_TARGET_SOFT_DATASETS_V1 = (
    CATALOG_REPLACEMENT_TARGET_TRAIN_SETS_V1[1],
    CATALOG_REPLACEMENT_TARGET_TRAIN_SETS_V1[2],
)
MODEL_MSA_QUERY_INCLUSION_POLICY_VERSION = (
    "reserve_global_query_row_zero_when_sampled_depth_le_threshold_v1"
)
MODEL_MSA_QUERY_INCLUSION_DEPTH_ACCOUNTING = (
    "replace_one_slot_keep_exact_sampled_depth"
)
MODEL_MSA_QUERY_INCLUSION_ABOVE_THRESHOLD = (
    "legacy_random_without_replacement"
)
STRICT_RESUME_CONTRACT_MIGRATIONS = {
    STRICT_RESUME_MSA_QUERY_PIN_MIGRATION_V1: {
        "source_training_contract_id": (
            "crop384_db48_ddp_scratch_50k_ws128_acc4_data_"
            "weightedpdb_afdb_mgnify_short_long_disordered_v4_"
            "pdb490_short005_long505_dis020_"
            "afq_mgn_cluster_replen_le200_gt200_v1_"
            "proteinonlydiffusion_v1_resolvedproteincrop_minloss4_v2_"
            "allmsashuffle_multichain_seg50_whole50_gap24_min9_v2_"
            "partialquery_max2_keep1_v1_"
            "arch_pf64_cs384_dct768_pfckpt8_c48_"
            "msamix_q10_u30_p60_"
            "msaraw_p10000_u10000_distillu16384_final16384_"
            "templatecache_top60u20id90_chaindummy_ckpt200_"
            "dlpersist1_prefetch2_deferredhealth_v10"
        ),
        "source_data_contract_sha256": (
            "cffda1e6694a9f293c8ed213d1b33e188d828d10a4d0eebdf66b2f428e75756c"
        ),
        "source_model_architecture_sha256": (
            "cd1cbb658e352f0d56a2dcd65c5e0e95ae2e29c321f4fd0da9f79c6145edc34c"
        ),
        "source_run_id": (
            "protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_"
            "v9_straggler_templatecache_chaindummy_msabase10ku10k_"
            "distillu16384_ckpt200_20260726_061627"
        ),
        "target_training_contract_id": (
            "crop384_db48_ddp_scratch_50k_ws128_acc4_data_"
            "weightedpdb_afdb_mgnify_short_long_disordered_v4_"
            "pdb490_short005_long505_dis020_"
            "afq_mgn_cluster_replen_le200_gt200_v1_"
            "proteinonlydiffusion_v1_resolvedproteincrop_minloss4_v2_"
            "allmsashuffle_multichain_seg50_whole50_gap24_min9_v2_"
            "partialquery_max2_keep1_v1_"
            "arch_pf64_cs384_dct768_pfckpt8_c48_"
            "msamix_q10_u30_p60_"
            "msaraw_p10000_u10000_distillu16384_final16384_"
            "msaquerypinle1024_"
            "templatecache_top60u20id90_chaindummy_ckpt200_"
            "dlpersist1_prefetch2_deferredhealth_v11"
        ),
        "target_data_contract_sha256": (
            "6c98c5cab75197b7e4d41b7ce5287af65ca3f750a707ec69a1bdaf29ed24b8de"
        ),
        "allowed_delta": "model_msa_query_inclusion_train_test_le1024_v1",
    },
}


def strict_resume_contract_migration_mode() -> str:
    """Return the explicitly authorized strict-resume contract migration."""
    value = os.environ.get(
        "PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION",
        "",
    ).strip()
    if value not in {
        "",
        STRICT_RESUME_MSA_QUERY_PIN_MIGRATION_V1,
        STRICT_RESUME_SOFT_CLUSTER_SCHEDULE_MIGRATION_V1,
        STRICT_RESUME_SOFT_CLUSTER_CATALOG_REPLACEMENT_MIGRATION_V2,
        STRICT_RESUME_WEIGHTEDPDB_MSA_INDEPENDENT_MIGRATION_V1,
    }:
        raise ValueError(
            "unsupported PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION: "
            f"{value!r}"
        )
    return value


def _strict_sha256(value: Any, *, label: str) -> str:
    """Return one canonical SHA-256 string or fail closed."""
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256")
    return digest


def _sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash one file without materializing it in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_soft_cluster_schedule_migration_manifest() -> dict[str, Any]:
    """Load the schedule manifest through its shared fail-closed validator."""
    schedule_path = os.environ.get(SOFT_CLUSTER_SCHEDULE_DIR_ENV, "").strip()
    if not schedule_path:
        raise ValueError(
            "soft-cluster schedule migration requires "
            f"{SOFT_CLUSTER_SCHEDULE_DIR_ENV}"
        )
    try:
        from protenix.data.pipeline.soft_cluster_schedule import (
            build_soft_cluster_schedule_contract,
            load_soft_cluster_schedule_manifest,
        )
    except ImportError as exc:
        raise RuntimeError(
            "soft-cluster schedule migration validator is unavailable"
        ) from exc

    info = load_soft_cluster_schedule_manifest(schedule_path)
    manifest = getattr(info, "manifest", None)
    if not isinstance(manifest, Mapping):
        raise TypeError(
            "soft-cluster schedule manifest loader returned invalid metadata"
        )
    manifest_path = Path(info.manifest_path).resolve(strict=True)
    return {
        "manifest": dict(manifest),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _strict_sha256(
            info.manifest_sha256,
            label="soft-cluster schedule manifest SHA-256",
        ),
        "schedule_contract": build_soft_cluster_schedule_contract(info),
    }


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    """Hash one JSON-compatible mapping with the training-contract codec."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validated_catalog_dataset_binding(
    record: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], str]:
    """Return one exact catalog-migration dataset binding and its hash."""
    if not isinstance(record, Mapping) or set(record) != {
        "binding",
        "binding_sha256",
    }:
        raise ValueError(
            f"{label} must contain exactly binding and binding_sha256"
        )
    binding = record["binding"]
    if not isinstance(binding, Mapping):
        raise ValueError(f"{label}.binding must be an object")
    binding = json.loads(json.dumps(binding))
    declared_sha256 = _strict_sha256(
        record["binding_sha256"],
        label=f"{label}.binding_sha256",
    )
    actual_sha256 = _canonical_mapping_sha256(binding)
    if declared_sha256 != actual_sha256:
        raise ValueError(
            f"{label} binding SHA-256 mismatch: "
            f"declared={declared_sha256}, actual={actual_sha256}"
        )
    return binding, declared_sha256


def _load_bound_source_run_data_contract(
    manifest: Mapping[str, Any],
    *,
    source_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the source run contract only after checking its manifest binding."""
    source_run = manifest.get("source_run")
    if not isinstance(source_run, Mapping):
        raise ValueError(
            "catalog replacement requires a bound source_run record"
        )
    required = {
        "run_dir",
        "run_name",
        "run_manifest_path",
        "run_manifest_sha256",
    }
    missing = required - set(source_run)
    if missing:
        raise ValueError(
            "catalog replacement source_run is missing fields: "
            f"{sorted(missing)}"
        )
    run_dir = Path(str(source_run["run_dir"])).resolve(strict=True)
    run_manifest_path = Path(
        str(source_run["run_manifest_path"])
    ).resolve(strict=True)
    if not run_dir.is_dir() or not run_manifest_path.is_file():
        raise ValueError("catalog replacement source run artifacts are invalid")
    try:
        run_manifest_path.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(
            "catalog replacement run manifest must be inside source run"
        ) from exc
    if run_dir.name != source_checkpoint["source_run_id"]:
        raise ValueError(
            "catalog replacement source run directory does not match the "
            "checkpoint run id"
        )
    if source_run["run_name"] != source_checkpoint["source_run_id"]:
        raise ValueError(
            "catalog replacement source_run.run_name does not match the "
            "checkpoint run id"
        )
    expected_manifest_sha256 = _strict_sha256(
        source_run["run_manifest_sha256"],
        label="catalog replacement source run manifest SHA-256",
    )
    actual_manifest_sha256 = _sha256_file(run_manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "catalog replacement source run manifest SHA-256 mismatch"
        )
    with run_manifest_path.open("r", encoding="utf-8") as handle:
        run_manifest = json.load(handle)
    if not isinstance(run_manifest, Mapping):
        raise ValueError("source run manifest must contain one object")
    if run_manifest.get("run_name") != source_checkpoint["source_run_id"]:
        raise ValueError("source run manifest has the wrong run_name")
    source_contract = (
        run_manifest.get("data", {}).get("contract")
        if isinstance(run_manifest.get("data"), Mapping)
        else None
    )
    if not isinstance(source_contract, Mapping):
        raise ValueError("source run manifest is missing data.contract")
    source_contract = json.loads(json.dumps(source_contract))
    declared_contract_sha256 = _strict_sha256(
        source_contract.get("sha256"),
        label="source run data contract SHA-256",
    )
    contract_payload = dict(source_contract)
    contract_payload.pop("sha256")
    actual_contract_sha256 = _canonical_mapping_sha256(contract_payload)
    expected_contract_sha256 = _strict_sha256(
        source_checkpoint["source_data_contract_sha256"],
        label="catalog replacement source checkpoint data contract SHA-256",
    )
    if not (
        declared_contract_sha256
        == actual_contract_sha256
        == expected_contract_sha256
    ):
        raise ValueError(
            "catalog replacement source run/checkpoint data contracts differ"
        )
    return source_contract


def _drop_catalog_dataset_entries(
    value: dict[str, Any],
    *,
    catalog_dataset_names: Sequence[str],
) -> None:
    """Remove only per-catalog entries from a copied data contract."""
    catalog_names = set(catalog_dataset_names)
    augmentation = value.get("augmentation")
    if isinstance(augmentation, dict):
        effective = augmentation.get("effective_by_dataset")
        if isinstance(effective, dict):
            for name in catalog_names:
                effective.pop(name, None)
    for section_name in (
        "dataset_runtime_semantics",
        "loss_and_crop_policy",
    ):
        section = value.get(section_name)
        if not isinstance(section, dict):
            continue
        effective = section.get("effective_by_dataset")
        if isinstance(effective, dict):
            for name in catalog_names:
                effective.pop(name, None)
    runtime_indices = value.get("runtime_distillation_indices")
    if isinstance(runtime_indices, dict):
        for name in catalog_names:
            runtime_indices.pop(name, None)


def _catalog_invariant_projection(
    contract: Mapping[str, Any],
    *,
    catalog_dataset_names: Sequence[str],
) -> dict[str, Any]:
    """Project a contract onto fields that catalog replacement cannot alter."""
    projected = json.loads(json.dumps(contract))
    projected.pop("sha256", None)
    projected.pop("profile_id", None)
    projected.pop("train_sets", None)
    projected.pop("soft_cluster_schedule", None)
    _drop_catalog_dataset_entries(
        projected,
        catalog_dataset_names=catalog_dataset_names,
    )
    return projected


def _scrub_catalog_transport_fields(value: Any) -> Any:
    """Remove catalog identity/path fields while retaining scientific policy."""
    transport_keys = {
        "dataset_name",
        "indices_fpath",
        "runtime_index_dir",
        "source_roots_fpath",
        "mmcif_dir",
        "bioassembly_dict_dir",
        "pdb_list",
    }
    if isinstance(value, Mapping):
        return {
            str(key): _scrub_catalog_transport_fields(item)
            for key, item in value.items()
            if key not in transport_keys
        }
    if isinstance(value, list):
        return [_scrub_catalog_transport_fields(item) for item in value]
    return value


def _catalog_dataset_semantics(
    contract: Mapping[str, Any],
    *,
    dataset_names: Sequence[str],
) -> dict[str, Any]:
    """Require every replaced child to retain one common non-path policy."""
    result: dict[str, Any] = {}
    section_paths = {
        "augmentation": ("augmentation", "effective_by_dataset"),
        "runtime": ("dataset_runtime_semantics", "effective_by_dataset"),
        "loss_and_crop": ("loss_and_crop_policy", "effective_by_dataset"),
    }
    for output_name, path in section_paths.items():
        current: Any = contract
        for component in path:
            current = current.get(component) if isinstance(current, Mapping) else None
        if not isinstance(current, Mapping):
            raise ValueError(
                f"catalog replacement data contract lacks {'.'.join(path)}"
            )
        values = []
        for dataset_name in dataset_names:
            if dataset_name not in current:
                raise ValueError(
                    "catalog replacement data contract lacks semantics for "
                    f"{dataset_name}"
                )
            values.append(
                _scrub_catalog_transport_fields(current[dataset_name])
            )
        reference = values[0]
        if any(value != reference for value in values[1:]):
            raise ValueError(
                "catalog replacement children do not share identical "
                f"{output_name} policy"
            )
        result[output_name] = reference
    return result


def _validate_catalog_replacement_transition(
    *,
    loaded: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    target_data_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Authorize exactly the manifest-declared Short/Long catalog replacement."""
    manifest = loaded["manifest"]
    catalog = manifest.get("catalog_migration")
    if not isinstance(catalog, Mapping):
        raise ValueError(
            "soft-cluster catalog migration requires catalog_migration"
        )
    if set(catalog) != {
        "format",
        "allowed_delta",
        "source_dataset",
        "target_dataset",
        "crosswalk",
    }:
        raise ValueError("catalog_migration contains unsupported fields")
    if catalog["format"] != CATALOG_REPLACEMENT_FORMAT_V1:
        raise ValueError(
            f"unsupported catalog replacement format: {catalog['format']!r}"
        )
    if catalog["allowed_delta"] != CATALOG_REPLACEMENT_ALLOWED_DELTA_V1:
        raise ValueError(
            f"unsupported catalog replacement delta: {catalog['allowed_delta']!r}"
        )
    source_binding, source_binding_sha256 = (
        _validated_catalog_dataset_binding(
            catalog["source_dataset"],
            label="catalog_migration.source_dataset",
        )
    )
    target_binding, target_binding_sha256 = (
        _validated_catalog_dataset_binding(
            catalog["target_dataset"],
            label="catalog_migration.target_dataset",
        )
    )
    if target_binding != manifest.get("dataset"):
        raise ValueError(
            "catalog replacement target binding must equal manifest.dataset"
        )
    if target_binding_sha256 != loaded["schedule_contract"].get(
        "dataset_binding_sha256"
    ):
        raise ValueError(
            "catalog replacement target binding SHA does not match schedule"
        )
    expected_source_sets = list(
        CATALOG_REPLACEMENT_SOURCE_TRAIN_SETS_V1
    )
    expected_target_sets = list(
        CATALOG_REPLACEMENT_TARGET_TRAIN_SETS_V1
    )
    expected_masses = list(CATALOG_REPLACEMENT_MASSES_V1)
    if source_binding.get("train_sets") != expected_source_sets:
        raise ValueError("catalog replacement source train_sets mismatch")
    if target_binding.get("train_sets") != expected_target_sets:
        raise ValueError("catalog replacement target train_sets mismatch")
    if source_binding.get("dataset_masses") != expected_masses:
        raise ValueError("catalog replacement source dataset masses mismatch")
    if target_binding.get("dataset_masses") != expected_masses:
        raise ValueError("catalog replacement target dataset masses mismatch")
    for unchanged_index in (0, 3):
        source_child = source_binding["child_fingerprints"][unchanged_index]
        target_child = target_binding["child_fingerprints"][unchanged_index]
        if source_child != target_child:
            raise ValueError(
                "catalog replacement changed an unchanged PDB/disordered child"
            )

    crosswalk = catalog["crosswalk"]
    if not isinstance(crosswalk, Mapping):
        raise ValueError("catalog_migration.crosswalk must be an object")
    required_crosswalk = {
        "format": "protenix.global_id_history_projection.v1",
        "global_id_policy": "source_prefix_pipe_raw_id_v1",
        "exposure_policy": "occurrence_sum_v1",
        "ambiguous_alternate_policy": "seen_floor_1_v1",
        "source_soft_datasets": list(
            CATALOG_REPLACEMENT_SOURCE_SOFT_DATASETS_V1
        ),
        "target_soft_datasets": list(
            CATALOG_REPLACEMENT_TARGET_SOFT_DATASETS_V1
        ),
    }
    for field, expected_value in required_crosswalk.items():
        if crosswalk.get(field) != expected_value:
            raise ValueError(
                "catalog replacement crosswalk mismatch for "
                f"{field}: {crosswalk.get(field)!r}"
            )
    lookup_bindings = crosswalk.get("lookup_bindings")
    if not isinstance(lookup_bindings, Mapping) or set(lookup_bindings) != set(
        CATALOG_REPLACEMENT_TARGET_SOFT_DATASETS_V1
    ):
        raise ValueError(
            "catalog replacement crosswalk must bind both target runtime "
            "indices"
        )
    runtime_contracts = target_data_contract.get(
        "runtime_distillation_indices"
    )
    if not isinstance(runtime_contracts, Mapping):
        raise ValueError(
            "target data contract lacks runtime distillation identities"
        )
    for dataset_name in CATALOG_REPLACEMENT_TARGET_SOFT_DATASETS_V1:
        lookup_binding = lookup_bindings[dataset_name]
        runtime_contract = runtime_contracts.get(dataset_name)
        if not isinstance(lookup_binding, Mapping) or not isinstance(
            runtime_contract,
            Mapping,
        ):
            raise ValueError(
                f"catalog replacement has no runtime binding for {dataset_name}"
            )
        live_manifest_path = os.path.realpath(
            os.path.join(
                str(runtime_contract.get("runtime_index_dir", "")),
                "manifest.json",
            )
        )
        if live_manifest_path != lookup_binding.get("runtime_manifest_path"):
            raise ValueError(
                "catalog replacement live runtime manifest path differs from "
                f"crosswalk binding for {dataset_name}"
            )
        if runtime_contract.get("manifest_sha256") != lookup_binding.get(
            "runtime_manifest_sha256"
        ):
            raise ValueError(
                "catalog replacement live runtime manifest SHA differs from "
                f"crosswalk binding for {dataset_name}"
            )
    if manifest.get("algorithm", {}).get("soft_cluster_datasets") != list(
        CATALOG_REPLACEMENT_TARGET_SOFT_DATASETS_V1
    ):
        raise ValueError(
            "catalog replacement schedule has the wrong soft datasets"
        )

    source_data_contract = _load_bound_source_run_data_contract(
        manifest,
        source_checkpoint=source_checkpoint,
    )
    if source_data_contract.get("train_sets") != expected_source_sets:
        raise ValueError("source run data contract train_sets mismatch")
    if source_data_contract.get("train_sample_weights") != expected_masses:
        raise ValueError("source run data contract masses mismatch")
    if target_data_contract.get("train_sets") != expected_target_sets:
        raise ValueError("target live data contract train_sets mismatch")
    if target_data_contract.get("train_sample_weights") != expected_masses:
        raise ValueError("target live data contract masses mismatch")

    source_projection = _catalog_invariant_projection(
        source_data_contract,
        catalog_dataset_names=CATALOG_REPLACEMENT_SOURCE_SOFT_DATASETS_V1,
    )
    target_projection = _catalog_invariant_projection(
        target_data_contract,
        catalog_dataset_names=CATALOG_REPLACEMENT_TARGET_SOFT_DATASETS_V1,
    )
    if source_projection != target_projection:
        raise ValueError(
            "catalog replacement has an unauthorized non-catalog "
            "training-data policy delta"
        )
    source_semantics = _catalog_dataset_semantics(
        source_data_contract,
        dataset_names=CATALOG_REPLACEMENT_SOURCE_SOFT_DATASETS_V1,
    )
    target_semantics = _catalog_dataset_semantics(
        target_data_contract,
        dataset_names=CATALOG_REPLACEMENT_TARGET_SOFT_DATASETS_V1,
    )
    if source_semantics != target_semantics:
        raise ValueError(
            "catalog replacement changed distillation MSA/crop/loss policy"
        )
    return {
        "catalog_migration": json.loads(json.dumps(catalog)),
        "source_dataset_binding": source_binding,
        "source_dataset_binding_sha256": source_binding_sha256,
        "target_dataset_binding": target_binding,
        "target_dataset_binding_sha256": target_binding_sha256,
        "source_data_contract": source_data_contract,
        "source_augmentation": source_data_contract["augmentation"],
        "catalog_invariant_projection_sha256": (
            _canonical_mapping_sha256(source_projection)
        ),
        "catalog_dataset_semantics_sha256": (
            _canonical_mapping_sha256(source_semantics)
        ),
    }


def _soft_cluster_schedule_migration_expectations(
    configs: Any,
) -> dict[str, Any]:
    """Resolve a manifest-authorized legacy-to-schedule sampler migration."""
    migration_mode = strict_resume_contract_migration_mode()
    if migration_mode not in SOFT_CLUSTER_MIGRATION_MODES:
        raise ValueError(
            "soft-cluster migration resolver requires a supported "
            "soft-cluster migration mode"
        )
    loaded = _load_soft_cluster_schedule_migration_manifest()
    data_cfg = _get_value(configs, "data", {})
    sampler_cfg = _get_value(data_cfg, "train_sampler", {})
    configured_schedule_path = str(
        _get_value(
            sampler_cfg,
            "soft_cluster_schedule_manifest",
            "",
        )
        or ""
    ).strip()
    if not configured_schedule_path:
        raise ValueError(
            "soft-cluster migration requires "
            "data.train_sampler.soft_cluster_schedule_manifest"
        )
    configured_manifest_path = Path(configured_schedule_path)
    if configured_manifest_path.is_dir():
        configured_manifest_path = configured_manifest_path / "manifest.json"
    configured_manifest_path = configured_manifest_path.resolve(strict=False)
    if str(configured_manifest_path) != loaded["manifest_path"]:
        raise ValueError(
            "soft-cluster migration environment/config manifest mismatch: "
            f"environment={loaded['manifest_path']!r}, "
            f"config={str(configured_manifest_path)!r}"
        )
    manifest = loaded["manifest"]
    source = manifest.get("source_checkpoint")
    if not isinstance(source, Mapping):
        raise ValueError(
            "soft-cluster schedule manifest is missing source_checkpoint"
        )
    if source.get("mode") != "checkpoint_migration":
        raise ValueError(
            "strict soft-cluster resume requires "
            "source_checkpoint.mode='checkpoint_migration'"
        )

    required_text = {
        "source_run_id",
        "source_training_contract_id",
    }
    for field in sorted(required_text):
        if not isinstance(source.get(field), str) or not source[field].strip():
            raise ValueError(
                "soft-cluster schedule source_checkpoint requires non-empty "
                f"{field}"
            )
    required_hashes = {
        "checkpoint_pair_id",
        "raw_checkpoint_sha256",
        "ema_checkpoint_sha256",
        "source_data_contract_sha256",
        "model_architecture_sha256",
    }
    hashes = {
        field: _strict_sha256(
            source.get(field),
            label=f"soft-cluster source_checkpoint.{field}",
        )
        for field in sorted(required_hashes)
    }
    try:
        checkpoint_step = int(source["checkpoint_step"])
        next_global_step = int(source["next_global_step"])
        source_accumulation = int(source["iters_to_accumulate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "soft-cluster source checkpoint requires integer checkpoint_step, "
            "next_global_step, and iters_to_accumulate"
        ) from exc
    if checkpoint_step < 0 or source_accumulation <= 0:
        raise ValueError(
            "soft-cluster source checkpoint step must be non-negative and "
            "iters_to_accumulate must be positive"
        )
    expected_next_global_step = (
        checkpoint_step + 1
    ) * source_accumulation
    if next_global_step != expected_next_global_step:
        raise ValueError(
            "soft-cluster source checkpoint boundary mismatch: "
            f"next_global_step={next_global_step}, "
            f"expected={expected_next_global_step}"
        )
    current_accumulation = int(_get_value(configs, "iters_to_accumulate", 1))
    if source_accumulation != current_accumulation:
        raise ValueError(
            "soft-cluster source/target accumulation mismatch: "
            f"source={source_accumulation}, target={current_accumulation}"
        )

    expected_step_text = os.environ.get(
        "PROTENIX_RESUME_CHECKPOINT_STEP", ""
    ).strip()
    if not expected_step_text:
        raise ValueError(
            "soft-cluster schedule migration requires "
            "PROTENIX_RESUME_CHECKPOINT_STEP"
        )
    try:
        expected_step = int(expected_step_text)
    except ValueError as exc:
        raise ValueError(
            "PROTENIX_RESUME_CHECKPOINT_STEP must be an integer, got "
            f"{expected_step_text!r}"
        ) from exc
    if expected_step != checkpoint_step:
        raise ValueError(
            "soft-cluster manifest/checkpoint boundary mismatch: "
            f"manifest step={checkpoint_step}, launcher step={expected_step}"
        )

    checkpoint_paths = {
        "raw_checkpoint_path": str(
            _get_value(configs, "load_checkpoint_path", "")
        ),
        "ema_checkpoint_path": str(
            _get_value(configs, "load_ema_checkpoint_path", "")
        ),
    }
    for field, configured_path in checkpoint_paths.items():
        configured_realpath = os.path.realpath(configured_path)
        if configured_realpath != source.get(field):
            raise ValueError(
                f"soft-cluster migration source {field} mismatch: "
                f"configured={configured_realpath!r}, "
                f"manifest={source.get(field)!r}"
            )

    target_training_contract_id = os.environ.get(
        "PROTENIX_TRAINING_CONTRACT_ID", ""
    ).strip()
    if not target_training_contract_id:
        raise ValueError(
            "soft-cluster migration requires a non-empty target training "
            "contract"
        )
    if target_training_contract_id == source["source_training_contract_id"]:
        raise ValueError(
            "soft-cluster migration target training contract must differ "
            "from the source contract"
        )

    target_data_contract = build_training_data_contract(configs)
    target_data_contract_sha256 = _strict_sha256(
        target_data_contract.get("sha256"),
        label="soft-cluster target data contract SHA-256",
    )

    if (
        migration_mode
        == STRICT_RESUME_SOFT_CLUSTER_CATALOG_REPLACEMENT_MIGRATION_V2
    ):
        if int(loaded["manifest"].get("schema_version", -1)) != 2:
            raise ValueError(
                "catalog replacement migration requires schedule "
                "schema_version=2"
            )
        catalog_transition = _validate_catalog_replacement_transition(
            loaded=loaded,
            source_checkpoint=source,
            target_data_contract=target_data_contract,
        )
        target_architecture = build_model_architecture_contract(configs)
        if target_architecture.get("sha256") != hashes[
            "model_architecture_sha256"
        ]:
            raise ValueError(
                "catalog replacement requires identical source and target "
                "model architecture"
            )
        return {
            "mode": migration_mode,
            "allowed_delta": CATALOG_REPLACEMENT_ALLOWED_DELTA_V1,
            "source_run_id": source["source_run_id"],
            "source_training_contract_id": source[
                "source_training_contract_id"
            ],
            "source_data_contract_sha256": hashes[
                "source_data_contract_sha256"
            ],
            "source_model_architecture_sha256": hashes[
                "model_architecture_sha256"
            ],
            "source_checkpoint_pair_id": hashes["checkpoint_pair_id"],
            "source_checkpoint_step": checkpoint_step,
            "source_next_global_step": next_global_step,
            "source_raw_checkpoint_sha256": hashes[
                "raw_checkpoint_sha256"
            ],
            "source_ema_checkpoint_sha256": hashes[
                "ema_checkpoint_sha256"
            ],
            "target_training_contract_id": target_training_contract_id,
            "target_data_contract_sha256": target_data_contract_sha256,
            "target_data_contract": target_data_contract,
            "target_model_architecture_sha256": target_architecture[
                "sha256"
            ],
            "soft_cluster_schedule_contract": dict(
                loaded["schedule_contract"]
            ),
            "soft_cluster_schedule_manifest_path": loaded[
                "manifest_path"
            ],
            "soft_cluster_schedule_manifest_sha256": loaded[
                "manifest_sha256"
            ],
            "trajectory_note": (
                "full model/EMA/optimizer/scheduler/scaler/step and all "
                "128 rank RNG states are restored at the manifest-bound "
                "boundary; only the declared legacy Short/Long to legacy "
                "residual plus ge200 structural catalog replacement and "
                "future soft-cluster schedule change"
            ),
            **catalog_transition,
        }

    if int(loaded["manifest"].get("schema_version", 1)) != 1:
        raise ValueError(
            "soft-cluster schedule-only migration v1 requires schedule "
            "schema_version=1"
        )

    # This migration authorizes exactly one new immutable schedule contract.
    # Deleting that field must reconstruct the source contract byte-for-byte;
    # a signed manifest cannot be used to smuggle in another data-policy delta.
    source_payload = json.loads(json.dumps(target_data_contract))
    source_payload.pop("sha256", None)
    schedule_contract = source_payload.pop("soft_cluster_schedule", None)
    if not isinstance(schedule_contract, Mapping):
        raise ValueError(
            "soft-cluster target data contract is missing "
            "soft_cluster_schedule"
        )
    expected_schedule_contract = loaded.get("schedule_contract")
    if (
        not isinstance(expected_schedule_contract, Mapping)
        or dict(schedule_contract) != dict(expected_schedule_contract)
    ):
        raise ValueError(
            "soft-cluster migration target data contract does not match the "
            "validated manifest scientific subset"
        )
    source_canonical = json.dumps(
        source_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    reconstructed_source_sha256 = hashlib.sha256(
        source_canonical
    ).hexdigest()
    if reconstructed_source_sha256 != hashes[
        "source_data_contract_sha256"
    ]:
        raise ValueError(
            "soft-cluster migration has an unauthorized data-contract delta: "
            "removing only soft_cluster_schedule reconstructs "
            f"{reconstructed_source_sha256}, expected "
            f"{hashes['source_data_contract_sha256']}"
        )

    target_architecture = build_model_architecture_contract(configs)
    if target_architecture.get("sha256") != hashes[
        "model_architecture_sha256"
    ]:
        raise ValueError(
            "soft-cluster migration requires identical source and target "
            "model architecture"
        )

    return {
        "mode": migration_mode,
        "allowed_delta": "soft_cluster_schedule_only_v1",
        "source_run_id": source["source_run_id"],
        "source_training_contract_id": source[
            "source_training_contract_id"
        ],
        "source_data_contract_sha256": hashes[
            "source_data_contract_sha256"
        ],
        "source_model_architecture_sha256": hashes[
            "model_architecture_sha256"
        ],
        "source_checkpoint_pair_id": hashes["checkpoint_pair_id"],
        "source_checkpoint_step": checkpoint_step,
        "source_next_global_step": next_global_step,
        "source_raw_checkpoint_sha256": hashes[
            "raw_checkpoint_sha256"
        ],
        "source_ema_checkpoint_sha256": hashes[
            "ema_checkpoint_sha256"
        ],
        "target_training_contract_id": target_training_contract_id,
        "target_data_contract_sha256": target_data_contract_sha256,
        "target_data_contract": target_data_contract,
        "target_model_architecture_sha256": target_architecture["sha256"],
        "source_augmentation": target_data_contract["augmentation"],
        "reconstructed_source_data_contract_sha256": (
            reconstructed_source_sha256
        ),
        "soft_cluster_schedule_contract": dict(schedule_contract),
        "soft_cluster_schedule_manifest_path": loaded["manifest_path"],
        "soft_cluster_schedule_manifest_sha256": loaded[
            "manifest_sha256"
        ],
        "trajectory_note": (
            "full model/EMA/optimizer/scheduler/scaler/step/RNG state is "
            "restored at the manifest-bound boundary; only the future "
            "cluster schedule changes"
        ),
    }


def _reconstruct_step27799_source_msa_policy(
    target_data_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove exactly the authorized weightedPDB source-budget delta."""

    source_payload = json.loads(json.dumps(target_data_contract))
    source_payload.pop("sha256", None)
    runtime_semantics = source_payload.get("dataset_runtime_semantics")
    effective_by_dataset = (
        runtime_semantics.get("effective_by_dataset")
        if isinstance(runtime_semantics, Mapping)
        else None
    )
    if not isinstance(effective_by_dataset, Mapping):
        raise ValueError(
            "weightedPDB MSA-only migration requires schema-7 dataset "
            "runtime semantics"
        )
    if set(effective_by_dataset) != set(source_payload.get("train_sets", [])):
        raise ValueError(
            "weightedPDB MSA-only migration effective dataset set does not "
            "match train_sets"
        )

    weighted_pdb = effective_by_dataset.get(
        WEIGHTEDPDB_MSA_INDEPENDENT_DATASET
    )
    if not isinstance(weighted_pdb, Mapping):
        raise ValueError(
            "weightedPDB MSA-only migration is missing weightedPDB runtime "
            "semantics"
        )
    effective_msa = weighted_pdb.get("effective_msa")
    if not isinstance(effective_msa, dict):
        raise ValueError(
            "weightedPDB MSA-only migration is missing weightedPDB effective "
            "MSA settings"
        )
    if effective_msa.get("independent_pairing_non_pairing_budget") is not True:
        raise ValueError(
            "weightedPDB MSA-only migration requires "
            "independent_pairing_non_pairing_budget=true for weightedPDB"
        )

    expected_values = {
        "prot_pairing_dbs": ["pairing"],
        "prot_non_pairing_dbs": ["pairing-non_pairing"],
        "max_input_paired_msa_depth": (
            WEIGHTEDPDB_MSA_INDEPENDENT_PAIRED_RAW_DEPTH
        ),
        "max_input_unpaired_msa_depth": (
            WEIGHTEDPDB_MSA_INDEPENDENT_UNPAIRED_RAW_DEPTH
        ),
    }
    for field, expected in expected_values.items():
        if effective_msa.get(field) != expected:
            raise ValueError(
                "weightedPDB MSA-only policy mismatch for "
                f"{field}: current={effective_msa.get(field)!r}, "
                f"required={expected!r}"
            )
    max_size = effective_msa.get("max_size")
    if (
        not isinstance(max_size, Mapping)
        or int(max_size.get("train", -1))
        != WEIGHTEDPDB_MSA_INDEPENDENT_FINAL_DEPTH
    ):
        raise ValueError(
            "weightedPDB MSA-only policy requires max_size.train="
            f"{WEIGHTEDPDB_MSA_INDEPENDENT_FINAL_DEPTH}"
        )

    enabled_datasets = []
    for dataset_name, dataset_semantics in effective_by_dataset.items():
        if not isinstance(dataset_semantics, Mapping):
            raise ValueError(
                f"dataset runtime semantics for {dataset_name} must be an object"
            )
        dataset_effective_msa = dataset_semantics.get("effective_msa")
        if not isinstance(dataset_effective_msa, dict):
            raise ValueError(
                f"dataset runtime semantics for {dataset_name} lacks "
                "effective_msa"
            )
        removed = dataset_effective_msa.pop(
            "independent_pairing_non_pairing_budget",
            False,
        )
        if not isinstance(removed, bool):
            raise ValueError(
                "independent_pairing_non_pairing_budget must be boolean in "
                f"dataset {dataset_name}, got {removed!r}"
            )
        if removed is True:
            enabled_datasets.append(str(dataset_name))
    if enabled_datasets != [WEIGHTEDPDB_MSA_INDEPENDENT_DATASET]:
        raise ValueError(
            "independent pairing/non-pairing budget must be active only for "
            f"weightedPDB, got {enabled_datasets}"
        )

    policy = {
        "policy_version": PAIRING_NON_PAIRING_BUDGET_POLICY_VERSION,
        "dataset": WEIGHTEDPDB_MSA_INDEPENDENT_DATASET,
        "paired_source": "pairing.a3m",
        "genuine_non_pairing_source": "non_pairing.a3m",
        "paired_source_raw_depth": (
            WEIGHTEDPDB_MSA_INDEPENDENT_PAIRED_RAW_DEPTH
        ),
        "genuine_non_pairing_raw_depth": (
            WEIGHTEDPDB_MSA_INDEPENDENT_UNPAIRED_RAW_DEPTH
        ),
        "final_assembled_depth": WEIGHTEDPDB_MSA_INDEPENDENT_FINAL_DEPTH,
        "paired_block_deduplicate": False,
        "unpaired_merge_deduplicate": True,
        "inference_policy_changed": False,
    }
    return source_payload, policy


def _weightedpdb_msa_independent_migration_expectations(
    configs: Any,
) -> dict[str, Any]:
    """Resolve the checkpoint-bound step-27799 MSA-only migration."""

    target_training_contract_id = os.environ.get(
        "PROTENIX_TRAINING_CONTRACT_ID",
        "",
    ).strip()
    if (
        target_training_contract_id
        != WEIGHTEDPDB_MSA_INDEPENDENT_TARGET_TRAINING_CONTRACT_ID
    ):
        raise ValueError(
            "weightedPDB MSA-only target training contract mismatch: "
            f"current={target_training_contract_id!r}, required="
            f"{WEIGHTEDPDB_MSA_INDEPENDENT_TARGET_TRAINING_CONTRACT_ID!r}"
        )

    target_data_contract = build_training_data_contract(configs)
    target_data_contract_sha256 = target_data_contract.get("sha256")
    if (
        target_data_contract_sha256
        != WEIGHTEDPDB_MSA_INDEPENDENT_TARGET_DATA_CONTRACT_SHA256
    ):
        raise ValueError(
            "weightedPDB MSA-only target data contract mismatch: "
            f"current={target_data_contract_sha256}, required="
            f"{WEIGHTEDPDB_MSA_INDEPENDENT_TARGET_DATA_CONTRACT_SHA256}"
        )
    if (
        target_data_contract.get("profile_id")
        != WEIGHTEDPDB_MSA_INDEPENDENT_DATA_PROFILE
    ):
        raise ValueError(
            "weightedPDB MSA-only migration requires the frozen data profile "
            f"{WEIGHTEDPDB_MSA_INDEPENDENT_DATA_PROFILE!r}"
        )

    source_payload, weighted_pdb_msa_policy = (
        _reconstruct_step27799_source_msa_policy(target_data_contract)
    )
    reconstructed_source_sha256 = _canonical_mapping_sha256(source_payload)
    if (
        reconstructed_source_sha256
        != WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_DATA_CONTRACT_SHA256
    ):
        raise ValueError(
            "weightedPDB MSA-only migration has an unauthorized data-contract "
            "delta: removing only the independent source-budget flag "
            f"reconstructs {reconstructed_source_sha256}, expected "
            f"{WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_DATA_CONTRACT_SHA256}"
        )

    target_architecture = build_model_architecture_contract(configs)
    if (
        target_architecture.get("sha256")
        != WEIGHTEDPDB_MSA_INDEPENDENT_MODEL_ARCHITECTURE_SHA256
    ):
        raise ValueError(
            "weightedPDB MSA-only migration requires identical source and "
            "target model architecture"
        )

    return {
        "mode": STRICT_RESUME_WEIGHTEDPDB_MSA_INDEPENDENT_MIGRATION_V1,
        "allowed_delta": (
            "weightedpdb_independent_pairing_nonpairing_top10k_only_v1"
        ),
        "source_run_id": WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_RUN_ID,
        "source_training_contract_id": (
            WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_TRAINING_CONTRACT_ID
        ),
        "target_training_contract_id": (
            WEIGHTEDPDB_MSA_INDEPENDENT_TARGET_TRAINING_CONTRACT_ID
        ),
        "source_data_contract_sha256": (
            WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_DATA_CONTRACT_SHA256
        ),
        "target_data_contract_sha256": (
            WEIGHTEDPDB_MSA_INDEPENDENT_TARGET_DATA_CONTRACT_SHA256
        ),
        "target_data_contract": target_data_contract,
        "source_augmentation": source_payload["augmentation"],
        "source_model_architecture_sha256": (
            WEIGHTEDPDB_MSA_INDEPENDENT_MODEL_ARCHITECTURE_SHA256
        ),
        "target_model_architecture_sha256": (
            WEIGHTEDPDB_MSA_INDEPENDENT_MODEL_ARCHITECTURE_SHA256
        ),
        "source_checkpoint_step": WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_STEP,
        "source_completed_updates": (
            WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_COMPLETED_UPDATES
        ),
        "source_checkpoint_pair_id": (
            WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_PAIR_ID
        ),
        "source_raw_checkpoint_sha256": (
            WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_RAW_SHA256
        ),
        "source_ema_checkpoint_sha256": (
            WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_EMA_SHA256
        ),
        "source_next_global_step": (
            WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_NEXT_GLOBAL_STEP
        ),
        "source_scheduler_step": (
            WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_SCHEDULER_STEP
        ),
        "source_scheduler_state": dict(
            WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_SCHEDULER_STATE
        ),
        "weightedpdb_msa_policy": weighted_pdb_msa_policy,
        "reconstructed_source_data_contract_sha256": (
            reconstructed_source_sha256
        ),
        "trajectory_note": (
            "model/EMA/optimizer/scheduler/step/scaler/RNG/sampler cursor and "
            "the existing step-21999 soft-cluster schedule are restored "
            "unchanged; only weightedPDB MSA source-budget accounting changes"
        ),
    }


def _strict_resume_migration_expectations(
    configs: Any,
) -> dict[str, Any] | None:
    """Validate and resolve one fail-closed source-to-target migration."""
    mode = strict_resume_contract_migration_mode()
    if not mode:
        return None

    if mode in SOFT_CLUSTER_MIGRATION_MODES:
        return _soft_cluster_schedule_migration_expectations(configs)
    if mode == STRICT_RESUME_WEIGHTEDPDB_MSA_INDEPENDENT_MIGRATION_V1:
        return _weightedpdb_msa_independent_migration_expectations(configs)

    migration = dict(STRICT_RESUME_CONTRACT_MIGRATIONS[mode])
    target_training_contract_id = os.environ.get(
        "PROTENIX_TRAINING_CONTRACT_ID",
        "",
    ).strip()
    if (
        target_training_contract_id
        != migration["target_training_contract_id"]
    ):
        raise ValueError(
            "MSA query-pin migration target training contract mismatch: "
            f"current={target_training_contract_id!r}, "
            f"required={migration['target_training_contract_id']!r}"
        )

    target_data_contract = build_training_data_contract(configs)
    expected_query_policy = {
        "policy_version": MODEL_MSA_QUERY_INCLUSION_POLICY_VERSION,
        "query_row_index": 0,
        "max_sample_size": {
            "train": 1024,
            "test": 1024,
        },
        "depth_accounting": MODEL_MSA_QUERY_INCLUSION_DEPTH_ACCOUNTING,
        "above_threshold": MODEL_MSA_QUERY_INCLUSION_ABOVE_THRESHOLD,
    }
    observed_query_policy = target_data_contract["augmentation"].get(
        "model_msa_query_inclusion"
    )
    if observed_query_policy != expected_query_policy:
        raise ValueError(
            "MSA query-pin migration target must use the exact train/test "
            "1024 query-inclusion policy"
        )
    if (
        target_data_contract["sha256"]
        != migration["target_data_contract_sha256"]
    ):
        raise ValueError(
            "MSA query-pin migration target data contract mismatch: "
            f"current={target_data_contract['sha256']}, "
            f"required={migration['target_data_contract_sha256']}"
        )

    target_architecture = build_model_architecture_contract(configs)
    if (
        migration["source_model_architecture_sha256"]
        != target_architecture["sha256"]
    ):
        raise ValueError(
            "strict-resume migration requires identical source and target "
            "model architecture"
        )

    # Reconstruct the exact legacy v10 payload by removing only the authorized
    # query-pin fields.  Matching its frozen SHA proves there is no hidden
    # data-policy delta in this migration.
    source_payload = json.loads(json.dumps(target_data_contract))
    source_payload.pop("sha256", None)
    source_augmentation = source_payload["augmentation"]
    removed_policy = source_augmentation.pop(
        "model_msa_query_inclusion",
        None,
    )
    if removed_policy != expected_query_policy:
        raise ValueError(
            "failed to derive the legacy source augmentation contract"
        )
    effective_by_dataset = source_payload.get(
        "dataset_runtime_semantics",
        {},
    ).get("effective_by_dataset")
    if not isinstance(effective_by_dataset, Mapping):
        raise ValueError(
            "MSA query-pin migration target lacks effective dataset MSA "
            "semantics"
        )
    if set(effective_by_dataset) != set(target_data_contract["train_sets"]):
        raise ValueError(
            "MSA query-pin migration effective dataset set does not match "
            "the target train_sets"
        )
    expected_thresholds = {"train": 1024, "test": 1024}
    for dataset_name, runtime_semantics in effective_by_dataset.items():
        effective_msa = runtime_semantics.get("effective_msa")
        if not isinstance(effective_msa, dict):
            raise ValueError(
                "MSA query-pin migration target lacks effective MSA config "
                f"for dataset {dataset_name}"
            )
        removed_thresholds = effective_msa.pop(
            "query_inclusion_max_sample_size",
            None,
        )
        if removed_thresholds != expected_thresholds:
            raise ValueError(
                "MSA query-pin migration expected train/test 1024 in every "
                f"effective dataset MSA config; {dataset_name} has "
                f"{removed_thresholds!r}"
            )
    source_canonical = json.dumps(
        source_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    reconstructed_source_sha256 = hashlib.sha256(
        source_canonical
    ).hexdigest()
    if (
        reconstructed_source_sha256
        != migration["source_data_contract_sha256"]
    ):
        raise ValueError(
            "MSA query-pin migration has an unauthorized data-contract "
            "delta: removing only query-pin reconstructs "
            f"{reconstructed_source_sha256}, expected "
            f"{migration['source_data_contract_sha256']}"
        )
    return {
        "mode": mode,
        **migration,
        "target_data_contract": target_data_contract,
        "target_model_architecture_sha256": target_architecture["sha256"],
        "expected_query_policy": expected_query_policy,
        "source_augmentation": source_augmentation,
        "reconstructed_source_data_contract_sha256": (
            reconstructed_source_sha256
        ),
    }


def validate_soft_cluster_migration_checkpoint_sha256(
    checkpoint_path: str,
    *,
    expected_kind: str,
    migration: Mapping[str, Any] | None,
) -> str | None:
    """Bind raw/EMA files to a checkpoint-bound migration before loading."""
    if (
        migration is None
        or migration.get("mode") not in CHECKPOINT_BOUND_MIGRATION_MODES
    ):
        return None
    if expected_kind not in {"raw", "ema"}:
        raise ValueError(
            f"unsupported checkpoint-bound source kind: {expected_kind!r}"
        )
    field = (
        "source_raw_checkpoint_sha256"
        if expected_kind == "raw"
        else "source_ema_checkpoint_sha256"
    )
    expected_sha256 = _strict_sha256(
        migration.get(field),
        label=f"checkpoint-bound migration {field}",
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"checkpoint-bound source {expected_kind} checkpoint is missing: "
            f"{checkpoint_path}"
        )
    actual_sha256 = _sha256_file(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"checkpoint-bound source {expected_kind} checkpoint SHA-256 "
            f"mismatch: actual={actual_sha256}, expected={expected_sha256}"
        )
    return actual_sha256


def _validate_weightedpdb_msa_source_training_state(
    checkpoint: Mapping[str, Any],
    migration: Mapping[str, Any] | None,
) -> None:
    """Prove the MSA-only migration preserves the source LR trajectory."""

    if (
        migration is None
        or migration.get("mode")
        != STRICT_RESUME_WEIGHTEDPDB_MSA_INDEPENDENT_MIGRATION_V1
    ):
        return
    scheduler = checkpoint.get("scheduler")
    if not isinstance(scheduler, Mapping):
        raise ValueError(
            "weightedPDB MSA-only migration requires source scheduler state"
        )
    expected_scheduler = migration["source_scheduler_state"]
    for field, expected in expected_scheduler.items():
        if scheduler.get(field) != expected:
            raise ValueError(
                "weightedPDB MSA-only source scheduler mismatch for "
                f"{field}: checkpoint={scheduler.get(field)!r}, "
                f"required={expected!r}"
            )
    optimizer = checkpoint.get("optimizer")
    param_groups = (
        optimizer.get("param_groups")
        if isinstance(optimizer, Mapping)
        else None
    )
    if not isinstance(param_groups, list) or not param_groups:
        raise ValueError(
            "weightedPDB MSA-only migration requires optimizer param_groups"
        )
    for index, param_group in enumerate(param_groups):
        if not isinstance(param_group, Mapping):
            raise ValueError(
                "weightedPDB MSA-only optimizer param_group must be an object"
            )
        try:
            current_lr = float(param_group["lr"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "weightedPDB MSA-only optimizer param_group lacks a valid lr"
            ) from exc
        if not math.isclose(current_lr, 0.00095, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(
                "weightedPDB MSA-only source optimizer LR mismatch for "
                f"param_group {index}: checkpoint={current_lr}, "
                "required=0.00095"
            )


def strict_full_resume_enabled() -> bool:
    """Return whether the fail-closed full-state resume contract is active."""
    value = os.environ.get("PROTENIX_STRICT_FULL_RESUME", "0").strip()
    if value not in {"0", "1"}:
        raise ValueError("PROTENIX_STRICT_FULL_RESUME must be 0 or 1")
    return value == "1"


def validate_strict_full_resume_configuration(configs: Any) -> bool:
    """Reject every configuration that could silently degrade to weights-only."""
    migration_mode = strict_resume_contract_migration_mode()
    if not strict_full_resume_enabled():
        if migration_mode:
            raise ValueError(
                "strict-resume contract migration requires "
                "PROTENIX_STRICT_FULL_RESUME=1"
            )
        return False
    if os.environ.get("PROTENIX_FULL_RESUME", "").strip() != "1":
        raise ValueError(
            "PROTENIX_STRICT_FULL_RESUME=1 requires PROTENIX_FULL_RESUME=1"
        )
    expected_contract = os.environ.get(
        "PROTENIX_TRAINING_CONTRACT_ID", ""
    ).strip()
    if not expected_contract:
        raise ValueError(
            "strict full resume requires PROTENIX_TRAINING_CONTRACT_ID"
        )
    if not os.environ.get("PROTENIX_TRAINING_DATA_PROFILE", "").strip():
        raise ValueError(
            "strict full resume requires PROTENIX_TRAINING_DATA_PROFILE"
        )

    required_values = {
        "load_checkpoint_path": _get_value(configs, "load_checkpoint_path", ""),
        "load_ema_checkpoint_path": _get_value(
            configs, "load_ema_checkpoint_path", ""
        ),
    }
    for name, value in required_values.items():
        if not str(value).strip():
            raise ValueError(f"strict full resume requires non-empty {name}")
    if os.path.realpath(str(required_values["load_checkpoint_path"])) == os.path.realpath(
        str(required_values["load_ema_checkpoint_path"])
    ):
        raise ValueError("strict full resume requires distinct raw and EMA checkpoints")

    forbidden_true = {
        "load_params_only": _get_value(configs, "load_params_only", True),
        "skip_load_optimizer": _get_value(
            configs, "skip_load_optimizer", False
        ),
        "skip_load_scheduler": _get_value(
            configs, "skip_load_scheduler", False
        ),
        "skip_load_step": _get_value(configs, "skip_load_step", False),
        "load_step_for_scheduler": _get_value(
            configs, "load_step_for_scheduler", False
        ),
    }
    enabled = [name for name, value in forbidden_true.items() if bool(value)]
    if enabled:
        raise ValueError(
            "strict full resume forbids state-skipping options: "
            f"{', '.join(sorted(enabled))}"
        )
    if not bool(_get_value(configs, "load_strict", False)):
        raise ValueError("strict full resume requires load_strict=true")
    if _get_value(configs, "distributed_strategy", "") != "ddp":
        raise ValueError("crop384/db48 strict full resume requires DDP")
    if float(_get_value(configs, "ema_decay", -1.0)) <= 0:
        raise ValueError("strict full resume requires EMA to be enabled")
    if migration_mode:
        _strict_resume_migration_expectations(configs)
    return True


def build_checkpoint_resume_contract(trainer: Any) -> dict[str, Any]:
    """Build metadata shared by the raw and EMA snapshots of one save event."""
    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else int(
        _get_value(DIST_WRAPPER, "world_size", 1)
    )
    configs = trainer.configs
    data_configs = _get_value(configs, "data", {})
    training_data_contract = build_training_data_contract(configs)
    train_dataloader_runtime = build_train_dataloader_runtime_contract(configs)
    model_architecture_contract = build_model_architecture_contract(configs)
    scaler_enabled = bool(trainer.grad_scaler.is_enabled())
    contract = {
        "schema_version": STRICT_FULL_RESUME_SCHEMA_VERSION,
        "training_contract_id": os.environ.get(
            "PROTENIX_TRAINING_CONTRACT_ID", ""
        ).strip(),
        "run_id": str(getattr(trainer, "run_name", "")),
        "step": int(trainer.step),
        "global_step": int(trainer.global_step),
        "next_step": int(trainer.step) + 1,
        "next_global_step": int(trainer.global_step) + 1,
        "iters_to_accumulate": int(trainer.iters_to_accumulate),
        "world_size": world_size,
        "distributed_strategy": str(
            _get_value(configs, "distributed_strategy", "")
        ),
        "dtype": str(_get_value(configs, "dtype", "")),
        "grad_scaler_enabled": scaler_enabled,
        "ema_decay": float(_get_value(configs, "ema_decay", -1.0)),
        "ema_shadow_device": str(
            _get_value(configs, "ema_shadow_device", "")
        ),
        "data_num_workers": train_dataloader_runtime["num_workers"],
        "data_train_dl_persistent_workers": train_dataloader_runtime[
            "persistent_workers"
        ],
        "data_train_dl_prefetch_factor": train_dataloader_runtime[
            "prefetch_factor"
        ],
        "data_train_dl_pin_memory": train_dataloader_runtime["pin_memory"],
        "data_profile_id": training_data_contract["profile_id"],
        "data_contract_sha256": training_data_contract["sha256"],
        "data_train_sets": training_data_contract["train_sets"],
        "data_train_sample_weights": training_data_contract[
            "train_sample_weights"
        ],
        "data_epoch_size": training_data_contract["epoch_size"],
        "data_sampler_seed": training_data_contract["sampler_seed"],
        "data_sampler_replacement": training_data_contract[
            "sampler_replacement"
        ],
        "data_augmentation": training_data_contract["augmentation"],
        "model_architecture_contract": model_architecture_contract,
        "model_architecture_sha256": model_architecture_contract["sha256"],
        "data_worker_rng_state_checkpointed": (
            train_dataloader_runtime["num_workers"] == 0
        ),
        "data_resume_scope": "sampler_cursor_and_rank_main_process_rng",
    }
    pair_material = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    contract["checkpoint_pair_id"] = hashlib.sha256(pair_material).hexdigest()
    return contract


def validate_matching_resume_contracts(
    raw_contract: Mapping[str, Any],
    ema_contract: Mapping[str, Any],
) -> None:
    """Ensure raw parameters/state and EMA parameters came from one save event."""
    matching_fields = {
        "checkpoint_pair_id",
        "training_contract_id",
        "run_id",
        "step",
        "global_step",
        "world_size",
        "ema_decay",
        "data_profile_id",
        "data_contract_sha256",
        "model_architecture_sha256",
    }
    mismatches = [
        field
        for field in sorted(matching_fields)
        if raw_contract.get(field) != ema_contract.get(field)
    ]
    if mismatches:
        raise ValueError(
            "raw and EMA checkpoints are not a matching pair; mismatched "
            f"resume-contract fields: {', '.join(mismatches)}"
        )


def should_reset_training_data(contract: Mapping[str, Any], configs: Any) -> bool:
    """Start an explicitly requested fresh data plan once, without resetting Adam."""
    if os.environ.get("PROTENIX_RESET_DATA_SAMPLER", "0") != "1":
        return False
    current = build_training_data_contract(configs)
    if contract.get("data_contract_sha256") == current["sha256"]:
        return False
    from protenix.data.pipeline.soft_cluster_schedule import (
        load_soft_cluster_schedule_manifest,
    )

    sampler = _get_value(_get_value(configs, "data", {}), "train_sampler", {})
    info = load_soft_cluster_schedule_manifest(
        str(_get_value(sampler, "soft_cluster_schedule_manifest", ""))
    )
    if info.manifest["source_checkpoint"]["mode"] != "fresh":
        raise ValueError("resetting training data requires a fresh sampling plan")
    start_step = int(os.environ.get("START_STEP", contract["next_step"]))
    if info.origin_consumed_global_batches != start_step * int(_get_value(configs, "iters_to_accumulate", 1)):
        raise ValueError("fresh sampling plan must begin at START_STEP")
    return True


def validate_strict_checkpoint_payload(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: str,
    expected_kind: str,
    configs: Any,
    grad_scaler_enabled: bool,
    device: torch.device,
    migration_expectations: Mapping[str, Any] | None | object = (
        _MIGRATION_EXPECTATIONS_UNSET
    ),
) -> dict[str, Any]:
    """Validate every state component before a strict resume mutates training."""
    checkpoint_kind = validate_checkpoint_header(
        checkpoint,
        checkpoint_path=checkpoint_path,
        load_params_only=expected_kind == "ema",
    )
    if checkpoint_kind != expected_kind:
        raise ValueError(
            f"strict resume expected checkpoint_kind={expected_kind!r}, "
            f"got {checkpoint_kind!r}"
        )
    contract = checkpoint.get("resume_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("strict resume checkpoint is missing resume_contract")
    if contract.get("schema_version") != STRICT_FULL_RESUME_SCHEMA_VERSION:
        raise ValueError(
            "unsupported strict resume-contract schema_version: "
            f"{contract.get('schema_version')!r}"
        )

    required_contract_fields = {
        "training_contract_id",
        "run_id",
        "checkpoint_pair_id",
        "step",
        "global_step",
        "next_step",
        "next_global_step",
        "iters_to_accumulate",
        "world_size",
        "distributed_strategy",
        "dtype",
        "grad_scaler_enabled",
        "ema_decay",
        "ema_shadow_device",
        "data_num_workers",
        "data_train_dl_persistent_workers",
        "data_train_dl_prefetch_factor",
        "data_train_dl_pin_memory",
        "data_profile_id",
        "data_contract_sha256",
        "data_train_sets",
        "data_train_sample_weights",
        "data_epoch_size",
        "data_sampler_seed",
        "data_sampler_replacement",
        "data_augmentation",
        "model_architecture_contract",
        "model_architecture_sha256",
        "data_worker_rng_state_checkpointed",
        "data_resume_scope",
    }
    missing_contract_fields = required_contract_fields - set(contract)
    if missing_contract_fields:
        raise ValueError(
            "strict resume contract is missing fields: "
            f"{sorted(missing_contract_fields)}"
        )

    reset_training_data = should_reset_training_data(contract, configs)
    migration = (
        _strict_resume_migration_expectations(configs)
        if migration_expectations is _MIGRATION_EXPECTATIONS_UNSET
        else migration_expectations
    )
    if migration is not None and not isinstance(migration, Mapping):
        raise TypeError("strict resume migration expectations must be a mapping")
    expected_contract_id = (
        migration["source_training_contract_id"]
        if migration is not None
        else os.environ.get(
            "PROTENIX_TRAINING_CONTRACT_ID",
            "",
        ).strip()
    )
    if not reset_training_data and contract["training_contract_id"] != expected_contract_id:
        raise ValueError(
            "training contract mismatch: "
            f"checkpoint={contract['training_contract_id']!r}, "
            f"current={expected_contract_id!r}"
        )
    if not str(contract["run_id"]).strip():
        raise ValueError("strict resume contract requires a non-empty run_id")
    if (
        migration is not None
        and contract["run_id"] != migration["source_run_id"]
    ):
        raise ValueError(
            "strict-resume migration source run mismatch: "
            f"checkpoint={contract['run_id']!r}, "
            f"required={migration['source_run_id']!r}"
        )
    pair_id = str(contract["checkpoint_pair_id"])
    if len(pair_id) != 64 or any(
        character not in "0123456789abcdef" for character in pair_id
    ):
        raise ValueError("strict resume contract has an invalid checkpoint_pair_id")
    pair_material = dict(contract)
    pair_material.pop("checkpoint_pair_id")
    expected_pair_id = hashlib.sha256(
        json.dumps(
            pair_material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if pair_id != expected_pair_id:
        raise ValueError(
            "strict resume contract checkpoint_pair_id does not match its metadata"
        )
    if (
        migration is not None
        and migration.get("mode") in CHECKPOINT_BOUND_MIGRATION_MODES
        and pair_id != migration["source_checkpoint_pair_id"]
    ):
        raise ValueError(
            "checkpoint-bound migration source checkpoint pair mismatch: "
            f"checkpoint={pair_id}, "
            f"required={migration['source_checkpoint_pair_id']}"
        )

    checkpoint_step = checkpoint.get("step")
    validate_expected_resume_checkpoint_step(
        checkpoint_step,
        checkpoint_path=checkpoint_path,
    )
    step = int(checkpoint_step)
    iters_to_accumulate = int(_get_value(configs, "iters_to_accumulate", 1))
    expected_global_step = (step + 1) * iters_to_accumulate - 1
    if (
        migration is not None
        and migration.get("mode") in CHECKPOINT_BOUND_MIGRATION_MODES
    ):
        if step != migration["source_checkpoint_step"]:
            raise ValueError(
                "checkpoint-bound migration source checkpoint step mismatch: "
                f"checkpoint={step}, "
                f"required={migration['source_checkpoint_step']}"
            )
        if int(contract["next_global_step"]) != migration[
            "source_next_global_step"
        ]:
            raise ValueError(
                "checkpoint-bound migration source sampler boundary mismatch: "
                f"checkpoint={contract['next_global_step']}, "
                f"required={migration['source_next_global_step']}"
            )
        source_completed_updates = migration.get("source_completed_updates")
        if (
            source_completed_updates is not None
            and int(contract["next_step"]) != int(source_completed_updates)
        ):
            raise ValueError(
                "checkpoint-bound migration source completed-update boundary "
                f"mismatch: checkpoint={contract['next_step']}, "
                f"required={source_completed_updates}"
            )
    train_dataloader_runtime = build_train_dataloader_runtime_contract(configs)
    num_workers = train_dataloader_runtime["num_workers"]
    scalar_expectations = {
        "step": step,
        "next_step": step + 1,
        "global_step": expected_global_step,
        "next_global_step": expected_global_step + 1,
        "iters_to_accumulate": iters_to_accumulate,
        "distributed_strategy": _get_value(
            configs, "distributed_strategy", ""
        ),
        "dtype": _get_value(configs, "dtype", ""),
        "grad_scaler_enabled": bool(grad_scaler_enabled),
        "ema_shadow_device": _get_value(configs, "ema_shadow_device", ""),
        "data_num_workers": train_dataloader_runtime["num_workers"],
        "data_train_dl_persistent_workers": train_dataloader_runtime[
            "persistent_workers"
        ],
        "data_train_dl_prefetch_factor": train_dataloader_runtime[
            "prefetch_factor"
        ],
        "data_train_dl_pin_memory": train_dataloader_runtime["pin_memory"],
        "data_worker_rng_state_checkpointed": (
            train_dataloader_runtime["num_workers"] == 0
        ),
        "data_resume_scope": "sampler_cursor_and_rank_main_process_rng",
    }
    for field, expected_value in scalar_expectations.items():
        if contract[field] != expected_value:
            raise ValueError(
                f"strict resume contract mismatch for {field}: "
                f"checkpoint={contract[field]!r}, current={expected_value!r}"
            )

    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else int(
        _get_value(DIST_WRAPPER, "world_size", 1)
    )
    if int(contract["world_size"]) != world_size:
        raise ValueError(
            "strict resume requires the original world size: "
            f"checkpoint={contract['world_size']}, current={world_size}"
        )
    configured_ema_decay = float(_get_value(configs, "ema_decay", -1.0))
    if not math.isclose(
        float(contract["ema_decay"]),
        configured_ema_decay,
        rel_tol=0,
        abs_tol=0,
    ):
        raise ValueError(
            "strict resume EMA decay mismatch: "
            f"checkpoint={contract['ema_decay']}, current={configured_ema_decay}"
        )
    data_configs = _get_value(configs, "data", {})
    current_data_contract = (
        migration["target_data_contract"]
        if migration is not None
        else build_training_data_contract(configs)
    )
    checkpoint_data_contract = current_data_contract
    if (
        migration is not None
        and migration.get("mode")
        == STRICT_RESUME_SOFT_CLUSTER_CATALOG_REPLACEMENT_MIGRATION_V2
    ):
        checkpoint_data_contract = migration["source_data_contract"]
    data_expectations = {
        "data_profile_id": checkpoint_data_contract["profile_id"],
        "data_contract_sha256": (
            migration["source_data_contract_sha256"]
            if migration is not None
            else current_data_contract["sha256"]
        ),
        "data_train_sets": checkpoint_data_contract["train_sets"],
        "data_train_sample_weights": checkpoint_data_contract[
            "train_sample_weights"
        ],
        "data_epoch_size": checkpoint_data_contract["epoch_size"],
        "data_sampler_seed": checkpoint_data_contract["sampler_seed"],
        "data_sampler_replacement": checkpoint_data_contract[
            "sampler_replacement"
        ],
        "data_augmentation": (
            migration["source_augmentation"]
            if migration is not None
            else current_data_contract["augmentation"]
        ),
    }
    for field, expected_value in (() if reset_training_data else data_expectations.items()):
        if contract[field] != expected_value:
            raise ValueError(
                f"strict resume data contract mismatch for {field}: "
                f"checkpoint={contract[field]!r}, current={expected_value!r}"
            )
    current_architecture_contract = build_model_architecture_contract(
        configs
    )
    architecture_expectations = {
        "model_architecture_contract": current_architecture_contract,
        "model_architecture_sha256": current_architecture_contract["sha256"],
    }
    for field, expected_value in architecture_expectations.items():
        if contract[field] != expected_value:
            raise ValueError(
                f"strict resume model architecture mismatch for {field}: "
                f"checkpoint={contract[field]!r}, current={expected_value!r}"
            )
    if expected_kind == "ema":
        for state_name in ("optimizer", "scheduler", "grad_scaler"):
            if checkpoint.get(state_name) is not None:
                raise ValueError(
                    f"strict EMA checkpoint must not contain {state_name} state"
                )
        if "rank_runtime_states" in checkpoint:
            raise ValueError(
                "strict EMA checkpoint must not contain rank runtime state"
            )
        return dict(contract)

    optimizer_state = checkpoint.get("optimizer")
    if (
        not isinstance(optimizer_state, Mapping)
        or not optimizer_state.get("param_groups")
        or not optimizer_state.get("state")
    ):
        raise ValueError(
            "strict raw checkpoint requires non-empty optimizer moments and "
            "param_groups"
        )
    if not isinstance(checkpoint.get("scheduler"), Mapping):
        raise ValueError("strict raw checkpoint requires scheduler state")
    _validate_weightedpdb_msa_source_training_state(checkpoint, migration)
    if not isinstance(checkpoint.get("grad_scaler"), Mapping):
        raise ValueError(
            "strict raw checkpoint requires GradScaler state, including the "
            "disabled-state mapping used by BF16"
        )
    rank_runtime_states = checkpoint.get("rank_runtime_states")
    if not isinstance(rank_runtime_states, Mapping):
        raise ValueError(
            "strict raw checkpoint requires per-rank RNG and dataloader state"
        )
    if rank_runtime_states.get("schema_version") != 1:
        raise ValueError(
            "strict raw checkpoint has unsupported rank runtime-state schema"
        )
    if int(rank_runtime_states.get("world_size", -1)) != world_size:
        raise ValueError(
            "strict raw checkpoint rank runtime-state world size mismatch"
        )
    states = rank_runtime_states.get("states")
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        raise ValueError("strict raw checkpoint requires a rank-state sequence")
    if len(states) != world_size:
        raise ValueError(
            "strict raw checkpoint rank-state count does not match world size"
        )
    observed_ranks = set()
    for state in states:
        if not isinstance(state, Mapping):
            raise ValueError("strict raw checkpoint contains an invalid rank state")
        rank = int(state.get("rank", -1))
        observed_ranks.add(rank)
        rng_state = state.get("rng")
        if not isinstance(rng_state, Mapping) or set(rng_state) != {
            "python",
            "numpy",
            "torch",
            "cuda",
        }:
            raise ValueError(
                f"strict raw checkpoint rank {rank} has incomplete RNG state"
            )
        if torch.device(device).type == "cuda" and rng_state["cuda"] is None:
            raise ValueError(
                f"strict raw checkpoint rank {rank} is missing CUDA RNG state"
            )
        dataloader_state = state.get("train_dataloader")
        if not isinstance(dataloader_state, Mapping):
            raise ValueError(
                f"strict raw checkpoint rank {rank} is missing dataloader state"
            )
        if dataloader_state.get("schema_version") != 1:
            raise ValueError(
                f"strict raw checkpoint rank {rank} has unsupported dataloader state"
            )
        if int(dataloader_state.get("num_workers", -1)) != num_workers:
            raise ValueError(
                f"strict raw checkpoint rank {rank} dataloader worker count mismatch"
            )
        if int(dataloader_state.get("consumed_global_batches", -1)) != int(
            contract["next_global_step"]
        ):
            raise ValueError(
                f"strict raw checkpoint rank {rank} dataloader cursor does not "
                "match next_global_step"
            )
    if observed_ranks != set(range(world_size)):
        raise ValueError(
            "strict raw checkpoint rank states are missing or duplicated"
        )
    return dict(contract)


def capture_rank_rng_state(device: torch.device) -> dict[str, Any]:
    """Capture the process RNG streams needed by a full training resume."""
    device = torch.device(device)
    cuda_state = None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "cannot capture CUDA RNG state because CUDA is unavailable"
            )
        cuda_state = torch.cuda.get_rng_state(device=device)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": cuda_state,
    }


def restore_rank_rng_state(
    rng_state: Mapping[str, Any],
    device: torch.device,
) -> None:
    """Restore one rank's Python, NumPy, torch CPU, and local CUDA RNG."""
    required_keys = {"python", "numpy", "torch", "cuda"}
    missing_keys = required_keys - set(rng_state)
    if missing_keys:
        raise ValueError(
            "checkpoint rank RNG state is missing keys: "
            f"{sorted(missing_keys)}"
        )
    random.setstate(rng_state["python"])
    np.random.set_state(tuple(rng_state["numpy"]))
    torch.set_rng_state(rng_state["torch"].cpu())

    cuda_state = rng_state["cuda"]
    device = torch.device(device)
    if cuda_state is not None:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "checkpoint contains CUDA RNG state but the resumed rank has "
                f"device={device}"
            )
        torch.cuda.set_rng_state(cuda_state.cpu(), device=device)


def synchronize_rank_exception(
    phase: str,
    exception: BaseException | None,
) -> None:
    """Make ordinary per-rank DDP failures visible before the next collective."""
    if not dist.is_available() or not dist.is_initialized():
        if exception is not None:
            raise exception
        return

    local_record = (
        None
        if exception is None
        else {
            "rank": dist.get_rank(),
            "error_type": type(exception).__name__,
            "error": str(exception),
        }
    )
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, local_record)
    failures = [record for record in records if record is not None]
    if not failures:
        return

    details = "; ".join(
        f"rank {record['rank']}: {record['error_type']}: {record['error']}"
        for record in failures
    )
    error = RuntimeError(f"{phase} failed on one or more ranks: {details}")
    if exception is not None:
        raise error from exception
    raise error


def _last_shape_dim(value: Any, dim: int) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        return None
    try:
        return int(shape[dim])
    except (IndexError, TypeError, ValueError):
        return None


DATA_PIPELINE_TIMING_KEYS = (
    "bioassembly_load_seconds",
    "crop_seconds",
    "feature_label_seconds",
    "msa_featurizer_seconds",
    "template_featurizer_seconds",
    "template_load_seconds",
    "template_align_seconds",
    "template_feature_seconds",
    "sequence_template_cache_load_seconds",
    "sequence_template_cache_files_loaded",
    "sequence_template_cache_candidates",
    "sequence_template_cache_selected",
    "sequence_template_cache_reuses",
    "data_process_seconds",
)

SEQUENCE_TEMPLATE_CACHE_CHAIN_DUMMY_REASONS = (
    "mapping_missing",
    "unsafe_path",
    "missing",
    "unreadable",
    "invalid_payload",
    "unknown",
)
SEQUENCE_TEMPLATE_CACHE_CHAIN_DUMMY_METRIC_KEYS = (
    "N_sequence_template_cache_chain_dummy_fallback_unique_sequence",
    "N_sequence_template_cache_chain_dummy_fallback_chain",
    *(
        "N_sequence_template_cache_chain_dummy_fallback_reason_" + reason
        for reason in SEQUENCE_TEMPLATE_CACHE_CHAIN_DUMMY_REASONS
    ),
)


def summarize_batch(batch: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract small, stable sample metadata from a training/eval batch."""
    basic = _get_value(batch, "basic", {})
    input_feature_dict = _get_value(batch, "input_feature_dict", {})
    summary = {}
    for key in [
        "pdb_id",
        "dataset_name",
        "data_id",
        "chain_id",
        "entity_id",
        "mol_type",
        "N_msa",
        "N_msa_prot_pair",
        "N_msa_prot_unpair",
        "N_msa_rna_pair",
        "N_msa_rna_unpair",
        "N_msa_virtual_segment",
        "msa_virtual_segment_shuffle_applied",
        "N_msa_shuffle_group",
        "msa_shuffle_requested_variant_code",
        "msa_shuffle_variant_eligible",
        "msa_shuffle_variant_applied",
        "msa_shuffle_whole_physical_chain_applied",
        "N_msa_crop_protein_chain",
        "N_msa_query_only_protein_chain",
        "N_msa_retained_protein_chain",
        "msa_partial_query_only_applied",
        "msa_global_query_only_applied",
        "N_openfold3_representative_query_mismatch",
        "N_openfold3_representative_remap",
        "N_openfold3_representative_query_only_fallback",
        "N_openfold3_explicit_query_only",
        "msa_mode",
        "query_only_msa",
        "unpaired_msa",
        "pair_msa_as_unpaired_only",
        "shuffle_pair_msa_when_unpaired_only",
        "template_use_requested",
        "template_dropout_applied",
        "N_template_valid",
        *SEQUENCE_TEMPLATE_CACHE_CHAIN_DUMMY_METRIC_KEYS,
        "sample_sampler_epoch",
        "sample_global_position",
        "N_prot_token",
        "N_lig_token",
        "N_rna_token",
        "N_dna_token",
        "N_diffusion_supervised_token",
        *DATA_PIPELINE_TIMING_KEYS,
    ]:
        value = _get_value(basic, key)
        if value is not None:
            summary[key] = to_jsonable(value)

    residue_index = _get_value(input_feature_dict, "residue_index")
    if residue_index is not None:
        summary["n_token"] = _last_shape_dim(residue_index, -1)

    ref_pos = _get_value(input_feature_dict, "ref_pos")
    if ref_pos is not None:
        summary["n_atom"] = _last_shape_dim(ref_pos, -2)
    else:
        atom_to_token_idx = _get_value(input_feature_dict, "atom_to_token_idx")
        if atom_to_token_idx is not None:
            summary["n_atom"] = _last_shape_dim(atom_to_token_idx, -1)
    return {k: v for k, v in summary.items() if v is not None}


def iter_dataloader_with_timing(data_loader: Any):
    """Yield batches together with time spent waiting for each batch.

    Iterator construction is charged to the first batch. This makes worker
    startup and epoch-boundary iterator reset costs visible without changing
    the sampler or DataLoader iteration semantics.
    """
    iterator_start = time.perf_counter()
    iterator = iter(data_loader)
    iterator_setup_seconds = time.perf_counter() - iterator_start
    while True:
        wait_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            return
        data_wait_seconds = (
            iterator_setup_seconds + time.perf_counter() - wait_start
        )
        iterator_setup_seconds = 0.0
        yield batch, data_wait_seconds


def benchmark_sample_identity(batch_summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a stable source-sample identity for exact-coverage audits."""
    identity = {
        key: _single_value(batch_summary.get(key))
        for key in ("dataset_name", "pdb_id", "data_id")
    }
    return {key: to_jsonable(value) for key, value in identity.items() if value is not None}


def build_benchmark_sample_coverage(
    gathered_sample_records: Sequence[Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Aggregate global uniqueness evidence from rank-local sample identities."""
    all_sample_records = [
        dict(record)
        for rank_sample_records in gathered_sample_records
        for record in (rank_sample_records or [])
    ]
    sample_identity_strings = [
        json.dumps(
            record.get("identity", {}),
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in all_sample_records
    ]
    coverage_digest = hashlib.sha256()
    for record in all_sample_records:
        coverage_digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        coverage_digest.update(b"\n")
    unique_sample_count = len(set(sample_identity_strings))
    return {
        "sample_count": len(all_sample_records),
        "unique_sample_count": unique_sample_count,
        "duplicate_sample_count": len(all_sample_records) - unique_sample_count,
        "missing_identity_count": sum(
            not bool(record.get("identity_complete"))
            for record in all_sample_records
        ),
        "per_rank_sample_count": [
            len(rank_sample_records or [])
            for rank_sample_records in gathered_sample_records
        ],
        "ordered_sha256": coverage_digest.hexdigest(),
    }


BENCHMARK_TOKEN_COMPOSITION_KEYS = (
    "n_token",
    "N_prot_token",
    "N_lig_token",
    "N_rna_token",
    "N_dna_token",
)


def build_benchmark_token_composition_coverage(
    gathered_sample_records: Sequence[Sequence[Mapping[str, Any]]],
    *,
    max_nuc_tokens: int,
) -> Dict[str, Any]:
    """Aggregate crop-level molecule token counts for every benchmark sample."""
    rank_records = [list(records or []) for records in gathered_sample_records]
    all_records = [record for records in rank_records for record in records]
    complete_records: list[dict[str, Any]] = []
    missing_composition_count = 0

    for record in all_records:
        composition = record.get("token_composition")
        if not isinstance(composition, Mapping) or any(
            key not in composition for key in BENCHMARK_TOKEN_COMPOSITION_KEYS
        ):
            missing_composition_count += 1
            continue
        values = {
            key: _int_or_zero(composition.get(key))
            for key in BENCHMARK_TOKEN_COMPOSITION_KEYS
        }
        values["N_nuc_token"] = values["N_rna_token"] + values["N_dna_token"]
        identity = record.get("identity") or {}
        complete_records.append(
            {
                "rank": _int_or_zero(record.get("rank")),
                "step": _int_or_zero(record.get("step")),
                "dataset_name": identity.get("dataset_name"),
                "pdb_id": identity.get("pdb_id"),
                "data_id": identity.get("data_id"),
                **values,
            }
        )

    token_counts = [record["n_token"] for record in complete_records]
    nuc_counts = [record["N_nuc_token"] for record in complete_records]
    nuc_present_counts = [value for value in nuc_counts if value > 0]

    def distribution(values: Sequence[int]) -> Dict[str, Any]:
        numeric_values = list(values)
        return {
            "count": len(numeric_values),
            "min": min(numeric_values) if numeric_values else None,
            "median": _percentile(numeric_values, 0.5),
            "p95": _percentile(numeric_values, 0.95),
            "p99": _percentile(numeric_values, 0.99),
            "max": max(numeric_values) if numeric_values else None,
            "mean": (
                sum(numeric_values) / len(numeric_values)
                if numeric_values
                else None
            ),
            "sum": sum(numeric_values),
        }

    return {
        "target_max_nuc_tokens": int(max_nuc_tokens),
        "sample_count": len(all_records),
        "complete_composition_count": len(complete_records),
        "missing_composition_count": missing_composition_count,
        "per_rank_sample_count": [len(records) for records in rank_records],
        "token_count": distribution(token_counts),
        "nuc_token_count": distribution(nuc_counts),
        "nuc_present_token_count": distribution(nuc_present_counts),
        "nuc_present_sample_count": len(nuc_present_counts),
        "over_nuc_limit_count": (
            sum(value > max_nuc_tokens for value in nuc_counts)
            if max_nuc_tokens > 0
            else 0
        ),
        "token_type_sum_mismatch_count": sum(
            record["N_prot_token"]
            + record["N_lig_token"]
            + record["N_rna_token"]
            + record["N_dna_token"]
            != record["n_token"]
            for record in complete_records
        ),
        "type_token_sums": {
            key: sum(record[key] for record in complete_records)
            for key in BENCHMARK_TOKEN_COMPOSITION_KEYS[1:]
        },
        "sample_records": complete_records,
    }


def build_benchmark_msa_depth_coverage(
    gathered_step_records: Sequence[Sequence[Mapping[str, Any]]],
    *,
    target_depth: int,
) -> Dict[str, Any]:
    """Aggregate per-recycle MSA input/sample depths across all ranks.

    A training step may invoke the MSA module more than once because of
    recycling.  Keeping every invocation prevents a final-cycle-only check
    from silently accepting shallower MSA sampling in an earlier recycle.
    """
    rank_records = [list(records or []) for records in gathered_step_records]
    all_step_records = [record for records in rank_records for record in records]
    all_calls = [
        call
        for record in all_step_records
        for call in (record.get("calls") or [])
        if isinstance(call, Mapping)
    ]

    def _valid_depths(key: str) -> list[int]:
        values = []
        for call in all_calls:
            value = call.get(key)
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                values.append(value)
        return values

    input_depths = _valid_depths("input_depth")
    sampled_depths = _valid_depths("sampled_depth")
    return {
        "target_depth": int(target_depth),
        "step_record_count": len(all_step_records),
        "call_count": len(all_calls),
        "missing_trace_step_count": sum(
            not bool(record.get("calls")) for record in all_step_records
        ),
        "missing_input_depth_call_count": len(all_calls) - len(input_depths),
        "missing_sampled_depth_call_count": len(all_calls) - len(sampled_depths),
        "input_depth_min": min(input_depths) if input_depths else None,
        "input_depth_max": max(input_depths) if input_depths else None,
        "sampled_depth_min": min(sampled_depths) if sampled_depths else None,
        "sampled_depth_max": max(sampled_depths) if sampled_depths else None,
        "sampled_target_call_count": sum(
            value == target_depth for value in sampled_depths
        ),
        "below_target_call_count": sum(
            value < target_depth for value in sampled_depths
        ),
        "per_rank_step_count": [len(records) for records in rank_records],
        "per_rank_call_count": [
            sum(len(record.get("calls") or []) for record in records)
            for records in rank_records
        ],
    }


def _single_value(value: Any) -> Any:
    value = to_jsonable(value)
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _dataset_name(value: Any) -> str:
    value = _single_value(value)
    return str(value) if value is not None else "unknown"


def _int_or_zero(value: Any) -> int:
    value = _single_value(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _env_int(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logging.warning("Ignoring invalid integer env %s=%r", name, value)
        return default


def _finite_scalar_float(value: Any) -> float | None:
    value = _scalar_float(value)
    return value if value is not None and math.isfinite(value) else None


def _scalar_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    elif hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value


def _global_scalar_float(value: Any) -> float | None:
    """Materialize a scalar DTensor (including Partial) before logging it."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = ()
    if DTensor and isinstance(value, DTensor):
        # clip_grad_norm_ returns a scalar with a partial norm placement under
        # FSDP2. full_tensor() performs the required reduction on every rank.
        value = value.full_tensor()
    return _scalar_float(value)


def _has_nonfinite_local_gradients(model: torch.nn.Module) -> bool:
    """Check rank-local gradient shards when no scalar norm is available."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = ()
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient = gradient.detach()
        if DTensor and isinstance(gradient, DTensor):
            gradient = gradient.to_local()
        if not bool(torch.isfinite(gradient).all().item()):
            return True
    return False


def _bool_value(value: Any) -> bool:
    value = _single_value(value)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[index]


def _numeric_window_stats(
    records: list[Mapping[str, Any]],
    key: str,
) -> Dict[str, Any]:
    values: list[tuple[float, int | None]] = []
    for record in records:
        value = _finite_scalar_float(record.get(key))
        if value is None:
            continue
        rank = record.get("rank")
        values.append((value, int(rank) if rank is not None else None))
    nums = [value for value, _ in values]
    if not nums:
        return {
            "count": 0,
            "avg": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
            "max_rank": None,
        }
    max_index = max(range(len(values)), key=lambda i: values[i][0])
    return {
        "count": len(nums),
        "avg": sum(nums) / len(nums),
        "p50": _percentile(nums, 0.50),
        "p95": _percentile(nums, 0.95),
        "p99": _percentile(nums, 0.99),
        "min": min(nums),
        "max": values[max_index][0],
        "max_rank": values[max_index][1],
    }


def _top_runtime_outliers(
    records: list[Mapping[str, Any]],
    key: str,
    *,
    limit: int = 5,
) -> list[Dict[str, Any]]:
    """Return compact sample identities for the slowest records."""
    ranked: list[tuple[float, Mapping[str, Any]]] = []
    for record in records:
        value = _finite_scalar_float(record.get(key))
        if value is None:
            timings = record.get("timings", {})
            if isinstance(timings, Mapping):
                value = _finite_scalar_float(timings.get(key))
        if value is not None:
            ranked.append((value, record))

    result = []
    for value, record in sorted(ranked, key=lambda item: item[0], reverse=True)[
        :limit
    ]:
        outlier = {
            key: value,
            "rank": record.get("rank"),
            "step": record.get("step"),
            "global_step": record.get("global_step"),
        }
        for identity_key in (
            "pdb_id",
            "dataset_name",
            "data_id",
            "n_token",
            "n_atom",
            "N_cycle",
            "N_msa",
            "N_msa_prot_pair",
            "N_msa_prot_unpair",
            "N_msa_rna_pair",
            "N_msa_rna_unpair",
            "N_msa_virtual_segment",
            "msa_virtual_segment_shuffle_applied",
            "N_msa_shuffle_group",
            "msa_shuffle_requested_variant_code",
            "msa_shuffle_variant_eligible",
            "msa_shuffle_variant_applied",
            "msa_shuffle_whole_physical_chain_applied",
            "N_msa_crop_protein_chain",
            "N_msa_query_only_protein_chain",
            "N_msa_retained_protein_chain",
            "msa_partial_query_only_applied",
            "msa_global_query_only_applied",
            "N_openfold3_representative_query_mismatch",
            "N_openfold3_representative_remap",
            "N_openfold3_representative_query_only_fallback",
            "N_openfold3_explicit_query_only",
            "N_prot_token",
            "N_lig_token",
            "N_rna_token",
            "N_dna_token",
            "msa_mode",
            "query_only_msa",
            "unpaired_msa",
            "N_template_valid",
            "template_use_requested",
            "chain_id",
            "sample_sampler_epoch",
            "sample_global_position",
        ):
            identity_value = record.get(identity_key)
            if identity_value is not None:
                outlier[identity_key] = to_jsonable(identity_value)
        result.append(outlier)
    return result


def _memory_extreme(
    records: list[Mapping[str, Any]],
    key: str,
    mode: str,
) -> Dict[str, Any]:
    values: list[tuple[float, int | None]] = []
    for record in records:
        memory = record.get("cuda_memory", {})
        if not isinstance(memory, Mapping):
            continue
        value = _finite_scalar_float(memory.get(key))
        if value is None:
            continue
        rank = record.get("rank")
        values.append((value, int(rank) if rank is not None else None))
    if not values:
        return {mode: None, "rank": None}
    index = (
        max(range(len(values)), key=lambda i: values[i][0])
        if mode == "max"
        else min(range(len(values)), key=lambda i: values[i][0])
    )
    return {mode: values[index][0], "rank": values[index][1]}


TOKEN_LENGTH_BINS: tuple[tuple[int, int | None, str], ...] = (
    (0, 128, "0000_0128"),
    (129, 256, "0129_0256"),
    (257, 512, "0257_0512"),
    (513, 768, "0513_0768"),
    (769, None, "0769_plus"),
)


def _token_length_bin(n_token: int) -> str:
    for lower, upper, name in TOKEN_LENGTH_BINS:
        if n_token >= lower and (upper is None or n_token <= upper):
            return name
    return TOKEN_LENGTH_BINS[0][2]


def _avg_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def build_token_length_bin_stats(
    records: list[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate runtime and losses by n_token ranges."""
    bin_accumulators: dict[str, dict[str, Any]] = {
        name: {
            "sample_count": 0,
            "token_count": 0,
            "atom_count": 0,
            "step_wall_seconds": [],
            "forward_seconds": [],
            "backward_seconds": [],
            "losses": {},
        }
        for _, _, name in TOKEN_LENGTH_BINS
    }

    for record in records:
        n_token = _int_or_zero(record.get("n_token"))
        n_atom = _int_or_zero(record.get("n_atom"))
        stats = bin_accumulators[_token_length_bin(n_token)]
        stats["sample_count"] += 1
        stats["token_count"] += n_token
        stats["atom_count"] += n_atom

        step_wall_seconds = _finite_scalar_float(record.get("step_wall_seconds"))
        if step_wall_seconds is not None:
            stats["step_wall_seconds"].append(step_wall_seconds)

        timings = record.get("timings", {})
        if isinstance(timings, Mapping):
            for timing_key in ["forward_seconds", "backward_seconds"]:
                timing_value = _finite_scalar_float(timings.get(timing_key))
                if timing_value is not None:
                    stats[timing_key].append(timing_value)

        losses = record.get("losses", {})
        if not isinstance(losses, Mapping):
            continue
        loss_accumulator = stats["losses"]
        for loss_name, loss_value in losses.items():
            if "loss" not in str(loss_name):
                continue
            loss_value = _finite_scalar_float(loss_value)
            if loss_value is not None:
                loss_accumulator.setdefault(str(loss_name), []).append(loss_value)

    result: Dict[str, Dict[str, Any]] = {}
    for _, _, name in TOKEN_LENGTH_BINS:
        stats = bin_accumulators[name]
        sample_count = stats["sample_count"]
        step_wall_seconds = stats["step_wall_seconds"]
        result[name] = {
            "sample_count": sample_count,
            "token_count": stats["token_count"],
            "atom_count": stats["atom_count"],
            "avg_n_token": (
                stats["token_count"] / sample_count if sample_count else 0.0
            ),
            "avg_n_atom": stats["atom_count"] / sample_count if sample_count else 0.0,
            "loss_avg": {
                loss_name: sum(values) / len(values)
                for loss_name, values in sorted(stats["losses"].items())
                if values
            },
            "step_wall_seconds_avg": _avg_or_none(step_wall_seconds),
            "step_wall_seconds_p95": _percentile(step_wall_seconds, 0.95),
            "step_wall_seconds_max": (
                max(step_wall_seconds) if step_wall_seconds else None
            ),
            "forward_seconds_avg": _avg_or_none(stats["forward_seconds"]),
            "backward_seconds_avg": _avg_or_none(stats["backward_seconds"]),
        }
    return result


def build_train_runtime_window_stats(records: list[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate recent runtime, memory, and optimizer health diagnostics."""
    total_samples = len(records)
    total_tokens = sum(_int_or_zero(record.get("n_token")) for record in records)
    total_atoms = sum(_int_or_zero(record.get("n_atom")) for record in records)
    template_cache_chain_dummy_unique_sequence_count = sum(
        _int_or_zero(
            record.get(
                "N_sequence_template_cache_chain_dummy_fallback_unique_sequence"
            )
        )
        for record in records
    )
    template_cache_chain_dummy_affected_chain_count = sum(
        _int_or_zero(
            record.get(
                "N_sequence_template_cache_chain_dummy_fallback_chain"
            )
        )
        for record in records
    )
    template_cache_chain_dummy_reason_counts = {
        reason: sum(
            _int_or_zero(
                record.get(
                    "N_sequence_template_cache_chain_dummy_fallback_reason_"
                    + reason
                )
            )
            for record in records
        )
        for reason in SEQUENCE_TEMPLATE_CACHE_CHAIN_DUMMY_REASONS
    }

    elapsed_by_step: dict[str, float] = {}
    end_to_end_by_step: dict[str, float] = {}
    wall_by_rank: dict[int, float] = {}
    end_to_end_by_rank: dict[int, float] = {}
    for index, record in enumerate(records):
        wall_seconds = _finite_scalar_float(record.get("step_wall_seconds"))
        if wall_seconds is None:
            continue
        step_key = str(record.get("global_step", record.get("step", index)))
        elapsed_by_step[step_key] = max(elapsed_by_step.get(step_key, 0.0), wall_seconds)
        rank = int(record.get("rank", -1))
        wall_by_rank[rank] = wall_by_rank.get(rank, 0.0) + wall_seconds
        data_wait_seconds = (
            _finite_scalar_float(record.get("data_wait_seconds")) or 0.0
        )
        end_to_end_seconds = wall_seconds + data_wait_seconds
        end_to_end_by_step[step_key] = max(
            end_to_end_by_step.get(step_key, 0.0),
            end_to_end_seconds,
        )
        end_to_end_by_rank[rank] = (
            end_to_end_by_rank.get(rank, 0.0) + end_to_end_seconds
        )

    elapsed_seconds = sum(elapsed_by_step.values())
    end_to_end_elapsed_seconds = sum(end_to_end_by_step.values())
    slowest_rank = (
        max(wall_by_rank.items(), key=lambda item: item[1])[0]
        if wall_by_rank
        else None
    )
    slowest_end_to_end_rank = (
        max(end_to_end_by_rank.items(), key=lambda item: item[1])[0]
        if end_to_end_by_rank
        else None
    )

    timing_records: list[dict[str, Any]] = []
    for record in records:
        record_step_wall = _finite_scalar_float(
            record.get("step_wall_seconds")
        )
        record_data_wait = _finite_scalar_float(
            record.get("data_wait_seconds")
        )
        flat = {
            "rank": record.get("rank"),
            "step_wall_seconds": record_step_wall,
            "data_wait_seconds": record_data_wait,
            "end_to_end_seconds": (
                record_step_wall + (record_data_wait or 0.0)
                if record_step_wall is not None
                else None
            ),
            "to_device_seconds": record.get("to_device_seconds"),
            "ema_seconds": record.get("ema_seconds"),
            "sampler_epoch_boundary": _bool_value(
                record.get("sampler_epoch_boundary")
            ),
        }
        flat.update(
            {
                key: record.get(key)
                for key in DATA_PIPELINE_TIMING_KEYS
            }
        )
        timings = record.get("timings", {})
        if isinstance(timings, Mapping):
            flat.update(timings)
        timing_records.append(flat)

    epoch_boundary_timing_records = [
        record
        for record in timing_records
        if record["sampler_epoch_boundary"]
    ]
    non_boundary_timing_records = [
        record
        for record in timing_records
        if not record["sampler_epoch_boundary"]
    ]
    epoch_comparison_keys = (
        "data_wait_seconds",
        "data_process_seconds",
        "end_to_end_seconds",
        "step_wall_seconds",
    )

    grad_norms = [
        value
        for value in (_finite_scalar_float(record.get("grad_norm")) for record in records)
        if value is not None
    ]
    amp_scales = [
        value
        for value in (_finite_scalar_float(record.get("amp_scale")) for record in records)
        if value is not None
    ]
    pre_permutation_supervised_tokens = [
        value
        for value in (
            _finite_scalar_float(
                record.get("N_diffusion_supervised_token")
            )
            for record in records
        )
        if value is not None
    ]
    post_permutation_supervised_tokens = [
        value
        for value in (
            _finite_scalar_float(
                record.get(
                    "post_permutation_diffusion_supervised_tokens"
                )
            )
            for record in records
        )
        if value is not None
    ]

    return {
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "total_atoms": total_atoms,
        "elapsed_seconds": elapsed_seconds,
        "compute_elapsed_seconds": elapsed_seconds,
        "end_to_end_elapsed_seconds": end_to_end_elapsed_seconds,
        "samples_per_sec": total_samples / elapsed_seconds if elapsed_seconds else 0.0,
        "tokens_per_sec": total_tokens / elapsed_seconds if elapsed_seconds else 0.0,
        "atoms_per_sec": total_atoms / elapsed_seconds if elapsed_seconds else 0.0,
        "end_to_end_samples_per_sec": (
            total_samples / end_to_end_elapsed_seconds
            if end_to_end_elapsed_seconds
            else 0.0
        ),
        "end_to_end_tokens_per_sec": (
            total_tokens / end_to_end_elapsed_seconds
            if end_to_end_elapsed_seconds
            else 0.0
        ),
        "end_to_end_atoms_per_sec": (
            total_atoms / end_to_end_elapsed_seconds
            if end_to_end_elapsed_seconds
            else 0.0
        ),
        "template_cache": {
            "chain_dummy_fallback_unique_sequence_count": (
                template_cache_chain_dummy_unique_sequence_count
            ),
            "chain_dummy_fallback_affected_chain_count": (
                template_cache_chain_dummy_affected_chain_count
            ),
            "chain_dummy_fallback_target_rate": (
                sum(
                    int(
                        _int_or_zero(
                            record.get(
                                "N_sequence_template_cache_chain_dummy_fallback_unique_sequence"
                            )
                        )
                        > 0
                    )
                    for record in records
                )
                / total_samples
                if total_samples
                else 0.0
            ),
            "chain_dummy_fallback_reason_counts": (
                template_cache_chain_dummy_reason_counts
            ),
        },
        "slowest_rank": slowest_rank,
        "slowest_end_to_end_rank": slowest_end_to_end_rank,
        "diffusion_supervision": {
            "pre_permutation_min": (
                min(pre_permutation_supervised_tokens)
                if pre_permutation_supervised_tokens
                else None
            ),
            "post_permutation_min": (
                min(post_permutation_supervised_tokens)
                if post_permutation_supervised_tokens
                else None
            ),
            "chain_candidate_rejection_count": sum(
                1
                for record in records
                if _bool_value(
                    record.get(
                        "chain_permutation_min_supervision_rejected"
                    )
                )
            ),
            "atom_candidate_rejection_count": sum(
                1
                for record in records
                if _bool_value(
                    record.get(
                        "atom_permutation_min_supervision_rejected"
                    )
                )
            ),
        },
        "rank_step_wall_seconds": {
            str(rank): seconds for rank, seconds in sorted(wall_by_rank.items())
        },
        "rank_end_to_end_seconds": {
            str(rank): seconds
            for rank, seconds in sorted(end_to_end_by_rank.items())
        },
        "length_bins": {
            "token_bins": build_token_length_bin_stats(records),
        },
        "timings": {
            key: _numeric_window_stats(timing_records, key)
            for key in [
                "step_wall_seconds",
                "data_wait_seconds",
                "end_to_end_seconds",
                "to_device_seconds",
                "ema_seconds",
                *DATA_PIPELINE_TIMING_KEYS,
                "forward_seconds",
                "loss_seconds",
                "loss_health_sync_seconds",
                "backward_seconds",
                "optimizer_seconds",
                "train_step_seconds",
            ]
        },
        "top_outliers": {
            key: _top_runtime_outliers(records, key)
            for key in (
                "data_wait_seconds",
                "data_process_seconds",
                "bioassembly_load_seconds",
                "msa_featurizer_seconds",
                "template_featurizer_seconds",
                "loss_health_sync_seconds",
                "step_wall_seconds",
            )
        },
        "sampler_epoch_boundary_comparison": {
            "boundary_record_count": len(epoch_boundary_timing_records),
            "non_boundary_record_count": len(non_boundary_timing_records),
            "boundary_timings": {
                key: _numeric_window_stats(epoch_boundary_timing_records, key)
                for key in epoch_comparison_keys
            },
            "non_boundary_timings": {
                key: _numeric_window_stats(non_boundary_timing_records, key)
                for key in epoch_comparison_keys
            },
        },
        "memory": {
            "allocated_bytes": _memory_extreme(records, "allocated_bytes", "max"),
            "reserved_bytes": _memory_extreme(records, "reserved_bytes", "max"),
            "max_allocated_bytes": _memory_extreme(
                records, "max_allocated_bytes", "max"
            ),
            "max_reserved_bytes": _memory_extreme(records, "max_reserved_bytes", "max"),
            "free_bytes": _memory_extreme(records, "free_bytes", "min"),
            "total_bytes": _memory_extreme(records, "total_bytes", "max"),
        },
        "grad_health": {
            "grad_norm_avg": sum(grad_norms) / len(grad_norms) if grad_norms else None,
            "grad_norm_max": max(grad_norms) if grad_norms else None,
            "grad_clip_count": sum(
                1 for record in records if _bool_value(record.get("grad_clipped"))
            ),
            "loss_nan_count": sum(
                1 for record in records if _bool_value(record.get("loss_nan"))
            ),
            "nonfinite_grad_count": sum(
                1
                for record in records
                if _bool_value(record.get("nonfinite_grad"))
                or (
                    (value := _scalar_float(record.get("grad_norm"))) is not None
                    and not math.isfinite(value)
                )
            ),
            "optimizer_skipped_count": sum(
                1 for record in records if _bool_value(record.get("optimizer_skipped"))
            ),
            "amp_scale_min": min(amp_scales) if amp_scales else None,
            "amp_scale_max": max(amp_scales) if amp_scales else None,
        },
    }


def build_train_dataset_window_stats(records: list[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate recent train samples by dataset for data-mix diagnostics."""
    per_dataset: dict[str, dict[str, Any]] = {}
    loss_values: dict[str, dict[str, list[float]]] = {}

    for record in records:
        dataset_name = _dataset_name(record.get("dataset_name"))
        n_token = _int_or_zero(record.get("n_token"))
        n_atom = _int_or_zero(record.get("n_atom"))
        stats = per_dataset.setdefault(
            dataset_name,
            {
                "sample_count": 0,
                "token_count": 0,
                "atom_count": 0,
            },
        )
        stats["sample_count"] += 1
        stats["token_count"] += n_token
        stats["atom_count"] += n_atom

        dataset_losses = loss_values.setdefault(dataset_name, {})
        for loss_name, loss_value in record.get("losses", {}).items():
            loss_value = _finite_scalar_float(loss_value)
            if loss_value is None:
                continue
            dataset_losses.setdefault(str(loss_name), []).append(loss_value)

    total_samples = sum(v["sample_count"] for v in per_dataset.values())
    total_tokens = sum(v["token_count"] for v in per_dataset.values())
    total_atoms = sum(v["atom_count"] for v in per_dataset.values())

    sample_count_by_dataset = {
        dataset_name: stats["sample_count"]
        for dataset_name, stats in sorted(per_dataset.items())
    }
    token_count_by_dataset = {
        dataset_name: stats["token_count"]
        for dataset_name, stats in sorted(per_dataset.items())
    }
    atom_count_by_dataset = {
        dataset_name: stats["atom_count"]
        for dataset_name, stats in sorted(per_dataset.items())
    }

    return {
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "total_atoms": total_atoms,
        "sample_count_by_dataset": sample_count_by_dataset,
        "sample_ratio_by_dataset": {
            dataset_name: count / total_samples if total_samples else 0.0
            for dataset_name, count in sample_count_by_dataset.items()
        },
        "token_count_by_dataset": token_count_by_dataset,
        "token_ratio_by_dataset": {
            dataset_name: count / total_tokens if total_tokens else 0.0
            for dataset_name, count in token_count_by_dataset.items()
        },
        "atom_count_by_dataset": atom_count_by_dataset,
        "atom_ratio_by_dataset": {
            dataset_name: count / total_atoms if total_atoms else 0.0
            for dataset_name, count in atom_count_by_dataset.items()
        },
        "avg_n_token_by_dataset": {
            dataset_name: (
                stats["token_count"] / stats["sample_count"]
                if stats["sample_count"]
                else 0.0
            )
            for dataset_name, stats in sorted(per_dataset.items())
        },
        "avg_n_atom_by_dataset": {
            dataset_name: (
                stats["atom_count"] / stats["sample_count"]
                if stats["sample_count"]
                else 0.0
            )
            for dataset_name, stats in sorted(per_dataset.items())
        },
        "loss_avg_by_dataset": {
            dataset_name: {
                loss_name: sum(values) / len(values)
                for loss_name, values in sorted(dataset_losses.items())
                if values
            }
            for dataset_name, dataset_losses in sorted(loss_values.items())
        },
        "augmentation": build_train_augmentation_window_stats(records),
        "augmentation_by_dataset": {
            dataset_name: build_train_augmentation_window_stats(
                [
                    record
                    for record in records
                    if _dataset_name(record.get("dataset_name")) == dataset_name
                ]
            )
            for dataset_name in sorted(per_dataset)
        },
    }


def build_train_augmentation_window_stats(
    records: list[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate requested and realized occurrence-level data augmentations."""
    decision_records = [
        record
        for record in records
        if record.get("query_only_msa") is not None
        and record.get("template_use_requested") is not None
    ]
    observation_count = len(decision_records)
    query_only_count = sum(
        1 for record in decision_records if _bool_value(record["query_only_msa"])
    )
    template_requested_count = sum(
        1
        for record in decision_records
        if _bool_value(record["template_use_requested"])
    )

    valid_template_records = [
        record
        for record in decision_records
        if record.get("N_template_valid") is not None
    ]
    template_realized_count = sum(
        1
        for record in valid_template_records
        if _int_or_zero(record["N_template_valid"]) > 0
    )
    template_requested_but_unavailable_count = sum(
        1
        for record in valid_template_records
        if _bool_value(record["template_use_requested"])
        and _int_or_zero(record["N_template_valid"]) == 0
    )

    joint_mode_counts = {
        "query_only__template_requested": 0,
        "query_only__no_template": 0,
        "full_msa__template_requested": 0,
        "full_msa__no_template": 0,
    }
    msa_mode_counts = {
        MSA_MODE_QUERY_ONLY: 0,
        MSA_MODE_UNPAIRED_SHUFFLE: 0,
        MSA_MODE_SPECIES_PAIRED: 0,
    }
    msa_template_joint_counts = {
        f"{msa_mode}__{template_mode}": 0
        for msa_mode in msa_mode_counts
        for template_mode in ("template_requested", "no_template")
    }
    for record in decision_records:
        legacy_msa_mode = (
            "query_only"
            if _bool_value(record["query_only_msa"])
            else "full_msa"
        )
        msa_mode = str(record.get("msa_mode") or "")
        if msa_mode not in msa_mode_counts:
            if legacy_msa_mode == "query_only":
                msa_mode = MSA_MODE_QUERY_ONLY
            elif _bool_value(record.get("pair_msa_as_unpaired_only")):
                msa_mode = MSA_MODE_UNPAIRED_SHUFFLE
            else:
                msa_mode = MSA_MODE_SPECIES_PAIRED
        template_mode = (
            "template_requested"
            if _bool_value(record["template_use_requested"])
            else "no_template"
        )
        joint_mode_counts[f"{legacy_msa_mode}__{template_mode}"] += 1
        msa_mode_counts[msa_mode] += 1
        msa_template_joint_counts[f"{msa_mode}__{template_mode}"] += 1

    def ratio(count: int, denominator: int) -> float | None:
        return count / denominator if denominator else None

    return {
        "observation_count": observation_count,
        "query_only_msa_count": query_only_count,
        "query_only_msa_ratio": ratio(query_only_count, observation_count),
        "msa_mode_counts": msa_mode_counts,
        "msa_mode_ratios": {
            name: ratio(count, observation_count)
            for name, count in msa_mode_counts.items()
        },
        "template_requested_count": template_requested_count,
        "template_requested_ratio": ratio(
            template_requested_count,
            observation_count,
        ),
        "template_realized_observation_count": len(valid_template_records),
        "template_realized_count": template_realized_count,
        "template_realized_ratio": ratio(
            template_realized_count,
            len(valid_template_records),
        ),
        "template_requested_but_unavailable_count": (
            template_requested_but_unavailable_count
        ),
        "joint_mode_counts": joint_mode_counts,
        "joint_mode_ratios": {
            name: ratio(count, observation_count)
            for name, count in joint_mode_counts.items()
        },
        "msa_template_joint_counts": msa_template_joint_counts,
        "msa_template_joint_ratios": {
            name: ratio(count, observation_count)
            for name, count in msa_template_joint_counts.items()
        },
    }


class TrainDatasetWindowStats:
    """Collect train data-mix diagnostics over one logging window."""

    def __init__(self, need_gather: bool = True) -> None:
        self.need_gather = need_gather
        self.records: list[dict[str, Any]] = []

    def add(
        self,
        batch_summary: Mapping[str, Any],
        loss_dict: Mapping[str, Any],
    ) -> None:
        losses = {}
        for loss_name, loss_value in loss_dict.items():
            if "loss" not in str(loss_name):
                continue
            loss_value = _finite_scalar_float(loss_value)
            if loss_value is not None:
                losses[str(loss_name)] = loss_value
        self.records.append(
            {
                "dataset_name": _dataset_name(batch_summary.get("dataset_name")),
                "n_token": _int_or_zero(batch_summary.get("n_token")),
                "n_atom": _int_or_zero(batch_summary.get("n_atom")),
                "query_only_msa": _single_value(
                    batch_summary.get("query_only_msa")
                ),
                "msa_mode": _single_value(batch_summary.get("msa_mode")),
                "unpaired_msa": _single_value(
                    batch_summary.get("unpaired_msa")
                ),
                "pair_msa_as_unpaired_only": _single_value(
                    batch_summary.get("pair_msa_as_unpaired_only")
                ),
                "template_use_requested": _single_value(
                    batch_summary.get("template_use_requested")
                ),
                "N_template_valid": _single_value(
                    batch_summary.get("N_template_valid")
                ),
                "losses": losses,
            }
        )

    def calc(self) -> Dict[str, Any]:
        records, self.records = self.records, []
        if self.need_gather:
            gathered_records = DIST_WRAPPER.all_gather_object(records)
            records = [
                record
                for rank_records in gathered_records
                for record in (rank_records or [])
            ]
        return build_train_dataset_window_stats(records)


class TrainRuntimeWindowStats:
    """Collect runtime, memory, and optimizer diagnostics over one logging window."""

    def __init__(self, need_gather: bool = True) -> None:
        self.need_gather = need_gather
        self.records: list[dict[str, Any]] = []

    def add(
        self,
        *,
        rank: int,
        step: int,
        global_step: int,
        batch_summary: Mapping[str, Any],
        train_info: Mapping[str, Any],
        to_device_seconds: Any,
        ema_seconds: Any,
        step_wall_seconds: Any,
        cuda_memory: Mapping[str, Any],
        data_wait_seconds: Any = None,
    ) -> None:
        losses = {}
        for loss_name, loss_value in train_info.get("loss_dict", {}).items():
            if "loss" not in str(loss_name):
                continue
            loss_value = _finite_scalar_float(loss_value)
            if loss_value is not None:
                losses[str(loss_name)] = loss_value
        record = {
            "rank": rank,
            "step": step,
            "global_step": global_step,
            "n_token": _int_or_zero(batch_summary.get("n_token")),
            "n_atom": _int_or_zero(batch_summary.get("n_atom")),
            "N_cycle": _int_or_zero(train_info.get("N_cycle")),
            "losses": losses,
            "step_wall_seconds": _finite_scalar_float(step_wall_seconds),
            "data_wait_seconds": _finite_scalar_float(data_wait_seconds),
            "to_device_seconds": _finite_scalar_float(to_device_seconds),
            "ema_seconds": _finite_scalar_float(ema_seconds),
            "timings": to_jsonable(train_info.get("timings", {})),
            "grad_norm": _scalar_float(train_info.get("grad_norm")),
            "grad_clipped": _bool_value(train_info.get("grad_clipped")),
            "loss_nan": _bool_value(train_info.get("loss_nan")),
            "nonfinite_grad": _bool_value(train_info.get("nonfinite_grad")),
            "optimizer_skipped": _bool_value(train_info.get("optimizer_skipped")),
            "amp_scale": _finite_scalar_float(train_info.get("amp_scale")),
            "post_permutation_diffusion_supervised_tokens": (
                _finite_scalar_float(
                    train_info.get(
                        "post_permutation_diffusion_supervised_tokens"
                    )
                )
            ),
            "chain_permutation_min_supervision_rejected": _bool_value(
                train_info.get(
                    "chain_permutation_min_supervision_rejected"
                )
            ),
            "atom_permutation_min_supervision_rejected": _bool_value(
                train_info.get(
                    "atom_permutation_min_supervision_rejected"
                )
            ),
            "cuda_memory": to_jsonable(cuda_memory),
        }
        for identity_key in (
            "pdb_id",
            "dataset_name",
            "data_id",
            "N_msa",
            "N_msa_prot_pair",
            "N_msa_prot_unpair",
            "N_msa_rna_pair",
            "N_msa_rna_unpair",
            "N_msa_virtual_segment",
            "msa_virtual_segment_shuffle_applied",
            "N_msa_shuffle_group",
            "msa_shuffle_requested_variant_code",
            "msa_shuffle_variant_eligible",
            "msa_shuffle_variant_applied",
            "msa_shuffle_whole_physical_chain_applied",
            "N_msa_crop_protein_chain",
            "N_msa_query_only_protein_chain",
            "N_msa_retained_protein_chain",
            "msa_partial_query_only_applied",
            "msa_global_query_only_applied",
            "N_openfold3_representative_query_mismatch",
            "N_openfold3_representative_remap",
            "N_openfold3_representative_query_only_fallback",
            "N_openfold3_explicit_query_only",
            "N_prot_token",
            "N_lig_token",
            "N_rna_token",
            "N_dna_token",
            "N_diffusion_supervised_token",
            "msa_mode",
            "query_only_msa",
            "unpaired_msa",
            "N_template_valid",
            "template_use_requested",
            *SEQUENCE_TEMPLATE_CACHE_CHAIN_DUMMY_METRIC_KEYS,
            "chain_id",
            "sample_sampler_epoch",
            "sample_global_position",
        ):
            identity_value = _single_value(batch_summary.get(identity_key))
            if identity_value is not None:
                record[identity_key] = to_jsonable(identity_value)
        for timing_key in DATA_PIPELINE_TIMING_KEYS:
            timing_value = _finite_scalar_float(
                _single_value(batch_summary.get(timing_key))
            )
            if timing_value is not None:
                record[timing_key] = timing_value
        sample_global_position = record.get("sample_global_position")
        record["sampler_epoch_boundary"] = (
            sample_global_position is not None
            and int(sample_global_position) == int(rank)
        )
        self.records.append(record)

    def calc(self) -> Dict[str, Any]:
        records, self.records = self.records, []
        if self.need_gather:
            gathered_records = DIST_WRAPPER.all_gather_object(records)
            records = [
                record
                for rank_records in gathered_records
                for record in (rank_records or [])
            ]
        return build_train_runtime_window_stats(records)


def extract_eval_key_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract common scalar metrics used for quick eval triage."""
    candidates = {
        "loss": ["loss"],
        "lddt_complex_mean": ["lddt/complex/mean"],
        "ranking_score": ["lddt/complex/ranking_score"],
        "plddt": ["lddt/complex/plddt"],
        "gpde": ["lddt/complex/gpde"],
    }
    key_metrics = {}
    for output_key, metric_keys in candidates.items():
        for metric_key in metric_keys:
            if metric_key in metrics:
                key_metrics[output_key] = to_jsonable(metrics[metric_key])
                break
            prefix = f"{metric_key}."
            suffixed_key = next(
                (key for key in metrics.keys() if str(key).startswith(prefix)),
                None,
            )
            if suffixed_key is not None:
                key_metrics[output_key] = to_jsonable(metrics[suffixed_key])
                break
    return key_metrics


def build_eval_sample_record(
    test_name: str,
    ema_suffix: str,
    step: int,
    global_step: int,
    rank: int,
    world_size: int,
    batch_index: int,
    total_batches: int,
    sample: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build one per-sample evaluation record."""
    return {
        "event": "eval_sample",
        "test_name": test_name,
        "ema_suffix": ema_suffix,
        "is_ema": bool(ema_suffix),
        "step": step,
        "global_step": global_step,
        "rank": rank,
        "world_size": world_size,
        "batch_index": batch_index,
        "total_batches": total_batches,
        "sample": sample,
        "key_metrics": extract_eval_key_metrics(metrics),
        "metrics": metrics,
    }


def _metric_for_sort(record: Mapping[str, Any], metric_name: str, default: float) -> float:
    value = _get_value(_get_value(record, "key_metrics", {}), metric_name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compact_eval_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "rank": _get_value(record, "rank"),
        "batch_index": _get_value(record, "batch_index"),
        "sample": _get_value(record, "sample", {}),
        "key_metrics": _get_value(record, "key_metrics", {}),
    }


def select_worst_eval_samples(
    records: list[Mapping[str, Any]],
    n: int,
) -> Dict[str, Any]:
    """Select samples that are most useful to inspect after evaluation."""
    if n <= 0:
        return {
            "highest_loss": [],
            "lowest_lddt_complex_mean": [],
            "lowest_ranking_score": [],
        }
    records_with_loss = [
        record for record in records if "loss" in _get_value(record, "key_metrics", {})
    ]
    records_with_lddt = [
        record
        for record in records
        if "lddt_complex_mean" in _get_value(record, "key_metrics", {})
    ]
    records_with_ranking = [
        record
        for record in records
        if "ranking_score" in _get_value(record, "key_metrics", {})
    ]
    return {
        "highest_loss": [
            _compact_eval_record(record)
            for record in sorted(
                records_with_loss,
                key=lambda record: _metric_for_sort(record, "loss", float("-inf")),
                reverse=True,
            )[:n]
        ],
        "lowest_lddt_complex_mean": [
            _compact_eval_record(record)
            for record in sorted(
                records_with_lddt,
                key=lambda record: _metric_for_sort(
                    record, "lddt_complex_mean", float("inf")
                ),
            )[:n]
        ],
        "lowest_ranking_score": [
            _compact_eval_record(record)
            for record in sorted(
                records_with_ranking,
                key=lambda record: _metric_for_sort(
                    record, "ranking_score", float("inf")
                ),
            )[:n]
        ],
    }


def get_cuda_memory_stats(device: torch.device | None = None) -> Dict[str, Any]:
    """Return lightweight CUDA memory stats for diagnostics."""
    if not torch.cuda.is_available():
        return {}
    device = device or torch.device("cuda")
    stats = {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except (RuntimeError, TypeError, ValueError):
        return stats
    stats["free_bytes"] = free_bytes
    stats["total_bytes"] = total_bytes
    return stats


class CudaPhaseMemoryRecorder:
    """Record synchronized, rank-local CUDA peaks for semantic train phases.

    This diagnostic deliberately synchronizes and resets allocator peak counters
    at phase boundaries.  It is therefore disabled by default and its timing is
    not comparable with a normal performance benchmark.
    """

    def __init__(self, *, enabled: bool, device: torch.device) -> None:
        self.enabled = bool(enabled and torch.cuda.is_available())
        self.device = device
        self.window_active = False
        self.current_step: Dict[str, Any] | None = None
        self.current_phase: Dict[str, Any] | None = None
        self.steps: list[Dict[str, Any]] = []
        self.overall_max_allocated_bytes = 0
        self.overall_max_reserved_bytes = 0

    @property
    def recording_step(self) -> bool:
        return self.enabled and self.window_active and self.current_step is not None

    def start_window(self) -> None:
        if not self.enabled:
            return
        if self.window_active:
            raise RuntimeError("CUDA phase recorder window is already active")
        self.window_active = True
        self.current_step = None
        self.current_phase = None
        self.steps = []
        self.overall_max_allocated_bytes = 0
        self.overall_max_reserved_bytes = 0

    def start_step(self, *, step: int, global_step: int, is_update_step: bool) -> None:
        if not self.enabled or not self.window_active:
            return
        if self.current_step is not None:
            raise RuntimeError("CUDA phase recorder step was not closed")
        self.current_step = {
            "step": int(step),
            "global_step": int(global_step),
            "is_update_step": bool(is_update_step),
            "sample": {},
            "phases": [],
        }
        # Preserve any peak reached since the previous measured step ended
        # before resetting the allocator counters for this step.  The benchmark
        # window includes inter-step work, so dropping this peak would make the
        # software-reconstructed whole-window maximum too small.
        self._start_phase(
            "to_device",
            synchronize=True,
            preserve_existing_peak=True,
        )

    def set_sample(self, sample: Mapping[str, Any]) -> None:
        if self.recording_step:
            self.current_step["sample"] = to_jsonable(dict(sample))

    def transition(self, phase_name: str) -> None:
        if not self.recording_step:
            return
        if not str(phase_name):
            raise ValueError("CUDA phase name must not be empty")
        self._close_phase()
        self._start_phase(
            phase_name,
            synchronize=False,
            preserve_existing_peak=False,
        )

    def end_step(self) -> None:
        if not self.recording_step:
            return
        self._close_phase()
        self.steps.append(self.current_step)
        self.current_step = None
        self.current_phase = None

    def finish_window(self) -> None:
        if not self.enabled or not self.window_active:
            return
        if self.recording_step:
            self.end_step()
        else:
            # Capture work between the final closed step and the window
            # boundary.  Normally this gap is tiny, but retaining it preserves
            # the original whole-window peak semantics.
            torch.cuda.synchronize(self.device)
            self._update_overall_peak(self._snapshot())
        self.window_active = False

    def _snapshot(self) -> Dict[str, int]:
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
        }

    def _update_overall_peak(self, snapshot: Mapping[str, Any]) -> None:
        self.overall_max_allocated_bytes = max(
            self.overall_max_allocated_bytes,
            int(snapshot["max_allocated_bytes"]),
        )
        self.overall_max_reserved_bytes = max(
            self.overall_max_reserved_bytes,
            int(snapshot["max_reserved_bytes"]),
        )

    def _start_phase(
        self,
        phase_name: str,
        *,
        synchronize: bool,
        preserve_existing_peak: bool,
    ) -> None:
        if synchronize:
            torch.cuda.synchronize(self.device)
        if preserve_existing_peak:
            self._update_overall_peak(self._snapshot())
        torch.cuda.reset_peak_memory_stats(self.device)
        start = self._snapshot()
        self.current_phase = {
            "name": str(phase_name),
            "started_at_perf_counter": time.perf_counter(),
            "start_allocated_bytes": start["allocated_bytes"],
            "start_reserved_bytes": start["reserved_bytes"],
        }

    def _close_phase(self) -> None:
        if self.current_phase is None:
            return
        torch.cuda.synchronize(self.device)
        end = self._snapshot()
        peak_allocated = end["max_allocated_bytes"]
        peak_reserved = end["max_reserved_bytes"]
        self._update_overall_peak(end)
        record = {
            "name": self.current_phase["name"],
            "duration_seconds": (
                time.perf_counter()
                - float(self.current_phase["started_at_perf_counter"])
            ),
            "start_allocated_bytes": self.current_phase[
                "start_allocated_bytes"
            ],
            "start_reserved_bytes": self.current_phase["start_reserved_bytes"],
            "end_allocated_bytes": end["allocated_bytes"],
            "end_reserved_bytes": end["reserved_bytes"],
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_increase_bytes": max(
                0,
                peak_allocated
                - int(self.current_phase["start_allocated_bytes"]),
            ),
        }
        self.current_step["phases"].append(record)
        self.current_phase = None

    def result(self) -> Dict[str, Any] | None:
        if not self.enabled:
            return None
        return {
            "schema_version": 1,
            "mode": "synchronized_phase_peaks",
            "throughput_comparable": False,
            "overall_max_allocated_bytes": self.overall_max_allocated_bytes,
            "overall_max_reserved_bytes": self.overall_max_reserved_bytes,
            "steps": self.steps,
        }


def build_phase_cuda_memory_summary(
    rank_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any] | None:
    """Aggregate optional phase-memory payloads without phase collectives."""
    payloads = [
        (int(record.get("rank", -1)), record.get("phase_cuda_memory"))
        for record in rank_records
        if isinstance(record.get("phase_cuda_memory"), Mapping)
    ]
    if not payloads:
        return None
    overall_max_allocated_bytes = 0
    overall_max_reserved_bytes = 0
    overall_max_allocated_rank = None
    aggregate: Dict[str, Dict[str, Any]] = {}
    phase_order: list[str] = []
    for rank, payload in payloads:
        rank_max_allocated = int(payload.get("overall_max_allocated_bytes", 0))
        if rank_max_allocated >= overall_max_allocated_bytes:
            overall_max_allocated_bytes = rank_max_allocated
            overall_max_allocated_rank = rank
        overall_max_reserved_bytes = max(
            overall_max_reserved_bytes,
            int(payload.get("overall_max_reserved_bytes", 0)),
        )
        for step_record in payload.get("steps", []):
            step = int(step_record.get("step", -1))
            for phase in step_record.get("phases", []):
                name = str(phase.get("name", ""))
                if not name:
                    continue
                if name not in aggregate:
                    phase_order.append(name)
                    aggregate[name] = {
                        "observation_count": 0,
                        "max_peak_allocated_bytes": 0,
                        "max_peak_reserved_bytes": 0,
                        "max_end_allocated_bytes": 0,
                        "max_peak_increase_bytes": 0,
                        "max_rank": None,
                        "max_step": None,
                    }
                item = aggregate[name]
                item["observation_count"] += 1
                peak_allocated = int(phase.get("peak_allocated_bytes", 0))
                if peak_allocated >= item["max_peak_allocated_bytes"]:
                    item["max_peak_allocated_bytes"] = peak_allocated
                    item["max_rank"] = rank
                    item["max_step"] = step
                item["max_peak_reserved_bytes"] = max(
                    item["max_peak_reserved_bytes"],
                    int(phase.get("peak_reserved_bytes", 0)),
                )
                item["max_end_allocated_bytes"] = max(
                    item["max_end_allocated_bytes"],
                    int(phase.get("end_allocated_bytes", 0)),
                )
                item["max_peak_increase_bytes"] = max(
                    item["max_peak_increase_bytes"],
                    int(phase.get("peak_increase_bytes", 0)),
                )
    return {
        "schema_version": 1,
        "mode": "synchronized_phase_peaks",
        "throughput_comparable": False,
        "rank_record_count": len(payloads),
        "overall_max_allocated_bytes": overall_max_allocated_bytes,
        "overall_max_reserved_bytes": overall_max_reserved_bytes,
        "overall_max_allocated_rank": overall_max_allocated_rank,
        "phase_order": phase_order,
        "aggregate_by_phase": aggregate,
        "rank_records": [
            {"rank": rank, **dict(payload)} for rank, payload in payloads
        ],
    }


class AF3Trainer(object):
    """
    Trainer class for the Alphafold3 model.

    Args:
        configs (Any): Configuration object for training.
    """

    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_log()
        self.init_model()
        self.init_loss()
        self.init_data()
        self.try_load_checkpoint()
        self.update_run_status(
            status="running",
            exit_reason="initialized",
            write_heartbeat=True,
        )

    def init_basics(self) -> None:
        """
        Initialize basic training parameters and directory structure.
        """
        # Step means effective step considering accumulation
        self.step = 0
        # Global_step equals to self.step * self.iters_to_accumulate
        self.global_step = 0
        self.start_step = 0
        # If any microbatch in an accumulation window is non-finite, all ranks
        # finish backward but discard that whole optimizer update.
        self._skip_accumulated_update = False
        # Add for grad accumulation, it can increase real batch size
        self.iters_to_accumulate = self.configs.iters_to_accumulate
        self.benchmark_warmup_steps = int(
            self.configs.get("benchmark_warmup_steps", -1)
        )
        self.benchmark_measure_steps = int(
            self.configs.get("benchmark_measure_steps", -1)
        )
        self.benchmark_enabled = validate_benchmark_window(
            self.benchmark_warmup_steps,
            self.benchmark_measure_steps,
            int(self.configs.max_steps),
        )
        phase_memory_value = os.environ.get(
            "PROTENIX_BENCHMARK_PHASE_MEMORY", "false"
        ).strip().lower()
        if phase_memory_value not in {"true", "false"}:
            raise ValueError(
                "PROTENIX_BENCHMARK_PHASE_MEMORY must be true or false"
            )
        if phase_memory_value == "true" and not self.benchmark_enabled:
            raise ValueError(
                "CUDA phase memory diagnostics require a benchmark window"
            )
        self.benchmark_phase_memory = CudaPhaseMemoryRecorder(
            enabled=phase_memory_value == "true",
            device=self.device,
        )
        self.benchmark_window_active = False
        self.benchmark_window_completed = False
        self.benchmark_window_started_at = None
        self.benchmark_window_started_epoch = None
        self.benchmark_window_start_time = None
        self.benchmark_local_sample_count = 0
        self.benchmark_local_token_count = 0
        self.benchmark_local_atom_count = 0
        self.benchmark_local_min_token_count = None
        self.benchmark_local_max_token_count = None
        self.benchmark_workload_digest = hashlib.sha256()
        self.benchmark_run_sample_records: list[dict[str, Any]] = []
        self.benchmark_run_msa_depth_records: list[dict[str, Any]] = []
        self.benchmark_local_loss_nonfinite_count = 0
        self.benchmark_local_gradient_nonfinite_count = 0
        self.benchmark_local_optimizer_skipped_count = 0

        self.run_name = self.configs.run_name + "_" + time.strftime("%Y%m%d_%H%M%S")
        run_names = DIST_WRAPPER.all_gather_object(
            self.run_name if DIST_WRAPPER.rank == 0 else None
        )
        self.run_name = [name for name in run_names if name is not None][0]
        self.run_dir = f"{self.configs.base_dir}/{self.run_name}"
        self.checkpoint_dir = f"{self.run_dir}/checkpoints"
        self.prediction_dir = f"{self.run_dir}/predictions"
        self.structure_dir = f"{self.run_dir}/structures"
        self.dump_dir = f"{self.run_dir}/dumps"
        self.error_dir = f"{self.run_dir}/errors"
        self.log_dir = f"{self.run_dir}/logs"
        self.manifest_dir = f"{self.run_dir}/manifest"
        self.numeric_error_dir = f"{self.error_dir}/numerics"
        self.benchmark_dir = f"{self.run_dir}/benchmark"
        self.benchmark_result_path = f"{self.benchmark_dir}/benchmark_result.json"
        self.train_metrics_log_path = f"{self.log_dir}/train_metrics.jsonl"
        self.eval_metrics_log_path = f"{self.log_dir}/eval_metrics.jsonl"
        self.heartbeat_log_path = f"{self.log_dir}/heartbeat_rank0.jsonl"
        self.runtime_log_path = f"{self.log_dir}/runtime_rank{DIST_WRAPPER.rank}.jsonl"
        self.data_samples_log_path = (
            f"{self.log_dir}/data_samples_rank{DIST_WRAPPER.rank}.jsonl"
        )
        self.eval_samples_log_path = (
            f"{self.log_dir}/eval_samples_rank{DIST_WRAPPER.rank}.jsonl"
        )
        self.eval_worst_samples_log_path = f"{self.log_dir}/eval_worst_samples.jsonl"
        self.numeric_error_log_path = (
            f"{self.numeric_error_dir}/numerics_rank{DIST_WRAPPER.rank}.jsonl"
        )
        self.checkpoint_index_path = f"{self.checkpoint_dir}/checkpoints.jsonl"
        self.run_manifest_path = f"{self.manifest_dir}/run_manifest.json"
        self.run_status_path = f"{self.manifest_dir}/run_status.json"
        self.run_started_at = _now_timestamp()
        self.latest_checkpoint_path = None
        self.latest_ema_checkpoint_path = None

        for directory in [
            self.run_dir,
            self.checkpoint_dir,
            self.prediction_dir,
            self.structure_dir,
            self.dump_dir,
            self.error_dir,
            self.log_dir,
            self.manifest_dir,
            self.numeric_error_dir,
            self.benchmark_dir,
        ]:
            os.makedirs(directory, exist_ok=True)

        if DIST_WRAPPER.rank == 0:
            save_config(
                self.configs,
                os.path.join(self.configs.base_dir, self.run_name, "config.yaml"),
            )
            write_json(
                self.run_manifest_path,
                build_run_manifest(self.configs, self.run_name, self.run_dir),
            )
            self.update_run_status(
                status="running",
                exit_reason="initializing",
                write_heartbeat=True,
            )

        self.print(
            f"Using run name: {self.run_name}, run dir: {self.run_dir}, "
            f"checkpoint_dir: {self.checkpoint_dir}, "
            f"prediction_dir: {self.prediction_dir}, "
            f"structure_dir: {self.structure_dir}, "
            f"log_dir: {self.log_dir}, "
            f"error_dir: {self.error_dir}"
        )
        if self.benchmark_enabled:
            self.print(
                "Synchronized benchmark window enabled: "
                f"warmup_steps={self.benchmark_warmup_steps}, "
                f"measure_steps={self.benchmark_measure_steps}, "
                f"result={self.benchmark_result_path}"
            )
            if self.benchmark_phase_memory.enabled:
                self.print(
                    "Synchronized CUDA phase-memory diagnostics enabled; "
                    "throughput from this run is not comparable"
                )

    def init_log(self) -> None:
        """
        Initialize logging and metrics aggregation.
        """
        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
            wandb.init(
                project=self.configs.project,
                name=self.run_name,
                config=vars(self.configs),
                id=self.configs.wandb_id or None,
            )
            define_training_metrics(wandb.run)
        self.train_metric_wrapper = SimpleMetricAggregator(["avg"])
        self.train_dataset_window_stats = TrainDatasetWindowStats()
        self.train_runtime_window_stats = TrainRuntimeWindowStats()
        self.msa_debug_log_limit = _env_int("PROTENIX_MSA_DEBUG_LOG_LIMIT", 0)
        self.msa_debug_log_count = 0

    def update_run_status(
        self,
        status: str,
        exit_reason: str | None = None,
        exception: BaseException | None = None,
        write_heartbeat: bool = False,
    ) -> None:
        """Update rank0 lifecycle status and optionally append heartbeat."""
        if DIST_WRAPPER.rank != 0:
            return
        record = build_run_status_record(
            status=status,
            run_name=self.run_name,
            run_dir=self.run_dir,
            step=(
                int(os.environ.get("RESUME_STEP", self.step))
                if exit_reason == "initializing"
                else self.step
            ),
            global_step=self.global_step,
            max_steps=self.configs.max_steps,
            latest_checkpoint=self.latest_checkpoint_path,
            latest_ema_checkpoint=self.latest_ema_checkpoint_path,
            exit_reason=exit_reason,
            exception=exception,
            started_at=self.run_started_at,
        )
        write_run_status(self.run_status_path, record)
        if write_heartbeat:
            append_run_heartbeat(self.heartbeat_log_path, record)

    def maybe_start_benchmark_window(self) -> None:
        """Start measurement on an optimizer-step boundary after warm-up."""
        if (
            not self.benchmark_enabled
            or self.benchmark_window_active
            or self.benchmark_window_completed
            or self.step != self.benchmark_warmup_steps
            or self.global_step % self.iters_to_accumulate != 0
        ):
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        self.benchmark_window_started_at = _now_timestamp()
        self.benchmark_window_started_epoch = time.time()
        self.benchmark_window_start_time = time.perf_counter()
        self.benchmark_window_active = True
        self.benchmark_phase_memory.start_window()
        self.benchmark_local_sample_count = 0
        self.benchmark_local_token_count = 0
        self.benchmark_local_atom_count = 0
        self.benchmark_local_min_token_count = None
        self.benchmark_local_max_token_count = None
        self.benchmark_workload_digest = hashlib.sha256()
        self.benchmark_local_loss_nonfinite_count = 0
        self.benchmark_local_gradient_nonfinite_count = 0
        self.benchmark_local_optimizer_skipped_count = 0
        self.print(
            f"Benchmark measurement window started at training step {self.step}"
        )

    def record_benchmark_batch(
        self,
        batch_summary: Mapping[str, Any],
        train_info: Mapping[str, Any],
    ) -> None:
        """Accumulate local work and numerical-health counters in the window."""
        if not self.benchmark_window_active:
            return
        self.benchmark_local_sample_count += 1
        batch_token_count = _int_or_zero(batch_summary.get("n_token"))
        self.benchmark_local_token_count += batch_token_count
        self.benchmark_local_atom_count += _int_or_zero(batch_summary.get("n_atom"))
        if self.benchmark_local_min_token_count is None:
            self.benchmark_local_min_token_count = batch_token_count
            self.benchmark_local_max_token_count = batch_token_count
        else:
            self.benchmark_local_min_token_count = min(
                self.benchmark_local_min_token_count,
                batch_token_count,
            )
            self.benchmark_local_max_token_count = max(
                self.benchmark_local_max_token_count,
                batch_token_count,
            )
        # The ordered digest proves that a paired DDP/FSDP2 run consumed the
        # same samples and crop/MSA metadata on every rank. Aggregate token or
        # atom counts alone cannot detect a different sample order.
        workload_summary = dict(batch_summary)
        workload_summary["msa_depth_trace"] = to_jsonable(
            train_info.get("msa_depth_trace", [])
        )
        workload_record = json.dumps(
            to_jsonable(workload_summary),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.benchmark_workload_digest.update(workload_record)
        self.benchmark_workload_digest.update(b"\n")
        self.benchmark_local_loss_nonfinite_count += int(
            _bool_value(train_info.get("loss_nan"))
        )
        self.benchmark_local_gradient_nonfinite_count += int(
            _bool_value(train_info.get("nonfinite_grad"))
        )
        self.benchmark_local_optimizer_skipped_count += int(
            _bool_value(train_info.get("optimizer_skipped"))
        )

    def record_benchmark_run_sample(
        self,
        batch_summary: Mapping[str, Any],
    ) -> None:
        """Track all warm-up and measured source samples without per-step I/O."""
        if not self.benchmark_enabled or self.benchmark_window_completed:
            return
        end_step = self.benchmark_warmup_steps + self.benchmark_measure_steps
        if self.step >= end_step:
            return
        identity = benchmark_sample_identity(batch_summary)
        token_composition = {
            key: _int_or_zero(batch_summary.get(key))
            for key in BENCHMARK_TOKEN_COMPOSITION_KEYS
            if batch_summary.get(key) is not None
        }
        self.benchmark_run_sample_records.append(
            {
                "rank": DIST_WRAPPER.rank,
                "step": self.step,
                "identity": identity,
                "identity_complete": bool(identity.get("dataset_name"))
                and bool(identity.get("pdb_id")),
                "token_composition": token_composition,
            }
        )

    def record_benchmark_msa_depth_trace(
        self,
        trace: Sequence[Mapping[str, Any]],
    ) -> None:
        """Track every MSA-module invocation over warm-up and measurement."""
        if not self.benchmark_enabled or self.benchmark_window_completed:
            return
        end_step = self.benchmark_warmup_steps + self.benchmark_measure_steps
        if self.step >= end_step:
            return
        self.benchmark_run_msa_depth_records.append(
            {
                "rank": DIST_WRAPPER.rank,
                "step": self.step,
                "calls": [
                    {
                        "input_depth": call.get("input_depth"),
                        "sampled_depth": call.get("sampled_depth"),
                    }
                    for call in (trace or [])
                    if isinstance(call, Mapping)
                ],
            }
        )

    def maybe_finish_benchmark_window(self, *, is_update_step: bool) -> None:
        """Synchronize, aggregate, and persist the completed measurement window."""
        end_step = self.benchmark_warmup_steps + self.benchmark_measure_steps
        if (
            not self.benchmark_window_active
            or not is_update_step
            or self.step + 1 != end_step
        ):
            return

        self.benchmark_phase_memory.finish_window()
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        elapsed_seconds = time.perf_counter() - self.benchmark_window_start_time
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        completed_epoch = time.time()
        phase_memory_result = self.benchmark_phase_memory.result()
        observed_max_allocated = (
            torch.cuda.max_memory_allocated(self.device)
            if torch.cuda.is_available()
            else 0
        )
        observed_max_reserved = (
            torch.cuda.max_memory_reserved(self.device)
            if torch.cuda.is_available()
            else 0
        )
        if phase_memory_result is not None:
            observed_max_allocated = max(
                observed_max_allocated,
                int(phase_memory_result["overall_max_allocated_bytes"]),
            )
            observed_max_reserved = max(
                observed_max_reserved,
                int(phase_memory_result["overall_max_reserved_bytes"]),
            )
        local_record = {
            "rank": DIST_WRAPPER.rank,
            "local_rank": DIST_WRAPPER.local_rank,
            "node_rank": DIST_WRAPPER.node_rank,
            "hostname": socket.gethostname(),
            "elapsed_seconds": elapsed_seconds,
            "sample_count": self.benchmark_local_sample_count,
            "token_count": self.benchmark_local_token_count,
            "atom_count": self.benchmark_local_atom_count,
            "min_token_count": self.benchmark_local_min_token_count,
            "max_token_count": self.benchmark_local_max_token_count,
            "workload_sha256": self.benchmark_workload_digest.hexdigest(),
            "loss_nonfinite_count": self.benchmark_local_loss_nonfinite_count,
            "gradient_nonfinite_count": (
                self.benchmark_local_gradient_nonfinite_count
            ),
            "optimizer_skipped_count": (
                self.benchmark_local_optimizer_skipped_count
            ),
            "max_allocated_bytes": observed_max_allocated,
            "max_reserved_bytes": observed_max_reserved,
            "phase_cuda_memory": phase_memory_result,
            "gpu_name": (
                torch.cuda.get_device_name(self.device)
                if torch.cuda.is_available()
                else None
            ),
        }
        rank_records = DIST_WRAPPER.all_gather_object(local_record)
        gathered_sample_records = DIST_WRAPPER.all_gather_object(
            self.benchmark_run_sample_records
        )
        run_sample_coverage = build_benchmark_sample_coverage(
            gathered_sample_records
        )
        benchmark_dataset_name = os.environ.get("BENCHMARK_DATASET_NAME", "")
        benchmark_dataset_cfg = _get_value(
            self.configs.data, benchmark_dataset_name, {}
        )
        benchmark_crop_cfg = _get_value(
            benchmark_dataset_cfg, "cropping_configs", {}
        )
        benchmark_max_nuc_tokens = _int_or_zero(
            _get_value(benchmark_crop_cfg, "max_nuc_tokens", -1)
        )
        token_composition_coverage = build_benchmark_token_composition_coverage(
            gathered_sample_records,
            max_nuc_tokens=benchmark_max_nuc_tokens,
        )
        gathered_msa_depth_records = DIST_WRAPPER.all_gather_object(
            self.benchmark_run_msa_depth_records
        )
        msa_target_depth = _int_or_zero(
            _get_value(
                _get_value(self.configs.data.msa, "sample_cutoff", {}),
                "train",
                0,
            )
        )
        msa_depth_coverage = build_benchmark_msa_depth_coverage(
            gathered_msa_depth_records,
            target_depth=msa_target_depth,
        )
        result = build_benchmark_window_result(
            rank_records,
            warmup_steps=self.benchmark_warmup_steps,
            measure_steps=self.benchmark_measure_steps,
        )
        phase_cuda_memory = build_phase_cuda_memory_summary(rank_records)
        if phase_cuda_memory is not None:
            result["phase_cuda_memory"] = phase_cuda_memory
        result.update(
            {
                "run_name": self.run_name,
                "run_dir": self.run_dir,
                "started_at": self.benchmark_window_started_at,
                "started_at_epoch": self.benchmark_window_started_epoch,
                "completed_at": _now_timestamp(),
                "completed_at_epoch": completed_epoch,
                "scope": (
                    "steady-state training window after DataLoader yielded the "
                    "first measured batch; includes device transfer, forward, "
                    "loss, backward, optimizer, EMA, and later inter-step waits"
                ),
                "benchmark_variant": os.environ.get("BENCHMARK_VARIANT", ""),
                "benchmark_dataset_name": benchmark_dataset_name,
                "benchmark_workload_profile": os.environ.get(
                    "BENCHMARK_WORKLOAD_PROFILE", ""
                ),
                "train_crop_size": int(self.configs.train_crop_size),
                "distributed_strategy": self.configs.distributed_strategy,
                "optimizer_sharding": self.configs.optimizer_sharding,
                "fsdp": to_jsonable(self.configs.fsdp),
                "resolved_fsdp_mesh": (
                    str(self.fsdp_mesh) if self.use_fsdp else None
                ),
                "ema": {
                    "enabled": hasattr(self, "ema_wrapper"),
                    "decay": self.configs.ema_decay,
                    "shadow_device": self.configs.ema_shadow_device,
                },
                "model_parameters": self.model_parameter_counts,
                "skip_final_checkpoint": bool(
                    self.configs.skip_final_checkpoint
                ),
                "load_params_only": bool(self.configs.load_params_only),
                "initial_checkpoint": {
                    "model": self.configs.load_checkpoint_path,
                    "ema": self.configs.load_ema_checkpoint_path,
                },
                "topology": {
                    "world_size": DIST_WRAPPER.world_size,
                    "local_world_size": DIST_WRAPPER.local_world_size,
                    "num_nodes": DIST_WRAPPER.num_nodes,
                },
                "iters_to_accumulate": self.iters_to_accumulate,
                "train_sampler_replacement": _get_value(
                    _get_value(self.configs.data, "train_sampler", {}),
                    "replacement",
                    True,
                ),
                "train_sampler_seed": _get_value(
                    _get_value(self.configs.data, "train_sampler", {}),
                    "seed",
                    -1,
                ),
                "run_sample_coverage": run_sample_coverage,
                "token_composition_coverage": token_composition_coverage,
                "msa_depth_coverage": msa_depth_coverage,
            }
        )
        if DIST_WRAPPER.rank == 0:
            write_json(self.benchmark_result_path, result)
        self.benchmark_window_active = False
        self.benchmark_window_completed = True
        self.print(
            "Benchmark measurement window completed: "
            f"seconds_per_step={result['seconds_per_step']:.3f}, "
            f"tokens_per_sec={result['tokens_per_sec']:.3f}, "
            f"max_allocated_bytes={result['max_allocated_bytes']}"
        )

    def init_env(self) -> None:
        """
        Initialize the PyTorch and CUDA environment for training.
        Sets up distributed training if world_size > 1 and sets random seeds.
        """
        validate_expected_distributed_topology(
            world_size=DIST_WRAPPER.world_size,
            local_world_size=DIST_WRAPPER.local_world_size,
            num_nodes=DIST_WRAPPER.num_nodes,
        )
        logging.info(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device(f"cuda:{DIST_WRAPPER.local_rank}")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if DIST_WRAPPER.world_size > 1:
            timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
            dist.init_process_group(
                backend="nccl", timeout=datetime.timedelta(seconds=timeout_seconds)
            )

        if not self.configs.deterministic_seed:
            # use rank-specific seed
            hash_string = f"({self.configs.seed},{DIST_WRAPPER.rank},init_seed)"
            rank_seed = int(hashlib.sha256(hash_string.encode("utf8")).hexdigest(), 16)
            rank_seed = rank_seed % (2**32)
        else:
            rank_seed = self.configs.seed

        seed_everything(
            seed=rank_seed,
            deterministic=self.configs.deterministic,
        )  # Different DDP processes get different seeds

        if self.configs.triangle_attention == "deepspeed":
            env = os.getenv("CUTLASS_PATH", None)
            print(f"env: {env}")
            assert env is not None, (
                "If use deepspeed (ds4sci), set CUTLASS_PATH env as per "
                "https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            )
        logging.info("Finished environment initialization.")

    def init_loss(self) -> None:
        """
        Initialize loss functions, metrics, and permutation utilities.
        """
        self.loss = ProtenixLoss(self.configs)
        self.symmetric_permutation = SymmetricPermutation(
            self.configs, error_dir=self.error_dir
        )
        self.lddt_metrics = LDDTMetrics(self.configs)

    def init_model(self) -> None:
        """
        Initialize the Protenix model, optimizer, and exponential moving average (EMA).
        Sets up DDP or FSDP2 if multiple GPUs are used.
        """
        self.raw_model = Protenix(self.configs).to(self.device)
        strategy = validate_distributed_strategy(
            self.configs,
            world_size=DIST_WRAPPER.world_size,
        )
        self.use_ddp = False
        self.use_fsdp = False
        self._ddp_first_backward_sync_pending = False
        if strategy == "fsdp2":
            self.print("Using FSDP2 fully_shard")
            self.use_fsdp = True
            # DDP synchronizes rank-0 initialization in its constructor;
            # composable FSDP2 does not, so do it before creating shards.
            sync_module_states(self.raw_model)
            self.fsdp_mesh = build_fsdp_device_mesh(self.configs.fsdp.mesh)
            shard_modules = get_fsdp2_shard_modules(self.raw_model)
            sharded_params = sum(
                parameter.numel()
                for module in shard_modules
                for parameter in module.parameters()
            )
            self.print(
                f"FSDP2 mesh: {self.fsdp_mesh}; child units: "
                f"{len(shard_modules)}; child parameters: "
                f"{sharded_params / 1e6:.2f}M"
            )
            # Wrap children bottom-up and the root last. The optimizer must be
            # constructed only after fully_shard has replaced the parameters.
            self.model = apply_fsdp2(
                self.raw_model,
                mesh=self.fsdp_mesh,
                param_dtype=self.configs.fsdp.param_dtype,
                reduce_dtype=self.configs.fsdp.reduce_dtype,
                reshard_after_forward=self.configs.fsdp.reshard_after_forward,
            )
        elif DIST_WRAPPER.world_size > 1:
            self.print("Using DistributedDataParallel (DDP)")
            self.use_ddp = True
            # Fix DDP/checkpoint compatibility:
            # https://discuss.pytorch.org/t/ddp-and-gradient-checkpointing/132244
            ddp_kwargs = build_ddp_kwargs(
                self.configs,
                local_rank=DIST_WRAPPER.local_rank,
            )
            self.print(
                "DDP gradient_as_bucket_view: "
                f"{ddp_kwargs['gradient_as_bucket_view']}"
            )
            self.model = DDP(self.raw_model, **ddp_kwargs)
            self._ddp_first_backward_sync_pending = bool(
                ddp_kwargs["static_graph"]
            )
            if (
                self._ddp_first_backward_sync_pending
                and self.iters_to_accumulate > 1
            ):
                self.print(
                    "DDP static-graph initialization: forcing synchronization "
                    "for the first backward before no_sync accumulation"
                )
        else:
            self.model = self.raw_model

        def count_parameters(model: torch.nn.Module) -> float:
            """Count total parameters in millions."""
            total_params = sum(p.numel() for p in model.parameters())
            return total_params / 1e6

        total_parameters = sum(p.numel() for p in self.model.parameters())
        trainable_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        fixed_parameters = {
            name.removeprefix("module."): parameter.numel()
            for name, parameter in self.model.named_parameters()
            if not parameter.requires_grad
        }
        self.model_parameter_counts = {
            "total": total_parameters,
            "trainable": trainable_parameters,
            "frozen": total_parameters - trainable_parameters,
            "fixed_parameters": fixed_parameters,
        }
        self.print(f"Model Parameters: {count_parameters(self.model):.2f}M")
        self.print(f"Model parameter counts: {self.model_parameter_counts}")
        if self.configs.get("ema_decay", -1) > 0:
            assert self.configs.ema_decay < 1
            self.ema_wrapper = EMAWrapper(
                self.model,
                self.configs.ema_decay,
                self.configs.ema_mutable_param_keywords,
                self.configs.ema_shadow_device,
            )

        torch.cuda.empty_cache()
        self.optimizer = get_optimizer(
            self.configs,
            self.model,
            param_names=self.configs.get("finetune_params_with_substring", [""]),
        )
        self.grad_scaler = torch.GradScaler(
            device="cuda" if torch.cuda.is_available() else "cpu",
            enabled=(self.configs.dtype == "fp16"),
        )
        self.init_scheduler()

    def init_scheduler(self, **kwargs: Any) -> None:
        """
        Initialize the learning rate scheduler.
        Supports both standard and fine-tuning schedulers.

        Args:
            **kwargs: Additional arguments passed to the scheduler.
        """
        # init finetune lr scheduler if available
        finetune_params = self.configs.get("finetune_params_with_substring", [""])
        is_finetune = len(finetune_params[0]) > 0

        if is_finetune:
            self.lr_scheduler = FinetuneLRScheduler(
                self.optimizer,
                self.configs,
                self.configs.finetune,
                **kwargs,
            )
        else:
            self.lr_scheduler = get_lr_scheduler(self.configs, self.optimizer, **kwargs)

    def init_data(self) -> None:
        """
        Initialize training and test dataloaders.
        """
        self.train_dl, self.test_dls = get_dataloaders(
            self.configs,
            DIST_WRAPPER.world_size,
            seed=self.configs.seed,
            error_dir=self.error_dir,
        )

    def _collect_rank_runtime_states(self) -> dict[str, Any]:
        """Collect RNG and next training-sample cursor from every rank."""
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        world_size = dist.get_world_size() if distributed else 1

        local_state = None
        capture_exception = None
        try:
            train_dataloader_state = None
            train_dl = getattr(self, "train_dl", None)
            state_dict_fn = getattr(train_dl, "state_dict", None)
            if callable(state_dict_fn):
                train_dataloader_state = state_dict_fn()
                consumed_batches = int(
                    train_dataloader_state["consumed_global_batches"]
                )
                # save_checkpoint is called after this batch was yielded and
                # trained, but before global_step is incremented.
                expected_consumed_batches = int(self.global_step) + 1
                if consumed_batches != expected_consumed_batches:
                    raise RuntimeError(
                        "train dataloader/checkpoint progress mismatch: "
                        f"loader consumed {consumed_batches} batches, expected "
                        f"{expected_consumed_batches} from global_step="
                        f"{self.global_step}"
                    )
            local_state = {
                "rank": rank,
                "rng": capture_rank_rng_state(self.device),
                "train_dataloader": train_dataloader_state,
            }
        except Exception as exc:
            capture_exception = exc

        synchronize_rank_exception(
            "checkpoint rank runtime-state capture",
            capture_exception,
        )
        assert local_state is not None

        if distributed:
            rank_states = [None] * world_size
            dist.all_gather_object(rank_states, local_state)
        else:
            rank_states = [local_state]
        return {
            "schema_version": 1,
            "world_size": world_size,
            "states": rank_states,
        }

    def _restore_rank_runtime_state(
        self,
        rank_runtime_states: Mapping[str, Any] | None,
        *,
        checkpoint_path: str,
    ) -> None:
        """Restore exact new-format state or reconstruct an old v2 cursor."""
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        world_size = dist.get_world_size() if distributed else 1
        train_dl = getattr(self, "train_dl", None)

        if rank_runtime_states is None:
            reconstruct_fn = getattr(train_dl, "reconstruct_state", None)
            if callable(reconstruct_fn):
                dataloader_state = reconstruct_fn(self.global_step)
                logging.warning(
                    "[rank %d] Legacy checkpoint %s has no per-rank RNG/data "
                    "state. Reconstructed the historical sampler cursor from "
                    "%d consumed global batches: epoch=%d, offset=%d. Python/"
                    "NumPy/torch/CUDA RNG streams cannot be recovered from this "
                    "checkpoint.",
                    rank,
                    checkpoint_path,
                    self.global_step,
                    dataloader_state["sampler_epoch"],
                    dataloader_state["next_local_offset"],
                )
            else:
                logging.warning(
                    "[rank %d] Legacy checkpoint %s has no per-rank RNG/data "
                    "state, and this trainer has no reconstructable train "
                    "dataloader. RNG streams were not restored.",
                    rank,
                    checkpoint_path,
                )
            return

        if rank_runtime_states.get("schema_version") != 1:
            raise ValueError(
                "unsupported rank runtime-state schema_version: "
                f"{rank_runtime_states.get('schema_version')!r}"
            )
        saved_world_size = int(rank_runtime_states.get("world_size", -1))
        if saved_world_size != world_size:
            raise ValueError(
                "full resume requires the same world size for per-rank RNG and "
                f"sampler state: checkpoint={saved_world_size}, current={world_size}"
            )
        states = rank_runtime_states.get("states")
        if not isinstance(states, Sequence) or len(states) != world_size:
            raise ValueError(
                "rank runtime-state list does not match checkpoint world size"
            )
        matching_states = [
            state
            for state in states
            if isinstance(state, Mapping) and int(state.get("rank", -1)) == rank
        ]
        if len(matching_states) != 1:
            raise ValueError(
                f"checkpoint must contain exactly one runtime state for rank {rank}"
            )
        local_state = matching_states[0]

        train_dataloader_state = local_state.get("train_dataloader")
        load_dataloader_state = getattr(train_dl, "load_state_dict", None)
        if getattr(self, "_reset_train_sampler", False):
            # Rebase the existing counters once; Adam and scheduler are untouched.
            self.step = int(os.environ.get("START_STEP", self.step))
            self.start_step = self.step
            self.global_step = self.step * self.iters_to_accumulate
            train_dataloader_state = train_dl.reconstruct_state(self.global_step)
            logging.info(
                "[rank %d] Started fresh data sampling with zero cluster exposure; "
                "restoring model, Adam, scheduler and rank RNG without reset",
                rank,
            )
        elif train_dataloader_state is not None:
            if not callable(load_dataloader_state):
                raise TypeError(
                    "checkpoint contains train dataloader state but the current "
                    "dataloader cannot restore it"
                )
            load_dataloader_state(train_dataloader_state)
        elif callable(load_dataloader_state):
            raise ValueError(
                "checkpoint rank runtime state is missing train dataloader state"
            )

        rng_state = local_state.get("rng")
        if not isinstance(rng_state, Mapping):
            raise ValueError("checkpoint rank runtime state is missing RNG state")
        # Restore RNG last so checkpoint application itself cannot advance the
        # resumed streams.
        restore_rank_rng_state(rng_state, self.device)
        if train_dataloader_state is None:
            logging.info(
                "[rank %d] Restored Python/NumPy/torch/CUDA RNG state from %s",
                rank,
                checkpoint_path,
            )
        else:
            logging.info(
                "[rank %d] Restored RNG and train sampler cursor from %s: "
                "epoch=%d, offset=%d, consumed_global_batches=%d",
                rank,
                checkpoint_path,
                train_dataloader_state["sampler_epoch"],
                train_dataloader_state["next_local_offset"],
                train_dataloader_state["consumed_global_batches"],
            )

    def _record_full_resume_manifest(
        self,
        *,
        checkpoint_path: str,
        checkpoint_step: int,
        rank_runtime_states: Mapping[str, Any] | None,
        resume_contract: Mapping[str, Any] | None = None,
    ) -> None:
        """Record the verified raw resume source and state schema on rank 0."""
        if DIST_WRAPPER.rank != 0:
            return
        manifest_path = getattr(self, "run_manifest_path", "")
        if not manifest_path or not os.path.exists(manifest_path):
            return
        migration = (
            _strict_resume_migration_expectations(self.configs)
            if strict_full_resume_enabled()
            else None
        )
        source_training_contract_id = (
            resume_contract.get("training_contract_id")
            if resume_contract is not None
            else None
        )
        source_data_contract_sha256 = (
            resume_contract.get("data_contract_sha256")
            if resume_contract is not None
            else None
        )
        target_training_contract_id = (
            migration["target_training_contract_id"]
            if migration is not None
            else source_training_contract_id
        )
        target_data_contract_sha256 = (
            migration["target_data_contract_sha256"]
            if migration is not None
            else source_data_contract_sha256
        )
        if getattr(self, "_reset_train_sampler", False):
            target_training_contract_id = os.environ.get("PROTENIX_TRAINING_CONTRACT_ID", "")
            target_data_contract_sha256 = build_training_data_contract(self.configs)["sha256"]
        temporary_path = f"{manifest_path}.tmp.resume.{os.getpid()}"
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            resume = manifest.setdefault("resume", {})
            resume.update(
                {
                    "raw_checkpoint": checkpoint_path,
                    "ema_checkpoint": getattr(
                        self.configs, "load_ema_checkpoint_path", ""
                    ),
                    "loaded_checkpoint_step": int(checkpoint_step),
                    "next_training_step": self.step,
                    "checkpoint_format_version": 2,
                    "rank_runtime_state_schema_version": (
                        rank_runtime_states.get("schema_version")
                        if rank_runtime_states is not None
                        else None
                    ),
                    "train_dataloader_state_schema_version": (
                        1 if rank_runtime_states is not None else None
                    ),
                    "legacy_sampler_cursor_reconstructed": (
                        rank_runtime_states is None
                    ),
                    "strict_full_resume": strict_full_resume_enabled(),
                    "full_state_restored": not getattr(self, "_reset_train_sampler", False),
                    "fresh_data_sampling": getattr(self, "_reset_train_sampler", False),
                    "restored_state_components": [
                        "raw_model",
                        "matched_ema_shadow",
                        "optimizer",
                        "scheduler",
                        *([] if getattr(self, "_reset_train_sampler", False) else ["step_and_global_step"]),
                        "grad_scaler",
                        "rank_python_numpy_torch_cuda_rng",
                        *([] if getattr(self, "_reset_train_sampler", False) else ["train_sampler_cursor"]),
                    ],
                    "training_contract_id": target_training_contract_id,
                    "source_training_contract_id": (
                        source_training_contract_id
                    ),
                    "target_training_contract_id": (
                        target_training_contract_id
                    ),
                    "checkpoint_pair_id": (
                        resume_contract.get("checkpoint_pair_id")
                        if resume_contract is not None
                        else None
                    ),
                    "data_profile_id": (
                        build_training_data_contract(self.configs)["profile_id"]
                    ),
                    "data_contract_sha256": target_data_contract_sha256,
                    "source_data_contract_sha256": (
                        source_data_contract_sha256
                    ),
                    "target_data_contract_sha256": (
                        target_data_contract_sha256
                    ),
                    "data_resume_scope": (
                        resume_contract.get("data_resume_scope")
                        if resume_contract is not None
                        else None
                    ),
                    "data_worker_rng_state_checkpointed": (
                        resume_contract.get(
                            "data_worker_rng_state_checkpointed"
                        )
                        if resume_contract is not None
                        else None
                    ),
                    "contract_migration": (
                        {
                            "mode": migration["mode"],
                            "allowed_delta": migration["allowed_delta"],
                            "source_run_id": migration["source_run_id"],
                            "source_training_contract_id": (
                                migration["source_training_contract_id"]
                            ),
                            "source_data_contract_sha256": (
                                migration["source_data_contract_sha256"]
                            ),
                            "target_training_contract_id": (
                                migration["target_training_contract_id"]
                            ),
                            "target_data_contract_sha256": (
                                migration["target_data_contract_sha256"]
                            ),
                            "model_architecture_sha256": (
                                migration[
                                    "target_model_architecture_sha256"
                                ]
                            ),
                            "trajectory_exact": False,
                            "trajectory_note": (
                                migration.get(
                                    "trajectory_note",
                                    "full state restored; training data policy "
                                    "changes only by the authorized MSA "
                                    "query-pin rule",
                                )
                            ),
                            **(
                                {
                                    "source_checkpoint_step": migration[
                                        "source_checkpoint_step"
                                    ],
                                    "source_checkpoint_pair_id": migration[
                                        "source_checkpoint_pair_id"
                                    ],
                                    "source_raw_checkpoint_sha256": migration[
                                        "source_raw_checkpoint_sha256"
                                    ],
                                    "source_ema_checkpoint_sha256": migration[
                                        "source_ema_checkpoint_sha256"
                                    ],
                                    "source_next_global_step": migration[
                                        "source_next_global_step"
                                    ],
                                }
                                if migration["mode"]
                                in CHECKPOINT_BOUND_MIGRATION_MODES
                                else {}
                            ),
                            **(
                                {
                                    "weightedpdb_msa_policy": migration[
                                        "weightedpdb_msa_policy"
                                    ],
                                    "source_scheduler_step": migration[
                                        "source_scheduler_step"
                                    ],
                                    "scheduler_migration": "none",
                                    "restored_source_lr": 0.00095,
                                }
                                if migration["mode"]
                                == (
                                    STRICT_RESUME_WEIGHTEDPDB_MSA_INDEPENDENT_MIGRATION_V1
                                )
                                else {}
                            ),
                            **(
                                {
                                    "soft_cluster_schedule_manifest_path": (
                                        migration[
                                            "soft_cluster_schedule_manifest_path"
                                        ]
                                    ),
                                    "soft_cluster_schedule_manifest_sha256": (
                                        migration[
                                            "soft_cluster_schedule_manifest_sha256"
                                        ]
                                    ),
                                    "soft_cluster_schedule_contract": (
                                        migration[
                                            "soft_cluster_schedule_contract"
                                        ]
                                    ),
                                }
                                if migration["mode"] in SOFT_CLUSTER_MIGRATION_MODES
                                else {}
                            ),
                            **(
                                {
                                    "catalog_migration": migration[
                                        "catalog_migration"
                                    ],
                                    "source_dataset_binding_sha256": migration[
                                        "source_dataset_binding_sha256"
                                    ],
                                    "target_dataset_binding_sha256": migration[
                                        "target_dataset_binding_sha256"
                                    ],
                                    "catalog_invariant_projection_sha256": migration[
                                        "catalog_invariant_projection_sha256"
                                    ],
                                    "catalog_dataset_semantics_sha256": migration[
                                        "catalog_dataset_semantics_sha256"
                                    ],
                                }
                                if migration["mode"]
                                == STRICT_RESUME_SOFT_CLUSTER_CATALOG_REPLACEMENT_MIGRATION_V2
                                else {}
                            ),
                        }
                        if migration is not None
                        else None
                    ),
                }
            )
            write_json(temporary_path, manifest)
            os.replace(temporary_path, manifest_path)
        except Exception:
            try:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
            except OSError:
                pass
            logging.exception(
                "Full resume succeeded, but run manifest %s could not be updated",
                manifest_path,
            )

    def save_checkpoint(self, ema_suffix: str = "") -> None:
        """
        Save the current model state, optimizer, and scheduler to a checkpoint file.

        Args:
            ema_suffix (str): Optional suffix for EMA checkpoints.
        """
        checkpoint_kind = "ema" if ema_suffix else "raw"
        resume_contract = build_checkpoint_resume_contract(self)
        rank_runtime_states = (
            self._collect_rank_runtime_states()
            if checkpoint_kind == "raw"
            else None
        )

        # Full FSDP state collection is collective: every rank must enter it,
        # even though only rank 0 writes the current single-file format.
        if self.use_fsdp:
            if checkpoint_kind == "ema":
                model_state = get_fsdp2_full_model_state_dict(self.model)
                optimizer_state = None
            else:
                model_state, optimizer_state = get_fsdp2_full_state_dict(
                    self.model,
                    self.optimizer,
                )
        else:
            model_state = None
            optimizer_state = None

        if (
            checkpoint_kind == "raw"
            and self.configs.optimizer_sharding == "zero1"
        ):
            self.optimizer.consolidate_state_dict(to=0)

        path = f"{self.checkpoint_dir}/{self.step}{ema_suffix}.pt"
        save_status = None
        rank0_save_exception = None
        if DIST_WRAPPER.rank == 0:
            temporary_path = f"{path}.tmp.{os.getpid()}"
            try:
                checkpoint = {
                    "format_version": 2,
                    "checkpoint_kind": checkpoint_kind,
                    "model": (
                        model_state if self.use_fsdp else self.model.state_dict()
                    ),
                    "optimizer": (
                        optimizer_state
                        if self.use_fsdp
                        else (
                            self.optimizer.state_dict()
                            if checkpoint_kind == "raw"
                            else None
                        )
                    ),
                    "scheduler": (
                        self.lr_scheduler.state_dict()
                        if checkpoint_kind == "raw"
                        and self.lr_scheduler is not None
                        else None
                    ),
                    "grad_scaler": (
                        self.grad_scaler.state_dict()
                        if checkpoint_kind == "raw"
                        else None
                    ),
                    "step": self.step,
                    "resume_contract": resume_contract,
                }
                if checkpoint_kind == "raw":
                    checkpoint["rank_runtime_states"] = rank_runtime_states
                torch.save(checkpoint, temporary_path)
                os.replace(temporary_path, path)
                if ema_suffix:
                    self.latest_ema_checkpoint_path = path
                else:
                    self.latest_checkpoint_path = path
                # The checkpoint itself is committed once os.replace succeeds.
                # Auxiliary audit files are best-effort and must not make peers
                # report a valid checkpoint as a failed/corrupt save.
                try:
                    append_jsonl(
                        self.checkpoint_index_path,
                        build_checkpoint_record(path, self.step, ema_suffix),
                    )
                    self.update_run_status(
                        status="running",
                        exit_reason="checkpoint_saved",
                    )
                except Exception:
                    logging.exception(
                        "Checkpoint %s was saved, but its audit metadata could "
                        "not be updated",
                        path,
                    )
                self.print(f"Saved checkpoint to {path}")
                save_status = {"ok": True, "path": path}
            except Exception as exc:  # synchronize ordinary rank-0 I/O failures
                rank0_save_exception = exc
                try:
                    if os.path.exists(temporary_path):
                        os.remove(temporary_path)
                except OSError:
                    pass
                save_status = {
                    "ok": False,
                    "path": path,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }

        status_list = [save_status]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(status_list, src=0)
        save_status = status_list[0]
        if not save_status["ok"]:
            error = RuntimeError(
                "Checkpoint save failed on rank 0: "
                f"{save_status['error_type']}: {save_status['error']}"
            )
            if rank0_save_exception is not None:
                raise error from rank0_save_exception
            raise error

    def try_load_checkpoint(self) -> None:
        """
        Attempt to load model and training state from specified checkpoint paths.
        Loads both standard and EMA checkpoints if configured.
        """
        strict_resume = validate_strict_full_resume_configuration(self.configs)
        params_only_source_pair = (
            os.environ.get("PROTENIX_INITIALIZATION_MODE", "").strip()
            == "params_only_source_pair"
        )
        if params_only_source_pair:
            if strict_resume or not self.configs.load_params_only:
                raise ValueError(
                    "params-only source-pair initialization cannot be a "
                    "strict/full-state resume"
                )
            if (
                not self.configs.load_checkpoint_path
                or not self.configs.load_ema_checkpoint_path
                or not self.configs.load_strict
                or not self.configs.skip_load_optimizer
                or not self.configs.skip_load_scheduler
                or not self.configs.skip_load_step
                or self.configs.load_step_for_scheduler
            ):
                raise ValueError(
                    "params-only source-pair initialization requires distinct "
                    "raw/EMA paths, strict model loading, and every training "
                    "state load disabled"
                )
            expected_architecture = os.environ.get(
                "PROTENIX_WARMSTART_SOURCE_MODEL_ARCHITECTURE_SHA256",
                "",
            )
            current_architecture = build_model_architecture_contract(
                self.configs
            )["sha256"]
            if expected_architecture != current_architecture:
                raise ValueError(
                    "warm-start source and target model architecture differ: "
                    f"source={expected_architecture}, "
                    f"target={current_architecture}"
                )
        strict_migration = (
            _strict_resume_migration_expectations(self.configs)
            if strict_resume
            else None
        )
        strict_contracts: dict[str, dict[str, Any]] = {}
        validated_checkpoint_bound_files: set[tuple[str, str]] = set()

        def _validate_checkpoint_bound_source_file(
            checkpoint_path: str,
            expected_kind: str,
        ) -> None:
            if (
                strict_migration is None
                or strict_migration.get("mode")
                not in CHECKPOINT_BOUND_MIGRATION_MODES
            ):
                return
            validation_key = (
                expected_kind,
                os.path.realpath(checkpoint_path),
            )
            if validation_key in validated_checkpoint_bound_files:
                return

            validation_status = None
            validation_exception = None
            distributed = dist.is_available() and dist.is_initialized()
            if not distributed or DIST_WRAPPER.rank == 0:
                try:
                    actual_sha256 = validate_soft_cluster_migration_checkpoint_sha256(
                        checkpoint_path,
                        expected_kind=expected_kind,
                        migration=strict_migration,
                    )
                    validation_status = {
                        "ok": True,
                        "actual_sha256": actual_sha256,
                    }
                except Exception as exc:
                    validation_exception = exc
                    validation_status = {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
            if distributed:
                status_list = [validation_status]
                dist.broadcast_object_list(status_list, src=0)
                validation_status = status_list[0]
            if not validation_status["ok"]:
                error = RuntimeError(
                    "Checkpoint-bound source validation failed on rank 0: "
                    f"{validation_status['error_type']}: "
                    f"{validation_status['error']}"
                )
                if validation_exception is not None:
                    raise error from validation_exception
                raise error
            validated_checkpoint_bound_files.add(validation_key)

        def _remember_strict_contract(
            checkpoint_kind: str,
            contract: Mapping[str, Any] | None,
        ) -> None:
            if not strict_resume:
                return
            if contract is None:
                raise ValueError(
                    f"strict {checkpoint_kind} checkpoint has no resume contract"
                )
            strict_contracts[checkpoint_kind] = dict(contract)
            if checkpoint_kind == "raw":
                self._reset_train_sampler = should_reset_training_data(contract, self.configs)
            other_kind = "ema" if checkpoint_kind == "raw" else "raw"
            if other_kind in strict_contracts:
                validate_matching_resume_contracts(
                    strict_contracts["raw"],
                    strict_contracts["ema"],
                )

        def _load_checkpoint(
            checkpoint_path: str,
            load_params_only: bool,
            skip_load_optimizer: bool = False,
            skip_load_step: bool = False,
            skip_load_scheduler: bool = False,
            load_step_for_scheduler: bool = True,
        ) -> None:
            """
            Internal helper to load a single checkpoint.

            Args:
                checkpoint_path (str): Path to the checkpoint file.
                load_params_only (bool): If True, only load model parameters.
                skip_load_optimizer (bool): If True, do not load optimizer state.
                skip_load_step (bool): If True, do not load training step.
                skip_load_scheduler (bool): If True, do not load scheduler state.
                load_step_for_scheduler (bool): If True, re-initialize scheduler with loaded step.
            """
            if (
                not load_params_only
                and skip_load_scheduler
                and load_step_for_scheduler
                and skip_load_step
            ):
                raise ValueError(
                    "load_step_for_scheduler requires checkpoint step loading"
                )
            self.print(
                f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
            )
            expected_kind = (
                "ema"
                if checkpoint_path == self.configs.load_ema_checkpoint_path
                else "raw"
            )
            _validate_checkpoint_bound_source_file(
                checkpoint_path,
                expected_kind,
            )
            if self.use_fsdp:
                rank0_load_exception = None
                if DIST_WRAPPER.rank == 0:
                    try:
                        if not os.path.exists(checkpoint_path):
                            raise FileNotFoundError(
                                "Given checkpoint path does not exist: "
                                f"[{checkpoint_path}]"
                            )
                        checkpoint = torch.load(
                            checkpoint_path,
                            map_location="cpu",
                            weights_only=False,
                        )
                        if not isinstance(checkpoint, Mapping):
                            raise TypeError("checkpoint must be a mapping")
                        model_state = checkpoint.get("model")
                        if not isinstance(model_state, Mapping) or not model_state:
                            raise ValueError(
                                "checkpoint model state must be a non-empty mapping"
                            )
                        checkpoint_kind = validate_checkpoint_header(
                            checkpoint,
                            checkpoint_path=checkpoint_path,
                            load_params_only=load_params_only,
                        )
                        strict_contract = None
                        if strict_resume:
                            strict_contract = validate_strict_checkpoint_payload(
                                checkpoint,
                                checkpoint_path=checkpoint_path,
                                expected_kind=expected_kind,
                                configs=self.configs,
                                grad_scaler_enabled=self.grad_scaler.is_enabled(),
                                device=self.device,
                                migration_expectations=strict_migration,
                            )
                        if (
                            os.environ.get("PROTENIX_FULL_RESUME", "").strip()
                            == "1"
                            and checkpoint_path
                            == self.configs.load_ema_checkpoint_path
                        ):
                            if checkpoint_kind != "ema":
                                raise ValueError(
                                    "PROTENIX_FULL_RESUME=1 requires the EMA "
                                    "companion checkpoint to declare "
                                    "checkpoint_kind='ema'"
                                )
                            validate_expected_resume_checkpoint_step(
                                checkpoint.get("step"),
                                checkpoint_path=checkpoint_path,
                            )
                        optimizer_state = checkpoint.get("optimizer")
                        optimizer_required = (
                            not load_params_only and not skip_load_optimizer
                        )
                        if optimizer_required and (
                            not isinstance(optimizer_state, Mapping)
                            or "param_groups" not in optimizer_state
                        ):
                            raise ValueError(
                                "full resume requires an optimizer state with "
                                "param_groups"
                            )
                        if optimizer_state is None:
                            optimizer_state = {}
                        sample_key = next(iter(model_state))
                        if sample_key.startswith("module."):
                            model_state = {
                                key[len("module.") :]: value
                                for key, value in model_state.items()
                            }
                        if (
                            not load_params_only
                            and not skip_load_step
                            and checkpoint.get("step") is None
                        ):
                            raise ValueError("full resume requires checkpoint step")
                        if not load_params_only and not skip_load_step:
                            validate_expected_resume_checkpoint_step(
                                checkpoint.get("step"),
                                checkpoint_path=checkpoint_path,
                            )
                        if (
                            not load_params_only
                            and not skip_load_scheduler
                            and checkpoint.get("scheduler") is None
                        ):
                            raise ValueError(
                                "full resume requires scheduler state unless "
                                "skip_load_scheduler=true"
                            )
                        metadata = {
                            "sample_key": sample_key,
                            "checkpoint_kind": checkpoint_kind,
                            "step": checkpoint.get("step"),
                            "scheduler": checkpoint.get("scheduler"),
                            "grad_scaler": checkpoint.get("grad_scaler"),
                            "rank_runtime_states": checkpoint.get(
                                "rank_runtime_states"
                            ),
                            "resume_contract": strict_contract,
                            "optimizer_is_canonical": (
                                optimizer_state_uses_canonical_fqns(optimizer_state)
                            ),
                        }
                        load_payload = {"ok": True, "metadata": metadata}
                    except Exception as exc:
                        rank0_load_exception = exc
                        model_state = {}
                        optimizer_state = {}
                        load_payload = {
                            "ok": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                else:
                    model_state = {}
                    optimizer_state = {}
                    load_payload = None

                payload_list = [load_payload]
                dist.broadcast_object_list(payload_list, src=0)
                load_payload = payload_list[0]
                if not load_payload["ok"]:
                    error = RuntimeError(
                        "Checkpoint load/validation failed on rank 0: "
                        f"{load_payload['error_type']}: {load_payload['error']}"
                    )
                    if rank0_load_exception is not None:
                        raise error from rank0_load_exception
                    raise error
                metadata = load_payload["metadata"]
                _remember_strict_contract(
                    metadata["checkpoint_kind"],
                    metadata["resume_contract"],
                )
                self.print(f"Sampled key: {metadata['sample_key']}")
                if (
                    not load_params_only
                    and not skip_load_optimizer
                    and not metadata["optimizer_is_canonical"]
                ):
                    raise ValueError(
                        "FSDP2 optimizer resume requires a non-empty FSDP2 "
                        "optimizer state with canonical parameter names; use "
                        "--skip_load_optimizer true for a legacy DDP checkpoint"
                    )
                set_fsdp2_full_model_state_dict(
                    self.model,
                    model_state,
                    strict=self.configs.load_strict,
                )

                if not load_params_only and not skip_load_optimizer:
                    self.print("Loading optimizer state")
                    set_fsdp2_full_optimizer_state_dict(
                        self.model,
                        self.optimizer,
                        optimizer_state,
                    )
                if not load_params_only:
                    if metadata["grad_scaler"] is not None:
                        self.grad_scaler.load_state_dict(metadata["grad_scaler"])
                    if not skip_load_step:
                        self.print("Loading checkpoint step")
                        self.step = metadata["step"] + 1
                        self.start_step = self.step
                        self.global_step = self.step * self.iters_to_accumulate
                    if not skip_load_scheduler:
                        self.print("Loading scheduler state")
                        self.lr_scheduler.load_state_dict(metadata["scheduler"])
                    elif load_step_for_scheduler:
                        assert not skip_load_step, (
                            "If load_step_for_scheduler is True, you must load step first"
                        )
                        self.init_scheduler(last_epoch=self.step - 1)
                    if not skip_load_step:
                        self._restore_rank_runtime_state(
                            metadata["rank_runtime_states"],
                            checkpoint_path=checkpoint_path,
                        )
                        self._record_full_resume_manifest(
                            checkpoint_path=checkpoint_path,
                            checkpoint_step=metadata["step"],
                            rank_runtime_states=metadata[
                                "rank_runtime_states"
                            ],
                            resume_contract=metadata["resume_contract"],
                        )

                self.print(
                    "Finish loading model parameters"
                    if load_params_only
                    else f"Finish loading checkpoint, current step: {self.step}"
                )
                return

            checkpoint = None
            model_state = None
            optimizer_state = None
            sample_key = None
            ddp_load_exception = None
            try:
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(
                        "Given checkpoint path does not exist: "
                        f"[{checkpoint_path}]"
                    )
                checkpoint = torch.load(
                    checkpoint_path,
                    # Params-only stage initialization must not materialize
                    # the source optimizer state on every GPU. mmap keeps the
                    # checkpoint on CPU and load_state_dict touches only model
                    # storages while copying them to the DDP module.
                    map_location="cpu" if load_params_only else self.device,
                    weights_only=False,
                    mmap=load_params_only,
                )
                if not isinstance(checkpoint, Mapping):
                    raise TypeError("checkpoint must be a mapping")
                checkpoint_kind = validate_checkpoint_header(
                    checkpoint,
                    checkpoint_path=checkpoint_path,
                    load_params_only=load_params_only,
                )
                strict_contract = None
                if strict_resume:
                    strict_contract = validate_strict_checkpoint_payload(
                        checkpoint,
                        checkpoint_path=checkpoint_path,
                        expected_kind=expected_kind,
                        configs=self.configs,
                        grad_scaler_enabled=self.grad_scaler.is_enabled(),
                        device=self.device,
                        migration_expectations=strict_migration,
                    )
                    _remember_strict_contract(
                        checkpoint_kind,
                        strict_contract,
                    )
                if (
                    os.environ.get("PROTENIX_FULL_RESUME", "").strip() == "1"
                    and checkpoint_path
                    == self.configs.load_ema_checkpoint_path
                ):
                    if checkpoint_kind != "ema":
                        raise ValueError(
                            "PROTENIX_FULL_RESUME=1 requires the EMA companion "
                            "checkpoint to declare checkpoint_kind='ema'"
                        )
                    validate_expected_resume_checkpoint_step(
                        checkpoint.get("step"),
                        checkpoint_path=checkpoint_path,
                    )
                model_state = checkpoint.get("model")
                if not isinstance(model_state, Mapping) or not model_state:
                    raise ValueError(
                        "checkpoint model state must be a non-empty mapping"
                    )
                if not load_params_only:
                    if not skip_load_optimizer:
                        optimizer_state = checkpoint.get("optimizer")
                        if (
                            not isinstance(optimizer_state, Mapping)
                            or "param_groups" not in optimizer_state
                            or not optimizer_state["param_groups"]
                        ):
                            raise ValueError(
                                "full resume requires a non-empty optimizer state "
                                "with param_groups"
                            )
                    if not skip_load_step and checkpoint.get("step") is None:
                        raise ValueError("full resume requires checkpoint step")
                    if not skip_load_step:
                        validate_expected_resume_checkpoint_step(
                            checkpoint.get("step"),
                            checkpoint_path=checkpoint_path,
                        )
                    if (
                        not skip_load_scheduler
                        and checkpoint.get("scheduler") is None
                    ):
                        raise ValueError(
                            "full resume requires scheduler state unless "
                            "skip_load_scheduler=true"
                        )

                sample_key = next(iter(model_state))
                if sample_key.startswith("module.") and not self.use_ddp:
                    # DDP checkpoint has module. prefix, remove it if needed.
                    model_state = {
                        key[len("module.") :]: value
                        for key, value in model_state.items()
                    }
                elif not sample_key.startswith("module.") and self.use_ddp:
                    # Canonical FSDP2/full states can initialize DDP as well.
                    model_state = {
                        f"module.{key}": value
                        for key, value in model_state.items()
                    }
            except Exception as exc:
                ddp_load_exception = exc

            synchronize_rank_exception(
                "DDP checkpoint load/validation",
                ddp_load_exception,
            )
            assert checkpoint is not None
            assert model_state is not None
            assert sample_key is not None
            self.print(f"Sampled key: {sample_key}")

            ddp_apply_exception = None
            try:
                self.model.load_state_dict(
                    state_dict=model_state,
                    strict=self.configs.load_strict,
                )
                if not load_params_only:
                    if not skip_load_optimizer:
                        self.print("Loading optimizer state")
                        self.optimizer.load_state_dict(optimizer_state)
                    if checkpoint.get("grad_scaler") is not None:
                        self.grad_scaler.load_state_dict(checkpoint["grad_scaler"])
                    if not skip_load_step:
                        self.print("Loading checkpoint step")
                        self.step = checkpoint["step"] + 1
                        self.start_step = self.step
                        self.global_step = self.step * self.iters_to_accumulate
                    if not skip_load_scheduler:
                        self.print("Loading scheduler state")
                        self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
                    elif load_step_for_scheduler:
                        self.init_scheduler(last_epoch=self.step - 1)
                    if not skip_load_step:
                        rank_runtime_states = checkpoint.get(
                            "rank_runtime_states"
                        )
                        self._restore_rank_runtime_state(
                            rank_runtime_states,
                            checkpoint_path=checkpoint_path,
                        )
                        self._record_full_resume_manifest(
                            checkpoint_path=checkpoint_path,
                            checkpoint_step=checkpoint["step"],
                            rank_runtime_states=rank_runtime_states,
                            resume_contract=strict_contract,
                        )
            except Exception as exc:
                ddp_apply_exception = exc

            synchronize_rank_exception(
                "DDP checkpoint state application",
                ddp_apply_exception,
            )

            self.print(
                "Finish loading model parameters"
                if load_params_only
                else f"Finish loading checkpoint, current step: {self.step}"
            )

        if (
            strict_migration is not None
            and strict_migration.get("mode")
            in CHECKPOINT_BOUND_MIGRATION_MODES
        ):
            # Authenticate the complete raw/EMA pair before deserializing or
            # applying either member.  A bad raw file therefore cannot leave
            # the live EMA shadow partially changed (or vice versa).
            _validate_checkpoint_bound_source_file(
                self.configs.load_checkpoint_path,
                "raw",
            )
            _validate_checkpoint_bound_source_file(
                self.configs.load_ema_checkpoint_path,
                "ema",
            )

        ema_registered = False
        # Load EMA model parameters first so its rank-local values become the
        # shadow, then restore the live/raw checkpoint below.
        if self.configs.load_ema_checkpoint_path:
            if not hasattr(self, "ema_wrapper"):
                raise ValueError(
                    "load_ema_checkpoint_path requires ema_decay to be enabled"
                )
            _load_checkpoint(
                self.configs.load_ema_checkpoint_path,
                load_params_only=True,
            )
            self.ema_wrapper.register()
            ema_registered = True

        # Load standard model checkpoint if configured
        if self.configs.load_checkpoint_path:
            _load_checkpoint(
                self.configs.load_checkpoint_path,
                self.configs.load_params_only,
                skip_load_optimizer=self.configs.skip_load_optimizer,
                skip_load_scheduler=self.configs.skip_load_scheduler,
                skip_load_step=self.configs.skip_load_step,
                load_step_for_scheduler=self.configs.load_step_for_scheduler,
            )

        # With no separate EMA checkpoint, initialize EMA from the final live
        # weights (including an optional raw checkpoint), not random pre-load weights.
        if hasattr(self, "ema_wrapper") and not ema_registered:
            self.ema_wrapper.register()
        if params_only_source_pair:
            if (self.step, self.start_step, self.global_step) != (0, 0, 0):
                raise ValueError(
                    "params-only source-pair initialization must start the "
                    "target stage at step/global_step 0"
                )
            if self.optimizer.state:
                raise ValueError(
                    "params-only source-pair initialization unexpectedly "
                    "loaded optimizer state"
                )
            self.print(
                "Params-only source pair loaded: live model=raw, EMA "
                "shadow=EMA; optimizer/scheduler/step/RNG/sampler are fresh"
            )
        if strict_resume and set(strict_contracts) != {"raw", "ema"}:
            raise ValueError(
                "strict full resume did not validate both raw and EMA checkpoints"
            )
        if strict_resume:
            migration = strict_migration
            if migration is not None:
                source_contract = strict_contracts["raw"]
                if (
                    migration["mode"] in SOFT_CLUSTER_MIGRATION_MODES
                ):
                    self.print(
                        "Strict full-state contract migration validated: "
                        f"mode={migration['mode']} "
                        f"source_run={source_contract['run_id']} "
                        f"source_step={source_contract['step']} "
                        f"source_pair={source_contract['checkpoint_pair_id']} "
                        f"source_contract={migration['source_training_contract_id']} "
                        f"target_contract={migration['target_training_contract_id']} "
                        f"schedule_manifest_sha256="
                        f"{migration['soft_cluster_schedule_manifest_sha256']} "
                        f"allowed_delta={migration['allowed_delta']}; "
                        "restored=model+matched_ema+optimizer+scheduler+step+"
                        "grad_scaler+rank_rng; migrated=train_sampler_cursor+"
                        "future_soft_cluster_schedule"
                    )
                else:
                    self.print(
                        "Strict full-state contract migration validated: "
                        f"mode={migration['mode']} "
                        f"source_run={source_contract['run_id']} "
                        f"source_step={source_contract['step']} "
                        f"source_pair={source_contract['checkpoint_pair_id']} "
                        f"source_contract={migration['source_training_contract_id']} "
                        f"target_contract={migration['target_training_contract_id']} "
                        f"allowed_delta={migration['allowed_delta']}; "
                        "restored=model+matched_ema+optimizer+scheduler+step+"
                        "grad_scaler+rank_rng+sampler_cursor; "
                        "data_worker_rng=not_checkpointed"
                    )

        smoke_steps = int(os.environ.get("PROTENIX_RESUME_SMOKE_STEPS", "0"))
        if smoke_steps > 0:
            self.configs.max_steps = min(self.configs.max_steps, self.step + smoke_steps)

    def print(self, msg: str) -> None:
        """
        Print message to log only on the master rank (rank 0).

        Args:
            msg (str): The message to log.
        """
        if DIST_WRAPPER.rank == 0:
            logging.info(msg)

    def log_msa_debug_batch(
        self,
        batch: Mapping[str, Any],
        batch_summary: Mapping[str, Any],
        mode: str,
    ) -> None:
        """Log data-side MSA counts for the first configured batches."""
        if self.msa_debug_log_count >= self.msa_debug_log_limit:
            return

        input_feature_dict = _get_value(batch, "input_feature_dict", {})
        msa = _get_value(input_feature_dict, "msa")
        msa_shape = list(getattr(msa, "shape", []))
        record = {
            "event": "msa_debug_batch",
            "mode": mode,
            "step": self.step,
            "global_step": self.global_step,
            "rank": DIST_WRAPPER.rank,
            "world_size": DIST_WRAPPER.world_size,
            "sample": batch_summary,
            "msa_shape": msa_shape,
        }
        append_jsonl(self.runtime_log_path, record)
        logging.info(
            "[MSA_DEBUG][batch] mode=%s rank=%s step=%s global_step=%s "
            "dataset=%s pdb_id=%s n_token=%s n_atom=%s N_msa=%s "
            "prot_pair=%s prot_unpair=%s rna_pair=%s rna_unpair=%s "
            "virtual_segments=%s segment_shuffle_applied=%s "
            "shuffle_groups=%s shuffle_variant=%s "
            "variant_eligible=%s variant_applied=%s "
            "whole_chain_applied=%s "
            "crop_protein_chains=%s query_only_protein_chains=%s "
            "retained_protein_chains=%s partial_query_only=%s "
            "global_query_only=%s "
            "msa_mode=%s template_requested=%s valid_templates=%s "
            "msa_shape=%s",
            mode,
            DIST_WRAPPER.rank,
            self.step,
            self.global_step,
            _single_value(batch_summary.get("dataset_name")),
            _single_value(batch_summary.get("pdb_id")),
            _single_value(batch_summary.get("n_token")),
            _single_value(batch_summary.get("n_atom")),
            _single_value(batch_summary.get("N_msa")),
            _single_value(batch_summary.get("N_msa_prot_pair")),
            _single_value(batch_summary.get("N_msa_prot_unpair")),
            _single_value(batch_summary.get("N_msa_rna_pair")),
            _single_value(batch_summary.get("N_msa_rna_unpair")),
            _single_value(batch_summary.get("N_msa_virtual_segment")),
            _single_value(
                batch_summary.get(
                    "msa_virtual_segment_shuffle_applied"
                )
            ),
            _single_value(batch_summary.get("N_msa_shuffle_group")),
            _single_value(
                batch_summary.get(
                    "msa_shuffle_requested_variant_code"
                )
            ),
            _single_value(
                batch_summary.get("msa_shuffle_variant_eligible")
            ),
            _single_value(
                batch_summary.get("msa_shuffle_variant_applied")
            ),
            _single_value(
                batch_summary.get(
                    "msa_shuffle_whole_physical_chain_applied"
                )
            ),
            _single_value(
                batch_summary.get("N_msa_crop_protein_chain")
            ),
            _single_value(
                batch_summary.get("N_msa_query_only_protein_chain")
            ),
            _single_value(
                batch_summary.get("N_msa_retained_protein_chain")
            ),
            _single_value(
                batch_summary.get("msa_partial_query_only_applied")
            ),
            _single_value(
                batch_summary.get("msa_global_query_only_applied")
            ),
            _single_value(batch_summary.get("msa_mode")),
            _single_value(batch_summary.get("template_use_requested")),
            _single_value(batch_summary.get("N_template_valid")),
            msa_shape,
        )
        self.msa_debug_log_count += 1

    def model_forward(
        self, batch: Dict[str, Any], mode: str = "train"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform a forward pass of the model.

        Args:
            batch (Dict[str, Any]): Input batch containing features and labels.
            mode (str): Execution mode, either 'train' or 'eval'.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Updated batch with
                predictions and logging dictionary.
        """
        assert mode in ["train", "eval"]
        msa_module = getattr(self.raw_model, "msa_module", None)
        reset_msa_depth_trace = getattr(
            msa_module, "reset_msa_depth_trace", None
        )
        if callable(reset_msa_depth_trace):
            reset_msa_depth_trace()

        model_kwargs = {}
        phase_memory = getattr(self, "benchmark_phase_memory", None)
        if phase_memory is not None and phase_memory.recording_step:
            model_kwargs["memory_phase_callback"] = self.benchmark_phase_transition
        batch["pred_dict"], batch["label_dict"], log_dict = self.model(
            input_feature_dict=batch["input_feature_dict"],
            label_dict=batch["label_dict"],
            label_full_dict=batch["label_full_dict"],
            mode=mode,
            current_step=self.step if mode == "train" else None,
            symmetric_permutation=self.symmetric_permutation,
            mc_dropout_apply_rate=(
                0 if mode == "train" else self.configs.mc_dropout_apply_rate
            ),
            **model_kwargs,
        )
        msa_depth_trace = getattr(msa_module, "msa_depth_trace", ())
        log_dict = dict(log_dict or {})
        log_dict["msa_depth_trace"] = [
            {
                "input_depth": call.get("input_depth"),
                "sampled_depth": call.get("sampled_depth"),
            }
            for call in (msa_depth_trace or ())
            if isinstance(call, Mapping)
        ]
        return batch, log_dict

    def benchmark_phase_transition(self, phase_name: str) -> None:
        """Advance the optional rank-local CUDA phase recorder."""
        recorder = getattr(self, "benchmark_phase_memory", None)
        if recorder is not None:
            recorder.transition(phase_name)

    def get_loss(
        self, batch: Dict[str, Any], mode: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        """
        Compute the loss for a given batch.

        Args:
            batch (Dict[str, Any]): Batch containing features, labels, and predictions.
            mode (str): Execution mode, either 'train' or 'eval'.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
                - Total loss tensor.
                - Dictionary containing individual loss components.
                - Updated batch dictionary.
        """
        assert mode in ["train", "eval"]

        loss, loss_dict = autocasting_disable_decorator(self.configs.skip_amp.loss)(
            self.loss
        )(
            feat_dict=batch["input_feature_dict"],
            pred_dict=batch["pred_dict"],
            label_dict=batch["label_dict"],
            mode=mode,
        )
        return loss, loss_dict, batch

    @torch.no_grad()
    def get_metrics(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute evaluation metrics (e.g., lDDT) for a batch.

        Args:
            batch (Dict[str, Any]): Batch containing predictions and labels.

        Returns:
            Dict[str, Any]: Dictionary containing computed metrics.
        """
        lddt_dict = self.lddt_metrics.compute_lddt(
            batch["pred_dict"], batch["label_dict"]
        )
        return lddt_dict

    @torch.no_grad()
    def aggregate_metrics(
        self, lddt_dict: Dict[str, Any], batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across the batch.

        Args:
            lddt_dict (Dict[str, Any]): Dictionary of per-structure metrics.
            batch (Dict[str, Any]): Batch containing confidence summaries.

        Returns:
            Dict[str, Any]: Aggregated metrics.
        """
        simple_metrics, _ = self.lddt_metrics.aggregate_lddt(
            lddt_dict, batch["pred_dict"]["summary_confidence"]
        )
        return simple_metrics

    @torch.no_grad()
    def evaluate(self, mode: str = "eval") -> None:
        """
        Evaluate the model on all test datasets.
        Handles both standard and EMA model evaluation if applicable.

        Args:
            mode (str): Execution mode, typically 'eval'.
        """
        if not self.configs.eval_ema_only:
            self._evaluate(mode=mode)
        if hasattr(self, "ema_wrapper"):
            self.ema_wrapper.apply_shadow()
            try:
                self._evaluate(
                    ema_suffix=f"ema{self.ema_wrapper.decay}_",
                    mode=mode,
                )
            finally:
                self.ema_wrapper.restore()

    @torch.no_grad()
    def _evaluate(self, ema_suffix: str = "", mode: str = "eval") -> None:
        """
        Internal evaluation loop for a specific model state.

        Args:
            ema_suffix (str): Suffix for metric names if using EMA.
            mode (str): Execution mode.
        """
        # Init Metric Aggregator
        simple_metric_wrapper = SimpleMetricAggregator(get_eval_aggregator_names())
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )
        self.model.eval()

        for test_name, test_dl in self.test_dls.items():
            self.print(f"Testing on {test_name}")
            eval_start = time.perf_counter()
            evaluated_pids = []
            eval_sample_records = []
            total_batch_num = len(test_dl)
            for index, batch in enumerate(tqdm(test_dl)):
                batch = to_device(batch, self.device)
                self.log_msa_debug_batch(
                    batch=batch,
                    batch_summary=summarize_batch(batch),
                    mode=f"eval:{test_name}",
                )
                pid = batch["basic"]["pdb_id"]

                if index + 1 == total_batch_num and DIST_WRAPPER.world_size > 1:
                    # Gather all pids across ranks to avoid duplicate evaluation
                    # when drop_last=False
                    all_data_ids = DIST_WRAPPER.all_gather_object(evaluated_pids)
                    dedup_ids = set(sum(all_data_ids, []))
                    if pid in dedup_ids:
                        print(
                            f"Rank {DIST_WRAPPER.rank}: Dropping data_id {pid} "
                            "as it is already evaluated."
                        )
                        break
                evaluated_pids.append(pid)

                simple_metrics = {}
                with enable_amp:
                    # Model forward
                    batch, _ = self.model_forward(batch, mode=mode)
                    # Loss forward
                    _, loss_dict, batch = self.get_loss(batch, mode="eval")
                    # lDDT metrics
                    lddt_dict = self.get_metrics(batch)
                    lddt_metrics = self.aggregate_metrics(lddt_dict, batch)
                    simple_metrics.update(
                        {k: v for k, v in lddt_metrics.items() if "diff" not in k}
                    )
                    simple_metrics.update(loss_dict)

                # Update metric aggregator
                for key, value in simple_metrics.items():
                    simple_metric_wrapper.add(
                        f"{ema_suffix}{key}", value, namespace=test_name
                    )
                sample_record = build_eval_sample_record(
                    test_name=test_name,
                    ema_suffix=ema_suffix,
                    step=self.step,
                    global_step=self.global_step,
                    rank=DIST_WRAPPER.rank,
                    world_size=DIST_WRAPPER.world_size,
                    batch_index=index,
                    total_batches=total_batch_num,
                    sample=summarize_batch(batch),
                    metrics=simple_metrics,
                )
                eval_sample_records.append(sample_record)
                append_jsonl(self.eval_samples_log_path, sample_record)

                del batch, simple_metrics
                if index % 5 == 0:
                    # Release memory periodically
                    torch.cuda.empty_cache()

            metrics = simple_metric_wrapper.calc()
            gathered_records = DIST_WRAPPER.all_gather_object(eval_sample_records)
            all_eval_sample_records = [
                record for rank_records in gathered_records for record in rank_records
            ]
            self.print(f"Step {self.step}, eval {test_name}: {metrics}")
            if DIST_WRAPPER.rank == 0:
                append_jsonl(
                    self.eval_worst_samples_log_path,
                    {
                        "event": "eval_worst_samples",
                        "step": self.step,
                        "global_step": self.global_step,
                        "test_name": test_name,
                        "ema_suffix": ema_suffix,
                        "is_ema": bool(ema_suffix),
                        "sample_count": len(all_eval_sample_records),
                        "worst": select_worst_eval_samples(
                            all_eval_sample_records,
                            self.configs.eval_worst_sample_count,
                        ),
                    },
                )
                append_jsonl(
                    self.eval_metrics_log_path,
                    {
                        "event": "eval_metrics",
                        "step": self.step,
                        "global_step": self.global_step,
                        "rank": DIST_WRAPPER.rank,
                        "world_size": DIST_WRAPPER.world_size,
                        "test_name": test_name,
                        "ema_suffix": ema_suffix,
                        "is_ema": bool(ema_suffix),
                        "num_batches": total_batch_num,
                        "sample_count": len(all_eval_sample_records),
                        "eval_seconds": time.perf_counter() - eval_start,
                        "metrics": metrics,
                    },
                )
            if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                wandb.log(metrics, step=self.step)

    def update(self) -> float | None:
        """
        Apply gradient clipping to model parameters.
        """
        if self.configs.grad_clip_norm != 0.0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.configs.grad_clip_norm
            )
            return _global_scalar_float(total_norm)
        return None

    def log_numeric_error(
        self,
        reason: str,
        batch: Mapping[str, Any],
        loss_dict: Mapping[str, Any],
    ) -> None:
        """Persist a compact numeric-error record for later debugging."""
        append_jsonl(
            self.numeric_error_log_path,
            {
                "event": "numeric_error",
                "reason": reason,
                "step": self.step,
                "global_step": self.global_step,
                "rank": DIST_WRAPPER.rank,
                "world_size": DIST_WRAPPER.world_size,
                "sample": summarize_batch(batch),
                "losses": loss_dict,
                "cuda_memory": get_cuda_memory_stats(self.device),
            },
        )

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a single training step including forward, backward, and optimization.

        Args:
            batch (Dict[str, Any]): Input batch for training.
        """
        step_start = time.perf_counter()
        self.model.train()
        train_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        enable_amp = (
            torch.autocast(
                device_type="cuda", dtype=train_precision, cache_enabled=False
            )
            if torch.cuda.is_available()
            else nullcontext()
        )

        scaler = self.grad_scaler
        amp_scale = _finite_scalar_float(scaler.get_scale())

        forward_start = time.perf_counter()
        with enable_amp:
            batch, model_log_dict = self.model_forward(batch, mode="train")
            forward_seconds = time.perf_counter() - forward_start
            self.benchmark_phase_transition("loss")
            loss_start = time.perf_counter()
            loss, loss_dict, _ = self.get_loss(batch, mode="train")
            loss_seconds = time.perf_counter() - loss_start
        self.benchmark_phase_transition("loss_validation")

        is_update_step = (
            (self.global_step + 1) % self.iters_to_accumulate == 0
        )
        # Loss health is checked locally on every microbatch.  The Python flag
        # intentionally survives until the accumulation boundary, where one
        # MAX reduction makes the skip decision identical on every rank.  This
        # avoids turning every DDP no_sync() microbatch into a 128-rank barrier.
        local_loss_nan = not bool(torch.isfinite(loss.detach()).all().item())
        loss_nan = local_loss_nan
        if local_loss_nan:
            self._skip_accumulated_update = True
            self.log_numeric_error("nan_or_inf_loss", batch, loss_dict)

        loss_health_sync_seconds = 0.0
        loss_nonfinite_in_window = bool(self._skip_accumulated_update)
        if is_update_step:
            if dist.is_available() and dist.is_initialized():
                self.benchmark_phase_transition("loss_health_sync")
                loss_health_sync_start = time.perf_counter()
                nonfinite_loss_flag = torch.tensor(
                    int(loss_nonfinite_in_window),
                    device=loss.device,
                    dtype=torch.int32,
                )
                dist.all_reduce(nonfinite_loss_flag, op=dist.ReduceOp.MAX)
                loss_nonfinite_in_window = bool(nonfinite_loss_flag.item())
                loss_health_sync_seconds = (
                    time.perf_counter() - loss_health_sync_start
                )
            self._skip_accumulated_update = loss_nonfinite_in_window
            # With accumulation=1 this preserves the former per-microbatch
            # global loss_nan result exactly.  With accumulation>1 loss_nan is
            # rank-local for the current sample, while the separate window
            # field records the synchronized optimizer-skip decision.
            if self.iters_to_accumulate == 1:
                loss_nan = loss_nonfinite_in_window
                if loss_nan and not local_loss_nan:
                    # Preserve the former accumulation=1 diagnostics: a
                    # remote bad loss produces a numeric-error record on every
                    # rank as well as the same global skip decision.
                    self.log_numeric_error("nan_or_inf_loss", batch, loss_dict)
            if loss_nonfinite_in_window:
                self.print(
                    "Non-finite loss in accumulation window ending at "
                    f"step {self.step}; finish backward and skip this "
                    "optimizer update"
                )

        # Preserve the original graph so every FSDP pre/post-backward hook and
        # reduce-scatter collective still runs on a bad-loss iteration.
        loss_for_backward = (
            torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
            if local_loss_nan
            else loss
        )
        backward_start = time.perf_counter()
        self.benchmark_phase_transition("backward")
        scaler.scale(loss_for_backward / self.iters_to_accumulate).backward()
        backward_seconds = time.perf_counter() - backward_start
        self.benchmark_phase_transition("grad_prepare")

        grad_norm = None
        grad_clipped = False
        nonfinite_grad = False
        optimizer_skipped = False
        did_update = False
        optimizer_seconds = 0.0
        # Global training step used for accumulation logic
        if is_update_step:
            optimizer_start = time.perf_counter()
            old_amp_scale = _scalar_float(scaler.get_scale())
            if not self._skip_accumulated_update:
                self.benchmark_phase_transition("clip")
                # Unscale before computing/clipping the true global norm. Keep
                # both operations in the clip phase because unscale is required
                # to expose the true gradient norm and may use foreach kernels.
                scaler.unscale_(self.optimizer)
                grad_norm = self.update()
                self.benchmark_phase_transition("gradient_health")
                grad_norm_value = _scalar_float(grad_norm)
                if grad_norm_value is not None:
                    nonfinite_grad = not math.isfinite(grad_norm_value)
                    grad_clipped = (
                        not nonfinite_grad
                        and self.configs.grad_clip_norm != 0.0
                        and grad_norm_value > self.configs.grad_clip_norm
                    )
                else:
                    nonfinite_grad = _has_nonfinite_local_gradients(self.model)

                nonfinite_grad_flag = torch.tensor(
                    int(nonfinite_grad),
                    device=self.device,
                    dtype=torch.int32,
                )
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(nonfinite_grad_flag, op=dist.ReduceOp.MAX)
                nonfinite_grad = bool(nonfinite_grad_flag.item())
                if nonfinite_grad:
                    self._skip_accumulated_update = True
                    self.log_numeric_error("nan_or_inf_gradient", batch, loss_dict)

            self.benchmark_phase_transition("optimizer")
            if not self._skip_accumulated_update:
                scaler.step(self.optimizer)
                scaler.update()
                new_amp_scale = _scalar_float(scaler.get_scale())
                amp_scale = (
                    new_amp_scale if new_amp_scale is not None else old_amp_scale
                )
                optimizer_skipped = (
                    scaler.is_enabled()
                    and old_amp_scale is not None
                    and new_amp_scale is not None
                    and new_amp_scale < old_amp_scale
                )
                did_update = not optimizer_skipped
                if did_update:
                    self.lr_scheduler.step()
            else:
                optimizer_skipped = True
                if scaler.is_enabled() and old_amp_scale is not None:
                    # Finish/reset GradScaler's per-optimizer state on every
                    # rank. Supplying the same explicit backoff avoids scale
                    # divergence when only one rank owned the non-finite shard.
                    new_amp_scale = old_amp_scale * scaler.get_backoff_factor()
                    scaler.update(new_scale=new_amp_scale)
                    amp_scale = new_amp_scale
            self.optimizer.zero_grad(set_to_none=True)
            self._skip_accumulated_update = False
            optimizer_seconds = time.perf_counter() - optimizer_start

        self.benchmark_phase_transition("post_train_step")

        for key, value in loss_dict.items():
            if "loss" not in key:
                continue
            self.train_metric_wrapper.add(key, value, namespace="train")
        if should_empty_cache(self.step + 1, self.configs.empty_cache_interval):
            torch.cuda.empty_cache()
        return {
            "loss_nan": loss_nan,
            "loss_nonfinite_in_window": loss_nonfinite_in_window,
            "loss": loss.detach(),
            "loss_dict": loss_dict,
            "grad_norm": grad_norm,
            "grad_clipped": grad_clipped,
            "nonfinite_grad": nonfinite_grad,
            "optimizer_skipped": optimizer_skipped,
            "did_update": did_update,
            "amp_scale": amp_scale,
            "N_cycle": model_log_dict.get("N_cycle"),
            "post_permutation_diffusion_supervised_tokens": (
                model_log_dict.get(
                    "minirollout_perm/N_diffusion_supervised_token"
                )
            ),
            "chain_permutation_min_supervision_rejected": (
                model_log_dict.get(
                    "minirollout_perm/Chain-min_supervision_rejected",
                    0.0,
                )
            ),
            "atom_permutation_min_supervision_rejected": (
                model_log_dict.get(
                    "minirollout_perm/Atom-min_supervision_rejected",
                    0.0,
                )
            ),
            "msa_depth_trace": model_log_dict.get("msa_depth_trace", []),
            "timings": {
                "forward_seconds": forward_seconds,
                "loss_seconds": loss_seconds,
                "loss_health_sync_seconds": loss_health_sync_seconds,
                "backward_seconds": backward_seconds,
                "optimizer_seconds": optimizer_seconds,
                "train_step_seconds": time.perf_counter() - step_start,
            },
        }

    def progress_bar(self, desc: str = "") -> None:
        """
        Update the training progress bar on the master rank.

        Args:
            desc (str): Description to display on the progress bar.
        """
        if DIST_WRAPPER.rank != 0:
            return
        eval_steps = get_progress_bar_steps(
            self.configs.eval_interval,
            self.iters_to_accumulate,
        )
        if self.global_step % eval_steps == 0 or (not hasattr(self, "_ipbar")):
            # Start a new progress bar
            self._pbar = tqdm(
                range(
                    self.global_step % eval_steps,
                    eval_steps,
                )
            )
            self._ipbar = iter(self._pbar)

        try:
            step = next(self._ipbar)
            self._pbar.set_description(
                f"[step {self.step}: {step}/{eval_steps}] {desc}"
            )
        except StopIteration:
            # If the iterator is exhausted, just return
            pass

    def run(self) -> None:
        try:
            self._run_impl()
        except KeyboardInterrupt as exc:
            self.update_run_status(
                status="failed",
                exit_reason="interrupted",
                exception=exc,
                write_heartbeat=True,
            )
            raise
        except Exception as exc:
            self.update_run_status(
                status="failed",
                exit_reason="exception",
                exception=exc,
                write_heartbeat=True,
            )
            raise

    def _run_impl(self) -> None:
        """
        Main entry point for the AF3Trainer.
        Handles the complete training cycle including evaluation, logging, and checkpointing.
        """
        if self.configs.eval_only or self.configs.eval_first:
            self.evaluate()
            if self.configs.eval_only:
                self.update_run_status(
                    status="completed",
                    exit_reason="eval_only_completed",
                    write_heartbeat=True,
                )
                return
        if self.step >= self.configs.max_steps:
            raise ValueError("No training updates remain before max_steps")
        use_ema = hasattr(self, "ema_wrapper")
        self.print(f"Using EMA: {use_ema}")

        while True:
            for batch, data_wait_seconds in iter_dataloader_with_timing(
                self.train_dl
            ):
                self.maybe_start_benchmark_window()
                step_wall_start = time.perf_counter()
                is_update_step = (self.global_step + 1) % self.iters_to_accumulate == 0
                is_last_step = (self.step + 1) == self.configs.max_steps
                step_need_log = (
                    self.step == self.start_step
                    or (self.step + 1) % self.configs.log_interval == 0
                )

                step_need_eval = (
                    self.configs.eval_interval > 0
                    and (self.step + 1) % self.configs.eval_interval == 0
                )
                step_need_save = (
                    self.configs.checkpoint_interval > 0
                    and (self.step + 1) % self.configs.checkpoint_interval == 0
                )

                is_last_step &= is_update_step
                step_need_log &= is_update_step
                step_need_eval &= is_update_step
                step_need_save &= is_update_step

                self.benchmark_phase_memory.start_step(
                    step=self.step,
                    global_step=self.global_step,
                    is_update_step=is_update_step,
                )
                to_device_start = time.perf_counter()
                batch = to_device(batch, self.device)
                to_device_seconds = time.perf_counter() - to_device_start
                self.benchmark_phase_transition("forward_prelude")
                batch_summary = summarize_batch(batch)
                self.benchmark_phase_memory.set_sample(batch_summary)
                self.record_benchmark_run_sample(batch_summary)
                self.log_msa_debug_batch(
                    batch=batch,
                    batch_summary=batch_summary,
                    mode="train",
                )
                self.progress_bar()
                force_ddp_first_backward_sync = (
                    self.use_ddp and self._ddp_first_backward_sync_pending
                )
                with gradient_accumulation_sync_context(
                    self.model,
                    use_ddp=self.use_ddp,
                    global_step=self.global_step,
                    iters_to_accumulate=self.iters_to_accumulate,
                    force_sync=force_ddp_first_backward_sync,
                ):
                    train_info = self.train_step(batch)
                if force_ddp_first_backward_sync:
                    # This state belongs to the newly constructed DDP reducer,
                    # not to the training step/checkpoint. A resumed process
                    # creates a new reducer and must perform this once again.
                    self._ddp_first_backward_sync_pending = False
                    if not is_update_step:
                        self.print(
                            "DDP static-graph first backward synchronization "
                            "completed; enabling no_sync accumulation"
                        )
                train_info["ddp_gradient_sync"] = (
                    not self.use_ddp
                    or is_update_step
                    or force_ddp_first_backward_sync
                )
                train_info["ddp_forced_first_backward_sync"] = (
                    force_ddp_first_backward_sync and not is_update_step
                )
                self.record_benchmark_msa_depth_trace(
                    train_info.get("msa_depth_trace", [])
                )
                self.record_benchmark_batch(batch_summary, train_info)
                self.train_dataset_window_stats.add(
                    batch_summary=batch_summary,
                    loss_dict=train_info["loss_dict"],
                )
                train_log_info = {
                    key: value for key, value in train_info.items() if key != "loss_dict"
                }
                ema_seconds = 0.0
                self.benchmark_phase_transition("ema")
                if use_ema and train_info["did_update"]:
                    ema_start = time.perf_counter()
                    self.ema_wrapper.update()
                    ema_seconds = time.perf_counter() - ema_start
                self.benchmark_phase_transition("post_ema")
                self.benchmark_phase_memory.end_step()
                self.maybe_finish_benchmark_window(is_update_step=is_update_step)
                step_wall_seconds = time.perf_counter() - step_wall_start
                cuda_memory_stats = get_cuda_memory_stats(self.device)
                self.train_runtime_window_stats.add(
                    rank=DIST_WRAPPER.rank,
                    step=self.step,
                    global_step=self.global_step,
                    batch_summary=batch_summary,
                    train_info=train_info,
                    to_device_seconds=to_device_seconds,
                    ema_seconds=ema_seconds,
                    step_wall_seconds=step_wall_seconds,
                    cuda_memory=cuda_memory_stats,
                    data_wait_seconds=data_wait_seconds,
                )

                if step_need_log or is_last_step:
                    metrics = self.train_metric_wrapper.calc()
                    dataset_window_stats = self.train_dataset_window_stats.calc()
                    runtime_window_stats = self.train_runtime_window_stats.calc()
                    self.print(f"Step {self.step} train metrics: {metrics}")
                    last_lr = self.lr_scheduler.get_last_lr()
                    append_jsonl(
                        self.data_samples_log_path,
                        {
                            "event": "data_sample",
                            "step": self.step,
                            "global_step": self.global_step,
                            "rank": DIST_WRAPPER.rank,
                            "world_size": DIST_WRAPPER.world_size,
                            "sample": batch_summary,
                        },
                    )
                    append_jsonl(
                        self.runtime_log_path,
                        {
                            "event": "runtime",
                            "step": self.step,
                            "global_step": self.global_step,
                            "rank": DIST_WRAPPER.rank,
                            "world_size": DIST_WRAPPER.world_size,
                            "is_update_step": is_update_step,
                            "sample": batch_summary,
                            "data_wait_seconds": data_wait_seconds,
                            "to_device_seconds": to_device_seconds,
                            "ema_seconds": ema_seconds,
                            "step_wall_seconds": step_wall_seconds,
                            "train": train_log_info,
                            "cuda_memory": cuda_memory_stats,
                        },
                    )
                    if DIST_WRAPPER.rank == 0:
                        append_jsonl(
                            self.train_metrics_log_path,
                            {
                                "event": "train_metrics",
                                "step": self.step,
                                "global_step": self.global_step,
                                "rank": DIST_WRAPPER.rank,
                                "world_size": DIST_WRAPPER.world_size,
                                "sample": batch_summary,
                                "metrics": metrics,
                                "dataset_window": dataset_window_stats,
                                "runtime_window": runtime_window_stats,
                                "lr": last_lr,
                                "loss_nan": train_log_info["loss_nan"],
                                "grad_norm": train_log_info["grad_norm"],
                                "ema_enabled": use_ema,
                                "ema_decay": (
                                    self.ema_wrapper.decay if use_ema else None
                                ),
                            },
                        )
                        self.update_run_status(
                            status="running",
                            exit_reason="heartbeat",
                            write_heartbeat=True,
                        )
                        if self.configs.use_wandb:
                            log_training_metrics(wandb.run, metrics, self.step, last_lr)
                        self.print(f"Step {self.step}, learning rate: {last_lr}")

                if step_need_save or (
                    is_last_step and not self.configs.skip_final_checkpoint
                ):
                    self.save_checkpoint()
                    if use_ema:
                        self.ema_wrapper.apply_shadow()
                        try:
                            self.save_checkpoint(
                                ema_suffix=f"_ema_{self.ema_wrapper.decay}"
                            )
                        finally:
                            self.ema_wrapper.restore()

                if should_run_evaluation(
                    step_need_eval,
                    is_last_step,
                    self.configs.eval_at_end,
                ):
                    self.evaluate()

                self.global_step += 1
                if self.global_step % self.iters_to_accumulate == 0:
                    self.step += 1

                if self.step >= self.configs.max_steps:
                    self.print(f"Finished training at step {self.step - 1}")
                    break
            if self.step >= self.configs.max_steps:
                break
        if self.benchmark_enabled and not self.benchmark_window_completed:
            raise RuntimeError(
                "Training ended before the configured benchmark window completed"
            )
        self.update_run_status(
            status="completed",
            exit_reason="max_steps_reached",
            write_heartbeat=True,
        )


def main() -> None:
    """
    Parse configurations and start the training process.
    """
    log_format = (
        "%(asctime)s,%(msecs)-3d %(levelname)-8s "
        "[%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    )
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    configs_base["triangle_attention"] = os.environ.get(
        "TRIANGLE_ATTENTION", "cuequivariance"
    )
    configs_base["triangle_multiplicative"] = os.environ.get(
        "TRIANGLE_MULTIPLICATIVE", "cuequivariance"
    )
    arg_str = parse_sys_args()
    configs = {**configs_base, **{"data": data_configs}}
    # 1. First pass to get model_name
    configs = parse_configs(
        configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    model_name = configs.model_name

    # 2. Get model specifics and merge into base defaults
    base_configs = {**configs_base, **{"data": data_configs}}
    model_specfics_configs = model_configs[model_name]

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    deep_update(base_configs, model_specfics_configs)

    # 3. Second pass to apply sys_args with higher priority
    configs = parse_configs(
        configs=base_configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    logging.info(
        f"Using params for model {model_name}: "
        f"cycle={configs.model.N_cycle}, step={configs.sample_diffusion.N_step}"
    )
    print(f"Run Name: {configs.run_name}")
    trainer = AF3Trainer(configs)
    trainer.run()


if __name__ == "__main__":
    main()
