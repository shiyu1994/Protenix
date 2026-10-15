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

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import runner.train as train


SOURCE_SETS = list(train.CATALOG_REPLACEMENT_SOURCE_TRAIN_SETS_V1)
TARGET_SETS = list(train.CATALOG_REPLACEMENT_TARGET_TRAIN_SETS_V1)
MASSES = list(train.CATALOG_REPLACEMENT_MASSES_V1)
SOURCE_SOFT = list(train.CATALOG_REPLACEMENT_SOURCE_SOFT_DATASETS_V1)
TARGET_SOFT = list(train.CATALOG_REPLACEMENT_TARGET_SOFT_DATASETS_V1)
SOURCE_RUN_ID = "source_run_v11"
SOURCE_TRAINING_CONTRACT = "source-contract-v11"
TARGET_TRAINING_CONTRACT = "target-catalog-contract-v2"
ARCHITECTURE_SHA256 = "a" * 64
REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "migrate_step21399_to_residual_ge200_softcluster_g02.sh"


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _semantics(dataset_names: list[str]) -> tuple[dict, dict, dict]:
    augmentation = {}
    runtime = {}
    loss = {}
    for name in dataset_names:
        augmentation[name] = {
            "query_only_msa_probability": 0.1,
            "pair_msa_as_unpaired_probability": 0.3,
        }
        runtime[name] = {
            "stage": "train",
            "dataset_config": {
                "base_info": {
                    "runtime_index_dir": f"/runtime/{name}",
                    "require_resolved_protein_crop": True,
                },
                "msa": {"max_input_unpaired_msa_depth": 16384},
            },
            "effective_msa": {
                "dataset_name": name,
                "max_input_unpaired_msa_depth": 16384,
            },
            "effective_template": {"dataset_name": name},
        }
        loss[name] = {
            "is_distillation": True,
            "require_resolved_protein_crop": True,
            "min_diffusion_supervised_tokens": 4,
        }
    return augmentation, runtime, loss


def _data_contract(train_sets: list[str], profile_id: str) -> dict:
    soft = train_sets[1:3]
    soft_augmentation, soft_runtime, soft_loss = _semantics(soft)
    unchanged_augmentation, unchanged_runtime, unchanged_loss = _semantics(
        [train_sets[0], train_sets[3]]
    )
    augmentation_by_dataset = {
        **unchanged_augmentation,
        **soft_augmentation,
    }
    runtime_by_dataset = {**unchanged_runtime, **soft_runtime}
    loss_by_dataset = {**unchanged_loss, **soft_loss}
    # Preserve the actual PDB/non-distillation difference while keeping it
    # identical across source and target contracts.
    loss_by_dataset[train_sets[0]]["is_distillation"] = False
    payload = {
        "schema_version": 7,
        "profile_id": profile_id,
        "train_sets": train_sets,
        "train_sample_weights": MASSES,
        "epoch_size": 10_240,
        "sampler_seed": -1,
        "sampler_replacement": True,
        "effective_sampler_seed": 42,
        "sampler_seed_resolution": "negative_or_none_inherits_global_seed_v1",
        "augmentation": {
            "decision_version": "sample_occurrence_sha256_v1",
            "query_only_msa_probability": 0.1,
            "pair_msa_as_unpaired_probability": 0.3,
            "effective_by_dataset": augmentation_by_dataset,
        },
        "dataset_runtime_semantics": {
            "effective_by_dataset": runtime_by_dataset,
        },
        "loss_and_crop_policy": {"effective_by_dataset": loss_by_dataset},
        "runtime_distillation_indices": {
            name: {
                "runtime_index_dir": f"/runtime/{name}",
                "manifest_sha256": _canonical_sha256({"name": name}),
            }
            for name in soft
        },
    }
    return {**payload, "sha256": _canonical_sha256(payload)}


def _dataset_binding(train_sets: list[str], sizes: list[int]) -> dict:
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    children = [
        {"kind": "weighted", "size": sizes[0], "sha256": "1" * 64},
        {"kind": "clustered", "size": sizes[1], "sha256": "2" * 64},
        {"kind": "clustered", "size": sizes[2], "sha256": "3" * 64},
        {"kind": "weighted", "size": sizes[3], "sha256": "4" * 64},
    ]
    return {
        "flat_row_count": offsets[-1],
        "train_sets": train_sets,
        "dataset_offsets": offsets,
        "dataset_masses": MASSES,
        "child_fingerprints": children,
    }


def _write_source_run(tmp_path: Path, source_contract: dict) -> Path:
    run_dir = tmp_path / SOURCE_RUN_ID
    manifest_dir = run_dir / "manifest"
    manifest_dir.mkdir(parents=True)
    document = {
        "run_name": SOURCE_RUN_ID,
        "data": {"contract": source_contract},
    }
    (manifest_dir / "run_manifest.json").write_text(
        json.dumps(document, sort_keys=True),
        encoding="utf-8",
    )
    return run_dir


def _loaded_manifest(tmp_path: Path):
    source_contract = _data_contract(SOURCE_SETS, "source-profile")
    run_dir = _write_source_run(tmp_path, source_contract)
    run_manifest_path = run_dir / "manifest" / "run_manifest.json"
    source_binding = _dataset_binding(SOURCE_SETS, [10, 20, 30, 40])
    target_binding = _dataset_binding(TARGET_SETS, [10, 25, 35, 40])
    source = {
        "mode": "checkpoint_migration",
        "raw_checkpoint_path": str(run_dir / "checkpoints" / "4.pt"),
        "raw_checkpoint_sha256": "b" * 64,
        "ema_checkpoint_path": str(run_dir / "checkpoints" / "4_ema_0.999.pt"),
        "ema_checkpoint_sha256": "c" * 64,
        "checkpoint_step": 4,
        "completed_updates": 5,
        "next_global_step": 20,
        "origin_consumed_global_batches_per_rank": 20,
        "iters_to_accumulate": 4,
        "checkpoint_pair_id": "d" * 64,
        "source_training_contract_id": SOURCE_TRAINING_CONTRACT,
        "source_data_contract_sha256": source_contract["sha256"],
        "source_run_id": SOURCE_RUN_ID,
        "model_architecture_sha256": ARCHITECTURE_SHA256,
    }
    catalog = {
        "format": train.CATALOG_REPLACEMENT_FORMAT_V1,
        "allowed_delta": train.CATALOG_REPLACEMENT_ALLOWED_DELTA_V1,
        "source_dataset": {
            "binding": source_binding,
            "binding_sha256": _canonical_sha256(source_binding),
        },
        "target_dataset": {
            "binding": target_binding,
            "binding_sha256": _canonical_sha256(target_binding),
        },
        "crosswalk": {
            "format": "protenix.global_id_history_projection.v1",
            "global_id_policy": "source_prefix_pipe_raw_id_v1",
            "exposure_policy": "occurrence_sum_v1",
            "ambiguous_alternate_policy": "seen_floor_1_v1",
            "source_soft_datasets": SOURCE_SOFT,
            "target_soft_datasets": TARGET_SOFT,
            "lookup_bindings": {
                name: {
                    "runtime_manifest_path": f"/runtime/{name}/manifest.json",
                    "runtime_manifest_sha256": _canonical_sha256(
                        {"name": name}
                    ),
                }
                for name in TARGET_SOFT
            },
            "intentional_missing_binding": None,
            "audit_artifact": "catalog_migration.audit",
            "accounting": {},
        },
    }
    manifest = {
        "schema_version": 2,
        "dataset": target_binding,
        "algorithm": {"soft_cluster_datasets": TARGET_SOFT},
        "source_checkpoint": source,
        "source_run": {
            "run_dir": str(run_dir),
            "run_name": SOURCE_RUN_ID,
            "run_manifest_path": str(run_manifest_path),
            "run_manifest_sha256": hashlib.sha256(
                run_manifest_path.read_bytes()
            ).hexdigest(),
        },
        "catalog_migration": catalog,
    }
    schedule_contract = {
        "dataset_binding_sha256": _canonical_sha256(target_binding),
        "manifest_sha256": "e" * 64,
    }
    return source_contract, manifest, schedule_contract


def _configs(raw_path: str, ema_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        load_checkpoint_path=raw_path,
        load_ema_checkpoint_path=ema_path,
        iters_to_accumulate=4,
        distributed_strategy="ddp",
        optimizer_sharding="none",
        dtype="bf16",
        ema_decay=0.999,
        ema_shadow_device="model",
        load_strict=True,
        load_params_only=False,
        skip_load_optimizer=False,
        skip_load_scheduler=False,
        skip_load_step=False,
        load_step_for_scheduler=False,
        data=SimpleNamespace(
            num_dl_workers=0,
            train_dl_persistent_workers=False,
            train_dl_prefetch_factor=2,
            train_dl_pin_memory=False,
            train_sampler=SimpleNamespace(
                soft_cluster_schedule_manifest="/schedule/manifest.json"
            )
        ),
    )


@pytest.fixture(autouse=True)
def _migration_environment(monkeypatch):
    monkeypatch.setenv(
        "PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION",
        train.STRICT_RESUME_SOFT_CLUSTER_CATALOG_REPLACEMENT_MIGRATION_V2,
    )
    monkeypatch.setenv("PROTENIX_RESUME_CHECKPOINT_STEP", "4")
    monkeypatch.setenv("PROTENIX_TRAINING_CONTRACT_ID", TARGET_TRAINING_CONTRACT)
    monkeypatch.setenv(train.SOFT_CLUSTER_SCHEDULE_DIR_ENV, "/schedule")


def _patch_runtime(monkeypatch, tmp_path: Path):
    source_contract, manifest, schedule_contract = _loaded_manifest(tmp_path)
    target_contract = _data_contract(TARGET_SETS, "target-profile")
    target_payload = dict(target_contract)
    target_payload.pop("sha256")
    target_payload["soft_cluster_schedule"] = schedule_contract
    target_contract = {
        **target_payload,
        "sha256": _canonical_sha256(target_payload),
    }
    monkeypatch.setattr(
        train,
        "_load_soft_cluster_schedule_migration_manifest",
        lambda: {
            "manifest": copy.deepcopy(manifest),
            "manifest_path": "/schedule/manifest.json",
            "manifest_sha256": "e" * 64,
            "schedule_contract": copy.deepcopy(schedule_contract),
        },
    )
    monkeypatch.setattr(
        train,
        "build_training_data_contract",
        lambda configs: copy.deepcopy(target_contract),
    )
    monkeypatch.setattr(
        train,
        "build_model_architecture_contract",
        lambda configs: {"schema_version": 1, "sha256": ARCHITECTURE_SHA256},
    )
    source = manifest["source_checkpoint"]
    configs = _configs(
        source["raw_checkpoint_path"],
        source["ema_checkpoint_path"],
    )
    return configs, source_contract, target_contract


def test_v2_transition_binds_source_and_target_catalogs(monkeypatch, tmp_path):
    configs, source_contract, target_contract = _patch_runtime(
        monkeypatch, tmp_path
    )

    migration = train._strict_resume_migration_expectations(configs)

    assert migration["mode"] == (
        train.STRICT_RESUME_SOFT_CLUSTER_CATALOG_REPLACEMENT_MIGRATION_V2
    )
    assert migration["allowed_delta"] == (
        train.CATALOG_REPLACEMENT_ALLOWED_DELTA_V1
    )
    assert migration["source_data_contract"] == source_contract
    assert migration["target_data_contract"] == target_contract
    assert migration["source_dataset_binding"]["train_sets"] == SOURCE_SETS
    assert migration["target_dataset_binding"]["train_sets"] == TARGET_SETS


def test_v2_transition_rejects_non_catalog_policy_delta(monkeypatch, tmp_path):
    configs, _, target_contract = _patch_runtime(monkeypatch, tmp_path)
    changed = copy.deepcopy(target_contract)
    changed["augmentation"]["query_only_msa_probability"] = 0.2
    changed_payload = dict(changed)
    changed_payload.pop("sha256")
    changed["sha256"] = _canonical_sha256(changed_payload)
    monkeypatch.setattr(
        train,
        "build_training_data_contract",
        lambda configs: copy.deepcopy(changed),
    )

    with pytest.raises(ValueError, match="unauthorized non-catalog"):
        train._strict_resume_migration_expectations(configs)


def test_v2_checkpoint_validation_requires_every_full_state_component(
    monkeypatch,
    tmp_path,
):
    configs, source_contract, _ = _patch_runtime(monkeypatch, tmp_path)
    migration = train._strict_resume_migration_expectations(configs)
    monkeypatch.setattr(
        train,
        "_strict_resume_migration_expectations",
        lambda configs: copy.deepcopy(migration),
    )
    monkeypatch.setattr(train.DIST_WRAPPER, "world_size", 1)
    architecture = {"schema_version": 1, "sha256": ARCHITECTURE_SHA256}
    resume_contract = {
        "schema_version": train.STRICT_FULL_RESUME_SCHEMA_VERSION,
        "training_contract_id": SOURCE_TRAINING_CONTRACT,
        "run_id": SOURCE_RUN_ID,
        "step": 4,
        "global_step": 19,
        "next_step": 5,
        "next_global_step": 20,
        "iters_to_accumulate": 4,
        "world_size": 1,
        "distributed_strategy": "ddp",
        "dtype": "bf16",
        "grad_scaler_enabled": False,
        "ema_decay": 0.999,
        "ema_shadow_device": "model",
        "data_num_workers": 0,
        "data_train_dl_persistent_workers": False,
        "data_train_dl_prefetch_factor": None,
        "data_train_dl_pin_memory": False,
        "data_profile_id": source_contract["profile_id"],
        "data_contract_sha256": source_contract["sha256"],
        "data_train_sets": source_contract["train_sets"],
        "data_train_sample_weights": source_contract["train_sample_weights"],
        "data_epoch_size": source_contract["epoch_size"],
        "data_sampler_seed": source_contract["sampler_seed"],
        "data_sampler_replacement": source_contract["sampler_replacement"],
        "data_augmentation": source_contract["augmentation"],
        "model_architecture_contract": architecture,
        "model_architecture_sha256": ARCHITECTURE_SHA256,
        "data_worker_rng_state_checkpointed": True,
        "data_resume_scope": "sampler_cursor_and_rank_main_process_rng",
    }
    resume_contract["checkpoint_pair_id"] = _canonical_sha256(resume_contract)
    # The migration manifest pins this pair id; update the test expectation to
    # the self-authenticating resume contract produced above.
    migration["source_checkpoint_pair_id"] = resume_contract[
        "checkpoint_pair_id"
    ]
    monkeypatch.setattr(
        train,
        "_strict_resume_migration_expectations",
        lambda configs: copy.deepcopy(migration),
    )
    checkpoint = {
        "format_version": 2,
        "checkpoint_kind": "raw",
        "model": {"weight": torch.ones(1)},
        "optimizer": {
            "state": {0: {"exp_avg": torch.ones(1)}},
            "param_groups": [{"params": [0]}],
        },
        "scheduler": {"last_epoch": 4},
        "grad_scaler": {},
        "step": 4,
        "resume_contract": resume_contract,
        "rank_runtime_states": {
            "schema_version": 1,
            "world_size": 1,
            "states": [
                {
                    "rank": 0,
                    "rng": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": None,
                    },
                    "train_dataloader": {
                        "schema_version": 1,
                        "sampler_epoch": 0,
                        "next_local_offset": 20,
                        "consumed_global_batches": 20,
                        "batch_size": 1,
                        "num_workers": 0,
                        "sampler": {"source": True},
                    },
                }
            ],
        },
    }
    validated = train.validate_strict_checkpoint_payload(
        checkpoint,
        checkpoint_path=configs.load_checkpoint_path,
        expected_kind="raw",
        configs=configs,
        grad_scaler_enabled=False,
        device=torch.device("cpu"),
    )
    assert validated["next_global_step"] == 20

    checkpoint.pop("optimizer")
    with pytest.raises(ValueError, match="optimizer moments"):
        train.validate_strict_checkpoint_payload(
            checkpoint,
            checkpoint_path=configs.load_checkpoint_path,
            expected_kind="raw",
            configs=configs,
            grad_scaler_enabled=False,
            device=torch.device("cpu"),
        )


def test_v2_mode_is_required_for_schema_v2_manifest(monkeypatch, tmp_path):
    configs, _, _ = _patch_runtime(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION",
        train.STRICT_RESUME_SOFT_CLUSTER_SCHEDULE_MIGRATION_V1,
    )

    with pytest.raises(ValueError, match="schema_version=1"):
        train._strict_resume_migration_expectations(configs)


def test_formal_launcher_locks_full_state_and_target_profile(tmp_path):
    schedule_dir = tmp_path / "schedule"
    schedule_dir.mkdir()
    env = {
        key: os.environ[key]
        for key in ("HOME", "PATH", "LANG", "LC_ALL")
        if key in os.environ
    }
    env.update(
        {
            "DRY_RUN": "1",
            "SKIP_PATH_CHECK": "1",
            "MLP_WORKER_NUM": "16",
            "MLP_ROLE_INDEX": "0",
            "MLP_WORKER_0_HOST": "worker-0",
            "MLP_WORKER_0_PORT": "23456",
            "NPROC_PER_NODE": "8",
            "RUN_NAME": "v9_straggler_residual_ge200_softcluster_g02_test",
            "BASE_DIR": str(tmp_path / "output"),
            "SOFT_CLUSTER_SCHEDULE_DIR": str(schedule_dir),
            "PROTENIX_LEGACY_RESIDUAL_RUNTIME_INDEX_DIR": str(
                tmp_path / "residual"
            ),
            "PROTENIX_GE200_STRUCTURAL_RUNTIME_INDEX_DIR": str(
                tmp_path / "ge200"
            ),
        }
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    command_line = next(
        line for line in result.stdout.splitlines() if line.startswith("torchrun ")
    )
    command = shlex.split(command_line)

    def arg(name: str) -> str:
        return command[command.index(name) + 1]

    assert arg("--load_params_only") == "false"
    assert arg("--skip_load_optimizer") == "false"
    assert arg("--skip_load_scheduler") == "false"
    assert arg("--skip_load_step") == "false"
    assert arg("--max_steps") == "50000"
    assert arg("--data.train_sampler.soft_cluster_gamma") == "0.2"
    assert arg("--data.train_sets") == ",".join(TARGET_SETS)
    assert arg("--data.train_sampler.train_sample_weights") == ",".join(
        str(value) for value in MASSES
    )
    assert "source_step=21399 target_start_step=21400" in result.stdout
    assert "restored=raw_model+matched_ema+optimizer+scheduler+step+" in (
        result.stdout
    )
