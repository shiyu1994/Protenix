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

"""Fail-closed runtime support for precomputed soft-cluster schedules.

The schedule is deliberately an immutable external artifact rather than mutable
sampler state.  Entries are stored in synchronized-wave, rank-major order::

    offset = (absolute_consumed_batches - origin_consumed_batches) * world_size + rank

This representation remains unambiguous when ``epoch_size`` is not divisible by
``world_size``.  A rank's sampler epoch and local offset are only a view onto the
absolute per-rank consumed-batch counter.

The manifest, completion receipt, and schedule file are authenticated before a
single row is exposed.  Checkpoints retain only a compact signature and cursor;
the large exposure state used to build the schedule never enters per-rank
checkpoint state.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import threading
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler

from protenix.data.pipeline.sample_context import SampleOccurrence


SOFT_CLUSTER_SCHEDULE_SCHEMA_VERSION = 1
SOFT_CLUSTER_SCHEDULE_SCHEMA_VERSION_V2 = 2
SOFT_CLUSTER_SCHEDULE_SUCCESS_SCHEMA_VERSION = 1
SOFT_CLUSTER_SCHEDULE_FORMAT = "protenix.soft_cluster_schedule.v1"
SOFT_CLUSTER_SCHEDULE_FORMAT_V2 = "protenix.soft_cluster_schedule.v2"
SOFT_CLUSTER_SCHEDULE_LAYOUT = "wave_major_rank_v1"
SOFT_CLUSTER_SCHEDULE_DTYPE = "<u4"
SOFT_CLUSTER_SCHEDULE_MANIFEST_FILENAME = "manifest.json"
SOFT_CLUSTER_SCHEDULE_SUCCESS_FILENAME = "_SUCCESS.json"
SOFT_CLUSTER_SCHEDULE_ROWS_FILENAME = "global_rows.u32"

SOFT_CLUSTER_ALGORITHM_FORMAT = "protenix.soft_exposure_bucket_sampler.v1"
SOFT_CLUSTER_CONSUMPTION_ORDER = (
    "synchronized_microstep_then_rank_ascending_v1"
)
SOFT_CLUSTER_MEMBER_POLICY = "deterministic_rotation_v1"
SOFT_CLUSTER_SAMPLER_LAYOUT = (
    "soft_cluster_schedule_strided_unpadded_v1"
)
LEGACY_HIERARCHICAL_SAMPLER_LAYOUT = "hierarchical_strided_unpadded_v1"

SOFT_CLUSTER_CATALOG_MIGRATION_FORMAT = "protenix.catalog_replacement.v1"
SOFT_CLUSTER_CATALOG_MIGRATION_ALLOWED_DELTA = (
    "legacy_short_long_to_legacy_residual_ge200_v1"
)
SOFT_CLUSTER_CROSSWALK_FORMAT = "protenix.global_id_history_projection.v1"
SOFT_CLUSTER_GLOBAL_ID_POLICY = "source_prefix_pipe_raw_id_v1"
SOFT_CLUSTER_EXPOSURE_POLICY = "occurrence_sum_v1"
SOFT_CLUSTER_AMBIGUOUS_POLICY = "seen_floor_1_v1"

_SHA256_HEX_LENGTH = 64
_UINT32_ITEMSIZE = np.dtype(SOFT_CLUSTER_SCHEDULE_DTYPE).itemsize


class SoftClusterScheduleError(ValueError):
    """Raised when a schedule artifact or binding is invalid."""


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _require_int(
    value: Any,
    *,
    label: str,
    minimum: Optional[int] = None,
) -> int:
    if not _is_int(value):
        raise SoftClusterScheduleError(f"{label} must be an integer, got {value!r}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise SoftClusterScheduleError(
            f"{label} must be >= {minimum}, got {result}"
        )
    return result


def _require_float(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise SoftClusterScheduleError(f"{label} must be numeric, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SoftClusterScheduleError(
            f"{label} must be numeric, got {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise SoftClusterScheduleError(f"{label} must be finite, got {result!r}")
    return result


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SoftClusterScheduleError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    label: str,
) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise SoftClusterScheduleError(
            f"{label} is missing fields: {sorted(missing)}"
        )
    if unknown:
        raise SoftClusterScheduleError(
            f"{label} has unsupported fields: {sorted(unknown)}"
        )


def _require_nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SoftClusterScheduleError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    text = _require_nonempty_string(value, label=label)
    if len(text) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise SoftClusterScheduleError(
            f"{label} must be a lowercase 64-character SHA-256"
        )
    return text


def _require_real_absolute_path(value: Any, *, label: str) -> str:
    path = _require_nonempty_string(value, label=label)
    if not os.path.isabs(path) or os.path.realpath(path) != path:
        raise SoftClusterScheduleError(
            f"{label} must be an absolute normalized real path, got {path!r}"
        )
    return path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise SoftClusterScheduleError(f"cannot hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _load_json_bytes(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise SoftClusterScheduleError(f"cannot read {label} {path}: {exc}") from exc
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SoftClusterScheduleError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise SoftClusterScheduleError(f"{label} must contain one JSON object")
    return document, payload


def _safe_artifact_path(root: Path, relative_path: Any, *, label: str) -> Path:
    relative_text = _require_nonempty_string(relative_path, label=label)
    pure = PurePosixPath(relative_text)
    if pure.is_absolute() or not pure.parts or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise SoftClusterScheduleError(
            f"{label} must be a safe relative POSIX path, got {relative_text!r}"
        )
    root_real = root.resolve()
    candidate = root_real.joinpath(*pure.parts).resolve()
    try:
        candidate.relative_to(root_real)
    except ValueError as exc:
        raise SoftClusterScheduleError(
            f"{label} escapes schedule directory: {relative_text!r}"
        ) from exc
    if not candidate.is_file():
        raise SoftClusterScheduleError(f"required artifact is missing: {candidate}")
    return candidate


def rank_local_epoch_size(epoch_size: int, world_size: int, rank: int) -> int:
    """Return the actual unpadded strided sample count for one rank."""

    epoch_size = _require_int(epoch_size, label="epoch_size", minimum=1)
    world_size = _require_int(world_size, label="world_size", minimum=1)
    rank = _require_int(rank, label="rank", minimum=0)
    if rank >= world_size:
        raise SoftClusterScheduleError(
            f"rank must be inside [0, {world_size}), got {rank}"
        )
    if rank >= epoch_size:
        return 0
    return 1 + (epoch_size - 1 - rank) // world_size


def sampler_cursor_from_consumed_batches(
    consumed_global_batches: int,
    *,
    epoch_size: int,
    world_size: int,
    rank: int,
) -> tuple[int, int]:
    """Map a synchronized per-rank batch counter to epoch/local offset."""

    consumed = _require_int(
        consumed_global_batches,
        label="consumed_global_batches",
        minimum=0,
    )
    local_size = rank_local_epoch_size(epoch_size, world_size, rank)
    if local_size <= 0:
        raise SoftClusterScheduleError(
            "epoch_size must be >= world_size for resumable distributed training"
        )
    return divmod(consumed, local_size)


def consumed_batches_from_sampler_cursor(
    sampler_epoch: int,
    next_local_offset: int,
    *,
    epoch_size: int,
    world_size: int,
    rank: int,
) -> int:
    """Inverse of :func:`sampler_cursor_from_consumed_batches`."""

    epoch = _require_int(sampler_epoch, label="sampler_epoch", minimum=0)
    offset = _require_int(next_local_offset, label="next_local_offset", minimum=0)
    local_size = rank_local_epoch_size(epoch_size, world_size, rank)
    if local_size <= 0:
        raise SoftClusterScheduleError(
            "epoch_size must be >= world_size for resumable distributed training"
        )
    if offset >= local_size:
        raise SoftClusterScheduleError(
            f"next_local_offset must be inside [0, {local_size}), got {offset}"
        )
    return epoch * local_size + offset


def sampler_global_position(
    next_local_offset: int,
    *,
    epoch_size: int,
    world_size: int,
    rank: int,
) -> int:
    """Map one rank-local offset to the historical strided global position."""

    offset = _require_int(next_local_offset, label="next_local_offset", minimum=0)
    local_size = rank_local_epoch_size(epoch_size, world_size, rank)
    if offset >= local_size:
        raise SoftClusterScheduleError(
            f"next_local_offset must be inside [0, {local_size}), got {offset}"
        )
    position = rank + offset * world_size
    if position >= epoch_size:
        raise SoftClusterScheduleError(
            f"derived global sampler position {position} exceeds epoch_size={epoch_size}"
        )
    return position


@dataclass(frozen=True)
class SoftClusterScheduleManifest:
    """Validated identities and runtime dimensions for one schedule."""

    root: Path
    manifest_path: Path
    success_path: Path
    schedule_path: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    schedule_sha256: str
    schedule_size_bytes: int
    schema_version: int
    format: str
    layout: str
    gamma: float
    seed: int
    epoch_size: int
    world_size: int
    origin_consumed_global_batches: int
    target_consumed_global_batches: int
    wave_count: int
    row_count: int
    flat_row_count: int
    dtype: str
    dataset_binding_sha256: str
    source_checkpoint_binding_sha256: str
    artifact_stat_paths: tuple[Path, ...]

    @property
    def source_checkpoint(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.manifest["source_checkpoint"]))

    @property
    def algorithm(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.manifest["algorithm"]))

    @property
    def dataset(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.manifest["dataset"]))

    @property
    def catalog_migration(self) -> Optional[dict[str, Any]]:
        """Return the authenticated source-to-target catalog contract, if any."""

        migration = self.manifest.get("catalog_migration")
        if migration is None:
            return None
        return copy.deepcopy(dict(migration))


@dataclass(frozen=True)
class _ManifestCacheEntry:
    info: SoftClusterScheduleManifest
    stat_signature: tuple[tuple[str, int, int, int, int], ...]


_MANIFEST_CACHE: dict[str, _ManifestCacheEntry] = {}
_MANIFEST_CACHE_LOCK = threading.Lock()


def clear_soft_cluster_schedule_cache() -> None:
    """Clear process-local validation results (primarily useful in tests)."""

    with _MANIFEST_CACHE_LOCK:
        _MANIFEST_CACHE.clear()


def _stat_signature(paths: Sequence[Path]) -> tuple[tuple[str, int, int, int, int], ...]:
    result = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError as exc:
            raise SoftClusterScheduleError(
                f"cannot stat schedule artifact {path}: {exc}"
            ) from exc
        if not path.is_file():
            raise SoftClusterScheduleError(f"schedule artifact is not a file: {path}")
        result.append(
            (
                str(path),
                int(stat.st_dev),
                int(stat.st_ino),
                int(stat.st_size),
                int(stat.st_mtime_ns),
            )
        )
    return tuple(result)


def _resolve_manifest_path(manifest_path: os.PathLike[str] | str) -> Path:
    path = Path(manifest_path).expanduser()
    if path.is_dir():
        path = path / SOFT_CLUSTER_SCHEDULE_MANIFEST_FILENAME
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SoftClusterScheduleError(
            f"soft-cluster schedule manifest does not exist: {path}"
        ) from exc
    if not resolved.is_file():
        raise SoftClusterScheduleError(
            f"soft-cluster schedule manifest is not a file: {resolved}"
        )
    return resolved


def _validate_artifact_record(
    root: Path,
    name: str,
    record: Any,
) -> tuple[Path, int, str]:
    mapping = _require_mapping(record, label=f"artifacts.{name}")
    _require_exact_keys(
        mapping,
        required={"path", "size_bytes", "sha256"},
        label=f"artifacts.{name}",
    )
    path = _safe_artifact_path(
        root,
        mapping["path"],
        label=f"artifacts.{name}.path",
    )
    declared_size = _require_int(
        mapping["size_bytes"],
        label=f"artifacts.{name}.size_bytes",
        minimum=0,
    )
    declared_sha = _require_sha256(
        mapping["sha256"],
        label=f"artifacts.{name}.sha256",
    )
    actual_size = int(path.stat().st_size)
    if actual_size != declared_size:
        raise SoftClusterScheduleError(
            f"artifact {name} size mismatch: manifest={declared_size}, actual={actual_size}"
        )
    actual_sha = _sha256_file(path)
    if actual_sha != declared_sha:
        raise SoftClusterScheduleError(
            f"artifact {name} SHA-256 mismatch: manifest={declared_sha}, actual={actual_sha}"
        )
    return path, declared_size, declared_sha


def _validate_source_checkpoint(
    source: Any,
    *,
    sampler_origin: int,
) -> dict[str, Any]:
    source = dict(_require_mapping(source, label="source_checkpoint"))
    required = {
        "mode",
        "raw_checkpoint_path",
        "raw_checkpoint_sha256",
        "ema_checkpoint_path",
        "ema_checkpoint_sha256",
        "checkpoint_step",
        "completed_updates",
        "next_global_step",
        "origin_consumed_global_batches_per_rank",
        "iters_to_accumulate",
        "checkpoint_pair_id",
        "source_training_contract_id",
        "source_data_contract_sha256",
        "source_run_id",
        "model_architecture_sha256",
    }
    _require_exact_keys(source, required=required, label="source_checkpoint")
    mode = source["mode"]
    if mode not in {"scratch", "checkpoint_migration", "fresh"}:
        raise SoftClusterScheduleError(
            "source_checkpoint.mode must be 'scratch', 'checkpoint_migration', or 'fresh'"
        )
    accumulation = _require_int(
        source["iters_to_accumulate"],
        label="source_checkpoint.iters_to_accumulate",
        minimum=1,
    )
    completed_updates = _require_int(
        source["completed_updates"],
        label="source_checkpoint.completed_updates",
        minimum=0,
    )
    next_global_step = _require_int(
        source["next_global_step"],
        label="source_checkpoint.next_global_step",
        minimum=0,
    )
    source_origin = _require_int(
        source["origin_consumed_global_batches_per_rank"],
        label="source_checkpoint.origin_consumed_global_batches_per_rank",
        minimum=0,
    )
    if source_origin != sampler_origin or next_global_step != sampler_origin:
        raise SoftClusterScheduleError(
            "source checkpoint cursor must equal sampler origin: "
            f"source_origin={source_origin}, next_global_step={next_global_step}, "
            f"sampler_origin={sampler_origin}"
        )
    if mode == "fresh":
        # A fresh data stage can start at an existing training cursor without
        # binding the sampling stream to a particular model/checkpoint file.
        if source["checkpoint_step"] != completed_updates - 1:
            raise SoftClusterScheduleError("fresh checkpoint_step must equal completed_updates - 1")
        if next_global_step != completed_updates * accumulation:
            raise SoftClusterScheduleError("fresh origin must equal completed_updates * iters_to_accumulate")
        nullable_fields = {
            "raw_checkpoint_path", "raw_checkpoint_sha256",
            "ema_checkpoint_path", "ema_checkpoint_sha256", "checkpoint_pair_id",
            "source_training_contract_id", "source_data_contract_sha256",
            "source_run_id", "model_architecture_sha256",
        }
        non_null = sorted(name for name in nullable_fields if source[name] is not None)
        if non_null:
            raise SoftClusterScheduleError(f"fresh schedule must not bind model checkpoint identity: {non_null}")
        return source
    _require_sha256(
        source["model_architecture_sha256"],
        label="source_checkpoint.model_architecture_sha256",
    )

    if mode == "scratch":
        if source["checkpoint_step"] != -1:
            raise SoftClusterScheduleError(
                "scratch source_checkpoint.checkpoint_step must be -1"
            )
        if completed_updates != 0 or next_global_step != 0:
            raise SoftClusterScheduleError(
                "scratch schedule must start at completed_updates=0 and next_global_step=0"
            )
        nullable_fields = {
            "raw_checkpoint_path",
            "raw_checkpoint_sha256",
            "ema_checkpoint_path",
            "ema_checkpoint_sha256",
            "checkpoint_pair_id",
            "source_training_contract_id",
            "source_data_contract_sha256",
            "source_run_id",
        }
        non_null = sorted(name for name in nullable_fields if source[name] is not None)
        if non_null:
            raise SoftClusterScheduleError(
                f"scratch source checkpoint fields must be null: {non_null}"
            )
        return source

    checkpoint_step = _require_int(
        source["checkpoint_step"],
        label="source_checkpoint.checkpoint_step",
        minimum=0,
    )
    if completed_updates != checkpoint_step + 1:
        raise SoftClusterScheduleError(
            "source checkpoint completed_updates must equal checkpoint_step + 1"
        )
    expected_next_global_step = completed_updates * accumulation
    if next_global_step != expected_next_global_step:
        raise SoftClusterScheduleError(
            "source checkpoint next_global_step must equal "
            "completed_updates * iters_to_accumulate: "
            f"expected={expected_next_global_step}, actual={next_global_step}"
        )
    _require_real_absolute_path(
        source["raw_checkpoint_path"],
        label="source_checkpoint.raw_checkpoint_path",
    )
    _require_real_absolute_path(
        source["ema_checkpoint_path"],
        label="source_checkpoint.ema_checkpoint_path",
    )
    _require_sha256(
        source["raw_checkpoint_sha256"],
        label="source_checkpoint.raw_checkpoint_sha256",
    )
    _require_sha256(
        source["ema_checkpoint_sha256"],
        label="source_checkpoint.ema_checkpoint_sha256",
    )
    _require_sha256(
        source["checkpoint_pair_id"],
        label="source_checkpoint.checkpoint_pair_id",
    )
    _require_nonempty_string(
        source["source_training_contract_id"],
        label="source_checkpoint.source_training_contract_id",
    )
    _require_sha256(
        source["source_data_contract_sha256"],
        label="source_checkpoint.source_data_contract_sha256",
    )
    _require_nonempty_string(
        source["source_run_id"],
        label="source_checkpoint.source_run_id",
    )
    return source


def _validate_dataset_binding(
    dataset: Any,
    *,
    flat_row_count: int,
) -> dict[str, Any]:
    dataset = dict(_require_mapping(dataset, label="dataset"))
    _require_exact_keys(
        dataset,
        required={
            "flat_row_count",
            "train_sets",
            "dataset_offsets",
            "dataset_masses",
            "child_fingerprints",
        },
        optional={"runtime_bindings"},
        label="dataset",
    )
    if _require_int(
        dataset["flat_row_count"],
        label="dataset.flat_row_count",
        minimum=1,
    ) != flat_row_count:
        raise SoftClusterScheduleError("dataset.flat_row_count is internally inconsistent")
    train_sets_raw = dataset["train_sets"]
    if not isinstance(train_sets_raw, Sequence) or isinstance(
        train_sets_raw, (str, bytes)
    ):
        raise SoftClusterScheduleError("dataset.train_sets must be a list")
    train_sets = [
        _require_nonempty_string(value, label="dataset.train_sets[]")
        for value in train_sets_raw
    ]
    if not train_sets or len(set(train_sets)) != len(train_sets):
        raise SoftClusterScheduleError("dataset.train_sets must be non-empty and unique")

    offsets_raw = dataset["dataset_offsets"]
    masses_raw = dataset["dataset_masses"]
    children_raw = dataset["child_fingerprints"]
    if not isinstance(offsets_raw, Sequence) or isinstance(offsets_raw, (str, bytes)):
        raise SoftClusterScheduleError("dataset.dataset_offsets must be a list")
    if not isinstance(masses_raw, Sequence) or isinstance(masses_raw, (str, bytes)):
        raise SoftClusterScheduleError("dataset.dataset_masses must be a list")
    if not isinstance(children_raw, Sequence) or isinstance(children_raw, (str, bytes)):
        raise SoftClusterScheduleError("dataset.child_fingerprints must be a list")
    offsets = [
        _require_int(value, label="dataset.dataset_offsets[]", minimum=0)
        for value in offsets_raw
    ]
    if len(offsets) != len(train_sets) + 1:
        raise SoftClusterScheduleError(
            "dataset.dataset_offsets must contain every child boundary"
        )
    if offsets[0] != 0 or offsets[-1] != flat_row_count or any(
        left >= right for left, right in zip(offsets, offsets[1:])
    ):
        raise SoftClusterScheduleError(
            "dataset.dataset_offsets must be strictly increasing from 0 to flat_row_count"
        )
    masses = [
        _require_float(value, label="dataset.dataset_masses[]")
        for value in masses_raw
    ]
    if len(masses) != len(train_sets) or any(value < 0 for value in masses) or not any(
        value > 0 for value in masses
    ):
        raise SoftClusterScheduleError(
            "dataset.dataset_masses must contain one non-negative mass per child and one positive mass"
        )
    if len(children_raw) != len(train_sets):
        raise SoftClusterScheduleError(
            "dataset.child_fingerprints must contain one record per child"
        )
    for index, child in enumerate(children_raw):
        child = _require_mapping(
            child,
            label=f"dataset.child_fingerprints[{index}]",
        )
        for field in ("kind", "size", "sha256"):
            if field not in child:
                raise SoftClusterScheduleError(
                    f"dataset.child_fingerprints[{index}] lacks {field}"
                )
        _require_nonempty_string(
            child["kind"],
            label=f"dataset.child_fingerprints[{index}].kind",
        )
        expected_size = offsets[index + 1] - offsets[index]
        if _require_int(
            child["size"],
            label=f"dataset.child_fingerprints[{index}].size",
            minimum=1,
        ) != expected_size:
            raise SoftClusterScheduleError(
                f"dataset child {index} size does not match its offsets"
            )
        _require_sha256(
            child["sha256"],
            label=f"dataset.child_fingerprints[{index}].sha256",
        )
    if "runtime_bindings" in dataset:
        _require_mapping(dataset["runtime_bindings"], label="dataset.runtime_bindings")
    return dataset


def _validate_bound_dataset_record(
    value: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], str]:
    record = dict(_require_mapping(value, label=label))
    _require_exact_keys(
        record,
        required={"binding", "binding_sha256"},
        label=label,
    )
    binding_raw = _require_mapping(record["binding"], label=f"{label}.binding")
    flat_row_count = _require_int(
        binding_raw.get("flat_row_count"),
        label=f"{label}.binding.flat_row_count",
        minimum=1,
    )
    binding = _validate_dataset_binding(
        binding_raw,
        flat_row_count=flat_row_count,
    )
    declared_sha = _require_sha256(
        record["binding_sha256"],
        label=f"{label}.binding_sha256",
    )
    observed_sha = _canonical_sha256(binding)
    if declared_sha != observed_sha:
        raise SoftClusterScheduleError(
            f"{label}.binding_sha256 does not authenticate its binding"
        )
    return binding, declared_sha


def _validate_external_tsv_binding(value: Any, *, label: str) -> dict[str, Any]:
    record = dict(_require_mapping(value, label=label))
    _require_exact_keys(
        record,
        required={"path", "size_bytes", "sha256", "count", "dtype", "columns"},
        label=label,
    )
    _require_real_absolute_path(record["path"], label=f"{label}.path")
    _require_int(record["size_bytes"], label=f"{label}.size_bytes", minimum=0)
    _require_int(record["count"], label=f"{label}.count", minimum=0)
    _require_sha256(record["sha256"], label=f"{label}.sha256")
    if record["dtype"] != "tsv_gzip":
        raise SoftClusterScheduleError(f"{label}.dtype must be 'tsv_gzip'")
    columns = record["columns"]
    if not isinstance(columns, Sequence) or isinstance(columns, (str, bytes)):
        raise SoftClusterScheduleError(f"{label}.columns must be a list")
    normalized = [
        _require_nonempty_string(column, label=f"{label}.columns[]")
        for column in columns
    ]
    if not normalized or len(set(normalized)) != len(normalized):
        raise SoftClusterScheduleError(f"{label}.columns must be non-empty and unique")
    record["columns"] = normalized
    return record


def _validate_lookup_binding(value: Any, *, label: str) -> dict[str, Any]:
    binding = dict(_require_mapping(value, label=label))
    _require_exact_keys(
        binding,
        required={
            "role",
            "runtime_manifest_path",
            "runtime_manifest_sha256",
            "pair_id",
            "pair_receipt_path",
            "pair_receipt_size_bytes",
            "pair_receipt_sha256",
            "member_to_cluster",
            "ambiguity_alternates",
        },
        label=label,
    )
    if binding["role"] not in {"legacy_residual", "ge200_structural"}:
        raise SoftClusterScheduleError(f"{label}.role is unsupported")
    for field in ("runtime_manifest_path", "pair_receipt_path"):
        _require_real_absolute_path(binding[field], label=f"{label}.{field}")
    for field in (
        "runtime_manifest_sha256",
        "pair_id",
        "pair_receipt_sha256",
    ):
        _require_sha256(binding[field], label=f"{label}.{field}")
    _require_int(
        binding["pair_receipt_size_bytes"],
        label=f"{label}.pair_receipt_size_bytes",
        minimum=0,
    )
    member = _validate_external_tsv_binding(
        binding["member_to_cluster"],
        label=f"{label}.member_to_cluster",
    )
    expected_member_columns = [
        "runtime_row_id",
        "global_id",
        "cluster_ordinal",
        "stable_cluster_id",
        "source_member_row_id",
        "source_partition",
        "cluster_representative_length_bucket",
    ]
    if member["columns"] != expected_member_columns:
        raise SoftClusterScheduleError(
            f"{label}.member_to_cluster columns differ from the v1 contract"
        )
    ambiguity_raw = binding["ambiguity_alternates"]
    if binding["role"] == "ge200_structural":
        ambiguity = _validate_external_tsv_binding(
            ambiguity_raw,
            label=f"{label}.ambiguity_alternates",
        )
        if ambiguity["columns"] != [
            "global_id",
            "canonical_cluster_ordinal",
            "alternate_cluster_ordinal",
            "raw_id",
            "resolution_rule",
        ]:
            raise SoftClusterScheduleError(
                f"{label}.ambiguity_alternates columns differ from contract"
            )
    elif ambiguity_raw is not None:
        raise SoftClusterScheduleError(
            f"{label}.ambiguity_alternates must be null for legacy_residual"
        )
    return binding


def _validate_projection_accounting(value: Any) -> dict[str, Any]:
    accounting = dict(_require_mapping(value, label="catalog_migration.crosswalk.accounting"))
    required = {
        "schema_version",
        "event",
        "global_id_policy",
        "exposure_policy",
        "ambiguous_alternate_policy",
        "source_unique_seen_members",
        "source_history_occurrences",
        "mapped_unique_members",
        "mapped_occurrences",
        "intentional_missing_unique_members",
        "intentional_missing_occurrences",
        "intentional_missing_by_reason",
        "unresolved_unique_members",
        "unresolved_occurrences",
        "ambiguous_alternate_records",
        "ambiguous_seen_records",
        "ambiguous_alternate_clusters",
        "ambiguous_floor_increment",
        "target_projected_occurrences",
        "target_initial_exposure_total",
        "target_exposure",
    }
    missing = required - set(accounting)
    if missing:
        raise SoftClusterScheduleError(
            "catalog migration accounting lacks fields: " f"{sorted(missing)}"
        )
    if accounting["schema_version"] != 1 or accounting["event"] != (
        "source_target_catalog_history_projection"
    ):
        raise SoftClusterScheduleError("unsupported catalog migration accounting")
    policies = {
        "global_id_policy": SOFT_CLUSTER_GLOBAL_ID_POLICY,
        "exposure_policy": SOFT_CLUSTER_EXPOSURE_POLICY,
        "ambiguous_alternate_policy": SOFT_CLUSTER_AMBIGUOUS_POLICY,
    }
    for field, expected in policies.items():
        if accounting[field] != expected:
            raise SoftClusterScheduleError(
                f"catalog migration accounting {field} is unsupported"
            )
    integers = {
        field: _require_int(accounting[field], label=f"accounting.{field}", minimum=0)
        for field in required
        if field not in {
            "schema_version",
            "event",
            "global_id_policy",
            "exposure_policy",
            "ambiguous_alternate_policy",
            "intentional_missing_by_reason",
            "target_exposure",
        }
    }
    if integers["unresolved_unique_members"] or integers["unresolved_occurrences"]:
        raise SoftClusterScheduleError(
            "catalog migration contains unresolved historical identities"
        )
    if integers["source_history_occurrences"] != (
        integers["mapped_occurrences"]
        + integers["intentional_missing_occurrences"]
    ):
        raise SoftClusterScheduleError(
            "catalog migration source occurrence conservation failed"
        )
    if integers["target_projected_occurrences"] != integers["mapped_occurrences"]:
        raise SoftClusterScheduleError(
            "catalog migration projected occurrence accounting failed"
        )
    if integers["target_initial_exposure_total"] != (
        integers["mapped_occurrences"] + integers["ambiguous_floor_increment"]
    ):
        raise SoftClusterScheduleError(
            "catalog migration target exposure accounting failed"
        )
    _require_mapping(
        accounting["intentional_missing_by_reason"],
        label="accounting.intentional_missing_by_reason",
    )
    _require_mapping(accounting["target_exposure"], label="accounting.target_exposure")
    return accounting


def _validate_catalog_migration(
    value: Any,
    *,
    target_dataset: Mapping[str, Any],
) -> dict[str, Any]:
    migration = dict(_require_mapping(value, label="catalog_migration"))
    _require_exact_keys(
        migration,
        required={
            "format",
            "allowed_delta",
            "source_dataset",
            "target_dataset",
            "crosswalk",
        },
        label="catalog_migration",
    )
    if migration["format"] != SOFT_CLUSTER_CATALOG_MIGRATION_FORMAT:
        raise SoftClusterScheduleError("unsupported catalog_migration.format")
    if migration["allowed_delta"] != SOFT_CLUSTER_CATALOG_MIGRATION_ALLOWED_DELTA:
        raise SoftClusterScheduleError("unsupported catalog_migration.allowed_delta")
    source_binding, _ = _validate_bound_dataset_record(
        migration["source_dataset"], label="catalog_migration.source_dataset"
    )
    target_binding, target_sha = _validate_bound_dataset_record(
        migration["target_dataset"], label="catalog_migration.target_dataset"
    )
    if target_binding != dict(target_dataset) or target_sha != _canonical_sha256(target_dataset):
        raise SoftClusterScheduleError(
            "catalog_migration target dataset differs from manifest.dataset"
        )
    if source_binding == target_binding:
        raise SoftClusterScheduleError("catalog migration source and target are identical")

    crosswalk = dict(_require_mapping(migration["crosswalk"], label="catalog_migration.crosswalk"))
    _require_exact_keys(
        crosswalk,
        required={
            "format",
            "global_id_policy",
            "exposure_policy",
            "ambiguous_alternate_policy",
            "source_soft_datasets",
            "target_soft_datasets",
            "lookup_bindings",
            "intentional_missing_binding",
            "audit_artifact",
            "accounting",
        },
        label="catalog_migration.crosswalk",
    )
    expected_values = {
        "format": SOFT_CLUSTER_CROSSWALK_FORMAT,
        "global_id_policy": SOFT_CLUSTER_GLOBAL_ID_POLICY,
        "exposure_policy": SOFT_CLUSTER_EXPOSURE_POLICY,
        "ambiguous_alternate_policy": SOFT_CLUSTER_AMBIGUOUS_POLICY,
    }
    for field, expected in expected_values.items():
        if crosswalk[field] != expected:
            raise SoftClusterScheduleError(f"unsupported catalog crosswalk {field}")
    source_names = crosswalk["source_soft_datasets"]
    target_names = crosswalk["target_soft_datasets"]
    for names, label in ((source_names, "source"), (target_names, "target")):
        if (
            not isinstance(names, Sequence)
            or isinstance(names, (str, bytes))
            or len(names) != 2
            or len(set(names)) != 2
        ):
            raise SoftClusterScheduleError(
                f"catalog crosswalk {label}_soft_datasets must name two children"
            )
    if not set(source_names).issubset(set(source_binding["train_sets"])):
        raise SoftClusterScheduleError("source soft datasets are absent from source binding")
    if not set(target_names).issubset(set(target_binding["train_sets"])):
        raise SoftClusterScheduleError("target soft datasets are absent from target binding")
    lookups = _require_mapping(crosswalk["lookup_bindings"], label="crosswalk.lookup_bindings")
    if set(lookups) != set(target_names):
        raise SoftClusterScheduleError("lookup bindings do not exactly cover target soft datasets")
    validated_lookups = {
        name: _validate_lookup_binding(value, label=f"crosswalk.lookup_bindings.{name}")
        for name, value in lookups.items()
    }
    if {binding["role"] for binding in validated_lookups.values()} != {
        "legacy_residual",
        "ge200_structural",
    }:
        raise SoftClusterScheduleError("target lookup roles are incomplete")
    if len({binding["pair_id"] for binding in validated_lookups.values()}) != 1 or len(
        {binding["pair_receipt_sha256"] for binding in validated_lookups.values()}
    ) != 1:
        raise SoftClusterScheduleError("target lookup pair bindings differ")
    missing_binding = crosswalk["intentional_missing_binding"]
    if missing_binding is not None:
        missing_binding = dict(_require_mapping(missing_binding, label="crosswalk.intentional_missing_binding"))
        _require_exact_keys(
            missing_binding,
            required={"path", "size_bytes", "sha256", "count", "columns"},
            label="crosswalk.intentional_missing_binding",
        )
        _require_real_absolute_path(missing_binding["path"], label="intentional_missing_binding.path")
        _require_int(missing_binding["size_bytes"], label="intentional_missing_binding.size_bytes", minimum=0)
        _require_int(missing_binding["count"], label="intentional_missing_binding.count", minimum=0)
        _require_sha256(missing_binding["sha256"], label="intentional_missing_binding.sha256")
        if missing_binding["columns"] != ["global_id", "reason"]:
            raise SoftClusterScheduleError("intentional-missing columns differ from contract")
    if crosswalk["audit_artifact"] != "catalog_migration.audit":
        raise SoftClusterScheduleError("catalog migration audit artifact key is invalid")
    _validate_projection_accounting(crosswalk["accounting"])
    return migration


def _validate_migrated_initial_exposure_artifacts(
    *,
    migration: Mapping[str, Any],
    validated_artifacts: Mapping[str, tuple[Path, int, str]],
) -> None:
    crosswalk = migration["crosswalk"]
    accounting = crosswalk["accounting"]
    expected_by_dataset = _require_mapping(
        accounting["target_exposure"],
        label="accounting.target_exposure",
    )
    target_names = list(crosswalk["target_soft_datasets"])
    if set(expected_by_dataset) != set(target_names):
        raise SoftClusterScheduleError(
            "accounting.target_exposure does not exactly cover target soft datasets"
        )
    observed_total = 0
    for name in target_names:
        expected = _require_mapping(
            expected_by_dataset[name],
            label=f"accounting.target_exposure.{name}",
        )
        required_stats = {
            "cluster_count",
            "total_exposure",
            "seen_clusters",
            "maximum_exposure",
        }
        _require_exact_keys(
            expected,
            required=required_stats,
            label=f"accounting.target_exposure.{name}",
        )
        expected_stats = {
            field: _require_int(
                expected[field],
                label=f"accounting.target_exposure.{name}.{field}",
                minimum=0,
            )
            for field in required_stats
        }
        artifact_key = f"{name}.initial_exposure"
        record = validated_artifacts.get(artifact_key)
        if record is None:
            raise SoftClusterScheduleError(
                f"catalog migration lacks initial exposure artifact {artifact_key!r}"
            )
        path, size, _ = record
        cluster_count = expected_stats["cluster_count"]
        if cluster_count <= 0 or size != cluster_count * np.dtype("<u4").itemsize:
            raise SoftClusterScheduleError(
                f"initial exposure artifact size differs for {name}"
            )
        values = np.memmap(path, dtype="<u4", mode="r", shape=(cluster_count,))
        observed = {
            "cluster_count": cluster_count,
            "total_exposure": int(values.sum(dtype=np.uint64)),
            "seen_clusters": int(np.count_nonzero(values)),
            "maximum_exposure": int(values.max()) if cluster_count else 0,
        }
        del values
        if observed != expected_stats:
            raise SoftClusterScheduleError(
                f"initial exposure artifact accounting differs for {name}: "
                f"expected={expected_stats}, observed={observed}"
            )
        observed_total += observed["total_exposure"]
    if observed_total != int(accounting["target_initial_exposure_total"]):
        raise SoftClusterScheduleError(
            "initial exposure artifacts do not sum to target_initial_exposure_total"
        )


def _validate_expected(
    info: SoftClusterScheduleManifest,
    expected: Optional[Mapping[str, Any]],
) -> None:
    if expected is None:
        return
    expected = _require_mapping(expected, label="expected")
    supported = {
        "dataset_size",
        "world_size",
        "epoch_size",
        "gamma",
        "seed",
        "origin_consumed_global_batches",
        "target_consumed_global_batches",
        "source_checkpoint",
    }
    unknown = set(expected) - supported
    if unknown:
        raise SoftClusterScheduleError(
            f"expected has unsupported fields: {sorted(unknown)}"
        )
    scalar_fields = {
        "dataset_size": info.flat_row_count,
        "world_size": info.world_size,
        "epoch_size": info.epoch_size,
        "gamma": info.gamma,
        "seed": info.seed,
        "origin_consumed_global_batches": info.origin_consumed_global_batches,
        "target_consumed_global_batches": info.target_consumed_global_batches,
    }
    for field, actual in scalar_fields.items():
        if field in expected and expected[field] != actual:
            raise SoftClusterScheduleError(
                f"schedule {field} mismatch: expected={expected[field]!r}, actual={actual!r}"
            )
    if "source_checkpoint" in expected:
        expected_source = _require_mapping(
            expected["source_checkpoint"],
            label="expected.source_checkpoint",
        )
        actual_source = info.source_checkpoint
        unknown_source = set(expected_source) - set(actual_source)
        if unknown_source:
            raise SoftClusterScheduleError(
                "expected.source_checkpoint has unsupported fields: "
                f"{sorted(unknown_source)}"
            )
        for field, value in expected_source.items():
            if actual_source[field] != value:
                raise SoftClusterScheduleError(
                    f"schedule source checkpoint mismatch for {field}: "
                    f"expected={value!r}, actual={actual_source[field]!r}"
                )


def _validate_manifest_uncached(manifest_path: Path) -> SoftClusterScheduleManifest:
    root = manifest_path.parent
    success_path = root / SOFT_CLUSTER_SCHEDULE_SUCCESS_FILENAME
    if not success_path.is_file():
        raise SoftClusterScheduleError(
            f"completed schedule receipt is missing: {success_path}"
        )
    manifest, manifest_bytes = _load_json_bytes(manifest_path, label="manifest")
    success, _ = _load_json_bytes(success_path, label="success receipt")

    schema_version = manifest.get("schema_version")
    format_name = manifest.get("format")
    supported_schema_format = {
        SOFT_CLUSTER_SCHEDULE_SCHEMA_VERSION: SOFT_CLUSTER_SCHEDULE_FORMAT,
        SOFT_CLUSTER_SCHEDULE_SCHEMA_VERSION_V2: SOFT_CLUSTER_SCHEDULE_FORMAT_V2,
    }
    if schema_version not in supported_schema_format:
        raise SoftClusterScheduleError(
            "unsupported soft-cluster schedule schema_version: "
            f"{schema_version!r}"
        )
    if format_name != supported_schema_format[schema_version]:
        raise SoftClusterScheduleError(
            "soft-cluster schedule format/schema mismatch: "
            f"schema_version={schema_version!r}, format={format_name!r}"
        )

    _require_exact_keys(
        manifest,
        required={
            "schema_version",
            "format",
            "status",
            "layout",
            "algorithm",
            "shape",
            "dataset",
            "sampler",
            "source_checkpoint",
            "artifacts",
        }
        | ({"catalog_migration"} if schema_version == 2 else set()),
        optional={"source_run", "provenance"},
        label="manifest",
    )
    if manifest["status"] != "complete":
        raise SoftClusterScheduleError("soft-cluster schedule status must be complete")
    if manifest["layout"] != SOFT_CLUSTER_SCHEDULE_LAYOUT:
        raise SoftClusterScheduleError(
            f"unsupported soft-cluster schedule layout: {manifest['layout']!r}"
        )

    algorithm = _require_mapping(manifest["algorithm"], label="algorithm")
    _require_exact_keys(
        algorithm,
        required={
            "format",
            "gamma",
            "member_policy",
            "consumption_order",
            "soft_cluster_datasets",
        },
        label="algorithm",
    )
    if algorithm["format"] != SOFT_CLUSTER_ALGORITHM_FORMAT:
        raise SoftClusterScheduleError(
            f"unsupported soft-cluster algorithm: {algorithm['format']!r}"
        )
    gamma = _require_float(algorithm["gamma"], label="algorithm.gamma")
    if not 0.0 < gamma <= 1.0:
        raise SoftClusterScheduleError(
            f"algorithm.gamma must be inside (0, 1], got {gamma}"
        )
    if algorithm["member_policy"] != SOFT_CLUSTER_MEMBER_POLICY:
        raise SoftClusterScheduleError(
            f"unsupported member policy: {algorithm['member_policy']!r}"
        )
    if algorithm["consumption_order"] != SOFT_CLUSTER_CONSUMPTION_ORDER:
        raise SoftClusterScheduleError(
            "unsupported schedule consumption order: "
            f"{algorithm['consumption_order']!r}"
        )
    soft_datasets_raw = algorithm["soft_cluster_datasets"]
    if not isinstance(soft_datasets_raw, Sequence) or isinstance(
        soft_datasets_raw, (str, bytes)
    ):
        raise SoftClusterScheduleError("algorithm.soft_cluster_datasets must be a list")
    soft_datasets = [
        _require_nonempty_string(value, label="algorithm.soft_cluster_datasets[]")
        for value in soft_datasets_raw
    ]
    if not soft_datasets or len(set(soft_datasets)) != len(soft_datasets):
        raise SoftClusterScheduleError(
            "algorithm.soft_cluster_datasets must be non-empty and unique"
        )

    shape = _require_mapping(manifest["shape"], label="shape")
    _require_exact_keys(
        shape,
        required={"dtype", "row_count", "wave_count"},
        label="shape",
    )
    if shape["dtype"] != SOFT_CLUSTER_SCHEDULE_DTYPE:
        raise SoftClusterScheduleError(
            f"schedule dtype must be {SOFT_CLUSTER_SCHEDULE_DTYPE!r}"
        )
    row_count = _require_int(shape["row_count"], label="shape.row_count", minimum=1)
    wave_count = _require_int(shape["wave_count"], label="shape.wave_count", minimum=1)

    sampler = _require_mapping(manifest["sampler"], label="sampler")
    _require_exact_keys(
        sampler,
        required={
            "seed",
            "epoch_size",
            "world_size",
            "origin_consumed_global_batches",
            "target_consumed_global_batches",
            "target_total_rows",
        },
        label="sampler",
    )
    seed = _require_int(sampler["seed"], label="sampler.seed")
    epoch_size = _require_int(
        sampler["epoch_size"], label="sampler.epoch_size", minimum=1
    )
    world_size = _require_int(
        sampler["world_size"], label="sampler.world_size", minimum=1
    )
    if epoch_size < world_size:
        raise SoftClusterScheduleError(
            "sampler.epoch_size must be >= sampler.world_size"
        )
    origin = _require_int(
        sampler["origin_consumed_global_batches"],
        label="sampler.origin_consumed_global_batches",
        minimum=0,
    )
    target = _require_int(
        sampler["target_consumed_global_batches"],
        label="sampler.target_consumed_global_batches",
        minimum=1,
    )
    if target <= origin:
        raise SoftClusterScheduleError(
            "sampler.target_consumed_global_batches must be greater than origin"
        )
    target_total_rows = _require_int(
        sampler["target_total_rows"],
        label="sampler.target_total_rows",
        minimum=1,
    )
    if wave_count != target - origin:
        raise SoftClusterScheduleError(
            "shape.wave_count must equal target_consumed_global_batches - "
            "origin_consumed_global_batches"
        )
    if row_count != wave_count * world_size or target_total_rows != row_count:
        raise SoftClusterScheduleError(
            "wave-major schedule row_count must equal wave_count * world_size "
            "and sampler.target_total_rows"
        )

    dataset_mapping = _require_mapping(manifest["dataset"], label="dataset")
    flat_row_count = _require_int(
        dataset_mapping.get("flat_row_count"),
        label="dataset.flat_row_count",
        minimum=1,
    )
    dataset_binding = _validate_dataset_binding(
        dataset_mapping,
        flat_row_count=flat_row_count,
    )
    catalog_migration = None
    if schema_version == SOFT_CLUSTER_SCHEDULE_SCHEMA_VERSION_V2:
        catalog_migration = _validate_catalog_migration(
            manifest["catalog_migration"],
            target_dataset=dataset_binding,
        )
    train_sets = set(dataset_binding["train_sets"])
    if not set(soft_datasets).issubset(train_sets):
        raise SoftClusterScheduleError(
            "algorithm.soft_cluster_datasets must be members of dataset.train_sets"
        )

    source_checkpoint = _validate_source_checkpoint(
        manifest["source_checkpoint"],
        sampler_origin=origin,
    )
    if "source_run" in manifest:
        source_run = _require_mapping(manifest["source_run"], label="source_run")
        if not source_run:
            raise SoftClusterScheduleError("source_run must not be empty")
    if "provenance" in manifest:
        _require_mapping(manifest["provenance"], label="provenance")

    artifacts = _require_mapping(manifest["artifacts"], label="artifacts")
    if "global_rows" not in artifacts:
        raise SoftClusterScheduleError("manifest artifacts lacks global_rows")
    allowed_scalar_artifacts = {"artifact_set_sha256"}
    unknown_scalars = [
        name
        for name, value in artifacts.items()
        if not isinstance(value, Mapping) and name not in allowed_scalar_artifacts
    ]
    if unknown_scalars:
        raise SoftClusterScheduleError(
            f"manifest artifacts has unsupported scalar fields: {sorted(unknown_scalars)}"
        )
    artifact_set_sha = None
    if "artifact_set_sha256" in artifacts:
        artifact_set_sha = _require_sha256(
            artifacts["artifact_set_sha256"],
            label="artifacts.artifact_set_sha256",
        )
    validated_artifacts: dict[str, tuple[Path, int, str]] = {}
    for name, record in artifacts.items():
        if name in allowed_scalar_artifacts:
            continue
        validated_artifacts[name] = _validate_artifact_record(root, name, record)
    if catalog_migration is not None:
        audit_key = catalog_migration["crosswalk"]["audit_artifact"]
        if audit_key not in validated_artifacts:
            raise SoftClusterScheduleError(
                "catalog migration audit artifact is absent from manifest.artifacts"
            )
        audit_document, _ = _load_json_bytes(
            validated_artifacts[audit_key][0],
            label="catalog migration audit",
        )
        if audit_document != catalog_migration["crosswalk"]["accounting"]:
            raise SoftClusterScheduleError(
                "catalog migration audit artifact differs from manifest accounting"
            )
        _validate_migrated_initial_exposure_artifacts(
            migration=catalog_migration,
            validated_artifacts=validated_artifacts,
        )
    schedule_path, schedule_size, schedule_sha = validated_artifacts["global_rows"]
    if artifacts["global_rows"]["path"] != SOFT_CLUSTER_SCHEDULE_ROWS_FILENAME:
        raise SoftClusterScheduleError(
            f"global_rows path must be {SOFT_CLUSTER_SCHEDULE_ROWS_FILENAME!r}"
        )
    expected_schedule_size = row_count * _UINT32_ITEMSIZE
    if schedule_size != expected_schedule_size:
        raise SoftClusterScheduleError(
            "global_rows size does not match shape.row_count and uint32 dtype: "
            f"expected={expected_schedule_size}, actual={schedule_size}"
        )

    manifest_sha = _sha256_bytes(manifest_bytes)
    _require_exact_keys(
        success,
        required={
            "schema_version",
            "status",
            "manifest_sha256",
            "schedule_path",
            "schedule_size_bytes",
            "schedule_sha256",
        },
        optional={"artifact_set_sha256"},
        label="success receipt",
    )
    if success["schema_version"] != SOFT_CLUSTER_SCHEDULE_SUCCESS_SCHEMA_VERSION:
        raise SoftClusterScheduleError(
            "unsupported schedule success schema_version: "
            f"{success['schema_version']!r}"
        )
    if success["status"] != "complete":
        raise SoftClusterScheduleError("schedule success status must be complete")
    if success["manifest_sha256"] != manifest_sha:
        raise SoftClusterScheduleError(
            "success receipt manifest SHA-256 does not match manifest bytes"
        )
    if success["schedule_path"] != SOFT_CLUSTER_SCHEDULE_ROWS_FILENAME:
        raise SoftClusterScheduleError("success receipt schedule_path is invalid")
    if success["schedule_size_bytes"] != schedule_size:
        raise SoftClusterScheduleError("success receipt schedule size is invalid")
    if success["schedule_sha256"] != schedule_sha:
        raise SoftClusterScheduleError("success receipt schedule SHA-256 is invalid")
    if success.get("artifact_set_sha256") != artifact_set_sha:
        raise SoftClusterScheduleError(
            "success receipt artifact_set_sha256 does not match manifest"
        )

    # The authenticated file must never address outside the flattened dataset.
    rows = np.memmap(
        schedule_path,
        dtype=SOFT_CLUSTER_SCHEDULE_DTYPE,
        mode="r",
        shape=(row_count,),
    )
    chunk_size = 4 * 1024 * 1024
    for start in range(0, row_count, chunk_size):
        chunk = rows[start : start + chunk_size]
        if chunk.size and int(chunk.max()) >= flat_row_count:
            del rows
            raise SoftClusterScheduleError(
                "global_rows contains a row outside dataset.flat_row_count"
            )
    del rows

    artifact_paths = tuple(
        [manifest_path, success_path]
        + [record[0] for record in validated_artifacts.values()]
    )
    return SoftClusterScheduleManifest(
        root=root,
        manifest_path=manifest_path,
        success_path=success_path,
        schedule_path=schedule_path,
        manifest=copy.deepcopy(manifest),
        manifest_sha256=manifest_sha,
        schedule_sha256=schedule_sha,
        schedule_size_bytes=schedule_size,
        schema_version=int(schema_version),
        format=str(format_name),
        layout=SOFT_CLUSTER_SCHEDULE_LAYOUT,
        gamma=gamma,
        seed=seed,
        epoch_size=epoch_size,
        world_size=world_size,
        origin_consumed_global_batches=origin,
        target_consumed_global_batches=target,
        wave_count=wave_count,
        row_count=row_count,
        flat_row_count=flat_row_count,
        dtype=SOFT_CLUSTER_SCHEDULE_DTYPE,
        dataset_binding_sha256=_canonical_sha256(dataset_binding),
        source_checkpoint_binding_sha256=_canonical_sha256(source_checkpoint),
        artifact_stat_paths=artifact_paths,
    )


def load_soft_cluster_schedule_manifest(
    manifest_path: os.PathLike[str] | str,
    *,
    expected: Optional[Mapping[str, Any]] = None,
) -> SoftClusterScheduleManifest:
    """Validate and load manifest metadata with process-local stat caching."""

    resolved = _resolve_manifest_path(manifest_path)
    cache_key = str(resolved)
    with _MANIFEST_CACHE_LOCK:
        cached = _MANIFEST_CACHE.get(cache_key)
    if cached is not None:
        try:
            current_signature = _stat_signature(cached.info.artifact_stat_paths)
        except SoftClusterScheduleError:
            with _MANIFEST_CACHE_LOCK:
                _MANIFEST_CACHE.pop(cache_key, None)
            raise
        if current_signature == cached.stat_signature:
            _validate_expected(cached.info, expected)
            return cached.info

    info = _validate_manifest_uncached(resolved)
    stat_signature = _stat_signature(info.artifact_stat_paths)
    with _MANIFEST_CACHE_LOCK:
        _MANIFEST_CACHE[cache_key] = _ManifestCacheEntry(
            info=info,
            stat_signature=stat_signature,
        )
    _validate_expected(info, expected)
    return info


def build_soft_cluster_schedule_contract(
    manifest_info: SoftClusterScheduleManifest,
) -> dict[str, Any]:
    """Return the immutable scientific subset for the training-data contract."""

    if not isinstance(manifest_info, SoftClusterScheduleManifest):
        raise TypeError("manifest_info must be a validated SoftClusterScheduleManifest")
    source_identity = manifest_info.source_checkpoint
    contract = {
        "schema_version": manifest_info.schema_version,
        "format": manifest_info.format,
        "layout": manifest_info.layout,
        "algorithm": manifest_info.algorithm,
        "source_identity": source_identity,
        "manifest_sha256": manifest_info.manifest_sha256,
        "dataset_binding_sha256": manifest_info.dataset_binding_sha256,
        "source_checkpoint_binding_sha256": (
            manifest_info.source_checkpoint_binding_sha256
        ),
        "sampler": {
            "seed": manifest_info.seed,
            "epoch_size": manifest_info.epoch_size,
            "world_size": manifest_info.world_size,
            "origin_consumed_global_batches_per_rank": (
                manifest_info.origin_consumed_global_batches
            ),
            "target_consumed_global_batches_per_rank": (
                manifest_info.target_consumed_global_batches
            ),
        },
        "schedule": {
            "dtype": manifest_info.dtype,
            "row_count": manifest_info.row_count,
            "wave_count": manifest_info.wave_count,
            "size_bytes": manifest_info.schedule_size_bytes,
            "sha256": manifest_info.schedule_sha256,
        },
    }
    if manifest_info.catalog_migration is not None:
        contract["catalog_migration"] = manifest_info.catalog_migration
    return contract


class SoftClusterSchedule:
    """An authenticated schedule backed by a uint32 array or read-only mmap."""

    def __init__(
        self,
        info: SoftClusterScheduleManifest,
        rows: np.ndarray,
    ) -> None:
        if rows.dtype != np.dtype(SOFT_CLUSTER_SCHEDULE_DTYPE):
            raise SoftClusterScheduleError(
                f"schedule rows must use dtype {SOFT_CLUSTER_SCHEDULE_DTYPE}"
            )
        if rows.ndim != 1 or rows.shape[0] != info.row_count:
            raise SoftClusterScheduleError(
                "schedule rows shape does not match validated manifest"
            )
        self.info = info
        self._rows = rows

    def __len__(self) -> int:
        return self.info.row_count

    def row_at(self, absolute_consumed_batches: int, rank: int) -> int:
        """Return one flattened dataset row from the wave-major schedule."""

        absolute = _require_int(
            absolute_consumed_batches,
            label="absolute_consumed_batches",
            minimum=0,
        )
        rank = _require_int(rank, label="rank", minimum=0)
        if rank >= self.info.world_size:
            raise SoftClusterScheduleError(
                f"rank must be inside [0, {self.info.world_size}), got {rank}"
            )
        relative_wave = absolute - self.info.origin_consumed_global_batches
        if relative_wave < 0:
            raise SoftClusterScheduleError(
                "requested sample precedes the checkpoint-bound schedule origin: "
                f"requested={absolute}, origin={self.info.origin_consumed_global_batches}"
            )
        if relative_wave >= self.info.wave_count:
            raise SoftClusterScheduleError(
                "soft-cluster schedule is exhausted: "
                f"requested={absolute}, target={self.info.target_consumed_global_batches}"
            )
        offset = relative_wave * self.info.world_size + rank
        return int(self._rows[offset])

    def rows_for_rank(
        self,
        start_consumed_batches: int,
        count: int,
        rank: int,
    ) -> np.ndarray:
        """Return a read-only strided view/copy of consecutive waves for one rank."""

        start = _require_int(
            start_consumed_batches,
            label="start_consumed_batches",
            minimum=0,
        )
        count = _require_int(count, label="count", minimum=0)
        rank = _require_int(rank, label="rank", minimum=0)
        if rank >= self.info.world_size:
            raise SoftClusterScheduleError(
                f"rank must be inside [0, {self.info.world_size}), got {rank}"
            )
        relative_start = start - self.info.origin_consumed_global_batches
        if relative_start < 0 or relative_start + count > self.info.wave_count:
            raise SoftClusterScheduleError(
                "requested rank slice is outside the schedule span"
            )
        first = relative_start * self.info.world_size + rank
        stop = first + count * self.info.world_size
        result = self._rows[first:stop:self.info.world_size]
        result.flags.writeable = False
        return result

    def close(self) -> None:
        """Close a numpy memmap owned by this object, if present."""

        mapping = getattr(self._rows, "_mmap", None)
        if mapping is not None:
            mapping.close()


def load_soft_cluster_schedule(
    manifest_path: os.PathLike[str] | str,
    *,
    expected: Optional[Mapping[str, Any]] = None,
    mmap: bool = True,
) -> SoftClusterSchedule:
    """Load an authenticated schedule as a read-only mmap or in-memory array."""

    info = load_soft_cluster_schedule_manifest(manifest_path, expected=expected)
    if mmap:
        rows: np.ndarray = np.memmap(
            info.schedule_path,
            dtype=SOFT_CLUSTER_SCHEDULE_DTYPE,
            mode="r",
            shape=(info.row_count,),
        )
    else:
        rows = np.fromfile(info.schedule_path, dtype=SOFT_CLUSTER_SCHEDULE_DTYPE)
        if rows.shape != (info.row_count,):
            raise SoftClusterScheduleError(
                "global_rows changed after manifest validation"
            )
        rows.flags.writeable = False
    return SoftClusterSchedule(info, rows)


class DistributedPrecomputedScheduleSampler(Sampler):
    """Distributed sampler backed by a checkpoint-bound wave-major schedule."""

    _STATE_VERSION = 1

    def __init__(
        self,
        dataset: Dataset,
        *,
        manifest_path: Optional[os.PathLike[str] | str] = None,
        schedule: Optional[SoftClusterSchedule] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        epoch_size: Optional[int] = None,
        gamma: Optional[float] = None,
        seed: Optional[int] = None,
        emit_sample_occurrence: bool = False,
        mmap: bool = True,
    ) -> None:
        if (manifest_path is None) == (schedule is None):
            raise SoftClusterScheduleError(
                "provide exactly one of manifest_path or schedule"
            )
        if schedule is None:
            expected: dict[str, Any] = {"dataset_size": len(dataset)}
            if num_replicas is not None:
                expected["world_size"] = int(num_replicas)
            if epoch_size is not None:
                expected["epoch_size"] = int(epoch_size)
            if gamma is not None:
                expected["gamma"] = float(gamma)
            if seed is not None:
                expected["seed"] = int(seed)
            assert manifest_path is not None
            schedule = load_soft_cluster_schedule(
                manifest_path,
                expected=expected,
                mmap=mmap,
            )
        self.schedule = schedule
        info = schedule.info
        if len(dataset) != info.flat_row_count:
            raise SoftClusterScheduleError(
                "dataset size does not match the schedule's flattened dataset"
            )

        distributed = dist.is_available() and dist.is_initialized()
        resolved_world_size = (
            dist.get_world_size()
            if num_replicas is None and distributed
            else info.world_size
            if num_replicas is None
            else int(num_replicas)
        )
        resolved_rank = (
            dist.get_rank()
            if rank is None and distributed
            else 0
            if rank is None
            else int(rank)
        )
        if resolved_world_size != info.world_size:
            raise SoftClusterScheduleError(
                "distributed world size does not match schedule manifest: "
                f"runtime={resolved_world_size}, schedule={info.world_size}"
            )
        if resolved_rank < 0 or resolved_rank >= resolved_world_size:
            raise SoftClusterScheduleError(
                f"rank must be inside [0, {resolved_world_size}), got {resolved_rank}"
            )
        if epoch_size is not None and int(epoch_size) != info.epoch_size:
            raise SoftClusterScheduleError("epoch_size does not match schedule manifest")
        if gamma is not None and float(gamma) != info.gamma:
            raise SoftClusterScheduleError("gamma does not match schedule manifest")
        if seed is not None and int(seed) != info.seed:
            raise SoftClusterScheduleError("seed does not match schedule manifest")

        self.dataset = dataset
        self.num_replicas = resolved_world_size
        self.rank = resolved_rank
        self.num_samples = info.epoch_size
        self.seed = info.seed
        self.replacement = True
        self.emit_sample_occurrence = bool(emit_sample_occurrence)
        self.epoch = 0
        self.start_offset = 0
        self.num_samples_per_replica = int(
            math.ceil(self.num_samples / self.num_replicas)
        )
        self.total_size = self.num_samples_per_replica * self.num_replicas

    def local_epoch_size(self) -> int:
        return rank_local_epoch_size(self.num_samples, self.num_replicas, self.rank)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _require_int(epoch, label="sampler_epoch", minimum=0)

    def set_start_offset(self, offset: int) -> None:
        offset = _require_int(offset, label="sampler start offset", minimum=0)
        local_size = self.local_epoch_size()
        if offset > local_size:
            raise SoftClusterScheduleError(
                f"sampler start offset must be inside [0, {local_size}], got {offset}"
            )
        self.start_offset = offset

    def set_consumed_global_batches(self, consumed_global_batches: int) -> None:
        consumed_global_batches = self._validate_schedule_consumed_span(
            consumed_global_batches
        )
        epoch, offset = sampler_cursor_from_consumed_batches(
            consumed_global_batches,
            epoch_size=self.num_samples,
            world_size=self.num_replicas,
            rank=self.rank,
        )
        self.epoch = epoch
        self.start_offset = offset

    def _validate_schedule_consumed_span(self, consumed_global_batches: int) -> int:
        consumed = _require_int(
            consumed_global_batches,
            label="consumed_global_batches",
            minimum=0,
        )
        info = self.schedule.info
        if not (
            info.origin_consumed_global_batches
            <= consumed
            <= info.target_consumed_global_batches
        ):
            raise SoftClusterScheduleError(
                "consumed_global_batches is outside the schedule span: "
                f"origin={info.origin_consumed_global_batches}, "
                f"consumed={consumed}, "
                f"target={info.target_consumed_global_batches}"
            )
        return consumed

    def consumed_global_batches(self) -> int:
        local_size = self.local_epoch_size()
        if self.start_offset == local_size:
            return (self.epoch + 1) * local_size
        return consumed_batches_from_sampler_cursor(
            self.epoch,
            self.start_offset,
            epoch_size=self.num_samples,
            world_size=self.num_replicas,
            rank=self.rank,
        )

    def __iter__(self) -> Iterator[int | SampleOccurrence]:
        local_size = self.local_epoch_size()
        if self.start_offset > local_size:
            raise SoftClusterScheduleError("invalid sampler start offset")
        entries: list[int | SampleOccurrence] = []
        for local_offset in range(self.start_offset, local_size):
            absolute_consumed = self.epoch * local_size + local_offset
            row = self.schedule.row_at(absolute_consumed, self.rank)
            if not self.emit_sample_occurrence:
                entries.append(row)
                continue
            position = sampler_global_position(
                local_offset,
                epoch_size=self.num_samples,
                world_size=self.num_replicas,
                rank=self.rank,
            )
            entries.append(
                SampleOccurrence(
                    index=row,
                    sampler_seed=self.seed,
                    sampler_epoch=self.epoch,
                    global_position=position,
                )
            )
        return iter(entries)

    def __len__(self) -> int:
        # Retain the existing distributed sampler's historical ceil value. The
        # resumable loader uses local_epoch_size() for its exact cursor math.
        return self.num_samples_per_replica

    def resume_signature(self) -> dict[str, Any]:
        info = self.schedule.info
        return {
            "sampler_type": type(self).__name__,
            "layout": SOFT_CLUSTER_SAMPLER_LAYOUT,
            "seed": self.seed,
            "num_samples": self.num_samples,
            "replacement": self.replacement,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "emit_sample_occurrence": self.emit_sample_occurrence,
            "gamma": info.gamma,
            "schedule_schema_version": info.schema_version,
            "schedule_format": info.format,
            "schedule_layout": info.layout,
            "schedule_manifest_sha256": info.manifest_sha256,
            "schedule_array_sha256": info.schedule_sha256,
            "dataset_binding_sha256": info.dataset_binding_sha256,
            "source_checkpoint_binding_sha256": (
                info.source_checkpoint_binding_sha256
            ),
            "origin_consumed_global_batches": (
                info.origin_consumed_global_batches
            ),
            "target_consumed_global_batches": (
                info.target_consumed_global_batches
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self._STATE_VERSION,
            "sampler_epoch": self.epoch,
            "start_offset": self.start_offset,
            "consumed_global_batches": self.consumed_global_batches(),
            "signature": self.resume_signature(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        state = _require_mapping(state, label="sampler state")
        _require_exact_keys(
            state,
            required={
                "schema_version",
                "sampler_epoch",
                "start_offset",
                "consumed_global_batches",
                "signature",
            },
            label="sampler state",
        )
        if state["schema_version"] != self._STATE_VERSION:
            raise SoftClusterScheduleError(
                f"unsupported sampler state version: {state['schema_version']!r}"
            )
        if state["signature"] != self.resume_signature():
            raise SoftClusterScheduleError("sampler state signature mismatch")
        consumed = _require_int(
            state["consumed_global_batches"],
            label="sampler state consumed_global_batches",
            minimum=0,
        )
        expected_epoch, expected_offset = sampler_cursor_from_consumed_batches(
            consumed,
            epoch_size=self.num_samples,
            world_size=self.num_replicas,
            rank=self.rank,
        )
        if (state["sampler_epoch"], state["start_offset"]) != (
            expected_epoch,
            expected_offset,
        ):
            raise SoftClusterScheduleError("sampler state cursor is inconsistent")
        self.set_consumed_global_batches(consumed)

    def validate_legacy_resume_signature(
        self,
        saved_signature: Mapping[str, Any],
        *,
        consumed_global_batches: int,
    ) -> bool:
        """Validate the one-time legacy hierarchical-to-schedule migration."""

        source = self.schedule.info.source_checkpoint
        if source["mode"] != "checkpoint_migration":
            raise SoftClusterScheduleError(
                "legacy sampler migration requires a checkpoint_migration schedule"
            )
        consumed = _require_int(
            consumed_global_batches,
            label="consumed_global_batches",
            minimum=0,
        )
        origin = self.schedule.info.origin_consumed_global_batches
        if consumed != origin or consumed != source["next_global_step"]:
            raise SoftClusterScheduleError(
                "legacy sampler migration cursor does not match the bound checkpoint: "
                f"consumed={consumed}, origin={origin}"
            )
        migration = self.schedule.info.catalog_migration
        dataset = (
            migration["source_dataset"]["binding"]
            if migration is not None
            else self.schedule.info.dataset
        )
        self._validate_hierarchical_signature(
            saved_signature,
            dataset=dataset,
            label="legacy sampler signature",
        )
        return True

    def _validate_hierarchical_signature(
        self,
        signature: Mapping[str, Any],
        *,
        dataset: Mapping[str, Any],
        label: str,
    ) -> None:
        saved = dict(_require_mapping(signature, label=label))
        required = {
            "sampler_type",
            "layout",
            "seed",
            "num_samples",
            "replacement",
            "num_replicas",
            "rank",
            "emit_sample_occurrence",
            "dataset_offsets",
            "dataset_masses",
            "child_fingerprints",
        }
        _require_exact_keys(
            saved,
            required=required,
            label=label,
        )
        expected_values = {
            "sampler_type": "DistributedHierarchicalWeightedSampler",
            "layout": LEGACY_HIERARCHICAL_SAMPLER_LAYOUT,
            "seed": self.seed,
            "num_samples": self.num_samples,
            "replacement": True,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "emit_sample_occurrence": self.emit_sample_occurrence,
            "dataset_offsets": dataset["dataset_offsets"][:-1],
            "dataset_masses": dataset["dataset_masses"],
            "child_fingerprints": dataset["child_fingerprints"],
        }
        for field, expected_value in expected_values.items():
            if saved[field] != expected_value:
                raise SoftClusterScheduleError(
                    f"{label} mismatch for {field}: "
                    f"checkpoint={saved[field]!r}, schedule={expected_value!r}"
                )

    def validate_target_dataset_signature(
        self,
        signature: Mapping[str, Any],
    ) -> bool:
        """Validate the runtime target catalog before replacing its sampler.

        Schema-v2 schedules resume a checkpoint whose saved hierarchical
        signature describes the *source* catalog, while ``get_dataloaders``
        first constructs a hierarchical sampler for the *target* catalog.  The
        two signatures must therefore be authenticated against different
        bindings; conflating them would make every legitimate replacement look
        like unsafe resume drift.
        """

        self._validate_hierarchical_signature(
            signature,
            dataset=self.schedule.info.dataset,
            label="target sampler signature",
        )
        return True

    def validate_resume_signature(
        self,
        saved_signature: Mapping[str, Any],
        *,
        consumed_global_batches: int,
        allow_legacy_migration: bool = False,
    ) -> str:
        """Validate a normal resume or an explicitly authorized legacy migration."""

        self._validate_schedule_consumed_span(consumed_global_batches)
        if dict(saved_signature) == self.resume_signature():
            return "current"
        if not allow_legacy_migration:
            raise SoftClusterScheduleError(
                "train sampler configuration changed across resume"
            )
        self.validate_legacy_resume_signature(
            saved_signature,
            consumed_global_batches=consumed_global_batches,
        )
        return "legacy_migration"
