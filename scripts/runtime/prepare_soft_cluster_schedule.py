#!/usr/bin/env python3
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

"""Prepare an immutable soft-cluster schedule from a raw checkpoint boundary.

The builder is deliberately offline.  It replays the checkpoint's historical
legacy sampler on CPU, converts historical short/long row occurrences to
cluster exposure counts, and publishes an exact future wave-major row stream.
No structure, MSA, template, model parameter, or CUDA device is opened.

For future positions the complete legacy hierarchical epoch is generated first.
Only rows belonging to the explicitly selected clustered children are replaced.
Consequently the outer child sequence and every weightedPDB/disorder row remain
bit-for-bit identical to the legacy sampler stream.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import gzip
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protenix.data.pipeline.distillation_runtime_index import (  # noqa: E402
    DistillationRuntimeIndex,
)
from protenix.data.pipeline.soft_exposure_sampler import (  # noqa: E402
    SoftExposureBucketSampler,
    rotating_member_ordinal,
)
from scripts.runtime.audit_train_sampler_coverage import (  # noqa: E402
    DatasetCatalog,
    _effective_sampler_seed,
    _hierarchical_sampler,
    _load_json,
    _load_yaml,
    _world_size,
    build_catalogs,
    merge_initial_coverages,
    replay_sampler_counts,
)


LOGGER = logging.getLogger("prepare_soft_cluster_schedule")
FORMAT = "protenix.soft_cluster_schedule.v1"
SCHEMA_VERSION = 1
FORMAT_V2 = "protenix.soft_cluster_schedule.v2"
SCHEMA_VERSION_V2 = 2
LAYOUT = "wave_major_rank_v1"
ALGORITHM_FORMAT = "protenix.soft_exposure_bucket_sampler.v1"
MEMBER_POLICY = "deterministic_rotation_v1"
CONSUMPTION_ORDER = "synchronized_microstep_then_rank_ascending_v1"
CATALOG_MIGRATION_FORMAT = "protenix.catalog_replacement.v1"
CATALOG_MIGRATION_ALLOWED_DELTA = (
    "legacy_short_long_to_legacy_residual_ge200_v1"
)
CROSSWALK_FORMAT = "protenix.global_id_history_projection.v1"
GLOBAL_ID_POLICY = "source_prefix_pipe_raw_id_v1"
EXPOSURE_POLICY = "occurrence_sum_v1"
AMBIGUOUS_ALTERNATE_POLICY = "seen_floor_1_v1"

TARGET_RESIDUAL_DATASET = (
    "afdb_mgnify_legacy_residual_distillation_v20260804"
)
TARGET_GE200_DATASET = "all_sources_ge200_structural_distillation_v20260804"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _sha256_stable_file(
    path: Path, *, chunk_size: int = 8 * 1024 * 1024
) -> str:
    """Hash one file and reject mutation/replacement during the read."""

    path = path.resolve(strict=True)
    before = _file_identity(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        descriptor_before = os.fstat(handle.fileno())
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
        descriptor_after = os.fstat(handle.fileno())
    after = _file_identity(path)
    descriptor_identity_before = (
        descriptor_before.st_dev,
        descriptor_before.st_ino,
        descriptor_before.st_size,
        descriptor_before.st_mtime_ns,
    )
    descriptor_identity_after = (
        descriptor_after.st_dev,
        descriptor_after.st_ino,
        descriptor_after.st_size,
        descriptor_after.st_mtime_ns,
    )
    if not before == after == descriptor_identity_before == descriptor_identity_after:
        raise RuntimeError(f"File changed while it was being hashed: {path}")
    return digest.hexdigest()


def _artifact_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": resolved.relative_to(relative_to.resolve(strict=True)).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_stable_file(resolved),
    }


def _write_json_fsync(path: Path, document: Mapping[str, Any]) -> None:
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write while publishing {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def _atomic_output_directory(output_dir: Path) -> Iterator[Path]:
    """Build in a sibling temporary directory and publish once, fail closed."""

    output_dir = output_dir.resolve(strict=False)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_dir}")
    lock_path = output_dir.with_name(f".{output_dir.name}.prepare.lock")
    lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    temporary = output_dir.with_name(
        f".{output_dir.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        if output_dir.exists():
            raise FileExistsError(f"Output appeared after lock acquisition: {output_dir}")
        temporary.mkdir(mode=0o755)
        yield temporary
        if output_dir.exists():
            raise FileExistsError(f"Output appeared before publication: {output_dir}")
        directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(temporary, output_dir)
        parent_fd = os.open(output_dir.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _load_checkpoint_metadata(path: Path) -> Mapping[str, Any]:
    """Load checkpoint structure using fake tensors rather than model storage."""

    with FakeTensorMode():
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint payload must be a mapping: {path}")
    return checkpoint


def _required_int(mapping: Mapping[str, Any], key: str, *, label: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{label}.{key} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.{key} must be an integer") from exc
    return converted


def _infer_ema_checkpoint(
    *,
    raw_path: Path,
    raw_checkpoint: Mapping[str, Any],
    override: Path | None,
) -> Path:
    if override is not None:
        return override.expanduser().resolve(strict=True)
    step = _required_int(raw_checkpoint, "step", label="raw checkpoint")
    candidates = sorted(raw_path.parent.glob(f"{step}_ema_*.pt"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one EMA peer for raw step {step} beside "
            f"{raw_path}, found {len(candidates)}; pass --source-ema-checkpoint"
        )
    return candidates[0].resolve(strict=True)


def _validate_checkpoint_pair_headers(
    *,
    raw_checkpoint: Mapping[str, Any],
    ema_checkpoint: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    world_size: int,
) -> tuple[int, Mapping[str, Any]]:
    if raw_checkpoint.get("format_version") != 2:
        raise ValueError("source raw checkpoint must have format_version=2")
    if raw_checkpoint.get("checkpoint_kind") != "raw":
        raise ValueError("source checkpoint must have checkpoint_kind='raw'")
    if ema_checkpoint.get("format_version") != 2:
        raise ValueError("source EMA checkpoint must have format_version=2")
    if ema_checkpoint.get("checkpoint_kind") != "ema":
        raise ValueError("source EMA checkpoint must have checkpoint_kind='ema'")
    step = _required_int(raw_checkpoint, "step", label="raw checkpoint")
    if step < 0 or _required_int(ema_checkpoint, "step", label="EMA checkpoint") != step:
        raise ValueError("raw/EMA checkpoint steps must match and be non-negative")

    raw_contract = raw_checkpoint.get("resume_contract")
    ema_contract = ema_checkpoint.get("resume_contract")
    if not isinstance(raw_contract, Mapping) or not isinstance(ema_contract, Mapping):
        raise ValueError("raw and EMA checkpoints must contain resume_contract")
    required_contract_fields = (
        "checkpoint_pair_id",
        "training_contract_id",
        "data_contract_sha256",
        "run_id",
        "model_architecture_sha256",
        "step",
        "next_step",
        "next_global_step",
        "iters_to_accumulate",
        "world_size",
    )
    raw_missing = [
        field for field in required_contract_fields if field not in raw_contract
    ]
    ema_missing = [
        field for field in required_contract_fields if field not in ema_contract
    ]
    if raw_missing or ema_missing:
        raise ValueError(
            "raw/EMA resume contracts are missing fields: "
            f"raw={raw_missing}, EMA={ema_missing}"
        )
    # The raw and EMA files are one checkpoint pair and are created from the
    # exact same resume-contract object.  Compare the complete canonical
    # mapping, including future schema additions, rather than maintaining a
    # partial field allowlist that could silently become stale.
    if dict(raw_contract) != dict(ema_contract):
        mismatched = sorted(
            field
            for field in set(raw_contract) | set(ema_contract)
            if raw_contract.get(field) != ema_contract.get(field)
        )
        raise ValueError(f"raw/EMA resume contracts differ: {mismatched}")
    completed_updates = step + 1
    accumulation = int(config.get("iters_to_accumulate", 1))
    expectations = {
        "step": step,
        "next_step": completed_updates,
        "next_global_step": completed_updates * accumulation,
        "iters_to_accumulate": accumulation,
        "world_size": world_size,
        "run_id": run_manifest.get("run_name"),
    }
    data_contract = run_manifest.get("data", {}).get("contract", {})
    if not isinstance(data_contract, Mapping) or not data_contract.get("sha256"):
        raise ValueError("source run manifest has no bound data contract SHA")
    expectations["data_contract_sha256"] = data_contract["sha256"]
    contract_mismatches = {
        key: {"expected": value, "observed": raw_contract.get(key)}
        for key, value in expectations.items()
        if raw_contract.get(key) != value
    }
    if contract_mismatches:
        raise ValueError(
            "checkpoint/run contract mismatch: "
            + json.dumps(contract_mismatches, sort_keys=True)
        )
    return completed_updates, raw_contract


def _rank_cursor_binding(
    *,
    raw_checkpoint: Mapping[str, Any],
    expected_signature_rank0: Mapping[str, Any],
    completed_updates: int,
    accumulation: int,
    world_size: int,
    epoch_size: int,
) -> tuple[list[dict[str, Any]], str]:
    if epoch_size % world_size != 0:
        raise ValueError(
            "soft schedule migration currently requires epoch_size divisible "
            f"by world_size, got {epoch_size} and {world_size}"
        )
    runtime = raw_checkpoint.get("rank_runtime_states")
    if not isinstance(runtime, Mapping) or runtime.get("schema_version") != 1:
        raise ValueError("raw checkpoint has no supported rank_runtime_states")
    if _required_int(runtime, "world_size", label="rank runtime") != world_size:
        raise ValueError("rank runtime-state world size differs from source run")
    states = runtime.get("states")
    if not isinstance(states, Sequence) or len(states) != world_size:
        raise ValueError("rank runtime-state list does not match world size")
    expected_batches = completed_updates * accumulation
    local_epoch_size = epoch_size // world_size
    expected_epoch, expected_offset = divmod(expected_batches, local_epoch_size)
    cursors: list[dict[str, Any]] = []
    seen_ranks: set[int] = set()
    for state in states:
        if not isinstance(state, Mapping):
            raise ValueError("rank runtime state must be a mapping")
        rank = _required_int(state, "rank", label="rank runtime state")
        if not 0 <= rank < world_size or rank in seen_ranks:
            raise ValueError(f"invalid or duplicate rank runtime state: {rank}")
        seen_ranks.add(rank)
        dataloader = state.get("train_dataloader")
        if not isinstance(dataloader, Mapping) or dataloader.get("schema_version") != 1:
            raise ValueError(f"rank {rank} has no supported train dataloader state")
        observed_cursor = {
            "sampler_epoch": _required_int(
                dataloader, "sampler_epoch", label=f"rank {rank} dataloader"
            ),
            "next_local_offset": _required_int(
                dataloader, "next_local_offset", label=f"rank {rank} dataloader"
            ),
            "consumed_global_batches": _required_int(
                dataloader,
                "consumed_global_batches",
                label=f"rank {rank} dataloader",
            ),
        }
        batch_size = _required_int(
            dataloader,
            "batch_size",
            label=f"rank {rank} dataloader",
        )
        if batch_size != 1:
            raise ValueError(
                "soft schedule migration requires rank-local dataloader "
                f"batch_size=1, got rank {rank} batch_size={batch_size}"
            )
        expected_cursor = {
            "sampler_epoch": expected_epoch,
            "next_local_offset": expected_offset,
            "consumed_global_batches": expected_batches,
        }
        if observed_cursor != expected_cursor:
            raise ValueError(
                f"rank {rank} checkpoint cursor mismatch: "
                f"observed={observed_cursor}, expected={expected_cursor}"
            )
        signature = dataloader.get("sampler")
        if not isinstance(signature, Mapping):
            raise ValueError(f"rank {rank} dataloader has no sampler signature")
        expected_signature = dict(expected_signature_rank0)
        expected_signature["rank"] = rank
        if dict(signature) != expected_signature:
            raise ValueError(
                f"rank {rank} sampler signature differs from current run config"
            )
        cursors.append(
            {
                "rank": rank,
                **observed_cursor,
                "sampler_signature_sha256": _canonical_sha256(signature),
            }
        )
    cursors.sort(key=lambda record: record["rank"])
    return cursors, _canonical_sha256(cursors)


def _runtime_cluster_arrays(
    runtime_index: DistillationRuntimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    first_rows = np.frombuffer(
        runtime_index._map("first_row_ids"),
        dtype="<u8",
        count=runtime_index.cluster_count,
    )
    cluster_sizes = np.frombuffer(
        runtime_index._map("cluster_sizes"),
        dtype="<u4",
        count=runtime_index.cluster_count,
    )
    if (
        int(first_rows[0]) != 0
        or bool(np.any(cluster_sizes == 0))
        or bool(np.any(first_rows[1:] != first_rows[:-1] + cluster_sizes[:-1]))
        or int(first_rows[-1] + cluster_sizes[-1]) != runtime_index.row_count
    ):
        raise ValueError(
            f"Runtime cluster intervals do not exactly partition rows: "
            f"{runtime_index.runtime_dir}"
        )
    return first_rows, cluster_sizes


def row_counts_to_cluster_exposure(
    runtime_index: DistillationRuntimeIndex,
    row_counts: np.ndarray,
) -> np.ndarray:
    """Sum historical member-row occurrences for every compact cluster."""

    row_counts = np.asarray(row_counts)
    if row_counts.shape != (runtime_index.row_count,):
        raise ValueError(
            "row count vector shape differs from runtime index: "
            f"{row_counts.shape} != {(runtime_index.row_count,)}"
        )
    if not np.issubdtype(row_counts.dtype, np.unsignedinteger):
        raise TypeError("historical row counts must have an unsigned dtype")
    first_rows, _ = _runtime_cluster_arrays(runtime_index)
    exposure = np.add.reduceat(
        row_counts,
        first_rows.astype(np.int64, copy=False),
        dtype=np.uint64,
    )
    if exposure.size and int(exposure.max()) > np.iinfo(np.uint32).max:
        raise OverflowError("historical cluster exposure exceeds uint32")
    if int(exposure.sum(dtype=np.uint64)) != int(row_counts.sum(dtype=np.uint64)):
        raise RuntimeError("row-to-cluster occurrence accounting mismatch")
    return exposure.astype(np.uint32)


def project_global_id_occurrences(
    *,
    source_occurrences: Mapping[str, int],
    target_assignments: Mapping[str, tuple[str, int]],
    target_cluster_counts: Mapping[str, int],
    intentional_missing: Mapping[str, str] | None = None,
    ambiguity_alternates: Iterable[tuple[str, str, int]] = (),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Project historical member occurrences into replacement clusters.

    ``source_occurrences`` is keyed by the source-qualified global identity
    (for example ``AFQ|AF-A0A...``).  Every identity must map to exactly one
    target member, or be explicitly allowlisted as intentionally missing.
    Historical counts are summed at the target cluster.  For the small set of
    AFQ/AFU raw-name collisions, a seen source identity may additionally place
    a conservative exposure floor of one on the recorded alternate cluster.

    This function is deliberately independent of TSV/runtime-index I/O so the
    accounting contract can be exhaustively exercised with tiny fixtures.
    """

    missing_reasons = dict(intentional_missing or {})
    exposures = {
        name: np.zeros(int(cluster_count), dtype=np.uint32)
        for name, cluster_count in target_cluster_counts.items()
    }
    if not exposures or any(values.size <= 0 for values in exposures.values()):
        raise ValueError("target cluster counts must be positive")

    source_total = 0
    mapped_total = 0
    intentional_missing_total = 0
    intentional_missing_rows = 0
    assignments_seen: dict[str, tuple[str, int]] = {}
    missing_by_reason: dict[str, dict[str, int]] = {}
    for global_id, raw_count in source_occurrences.items():
        if not isinstance(global_id, str) or "|" not in global_id:
            raise ValueError(f"invalid source-qualified global ID: {global_id!r}")
        count = int(raw_count)
        if count <= 0:
            raise ValueError(
                f"historical occurrence count must be positive: {global_id!r}"
            )
        source_total += count
        assignment = target_assignments.get(global_id)
        if assignment is None:
            reason = missing_reasons.get(global_id)
            if not reason:
                raise ValueError(
                    "historical global ID is absent from the target union and "
                    f"is not intentional-missing: {global_id}"
                )
            intentional_missing_total += count
            intentional_missing_rows += 1
            record = missing_by_reason.setdefault(
                reason,
                {"unique_source_rows": 0, "historical_occurrences": 0},
            )
            record["unique_source_rows"] += 1
            record["historical_occurrences"] += count
            continue
        if global_id in missing_reasons:
            raise ValueError(
                "intentional-missing allowlist contains a mapped target identity: "
                f"{global_id}"
            )
        target_name, raw_cluster = assignment
        values = exposures.get(target_name)
        if values is None:
            raise ValueError(
                f"target assignment references unknown dataset {target_name!r}"
            )
        cluster = int(raw_cluster)
        if cluster < 0 or cluster >= values.size:
            raise ValueError(
                f"target cluster ordinal is outside {target_name}: {cluster}"
            )
        updated = int(values[cluster]) + count
        if updated > np.iinfo(np.uint32).max:
            raise OverflowError("projected cluster exposure exceeds uint32")
        values[cluster] = updated
        mapped_total += count
        assignments_seen[global_id] = (target_name, cluster)

    alternate_records = 0
    alternate_seen_records = 0
    alternate_floor_increments = 0
    alternate_clusters: set[tuple[str, int]] = set()
    seen_alternate_keys: set[tuple[str, str, int]] = set()
    for global_id, target_name, raw_cluster in ambiguity_alternates:
        key = (str(global_id), str(target_name), int(raw_cluster))
        if key in seen_alternate_keys:
            raise ValueError(f"duplicate ambiguity alternate record: {key!r}")
        seen_alternate_keys.add(key)
        alternate_records += 1
        if global_id not in assignments_seen:
            continue
        values = exposures.get(target_name)
        if values is None:
            raise ValueError(
                f"ambiguity alternate references unknown dataset {target_name!r}"
            )
        cluster = int(raw_cluster)
        if cluster < 0 or cluster >= values.size:
            raise ValueError(
                f"ambiguity alternate cluster is outside {target_name}: {cluster}"
            )
        canonical_target = assignments_seen[global_id]
        if canonical_target == (target_name, cluster):
            raise ValueError(
                "ambiguity alternate equals the canonical target assignment: "
                f"{key!r}"
            )
        alternate_seen_records += 1
        alternate_clusters.add((target_name, cluster))
        if int(values[cluster]) == 0:
            values[cluster] = 1
            alternate_floor_increments += 1

    projected_total = sum(
        int(values.sum(dtype=np.uint64)) for values in exposures.values()
    )
    if source_total != mapped_total + intentional_missing_total:
        raise RuntimeError("source history projection does not conserve occurrences")
    if projected_total != mapped_total + alternate_floor_increments:
        raise RuntimeError("target exposure projection accounting mismatch")
    audit = {
        "schema_version": 1,
        "event": "source_target_catalog_history_projection",
        "global_id_policy": GLOBAL_ID_POLICY,
        "exposure_policy": EXPOSURE_POLICY,
        "ambiguous_alternate_policy": AMBIGUOUS_ALTERNATE_POLICY,
        "source_unique_seen_members": len(source_occurrences),
        "source_history_occurrences": source_total,
        "mapped_unique_members": len(assignments_seen),
        "mapped_occurrences": mapped_total,
        "intentional_missing_unique_members": intentional_missing_rows,
        "intentional_missing_occurrences": intentional_missing_total,
        "intentional_missing_by_reason": missing_by_reason,
        "unresolved_unique_members": 0,
        "unresolved_occurrences": 0,
        "ambiguous_alternate_records": alternate_records,
        "ambiguous_seen_records": alternate_seen_records,
        "ambiguous_alternate_clusters": len(alternate_clusters),
        "ambiguous_floor_increment": alternate_floor_increments,
        "target_projected_occurrences": mapped_total,
        "target_initial_exposure_total": projected_total,
        "target_exposure": {
            name: _exposure_stats(values) for name, values in exposures.items()
        },
    }
    return exposures, audit


def _runtime_binding(runtime_index: DistillationRuntimeIndex) -> dict[str, Any]:
    manifest_path = runtime_index.runtime_dir / "manifest.json"
    sampling_records: dict[str, Any] = {}
    declared_arrays = runtime_index._manifest["sampling"]["arrays"]
    for name in ("cluster_ids", "first_row_ids", "cluster_sizes"):
        path = runtime_index._array_paths[name]
        observed = _sha256_stable_file(path)
        declared = str(declared_arrays[name]["sha256"]).lower()
        if observed != declared:
            raise ValueError(
                f"Runtime sampling artifact SHA mismatch for {path}: "
                f"{observed} != {declared}"
            )
        sampling_records[name] = {
            "path": str(path.resolve(strict=True)),
            "size_bytes": path.stat().st_size,
            "sha256": observed,
        }
    return {
        "runtime_dir": str(runtime_index.runtime_dir),
        "row_count": runtime_index.row_count,
        "cluster_count": runtime_index.cluster_count,
        "manifest_path": str(manifest_path.resolve(strict=True)),
        "manifest_sha256": _sha256_stable_file(manifest_path),
        "sampling_fingerprint": runtime_index.sampling_fingerprint,
        "sampling_artifacts": sampling_records,
    }


def _declared_runtime_artifact(
    runtime_index: DistillationRuntimeIndex,
    *,
    name: str,
    required: bool,
) -> dict[str, Any] | None:
    """Resolve and hash one immutable target-catalog artifact."""

    record = runtime_index._manifest.get("artifacts", {}).get(name)
    if record is None and not required:
        return None
    if not isinstance(record, Mapping):
        raise ValueError(
            f"Target runtime manifest lacks artifacts.{name}: "
            f"{runtime_index.runtime_dir}"
        )
    required_fields = {"path", "size_bytes", "sha256", "count", "dtype", "columns"}
    missing = required_fields - set(record)
    if missing:
        raise ValueError(
            f"Target runtime artifacts.{name} lacks fields: {sorted(missing)}"
        )
    relative = Path(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ValueError(f"Unsafe target runtime artifact path: {relative}")
    path = runtime_index.runtime_dir / relative
    path = path.resolve(strict=True)
    if not path.is_relative_to(runtime_index.runtime_dir):
        raise ValueError(f"Target runtime artifact escapes runtime directory: {path}")
    size = int(record["size_bytes"])
    count = int(record["count"])
    declared_sha = str(record["sha256"]).lower()
    if size < 0 or count < 0 or path.stat().st_size != size:
        raise ValueError(f"Invalid size/count for target runtime artifact: {path}")
    if record["dtype"] != "tsv_gzip":
        raise ValueError(f"Target lookup must use dtype='tsv_gzip': {path}")
    if len(declared_sha) != 64 or any(char not in "0123456789abcdef" for char in declared_sha):
        raise ValueError(f"Invalid target runtime artifact SHA-256: {path}")
    observed_sha = _sha256_stable_file(path)
    if observed_sha != declared_sha:
        raise ValueError(
            f"Target runtime artifact SHA mismatch for {path}: "
            f"{observed_sha} != {declared_sha}"
        )
    return {
        "path": str(path),
        "size_bytes": size,
        "sha256": declared_sha,
        "count": count,
        "dtype": "tsv_gzip",
        "columns": list(record["columns"]),
    }


def _target_lookup_binding(
    *,
    dataset_name: str,
    runtime_index: DistillationRuntimeIndex,
) -> dict[str, Any]:
    target_catalog = runtime_index._manifest.get("target_catalog")
    if not isinstance(target_catalog, Mapping):
        raise ValueError(
            f"Target runtime has no target_catalog contract: {runtime_index.runtime_dir}"
        )
    required = {
        "schema_version",
        "role",
        "pair_id",
        "pair_receipt",
        "set_operation",
        "counts",
        "source_identity",
    }
    if set(target_catalog) != required:
        raise ValueError(
            "Target runtime target_catalog fields differ from the v1 contract: "
            f"{sorted(set(target_catalog))}"
        )
    if target_catalog["schema_version"] != 1:
        raise ValueError("Target runtime target_catalog schema_version must be 1")
    role = str(target_catalog["role"])
    expected_role = {
        TARGET_RESIDUAL_DATASET: "legacy_residual",
        TARGET_GE200_DATASET: "ge200_structural",
    }.get(dataset_name)
    if expected_role is None or role != expected_role:
        raise ValueError(
            f"Target dataset {dataset_name!r} requires role {expected_role!r}, "
            f"got {role!r}"
        )
    pair_id = str(target_catalog["pair_id"])
    if len(pair_id) != 64 or any(char not in "0123456789abcdef" for char in pair_id):
        raise ValueError("target_catalog.pair_id must be a lowercase SHA-256")
    pair_receipt = target_catalog["pair_receipt"]
    if not isinstance(pair_receipt, Mapping) or set(pair_receipt) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise ValueError("target_catalog.pair_receipt has an invalid record")
    receipt_relative = Path(str(pair_receipt["path"]))
    if receipt_relative.is_absolute() or ".." in receipt_relative.parts:
        raise ValueError("target_catalog.pair_receipt path is unsafe")
    receipt_path = (runtime_index.runtime_dir / receipt_relative).resolve(strict=True)
    if not receipt_path.is_relative_to(runtime_index.runtime_dir):
        raise ValueError("target_catalog.pair_receipt escapes runtime directory")
    receipt_sha = _sha256_stable_file(receipt_path)
    if (
        receipt_path.stat().st_size != int(pair_receipt["size_bytes"])
        or receipt_sha != str(pair_receipt["sha256"]).lower()
    ):
        raise ValueError("target_catalog pair receipt binding mismatch")

    lookup = _declared_runtime_artifact(
        runtime_index,
        name="member_to_cluster_lookup",
        required=True,
    )
    ambiguity = _declared_runtime_artifact(
        runtime_index,
        name="ambiguity_alternates",
        required=role == "ge200_structural",
    )
    assert lookup is not None
    expected_lookup_columns = [
        "runtime_row_id",
        "global_id",
        "cluster_ordinal",
        "stable_cluster_id",
        "source_member_row_id",
        "source_partition",
        "cluster_representative_length_bucket",
    ]
    if lookup["columns"] != expected_lookup_columns:
        raise ValueError("member_to_cluster lookup columns differ from contract")
    if lookup["count"] != runtime_index.row_count:
        raise ValueError("member_to_cluster lookup must contain every runtime row")
    if ambiguity is not None and ambiguity["columns"] != [
        "global_id",
        "canonical_cluster_ordinal",
        "alternate_cluster_ordinal",
        "raw_id",
        "resolution_rule",
    ]:
        raise ValueError("ambiguity_alternates columns differ from contract")
    manifest_path = runtime_index.runtime_dir / "manifest.json"
    return {
        "role": role,
        "runtime_manifest_path": str(manifest_path.resolve(strict=True)),
        "runtime_manifest_sha256": _sha256_stable_file(manifest_path),
        "pair_id": pair_id,
        "pair_receipt_path": str(receipt_path),
        "pair_receipt_size_bytes": receipt_path.stat().st_size,
        "pair_receipt_sha256": receipt_sha,
        "member_to_cluster": lookup,
        "ambiguity_alternates": ambiguity,
    }


def _runtime_source_prefixes(
    runtime_index: DistillationRuntimeIndex,
) -> dict[str, str]:
    roots = _load_json(runtime_index.roots_path)
    records = roots.get("roots")
    if not isinstance(records, Sequence):
        raise ValueError(f"Runtime roots has no roots list: {runtime_index.roots_path}")
    prefixes: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Runtime roots entry must be an object")
        root_id = str(record.get("root_id", ""))
        prefix = str(record.get("source_prefix", ""))
        if root_id in prefixes or not root_id or not prefix.endswith("|"):
            raise ValueError(
                f"Runtime roots lacks a unique source_prefix for {root_id!r}"
            )
        prefixes[root_id] = prefix
    if set(prefixes) != set(runtime_index.root_ids):
        raise ValueError("Runtime source-prefix roots differ from runtime root IDs")
    return prefixes


def collect_seen_source_global_occurrences(
    *,
    catalogs: Sequence[DatasetCatalog],
    history_counts: Sequence[np.ndarray],
    source_soft_datasets: Sequence[str],
) -> tuple[dict[str, int], dict[str, Any]]:
    """Convert nonzero legacy Short/Long rows into qualified global IDs."""

    if len(catalogs) != len(history_counts):
        raise ValueError("catalog/history count lengths differ")
    selected = set(source_soft_datasets)
    if len(selected) != 2:
        raise ValueError("source catalog migration requires exactly two datasets")
    occurrences: dict[str, int] = {}
    dataset_audit: dict[str, Any] = {}
    for catalog, counts in zip(catalogs, history_counts):
        if catalog.name not in selected:
            continue
        runtime = catalog.runtime_index
        if runtime is None:
            raise ValueError(f"Source dataset is not runtime indexed: {catalog.name}")
        counts = np.asarray(counts)
        if counts.shape != (runtime.row_count,):
            raise ValueError(f"Source history shape differs for {catalog.name}")
        prefixes = _runtime_source_prefixes(runtime)
        seen_rows = np.flatnonzero(counts)
        dataset_occurrences = 0
        duplicate_global_ids = 0
        for row_id in seen_rows.tolist():
            count = int(counts[row_id])
            root_id, cif_relpath = runtime.get_address(row_id)
            raw_id = Path(cif_relpath).stem
            if not raw_id or "\t" in raw_id or "\n" in raw_id:
                raise ValueError(
                    f"Invalid source raw ID at {catalog.name} row {row_id}"
                )
            global_id = f"{prefixes[root_id]}{raw_id}"
            if global_id in occurrences:
                duplicate_global_ids += 1
            occurrences[global_id] = occurrences.get(global_id, 0) + count
            dataset_occurrences += count
        dataset_audit[catalog.name] = {
            "catalog_rows": runtime.row_count,
            "unique_seen_rows": int(seen_rows.size),
            "historical_occurrences": dataset_occurrences,
            "duplicate_global_ids_across_source_rows": duplicate_global_ids,
        }
    if set(dataset_audit) != selected:
        raise ValueError(
            f"Source soft datasets are absent from source catalogs: "
            f"{sorted(selected - set(dataset_audit))}"
        )
    source_total = sum(
        int(record["historical_occurrences"])
        for record in dataset_audit.values()
    )
    if source_total != sum(occurrences.values()):
        raise RuntimeError("source global-ID occurrence aggregation is inconsistent")
    return occurrences, {
        "datasets": dataset_audit,
        "unique_seen_global_ids": len(occurrences),
        "historical_occurrences": source_total,
    }


def _scan_target_member_lookups(
    *,
    source_global_ids: set[str],
    target_catalogs: Sequence[DatasetCatalog],
    lookup_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[str, int]]:
    assignments: dict[str, tuple[str, int]] = {}
    expected_header = [
        "runtime_row_id",
        "global_id",
        "cluster_ordinal",
        "stable_cluster_id",
        "source_member_row_id",
        "source_partition",
        "cluster_representative_length_bucket",
    ]
    for catalog in target_catalogs:
        binding = lookup_bindings[catalog.name]
        runtime = catalog.runtime_index
        assert runtime is not None
        lookup = binding["member_to_cluster"]
        path = Path(lookup["path"])
        observed_rows = 0
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != expected_header:
                raise ValueError(f"Target lookup header differs from contract: {path}")
            for expected_row, record in enumerate(reader):
                observed_rows += 1
                row_id = int(record["runtime_row_id"])
                if row_id != expected_row:
                    raise ValueError(
                        f"Target lookup is not in runtime row order at {path}: "
                        f"{row_id} != {expected_row}"
                    )
                global_id = record["global_id"]
                if global_id not in source_global_ids:
                    continue
                cluster = int(record["cluster_ordinal"])
                stable_cluster = int(record["stable_cluster_id"])
                if not 0 <= cluster < runtime.cluster_count:
                    raise ValueError(f"Target lookup cluster outside runtime: {path}")
                if runtime.get_cluster_id(cluster) != stable_cluster:
                    raise ValueError(
                        f"Target lookup stable cluster ID mismatch: {path}"
                    )
                assignment = (catalog.name, cluster)
                previous = assignments.get(global_id)
                if previous is not None:
                    raise ValueError(
                        "Target residual/ge200 union contains duplicate source "
                        f"identity {global_id!r}: {previous!r} vs {assignment!r}"
                    )
                assignments[global_id] = assignment
        if observed_rows != runtime.row_count or observed_rows != int(lookup["count"]):
            raise ValueError(
                f"Target lookup row count mismatch for {path}: {observed_rows}"
            )
    return assignments


def _scan_ambiguity_alternates(
    *,
    ge200_dataset_name: str,
    binding: Mapping[str, Any],
    cluster_count: int,
    target_assignments: Mapping[str, tuple[str, int]],
) -> tuple[list[tuple[str, str, int]], dict[str, int]]:
    artifact = binding.get("ambiguity_alternates")
    if not isinstance(artifact, Mapping):
        raise ValueError("ge200 target lookup lacks ambiguity_alternates")
    path = Path(artifact["path"])
    expected_header = [
        "global_id",
        "canonical_cluster_ordinal",
        "alternate_cluster_ordinal",
        "raw_id",
        "resolution_rule",
    ]
    records: list[tuple[str, str, int]] = []
    raw_ids: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != expected_header:
            raise ValueError("ambiguity_alternates header differs from contract")
        for record in reader:
            global_id = record["global_id"]
            canonical = int(record["canonical_cluster_ordinal"])
            alternate = int(record["alternate_cluster_ordinal"])
            if (
                canonical < 0
                or alternate < 0
                or canonical >= cluster_count
                or alternate >= cluster_count
                or canonical == alternate
            ):
                raise ValueError("invalid canonical/alternate ambiguity clusters")
            raw_id = record["raw_id"]
            if not global_id.endswith(f"|{raw_id}"):
                raise ValueError("ambiguity raw_id does not match global_id")
            raw_ids.add(raw_id)
            assignment = target_assignments.get(global_id)
            if assignment is not None and assignment != (
                ge200_dataset_name,
                canonical,
            ):
                raise ValueError(
                    "ambiguity canonical cluster differs from target lookup for "
                    f"{global_id}: {canonical} != {assignment}"
                )
            records.append((global_id, ge200_dataset_name, alternate))
    if len(records) != int(artifact["count"]):
        raise ValueError("ambiguity_alternates count differs from manifest")
    return records, {
        "artifact_records": len(records),
        "ambiguous_raw_ids": len(raw_ids),
    }


def _load_intentional_missing(
    path: Path | None,
) -> tuple[dict[str, str], dict[str, Any] | None]:
    if path is None:
        return {}, None
    resolved = path.expanduser().resolve(strict=True)
    opener = gzip.open if resolved.suffix == ".gz" else open
    values: dict[str, str] = {}
    with opener(resolved, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["global_id", "reason"]:
            raise ValueError(
                "intentional-missing TSV must have global_id and reason columns"
            )
        for record in reader:
            global_id = record["global_id"]
            reason = record["reason"]
            if not global_id or "|" not in global_id or not reason:
                raise ValueError("invalid intentional-missing record")
            if global_id in values:
                raise ValueError(f"duplicate intentional-missing ID: {global_id}")
            values[global_id] = reason
    return values, {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_stable_file(resolved),
        "count": len(values),
        "columns": ["global_id", "reason"],
    }


def _derive_seed(base_seed: int, dataset_name: str, stream: str) -> int:
    material = f"{int(base_seed)}\0{dataset_name}\0{stream}\0{FORMAT}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def _build_dataset_binding(
    *,
    catalogs: Sequence[DatasetCatalog],
    masses: Sequence[float],
    epoch_size: int,
    world_size: int,
    seed: int,
    runtime_bindings: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    offsets = np.cumsum(
        np.asarray([0] + [catalog.size for catalog in catalogs], dtype=np.int64)
    ).tolist()
    hierarchical = _hierarchical_sampler(
        catalogs=catalogs,
        masses=masses,
        epoch_size=epoch_size,
        world_size=world_size,
        seed=seed,
    )
    if hierarchical is None:
        raise ValueError("catalog migration requires a hierarchical sampler")
    signature = hierarchical.resume_signature()
    binding = {
        "flat_row_count": offsets[-1],
        "train_sets": [catalog.name for catalog in catalogs],
        "dataset_offsets": offsets,
        "dataset_masses": [float(value) for value in masses],
        "child_fingerprints": signature["child_fingerprints"],
        "runtime_bindings": dict(runtime_bindings),
    }
    return binding, signature


def _validate_target_lookup_pair(
    lookup_bindings: Mapping[str, Mapping[str, Any]],
) -> None:
    pair_ids = {str(binding["pair_id"]) for binding in lookup_bindings.values()}
    receipt_hashes = {
        str(binding["pair_receipt_sha256"]) for binding in lookup_bindings.values()
    }
    if len(pair_ids) != 1 or len(receipt_hashes) != 1:
        raise ValueError(
            "Target residual/ge200 runtime indices do not share one catalog-pair receipt"
        )


def _identify_soft_catalogs(
    catalogs: Sequence[DatasetCatalog],
    requested_names: Sequence[str],
) -> list[DatasetCatalog]:
    catalog_by_name = {catalog.name: catalog for catalog in catalogs}
    if requested_names:
        names = list(requested_names)
    else:
        short = [
            catalog.name
            for catalog in catalogs
            if "short" in catalog.name.lower() and catalog.runtime_index is not None
        ]
        long = [
            catalog.name
            for catalog in catalogs
            if "long" in catalog.name.lower() and catalog.runtime_index is not None
        ]
        if len(short) != 1 or len(long) != 1:
            raise ValueError(
                "Could not identify exactly one short and one long clustered "
                "dataset; pass --soft-dataset twice"
            )
        names = [short[0], long[0]]
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError("exactly two distinct --soft-dataset names are required")
    selected: list[DatasetCatalog] = []
    for name in names:
        catalog = catalog_by_name.get(name)
        if catalog is None:
            raise ValueError(f"soft dataset is absent from train_sets: {name!r}")
        if catalog.runtime_index is None:
            raise ValueError(f"soft dataset is not runtime-clustered: {name!r}")
        if not (
            isinstance(catalog.sampling_spec, Mapping)
            and catalog.sampling_spec.get("kind") == "clustered"
        ):
            raise ValueError(f"soft dataset has no clustered sampler: {name!r}")
        selected.append(catalog)
    return selected


def _replay_history_segment(
    *,
    catalogs: Sequence[DatasetCatalog],
    masses: Sequence[float],
    start_update: int,
    completed_updates: int,
    accumulation: int,
    world_size: int,
    epoch_size: int,
    seed: int,
    replacement: bool,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    end_counts, end_replay = replay_sampler_counts(
        catalogs=catalogs,
        masses=masses,
        updates=completed_updates,
        accumulation=accumulation,
        world_size=world_size,
        epoch_size=epoch_size,
        seed=seed,
        replacement=replacement,
    )
    if start_update == 0:
        return end_counts, {
            "start_update": 0,
            "completed_updates": completed_updates,
            "end_prefix": end_replay,
            "start_prefix": None,
        }
    start_counts, start_replay = replay_sampler_counts(
        catalogs=catalogs,
        masses=masses,
        updates=start_update,
        accumulation=accumulation,
        world_size=world_size,
        epoch_size=epoch_size,
        seed=seed,
        replacement=replacement,
    )
    segment: list[np.ndarray] = []
    for catalog, end, start in zip(catalogs, end_counts, start_counts):
        if bool(np.any(start > end)):
            raise RuntimeError(f"replay prefix underflow for dataset {catalog.name}")
        segment.append((end.astype(np.uint64) - start).astype(np.uint32))
    return segment, {
        "start_update": start_update,
        "completed_updates": completed_updates,
        "end_prefix": end_replay,
        "start_prefix": start_replay,
    }


def _bind_initial_coverage_artifacts(
    merged_coverages: Sequence[dict[str, Any]],
) -> None:
    """Bind every sparse row-count artifact consumed by coverage merging."""

    for record in merged_coverages:
        coverage_dir = Path(record["coverage_dir"]).resolve(strict=True)
        summary_path = Path(record["summary_path"]).resolve(strict=True)
        summary = _load_json(summary_path)
        datasets = summary.get("datasets")
        if not isinstance(datasets, Mapping):
            raise ValueError(f"initial coverage has no dataset records: {summary_path}")
        bindings: dict[str, Any] = {}
        for name in record["datasets"]:
            dataset = datasets.get(name)
            if not isinstance(dataset, Mapping):
                raise ValueError(f"initial coverage lacks dataset {name!r}")
            artifacts = dataset.get("artifacts")
            if not isinstance(artifacts, Mapping):
                raise ValueError(f"initial coverage dataset {name!r} lacks artifacts")
            relative = artifacts.get("seen_rows_npz")
            if not isinstance(relative, str) or Path(relative).name != relative:
                raise ValueError(
                    f"initial coverage dataset {name!r} has unsafe seen_rows_npz"
                )
            path = (coverage_dir / relative).resolve(strict=True)
            if not path.is_relative_to(coverage_dir) or not path.is_file():
                raise ValueError(f"invalid initial coverage artifact: {path}")
            bindings[name] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_stable_file(path),
            }
        record["consumed_artifacts"] = bindings


def _snapshot_initial_coverage_inputs(
    coverage_dirs: Sequence[Path],
) -> dict[Path, dict[str, Any]]:
    """Hash coverage inputs before merging so concurrent changes are rejected."""

    snapshots: dict[Path, dict[str, Any]] = {}
    for raw_dir in coverage_dirs:
        coverage_dir = raw_dir.expanduser().resolve(strict=True)
        summary_path = (coverage_dir / "summary.json").resolve(strict=True)
        summary = _load_json(summary_path)
        datasets = summary.get("datasets")
        if not isinstance(datasets, Mapping):
            raise ValueError(f"initial coverage has no datasets: {summary_path}")
        paths = [summary_path]
        for name, dataset in datasets.items():
            if not isinstance(dataset, Mapping):
                raise ValueError(
                    f"initial coverage dataset {name!r} is not a mapping"
                )
            artifacts = dataset.get("artifacts")
            if not isinstance(artifacts, Mapping):
                raise ValueError(
                    f"initial coverage dataset {name!r} lacks artifacts"
                )
            relative = artifacts.get("seen_rows_npz")
            if not isinstance(relative, str) or Path(relative).name != relative:
                raise ValueError(
                    f"initial coverage dataset {name!r} has unsafe seen_rows_npz"
                )
            path = (coverage_dir / relative).resolve(strict=True)
            if not path.is_relative_to(coverage_dir) or not path.is_file():
                raise ValueError(f"invalid initial coverage artifact: {path}")
            paths.append(path)
        for path in paths:
            snapshots[path] = {
                "identity": _file_identity(path),
                "sha256": _sha256_stable_file(path),
            }
    return snapshots


def _verify_hashed_input_snapshots(
    snapshots: Mapping[Path, Mapping[str, Any]],
) -> None:
    changed: list[str] = []
    for path, record in snapshots.items():
        if _file_identity(path) != record["identity"]:
            changed.append(str(path))
            continue
        if _sha256_stable_file(path) != record["sha256"]:
            changed.append(str(path))
    if changed:
        raise RuntimeError(
            "Bound sampler inputs changed while preparing the schedule: "
            + ", ".join(changed)
        )


def _validate_initial_coverage_boundary(
    *,
    coverage_dirs: Sequence[Path],
    history_start_update: int,
    accumulation: int,
    world_size: int,
) -> None:
    """Require one provable prefix rather than silently unioning overlap."""

    if history_start_update == 0:
        if coverage_dirs:
            raise ValueError(
                "--initial-coverage is forbidden when history-start-update=0; "
                "it would double-count the replayed prefix"
            )
        return
    if len(coverage_dirs) != 1:
        raise ValueError(
            "history-start-update > 0 currently requires exactly one "
            "--initial-coverage representing [0, history-start-update); "
            "multiple segments cannot be proven non-overlapping"
        )
    coverage_dir = coverage_dirs[0].expanduser().resolve(strict=True)
    summary_path = coverage_dir / "summary.json"
    summary = _load_json(summary_path)
    if summary.get("event") != "planned_train_sampler_coverage":
        raise ValueError(f"not a planned sampler coverage prefix: {summary_path}")
    if _required_int(summary, "updates", label="initial coverage summary") != (
        history_start_update
    ):
        raise ValueError(
            "initial coverage updates must exactly equal history-start-update"
        )
    expected_occurrences = history_start_update * accumulation * world_size
    observed_occurrences = _required_int(
        summary,
        "coverage_total_occurrences",
        label="initial coverage summary",
    )
    if observed_occurrences != expected_occurrences:
        raise ValueError(
            "initial coverage occurrence count does not prove the complete "
            f"prefix: observed={observed_occurrences}, "
            f"expected={expected_occurrences}"
        )


def _rows_for_soft_draws(
    *,
    runtime_index: DistillationRuntimeIndex,
    first_rows: np.ndarray,
    cluster_sizes: np.ndarray,
    draws: Sequence[Any],
    member_seed: int,
) -> np.ndarray:
    if not draws:
        return np.empty(0, dtype=np.int64)
    ordinals = np.fromiter(
        (draw.cluster_ordinal for draw in draws),
        dtype=np.int64,
        count=len(draws),
    )
    previous = np.fromiter(
        (draw.previous_exposure_count for draw in draws),
        dtype=np.uint32,
        count=len(draws),
    )
    selected_sizes = cluster_sizes[ordinals]
    member_offsets = np.fromiter(
        (
            rotating_member_ordinal(
                cluster_ordinal=int(ordinal),
                previous_exposure_count=int(exposure),
                member_count=int(size),
                seed=member_seed,
            )
            for ordinal, exposure, size in zip(ordinals, previous, selected_sizes)
        ),
        dtype=np.uint64,
        count=len(draws),
    )
    rows = first_rows[ordinals] + member_offsets
    if bool(np.any(rows >= runtime_index.row_count)):
        raise RuntimeError("soft sampler produced a row outside its runtime index")
    return rows.astype(np.int64, copy=False)


def generate_global_rows(
    *,
    output_path: Path,
    catalogs: Sequence[DatasetCatalog],
    masses: Sequence[float],
    soft_samplers: Mapping[str, SoftExposureBucketSampler],
    member_seeds: Mapping[str, int],
    origin_consumed_batches_per_rank: int,
    target_consumed_batches_per_rank: int,
    accumulation: int,
    world_size: int,
    epoch_size: int,
    seed: int,
) -> dict[str, int]:
    """Write the exact future wave-major flattened-row stream."""

    del accumulation  # The caller already expresses both boundaries in waves.
    if target_consumed_batches_per_rank <= origin_consumed_batches_per_rank:
        raise ValueError("target sampler wave must be after source wave")
    if epoch_size % world_size != 0:
        raise ValueError("wave-major generation requires epoch_size % world_size == 0")
    offsets = np.cumsum(
        np.asarray([0] + [catalog.size for catalog in catalogs], dtype=np.int64)
    )
    if int(offsets[-1]) > np.iinfo(np.uint32).max + 1:
        raise OverflowError("flattened dataset rows do not fit uint32")
    hierarchical = _hierarchical_sampler(
        catalogs=catalogs,
        masses=masses,
        epoch_size=epoch_size,
        world_size=world_size,
        seed=seed,
    )
    if hierarchical is None:
        raise ValueError("soft schedule requires a hierarchical sampler")
    runtime_arrays = {
        catalog.name: _runtime_cluster_arrays(catalog.runtime_index)
        for catalog in catalogs
        if catalog.name in soft_samplers and catalog.runtime_index is not None
    }

    wave_count = target_consumed_batches_per_rank - origin_consumed_batches_per_rank
    row_count = wave_count * world_size
    global_start = origin_consumed_batches_per_rank * world_size
    schedule = np.memmap(output_path, mode="w+", dtype="<u4", shape=(row_count,))
    written = 0
    soft_draw_counts = {name: 0 for name in soft_samplers}
    try:
        while written < row_count:
            absolute_position = global_start + written
            epoch, epoch_offset = divmod(absolute_position, epoch_size)
            hierarchical.set_epoch(epoch)
            legacy_epoch = np.asarray(
                hierarchical._global_indices(), dtype=np.int64
            )
            take = min(row_count - written, epoch_size - epoch_offset)
            rows = legacy_epoch[epoch_offset : epoch_offset + take].copy()
            child_ordinals = np.searchsorted(offsets[1:], rows, side="right")
            for child_ordinal, catalog in enumerate(catalogs):
                soft_sampler = soft_samplers.get(catalog.name)
                if soft_sampler is None:
                    continue
                positions = np.flatnonzero(child_ordinals == child_ordinal)
                if positions.size == 0:
                    continue
                runtime_index = catalog.runtime_index
                assert runtime_index is not None
                draws = soft_sampler.draw_many(int(positions.size))
                first_rows, cluster_sizes = runtime_arrays[catalog.name]
                local_rows = _rows_for_soft_draws(
                    runtime_index=runtime_index,
                    first_rows=first_rows,
                    cluster_sizes=cluster_sizes,
                    draws=draws,
                    member_seed=member_seeds[catalog.name],
                )
                rows[positions] = int(offsets[child_ordinal]) + local_rows
                soft_draw_counts[catalog.name] += int(positions.size)
            if bool(np.any(rows < 0)) or bool(np.any(rows >= int(offsets[-1]))):
                raise RuntimeError("generated schedule contains invalid flattened row")
            schedule[written : written + take] = rows.astype(np.uint32)
            written += take
        schedule.flush()
        descriptor = os.open(output_path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        del schedule
    if output_path.stat().st_size != row_count * np.dtype("<u4").itemsize:
        raise RuntimeError("global schedule file size mismatch")
    return soft_draw_counts


def _exposure_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    return {
        "cluster_count": int(values.size),
        "total_exposure": int(values.sum(dtype=np.uint64)),
        "seen_clusters": int(np.count_nonzero(values)),
        "maximum_exposure": int(values.max()) if values.size else 0,
    }


def _write_u32(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.unsignedinteger):
        raise TypeError("exposure artifact must be a one-dimensional uint array")
    little = np.asarray(values, dtype="<u4")
    with path.open("xb") as handle:
        little.tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a gamma-decayed clustered row schedule offline."
    )
    parser.add_argument("--fresh", action="store_true", help="Start data order and cluster exposure from zero.")
    parser.add_argument(
        "--origin-updates", type=int, default=0, help="Existing trainer update cursor for a fresh stage."
    )
    parser.add_argument("--world-size", type=int, default=128, help="World size for --fresh generation.")
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--source-checkpoint-step",
        type=int,
        default=None,
        help=(
            "Expected source checkpoint step. If --source-checkpoint is omitted, "
            "the path is inferred as source-run-dir/checkpoints/<step>.pt. This "
            "allows a future boundary such as 21399 to be declared before the "
            "file appears; publication remains blocked until the real raw/EMA "
            "files can be hashed and contract-validated."
        ),
    )
    parser.add_argument("--source-ema-checkpoint", type=Path, default=None)
    parser.add_argument("--source-run-dir", type=Path, default=None)
    parser.add_argument("--target-updates", type=int, required=True)
    parser.add_argument(
        "--target-config",
        type=Path,
        default=None,
        help=(
            "Target catalog training config. Required for --fresh; otherwise "
            "enables schema-v2 source-to-target catalog replacement."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument(
        "--history-start-update",
        type=int,
        default=0,
        help="Absolute first update governed by this config's sampler.",
    )
    parser.add_argument(
        "--initial-coverage",
        type=Path,
        action="append",
        default=[],
        help="Existing audit coverage to add before history-start-update.",
    )
    parser.add_argument(
        "--soft-dataset",
        action="append",
        default=[],
        help="Repeat exactly twice; otherwise short/long names are auto-detected.",
    )
    parser.add_argument(
        "--target-soft-dataset",
        action="append",
        default=[],
        help=(
            "Schema-v2 target clustered datasets. Defaults to the locked "
            "legacy-residual and all-sources-ge200 dataset names."
        ),
    )
    parser.add_argument(
        "--intentional-missing-global-ids",
        type=Path,
        default=None,
        help="Optional TSV(.gz) allowlist with global_id and reason columns.",
    )
    parser.add_argument(
        "--expected-ambiguity-raw-ids",
        type=int,
        default=142,
        help="Expected AFQ/AFU ambiguous raw-ID groups in the ge200 receipt.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if not args.fresh and args.source_run_dir is None:
        parser.error("--source-run-dir is required unless --fresh is used")
    return args


def _resolve_source_checkpoint_argument(
    *,
    source_run_dir: Path,
    source_checkpoint: Path | None,
    source_checkpoint_step: int | None,
) -> tuple[Path, int | None]:
    if source_checkpoint is None and source_checkpoint_step is None:
        raise ValueError(
            "provide --source-checkpoint or --source-checkpoint-step"
        )
    expected_step = (
        None if source_checkpoint_step is None else int(source_checkpoint_step)
    )
    if expected_step is not None and expected_step < 0:
        raise ValueError("source-checkpoint-step must be non-negative")
    candidate = (
        source_checkpoint
        if source_checkpoint is not None
        else source_run_dir / "checkpoints" / f"{expected_step}.pt"
    )
    candidate = candidate.expanduser().resolve(strict=False)
    if not candidate.is_file():
        if expected_step is not None:
            raise FileNotFoundError(
                "Pending source checkpoint has not appeared; no complete schedule "
                "was published. Re-run after the raw/EMA pair is present and it "
                f"will be cryptographically bound: step={expected_step}, path={candidate}"
            )
        raise FileNotFoundError(candidate)
    return candidate.resolve(strict=True), expected_step


def prepare_fresh_schedule(args: argparse.Namespace) -> Path:
    """Publish the existing schedule format with fresh data order/exposure.

    Rows are generated from sampler epoch zero but addressed from the current
    trainer cursor. No model or checkpoint file is opened or hashed.
    """
    if args.target_config is None:
        raise ValueError("--fresh requires --target-config")
    if any(getattr(args, name, None) is not None for name in (
        "source_checkpoint", "source_checkpoint_step", "source_ema_checkpoint", "source_run_dir"
    )):
        raise ValueError("--fresh does not accept source checkpoint identity")
    if getattr(args, "initial_coverage", []) or getattr(args, "history_start_update", 0):
        raise ValueError("--fresh starts exposure from zero; history inputs are not accepted")
    config_path = Path(args.target_config).expanduser().resolve(strict=True)
    config_identity = _file_identity(config_path)
    config = _load_yaml(config_path)
    data_config = config["data"]
    sampler_config = data_config["train_sampler"]
    accumulation = int(config["iters_to_accumulate"])
    epoch_size = int(data_config["epoch_size"])
    world_size = int(args.world_size)
    origin_updates = int(args.origin_updates)
    target_updates = int(args.target_updates)
    seed = _effective_sampler_seed(config)
    if accumulation <= 0 or world_size <= 0 or epoch_size <= 0 or epoch_size % world_size:
        raise ValueError("fresh schedule requires positive accumulation/world/epoch and divisible epoch size")
    if not bool(sampler_config.get("replacement", True)):
        raise ValueError("fresh schedule requires replacement=True")
    if origin_updates < 0 or target_updates <= origin_updates:
        raise ValueError("fresh target updates must exceed a non-negative origin")
    if int(config["max_steps"]) < target_updates:
        raise ValueError("target-updates exceeds target config max_steps")
    catalogs = build_catalogs(config)
    soft_catalogs = _identify_soft_catalogs(catalogs, args.soft_dataset)
    masses = [float(value) for value in sampler_config["train_sample_weights"]]
    runtime_bindings = {
        catalog.name: _runtime_binding(catalog.runtime_index)
        for catalog in soft_catalogs
    }
    dataset_binding, _ = _build_dataset_binding(
        catalogs=catalogs, masses=masses, epoch_size=epoch_size,
        world_size=world_size, seed=seed, runtime_bindings=runtime_bindings,
    )
    origin_waves = origin_updates * accumulation
    target_waves = target_updates * accumulation
    wave_count = target_waves - origin_waves
    row_count = wave_count * world_size
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    with _atomic_output_directory(output_dir) as temporary:
        artifacts = {}
        exposure = {}
        soft_samplers = {}
        member_seeds = {}
        for catalog in soft_catalogs:
            name = catalog.name
            if Path(name).name != name:
                raise ValueError(f"unsafe dataset name for artifact: {name!r}")
            initial = np.zeros(catalog.runtime_index.cluster_count, dtype=np.uint32)
            initial_path = temporary / f"{name}.initial_exposure.u32"
            _write_u32(initial_path, initial)
            artifacts[f"{name}.initial_exposure"] = _artifact_record(initial_path, relative_to=temporary)
            bucket_seed = _derive_seed(seed, name, "soft_exposure_bucket")
            member_seed = _derive_seed(seed, name, "member_rotation")
            soft_samplers[name] = SoftExposureBucketSampler(initial, gamma=float(args.gamma), seed=bucket_seed)
            member_seeds[name] = member_seed
            exposure[name] = {
                "dtype": "<u4", "initial": _exposure_stats(initial),
                "bucket_seed": bucket_seed, "member_seed": member_seed,
            }
        schedule_path = temporary / "global_rows.u32"
        draw_counts = generate_global_rows(
            output_path=schedule_path, catalogs=catalogs, masses=masses,
            soft_samplers=soft_samplers, member_seeds=member_seeds,
            origin_consumed_batches_per_rank=0,
            target_consumed_batches_per_rank=wave_count,
            accumulation=accumulation, world_size=world_size, epoch_size=epoch_size, seed=seed,
        )
        artifacts["global_rows"] = _artifact_record(schedule_path, relative_to=temporary)
        for name, sampler in soft_samplers.items():
            final = sampler.exposure_counts
            final_path = temporary / f"{name}.final_exposure.u32"
            _write_u32(final_path, final)
            artifacts[f"{name}.final_exposure"] = _artifact_record(final_path, relative_to=temporary)
            exposure[name].update({"final": _exposure_stats(final), "future_draws": draw_counts[name]})
            if int(final.sum(dtype=np.uint64)) != draw_counts[name]:
                raise RuntimeError(f"fresh exposure accounting mismatch: {name}")
        artifact_set_sha = _canonical_sha256(artifacts)
        artifacts["artifact_set_sha256"] = artifact_set_sha
        if _file_identity(config_path) != config_identity:
            raise RuntimeError("target config changed while preparing the fresh schedule")
        manifest = {
            "schema_version": SCHEMA_VERSION, "format": FORMAT, "status": "complete", "layout": LAYOUT,
            "algorithm": {
                "format": ALGORITHM_FORMAT, "gamma": float(args.gamma),
                "member_policy": MEMBER_POLICY, "consumption_order": CONSUMPTION_ORDER,
                "soft_cluster_datasets": [catalog.name for catalog in soft_catalogs],
            },
            "shape": {"dtype": "<u4", "row_count": row_count, "wave_count": wave_count},
            "dataset": dataset_binding,
            "sampler": {
                "seed": seed, "epoch_size": epoch_size, "world_size": world_size,
                "origin_consumed_global_batches": origin_waves,
                "target_consumed_global_batches": target_waves, "target_total_rows": row_count,
            },
            "source_checkpoint": {
                "mode": "fresh", "raw_checkpoint_path": None, "raw_checkpoint_sha256": None,
                "ema_checkpoint_path": None, "ema_checkpoint_sha256": None,
                "checkpoint_step": origin_updates - 1, "completed_updates": origin_updates,
                "next_global_step": origin_waves, "origin_consumed_global_batches_per_rank": origin_waves,
                "iters_to_accumulate": accumulation, "checkpoint_pair_id": None,
                "source_training_contract_id": None, "source_data_contract_sha256": None,
                "source_run_id": None, "model_architecture_sha256": None,
            },
            "provenance": {
                "generated_at": datetime.now(timezone.utc).isoformat(), "target_config_path": str(config_path),
                "data_sequence_start": "epoch_zero", "initial_exposure": "zero", "exposure": exposure,
            },
            "artifacts": artifacts,
        }
        _write_json_fsync(temporary / "manifest.json", manifest)
        _write_json_fsync(temporary / "_SUCCESS.json", {
            "schema_version": 1, "status": "complete",
            "manifest_sha256": _sha256_stable_file(temporary / "manifest.json"),
            "schedule_path": "global_rows.u32", "schedule_size_bytes": artifacts["global_rows"]["size_bytes"],
            "schedule_sha256": artifacts["global_rows"]["sha256"], "artifact_set_sha256": artifact_set_sha,
        })
    LOGGER.info("published fresh soft cluster schedule: %s (%d rows)", output_dir, row_count)
    return output_dir / "manifest.json"


def prepare_schedule(args: argparse.Namespace) -> Path:
    if getattr(args, "fresh", False):
        return prepare_fresh_schedule(args)
    started = time.perf_counter()
    source_run_dir = args.source_run_dir.expanduser().resolve(strict=True)
    if not source_run_dir.is_dir():
        raise NotADirectoryError(source_run_dir)
    config_path = source_run_dir / "config.yaml"
    run_manifest_path = source_run_dir / "manifest" / "run_manifest.json"
    if not config_path.is_file() or not run_manifest_path.is_file():
        raise FileNotFoundError(
            f"source run requires config.yaml and manifest/run_manifest.json: "
            f"{source_run_dir}"
        )
    config_identity = _file_identity(config_path)
    run_manifest_identity = _file_identity(run_manifest_path)
    config_sha256 = _sha256_stable_file(config_path)
    run_manifest_sha256 = _sha256_stable_file(run_manifest_path)
    raw_path, expected_checkpoint_step = _resolve_source_checkpoint_argument(
        source_run_dir=source_run_dir,
        source_checkpoint=getattr(args, "source_checkpoint", None),
        source_checkpoint_step=getattr(args, "source_checkpoint_step", None),
    )
    if not raw_path.is_file() or not raw_path.is_relative_to(source_run_dir):
        raise ValueError("source raw checkpoint must be a file inside source-run-dir")
    config = _load_yaml(config_path)
    run_manifest = _load_json(run_manifest_path)
    world_size = _world_size(source_run_dir, None)
    accumulation = int(config.get("iters_to_accumulate", 1))
    epoch_size = int(config["data"]["epoch_size"])
    seed = _effective_sampler_seed(config)
    replacement = bool(config["data"]["train_sampler"].get("replacement", True))
    masses = [
        float(value)
        for value in config["data"]["train_sampler"]["train_sample_weights"]
    ]
    if accumulation <= 0 or epoch_size <= 0 or not replacement:
        raise ValueError("source run has unsupported accumulation/epoch/replacement")

    raw_identity = _file_identity(raw_path)
    raw_sha256 = _sha256_stable_file(raw_path)
    raw_checkpoint = _load_checkpoint_metadata(raw_path)
    ema_path = _infer_ema_checkpoint(
        raw_path=raw_path,
        raw_checkpoint=raw_checkpoint,
        override=args.source_ema_checkpoint,
    )
    if not ema_path.is_relative_to(source_run_dir):
        raise ValueError("source EMA checkpoint must be inside source-run-dir")
    ema_identity = _file_identity(ema_path)
    ema_sha256 = _sha256_stable_file(ema_path)
    ema_checkpoint = _load_checkpoint_metadata(ema_path)
    completed_updates, contract = _validate_checkpoint_pair_headers(
        raw_checkpoint=raw_checkpoint,
        ema_checkpoint=ema_checkpoint,
        run_manifest=run_manifest,
        config=config,
        world_size=world_size,
    )
    if (
        expected_checkpoint_step is not None
        and completed_updates - 1 != expected_checkpoint_step
    ):
        raise ValueError(
            "source checkpoint step differs from the declared placeholder: "
            f"declared={expected_checkpoint_step}, observed={completed_updates - 1}"
        )
    target_updates = int(args.target_updates)
    if target_updates <= completed_updates:
        raise ValueError(
            f"target-updates must exceed source completed updates "
            f"{completed_updates}, got {target_updates}"
        )
    max_steps = config.get("max_steps")
    if max_steps is not None and int(max_steps) > 0 and target_updates > int(max_steps):
        raise ValueError(
            f"target-updates {target_updates} exceeds source config max_steps "
            f"{max_steps}"
        )
    history_start = int(args.history_start_update)
    if not 0 <= history_start <= completed_updates:
        raise ValueError("history-start-update must be inside checkpoint history")
    _validate_initial_coverage_boundary(
        coverage_dirs=args.initial_coverage,
        history_start_update=history_start,
        accumulation=accumulation,
        world_size=world_size,
    )
    coverage_input_snapshots = _snapshot_initial_coverage_inputs(
        args.initial_coverage
    )

    LOGGER.info("building source run catalogs")
    source_catalogs = build_catalogs(config)
    if len(source_catalogs) != len(masses):
        raise ValueError("catalog count differs from mixture mass count")
    source_soft_catalogs = _identify_soft_catalogs(
        source_catalogs, getattr(args, "soft_dataset", [])
    )
    history_counts, replay = _replay_history_segment(
        catalogs=source_catalogs,
        masses=masses,
        start_update=history_start,
        completed_updates=completed_updates,
        accumulation=accumulation,
        world_size=world_size,
        epoch_size=epoch_size,
        seed=seed,
        replacement=replacement,
    )
    merged_coverages = merge_initial_coverages(
        catalogs=source_catalogs,
        counts=history_counts,
        coverage_dirs=args.initial_coverage,
    )
    if history_start > 0:
        expected_initial_occurrences = history_start * accumulation * world_size
        observed_initial_occurrences = sum(
            int(record["sampled_occurrences_added"])
            for record in merged_coverages
        )
        if observed_initial_occurrences != expected_initial_occurrences:
            raise ValueError(
                "initial coverage sparse row counts do not equal the declared "
                f"complete prefix: observed={observed_initial_occurrences}, "
                f"expected={expected_initial_occurrences}"
            )
    _bind_initial_coverage_artifacts(merged_coverages)
    end_prefix = replay["end_prefix"]
    rank_cursors, rank_cursor_sha256 = _rank_cursor_binding(
        raw_checkpoint=raw_checkpoint,
        expected_signature_rank0=end_prefix["sampler_signature_rank0"],
        completed_updates=completed_updates,
        accumulation=accumulation,
        world_size=world_size,
        epoch_size=epoch_size,
    )

    initial_exposure: dict[str, np.ndarray] = {}
    source_runtime_bindings: dict[str, Any] = {}
    runtime_input_snapshots: dict[Path, dict[str, Any]] = {}
    for catalog in source_soft_catalogs:
        runtime_index = catalog.runtime_index
        assert runtime_index is not None
        runtime_binding = _runtime_binding(runtime_index)
        source_runtime_bindings[catalog.name] = runtime_binding
        manifest_path_text = runtime_binding.get("manifest_path")
        if manifest_path_text is not None:
            manifest_path = Path(manifest_path_text).resolve(strict=True)
            runtime_input_snapshots[manifest_path] = {
                "identity": _file_identity(manifest_path),
                "sha256": runtime_binding["manifest_sha256"],
            }
        for artifact in runtime_binding.get("sampling_artifacts", {}).values():
            artifact_path = Path(artifact["path"]).resolve(strict=True)
            runtime_input_snapshots[artifact_path] = {
                "identity": _file_identity(artifact_path),
                "sha256": artifact["sha256"],
            }
    source_dataset_binding, _ = _build_dataset_binding(
        catalogs=source_catalogs,
        masses=masses,
        epoch_size=epoch_size,
        world_size=world_size,
        seed=seed,
        runtime_bindings=source_runtime_bindings,
    )

    target_config_path_raw = getattr(args, "target_config", None)
    schema_v2 = target_config_path_raw is not None
    catalog_migration: dict[str, Any] | None = None
    target_config_path: Path | None = None
    target_config_identity: tuple[int, int, int, int] | None = None
    target_config_sha256: str | None = None
    if not schema_v2:
        catalogs = source_catalogs
        soft_catalogs = source_soft_catalogs
        schedule_masses = masses
        runtime_bindings = source_runtime_bindings
        dataset_binding = source_dataset_binding
        for catalog in soft_catalogs:
            runtime_index = catalog.runtime_index
            assert runtime_index is not None
            child_ordinal = source_catalogs.index(catalog)
            initial_exposure[catalog.name] = row_counts_to_cluster_exposure(
                runtime_index, history_counts[child_ordinal]
            )
    else:
        target_config_path = Path(target_config_path_raw).expanduser().resolve(
            strict=True
        )
        target_config_identity = _file_identity(target_config_path)
        target_config_sha256 = _sha256_stable_file(target_config_path)
        target_config = _load_yaml(target_config_path)
        target_accumulation = int(target_config.get("iters_to_accumulate", 1))
        target_epoch_size = int(target_config["data"]["epoch_size"])
        target_seed = _effective_sampler_seed(target_config)
        target_replacement = bool(
            target_config["data"]["train_sampler"].get("replacement", True)
        )
        if (
            target_accumulation != accumulation
            or target_epoch_size != epoch_size
            or target_seed != seed
            or not target_replacement
        ):
            raise ValueError(
                "Target catalog migration must preserve accumulation, epoch_size, "
                "sampler seed, and replacement=True"
            )
        target_max_steps = target_config.get("max_steps")
        if (
            target_max_steps is not None
            and int(target_max_steps) > 0
            and target_updates > int(target_max_steps)
        ):
            raise ValueError("target-updates exceeds target config max_steps")
        schedule_masses = [
            float(value)
            for value in target_config["data"]["train_sampler"][
                "train_sample_weights"
            ]
        ]
        if schedule_masses != masses:
            raise ValueError(
                "Catalog replacement must preserve the source dataset masses"
            )
        LOGGER.info("building target replacement catalogs")
        catalogs = build_catalogs(target_config)
        if len(catalogs) != len(schedule_masses):
            raise ValueError("target catalog count differs from mixture masses")
        requested_target_names = list(
            getattr(args, "target_soft_dataset", [])
        ) or [TARGET_RESIDUAL_DATASET, TARGET_GE200_DATASET]
        if requested_target_names != [TARGET_RESIDUAL_DATASET, TARGET_GE200_DATASET]:
            raise ValueError(
                "schema-v2 target datasets must be the locked residual/ge200 pair"
            )
        soft_catalogs = _identify_soft_catalogs(catalogs, requested_target_names)
        source_names = [catalog.name for catalog in source_catalogs]
        target_names = [catalog.name for catalog in catalogs]
        source_soft_names = [catalog.name for catalog in source_soft_catalogs]
        expected_target_names = [
            TARGET_RESIDUAL_DATASET if name == source_soft_names[0]
            else TARGET_GE200_DATASET if name == source_soft_names[1]
            else name
            for name in source_names
        ]
        if target_names != expected_target_names:
            raise ValueError(
                "Target train_sets must replace legacy Short/Long in place and "
                "preserve all other children"
            )

        target_runtime_bindings: dict[str, Any] = {}
        lookup_bindings: dict[str, Any] = {}
        for catalog in soft_catalogs:
            runtime_index = catalog.runtime_index
            assert runtime_index is not None
            target_runtime_bindings[catalog.name] = _runtime_binding(runtime_index)
            lookup_bindings[catalog.name] = _target_lookup_binding(
                dataset_name=catalog.name,
                runtime_index=runtime_index,
            )
            target_binding = target_runtime_bindings[catalog.name]
            target_manifest_path = Path(
                target_binding["manifest_path"]
            ).resolve(strict=True)
            runtime_input_snapshots[target_manifest_path] = {
                "identity": _file_identity(target_manifest_path),
                "sha256": target_binding["manifest_sha256"],
            }
            for artifact in target_binding.get("sampling_artifacts", {}).values():
                artifact_path = Path(artifact["path"]).resolve(strict=True)
                runtime_input_snapshots[artifact_path] = {
                    "identity": _file_identity(artifact_path),
                    "sha256": artifact["sha256"],
                }
            lookup_binding = lookup_bindings[catalog.name]
            external_records = [lookup_binding["member_to_cluster"]]
            if lookup_binding["ambiguity_alternates"] is not None:
                external_records.append(lookup_binding["ambiguity_alternates"])
            for artifact in external_records:
                artifact_path = Path(artifact["path"]).resolve(strict=True)
                runtime_input_snapshots[artifact_path] = {
                    "identity": _file_identity(artifact_path),
                    "sha256": artifact["sha256"],
                }
            receipt_path = Path(lookup_binding["pair_receipt_path"]).resolve(
                strict=True
            )
            runtime_input_snapshots[receipt_path] = {
                "identity": _file_identity(receipt_path),
                "sha256": lookup_binding["pair_receipt_sha256"],
            }
        _validate_target_lookup_pair(lookup_bindings)
        runtime_bindings = target_runtime_bindings
        dataset_binding, _ = _build_dataset_binding(
            catalogs=catalogs,
            masses=schedule_masses,
            epoch_size=epoch_size,
            world_size=world_size,
            seed=seed,
            runtime_bindings=runtime_bindings,
        )

        source_occurrences, source_occurrence_audit = (
            collect_seen_source_global_occurrences(
                catalogs=source_catalogs,
                history_counts=history_counts,
                source_soft_datasets=source_soft_names,
            )
        )
        target_assignments = _scan_target_member_lookups(
            source_global_ids=set(source_occurrences),
            target_catalogs=soft_catalogs,
            lookup_bindings=lookup_bindings,
        )
        intentional_missing, intentional_missing_binding = (
            _load_intentional_missing(
                getattr(args, "intentional_missing_global_ids", None)
            )
        )
        if intentional_missing_binding is not None:
            missing_path = Path(intentional_missing_binding["path"])
            runtime_input_snapshots[missing_path] = {
                "identity": _file_identity(missing_path),
                "sha256": intentional_missing_binding["sha256"],
            }
        ambiguity_records, ambiguity_audit = _scan_ambiguity_alternates(
            ge200_dataset_name=TARGET_GE200_DATASET,
            binding=lookup_bindings[TARGET_GE200_DATASET],
            cluster_count=(
                next(
                    catalog.runtime_index.cluster_count
                    for catalog in soft_catalogs
                    if catalog.name == TARGET_GE200_DATASET
                    and catalog.runtime_index is not None
                )
            ),
            target_assignments=target_assignments,
        )
        expected_ambiguity = int(
            getattr(args, "expected_ambiguity_raw_ids", 142)
        )
        if ambiguity_audit["ambiguous_raw_ids"] != expected_ambiguity:
            raise ValueError(
                "ge200 ambiguity raw-ID count differs from the declared contract: "
                f"{ambiguity_audit['ambiguous_raw_ids']} != {expected_ambiguity}"
            )
        initial_exposure, projection_audit = project_global_id_occurrences(
            source_occurrences=source_occurrences,
            target_assignments=target_assignments,
            target_cluster_counts={
                catalog.name: catalog.runtime_index.cluster_count
                for catalog in soft_catalogs
                if catalog.runtime_index is not None
            },
            intentional_missing=intentional_missing,
            ambiguity_alternates=ambiguity_records,
        )
        projection_audit["source"] = source_occurrence_audit
        projection_audit["ambiguity_input"] = ambiguity_audit
        catalog_migration = {
            "format": CATALOG_MIGRATION_FORMAT,
            "allowed_delta": CATALOG_MIGRATION_ALLOWED_DELTA,
            "source_dataset": {
                "binding": source_dataset_binding,
                "binding_sha256": _canonical_sha256(source_dataset_binding),
            },
            "target_dataset": {
                "binding": dataset_binding,
                "binding_sha256": _canonical_sha256(dataset_binding),
            },
            "crosswalk": {
                "format": CROSSWALK_FORMAT,
                "global_id_policy": GLOBAL_ID_POLICY,
                "exposure_policy": EXPOSURE_POLICY,
                "ambiguous_alternate_policy": AMBIGUOUS_ALTERNATE_POLICY,
                "source_soft_datasets": source_soft_names,
                "target_soft_datasets": [catalog.name for catalog in soft_catalogs],
                "lookup_bindings": lookup_bindings,
                "intentional_missing_binding": intentional_missing_binding,
                "audit_artifact": "catalog_migration.audit",
                "accounting": projection_audit,
            },
        }
        del source_occurrences, target_assignments
    del history_counts
    gc.collect()

    output_dir = args.output_dir.expanduser().resolve(strict=False)
    with _atomic_output_directory(output_dir) as temporary:
        artifacts: dict[str, Any] = {}
        exposure_manifest: dict[str, Any] = {}
        soft_samplers: dict[str, SoftExposureBucketSampler] = {}
        member_seeds: dict[str, int] = {}
        for catalog in soft_catalogs:
            name = catalog.name
            if Path(name).name != name:
                raise ValueError(f"unsafe dataset name for artifact: {name!r}")
            initial_path = temporary / f"{name}.initial_exposure.u32"
            _write_u32(initial_path, initial_exposure[name])
            artifacts[f"{name}.initial_exposure"] = _artifact_record(
                initial_path, relative_to=temporary
            )
            bucket_seed = _derive_seed(seed, name, "soft_exposure_bucket")
            member_seed = _derive_seed(seed, name, "member_rotation")
            soft_samplers[name] = SoftExposureBucketSampler(
                initial_exposure[name], gamma=float(args.gamma), seed=bucket_seed
            )
            member_seeds[name] = member_seed
            exposure_manifest[name] = {
                "dtype": "<u4",
                "initial": _exposure_stats(initial_exposure[name]),
                "bucket_seed": bucket_seed,
                "member_seed": member_seed,
                "initial_artifact": f"{name}.initial_exposure",
            }

        origin_waves = completed_updates * accumulation
        target_waves = target_updates * accumulation
        schedule_path = temporary / "global_rows.u32"
        soft_draw_counts = generate_global_rows(
            output_path=schedule_path,
            catalogs=catalogs,
            masses=schedule_masses,
            soft_samplers=soft_samplers,
            member_seeds=member_seeds,
            origin_consumed_batches_per_rank=origin_waves,
            target_consumed_batches_per_rank=target_waves,
            accumulation=accumulation,
            world_size=world_size,
            epoch_size=epoch_size,
            seed=seed,
        )
        artifacts["global_rows"] = _artifact_record(
            schedule_path, relative_to=temporary
        )

        for catalog in soft_catalogs:
            name = catalog.name
            final_exposure = soft_samplers[name].exposure_counts
            final_path = temporary / f"{name}.final_exposure.u32"
            _write_u32(final_path, final_exposure)
            artifacts[f"{name}.final_exposure"] = _artifact_record(
                final_path, relative_to=temporary
            )
            exposure_manifest[name].update(
                {
                    "final": _exposure_stats(final_exposure),
                    "future_draws": soft_draw_counts[name],
                    "final_artifact": f"{name}.final_exposure",
                }
            )
            if (
                exposure_manifest[name]["final"]["total_exposure"]
                - exposure_manifest[name]["initial"]["total_exposure"]
                != soft_draw_counts[name]
            ):
                raise RuntimeError(f"future exposure accounting mismatch for {name}")

        if catalog_migration is not None:
            audit_path = temporary / "catalog_migration.audit.json"
            _write_json_fsync(
                audit_path,
                catalog_migration["crosswalk"]["accounting"],
            )
            artifacts["catalog_migration.audit"] = _artifact_record(
                audit_path,
                relative_to=temporary,
            )

        artifact_set_sha256 = _canonical_sha256(artifacts)
        artifacts["artifact_set_sha256"] = artifact_set_sha256
        hierarchical = _hierarchical_sampler(
            catalogs=catalogs,
            masses=schedule_masses,
            epoch_size=epoch_size,
            world_size=world_size,
            seed=seed,
        )
        assert hierarchical is not None
        signature = hierarchical.resume_signature()
        row_count = (target_waves - origin_waves) * world_size
        source_checkpoint = {
            "mode": "checkpoint_migration",
            "raw_checkpoint_path": str(raw_path),
            "raw_checkpoint_sha256": raw_sha256,
            "ema_checkpoint_path": str(ema_path),
            "ema_checkpoint_sha256": ema_sha256,
            "checkpoint_step": completed_updates - 1,
            "completed_updates": completed_updates,
            "next_global_step": origin_waves,
            "origin_consumed_global_batches_per_rank": origin_waves,
            "iters_to_accumulate": accumulation,
            "checkpoint_pair_id": contract["checkpoint_pair_id"],
            "source_training_contract_id": contract["training_contract_id"],
            "source_data_contract_sha256": contract["data_contract_sha256"],
            "source_run_id": contract["run_id"],
            "model_architecture_sha256": contract["model_architecture_sha256"],
        }
        source_run = {
            "run_dir": str(source_run_dir),
            "run_name": run_manifest["run_name"],
            "config_path": str(config_path.resolve(strict=True)),
            "config_sha256": config_sha256,
            "run_manifest_path": str(run_manifest_path.resolve(strict=True)),
            "run_manifest_sha256": run_manifest_sha256,
            "history_start_update": history_start,
            "initial_coverages": merged_coverages,
            "rank_cursors": rank_cursors,
            "rank_cursor_sha256": rank_cursor_sha256,
        }
        if target_config_path is not None:
            source_run["target_config_path"] = str(target_config_path)
            source_run["target_config_sha256"] = target_config_sha256
        immutable_inputs = {
            config_path: config_identity,
            run_manifest_path: run_manifest_identity,
            raw_path: raw_identity,
            ema_path: ema_identity,
        }
        if target_config_path is not None and target_config_identity is not None:
            immutable_inputs[target_config_path] = target_config_identity
        changed_inputs = [
            str(path)
            for path, identity in immutable_inputs.items()
            if _file_identity(path) != identity
        ]
        if changed_inputs:
            raise RuntimeError(
                "Source inputs changed while preparing the schedule: "
                + ", ".join(changed_inputs)
            )
        _verify_hashed_input_snapshots(coverage_input_snapshots)
        _verify_hashed_input_snapshots(runtime_input_snapshots)
        manifest = {
            "schema_version": SCHEMA_VERSION_V2 if schema_v2 else SCHEMA_VERSION,
            "format": FORMAT_V2 if schema_v2 else FORMAT,
            "status": "complete",
            "layout": LAYOUT,
            "algorithm": {
                "format": ALGORITHM_FORMAT,
                "gamma": float(args.gamma),
                "member_policy": MEMBER_POLICY,
                "consumption_order": CONSUMPTION_ORDER,
                "soft_cluster_datasets": [catalog.name for catalog in soft_catalogs],
            },
            "shape": {
                "dtype": "<u4",
                "row_count": row_count,
                "wave_count": target_waves - origin_waves,
            },
            "dataset": dataset_binding,
            "sampler": {
                "seed": seed,
                "epoch_size": epoch_size,
                "world_size": world_size,
                "origin_consumed_global_batches": origin_waves,
                "target_consumed_global_batches": target_waves,
                "target_total_rows": row_count,
            },
            "source_checkpoint": source_checkpoint,
            "source_run": source_run,
            "provenance": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "history_replay": replay,
                "exposure": exposure_manifest,
                "epoch_size_divisible_by_world_size": True,
            },
            "artifacts": artifacts,
        }
        if catalog_migration is not None:
            manifest["catalog_migration"] = catalog_migration
        manifest_path = temporary / "manifest.json"
        _write_json_fsync(manifest_path, manifest)
        manifest_sha256 = _sha256_stable_file(manifest_path)
        success = {
            "schema_version": 1,
            "status": "complete",
            "manifest_sha256": manifest_sha256,
            "schedule_path": "global_rows.u32",
            "schedule_size_bytes": artifacts["global_rows"]["size_bytes"],
            "schedule_sha256": artifacts["global_rows"]["sha256"],
            "artifact_set_sha256": artifact_set_sha256,
        }
        _write_json_fsync(temporary / "_SUCCESS.json", success)

    LOGGER.info(
        "published soft cluster schedule in %.2fs: %s",
        time.perf_counter() - started,
        output_dir,
    )
    return output_dir / "manifest.json"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    manifest = prepare_schedule(args)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
