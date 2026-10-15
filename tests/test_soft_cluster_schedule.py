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

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from ml_collections import ConfigDict

from protenix.data.pipeline.sample_context import SampleOccurrence
from protenix.data.pipeline.dataloader import (
    DistributedDataLoader,
    DistributedHierarchicalWeightedSampler,
    get_dataloaders,
)
from protenix.data.pipeline.soft_cluster_schedule import (
    DistributedPrecomputedScheduleSampler,
    SOFT_CLUSTER_ALGORITHM_FORMAT,
    SOFT_CLUSTER_CONSUMPTION_ORDER,
    SOFT_CLUSTER_MEMBER_POLICY,
    SOFT_CLUSTER_SCHEDULE_DTYPE,
    SOFT_CLUSTER_SCHEDULE_FORMAT,
    SOFT_CLUSTER_SCHEDULE_FORMAT_V2,
    SOFT_CLUSTER_SCHEDULE_LAYOUT,
    SoftClusterScheduleError,
    build_soft_cluster_schedule_contract,
    clear_soft_cluster_schedule_cache,
    consumed_batches_from_sampler_cursor,
    load_soft_cluster_schedule,
    load_soft_cluster_schedule_manifest,
    rank_local_epoch_size,
    sampler_cursor_from_consumed_batches,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


class _SizedDataset:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return int(getattr(index, "index", index))


class _ClusterSource:
    def __init__(self) -> None:
        self.first_row_ids = np.asarray([0, 2], dtype=np.uint64)
        self.cluster_sizes = np.asarray([2, 3], dtype=np.uint32)
        self.cluster_count = 2


class _HierarchicalSizedDataset(_SizedDataset):
    def __init__(self) -> None:
        super().__init__(20)
        self.hierarchical_dataset_offsets = (0, 5, 10, 15, 20)
        self.hierarchical_dataset_masses = [0.49, 0.005, 0.505, 0.02]
        self.hierarchical_child_sampling_specs = [
            {"kind": "weighted", "weights": torch.ones(5)},
            {"kind": "clustered", "source": _ClusterSource()},
            {"kind": "clustered", "source": _ClusterSource()},
            {"kind": "weighted", "weights": torch.ones(5)},
        ]


def _build_schedule(
    root: Path,
    *,
    world_size: int = 3,
    epoch_size: int = 8,
    origin: int = 4,
    target: int = 10,
    flat_row_count: int = 20,
    gamma: float = 0.2,
) -> tuple[Path, dict, np.ndarray]:
    root.mkdir(parents=True, exist_ok=True)
    wave_count = target - origin
    rows = (
        np.arange(wave_count * world_size, dtype=np.uint32) % flat_row_count
    ).astype(SOFT_CLUSTER_SCHEDULE_DTYPE)
    rows_path = root / "global_rows.u32"
    rows.tofile(rows_path)

    child_sizes = [5, 5, 5, 5]
    child_hashes = [
        hashlib.sha256(f"child-{index}".encode()).hexdigest()
        for index in range(4)
    ]
    source = {
        "mode": "checkpoint_migration",
        "raw_checkpoint_path": os.path.realpath(root / "raw.pt"),
        "raw_checkpoint_sha256": "1" * 64,
        "ema_checkpoint_path": os.path.realpath(root / "ema.pt"),
        "ema_checkpoint_sha256": "2" * 64,
        "checkpoint_step": 0,
        "completed_updates": 1,
        "next_global_step": origin,
        "origin_consumed_global_batches_per_rank": origin,
        "iters_to_accumulate": origin,
        "checkpoint_pair_id": "3" * 64,
        "source_training_contract_id": "source-contract-v1",
        "source_data_contract_sha256": "4" * 64,
        "source_run_id": "source-run-v1",
        "model_architecture_sha256": "5" * 64,
    }
    manifest = {
        "schema_version": 1,
        "format": SOFT_CLUSTER_SCHEDULE_FORMAT,
        "status": "complete",
        "layout": SOFT_CLUSTER_SCHEDULE_LAYOUT,
        "algorithm": {
            "format": SOFT_CLUSTER_ALGORITHM_FORMAT,
            "gamma": gamma,
            "member_policy": SOFT_CLUSTER_MEMBER_POLICY,
            "consumption_order": SOFT_CLUSTER_CONSUMPTION_ORDER,
            "soft_cluster_datasets": ["short", "long"],
        },
        "shape": {
            "dtype": SOFT_CLUSTER_SCHEDULE_DTYPE,
            "row_count": int(rows.size),
            "wave_count": wave_count,
        },
        "dataset": {
            "flat_row_count": flat_row_count,
            "train_sets": ["pdb", "short", "long", "disorder"],
            "dataset_offsets": [0, 5, 10, 15, 20],
            "dataset_masses": [0.49, 0.005, 0.505, 0.02],
            "child_fingerprints": [
                {
                    "kind": "clustered" if index in {1, 2} else "weighted",
                    "size": child_sizes[index],
                    "sha256": child_hashes[index],
                }
                for index in range(4)
            ],
        },
        "sampler": {
            "seed": 42,
            "epoch_size": epoch_size,
            "world_size": world_size,
            "origin_consumed_global_batches": origin,
            "target_consumed_global_batches": target,
            "target_total_rows": int(rows.size),
        },
        "source_checkpoint": source,
        "artifacts": {
            "global_rows": {
                "path": "global_rows.u32",
                "size_bytes": rows_path.stat().st_size,
                "sha256": _sha256(rows_path),
            }
        },
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    success = {
        "schema_version": 1,
        "status": "complete",
        "manifest_sha256": _sha256(manifest_path),
        "schedule_path": "global_rows.u32",
        "schedule_size_bytes": rows_path.stat().st_size,
        "schedule_sha256": _sha256(rows_path),
    }
    _write_json(root / "_SUCCESS.json", success)
    clear_soft_cluster_schedule_cache()
    return manifest_path, manifest, rows


def _upgrade_schedule_to_catalog_migration_v2(
    manifest_path: Path,
    manifest: dict,
) -> dict:
    root = manifest_path.parent
    target_binding = manifest["dataset"]
    source_binding = json.loads(json.dumps(target_binding))
    source_binding["train_sets"] = [
        "pdb",
        "legacy_short",
        "legacy_long",
        "disorder",
    ]
    for name, values in {
        "short": np.asarray([3, 0], dtype="<u4"),
        "long": np.asarray([3, 1], dtype="<u4"),
    }.items():
        path = root / f"{name}.initial_exposure.u32"
        values.tofile(path)
        manifest["artifacts"][f"{name}.initial_exposure"] = {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    accounting = {
        "schema_version": 1,
        "event": "source_target_catalog_history_projection",
        "global_id_policy": "source_prefix_pipe_raw_id_v1",
        "exposure_policy": "occurrence_sum_v1",
        "ambiguous_alternate_policy": "seen_floor_1_v1",
        "source_unique_seen_members": 3,
        "source_history_occurrences": 7,
        "mapped_unique_members": 2,
        "mapped_occurrences": 6,
        "intentional_missing_unique_members": 1,
        "intentional_missing_occurrences": 1,
        "intentional_missing_by_reason": {
            "fixture": {"unique_source_rows": 1, "historical_occurrences": 1}
        },
        "unresolved_unique_members": 0,
        "unresolved_occurrences": 0,
        "ambiguous_alternate_records": 2,
        "ambiguous_seen_records": 1,
        "ambiguous_alternate_clusters": 1,
        "ambiguous_floor_increment": 1,
        "target_projected_occurrences": 6,
        "target_initial_exposure_total": 7,
        "target_exposure": {
            "short": {
                "cluster_count": 2,
                "total_exposure": 3,
                "seen_clusters": 1,
                "maximum_exposure": 3,
            },
            "long": {
                "cluster_count": 2,
                "total_exposure": 4,
                "seen_clusters": 2,
                "maximum_exposure": 3,
            },
        },
    }
    audit_path = root / "catalog_migration.audit.json"
    _write_json(audit_path, accounting)
    manifest["artifacts"]["catalog_migration.audit"] = {
        "path": audit_path.name,
        "size_bytes": audit_path.stat().st_size,
        "sha256": _sha256(audit_path),
    }

    pair_id = "a" * 64
    pair_receipt_sha = "b" * 64

    def tsv_record(name: str, columns: list[str], count: int) -> dict:
        return {
            "path": os.path.realpath(root / name),
            "size_bytes": 123,
            "sha256": "c" * 64,
            "count": count,
            "dtype": "tsv_gzip",
            "columns": columns,
        }

    member_columns = [
        "runtime_row_id",
        "global_id",
        "cluster_ordinal",
        "stable_cluster_id",
        "source_member_row_id",
        "source_partition",
        "cluster_representative_length_bucket",
    ]
    ambiguity_columns = [
        "global_id",
        "canonical_cluster_ordinal",
        "alternate_cluster_ordinal",
        "raw_id",
        "resolution_rule",
    ]

    def lookup(role: str, name: str) -> dict:
        return {
            "role": role,
            "runtime_manifest_path": os.path.realpath(root / f"{name}.manifest.json"),
            "runtime_manifest_sha256": "d" * 64,
            "pair_id": pair_id,
            "pair_receipt_path": os.path.realpath(root / "target_catalog_pair.json"),
            "pair_receipt_size_bytes": 100,
            "pair_receipt_sha256": pair_receipt_sha,
            "member_to_cluster": tsv_record(
                f"{name}.member_to_cluster.tsv.gz", member_columns, 5
            ),
            "ambiguity_alternates": (
                tsv_record("ambiguity.tsv.gz", ambiguity_columns, 2)
                if role == "ge200_structural"
                else None
            ),
        }

    def canonical_sha(value: dict) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    manifest["schema_version"] = 2
    manifest["format"] = SOFT_CLUSTER_SCHEDULE_FORMAT_V2
    manifest["catalog_migration"] = {
        "format": "protenix.catalog_replacement.v1",
        "allowed_delta": "legacy_short_long_to_legacy_residual_ge200_v1",
        "source_dataset": {
            "binding": source_binding,
            "binding_sha256": canonical_sha(source_binding),
        },
        "target_dataset": {
            "binding": target_binding,
            "binding_sha256": canonical_sha(target_binding),
        },
        "crosswalk": {
            "format": "protenix.global_id_history_projection.v1",
            "global_id_policy": "source_prefix_pipe_raw_id_v1",
            "exposure_policy": "occurrence_sum_v1",
            "ambiguous_alternate_policy": "seen_floor_1_v1",
            "source_soft_datasets": ["legacy_short", "legacy_long"],
            "target_soft_datasets": ["short", "long"],
            "lookup_bindings": {
                "short": lookup("legacy_residual", "short"),
                "long": lookup("ge200_structural", "long"),
            },
            "intentional_missing_binding": None,
            "audit_artifact": "catalog_migration.audit",
            "accounting": accounting,
        },
    }
    _write_json(manifest_path, manifest)
    success_path = root / "_SUCCESS.json"
    success = json.loads(success_path.read_text(encoding="utf-8"))
    success["manifest_sha256"] = _sha256(manifest_path)
    _write_json(success_path, success)
    clear_soft_cluster_schedule_cache()
    return manifest


def test_manifest_and_wave_major_schedule_are_fully_validated(tmp_path: Path):
    manifest_path, manifest, rows = _build_schedule(tmp_path / "schedule")

    info = load_soft_cluster_schedule_manifest(
        manifest_path,
        expected={
            "dataset_size": 20,
            "world_size": 3,
            "epoch_size": 8,
            "gamma": 0.2,
            "seed": 42,
            "origin_consumed_global_batches": 4,
            "source_checkpoint": {
                "checkpoint_pair_id": "3" * 64,
                "checkpoint_step": 0,
            },
        },
    )
    schedule = load_soft_cluster_schedule(manifest_path, mmap=False)

    assert info.schedule_sha256 == manifest["artifacts"]["global_rows"]["sha256"]
    assert schedule.row_at(4, 0) == int(rows[0])
    assert schedule.row_at(4, 2) == int(rows[2])
    assert schedule.row_at(7, 1) == int(rows[3 * 3 + 1])
    assert schedule.rows_for_rank(5, 3, 2).tolist() == [5, 8, 11]
    with pytest.raises(SoftClusterScheduleError, match="precedes"):
        schedule.row_at(3, 0)
    with pytest.raises(SoftClusterScheduleError, match="exhausted"):
        schedule.row_at(10, 0)

    contract = build_soft_cluster_schedule_contract(info)
    assert contract["algorithm"]["gamma"] == 0.2
    assert contract["layout"] == SOFT_CLUSTER_SCHEDULE_LAYOUT
    assert contract["sampler"]["origin_consumed_global_batches_per_rank"] == 4
    assert contract["schedule"]["sha256"] == info.schedule_sha256
    assert "target_data_contract_sha256" not in json.dumps(contract)


def test_v2_catalog_migration_authenticates_both_catalogs_and_accounting(
    tmp_path: Path,
):
    manifest_path, manifest, _ = _build_schedule(tmp_path / "schedule")
    manifest = _upgrade_schedule_to_catalog_migration_v2(
        manifest_path, manifest
    )

    info = load_soft_cluster_schedule_manifest(manifest_path)
    contract = build_soft_cluster_schedule_contract(info)

    assert info.schema_version == 2
    assert info.format == SOFT_CLUSTER_SCHEDULE_FORMAT_V2
    assert info.catalog_migration is not None
    assert info.catalog_migration["source_dataset"]["binding"]["train_sets"] == [
        "pdb",
        "legacy_short",
        "legacy_long",
        "disorder",
    ]
    assert contract["catalog_migration"] == info.catalog_migration

    sampler = DistributedPrecomputedScheduleSampler(
        _SizedDataset(20),
        manifest_path=manifest_path,
        num_replicas=3,
        rank=1,
        emit_sample_occurrence=True,
    )
    target_signature = _legacy_signature(manifest, rank=1, emit=True)
    assert sampler.validate_target_dataset_signature(target_signature)
    source_manifest = dict(manifest)
    source_manifest["dataset"] = manifest["catalog_migration"]["source_dataset"][
        "binding"
    ]
    source_signature = _legacy_signature(source_manifest, rank=1, emit=True)
    assert sampler.validate_legacy_resume_signature(
        source_signature,
        consumed_global_batches=manifest["sampler"][
            "origin_consumed_global_batches"
        ],
    )
    with pytest.raises(SoftClusterScheduleError, match="child_fingerprints"):
        bad_target = dict(target_signature)
        bad_target["child_fingerprints"] = list(
            target_signature["child_fingerprints"]
        )
        bad_target["child_fingerprints"][1] = dict(
            bad_target["child_fingerprints"][1]
        )
        bad_target["child_fingerprints"][1]["sha256"] = "f" * 64
        sampler.validate_target_dataset_signature(bad_target)


def test_v2_catalog_migration_rejects_occurrence_loss(tmp_path: Path):
    manifest_path, manifest, _ = _build_schedule(tmp_path / "schedule")
    manifest = _upgrade_schedule_to_catalog_migration_v2(
        manifest_path, manifest
    )
    manifest["catalog_migration"]["crosswalk"]["accounting"][
        "mapped_occurrences"
    ] = 5
    _write_json(manifest_path, manifest)
    success_path = manifest_path.parent / "_SUCCESS.json"
    success = json.loads(success_path.read_text(encoding="utf-8"))
    success["manifest_sha256"] = _sha256(manifest_path)
    _write_json(success_path, success)
    clear_soft_cluster_schedule_cache()

    with pytest.raises(SoftClusterScheduleError, match="conservation"):
        load_soft_cluster_schedule_manifest(manifest_path)


def test_schedule_loader_rejects_missing_receipt_and_tampering(tmp_path: Path):
    manifest_path, _, _ = _build_schedule(tmp_path / "schedule")
    success_path = manifest_path.parent / "_SUCCESS.json"
    success_path.unlink()
    clear_soft_cluster_schedule_cache()
    with pytest.raises(SoftClusterScheduleError, match="receipt is missing"):
        load_soft_cluster_schedule_manifest(manifest_path)

    _, _, _ = _build_schedule(tmp_path / "schedule")
    rows_path = manifest_path.parent / "global_rows.u32"
    with rows_path.open("r+b") as handle:
        handle.seek(0)
        handle.write((19).to_bytes(4, "little"))
    clear_soft_cluster_schedule_cache()
    with pytest.raises(SoftClusterScheduleError, match="SHA-256 mismatch"):
        load_soft_cluster_schedule_manifest(manifest_path)


def test_schedule_loader_rejects_out_of_range_authenticated_row(tmp_path: Path):
    manifest_path, manifest, rows = _build_schedule(tmp_path / "schedule")
    rows[0] = 20
    rows.tofile(manifest_path.parent / "global_rows.u32")
    manifest["artifacts"]["global_rows"]["sha256"] = _sha256(
        manifest_path.parent / "global_rows.u32"
    )
    _write_json(manifest_path, manifest)
    success = json.loads((manifest_path.parent / "_SUCCESS.json").read_text())
    success["manifest_sha256"] = _sha256(manifest_path)
    success["schedule_sha256"] = manifest["artifacts"]["global_rows"]["sha256"]
    _write_json(manifest_path.parent / "_SUCCESS.json", success)
    clear_soft_cluster_schedule_cache()

    with pytest.raises(SoftClusterScheduleError, match="outside"):
        load_soft_cluster_schedule_manifest(manifest_path)


def test_expected_runtime_binding_is_fail_closed(tmp_path: Path):
    manifest_path, _, _ = _build_schedule(tmp_path / "schedule")

    with pytest.raises(SoftClusterScheduleError, match="world_size mismatch"):
        load_soft_cluster_schedule_manifest(
            manifest_path,
            expected={"world_size": 4},
        )
    with pytest.raises(SoftClusterScheduleError, match="unsupported fields"):
        load_soft_cluster_schedule_manifest(
            manifest_path,
            expected={"unknown": 1},
        )
    with pytest.raises(SoftClusterScheduleError, match="checkpoint_pair_id"):
        load_soft_cluster_schedule_manifest(
            manifest_path,
            expected={"source_checkpoint": {"checkpoint_pair_id": "9" * 64}},
        )


def test_cursor_mapping_handles_epoch_size_not_divisible_by_world_size():
    assert [rank_local_epoch_size(8, 3, rank) for rank in range(3)] == [3, 3, 2]
    for rank in range(3):
        for consumed in range(20):
            epoch, offset = sampler_cursor_from_consumed_batches(
                consumed,
                epoch_size=8,
                world_size=3,
                rank=rank,
            )
            assert (
                consumed_batches_from_sampler_cursor(
                    epoch,
                    offset,
                    epoch_size=8,
                    world_size=3,
                    rank=rank,
                )
                == consumed
            )


def test_distributed_sampler_reads_synchronized_waves_across_uneven_epochs(
    tmp_path: Path,
):
    manifest_path, _, rows = _build_schedule(tmp_path / "schedule")
    schedule = load_soft_cluster_schedule(manifest_path)
    dataset = _SizedDataset(20)

    for rank in range(3):
        sampler = DistributedPrecomputedScheduleSampler(
            dataset,
            schedule=schedule,
            num_replicas=3,
            rank=rank,
            epoch_size=8,
            gamma=0.2,
            seed=42,
            emit_sample_occurrence=True,
        )
        sampler.set_consumed_global_batches(4)
        actual = list(sampler)
        assert all(isinstance(item, SampleOccurrence) for item in actual)
        assert [item.index for item in actual] == [
            int(rows[rank]),
            int(rows[3 + rank]),
        ]
        assert sampler.consumed_global_batches() == 4
        assert len(json.dumps(sampler.resume_signature())) < 2_000


def _legacy_signature(manifest: dict, *, rank: int, emit: bool) -> dict:
    dataset = manifest["dataset"]
    sampler = manifest["sampler"]
    return {
        "sampler_type": "DistributedHierarchicalWeightedSampler",
        "layout": "hierarchical_strided_unpadded_v1",
        "seed": sampler["seed"],
        "num_samples": sampler["epoch_size"],
        "replacement": True,
        "num_replicas": sampler["world_size"],
        "rank": rank,
        "emit_sample_occurrence": emit,
        "dataset_offsets": dataset["dataset_offsets"][:-1],
        "dataset_masses": dataset["dataset_masses"],
        "child_fingerprints": dataset["child_fingerprints"],
    }


def test_legacy_signature_migration_is_checkpoint_bound_and_one_time(tmp_path: Path):
    manifest_path, manifest, _ = _build_schedule(tmp_path / "schedule")
    sampler = DistributedPrecomputedScheduleSampler(
        _SizedDataset(20),
        manifest_path=manifest_path,
        num_replicas=3,
        rank=1,
        epoch_size=8,
        gamma=0.2,
        seed=42,
        emit_sample_occurrence=True,
    )
    legacy = _legacy_signature(manifest, rank=1, emit=True)

    assert sampler.validate_legacy_resume_signature(
        legacy,
        consumed_global_batches=4,
    )
    assert (
        sampler.validate_resume_signature(
            legacy,
            consumed_global_batches=4,
            allow_legacy_migration=True,
        )
        == "legacy_migration"
    )
    with pytest.raises(SoftClusterScheduleError, match="configuration changed"):
        sampler.validate_resume_signature(
            legacy,
            consumed_global_batches=4,
            allow_legacy_migration=False,
        )
    with pytest.raises(SoftClusterScheduleError, match="cursor"):
        sampler.validate_legacy_resume_signature(
            legacy,
            consumed_global_batches=5,
        )
    bad = dict(legacy)
    bad["dataset_masses"] = [0.48, 0.005, 0.505, 0.02]
    with pytest.raises(SoftClusterScheduleError, match="dataset_masses"):
        sampler.validate_legacy_resume_signature(
            bad,
            consumed_global_batches=4,
        )

    assert (
        sampler.validate_resume_signature(
            sampler.resume_signature(),
            consumed_global_batches=5,
        )
        == "current"
    )


def test_sampler_state_is_small_and_rejects_schedule_change(tmp_path: Path):
    manifest_path, _, _ = _build_schedule(tmp_path / "schedule")
    source = DistributedPrecomputedScheduleSampler(
        _SizedDataset(20),
        manifest_path=manifest_path,
        num_replicas=3,
        rank=0,
        emit_sample_occurrence=False,
    )
    source.set_consumed_global_batches(5)
    state = source.state_dict()
    assert len(json.dumps(state)) < 3_000

    restored = DistributedPrecomputedScheduleSampler(
        _SizedDataset(20),
        manifest_path=manifest_path,
        num_replicas=3,
        rank=0,
        emit_sample_occurrence=False,
    )
    restored.load_state_dict(state)
    assert restored.consumed_global_batches() == 5
    assert list(restored) == list(source)

    state["signature"] = dict(state["signature"])
    state["signature"]["gamma"] = 0.1
    with pytest.raises(SoftClusterScheduleError, match="signature mismatch"):
        restored.load_state_dict(state)


def _legacy_loader_state(manifest: dict, *, rank: int, consumed: int) -> dict:
    epoch, offset = sampler_cursor_from_consumed_batches(
        consumed,
        epoch_size=manifest["sampler"]["epoch_size"],
        world_size=manifest["sampler"]["world_size"],
        rank=rank,
    )
    return {
        "schema_version": 1,
        "sampler_epoch": epoch,
        "next_local_offset": offset,
        "consumed_global_batches": consumed,
        "batch_size": 1,
        "num_workers": 0,
        "persistent_workers": False,
        "prefetch_factor": None,
        "pin_memory": False,
        "sampler": _legacy_signature(manifest, rank=rank, emit=True),
    }


def test_resumable_loader_allows_only_explicit_checkpoint_bound_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, manifest, rows = _build_schedule(tmp_path / "schedule")
    dataset = _SizedDataset(20)

    def make_loader() -> DistributedDataLoader:
        sampler = DistributedPrecomputedScheduleSampler(
            dataset,
            manifest_path=manifest_path,
            num_replicas=3,
            rank=1,
            epoch_size=8,
            gamma=0.2,
            seed=42,
            emit_sample_occurrence=True,
        )
        return DistributedDataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            sampler=sampler,
        )

    legacy_state = _legacy_loader_state(manifest, rank=1, consumed=4)
    monkeypatch.delenv(
        "PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION",
        raising=False,
    )
    with pytest.raises(ValueError, match="migration was rejected"):
        make_loader().load_state_dict(legacy_state)

    monkeypatch.setenv(
        "PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION",
        "soft_cluster_schedule_manifest_full_state_v1",
    )
    migrated = make_loader()
    migrated.load_state_dict(legacy_state)
    assert migrated.state_dict()["consumed_global_batches"] == 4
    first = next(iter(migrated))
    assert int(first.item()) == int(rows[1])

    wrong_boundary = dict(legacy_state)
    wrong_boundary["consumed_global_batches"] = 5
    wrong_boundary["sampler_epoch"], wrong_boundary["next_local_offset"] = (
        sampler_cursor_from_consumed_batches(
            5,
            epoch_size=8,
            world_size=3,
            rank=1,
        )
    )
    with pytest.raises(ValueError, match="migration was rejected"):
        make_loader().load_state_dict(wrong_boundary)


def test_get_dataloaders_selects_and_binds_precomputed_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, manifest, _ = _build_schedule(
        tmp_path / "schedule",
        world_size=1,
        epoch_size=8,
    )
    dataset = _HierarchicalSizedDataset()
    legacy = DistributedHierarchicalWeightedSampler(
        dataset,
        dataset_offsets=dataset.hierarchical_dataset_offsets,
        dataset_masses=dataset.hierarchical_dataset_masses,
        child_sampling_specs=dataset.hierarchical_child_sampling_specs,
        num_samples=8,
        num_replicas=1,
        rank=0,
        replacement=True,
        seed=42,
        emit_sample_occurrence=True,
    )
    manifest["dataset"]["child_fingerprints"] = legacy.resume_signature()[
        "child_fingerprints"
    ]
    _write_json(manifest_path, manifest)
    success_path = manifest_path.parent / "_SUCCESS.json"
    success = json.loads(success_path.read_text(encoding="utf-8"))
    success["manifest_sha256"] = _sha256(manifest_path)
    _write_json(success_path, success)
    clear_soft_cluster_schedule_cache()

    monkeypatch.setattr(
        "protenix.data.pipeline.dataloader.get_datasets",
        lambda *_args, **_kwargs: (dataset, {}),
    )
    configs = ConfigDict(
        {
            "data": {
                "num_dl_workers": 0,
                "epoch_size": 8,
                "train_sampler": {
                    "seed": 42,
                    "replacement": True,
                    "soft_cluster_schedule_manifest": str(manifest_path),
                    "soft_cluster_gamma": 0.2,
                },
            }
        }
    )

    train_dl, _ = get_dataloaders(configs, world_size=1, seed=42)

    assert isinstance(
        train_dl.sampler,
        DistributedPrecomputedScheduleSampler,
    )
    assert train_dl.sampler.schedule.info.manifest_path == manifest_path
