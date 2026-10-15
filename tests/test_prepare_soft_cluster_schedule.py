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

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from protenix.data.pipeline.soft_exposure_sampler import (
    SoftExposureBucketSampler,
)
from protenix.data.pipeline.soft_cluster_schedule import (
    clear_soft_cluster_schedule_cache,
    load_soft_cluster_schedule_manifest,
)
from scripts.runtime.audit_train_sampler_coverage import (
    DatasetCatalog,
    _hierarchical_sampler,
)
from scripts.runtime.prepare_soft_cluster_schedule import (
    _canonical_sha256,
    _atomic_output_directory,
    _bind_initial_coverage_artifacts,
    _file_identity,
    _sha256_stable_file,
    _validate_initial_coverage_boundary,
    _verify_hashed_input_snapshots,
    _rank_cursor_binding,
    generate_global_rows,
    project_global_id_occurrences,
    row_counts_to_cluster_exposure,
)
from scripts.runtime import prepare_soft_cluster_schedule as prepare_module


class _SizedDataset:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


class _ToyRuntimeIndex:
    def __init__(self, cluster_sizes) -> None:
        self.cluster_sizes = np.asarray(cluster_sizes, dtype="<u4")
        self.cluster_count = len(self.cluster_sizes)
        starts = np.zeros(self.cluster_count, dtype="<u8")
        if self.cluster_count > 1:
            np.cumsum(self.cluster_sizes[:-1], out=starts[1:])
        self.first_rows = starts
        self.row_count = int(self.cluster_sizes.sum())
        self.runtime_dir = Path("/toy/runtime")
        self.sampling_fingerprint = (
            f"toy-{','.join(str(int(value)) for value in self.cluster_sizes)}"
        )
        self._buffers = {
            "first_row_ids": self.first_rows.tobytes(),
            "cluster_sizes": self.cluster_sizes.tobytes(),
        }

    def _map(self, name):
        return self._buffers[name]

    def sample_member_rows(self, ordinals, uniforms):
        ordinals = np.asarray(ordinals, dtype=np.int64)
        return self.first_rows[ordinals] + np.floor(
            np.asarray(uniforms) * self.cluster_sizes[ordinals]
        ).astype(np.int64)

    def resume_signature(self):
        return {
            "fixture": self.sampling_fingerprint,
            "row_count": self.row_count,
            "cluster_count": self.cluster_count,
        }


def _catalog(name, size, sampling_spec, runtime_index=None):
    return DatasetCatalog(
        name=name,
        dataset=_SizedDataset(size),
        sampling_spec=sampling_spec,
        runtime_index=runtime_index,
        random_sample_if_failed=False,
        source_identity={"fixture": name},
    )


def _toy_catalogs():
    short = _ToyRuntimeIndex([2, 1, 1])
    long = _ToyRuntimeIndex([1, 2])
    catalogs = [
        _catalog("weighted", 3, np.asarray([1.0, 2.0, 3.0])),
        _catalog(
            "short_distillation",
            short.row_count,
            {"kind": "clustered", "source": short},
            short,
        ),
        _catalog(
            "long_distillation",
            long.row_count,
            {"kind": "clustered", "source": long},
            long,
        ),
        _catalog("disorder", 2, np.asarray([4.0, 1.0])),
    ]
    return catalogs, short, long


def _legacy_slice(catalogs, masses, *, start, count, epoch_size, world_size, seed):
    sampler = _hierarchical_sampler(
        catalogs=catalogs,
        masses=masses,
        epoch_size=epoch_size,
        world_size=world_size,
        seed=seed,
    )
    assert sampler is not None
    chunks = []
    consumed = 0
    while consumed < count:
        absolute = start + consumed
        epoch, offset = divmod(absolute, epoch_size)
        sampler.set_epoch(epoch)
        values = np.asarray(sampler._global_indices(), dtype=np.int64)
        take = min(count - consumed, epoch_size - offset)
        chunks.append(values[offset : offset + take])
        consumed += take
    return np.concatenate(chunks)


def test_row_occurrences_are_summed_over_cluster_members():
    runtime = _ToyRuntimeIndex([2, 1, 3])
    rows = np.asarray([2, 5, 7, 11, 13, 17], dtype=np.uint32)

    exposure = row_counts_to_cluster_exposure(runtime, rows)

    np.testing.assert_array_equal(exposure, [7, 7, 41])
    assert exposure.dtype == np.uint32
    assert int(exposure.sum()) == int(rows.sum())


def test_v2_global_id_history_projection_conserves_occurrences_and_floors_ambiguity():
    source = {
        "AFQ|seen-in-ge200": 4,
        "MGN|seen-in-residual": 3,
        "AFQ|intentional-gap": 2,
    }
    assignments = {
        "AFQ|seen-in-ge200": ("ge200", 1),
        "MGN|seen-in-residual": ("residual", 0),
    }

    exposures, audit = project_global_id_occurrences(
        source_occurrences=source,
        target_assignments=assignments,
        target_cluster_counts={"residual": 2, "ge200": 3},
        intentional_missing={"AFQ|intentional-gap": "fixture_missing_asset"},
        ambiguity_alternates=[("AFQ|seen-in-ge200", "ge200", 2)],
    )

    np.testing.assert_array_equal(exposures["residual"], [3, 0])
    np.testing.assert_array_equal(exposures["ge200"], [0, 4, 1])
    assert audit["source_history_occurrences"] == 9
    assert audit["mapped_occurrences"] == 7
    assert audit["intentional_missing_occurrences"] == 2
    assert audit["target_projected_occurrences"] == 7
    assert audit["ambiguous_floor_increment"] == 1
    assert audit["target_initial_exposure_total"] == 8
    assert audit["unresolved_occurrences"] == 0


def test_v2_global_id_history_projection_rejects_unexplained_loss_and_duplicate_alternate():
    with pytest.raises(ValueError, match="not intentional-missing"):
        project_global_id_occurrences(
            source_occurrences={"AFQ|missing": 2},
            target_assignments={},
            target_cluster_counts={"ge200": 2},
        )

    with pytest.raises(ValueError, match="duplicate ambiguity"):
        project_global_id_occurrences(
            source_occurrences={"AFQ|seen": 1},
            target_assignments={"AFQ|seen": ("ge200", 0)},
            target_cluster_counts={"ge200": 2},
            ambiguity_alternates=[
                ("AFQ|seen", "ge200", 1),
                ("AFQ|seen", "ge200", 1),
            ],
        )


def test_exact_non_epoch_aligned_schedule_preserves_outer_and_nonsoft_rows(
    tmp_path,
):
    catalogs, short, long = _toy_catalogs()
    masses = [0.4, 0.2, 0.3, 0.1]
    epoch_size = 8
    world_size = 2
    seed = 37
    origin_waves = 3
    target_waves = 10
    expected_count = (target_waves - origin_waves) * world_size
    legacy = _legacy_slice(
        catalogs,
        masses,
        start=origin_waves * world_size,
        count=expected_count,
        epoch_size=epoch_size,
        world_size=world_size,
        seed=seed,
    )
    initial = {
        "short_distillation": np.asarray([2, 0, 1], dtype=np.uint32),
        "long_distillation": np.asarray([0, 3], dtype=np.uint32),
    }
    samplers = {
        name: SoftExposureBucketSampler(values, gamma=0.2, seed=index + 100)
        for index, (name, values) in enumerate(initial.items())
    }
    path = tmp_path / "global_rows.u32"

    draws = generate_global_rows(
        output_path=path,
        catalogs=catalogs,
        masses=masses,
        soft_samplers=samplers,
        member_seeds={"short_distillation": 11, "long_distillation": 12},
        origin_consumed_batches_per_rank=origin_waves,
        target_consumed_batches_per_rank=target_waves,
        accumulation=4,
        world_size=world_size,
        epoch_size=epoch_size,
        seed=seed,
    )

    generated = np.fromfile(path, dtype="<u4").astype(np.int64)
    assert len(generated) == expected_count
    offsets = np.cumsum([0] + [catalog.size for catalog in catalogs])
    legacy_children = np.searchsorted(offsets[1:], legacy, side="right")
    generated_children = np.searchsorted(offsets[1:], generated, side="right")
    np.testing.assert_array_equal(generated_children, legacy_children)
    nonsoft = np.isin(legacy_children, [0, 3])
    np.testing.assert_array_equal(generated[nonsoft], legacy[nonsoft])
    assert draws["short_distillation"] == int((legacy_children == 1).sum())
    assert draws["long_distillation"] == int((legacy_children == 2).sum())
    for name, values in initial.items():
        assert int(samplers[name].exposure_counts.sum()) == (
            int(values.sum()) + draws[name]
        )
    assert short.row_count == 4
    assert long.row_count == 3


def test_schedule_is_deterministic_for_same_source_state(tmp_path):
    catalogs, _, _ = _toy_catalogs()
    masses = [0.4, 0.2, 0.3, 0.1]
    paths = [tmp_path / "left.u32", tmp_path / "right.u32"]
    for path in paths:
        generate_global_rows(
            output_path=path,
            catalogs=catalogs,
            masses=masses,
            soft_samplers={
                "short_distillation": SoftExposureBucketSampler(
                    np.asarray([0, 2, 0], dtype=np.uint32),
                    gamma=0.2,
                    seed=100,
                ),
                "long_distillation": SoftExposureBucketSampler(
                    np.asarray([1, 0], dtype=np.uint32),
                    gamma=0.2,
                    seed=200,
                ),
            },
            member_seeds={"short_distillation": 11, "long_distillation": 12},
            origin_consumed_batches_per_rank=1,
            target_consumed_batches_per_rank=14,
            accumulation=4,
            world_size=2,
            epoch_size=8,
            seed=9,
        )
    assert paths[0].read_bytes() == paths[1].read_bytes()


def test_checkpoint_rank_cursor_and_sampler_signature_are_fail_closed():
    catalogs, _, _ = _toy_catalogs()
    masses = [0.4, 0.2, 0.3, 0.1]
    hierarchical = _hierarchical_sampler(
        catalogs=catalogs,
        masses=masses,
        epoch_size=8,
        world_size=2,
        seed=5,
    )
    assert hierarchical is not None
    rank0 = hierarchical.resume_signature()
    states = []
    for rank in range(2):
        signature = dict(rank0)
        signature["rank"] = rank
        states.append(
            {
                "rank": rank,
                "train_dataloader": {
                    "schema_version": 1,
                    "batch_size": 1,
                    "sampler_epoch": 1,
                    "next_local_offset": 2,
                    "consumed_global_batches": 6,
                    "sampler": signature,
                },
            }
        )
    raw = {
        "rank_runtime_states": {
            "schema_version": 1,
            "world_size": 2,
            "states": states,
        }
    }

    cursors, digest = _rank_cursor_binding(
        raw_checkpoint=raw,
        expected_signature_rank0=rank0,
        completed_updates=3,
        accumulation=2,
        world_size=2,
        epoch_size=8,
    )

    assert [record["rank"] for record in cursors] == [0, 1]
    assert len(digest) == 64
    raw["rank_runtime_states"]["states"][1]["train_dataloader"][
        "next_local_offset"
    ] = 3
    with pytest.raises(ValueError, match="cursor mismatch"):
        _rank_cursor_binding(
            raw_checkpoint=raw,
            expected_signature_rank0=rank0,
            completed_updates=3,
            accumulation=2,
            world_size=2,
            epoch_size=8,
        )

    raw["rank_runtime_states"]["states"][1]["train_dataloader"][
        "next_local_offset"
    ] = 2
    raw["rank_runtime_states"]["states"][1]["train_dataloader"][
        "batch_size"
    ] = 2
    with pytest.raises(ValueError, match="batch_size=1"):
        _rank_cursor_binding(
            raw_checkpoint=raw,
            expected_signature_rank0=rank0,
            completed_updates=3,
            accumulation=2,
            world_size=2,
            epoch_size=8,
        )


def test_atomic_output_is_invisible_on_failure_and_refuses_overwrite(tmp_path):
    destination = tmp_path / "schedule"
    with pytest.raises(RuntimeError, match="abort"):
        with _atomic_output_directory(destination) as temporary:
            (temporary / "partial").write_text("not published", encoding="utf-8")
            raise RuntimeError("abort")
    assert not destination.exists()
    assert not list(tmp_path.glob(".schedule.tmp.*"))

    with _atomic_output_directory(destination) as temporary:
        (temporary / "complete").write_text("ok", encoding="utf-8")
    assert (destination / "complete").read_text(encoding="utf-8") == "ok"
    with pytest.raises(FileExistsError, match="overwrite"):
        with _atomic_output_directory(destination):
            pass


def test_bound_input_mutation_aborts_atomic_publication(tmp_path):
    source = tmp_path / "runtime-array.bin"
    source.write_bytes(b"before")
    snapshots = {
        source: {
            "identity": _file_identity(source),
            "sha256": _sha256_stable_file(source),
        }
    }
    destination = tmp_path / "schedule"

    with pytest.raises(RuntimeError, match="Bound sampler inputs changed"):
        with _atomic_output_directory(destination) as temporary:
            (temporary / "global_rows.u32").write_bytes(b"partial")
            source.write_bytes(b"after!")
            _verify_hashed_input_snapshots(snapshots)

    assert not destination.exists()
    assert not list(tmp_path.glob(".schedule.tmp.*"))


def test_initial_coverage_binds_consumed_sparse_row_counts(tmp_path):
    np.savez_compressed(
        tmp_path / "short.seen_rows.npz",
        row_id=np.asarray([1], dtype=np.int64),
        sample_count=np.asarray([2], dtype=np.uint32),
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "datasets": {
                    "short": {
                        "artifacts": {
                            "seen_rows_npz": "short.seen_rows.npz"
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    records = [
        {
            "coverage_dir": str(tmp_path),
            "summary_path": str(summary_path),
            "datasets": ["short"],
        }
    ]

    _bind_initial_coverage_artifacts(records)

    binding = records[0]["consumed_artifacts"]["short"]
    assert binding["size_bytes"] == (tmp_path / "short.seen_rows.npz").stat().st_size
    assert len(binding["sha256"]) == 64


def test_initial_coverage_must_prove_exact_nonoverlapping_prefix(tmp_path):
    coverage = tmp_path / "coverage"
    coverage.mkdir()
    summary_path = coverage / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "event": "planned_train_sampler_coverage",
                "updates": 5,
                "coverage_total_occurrences": 5 * 2 * 3,
            }
        ),
        encoding="utf-8",
    )

    _validate_initial_coverage_boundary(
        coverage_dirs=[coverage],
        history_start_update=5,
        accumulation=2,
        world_size=3,
    )
    with pytest.raises(ValueError, match="forbidden"):
        _validate_initial_coverage_boundary(
            coverage_dirs=[coverage],
            history_start_update=0,
            accumulation=2,
            world_size=3,
        )
    with pytest.raises(ValueError, match="exactly one"):
        _validate_initial_coverage_boundary(
            coverage_dirs=[],
            history_start_update=5,
            accumulation=2,
            world_size=3,
        )
    with pytest.raises(ValueError, match="updates"):
        _validate_initial_coverage_boundary(
            coverage_dirs=[coverage],
            history_start_update=4,
            accumulation=2,
            world_size=3,
        )


def test_builder_manifest_extensions_are_accepted_by_runtime(tmp_path):
    root = tmp_path / "schedule"
    root.mkdir()
    rows = np.asarray([0, 1, 2, 3], dtype="<u4")
    rows.tofile(root / "global_rows.u32")
    np.asarray([0, 1], dtype="<u4").tofile(root / "short.initial.u32")
    np.asarray([1, 2], dtype="<u4").tofile(root / "short.final.u32")

    def record(name):
        path = root / name
        return {
            "path": name,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    artifacts = {
        "global_rows": record("global_rows.u32"),
        "short.initial_exposure": record("short.initial.u32"),
        "short.final_exposure": record("short.final.u32"),
    }
    artifact_set_sha256 = _canonical_sha256(artifacts)
    artifacts["artifact_set_sha256"] = artifact_set_sha256
    manifest = {
        "schema_version": 1,
        "format": "protenix.soft_cluster_schedule.v1",
        "status": "complete",
        "layout": "wave_major_rank_v1",
        "algorithm": {
            "format": "protenix.soft_exposure_bucket_sampler.v1",
            "gamma": 0.2,
            "member_policy": "deterministic_rotation_v1",
            "consumption_order": "synchronized_microstep_then_rank_ascending_v1",
            "soft_cluster_datasets": ["short"],
        },
        "shape": {"dtype": "<u4", "row_count": 4, "wave_count": 2},
        "dataset": {
            "flat_row_count": 4,
            "train_sets": ["short"],
            "dataset_offsets": [0, 4],
            "dataset_masses": [1.0],
            "child_fingerprints": [
                {"kind": "clustered", "size": 4, "sha256": "a" * 64}
            ],
            "runtime_bindings": {
                "short": {
                    "runtime_dir": "/toy/runtime",
                    "row_count": 4,
                    "cluster_count": 2,
                    "manifest_sha256": "b" * 64,
                    "sampling_fingerprint": "c" * 64,
                }
            },
        },
        "sampler": {
            "seed": 42,
            "epoch_size": 4,
            "world_size": 2,
            "origin_consumed_global_batches": 4,
            "target_consumed_global_batches": 6,
            "target_total_rows": 4,
        },
        "source_checkpoint": {
            "mode": "checkpoint_migration",
            "raw_checkpoint_path": os.path.realpath(root / "raw.pt"),
            "raw_checkpoint_sha256": "1" * 64,
            "ema_checkpoint_path": os.path.realpath(root / "ema.pt"),
            "ema_checkpoint_sha256": "2" * 64,
            "checkpoint_step": 1,
            "completed_updates": 2,
            "next_global_step": 4,
            "origin_consumed_global_batches_per_rank": 4,
            "iters_to_accumulate": 2,
            "checkpoint_pair_id": "3" * 64,
            "source_training_contract_id": "toy-contract",
            "source_data_contract_sha256": "4" * 64,
            "source_run_id": "toy-run",
            "model_architecture_sha256": "5" * 64,
        },
        "source_run": {"config_sha256": "6" * 64},
        "provenance": {
            "exposure": {"short": {"future_draws": 2}},
            "epoch_size_divisible_by_world_size": True,
        },
        "artifacts": artifacts,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    success = {
        "schema_version": 1,
        "status": "complete",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "schedule_path": "global_rows.u32",
        "schedule_size_bytes": artifacts["global_rows"]["size_bytes"],
        "schedule_sha256": artifacts["global_rows"]["sha256"],
        "artifact_set_sha256": artifact_set_sha256,
    }
    (root / "_SUCCESS.json").write_text(
        json.dumps(success), encoding="utf-8"
    )
    clear_soft_cluster_schedule_cache()

    info = load_soft_cluster_schedule_manifest(manifest_path)

    assert info.row_count == 4
    assert info.gamma == 0.2


@pytest.mark.parametrize(
    ("checkpoint_step", "target_updates"),
    [(1, 4), (4, 7)],
)
def test_prepare_schedule_toy_checkpoint_to_runtime_end_to_end(
    tmp_path, monkeypatch, checkpoint_step, target_updates
):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    manifest_dir = run_dir / "manifest"
    checkpoint_dir.mkdir(parents=True)
    manifest_dir.mkdir()
    catalogs, _, _ = _toy_catalogs()
    masses = [0.4, 0.2, 0.3, 0.1]
    config = {
        "seed": 42,
        "iters_to_accumulate": 2,
        "max_steps": target_updates,
        "data": {
            "epoch_size": 8,
            "train_sets": [catalog.name for catalog in catalogs],
            "train_sampler": {
                "seed": -1,
                "replacement": True,
                "train_sample_weights": masses,
            },
        },
    }
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    run_name = "toy-source-run"
    data_contract_sha = "d" * 64
    (manifest_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "distributed": {"world_size": 2},
                "data": {"contract": {"sha256": data_contract_sha}},
            }
        ),
        encoding="utf-8",
    )
    hierarchical = _hierarchical_sampler(
        catalogs=catalogs,
        masses=masses,
        epoch_size=8,
        world_size=2,
        seed=42,
    )
    assert hierarchical is not None
    rank0_signature = hierarchical.resume_signature()
    completed_updates = checkpoint_step + 1
    origin_waves = completed_updates * 2
    cursor_epoch, cursor_offset = divmod(origin_waves, 4)
    rank_states = []
    for rank in range(2):
        signature = dict(rank0_signature)
        signature["rank"] = rank
        rank_states.append(
            {
                "rank": rank,
                "train_dataloader": {
                    "schema_version": 1,
                    "batch_size": 1,
                    "sampler_epoch": cursor_epoch,
                    "next_local_offset": cursor_offset,
                    "consumed_global_batches": origin_waves,
                    "sampler": signature,
                },
            }
        )
    contract = {
        "checkpoint_pair_id": "a" * 64,
        "training_contract_id": "toy-training-contract",
        "data_contract_sha256": data_contract_sha,
        "run_id": run_name,
        "model_architecture_sha256": "b" * 64,
        "step": checkpoint_step,
        "next_step": completed_updates,
        "next_global_step": origin_waves,
        "iters_to_accumulate": 2,
        "world_size": 2,
    }
    raw_path = checkpoint_dir / f"{checkpoint_step}.pt"
    ema_path = checkpoint_dir / f"{checkpoint_step}_ema_0.999.pt"
    raw_payload = {
        "format_version": 2,
        "checkpoint_kind": "raw",
        "step": checkpoint_step,
        "resume_contract": contract,
        "rank_runtime_states": {
            "schema_version": 1,
            "world_size": 2,
            "states": rank_states,
        },
    }
    torch.save(raw_payload, raw_path)
    torch.save(
        {
            "format_version": 2,
            "checkpoint_kind": "ema",
            "step": checkpoint_step,
            "resume_contract": contract,
        },
        ema_path,
    )
    monkeypatch.setattr(prepare_module, "build_catalogs", lambda _: catalogs)
    monkeypatch.setattr(
        prepare_module,
        "_runtime_binding",
        lambda index: {
            "runtime_dir": str(index.runtime_dir),
            "row_count": index.row_count,
            "cluster_count": index.cluster_count,
            "manifest_sha256": "c" * 64,
            "sampling_fingerprint": hashlib.sha256(
                index.sampling_fingerprint.encode()
            ).hexdigest(),
        },
    )
    output_dir = tmp_path / "published"
    args = argparse.Namespace(
        source_checkpoint=raw_path,
        source_ema_checkpoint=None,
        source_run_dir=run_dir,
        target_updates=target_updates,
        output_dir=output_dir,
        gamma=0.2,
        history_start_update=0,
        initial_coverage=[],
        soft_dataset=[],
        log_level="INFO",
    )

    manifest_path = prepare_module.prepare_schedule(args)
    info = load_soft_cluster_schedule_manifest(manifest_path)

    assert info.row_count == (
        target_updates - completed_updates
    ) * 2 * 2
    assert info.origin_consumed_global_batches == origin_waves
    assert info.target_consumed_global_batches == target_updates * 2
    assert info.source_checkpoint["checkpoint_step"] == checkpoint_step
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert document["source_run"]["rank_cursors"][1]["rank"] == 1
    assert document["provenance"]["exposure"][
        "short_distillation"
    ]["future_draws"] >= 0
    assert (output_dir / "_SUCCESS.json").is_file()

    if checkpoint_step == 1:
        bad_payload = copy.deepcopy(raw_payload)
        bad_payload["rank_runtime_states"]["states"][1]["train_dataloader"][
            "next_local_offset"
        ] += 1
        bad_path = checkpoint_dir / "bad_cursor.pt"
        torch.save(bad_payload, bad_path)
        args.source_checkpoint = bad_path
        args.output_dir = tmp_path / "bad-published"
        with pytest.raises(ValueError, match="cursor mismatch"):
            prepare_module.prepare_schedule(args)
        assert not args.output_dir.exists()

        bad_ema_payload = {
            "format_version": 2,
            "checkpoint_kind": "ema",
            "step": checkpoint_step,
            "resume_contract": dict(contract),
        }
        bad_ema_payload["resume_contract"]["next_global_step"] += 1
        bad_ema_path = checkpoint_dir / "bad_ema_0.999.pt"
        torch.save(bad_ema_payload, bad_ema_path)
        args.source_checkpoint = raw_path
        args.source_ema_checkpoint = bad_ema_path
        args.output_dir = tmp_path / "bad-ema-published"
        with pytest.raises(ValueError, match="raw/EMA resume contracts differ"):
            prepare_module.prepare_schedule(args)
        assert not args.output_dir.exists()
