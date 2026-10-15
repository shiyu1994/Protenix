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
import json

import numpy as np
import pytest
import torch
import yaml

from protenix.data.pipeline.dataloader import DistributedDataLoader
from protenix.data.pipeline.soft_cluster_schedule import (
    DistributedPrecomputedScheduleSampler,
    SoftClusterScheduleError,
    _validate_source_checkpoint,
    load_soft_cluster_schedule,
    load_soft_cluster_schedule_manifest,
)
from scripts.runtime import prepare_soft_cluster_schedule as prepare_module
from tests.test_prepare_soft_cluster_schedule import (
    _catalog,
    _legacy_slice,
    _toy_catalogs,
)


@pytest.fixture
def fresh_plan(tmp_path, monkeypatch):
    catalogs, _, _ = _toy_catalogs()
    catalogs.extend([
        _catalog("sabdab_train", 3, np.asarray([1.0, 2.0, 1.0])),
        _catalog("sabdab_distill", 4, np.asarray([1.0, 1.0, 2.0, 1.0])),
    ])
    masses = [0.35, 0.15, 0.2, 0.1, 0.1, 0.1]
    config = {
        "seed": 42, "iters_to_accumulate": 2, "max_steps": 40,
        "data": {
            "train_sets": [catalog.name for catalog in catalogs],
            "epoch_size": 8,
            "train_sampler": {"seed": -1, "replacement": True, "train_sample_weights": masses},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(prepare_module, "build_catalogs", lambda _: catalogs)
    monkeypatch.setattr(prepare_module, "_runtime_binding", lambda index: {
        "runtime_dir": str(index.runtime_dir), "row_count": index.row_count,
        "cluster_count": index.cluster_count,
    })

    def forbid_checkpoint_read(*args, **kwargs):
        pytest.fail("fresh schedule must not load model checkpoints")

    monkeypatch.setattr(torch, "load", forbid_checkpoint_read)
    args = prepare_module._parse_args([
        "--fresh", "--target-config", str(config_path), "--origin-updates", "8",
        "--target-updates", "40", "--world-size", "2", "--output-dir", str(tmp_path / "plan"),
        "--soft-dataset", "short_distillation", "--soft-dataset", "long_distillation",
    ])
    manifest_path = prepare_module.prepare_schedule(args)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = np.fromfile(manifest_path.parent / "global_rows.u32", dtype="<u4")
    return args, manifest_path, manifest, rows, catalogs, masses


def test_fresh_schema_one_covers_existing_cursor_without_checkpoint_identity(fresh_plan):
    _, path, manifest, rows, _, _ = fresh_plan
    info = load_soft_cluster_schedule_manifest(path)
    assert info.schema_version == 1
    assert info.origin_consumed_global_batches == 16
    assert info.target_consumed_global_batches == 80
    assert len(rows) == 128
    assert info.seed == 42 and info.gamma == 0.2
    assert info.source_checkpoint["mode"] == "fresh"
    assert info.source_checkpoint["completed_updates"] == 8
    assert info.source_checkpoint["checkpoint_step"] == 7
    for name in ("raw_checkpoint_path", "ema_checkpoint_path", "checkpoint_pair_id", "model_architecture_sha256"):
        assert info.source_checkpoint[name] is None
    assert len(manifest["dataset"]["train_sets"]) == 6


def test_fresh_data_order_is_epoch_zero_and_exposure_is_zero(fresh_plan):
    _, path, manifest, rows, catalogs, masses = fresh_plan
    legacy = _legacy_slice(catalogs, masses, start=0, count=len(rows), epoch_size=8, world_size=2, seed=42)
    offsets = np.cumsum([catalog.size for catalog in catalogs])
    actual_children = np.searchsorted(offsets, rows, side="right")
    expected_children = np.searchsorted(offsets, legacy, side="right")
    np.testing.assert_array_equal(actual_children, expected_children)
    non_soft = ~np.isin(actual_children, [1, 2])
    np.testing.assert_array_equal(rows[non_soft], legacy[non_soft])
    assert {4, 5}.issubset(set(actual_children))
    for name in ("short_distillation", "long_distillation"):
        initial = np.fromfile(path.parent / f"{name}.initial_exposure.u32", dtype="<u4")
        final = np.fromfile(path.parent / f"{name}.final_exposure.u32", dtype="<u4")
        assert np.count_nonzero(initial) == 0
        assert int(final.sum()) == manifest["provenance"]["exposure"][name]["future_draws"]


def test_fresh_stream_is_reproducible_and_independent_of_training_cursor(fresh_plan):
    args, _, _, rows, _, _ = fresh_plan
    other = copy.copy(args)
    other.origin_updates = 0
    other.target_updates = args.target_updates - args.origin_updates
    other.output_dir = args.output_dir.parent / "plan_at_zero"
    path = prepare_module.prepare_schedule(other)
    np.testing.assert_array_equal(rows, np.fromfile(path.parent / "global_rows.u32", dtype="<u4"))
    assert load_soft_cluster_schedule_manifest(path).origin_consumed_global_batches == 0


def test_existing_loader_reconstruct_and_resume_preserve_fresh_stream_cursor(fresh_plan):
    _, path, manifest, rows, _, _ = fresh_plan

    class Dataset:
        def __len__(self):
            return manifest["dataset"]["flat_row_count"]

        def __getitem__(self, index):
            return int(getattr(index, "index", index))

    def loader():
        dataset = Dataset()
        sampler = DistributedPrecomputedScheduleSampler(
            dataset, manifest_path=path, num_replicas=2, rank=0,
            epoch_size=8, gamma=0.2, seed=42, emit_sample_occurrence=True,
        )
        return DistributedDataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)

    original = loader()
    original.reconstruct_state(16)
    iterator = iter(original)
    assert [next(iterator).item() for _ in range(3)] == rows[:6:2].tolist()
    state = original.state_dict()
    assert state["consumed_global_batches"] == 19
    resumed = loader()
    resumed.load_state_dict(state)
    assert next(iter(resumed)).item() == rows[6]
    assert resumed.state_dict()["consumed_global_batches"] == 20


@pytest.mark.parametrize("field,value", [
    ("checkpoint_step", 8), ("completed_updates", 9),
    ("raw_checkpoint_path", "/tmp/model.pt"), ("model_architecture_sha256", "f" * 64),
])
def test_fresh_source_validation_keeps_cursor_and_no_model_binding_checks(fresh_plan, field, value):
    _, _, manifest, _, _, _ = fresh_plan
    source = dict(manifest["source_checkpoint"])
    source[field] = value
    with pytest.raises(SoftClusterScheduleError):
        _validate_source_checkpoint(source, sampler_origin=16)


def test_fresh_schedule_rejects_exhaustion_and_does_not_overwrite_plan(fresh_plan):
    args, path, _, _, _, _ = fresh_plan
    schedule = load_soft_cluster_schedule(path)
    with pytest.raises(SoftClusterScheduleError, match="precedes"):
        schedule.row_at(0, 0)
    with pytest.raises(SoftClusterScheduleError, match="exhausted"):
        schedule.row_at(80, 0)
    with pytest.raises(FileExistsError):
        prepare_module.prepare_schedule(args)


@pytest.mark.parametrize("option,value", [
    ("source_checkpoint", "/tmp/model.pt"), ("origin_updates", -1),
    ("target_updates", 41), ("history_start_update", 1),
])
def test_fresh_generator_rejects_ambiguous_or_out_of_range_inputs(fresh_plan, option, value):
    args, _, _, _, _, _ = fresh_plan
    changed = copy.copy(args)
    setattr(changed, option, value)
    changed.output_dir = args.output_dir.parent / "invalid"
    with pytest.raises(ValueError):
        prepare_module.prepare_schedule(changed)
    assert not changed.output_dir.exists()
