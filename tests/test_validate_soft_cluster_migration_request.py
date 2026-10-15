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
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.runtime.validate_soft_cluster_migration_request as validator


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path: Path, *, step: int = 21_999):
    run_dir = tmp_path / "source_run_v11"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    raw_path = checkpoint_dir / f"{step}.pt"
    ema_path = checkpoint_dir / f"{step}_ema_0.999.pt"
    raw_payload = b"raw-checkpoint-at-selected-boundary"
    ema_payload = b"ema-checkpoint-at-selected-boundary"
    raw_path.write_bytes(raw_payload)
    ema_path.write_bytes(ema_payload)

    dataset = {
        "train_sets": list(validator.EXPECTED_TARGET_TRAIN_SETS_V2),
        "dataset_masses": list(validator.EXPECTED_MASSES),
    }
    source = {
        "mode": "checkpoint_migration",
        "raw_checkpoint_path": str(raw_path.resolve()),
        "raw_checkpoint_sha256": _sha256(raw_payload),
        "ema_checkpoint_path": str(ema_path.resolve()),
        "ema_checkpoint_sha256": _sha256(ema_payload),
        "checkpoint_step": step,
        "completed_updates": step + 1,
        "next_global_step": (step + 1) * 4,
        "origin_consumed_global_batches_per_rank": (step + 1) * 4,
        "iters_to_accumulate": 4,
        "checkpoint_pair_id": "d" * 64,
        "source_training_contract_id": "source-contract-v11",
        "source_data_contract_sha256": validator.LOCKED_SOURCE_INVARIANTS_V2[
            "source_data_contract_sha256"
        ],
        "source_run_id": run_dir.name,
        "model_architecture_sha256": validator.LOCKED_SOURCE_INVARIANTS_V2[
            "model_architecture_sha256"
        ],
    }
    catalog = {
        "format": validator.CATALOG_REPLACEMENT_FORMAT,
        "allowed_delta": validator.CATALOG_REPLACEMENT_ALLOWED_DELTA,
        "source_dataset": {
            "binding": {
                "train_sets": list(validator.EXPECTED_TRAIN_SETS),
                "dataset_masses": list(validator.EXPECTED_MASSES),
            },
            "binding_sha256": "1" * 64,
        },
        "target_dataset": {
            "binding": dataset,
            "binding_sha256": "2" * 64,
        },
        "crosswalk": {
            "format": "protenix.global_id_history_projection.v1",
            "exposure_policy": "occurrence_sum_v1",
            "ambiguous_alternate_policy": "seen_floor_1_v1",
        },
    }
    info = SimpleNamespace(
        source_checkpoint=source,
        origin_consumed_global_batches=(step + 1) * 4,
        target_consumed_global_batches=50_000 * 4,
        schema_version=2,
        dataset=dataset,
        algorithm={
            "soft_cluster_datasets": list(
                validator.EXPECTED_TARGET_SOFT_DATASETS_V2
            )
        },
        manifest={
            "catalog_migration": catalog,
            "source_run": {
                "run_dir": str(run_dir.resolve()),
                "run_name": run_dir.name,
            },
        },
        manifest_path=tmp_path / "schedule" / "manifest.json",
        manifest_sha256="3" * 64,
        row_count=1,
        schedule_sha256="4" * 64,
        source_checkpoint_binding_sha256="5" * 64,
        gamma=0.2,
    )
    args = argparse.Namespace(
        schedule=tmp_path / "schedule",
        source_run_dir=run_dir,
        source_step=step,
        source_raw_checkpoint=raw_path,
        source_ema_checkpoint=ema_path,
        target_updates=50_000,
        migration_mode="catalog-replacement-v2",
    )
    return args, info


def _patch_request(monkeypatch, args, info):
    monkeypatch.setattr(validator, "_parse_args", lambda: args)

    def load(schedule, *, expected):
        assert schedule == args.schedule
        assert expected["source_checkpoint"] == {
            "checkpoint_step": args.source_step,
            "raw_checkpoint_path": str(args.source_raw_checkpoint.resolve()),
            "ema_checkpoint_path": str(args.source_ema_checkpoint.resolve()),
        }
        return info

    monkeypatch.setattr(validator, "load_soft_cluster_schedule_manifest", load)


def test_accepts_schedule_bound_step21999_and_hashes_checkpoint_pair(
    tmp_path: Path, monkeypatch, capsys
):
    args, info = _fixture(tmp_path)
    _patch_request(monkeypatch, args, info)

    assert validator.main() == 0

    output = json.loads(capsys.readouterr().out)
    assert output["source_checkpoint_step"] == 21999
    assert output["source_checkpoint_pair_id"] == "d" * 64
    assert output["source_completed_updates"] == 22000
    assert output["source_checkpoint_binding_sha256"] == "5" * 64
    assert output["source_checkpoint_hashes"] == {
        "raw_checkpoint_sha256": info.source_checkpoint[
            "raw_checkpoint_sha256"
        ],
        "ema_checkpoint_sha256": info.source_checkpoint[
            "ema_checkpoint_sha256"
        ],
    }


def test_rejects_checkpoint_bytes_outside_validated_schedule_binding(
    tmp_path: Path, monkeypatch
):
    args, info = _fixture(tmp_path)
    info.source_checkpoint["raw_checkpoint_sha256"] = "0" * 64
    _patch_request(monkeypatch, args, info)

    with pytest.raises(ValueError, match="validated schedule binding"):
        validator.main()


@pytest.mark.parametrize(
    "field",
    ["source_data_contract_sha256", "model_architecture_sha256"],
)
def test_rejects_source_training_invariant_drift(
    tmp_path: Path, monkeypatch, field: str
):
    args, info = _fixture(tmp_path)
    info.source_checkpoint[field] = "0" * 64
    _patch_request(monkeypatch, args, info)

    with pytest.raises(ValueError, match=f"source invariant mismatch for {field}"):
        validator.main()


def _same_schedule_continuation_fixture(tmp_path: Path):
    step = validator.MSA_INDEPENDENT_SOURCE_STEP
    run_dir = tmp_path / validator.MSA_INDEPENDENT_SOURCE_RUN_ID
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    raw_path = checkpoint_dir / f"{step}.pt"
    ema_path = checkpoint_dir / f"{step}_ema_0.999.pt"
    raw_path.write_bytes(b"locked-step-27799-raw-test-double")
    ema_path.write_bytes(b"locked-step-27799-ema-test-double")

    dataset = {
        "train_sets": list(validator.EXPECTED_TARGET_TRAIN_SETS_V2),
        "dataset_masses": list(validator.EXPECTED_MASSES),
    }
    anchor_source = {
        "mode": "checkpoint_migration",
        "raw_checkpoint_path": "/locked/parent/checkpoints/21999.pt",
        "raw_checkpoint_sha256": "1" * 64,
        "ema_checkpoint_path": (
            "/locked/parent/checkpoints/21999_ema_0.999.pt"
        ),
        "ema_checkpoint_sha256": "2" * 64,
        "checkpoint_step": 21_999,
        "completed_updates": 22_000,
        "next_global_step": 88_000,
        "origin_consumed_global_batches_per_rank": 88_000,
        "iters_to_accumulate": 4,
        "checkpoint_pair_id": "3" * 64,
        "source_training_contract_id": "parent-contract",
        "source_data_contract_sha256": validator.LOCKED_SOURCE_INVARIANTS_V2[
            "source_data_contract_sha256"
        ],
        "source_run_id": validator.SAME_SCHEDULE_PARENT_RUN_ID,
        "model_architecture_sha256": validator.LOCKED_SOURCE_INVARIANTS_V2[
            "model_architecture_sha256"
        ],
    }
    catalog = {
        "format": validator.CATALOG_REPLACEMENT_FORMAT,
        "allowed_delta": validator.CATALOG_REPLACEMENT_ALLOWED_DELTA,
        "source_dataset": {
            "binding": {
                "train_sets": list(validator.EXPECTED_TRAIN_SETS),
                "dataset_masses": list(validator.EXPECTED_MASSES),
            },
            "binding_sha256": "4" * 64,
        },
        "target_dataset": {
            "binding": dataset,
            "binding_sha256": (
                validator.SAME_SCHEDULE_PARENT_DATASET_BINDING_SHA256
            ),
        },
        "crosswalk": {
            "format": "protenix.global_id_history_projection.v1",
            "exposure_policy": "occurrence_sum_v1",
            "ambiguous_alternate_policy": "seen_floor_1_v1",
        },
    }
    info = SimpleNamespace(
        source_checkpoint=anchor_source,
        origin_consumed_global_batches=88_000,
        target_consumed_global_batches=200_000,
        schema_version=2,
        dataset=dataset,
        algorithm={
            "soft_cluster_datasets": list(
                validator.EXPECTED_TARGET_SOFT_DATASETS_V2
            )
        },
        manifest={
            "catalog_migration": catalog,
            "source_run": {
                "run_dir": f"/locked/{validator.SAME_SCHEDULE_PARENT_RUN_ID}",
                "run_name": validator.SAME_SCHEDULE_PARENT_RUN_ID,
            },
        },
        manifest_path=tmp_path / "schedule" / "manifest.json",
        manifest_sha256=validator.SAME_SCHEDULE_PARENT_MANIFEST_SHA256,
        row_count=(200_000 - 88_000) * 128,
        schedule_sha256=validator.SAME_SCHEDULE_PARENT_ROWS_SHA256,
        source_checkpoint_binding_sha256=(
            validator.SAME_SCHEDULE_PARENT_SOURCE_BINDING_SHA256
        ),
        gamma=0.2,
    )
    args = argparse.Namespace(
        schedule=tmp_path / "schedule",
        source_run_dir=run_dir,
        source_step=step,
        source_raw_checkpoint=raw_path,
        source_ema_checkpoint=ema_path,
        target_updates=50_000,
        migration_mode=validator.SAME_SCHEDULE_CONTINUATION_MIGRATION_MODE,
    )
    return args, info


def _patch_same_schedule_request(monkeypatch, args, info):
    monkeypatch.setattr(validator, "_parse_args", lambda: args)

    def load(schedule, *, expected):
        assert schedule == args.schedule
        assert expected == {
            "world_size": 128,
            "epoch_size": 10_240,
            "gamma": 0.2,
            "seed": 42,
        }
        return info

    monkeypatch.setattr(validator, "load_soft_cluster_schedule_manifest", load)


def _patch_locked_step27799_hashes(monkeypatch, args):
    def fake_hash(path: Path) -> str:
        if path == args.source_raw_checkpoint.resolve():
            return validator.MSA_INDEPENDENT_SOURCE_RAW_SHA256
        if path == args.source_ema_checkpoint.resolve():
            return validator.MSA_INDEPENDENT_SOURCE_EMA_SHA256
        raise AssertionError(f"unexpected hash request: {path}")

    monkeypatch.setattr(validator, "_sha256_stable_file", fake_hash)


def test_accepts_step27799_continuation_on_unchanged_parent_schedule(
    tmp_path: Path, monkeypatch, capsys
):
    args, info = _same_schedule_continuation_fixture(tmp_path)
    _patch_same_schedule_request(monkeypatch, args, info)
    _patch_locked_step27799_hashes(monkeypatch, args)

    assert validator.main() == 0

    output = json.loads(capsys.readouterr().out)
    assert output["migration_mode"] == "same-schedule-continuation-v1"
    assert output["source_run_id"] == validator.MSA_INDEPENDENT_SOURCE_RUN_ID
    assert output["source_checkpoint_step"] == 27_799
    assert output["source_checkpoint_pair_id"] == (
        validator.MSA_INDEPENDENT_SOURCE_PAIR_ID
    )
    assert output["source_completed_updates"] == 27_800
    assert output["origin_consumed_global_batches_per_rank"] == 88_000
    assert output["selected_consumed_global_batches_per_rank"] == 111_200
    assert output["source_checkpoint_hashes"] == {
        "raw_checkpoint_sha256": validator.MSA_INDEPENDENT_SOURCE_RAW_SHA256,
        "ema_checkpoint_sha256": validator.MSA_INDEPENDENT_SOURCE_EMA_SHA256,
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("manifest_sha256", "0" * 64),
        ("schedule_sha256", "0" * 64),
        ("source_checkpoint_binding_sha256", "0" * 64),
    ],
)
def test_rejects_same_schedule_identity_drift(
    tmp_path: Path, monkeypatch, field: str, replacement: str
):
    args, info = _same_schedule_continuation_fixture(tmp_path)
    setattr(info, field, replacement)
    _patch_same_schedule_request(monkeypatch, args, info)
    _patch_locked_step27799_hashes(monkeypatch, args)

    with pytest.raises(ValueError, match=f"{field} mismatch"):
        validator.main()


def test_rejects_step27799_continuation_checkpoint_hash_drift(
    tmp_path: Path, monkeypatch
):
    args, info = _same_schedule_continuation_fixture(tmp_path)
    _patch_same_schedule_request(monkeypatch, args, info)
    monkeypatch.setattr(validator, "_sha256_stable_file", lambda path: "0" * 64)

    with pytest.raises(ValueError, match="locked step-27799 boundary"):
        validator.main()
