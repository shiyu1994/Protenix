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
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from protenix.data.pipeline import soft_cluster_schedule
import runner.train as train


SOURCE_TRAINING_CONTRACT = "arbitrary-source-training-contract-v1"
TARGET_TRAINING_CONTRACT = "target-soft-cluster-training-contract-v1"
SOURCE_RUN_ID = "arbitrary-source-run_20260801_000000"
ARCHITECTURE_SHA256 = "a" * 64


def _canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _contracts() -> tuple[dict, dict, dict]:
    source_payload = {
        "schema_version": 7,
        "profile_id": "four-source-v4",
        "train_sets": ["pdb", "short", "long", "disordered"],
        "train_sample_weights": [0.49, 0.005, 0.505, 0.02],
        "epoch_size": 10240,
        "sampler_seed": -1,
        "sampler_replacement": True,
        "augmentation": {"decision_version": "occurrence-v1"},
    }
    source = {**source_payload, "sha256": _canonical_hash(source_payload)}
    schedule_contract = {
        "schema_version": 1,
        "algorithm": "soft_exposure_gamma_v1",
        "gamma": 0.2,
        "layout": "wave_major_rank_v1",
        "manifest_sha256": "b" * 64,
        "schedule_sha256": "c" * 64,
        "origin_consumed_global_batches": 20,
        "schedule_global_batches": 1000,
    }
    target_payload = {
        **source_payload,
        "soft_cluster_schedule": schedule_contract,
    }
    target = {**target_payload, "sha256": _canonical_hash(target_payload)}
    return source, target, schedule_contract


def _architecture_contract() -> dict:
    return {
        "schema_version": 1,
        "test_architecture": "unchanged",
        "sha256": ARCHITECTURE_SHA256,
    }


def _configs(**overrides) -> SimpleNamespace:
    values = {
        "load_checkpoint_path": "/tmp/source/checkpoints/4.pt",
        "load_ema_checkpoint_path": "/tmp/source/checkpoints/4_ema_0.999.pt",
        "load_strict": True,
        "load_params_only": False,
        "skip_load_optimizer": False,
        "skip_load_scheduler": False,
        "skip_load_step": False,
        "load_step_for_scheduler": False,
        "distributed_strategy": "ddp",
        "optimizer_sharding": "none",
        "iters_to_accumulate": 4,
        "dtype": "bf16",
        "ema_decay": 0.999,
        "ema_shadow_device": "model",
        "data": SimpleNamespace(
            num_dl_workers=0,
            train_sampler=SimpleNamespace(
                soft_cluster_schedule_manifest="/schedule/manifest.json",
            ),
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }


def _rehash_resume_contract(contract: dict) -> None:
    material = dict(contract)
    material.pop("checkpoint_pair_id", None)
    contract["checkpoint_pair_id"] = _canonical_hash(material)


def _source_resume_contract(source_data_contract: dict) -> dict:
    architecture = _architecture_contract()
    contract = {
        "schema_version": train.STRICT_FULL_RESUME_SCHEMA_VERSION,
        "training_contract_id": SOURCE_TRAINING_CONTRACT,
        "run_id": SOURCE_RUN_ID,
        "checkpoint_pair_id": "",
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
        "data_profile_id": source_data_contract["profile_id"],
        "data_contract_sha256": source_data_contract["sha256"],
        "data_train_sets": source_data_contract["train_sets"],
        "data_train_sample_weights": source_data_contract[
            "train_sample_weights"
        ],
        "data_epoch_size": source_data_contract["epoch_size"],
        "data_sampler_seed": source_data_contract["sampler_seed"],
        "data_sampler_replacement": source_data_contract[
            "sampler_replacement"
        ],
        "data_augmentation": source_data_contract["augmentation"],
        "model_architecture_contract": architecture,
        "model_architecture_sha256": architecture["sha256"],
        "data_worker_rng_state_checkpointed": True,
        "data_resume_scope": "sampler_cursor_and_rank_main_process_rng",
    }
    _rehash_resume_contract(contract)
    return contract


def _raw_checkpoint(contract: dict) -> dict:
    return {
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
        "resume_contract": copy.deepcopy(contract),
        "rank_runtime_states": {
            "schema_version": 1,
            "world_size": 1,
            "states": [
                {
                    "rank": 0,
                    "rng": _rng_state(),
                    "train_dataloader": {
                        "schema_version": 1,
                        "sampler_epoch": 4,
                        "next_local_offset": 0,
                        "consumed_global_batches": 20,
                        "batch_size": 1,
                        "num_workers": 0,
                        "sampler": {"legacy": True},
                    },
                }
            ],
        },
    }


def _manifest_source(
    source_data_contract: dict,
    target_data_contract: dict,
    *,
    checkpoint_pair_id: str,
) -> dict:
    return {
        "mode": "checkpoint_migration",
        "raw_checkpoint_path": "/tmp/source/checkpoints/4.pt",
        "checkpoint_step": 4,
        "completed_updates": 5,
        "next_global_step": 20,
        "origin_consumed_global_batches_per_rank": 20,
        "iters_to_accumulate": 4,
        "checkpoint_pair_id": checkpoint_pair_id,
        "raw_checkpoint_sha256": "d" * 64,
        "ema_checkpoint_path": "/tmp/source/checkpoints/4_ema_0.999.pt",
        "ema_checkpoint_sha256": "e" * 64,
        "source_run_id": SOURCE_RUN_ID,
        "source_training_contract_id": SOURCE_TRAINING_CONTRACT,
        "source_data_contract_sha256": source_data_contract["sha256"],
        "model_architecture_sha256": ARCHITECTURE_SHA256,
    }


@pytest.fixture(autouse=True)
def _environment(monkeypatch):
    monkeypatch.setenv("PROTENIX_STRICT_FULL_RESUME", "1")
    monkeypatch.setenv("PROTENIX_FULL_RESUME", "1")
    monkeypatch.setenv(
        "PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION",
        train.STRICT_RESUME_SOFT_CLUSTER_SCHEDULE_MIGRATION_V1,
    )
    monkeypatch.setenv(
        "PROTENIX_TRAINING_CONTRACT_ID",
        TARGET_TRAINING_CONTRACT,
    )
    monkeypatch.setenv("PROTENIX_TRAINING_DATA_PROFILE", "four-source-v4")
    monkeypatch.setenv("PROTENIX_RESUME_CHECKPOINT_STEP", "4")
    monkeypatch.setenv(train.SOFT_CLUSTER_SCHEDULE_DIR_ENV, "/schedule")


def _patch_expectation_inputs(
    monkeypatch,
    *,
    source_data_contract: dict | None = None,
    target_data_contract: dict | None = None,
    source_checkpoint: dict | None = None,
) -> tuple[dict, dict]:
    default_source, default_target, _ = _contracts()
    source_contract = source_data_contract or default_source
    target_contract = target_data_contract or default_target
    resume_contract = _source_resume_contract(source_contract)
    checkpoint_source = source_checkpoint or _manifest_source(
        source_contract,
        target_contract,
        checkpoint_pair_id=resume_contract["checkpoint_pair_id"],
    )
    monkeypatch.setattr(
        train,
        "_load_soft_cluster_schedule_migration_manifest",
        lambda: {
            "manifest": {"source_checkpoint": copy.deepcopy(checkpoint_source)},
            "manifest_path": "/schedule/manifest.json",
            "manifest_sha256": "b" * 64,
            "schedule_contract": copy.deepcopy(
                target_contract["soft_cluster_schedule"]
            ),
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
        lambda configs: copy.deepcopy(_architecture_contract()),
    )
    monkeypatch.setattr(train.DIST_WRAPPER, "world_size", 1)
    monkeypatch.setattr(train.DIST_WRAPPER, "rank", 0)
    return resume_contract, checkpoint_source


def test_manifest_migration_reconstructs_the_source_contract(monkeypatch):
    source, target, schedule_contract = _contracts()
    _patch_expectation_inputs(
        monkeypatch,
        source_data_contract=source,
        target_data_contract=target,
    )

    migration = train._strict_resume_migration_expectations(_configs())

    assert migration is not None
    assert migration["source_data_contract_sha256"] == source["sha256"]
    assert migration["target_data_contract_sha256"] == target["sha256"]
    assert migration["soft_cluster_schedule_contract"] == schedule_contract
    assert migration["source_checkpoint_step"] == 4
    assert migration["source_next_global_step"] == 20


def test_training_data_contract_binds_only_the_schedule_subset(monkeypatch):
    _, _, schedule_contract = _contracts()
    schedule_info = object()
    monkeypatch.setattr(
        soft_cluster_schedule,
        "load_soft_cluster_schedule_manifest",
        lambda path: schedule_info,
    )
    monkeypatch.setattr(
        soft_cluster_schedule,
        "build_soft_cluster_schedule_contract",
        lambda info: copy.deepcopy(schedule_contract),
    )

    def configs(schedule_path: str) -> SimpleNamespace:
        return SimpleNamespace(
            data=SimpleNamespace(
                num_dl_workers=0,
                train_sampler=SimpleNamespace(
                    train_sample_weights=[],
                    seed=-1,
                    replacement=True,
                    soft_cluster_schedule_manifest=schedule_path,
                ),
            )
        )

    source = train.build_training_data_contract(configs(""))
    target = train.build_training_data_contract(configs("/schedule/manifest.json"))
    assert target["soft_cluster_schedule"] == schedule_contract

    reconstructed = copy.deepcopy(target)
    reconstructed.pop("sha256")
    reconstructed.pop("soft_cluster_schedule")
    assert _canonical_hash(reconstructed) == source["sha256"]


def test_environment_and_runtime_manifest_must_match(monkeypatch):
    _patch_expectation_inputs(monkeypatch)
    configs = _configs()
    configs.data.train_sampler.soft_cluster_schedule_manifest = (
        "/another/manifest.json"
    )

    with pytest.raises(ValueError, match="environment/config manifest mismatch"):
        train._strict_resume_migration_expectations(configs)


def test_manifest_migration_rejects_any_second_data_delta(monkeypatch):
    source, target, _ = _contracts()
    target = copy.deepcopy(target)
    target["hidden_sampler_change"] = True
    target_payload = dict(target)
    target_payload.pop("sha256")
    target["sha256"] = _canonical_hash(target_payload)
    source_resume = _source_resume_contract(source)
    manifest_source = _manifest_source(
        source,
        target,
        checkpoint_pair_id=source_resume["checkpoint_pair_id"],
    )
    _patch_expectation_inputs(
        monkeypatch,
        source_data_contract=source,
        target_data_contract=target,
        source_checkpoint=manifest_source,
    )

    with pytest.raises(ValueError, match="unauthorized data-contract delta"):
        train._strict_resume_migration_expectations(_configs())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_step", 3, "source checkpoint boundary mismatch"),
        ("next_global_step", 19, "source checkpoint boundary mismatch"),
        (
            "source_training_contract_id",
            TARGET_TRAINING_CONTRACT,
            "target training contract must differ",
        ),
        ("raw_checkpoint_sha256", "bad", "64-character lowercase SHA-256"),
    ],
)
def test_manifest_boundary_and_identity_fail_closed(
    monkeypatch,
    field,
    value,
    message,
):
    source, target, _ = _contracts()
    source_resume = _source_resume_contract(source)
    manifest_source = _manifest_source(
        source,
        target,
        checkpoint_pair_id=source_resume["checkpoint_pair_id"],
    )
    manifest_source[field] = value
    _patch_expectation_inputs(
        monkeypatch,
        source_data_contract=source,
        target_data_contract=target,
        source_checkpoint=manifest_source,
    )

    with pytest.raises(ValueError, match=message):
        train._strict_resume_migration_expectations(_configs())


def test_soft_migration_still_validates_full_raw_state(monkeypatch):
    source, target, _ = _contracts()
    source_resume, _ = _patch_expectation_inputs(
        monkeypatch,
        source_data_contract=source,
        target_data_contract=target,
    )
    migration = train._strict_resume_migration_expectations(_configs())
    monkeypatch.setattr(
        train,
        "_strict_resume_migration_expectations",
        lambda configs: copy.deepcopy(migration),
    )
    checkpoint = _raw_checkpoint(source_resume)

    validated = train.validate_strict_checkpoint_payload(
        checkpoint,
        checkpoint_path="/tmp/source/checkpoints/4.pt",
        expected_kind="raw",
        configs=_configs(),
        grad_scaler_enabled=False,
        device=torch.device("cpu"),
    )
    assert validated["next_global_step"] == 20

    checkpoint.pop("optimizer")
    with pytest.raises(ValueError, match="optimizer moments"):
        train.validate_strict_checkpoint_payload(
            checkpoint,
            checkpoint_path="/tmp/source/checkpoints/4.pt",
            expected_kind="raw",
            configs=_configs(),
            grad_scaler_enabled=False,
            device=torch.device("cpu"),
        )


def test_soft_migration_rejects_wrong_pair_or_sampler_boundary(monkeypatch):
    source, target, _ = _contracts()
    source_resume, _ = _patch_expectation_inputs(
        monkeypatch,
        source_data_contract=source,
        target_data_contract=target,
    )
    migration = train._strict_resume_migration_expectations(_configs())
    monkeypatch.setattr(
        train,
        "_strict_resume_migration_expectations",
        lambda configs: copy.deepcopy(migration),
    )

    wrong_pair = copy.deepcopy(migration)
    wrong_pair["source_checkpoint_pair_id"] = "f" * 64
    monkeypatch.setattr(
        train,
        "_strict_resume_migration_expectations",
        lambda configs: copy.deepcopy(wrong_pair),
    )
    with pytest.raises(ValueError, match="source checkpoint pair mismatch"):
        train.validate_strict_checkpoint_payload(
            _raw_checkpoint(source_resume),
            checkpoint_path="/tmp/source/checkpoints/4.pt",
            expected_kind="raw",
            configs=_configs(),
            grad_scaler_enabled=False,
            device=torch.device("cpu"),
        )

    wrong_boundary = copy.deepcopy(migration)
    wrong_boundary["source_next_global_step"] = 21
    monkeypatch.setattr(
        train,
        "_strict_resume_migration_expectations",
        lambda configs: copy.deepcopy(wrong_boundary),
    )
    with pytest.raises(ValueError, match="source sampler boundary mismatch"):
        train.validate_strict_checkpoint_payload(
            _raw_checkpoint(source_resume),
            checkpoint_path="/tmp/source/checkpoints/4.pt",
            expected_kind="raw",
            configs=_configs(),
            grad_scaler_enabled=False,
            device=torch.device("cpu"),
        )


def test_checkpoint_file_sha_is_bound_before_loading(tmp_path):
    checkpoint_path = tmp_path / "source.pt"
    checkpoint_path.write_bytes(b"raw-checkpoint-bytes")
    actual = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    migration = {
        "mode": train.STRICT_RESUME_SOFT_CLUSTER_SCHEDULE_MIGRATION_V1,
        "source_raw_checkpoint_sha256": actual,
        "source_ema_checkpoint_sha256": "e" * 64,
    }

    assert (
        train.validate_soft_cluster_migration_checkpoint_sha256(
            str(checkpoint_path),
            expected_kind="raw",
            migration=migration,
        )
        == actual
    )

    migration["source_raw_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        train.validate_soft_cluster_migration_checkpoint_sha256(
            str(checkpoint_path),
            expected_kind="raw",
            migration=migration,
        )


def test_schedule_manifest_environment_is_mandatory(monkeypatch):
    monkeypatch.delenv(train.SOFT_CLUSTER_SCHEDULE_DIR_ENV)

    with pytest.raises(ValueError, match=train.SOFT_CLUSTER_SCHEDULE_DIR_ENV):
        train._load_soft_cluster_schedule_migration_manifest()
