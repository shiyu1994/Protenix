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

import runner.train as train


MIGRATION_MODE = train.STRICT_RESUME_MSA_QUERY_PIN_MIGRATION_V1
MIGRATION = train.STRICT_RESUME_CONTRACT_MIGRATIONS[MIGRATION_MODE]
SOURCE_DATA_SHA256 = (
    "cffda1e6694a9f293c8ed213d1b33e188d828d10a4d0eebdf66b2f428e75756c"
)
TARGET_DATA_SHA256 = (
    "6c98c5cab75197b7e4d41b7ce5287af65ca3f750a707ec69a1bdaf29ed24b8de"
)
ARCHITECTURE_SHA256 = (
    "cd1cbb658e352f0d56a2dcd65c5e0e95ae2e29c321f4fd0da9f79c6145edc34c"
)
REAL_SHA256 = hashlib.sha256
DATASET_NAME = "migration_test_dataset"


class _FixedDigest:
    def __init__(self, digest: str):
        self._digest = digest

    def hexdigest(self) -> str:
        return self._digest


class _Scheduler:
    def __init__(self, last_epoch: int):
        self.last_epoch = last_epoch

    def state_dict(self) -> dict[str, int]:
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.last_epoch = int(state_dict["last_epoch"])


class _RuntimeLoader:
    def __init__(self) -> None:
        self.loaded_state = None

    def load_state_dict(self, state_dict) -> None:
        self.loaded_state = copy.deepcopy(state_dict)


class _EMAProbe:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.shadow = None

    def register(self) -> None:
        self.shadow = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }


def _query_policy() -> dict:
    return {
        "policy_version": train.MODEL_MSA_QUERY_INCLUSION_POLICY_VERSION,
        "query_row_index": 0,
        "max_sample_size": {"train": 1024, "test": 1024},
        "depth_accounting": train.MODEL_MSA_QUERY_INCLUSION_DEPTH_ACCOUNTING,
        "above_threshold": train.MODEL_MSA_QUERY_INCLUSION_ABOVE_THRESHOLD,
    }


def _target_data_contract() -> dict:
    return {
        "schema_version": 7,
        "profile_id": "weighted_pdb_afdb_mgnify_short_long_disordered_v4",
        "train_sets": [DATASET_NAME],
        "train_sample_weights": [1.0],
        "epoch_size": 10240,
        "sampler_seed": -1,
        "sampler_replacement": True,
        "augmentation": {
            "decision_version": "sample_occurrence_sha256_v1",
            "model_msa_query_inclusion": _query_policy(),
        },
        "dataset_runtime_semantics": {
            "effective_by_dataset": {
                DATASET_NAME: {
                    "effective_msa": {
                        "query_inclusion_max_sample_size": {
                            "train": 1024,
                            "test": 1024,
                        }
                    }
                }
            }
        },
        "sha256": TARGET_DATA_SHA256,
    }


def _source_payload_bytes(target_data_contract: dict) -> bytes:
    source_payload = copy.deepcopy(target_data_contract)
    source_payload.pop("sha256")
    source_payload["augmentation"].pop("model_msa_query_inclusion")
    for runtime_semantics in source_payload["dataset_runtime_semantics"][
        "effective_by_dataset"
    ].values():
        runtime_semantics["effective_msa"].pop(
            "query_inclusion_max_sample_size"
        )
    return json.dumps(
        source_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _architecture_contract() -> dict:
    return {
        "schema_version": 1,
        "test_architecture": "linear_2_to_1",
        "sha256": ARCHITECTURE_SHA256,
    }


def _configs(**overrides) -> SimpleNamespace:
    values = {
        "load_checkpoint_path": "/tmp/source/checkpoints/4.pt",
        "load_ema_checkpoint_path": (
            "/tmp/source/checkpoints/4_ema_0.999.pt"
        ),
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
        "data": SimpleNamespace(num_dl_workers=0),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _migration_expectations() -> dict:
    target_data_contract = _target_data_contract()
    source_augmentation = copy.deepcopy(
        target_data_contract["augmentation"]
    )
    source_augmentation.pop("model_msa_query_inclusion")
    return {
        "mode": MIGRATION_MODE,
        **MIGRATION,
        "target_data_contract": target_data_contract,
        "target_model_architecture_sha256": ARCHITECTURE_SHA256,
        "expected_query_policy": _query_policy(),
        "source_augmentation": source_augmentation,
        "reconstructed_source_data_contract_sha256": SOURCE_DATA_SHA256,
    }


def _patch_checkpoint_expectations(monkeypatch) -> None:
    expectations = _migration_expectations()
    monkeypatch.setattr(
        train,
        "_strict_resume_migration_expectations",
        lambda configs: copy.deepcopy(expectations),
    )
    monkeypatch.setattr(
        train,
        "build_model_architecture_contract",
        lambda configs: copy.deepcopy(_architecture_contract()),
    )
    monkeypatch.setattr(train.DIST_WRAPPER, "world_size", 1)
    monkeypatch.setattr(train.DIST_WRAPPER, "rank", 0)


def _rehash_contract(contract: dict) -> None:
    material = dict(contract)
    material.pop("checkpoint_pair_id", None)
    contract["checkpoint_pair_id"] = REAL_SHA256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _rng_state() -> dict:
    original_python = random.getstate()
    original_numpy = np.random.get_state()
    original_torch = torch.get_rng_state()
    try:
        random.seed(1234)
        np.random.seed(5678)
        torch.manual_seed(9012)
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": None,
        }
    finally:
        random.setstate(original_python)
        np.random.set_state(original_numpy)
        torch.set_rng_state(original_torch)


def _rank_runtime_states(rng_state: dict | None = None) -> dict:
    return {
        "schema_version": 1,
        "world_size": 1,
        "states": [
            {
                "rank": 0,
                "rng": copy.deepcopy(rng_state or _rng_state()),
                "train_dataloader": {
                    "schema_version": 1,
                    "sampler_epoch": 4,
                    "next_local_offset": 0,
                    "consumed_global_batches": 20,
                    "batch_size": 1,
                    "num_workers": 0,
                    "sampler": {"test_state": True},
                },
            }
        ],
    }


def _source_contract() -> dict:
    target_data_contract = _target_data_contract()
    source_augmentation = copy.deepcopy(
        target_data_contract["augmentation"]
    )
    source_augmentation.pop("model_msa_query_inclusion")
    contract = {
        "schema_version": train.STRICT_FULL_RESUME_SCHEMA_VERSION,
        "training_contract_id": MIGRATION["source_training_contract_id"],
        "run_id": MIGRATION["source_run_id"],
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
        "data_profile_id": target_data_contract["profile_id"],
        "data_contract_sha256": MIGRATION["source_data_contract_sha256"],
        "data_train_sets": target_data_contract["train_sets"],
        "data_train_sample_weights": target_data_contract[
            "train_sample_weights"
        ],
        "data_epoch_size": target_data_contract["epoch_size"],
        "data_sampler_seed": target_data_contract["sampler_seed"],
        "data_sampler_replacement": target_data_contract[
            "sampler_replacement"
        ],
        "data_augmentation": source_augmentation,
        "model_architecture_contract": _architecture_contract(),
        "model_architecture_sha256": ARCHITECTURE_SHA256,
        "data_worker_rng_state_checkpointed": True,
        "data_resume_scope": "sampler_cursor_and_rank_main_process_rng",
    }
    _rehash_contract(contract)
    return contract


def _raw_checkpoint(
    *,
    contract: dict | None = None,
    model_state: dict | None = None,
    optimizer_state: dict | None = None,
    rng_state: dict | None = None,
) -> dict:
    return {
        "format_version": 2,
        "checkpoint_kind": "raw",
        "model": model_state or {"weight": torch.ones(1)},
        "optimizer": optimizer_state
        or {
            "state": {0: {"exp_avg": torch.ones(1)}},
            "param_groups": [{"params": [0]}],
        },
        "scheduler": {"last_epoch": 7},
        "grad_scaler": {},
        "step": 4,
        "resume_contract": copy.deepcopy(contract or _source_contract()),
        "rank_runtime_states": _rank_runtime_states(rng_state),
    }


def _ema_checkpoint(
    *,
    contract: dict | None = None,
    model_state: dict | None = None,
) -> dict:
    return {
        "format_version": 2,
        "checkpoint_kind": "ema",
        "model": model_state or {"weight": torch.ones(1)},
        "optimizer": None,
        "scheduler": None,
        "grad_scaler": None,
        "step": 4,
        "resume_contract": copy.deepcopy(contract or _source_contract()),
    }


@pytest.fixture(autouse=True)
def _migration_environment(monkeypatch):
    monkeypatch.setenv("PROTENIX_STRICT_FULL_RESUME", "1")
    monkeypatch.setenv("PROTENIX_FULL_RESUME", "1")
    monkeypatch.setenv(
        "PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION",
        MIGRATION_MODE,
    )
    monkeypatch.setenv(
        "PROTENIX_TRAINING_CONTRACT_ID",
        MIGRATION["target_training_contract_id"],
    )
    monkeypatch.setenv(
        "PROTENIX_TRAINING_DATA_PROFILE",
        "weighted_pdb_afdb_mgnify_short_long_disordered_v4",
    )
    monkeypatch.setenv("PROTENIX_RESUME_CHECKPOINT_STEP", "4")
    monkeypatch.delenv("PROTENIX_INITIALIZATION_MODE", raising=False)


def _patch_target_builders(
    monkeypatch,
    *,
    target_data_contract: dict | None = None,
    architecture_contract: dict | None = None,
) -> dict:
    target = copy.deepcopy(target_data_contract or _target_data_contract())
    # Freeze the digest override to the one authorized legacy payload.  A
    # caller-supplied target mutation must therefore fail the downgrade-hash
    # proof instead of teaching the mock to accept the mutation.
    source_bytes = _source_payload_bytes(_target_data_contract())

    def sha256(data=b""):
        if data == source_bytes:
            return _FixedDigest(SOURCE_DATA_SHA256)
        return REAL_SHA256(data)

    monkeypatch.setattr(train.hashlib, "sha256", sha256)
    monkeypatch.setattr(
        train,
        "build_training_data_contract",
        lambda configs: copy.deepcopy(target),
    )
    monkeypatch.setattr(
        train,
        "build_model_architecture_contract",
        lambda configs: copy.deepcopy(
            architecture_contract or _architecture_contract()
        ),
    )
    return target


def test_registry_is_frozen_to_the_audited_v10_and_v11_contracts():
    assert set(train.STRICT_RESUME_CONTRACT_MIGRATIONS) == {MIGRATION_MODE}
    assert MIGRATION["source_data_contract_sha256"] == SOURCE_DATA_SHA256
    assert MIGRATION["target_data_contract_sha256"] == TARGET_DATA_SHA256
    assert (
        MIGRATION["source_model_architecture_sha256"]
        == ARCHITECTURE_SHA256
    )
    assert MIGRATION["source_training_contract_id"].endswith(
        "deferredhealth_v10"
    )
    assert "msaquerypinle1024" in MIGRATION["target_training_contract_id"]
    assert MIGRATION["target_training_contract_id"].endswith(
        "deferredhealth_v11"
    )


def test_exact_target_contract_resolves_the_fixed_migration(monkeypatch):
    _patch_target_builders(monkeypatch)

    resolved = train._strict_resume_migration_expectations(_configs())

    assert resolved is not None
    assert resolved["mode"] == MIGRATION_MODE
    assert resolved["target_data_contract_sha256"] == TARGET_DATA_SHA256
    assert (
        resolved["reconstructed_source_data_contract_sha256"]
        == SOURCE_DATA_SHA256
    )
    assert resolved["expected_query_policy"] == _query_policy()


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("target_contract", "target training contract mismatch"),
        ("target_hash", "target data contract mismatch"),
        ("target_architecture", "identical source and target model architecture"),
        ("target_query_policy", "exact train/test 1024 query-inclusion policy"),
    ],
)
def test_migration_rejects_wrong_target(monkeypatch, fault, message):
    target = _target_data_contract()
    architecture = _architecture_contract()
    if fault == "target_contract":
        monkeypatch.setenv("PROTENIX_TRAINING_CONTRACT_ID", "wrong-v11")
    elif fault == "target_hash":
        target["sha256"] = "0" * 64
    elif fault == "target_architecture":
        architecture["sha256"] = "1" * 64
    elif fault == "target_query_policy":
        target["augmentation"]["model_msa_query_inclusion"][
            "max_sample_size"
        ]["train"] = 1023
    _patch_target_builders(
        monkeypatch,
        target_data_contract=target,
        architecture_contract=architecture,
    )

    with pytest.raises(ValueError, match=message):
        train._strict_resume_migration_expectations(_configs())


def test_migration_rejects_hidden_target_data_delta(monkeypatch):
    target = _target_data_contract()
    target["unauthorized_training_policy"] = True
    _patch_target_builders(monkeypatch, target_data_contract=target)

    with pytest.raises(ValueError, match="unauthorized data-contract delta"):
        train._strict_resume_migration_expectations(_configs())


def test_exact_source_raw_and_ema_contracts_validate_as_a_pair(monkeypatch):
    _patch_checkpoint_expectations(monkeypatch)
    contract = _source_contract()
    raw_contract = train.validate_strict_checkpoint_payload(
        _raw_checkpoint(contract=contract),
        checkpoint_path="/tmp/4.pt",
        expected_kind="raw",
        configs=_configs(),
        grad_scaler_enabled=False,
        device=torch.device("cpu"),
    )
    ema_contract = train.validate_strict_checkpoint_payload(
        _ema_checkpoint(contract=contract),
        checkpoint_path="/tmp/4_ema_0.999.pt",
        expected_kind="ema",
        configs=_configs(),
        grad_scaler_enabled=False,
        device=torch.device("cpu"),
    )

    train.validate_matching_resume_contracts(raw_contract, ema_contract)
    assert raw_contract["training_contract_id"] == MIGRATION[
        "source_training_contract_id"
    ]
    assert raw_contract["data_contract_sha256"] == SOURCE_DATA_SHA256


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run_id", "wrong-source-run", "source run mismatch"),
        ("data_contract_sha256", "2" * 64, "data contract mismatch"),
        ("training_contract_id", "wrong-v10", "training contract mismatch"),
    ],
)
def test_migration_rejects_wrong_source_contract(
    monkeypatch,
    field,
    value,
    message,
):
    _patch_checkpoint_expectations(monkeypatch)
    contract = _source_contract()
    contract[field] = value
    _rehash_contract(contract)

    with pytest.raises(ValueError, match=message):
        train.validate_strict_checkpoint_payload(
            _raw_checkpoint(contract=contract),
            checkpoint_path="/tmp/4.pt",
            expected_kind="raw",
            configs=_configs(),
            grad_scaler_enabled=False,
            device=torch.device("cpu"),
        )


def test_migration_rejects_tampered_source_pair_id(monkeypatch):
    _patch_checkpoint_expectations(monkeypatch)
    contract = _source_contract()
    contract["checkpoint_pair_id"] = "f" * 64

    with pytest.raises(ValueError, match="checkpoint_pair_id does not match"):
        train.validate_strict_checkpoint_payload(
            _raw_checkpoint(contract=contract),
            checkpoint_path="/tmp/4.pt",
            expected_kind="raw",
            configs=_configs(),
            grad_scaler_enabled=False,
            device=torch.device("cpu"),
        )


def test_cpu_full_state_migration_restores_raw_ema_optimizer_and_runtime(
    tmp_path,
    monkeypatch,
):
    _patch_checkpoint_expectations(monkeypatch)

    source_model = torch.nn.Linear(2, 1)
    source_optimizer = torch.optim.Adam(source_model.parameters(), lr=1e-3)
    source_model(torch.ones(1, 2)).sum().backward()
    source_optimizer.step()
    source_optimizer.zero_grad(set_to_none=True)
    raw_model_state = copy.deepcopy(source_model.state_dict())
    optimizer_state = copy.deepcopy(source_optimizer.state_dict())
    ema_model_state = {
        name: value.detach().clone() + 0.25
        for name, value in raw_model_state.items()
    }
    saved_rng_state = _rng_state()
    contract = _source_contract()

    raw_path = tmp_path / "4.pt"
    ema_path = tmp_path / "4_ema_0.999.pt"
    torch.save(
        _raw_checkpoint(
            contract=contract,
            model_state=raw_model_state,
            optimizer_state=optimizer_state,
            rng_state=saved_rng_state,
        ),
        raw_path,
    )
    torch.save(
        _ema_checkpoint(
            contract=contract,
            model_state=ema_model_state,
        ),
        ema_path,
    )

    python_rng = random.Random()
    python_rng.setstate(saved_rng_state["python"])
    expected_python = python_rng.random()
    numpy_rng = np.random.RandomState()
    numpy_rng.set_state(saved_rng_state["numpy"])
    expected_numpy = numpy_rng.rand()
    torch_rng = torch.Generator()
    torch_rng.set_state(saved_rng_state["torch"])
    expected_torch = torch.rand((), generator=torch_rng)

    random.seed(91)
    np.random.seed(92)
    torch.manual_seed(93)
    restored = train.AF3Trainer.__new__(train.AF3Trainer)
    restored.model = torch.nn.Linear(2, 1)
    restored.optimizer = torch.optim.Adam(
        restored.model.parameters(),
        lr=1e-3,
    )
    restored.lr_scheduler = _Scheduler(last_epoch=-1)
    restored.grad_scaler = torch.GradScaler(device="cpu", enabled=False)
    restored.configs = _configs(
        load_checkpoint_path=str(raw_path),
        load_ema_checkpoint_path=str(ema_path),
    )
    restored.device = torch.device("cpu")
    restored.step = 0
    restored.global_step = 0
    restored.start_step = 0
    restored.iters_to_accumulate = 4
    restored.train_dl = _RuntimeLoader()
    restored.use_fsdp = False
    restored.use_ddp = False
    restored.ema_wrapper = _EMAProbe(restored.model)
    restored.run_manifest_path = ""
    restored.print = lambda message: None

    restored.try_load_checkpoint()

    assert restored.step == 5
    assert restored.start_step == 5
    assert restored.global_step == 20
    assert restored.lr_scheduler.last_epoch == 7
    assert restored.optimizer.state_dict()["state"]
    assert restored.train_dl.loaded_state is not None
    assert (
        restored.train_dl.loaded_state["consumed_global_batches"] == 20
    )
    for name, expected in raw_model_state.items():
        assert torch.equal(restored.model.state_dict()[name], expected)
    assert restored.ema_wrapper.shadow is not None
    for name, expected in ema_model_state.items():
        assert torch.equal(restored.ema_wrapper.shadow[name], expected)
    assert random.random() == expected_python
    assert np.random.rand() == expected_numpy
    assert torch.equal(torch.rand(()), expected_torch)
