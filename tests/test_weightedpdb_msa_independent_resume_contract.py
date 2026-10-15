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

import pytest

import runner.train as train


PDB_DATASET = train.WEIGHTEDPDB_MSA_INDEPENDENT_DATASET
OTHER_DATASET = "msa_independent_contract_test_distillation"
TEST_ARCHITECTURE_SHA256 = "a" * 64


def _canonical_contract_sha256(contract: dict) -> str:
    payload = copy.deepcopy(contract)
    payload.pop("sha256", None)
    return train._canonical_mapping_sha256(payload)


def _target_data_contract() -> dict:
    payload = {
        "schema_version": 7,
        "profile_id": train.WEIGHTEDPDB_MSA_INDEPENDENT_DATA_PROFILE,
        "train_sets": [PDB_DATASET, OTHER_DATASET],
        "train_sample_weights": [0.49, 0.51],
        "epoch_size": 10240,
        "sampler_seed": 42,
        "sampler_replacement": True,
        "augmentation": {
            "decision_version": "sample_occurrence_sha256_v1",
        },
        "soft_cluster_schedule": {
            "format": "protenix.soft_cluster_schedule.v2",
            "sha256": "b" * 64,
        },
        "dataset_runtime_semantics": {
            "effective_by_dataset": {
                PDB_DATASET: {
                    "stage": "train",
                    "effective_msa": {
                        "dataset_name": PDB_DATASET,
                        "prot_pairing_dbs": ["pairing"],
                        "prot_non_pairing_dbs": [
                            "pairing-non_pairing"
                        ],
                        "max_input_paired_msa_depth": 10000,
                        "max_input_unpaired_msa_depth": 10000,
                        "max_size": {"train": 16384, "test": 16384},
                        "sample_cutoff": {"train": 2048, "test": 2048},
                        "independent_pairing_non_pairing_budget": True,
                    },
                    "effective_template": {"dataset_name": PDB_DATASET},
                },
                OTHER_DATASET: {
                    "stage": "train",
                    "effective_msa": {
                        "dataset_name": OTHER_DATASET,
                        "prot_pairing_dbs": [""],
                        "prot_non_pairing_dbs": ["non_pairing"],
                        "max_input_unpaired_msa_depth": 16384,
                        "max_size": {"train": 16384, "test": 16384},
                    },
                    "effective_template": {"dataset_name": OTHER_DATASET},
                },
            }
        },
    }
    payload["sha256"] = _canonical_contract_sha256(payload)
    return payload


def _source_data_sha256(target: dict) -> str:
    source = copy.deepcopy(target)
    source.pop("sha256", None)
    effective_by_dataset = source["dataset_runtime_semantics"][
        "effective_by_dataset"
    ]
    for semantics in effective_by_dataset.values():
        semantics["effective_msa"].pop(
            "independent_pairing_non_pairing_budget",
            None,
        )
    return train._canonical_mapping_sha256(source)


def _bind_fixture_contract(
    monkeypatch: pytest.MonkeyPatch,
    target: dict,
    *,
    source_reference: dict | None = None,
) -> None:
    source_reference = source_reference or _target_data_contract()
    monkeypatch.setattr(
        train,
        "WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_DATA_CONTRACT_SHA256",
        _source_data_sha256(source_reference),
    )
    monkeypatch.setattr(
        train,
        "WEIGHTEDPDB_MSA_INDEPENDENT_TARGET_DATA_CONTRACT_SHA256",
        target["sha256"],
    )
    monkeypatch.setattr(
        train,
        "WEIGHTEDPDB_MSA_INDEPENDENT_MODEL_ARCHITECTURE_SHA256",
        TEST_ARCHITECTURE_SHA256,
    )
    monkeypatch.setattr(
        train,
        "build_training_data_contract",
        lambda configs: copy.deepcopy(target),
    )
    monkeypatch.setattr(
        train,
        "build_model_architecture_contract",
        lambda configs: {"sha256": TEST_ARCHITECTURE_SHA256},
    )
    monkeypatch.setenv(
        "PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION",
        train.STRICT_RESUME_WEIGHTEDPDB_MSA_INDEPENDENT_MIGRATION_V1,
    )
    monkeypatch.setenv(
        "PROTENIX_TRAINING_CONTRACT_ID",
        train.WEIGHTEDPDB_MSA_INDEPENDENT_TARGET_TRAINING_CONTRACT_ID,
    )


def test_production_step27799_msa_only_constants_are_frozen():
    assert train.STRICT_RESUME_WEIGHTEDPDB_MSA_INDEPENDENT_MIGRATION_V1 == (
        "weightedpdb_independent_pairing_nonpairing_full_state_v1"
    )
    assert train.WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_STEP == 27799
    assert train.WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_NEXT_GLOBAL_STEP == 111200
    assert train.WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_PAIR_ID == (
        "d40d64151cc8a8ff17458c310ca0a53f40047d3b5d4adf5577db3d653ae1c8f2"
    )
    assert train.WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_DATA_CONTRACT_SHA256 == (
        "bbb8d9b0a987e2aed17743df24024daa832d7b66d05ab45783a11a0f03a3f42c"
    )
    assert train.WEIGHTEDPDB_MSA_INDEPENDENT_TARGET_DATA_CONTRACT_SHA256 == (
        "ccf4fcfd799521fbc7cd0be1c6cc0eea744cf18fe87ab906c24bc485486dce00"
    )
    assert train.WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_RAW_SHA256 == (
        "4cefb9af5088a4671b4287149c95efcd72cac37449136fe37b4e4d060754f94a"
    )
    assert train.WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_EMA_SHA256 == (
        "b3ad325bde97f2b249a03c4004a2a00d704e92a3aa0be925b0705e1a23ee67c8"
    )


def test_valid_target_resolves_msa_only_expectations(monkeypatch):
    target = _target_data_contract()
    _bind_fixture_contract(monkeypatch, target)

    migration = train._strict_resume_migration_expectations(object())

    assert migration["mode"] == (
        train.STRICT_RESUME_WEIGHTEDPDB_MSA_INDEPENDENT_MIGRATION_V1
    )
    assert migration["mode"] not in train.SOFT_CLUSTER_MIGRATION_MODES
    assert migration["mode"] in train.CHECKPOINT_BOUND_MIGRATION_MODES
    assert migration["target_data_contract_sha256"] == target["sha256"]
    assert migration["reconstructed_source_data_contract_sha256"] == (
        _source_data_sha256(target)
    )
    assert migration["source_checkpoint_step"] == 27799
    assert migration["source_next_global_step"] == 111200
    assert migration["weightedpdb_msa_policy"] == {
        "policy_version": train.PAIRING_NON_PAIRING_BUDGET_POLICY_VERSION,
        "dataset": PDB_DATASET,
        "paired_source": "pairing.a3m",
        "genuine_non_pairing_source": "non_pairing.a3m",
        "paired_source_raw_depth": 10000,
        "genuine_non_pairing_raw_depth": 10000,
        "final_assembled_depth": 16384,
        "paired_block_deduplicate": False,
        "unpaired_merge_deduplicate": True,
        "inference_policy_changed": False,
    }

    lowered_keys = {key.lower() for key in migration}
    assert not any("rewarm" in key or "cosine" in key for key in lowered_keys)
    assert "target_scheduler_state" not in migration
    assert migration["source_scheduler_step"] == 27800
    assert migration["source_scheduler_state"]["_last_lr"] == [0.00095]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("flag_disabled", "requires independent_pairing"),
        ("other_dataset_enabled", "active only for weightedPDB"),
        ("paired_10k_drift", "max_input_paired_msa_depth"),
        ("unpaired_10k_drift", "max_input_unpaired_msa_depth"),
        ("final_16384_drift", "max_size.train=16384"),
    ],
)
def test_msa_policy_drift_fails_closed(monkeypatch, case, message):
    reference = _target_data_contract()
    target = copy.deepcopy(reference)
    effective = target["dataset_runtime_semantics"]["effective_by_dataset"]
    weighted_msa = effective[PDB_DATASET]["effective_msa"]
    if case == "flag_disabled":
        weighted_msa["independent_pairing_non_pairing_budget"] = False
    elif case == "other_dataset_enabled":
        effective[OTHER_DATASET]["effective_msa"][
            "independent_pairing_non_pairing_budget"
        ] = True
    elif case == "paired_10k_drift":
        weighted_msa["max_input_paired_msa_depth"] = 9999
    elif case == "unpaired_10k_drift":
        weighted_msa["max_input_unpaired_msa_depth"] = 10001
    elif case == "final_16384_drift":
        weighted_msa["max_size"]["train"] = 16383
    else:  # pragma: no cover - protects the table itself.
        raise AssertionError(case)
    target["sha256"] = _canonical_contract_sha256(target)
    _bind_fixture_contract(
        monkeypatch,
        target,
        source_reference=reference,
    )

    with pytest.raises(ValueError, match=message):
        train._strict_resume_migration_expectations(object())


def test_target_training_contract_drift_fails_closed(monkeypatch):
    target = _target_data_contract()
    _bind_fixture_contract(monkeypatch, target)
    monkeypatch.setenv("PROTENIX_TRAINING_CONTRACT_ID", "wrong-target")

    with pytest.raises(ValueError, match="target training contract mismatch"):
        train._strict_resume_migration_expectations(object())


def test_target_data_sha_drift_fails_closed(monkeypatch):
    target = _target_data_contract()
    _bind_fixture_contract(monkeypatch, target)
    drifted = copy.deepcopy(target)
    drifted["sha256"] = "f" * 64
    monkeypatch.setattr(
        train,
        "build_training_data_contract",
        lambda configs: copy.deepcopy(drifted),
    )

    with pytest.raises(ValueError, match="target data contract mismatch"):
        train._strict_resume_migration_expectations(object())


def test_source_scheduler_is_validated_without_rewrite(monkeypatch):
    target = _target_data_contract()
    _bind_fixture_contract(monkeypatch, target)
    migration = train._strict_resume_migration_expectations(object())
    checkpoint = {
        "scheduler": {
            **migration["source_scheduler_state"],
            "_get_lr_called_within_step": False,
        },
        "optimizer": {"param_groups": [{"lr": 0.00095}]},
    }
    original = copy.deepcopy(checkpoint)

    train._validate_weightedpdb_msa_source_training_state(
        checkpoint,
        migration,
    )

    assert checkpoint == original
    assert migration["source_scheduler_state"] == (
        train.WEIGHTEDPDB_MSA_INDEPENDENT_SOURCE_SCHEDULER_STATE
    )

    checkpoint["scheduler"]["last_epoch"] += 1
    with pytest.raises(ValueError, match="source scheduler mismatch"):
        train._validate_weightedpdb_msa_source_training_state(
            checkpoint,
            migration,
        )

