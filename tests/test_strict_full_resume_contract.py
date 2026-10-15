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

from runner.train import (
    AF3Trainer,
    build_checkpoint_resume_contract,
    build_model_architecture_contract,
    build_training_data_contract,
    validate_matching_resume_contracts,
    validate_strict_checkpoint_payload,
    validate_strict_full_resume_configuration,
)


TRAINING_CONTRACT_ID = (
    "crop384_db48_ddp_scratch_50k_ws128_acc4_data_"
    "weightedpdb_only_v1_arch_pf64_cs384_dct768_pfckpt8_c48_"
    "msamix_q20_u30_p50_v8"
)


class _Scheduler:
    def __init__(self, last_epoch=4):
        self.last_epoch = last_epoch

    def state_dict(self):
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state_dict):
        self.last_epoch = int(state_dict["last_epoch"])


class _RuntimeLoader:
    def __init__(self, consumed=0):
        self.loaded_state = None
        self.state = {
            "schema_version": 1,
            "sampler_epoch": consumed // 5,
            "next_local_offset": consumed % 5,
            "consumed_global_batches": consumed,
            "batch_size": 1,
            "num_workers": 0,
            "sampler": {"test": True},
        }

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state_dict):
        self.loaded_state = dict(state_dict)
        self.state = dict(state_dict)


class _EMAProbe:
    def __init__(self, model):
        self.model = model
        self.shadow = None

    def register(self):
        self.shadow = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }


def _configs(**overrides):
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
        "model_name": "protenix-v2",
        "c_s": 384,
        "c_z": 256,
        "c_s_inputs": 449,
        "c_atom": 128,
        "c_atompair": 16,
        "c_token": 384,
        "max_atoms_per_token": 24,
        "no_bins": 64,
        "model": SimpleNamespace(
            pairformer=SimpleNamespace(
                n_blocks=64,
                n_heads=16,
                c_z=256,
                c_s=384,
                hidden_scale_up=True,
            ),
            diffusion_module=SimpleNamespace(
                c_token=768,
                c_atom=128,
                c_atompair=16,
                c_z=256,
                c_s=384,
                c_s_inputs=449,
                atom_encoder=SimpleNamespace(n_blocks=3, n_heads=4),
                transformer=SimpleNamespace(n_blocks=24, n_heads=16),
                atom_decoder=SimpleNamespace(n_blocks=3, n_heads=4),
            ),
        ),
        "data": SimpleNamespace(
            num_dl_workers=8,
            epoch_size=10240,
            query_only_msa_probability=0.2,
            pair_msa_as_unpaired_probability=0.3,
            template_use_probability=0.2,
            augmentation_decision_version="sample_occurrence_sha256_v1",
            template=SimpleNamespace(template_dropout_rate=0.0),
            train_sets=["weightedPDB_before210930_v20260101"],
            train_sampler=SimpleNamespace(
                train_sample_weights=[1.0],
                seed=-1,
                replacement=True,
            ),
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _contract(monkeypatch):
    monkeypatch.setenv("PROTENIX_TRAINING_CONTRACT_ID", TRAINING_CONTRACT_ID)
    trainer = SimpleNamespace(
        configs=_configs(),
        grad_scaler=torch.GradScaler(device="cpu", enabled=False),
        run_name="crop384_source_20260724_000000",
        step=4,
        global_step=19,
        iters_to_accumulate=4,
    )
    return build_checkpoint_resume_contract(trainer)


def _rank_runtime_state(*, next_local_offset=0, num_workers=8):
    return {
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
                    "sampler_epoch": 2,
                    "next_local_offset": next_local_offset,
                    "consumed_global_batches": 20,
                    "batch_size": 1,
                    "num_workers": num_workers,
                    "sampler": {"test": True},
                },
            }
        ],
    }


def _raw_checkpoint(monkeypatch):
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
        "resume_contract": _contract(monkeypatch),
        "rank_runtime_states": _rank_runtime_state(),
    }


def _rehash_contract(contract):
    material = dict(contract)
    material.pop("checkpoint_pair_id", None)
    contract["checkpoint_pair_id"] = hashlib.sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _rehash_architecture_contract(contract):
    material = dict(contract)
    material.pop("sha256", None)
    contract["sha256"] = hashlib.sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _ema_checkpoint(monkeypatch):
    return {
        "format_version": 2,
        "checkpoint_kind": "ema",
        "model": {"weight": torch.ones(1)},
        "optimizer": None,
        "scheduler": None,
        "grad_scaler": None,
        "step": 4,
        "resume_contract": _contract(monkeypatch),
    }


@pytest.fixture(autouse=True)
def _strict_environment(monkeypatch):
    monkeypatch.setenv("PROTENIX_STRICT_FULL_RESUME", "1")
    monkeypatch.setenv("PROTENIX_FULL_RESUME", "1")
    monkeypatch.setenv("PROTENIX_TRAINING_CONTRACT_ID", TRAINING_CONTRACT_ID)
    monkeypatch.setenv(
        "PROTENIX_TRAINING_DATA_PROFILE",
        "weighted_pdb_only_v1",
    )
    monkeypatch.setenv("PROTENIX_RESUME_CHECKPOINT_STEP", "4")


def _validate(checkpoint, kind):
    return validate_strict_checkpoint_payload(
        checkpoint,
        checkpoint_path=f"/tmp/{kind}.pt",
        expected_kind=kind,
        configs=_configs(),
        grad_scaler_enabled=False,
        device=torch.device("cpu"),
    )


def test_pdb_only_contract_binds_model_msa_query_inclusion_policy():
    legacy_data = copy.deepcopy(_configs().data)
    legacy_contract = build_training_data_contract(
        SimpleNamespace(data=legacy_data)
    )
    assert legacy_contract["schema_version"] == 3
    assert (
        "model_msa_query_inclusion"
        not in legacy_contract["augmentation"]
    )

    data = copy.deepcopy(legacy_data)
    data.msa = SimpleNamespace(
        query_inclusion_max_sample_size=SimpleNamespace(
            train=1024,
            test=1024,
        ),
    )
    contract = build_training_data_contract(SimpleNamespace(data=data))

    assert contract["schema_version"] == 3
    assert contract["augmentation"]["model_msa_query_inclusion"] == {
        "policy_version": (
            "reserve_global_query_row_zero_when_sampled_depth_le_threshold_v1"
        ),
        "query_row_index": 0,
        "max_sample_size": {
            "train": 1024,
            "test": 1024,
        },
        "depth_accounting": "replace_one_slot_keep_exact_sampled_depth",
        "above_threshold": "legacy_random_without_replacement",
    }
    assert contract["sha256"] != legacy_contract["sha256"]

    changed_train = copy.deepcopy(data)
    changed_train.msa.query_inclusion_max_sample_size.train = 1023
    assert build_training_data_contract(
        SimpleNamespace(data=changed_train)
    )["sha256"] != contract["sha256"]

    changed_test = copy.deepcopy(data)
    changed_test.msa.query_inclusion_max_sample_size.test = 0
    assert build_training_data_contract(
        SimpleNamespace(data=changed_test)
    )["sha256"] != contract["sha256"]


def test_training_data_contract_binds_runtime_index_source_hash(tmp_path):
    runtime_dir = tmp_path / "runtime-index"
    runtime_dir.mkdir()
    source_sha256 = "a" * 64
    (runtime_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "protenix_runtime_distillation_index",
                "row_count": 33_498_651,
                "cluster_count": 30_490_556,
                "source": {"sha256": source_sha256},
            }
        ),
        encoding="utf-8",
    )
    data = copy.deepcopy(_configs().data)
    data.train_sets = ["distillation"]
    data.train_sampler.train_sample_weights = [1.0]
    data.distillation = SimpleNamespace(
        base_info=SimpleNamespace(runtime_index_dir=str(runtime_dir)),
        augmentation=SimpleNamespace(),
        template=SimpleNamespace(),
    )

    contract = build_training_data_contract(SimpleNamespace(data=data))

    assert contract["schema_version"] == 4
    runtime_contract = contract["runtime_distillation_indices"]["distillation"]
    assert runtime_contract["source_index_sha256"] == source_sha256
    assert runtime_contract["row_count"] == 33_498_651
    assert len(runtime_contract["manifest_sha256"]) == 64


def test_training_data_contract_binds_disordered_loss_and_crop_policy():
    data = copy.deepcopy(_configs().data)
    data.train_sets = ["disordered"]
    data.train_sampler.train_sample_weights = [1.0]
    data.disordered = SimpleNamespace(
        base_info=SimpleNamespace(
            is_distillation=True,
            disable_non_protein_diffusion_weights=True,
            require_resolved_protein_crop=True,
            min_diffusion_supervised_tokens=4,
        ),
        augmentation=SimpleNamespace(),
        template=SimpleNamespace(),
    )

    protected_contract = build_training_data_contract(
        SimpleNamespace(data=data)
    )
    policy = protected_contract["loss_and_crop_policy"][
        "effective_by_dataset"
    ]["disordered"]

    assert protected_contract["schema_version"] == 6
    assert policy == {
        "disable_non_protein_diffusion_weights": True,
        "is_distillation": True,
        "diffusion_supervision_policy": (
            "protein_mse_residual_and_smooth_lddt_endpoints_v1"
        ),
        "require_resolved_protein_crop": True,
        "min_diffusion_supervised_tokens": 4,
        "diffusion_supervised_token_definition": (
            "token_has_resolved_protein_atom_v1"
        ),
        "minimum_validation_policy": (
            "pre_msa_crop_post_crop_final_feature_and_post_permutation_v2"
        ),
        "required_token_crop_policy": (
            "nearest_eligible_min_k_remove_farthest_complete_ref_space_uid_v2"
        ),
    }

    data.disordered.base_info.disable_non_protein_diffusion_weights = False
    all_molecule_contract = build_training_data_contract(
        SimpleNamespace(data=data)
    )
    assert all_molecule_contract["sha256"] != protected_contract["sha256"]

    data.disordered.base_info.require_resolved_protein_crop = False
    unrestricted_crop_contract = build_training_data_contract(
        SimpleNamespace(data=data)
    )
    assert unrestricted_crop_contract["sha256"] != all_molecule_contract[
        "sha256"
    ]

    data.disordered.base_info.min_diffusion_supervised_tokens = 3
    lower_minimum_contract = build_training_data_contract(
        SimpleNamespace(data=data)
    )
    assert lower_minimum_contract["sha256"] != unrestricted_crop_contract[
        "sha256"
    ]


def test_training_data_contract_binds_virtual_segment_msa_shuffle():
    data = copy.deepcopy(_configs().data)
    data.msa = SimpleNamespace(
        enable_virtual_segment_msa_shuffle=True,
        virtual_segment_residue_gap=24,
        virtual_segment_min_length=9,
    )

    contract = build_training_data_contract(
        SimpleNamespace(data=data, seed=42)
    )

    assert contract["schema_version"] == 7
    assert contract["sampler_seed"] == -1
    assert contract["effective_sampler_seed"] == 42
    assert contract["sampler_seed_resolution"] == (
        "negative_or_none_inherits_global_seed_v1"
    )
    assert contract["augmentation"]["msa_mode_policy"] == (
        "categorical_query_only_all_protein_segment_shuffle_species_paired_v2"
    )
    assert "crop_aware_partial_query_only" not in contract["augmentation"]
    assert contract["augmentation"]["virtual_segment_msa_shuffle"] == {
        "enabled": True,
        "policy_version": (
            "all_protein_msa_crop_segments_nearest_eligible_v1"
        ),
        "seed_stream": "virtual_segment_msa_shuffle",
        "residue_gap": 24,
        "min_segment_length": 9,
        "grouping_axis": (
            "unique_physical_asym_id_canonical_res_id"
        ),
        "short_segment_policy": (
            "nearest_eligible_same_chain_tie_left_v1"
        ),
        "protein_msa_sources": "all_merged_paired_and_unpaired_rows",
        "query_row_fixed": True,
        "existing_paired_query_placeholder_fixed": True,
    }
    assert contract["sha256"] == (
        "958be172060d37dee32181c38d46396eb9dc2dc8a3353046254919dd3e66ac46"
    )

    data.msa.enable_crop_aware_partial_query_only = False
    data.msa.partial_query_only_max_chains = 2
    explicit_disabled_defaults = build_training_data_contract(
        SimpleNamespace(data=data, seed=42)
    )
    assert explicit_disabled_defaults == contract

    data.msa.virtual_segment_min_length = 8
    changed = build_training_data_contract(
        SimpleNamespace(data=data, seed=42)
    )
    assert changed["sha256"] != contract["sha256"]


def test_schema7_contract_binds_crop_aware_partial_query_only_policy():
    data = copy.deepcopy(_configs().data)
    data.msa = SimpleNamespace(
        enable_virtual_segment_msa_shuffle=True,
        virtual_segment_residue_gap=24,
        virtual_segment_min_length=9,
        enable_crop_aware_partial_query_only=True,
        partial_query_only_max_chains=2,
    )

    contract = build_training_data_contract(
        SimpleNamespace(data=data, seed=42)
    )

    assert contract["schema_version"] == 7
    assert contract["augmentation"]["msa_mode_policy"] == (
        "categorical_crop_aware_partial_query_only_"
        "all_protein_segment_shuffle_species_paired_v3"
    )
    assert contract["augmentation"]["crop_aware_partial_query_only"] == {
        "enabled": True,
        "policy_version": (
            "crop_selected_physical_protein_asym_"
            "uniform_1_to_max_retain_one_v1"
        ),
        "seed_stream": "partial_query_only_chain_selection",
        "selection_stage": "post_crop_pre_homolog_io",
        "candidate_axis": "crop_selected_physical_protein_asym_id",
        "candidate_order": "sorted_unique_asym_id",
        "homomer_copies_are_distinct_candidates": True,
        "selected_count_policy": (
            "uniform_integer_1_to_min_max_chains_n_minus_1"
        ),
        "max_query_only_chains": 2,
        "minimum_retained_protein_chains": 1,
        "zero_or_one_protein_policy": "legacy_global_query_only_v1",
        "selected_chain_homolog_io": "skipped",
        "selected_chain_non_query_rows": (
            "gap_with_zero_deletion_query_profile_v1"
        ),
        "retained_chain_policy": "normal_protenix_msa_flow",
        "sole_retained_chain_paired_hits": (
            "merge_into_unpaired_to_preserve_all_sources_v1"
        ),
        "multi_protein_non_protein_policy": (
            "not_candidates_and_normal_msa_flow"
        ),
        "query_row_fixed": True,
        "deterministic_scope": "occurrence_seeded_given_crop",
    }
    assert contract["dataset_runtime_semantics"]["effective_by_dataset"][
        "weightedPDB_before210930_v20260101"
    ]["effective_msa"]["partial_query_only_max_chains"] == 2
    assert contract["sha256"] == (
        "cef9da75c007c5f30cce2275b3e93264c5ff4ecd779e3445f1e34a29a38798bc"
    )

    data.msa.partial_query_only_max_chains = 1
    max_one_contract = build_training_data_contract(
        SimpleNamespace(data=data, seed=42)
    )
    assert max_one_contract["augmentation"][
        "crop_aware_partial_query_only"
    ]["max_query_only_chains"] == 1
    assert max_one_contract["sha256"] == (
        "bf103aa39502c92def7119e9b83107921d69fe3bfd4da77bea87415c70b476b7"
    )
    assert max_one_contract["sha256"] != contract["sha256"]


def test_schema7_contract_binds_effective_runtime_data_semantics():
    dataset_name = "weightedPDB_before210930_v20260101"
    data = copy.deepcopy(_configs().data)
    data.train_ref_pos_augment = True
    data.limits = -1
    data.msa = SimpleNamespace(
        enable_virtual_segment_msa_shuffle=True,
        virtual_segment_residue_gap=24,
        virtual_segment_min_length=9,
        prot_non_pairing_dbs=["global_uniref90"],
    )
    data.template = SimpleNamespace(
        template_dropout_rate=0.0,
        template_mmcif_dir="/global/templates",
        evaluation_sequence_template_cache_root="/eval/cache-a",
        evaluation_sequence_template_cache_mode="required",
        evaluation_sequence_template_cache_filename="features-a.pkl.gz",
        evaluation_sequence_template_cache_policy="locked_cutoff_a",
    )
    setattr(
        data,
        dataset_name,
        SimpleNamespace(
            base_info=SimpleNamespace(),
            cropping_configs=SimpleNamespace(crop_size=384),
            sampler_configs=SimpleNamespace(sampler_type="weighted"),
            msa=SimpleNamespace(
                prot_non_pairing_dbs=["dataset_uniref90"],
            ),
            template=SimpleNamespace(enable_prot_template=False),
            lig_atom_rename=False,
            shuffle_mols=False,
            shuffle_sym_ids=False,
            constraint=SimpleNamespace(enable=False),
        ),
    )

    base_configs = SimpleNamespace(data=data, seed=42)
    contract = build_training_data_contract(base_configs)
    runtime = contract["dataset_runtime_semantics"][
        "effective_by_dataset"
    ][dataset_name]

    assert runtime["stage"] == "train"
    assert runtime["effective_msa"]["dataset_name"] == dataset_name
    assert runtime["effective_msa"]["prot_non_pairing_dbs"] == [
        "dataset_uniref90"
    ]
    assert runtime["effective_template"]["dataset_name"] == dataset_name
    assert runtime["effective_template"]["stage"] == "train"
    assert runtime["effective_template"]["template_mmcif_dir"] == (
        "/global/templates"
    )
    assert runtime["effective_template"]["enable_prot_template"] is False
    for evaluation_only_key in (
        "evaluation_sequence_template_cache_root",
        "evaluation_sequence_template_cache_mode",
        "evaluation_sequence_template_cache_filename",
        "evaluation_sequence_template_cache_policy",
    ):
        assert evaluation_only_key not in runtime["effective_template"]
    assert runtime["dataset_config"]["lig_atom_rename"] is False
    assert runtime["train_ref_pos_augment"] is True
    assert runtime["limits"] == -1

    changed_global_msa = copy.deepcopy(base_configs)
    changed_global_msa.data.msa.prot_indexing_methods = ["sequence"]
    assert build_training_data_contract(changed_global_msa)["sha256"] != (
        contract["sha256"]
    )

    changed_global_template = copy.deepcopy(base_configs)
    changed_global_template.data.template.template_mmcif_dir = (
        "/different/templates"
    )
    assert build_training_data_contract(changed_global_template)["sha256"] != (
        contract["sha256"]
    )

    changed_evaluation_template = copy.deepcopy(base_configs)
    changed_evaluation_template.data.template.evaluation_sequence_template_cache_root = (
        "/eval/cache-b"
    )
    changed_evaluation_template.data.template.evaluation_sequence_template_cache_mode = (
        "disabled"
    )
    changed_evaluation_template.data.template.evaluation_sequence_template_cache_filename = (
        "features-b.pkl.gz"
    )
    changed_evaluation_template.data.template.evaluation_sequence_template_cache_policy = (
        "locked_cutoff_b"
    )
    assert build_training_data_contract(
        changed_evaluation_template
    )["sha256"] == contract["sha256"]

    changed_dataset_option = copy.deepcopy(base_configs)
    getattr(changed_dataset_option.data, dataset_name).shuffle_mols = True
    assert build_training_data_contract(changed_dataset_option)["sha256"] != (
        contract["sha256"]
    )

    changed_ref_augment = copy.deepcopy(base_configs)
    changed_ref_augment.data.train_ref_pos_augment = False
    assert build_training_data_contract(changed_ref_augment)["sha256"] != (
        contract["sha256"]
    )

    changed_global_seed = copy.deepcopy(base_configs)
    changed_global_seed.seed = 43
    changed_seed_contract = build_training_data_contract(
        changed_global_seed
    )
    assert changed_seed_contract["effective_sampler_seed"] == 43
    assert changed_seed_contract["sha256"] != contract["sha256"]

    explicit_seed = copy.deepcopy(base_configs)
    explicit_seed.data.train_sampler.seed = 17
    explicit_seed_contract = build_training_data_contract(explicit_seed)
    changed_unused_global_seed = copy.deepcopy(explicit_seed)
    changed_unused_global_seed.seed = 99
    changed_unused_global_seed_contract = build_training_data_contract(
        changed_unused_global_seed
    )
    assert explicit_seed_contract["effective_sampler_seed"] == 17
    assert changed_unused_global_seed_contract["effective_sampler_seed"] == 17
    assert changed_unused_global_seed_contract["sha256"] == (
        explicit_seed_contract["sha256"]
    )


def test_independent_pairing_budget_contract_requires_genuine_source():
    data = copy.deepcopy(_configs().data)
    data.msa = SimpleNamespace(
        independent_pairing_non_pairing_budget=True,
        prot_pairing_dbs=["pairing"],
        prot_non_pairing_dbs=["pairing"],
    )
    pairing_only = build_training_data_contract(
        SimpleNamespace(data=data, seed=42)
    )

    disabled_data = copy.deepcopy(data)
    disabled_data.msa.independent_pairing_non_pairing_budget = False
    disabled = build_training_data_contract(
        SimpleNamespace(data=disabled_data, seed=42)
    )
    assert pairing_only["sha256"] == disabled["sha256"]

    genuine_data = copy.deepcopy(data)
    genuine_data.msa.prot_non_pairing_dbs = ["pairing-non_pairing"]
    genuine = build_training_data_contract(
        SimpleNamespace(data=genuine_data, seed=42)
    )
    genuine_effective = genuine["dataset_runtime_semantics"][
        "effective_by_dataset"
    ]["weightedPDB_before210930_v20260101"]["effective_msa"]
    assert genuine_effective[
        "independent_pairing_non_pairing_budget"
    ] is True
    assert genuine["sha256"] != disabled["sha256"]


def test_schema7_contract_rejects_unresolved_inherited_sampler_seed():
    data = copy.deepcopy(_configs().data)
    data.msa = SimpleNamespace(
        enable_virtual_segment_msa_shuffle=True,
        virtual_segment_residue_gap=24,
        virtual_segment_min_length=9,
    )

    with pytest.raises(ValueError, match="global training seed is missing"):
        build_training_data_contract(SimpleNamespace(data=data))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("enable_virtual_segment_msa_shuffle", "true", "must be a boolean"),
        ("virtual_segment_residue_gap", 0, "must be >= 1"),
        ("virtual_segment_min_length", True, "non-negative integer"),
    ],
)
def test_training_data_contract_rejects_invalid_virtual_segment_options(
    field,
    value,
    message,
):
    data = copy.deepcopy(_configs().data)
    data.msa = SimpleNamespace(
        enable_virtual_segment_msa_shuffle=True,
        virtual_segment_residue_gap=24,
        virtual_segment_min_length=9,
    )
    setattr(data.msa, field, value)

    with pytest.raises((TypeError, ValueError), match=message):
        build_training_data_contract(SimpleNamespace(data=data))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "enable_crop_aware_partial_query_only",
            "true",
            "must be a boolean",
        ),
        (
            "partial_query_only_max_chains",
            True,
            "non-negative integer",
        ),
        (
            "partial_query_only_max_chains",
            0,
            "must be >= 1",
        ),
    ],
)
def test_training_data_contract_rejects_invalid_partial_query_only_options(
    field,
    value,
    message,
):
    data = copy.deepcopy(_configs().data)
    data.msa = SimpleNamespace(
        enable_virtual_segment_msa_shuffle=True,
        virtual_segment_residue_gap=24,
        virtual_segment_min_length=9,
        enable_crop_aware_partial_query_only=True,
        partial_query_only_max_chains=2,
    )
    setattr(data.msa, field, value)

    with pytest.raises((TypeError, ValueError), match=message):
        build_training_data_contract(
            SimpleNamespace(data=data, seed=42)
        )


def test_training_data_contract_binds_disordered_artifacts(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    indices_path = tmp_path / "indices.csv.gz"
    indices_path.write_bytes(b"first-index")
    first_indices_sha256 = hashlib.sha256(b"first-index").hexdigest()
    manifest_payload = {
        "schema_version": 3,
        "artifact_role": "openfold3_disordered_protenix_dataset",
        "status": "complete",
        "success_eligible": True,
        "artifacts": {
            "indices": {"sha256": first_indices_sha256},
        },
    }
    manifest_bytes = json.dumps(
        manifest_payload,
        sort_keys=True,
    ).encode()
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    success_path = tmp_path / "_SUCCESS.json"
    success_path.write_text(
        json.dumps(
            {
                "manifest_sha256": manifest_sha256,
                "cache_sha256": "b" * 64,
                "entry_count": 2,
                "index_row_count": 3,
                "pickle_count": 2,
            }
        ),
        encoding="utf-8",
    )
    data = copy.deepcopy(_configs().data)
    data.train_sets = ["disordered"]
    data.train_sampler.train_sample_weights = [1.0]
    data.disordered = SimpleNamespace(
        base_info=SimpleNamespace(
            indices_fpath=str(indices_path),
            dataset_success_fpath=str(success_path),
            dataset_manifest_fpath=str(manifest_path),
            is_distillation=True,
            disable_non_protein_diffusion_weights=True,
            require_resolved_protein_crop=True,
            min_diffusion_supervised_tokens=4,
        ),
        augmentation=SimpleNamespace(),
        cropping_configs=SimpleNamespace(crop_size=384),
        sampler_configs=SimpleNamespace(force_recompute_weight=False),
        msa=SimpleNamespace(openfold3_alignment_npz_dirs=["/msa"]),
        template=SimpleNamespace(enable_prot_template=False),
    )

    first = build_training_data_contract(SimpleNamespace(data=data))
    artifact = first["immutable_dataset_artifacts"]["disordered"]
    assert artifact["manifest"]["sha256"] == manifest_sha256
    assert artifact["indices"]["sha256"] == first_indices_sha256

    indices_path.write_bytes(b"second-index")
    with pytest.raises(ValueError, match="indices sha256"):
        build_training_data_contract(SimpleNamespace(data=data))

    second_indices_sha256 = hashlib.sha256(b"second-index").hexdigest()
    manifest_payload["artifacts"]["indices"]["sha256"] = (
        second_indices_sha256
    )
    second_manifest_bytes = json.dumps(
        manifest_payload,
        sort_keys=True,
    ).encode()
    manifest_path.write_bytes(second_manifest_bytes)
    success_payload = json.loads(success_path.read_text())
    success_payload["manifest_sha256"] = hashlib.sha256(
        second_manifest_bytes
    ).hexdigest()
    success_path.write_text(json.dumps(success_payload), encoding="utf-8")
    second = build_training_data_contract(SimpleNamespace(data=data))
    assert second["sha256"] != first["sha256"]

    data.disordered.cropping_configs.crop_size = 256
    third = build_training_data_contract(SimpleNamespace(data=data))
    assert third["sha256"] != second["sha256"]


def test_three_source_profile_rejects_missing_minimum_overrides(monkeypatch):
    dataset_names = [
        "weightedPDB_before210930_v20260101",
        "afdb_mgnify_distillation_v20260725",
        "openfold3_disordered_v20260725",
    ]
    data = copy.deepcopy(_configs().data)
    data.train_sets = dataset_names
    data.train_sampler.train_sample_weights = [0.5, 0.5, 0.02]
    for dataset_name in dataset_names:
        setattr(
            data,
            dataset_name,
            SimpleNamespace(
                base_info=SimpleNamespace(
                    is_distillation=dataset_name
                    != "weightedPDB_before210930_v20260101",
                    disable_non_protein_diffusion_weights=False,
                    require_resolved_protein_crop=False,
                    min_diffusion_supervised_tokens=0,
                ),
                augmentation=SimpleNamespace(),
                template=SimpleNamespace(),
            ),
        )
    monkeypatch.setenv("PROTENIX_TRAINING_DATA_PROFILE", "custom_three_source")

    with pytest.raises(
        ValueError,
        match="min_diffusion_supervised_tokens=4",
    ):
        build_training_data_contract(SimpleNamespace(data=data))


def test_reserved_three_source_profile_requires_artifact_contract(monkeypatch):
    dataset_names = [
        "weightedPDB_before210930_v20260101",
        "afdb_mgnify_distillation_v20260725",
        "openfold3_disordered_v20260725",
    ]
    data = copy.deepcopy(_configs().data)
    data.train_sets = dataset_names
    data.train_sampler.train_sample_weights = [0.5, 0.5, 0.02]
    for dataset_name in dataset_names:
        is_predicted = dataset_name != dataset_names[0]
        is_disordered = dataset_name == dataset_names[2]
        setattr(
            data,
            dataset_name,
            SimpleNamespace(
                base_info=SimpleNamespace(
                    is_distillation=is_predicted,
                    disable_non_protein_diffusion_weights=is_disordered,
                    require_resolved_protein_crop=is_predicted,
                    min_diffusion_supervised_tokens=4,
                ),
                augmentation=SimpleNamespace(),
                template=SimpleNamespace(),
            ),
        )
    monkeypatch.setenv(
        "PROTENIX_TRAINING_DATA_PROFILE",
        "weighted_pdb_afdb_mgnify_disordered_v2",
    )

    with pytest.raises(
        ValueError,
        match="immutable disordered indices/_SUCCESS/manifest",
    ):
        build_training_data_contract(SimpleNamespace(data=data))


def test_reserved_three_source_profile_rejects_dataset_msa_overrides(
    monkeypatch,
    tmp_path,
):
    dataset_names = [
        "weightedPDB_before210930_v20260101",
        "afdb_mgnify_distillation_v20260725",
        "openfold3_disordered_v20260725",
    ]
    runtime_index_dir = tmp_path / "runtime-index"
    runtime_index_dir.mkdir()
    (runtime_index_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "test",
                "row_count": 1,
                "cluster_count": 1,
                "source": {"sha256": "a" * 64},
            }
        ),
        encoding="utf-8",
    )

    indices_path = tmp_path / "indices.csv.gz"
    indices_path.write_bytes(b"index")
    indices_sha256 = hashlib.sha256(b"index").hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_bytes = json.dumps(
        {
            "schema_version": 1,
            "status": "complete",
            "success_eligible": True,
            "artifacts": {
                "indices": {"sha256": indices_sha256},
            },
        },
        sort_keys=True,
    ).encode()
    manifest_path.write_bytes(manifest_bytes)
    success_path = tmp_path / "_SUCCESS.json"
    success_path.write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(
                    manifest_bytes
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    data = copy.deepcopy(_configs().data)
    data.train_sets = dataset_names
    data.train_sampler.train_sample_weights = [0.5, 0.5, 0.02]
    data.msa = SimpleNamespace(
        enable_virtual_segment_msa_shuffle=True,
        virtual_segment_residue_gap=24,
        virtual_segment_min_length=9,
        enable_crop_aware_partial_query_only=True,
        partial_query_only_max_chains=2,
    )
    for dataset_name in dataset_names:
        is_predicted = dataset_name != dataset_names[0]
        is_disordered = dataset_name == dataset_names[2]
        base_info = SimpleNamespace(
            is_distillation=is_predicted,
            disable_non_protein_diffusion_weights=is_disordered,
            require_resolved_protein_crop=is_predicted,
            min_diffusion_supervised_tokens=4,
        )
        if dataset_name == dataset_names[1]:
            base_info.runtime_index_dir = str(runtime_index_dir)
        elif is_disordered:
            base_info.indices_fpath = str(indices_path)
            base_info.dataset_success_fpath = str(success_path)
            base_info.dataset_manifest_fpath = str(manifest_path)
        setattr(
            data,
            dataset_name,
            SimpleNamespace(
                base_info=base_info,
                augmentation=SimpleNamespace(),
                cropping_configs=SimpleNamespace(crop_size=384),
                sampler_configs=SimpleNamespace(),
                msa=SimpleNamespace(),
                template=SimpleNamespace(),
            ),
        )
    monkeypatch.setenv(
        "PROTENIX_TRAINING_DATA_PROFILE",
        "weighted_pdb_afdb_mgnify_disordered_v2",
    )

    valid_contract = build_training_data_contract(
        SimpleNamespace(data=data, seed=42)
    )
    assert valid_contract["schema_version"] == 7

    for field, value in (
        ("enable_virtual_segment_msa_shuffle", False),
        ("virtual_segment_residue_gap", 23),
        ("virtual_segment_min_length", 8),
        ("enable_crop_aware_partial_query_only", False),
        ("partial_query_only_max_chains", 1),
    ):
        overridden = copy.deepcopy(data)
        setattr(
            getattr(overridden, dataset_names[1]).msa,
            field,
            value,
        )
        with pytest.raises(
            ValueError,
            match="effective MSA settings",
        ):
            build_training_data_contract(
                SimpleNamespace(data=overridden, seed=42)
            )


def _four_source_v3_contract_config(tmp_path, monkeypatch):
    dataset_names = [
        "weightedPDB_before210930_v20260101",
        "afdb_mgnify_short_distillation_v20260725",
        "afdb_mgnify_long_distillation_v20260725",
        "openfold3_disordered_v20260725",
    ]
    runtime_specs = {
        dataset_names[1]: (15_425_587, 14_327_216),
        dataset_names[2]: (18_073_064, 16_163_340),
    }
    partition_roles = {
        dataset_names[1]: "short_le200",
        dataset_names[2]: "long_gt200",
    }
    source_hashes = {
        "full_runtime_manifest_sha256": (
            "23ab7c5563180da276f482596310a0b4064ded8c0ae6796f10ac86bbd1ace6d7"
        ),
        "clusters": (
            "d7e94ff572a9496900ac821f49c05c7b78f077ab6827780a27e465f7b96cd352"
        ),
        "short_paths": (
            "72c5655950119771f79f9ebdaf75b2917b22c0d2597791df1297ab60158afe38"
        ),
        "short_paths_stats": (
            "da2e3cbe6580a6e98b526e5d47bc08000140af56a46584b1c68af23a0d0ef569"
        ),
        "canonical_audit": (
            "c2ab75330588b70fcabccd3fa66cc07beb3c6ef6e4ecb0a015452657ac0c5439"
        ),
        "canonical_audit_success": (
            "788e32e367a09d9b73f045eb9a71362021b683c9a68f12fc0867e6bb9fac26cc"
        ),
    }
    pair_material = {
        "partition_schema_version": 1,
        "full_runtime_manifest_sha256": source_hashes[
            "full_runtime_manifest_sha256"
        ],
        "clusters_sha256": source_hashes["clusters"],
        "short_paths_sha256": source_hashes["short_paths"],
        "short_stats_sha256": source_hashes["short_paths_stats"],
        "canonical_audit_sha256": source_hashes["canonical_audit"],
        "canonical_audit_success_sha256": source_hashes[
            "canonical_audit_success"
        ],
        "max_representative_length": 200,
    }
    pair_id = hashlib.sha256(
        json.dumps(
            pair_material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert (
        pair_id
        == "9c87d52dd446f622be689b7a09611f302d3791ac2448ecd86f8aee2488fb8734"
    )
    criterion = {
        "representative": "final representative_row_id from clusters.csv",
        "short": "canonical_length <= 200",
        "long": "canonical_length > 200",
        "max_representative_length": 200,
    }
    pair_receipt = {
        "schema_version": 1,
        "format": "protenix_distillation_length_partition_pair",
        "pair_id": pair_id,
        "criterion": criterion,
        "member_policy": "retain_whole_source_cluster",
        "source": {
            "row_count": 33_498_651,
            "cluster_count": 30_490_556,
            "full_runtime_manifest_sha256": source_hashes[
                "full_runtime_manifest_sha256"
            ],
            **{
                name: {"sha256": source_hashes[name]}
                for name in (
                    "clusters",
                    "short_paths",
                    "short_paths_stats",
                    "canonical_audit",
                    "canonical_audit_success",
                )
            },
        },
        "partitions": {
            partition_roles[name]: {
                "row_count": counts[0],
                "cluster_count": counts[1],
            }
            for name, counts in runtime_specs.items()
        },
        "coverage": {
            "row_count_sum": 33_498_651,
            "cluster_count_sum": 30_490_556,
            "overlap": 0,
        },
    }
    pair_bytes = (
        json.dumps(pair_receipt, indent=2, sort_keys=True) + "\n"
    ).encode()
    pair_sha256 = hashlib.sha256(pair_bytes).hexdigest()
    runtime_dirs = {}
    for ordinal, (dataset_name, counts) in enumerate(
        runtime_specs.items()
    ):
        runtime_dir = tmp_path / f"runtime-{ordinal}"
        runtime_dir.mkdir()
        (runtime_dir / "partition_pair.json").write_bytes(pair_bytes)
        (runtime_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "format": "protenix_runtime_distillation_index",
                    "row_count": counts[0],
                    "cluster_count": counts[1],
                    "source": {"sha256": f"{ordinal + 1}" * 64},
                    "artifacts": {
                        "partition_pair": {
                            "path": "partition_pair.json",
                            "size_bytes": len(pair_bytes),
                            "sha256": pair_sha256,
                        },
                    },
                    "partition": {
                        "schema_version": 1,
                        "pair_id": pair_id,
                        "role": partition_roles[dataset_name],
                        "criterion": criterion,
                        "member_policy": "retain_whole_source_cluster",
                        "source_cluster_id_policy": (
                            "preserve_stable_cluster_id"
                        ),
                        "row_order": "source order",
                        "pair_receipt_sha256": pair_sha256,
                        "source_row_count": 33_498_651,
                        "source_cluster_count": 30_490_556,
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        runtime_dirs[dataset_name] = runtime_dir

    indices_path = tmp_path / "disordered_indices.csv.gz"
    indices_path.write_bytes(b"disordered-index")
    indices_sha256 = hashlib.sha256(b"disordered-index").hexdigest()
    manifest_path = tmp_path / "disordered_manifest.json"
    manifest_bytes = json.dumps(
        {
            "schema_version": 1,
            "status": "complete",
            "success_eligible": True,
            "artifacts": {
                "indices": {"sha256": indices_sha256},
            },
        },
        sort_keys=True,
    ).encode()
    manifest_path.write_bytes(manifest_bytes)
    success_path = tmp_path / "disordered_SUCCESS.json"
    success_path.write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(
                    manifest_bytes
                ).hexdigest(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    data = copy.deepcopy(_configs().data)
    data.query_only_msa_probability = 0.1
    data.pair_msa_as_unpaired_probability = 0.3
    data.train_sets = dataset_names
    data.train_sampler.train_sample_weights = [
        0.5,
        0.005,
        0.495,
        0.02,
    ]
    data.msa = SimpleNamespace(
        enable_virtual_segment_msa_shuffle=True,
        virtual_segment_residue_gap=24,
        virtual_segment_min_length=9,
        enable_multichain_msa_shuffle_variant_sampling=True,
        multichain_msa_shuffle_segment_probability=0.5,
        enable_crop_aware_partial_query_only=True,
        partial_query_only_max_chains=2,
        max_input_paired_msa_depth=10000,
        max_input_unpaired_msa_depth=10000,
    )
    for dataset_name in dataset_names:
        is_predicted = dataset_name != dataset_names[0]
        is_disordered = dataset_name == dataset_names[-1]
        base_info = SimpleNamespace(
            is_distillation=is_predicted,
            disable_non_protein_diffusion_weights=is_disordered,
            require_resolved_protein_crop=is_predicted,
            min_diffusion_supervised_tokens=4,
        )
        if dataset_name in runtime_dirs:
            base_info.runtime_index_dir = str(
                runtime_dirs[dataset_name]
            )
        elif is_disordered:
            base_info.indices_fpath = str(indices_path)
            base_info.dataset_success_fpath = str(success_path)
            base_info.dataset_manifest_fpath = str(manifest_path)
        setattr(
            data,
            dataset_name,
            SimpleNamespace(
                base_info=base_info,
                augmentation=SimpleNamespace(
                    query_only_msa_probability=0.1,
                    pair_msa_as_unpaired_probability=0.3,
                ),
                cropping_configs=SimpleNamespace(crop_size=384),
                sampler_configs=SimpleNamespace(
                    sampler_type=(
                        "clustered"
                        if dataset_name in runtime_dirs
                        else "weighted"
                    ),
                ),
                msa=(
                    SimpleNamespace(
                        max_input_unpaired_msa_depth=16384,
                    )
                    if dataset_name in runtime_dirs
                    else SimpleNamespace()
                ),
                template=SimpleNamespace(),
            ),
        )
    monkeypatch.setenv(
        "PROTENIX_TRAINING_DATA_PROFILE",
        "weighted_pdb_afdb_mgnify_short_long_disordered_v3",
    )
    return SimpleNamespace(data=data, seed=42), dataset_names


def test_reserved_four_source_v3_contract_binds_partition_and_msa_policy(
    monkeypatch,
    tmp_path,
):
    configs, dataset_names = _four_source_v3_contract_config(
        tmp_path,
        monkeypatch,
    )
    contract = build_training_data_contract(configs)

    assert contract["schema_version"] == 7
    assert contract["train_sets"] == dataset_names
    assert contract["train_sample_weights"] == [
        0.5,
        0.005,
        0.495,
        0.02,
    ]
    augmentation = contract["augmentation"]
    assert augmentation["msa_mode_policy"].endswith(
        "multichain_segment_or_whole_shuffle_species_paired_v4"
    )
    variant = augmentation["multichain_msa_shuffle_variant"]
    assert variant["segment_probability"] == 0.5
    assert variant["whole_physical_chain_probability"] == 0.5
    assert variant["single_protein_chain_policy"] == "segment"
    assert variant["query_row_fixed"] is True
    runtime = contract["runtime_distillation_indices"]
    assert runtime[dataset_names[1]]["cluster_count"] == 14_327_216
    assert runtime[dataset_names[2]]["cluster_count"] == 16_163_340
    assert (
        runtime[dataset_names[1]]["partition"]["manifest_metadata"]["role"]
        == "short_le200"
    )
    assert (
        runtime[dataset_names[2]]["partition"]["manifest_metadata"]["role"]
        == "long_gt200"
    )
    assert (
        runtime[dataset_names[1]]["partition"]["receipt_artifact"]["sha256"]
        == runtime[dataset_names[2]]["partition"]["receipt_artifact"]["sha256"]
    )
    assert (
        runtime[dataset_names[1]]["row_count"]
        + runtime[dataset_names[2]]["row_count"]
        == 33_498_651
    )
    effective = contract["dataset_runtime_semantics"][
        "effective_by_dataset"
    ]
    for dataset_name in (dataset_names[0], dataset_names[3]):
        assert (
            effective[dataset_name]["effective_msa"][
                "max_input_paired_msa_depth"
            ]
            == 10000
        )
        assert (
            effective[dataset_name]["effective_msa"][
                "max_input_unpaired_msa_depth"
            ]
            == 10000
        )
    for dataset_name in (dataset_names[1], dataset_names[2]):
        assert (
            effective[dataset_name]["effective_msa"][
                "max_input_paired_msa_depth"
            ]
            == 10000
        )
        assert (
            effective[dataset_name]["effective_msa"][
                "max_input_unpaired_msa_depth"
            ]
            == 16384
        )

    old_distillation_budget = copy.deepcopy(configs)
    getattr(
        old_distillation_budget.data,
        dataset_names[1],
    ).msa.max_input_unpaired_msa_depth = 10000
    assert build_training_data_contract(
        old_distillation_budget
    )["sha256"] != contract["sha256"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("weights", "raw outer weights"),
        ("dataset_probability", "MSA mode probabilities"),
        ("variant_override", "effective multichain variant sampling"),
        ("partition_row_count", "row_count=15425587"),
        ("partition_role", "length-partition metadata"),
        ("partition_pair_id", "production partition pair_id"),
        ("partition_source_hash", "does not match its recorded source hashes"),
        ("partition_pair_id_noncanonical", "length-partition metadata"),
        ("partition_receipt", "receipt identity"),
        ("partition_criterion", "length-partition metadata"),
    ],
)
def test_reserved_four_source_v3_contract_fails_closed(
    monkeypatch,
    tmp_path,
    mutation,
    message,
):
    configs, dataset_names = _four_source_v3_contract_config(
        tmp_path,
        monkeypatch,
    )
    if mutation == "weights":
        configs.data.train_sampler.train_sample_weights[1] = 0.01
    elif mutation == "dataset_probability":
        getattr(
            configs.data,
            dataset_names[2],
        ).augmentation.query_only_msa_probability = 0.2
    elif mutation == "variant_override":
        getattr(
            configs.data,
            dataset_names[1],
        ).msa.enable_multichain_msa_shuffle_variant_sampling = False
    elif mutation in {"partition_role", "partition_criterion"}:
        runtime_dir = tmp_path / "runtime-0"
        manifest_path = runtime_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if mutation == "partition_role":
            manifest["partition"]["role"] = "long_gt200"
        else:
            manifest["partition"]["criterion"][
                "max_representative_length"
            ] = 201
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True),
            encoding="utf-8",
        )
    elif mutation == "partition_row_count":
        manifest_path = tmp_path / "runtime-0" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["row_count"] += 1
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True),
            encoding="utf-8",
        )
    elif mutation in {"partition_pair_id", "partition_source_hash"}:
        runtime_dir = tmp_path / "runtime-1"
        receipt_path = runtime_dir / "partition_pair.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["source"]["clusters"]["sha256"] = "cd" * 32
        if mutation == "partition_pair_id":
            source = receipt["source"]
            pair_material = {
                "partition_schema_version": 1,
                "full_runtime_manifest_sha256": source[
                    "full_runtime_manifest_sha256"
                ],
                "clusters_sha256": source["clusters"]["sha256"],
                "short_paths_sha256": source["short_paths"]["sha256"],
                "short_stats_sha256": source["short_paths_stats"]["sha256"],
                "canonical_audit_sha256": source["canonical_audit"][
                    "sha256"
                ],
                "canonical_audit_success_sha256": source[
                    "canonical_audit_success"
                ]["sha256"],
                "max_representative_length": 200,
            }
            receipt["pair_id"] = hashlib.sha256(
                json.dumps(
                    pair_material,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        receipt_bytes = (
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        ).encode()
        receipt_path.write_bytes(receipt_bytes)
        manifest_path = runtime_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        if mutation == "partition_pair_id":
            manifest["partition"]["pair_id"] = receipt["pair_id"]
        manifest["partition"]["pair_receipt_sha256"] = receipt_sha256
        manifest["artifacts"]["partition_pair"].update(
            size_bytes=len(receipt_bytes),
            sha256=receipt_sha256,
        )
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True),
            encoding="utf-8",
        )
    elif mutation == "partition_pair_id_noncanonical":
        for ordinal in range(2):
            runtime_dir = tmp_path / f"runtime-{ordinal}"
            receipt_path = runtime_dir / "partition_pair.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["pair_id"] = "+" + "a" * 63
            receipt_bytes = (
                json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            ).encode()
            receipt_path.write_bytes(receipt_bytes)
            manifest_path = runtime_dir / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
            manifest["partition"]["pair_id"] = receipt["pair_id"]
            manifest["partition"]["pair_receipt_sha256"] = receipt_sha256
            manifest["artifacts"]["partition_pair"].update(
                size_bytes=len(receipt_bytes),
                sha256=receipt_sha256,
            )
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True),
                encoding="utf-8",
            )
    elif mutation == "partition_receipt":
        receipt_path = tmp_path / "runtime-0" / "partition_pair.json"
        receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=message):
        build_training_data_contract(configs)


def test_strict_pair_accepts_every_required_state_and_matching_identity(
    monkeypatch,
):
    raw_contract = _validate(_raw_checkpoint(monkeypatch), "raw")
    ema_contract = _validate(_ema_checkpoint(monkeypatch), "ema")

    validate_matching_resume_contracts(raw_contract, ema_contract)
    assert raw_contract["checkpoint_pair_id"] == ema_contract["checkpoint_pair_id"]
    assert raw_contract["next_step"] == 5
    assert raw_contract["next_global_step"] == 20
    assert raw_contract["data_num_workers"] == 8
    assert raw_contract["data_train_dl_persistent_workers"] is True
    assert raw_contract["data_train_dl_prefetch_factor"] == 2
    assert raw_contract["data_train_dl_pin_memory"] is False
    assert raw_contract["schema_version"] == 3
    assert raw_contract["data_profile_id"] == "weighted_pdb_only_v1"
    assert raw_contract["data_train_sets"] == [
        "weightedPDB_before210930_v20260101"
    ]
    assert raw_contract["data_train_sample_weights"] == [1.0]
    assert raw_contract["data_contract_sha256"] == (
        "835a4089f4b20b138c310be537b95dedcdd0ca65cb7a8fd39eddc460947c3a08"
    )
    assert raw_contract["data_augmentation"] == {
        "decision_version": "sample_occurrence_sha256_v1",
        "msa_mode_policy": (
            "categorical_query_only_unpaired_shuffle_species_paired_v1"
        ),
        "msa_mode_stream": "msa_mode",
        "msa_mode_order": [
            "query_only",
            "unpaired_shuffle",
            "species_paired",
        ],
        "query_only_msa_probability": 0.2,
        "pair_msa_as_unpaired_probability": 0.3,
        "species_paired_msa_probability": 0.5,
        "unpaired_shuffle_flags": {
            "pair_msa_as_unpaired_only": True,
            "shuffle_pair_msa_when_unpaired_only": True,
        },
        "species_paired_flags": {
            "pair_msa_as_unpaired_only": False,
            "shuffle_pair_msa_when_unpaired_only": False,
        },
        "template_use_probability": 0.2,
        "legacy_template_dropout_rate": 0.0,
        "effective_by_dataset": {
            "weightedPDB_before210930_v20260101": {
                "query_only_msa_probability": 0.2,
                "pair_msa_as_unpaired_probability": 0.3,
                "species_paired_msa_probability": 0.5,
                "template_use_probability": 0.2,
                "legacy_template_dropout_rate": 0.0,
            }
        },
    }
    assert raw_contract["model_architecture_contract"]["pairformer"][
        "n_blocks"
    ] == 64
    assert raw_contract["model_architecture_contract"]["diffusion_module"][
        "c_token"
    ] == 768
    assert len(raw_contract["model_architecture_sha256"]) == 64


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("optimizer", "optimizer"),
        ("scheduler", "scheduler"),
        ("grad_scaler", "GradScaler"),
        ("rank_runtime_states", "per-rank RNG and dataloader"),
    ],
)
def test_strict_raw_rejects_missing_training_state(monkeypatch, field, message):
    checkpoint = _raw_checkpoint(monkeypatch)
    checkpoint.pop(field)

    with pytest.raises(ValueError, match=message):
        _validate(checkpoint, "raw")


def test_strict_raw_rejects_incomplete_rng_or_data_cursor(monkeypatch):
    checkpoint = _raw_checkpoint(monkeypatch)
    checkpoint["rank_runtime_states"]["states"][0]["rng"].pop("numpy")
    with pytest.raises(ValueError, match="incomplete RNG"):
        _validate(checkpoint, "raw")

    checkpoint = _raw_checkpoint(monkeypatch)
    checkpoint["rank_runtime_states"]["states"][0].pop("train_dataloader")
    with pytest.raises(ValueError, match="missing dataloader"):
        _validate(checkpoint, "raw")


def test_strict_multiworker_resume_restores_cursor_without_claiming_worker_rng(
    monkeypatch,
):
    checkpoint = _raw_checkpoint(monkeypatch)
    checkpoint["rank_runtime_states"] = _rank_runtime_state(
        next_local_offset=4,
    )

    contract = _validate(checkpoint, "raw")

    assert contract["data_resume_scope"] == (
        "sampler_cursor_and_rank_main_process_rng"
    )
    assert contract["data_worker_rng_state_checkpointed"] is False


def test_strict_resume_rejects_mismatched_raw_and_ema_pair(monkeypatch):
    raw_contract = _validate(_raw_checkpoint(monkeypatch), "raw")
    ema_contract = copy.deepcopy(_validate(_ema_checkpoint(monkeypatch), "ema"))
    ema_contract["checkpoint_pair_id"] = "f" * 64

    with pytest.raises(ValueError, match="not a matching pair"):
        validate_matching_resume_contracts(raw_contract, ema_contract)


def test_strict_resume_rejects_legacy_or_wrong_training_contract(monkeypatch):
    checkpoint = _raw_checkpoint(monkeypatch)
    checkpoint.pop("resume_contract")
    with pytest.raises(ValueError, match="missing resume_contract"):
        _validate(checkpoint, "raw")

    checkpoint = _raw_checkpoint(monkeypatch)
    checkpoint["resume_contract"]["training_contract_id"] = (
        "crop384_db48_ddp_scratch_50k_ws128_acc4_data_"
        "weightedpdb_only_v1_arch_pf64_cs384_dct768_pfckpt8_c48_"
        "msamix_q20_u30_p50_v7"
    )
    with pytest.raises(ValueError, match="training contract mismatch"):
        _validate(checkpoint, "raw")

    checkpoint = _raw_checkpoint(monkeypatch)
    checkpoint["resume_contract"]["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        _validate(checkpoint, "raw")


def test_strict_resume_rejects_changed_data_profile_or_weights(monkeypatch):
    checkpoint = _raw_checkpoint(monkeypatch)
    checkpoint["resume_contract"]["data_profile_id"] = "future_distillation_v1"
    _rehash_contract(checkpoint["resume_contract"])
    with pytest.raises(ValueError, match="data_profile_id"):
        _validate(checkpoint, "raw")

    checkpoint = _raw_checkpoint(monkeypatch)
    checkpoint["resume_contract"]["data_train_sample_weights"] = [0.5, 0.5]
    _rehash_contract(checkpoint["resume_contract"])
    with pytest.raises(ValueError, match="data_train_sample_weights"):
        _validate(checkpoint, "raw")


def test_strict_resume_rejects_changed_dataloader_runtime(monkeypatch):
    for field, value in (
        ("data_train_dl_persistent_workers", False),
        ("data_train_dl_prefetch_factor", 4),
        ("data_train_dl_pin_memory", True),
    ):
        checkpoint = _raw_checkpoint(monkeypatch)
        checkpoint["resume_contract"][field] = value
        _rehash_contract(checkpoint["resume_contract"])
        with pytest.raises(ValueError, match=field):
            _validate(checkpoint, "raw")


def test_architecture_contract_is_canonical_and_rejects_old_pairformer_depth(
    monkeypatch,
):
    current = build_model_architecture_contract(_configs())
    duplicate = build_model_architecture_contract(_configs())
    assert current == duplicate
    assert current["pairformer"]["n_blocks"] == 64

    checkpoint = _raw_checkpoint(monkeypatch)
    old_architecture = checkpoint["resume_contract"][
        "model_architecture_contract"
    ]
    old_architecture["pairformer"]["n_blocks"] = 48
    _rehash_architecture_contract(old_architecture)
    checkpoint["resume_contract"]["model_architecture_sha256"] = old_architecture[
        "sha256"
    ]
    _rehash_contract(checkpoint["resume_contract"])

    with pytest.raises(ValueError, match="model architecture mismatch"):
        _validate(checkpoint, "raw")


def test_strict_resume_requires_exact_world_size_and_all_rank_states(
    monkeypatch,
):
    checkpoint = _raw_checkpoint(monkeypatch)
    monkeypatch.setattr("runner.train.dist.is_available", lambda: True)
    monkeypatch.setattr("runner.train.dist.is_initialized", lambda: True)
    monkeypatch.setattr("runner.train.dist.get_world_size", lambda: 2)

    with pytest.raises(ValueError, match="original world size"):
        _validate(checkpoint, "raw")

    checkpoint["resume_contract"]["world_size"] = 2
    _rehash_contract(checkpoint["resume_contract"])
    checkpoint["rank_runtime_states"]["world_size"] = 2
    with pytest.raises(ValueError, match="rank-state count"):
        _validate(checkpoint, "raw")


@pytest.mark.parametrize(
    "overrides",
    [
        {"load_params_only": True},
        {"skip_load_optimizer": True},
        {"skip_load_scheduler": True},
        {"skip_load_step": True},
        {"load_step_for_scheduler": True},
    ],
)
def test_strict_configuration_rejects_every_weights_only_escape(overrides):
    with pytest.raises(ValueError, match="state-skipping options"):
        validate_strict_full_resume_configuration(_configs(**overrides))


def test_strict_configuration_requires_both_paths_and_ddp():
    with pytest.raises(ValueError, match="load_ema_checkpoint_path"):
        validate_strict_full_resume_configuration(
            _configs(load_ema_checkpoint_path="")
        )
    with pytest.raises(ValueError, match="requires DDP"):
        validate_strict_full_resume_configuration(
            _configs(distributed_strategy="fsdp2")
        )


def test_strict_try_load_checkpoint_round_trip_restores_pair_and_runtime(
    tmp_path,
    monkeypatch,
):
    source = AF3Trainer.__new__(AF3Trainer)
    source.model = torch.nn.Linear(2, 1)
    source.optimizer = torch.optim.Adam(source.model.parameters(), lr=1e-3)
    source.model(torch.ones(1, 2)).sum().backward()
    source.optimizer.step()
    source.optimizer.zero_grad(set_to_none=True)
    source.lr_scheduler = _Scheduler(last_epoch=4)
    source.grad_scaler = torch.GradScaler(device="cpu", enabled=False)
    source.configs = _configs(data=SimpleNamespace(num_dl_workers=0))
    source.device = torch.device("cpu")
    source.step = 4
    source.global_step = 19
    source.iters_to_accumulate = 4
    source.run_name = "crop384_source_20260724_000000"
    source.train_dl = _RuntimeLoader(consumed=20)
    source.use_fsdp = False
    source.use_ddp = False
    source.checkpoint_dir = str(tmp_path)
    source.checkpoint_index_path = str(tmp_path / "checkpoints.jsonl")
    source.latest_checkpoint_path = None
    source.latest_ema_checkpoint_path = None
    source.update_run_status = lambda **kwargs: None
    source.print = lambda message: None
    monkeypatch.setattr("runner.train.DIST_WRAPPER.rank", 0)

    source.save_checkpoint()
    source.save_checkpoint("_ema_0.999")
    raw_path = tmp_path / "4.pt"
    ema_path = tmp_path / "4_ema_0.999.pt"
    expected_model = {
        name: value.detach().clone()
        for name, value in source.model.state_dict().items()
    }

    restored = AF3Trainer.__new__(AF3Trainer)
    restored.model = torch.nn.Linear(2, 1)
    restored.optimizer = torch.optim.Adam(restored.model.parameters(), lr=1e-3)
    restored.lr_scheduler = _Scheduler(last_epoch=-1)
    restored.grad_scaler = torch.GradScaler(device="cpu", enabled=False)
    restored.configs = _configs(
        load_checkpoint_path=str(raw_path),
        load_ema_checkpoint_path=str(ema_path),
        data=SimpleNamespace(num_dl_workers=0),
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
    assert restored.global_step == 20
    assert restored.lr_scheduler.last_epoch == 4
    assert restored.train_dl.loaded_state is not None
    assert restored.train_dl.loaded_state["consumed_global_batches"] == 20
    assert restored.ema_wrapper.shadow is not None
    for name, expected in expected_model.items():
        assert torch.equal(restored.model.state_dict()[name], expected)
        assert torch.equal(restored.ema_wrapper.shadow[name], expected)
    assert restored.optimizer.state_dict()["state"]
