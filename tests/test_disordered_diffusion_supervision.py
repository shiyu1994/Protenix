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

"""Contracts for AF3-style disordered diffusion supervision."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray

from protenix.data.pipeline.dataset import (
    _count_diffusion_supervised_feature_tokens,
    _diffusion_supervised_token_mask,
    _resolved_protein_token_mask,
    _strict_bool_option,
    _strict_nonnegative_int_option,
)
from protenix.data.pipeline.data_pipeline import DataPipeline
from protenix.data.tokenizer import Token, TokenArray
from protenix.model.loss import (
    MSELoss,
    SmoothLDDTLoss,
    get_diffusion_supervision_masks,
)
from protenix.utils.cropping import CropData
from protenix.utils.permutation import permutation as permutation_module
from protenix.utils.permutation.chain_permutation.heuristic import (
    _correct_symmetric_chains_for_one_sample,
)
from protenix.utils.permutation.permutation import SymmetricPermutation


def _mask_inputs(flag: bool | None):
    feat_dict = {
        "is_protein": torch.tensor([1, 1, 0, 0]),
    }
    if flag is not None:
        feat_dict["disable_non_protein_diffusion_weights"] = torch.tensor(
            [flag], dtype=torch.bool
        )
    label_dict = {
        "coordinate_mask": torch.tensor([1, 0, 1, 1]),
        "lddt_mask": torch.ones((4, 4)),
    }
    return feat_dict, label_dict


@pytest.mark.parametrize("flag", (False, None))
def test_diffusion_supervision_mask_preserves_legacy_semantics(flag):
    feat_dict, label_dict = _mask_inputs(flag)
    original_lddt_mask = label_dict["lddt_mask"].clone()

    atom_mask, lddt_mask = get_diffusion_supervision_masks(
        feat_dict,
        label_dict,
    )

    assert torch.equal(atom_mask, label_dict["coordinate_mask"])
    expected_pair_mask = (
        atom_mask[..., None] * atom_mask[..., None, :]
    ).to(lddt_mask.dtype)
    assert torch.equal(lddt_mask, original_lddt_mask * expected_pair_mask)
    assert torch.equal(label_dict["lddt_mask"], original_lddt_mask)


def test_diffusion_supervision_mask_keeps_only_resolved_protein_atoms():
    feat_dict, label_dict = _mask_inputs(True)

    atom_mask, lddt_mask = get_diffusion_supervision_masks(
        feat_dict,
        label_dict,
    )

    assert torch.equal(atom_mask, torch.tensor([1, 0, 0, 0]))
    assert torch.equal(
        lddt_mask,
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_mse_supervision_mask_does_not_change_rigid_alignment_mask():
    loss_fn = MSELoss()
    captured = {}

    def fake_weighted_rigid_align(
        _self,
        pred_coordinate,
        true_coordinate,
        coordinate_mask,
        **_kwargs,
    ):
        captured["coordinate_mask"] = coordinate_mask.clone()
        aligned = true_coordinate.unsqueeze(-3).expand_as(pred_coordinate)
        return aligned, torch.ones_like(coordinate_mask)

    loss_fn.weighted_rigid_align = types.MethodType(
        fake_weighted_rigid_align,
        loss_fn,
    )
    true_coordinate = torch.zeros((2, 3))
    pred_coordinate = torch.tensor(
        [[[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]]]
    )
    coordinate_mask = torch.ones(2)
    protein_only_mask = torch.tensor([1.0, 0.0])

    protein_only_loss = loss_fn(
        pred_coordinate=pred_coordinate,
        true_coordinate=true_coordinate,
        coordinate_mask=coordinate_mask,
        is_dna=torch.zeros(2),
        is_rna=torch.zeros(2),
        is_ligand=torch.tensor([0, 1]),
        supervision_atom_mask=protein_only_mask,
    )

    assert torch.equal(captured["coordinate_mask"], coordinate_mask)
    assert torch.isclose(protein_only_loss, torch.tensor(1.0 / 3.0))


def test_sparse_smooth_lddt_empty_mask_is_finite_and_graph_connected():
    pred_coordinate = torch.randn((2, 3, 3), requires_grad=True)
    true_coordinate = torch.randn((3, 3))

    loss = SmoothLDDTLoss().sparse_forward(
        pred_coordinate=pred_coordinate,
        true_coordinate=true_coordinate,
        lddt_mask=torch.zeros((3, 3)),
    )

    assert torch.equal(loss, torch.tensor(1.0))
    loss.backward()
    assert pred_coordinate.grad is not None
    assert torch.count_nonzero(pred_coordinate.grad) == 0


def test_resolved_protein_token_mask_checks_all_atoms_not_only_centre():
    first = Token(0)
    first.atom_indices = [0, 1]
    first.centre_atom_index = 0
    second = Token(0)
    second.atom_indices = [2]
    second.centre_atom_index = 2
    token_array = TokenArray([first, second])
    atom_array = AtomArray(3)
    atom_array.set_annotation("is_protein", np.asarray([1, 1, 0]))
    atom_array.set_annotation("is_resolved", np.asarray([0, 1, 1]))

    mask = _resolved_protein_token_mask(token_array, atom_array)

    assert torch.equal(mask, torch.tensor([True, False]))


def test_diffusion_supervised_token_mask_uses_effective_loss_policy():
    first = Token(0)
    first.atom_indices = [0]
    first.centre_atom_index = 0
    second = Token(1)
    second.atom_indices = [1]
    second.centre_atom_index = 1
    token_array = TokenArray([first, second])
    atom_array = AtomArray(2)
    atom_array.set_annotation("is_protein", np.asarray([1, 0]))
    atom_array.set_annotation("is_resolved", np.asarray([0, 1]))

    all_resolved = _diffusion_supervised_token_mask(
        token_array,
        atom_array,
        protein_only=False,
    )
    protein_only = _diffusion_supervised_token_mask(
        token_array,
        atom_array,
        protein_only=True,
    )

    assert torch.equal(all_resolved, torch.tensor([False, True]))
    assert torch.equal(protein_only, torch.tensor([False, False]))


def test_final_feature_label_count_uses_unique_effective_tokens():
    features = {
        "atom_to_token_idx": torch.tensor([0, 0, 1, 2, 3, 3]),
        "is_protein": torch.tensor([1, 1, 1, 0, 1, 1]),
    }
    labels = {
        "coordinate_mask": torch.tensor([1, 1, 0, 1, 1, 0]),
    }

    assert (
        _count_diffusion_supervised_feature_tokens(
            features,
            labels,
            protein_only=False,
        )
        == 3
    )
    assert (
        _count_diffusion_supervised_feature_tokens(
            features,
            labels,
            protein_only=True,
        )
        == 2
    )


def _mini_rollout_permutation_fixture():
    configs = types.SimpleNamespace(
        chain_permutation=types.SimpleNamespace(
            train=types.SimpleNamespace(mini_rollout=True),
            configs={},
        ),
        atom_permutation=types.SimpleNamespace(
            train=types.SimpleNamespace(mini_rollout=True),
            global_align_wo_symmetric_atom=False,
        ),
    )
    input_features = {
        "atom_to_token_idx": torch.arange(4),
        "is_protein": torch.ones(4, dtype=torch.bool),
        "disable_non_protein_diffusion_weights": torch.tensor([True]),
        "min_diffusion_supervised_tokens": torch.tensor([4]),
        "ref_space_uid": torch.arange(4),
        "atom_perm_list": [[0], [1], [2], [3]],
    }
    labels = {
        "coordinate": torch.zeros((4, 3)),
        "coordinate_mask": torch.ones(4, dtype=torch.int64),
    }
    return configs, input_features, labels


def test_chain_permutation_cannot_reduce_supervision_below_minimum(
    monkeypatch,
):
    configs, input_features, labels = _mini_rollout_permutation_fixture()

    monkeypatch.setattr(
        permutation_module.chain_permutation,
        "run",
        lambda *_args, **_kwargs: (
            {
                "coordinate": torch.ones((4, 3)),
                "coordinate_mask": torch.tensor([1, 0, 0, 0]),
            },
            {},
            [],
            [],
        ),
    )
    monkeypatch.setattr(
        permutation_module.atom_permutation,
        "run",
        lambda **_kwargs: ({}, {}, []),
    )

    output, log = SymmetricPermutation(
        configs
    ).permute_label_to_match_mini_rollout(
        mini_coord=torch.zeros((1, 4, 3)),
        input_feature_dict=input_features,
        label_dict=labels,
        label_full_dict={},
    )

    assert int(output["coordinate_mask"].sum()) == 4
    assert torch.equal(output["coordinate"], torch.zeros((4, 3)))
    assert log["minirollout_perm/Chain-min_supervision_rejected"] == 1.0
    assert log["minirollout_perm/Chain-rejected_supervised_tokens"] == 1.0
    assert log["minirollout_perm/N_diffusion_supervised_token"] == 4.0


def test_atom_permutation_cannot_reduce_supervision_below_minimum(
    monkeypatch,
):
    configs, input_features, labels = _mini_rollout_permutation_fixture()

    monkeypatch.setattr(
        permutation_module.chain_permutation,
        "run",
        lambda *_args, **_kwargs: ({}, {}, [], []),
    )
    monkeypatch.setattr(
        permutation_module.atom_permutation,
        "run",
        lambda **_kwargs: (
            {
                "coordinate": torch.ones((4, 3)),
                "coordinate_mask": torch.tensor([1, 1, 1, 0]),
            },
            {},
            [],
        ),
    )

    output, log = SymmetricPermutation(
        configs
    ).permute_label_to_match_mini_rollout(
        mini_coord=torch.zeros((1, 4, 3)),
        input_feature_dict=input_features,
        label_dict=labels,
        label_full_dict={},
    )

    assert int(output["coordinate_mask"].sum()) == 4
    assert torch.equal(output["coordinate"], torch.zeros((4, 3)))
    assert log["minirollout_perm/Atom-min_supervision_rejected"] == 1.0
    assert log["minirollout_perm/Atom-rejected_supervised_tokens"] == 3.0
    assert log["minirollout_perm/N_diffusion_supervised_token"] == 4.0


def test_real_chain_heuristic_low_resolution_match_is_rejected(monkeypatch):
    pred = {
        "coordinate": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        ),
        "entity_mol_id": torch.tensor([0, 0, 0, 0]),
        "mol_id": torch.tensor([0, 0, 0, 0]),
        "mol_atom_index": torch.tensor([0, 1, 2, 3]),
        "pae_rep_atom_mask": torch.ones(4, dtype=torch.int64),
        "is_ligand": torch.zeros(4, dtype=torch.int64),
    }
    label_full = {
        "coordinate": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [0.0, 100.0, 0.0],
                [0.0, 0.0, 100.0],
            ]
        ),
        "coordinate_mask": torch.tensor([1, 1, 1, 1, 1, 0, 0, 0]),
        "entity_mol_id": torch.tensor([0] * 8),
        "mol_id": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        "mol_atom_index": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
        "pae_rep_atom_mask": torch.ones(8, dtype=torch.int64),
    }

    (
        best_match,
        _pred_indices,
        label_indices,
        raw_output,
        _raw_log,
    ) = _correct_symmetric_chains_for_one_sample(
        pred,
        label_full,
        max_num_chains=20,
        permute_label=True,
        use_center_rmsd=False,
        find_gt_anchor_first=False,
        accept_it_as_it_is=False,
    )
    assert best_match == {0: 1}
    assert torch.equal(label_indices, torch.tensor([4, 5, 6, 7]))
    assert int(raw_output["coordinate_mask"].sum()) == 1

    configs, minimum_features, labels = (
        _mini_rollout_permutation_fixture()
    )
    configs.chain_permutation.configs = {
        "use_center_rmsd": False,
        "find_gt_anchor_first": False,
        "accept_it_as_it_is": False,
    }
    minimum_features.update(
        {
            key: value
            for key, value in pred.items()
            if key != "coordinate"
        }
    )
    monkeypatch.setattr(
        permutation_module.atom_permutation,
        "run",
        lambda **_kwargs: ({}, {}, []),
    )

    output, log = SymmetricPermutation(
        configs
    ).permute_label_to_match_mini_rollout(
        mini_coord=pred["coordinate"].unsqueeze(0),
        input_feature_dict=minimum_features,
        label_dict=labels,
        label_full_dict=label_full,
    )

    assert int(output["coordinate_mask"].sum()) == 4
    assert log["minirollout_perm/Chain-min_supervision_rejected"] == 1.0
    assert log["minirollout_perm/Chain-rejected_supervised_tokens"] == 1.0


def test_mini_rollout_rejects_invalid_incoming_supervision():
    configs, input_features, labels = _mini_rollout_permutation_fixture()
    labels["coordinate_mask"] = torch.tensor([1, 1, 1, 0])

    with pytest.raises(RuntimeError, match="before mini-rollout permutation"):
        SymmetricPermutation(configs).permute_label_to_match_mini_rollout(
            mini_coord=torch.zeros((1, 4, 3)),
            input_feature_dict=input_features,
            label_dict=labels,
            label_full_dict={},
        )


def test_new_dataset_booleans_reject_truthy_strings():
    with pytest.raises(TypeError, match="must be a boolean"):
        _strict_bool_option(
            {"require_resolved_protein_crop": "false"},
            "require_resolved_protein_crop",
        )


@pytest.mark.parametrize("value", (True, 4.0, "4"))
def test_minimum_diffusion_tokens_rejects_non_integer_values(value):
    with pytest.raises(TypeError, match="non-negative integer"):
        _strict_nonnegative_int_option(
            {"min_diffusion_supervised_tokens": value},
            "min_diffusion_supervised_tokens",
        )


def test_minimum_diffusion_tokens_rejects_negative_value():
    with pytest.raises(ValueError, match="must be >= 0"):
        _strict_nonnegative_int_option(
            {"min_diffusion_supervised_tokens": -1},
            "min_diffusion_supervised_tokens",
        )


def _crop_fixture(
    required_mask: torch.Tensor,
    crop_size: int = 3,
    min_required_tokens: int = 1,
) -> CropData:
    tokens = []
    for atom_index in range(5):
        token = Token(atom_index)
        token.atom_indices = [atom_index]
        token.centre_atom_index = atom_index
        tokens.append(token)
    token_array = TokenArray(tokens)
    atom_array = AtomArray(5)
    atom_array.coord = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    atom_array.set_annotation(
        "ref_space_uid",
        np.asarray([10, 20, 20, 30, 40]),
    )
    return CropData(
        crop_size=crop_size,
        ref_chain_indices=[0],
        token_array=token_array,
        atom_array=atom_array,
        required_token_mask=required_mask,
        min_required_tokens=min_required_tokens,
    )


def test_crop_inserts_required_token_without_fragmenting_uid_group():
    crop = _crop_fixture(torch.tensor([0, 0, 0, 0, 1]))

    selected = crop._ensure_required_token(
        torch.tensor([0, 1, 2]),
        reference_token_index=0,
    )

    assert 4 in selected.tolist()
    assert not ({1, 2} & set(selected.tolist()))
    assert len(selected) <= crop.crop_size


def test_crop_rejects_required_mask_without_eligible_token():
    with pytest.raises(ValueError, match="too few eligible tokens"):
        _crop_fixture(torch.zeros(5, dtype=torch.bool))


def test_crop_enforces_four_required_tokens_before_feature_building():
    crop = _crop_fixture(
        torch.tensor([1, 0, 1, 1, 1]),
        crop_size=4,
        min_required_tokens=4,
    )

    selected = crop._ensure_required_tokens(
        torch.tensor([0, 1, 2]),
        reference_token_index=0,
    )

    assert selected.tolist() == [0, 2, 3, 4]
    assert int(crop.required_token_mask[selected].sum()) == 4
    assert len(selected) <= crop.crop_size


def test_crop_rejects_source_with_fewer_than_four_required_tokens():
    with pytest.raises(ValueError, match="3 < 4"):
        _crop_fixture(
            torch.tensor([1, 1, 1, 0, 0]),
            crop_size=4,
            min_required_tokens=4,
        )


def test_supervision_minimum_is_checked_before_msa_io(monkeypatch):
    fixture = _crop_fixture(
        torch.tensor([1, 1, 1, 0, 1]),
        crop_size=4,
    )
    monkeypatch.setattr(
        DataPipeline,
        "_map_ref_chain",
        staticmethod(lambda **_kwargs: [0]),
    )
    monkeypatch.setattr(
        CropData,
        "random_crop_method",
        lambda _self: "ContiguousCropping",
    )
    monkeypatch.setattr(
        CropData,
        "get_crop_indices",
        lambda _self, crop_method: (torch.tensor([0, 1, 2, 3]), 0),
    )

    def fail_if_msa_is_read(**_kwargs):
        raise AssertionError("MSA featurizer must not run")

    with pytest.raises(RuntimeError, match="before MSA I/O"):
        DataPipeline.crop(
            one_sample={},
            bioassembly_dict={
                "token_array": fixture.token_array,
                "atom_array": fixture.atom_array,
            },
            crop_size=4,
            msa_featurizer=fail_if_msa_is_read,
            template_featurizer=None,
            supervision_token_mask=torch.tensor([1, 1, 1, 0, 1]),
            min_supervision_tokens=4,
        )


def test_crop_hard_requirement_overrides_single_reference_uid():
    crop = _crop_fixture(
        torch.tensor([0, 0, 0, 0, 1]),
        crop_size=2,
    )

    selected = crop._ensure_required_token(
        torch.tensor([1, 2]),
        reference_token_index=1,
    )

    assert selected.tolist() == [4]


def test_crop_uses_resolved_atom_when_required_token_centre_is_nan():
    tokens = []
    for atom_index in range(3):
        token = Token(atom_index)
        token.atom_indices = [atom_index]
        token.centre_atom_index = atom_index
        tokens.append(token)
    required_token = Token(3)
    required_token.atom_indices = [3, 4]
    required_token.centre_atom_index = 3
    tokens.append(required_token)
    token_array = TokenArray(tokens)
    token_array.set_annotation(
        "centre_atom_index",
        np.asarray([0, 1, 2, 3]),
    )
    atom_array = AtomArray(5)
    atom_array.coord = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [np.nan, np.nan, np.nan],
            [2.0, 0.0, 0.0],
        ]
    )
    atom_array.set_annotation(
        "ref_space_uid",
        np.asarray([10, 20, 30, 40, 40]),
    )
    atom_array.set_annotation(
        "is_resolved",
        np.asarray([1, 1, 1, 0, 1]),
    )
    atom_array.set_annotation(
        "is_protein",
        np.asarray([0, 0, 0, 1, 1]),
    )
    crop = CropData(
        crop_size=3,
        ref_chain_indices=[0],
        token_array=token_array,
        atom_array=atom_array,
        required_token_mask=torch.tensor([0, 0, 0, 1]),
    )

    selected = crop._ensure_required_token(torch.tensor([0, 1, 2]))

    assert selected.tolist() == [0, 1, 3]
