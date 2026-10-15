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

import numpy as np
import pytest

from protenix.data.constants import PROTEIN_CHAIN, RNA_CHAIN
from protenix.data.msa.msa_featurizer import (
    FeatureAssemblyLine,
    MSA_SHUFFLE_VARIANT_NONE,
    MSA_SHUFFLE_VARIANT_SEGMENT,
    MSA_SHUFFLE_VARIANT_WHOLE_PHYSICAL_CHAIN,
    MSAFeaturizer,
    build_virtual_msa_segment_groups,
    build_whole_physical_chain_msa_groups,
    select_multichain_msa_shuffle_variant,
    shuffle_msa_feature_rows_by_segment,
)
from protenix.data.msa.msa_utils import MSACore, RawMsa
from protenix.data.pipeline.sample_context import (
    SampleOccurrence,
    deterministic_occurrence_seed,
)


_PROTEIN_QUERY = "A" * 18
_PROTEIN_HITS = tuple(
    (
        fill * insertion_position
        + fill.lower() * insertion_position
        + fill * (18 - insertion_position)
    )
    for fill, insertion_position in zip("CDEFGH", range(1, 7))
)


class _FakeTokenArray:
    def __init__(self, centre_atom_indices):
        self.centre_atom_indices = np.asarray(
            centre_atom_indices,
            dtype=np.int64,
        )

    def __getitem__(self, indices):
        return _FakeTokenArray(self.centre_atom_indices[indices])

    def get_annotation(self, name):
        if name != "centre_atom_index":
            raise KeyError(name)
        return self.centre_atom_indices


class _FakeAtomArray:
    def __init__(self, asym_id_int, chain_id, res_id):
        self.asym_id_int = np.asarray(asym_id_int, dtype=np.int64)
        self.chain_id = np.asarray(chain_id)
        self.res_id = np.asarray(res_id, dtype=np.int64)

    def __getitem__(self, indices):
        return _FakeAtomArray(
            self.asym_id_int[indices],
            self.chain_id[indices],
            self.res_id[indices],
        )


def _single_chain_bioassembly() -> dict[int, dict[str, str]]:
    query = _PROTEIN_QUERY

    def a3m(rows: tuple[tuple[str, str], ...]) -> str:
        records = [">query", query]
        for name, sequence in rows:
            records.extend((f">{name}", sequence))
        return "\n".join(records) + "\n"

    return {
        0: {
            "entity_id": "1",
            "chain_id": "A",
            "sequence": query,
            "chain_entity_type": PROTEIN_CHAIN,
            "paired_msa": a3m(
                (
                    ("pair_1", _PROTEIN_HITS[0]),
                    ("pair_2", _PROTEIN_HITS[1]),
                    ("pair_3", _PROTEIN_HITS[2]),
                )
            ),
            "unpaired_msa": a3m(
                (
                    ("unpaired_1", _PROTEIN_HITS[3]),
                    ("unpaired_2", _PROTEIN_HITS[4]),
                    ("unpaired_3", _PROTEIN_HITS[5]),
                )
            ),
        }
    }


@pytest.mark.parametrize(
    (
        "protein_asym_ids",
        "sampling_enabled",
        "segment_probability",
        "variant_seed",
        "expected_variant",
        "expected_eligible",
    ),
    [
        ([], True, 0.5, 0, MSA_SHUFFLE_VARIANT_NONE, False),
        ([1], True, 0.0, None, MSA_SHUFFLE_VARIANT_SEGMENT, False),
        ([1, 2], False, 0.0, None, MSA_SHUFFLE_VARIANT_SEGMENT, False),
        ([1, 2], True, 1.0, None, MSA_SHUFFLE_VARIANT_SEGMENT, True),
        (
            [1, 2],
            True,
            0.0,
            None,
            MSA_SHUFFLE_VARIANT_WHOLE_PHYSICAL_CHAIN,
            True,
        ),
        ([1, 2], True, 0.5, 0, MSA_SHUFFLE_VARIANT_SEGMENT, True),
        (
            [1, 2],
            True,
            0.5,
            (1 << 64) - 1,
            MSA_SHUFFLE_VARIANT_WHOLE_PHYSICAL_CHAIN,
            True,
        ),
    ],
)
def test_multichain_shuffle_variant_selection(
    protein_asym_ids,
    sampling_enabled,
    segment_probability,
    variant_seed,
    expected_variant,
    expected_eligible,
):
    assert select_multichain_msa_shuffle_variant(
        protein_asym_ids=protein_asym_ids,
        sampling_enabled=sampling_enabled,
        segment_probability=segment_probability,
        variant_seed=variant_seed,
    ) == (expected_variant, expected_eligible)


def test_nontrivial_multichain_variant_requires_independent_seed():
    with pytest.raises(TypeError, match="variant_seed"):
        select_multichain_msa_shuffle_variant(
            protein_asym_ids=[1, 2],
            sampling_enabled=True,
            segment_probability=0.5,
            variant_seed=None,
        )


@pytest.mark.parametrize("probability", [True, np.nan, -0.1, 1.1])
def test_invalid_multichain_segment_probabilities_are_rejected(probability):
    expected_error = TypeError if probability is True else ValueError
    with pytest.raises(expected_error):
        select_multichain_msa_shuffle_variant(
            protein_asym_ids=[1, 2],
            sampling_enabled=True,
            segment_probability=probability,
            variant_seed=0,
        )


def test_multichain_variant_constructor_is_fail_closed():
    common = {
        "prot_seq_or_filename_to_msadir_jsons": [],
        "prot_msadir_raw_paths": [],
        "rna_seq_or_filename_to_msadir_jsons": [],
        "rna_msadir_raw_paths": [],
        "prot_pairing_dbs": [],
        "prot_non_pairing_dbs": [],
        "prot_indexing_methods": [],
        "rna_indexing_methods": [],
        "enable_prot_msa": False,
        "enable_rna_msa": False,
    }
    with pytest.raises(ValueError, match="requires"):
        MSAFeaturizer(
            **common,
            enable_multichain_msa_shuffle_variant_sampling=True,
        )
    with pytest.raises(TypeError, match="real number"):
        MSAFeaturizer(
            **common,
            multichain_msa_shuffle_segment_probability=True,
        )


def test_whole_physical_chain_groups_keep_homomer_copies_distinct():
    asym_ids = np.array([1, 1, 1, 1, 2, 2, 2, 3], dtype=np.int64)
    res_ids = np.array([1, 2, 2, 30, 1, 25, 25, 1], dtype=np.int64)

    groups = build_whole_physical_chain_msa_groups(
        asym_ids=asym_ids,
        res_ids=res_ids,
        protein_sequence_lengths={1: 40, 2: 30},
    )

    assert set(groups) == {1, 2}
    assert len(groups[1]) == len(groups[2]) == 1
    np.testing.assert_array_equal(groups[1][0], np.array([0, 1, 29]))
    np.testing.assert_array_equal(groups[2][0], np.array([0, 24]))


def test_segment_groups_count_unique_residues_and_merge_short_tie_left():
    left = np.arange(1, 10)
    short = np.arange(33, 41)
    right = np.arange(64, 73)
    chain_one_residues = np.concatenate(
        (left, short, np.repeat(35, 6), right)
    )
    chain_two_residues = np.array([1, 25])
    asym_ids = np.concatenate(
        (
            np.ones(chain_one_residues.size, dtype=np.int64),
            np.full(chain_two_residues.size, 2, dtype=np.int64),
        )
    )
    res_ids = np.concatenate((chain_one_residues, chain_two_residues))

    groups = build_virtual_msa_segment_groups(
        asym_ids=asym_ids,
        res_ids=res_ids,
        protein_sequence_lengths={1: 80, 2: 30},
        residue_gap=24,
        min_segment_length=9,
    )

    assert len(groups[1]) == 2
    np.testing.assert_array_equal(
        groups[1][0],
        np.concatenate((left, short)) - 1,
    )
    np.testing.assert_array_equal(groups[1][1], right - 1)
    # Neither short raw segment is eligible, so this physical chain remains
    # one group and is never merged into chain 1.
    assert len(groups[2]) == 1
    np.testing.assert_array_equal(groups[2][0], chain_two_residues - 1)


def test_segment_gap_23_stays_joined_and_gap_24_splits():
    joined = build_virtual_msa_segment_groups(
        asym_ids=np.ones(18, dtype=np.int64),
        res_ids=np.concatenate((np.arange(1, 10), np.arange(32, 41))),
        protein_sequence_lengths={1: 41},
    )
    split = build_virtual_msa_segment_groups(
        asym_ids=np.ones(18, dtype=np.int64),
        res_ids=np.concatenate((np.arange(1, 10), np.arange(33, 42))),
        protein_sequence_lengths={1: 42},
    )

    assert len(joined[1]) == 1
    assert len(split[1]) == 2


@pytest.mark.parametrize(
    ("short", "expected_group"),
    [
        (np.arange(33, 36), 0),
        (np.arange(60, 63), 1),
        (np.arange(1, 4), 0),
        (np.arange(91, 94), 0),
    ],
)
def test_short_segments_merge_to_the_nearest_eligible_group(
    short,
    expected_group,
):
    if short[0] < 10:
        eligible_segments = (np.arange(27, 36),)
    elif short[0] > 90:
        eligible_segments = (np.arange(1, 10),)
    else:
        eligible_segments = (np.arange(1, 10), np.arange(86, 95))
    residues = np.concatenate(
        tuple(sorted((*eligible_segments, short), key=lambda value: value[0]))
    )

    groups = build_virtual_msa_segment_groups(
        asym_ids=np.ones(residues.size, dtype=np.int64),
        res_ids=residues,
        protein_sequence_lengths={1: 100},
    )[1]

    assert len(groups) == len(eligible_segments)
    assert any(np.isin(short - 1, group).all() for group in groups)
    if len(eligible_segments) == 2:
        assert np.isin(short - 1, groups[expected_group]).all()


def test_segment_row_shuffle_is_synchronized_and_query_fixed():
    msa = np.arange(6, dtype=np.int64)[:, None] * 10
    msa = msa + np.arange(8, dtype=np.int64)[None, :]
    deletion_matrix = msa + 100
    segments = (np.array([0, 1, 2]), np.array([4, 5]))

    shuffled_msa, shuffled_deletion, attempted = (
        shuffle_msa_feature_rows_by_segment(
            msa=msa,
            deletion_matrix=deletion_matrix,
            segment_columns=segments,
            rng=np.random.default_rng(7),
        )
    )

    expected_rng = np.random.default_rng(7)
    for columns in segments:
        order = np.concatenate(
            (
                np.array([0]),
                expected_rng.permutation(np.arange(1, msa.shape[0])),
            )
        )
        np.testing.assert_array_equal(
            shuffled_msa[:, columns],
            msa[order][:, columns],
        )
        np.testing.assert_array_equal(
            shuffled_deletion[:, columns],
            deletion_matrix[order][:, columns],
        )
    assert attempted is True
    np.testing.assert_array_equal(shuffled_msa[0], msa[0])
    np.testing.assert_array_equal(shuffled_deletion[0], deletion_matrix[0])
    np.testing.assert_array_equal(shuffled_msa[:, 3], msa[:, 3])
    np.testing.assert_array_equal(shuffled_msa[:, 6:], msa[:, 6:])
    np.testing.assert_array_equal(
        shuffled_deletion - shuffled_msa,
        np.full_like(msa, 100),
    )


def test_invalid_inplace_segments_are_rejected_before_mutation():
    msa = np.arange(24, dtype=np.int32).reshape(4, 6)
    deletion = msa + 50
    original_msa = msa.copy()
    original_deletion = deletion.copy()

    with pytest.raises(ValueError, match="disjoint"):
        shuffle_msa_feature_rows_by_segment(
            msa=msa,
            deletion_matrix=deletion,
            segment_columns=(np.array([0, 1]), np.array([1, 2])),
            rng=np.random.default_rng(3),
            copy=False,
        )

    np.testing.assert_array_equal(msa, original_msa)
    np.testing.assert_array_equal(deletion, original_deletion)


def test_feature_assembly_shuffles_paired_and_unpaired_rows_by_segment():
    assembly = FeatureAssemblyLine(
        max_msa_size=32,
        pair_msa_as_unpaired_only=True,
        shuffle_pair_msa_when_unpaired_only=True,
        enable_virtual_segment_msa_shuffle=True,
    )
    features = assembly.assemble(
        _single_chain_bioassembly(),
        np.arange(18, dtype=np.int32),
        virtual_segment_groups={
            0: (np.arange(9), np.arange(9, 18)),
        },
        virtual_segment_shuffle_seed=0,
    ).to_dict()

    assert assembly.last_virtual_segment_count == 2
    assert assembly.last_virtual_segment_shuffle_applied is True
    source_msa, source_deletion = MSACore.sequences_to_array(
        [_PROTEIN_QUERY, *_PROTEIN_HITS],
        PROTEIN_CHAIN,
    )
    expected_msa = source_msa.copy()
    expected_deletion = source_deletion.copy()
    expected_rng = np.random.default_rng(0)
    for columns in (np.arange(9), np.arange(9, 18)):
        order = np.concatenate(
            (
                np.array([0]),
                expected_rng.permutation(np.arange(1, 7)),
            )
        )
        expected_msa[:, columns] = source_msa[order][:, columns]
        expected_deletion[:, columns] = source_deletion[order][:, columns]
    # Preserve the existing paired-query placeholder ahead of the assembled
    # chain MSA. The source query row and this placeholder both stay fixed.
    expected_msa = np.concatenate((source_msa[:1], expected_msa), axis=0)
    expected_deletion = np.concatenate(
        (source_deletion[:1], expected_deletion),
        axis=0,
    )
    np.testing.assert_array_equal(features["msa"], expected_msa)
    np.testing.assert_array_equal(
        features["deletion_matrix"],
        expected_deletion,
    )
    assert int(features["prot_pair_num_alignments"]) == 1
    assert int(features["prot_unpair_num_alignments"]) == 7


def test_whole_chain_grouping_couples_segments_but_keeps_chains_independent():
    def a3m(query, fills, prefix):
        records = [">query", query]
        for index, fill in enumerate(fills):
            records.extend((f">{prefix}_{index}", fill * len(query)))
        return "\n".join(records) + "\n"

    chain_a_query = "A" * 18
    chain_b_query = "V" * 18
    bioassembly = {
        0: {
            "entity_id": "1",
            "chain_id": "A",
            "sequence": chain_a_query,
            "chain_entity_type": PROTEIN_CHAIN,
            "paired_msa": a3m(chain_a_query, "CDE", "a_pair"),
            "unpaired_msa": a3m(chain_a_query, "FGH", "a_unpair"),
        },
        1: {
            "entity_id": "2",
            "chain_id": "B",
            "sequence": chain_b_query,
            "chain_entity_type": PROTEIN_CHAIN,
            "paired_msa": a3m(chain_b_query, "KLM", "b_pair"),
            "unpaired_msa": a3m(chain_b_query, "NPQ", "b_unpair"),
        },
    }
    common = {
        "bioassembly": bioassembly,
        "std_idxs": np.arange(36, dtype=np.int32),
        "virtual_segment_shuffle_seed": 0,
    }
    segment = FeatureAssemblyLine(
        max_msa_size=32,
        pair_msa_as_unpaired_only=True,
        shuffle_pair_msa_when_unpaired_only=True,
        enable_virtual_segment_msa_shuffle=True,
    ).assemble(
        **common,
        virtual_segment_groups={
            0: (np.arange(9), np.arange(9, 18)),
            1: (np.arange(9), np.arange(9, 18)),
        },
    ).to_dict()
    whole = FeatureAssemblyLine(
        max_msa_size=32,
        pair_msa_as_unpaired_only=True,
        shuffle_pair_msa_when_unpaired_only=True,
        enable_virtual_segment_msa_shuffle=True,
    ).assemble(
        **common,
        virtual_segment_groups={
            0: (np.arange(18),),
            1: (np.arange(18),),
        },
    ).to_dict()

    # All three paired and three unpaired hits participate in the same
    # non-query pool, while the target row remains fixed.
    assert whole["msa"].shape[0] == 7
    assert int(whole["prot_unpair_num_alignments"]) == 6
    expected_query, _ = MSACore.sequences_to_array(
        [chain_a_query + chain_b_query],
        PROTEIN_CHAIN,
    )
    np.testing.assert_array_equal(whole["msa"][0], expected_query[0])

    # Whole-chain mode gives both crop segments of one chain the same row
    # permutation. Segment mode draws independently, and the RNG advances
    # independently again for the second physical chain.
    np.testing.assert_array_equal(whole["msa"][:, 0], whole["msa"][:, 9])
    np.testing.assert_array_equal(whole["msa"][:, 18], whole["msa"][:, 27])
    assert np.any(segment["msa"][1:, 0] != segment["msa"][1:, 9])
    assert np.any(segment["msa"][1:, 18] != segment["msa"][1:, 27])
    assert np.any(whole["msa"][1:, 0] != whole["msa"][1:, 18])


def test_modified_residue_token_broadcast_remains_consistent_after_shuffle():
    assembly = FeatureAssemblyLine(
        max_msa_size=32,
        pair_msa_as_unpaired_only=True,
        shuffle_pair_msa_when_unpaired_only=True,
        enable_virtual_segment_msa_shuffle=True,
    )
    # Canonical residue 2 is represented by three atom tokens. This is the
    # same repeated std_idx mapping used for a modified amino acid.
    std_idxs = np.array(
        [0, 1, 1, 1, *range(2, 18)],
        dtype=np.int32,
    )
    features = assembly.assemble(
        _single_chain_bioassembly(),
        std_idxs,
        virtual_segment_groups={
            0: (np.arange(9), np.arange(9, 18)),
        },
        virtual_segment_shuffle_seed=3,
    ).to_dict()

    for name in ("msa", "deletion_matrix", "has_deletion", "deletion_value"):
        np.testing.assert_array_equal(
            features[name][:, 1],
            features[name][:, 2],
        )
        np.testing.assert_array_equal(
            features[name][:, 1],
            features[name][:, 3],
        )
    for name in ("profile", "deletion_mean"):
        np.testing.assert_array_equal(features[name][1], features[name][2])
        np.testing.assert_array_equal(features[name][1], features[name][3])


def test_virtual_protein_segments_do_not_reorder_rna_msa_columns():
    bioassembly = _single_chain_bioassembly()
    rna_query = "ACGUACGUA"
    bioassembly[1] = {
        "entity_id": "2",
        "chain_id": "R",
        "sequence": rna_query,
        "chain_entity_type": RNA_CHAIN,
        "paired_msa": "",
        "unpaired_msa": (
            f">query\n{rna_query}\n"
            ">rna_1\nACGUACGUC\n"
            ">rna_2\nACGUACGUG\n"
        ),
    }
    common = {
        "bioassembly": bioassembly,
        "std_idxs": np.arange(27, dtype=np.int32),
        "virtual_segment_shuffle_seed": 0,
    }
    baseline = FeatureAssemblyLine(
        max_msa_size=32,
        pair_msa_as_unpaired_only=True,
        shuffle_pair_msa_when_unpaired_only=True,
        enable_virtual_segment_msa_shuffle=True,
    ).assemble(
        **common,
        virtual_segment_groups={0: (np.arange(18),)},
    ).to_dict()
    augmented = FeatureAssemblyLine(
        max_msa_size=32,
        pair_msa_as_unpaired_only=True,
        shuffle_pair_msa_when_unpaired_only=True,
        enable_virtual_segment_msa_shuffle=True,
    ).assemble(
        **common,
        virtual_segment_groups={
            0: (np.arange(9), np.arange(9, 18)),
        },
    ).to_dict()

    assert np.any(baseline["msa"][:, :18] != augmented["msa"][:, :18])
    np.testing.assert_array_equal(
        baseline["msa"][:, 18:],
        augmented["msa"][:, 18:],
    )
    np.testing.assert_array_equal(
        baseline["deletion_matrix"][:, 18:],
        augmented["deletion_matrix"][:, 18:],
    )


def test_legacy_pdb_msa_loader_uses_crop_segments_for_all_protein_rows(
    monkeypatch,
):
    sequence = "A" * 42
    paired_a3m = (
        f">query\n{sequence}\n"
        f">paired_1\n{'C' * 42}\n"
        f">paired_2\n{'D' * 42}\n"
    )
    unpaired_a3m = (
        f">query\n{sequence}\n"
        f">unpaired_1\n{'E' * 42}\n"
        f">unpaired_2\n{'F' * 42}\n"
    )
    atom_array = _FakeAtomArray(
        asym_id_int=[1] * len(sequence),
        chain_id=["A"] * len(sequence),
        res_id=np.arange(1, len(sequence) + 1),
    )
    bioassembly = {
        "pdb_id": "legacy-pdb",
        "atom_array": atom_array,
        "token_array": _FakeTokenArray(np.arange(len(sequence))),
        "sequences": {"1": sequence},
        "entity_poly_type": {"1": "polypeptide(L)"},
    }
    featurizer = MSAFeaturizer(
        dataset_name="legacy-pdb-test",
        prot_seq_or_filename_to_msadir_jsons=[],
        prot_msadir_raw_paths=[],
        rna_seq_or_filename_to_msadir_jsons=[],
        rna_msadir_raw_paths=[],
        prot_pairing_dbs=[],
        prot_non_pairing_dbs=[],
        prot_indexing_methods=[],
        rna_indexing_methods=[],
        enable_prot_msa=True,
        enable_rna_msa=False,
        max_msa_size=16,
        enable_virtual_segment_msa_shuffle=True,
    )
    monkeypatch.setattr(
        featurizer.prot_mgr,
        "fetch_msas",
        lambda *args, **kwargs: (
            [RawMsa.from_a3m(sequence, PROTEIN_CHAIN, unpaired_a3m)],
            [
                RawMsa.from_a3m(
                    sequence,
                    PROTEIN_CHAIN,
                    paired_a3m,
                    dedup=False,
                )
            ],
        ),
    )
    selected_indices = np.concatenate((np.arange(9), np.arange(32, 41)))

    features = featurizer.make_msa_features(
        bioassembly_dict=bioassembly,
        selected_indices=selected_indices,
        entity_to_asym_id_int={"1": [1]},
        pair_msa_as_unpaired_only=True,
        shuffle_pair_msa_when_unpaired_only=True,
        virtual_segment_shuffle_seed=5,
    )

    profile = featurizer.get_last_profile()
    assert profile["virtual_segment_count"] == 2
    assert profile["virtual_segment_shuffle_applied"] is True
    assert (
        profile["msa_shuffle_requested_variant"]
        == MSA_SHUFFLE_VARIANT_SEGMENT
    )
    assert profile["msa_shuffle_variant_sampling_enabled"] is False
    assert profile["msa_shuffle_variant_eligible"] is False
    np.testing.assert_array_equal(features["msa"][0, :9], features["msa"][0, 9:])
    assert np.any(features["msa"][2:, 0] != features["msa"][2:, 9])


@pytest.mark.parametrize(
    (
        "variant_sampling_enabled",
        "segment_probability",
        "expected_variant",
        "expected_eligible",
        "expected_segment_count",
        "expected_group_count",
        "variant_seed",
    ),
    [
        (
            False,
            0.5,
            MSA_SHUFFLE_VARIANT_SEGMENT,
            False,
            4,
            4,
            17,
        ),
        (
            True,
            0.5,
            MSA_SHUFFLE_VARIANT_SEGMENT,
            True,
            4,
            4,
            0,
        ),
        (
            True,
            0.5,
            MSA_SHUFFLE_VARIANT_WHOLE_PHYSICAL_CHAIN,
            True,
            0,
            2,
            (1 << 64) - 1,
        ),
    ],
)
def test_msa_featurizer_wiring_preserves_modified_residue_tokens_per_asym(
    monkeypatch,
    variant_sampling_enabled,
    segment_probability,
    expected_variant,
    expected_eligible,
    expected_segment_count,
    expected_group_count,
    variant_seed,
):
    sequence = "A" * 42

    def a3m(prefix, fills):
        records = [">query", sequence]
        for index, fill in enumerate(fills, start=1):
            insertion_position = index + 1
            hit = (
                fill * insertion_position
                + fill.lower()
                + fill * (42 - insertion_position)
            )
            records.extend((f">{prefix}_{index}", hit))
        return "\n".join(records) + "\n"

    paired_a3m = a3m("paired", "CD")
    unpaired_a3m = a3m("unpaired", "EF")
    chain_a_res_ids = np.concatenate(
        (np.array([1, 2, 2, 2]), np.arange(3, 43))
    )
    chain_b_res_ids = np.arange(1, 43)
    asym_ids = np.concatenate(
        (
            np.ones(chain_a_res_ids.size, dtype=np.int64),
            np.full(chain_b_res_ids.size, 2, dtype=np.int64),
        )
    )
    res_ids = np.concatenate((chain_a_res_ids, chain_b_res_ids))
    chain_ids = np.concatenate(
        (
            np.full(chain_a_res_ids.size, "A"),
            np.full(chain_b_res_ids.size, "B"),
        )
    )
    atom_array = _FakeAtomArray(asym_ids, chain_ids, res_ids)
    bioassembly = {
        "pdb_id": "modified-two-asym",
        "atom_array": atom_array,
        "token_array": _FakeTokenArray(np.arange(res_ids.size)),
        "sequences": {"1": sequence},
        "entity_poly_type": {"1": "polypeptide(L)"},
    }
    def make_featurizer(*, sampling_enabled, probability):
        return MSAFeaturizer(
            dataset_name="modified-two-asym-test",
            prot_seq_or_filename_to_msadir_jsons=[],
            prot_msadir_raw_paths=[],
            rna_seq_or_filename_to_msadir_jsons=[],
            rna_msadir_raw_paths=[],
            prot_pairing_dbs=[],
            prot_non_pairing_dbs=[],
            prot_indexing_methods=[],
            rna_indexing_methods=[],
            enable_prot_msa=True,
            enable_rna_msa=False,
            max_msa_size=16,
            enable_virtual_segment_msa_shuffle=True,
            enable_multichain_msa_shuffle_variant_sampling=(
                sampling_enabled
            ),
            multichain_msa_shuffle_segment_probability=probability,
        )

    featurizer = make_featurizer(
        sampling_enabled=variant_sampling_enabled,
        probability=segment_probability,
    )

    def fetch_msas(*args, **kwargs):
        return (
            [RawMsa.from_a3m(sequence, PROTEIN_CHAIN, unpaired_a3m)],
            [
                RawMsa.from_a3m(
                    sequence,
                    PROTEIN_CHAIN,
                    paired_a3m,
                    dedup=False,
                )
            ],
        )

    monkeypatch.setattr(featurizer.prot_mgr, "fetch_msas", fetch_msas)
    selected_indices = np.flatnonzero(
        np.isin(asym_ids, (1, 2))
        & (
            np.isin(res_ids, np.arange(1, 10))
            | np.isin(res_ids, np.arange(33, 42))
        )
    )
    original_asym_ids = atom_array.asym_id_int.copy()
    original_res_ids = atom_array.res_id.copy()

    features = featurizer.make_msa_features(
        bioassembly_dict=bioassembly,
        selected_indices=selected_indices,
        entity_to_asym_id_int={"1": [1, 2]},
        pair_msa_as_unpaired_only=True,
        shuffle_pair_msa_when_unpaired_only=True,
        virtual_segment_shuffle_seed=11,
        msa_shuffle_variant_seed=variant_seed,
    )

    profile = featurizer.get_last_profile()
    assert profile["virtual_segment_count"] == expected_segment_count
    assert profile["msa_shuffle_group_count"] == expected_group_count
    assert profile["msa_shuffle_requested_variant"] == expected_variant
    assert (
        profile["msa_shuffle_variant_eligible"]
        is expected_eligible
    )
    assert profile["msa_shuffle_variant_applied"] is True
    assert profile["virtual_segment_shuffle_applied"] is (
        expected_variant == MSA_SHUFFLE_VARIANT_SEGMENT
    )
    assert profile["msa_shuffle_whole_physical_chain_applied"] is (
        expected_variant == MSA_SHUFFLE_VARIANT_WHOLE_PHYSICAL_CHAIN
    )
    selected_asym_ids = asym_ids[selected_indices]
    selected_res_ids = res_ids[selected_indices]
    modified_columns = np.flatnonzero(
        (selected_asym_ids == 1) & (selected_res_ids == 2)
    )
    assert modified_columns.size == 3
    for name in ("msa", "has_deletion", "deletion_value"):
        np.testing.assert_array_equal(
            features[name][:, modified_columns],
            np.repeat(
                features[name][:, modified_columns[:1]],
                modified_columns.size,
                axis=1,
            ),
        )
    for name in ("profile", "deletion_mean"):
        np.testing.assert_array_equal(
            features[name][modified_columns],
            np.repeat(
                features[name][modified_columns[:1]],
                modified_columns.size,
                axis=0,
            ),
        )
    np.testing.assert_array_equal(atom_array.asym_id_int, original_asym_ids)
    np.testing.assert_array_equal(atom_array.res_id, original_res_ids)

    first_segment_columns = {}
    second_segment_columns = {}
    for asym_id in (1, 2):
        first_segment_columns[asym_id] = int(
            np.flatnonzero(
                (selected_asym_ids == asym_id) & (selected_res_ids == 1)
            )[0]
        )
        second_segment_columns[asym_id] = int(
            np.flatnonzero(
                (selected_asym_ids == asym_id) & (selected_res_ids == 33)
            )[0]
        )
    for asym_id in (1, 2):
        first_column = first_segment_columns[asym_id]
        second_column = second_segment_columns[asym_id]
        if expected_variant == MSA_SHUFFLE_VARIANT_WHOLE_PHYSICAL_CHAIN:
            np.testing.assert_array_equal(
                features["msa"][:, first_column],
                features["msa"][:, second_column],
            )

    if (
        variant_sampling_enabled
        and expected_variant == MSA_SHUFFLE_VARIANT_SEGMENT
    ):
        # The independent variant gate must not consume or perturb the row
        # permutation stream. Forced segment and legacy static segment are
        # byte-for-byte identical with the same row seed.
        baseline_featurizer = make_featurizer(
            sampling_enabled=False,
            probability=0.5,
        )
        monkeypatch.setattr(
            baseline_featurizer.prot_mgr,
            "fetch_msas",
            fetch_msas,
        )
        baseline = baseline_featurizer.make_msa_features(
            bioassembly_dict=bioassembly,
            selected_indices=selected_indices,
            entity_to_asym_id_int={"1": [1, 2]},
            pair_msa_as_unpaired_only=True,
            shuffle_pair_msa_when_unpaired_only=True,
            virtual_segment_shuffle_seed=11,
        )
        for name in baseline:
            np.testing.assert_array_equal(features[name], baseline[name])

    if expected_variant == MSA_SHUFFLE_VARIANT_WHOLE_PHYSICAL_CHAIN:
        chain_one_indices = selected_indices[
            asym_ids[selected_indices] == 1
        ]
        featurizer.make_msa_features(
            bioassembly_dict=bioassembly,
            selected_indices=chain_one_indices,
            entity_to_asym_id_int={"1": [1, 2]},
            pair_msa_as_unpaired_only=True,
            shuffle_pair_msa_when_unpaired_only=True,
            virtual_segment_shuffle_seed=11,
            msa_shuffle_variant_seed=(1 << 64) - 1,
        )
        crop_profile = featurizer.get_last_profile()
        assert crop_profile["msa_shuffle_crop_protein_chain_count"] == 1
        assert (
            crop_profile["msa_shuffle_requested_variant"]
            == MSA_SHUFFLE_VARIANT_SEGMENT
        )
        assert crop_profile["msa_shuffle_variant_eligible"] is False

        for inactive_kwargs in (
            {
                "query_only": True,
                "pair_msa_as_unpaired_only": False,
                "shuffle_pair_msa_when_unpaired_only": False,
            },
            {
                "query_only": False,
                "pair_msa_as_unpaired_only": False,
                "shuffle_pair_msa_when_unpaired_only": False,
            },
        ):
            featurizer.make_msa_features(
                bioassembly_dict=bioassembly,
                selected_indices=selected_indices,
                entity_to_asym_id_int={"1": [1, 2]},
                **inactive_kwargs,
            )
            inactive_profile = featurizer.get_last_profile()
            assert (
                inactive_profile["msa_shuffle_requested_variant"]
                == MSA_SHUFFLE_VARIANT_NONE
            )
            assert (
                inactive_profile["msa_shuffle_variant_eligible"] is False
            )
            assert (
                inactive_profile["msa_shuffle_variant_applied"] is False
            )


def test_query_only_msa_skips_io_and_virtual_segment_shuffle(monkeypatch):
    sequence = "A" * 42
    atom_array = _FakeAtomArray(
        asym_id_int=[1] * len(sequence),
        chain_id=["A"] * len(sequence),
        res_id=np.arange(1, len(sequence) + 1),
    )
    bioassembly = {
        "pdb_id": "query-only",
        "atom_array": atom_array,
        "token_array": _FakeTokenArray(np.arange(len(sequence))),
        "sequences": {"1": sequence},
        "entity_poly_type": {"1": "polypeptide(L)"},
    }
    featurizer = MSAFeaturizer(
        dataset_name="query-only-test",
        prot_seq_or_filename_to_msadir_jsons=[],
        prot_msadir_raw_paths=[],
        rna_seq_or_filename_to_msadir_jsons=[],
        rna_msadir_raw_paths=[],
        prot_pairing_dbs=[],
        prot_non_pairing_dbs=[],
        prot_indexing_methods=[],
        rna_indexing_methods=[],
        enable_prot_msa=True,
        enable_rna_msa=False,
        max_msa_size=16,
        enable_virtual_segment_msa_shuffle=True,
        enable_multichain_msa_shuffle_variant_sampling=True,
    )

    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("query-only MSA must not fetch homologs")

    monkeypatch.setattr(
        featurizer.prot_mgr,
        "fetch_msas",
        unexpected_fetch,
    )
    selected_indices = np.concatenate((np.arange(9), np.arange(32, 41)))

    features = featurizer.make_msa_features(
        bioassembly_dict=bioassembly,
        selected_indices=selected_indices,
        entity_to_asym_id_int={"1": [1]},
        query_only=True,
        pair_msa_as_unpaired_only=False,
        shuffle_pair_msa_when_unpaired_only=False,
        virtual_segment_shuffle_seed=13,
    )

    profile = featurizer.get_last_profile()
    assert profile["virtual_segment_count"] == 0
    assert profile["virtual_segment_shuffle_applied"] is False
    assert profile["msa_shuffle_requested_variant"] == MSA_SHUFFLE_VARIANT_NONE
    assert profile["msa_shuffle_variant_eligible"] is False
    assert features["msa"].shape == (1, selected_indices.size)
    # Query-only preserves the existing count convention: every MSA count
    # field reports the one retained global query row.
    assert int(features["prot_pair_num_alignments"]) == 1
    assert int(features["prot_unpair_num_alignments"]) == 1


@pytest.mark.parametrize(
    ("pair_as_unpaired", "shuffle"),
    [
        (False, False),
        (False, True),
        (True, False),
    ],
)
def test_virtual_segments_are_inactive_outside_unpaired_shuffle(
    pair_as_unpaired,
    shuffle,
):
    kwargs = {
        "max_msa_size": 32,
        "pair_msa_as_unpaired_only": pair_as_unpaired,
        "shuffle_pair_msa_when_unpaired_only": shuffle,
    }
    baseline = FeatureAssemblyLine(**kwargs).assemble(
        _single_chain_bioassembly(),
        np.arange(18, dtype=np.int32),
    ).to_dict()
    candidate_line = FeatureAssemblyLine(
        **kwargs,
        enable_virtual_segment_msa_shuffle=True,
    )
    candidate = candidate_line.assemble(
        _single_chain_bioassembly(),
        np.arange(18, dtype=np.int32),
        virtual_segment_groups={
            0: (np.arange(9), np.arange(9, 18)),
        },
        virtual_segment_shuffle_seed=2,
    ).to_dict()

    assert candidate_line.last_virtual_segment_shuffle_applied is False
    for name in (
        "msa",
        "deletion_matrix",
        "profile",
        "deletion_mean",
        "prot_pair_num_alignments",
        "prot_unpair_num_alignments",
    ):
        np.testing.assert_array_equal(candidate[name], baseline[name])


def test_one_group_or_one_hit_is_a_noop():
    assembly = FeatureAssemblyLine(
        max_msa_size=32,
        pair_msa_as_unpaired_only=True,
        shuffle_pair_msa_when_unpaired_only=True,
        enable_virtual_segment_msa_shuffle=True,
    )
    one_group = assembly.assemble(
        _single_chain_bioassembly(),
        np.arange(18, dtype=np.int32),
        virtual_segment_groups={0: (np.arange(18),)},
        virtual_segment_shuffle_seed=9,
    ).to_dict()

    assert assembly.last_virtual_segment_count == 1
    assert assembly.last_virtual_segment_shuffle_applied is False
    baseline = FeatureAssemblyLine(
        max_msa_size=32,
        pair_msa_as_unpaired_only=True,
        shuffle_pair_msa_when_unpaired_only=False,
        enable_virtual_segment_msa_shuffle=True,
    ).assemble(
        _single_chain_bioassembly(),
        np.arange(18, dtype=np.int32),
        virtual_segment_groups={0: (np.arange(18),)},
        virtual_segment_shuffle_seed=9,
    ).to_dict()
    for name in (
        "msa",
        "deletion_matrix",
        "profile",
        "deletion_mean",
        "prot_pair_num_alignments",
        "prot_unpair_num_alignments",
    ):
        np.testing.assert_array_equal(one_group[name], baseline[name])

    msa = np.arange(10, dtype=np.int64).reshape(2, 5)
    deletion = msa + 20
    shuffled_msa, shuffled_deletion, attempted = (
        shuffle_msa_feature_rows_by_segment(
            msa=msa,
            deletion_matrix=deletion,
            segment_columns=(np.arange(2), np.arange(3, 5)),
            rng=np.random.default_rng(1),
        )
    )
    assert attempted is False
    np.testing.assert_array_equal(shuffled_msa, msa)
    np.testing.assert_array_equal(shuffled_deletion, deletion)


def test_occurrence_seed_is_stable_and_stream_specific():
    occurrence = SampleOccurrence(
        index=17,
        sampler_seed=42,
        sampler_epoch=3,
        global_position=91,
    )

    first = deterministic_occurrence_seed(
        occurrence,
        stream_name="virtual_segment_msa_shuffle",
    )
    second = deterministic_occurrence_seed(
        occurrence,
        stream_name="virtual_segment_msa_shuffle",
    )
    other = deterministic_occurrence_seed(
        occurrence,
        stream_name="another_stream",
    )
    different_epoch = deterministic_occurrence_seed(
        occurrence._replace(sampler_epoch=4),
        stream_name="virtual_segment_msa_shuffle",
    )
    different_position = deterministic_occurrence_seed(
        occurrence._replace(global_position=92),
        stream_name="virtual_segment_msa_shuffle",
    )

    assert first == second
    assert first != other
    assert len({first, different_epoch, different_position}) == 3
