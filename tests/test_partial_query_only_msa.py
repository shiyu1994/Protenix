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

from unittest.mock import Mock

import numpy as np

from protenix.data.constants import (
    PROTEIN_CHAIN,
    RNA_CHAIN,
    STD_RESIDUES_WITH_GAP,
)
from protenix.data.msa.msa_featurizer import (
    MSAFeaturizer,
    apply_partial_query_only_msa_features,
    select_query_only_protein_asym_ids,
)
from protenix.data.msa.msa_utils import (
    MSA_GAP_IDX,
    MSACore,
    RawMsa,
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


def _make_featurizer() -> MSAFeaturizer:
    featurizer = object.__new__(MSAFeaturizer)
    featurizer.prot_mgr = Mock()
    featurizer.rna_mgr = Mock()
    featurizer.prot_p_dbs = ["pairing"]
    featurizer.prot_np_dbs = ["unpaired"]
    featurizer.max_msa_size = 32
    featurizer.pair_msa_as_unpaired_only = False
    featurizer.shuffle_pair_msa_when_unpaired_only = False
    featurizer.enable_crop_aware_partial_query_only = True
    featurizer.partial_query_only_max_chains = 2
    return featurizer


def _raw_msa(
    query: str,
    chain_type: str,
    description: str,
    hit: str,
) -> RawMsa:
    return RawMsa.from_a3m(
        query,
        chain_type,
        f">query\n{query}\n>{description}\n{hit}\n",
        dedup=False,
    )


def _bioassembly(
    protein_sequences: list[str],
    *,
    homomer: bool = False,
    rna_sequence: str | None = None,
):
    sequences = {}
    entity_poly_type = {}
    entity_to_asym = {}
    asym_ids = []
    chain_ids = []
    res_ids = []
    atom_offset = 0
    for index, sequence in enumerate(protein_sequences, start=1):
        entity_id = "1" if homomer else str(index)
        sequences.setdefault(entity_id, sequence)
        entity_poly_type[entity_id] = "polypeptide(L)"
        entity_to_asym.setdefault(entity_id, []).append(index)
        asym_ids.extend([index] * len(sequence))
        chain_ids.extend([chr(ord("A") + index - 1)] * len(sequence))
        res_ids.extend(range(1, len(sequence) + 1))
        atom_offset += len(sequence)
    if rna_sequence is not None:
        entity_id = str(len(sequences) + 1)
        rna_asym_id = len(protein_sequences) + 1
        sequences[entity_id] = rna_sequence
        entity_poly_type[entity_id] = "polyribonucleotide"
        entity_to_asym[entity_id] = [rna_asym_id]
        asym_ids.extend([rna_asym_id] * len(rna_sequence))
        chain_ids.extend(["R"] * len(rna_sequence))
        res_ids.extend(range(1, len(rna_sequence) + 1))
    atom_array = _FakeAtomArray(asym_ids, chain_ids, res_ids)
    return (
        {
            "pdb_id": "partial-query-only",
            "atom_array": atom_array,
            "token_array": _FakeTokenArray(np.arange(len(asym_ids))),
            "sequences": sequences,
            "entity_poly_type": entity_poly_type,
        },
        entity_to_asym,
    )


def test_partial_query_only_selector_limits_and_is_deterministic():
    numpy_state = np.random.get_state()
    for chain_count in (0, 1, 2, 3, 5):
        asym_ids = list(range(1, chain_count + 1))
        selected_counts = set()
        for seed in range(32):
            first = select_query_only_protein_asym_ids(
                protein_asym_ids=asym_ids,
                max_query_only_chains=2,
                rng=np.random.default_rng(seed),
            )
            second = select_query_only_protein_asym_ids(
                protein_asym_ids=list(reversed(asym_ids)),
                max_query_only_chains=2,
                rng=np.random.default_rng(seed),
            )
            assert first == second
            selected_counts.add(len(first))
            if chain_count == 0:
                assert first == ()
            elif chain_count == 1:
                assert first == (1,)
            else:
                assert 1 <= len(first) <= min(2, chain_count - 1)
                assert chain_count - len(first) >= 1
                assert set(first).issubset(asym_ids)
        if chain_count >= 3:
            assert selected_counts == {1, 2}
    restored_state = np.random.get_state()
    assert numpy_state[0] == restored_state[0]
    np.testing.assert_array_equal(numpy_state[1], restored_state[1])
    assert numpy_state[2:] == restored_state[2:]


def test_partial_transform_keeps_query_and_unselected_columns_exact():
    msa = np.arange(32, dtype=np.int16).reshape(4, 8) % 20
    msa[0] = np.arange(8) % 5
    deletion = np.arange(32, dtype=np.int16).reshape(4, 8)
    profile = np.arange(
        8 * len(STD_RESIDUES_WITH_GAP),
        dtype=np.float32,
    ).reshape(8, len(STD_RESIDUES_WITH_GAP))
    features = {
        "msa": msa,
        "deletion_matrix": deletion,
        "has_deletion": np.clip(deletion, 0, 1).astype(np.float32),
        "deletion_value": deletion.astype(np.float32) / 10,
        "profile": profile,
        "deletion_mean": deletion.mean(axis=0).astype(np.float32),
    }
    originals = {name: value.copy() for name, value in features.items()}
    token_asym_ids = np.array([1, 1, 2, 2, 2, 3, 3, 4])

    transformed = apply_partial_query_only_msa_features(
        features,
        token_asym_ids=token_asym_ids,
        query_only_asym_ids=[2],
    )

    selected = token_asym_ids == 2
    retained = ~selected
    np.testing.assert_array_equal(transformed["msa"][0], msa[0])
    np.testing.assert_array_equal(
        transformed["msa"][1:, selected],
        np.full((3, selected.sum()), MSA_GAP_IDX),
    )
    np.testing.assert_array_equal(
        transformed["msa"][:, retained],
        msa[:, retained],
    )
    for name in ("deletion_matrix", "has_deletion", "deletion_value"):
        np.testing.assert_array_equal(
            transformed[name][1:, selected],
            np.zeros((3, selected.sum())),
        )
        np.testing.assert_array_equal(
            transformed[name][:, retained],
            features[name][:, retained],
        )
    expected_profile = (
        msa[0, selected, None]
        == np.arange(len(STD_RESIDUES_WITH_GAP))[None, :]
    ).astype(np.float32)
    np.testing.assert_array_equal(
        transformed["profile"][selected],
        expected_profile,
    )
    np.testing.assert_array_equal(
        transformed["profile"][retained],
        profile[retained],
    )
    np.testing.assert_array_equal(
        transformed["deletion_mean"][selected],
        deletion[0, selected],
    )
    for name, original in originals.items():
        np.testing.assert_array_equal(features[name], original)


def test_two_chain_partial_query_only_keeps_all_retained_msa_sources_and_rna():
    protein_sequences = ["ACDEF", "GHIKL"]
    rna_sequence = "ACGUU"
    bioassembly, entity_to_asym = _bioassembly(
        protein_sequences,
        rna_sequence=rna_sequence,
    )
    featurizer = _make_featurizer()
    protein_sources = {
        protein_sequences[0]: ("CCCCC", "DDDDD"),
        protein_sequences[1]: ("MMMMM", "NNNNN"),
    }
    rna_hit = "UGCAA"

    def fetch_protein(query, *args, **kwargs):
        unpaired_hit, paired_hit = protein_sources[query]
        return (
            [
                _raw_msa(
                    query,
                    PROTEIN_CHAIN,
                    f"{query}_unpaired",
                    unpaired_hit,
                )
            ],
            [
                _raw_msa(
                    query,
                    PROTEIN_CHAIN,
                    f"sp|P12345|{query}_9606",
                    paired_hit,
                )
            ],
        )

    featurizer.prot_mgr.fetch_msas.side_effect = fetch_protein
    featurizer.rna_mgr.fetch_msas.return_value = (
        [_raw_msa(rna_sequence, RNA_CHAIN, "rna_hit", rna_hit)],
        [],
    )

    features = featurizer.make_msa_features(
        bioassembly_dict=bioassembly,
        selected_indices=None,
        entity_to_asym_id_int=entity_to_asym,
        query_only=True,
        pair_msa_as_unpaired_only=False,
        shuffle_pair_msa_when_unpaired_only=False,
        query_only_chain_selection_seed=0,
    )

    profile = featurizer.get_last_profile()
    assert profile["query_only_partial_applied"] is True
    assert profile["query_only_global_applied"] is False
    assert profile["query_only_crop_protein_chain_count"] == 2
    assert profile["query_only_selected_protein_chain_count"] == 1
    assert profile["query_only_retained_protein_chain_count"] == 1
    selected_asym = profile["query_only_selected_protein_asym_ids"][0]
    selected_slice = (
        slice(0, 5) if selected_asym == 1 else slice(5, 10)
    )
    retained_slice = (
        slice(5, 10) if selected_asym == 1 else slice(0, 5)
    )
    retained_query = protein_sequences[1 if selected_asym == 1 else 0]
    retained_hits = protein_sources[retained_query]

    np.testing.assert_array_equal(
        features["msa"][1:, selected_slice],
        np.full(
            (features["msa"].shape[0] - 1, 5),
            MSA_GAP_IDX,
        ),
    )
    np.testing.assert_array_equal(
        features["has_deletion"][1:, selected_slice],
        np.zeros((features["msa"].shape[0] - 1, 5)),
    )
    for hit in retained_hits:
        encoded_hit, _ = MSACore.sequences_to_array(
            [hit],
            PROTEIN_CHAIN,
        )
        assert any(
            np.array_equal(row, encoded_hit[0])
            for row in features["msa"][:, retained_slice]
        )
    encoded_rna_hit, _ = MSACore.sequences_to_array(
        [rna_hit],
        RNA_CHAIN,
    )
    assert any(
        np.array_equal(row, encoded_rna_hit[0])
        for row in features["msa"][:, 10:15]
    )
    assert featurizer.prot_mgr.fetch_msas.call_count == 1
    assert featurizer.rna_mgr.fetch_msas.call_count == 1
    # The paired query placeholder is tracked separately; the unpaired count
    # here is the two retained homolog rows.
    assert int(features["prot_unpair_num_alignments"]) == 2


def test_direct_query_only_call_forces_normal_pairing_on_retained_chains():
    protein_sequences = ["ACDEF", "GHIKL", "MNPQR"]
    bioassembly, entity_to_asym = _bioassembly(protein_sequences)
    featurizer = _make_featurizer()
    paired_hits = {
        protein_sequences[0]: "CCCCC",
        protein_sequences[1]: "DDDDD",
        protein_sequences[2]: "EEEEE",
    }

    def fetch_protein(query, *args, **kwargs):
        return (
            [],
            [
                _raw_msa(
                    query,
                    PROTEIN_CHAIN,
                    f"UniRef100_{query}_9606",
                    paired_hits[query],
                )
            ],
        )

    featurizer.prot_mgr.fetch_msas.side_effect = fetch_protein
    features = featurizer.make_msa_features(
        bioassembly_dict=bioassembly,
        selected_indices=None,
        entity_to_asym_id_int=entity_to_asym,
        query_only=True,
        # A direct caller cannot change the retained-chain policy away from
        # the contract's normal species-paired flow.
        pair_msa_as_unpaired_only=True,
        shuffle_pair_msa_when_unpaired_only=True,
        query_only_chain_selection_seed=1,
    )

    profile = featurizer.get_last_profile()
    assert profile["query_only_selected_protein_chain_count"] == 1
    assert profile["query_only_retained_protein_chain_count"] == 2
    assert int(features["prot_pair_num_alignments"]) == 2
    assert int(features["prot_unpair_num_alignments"]) == 1


def test_homomer_selection_is_per_physical_asym_and_gaps_extra_query_rows():
    sequence = "ACDEF"
    bioassembly, entity_to_asym = _bioassembly(
        [sequence, sequence],
        homomer=True,
    )
    featurizer = _make_featurizer()
    featurizer.prot_mgr.fetch_msas.return_value = (
        [_raw_msa(sequence, PROTEIN_CHAIN, "unpaired", "CCCCC")],
        [_raw_msa(sequence, PROTEIN_CHAIN, "paired", "DDDDD")],
    )

    features = featurizer.make_msa_features(
        bioassembly_dict=bioassembly,
        selected_indices=None,
        entity_to_asym_id_int=entity_to_asym,
        query_only=True,
        pair_msa_as_unpaired_only=False,
        shuffle_pair_msa_when_unpaired_only=False,
        query_only_chain_selection_seed=9,
    )

    profile = featurizer.get_last_profile()
    assert profile["query_only_crop_protein_chain_count"] == 2
    assert profile["query_only_selected_protein_chain_count"] == 1
    selected_asym = profile["query_only_selected_protein_asym_ids"][0]
    selected_slice = (
        slice(0, 5) if selected_asym == 1 else slice(5, 10)
    )
    retained_slice = (
        slice(5, 10) if selected_asym == 1 else slice(0, 5)
    )
    np.testing.assert_array_equal(
        features["msa"][1:, selected_slice],
        np.full(
            (features["msa"].shape[0] - 1, 5),
            MSA_GAP_IDX,
        ),
    )
    assert np.any(features["msa"][1:, retained_slice] != MSA_GAP_IDX)
    assert featurizer.prot_mgr.fetch_msas.call_count == 1


def test_modified_residue_multiple_tokens_share_query_only_msa_column():
    """A residue tokenized by atom must inherit one canonical MSA column."""
    protein_sequences = ["ACDEF", "GHIKL"]
    # Residue 2 represents a non-standard amino acid expanded to three
    # atom-level tokens.  map_to_standard() must broadcast canonical MSA
    # column 2 to every one of those tokens before partial query-only masking.
    expanded_res_ids = [1, 2, 2, 2, 3, 4, 5]
    token_asym_ids = np.asarray([1] * 7 + [2] * 7, dtype=np.int64)
    bioassembly = {
        "pdb_id": "partial-query-only-modified-residue",
        "atom_array": _FakeAtomArray(
            asym_id_int=token_asym_ids,
            chain_id=["A"] * 7 + ["B"] * 7,
            res_id=expanded_res_ids + expanded_res_ids,
        ),
        "token_array": _FakeTokenArray(np.arange(14)),
        "sequences": {
            "1": protein_sequences[0],
            "2": protein_sequences[1],
        },
        "entity_poly_type": {
            "1": "polypeptide(L)",
            "2": "polypeptide(L)",
        },
    }
    entity_to_asym = {"1": [1], "2": [2]}
    retained_hits = {
        protein_sequences[0]: "CCCCC",
        protein_sequences[1]: "MNPQR",
    }
    featurizer = _make_featurizer()

    def fetch_protein(query, *args, **kwargs):
        return (
            [
                _raw_msa(
                    query,
                    PROTEIN_CHAIN,
                    f"{query}_hit",
                    retained_hits[query],
                )
            ],
            [],
        )

    featurizer.prot_mgr.fetch_msas.side_effect = fetch_protein
    features = featurizer.make_msa_features(
        bioassembly_dict=bioassembly,
        selected_indices=None,
        entity_to_asym_id_int=entity_to_asym,
        query_only=True,
        pair_msa_as_unpaired_only=False,
        shuffle_pair_msa_when_unpaired_only=False,
        query_only_chain_selection_seed=5,
    )

    profile = featurizer.get_last_profile()
    selected_asym = profile["query_only_selected_protein_asym_ids"][0]
    selected_columns = token_asym_ids == selected_asym
    retained_asym = 2 if selected_asym == 1 else 1
    retained_columns = token_asym_ids == retained_asym

    canonical_queries = [
        MSACore.sequences_to_array([sequence], PROTEIN_CHAIN)[0][0]
        for sequence in protein_sequences
    ]
    broadcast_indices = np.asarray([0, 1, 1, 1, 2, 3, 4])
    expected_query = np.concatenate(
        [query[broadcast_indices] for query in canonical_queries]
    )
    np.testing.assert_array_equal(features["msa"][0], expected_query)

    selected_start = 0 if selected_asym == 1 else 7
    modified_token_columns = selected_start + np.asarray([1, 2, 3])
    np.testing.assert_array_equal(
        features["msa"][0, modified_token_columns],
        np.full(3, expected_query[modified_token_columns[0]]),
    )
    np.testing.assert_array_equal(
        features["msa"][1:, selected_columns],
        np.full(
            (features["msa"].shape[0] - 1, int(selected_columns.sum())),
            MSA_GAP_IDX,
        ),
    )
    np.testing.assert_array_equal(
        features["has_deletion"][1:, selected_columns],
        np.zeros(
            (features["msa"].shape[0] - 1, int(selected_columns.sum())),
        ),
    )

    expected_selected_profile = (
        expected_query[selected_columns, None]
        == np.arange(len(STD_RESIDUES_WITH_GAP))[None, :]
    ).astype(features["profile"].dtype)
    np.testing.assert_array_equal(
        features["profile"][selected_columns],
        expected_selected_profile,
    )
    np.testing.assert_array_equal(
        features["profile"][modified_token_columns],
        np.repeat(
            expected_selected_profile[1:2],
            repeats=3,
            axis=0,
        ),
    )
    np.testing.assert_array_equal(
        features["deletion_mean"][selected_columns],
        np.zeros(
            int(selected_columns.sum()),
            dtype=features["deletion_mean"].dtype,
        ),
    )

    retained_index = retained_asym - 1
    retained_hit, _ = MSACore.sequences_to_array(
        [retained_hits[protein_sequences[retained_index]]],
        PROTEIN_CHAIN,
    )
    expected_retained_hit = retained_hit[0, broadcast_indices]
    assert any(
        np.array_equal(row, expected_retained_hit)
        for row in features["msa"][:, retained_columns]
    )
    assert featurizer.prot_mgr.fetch_msas.call_count == 1


def test_chain_selection_uses_only_crop_visible_protein_asym_ids():
    sequences = ["ACDEF", "GHIKL", "MNPQR"]
    bioassembly, entity_to_asym = _bioassembly(sequences)
    featurizer = _make_featurizer()

    def fetch_protein(query, *args, **kwargs):
        return (
            [_raw_msa(query, PROTEIN_CHAIN, "hit", "CCCCC")],
            [],
        )

    featurizer.prot_mgr.fetch_msas.side_effect = fetch_protein
    selected_indices = np.arange(5, 15)

    features = featurizer.make_msa_features(
        bioassembly_dict=bioassembly,
        selected_indices=selected_indices,
        entity_to_asym_id_int=entity_to_asym,
        query_only=True,
        pair_msa_as_unpaired_only=False,
        shuffle_pair_msa_when_unpaired_only=False,
        query_only_chain_selection_seed=4,
    )

    profile = featurizer.get_last_profile()
    assert profile["query_only_crop_protein_chain_count"] == 2
    assert set(profile["query_only_selected_protein_asym_ids"]).issubset(
        {2, 3}
    )
    assert features["msa"].shape[1] == 10


def test_enabled_single_protein_keeps_historical_global_query_only():
    sequence = "ACDEF"
    bioassembly, entity_to_asym = _bioassembly([sequence])
    featurizer = _make_featurizer()

    features = featurizer.make_msa_features(
        bioassembly_dict=bioassembly,
        selected_indices=None,
        entity_to_asym_id_int=entity_to_asym,
        query_only=True,
        pair_msa_as_unpaired_only=False,
        shuffle_pair_msa_when_unpaired_only=False,
        query_only_chain_selection_seed=3,
    )

    profile = featurizer.get_last_profile()
    assert profile["query_only_global_applied"] is True
    assert profile["query_only_partial_applied"] is False
    assert features["msa"].shape == (1, len(sequence))
    featurizer.prot_mgr.fetch_msas.assert_not_called()
