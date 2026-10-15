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

"""Focused runtime recovery tests for disordered OpenFold3 representatives."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from protenix.data.constants import PROTEIN_CHAIN
from protenix.data.msa.msa_featurizer import MSAFeaturizer
from protenix.data.msa.msa_utils import MSA_GAP_IDX, MSACore


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
    def __init__(
        self,
        *,
        asym_id_int,
        chain_id,
        res_id,
        label_asym_id,
    ):
        self.asym_id_int = np.asarray(asym_id_int, dtype=np.int64)
        self.chain_id = np.asarray(chain_id)
        self.res_id = np.asarray(res_id, dtype=np.int64)
        self.label_asym_id = np.asarray(label_asym_id)

    def __getitem__(self, indices):
        return _FakeAtomArray(
            asym_id_int=self.asym_id_int[indices],
            chain_id=self.chain_id[indices],
            res_id=self.res_id[indices],
            label_asym_id=self.label_asym_id[indices],
        )


def _alignment_payload(query: str, hits: list[str]) -> np.ndarray:
    payload = np.empty((), dtype=object)
    rows = [query, *hits]
    payload[()] = {
        "msa": np.asarray([list(row) for row in rows], dtype="<U1"),
        "deletion_matrix": np.zeros(
            (len(rows), len(query)),
            dtype=np.int32,
        ),
        "metadata": np.asarray(
            ["query", *(f"hit_{index}" for index in range(len(hits)))],
            dtype=object,
        ),
    }
    return payload


def _write_alignment(
    directory: Path,
    representative_id: str,
    *,
    query: str,
    hits: list[str],
) -> None:
    np.savez(
        directory / f"{representative_id}.npz",
        uniref90_hits=_alignment_payload(query, hits),
    )


def _featurizer(
    alignment_dir: Path,
    *,
    dataset_name: str = "openfold3_disordered_v20260725",
    max_input_msa_depth: int | None = None,
) -> MSAFeaturizer:
    return MSAFeaturizer(
        dataset_name=dataset_name,
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
        max_msa_size=32,
        max_input_msa_depth=max_input_msa_depth,
        openfold3_alignment_npz_dirs=[str(alignment_dir)],
    )


def _bioassembly(
    chains: list[tuple[str, str, str]],
) -> tuple[dict, dict[str, list[int]], list[slice]]:
    """Build chains as ``(chain_id, sequence, representative_id)``."""
    asym_id_int: list[int] = []
    chain_ids: list[str] = []
    res_ids: list[int] = []
    label_asym_ids: list[str] = []
    sequences: dict[str, str] = {}
    sequences_by_chain_id: dict[str, str] = {}
    entity_poly_type: dict[str, str] = {}
    representatives: dict[str, str] = {}
    entity_to_asym: dict[str, list[int]] = {}
    chain_slices: list[slice] = []
    offset = 0

    for asym_id, (chain_id, sequence, representative_id) in enumerate(
        chains,
        start=1,
    ):
        entity_id = str(asym_id)
        length = len(sequence)
        asym_id_int.extend([asym_id] * length)
        chain_ids.extend([chain_id] * length)
        res_ids.extend(range(1, length + 1))
        label_asym_ids.extend([chain_id] * length)
        sequences[entity_id] = sequence
        sequences_by_chain_id[chain_id] = sequence
        entity_poly_type[entity_id] = "polypeptide(L)"
        representatives[chain_id] = representative_id
        entity_to_asym[entity_id] = [asym_id]
        chain_slices.append(slice(offset, offset + length))
        offset += length

    atom_array = _FakeAtomArray(
        asym_id_int=asym_id_int,
        chain_id=chain_ids,
        res_id=res_ids,
        label_asym_id=label_asym_ids,
    )
    return (
        {
            "pdb_id": "entry",
            "atom_array": atom_array,
            "token_array": _FakeTokenArray(np.arange(offset)),
            "sequences": sequences,
            "sequences_by_chain_id": sequences_by_chain_id,
            "entity_poly_type": entity_poly_type,
            "alignment_representative_id_by_chain_id": representatives,
        },
        entity_to_asym,
        chain_slices,
    )


def _encoded(sequence: str) -> np.ndarray:
    encoded, _ = MSACore.sequences_to_array(
        [sequence],
        PROTEIN_CHAIN,
    )
    return encoded[0]


def _make_features(
    featurizer: MSAFeaturizer,
    bioassembly: dict,
    entity_to_asym: dict[str, list[int]],
) -> dict:
    return featurizer.make_msa_features(
        bioassembly_dict=bioassembly,
        selected_indices=None,
        entity_to_asym_id_int=entity_to_asym,
    )


def _assert_chain_is_query_only(
    features: dict,
    chain_slice: slice,
    query: str,
) -> None:
    encoded_query = _encoded(query)
    for row in features["msa"][:, chain_slice]:
        assert np.array_equal(row, encoded_query) or np.all(
            row == MSA_GAP_IDX
        )


def test_normal_representative_keeps_original_path(tmp_path):
    _write_alignment(
        tmp_path,
        "entry_A",
        query="AAAAA",
        hits=["AAAAC"],
    )
    featurizer = _featurizer(tmp_path)
    bioassembly, entity_to_asym, chain_slices = _bioassembly(
        [("A", "AAAAA", "entry_A")]
    )

    with patch.object(
        featurizer,
        "_load_openfold3_recovery_candidate",
        wraps=featurizer._load_openfold3_recovery_candidate,
    ) as recovery_loader:
        features = _make_features(
            featurizer,
            bioassembly,
            entity_to_asym,
        )

    recovery_loader.assert_not_called()
    profile = featurizer.get_last_profile()
    assert profile["openfold3_representative_query_mismatch_count"] == 0
    assert profile["openfold3_representative_remap_count"] == 0
    assert profile["openfold3_representative_query_only_fallback_count"] == 0
    assert any(
        np.array_equal(row[chain_slices[0]], _encoded("AAAAC"))
        for row in features["msa"]
    )


def test_2uy1_style_ab_swap_remaps_unique_exact_queries_and_caches_candidates(
    tmp_path,
):
    _write_alignment(
        tmp_path,
        "entry_A",
        query="AAAAA",
        hits=["AAAAC"],
    )
    _write_alignment(
        tmp_path,
        "entry_B",
        query="CCCCC",
        hits=["CCCCA"],
    )
    featurizer = _featurizer(tmp_path)
    bioassembly, entity_to_asym, chain_slices = _bioassembly(
        [
            ("A", "CCCCC", "entry_A"),
            ("B", "AAAAA", "entry_B"),
        ]
    )

    with patch.object(
        featurizer,
        "_load_openfold3_recovery_candidate",
        wraps=featurizer._load_openfold3_recovery_candidate,
    ) as recovery_loader:
        features = _make_features(
            featurizer,
            bioassembly,
            entity_to_asym,
        )

    assert recovery_loader.call_count == 2
    profile = featurizer.get_last_profile()
    assert profile["openfold3_representative_query_mismatch_count"] == 2
    assert profile["openfold3_representative_remap_count"] == 2
    assert profile["openfold3_representative_query_only_fallback_count"] == 0
    assert profile["openfold3_recovery_candidate_load_count"] == 2
    assert profile["openfold3_recovery_candidate_cache_hit_count"] == 2
    assert profile["openfold3_remapped_chain_ids"] == ["A", "B"]
    assert profile["openfold3_representative_remaps"] == [
        {"chain_id": "A", "from": "entry_A", "to": "entry_B"},
        {"chain_id": "B", "from": "entry_B", "to": "entry_A"},
    ]
    assert any(
        np.array_equal(row[chain_slices[0]], _encoded("CCCCA"))
        for row in features["msa"]
    )
    assert any(
        np.array_equal(row[chain_slices[1]], _encoded("AAAAC"))
        for row in features["msa"]
    )


def test_remap_recovery_candidate_obeys_input_depth_cap(tmp_path):
    _write_alignment(
        tmp_path,
        "entry_A",
        query="AAAAA",
        hits=["AAAAC", "AAAAD", "AAAAE", "AAAAG"],
    )
    _write_alignment(
        tmp_path,
        "entry_B",
        query="CCCCC",
        hits=["CCCCA", "CCCCD", "CCCCE", "CCCCG"],
    )
    featurizer = _featurizer(
        tmp_path,
        max_input_msa_depth=3,
    )
    bioassembly, entity_to_asym, _ = _bioassembly(
        [
            ("A", "CCCCC", "entry_A"),
            ("B", "AAAAA", "entry_B"),
        ]
    )
    recovered_candidates = []
    original_loader = featurizer._load_openfold3_recovery_candidate

    def record_candidate(**kwargs):
        candidate = original_loader(**kwargs)
        recovered_candidates.append(candidate)
        return candidate

    with patch.object(
        featurizer,
        "_load_openfold3_recovery_candidate",
        side_effect=record_candidate,
    ) as recovery_loader:
        _make_features(
            featurizer,
            bioassembly,
            entity_to_asym,
        )

    assert recovery_loader.call_count == 2
    assert all(
        call.kwargs["paired_depth_limit"] == 3
        and call.kwargs["unpaired_depth_limit"] == 3
        for call in recovery_loader.call_args_list
    )
    assert recovered_candidates
    assert all(
        [raw_msa.depth for raw_msa in candidate.unpaired_msas] == [3]
        and candidate.paired_msas == ()
        for candidate in recovered_candidates
    )
    profile = featurizer.get_last_profile()
    assert profile["openfold3_representative_remap_count"] == 2


def test_no_exact_match_uses_per_chain_query_only_and_warns(
    tmp_path,
    caplog,
):
    _write_alignment(
        tmp_path,
        "entry_A",
        query="AAAAA",
        hits=["AAAAC"],
    )
    _write_alignment(
        tmp_path,
        "entry_B",
        query="CCCCC",
        hits=["CCCCA"],
    )
    featurizer = _featurizer(tmp_path)
    bioassembly, entity_to_asym, chain_slices = _bioassembly(
        [
            ("A", "GGGGG", "entry_A"),
            ("B", "CCCCC", "entry_B"),
        ]
    )
    bioassembly["alignment_representative_id_by_chain_id"][
        "missing"
    ] = "entry_MISSING"

    with caplog.at_level(logging.WARNING):
        features = _make_features(
            featurizer,
            bioassembly,
            entity_to_asym,
        )

    profile = featurizer.get_last_profile()
    assert profile["openfold3_representative_query_mismatch_count"] == 1
    assert profile["openfold3_representative_remap_count"] == 0
    assert profile["openfold3_representative_query_only_fallback_count"] == 1
    assert profile["openfold3_query_only_fallback_chain_ids"] == ["A"]
    assert any(
        key.startswith("entry_MISSING:")
        for key in profile["openfold3_recovery_candidate_errors"]
    )
    np.testing.assert_array_equal(
        features["msa"][0, chain_slices[0]],
        _encoded("GGGGG"),
    )
    _assert_chain_is_query_only(
        features,
        chain_slices[0],
        "GGGGG",
    )
    assert "no unique exact declared canonical-query match" in caplog.text
    assert "using canonical query-only MSA for this chain" in caplog.text
    assert "entry_MISSING" in caplog.text


def test_explicit_query_only_marker_skips_all_alignment_io(tmp_path):
    _write_alignment(
        tmp_path,
        "entry_A",
        query="AAAAA",
        hits=["AAAAC"],
    )
    featurizer = _featurizer(tmp_path)
    bioassembly, entity_to_asym, chain_slices = _bioassembly(
        [("A", "GGGGG", "entry_A")]
    )
    bioassembly["openfold3_query_only_chain_ids"] = ["A"]

    with (
        patch.object(
            featurizer,
            "_load_openfold3_chain_msas",
            wraps=featurizer._load_openfold3_chain_msas,
        ) as alignment_loader,
        patch.object(
            featurizer,
            "_load_openfold3_recovery_candidate",
            wraps=featurizer._load_openfold3_recovery_candidate,
        ) as recovery_loader,
    ):
        features = _make_features(
            featurizer,
            bioassembly,
            entity_to_asym,
        )

    alignment_loader.assert_not_called()
    recovery_loader.assert_not_called()
    profile = featurizer.get_last_profile()
    assert profile["openfold3_explicit_query_only_chain_count"] == 1
    assert profile["openfold3_explicit_query_only_chain_ids"] == ["A"]
    assert profile["openfold3_representative_query_mismatch_count"] == 0
    assert profile["openfold3_representative_remap_count"] == 0
    assert (
        profile["openfold3_representative_query_only_fallback_count"]
        == 0
    )
    assert profile["openfold3_recovery_candidate_load_count"] == 0
    np.testing.assert_array_equal(
        features["msa"][0, chain_slices[0]],
        _encoded("GGGGG"),
    )
    _assert_chain_is_query_only(
        features,
        chain_slices[0],
        "GGGGG",
    )


def test_ambiguous_exact_matches_use_per_chain_query_only_and_warn(
    tmp_path,
    caplog,
):
    _write_alignment(
        tmp_path,
        "entry_A",
        query="AAAAA",
        hits=["AAAAC"],
    )
    _write_alignment(
        tmp_path,
        "entry_B",
        query="AAAAA",
        hits=["AAAAT"],
    )
    _write_alignment(
        tmp_path,
        "entry_C",
        query="CCCCC",
        hits=["CCCCA"],
    )
    featurizer = _featurizer(tmp_path)
    bioassembly, entity_to_asym, chain_slices = _bioassembly(
        [
            ("C", "AAAAA", "entry_C"),
            ("A", "AAAAA", "entry_A"),
            ("B", "AAAAA", "entry_B"),
        ]
    )

    with caplog.at_level(logging.WARNING):
        features = _make_features(
            featurizer,
            bioassembly,
            entity_to_asym,
        )

    profile = featurizer.get_last_profile()
    assert profile["openfold3_representative_query_mismatch_count"] == 1
    assert profile["openfold3_representative_remap_count"] == 0
    assert profile["openfold3_representative_query_only_fallback_count"] == 1
    assert profile["openfold3_query_only_fallback_chain_ids"] == ["C"]
    np.testing.assert_array_equal(
        features["msa"][0, chain_slices[0]],
        _encoded("AAAAA"),
    )
    _assert_chain_is_query_only(
        features,
        chain_slices[0],
        "AAAAA",
    )
    assert "exact_matches=['entry_A', 'entry_B']" in caplog.text


def test_disordered_cache_does_not_reuse_representative_across_queries(
    tmp_path,
):
    _write_alignment(
        tmp_path,
        "entry_A",
        query="AAAAA",
        hits=["AAAAC"],
    )
    featurizer = _featurizer(tmp_path)
    bioassembly, entity_to_asym, chain_slices = _bioassembly(
        [
            ("A", "AAAAA", "entry_A"),
            ("B", "CCCCC", "entry_A"),
        ]
    )

    features = _make_features(
        featurizer,
        bioassembly,
        entity_to_asym,
    )

    profile = featurizer.get_last_profile()
    assert profile["openfold3_representative_query_mismatch_count"] == 1
    assert profile["openfold3_representative_remap_count"] == 0
    assert profile["openfold3_representative_query_only_fallback_count"] == 1
    assert profile["openfold3_query_only_fallback_chain_ids"] == ["B"]
    _assert_chain_is_query_only(
        features,
        chain_slices[1],
        "CCCCC",
    )


def test_mismatch_recovery_is_gated_to_disordered_dataset(tmp_path):
    _write_alignment(
        tmp_path,
        "entry_A",
        query="AAAAA",
        hits=["AAAAC"],
    )
    featurizer = _featurizer(
        tmp_path,
        dataset_name="another_openfold3_npz_consumer",
    )
    bioassembly, entity_to_asym, _ = _bioassembly(
        [("A", "CCCCC", "entry_A")]
    )

    with patch.object(
        featurizer,
        "_load_openfold3_recovery_candidate",
        wraps=featurizer._load_openfold3_recovery_candidate,
    ) as recovery_loader:
        with pytest.raises(
            ValueError,
            match="alignment query row does not match the canonical sequence",
        ):
            _make_features(
                featurizer,
                bioassembly,
                entity_to_asym,
            )

    recovery_loader.assert_not_called()
