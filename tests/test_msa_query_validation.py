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

import numpy as np
import pytest
from biotite.structure import AtomArray

from protenix.data.constants import DNA_CHAIN, PROTEIN_CHAIN, RNA_CHAIN
from protenix.data.msa.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.msa.msa_utils import MSACore, RawMsa


def _a3m(*records: tuple[str, str]) -> str:
    return "".join(
        f">{description}\n{sequence}\n"
        for description, sequence in records
    )


def test_exact_query_keeps_protein_hits():
    raw = RawMsa.from_a3m(
        "ACDEF",
        PROTEIN_CHAIN,
        _a3m(("query", "ACDEF"), ("hit", "AAAAA")),
        dedup=False,
    )

    assert raw.seqs == ["ACDEF", "AAAAA"]
    assert raw.descs == ["query", "hit"]


@pytest.mark.parametrize(
    ("query", "wrong_query", "chain_type"),
    [
        ("ACDEF", "AAAAA", PROTEIN_CHAIN),
        ("ACGUU", "AAAAA", RNA_CHAIN),
        ("ACGTT", "AAAAA", DNA_CHAIN),
    ],
)
def test_same_length_wrong_query_falls_back_to_query_only(
    query: str,
    wrong_query: str,
    chain_type: str,
):
    raw = RawMsa.from_a3m(
        query,
        chain_type,
        _a3m(("wrong", wrong_query), ("canonical", query)),
        dedup=False,
    )

    assert raw.seqs == [query]
    assert raw.descs == ["Original query"]
    np.testing.assert_array_equal(
        raw.featurize()["msa"],
        MSACore.sequences_to_array([query], chain_type)[0],
    )


def test_lowercase_query_insertions_are_feature_equivalent():
    raw = RawMsa.from_a3m(
        "ACDEF",
        PROTEIN_CHAIN,
        _a3m(("query", "ACdDEF"), ("hit", "AAAAA")),
        dedup=False,
    )

    assert raw.seqs == ["ACdDEF", "AAAAA"]
    np.testing.assert_array_equal(
        raw.featurize()["deletion_matrix"][0],
        np.asarray([0, 0, 1, 0, 0], dtype=np.int32),
    )


def test_model_token_aliases_are_query_equivalent():
    raw = RawMsa.from_a3m(
        "DCEX",
        PROTEIN_CHAIN,
        _a3m(("query", "BUZO"), ("hit", "AAAA")),
        dedup=False,
    )

    assert raw.seqs == ["BUZO", "AAAA"]


@pytest.mark.parametrize("row_zero", ["AC-EF", "AC!DEF"])
def test_gap_or_invalid_character_is_not_ignored(row_zero: str):
    raw = RawMsa.from_a3m(
        "ACDEF",
        PROTEIN_CHAIN,
        _a3m(("wrong", row_zero), ("canonical", "ACDEF")),
        dedup=False,
    )

    assert raw.seqs == ["ACDEF"]
    assert raw.descs == ["Original query"]


def test_file_a3m_uses_the_same_query_validation(tmp_path):
    path = tmp_path / "wrong_query.a3m"
    path.write_text(
        _a3m(("wrong", "AAAAA"), ("canonical", "ACDEF")),
        encoding="utf-8",
    )

    raw = RawMsa.from_a3m_file(
        "ACDEF",
        PROTEIN_CHAIN,
        path,
        dedup=False,
    )

    assert raw.seqs == ["ACDEF"]
    assert raw.descs == ["Original query"]


def _protein_atom_array(sequence_length: int) -> AtomArray:
    atom_array = AtomArray(sequence_length)
    atom_array.chain_id[:] = "A"
    atom_array.res_id[:] = np.arange(1, sequence_length + 1)
    atom_array.set_annotation(
        "asym_id_int",
        np.zeros(sequence_length, dtype=np.int64),
    )
    atom_array.set_annotation(
        "centre_atom_mask",
        np.ones(sequence_length, dtype=bool),
    )
    return atom_array


@pytest.mark.parametrize("input_field", ["unpairedMsa", "unpairedMsaPath"])
def test_inference_inputs_use_the_same_query_validation(
    input_field: str,
    tmp_path,
):
    query = "ACDEF"
    wrong_a3m = _a3m(("wrong", "AAAAA"), ("canonical", query))
    if input_field == "unpairedMsa":
        input_value = wrong_a3m
    else:
        path = tmp_path / "wrong_query.a3m"
        path.write_text(wrong_a3m, encoding="utf-8")
        input_value = str(path)

    features = InferenceMSAFeaturizer.make_msa_feature(
        [
            {
                "proteinChain": {
                    "sequence": query,
                    "count": 1,
                    input_field: input_value,
                    "pairedMsa": "",
                }
            }
        ],
        _protein_atom_array(len(query)),
    )

    query_ids = MSACore.sequences_to_array([query], PROTEIN_CHAIN)[0][0]
    assert int(features["prot_unpaired_num_alignments"]) == 1
    assert features["msa"].shape == (2, len(query))
    np.testing.assert_array_equal(
        features["msa"],
        np.repeat(query_ids[None, :], 2, axis=0),
    )
