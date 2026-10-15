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
import unittest
from unittest.mock import Mock, call, patch

import numpy as np

from protenix.data.constants import (
    PROTEIN_CHAIN,
    RNA_CHAIN,
    STD_RESIDUES_WITH_GAP,
)
from protenix.data.msa.msa_featurizer import (
    FeatureAssemblyLine,
    MSAFeaturizer,
    make_query_only_msa_features,
)
from protenix.data.msa.msa_utils import RawMsa


class _FakeTokenArray:
    def __init__(self, centre_atom_indices):
        self.centre_atom_indices = np.asarray(centre_atom_indices, dtype=np.int64)

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


class TestQueryOnlyMSAFeatures(unittest.TestCase):
    @staticmethod
    def _a3m_with_insertion_homolog(sequence: str, fill: str) -> str:
        homolog = sequence[0] + "a" + fill * (len(sequence) - 1)
        return f">query\n{sequence}\n>homolog\n{homolog}\n"

    @classmethod
    def _assemble_two_protein_chains(
        cls,
        sequences: tuple[str, str],
        *,
        homomer: bool,
    ) -> dict[str, np.ndarray]:
        meta = {}
        for index, sequence in enumerate(sequences, start=1):
            meta[index] = {
                "entity_id": "1" if homomer else str(index),
                "chain_id": chr(ord("A") + index - 1),
                "sequence": sequence,
                "chain_entity_type": PROTEIN_CHAIN,
                "paired_msa": "",
                "unpaired_msa": cls._a3m_with_insertion_homolog(
                    sequence,
                    "A" if index == 1 else "C",
                ),
            }
        return FeatureAssemblyLine(max_msa_size=16).assemble(
            meta,
            std_idxs=np.arange(sum(map(len, sequences)), dtype=np.int32),
        ).to_dict()

    @staticmethod
    def _query_only_featurizer() -> MSAFeaturizer:
        featurizer = object.__new__(MSAFeaturizer)
        featurizer.prot_mgr = Mock()
        featurizer.rna_mgr = Mock()
        featurizer.prot_p_dbs = ["pairing"]
        featurizer.prot_np_dbs = ["non_pairing"]
        featurizer.max_msa_size = 16
        featurizer.pair_msa_as_unpaired_only = False
        featurizer.shuffle_pair_msa_when_unpaired_only = False
        return featurizer

    @staticmethod
    def _protein_rna_bioassembly():
        protein = "ACDEF"
        rna = "ACGUU"
        asym_id_int = [1] * len(protein) + [2] * len(rna)
        chain_id = ["A"] * len(protein) + ["R"] * len(rna)
        res_id = list(range(1, len(protein) + 1)) + list(
            range(1, len(rna) + 1)
        )
        atom_array = _FakeAtomArray(asym_id_int, chain_id, res_id)
        token_array = _FakeTokenArray(np.arange(len(asym_id_int)))
        return {
            "pdb_id": "protein-rna-target",
            "atom_array": atom_array,
            "token_array": token_array,
            "sequences": {"1": protein, "2": rna},
            "entity_poly_type": {
                "1": "polypeptide(L)",
                "2": "polyribonucleotide",
            },
        }

    def _assert_query_only_features(
        self,
        full_features: dict[str, np.ndarray],
        expected_query: np.ndarray,
    ) -> None:
        original = {
            name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in full_features.items()
        }
        self.assertGreater(full_features["msa"].shape[0], 1)
        self.assertTrue(
            np.any(
                full_features["deletion_mean"]
                != full_features["deletion_matrix"][0]
            )
        )

        query_only = make_query_only_msa_features(full_features)

        self.assertEqual(query_only["msa"].shape, (1, expected_query.size))
        self.assertEqual(
            query_only["msa_mask"].shape,
            (1, expected_query.size),
        )
        self.assertEqual(
            query_only["deletion_matrix"].shape,
            (1, expected_query.size),
        )
        np.testing.assert_array_equal(query_only["msa"][0], expected_query)

        expected_profile = (
            expected_query[:, None]
            == np.arange(len(STD_RESIDUES_WITH_GAP))[None, :]
        ).astype(query_only["profile"].dtype)
        np.testing.assert_array_equal(query_only["profile"], expected_profile)
        np.testing.assert_array_equal(
            query_only["deletion_mean"],
            full_features["deletion_matrix"][0].astype(
                query_only["deletion_mean"].dtype
            ),
        )

        for name in (
            "prot_unpaired_num_alignments",
            "prot_paired_num_alignments",
            "rna_unpaired_num_alignments",
            "prot_pair_num_alignments",
            "prot_unpair_num_alignments",
            "rna_pair_num_alignments",
            "rna_unpair_num_alignments",
        ):
            self.assertEqual(int(query_only[name]), 1)

        for name, value in original.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(full_features[name], value)
            else:
                self.assertEqual(full_features[name], value)

    def test_heteromer_is_one_global_query_row_without_summary_leakage(self):
        full = self._assemble_two_protein_chains(
            ("ACDEF", "GHIKL"),
            homomer=False,
        )
        self._assert_query_only_features(full, full["msa"][0].copy())

    def test_homomer_keeps_both_chain_copies_in_one_global_row(self):
        full = self._assemble_two_protein_chains(
            ("ACDEF", "ACDEF"),
            homomer=True,
        )
        expected = full["msa"][0].copy()
        np.testing.assert_array_equal(expected[:5], expected[5:])
        self._assert_query_only_features(full, expected)

    def test_explicit_query_only_skips_protein_and_rna_homolog_io(self):
        featurizer = self._query_only_featurizer()
        bioassembly = self._protein_rna_bioassembly()

        features = featurizer.make_msa_features(
            bioassembly_dict=bioassembly,
            selected_indices=None,
            entity_to_asym_id_int={"1": [1], "2": [2]},
            query_only=True,
        )

        featurizer.prot_mgr.fetch_msas.assert_not_called()
        featurizer.rna_mgr.fetch_msas.assert_not_called()
        self.assertEqual(features["msa"].shape, (1, 10))

        expected_protein = RawMsa.from_a3m(
            "ACDEF", PROTEIN_CHAIN, "", dedup=True
        ).featurize()["msa"][0]
        expected_rna = RawMsa.from_a3m(
            "ACGUU", RNA_CHAIN, "", dedup=True
        ).featurize()["msa"][0]
        expected_query = np.concatenate([expected_protein, expected_rna])
        np.testing.assert_array_equal(features["msa"][0], expected_query)

        expected_profile = (
            expected_query[:, None]
            == np.arange(len(STD_RESIDUES_WITH_GAP))[None, :]
        ).astype(features["profile"].dtype)
        np.testing.assert_array_equal(features["profile"], expected_profile)
        np.testing.assert_array_equal(
            features["deletion_mean"],
            np.zeros(10, dtype=features["deletion_mean"].dtype),
        )
        for name in (
            "prot_pair_num_alignments",
            "prot_unpair_num_alignments",
            "rna_pair_num_alignments",
            "rna_unpair_num_alignments",
        ):
            self.assertEqual(int(features[name]), 1)

    def test_default_full_mode_still_reads_protein_and_rna_sources(self):
        featurizer = self._query_only_featurizer()
        featurizer.prot_mgr.fetch_msas.return_value = ([], [])
        featurizer.rna_mgr.fetch_msas.return_value = ([], [])

        featurizer.make_msa_features(
            bioassembly_dict=self._protein_rna_bioassembly(),
            selected_indices=None,
            entity_to_asym_id_int={"1": [1], "2": [2]},
        )

        featurizer.prot_mgr.fetch_msas.assert_called_once()
        featurizer.rna_mgr.fetch_msas.assert_called_once()

    def test_per_occurrence_pairing_overrides_do_not_mutate_featurizer(self):
        featurizer = self._query_only_featurizer()
        featurizer.prot_mgr.fetch_msas.return_value = ([], [])
        featurizer.rna_mgr.fetch_msas.return_value = ([], [])
        assembled = Mock()
        assembled.to_dict.return_value = {
            "msa": np.zeros((1, 10), dtype=np.int8)
        }

        with patch(
            "protenix.data.msa.msa_featurizer.FeatureAssemblyLine"
        ) as assembly_cls:
            assembly_cls.return_value.assemble.return_value = assembled
            kwargs = {
                "bioassembly_dict": self._protein_rna_bioassembly(),
                "selected_indices": None,
                "entity_to_asym_id_int": {"1": [1], "2": [2]},
            }
            featurizer.make_msa_features(
                **kwargs,
                pair_msa_as_unpaired_only=True,
                shuffle_pair_msa_when_unpaired_only=True,
            )
            featurizer.make_msa_features(
                **kwargs,
                pair_msa_as_unpaired_only=False,
                shuffle_pair_msa_when_unpaired_only=False,
            )

        self.assertEqual(
            assembly_cls.call_args_list,
            [
                call(
                    max_msa_size=16,
                    pair_msa_as_unpaired_only=True,
                    shuffle_pair_msa_when_unpaired_only=True,
                ),
                call(
                    max_msa_size=16,
                    pair_msa_as_unpaired_only=False,
                    shuffle_pair_msa_when_unpaired_only=False,
                ),
            ],
        )
        self.assertFalse(featurizer.pair_msa_as_unpaired_only)
        self.assertFalse(featurizer.shuffle_pair_msa_when_unpaired_only)

    def test_invalid_or_inconsistent_row_features_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "non-empty rank-2"):
            make_query_only_msa_features(
                {"msa": np.empty((0, 4), dtype=np.int8)}
            )

        with self.assertRaisesRegex(ValueError, "must share msa depth 2"):
            make_query_only_msa_features(
                {
                    "msa": np.zeros((2, 4), dtype=np.int8),
                    "deletion_matrix": np.zeros((1, 4), dtype=np.int8),
                }
            )


if __name__ == "__main__":
    unittest.main()
