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

import gzip
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from ml_collections.config_dict import ConfigDict

from protenix.data.constants import PROTEIN_CHAIN
from protenix.data.msa.msa_featurizer import (
    DIRECT_PROTEIN_UNPAIRED_MSA_PATH,
    MSAFeaturizer,
)
from protenix.data.msa.msa_utils import RawMsa
from protenix.data.pipeline.dataset import (
    BaseSingleDataset,
    get_template_featurizer,
)


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


class TestDistillationDirectIO(unittest.TestCase):
    def test_catalog_row_resolves_nested_pickle_and_sibling_a3m(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_root = tmp_path / "source"
            leaf = source_root / "bucket" / "0001"
            bioassembly_path = leaf / "bioassembly" / "sample.pkl.gz"
            a3m_path = leaf / "sample.a3m"
            bioassembly_path.parent.mkdir(parents=True)
            a3m_path.write_text(">query\nACDEF\n", encoding="utf-8")
            with gzip.open(bioassembly_path, "wb") as f:
                pickle.dump({"sentinel": "loaded"}, f)

            roots_path = tmp_path / "roots.json"
            roots_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "roots": [{"root_id": "0", "path": str(source_root)}],
                    }
                ),
                encoding="utf-8",
            )

            dataset = object.__new__(BaseSingleDataset)
            dataset.source_roots = BaseSingleDataset._load_source_roots(
                roots_path
            )
            dataset.bioassembly_dict_dir = tmp_path / "legacy-flat-dir"
            dataset.group_by_pdb_id = False
            dataset.indices_list = pd.DataFrame(
                [
                    {
                        "pdb_id": "sample",
                        "type": "chain",
                        "source_root_id": "0",
                        "cif_relpath": "bucket/0001/sample.cif",
                    }
                ]
            )

            sample, bioassembly, resolved_path = dataset._get_bioassembly_data(0)

            self.assertEqual(sample["pdb_id"], "sample")
            self.assertEqual(Path(resolved_path), bioassembly_path)
            self.assertEqual(bioassembly["sentinel"], "loaded")
            self.assertEqual(
                Path(bioassembly[DIRECT_PROTEIN_UNPAIRED_MSA_PATH]),
                a3m_path,
            )

    def test_catalog_relative_path_cannot_escape_source_root(self):
        dataset = object.__new__(BaseSingleDataset)
        dataset.source_roots = {"0": Path("/source")}
        row = pd.Series(
            {
                "source_root_id": "0",
                "cif_relpath": "../outside/sample.cif",
            }
        )

        with self.assertRaisesRegex(ValueError, "Unsafe"):
            dataset._resolve_row_addressed_distillation_paths(row)

    def test_direct_a3m_is_used_as_unpaired_protein_msa(self):
        sequence = "ACDEF"
        a3m = ">query\nACDEF\n>homolog\nACDDF\n"
        with tempfile.TemporaryDirectory() as tmp:
            a3m_path = Path(tmp) / "sample.a3m"
            a3m_path.write_text(a3m, encoding="utf-8")

            atom_array = _FakeAtomArray(
                asym_id_int=[1] * len(sequence),
                chain_id=["A"] * len(sequence),
                res_id=range(1, len(sequence) + 1),
            )
            bioassembly = {
                "pdb_id": "sample",
                "atom_array": atom_array,
                "token_array": _FakeTokenArray(np.arange(len(sequence))),
                "sequences": {"1": sequence},
                "entity_poly_type": {"1": "polypeptide(L)"},
                DIRECT_PROTEIN_UNPAIRED_MSA_PATH: str(a3m_path),
            }
            featurizer = MSAFeaturizer(
                dataset_name="direct-distillation",
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
            )

            with patch.object(featurizer.prot_mgr, "fetch_msas") as legacy_fetch:
                features = featurizer.make_msa_features(
                    bioassembly_dict=bioassembly,
                    selected_indices=None,
                    entity_to_asym_id_int={"1": [1]},
                )

            legacy_fetch.assert_not_called()
            unpaired = RawMsa.from_a3m(
                sequence, PROTEIN_CHAIN, a3m, dedup=True
            ).featurize()["msa"]
            # The standard assembly line retains its paired query placeholder
            # ahead of the unpaired rows, matching every existing protein MSA.
            expected = np.concatenate([unpaired[:1], unpaired], axis=0)
            np.testing.assert_array_equal(features["msa"], expected)
            self.assertEqual(int(features["prot_unpair_num_alignments"]), 2)
            self.assertEqual(int(features["prot_pair_num_alignments"]), 1)

    def test_direct_a3m_rows_use_independent_crop_segment_permutations(self):
        sequence = "A" * 42
        a3m = (
            f">query\n{sequence}\n"
            f">hit_1\n{'C' * 42}\n"
            f">hit_2\n{'D' * 42}\n"
            f">hit_3\n{'E' * 42}\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            a3m_path = Path(tmp) / "sample.a3m"
            a3m_path.write_text(a3m, encoding="utf-8")
            atom_array = _FakeAtomArray(
                asym_id_int=[1] * len(sequence),
                chain_id=["A"] * len(sequence),
                res_id=range(1, len(sequence) + 1),
            )
            bioassembly = {
                "pdb_id": "sample",
                "atom_array": atom_array,
                "token_array": _FakeTokenArray(np.arange(len(sequence))),
                "sequences": {"1": sequence},
                "entity_poly_type": {"1": "polypeptide(L)"},
                DIRECT_PROTEIN_UNPAIRED_MSA_PATH: str(a3m_path),
            }
            featurizer = MSAFeaturizer(
                dataset_name="direct-distillation-segment-test",
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
            selected_indices = np.concatenate(
                (np.arange(9), np.arange(32, 41))
            )

            with patch.object(featurizer.prot_mgr, "fetch_msas") as legacy_fetch:
                features = featurizer.make_msa_features(
                    bioassembly_dict=bioassembly,
                    selected_indices=selected_indices,
                    entity_to_asym_id_int={"1": [1]},
                    pair_msa_as_unpaired_only=True,
                    shuffle_pair_msa_when_unpaired_only=True,
                    virtual_segment_shuffle_seed=4,
                )

        legacy_fetch.assert_not_called()
        profile = featurizer.get_last_profile()
        self.assertEqual(profile["virtual_segment_count"], 2)
        self.assertTrue(profile["virtual_segment_shuffle_applied"])
        np.testing.assert_array_equal(
            features["msa"][0, :9],
            features["msa"][0, 9:],
        )
        self.assertTrue(
            np.any(features["msa"][2:, 0] != features["msa"][2:, 9])
        )

    def test_template_can_be_disabled_before_loading_template_mappings(self):
        configs = ConfigDict(
            {
                "data": {
                    "template": {
                        "enable_prot_template": True,
                        "prot_seq_or_filename_to_templatedir_jsons": [
                            "/does/not/exist.json"
                        ],
                    },
                    "distillation": {
                        "template": {"enable_prot_template": False}
                    },
                }
            }
        )

        self.assertIsNone(
            get_template_featurizer(
                configs,
                dataset_name="distillation",
                stage="train",
            )
        )


if __name__ == "__main__":
    unittest.main()
