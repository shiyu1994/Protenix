import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from ml_collections.config_dict import ConfigDict

from protenix.data.constants import RNA_CHAIN
from protenix.data.msa.msa_utils import RawMsa
from protenix.utils.permutation.chain_permutation.heuristic import MultiChainPermutation


class DataEdgeCaseTest(unittest.TestCase):
    def test_raw_msa_uses_query_only_when_first_rna_sequence_length_mismatches(self):
        msa = RawMsa(
            query="GCU",
            chain_type=RNA_CHAIN,
            sequences=["CU"],
            descriptions=["mismatched first sequence"],
            deduplicate=False,
        )

        self.assertEqual(msa.seqs, ["GCU"])
        self.assertEqual(msa.descs, ["Original query"])

    def test_chain_permutation_keeps_original_match_when_no_valid_anchor_exists(self):
        permutation = MultiChainPermutation(
            use_center_rmsd=False,
            find_gt_anchor_first=False,
            accept_it_as_it_is=False,
        )
        permutation.unpermuted_match = {1: 1}
        permutation.pred_asym_dict = {
            1: {"coordinate": torch.zeros(3, 3), "mol_atom_index": torch.arange(3)}
        }
        permutation.label_asym_dict = {
            1: {"coordinate_mask": torch.ones(4), "mol_atom_index": torch.arange(4)}
        }
        permutation.label_token_dict = {"entity_to_asym": {10: torch.tensor([1])}}
        permutation.pred_token_dict = {
            "entity_to_asym": {10: torch.tensor([1])},
            "entity_mol_id": torch.tensor([10, 10, 10]),
            "is_ligand": torch.tensor([False, False, False]),
        }

        self.assertEqual(permutation.compute_best_match_heuristic(), {1: 1})

    def test_get_msa_featurizer_uses_stage_specific_max_size(self):
        from protenix.data.pipeline.dataset import get_msa_featurizer

        with TemporaryDirectory() as tmpdir:
            mapping_path = Path(tmpdir) / "empty.json"
            mapping_path.write_text("{}", encoding="utf-8")
            configs = ConfigDict(
                {
                    "data": {
                        "msa": {
                            "prot_seq_or_filename_to_msadir_jsons": [str(mapping_path)],
                            "prot_msadir_raw_paths": [tmpdir],
                            "rna_seq_or_filename_to_msadir_jsons": [str(mapping_path)],
                            "rna_msadir_raw_paths": [tmpdir],
                            "prot_pairing_dbs": ["pairing"],
                            "prot_non_pairing_dbs": ["non_pairing"],
                            "prot_indexing_methods": ["sequence"],
                            "rna_indexing_methods": ["sequence"],
                            "enable_prot_msa": True,
                            "enable_rna_msa": True,
                            "pair_msa_as_unpaired_only": True,
                            "shuffle_pair_msa_when_unpaired_only": True,
                            "max_size": {
                                "train": 7,
                                "test": 11,
                            },
                            "max_input_paired_msa_depth": 5,
                            "max_input_unpaired_msa_depth": 6,
                        },
                        "toy_dataset": {},
                        "antibody_dataset": {
                            "msa": {
                                "pair_msa_as_unpaired_only": False,
                                "shuffle_pair_msa_when_unpaired_only": False,
                                "max_input_unpaired_msa_depth": 11,
                            },
                        },
                    }
                }
            )

            featurizer = get_msa_featurizer(configs, "toy_dataset", stage="train")
            antibody_featurizer = get_msa_featurizer(
                configs, "antibody_dataset", stage="train"
            )
            eval_featurizer = get_msa_featurizer(
                configs, "toy_dataset", stage="test"
            )

        self.assertEqual(featurizer.max_msa_size, 7)
        self.assertEqual(featurizer.max_input_paired_msa_depth, 5)
        self.assertEqual(featurizer.max_input_unpaired_msa_depth, 6)
        self.assertIsNone(featurizer.max_input_msa_depth)
        self.assertEqual(
            antibody_featurizer.max_input_paired_msa_depth,
            5,
        )
        self.assertEqual(
            antibody_featurizer.max_input_unpaired_msa_depth,
            11,
        )
        self.assertEqual(eval_featurizer.max_msa_size, 11)
        self.assertIsNone(eval_featurizer.max_input_paired_msa_depth)
        self.assertIsNone(eval_featurizer.max_input_unpaired_msa_depth)
        self.assertIsNone(eval_featurizer.max_input_msa_depth)
        self.assertTrue(featurizer.pair_msa_as_unpaired_only)
        self.assertTrue(featurizer.shuffle_pair_msa_when_unpaired_only)
        self.assertFalse(antibody_featurizer.pair_msa_as_unpaired_only)
        self.assertFalse(antibody_featurizer.shuffle_pair_msa_when_unpaired_only)

    def test_antibody_datasets_override_unpaired_only_pairing_mode(self):
        from configs.configs_data import data_configs

        self.assertFalse(
            data_configs["sabdab-train"]["msa"]["pair_msa_as_unpaired_only"]
        )
        self.assertFalse(
            data_configs["distill-sabdab"]["msa"]["pair_msa_as_unpaired_only"]
        )


if __name__ == "__main__":
    unittest.main()
