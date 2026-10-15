import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from protenix.data.constants import PROTEIN_CHAIN
from protenix.data.msa.msa_featurizer import FeatureAssemblyLine
from protenix.data.msa.msa_utils import MSACore
from protenix.data.pipeline.data_pipeline import DataPipeline
from protenix.data.pipeline.dataset import BaseSingleDataset
from protenix.data.pipeline.sample_context import (
    AUGMENTATION_DECISION_VERSION,
    MSA_MODE_CONFIGURED,
    MSA_MODE_QUERY_ONLY,
    MSA_MODE_UNPAIRED_SHUFFLE,
    MULTICHAIN_MSA_SHUFFLE_VARIANT_STREAM,
    QUERY_ONLY_CHAIN_SELECTION_STREAM,
    SampleOccurrence,
    deterministic_occurrence_seed,
)


class MSAPairingModeTest(unittest.TestCase):
    def test_data_pipeline_records_msa_featurizer_profile(self):
        class FakeMSAFeaturizer:
            def __init__(self):
                self.profile = {
                    "virtual_segment_count": 2,
                    "virtual_segment_shuffle_applied": True,
                }

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return {"msa": np.zeros((1, 5), dtype=np.int8)}

            def set_last_profile(self, profile):
                self.profile = profile

            def get_last_profile(self):
                return self.profile

        featurizer = FakeMSAFeaturizer()
        with patch.object(
            DataPipeline,
            "get_label_entity_id_to_asym_id_int",
            return_value={"1": [1]},
        ):
            result = DataPipeline.get_msa_raw_features(
                bioassembly_dict={"atom_array": object()},
                selected_indices=np.arange(5),
                msa_featurizer=featurizer,
                query_only=False,
                pair_msa_as_unpaired_only=True,
                shuffle_pair_msa_when_unpaired_only=True,
                virtual_segment_shuffle_seed=123,
                msa_shuffle_variant_seed=234,
                query_only_chain_selection_seed=456,
            )

        self.assertEqual(result["msa"].shape, (1, 5))
        self.assertIsNotNone(featurizer.profile)
        self.assertGreaterEqual(featurizer.profile["elapsed_seconds"], 0.0)
        self.assertFalse(featurizer.profile["query_only"])
        self.assertTrue(featurizer.profile["pair_msa_as_unpaired_only"])
        self.assertTrue(
            featurizer.profile["shuffle_pair_msa_when_unpaired_only"]
        )
        self.assertEqual(featurizer.kwargs["virtual_segment_shuffle_seed"], 123)
        self.assertEqual(featurizer.kwargs["msa_shuffle_variant_seed"], 234)
        self.assertEqual(
            featurizer.kwargs["query_only_chain_selection_seed"],
            456,
        )
        self.assertEqual(featurizer.profile["virtual_segment_count"], 2)
        self.assertTrue(
            featurizer.profile["virtual_segment_shuffle_applied"]
        )

    def test_dataset_crop_forwards_query_only_chain_selection_seed(self):
        dataset = object.__new__(BaseSingleDataset)
        dataset.min_diffusion_supervised_tokens = 0
        dataset.require_resolved_protein_crop = False
        dataset.disable_non_protein_diffusion_weights = False
        dataset.msa_featurizer = object()
        dataset.template_featurizer = object()
        crop_result = (
            "no_crop",
            object(),
            object(),
            {},
            {},
            -1,
        )

        with patch.object(
            DataPipeline,
            "crop",
            return_value=crop_result,
        ) as pipeline_crop:
            result = BaseSingleDataset.crop(
                dataset,
                sample_indice=object(),
                bioassembly_dict={},
                crop_size=16,
                method_weights=[1.0, 0.0, 0.0],
                query_only_msa=True,
                msa_shuffle_variant_seed=678,
                query_only_chain_selection_seed=789,
            )

        self.assertIs(result, crop_result)
        self.assertEqual(
            pipeline_crop.call_args.kwargs["msa_shuffle_variant_seed"],
            678,
        )
        self.assertEqual(
            pipeline_crop.call_args.kwargs[
                "query_only_chain_selection_seed"
            ],
            789,
        )

    def test_process_one_derives_and_forwards_query_only_chain_selection_seed(
        self,
    ):
        dataset = object.__new__(BaseSingleDataset)
        dataset.augmentation_decision_version = AUGMENTATION_DECISION_VERSION
        dataset.msa_featurizer = SimpleNamespace(
            pair_msa_as_unpaired_only=True,
            shuffle_pair_msa_when_unpaired_only=True,
            enable_multichain_msa_shuffle_variant_sampling=True,
        )
        dataset.use_reference_chains_only = False
        dataset.shuffle_mols = False
        dataset.shuffle_sym_ids = False
        dataset.reassign_continuous_chain_ids = False
        dataset.cropping_configs = {}
        dataset._get_bioassembly_data = Mock(
            return_value=(
                object(),
                {
                    "pdb_id": "seed-forwarding",
                    "atom_array": SimpleNamespace(
                        entity_mol_id=np.asarray([0], dtype=np.int64)
                    ),
                    "token_array": object(),
                },
                "seed-forwarding.pkl",
            )
        )
        dataset.crop = Mock(side_effect=RuntimeError("stop after crop call"))
        occurrence = SampleOccurrence(
            index=3,
            sampler_seed=41,
            sampler_epoch=7,
            global_position=101,
        )
        expected_seed = deterministic_occurrence_seed(
            occurrence,
            stream_name=QUERY_ONLY_CHAIN_SELECTION_STREAM,
            algorithm_version=AUGMENTATION_DECISION_VERSION,
        )

        with self.assertRaisesRegex(RuntimeError, "stop after crop call"):
            BaseSingleDataset.process_one(
                dataset,
                idx=3,
                msa_mode=MSA_MODE_QUERY_ONLY,
                sample_occurrence=occurrence,
            )

        crop_kwargs = dataset.crop.call_args.kwargs
        self.assertTrue(crop_kwargs["query_only_msa"])
        self.assertFalse(crop_kwargs["pair_msa_as_unpaired_only"])
        self.assertFalse(
            crop_kwargs["shuffle_pair_msa_when_unpaired_only"]
        )
        self.assertIsNone(crop_kwargs["virtual_segment_shuffle_seed"])
        self.assertIsNone(crop_kwargs["msa_shuffle_variant_seed"])
        self.assertEqual(
            crop_kwargs["query_only_chain_selection_seed"],
            expected_seed,
        )

    def test_process_one_derives_independent_msa_shuffle_variant_seed(self):
        dataset = object.__new__(BaseSingleDataset)
        dataset.augmentation_decision_version = AUGMENTATION_DECISION_VERSION
        dataset.msa_featurizer = SimpleNamespace(
            pair_msa_as_unpaired_only=False,
            shuffle_pair_msa_when_unpaired_only=False,
            enable_multichain_msa_shuffle_variant_sampling=True,
        )
        dataset.use_reference_chains_only = False
        dataset.shuffle_mols = False
        dataset.shuffle_sym_ids = False
        dataset.reassign_continuous_chain_ids = False
        dataset.cropping_configs = {}
        dataset._get_bioassembly_data = Mock(
            return_value=(
                object(),
                {
                    "pdb_id": "variant-seed-forwarding",
                    "atom_array": SimpleNamespace(
                        entity_mol_id=np.asarray([0], dtype=np.int64)
                    ),
                    "token_array": object(),
                },
                "variant-seed-forwarding.pkl",
            )
        )
        dataset.crop = Mock(side_effect=RuntimeError("stop after crop call"))
        occurrence = SampleOccurrence(
            index=5,
            sampler_seed=43,
            sampler_epoch=11,
            global_position=109,
        )
        expected_variant_seed = deterministic_occurrence_seed(
            occurrence,
            stream_name=MULTICHAIN_MSA_SHUFFLE_VARIANT_STREAM,
            algorithm_version=AUGMENTATION_DECISION_VERSION,
        )

        with self.assertRaisesRegex(RuntimeError, "stop after crop call"):
            BaseSingleDataset.process_one(
                dataset,
                idx=5,
                msa_mode=MSA_MODE_UNPAIRED_SHUFFLE,
                sample_occurrence=occurrence,
            )

        crop_kwargs = dataset.crop.call_args.kwargs
        self.assertTrue(crop_kwargs["pair_msa_as_unpaired_only"])
        self.assertTrue(
            crop_kwargs["shuffle_pair_msa_when_unpaired_only"]
        )
        self.assertEqual(
            crop_kwargs["msa_shuffle_variant_seed"],
            expected_variant_seed,
        )
        self.assertNotEqual(
            crop_kwargs["msa_shuffle_variant_seed"],
            crop_kwargs["virtual_segment_shuffle_seed"],
        )
        self.assertIsNone(crop_kwargs["query_only_chain_selection_seed"])

        # Configured mode may also resolve to static all-row unpaired shuffle.
        # It uses the same independent occurrence stream rather than reaching
        # the crop with a missing seed.
        dataset.msa_featurizer.pair_msa_as_unpaired_only = True
        dataset.msa_featurizer.shuffle_pair_msa_when_unpaired_only = True
        dataset.crop.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "stop after crop call"):
            BaseSingleDataset.process_one(
                dataset,
                idx=5,
                msa_mode=MSA_MODE_CONFIGURED,
                sample_occurrence=occurrence,
            )
        configured_crop_kwargs = dataset.crop.call_args.kwargs
        self.assertEqual(
            configured_crop_kwargs["msa_shuffle_variant_seed"],
            expected_variant_seed,
        )

    def test_pair_msa_as_unpaired_only_keeps_query_as_only_paired_row(self):
        bioassembly = {
            0: {
                "entity_id": "1",
                "chain_id": "A",
                "sequence": "AAAAA",
                "chain_entity_type": PROTEIN_CHAIN,
                "paired_msa": (
                    ">query\n"
                    "AAAAA\n"
                    ">UniRef100_chainA_9606/\n"
                    "AAAAT\n"
                ),
                "unpaired_msa": (
                    ">query\n"
                    "AAAAA\n"
                    ">chainA_unpaired\n"
                    "AAATA\n"
                ),
            },
            1: {
                "entity_id": "2",
                "chain_id": "B",
                "sequence": "CCCCC",
                "chain_entity_type": PROTEIN_CHAIN,
                "paired_msa": (
                    ">query\n"
                    "CCCCC\n"
                    ">UniRef100_chainB_9606/\n"
                    "CCCCT\n"
                ),
                "unpaired_msa": (
                    ">query\n"
                    "CCCCC\n"
                    ">chainB_unpaired\n"
                    "CCACC\n"
                ),
            },
        }
        std_idxs = np.arange(10, dtype=np.int32)

        features = FeatureAssemblyLine(
            max_msa_size=10,
            pair_msa_as_unpaired_only=True,
        ).assemble(bioassembly, std_idxs).to_dict()

        expected_msa, _ = MSACore.sequences_to_array(
            [
                "AAAAACCCCC",
                "AAAATCCCCT",
                "AAATACCACC",
            ],
            PROTEIN_CHAIN,
        )

        self.assertEqual(int(features["prot_pair_num_alignments"]), 1)
        self.assertEqual(int(features["prot_unpair_num_alignments"]), 2)
        np.testing.assert_array_equal(features["msa"][:3], expected_msa)

    def test_pair_msa_as_unpaired_only_can_shuffle_paired_hits_before_merging(self):
        bioassembly = {
            0: {
                "entity_id": "1",
                "chain_id": "A",
                "sequence": "AAAAA",
                "chain_entity_type": PROTEIN_CHAIN,
                "paired_msa": (
                    ">query\n"
                    "AAAAA\n"
                    ">pair_a1\n"
                    "AAAAT\n"
                    ">pair_a2\n"
                    "AAATA\n"
                    ">pair_a3\n"
                    "AATAA\n"
                    ">pair_a4\n"
                    "ATAAA\n"
                ),
                "unpaired_msa": ">query\nAAAAA\n",
            },
            1: {
                "entity_id": "2",
                "chain_id": "B",
                "sequence": "CCCCC",
                "chain_entity_type": PROTEIN_CHAIN,
                "paired_msa": (
                    ">query\n"
                    "CCCCC\n"
                    ">pair_b1\n"
                    "CCCCT\n"
                    ">pair_b2\n"
                    "CCCAC\n"
                    ">pair_b3\n"
                    "CCACC\n"
                    ">pair_b4\n"
                    "CACCC\n"
                ),
                "unpaired_msa": ">query\nCCCCC\n",
            },
        }
        std_idxs = np.arange(10, dtype=np.int32)

        rng_state = np.random.get_state()
        try:
            np.random.seed(0)
            features = FeatureAssemblyLine(
                max_msa_size=10,
                pair_msa_as_unpaired_only=True,
                shuffle_pair_msa_when_unpaired_only=True,
            ).assemble(bioassembly, std_idxs).to_dict()
        finally:
            np.random.set_state(rng_state)

        expected_msa, _ = MSACore.sequences_to_array(
            [
                "AAAAACCCCC",
                "AATAACCCCT",
                "ATAAACCACC",
                "AAATACCCAC",
                "AAAATCACCC",
            ],
            PROTEIN_CHAIN,
        )

        self.assertEqual(int(features["prot_pair_num_alignments"]), 1)
        self.assertEqual(int(features["prot_unpair_num_alignments"]), 4)
        np.testing.assert_array_equal(features["msa"][:5], expected_msa)


if __name__ == "__main__":
    unittest.main()
