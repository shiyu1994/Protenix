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
from types import MethodType
from unittest.mock import patch

from protenix.data.pipeline.dataloader import (
    DistributedWeightedSampler,
    WeightedSampler,
)
from protenix.data.pipeline.dataset import BaseSingleDataset, WeightedMultiDataset
from protenix.data.pipeline.sample_context import (
    MSA_MODE_QUERY_ONLY,
    MSA_MODE_SPECIES_PAIRED,
    MSA_MODE_UNPAIRED_SHUFFLE,
    SampleOccurrence,
    deterministic_occurrence_bernoulli,
    deterministic_occurrence_categorical,
)


class _RecordingDataset:
    def __init__(self, size):
        self.size = size
        self.last_index = None

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.last_index = index
        return {"basic": {}}


class TestSampleOccurrenceAugmentations(unittest.TestCase):
    def test_weighted_sampler_emits_resumable_occurrence_positions(self):
        sampler = WeightedSampler(
            weights=[1.0, 2.0, 3.0],
            num_samples=8,
            replacement=True,
            seed=17,
            emit_sample_occurrence=True,
        )
        sampler.set_epoch(3)
        complete = list(sampler)
        self.assertEqual(
            [item.global_position for item in complete],
            list(range(8)),
        )
        self.assertTrue(
            all(
                item.sampler_seed == 17 and item.sampler_epoch == 3
                for item in complete
            )
        )

        sampler.set_start_offset(5)
        self.assertEqual(list(sampler), complete[5:])

    def test_distributed_occurrence_positions_are_global_and_disjoint(self):
        positions = []
        for rank in range(4):
            sampler = DistributedWeightedSampler(
                dataset=list(range(5)),
                weights=[1.0] * 5,
                num_samples=12,
                num_replicas=4,
                rank=rank,
                replacement=True,
                seed=9,
                emit_sample_occurrence=True,
            )
            rank_items = list(sampler)
            self.assertTrue(
                all(item.global_position % 4 == rank for item in rank_items)
            )
            positions.extend(item.global_position for item in rank_items)
        self.assertEqual(sorted(positions), list(range(12)))

    def test_independent_hash_streams_are_stable_and_near_target_rates(self):
        msa_modes = []
        template = []
        for position in range(20_000):
            occurrence = SampleOccurrence(
                index=position % 7,
                sampler_seed=42,
                sampler_epoch=position // 10_240,
                global_position=position % 10_240,
            )
            msa_modes.append(
                deterministic_occurrence_categorical(
                    occurrence,
                    probabilities=(
                        (MSA_MODE_QUERY_ONLY, 0.2),
                        (MSA_MODE_UNPAIRED_SHUFFLE, 0.3),
                        (MSA_MODE_SPECIES_PAIRED, 0.5),
                    ),
                    stream_name="msa_mode",
                )
            )
            template.append(
                deterministic_occurrence_bernoulli(
                    occurrence,
                    probability=0.2,
                    stream_name="template_use",
                )
            )
        for mode, expected in (
            (MSA_MODE_QUERY_ONLY, 0.2),
            (MSA_MODE_UNPAIRED_SHUFFLE, 0.3),
            (MSA_MODE_SPECIES_PAIRED, 0.5),
        ):
            self.assertAlmostEqual(
                msa_modes.count(mode) / len(msa_modes),
                expected,
                delta=0.015,
            )
        self.assertAlmostEqual(sum(template) / len(template), 0.2, delta=0.015)
        self.assertNotEqual(
            [mode == MSA_MODE_QUERY_ONLY for mode in msa_modes],
            template,
        )

        occurrence = SampleOccurrence(3, 42, 7, 91)
        first = deterministic_occurrence_categorical(
            occurrence,
            probabilities=(
                (MSA_MODE_QUERY_ONLY, 0.2),
                (MSA_MODE_UNPAIRED_SHUFFLE, 0.3),
                (MSA_MODE_SPECIES_PAIRED, 0.5),
            ),
            stream_name="msa_mode",
        )
        second = deterministic_occurrence_categorical(
            occurrence,
            probabilities=(
                (MSA_MODE_QUERY_ONLY, 0.2),
                (MSA_MODE_UNPAIRED_SHUFFLE, 0.3),
                (MSA_MODE_SPECIES_PAIRED, 0.5),
            ),
            stream_name="msa_mode",
        )
        self.assertEqual(first, second)

    def test_multi_dataset_mapping_preserves_occurrence_identity(self):
        left = _RecordingDataset(2)
        right = _RecordingDataset(3)
        dataset = WeightedMultiDataset(
            datasets=[left, right],
            dataset_names=["left", "right"],
            datapoint_weights=[[1.0, 1.0], [1.0, 1.0, 1.0]],
            dataset_sample_weights=[0.4, 0.6],
        )
        occurrence = SampleOccurrence(
            index=4,
            sampler_seed=42,
            sampler_epoch=2,
            global_position=99,
        )

        result = dataset[occurrence]

        self.assertIsNone(left.last_index)
        self.assertEqual(
            right.last_index,
            SampleOccurrence(2, 42, 2, 99),
        )
        self.assertEqual(result["basic"]["dataset_name"], "right")

    def test_dataset_decisions_stay_fixed_when_failed_row_is_replaced(self):
        dataset = object.__new__(BaseSingleDataset)
        dataset.query_only_msa_probability = 0.2
        dataset.pair_msa_as_unpaired_probability = 0.3
        dataset.template_use_probability = 0.2
        dataset.augmentation_decision_version = "sample_occurrence_sha256_v1"
        dataset.random_sample_if_failed = True
        dataset.indices_list = [0, 1]
        calls = []

        def process_one(
            self,
            index,
            *,
            msa_mode,
            template_use_requested,
            sample_occurrence,
        ):
            calls.append(
                (
                    index,
                    msa_mode,
                    template_use_requested,
                    sample_occurrence,
                )
            )
            if len(calls) == 1:
                raise RuntimeError("synthetic broken row")
            return {"basic": {}}

        dataset.process_one = MethodType(process_one, dataset)
        dataset.save_error_data = MethodType(lambda self, index, error: None, dataset)
        occurrence = SampleOccurrence(0, 42, 3, 77)

        with patch(
            "protenix.data.pipeline.dataset.random.choice",
            return_value=1,
        ):
            result = dataset[occurrence]

        self.assertEqual(result, {"basic": {}})
        self.assertEqual([call[0] for call in calls], [0, 1])
        self.assertEqual(calls[0][1:3], calls[1][1:3])
        self.assertEqual(calls[0][3], calls[1][3])

    def test_nontrivial_probability_requires_occurrence_context(self):
        with self.assertRaisesRegex(ValueError, "SampleOccurrence"):
            deterministic_occurrence_categorical(
                None,
                probabilities=(
                    (MSA_MODE_QUERY_ONLY, 0.2),
                    (MSA_MODE_UNPAIRED_SHUFFLE, 0.3),
                    (MSA_MODE_SPECIES_PAIRED, 0.5),
                ),
                stream_name="msa_mode",
            )

    def test_categorical_rejects_invalid_probability_contract(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            deterministic_occurrence_categorical(
                SampleOccurrence(0, 42, 0, 0),
                probabilities=(("query_only", 0.2), ("full", 0.7)),
                stream_name="msa_mode",
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            deterministic_occurrence_categorical(
                SampleOccurrence(0, 42, 0, 0),
                probabilities=(("same", 0.2), ("same", 0.8)),
                stream_name="msa_mode",
            )

    def test_dataset_rejects_overlapping_msa_mode_probability_mass(self):
        with self.assertRaisesRegex(ValueError, "must be <= 1"):
            BaseSingleDataset(
                mmcif_dir="",
                bioassembly_dict_dir=None,
                indices_fpath="",
                cropping_configs={},
                query_only_msa_probability=0.8,
                pair_msa_as_unpaired_probability=0.3,
            )


if __name__ == "__main__":
    unittest.main()
