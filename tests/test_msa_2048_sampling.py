import unittest
from unittest.mock import patch

import torch

from protenix.model.utils import (
    sample_indices,
    sample_msa_feature_dict_random_without_replacement,
)


class Msa2048SamplingTest(unittest.TestCase):
    @staticmethod
    def _features(depth: int):
        row_ids = torch.arange(depth, dtype=torch.int64)
        return (
            {
                "msa": row_ids[:, None],
                "deletion_value": (row_ids * 10)[:, None],
                "profile": torch.arange(5, dtype=torch.float32),
            },
            {"msa": 0, "deletion_value": 0},
        )

    def test_lower_and_cutoff_2048_select_exactly_2048_without_replacement(self):
        torch.manual_seed(20260715)
        features, dimensions = self._features(4096)

        sampled = sample_msa_feature_dict_random_without_replacement(
            feat_dict=features,
            dim_dict=dimensions,
            cutoff=2048,
            lower_bound=2048,
            strategy="random",
        )

        selected = sampled["msa"].squeeze(-1)
        self.assertEqual(selected.numel(), 2048)
        self.assertEqual(torch.unique(selected).numel(), 2048)
        torch.testing.assert_close(sampled["deletion_value"].squeeze(-1), selected * 10)

    def test_lower_and_cutoff_2048_take_all_rows_when_source_is_shallow(self):
        torch.manual_seed(20260715)
        features, dimensions = self._features(127)

        sampled = sample_msa_feature_dict_random_without_replacement(
            feat_dict=features,
            dim_dict=dimensions,
            cutoff=2048,
            lower_bound=2048,
            strategy="random",
        )

        selected = sampled["msa"].squeeze(-1)
        self.assertEqual(selected.numel(), 127)
        self.assertEqual(set(selected.tolist()), set(range(127)))

    def test_sample_indices_obeys_the_same_exact_depth_boundary(self):
        torch.manual_seed(7)
        deep = sample_indices(n=3000, lower_bound=2048, strategy="random")[:2048]
        shallow = sample_indices(n=73, lower_bound=2048, strategy="random")[:2048]

        self.assertEqual(deep.numel(), 2048)
        self.assertEqual(torch.unique(deep).numel(), 2048)
        self.assertEqual(shallow.numel(), 73)
        self.assertEqual(torch.unique(shallow).numel(), 73)

    def test_minimum_16_does_not_mean_a_fixed_2048_rows(self):
        # Force the legal lower endpoint of randint.  This directly proves
        # that lower_bound=16 is a variable-depth policy even with cutoff=2048.
        with patch(
            "protenix.model.utils.torch.randint",
            return_value=torch.tensor([16]),
        ):
            indices = sample_indices(n=4096, lower_bound=16, strategy="topk")

        self.assertEqual(indices.numel(), 16)
        self.assertNotEqual(indices.numel(), 2048)


if __name__ == "__main__":
    unittest.main()
