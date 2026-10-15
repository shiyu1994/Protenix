import unittest

import torch

from protenix.data.pipeline.dataloader import DistributedWeightedSampler


class _SizedDataset:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


class TestDistributedWeightedSampler(unittest.TestCase):
    def test_eight_ranks_cover_800_samples_exactly_once(self):
        dataset = _SizedDataset(800)
        weights = torch.ones(800)
        per_rank = []
        for rank in range(8):
            sampler = DistributedWeightedSampler(
                dataset,
                weights,
                num_samples=800,
                num_replicas=8,
                rank=rank,
                replacement=False,
                seed=42,
            )
            indices = list(sampler)
            self.assertEqual(len(sampler), 100)
            self.assertEqual(len(indices), 100)
            self.assertEqual(len(set(indices)), 100)
            per_rank.append(set(indices))

        for left in range(8):
            for right in range(left + 1, 8):
                self.assertFalse(per_rank[left] & per_rank[right])
        self.assertEqual(set().union(*per_rank), set(range(800)))

    def test_without_replacement_rejects_invalid_global_sample_count(self):
        dataset = _SizedDataset(800)
        weights = torch.ones(800)
        with self.assertRaisesRegex(ValueError, "divisible"):
            DistributedWeightedSampler(
                dataset,
                weights,
                num_samples=799,
                num_replicas=8,
                rank=0,
                replacement=False,
            )

    def test_without_replacement_rejects_zero_weight_shortfall(self):
        dataset = _SizedDataset(800)
        weights = torch.ones(800)
        weights[-1] = 0
        with self.assertRaisesRegex(ValueError, "positive-weight"):
            DistributedWeightedSampler(
                dataset,
                weights,
                num_samples=800,
                num_replicas=8,
                rank=0,
                replacement=False,
            )


if __name__ == "__main__":
    unittest.main()
