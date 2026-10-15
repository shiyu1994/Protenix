import unittest
from collections import Counter
from unittest.mock import patch

import torch

from protenix.model.modules import pairformer as pairformer_module
from protenix.model.modules.pairformer import MSAModule, MSAStack
from protenix.model.utils import (
    sample_indices,
    sample_msa_feature_dict_random_without_replacement,
)


class TestMSARandomCappedSampling(unittest.TestCase):
    @staticmethod
    def _row_id_features(depth: int):
        row_ids = torch.arange(depth, dtype=torch.int64)
        return (
            {
                "msa": row_ids[:, None],
                "has_deletion": (row_ids + 100_000)[:, None],
                "deletion_value": (row_ids + 200_000)[:, None],
            },
            {
                "msa": 0,
                "has_deletion": 0,
                "deletion_value": 0,
            },
        )

    def _draw_depth_histogram(
        self,
        *,
        source_depth: int,
        cutoff: int,
        draws: int,
        seed: int,
    ) -> Counter:
        features, dimensions = self._row_id_features(source_depth)
        histogram = Counter()
        torch.manual_seed(seed)

        for _ in range(draws):
            sampled = sample_msa_feature_dict_random_without_replacement(
                feat_dict=features,
                dim_dict=dimensions,
                cutoff=cutoff,
                lower_bound=1,
                strategy="random",
            )
            selected = sampled["msa"].squeeze(-1)
            sample_depth = selected.numel()
            histogram[sample_depth] += 1

            # randperm-based row selection must be without replacement, and
            # every MSA-aligned feature must use exactly the same row indices.
            self.assertEqual(torch.unique(selected).numel(), sample_depth)
            torch.testing.assert_close(
                sampled["has_deletion"].squeeze(-1),
                selected + 100_000,
            )
            torch.testing.assert_close(
                sampled["deletion_value"].squeeze(-1),
                selected + 200_000,
            )

        return histogram

    def assert_approximately_discrete_uniform(
        self,
        histogram: Counter,
        *,
        upper_bound: int,
        draws: int,
    ) -> None:
        # This checks both reachable endpoints and the intended distribution.
        # A 20% deterministic tolerance is much wider than ordinary sampling
        # noise here, while decisively rejecting min(U(1, n), cutoff), whose
        # cutoff endpoint accumulates all probability mass above the cap.
        self.assertEqual(set(histogram), set(range(1, upper_bound + 1)))
        expected = draws / upper_bound
        for sample_depth in range(1, upper_bound + 1):
            relative_error = abs(histogram[sample_depth] - expected) / expected
            self.assertLess(
                relative_error,
                0.20,
                msg=(
                    f"depth {sample_depth} count {histogram[sample_depth]} is "
                    f"not approximately UniformInteger(1, {upper_bound})"
                ),
            )

    def test_shallow_pool_is_uniform_from_one_through_actual_depth(self):
        draws = 4096
        histogram = self._draw_depth_histogram(
            source_depth=9,
            cutoff=16,
            draws=draws,
            seed=20260716,
        )

        self.assert_approximately_discrete_uniform(
            histogram,
            upper_bound=9,
            draws=draws,
        )

    def test_deep_pool_caps_before_the_uniform_depth_draw(self):
        draws = 4096
        histogram = self._draw_depth_histogram(
            source_depth=17,
            cutoff=7,
            draws=draws,
            seed=20260717,
        )

        self.assert_approximately_discrete_uniform(
            histogram,
            upper_bound=7,
            draws=draws,
        )

    def test_16384_row_pool_never_samples_more_than_2048_unique_rows(self):
        features, dimensions = self._row_id_features(16_384)
        torch.manual_seed(20260718)

        for _ in range(16):
            sampled = sample_msa_feature_dict_random_without_replacement(
                feat_dict=features,
                dim_dict=dimensions,
                cutoff=2048,
                lower_bound=1,
                strategy="random",
                query_inclusion_max_sample_size=1024,
            )
            selected = sampled["msa"].squeeze(-1)
            self.assertGreaterEqual(selected.numel(), 1)
            self.assertLessEqual(selected.numel(), 2048)
            self.assertEqual(torch.unique(selected).numel(), selected.numel())
            self.assertGreaterEqual(int(selected.min()), 0)
            self.assertLess(int(selected.max()), 16_384)

    def test_deep_pool_draws_depth_from_exact_inclusive_one_to_2048_range(self):
        with patch(
            "protenix.model.utils.torch.randint",
            return_value=torch.tensor([1]),
        ) as randint:
            selected = sample_indices(
                n=16_384,
                lower_bound=1,
                upper_bound=2048,
                strategy="topk",
            )

        self.assertEqual(selected.numel(), 1)
        randint.assert_called_once_with(low=1, high=2049, size=(1,))

    def test_sampled_depth_one_is_the_query_row(self):
        features, dimensions = self._row_id_features(16_384)
        reverse_permutation = torch.arange(16_383, -1, -1)

        with (
            patch(
                "protenix.model.utils.torch.randint",
                return_value=torch.tensor([1]),
            ),
            patch(
                "protenix.model.utils.torch.randperm",
                return_value=reverse_permutation,
            ) as randperm,
        ):
            sampled = sample_msa_feature_dict_random_without_replacement(
                feat_dict=features,
                dim_dict=dimensions,
                cutoff=2048,
                lower_bound=1,
                strategy="random",
                query_inclusion_max_sample_size=1024,
            )

        selected = sampled["msa"].squeeze(-1)
        self.assertEqual(selected.tolist(), [0])
        self.assertEqual(sampled["has_deletion"].item(), 100_000)
        self.assertEqual(sampled["deletion_value"].item(), 200_000)
        randperm.assert_called_once_with(n=16_384, device=torch.device("cpu"))

    def test_query_is_reserved_without_growing_depth_at_1024_boundary(self):
        features, dimensions = self._row_id_features(2048)
        reverse_permutation = torch.arange(2047, -1, -1)

        with (
            patch(
                "protenix.model.utils.torch.randint",
                return_value=torch.tensor([1024]),
            ),
            patch(
                "protenix.model.utils.torch.randperm",
                return_value=reverse_permutation,
            ),
        ):
            sampled = sample_msa_feature_dict_random_without_replacement(
                feat_dict=features,
                dim_dict=dimensions,
                cutoff=2048,
                lower_bound=1,
                strategy="random",
                query_inclusion_max_sample_size=1024,
            )

        selected = sampled["msa"].squeeze(-1)
        self.assertEqual(selected.numel(), 1024)
        self.assertEqual(torch.unique(selected).numel(), 1024)
        self.assertEqual(selected[0].item(), 0)
        self.assertEqual((selected == 0).sum().item(), 1)
        torch.testing.assert_close(
            sampled["has_deletion"].squeeze(-1),
            selected + 100_000,
        )
        torch.testing.assert_close(
            sampled["deletion_value"].squeeze(-1),
            selected + 200_000,
        )

    def test_naturally_selected_query_is_not_duplicated(self):
        permutation = torch.cat(
            [torch.zeros(1, dtype=torch.long), torch.arange(1, 64)]
        )

        with (
            patch(
                "protenix.model.utils.torch.randint",
                return_value=torch.tensor([17]),
            ),
            patch(
                "protenix.model.utils.torch.randperm",
                return_value=permutation,
            ),
        ):
            selected = sample_indices(
                n=64,
                lower_bound=1,
                upper_bound=64,
                strategy="random",
                query_inclusion_max_sample_size=1024,
            )

        self.assertEqual(selected.numel(), 17)
        self.assertEqual(torch.unique(selected).numel(), 17)
        self.assertEqual(selected[0].item(), 0)
        self.assertEqual((selected == 0).sum().item(), 1)

    def test_sampled_depth_above_1024_keeps_the_original_random_policy(self):
        reverse_permutation = torch.arange(2047, -1, -1)

        with (
            patch(
                "protenix.model.utils.torch.randint",
                return_value=torch.tensor([1025]),
            ),
            patch(
                "protenix.model.utils.torch.randperm",
                return_value=reverse_permutation,
            ),
        ):
            selected = sample_indices(
                n=2048,
                lower_bound=1,
                upper_bound=2048,
                strategy="random",
                query_inclusion_max_sample_size=1024,
            )

        self.assertEqual(selected.numel(), 1025)
        self.assertEqual(torch.unique(selected).numel(), 1025)
        self.assertNotIn(0, selected.tolist())
        torch.testing.assert_close(selected, reverse_permutation[:1025])

    def test_general_profile_is_exact_2048_for_deep_pool_and_all_for_shallow_pool(self):
        torch.manual_seed(20260720)
        deep_features, deep_dimensions = self._row_id_features(4096)
        deep = sample_msa_feature_dict_random_without_replacement(
            feat_dict=deep_features,
            dim_dict=deep_dimensions,
            cutoff=2048,
            lower_bound=2048,
            strategy="random",
        )
        deep_rows = deep["msa"].squeeze(-1)
        self.assertEqual(deep_rows.numel(), 2048)
        self.assertEqual(torch.unique(deep_rows).numel(), 2048)

        shallow_features, shallow_dimensions = self._row_id_features(127)
        shallow = sample_msa_feature_dict_random_without_replacement(
            feat_dict=shallow_features,
            dim_dict=shallow_dimensions,
            cutoff=2048,
            lower_bound=2048,
            strategy="random",
        )
        shallow_rows = shallow["msa"].squeeze(-1)
        self.assertEqual(shallow_rows.numel(), 127)
        self.assertEqual(set(shallow_rows.tolist()), set(range(127)))

    def test_msa_module_resamples_and_pins_query_in_train_and_eval(self):
        real_sampler = (
            pairformer_module.sample_msa_feature_dict_random_without_replacement
        )
        selected_rows = []
        query_inclusion_limits = []

        def recording_sampler(*args, **kwargs):
            sampled = real_sampler(*args, **kwargs)
            selected_rows.append(tuple(sampled["msa"][:, 0].tolist()))
            query_inclusion_limits.append(
                kwargs["query_inclusion_max_sample_size"]
            )
            return sampled

        def bypass_blocks(
            blocks,
            args,
            blocks_per_ckpt,
            checkpoint_cpu_offload_every_n=None,
        ):
            del blocks, blocks_per_ckpt, checkpoint_cpu_offload_every_n
            return args

        module = MSAModule(
            n_blocks=1,
            c_m=2,
            c_z=2,
            c_s_inputs=3,
            blocks_per_ckpt=1,
            msa_chunk_size=2048,
            msa_max_size=16_384,
            msa_configs={
                "enable": True,
                "strategy": "random",
                "sample_cutoff": {"train": 2048, "test": 2048},
                "min_size": {"train": 1, "test": 1},
                "query_inclusion_max_sample_size": {
                    "train": 1024,
                    "test": 512,
                },
            },
        )
        module.train()
        row_ids = torch.arange(32, dtype=torch.long)[:, None]
        features = {
            "msa": row_ids,
            "has_deletion": torch.zeros(32, 1),
            "deletion_value": torch.zeros(32, 1),
        }
        z = torch.zeros(1, 1, 2)
        s_inputs = torch.zeros(1, 3)
        pair_mask = torch.ones(1, 1)

        torch.manual_seed(20260719)
        with (
            patch.object(
                pairformer_module,
                "sample_msa_feature_dict_random_without_replacement",
                side_effect=recording_sampler,
            ),
            patch.object(
                pairformer_module,
                "checkpoint_blocks",
                side_effect=bypass_blocks,
            ),
            torch.no_grad(),
        ):
            for _ in range(6):
                module(features, z, s_inputs, pair_mask)
            module.eval()
            module(features, z, s_inputs, pair_mask)

        self.assertEqual(len(selected_rows), 7)
        self.assertGreater(len(set(selected_rows)), 1)
        self.assertTrue(all(rows[0] == 0 for rows in selected_rows))
        self.assertEqual(query_inclusion_limits, [1024] * 6 + [512])
        self.assertEqual(len(module.msa_depth_trace), 7)
        self.assertTrue(
            all(record["input_depth"] == 32 for record in module.msa_depth_trace)
        )
        self.assertTrue(
            all(
                1 <= record["sampled_depth"] <= 32
                for record in module.msa_depth_trace
            )
        )
        self.assertEqual(module.msa_max_size, 2048)
        self.assertEqual(module.msa_chunk_size, 2048)

    def test_training_compute_height_is_2048_and_chunk2048_is_one_block(self):
        weighted_depths = []
        transition_depths = []

        class RecordingWeightedAverage(torch.nn.Module):
            def forward(self, m, z):
                del z
                weighted_depths.append(m.shape[0])
                return torch.zeros_like(m)

        class RecordingTransition(torch.nn.Module):
            def forward(self, m):
                transition_depths.append(m.shape[0])
                return torch.zeros_like(m)

        stack = MSAStack(
            c_m=2,
            c_z=3,
            c=1,
            dropout=0.0,
            msa_chunk_size=2048,
            msa_max_size=2048,
        )
        stack.msa_pair_weighted_averaging = RecordingWeightedAverage()
        stack.transition_m = RecordingTransition()
        stack.train()
        m = torch.randn(37, 2, 2)
        z = torch.randn(2, 2, 3)

        direct_checkpoint = lambda module, *args: module(*args)
        with patch.object(
            pairformer_module,
            "get_checkpoint_fn",
            return_value=direct_checkpoint,
        ):
            output = stack(m, z)

        self.assertEqual(output.shape, m.shape)
        self.assertEqual(weighted_depths, [2048])
        self.assertEqual(transition_depths, [2048])


if __name__ == "__main__":
    unittest.main()
