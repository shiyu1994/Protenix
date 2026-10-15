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

import math
import unittest

import torch

from protenix.model.modules.primitives import (
    rearrange_qk_to_dense_trunk,
    rearrange_to_dense_trunk,
)


def create_qkv(batch_size_dims, n_q, n_kv, d):
    q = torch.rand(size=(*batch_size_dims, n_q, d))
    k = torch.rand(size=(*batch_size_dims, n_kv, d))
    v = torch.rand(size=(*batch_size_dims, n_kv, d))
    return q, k, v


class TestUtils(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def test_equivalence(self):

        batch_size_dims = (3, 5)
        n = 128 * 2 + 18
        d = 9
        n_queries = 32
        n_keys = 128
        inf = 10e10

        torch.random.manual_seed(42)
        q, k, v = create_qkv(batch_size_dims, n, n, d)
        (
            q_trunked,
            k_trunked,
            _,
            attn_bias_trunked,
            q_pad_length,
        ) = rearrange_to_dense_trunk(
            q,
            k,
            v,
            n_queries,
            n_keys,
            inf=inf,
        )

        q_b, k_b, padding_info = rearrange_qk_to_dense_trunk(
            q, k, dim_q=-2, dim_k=-2, n_queries=n_queries, n_keys=n_keys
        )
        self.assertTrue(
            torch.allclose(
                padding_info["mask_trunked"] > 0, attn_bias_trunked[0, 0] > -1
            )
        )
        self.assertTrue(torch.allclose(q_b, q_trunked))
        self.assertTrue(torch.allclose(k_b, k_trunked))

    def test_padding_mask_boundaries(self):
        n_queries = 32
        n_keys = 128

        for n in (1, 31, 32, 33, 127, 128, 129, 274):
            with self.subTest(n=n):
                q, k, _ = create_qkv((), n, n, 3)
                _, _, padding_info = rearrange_qk_to_dense_trunk(
                    q,
                    k,
                    dim_q=-2,
                    dim_k=-2,
                    n_queries=n_queries,
                    n_keys=n_keys,
                )
                mask = padding_info["mask_trunked"]

                n_trunks = math.ceil(n / n_queries)
                block_starts = torch.arange(n_trunks) * n_queries
                q_indices = block_starts[:, None] + torch.arange(n_queries)
                k_indices = (
                    block_starts[:, None]
                    - (n_keys - n_queries) // 2
                    + torch.arange(n_keys)
                )
                expected = (q_indices < n).unsqueeze(-1) & (
                    (k_indices >= 0) & (k_indices < n)
                ).unsqueeze(-2)

                self.assertEqual(mask.dtype, torch.bool)
                self.assertEqual(mask.shape, (n_trunks, n_queries, n_keys))
                self.assertTrue(torch.equal(mask, expected))

    def test_dense_attention_bias_is_gathered_into_local_blocks(self):
        batch_size_dims = (2, 3)
        n = 73
        d = 5
        n_queries = 16
        n_keys = 32
        inf = 1e10

        q, k, v = create_qkv(batch_size_dims, n, n, d)
        attn_bias = torch.randn(*batch_size_dims, n, n)
        _, _, _, attn_bias_trunked, _ = rearrange_to_dense_trunk(
            q,
            k,
            v,
            n_queries,
            n_keys,
            attn_bias=attn_bias,
            inf=inf,
        )

        n_trunks = math.ceil(n / n_queries)
        block_starts = torch.arange(n_trunks) * n_queries
        q_indices = block_starts[:, None] + torch.arange(n_queries)
        k_indices = (
            block_starts[:, None]
            - (n_keys - n_queries) // 2
            + torch.arange(n_keys)
        )
        valid = (q_indices < n).unsqueeze(-1) & (
            (k_indices >= 0) & (k_indices < n)
        ).unsqueeze(-2)
        expected = attn_bias[
            ...,
            q_indices.clamp(max=n - 1).unsqueeze(-1),
            k_indices.clamp(min=0, max=n - 1).unsqueeze(-2),
        ]
        expected = expected.masked_fill(~valid, -inf)

        self.assertEqual(
            attn_bias_trunked.shape,
            (*batch_size_dims, n_trunks, n_queries, n_keys),
        )
        self.assertTrue(torch.equal(attn_bias_trunked, expected))

    def test_padding_mask_storage_scales_with_local_windows(self):
        n = 4097
        n_queries = 32
        n_keys = 128
        q, k, _ = create_qkv((), n, n, 1)
        _, _, padding_info = rearrange_qk_to_dense_trunk(
            q,
            k,
            dim_q=-2,
            dim_k=-2,
            n_queries=n_queries,
            n_keys=n_keys,
        )

        mask = padding_info["mask_trunked"]
        self.assertEqual(
            mask.numel(), math.ceil(n / n_queries) * n_queries * n_keys
        )


if __name__ == "__main__":
    unittest.main()
