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

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from protenix.metrics.lddt_metrics import LDDTMetrics


class _MetricsConfig(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def _configs():
    return SimpleNamespace(
        metrics=_MetricsConfig(
            lddt=SimpleNamespace(eps=1e-10),
            complex_ranker_keys=["gpde"],
        ),
        infer_setting=SimpleNamespace(lddt_metrics_chunk_size=None),
    )


def _summary_confidence(gpde):
    return [{"gpde": torch.tensor(value)} for value in gpde]


class TestAF3ProxyLDDTMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = LDDTMetrics(_configs())

    def test_aggregate_uses_global_gpde_top1_plus_per_entry_best_of_five(self):
        # Lower gPDE is better, so sample index 2 is the single global top-1
        # sample shared by every chain/interface entry.
        lddt_dict = {
            "complex": torch.tensor([0.1, 0.2, 0.9, 0.4, 0.5]),
            "af3_entries": torch.tensor(
                [
                    [0.1, 0.2, 0.9, 0.4, 0.5],
                    [0.6, 0.8, 0.7, 0.1, 0.2],
                    [0.2, 0.3, 0.4, 0.5, 0.6],
                ]
            ),
            "af3_eval_type": np.array(
                ["intra_prot", "intra_prot", "prot_prot"]
            ),
        }

        aggregated, _ = self.metrics.aggregate_lddt(
            lddt_dict,
            _summary_confidence([4.0, 3.0, 1.0, 2.0, 5.0]),
        )

        # intra_prot:
        #   entry 1: (gPDE top-1 0.9 + best 0.9) / 2 = 0.90
        #   entry 2: (gPDE top-1 0.7 + best 0.8) / 2 = 0.75
        # mean across entries = 0.825, reported on the AF3 0--100 scale.
        self.assertAlmostEqual(
            aggregated["lddt/af3_proxy/intra_prot/top1_best5_pct"].item(),
            82.5,
            places=5,
        )
        self.assertAlmostEqual(
            aggregated["lddt/af3_proxy/prot_prot/top1_best5_pct"].item(),
            50.0,
            places=5,
        )

    def test_protein_nucleic_interface_uses_nucleic_30_angstrom_radius(self):
        # This 20 A protein-DNA pair is absent in the protein-row (15 A) mask
        # but present in the nucleotide-row (30 A) mask. dna_prot must therefore
        # use the transposed/orientation-correct mask and retain the pair.
        true_coordinate = torch.tensor(
            [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]
        )
        pred_coordinate = true_coordinate.unsqueeze(0)
        label_dict = {
            "coordinate": true_coordinate,
            "lddt_mask": torch.tensor([[False, False], [True, False]]),
            "eval_type": np.array(["dna_prot"]),
            "chain_1_mask": torch.tensor([[True, False]]),
            "chain_2_mask": torch.tensor([[False, True]]),
        }

        out = self.metrics.compute_lddt(
            {"coordinate": pred_coordinate},
            label_dict,
        )

        self.assertEqual(tuple(out["af3_entries"].shape), (1, 1))
        self.assertEqual(list(out["af3_eval_type"]), ["dna_prot"])
        torch.testing.assert_close(out["af3_entries"], torch.ones(1, 1))

    def test_empty_chain_or_interface_mask_is_skipped(self):
        true_coordinate = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )
        label_dict = {
            "coordinate": true_coordinate,
            "lddt_mask": torch.tensor([[False, True], [True, False]]),
            "eval_type": np.array(["intra_prot", "prot_prot"]),
            "chain_1_mask": torch.tensor(
                [[True, True], [True, False]]
            ),
            "chain_2_mask": torch.tensor(
                [[True, True], [False, False]]
            ),
        }

        out = self.metrics.compute_lddt(
            {"coordinate": true_coordinate.unsqueeze(0)},
            label_dict,
        )

        self.assertEqual(tuple(out["af3_entries"].shape), (1, 1))
        self.assertEqual(list(out["af3_eval_type"]), ["intra_prot"])
        torch.testing.assert_close(out["af3_entries"], torch.ones(1, 1))

    def test_labels_without_chain_interface_fields_preserve_complex_only_behavior(self):
        true_coordinate = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )
        out = self.metrics.compute_lddt(
            {"coordinate": true_coordinate.unsqueeze(0).repeat(5, 1, 1)},
            {
                "coordinate": true_coordinate,
                "lddt_mask": torch.tensor([[False, True], [True, False]]),
            },
        )

        self.assertEqual(set(out), {"complex"})
        torch.testing.assert_close(out["complex"], torch.ones(5))

        aggregated, details = self.metrics.aggregate_lddt(
            out,
            _summary_confidence([4.0, 3.0, 1.0, 2.0, 5.0]),
        )
        self.assertEqual(details, {})
        self.assertFalse(any(key.startswith("lddt/af3_proxy/") for key in aggregated))
        self.assertIn("lddt/complex/gpde.rank1", aggregated)


if __name__ == "__main__":
    unittest.main()
