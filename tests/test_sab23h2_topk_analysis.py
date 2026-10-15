import unittest

from analyze_sab23h2_topk import topk_best_dockq, topk_success


class TopKAnalysisTest(unittest.TestCase):
    def test_topk_success_uses_ranking_order(self):
        rows = [
            {"ranking_score": 0.9, "immune_other_dockq": 0.1},
            {"ranking_score": 0.8, "immune_other_dockq": 0.2},
            {"ranking_score": 0.7, "immune_other_dockq": 0.5},
        ]
        self.assertFalse(topk_success(rows, k=2, threshold=0.23))
        self.assertTrue(topk_success(rows, k=3, threshold=0.23))

    def test_topk_best_dockq_returns_best_within_topk(self):
        rows = [
            {"ranking_score": 0.9, "immune_other_dockq": 0.1},
            {"ranking_score": 0.8, "immune_other_dockq": 0.4},
            {"ranking_score": 0.7, "immune_other_dockq": 0.8},
        ]
        self.assertAlmostEqual(topk_best_dockq(rows, k=2), 0.4)
        self.assertAlmostEqual(topk_best_dockq(rows, k=3), 0.8)


if __name__ == "__main__":
    unittest.main()
