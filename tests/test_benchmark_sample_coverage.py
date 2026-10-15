import unittest

from runner.train import (
    build_benchmark_msa_depth_coverage,
    build_benchmark_sample_coverage,
    build_benchmark_token_composition_coverage,
)


class TestBenchmarkSampleCoverage(unittest.TestCase):
    @staticmethod
    def _record(rank: int, step: int, pdb_index: int):
        return {
            "rank": rank,
            "step": step,
            "identity": {
                "dataset_name": "benchmark_long_protein_context_1024_800",
                "pdb_id": f"p{pdb_index:04d}",
                "data_id": f"sample_{pdb_index:04d}",
            },
            "identity_complete": True,
        }

    def test_reports_exact_800_sample_global_coverage(self):
        gathered = [
            [self._record(rank, step, rank + step * 8) for step in range(100)]
            for rank in range(8)
        ]
        coverage = build_benchmark_sample_coverage(gathered)

        self.assertEqual(coverage["sample_count"], 800)
        self.assertEqual(coverage["unique_sample_count"], 800)
        self.assertEqual(coverage["duplicate_sample_count"], 0)
        self.assertEqual(coverage["missing_identity_count"], 0)
        self.assertEqual(coverage["per_rank_sample_count"], [100] * 8)
        self.assertEqual(len(coverage["ordered_sha256"]), 64)

    def test_detects_cross_rank_duplicate_and_missing_identity(self):
        gathered = [
            [self._record(0, 0, 0)],
            [self._record(1, 0, 0)],
        ]
        gathered[1][0]["identity_complete"] = False
        coverage = build_benchmark_sample_coverage(gathered)

        self.assertEqual(coverage["sample_count"], 2)
        self.assertEqual(coverage["unique_sample_count"], 1)
        self.assertEqual(coverage["duplicate_sample_count"], 1)
        self.assertEqual(coverage["missing_identity_count"], 1)


class TestBenchmarkMsaDepthCoverage(unittest.TestCase):
    def test_accepts_multiple_recycles_at_exact_target_depth(self):
        gathered = [
            [
                {
                    "rank": rank,
                    "step": step,
                    "calls": [
                        {"input_depth": 4096, "sampled_depth": 2048},
                        {"input_depth": 4096, "sampled_depth": 2048},
                    ],
                }
                for step in range(2)
            ]
            for rank in range(2)
        ]

        coverage = build_benchmark_msa_depth_coverage(
            gathered, target_depth=2048
        )

        self.assertEqual(coverage["step_record_count"], 4)
        self.assertEqual(coverage["call_count"], 8)
        self.assertEqual(coverage["missing_trace_step_count"], 0)
        self.assertEqual(coverage["sampled_depth_min"], 2048)
        self.assertEqual(coverage["sampled_depth_max"], 2048)
        self.assertEqual(coverage["sampled_target_call_count"], 8)
        self.assertEqual(coverage["below_target_call_count"], 0)
        self.assertEqual(coverage["per_rank_step_count"], [2, 2])
        self.assertEqual(coverage["per_rank_call_count"], [4, 4])

    def test_detects_shallow_and_missing_msa_trace(self):
        gathered = [
            [
                {
                    "rank": 0,
                    "step": 0,
                    "calls": [{"input_depth": 1024, "sampled_depth": 1024}],
                },
                {"rank": 0, "step": 1, "calls": []},
            ]
        ]

        coverage = build_benchmark_msa_depth_coverage(
            gathered, target_depth=2048
        )

        self.assertEqual(coverage["missing_trace_step_count"], 1)
        self.assertEqual(coverage["sampled_depth_min"], 1024)
        self.assertEqual(coverage["below_target_call_count"], 1)


class TestBenchmarkTokenCompositionCoverage(unittest.TestCase):
    @staticmethod
    def _record(rank: int, step: int, *, nuc_tokens: int = 128):
        return {
            "rank": rank,
            "step": step,
            "identity": {
                "dataset_name": "benchmark_long_protein_context_960_800_msa2048",
                "pdb_id": f"p{rank}{step}",
            },
            "token_composition": {
                "n_token": 960,
                "N_prot_token": 960 - nuc_tokens,
                "N_lig_token": 0,
                "N_rna_token": nuc_tokens,
                "N_dna_token": 0,
            },
        }

    def test_records_complete_composition_at_the_nucleic_limit(self):
        gathered = [[self._record(rank, step) for step in range(2)] for rank in range(2)]

        coverage = build_benchmark_token_composition_coverage(
            gathered, max_nuc_tokens=128
        )

        self.assertEqual(coverage["sample_count"], 4)
        self.assertEqual(coverage["complete_composition_count"], 4)
        self.assertEqual(coverage["missing_composition_count"], 0)
        self.assertEqual(coverage["nuc_token_count"]["max"], 128)
        self.assertEqual(coverage["over_nuc_limit_count"], 0)
        self.assertEqual(coverage["token_type_sum_mismatch_count"], 0)
        self.assertEqual(len(coverage["sample_records"]), 4)

    def test_detects_over_limit_missing_and_partition_mismatch(self):
        over_limit = self._record(0, 0, nuc_tokens=129)
        missing = self._record(0, 1)
        missing["token_composition"].pop("N_dna_token")
        mismatch = self._record(0, 2)
        mismatch["token_composition"]["N_prot_token"] -= 1

        coverage = build_benchmark_token_composition_coverage(
            [[over_limit, missing, mismatch]], max_nuc_tokens=128
        )

        self.assertEqual(coverage["sample_count"], 3)
        self.assertEqual(coverage["complete_composition_count"], 2)
        self.assertEqual(coverage["missing_composition_count"], 1)
        self.assertEqual(coverage["over_nuc_limit_count"], 1)
        self.assertEqual(coverage["token_type_sum_mismatch_count"], 1)


if __name__ == "__main__":
    unittest.main()
