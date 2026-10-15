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

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.runtime.summarize_v9_train_metrics import (
    MetricsSummaryError,
    load_summary,
    main,
    render_markdown,
)


def _runtime_window(*, wait_max: float, rank: int) -> dict:
    return {
        "end_to_end_elapsed_seconds": 93.5,
        "end_to_end_samples_per_sec": 5.475,
        "timings": {
            "data_wait_seconds": {
                "p50": 0.2,
                "p95": 40.0,
                "p99": wait_max,
                "max": wait_max,
                "max_rank": rank,
            },
            "data_process_seconds": {"p95": 35.0, "max": 80.0},
            "loss_health_sync_seconds": {"p95": 0.8, "max": 1.2},
            "step_wall_seconds": {"p95": 7.0, "p99": 8.0, "max": 8.1},
        },
        "grad_health": {
            "grad_clip_count": 3,
            "loss_nan_count": 0,
            "nonfinite_grad_count": 0,
            "optimizer_skipped_count": 0,
        },
        "top_outliers": {
            "data_wait_seconds": [
                {
                    "data_wait_seconds": wait_max,
                    "rank": rank,
                    "step": 399,
                    "global_step": 99,
                    "pdb_id": "6w2t",
                    "data_id": "row-7",
                    "dataset_name": "weightedPDB",
                    "n_token": 361,
                    "n_atom": 4641,
                    "N_msa": 16384,
                    "N_msa_shuffle_group": 3,
                    "msa_shuffle_requested_variant_code": 2,
                    "msa_shuffle_variant_eligible": True,
                    "msa_shuffle_variant_applied": True,
                    "msa_shuffle_whole_physical_chain_applied": True,
                    "N_msa_crop_protein_chain": 3,
                    "N_msa_query_only_protein_chain": 2,
                    "N_msa_retained_protein_chain": 1,
                    "msa_partial_query_only_applied": True,
                    "msa_global_query_only_applied": False,
                }
            ],
            "step_wall_seconds": [
                {
                    "step_wall_seconds": 8.1,
                    "rank": rank,
                    "step": 399,
                    "global_step": 99,
                    "pdb_id": "6w2t",
                    "data_id": "row-7",
                    "dataset_name": "weightedPDB",
                    "n_token": 361,
                    "n_atom": 4641,
                }
            ],
            "msa_featurizer_seconds": [
                {
                    "msa_featurizer_seconds": 22.0,
                    "rank": 4,
                    "step": 398,
                    "global_step": 99,
                    "pdb_id": "7abc",
                    "dataset_name": "weightedPDB",
                }
            ],
        },
    }


class TestV9TrainMetricsSummary(unittest.TestCase):
    def _write_metrics(self, root: Path) -> Path:
        path = root / "train_metrics.jsonl"
        records = [
            {
                "event": "train_metrics",
                "global_step": 99,
                "runtime_window": _runtime_window(wait_max=88.0, rank=123),
            },
            {
                "event": "train_metrics",
                "global_step": 199,
                "runtime_window": _runtime_window(wait_max=70.0, rank=17),
            },
        ]
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        return path

    def test_load_summary_extracts_windows_and_merges_outlier_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = load_summary(self._write_metrics(Path(tmpdir)))

        self.assertEqual(summary["window_count"], 2)
        window = summary["windows"][0]
        self.assertEqual(window["global_step"], 99)
        self.assertEqual(window["end_to_end_elapsed_seconds"], 93.5)
        self.assertEqual(
            window["timings"]["data_wait_seconds"],
            {
                "p50": 0.2,
                "p95": 40.0,
                "p99": 88.0,
                "max": 88.0,
                "max_rank": 123,
            },
        )
        self.assertEqual(window["health"]["grad_clip_count"], 3)
        self.assertEqual(len(window["top_outliers"]), 2)
        merged = window["top_outliers"][0]
        self.assertEqual(merged["pdb_id"], "6w2t")
        self.assertEqual(merged["N_msa_shuffle_group"], 3)
        self.assertEqual(
            merged["msa_shuffle_requested_variant_code"],
            2,
        )
        self.assertIs(merged["msa_shuffle_variant_eligible"], True)
        self.assertIs(merged["msa_shuffle_variant_applied"], True)
        self.assertIs(
            merged["msa_shuffle_whole_physical_chain_applied"],
            True,
        )
        self.assertEqual(merged["N_msa_crop_protein_chain"], 3)
        self.assertEqual(merged["N_msa_query_only_protein_chain"], 2)
        self.assertEqual(merged["N_msa_retained_protein_chain"], 1)
        self.assertIs(merged["msa_partial_query_only_applied"], True)
        self.assertIs(merged["msa_global_query_only_applied"], False)
        self.assertEqual(
            merged["metrics"],
            {"data_wait_seconds": 88.0, "step_wall_seconds": 8.1},
        )

    def test_markdown_and_json_cli_outputs_are_compact_and_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = self._write_metrics(Path(tmpdir))
            summary = load_summary(metrics_path)
            markdown = render_markdown(summary)

            self.assertIn("| 99 | 93.500 | 5.475 |", markdown)
            self.assertIn("88.000 (r123)", markdown)
            self.assertEqual(markdown.count("| 99 | 399 | 123 |"), 1)
            self.assertIn("data_wait_seconds=88.000", markdown)
            self.assertIn("step_wall_seconds=8.100", markdown)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                return_code = main([str(metrics_path), "--format", "json"])
            payload = json.loads(stdout.getvalue())
            self.assertEqual(return_code, 0)
            self.assertEqual(payload["window_count"], 2)
            self.assertEqual(payload["windows"][1]["global_step"], 199)

    def test_missing_and_malformed_metrics_fail_with_line_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(
                MetricsSummaryError,
                "metrics file does not exist",
            ):
                load_summary(root / "missing.jsonl")

            malformed = root / "malformed.jsonl"
            malformed.write_text('{"event": "train_metrics", nope}\\n', encoding="utf-8")
            with self.assertRaisesRegex(
                MetricsSummaryError,
                r"malformed[.]jsonl:line 1 contains malformed JSON",
            ):
                load_summary(malformed)

            missing = root / "missing-runtime.jsonl"
            missing.write_text(
                json.dumps({"event": "train_metrics", "global_step": 4}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                MetricsSummaryError,
                r"line 1[.]runtime_window must be a JSON object",
            ):
                load_summary(missing)

    def test_missing_required_timing_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train_metrics.jsonl"
            runtime = _runtime_window(wait_max=88.0, rank=123)
            del runtime["timings"]["loss_health_sync_seconds"]["max"]
            path.write_text(
                json.dumps(
                    {
                        "event": "train_metrics",
                        "global_step": 99,
                        "runtime_window": runtime,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                MetricsSummaryError,
                "missing required field 'max'",
            ):
                load_summary(path)


if __name__ == "__main__":
    unittest.main()
