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

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.distillation.audit_foldseek_clusters import (
    FoldseekAuditError,
    audit_foldseek_clusters,
    foldseek_id_from_manifest_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDITOR = REPO_ROOT / "scripts" / "distillation" / "audit_foldseek_clusters.py"


@unittest.skipUnless(
    Path("/dev/shm").is_dir() and shutil.which("sort"),
    "/dev/shm and GNU sort are required",
)
class FoldseekClusterAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(
            tempfile.mkdtemp(prefix="foldseek-audit-test-", dir="/dev/shm")
        )
        self.addCleanup(shutil.rmtree, self.temp_dir, True)
        self.manifest = self.temp_dir / "paths.tsv"
        self.clusters = self.temp_dir / "clusters.tsv"
        self.report = self.temp_dir / "audit.json"
        self.scratch = self.temp_dir / "scratch"
        self.scratch.mkdir()

    def run_cli(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(AUDITOR),
                "--manifest-paths-tsv",
                str(self.manifest),
                "--clusters-tsv",
                str(self.clusters),
                "--report-json",
                str(self.report),
                "--tmp-dir",
                str(self.scratch),
                "--sort-parallel",
                "2",
                "--sort-memory",
                "1M",
                "--progress-every",
                "0",
                "--min-free-gb",
                "0",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_passes_unsorted_complete_partition(self) -> None:
        self.manifest.write_text(
            "\n".join(
                [
                    "0/bucket/alpha.cif",
                    "1/beta.mmcif.gz",
                    "2/gamma.pdb",
                    "2/delta.cif.gz",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        self.clusters.write_text(
            "\n".join(
                [
                    "gamma\tgamma",
                    "alpha\tbeta",
                    "delta\tdelta",
                    "alpha\talpha",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        completed = self.run_cli()

        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(self.report.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(
            report["counts"],
            {
                "cluster_count": 3,
                "cluster_rows": 4,
                "manifest_rows": 4,
                "manifest_unique_ids": 4,
                "maximum_cluster_size": 2,
                "member_unique_ids": 4,
                "representatives_with_self_edge": 3,
                "self_edge_rows": 3,
            },
        )
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(report["inputs"]["manifest_paths_tsv_sha256"])
        self.assertTrue(report["inputs"]["clusters_tsv_sha256"])
        self.assertEqual(list(self.scratch.iterdir()), [])
        self.assertEqual(list(self.temp_dir.glob(".audit.json.tmp.*")), [])

    def test_reports_all_partition_violations_and_returns_one(self) -> None:
        self.manifest.write_text(
            "0/a.cif\n0/b.cif\n0/c.cif\n0/d.cif\n",
            encoding="utf-8",
        )
        self.clusters.write_text(
            "a\ta\na\tb\nghost\tc\nghost\tc\nz\tz\n",
            encoding="utf-8",
        )

        completed = self.run_cli()

        self.assertEqual(completed.returncode, 1, completed.stderr)
        report = json.loads(self.report.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["counts"]["cluster_rows"], 5)
        self.assertEqual(report["counts"]["cluster_count"], 3)
        self.assertEqual(report["counts"]["maximum_cluster_size"], 2)
        self.assertEqual(report["violations"]["duplicate_member_ids"], 1)
        self.assertEqual(report["violations"]["duplicate_member_excess_rows"], 1)
        self.assertEqual(report["violations"]["missing_members"], 1)
        self.assertEqual(report["samples"]["missing_members"], ["d"])
        self.assertEqual(report["violations"]["extra_members"], 1)
        self.assertEqual(report["samples"]["extra_members"], ["z"])
        self.assertEqual(report["violations"]["representatives_not_in_manifest"], 2)
        self.assertEqual(
            report["samples"]["representatives_not_in_manifest"],
            ["ghost", "z"],
        )
        self.assertEqual(
            report["violations"]["representatives_not_observed_as_members"],
            1,
        )
        self.assertEqual(
            report["violations"]["representatives_without_self_edge"],
            1,
        )
        self.assertEqual(
            report["samples"]["representatives_without_self_edge"],
            ["ghost"],
        )
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_duplicate_manifest_ids_are_distinct_from_duplicate_members(self) -> None:
        self.manifest.write_text(
            "0/a.cif\n1/nested/a.pdb\n0/b.cif\n",
            encoding="utf-8",
        )
        self.clusters.write_text("a\ta\nb\tb\n", encoding="utf-8")

        report = audit_foldseek_clusters(
            manifest_paths_tsv=self.manifest,
            clusters_tsv=self.clusters,
            report_json=self.report,
            tmp_dir=self.scratch,
            sort_parallel=1,
            sort_memory="1M",
            progress_every=0,
            min_free_gb=0,
        )

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["violations"]["manifest_duplicate_ids"], 1)
        self.assertEqual(report["violations"]["manifest_duplicate_excess_rows"], 1)
        self.assertEqual(report["violations"]["duplicate_member_ids"], 0)
        self.assertFalse(report["checks"]["row_count_matches_manifest"])

    def test_malformed_cluster_publishes_atomic_error_report(self) -> None:
        self.manifest.write_text("0/a.cif\n", encoding="utf-8")
        self.clusters.write_text("a\ta\textra\n", encoding="utf-8")

        completed = self.run_cli()

        self.assertEqual(completed.returncode, 2)
        report = json.loads(self.report.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "error")
        self.assertIn("exactly two", report["error"]["message"])
        self.assertEqual(list(self.scratch.iterdir()), [])
        self.assertEqual(list(self.temp_dir.glob(".audit.json.tmp.*")), [])

    def test_rejects_root_filesystem_scratch_without_creating_it(self) -> None:
        self.manifest.write_text("0/a.cif\n", encoding="utf-8")
        self.clusters.write_text("a\ta\n", encoding="utf-8")
        unsafe = Path("/tmp/foldseek-audit-unsafe-must-not-create")

        with self.assertRaisesRegex(FoldseekAuditError, "below /mnt or /dev/shm"):
            audit_foldseek_clusters(
                manifest_paths_tsv=self.manifest,
                clusters_tsv=self.clusters,
                report_json=self.report,
                tmp_dir=unsafe.parent,
                sort_parallel=1,
                sort_memory="1M",
                progress_every=0,
                min_free_gb=0,
            )

        self.assertFalse(unsafe.exists())
        error_report = json.loads(self.report.read_text(encoding="utf-8"))
        self.assertEqual(error_report["status"], "error")

    def test_foldseek_identifier_suffix_contract(self) -> None:
        self.assertEqual(
            foldseek_id_from_manifest_path("0/x/model.CIF.GZ"),
            "model",
        )
        self.assertEqual(
            foldseek_id_from_manifest_path("1/x/model.mmcif"),
            "model",
        )
        with self.assertRaisesRegex(FoldseekAuditError, "does not end"):
            foldseek_id_from_manifest_path("0/x/model.txt")


if __name__ == "__main__":
    unittest.main()
