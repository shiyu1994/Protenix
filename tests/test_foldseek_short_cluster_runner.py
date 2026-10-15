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

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "distillation" / "run_foldseek_short_cluster.sh"

FAKE_FOLDSEEK = """#!/usr/bin/env bash
set -euo pipefail
printf '%s|%s\\n' "$PWD" "$*" >>"$FAKE_FOLDSEEK_LOG"
command_name="$1"
shift
case "$command_name" in
  version)
    printf 'foldseek fake-stable-1\\n'
    ;;
  createdb)
    output_prefix="$2"
    printf 'db\\n' >"$output_prefix"
    printf '0\\n' >"${output_prefix}.dbtype"
    ;;
  cluster)
    output_prefix="$2"
    tmp_dir="$3"
    printf 'clu\\n' >"$output_prefix"
    printf '6\\n' >"${output_prefix}.dbtype"
    printf 'retained\\n' >"$tmp_dir/fake.tmp"
    ;;
  createtsv)
    output_tsv="$4"
    mkdir -p "$(dirname "$output_tsv")"
    printf 'rep\\tmember\\n' >"$output_tsv"
    ;;
  *)
    printf 'unexpected command: %s\\n' "$command_name" >&2
    exit 9
    ;;
esac
"""


@unittest.skipUnless(Path("/dev/shm").is_dir(), "/dev/shm is required by runner safety tests")
class FoldseekShortClusterRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="foldseek-runner-test-", dir="/dev/shm"))
        self.addCleanup(shutil.rmtree, self.temp_dir, True)
        self.bundle = self.temp_dir / "input"
        self.roots = self.bundle / "roots"
        self.source = self.temp_dir / "source"
        self.roots.mkdir(parents=True)
        self.source.mkdir()
        (self.roots / "afdb").symlink_to(self.source, target_is_directory=True)
        self.paths_tsv = self.bundle / "paths.tsv"
        self.paths_tsv.write_text(
            "afdb/a.cif\nafdb/b.cif\nafdb/c.cif\n",
            encoding="utf-8",
        )
        self.fake_foldseek = self.temp_dir / "foldseek"
        self.fake_foldseek.write_text(FAKE_FOLDSEEK, encoding="utf-8")
        self.fake_foldseek.chmod(0o755)
        self.fake_log = self.temp_dir / "foldseek.calls"
        self.output = self.temp_dir / "persistent"
        self.work = self.temp_dir / "work"

    def run_runner(
        self,
        *extra_args: str,
        output: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            "bash",
            str(RUNNER),
            "--paths-tsv",
            str(self.paths_tsv),
            "--roots-dir",
            str(self.roots),
            "--output-dir",
            str(output or self.output),
            "--foldseek-bin",
            str(self.fake_foldseek),
            "--work-dir",
            str(self.work),
            "--min-free-gb",
            "0",
            *extra_args,
        ]
        environment = os.environ.copy()
        environment["FAKE_FOLDSEEK_LOG"] = str(self.fake_log)
        return subprocess.run(
            command,
            check=check,
            capture_output=True,
            text=True,
            env=environment,
        )

    def calls(self, command_name: str) -> list[str]:
        if not self.fake_log.exists():
            return []
        return [
            line
            for line in self.fake_log.read_text(encoding="utf-8").splitlines()
            if f"|{command_name} " in line or line.endswith(f"|{command_name}")
        ]

    def test_full_run_uses_fixed_fast_profile_and_persists_artifacts(self) -> None:
        self.run_runner("--mode", "full", "--threads", "128")

        createdb_call = self.calls("createdb")
        cluster_call = self.calls("cluster")
        createtsv_call = self.calls("createtsv")
        self.assertEqual(len(createdb_call), 1)
        self.assertEqual(len(cluster_call), 1)
        self.assertEqual(len(createtsv_call), 1)
        self.assertTrue(createdb_call[0].startswith(f"{self.roots.resolve()}|"))

        cluster_arguments = cluster_call[0].split("|", maxsplit=1)[1]
        for expected in (
            "--threads 128",
            "--alignment-type 2",
            "--tmscore-threshold 0.8",
            "--tmscore-threshold-mode 2",
            "--exact-tmscore 1",
            "-c 0.9",
            "--cov-mode 0",
            "--cluster-mode 0",
            "--cluster-reassign 1",
            "--remove-tmp-files 0",
        ):
            self.assertIn(expected, cluster_arguments)

        self.assertTrue((self.output / "db" / "input" / "short_le200.dbtype").is_file())
        self.assertTrue((self.output / "db" / "cluster" / "short_le200.dbtype").is_file())
        self.assertEqual(
            (self.output / "result" / "short_le200_clusters.tsv").read_text(encoding="utf-8"),
            "rep\tmember\n",
        )
        self.assertTrue((self.output / "provenance" / "run-config.env").is_file())
        self.assertTrue((self.output / "provenance" / "completion.env").is_file())
        self.assertTrue((self.output / "logs" / "cluster.log").is_file())
        self.assertTrue((self.work / "tmp" / "fake.tmp").is_file())

    def test_resume_skips_all_completed_stages(self) -> None:
        self.run_runner("--mode", "full")
        shutil.rmtree(self.work)
        self.run_runner("--mode", "full")

        self.assertEqual(len(self.calls("createdb")), 1)
        self.assertEqual(len(self.calls("cluster")), 1)
        self.assertEqual(len(self.calls("createtsv")), 1)
        self.assertTrue((self.work / "db" / "input" / "short_le200.dbtype").is_file())
        self.assertTrue((self.work / "db" / "cluster" / "short_le200.dbtype").is_file())

    def test_pilot_materializes_deterministic_prefix_manifest(self) -> None:
        self.run_runner("--mode", "pilot", "--pilot-size", "2")

        pilot_manifest = self.output / "provenance" / "pilot.paths.tsv"
        self.assertEqual(
            pilot_manifest.read_text(encoding="utf-8"),
            "afdb/a.cif\nafdb/b.cif\n",
        )
        createdb_arguments = self.calls("createdb")[0].split("|", maxsplit=1)[1]
        self.assertIn(str(pilot_manifest), createdb_arguments)

    def test_refuses_root_filesystem_and_tmp_output(self) -> None:
        result = self.run_runner(
            "--mode",
            "full",
            output=Path("/tmp/foldseek-runner-must-not-create"),
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be below /mnt or /dev/shm", result.stderr)


if __name__ == "__main__":
    unittest.main()
