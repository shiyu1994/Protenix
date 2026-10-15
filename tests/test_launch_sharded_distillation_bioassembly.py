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

import json
import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from scripts import launch_sharded_distillation_bioassembly as launcher


def _write_plan(path: Path, world_size: int = 2) -> None:
    path.mkdir(parents=True)
    (path / "plan.json").write_text(
        json.dumps(
            {
                "version": 3,
                "plan_id": "test-plan",
                "world_size": world_size,
            }
        ),
        encoding="utf-8",
    )


def _write_remote_fixture(
    root: Path,
    *,
    builder_source: str,
) -> tuple[Path, Path, Path]:
    work_dir = root / "work"
    work_dir.mkdir()
    builder = work_dir / "fake_builder.py"
    builder.write_text(builder_source, encoding="utf-8")
    plan_dir = root / "plan"
    _write_plan(plan_dir, world_size=1)
    (plan_dir / "rank-000.jsonl").write_text("{}\n", encoding="utf-8")
    common_dir = root / "common"
    common_dir.mkdir()
    (common_dir / "components.cif").write_text("data\n", encoding="utf-8")
    (common_dir / "components.cif.rdkit_mol.pkl").write_text(
        "cache\n",
        encoding="utf-8",
    )
    return work_dir, builder, plan_dir


def _terminate_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.01)


class HostsTest(unittest.TestCase):
    def test_comments_are_removed_and_file_order_is_rank_order(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "hosts.txt"
            path.write_text(
                "\n# workers\nworker-a # rank 0\nuser@worker-b\n",
                encoding="utf-8",
            )

            self.assertEqual(
                launcher.read_hosts(path),
                ["worker-a", "user@worker-b"],
            )

    def test_duplicate_and_shell_syntax_are_rejected(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "hosts.txt"
            path.write_text("worker-a\nworker-a\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicates"):
                launcher.read_hosts(path)

            path.write_text(
                "worker-a;touch-pwned\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                launcher.read_hosts(path)


class RemoteScriptTest(unittest.TestCase):
    def _arguments(self, root: Path):
        return {
            "rank": 7,
            "plan_dir": root / "plan with spaces",
            "builder": root / "builder's script.py",
            "work_dir": root / "work $(not-run)",
            "protenix_root_dir": root / "data root",
            "python_executable": "/shared/env/bin/python",
            "workers": 128,
            "compresslevel": 3,
            "log_path": root / "logs" / "rank-007.log",
            "pid_path": root / "logs" / "rank-007.pid",
            "resume": False,
        }

    def test_script_has_valid_shell_syntax_and_locked_background_lease(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            script = launcher.build_remote_script(**self._arguments(root))

            completed = subprocess.run(
                ["bash", "-n"],
                input=script,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("command -v flock", script)
        self.assertIn('exec 9>"$lock_file"', script)
        self.assertIn("flock -n 9", script)
        self.assertIn("9>&9 &", script)
        self.assertIn("--workers 128", script)
        self.assertIn("--compresslevel 3", script)
        self.assertIn("--tail-max-lanes-per-leaf 0", script)
        self.assertIn("umask 0022", script)
        self.assertIn("OMP_NUM_THREADS=1", script)
        self.assertIn("OPENBLAS_NUM_THREADS=1", script)
        self.assertIn("PROTENIX_ROOT_DIR=", script)

    def test_fast_path_flags_are_fail_closed(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            arguments = self._arguments(root)
            auto = launcher.build_remote_script(
                **arguments,
                fast_path_mode="auto",
            )
            required = launcher.build_remote_script(
                **arguments,
                fast_path_mode="require",
            )
            disabled = launcher.build_remote_script(
                **arguments,
                fast_path_mode="off",
            )

        self.assertNotIn("--require-fast-path", auto)
        self.assertNotIn("--disable-fast-path", auto)
        self.assertIn("--require-fast-path", required)
        self.assertNotIn("--disable-fast-path", required)
        self.assertIn("--disable-fast-path", disabled)
        self.assertNotIn("--require-fast-path", disabled)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            launcher.build_remote_script(
                **arguments,
                fast_path_mode="invalid",
            )

    def test_ssh_command_keeps_remote_script_one_argument(self):
        script = "echo '$HOME'; touch 'path with spaces'\n"

        command = launcher.build_ssh_command(
            "user@worker-a",
            script,
            ssh_options=["ConnectTimeout=10"],
        )

        self.assertEqual(command[0], "ssh")
        self.assertIn("BatchMode=yes", command)
        self.assertIn("ConnectTimeout=10", command)
        self.assertIn("ServerAliveInterval=30", command)
        self.assertIn("ServerAliveCountMax=3", command)
        self.assertEqual(command[-2], "user@worker-a")
        self.assertTrue(command[-1].startswith("bash -lc "))

    def test_background_process_holds_rank_lease(self):
        with TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            work_dir, builder, plan_dir = _write_remote_fixture(
                root,
                builder_source="import time\ntime.sleep(30)\n",
            )
            log_dir = root / "logs $(touch injected)"
            log_path = log_dir / "rank-000.log"
            pid_path = log_dir / "rank-000.pid"
            script = launcher.build_remote_script(
                rank=0,
                plan_dir=plan_dir,
                builder=builder,
                work_dir=work_dir,
                protenix_root_dir=root,
                python_executable=sys.executable,
                workers=1,
                compresslevel=3,
                log_path=log_path,
                pid_path=pid_path,
                resume=False,
            )

            first = subprocess.run(
                ["bash", "-lc", script],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            try:
                self.assertEqual(first.returncode, 0, first.stderr)
                os.kill(pid, 0)
                second = subprocess.run(
                    ["bash", "-lc", script],
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.assertEqual(second.returncode, 5)
                self.assertIn("lease is already held", second.stderr)
                self.assertFalse((work_dir / "injected").exists())
                self.assertFalse((root / "injected").exists())
            finally:
                _terminate_process(pid)

    def test_remote_preflight_requires_rank_plan_and_ccd_cache(self):
        with TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            work_dir, builder, plan_dir = _write_remote_fixture(
                root,
                builder_source="import time\ntime.sleep(30)\n",
            )
            rank_plan = plan_dir / "rank-000.jsonl"
            rank_plan.unlink()
            arguments = {
                "rank": 0,
                "plan_dir": plan_dir,
                "builder": builder,
                "work_dir": work_dir,
                "protenix_root_dir": root,
                "python_executable": sys.executable,
                "workers": 1,
                "compresslevel": 3,
                "log_path": plan_dir / "logs" / "rank-000.log",
                "pid_path": plan_dir / "logs" / "rank-000.pid",
                "resume": False,
            }

            missing_rank = subprocess.run(
                ["bash", "-lc", launcher.build_remote_script(**arguments)],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(missing_rank.returncode, 2)
            self.assertIn(str(rank_plan), missing_rank.stderr)

            rank_plan.write_text("{}\n", encoding="utf-8")
            ccd_cache = root / "common" / "components.cif.rdkit_mol.pkl"
            ccd_cache.unlink()
            missing_cache = subprocess.run(
                ["bash", "-lc", launcher.build_remote_script(**arguments)],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(missing_cache.returncode, 2)
            self.assertIn(str(ccd_cache), missing_cache.stderr)
            self.assertFalse(arguments["pid_path"].exists())

    def test_immediate_builder_failure_is_reported_and_pid_is_removed(self):
        with TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            work_dir, builder, plan_dir = _write_remote_fixture(
                root,
                builder_source="raise SystemExit(7)\n",
            )
            status_dir = plan_dir / "status"
            status_dir.mkdir()
            (status_dir / "rank-000.json").write_text(
                json.dumps(
                    {
                        "plan_id": "an-old-plan",
                        "state": "complete",
                    }
                ),
                encoding="utf-8",
            )
            pid_path = plan_dir / "logs" / "rank-000.pid"
            script = launcher.build_remote_script(
                rank=0,
                plan_dir=plan_dir,
                builder=builder,
                work_dir=work_dir,
                protenix_root_dir=root,
                python_executable=sys.executable,
                workers=1,
                compresslevel=3,
                log_path=plan_dir / "logs" / "rank-000.log",
                pid_path=pid_path,
                resume=False,
            )

            completed = subprocess.run(
                ["bash", "-lc", script],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(completed.returncode, 6, completed.stderr)
            self.assertIn("exited during startup", completed.stderr)
            self.assertFalse(pid_path.exists())

    def test_stale_pid_from_another_host_does_not_override_shared_lease(self):
        with TemporaryDirectory(dir="/tmp") as temporary_directory:
            root = Path(temporary_directory)
            work_dir, builder, plan_dir = _write_remote_fixture(
                root,
                builder_source="import time\ntime.sleep(30)\n",
            )
            log_dir = plan_dir / "logs"
            log_dir.mkdir()
            pid_path = log_dir / "rank-000.pid"
            pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
            script = launcher.build_remote_script(
                rank=0,
                plan_dir=plan_dir,
                builder=builder,
                work_dir=work_dir,
                protenix_root_dir=root,
                python_executable=sys.executable,
                workers=1,
                compresslevel=3,
                log_path=log_dir / "rank-000.log",
                pid_path=pid_path,
                resume=True,
            )

            completed = subprocess.run(
                ["bash", "-lc", script],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            launched_pid = int(pid_path.read_text(encoding="utf-8").strip())
            try:
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertNotEqual(launched_pid, os.getpid())
                os.kill(launched_pid, 0)
            finally:
                _terminate_process(launched_pid)


class LaunchAndStatusTest(unittest.TestCase):
    def _fixture(self, base: Path, world_size: int = 2):
        plan_dir = base / "plan"
        _write_plan(plan_dir, world_size=world_size)
        builder = base / "builder.py"
        builder.write_text("pass\n", encoding="utf-8")
        work_dir = base / "work"
        work_dir.mkdir()
        data_dir = base / "data"
        data_dir.mkdir()
        return plan_dir, builder, work_dir, data_dir

    def test_dry_run_does_not_call_ssh(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan_dir, builder, work_dir, data_dir = self._fixture(root)
            with mock.patch.object(
                launcher.subprocess,
                "run",
                side_effect=AssertionError("must not execute"),
            ):
                result = launcher.launch_ranks(
                    hosts=["worker-a", "worker-b"],
                    plan_dir=plan_dir,
                    builder=builder,
                    work_dir=work_dir,
                    protenix_root_dir=data_dir,
                    python_executable=sys.executable,
                    execute=False,
                    fast_path_mode="require",
                )

        self.assertFalse(result["executed"])
        self.assertEqual(result["world_size"], 2)
        self.assertEqual(
            [item["rank"] for item in result["launches"]],
            [0, 1],
        )
        self.assertTrue(
            all(
                "--require-fast-path" in item["command"]
                for item in result["launches"]
            )
        )

    def test_host_count_must_equal_world_size(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan_dir, builder, work_dir, data_dir = self._fixture(root)

            with self.assertRaisesRegex(ValueError, "world_size=2"):
                launcher.launch_ranks(
                    hosts=["only-one"],
                    plan_dir=plan_dir,
                    builder=builder,
                    work_dir=work_dir,
                    protenix_root_dir=data_dir,
                )

    def test_log_directory_must_stay_under_shared_plan(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan_dir, builder, work_dir, data_dir = self._fixture(root)

            with self.assertRaisesRegex(
                ValueError,
                "inside the shared",
            ):
                launcher.launch_ranks(
                    hosts=["worker-a", "worker-b"],
                    plan_dir=plan_dir,
                    builder=builder,
                    work_dir=work_dir,
                    protenix_root_dir=data_dir,
                    log_dir=root / "node-local-logs",
                )

    def test_execute_runs_all_ssh_commands_and_sorts_results(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan_dir, builder, work_dir, data_dir = self._fixture(root)

            def fake_run(launch):
                return {
                    "rank": launch.rank,
                    "host": launch.host,
                    "returncode": 0,
                    "stdout": "launched",
                    "stderr": "",
                }

            with mock.patch.object(
                launcher,
                "_run_ssh",
                side_effect=fake_run,
            ) as run:
                result = launcher.launch_ranks(
                    hosts=["worker-a", "worker-b"],
                    plan_dir=plan_dir,
                    builder=builder,
                    work_dir=work_dir,
                    protenix_root_dir=data_dir,
                    execute=True,
                )

        self.assertTrue(result["executed"])
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            [item["rank"] for item in result["results"]],
            [0, 1],
        )

    def test_status_ignores_heartbeat_from_an_old_plan(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan_dir, _, _, _ = self._fixture(root)
            status_dir = plan_dir / "status"
            status_dir.mkdir()
            (status_dir / "rank-000.json").write_text(
                json.dumps(
                    {
                        "plan_id": "old-plan",
                        "rank": 0,
                        "state": "complete",
                        "completed": 99,
                    }
                ),
                encoding="utf-8",
            )
            (status_dir / "rank-001.json").write_text(
                json.dumps(
                    {
                        "plan_id": "test-plan",
                        "rank": 1,
                        "state": "running",
                        "completed": 7,
                    }
                ),
                encoding="utf-8",
            )

            status = launcher.aggregate_status(
                plan_dir,
                hosts=["worker-a", "worker-b"],
            )

        self.assertEqual(status["states"], {"stale": 1, "running": 1})
        self.assertEqual(status["completed"], 7)
        self.assertEqual(status["ranks"][0]["host"], "worker-a")


if __name__ == "__main__":
    unittest.main()
