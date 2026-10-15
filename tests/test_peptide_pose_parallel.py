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

"""Case partitioning and real spawned-worker orchestration without GPU inference."""

import hashlib
import importlib.util
import json
import multiprocessing
import os
import signal
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.evaluation.peptide_pose_parallel import launch_workers, partition_cases

HAS_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "scipy", "biotite", "torch", "rdkit")
)


def fake_predict(args, manifest, config, cases, progress_path, input_json):
    """Exercise process isolation, atomic progress, and a transient worker crash."""
    progress = {}
    for case in cases:
        name = case["name"]
        if (args.output_dir / f"fail_{name}").exists():
            raise RuntimeError("simulated worker crash")
        artifact = args.output_dir / f"fake_{name}.json"
        artifact.write_text(
            json.dumps(
                {
                    "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
                    "world_size": os.environ.get("WORLD_SIZE"),
                    "pid": os.getpid(),
                }
            )
        )
        progress[name] = {
            "status": "complete",
            "run_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest(),
            "files": {artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()},
        }
        temp = progress_path.with_suffix(".tmp")
        temp.write_text(json.dumps(progress))
        temp.replace(progress_path)
    return True


def waiting_predict(args, manifest, config, cases, progress_path, input_json):
    progress_path.write_text(str(os.getpid()))
    while True:
        time.sleep(1)


def wait_for_metrics_predict(args, manifest, config, cases, progress_path, input_json):
    if cases[0]["name"] == "b":
        deadline = time.monotonic() + 10
        while not (args.output_dir / "metrics_a").exists():
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Coordinator did not report metrics while another worker was running"
                )
            time.sleep(0.05)
    return fake_predict(args, manifest, config, cases, progress_path, input_json)


class PartitionTests(unittest.TestCase):
    def test_51_cases_assigned_once_across_eight_workers(self):
        cases = [
            {"name": f"case_{i:02d}", "validation": {"input_tokens": 100 + i * 10}}
            for i in range(51)
        ]
        shards = partition_cases(cases, 8)
        names = [case["name"] for shard in shards for case in shard]
        self.assertEqual(len(shards), 8)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), {c["name"] for c in cases})
        self.assertEqual(shards, partition_cases(list(reversed(cases)), 8))
        loads = [
            sum(c["validation"]["input_tokens"] ** 2 for c in shard) for shard in shards
        ]
        self.assertLess(max(loads) - min(loads), 600**2)

    def test_fewer_cases_than_gpus_and_empty_work(self):
        cases = [{"name": "one", "validation": {"input_tokens": 100}}]
        self.assertEqual(partition_cases(cases, 8), [cases])
        self.assertEqual(partition_cases([], 8), [])
        with self.assertRaises(ValueError):
            partition_cases(cases, 0)

    def test_spawned_workers_get_isolated_devices_and_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(output_dir=root)
            jobs = [
                {
                    "gpu": str(i + 3),
                    "cases": [{"name": str(i)}],
                    "progress": root / f"progress_{i}.json",
                    "input": root / f"inputs_{i}.json",
                    "log": root / f"worker_{i}.log",
                }
                for i in range(2)
            ]
            with patch.dict(
                os.environ, {"WORLD_SIZE": "99", "CUDA_VISIBLE_DEVICES": "parent"}
            ):
                codes = launch_workers(fake_predict, (args, {}, {}), jobs, lambda: None)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "parent")
                self.assertEqual(os.environ["WORLD_SIZE"], "99")
            self.assertEqual(codes, [0, 0])
            results = [
                json.loads((root / f"fake_{i}.json").read_text()) for i in range(2)
            ]
            self.assertEqual([r["gpu"] for r in results], ["3", "4"])
            self.assertTrue(all(r["world_size"] is None for r in results))
            self.assertEqual(len({r["pid"] for r in results}), 2)
            self.assertTrue(all(j["log"].exists() for j in jobs))

    def test_interrupt_reaps_workers_and_restores_signal_handlers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(output_dir=root)
            job = {
                "gpu": "0",
                "cases": [{"name": "one"}],
                "progress": root / "ready",
                "input": root / "input.json",
                "log": root / "worker.log",
            }
            interrupted = False
            original = signal.getsignal(signal.SIGTERM)

            def update():
                nonlocal interrupted
                if job["progress"].exists() and not interrupted:
                    interrupted = True
                    os.kill(os.getpid(), signal.SIGTERM)

            with self.assertRaises(KeyboardInterrupt):
                launch_workers(waiting_predict, (args, {}, {}), [job], update)
            child_pid = int(job["progress"].read_text())
            self.assertNotIn(
                child_pid, {child.pid for child in multiprocessing.active_children()}
            )
            self.assertEqual(signal.getsignal(signal.SIGTERM), original)


@unittest.skipUnless(HAS_DEPS, "Requires Protenix scientific Python dependencies")
class ParallelResumeTests(unittest.TestCase):
    def test_metric_callback_runs_while_another_worker_is_still_active(self):
        from scripts.evaluation import evaluate_peptide_pose as driver

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(output_dir=root, gpus=["0", "1"])
            cases = [
                {
                    "name": name,
                    "status": "ready",
                    "validation": {"input_tokens": tokens},
                }
                for name, tokens in (("a", 100), ("b", 90))
            ]
            progress_path = root / "prediction_manifest.json"
            driver.write_json(progress_path, {})
            observed = []

            def completed(case, record):
                self.assertEqual(driver.read_json(progress_path)[case["name"]], record)
                observed.append(case["name"])
                if case["name"] == "a":
                    (root / "metrics_a").touch()

            with patch.object(driver, "predict_cases", wait_for_metrics_predict):
                driver.predict_parallel(
                    args, {}, {}, cases, {}, progress_path, completed
                )
            self.assertEqual(observed, ["a", "b"])
            self.assertTrue(
                all(
                    r["status"] == "complete"
                    for r in driver.read_json(progress_path).values()
                )
            )

    def test_worker_crash_is_recorded_and_resume_only_runs_unfinished_cases(self):
        from scripts.evaluation import evaluate_peptide_pose as driver

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(output_dir=root, gpus=["0", "1"])
            cases = [
                {
                    "name": name,
                    "status": "ready",
                    "validation": {"input_tokens": tokens},
                }
                for name, tokens in (("a", 100), ("b", 90), ("c", 80), ("d", 70))
            ]
            manifest = {"signature": {"input_environment": {}}, "cases": cases}
            config = {"samples": 1, "seeds": [101]}
            (root / "fail_d").touch()
            with patch.object(
                driver, "input_environment", return_value={}
            ), patch.object(
                driver, "make_run_config", return_value=config
            ), patch.object(
                driver, "predict_cases", fake_predict
            ):
                self.assertFalse(driver.predict(args, manifest))
                progress = driver.read_json(root / "prediction_manifest.json")
                self.assertEqual(
                    {name for name, r in progress.items() if r["status"] == "complete"},
                    {"a", "b", "c"},
                )
                self.assertEqual(progress["d"]["status"], "failed")
                self.assertIn("worker exited", progress["d"]["error"])
                saved = {
                    name: (root / f"fake_{name}.json").stat().st_mtime_ns
                    for name in ("a", "b", "c")
                }
                (root / "fail_d").unlink()
                args.gpus = [str(i) for i in range(8)]
                self.assertTrue(driver.predict(args, manifest))
                self.assertEqual(
                    saved,
                    {
                        name: (root / f"fake_{name}.json").stat().st_mtime_ns
                        for name in saved
                    },
                )
                progress = driver.read_json(root / "prediction_manifest.json")
                self.assertEqual(set(progress), {"a", "b", "c", "d"})
                self.assertTrue(
                    all(r["status"] == "complete" for r in progress.values())
                )
                assignments = [
                    driver.read_json(p)
                    for p in (root / "workers").glob("*/assignment.json")
                ]
                self.assertIn(
                    ["d"],
                    [
                        [c for w in a["workers"] for c in w["cases"]]
                        for a in assignments
                    ],
                )
                with patch.object(
                    driver,
                    "predict_parallel",
                    side_effect=AssertionError("Unnecessary rerun"),
                ):
                    self.assertTrue(driver.predict(args, manifest))


if __name__ == "__main__":
    unittest.main()
