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

"""Live metric output, verified result reuse, and monitoring an active evaluation."""

import contextlib
import fcntl
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HAS_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "scipy", "biotite", "torch", "rdkit")
)
if HAS_DEPS:
    from scripts.evaluation import evaluate_peptide_pose as driver
    from scripts.evaluation.peptide_pose_inputs import write_json

    from tests.test_peptide_pose_evaluation import coordinates, reference_fixture


@unittest.skipUnless(HAS_DEPS, "Requires Protenix scientific Python dependencies")
class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = reference_fixture()
        self.prediction = coordinates(self.reference)
        source = self.root / "native.cif"
        source.write_text(
            "source is hashed; native atom mapping comes from the reference"
        )
        write_json(self.root / "reference.json", self.reference)
        self.case = {
            "name": "fixture",
            "status": "ready",
            "source": str(source),
            "source_sha256": driver.file_hash(source),
            "reference": "reference.json",
            "artifacts": {
                "reference.json": driver.file_hash(self.root / "reference.json")
            },
        }
        self.manifest = {"cases": [self.case], "signature": {}}
        write_json(self.root / "manifest.json", self.manifest)
        self.run = {
            "seeds": [101],
            "samples": 1,
            "manifest_sha256": driver.file_hash(self.root / "manifest.json"),
        }
        write_json(self.root / "run_config.json", self.run)
        self.cif, conf = driver.prediction_paths(self.root, "fixture", 101, 0)
        write_json(conf, {"ranking_score": 0.9})
        self.cif.write_text("prediction reader is mocked in these CPU tests")
        self.record = {
            "status": "complete",
            "run_sha256": driver.digest(self.run),
            "files": {
                str(p.relative_to(self.root)): driver.file_hash(p)
                for p in (self.cif, conf)
            },
        }
        write_json(self.root / "prediction_manifest.json", {"fixture": self.record})
        self.args = SimpleNamespace(
            output_dir=self.root,
            contact_cutoff=5.0,
            no_atom_symmetry=True,
            min_receptor_coverage=0.9,
            thresholds=[2, 5],
            ids=None,
            stage="all",
            once=False,
            poll_interval=0.01,
        )

    def test_console_reports_each_sample_and_distinguishes_top1_from_best(self):
        report = {"name": "fixture", "poses": [], "peptides": [], "errors": []}
        for sample, rmsd, confidence in ((0, 4.0, 0.9), (1, 1.0, 0.2)):
            report["poses"].append(
                {
                    "seed": 101,
                    "sample": sample,
                    "ranking_score": confidence,
                    "max_peptide_rmsd": rmsd,
                    "receptor_rmsd": 0.5,
                }
            )
            for chain in ("L", "M"):
                report["peptides"].append(
                    {
                        "seed": 101,
                        "sample": sample,
                        "peptide_chain": chain,
                        "pose_rmsd": rmsd,
                        "backbone_pose_rmsd": None,
                        "fnat": None,
                    }
                )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            driver.display_case_metrics(report, 2)
        text = output.getvalue()
        self.assertEqual(text.count("peptide="), 4)
        self.assertIn("sample=0 peptide=L: RMSD=4.000 A", text)
        self.assertIn("sample=1 peptide=M: RMSD=1.000 A", text)
        self.assertIn("top1 max peptide RMSD=4.000 A", text)
        self.assertIn("best max peptide RMSD=1.000 A", text)
        self.assertIn("backbone RMSD=n/a, fnat=n/a", text)

    def test_cached_live_scores_are_reused_but_changed_prediction_is_rejected(self):
        with patch.object(driver, "read_prediction", return_value=self.prediction):
            report = driver.score_case(self.args, self.case, self.run, self.record)
        self.assertFalse(report["errors"])
        with patch.object(
            driver, "score_pose", side_effect=AssertionError("Unnecessary rescoring")
        ), patch("builtins.print"):
            self.assertTrue(driver.score(self.args, self.manifest, {"fixture": report}))
            self.cif.write_text("changed")
            self.assertFalse(
                driver.score(self.args, self.manifest, {"fixture": report})
            )
        self.assertIn(
            "artifact changed",
            driver.read_json(self.root / "scores/errors.json")[0]["error"],
        )

    def test_changing_scoring_settings_invalidates_live_cache(self):
        with patch.object(driver, "read_prediction", return_value=self.prediction):
            report = driver.score_case(self.args, self.case, self.run, self.record)
            self.args.contact_cutoff = 8.0
            with patch.object(driver, "score_pose", wraps=driver.score_pose) as scorer:
                updated = driver.score_case(
                    self.args, self.case, self.run, self.record, report
                )
        scorer.assert_called_once()
        self.assertNotEqual(report["context"], updated["context"])

    def test_failed_evaluation_shows_error_not_zero_rmsd(self):
        report = driver.score_case(
            self.args,
            self.case,
            self.run,
            {"status": "failed", "error": "worker crashed"},
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            driver.display_case_metrics(report, 5)
        self.assertIn("0/5 samples (incomplete)", output.getvalue())
        self.assertIn("worker crashed", output.getvalue())
        self.assertNotIn("RMSD=0", output.getvalue())

    def test_changed_native_reference_is_rejected(self):
        write_json(self.root / "reference.json", {**self.reference, "changed": True})
        report = driver.score_case(self.args, self.case, self.run, self.record)
        self.assertIn("Prepared reference changed", report["errors"][0]["error"])

    def test_single_worker_reports_before_predicting_the_next_case(self):
        pending = [self.case, {**self.case, "name": "second"}]
        for case in pending:
            case["input"] = f"{case['name']}_input.json"
            write_json(self.root / case["input"], [{"name": case["name"]}])
        events = []
        self.args.seeds, self.args.samples = [101], 1
        self.manifest["msa_enabled"] = False

        class Dataset:
            def __getitem__(inner, index):
                return (
                    {
                        "sample_name": pending[index]["name"],
                        "N_token": 4,
                        "entity_poly_type": {"1": "polypeptide(L)"},
                    },
                    None,
                    "",
                )

        def predict(data):
            events.append("predict:" + data["sample_name"])
            return {}

        def dump(dataset, name, seed, prediction, atoms, polymers):
            cif, conf = driver.prediction_paths(self.root, name, seed, 0)
            write_json(conf, {"ranking_score": 0.9})
            cif.write_text("synthetic prediction")

        progress_path = self.root / "worker.json"

        def completed(case, record):
            self.assertEqual(driver.read_json(progress_path)[case["name"]], record)
            report = driver.score_case(self.args, case, self.run, record)
            self.assertFalse(report["errors"])
            driver.display_case_metrics(report, 1)
            events.append("metrics:" + case["name"])

        runner = SimpleNamespace(predict=predict, dumper=SimpleNamespace(dump=dump))
        modules = {
            "runner.inference": SimpleNamespace(InferenceRunner=lambda configs: runner),
            "protenix.data.inference.infer_dataloader": SimpleNamespace(
                InferenceDataset=lambda configs: Dataset()
            ),
            "protenix.utils.seed": SimpleNamespace(
                seed_everything=lambda **kwargs: None
            ),
        }
        with patch.dict("sys.modules", modules), patch.object(
            driver, "model_configs", return_value=None
        ), patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.empty_cache"
        ), patch.object(
            driver, "read_prediction", return_value=self.prediction
        ), patch(
            "builtins.print"
        ):
            success = driver.predict_cases(
                self.args,
                self.manifest,
                self.run,
                pending,
                progress_path,
                self.root / "worker_inputs.json",
                completed,
            )
        self.assertTrue(success)
        self.assertEqual(
            events,
            ["predict:fixture", "metrics:fixture", "predict:second", "metrics:second"],
        )

    def test_all_mode_prints_metrics_before_predict_returns_and_does_not_repeat_at_end(
        self,
    ):
        output = io.StringIO()

        def predict(args, manifest, on_complete):
            on_complete(self.case, self.record)
            self.assertIn("peptide=L: RMSD=0.000 A", output.getvalue())
            self.assertFalse((self.root / "scores/summary.json").exists())
            return True

        with patch.object(driver, "prepare", return_value=self.manifest), patch.object(
            driver, "predict", side_effect=predict
        ), patch.object(
            driver, "read_prediction", return_value=self.prediction
        ), patch.object(
            driver, "score_pose", wraps=driver.score_pose
        ) as scorer, contextlib.redirect_stdout(
            output
        ):
            driver.run_stages(self.args)
        scorer.assert_called_once()
        self.assertEqual(output.getvalue().count("peptide=L:"), 1)
        self.assertEqual(
            driver.read_json(self.root / "scores/summary.json")["complete_complexes"], 1
        )

    def test_watch_once_works_with_active_output_lock_and_never_writes(self):
        with (self.root / ".evaluation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            before = {
                p: p.stat().st_mtime_ns for p in self.root.rglob("*") if p.is_file()
            }
            output = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "evaluate",
                    "watch",
                    "--once",
                    "--no-atom-symmetry",
                    "--output-dir",
                    str(self.root),
                ],
            ), patch.object(
                driver, "read_prediction", return_value=self.prediction
            ), patch.object(
                driver,
                "write_json",
                side_effect=AssertionError("Monitor must be read-only"),
            ), contextlib.redirect_stdout(
                output
            ):
                driver.main()
            after = {
                p: p.stat().st_mtime_ns for p in self.root.rglob("*") if p.is_file()
            }
        self.assertEqual(before, after)
        self.assertIn("peptide=L: RMSD=0.000 A", output.getvalue())
        self.assertFalse((self.root / "scores").exists())

    def test_watch_displays_newly_completed_case_once(self):
        self.manifest["cases"].append({**self.case, "name": "second"})
        write_json(self.root / "manifest.json", self.manifest)
        self.run["manifest_sha256"] = driver.file_hash(self.root / "manifest.json")
        write_json(self.root / "run_config.json", self.run)
        self.record["run_sha256"] = driver.digest(self.run)
        write_json(self.root / "prediction_manifest.json", {})
        calls = 0

        def tick(interval):
            nonlocal calls
            calls += 1
            if calls > 3:
                self.fail(
                    "Monitor did not stop after all cases reached terminal states"
                )
            progress = {"fixture": self.record}
            if calls == 3:
                progress["second"] = {"status": "failed", "error": "simulated crash"}
            (self.root / "prediction_manifest.json").write_text(json.dumps(progress))

        output = io.StringIO()
        with patch.object(driver.time, "sleep", side_effect=tick), patch.object(
            driver, "read_prediction", return_value=self.prediction
        ), patch.object(
            driver,
            "write_json",
            side_effect=AssertionError("Monitor must be read-only"),
        ), contextlib.redirect_stdout(
            output
        ):
            driver.watch(self.args)
        self.assertEqual(output.getvalue().count("peptide=L:"), 1)
        self.assertEqual(output.getvalue().count("simulated crash"), 1)
        self.assertEqual(calls, 3)


if __name__ == "__main__":
    unittest.main()
