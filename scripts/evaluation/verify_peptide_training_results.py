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

"""Verify training reconstruction scores against saved coordinates and source atoms."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from protenix.data.core.featurizer import Featurizer
from protenix.utils.file_io import load_gzip_pickle
from scripts.evaluation.evaluate_peptide_training import REPO, score_interface, select_predictions
from scripts.evaluation.peptide_pose_inputs import file_hash, write_json


def verify(root, recovery_dir=None):
    manifest = json.loads((root / "manifest.json").read_text())
    run = json.loads((root / "run_config.json").read_text())
    seeds = run.get("seeds", [run.get("seed", 101)])
    expected_ids = [(seed, sample) for seed in seeds for sample in range(run["samples"])]
    run_hash = file_hash(root / "run_config.json")
    assert run["manifest_sha256"] == file_hash(root / "manifest.json")
    assert len(manifest["cases"]) == len({c["pdb_id"] for c in manifest["cases"]})
    assert len(manifest["cases"]) == manifest["selection"]["count"]
    assert all(c["training_exposures"] > 0 for c in manifest["cases"])
    validation = {
        r["name"]: r for r in json.loads((root / "final_crop_validation.json").read_text())
    }
    recovery_run = None
    recovery_cases = {}
    recovery_validation = {}
    if recovery_dir is not None:
        recovery_run = json.loads((recovery_dir / "run_config.json").read_text())
        for key in run:
            if key != "manifest_sha256":
                assert recovery_run[key] == run[key], key
        recovery_manifest = json.loads((recovery_dir / "manifest.json").read_text())
        assert recovery_run["manifest_sha256"] == file_hash(recovery_dir / "manifest.json")
        recovery_cases = {r["name"]: r for r in recovery_manifest["cases"]}
        recovery_validation = {
            r["name"]: r for r in json.loads((recovery_dir / "final_crop_validation.json").read_text())
        }
    sample_count = 0
    max_metric_error = 0.0
    checked = []
    combined = []
    recovered = []
    for case in manifest["cases"]:
        source = root
        expected = case
        case_validation = validation
        result = json.loads((source / "cases" / (case["name"] + ".json")).read_text())
        expected_run_hash = run_hash
        if result["status"] != "complete" and case["name"] in recovery_cases:
            source = recovery_dir
            expected = recovery_cases[case["name"]]
            for key in case:
                if key != "crop_seed":
                    assert expected[key] == case[key], key
            result = json.loads((source / "cases" / (case["name"] + ".json")).read_text())
            expected_run_hash = file_hash(source / "run_config.json")
            case_validation = recovery_validation
            recovered.append(case["name"])
        assert result["status"] == "complete", case["name"]
        assert result["run_sha256"] == expected_run_hash
        for key, value in expected.items():
            assert result[key] == value, (case["name"], key)
        coordinate_path = source / "cases" / (case["name"] + ".npz")
        assert result["coordinates_sha256"] == file_hash(coordinate_path)
        meta = result["crop"]
        assert case_validation[case["name"]]["status"] == "ok"
        assert case_validation[case["name"]]["metadata"]["feature_sha256"] == meta["feature_sha256"]
        cache_path = Path(meta["cache_path"])
        if not cache_path.is_absolute():
            cache_path = REPO / cache_path
        assert meta["cache_sha256"] == file_hash(cache_path)
        full = load_gzip_pickle(cache_path)["atom_array"]
        inverse = np.empty(len(full), dtype=int)
        inverse[full.peptide_source_atom_index] = np.arange(len(full))
        selected = np.asarray(meta["source_atom_indices"], dtype=int)
        assert len(np.unique(selected)) == len(selected)
        atoms = full[inverse[selected]]
        permutations = Featurizer(None, atoms).get_atom_permutation_list()
        with np.load(coordinate_path) as coordinates:
            np.testing.assert_array_equal(coordinates["native"], atoms.coord)
            np.testing.assert_array_equal(coordinates["source_atom_indices"], selected)
            if "seeds" in run:
                np.testing.assert_array_equal(coordinates["prediction_seeds"], [s for s, _ in expected_ids])
                np.testing.assert_array_equal(coordinates["prediction_samples"], [i for _, i in expected_ids])
            for label in ("before", "after"):
                saved = result[label]
                assert len(coordinates[label]) == len(saved["samples"]) == len(expected_ids)
                assert [(s.get("seed", seeds[0]), s["sample"]) for s in saved["samples"]] == expected_ids
                for sample, xyz in enumerate(coordinates[label]):
                    score = score_interface(atoms, {"atom_perm_list": permutations}, meta, xyz)
                    reported = saved["samples"][sample]
                    for metric in ("peptide_rmsd", "peptide_shape_rmsd", "receptor_rmsd"):
                        difference = abs(score[metric] - reported[metric])
                        assert math.isfinite(difference) and difference < 1e-6, (case["name"], metric)
                        max_metric_error = max(max_metric_error, difference)
                    sample_count += 1
                selected_predictions = select_predictions(saved["samples"])
                assert saved["top1"] == selected_predictions["top1"]
                assert saved["best"] == selected_predictions["best"]
        checked.append(case["name"])
        combined.append({**result, "source_run": str(source)})
        print(f"Verified {len(checked)}/{len(manifest['cases'])}: {case['name']}", flush=True)
    report = {
        "verified_structures": len(checked), "verified_predictions": sample_count,
        "maximum_metric_recomputation_error": max_metric_error,
        "coordinate_integrity": True, "native_source_atom_integrity": True,
        "matches_cpu_validated_features": True, "selection_integrity": True,
        "ranking_and_oracle_selection_verified": True, "cases": checked,
        "recovered_cases": recovered,
    }
    write_json(root / "verification.json", report)
    write_json(root / "combined_results.json", combined)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--recovery-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir.resolve(), args.recovery_dir), indent=2))
