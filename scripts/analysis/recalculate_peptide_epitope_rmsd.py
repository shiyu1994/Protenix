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

"""Rescore saved test predictions after native receptor epitope C-alpha alignment."""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.recalculate_peptide_ca_rmsd import Inputs, ca_reference, compare, require, write_csv
from scripts.evaluation.peptide_pose_inputs import write_json
from scripts.evaluation.peptide_pose_metrics import read_prediction, score_pose

FIGURE_SHARED_IDS = frozenset(
    "8are 7ykd 8arn 8j5u 8rte 8qm0 8qlg 8jgf 8fsr 8s6n 8sg1 8jgg 8wx5 8qlk "
    "8h0q 8fg6 8jzd 8hdj 8qci 7udk 8sao 8h0p 7udj 8zlz 8brh 8qlm 8s6o".split()
)


def epitope_reference(reference, cutoff=10.0):
    """Select native receptor CA atoms within cutoff of any native target-peptide CA.

    Multiple peptides contribute a union of epitopes to one consistent rigid fit.
    Receptor chain groups are retained for the existing equivalent-copy assignment.
    """
    require(np.isfinite(cutoff) and cutoff > 0, "Epitope cutoff must be finite and positive")
    ca, peptide_audit = ca_reference(reference)
    peptide = [r for r in ca["atoms"] if r[0] in reference["peptide_chains"]]
    receptor = [
        r for r in reference["atoms"]
        if reference["chains"][r[0]]["role"] == "receptor" and r[3] == "CA" and r[4] == "C"
    ]
    require(bool(receptor), f"No native receptor CA atoms: {reference['name']}")
    distances = cdist(np.asarray([r[5:8] for r in receptor]), np.asarray([r[5:8] for r in peptide])).min(axis=1)
    selected = [r for r, distance in zip(receptor, distances) if distance <= cutoff]
    require(len(selected) >= 3, f"Fewer than three epitope CA atoms: {reference['name']}")
    xyz = np.asarray([r[5:8] for r in selected])
    require(np.linalg.matrix_rank(xyz - xyz.mean(axis=0)) >= 2,
            f"Collinear native epitope atoms: {reference['name']}")
    result = dict(reference)
    result["atoms"] = selected + [r for r in reference["atoms"] if r[0] in reference["peptide_chains"]]
    return result, {
        "cutoff_angstrom": cutoff, "distance_atoms": "native receptor CA to native peptide CA",
        "multi_peptide_policy": "union of epitopes; one fit; maximum peptide RMSD",
        "global_receptor_ca_atoms": len(receptor), "epitope_receptor_ca_atoms": len(selected),
        "peptide_atoms": peptide_audit,
        "epitope_atoms": [
            {"chain": r[0], "position": r[1], "residue": r[2], "distance_to_peptide_ca": float(distance)}
            for r, distance in zip(receptor, distances) if distance <= cutoff
        ],
    }


def rescore(root, global_scores, cutoff, inputs):
    run = inputs.json(root / "run_config.json")
    require(run == global_scores["run"], "Global and epitope scoring must use the same prediction run")
    manifest = inputs.json(root / "manifest.json", run["manifest_sha256"])
    progress = inputs.json(root / "prediction_manifest.json")
    original_summary = inputs.json(root / "scores/summary.json")
    run_digest = hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()
    old = {(p["name"], p["seed"], p["sample"]): p for p in global_scores["poses"]}
    poses, audits, errors = [], {}, []
    for case in manifest["cases"]:
        name = case["name"]
        record = progress.get(name, {})
        if case["status"] != "ready" or record.get("status") != "complete":
            errors.append({"name": name, "error": "Original evaluation incomplete"})
            continue
        require(record["run_sha256"] == run_digest, f"Run identity mismatch: {name}")
        reference = inputs.json(root / case["reference"], case["artifacts"][case["reference"]])
        try:
            heavy_reference, audits[name] = epitope_reference(reference, cutoff)
            reduced, _ = ca_reference(heavy_reference)
        except ValueError as exc:
            errors.append({"name": name, "error": str(exc)})
            continue
        pending = []
        for seed in run["seeds"]:
            for sample in range(run["samples"]):
                folder = root / "predictions" / name / f"seed_{seed}" / "predictions"
                cif = folder / f"{name}_sample_{sample}.cif"
                confidence = folder / f"{name}_summary_confidence_sample_{sample}.json"
                for artifact in (cif, confidence):
                    inputs.check(artifact, record["files"][str(artifact.relative_to(root))])
                ranking = inputs.json(confidence)["ranking_score"]
                previous = old[name, seed, sample]
                require(ranking == previous["ranking_score"], f"Ranking changed: {name}")
                prediction = read_prediction(cif)
                try:
                    scores = {
                        "ca": score_pose(reduced, prediction, symmetry=False,
                                         min_receptor_coverage=original_summary["min_receptor_ca_coverage"]),
                        "heavy": score_pose(heavy_reference, prediction, symmetry=original_summary["atom_symmetry"],
                                            min_receptor_coverage=original_summary["min_receptor_ca_coverage"]),
                    }
                except ValueError as exc:
                    errors.append({"name": name, "seed": seed, "sample": sample, "error": str(exc)})
                    continue
                for result in scores.values():
                    result.pop("receptor_contact_atom_coverage", None)
                    for peptide_score in result["peptides"]:
                        for field in ("native_contacts", "predicted_contacts", "fnat", "contact_precision"):
                            peptide_score.pop(field, None)
                pending.append({
                    "name": name, "seed": seed, "sample": sample, "ranking_score": ranking,
                    "peptide_ca_rmsd": scores["ca"]["max_peptide_rmsd"],
                    "peptide_heavy_rmsd": scores["heavy"]["max_peptide_rmsd"],
                    "global_peptide_ca_rmsd": previous["peptide_ca_rmsd"],
                    "global_peptide_heavy_rmsd": previous["original_peptide_heavy_rmsd"],
                    "receptor_rmsd": scores["ca"]["receptor_rmsd"],
                    "global_receptor_rmsd": previous["receptor_rmsd"],
                    "scores": scores,
                })
        if len(pending) == len(run["seeds"]) * run["samples"]:
            poses.extend(pending)
        print(f"{root.name}: {name}, epitope {audits[name]['epitope_receptor_ca_atoms']} CA, "
              f"{len(pending)} poses", flush=True)
    return {"run": run, "requested": len(manifest["cases"]), "poses": poses, "atom_audit": audits, "errors": errors}


def metric_rows(rows, metric, shared_ids=None):
    selected = []
    for row in rows:
        if shared_ids is not None and row["name"].split("-")[0] not in shared_ids:
            continue
        receptor_rmsd = row["receptor_rmsd"]
        if metric.startswith("global_"):
            receptor_rmsd = row["global_receptor_rmsd"]
        elif metric == "peptide_heavy_rmsd":
            receptor_rmsd = row["scores"]["heavy"]["receptor_rmsd"]
        selected.append({**row, "peptide_ca_rmsd": row[metric], "receptor_rmsd": receptor_rmsd})
    return selected


def render_report(summary):
    lines = [
        "Epitope alignment of saved October 6 peptide test predictions.", "",
        f"Cutoff: {summary['cutoff_angstrom']:g} Å between native receptor Cα and native peptide Cα atoms. "
        "Fit the selected receptor Cα atoms and apply that transform to the peptide; no peptide superposition. "
        "Multiple peptides use the union of epitopes and the maximum peptide RMSD per complex.", "",
        "Five saved predictions per complex, seed 101. Top1 uses confidence ranking; "
        "best-of-5 is selected independently for each RMSD definition. All comparisons use shared completed cases.", "",
    ]
    for group, title in (("all_test", "Full test set"), ("figure_shared", "27 PDB IDs shared with the figure")):
        lines += [title, "", "| Atoms / alignment | Selection | N | Official mean / median Å | "
                  "Finetuned mean / median Å | Official ≤2 / ≤5 Å | Finetuned ≤2 / ≤5 Å |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for metric, data in summary[group].items():
            n = data["shared_successful_cases"]
            for selection in ("top1", "best"):
                a, b = data["before"][selection], data["after"][selection]
                ac, bc = a["success_counts"], b["success_counts"]
                lines.append(f"| {metric} | {selection} | {n} | {a['mean']:.2f} / {a['median']:.2f} | "
                             f"{b['mean']:.2f} / {b['median']:.2f} | "
                             f"{100*ac['2']/n:.1f}% / {100*ac['5']/n:.1f}% | "
                             f"{100*bc['2']/n:.1f}% / {100*bc['5']/n:.1f}% |")
        lines.append("")
    lines += [
        "The figure uses 25 predictions per case, and its raw scores are unavailable. "
        "The earlier visual estimate on the 27 shared PDB IDs was approximately 6/27 at ≤2 Å "
        "and 19/27 at ≤5 Å. These are approximate figure readings, not machine-verified results. "
        "Matching PDB IDs does not verify identical receptor/peptide selections; 8rte has two target peptide chains "
        "in our evaluation. The shared IDs appear in the figure's non-cyclic group.", "",
        "Original prediction files and previous score reports were not modified. No inference was run.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-ca-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=10.0)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "Use a new output directory")
    args.output_dir.mkdir(parents=True)
    inputs = Inputs()
    previous = inputs.json(args.global_ca_dir / "summary.json")
    roots = {"before": Path(previous["baseline_test"]), "after": Path(previous["evaluation_root"]) / "test51"}
    results = {}
    for label, root in roots.items():
        old = inputs.json(args.global_ca_dir / f"test_{label}.json")
        resources = old["run"]["input_environment"]["ccd_resources"]
        ccd_pickle = Path(resources["ccd_components_rdkit_mol_file"]["path"])
        os.environ.setdefault("PROTENIX_ROOT_DIR", str(ccd_pickle.parent.parent))
        from protenix.data.core import ccd

        require(Path(ccd.RKDIT_MOL_PKL).resolve() == ccd_pickle.resolve(), "CCD path differs from original run")
        for resource in resources.values():
            stat = Path(resource["path"]).stat()
            require(stat.st_size == resource["size"] and stat.st_mtime_ns == resource["mtime_ns"],
                    f"CCD resource changed since evaluation: {resource['path']}")
            inputs.check(resource["path"])
        results[label] = rescore(root, old, args.cutoff, inputs)
        write_json(args.output_dir / f"test_{label}.json", results[label])
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "cutoff_angstrom": args.cutoff,
               "checkpoint": previous["checkpoint"], "figure_shared_ids": sorted(FIGURE_SHARED_IDS),
               "errors": {label: d["errors"] for label, d in results.items()}}
    for group, ids in (("all_test", None), ("figure_shared", FIGURE_SHARED_IDS)):
        summary[group] = {}
        for label, metric in (
            ("CA / global", "global_peptide_ca_rmsd"), ("CA / epitope", "peptide_ca_rmsd"),
            ("heavy / global", "global_peptide_heavy_rmsd"), ("heavy / epitope", "peptide_heavy_rmsd"),
        ):
            before = metric_rows(results["before"]["poses"], metric, ids)
            after = metric_rows(results["after"]["poses"], metric, ids)
            summary[group][label], cases = compare(before, after)
            write_csv(args.output_dir / f"{group}_{metric}.csv", cases)
    for source in (
        Path(__file__), REPO / "scripts/analysis/recalculate_peptide_ca_rmsd.py",
        REPO / "scripts/evaluation/peptide_pose_metrics.py", REPO / "scripts/evaluation/peptide_pose_inputs.py",
    ):
        inputs.check(source)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "input_hashes.json", inputs.hashes)
    (args.output_dir / "comparison.md").write_text(render_report(summary))
    shutil.copyfile(__file__, args.output_dir / Path(__file__).name)
    print(render_report(summary), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
