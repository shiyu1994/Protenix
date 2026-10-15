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

"""Recalculate peptide C-alpha pose RMSDs from saved test CIFs and training NPZs.

This CPU analysis never creates model predictions or changes the original results.
Training C-alpha identities are recovered by matching native atoms to original PDB
CIF coordinates and elements, since preparation replaced peptide atom names.
"""

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from biotite.structure.io.pdbx import CIFFile
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.evaluation.evaluate_peptide_training import equivalent_chain_indices, score_interface
from scripts.evaluation.peptide_pose_inputs import file_hash, write_json
from scripts.evaluation.peptide_pose_metrics import read_prediction, receptor_fit, score_pose


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Inputs:
    """Record and verify the exact files used for this recalculation."""

    def __init__(self):
        self.hashes = {}

    def check(self, path, expected=None):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.hashes:
            self.hashes[key] = file_hash(path)
        actual = self.hashes[key]
        require(expected is None or actual == expected, f"File hash mismatch: {path}")
        return actual

    def json(self, path, expected=None):
        self.check(path, expected)
        return json.loads(Path(path).read_text())


def select(rows):
    top = min(rows, key=lambda r: (-r["ranking_score"], r["seed"], r["sample"]))
    best = min(rows, key=lambda r: (r["peptide_ca_rmsd"], r["seed"], r["sample"]))
    return {"top1": top, "best": best}


def ca_reference(reference):
    """Keep all receptor atoms but score only observed peptide carbon CA atoms."""
    peptide_chains = set(reference["peptide_chains"])
    result = dict(reference)
    result["atoms"] = [
        r for r in reference["atoms"] if r[0] not in peptide_chains or (r[3] == "CA" and r[4] == "C")
    ]
    audit = {}
    for chain in sorted(peptide_chains):
        atoms = [r for r in reference["atoms"] if r[0] == chain]
        ca_atoms = [r for r in result["atoms"] if r[0] == chain]
        require(bool(ca_atoms), f"No observed peptide CA atoms: {reference['name']} {chain}")
        represented = {r[1] for r in ca_atoms}
        audit[chain] = {
            "ca_atoms": len(ca_atoms),
            "residues_without_observed_named_ca": sorted({(r[1], r[2]) for r in atoms if r[1] not in represented}),
        }
    return result, audit


def rescore_test(root, inputs):
    run = inputs.json(root / "run_config.json")
    manifest = inputs.json(root / "manifest.json", run["manifest_sha256"])
    progress = inputs.json(root / "prediction_manifest.json")
    original_summary = inputs.json(root / "scores/summary.json")
    inputs.check(root / "scores/poses.csv")
    with (root / "scores/poses.csv").open() as handle:
        original = {(r["name"], int(r["seed"]), int(r["sample"])): r for r in csv.DictReader(handle)}
    digest = hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()
    poses, audits, references = [], {}, {}
    complete = []
    for case in manifest["cases"]:
        name = case["name"]
        record = progress.get(name, {})
        if case["status"] != "ready" or record.get("status") != "complete":
            continue
        require(record["run_sha256"] == digest, f"Run identity mismatch: {name}")
        ref_path = root / case["reference"]
        ref = inputs.json(ref_path, case["artifacts"][case["reference"]])
        references[name] = inputs.check(ref_path)
        reduced, audits[name] = ca_reference(ref)
        for seed in run["seeds"]:
            for sample in range(run["samples"]):
                folder = root / "predictions" / name / f"seed_{seed}" / "predictions"
                cif = folder / f"{name}_sample_{sample}.cif"
                conf_path = folder / f"{name}_summary_confidence_sample_{sample}.json"
                for path in (cif, conf_path):
                    inputs.check(path, record["files"][str(path.relative_to(root))])
                confidence = inputs.json(conf_path)["ranking_score"]
                old = original[name, seed, sample]
                require(abs(confidence - float(old["ranking_score"])) < 1e-8, f"Confidence mismatch: {name}")
                result = score_pose(
                    reduced, read_prediction(cif), symmetry=False,
                    min_receptor_coverage=original_summary["min_receptor_ca_coverage"],
                )
                error = abs(result["receptor_rmsd"] - float(old["receptor_rmsd"]))
                require(error <= 2e-6, f"Receptor fit changed beyond tie tolerance: {name} {error}")
                poses.append({
                    "name": name, "seed": seed, "sample": sample, "ranking_score": confidence,
                    "peptide_ca_rmsd": result["max_peptide_rmsd"],
                    "mean_peptide_ca_rmsd": result["mean_peptide_rmsd"],
                    "receptor_rmsd": result["receptor_rmsd"],
                    "original_peptide_heavy_rmsd": float(old["max_peptide_rmsd"]),
                    "receptor_mapping": result["receptor_mapping"],
                    "original_receptor_rmsd_error": error,
                    "receptor_mapping_changed": result["receptor_mapping"] != json.loads(old["receptor_mapping"]),
                    "peptides": [{
                        "chain": p["peptide_chain"], "predicted_chain": p["predicted_peptide_chain"],
                        "ca_rmsd": p["pose_rmsd"], "ca_atoms": p["n_atoms"],
                    } for p in result["peptides"]],
                })
        complete.append(name)
        print(f"Test {root.name}: {len(complete)}/{len(manifest['cases'])} {name}", flush=True)
    return {"run": run, "requested": len(manifest["cases"]), "poses": poses,
            "atom_audit": audits, "reference_hashes": references}


def recover_ca(atoms, peptide, original_cif):
    """Recover identities with unambiguous coordinate/element matches within 0.001 A."""
    site = CIFFile.read(original_cif).block["atom_site"]
    xyz = np.stack([site[k].as_array(float) for k in ("Cartn_x", "Cartn_y", "Cartn_z")], axis=-1)
    names = site["label_atom_id"].as_array()
    elements = np.char.upper(site["type_symbol"].as_array())
    chains = site["label_asym_id"].as_array()
    positions = site["label_seq_id"].as_array()
    residues = site["label_comp_id"].as_array()
    matches = cKDTree(xyz).query_ball_point(atoms.coord[peptide], r=0.001)
    mask, audit = [], []
    for index, candidates in zip(peptide, matches):
        candidates = [i for i in candidates if elements[i] == atoms.element[index].upper()]
        require(bool(candidates), f"Native peptide atom cannot be identified: {original_cif} atom {index}")
        identities = {(str(chains[i]), str(positions[i]), str(residues[i]), str(names[i])) for i in candidates}
        require(len(identities) == 1, f"Ambiguous native peptide atom: {original_cif} {identities}")
        best = min(candidates, key=lambda i: np.linalg.norm(xyz[i] - atoms.coord[index]))
        is_ca = names[best] == "CA" and elements[best] == "C"
        mask.append(is_ca)
        audit.append({
            "crop_atom_index": int(index), "source_atom_index": int(atoms.peptide_source_atom_index[index]),
            "original_atom_site_row": int(best), "chain": str(chains[best]),
            "position": str(positions[best]), "residue": str(residues[best]), "atom_name": str(names[best]),
            "is_ca": bool(is_ca), "match_distance": float(np.linalg.norm(xyz[best] - atoms.coord[index])),
        })
    require(any(mask), f"No identified peptide CA atoms: {original_cif}")
    return np.asarray(mask), audit


def training_ca_score(atoms, features, meta, predicted, ca_mask):
    receptor, receptor_copies = equivalent_chain_indices(atoms, meta["receptor_chain"])
    peptide, peptide_copies = equivalent_chain_indices(atoms, meta["peptide_chain"])
    protein = atoms.is_protein[receptor].astype(bool)
    anchors = (((atoms.atom_name[receptor] == "CA") & protein)
               | ((atoms.atom_name[receptor] == "C4'") & ~protein)) & atoms.is_resolved[receptor].astype(bool)
    perms = np.asarray([features["atom_perm_list"][i] for i in peptide], dtype=int).T
    ca_indices = np.flatnonzero(ca_mask)
    permutations = sorted({tuple(p[ca_indices]) for p in perms if set(p[ca_indices]) == set(ca_indices)})
    require(bool(permutations), "No CA-preserving peptide symmetry permutations")
    permutations = np.asarray(permutations)
    fixed = np.asarray(atoms.coord, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    best = None
    for receptor_name, receptor_ids in receptor_copies:
        rotation, translation, receptor_rmsd = receptor_fit(predicted[receptor_ids[anchors]], fixed[receptor[anchors]])
        for peptide_name, peptide_ids in peptide_copies:
            aligned = predicted[peptide_ids] @ rotation + translation
            errors = np.mean(np.sum((aligned[permutations] - fixed[peptide[ca_mask]]) ** 2, axis=-1), axis=-1)
            winner = int(np.argmin(errors))
            rmsd = float(np.sqrt(errors[winner]))
            candidate = {
                "peptide_ca_rmsd": rmsd, "receptor_rmsd": receptor_rmsd,
                "predicted_receptor_chain": receptor_name, "predicted_peptide_chain": peptide_name,
                "ca_atoms": int(ca_mask.sum()), "ca_symmetry_permutations": len(permutations),
                "selected_peptide_local_indices": permutations[winner].tolist(),
            }
            if best is None or receptor_rmsd < best["receptor_rmsd"] - 1e-6 or (
                abs(receptor_rmsd - best["receptor_rmsd"]) <= 1e-6 and rmsd < best["peptide_ca_rmsd"]
            ):
                best = candidate
    require(best is not None, "No valid training receptor/peptide correspondence")
    return best


def rescore_training(root, native_dir, inputs):
    from protenix.data.core.featurizer import Featurizer

    run = inputs.json(root / "run_config.json")
    manifest = inputs.json(root / "manifest.json", run["manifest_sha256"])
    seeds = run.get("seeds", [run.get("seed", 101)])
    expected = [(seed, sample) for seed in seeds for sample in range(run["samples"])]
    poses = {"before": [], "after": []}
    audits = {}
    for case in manifest["cases"]:
        name = case["name"]
        result = inputs.json(root / "cases" / f"{name}.json")
        if result["status"] != "complete":
            continue
        require(result["run_sha256"] == inputs.check(root / "run_config.json"), f"Run mismatch: {name}")
        meta = result["crop"]
        cache = Path(meta["cache_path"])
        cache = cache if cache.is_absolute() else REPO / cache
        inputs.check(cache, meta["cache_sha256"])
        with gzip.open(cache, "rb") as handle:
            full = pickle.load(handle)["atom_array"]
        inverse = {int(v): i for i, v in enumerate(full.peptide_source_atom_index)}
        selected = np.asarray(meta["source_atom_indices"])
        require(len(set(selected)) == len(selected), f"Duplicate cropped source atoms: {name}")
        atoms = full[[inverse[int(i)] for i in selected]]
        peptide, _ = equivalent_chain_indices(atoms, meta["peptide_chain"])
        native_cif = native_dir / f"{case['pdb_id']}.cif"
        inputs.check(native_cif)
        ca_mask, atom_audit = recover_ca(atoms, peptide, native_cif)
        audits[name] = {"source_cif": str(native_cif), "atoms": atom_audit}
        features = {"atom_perm_list": Featurizer(None, atoms).get_atom_permutation_list()}
        npz = root / "cases" / f"{name}.npz"
        inputs.check(npz, result["coordinates_sha256"])
        with np.load(npz) as coordinates:
            np.testing.assert_array_equal(coordinates["native"], atoms.coord)
            np.testing.assert_array_equal(coordinates["source_atom_indices"], selected)
            if "prediction_seeds" in coordinates:
                require(list(zip(coordinates["prediction_seeds"], coordinates["prediction_samples"])) == expected,
                        f"Seed/sample ordering mismatch: {name}")
            for label in poses:
                require(len(coordinates[label]) == len(result[label]["samples"]) == len(expected),
                        f"Incomplete prediction pool: {name} {label}")
                for xyz, old, (seed, sample) in zip(coordinates[label], result[label]["samples"], expected):
                    require((old.get("seed", seeds[0]), old["sample"]) == (seed, sample), f"Order mismatch: {name}")
                    heavy = score_interface(atoms, features, meta, xyz)
                    error = max(abs(heavy[k] - old[k]) for k in ("peptide_rmsd", "receptor_rmsd"))
                    require(error < 1e-6, f"Original heavy-atom metric not reproduced: {name} {error}")
                    score = training_ca_score(atoms, features, meta, xyz, ca_mask)
                    require(abs(score["receptor_rmsd"] - old["receptor_rmsd"]) <= 2e-6,
                            f"Receptor fit changed: {name}")
                    poses[label].append({
                        "name": name, "seed": seed, "sample": sample, "ranking_score": old["ranking_score"],
                        **score, "original_peptide_heavy_rmsd": old["peptide_rmsd"],
                        "original_heavy_metric_recomputation_error": error,
                    })
        print(f"Training: {len(audits)}/{len(manifest['cases'])} {name} ({int(ca_mask.sum())} CA)", flush=True)
    return {"run": run, "requested": len(manifest["cases"]), "poses": poses, "atom_audit": audits}


def compare(before, after):
    grouped = {}
    for label, rows in (("before", before), ("after", after)):
        pools = defaultdict(list)
        for row in rows:
            pools[row["name"]].append(row)
        grouped[label] = {name: select(pool) for name, pool in pools.items()}
    shared = sorted(set(grouped["before"]) & set(grouped["after"]))
    require(bool(shared), "No shared successful cases")
    summary = {"shared_successful_cases": len(shared)}
    per_case = []
    for label in grouped:
        summary[label] = {}
        for selection in ("top1", "best"):
            values = [grouped[label][name][selection]["peptide_ca_rmsd"] for name in shared]
            summary[label][selection] = {
                "mean": float(np.mean(values)), "median": float(np.median(values)),
                "success_counts": {str(t): sum(v <= t for v in values) for t in (2, 5)},
            }
    for name in shared:
        row = {"name": name}
        for label in grouped:
            for selection in ("top1", "best"):
                pose = grouped[label][name][selection]
                for field in ("peptide_ca_rmsd", "receptor_rmsd", "seed", "sample"):
                    row[f"{label}_{selection}_{field}"] = pose[field]
        row["top1_change"] = row["after_top1_peptide_ca_rmsd"] - row["before_top1_peptide_ca_rmsd"]
        per_case.append(row)
    summary["top1_improved"] = sum(r["top1_change"] < -1e-8 for r in per_case)
    summary["top1_worsened"] = sum(r["top1_change"] > 1e-8 for r in per_case)
    summary["top1_tied"] = len(shared) - summary["top1_improved"] - summary["top1_worsened"]
    summary["excluded_before_only"] = sorted(set(grouped["before"]) - set(shared))
    summary["excluded_after_only"] = sorted(set(grouped["after"]) - set(shared))
    return summary, per_case


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def report(summary):
    lines = [
        "Peptide Cα RMSD recalculation from saved October 6 evaluation predictions.", "",
        "All RMSDs are in Å. Peptides are measured after receptor-only rigid alignment; "
        "there is no independent peptide superposition. Only observed carbon atoms named CA are used.", "",
        "Top1 uses the original confidence ranking. Best-of-5 is reselected by Cα RMSD. "
        "Test complexes use the maximum across peptide chains; training uses the indexed peptide. "
        "Equivalent chain matching follows the original receptor-first rule with Cα RMSD breaking ties. "
        "Training atom symmetries must preserve the recovered Cα atom set.", "",
    ]
    for key, title in (("test", "Test set"), ("training", "Training subset")):
        data = summary[key]
        n = data["shared_successful_cases"]
        lines += [f"{title}: {n} shared successful cases.", "",
                  "| Peptide Cα RMSD | Official mean / median | Finetuned mean / median |",
                  "|---|---|---|"]
        for selection in ("top1", "best"):
            a, b = data["before"][selection], data["after"][selection]
            lines.append(f"| {selection} | {a['mean']:.2f} / {a['median']:.2f} | "
                         f"{b['mean']:.2f} / {b['median']:.2f} |")
        lines += ["", "| Success | Official | Finetuned |", "|---|---|---|"]
        for selection in ("top1", "best"):
            for t in ("2", "5"):
                a = data["before"][selection]["success_counts"][t]
                b = data["after"][selection]["success_counts"][t]
                lines.append(f"| {selection} ≤ {t} Å | {a}/{n} ({100*a/n:.1f}%) | {b}/{n} ({100*b/n:.1f}%) |")
        lines += ["", f"Top1: {data['top1_improved']} improved, {data['top1_worsened']} worsened, "
                  f"{data['top1_tied']} tied.", ""]
    lines += [
        "Training atom names were recovered from original PDB CIFs by native coordinate and element "
        "matching within 0.001 Å. Atom mappings and matching distances are recorded in training.json. "
        "Original heavy-atom training RMSDs were reproduced to verify saved coordinate correspondence.", "",
        "Residues/caps without an observed atom named CA contribute no atom to this metric. "
        "Test residue exclusions are recorded in test_before.json and test_after.json.", "",
        "Both test runs used MSA, seed 101, five samples, 200 diffusion steps, ten recycles, and no templates. "
        "The original baseline requested chunk64/dynamic chunking; finetuned used fixed chunk32. "
        "Test complexes are uncropped; training-subset crops contain up to 1024 tokens.", "",
        "No model inference was run. Original evaluation files were not changed.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--baseline-test", type=Path, required=True)
    parser.add_argument("--native-cif-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "Use a new output directory")
    args.output_dir.mkdir(parents=True)
    inputs = Inputs()
    inputs.check(__file__)
    for source in ("peptide_pose_metrics.py", "peptide_pose_inputs.py", "evaluate_peptide_training.py"):
        inputs.check(REPO / "scripts/evaluation" / source)
    before = rescore_test(args.baseline_test, inputs)
    write_json(args.output_dir / "test_before.json", before)
    after = rescore_test(args.evaluation_root / "test51", inputs)
    write_json(args.output_dir / "test_after.json", after)
    for name in set(before["reference_hashes"]) & set(after["reference_hashes"]):
        require(before["reference_hashes"][name] == after["reference_hashes"][name], f"Reference mismatch: {name}")
    for key in ("seeds", "samples", "steps", "cycles", "msa_enabled", "templates_enabled"):
        require(before["run"][key] == after["run"][key], f"Unmatched test sampling settings: {key}")
    training = rescore_training(args.evaluation_root / "train_subset", args.native_cif_dir, inputs)
    write_json(args.output_dir / "training.json", training)
    test_summary, test_cases = compare(before["poses"], after["poses"])
    train_summary, train_cases = compare(training["poses"]["before"], training["poses"]["after"])
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_root": str(args.evaluation_root), "baseline_test": str(args.baseline_test),
        "checkpoint": after["run"]["checkpoint"], "test": test_summary, "training": train_summary,
        "test_predictions_per_model": len(after["poses"]),
        "training_predictions_per_model": len(training["poses"]["after"]),
    }
    write_csv(args.output_dir / "test_per_case.csv", test_cases)
    write_csv(args.output_dir / "training_per_case.csv", train_cases)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "input_hashes.json", inputs.hashes)
    (args.output_dir / "comparison.md").write_text(report(summary))
    shutil.copyfile(__file__, args.output_dir / "recalculate_peptide_ca_rmsd.py")
    print(report(summary), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
