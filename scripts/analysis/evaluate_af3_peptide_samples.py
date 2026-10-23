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

"""Evaluate paired original/ligand AF3 samples with the native epitope CA protocol."""

import argparse
import copy
import json
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from biotite.structure.io.pdbx import CIFFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.evaluate_af3_peptide_top1 import normalize_ca, save, write_crop, write_csv
from scripts.analysis.recalculate_peptide_ca_rmsd import ca_reference
from scripts.analysis.recalculate_peptide_epitope_rmsd import epitope_reference
from scripts.evaluation.peptide_pose_inputs import atom_rows, file_hash
from scripts.evaluation.peptide_pose_metrics import receptor_fit, score_pose
from scripts.evaluation.peptide_training_aligned import epitope_summary


def alias_original(block, ligands):
    """Alias source peptide chain IDs, preserving residue identifiers and coordinates."""
    result = copy.deepcopy(block)
    aliases = {r["source_label_asym_id"]: r["id"] for r in ligands}
    if len(aliases) != len(ligands) or len(set(aliases.values())) != len(ligands):
        raise ValueError("Non-bijective peptide chain aliases")
    for category, column in (("atom_site", "label_asym_id"), ("struct_asym", "id")):
        if category in result:
            values = result[category][column].as_array(str)
            result[category][column] = [aliases.get(v, v) for v in values]
    return result


def ca_lookup(block):
    rows = [r for r in atom_rows(block) if r["atom"] == "CA" and r["element"] == "C"]
    lookup = {(r["chain"], r["position"], r["atom"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate CA identifiers")
    signature = sorted((*key, row["resname"]) for key, row in lookup.items())
    return lookup, signature


def mapping_routes(reference, audits, lookup):
    """Cache explicit identity routes; subsequent samples must have identical topology."""
    routes = {}
    receptor_map = {}
    for row in audits:
        chain, index, _, atom = row["reference"]
        source = tuple(row["af3_source"][:3])
        routes[chain, int(index), atom] = source
        if reference["chains"][chain]["role"] == "receptor":
            receptor_map[chain] = source[0]
    for chain, source_chain in receptor_map.items():
        for key in lookup:
            if key[0] == source_chain:
                routes[chain, int(key[1]), key[2]] = key
    return routes


def export_selection(output, mode, name, selection, reference, reduced, prediction, score):
    mapping = dict(score["receptor_mapping"])
    mapping.update({r["peptide_chain"]: r["predicted_peptide_chain"] for r in score["peptides"]})
    rows = reduced["atoms"]
    anchors = [r for r in rows if r[0] not in reference["peptide_chains"]]
    rotation, translation, _ = receptor_fit(
        [prediction[mapping[r[0]], r[1], r[3]] for r in anchors], [r[5:8] for r in anchors]
    )
    folder = output / mode / "top_best" / name
    native = write_crop(folder / "reference.cif", rows, [r[5:8] for r in rows], reference["peptide_chains"])
    fitted = write_crop(
        folder / f"{selection}.cif", rows,
        [prediction[mapping[r[0]], r[1], r[3]] @ rotation + translation for r in rows],
        reference["peptide_chains"],
    )
    independent = max(float(np.sqrt(np.mean([
        np.sum((fitted[k] - native[k]) ** 2) for k in native if k[0] == chain
    ]))) for chain in reference["peptide_chains"])
    difference = abs(independent - score["max_peptide_rmsd"])
    if difference > 1e-6:
        raise ValueError(f"Export RMSD differs: {name}/{selection}: {difference}")
    return {"rotation": rotation.tolist(), "translation": translation.tolist(),
            "exported_rmsd": independent, "export_difference": difference}


def evaluate_case(task):
    root, reference_run, output, case, records = task
    name = case["name"]
    ref_path = reference_run / case["reference"]
    assert file_hash(ref_path) == case["artifacts"][case["reference"]]
    reference = json.loads(ref_path.read_text())
    native_path = Path(reference["source"])
    assert file_hash(native_path) == reference["source_sha256"]
    native = CIFFile.read(native_path).block
    reduced, epitope_audit = epitope_reference(reference, 10.0)
    reduced, _ = ca_reference(reduced)
    reduced.pop("prediction_atom_mapping", None)
    caches, results, selected = {}, {"ligand": [], "original": []}, {}
    max_coordinate_difference = 0.0
    for record in sorted(records, key=lambda r: r["sample_id"]):
        match = re.fullmatch(r"seed-(\d+)_sample-(\d+)", record["sample_id"])
        if not match:
            raise ValueError(f"Invalid sample ID: {record['sample_id']}")
        seed, sample = map(int, match.groups())
        pair = {}
        for mode in results:
            path = root / mode / name / record["sample_id"] / f"{name}_{record['sample_id']}_model.cif"
            block = CIFFile.read(path).block
            if mode == "original":
                block = alias_original(block, record["ligands"])
            lookup, signature = ca_lookup(block)
            if mode not in caches:
                normalized, audits = normalize_ca(reference, native, block)
                routes = mapping_routes(reference, audits, lookup)
                caches[mode] = signature, routes
                assert set(routes) == set(normalized)
                assert all(np.array_equal(normalized[k], lookup[v]["coord"]) for k, v in routes.items())
                save(output / mode / "audits" / f"{name}.json", {
                    "reference_sha256": file_hash(ref_path), "native_sha256": file_hash(native_path),
                    "mappings": audits, "epitope": epitope_audit,
                    "original_chain_aliases": record["ligands"] if mode == "original" else [],
                    "ca_topology": signature,
                })
            old_signature, routes = caches[mode]
            if signature != old_signature:
                raise ValueError(f"CA topology changed: {path}")
            prediction = {k: np.asarray(lookup[v]["coord"], dtype=float) for k, v in routes.items()}
            pair[mode] = prediction
            score = score_pose(reduced, prediction, symmetry=False, min_receptor_coverage=0.9)
            row = {"name": name, "seed": seed, "sample": sample, "sample_id": record["sample_id"],
                   "ranking_score": float(record["ranking_score"]), "source": str(path),
                   "source_sha256": file_hash(path), "epitope_ca_rmsd": score["max_peptide_rmsd"],
                   "receptor_ca_coverage": score["receptor_ca_coverage"]}
            results[mode].append(row)
            for selection, key in (("top", (-row["ranking_score"], seed, sample)),
                                   ("best", (row["epitope_ca_rmsd"], seed, sample))):
                previous = selected.get((mode, selection))
                if previous is None or key < previous[0]:
                    selected[mode, selection] = key, prediction, score, row
        if set(pair["ligand"]) != set(pair["original"]):
            raise ValueError(f"Representation CA key sets differ: {name}")
        max_coordinate_difference = max(max_coordinate_difference, max(
            float(np.max(np.abs(pair["ligand"][k] - pair["original"][k]))) for k in pair["ligand"]
        ))
    exported = {}
    for (mode, selection), (_, prediction, score, row) in selected.items():
        audit = export_selection(output, mode, name, selection, reference, reduced, prediction, score)
        save(output / mode / "top_best" / name / f"{selection}.json", {**row, **audit, "score": score})
        exported[f"{mode}_{selection}"] = audit["export_difference"]
    save(output / "per_case_details" / f"{name}.json", results)
    print(f"{name}: {len(records)} poses/mode; best {min(r['epitope_ca_rmsd'] for r in results['ligand']):.4f}",
          flush=True)
    return {"name": name, "poses": results, "max_coordinate_difference": max_coordinate_difference,
            "export_differences": exported}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--reference-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.reference_run / "manifest.json").read_text())
    records = json.loads((args.predictions / "manifest.json").read_text())
    cases = manifest["cases"]
    grouped = {c["name"]: [r for r in records if r["complex_id"] == c["name"]] for c in cases}
    assert len(cases) == 51 and len(records) == 1275
    assert {r["complex_id"] for r in records} == set(grouped)
    for rows in grouped.values():
        assert len(rows) == len({r["sample_id"] for r in rows}) == 25
        samples = [re.fullmatch(r"seed-(\d+)_sample-(\d+)", r["sample_id"]).groups() for r in rows]
        assert len({s for s, _ in samples}) == 5
        assert all({int(i) for s, i in samples if s == seed} == set(range(5)) for seed, _ in samples)
    for mode in ("ligand", "original"):
        actual = set((args.predictions / mode).rglob("*.cif"))
        expected = {args.predictions / mode / r["complex_id"] / r["sample_id"] /
                    f"{r['complex_id']}_{r['sample_id']}_model.cif" for r in records}
        assert actual == expected
    provenance = {"input": str(args.predictions), "reference_run": str(args.reference_run),
                  "started_utc": datetime.now(timezone.utc).isoformat(),
                  "input_manifest_sha256": file_hash(args.predictions / "manifest.json"),
                  "reference_manifest_sha256": file_hash(args.reference_run / "manifest.json"),
                  "code_sha256": {str(p): file_hash(p) for p in (
                      Path(__file__), Path(__file__).with_name("evaluate_af3_peptide_top1.py"),
                      Path(__file__).with_name("recalculate_peptide_ca_rmsd.py"),
                      Path(__file__).with_name("recalculate_peptide_epitope_rmsd.py"),
                      Path(__file__).parents[1] / "evaluation/peptide_pose_metrics.py")}}
    save(args.output / "provenance.json", provenance)
    for path in provenance["code_sha256"]:
        shutil.copy2(path, args.output / Path(path).name)
    tasks = [(args.predictions, args.reference_run, args.output, c, grouped[c["name"]]) for c in cases]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(evaluate_case, tasks):
            results.append(result)
            save(args.output / "progress.json", {"complete_cases": len(results), "total_cases": len(cases)})
    comparisons, per_case = [], []
    for mode in ("ligand", "original"):
        poses = [p for r in results for p in r["poses"][mode]]
        summary = epitope_summary(cases, poses, 25)
        assert summary["complete_cases"] == 51
        save(args.output / mode / "summary.json", summary)
        write_csv(args.output / mode / "poses.csv", poses)
        comparisons.append({"Model": f"AF3 {mode}", **summary["table"]})
        for c in summary["cases"]:
            per_case.append({"mode": mode, "name": c["name"], "top_rmsd": c["top"]["epitope_ca_rmsd"],
                             "best_rmsd": c["best"]["epitope_ca_rmsd"],
                             "top_sample": c["top"]["sample_id"], "best_sample": c["best"]["sample_id"]})
    for checkpoint in (7999, 8499, 8999, 9499):
        path = args.reference_run.parent / f"peptide_eval_training_aligned_{checkpoint}_20261022"
        summary = json.loads((path / "scores/epitope_ca_summary.json").read_text())
        assert {c["name"] for c in summary["cases"]} == set(grouped)
        comparisons.append({"Model": f"Protenix {checkpoint}", **summary["table"]})
    write_csv(args.output / "comparison.csv", comparisons)
    write_csv(args.output / "per_case.csv", per_case)
    validation = {"complete_cases": len(results), "poses_per_mode": 1275,
                  "max_representation_coordinate_difference": max(r["max_coordinate_difference"] for r in results),
                  "max_export_rmsd_difference": max(v for r in results for v in r["export_differences"].values()),
                  "finished_utc": datetime.now(timezone.utc).isoformat()}
    save(args.output / "validation.json", validation)
    (args.output / "README.txt").write_text(
        "51 cases, five seeds and five samples per seed (25 poses/case) for each representation.\n"
        "original contains residue annotations; ligand is a conversion of the original AF3 predictions.\n"
        "Top: maximum manifest ranking_score; best: minimum epitope CA RMSD. Ties: ascending seed/sample.\n"
        "Native receptor CA within <=10 A of any observed native peptide CA define the fit.\n"
        "No peptide superposition. Multiple peptides: maximum chain RMSD. Strict <2 A and <5 A.\n"
        "Mapping uses source chain annotations and original native residue numbering, never prediction geometry.\n"
        "PHE/NFA (7yvb) and 7MD/ASP (8fsq) chemistry differences are audited; corresponding CA are included.\n"
        "Both representations scored independently, with every normalized CA coordinate compared.\n"
        "top_best contains CA-only native and fitted top/best CIFs, with transformations and source hashes.\n"
        "These crops are already aligned; do not superpose them again to reproduce the reported RMSD.\n"
        "Original predictions and ongoing evaluation output were not modified.\n"
    )
    print(json.dumps({"validation": validation, "comparison": comparisons}, indent=2), flush=True)


if __name__ == "__main__":
    main()
