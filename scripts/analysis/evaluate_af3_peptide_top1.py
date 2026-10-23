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

"""Score supplied AF3 top1 files with the existing peptide CA/epitope protocol."""

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from biotite.structure.io.pdbx import CIFBlock, CIFCategory, CIFFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.recalculate_peptide_ca_rmsd import ca_reference
from scripts.analysis.recalculate_peptide_epitope_rmsd import epitope_reference
from scripts.evaluation.peptide_pose_inputs import atom_rows, category_rows, file_hash
from scripts.evaluation.peptide_pose_metrics import read_prediction, receptor_fit, score_pose


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path, rows):
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_ca(reference, native_block, af3_block):
    """Recover original indices by native-native coordinate matching, never pose geometry."""
    native_atoms = atom_rows(native_block)
    predicted_atoms = atom_rows(af3_block)
    annotations = {r["id"]: r for r in category_rows(native_block, "foldbench_peptide_ligand")}
    pred_lookup = {(r["chain"], r["position"], r["atom"]): r for r in predicted_atoms}
    assert len(pred_lookup) == len(predicted_atoms), "Duplicate AF3 atom identifiers"
    # AF3 input chains may have been assigned new IDs (e.g. B-2 -> C).
    # Match complete receptor sequences, then let the existing scorer enumerate
    # equivalent copies. No peptide RMSD is used to assign sequence identities.
    sequences = {}
    for r in category_rows(af3_block, "entity_poly_seq"):
        sequences.setdefault(r["entity_id"], {})[int(r["num"])] = r["mon_id"]
    canonical = lambda names: tuple("MET" if n == "MSE" else n for n in names)
    predicted_sequences = {
        r["id"]: canonical([v for _, v in sorted(sequences[r["entity_id"]].items())])
        for r in category_rows(af3_block, "struct_asym") if r["entity_id"] in sequences
    }
    receptor_map, used_chains = {}, set()
    for chain, info in reference["chains"].items():
        if info["role"] != "receptor":
            continue
        candidates = sorted(c for c, seq in predicted_sequences.items()
                            if c not in used_chains and seq == canonical(info["residues"]))
        if not candidates:
            raise ValueError(f"No exact full-sequence receptor counterpart for {chain}")
        chosen = chain if chain in candidates else candidates[0]
        receptor_map[chain] = chosen
        used_chains.add(chosen)
    mapped, audits = {}, []
    # Retain all receptor CA atoms, including residues absent from a particular
    # native copy: other equivalent copies can need them during permutation fits.
    for chain, predicted_chain in receptor_map.items():
        for r in predicted_atoms:
            if r["chain"] == predicted_chain and r["atom"] == "CA" and r["element"] == "C":
                mapped[chain, int(r["position"]), "CA"] = np.asarray(r["coord"], dtype=float)
    for row in reference["atoms"]:
        chain, index, residue, atom, element = row[:5]
        if atom != "CA" or element != "C":
            continue
        if reference["chains"][chain]["role"] not in {"receptor", "peptide"}:
            continue
        matches = [r for r in native_atoms if r["chain"] == chain and r["atom"] == atom
                   and r["element"] == element and np.linalg.norm(np.asarray(r["coord"]) - row[5:8]) < 1e-5]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous native atom correspondence: {chain}/{index}/{atom}")
        source = matches[0]
        source_key = (chain, source["position"], atom)
        method = "original_native_label_seq_id"
        if chain in receptor_map:
            source_key = (receptor_map[chain], str(index), atom)
            method = "exact_receptor_sequence_and_sequence_index"
        elif source["position"].startswith("auth:"):
            # The native conversion can merge nonpolymer attachments into LIG1.
            # These source-chain IDs are explicitly recorded in the native CIF.
            original_chains = annotations[chain]["source_label_asym_ids"].split(",")
            candidates = [r for r in predicted_atoms if r["chain"] in original_chains
                          and r["atom"] == atom and r["resname"] == source["resname"]]
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous/missing appended ligand atom: {chain}/{index}: {candidates}")
            candidate = candidates[0]
            source_key = (candidate["chain"], candidate["position"], atom)
            method = "native_annotation_unique_appended_ligand"
        if source_key not in pred_lookup:
            raise ValueError(f"AF3 lacks required CA: {source_key}")
        predicted = pred_lookup[source_key]
        if predicted["element"] != element:
            raise ValueError(f"Element mismatch: {source_key}")
        mismatch = residue != predicted["resname"]
        # These are explicitly reported molecular-input differences, not hidden
        # sequence substitutions. Only their named carbon CA atoms are compared.
        if mismatch and (residue, predicted["resname"]) not in {("PHE", "NFA"), ("7MD", "ASP")}:
            raise ValueError(f"Unexpected residue mismatch: {chain}/{index}: {residue}/{predicted['resname']}")
        mapped[chain, int(index), atom] = np.asarray(predicted["coord"], dtype=float)
        audits.append({"reference": [chain, index, residue, atom],
                       "native_source": [chain, source["position"], source["resname"], atom],
                       "af3_source": [*source_key, predicted["resname"]], "method": method,
                       "chemistry_mismatch": mismatch})
    used = [tuple(r["af3_source"][:3]) for r in audits]
    if len(used) != len(set(used)):
        raise ValueError("An AF3 CA atom was assigned to more than one native atom")
    return mapped, audits


def write_crop(path, rows, coords, peptide_chains):
    columns = {"group_PDB": [], "id": [], "type_symbol": [], "label_atom_id": [],
               "label_comp_id": [], "label_asym_id": [], "label_entity_id": [],
               "label_seq_id": [], "auth_asym_id": [], "auth_seq_id": [],
               "auth_comp_id": [], "auth_atom_id": [], "label_alt_id": [],
               "occupancy": [], "B_iso_or_equiv": [], "pdbx_PDB_model_num": [],
               "Cartn_x": [], "Cartn_y": [], "Cartn_z": []}
    chains = list(dict.fromkeys(r[0] for r in rows))
    for i, (r, xyz) in enumerate(zip(rows, coords), 1):
        values = ["HETATM" if r[0] in peptide_chains else "ATOM", i, r[4], r[3], r[2], r[0],
                  chains.index(r[0]) + 1, r[1], r[0], r[1], r[2], r[3], ".", 1, 0, 1,
                  *[f"{v:.8f}" for v in xyz]]
        for k, v in zip(columns, values):
            columns[k].append(str(v))
    cif = CIFFile()
    block = CIFBlock()
    cif["epitope_ca"] = block
    block["atom_site"] = CIFCategory(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    cif.write(path)
    return read_prediction(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    shutil.copy2(__file__, args.output / Path(__file__).name)
    manifest = json.loads((args.reference_run / "manifest.json").read_text())
    expected = {c["name"] + "_model.cif" for c in manifest["cases"]}
    assert {p.name for p in args.predictions.glob("*.cif")} == expected
    save(args.output / "provenance.json", {
        "predictions": str(args.predictions), "reference_run": str(args.reference_run),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": file_hash(args.reference_run / "manifest.json"),
        "script_sha256": file_hash(Path(__file__)),
        "selection": "supplied top1 files; ranking provenance and other samples not provided",
        "scorer_sha256": file_hash(Path(__file__).parents[1] / "evaluation/peptide_pose_metrics.py"),
        "epitope_sha256": file_hash(Path(__file__).with_name("recalculate_peptide_epitope_rmsd.py")),
        "ca_reference_sha256": file_hash(Path(__file__).with_name("recalculate_peptide_ca_rmsd.py")),
    })
    results, errors, all_audits = [], [], {}
    for case in manifest["cases"]:
        name = case["name"]
        try:
            ref_path = args.reference_run / case["reference"]
            assert file_hash(ref_path) == case["artifacts"][case["reference"]]
            ref = json.loads(ref_path.read_text())
            native = Path(ref["source"])
            assert file_hash(native) == ref["source_sha256"]
            prediction = args.predictions / (name + "_model.cif")
            pred, mappings = normalize_ca(ref, CIFFile.read(native).block, CIFFile.read(prediction).block)
            reduced, epitope_audit = epitope_reference(ref, 10.0)
            reduced, peptide_audit = ca_reference(reduced)
            reduced.pop("prediction_atom_mapping", None)  # AF3 uses native residue atom names, not Cxxx.
            score = score_pose(reduced, pred, symmetry=False, min_receptor_coverage=0.9)
            mapping = dict(score["receptor_mapping"])
            mapping.update({r["peptide_chain"]: r["predicted_peptide_chain"] for r in score["peptides"]})
            anchors = [r for r in reduced["atoms"] if r[0] not in ref["peptide_chains"]]
            rotation, translation, _ = receptor_fit(
                [pred[mapping[r[0]], r[1], r[3]] for r in anchors], [r[5:8] for r in anchors])
            rows = reduced["atoms"]
            native_crop = write_crop(args.output / "metric_ca/references" / (name + ".cif"), rows,
                                     [r[5:8] for r in rows], ref["peptide_chains"])
            pred_crop = write_crop(args.output / "metric_ca/predictions" / (name + ".cif"), rows,
                [pred[mapping[r[0]], r[1], r[3]] @ rotation + translation for r in rows], ref["peptide_chains"])
            independent = max(float(np.sqrt(np.mean([
                np.sum((pred_crop[k] - native_crop[k]) ** 2) for k in native_crop if k[0] == chain])))
                for chain in ref["peptide_chains"])
            assert abs(independent - score["max_peptide_rmsd"]) < 1e-6
            audit = {"name": name, "source_sha256": file_hash(prediction),
                     "native_source_sha256": file_hash(native), "reference_sha256": file_hash(ref_path),
                     "mappings": mappings, "epitope": epitope_audit, "peptide_ca": peptide_audit,
                     "score": score, "rotation": rotation.tolist(), "translation": translation.tolist(),
                     "independent_exported_rmsd": independent}
            all_audits[name] = audit
            save(args.output / "audits" / (name + ".json"), audit)
            results.append({"name": name, "top_rmsd": score["max_peptide_rmsd"],
                            "top_lt2": score["max_peptide_rmsd"] < 2,
                            "top_lt5": score["max_peptide_rmsd"] < 5,
                            "receptor_ca_coverage": score["receptor_ca_coverage"],
                            "renumbered_ca_atoms": sum(str(r["reference"][1]) != r["native_source"][1]
                                                       for r in mappings),
                            "appended_ligand_ca_atoms": sum(r["method"].endswith("appended_ligand") for r in mappings),
                            "chemistry_mismatch_ca_atoms": sum(r["chemistry_mismatch"] for r in mappings)})
            print(name, f"{score['max_peptide_rmsd']:.4f}", flush=True)
        except Exception as exc:
            errors.append({"name": name, "error": repr(exc)})
            print("ERROR", name, repr(exc), flush=True)
    save(args.output / "errors.json", errors)
    if results:
        write_csv(args.output / "per_case.csv", results)
    n = len(manifest["cases"])
    vals = [r["top_rmsd"] for r in results]
    table = {"Top < 2A": f"{sum(v < 2 for v in vals)}/{n} ({100 * sum(v < 2 for v in vals) / n:.1f}%)",
             "Top < 5A": f"{sum(v < 5 for v in vals)}/{n} ({100 * sum(v < 5 for v in vals) / n:.1f}%)",
             "Best < 2A": "N/A", "Best < 5A": "N/A",
             "Top Mean RMSD": float(np.mean(vals)), "Top Median RMSD": float(np.median(vals)),
             "Best Mean RMSD": "N/A", "Best Median RMSD": "N/A"}
    summary = {"cases_requested": n, "cases_scored": len(results), "errors": len(errors), "table": table,
               "protocol": "native receptor CA <=10 A from native peptide CA; fit receptor only; max peptide CA RMSD",
               "best_metrics": "Unavailable across original AF3 sampling; singleton best equals supplied top only",
               "chemistry_mismatch_cases": [r["name"] for r in results if r["chemistry_mismatch_ca_atoms"]],
               "max_export_rmsd_difference": max(abs(a["independent_exported_rmsd"] - a["score"]["max_peptide_rmsd"])
                                                 for a in all_audits.values())}
    save(args.output / "summary.json", summary)
    write_csv(args.output / "summary.csv", [table])
    comparisons = []
    for checkpoint in (7999, 8499, 8999, 9499):
        root = args.reference_run.parent / f"peptide_eval_training_aligned_{checkpoint}_20261022"
        previous = json.loads((root / "scores/epitope_ca_summary.json").read_text())
        comparisons.append({"Model": str(checkpoint), **previous["table"]})
    comparisons.append({"Model": "AF3 supplied top1", **table})
    write_csv(args.output / "comparison.csv", comparisons)
    (args.output / "README.txt").write_text(
        "AF3 supplied top1 evaluation with the existing epitope/CA protocol.\n"
        "One CIF per case is supplied. The folder's top1 selection cannot be independently verified.\n"
        "Independent best-of-samples metrics require the other AF3 predictions and are N/A here.\n"
        "All comparisons use label identifiers. Original native coordinates identify which original\n"
        "sequence position corresponds to each prepared reference residue; AF3 pose geometry is\n"
        "never used to choose that numbering. Appended GLU ligands use native source-chain annotations.\n"
        "Audits explicitly flag modified-residue chemistry differences. CA correspondence is retained.\n"
        "metric_ca contains references and already aligned predictions; do not superpose again.\n"
        "For each peptide chain calculate direct CA RMSD, then take the maximum across peptides.\n"
        "per_case.csv, audits/, errors.json and summary.json record scores, mappings and coverage.\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
