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

"""Export observed references and pose crops in the exact evaluation frame."""

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from biotite.structure.io.pdbx import CIFBlock, CIFCategory, CIFFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.recalculate_peptide_ca_rmsd import ca_reference
from scripts.analysis.recalculate_peptide_epitope_rmsd import epitope_reference
from scripts.evaluation.evaluate_peptide_pose import digest, prediction_paths
from scripts.evaluation.peptide_pose_inputs import category_rows, file_hash
from scripts.evaluation.peptide_pose_metrics import read_prediction, receptor_fit, score_pose
from scripts.evaluation.peptide_training_aligned import restore_prediction_keys


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def set_rows(block, category, rows):
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        block[category] = CIFCategory({k: [str(r.get(k, "?")) for r in rows] for k in keys})


def key(row):
    return row["label_asym_id"], int(row["label_seq_id"]), row["label_atom_id"]


def original_key(row):
    return row[0], int(row[1]), row[3]


def model_key(atom_key, reference):
    chain, position, name = atom_key
    for entry in reference.get("prediction_atom_mapping", {}).get(chain, []):
        if entry["original"] == [position, name]:
            return chain, *entry["model"]
    if chain in reference.get("prediction_atom_mapping", {}):
        raise ValueError(f"Missing peptide atom mapping: {atom_key}")
    return atom_key


def custom_chemistry(job):
    """Use actual local peptide chemistry, never the unrelated CCD X00 entry."""
    components, atoms, bonds = [], [], []
    for spec in job["ligand_reference_overrides"].values():
        comp = spec["res_name"]
        components.append({"id": comp, "type": "NON-POLYMER", "name": "custom whole-peptide ligand"})
        for name, element, charge in zip(spec["atom_names"], spec["elements"], spec["charges"]):
            atoms.append({"comp_id": comp, "atom_id": name, "type_symbol": element, "charge": charge})
        for a, b, order in spec["bonds"]:
            bonds.append({"comp_id": comp, "atom_id_1": spec["atom_names"][a],
                          "atom_id_2": spec["atom_names"][b],
                          "value_order": {1: "SING", 2: "DOUB", 3: "TRIP"}[order]})
    return components, atoms, bonds


def write_cif(path, rows, template, chemistry, reference, ca_keys, connections):
    """Write consistent label/auth identities and only bonds with present endpoints."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cif = CIFFile()
    block = CIFBlock()
    cif[reference["name"].replace("-", "_")] = block
    rows = [dict(r, id=str(i)) for i, r in enumerate(rows, 1)]
    set_rows(block, "atom_site", rows)
    entities = {r["label_entity_id"] for r in rows}
    comps = {r["label_comp_id"] for r in rows}
    chains = dict.fromkeys(r["label_asym_id"] for r in rows)
    for category in ("entity", "entity_poly", "entity_poly_seq"):
        field = "id" if category == "entity" else "entity_id"
        subset = [r.copy() for r in category_rows(template, category) if r[field] in entities]
        if category == "entity_poly":
            for r in subset:
                r["pdbx_strand_id"] = ",".join(c for c in chains if any(
                    a["label_asym_id"] == c and a["label_entity_id"] == r[field] for a in rows))
        set_rows(block, category, subset)
    set_rows(block, "struct_asym", [{"id": c, "entity_id": next(
        r["label_entity_id"] for r in rows if r["label_asym_id"] == c)} for c in chains])
    custom, chem_atoms, chem_bonds = chemistry
    custom_ids = {r["id"] for r in custom}
    set_rows(block, "chem_comp", [r for r in category_rows(template, "chem_comp")
                                  if r["id"] in comps and r["id"] not in custom_ids]
             + [r for r in custom if r["id"] in comps])
    present = {(r["label_comp_id"], r["label_atom_id"]) for r in rows}
    set_rows(block, "chem_comp_atom", [r for r in chem_atoms if (r["comp_id"], r["atom_id"]) in present])
    set_rows(block, "chem_comp_bond", [r for r in chem_bonds
             if (r["comp_id"], r["atom_id_1"]) in present and (r["comp_id"], r["atom_id_2"]) in present])
    lookup = {key(r): r for r in rows}
    links = []
    for a, b in connections:
        if a not in lookup or b not in lookup:
            continue
        link = {"id": f"covale{len(links) + 1}", "conn_type_id": "covale"}
        for i, atom_key in enumerate((a, b), 1):
            r = lookup[atom_key]
            for field in ("asym_id", "comp_id", "seq_id", "atom_id"):
                link[f"ptnr{i}_label_{field}"] = r[f"label_{field}"]
        links.append(link)
    set_rows(block, "struct_conn", links)
    reverse = {model_key(original_key(r), reference): r for r in reference["atoms"]}
    metadata = []
    for r in rows:
        original = reverse[key(r)]
        role = reference["chains"][original[0]]["role"]
        metadata.append({"atom_site_id": r["id"], "original_chain": original[0],
                         "original_residue_index": original[1], "original_residue_name": original[2],
                         "original_atom_name": original[3], "role": role,
                         "used_for_fit": "yes" if original_key(original) in ca_keys and role == "receptor" else "no",
                         "used_for_rmsd": "yes" if original_key(original) in ca_keys and role == "peptide" else "no"})
    set_rows(block, "inspection_atom_mapping", metadata)
    cif.write(path)
    reread = read_prediction(path)
    assert len(reread) == len(rows) == len(lookup), "Duplicate/missing exported atoms"
    for r in rows:
        np.testing.assert_allclose(reread[key(r)], [float(r[c]) for c in ("Cartn_x", "Cartn_y", "Cartn_z")],
                                   atol=1e-8, rtol=0)
    return reread


def coordinates(row, xyz):
    return dict(row, **{c: f"{float(v):.8f}" for c, v in zip(("Cartn_x", "Cartn_y", "Cartn_z"), xyz)})


def export_case(run_root, output, case, run, progress, baseline):
    name = case["name"]
    record = progress[name]
    assert record["status"] == "complete" and record["run_sha256"] == digest(run)
    ref_path = run_root / case["reference"]
    assert file_hash(ref_path) == case["artifacts"][case["reference"]]
    reference = json.loads(ref_path.read_text())
    input_path = run_root / case["input"]
    assert file_hash(input_path) == case["artifacts"][case["input"]]
    job = json.loads(input_path.read_text())[0]
    chemistry = custom_chemistry(job)
    local, epitope_audit = epitope_reference(reference, 10.0)
    ca, ca_audit = ca_reference(local)
    ca_keys = {original_key(r) for r in ca["atoms"]}
    epitope_residues = {(r[0], r[1]) for r in ca["atoms"] if r[0] not in reference["peptide_chains"]}
    context = [r for r in reference["atoms"] if r[0] in reference["peptide_chains"]
               or (r[0], r[1]) in epitope_residues]
    selections = {"metric_ca": ca["atoms"], "epitope_context": context}
    template_path, _ = prediction_paths(run_root, name, run["seeds"][0], 0)
    template = CIFFile.read(template_path).block
    template_rows = {key(r): r for r in category_rows(template, "atom_site")}
    native_rows = {}
    for r in reference["atoms"]:
        k = model_key(original_key(r), reference)
        base = dict(template_rows[k], B_iso_or_equiv="0.0", occupancy="1.0")
        native_rows[original_key(r)] = coordinates(base, r[5:8])
    # Covalent links between entities are taken from the validated inference job.
    entity_copies = {(info["entity"], info["copy"]): chain for chain, info in reference["chains"].items()}
    connections = []
    for bond in job.get("covalent_bonds", []):
        ends = []
        for i in (1, 2):
            c = entity_copies[bond[f"entity{i}"], bond[f"copy{i}"]]
            ends.append(model_key((c, int(bond[f"position{i}"]), bond[f"atom{i}"]), reference))
        connections.append(tuple(ends))
    write_cif(output / "references_full" / f"{name}.cif", list(native_rows.values()),
              template, chemistry, reference, ca_keys, connections)
    for kind, selected in selections.items():
        write_cif(output / kind / "references" / f"{name}.cif",
                  [native_rows[original_key(r)] for r in selected],
                  template, chemistry, reference, ca_keys, connections)
    save_json(output / "mappings" / f"{name}.json", {
        "prediction_atom_mapping": reference["prediction_atom_mapping"],
        "epitope": epitope_audit, "peptide_ca": ca_audit,
        "reference_sha256": file_hash(ref_path), "omitted_water_atoms": reference["omitted_water_atoms"],
    })
    reports = []
    for seed in run["seeds"]:
        for sample in range(run["samples"]):
            source, confidence = prediction_paths(run_root, name, seed, sample)
            for path in (source, confidence):
                assert file_hash(path) == record["files"][str(path.relative_to(run_root))]
            raw = read_prediction(source)
            restored = restore_prediction_keys(raw, reference)
            scored = score_pose(ca, raw, symmetry=False)
            saved = baseline[name, seed, sample]
            np.testing.assert_allclose(scored["max_peptide_rmsd"], float(saved["epitope_ca_rmsd"]),
                                       atol=1e-7, rtol=0)
            mapping = dict(scored["receptor_mapping"])
            mapping.update({r["peptide_chain"]: r["predicted_peptide_chain"] for r in scored["peptides"]})
            anchors = [r for r in ca["atoms"] if r[0] not in reference["peptide_chains"]
                       and (mapping[r[0]], r[1], r[3]) in restored]
            rotation, translation, fit_rmsd = receptor_fit(
                [restored[mapping[r[0]], r[1], r[3]] for r in anchors], [r[5:8] for r in anchors])
            np.testing.assert_allclose(fit_rmsd, scored["receptor_rmsd"], atol=1e-8, rtol=0)
            source_rows = {key(r): r for r in category_rows(CIFFile.read(source).block, "atom_site")}
            stem = f"seed_{seed}_sample_{sample}"
            written = {}
            for kind, selected in selections.items():
                rows = []
                for r in selected:
                    source_key = (mapping[r[0]], r[1], r[3])
                    if source_key not in restored:
                        continue
                    source_model = model_key(source_key, reference)
                    row = dict(native_rows[original_key(r)],
                               B_iso_or_equiv=source_rows[source_model]["B_iso_or_equiv"])
                    rows.append(coordinates(row, restored[source_key] @ rotation + translation))
                written[kind] = write_cif(output / kind / "predictions" / name / f"{stem}.cif", rows,
                                          template, chemistry, reference, ca_keys, connections)
            per_chain = {}
            for chain in reference["peptide_chains"]:
                rows = [r for r in ca["atoms"] if r[0] == chain]
                delta = np.asarray([written["metric_ca"][model_key(original_key(r), reference)]
                                    - np.asarray(r[5:8]) for r in rows])
                per_chain[chain] = float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1))))
            actual = max(per_chain.values())
            assert abs(actual - scored["max_peptide_rmsd"]) < 1e-6
            audit = {"name": name, "seed": seed, "sample": sample, "source": str(source),
                     "source_sha256": record["files"][str(source.relative_to(run_root))],
                     "native_to_prediction_chain_mapping": mapping,
                     "rotation_row_vectors": rotation.tolist(), "translation": translation.tolist(),
                     "receptor_fit_rmsd": fit_rmsd, "per_peptide_rmsd": per_chain,
                     "saved_rmsd": float(saved["epitope_ca_rmsd"]), "exported_rmsd": actual,
                     "ranking_score": float(saved["ranking_score"])}
            save_json(output / "transforms" / name / f"{stem}.json", audit)
            reports.append({k: v for k, v in audit.items() if not isinstance(v, (dict, list))})
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="*", help="Optional smoke-test case names")
    args = parser.parse_args()
    root, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    run = json.loads((root / "run_config.json").read_text())
    assert file_hash(root / "manifest.json") == run["manifest_sha256"]
    manifest = json.loads((root / "manifest.json").read_text())
    progress = json.loads((root / "prediction_manifest.json").read_text())
    with (root / "scores/poses.csv").open() as handle:
        baseline = {(r["name"], int(r["seed"]), int(r["sample"])): r for r in csv.DictReader(handle)}
    cases = [c for c in manifest["cases"] if not args.cases or c["name"] in args.cases]
    output.mkdir(parents=True)
    save_json(output / "provenance.json", {"run": run, "source_run": str(root),
              "script_sha256": file_hash(Path(__file__)), "started_utc": datetime.now(timezone.utc).isoformat(),
              "baseline_sha256": file_hash(root / "scores/poses.csv")})
    shutil.copy2(__file__, output / "export_peptide_inspection_cifs.py")
    reports = []
    for i, case in enumerate(cases, 1):
        reports.extend(export_case(root, output, case, run, progress, baseline))
        save_json(output / "progress.json", {"cases": i, "requested": len(cases), "poses": len(reports)})
        print(f"Exported {i}/{len(cases)}: {case['name']}", flush=True)
    with (output / "poses.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    summary = json.loads((root / "scores/epitope_ca_summary.json").read_text())
    for case in summary["cases"]:
        name = case["name"]
        if name not in {c["name"] for c in cases}:
            continue
        folder = output / "top_best" / name
        folder.mkdir(parents=True)
        targets = {"reference": output / "epitope_context/references" / f"{name}.cif"}
        for kind in ("top", "best"):
            pose = case[kind]
            targets[kind] = output / "epitope_context/predictions" / name / (
                f"seed_{pose['seed']}_sample_{pose['sample']}.cif")
        for kind, target in targets.items():
            (folder / f"{kind}.cif").symlink_to(os.path.relpath(target, folder))
        (folder / "view.pml").write_text("\n".join([
            f"load {folder / 'reference.cif'}, reference", f"load {folder / 'top.cif'}, top",
            f"load {folder / 'best.cif'}, best", "hide everything", "show sticks",
            "color gray70, reference", "color cyan, top", "color magenta, best", "zoom", ""]))
    expected = len(cases) * len(run["seeds"]) * run["samples"]
    assert len(reports) == expected
    counts = {d: len(list((output / d).rglob('*.cif'))) for d in
              ("references_full", "metric_ca/references", "metric_ca/predictions",
               "epitope_context/references", "epitope_context/predictions")}
    assert list(counts.values()) == [len(cases), len(cases), expected, len(cases), expected]
    (output / "README.txt").write_text(
        "Training-aligned peptide inspection CIFs\n\n"
        f"Source evaluation: {root}\n"
        "references_full/: observed evaluation-reference heavy atoms, in native coordinates.\n"
        "Water is omitted just as in evaluation; other observed cofactors are retained.\n"
        "No missing atoms are invented. Reference occupancy=1 and B-factor=0 are placeholders.\n"
        "Peptides use the prediction's X00/X01/etc names, residue index 1 and generated atom names.\n"
        "The actual custom peptide atom/bond dictionaries replace unrelated global CCD X00 metadata.\n\n"
        "metric_ca/references/ and metric_ca/predictions/: exactly the scoring atoms: native receptor\n"
        "CA within <=10 A of any observed native peptide CA, plus observed peptide CA atoms.\n"
        "epitope_context/references/ and epitope_context/predictions/: all observed atoms of those\n"
        "same receptor residues, plus all observed peptide atoms. Extra side chains are for viewing\n"
        "only; they are NOT included in the reported CA RMSD. Cofactors and water are absent in crops.\n\n"
        "Every prediction crop is already rigidly superposed on its corresponding native reference.\n"
        "Only receptor CA determine the fit. Peptides are never independently superposed.\n"
        "Equivalent receptor/peptide copies use the original scorer's assignment; prediction crop\n"
        "chain/atom labels are remapped to reference identities for direct overlay. The original\n"
        "assignment and rotation/translation are recorded under transforms/. No inference is run.\n"
        "Peptide atoms corresponding to CA keep generated Cxxx names, NOT the literal name CA.\n"
        "Use inspection_atom_mapping in each CIF, or mappings/<case>.json, to identify them.\n"
        "For multiple peptides, compute each chain's CA RMSD without further fitting and take\n"
        "the maximum (not a pooled average). This is the evaluation's best/top metric.\n\n"
        "Prediction filenames: <case>/seed_101_sample_0.cif; samples are zero-based.\n"
        "top_best/<case>/ contains reference.cif, top.cif and best.cif symlinks plus view.pml.\n"
        "Open view.pml in PyMOL to overlay reference (gray), top (cyan), best (magenta).\n"
        "Do NOT run another align/super command: it would change the evaluation frame.\n"
        "For metric_ca files, use spheres/nonbonded display; CA-only ligand crops lack most bonds.\n"
        "Top is maximum original ranking_score across 25 poses; best is minimum peptide CA RMSD.\n\n"
        "poses.csv records all saved/exported RMSDs. validation.json records counts and maximum\n"
        "numerical difference after reading exported CIFs back. provenance.json and the copied\n"
        "export script document inputs. Original evaluation files are unchanged.\n"
    )
    save_json(output / "validation.json", {"status": "complete", "cases": len(cases), "poses": expected,
              "counts": counts, "max_rmsd_difference_angstrom": max(
                  abs(r["saved_rmsd"] - r["exported_rmsd"]) for r in reports),
              "all_input_hashes_verified": True, "all_exported_cifs_read_back": True})
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
