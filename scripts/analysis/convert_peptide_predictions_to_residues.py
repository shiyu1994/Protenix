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

"""Export ligand-form peptide predictions as polymers and rescore their saved coordinates.

Requires residue-resolved CCD input: it does not infer residues from a single
flattened ligand. Modified residues and terminal caps retain their CCD identities.
No coordinates, atom order, explicit bonds, or confidence values are changed.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np
from biotite.structure.io.pdbx import CIFCategory, CIFColumn, CIFFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.recalculate_peptide_ca_rmsd import ca_reference, require, select
from scripts.analysis.recalculate_peptide_epitope_rmsd import epitope_reference
from scripts.evaluation.peptide_pose_inputs import category_rows, write_json
from scripts.evaluation.peptide_pose_metrics import read_prediction, score_pose

STANDARD = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())


def set_rows(block, category, rows):
    """Replace a category while retaining every existing column."""
    keys = list(dict.fromkeys(key for row in rows for key in row))
    block[category] = CIFCategory({key: [row.get(key, "?") for row in rows] for key in keys})


def convert_prediction(source, destination, reference):
    """Promote explicitly annotated peptide chains to residue-based polymer entities."""
    source, destination = Path(source), Path(destination)
    require(not destination.exists(), f"Output already exists: {destination}")
    cif = CIFFile.read(source)
    block = cif.block
    atoms = category_rows(block, "atom_site")
    peptide_chains = set(reference["peptide_chains"])
    require(bool(peptide_chains), "No annotated peptide chains")
    chain_atoms = defaultdict(list)
    entity_chains = defaultdict(set)
    for atom in atoms:
        chain_atoms[atom["label_asym_id"]].append(atom)
        entity_chains[atom["label_entity_id"]].add(atom["label_asym_id"])

    entity_sequences = {}
    for chain in sorted(peptide_chains):
        require(chain in chain_atoms, f"Missing peptide chain: {chain}")
        rows = chain_atoms[chain]
        entities = {row["label_entity_id"] for row in rows}
        require(len(entities) == 1, f"Multiple entities in peptide chain: {chain}")
        entity = next(iter(entities))
        require(entity_chains[entity] <= peptide_chains, f"Peptide shares entity with non-peptide: {entity}")
        sequence = reference["chains"][chain]["residues"]
        observed = defaultdict(set)
        for row in rows:
            position = row["label_seq_id"]
            require(position not in {".", "?"}, f"Missing residue numbering in {chain}")
            observed[int(position)].add(row["label_comp_id"])
        require(set(observed) == set(range(1, len(sequence) + 1)), f"Incomplete residue mapping in {chain}")
        for position, residue in enumerate(sequence, 1):
            require(observed[position] == {residue}, f"Residue identity mismatch: {chain}/{position}")
        require(entity not in entity_sequences or entity_sequences[entity] == sequence,
                f"Inconsistent sequence in shared entity: {entity}")
        entity_sequences[entity] = sequence

    # Standard amino acids become ATOM records; modified residues/caps remain HETATM
    # members of a polymer, as in conventional residue-based mmCIF files.
    groups = [
        ("ATOM" if atom["label_comp_id"] in STANDARD else "HETATM")
        if atom["label_asym_id"] in peptide_chains else atom["group_PDB"]
        for atom in atoms
    ]
    block["atom_site"]["group_PDB"] = CIFColumn(groups)
    entities = category_rows(block, "entity")
    require(set(entity_sequences) <= {row["id"] for row in entities}, "Missing entity metadata")
    for row in entities:
        if row["id"] in entity_sequences:
            row["type"] = "polymer"
    set_rows(block, "entity", entities)

    polymers = [r for r in category_rows(block, "entity_poly") if r["entity_id"] not in entity_sequences]
    sequences = [r for r in category_rows(block, "entity_poly_seq") if r["entity_id"] not in entity_sequences]
    chemistry = {r["id"]: r.get("type", "").upper() for r in category_rows(block, "chem_comp")}
    for entity, sequence in entity_sequences.items():
        chiral = {chemistry.get(residue, "") for residue in sequence} & {"L-PEPTIDE LINKING", "D-PEPTIDE LINKING"}
        polymer_type = "polypeptide(D)" if chiral == {"D-PEPTIDE LINKING"} else "polypeptide(L)"
        polymers.append({
            "entity_id": entity,
            "type": polymer_type,
            "pdbx_strand_id": ",".join(sorted({
                row["auth_asym_id"] for chain in entity_chains[entity] for row in chain_atoms[chain]
            })),
        })
        sequences.extend({"entity_id": entity, "num": str(i), "mon_id": residue, "hetero": "n"}
                         for i, residue in enumerate(sequence, 1))
    set_rows(block, "entity_poly", polymers)
    set_rows(block, "entity_poly_seq", sequences)
    if "struct_asym" not in block:
        set_rows(block, "struct_asym", [
            {"id": chain, "entity_id": rows[0]["label_entity_id"]} for chain, rows in chain_atoms.items()
        ])
    # These exporters have no scheme tables. Refuse other formats rather than leave
    # stale nonpolymer/scheme metadata in a converted file.
    require(not any(k in block for k in ("pdbx_nonpoly_scheme", "pdbx_poly_seq_scheme", "pdbx_entity_nonpoly")),
            "Scheme metadata requires an explicit conversion")
    destination.parent.mkdir(parents=True, exist_ok=True)
    cif.write(destination)

    saved = CIFFile.read(destination).block
    after = category_rows(saved, "atom_site")
    require(len(atoms) == len(after), "Atom count changed")
    for before, current in zip(atoms, after):
        require({k: v for k, v in before.items() if k != "group_PDB"}
                == {k: v for k, v in current.items() if k != "group_PDB"},
                "Atom identities, coordinates, order or confidence changed")
        if before["label_asym_id"] not in peptide_chains:
            require(before == current, "Non-peptide atom changed")
    require(category_rows(block, "struct_conn") == category_rows(saved, "struct_conn"), "Bonds changed")
    return {"peptide_chains": len(peptide_chains), "peptide_residues": sum(
        len(reference["chains"][c]["residues"]) for c in peptide_chains
    ), "atoms_preserved": len(atoms)}


def statistics(rows):
    result = {"cases": len(rows)}
    for metric in ("top", "best"):
        values = [r[metric + "_rmsd"] for r in rows]
        result.update({metric + "_lt2": sum(v < 2 for v in values), metric + "_lt5": sum(v < 5 for v in values),
                       metric + "_mean": mean(values), metric + "_median": median(values)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Original test51 directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--baseline", type=Path, required=True, help="Original C-alpha/epitope results.json")
    args = parser.parse_args()
    root, output = args.run.resolve(), args.output.resolve()
    require(not output.exists(), f"Output already exists: {output}")
    run = json.loads((root / "run_config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    progress = json.loads((root / "prediction_manifest.json").read_text())
    baseline = json.loads(args.baseline.read_text())
    require(run == baseline["run"], "Baseline checkpoint/run mismatch")
    previous = {(p["name"], p["seed"], p["sample"]): p for p in baseline["poses"]}
    run_hash = hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()
    require(hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest() == run["manifest_sha256"],
            "Manifest hash mismatch")
    output.mkdir(parents=True)
    poses, cases, conversions, hashes = [], [], {}, {}

    def check(path, expected=None):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(expected is None or digest == expected, f"Hash mismatch: {path}")
        hashes[str(path)] = digest
        return digest

    check(args.baseline)
    for case in manifest["cases"]:
        name = case["name"]
        record = progress[name]
        require(case["status"] == "ready" and record["status"] == "complete", f"Incomplete case: {name}")
        require(record["run_sha256"] == run_hash, f"Run identity mismatch: {name}")
        refpath = root / case["reference"]
        check(refpath, case["artifacts"][case["reference"]])
        reference = json.loads(refpath.read_text())
        epitope, audit = epitope_reference(reference, 10.0)
        require(json.loads(json.dumps(audit)) == baseline["atom_audit"][name], f"Epitope changed: {name}")
        ca, _ = ca_reference(epitope)
        current = []
        for seed in run["seeds"]:
            for sample in range(run["samples"]):
                folder = Path("predictions") / name / f"seed_{seed}" / "predictions"
                relative = folder / f"{name}_sample_{sample}.cif"
                confidence = folder / f"{name}_summary_confidence_sample_{sample}.json"
                for rel in (relative, confidence):
                    check(root / rel, record["files"][str(rel)])
                    check(root / rel, baseline["input_hashes"][str(root / rel)])
                conversions[str(relative)] = convert_prediction(root / relative, output / relative, reference)
                shutil.copy2(root / confidence, output / confidence)
                prediction = read_prediction(output / relative)
                original = read_prediction(root / relative)
                require(prediction.keys() == original.keys(), "Prediction atom keys changed")
                require(all(np.array_equal(prediction[k], original[k]) for k in original), "Coordinates changed")
                score = score_pose(ca, prediction, symmetry=False, min_receptor_coverage=0.9)
                ranking = json.loads((output / confidence).read_text())["ranking_score"]
                old = previous[name, seed, sample]
                require(ranking == old["ranking_score"], "Ranking score changed")
                row = {"name": name, "seed": seed, "sample": sample, "ranking_score": ranking,
                       "peptide_ca_rmsd": score["max_peptide_rmsd"], "original_rmsd": old["peptide_ca_rmsd"],
                       "delta_rmsd": score["max_peptide_rmsd"] - old["peptide_ca_rmsd"],
                       "receptor_rmsd": score["receptor_rmsd"], "file": str(relative)}
                current.append(row)
                check(output / relative)
        poses.extend(current)
        selected = select(current)
        original_selected = select([previous[name, p["seed"], p["sample"]] for p in current])
        row = {"name": name, "poses": len(current)}
        for label, key in (("top", "top1"), ("best", "best")):
            chosen = selected[key]
            old = original_selected[key]
            row.update({label + "_rmsd": chosen["peptide_ca_rmsd"], label + "_seed": chosen["seed"],
                        label + "_sample": chosen["sample"], label + "_original_rmsd": old["peptide_ca_rmsd"],
                        label + "_selection_unchanged": (
                            chosen["seed"], chosen["sample"]
                        ) == (old["seed"], old["sample"])})
            target = output / "selected" / label / f"{name}.cif"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.path.relpath(output / chosen["file"], target.parent))
        cases.append(row)
        print(f"Converted and rescored {name}: {len(current)} poses", flush=True)
    require(len(poses) == len(previous), "Baseline and converted pose sets differ")
    report = {"run": run, "source_run": str(root), "baseline": str(args.baseline.resolve()),
              "protocol": {"metric": "peptide CA RMSD", "epitope_cutoff_angstrom": 10,
                           "alignment": "native receptor epitope CA only", "multi_peptide": "maximum chain RMSD",
                           "top_selector": "original ranking_score", "best_selector": "minimum RMSD",
                           "threshold_comparison": "strictly less than"},
              "statistics": statistics(cases), "poses": poses, "per_case": cases,
              "maximum_absolute_rmsd_change": max(abs(p["delta_rmsd"]) for p in poses),
              "all_top_and_best_selections_unchanged": all(c["top_selection_unchanged"] and
                                                          c["best_selection_unchanged"] for c in cases),
              "conversion_audit": conversions, "file_hashes": hashes}
    write_json(output / "results.json", report)
    write_json(output / "summary.json", {k: v for k, v in report.items()
                                        if k not in {"poses", "per_case", "conversion_audit", "file_hashes"}})
    for filename, rows in (("poses.csv", poses), ("per_case.csv", cases)):
        with (output / filename).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(report["statistics"], indent=2))
    print("Maximum RMSD change:", report["maximum_absolute_rmsd_change"])


if __name__ == "__main__":
    main()
