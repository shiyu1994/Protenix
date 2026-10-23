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

"""Prepare observed peptide atoms using the training ligand representation."""

import numpy as np
import biotite.structure as struc
from rdkit import Chem

from protenix.data.inference.json_to_feature import SampleDictToFeatures
from scripts.prepare_peptide_training_data import make_reference


class SourceChemistry:
    """Resolve aromatic hydrogens from known CCD identities in the source map."""

    def __init__(self, rows):
        self.rows = rows

    def match(self, coord, element):
        matches = {
            (r[0], str(r[1]), r[2], r[3], "") for r in self.rows
            if r[4] == element and np.linalg.norm(np.asarray(r[5:8]) - coord) < 0.002
        }
        return next(iter(matches)) if len(matches) == 1 else None


def align_peptide_inputs(job, reference, seed=42):
    """Create whole-peptide references and retain original keys for scoring.

    Native coordinates are used solely for stereochemistry assignment by the
    same make_reference() routine as training, then discarded before embedding.
    Chemistry/topology comes from the existing validated CCD input construction.
    """
    atoms = SampleDictToFeatures(job).get_atom_array()
    lookup = {(str(a.chain_id), int(a.res_id), str(a.atom_name)): i for i, a in enumerate(atoms)}
    overrides, mappings, audits = {}, {}, {}
    for number, chain in enumerate(reference["peptide_chains"]):
        rows = [r for r in reference["atoms"] if r[0] == chain]
        ids = np.asarray([lookup[(r[0], r[1], r[3])] for r in rows])
        bonds = atoms.bonds.as_array()
        in_a, in_b = np.isin(bonds[:, 0], ids), np.isin(bonds[:, 1], ids)
        external = (atoms.chain_id[bonds[:, 0]] != chain) | (atoms.chain_id[bonds[:, 1]] != chain)
        external_count = int(np.sum((in_a != in_b) & external))
        peptide = atoms[ids]
        # CCD stores Kekule aromatic single/double orders as Biotite 5/6.
        # Training struct_conn uses ordinary SING/DOUB for the same graph.
        peptide_bonds = peptide.bonds.as_array().copy()
        for aromatic, ordinary in ((5, 1), (6, 2), (7, 3)):
            peptide_bonds[peptide_bonds[:, 2] == aromatic, 2] = ordinary
        peptide.bonds = struc.BondList(len(peptide), peptide_bonds)
        peptide.coord[:] = np.asarray([r[5:8] for r in rows])
        peptide.set_annotation("charge", peptide.ref_charge.astype(int))
        mol, names = make_reference(peptide, np.arange(len(peptide)), seed, SourceChemistry(rows))
        name = f"X{number:02d}"
        if len(name) > 3 or name in set(atoms.res_name):
            raise ValueError(f"Custom residue name collision: {name}")
        overrides[chain] = {
            "source_keys": [[int(r[1]), r[3]] for r in rows],
            "atom_names": names,
            "res_name": name,
            "elements": peptide.element.tolist(),
            "charges": [a.GetFormalCharge() for a in mol.GetAtoms()],
            "ref_pos": mol.GetConformer().GetPositions().tolist(),
            "bonds": peptide.bonds.as_array().astype(int).tolist(),
        }
        mappings[chain] = [
            {"model": [1, n], "original": [int(r[1]), r[3]]}
            for n, r in zip(names, rows)
        ]
        audits[chain] = {
            "atoms": len(peptide), "reference_groups": 1,
            "smiles": Chem.MolToSmiles(mol),
            "embedding_attempts": mol.embedding_attempts,
            "inferred_aromatic_hydrogen_atoms": mol.inferred_hydrogen_atoms,
            "omitted_unobserved_atoms": int(np.sum(atoms.chain_id == chain) - len(peptide)),
            "external_bonds_preserved": external_count,
            "training_exception": (
                "Training converter excludes external peptide attachments; evaluation retains them"
                if external_count else None
            ),
        }
    job["ligand_reference_overrides"] = overrides
    reference["prediction_atom_mapping"] = mappings
    reference["training_alignment"] = {
        "reference_seed": seed, "peptides": audits,
        "method": "training make_reference; observed atoms; one ligand residue per peptide",
        "chemistry": "validated CCD components and explicit connections",
        "native_coordinate_use": "stereochemistry assignment only; discarded before ETKDG",
    }
    reference["protocol"] += "; training-aligned whole-peptide references"
    return job, reference


def restore_prediction_keys(prediction, reference):
    """Map model atom identities back to native residue identities for scoring."""
    mappings = reference.get("prediction_atom_mapping", {})
    result = {k: v for k, v in prediction.items() if k[0] not in mappings}
    for chain, entries in mappings.items():
        for entry in entries:
            model = (chain, *entry["model"])
            target = (chain, *entry["original"])
            if model not in prediction:
                raise ValueError(f"Missing training-aligned prediction atom: {model}")
            if target in result:
                raise ValueError(f"Duplicate prediction atom mapping: {target}")
            result[target] = prediction[model]
    return result


def epitope_ca_score(reference, prediction, min_coverage=0.9):
    """Apply the existing 10-Angstrom native-epitope/C-alpha protocol."""
    from scripts.analysis.recalculate_peptide_ca_rmsd import ca_reference
    from scripts.analysis.recalculate_peptide_epitope_rmsd import epitope_reference
    from scripts.evaluation.peptide_pose_metrics import score_pose

    local, _ = epitope_reference(reference, 10.0)
    ca, _ = ca_reference(local)
    return score_pose(ca, prediction, symmetry=False, min_receptor_coverage=min_coverage)["max_peptide_rmsd"]


def epitope_summary(cases, poses, expected):
    """Summarize complete cases, using all requested cases for success rates."""
    selected = []
    for case in cases:
        rows = [p for p in poses if p["name"] == case["name"] and "epitope_ca_rmsd" in p]
        if len(rows) != expected:
            continue
        top = min(rows, key=lambda r: (-r["ranking_score"], r["seed"], r["sample"]))
        best = min(rows, key=lambda r: (r["epitope_ca_rmsd"], r["seed"], r["sample"]))
        selected.append({"name": case["name"], "top": top, "best": best})
    n = len(cases)
    values = {key: [r[key]["epitope_ca_rmsd"] for r in selected] for key in ("top", "best")}
    cells = {}
    for key in ("top", "best"):
        for cutoff in (2, 5):
            count = sum(v < cutoff for v in values[key])
            cells[f"{key.title()} < {cutoff}A"] = f"{count}/{n} ({100 * count / n:.1f}%)" if n else "0/0"
    for key in ("top", "best"):
        for title, function in (("Mean", np.mean), ("Median", np.median)):
            cells[f"{key.title()} {title} RMSD"] = float(function(values[key])) if values[key] else None
    return {
        "protocol": "peptide CA RMSD after native receptor CA epitope fit (10 A); max over peptides",
        "ranking": "maximum ranking_score; tie-break ascending seed/sample; best is minimum RMSD",
        "thresholds": "strict <; incomplete cases count as failures; mean/median use complete cases",
        "requested_cases": n, "complete_cases": len(selected), "table": cells, "cases": selected,
    }
