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

"""Receptor-aligned peptide pose RMSD, chain assignment, and residue contacts."""

import itertools
import math
from collections import defaultdict
from functools import lru_cache

import numpy as np
from biotite.structure.io.pdbx import CIFFile
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from scripts.evaluation.peptide_pose_inputs import atom_rows


def receptor_fit(moving, fixed):
    """Fit row-vector coordinates using receptor atoms only; disallow reflections."""
    moving, fixed = np.asarray(moving, dtype=float), np.asarray(fixed, dtype=float)
    if (
        moving.shape != fixed.shape
        or moving.ndim != 2
        or moving.shape[1] != 3
        or len(moving) < 3
    ):
        raise ValueError(
            "Receptor alignment requires at least three matched 3D coordinates"
        )
    if not np.isfinite(moving).all() or not np.isfinite(fixed).all():
        raise ValueError("Nonfinite receptor coordinates")
    a, b = moving.mean(axis=0), fixed.mean(axis=0)
    if min(np.linalg.matrix_rank(moving - a), np.linalg.matrix_rank(fixed - b)) < 2:
        raise ValueError("Receptor alignment atoms are collinear")
    u, _, vt = np.linalg.svd((moving - a).T @ (fixed - b))
    sign = np.eye(3)
    sign[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ sign @ vt
    translation = b - a @ rotation
    rmsd = np.sqrt(
        np.mean(np.sum((moving @ rotation + translation - fixed) ** 2, axis=-1))
    )
    return rotation, translation, float(rmsd)


@lru_cache(maxsize=None)
def residue_permutations(resname, names, fixed_names):
    """CCD graph automorphisms closed on observed atoms and fixing external bonds."""
    from protenix.data.core import ccd

    identity = tuple(range(len(names)))
    info = ccd.get_ccd_ref_info(resname, return_atomic_number=True)
    if not info or "perm" not in info:
        return (identity,)
    atom_map = info["atom_map"]
    if any(name not in atom_map for name in names):
        return (identity,)
    indices = [atom_map[name] for name in names]
    # Normal CCD permutations include hydrogen rows; the sanitization fallback
    # contains only heavy-atom rows. The first column need not be the identity.
    numbers = np.asarray(info["atomic_number"])
    row_indices = np.arange(len(numbers))
    if info["perm"].shape[0] != len(numbers):
        row_indices = np.flatnonzero(numbers != 1)
    if info["perm"].shape[0] != len(row_indices):
        return (identity,)
    row_lookup = {int(index): row for row, index in enumerate(row_indices)}
    if any(index not in row_lookup for index in indices):
        return (identity,)
    current = {index: i for i, index in enumerate(indices)}
    perms = {identity}
    for column in info["perm"][[row_lookup[i] for i in indices]].T:
        if set(column.tolist()) != set(indices):
            continue
        perm = tuple(current[int(index)] for index in column)
        if all(perm[i] == i for i, name in enumerate(names) if name in fixed_names):
            perms.add(perm)
    return tuple(sorted(perms))


def permutation_mse(moving, fixed, permutations):
    """Compare coordinates in the receptor frame, without any ligand superposition."""
    best = None
    for perm in permutations:
        perm = np.asarray(perm, dtype=int)
        candidate = moving[perm]
        mse = float(np.sum((candidate - fixed) ** 2))
        if best is None or mse < best[0]:
            best = (mse, candidate)
    return best


def peptide_coordinates(
    rows, prediction, predicted_chain, rotation, translation, fixed_atoms, symmetry
):
    fixed = np.asarray([row[5:8] for row in rows], dtype=float)
    keys = [(predicted_chain, row[1], row[3]) for row in rows]
    missing = [key for key in keys if key not in prediction]
    if missing:
        raise ValueError(f"Missing peptide atoms ({len(missing)}): {missing[:8]}")
    moving = np.asarray([prediction[key] for key in keys]) @ rotation + translation
    if not np.isfinite(moving).all() or not np.isfinite(fixed).all():
        raise ValueError("Nonfinite peptide coordinates")
    corrected = moving.copy()
    if symmetry:
        residues = defaultdict(list)
        for i, row in enumerate(rows):
            residues[(row[1], row[2])].append(i)
        for (position, resname), indices in residues.items():
            names = tuple(rows[i][3] for i in indices)
            anchors = tuple(fixed_atoms.get(f"{rows[0][0]}:{position}", []))
            # N and C participate in peptide bonds even if absent from struct_conn.
            perms = residue_permutations(
                resname, names, tuple(sorted(set(anchors) | {"N", "C"}))
            )
            _, corrected[indices] = permutation_mse(
                moving[indices], fixed[indices], perms
            )
    raw = float(np.sqrt(np.mean(np.sum((moving - fixed) ** 2, axis=-1))))
    rmsd = float(np.sqrt(np.mean(np.sum((corrected - fixed) ** 2, axis=-1))))
    backbone = np.asarray([row[3] in {"N", "CA", "C", "O"} for row in rows])
    backbone_rmsd = None
    if backbone.any():
        backbone_rmsd = float(
            np.sqrt(np.mean(np.sum((moving[backbone] - fixed[backbone]) ** 2, axis=-1)))
        )
    return {
        "pose_rmsd": rmsd,
        "atom_name_rmsd": raw,
        "backbone_pose_rmsd": backbone_rmsd,
        "native": fixed,
        "predicted": corrected,
        "n_atoms": len(rows),
    }


def residue_contacts(
    receptor_coords, receptor_keys, peptide_coords, peptide_keys, cutoff
):
    tree = cKDTree(receptor_coords)
    contacts = set()
    for i, neighbors in enumerate(tree.query_ball_point(peptide_coords, r=cutoff)):
        for j in neighbors:
            contacts.add((*receptor_keys[j], peptide_keys[i]))
    return contacts


def score_pose(
    reference, prediction, contact_cutoff=5.0, symmetry=True, min_receptor_coverage=0.9
):
    """Score every peptide with one consistent receptor/peptide chain assignment."""
    if reference.get("prediction_atom_mapping"):
        from scripts.evaluation.peptide_training_aligned import restore_prediction_keys

        prediction = restore_prediction_keys(prediction, reference)
    chains = reference["chains"]
    receptor_rows = [
        r for r in reference["atoms"] if chains[r[0]]["role"] == "receptor"
    ]
    anchors = [r for r in receptor_rows if r[3] == "CA"]
    if len(anchors) < 3:
        raise ValueError("Reference has fewer than three receptor CA atoms")
    groups = defaultdict(list)
    for chain, info in chains.items():
        if info["role"] == "receptor":
            groups[info["group"]].append(chain)
    group_list = list(groups.values())
    n_mappings = math.prod(math.factorial(len(g)) for g in group_list)
    if n_mappings > 4096:
        raise ValueError(
            f"Too many equivalent receptor assignments ({n_mappings}); explicit mapping required"
        )
    peptide_groups = defaultdict(list)
    for chain in reference["peptide_chains"]:
        peptide_groups[chains[chain]["group"]].append(chain)
    peptide_rows = {
        chain: [r for r in reference["atoms"] if r[0] == chain]
        for chain in reference["peptide_chains"]
    }
    best = None
    errors = []
    choices = [itertools.permutations(g) for g in group_list]
    for perm_groups in itertools.product(*choices):
        mapping = dict(
            itertools.chain.from_iterable(
                zip(g, p) for g, p in zip(group_list, perm_groups)
            )
        )
        anchor_pairs = [(r, (mapping[r[0]], r[1], r[3])) for r in anchors]
        anchor_pairs = [(r, key) for r, key in anchor_pairs if key in prediction]
        coverage = len(anchor_pairs) / len(anchors)
        if coverage < min_receptor_coverage:
            errors.append(
                f"Receptor CA coverage {coverage:.3f} < {min_receptor_coverage}"
            )
            continue
        try:
            rotation, translation, receptor_rmsd = receptor_fit(
                [prediction[key] for _, key in anchor_pairs],
                [r[5:8] for r, _ in anchor_pairs],
            )
            assignments = []
            for group in peptide_groups.values():
                options = {}
                cost = np.full((len(group), len(group)), np.inf)
                for i, native_chain in enumerate(group):
                    for j, predicted_chain in enumerate(group):
                        try:
                            value = peptide_coordinates(
                                peptide_rows[native_chain],
                                prediction,
                                predicted_chain,
                                rotation,
                                translation,
                                reference["fixed_atoms"],
                                symmetry,
                            )
                            options[i, j] = value
                            cost[i, j] = value["pose_rmsd"] ** 2
                        except ValueError as exc:
                            errors.append(str(exc))
                rows, columns = linear_sum_assignment(cost)
                for i, j in zip(rows, columns):
                    assignments.append((group[i], group[j], options[i, j]))
            score = float(np.mean([v["pose_rmsd"] ** 2 for _, _, v in assignments]))
            # Choose receptor correspondence from receptor geometry. A tiny
            # numerical improvement on the peptide must never select a poor
            # receptor fit. Peptides break ties between equivalent fits only.
            if (
                best is None
                or receptor_rmsd < best[1] - 1e-6
                or (abs(receptor_rmsd - best[1]) <= 1e-6 and score < best[0])
            ):
                best = (
                    score,
                    receptor_rmsd,
                    mapping,
                    assignments,
                    rotation,
                    translation,
                    coverage,
                )
        except ValueError as exc:
            errors.append(str(exc))
    if best is None:
        raise ValueError(
            "No valid chain/atom assignment: " + "; ".join(dict.fromkeys(errors))[:1500]
        )
    _, receptor_rmsd, mapping, assignments, rotation, translation, coverage = best
    matched = [(r, (mapping[r[0]], r[1], r[3])) for r in receptor_rows]
    matched = [(r, key) for r, key in matched if key in prediction]
    native_receptor = np.asarray([r[5:8] for r, _ in matched])
    predicted_receptor = (
        np.asarray([prediction[key] for _, key in matched]) @ rotation + translation
    )
    receptor_keys = [(r[0], r[1]) for r, _ in matched]
    output = []
    for native_chain, predicted_chain, value in assignments:
        peptide_keys = [r[1] for r in peptide_rows[native_chain]]
        native_contacts = residue_contacts(
            native_receptor,
            receptor_keys,
            value["native"],
            peptide_keys,
            contact_cutoff,
        )
        predicted_contacts = residue_contacts(
            predicted_receptor,
            receptor_keys,
            value["predicted"],
            peptide_keys,
            contact_cutoff,
        )
        common = len(native_contacts & predicted_contacts)
        output.append(
            {
                "peptide_chain": native_chain,
                "predicted_peptide_chain": predicted_chain,
                **{k: v for k, v in value.items() if k not in {"native", "predicted"}},
                "native_contacts": len(native_contacts),
                "predicted_contacts": len(predicted_contacts),
                "fnat": common / len(native_contacts) if native_contacts else None,
                "contact_precision": common / len(predicted_contacts)
                if predicted_contacts
                else 0.0,
            }
        )
    return {
        "receptor_rmsd": receptor_rmsd,
        "receptor_ca_coverage": coverage,
        "receptor_contact_atom_coverage": len(matched) / len(receptor_rows),
        "receptor_mapping": mapping,
        "peptides": output,
        "mean_peptide_rmsd": float(np.mean([r["pose_rmsd"] for r in output])),
        "max_peptide_rmsd": float(max(r["pose_rmsd"] for r in output)),
    }


def read_prediction(path):
    result = {}
    for atom in atom_rows(CIFFile.read(path).block):
        position = atom["position"]
        if position.startswith("auth:"):
            position = position.split(":")[1]
        key = (atom["chain"], int(position), atom["atom"])
        if key in result:
            raise ValueError(f"Duplicate prediction atom: {key}")
        result[key] = np.asarray(atom["coord"], dtype=float)
    return result
