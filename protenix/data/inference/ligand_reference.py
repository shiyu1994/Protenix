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

"""Apply explicitly supplied single-residue ligand reference conformers."""

import biotite.structure as struc
import numpy as np


def apply_ligand_references(atoms, references, entity_poly_type):
    """Replace selected ligand chains before token/atom annotations are derived.

    References contain independent conformers, never target coordinates. Original
    residue/atom keys identify the selected heavy atoms and preserve their order.
    """
    selected = {}
    for chain, ref in references.items():
        chain_ids = np.flatnonzero(atoms.chain_id == chain)
        if not len(chain_ids):
            raise ValueError(f"Unknown reference ligand chain: {chain}")
        if any(entity_poly_type.get(str(e), "non-polymer") != "non-polymer"
               for e in atoms.label_entity_id[chain_ids]):
            raise ValueError(f"Cannot override a polymer reference: {chain}")
        lookup = {(int(atoms.res_id[i]), str(atoms.atom_name[i])): int(i) for i in chain_ids}
        keys = [tuple(k) for k in ref["source_keys"]]
        if len(set(keys)) != len(keys) or not keys or any(k not in lookup for k in keys):
            raise ValueError(f"Invalid reference atom mapping: {chain}")
        ids = np.array([lookup[k] for k in keys])
        xyz = np.asarray(ref["ref_pos"], dtype=float)
        names = ref["atom_names"]
        if (xyz.shape != (len(ids), 3) or not np.isfinite(xyz).all()
                or len(names) != len(ids) or len(set(names)) != len(names)
                or any(not n or len(n) > 4 for n in names)):
            raise ValueError(f"Invalid reference conformer: {chain}")
        if atoms.element[ids].tolist() != ref["elements"] or len(ref["charges"]) != len(ids):
            raise ValueError(f"Reference chemistry does not match atoms: {chain}")
        selected[chain] = ids

    # Keep non-target chains intact; preserve the supplied within-peptide order.
    order = []
    for chain in dict.fromkeys(atoms.chain_id):
        order.extend(selected.get(chain, np.flatnonzero(atoms.chain_id == chain)))
    result = atoms[np.asarray(order)]
    for chain, ref in references.items():
        ids = np.flatnonzero(result.chain_id == chain)
        bonds = result.bonds.as_array()
        inside_a, inside_b = np.isin(bonds[:, 0], ids), np.isin(bonds[:, 1], ids)
        # Keep experimentally declared attachments to other chains. Only the
        # within-peptide graph is replaced; no attachment is silently removed.
        kept = bonds[~(inside_a & inside_b)]
        local = np.asarray(ref["bonds"], dtype=int).reshape(-1, 3)
        if len(local) and (local[:, :2].min() < 0 or local[:, :2].max() >= len(ids)):
            raise ValueError(f"Invalid reference bonds: {chain}")
        mapped = local.copy()
        mapped[:, :2] = ids[local[:, :2]]
        result.bonds = struc.BondList(len(result), np.concatenate([kept, mapped]))
        result.res_id[ids] = 1
        result.label_seq_id[ids] = 1
        result.res_name[ids] = ref["res_name"]
        result.atom_name[ids] = ref["atom_names"]
        result.coord[ids] = ref["ref_pos"]
        result.ref_pos[ids] = ref["ref_pos"]
        result.ref_charge[ids] = ref["charges"]
        result.ref_mask[ids] = True
        result.hetero[ids] = True
        if "charge" in result.get_annotation_categories():
            result.charge[ids] = ref["charges"]
    return result
