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

"""Repairs for exported peptide CIFs, without changing observed coordinates."""

from itertools import pairwise

import biotite.structure as struc
import numpy as np
from biotite.structure.io.pdbx import CIFCategory
from rdkit import Chem
from scipy.spatial import cKDTree

from protenix.data.constants import STD_RESIDUES
from protenix.data.core import ccd
from protenix.data.core.parser import DistillationMMCIFParser, MMCIFParser


class OriginalAtoms:
    """Match exported atoms to original CIF identities, never by proximity alone."""

    def __init__(self, path):
        self.parser = MMCIFParser(mmcif_file=path)
        site = self.parser.cif.block["atom_site"]
        self.columns = {key: site[key].as_array() for key in site}
        self.coords = np.column_stack(
            [site[f"Cartn_{axis}"].as_array(float) for axis in "xyz"]
        )
        self.tree = cKDTree(self.coords)

    def match(self, coord, element, res_name=None):
        """Require one chemical identity at the same position within CIF precision."""
        candidates = self.tree.query_ball_point(coord, 0.002)
        identities = set()
        for i in candidates:
            c = self.columns
            if c["type_symbol"][i].upper() != element.upper():
                continue
            if res_name is not None and c["label_comp_id"][i] != res_name:
                continue
            identities.add(
                tuple(
                    str(c[key][i])
                    for key in (
                        "label_asym_id",
                        "label_seq_id",
                        "label_comp_id",
                        "label_atom_id",
                        "label_entity_id",
                    )
                )
            )
        return next(iter(identities)) if len(identities) == 1 else None

    def polymer_type(self, columns, mask):
        """Recover a single original polymer type for a converted entity."""
        types = set()
        for i in np.flatnonzero(mask):
            match = self.match(
                [float(columns[f"Cartn_{axis}"][i]) for axis in "xyz"],
                str(columns["type_symbol"][i]),
                str(columns["label_comp_id"][i]),
            )
            if match:
                kind = self.parser.entity_poly_type.get(match[4])
                if kind:
                    types.add(kind)
        if len(types) != 1:
            raise ValueError(
                f"Cannot uniquely recover original polymer type: {sorted(types)}"
            )
        return types.pop()


def repair_atom_names(parser, original):
    """Restore entire affected residues, including valid-looking renamed collisions."""
    site = parser.cif.block["atom_site"]
    columns = {key: site[key].as_array().astype(object) for key in site}
    residues = {}
    for i in range(site.row_count):
        key = tuple(
            str(columns[k][i])
            for k in ("label_asym_id", "label_seq_id", "label_comp_id")
        )
        residues.setdefault(key, []).append(i)
    changes, replacements = [], {}
    for key, ids in residues.items():
        if key[2] == "PEP":
            continue
        reference = ccd.get_component_rdkit_mol(key[2])
        if reference is None:
            continue
        old_names = [str(columns["label_atom_id"][i]) for i in ids]
        if all(name in reference.atom_map for name in old_names):
            continue
        matched = [
            original.match(
                [float(columns[f"Cartn_{axis}"][i]) for axis in "xyz"],
                str(columns["type_symbol"][i]),
                key[2],
            )
            for i in ids
        ]
        if any(m is None for m in matched) or len({m[:3] for m in matched}) != 1:
            raise ValueError(
                f"Cannot uniquely restore CCD atom names for residue {key}"
            )
        new_names = [m[3] for m in matched]
        if len(set(new_names)) != len(new_names) or any(
            n not in reference.atom_map for n in new_names
        ):
            raise ValueError(
                f"Original atom names do not uniquely match CCD for residue {key}"
            )
        for i, old, new in zip(ids, old_names, new_names):
            replacements[(*key, old)] = new
            columns["label_atom_id"][i] = new
            columns["auth_atom_id"][i] = new
        changes.append(
            {"residue": list(key), "atom_names": dict(zip(old_names, new_names))}
        )
    if changes:
        parser.cif.block["atom_site"] = CIFCategory(
            {k: v.tolist() for k, v in columns.items()}
        )
        if "struct_conn" in parser.cif.block:
            conn = parser.cif.block["struct_conn"]
            cc = {k: conn[k].as_array().astype(object) for k in conn}
            for row in range(conn.row_count):
                for n in (1, 2):
                    key = tuple(
                        str(cc[f"ptnr{n}_label_{f}"][row])
                        for f in ("asym_id", "seq_id", "comp_id", "atom_id")
                    )
                    if key in replacements:
                        cc[f"ptnr{n}_label_atom_id"][row] = replacements[key]
            parser.cif.block["struct_conn"] = CIFCategory(
                {k: v.tolist() for k, v in cc.items()}
            )
    return changes


def repair_aromatic_hydrogens(mol, atoms, ids, original=None):
    """Recover omitted aromatic N hydrogens using valence and original CCD identity."""
    mol.UpdatePropertyCache(strict=False)
    inferred = set()
    for ring in Chem.GetSymmSSSR(mol):
        ring = list(ring)
        if not all(
            mol.GetBondBetweenAtoms(a, b).GetIsAromatic()
            for a, b in zip(ring, ring[1:] + ring[:1])
        ):
            continue
        single_n_pyrrole = sorted(
            mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring
        ) == [6, 6, 6, 6, 7]
        for i in ring:
            nitrogen = mol.GetAtomWithIdx(i)
            if nitrogen.GetAtomicNum() != 7 or nitrogen.GetDegree() != 2:
                continue
            charge = nitrogen.GetFormalCharge()
            # An unsubstituted aromatic [n+] needs an H; a neutral pyrrole does too.
            needs_h = charge == 1 or (charge == 0 and single_n_pyrrole)
            if charge == 0 and not single_n_pyrrole and original is not None:
                source = int(ids[i])
                match = original.match(atoms.coord[source], str(atoms.element[source]))
                if match:
                    template = ccd.get_component_rdkit_mol(match[2])
                    if template is not None and match[3] in template.atom_map:
                        atom = template.GetAtomWithIdx(template.atom_map[match[3]])
                        needs_h = (
                            atom.GetIsAromatic()
                            and atom.GetTotalNumHs(includeNeighbors=True) > 0
                        )
            if needs_h:
                nitrogen.SetNumExplicitHs(1)
                nitrogen.SetNoImplicit(True)
                inferred.add(i)
    return sorted(inferred)


def add_missing_representatives(atoms, polymer_types):
    """Add unresolved centre/distogram atoms only; keep every observed atom and bond."""
    atoms = atoms.copy()
    atoms.set_annotation("is_resolved", np.ones(len(atoms), dtype=bool))
    pieces, old_to_new, added_bonds = [], {}, []
    offset = 0
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    for start, stop in pairwise(starts):
        residue = atoms[start:stop]
        name = str(residue.res_name[0])
        required = []
        if residue.label_entity_id[0] in polymer_types and name in STD_RESIDUES:
            if len(name) == 3:
                required = ["CA"] if name == "GLY" else ["CA", "CB"]
            else:
                rep = "C4" if name in {"A", "G", "DA", "DG"} else "C2"
                required = ["C1'"] if name in {"N", "DN"} else ["C1'", rep]
        missing = [n for n in required if n not in residue.atom_name]
        old_to_new.update({i: offset + i - start for i in range(start, stop)})
        if missing:
            template = ccd.get_component_atom_array(name)
            if template is None:
                raise ValueError(f"No CCD for missing representatives of {name}")
            for atom_name in missing:
                matches = np.flatnonzero(template.atom_name == atom_name)
                if len(matches) != 1:
                    raise ValueError(f"No unique CCD representative {name}:{atom_name}")
                new = atoms[start : start + 1].copy()
                new.bonds = struc.BondList(1)
                new.atom_name[:] = atom_name
                new.element[:] = str(template.element[matches[0]]).upper()
                new.coord[:] = 0
                new.is_resolved[:] = False
                new.peptide_source_atom_index[:] = -1
                if "charge" in new.get_annotation_categories():
                    new.charge[:] = template.charge[matches[0]]
                if "occupancy" in new.get_annotation_categories():
                    new.occupancy[:] = 0
                residue = residue + new
            lookup = {n: i for i, n in enumerate(residue.atom_name)}
            for a, b, order in template.bonds.as_array():
                na, nb = template.atom_name[a], template.atom_name[b]
                if na in lookup and nb in lookup and (na in missing or nb in missing):
                    added_bonds.append(
                        (offset + lookup[na], offset + lookup[nb], int(order))
                    )
        pieces.append(residue)
        offset += len(residue)
    result = struc.concatenate(pieces)
    bonds = [
        (old_to_new[int(a)], old_to_new[int(b)], int(order))
        for a, b, order in atoms.bonds.as_array()
    ]
    result.bonds = struc.BondList(
        len(result), np.asarray(bonds + added_bonds, dtype=int).reshape(-1, 3)
    )
    return result


class PeptideParser(DistillationMMCIFParser):
    """Use standard annotations while preserving observed geometry and missingness."""

    @staticmethod
    def fix_arginine(atom_array):
        """Keep source NH1/NH2 identities instead of swapping their coordinates."""
        return atom_array

    def add_missing_atoms_and_residues(self, atom_array):
        """Supply masked representatives without completing residues or peptides."""
        return add_missing_representatives(atom_array, self.entity_poly_type)
