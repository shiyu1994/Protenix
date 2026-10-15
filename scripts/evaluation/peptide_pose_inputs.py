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

"""Coordinate-free Protenix inputs and scoring references for peptide assemblies."""

import hashlib
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from biotite.structure.io.pdbx import CIFFile


SCHEMA_VERSION = 1


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def category_rows(block, name):
    if name not in block:
        return []
    category = block[name]
    columns = {key: category[key].as_array().tolist() for key in category}
    return [dict(zip(columns, values)) for values in zip(*columns.values())]


def atom_rows(block):
    """Read observed heavy atoms using label identifiers and the first model."""
    rows = category_rows(block, "atom_site")
    if not rows:
        raise ValueError("Empty atom_site")
    model = rows[0].get("pdbx_PDB_model_num", "1")
    atoms = {}
    for row in rows:
        if row.get("pdbx_PDB_model_num", "1") != model:
            continue
        if row["type_symbol"].upper() in {"H", "D"}:
            continue
        if float(row.get("occupancy", "1")) <= 0:
            continue
        chain = row["label_asym_id"]
        pos = row["label_seq_id"]
        if pos in {".", "?"}:
            pos = "auth:" + row["auth_seq_id"] + ":" + row.get("pdbx_PDB_ins_code", "?")
        key = (chain, pos, row["label_comp_id"], row["label_atom_id"])
        coord = [float(row[col]) for col in ("Cartn_x", "Cartn_y", "Cartn_z")]
        if not np.isfinite(coord).all():
            raise ValueError(f"Nonfinite coordinate: {key}")
        item = {
            "chain": chain,
            "position": pos,
            "resname": row["label_comp_id"],
            "atom": row["label_atom_id"],
            "element": row["type_symbol"].upper(),
            "entity": row["label_entity_id"],
            "coord": coord,
            "occupancy": float(row.get("occupancy", "1")),
            "altloc": row.get("label_alt_id", "."),
        }
        # Occupancy then altloc gives a deterministic observed conformation.
        if key not in atoms or (-item["occupancy"], item["altloc"]) < (
            -atoms[key]["occupancy"],
            atoms[key]["altloc"],
        ):
            atoms[key] = item
    return list(atoms.values())


@lru_cache(maxsize=None)
def canonical_atom_names(ccd_code):
    """Map unambiguous CCD alternative atom names to canonical names."""
    from protenix.data.core import ccd

    component = ccd.get_component_atom_array(ccd_code, keep_leaving_atoms=True)
    if component is None:
        raise ValueError(f"CCD component unavailable: {ccd_code}")
    names = set(component.atom_name.tolist())
    candidates = defaultdict(set)
    for column in ("alt_atom_id", "pdbx_component_atom_id"):
        for alias, canonical in zip(
            component.get_annotation(column), component.atom_name
        ):
            candidates[str(alias)].add(str(canonical))
    aliases = {name: name for name in names}
    for alias, values in candidates.items():
        if alias not in names and len(values) == 1:
            aliases[alias] = next(iter(values))
    return aliases


@lru_cache(maxsize=None)
def canonical_bonds(ccd_code):
    """Return the CCD intramolecular connectivity in canonical atom names."""
    from protenix.data.core import ccd

    atoms = ccd.get_component_atom_array(ccd_code, keep_leaving_atoms=True)
    if atoms is None or atoms.bonds is None:
        raise ValueError(f"CCD bond topology unavailable: {ccd_code}")
    return {
        tuple(sorted((str(atoms.atom_name[i]), str(atoms.atom_name[j]))))
        for i, j, _ in atoms.bonds.as_array()
    }


def polymer_spec(poly_type, residue_names):
    from protenix.data.constants import PROT_STD_RESIDUES_ONE_TO_THREE
    from protenix.data.core import ccd

    inverse = {value: key for key, value in PROT_STD_RESIDUES_ONE_TO_THREE.items()}
    if poly_type != "polypeptide(L)":
        raise ValueError(f"Unsupported receptor polymer type: {poly_type}")
    seq, modifications = [], []
    for index, name in enumerate(residue_names, 1):
        code = inverse.get(name)
        if code is None:
            code = ccd.get_one_letter_code(name)
            if (
                code is None
                or len(code) != 1
                or code not in PROT_STD_RESIDUES_ONE_TO_THREE
            ):
                code = "X"
            modifications.append({"ptmType": "CCD_" + name, "ptmPosition": index})
        seq.append(code)
    spec = {"sequence": "".join(seq)}
    if modifications:
        spec["modifications"] = modifications
    return "proteinChain", spec


def prepare_case(path, msa_map=None):
    """Build chemistry-only inputs; native coordinates are confined to reference."""
    path = Path(path).resolve()
    block = CIFFile.read(path).block
    native = atom_rows(block)
    poly_types = {
        r["entity_id"]: r["type"] for r in category_rows(block, "entity_poly")
    }
    poly_seq = defaultdict(dict)
    for row in category_rows(block, "entity_poly_seq"):
        number = int(row["num"])
        if (
            number in poly_seq[row["entity_id"]]
            and poly_seq[row["entity_id"]][number] != row["mon_id"]
        ):
            raise ValueError(
                "Microheterogeneous polymer sequence requires explicit selection"
            )
        poly_seq[row["entity_id"]][number] = row["mon_id"]
    peptide_rows = category_rows(block, "foldbench_peptide_ligand")
    if not peptide_rows:
        raise ValueError(
            "Missing _foldbench_peptide_ligand annotation; peptide identities must be explicit"
        )
    peptide_chains = [row["id"] for row in peptide_rows]
    chains = defaultdict(list)
    omitted_water_atoms = 0
    for atom in native:
        if atom["resname"] in {"HOH", "DOD", "WAT"}:
            omitted_water_atoms += 1
        else:
            chains[atom["chain"]].append(atom)
    if not set(peptide_chains) <= chains.keys():
        raise ValueError("Annotated peptide chain is absent from atom_site")

    jobs, groups, chain_info, positions = [], {}, {}, {}
    terminal_caps, normalizations = {}, []
    for chain, atoms in chains.items():
        entities = {a["entity"] for a in atoms}
        if len(entities) != 1:
            raise ValueError(f"Chain {chain} spans multiple entities")
        entity = next(iter(entities))
        observed_residues = list(
            dict.fromkeys((a["position"], a["resname"]) for a in atoms)
        )
        if entity in poly_types:
            if chain in peptide_chains:
                raise ValueError(
                    "Peptide annotation conflicts with nonpolymer ligand representation"
                )
            sequence = poly_seq[entity]
            if not sequence:
                raise ValueError(f"Missing polymer sequence for entity {entity}")
            source_positions = sorted(sequence)
            residues = [sequence[number] for number in source_positions]
            kind, spec = polymer_spec(poly_types[entity], residues)
            pos_map = {str(number): i for i, number in enumerate(source_positions, 1)}
            role = "receptor"
            if msa_map is not None:
                if spec["sequence"] not in msa_map:
                    raise ValueError(
                        f"No supplied MSA for receptor sequence in chain {chain}"
                    )
                entry = msa_map[spec["sequence"]]
                if (
                    not isinstance(entry, dict)
                    or not entry
                    or set(entry) - {"unpairedMsaPath", "pairedMsaPath"}
                ):
                    raise ValueError(
                        "MSA map values must contain pairedMsaPath and/or unpairedMsaPath"
                    )
                for key, filename in entry.items():
                    target = Path(filename).resolve(strict=True)
                    spec[key] = str(target)
        else:
            residues = [name for _, name in observed_residues]
            pos_map = {pos: i for i, (pos, _) in enumerate(observed_residues, 1)}
            if len(pos_map) != len(observed_residues):
                raise ValueError(f"Ambiguous residue identifiers in chain {chain}")
            nxt = [
                a
                for a in atoms
                if a["atom"] == "NXT"
                and "NXT" not in canonical_atom_names(a["resname"])
            ]
            if nxt:
                if (
                    len(nxt) != 1
                    or nxt[0]["position"] != observed_residues[-1][0]
                    or nxt[0]["element"] != "N"
                ):
                    raise ValueError(
                        "NXT can only denote a single terminal amide nitrogen"
                    )
                terminal = [a for a in atoms if a["position"] == nxt[0]["position"]]
                if not {"C", "O"} <= {a["atom"] for a in terminal} or any(
                    a["atom"] == "OXT" for a in terminal
                ):
                    raise ValueError("Inconsistent terminal NXT amide chemistry")
                terminal_caps[chain] = (len(residues), len(residues) + 1)
                residues.append("NH2")
                normalizations.append(
                    {
                        "chain": chain,
                        "change": "terminal NXT represented by bonded CCD_NH2",
                    }
                )
            kind, spec = "ligand", {"ligand": "CCD_" + "_".join(residues)}
            role = "peptide" if chain in peptide_chains else "cofactor"
        identity = json.dumps([kind, spec], sort_keys=True)
        if identity not in groups:
            groups[identity] = len(jobs)
            jobs.append({kind: {**spec, "count": 0, "id": []}})
        entity_index = groups[identity]
        body = jobs[entity_index][kind]
        body["count"] += 1
        body["id"].append(chain)
        chain_info[chain] = {
            "role": role,
            "group": identity,
            "entity": entity_index + 1,
            "copy": body["count"],
            "residues": residues,
        }
        positions[chain] = pos_map

    if not any(info["role"] == "receptor" for info in chain_info.values()):
        raise ValueError("No receptor protein chains")
    reference_atoms = []
    endpoint_map = {}
    endpoint_coords = {}
    for chain, atoms in chains.items():
        for atom in atoms:
            residue = atom["resname"]
            aliases = canonical_atom_names(residue)
            name = aliases.get(atom["atom"])
            if (
                name is None
                and atom["atom"].endswith("''")
                and atom["atom"][:-1] in aliases
            ):
                name = aliases[atom["atom"][:-1]]
                normalizations.append(
                    {
                        "chain": chain,
                        "position": atom["position"],
                        "atom": atom["atom"],
                        "canonical_atom": name,
                    }
                )
            position = positions[chain].get(atom["position"])
            if atom["atom"] == "NXT" and chain in terminal_caps:
                position, name, residue = terminal_caps[chain][1], "N", "NH2"
            if name is None:
                raise ValueError(
                    f"Unknown CCD atom: {chain}/{atom['position']}/{residue}/{atom['atom']}"
                )
            if position is None:
                raise ValueError(
                    f"Observed residue absent from polymer sequence: {chain}/{atom['position']}"
                )
            source_key = (chain, atom["position"], atom["resname"], atom["atom"])
            endpoint_map[source_key] = (position, name)
            endpoint_coords[source_key] = atom["coord"]
            if residue == "MSE":
                residue = "MET"
                if name == "SE":
                    name = "SD"
            reference_atoms.append(
                [chain, position, residue, name, atom["element"], *atom["coord"]]
            )

    bonds, fixed_atoms, ignored_connections = [], defaultdict(set), []
    for connection in category_rows(block, "struct_conn"):
        if connection["conn_type_id"] not in {"covale", "disulf"}:
            ignored_connections.append(connection)
            continue
        if connection.get("pdbx_value_order", "SING").upper() not in {"SING", "?", "."}:
            raise ValueError(
                "Protenix JSON inter-residue bonds support single bond order only"
            )
        bond = {}
        candidates_by_side = []
        for side in (1, 2):
            prefix = f"ptnr{side}_label_"
            chain = connection[prefix + "asym_id"]
            if chain not in chain_info:
                raise ValueError(
                    "Covalent connection references an omitted or missing chain"
                )
            seq = connection[prefix + "seq_id"]
            residue = connection[prefix + "comp_id"]
            atom = connection[prefix + "atom_id"]
            candidates = [
                (key, value)
                for key, value in endpoint_map.items()
                if key[0] == chain
                and key[2:] == (residue, atom)
                and (
                    (seq in {".", "?"} and key[1].startswith("auth:")) or key[1] == seq
                )
            ]
            if not candidates:
                raise ValueError(
                    f"Ambiguous or absent bond endpoint: {chain}/{seq}/{residue}/{atom}"
                )
            candidates_by_side.append(candidates)
        if any(len(values) > 1 for values in candidates_by_side):
            # The exported struct_conn drops auth residue identifiers for glycans.
            # Resolve only the explicitly declared bond, never proximity contacts.
            plausible = []
            for left in candidates_by_side[0]:
                for right in candidates_by_side[1]:
                    distance = np.linalg.norm(
                        np.asarray(endpoint_coords[left[0]]) - endpoint_coords[right[0]]
                    )
                    if left[0] != right[0] and 0.8 <= distance <= 2.3:
                        plausible.append((left, right))
            if len(plausible) != 1:
                raise ValueError(
                    f"Cannot uniquely resolve explicit connection {connection['id']}"
                )
            selected = plausible[0]
            normalizations.append(
                {
                    "connection": connection["id"],
                    "resolved_endpoints": [list(x[0]) for x in selected],
                }
            )
        else:
            selected = [values[0] for values in candidates_by_side]
        left, right = selected
        if left[0][:3] == right[0][:3]:
            # The export contains spurious C--N self-links within amino acids.
            # CCD already defines N--CA--C; the shortcut would create an
            # unsupported three-membered ring.
            pair = tuple(sorted((left[1][1], right[1][1])))
            topology = canonical_bonds(left[0][2])
            if pair in topology:
                reason = "redundant intra-residue bond already supplied by CCD"
            elif pair == ("C", "N") and {("C", "CA"), ("CA", "N")} <= topology:
                reason = "omitted source C--N self-link incompatible with CCD N--CA--C backbone"
            else:
                raise ValueError(
                    f"Explicit intra-residue bond conflicts with CCD: {connection['id']}"
                )
            normalizations.append(
                {
                    "connection": connection["id"],
                    "change": reason,
                    "source_endpoints": [list(x[0]) for x in selected],
                }
            )
            continue
        for side, (source_key, (position, name)) in enumerate(selected, 1):
            chain = source_key[0]
            info = chain_info[chain]
            bond.update(
                {
                    f"entity{side}": info["entity"],
                    f"copy{side}": info["copy"],
                    f"position{side}": position,
                    f"atom{side}": name,
                }
            )
            fixed_atoms[(chain, position)].add(name)
        bonds.append(bond)
    for chain, (position, cap_position) in terminal_caps.items():
        info = chain_info[chain]
        bonds.append(
            {
                "entity1": info["entity"],
                "copy1": info["copy"],
                "position1": position,
                "atom1": "C",
                "entity2": info["entity"],
                "copy2": info["copy"],
                "position2": cap_position,
                "atom2": "N",
            }
        )
        fixed_atoms[(chain, position)].add("C")
        fixed_atoms[(chain, cap_position)].add("N")
    unique_bonds = {}
    for bond in bonds:
        endpoints = tuple(
            sorted(
                tuple(
                    bond[f"{key}{side}"]
                    for key in ("entity", "copy", "position", "atom")
                )
                for side in (1, 2)
            )
        )
        unique_bonds.setdefault(endpoints, bond)
    bonds = list(unique_bonds.values())
    job = {"name": path.stem, "sequences": jobs, "covalent_bonds": bonds}
    # Peptide-copy equivalence requires identical explicit internal connectivity.
    for chain in peptide_chains:
        info = chain_info[chain]
        local_bonds = []
        for bond in bonds:
            ends = [(bond[f"entity{i}"], bond[f"copy{i}"]) for i in (1, 2)]
            if ends == [(info["entity"], info["copy"])] * 2:
                local_bonds.append(
                    sorted(
                        [
                            (bond["position1"], bond["atom1"]),
                            (bond["position2"], bond["atom2"]),
                        ]
                    )
                )
        info["group"] += json.dumps(sorted(local_bonds))
    reference = {
        "schema_version": SCHEMA_VERSION,
        "name": path.stem,
        "source": str(path),
        "source_sha256": file_hash(path),
        "chains": chain_info,
        "peptide_chains": peptide_chains,
        "atoms": reference_atoms,
        "fixed_atoms": {
            f"{chain}:{pos}": sorted(names)
            for (chain, pos), names in fixed_atoms.items()
        },
        "omitted_water_atoms": omitted_water_atoms,
        "ignored_connections": ignored_connections,
        "normalizations": normalizations,
        "protocol": "full assembly; ligand peptides; CCD references; no native coordinates/templates/contacts",
    }
    return job, reference


def validate_case(job, reference):
    """Exercise actual Protenix atom construction and verify every target atom maps."""
    from protenix.data.inference.json_to_feature import SampleDictToFeatures
    from protenix.data.tokenizer import AtomArrayTokenizer

    parser = SampleDictToFeatures(job)
    atoms = parser.get_atom_array()
    tokens = AtomArrayTokenizer(atoms).get_token_array()
    predicted_keys = {(str(a.chain_id), int(a.res_id), str(a.atom_name)) for a in atoms}
    missing = [
        row[:4]
        for row in reference["atoms"]
        if (row[0], row[1], row[3]) not in predicted_keys
    ]
    peptide_missing = [row for row in missing if row[0] in reference["peptide_chains"]]
    if peptide_missing:
        raise ValueError(
            f"Peptide atoms absent from generated input: {peptide_missing[:12]}"
        )
    peptide_mask = np.isin(atoms.chain_id, reference["peptide_chains"])
    if not peptide_mask.any() or not np.all(atoms.is_ligand[peptide_mask]):
        raise ValueError("Peptide atoms were not constructed as ligands")
    if not np.isfinite(atoms.ref_pos).all():
        raise ValueError("Nonfinite reference features")
    reference["validation"] = {
        "input_atoms": len(atoms),
        "input_tokens": len(tokens),
        "native_atoms_not_in_input": missing,
        "peptide_atom_coverage": 1.0,
    }
    return reference
