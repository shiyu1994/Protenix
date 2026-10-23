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

"""Prepare protein/PEP CIFs with custom peptide chemistry for finetuning.

The converted CIFs omit polymer metadata and use PEP for arbitrary peptides.
Preserve their observed atoms and explicit bonds, recover full polymer sequences
from the original PDB CIF, and generate independent ligand reference conformers.
No assembly expansion or completion of unobserved atoms is performed.
"""

import argparse
import fcntl
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import biotite.structure as struc
import numpy as np
import pandas as pd
from biotite.structure.io.pdbx import CIFCategory

from configs.configs_data import data_configs
from protenix.data.core import ccd
from protenix.data.core.parser import DistillationMMCIFParser
from protenix.data.pipeline import peptide_sequence
from protenix.data.tokenizer import AtomArrayTokenizer
from protenix.utils.file_io import dump_gzip_pickle, load_gzip_pickle
from rdkit import Chem
from rdkit.Chem import AllChem


def file_hash(path):
    """Hash file contents without reading the entire file into memory."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    """Atomically publish a JSON record."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=json_scalar) + "\n")
    temp.replace(path)


def json_scalar(value):
    """Convert NumPy scalars produced by structure metadata to JSON scalars."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def atom_name(index, element):
    """Generate a unique four-character name without truncating large indices."""
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if not 0 <= index < 36**3:
        raise ValueError("A converted peptide must have fewer than 46656 atoms")
    suffix = ""
    for _ in range(3):
        index, remainder = divmod(index, 36)
        suffix = digits[remainder] + suffix
    return element[0].upper() + suffix


def add_polymer_metadata(parser):
    """Recover observed polymer sequences while leaving nonpolymer entities intact."""
    site = parser.cif.block["atom_site"]
    columns = {key: site[key].as_array() for key in site}
    polymers, seq_rows = {}, []
    for entity in dict.fromkeys(columns["label_entity_id"]):
        mask = columns["label_entity_id"] == entity
        names = set(columns["label_comp_id"][mask])
        if "PEP" in names:
            if names != {"PEP"}:
                raise ValueError(f"PEP shares entity {entity} with other residues")
            continue
        if not np.any(columns["group_PDB"][mask] == "ATOM"):
            continue
        types = {ccd.get_mol_type(name) for name in names}
        if types == {"protein"}:
            polymers[entity] = "polypeptide(L)"
        elif types == {"rna"}:
            polymers[entity] = "polyribonucleotide"
        elif types == {"dna"}:
            polymers[entity] = "polydeoxyribonucleotide"
        else:
            raise ValueError(f"Cannot infer polymer type for entity {entity}: {names}")
        residues = {}
        for seq_id, name in zip(
            columns["label_seq_id"][mask], columns["label_comp_id"][mask]
        ):
            if int(seq_id) in residues and residues[int(seq_id)] != name:
                raise ValueError(
                    f"Conflicting residue names in entity {entity} at {seq_id}"
                )
            residues[int(seq_id)] = name
        seq_rows.extend(
            (entity, str(seq_id), name) for seq_id, name in sorted(residues.items())
        )
    if not polymers:
        raise ValueError("No protein or nucleic-acid entity found")
    parser.cif.block["entity_poly"] = CIFCategory(
        {
            "entity_id": list(polymers),
            "type": list(polymers.values()),
        }
    )
    parser.cif.block["entity_poly_seq"] = CIFCategory(
        {
            key: [row[i] for row in seq_rows]
            for i, key in enumerate(("entity_id", "num", "mon_id"))
        }
    )


def restore_explicit_bonds(parser, atoms):
    """Restore struct_conn bond orders, including bonds in ordinary ligands."""
    lookup = {
        (
            str(a.label_asym_id),
            str(a.label_seq_id),
            str(a.res_name),
            str(a.atom_name),
        ): i
        for i, a in enumerate(atoms)
    }
    if len(lookup) != len(atoms):
        raise ValueError("Ambiguous atom identifiers or alternate conformations")
    conn = parser.cif.block["struct_conn"]
    columns = {key: conn[key].as_array() for key in conn}
    orders = {"SING": 1, "DOUB": 2, "TRIP": 3, "AROM": int(struc.BondType.AROMATIC)}
    explicit = {}
    for row in range(conn.row_count):
        if str(columns["conn_type_id"][row]) not in {"covale", "disulf"}:
            raise ValueError(
                f"Unsupported connection type: {columns['conn_type_id'][row]}"
            )
        keys = [
            tuple(
                str(columns[f"ptnr{n}_label_{field}"][row])
                for field in ("asym_id", "seq_id", "comp_id", "atom_id")
            )
            for n in (1, 2)
        ]
        if not all(key in lookup for key in keys):
            raise ValueError(f"Bond endpoint is absent from atom_site: {keys}")
        pair = tuple(sorted(lookup[key] for key in keys))
        order = orders[str(columns["pdbx_value_order"][row]).upper()]
        if pair[0] == pair[1] or (pair in explicit and explicit[pair] != order):
            raise ValueError(f"Invalid or conflicting bond: {keys}")
        explicit[pair] = order
    pep = atoms.res_name == "PEP"
    # PEP is a custom molecule here, not the CCD component with the same name.
    combined = {
        tuple(sorted((int(a), int(b)))): int(order)
        for a, b, order in atoms.bonds.as_array()
        if not (pep[a] or pep[b])
    }
    combined.update(explicit)
    atoms.bonds = struc.BondList(
        len(atoms), np.asarray([(*pair, order) for pair, order in combined.items()])
    )
    return explicit


def make_reference(atoms, ids, seed):
    """Build a peptide reference from source chemistry and an independent conformer."""
    local = {int(atom_id): i for i, atom_id in enumerate(ids)}
    mol = Chem.RWMol()
    names = []
    for i, atom_id in enumerate(ids):
        element = str(atoms.element[atom_id]).title()
        atom = Chem.Atom(element)
        if "charge" in atoms.get_annotation_categories():
            atom.SetFormalCharge(int(atoms.charge[atom_id]))
        mol.AddAtom(atom)
        names.append(atom_name(i, element))
    rdkit_orders = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        int(struc.BondType.AROMATIC): Chem.BondType.AROMATIC,
    }
    for a, b, order in atoms.bonds.as_array():
        if a in local and b in local:
            mol.AddBond(local[a], local[b], rdkit_orders[int(order)])
        elif (a in local) != (b in local):
            raise ValueError(
                "A converted PEP has covalent bonds outside its residue; separate treatment is required"
            )
    mol = mol.GetMol()
    # Heavy-atom CIFs omit the [nH] required by a neutral pyrrole/indole ring.
    # Recover only the unambiguous single-nitrogen, five-membered carbon ring.
    # Multiple-nitrogen rings (e.g. histidine) still require explicit chemistry.
    mol.UpdatePropertyCache(strict=False)
    inferred_hydrogens = []
    for ring in Chem.GetSymmSSSR(mol):
        ring = list(ring)
        if sorted(mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring) != [
            6,
            6,
            6,
            6,
            7,
        ]:
            continue
        if not all(
            mol.GetBondBetweenAtoms(a, b).GetIsAromatic()
            for a, b in zip(ring, ring[1:] + ring[:1])
        ):
            continue
        nitrogen = next(
            mol.GetAtomWithIdx(i)
            for i in ring
            if mol.GetAtomWithIdx(i).GetAtomicNum() == 7
        )
        if nitrogen.GetDegree() == 2 and nitrogen.GetFormalCharge() == 0:
            nitrogen.SetNumExplicitHs(1)
            nitrogen.SetNoImplicit(True)
            inferred_hydrogens.append(names[nitrogen.GetIdx()])
    Chem.SanitizeMol(mol)
    native = Chem.Conformer(len(ids))
    for i, atom_id in enumerate(ids):
        native.SetAtomPosition(i, atoms.coord[atom_id].astype(float))
    mol.AddConformer(native)
    Chem.AssignAtomChiralTagsFromStructure(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    mol.RemoveAllConformers()
    mol_h = Chem.AddHs(mol)
    options = AllChem.ETKDGv3()
    options.randomSeed = seed
    options.numThreads = 1
    options.useRandomCoords = True
    options.maxIterations = 1000
    if hasattr(options, "timeout"):
        options.timeout = 120
    if AllChem.EmbedMolecule(mol_h, options) < 0:
        raise ValueError("RDKit reference conformer generation failed")
    mol = Chem.RemoveHs(mol_h)
    mol.atom_map = {name: i for i, name in enumerate(names)}
    mol.ref_conf_id = 0
    mol.ref_mask = np.ones(len(ids), dtype=bool)
    mol.inferred_hydrogen_atoms = inferred_hydrogens
    return mol, names


def convert(path, output, seed, original_cif=None):
    """Convert one CIF and return peptide interface rows and preservation diagnostics."""
    parser = DistillationMMCIFParser(mmcif_file=path)
    add_polymer_metadata(parser)
    atoms = parser.get_structure(bond_lenth_threshold=None)
    if atoms is None or len(atoms) != parser.cif.block["atom_site"].row_count:
        raise ValueError("Input must contain one model without alternate atom records")
    if (
        not np.isfinite(atoms.coord).all()
        or np.isin(atoms.element, ["H", "D", "X"]).any()
    ):
        raise ValueError(
            "Input must contain finite heavy-atom coordinates and known elements"
        )
    atoms.set_annotation("peptide_source_atom_index", np.arange(len(atoms)))
    # Tokenizer element IDs use upper-case symbols, including CL, MG, and ZN.
    atoms.element = np.char.upper(atoms.element)
    native_coord = atoms.coord.copy()
    explicit = restore_explicit_bonds(parser, atoms)
    pep = atoms.res_name == "PEP"
    if not pep.any():
        raise ValueError("No converted PEP residue found")
    peptide_entities = set(atoms.label_entity_id[pep])
    custom_mols, ligands = {}, []
    for chain_number, chain in enumerate(dict.fromkeys(atoms.label_asym_id[pep])):
        ids = np.flatnonzero(pep & (atoms.label_asym_id == chain))
        if len(set(atoms.res_id[ids])) != 1:
            raise ValueError("Multiple PEP residues in one chain are unsupported")
        mol, names = make_reference(atoms, ids, seed)
        # Pick an unused local identifier; custom references are passed explicitly.
        name = f"X{chain_number:02d}"
        if len(name) > 3 or name in set(atoms.res_name):
            raise ValueError("Custom peptide residue-name collision")
        custom_mols[name] = mol
        ligands.append(
            {
                "chain": str(chain),
                "entity": str(atoms.label_entity_id[ids[0]]),
                "residue": name,
                "atoms": len(ids),
                "bonds": mol.GetNumBonds(),
                "smiles": Chem.MolToSmiles(mol),
                "atom_name_mapping": dict(zip(atoms.atom_name[ids].tolist(), names)),
                "inferred_pyrrole_hydrogen_atoms": mol.inferred_hydrogen_atoms,
            }
        )
        atoms.res_name[ids] = name
        atoms.atom_name[ids] = names
    # Reuse the repository's annotation/index pipeline on our corrected structure.
    parser.get_structure = lambda: atoms.copy()
    structure = parser.get_structure_dict(add_missing_atom=False, ccd_mols=custom_mols)
    aa = structure["atom_array"]
    if aa is None or len(aa) != len(atoms):
        raise ValueError("Structure processing removed source atoms")
    source_ids = aa.peptide_source_atom_index
    if not np.array_equal(np.sort(source_ids), np.arange(len(atoms))):
        raise ValueError("Source atom identities changed")
    if not np.array_equal(aa.coord, native_coord[source_ids]):
        raise ValueError("Source atom coordinates changed")
    observed = {
        tuple(sorted((int(source_ids[a]), int(source_ids[b])))): int(order)
        for a, b, order in aa.bonds.as_array()
    }
    if any(observed.get(pair) != order for pair, order in explicit.items()):
        raise ValueError("Source bond connectivity or order changed")
    peptide_mask = np.isin(aa.label_entity_id, list(peptide_entities))
    if (
        int(peptide_mask.sum()) != int(pep.sum())
        or not aa.is_ligand[peptide_mask].all()
    ):
        raise ValueError("Converted peptide atoms lost their ligand classification")
    if not aa.ref_mask[peptide_mask].all() or not np.isfinite(aa.ref_pos).all():
        raise ValueError("Invalid peptide reference features")
    aa.set_annotation("resolution", np.full(len(aa), parser.resolution))
    structure["token_array"] = AtomArrayTokenizer(aa).get_token_array()
    structure.update(
        msa_features=None, template_features=None, source_cif_path=str(path)
    )
    rows = [
        row
        for row in parser.make_indices(structure)
        if row["type"] == "interface"
        and row["mol_type_group"] in {"ligand_prot", "ligand_nuc"}
        and (
            row["entity_1_id"] in peptide_entities
            or row["entity_2_id"] in peptide_entities
        )
    ]
    if not rows:
        raise ValueError("No polymer/converted-peptide interface within 5 angstroms")
    # Loader paths use pdb_id, so force a unique filename-derived identifier.
    structure["pdb_id"] = path.stem
    for row in rows:
        row["pdb_id"] = path.stem
    if original_cif is None:
        original_cif = (
            Path(data_configs["weightedPDB_before2109_wopb_nometalc_0925"]["base_info"]["mmcif_dir"])
            / f"{path.stem}.cif"
        )
    sequence_audit = peptide_sequence.recover_polymer_sequences(structure, original_cif)
    dump_gzip_pickle(structure, output, compression_level=3, atomic=True)
    return rows, {
        "atoms": len(aa),
        "tokens": len(structure["token_array"]),
        "peptide_atoms": int(peptide_mask.sum()),
        "explicit_bonds": len(explicit),
        "resolution": parser.resolution,
        "ligands": ligands,
        "reference_method": "ETKDGv3; native coordinates discarded after stereo assignment",
        "missing_atom_completion": False,
        "assembly_expansion": False,
        "polymer_sequence_recovery": sequence_audit,
    }


def process_file(task):
    """Reuse a content-verified cache entry or convert one structure."""
    path, root, settings, check_only = task
    record_path = root / "records" / f"{path.stem}.json"
    output = root / "bioassembly" / f"{path.stem}.pkl.gz"
    signature = {
        "source": str(path),
        "source_sha256": file_hash(path),
        "settings": {key: value for key, value in settings.items() if key != "reuse_prepared_dir"},
    }
    original_cif = None
    if settings.get("original_mmcif_dir"):
        original_cif = Path(settings["original_mmcif_dir"]) / f"{path.stem}.cif"
        signature["original_cif_sha256"] = (
            file_hash(original_cif) if original_cif.is_file() else None
        )
    if record_path.is_file():
        try:
            record = json.loads(record_path.read_text())
            if record["signature"] == signature:
                if record["status"] == "failed" and check_only:
                    return record
                if (
                    record["status"] == "ok"
                    and output.is_file()
                    and file_hash(output) == record["output_sha256"]
                ):
                    return record
        except (KeyError, ValueError, OSError):
            pass
    if check_only:
        return {
            "status": "failed",
            "signature": signature,
            "error": "Missing, stale, or corrupt prepared cache",
        }
    try:
        if settings.get("reuse_prepared_dir"):
            old_root = Path(settings["reuse_prepared_dir"])
            old = json.loads((old_root / "records" / f"{path.stem}.json").read_text())
            if old["signature"]["source_sha256"] != signature["source_sha256"]:
                raise ValueError("Source CIF differs from the cache being repaired")
            if old["status"] != "ok":
                raise ValueError(f"Previously rejected source: {old['error']}")
            if old["signature"]["settings"]["seed"] != settings["seed"]:
                raise ValueError("Reference seed differs from the cache being repaired")
            old_output = old_root / "bioassembly" / f"{path.stem}.pkl.gz"
            if file_hash(old_output) != old["output_sha256"]:
                raise ValueError("Corrupt cache being repaired")
            structure = load_gzip_pickle(old_output)
            sequence_audit = peptide_sequence.recover_polymer_sequences(structure, original_cif)
            dump_gzip_pickle(structure, output, compression_level=3, atomic=True)
            rows = old["rows"]
            audit = {
                **old["audit"],
                "polymer_sequence_recovery": sequence_audit,
                "reused_cache_sha256": old["output_sha256"],
            }
        else:
            rows, audit = convert(path, output, settings["seed"], original_cif)
        record = {
            "status": "ok",
            "signature": signature,
            "output_sha256": file_hash(output),
            "rows": rows,
            "audit": audit,
        }
    except Exception as exc:
        record = {
            "status": "failed",
            "signature": signature,
            "error": f"{type(exc).__name__}: {exc}",
        }
    write_json(record_path, record)
    return record


def main():
    """Prepare all selected CIFs, publish an index, and report any rejected files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--original-mmcif-dir", type=Path,
        default=Path(data_configs["weightedPDB_before2109_wopb_nometalc_0925"]["base_info"]["mmcif_dir"]),
        help="Original PDB CIFs used to recover full polymer sequences and validate residue positions",
    )
    parser.add_argument(
        "--reuse-prepared-dir", type=Path,
        help="Repair a verified existing cache into a different output directory without regenerating chemistry",
    )
    parser.add_argument(
        "--ids", help="Optional comma-separated CIF stems for a small run"
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Explicitly allow excluding rejected CIFs",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify existing preparation; do not convert files",
    )
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.seed < 2**31:
        parser.error("workers must be positive and seed must be in [0, 2**31)")
    source, root = args.input_dir.resolve(), args.output_dir.resolve()
    if not args.original_mmcif_dir.is_dir():
        parser.error("original-mmcif-dir must contain the original PDB CIFs")
    if args.reuse_prepared_dir and args.reuse_prepared_dir.resolve() == root:
        parser.error("Repair must write a separate output directory")
    files = sorted(source.glob("*.cif"))
    if args.ids:
        requested = set(args.ids.split(","))
        files = [path for path in files if path.stem in requested]
        if requested != {path.stem for path in files}:
            parser.error("Some requested CIF IDs are missing")
    if not files:
        parser.error("No .cif files found")
    if root == source or source in root.parents:
        parser.error("output-dir must be outside the source CIF directory")
    ccd_paths = [
        Path(data_configs[key])
        for key in ("ccd_components_file", "ccd_components_rdkit_mol_file")
    ]
    for path in ccd_paths:
        if not path.is_file():
            parser.error(f"Missing CCD resource: {path}; set PROTENIX_ROOT_DIR")
    settings = {
        "script_sha256": file_hash(__file__),
        "seed": args.seed,
        "original_mmcif_dir": str(args.original_mmcif_dir.resolve()),
        "sequence_recovery_sha256": file_hash(peptide_sequence.__file__),
        "reuse_prepared_dir": str(args.reuse_prepared_dir.resolve()) if args.reuse_prepared_dir else None,
        "ccd": [
            {
                "path": str(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in ccd_paths
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another preparation/check is using this output directory")
        for directory in ("records", "bioassembly"):
            (root / directory).mkdir(exist_ok=True)
        tasks = [(path, root, settings, args.check) for path in files]
        records = []
        if args.workers == 1 or args.check:
            iterator = map(process_file, tasks)
            pool = None
        else:
            pool = ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
            iterator = pool.map(process_file, tasks)
        try:
            for i, record in enumerate(iterator, 1):
                records.append(record)
                name = Path(record["signature"]["source"]).name
                detail = record.get(
                    "error", f"{len(record.get('rows', []))} interfaces"
                )
                print(
                    f"[{i}/{len(files)}] {name}: {record['status']} ({detail})",
                    flush=True,
                )
        finally:
            if pool is not None:
                pool.shutdown()
        failures = [rec for rec in records if rec["status"] != "ok"]
        rows = [row for rec in records if rec["status"] == "ok" for row in rec["rows"]]
        ready = bool(rows) and (not failures or args.skip_errors)
        summary = {
            "ready": ready,
            "input_dir": str(source),
            "settings": settings,
            "requested_files": len(files),
            "prepared_files": len(records) - len(failures),
            "failed_files": len(failures),
            "interface_rows": len(rows),
            "skip_errors": args.skip_errors,
            "ids": args.ids,
        }
        frame = pd.DataFrame(rows)
        if args.check:
            index = root / "indices.csv"
            if not index.is_file() or index.read_text() != frame.to_csv(index=False):
                ready = False
                print(
                    "Prepared index does not match this selection; run preprocessing first",
                    flush=True,
                )
        else:
            temp = root / "indices.csv.tmp"
            frame.to_csv(temp, index=False)
            temp.replace(root / "indices.csv")
            write_json(root / "failures.json", failures)
            write_json(root / "preparation.json", summary)
        print(json.dumps({**summary, "ready": ready}), flush=True)
        if not ready:
            raise SystemExit(
                "Preparation is incomplete. Inspect failures.json; exclusions require --skip-errors."
            )


if __name__ == "__main__":
    main()
