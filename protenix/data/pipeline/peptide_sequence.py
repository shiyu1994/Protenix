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

"""Recover full polymer queries for converted peptide complexes."""

import hashlib
from pathlib import Path

import biotite.structure as struc

from protenix.data.core import ccd
from protenix.data.core.parser import MMCIFParser


def match_full_sequence(observed, chain_id, polymer_type, chain_entities, sequences, types):
    """Match original chain identity and validate every observed sequence position."""
    if not observed or min(observed) < 1:
        raise ValueError(f"Invalid polymer residue IDs in chain {chain_id}")

    def matches(entity):
        sequence = sequences.get(entity, "")
        return (
            types.get(entity) == polymer_type
            and max(observed) <= len(sequence)
            and all(sequence[position - 1] == residue for position, residue in observed.items())
        )

    preferred = chain_entities.get(chain_id)
    if preferred is not None and matches(preferred):
        return sequences[preferred], [preferred], "original_label_asym_id"
    # Some converted inputs rename chains. Only accept a unique full sequence
    # from this same PDB that agrees at *all* observed canonical residue IDs.
    candidates = [entity for entity in sequences if matches(entity)]
    distinct = {sequences[entity] for entity in candidates}
    if len(distinct) != 1:
        raise ValueError(
            f"Cannot uniquely recover chain {chain_id}: "
            f"{len(distinct)} full sequences match the observed residue positions"
        )
    return distinct.pop(), candidates, "unique_sequence_at_original_residue_ids"


def recover_polymer_sequences(structure, original_cif):
    """Restore full sequences without changing atoms, residue IDs, or chemistry."""
    original_cif = Path(original_cif)
    parser = MMCIFParser(mmcif_file=original_cif)
    sequences = parser.get_sequences()
    types = parser.entity_poly_type
    asym = parser.cif.block["struct_asym"]
    chain_entities = dict(zip(asym["id"].as_array(), asym["entity_id"].as_array()))
    atoms = structure["atom_array"]
    observed_chains = {}
    for start in struc.get_residue_starts(atoms):
        entity = str(atoms.label_entity_id[start])
        if entity not in structure["entity_poly_type"]:
            continue
        chain = str(atoms.chain_id[start])
        label = str(atoms.label_asym_id[start])
        key = (entity, chain, label)
        observed = observed_chains.setdefault(key, {})
        position = int(atoms.res_id[start])
        residue = ccd.res_names_to_sequence([str(atoms.res_name[start])])
        if position in observed and observed[position] != residue:
            raise ValueError(f"Conflicting residues in chain {chain} at {position}")
        observed[position] = residue

    corrected, by_chain, records = {}, {}, []
    for (entity, chain, label), observed in observed_chains.items():
        full, original_entities, method = match_full_sequence(
            observed, label, structure["entity_poly_type"][entity], chain_entities, sequences, types
        )
        if entity in corrected and corrected[entity] != full:
            raise ValueError(f"Converted entity {entity} maps to different original sequences")
        corrected[entity] = full
        by_chain[chain] = full
        records.append(
            {
                "entity_id": entity,
                "chain_id": chain,
                "label_asym_id": label,
                "original_entity_ids": original_entities,
                "method": method,
                "observed_residues": len(observed),
                "previous_sequence_length": len(structure["sequences"][entity]),
                "full_sequence_length": len(full),
                "sequence_changed": structure["sequences"][entity] != full,
                "residue_ids": sorted(observed),
            }
        )
    if not corrected:
        raise ValueError("No observed polymer chain to recover")
    with original_cif.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    audit = {
        "version": 1,
        "original_cif": str(original_cif.resolve()),
        "original_cif_sha256": source_hash,
        "residue_mapping": "original_label_seq_id; every observed residue validated",
        "chains": records,
    }
    # Publish only after every chain has passed validation. Never guess or
    # silently retain an observed-only query on failure.
    structure["sequences"] = {**structure["sequences"], **corrected}
    if "sequences_by_chain_id" in structure:
        structure["sequences_by_chain_id"] = {
            **structure["sequences_by_chain_id"], **by_chain
        }
    structure["polymer_sequence_recovery"] = audit
    return audit
