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

"""Read OpenFold3 disordered structure arrays without relying on their CIF metadata."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from rdkit import Chem

from protenix.data.constants import (
    DNA_STD_RESIDUES,
    PROT_STD_RESIDUES_ONE_TO_THREE,
    RNA_STD_RESIDUES,
)
from protenix.data.core import ccd
from protenix.data.core.filter import Filter
from protenix.data.core.parser import AddAtomArrayAnnot, MMCIFParser
from protenix.data.tokenizer import AtomArrayTokenizer
from protenix.data.utils import get_ligand_polymer_bond_mask

OPENFOLD3_MOLECULE_TYPE_NAMES = {
    0: "PROTEIN",
    1: "RNA",
    2: "DNA",
    3: "LIGAND",
}
OPENFOLD3_TO_PROTENIX_MOL_TYPE = {
    0: "protein",
    1: "rna",
    2: "dna",
    3: "ligand",
}
OPENFOLD3_TO_ENTITY_POLY_TYPE = {
    0: "polypeptide(L)",
    1: "polyribonucleotide",
    2: "polydeoxyribonucleotide",
}
OPENFOLD3_TO_INDEX_MOL_TYPE = {
    0: "prot",
    1: "nuc",
    2: "nuc",
    3: "ligand",
}

_REQUIRED_NPZ_FIELDS = {
    "coord",
    "bonds",
    "chain_id",
    "res_id",
    "ins_code",
    "res_name",
    "hetero",
    "atom_name",
    "element",
    "charge",
    "occupancy",
    "label_asym_id",
    "auth_asym_id",
    "molecule_type_id",
    "entity_id",
}
_PROTEIN_THREE_TO_ONE = {
    three_letter: one_letter
    for one_letter, three_letter in PROT_STD_RESIDUES_ONE_TO_THREE.items()
}
_ALPHA_BY_MOL_TYPE = {"prot": 3.0, "nuc": 3.0, "ligand": 1.0}
_BETA_BY_SAMPLE_TYPE = {"chain": 0.5, "interface": 1.0}
_WEIGHT_EPS = 1e-9
_KNOWN_3NZI_CHAIN_REMAP = {"D": "4", "E": "6", "F": "2"}
_KNOWN_3NZI_PEPTIDE_SEQUENCE = "DPMFKLV"


class OpenFold3DisorderedError(ValueError):
    """Indicate invalid or inconsistent OpenFold3 disordered input data."""


def _as_string_array(values: np.ndarray) -> np.ndarray:
    """Return a string array without the fixed-width truncation of AtomArray defaults."""
    values = np.asarray(values).astype(str)
    max_length = max((len(value) for value in values), default=1)
    return values.astype(f"<U{max_length}", copy=False)


def _set_string_annotation(
    atom_array: AtomArray,
    name: str,
    values: np.ndarray,
) -> None:
    """Set an exact-width string annotation, replacing a built-in category if needed."""
    if name in atom_array.get_annotation_categories():
        atom_array.del_annotation(name)
    atom_array.set_annotation(name, _as_string_array(values))


def _chain_ids_in_order(atom_array: AtomArray) -> list[str]:
    """Return chain IDs in atom order and reject non-contiguous repeated chains."""
    if len(atom_array) == 0:
        return []
    chain_start_mask = np.concatenate(
        ([True], atom_array.chain_id[1:] != atom_array.chain_id[:-1])
    )
    chain_ids = atom_array.chain_id[chain_start_mask].tolist()
    if len(chain_ids) != len(set(chain_ids)):
        raise OpenFold3DisorderedError(
            "NPZ chain_id values must each occupy exactly one contiguous atom block"
        )
    return chain_ids


def _restore_cross_site_bonds_after_alternate_selection(
    *,
    original_bonds: np.ndarray,
    keep: np.ndarray,
    replacement_old_indices: Mapping[int, int],
    source_keys_by_atom: list[tuple[str, int, str]],
    selected_atom_array: AtomArray,
) -> AtomArray:
    """Redirect only source-declared bonds broken by alternate selection.

    A source bond can point from an alternate component that is discarded to
    the next retained residue/component.  Recreate that exact topology through
    the selected component's atom with the same name and element.  Bonds whose
    two endpoints share one source residue key are deliberately not redirected:
    some archives contain cross-alternate bonds inside a microheterogeneous
    site, and folding those onto one component would create spurious chemistry.
    """
    old_to_new = np.full(len(keep), -1, dtype=np.int64)
    old_to_new[keep] = np.arange(np.count_nonzero(keep), dtype=np.int64)
    existing_pairs = {
        tuple(sorted((int(atom_1), int(atom_2))))
        for atom_1, atom_2, _bond_type in selected_atom_array.bonds.as_array()
    }
    new_bonds = []
    for raw_atom_1, raw_atom_2, raw_bond_type in original_bonds:
        atom_1 = int(raw_atom_1)
        atom_2 = int(raw_atom_2)
        if keep[atom_1] and keep[atom_2]:
            continue
        if source_keys_by_atom[atom_1] == source_keys_by_atom[atom_2]:
            continue
        mapped_old_1 = (
            atom_1
            if keep[atom_1]
            else replacement_old_indices.get(atom_1)
        )
        mapped_old_2 = (
            atom_2
            if keep[atom_2]
            else replacement_old_indices.get(atom_2)
        )
        if mapped_old_1 is None or mapped_old_2 is None:
            continue
        mapped_1 = int(old_to_new[mapped_old_1])
        mapped_2 = int(old_to_new[mapped_old_2])
        if mapped_1 < 0 or mapped_2 < 0 or mapped_1 == mapped_2:
            continue
        pair = tuple(sorted((mapped_1, mapped_2)))
        if pair in existing_pairs:
            continue
        existing_pairs.add(pair)
        new_bonds.append((mapped_1, mapped_2, int(raw_bond_type)))
    if new_bonds:
        selected_atom_array.bonds = selected_atom_array.bonds.merge(
            struc.BondList(
                len(selected_atom_array),
                np.asarray(new_bonds, dtype=np.uint32),
            )
        )
    return selected_atom_array


def _select_alternate_components(atom_array: AtomArray) -> AtomArray:
    """Keep one component for each source residue microheterogeneity site.

    OpenFold3 archives can contain different monomers at the same
    ``(chain_id, res_id, ins_code)``.  They are alternate sequence/chemical
    states, not additional canonical positions.  Protenix has no alternate
    component sampler, so select the occurrence with the highest finite atom
    occupancy and use source order as the deterministic tie breaker.  The
    latter preserves the canonical first monomer for equal-occupancy
    microheterogeneity instead of changing sequence identity merely because
    one alternate has more unresolved atoms.
    """
    original_res_id = np.asarray(atom_array.res_id, dtype=np.int64).copy()
    original_ins_code = np.asarray(atom_array.ins_code).astype(str).copy()
    atom_array.set_annotation("openfold3_res_id", original_res_id)
    _set_string_annotation(
        atom_array,
        "openfold3_ins_code",
        original_ins_code,
    )

    residue_starts = struc.get_residue_starts(
        atom_array,
        add_exclusive_stop=True,
    )
    grouped_occurrences: dict[
        tuple[str, int, str],
        list[tuple[int, int]],
    ] = {}
    source_keys_by_atom: list[tuple[str, int, str]] = [
        ("", 0, "")
    ] * len(atom_array)
    for start, stop in zip(residue_starts[:-1], residue_starts[1:]):
        key = (
            str(atom_array.chain_id[start]),
            int(original_res_id[start]),
            str(original_ins_code[start]),
        )
        grouped_occurrences.setdefault(key, []).append((int(start), int(stop)))
        source_keys_by_atom[start:stop] = [key] * (int(stop) - int(start))

    keep = np.ones(len(atom_array), dtype=bool)
    alternate_group_size = np.ones(len(atom_array), dtype=np.int64)
    replacement_old_indices: dict[int, int] = {}
    original_bonds = atom_array.bonds.as_array().copy()
    for occurrences in grouped_occurrences.values():
        if len(occurrences) == 1:
            continue
        molecule_types = {
            int(atom_array.molecule_type_id[start])
            for start, _stop in occurrences
        }
        if 3 in molecule_types and molecule_types != {3}:
            raise OpenFold3DisorderedError(
                "one source residue key mixes polymer and ligand "
                "component occurrences"
            )
        occurrence_scores = []
        for start, stop in occurrences:
            occupancies = np.asarray(
                atom_array.occupancy[start:stop],
                dtype=np.float64,
            )
            finite_occupancies = occupancies[np.isfinite(occupancies)]
            occurrence_scores.append(
                float(finite_occupancies.max())
                if finite_occupancies.size
                else -np.inf
            )
        selected_ordinal = max(
            range(len(occurrences)),
            key=lambda ordinal: (
                occurrence_scores[ordinal],
                -ordinal,
            ),
        )
        selected_start, selected_stop = occurrences[selected_ordinal]
        selected_atoms: dict[tuple[str, str], int] = {}
        for selected_index in range(selected_start, selected_stop):
            atom_key = (
                str(atom_array.atom_name[selected_index]),
                str(atom_array.element[selected_index]),
            )
            if atom_key in selected_atoms:
                raise OpenFold3DisorderedError(
                    "selected alternate component contains duplicate atom "
                    f"identity {atom_key!r}"
                )
            selected_atoms[atom_key] = selected_index
        for ordinal, (start, stop) in enumerate(occurrences):
            if ordinal != selected_ordinal:
                keep[start:stop] = False
                for discarded_index in range(start, stop):
                    atom_key = (
                        str(atom_array.atom_name[discarded_index]),
                        str(atom_array.element[discarded_index]),
                    )
                    selected_index = selected_atoms.get(atom_key)
                    if selected_index is not None:
                        replacement_old_indices[discarded_index] = (
                            selected_index
                        )
            else:
                alternate_group_size[start:stop] = len(occurrences)
    atom_array.set_annotation(
        "openfold3_alternate_group_size",
        alternate_group_size,
    )
    if not np.all(keep):
        atom_array = atom_array[keep]
        atom_array = _restore_cross_site_bonds_after_alternate_selection(
            original_bonds=original_bonds,
            keep=keep,
            replacement_old_indices=replacement_old_indices,
            source_keys_by_atom=source_keys_by_atom,
            selected_atom_array=atom_array,
        )
    return atom_array


def _add_occurrence_ref_space_uid(atom_array: AtomArray) -> AtomArray:
    """Give each retained component occurrence its own conformer identity."""
    ref_space_uid = np.empty(len(atom_array), dtype=np.int64)
    residue_starts = struc.get_residue_starts(
        atom_array,
        add_exclusive_stop=True,
    )
    for uid, (start, stop) in enumerate(
        zip(residue_starts[:-1], residue_starts[1:])
    ):
        ref_space_uid[start:stop] = uid
    atom_array.set_annotation("ref_space_uid", ref_space_uid)
    return atom_array


def _split_cross_molecule_type_entities(atom_array: AtomArray) -> AtomArray:
    """Give Protenix distinct numeric entities when an OF3 ID spans mol types."""
    source_entity_ids = np.asarray(atom_array.entity_id, dtype=np.int64)
    atom_array.set_annotation(
        "openfold3_entity_id",
        source_entity_ids.copy(),
    )
    internal_entity_ids = source_entity_ids.astype(str)
    next_entity_id = int(max(np.max(source_entity_ids, initial=0), 0)) + 1
    for source_entity_id in np.unique(source_entity_ids):
        entity_mask = source_entity_ids == source_entity_id
        molecule_types = sorted(
            np.unique(atom_array.molecule_type_id[entity_mask]).astype(int)
        )
        if len(molecule_types) <= 1:
            continue
        # Keep the polymer (or otherwise lowest OF3 molecule type) on the
        # source ID and allocate deterministic numeric IDs to the remaining
        # molecule types. Numeric strings are required by Protenix CSV loading.
        for molecule_type_id in molecule_types[1:]:
            type_mask = entity_mask & (
                atom_array.molecule_type_id == molecule_type_id
            )
            internal_entity_ids[type_mask] = str(next_entity_id)
            next_entity_id += 1
    _set_string_annotation(
        atom_array,
        "label_entity_id",
        internal_entity_ids,
    )
    return atom_array


def _validate_npz_arrays(arrays: Mapping[str, np.ndarray], path: Path) -> int:
    """Validate the machine-readable OpenFold3 structure schema."""
    missing = sorted(_REQUIRED_NPZ_FIELDS - set(arrays))
    if missing:
        raise OpenFold3DisorderedError(
            f"{path}: missing required NPZ fields: {', '.join(missing)}"
        )

    coord = np.asarray(arrays["coord"])
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise OpenFold3DisorderedError(
            f"{path}: coord must have shape (num_atoms, 3), got {coord.shape}"
        )
    atom_count = coord.shape[0]
    if atom_count == 0:
        raise OpenFold3DisorderedError(f"{path}: structure contains no atoms")

    atom_fields = _REQUIRED_NPZ_FIELDS - {"coord", "bonds"}
    bad_lengths = {
        field: np.asarray(arrays[field]).shape
        for field in atom_fields
        if np.asarray(arrays[field]).shape != (atom_count,)
    }
    if bad_lengths:
        raise OpenFold3DisorderedError(
            f"{path}: atom fields have inconsistent shapes: {bad_lengths}"
        )

    bonds = np.asarray(arrays["bonds"])
    if bonds.ndim != 2 or bonds.shape[1] != 3:
        raise OpenFold3DisorderedError(
            f"{path}: bonds must have shape (num_bonds, 3), got {bonds.shape}"
        )
    if bonds.dtype.kind not in "iu":
        raise OpenFold3DisorderedError(f"{path}: bonds must use an integer dtype")
    if bonds.size:
        if np.any(bonds[:, :2] >= atom_count):
            raise OpenFold3DisorderedError(f"{path}: bond atom index is out of range")
        if np.any(bonds[:, 2] > int(struc.BondType.AROMATIC)):
            raise OpenFold3DisorderedError(f"{path}: unsupported bond type")

    molecule_type_id = np.asarray(arrays["molecule_type_id"])
    invalid_types = sorted(
        set(molecule_type_id.astype(int).tolist())
        - set(OPENFOLD3_MOLECULE_TYPE_NAMES)
    )
    if invalid_types:
        raise OpenFold3DisorderedError(
            f"{path}: unsupported molecule_type_id values: {invalid_types}"
        )
    return atom_count


def read_openfold3_disordered_npz(
    npz_path: Union[str, Path],
) -> AtomArray:
    """Read one trusted OpenFold3 structure NPZ into a Protenix-compatible AtomArray.

    The OpenFold3 ``chain_id`` remains the primary chain identifier because the
    official dataset cache and its interface keys use this ID.  Author and label
    asym IDs remain separate annotations.  Coordinates are supervised only when
    they are finite and have positive occupancy; all unsupervised coordinates are
    normalized to zero.

    Args:
        npz_path: Path to an OpenFold3 structure ``.npz`` file.

    Returns:
        AtomArray containing the retained and normalized source annotations
        and bond graph.

    Raises:
        OpenFold3DisorderedError: If the NPZ schema or values are invalid.
    """
    path = Path(npz_path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise OpenFold3DisorderedError(f"cannot read OpenFold3 NPZ {path}: {exc}") from exc

    atom_count = _validate_npz_arrays(arrays, path)
    atom_array = AtomArray(atom_count)

    _set_string_annotation(atom_array, "chain_id", arrays["chain_id"])
    atom_array.res_id = np.asarray(arrays["res_id"], dtype=np.int64)
    atom_array.ins_code = _as_string_array(arrays["ins_code"])
    atom_array.res_name = _as_string_array(arrays["res_name"])
    atom_array.hetero = np.asarray(arrays["hetero"], dtype=bool)
    atom_array.atom_name = _as_string_array(arrays["atom_name"])
    atom_array.element = np.char.upper(_as_string_array(arrays["element"]))

    _set_string_annotation(atom_array, "label_asym_id", arrays["label_asym_id"])
    _set_string_annotation(atom_array, "auth_asym_id", arrays["auth_asym_id"])
    atom_array.set_annotation(
        "entity_id", np.asarray(arrays["entity_id"], dtype=np.int64)
    )
    _set_string_annotation(
        atom_array,
        "label_entity_id",
        np.asarray(arrays["entity_id"], dtype=np.int64).astype(str),
    )
    atom_array.set_annotation(
        "molecule_type_id",
        np.asarray(arrays["molecule_type_id"], dtype=np.int64),
    )
    atom_array.set_annotation(
        "occupancy", np.asarray(arrays["occupancy"], dtype=np.float64)
    )
    atom_array.set_annotation(
        "charge", np.asarray(arrays["charge"], dtype=np.int64)
    )
    atom_array.set_annotation("label_seq_id", atom_array.res_id.copy())
    atom_array.set_annotation("auth_seq_id", atom_array.res_id.copy())
    atom_array.set_annotation("label_alt_id", np.full(atom_count, ".", dtype="<U1"))

    coord = np.asarray(arrays["coord"], dtype=np.float32)
    finite = np.isfinite(coord).all(axis=1)
    occupancy = atom_array.occupancy
    is_resolved = finite & np.isfinite(occupancy) & (occupancy > 0)
    atom_array.coord = np.array(coord, copy=True)
    atom_array.coord[~is_resolved] = 0.0
    atom_array.set_annotation("is_resolved", is_resolved)

    bonds = np.asarray(arrays["bonds"], dtype=np.uint32)
    atom_array.bonds = struc.BondList(atom_count, np.array(bonds, copy=True))

    chain_ids = _chain_ids_in_order(atom_array)
    if not chain_ids:
        raise OpenFold3DisorderedError(f"{path}: structure contains no chains")
    for chain_id in chain_ids:
        chain_mask = atom_array.chain_id == chain_id
        for name in (
            "label_asym_id",
            "auth_asym_id",
            "label_entity_id",
            "molecule_type_id",
        ):
            if np.unique(atom_array.get_annotation(name)[chain_mask]).size != 1:
                raise OpenFold3DisorderedError(
                    f"{path}: {name} is not constant within chain_id {chain_id!r}"
                )
    atom_array = _select_alternate_components(atom_array)
    # Match the standard Protenix mmCIF preprocessing contract.  Unknown
    # UNX/UNL components cannot be tokenized because their chemical element is
    # genuinely unspecified; ASX/GLX are deterministically normalized by the
    # same helper rather than inventing an element for them here.
    atom_array = Filter.remove_element_X(atom_array)
    return _split_cross_molecule_type_entities(atom_array)


def _residue_names_for_chain(
    atom_array: AtomArray,
    chain_id: str,
) -> np.ndarray:
    """Return one residue name per residue for a chain."""
    chain = atom_array[atom_array.chain_id == chain_id]
    starts = struc.get_residue_starts(chain)
    return chain.res_name[starts]


def _residue_to_one_letter(res_name: str, molecule_type_id: int) -> str:
    """Convert a residue name with deterministic standard-residue fast paths."""
    if molecule_type_id == 0:
        if res_name in _PROTEIN_THREE_TO_ONE:
            return _PROTEIN_THREE_TO_ONE[res_name]
        unknown = "X"
    elif molecule_type_id == 1:
        if res_name in RNA_STD_RESIDUES:
            return res_name if res_name != "N" else "N"
        unknown = "N"
    elif molecule_type_id == 2:
        if res_name in DNA_STD_RESIDUES:
            return res_name[1:] if res_name != "DN" else "N"
        unknown = "N"
    else:
        raise OpenFold3DisorderedError(
            f"cannot derive a polymer sequence for molecule_type_id={molecule_type_id}"
        )

    one_letter = ccd.get_one_letter_code(res_name)
    if one_letter is None or len(one_letter) != 1:
        return unknown
    return one_letter


def _get_polymer_metadata(
    atom_array: AtomArray,
    *,
    allow_inconsistent_entity_copies: bool = False,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Recover canonical polymer sequences and Protenix entity polymer types."""
    chain_ids = _chain_ids_in_order(atom_array)
    sequences: dict[str, str] = {}
    entity_poly_type: dict[str, str] = {}
    sequences_by_chain_id: dict[str, str] = {}

    for chain_id in chain_ids:
        chain_mask = atom_array.chain_id == chain_id
        entity_id = str(atom_array.label_entity_id[chain_mask][0])
        molecule_type_id = int(atom_array.molecule_type_id[chain_mask][0])
        if molecule_type_id == 3:
            continue

        if entity_id in entity_poly_type:
            if entity_poly_type[entity_id] != OPENFOLD3_TO_ENTITY_POLY_TYPE[molecule_type_id]:
                raise OpenFold3DisorderedError(
                    f"entity_id {entity_id!r} has inconsistent molecule types"
                )
        else:
            entity_poly_type[entity_id] = OPENFOLD3_TO_ENTITY_POLY_TYPE[
                molecule_type_id
            ]

        residue_names = _residue_names_for_chain(atom_array, chain_id)
        sequence = "".join(
            _residue_to_one_letter(str(res_name), molecule_type_id)
            for res_name in residue_names
        )
        sequences_by_chain_id[chain_id] = sequence
        previous = sequences.setdefault(entity_id, sequence)
        if previous != sequence and not allow_inconsistent_entity_copies:
            raise OpenFold3DisorderedError(
                f"entity_id {entity_id!r} has inconsistent sequences across chain copies"
            )
    return sequences, entity_poly_type, sequences_by_chain_id


def _resolve_alignment_representatives(
    *,
    cache_chains: Mapping[str, Mapping[str, Any]],
    sequences_by_chain_id: Mapping[str, str],
    canonical_sequences_by_representative: Optional[Mapping[str, str]],
) -> tuple[dict[str, Optional[str]], dict[str, str], list[str]]:
    """Resolve chain MSAs by exact observed-query matching when mappings swapped.

    Most entries retain the cache representative verbatim.  For the few
    audited structures where near-identical predicted chains were mapped onto
    the wrong experimental copies, the builder supplies the query sequences
    for this entry's representatives.  A protein/RNA representative is
    reassigned only when exactly one same-molecule candidate matches the
    physical chain sequence.  Zero or multiple matches leave the cache mapping
    unchanged and explicitly mark that physical chain as query-only.
    """
    resolved = {
        chain_id: chain_data.get("alignment_representative_id")
        for chain_id, chain_data in cache_chains.items()
    }
    if canonical_sequences_by_representative is None:
        return resolved, {}, []

    canonical = {
        str(representative): str(sequence)
        for representative, sequence in (
            canonical_sequences_by_representative.items()
        )
    }
    representatives_by_molecule_type: dict[str, set[str]] = {}
    for chain_data in cache_chains.values():
        representative = chain_data.get("alignment_representative_id")
        if representative is None:
            continue
        representative = str(representative)
        if representative not in canonical:
            raise OpenFold3DisorderedError(
                f"missing canonical query for representative {representative!r}"
            )
        representatives_by_molecule_type.setdefault(
            str(chain_data["molecule_type"]),
            set(),
        ).add(representative)

    query_only_chain_ids: set[str] = set()
    for chain_id, observed_sequence in sequences_by_chain_id.items():
        chain_data = cache_chains[chain_id]
        molecule_type = str(chain_data["molecule_type"])
        original = chain_data.get("alignment_representative_id")
        if original is None:
            if molecule_type in {"PROTEIN", "RNA"}:
                query_only_chain_ids.add(chain_id)
            continue
        original = str(original)
        candidates = sorted(
            representative
            for representative in representatives_by_molecule_type.get(
                molecule_type,
                set(),
            )
            if canonical[representative] == observed_sequence
        )
        if molecule_type in {"PROTEIN", "RNA"}:
            if len(candidates) != 1:
                query_only_chain_ids.add(chain_id)
                continue
            resolved[chain_id] = candidates[0]
            continue

        # Preserve the pre-existing fail-closed behavior for other molecule
        # types, which are outside the protein/RNA query-only contract.
        if canonical[original] == observed_sequence:
            continue
        if not candidates:
            raise OpenFold3DisorderedError(
                f"chain {chain_id!r} sequence does not exactly match any "
                "same-molecule alignment representative in its entry"
            )
        resolved[chain_id] = candidates[0]

    # Build canonical entity sequences only after every physical chain has
    # been resolved.  Deriving them from the cache's original representatives
    # before the loop above would swap the entity sequences alongside an
    # audited distinct-entity A/B representative repair.
    entity_sequences: dict[str, str] = {}
    query_only_entity_ids = {
        str(cache_chains[chain_id]["entity_id"])
        for chain_id in query_only_chain_ids
    }
    for chain_id, chain_data in cache_chains.items():
        entity_id = str(chain_data["entity_id"])
        if entity_id in query_only_entity_ids:
            continue
        representative = resolved[chain_id]
        if representative is None:
            continue
        representative = str(representative)
        sequence = canonical[representative]
        previous = entity_sequences.setdefault(entity_id, sequence)
        if previous != sequence:
            raise OpenFold3DisorderedError(
                f"entity_id {entity_id!r} has inconsistent canonical "
                "alignment queries after representative resolution"
            )
    return resolved, entity_sequences, sorted(query_only_chain_ids)


def _chain_metadata(atom_array: AtomArray) -> dict[str, dict[str, Any]]:
    """Return invariant OpenFold3 metadata for every NPZ chain."""
    metadata = {}
    for chain_id in _chain_ids_in_order(atom_array):
        chain_mask = atom_array.chain_id == chain_id
        atom_index = int(np.flatnonzero(chain_mask)[0])
        metadata[chain_id] = {
            "label_asym_id": str(atom_array.label_asym_id[atom_index]),
            "auth_asym_id": str(atom_array.auth_asym_id[atom_index]),
            "entity_id": int(atom_array.entity_id[atom_index]),
            "molecule_type": OPENFOLD3_MOLECULE_TYPE_NAMES[
                int(atom_array.molecule_type_id[atom_index])
            ],
            "molecule_type_id": int(atom_array.molecule_type_id[atom_index]),
        }
    return metadata


def _normalize_cache_chains(
    cache_entry: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Normalize cache chain keys while retaining exact case-sensitive values."""
    chains = cache_entry.get("chains")
    if not isinstance(chains, Mapping):
        raise OpenFold3DisorderedError("cache_entry.chains must be a mapping")
    normalized = {str(chain_id): value for chain_id, value in chains.items()}
    if len(normalized) != len(chains):
        raise OpenFold3DisorderedError("cache_entry.chains has duplicate normalized IDs")
    if not all(isinstance(value, Mapping) for value in normalized.values()):
        raise OpenFold3DisorderedError("every cache_entry.chains value must be a mapping")
    return normalized


def _repair_known_3nzi_chain_ids(
    atom_array: AtomArray,
    *,
    pdb_id: str,
    cache_entry: Mapping[str, Any],
) -> AtomArray:
    """Repair the one audited OF3 chain-map omission without a broad heuristic.

    The official 3nzi NPZ retains three predicted peptide IDs (D/E/F) instead
    of the cache IDs (4/6/2).  The mapping is uniquely fixed by the peptide
    sequence and the resolved 5-A contact graph.  Every invariant is checked
    before mutation so a changed or corrupt source fails closed.
    """
    cache_chains = _normalize_cache_chains(cache_entry)
    npz_chain_ids = set(_chain_ids_in_order(atom_array))
    cache_chain_ids = set(cache_chains)
    extra = npz_chain_ids - cache_chain_ids
    missing = cache_chain_ids - npz_chain_ids
    # Cache-only chains are expected after OpenFold3 pocket filtering.
    if not extra:
        return atom_array
    if (
        pdb_id != "3nzi"
        or extra != set(_KNOWN_3NZI_CHAIN_REMAP)
        or missing != set(_KNOWN_3NZI_CHAIN_REMAP.values())
    ):
        raise OpenFold3DisorderedError(
            "NPZ contains chain IDs absent from the official cache outside "
            "the audited 3nzi repair: "
            f"pdb_id={pdb_id!r}, extra={sorted(extra)}, "
            f"missing={sorted(missing)}"
        )

    for source_chain, target_chain in _KNOWN_3NZI_CHAIN_REMAP.items():
        chain_mask = atom_array.chain_id == source_chain
        sequence = "".join(
            _residue_to_one_letter(str(res_name), 0)
            for res_name in _residue_names_for_chain(atom_array, source_chain)
        )
        source_metadata = {
            "label_asym_id": str(atom_array.label_asym_id[chain_mask][0]),
            "auth_asym_id": str(atom_array.auth_asym_id[chain_mask][0]),
            "entity_id": int(atom_array.entity_id[chain_mask][0]),
            "molecule_type_id": int(
                atom_array.molecule_type_id[chain_mask][0]
            ),
        }
        if (
            sequence != _KNOWN_3NZI_PEPTIDE_SEQUENCE
            or source_metadata
            != {
                "label_asym_id": source_chain,
                "auth_asym_id": source_chain,
                "entity_id": -1,
                "molecule_type_id": 0,
            }
        ):
            raise OpenFold3DisorderedError(
                f"3nzi chain {source_chain!r} no longer matches the audited "
                "unmapped peptide"
            )
        target_metadata = cache_chains[target_chain]
        if (
            str(target_metadata.get("label_asym_id")) != "B"
            or str(target_metadata.get("auth_asym_id")) != "B"
            or int(target_metadata.get("entity_id")) != 2
            or str(target_metadata.get("molecule_type")) != "PROTEIN"
            or str(target_metadata.get("alignment_representative_id"))
            != "3nzi_B"
        ):
            raise OpenFold3DisorderedError(
                f"3nzi cache chain {target_chain!r} no longer matches the "
                "audited peptide metadata"
            )

    _entity_pairs, actual_chain_pairs = MMCIFParser.find_interfaces(atom_array)
    mapped_actual_pairs = {
        frozenset(
            (
                _KNOWN_3NZI_CHAIN_REMAP.get(chain_1, chain_1),
                _KNOWN_3NZI_CHAIN_REMAP.get(chain_2, chain_2),
            )
        )
        for chain_1, chain_2 in actual_chain_pairs
    }
    cache_interface_pairs = {
        frozenset(
            _parse_cache_interface_id(
                str(interface_id),
                cache_chain_ids,
            )
        )
        for interface_id in cache_entry.get("interfaces", {})
    }
    if mapped_actual_pairs != cache_interface_pairs:
        raise OpenFold3DisorderedError(
            "3nzi resolved contact graph no longer uniquely matches the "
            "audited cache interfaces"
        )

    for source_chain, target_chain in _KNOWN_3NZI_CHAIN_REMAP.items():
        chain_mask = atom_array.chain_id == source_chain
        target_metadata = cache_chains[target_chain]
        atom_array.chain_id[chain_mask] = target_chain
        atom_array.label_asym_id[chain_mask] = str(
            target_metadata["label_asym_id"]
        )
        atom_array.auth_asym_id[chain_mask] = str(
            target_metadata["auth_asym_id"]
        )
        atom_array.entity_id[chain_mask] = int(target_metadata["entity_id"])
        atom_array.label_entity_id[chain_mask] = str(
            target_metadata["entity_id"]
        )
        atom_array.molecule_type_id[chain_mask] = 0
    return atom_array


def _validate_cache_chains(
    atom_array: AtomArray,
    cache_entry: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Validate cache metadata for every chain retained in the final NPZ.

    The disordered cache is inherited from the experimental parent structure
    before non-protein chains outside the retained protein pockets are
    removed.  Cache-only chains are therefore expected and are not trainable
    datapoints.  Conversely, every chain present in the NPZ must still have
    matching cache metadata.
    """
    npz_metadata = _chain_metadata(atom_array)
    cache_chains = _normalize_cache_chains(cache_entry)
    uncached_npz_chains = sorted(set(npz_metadata) - set(cache_chains))
    if uncached_npz_chains:
        raise OpenFold3DisorderedError(
            "NPZ contains chain IDs absent from the official cache after "
            f"known repairs: {uncached_npz_chains}"
        )

    for chain_id, expected in npz_metadata.items():
        cached = cache_chains[chain_id]
        for field in (
            "label_asym_id",
            "auth_asym_id",
            "entity_id",
            "molecule_type",
        ):
            if field not in cached:
                raise OpenFold3DisorderedError(
                    f"cache chain {chain_id!r} is missing {field!r}"
                )
            actual = cached[field]
            if field == "entity_id":
                try:
                    actual = int(actual)
                except (TypeError, ValueError) as exc:
                    raise OpenFold3DisorderedError(
                        f"cache chain {chain_id!r} has invalid entity_id"
                    ) from exc
            else:
                actual = str(actual)
            if actual != expected[field]:
                raise OpenFold3DisorderedError(
                    f"cache chain {chain_id!r} {field} mismatch: "
                    f"cache={actual!r}, NPZ={expected[field]!r}"
                )

        cluster_id = cached.get("cluster_id")
        try:
            cluster_size = int(cached.get("cluster_size"))
        except (TypeError, ValueError) as exc:
            raise OpenFold3DisorderedError(
                f"cache chain {chain_id!r} has invalid cluster_size"
            ) from exc
        if cluster_id is None or cluster_size <= 0:
            raise OpenFold3DisorderedError(
                f"cache chain {chain_id!r} must have cluster_id and positive cluster_size"
            )
        alignment_id = cached.get("alignment_representative_id")
        if alignment_id is not None and not isinstance(alignment_id, str):
            raise OpenFold3DisorderedError(
                f"cache chain {chain_id!r} has invalid alignment_representative_id"
            )
    return {
        chain_id: cache_chains[chain_id]
        for chain_id in _chain_ids_in_order(atom_array)
    }


def _parse_cache_interface_id(
    interface_id: str,
    chain_ids: set[str],
) -> tuple[str, str]:
    """Resolve an official ``chain1_chain2`` interface key without lowercasing IDs."""
    matches = [
        (chain_1, chain_2)
        for chain_1 in chain_ids
        for chain_2 in chain_ids
        if chain_1 != chain_2 and f"{chain_1}_{chain_2}" == interface_id
    ]
    if len(matches) != 1:
        raise OpenFold3DisorderedError(
            f"cache interface ID {interface_id!r} does not identify exactly two chains"
        )
    return matches[0]


def _validate_cache_interfaces(
    cache_entry: Mapping[str, Any],
    present_chain_ids: set[str],
    actual_chain_pairs: list[tuple[str, str]],
    chain_entity_ids: Mapping[str, str],
) -> list[tuple[str, str, str, Mapping[str, Any]]]:
    """Validate cache interfaces that remain sampleable in the predicted NPZ.

    The cache interface IDs come from the experimental parent structure.  In a
    disordered NPZ, equivalent copies of an entity can contact different chain
    IDs after the protein coordinates are replaced by an AF-Multimer
    prediction.  Protenix samples interfaces by entity pair and then chooses an
    actually contacting chain pair from ``entity_pair_to_chain_pairs`` at load
    time.  Therefore a cache row remains valid when its *entity pair* has a
    resolved 5-A interface in the NPZ; the exact cache chain pair need not be
    the contacting copy.
    """
    interfaces = cache_entry.get("interfaces")
    if not isinstance(interfaces, Mapping):
        raise OpenFold3DisorderedError("cache_entry.interfaces must be a mapping")
    cache_chains = _normalize_cache_chains(cache_entry)
    cache_chain_ids = set(cache_chains)

    actual_entity_pairs = set()
    for chain_1, chain_2 in actual_chain_pairs:
        if (
            chain_1 not in cache_chains
            or chain_2 not in cache_chains
            or chain_1 not in chain_entity_ids
            or chain_2 not in chain_entity_ids
        ):
            raise OpenFold3DisorderedError(
                "resolved NPZ interface contains a chain absent from the "
                f"official cache: {(chain_1, chain_2)!r}"
            )
        actual_entity_pairs.add(
            tuple(
                sorted(
                    (
                        str(chain_entity_ids[chain_1]),
                        str(chain_entity_ids[chain_2]),
                    )
                )
            )
        )

    seen_pairs = set()
    validated = []
    for raw_interface_id, interface_data in interfaces.items():
        interface_id = str(raw_interface_id)
        if not isinstance(interface_data, Mapping):
            raise OpenFold3DisorderedError(
                f"cache interface {interface_id!r} must be a mapping"
            )
        chain_1, chain_2 = _parse_cache_interface_id(
            interface_id,
            cache_chain_ids,
        )
        unordered_pair = frozenset((chain_1, chain_2))
        if unordered_pair in seen_pairs:
            raise OpenFold3DisorderedError(
                f"cache contains duplicate interface pair {interface_id!r}"
            )
        seen_pairs.add(unordered_pair)

        cluster_id = interface_data.get("cluster_id")
        try:
            cluster_size = int(interface_data.get("cluster_size"))
        except (TypeError, ValueError) as exc:
            raise OpenFold3DisorderedError(
                f"cache interface {interface_id!r} has invalid cluster_size"
            ) from exc
        if cluster_id is None or cluster_size <= 0:
            raise OpenFold3DisorderedError(
                f"cache interface {interface_id!r} must have cluster_id "
                "and positive cluster_size"
            )
        if not {chain_1, chain_2}.issubset(present_chain_ids):
            continue
        cache_entity_pair = tuple(
            sorted(
                (
                    str(chain_entity_ids[chain_1]),
                    str(chain_entity_ids[chain_2]),
                )
            )
        )
        if cache_entity_pair not in actual_entity_pairs:
            continue
        validated.append((interface_id, chain_1, chain_2, interface_data))
    return validated


def _annotate_atom_array(
    atom_array: AtomArray,
    resolution: float,
    entity_poly_type: dict[str, str],
    ccd_mols: Optional[dict[str, Chem.Mol]],
) -> AtomArray:
    """Add the annotations consumed by Protenix tokenization and featurization."""
    molecule_type = np.empty(len(atom_array), dtype="<U7")
    for molecule_type_id, protenix_type in OPENFOLD3_TO_PROTENIX_MOL_TYPE.items():
        molecule_type[atom_array.molecule_type_id == molecule_type_id] = protenix_type
    atom_array.set_annotation("mol_type", molecule_type)

    atom_array = AddAtomArrayAnnot.add_centre_atom_mask(atom_array)
    atom_array = AddAtomArrayAnnot.add_atom_mol_type_mask(atom_array)
    atom_array = AddAtomArrayAnnot.add_distogram_rep_atom_mask(atom_array)
    atom_array = AddAtomArrayAnnot.add_plddt_m_rep_atom_mask(atom_array)
    atom_array = AddAtomArrayAnnot.add_cano_seq_resname(atom_array)
    atom_array = AddAtomArrayAnnot.add_tokatom_idx(atom_array)
    atom_array = AddAtomArrayAnnot.add_modified_res_mask(atom_array)
    if atom_array.centre_atom_mask.sum() != atom_array.distogram_rep_atom_mask.sum():
        raise OpenFold3DisorderedError(
            "centre and distogram representative atom counts differ"
        )

    atom_array = AddAtomArrayAnnot.add_int_id(atom_array)
    atom_array = AddAtomArrayAnnot.find_equiv_mol_and_assign_ids(
        atom_array,
        entity_poly_type=entity_poly_type,
    )
    atom_array = _add_occurrence_ref_space_uid(atom_array)
    atom_array = AddAtomArrayAnnot.add_ref_info_and_res_perm(
        atom_array,
        ccd_mols=ccd_mols,
    )
    atom_array.set_annotation(
        "resolution", np.full(len(atom_array), resolution, dtype=np.float32)
    )
    return atom_array


def _chain_index_metadata(
    atom_array: AtomArray,
    sequences: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Build base index metadata for chains with resolved token centres."""
    metadata = {}
    for chain_id in _chain_ids_in_order(atom_array):
        chain_mask = atom_array.chain_id == chain_id
        resolved_centre = (
            chain_mask
            & atom_array.is_resolved
            & atom_array.centre_atom_mask.astype(bool)
        )
        if not np.any(resolved_centre):
            continue
        atom_index = int(np.flatnonzero(chain_mask)[0])
        entity_id = str(atom_array.label_entity_id[atom_index])
        molecule_type_id = int(atom_array.molecule_type_id[atom_index])
        if molecule_type_id == 0:
            descriptor = sequences[entity_id]
        elif molecule_type_id in (1, 2):
            descriptor = sequences[entity_id]
        else:
            descriptor = "_".join(
                _residue_names_for_chain(atom_array, chain_id).astype(str).tolist()
            )
        metadata[chain_id] = {
            "entity_id": entity_id,
            "chain_id": chain_id,
            "mol_type": OPENFOLD3_TO_INDEX_MOL_TYPE[molecule_type_id],
            "descriptor": descriptor,
        }
    return metadata


def _singleton_cluster_id(
    pdb_id: str,
    sample_type: str,
    chain_ids: tuple[str, ...],
) -> str:
    """Create a stable, globally scoped fallback singleton cluster ID."""
    members = ":".join(chain_ids)
    return f"openfold3-singleton:{pdb_id}:{sample_type}:{members}"


def _base_index_row(
    chain_1: Mapping[str, Any],
    chain_2: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create the chain-side columns shared by chain and interface index rows."""
    row = {
        "entity_1_id": chain_1["entity_id"],
        "chain_1_id": chain_1["chain_id"],
        "mol_1_type": chain_1["mol_type"],
        "cluster_1_id": chain_1["descriptor"],
        "_descriptor_1": chain_1["descriptor"],
    }
    if chain_2 is None:
        row.update(
            {
                "entity_2_id": "",
                "chain_2_id": "",
                "mol_2_type": "",
                "cluster_2_id": "",
                "_descriptor_2": "",
            }
        )
    else:
        row.update(
            {
                "entity_2_id": chain_2["entity_id"],
                "chain_2_id": chain_2["chain_id"],
                "mol_2_type": chain_2["mol_type"],
                "cluster_2_id": chain_2["descriptor"],
                "_descriptor_2": chain_2["descriptor"],
            }
        )
    return row


def _add_standard_index_types(
    atom_array: AtomArray,
    rows: list[dict[str, Any]],
) -> None:
    """Apply Protenix's standard sub-molecule and evaluation type annotations."""
    polymer_ligand_bonds = get_ligand_polymer_bond_mask(atom_array)
    if polymer_ligand_bonds.size:
        ligand_polymer_chain_ids = atom_array.chain_id[
            np.unique(polymer_ligand_bonds[:, :2])
        ]
    else:
        ligand_polymer_chain_ids = []

    for row in rows:
        cluster_ids = (row["cluster_1_id"], row["cluster_2_id"])
        row["cluster_1_id"] = row.pop("_descriptor_1")
        row["cluster_2_id"] = row.pop("_descriptor_2")
        if row["type"] == "chain":
            row["mol_type_group"] = f'intra_{row["mol_1_type"]}'
        else:
            row["mol_type_group"] = "_".join(
                sorted((row["mol_1_type"], row["mol_2_type"]))
            )
        MMCIFParser.add_sub_mol_type(
            atom_array,
            ligand_polymer_chain_ids,
            row,
        )
        MMCIFParser.add_eval_type(row)
        row["cluster_1_id"], row["cluster_2_id"] = cluster_ids


def _add_sampling_weights(rows: list[dict[str, Any]]) -> None:
    """Add official-cluster weights while neutralizing bioassembly copy multiplicity."""
    entity_pair_counts = Counter(
        (
            str(row["pdb_id"]),
            *sorted((str(row["entity_1_id"]), str(row["entity_2_id"]))),
        )
        for row in rows
    )
    for row in rows:
        weighted_chain_count = _ALPHA_BY_MOL_TYPE[row["mol_1_type"]]
        if row["mol_2_type"]:
            weighted_chain_count += _ALPHA_BY_MOL_TYPE[row["mol_2_type"]]
        entity_pair_key = (
            str(row["pdb_id"]),
            *sorted((str(row["entity_1_id"]), str(row["entity_2_id"]))),
        )
        copy_count = entity_pair_counts[entity_pair_key]
        row["weights"] = (
            _BETA_BY_SAMPLE_TYPE[row["type"]]
            * weighted_chain_count
            / (int(row["cluster_size"]) + _WEIGHT_EPS)
            / copy_count
        )


def _make_sample_indices(
    *,
    atom_array: AtomArray,
    bioassembly: Mapping[str, Any],
    cache_entry: Optional[Mapping[str, Any]],
    cache_chains: Optional[dict[str, Mapping[str, Any]]],
    actual_chain_pairs: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Create Protenix chain/interface rows from cache clusters or singleton fallbacks."""
    pdb_id = str(bioassembly["pdb_id"])
    chain_metadata = _chain_index_metadata(atom_array, bioassembly["sequences"])
    rows: list[dict[str, Any]] = []

    for chain_id in _chain_ids_in_order(atom_array):
        chain = chain_metadata.get(chain_id)
        if chain is None:
            continue
        row = _base_index_row(chain, None)
        row["type"] = "chain"
        if cache_chains is None:
            cluster_id = _singleton_cluster_id(pdb_id, "chain", (chain_id,))
            cluster_size = 1
        else:
            cluster_id = str(cache_chains[chain_id]["cluster_id"])
            cluster_size = int(cache_chains[chain_id]["cluster_size"])
        row["cluster_1_id"] = cluster_id
        row["cluster_id"] = cluster_id
        row["cluster_size"] = cluster_size
        rows.append(row)

    if cache_entry is None:
        interfaces = [
            (
                _singleton_cluster_id(
                    pdb_id,
                    "interface",
                    tuple(sorted((chain_1, chain_2))),
                ),
                chain_1,
                chain_2,
                1,
            )
            for chain_1, chain_2 in actual_chain_pairs
        ]
    else:
        validated_interfaces = _validate_cache_interfaces(
            cache_entry,
            set(_chain_ids_in_order(atom_array)),
            actual_chain_pairs,
            {
                chain_id: str(
                    atom_array.label_entity_id[
                        np.flatnonzero(atom_array.chain_id == chain_id)[0]
                    ]
                )
                for chain_id in _chain_ids_in_order(atom_array)
            },
        )
        interfaces = [
            (
                str(interface_data["cluster_id"]),
                chain_1,
                chain_2,
                int(interface_data["cluster_size"]),
            )
            for _interface_id, chain_1, chain_2, interface_data in validated_interfaces
        ]

    for cluster_id, chain_1_id, chain_2_id, cluster_size in interfaces:
        chain_1 = chain_metadata.get(chain_1_id)
        chain_2 = chain_metadata.get(chain_2_id)
        if chain_1 is None or chain_2 is None:
            continue
        # MMCIFParser.find_interfaces() orders both the entity-pair key and its
        # corresponding chain pair by the (string) entity IDs.  Keep the CSV
        # row in the same order so Dataset can look up this entity pair
        # directly and replace it with an actually contacting chain copy.
        if str(chain_2["entity_id"]) < str(chain_1["entity_id"]):
            chain_1_id, chain_2_id = chain_2_id, chain_1_id
            chain_1, chain_2 = chain_2, chain_1
        entity_pair = (
            str(chain_1["entity_id"]),
            str(chain_2["entity_id"]),
        )
        entity_pair_to_chain_pairs = bioassembly.get(
            "entity_pair_to_chain_pairs"
        )
        if (
            not isinstance(entity_pair_to_chain_pairs, Mapping)
            or entity_pair not in entity_pair_to_chain_pairs
        ):
            raise OpenFold3DisorderedError(
                "interface index entity pair is absent from resolved "
                f"entity_pair_to_chain_pairs: {entity_pair!r}"
            )
        row = _base_index_row(chain_1, chain_2)
        row["type"] = "interface"
        row["cluster_1_id"] = (
            str(cache_chains[chain_1_id]["cluster_id"])
            if cache_chains is not None
            else _singleton_cluster_id(pdb_id, "chain", (chain_1_id,))
        )
        row["cluster_2_id"] = (
            str(cache_chains[chain_2_id]["cluster_id"])
            if cache_chains is not None
            else _singleton_cluster_id(pdb_id, "chain", (chain_2_id,))
        )
        row["cluster_id"] = cluster_id
        row["cluster_size"] = cluster_size
        rows.append(row)

    _add_standard_index_types(atom_array, rows)
    meta = {
        "pdb_id": pdb_id,
        "release_date": bioassembly["release_date"],
        "num_tokens": bioassembly["num_tokens"],
        "num_prot_chains": bioassembly["num_prot_chains"],
        "resolution": bioassembly["resolution"],
    }
    for row in rows:
        row.update(meta)
    _add_sampling_weights(rows)
    return rows


def build_openfold3_disordered_data(
    npz_path: Union[str, Path],
    *,
    pdb_id: Optional[str] = None,
    cache_entry: Optional[Mapping[str, Any]] = None,
    canonical_sequences_by_representative: Optional[
        Mapping[str, str]
    ] = None,
    ccd_mols: Optional[dict[str, Chem.Mol]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build Protenix indices and a bioassembly from one OpenFold3 disordered NPZ.

    Args:
        npz_path: Path to ``structures/<pdb_id>/<pdb_id>.npz``.
        pdb_id: Optional explicit entry ID. Defaults to the NPZ file stem.
        cache_entry: Optional official OpenFold3 training-cache entry. When
            supplied, chain metadata are checked exactly and official cluster
            IDs, sizes, alignments, release date, and resolution are retained.
        canonical_sequences_by_representative: Optional exact query sequences
            used only to repair audited swaps among near-identical predicted
            chain copies. Normal entries do not need this mapping.
        ccd_mols: Optional custom CCD molecule mapping passed to Protenix
            reference feature construction.

    Returns:
        A pair ``(sample_indices, bioassembly)`` ready for serialization by a
        data-preparation script.

    Raises:
        OpenFold3DisorderedError: If the NPZ or cache metadata are inconsistent.
    """
    path = Path(npz_path)
    entry_id = str(pdb_id if pdb_id is not None else path.stem)
    if not entry_id:
        raise OpenFold3DisorderedError("pdb_id cannot be empty")

    atom_array = read_openfold3_disordered_npz(path)
    if cache_entry is None:
        cache_chains = None
    else:
        atom_array = _repair_known_3nzi_chain_ids(
            atom_array,
            pdb_id=entry_id,
            cache_entry=cache_entry,
        )
        cache_chains = _validate_cache_chains(atom_array, cache_entry)

    (
        sequences,
        entity_poly_type,
        sequences_by_chain_id,
    ) = _get_polymer_metadata(
        atom_array,
        allow_inconsistent_entity_copies=(
            canonical_sequences_by_representative is not None
        ),
    )

    if cache_entry is None:
        release_date = ""
        resolution = -1.0
        alignment_representative_id_by_chain_id = {
            chain_id: None for chain_id in _chain_ids_in_order(atom_array)
        }
        openfold3_query_only_chain_ids: list[str] = []
    else:
        release_date = str(cache_entry.get("release_date", ""))
        raw_resolution = cache_entry.get("resolution")
        try:
            resolution = (
                -1.0
                if raw_resolution is None
                else float(raw_resolution)
            )
        except (TypeError, ValueError) as exc:
            raise OpenFold3DisorderedError(
                "cache_entry.resolution must be numeric or null"
            ) from exc
        # The official cache uses JSON NaN for 124 entries whose experimental
        # resolution is unknown.  Protenix already represents unknown
        # resolution as -1.0, as it does for a JSON null above.
        if not np.isfinite(resolution):
            resolution = -1.0
        (
            alignment_representative_id_by_chain_id,
            canonical_entity_sequences,
            openfold3_query_only_chain_ids,
        ) = _resolve_alignment_representatives(
            cache_chains=cache_chains,
            sequences_by_chain_id=sequences_by_chain_id,
            canonical_sequences_by_representative=(
                canonical_sequences_by_representative
            ),
        )
        sequences.update(canonical_entity_sequences)

    atom_array = _annotate_atom_array(
        atom_array,
        resolution=resolution,
        entity_poly_type=entity_poly_type,
        ccd_mols=ccd_mols,
    )
    entity_pair_to_chain_pairs, actual_chain_pairs = MMCIFParser.find_interfaces(
        atom_array
    )
    token_array = AtomArrayTokenizer(atom_array).get_token_array()

    polymer_mask = atom_array.molecule_type_id != 3
    protein_mask = atom_array.molecule_type_id == 0
    residue_starts = struc.get_residue_starts(atom_array)
    alternate_group_sizes = atom_array.openfold3_alternate_group_size[
        residue_starts
    ]
    bioassembly = {
        "pdb_id": entry_id,
        "assembly_id": None,
        "release_date": release_date,
        "resolution": resolution,
        "sequences": sequences,
        "sequences_by_chain_id": sequences_by_chain_id,
        "entity_poly_type": entity_poly_type,
        "atom_array": atom_array,
        "token_array": token_array,
        "num_tokens": int(atom_array.centre_atom_mask.sum()),
        "num_prot_chains": int(np.unique(atom_array.chain_id[protein_mask]).size),
        "num_assembly_polymer_chains": int(
            np.unique(atom_array.chain_id[polymer_mask]).size
        ),
        "openfold3_alternate_component_site_count": int(
            np.count_nonzero(alternate_group_sizes > 1)
        ),
        "openfold3_discarded_alternate_component_count": int(
            np.maximum(alternate_group_sizes - 1, 0).sum()
        ),
        "entity_pair_to_chain_pairs": entity_pair_to_chain_pairs,
        "alignment_representative_id_by_chain_id": (
            alignment_representative_id_by_chain_id
        ),
        "openfold3_query_only_chain_ids": openfold3_query_only_chain_ids,
        "msa_features": None,
        "template_features": None,
        "openfold3_cache_backed": cache_entry is not None,
    }
    sample_indices = _make_sample_indices(
        atom_array=atom_array,
        bioassembly=bioassembly,
        cache_entry=cache_entry,
        cache_chains=cache_chains,
        actual_chain_pairs=actual_chain_pairs,
    )
    return sample_indices, bioassembly
