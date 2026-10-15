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

"""Strict Gemmi reader for predicted standard-protein monomers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import gemmi
import numpy as np
from biotite.structure import AtomArray, BondList, BondType

from protenix.data.constants import (
    PROTEIN_COMMON_ONE_TO_THREE,
    PROTEIN_TYPES_ONE_LETTER,
    RES_ATOMS_DICT,
)
from protenix.data.core.predicted_protein_fast_path import (
    FastPathNotApplicable,
    _get_standard_protein_template_cache,
)


_MAX_ATOM_COUNT = 1_000_000
_STANDARD_RESIDUE_NAMES = np.asarray(
    sorted(PROTEIN_COMMON_ONE_TO_THREE.values())
)
_SUPPORTED_ELEMENTS = np.asarray(["C", "N", "O", "S"])

_RESIDUE_NAME_BY_CODE = np.asarray(
    [
        PROTEIN_COMMON_ONE_TO_THREE[one_letter]
        for one_letter in PROTEIN_TYPES_ONE_LETTER
    ]
)
_SORTED_RESIDUE_CODE = np.argsort(_RESIDUE_NAME_BY_CODE)
_SORTED_RESIDUE_NAME = _RESIDUE_NAME_BY_CODE[_SORTED_RESIDUE_CODE]

_ATOM_KEY_SEPARATOR = "\x1f"
_ATOM_KEY_ITEMS = sorted(
    (
        f"{residue_name}{_ATOM_KEY_SEPARATOR}{atom_name}",
        local_index,
    )
    for residue_name in _RESIDUE_NAME_BY_CODE
    for atom_name, local_index in RES_ATOMS_DICT[residue_name].items()
)
_ATOM_KEYS = np.asarray([item[0] for item in _ATOM_KEY_ITEMS])
_LOCAL_ATOM_INDEX = np.asarray(
    [item[1] for item in _ATOM_KEY_ITEMS],
    dtype=np.int16,
)
_MAX_RESIDUE_ATOM_COUNT = max(
    len(RES_ATOMS_DICT[residue_name])
    for residue_name in _RESIDUE_NAME_BY_CODE
)

_REQUIRED_ATOM_SITE_FIELDS = frozenset(
    {
        "group_PDB",
        "id",
        "type_symbol",
        "label_atom_id",
        "label_alt_id",
        "label_comp_id",
        "label_asym_id",
        "label_entity_id",
        "label_seq_id",
        "pdbx_PDB_ins_code",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "occupancy",
        "B_iso_or_equiv",
        "auth_seq_id",
        "auth_comp_id",
        "auth_asym_id",
        "auth_atom_id",
        "pdbx_PDB_model_num",
    }
)

_STRING_FIELDS = (
    "group_PDB",
    "type_symbol",
    "label_atom_id",
    "label_alt_id",
    "label_comp_id",
    "label_asym_id",
    "label_entity_id",
    "pdbx_PDB_ins_code",
    "auth_comp_id",
    "auth_asym_id",
    "auth_atom_id",
)


def _reject(reason: str, detail: str | None = None) -> None:
    raise FastPathNotApplicable(reason, detail)


def _column(
    block: gemmi.cif.Block,
    category: str,
    field: str,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    tag = f"_{category}.{field}"
    values = block.find_values(tag)
    if len(values) == 0:
        _reject("missing_cif_field", tag)
    try:
        return np.asarray(values, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise FastPathNotApplicable(
            "invalid_cif_value",
            tag,
        ) from error


def _require_size(
    values: np.ndarray,
    atom_count: int,
    field: str,
) -> None:
    if values.ndim != 1 or values.size != atom_count:
        _reject("invalid_atom_site_column", field)


def _read_document(path: Path) -> gemmi.cif.Document:
    try:
        document = gemmi.cif.read_file(os.fspath(path))
    except (RuntimeError, ValueError) as error:
        raise FastPathNotApplicable(
            "invalid_cif",
            str(error),
        ) from error
    if len(document) != 1:
        _reject("cif_block_count", str(len(document)))
    return document


def _validate_categories(block: gemmi.cif.Block) -> gemmi.cif.Table:
    atom_site = block.find_mmcif_category("_atom_site.")
    if len(atom_site) == 0:
        _reject("empty_atom_site")
    if len(atom_site) > _MAX_ATOM_COUNT:
        _reject("atom_count_limit", str(len(atom_site)))

    atom_site_fields = {
        tag.removeprefix("_atom_site.") for tag in atom_site.tags
    }
    missing_fields = sorted(
        _REQUIRED_ATOM_SITE_FIELDS - atom_site_fields
    )
    if missing_fields:
        _reject("missing_cif_field", f"_atom_site.{missing_fields[0]}")

    if len(block.find_mmcif_category("_struct_conn.")) != 0:
        _reject("struct_conn_present")
    if len(block.find_mmcif_category("_cell.")) != 0:
        _reject("cell_present")
    if len(block.find_mmcif_category("_atom_sites.")) != 0:
        _reject("atom_sites_present")
    return atom_site


def _read_atom_site_columns(
    block: gemmi.cif.Block,
    atom_count: int,
) -> dict[str, np.ndarray]:
    columns = {
        field: _column(block, "atom_site", field)
        for field in _STRING_FIELDS
    }
    for field in ("label_seq_id", "auth_seq_id"):
        text_values = _column(block, "atom_site", field)
        columns[f"{field}_text"] = text_values
        try:
            columns[field] = text_values.astype(np.int64)
        except (TypeError, ValueError, OverflowError) as error:
            raise FastPathNotApplicable(
                "invalid_cif_value",
                f"_atom_site.{field}",
            ) from error
    columns["pdbx_PDB_model_num"] = _column(
        block,
        "atom_site",
        "pdbx_PDB_model_num",
        np.int32,
    )
    for field in ("Cartn_x", "Cartn_y", "Cartn_z"):
        columns[field] = _column(
            block,
            "atom_site",
            field,
            np.float32,
        )
    for field in ("occupancy", "B_iso_or_equiv"):
        columns[field] = _column(
            block,
            "atom_site",
            field,
            np.float64,
        )

    charge_values = block.find_values("_atom_site.pdbx_formal_charge")
    if len(charge_values) != 0:
        raw_charge = np.asarray(charge_values)
        missing_charge = np.isin(raw_charge, (".", "?"))
        charge = np.zeros(atom_count, dtype=np.int64)
        try:
            charge[~missing_charge] = raw_charge[
                ~missing_charge
            ].astype(np.int64)
        except (TypeError, ValueError, OverflowError) as error:
            raise FastPathNotApplicable(
                "invalid_cif_value",
                "_atom_site.pdbx_formal_charge",
            ) from error
        columns["charge"] = charge

    for field, values in columns.items():
        _require_size(values, atom_count, field)
    return columns


def _validate_atom_site(columns: dict[str, np.ndarray]) -> None:
    if not np.all(columns["pdbx_PDB_model_num"] == 1):
        _reject("model_count")
    if not np.all(columns["label_alt_id"] == "."):
        _reject("alternate_location")
    if not np.all(
        np.isin(columns["pdbx_PDB_ins_code"], (".", "?"))
    ):
        _reject("insertion_code")
    if not np.all(columns["group_PDB"] == "ATOM"):
        _reject("non_atom_record")

    if np.unique(columns["label_entity_id"]).size != 1:
        _reject("coordinate_entity_count")
    if np.unique(columns["label_asym_id"]).size != 1:
        _reject("coordinate_chain_count")
    if np.unique(columns["auth_asym_id"]).size != 1:
        _reject("author_chain_count")

    if not np.array_equal(
        columns["label_atom_id"],
        columns["auth_atom_id"],
    ):
        _reject("author_label_atom_mismatch")
    if not np.array_equal(
        columns["label_comp_id"],
        columns["auth_comp_id"],
    ):
        _reject("author_label_residue_mismatch")
    if not np.array_equal(
        columns["label_seq_id"],
        columns["auth_seq_id"],
    ):
        _reject("author_label_sequence_mismatch")

    if not np.all(
        np.isin(columns["label_comp_id"], _STANDARD_RESIDUE_NAMES)
    ):
        _reject("nonstandard_residue")
    if not np.all(
        np.isin(columns["type_symbol"], _SUPPORTED_ELEMENTS)
    ):
        _reject("unsupported_element")

    coordinate_values = np.column_stack(
        (
            columns["Cartn_x"],
            columns["Cartn_y"],
            columns["Cartn_z"],
        )
    )
    if not np.all(np.isfinite(coordinate_values)):
        _reject("nonfinite_coordinates")
    if not np.all(np.isfinite(columns["occupancy"])):
        _reject("nonfinite_occupancy")
    if not np.all(np.isfinite(columns["B_iso_or_equiv"])):
        _reject("nonfinite_b_factor")


def _residue_starts_and_count(
    columns: dict[str, np.ndarray],
) -> tuple[np.ndarray, int]:
    label_seq_id = columns["label_seq_id"]
    residue_starts = np.flatnonzero(
        np.concatenate(
            ([True], label_seq_id[1:] != label_seq_id[:-1])
        )
    )
    residue_ids = label_seq_id[residue_starts]
    expected_ids = np.arange(
        1,
        residue_starts.size + 1,
        dtype=np.int64,
    )
    if not np.array_equal(residue_ids, expected_ids):
        _reject("residue_ids_not_contiguous")

    residue_stops = np.append(residue_starts[1:], label_seq_id.size)
    expected_residue_ids = np.repeat(
        residue_ids,
        residue_stops - residue_starts,
    )
    if not np.array_equal(label_seq_id, expected_residue_ids):
        _reject("residue_rows_not_grouped")

    residue_names = columns["label_comp_id"][residue_starts]
    expected_residue_names = np.repeat(
        residue_names,
        residue_stops - residue_starts,
    )
    if not np.array_equal(
        columns["label_comp_id"],
        expected_residue_names,
    ):
        _reject("mixed_residue_names")
    return residue_starts, residue_starts.size


def _validate_polymer_metadata(
    block: gemmi.cif.Block,
    columns: dict[str, np.ndarray],
    residue_starts: np.ndarray,
    residue_count: int,
) -> None:
    entity_ids = _column(block, "entity_poly", "entity_id")
    polymer_types = _column(block, "entity_poly", "type")
    if entity_ids.size != 1 or polymer_types.size != 1:
        _reject("polymer_entity_count")
    polymer_type = gemmi.cif.as_string(str(polymer_types[0]))
    if polymer_type != "polypeptide(L)":
        _reject("unsupported_polymer_type")

    coordinate_entity_id = str(columns["label_entity_id"][0])
    if str(entity_ids[0]) != coordinate_entity_id:
        _reject("coordinate_entity_mismatch")

    sequence_entity_ids = _column(
        block,
        "entity_poly_seq",
        "entity_id",
    )
    sequence_numbers = _column(
        block,
        "entity_poly_seq",
        "num",
        np.int64,
    )
    sequence_residue_names = _column(
        block,
        "entity_poly_seq",
        "mon_id",
    )
    if not (
        sequence_entity_ids.size
        == sequence_numbers.size
        == sequence_residue_names.size
        == residue_count
    ):
        _reject("canonical_sequence_mismatch")
    if not np.all(sequence_entity_ids == coordinate_entity_id):
        _reject("sequence_entity_mismatch")
    if not np.array_equal(
        sequence_numbers,
        np.arange(1, residue_count + 1, dtype=np.int64),
    ):
        _reject("canonical_sequence_mismatch")
    if not np.array_equal(
        sequence_residue_names,
        columns["label_comp_id"][residue_starts],
    ):
        _reject("canonical_sequence_mismatch")


def _build_bonds(
    residue_names: np.ndarray,
    atom_names: np.ndarray,
    residue_ids: np.ndarray,
    residue_count: int,
) -> BondList:
    n_indices = np.flatnonzero(atom_names == "N")
    c_indices = np.flatnonzero(atom_names == "C")
    expected_residue_ids = np.arange(
        1,
        residue_count + 1,
        dtype=np.int64,
    )
    if (
        n_indices.size != residue_count
        or c_indices.size != residue_count
        or not np.array_equal(
            residue_ids[n_indices],
            expected_residue_ids,
        )
        or not np.array_equal(
            residue_ids[c_indices],
            expected_residue_ids,
        )
    ):
        _reject("peptide_connector_count")

    residue_code_position = np.searchsorted(
        _SORTED_RESIDUE_NAME,
        residue_names,
    )
    safe_residue_code_position = np.minimum(
        residue_code_position,
        _SORTED_RESIDUE_NAME.size - 1,
    )
    if not np.all(
        (residue_code_position < _SORTED_RESIDUE_NAME.size)
        & (
            _SORTED_RESIDUE_NAME[safe_residue_code_position]
            == residue_names
        )
    ):
        _reject("nonstandard_residue")
    residue_code = _SORTED_RESIDUE_CODE[residue_code_position]

    pair_keys = np.char.add(
        np.char.add(
            residue_names[residue_ids - 1],
            _ATOM_KEY_SEPARATOR,
        ),
        atom_names,
    )
    atom_key_position = np.searchsorted(_ATOM_KEYS, pair_keys)
    safe_atom_key_position = np.minimum(
        atom_key_position,
        _ATOM_KEYS.size - 1,
    )
    atom_key_match = (atom_key_position < _ATOM_KEYS.size) & (
        _ATOM_KEYS[safe_atom_key_position] == pair_keys
    )
    if not np.all(atom_key_match):
        first_missing = np.flatnonzero(~atom_key_match)[0]
        _reject("unexpected_atom", str(atom_names[first_missing]))

    local_atom_index = _LOCAL_ATOM_INDEX[atom_key_position].astype(
        np.int64,
        copy=False,
    )
    flat_atom_index = (
        (residue_ids - 1) * _MAX_RESIDUE_ATOM_COUNT
        + local_atom_index
    )
    if np.unique(flat_atom_index).size != atom_names.size:
        _reject("duplicate_atom")
    raw_index_by_local_atom = np.full(
        (residue_count, _MAX_RESIDUE_ATOM_COUNT),
        -1,
        dtype=np.int64,
    )
    raw_index_by_local_atom.flat[flat_atom_index] = np.arange(
        atom_names.size,
        dtype=np.int64,
    )

    template_cache = _get_standard_protein_template_cache()
    selected_bonds = template_cache.bond_table[residue_code]
    bond_mask = (
        np.arange(template_cache.bond_table.shape[1])[None, :]
        < template_cache.bond_counts[residue_code][:, None]
    )
    left_raw_index = np.take_along_axis(
        raw_index_by_local_atom,
        selected_bonds[:, :, 0],
        axis=1,
    )
    right_raw_index = np.take_along_axis(
        raw_index_by_local_atom,
        selected_bonds[:, :, 1],
        axis=1,
    )
    bond_mask &= (left_raw_index >= 0) & (right_raw_index >= 0)
    internal_bonds = selected_bonds[bond_mask].copy()
    internal_bonds[:, 0] = left_raw_index[bond_mask]
    internal_bonds[:, 1] = right_raw_index[bond_mask]

    if residue_count == 1:
        peptide_bonds = np.empty((0, 3), dtype=np.uint32)
    else:
        peptide_bonds = np.empty(
            (residue_count - 1, 3),
            dtype=np.uint32,
        )
        peptide_bonds[:, 0] = c_indices[:-1]
        peptide_bonds[:, 1] = n_indices[1:]
        peptide_bonds[:, 2] = int(BondType.SINGLE)
    bond_array = np.concatenate((peptide_bonds, internal_bonds))
    return BondList(atom_names.size, bond_array)


def _build_atom_array(
    columns: dict[str, np.ndarray],
    residue_starts: np.ndarray,
    residue_count: int,
) -> AtomArray:
    atom_count = columns["label_atom_id"].size
    atom_array = AtomArray(atom_count)
    atom_array.coord[:, 0] = columns["Cartn_x"]
    atom_array.coord[:, 1] = columns["Cartn_y"]
    atom_array.coord[:, 2] = columns["Cartn_z"]

    atom_array.chain_id = columns["label_asym_id"]
    atom_array.res_id = columns["label_seq_id"]
    atom_array.ins_code = np.full(atom_count, "")
    atom_array.res_name = columns["auth_comp_id"]
    atom_array.hetero = columns["group_PDB"] == "HETATM"
    atom_array.atom_name = columns["auth_atom_id"]
    atom_array.element = columns["type_symbol"]

    for name in (
        "label_asym_id",
        "label_entity_id",
        "auth_asym_id",
        "label_seq_id",
        "auth_seq_id",
    ):
        if name in ("label_seq_id", "auth_seq_id"):
            values = columns[f"{name}_text"]
        else:
            values = columns[name]
        atom_array.set_annotation(name, values)
    atom_array.set_annotation("occupancy", columns["occupancy"])
    atom_array.set_annotation("b_factor", columns["B_iso_or_equiv"])
    atom_array.set_annotation("label_alt_id", columns["label_alt_id"])
    if "charge" in columns:
        atom_array.set_annotation("charge", columns["charge"])

    atom_array.bonds = _build_bonds(
        columns["label_comp_id"][residue_starts],
        atom_array.atom_name,
        atom_array.res_id,
        residue_count,
    )
    return atom_array


def read_predicted_protein_atom_array_fast(
    cif_path: Union[str, Path],
) -> AtomArray:
    """Read a strictly eligible predicted monomer with Gemmi.

    Residue-internal bonds are generated from the cached 20-amino-acid CCD
    tables and adjacent peptide C--N bonds are added without invoking
    Biotite's generic residue-name connector.

    Args:
        cif_path: Path to an uncompressed ``.cif`` file.

    Returns:
        AtomArray carrying the same source annotations as the generic
        distillation reader, including the same predicted-protein bonds.

    Raises:
        FastPathNotApplicable: If the CIF is outside the deliberately narrow
            predicted-monomer contract.
    """
    path = Path(cif_path)
    if path.suffix != ".cif":
        _reject("unsupported_cif_path", os.fspath(path))

    document = _read_document(path)
    block = document.sole_block()
    atom_site = _validate_categories(block)
    columns = _read_atom_site_columns(block, len(atom_site))
    _validate_atom_site(columns)
    residue_starts, residue_count = _residue_starts_and_count(columns)
    _validate_polymer_metadata(
        block,
        columns,
        residue_starts,
        residue_count,
    )
    return _build_atom_array(
        columns,
        residue_starts,
        residue_count,
    )


__all__ = ["read_predicted_protein_atom_array_fast"]
