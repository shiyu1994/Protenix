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

"""Strict eligibility and atom layout for predicted-protein fast paths."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from protenix.data.constants import (
    PROTEIN_COMMON_ONE_TO_THREE,
    PROTEIN_TYPES_ONE_LETTER,
    RES_ATOMS_DICT,
)
from protenix.data.core import ccd


class FastPathNotApplicable(ValueError):
    """Indicate that an input must use the generic structure pipeline.

    ``reason`` is intentionally independent from the optional diagnostic detail,
    so callers can use it as a stable metrics key.
    """

    def __init__(self, reason: str, detail: str | None = None) -> None:
        """Initialize the exception.

        Args:
            reason: Stable machine-readable fallback reason.
            detail: Optional human-readable diagnostic.
        """
        self.reason = reason
        message = reason if detail is None else f"{reason}: {detail}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class PredictedProteinFastPathLayout:
    """Precomputed mapping from parsed atoms to the complete protein template."""

    entity_id: str
    chain_id: str
    canonical_sequence: str
    raw_to_template: np.ndarray
    template_to_raw: np.ndarray
    template_residue_index: np.ndarray
    template_atom_name: np.ndarray
    residue_offsets: np.ndarray
    missing_template_mask: np.ndarray
    terminal_oxt_template_index: int | None

    @property
    def terminal_oxt_missing(self) -> bool:
        """Whether the generic pipeline would need to synthesize terminal OXT."""
        if self.terminal_oxt_template_index is None:
            return False
        return bool(self.missing_template_mask[self.terminal_oxt_template_index])

    @property
    def template_atom_count(self) -> int:
        """Return the number of atoms in the completed template layout."""
        return int(self.template_to_raw.size)


@dataclass(frozen=True, slots=True)
class _StandardProteinTemplateCache:
    annotations: dict[str, np.ndarray]
    annotation_dtypes: dict[str, tuple[np.dtype, ...]]
    residue_offsets: np.ndarray
    bond_table: np.ndarray
    bond_counts: np.ndarray


_RESIDUE_NAME_BY_CODE = np.asarray(
    [
        PROTEIN_COMMON_ONE_TO_THREE[one_letter]
        for one_letter in PROTEIN_TYPES_ONE_LETTER
    ]
)
_STANDARD_RESIDUE_NAMES = np.sort(_RESIDUE_NAME_BY_CODE)

_CANONICAL_CODE_BY_ASCII = np.full(256, -1, dtype=np.int16)
_CANONICAL_ASCII = np.frombuffer(
    "".join(PROTEIN_TYPES_ONE_LETTER).encode("ascii"),
    dtype=np.uint8,
)
_CANONICAL_CODE_BY_ASCII[_CANONICAL_ASCII] = np.arange(
    len(PROTEIN_TYPES_ONE_LETTER),
    dtype=np.int16,
)

_ALL_ATOM_NAMES = np.asarray(
    sorted(
        {
            atom_name
            for residue_name in _RESIDUE_NAME_BY_CODE
            for atom_name in RES_ATOMS_DICT[residue_name]
        }
    )
)
_MAX_ATOM_NAME_LENGTH = max(len(atom_name) for atom_name in _ALL_ATOM_NAMES)
_MAX_TEMPLATE_ATOMS = max(
    len(RES_ATOMS_DICT[residue_name]) for residue_name in _RESIDUE_NAME_BY_CODE
)

_TEMPLATE_ATOM_RANK = np.full(
    (len(_RESIDUE_NAME_BY_CODE), len(_ALL_ATOM_NAMES)),
    -1,
    dtype=np.int16,
)
_TEMPLATE_ATOM_NAME = np.full(
    (len(_RESIDUE_NAME_BY_CODE), _MAX_TEMPLATE_ATOMS),
    "",
    dtype=f"<U{_MAX_ATOM_NAME_LENGTH}",
)
_INTERNAL_ATOM_COUNT = np.empty(len(_RESIDUE_NAME_BY_CODE), dtype=np.int16)
_TERMINAL_ATOM_COUNT = np.empty(len(_RESIDUE_NAME_BY_CODE), dtype=np.int16)

for _residue_code, _residue_name in enumerate(_RESIDUE_NAME_BY_CODE):
    _atom_names = tuple(
        sorted(
            RES_ATOMS_DICT[_residue_name],
            key=RES_ATOMS_DICT[_residue_name].__getitem__,
        )
    )
    if _atom_names[-1] != "OXT":
        raise RuntimeError(f"{_residue_name} template does not end in OXT")
    _atom_codes = np.searchsorted(_ALL_ATOM_NAMES, _atom_names)
    _TEMPLATE_ATOM_RANK[_residue_code, _atom_codes] = np.arange(
        len(_atom_names),
        dtype=np.int16,
    )
    _TEMPLATE_ATOM_NAME[_residue_code, : len(_atom_names)] = _atom_names
    _INTERNAL_ATOM_COUNT[_residue_code] = len(_atom_names) - 1
    _TERMINAL_ATOM_COUNT[_residue_code] = len(_atom_names)


def _reject(reason: str, detail: str | None = None) -> None:
    raise FastPathNotApplicable(reason, detail)


def _get_annotation(atom_array: Any, name: str, atom_count: int) -> np.ndarray:
    try:
        values = np.asarray(getattr(atom_array, name))
    except (AttributeError, KeyError):
        _reject("missing_annotation", name)
    if values.ndim != 1 or values.size != atom_count:
        _reject("invalid_annotation", name)
    return values


def _as_integer_ids(
    values: np.ndarray,
    annotation_name: str,
) -> np.ndarray:
    try:
        integer_values = values.astype(np.int64)
    except (TypeError, ValueError, OverflowError):
        _reject("invalid_residue_id", annotation_name)

    if values.dtype.kind in "fc":
        if not np.all(np.isfinite(values)) or not np.array_equal(
            values,
            integer_values,
        ):
            _reject("invalid_residue_id", annotation_name)
    return integer_values


def _canonical_codes(canonical_sequence: str) -> np.ndarray:
    if not isinstance(canonical_sequence, str):
        _reject("invalid_canonical_sequence")
    if not canonical_sequence:
        _reject("empty_canonical_sequence")
    try:
        sequence_ascii = np.frombuffer(
            canonical_sequence.encode("ascii"),
            dtype=np.uint8,
        )
    except UnicodeEncodeError:
        _reject("nonstandard_canonical_sequence")
    codes = _CANONICAL_CODE_BY_ASCII[sequence_ascii]
    if np.any(codes < 0):
        _reject("nonstandard_canonical_sequence")
    return codes.astype(np.intp, copy=False)


def _single_entity_metadata(
    sequences: Mapping[str, str],
    entity_poly_type: Mapping[str, str],
) -> tuple[str, str]:
    if not isinstance(entity_poly_type, Mapping) or len(entity_poly_type) != 1:
        _reject("polymer_entity_count")
    entity_key, polymer_type = next(iter(entity_poly_type.items()))
    if polymer_type != "polypeptide(L)":
        _reject("unsupported_polymer_type")
    if (
        not isinstance(sequences, Mapping)
        or len(sequences) != 1
        or entity_key not in sequences
    ):
        _reject("sequence_entity_mismatch")
    return str(entity_key), sequences[entity_key]


def _get_bond_array(atom_array: Any, atom_count: int) -> np.ndarray:
    bonds = getattr(atom_array, "bonds", None)
    if bonds is None:
        _reject("missing_bonds")
    bond_array = np.asarray(bonds.as_array())
    if bond_array.ndim != 2 or bond_array.shape[1] != 3:
        _reject("invalid_bonds")
    if bond_array.size > 0 and (
        np.any(bond_array[:, :2] < 0)
        or np.any(bond_array[:, :2] >= atom_count)
    ):
        _reject("invalid_bonds")
    return bond_array


def _encoded_bond_pairs(
    bond_endpoints: np.ndarray,
    atom_count: int,
) -> np.ndarray:
    if bond_endpoints.size == 0:
        return np.empty(0, dtype=np.int64)
    endpoints = bond_endpoints.astype(np.int64, copy=False)
    lower = np.minimum(endpoints[:, 0], endpoints[:, 1])
    upper = np.maximum(endpoints[:, 0], endpoints[:, 1])
    return lower * atom_count + upper


def prepare_predicted_protein_fast_path(
    atom_array: Any,
    sequences: Mapping[str, str],
    entity_poly_type: Mapping[str, str],
) -> PredictedProteinFastPathLayout:
    """Validate a predicted monomer and precompute its completed atom layout.

    The input is expected after MSE-to-MET conversion, arginine name fixing, and
    water/H/D/X filtering. No input array is modified.

    Args:
        atom_array: A Biotite ``AtomArray`` containing the filtered coordinates.
        sequences: Canonical sequence keyed by label entity ID.
        entity_poly_type: Polymer type keyed by label entity ID.

    Returns:
        A mapping from arbitrary raw atom order to the complete CCD-style layout.

    Raises:
        FastPathNotApplicable: If generic processing is required for any reason.
    """
    atom_count = len(atom_array)
    if atom_count == 0:
        _reject("empty_atom_array")

    entity_id, canonical_sequence = _single_entity_metadata(
        sequences,
        entity_poly_type,
    )
    canonical_code = _canonical_codes(canonical_sequence)
    residue_count = canonical_code.size

    chain_id = _get_annotation(
        atom_array,
        "chain_id",
        atom_count,
    ).astype(str, copy=False)
    label_entity_id = _get_annotation(
        atom_array,
        "label_entity_id",
        atom_count,
    ).astype(str, copy=False)
    res_id = _as_integer_ids(
        _get_annotation(atom_array, "res_id", atom_count),
        "res_id",
    )
    label_seq_id = _as_integer_ids(
        _get_annotation(atom_array, "label_seq_id", atom_count),
        "label_seq_id",
    )
    res_name = _get_annotation(
        atom_array,
        "res_name",
        atom_count,
    ).astype(str, copy=False)
    atom_name = _get_annotation(
        atom_array,
        "atom_name",
        atom_count,
    ).astype(str, copy=False)
    element = _get_annotation(
        atom_array,
        "element",
        atom_count,
    ).astype(str, copy=False)

    coordinate_entity_ids = np.unique(label_entity_id)
    if coordinate_entity_ids.size != 1:
        _reject("coordinate_entity_count")
    if coordinate_entity_ids[0] != entity_id:
        _reject("coordinate_entity_mismatch")

    coordinate_chain_ids = np.unique(chain_id)
    if coordinate_chain_ids.size != 1:
        _reject("coordinate_chain_count")

    if np.any(res_id < 1) or np.any(res_id > residue_count):
        _reject("residue_ids_not_contiguous")
    residue_index = res_id - 1
    if np.any(np.bincount(residue_index, minlength=residue_count) == 0):
        _reject("residue_ids_not_contiguous")
    if not np.array_equal(label_seq_id, res_id):
        _reject("label_seq_id_mismatch")

    sorted_residue_position = np.searchsorted(
        _STANDARD_RESIDUE_NAMES,
        res_name,
    )
    safe_residue_position = np.minimum(
        sorted_residue_position,
        _STANDARD_RESIDUE_NAMES.size - 1,
    )
    standard_residue_mask = (
        sorted_residue_position < _STANDARD_RESIDUE_NAMES.size
    ) & (_STANDARD_RESIDUE_NAMES[safe_residue_position] == res_name)
    if not np.all(standard_residue_mask):
        _reject("nonstandard_residue")

    expected_res_name = _RESIDUE_NAME_BY_CODE[canonical_code]
    if not np.all(res_name == expected_res_name[residue_index]):
        _reject("canonical_sequence_mismatch")

    if np.any(np.isin(element, ("H", "D", "X"))):
        _reject("unsupported_element")
    coord = np.asarray(atom_array.coord)
    if coord.shape != (atom_count, 3) or not np.all(np.isfinite(coord)):
        _reject("nonfinite_coordinates")

    atom_code = np.searchsorted(_ALL_ATOM_NAMES, atom_name)
    safe_atom_code = np.minimum(atom_code, _ALL_ATOM_NAMES.size - 1)
    known_atom_name = (atom_code < _ALL_ATOM_NAMES.size) & (
        _ALL_ATOM_NAMES[safe_atom_code] == atom_name
    )
    residue_code = canonical_code[residue_index]
    local_template_index = _TEMPLATE_ATOM_RANK[
        residue_code,
        safe_atom_code,
    ].astype(np.int64)
    if not np.all(known_atom_name & (local_template_index >= 0)):
        _reject("unexpected_atom")

    flat_residue_atom_index = (
        residue_index * _MAX_TEMPLATE_ATOMS + local_template_index
    )
    flat_occupancy = np.bincount(
        flat_residue_atom_index,
        minlength=residue_count * _MAX_TEMPLATE_ATOMS,
    )
    residue_atom_occupancy = flat_occupancy.reshape(
        residue_count,
        _MAX_TEMPLATE_ATOMS,
    )
    if np.any(residue_atom_occupancy > 1):
        _reject("duplicate_atom")

    required_atom_mask = (
        np.arange(_MAX_TEMPLATE_ATOMS)[None, :]
        < _INTERNAL_ATOM_COUNT[canonical_code][:, None]
    )
    if np.any(residue_atom_occupancy[required_atom_mask] == 0):
        _reject("incomplete_residue")

    raw_index_by_residue_atom = np.full(
        (residue_count, _MAX_TEMPLATE_ATOMS),
        -1,
        dtype=np.int64,
    )
    raw_index_by_residue_atom.flat[flat_residue_atom_index] = np.arange(
        atom_count,
        dtype=np.int64,
    )

    bond_array = _get_bond_array(atom_array, atom_count)
    encoded_source_bonds = _encoded_bond_pairs(
        bond_array[:, :2],
        atom_count,
    )
    if residue_count > 1:
        peptide_c_indices = raw_index_by_residue_atom[:-1, 2]
        peptide_n_indices = raw_index_by_residue_atom[1:, 0]
        expected_peptide_bonds = _encoded_bond_pairs(
            np.column_stack((peptide_c_indices, peptide_n_indices)),
            atom_count,
        )
        if not np.all(np.isin(expected_peptide_bonds, encoded_source_bonds)):
            _reject("peptide_bond_mismatch")

    terminal_c_index = raw_index_by_residue_atom[-1, 2]
    terminal_c_at_left = bond_array[:, 0] == terminal_c_index
    terminal_c_at_right = bond_array[:, 1] == terminal_c_index
    terminal_neighbours = np.concatenate(
        (
            bond_array[terminal_c_at_left, 1],
            bond_array[terminal_c_at_right, 0],
        )
    ).astype(np.int64, copy=False)
    terminal_c_has_inter_residue_bond = bool(
        terminal_neighbours.size > 0
        and np.any(residue_index[terminal_neighbours] != residue_count - 1)
    )

    template_atom_counts = _INTERNAL_ATOM_COUNT[canonical_code].astype(
        np.int64
    )
    if not terminal_c_has_inter_residue_bond:
        template_atom_counts[-1] = _TERMINAL_ATOM_COUNT[canonical_code[-1]]

    oxt_atom = local_template_index == _INTERNAL_ATOM_COUNT[residue_code]
    allowed_oxt = (residue_index == residue_count - 1) & (
        not terminal_c_has_inter_residue_bond
    )
    if np.any(oxt_atom & ~allowed_oxt):
        _reject("unexpected_atom")

    residue_offsets = np.empty(residue_count + 1, dtype=np.int64)
    residue_offsets[0] = 0
    np.cumsum(template_atom_counts, out=residue_offsets[1:])
    raw_to_template = residue_offsets[residue_index] + local_template_index

    template_atom_count = int(residue_offsets[-1])
    template_occupancy = np.bincount(
        raw_to_template,
        minlength=template_atom_count,
    )
    missing_template_mask = template_occupancy == 0
    disallowed_missing = missing_template_mask.copy()
    terminal_oxt_template_index = None
    if not terminal_c_has_inter_residue_bond:
        terminal_oxt_template_index = template_atom_count - 1
        disallowed_missing[terminal_oxt_template_index] = False
    if np.any(disallowed_missing):
        _reject("incomplete_residue")

    template_to_raw = np.full(template_atom_count, -1, dtype=np.int64)
    template_to_raw[raw_to_template] = np.arange(
        atom_count,
        dtype=np.int64,
    )

    template_column = np.arange(_MAX_TEMPLATE_ATOMS)
    template_mask = template_column[None, :] < template_atom_counts[:, None]
    template_atom_name = _TEMPLATE_ATOM_NAME[canonical_code][template_mask]
    template_residue_index = np.broadcast_to(
        np.arange(residue_count, dtype=np.int32)[:, None],
        template_mask.shape,
    )[template_mask]

    return PredictedProteinFastPathLayout(
        entity_id=entity_id,
        chain_id=str(coordinate_chain_ids[0]),
        canonical_sequence=canonical_sequence,
        raw_to_template=raw_to_template,
        template_to_raw=template_to_raw,
        template_residue_index=template_residue_index,
        template_atom_name=template_atom_name,
        residue_offsets=residue_offsets,
        missing_template_mask=missing_template_mask,
        terminal_oxt_template_index=terminal_oxt_template_index,
    )


@functools.lru_cache(maxsize=1)
def _get_standard_protein_template_cache() -> _StandardProteinTemplateCache:
    components = []
    for residue_code, residue_name in enumerate(_RESIDUE_NAME_BY_CODE):
        component = ccd.get_component_atom_array(
            residue_name,
            keep_leaving_atoms=True,
            keep_hydrogens=False,
        )
        if component is None:
            raise RuntimeError(
                f"CCD template is unavailable for {residue_name}"
            )
        expected_names = _TEMPLATE_ATOM_NAME[
            residue_code,
            : _TERMINAL_ATOM_COUNT[residue_code],
        ]
        if not np.array_equal(component.atom_name, expected_names):
            raise RuntimeError(
                "CCD atom order differs from RES_ATOMS_DICT for "
                f"{residue_name}"
            )
        if component.bonds is None:
            raise RuntimeError(
                f"CCD bonds are unavailable for {residue_name}"
            )
        components.append(component)

    common_annotations = set(components[0].get_annotation_categories())
    for component in components[1:]:
        common_annotations &= set(component.get_annotation_categories())

    annotations = {
        name: np.concatenate(
            [component.get_annotation(name) for component in components]
        )
        for name in sorted(common_annotations)
    }
    annotation_dtypes = {
        name: tuple(
            component.get_annotation(name).dtype for component in components
        )
        for name in sorted(common_annotations)
    }

    residue_offsets = np.empty(len(components) + 1, dtype=np.int64)
    residue_offsets[0] = 0
    np.cumsum(
        np.asarray([len(component) for component in components]),
        out=residue_offsets[1:],
    )

    component_bonds = [
        component.bonds.as_array() for component in components
    ]
    bond_counts = np.asarray(
        [len(bonds) for bonds in component_bonds],
        dtype=np.int16,
    )
    bond_table = np.zeros(
        (len(components), int(bond_counts.max()), 3),
        dtype=np.uint32,
    )
    for residue_code, bonds in enumerate(component_bonds):
        bond_table[residue_code, : len(bonds)] = bonds

    return _StandardProteinTemplateCache(
        annotations=annotations,
        annotation_dtypes=annotation_dtypes,
        residue_offsets=residue_offsets,
        bond_table=bond_table,
        bond_counts=bond_counts,
    )


def prewarm_predicted_protein_fast_path() -> None:
    """Load and cache the 20 standard amino-acid CCD templates."""
    _get_standard_protein_template_cache()


def _build_template_bonds(
    canonical_code: np.ndarray,
    layout: PredictedProteinFastPathLayout,
    template_cache: _StandardProteinTemplateCache,
) -> np.ndarray:
    template_atom_counts = np.diff(layout.residue_offsets)
    residue_count = canonical_code.size
    max_bond_count = template_cache.bond_table.shape[1]

    selected_bonds = template_cache.bond_table[canonical_code]
    bond_mask = (
        np.arange(max_bond_count)[None, :]
        < template_cache.bond_counts[canonical_code][:, None]
    )
    bond_mask &= (
        selected_bonds[:, :, 0] < template_atom_counts[:, None]
    )
    bond_mask &= (
        selected_bonds[:, :, 1] < template_atom_counts[:, None]
    )

    residue_index = np.broadcast_to(
        np.arange(residue_count, dtype=np.int64)[:, None],
        bond_mask.shape,
    )[bond_mask]
    template_bonds = selected_bonds[bond_mask].copy()
    template_bonds[:, :2] += layout.residue_offsets[
        residue_index,
        None,
    ].astype(
        template_bonds.dtype,
        copy=False,
    )
    return template_bonds


def complete_predicted_protein_atom_array(
    atom_array: AtomArray,
    layout: PredictedProteinFastPathLayout,
) -> AtomArray:
    """Reorder eligible coordinates and add a missing terminal OXT.

    This function intentionally assumes that ``layout`` came from
    :func:`prepare_predicted_protein_fast_path`. It does not catch exceptions:
    implementation errors must not be mistaken for an eligibility fallback.

    Args:
        atom_array: The filtered source coordinates used to create ``layout``.
        layout: Validated raw-to-template atom mapping.

    Returns:
        An ``AtomArray`` equivalent to
        ``DistillationMMCIFParser.add_missing_atoms_and_residues()``.
    """
    atom_count = len(atom_array)
    if layout.raw_to_template.shape != (atom_count,):
        raise ValueError("layout raw atom count does not match atom_array")

    canonical_code = _canonical_codes(layout.canonical_sequence)
    residue_count = canonical_code.size
    if layout.residue_offsets.shape != (residue_count + 1,):
        raise ValueError(
            "layout residue count does not match canonical_sequence"
        )

    template_cache = _get_standard_protein_template_cache()
    template_atom_count = layout.template_atom_count
    local_template_index = (
        np.arange(template_atom_count, dtype=np.int64)
        - layout.residue_offsets[layout.template_residue_index]
    )
    packed_template_index = (
        template_cache.residue_offsets[
            canonical_code[layout.template_residue_index]
        ]
        + local_template_index
    )

    completed = AtomArray(template_atom_count)
    selected_residue_codes = np.unique(canonical_code)
    for name, packed_values in template_cache.annotations.items():
        selected_dtypes = tuple(
            template_cache.annotation_dtypes[name][code]
            for code in selected_residue_codes
        )
        output_dtype = np.result_type(*selected_dtypes)
        completed.set_annotation(
            name,
            packed_values[packed_template_index].astype(
                output_dtype,
                copy=False,
            ),
        )

    if not np.array_equal(completed.atom_name, layout.template_atom_name):
        raise RuntimeError(
            "cached CCD template does not match the validated layout"
        )

    for name, source_values in atom_array._annot.items():
        if name not in completed._annot:
            completed.set_annotation(
                name,
                np.zeros(template_atom_count, dtype=source_values.dtype),
            )

    residue_source_index = layout.template_to_raw[
        layout.residue_offsets[:-1]
    ]
    if np.any(residue_source_index < 0):
        raise RuntimeError(
            "validated layout is missing a residue backbone N atom"
        )
    for name in ("res_id", "hetero", "label_seq_id", "auth_seq_id"):
        completed._annot[name][:] = atom_array._annot[name][
            residue_source_index
        ][layout.template_residue_index]

    for name in (
        "chain_id",
        "label_asym_id",
        "label_entity_id",
        "auth_asym_id",
    ):
        completed._annot[name][:] = atom_array._annot[name][0]

    observed_mask = layout.template_to_raw >= 0
    observed_source_index = layout.template_to_raw[observed_mask]
    for name in ("b_factor", "occupancy", "charge", "label_alt_id"):
        if name in atom_array._annot:
            completed._annot[name][observed_mask] = atom_array._annot[name][
                observed_source_index
            ]

    completed.set_annotation("is_resolved", observed_mask)
    completed.coord[:] = 0.0
    completed.coord[observed_mask] = atom_array.coord[
        observed_source_index
    ]

    template_bonds = _build_template_bonds(
        canonical_code,
        layout,
        template_cache,
    )
    completed.bonds = struc.BondList(template_atom_count, template_bonds)

    source_bonds = _get_bond_array(atom_array, atom_count).copy()
    source_bonds[:, 0] = layout.raw_to_template[source_bonds[:, 0]]
    source_bonds[:, 1] = layout.raw_to_template[source_bonds[:, 1]]
    completed.bonds = completed.bonds.merge(
        struc.BondList(template_atom_count, source_bonds)
    )
    return ccd.add_inter_residue_bonds(
        completed,
        exclude_struct_conn_pairs=True,
    )


__all__ = [
    "FastPathNotApplicable",
    "PredictedProteinFastPathLayout",
    "complete_predicted_protein_atom_array",
    "prewarm_predicted_protein_fast_path",
    "prepare_predicted_protein_fast_path",
]
