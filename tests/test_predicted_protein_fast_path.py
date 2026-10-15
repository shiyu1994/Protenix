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

import unittest
from pathlib import Path

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from configs.configs_data import data_configs
from protenix.data.constants import (
    PROTEIN_COMMON_ONE_TO_THREE,
    RES_ATOMS_DICT,
)
from protenix.data.core.filter import Filter
from protenix.data.core.parser import DistillationMMCIFParser
from protenix.data.core.predicted_protein_fast_path import (
    FastPathNotApplicable,
    complete_predicted_protein_atom_array,
    prepare_predicted_protein_fast_path,
)

_HAS_CCD = Path(data_configs["ccd_components_file"]).is_file()
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _element_for_atom(atom_name: str) -> str:
    if atom_name.startswith("N"):
        return "N"
    if atom_name.startswith("O"):
        return "O"
    if atom_name.startswith("S"):
        return "S"
    return "C"


def _make_atom_array(
    sequence: str = "AG",
    include_terminal_oxt: bool = True,
) -> AtomArray:
    residue_ids = []
    residue_names = []
    atom_names = []
    for residue_id, one_letter in enumerate(sequence, start=1):
        residue_name = PROTEIN_COMMON_ONE_TO_THREE[one_letter]
        names = tuple(
            sorted(
                RES_ATOMS_DICT[residue_name],
                key=RES_ATOMS_DICT[residue_name].__getitem__,
            )
        )
        if residue_id < len(sequence) or not include_terminal_oxt:
            names = tuple(name for name in names if name != "OXT")
        residue_ids.extend([residue_id] * len(names))
        residue_names.extend([residue_name] * len(names))
        atom_names.extend(names)

    atom_array = AtomArray(len(atom_names))
    atom_array.coord = np.arange(
        len(atom_names) * 3,
        dtype=np.float32,
    ).reshape(-1, 3)
    atom_array.chain_id = np.full(len(atom_names), "A")
    atom_array.res_id = np.asarray(residue_ids)
    atom_array.res_name = np.asarray(residue_names)
    atom_array.atom_name = np.asarray(atom_names)
    atom_array.element = np.asarray(
        [_element_for_atom(name) for name in atom_names]
    )
    atom_array.hetero = np.zeros(len(atom_names), dtype=bool)
    atom_array.set_annotation(
        "label_entity_id",
        np.full(len(atom_names), "1"),
    )
    atom_array.set_annotation(
        "label_seq_id",
        np.asarray(residue_ids),
    )
    atom_array.set_annotation(
        "auth_seq_id",
        np.asarray(residue_ids).astype(str),
    )
    atom_array.set_annotation(
        "label_asym_id",
        np.full(len(atom_names), "A"),
    )
    atom_array.set_annotation(
        "auth_asym_id",
        np.full(len(atom_names), "A"),
    )
    atom_array.set_annotation(
        "b_factor",
        np.arange(len(atom_names), dtype=np.float64),
    )
    atom_array.set_annotation(
        "occupancy",
        np.ones(len(atom_names), dtype=np.float64),
    )
    atom_array.set_annotation(
        "charge",
        np.zeros(len(atom_names), dtype=np.int64),
    )
    atom_array.set_annotation(
        "label_alt_id",
        np.full(len(atom_names), "."),
    )
    atom_array.bonds = struc.connect_via_residue_names(
        atom_array,
        inter_residue=True,
    )
    return atom_array


def _prepare(atom_array: AtomArray, sequence: str = "AG"):
    return prepare_predicted_protein_fast_path(
        atom_array,
        sequences={"1": sequence},
        entity_poly_type={"1": "polypeptide(L)"},
    )


class PredictedProteinFastPathTest(unittest.TestCase):
    def assert_fallback(
        self,
        reason: str,
        atom_array: AtomArray,
        sequence: str = "AG",
    ) -> None:
        with self.assertRaises(FastPathNotApplicable) as context:
            _prepare(atom_array, sequence)
        self.assertEqual(context.exception.reason, reason)

    def test_complete_template_layout(self):
        atom_array = _make_atom_array()

        layout = _prepare(atom_array)

        np.testing.assert_array_equal(
            layout.raw_to_template,
            np.arange(len(atom_array)),
        )
        np.testing.assert_array_equal(
            layout.template_to_raw,
            np.arange(len(atom_array)),
        )
        np.testing.assert_array_equal(
            layout.missing_template_mask,
            False,
        )
        np.testing.assert_array_equal(
            layout.residue_offsets,
            [0, 5, 10],
        )
        np.testing.assert_array_equal(
            layout.template_atom_name,
            [
                "N",
                "CA",
                "C",
                "O",
                "CB",
                "N",
                "CA",
                "C",
                "O",
                "OXT",
            ],
        )
        np.testing.assert_array_equal(
            layout.template_residue_index,
            [0] * 5 + [1] * 5,
        )
        self.assertEqual(layout.entity_id, "1")
        self.assertEqual(layout.chain_id, "A")
        self.assertFalse(layout.terminal_oxt_missing)

    def test_missing_terminal_oxt_has_synthetic_template_slot(self):
        atom_array = _make_atom_array(include_terminal_oxt=False)

        layout = _prepare(atom_array)

        self.assertEqual(
            layout.template_atom_count,
            len(atom_array) + 1,
        )
        self.assertEqual(
            layout.terminal_oxt_template_index,
            layout.template_atom_count - 1,
        )
        self.assertEqual(layout.template_to_raw[-1], -1)
        self.assertEqual(layout.template_atom_name[-1], "OXT")
        self.assertTrue(layout.missing_template_mask[-1])
        self.assertEqual(
            np.count_nonzero(layout.missing_template_mask),
            1,
        )
        self.assertTrue(layout.terminal_oxt_missing)

    def test_arbitrary_raw_atom_order_is_mapped_to_template(self):
        ordered = _make_atom_array()
        permutation = np.asarray(
            [6, 0, 9, 4, 1, 7, 3, 8, 2, 5]
        )
        shuffled = ordered[permutation]

        layout = _prepare(shuffled)

        np.testing.assert_array_equal(
            shuffled.atom_name[layout.template_to_raw],
            ordered.atom_name,
        )
        np.testing.assert_array_equal(
            layout.raw_to_template[layout.template_to_raw],
            np.arange(len(ordered)),
        )

    def test_metadata_fallback_reasons(self):
        atom_array = _make_atom_array()
        cases = [
            (
                "polymer_entity_count",
                {"1": "AG"},
                {"1": "polypeptide(L)", "2": "polypeptide(L)"},
            ),
            (
                "unsupported_polymer_type",
                {"1": "AG"},
                {"1": "polyribonucleotide"},
            ),
            (
                "sequence_entity_mismatch",
                {"2": "AG"},
                {"1": "polypeptide(L)"},
            ),
            (
                "nonstandard_canonical_sequence",
                {"1": "AX"},
                {"1": "polypeptide(L)"},
            ),
        ]
        for reason, sequences, entity_poly_type in cases:
            with self.subTest(reason=reason):
                with self.assertRaises(FastPathNotApplicable) as context:
                    prepare_predicted_protein_fast_path(
                        atom_array,
                        sequences,
                        entity_poly_type,
                    )
                self.assertEqual(context.exception.reason, reason)

    def test_coordinate_identity_fallback_reasons(self):
        cases = []

        atom_array = _make_atom_array()
        atom_array.label_entity_id[-1] = "2"
        cases.append(("coordinate_entity_count", atom_array))

        atom_array = _make_atom_array()
        atom_array.label_entity_id[:] = "2"
        cases.append(("coordinate_entity_mismatch", atom_array))

        atom_array = _make_atom_array()
        atom_array.chain_id[-1] = "B"
        cases.append(("coordinate_chain_count", atom_array))

        for reason, atom_array in cases:
            with self.subTest(reason=reason):
                self.assert_fallback(reason, atom_array)

    def test_residue_and_sequence_fallback_reasons(self):
        cases = []

        atom_array = _make_atom_array()
        atom_array.res_id[5:] = 3
        atom_array.label_seq_id[5:] = 3
        cases.append(
            ("residue_ids_not_contiguous", atom_array, "AG")
        )

        atom_array = _make_atom_array()
        atom_array.label_seq_id[0] = 2
        cases.append(("label_seq_id_mismatch", atom_array, "AG"))

        atom_array = _make_atom_array()
        atom_array.res_name[5:] = "UNK"
        cases.append(("nonstandard_residue", atom_array, "AG"))

        cases.append(
            (
                "canonical_sequence_mismatch",
                _make_atom_array(),
                "AA",
            )
        )

        for reason, atom_array, sequence in cases:
            with self.subTest(reason=reason):
                self.assert_fallback(reason, atom_array, sequence)

    def test_atom_set_fallback_reasons(self):
        cases = []

        atom_array = _make_atom_array()
        atom_array.atom_name[4] = "CA"
        cases.append(("duplicate_atom", atom_array))

        atom_array = _make_atom_array()
        atom_array.atom_name[4] = "CG"
        cases.append(("unexpected_atom", atom_array))

        atom_array = _make_atom_array()
        atom_array = atom_array[np.arange(len(atom_array)) != 4]
        cases.append(("incomplete_residue", atom_array))

        atom_array = _make_atom_array()
        atom_array.element[0] = "X"
        cases.append(("unsupported_element", atom_array))

        atom_array = _make_atom_array()
        atom_array.coord[0, 0] = np.nan
        cases.append(("nonfinite_coordinates", atom_array))

        for reason, atom_array in cases:
            with self.subTest(reason=reason):
                self.assert_fallback(reason, atom_array)

    def test_missing_peptide_bond_falls_back(self):
        atom_array = _make_atom_array()
        atom_array.bonds.remove_bond(2, 5)

        self.assert_fallback(
            "peptide_bond_mismatch",
            atom_array,
        )


@unittest.skipUnless(
    _HAS_CCD,
    "standard CCD components file is unavailable",
)
class PredictedProteinFastPathCompletionTest(unittest.TestCase):
    def assert_atom_arrays_equal(
        self,
        expected: AtomArray,
        actual: AtomArray,
    ) -> None:
        self.assertEqual(
            set(expected.get_annotation_categories()),
            set(actual.get_annotation_categories()),
        )
        for name in expected.get_annotation_categories():
            with self.subTest(annotation=name):
                expected_values = expected.get_annotation(name)
                actual_values = actual.get_annotation(name)
                self.assertEqual(
                    expected_values.dtype,
                    actual_values.dtype,
                )
                np.testing.assert_array_equal(
                    expected_values,
                    actual_values,
                )

        self.assertEqual(expected.coord.dtype, actual.coord.dtype)
        np.testing.assert_array_equal(expected.coord, actual.coord)
        if expected.box is None:
            self.assertIsNone(actual.box)
        else:
            np.testing.assert_array_equal(expected.box, actual.box)

        expected_bonds = expected.bonds.as_array()
        actual_bonds = actual.bonds.as_array()
        self.assertEqual(expected_bonds.dtype, actual_bonds.dtype)
        np.testing.assert_array_equal(expected_bonds, actual_bonds)

    def test_complete_and_shuffled_raw_atoms_have_identical_output(self):
        ordered = _make_atom_array()
        ordered_layout = _prepare(ordered)
        ordered_completed = complete_predicted_protein_atom_array(
            ordered,
            ordered_layout,
        )

        permutation = np.asarray(
            [2, 0, 4, 1, 3, 7, 5, 9, 6, 8]
        )
        shuffled = ordered[permutation]
        shuffled_layout = _prepare(shuffled)
        shuffled_completed = complete_predicted_protein_atom_array(
            shuffled,
            shuffled_layout,
        )

        self.assert_atom_arrays_equal(
            ordered_completed,
            shuffled_completed,
        )
        self.assertTrue(np.all(ordered_completed.is_resolved))

    def test_missing_terminal_oxt_is_zero_and_unresolved(self):
        atom_array = _make_atom_array(include_terminal_oxt=False)

        completed = complete_predicted_protein_atom_array(
            atom_array,
            _prepare(atom_array),
        )

        self.assertEqual(completed.atom_name[-1], "OXT")
        np.testing.assert_array_equal(
            completed.coord[-1],
            np.zeros(3),
        )
        self.assertFalse(completed.is_resolved[-1])
        self.assertEqual(completed.b_factor[-1], 0.0)
        self.assertEqual(completed.occupancy[-1], 0.0)
        self.assertEqual(completed.label_alt_id[-1], "")

    def test_2lwu_matches_generic_add_missing_atom_pipeline(self):
        parser = DistillationMMCIFParser(
            _REPOSITORY_ROOT / "examples" / "2lwu.cif"
        )
        atom_array = parser.get_structure()
        atom_array = parser.mse_to_met(atom_array)
        atom_array = parser.fix_arginine(atom_array)
        atom_array = Filter.remove_water(atom_array)
        atom_array = Filter.remove_hydrogens(atom_array)
        atom_array = Filter.remove_element_X(atom_array)

        layout = prepare_predicted_protein_fast_path(
            atom_array,
            parser.get_sequences(atom_array),
            parser.entity_poly_type,
        )
        expected = parser.add_missing_atoms_and_residues(
            atom_array.copy()
        )
        actual = complete_predicted_protein_atom_array(
            atom_array,
            layout,
        )

        self.assertIsNone(layout.terminal_oxt_template_index)
        self.assert_atom_arrays_equal(expected, actual)


if __name__ == "__main__":
    unittest.main()
