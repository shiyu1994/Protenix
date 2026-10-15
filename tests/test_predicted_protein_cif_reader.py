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

import tempfile
import unittest
from pathlib import Path

import numpy as np

from configs.configs_data import data_configs
from protenix.data.core.filter import Filter
from protenix.data.core.parser import DistillationMMCIFParser
from protenix.data.core.predicted_protein_cif_reader import (
    read_predicted_protein_atom_array_fast,
)
from protenix.data.core.predicted_protein_fast_path import (
    FastPathNotApplicable,
    complete_predicted_protein_atom_array,
    prepare_predicted_protein_fast_path,
)


_HAS_CCD = Path(data_configs["ccd_components_file"]).is_file()

_ATOM_SITE_FIELDS = (
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
    "pdbx_formal_charge",
    "auth_seq_id",
    "auth_comp_id",
    "auth_asym_id",
    "auth_atom_id",
    "pdbx_PDB_model_num",
)


def _atom_row(
    atom_id,
    atom_name,
    element,
    residue_name,
    residue_id,
    coordinates,
    model=1,
    altloc=".",
):
    x, y, z = coordinates
    return {
        "group_PDB": "ATOM",
        "id": str(atom_id),
        "type_symbol": element,
        "label_atom_id": atom_name,
        "label_alt_id": altloc,
        "label_comp_id": residue_name,
        "label_asym_id": "A",
        "label_entity_id": "1",
        "label_seq_id": str(residue_id),
        "pdbx_PDB_ins_code": "?",
        "Cartn_x": f"{x:.3f}",
        "Cartn_y": f"{y:.3f}",
        "Cartn_z": f"{z:.3f}",
        "occupancy": "1.00",
        "B_iso_or_equiv": f"{90.0 + atom_id:.2f}",
        "pdbx_formal_charge": "?",
        "auth_seq_id": str(residue_id),
        "auth_comp_id": residue_name,
        "auth_asym_id": "A",
        "auth_atom_id": atom_name,
        "pdbx_PDB_model_num": str(model),
    }


def _predicted_cif_text(
    *,
    altloc=".",
    last_model=1,
    missing_field=None,
    include_cell=False,
    include_struct_conn=False,
    second_sequence_residue="GLY",
):
    atom_specs = [
        ("N", "N", "ALA", 1),
        ("CA", "C", "ALA", 1),
        ("C", "C", "ALA", 1),
        ("CB", "C", "ALA", 1),
        ("O", "O", "ALA", 1),
        ("N", "N", "GLY", 2),
        ("CA", "C", "GLY", 2),
        ("C", "C", "GLY", 2),
        ("O", "O", "GLY", 2),
    ]
    rows = []
    for atom_id, (
        atom_name,
        element,
        residue_name,
        residue_id,
    ) in enumerate(atom_specs, start=1):
        model = last_model if atom_id == len(atom_specs) else 1
        rows.append(
            _atom_row(
                atom_id,
                atom_name,
                element,
                residue_name,
                residue_id,
                (atom_id, atom_id + 0.25, atom_id + 0.5),
                model=model,
                altloc=altloc,
            )
        )

    fields = [
        field for field in _ATOM_SITE_FIELDS if field != missing_field
    ]
    lines = [
        "data_TEST",
        "_entry.id TEST",
        "_entity_poly.entity_id 1",
        "_entity_poly.type polypeptide(L)",
        "loop_",
        "_entity_poly_seq.entity_id",
        "_entity_poly_seq.num",
        "_entity_poly_seq.mon_id",
        "1 1 ALA",
        f"1 2 {second_sequence_residue}",
        "loop_",
    ]
    lines.extend(f"_atom_site.{field}" for field in fields)
    lines.extend(
        " ".join(row[field] for field in fields) for row in rows
    )

    if include_cell:
        lines.extend(
            [
                "_cell.length_a 10.0",
                "_cell.length_b 10.0",
                "_cell.length_c 10.0",
                "_cell.angle_alpha 90.0",
                "_cell.angle_beta 90.0",
                "_cell.angle_gamma 90.0",
            ]
        )
    if include_struct_conn:
        lines.extend(
            [
                "loop_",
                "_struct_conn.id",
                "_struct_conn.conn_type_id",
                "link1 covale",
            ]
        )
    return "\n".join(lines) + "\n"


def _write_cif(directory, text=None, filename="input.cif"):
    path = Path(directory) / filename
    path.write_text(
        _predicted_cif_text() if text is None else text,
        encoding="utf-8",
    )
    return path


def _filter_source(parser, atom_array):
    atom_array = parser.mse_to_met(atom_array)
    atom_array = parser.fix_arginine(atom_array)
    atom_array = Filter.remove_water(atom_array)
    atom_array = Filter.remove_hydrogens(atom_array)
    return Filter.remove_element_X(atom_array)


class PredictedProteinCIFReaderTest(unittest.TestCase):
    def assert_fallback(
        self,
        reason,
        text,
        filename="input.cif",
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_cif(
                directory,
                text=text,
                filename=filename,
            )
            with self.assertRaises(FastPathNotApplicable) as context:
                read_predicted_protein_atom_array_fast(path)
        self.assertEqual(context.exception.reason, reason)

    def test_reader_matches_generic_source_without_generic_connector(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_cif(directory)
            generic = DistillationMMCIFParser(path).get_structure()
            actual = read_predicted_protein_atom_array_fast(path)

        self.assertEqual(
            set(generic.get_annotation_categories()),
            set(actual.get_annotation_categories()),
        )
        for name in generic.get_annotation_categories():
            with self.subTest(annotation=name):
                self.assertEqual(
                    generic.get_annotation(name).dtype,
                    actual.get_annotation(name).dtype,
                )
                np.testing.assert_array_equal(
                    generic.get_annotation(name),
                    actual.get_annotation(name),
                )
        np.testing.assert_array_equal(generic.coord, actual.coord)
        self.assertIsNone(actual.box)

        np.testing.assert_array_equal(
            generic.bonds.as_array(),
            actual.bonds.as_array(),
        )
        first_bond = actual.bonds.as_array()[0]
        self.assertEqual(actual.atom_name[first_bond[0]], "C")
        self.assertEqual(actual.res_id[first_bond[0]], 1)
        self.assertEqual(actual.atom_name[first_bond[1]], "N")
        self.assertEqual(actual.res_id[first_bond[1]], 2)

    def test_rejects_non_cif_path(self):
        self.assert_fallback(
            "unsupported_cif_path",
            _predicted_cif_text(),
            filename="input.cif.gz",
        )

    def test_rejects_missing_expected_field(self):
        self.assert_fallback(
            "missing_cif_field",
            _predicted_cif_text(missing_field="occupancy"),
        )

    def test_rejects_multiple_models_and_true_altloc(self):
        cases = [
            ("model_count", _predicted_cif_text(last_model=2)),
            (
                "alternate_location",
                _predicted_cif_text(altloc="A"),
            ),
        ]
        for reason, text in cases:
            with self.subTest(reason=reason):
                self.assert_fallback(reason, text)

    def test_rejects_cell_and_struct_conn(self):
        cases = [
            (
                "cell_present",
                _predicted_cif_text(include_cell=True),
            ),
            (
                "struct_conn_present",
                _predicted_cif_text(include_struct_conn=True),
            ),
        ]
        for reason, text in cases:
            with self.subTest(reason=reason):
                self.assert_fallback(reason, text)

    def test_rejects_canonical_metadata_mismatch(self):
        self.assert_fallback(
            "canonical_sequence_mismatch",
            _predicted_cif_text(second_sequence_residue="ALA"),
        )


@unittest.skipUnless(
    _HAS_CCD,
    "standard CCD components file is unavailable",
)
class PredictedProteinCIFReaderIntegrationTest(unittest.TestCase):
    def test_fast_completion_matches_generic_add_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_cif(directory)
            parser = DistillationMMCIFParser(path)

            generic_source = _filter_source(
                parser,
                parser.get_structure(),
            )
            expected = parser.add_missing_atoms_and_residues(
                generic_source.copy()
            )

            fast_source = _filter_source(
                parser,
                read_predicted_protein_atom_array_fast(path),
            )
            layout = prepare_predicted_protein_fast_path(
                fast_source,
                parser.get_sequences(fast_source),
                parser.entity_poly_type,
            )
            actual = complete_predicted_protein_atom_array(
                fast_source,
                layout,
            )

        self.assertEqual(
            set(expected.get_annotation_categories()),
            set(actual.get_annotation_categories()),
        )
        for name in expected.get_annotation_categories():
            with self.subTest(annotation=name):
                self.assertEqual(
                    expected.get_annotation(name).dtype,
                    actual.get_annotation(name).dtype,
                )
                np.testing.assert_array_equal(
                    expected.get_annotation(name),
                    actual.get_annotation(name),
                )
        self.assertEqual(expected.coord.dtype, actual.coord.dtype)
        np.testing.assert_array_equal(expected.coord, actual.coord)
        self.assertIsNone(expected.box)
        self.assertIsNone(actual.box)
        self.assertEqual(
            expected.bonds.as_array().dtype,
            actual.bonds.as_array().dtype,
        )
        np.testing.assert_array_equal(
            expected.bonds.as_array(),
            actual.bonds.as_array(),
        )


if __name__ == "__main__":
    unittest.main()
