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

"""Regressions for AF3/native peptide numbering and chain identity mapping."""

import unittest

import numpy as np
from biotite.structure.io.pdbx import CIFBlock, CIFCategory

from scripts.analysis.evaluate_af3_peptide_top1 import normalize_ca
from scripts.analysis.evaluate_af3_peptide_samples import alias_original, ca_lookup, mapping_routes


def block(atoms, receptor=None, annotation=False):
    result = CIFBlock()
    columns = ["label_asym_id", "label_seq_id", "label_comp_id", "Cartn_x", "Cartn_y", "Cartn_z"]
    data = {key: [str(row[i]) for row in atoms] for i, key in enumerate(columns)}
    data.update({"label_atom_id": ["CA"] * len(atoms), "type_symbol": ["C"] * len(atoms),
                 "label_entity_id": ["1"] * len(atoms), "auth_seq_id": ["1001"] * len(atoms)})
    result["atom_site"] = CIFCategory(data)
    if receptor:
        result["struct_asym"] = CIFCategory({"id": [receptor], "entity_id": ["1"]})
        result["entity_poly_seq"] = CIFCategory({"entity_id": ["1"] * 3,
                                                "num": ["1", "2", "3"], "mon_id": ["GLY"] * 3})
    if annotation:
        result["foldbench_peptide_ligand"] = CIFCategory({"id": ["LIG1"], "source_label_asym_ids": ["P,D"]})
    return result


class AF3MappingTests(unittest.TestCase):
    def fixture(self):
        ref = {"chains": {"R": {"role": "receptor", "residues": ["GLY"] * 3},
                          "LIG1": {"role": "peptide"}},
               "atoms": [["R", 1, "GLY", "CA", "C", 0, 0, 0],
                         ["R", 3, "GLY", "CA", "C", 0, 3, 0],
                         ["LIG1", 1, "PHE", "CA", "C", 1, 1, 1],
                         ["LIG1", 2, "GLU", "CA", "C", 2, 2, 2]]}
        native = block([("R", 1, "GLY", 0, 0, 0), ("R", 3, "GLY", 0, 3, 0),
                        ("LIG1", 3, "PHE", 1, 1, 1), ("LIG1", ".", "GLU", 2, 2, 2)], annotation=True)
        predicted = block([("S", 1, "GLY", 10, 0, 0), ("S", 2, "GLY", 10, 2, 0),
                           ("S", 3, "GLY", 10, 3, 0), ("LIG1", 3, "PHE", 11, 1, 1),
                           ("D", 1, "GLU", 12, 2, 2)], receptor="S")
        return ref, native, predicted

    def test_numbering_renamed_receptor_and_appended_glu(self):
        ref, native, predicted = self.fixture()
        mapped, audit = normalize_ca(ref, native, predicted)
        np.testing.assert_array_equal(mapped["LIG1", 1, "CA"], [11, 1, 1])
        np.testing.assert_array_equal(mapped["LIG1", 2, "CA"], [12, 2, 2])
        np.testing.assert_array_equal(mapped["R", 2, "CA"], [10, 2, 0])
        self.assertEqual(audit[-1]["method"], "native_annotation_unique_appended_ligand")

    def test_unexpected_peptide_mutation_is_rejected(self):
        ref, native, predicted = self.fixture()
        predicted["atom_site"]["label_comp_id"] = ["GLY", "GLY", "GLY", "ALA", "GLU"]
        with self.assertRaisesRegex(ValueError, "Unexpected residue mismatch"):
            normalize_ca(ref, native, predicted)

    def test_modified_residue_difference_is_explicitly_audited(self):
        ref, native, predicted = self.fixture()
        predicted["atom_site"]["label_comp_id"] = ["GLY", "GLY", "GLY", "NFA", "GLU"]
        _, audit = normalize_ca(ref, native, predicted)
        self.assertEqual(sum(r["chemistry_mismatch"] for r in audit), 1)

    def test_original_alias_preserves_coordinates_and_missing_native_receptor_ca(self):
        ref, native, predicted = self.fixture()
        expected, _ = normalize_ca(ref, native, predicted)
        predicted["atom_site"]["label_asym_id"] = ["S", "S", "S", "P", "D"]
        aliased = alias_original(predicted, [{"source_label_asym_id": "P", "id": "LIG1"}])
        actual, audit = normalize_ca(ref, native, aliased)
        lookup, _ = ca_lookup(aliased)
        routes = mapping_routes(ref, audit, lookup)
        self.assertEqual(set(expected), set(routes))
        self.assertIn(("R", 2, "CA"), routes)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
            np.testing.assert_array_equal(lookup[routes[key]]["coord"], expected[key])
        self.assertEqual(predicted["atom_site"]["label_asym_id"].as_array(str)[3], "P")

    def test_duplicate_original_alias_is_rejected(self):
        _, _, predicted = self.fixture()
        with self.assertRaisesRegex(ValueError, "Non-bijective"):
            alias_original(predicted, [{"source_label_asym_id": "P", "id": "LIG1"},
                                       {"source_label_asym_id": "Q", "id": "LIG1"}])


if __name__ == "__main__":
    unittest.main()
