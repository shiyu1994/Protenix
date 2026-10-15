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

"""Check C-alpha scoring against simple structures with known displacements."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray

from scripts.analysis.recalculate_peptide_ca_rmsd import ca_reference, recover_ca, select
from scripts.analysis.recalculate_peptide_epitope_rmsd import epitope_reference
from scripts.evaluation.peptide_pose_metrics import score_pose


class TestPeptideCaRecalculation(unittest.TestCase):
    def reference(self):
        return {
            "name": "fixture", "peptide_chains": ["P"], "fixed_atoms": {},
            "chains": {"R": {"role": "receptor", "group": "r"}, "P": {"role": "peptide", "group": "p"}},
            "atoms": [
                ["R", 1, "ALA", "CA", "C", 0, 0, 0],
                ["R", 2, "ALA", "CA", "C", 1, 0, 0],
                ["R", 3, "ALA", "CA", "C", 0, 1, 0],
                ["P", 1, "ALA", "CA", "C", 0, 0, 1],
                ["P", 1, "ALA", "CB", "C", 0, 0, 2],
            ],
        }

    def test_pose_displacement_ignores_sidechain_and_global_transform(self):
        ref, _ = ca_reference(self.reference())
        prediction = {(r[0], r[1], r[3]): np.asarray(r[5:8], dtype=float) for r in self.reference()["atoms"]}
        prediction["P", 1, "CA"] += [0, 0, 3]
        prediction["P", 1, "CB"] += [100, 100, 100]
        rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        prediction = {key: value @ rotation + [17, 3, -5] for key, value in prediction.items()}
        result = score_pose(ref, prediction, symmetry=False)
        self.assertAlmostEqual(result["receptor_rmsd"], 0)
        self.assertAlmostEqual(result["max_peptide_rmsd"], 3)
        self.assertEqual(result["peptides"][0]["n_atoms"], 1)

    def test_equivalent_peptide_assignment(self):
        ref = self.reference()
        ref["peptide_chains"].append("Q")
        ref["chains"]["Q"] = {"role": "peptide", "group": "p"}
        ref["atoms"].append(["Q", 1, "ALA", "CA", "C", 10, 0, 1])
        ref, _ = ca_reference(ref)
        prediction = {(r[0], r[1], r[3]): np.asarray(r[5:8], dtype=float) for r in ref["atoms"]}
        prediction["P", 1, "CA"], prediction["Q", 1, "CA"] = prediction["Q", 1, "CA"], prediction["P", 1, "CA"]
        result = score_pose(ref, prediction, symmetry=False)
        self.assertAlmostEqual(result["max_peptide_rmsd"], 0)
        self.assertEqual(result["peptides"][0]["predicted_peptide_chain"], "Q")

    def test_missing_ca_is_not_a_zero_score(self):
        ref = self.reference()
        ref["atoms"] = [r for r in ref["atoms"] if not (r[0] == "P" and r[3] == "CA")]
        with self.assertRaisesRegex(ValueError, "No observed peptide CA"):
            ca_reference(ref)

    def test_best_is_reselected_using_ca_across_seeds(self):
        rows = [
            {"seed": 101, "sample": 0, "ranking_score": 0.9, "peptide_ca_rmsd": 4},
            {"seed": 102, "sample": 0, "ranking_score": 0.8, "peptide_ca_rmsd": 1},
        ]
        result = select(rows)
        self.assertEqual(result["top1"]["seed"], 101)
        self.assertEqual(result["best"]["seed"], 102)

    def test_epitope_uses_native_ca_and_excludes_distal_receptor_motion(self):
        reference = self.reference()
        reference["atoms"] += [
            ["R", 4, "ALA", "CA", "C", 100, 0, 0],
            ["R", 4, "ALA", "CB", "C", 0, 0, 1],
            ["R", 5, "ALA", "CA", "C", 100, 1, 0],
            ["R", 6, "ALA", "CA", "C", 100, 0, 1],
        ]
        local, audit = epitope_reference(reference, 10)
        self.assertEqual(audit["epitope_receptor_ca_atoms"], 3)
        self.assertEqual(audit["global_receptor_ca_atoms"], 6)
        prediction = {(r[0], r[1], r[3]): np.asarray(r[5:8], dtype=float) for r in reference["atoms"]}
        for position in (4, 5, 6):
            prediction["R", position, "CA"] += [0, 50, 0]
        prediction["P", 1, "CA"] += [0, 0, 3]
        local, _ = ca_reference(local)
        result = score_pose(local, prediction, symmetry=False)
        self.assertAlmostEqual(result["max_peptide_rmsd"], 3)
        self.assertAlmostEqual(result["receptor_rmsd"], 0)
        global_ref, _ = ca_reference(reference)
        self.assertGreater(score_pose(global_ref, prediction, symmetry=False)["max_peptide_rmsd"], 3)

    def test_epitope_cutoff_is_inclusive_and_requires_three_anchors(self):
        reference = self.reference()
        reference["atoms"].append(["R", 4, "ALA", "CA", "C", 10, 0, 1])
        _, audit = epitope_reference(reference, 10)
        self.assertEqual(audit["epitope_receptor_ca_atoms"], 4)
        _, audit = epitope_reference(reference, 9.999)
        self.assertEqual(audit["epitope_receptor_ca_atoms"], 3)
        with self.assertRaisesRegex(ValueError, "Fewer than three epitope"):
            epitope_reference(reference, 0.5)

    def test_epitope_union_includes_both_native_binding_sites(self):
        reference = self.reference()
        reference["peptide_chains"].append("Q")
        reference["chains"]["Q"] = {"role": "peptide", "group": "p"}
        reference["atoms"] += [
            ["Q", 1, "ALA", "CA", "C", 100, 0, 1],
            ["R", 4, "ALA", "CA", "C", 100, 0, 0],
        ]
        _, audit = epitope_reference(reference, 10)
        self.assertEqual(audit["epitope_receptor_ca_atoms"], 4)

    def test_training_ca_identity_uses_coordinates_and_elements(self):
        atoms = AtomArray(2)
        atoms.atom_name = ["C001", "C002"]
        atoms.element = ["C", "C"]
        atoms.coord = np.asarray([[1, 2, 3], [2, 2, 3]], dtype=float)
        atoms.set_annotation("peptide_source_atom_index", np.asarray([10, 11]))
        cif = """data_test
loop_
_atom_site.label_atom_id
_atom_site.type_symbol
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.label_comp_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
CB C A 1 ALA 2 2 3
CA C A 1 ALA 1 2 3
#
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.cif"
            path.write_text(cif)
            mask, audit = recover_ca(atoms, np.asarray([0, 1]), path)
            np.testing.assert_array_equal(mask, [True, False])
            self.assertEqual(audit[0]["original_atom_site_row"], 1)
            atoms.coord[0] += [1, 0, 0.1]
            with self.assertRaisesRegex(ValueError, "cannot be identified"):
                recover_ca(atoms, np.asarray([0, 1]), path)


if __name__ == "__main__":
    unittest.main()
