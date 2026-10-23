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

"""Feature and scoring regressions for the single-residue peptide input mode."""

import importlib.util
import unittest

HAS_DEPS = all(importlib.util.find_spec(n) is not None for n in ("biotite", "torch", "rdkit", "scipy"))
if HAS_DEPS:
    import biotite.structure as struc
    import numpy as np

    from protenix.data.inference.ligand_reference import apply_ligand_references
    from scripts.evaluation.peptide_pose_metrics import score_pose
    from scripts.evaluation.peptide_training_aligned import epitope_ca_score, epitope_summary


@unittest.skipUnless(HAS_DEPS, "Requires scientific Python dependencies")
class TrainingAlignedTests(unittest.TestCase):
    def atom_fixture(self):
        a = struc.AtomArray(4)
        a.chain_id[:] = ["R", "L", "L", "L"]
        a.res_id[:] = [1, 1, 1, 2]
        a.res_name[:] = ["CYS", "CYS", "CYS", "GLY"]
        a.atom_name[:] = ["SG", "SG", "CA", "CA"]
        a.element[:] = ["S", "S", "C", "C"]
        a.set_annotation("label_entity_id", np.array(["1", "2", "2", "2"]))
        a.set_annotation("label_seq_id", a.res_id.copy())
        a.set_annotation("ref_pos", np.zeros((4, 3)))
        a.set_annotation("ref_charge", np.zeros(4, dtype=int))
        a.set_annotation("ref_mask", np.ones(4, dtype=bool))
        a.bonds = struc.BondList(4, np.array([[0, 1, 1], [1, 2, 1], [2, 3, 1]]))
        refs = {"L": {
            "source_keys": [[1, "SG"], [1, "CA"], [2, "CA"]],
            "atom_names": ["S000", "C001", "C002"], "res_name": "X00",
            "elements": ["S", "C", "C"], "charges": [0, 1, 0],
            "ref_pos": [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
            "bonds": [[0, 1, 1], [1, 2, 1]],
        }}
        return a, refs

    def test_collapse_preserves_external_bond_and_reference_features(self):
        a, refs = self.atom_fixture()
        b = apply_ligand_references(a, refs, {"1": "polypeptide(L)", "2": "non-polymer"})
        np.testing.assert_array_equal(b.res_id[1:], [1, 1, 1])
        np.testing.assert_array_equal(b.atom_name[1:], refs["L"]["atom_names"])
        np.testing.assert_array_equal(b.ref_pos[1:], refs["L"]["ref_pos"])
        np.testing.assert_array_equal(b.ref_charge[1:], [0, 1, 0])
        np.testing.assert_array_equal(b.bonds.as_array(), a.bonds.as_array())
        self.assertEqual(a.atom_name[1], "SG")

    def test_rejects_polymer_override_and_bad_mapping(self):
        a, refs = self.atom_fixture()
        with self.assertRaisesRegex(ValueError, "polymer"):
            apply_ligand_references(a, refs, {"2": "polypeptide(L)"})
        refs["L"]["source_keys"][0] = [999, "SG"]
        with self.assertRaisesRegex(ValueError, "mapping"):
            apply_ligand_references(a, refs, {})

    def scoring_fixture(self):
        ref = {
            "name": "fixture", "peptide_chains": ["L"], "fixed_atoms": {},
            "chains": {"R": {"role": "receptor", "group": "r"},
                       "L": {"role": "peptide", "group": "l"}},
            "atoms": [["R", i, "ALA", "CA", "C", *xyz] for i, xyz in enumerate(
                [(0, 0, 0), (4, 0, 0), (0, 4, 0), (0, 0, 4)], 1)]
            + [["L", 1, "ALA", "CA", "C", 1, 1, 1], ["L", 2, "GLY", "CA", "C", 2, 1, 1]],
            "prediction_atom_mapping": {"L": [
                {"model": [1, "C000"], "original": [1, "CA"]},
                {"model": [1, "C001"], "original": [2, "CA"]},
            ]},
        }
        prediction = {(r[0], r[1], r[3]): np.asarray(r[5:8], dtype=float)
                      for r in ref["atoms"] if r[0] == "R"}
        prediction.update({("L", 1, "C000"): np.array([1., 1., 1.]),
                           ("L", 1, "C001"): np.array([2., 1., 1.])})
        return ref, prediction

    def test_mapping_and_epitope_scoring_do_not_fit_away_pose_error(self):
        ref, pred = self.scoring_fixture()
        self.assertAlmostEqual(score_pose(ref, pred, symmetry=False)["max_peptide_rmsd"], 0)
        self.assertAlmostEqual(epitope_ca_score(ref, pred), 0)
        for k in pred:
            if k[0] == "L":
                pred[k] += [7, 0, 0]
        self.assertAlmostEqual(epitope_ca_score(ref, pred), 7)
        del pred["L", 1, "C001"]
        with self.assertRaisesRegex(ValueError, "Missing"):
            epitope_ca_score(ref, pred)

    def test_strict_thresholds_incomplete_denominator_and_top_selection(self):
        cases = [{"name": "a"}, {"name": "b"}]
        rows = [{"name": "a", "seed": 101, "sample": i, "ranking_score": score,
                 "epitope_ca_rmsd": rmsd} for i, score, rmsd in [(0, .9, 5.), (1, .8, 2.)]]
        s = epitope_summary(cases, rows, 2)
        self.assertEqual(s["complete_cases"], 1)
        self.assertEqual(s["table"]["Top < 5A"], "0/2 (0.0%)")
        self.assertEqual(s["table"]["Best < 2A"], "0/2 (0.0%)")
        self.assertEqual(s["table"]["Best < 5A"], "1/2 (50.0%)")
        self.assertEqual(s["table"]["Best Mean RMSD"], 2.)


if __name__ == "__main__":
    unittest.main()
