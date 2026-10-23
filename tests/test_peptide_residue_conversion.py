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

"""Check residue export preserves chemistry and receptor-aligned pose scores."""

import tempfile
import unittest
from pathlib import Path

from biotite.structure.io.pdbx import CIFBlock, CIFCategory, CIFFile

from scripts.analysis.convert_peptide_predictions_to_residues import convert_prediction
from scripts.analysis.recalculate_peptide_ca_rmsd import ca_reference
from scripts.evaluation.peptide_pose_inputs import category_rows
from scripts.evaluation.peptide_pose_metrics import read_prediction, score_pose


class TestPeptideResidueConversion(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source.cif"
        self.output = Path(self.temp.name) / "converted.cif"
        self.reference = {
            "name": "fixture", "peptide_chains": ["P"], "fixed_atoms": {},
            "chains": {"R": {"role": "receptor", "group": "r"},
                       "P": {"role": "peptide", "group": "p", "residues": ["ALA", "DPN", "NH2"]}},
            "atoms": [["R", 1, "ALA", "CA", "C", 0, 0, 0],
                      ["R", 2, "ALA", "CA", "C", 4, 0, 0],
                      ["R", 3, "ALA", "CA", "C", 0, 4, 0],
                      ["P", 1, "ALA", "CA", "C", 1, 1, 1],
                      ["P", 2, "DPN", "CA", "C", 2, 1, 1],
                      ["P", 3, "NH2", "N", "N", 3, 1, 1]],
        }
        atoms = []
        for chain, position, resname, atom, element, x, y, z in self.reference["atoms"]:
            atoms.append({
                "group_PDB": "ATOM" if chain == "R" else "HETATM", "label_asym_id": chain,
                "auth_asym_id": chain, "label_entity_id": "1" if chain == "R" else "2",
                "label_seq_id": str(position), "auth_seq_id": str(position), "label_comp_id": resname,
                "label_atom_id": atom, "type_symbol": element, "Cartn_x": str(x), "Cartn_y": str(y),
                "Cartn_z": str(z + (3 if chain == "P" else 0)), "B_iso_or_equiv": "82.12", "occupancy": "1",
            })
        self.cif = CIFFile({"fixture": CIFBlock()})
        self.cif.block["atom_site"] = CIFCategory({k: [row[k] for row in atoms] for k in atoms[0]})
        self.cif.block["entity"] = CIFCategory({"id": ["1", "2"], "type": ["polymer", "non-polymer"]})
        self.cif.block["entity_poly"] = CIFCategory({"entity_id": ["1"], "type": ["polypeptide(L)"]})
        self.cif.block["entity_poly_seq"] = CIFCategory({
            "entity_id": ["1"] * 3, "num": ["1", "2", "3"], "mon_id": ["ALA"] * 3, "hetero": ["n"] * 3,
        })
        self.cif.block["struct_conn"] = CIFCategory({
            "id": ["cap"], "conn_type_id": ["covale"], "ptnr1_label_asym_id": ["P"],
            "ptnr2_label_asym_id": ["P"], "ptnr1_label_seq_id": ["2"], "ptnr2_label_seq_id": ["3"],
            "ptnr1_label_atom_id": ["C"], "ptnr2_label_atom_id": ["N"],
        })
        self.cif.write(self.source)

    def test_export_preserves_modifications_caps_bonds_and_score(self):
        original_bytes = self.source.read_bytes()
        convert_prediction(self.source, self.output, self.reference)
        block = CIFFile.read(self.output).block
        self.assertEqual([r["type"] for r in category_rows(block, "entity")], ["polymer", "polymer"])
        sequence = [r["mon_id"] for r in category_rows(block, "entity_poly_seq") if r["entity_id"] == "2"]
        self.assertEqual(sequence, ["ALA", "DPN", "NH2"])
        peptide = [r for r in category_rows(block, "atom_site") if r["label_asym_id"] == "P"]
        self.assertEqual([r["group_PDB"] for r in peptide], ["ATOM", "HETATM", "HETATM"])
        self.assertEqual(category_rows(block, "struct_conn"), category_rows(self.cif.block, "struct_conn"))
        self.assertEqual(original_bytes, self.source.read_bytes())
        ca, _ = ca_reference(self.reference)
        for path in (self.source, self.output):
            score = score_pose(ca, read_prediction(path), symmetry=False)
            self.assertAlmostEqual(score["max_peptide_rmsd"], 3)

    def test_rejects_incorrect_residue_mapping(self):
        self.reference["chains"]["P"]["residues"][0] = "GLY"
        with self.assertRaisesRegex(ValueError, "Residue identity mismatch"):
            convert_prediction(self.source, self.output, self.reference)
        self.assertFalse(self.output.exists())

    def test_rejects_existing_output(self):
        self.output.write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            convert_prediction(self.source, self.output, self.reference)
        self.assertEqual(self.output.read_text(), "keep")

    def test_rejects_entity_shared_with_receptor(self):
        self.cif.block["atom_site"]["label_entity_id"] = ["1"] * 6
        self.cif.write(self.source)
        with self.assertRaisesRegex(ValueError, "shares entity"):
            convert_prediction(self.source, self.output, self.reference)


if __name__ == "__main__":
    unittest.main()
