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

"""Fixed-coordinate matching and controlled peptide MSA conversion tests."""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts.analysis.rescore_peptide_residue_confidence import (
    audit_features,
    coordinate_batch,
    msa_bioassembly,
    set_single_atom_residue_centres,
)


class TestResidueConfidence(unittest.TestCase):
    def test_single_nitrogen_glycine_keeps_coordinates_and_invalid_frame(self):
        import biotite.structure as struc
        from protenix.data.core.featurizer import Featurizer
        from protenix.data.tokenizer import AtomArrayTokenizer

        atoms = struc.AtomArray(1)
        atoms.chain_id[:] = "P"
        atoms.res_name[:] = "GLY"
        atoms.res_id[:] = 10
        atoms.atom_name[:] = "N"
        atoms.element[:] = "N"
        atoms.coord[:] = [[1, 2, 3]]
        atoms.set_annotation("mol_type", np.array(["protein"]))
        atoms.set_annotation("centre_atom_mask", np.array([0]))
        atoms.set_annotation("ref_space_uid", np.array([0]))
        before = atoms.coord.copy()
        set_single_atom_residue_centres(atoms, ["P"])
        tokens = AtomArrayTokenizer(atoms).get_token_array()
        tokens = Featurizer.get_token_frame(tokens, atoms, torch.tensor(before), torch.ones(1))
        self.assertEqual(len(tokens), 1)
        self.assertFalse(tokens[0].has_frame)
        self.assertTrue(atoms.rescore_centre_fallback[0])
        np.testing.assert_array_equal(atoms.coord, before)

    def test_coordinate_mapping_uses_identity_not_file_order(self):
        atoms = SimpleNamespace(chain_id=np.array(["P", "R"]), res_id=np.array([1, 2]),
                                atom_name=np.array(["CA", "N"]))
        saved = {("R", 2, "N"): np.array([4, 5, 6]), ("P", 1, "CA"): np.array([1, 2, 3])}
        with patch("scripts.analysis.rescore_peptide_residue_confidence.read_prediction", return_value=saved):
            coords = coordinate_batch(atoms, ["pose.cif"])
        np.testing.assert_array_equal(coords, [[[1, 2, 3], [4, 5, 6]]])

    def test_missing_or_extra_atoms_are_rejected(self):
        atoms = SimpleNamespace(chain_id=np.array(["P"]), res_id=np.array([1]), atom_name=np.array(["CA"]))
        for saved in ({}, {("P", 1, "CA"): [1, 2, 3], ("P", 1, "CB"): [2, 3, 4]}):
            with patch("scripts.analysis.rescore_peptide_residue_confidence.read_prediction", return_value=saved):
                with self.assertRaisesRegex(ValueError, "Atom inventory mismatch"):
                    coordinate_batch(atoms, ["pose.cif"])

    def test_training_aligned_names_restore_residue_coordinates(self):
        atoms = SimpleNamespace(chain_id=np.array(["P", "P"]), res_id=np.array([1, 2]),
                                atom_name=np.array(["CA", "CA"]))
        saved = {("P", 1, "C004"): np.array([1, 2, 3]), ("P", 1, "C00A"): np.array([4, 5, 6])}
        reference = {"prediction_atom_mapping": {"P": [
            {"model": [1, "C004"], "original": [1, "CA"]},
            {"model": [1, "C00A"], "original": [2, "CA"]},
        ]}}
        with patch("scripts.analysis.rescore_peptide_residue_confidence.read_prediction", return_value=saved):
            coords = coordinate_batch(atoms, ["pose.cif"], reference)
        np.testing.assert_array_equal(coords, [[[1, 2, 3], [4, 5, 6]]])

    def test_msa_conversion_preserves_receptor_and_cofactor(self):
        sample = {"sequences": [
            {"proteinChain": {"sequence": "AAA", "id": ["R"], "count": 1, "unpairedMsaPath": "original.a3m"}},
            {"ligand": {"ligand": "CCD_ALA_GLY", "id": ["P"], "count": 1}},
            {"ligand": {"ligand": "CCD_NAG", "id": ["C"], "count": 1}},
        ]}
        converted = msa_bioassembly(sample, ["P"])
        self.assertEqual(converted[0], sample["sequences"][0])
        self.assertEqual(converted[2], sample["sequences"][2])
        self.assertEqual(converted[1], {"proteinChain": {"sequence": "AG", "id": ["P"], "count": 1}})
        self.assertIn("ligand", sample["sequences"][1])

    def test_audit_rejects_atomized_standard_peptide(self):
        class Atoms(SimpleNamespace):
            def __len__(self):
                return len(self.chain_id)

        atoms = Atoms(chain_id=np.array(["P", "P"]), res_id=np.array([1, 1]),
                      res_name=np.array(["ALA", "ALA"]), mol_type=np.array(["ligand", "ligand"]))
        data = {"N_token": torch.tensor([2]), "N_msa": torch.tensor([1]), "input_feature_dict": {
            "atom_to_token_idx": torch.tensor([0, 1]), "is_protein": torch.tensor([0, 0]),
        }}
        with self.assertRaisesRegex(ValueError, "not tokenized as residue"):
            audit_features(atoms, data, ["P"])

    def test_peptide_msa_is_attached_by_exact_sequence(self):
        sample = {"sequences": [
            {"proteinChain": {"sequence": "AAA", "id": ["R"], "count": 1, "unpairedMsaPath": "receptor.a3m"}},
            {"ligand": {"ligand": "CCD_ALA_GLY", "id": ["P"], "count": 1}},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "peptide.a3m"
            path.write_text(">query\nAG\n>homolog\nSG\n")
            mapping = {"AG": {"unpairedMsaPath": str(path)}}
            result = msa_bioassembly(sample, ["P"], mapping)
            self.assertEqual(result[0], sample["sequences"][0])
            self.assertEqual(result[1]["proteinChain"]["unpairedMsaPath"], str(path))
            self.assertIn("ligand", sample["sequences"][1])
            with self.assertRaises(KeyError):
                msa_bioassembly(sample, ["P"], {"GA": mapping["AG"]})
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                msa_bioassembly(sample, ["P"], mapping)

    def test_ligand_audit_requires_atom_tokens_and_ligand_classification(self):
        class Atoms(SimpleNamespace):
            def __len__(self):
                return len(self.chain_id)

        atoms = Atoms(chain_id=np.array(["P", "P"]), res_id=np.array([1, 1]),
                      res_name=np.array(["ALA", "ALA"]), mol_type=np.array(["ligand", "ligand"]))
        features = {"atom_to_token_idx": torch.tensor([0, 1]), "is_protein": torch.tensor([0, 0]),
                    "is_ligand": torch.tensor([1, 1])}
        data = {"N_token": torch.tensor([2]), "N_msa": torch.tensor([1]), "input_feature_dict": features}
        audit = audit_features(atoms, data, ["P"], "ligand")
        self.assertEqual(audit["peptides"]["P"]["tokens"], 2)
        features["atom_to_token_idx"] = torch.tensor([0, 0])
        with self.assertRaisesRegex(ValueError, "not tokenized per atom"):
            audit_features(atoms, data, ["P"], "ligand")
        features["atom_to_token_idx"] = torch.tensor([0, 1])
        features["is_ligand"] = torch.tensor([1, 0])
        with self.assertRaisesRegex(ValueError, "not classified as ligand"):
            audit_features(atoms, data, ["P"], "ligand")


if __name__ == "__main__":
    unittest.main()
