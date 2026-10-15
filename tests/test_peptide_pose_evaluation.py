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

"""Deterministic binding-pose tests; no GPU, checkpoint, or CCD download needed."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HAS_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "scipy", "biotite", "torch", "rdkit")
)
if HAS_DEPS:
    import biotite.structure as struc
    import numpy as np
    from biotite.structure.io.pdbx import CIFBlock, CIFCategory, CIFFile

    from protenix.data.core.parser import AddAtomArrayAnnot
    from protenix.data.tokenizer import AtomArrayTokenizer
    from scripts.evaluation import (
        evaluate_peptide_pose as driver,
        peptide_pose_inputs as inputs,
        peptide_pose_metrics as metrics,
    )


def reference_fixture():
    receptor = [(0, 0, 0), (4, 0, 0), (0, 4, 0), (0, 0, 4)]
    atoms = [["A", i, "ALA", "CA", "C", *xyz] for i, xyz in enumerate(receptor, 1)]
    atoms += [
        ["L", 1, "ASP", name, element, *xyz]
        for name, element, xyz in (
            ("CA", "C", (1, 1, 1)),
            ("OD1", "O", (1, 2, 1)),
            ("OD2", "O", (2, 1, 1)),
        )
    ]
    return {
        "chains": {
            "A": {"role": "receptor", "group": "receptor"},
            "L": {"role": "peptide", "group": "peptide"},
        },
        "peptide_chains": ["L"],
        "atoms": atoms,
        "fixed_atoms": {},
    }


def coordinates(reference):
    return {
        (r[0], r[1], r[3]): np.asarray(r[5:8], dtype=float) for r in reference["atoms"]
    }


@unittest.skipUnless(HAS_DEPS, "Requires Protenix scientific Python dependencies")
class PoseMetricsTests(unittest.TestCase):
    def setUp(self):
        self.reference = reference_fixture()
        self.prediction = coordinates(self.reference)

    def test_common_rigid_transform_preserves_pose_and_contacts(self):
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        prediction = {k: v @ rotation + [10, 8, -6] for k, v in self.prediction.items()}
        result = metrics.score_pose(self.reference, prediction, symmetry=False)
        self.assertAlmostEqual(result["max_peptide_rmsd"], 0)
        self.assertAlmostEqual(result["receptor_rmsd"], 0)
        self.assertEqual(result["peptides"][0]["fnat"], 1)

    def test_peptide_translation_is_not_fitted_away(self):
        for key in self.prediction:
            if key[0] == "L":
                self.prediction[key] += [12, 0, 0]
        result = metrics.score_pose(self.reference, self.prediction, symmetry=False)
        self.assertAlmostEqual(result["max_peptide_rmsd"], 12)
        self.assertEqual(result["peptides"][0]["fnat"], 0)

    def test_symmetry_swap_and_external_bond_anchor(self):
        info = {
            "atom_map": {"CA": 0, "OD1": 1, "OD2": 2},
            "atomic_number": np.array([6, 8, 8]),
            "perm": np.array([[0, 0], [1, 2], [2, 1]]),
        }
        self.prediction["L", 1, "OD1"], self.prediction["L", 1, "OD2"] = (
            self.prediction["L", 1, "OD2"],
            self.prediction["L", 1, "OD1"],
        )
        metrics.residue_permutations.cache_clear()
        try:
            with patch("protenix.data.core.ccd.get_ccd_ref_info", return_value=info):
                result = metrics.score_pose(self.reference, self.prediction)
                self.assertAlmostEqual(result["max_peptide_rmsd"], 0)
                self.assertGreater(result["peptides"][0]["atom_name_rmsd"], 1)
                self.reference["fixed_atoms"] = {"L:1": ["OD1"]}
                result = metrics.score_pose(self.reference, self.prediction)
                self.assertGreater(result["max_peptide_rmsd"], 1)
        finally:
            metrics.residue_permutations.cache_clear()

    def test_equivalent_peptides_have_one_to_one_assignment(self):
        self.reference["chains"]["M"] = dict(self.reference["chains"]["L"])
        self.reference["peptide_chains"].append("M")
        for row in list(self.reference["atoms"]):
            if row[0] == "L":
                self.reference["atoms"].append(["M", *row[1:5], row[5] + 20, *row[6:]])
        prediction = coordinates(self.reference)
        for key, value in list(prediction.items()):
            if key[0] == "L":
                prediction[key], prediction[("M", *key[1:])] = (
                    prediction[("M", *key[1:])],
                    value,
                )
        result = metrics.score_pose(self.reference, prediction, symmetry=False)
        self.assertAlmostEqual(result["max_peptide_rmsd"], 0)
        self.assertEqual(
            {p["predicted_peptide_chain"] for p in result["peptides"]}, {"L", "M"}
        )
        for key in prediction:
            if key[0] == "M":
                prediction[key] = prediction[("L", *key[1:])].copy()
        result = metrics.score_pose(self.reference, prediction, symmetry=False)
        self.assertGreater(result["max_peptide_rmsd"], 19)

    def test_symmetry_handles_hydrogen_rows_and_nonidentity_first_column(self):
        for perm in (
            np.array([[3, 0], [1, 1], [2, 2], [0, 3]]),
            np.array([[3, 0], [2, 2], [0, 3]]),
        ):
            info = {
                "atom_map": {"OD1": 0, "H": 1, "CA": 2, "OD2": 3},
                "atomic_number": np.array([8, 1, 6, 8]),
                "perm": perm,
            }
            metrics.residue_permutations.cache_clear()
            with patch("protenix.data.core.ccd.get_ccd_ref_info", return_value=info):
                permutations = metrics.residue_permutations(
                    "ASP", ("CA", "OD1", "OD2"), ()
                )
                self.assertEqual(set(permutations), {(0, 1, 2), (0, 2, 1)})
        metrics.residue_permutations.cache_clear()

    def test_equivalent_receptor_chains_can_swap(self):
        self.reference["chains"]["B"] = dict(self.reference["chains"]["A"])
        for row in list(self.reference["atoms"]):
            if row[0] == "A":
                self.reference["atoms"].append(["B", *row[1:5], row[5] + 20, *row[6:]])
        prediction = coordinates(self.reference)
        for key, value in list(prediction.items()):
            if key[0] == "A":
                prediction[key], prediction[("B", *key[1:])] = (
                    prediction[("B", *key[1:])],
                    value,
                )
        result = metrics.score_pose(self.reference, prediction, symmetry=False)
        self.assertEqual(result["receptor_mapping"], {"A": "B", "B": "A"})
        self.assertAlmostEqual(result["max_peptide_rmsd"], 0)

    def test_missing_peptide_atoms_fail(self):
        del self.prediction["L", 1, "OD1"]
        with self.assertRaisesRegex(ValueError, "Missing peptide atoms"):
            metrics.score_pose(self.reference, self.prediction, symmetry=False)

    def test_receptor_mapping_uses_receptor_fit_before_peptide_error(self):
        self.reference["chains"]["B"] = dict(self.reference["chains"]["A"])
        for row in list(self.reference["atoms"]):
            if row[0] == "A":
                self.reference["atoms"].append(["B", *row[1:5], row[5] + 20, *row[6:]])
        prediction = coordinates(self.reference)
        fits = [
            (np.eye(3), np.array([1e-9, 0, 0]), 0.0),
            (np.eye(3), np.zeros(3), 20.0),
        ]
        with patch.object(metrics, "receptor_fit", side_effect=fits):
            result = metrics.score_pose(self.reference, prediction, symmetry=False)
        self.assertEqual(result["receptor_mapping"], {"A": "A", "B": "B"})
        self.assertEqual(result["receptor_rmsd"], 0)

    def test_insufficient_receptor_coverage_fails(self):
        del self.prediction["A", 1, "CA"]
        with self.assertRaisesRegex(ValueError, "coverage"):
            metrics.score_pose(self.reference, self.prediction, symmetry=False)

    def test_collinear_alignment_and_nonfinite_coordinates_fail(self):
        with self.assertRaisesRegex(ValueError, "collinear"):
            metrics.receptor_fit([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0]] * 3)
        self.prediction["L", 1, "CA"][:] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite peptide"):
            metrics.score_pose(self.reference, self.prediction, symmetry=False)

    def test_confidence_top1_oracle_and_full_denominator(self):
        cases = [
            {"name": "good", "status": "ready"},
            {"name": "failed", "status": "failed"},
            {"name": "partial", "status": "ready"},
        ]
        rows = [
            {
                "name": "good",
                "seed": 101,
                "sample": 0,
                "ranking_score": 0.9,
                "max_peptide_rmsd": 8,
            },
            {
                "name": "good",
                "seed": 101,
                "sample": 1,
                "ranking_score": 0.5,
                "max_peptide_rmsd": 1,
            },
            {
                "name": "partial",
                "seed": 101,
                "sample": 0,
                "ranking_score": 1,
                "max_peptide_rmsd": 0,
            },
        ]
        summary, rates = driver.summarize_cases(cases, rows, 2, [2, 5])
        self.assertEqual(summary[0]["top1_sample"], 0)
        self.assertEqual(summary[0]["best_sample"], 1)
        self.assertEqual(rates["top1_success_2A"], 0)
        self.assertAlmostEqual(rates["best_success_2A"], 1 / 3)
        self.assertFalse(summary[2]["complete"])

    def test_multiple_seeds_rank_and_score_over_the_full_prediction_pool(self):
        rows = [
            {"name": "fixture", "seed": seed, "sample": sample, "ranking_score": rank,
             "max_peptide_rmsd": rmsd}
            for seed, sample, rank, rmsd in [(101, 0, 0.8, 8), (101, 1, 0.5, 1),
                                            (102, 0, 0.9, 4), (102, 1, 0.6, 2)]
        ]
        summary, _ = driver.summarize_cases([{"name": "fixture", "status": "ready"}], rows, 4, [2, 5])
        self.assertTrue(summary[0]['complete'])
        self.assertEqual((summary[0]['top1_seed'], summary[0]['top1_sample']), (102, 0))
        self.assertEqual((summary[0]['best_seed'], summary[0]['best_sample']), (101, 1))
        incomplete, _ = driver.summarize_cases(
            [{"name": "fixture", "status": "ready"}], rows[:2], 4, [2, 5],
        )
        self.assertFalse(incomplete[0]['complete'])

    def test_scoring_rejects_changed_prediction_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            case = {"name": "fixture", "status": "ready", "reference": "reference.json"}
            manifest = {"cases": [case]}
            inputs.write_json(root / "manifest.json", manifest)
            inputs.write_json(root / "reference.json", self.reference)
            run = {
                "seeds": [101],
                "samples": 1,
                "manifest_sha256": inputs.file_hash(root / "manifest.json"),
            }
            inputs.write_json(root / "run_config.json", run)
            cif, confidence = driver.prediction_paths(root, "fixture", 101, 0)
            inputs.write_json(confidence, {"ranking_score": 0.9})
            cif.write_text(
                "synthetic prediction; reader mocked for this provenance test"
            )
            progress = {
                "fixture": {
                    "status": "complete",
                    "run_sha256": driver.digest(run),
                    "files": {
                        str(p.relative_to(root)): inputs.file_hash(p)
                        for p in (cif, confidence)
                    },
                }
            }
            inputs.write_json(root / "prediction_manifest.json", progress)
            args = SimpleNamespace(
                output_dir=root,
                contact_cutoff=5.0,
                no_atom_symmetry=True,
                min_receptor_coverage=0.9,
                thresholds=[2, 5],
            )
            with patch.object(
                driver, "read_prediction", return_value=self.prediction
            ), patch("builtins.print"):
                self.assertTrue(driver.score(args, manifest))
                cif.write_text("changed")
                self.assertFalse(driver.score(args, manifest))
            summary = driver.read_json(root / "scores/summary.json")
            self.assertEqual(summary["complete_complexes"], 0)
            self.assertEqual(summary["requested_complexes"], 1)
            self.assertEqual(
                summary["success_rates_all_requested_complexes"]["best_success_2A"], 0
            )


@unittest.skipUnless(HAS_DEPS, "Requires Protenix scientific Python dependencies")
class PoseInputTests(unittest.TestCase):
    def test_polymer_amino_acids_and_explicit_nonpolymer_have_different_tokens(self):
        atoms = struc.AtomArray(10)
        atoms.chain_id = np.array(["A"] * 5 + ["L"] * 5)
        atoms.res_name[:] = "ALA"
        atoms.res_id[:] = 1
        atoms.atom_name = np.array(["N", "CA", "C", "O", "CB"] * 2)
        atoms.element = np.array(["N", "C", "C", "O", "C"] * 2)
        atoms.set_annotation("label_entity_id", np.array(["1"] * 5 + ["2"] * 5))
        with patch("protenix.data.core.ccd.get_mol_type", return_value="protein"):
            AddAtomArrayAnnot.add_token_mol_type(
                atoms, {"1": "polypeptide(L)", "2": "non-polymer"}
            )
        AddAtomArrayAnnot.add_atom_mol_type_mask(atoms)
        AddAtomArrayAnnot.add_centre_atom_mask(atoms)
        tokens = AtomArrayTokenizer(atoms).get_token_array()
        self.assertEqual(len(tokens), 6)
        self.assertEqual(len(tokens[0].atom_indices), 5)
        self.assertTrue(np.all(atoms.is_protein[:5]))
        self.assertTrue(np.all(atoms.is_ligand[5:]))
        self.assertTrue(all(len(token.atom_indices) == 1 for token in list(tokens)[1:]))

    def test_prepared_input_has_no_native_geometry_and_deduplicates_bonds(self):
        columns = {
            "label_asym_id": ["A", "L", "L"],
            "label_entity_id": ["1", "2", "2"],
            "label_seq_id": ["1", ".", "."],
            "auth_seq_id": ["1", "1", "2"],
            "label_comp_id": ["ALA", "ALA", "GLY"],
            "label_atom_id": ["CA", "C", "N"],
            "type_symbol": ["C", "C", "N"],
            "Cartn_x": ["0", "1", "2"],
            "Cartn_y": ["0", "0", "0"],
            "Cartn_z": ["0", "0", "0"],
        }
        bonds = {
            "id": ["b1", "b2"],
            "conn_type_id": ["covale"] * 2,
            "pdbx_value_order": ["SING"] * 2,
        }
        for side, residue, atom in ((1, "ALA", "C"), (2, "GLY", "N")):
            bonds.update(
                {
                    f"ptnr{side}_label_asym_id": ["L"] * 2,
                    f"ptnr{side}_label_seq_id": ["."] * 2,
                    f"ptnr{side}_label_comp_id": [residue] * 2,
                    f"ptnr{side}_label_atom_id": [atom] * 2,
                }
            )
        # Add the export's spurious self-link and its observed N atom.
        for key, values in columns.items():
            values.append("N" if key in {"label_atom_id", "type_symbol"} else values[1])
        for key, values in bonds.items():
            values.append("self" if key == "id" else values[0])
        bonds["ptnr2_label_comp_id"][-1] = "ALA"
        block = CIFBlock()
        block["atom_site"] = CIFCategory(columns)
        block["entity_poly"] = CIFCategory(
            {"entity_id": ["1"], "type": ["polypeptide(L)"]}
        )
        block["entity_poly_seq"] = CIFCategory(
            {"entity_id": ["1"], "num": ["1"], "mon_id": ["ALA"]}
        )
        block["foldbench_peptide_ligand"] = CIFCategory({"id": ["L"]})
        block["struct_conn"] = CIFCategory(bonds)
        cif = CIFFile()
        cif["fixture"] = block
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "fixture.cif"
            cif.write(source)
            aliases = {name: name for name in ("N", "CA", "C", "O", "CB")}
            with patch.object(
                inputs, "canonical_atom_names", return_value=aliases
            ), patch.object(
                inputs, "canonical_bonds", return_value={("C", "CA"), ("CA", "N")}
            ):
                job, reference = inputs.prepare_case(source)
                block["atom_site"]["Cartn_x"] = ["10", "11", "12", "11"]
                cif.write(source)
                shifted_job, shifted_reference = inputs.prepare_case(source)
            self.assertEqual(job, shifted_job)
            self.assertNotEqual(reference["atoms"], shifted_reference["atoms"])
            self.assertEqual(len(job["covalent_bonds"]), 1)
            self.assertIn("self-link", reference["normalizations"][0]["change"])
            self.assertEqual(set(job), {"name", "sequences", "covalent_bonds"})
            self.assertEqual(job["sequences"][1]["ligand"]["ligand"], "CCD_ALA_GLY")
            self.assertNotIn("coord", json.dumps(job))

    def test_modified_amino_acid_preserves_parent_sequence(self):
        with patch("protenix.data.core.ccd.get_one_letter_code", return_value="M"):
            kind, spec = inputs.polymer_spec("polypeptide(L)", ["ALA", "MSE"])
        self.assertEqual(kind, "proteinChain")
        self.assertEqual(spec["sequence"], "AM")
        self.assertEqual(
            spec["modifications"], [{"ptmPosition": 2, "ptmType": "CCD_MSE"}]
        )


if __name__ == "__main__":
    unittest.main()
