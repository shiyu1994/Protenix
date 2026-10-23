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

"""Exercise custom chemistry, cache validity, and launcher selection without GPUs."""

import importlib.util
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
HAS_CHEMISTRY = all(
    importlib.util.find_spec(name) is not None for name in ("biotite", "rdkit", "torch")
)
if HAS_CHEMISTRY:
    import biotite.structure as struc
    import numpy as np
    from biotite.structure.io.pdbx import CIFCategory
    from rdkit import Chem

    from protenix.data.core.featurizer import Featurizer
    from protenix.data.pipeline import peptide_repair
    from protenix.data.tokenizer import Token
    from scripts import prepare_peptide_training_data as preparation


@unittest.skipUnless(HAS_CHEMISTRY, "Requires the Protenix chemistry dependencies")
class PeptidePreparationTests(unittest.TestCase):
    def make_atoms(self):
        atoms = struc.AtomArray(3)
        atoms.res_name[:] = "PEP"
        atoms.atom_name = np.array(["ATM000", "ATM001", "ATM002"])
        atoms.element = np.array(["C", "C", "O"])
        atoms.coord = np.array([[0, 0, 0], [4, 0, 0], [8, 0.2, 0]], dtype=np.float32)
        atoms.set_annotation("label_asym_id", np.array(["B"] * 3))
        atoms.set_annotation("label_seq_id", np.array(["1"] * 3))
        atoms.bonds = struc.BondList(3, np.array([[0, 1, 1], [1, 2, 1]]))
        return atoms

    def test_explicit_double_bond_and_independent_reference(self):
        atoms = self.make_atoms()
        columns = {
            "conn_type_id": ["covale", "covale"],
            "pdbx_value_order": ["SING", "DOUB"],
        }
        for n, names in ((1, ["ATM000", "ATM001"]), (2, ["ATM001", "ATM002"])):
            columns.update(
                {
                    f"ptnr{n}_label_asym_id": ["B", "B"],
                    f"ptnr{n}_label_seq_id": ["1", "1"],
                    f"ptnr{n}_label_comp_id": ["PEP", "PEP"],
                    f"ptnr{n}_label_atom_id": names,
                }
            )
        parser = SimpleNamespace(
            cif=SimpleNamespace(block={"struct_conn": CIFCategory(columns)})
        )
        original = atoms.coord.copy()
        explicit = preparation.restore_explicit_bonds(parser, atoms)
        self.assertEqual(explicit[(1, 2)], 2)
        self.assertEqual(atoms.bonds.as_array()[1, 2], 2)
        mol, names = preparation.make_reference(atoms, np.arange(3), seed=42)
        self.assertEqual(mol.GetNumAtoms(), 3)
        self.assertEqual(
            float(mol.GetBondBetweenAtoms(1, 2).GetBondTypeAsDouble()), 2.0
        )
        self.assertEqual(len(set(names)), 3)
        ref = mol.GetConformer().GetPositions()
        self.assertLess(np.linalg.norm(ref[1] - ref[2]), 2)
        np.testing.assert_array_equal(atoms.coord, original)

    def test_atom_names_do_not_collide_after_four_character_encoding(self):
        names = [preparation.atom_name(i, "C") for i in range(2000)]
        self.assertTrue(all(len(name) == 4 for name in names))
        self.assertEqual(len(set(name[:4] for name in names)), 2000)

    def test_unambiguous_pyrrole_hydrogen_is_recovered(self):
        atoms = struc.AtomArray(5)
        atoms.element = np.array(["N", "C", "C", "C", "C"])
        angles = np.arange(5) * 2 * np.pi / 5
        atoms.coord = np.column_stack(
            (np.cos(angles), np.sin(angles), np.zeros(5))
        ).astype(np.float32)
        atoms.bonds = struc.BondList(
            5,
            np.array(
                [[i, (i + 1) % 5, int(struc.BondType.AROMATIC)] for i in range(5)]
            ),
        )
        mol, names = preparation.make_reference(atoms, np.arange(5), seed=42)
        self.assertEqual(mol.inferred_hydrogen_atoms, [names[0]])
        self.assertEqual(mol.GetAtomWithIdx(0).GetTotalNumHs(), 1)

    def test_missing_bond_endpoint_is_rejected(self):
        atoms = self.make_atoms()
        columns = {"conn_type_id": ["covale"], "pdbx_value_order": ["SING"]}
        for n, name in ((1, "ATM000"), (2, "ABSENT")):
            columns.update(
                {
                    f"ptnr{n}_label_asym_id": ["B"],
                    f"ptnr{n}_label_seq_id": ["1"],
                    f"ptnr{n}_label_comp_id": ["PEP"],
                    f"ptnr{n}_label_atom_id": [name],
                }
            )
        parser = SimpleNamespace(
            cif=SimpleNamespace(block={"struct_conn": CIFCategory(columns)})
        )
        with self.assertRaisesRegex(ValueError, "endpoint"):
            preparation.restore_explicit_bonds(parser, atoms)

    def test_imidazolium_hydrogens_preserve_charge_and_connectivity(self):
        template = Chem.MolFromSmiles("c1[nH+]cc[nH]1")
        template.atom_map = {str(i): i for i in range(template.GetNumAtoms())}
        mol = Chem.Mol(template)
        for atom in mol.GetAtoms():
            atom.SetNumExplicitHs(0)
        atoms = struc.AtomArray(5)
        atoms.element = np.array([a.GetSymbol() for a in mol.GetAtoms()])
        atoms.coord[:, 0] = np.arange(5)
        original = SimpleNamespace(
            match=lambda coord, element: ("A", "1", "HIS", str(int(coord[0])), "1")
        )
        with patch.object(
            peptide_repair.ccd, "get_component_rdkit_mol", return_value=template
        ):
            inferred = peptide_repair.repair_aromatic_hydrogens(
                mol, atoms, np.arange(5), original
            )
        Chem.SanitizeMol(mol)
        self.assertEqual(inferred, [1, 4])
        self.assertEqual(Chem.MolToSmiles(mol), Chem.MolToSmiles(template))

    def test_failed_embedding_retries_with_independent_initialization(self):
        atoms = self.make_atoms()
        embed = preparation.AllChem.EmbedMolecule
        calls = []

        def fail_once(mol, options):
            calls.append((options.randomSeed, options.useRandomCoords))
            return -1 if len(calls) == 1 else embed(mol, options)

        with patch.object(preparation.AllChem, "EmbedMolecule", side_effect=fail_once):
            mol, _ = preparation.make_reference(atoms, np.arange(3), seed=42)
        self.assertEqual(calls, [(42, True), (43, False)])
        self.assertEqual(mol.embedding_attempts[0]["result"], -1)
        self.assertEqual(mol.embedding_attempts[1]["result"], 0)
        self.assertTrue(np.isfinite(mol.GetConformer().GetPositions()).all())

    def test_failed_embeddings_never_fall_back_to_native_coordinates(self):
        atoms = self.make_atoms()
        with patch.object(
            preparation.AllChem, "EmbedMolecule", return_value=-1
        ) as embed:
            with self.assertRaisesRegex(
                ValueError, "reference conformer generation failed"
            ):
                preparation.make_reference(atoms, np.arange(3), seed=42)
        self.assertEqual(embed.call_count, 2)

    def test_atom_name_repair_handles_collisions_and_bond_endpoints(self):
        site = CIFCategory(
            {
                "label_asym_id": ["B", "B"],
                "label_seq_id": ["1", "1"],
                "label_comp_id": ["APA", "APA"],
                "label_atom_id": ["C1", "C1_2"],
                "auth_atom_id": ["C1", "C1_2"],
                "type_symbol": ["C", "C"],
                "Cartn_x": ["0", "1"],
                "Cartn_y": ["0", "0"],
                "Cartn_z": ["0", "0"],
            }
        )
        conn = {}
        for n, atom in ((1, "C1"), (2, "C1_2")):
            for field, value in zip(
                ("asym_id", "seq_id", "comp_id", "atom_id"), ("B", "1", "APA", atom)
            ):
                conn[f"ptnr{n}_label_{field}"] = [value]
        parser = SimpleNamespace(
            cif=SimpleNamespace(
                block={"atom_site": site, "struct_conn": CIFCategory(conn)}
            )
        )
        original = SimpleNamespace(
            match=lambda coord, element, residue: (
                "A",
                "1",
                "APA",
                ["C1'", "C1"][int(coord[0])],
                "1",
            )
        )
        with patch.object(
            peptide_repair.ccd,
            "get_component_rdkit_mol",
            return_value=SimpleNamespace(atom_map={"C1": 0, "C1'": 1}),
        ):
            changes = peptide_repair.repair_atom_names(parser, original)
        self.assertEqual(len(changes), 1)
        self.assertEqual(
            parser.cif.block["atom_site"]["label_atom_id"].as_array().tolist(),
            ["C1'", "C1"],
        )
        self.assertEqual(
            parser.cif.block["struct_conn"]["ptnr1_label_atom_id"].as_array().tolist(),
            ["C1'"],
        )

    def test_ambiguous_original_names_fail_closed(self):
        columns = {
            "label_asym_id": ["B"],
            "label_seq_id": ["1"],
            "label_comp_id": ["APA"],
            "label_atom_id": ["C1_2"],
            "auth_atom_id": ["C1_2"],
            "type_symbol": ["C"],
            "Cartn_x": ["0"],
            "Cartn_y": ["0"],
            "Cartn_z": ["0"],
        }
        parser = SimpleNamespace(
            cif=SimpleNamespace(block={"atom_site": CIFCategory(columns)})
        )
        with patch.object(
            peptide_repair.ccd,
            "get_component_rdkit_mol",
            return_value=SimpleNamespace(atom_map={"C1'": 0}),
        ):
            with self.assertRaisesRegex(ValueError, "Cannot uniquely restore"):
                peptide_repair.repair_atom_names(
                    parser, SimpleNamespace(match=lambda *args: None)
                )
        self.assertEqual(
            parser.cif.block["atom_site"]["label_atom_id"].as_array().tolist(), ["C1_2"]
        )

    def test_missing_centres_are_unresolved_without_dropping_observed_atoms(self):
        atoms = self.make_atoms()
        atoms.res_name[:] = "ALA"
        atoms.atom_name = np.array(["N", "C", "O"])
        atoms.element = np.array(["N", "C", "O"])
        atoms.set_annotation("label_entity_id", np.array(["1"] * 3))
        atoms.set_annotation("peptide_source_atom_index", np.arange(3))
        template = struc.AtomArray(3)
        template.atom_name = np.array(["N", "CA", "CB"])
        template.element = np.array(["N", "C", "C"])
        template.bonds = struc.BondList(3, np.array([[0, 1, 1], [1, 2, 1]]))
        with patch.object(
            peptide_repair.ccd, "get_component_atom_array", return_value=template
        ):
            result = peptide_repair.add_missing_representatives(
                atoms, {"1": "polypeptide(L)"}
            )
        self.assertEqual(result.atom_name.tolist(), ["N", "C", "O", "CA", "CB"])
        self.assertEqual(result.is_resolved.tolist(), [True, True, True, False, False])
        self.assertEqual(result.peptide_source_atom_index.tolist(), [0, 1, 2, -1, -1])
        np.testing.assert_array_equal(result.coord[:3], atoms.coord)
        np.testing.assert_array_equal(result.coord[3:], 0)
        self.assertEqual(len(result.bonds.as_array()), 4)

    def test_reuse_retries_previously_failed_entries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root / "old"
            output = root / "new"
            for p in (old / "records", output / "records", output / "bioassembly"):
                p.mkdir(parents=True)
            source = root / "example.cif"
            source.write_text("source")
            preparation.write_json(
                old / "records/example.json",
                {"status": "failed", "error": "old failure"},
            )

            def convert(path, destination, seed, original):
                destination.write_bytes(b"repaired")
                return [{"pdb_id": "example"}], {"repaired": True}

            with patch.object(preparation, "convert", side_effect=convert) as retry:
                result = preparation.process_file(
                    (
                        source,
                        output,
                        {"reuse_prepared_dir": str(old), "seed": 42},
                        False,
                    )
                )
            self.assertEqual(result["status"], "ok")
            retry.assert_called_once()

    def test_cache_detects_changed_source_and_corrupt_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "records").mkdir()
            (root / "bioassembly").mkdir()
            source = root / "example.cif"
            source.write_text("source")
            output = root / "bioassembly/example.pkl.gz"
            output.write_bytes(b"test cache")
            signature = {
                "source": str(source),
                "source_sha256": preparation.file_hash(source),
                "settings": {},
            }
            preparation.write_json(
                root / "records/example.json",
                {
                    "status": "ok",
                    "signature": signature,
                    "rows": [{"num_tokens": np.int64(3)}],
                    "output_sha256": preparation.file_hash(output),
                },
            )
            with patch.object(
                preparation,
                "convert",
                side_effect=AssertionError("Must not reconvert in --check"),
            ):
                self.assertEqual(
                    preparation.process_file((source, root, {}, True))["status"], "ok"
                )
                source.write_text("changed")
                self.assertEqual(
                    preparation.process_file((source, root, {}, True))["status"],
                    "failed",
                )
                source.write_text("source")
                output.write_bytes(b"corrupt")
                self.assertEqual(
                    preparation.process_file((source, root, {}, True))["status"],
                    "failed",
                )
            self.assertIsInstance(
                json.loads((root / "records/example.json").read_text())["rows"][0][
                    "num_tokens"
                ],
                int,
            )


@unittest.skipUnless(HAS_CHEMISTRY, "Requires the Protenix chemistry dependencies")
class PeptideFrameTests(unittest.TestCase):
    def test_incomplete_polymer_backbones_have_masked_frames(self):
        for mol_type, names in (
            ("protein", ["N", "CA", "C"]),
            ("rna", ["C1'", "C3'", "C4'"]),
            ("dna", ["C1'", "C3'", "C4'"]),
        ):
            for missing in names:
                with self.subTest(mol_type=mol_type, missing=missing):
                    observed = [name for name in names if name != missing]
                    token = Token(0, atom_names=observed, atom_indices=[10, 20])
                    self.assertEqual(
                        Featurizer.get_prot_nuc_frame(
                            token, SimpleNamespace(mol_type=mol_type)
                        ),
                        (0, [-1, -1, -1]),
                    )

    def test_complete_polymer_frames_keep_global_atom_indices(self):
        for mol_type, names in (
            ("protein", ["N", "CA", "C"]),
            ("rna", ["C1'", "C3'", "C4'"]),
            ("dna", ["C1'", "C3'", "C4'"]),
        ):
            with self.subTest(mol_type=mol_type):
                token = Token(0, atom_names=names[::-1], atom_indices=[30, 20, 10])
                self.assertEqual(
                    Featurizer.get_prot_nuc_frame(
                        token, SimpleNamespace(mol_type=mol_type)
                    ),
                    (1, [10, 20, 30]),
                )


class PeptideLauncherTests(unittest.TestCase):
    def command(self, mode=None, script="finetune_peptide.sh", **settings):
        env = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "CIF_IDS",
                "RESUME_CHECKPOINT",
                "CHECKPOINT_PATH",
                "SKIP_PREP_ERRORS",
                "NPROC_PER_NODE",
            }
        }
        env.update(settings)
        args = ["bash", str(REPO / script)]
        if mode:
            args.append(mode)
        result = subprocess.run(
            [*args, "--dry-run"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return {
            line.split(": ", 1)[0]: shlex.split(line.split(": ", 1)[1])
            for line in result.stdout.splitlines()
            if "command: " in line
        }

    def test_default_checks_cache_and_loads_pretrained_parameters(self):
        commands = self.command()
        self.assertIn("--check", commands["Preprocessing command"])
        self.assertNotIn("--ids", commands["Preprocessing command"])
        self.assertNotIn("--skip-errors", commands["Preprocessing command"])
        train = commands["Training command"]
        self.assertEqual(train[train.index("--load_params_only") + 1], "true")
        self.assertEqual(train[train.index("--model_name") + 1], "protenix-v2")

    def test_all_mode_explicitly_prepares_before_training(self):
        commands = self.command("all")
        self.assertNotIn("--check", commands["Preprocessing command"])
        self.assertIn("Training command", commands)

    def test_prepare_mode_does_not_launch_training(self):
        commands = self.command("prepare", CIF_IDS="1a07,1a0n")
        self.assertNotIn("Training command", commands)
        self.assertIn("1a07,1a0n", commands["Preprocessing command"])

    def test_standalone_preprocessing_matches_training_cache_contract(self):
        for settings in (
            {},
            {
                "DATA_DIR": "/tmp/peptide inputs",
                "PREPARED_DIR": "/tmp/peptide cache",
                "PYTHON_BIN": "/tmp/custom python",
                "PREP_WORKERS": "3",
                "SEED": "123",
                "CIF_IDS": "1a07,1a0n",
                "SKIP_PREP_ERRORS": "1",
            },
        ):
            for mode, standalone_mode in (("prepare", None), ("train", "--check")):
                with self.subTest(settings=settings, mode=mode):
                    commands = self.command(mode, **settings)
                    standalone = self.command(
                        standalone_mode, script="preprocess_peptide.sh", **settings
                    )
                    self.assertNotIn("Training command", standalone)
                    self.assertEqual(
                        standalone["Preprocessing command"],
                        commands["Preprocessing command"],
                    )

    def test_resume_uses_raw_checkpoint_and_checks_preparation(self):
        commands = self.command(
            "train", RESUME_CHECKPOINT="/tmp/raw.pt", NPROC_PER_NODE="2"
        )
        self.assertIn("--check", commands["Preprocessing command"])
        train = commands["Training command"]
        self.assertEqual(
            train[train.index("--load_checkpoint_path") + 1], "/tmp/raw.pt"
        )
        self.assertEqual(train[train.index("--load_params_only") + 1], "false")
        self.assertEqual(train[train.index("--nproc_per_node") + 1], "2")


if __name__ == "__main__":
    unittest.main()
