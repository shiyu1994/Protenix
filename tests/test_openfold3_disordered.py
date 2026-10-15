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

"""Focused tests for the OpenFold3 disordered structure adapter."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import biotite.structure as struc
import numpy as np

from protenix.data.core import ccd
from protenix.data.core.openfold3_disordered import (
    OpenFold3DisorderedError,
    _validate_cache_interfaces,
    build_openfold3_disordered_data,
    read_openfold3_disordered_npz,
)
from protenix.data.core.parser import AddAtomArrayAnnot

_REAL_1A02 = Path(
    "/mnt/project_abd_hs/klp/databases/pdb_disordered/"
    "structures/structures/1a02/1a02.npz"
)


def _write_synthetic_structure(path: Path) -> np.ndarray:
    """Write a compact structure containing every OpenFold3 molecule type."""
    coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=np.float32,
    )
    bonds = np.array(
        [
            [0, 1, int(struc.BondType.SINGLE)],
            [2, 3, int(struc.BondType.SINGLE)],
            [6, 7, int(struc.BondType.SINGLE)],
        ],
        dtype=np.uint32,
    )
    np.savez_compressed(
        path,
        coord=coord,
        bonds=bonds,
        chain_id=np.array(["1", "1", "5", "5", "2", "3", "4", "4"]),
        res_id=np.ones(8, dtype=np.int64),
        ins_code=np.array([""] * 8),
        res_name=np.array(
            ["ALA", "ALA", "ALA", "ALA", "A", "DA", "LIG", "LIG"]
        ),
        hetero=np.array([False] * 6 + [True, True]),
        atom_name=np.array(["CA", "CB", "CA", "CB", "C1'", "C1'", "C1", "O1"]),
        element=np.array(["C", "C", "C", "C", "C", "C", "C", "O"]),
        charge=np.zeros(8, dtype=np.int64),
        occupancy=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        label_asym_id=np.array(["A", "A", "A", "A", "B", "C", "D", "D"]),
        auth_asym_id=np.array(["X", "X", "X", "X", "Y", "Z", "L", "L"]),
        molecule_type_id=np.array([0, 0, 0, 0, 1, 2, 3, 3]),
        entity_id=np.array([1, 1, 1, 1, 2, 3, 4, 4]),
    )
    return bonds


def _write_microheterogeneous_dna(
    path: Path,
    *,
    incomplete_first: bool = False,
) -> None:
    """Write two alternate bases at site 1 followed by canonical site 2."""
    coord = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [1.6, 0.1, 0.0],
            [3.2, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    occupancy = np.asarray([0.5] * 6)
    if incomplete_first:
        coord[1] = np.nan
        occupancy[1] = 0.0
    np.savez_compressed(
        path,
        coord=coord,
        bonds=np.asarray(
            [
                [0, 1, int(struc.BondType.SINGLE)],
                [2, 3, int(struc.BondType.SINGLE)],
                [3, 4, int(struc.BondType.SINGLE)],
                [4, 5, int(struc.BondType.SINGLE)],
            ],
            dtype=np.uint32,
        ),
        chain_id=np.asarray(["1"] * 6),
        res_id=np.asarray([1, 1, 1, 1, 2, 2]),
        ins_code=np.asarray([""] * 6),
        res_name=np.asarray(["DA", "DA", "DT", "DT", "DG", "DG"]),
        hetero=np.asarray([False] * 6),
        atom_name=np.asarray(["C1'", "O3'", "C1'", "O3'", "P", "C1'"]),
        element=np.asarray(["C", "O", "C", "O", "P", "C"]),
        charge=np.zeros(6, dtype=np.int64),
        occupancy=occupancy,
        label_asym_id=np.asarray(["A"] * 6),
        auth_asym_id=np.asarray(["A"] * 6),
        molecule_type_id=np.asarray([2] * 6),
        entity_id=np.asarray([1] * 6),
    )


def _write_microheterogeneous_ligand(path: Path) -> None:
    """Write two mutually exclusive sugars connected to one next component."""
    np.savez_compressed(
        path,
        coord=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [1.0, 0.1, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        bonds=np.asarray(
            [
                [0, 1, int(struc.BondType.SINGLE)],
                [2, 3, int(struc.BondType.SINGLE)],
                [1, 4, int(struc.BondType.SINGLE)],
                [3, 4, int(struc.BondType.SINGLE)],
                [4, 5, int(struc.BondType.SINGLE)],
            ],
            dtype=np.uint32,
        ),
        chain_id=np.asarray(["10"] * 6),
        res_id=np.asarray([1, 1, 1, 1, 2, 2]),
        ins_code=np.asarray([""] * 6),
        res_name=np.asarray(["BGC", "BGC", "GLC", "GLC", "GAL", "GAL"]),
        hetero=np.asarray([True] * 6),
        atom_name=np.asarray(["C1", "O3", "C1", "O3", "C1", "O4"]),
        element=np.asarray(["C", "O", "C", "O", "C", "O"]),
        charge=np.zeros(6, dtype=np.int64),
        occupancy=np.asarray([0.7, 0.7, 0.3, 0.3, 1.0, 1.0]),
        label_asym_id=np.asarray(["J"] * 6),
        auth_asym_id=np.asarray(["J"] * 6),
        molecule_type_id=np.asarray([3] * 6),
        entity_id=np.asarray([1] * 6),
    )


def _cache_entry() -> dict:
    """Return official-cache-shaped metadata for the synthetic structure."""

    def chain(
        label_asym_id,
        auth_asym_id,
        entity_id,
        molecule_type,
        alignment_representative_id,
        cluster_id,
        cluster_size,
    ):
        return {
            "label_asym_id": label_asym_id,
            "auth_asym_id": auth_asym_id,
            "entity_id": entity_id,
            "molecule_type": molecule_type,
            "reference_mol_id": None,
            "alignment_representative_id": alignment_representative_id,
            "template_ids": None,
            "cluster_id": cluster_id,
            "cluster_size": cluster_size,
        }

    return {
        "release_date": "2020-01-02",
        "resolution": 2.5,
        "chains": {
            "1": chain("A", "X", 1, "PROTEIN", "rep_A", "c_protein", 10),
            "5": chain("A", "X", 1, "PROTEIN", "rep_A", "c_protein", 10),
            "2": chain("B", "Y", 2, "RNA", "rep_B", "c_rna", 20),
            "3": chain("C", "Z", 3, "DNA", None, "c_dna", 30),
            "4": chain("D", "L", 4, "LIGAND", None, "c_ligand", 40),
        },
        "interfaces": {
            "1_2": {"cluster_id": "c_protein_c_rna", "cluster_size": 5},
            "5_2": {"cluster_id": "c_protein_c_rna", "cluster_size": 5},
        },
    }


def _fake_reference_features(atom_array, ccd_mols=None):
    """Add deterministic reference annotations without requiring an external CCD."""
    del ccd_mols
    atom_array.set_annotation(
        "ref_pos", np.zeros((len(atom_array), 3), dtype=np.float32)
    )
    atom_array.set_annotation("ref_charge", np.zeros(len(atom_array), dtype=int))
    atom_array.set_annotation("ref_mask", np.ones(len(atom_array), dtype=int))
    residue_starts = struc.get_residue_starts(
        atom_array, add_exclusive_stop=True
    )
    residue_permutations = []
    for start, stop in zip(residue_starts[:-1], residue_starts[1:]):
        residue_permutations.extend(str(index) for index in range(stop - start))
    atom_array.set_annotation("res_perm", residue_permutations)
    return atom_array


def _one_letter_code(res_name):
    """Return one-letter codes used by the synthetic fixture."""
    return {
        "ALA": "A",
        "A": "A",
        "DA": "A",
        "LIG": None,
    }.get(res_name)


class OpenFold3DisorderedTest(unittest.TestCase):
    """Exercise NPZ conversion, cache validation, and sampling metadata."""

    def setUp(self):
        """Create one isolated synthetic NPZ per test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.npz_path = Path(self.temp_dir.name) / "synthetic.npz"
        self.bonds = _write_synthetic_structure(self.npz_path)

    def _build(
        self,
        cache_entry=None,
        canonical_sequences_by_representative=None,
    ):
        with (
            mock.patch.object(
                AddAtomArrayAnnot,
                "add_ref_info_and_res_perm",
                side_effect=_fake_reference_features,
            ),
            mock.patch.object(
                ccd,
                "get_one_letter_code",
                side_effect=_one_letter_code,
            ),
        ):
            return build_openfold3_disordered_data(
                self.npz_path,
                pdb_id="synthetic",
                cache_entry=cache_entry,
                canonical_sequences_by_representative=(
                    canonical_sequences_by_representative
                ),
            )

    def test_builds_cache_backed_bioassembly_and_weighted_indices(self):
        """The adapter preserves structure data and official sampling metadata."""
        rows, bioassembly = self._build(_cache_entry())
        atom_array = bioassembly["atom_array"]

        self.assertEqual(
            bioassembly["entity_poly_type"],
            {
                "1": "polypeptide(L)",
                "2": "polyribonucleotide",
                "3": "polydeoxyribonucleotide",
            },
        )
        self.assertEqual(
            bioassembly["sequences"],
            {"1": "A", "2": "A", "3": "A"},
        )
        self.assertEqual(
            bioassembly["alignment_representative_id_by_chain_id"],
            {
                "1": "rep_A",
                "5": "rep_A",
                "2": "rep_B",
                "3": None,
                "4": None,
            },
        )
        self.assertEqual(bioassembly["openfold3_query_only_chain_ids"], [])
        np.testing.assert_array_equal(
            atom_array.bonds.as_array(),
            self.bonds,
        )
        np.testing.assert_array_equal(
            atom_array.is_resolved,
            [True, False, True, True, True, True, True, False],
        )
        np.testing.assert_array_equal(atom_array.coord[1], np.zeros(3))
        np.testing.assert_array_equal(atom_array.coord[7], np.zeros(3))
        self.assertTrue(np.isfinite(atom_array.coord).all())
        self.assertEqual(
            atom_array.mol_type.tolist(),
            [
                "protein",
                "protein",
                "protein",
                "protein",
                "rna",
                "dna",
                "ligand",
                "ligand",
            ],
        )
        self.assertEqual(len(bioassembly["token_array"]), 6)
        self.assertEqual(bioassembly["release_date"], "2020-01-02")
        self.assertEqual(bioassembly["resolution"], 2.5)

        chain_rows = {
            row["chain_1_id"]: row for row in rows if row["type"] == "chain"
        }
        self.assertEqual(set(chain_rows), {"1", "2", "3", "4", "5"})
        self.assertEqual(chain_rows["1"]["cluster_id"], "c_protein")
        self.assertEqual(chain_rows["1"]["cluster_size"], 10)
        self.assertAlmostEqual(chain_rows["1"]["weights"], 0.075)
        self.assertAlmostEqual(chain_rows["5"]["weights"], 0.075)
        self.assertAlmostEqual(chain_rows["2"]["weights"], 0.075)
        self.assertAlmostEqual(chain_rows["3"]["weights"], 0.05)
        self.assertAlmostEqual(chain_rows["4"]["weights"], 0.0125)

        interface_rows = [row for row in rows if row["type"] == "interface"]
        self.assertEqual(len(interface_rows), 2)
        for row in interface_rows:
            self.assertEqual(row["cluster_id"], "c_protein_c_rna")
            self.assertEqual(row["cluster_size"], 5)
            self.assertAlmostEqual(row["weights"], 0.6)
            self.assertEqual(row["mol_type_group"], "nuc_prot")
            self.assertEqual(row["eval_type"], "rna_prot")

    def test_query_only_field_preserves_physical_sequences(self):
        """Zero/ambiguous protein/RNA matches remain physical query-only chains."""
        cache_entry = copy.deepcopy(_cache_entry())
        cache_entry["chains"]["5"]["alignment_representative_id"] = "rep_C"

        _rows, bioassembly = self._build(
            cache_entry,
            canonical_sequences_by_representative={
                "rep_A": "A",
                "rep_B": "G",
                "rep_C": "A",
            },
        )

        self.assertEqual(
            bioassembly["alignment_representative_id_by_chain_id"],
            {
                "1": "rep_A",
                "5": "rep_C",
                "2": "rep_B",
                "3": None,
                "4": None,
            },
        )
        self.assertEqual(
            bioassembly["openfold3_query_only_chain_ids"],
            ["1", "2", "5"],
        )
        self.assertEqual(
            bioassembly["sequences"],
            {"1": "A", "2": "A", "3": "A"},
        )
        self.assertEqual(
            bioassembly["sequences_by_chain_id"],
            {"1": "A", "5": "A", "2": "A", "3": "A"},
        )

    def test_cache_chain_metadata_mismatch_is_rejected(self):
        """Retained NPZ chains must agree with their cache annotations."""
        cache_entry = copy.deepcopy(_cache_entry())
        cache_entry["chains"]["2"]["label_asym_id"] = "b"
        with self.assertRaisesRegex(
            OpenFold3DisorderedError,
            "label_asym_id mismatch",
        ):
            self._build(cache_entry)

    def test_nonfinite_cache_resolution_uses_unknown_sentinel(self):
        """Official JSON NaN resolution maps to Protenix's -1 sentinel."""
        cache_entry = copy.deepcopy(_cache_entry())
        cache_entry["resolution"] = float("nan")

        _rows, bioassembly = self._build(cache_entry)

        self.assertEqual(bioassembly["resolution"], -1.0)
        np.testing.assert_array_equal(
            bioassembly["atom_array"].resolution,
            np.full(len(bioassembly["atom_array"]), -1.0),
        )

    def test_polymer_microheterogeneity_keeps_one_site_and_repairs_bond(self):
        """Alternate polymer monomers are not treated as extra sequence sites."""
        _write_microheterogeneous_dna(self.npz_path)

        atom_array = read_openfold3_disordered_npz(self.npz_path)

        residue_starts = struc.get_residue_starts(atom_array)
        self.assertEqual(atom_array.res_name[residue_starts].tolist(), ["DA", "DG"])
        self.assertEqual(atom_array.res_id[residue_starts].tolist(), [1, 2])
        self.assertEqual(
            atom_array.openfold3_alternate_group_size[
                residue_starts
            ].tolist(),
            [2, 1],
        )
        o3_index = int(np.flatnonzero(atom_array.atom_name == "O3'")[0])
        p_index = int(np.flatnonzero(atom_array.atom_name == "P")[0])
        bonded_indices, _bond_types = atom_array.bonds.get_bonds(o3_index)
        self.assertIn(p_index, bonded_indices)

    def test_microheterogeneity_tie_preserves_canonical_source_component(self):
        """Missing atoms do not override the canonical first component on a tie."""
        _write_microheterogeneous_dna(
            self.npz_path,
            incomplete_first=True,
        )

        atom_array = read_openfold3_disordered_npz(self.npz_path)

        residue_starts = struc.get_residue_starts(atom_array)
        self.assertEqual(atom_array.res_name[residue_starts].tolist(), ["DA", "DG"])
        o3_index = int(np.flatnonzero(atom_array.atom_name == "O3'")[0])
        p_index = int(np.flatnonzero(atom_array.atom_name == "P")[0])
        self.assertFalse(atom_array.is_resolved[o3_index])
        bonded_indices, _bond_types = atom_array.bonds.get_bonds(o3_index)
        self.assertIn(p_index, bonded_indices)

    def test_ligand_microheterogeneity_keeps_highest_occupancy_component(self):
        """Mutually exclusive covalent sugars do not survive simultaneously."""
        _write_microheterogeneous_ligand(self.npz_path)

        atom_array = read_openfold3_disordered_npz(self.npz_path)

        residue_starts = struc.get_residue_starts(atom_array)
        self.assertEqual(
            atom_array.res_name[residue_starts].tolist(),
            ["BGC", "GAL"],
        )
        self.assertNotIn("GLC", atom_array.res_name.tolist())
        self.assertEqual(
            atom_array.openfold3_alternate_group_size[
                residue_starts
            ].tolist(),
            [2, 1],
        )
        o3_index = int(np.flatnonzero(atom_array.atom_name == "O3")[0])
        gal_c1_index = int(
            np.flatnonzero(
                (atom_array.res_name == "GAL")
                & (atom_array.atom_name == "C1")
            )[0]
        )
        bonded_indices, _bond_types = atom_array.bonds.get_bonds(o3_index)
        self.assertIn(gal_c1_index, bonded_indices)

    def test_cache_only_chain_and_interface_are_filtered(self):
        """Pocket filtering may remove a non-protein chain retained in the cache."""
        cache_entry = copy.deepcopy(_cache_entry())
        cache_entry["chains"]["9"] = {
            **copy.deepcopy(cache_entry["chains"]["4"]),
            "label_asym_id": "Z",
            "auth_asym_id": "Z",
            "entity_id": 9,
            "cluster_id": "cache_only_ligand",
        }
        cache_entry["interfaces"]["1_9"] = {
            "cluster_id": "cache_only_interface",
            "cluster_size": 1,
        }

        rows, bioassembly = self._build(cache_entry)

        self.assertNotIn(
            "9",
            bioassembly["alignment_representative_id_by_chain_id"],
        )
        self.assertFalse(
            any(
                row["chain_1_id"] == "9" or row["chain_2_id"] == "9"
                for row in rows
            )
        )

    def test_npz_chain_absent_from_cache_is_rejected(self):
        """A retained structural chain still requires authoritative metadata."""
        cache_entry = copy.deepcopy(_cache_entry())
        del cache_entry["chains"]["4"]
        with self.assertRaisesRegex(
            OpenFold3DisorderedError,
            "NPZ contains chain IDs absent",
        ):
            self._build(cache_entry)

    def test_cache_interface_without_a_live_entity_pair_is_filtered(self):
        """A parent interface is skipped when that entity pair no longer contacts."""
        cache_entry = copy.deepcopy(_cache_entry())
        cache_entry["interfaces"]["1_3"] = {
            "cluster_id": "not_an_interface",
            "cluster_size": 1,
        }
        rows, _ = self._build(cache_entry)
        self.assertFalse(
            any(row["cluster_id"] == "not_an_interface" for row in rows)
        )

    def test_interface_with_no_resolved_token_centre_is_safely_skipped(self):
        """Contacting side-chain atoms do not require a sampleable chain row."""
        with np.load(self.npz_path, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
        arrays["coord"][0] = np.nan
        arrays["occupancy"][0] = 0.0
        arrays["coord"][1] = np.asarray([0.0, 0.0, 0.0])
        arrays["occupancy"][1] = 1.0
        np.savez_compressed(self.npz_path, **arrays)

        rows, _bioassembly = self._build(_cache_entry())

        self.assertFalse(
            any(
                row["type"] == "interface"
                and row["chain_1_id"] == "1"
                for row in rows
            )
        )
        self.assertTrue(
            any(
                row["type"] == "interface"
                and row["chain_1_id"] == "5"
                for row in rows
            )
        )

    def test_cache_interface_accepts_an_equivalent_contacting_chain_copy(self):
        """Entity-level remapping may use a different homomer chain copy."""
        cache_entry = _cache_entry()
        interfaces = _validate_cache_interfaces(
            cache_entry,
            present_chain_ids=set(cache_entry["chains"]),
            actual_chain_pairs=[("5", "2")],
            chain_entity_ids={
                chain_id: str(chain["entity_id"])
                for chain_id, chain in cache_entry["chains"].items()
            },
        )
        self.assertEqual(
            [interface_id for interface_id, *_ in interfaces],
            ["1_2", "5_2"],
        )

    def test_interface_rows_follow_runtime_entity_pair_order(self):
        """Descending cache chain order is normalized to the runtime key."""
        cache_entry = copy.deepcopy(_cache_entry())
        cache_entry["interfaces"]["4_1"] = {
            "cluster_id": "c_ligand_c_protein",
            "cluster_size": 7,
        }

        rows, bioassembly = self._build(cache_entry)

        row = next(
            row
            for row in rows
            if row["cluster_id"] == "c_ligand_c_protein"
        )
        self.assertEqual(
            (row["entity_1_id"], row["entity_2_id"]),
            ("1", "4"),
        )
        self.assertEqual(
            (row["chain_1_id"], row["chain_2_id"]),
            ("1", "4"),
        )
        self.assertIn(
            ("1", "4"),
            bioassembly["entity_pair_to_chain_pairs"],
        )

    def test_no_cache_uses_stable_singleton_clusters(self):
        """Cache-free conversion remains deterministic for diagnostics and tests."""
        rows, bioassembly = self._build()
        self.assertFalse(bioassembly["openfold3_cache_backed"])
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["cluster_size"], 1)
            self.assertTrue(
                row["cluster_id"].startswith(
                    "openfold3-singleton:synthetic:"
                )
            )

    def test_reader_rejects_unknown_molecule_type(self):
        """Only the four documented OpenFold3 molecule type IDs are accepted."""
        with np.load(self.npz_path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name]) for name in archive.files}
        arrays["molecule_type_id"][0] = 9
        bad_path = Path(self.temp_dir.name) / "bad_type.npz"
        np.savez_compressed(bad_path, **arrays)
        with self.assertRaisesRegex(
            OpenFold3DisorderedError,
            "unsupported molecule_type_id",
        ):
            read_openfold3_disordered_npz(bad_path)

    def test_unknown_element_component_uses_standard_protenix_filter(self):
        """UNX is removed instead of inventing an unsupported atom element."""
        with np.load(self.npz_path, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
        ligand_mask = arrays["chain_id"] == "4"
        arrays["res_name"][ligand_mask] = "UNX"
        arrays["atom_name"][ligand_mask] = "UNK"
        arrays["element"][ligand_mask] = "X"
        unknown_path = Path(self.temp_dir.name) / "unknown_element.npz"
        np.savez_compressed(unknown_path, **arrays)

        atom_array = read_openfold3_disordered_npz(unknown_path)

        self.assertNotIn("4", atom_array.chain_id.tolist())
        self.assertNotIn("X", atom_array.element.tolist())

    def test_cache_backed_unknown_component_is_not_sampleable(self):
        """A cache chain/interface cannot revive a filtered UNX component."""
        with np.load(self.npz_path, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
        ligand_mask = arrays["chain_id"] == "4"
        arrays["res_name"][ligand_mask] = "UNX"
        arrays["atom_name"][ligand_mask] = "UNK"
        arrays["element"][ligand_mask] = "X"
        np.savez_compressed(self.npz_path, **arrays)

        cache_entry = _cache_entry()
        cache_entry["interfaces"]["1_4"] = {
            "cluster_id": "c_protein_c_unknown",
            "cluster_size": 7,
        }
        rows, bioassembly = self._build(cache_entry)

        self.assertNotIn("4", bioassembly["atom_array"].chain_id.tolist())
        self.assertNotIn(
            "4",
            bioassembly["alignment_representative_id_by_chain_id"],
        )
        self.assertTrue(
            all(
                row["chain_1_id"] != "4" and row["chain_2_id"] != "4"
                for row in rows
            )
        )
        self.assertTrue(
            all("4" not in pair for pair in bioassembly["entity_pair_to_chain_pairs"])
        )

    def test_unrecognized_x_element_remains_fail_closed(self):
        """Element X outside Protenix's audited residue mappings is rejected."""
        with np.load(self.npz_path, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
        ligand_atom = np.flatnonzero(arrays["chain_id"] == "4")[0]
        arrays["element"][ligand_atom] = "X"
        np.savez_compressed(self.npz_path, **arrays)

        with self.assertRaisesRegex(ValueError, "Unknown atom element: X"):
            self._build(_cache_entry())

    @unittest.skipUnless(_REAL_1A02.is_file(), "local OpenFold3 1a02 NPZ unavailable")
    def test_real_1a02_npz_reader(self):
        """The reader accepts a real local OpenFold3 disordered structure."""
        atom_array = read_openfold3_disordered_npz(_REAL_1A02)
        self.assertEqual(len(atom_array), 4151)
        self.assertEqual(atom_array.bonds.as_array().shape, (4305, 3))
        self.assertEqual(
            set(np.unique(atom_array.molecule_type_id).tolist()),
            {0, 2},
        )
        self.assertEqual(
            set(np.unique(atom_array.label_asym_id).tolist()),
            {"A", "B", "C", "D", "E"},
        )
        self.assertEqual(int(atom_array.is_resolved.sum()), 4145)
        self.assertTrue(np.isfinite(atom_array.coord).all())


if __name__ == "__main__":
    unittest.main()
