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

"""Regress sequence identity and residue-column mapping in peptide caches."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("torch")
struc = pytest.importorskip("biotite.structure")
from biotite.structure.io.pdbx import CIFCategory

from protenix.data.msa.msa_utils import map_to_standard
from protenix.data.pipeline.peptide_sequence import (
    match_full_sequence,
    recover_polymer_sequences,
)

PROTEIN = "polypeptide(L)"


@pytest.mark.parametrize(
    "observed,full",
    [({3: "S", 4: "I"}, "MDSI"), ({1: "A", 4: "E"}, "ACDE")],
)
def test_terminal_and_internal_missing_residues_keep_full_query(observed, full):
    query, entities, method = match_full_sequence(
        observed, "B", PROTEIN, {"B": "9"}, {"9": full}, {"9": PROTEIN}
    )
    assert query == full
    assert entities == ["9"]
    indices = map_to_standard(
        np.ones(len(observed), dtype=int), np.array(list(observed)), {1: {"sequence": query}}
    )
    assert "".join(query[i] for i in indices) == "".join(observed.values())


def test_renamed_chain_requires_unique_sequence_and_original_positions():
    query, _, method = match_full_sequence(
        {2: "C", 4: "E"}, "renamed", PROTEIN, {},
        {"1": "ACDE", "2": "WXYZ"}, {"1": PROTEIN, "2": PROTEIN}
    )
    assert query == "ACDE"
    assert method == "unique_sequence_at_original_residue_ids"


def test_ambiguous_full_sequences_fail_closed():
    with pytest.raises(ValueError, match="2 full sequences"):
        match_full_sequence(
            {2: "C", 4: "E"}, "renamed", PROTEIN, {},
            {"1": "ACDE", "2": "WCDE"}, {"1": PROTEIN, "2": PROTEIN}
        )


@pytest.mark.parametrize("observed", [{0: "A"}, {8: "A"}, {1: "W"}])
def test_invalid_positions_or_residue_identity_rejected(observed):
    with pytest.raises(ValueError):
        match_full_sequence(observed, "A", PROTEIN, {"A": "1"}, {"1": "ACDE"}, {"1": PROTEIN})


def test_recovery_preserves_atoms_ligand_and_renumbered_entity(tmp_path):
    atoms = struc.AtomArray(3)
    atoms.chain_id = np.array(["A", "A", "B"])
    atoms.res_id = np.array([3, 4, 1])
    atoms.res_name = np.array(["SER", "ILE", "X00"])
    atoms.atom_name = np.array(["CA", "CA", "C00"])
    atoms.set_annotation("label_entity_id", np.array(["2", "2", "3"]))
    atoms.set_annotation("label_asym_id", np.array(["A", "A", "B"]))
    atoms.coord[:] = np.arange(9).reshape(3, 3)
    atoms.bonds = struc.BondList(3, np.array([[0, 1, 1]]))
    structure = {"atom_array": atoms, "sequences": {"2": "SI"}, "entity_poly_type": {"2": PROTEIN}}
    original = tmp_path / "original.cif"
    original.write_text("test original CIF content")
    parser = SimpleNamespace(
        get_sequences=lambda: {"9": "MDSI"}, entity_poly_type={"9": PROTEIN},
        cif=SimpleNamespace(block={"struct_asym": CIFCategory({"id": ["A"], "entity_id": ["9"]})}),
    )
    coords, bonds, res_ids = atoms.coord.copy(), atoms.bonds.as_array().copy(), atoms.res_id.copy()
    with patch("protenix.data.pipeline.peptide_sequence.MMCIFParser", return_value=parser):
        audit = recover_polymer_sequences(structure, original)
    assert structure["sequences"] == {"2": "MDSI"}
    assert structure["atom_array"] is atoms
    assert audit["chains"][0]["original_entity_ids"] == ["9"]
    np.testing.assert_array_equal(atoms.coord, coords)
    np.testing.assert_array_equal(atoms.bonds.as_array(), bonds)
    np.testing.assert_array_equal(atoms.res_id, res_ids)
    assert atoms.res_name[-1] == "X00"
