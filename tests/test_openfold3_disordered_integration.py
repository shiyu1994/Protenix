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

"""Focused tests for OpenFold3 disordered-data training integration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from protenix.data.constants import PROTEIN_CHAIN
from protenix.data.msa.msa_featurizer import MSAFeaturizer
from protenix.data.msa.msa_utils import MSA_GAP_IDX, MSACore, RawMsa


class _FakeTokenArray:
    def __init__(self, centre_atom_indices):
        self.centre_atom_indices = np.asarray(
            centre_atom_indices,
            dtype=np.int64,
        )

    def __getitem__(self, indices):
        return _FakeTokenArray(self.centre_atom_indices[indices])

    def get_annotation(self, name):
        if name != "centre_atom_index":
            raise KeyError(name)
        return self.centre_atom_indices


class _FakeAtomArray:
    def __init__(
        self,
        *,
        asym_id_int,
        chain_id,
        res_id,
        label_asym_id,
        mol_type=None,
    ):
        self.asym_id_int = np.asarray(asym_id_int, dtype=np.int64)
        self.chain_id = np.asarray(chain_id)
        self.res_id = np.asarray(res_id, dtype=np.int64)
        self.label_asym_id = np.asarray(label_asym_id)
        if mol_type is not None:
            self.mol_type = np.asarray(mol_type)

    def __getitem__(self, indices):
        mol_type = (
            self.mol_type[indices]
            if hasattr(self, "mol_type")
            else None
        )
        return _FakeAtomArray(
            asym_id_int=self.asym_id_int[indices],
            chain_id=self.chain_id[indices],
            res_id=self.res_id[indices],
            label_asym_id=self.label_asym_id[indices],
            mol_type=mol_type,
        )


def _payload(
    query: str,
    hit: str,
    source_name: str,
    deletion_value: int = 0,
) -> np.ndarray:
    value = np.empty((), dtype=object)
    value[()] = {
        "msa": np.asarray(
            [list(query), list(hit)],
            dtype="<U1",
        ),
        "deletion_matrix": np.asarray(
            [
                [0] * len(query),
                [deletion_value] * len(query),
            ],
            dtype=np.int32,
        ),
        "metadata": np.asarray(
            ["query", f"{source_name}_hit"],
            dtype=object,
        ),
    }
    return value


def _write_alignment(
    path: Path,
    *,
    query: str,
    source_to_hit: dict[str, str],
) -> None:
    np.savez(
        path,
        **{
            source_name: _payload(
                query,
                hit,
                source_name,
                deletion_value=index,
            )
            for index, (source_name, hit) in enumerate(
                source_to_hit.items(),
                start=1,
            )
        },
    )


def _msa_featurizer(alignment_dir: Path) -> MSAFeaturizer:
    return MSAFeaturizer(
        dataset_name="openfold3-disordered-test",
        prot_seq_or_filename_to_msadir_jsons=[],
        prot_msadir_raw_paths=[],
        rna_seq_or_filename_to_msadir_jsons=[],
        rna_msadir_raw_paths=[],
        prot_pairing_dbs=[],
        prot_non_pairing_dbs=[],
        prot_indexing_methods=[],
        rna_indexing_methods=[],
        enable_prot_msa=True,
        enable_rna_msa=False,
        max_msa_size=32,
        openfold3_alignment_npz_dirs=[str(alignment_dir)],
    )


def _single_chain_bioassembly(
    sequence: str,
    *,
    pdb_id: str = "target",
    chain_id: str = "AuthA",
    label_asym_id: str = "TargetLabel",
    representative_id: str | None = None,
) -> tuple[dict, dict[str, list[int]]]:
    atom_array = _FakeAtomArray(
        asym_id_int=[1] * len(sequence),
        chain_id=[chain_id] * len(sequence),
        res_id=range(1, len(sequence) + 1),
        label_asym_id=[label_asym_id] * len(sequence),
    )
    bioassembly = {
        "pdb_id": pdb_id,
        "atom_array": atom_array,
        "token_array": _FakeTokenArray(np.arange(len(sequence))),
        "sequences": {"1": sequence},
        "entity_poly_type": {"1": "polypeptide(L)"},
    }
    if representative_id is not None:
        bioassembly["alignment_representative_id_by_chain_id"] = {
            chain_id: representative_id
        }
    return bioassembly, {"1": [1]}


class TestOpenFold3AlignmentAdapter(unittest.TestCase):
    def test_aligned_arrays_preserve_deletions_and_require_exact_query(self):
        msa = np.asarray(
            [list("ACDEF"), list("A-DEF")],
            dtype="<U1",
        )
        deletion_matrix = np.asarray(
            [
                [0, 0, 0, 0, 0],
                [3, 1, 4, 1, 5],
            ],
            dtype=np.int16,
        )

        raw_msa = RawMsa.from_aligned_arrays(
            "ACDEF",
            PROTEIN_CHAIN,
            msa,
            deletion_matrix,
            ["query", "hit"],
            dedup=False,
        )

        np.testing.assert_array_equal(
            raw_msa.featurize()["deletion_matrix"],
            deletion_matrix,
        )
        with self.assertRaisesRegex(
            ValueError,
            "query row does not match",
        ):
            RawMsa.from_aligned_arrays(
                "ACDDF",
                PROTEIN_CHAIN,
                msa,
                deletion_matrix,
                ["query", "hit"],
            )

    def test_uniprot_is_paired_and_other_protein_sources_are_unpaired(self):
        query = "ACDEF"
        with tempfile.TemporaryDirectory() as tmp:
            alignment_dir = Path(tmp)
            path = alignment_dir / "1abc_A.npz"
            _write_alignment(
                path,
                query=query,
                source_to_hit={
                    "uniref90_hits": "ACDDF",
                    "cfdb_hits": "ACEEF",
                    "mgnify_hits": "ACGEF",
                    "uniprot_hits": "ACKEF",
                },
            )
            featurizer = _msa_featurizer(alignment_dir)

            unpaired, paired = featurizer._load_openfold3_chain_msas(
                pdb_id="1abc",
                label_asym_id="A",
                query=query,
                chain_type=PROTEIN_CHAIN,
            )

        self.assertEqual(
            [msa.descs[1] for msa in unpaired],
            [
                "uniref90_hits_hit",
                "cfdb_hits_hit",
                "mgnify_hits_hit",
            ],
        )
        self.assertEqual(
            [msa.descs[1] for msa in paired],
            ["uniprot_hits_hit"],
        )
        np.testing.assert_array_equal(
            unpaired[1].featurize()["deletion_matrix"][1],
            np.full(len(query), 2, dtype=np.int32),
        )

    def test_representative_mapping_preserves_label_asym_id_case(self):
        query = "ACDEF"
        lower_hit = "ACDDF"
        upper_hit = "ACEEF"
        with tempfile.TemporaryDirectory() as tmp:
            alignment_dir = Path(tmp)
            lower_path = alignment_dir / "3os0_t.npz"
            upper_path = alignment_dir / "3os0_T.npz"
            _write_alignment(
                lower_path,
                query=query,
                source_to_hit={"uniref90_hits": lower_hit},
            )
            _write_alignment(
                upper_path,
                query=query,
                source_to_hit={"uniref90_hits": upper_hit},
            )
            featurizer = _msa_featurizer(alignment_dir)
            bioassembly, entity_to_asym = _single_chain_bioassembly(
                query,
                representative_id="3os0_t",
            )

            self.assertEqual(
                featurizer._resolve_openfold3_alignment_path("3OS0", "t"),
                lower_path,
            )
            self.assertEqual(
                featurizer._resolve_openfold3_alignment_path("3OS0", "T"),
                upper_path,
            )
            features = featurizer.make_msa_features(
                bioassembly_dict=bioassembly,
                selected_indices=None,
                entity_to_asym_id_int=entity_to_asym,
            )

        lower_encoded, _ = MSACore.sequences_to_array(
            [lower_hit],
            PROTEIN_CHAIN,
        )
        upper_encoded, _ = MSACore.sequences_to_array(
            [upper_hit],
            PROTEIN_CHAIN,
        )
        self.assertTrue(
            any(np.array_equal(row, lower_encoded[0]) for row in features["msa"])
        )
        self.assertFalse(
            any(np.array_equal(row, upper_encoded[0]) for row in features["msa"])
        )

    def test_openfold3_rows_use_independent_crop_segment_permutations(self):
        query = "A" * 42
        with tempfile.TemporaryDirectory() as tmp:
            alignment_dir = Path(tmp)
            _write_alignment(
                alignment_dir / "target_TargetLabel.npz",
                query=query,
                source_to_hit={
                    "uniref90_hits": "C" * 42,
                    "cfdb_hits": "D" * 42,
                    "mgnify_hits": "E" * 42,
                    "uniprot_hits": "F" * 42,
                },
            )
            featurizer = _msa_featurizer(alignment_dir)
            featurizer.enable_virtual_segment_msa_shuffle = True
            bioassembly, entity_to_asym = _single_chain_bioassembly(query)
            selected_indices = np.concatenate(
                (np.arange(9), np.arange(32, 41))
            )

            features = featurizer.make_msa_features(
                bioassembly_dict=bioassembly,
                selected_indices=selected_indices,
                entity_to_asym_id_int=entity_to_asym,
                pair_msa_as_unpaired_only=True,
                shuffle_pair_msa_when_unpaired_only=True,
                virtual_segment_shuffle_seed=6,
            )

        profile = featurizer.get_last_profile()
        self.assertEqual(profile["virtual_segment_count"], 2)
        self.assertTrue(profile["virtual_segment_shuffle_applied"])
        np.testing.assert_array_equal(
            features["msa"][0, :9],
            features["msa"][0, 9:],
        )
        self.assertTrue(
            np.any(features["msa"][2:, 0] != features["msa"][2:, 9])
        )

    def test_per_chain_sequence_selects_exact_remapped_representative(self):
        """Near-identical swapped copies use their own observed query."""
        with tempfile.TemporaryDirectory() as tmp:
            alignment_dir = Path(tmp)
            _write_alignment(
                alignment_dir / "entry_A.npz",
                query="AAA",
                source_to_hit={"uniref90_hits": "A-A"},
            )
            _write_alignment(
                alignment_dir / "entry_B.npz",
                query="CCC",
                source_to_hit={"uniref90_hits": "C-C"},
            )
            featurizer = _msa_featurizer(alignment_dir)
            atom_array = _FakeAtomArray(
                asym_id_int=[1, 1, 1, 2, 2, 2],
                chain_id=["A", "A", "A", "B", "B", "B"],
                res_id=[1, 2, 3, 1, 2, 3],
                label_asym_id=["A", "A", "A", "B", "B", "B"],
            )
            bioassembly = {
                "pdb_id": "entry",
                "atom_array": atom_array,
                "token_array": _FakeTokenArray(np.arange(6)),
                "sequences": {"1": "AAA"},
                "sequences_by_chain_id": {"A": "AAA", "B": "CCC"},
                "entity_poly_type": {"1": "polypeptide(L)"},
                "alignment_representative_id_by_chain_id": {
                    "A": "entry_A",
                    "B": "entry_B",
                },
            }

            features = featurizer.make_msa_features(
                bioassembly_dict=bioassembly,
                selected_indices=None,
                entity_to_asym_id_int={"1": [1, 2]},
            )

        encoded_query, _ = MSACore.sequences_to_array(
            ["AAACCC"],
            PROTEIN_CHAIN,
        )
        np.testing.assert_array_equal(features["msa"][0], encoded_query[0])

    def test_partial_query_only_skips_selected_openfold3_archive(self):
        queries = {"A": "ACDEF", "B": "GHIKL"}
        hits = {"A": "CCCCC", "B": "MMMMM"}
        with tempfile.TemporaryDirectory() as tmp:
            alignment_dir = Path(tmp)
            for label_asym_id in ("A", "B"):
                _write_alignment(
                    alignment_dir
                    / f"partial_{label_asym_id}.npz",
                    query=queries[label_asym_id],
                    source_to_hit={
                        "uniref90_hits": hits[label_asym_id],
                    },
                )
            featurizer = _msa_featurizer(alignment_dir)
            featurizer.enable_crop_aware_partial_query_only = True
            featurizer.partial_query_only_max_chains = 2
            atom_array = _FakeAtomArray(
                asym_id_int=[1] * 5 + [2] * 5,
                chain_id=["AuthA"] * 5 + ["AuthB"] * 5,
                res_id=list(range(1, 6)) * 2,
                label_asym_id=["A"] * 5 + ["B"] * 5,
            )
            bioassembly = {
                "pdb_id": "partial",
                "atom_array": atom_array,
                "token_array": _FakeTokenArray(np.arange(10)),
                "sequences": {"1": queries["A"], "2": queries["B"]},
                "entity_poly_type": {
                    "1": "polypeptide(L)",
                    "2": "polypeptide(L)",
                },
            }
            with patch.object(
                featurizer,
                "_load_openfold3_chain_msas",
                wraps=featurizer._load_openfold3_chain_msas,
            ) as load_alignment:
                features = featurizer.make_msa_features(
                    bioassembly_dict=bioassembly,
                    selected_indices=None,
                    entity_to_asym_id_int={"1": [1], "2": [2]},
                    query_only=True,
                    query_only_chain_selection_seed=17,
                )

        profile = featurizer.get_last_profile()
        selected_asym = profile[
            "query_only_selected_protein_asym_ids"
        ][0]
        retained_asym = 2 if selected_asym == 1 else 1
        selected_slice = (
            slice(0, 5) if selected_asym == 1 else slice(5, 10)
        )
        retained_slice = (
            slice(5, 10) if selected_asym == 1 else slice(0, 5)
        )
        retained_label = "A" if retained_asym == 1 else "B"

        self.assertEqual(load_alignment.call_count, 1)
        self.assertEqual(
            load_alignment.call_args.kwargs["label_asym_id"],
            retained_label,
        )
        np.testing.assert_array_equal(
            features["msa"][1:, selected_slice],
            np.full(
                (features["msa"].shape[0] - 1, 5),
                MSA_GAP_IDX,
            ),
        )
        encoded_hit, _ = MSACore.sequences_to_array(
            [hits[retained_label]],
            PROTEIN_CHAIN,
        )
        self.assertTrue(
            any(
                np.array_equal(row, encoded_hit[0])
                for row in features["msa"][:, retained_slice]
            )
        )

    def test_missing_alignment_archive_falls_back_to_canonical_query_only(self):
        query = "ACDEF"
        with tempfile.TemporaryDirectory() as tmp:
            featurizer = _msa_featurizer(Path(tmp))
            bioassembly, entity_to_asym = _single_chain_bioassembly(query)

            with patch.object(featurizer.prot_mgr, "fetch_msas") as legacy_fetch:
                features = featurizer.make_msa_features(
                    bioassembly_dict=bioassembly,
                    selected_indices=None,
                    entity_to_asym_id_int=entity_to_asym,
                )

        legacy_fetch.assert_not_called()
        encoded_query, _ = MSACore.sequences_to_array(
            [query],
            PROTEIN_CHAIN,
        )
        self.assertTrue(
            all(
                np.array_equal(row, encoded_query[0])
                for row in features["msa"]
            )
        )
        self.assertEqual(int(features["prot_unpair_num_alignments"]), 1)
        self.assertEqual(int(features["prot_pair_num_alignments"]), 1)


if __name__ == "__main__":
    unittest.main()
