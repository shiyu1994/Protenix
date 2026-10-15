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

import csv
import gzip
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
from biotite.structure import AtomArray

from protenix.data.core import openfold3_disordered
from scripts import build_openfold3_disordered_dataset as builder


def _protein_chain(
    *,
    label_asym_id: str,
    entity_id: int,
    representative: str,
    cluster_id: str,
    cluster_size: int,
) -> dict:
    return {
        "label_asym_id": label_asym_id,
        "auth_asym_id": label_asym_id,
        "entity_id": entity_id,
        "molecule_type": "PROTEIN",
        "reference_mol_id": None,
        "alignment_representative_id": representative,
        "template_ids": [],
        "cluster_id": cluster_id,
        "cluster_size": cluster_size,
    }


def _write_alignment_query(path: Path, query: str) -> None:
    payload = np.empty((), dtype=object)
    payload[()] = {
        "msa": np.asarray([list(query)], dtype="<U1"),
    }
    np.savez(path, uniref90_hits=payload)


class OpenFold3DisorderedDatasetBuilderTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.base = Path(self.temporary_directory.name)
        self.structures_root = self.base / "structures" / "structures"
        self.alignment_dir = self.base / "alignment_arrays"
        self.structures_root.mkdir(parents=True)
        self.alignment_dir.mkdir()

        self.cache = {
            "_type": "ClusteredDatasetCache",
            "name": "test-disordered",
            "structure_data": {
                "1aaa": {
                    "release_date": "2020-01-01",
                    "resolution": 2.0,
                    "chains": {
                        "1": _protein_chain(
                            label_asym_id="A",
                            entity_id=1,
                            representative="1aaa_A",
                            cluster_id="protein-cluster",
                            cluster_size=10,
                        ),
                        "2": _protein_chain(
                            label_asym_id="B",
                            entity_id=1,
                            representative="1aaa_A",
                            cluster_id="protein-cluster",
                            cluster_size=10,
                        ),
                    },
                    "interfaces": {
                        "1_2": {
                            "cluster_id": "interface-cluster",
                            "cluster_size": 5,
                        }
                    },
                },
                "2bbb": {
                    "release_date": "2021-01-01",
                    "resolution": 3.0,
                    "chains": {
                        "1": {
                            "label_asym_id": "A",
                            "auth_asym_id": "A",
                            "entity_id": 2,
                            "molecule_type": "LIGAND",
                            "reference_mol_id": "ATP",
                            "alignment_representative_id": None,
                            "template_ids": None,
                            "cluster_id": "ligand-cluster",
                            "cluster_size": 2,
                        }
                    },
                    "interfaces": {},
                },
            },
            "reference_molecule_data": {},
        }
        self.cache_path = self.base / "cache.json"
        self.cache_path.write_text(
            json.dumps(self.cache),
            encoding="utf-8",
        )
        for pdb_id in self.cache["structure_data"]:
            source_dir = self.structures_root / pdb_id
            source_dir.mkdir()
            (source_dir / f"{pdb_id}.npz").write_bytes(
                f"source-{pdb_id}".encode()
            )
        (self.alignment_dir / "1aaa_A.npz").write_bytes(b"alignment")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _fake_core(
        self,
        source_path,
        *,
        pdb_id,
        cache_entry,
        alignment_dirs=(),
    ):
        del source_path, cache_entry, alignment_dirs
        if pdb_id == "1aaa":
            rows = [
                {
                    "entity_1_id": "1",
                    "chain_1_id": "1",
                    "mol_1_type": "prot",
                    "cluster_1_id": "protein-cluster",
                    "entity_2_id": "",
                    "chain_2_id": "",
                    "mol_2_type": "",
                    "cluster_2_id": "",
                    "cluster_id": "protein-cluster",
                    "pdb_id": pdb_id,
                    "num_tokens": 2,
                    "type": "chain",
                },
                {
                    "entity_1_id": "1",
                    "chain_1_id": "2",
                    "mol_1_type": "prot",
                    "cluster_1_id": "protein-cluster",
                    "entity_2_id": "",
                    "chain_2_id": "",
                    "mol_2_type": "",
                    "cluster_2_id": "",
                    "cluster_id": "protein-cluster",
                    "pdb_id": pdb_id,
                    "num_tokens": 2,
                    "type": "chain",
                },
                {
                    "entity_1_id": "1",
                    "chain_1_id": "1",
                    "mol_1_type": "prot",
                    "cluster_1_id": "protein-cluster",
                    "entity_2_id": "1",
                    "chain_2_id": "2",
                    "mol_2_type": "prot",
                    "cluster_2_id": "protein-cluster",
                    "cluster_id": "interface-cluster",
                    "pdb_id": pdb_id,
                    "num_tokens": 2,
                    "type": "interface",
                },
            ]
            representatives = {"1": "1aaa_A", "2": "1aaa_A"}
            token_count = 2
            chain_ids = ["1", "2"]
        else:
            rows = [
                {
                    "entity_1_id": "2",
                    "chain_1_id": "1",
                    "mol_1_type": "ligand",
                    "cluster_1_id": "ligand-cluster",
                    "entity_2_id": "",
                    "chain_2_id": "",
                    "mol_2_type": "",
                    "cluster_2_id": "",
                    "cluster_id": "ligand-cluster",
                    "pdb_id": pdb_id,
                    "num_tokens": 1,
                    "type": "chain",
                }
            ]
            representatives = {}
            token_count = 1
            chain_ids = ["1"]
        atom_array = AtomArray(token_count)
        atom_array.chain_id = chain_ids
        bioassembly = {
            "pdb_id": pdb_id,
            "atom_array": atom_array,
            "token_array": list(range(token_count)),
            "num_tokens": token_count,
            "alignment_representative_id_by_chain_id": representatives,
        }
        return rows, bioassembly

    def _build(self, **overrides):
        arguments = {
            "cache_json": self.cache_path,
            "structures_root": self.structures_root,
            "alignment_dirs": [self.alignment_dir, self.alignment_dir],
            "workers": 1,
            "expected_entry_count": 2,
            "compresslevel": 1,
        }
        arguments.update(overrides)
        with mock.patch.object(
            builder,
            "_build_with_core",
            side_effect=self._fake_core,
        ):
            return builder.build_dataset(**arguments)

    def test_candidate_names_preserve_chain_case(self):
        self.assertEqual(
            builder._alignment_candidate_names("AbCd_t"),
            ("AbCd_t.npz", "abcd_t.npz", "ABCD_t.npz"),
        )

    def test_core_preloads_queries_for_distinct_entity_swap(self):
        cache_entry = {
            "chains": {
                "1": _protein_chain(
                    label_asym_id="A",
                    entity_id=1,
                    representative="3swap_A",
                    cluster_id="cluster-a",
                    cluster_size=1,
                ),
                "2": _protein_chain(
                    label_asym_id="B",
                    entity_id=2,
                    representative="3swap_B",
                    cluster_id="cluster-b",
                    cluster_size=1,
                ),
            },
            "interfaces": {},
        }
        _write_alignment_query(
            self.alignment_dir / "3swap_A.npz",
            "AAAA",
        )
        _write_alignment_query(
            self.alignment_dir / "3swap_B.npz",
            "CCCC",
        )
        received_queries = []

        def fake_core(
            source_path,
            *,
            pdb_id,
            cache_entry,
            canonical_sequences_by_representative=None,
            ccd_mols,
        ):
            del source_path, pdb_id, cache_entry, ccd_mols
            received_queries.append(canonical_sequences_by_representative)
            if canonical_sequences_by_representative is None:
                representatives = {"1": "3swap_A", "2": "3swap_B"}
            else:
                # The structure has distinct entities whose observed
                # sequences are swapped relative to the cache mapping.  This
                # does not raise the old inconsistent-copy exception.
                representatives = {"1": "3swap_B", "2": "3swap_A"}
            return [], {
                "alignment_representative_id_by_chain_id": representatives,
            }

        with mock.patch(
            "protenix.data.core.openfold3_disordered."
            "build_openfold3_disordered_data",
            side_effect=fake_core,
        ):
            _, bioassembly = builder._build_with_core(
                self.structures_root / "3swap" / "3swap.npz",
                pdb_id="3swap",
                cache_entry=cache_entry,
                alignment_dirs=(self.alignment_dir,),
            )

        self.assertEqual(
            received_queries,
            [{"3swap_A": "AAAA", "3swap_B": "CCCC"}],
        )
        self.assertEqual(
            bioassembly["alignment_representative_id_by_chain_id"],
            {"1": "3swap_B", "2": "3swap_A"},
        )

    def test_distinct_entity_swap_keeps_physical_entity_sequences(self):
        cache_chains = {
            "1": _protein_chain(
                label_asym_id="A",
                entity_id=1,
                representative="3swap_A",
                cluster_id="cluster-a",
                cluster_size=1,
            ),
            "2": _protein_chain(
                label_asym_id="B",
                entity_id=2,
                representative="3swap_B",
                cluster_id="cluster-b",
                cluster_size=1,
            ),
        }

        representatives, entity_sequences, query_only_chain_ids = (
            openfold3_disordered._resolve_alignment_representatives(
                cache_chains=cache_chains,
                sequences_by_chain_id={
                    "1": "CCCC",
                    "2": "AAAA",
                },
                canonical_sequences_by_representative={
                    "3swap_A": "AAAA",
                    "3swap_B": "CCCC",
                },
            )
        )

        self.assertEqual(
            representatives,
            {"1": "3swap_B", "2": "3swap_A"},
        )
        self.assertEqual(
            entity_sequences,
            {
                "1": "CCCC",
                "2": "AAAA",
            },
        )
        self.assertEqual(query_only_chain_ids, [])

    def test_zero_or_nonunique_match_is_explicitly_query_only(self):
        cache_chains = {
            "1": _protein_chain(
                label_asym_id="A",
                entity_id=1,
                representative="4query_A",
                cluster_id="cluster-a",
                cluster_size=1,
            ),
            "2": _protein_chain(
                label_asym_id="B",
                entity_id=2,
                representative="4query_B",
                cluster_id="cluster-b",
                cluster_size=1,
            ),
            "3": _protein_chain(
                label_asym_id="C",
                entity_id=3,
                representative="4query_C",
                cluster_id="cluster-c",
                cluster_size=1,
            ),
        }

        representatives, entity_sequences, query_only_chain_ids = (
            openfold3_disordered._resolve_alignment_representatives(
                cache_chains=cache_chains,
                sequences_by_chain_id={
                    "1": "ZZZZ",
                    "2": "BBBB",
                    "3": "BBBB",
                },
                canonical_sequences_by_representative={
                    "4query_A": "AAAA",
                    "4query_B": "BBBB",
                    "4query_C": "BBBB",
                },
            )
        )

        self.assertEqual(
            representatives,
            {
                "1": "4query_A",
                "2": "4query_B",
                "3": "4query_C",
            },
        )
        self.assertEqual(entity_sequences, {})
        self.assertEqual(query_only_chain_ids, ["1", "2", "3"])

    def test_core_without_alignment_dirs_keeps_legacy_call_signature(self):
        def legacy_core(
            source_path,
            *,
            pdb_id,
            cache_entry,
            ccd_mols,
        ):
            del source_path, pdb_id, cache_entry, ccd_mols
            return [], {}

        with mock.patch(
            "protenix.data.core.openfold3_disordered."
            "build_openfold3_disordered_data",
            side_effect=legacy_core,
        ):
            self.assertEqual(
                builder._build_with_core(
                    self.structures_root / "1aaa" / "1aaa.npz",
                    pdb_id="1aaa",
                    cache_entry=self.cache["structure_data"]["1aaa"],
                    alignment_dirs=(),
                ),
                ([], {}),
            )

    def test_official_weights_include_copy_normalization(self):
        rows, _ = self._fake_core(
            Path("1aaa.npz"),
            pdb_id="1aaa",
            cache_entry=self.cache["structure_data"]["1aaa"],
        )

        weighted = builder._weight_index_rows(
            pdb_id="1aaa",
            raw_rows=rows,
            cache_entry=self.cache["structure_data"]["1aaa"],
            bioassembly_relpath="1aaa/bioassembly/1aaa.pkl.gz",
        )

        chain_weights = [
            row["weights"] for row in weighted if row["type"] == "chain"
        ]
        interface_weights = [
            row["weights"] for row in weighted if row["type"] == "interface"
        ]
        self.assertEqual(chain_weights, [0.075, 0.075])
        self.assertEqual(interface_weights, [1.2])

    def test_build_resume_and_deterministic_aggregates(self):
        first = self._build()

        output_dir = self.base / "protenix"
        success_path = output_dir / "_SUCCESS.json"
        index_path = output_dir / "indices.csv.gz"
        alignment_path = output_dir / "alignment_representatives.tsv.gz"
        self.assertEqual(first["status"], "complete")
        self.assertTrue(success_path.is_file())
        self.assertEqual(first["counts"]["validated_pickles"], 2)
        self.assertEqual(first["counts"]["index_rows"], 4)
        first_index_bytes = index_path.read_bytes()
        first_alignment_bytes = alignment_path.read_bytes()

        with gzip.open(index_path, "rt", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["pdb_id"] for row in rows], [
            "1aaa",
            "1aaa",
            "1aaa",
            "2bbb",
        ])
        self.assertTrue(
            all(
                row["bioassembly_relpath"].startswith(row["pdb_id"] + "/")
                for row in rows
            )
        )
        with gzip.open(
            alignment_path,
            "rt",
            encoding="utf-8",
            newline="",
        ) as stream:
            alignment_rows = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(len(alignment_rows), 1)
        self.assertEqual(alignment_rows[0]["representative_id"], "1aaa_A")
        self.assertEqual(alignment_rows[0]["consumer_structure_count"], "1")
        self.assertEqual(alignment_rows[0]["reference_count"], "2")

        with mock.patch.object(
            builder,
            "_build_with_core",
            side_effect=AssertionError("resume unexpectedly rebuilt"),
        ):
            resumed = builder.build_dataset(
                cache_json=self.cache_path,
                structures_root=self.structures_root,
                alignment_dirs=[self.alignment_dir],
                workers=1,
                resume=True,
                expected_entry_count=2,
                compresslevel=1,
            )
        self.assertEqual(resumed["counts"]["resumed_entries"], 2)
        self.assertEqual(index_path.read_bytes(), first_index_bytes)
        self.assertEqual(alignment_path.read_bytes(), first_alignment_bytes)

    def test_missing_alignment_is_recorded_without_success_marker(self):
        (self.alignment_dir / "1aaa_A.npz").unlink()

        summary = self._build(allow_missing_alignments=True)

        output_dir = self.base / "protenix"
        self.assertTrue(summary["exit_ok"])
        self.assertIsNone(summary["success_marker"])
        self.assertFalse((output_dir / "_SUCCESS.json").exists())
        with (output_dir / "failures.tsv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as stream:
            failures = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(
            failures[0]["error_type"],
            "MissingAlignmentRepresentative",
        )

    def test_cache_entry_count_mismatch_fails_closed(self):
        with self.assertRaisesRegex(
            builder.DatasetBuildError,
            "expected 3",
        ):
            builder._load_cache(
                self.cache_path,
                expected_entry_count=3,
            )


if __name__ == "__main__":
    unittest.main()
