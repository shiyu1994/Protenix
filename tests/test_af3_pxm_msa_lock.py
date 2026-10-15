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
"""Focused tests for the complete PXM evaluation MSA lock."""

from __future__ import annotations

import gzip
import json
import pickle
import tempfile
import unittest
from pathlib import Path

from scripts.evaluation.build_af3_pxm_msa_lock import (
    MSA_LOCK_MANIFEST_FILENAME,
    MSA_LOCK_SUCCESS_FILENAME,
    build_msa_lock,
)
from scripts.evaluation.validate_af3_pxm_msa_lock import validate_msa_lock


class Af3PxmMsaLockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.bioassembly_dir = self.root / "bioassembly"
        self.protein_root = self.root / "protein"
        self.rna_root = self.root / "rna"
        self.output_root = self.root / "lock"
        self.bioassembly_dir.mkdir()
        self.protein_root.mkdir()
        self.rna_root.mkdir()

        self.protein_a = "ACDEFGHIKLMN"
        self.protein_b = "QRSTVWYACDEF"
        self.rna_a = "AUGCUAGCUA"
        self.rna_b = "GGCAUACCGU"
        self.dna = "ATCGATCG"
        self.hybrid = "AUGCATGC"

        self.pdb_list = self.root / "pdb_ids.txt"
        self.pdb_list.write_text("1abc\n2def\n", encoding="utf-8")
        self._write_bioassembly(
            "1abc",
            sequences={
                "1": self.protein_a,
                "2": self.rna_a,
                "3": self.dna,
                "4": self.hybrid,
            },
            polymer_types={
                "1": "polypeptide(L)",
                "2": "polyribonucleotide",
                "3": "polydeoxyribonucleotide",
                "4": "polydeoxyribonucleotide/polyribonucleotide hybrid",
            },
        )
        self._write_bioassembly(
            "2def",
            sequences={
                "1": self.protein_a,
                "2": self.protein_b,
                "3": self.rna_a,
                "4": self.rna_b,
            },
            polymer_types={
                "1": "polypeptide(L)",
                "2": "polypeptide(L)",
                "3": "polyribonucleotide",
                "4": "polyribonucleotide",
            },
        )

        self.protein_sources = self.root / "protein_sources.json"
        protein_mapping = {}
        for index, sequence in enumerate(
            (self.protein_a, self.protein_b),
            start=1,
        ):
            source_directory = self.protein_root / f"source_{index}"
            source_directory.mkdir()
            self._write_a3m(
                source_directory / "pairing.a3m",
                sequence,
                hit_sequence=sequence[:-1] + "-",
            )
            self._write_a3m(
                source_directory / "non_pairing.a3m",
                sequence,
                hit_sequence=sequence,
            )
            protein_mapping[sequence] = str(source_directory)
        self.protein_sources.write_text(
            json.dumps(protein_mapping),
            encoding="utf-8",
        )

        self.rna_mapping = self.root / "rna_mapping.json"
        rna_mapping = {}
        for index, sequence in enumerate((self.rna_a, self.rna_b), start=1):
            eid = f"rna_{index}"
            rna_mapping[sequence] = [eid, f"unused_{index}"]
            rna_directory = self.rna_root / eid
            rna_directory.mkdir()
            self._write_a3m(
                rna_directory / f"{eid}_all.a3m",
                sequence,
                hit_sequence=sequence,
            )
        self.rna_mapping.write_text(
            json.dumps(rna_mapping),
            encoding="utf-8",
        )

        self.manifest_path = (
            self.output_root / MSA_LOCK_MANIFEST_FILENAME
        )
        self.success_path = self.output_root / MSA_LOCK_SUCCESS_FILENAME

    def _write_bioassembly(
        self,
        pdb_id: str,
        *,
        sequences: dict[str, str],
        polymer_types: dict[str, str],
    ) -> None:
        with gzip.open(
            self.bioassembly_dir / f"{pdb_id}.pkl.gz",
            "wb",
        ) as handle:
            pickle.dump(
                {
                    "sequences": sequences,
                    "entity_poly_type": polymer_types,
                },
                handle,
            )

    @staticmethod
    def _write_a3m(
        path: Path,
        query_sequence: str,
        *,
        hit_sequence: str,
    ) -> None:
        path.write_text(
            f">query\n{query_sequence}\n>hit\n{hit_sequence}\n",
            encoding="utf-8",
        )

    def _build_kwargs(self) -> dict[str, Path]:
        return {
            "pdb_list": self.pdb_list,
            "bioassembly_dir": self.bioassembly_dir,
            "protein_sequence_sources_path": self.protein_sources,
            "rna_sequence_mapping_path": self.rna_mapping,
            "rna_msa_root": self.rna_root,
            "manifest_path": self.manifest_path,
            "success_path": self.success_path,
        }

    def test_builds_exact_deduplicated_protein_and_rna_lock(self) -> None:
        summary = build_msa_lock(**self._build_kwargs())
        self.assertEqual(summary["status"], "complete")
        self.assertTrue(self.manifest_path.is_file())
        self.assertTrue(self.success_path.is_file())

        manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["counts"]["target_count"], 2)
        self.assertEqual(manifest["target_ids"], ["1abc", "2def"])
        self.assertEqual(
            manifest["counts"]["protein_entity_occurrence_count"],
            3,
        )
        self.assertEqual(
            manifest["counts"]["unique_protein_sequence_count"],
            2,
        )
        self.assertEqual(
            manifest["counts"]["rna_entity_occurrence_count"],
            3,
        )
        self.assertEqual(
            manifest["counts"]["unique_rna_sequence_count"],
            2,
        )
        self.assertEqual(manifest["counts"]["file_count"], 6)

        records = manifest["records"]
        self.assertEqual(len(records), 4)
        self.assertNotIn(self.dna, {record["sequence"] for record in records})
        self.assertNotIn(
            self.hybrid,
            {record["sequence"] for record in records},
        )
        protein_a_record = next(
            record
            for record in records
            if record["sequence"] == self.protein_a
        )
        self.assertEqual(protein_a_record["occurrence_count"], 2)
        self.assertEqual(
            set(protein_a_record["files"]),
            {"pairing", "non_pairing"},
        )
        for record in records:
            for file_record in record["files"].values():
                self.assertTrue(Path(file_record["path"]).is_absolute())
                self.assertGreater(file_record["size_bytes"], 0)
                self.assertEqual(len(file_record["sha256"]), 64)
                self.assertEqual(file_record["record_depth"], 2)
                self.assertTrue(file_record["first_query_exact"])

        validation = validate_msa_lock(**self._build_kwargs())
        self.assertEqual(validation["status"], "valid")
        self.assertEqual(validation["unique_protein_sequence_count"], 2)
        self.assertEqual(validation["unique_rna_sequence_count"], 2)

        subset_path = self.root / "smoke_targets.txt"
        subset_path.write_text("2def\n", encoding="utf-8")
        subset_validation = validate_msa_lock(
            **self._build_kwargs(),
            requested_pdb_list=subset_path,
        )
        self.assertEqual(subset_validation["target_count"], 2)
        self.assertEqual(subset_validation["requested_target_count"], 1)

    def test_validator_rehashes_every_file_and_rejects_mutation(self) -> None:
        build_msa_lock(**self._build_kwargs())
        mapping = json.loads(self.protein_sources.read_text(encoding="utf-8"))
        mutated_path = Path(mapping[self.protein_a]) / "pairing.a3m"
        mutated_path.write_text(
            f">query\n{self.protein_a}\n>changed\n{self.protein_a}\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            ValueError,
            "manifest differs",
        ):
            validate_msa_lock(**self._build_kwargs())

    def test_validator_rejects_rna_first_eid_file_mutation(self) -> None:
        build_msa_lock(**self._build_kwargs())
        mapping = json.loads(self.rna_mapping.read_text(encoding="utf-8"))
        first_eid = mapping[self.rna_a][0]
        mutated_path = (
            self.rna_root / first_eid / f"{first_eid}_all.a3m"
        )
        mutated_path.write_text(
            f">query\n{self.rna_a}\n>changed\n{self.rna_a}\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            validate_msa_lock(**self._build_kwargs())

    def test_query_mismatch_fails_without_publishing_success(self) -> None:
        self.success_path.parent.mkdir(parents=True)
        self.success_path.write_text('{"stale": true}\n', encoding="utf-8")
        mapping = json.loads(self.protein_sources.read_text(encoding="utf-8"))
        bad_path = Path(mapping[self.protein_b]) / "non_pairing.a3m"
        self._write_a3m(
            bad_path,
            "AAAAAAAAAAAA",
            hit_sequence="AAAAAAAAAAAA",
        )

        with self.assertRaisesRegex(ValueError, "first query"):
            build_msa_lock(**self._build_kwargs())
        self.assertFalse(self.success_path.exists())

    def test_protein_source_coverage_must_be_exact(self) -> None:
        mapping = json.loads(self.protein_sources.read_text(encoding="utf-8"))
        del mapping[self.protein_b]
        self.protein_sources.write_text(
            json.dumps(mapping),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "coverage is not exact"):
            build_msa_lock(**self._build_kwargs())
        self.assertFalse(self.success_path.exists())

    def test_requested_target_must_be_inside_full_lock(self) -> None:
        build_msa_lock(**self._build_kwargs())
        requested_path = self.root / "custom_targets.txt"
        requested_path.write_text("1abc\n9xyz\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "outside the full MSA lock"):
            validate_msa_lock(
                **self._build_kwargs(),
                requested_pdb_list=requested_path,
            )


if __name__ == "__main__":
    unittest.main()
