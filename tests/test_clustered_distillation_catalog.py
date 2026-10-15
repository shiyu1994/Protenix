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
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from scripts.distillation.build_clustered_distillation_catalog import (
    CatalogBuildError,
    SourceRoot,
    _parse_expected_counts,
    build_clustered_distillation_catalog,
    build_source_inventory,
    iter_mmseqs_cluster_blocks,
    parse_source_root,
)


class ClusteredDistillationCatalogTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.afdb = self.root / "afdb"
        self.mgnify = self.root / "mgnify"
        self.afdb.mkdir()
        self.mgnify.mkdir()
        self.source_roots = [
            SourceRoot("afdb", "AFQ|", self.afdb),
            SourceRoot("mgnify", "MGN|", self.mgnify),
        ]

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def _touch_cif(root: Path, relative_path: str) -> None:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data_test\n", encoding="utf-8")

    @staticmethod
    def _read_csv_gz(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as file_obj:
            return list(csv.DictReader(file_obj, delimiter=delimiter))

    @staticmethod
    def _read_binary(path: Path, format_string: str) -> list[int]:
        payload = path.read_bytes()
        item_size = struct.calcsize(format_string)
        return [
            item[0]
            for item in struct.iter_unpack(format_string, payload)
            if len(payload) % item_size == 0
        ]

    def test_build_filters_replaces_and_writes_aligned_sampling_arrays(self):
        self._touch_cif(self.afdb, "leaf/repA.cif")
        self._touch_cif(self.afdb, "leaf/memberA.cif")
        self._touch_cif(self.mgnify, "p001/memberM.cif")
        self._touch_cif(self.mgnify, "p001/replacement.cif")
        self._touch_cif(
            self.afdb,
            "leaf/bioassembly/must_be_ignored.cif",
        )

        cluster_tsv = self.root / "clusters.tsv"
        cluster_tsv.write_text(
            "\n".join(
                [
                    "AFQ|repA\tAFQ|memberA",
                    "AFQ|repA\tMGN|memberM",
                    "AFQ|repA\tAFQ|repA",
                    "AFQ|missingRep\tMGN|replacement",
                    "AFQ|missingRep\tAFQ|missingRep",
                    "AFQ|gone\tAFQ|gone",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        output_dir = self.root / "output"

        manifest = build_clustered_distillation_catalog(
            cluster_tsv=cluster_tsv,
            source_roots=self.source_roots,
            output_dir=output_dir,
            scan_workers=4,
            compresslevel=1,
        )

        self.assertEqual(manifest["counts"]["input_rows"], 6)
        self.assertEqual(manifest["counts"]["input_clusters"], 3)
        self.assertEqual(manifest["counts"]["surviving_rows"], 4)
        self.assertEqual(manifest["counts"]["surviving_clusters"], 2)
        self.assertEqual(manifest["counts"]["removed_members"], 2)
        self.assertEqual(manifest["counts"]["dropped_clusters"], 1)
        self.assertEqual(manifest["counts"]["replaced_representatives"], 1)
        self.assertEqual(manifest["counts"]["cross_source_clusters"], 1)
        self.assertEqual(manifest["counts"]["unused_inventory_entries"], 0)

        index_rows = self._read_csv_gz(
            output_dir / "distillation_index.csv.gz"
        )
        self.assertEqual(
            [row["source_protein_id"] for row in index_rows],
            [
                "AFQ|memberA",
                "MGN|memberM",
                "AFQ|repA",
                "MGN|replacement",
            ],
        )
        self.assertEqual([row["row_id"] for row in index_rows], ["0", "1", "2", "3"])
        self.assertEqual(
            [row["cluster_id"] for row in index_rows],
            ["0", "0", "0", "1"],
        )
        self.assertEqual(index_rows[0]["cluster_1_id"], "0")
        self.assertEqual(index_rows[0]["num_tokens"], "-1")
        self.assertEqual(index_rows[0]["pdb_id"], "memberA")
        self.assertEqual(index_rows[0]["release_date"], "9999-12-31")
        self.assertEqual(index_rows[0]["resolution"], "-1.0")
        self.assertEqual(index_rows[0]["cif_relpath"], "leaf/memberA.cif")
        self.assertFalse(Path(index_rows[0]["cif_relpath"]).is_absolute())
        self.assertEqual(index_rows[0]["source_db"], "afdb")
        self.assertEqual(index_rows[1]["source_db"], "mgnify")

        cluster_rows = self._read_csv_gz(output_dir / "clusters.csv.gz")
        self.assertEqual(cluster_rows[0]["representative_id"], "AFQ|repA")
        self.assertEqual(cluster_rows[0]["representative_row_id"], "2")
        self.assertEqual(cluster_rows[0]["first_row_id"], "0")
        self.assertEqual(cluster_rows[0]["cluster_size"], "3")
        self.assertIn("cross_source", cluster_rows[0]["flags"])
        self.assertEqual(
            cluster_rows[1]["representative_id"],
            "MGN|replacement",
        )
        self.assertEqual(cluster_rows[1]["representative_row_id"], "3")
        self.assertIn("representative_replaced", cluster_rows[1]["flags"])

        with gzip.open(
            output_dir / "cleaned_clusters.tsv.gz",
            "rt",
            encoding="utf-8",
        ) as file_obj:
            cleaned_rows = [line.rstrip("\n") for line in file_obj]
        self.assertEqual(
            cleaned_rows,
            [
                "AFQ|repA\tAFQ|memberA",
                "AFQ|repA\tMGN|memberM",
                "AFQ|repA\tAFQ|repA",
                "MGN|replacement\tMGN|replacement",
            ],
        )

        removed_rows = self._read_csv_gz(
            output_dir / "removed_members.tsv.gz",
            delimiter="\t",
        )
        self.assertEqual(
            [row["source_protein_id"] for row in removed_rows],
            ["AFQ|missingRep", "AFQ|gone"],
        )
        dropped_rows = self._read_csv_gz(
            output_dir / "dropped_clusters.tsv.gz",
            delimiter="\t",
        )
        self.assertEqual(dropped_rows[0]["cluster_id"], "2")

        sampling_dir = output_dir / "sampling"
        self.assertEqual(
            self._read_binary(sampling_dir / "cluster_ids.u32", "<I"),
            [0, 1],
        )
        self.assertEqual(
            self._read_binary(sampling_dir / "first_row_ids.u64", "<Q"),
            [0, 3],
        )
        self.assertEqual(
            self._read_binary(sampling_dir / "cluster_sizes.u32", "<I"),
            [3, 1],
        )
        disk_manifest = json.loads(
            (output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        for array in disk_manifest["sampling"]["arrays"].values():
            payload = (output_dir / array["path"]).read_bytes()
            self.assertEqual(array["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(array["count"], 2)

    def test_inventory_skips_bioassembly_and_rejects_duplicate_ids(self):
        self._touch_cif(self.afdb, "a/sample.cif")
        self._touch_cif(self.afdb, "a/bioassembly/ignored.cif")
        inventory, counts = build_source_inventory(
            self.source_roots,
            scan_workers=3,
        )
        self.assertEqual(set(inventory), {"AFQ|sample"})
        self.assertEqual(counts["afdb"]["cif_files"], 1)

        self._touch_cif(self.afdb, "b/sample.cif")
        with self.assertRaisesRegex(CatalogBuildError, "duplicate"):
            build_source_inventory(self.source_roots, scan_workers=3)

    def test_duplicate_member_is_rejected(self):
        lines = [
            "AFQ|a\tAFQ|a\n",
            "AFQ|b\tAFQ|a\n",
            "AFQ|b\tAFQ|b\n",
        ]
        with self.assertRaisesRegex(CatalogBuildError, "Member.*more than once"):
            list(iter_mmseqs_cluster_blocks(lines))

    def test_non_contiguous_representative_is_rejected(self):
        lines = [
            "AFQ|a\tAFQ|a\n",
            "AFQ|b\tAFQ|b\n",
            "AFQ|a\tAFQ|c\n",
        ]
        with self.assertRaisesRegex(CatalogBuildError, "reappears"):
            list(iter_mmseqs_cluster_blocks(lines))

    def test_missing_representative_self_row_is_rejected(self):
        lines = [
            "AFQ|a\tAFQ|other\n",
            "AFQ|b\tAFQ|b\n",
        ]
        with self.assertRaisesRegex(CatalogBuildError, "exactly one"):
            list(iter_mmseqs_cluster_blocks(lines))

    def test_nonempty_output_is_refused_without_changes(self):
        self._touch_cif(self.afdb, "sample.cif")
        cluster_tsv = self.root / "clusters.tsv"
        cluster_tsv.write_text("AFQ|sample\tAFQ|sample\n", encoding="utf-8")
        output_dir = self.root / "output"
        output_dir.mkdir()
        marker = output_dir / "owned_by_user.txt"
        marker.write_text("keep", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "non-empty"):
            build_clustered_distillation_catalog(
                cluster_tsv=cluster_tsv,
                source_roots=self.source_roots,
                output_dir=output_dir,
                scan_workers=2,
            )
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.assertEqual(list(output_dir.iterdir()), [marker])

    def test_empty_result_and_expected_count_mismatch_are_not_published(self):
        self._touch_cif(self.afdb, "unclustered.cif")
        cluster_tsv = self.root / "clusters.tsv"
        cluster_tsv.write_text(
            "AFQ|missing\tAFQ|missing\n",
            encoding="utf-8",
        )
        empty_output = self.root / "empty-output"
        with self.assertRaisesRegex(CatalogBuildError, "No cluster members"):
            build_clustered_distillation_catalog(
                cluster_tsv=cluster_tsv,
                source_roots=self.source_roots,
                output_dir=empty_output,
                scan_workers=2,
            )
        self.assertFalse(empty_output.exists())

        cluster_tsv.write_text(
            "AFQ|unclustered\tAFQ|unclustered\n",
            encoding="utf-8",
        )
        mismatch_output = self.root / "mismatch-output"
        with self.assertRaisesRegex(CatalogBuildError, "Expected count"):
            build_clustered_distillation_catalog(
                cluster_tsv=cluster_tsv,
                source_roots=self.source_roots,
                output_dir=mismatch_output,
                scan_workers=2,
                expected_counts={"surviving_rows": 2},
            )
        self.assertFalse(mismatch_output.exists())

    def test_parse_expected_counts(self):
        self.assertEqual(
            _parse_expected_counts(
                ["inventory_entries=10", "source_rows.afdb=7"]
            ),
            {"inventory_entries": 10, "source_rows.afdb": 7},
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _parse_expected_counts(["input_rows=1", "input_rows=1"])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _parse_expected_counts(["input_rows=-1"])

    def test_parse_source_root_normalizes_prefix(self):
        parsed = parse_source_root(f"afdb,AFQ,{self.afdb}")
        self.assertEqual(parsed.root_id, "afdb")
        self.assertEqual(parsed.source_prefix, "AFQ|")
        self.assertEqual(parsed.path, self.afdb)


if __name__ == "__main__":
    unittest.main()
