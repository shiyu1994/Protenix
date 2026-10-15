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
import tempfile
import unittest
from pathlib import Path

from scripts.distillation.build_foldseek_short_protein_manifest import (
    EXPECTED_ARTIFACT_ROLE,
    FoldseekManifestError,
    _is_pilot_path,
    build_foldseek_short_protein_manifest,
    verify_root_links,
)


class FoldseekShortProteinManifestTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.afdb = self.root / "afdb"
        self.mgnify = self.root / "mgnify"
        self.afdb.mkdir()
        self.mgnify.mkdir()
        self.roots_json = self.root / "roots.json"
        self.roots_json.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "roots": [
                        {"root_id": "afdb", "path": str(self.afdb)},
                        {"root_id": "mgnify", "path": str(self.mgnify)},
                    ],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def _write_index(path: Path, rows: list[dict[str, object]]) -> None:
        with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "artifact_role",
                    "entry_id",
                    "source_root_id",
                    "cif_relpath",
                    "canonical_length",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def _row(
        self,
        entry_id: str,
        root_id: str,
        cif_relpath: str,
        canonical_length: int,
    ) -> dict[str, object]:
        return {
            "artifact_role": EXPECTED_ARTIFACT_ROLE,
            "entry_id": entry_id,
            "source_root_id": root_id,
            "cif_relpath": cif_relpath,
            "canonical_length": canonical_length,
        }

    def test_streams_exact_length_filter_and_builds_relative_outputs(self):
        index = self.root / "indices.csv.gz"
        rows = [
            self._row("a", "afdb", "bucket/a.cif", 200),
            self._row("b", "afdb", "bucket/b.cif", 201),
            self._row("c", "mgnify", "nested/c.cif", 7),
            self._row("d", "afdb", "d.cif", 199),
        ]
        self._write_index(index, rows)
        output = self.root / "manifest"

        stats = build_foldseek_short_protein_manifest(
            input_index=index,
            roots_json=self.roots_json,
            output_dir=output,
            pilot_hash_modulus=1,
            pilot_hash_buckets=1,
            progress_every_rows=0,
        )

        expected_paths = [
            "afdb/bucket/a.cif",
            "mgnify/nested/c.cif",
            "afdb/d.cif",
        ]
        self.assertEqual(
            (output / "short_le200.paths.tsv")
            .read_text(encoding="utf-8")
            .splitlines(),
            expected_paths,
        )
        self.assertEqual(
            (output / "short_le200.pilot.paths.tsv")
            .read_text(encoding="utf-8")
            .splitlines(),
            expected_paths,
        )
        with gzip.open(
            output / "short_le200.mapping.tsv.gz",
            "rt",
            encoding="utf-8",
        ) as stream:
            self.assertEqual(
                stream.read().splitlines(),
                [
                    "0\tafdb/bucket/a.cif\t200",
                    "1\tmgnify/nested/c.cif\t7",
                    "2\tafdb/d.cif\t199",
                ],
            )

        self.assertEqual(stats["counts"]["input_rows"], 4)
        self.assertEqual(stats["counts"]["selected_rows"], 3)
        self.assertEqual(stats["counts"]["excluded_too_short"], 0)
        self.assertEqual(stats["counts"]["excluded_too_long"], 1)
        self.assertEqual(stats["counts"]["pilot_rows"], 3)
        self.assertEqual(stats["counts"]["selected_length_min"], 7)
        self.assertEqual(stats["counts"]["selected_length_max"], 200)
        self.assertEqual(stats["counts"]["selected_length_sum"], 406)
        self.assertEqual(stats["by_root"]["afdb"]["selected_rows"], 2)
        self.assertEqual(stats["by_root"]["mgnify"]["selected_rows"], 1)
        self.assertTrue((output / "roots" / "afdb").is_symlink())
        self.assertEqual((output / "roots" / "afdb").resolve(), self.afdb)
        verify_root_links(output, self.roots_json)

        published_stats = json.loads(
            (output / "short_le200.stats.json").read_text(encoding="utf-8")
        )
        self.assertEqual(published_stats["counts"], stats["counts"])
        self.assertEqual(
            published_stats["path_contract"]["base_directory"],
            "roots",
        )

    def test_pilot_is_stable_and_is_a_selected_subset(self):
        index = self.root / "indices.csv.gz"
        rows = [
            self._row(
                str(ordinal),
                "afdb",
                f"bucket/model_{ordinal:02d}.cif",
                100 + ordinal,
            )
            for ordinal in range(20)
        ]
        self._write_index(index, rows)
        outputs = [self.root / "manifest-a", self.root / "manifest-b"]
        for output in outputs:
            build_foldseek_short_protein_manifest(
                input_index=index,
                roots_json=self.roots_json,
                output_dir=output,
                pilot_hash_modulus=7,
                pilot_hash_buckets=2,
                pilot_hash_seed="fixture",
                progress_every_rows=0,
            )

        pilots = [
            (output / "short_le200.pilot.paths.tsv")
            .read_text(encoding="utf-8")
            .splitlines()
            for output in outputs
        ]
        expected = [
            f"afdb/bucket/model_{ordinal:02d}.cif"
            for ordinal in range(20)
            if _is_pilot_path(
                f"afdb/bucket/model_{ordinal:02d}.cif",
                seed="fixture",
                modulus=7,
                buckets=2,
            )
        ]
        self.assertEqual(pilots[0], expected)
        self.assertEqual(pilots[0], pilots[1])
        self.assertEqual(
            (outputs[0] / "short_le200.mapping.tsv.gz").read_bytes(),
            (outputs[1] / "short_le200.mapping.tsv.gz").read_bytes(),
        )

    def test_invalid_late_row_leaves_no_partial_output(self):
        index = self.root / "indices.csv.gz"
        self._write_index(
            index,
            [
                self._row("a", "afdb", "valid/a.cif", 100),
                self._row("b", "afdb", "../escape.cif", 100),
            ],
        )
        output = self.root / "manifest"

        with self.assertRaisesRegex(FoldseekManifestError, "unsafe cif_relpath"):
            build_foldseek_short_protein_manifest(
                input_index=index,
                roots_json=self.roots_json,
                output_dir=output,
                progress_every_rows=0,
            )

        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".manifest.tmp.*")), [])

    def test_inclusive_minimum_length_selects_long_proteins(self):
        index = self.root / "indices.csv.gz"
        rows = [
            self._row("a", "afdb", "bucket/a.cif", 199),
            self._row("b", "afdb", "bucket/b.cif", 200),
            self._row("c", "mgnify", "nested/c.cif", 201),
        ]
        self._write_index(index, rows)
        output = self.root / "manifest-long"

        stats = build_foldseek_short_protein_manifest(
            input_index=index,
            roots_json=self.roots_json,
            output_dir=output,
            min_length=200,
            max_length=2_147_483_647,
            output_prefix="long_ge200",
            pilot_hash_modulus=1,
            pilot_hash_buckets=1,
            progress_every_rows=0,
        )

        self.assertEqual(
            (output / "long_ge200.paths.tsv").read_text(encoding="utf-8").splitlines(),
            ["afdb/bucket/b.cif", "mgnify/nested/c.cif"],
        )
        self.assertEqual(stats["counts"]["selected_rows"], 2)
        self.assertEqual(stats["counts"]["excluded_too_short"], 1)
        self.assertEqual(stats["counts"]["excluded_too_long"], 0)
        self.assertEqual(stats["counts"]["selected_length_min"], 200)
        self.assertEqual(stats["counts"]["selected_length_max"], 201)
        self.assertEqual(
            stats["selection"],
            {
                "canonical_length_operator": "closed_interval",
                "min_canonical_length": 200,
                "max_canonical_length": 2_147_483_647,
            },
        )


if __name__ == "__main__":
    unittest.main()
