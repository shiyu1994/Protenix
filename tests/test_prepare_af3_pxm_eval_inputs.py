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

"""Tests for the immutable PXM evaluation input inventory."""

from __future__ import annotations

import gzip
import importlib.util
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluation"
    / "prepare_af3_pxm_eval_inputs.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "prepare_af3_pxm_eval_inputs",
    _SCRIPT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class PrepareAf3PxmEvalInputsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.bioassembly_dir = self.root / "bioassembly"
        self.official_raw_root = self.root / "official_raw"
        self.overlay_raw_root = self.root / "overlay_raw"
        self.output_root = self.root / "output"
        self.bioassembly_dir.mkdir()

        self.official_sequence = "ACDEFGHIKLMNPQ"
        self.overlay_sequence = "RSTVWYACDEFGHI"
        self.pdb_list = self.root / "pdb_ids.txt"
        self.pdb_list.write_text("test\n", encoding="utf-8")
        with gzip.open(self.bioassembly_dir / "test.pkl.gz", "wb") as handle:
            pickle.dump(
                {
                    "sequences": {
                        "1": self.official_sequence,
                        "2": self.overlay_sequence,
                        "3": "ATCG",
                    },
                    "entity_poly_type": {
                        "1": "polypeptide(L)",
                        "2": "polypeptide(L)",
                        "3": "polydeoxyribonucleotide",
                    },
                },
                handle,
            )

        self.official_mapping = self.root / "official.json"
        self.overlay_mapping = self.root / "overlay.json"
        self.official_mapping.write_text(
            json.dumps({self.official_sequence: "official/entity_1"}),
            encoding="utf-8",
        )
        self.overlay_mapping.write_text(
            json.dumps({self.overlay_sequence: "test/entity_2"}),
            encoding="utf-8",
        )
        self._write_source(
            self.official_raw_root / "official/entity_1",
            self.official_sequence,
        )
        self._write_source(
            self.overlay_raw_root / "test/entity_2",
            self.overlay_sequence,
        )

    @staticmethod
    def _write_source(directory: Path, sequence: str) -> None:
        directory.mkdir(parents=True)
        for filename in ("pairing.a3m", "non_pairing.a3m"):
            (directory / filename).write_text(
                f">query\n{sequence}\n",
                encoding="utf-8",
            )
        # An empty completed search is a legitimate zero-template result.
        (directory / "hmmsearch.a3m").write_text("", encoding="utf-8")

    def _arguments(self) -> list[str]:
        return [
            str(_SCRIPT_PATH),
            "--pdb-list",
            str(self.pdb_list),
            "--bioassembly-dir",
            str(self.bioassembly_dir),
            "--official-sequence-mapping",
            str(self.official_mapping),
            "--official-raw-root",
            str(self.official_raw_root),
            "--overlay-sequence-mapping",
            str(self.overlay_mapping),
            "--overlay-raw-root",
            str(self.overlay_raw_root),
            "--output-root",
            str(self.output_root),
        ]

    def test_resolves_public_and_overlay_sources_without_copying(self) -> None:
        with mock.patch.object(sys, "argv", self._arguments()):
            _MODULE.main()

        success = json.loads(
            (
                self.output_root / "manifests/_INPUT_SUCCESS.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(success["pdb_count"], 1)
        self.assertEqual(success["unique_protein_sequence_count"], 2)
        self.assertEqual(success["resolved_sequence_count"], 2)
        self.assertEqual(success["missing_sequence_count"], 0)

        sources = json.loads(
            (
                self.output_root
                / "manifests/protein_sequence_sources.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            sources[self.official_sequence],
            str(self.official_raw_root / "official/entity_1"),
        )
        self.assertEqual(
            sources[self.overlay_sequence],
            str(self.overlay_raw_root / "test/entity_2"),
        )

    def test_query_mismatch_is_fail_closed_and_removes_success_marker(self) -> None:
        success_path = self.output_root / "manifests/_INPUT_SUCCESS.json"
        success_path.parent.mkdir(parents=True)
        success_path.write_text('{"stale": true}\n', encoding="utf-8")
        (
            self.overlay_raw_root / "test/entity_2/pairing.a3m"
        ).write_text(
            ">query\nAAAAAAAAAAAAAA\n",
            encoding="utf-8",
        )

        with mock.patch.object(sys, "argv", self._arguments()):
            with self.assertRaisesRegex(RuntimeError, "coverage is incomplete"):
                _MODULE.main()

        self.assertFalse(success_path.exists())
        inventory = json.loads(
            (
                self.output_root
                / "manifests/protein_sequence_inventory.json"
            ).read_text(encoding="utf-8")
        )
        missing_records = [
            record
            for record in inventory["records"]
            if record["status"] == "missing"
        ]
        self.assertEqual(len(missing_records), 1)
        self.assertIn(
            "pairing.a3m_query_mismatch",
            missing_records[0]["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
