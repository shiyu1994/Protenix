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

from protenix.data.pipeline.distillation_runtime_index import (
    DistillationRuntimeIndex,
)
from scripts.distillation.build_length_partitioned_runtime_indices import (
    PartitionBuildError,
    build_length_partitioned_runtime_indices,
    verify_length_partitioned_runtime_indices,
)
from scripts.distillation.build_runtime_distillation_index import (
    RuntimeIndexError,
    build_runtime_distillation_index,
)


class LengthPartitionedRuntimeIndicesTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.root0 = self.root / "root0"
        self.root1 = self.root / "root1"
        self.root0.mkdir()
        self.root1.mkdir()
        self.roots_json = self.source / "roots.json"
        self.roots_json.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "roots": [
                        {
                            "root_id": "0",
                            "path": str(self.root0),
                            "source_prefix": "AFQ|",
                        },
                        {
                            "root_id": "1",
                            "path": str(self.root1),
                            "source_prefix": "MGN|",
                        },
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.rows = [
            ("0", "c0_short_member.cif"),
            ("1", "c0_long_member.cif"),
            ("0", "c0_rep_short.cif"),
            ("1", "c1_rep_long.cif"),
            ("0", "c2_rep_long.cif"),
            ("1", "c2_short_member.cif"),
            ("0", "c3_rep_boundary.cif"),
        ]
        self.index_csv = self.source / "index.csv.gz"
        with gzip.open(
            self.index_csv,
            "wt",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["row_id", "source_root_id", "cif_relpath"])
            for row_id, (root_id, relpath) in enumerate(self.rows):
                writer.writerow([row_id, root_id, relpath])

        sampling = self.source / "sampling"
        sampling.mkdir()
        self._write_binary(
            sampling / "cluster_ids.u32",
            "<I",
            [10, 20, 42, 99],
        )
        self._write_binary(
            sampling / "first_row_ids.u64",
            "<Q",
            [0, 3, 4, 6],
        )
        self._write_binary(
            sampling / "cluster_sizes.u32",
            "<I",
            [3, 1, 2, 1],
        )
        self.full_runtime = self.root / "full_runtime"
        self.full_manifest = build_runtime_distillation_index(
            input_index=self.index_csv,
            output_dir=self.full_runtime,
            roots_json=self.roots_json,
            sampling_dir=sampling,
            progress_every_rows=0,
        )

        self.cluster_rows = [
            [10, "AFQ|c0_rep_short", "AFQ|c0_rep_short", 3, 3, 0, 0, 2, "cross_source"],
            [20, "MGN|c1_rep_long", "MGN|c1_rep_long", 1, 1, 0, 3, 3, ""],
            [42, "AFQ|c2_rep_long", "AFQ|c2_rep_long", 2, 2, 0, 4, 4, "cross_source"],
            [99, "AFQ|c3_rep_boundary", "AFQ|c3_rep_boundary", 1, 1, 0, 6, 6, ""],
        ]
        self.clusters_csv = self.source / "clusters.csv.gz"
        self._write_clusters(self.cluster_rows)

        self.short_paths = self.source / "short_le200.paths.tsv"
        self.short_stats = self.source / "short_le200.stats.json"
        self.canonical_audit = (
            self.source / "distillation_bioassembly_indices.csv.gz"
        )
        self.canonical_audit.write_bytes(b"canonical-length-fixture\n")
        self.canonical_audit_sha256 = hashlib.sha256(
            self.canonical_audit.read_bytes()
        ).hexdigest()
        (self.source / "_FINALIZE_SUCCESS.json").write_text(
            json.dumps(
                {
                    "artifact_role": "distillation_bioassembly_finalize",
                    "complete_cifs": len(self.rows),
                    "failure_count": 0,
                    "merged_audit_present": True,
                    "merged_audit_relpath": self.canonical_audit.name,
                    "merged_audit_rows": len(self.rows),
                    "merged_audit_sha256": self.canonical_audit_sha256,
                    "merged_audit_size": self.canonical_audit.stat().st_size,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.short_path_values = [
            "0/c0_short_member.cif",
            "0/c0_rep_short.cif",
            "1/c2_short_member.cif",
            "0/c3_rep_boundary.cif",
        ]
        self._write_short_inputs(self.short_path_values)
        self.short_output = self.root / "short_runtime"
        self.long_output = self.root / "long_runtime"

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def _write_binary(path: Path, fmt: str, values: list[int]) -> None:
        path.write_bytes(b"".join(struct.pack(fmt, value) for value in values))

    @staticmethod
    def _read_binary(path: Path, fmt: str) -> list[int]:
        return [
            value[0]
            for value in struct.iter_unpack(fmt, path.read_bytes())
        ]

    def _write_clusters(self, rows: list[list[object]]) -> None:
        with gzip.open(
            self.clusters_csv,
            "wt",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "cluster_id",
                    "source_representative_id",
                    "representative_id",
                    "original_size",
                    "cluster_size",
                    "removed_size",
                    "first_row_id",
                    "representative_row_id",
                    "flags",
                ]
            )
            writer.writerows(rows)

    def _write_short_inputs(self, values: list[str]) -> None:
        payload = "".join(f"{value}\n" for value in values).encode("utf-8")
        self.short_paths.write_bytes(payload)
        lengths = [50, 150, 199, 200]
        self.short_stats.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "selection": {
                        "canonical_length_operator": "<=",
                        "max_canonical_length": 200,
                    },
                    "counts": {
                        "input_rows": len(self.rows),
                        "selected_rows": len(values),
                        "excluded_too_long": len(self.rows) - len(values),
                        "selected_length_min": min(lengths[: len(values)]),
                        "selected_length_max": max(lengths[: len(values)]),
                        "selected_length_sum": sum(lengths[: len(values)]),
                    },
                    "input": {
                        "index": str(self.canonical_audit),
                        "index_size_bytes": self.canonical_audit.stat().st_size,
                        "roots_json_sha256": self.full_manifest["artifacts"][
                            "roots"
                        ]["sha256"],
                    },
                    "outputs": {
                        "paths": {
                            "lines": len(values),
                            "size_bytes": len(payload),
                            "content_sha256": hashlib.sha256(
                                payload
                            ).hexdigest(),
                        },
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _build(self) -> dict[str, object]:
        return build_length_partitioned_runtime_indices(
            full_runtime_dir=self.full_runtime,
            clusters_csv=self.clusters_csv,
            short_paths=self.short_paths,
            short_paths_stats=self.short_stats,
            short_output_dir=self.short_output,
            long_output_dir=self.long_output,
            progress_every_clusters=0,
        )

    def test_whole_clusters_are_partitioned_by_final_representative(self):
        receipt = self._build()
        self.assertEqual(receipt["coverage"]["row_count_sum"], 7)
        self.assertEqual(receipt["coverage"]["cluster_count_sum"], 4)
        self.assertEqual(
            receipt["partitions"]["short_le200"],
            {
                "row_count": 4,
                "cluster_count": 2,
                "short_members": 3,
                "long_members": 1,
                "mixed_clusters": 1,
            },
        )
        self.assertEqual(
            receipt["partitions"]["long_gt200"],
            {
                "row_count": 3,
                "cluster_count": 2,
                "short_members": 1,
                "long_members": 2,
                "mixed_clusters": 1,
            },
        )

        self.assertEqual(
            self._read_binary(
                self.short_output / "sampling/cluster_ids.u32",
                "<I",
            ),
            [10, 99],
        )
        self.assertEqual(
            self._read_binary(
                self.short_output / "sampling/first_row_ids.u64",
                "<Q",
            ),
            [0, 3],
        )
        self.assertEqual(
            self._read_binary(
                self.short_output / "sampling/cluster_sizes.u32",
                "<I",
            ),
            [3, 1],
        )
        self.assertEqual(
            self._read_binary(
                self.long_output / "sampling/cluster_ids.u32",
                "<I",
            ),
            [20, 42],
        )
        self.assertEqual(
            self._read_binary(
                self.long_output / "sampling/first_row_ids.u64",
                "<Q",
            ),
            [0, 1],
        )
        self.assertEqual(
            self._read_binary(
                self.long_output / "sampling/cluster_sizes.u32",
                "<I",
            ),
            [1, 2],
        )

        short_runtime = DistillationRuntimeIndex(self.short_output)
        long_runtime = DistillationRuntimeIndex(self.long_output)
        self.assertEqual(
            [short_runtime.get_address(i) for i in range(len(short_runtime))],
            [self.rows[index] for index in (0, 1, 2, 6)],
        )
        self.assertEqual(
            [long_runtime.get_address(i) for i in range(len(long_runtime))],
            [self.rows[index] for index in (3, 4, 5)],
        )
        verified = verify_length_partitioned_runtime_indices(
            short_runtime_dir=self.short_output,
            long_runtime_dir=self.long_output,
            full_runtime_dir=self.full_runtime,
            verify_recorded_sources=True,
        )
        self.assertEqual(verified["pair_id"], receipt["pair_id"])

    def test_existing_either_output_fails_closed(self):
        self.short_output.mkdir()
        (self.short_output / "sentinel").write_text(
            "keep\n",
            encoding="utf-8",
        )
        with self.assertRaises(FileExistsError):
            self._build()
        self.assertEqual(
            (self.short_output / "sentinel").read_text(encoding="utf-8"),
            "keep\n",
        )
        self.assertFalse(self.long_output.exists())

    def test_schema_rejects_a_non_200_threshold(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            build_length_partitioned_runtime_indices(
                full_runtime_dir=self.full_runtime,
                clusters_csv=self.clusters_csv,
                short_paths=self.short_paths,
                short_paths_stats=self.short_stats,
                short_output_dir=self.short_output,
                long_output_dir=self.long_output,
                max_representative_length=199,
                progress_every_clusters=0,
            )
        self.assertFalse(self.short_output.exists())
        self.assertFalse(self.long_output.exists())

    def test_unknown_short_path_rejects_without_publishing(self):
        values = list(self.short_path_values)
        values[-1] = "0/not_in_full_runtime.cif"
        self._write_short_inputs(values)
        with self.assertRaisesRegex(
            PartitionBuildError,
            "short paths are absent",
        ):
            self._build()
        self.assertFalse(self.short_output.exists())
        self.assertFalse(self.long_output.exists())

    def test_canonical_audit_must_match_completion_sha256(self):
        payload = bytearray(self.canonical_audit.read_bytes())
        payload[0] ^= 1
        self.canonical_audit.write_bytes(payload)
        self._write_short_inputs(self.short_path_values)
        with self.assertRaisesRegex(
            PartitionBuildError,
            "SHA-256 differs",
        ):
            self._build()
        self.assertFalse(self.short_output.exists())
        self.assertFalse(self.long_output.exists())

    def test_final_representative_id_must_match_representative_row(self):
        rows = [list(row) for row in self.cluster_rows]
        rows[0][2] = "AFQ|wrong"
        self._write_clusters(rows)
        with self.assertRaisesRegex(
            PartitionBuildError,
            "representative_id does not match",
        ):
            self._build()
        self.assertFalse(self.short_output.exists())
        self.assertFalse(self.long_output.exists())

    def test_verify_detects_tampered_runtime_artifact(self):
        self._build()
        path = self.short_output / "source_root_ids.u8"
        payload = bytearray(path.read_bytes())
        payload[0] = 1
        path.write_bytes(payload)
        with self.assertRaises(RuntimeIndexError):
            verify_length_partitioned_runtime_indices(
                short_runtime_dir=self.short_output,
                long_runtime_dir=self.long_output,
                full_runtime_dir=self.full_runtime,
            )


if __name__ == "__main__":
    unittest.main()
