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
import multiprocessing
import os
import pickle
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from protenix.data.pipeline.distillation_runtime_index import (
    DistillationRuntimeIndex,
)
from protenix.data.pipeline.dataset import BaseSingleDataset
from scripts.distillation.build_runtime_distillation_index import (
    RuntimeIndexError,
    build_runtime_distillation_index,
    verify_runtime_distillation_index,
)


def _fork_runtime_lookup(index, connection):
    try:
        connection.send(
            {
                "address": index.get_address(1),
                "owner_pid": index._owner_pid,
                "process_pid": os.getpid(),
            }
        )
    finally:
        connection.close()


class DistillationRuntimeIndexTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.catalog_dir = self.root / "catalog"
        self.catalog_dir.mkdir()
        self.source_a = self.root / "source-a"
        self.source_b = self.root / "source-b"
        self.source_a.mkdir()
        self.source_b.mkdir()
        (self.catalog_dir / "sampling").mkdir()
        (self.catalog_dir / "roots.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "roots": [
                        {"root_id": "afdb", "path": str(self.source_a)},
                        {"root_id": "mgnify", "path": str(self.source_b)},
                    ],
                }
            ),
            encoding="utf-8",
        )
        self._write_sampling(
            cluster_ids=[10, 12],
            first_row_ids=[0, 2],
            cluster_sizes=[2, 1],
        )
        self.index_path = self.catalog_dir / "distillation_index.csv.gz"
        self.rows = [
            {
                "row_id": "0",
                "source_root_id": "afdb",
                "cif_relpath": "bucket/000/a.cif",
                "pdb_id": "a",
            },
            {
                "row_id": "1",
                "source_root_id": "mgnify",
                "cif_relpath": "nested/b.cif",
                "pdb_id": "b",
            },
            {
                "row_id": "2",
                "source_root_id": "afdb",
                "cif_relpath": "c.cif",
                "pdb_id": "c",
            },
        ]
        self._write_index(self.rows)

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_sampling(self, *, cluster_ids, first_row_ids, cluster_sizes):
        sampling = self.catalog_dir / "sampling"
        (sampling / "cluster_ids.u32").write_bytes(
            b"".join(struct.pack("<I", value) for value in cluster_ids)
        )
        (sampling / "first_row_ids.u64").write_bytes(
            b"".join(struct.pack("<Q", value) for value in first_row_ids)
        )
        (sampling / "cluster_sizes.u32").write_bytes(
            b"".join(struct.pack("<I", value) for value in cluster_sizes)
        )
        arrays = {}
        for name, filename, dtype in (
            ("cluster_ids", "cluster_ids.u32", "uint32"),
            ("first_row_ids", "first_row_ids.u64", "uint64"),
            ("cluster_sizes", "cluster_sizes.u32", "uint32"),
        ):
            payload = (sampling / filename).read_bytes()
            item_size = 8 if dtype == "uint64" else 4
            arrays[name] = {
                "path": f"sampling/{filename}",
                "dtype": dtype,
                "endian": "little",
                "count": len(payload) // item_size,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        (self.catalog_dir / "manifest.json").write_text(
            json.dumps({"sampling": {"arrays": arrays}}),
            encoding="utf-8",
        )

    def _write_index(self, rows):
        if self.index_path.suffix == ".gz":
            file_obj = gzip.open(
                self.index_path,
                "wt",
                encoding="utf-8",
                newline="",
                compresslevel=1,
            )
        else:
            file_obj = self.index_path.open("w", encoding="utf-8", newline="")
        with file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "row_id",
                    "source_root_id",
                    "cif_relpath",
                    "pdb_id",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    def _build(self, name="runtime"):
        output = self.root / name
        manifest = build_runtime_distillation_index(
            input_index=self.index_path,
            output_dir=output,
            progress_every_rows=0,
        )
        return output, manifest

    def test_build_verify_random_access_cluster_lookup_and_pickle_reopen(self):
        runtime_dir, manifest = self._build()

        self.assertEqual(manifest["row_count"], 3)
        self.assertEqual(manifest["cluster_count"], 2)
        self.assertEqual(manifest["root_ids"], ["afdb", "mgnify"])
        self.assertEqual((runtime_dir / "row_offsets.u64").stat().st_size, 32)
        self.assertEqual((runtime_dir / "source_root_ids.u8").read_bytes(), b"\x00\x01\x00")
        self.assertTrue((runtime_dir / "sampling/first_row_ids.u64").is_file())
        self.assertEqual(
            verify_runtime_distillation_index(runtime_dir, verify_source=True)["row_count"],
            3,
        )

        index = DistillationRuntimeIndex(runtime_dir)
        self.assertEqual(index._maps, {})
        self.assertEqual(index.get_address(2), ("afdb", "c.cif"))
        self.assertEqual(index.get_address(0), ("afdb", "bucket/000/a.cif"))
        paths = index.resolve_paths(1)
        self.assertEqual(paths.cif_path, self.source_b / "nested/b.cif")
        self.assertEqual(paths.a3m_path, self.source_b / "nested/b.a3m")
        self.assertEqual(
            paths.bioassembly_path,
            self.source_b / "nested/bioassembly/b.pkl.gz",
        )
        self.assertEqual(index.get_cluster_id(1), 12)
        self.assertEqual(index.get_cluster_size(0), 2)
        self.assertEqual(index.get_member_row(0, 1), 1)
        self.assertEqual(index.get_member_row(1, 0), 2)
        np.testing.assert_array_equal(
            index.sample_member_rows(
                np.asarray([0, 0, 1], dtype=np.int64),
                np.asarray([0.0, 0.999, 0.5], dtype=np.float64),
            ),
            np.asarray([0, 1, 2], dtype=np.int64),
        )
        self.assertEqual(index.get_row_cluster(0), (10, 2))
        self.assertEqual(index.get_row_cluster(1), (10, 2))
        self.assertEqual(index.get_row_cluster(2), (12, 1))
        self.assertEqual(len(index.sampling_fingerprint), 64)
        self.assertEqual(
            index.resume_signature()["sampling_fingerprint"],
            index.sampling_fingerprint,
        )
        with self.assertRaises(IndexError):
            index.get_member_row(1, 1)
        with self.assertRaises(ValueError):
            index.sample_member_rows(
                np.asarray([0], dtype=np.int64),
                np.asarray([1.0], dtype=np.float64),
            )

        restored = pickle.loads(pickle.dumps(index))
        self.assertEqual(restored._maps, {})
        self.assertEqual(restored.get_address(1), ("mgnify", "nested/b.cif"))
        self.assertTrue(restored._maps)

        inherited_map = restored._maps["row_offsets"]
        with patch(
            "protenix.data.pipeline.distillation_runtime_index.os.getpid",
            return_value=(restored._owner_pid or 0) + 1,
        ):
            self.assertEqual(restored.get_address(0), ("afdb", "bucket/000/a.cif"))
        self.assertTrue(inherited_map.closed)
        self.assertIsNot(restored._maps["row_offsets"], inherited_map)
        index.close()
        restored.close()

    def test_unsafe_unknown_root_and_noncontiguous_rows_are_not_published(self):
        invalid_cases = [
            (
                "unsafe",
                [
                    {
                        "row_id": "0",
                        "source_root_id": "afdb",
                        "cif_relpath": "../escape.cif",
                        "pdb_id": "x",
                    }
                ],
                "Unsafe",
            ),
            (
                "unknown-root",
                [
                    {
                        "row_id": "0",
                        "source_root_id": "other",
                        "cif_relpath": "x.cif",
                        "pdb_id": "x",
                    }
                ],
                "unknown source_root_id",
            ),
            (
                "row-gap",
                [
                    {
                        "row_id": "1",
                        "source_root_id": "afdb",
                        "cif_relpath": "x.cif",
                        "pdb_id": "x",
                    }
                ],
                "contiguous",
            ),
        ]
        for name, rows, message in invalid_cases:
            with self.subTest(name=name):
                self._write_index(rows)
                output = self.root / name
                with self.assertRaisesRegex(RuntimeIndexError, message):
                    build_runtime_distillation_index(
                        input_index=self.index_path,
                        output_dir=output,
                        progress_every_rows=0,
                    )
                self.assertFalse(output.exists())

    def test_real_fork_reopens_inherited_mmaps(self):
        runtime_dir, _ = self._build()
        index = DistillationRuntimeIndex(runtime_dir)
        self.assertEqual(index.get_address(0), ("afdb", "bucket/000/a.cif"))
        parent_map = index._maps["row_offsets"]
        parent_pid = index._owner_pid

        context = multiprocessing.get_context("fork")
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_fork_runtime_lookup,
            args=(index, child_connection),
        )
        process.start()
        child_connection.close()
        self.assertTrue(
            parent_connection.poll(10),
            "forked runtime-index lookup did not respond",
        )
        result = parent_connection.recv()
        parent_connection.close()
        process.join(timeout=10)

        self.assertEqual(process.exitcode, 0)
        self.assertEqual(result["address"], ("mgnify", "nested/b.cif"))
        self.assertEqual(result["owner_pid"], result["process_pid"])
        self.assertNotEqual(result["owner_pid"], parent_pid)
        self.assertFalse(parent_map.closed)
        index.close()

    def test_sampling_interval_mismatch_is_rejected(self):
        self._write_sampling(
            cluster_ids=[10, 12],
            first_row_ids=[0, 1],
            cluster_sizes=[1, 1],
        )
        output = self.root / "bad-sampling"
        with self.assertRaisesRegex(RuntimeIndexError, "cover"):
            build_runtime_distillation_index(
                input_index=self.index_path,
                output_dir=output,
                progress_every_rows=0,
            )
        self.assertFalse(output.exists())

    def test_verify_and_accessor_reject_corruption(self):
        runtime_dir, _ = self._build()
        paths_file = runtime_dir / "cif_relpaths.bin"
        payload = bytearray(paths_file.read_bytes())
        payload[0] ^= 1
        paths_file.write_bytes(payload)
        with self.assertRaisesRegex(RuntimeIndexError, "SHA-256 mismatch"):
            verify_runtime_distillation_index(runtime_dir)

        runtime_dir_2, _ = self._build("runtime-truncated")
        roots_file = runtime_dir_2 / "source_root_ids.u8"
        roots_file.write_bytes(roots_file.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            DistillationRuntimeIndex(runtime_dir_2)

        runtime_dir_3, _ = self._build("runtime-roots-corrupt")
        runtime_roots = runtime_dir_3 / "roots.json"
        roots_payload = runtime_roots.read_bytes().replace(
            b"source-a",
            b"source-x",
        )
        self.assertEqual(len(roots_payload), runtime_roots.stat().st_size)
        runtime_roots.write_bytes(roots_payload)
        with self.assertRaisesRegex(ValueError, "roots SHA-256 mismatch"):
            DistillationRuntimeIndex(runtime_dir_3)

        runtime_dir_4, _ = self._build("runtime-manifest-corrupt")
        manifest_path = runtime_dir_4 / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"]["row_offsets"]["count"] = 99
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            DistillationRuntimeIndex(runtime_dir_4)
        with self.assertRaisesRegex(RuntimeIndexError, "count mismatch"):
            verify_runtime_distillation_index(runtime_dir_4)

    def test_source_sampling_sha_and_output_ancestor_are_rejected(self):
        cluster_ids_path = self.catalog_dir / "sampling/cluster_ids.u32"
        cluster_ids = bytearray(cluster_ids_path.read_bytes())
        cluster_ids[0] ^= 1
        cluster_ids_path.write_bytes(cluster_ids)
        with self.assertRaisesRegex(RuntimeIndexError, "Source sampling SHA-256"):
            build_runtime_distillation_index(
                input_index=self.index_path,
                output_dir=self.root / "stale-sampling",
                progress_every_rows=0,
            )

        # Recreate a consistent source, then ensure even explicit overwrite
        # cannot target an ancestor containing the authority catalog.
        self._write_sampling(
            cluster_ids=[10, 12],
            first_row_ids=[0, 2],
            cluster_sizes=[2, 1],
        )
        with self.assertRaisesRegex(ValueError, "must not contain"):
            build_runtime_distillation_index(
                input_index=self.index_path,
                output_dir=self.root,
                overwrite=True,
                progress_every_rows=0,
            )
        self.assertTrue(self.index_path.is_file())

    def test_existing_output_requires_explicit_overwrite(self):
        runtime_dir, first_manifest = self._build()
        with self.assertRaises(FileExistsError):
            build_runtime_distillation_index(
                input_index=self.index_path,
                output_dir=runtime_dir,
                progress_every_rows=0,
            )

        second_manifest = build_runtime_distillation_index(
            input_index=self.index_path,
            output_dir=runtime_dir,
            overwrite=True,
            progress_every_rows=0,
        )
        self.assertEqual(first_manifest["row_count"], second_manifest["row_count"])
        self.assertEqual(verify_runtime_distillation_index(runtime_dir)["row_count"], 3)

    def test_plain_csv_input_is_supported(self):
        self.index_path = self.catalog_dir / "distillation_index.csv"
        self._write_index(self.rows)
        runtime_dir, manifest = self._build("runtime-plain")

        self.assertEqual(manifest["row_count"], 3)
        self.assertEqual(
            DistillationRuntimeIndex(runtime_dir).get_address(1),
            ("mgnify", "nested/b.cif"),
        )

    def test_base_dataset_uses_runtime_rows_without_reading_authority_csv(self):
        runtime_dir, _ = self._build()
        with patch(
            "protenix.data.pipeline.dataset.read_indices_csv"
        ) as read_indices:
            dataset = BaseSingleDataset(
                mmcif_dir=runtime_dir,
                bioassembly_dict_dir=None,
                indices_fpath=self.root / "must-not-be-read.csv.gz",
                runtime_index_dir=runtime_dir,
                cropping_configs={"crop_size": 4},
                msa_featurizer=None,
                template_featurizer=None,
                source_roots_fpath=runtime_dir / "roots.json",
                pdb_list="",
                is_distillation=True,
                random_sample_if_failed=False,
            )

        read_indices.assert_not_called()
        self.assertEqual(len(dataset), 3)
        row = dataset._get_sample_indice(1)
        self.assertEqual(row["pdb_id"], "b")
        self.assertEqual(row["source_root_id"], "mgnify")
        self.assertEqual(row["cif_relpath"], "nested/b.cif")
        self.assertEqual(row["cluster_id"], 10)
        self.assertEqual(row["cluster_size"], 2)
        self.assertEqual(row["type"], "chain")
        self.assertTrue(dataset.is_distillation)

        alternate_roots = self.root / "alternate-roots.json"
        alternate_roots.write_text(
            json.dumps(
                {
                    "roots": [
                        {"root_id": "afdb", "path": "/alternate-a"},
                        {"root_id": "mgnify", "path": "/alternate-b"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "different source_roots_fpath"):
            BaseSingleDataset(
                mmcif_dir=runtime_dir,
                bioassembly_dict_dir=runtime_dir,
                indices_fpath=self.index_path,
                runtime_index_dir=runtime_dir,
                cropping_configs={"crop_size": 4},
                source_roots_fpath=alternate_roots,
                pdb_list="",
                is_distillation=True,
                random_sample_if_failed=False,
            )


if __name__ == "__main__":
    unittest.main()
