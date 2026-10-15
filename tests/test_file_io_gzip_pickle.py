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

import gzip
import os
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from protenix.utils.file_io import (
    dump_gzip_pickle,
    load_gzip_pickle,
    read_indices_csv,
)


class _Unpicklable:
    def __reduce__(self):
        raise RuntimeError("intentional pickle failure")


class DumpGzipPickleTest(unittest.TestCase):
    def test_training_reader_rejects_conversion_audit_before_pandas_load(self):
        with TemporaryDirectory() as temporary_directory:
            audit_path = Path(temporary_directory) / "audit.csv.gz"
            with gzip.open(
                audit_path,
                "wt",
                encoding="utf-8",
                newline="",
            ) as stream:
                stream.write("artifact_role,pdb_id,cluster_id\n")
                stream.write(
                    "distillation_bioassembly_conversion_audit,"
                    "sample,NotInClusterTxt\n"
                )

            with patch(
                "protenix.utils.file_io.pd.read_csv",
            ) as pandas_read:
                with self.assertRaisesRegex(
                    ValueError,
                    "non-training audit artifact",
                ):
                    read_indices_csv(audit_path)
            pandas_read.assert_not_called()

            allowed = read_indices_csv(
                audit_path,
                allow_non_training_audit=True,
            )
            self.assertEqual(len(allowed), 1)
            self.assertEqual(
                allowed.iloc[0]["artifact_role"],
                "distillation_bioassembly_conversion_audit",
            )

    def test_defaults_preserve_pickle_dump_behavior_without_fsync(self):
        data = {"sequence": "ACDEFG", "scores": [1, 2, 3]}

        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "data.pkl.gz"
            with patch("protenix.utils.file_io.os.fsync") as fsync:
                dump_gzip_pickle(data, output_path)

            with gzip.open(output_path, "rb") as f:
                pickle_payload = f.read()

        fsync.assert_not_called()
        self.assertEqual(pickle_payload, pickle.dumps(data))

    def test_accepts_compression_level_protocol_and_legacy_alias(self):
        data = {"sequence": "ACDEFG" * 100}

        for keyword in ("compression_level", "compresslevel"):
            with self.subTest(keyword=keyword):
                with TemporaryDirectory() as temporary_directory:
                    output_path = (
                        Path(temporary_directory) / "data.pkl.gz"
                    )
                    with patch(
                        "protenix.utils.file_io.gzip.open",
                        wraps=gzip.open,
                    ) as gzip_open:
                        dump_gzip_pickle(
                            data,
                            output_path,
                            protocol=2,
                            **{keyword: 1},
                        )

                    self.assertEqual(
                        gzip_open.call_args.kwargs["compresslevel"],
                        1,
                    )
                    with gzip.open(output_path, "rb") as f:
                        pickle_payload = f.read()

                self.assertTrue(
                    pickle_payload.startswith(b"\x80\x02")
                )
                self.assertEqual(pickle.loads(pickle_payload), data)

    def test_rejects_conflicting_compression_level_aliases(self):
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "data.pkl.gz"
            with self.assertRaisesRegex(ValueError, "disagree"):
                dump_gzip_pickle(
                    {},
                    output_path,
                    compression_level=1,
                    compresslevel=6,
                )
            self.assertFalse(output_path.exists())

    def test_atomic_write_replaces_destination_from_same_directory(self):
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            output_path = directory / "data.pkl.gz"
            dump_gzip_pickle({"version": 1}, output_path)
            output_path.chmod(0o640)

            with patch(
                "protenix.utils.file_io.os.replace",
                wraps=os.replace,
            ) as replace, patch(
                "protenix.utils.file_io.os.fsync"
            ) as fsync:
                dump_gzip_pickle(
                    {"version": 2},
                    output_path,
                    compression_level=6,
                    protocol=pickle.HIGHEST_PROTOCOL,
                    atomic=True,
                )

            source, destination = replace.call_args.args
            fsync.assert_not_called()
            self.assertEqual(
                Path(source).parent.resolve(),
                directory.resolve(),
            )
            self.assertEqual(Path(destination), output_path)
            self.assertEqual(
                load_gzip_pickle(output_path),
                {"version": 2},
            )
            self.assertEqual(output_path.stat().st_mode & 0o777, 0o640)
            self.assertEqual(list(directory.iterdir()), [output_path])

    def test_atomic_new_file_mode_follows_umask(self):
        for umask, expected_mode in ((0o022, 0o644), (0o027, 0o640)):
            with self.subTest(umask=oct(umask)):
                with TemporaryDirectory() as temporary_directory:
                    output_path = Path(temporary_directory) / "data.pkl.gz"
                    previous_umask = os.umask(umask)
                    try:
                        dump_gzip_pickle(
                            {"version": 1},
                            output_path,
                            atomic=True,
                        )
                    finally:
                        os.umask(previous_umask)

                    self.assertEqual(
                        output_path.stat().st_mode & 0o777,
                        expected_mode,
                    )
                    self.assertEqual(
                        load_gzip_pickle(output_path),
                        {"version": 1},
                    )

    def test_atomic_write_cleans_up_after_failure(self):
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            output_path = directory / "data.pkl.gz"
            original_data = {"version": 1}
            dump_gzip_pickle(original_data, output_path)

            with self.assertRaisesRegex(
                RuntimeError,
                "intentional pickle failure",
            ):
                dump_gzip_pickle(
                    _Unpicklable(),
                    output_path,
                    atomic=True,
                )

            self.assertEqual(
                load_gzip_pickle(output_path),
                original_data,
            )
            self.assertEqual(list(directory.iterdir()), [output_path])

    def test_durable_write_fsyncs_file_and_directory(self):
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "data.pkl.gz"
            with patch(
                "protenix.utils.file_io.os.fsync",
                wraps=os.fsync,
            ) as fsync:
                dump_gzip_pickle(
                    {"version": 1},
                    output_path,
                    atomic=True,
                    durable=True,
                )

            expected_calls = (
                2 if hasattr(os, "O_DIRECTORY") else 1
            )
            self.assertEqual(fsync.call_count, expected_calls)
            self.assertEqual(
                load_gzip_pickle(output_path),
                {"version": 1},
            )


if __name__ == "__main__":
    unittest.main()
