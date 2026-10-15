#!/usr/bin/env python3
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

"""Build the mutually-exclusive legacy-residual and all-source ge200 indices.

The set operation is performed on source-qualified protein identities::

    residual = (legacy Short union legacy Long) - ge200[AFQ or MGN]
    target   = residual union ge200[AFQ, AFU, MGN, ESM]

The two published children use the existing runtime-index schema and add
optional, row-aligned explicit A3M/PKL paths plus global-ID/provenance arrays.
This is required for AFU ``.cif.gz`` and ESM's materialized ``.pdb -> .cif``
training representation; deriving sibling files from the structure suffix is
not reliable for the four-source catalog.

The production defaults are intentionally pinned to the 2026-08-04 inputs and
their expected cardinalities.  Tests may pass a smaller ``ExpectedCounts``.
No source asset is modified.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import mmap
import os
import shutil
import struct
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import IO, Any, Iterable, Mapping, Sequence

LEGACY_SHORT_RUNTIME = Path(
    "/mnt/project_abd_hs/klp/databases/"
    "afdb_mgnify_distillation_short_le200_runtime_index_20260725"
)
LEGACY_LONG_RUNTIME = Path(
    "/mnt/project_abd_hs/klp/databases/"
    "afdb_mgnify_distillation_long_gt200_runtime_index_20260725"
)
GE200_ROOT = Path(
    "/mnt/project_abd_hs/klp/databases/"
    "foldseek_all_sources_ge200_tm0.7_cov0.8_20260728"
)
GE200_NUMERIC_GLOBAL = (
    GE200_ROOT
    / "result_a3m_pkl_filtered_20260804/numeric_global_export_20260804/"
    "all_ge200_tm070_cov080_a3m_pkl_filtered_numeric_global.tsv.gz"
)
GE200_MAPPING = GE200_ROOT / "manifest/all_ge200.mapping.tsv.gz"
GE200_AMBIGUITY = (
    GE200_ROOT
    / "result_a3m_pkl_filtered_20260804/numeric_global_export_20260804/"
    "cross_group_duplicate_member_resolution.tsv.gz"
)
ESM_MSA_ONLY_INDEX = (
    GE200_ROOT
    / "esm_msa_ge384_current_final_20260803/indices/"
    "esm_bioassembly_indices.msa_only.csv.gz"
)
AFU_BIOASSEMBLY_INDEX = (
    GE200_ROOT
    / "afu_bioassembly_plan_v4_3x136_20260801/"
    "afu_bioassembly_indices.csv.gz"
)
DEFAULT_RESIDUAL_OUTPUT = Path(
    "/mnt/project_abd_hs/klp/databases/"
    "afdb_mgnify_legacy_residual_not_ge200_runtime_index_20260804"
)
DEFAULT_GE200_OUTPUT = Path(
    "/mnt/project_abd_hs/klp/databases/"
    "all_sources_ge200_structural_runtime_index_20260804"
)

LOOKUP_COLUMNS = [
    "runtime_row_id",
    "global_id",
    "cluster_ordinal",
    "stable_cluster_id",
    "source_member_row_id",
    "source_partition",
    "cluster_representative_length_bucket",
]
AMBIGUITY_COLUMNS = [
    "global_id",
    "canonical_cluster_ordinal",
    "alternate_cluster_ordinal",
    "raw_id",
    "resolution_rule",
]

ROOTS = {
    "0": {
        "path": "/mnt/project_abd_hs/klp/databases/afdb/msas",
        "source_db": "afdb",
        "source_prefix": "AFQ|",
    },
    "1": {
        "path": "/mnt/project_abd_hs/klp/databases/afdb/models_unique_20260724",
        "source_db": "afdb",
        "source_prefix": "AFQ|",
    },
    "2": {
        "path": "/mnt/project_abd_hs/biran/data/mgnify/msas",
        "source_db": "mgnify",
        "source_prefix": "MGN|",
    },
    "3": {
        "path": "/mnt/project_abd_hs/klp/databases/esm/"
        "afdbv6_unmatched_query_mgnify_ge200_lt256_cif_gz",
        "source_db": "afdb_v6_unmatched",
        "source_prefix": "AFU|",
    },
    "4": {
        "path": "/mnt/project_abd_hs/klp/databases/esm/"
        "afdbv6_unmatched_query_mgnify_ge256_cif_gz",
        "source_db": "afdb_v6_unmatched",
        "source_prefix": "AFU|",
    },
    "5": {
        "path": "/mnt/project_abd_hs/klp/databases/esm/pdbs",
        "source_db": "esm_atlas",
        "source_prefix": "ESM|",
    },
}
SOURCE_CODES = {"AFQ": 0, "AFU": 1, "MGN": 2, "ESM": 3}
CODE_SOURCES = {value: key for key, value in SOURCE_CODES.items()}
EXPECTED_NUMERIC_GLOBAL_SHA256 = (
    "f9923ef3f52fac700f9cec89dd52d8efb14f1572bc4f591dd3c923b47f3e0f15"
)


class CatalogBuildError(RuntimeError):
    """A source or target catalog invariant was violated."""


@dataclass(frozen=True)
class ExpectedCounts:
    residual_rows: int
    ge200_rows: int
    target_rows: int
    ge200_clusters: int
    ge200_afq_mgn_rows: int | None = None
    residual_short_rows: int | None = None
    residual_long_rows: int | None = None


PRODUCTION_COUNTS = ExpectedCounts(
    residual_rows=15_323_139,
    ge200_rows=25_417_250,
    target_rows=40_740_389,
    ge200_clusters=9_050_355,
    ge200_afq_mgn_rows=18_175_512,
    residual_short_rows=15_313_927,
    residual_long_rows=9_212,
)


@dataclass(frozen=True)
class RowRecord:
    root_id: str
    cif_relpath: str
    a3m_relpath: str
    bioassembly_relpath: str
    global_id: str
    source_member_row_id: int
    source_partition: str
    cluster_representative_length_bucket: str


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(document: object) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _json_bytes(document: object) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def _safe_relative(value: str, *, kind: str) -> str:
    if not value or "\x00" in value or "\\" in value:
        raise CatalogBuildError(f"Unsafe {kind} path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise CatalogBuildError(f"Unsafe {kind} path: {value!r}")
    lower = path.name.lower()
    valid = {
        "cif": lower.endswith(".cif") or lower.endswith(".cif.gz"),
        "a3m": lower.endswith(".a3m"),
        "pkl": lower.endswith(".pkl.gz"),
    }[kind]
    if not valid:
        raise CatalogBuildError(f"Invalid {kind} suffix: {value!r}")
    return path.as_posix()


def _strip_structure_suffix(value: str) -> str:
    name = PurePosixPath(value).name
    lower = name.lower()
    for suffix in (".cif.gz", ".cif", ".pdb.gz", ".pdb"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    raise CatalogBuildError(f"Unsupported structure suffix: {value!r}")


def _load_runtime_index_class():
    """Load the lightweight reader without importing pipeline/__init__.py.

    ``protenix.data.pipeline.__init__`` imports the full training stack.  A
    catalog build and even ``--help`` must not require torch/biotite, so the
    standalone stdlib+NumPy reader is loaded directly from its source file.
    """

    module_name = "_protenix_distillation_runtime_index_standalone"
    module = sys.modules.get(module_name)
    if module is None:
        path = Path(__file__).resolve().parents[2] / (
            "protenix/data/pipeline/distillation_runtime_index.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load runtime-index reader: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.DistillationRuntimeIndex


def _explicit_paths(source: str, relative_structure: str) -> tuple[str, str, str]:
    """Translate the clustering structure path into training artifacts."""

    relative = PurePosixPath(relative_structure)
    raw_id = _strip_structure_suffix(relative.as_posix())
    if source == "ESM":
        cif = relative.parent / f"{raw_id}.cif"
    else:
        cif = relative
    a3m = cif.parent / f"{raw_id}.a3m"
    pkl = cif.parent / "bioassembly" / f"{raw_id}.pkl.gz"
    return (
        _safe_relative(cif.as_posix(), kind="cif"),
        _safe_relative(a3m.as_posix(), kind="a3m"),
        _safe_relative(pkl.as_posix(), kind="pkl"),
    )


def _load_esm_explicit_paths(path: Path) -> dict[str, tuple[str, str, str, str]]:
    """Load the authoritative ESM A3M+PKL training projection.

    The Foldseek mapping points at ``.pdb`` files.  It is intentionally not
    used to infer ESM training paths: the msa-only audit is the eligibility
    authority and explicitly names the materialized CIF, A3M and PKL.
    """

    values: dict[str, tuple[str, str, str, str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "pdb_id",
            "source_root_id",
            "cif_relpath",
            "msa_relpath",
            "bioassembly_relpath",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise CatalogBuildError("ESM msa-only index has an incompatible header")
        for record in reader:
            raw_id = record["pdb_id"]
            if not raw_id or raw_id in values:
                raise CatalogBuildError(f"duplicate/empty ESM msa-only ID: {raw_id!r}")
            root_id = record["source_root_id"]
            if root_id != "5":
                raise CatalogBuildError(f"ESM msa-only row uses root {root_id!r}")
            cif = _safe_relative(record["cif_relpath"], kind="cif")
            a3m = _safe_relative(record["msa_relpath"], kind="a3m")
            pkl = _safe_relative(record["bioassembly_relpath"], kind="pkl")
            if not (
                _strip_structure_suffix(cif) == raw_id
                and PurePosixPath(a3m).stem == raw_id
                and PurePosixPath(pkl).name == f"{raw_id}.pkl.gz"
            ):
                raise CatalogBuildError(f"ESM explicit paths disagree for {raw_id}")
            values[raw_id] = (root_id, cif, a3m, pkl)
    if not values:
        raise CatalogBuildError("ESM msa-only index contains no rows")
    return values


def _load_afu_explicit_paths(path: Path) -> dict[str, tuple[str, str, str, str]]:
    """Load authoritative AFU CIF/PKL rows and derive only the blank A3M."""

    values: dict[str, tuple[str, str, str, str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "pdb_id",
            "source_root_id",
            "cif_relpath",
            "msa_relpath",
            "bioassembly_relpath",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise CatalogBuildError("AFU bioassembly index has an incompatible header")
        for record in reader:
            raw_id = record["pdb_id"]
            if not raw_id or raw_id in values:
                raise CatalogBuildError(f"duplicate/empty AFU audit ID: {raw_id!r}")
            root_id = record["source_root_id"]
            if root_id not in {"3", "4"}:
                raise CatalogBuildError(f"AFU audit row uses root {root_id!r}")
            cif = _safe_relative(record["cif_relpath"], kind="cif")
            pkl = _safe_relative(record["bioassembly_relpath"], kind="pkl")
            msa_value = record["msa_relpath"].strip()
            if msa_value:
                a3m = _safe_relative(msa_value, kind="a3m")
            else:
                cif_path = PurePosixPath(cif)
                # Remove the full two-part suffix. Path.with_suffix(".a3m")
                # would incorrectly produce ``protein.cif.a3m``.
                a3m = _safe_relative(
                    (cif_path.parent / f"{_strip_structure_suffix(cif)}.a3m").as_posix(),
                    kind="a3m",
                )
            if not (
                _strip_structure_suffix(cif) == raw_id
                and PurePosixPath(a3m).stem == raw_id
                and PurePosixPath(pkl).name == f"{raw_id}.pkl.gz"
            ):
                raise CatalogBuildError(f"AFU explicit paths disagree for {raw_id}")
            values[raw_id] = (root_id, cif, a3m, pkl)
    if not values:
        raise CatalogBuildError("AFU bioassembly index contains no rows")
    return values


class _HashedWriter:
    def __init__(self, path: Path):
        self.path = path
        self.stream = path.open("wb", buffering=8 * 1024 * 1024)
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, payload: bytes) -> None:
        self.stream.write(payload)
        self.digest.update(payload)
        self.size += len(payload)

    def close(self) -> None:
        self.stream.close()

    @property
    def sha256(self) -> str:
        return self.digest.hexdigest()


def _artifact(
    path: str,
    writer: _HashedWriter,
    *,
    dtype: str,
    count: int,
) -> dict[str, Any]:
    return {
        "path": path,
        "size_bytes": writer.size,
        "sha256": writer.sha256,
        "dtype": dtype,
        "endian": "not_applicable" if dtype == "bytes" else "little",
        "count": count,
    }


class _MappingLookup:
    """Temporary mmap lookup keyed by the source manifest's numeric row ID."""

    def __init__(self, mapping_tsv: Path, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        offsets_path = directory / "mapping_path_offsets.u64"
        payload_path = directory / "mapping_paths.bin"
        roots_path = directory / "mapping_root_ids.u8"
        sources_path = directory / "mapping_source_ids.u8"
        offsets = offsets_path.open("wb")
        payload = payload_path.open("wb")
        roots = roots_path.open("wb")
        sources = sources_path.open("wb")
        path_bytes = 0
        row_count = 0
        offsets.write(struct.pack("<Q", 0))
        try:
            with gzip.open(mapping_tsv, "rt", encoding="utf-8", newline="") as stream:
                reader = csv.reader(stream, delimiter="\t")
                for line_number, fields in enumerate(reader, 1):
                    if len(fields) != 5:
                        raise CatalogBuildError(
                            f"mapping line {line_number} has {len(fields)} columns"
                        )
                    source_row, source, raw_id, _length, rooted_path = fields
                    if int(source_row) != row_count:
                        raise CatalogBuildError(
                            f"mapping row IDs are not contiguous at line {line_number}"
                        )
                    try:
                        source_code = SOURCE_CODES[source]
                    except KeyError as exc:
                        raise CatalogBuildError(f"Unknown source {source!r}") from exc
                    root_id, separator, relative = rooted_path.partition("/")
                    if not separator or root_id not in ROOTS:
                        raise CatalogBuildError(
                            f"Invalid rooted structure path {rooted_path!r}"
                        )
                    expected_prefix = ROOTS[root_id]["source_prefix"]
                    if expected_prefix != f"{source}|":
                        raise CatalogBuildError(
                            f"source/root mismatch at mapping row {row_count}: "
                            f"{source!r}/{root_id!r}"
                        )
                    if _strip_structure_suffix(relative) != raw_id:
                        raise CatalogBuildError(
                            f"mapping raw ID/path mismatch at row {row_count}"
                        )
                    encoded = relative.encode()
                    payload.write(encoded)
                    path_bytes += len(encoded)
                    offsets.write(struct.pack("<Q", path_bytes))
                    roots.write(bytes((int(root_id),)))
                    sources.write(bytes((source_code,)))
                    row_count += 1
        finally:
            offsets.close()
            payload.close()
            roots.close()
            sources.close()
        if row_count == 0:
            raise CatalogBuildError("ge200 mapping contains no rows")

        self.row_count = row_count
        self._files = [
            offsets_path.open("rb"),
            payload_path.open("rb"),
            roots_path.open("rb"),
            sources_path.open("rb"),
        ]
        self._offsets, self._payload, self._roots, self._sources = [
            mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            for stream in self._files
        ]

    def get(self, row_id: int) -> tuple[str, str, str]:
        if not 0 <= row_id < self.row_count:
            raise CatalogBuildError(f"member_row_id outside mapping: {row_id}")
        start = struct.unpack_from("<Q", self._offsets, row_id * 8)[0]
        end = struct.unpack_from("<Q", self._offsets, (row_id + 1) * 8)[0]
        try:
            relative = self._payload[start:end].decode()
        except UnicodeDecodeError as exc:
            raise CatalogBuildError(f"Invalid mapping UTF-8 at row {row_id}") from exc
        source = CODE_SOURCES[self._sources[row_id]]
        return source, str(self._roots[row_id]), relative

    def close(self) -> None:
        for mapped in (self._offsets, self._payload, self._roots, self._sources):
            mapped.close()
        for stream in self._files:
            stream.close()


class _RuntimeWriter:
    def __init__(self, directory: Path, roots_document: Mapping[str, Any]):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "sampling").mkdir()
        (directory / "lookup").mkdir()
        root_records = roots_document.get("roots")
        if not isinstance(root_records, Sequence) or not root_records:
            raise CatalogBuildError("roots document must contain records")
        self.root_ids = [str(record["root_id"]) for record in root_records]
        self.root_ordinals = {value: index for index, value in enumerate(self.root_ids)}
        if len(self.root_ids) != len(self.root_ordinals) or len(self.root_ids) > 256:
            raise CatalogBuildError("root IDs must be unique and fit uint8")
        roots_payload = _json_bytes(roots_document)
        (directory / "roots.json").write_bytes(roots_payload)
        self.roots_artifact = {
            "path": "roots.json",
            "size_bytes": len(roots_payload),
            "sha256": hashlib.sha256(roots_payload).hexdigest(),
        }

        names = {
            "row_offsets": "row_offsets.u64",
            "source_root_ids": "source_root_ids.u8",
            "cif_relpaths": "cif_relpaths.bin",
            "a3m_row_offsets": "a3m_row_offsets.u64",
            "a3m_relpaths": "a3m_relpaths.bin",
            "bioassembly_row_offsets": "bioassembly_row_offsets.u64",
            "bioassembly_relpaths": "bioassembly_relpaths.bin",
            "global_id_offsets": "global_id_offsets.u64",
            "global_ids": "global_ids.bin",
            "source_member_row_ids": "source_member_row_ids.u64",
            "cluster_ids": "sampling/cluster_ids.u32",
            "first_row_ids": "sampling/first_row_ids.u64",
            "cluster_sizes": "sampling/cluster_sizes.u32",
        }
        self.writers = {
            name: _HashedWriter(directory / relative) for name, relative in names.items()
        }
        self.names = names
        for name in (
            "row_offsets",
            "a3m_row_offsets",
            "bioassembly_row_offsets",
            "global_id_offsets",
        ):
            self.writers[name].write(struct.pack("<Q", 0))
        self.byte_counts = {"cif": 0, "a3m": 0, "pkl": 0, "global": 0}
        self.row_count = 0
        self.cluster_count = 0
        self._cluster_first = 0
        self._cluster_stable_id: int | None = None
        self.source_counts: Counter[str] = Counter()
        self.partition_counts: Counter[str] = Counter()
        self.lookup_path = directory / "lookup/member_to_cluster.tsv.gz"
        self.lookup_stream = gzip.open(
            self.lookup_path, "wt", encoding="utf-8", newline="", compresslevel=1
        )
        self.lookup = csv.writer(self.lookup_stream, delimiter="\t", lineterminator="\n")
        self.lookup.writerow(LOOKUP_COLUMNS)

    def begin_cluster(self, stable_cluster_id: int) -> None:
        if self._cluster_stable_id is not None:
            raise RuntimeError("previous cluster has not ended")
        if not 0 <= int(stable_cluster_id) <= (1 << 32) - 1:
            raise CatalogBuildError(f"cluster ID does not fit uint32: {stable_cluster_id}")
        self._cluster_stable_id = int(stable_cluster_id)
        self._cluster_first = self.row_count

    def add_row(self, row: RowRecord) -> None:
        if self._cluster_stable_id is None:
            raise RuntimeError("begin_cluster must precede add_row")
        try:
            root_ordinal = self.root_ordinals[row.root_id]
        except KeyError as exc:
            raise CatalogBuildError(f"Unknown output root ID {row.root_id!r}") from exc
        cif = _safe_relative(row.cif_relpath, kind="cif").encode()
        a3m = _safe_relative(row.a3m_relpath, kind="a3m").encode()
        pkl = _safe_relative(row.bioassembly_relpath, kind="pkl").encode()
        global_id = row.global_id.encode()
        if (
            "|" not in row.global_id
            or not global_id
            or any(char in row.global_id for char in "\t\r\n")
        ):
            raise CatalogBuildError(f"Invalid global ID: {row.global_id!r}")
        if row.source_member_row_id < 0:
            raise CatalogBuildError("source member row ID must be non-negative")

        self.writers["source_root_ids"].write(bytes((root_ordinal,)))
        for payload, data_name, offsets_name, byte_key in (
            (cif, "cif_relpaths", "row_offsets", "cif"),
            (a3m, "a3m_relpaths", "a3m_row_offsets", "a3m"),
            (pkl, "bioassembly_relpaths", "bioassembly_row_offsets", "pkl"),
            (global_id, "global_ids", "global_id_offsets", "global"),
        ):
            self.writers[data_name].write(payload)
            self.byte_counts[byte_key] += len(payload)
            self.writers[offsets_name].write(
                struct.pack("<Q", self.byte_counts[byte_key])
            )
        self.writers["source_member_row_ids"].write(
            struct.pack("<Q", row.source_member_row_id)
        )
        self.lookup.writerow(
            [
                self.row_count,
                row.global_id,
                self.cluster_count,
                self._cluster_stable_id,
                row.source_member_row_id,
                row.source_partition,
                row.cluster_representative_length_bucket,
            ]
        )
        self.source_counts[row.global_id.split("|", 1)[0]] += 1
        self.partition_counts[row.source_partition] += 1
        self.row_count += 1

    def end_cluster(self) -> None:
        if self._cluster_stable_id is None:
            raise RuntimeError("no active cluster")
        size = self.row_count - self._cluster_first
        if size <= 0:
            raise CatalogBuildError("cannot publish an empty cluster")
        self.writers["cluster_ids"].write(struct.pack("<I", self._cluster_stable_id))
        self.writers["first_row_ids"].write(struct.pack("<Q", self._cluster_first))
        self.writers["cluster_sizes"].write(struct.pack("<I", size))
        self.cluster_count += 1
        self._cluster_stable_id = None

    def finish(self) -> dict[str, Any]:
        if self._cluster_stable_id is not None:
            raise RuntimeError("active cluster must be ended before finish")
        self.lookup_stream.close()
        for writer in self.writers.values():
            writer.close()
        if self.row_count <= 0 or self.cluster_count <= 0:
            raise CatalogBuildError("runtime child must be non-empty")
        artifacts = {
            "row_offsets": _artifact(
                self.names["row_offsets"],
                self.writers["row_offsets"],
                dtype="uint64",
                count=self.row_count + 1,
            ),
            "source_root_ids": _artifact(
                self.names["source_root_ids"],
                self.writers["source_root_ids"],
                dtype="uint8",
                count=self.row_count,
            ),
            "cif_relpaths": _artifact(
                self.names["cif_relpaths"],
                self.writers["cif_relpaths"],
                dtype="bytes",
                count=self.byte_counts["cif"],
            ),
            "a3m_row_offsets": _artifact(
                self.names["a3m_row_offsets"],
                self.writers["a3m_row_offsets"],
                dtype="uint64",
                count=self.row_count + 1,
            ),
            "a3m_relpaths": _artifact(
                self.names["a3m_relpaths"],
                self.writers["a3m_relpaths"],
                dtype="bytes",
                count=self.byte_counts["a3m"],
            ),
            "bioassembly_row_offsets": _artifact(
                self.names["bioassembly_row_offsets"],
                self.writers["bioassembly_row_offsets"],
                dtype="uint64",
                count=self.row_count + 1,
            ),
            "bioassembly_relpaths": _artifact(
                self.names["bioassembly_relpaths"],
                self.writers["bioassembly_relpaths"],
                dtype="bytes",
                count=self.byte_counts["pkl"],
            ),
            "global_id_offsets": _artifact(
                self.names["global_id_offsets"],
                self.writers["global_id_offsets"],
                dtype="uint64",
                count=self.row_count + 1,
            ),
            "global_ids": _artifact(
                self.names["global_ids"],
                self.writers["global_ids"],
                dtype="bytes",
                count=self.byte_counts["global"],
            ),
            "source_member_row_ids": _artifact(
                self.names["source_member_row_ids"],
                self.writers["source_member_row_ids"],
                dtype="uint64",
                count=self.row_count,
            ),
            "roots": self.roots_artifact,
        }
        sampling = {
            "alignment": (
                "Array index i identifies one cluster. Members occupy "
                "[first_row_ids[i], first_row_ids[i] + cluster_sizes[i])."
            ),
            "arrays": {
                "cluster_ids": _artifact(
                    self.names["cluster_ids"],
                    self.writers["cluster_ids"],
                    dtype="uint32",
                    count=self.cluster_count,
                ),
                "first_row_ids": _artifact(
                    self.names["first_row_ids"],
                    self.writers["first_row_ids"],
                    dtype="uint64",
                    count=self.cluster_count,
                ),
                "cluster_sizes": _artifact(
                    self.names["cluster_sizes"],
                    self.writers["cluster_sizes"],
                    dtype="uint32",
                    count=self.cluster_count,
                ),
            },
        }
        lookup_sha = _sha256_file(self.lookup_path)
        artifacts["member_to_cluster_lookup"] = {
            "path": "lookup/member_to_cluster.tsv.gz",
            "size_bytes": self.lookup_path.stat().st_size,
            "sha256": lookup_sha,
            "count": self.row_count,
            "dtype": "tsv_gzip",
            "columns": LOOKUP_COLUMNS,
        }
        return {
            "row_count": self.row_count,
            "cluster_count": self.cluster_count,
            "root_ids": self.root_ids,
            "artifacts": artifacts,
            "sampling": sampling,
            "source_counts": dict(sorted(self.source_counts.items())),
            "partition_counts": dict(sorted(self.partition_counts.items())),
        }


def _roots_document(root_ids: Iterable[str]) -> dict[str, Any]:
    records = []
    for root_id in root_ids:
        record = ROOTS[str(root_id)]
        records.append({"root_id": str(root_id), **record})
    return {
        "schema_version": 1,
        "path_mode": "explicit_row_aligned_cif_a3m_bioassembly_v1",
        "roots": records,
    }


def _legacy_prefixes(runtime: Any) -> dict[str, str]:
    document = json.loads(runtime.roots_path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for record in document["roots"]:
        root_id = str(record["root_id"])
        prefix = str(record.get("source_prefix", ""))
        if not prefix.endswith("|"):
            raise CatalogBuildError(f"Legacy root {root_id} lacks source_prefix")
        result[root_id] = prefix
    return result


def _legacy_row(
    runtime: Any,
    row_id: int,
    partition: str,
    prefixes: Mapping[str, str],
) -> RowRecord:
    paths = runtime.resolve_paths(row_id)
    global_id = f"{prefixes[paths.root_id]}{_strip_structure_suffix(paths.cif_relpath)}"
    root = runtime.source_roots[paths.root_id]
    return RowRecord(
        root_id=paths.root_id,
        cif_relpath=paths.cif_path.relative_to(root).as_posix(),
        a3m_relpath=paths.a3m_path.relative_to(root).as_posix(),
        bioassembly_relpath=paths.bioassembly_path.relative_to(root).as_posix(),
        global_id=global_id,
        source_member_row_id=row_id,
        source_partition=partition,
        cluster_representative_length_bucket=(
            "source_cluster_representative_le200"
            if partition == "short_le200"
            else "source_cluster_representative_gt200"
        ),
    )


def _read_ambiguity_groups(path: Path) -> dict[str, dict[str, tuple[int, str]]]:
    groups: dict[str, dict[str, tuple[int, str]]] = defaultdict(dict)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {
            "raw_id",
            "cluster_run_index",
            "assigned_member_global_id",
            "resolution_rule",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise CatalogBuildError("ambiguity input has an incompatible header")
        for record in reader:
            raw_id = record["raw_id"]
            global_id = record["assigned_member_global_id"]
            source = global_id.split("|", 1)[0]
            if source in groups[raw_id]:
                raise CatalogBuildError(f"duplicate ambiguity source for {raw_id}")
            groups[raw_id][source] = (
                int(record["cluster_run_index"]),
                record["resolution_rule"],
            )
    for raw_id, records in groups.items():
        if set(records) != {"AFQ", "AFU"}:
            raise CatalogBuildError(
                f"cross-group ambiguity must pair AFQ/AFU for {raw_id}: {records}"
            )
    return groups


def _write_ambiguity_artifact(
    writer: _RuntimeWriter,
    groups: Mapping[str, Mapping[str, tuple[int, str]]],
) -> dict[str, Any]:
    path = writer.directory / "lookup/ambiguity_alternates.tsv.gz"
    count = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=1) as stream:
        output = csv.writer(stream, delimiter="\t", lineterminator="\n")
        output.writerow(AMBIGUITY_COLUMNS)
        for raw_id in sorted(groups):
            canonical, source_rule = groups[raw_id]["AFQ"]
            alternate, _ = groups[raw_id]["AFU"]
            if canonical == alternate:
                raise CatalogBuildError(f"ambiguity clusters are equal for {raw_id}")
            if canonical >= writer.cluster_count or alternate >= writer.cluster_count:
                raise CatalogBuildError(f"ambiguity cluster outside ge200 runtime for {raw_id}")
            output.writerow(
                [
                    f"AFQ|{raw_id}",
                    canonical,
                    alternate,
                    raw_id,
                    "seen_floor_1_v1:" + source_rule,
                ]
            )
            count += 1
    return {
        "path": "lookup/ambiguity_alternates.tsv.gz",
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "count": count,
        "dtype": "tsv_gzip",
        "columns": AMBIGUITY_COLUMNS,
    }


def _manifest(
    *,
    built: Mapping[str, Any],
    role: str,
    pair_id: str,
    pair_receipt: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    counts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "format": "protenix_runtime_distillation_index",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "row_count": built["row_count"],
        "cluster_count": built["cluster_count"],
        "root_ids": built["root_ids"],
        "artifacts": built["artifacts"],
        "sampling": built["sampling"],
        "path_rules": {
            "mode": "explicit_row_aligned_v1",
            "cif": "source root / cif_relpaths[row]",
            "a3m": "source root / a3m_relpaths[row]",
            "bioassembly": "source root / bioassembly_relpaths[row]",
        },
        "target_catalog": {
            "schema_version": 1,
            "role": role,
            "pair_id": pair_id,
            "pair_receipt": pair_receipt,
            "set_operation": (
                "legacy_short_union_long_minus_ge200_afq_mgn"
                if role == "legacy_residual"
                else "all_source_ge200_filtered_a3m_pkl"
            ),
            "counts": dict(counts),
            "source_identity": dict(source_identity),
        },
    }


def _write_completion(directory: Path, manifest: Mapping[str, Any]) -> None:
    manifest_path = directory / "manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    manifest_sha = _sha256_file(manifest_path)
    success = {
        "schema_version": 1,
        "status": "complete",
        "manifest_sha256": manifest_sha,
        "row_count": manifest["row_count"],
        "cluster_count": manifest["cluster_count"],
    }
    (directory / "_SUCCESS.json").write_bytes(_json_bytes(success))
    checksum_lines = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_lines.append(
                f"{_sha256_file(path)}  {path.relative_to(directory).as_posix()}\n"
            )
    (directory / "checksums.sha256").write_text(
        "".join(checksum_lines), encoding="utf-8"
    )


def _prepare_stage(output: Path, overwrite: bool) -> Path:
    output = output.resolve(strict=False)
    if os.path.lexists(output) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    stage.mkdir()
    return stage


def _publish_pair(stages: Sequence[Path], outputs: Sequence[Path], overwrite: bool) -> None:
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for output in outputs:
            if os.path.lexists(output):
                if not overwrite or not output.is_dir() or output.is_symlink():
                    raise FileExistsError(f"Cannot replace output {output}")
                backup = output.parent / f".{output.name}.old.{uuid.uuid4().hex}"
                os.replace(output, backup)
                backups.append((output, backup))
        for stage, output in zip(stages, outputs):
            os.replace(stage, output)
            published.append(output)
    except BaseException:
        for output in reversed(published):
            shutil.rmtree(output, ignore_errors=True)
        for output, backup in reversed(backups):
            os.replace(backup, output)
        raise
    for _output, backup in backups:
        shutil.rmtree(backup, ignore_errors=True)


def build_ge200_residual_runtime_indices(
    *,
    legacy_short_runtime_dir: str | Path,
    legacy_long_runtime_dir: str | Path,
    ge200_numeric_global_tsv: str | Path,
    ge200_mapping_tsv: str | Path,
    ge200_ambiguity_tsv: str | Path,
    esm_msa_only_index: str | Path,
    afu_bioassembly_index: str | Path,
    residual_output_dir: str | Path,
    ge200_output_dir: str | Path,
    expected_counts: ExpectedCounts = PRODUCTION_COUNTS,
    expected_numeric_global_sha256: str | None = None,
    overwrite: bool = False,
    progress_every_rows: int = 1_000_000,
) -> dict[str, Any]:
    """Build and atomically publish both target catalog children."""

    started = time.monotonic()
    short_path = Path(legacy_short_runtime_dir).resolve(strict=True)
    long_path = Path(legacy_long_runtime_dir).resolve(strict=True)
    numeric_path = Path(ge200_numeric_global_tsv).resolve(strict=True)
    mapping_path = Path(ge200_mapping_tsv).resolve(strict=True)
    ambiguity_path = Path(ge200_ambiguity_tsv).resolve(strict=True)
    esm_index_path = Path(esm_msa_only_index).resolve(strict=True)
    afu_index_path = Path(afu_bioassembly_index).resolve(strict=True)
    residual_output = Path(residual_output_dir).resolve(strict=False)
    ge200_output = Path(ge200_output_dir).resolve(strict=False)
    if residual_output == ge200_output:
        raise ValueError("residual and ge200 outputs must differ")
    numeric_sha = _sha256_file(numeric_path)
    if expected_numeric_global_sha256 and numeric_sha != expected_numeric_global_sha256:
        raise CatalogBuildError(
            f"numeric/global SHA-256 differs: {numeric_sha} != "
            f"{expected_numeric_global_sha256}"
        )
    input_identity = {
        "legacy_short_manifest_sha256": _sha256_file(short_path / "manifest.json"),
        "legacy_long_manifest_sha256": _sha256_file(long_path / "manifest.json"),
        "ge200_numeric_global_sha256": numeric_sha,
        "ge200_mapping_sha256": _sha256_file(mapping_path),
        "ge200_ambiguity_sha256": _sha256_file(ambiguity_path),
        "esm_msa_only_index_sha256": _sha256_file(esm_index_path),
        "afu_bioassembly_index_sha256": _sha256_file(afu_index_path),
    }
    residual_stage = _prepare_stage(residual_output, overwrite)
    ge200_stage = _prepare_stage(ge200_output, overwrite)
    lookup_tmp = ge200_stage / ".mapping_lookup"
    short = long = None
    try:
        mapping = _MappingLookup(mapping_path, lookup_tmp)
        ambiguity_groups = _read_ambiguity_groups(ambiguity_path)
        esm_explicit_paths = _load_esm_explicit_paths(esm_index_path)
        esm_rows_used: set[str] = set()
        afu_explicit_paths = _load_afu_explicit_paths(afu_index_path)
        afu_rows_used: set[str] = set()
        ge200_writer = _RuntimeWriter(ge200_stage, _roots_document(ROOTS))
        ge200_legacy_ids: set[str] = set()
        ge200_seen_ids: set[str] = set()
        current_representative: int | None = None
        current_representative_global: str | None = None
        ge200_rows = 0
        try:
            with gzip.open(numeric_path, "rt", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream, delimiter="\t")
                expected_header = [
                    "representative_row_id",
                    "member_row_id",
                    "representative_global_id",
                    "member_global_id",
                ]
                if reader.fieldnames != expected_header:
                    raise CatalogBuildError("numeric/global input header differs")
                for record in reader:
                    representative = int(record["representative_row_id"])
                    member_row = int(record["member_row_id"])
                    representative_global = record["representative_global_id"]
                    member_global = record["member_global_id"]
                    if representative != current_representative:
                        if current_representative is not None:
                            ge200_writer.end_cluster()
                        current_representative = representative
                        current_representative_global = representative_global
                        if representative != member_row or representative_global != member_global:
                            raise CatalogBuildError(
                                "every numeric/global cluster must begin with its self row"
                            )
                        ge200_writer.begin_cluster(representative)
                    elif representative_global != current_representative_global:
                        raise CatalogBuildError("representative global ID changed within cluster")
                    source, root_id, structure_relative = mapping.get(member_row)
                    raw_id = _strip_structure_suffix(structure_relative)
                    if member_global != f"{source}|{raw_id}":
                        raise CatalogBuildError(
                            f"numeric/global mapping mismatch for member row {member_row}"
                        )
                    if member_global in ge200_seen_ids:
                        raise CatalogBuildError(
                            f"duplicate source-qualified ge200 identity: {member_global}"
                        )
                    ge200_seen_ids.add(member_global)
                    if source in {"AFQ", "MGN"}:
                        ge200_legacy_ids.add(member_global)
                    if source == "ESM":
                        try:
                            explicit_root, cif, a3m, pkl = esm_explicit_paths[raw_id]
                        except KeyError as exc:
                            raise CatalogBuildError(
                                f"eligible ESM member is absent from msa-only index: {raw_id}"
                            ) from exc
                        if explicit_root != root_id:
                            raise CatalogBuildError(
                                f"ESM mapping/audit root mismatch for {raw_id}"
                            )
                        esm_rows_used.add(raw_id)
                    elif source == "AFU":
                        try:
                            explicit_root, cif, a3m, pkl = afu_explicit_paths[raw_id]
                        except KeyError as exc:
                            raise CatalogBuildError(
                                f"eligible AFU member is absent from audit index: {raw_id}"
                            ) from exc
                        if explicit_root != root_id:
                            raise CatalogBuildError(
                                f"AFU mapping/audit root mismatch for {raw_id}"
                            )
                        # The mapping and AFU audit must identify exactly the
                        # same physical structure, not merely the same ID.
                        if cif != _safe_relative(structure_relative, kind="cif"):
                            raise CatalogBuildError(
                                f"AFU mapping/audit CIF mismatch for {raw_id}"
                            )
                        afu_rows_used.add(raw_id)
                    else:
                        cif, a3m, pkl = _explicit_paths(source, structure_relative)
                    ge200_writer.add_row(
                        RowRecord(
                            root_id=root_id,
                            cif_relpath=cif,
                            a3m_relpath=a3m,
                            bioassembly_relpath=pkl,
                            global_id=member_global,
                            source_member_row_id=member_row,
                            source_partition="ge200_structural",
                            cluster_representative_length_bucket="ge200_min_length_filter",
                        )
                    )
                    ge200_rows += 1
                    if progress_every_rows and ge200_rows % progress_every_rows == 0:
                        print(f"[ge200] rows={ge200_rows:,}", file=sys.stderr, flush=True)
            if current_representative is not None:
                ge200_writer.end_cluster()
            if esm_rows_used != set(esm_explicit_paths):
                missing = set(esm_explicit_paths) - esm_rows_used
                extra = esm_rows_used - set(esm_explicit_paths)
                raise CatalogBuildError(
                    "ge200 ESM membership differs from msa-only audit: "
                    f"missing={len(missing)} extra={len(extra)}"
                )
            if afu_rows_used != set(afu_explicit_paths):
                missing = set(afu_explicit_paths) - afu_rows_used
                extra = afu_rows_used - set(afu_explicit_paths)
                raise CatalogBuildError(
                    "ge200 AFU membership differs from bioassembly audit: "
                    f"missing={len(missing)} extra={len(extra)}"
                )
        finally:
            mapping.close()
            shutil.rmtree(lookup_tmp, ignore_errors=True)
        ge200_built = ge200_writer.finish()
        ge200_built["artifacts"]["ambiguity_alternates"] = _write_ambiguity_artifact(
            ge200_writer, ambiguity_groups
        )

        DistillationRuntimeIndex = _load_runtime_index_class()
        short = DistillationRuntimeIndex(short_path)
        long = DistillationRuntimeIndex(long_path)
        residual_root_ids = list(dict.fromkeys((*short.root_ids, *long.root_ids)))
        if any(root_id not in {"0", "1", "2"} for root_id in residual_root_ids):
            raise CatalogBuildError("legacy runtime contains unexpected source roots")
        residual_writer = _RuntimeWriter(
            residual_stage, _roots_document(residual_root_ids)
        )
        matched_legacy_ids: set[str] = set()
        legacy_rows = 0
        for partition, runtime in (
            ("short_le200", short),
            ("long_gt200", long),
        ):
            prefixes = _legacy_prefixes(runtime)
            for ordinal in range(runtime.cluster_count):
                stable_cluster_id = runtime.get_cluster_id(ordinal)
                size = runtime.get_cluster_size(ordinal)
                opened = False
                for member_offset in range(size):
                    source_row = runtime.get_member_row(ordinal, member_offset)
                    row = _legacy_row(
                        runtime, source_row, partition, prefixes
                    )
                    legacy_rows += 1
                    if row.global_id in ge200_legacy_ids:
                        if row.global_id in matched_legacy_ids:
                            raise CatalogBuildError(
                                "legacy Short/Long contains duplicate ge200 identity: "
                                f"{row.global_id}"
                            )
                        matched_legacy_ids.add(row.global_id)
                        continue
                    if not opened:
                        residual_writer.begin_cluster(stable_cluster_id)
                        opened = True
                    residual_writer.add_row(row)
                    if progress_every_rows and legacy_rows % progress_every_rows == 0:
                        print(
                            f"[legacy] rows={legacy_rows:,} residual={residual_writer.row_count:,}",
                            file=sys.stderr,
                            flush=True,
                        )
                if opened:
                    residual_writer.end_cluster()
        missing_legacy = ge200_legacy_ids - matched_legacy_ids
        if missing_legacy:
            example = sorted(missing_legacy)[:5]
            raise CatalogBuildError(
                f"{len(missing_legacy):,} ge200 AFQ/MGN identities are absent "
                f"from legacy Short/Long; examples={example}"
            )
        residual_built = residual_writer.finish()

        if ge200_built["row_count"] != expected_counts.ge200_rows:
            raise CatalogBuildError(
                f"ge200 rows differ: {ge200_built['row_count']} != "
                f"{expected_counts.ge200_rows}"
            )
        if ge200_built["cluster_count"] != expected_counts.ge200_clusters:
            raise CatalogBuildError(
                f"ge200 clusters differ: {ge200_built['cluster_count']} != "
                f"{expected_counts.ge200_clusters}"
            )
        if residual_built["row_count"] != expected_counts.residual_rows:
            raise CatalogBuildError(
                f"residual rows differ: {residual_built['row_count']} != "
                f"{expected_counts.residual_rows}"
            )
        for partition, expected in (
            ("short_le200", expected_counts.residual_short_rows),
            ("long_gt200", expected_counts.residual_long_rows),
        ):
            observed = int(residual_built["partition_counts"].get(partition, 0))
            if expected is not None and observed != expected:
                raise CatalogBuildError(
                    f"residual {partition} rows differ: {observed} != {expected}"
                )
        target_rows = residual_built["row_count"] + ge200_built["row_count"]
        if target_rows != expected_counts.target_rows:
            raise CatalogBuildError(
                f"target rows differ: {target_rows} != {expected_counts.target_rows}"
            )
        if (
            expected_counts.ge200_afq_mgn_rows is not None
            and len(ge200_legacy_ids) != expected_counts.ge200_afq_mgn_rows
        ):
            raise CatalogBuildError(
                f"ge200 AFQ/MGN rows differ: {len(ge200_legacy_ids)} != "
                f"{expected_counts.ge200_afq_mgn_rows}"
            )
        if legacy_rows != target_rows - (
            ge200_built["row_count"] - len(ge200_legacy_ids)
        ):
            raise CatalogBuildError("legacy/new set accounting identity failed")

        pair_basis = {
            "format": "protenix.target_distillation_catalog_pair.v1",
            "set_operation": (
                "(legacy_short_union_long-minus-ge200_afq_mgn)"
                "-union-all_source_ge200"
            ),
            "global_id_policy": "source_prefix_pipe_raw_id_v1",
            "counts": {
                "legacy_input_rows": legacy_rows,
                "ge200_afq_mgn_overlap_rows": len(ge200_legacy_ids),
                "residual_rows": residual_built["row_count"],
                "ge200_rows": ge200_built["row_count"],
                "target_rows": target_rows,
            },
            "inputs": input_identity,
            "children": {
                "legacy_residual": {
                    "row_count": residual_built["row_count"],
                    "cluster_count": residual_built["cluster_count"],
                    "lookup_sha256": residual_built["artifacts"]
                    ["member_to_cluster_lookup"]["sha256"],
                },
                "ge200_structural": {
                    "row_count": ge200_built["row_count"],
                    "cluster_count": ge200_built["cluster_count"],
                    "lookup_sha256": ge200_built["artifacts"]
                    ["member_to_cluster_lookup"]["sha256"],
                    "ambiguity_sha256": ge200_built["artifacts"]
                    ["ambiguity_alternates"]["sha256"],
                },
            },
        }
        pair_id = _canonical_sha256(pair_basis)
        pair_document = {**pair_basis, "pair_id": pair_id}
        pair_payload = _json_bytes(pair_document)
        pair_sha = hashlib.sha256(pair_payload).hexdigest()
        pair_receipt = {
            "path": "target_catalog_pair.json",
            "size_bytes": len(pair_payload),
            "sha256": pair_sha,
        }
        audit = {
            "schema_version": 1,
            "status": "complete",
            "pair_id": pair_id,
            "counts": pair_basis["counts"],
            "source_counts": {
                "residual": residual_built["source_counts"],
                "ge200": ge200_built["source_counts"],
            },
            "source_partition_counts": {
                "residual": residual_built["partition_counts"],
                "ge200": ge200_built["partition_counts"],
            },
            "ambiguity_alternate_records": ge200_built["artifacts"]
            ["ambiguity_alternates"]["count"],
            "invariants": {
                "residual_ge200_global_ids_disjoint": True,
                "every_ge200_afq_mgn_identity_removed_from_residual": True,
                "every_legacy_identity_retained_or_moved_exactly_once": True,
                "target_row_count_identity": True,
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        audit_payload = _json_bytes(audit)
        source_identity = {
            **input_identity,
            "global_id_policy": "source_prefix_pipe_raw_id_v1",
            "numeric_lift_ambiguity_policy": "canonical_plus_seen_floor_alternate_v1",
        }
        for stage in (residual_stage, ge200_stage):
            (stage / "target_catalog_pair.json").write_bytes(pair_payload)
            (stage / "build_audit.json").write_bytes(audit_payload)

        residual_counts = {
            "row_count": residual_built["row_count"],
            "cluster_count": residual_built["cluster_count"],
            "target_union_row_count": target_rows,
        }
        ge200_counts = {
            "row_count": ge200_built["row_count"],
            "cluster_count": ge200_built["cluster_count"],
            "afq_mgn_overlap_rows": len(ge200_legacy_ids),
            "target_union_row_count": target_rows,
        }
        residual_manifest = _manifest(
            built=residual_built,
            role="legacy_residual",
            pair_id=pair_id,
            pair_receipt=pair_receipt,
            source_identity=source_identity,
            counts=residual_counts,
        )
        ge200_manifest = _manifest(
            built=ge200_built,
            role="ge200_structural",
            pair_id=pair_id,
            pair_receipt=pair_receipt,
            source_identity=source_identity,
            counts=ge200_counts,
        )
        _write_completion(residual_stage, residual_manifest)
        _write_completion(ge200_stage, ge200_manifest)
        _publish_pair(
            (residual_stage, ge200_stage),
            (residual_output, ge200_output),
            overwrite,
        )
        return {
            "pair_id": pair_id,
            "residual_output": str(residual_output),
            "ge200_output": str(ge200_output),
            "residual": residual_counts,
            "ge200": ge200_counts,
            "target_rows": target_rows,
            "audit": audit,
        }
    except BaseException:
        shutil.rmtree(residual_stage, ignore_errors=True)
        shutil.rmtree(ge200_stage, ignore_errors=True)
        raise
    finally:
        if short is not None:
            short.close()
        if long is not None:
            long.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-short-runtime-dir", type=Path, default=LEGACY_SHORT_RUNTIME)
    parser.add_argument("--legacy-long-runtime-dir", type=Path, default=LEGACY_LONG_RUNTIME)
    parser.add_argument("--ge200-numeric-global-tsv", type=Path, default=GE200_NUMERIC_GLOBAL)
    parser.add_argument("--ge200-mapping-tsv", type=Path, default=GE200_MAPPING)
    parser.add_argument("--ge200-ambiguity-tsv", type=Path, default=GE200_AMBIGUITY)
    parser.add_argument("--esm-msa-only-index", type=Path, default=ESM_MSA_ONLY_INDEX)
    parser.add_argument(
        "--afu-bioassembly-index", type=Path, default=AFU_BIOASSEMBLY_INDEX
    )
    parser.add_argument("--residual-output-dir", type=Path, default=DEFAULT_RESIDUAL_OUTPUT)
    parser.add_argument("--ge200-output-dir", type=Path, default=DEFAULT_GE200_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every-rows", type=int, default=1_000_000)
    parser.add_argument(
        "--expected-numeric-global-sha256",
        default=EXPECTED_NUMERIC_GLOBAL_SHA256,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = build_ge200_residual_runtime_indices(
        legacy_short_runtime_dir=args.legacy_short_runtime_dir,
        legacy_long_runtime_dir=args.legacy_long_runtime_dir,
        ge200_numeric_global_tsv=args.ge200_numeric_global_tsv,
        ge200_mapping_tsv=args.ge200_mapping_tsv,
        ge200_ambiguity_tsv=args.ge200_ambiguity_tsv,
        esm_msa_only_index=args.esm_msa_only_index,
        afu_bioassembly_index=args.afu_bioassembly_index,
        residual_output_dir=args.residual_output_dir,
        ge200_output_dir=args.ge200_output_dir,
        expected_counts=PRODUCTION_COUNTS,
        expected_numeric_global_sha256=args.expected_numeric_global_sha256,
        overwrite=args.overwrite,
        progress_every_rows=args.progress_every_rows,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
