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

"""Partition one clustered distillation runtime index by representative length.

The short/long decision is made once per source cluster from the *final*
``representative_row_id`` in ``clusters.csv.gz``.  All members of that cluster
are copied to the same output.  Stable ``cluster_id`` values and member order
are retained, while row IDs and ``first_row_ids`` are compacted independently
inside each output runtime index.

The preferred production length input is the already-audited
``short_le200.paths.tsv`` plus its ``short_le200.stats.json``.  No CIF, A3M, or
PKL file is opened.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import mmap
import os
import shutil
import struct
import sys
import tempfile
import time
import uuid
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import IO, Iterator, Mapping, Sequence

try:
    from scripts.distillation.build_runtime_distillation_index import (
        verify_runtime_distillation_index,
    )
except ModuleNotFoundError:  # Direct ``python scripts/distillation/<file>.py``.
    from build_runtime_distillation_index import (
        verify_runtime_distillation_index,
    )


PARTITION_SCHEMA_VERSION = 1
PAIR_FORMAT = "protenix_distillation_length_partition_pair"
RUNTIME_FORMAT = "protenix_runtime_distillation_index"
RUNTIME_SCHEMA_VERSION = 1
DEFAULT_MAX_REPRESENTATIVE_LENGTH = 200
_COPY_CHUNK_SIZE = 8 * 1024 * 1024
_BUFFER_FLUSH_SIZE = 8 * 1024 * 1024
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_CLUSTER_HEADER = (
    b"cluster_id",
    b"source_representative_id",
    b"representative_id",
    b"original_size",
    b"cluster_size",
    b"removed_size",
    b"first_row_id",
    b"representative_row_id",
    b"flags",
)
_PARTITION_ROLES = ("short_le200", "long_gt200")


class PartitionBuildError(RuntimeError):
    """Raised when a source or output violates the partition contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=_COPY_CHUNK_SIZE) as stream:
        while payload := stream.read(_COPY_CHUNK_SIZE):
            digest.update(payload)
    return digest.hexdigest()


def _stat_record(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _same_stat(record: Mapping[str, object], path: Path) -> bool:
    stat = path.stat()
    return (
        int(record["size"]) == stat.st_size
        and int(record["mtime_ns"]) == stat.st_mtime_ns
    )


def _json_bytes(document: object) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _write_json(path: Path, document: object) -> dict[str, int | str]:
    payload = _json_bytes(document)
    path.write_bytes(payload)
    return {
        "path": path.name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _artifact_record(
    *,
    path: str,
    size_bytes: int,
    sha256: str,
    dtype: str,
    count: int,
) -> dict[str, int | str]:
    return {
        "path": path,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "dtype": dtype,
        "endian": "not_applicable" if dtype == "bytes" else "little",
        "count": count,
    }


def _normalize_new_directory(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate.parent.mkdir(parents=True, exist_ok=True)
    parent = candidate.parent.resolve(strict=True)
    if candidate.name in {"", ".", ".."}:
        raise ValueError(f"Invalid output directory: {path}")
    return parent / candidate.name


def _safe_relative_path(raw: str, *, suffix: str | None = None) -> PurePosixPath:
    if not raw or "\x00" in raw or "\\" in raw:
        raise PartitionBuildError(f"Unsafe relative path: {raw!r}")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or path.as_posix() != raw
        or ".." in path.parts
        or "." in path.parts
    ):
        raise PartitionBuildError(f"Unsafe relative path: {raw!r}")
    if suffix is not None and path.suffix.lower() != suffix:
        raise PartitionBuildError(
            f"Expected a {suffix} relative path, got {raw!r}"
        )
    return path


@contextmanager
def _pair_build_lock(
    short_destination: Path,
    long_destination: Path,
) -> Iterator[None]:
    identity = hashlib.sha256(
        (
            f"{short_destination}\0{long_destination}"
        ).encode("utf-8")
    ).hexdigest()[:24]
    lock_path = Path(tempfile.gettempdir()) / (
        f"protenix-runtime-partition-{identity}.lock"
    )
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PartitionBuildError(
                "Another builder is already targeting this output pair"
            ) from exc
        yield


class _ArtifactWriter:
    """Buffered binary writer with an on-write SHA-256 digest."""

    def __init__(self, path: Path):
        self.path = path
        self._stream: IO[bytes] | None = None
        self._buffer = bytearray()
        self._digest = hashlib.sha256()
        self.size_bytes = 0

    def __enter__(self) -> "_ArtifactWriter":
        self._stream = self.path.open("wb", buffering=_COPY_CHUNK_SIZE)
        return self

    def append(self, payload: bytes) -> None:
        self._buffer.extend(payload)
        if len(self._buffer) >= _BUFFER_FLUSH_SIZE:
            self.flush()

    def append_u32(self, value: int) -> None:
        self.append(struct.pack("<I", value))

    def append_u64(self, value: int) -> None:
        self.append(struct.pack("<Q", value))

    def flush(self) -> None:
        if not self._buffer:
            return
        if self._stream is None:
            raise RuntimeError("Artifact writer is not open")
        self._stream.write(self._buffer)
        self._digest.update(self._buffer)
        self.size_bytes += len(self._buffer)
        self._buffer.clear()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.flush()
        if self._stream is not None:
            self._stream.close()

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


class _RuntimeMaps:
    """Minimal read-only view over one validated runtime index."""

    def __init__(self, directory: Path, manifest: Mapping[str, object]):
        self.directory = directory
        self.manifest = manifest
        self.row_count = int(manifest["row_count"])
        self.cluster_count = int(manifest["cluster_count"])
        self.root_ids = tuple(str(value) for value in manifest["root_ids"])
        self._stack: ExitStack | None = None
        self._maps: dict[str, mmap.mmap] = {}

    @staticmethod
    def _record_path(
        directory: Path,
        record: object,
    ) -> Path:
        if not isinstance(record, dict):
            raise PartitionBuildError("Runtime manifest artifact is not an object")
        raw = str(record.get("path", ""))
        relative = _safe_relative_path(raw)
        return directory.joinpath(*relative.parts)

    def __enter__(self) -> "_RuntimeMaps":
        stack = ExitStack()
        self._stack = stack
        artifacts = self.manifest["artifacts"]
        sampling = self.manifest["sampling"]["arrays"]
        records = {
            "row_offsets": artifacts["row_offsets"],
            "source_root_ids": artifacts["source_root_ids"],
            "cif_relpaths": artifacts["cif_relpaths"],
            "cluster_ids": sampling["cluster_ids"],
            "first_row_ids": sampling["first_row_ids"],
            "cluster_sizes": sampling["cluster_sizes"],
        }
        for name, record in records.items():
            path = self._record_path(self.directory, record)
            stream = stack.enter_context(path.open("rb"))
            self._maps[name] = mmap.mmap(
                stream.fileno(),
                0,
                access=mmap.ACCESS_READ,
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for mapped in reversed(tuple(self._maps.values())):
            mapped.close()
        self._maps.clear()
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def row_offset(self, row_id: int) -> int:
        return struct.unpack_from(
            "<Q",
            self._maps["row_offsets"],
            row_id * 8,
        )[0]

    def root_ordinal(self, row_id: int) -> int:
        return self._maps["source_root_ids"][row_id]

    def relpath_bytes(self, row_id: int) -> bytes:
        start = self.row_offset(row_id)
        end = self.row_offset(row_id + 1)
        return self._maps["cif_relpaths"][start:end]

    def root_block(self, first_row: int, end_row: int) -> bytes:
        return self._maps["source_root_ids"][first_row:end_row]

    def relpath_block(self, first_row: int, end_row: int) -> bytes:
        start = self.row_offset(first_row)
        end = self.row_offset(end_row)
        return self._maps["cif_relpaths"][start:end]

    def cluster_id(self, ordinal: int) -> int:
        return struct.unpack_from(
            "<I",
            self._maps["cluster_ids"],
            ordinal * 4,
        )[0]

    def first_row_id(self, ordinal: int) -> int:
        return struct.unpack_from(
            "<Q",
            self._maps["first_row_ids"],
            ordinal * 8,
        )[0]

    def cluster_size(self, ordinal: int) -> int:
        return struct.unpack_from(
            "<I",
            self._maps["cluster_sizes"],
            ordinal * 4,
        )[0]


class _PartitionWriter:
    """Write one compact runtime partition while retaining source cluster IDs."""

    def __init__(self, directory: Path, role: str):
        self.directory = directory
        self.role = role
        self.row_count = 0
        self.cluster_count = 0
        self.relpath_bytes = 0
        self._stack: ExitStack | None = None
        self.offsets: _ArtifactWriter
        self.roots: _ArtifactWriter
        self.relpaths: _ArtifactWriter
        self.cluster_ids: _ArtifactWriter
        self.first_rows: _ArtifactWriter
        self.cluster_sizes: _ArtifactWriter

    def __enter__(self) -> "_PartitionWriter":
        self.directory.mkdir()
        (self.directory / "sampling").mkdir()
        stack = ExitStack()
        self._stack = stack
        self.offsets = stack.enter_context(
            _ArtifactWriter(self.directory / "row_offsets.u64")
        )
        self.roots = stack.enter_context(
            _ArtifactWriter(self.directory / "source_root_ids.u8")
        )
        self.relpaths = stack.enter_context(
            _ArtifactWriter(self.directory / "cif_relpaths.bin")
        )
        self.cluster_ids = stack.enter_context(
            _ArtifactWriter(self.directory / "sampling/cluster_ids.u32")
        )
        self.first_rows = stack.enter_context(
            _ArtifactWriter(self.directory / "sampling/first_row_ids.u64")
        )
        self.cluster_sizes = stack.enter_context(
            _ArtifactWriter(self.directory / "sampling/cluster_sizes.u32")
        )
        self.offsets.append_u64(0)
        return self

    def append_cluster(
        self,
        *,
        source: _RuntimeMaps,
        cluster_id: int,
        first_row: int,
        cluster_size: int,
    ) -> None:
        if not 0 <= cluster_id <= _UINT32_MAX:
            raise PartitionBuildError(
                f"cluster_id is outside uint32: {cluster_id}"
            )
        if not 0 < cluster_size <= _UINT32_MAX:
            raise PartitionBuildError(
                f"cluster_size is outside uint32: {cluster_size}"
            )
        end_row = first_row + cluster_size
        self.cluster_ids.append_u32(cluster_id)
        self.first_rows.append_u64(self.row_count)
        self.cluster_sizes.append_u32(cluster_size)

        self.roots.append(source.root_block(first_row, end_row))
        self.relpaths.append(source.relpath_block(first_row, end_row))
        for source_row in range(first_row, end_row):
            path_size = (
                source.row_offset(source_row + 1)
                - source.row_offset(source_row)
            )
            if path_size <= 0:
                raise PartitionBuildError(
                    f"Source runtime row {source_row} has an empty path"
                )
            self.relpath_bytes += path_size
            if self.relpath_bytes > _UINT64_MAX:
                raise PartitionBuildError(
                    f"{self.role} CIF paths exceed uint64 offsets"
                )
            self.offsets.append_u64(self.relpath_bytes)

        self.row_count += cluster_size
        self.cluster_count += 1

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def runtime_artifacts(
        self,
        roots_artifact: Mapping[str, object],
        pair_artifact: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "row_offsets": _artifact_record(
                path="row_offsets.u64",
                size_bytes=self.offsets.size_bytes,
                sha256=self.offsets.sha256,
                dtype="uint64",
                count=self.row_count + 1,
            ),
            "source_root_ids": _artifact_record(
                path="source_root_ids.u8",
                size_bytes=self.roots.size_bytes,
                sha256=self.roots.sha256,
                dtype="uint8",
                count=self.row_count,
            ),
            "cif_relpaths": _artifact_record(
                path="cif_relpaths.bin",
                size_bytes=self.relpaths.size_bytes,
                sha256=self.relpaths.sha256,
                dtype="bytes",
                count=self.relpath_bytes,
            ),
            "roots": dict(roots_artifact),
            "partition_pair": dict(pair_artifact),
        }

    def sampling_manifest(self) -> dict[str, object]:
        return {
            "alignment": (
                "Array index i identifies one retained source cluster. Members "
                "occupy [first_row_ids[i], first_row_ids[i] + "
                "cluster_sizes[i]). Stable cluster_ids are retained."
            ),
            "arrays": {
                "cluster_ids": {
                    **_artifact_record(
                        path="sampling/cluster_ids.u32",
                        size_bytes=self.cluster_ids.size_bytes,
                        sha256=self.cluster_ids.sha256,
                        dtype="uint32",
                        count=self.cluster_count,
                    ),
                },
                "first_row_ids": {
                    **_artifact_record(
                        path="sampling/first_row_ids.u64",
                        size_bytes=self.first_rows.size_bytes,
                        sha256=self.first_rows.sha256,
                        dtype="uint64",
                        count=self.cluster_count,
                    ),
                },
                "cluster_sizes": {
                    **_artifact_record(
                        path="sampling/cluster_sizes.u32",
                        size_bytes=self.cluster_sizes.size_bytes,
                        sha256=self.cluster_sizes.sha256,
                        dtype="uint32",
                        count=self.cluster_count,
                    ),
                },
            },
        }


def _load_short_path_stats(
    stats_path: Path,
) -> tuple[dict[str, object], dict[str, int | str]]:
    stat = _stat_record(stats_path)
    sha256 = _sha256_file(stats_path)
    try:
        document = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PartitionBuildError(
            f"Cannot read short-path stats {stats_path}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise PartitionBuildError("Short-path stats must be a JSON object")
    if not _same_stat(stat, stats_path):
        raise PartitionBuildError(
            f"Short-path stats changed while being read: {stats_path}"
        )
    return document, {**stat, "sha256": sha256}


def _validate_canonical_audit_source(
    stats_document: Mapping[str, object],
    *,
    canonical_audit_index: Path | str | None,
    expected_rows: int,
) -> tuple[dict[str, int | str], dict[str, int | str]]:
    """Hash-bind the short-path manifest to its canonical-length audit."""

    input_record = stats_document.get("input")
    if not isinstance(input_record, dict):
        raise PartitionBuildError(
            "Short-path stats lacks canonical-length input provenance"
        )
    recorded_path = input_record.get("index")
    if not isinstance(recorded_path, str) or not recorded_path:
        raise PartitionBuildError(
            "Short-path stats does not record its canonical-length index"
        )
    audit_path = (
        Path(canonical_audit_index).expanduser().resolve(strict=True)
        if canonical_audit_index is not None
        else Path(recorded_path).expanduser().resolve(strict=True)
    )
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"Canonical-length audit is not a file: {audit_path}"
        )
    audit_stat = _stat_record(audit_path)
    try:
        recorded_size = int(input_record["index_size_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PartitionBuildError(
            "Short-path stats has invalid canonical-length index size"
        ) from exc
    if recorded_size != int(audit_stat["size"]):
        raise PartitionBuildError(
            "Canonical-length audit size differs from short-path stats"
        )

    success_path = audit_path.parent / "_FINALIZE_SUCCESS.json"
    if not success_path.is_file():
        raise FileNotFoundError(
            f"Canonical audit completion marker is missing: {success_path}"
        )
    success_stat = _stat_record(success_path)
    success_sha256 = _sha256_file(success_path)
    try:
        success = json.loads(success_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PartitionBuildError(
            f"Cannot read canonical audit completion marker: {success_path}"
        ) from exc
    if not isinstance(success, dict):
        raise PartitionBuildError(
            "Canonical audit completion marker must be a JSON object"
        )
    try:
        success_rows = int(success["merged_audit_rows"])
        success_size = int(success["merged_audit_size"])
        success_failures = int(success["failure_count"])
        complete_cifs = int(success["complete_cifs"])
        success_sha = str(success["merged_audit_sha256"])
        success_relpath = str(success["merged_audit_relpath"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PartitionBuildError(
            "Canonical audit completion marker has invalid fields"
        ) from exc
    if (
        success.get("merged_audit_present") is not True
        or success.get("artifact_role")
        != "distillation_bioassembly_finalize"
        or success_rows != expected_rows
        or complete_cifs != expected_rows
        or success_size != int(audit_stat["size"])
        or success_failures != 0
        or success_relpath != audit_path.name
        or len(success_sha) != 64
        or any(char not in "0123456789abcdef" for char in success_sha.lower())
    ):
        raise PartitionBuildError(
            "Canonical audit completion marker does not describe the "
            "requested complete audit"
        )
    audit_sha256 = _sha256_file(audit_path)
    if audit_sha256 != success_sha.lower():
        raise PartitionBuildError(
            "Canonical-length audit SHA-256 differs from its completion marker"
        )
    if not _same_stat(audit_stat, audit_path) or not _same_stat(
        success_stat,
        success_path,
    ):
        raise PartitionBuildError(
            "Canonical audit or completion marker changed while being hashed"
        )
    return (
        {**audit_stat, "sha256": audit_sha256},
        {**success_stat, "sha256": success_sha256},
    )


def _load_short_paths(
    path: Path,
    *,
    root_ids: Sequence[str],
) -> tuple[dict[bytes, int], dict[str, int | str]]:
    root_bytes = {root_id.encode("utf-8") for root_id in root_ids}
    if any(
        not value or b"/" in value or any(char in value for char in b"\0\t\r\n")
        for value in root_bytes
    ):
        raise PartitionBuildError(
            "Runtime root IDs are incompatible with root_id/cif_relpath paths"
        )

    stat = _stat_record(path)
    digest = hashlib.sha256()
    selected: dict[bytes, int] = {}
    with path.open("rb", buffering=_COPY_CHUNK_SIZE) as stream:
        for line_number, line in enumerate(stream, start=1):
            digest.update(line)
            if not line.endswith(b"\n"):
                raise PartitionBuildError(
                    f"Short-path line {line_number} is not newline terminated"
                )
            value = line[:-1]
            if (
                not value
                or b"\0" in value
                or b"\t" in value
                or b"\r" in value
                or b"\\" in value
            ):
                raise PartitionBuildError(
                    f"Short-path line {line_number} is unsafe"
                )
            root_id, separator, relpath = value.partition(b"/")
            if not separator or root_id not in root_bytes:
                raise PartitionBuildError(
                    f"Short-path line {line_number} has an unknown root ID"
                )
            try:
                decoded_relpath = relpath.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PartitionBuildError(
                    f"Short-path line {line_number} is not UTF-8"
                ) from exc
            _safe_relative_path(decoded_relpath, suffix=".cif")
            if value in selected:
                raise PartitionBuildError(
                    f"Duplicate short path at line {line_number}: "
                    f"{value.decode('utf-8')}"
                )
            selected[value] = line_number - 1
    if not selected:
        raise PartitionBuildError(f"Short-path manifest is empty: {path}")
    if not _same_stat(stat, path):
        raise PartitionBuildError(
            f"Short-path manifest changed while being read: {path}"
        )
    return selected, {
        **stat,
        "sha256": digest.hexdigest(),
        "line_count": len(selected),
    }


def _validate_short_path_stats(
    document: Mapping[str, object],
    *,
    stats_source: Mapping[str, object],
    paths_source: Mapping[str, object],
    full_manifest: Mapping[str, object],
    max_representative_length: int,
) -> None:
    selection = document.get("selection")
    counts = document.get("counts")
    outputs = document.get("outputs")
    path_output = outputs.get("paths") if isinstance(outputs, dict) else None
    if (
        not isinstance(selection, dict)
        or not isinstance(counts, dict)
        or not isinstance(path_output, dict)
    ):
        raise PartitionBuildError(
            f"{stats_source['path']} lacks selection/counts/outputs.paths"
        )
    try:
        maximum = int(selection["max_canonical_length"])
        input_rows = int(counts["input_rows"])
        selected_rows = int(counts["selected_rows"])
        excluded_too_short = int(counts.get("excluded_too_short", 0))
        excluded_too_long = int(counts["excluded_too_long"])
        output_lines = int(path_output["lines"])
        output_size = int(path_output["size_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PartitionBuildError(
            f"{stats_source['path']} has invalid numeric fields"
        ) from exc
    operator = selection.get("canonical_length_operator")
    if operator not in {"<=", "closed_interval"}:
        raise PartitionBuildError(
            f"Unsupported canonical-length operator in {stats_source['path']}: "
            f"{operator!r}"
        )
    if maximum != max_representative_length:
        raise PartitionBuildError(
            "Short-path threshold does not match requested representative "
            f"threshold: {maximum} != {max_representative_length}"
        )
    minimum = selection.get("min_canonical_length")
    if minimum is not None and int(minimum) != 1:
        raise PartitionBuildError(
            "Short-path manifest must include every positive canonical length"
        )
    if excluded_too_short != 0:
        raise PartitionBuildError(
            "Short-path manifest excluded positive proteins below its interval"
        )
    if input_rows != int(full_manifest["row_count"]):
        raise PartitionBuildError(
            f"Short-path input rows {input_rows} do not match full runtime rows "
            f"{full_manifest['row_count']}"
        )
    if selected_rows != int(paths_source["line_count"]):
        raise PartitionBuildError(
            "Short-path selected_rows does not match the path manifest"
        )
    if selected_rows + excluded_too_short + excluded_too_long != input_rows:
        raise PartitionBuildError(
            "Short-path selected/excluded counts do not cover the input"
        )
    if output_lines != selected_rows or output_size != int(paths_source["size"]):
        raise PartitionBuildError(
            "Short-path output metadata does not match the path manifest"
        )
    expected_content_sha = path_output.get("content_sha256")
    if expected_content_sha != paths_source["sha256"]:
        raise PartitionBuildError(
            "Short-path content SHA-256 does not match stats metadata"
        )
    selected_max = counts.get("selected_length_max")
    if selected_max is None or int(selected_max) > max_representative_length:
        raise PartitionBuildError(
            "Short-path selected_length_max exceeds the requested threshold"
        )
    selected_min = counts.get("selected_length_min")
    if selected_min is None or int(selected_min) <= 0:
        raise PartitionBuildError(
            "Short-path selected_length_min must be positive"
        )
    roots_sha = document.get("input", {}).get("roots_json_sha256")
    full_roots_sha = full_manifest["artifacts"]["roots"]["sha256"]
    if roots_sha != full_roots_sha:
        raise PartitionBuildError(
            "Short-path stats and full runtime index use different roots.json"
        )


def _source_runtime_paths(
    directory: Path,
    manifest: Mapping[str, object],
) -> list[Path]:
    records = [
        manifest["artifacts"]["row_offsets"],
        manifest["artifacts"]["source_root_ids"],
        manifest["artifacts"]["cif_relpaths"],
        manifest["artifacts"]["roots"],
        manifest["sampling"]["arrays"]["cluster_ids"],
        manifest["sampling"]["arrays"]["first_row_ids"],
        manifest["sampling"]["arrays"]["cluster_sizes"],
    ]
    paths = [directory / "manifest.json"]
    for record in records:
        relative = _safe_relative_path(str(record["path"]))
        paths.append(directory.joinpath(*relative.parts))
    return paths


def _copy_roots_artifact(
    source_path: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> dict[str, int | str]:
    shutil.copyfile(source_path, destination)
    actual_sha256 = _sha256_file(destination)
    if actual_sha256 != expected_sha256:
        raise PartitionBuildError(
            f"Copied roots.json SHA-256 mismatch: {actual_sha256} != "
            f"{expected_sha256}"
        )
    return {
        "path": "roots.json",
        "size_bytes": destination.stat().st_size,
        "sha256": actual_sha256,
    }


def _representative_source_id(
    source: _RuntimeMaps,
    representative_row_id: int,
    root_prefixes: Sequence[str],
) -> str:
    root_ordinal = source.root_ordinal(representative_row_id)
    if root_ordinal >= len(root_prefixes):
        raise PartitionBuildError(
            f"Unknown root ordinal {root_ordinal} at representative row "
            f"{representative_row_id}"
        )
    try:
        relpath = source.relpath_bytes(representative_row_id).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PartitionBuildError(
            f"Representative row {representative_row_id} has invalid UTF-8"
        ) from exc
    stem = _safe_relative_path(relpath, suffix=".cif").stem
    return f"{root_prefixes[root_ordinal]}{stem}"


def _parse_cluster_row(
    raw_line: bytes,
    *,
    line_number: int,
) -> tuple[int, str, int, int, int]:
    fields = raw_line.rstrip(b"\r\n").split(b",")
    if len(fields) != len(_CLUSTER_HEADER):
        raise PartitionBuildError(
            f"Cluster CSV row {line_number} has {len(fields)} fields; expected "
            f"{len(_CLUSTER_HEADER)}"
        )
    try:
        cluster_id = int(fields[0])
        cluster_size = int(fields[4])
        first_row_id = int(fields[6])
        representative_row_id = int(fields[7])
        representative_id = fields[2].decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise PartitionBuildError(
            f"Cluster CSV row {line_number} has invalid fields"
        ) from exc
    if not representative_id:
        raise PartitionBuildError(
            f"Cluster CSV row {line_number} has an empty representative_id"
        )
    return (
        cluster_id,
        representative_id,
        cluster_size,
        first_row_id,
        representative_row_id,
    )


def _partition_identity(
    *,
    full_manifest_sha256: str,
    clusters_sha256: str,
    short_paths_sha256: str,
    short_stats_sha256: str,
    canonical_audit_sha256: str,
    canonical_audit_success_sha256: str,
    max_representative_length: int,
) -> str:
    document = {
        "partition_schema_version": PARTITION_SCHEMA_VERSION,
        "full_runtime_manifest_sha256": full_manifest_sha256,
        "clusters_sha256": clusters_sha256,
        "short_paths_sha256": short_paths_sha256,
        "short_stats_sha256": short_stats_sha256,
        "canonical_audit_sha256": canonical_audit_sha256,
        "canonical_audit_success_sha256": (
            canonical_audit_success_sha256
        ),
        "max_representative_length": max_representative_length,
    }
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _partition_manifest(
    *,
    writer: _PartitionWriter,
    role: str,
    root_ids: Sequence[str],
    roots_artifact: Mapping[str, object],
    pair_artifact: Mapping[str, object],
    pair_receipt: Mapping[str, object],
    full_manifest_source: Mapping[str, object],
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "format": RUNTIME_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": dict(full_manifest_source),
        "row_count": writer.row_count,
        "cluster_count": writer.cluster_count,
        "root_ids": list(root_ids),
        "artifacts": writer.runtime_artifacts(
            roots_artifact,
            pair_artifact,
        ),
        "sampling": writer.sampling_manifest(),
        "path_rules": {
            "cif": "source root / cif_relpath",
            "a3m": "replace the final .cif suffix with .a3m",
            "bioassembly": (
                "CIF parent / bioassembly / <CIF stem>.pkl.gz"
            ),
        },
        "partition": {
            "schema_version": PARTITION_SCHEMA_VERSION,
            "pair_id": pair_receipt["pair_id"],
            "role": role,
            "criterion": pair_receipt["criterion"],
            "member_policy": "retain_whole_source_cluster",
            "source_cluster_id_policy": "preserve_stable_cluster_id",
            "row_order": (
                "source cluster ordinal order, then source member order; row "
                "IDs are compacted within this partition"
            ),
            "pair_receipt_sha256": pair_artifact["sha256"],
            "source_row_count": pair_receipt["source"]["row_count"],
            "source_cluster_count": pair_receipt["source"]["cluster_count"],
        },
        "elapsed_seconds": elapsed_seconds,
    }


def _validate_pair_receipts(
    short_dir: Path,
    long_dir: Path,
    short_manifest: Mapping[str, object],
    long_manifest: Mapping[str, object],
) -> dict[str, object]:
    short_partition = short_manifest.get("partition")
    long_partition = long_manifest.get("partition")
    if not isinstance(short_partition, dict) or not isinstance(
        long_partition,
        dict,
    ):
        raise PartitionBuildError("Runtime manifests lack partition metadata")
    if short_partition.get("role") != "short_le200":
        raise PartitionBuildError("Short runtime has the wrong partition role")
    if long_partition.get("role") != "long_gt200":
        raise PartitionBuildError("Long runtime has the wrong partition role")
    if (
        short_partition.get("pair_id") != long_partition.get("pair_id")
        or not short_partition.get("pair_id")
    ):
        raise PartitionBuildError("Runtime partitions have different pair IDs")

    short_receipt_path = short_dir / "partition_pair.json"
    long_receipt_path = long_dir / "partition_pair.json"
    short_payload = short_receipt_path.read_bytes()
    long_payload = long_receipt_path.read_bytes()
    if short_payload != long_payload:
        raise PartitionBuildError(
            "Short and long partition receipts are not byte-identical"
        )
    receipt_sha256 = hashlib.sha256(short_payload).hexdigest()
    if (
        short_partition.get("pair_receipt_sha256") != receipt_sha256
        or long_partition.get("pair_receipt_sha256") != receipt_sha256
    ):
        raise PartitionBuildError("Partition receipt SHA-256 mismatch")
    try:
        receipt = json.loads(short_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PartitionBuildError("Partition receipt is invalid JSON") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("format") != PAIR_FORMAT
        or receipt.get("pair_id") != short_partition["pair_id"]
    ):
        raise PartitionBuildError("Partition receipt identity is invalid")
    partitions = receipt.get("partitions")
    coverage = receipt.get("coverage")
    source = receipt.get("source")
    if (
        not isinstance(partitions, dict)
        or not isinstance(coverage, dict)
        or not isinstance(source, dict)
    ):
        raise PartitionBuildError(
            "Partition receipt lacks partitions/coverage/source metadata"
        )
    criterion = receipt.get("criterion")
    clusters_source = source.get("clusters")
    short_paths_source = source.get("short_paths")
    short_stats_source = source.get("short_paths_stats")
    canonical_audit_source = source.get("canonical_audit")
    canonical_success_source = source.get("canonical_audit_success")
    if (
        not isinstance(criterion, dict)
        or not isinstance(clusters_source, dict)
        or not isinstance(short_paths_source, dict)
        or not isinstance(short_stats_source, dict)
        or not isinstance(canonical_audit_source, dict)
        or not isinstance(canonical_success_source, dict)
    ):
        raise PartitionBuildError(
            "Partition receipt lacks identity-bearing source metadata"
        )
    try:
        expected_pair_id = _partition_identity(
            full_manifest_sha256=str(
                source["full_runtime_manifest_sha256"]
            ),
            clusters_sha256=str(clusters_source["sha256"]),
            short_paths_sha256=str(short_paths_source["sha256"]),
            short_stats_sha256=str(short_stats_source["sha256"]),
            canonical_audit_sha256=str(
                canonical_audit_source["sha256"]
            ),
            canonical_audit_success_sha256=str(
                canonical_success_source["sha256"]
            ),
            max_representative_length=int(
                criterion["max_representative_length"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PartitionBuildError(
            "Partition receipt has invalid identity metadata"
        ) from exc
    if expected_pair_id != receipt["pair_id"]:
        raise PartitionBuildError(
            "Partition pair ID does not match its recorded source hashes"
        )
    expected_manifests = {
        "short_le200": short_manifest,
        "long_gt200": long_manifest,
    }
    try:
        for role, manifest in expected_manifests.items():
            record = partitions.get(role)
            if not isinstance(record, dict):
                raise PartitionBuildError(
                    f"Partition receipt lacks {role} metadata"
                )
            if (
                int(record.get("row_count", -1))
                != int(manifest["row_count"])
                or int(record.get("cluster_count", -1))
                != int(manifest["cluster_count"])
            ):
                raise PartitionBuildError(
                    f"Partition receipt counts disagree with {role} manifest"
                )
            pair_record = manifest["artifacts"].get("partition_pair")
            if (
                not isinstance(pair_record, dict)
                or pair_record.get("path") != "partition_pair.json"
                or int(pair_record.get("size_bytes", -1))
                != len(short_payload)
                or pair_record.get("sha256") != receipt_sha256
            ):
                raise PartitionBuildError(
                    f"{role} manifest has invalid partition_pair artifact "
                    "metadata"
                )
        row_sum = int(short_manifest["row_count"]) + int(
            long_manifest["row_count"]
        )
        cluster_sum = int(short_manifest["cluster_count"]) + int(
            long_manifest["cluster_count"]
        )
        counts_match = (
            int(coverage.get("row_count_sum", -1)) == row_sum
            and int(coverage.get("cluster_count_sum", -1)) == cluster_sum
            and int(coverage.get("overlap", -1)) == 0
            and int(source.get("row_count", -1)) == row_sum
            and int(source.get("cluster_count", -1)) == cluster_sum
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PartitionBuildError(
            "Partition receipt contains invalid count metadata"
        ) from exc
    if not counts_match:
        raise PartitionBuildError(
            "Partition receipt coverage counts are inconsistent"
        )
    return receipt


def _compare_partition_union(
    *,
    full_dir: Path,
    full_manifest: Mapping[str, object],
    short_dir: Path,
    short_manifest: Mapping[str, object],
    long_dir: Path,
    long_manifest: Mapping[str, object],
) -> None:
    if (
        tuple(full_manifest["root_ids"])
        != tuple(short_manifest["root_ids"])
        or tuple(full_manifest["root_ids"])
        != tuple(long_manifest["root_ids"])
    ):
        raise PartitionBuildError(
            "Partition root ID tables differ from the full runtime"
        )
    with (
        _RuntimeMaps(full_dir, full_manifest) as full,
        _RuntimeMaps(short_dir, short_manifest) as short,
        _RuntimeMaps(long_dir, long_manifest) as long,
    ):
        short_ordinal = 0
        long_ordinal = 0
        for full_ordinal in range(full.cluster_count):
            cluster_id = full.cluster_id(full_ordinal)
            short_match = (
                short_ordinal < short.cluster_count
                and short.cluster_id(short_ordinal) == cluster_id
            )
            long_match = (
                long_ordinal < long.cluster_count
                and long.cluster_id(long_ordinal) == cluster_id
            )
            if short_match == long_match:
                state = "both" if short_match else "neither"
                raise PartitionBuildError(
                    f"Source cluster {cluster_id} appears in {state} partition"
                )
            partition = short if short_match else long
            partition_ordinal = (
                short_ordinal if short_match else long_ordinal
            )
            source_size = full.cluster_size(full_ordinal)
            partition_size = partition.cluster_size(partition_ordinal)
            if source_size != partition_size:
                raise PartitionBuildError(
                    f"Cluster {cluster_id} size changed: {source_size} != "
                    f"{partition_size}"
                )
            source_first = full.first_row_id(full_ordinal)
            partition_first = partition.first_row_id(partition_ordinal)
            source_end = source_first + source_size
            partition_end = partition_first + partition_size
            if full.root_block(source_first, source_end) != partition.root_block(
                partition_first,
                partition_end,
            ):
                raise PartitionBuildError(
                    f"Cluster {cluster_id} source-root ordinals changed"
                )
            if full.relpath_block(
                source_first,
                source_end,
            ) != partition.relpath_block(partition_first, partition_end):
                raise PartitionBuildError(
                    f"Cluster {cluster_id} CIF relative paths changed"
                )
            if short_match:
                short_ordinal += 1
            else:
                long_ordinal += 1
        if (
            short_ordinal != short.cluster_count
            or long_ordinal != long.cluster_count
        ):
            raise PartitionBuildError(
                "Partition sampling arrays contain clusters absent from source"
            )


def _publish_pair(
    *,
    short_staging: Path,
    long_staging: Path,
    short_destination: Path,
    long_destination: Path,
) -> None:
    if os.path.lexists(short_destination) or os.path.lexists(long_destination):
        raise FileExistsError(
            "Refusing to overwrite an existing short/long runtime directory"
        )
    moved_short = False
    try:
        os.replace(short_staging, short_destination)
        moved_short = True
        if os.path.lexists(long_destination):
            raise FileExistsError(
                f"Long runtime path appeared during publication: "
                f"{long_destination}"
            )
        os.replace(long_staging, long_destination)
    except BaseException:
        if moved_short and os.path.lexists(short_destination):
            os.replace(short_destination, short_staging)
        raise


def build_length_partitioned_runtime_indices(
    *,
    full_runtime_dir: Path | str,
    clusters_csv: Path | str,
    short_paths: Path | str,
    short_output_dir: Path | str,
    long_output_dir: Path | str,
    short_paths_stats: Path | str | None = None,
    canonical_audit_index: Path | str | None = None,
    max_representative_length: int = DEFAULT_MAX_REPRESENTATIVE_LENGTH,
    progress_every_clusters: int = 1_000_000,
) -> dict[str, object]:
    """Build and atomically publish two whole-cluster runtime partitions."""

    if max_representative_length != DEFAULT_MAX_REPRESENTATIVE_LENGTH:
        raise ValueError(
            "This short_le200/long_gt200 schema requires "
            f"max_representative_length={DEFAULT_MAX_REPRESENTATIVE_LENGTH}"
        )
    if progress_every_clusters < 0:
        raise ValueError("progress_every_clusters must be non-negative")

    full_dir = Path(full_runtime_dir).expanduser().resolve(strict=True)
    clusters_path = Path(clusters_csv).expanduser().resolve(strict=True)
    short_paths_path = Path(short_paths).expanduser().resolve(strict=True)
    if short_paths_stats is None:
        suffix = ".paths.tsv"
        if not short_paths_path.name.endswith(suffix):
            raise ValueError(
                "Cannot infer stats path unless short_paths ends in .paths.tsv"
            )
        stats_path = short_paths_path.with_name(
            short_paths_path.name[: -len(suffix)] + ".stats.json"
        )
        stats_path = stats_path.resolve(strict=True)
    else:
        stats_path = Path(short_paths_stats).expanduser().resolve(strict=True)
    if not full_dir.is_dir():
        raise NotADirectoryError(f"Full runtime is not a directory: {full_dir}")
    for path in (clusters_path, short_paths_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is not a file: {path}")

    short_destination = _normalize_new_directory(short_output_dir)
    long_destination = _normalize_new_directory(long_output_dir)
    if short_destination == long_destination:
        raise ValueError("Short and long output directories must differ")
    if short_destination.parent != long_destination.parent:
        raise ValueError(
            "Short and long outputs must share one parent for paired publication"
        )
    for destination in (short_destination, long_destination):
        if destination == full_dir or full_dir in destination.parents:
            raise ValueError(
                "Partition outputs must not replace or be nested in full runtime"
            )

    with _pair_build_lock(short_destination, long_destination):
        if os.path.lexists(short_destination) or os.path.lexists(
            long_destination
        ):
            raise FileExistsError(
                "Refusing to overwrite an existing short/long runtime directory"
            )

        started = time.monotonic()
        full_source_paths = _source_runtime_paths(
            full_dir,
            json.loads(
                (full_dir / "manifest.json").read_text(encoding="utf-8")
            ),
        )
        full_source_stats = {
            str(path): _stat_record(path) for path in full_source_paths
        }
        full_manifest = verify_runtime_distillation_index(full_dir)
        full_manifest_path = full_dir / "manifest.json"
        full_manifest_sha256 = _sha256_file(full_manifest_path)

        roots_records = full_manifest["artifacts"]["roots"]
        root_ids = tuple(str(value) for value in full_manifest["root_ids"])
        roots_document = json.loads(
            (full_dir / str(roots_records["path"])).read_text(encoding="utf-8")
        )
        root_records = roots_document.get("roots")
        if not isinstance(root_records, list) or len(root_records) != len(root_ids):
            raise PartitionBuildError("Full runtime roots.json is inconsistent")
        root_prefixes: list[str] = []
        for ordinal, record in enumerate(root_records):
            if (
                not isinstance(record, dict)
                or str(record.get("root_id")) != root_ids[ordinal]
            ):
                raise PartitionBuildError(
                    "Full runtime roots.json order is inconsistent"
                )
            prefix = record.get("source_prefix")
            if not isinstance(prefix, str) or not prefix.endswith("|"):
                raise PartitionBuildError(
                    f"Root {root_ids[ordinal]} lacks source_prefix"
                )
            root_prefixes.append(prefix)

        stats_document, stats_source = _load_short_path_stats(stats_path)
        selected_short_paths, paths_source = _load_short_paths(
            short_paths_path,
            root_ids=root_ids,
        )
        _validate_short_path_stats(
            stats_document,
            stats_source=stats_source,
            paths_source=paths_source,
            full_manifest=full_manifest,
            max_representative_length=max_representative_length,
        )
        canonical_source, canonical_success_source = (
            _validate_canonical_audit_source(
                stats_document,
                canonical_audit_index=canonical_audit_index,
                expected_rows=int(full_manifest["row_count"]),
            )
        )
        canonical_path = Path(str(canonical_source["path"]))
        canonical_success_path = Path(
            str(canonical_success_source["path"])
        )

        clusters_source = _stat_record(clusters_path)
        clusters_sha256 = _sha256_file(clusters_path)
        pair_staging = short_destination.parent / (
            f".distillation-length-partition.tmp.{os.getpid()}."
            f"{uuid.uuid4().hex}"
        )
        pair_staging.mkdir()
        short_staging = pair_staging / "short"
        long_staging = pair_staging / "long"
        matched_short_paths = bytearray(len(selected_short_paths))
        matched_short_count = 0
        expected_source_first = 0
        cluster_count = 0
        matrix = {
            "short_le200": {
                "short_members": 0,
                "long_members": 0,
                "mixed_clusters": 0,
            },
            "long_gt200": {
                "short_members": 0,
                "long_members": 0,
                "mixed_clusters": 0,
            },
        }

        try:
            with (
                _RuntimeMaps(full_dir, full_manifest) as source,
                _PartitionWriter(short_staging, "short_le200") as short_writer,
                _PartitionWriter(long_staging, "long_gt200") as long_writer,
            ):
                root_key_prefixes = tuple(
                    f"{root_id}/".encode("utf-8") for root_id in root_ids
                )
                opener = gzip.open if clusters_path.suffix.lower() == ".gz" else open
                with opener(clusters_path, "rb") as cluster_stream:
                    header = tuple(
                        cluster_stream.readline().rstrip(b"\r\n").split(b",")
                    )
                    if header != _CLUSTER_HEADER:
                        raise PartitionBuildError(
                            f"Unexpected clusters CSV header: {header}"
                        )
                    for cluster_count, raw_line in enumerate(
                        cluster_stream,
                        start=1,
                    ):
                        (
                            cluster_id,
                            representative_id,
                            cluster_size,
                            first_row_id,
                            representative_row_id,
                        ) = _parse_cluster_row(
                            raw_line,
                            line_number=cluster_count + 1,
                        )
                        source_ordinal = cluster_count - 1
                        if source_ordinal >= source.cluster_count:
                            raise PartitionBuildError(
                                "clusters.csv contains more clusters than the "
                                "full runtime"
                            )
                        if (
                            cluster_id != source.cluster_id(source_ordinal)
                            or cluster_size
                            != source.cluster_size(source_ordinal)
                            or first_row_id
                            != source.first_row_id(source_ordinal)
                        ):
                            raise PartitionBuildError(
                                f"clusters.csv row {cluster_count + 1} is not "
                                "aligned with full runtime sampling arrays"
                            )
                        if (
                            first_row_id != expected_source_first
                            or cluster_size <= 0
                            or not first_row_id
                            <= representative_row_id
                            < first_row_id + cluster_size
                        ):
                            raise PartitionBuildError(
                                f"Invalid source interval or representative at "
                                f"cluster {cluster_id}"
                            )
                        expected_source_first += cluster_size
                        expected_representative_id = _representative_source_id(
                            source,
                            representative_row_id,
                            root_prefixes,
                        )
                        if representative_id != expected_representative_id:
                            raise PartitionBuildError(
                                f"Cluster {cluster_id} representative_id does "
                                "not match representative_row_id: "
                                f"{representative_id!r} != "
                                f"{expected_representative_id!r}"
                            )

                        rep_root = source.root_ordinal(representative_row_id)
                        representative_key = (
                            root_key_prefixes[rep_root]
                            + source.relpath_bytes(representative_row_id)
                        )
                        representative_is_short = (
                            representative_key in selected_short_paths
                        )
                        role = (
                            "short_le200"
                            if representative_is_short
                            else "long_gt200"
                        )
                        writer = (
                            short_writer
                            if representative_is_short
                            else long_writer
                        )

                        short_members = 0
                        long_members = 0
                        for row_id in range(
                            first_row_id,
                            first_row_id + cluster_size,
                        ):
                            root_ordinal = source.root_ordinal(row_id)
                            path_key = (
                                root_key_prefixes[root_ordinal]
                                + source.relpath_bytes(row_id)
                            )
                            short_ordinal = selected_short_paths.get(path_key)
                            if short_ordinal is None:
                                long_members += 1
                            else:
                                if matched_short_paths[short_ordinal]:
                                    raise PartitionBuildError(
                                        "Full runtime contains duplicate short "
                                        f"path: {path_key.decode('utf-8')}"
                                    )
                                matched_short_paths[short_ordinal] = 1
                                matched_short_count += 1
                                short_members += 1
                        matrix[role]["short_members"] += short_members
                        matrix[role]["long_members"] += long_members
                        if short_members and long_members:
                            matrix[role]["mixed_clusters"] += 1

                        writer.append_cluster(
                            source=source,
                            cluster_id=cluster_id,
                            first_row=first_row_id,
                            cluster_size=cluster_size,
                        )
                        if (
                            progress_every_clusters
                            and cluster_count % progress_every_clusters == 0
                        ):
                            elapsed = max(time.monotonic() - started, 1e-9)
                            print(
                                "[runtime-length-partition] "
                                f"clusters={cluster_count:,} "
                                f"rows={expected_source_first:,} "
                                f"rate={cluster_count / elapsed:,.0f} "
                                "clusters/s",
                                file=sys.stderr,
                                flush=True,
                            )

                if cluster_count != source.cluster_count:
                    raise PartitionBuildError(
                        f"clusters.csv has {cluster_count} clusters; full "
                        f"runtime has {source.cluster_count}"
                    )
                if expected_source_first != source.row_count:
                    raise PartitionBuildError(
                        f"Cluster intervals cover {expected_source_first} rows; "
                        f"full runtime has {source.row_count}"
                    )
                if matched_short_count != len(selected_short_paths):
                    missing_examples = [
                        key.decode("utf-8")
                        for key, ordinal in selected_short_paths.items()
                        if not matched_short_paths[ordinal]
                    ][:5]
                    raise PartitionBuildError(
                        f"{len(selected_short_paths) - matched_short_count} "
                        "short paths are absent from full runtime; examples="
                        f"{missing_examples}"
                    )
                if short_writer.cluster_count == 0 or long_writer.cluster_count == 0:
                    raise PartitionBuildError(
                        "Both short and long runtime partitions must be non-empty"
                    )

            if not _same_stat(clusters_source, clusters_path):
                raise PartitionBuildError(
                    f"clusters.csv changed while being read: {clusters_path}"
                )
            if not _same_stat(paths_source, short_paths_path):
                raise PartitionBuildError(
                    "Short-path manifest changed during partition build"
                )
            if not _same_stat(stats_source, stats_path):
                raise PartitionBuildError(
                    "Short-path stats changed during partition build"
                )
            if not _same_stat(canonical_source, canonical_path):
                raise PartitionBuildError(
                    "Canonical-length audit changed during partition build"
                )
            if not _same_stat(
                canonical_success_source,
                canonical_success_path,
            ):
                raise PartitionBuildError(
                    "Canonical audit completion marker changed during build"
                )
            for path in full_source_paths:
                if not _same_stat(full_source_stats[str(path)], path):
                    raise PartitionBuildError(
                        f"Full runtime source changed during build: {path}"
                    )

            pair_id = _partition_identity(
                full_manifest_sha256=full_manifest_sha256,
                clusters_sha256=clusters_sha256,
                short_paths_sha256=str(paths_source["sha256"]),
                short_stats_sha256=str(stats_source["sha256"]),
                canonical_audit_sha256=str(
                    canonical_source["sha256"]
                ),
                canonical_audit_success_sha256=str(
                    canonical_success_source["sha256"]
                ),
                max_representative_length=max_representative_length,
            )
            pair_receipt: dict[str, object] = {
                "schema_version": PARTITION_SCHEMA_VERSION,
                "format": PAIR_FORMAT,
                "pair_id": pair_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "criterion": {
                    "representative": (
                        "final representative_row_id from clusters.csv"
                    ),
                    "short": (
                        f"canonical_length <= "
                        f"{max_representative_length}"
                    ),
                    "long": (
                        f"canonical_length > "
                        f"{max_representative_length}"
                    ),
                    "max_representative_length": (
                        max_representative_length
                    ),
                },
                "member_policy": "retain_whole_source_cluster",
                "source": {
                    "full_runtime_dir": str(full_dir),
                    "full_runtime_manifest_sha256": full_manifest_sha256,
                    "row_count": int(full_manifest["row_count"]),
                    "cluster_count": int(full_manifest["cluster_count"]),
                    "clusters": {
                        **clusters_source,
                        "sha256": clusters_sha256,
                    },
                    "short_paths": dict(paths_source),
                    "short_paths_stats": dict(stats_source),
                    "canonical_audit": dict(canonical_source),
                    "canonical_audit_success": dict(
                        canonical_success_source
                    ),
                },
                "partitions": {
                    "short_le200": {
                        "row_count": short_writer.row_count,
                        "cluster_count": short_writer.cluster_count,
                        **matrix["short_le200"],
                    },
                    "long_gt200": {
                        "row_count": long_writer.row_count,
                        "cluster_count": long_writer.cluster_count,
                        **matrix["long_gt200"],
                    },
                },
                "coverage": {
                    "row_count_sum": (
                        short_writer.row_count + long_writer.row_count
                    ),
                    "cluster_count_sum": (
                        short_writer.cluster_count
                        + long_writer.cluster_count
                    ),
                    "overlap": 0,
                },
            }
            if (
                pair_receipt["coverage"]["row_count_sum"]
                != int(full_manifest["row_count"])
                or pair_receipt["coverage"]["cluster_count_sum"]
                != int(full_manifest["cluster_count"])
            ):
                raise AssertionError("Partition coverage accounting is invalid")

            full_roots_path = full_dir / str(roots_records["path"])
            short_roots = _copy_roots_artifact(
                full_roots_path,
                short_staging / "roots.json",
                expected_sha256=str(roots_records["sha256"]),
            )
            long_roots = _copy_roots_artifact(
                full_roots_path,
                long_staging / "roots.json",
                expected_sha256=str(roots_records["sha256"]),
            )
            short_pair_artifact = _write_json(
                short_staging / "partition_pair.json",
                pair_receipt,
            )
            long_pair_artifact = _write_json(
                long_staging / "partition_pair.json",
                pair_receipt,
            )
            if short_pair_artifact != long_pair_artifact:
                raise AssertionError("Pair receipt artifacts differ")

            full_manifest_source = {
                **_stat_record(full_manifest_path),
                "sha256": full_manifest_sha256,
                "kind": "full_runtime_manifest",
            }
            elapsed_seconds = time.monotonic() - started
            short_manifest = _partition_manifest(
                writer=short_writer,
                role="short_le200",
                root_ids=root_ids,
                roots_artifact=short_roots,
                pair_artifact=short_pair_artifact,
                pair_receipt=pair_receipt,
                full_manifest_source=full_manifest_source,
                elapsed_seconds=elapsed_seconds,
            )
            long_manifest = _partition_manifest(
                writer=long_writer,
                role="long_gt200",
                root_ids=root_ids,
                roots_artifact=long_roots,
                pair_artifact=long_pair_artifact,
                pair_receipt=pair_receipt,
                full_manifest_source=full_manifest_source,
                elapsed_seconds=elapsed_seconds,
            )
            _write_json(short_staging / "manifest.json", short_manifest)
            _write_json(long_staging / "manifest.json", long_manifest)

            verified_short = verify_runtime_distillation_index(short_staging)
            verified_long = verify_runtime_distillation_index(long_staging)
            _validate_pair_receipts(
                short_staging,
                long_staging,
                verified_short,
                verified_long,
            )
            _compare_partition_union(
                full_dir=full_dir,
                full_manifest=full_manifest,
                short_dir=short_staging,
                short_manifest=verified_short,
                long_dir=long_staging,
                long_manifest=verified_long,
            )
            if (
                not _same_stat(clusters_source, clusters_path)
                or not _same_stat(paths_source, short_paths_path)
                or not _same_stat(stats_source, stats_path)
                or not _same_stat(canonical_source, canonical_path)
                or not _same_stat(
                    canonical_success_source,
                    canonical_success_path,
                )
            ):
                raise PartitionBuildError(
                    "A partition source changed during final verification"
                )
            for path in full_source_paths:
                if not _same_stat(full_source_stats[str(path)], path):
                    raise PartitionBuildError(
                        f"Full runtime source changed during final verification: "
                        f"{path}"
                    )
            _publish_pair(
                short_staging=short_staging,
                long_staging=long_staging,
                short_destination=short_destination,
                long_destination=long_destination,
            )
            shutil.rmtree(pair_staging, ignore_errors=True)
            return pair_receipt
        except BaseException:
            shutil.rmtree(pair_staging, ignore_errors=True)
            raise


def verify_length_partitioned_runtime_indices(
    *,
    short_runtime_dir: Path | str,
    long_runtime_dir: Path | str,
    full_runtime_dir: Path | str | None = None,
    verify_recorded_sources: bool = False,
) -> dict[str, object]:
    """Verify both runtime indices, their shared receipt, and source coverage."""

    short_dir = Path(short_runtime_dir).expanduser().resolve(strict=True)
    long_dir = Path(long_runtime_dir).expanduser().resolve(strict=True)
    short_manifest = verify_runtime_distillation_index(short_dir)
    long_manifest = verify_runtime_distillation_index(long_dir)
    receipt = _validate_pair_receipts(
        short_dir,
        long_dir,
        short_manifest,
        long_manifest,
    )
    source = receipt.get("source")
    if not isinstance(source, dict):
        raise PartitionBuildError("Partition receipt lacks source provenance")
    resolved_full_dir = (
        Path(full_runtime_dir).expanduser().resolve(strict=True)
        if full_runtime_dir is not None
        else Path(str(source.get("full_runtime_dir", ""))).resolve(strict=True)
    )
    full_manifest = verify_runtime_distillation_index(resolved_full_dir)
    actual_manifest_sha256 = _sha256_file(
        resolved_full_dir / "manifest.json"
    )
    if actual_manifest_sha256 != source.get("full_runtime_manifest_sha256"):
        raise PartitionBuildError(
            "Full runtime manifest SHA-256 differs from partition provenance"
        )
    _compare_partition_union(
        full_dir=resolved_full_dir,
        full_manifest=full_manifest,
        short_dir=short_dir,
        short_manifest=short_manifest,
        long_dir=long_dir,
        long_manifest=long_manifest,
    )

    if verify_recorded_sources:
        for label in (
            "clusters",
            "short_paths",
            "short_paths_stats",
            "canonical_audit",
            "canonical_audit_success",
        ):
            record = source.get(label)
            if not isinstance(record, dict):
                raise PartitionBuildError(
                    f"Partition receipt lacks {label} provenance"
                )
            path = Path(str(record.get("path", "")))
            if not path.is_file():
                raise PartitionBuildError(
                    f"Recorded {label} source is missing: {path}"
                )
            if not _same_stat(record, path):
                raise PartitionBuildError(
                    f"Recorded {label} source stat changed: {path}"
                )
            if _sha256_file(path) != record.get("sha256"):
                raise PartitionBuildError(
                    f"Recorded {label} source SHA-256 changed: {path}"
                )
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build",
        help="Build and publish the paired short/long runtime indices",
    )
    build_parser.add_argument("--full-runtime-dir", type=Path, required=True)
    build_parser.add_argument("--clusters-csv", type=Path, required=True)
    build_parser.add_argument("--short-paths", type=Path, required=True)
    build_parser.add_argument("--short-paths-stats", type=Path)
    build_parser.add_argument(
        "--canonical-audit-index",
        type=Path,
        help=(
            "Canonical-length audit to hash and bind; defaults to the index "
            "recorded by short-path stats"
        ),
    )
    build_parser.add_argument("--short-output-dir", type=Path, required=True)
    build_parser.add_argument("--long-output-dir", type=Path, required=True)
    build_parser.add_argument(
        "--max-representative-length",
        type=int,
        default=DEFAULT_MAX_REPRESENTATIVE_LENGTH,
    )
    build_parser.add_argument(
        "--progress-every-clusters",
        type=int,
        default=1_000_000,
    )

    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify both partitions and their exact union of the full runtime",
    )
    verify_parser.add_argument(
        "--short-runtime-dir",
        type=Path,
        required=True,
    )
    verify_parser.add_argument(
        "--long-runtime-dir",
        type=Path,
        required=True,
    )
    verify_parser.add_argument("--full-runtime-dir", type=Path)
    verify_parser.add_argument(
        "--verify-recorded-sources",
        action="store_true",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "build":
        receipt = build_length_partitioned_runtime_indices(
            full_runtime_dir=args.full_runtime_dir,
            clusters_csv=args.clusters_csv,
            short_paths=args.short_paths,
            short_paths_stats=args.short_paths_stats,
            canonical_audit_index=args.canonical_audit_index,
            short_output_dir=args.short_output_dir,
            long_output_dir=args.long_output_dir,
            max_representative_length=args.max_representative_length,
            progress_every_clusters=args.progress_every_clusters,
        )
    else:
        receipt = verify_length_partitioned_runtime_indices(
            short_runtime_dir=args.short_runtime_dir,
            long_runtime_dir=args.long_runtime_dir,
            full_runtime_dir=args.full_runtime_dir,
            verify_recorded_sources=args.verify_recorded_sources,
        )
    print(
        json.dumps(
            {
                "pair_id": receipt["pair_id"],
                "partitions": receipt["partitions"],
                "coverage": receipt["coverage"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
