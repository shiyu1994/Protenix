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

"""Build a compact, mmap-friendly runtime view of a distillation catalog.

The authoritative CSV remains the source of truth.  This script scans it once
and emits only the fields needed to locate a sampled row:

* ``row_offsets.u64``: byte offsets into ``cif_relpaths.bin`` (N + 1 values);
* ``source_root_ids.u8``: an ordinal into the manifest root table (N values);
* ``cif_relpaths.bin``: concatenated UTF-8 relative CIF paths.

The source catalog's roots document and aligned cluster-sampling arrays are
copied into the output, making the published directory self-contained.  All
files are built in a sibling staging directory and published atomically.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import mmap
import os
import shutil
import struct
import sys
import time
import uuid
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import BinaryIO, IO, Iterator, Mapping, Sequence


RUNTIME_SCHEMA_VERSION = 1
_COPY_CHUNK_SIZE = 8 * 1024 * 1024
_BUFFER_FLUSH_SIZE = 8 * 1024 * 1024
_UINT64_MAX = (1 << 64) - 1
_SAMPLING_SPECS = {
    "cluster_ids": ("sampling/cluster_ids.u32", 4, "uint32"),
    "first_row_ids": ("sampling/first_row_ids.u64", 8, "uint64"),
    "cluster_sizes": ("sampling/cluster_sizes.u32", 4, "uint32"),
}


class RuntimeIndexError(RuntimeError):
    """Raised when an input or runtime-index invariant is violated."""


class _HashingRawReader(io.RawIOBase):
    """Forward-only raw reader that hashes the exact source bytes consumed."""

    def __init__(self, raw: BinaryIO, hasher: "hashlib._Hash"):
        self._raw = raw
        self._hasher = hasher

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        payload = self._raw.read(size)
        if payload:
            self._hasher.update(payload)
        return payload

    def readinto(self, buffer) -> int:
        count = self._raw.readinto(buffer)
        if count:
            self._hasher.update(memoryview(buffer)[:count])
        return count

    def close(self) -> None:
        if not self.closed:
            self._raw.close()
        super().close()


class _BufferedArtifactWriter:
    """Buffered binary artifact writer with on-write size and SHA-256."""

    def __init__(self, path: Path):
        self.path = path
        self._file: IO[bytes] | None = None
        self._buffer = bytearray()
        self._hasher = hashlib.sha256()
        self.size_bytes = 0

    def __enter__(self) -> "_BufferedArtifactWriter":
        self._file = self.path.open("wb", buffering=_COPY_CHUNK_SIZE)
        return self

    def append(self, payload: bytes) -> None:
        self._buffer.extend(payload)
        if len(self._buffer) >= _BUFFER_FLUSH_SIZE:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        if self._file is None:
            raise RuntimeError("Artifact writer is not open")
        self._file.write(self._buffer)
        self._hasher.update(self._buffer)
        self.size_bytes += len(self._buffer)
        self._buffer.clear()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.flush()
        if self._file is not None:
            self._file.close()

    @property
    def sha256(self) -> str:
        return self._hasher.hexdigest()


def _stat_document(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _same_stat(before: Mapping[str, int | str], after: os.stat_result) -> bool:
    return before["size"] == after.st_size and before["mtime_ns"] == after.st_mtime_ns


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb", buffering=_COPY_CHUNK_SIZE) as file_obj:
        while payload := file_obj.read(_COPY_CHUNK_SIZE):
            hasher.update(payload)
    return hasher.hexdigest()


@contextmanager
def _open_index_text(
    path: Path,
    hasher: "hashlib._Hash",
) -> Iterator[IO[str]]:
    """Open a plain or gzip CSV while hashing the exact on-disk bytes."""

    raw = path.open("rb", buffering=0)
    hashing_raw = _HashingRawReader(raw, hasher)
    binary: IO[bytes]
    if path.suffix.lower() == ".gz":
        binary = gzip.GzipFile(fileobj=hashing_raw, mode="rb")
    else:
        binary = io.BufferedReader(hashing_raw, buffer_size=_COPY_CHUNK_SIZE)
    text = io.TextIOWrapper(binary, encoding="utf-8", newline="")
    try:
        yield text
    finally:
        text.close()


def _safe_relative_path(value: object, *, suffix: str | None = None) -> PurePosixPath:
    raw = str(value).strip()
    if not raw or "\x00" in raw or "\\" in raw:
        raise RuntimeIndexError(f"Unsafe relative path: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise RuntimeIndexError(f"Unsafe relative path: {raw!r}")
    if suffix is not None and path.suffix.lower() != suffix:
        raise RuntimeIndexError(f"Expected a {suffix} relative path, got {raw!r}")
    return path


def _safe_artifact_path(directory: Path, value: object) -> Path:
    relative = _safe_relative_path(value)
    return directory.joinpath(*relative.parts)


def _load_and_normalize_roots(roots_path: Path) -> tuple[dict[str, object], dict[str, int]]:
    try:
        document = json.loads(roots_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeIndexError(f"Cannot read roots document {roots_path}: {exc}") from exc

    records = document.get("roots") if isinstance(document, dict) else None
    if not isinstance(records, list) or not records:
        raise RuntimeIndexError(f"{roots_path} must contain a non-empty roots list")
    if len(records) > 256:
        raise RuntimeIndexError(f"u8 source-root ordinals support at most 256 roots, found {len(records)}")

    root_to_ordinal: dict[str, int] = {}
    normalized_records: list[dict[str, object]] = []
    for ordinal, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeIndexError(f"{roots_path} contains a non-object roots entry")
        root_id = str(record.get("root_id", "")).strip()
        root_value = record.get("path")
        if not root_id or not isinstance(root_value, str) or not root_value or "\x00" in root_value:
            raise RuntimeIndexError(f"{roots_path} roots entries require root_id and path")
        if root_id in root_to_ordinal:
            raise RuntimeIndexError(f"{roots_path} contains duplicate root_id {root_id!r}")
        root_path = Path(root_value).expanduser()
        if not root_path.is_absolute():
            root_path = roots_path.parent / root_path
        normalized_record = dict(record)
        normalized_record["root_id"] = root_id
        normalized_record["path"] = str(root_path.resolve(strict=False))
        normalized_records.append(normalized_record)
        root_to_ordinal[root_id] = ordinal

    normalized_document = dict(document)
    normalized_document["roots"] = normalized_records
    return normalized_document, root_to_ordinal


def _write_json_artifact(path: Path, document: object) -> dict[str, int | str]:
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return {
        "path": path.name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _copy_artifact(source: Path, destination: Path) -> dict[str, int | str]:
    source_stat = _stat_document(source)
    hasher = hashlib.sha256()
    size_bytes = 0
    with source.open("rb", buffering=_COPY_CHUNK_SIZE) as input_file, destination.open(
        "wb", buffering=_COPY_CHUNK_SIZE
    ) as output_file:
        while payload := input_file.read(_COPY_CHUNK_SIZE):
            output_file.write(payload)
            hasher.update(payload)
            size_bytes += len(payload)
    if not _same_stat(source_stat, source.stat()):
        raise RuntimeIndexError(f"Source artifact changed while being copied: {source}")
    return {
        "path": destination.relative_to(destination.parents[1]).as_posix(),
        "size_bytes": size_bytes,
        "sha256": hasher.hexdigest(),
        "source": source_stat,
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
        "endian": "little" if dtype != "bytes" else "not_applicable",
        "count": count,
    }


def _read_source_sampling_manifest(catalog_dir: Path) -> dict[str, object]:
    manifest_path = catalog_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeIndexError(f"Cannot read source manifest {manifest_path}: {exc}") from exc
    sampling = document.get("sampling")
    return sampling if isinstance(sampling, dict) else {}


def _copy_sampling_arrays(
    *,
    sampling_dir: Path,
    staging_dir: Path,
    source_sampling_manifest: Mapping[str, object],
) -> tuple[dict[str, object], int]:
    output_sampling_dir = staging_dir / "sampling"
    output_sampling_dir.mkdir()
    source_array_records = source_sampling_manifest.get("arrays", {})
    if not isinstance(source_array_records, dict):
        source_array_records = {}

    arrays: dict[str, object] = {}
    counts: dict[str, int] = {}
    for name, (relative_path, item_size, dtype) in _SAMPLING_SPECS.items():
        source_path = sampling_dir / Path(relative_path).name
        if not source_path.is_file():
            raise RuntimeIndexError(f"Missing source sampling array: {source_path}")
        size_bytes = source_path.stat().st_size
        if size_bytes % item_size:
            raise RuntimeIndexError(
                f"Sampling array {source_path} size {size_bytes} is not divisible by {item_size}"
            )
        count = size_bytes // item_size
        source_record = source_array_records.get(name)
        destination = staging_dir / relative_path
        copy_record = _copy_artifact(source_path, destination)
        if isinstance(source_record, dict):
            if "count" in source_record:
                try:
                    manifest_count = int(source_record["count"])
                except (TypeError, ValueError) as exc:
                    raise RuntimeIndexError(
                        f"Invalid source sampling count for {name}"
                    ) from exc
                if manifest_count != count:
                    raise RuntimeIndexError(
                        f"Source sampling manifest count mismatch for {name}: "
                        f"{manifest_count} != {count}"
                    )
            source_sha256 = source_record.get("sha256")
            if source_sha256 is not None:
                if (
                    not isinstance(source_sha256, str)
                    or len(source_sha256) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in source_sha256.lower()
                    )
                ):
                    raise RuntimeIndexError(
                        f"Invalid source sampling SHA-256 for {name}"
                    )
                if source_sha256.lower() != copy_record["sha256"]:
                    raise RuntimeIndexError(
                        f"Source sampling SHA-256 mismatch for {name}: "
                        f"{copy_record['sha256']} != {source_sha256}"
                    )
            source_dtype = source_record.get("dtype")
            if source_dtype is not None and source_dtype != dtype:
                raise RuntimeIndexError(
                    f"Source sampling dtype mismatch for {name}: "
                    f"{source_dtype!r} != {dtype!r}"
                )
            source_endian = source_record.get("endian")
            if source_endian is not None and source_endian != "little":
                raise RuntimeIndexError(
                    f"Source sampling endian mismatch for {name}: "
                    f"{source_endian!r} != 'little'"
                )
        arrays[name] = {
            **copy_record,
            "dtype": dtype,
            "endian": "little",
            "count": count,
        }
        counts[name] = count

    if len(set(counts.values())) != 1:
        raise RuntimeIndexError(f"Sampling arrays have different counts: {counts}")
    cluster_count = next(iter(counts.values()))
    if cluster_count == 0:
        raise RuntimeIndexError("Sampling arrays contain no clusters")
    return {
        "alignment": (
            "Array index i identifies one cluster. Members occupy "
            "[first_row_ids[i], first_row_ids[i] + cluster_sizes[i])."
        ),
        "arrays": arrays,
    }, cluster_count


def _validate_sampling_intervals(directory: Path, *, row_count: int, cluster_count: int) -> None:
    first_path = directory / _SAMPLING_SPECS["first_row_ids"][0]
    size_path = directory / _SAMPLING_SPECS["cluster_sizes"][0]
    with first_path.open("rb") as first_file, size_path.open("rb") as size_file:
        first_map = mmap.mmap(first_file.fileno(), 0, access=mmap.ACCESS_READ)
        size_map = mmap.mmap(size_file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            expected_first = 0
            for ordinal in range(cluster_count):
                first_row = struct.unpack_from("<Q", first_map, ordinal * 8)[0]
                cluster_size = struct.unpack_from("<I", size_map, ordinal * 4)[0]
                if cluster_size == 0:
                    raise RuntimeIndexError(f"Cluster ordinal {ordinal} has zero members")
                if first_row != expected_first:
                    raise RuntimeIndexError(
                        f"Cluster ordinal {ordinal} starts at row {first_row}, expected {expected_first}"
                    )
                expected_first += cluster_size
                if expected_first > row_count:
                    raise RuntimeIndexError(
                        f"Cluster ordinal {ordinal} extends beyond row count {row_count}"
                    )
            if expected_first != row_count:
                raise RuntimeIndexError(
                    f"Sampling intervals cover {expected_first} rows, runtime index has {row_count}"
                )
        finally:
            first_map.close()
            size_map.close()


def _prepare_destination(destination: Path, *, overwrite: bool) -> None:
    if os.path.lexists(destination):
        if not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing runtime-index path: {destination}"
            )
        if destination.is_symlink() or not destination.is_dir():
            raise FileExistsError(f"--overwrite requires an existing directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)


def _publish_staging(staging: Path, destination: Path, *, overwrite: bool) -> None:
    if not os.path.lexists(destination):
        os.replace(staging, destination)
        return
    if not overwrite:
        raise FileExistsError(f"Runtime-index path appeared during build: {destination}")

    backup = destination.parent / f".{destination.name}.old.{os.getpid()}.{uuid.uuid4().hex}"
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def build_runtime_distillation_index(
    *,
    input_index: Path | str,
    output_dir: Path | str,
    roots_json: Path | str | None = None,
    sampling_dir: Path | str | None = None,
    overwrite: bool = False,
    progress_every_rows: int = 1_000_000,
) -> dict[str, object]:
    """Stream an authoritative CSV catalog into an immutable runtime index."""

    if progress_every_rows < 0:
        raise ValueError("progress_every_rows must be non-negative")
    input_path = Path(input_index).expanduser().resolve(strict=True)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input index is not a file: {input_path}")
    catalog_dir = input_path.parent
    roots_path = (
        Path(roots_json).expanduser().resolve(strict=True)
        if roots_json is not None
        else catalog_dir / "roots.json"
    )
    source_sampling_dir = (
        Path(sampling_dir).expanduser().resolve(strict=True)
        if sampling_dir is not None
        else catalog_dir / "sampling"
    )
    if not roots_path.is_file():
        raise FileNotFoundError(f"Roots document is not a file: {roots_path}")
    if not source_sampling_dir.is_dir():
        raise NotADirectoryError(f"Sampling path is not a directory: {source_sampling_dir}")

    destination = Path(output_dir).expanduser().resolve(strict=False)
    if destination == catalog_dir:
        raise ValueError("Runtime output directory must not replace the authoritative catalog directory")
    protected_sources = (input_path, roots_path, source_sampling_dir)
    for source_path in protected_sources:
        if source_path == destination or destination in source_path.parents:
            raise ValueError(
                "Runtime output directory must not contain an authoritative "
                f"source path: output={destination}, source={source_path}"
            )
    _prepare_destination(destination, overwrite=overwrite)
    staging = destination.parent / f".{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    staging.mkdir()

    started = time.monotonic()
    try:
        roots_source_stat = _stat_document(roots_path)
        roots_source_sha256 = _sha256_file(roots_path)
        roots_document, root_to_ordinal = _load_and_normalize_roots(roots_path)
        if not _same_stat(roots_source_stat, roots_path.stat()):
            raise RuntimeIndexError(
                f"Roots document changed while it was being read: {roots_path}"
            )
        root_ids = [record["root_id"] for record in roots_document["roots"]]
        roots_artifact = _write_json_artifact(staging / "roots.json", roots_document)

        source_sampling_manifest = _read_source_sampling_manifest(catalog_dir)
        sampling_manifest, cluster_count = _copy_sampling_arrays(
            sampling_dir=source_sampling_dir,
            staging_dir=staging,
            source_sampling_manifest=source_sampling_manifest,
        )

        source_stat = _stat_document(input_path)
        source_hasher = hashlib.sha256()
        row_count = 0
        relpath_bytes = 0
        with ExitStack() as stack:
            offsets = stack.enter_context(_BufferedArtifactWriter(staging / "row_offsets.u64"))
            roots = stack.enter_context(_BufferedArtifactWriter(staging / "source_root_ids.u8"))
            relpaths = stack.enter_context(_BufferedArtifactWriter(staging / "cif_relpaths.bin"))
            offsets.append(struct.pack("<Q", 0))

            with _open_index_text(input_path, source_hasher) as input_file:
                reader = csv.reader(input_file)
                try:
                    header = next(reader)
                except StopIteration as exc:
                    raise RuntimeIndexError(f"Input index is empty: {input_path}") from exc
                required = ("row_id", "source_root_id", "cif_relpath")
                missing = [field for field in required if field not in header]
                if missing:
                    raise RuntimeIndexError(f"Input index is missing required columns: {', '.join(missing)}")
                if len(set(header)) != len(header):
                    raise RuntimeIndexError("Input index contains duplicate column names")
                row_id_column, root_column, relpath_column = (header.index(field) for field in required)
                minimum_columns = max(row_id_column, root_column, relpath_column) + 1

                for line_number, row in enumerate(reader, start=2):
                    if len(row) < minimum_columns:
                        raise RuntimeIndexError(
                            f"CSV row {line_number} has {len(row)} columns; expected at least {minimum_columns}"
                        )
                    try:
                        row_id = int(row[row_id_column])
                    except ValueError as exc:
                        raise RuntimeIndexError(
                            f"CSV row {line_number} has invalid row_id {row[row_id_column]!r}"
                        ) from exc
                    if row_id != row_count:
                        raise RuntimeIndexError(
                            f"row_id must be contiguous: CSV row {line_number} has {row_id}, expected {row_count}"
                        )
                    root_id = row[root_column].strip()
                    try:
                        root_ordinal = root_to_ordinal[root_id]
                    except KeyError as exc:
                        raise RuntimeIndexError(
                            f"CSV row {line_number} references unknown source_root_id {root_id!r}"
                        ) from exc
                    relpath = _safe_relative_path(row[relpath_column], suffix=".cif")
                    encoded = relpath.as_posix().encode("utf-8")
                    if not encoded:
                        raise RuntimeIndexError(f"CSV row {line_number} has an empty CIF path")
                    relpath_bytes += len(encoded)
                    if relpath_bytes > _UINT64_MAX:
                        raise RuntimeIndexError("Concatenated CIF paths exceed uint64 offsets")

                    roots.append(bytes((root_ordinal,)))
                    relpaths.append(encoded)
                    offsets.append(struct.pack("<Q", relpath_bytes))
                    row_count += 1

                    if progress_every_rows and row_count % progress_every_rows == 0:
                        elapsed = max(time.monotonic() - started, 1e-9)
                        print(
                            "[runtime-index] "
                            f"rows={row_count:,} rate={row_count / elapsed:,.0f} rows/s",
                            file=sys.stderr,
                            flush=True,
                        )

        if row_count == 0:
            raise RuntimeIndexError("Input index contains no data rows")
        if not _same_stat(source_stat, input_path.stat()):
            raise RuntimeIndexError(f"Input index changed while it was being read: {input_path}")

        _validate_sampling_intervals(staging, row_count=row_count, cluster_count=cluster_count)
        artifacts = {
            "row_offsets": _artifact_record(
                path="row_offsets.u64",
                size_bytes=offsets.size_bytes,
                sha256=offsets.sha256,
                dtype="uint64",
                count=row_count + 1,
            ),
            "source_root_ids": _artifact_record(
                path="source_root_ids.u8",
                size_bytes=roots.size_bytes,
                sha256=roots.sha256,
                dtype="uint8",
                count=row_count,
            ),
            "cif_relpaths": _artifact_record(
                path="cif_relpaths.bin",
                size_bytes=relpaths.size_bytes,
                sha256=relpaths.sha256,
                dtype="bytes",
                count=relpath_bytes,
            ),
            "roots": roots_artifact,
        }
        manifest: dict[str, object] = {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "format": "protenix_runtime_distillation_index",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                **source_stat,
                "sha256": source_hasher.hexdigest(),
                "roots_json": {
                    **roots_source_stat,
                    "sha256": roots_source_sha256,
                },
                "sampling_dir": str(source_sampling_dir),
            },
            "row_count": row_count,
            "cluster_count": cluster_count,
            "root_ids": root_ids,
            "artifacts": artifacts,
            "sampling": sampling_manifest,
            "path_rules": {
                "cif": "source root / cif_relpath",
                "a3m": "replace the final .cif suffix with .a3m",
                "bioassembly": "CIF parent / bioassembly / <CIF stem>.pkl.gz",
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        _write_json_artifact(staging / "manifest.json", manifest)
        _validate_runtime_index(staging, verify_checksums=True)
        _publish_staging(staging, destination, overwrite=overwrite)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _expect_artifact(
    directory: Path,
    record: object,
    *,
    expected_size: int | None = None,
    expected_count: int | None = None,
    count_matches_size: bool = False,
    expected_dtype: str | None = None,
    expected_endian: str | None = None,
    verify_checksum: bool,
) -> Path:
    if not isinstance(record, dict):
        raise RuntimeIndexError("Manifest artifact record must be an object")
    path = _safe_artifact_path(directory, record.get("path", ""))
    if not path.is_file():
        raise RuntimeIndexError(f"Missing runtime artifact: {path}")
    try:
        manifest_size = int(record["size_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeIndexError(f"Invalid size_bytes for runtime artifact {path}") from exc
    actual_size = path.stat().st_size
    if actual_size != manifest_size:
        raise RuntimeIndexError(
            f"Runtime artifact size mismatch for {path}: {actual_size} != {manifest_size}"
        )
    if expected_size is not None and actual_size != expected_size:
        raise RuntimeIndexError(
            f"Runtime artifact format size mismatch for {path}: {actual_size} != {expected_size}"
        )
    if count_matches_size:
        expected_count = actual_size
    if expected_count is not None:
        try:
            manifest_count = int(record["count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeIndexError(f"Invalid count for runtime artifact {path}") from exc
        if manifest_count != expected_count:
            raise RuntimeIndexError(
                f"Runtime artifact count mismatch for {path}: "
                f"{manifest_count} != {expected_count}"
            )
    if expected_dtype is not None and record.get("dtype") != expected_dtype:
        raise RuntimeIndexError(
            f"Runtime artifact dtype mismatch for {path}: "
            f"{record.get('dtype')!r} != {expected_dtype!r}"
        )
    if expected_endian is not None and record.get("endian") != expected_endian:
        raise RuntimeIndexError(
            f"Runtime artifact endian mismatch for {path}: "
            f"{record.get('endian')!r} != {expected_endian!r}"
        )
    if verify_checksum:
        expected_sha256 = record.get("sha256")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_sha256.lower()
            )
        ):
            raise RuntimeIndexError(
                f"Invalid SHA-256 for runtime artifact {path}"
            )
        actual_sha256 = _sha256_file(path)
        if expected_sha256.lower() != actual_sha256:
            raise RuntimeIndexError(
                f"Runtime artifact SHA-256 mismatch for {path}: {actual_sha256} != {expected_sha256}"
            )
    return path


def _validate_runtime_index(
    directory: Path,
    *,
    verify_checksums: bool,
) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeIndexError(f"Cannot read runtime manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeIndexError("Runtime manifest must be a JSON object")
    if manifest.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        raise RuntimeIndexError(f"Unsupported runtime schema version: {manifest.get('schema_version')!r}")
    if manifest.get("format") != "protenix_runtime_distillation_index":
        raise RuntimeIndexError(f"Unsupported runtime-index format: {manifest.get('format')!r}")
    try:
        row_count = int(manifest["row_count"])
        cluster_count = int(manifest["cluster_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeIndexError("Runtime manifest has invalid row or cluster count") from exc
    if row_count <= 0 or cluster_count <= 0:
        raise RuntimeIndexError("Runtime row and cluster counts must be positive")

    root_ids = manifest.get("root_ids")
    if (
        not isinstance(root_ids, list)
        or not root_ids
        or len(root_ids) > 256
        or not all(isinstance(root_id, str) and root_id for root_id in root_ids)
        or len(set(root_ids)) != len(root_ids)
    ):
        raise RuntimeIndexError("Runtime manifest has an invalid root_ids table")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeIndexError("Runtime manifest is missing artifacts")
    offsets_path = _expect_artifact(
        directory,
        artifacts.get("row_offsets"),
        expected_size=(row_count + 1) * 8,
        expected_count=row_count + 1,
        expected_dtype="uint64",
        expected_endian="little",
        verify_checksum=verify_checksums,
    )
    root_ordinals_path = _expect_artifact(
        directory,
        artifacts.get("source_root_ids"),
        expected_size=row_count,
        expected_count=row_count,
        expected_dtype="uint8",
        expected_endian="little",
        verify_checksum=verify_checksums,
    )
    relpaths_path = _expect_artifact(
        directory,
        artifacts.get("cif_relpaths"),
        count_matches_size=True,
        expected_dtype="bytes",
        expected_endian="not_applicable",
        verify_checksum=verify_checksums,
    )
    roots_path = _expect_artifact(
        directory,
        artifacts.get("roots"),
        verify_checksum=verify_checksums,
    )
    normalized_roots, root_to_ordinal = _load_and_normalize_roots(roots_path)
    if list(root_to_ordinal) != root_ids:
        raise RuntimeIndexError("Runtime roots.json order does not match manifest root_ids")
    del normalized_roots

    with offsets_path.open("rb") as offsets_file:
        offsets_map = mmap.mmap(offsets_file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            previous = struct.unpack_from("<Q", offsets_map, 0)[0]
            if previous != 0:
                raise RuntimeIndexError(f"First row offset must be zero, got {previous}")
            for row_id in range(row_count):
                current = struct.unpack_from("<Q", offsets_map, (row_id + 1) * 8)[0]
                if current <= previous:
                    raise RuntimeIndexError(
                        f"Row offsets must be strictly increasing at row {row_id}: {previous}, {current}"
                    )
                previous = current
            if previous != relpaths_path.stat().st_size:
                raise RuntimeIndexError(
                    f"Final row offset {previous} does not match CIF-path bytes {relpaths_path.stat().st_size}"
                )
        finally:
            offsets_map.close()

    with root_ordinals_path.open("rb") as roots_file:
        while payload := roots_file.read(_COPY_CHUNK_SIZE):
            if max(payload) >= len(root_ids):
                raise RuntimeIndexError("source_root_ids.u8 contains an unknown root ordinal")

    sampling = manifest.get("sampling")
    arrays = sampling.get("arrays") if isinstance(sampling, dict) else None
    if not isinstance(arrays, dict):
        raise RuntimeIndexError("Runtime manifest is missing sampling arrays")
    for name, (_relative, item_size, dtype) in _SAMPLING_SPECS.items():
        record = arrays.get(name)
        _expect_artifact(
            directory,
            record,
            expected_size=cluster_count * item_size,
            expected_count=cluster_count,
            expected_dtype=dtype,
            expected_endian="little",
            verify_checksum=verify_checksums,
        )
    _validate_sampling_intervals(directory, row_count=row_count, cluster_count=cluster_count)
    return manifest


def verify_runtime_distillation_index(
    runtime_dir: Path | str,
    *,
    verify_source: bool = False,
) -> dict[str, object]:
    """Fully verify runtime artifacts and optionally re-hash the source CSV."""

    directory = Path(runtime_dir).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise NotADirectoryError(f"Runtime-index path is not a directory: {directory}")
    manifest = _validate_runtime_index(directory, verify_checksums=True)
    if verify_source:
        source = manifest.get("source")
        if not isinstance(source, dict):
            raise RuntimeIndexError("Runtime manifest has no source record")
        source_path = Path(str(source.get("path", "")))
        if not source_path.is_file():
            raise RuntimeIndexError(f"Recorded source index is missing: {source_path}")
        stat = source_path.stat()
        if source.get("size") != stat.st_size or source.get("mtime_ns") != stat.st_mtime_ns:
            raise RuntimeIndexError(f"Recorded source index stat changed: {source_path}")
        sha256 = _sha256_file(source_path)
        if sha256 != source.get("sha256"):
            raise RuntimeIndexError(f"Recorded source index SHA-256 changed: {source_path}")
    return manifest


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build and atomically publish a runtime index")
    build_parser.add_argument("--input-index", type=Path, required=True)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--roots-json", type=Path)
    build_parser.add_argument("--sampling-dir", type=Path)
    build_parser.add_argument("--overwrite", action="store_true")
    build_parser.add_argument("--progress-every-rows", type=int, default=1_000_000)

    verify_parser = subparsers.add_parser("verify", help="Verify a published runtime index")
    verify_parser.add_argument("--runtime-dir", type=Path, required=True)
    verify_parser.add_argument("--verify-source", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    if args.command == "build":
        manifest = build_runtime_distillation_index(
            input_index=args.input_index,
            output_dir=args.output_dir,
            roots_json=args.roots_json,
            sampling_dir=args.sampling_dir,
            overwrite=args.overwrite,
            progress_every_rows=args.progress_every_rows,
        )
    else:
        manifest = verify_runtime_distillation_index(
            args.runtime_dir,
            verify_source=args.verify_source,
        )
    print(
        json.dumps(
            {
                "runtime_dir": str(
                    args.output_dir if args.command == "build" else args.runtime_dir
                ),
                "row_count": manifest["row_count"],
                "cluster_count": manifest["cluster_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
