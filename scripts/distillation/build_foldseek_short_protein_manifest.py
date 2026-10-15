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

"""Build a streaming Foldseek input manifest from conversion audit indices.

The generated path files are relative to ``OUTPUT_DIR/roots``.  Each source
root is exposed there as ``roots/<root_id>`` via a verified symlink, so a
consumer can change directory to ``OUTPUT_DIR/roots`` and resolve every line
without copying or flattening the structure files.

This builder never opens a CIF.  It streams the merged conversion audit once
and keeps memory proportional to the number of source roots, not input rows.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator, Mapping, TextIO


EXPECTED_ARTIFACT_ROLE = "distillation_bioassembly_conversion_audit"
REQUIRED_INDEX_FIELDS = (
    "source_root_id",
    "cif_relpath",
    "canonical_length",
)
ROOT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CONTROL_CHARACTERS = frozenset("\0\t\r\n")
SCHEMA_VERSION = 1


class FoldseekManifestError(RuntimeError):
    """Raised when an input violates the Foldseek manifest contract."""


@dataclass(frozen=True, slots=True)
class SourceRoot:
    """One source tree addressed by its compact root ID."""

    root_id: str
    path: Path


@dataclass(slots=True)
class RootCounts:
    """Streaming counters for one source root."""

    input_rows: int = 0
    selected_rows: int = 0
    pilot_rows: int = 0


class _ContentDigestWriter:
    """Write bytes while hashing the uncompressed logical content."""

    def __init__(self, stream: BinaryIO):
        self._stream = stream
        self._digest = hashlib.sha256()
        self.bytes_written = 0
        self.lines_written = 0

    def write_line(self, line: str) -> None:
        payload = line.encode("utf-8")
        self._stream.write(payload)
        self._digest.update(payload)
        self.bytes_written += len(payload)
        self.lines_written += 1

    @property
    def content_sha256(self) -> str:
        return self._digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _path_stat_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _open_index(path: Path) -> Iterator[TextIO]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("rt", encoding="utf-8", newline="")


def _validate_root_id(root_id: str, *, context: str) -> str:
    if not ROOT_ID_PATTERN.fullmatch(root_id):
        raise FoldseekManifestError(
            f"{context} has unsafe root_id {root_id!r}; expected one portable "
            "path component matching [A-Za-z0-9][A-Za-z0-9._-]*"
        )
    return root_id


def _load_roots(path: Path) -> list[SourceRoot]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FoldseekManifestError(f"Cannot read roots document {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("roots"), list):
        raise FoldseekManifestError(f"{path} must contain a JSON object with a roots list")

    roots: list[SourceRoot] = []
    seen_ids: set[str] = set()
    for ordinal, record in enumerate(payload["roots"]):
        if not isinstance(record, dict):
            raise FoldseekManifestError(f"roots[{ordinal}] is not a JSON object")
        try:
            root_id = _validate_root_id(str(record["root_id"]), context=f"roots[{ordinal}]")
            raw_path = record["path"]
        except KeyError as exc:
            raise FoldseekManifestError(f"roots[{ordinal}] is missing {exc.args[0]!r}") from exc
        if root_id in seen_ids:
            raise FoldseekManifestError(f"Duplicate root_id in {path}: {root_id!r}")
        if not isinstance(raw_path, str) or not raw_path:
            raise FoldseekManifestError(f"roots[{ordinal}].path must be a non-empty string")
        try:
            source_path = Path(raw_path).expanduser().resolve(strict=True)
        except OSError as exc:
            raise FoldseekManifestError(
                f"Source root {root_id!r} cannot be resolved: {raw_path!r}"
            ) from exc
        if not source_path.is_dir():
            raise FoldseekManifestError(
                f"Source root {root_id!r} is not a directory: {source_path}"
            )
        seen_ids.add(root_id)
        roots.append(SourceRoot(root_id=root_id, path=source_path))
    if not roots:
        raise FoldseekManifestError(f"{path} contains no source roots")
    return roots


def _validate_cif_relpath(raw_path: str, *, line_number: int) -> str:
    if not raw_path or any(character in raw_path for character in CONTROL_CHARACTERS):
        raise FoldseekManifestError(
            f"CSV row {line_number} has an empty or control-character-containing cif_relpath"
        )
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise FoldseekManifestError(
            f"CSV row {line_number} has unsafe cif_relpath {raw_path!r}"
        )
    normalized = relative.as_posix()
    if normalized in {".", ""} or normalized != raw_path or not normalized.endswith(".cif"):
        raise FoldseekManifestError(
            f"CSV row {line_number} has non-normalized .cif path {raw_path!r}"
        )
    return normalized


def _is_pilot_path(
    relative_path: str,
    *,
    seed: str,
    modulus: int,
    buckets: int,
) -> bool:
    digest = hashlib.sha256()
    digest.update(seed.encode("utf-8"))
    digest.update(b"\0")
    digest.update(relative_path.encode("utf-8"))
    bucket = int.from_bytes(digest.digest()[:8], byteorder="big") % modulus
    return bucket < buckets


def _create_and_verify_root_links(directory: Path, roots: list[SourceRoot]) -> None:
    directory.mkdir()
    for root in roots:
        link = directory / root.root_id
        link.symlink_to(root.path, target_is_directory=True)
        if not link.is_symlink():
            raise FoldseekManifestError(f"Failed to create source-root symlink: {link}")
        try:
            resolved = link.resolve(strict=True)
        except OSError as exc:
            raise FoldseekManifestError(f"Broken source-root symlink: {link}") from exc
        if resolved != root.path:
            raise FoldseekManifestError(
                f"Source-root symlink resolves incorrectly: {link} -> {resolved}, "
                f"expected {root.path}"
            )


def verify_root_links(output_dir: Path | str, roots_json: Path | str) -> None:
    """Verify a published manifest's root symlinks against ``roots.json``."""

    destination = Path(output_dir).expanduser().resolve(strict=True)
    roots_path = Path(roots_json).expanduser().resolve(strict=True)
    for root in _load_roots(roots_path):
        link = destination / "roots" / root.root_id
        if not link.is_symlink():
            raise FoldseekManifestError(f"Expected source-root symlink is missing: {link}")
        if link.resolve(strict=True) != root.path:
            raise FoldseekManifestError(
                f"Source-root symlink target mismatch: {link} -> "
                f"{link.resolve(strict=True)}, expected {root.path}"
            )


def _normalize_destination(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate.parent.mkdir(parents=True, exist_ok=True)
    parent = candidate.parent.resolve(strict=True)
    return parent / candidate.name


def _publish_directory(staging: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        os.replace(staging, destination)
        return
    if destination.is_symlink() or not destination.is_dir():
        raise FoldseekManifestError(
            f"Refusing to replace a non-directory destination: {destination}"
        )
    if not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {destination}; pass --overwrite to replace it"
        )

    backup = destination.parent / (
        f".{destination.name}.old.{os.getpid()}.{uuid.uuid4().hex}"
    )
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    shutil.rmtree(backup)


def _artifact_stats(
    path: Path,
    writer: _ContentDigestWriter,
    *,
    relative_to: Path,
) -> dict[str, object]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "lines": writer.lines_written,
        "uncompressed_bytes": writer.bytes_written,
        "content_sha256": writer.content_sha256,
        "size_bytes": path.stat().st_size,
    }


def build_foldseek_short_protein_manifest(
    *,
    input_index: Path | str,
    roots_json: Path | str,
    output_dir: Path | str,
    min_length: int = 1,
    max_length: int = 200,
    pilot_hash_modulus: int = 1_000,
    pilot_hash_buckets: int = 1,
    pilot_hash_seed: str = "protenix-foldseek-short-v1",
    output_prefix: str = "short_le200",
    compresslevel: int = 1,
    progress_every_rows: int = 1_000_000,
    overwrite: bool = False,
) -> dict[str, object]:
    """Stream a conversion audit into relative Foldseek manifests.

    The compact mapping is a headerless gzip TSV with columns
    ``selected_ordinal``, ``root_id/cif_relpath``, and ``canonical_length``.
    """

    if min_length <= 0:
        raise ValueError("min_length must be positive")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if min_length > max_length:
        raise ValueError("min_length must not exceed max_length")
    if pilot_hash_modulus <= 0:
        raise ValueError("pilot_hash_modulus must be positive")
    if not 0 <= pilot_hash_buckets <= pilot_hash_modulus:
        raise ValueError("pilot_hash_buckets must be between zero and pilot_hash_modulus")
    if not pilot_hash_seed or any(
        character in pilot_hash_seed for character in CONTROL_CHARACTERS
    ):
        raise ValueError("pilot_hash_seed must be non-empty and contain no control characters")
    if not ROOT_ID_PATTERN.fullmatch(output_prefix):
        raise ValueError(
            "output_prefix must match [A-Za-z0-9][A-Za-z0-9._-]*"
        )
    if not 0 <= compresslevel <= 9:
        raise ValueError("compresslevel must be between zero and nine")
    if progress_every_rows < 0:
        raise ValueError("progress_every_rows must be non-negative")

    index_path = Path(input_index).expanduser().resolve(strict=True)
    roots_path = Path(roots_json).expanduser().resolve(strict=True)
    if not index_path.is_file():
        raise FileNotFoundError(f"Input index is not a file: {index_path}")
    if not roots_path.is_file():
        raise FileNotFoundError(f"Roots document is not a file: {roots_path}")
    destination = _normalize_destination(output_dir)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {destination}; pass --overwrite to replace it"
        )

    roots_stat_before = _path_stat_identity(roots_path)
    index_stat_before = _path_stat_identity(index_path)
    roots_sha256 = _sha256_file(roots_path)
    roots = _load_roots(roots_path)
    root_by_id: Mapping[str, SourceRoot] = {root.root_id: root for root in roots}
    root_counts = {root.root_id: RootCounts() for root in roots}

    staging = destination.parent / (
        f".{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    staging.mkdir()
    started_monotonic = time.monotonic()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    input_rows = 0
    selected_rows = 0
    excluded_too_short = 0
    excluded_too_long = 0
    pilot_rows = 0
    selected_length_sum = 0
    selected_length_min: int | None = None
    selected_length_max: int | None = None

    paths_path = staging / f"{output_prefix}.paths.tsv"
    mapping_path = staging / f"{output_prefix}.mapping.tsv.gz"
    pilot_path = staging / f"{output_prefix}.pilot.paths.tsv"
    stats_path = staging / f"{output_prefix}.stats.json"

    try:
        _create_and_verify_root_links(staging / "roots", roots)
        with ExitStack() as stack:
            paths_raw = stack.enter_context(paths_path.open("wb", buffering=1024 * 1024))
            pilot_raw = stack.enter_context(pilot_path.open("wb", buffering=1024 * 1024))
            mapping_raw = stack.enter_context(mapping_path.open("wb", buffering=1024 * 1024))
            mapping_gzip = stack.enter_context(
                gzip.GzipFile(
                    filename="",
                    mode="wb",
                    compresslevel=compresslevel,
                    fileobj=mapping_raw,
                    mtime=0,
                )
            )
            paths_writer = _ContentDigestWriter(paths_raw)
            pilot_writer = _ContentDigestWriter(pilot_raw)
            mapping_writer = _ContentDigestWriter(mapping_gzip)

            with _open_index(index_path) as source:
                reader = csv.reader(source, strict=True)
                try:
                    header = next(reader)
                except StopIteration as exc:
                    raise FoldseekManifestError(f"Input index is empty: {index_path}") from exc
                if len(set(header)) != len(header):
                    raise FoldseekManifestError("Input index contains duplicate column names")
                missing = [field for field in REQUIRED_INDEX_FIELDS if field not in header]
                if missing:
                    raise FoldseekManifestError(
                        f"Input index is missing required columns: {', '.join(missing)}"
                    )
                root_column = header.index("source_root_id")
                relpath_column = header.index("cif_relpath")
                length_column = header.index("canonical_length")
                role_column = (
                    header.index("artifact_role") if "artifact_role" in header else None
                )
                minimum_columns = max(
                    root_column,
                    relpath_column,
                    length_column,
                    role_column if role_column is not None else 0,
                ) + 1

                for line_number, row in enumerate(reader, start=2):
                    input_rows += 1
                    if len(row) < minimum_columns:
                        raise FoldseekManifestError(
                            f"CSV row {line_number} has {len(row)} columns; "
                            f"expected at least {minimum_columns}"
                        )
                    if (
                        role_column is not None
                        and row[role_column] != EXPECTED_ARTIFACT_ROLE
                    ):
                        raise FoldseekManifestError(
                            f"CSV row {line_number} has artifact_role "
                            f"{row[role_column]!r}, expected {EXPECTED_ARTIFACT_ROLE!r}"
                        )
                    root_id = row[root_column]
                    if root_id not in root_by_id:
                        raise FoldseekManifestError(
                            f"CSV row {line_number} references unknown source_root_id "
                            f"{root_id!r}"
                        )
                    cif_relpath = _validate_cif_relpath(
                        row[relpath_column],
                        line_number=line_number,
                    )
                    try:
                        canonical_length = int(row[length_column])
                    except ValueError as exc:
                        raise FoldseekManifestError(
                            f"CSV row {line_number} has invalid canonical_length "
                            f"{row[length_column]!r}"
                        ) from exc
                    if canonical_length <= 0:
                        raise FoldseekManifestError(
                            f"CSV row {line_number} has non-positive canonical_length "
                            f"{canonical_length}"
                        )

                    counts = root_counts[root_id]
                    counts.input_rows += 1
                    if canonical_length < min_length:
                        excluded_too_short += 1
                    elif canonical_length > max_length:
                        excluded_too_long += 1
                    else:
                        relative_path = f"{root_id}/{cif_relpath}"
                        paths_writer.write_line(f"{relative_path}\n")
                        mapping_writer.write_line(
                            f"{selected_rows}\t{relative_path}\t{canonical_length}\n"
                        )
                        counts.selected_rows += 1
                        selected_rows += 1
                        selected_length_sum += canonical_length
                        selected_length_min = (
                            canonical_length
                            if selected_length_min is None
                            else min(selected_length_min, canonical_length)
                        )
                        selected_length_max = (
                            canonical_length
                            if selected_length_max is None
                            else max(selected_length_max, canonical_length)
                        )
                        if _is_pilot_path(
                            relative_path,
                            seed=pilot_hash_seed,
                            modulus=pilot_hash_modulus,
                            buckets=pilot_hash_buckets,
                        ):
                            pilot_writer.write_line(f"{relative_path}\n")
                            counts.pilot_rows += 1
                            pilot_rows += 1

                    if progress_every_rows and input_rows % progress_every_rows == 0:
                        print(
                            f"rows={input_rows:,} selected={selected_rows:,} "
                            f"pilot={pilot_rows:,}",
                            file=sys.stderr,
                            flush=True,
                        )

        if _path_stat_identity(index_path) != index_stat_before:
            raise FoldseekManifestError(f"Input index changed while streaming: {index_path}")
        if _path_stat_identity(roots_path) != roots_stat_before:
            raise FoldseekManifestError(f"Roots document changed while streaming: {roots_path}")

        elapsed_seconds = time.monotonic() - started_monotonic
        stats: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "created_at": started_at,
            "elapsed_seconds": elapsed_seconds,
            "selection": {
                "canonical_length_operator": "closed_interval",
                "min_canonical_length": min_length,
                "max_canonical_length": max_length,
            },
            "pilot": {
                "algorithm": "sha256_first_u64_modulo",
                "seed": pilot_hash_seed,
                "modulus": pilot_hash_modulus,
                "selected_buckets": f"[0,{pilot_hash_buckets})",
                "nominal_fraction": pilot_hash_buckets / pilot_hash_modulus,
            },
            "input": {
                "index": str(index_path),
                "index_size_bytes": index_stat_before[2],
                "roots_json": str(roots_path),
                "roots_json_size_bytes": roots_stat_before[2],
                "roots_json_sha256": roots_sha256,
            },
            "counts": {
                "input_rows": input_rows,
                "selected_rows": selected_rows,
                "excluded_too_short": excluded_too_short,
                "excluded_too_long": excluded_too_long,
                "pilot_rows": pilot_rows,
                "selected_length_min": selected_length_min,
                "selected_length_max": selected_length_max,
                "selected_length_sum": selected_length_sum,
            },
            "by_root": {
                root_id: {
                    "input_rows": counts.input_rows,
                    "selected_rows": counts.selected_rows,
                    "pilot_rows": counts.pilot_rows,
                }
                for root_id, counts in root_counts.items()
            },
            "path_contract": {
                "base_directory": "roots",
                "paths_format": "UTF-8, one root_id/cif_relpath per line, no header",
                "mapping_format": (
                    "gzip UTF-8 TSV, no header: "
                    "selected_ordinal, root_id/cif_relpath, canonical_length"
                ),
                "root_symlinks_verified": True,
            },
            "outputs": {
                "paths": _artifact_stats(paths_path, paths_writer, relative_to=staging),
                "mapping": _artifact_stats(mapping_path, mapping_writer, relative_to=staging),
                "pilot_paths": _artifact_stats(
                    pilot_path,
                    pilot_writer,
                    relative_to=staging,
                ),
            },
        }
        stats_path.write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _publish_directory(staging, destination, overwrite=overwrite)
        verify_root_links(destination, roots_path)
        return stats
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream distillation_bioassembly_indices.csv.gz into an inclusive "
            "length interval "
            "Foldseek path manifests without opening CIF files."
        )
    )
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--roots-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--pilot-hash-modulus", type=int, default=1_000)
    parser.add_argument("--pilot-hash-buckets", type=int, default=1)
    parser.add_argument(
        "--pilot-hash-seed",
        default="protenix-foldseek-short-v1",
    )
    parser.add_argument("--output-prefix", default="short_le200")
    parser.add_argument("--compresslevel", type=int, default=1)
    parser.add_argument("--progress-every-rows", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    stats = build_foldseek_short_protein_manifest(
        input_index=args.input_index,
        roots_json=args.roots_json,
        output_dir=args.output_dir,
        min_length=args.min_length,
        max_length=args.max_length,
        pilot_hash_modulus=args.pilot_hash_modulus,
        pilot_hash_buckets=args.pilot_hash_buckets,
        pilot_hash_seed=args.pilot_hash_seed,
        output_prefix=args.output_prefix,
        compresslevel=args.compresslevel,
        progress_every_rows=args.progress_every_rows,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output_dir": str(
                    Path(args.output_dir).expanduser().absolute()
                ),
                "counts": stats["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
