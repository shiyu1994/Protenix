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

"""Build a clustered AFDB/MGnify distillation catalog.

The MMseqs two-column input is streamed one representative block at a time.
Only the exact CIF filename inventory is kept in memory.  No CIF is opened and
no per-CIF ``stat()`` call is made.  The resulting catalog retains every
available member and uses the original zero-based block ordinal as its stable
integer cluster ID.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import queue
import re
import shutil
import struct
import sys
import threading
import time
import uuid
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import IO, Iterable, Iterator, Sequence


STANDARD_INDEX_FIELDS = [
    "entity_1_id",
    "chain_1_id",
    "mol_1_type",
    "cluster_1_id",
    "entity_2_id",
    "chain_2_id",
    "mol_2_type",
    "cluster_2_id",
    "cluster_id",
    "pdb_id",
    "release_date",
    "num_tokens",
    "num_prot_chains",
    "resolution",
    "type",
    "mol_type_group",
    "sub_mol_1_type",
    "sub_mol_2_type",
    "eval_type",
]

PROVENANCE_INDEX_FIELDS = [
    "row_id",
    "source_db",
    "source_root_id",
    "cif_relpath",
    "source_protein_id",
    "cluster_size",
]

INDEX_FIELDS = STANDARD_INDEX_FIELDS + PROVENANCE_INDEX_FIELDS

CLUSTER_FIELDS = [
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

REMOVED_MEMBER_FIELDS = [
    "cluster_id",
    "source_representative_id",
    "source_protein_id",
    "reason",
]

DROPPED_CLUSTER_FIELDS = [
    "cluster_id",
    "source_representative_id",
    "original_size",
    "removed_size",
    "reason",
]

_ROOT_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_SOURCE_NAMESPACE_TO_DB = {
    "AFQ": "afdb",
    "MGN": "mgnify",
}
_CORE_COUNT_KEYS = (
    "input_rows",
    "input_clusters",
    "surviving_rows",
    "surviving_clusters",
    "removed_members",
    "dropped_clusters",
    "replaced_representatives",
    "cross_source_clusters",
)


class CatalogBuildError(RuntimeError):
    """Raised when an input invariant would make the catalog ambiguous."""


@dataclass(frozen=True, slots=True)
class SourceRoot:
    """One independently relocatable source tree."""

    root_id: str
    source_prefix: str
    path: Path

    @property
    def source_db(self) -> str:
        namespace = self.source_prefix[:-1]
        return _SOURCE_NAMESPACE_TO_DB.get(namespace, namespace.lower())


@dataclass(frozen=True, slots=True)
class InventoryEntry:
    """Exact location of one source protein CIF."""

    root_id: str
    source_db: str
    cif_relpath: str


@dataclass(frozen=True, slots=True)
class ClusterBlock:
    """One validated, contiguous MMseqs representative block."""

    cluster_id: int
    representative_id: str
    members: tuple[str, ...]
    first_line_number: int


def parse_source_root(spec: str) -> SourceRoot:
    """Parse ``ROOT_ID,SOURCE_PREFIX,PATH`` from the repeatable CLI option."""

    parts = spec.split(",", maxsplit=2)
    if len(parts) != 3:
        raise ValueError(
            "--source-root must be ROOT_ID,SOURCE_PREFIX,PATH; "
            f"received {spec!r}"
        )
    root_id, source_prefix, raw_path = (part.strip() for part in parts)
    if not root_id or not _ROOT_ID_RE.fullmatch(root_id):
        raise ValueError(f"Invalid source root ID: {root_id!r}")
    if not source_prefix:
        raise ValueError("SOURCE_PREFIX must not be empty")
    if any(char in source_prefix for char in "\t\r\n,"):
        raise ValueError(f"Invalid SOURCE_PREFIX: {source_prefix!r}")
    if not source_prefix.endswith("|"):
        source_prefix += "|"
    if source_prefix == "|":
        raise ValueError("SOURCE_PREFIX must contain a name before '|'")
    if not raw_path:
        raise ValueError("Source root path must not be empty")
    return SourceRoot(
        root_id=root_id,
        source_prefix=source_prefix,
        path=Path(raw_path).expanduser(),
    )


def _normalize_source_roots(source_roots: Sequence[SourceRoot]) -> list[SourceRoot]:
    if not source_roots:
        raise ValueError("At least one source root is required")

    normalized: list[SourceRoot] = []
    seen_root_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for source_root in source_roots:
        if not _ROOT_ID_RE.fullmatch(source_root.root_id):
            raise ValueError(f"Invalid source root ID: {source_root.root_id!r}")
        if source_root.root_id in seen_root_ids:
            raise ValueError(f"Duplicate source root ID: {source_root.root_id}")
        prefix = source_root.source_prefix
        if not prefix.endswith("|"):
            prefix += "|"
        if prefix == "|" or any(char in prefix for char in "\t\r\n,"):
            raise ValueError(f"Invalid SOURCE_PREFIX: {prefix!r}")

        path = source_root.path.expanduser().resolve(strict=True)
        if not path.is_dir():
            raise NotADirectoryError(f"Source root is not a directory: {path}")
        if path in seen_paths:
            raise ValueError(f"Duplicate source root path: {path}")

        seen_root_ids.add(source_root.root_id)
        seen_paths.add(path)
        normalized.append(
            SourceRoot(
                root_id=source_root.root_id,
                source_prefix=prefix,
                path=path,
            )
        )
    return normalized


def build_source_inventory(
    source_roots: Sequence[SourceRoot],
    *,
    scan_workers: int = 64,
) -> tuple[dict[str, InventoryEntry], dict[str, dict[str, int]]]:
    """Enumerate CIF names with a bounded parallel ``scandir`` work queue.

    Directory entries are inspected by name.  No CIF is opened and no explicit
    file ``stat()`` is issued.  Directory symlinks are not followed.
    """

    roots = _normalize_source_roots(source_roots)
    if scan_workers <= 0:
        raise ValueError(f"scan_workers must be positive, got {scan_workers}")

    inventory: dict[str, InventoryEntry] = {}
    counts: dict[str, Counter[str]] = {
        source_root.root_id: Counter() for source_root in roots
    }
    work_queue: queue.Queue[tuple[SourceRoot, Path, str] | None] = queue.Queue()
    lock = threading.Lock()
    scan_errors: list[str] = []
    duplicate_examples: list[tuple[str, InventoryEntry, InventoryEntry]] = []
    duplicate_count = 0

    for source_root in roots:
        work_queue.put((source_root, source_root.path, ""))

    def scan_worker() -> None:
        nonlocal duplicate_count
        while True:
            task = work_queue.get()
            if task is None:
                work_queue.task_done()
                return
            source_root, directory, relative_directory = task
            discovered: list[tuple[str, InventoryEntry]] = []
            child_directories = 0
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        name = entry.name
                        if name == "bioassembly":
                            continue
                        relative_path = (
                            f"{relative_directory}/{name}"
                            if relative_directory
                            else name
                        )
                        # Check the suffix before is_dir(). On filesystems
                        # without d_type, DirEntry.is_dir() would otherwise
                        # issue one lstat per CIF. Source trees must not contain
                        # directories whose names end in ".cif".
                        if name.endswith(".cif"):
                            protein_id = source_root.source_prefix + name[:-4]
                            discovered.append(
                                (
                                    protein_id,
                                    InventoryEntry(
                                        root_id=source_root.root_id,
                                        source_db=source_root.source_db,
                                        cif_relpath=PurePosixPath(
                                            relative_path
                                        ).as_posix(),
                                    ),
                                )
                            )
                            continue
                        try:
                            is_directory = entry.is_dir(follow_symlinks=False)
                        except OSError as exc:
                            with lock:
                                scan_errors.append(
                                    f"{entry.path}: {type(exc).__name__}: {exc}"
                                )
                            continue
                        if is_directory:
                            child_directories += 1
                            work_queue.put(
                                (
                                    source_root,
                                    Path(entry.path),
                                    relative_path,
                                )
                            )
                            continue
            except OSError as exc:
                with lock:
                    scan_errors.append(
                        f"{directory}: {type(exc).__name__}: {exc}"
                    )
            finally:
                with lock:
                    root_counts = counts[source_root.root_id]
                    root_counts["directories"] += 1
                    root_counts["child_directories"] += child_directories
                    for protein_id, candidate in discovered:
                        previous = inventory.get(protein_id)
                        if previous is None:
                            inventory[protein_id] = candidate
                            root_counts["cif_files"] += 1
                            continue
                        duplicate_count += 1
                        if len(duplicate_examples) < 20:
                            duplicate_examples.append(
                                (protein_id, previous, candidate)
                            )
                work_queue.task_done()

    threads = [
        threading.Thread(
            target=scan_worker,
            name=f"cif-scanner-{worker_index:03d}",
        )
        for worker_index in range(scan_workers)
    ]
    for thread in threads:
        thread.start()

    work_queue.join()
    for _ in threads:
        work_queue.put(None)
    work_queue.join()
    for thread in threads:
        thread.join()

    if scan_errors:
        examples = "\n".join(sorted(scan_errors)[:20])
        raise CatalogBuildError(
            f"Failed to scan {len(scan_errors)} directories/entries:\n{examples}"
        )
    if duplicate_count:
        examples = "\n".join(
            f"{protein_id}: "
            f"{old.root_id}/{old.cif_relpath} and "
            f"{new.root_id}/{new.cif_relpath}"
            for protein_id, old, new in sorted(
                duplicate_examples,
                key=lambda item: (
                    item[0],
                    item[1].root_id,
                    item[1].cif_relpath,
                    item[2].root_id,
                    item[2].cif_relpath,
                ),
            )
        )
        raise CatalogBuildError(
            f"Found {duplicate_count} duplicate prefixed protein IDs; "
            f"the first examples are:\n{examples}"
        )

    return inventory, {
        root_id: dict(sorted(root_counts.items()))
        for root_id, root_counts in counts.items()
    }


def iter_mmseqs_cluster_blocks(lines: Iterable[str]) -> Iterator[ClusterBlock]:
    """Validate and yield contiguous MMseqs ``representative<TAB>member`` blocks."""

    seen_representatives: set[str] = set()
    seen_members: set[str] = set()
    current_representative: str | None = None
    current_members: list[str] = []
    current_self_count = 0
    current_first_line = 0
    cluster_id = 0

    def finish_current() -> ClusterBlock:
        if current_representative is None:
            raise AssertionError("No current representative")
        if current_self_count != 1:
            raise CatalogBuildError(
                f"Cluster {cluster_id} ({current_representative!r}), starting "
                f"at line {current_first_line}, must contain exactly one "
                f"representative self row; observed {current_self_count}"
            )
        return ClusterBlock(
            cluster_id=cluster_id,
            representative_id=current_representative,
            members=tuple(current_members),
            first_line_number=current_first_line,
        )

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line
        if line.endswith("\n"):
            line = line[:-1]
        if line.endswith("\r"):
            line = line[:-1]
        columns = line.split("\t")
        if len(columns) != 2 or not columns[0] or not columns[1]:
            raise CatalogBuildError(
                f"Malformed cluster TSV row at line {line_number}: {line!r}"
            )
        representative, member = columns

        if representative != current_representative:
            if current_representative is not None:
                yield finish_current()
                cluster_id += 1
            if representative in seen_representatives:
                raise CatalogBuildError(
                    f"Representative {representative!r} reappears in a "
                    f"non-contiguous block at line {line_number}"
                )
            seen_representatives.add(representative)
            current_representative = representative
            current_members = []
            current_self_count = 0
            current_first_line = line_number

        if member in seen_members:
            raise CatalogBuildError(
                f"Member {member!r} occurs more than once; duplicate at "
                f"line {line_number}"
            )
        seen_members.add(member)
        current_members.append(member)
        if member == representative:
            current_self_count += 1

    if current_representative is not None:
        yield finish_current()


@contextmanager
def _open_deterministic_gzip_text(
    path: Path,
    *,
    compresslevel: int,
) -> Iterator[IO[str]]:
    with path.open("wb", buffering=1024 * 1024) as raw_file:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=compresslevel,
            fileobj=raw_file,
            mtime=0,
        ) as gzip_file:
            with io.TextIOWrapper(
                gzip_file,
                encoding="utf-8",
                newline="",
                write_through=False,
            ) as text_file:
                yield text_file


class _SamplingArrayWriter:
    _SPECS = {
        "cluster_ids": ("cluster_ids.u32", "<I", "uint32"),
        "first_row_ids": ("first_row_ids.u64", "<Q", "uint64"),
        "cluster_sizes": ("cluster_sizes.u32", "<I", "uint32"),
    }

    def __init__(self, directory: Path):
        self.directory = directory
        self._files: dict[str, IO[bytes]] = {}
        self._hashers = {
            name: hashlib.sha256() for name in self._SPECS
        }
        self.count = 0

    def __enter__(self) -> "_SamplingArrayWriter":
        self.directory.mkdir(parents=True, exist_ok=False)
        for name, (filename, _format, _dtype) in self._SPECS.items():
            self._files[name] = (self.directory / filename).open(
                "wb", buffering=1024 * 1024
            )
        return self

    def append(self, cluster_id: int, first_row_id: int, cluster_size: int) -> None:
        if not 0 <= cluster_id <= _UINT32_MAX:
            raise CatalogBuildError(
                f"cluster_id does not fit uint32: {cluster_id}"
            )
        if not 0 <= cluster_size <= _UINT32_MAX:
            raise CatalogBuildError(
                f"cluster_size does not fit uint32: {cluster_size}"
            )
        if not 0 <= first_row_id <= _UINT64_MAX:
            raise CatalogBuildError(
                f"first_row_id does not fit uint64: {first_row_id}"
            )
        values = {
            "cluster_ids": cluster_id,
            "first_row_ids": first_row_id,
            "cluster_sizes": cluster_size,
        }
        for name, value in values.items():
            payload = struct.pack(self._SPECS[name][1], value)
            self._files[name].write(payload)
            self._hashers[name].update(payload)
        self.count += 1

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for file_obj in self._files.values():
            file_obj.close()

    def manifest(self) -> dict[str, object]:
        arrays: dict[str, object] = {}
        for name, (filename, _format, dtype) in self._SPECS.items():
            arrays[name] = {
                "path": f"sampling/{filename}",
                "dtype": dtype,
                "endian": "little",
                "count": self.count,
                "sha256": self._hashers[name].hexdigest(),
            }
        return {
            "alignment": (
                "Array index i identifies the same surviving cluster. "
                "Catalog members occupy the contiguous row_id interval "
                "[first_row_ids[i], first_row_ids[i] + cluster_sizes[i])."
            ),
            "arrays": arrays,
        }


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(value, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def _target_is_nonempty(path: Path) -> bool:
    return path.exists() and (not path.is_dir() or next(path.iterdir(), None) is not None)


def _make_staging_directory(output_dir: Path) -> tuple[Path, bool]:
    output_existed_empty = False
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output path is not a directory: {output_dir}")
        if next(output_dir.iterdir(), None) is not None:
            raise FileExistsError(
                f"Refusing to overwrite non-empty output directory: {output_dir}"
            )
        output_existed_empty = True
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / (
        f".{output_dir.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    staging_dir.mkdir()
    return staging_dir, output_existed_empty


def _commit_staging_directory(
    staging_dir: Path,
    output_dir: Path,
    *,
    output_existed_empty: bool,
) -> None:
    if output_dir.exists():
        if _target_is_nonempty(output_dir):
            raise FileExistsError(
                f"Output directory became non-empty during the build: {output_dir}"
            )
        if not output_existed_empty:
            raise FileExistsError(
                f"Output directory appeared during the build: {output_dir}"
            )
        output_dir.rmdir()
    os.replace(staging_dir, output_dir)


def _index_row(
    *,
    row_id: int,
    cluster_id: int,
    cluster_size: int,
    protein_id: str,
    inventory_entry: InventoryEntry,
) -> list[str | int]:
    pdb_id = PurePosixPath(inventory_entry.cif_relpath).stem
    return [
        "1",
        "A",
        "prot",
        cluster_id,
        "",
        "",
        "",
        "",
        cluster_id,
        pdb_id,
        "9999-12-31",
        -1,
        1,
        "-1.0",
        "chain",
        "intra_prot",
        "prot",
        "",
        "intra_prot",
        row_id,
        inventory_entry.source_db,
        inventory_entry.root_id,
        inventory_entry.cif_relpath,
        protein_id,
        cluster_size,
    ]


def build_clustered_distillation_catalog(
    *,
    cluster_tsv: Path | str,
    source_roots: Sequence[SourceRoot],
    output_dir: Path | str,
    scan_workers: int = 64,
    compresslevel: int = 1,
    progress_every_clusters: int = 0,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, object]:
    """Build the immutable full catalog and return its manifest."""

    if not 0 <= compresslevel <= 9:
        raise ValueError(
            f"compresslevel must be between 0 and 9, got {compresslevel}"
        )
    if progress_every_clusters < 0:
        raise ValueError("progress_every_clusters must be non-negative")

    cluster_path = Path(cluster_tsv).expanduser().resolve(strict=True)
    if not cluster_path.is_file():
        raise FileNotFoundError(f"Cluster TSV is not a file: {cluster_path}")
    destination = Path(output_dir).expanduser().resolve(strict=False)
    staging_dir, output_existed_empty = _make_staging_directory(destination)
    started = time.monotonic()

    try:
        roots = _normalize_source_roots(source_roots)
        inventory, inventory_counts = build_source_inventory(
            roots,
            scan_workers=scan_workers,
        )
        if not inventory:
            raise CatalogBuildError("Source roots contain no .cif inventory entries")
        input_stat_before = cluster_path.stat()
        input_sha256 = hashlib.sha256()
        counts: Counter[str] = Counter()
        source_row_counts: Counter[str] = Counter()

        roots_document = {
            "schema_version": 1,
            "a3m_path_rule": (
                "Replace the final .cif suffix of cif_relpath with .a3m "
                "under the same source_root_id."
            ),
            "roots": [
                {
                    "root_id": source_root.root_id,
                    "source_prefix": source_root.source_prefix,
                    "source_db": source_root.source_db,
                    "path": str(source_root.path),
                    **inventory_counts[source_root.root_id],
                }
                for source_root in roots
            ],
        }
        _write_json(staging_dir / "roots.json", roots_document)

        with ExitStack() as stack:
            index_file = stack.enter_context(
                _open_deterministic_gzip_text(
                    staging_dir / "distillation_index.csv.gz",
                    compresslevel=compresslevel,
                )
            )
            clusters_file = stack.enter_context(
                _open_deterministic_gzip_text(
                    staging_dir / "clusters.csv.gz",
                    compresslevel=compresslevel,
                )
            )
            cleaned_file = stack.enter_context(
                _open_deterministic_gzip_text(
                    staging_dir / "cleaned_clusters.tsv.gz",
                    compresslevel=compresslevel,
                )
            )
            removed_file = stack.enter_context(
                _open_deterministic_gzip_text(
                    staging_dir / "removed_members.tsv.gz",
                    compresslevel=compresslevel,
                )
            )
            dropped_file = stack.enter_context(
                _open_deterministic_gzip_text(
                    staging_dir / "dropped_clusters.tsv.gz",
                    compresslevel=compresslevel,
                )
            )
            sampling_writer = stack.enter_context(
                _SamplingArrayWriter(staging_dir / "sampling")
            )

            index_writer = csv.writer(index_file, lineterminator="\n")
            cluster_writer = csv.writer(clusters_file, lineterminator="\n")
            removed_writer = csv.writer(
                removed_file, delimiter="\t", lineterminator="\n"
            )
            dropped_writer = csv.writer(
                dropped_file, delimiter="\t", lineterminator="\n"
            )
            index_writer.writerow(INDEX_FIELDS)
            cluster_writer.writerow(CLUSTER_FIELDS)
            removed_writer.writerow(REMOVED_MEMBER_FIELDS)
            dropped_writer.writerow(DROPPED_CLUSTER_FIELDS)

            def decoded_input_lines() -> Iterator[str]:
                with cluster_path.open("rb", buffering=1024 * 1024) as file_obj:
                    for raw_line in file_obj:
                        input_sha256.update(raw_line)
                        counts["input_rows"] += 1
                        try:
                            yield raw_line.decode("utf-8")
                        except UnicodeDecodeError as exc:
                            raise CatalogBuildError(
                                "Cluster TSV is not valid UTF-8 near input "
                                f"line {counts['input_rows']}"
                            ) from exc

            next_row_id = 0
            for block in iter_mmseqs_cluster_blocks(decoded_input_lines()):
                counts["input_clusters"] += 1
                original_representative = block.representative_id
                surviving: list[tuple[str, InventoryEntry]] = []
                missing: list[str] = []
                for member in block.members:
                    entry = inventory.get(member)
                    if entry is None:
                        missing.append(member)
                    else:
                        surviving.append((member, entry))

                original_size = len(block.members)
                surviving_size = len(surviving)
                removed_size = len(missing)
                for missing_member in missing:
                    removed_writer.writerow(
                        [
                            block.cluster_id,
                            original_representative,
                            missing_member,
                            "missing_from_inventory",
                        ]
                    )
                    counts["removed_members"] += 1

                if not surviving:
                    dropped_writer.writerow(
                        [
                            block.cluster_id,
                            original_representative,
                            original_size,
                            removed_size,
                            "no_surviving_members",
                        ]
                    )
                    counts["dropped_clusters"] += 1
                    continue

                if original_representative in inventory:
                    representative = original_representative
                else:
                    representative = surviving[0][0]
                    counts["replaced_representatives"] += 1

                first_row_id = next_row_id
                representative_row_id = -1
                source_dbs: set[str] = set()
                for member, entry in surviving:
                    row_id = next_row_id
                    next_row_id += 1
                    if member == representative:
                        representative_row_id = row_id
                    source_dbs.add(entry.source_db)
                    source_row_counts[entry.source_db] += 1
                    index_writer.writerow(
                        _index_row(
                            row_id=row_id,
                            cluster_id=block.cluster_id,
                            cluster_size=surviving_size,
                            protein_id=member,
                            inventory_entry=entry,
                        )
                    )
                    cleaned_file.write(f"{representative}\t{member}\n")

                if representative_row_id < 0:
                    raise AssertionError(
                        f"Selected representative is not a survivor: {representative}"
                    )

                flags: list[str] = []
                if representative != original_representative:
                    flags.append("representative_replaced")
                if removed_size:
                    flags.append("members_removed")
                if len(source_dbs) > 1:
                    flags.append("cross_source")
                    counts["cross_source_clusters"] += 1
                cluster_writer.writerow(
                    [
                        block.cluster_id,
                        original_representative,
                        representative,
                        original_size,
                        surviving_size,
                        removed_size,
                        first_row_id,
                        representative_row_id,
                        ";".join(flags),
                    ]
                )
                sampling_writer.append(
                    block.cluster_id,
                    first_row_id,
                    surviving_size,
                )
                counts["surviving_clusters"] += 1
                counts["surviving_rows"] += surviving_size

                if (
                    progress_every_clusters
                    and counts["input_clusters"] % progress_every_clusters == 0
                ):
                    elapsed = max(time.monotonic() - started, 1e-9)
                    print(
                        "[cluster-catalog] "
                        f"clusters={counts['input_clusters']:,} "
                        f"rows={counts['input_rows']:,} "
                        f"rate={counts['input_rows'] / elapsed:,.0f} rows/s",
                        file=sys.stderr,
                        flush=True,
                    )

        input_stat_after = cluster_path.stat()
        if (
            input_stat_after.st_size != input_stat_before.st_size
            or input_stat_after.st_mtime_ns != input_stat_before.st_mtime_ns
        ):
            raise CatalogBuildError(
                "Cluster TSV size or mtime changed while it was being read"
            )

        if counts["input_rows"] != (
            counts["surviving_rows"] + counts["removed_members"]
        ):
            raise AssertionError("Input/output member count invariant failed")
        if counts["input_clusters"] != (
            counts["surviving_clusters"] + counts["dropped_clusters"]
        ):
            raise AssertionError("Input/output cluster count invariant failed")
        if sampling_writer.count != counts["surviving_clusters"]:
            raise AssertionError("Sampling array count invariant failed")

        for count_name in _CORE_COUNT_KEYS:
            counts.setdefault(count_name, 0)
        if counts["input_rows"] == 0:
            raise CatalogBuildError("Cluster TSV contains no rows")
        if counts["surviving_rows"] == 0:
            raise CatalogBuildError(
                "No cluster members survived the source inventory join"
            )

        unused_inventory_entries = len(inventory) - counts["surviving_rows"]
        if unused_inventory_entries < 0:
            raise AssertionError("Inventory intersection count invariant failed")
        counts_document: dict[str, object] = {
            "inventory_entries": len(inventory),
            "unused_inventory_entries": unused_inventory_entries,
            **{key: counts[key] for key in _CORE_COUNT_KEYS},
            "source_rows": dict(sorted(source_row_counts.items())),
        }
        flattened_counts = {
            key: value
            for key, value in counts_document.items()
            if isinstance(value, int)
        }
        flattened_counts.update(
            {
                f"source_rows.{source_db}": value
                for source_db, value in source_row_counts.items()
            }
        )
        for key, expected in (expected_counts or {}).items():
            if key not in flattened_counts:
                available = ", ".join(sorted(flattened_counts))
                raise ValueError(
                    f"Unknown expected count {key!r}; available keys: {available}"
                )
            actual = flattened_counts[key]
            if actual != expected:
                raise CatalogBuildError(
                    f"Expected count {key}={expected:,}, observed {actual:,}"
                )

        elapsed_seconds = time.monotonic() - started
        manifest: dict[str, object] = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": {
                "path": str(cluster_path),
                "size": input_stat_before.st_size,
                "mtime_ns": input_stat_before.st_mtime_ns,
                "sha256": input_sha256.hexdigest(),
            },
            "roots_file": "roots.json",
            "index": {
                "path": "distillation_index.csv.gz",
                "fields": INDEX_FIELDS,
                "standard_field_count": len(STANDARD_INDEX_FIELDS),
                "row_order": (
                    "Original cluster block order, then original member order "
                    "after missing members are removed."
                ),
            },
            "clusters": {
                "path": "clusters.csv.gz",
                "fields": CLUSTER_FIELDS,
                "cluster_id_rule": (
                    "Original zero-based contiguous representative-block "
                    "ordinal; IDs of dropped clusters remain unused gaps."
                ),
            },
            "outputs": {
                "cleaned_clusters": "cleaned_clusters.tsv.gz",
                "removed_members": "removed_members.tsv.gz",
                "dropped_clusters": "dropped_clusters.tsv.gz",
            },
            "rules": {
                "availability": (
                    "A prefixed protein ID is available iff an exactly named "
                    ".cif was enumerated under an authoritative source root."
                ),
                "representative_replacement": (
                    "Keep the source representative when available; otherwise "
                    "select the first surviving member in original block order."
                ),
                "a3m_path": (
                    "Derived from cif_relpath by replacing final .cif with .a3m."
                ),
                "bioassembly_directories_skipped": True,
                "per_cif_stat": False,
            },
            "sampling": sampling_writer.manifest(),
            "counts": counts_document,
            "elapsed_seconds": elapsed_seconds,
        }
        _write_json(staging_dir / "manifest.json", manifest)
        _commit_staging_directory(
            staging_dir,
            destination,
            output_existed_empty=output_existed_empty,
        )
        return manifest
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter an MMseqs cluster TSV against exact CIF inventories and "
            "build an immutable distillation catalog."
        )
    )
    parser.add_argument("--cluster-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        action="append",
        required=True,
        metavar="ROOT_ID,SOURCE_PREFIX,PATH",
        help=(
            "Repeat for every authoritative source tree, for example "
            "'afdb,AFQ,/data/afdb/msas'."
        ),
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=min(64, max(1, os.cpu_count() or 1)),
    )
    parser.add_argument("--compresslevel", type=int, default=1)
    parser.add_argument(
        "--progress-every-clusters",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--expect-count",
        action="append",
        default=[],
        metavar="KEY=INTEGER",
        help=(
            "Refuse to publish unless a final manifest count has this exact "
            "value. Repeat as needed; source counts use source_rows.NAME."
        ),
    )
    return parser


def _parse_expected_counts(specs: Sequence[str]) -> dict[str, int]:
    expected_counts: dict[str, int] = {}
    for spec in specs:
        key, separator, raw_value = spec.partition("=")
        if not separator or not key or not raw_value:
            raise ValueError(
                f"--expect-count must be KEY=INTEGER; received {spec!r}"
            )
        if key in expected_counts:
            raise ValueError(f"Duplicate --expect-count key: {key}")
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"--expect-count value must be an integer: {spec!r}"
            ) from exc
        if value < 0:
            raise ValueError(
                f"--expect-count value must be non-negative: {spec!r}"
            )
        expected_counts[key] = value
    return expected_counts


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    source_roots = [parse_source_root(spec) for spec in args.source_root]
    manifest = build_clustered_distillation_catalog(
        cluster_tsv=args.cluster_tsv,
        source_roots=source_roots,
        output_dir=args.output_dir,
        scan_workers=args.scan_workers,
        compresslevel=args.compresslevel,
        progress_every_clusters=args.progress_every_clusters,
        expected_counts=_parse_expected_counts(args.expect_count),
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
