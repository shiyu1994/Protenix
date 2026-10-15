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

"""Build directory-affine AFDB/MGnify distillation bioassembly caches.

The production unit of scheduling is a directory chunk, not an individual CIF.
Leaf directories are assigned to exactly one rank with largest-first balancing.
Large leaves are split into balanced chunks, while chunk ordering makes workers
visit distinct leaves before taking a second chunk from the same leaf.

No A3M file is opened by this program.  Every pickle is written below the
input CIF's own ``bioassembly`` directory.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import queue
import socket
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    sys.path.remove(str(REPO_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(REPO_ROOT))

PLAN_VERSION = 3
BIOASSEMBLY_DIRNAME = "bioassembly"
STATE_DIRNAME = ".state"
LEAF_INDEX_NAME = "indices.csv.gz"
LEAF_FAILURE_NAME = "failures.tsv"
LEAF_SUCCESS_NAME = "_SUCCESS"
FINALIZE_SUCCESS_NAME = "_FINALIZE_SUCCESS.json"
CONVERSION_AUDIT_ROLE = "distillation_bioassembly_conversion_audit"
LEAF_AUDIT_ROLE = "distillation_bioassembly_leaf_audit"

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
    "artifact_role",
    "entry_id",
    "source_db",
    "source_root_id",
    "cif_relpath",
    "msa_relpath",
    "bioassembly_relpath",
    "canonical_length",
    "canonical_sha256",
]

INDEX_FIELDS = STANDARD_INDEX_FIELDS + PROVENANCE_INDEX_FIELDS
INDEX_SCHEMA_SHA256 = hashlib.sha256(
    "\0".join(INDEX_FIELDS).encode("utf-8")
).hexdigest()


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


@dataclass(frozen=True, slots=True)
class SourceRoot:
    """One relocatable source tree."""

    root_id: str
    source_db: str
    path: str
    source_prefix: Optional[str] = None
    expected_cif_count: Optional[int] = None


@dataclass(frozen=True, slots=True)
class LeafJob:
    """Immutable inventory record for one directory containing CIF files."""

    leaf_id: int
    root_id: str
    leaf_relpath: str
    cif_count: int
    cif_names_sha256: str


@dataclass(frozen=True, slots=True)
class ChunkTask:
    """One contiguous slice of the lexicographically sorted CIF names."""

    leaf_id: int
    root_id: str
    leaf_relpath: str
    cif_count: int
    cif_names_sha256: str
    chunk_index: int
    chunk_count: int
    begin: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True, slots=True)
class PreparedLeaf:
    """Rank-local leaf inventory inherited by forked workers."""

    job: LeafJob
    source_root: SourceRoot
    leaf_dir: Path
    cif_names: tuple[str, ...]


@dataclass(slots=True)
class RankSummary:
    rank: int
    leaf_count: int = 0
    task_count: int = 0
    cif_count: int = 0
    rank_plan_sha256: str = ""


_PREPARED_LEAVES: dict[int, PreparedLeaf] = {}


def _atomic_json_dump(value: Any, path: Path, *, durable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{socket.gethostname()}.{os.getpid()}"
    )
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                default=_json_default,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_text_write(
    value: str,
    path: Path,
    *,
    durable: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{socket.gethostname()}.{os.getpid()}"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(value)
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_jsonl(values: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{socket.gethostname()}.{os.getpid()}"
    )
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for value in values:
                stream.write(
                    json.dumps(
                        value,
                        default=_json_default,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _file_provenance(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    metadata = resolved.stat()
    return {
        "path": str(resolved),
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "sha256": _sha256_file(resolved),
    }


def _parse_source_root(value: str) -> SourceRoot:
    parts = value.split(",", maxsplit=2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--source-root must be ROOT_ID,SOURCE_DB,PATH"
        )
    root_id, source_db, raw_path = (part.strip() for part in parts)
    if not root_id or any(character in root_id for character in "/\t\r\n,"):
        raise argparse.ArgumentTypeError(f"Invalid root ID: {root_id!r}")
    if not source_db or any(
        character in source_db for character in "/\t\r\n,"
    ):
        raise argparse.ArgumentTypeError(
            f"Invalid source database: {source_db!r}"
        )
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(
            f"Source root is not a directory: {path}"
        )
    return SourceRoot(root_id=root_id, source_db=source_db, path=str(path))


def _load_roots_json(path: Path) -> list[SourceRoot]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    records = payload.get("roots")
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a roots list")
    roots = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid root record in {path}: {record!r}")
        roots.append(
            SourceRoot(
                root_id=str(record["root_id"]),
                source_db=str(record["source_db"]),
                path=str(Path(record["path"]).expanduser().resolve()),
                source_prefix=(
                    str(record["source_prefix"])
                    if record.get("source_prefix") is not None
                    else None
                ),
                expected_cif_count=(
                    int(record["cif_files"])
                    if record.get("cif_files") is not None
                    else None
                ),
            )
        )
    return roots


def _normalize_source_roots(
    roots: Sequence[SourceRoot],
) -> list[SourceRoot]:
    if not roots:
        raise ValueError(
            "At least one --source-root or --roots-json is required"
        )
    normalized: list[SourceRoot] = []
    seen_ids: set[str] = set()
    seen_paths: list[tuple[str, Path]] = []
    for root in roots:
        if not root.root_id or root.root_id in seen_ids:
            raise ValueError(f"Duplicate or empty root ID: {root.root_id!r}")
        if not root.source_db:
            raise ValueError(f"Empty source database for root {root.root_id}")
        if (
            root.expected_cif_count is not None
            and root.expected_cif_count < 0
        ):
            raise ValueError(
                f"Negative expected CIF count for root {root.root_id}: "
                f"{root.expected_cif_count}"
            )
        path = Path(root.path).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise NotADirectoryError(path)
        for other_id, other_path in seen_paths:
            if (
                path == other_path
                or path in other_path.parents
                or other_path in path.parents
            ):
                raise ValueError(
                    "Overlapping source roots are not allowed: "
                    f"{root.root_id}={path}, {other_id}={other_path}"
                )
        seen_ids.add(root.root_id)
        seen_paths.append((root.root_id, path))
        normalized.append(
            SourceRoot(
                root_id=root.root_id,
                source_db=root.source_db,
                path=str(path),
                source_prefix=root.source_prefix,
                expected_cif_count=root.expected_cif_count,
            )
        )
    return sorted(normalized, key=lambda item: item.root_id)


def _hash_cif_names(cif_names: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    for name in cif_names:
        hasher.update(os.fsencode(name))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _classify_cif_directory_entry(
    entry: os.DirEntry[str],
) -> tuple[bool, bool]:
    """Return ``(is_real_directory, is_cif)`` with os.walk semantics."""

    try:
        is_directory = entry.is_dir(follow_symlinks=False)
    except OSError as error:
        raise OSError(
            f"Failed to inspect directory entry {entry.path}: {error}"
        ) from error
    if is_directory:
        return True, False
    if not entry.name.endswith(".cif"):
        return False, False
    try:
        is_directory_symlink = entry.is_symlink() and entry.is_dir()
    except OSError as error:
        raise OSError(
            f"Failed to inspect CIF entry {entry.path}: {error}"
        ) from error
    return False, not is_directory_symlink


def _scan_leaf_cif_names(leaf_dir: Path) -> tuple[str, ...]:
    """Read CIF basenames using the same entry semantics as plan inventory."""

    try:
        with os.scandir(leaf_dir) as entries:
            names = sorted(
                entry.name
                for entry in entries
                if _classify_cif_directory_entry(entry)[1]
            )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Leaf directory disappeared: {leaf_dir}"
        ) from error
    return tuple(names)


def _scan_inventory_directory(
    directory: Path,
    relative_directory: str,
) -> tuple[list[tuple[Path, str]], tuple[str, ...]]:
    """Scan one directory without opening or explicitly stating CIF files."""

    child_directories: list[tuple[Path, str]] = []
    cif_names: list[str] = []
    with os.scandir(directory) as entries:
        for entry in entries:
            name = entry.name
            if name == BIOASSEMBLY_DIRNAME:
                continue
            relative_path = (
                f"{relative_directory}/{name}"
                if relative_directory
                else name
            )
            is_directory, is_cif = _classify_cif_directory_entry(entry)
            if is_directory:
                child_directories.append(
                    (directory / name, relative_path)
                )
                continue
            if is_cif:
                cif_names.append(name)

    child_directories.sort(key=lambda item: item[1])
    return child_directories, tuple(sorted(cif_names))


def _inventory_leaves(
    roots: Sequence[SourceRoot],
    *,
    scan_workers: int = 64,
) -> list[LeafJob]:
    if scan_workers <= 0:
        raise ValueError(f"scan_workers must be positive, got {scan_workers}")

    records: list[tuple[str, str, int, str]] = []
    observed_counts = {root.root_id: 0 for root in roots}
    work_queue: queue.Queue[
        Optional[tuple[SourceRoot, Path, str]]
    ] = queue.Queue()
    state_lock = threading.Lock()
    scan_errors: list[tuple[str, BaseException]] = []

    for root in roots:
        work_queue.put((root, Path(root.path), ""))

    def scan_worker() -> None:
        while True:
            task = work_queue.get()
            try:
                if task is None:
                    return
                root, directory, relative_directory = task
                try:
                    child_directories, cif_names = (
                        _scan_inventory_directory(
                            directory,
                            relative_directory,
                        )
                    )
                    for child_directory, child_relpath in child_directories:
                        work_queue.put(
                            (root, child_directory, child_relpath)
                        )
                    if cif_names:
                        leaf_relpath = relative_directory or "."
                        cif_names_sha256 = _hash_cif_names(cif_names)
                        with state_lock:
                            observed_counts[root.root_id] += len(cif_names)
                            records.append(
                                (
                                    root.root_id,
                                    leaf_relpath,
                                    len(cif_names),
                                    cif_names_sha256,
                                )
                            )
                except BaseException as error:
                    detail = (
                        f"Failed to scan source root {root.root_id}: "
                        f"{directory}: {type(error).__name__}: {error}"
                    )
                    with state_lock:
                        scan_errors.append((detail, error))
            finally:
                work_queue.task_done()

    threads = [
        threading.Thread(
            target=scan_worker,
            name=f"plan-cif-scanner-{worker_index:03d}",
        )
        for worker_index in range(scan_workers)
    ]
    started_threads: list[threading.Thread] = []
    try:
        for thread in threads:
            thread.start()
            started_threads.append(thread)
    except BaseException:
        for _ in started_threads:
            work_queue.put(None)
        for thread in started_threads:
            thread.join()
        raise

    scan_completed = False
    try:
        work_queue.join()
        scan_completed = True
    finally:
        for _ in started_threads:
            work_queue.put(None)
        if scan_completed:
            work_queue.join()
        for thread in started_threads:
            thread.join()

    if scan_errors:
        ordered_errors = sorted(scan_errors, key=lambda item: item[0])
        first_error = ordered_errors[0][1]
        if isinstance(first_error, (KeyboardInterrupt, SystemExit)):
            raise first_error
        examples = "\n".join(
            detail for detail, _ in ordered_errors[:20]
        )
        raise OSError(
            f"Failed to scan {len(scan_errors)} directories/entries:\n"
            f"{examples}"
        ) from first_error

    for root in roots:
        if (
            root.expected_cif_count is not None
            and observed_counts[root.root_id] != root.expected_cif_count
        ):
            raise RuntimeError(
                f"Source root {root.root_id} CIF count does not match "
                f"roots.json: observed={observed_counts[root.root_id]}, "
                f"expected={root.expected_cif_count}"
            )
    records.sort(key=lambda item: (item[0], item[1]))
    return [
        LeafJob(
            leaf_id=leaf_id,
            root_id=root_id,
            leaf_relpath=leaf_relpath,
            cif_count=cif_count,
            cif_names_sha256=cif_names_sha256,
        )
        for leaf_id, (
            root_id,
            leaf_relpath,
            cif_count,
            cif_names_sha256,
        ) in enumerate(records)
    ]


def _assign_leaf_jobs(
    leaves: Sequence[LeafJob],
    world_size: int,
) -> list[list[LeafJob]]:
    """Largest-processing-time assignment while keeping a leaf on one rank."""

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    assignments: list[list[LeafJob]] = [[] for _ in range(world_size)]
    weights = [0] * world_size
    for leaf in sorted(
        leaves,
        key=lambda item: (
            -item.cif_count,
            item.root_id,
            item.leaf_relpath,
        ),
    ):
        rank = min(range(world_size), key=lambda idx: (weights[idx], idx))
        assignments[rank].append(leaf)
        weights[rank] += leaf.cif_count
    return assignments


def _split_leaf_job(
    leaf: LeafJob,
    max_files_per_task: int,
) -> list[ChunkTask]:
    if max_files_per_task <= 0:
        raise ValueError(
            "max_files_per_task must be positive, "
            f"got {max_files_per_task}"
        )
    chunk_count = max(1, math.ceil(leaf.cif_count / max_files_per_task))
    base_size, extra = divmod(leaf.cif_count, chunk_count)
    chunks = []
    begin = 0
    for chunk_index in range(chunk_count):
        size = base_size + int(chunk_index < extra)
        end = begin + size
        chunks.append(
            ChunkTask(
                leaf_id=leaf.leaf_id,
                root_id=leaf.root_id,
                leaf_relpath=leaf.leaf_relpath,
                cif_count=leaf.cif_count,
                cif_names_sha256=leaf.cif_names_sha256,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                begin=begin,
                end=end,
            )
        )
        begin = end
    assert begin == leaf.cif_count
    return chunks


def _order_rank_tasks(tasks: Sequence[ChunkTask]) -> list[ChunkTask]:
    """Start distinct leaves before later chunks from a large leaf."""

    return sorted(
        tasks,
        key=lambda item: (
            item.chunk_index,
            -item.size,
            item.root_id,
            item.leaf_relpath,
        ),
    )


def _build_plan_impl(
    source_roots: Sequence[SourceRoot],
    *,
    world_size: int,
    workers_per_rank: int,
    max_files_per_task: int,
    plan_dir: Path,
    scan_workers: int = 64,
    force: bool = False,
    roots_json_path: Optional[Path] = None,
    catalog_manifest_path: Optional[Path] = None,
) -> dict[str, Any]:
    if workers_per_rank <= 0:
        raise ValueError(
            f"workers_per_rank must be positive, got {workers_per_rank}"
        )
    if max_files_per_task <= 0:
        raise ValueError(
            "max_files_per_task must be positive, "
            f"got {max_files_per_task}"
        )
    if scan_workers <= 0:
        raise ValueError(
            f"scan_workers must be positive, got {scan_workers}"
        )
    plan_dir = plan_dir.expanduser().resolve()
    if plan_dir.exists() and any(plan_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Plan directory is not empty: {plan_dir}; use a new path "
            "or pass --force"
        )
    plan_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for old_path in plan_dir.glob("rank-*.jsonl"):
            old_path.unlink()
        for old_name in ("plan.json",):
            try:
                (plan_dir / old_name).unlink()
            except FileNotFoundError:
                pass

    roots = _normalize_source_roots(source_roots)
    leaves = _inventory_leaves(roots, scan_workers=scan_workers)
    assignments = _assign_leaf_jobs(leaves, world_size)
    plan_id = uuid.uuid4().hex
    rank_summaries: list[RankSummary] = []
    task_count = 0
    input_provenance: dict[str, dict[str, Any]] = {}
    if roots_json_path is not None:
        input_provenance["roots_json"] = _file_provenance(roots_json_path)
        if catalog_manifest_path is None:
            sibling_manifest = (
                roots_json_path.expanduser().resolve().parent
                / "manifest.json"
            )
            if sibling_manifest.is_file():
                catalog_manifest_path = sibling_manifest
    if catalog_manifest_path is not None:
        input_provenance["catalog_manifest"] = _file_provenance(
            catalog_manifest_path
        )
    root_cif_counts = {
        root.root_id: sum(
            leaf.cif_count for leaf in leaves if leaf.root_id == root.root_id
        )
        for root in roots
    }
    observed_cif_count = sum(root_cif_counts.values())
    if catalog_manifest_path is not None:
        manifest_payload = json.loads(
            catalog_manifest_path.expanduser()
            .resolve(strict=True)
            .read_text(encoding="utf-8")
        )
        if not isinstance(manifest_payload, dict):
            raise ValueError(
                f"Catalog manifest must be a JSON object: "
                f"{catalog_manifest_path}"
            )
        manifest_counts = manifest_payload.get("counts", {})
        if isinstance(manifest_counts, dict):
            authoritative_count = manifest_counts.get(
                "inventory_entries"
            )
            if (
                authoritative_count is not None
                and int(authoritative_count) != observed_cif_count
            ):
                raise RuntimeError(
                    "Planned CIF count does not match catalog manifest "
                    f"inventory_entries: {observed_cif_count} != "
                    f"{authoritative_count}"
                )

    for rank, rank_leaves in enumerate(assignments):
        tasks = _order_rank_tasks(
            [
                chunk
                for leaf in rank_leaves
                for chunk in _split_leaf_job(leaf, max_files_per_task)
            ]
        )
        task_count += len(tasks)
        rank_plan_path = plan_dir / f"rank-{rank:03d}.jsonl"
        _write_jsonl(
            (
                {
                    "plan_id": plan_id,
                    **asdict(task),
                }
                for task in tasks
            ),
            rank_plan_path,
        )
        rank_summaries.append(
            RankSummary(
                rank=rank,
                leaf_count=len(rank_leaves),
                task_count=len(tasks),
                cif_count=sum(leaf.cif_count for leaf in rank_leaves),
                rank_plan_sha256=_sha256_file(rank_plan_path),
            )
        )

    summary = {
        "version": PLAN_VERSION,
        "plan_id": plan_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "world_size": world_size,
        "workers_per_rank": workers_per_rank,
        "max_files_per_task": max_files_per_task,
        "scan_workers": scan_workers,
        "assignment": "leaf_lpt_then_balanced_chunks",
        "task_order": "chunk_index,-chunk_size,root_id,leaf_relpath",
        "roots": [asdict(root) for root in roots],
        "root_cif_counts": root_cif_counts,
        "input_provenance": input_provenance,
        "leaf_count": len(leaves),
        "task_count": task_count,
        "cif_count": observed_cif_count,
        "ranks": [asdict(summary) for summary in rank_summaries],
    }
    _atomic_json_dump(summary, plan_dir / "plan.json")
    _load_and_validate_all_rank_tasks(plan_dir, plan=summary)
    return summary


def _load_plan(plan_dir: Path) -> dict[str, Any]:
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if int(plan.get("version", -1)) != PLAN_VERSION:
        raise ValueError(
            f"Unsupported plan version in {plan_path}: "
            f"{plan.get('version')!r}"
        )
    if not isinstance(plan.get("plan_id"), str) or not plan["plan_id"]:
        raise ValueError(f"Missing plan_id in {plan_path}")
    return plan


def _roots_from_plan(plan: dict[str, Any]) -> dict[str, SourceRoot]:
    roots: dict[str, SourceRoot] = {}
    for record in plan.get("roots", []):
        root = SourceRoot(**record)
        if root.root_id in roots:
            raise ValueError(f"Duplicate root ID in plan: {root.root_id}")
        roots[root.root_id] = root
    if not roots:
        raise ValueError("Plan contains no roots")
    return roots


def _rank_summaries_from_plan(
    plan: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    try:
        world_size = int(plan["world_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Plan has an invalid world_size") from error
    if world_size <= 0:
        raise ValueError(f"Plan world_size must be positive: {world_size}")
    records = plan.get("ranks")
    if not isinstance(records, list):
        raise ValueError("Plan contains no rank summaries")
    summaries: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid rank summary: {record!r}")
        try:
            rank = int(record["rank"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid rank summary: {record!r}") from error
        if rank in summaries:
            raise ValueError(f"Duplicate rank summary: {rank}")
        summaries[rank] = record
    expected_ranks = set(range(world_size))
    if set(summaries) != expected_ranks:
        raise ValueError(
            "Rank summaries do not cover world_size exactly: "
            f"observed={sorted(summaries)}, "
            f"expected={sorted(expected_ranks)}"
        )
    return summaries


def _load_rank_tasks(
    plan_dir: Path,
    rank: int,
    *,
    plan: Optional[dict[str, Any]] = None,
) -> list[ChunkTask]:
    plan = plan or _load_plan(plan_dir)
    world_size = int(plan["world_size"])
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    plan_path = plan_dir / f"rank-{rank:03d}.jsonl"
    rank_summary = _rank_summaries_from_plan(plan)[rank]
    expected_sha256 = str(rank_summary.get("rank_plan_sha256", ""))
    if not expected_sha256:
        raise ValueError(
            f"Rank {rank} summary is missing rank_plan_sha256"
        )
    actual_sha256 = _sha256_file(plan_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Rank plan SHA256 mismatch for rank {rank}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    tasks = []
    with plan_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                record_plan_id = record.pop("plan_id")
                if record_plan_id != plan["plan_id"]:
                    raise ValueError(
                        f"rank plan ID {record_plan_id!r} does not match "
                        f"{plan['plan_id']!r}"
                    )
                tasks.append(ChunkTask(**record))
            except Exception as error:
                raise ValueError(
                    f"Invalid rank plan at {plan_path}:{line_number}"
                ) from error
    _validate_rank_tasks(tasks)
    leaf_tasks = _leaf_tasks(tasks)
    observed = {
        "task_count": len(tasks),
        "leaf_count": len(leaf_tasks),
        "cif_count": sum(
            leaf_chunks[0].cif_count
            for leaf_chunks in leaf_tasks.values()
        ),
    }
    for field, value in observed.items():
        try:
            planned_value = int(rank_summary[field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Rank {rank} summary has invalid {field}"
            ) from error
        if value != planned_value:
            raise ValueError(
                f"Rank {rank} {field} mismatch: "
                f"observed={value}, planned={planned_value}"
            )
    return tasks


def _validate_rank_tasks(tasks: Sequence[ChunkTask]) -> None:
    by_leaf: dict[int, list[ChunkTask]] = {}
    for task in tasks:
        if task.size <= 0:
            raise ValueError(f"Empty or inverted chunk: {task}")
        by_leaf.setdefault(task.leaf_id, []).append(task)
    for leaf_id, leaf_tasks in by_leaf.items():
        ordered = sorted(leaf_tasks, key=lambda item: item.chunk_index)
        first = ordered[0]
        expected_indices = list(range(first.chunk_count))
        if [task.chunk_index for task in ordered] != expected_indices:
            raise ValueError(f"Non-contiguous chunks for leaf {leaf_id}")
        begin = 0
        for task in ordered:
            stable_fields = (
                task.root_id,
                task.leaf_relpath,
                task.cif_count,
                task.cif_names_sha256,
                task.chunk_count,
            )
            first_fields = (
                first.root_id,
                first.leaf_relpath,
                first.cif_count,
                first.cif_names_sha256,
                first.chunk_count,
            )
            if stable_fields != first_fields or task.begin != begin:
                raise ValueError(f"Inconsistent chunks for leaf {leaf_id}")
            begin = task.end
        if begin != first.cif_count:
            raise ValueError(
                f"Chunks for leaf {leaf_id} cover {begin}, "
                f"expected {first.cif_count}"
            )


def _load_and_validate_all_rank_tasks(
    plan_dir: Path,
    *,
    plan: Optional[dict[str, Any]] = None,
) -> list[list[ChunkTask]]:
    """Load every immutable rank file and validate global plan coverage."""

    plan = plan or _load_plan(plan_dir)
    roots = _roots_from_plan(plan)
    rank_tasks = [
        _load_rank_tasks(plan_dir, rank, plan=plan)
        for rank in range(int(plan["world_size"]))
    ]
    leaf_owner: dict[int, int] = {}
    leaf_definitions: dict[int, tuple[Any, ...]] = {}
    physical_leaves: dict[tuple[str, str], int] = {}
    chunk_keys: set[tuple[int, int]] = set()
    total_task_cifs = 0

    for rank, tasks in enumerate(rank_tasks):
        for task in tasks:
            if task.root_id not in roots:
                raise ValueError(
                    f"Rank {rank} task references unknown root "
                    f"{task.root_id!r}"
                )
            chunk_key = (task.leaf_id, task.chunk_index)
            if chunk_key in chunk_keys:
                raise ValueError(
                    f"Duplicate global chunk assignment: {chunk_key}"
                )
            chunk_keys.add(chunk_key)
            total_task_cifs += task.size

            definition = (
                task.root_id,
                task.leaf_relpath,
                task.cif_count,
                task.cif_names_sha256,
                task.chunk_count,
            )
            previous_definition = leaf_definitions.setdefault(
                task.leaf_id,
                definition,
            )
            if previous_definition != definition:
                raise ValueError(
                    f"Leaf {task.leaf_id} has conflicting definitions"
                )
            previous_owner = leaf_owner.setdefault(task.leaf_id, rank)
            if previous_owner != rank:
                raise ValueError(
                    f"Leaf {task.leaf_id} is assigned to ranks "
                    f"{previous_owner} and {rank}"
                )
            physical_key = (task.root_id, task.leaf_relpath)
            previous_leaf_id = physical_leaves.setdefault(
                physical_key,
                task.leaf_id,
            )
            if previous_leaf_id != task.leaf_id:
                raise ValueError(
                    f"Physical leaf {physical_key} has leaf IDs "
                    f"{previous_leaf_id} and {task.leaf_id}"
                )

    expected_leaf_count = int(plan.get("leaf_count", -1))
    expected_task_count = int(plan.get("task_count", -1))
    expected_cif_count = int(plan.get("cif_count", -1))
    if len(leaf_definitions) != expected_leaf_count:
        raise ValueError(
            "Global leaf_count mismatch: "
            f"observed={len(leaf_definitions)}, "
            f"planned={expected_leaf_count}"
        )
    if len(chunk_keys) != expected_task_count:
        raise ValueError(
            "Global task_count mismatch: "
            f"observed={len(chunk_keys)}, "
            f"planned={expected_task_count}"
        )
    observed_leaf_ids = set(leaf_definitions)
    expected_leaf_ids = set(range(expected_leaf_count))
    if observed_leaf_ids != expected_leaf_ids:
        raise ValueError(
            "Global leaf IDs are not contiguous: "
            f"observed={sorted(observed_leaf_ids)}, "
            f"expected={sorted(expected_leaf_ids)}"
        )
    leaf_cif_count = sum(
        int(definition[2]) for definition in leaf_definitions.values()
    )
    if leaf_cif_count != expected_cif_count:
        raise ValueError(
            "Global leaf CIF count mismatch: "
            f"observed={leaf_cif_count}, planned={expected_cif_count}"
        )
    if total_task_cifs != expected_cif_count:
        raise ValueError(
            "Global chunk CIF coverage mismatch: "
            f"observed={total_task_cifs}, planned={expected_cif_count}"
        )

    observed_root_counts = {root_id: 0 for root_id in roots}
    for definition in leaf_definitions.values():
        observed_root_counts[str(definition[0])] += int(definition[2])
    planned_root_counts = plan.get("root_cif_counts")
    if not isinstance(planned_root_counts, dict):
        raise ValueError("Plan is missing root_cif_counts")
    normalized_planned_root_counts = {
        str(root_id): int(count)
        for root_id, count in planned_root_counts.items()
    }
    if observed_root_counts != normalized_planned_root_counts:
        raise ValueError(
            "Per-root CIF counts do not match plan: "
            f"observed={observed_root_counts}, "
            f"planned={normalized_planned_root_counts}"
        )
    for root_id, root in roots.items():
        if (
            root.expected_cif_count is not None
            and observed_root_counts[root_id] != root.expected_cif_count
        ):
            raise ValueError(
                f"Root {root_id} CIF coverage no longer matches its "
                f"authoritative count: "
                f"{observed_root_counts[root_id]} != "
                f"{root.expected_cif_count}"
            )
    return rank_tasks


def _leaf_job_from_task(task: ChunkTask) -> LeafJob:
    return LeafJob(
        leaf_id=task.leaf_id,
        root_id=task.root_id,
        leaf_relpath=task.leaf_relpath,
        cif_count=task.cif_count,
        cif_names_sha256=task.cif_names_sha256,
    )


def _prepare_rank_leaves(
    tasks: Sequence[ChunkTask],
    roots: dict[str, SourceRoot],
) -> dict[int, PreparedLeaf]:
    prepared: dict[int, PreparedLeaf] = {}
    for task in tasks:
        if task.leaf_id in prepared:
            continue
        try:
            source_root = roots[task.root_id]
        except KeyError as error:
            raise ValueError(
                f"Unknown root ID in rank plan: {task.root_id}"
            ) from error
        root_path = Path(source_root.path)
        leaf_dir = (root_path / task.leaf_relpath).resolve()
        try:
            leaf_dir.relative_to(root_path.resolve())
        except ValueError as error:
            raise ValueError(
                f"Leaf path escapes source root: {task.leaf_relpath}"
            ) from error
        cif_names = _scan_leaf_cif_names(leaf_dir)
        names_hash = _hash_cif_names(cif_names)
        if (
            len(cif_names) != task.cif_count
            or names_hash != task.cif_names_sha256
        ):
            raise RuntimeError(
                "Leaf inventory no longer matches the plan: "
                f"root={task.root_id}, leaf={task.leaf_relpath}, "
                f"count={len(cif_names)}/{task.cif_count}, "
                f"sha256={names_hash}/{task.cif_names_sha256}"
            )
        prepared[task.leaf_id] = PreparedLeaf(
            job=_leaf_job_from_task(task),
            source_root=source_root,
            leaf_dir=leaf_dir,
            cif_names=cif_names,
        )
    return prepared


def _configure_protenix_root(root: Optional[Path]) -> Path:
    if root is None:
        configured = os.environ.get("PROTENIX_ROOT_DIR")
        if not configured:
            raise ValueError(
                "PROTENIX_ROOT_DIR is unset; pass --protenix-root-dir"
            )
        root = Path(configured)
    root = root.expanduser().resolve()
    required = [
        root / "common" / "components.cif",
        root / "common" / "components.cif.rdkit_mol.pkl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Protenix CCD cache is incomplete; missing: "
            + ", ".join(missing)
        )
    loaded = sys.modules.get("configs.configs_data")
    if loaded is not None:
        configured_ccd = Path(
            loaded.data_configs["ccd_components_file"]
        ).resolve()
        if configured_ccd != required[0]:
            raise RuntimeError(
                "configs.configs_data was imported before PROTENIX_ROOT_DIR "
                f"was configured ({configured_ccd} != {required[0]})"
            )
    os.environ["PROTENIX_ROOT_DIR"] = str(root)
    return root


def _prewarm_parent() -> dict[str, Any]:
    """Populate parser, CCD, and RDKit caches before forking workers."""

    started = time.monotonic()
    from protenix.data.constants import PRO_STD_RESIDUES
    from protenix.data.core import ccd
    from protenix.data.core.predicted_protein_fast_path import (
        prewarm_predicted_protein_fast_path,
    )
    from protenix.data.pipeline.data_pipeline import DataPipeline
    from protenix.utils.file_io import dump_gzip_pickle

    preloaded_modules = (DataPipeline, dump_gzip_pickle)
    ccd.biotite_load_ccd_cif()
    ccd.get_component_rdkit_mol("ALA")
    for residue_name in PRO_STD_RESIDUES:
        if residue_name == "UNK":
            continue
        ccd.get_component_atom_array(
            residue_name,
            keep_leaving_atoms=True,
            keep_hydrogens=False,
        )
        ccd.get_ccd_ref_info(residue_name)
    prewarm_predicted_protein_fast_path()
    return {
        "elapsed_seconds": time.monotonic() - started,
        "preloaded_modules": len(preloaded_modules),
        "residue_templates": len(PRO_STD_RESIDUES) - 1,
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _validate_single_index_row(
    sample_indices: Sequence[dict[str, Any]],
) -> None:
    if len(sample_indices) != 1:
        raise ValueError(
            "Expected exactly one training index row per predicted "
            f"protein CIF, found {len(sample_indices)}"
        )


def _convert_one(
    *,
    prepared: PreparedLeaf,
    cif_name: str,
    compresslevel: int,
    fast_path_mode: str,
    durable_output: bool,
) -> dict[str, Any]:
    cif_path = prepared.leaf_dir / cif_name
    sample_id = Path(cif_name).stem
    output_dir = prepared.leaf_dir / BIOASSEMBLY_DIRNAME
    output_path = output_dir / f"{sample_id}.pkl.gz"
    root_path = Path(prepared.source_root.path)
    try:
        from protenix.data.pipeline.data_pipeline import DataPipeline
        from protenix.utils.file_io import dump_gzip_pickle

        sample_indices, bioassembly = DataPipeline.get_data_from_mmcif(
            cif_path,
            pdb_cluster_file=None,
            dataset="Distillation",
            distillation_fast_path=fast_path_mode,
            raise_on_error=True,
        )
        if not sample_indices or not bioassembly:
            raise RuntimeError("Distillation parser returned an empty result")
        _validate_single_index_row(sample_indices)

        sequences = bioassembly.get("sequences") or {}
        if len(sequences) != 1:
            raise ValueError(
                "Expected exactly one polymer sequence, "
                f"found {len(sequences)}"
            )
        canonical_sequence = str(next(iter(sequences.values())))
        if not canonical_sequence:
            raise ValueError("Canonical sequence is empty")
        num_tokens = int(bioassembly.get("num_tokens", -1))
        if num_tokens != len(canonical_sequence):
            raise ValueError(
                "Canonical sequence/token mismatch: "
                f"{len(canonical_sequence)} != {num_tokens}"
            )
        atom_array = bioassembly.get("atom_array")
        token_array = bioassembly.get("token_array")
        if atom_array is None or len(atom_array) == 0:
            raise ValueError("AtomArray is empty")
        if token_array is None or len(token_array) != num_tokens:
            raise ValueError(
                "TokenArray is empty or has an unexpected length"
            )

        fast_info = bioassembly.get("predicted_protein_fast_path") or {}
        fast_path_used = bool(fast_info.get("used", False))
        fast_reader_used = bool(fast_info.get("cif_reader_used", False))
        if fast_path_mode == "require" and not (
            fast_path_used and fast_reader_used
        ):
            raise RuntimeError(
                "Required predicted-protein CIF reader and completion fast "
                "paths were not both used"
            )

        entry_id = str(bioassembly.get("pdb_id", ""))
        cif_relpath = cif_path.relative_to(root_path).as_posix()
        msa_relpath = cif_path.with_suffix(".a3m").relative_to(
            root_path
        ).as_posix()
        bioassembly_relpath = output_path.relative_to(root_path).as_posix()
        canonical_sha256 = _sha256_text(canonical_sequence)

        bioassembly["entry_id"] = entry_id
        bioassembly["pdb_id"] = sample_id
        bioassembly["source_root_id"] = prepared.source_root.root_id
        bioassembly["source_cif_relpath"] = cif_relpath
        bioassembly["canonical_sha256"] = canonical_sha256

        enriched_indices = []
        for sample_index in sample_indices:
            row = dict(sample_index)
            row.update(
                {
                    "artifact_role": CONVERSION_AUDIT_ROLE,
                    "pdb_id": sample_id,
                    "entry_id": entry_id,
                    "source_db": prepared.source_root.source_db,
                    "source_root_id": prepared.source_root.root_id,
                    "cif_relpath": cif_relpath,
                    "msa_relpath": msa_relpath,
                    "bioassembly_relpath": bioassembly_relpath,
                    "canonical_length": len(canonical_sequence),
                    "canonical_sha256": canonical_sha256,
                }
            )
            enriched_indices.append(row)

        output_dir.mkdir(exist_ok=True)
        dump_gzip_pickle(
            bioassembly,
            output_path,
            compression_level=compresslevel,
            protocol=pickle.HIGHEST_PROTOCOL,
            atomic=True,
            durable=durable_output,
        )
        return {
            "status": "success",
            "cif_name": cif_name,
            "sample_id": sample_id,
            "output_name": output_path.name,
            "fast_path_used": fast_path_used,
            "fast_path_fallback_reason": fast_info.get("fallback_reason"),
            "fast_cif_reader_used": fast_reader_used,
            "fast_cif_reader_fallback_reason": fast_info.get(
                "cif_reader_fallback_reason"
            ),
            "index_rows": enriched_indices,
        }
    except Exception as error:
        return {
            "status": "failed",
            "cif_name": cif_name,
            "sample_id": sample_id,
            "output_name": output_path.name,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


def _chunk_prefix(task: ChunkTask) -> str:
    width = max(4, len(str(task.chunk_count - 1)))
    return f"chunk-{task.chunk_index:0{width}d}"


def _chunk_paths(
    prepared: PreparedLeaf,
    task: ChunkTask,
) -> dict[str, Path]:
    state_dir = prepared.leaf_dir / BIOASSEMBLY_DIRNAME / STATE_DIRNAME
    prefix = _chunk_prefix(task)
    return {
        "output_dir": prepared.leaf_dir / BIOASSEMBLY_DIRNAME,
        "state_dir": state_dir,
        "lock": state_dir / f"{prefix}.lock",
        "journal": state_dir / f"{prefix}.progress.jsonl",
        "index": state_dir / f"{prefix}.indices.csv.gz",
        "failures": state_dir / f"{prefix}.failures.tsv",
        "success": state_dir / f"{prefix}._SUCCESS.json",
    }


@contextmanager
def _chunk_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            raise RuntimeError(
                f"Chunk is already being processed: {path}"
            ) from error
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(
            json.dumps(
                {
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
                sort_keys=True,
            )
            + "\n"
        )
        lock_file.flush()
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _plan_activity_lock(
    plan_dir: Path,
    *,
    exclusive: bool,
) -> Iterator[None]:
    lock_path = plan_dir / ".activity.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(
                lock_file.fileno(),
                operation | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            action = "finalize" if exclusive else "run a rank"
            raise RuntimeError(
                f"Cannot {action} while the plan has incompatible active "
                f"work: {lock_path}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def build_plan(
    source_roots: Sequence[SourceRoot],
    *,
    world_size: int,
    workers_per_rank: int,
    max_files_per_task: int,
    plan_dir: Path,
    scan_workers: int = 64,
    force: bool = False,
    roots_json_path: Optional[Path] = None,
    catalog_manifest_path: Optional[Path] = None,
) -> dict[str, Any]:
    resolved_plan_dir = plan_dir.expanduser().resolve()
    if not force:
        return _build_plan_impl(
            source_roots,
            world_size=world_size,
            workers_per_rank=workers_per_rank,
            max_files_per_task=max_files_per_task,
            plan_dir=resolved_plan_dir,
            scan_workers=scan_workers,
            force=False,
            roots_json_path=roots_json_path,
            catalog_manifest_path=catalog_manifest_path,
        )
    resolved_plan_dir.mkdir(parents=True, exist_ok=True)
    with _plan_activity_lock(resolved_plan_dir, exclusive=True):
        try:
            (resolved_plan_dir / FINALIZE_SUCCESS_NAME).unlink()
        except FileNotFoundError:
            pass
        return _build_plan_impl(
            source_roots,
            world_size=world_size,
            workers_per_rank=workers_per_rank,
            max_files_per_task=max_files_per_task,
            plan_dir=resolved_plan_dir,
            scan_workers=scan_workers,
            force=True,
            roots_json_path=roots_json_path,
            catalog_manifest_path=catalog_manifest_path,
        )


def _read_json_marker(path: Path) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _file_marker_fields(path: Path, *, prefix: str) -> dict[str, Any]:
    metadata = path.stat()
    return {
        f"{prefix}_size": metadata.st_size,
        f"{prefix}_sha256": _sha256_file(path),
    }


def _marker_file_matches(
    marker: dict[str, Any],
    path: Path,
    *,
    prefix: str,
) -> bool:
    try:
        expected_size = int(marker[f"{prefix}_size"])
        expected_sha256 = str(marker[f"{prefix}_sha256"])
        metadata = path.stat()
        if metadata.st_size != expected_size or not expected_sha256:
            return False
        actual_sha256 = _sha256_file(path)
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return actual_sha256 == expected_sha256


def _chunk_marker_matches(
    task: ChunkTask,
    *,
    plan_id: str,
    fast_path_mode: str,
    paths: dict[str, Path],
) -> bool:
    if not paths["success"].is_file() or not paths["index"].is_file():
        return False
    marker = _read_json_marker(paths["success"])
    if marker is None:
        return False
    expected = {
        "plan_id": plan_id,
        "cif_names_sha256": task.cif_names_sha256,
        "chunk_index": task.chunk_index,
        "chunk_count": task.chunk_count,
        "begin": task.begin,
        "end": task.end,
        "success_count": task.size,
        "index_rows": task.size,
        "index_schema_sha256": INDEX_SCHEMA_SHA256,
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        return False
    if not _marker_file_matches(marker, paths["index"], prefix="index"):
        return False
    if fast_path_mode != "require":
        return True
    return (
        marker.get("fast_path_used") == task.size
        and marker.get("fast_cif_reader_used") == task.size
    )


def _journal_record_matches(
    record: dict[str, Any],
    *,
    task: ChunkTask,
    plan_id: str,
) -> bool:
    return (
        record.get("plan_id") == plan_id
        and record.get("cif_names_sha256") == task.cif_names_sha256
        and record.get("chunk_index") == task.chunk_index
        and record.get("chunk_count") == task.chunk_count
    )


def _load_chunk_journal(
    path: Path,
    *,
    task: ChunkTask,
    plan_id: str,
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    with path.open("r+b") as stream:
        line_number = 0
        while True:
            line_start = stream.tell()
            raw_line = stream.readline()
            if not raw_line:
                break
            line_number += 1
            terminated = raw_line.endswith(b"\n")
            try:
                record = json.loads(raw_line.decode("utf-8"))
                if not isinstance(record, dict):
                    raise ValueError("journal record is not an object")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                if not terminated:
                    stream.seek(line_start)
                    stream.truncate()
                    stream.flush()
                    os.fsync(stream.fileno())
                    break
                raise ValueError(
                    f"Invalid journal record at {path}:{line_number}"
                ) from error
            cif_name = record.get("cif_name")
            if cif_name and _journal_record_matches(
                record,
                task=task,
                plan_id=plan_id,
            ):
                latest[str(cif_name)] = record
            if not terminated:
                stream.seek(0, os.SEEK_END)
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
                break
    return latest


def _record_is_complete(
    record: dict[str, Any],
    *,
    output_dir: Path,
    fast_path_mode: str,
) -> bool:
    if record.get("status") != "success":
        return False
    index_rows = record.get("index_rows")
    if not isinstance(index_rows, list) or len(index_rows) != 1:
        return False
    output_name = str(record.get("output_name", ""))
    if not output_name or not (output_dir / output_name).is_file():
        return False
    return fast_path_mode != "require" or (
        bool(record.get("fast_path_used"))
        and bool(record.get("fast_cif_reader_used"))
    )


def _write_index_records(
    records: dict[str, dict[str, Any]],
    cif_names: Sequence[str],
    path: Path,
    *,
    compresslevel: int,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{socket.gethostname()}.{os.getpid()}"
    )
    row_count = 0
    try:
        with gzip.open(
            temporary,
            "wt",
            encoding="utf-8",
            newline="",
            compresslevel=compresslevel,
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=INDEX_FIELDS,
                extrasaction="ignore",
                quoting=csv.QUOTE_NONNUMERIC,
            )
            writer.writeheader()
            for cif_name in cif_names:
                for row in records[cif_name].get("index_rows") or []:
                    audit_row = dict(row)
                    audit_row["artifact_role"] = CONVERSION_AUDIT_ROLE
                    writer.writerow(audit_row)
                    row_count += 1
        if row_count != len(cif_names):
            raise ValueError(
                f"Chunk index row count mismatch: "
                f"{row_count} != {len(cif_names)}"
            )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return row_count


def _write_chunk_failures(
    records: dict[str, dict[str, Any]],
    cif_names: Sequence[str],
    path: Path,
    *,
    complete_names: Optional[set[str]] = None,
) -> int:
    failures = []
    for cif_name in cif_names:
        record = records.get(cif_name)
        complete = (
            cif_name in complete_names
            if complete_names is not None
            else record is not None and record.get("status") == "success"
        )
        if not complete:
            if record is not None and record.get("status") == "success":
                record = {
                    "cif_name": cif_name,
                    "error_type": "OutputMissing",
                    "error": "Successful journal record has no output file",
                }
            failures.append(
                record
                or {
                    "cif_name": cif_name,
                    "error_type": "Incomplete",
                    "error": "No successful journal record",
                }
            )
    lines = ["cif_name\terror_type\terror\n"]
    for record in failures:
        error = str(record.get("error", "")).replace("\t", " ").replace(
            "\n",
            " ",
        )
        lines.append(
            f"{record.get('cif_name', '')}\t"
            f"{record.get('error_type', '')}\t{error}\n"
        )
    _atomic_text_write("".join(lines), path)
    return len(failures)


def _process_chunk(
    invocation: tuple[ChunkTask, str, int, str, bool, Optional[int]],
) -> dict[str, Any]:
    (
        task,
        plan_id,
        compresslevel,
        fast_path_mode,
        durable_output,
        max_files,
    ) = invocation
    started = time.monotonic()
    prepared = _PREPARED_LEAVES[task.leaf_id]
    paths = _chunk_paths(prepared, task)
    try:
        with _chunk_lock(paths["lock"]):
            if _chunk_marker_matches(
                task,
                plan_id=plan_id,
                fast_path_mode=fast_path_mode,
                paths=paths,
            ):
                return {
                    "status": "complete",
                    "leaf_id": task.leaf_id,
                    "chunk_index": task.chunk_index,
                    "completed": task.size,
                    "failed": 0,
                    "skipped": task.size,
                    "elapsed_seconds": time.monotonic() - started,
                }

            try:
                paths["success"].unlink()
            except FileNotFoundError:
                pass
            expected_names = prepared.cif_names[task.begin : task.end]
            expected_name_set = set(expected_names)
            records = _load_chunk_journal(
                paths["journal"],
                task=task,
                plan_id=plan_id,
            )
            completed_names = {
                cif_name
                for cif_name, record in records.items()
                if cif_name in expected_name_set
                and _record_is_complete(
                    record,
                    output_dir=paths["output_dir"],
                    fast_path_mode=fast_path_mode,
                )
            }
            pending = [
                cif_name
                for cif_name in expected_names
                if cif_name not in completed_names
            ]
            if max_files is not None:
                pending = pending[:max_files]

            paths["state_dir"].mkdir(parents=True, exist_ok=True)
            with paths["journal"].open("a", encoding="utf-8") as journal:
                last_flush = time.monotonic()
                since_flush = 0
                for cif_name in pending:
                    result = _convert_one(
                        prepared=prepared,
                        cif_name=cif_name,
                        compresslevel=compresslevel,
                        fast_path_mode=fast_path_mode,
                        durable_output=durable_output,
                    )
                    result.update(
                        {
                            "plan_id": plan_id,
                            "cif_names_sha256": task.cif_names_sha256,
                            "chunk_index": task.chunk_index,
                            "chunk_count": task.chunk_count,
                        }
                    )
                    records[cif_name] = result
                    journal.write(
                        json.dumps(
                            result,
                            default=_json_default,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    )
                    journal.write("\n")
                    since_flush += 1
                    now = time.monotonic()
                    if since_flush >= 64 or now - last_flush >= 2.0:
                        journal.flush()
                        since_flush = 0
                        last_flush = now
                journal.flush()
                os.fsync(journal.fileno())

            success_names = {
                cif_name
                for cif_name in expected_names
                if cif_name in records
                and _record_is_complete(
                    records[cif_name],
                    output_dir=paths["output_dir"],
                    fast_path_mode=fast_path_mode,
                )
            }
            failed = _write_chunk_failures(
                records,
                expected_names,
                paths["failures"],
                complete_names=success_names,
            )
            complete = max_files is None and failed == 0
            index_rows = 0
            if complete:
                index_rows = _write_index_records(
                    records,
                    expected_names,
                    paths["index"],
                    compresslevel=compresslevel,
                )
                index_marker = _file_marker_fields(
                    paths["index"],
                    prefix="index",
                )
                fast_path_used = sum(
                    bool(records[name].get("fast_path_used"))
                    for name in expected_names
                )
                fast_reader_used = sum(
                    bool(records[name].get("fast_cif_reader_used"))
                    for name in expected_names
                )
                _atomic_json_dump(
                    {
                        "plan_id": plan_id,
                        "cif_names_sha256": task.cif_names_sha256,
                        "chunk_index": task.chunk_index,
                        "chunk_count": task.chunk_count,
                        "begin": task.begin,
                        "end": task.end,
                        "success_count": task.size,
                        "index_rows": index_rows,
                        "index_schema_sha256": INDEX_SCHEMA_SHA256,
                        **index_marker,
                        "fast_path_mode": fast_path_mode,
                        "fast_path_used": fast_path_used,
                        "fast_cif_reader_used": fast_reader_used,
                        "completed_at": time.strftime(
                            "%Y-%m-%dT%H:%M:%S%z"
                        ),
                    },
                    paths["success"],
                )

            return {
                "status": "complete" if complete else "incomplete",
                "leaf_id": task.leaf_id,
                "chunk_index": task.chunk_index,
                "completed": len(success_names),
                "failed": task.size - len(success_names),
                "skipped": len(completed_names),
                "index_rows": index_rows,
                "elapsed_seconds": time.monotonic() - started,
            }
    except Exception as error:
        return {
            "status": "crashed",
            "leaf_id": task.leaf_id,
            "chunk_index": task.chunk_index,
            "completed": 0,
            "failed": task.size,
            "skipped": 0,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - started,
        }


def _task_priority(task: ChunkTask) -> tuple[Any, ...]:
    return (
        task.chunk_index,
        -task.size,
        task.root_id,
        task.leaf_relpath,
    )


def _choose_next_task(
    queues: dict[int, deque[ChunkTask]],
    active_by_leaf: dict[int, int],
    *,
    tail_max_lanes_per_leaf: int,
) -> Optional[ChunkTask]:
    """Prefer an inactive leaf; use another lane only in the rank tail."""

    inactive = [
        queue[0]
        for leaf_id, queue in queues.items()
        if queue and active_by_leaf.get(leaf_id, 0) == 0
    ]
    if inactive:
        return min(inactive, key=_task_priority)

    parallel = [
        queue[0]
        for leaf_id, queue in queues.items()
        if queue
        and (
            tail_max_lanes_per_leaf == 0
            or active_by_leaf.get(leaf_id, 0)
            < tail_max_lanes_per_leaf
        )
    ]
    if not parallel:
        return None
    return min(
        parallel,
        key=lambda task: (
            active_by_leaf.get(task.leaf_id, 0),
            *_task_priority(task),
        ),
    )


def _run_directory_affine_tasks(
    invocations: Sequence[
        tuple[ChunkTask, str, int, str, bool, Optional[int]]
    ],
    *,
    workers: int,
    tail_max_lanes_per_leaf: int,
) -> Iterator[dict[str, Any]]:
    """Dynamically submit chunks while minimizing same-directory overlap."""

    if tail_max_lanes_per_leaf < 0:
        raise ValueError(
            "tail_max_lanes_per_leaf must be non-negative, "
            f"got {tail_max_lanes_per_leaf}"
        )
    invocation_by_task = {
        invocation[0]: invocation for invocation in invocations
    }
    queues: dict[int, deque[ChunkTask]] = {}
    for task in sorted(invocation_by_task, key=_task_priority):
        queues.setdefault(task.leaf_id, deque()).append(task)
    active_by_leaf: dict[int, int] = {}
    future_to_task = {}
    context = mp.get_context("fork")
    process_count = min(workers, len(invocations))

    with ProcessPoolExecutor(
        max_workers=process_count,
        mp_context=context,
    ) as executor:
        while queues or future_to_task:
            while len(future_to_task) < process_count and queues:
                task = _choose_next_task(
                    queues,
                    active_by_leaf,
                    tail_max_lanes_per_leaf=tail_max_lanes_per_leaf,
                )
                if task is None:
                    break
                queue = queues[task.leaf_id]
                popped = queue.popleft()
                assert popped == task
                if not queue:
                    del queues[task.leaf_id]
                active_by_leaf[task.leaf_id] = (
                    active_by_leaf.get(task.leaf_id, 0) + 1
                )
                future = executor.submit(
                    _process_chunk,
                    invocation_by_task[task],
                )
                future_to_task[future] = task

            if not future_to_task:
                if queues:
                    raise RuntimeError(
                        "Directory-affine scheduler made no progress"
                    )
                break

            completed_futures, _ = wait(
                future_to_task,
                timeout=30.0,
                return_when=FIRST_COMPLETED,
            )
            if not completed_futures:
                yield {
                    "status": "heartbeat",
                    "active_tasks": len(future_to_task),
                    "queued_tasks": sum(len(queue) for queue in queues.values()),
                }
                continue
            for future in completed_futures:
                task = future_to_task.pop(future)
                active_by_leaf[task.leaf_id] -= 1
                try:
                    yield future.result()
                except BrokenProcessPool:
                    raise
                except Exception as error:
                    yield {
                        "status": "crashed",
                        "leaf_id": task.leaf_id,
                        "chunk_index": task.chunk_index,
                        "completed": 0,
                        "failed": task.size,
                        "skipped": 0,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }


def _leaf_tasks(
    tasks: Sequence[ChunkTask],
) -> dict[int, list[ChunkTask]]:
    grouped: dict[int, list[ChunkTask]] = {}
    for task in tasks:
        grouped.setdefault(task.leaf_id, []).append(task)
    return {
        leaf_id: sorted(
            leaf_tasks,
            key=lambda item: item.chunk_index,
        )
        for leaf_id, leaf_tasks in grouped.items()
    }


def _merge_chunk_indices(
    chunk_paths: Sequence[Path],
    output_path: Path,
    *,
    compresslevel: int,
    expected_row_count: Optional[int] = None,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.tmp.{socket.gethostname()}.{os.getpid()}"
    )
    row_count = 0
    try:
        with gzip.open(
            temporary,
            "wt",
            encoding="utf-8",
            newline="",
            compresslevel=compresslevel,
        ) as destination:
            writer = csv.DictWriter(
                destination,
                fieldnames=INDEX_FIELDS,
                extrasaction="ignore",
                quoting=csv.QUOTE_NONNUMERIC,
            )
            writer.writeheader()
            for chunk_path in chunk_paths:
                with gzip.open(
                    chunk_path,
                    "rt",
                    encoding="utf-8",
                    newline="",
                ) as source:
                    reader = csv.DictReader(source)
                    missing = [
                        field
                        for field in INDEX_FIELDS
                        if field not in (reader.fieldnames or [])
                    ]
                    if missing:
                        raise ValueError(
                            f"Index shard {chunk_path} is missing {missing}"
                        )
                    for row in reader:
                        writer.writerow(row)
                        row_count += 1
        if (
            expected_row_count is not None
            and row_count != expected_row_count
        ):
            raise ValueError(
                f"Merged index row count mismatch for {output_path}: "
                f"{row_count} != {expected_row_count}"
            )
        os.replace(temporary, output_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return row_count


def _merge_chunk_failures(
    chunk_paths: Sequence[Path],
    output_path: Path,
) -> int:
    lines = ["cif_name\terror_type\terror\n"]
    failure_count = 0
    for chunk_path in chunk_paths:
        if not chunk_path.is_file():
            continue
        with chunk_path.open(encoding="utf-8") as stream:
            next(stream, None)
            for line in stream:
                if line.strip():
                    lines.append(line)
                    failure_count += 1
    _atomic_text_write("".join(lines), output_path)
    return failure_count


def _expected_leaf_output_names(
    prepared: PreparedLeaf,
) -> tuple[str, ...]:
    cif_names = prepared.cif_names or _scan_leaf_cif_names(
        prepared.leaf_dir
    )
    if (
        len(cif_names) != prepared.job.cif_count
        or _hash_cif_names(cif_names) != prepared.job.cif_names_sha256
    ):
        raise RuntimeError(
            f"Leaf inventory no longer matches plan: "
            f"{prepared.source_root.root_id}:"
            f"{prepared.job.leaf_relpath}"
        )
    output_names = tuple(
        f"{Path(cif_name).stem}.pkl.gz" for cif_name in cif_names
    )
    if len(set(output_names)) != len(output_names):
        raise ValueError(
            f"CIF stems are not unique in leaf "
            f"{prepared.source_root.root_id}:"
            f"{prepared.job.leaf_relpath}"
        )
    return output_names


def _scan_leaf_output_names(output_dir: Path) -> set[str]:
    try:
        with os.scandir(output_dir) as entries:
            return {
                entry.name
                for entry in entries
                if entry.name.endswith(".pkl.gz")
                and entry.is_file(follow_symlinks=False)
            }
    except FileNotFoundError:
        return set()


def _missing_chunk_outputs(
    leaf_tasks: Sequence[ChunkTask],
    expected_output_names: Sequence[str],
    actual_output_names: set[str],
) -> dict[int, list[str]]:
    missing: dict[int, list[str]] = {}
    for task in leaf_tasks:
        task_missing = [
            name
            for name in expected_output_names[task.begin : task.end]
            if name not in actual_output_names
        ]
        if task_missing:
            missing[task.chunk_index] = task_missing
    return missing


def _invalidate_missing_chunk_outputs(
    leaf_tasks: Sequence[ChunkTask],
    prepared: PreparedLeaf,
    *,
    expected_output_names: Optional[Sequence[str]] = None,
) -> dict[int, list[str]]:
    output_dir = prepared.leaf_dir / BIOASSEMBLY_DIRNAME
    if expected_output_names is None:
        expected_output_names = _expected_leaf_output_names(prepared)
    missing_outputs = _missing_chunk_outputs(
        leaf_tasks,
        expected_output_names,
        _scan_leaf_output_names(output_dir),
    )
    if not missing_outputs:
        return {}
    for task in leaf_tasks:
        if task.chunk_index not in missing_outputs:
            continue
        paths = _chunk_paths(prepared, task)
        try:
            paths["success"].unlink()
        except FileNotFoundError:
            pass
    try:
        (output_dir / LEAF_SUCCESS_NAME).unlink()
    except FileNotFoundError:
        pass
    return missing_outputs


def _finalize_leaf(
    leaf_tasks: Sequence[ChunkTask],
    prepared: PreparedLeaf,
    *,
    plan_id: str,
    fast_path_mode: str,
    compresslevel: int,
) -> bool:
    output_dir = prepared.leaf_dir / BIOASSEMBLY_DIRNAME
    marker_path = output_dir / LEAF_SUCCESS_NAME
    chunk_state = [
        (task, _chunk_paths(prepared, task)) for task in leaf_tasks
    ]
    complete = all(
        _chunk_marker_matches(
            task,
            plan_id=plan_id,
            fast_path_mode=fast_path_mode,
            paths=paths,
        )
        for task, paths in chunk_state
    )
    if not complete:
        try:
            marker_path.unlink()
        except FileNotFoundError:
            pass
        _merge_chunk_failures(
            [paths["failures"] for _, paths in chunk_state],
            output_dir / LEAF_FAILURE_NAME,
        )
        return False

    markers = [
        _read_json_marker(paths["success"]) for _, paths in chunk_state
    ]
    if any(marker is None for marker in markers):
        return False
    expected_output_names = _expected_leaf_output_names(prepared)
    if _invalidate_missing_chunk_outputs(
        leaf_tasks,
        prepared,
        expected_output_names=expected_output_names,
    ):
        return False
    index_rows = _merge_chunk_indices(
        [paths["index"] for _, paths in chunk_state],
        output_dir / LEAF_INDEX_NAME,
        compresslevel=compresslevel,
        expected_row_count=prepared.job.cif_count,
    )
    _merge_chunk_failures(
        [paths["failures"] for _, paths in chunk_state],
        output_dir / LEAF_FAILURE_NAME,
    )
    fast_path_used = sum(
        int(marker["fast_path_used"]) for marker in markers if marker
    )
    fast_reader_used = sum(
        int(marker["fast_cif_reader_used"]) for marker in markers if marker
    )
    leaf_index_path = output_dir / LEAF_INDEX_NAME
    _atomic_json_dump(
        {
            "plan_id": plan_id,
            "root_id": prepared.source_root.root_id,
            "source_db": prepared.source_root.source_db,
            "leaf_relpath": prepared.job.leaf_relpath,
            "cif_names_sha256": prepared.job.cif_names_sha256,
            "cif_count": prepared.job.cif_count,
            "chunk_count": len(leaf_tasks),
            "index_rows": index_rows,
            "index_schema_sha256": INDEX_SCHEMA_SHA256,
            **_file_marker_fields(leaf_index_path, prefix="index"),
            "pkl_count": len(expected_output_names),
            "pkl_names_sha256": _hash_cif_names(expected_output_names),
            "fast_path_mode": fast_path_mode,
            "fast_path_used": fast_path_used,
            "fast_cif_reader_used": fast_reader_used,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        marker_path,
    )
    return True


def _leaf_marker_matches(
    prepared: PreparedLeaf,
    *,
    plan_id: str,
    fast_path_mode: str = "auto",
    verify_outputs: bool = False,
) -> bool:
    output_dir = prepared.leaf_dir / BIOASSEMBLY_DIRNAME
    index_path = output_dir / LEAF_INDEX_NAME
    marker = _read_json_marker(output_dir / LEAF_SUCCESS_NAME)
    if marker is None or not index_path.is_file():
        return False
    try:
        expected_output_names = _expected_leaf_output_names(prepared)
    except (OSError, RuntimeError, ValueError):
        return False
    expected = {
        "plan_id": plan_id,
        "root_id": prepared.source_root.root_id,
        "leaf_relpath": prepared.job.leaf_relpath,
        "cif_names_sha256": prepared.job.cif_names_sha256,
        "cif_count": prepared.job.cif_count,
        "index_rows": prepared.job.cif_count,
        "index_schema_sha256": INDEX_SCHEMA_SHA256,
        "pkl_count": prepared.job.cif_count,
        "pkl_names_sha256": _hash_cif_names(expected_output_names),
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        return False
    if not _marker_file_matches(marker, index_path, prefix="index"):
        return False
    if verify_outputs:
        actual_output_names = _scan_leaf_output_names(output_dir)
        if any(
            name not in actual_output_names
            for name in expected_output_names
        ):
            return False
    if fast_path_mode != "require":
        return True
    return (
        marker.get("fast_path_used") == prepared.job.cif_count
        and marker.get("fast_cif_reader_used") == prepared.job.cif_count
    )


def _update_rank_status(
    path: Path,
    *,
    plan_id: str,
    rank: int,
    state: str,
    task_count: int,
    finished_tasks: int,
    completed: int,
    failed: int,
    started: float,
    message: Optional[str] = None,
) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    _atomic_json_dump(
        {
            "plan_id": plan_id,
            "rank": rank,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "state": state,
            "task_count": task_count,
            "finished_tasks": finished_tasks,
            "completed": completed,
            "failed": failed,
            "elapsed_seconds": elapsed,
            "files_per_second": completed / elapsed,
            "message": message,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        path,
    )


def _run_rank_impl(
    plan_dir: Path,
    *,
    rank: int,
    workers: int,
    compresslevel: int = 3,
    protenix_root_dir: Optional[Path] = None,
    fast_path_mode: str = "auto",
    durable_output: bool = False,
    max_files: Optional[int] = None,
    tail_max_lanes_per_leaf: int = 0,
) -> dict[str, Any]:
    if os.name != "posix":
        raise RuntimeError("The preloaded worker pool requires POSIX fork")
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    if not 0 <= compresslevel <= 9:
        raise ValueError(
            f"compresslevel must be in [0, 9], got {compresslevel}"
        )
    if fast_path_mode not in {"off", "auto", "require"}:
        raise ValueError(
            f"Unsupported fast path mode: {fast_path_mode!r}"
        )
    if max_files is not None and max_files <= 0:
        raise ValueError(f"max_files must be positive, got {max_files}")
    if tail_max_lanes_per_leaf < 0:
        raise ValueError(
            "tail_max_lanes_per_leaf must be non-negative, "
            f"got {tail_max_lanes_per_leaf}"
        )

    plan_dir = plan_dir.expanduser().resolve()
    plan = _load_plan(plan_dir)
    plan_id = str(plan["plan_id"])
    try:
        (plan_dir / FINALIZE_SUCCESS_NAME).unlink()
    except FileNotFoundError:
        pass
    roots = _roots_from_plan(plan)
    world_size = int(plan["world_size"])
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    rank_tasks = _load_and_validate_all_rank_tasks(
        plan_dir,
        plan=plan,
    )
    tasks = rank_tasks[rank]
    if max_files is not None:
        if workers != 1 or len(tasks) != 1:
            raise ValueError(
                "--max-files pilot mode requires --workers 1 and a rank "
                "containing exactly one chunk"
            )

    _configure_protenix_root(protenix_root_dir)
    global _PREPARED_LEAVES
    _PREPARED_LEAVES = _prepare_rank_leaves(tasks, roots)

    status_path = plan_dir / "status" / f"rank-{rank:03d}.json"
    started = time.monotonic()
    totals = {"completed": 0, "failed": 0, "skipped": 0}
    _update_rank_status(
        status_path,
        plan_id=plan_id,
        rank=rank,
        state="preparing",
        task_count=len(tasks),
        finished_tasks=0,
        completed=0,
        failed=0,
        started=started,
    )

    rank_leaf_tasks = _leaf_tasks(tasks)
    audit_last_update = time.monotonic()
    try:
        for audited_leaves, (leaf_id, leaf_tasks) in enumerate(
            rank_leaf_tasks.items(),
            start=1,
        ):
            prepared_leaf = _PREPARED_LEAVES[leaf_id]
            output_dir = (
                prepared_leaf.leaf_dir / BIOASSEMBLY_DIRNAME
            )
            has_completion_marker = (
                output_dir / LEAF_SUCCESS_NAME
            ).is_file() or any(
                _chunk_paths(prepared_leaf, task)["success"].is_file()
                for task in leaf_tasks
            )
            if has_completion_marker:
                _invalidate_missing_chunk_outputs(
                    leaf_tasks,
                    prepared_leaf,
                )
            now = time.monotonic()
            if now - audit_last_update >= 30.0:
                _update_rank_status(
                    status_path,
                    plan_id=plan_id,
                    rank=rank,
                    state="preparing",
                    task_count=len(tasks),
                    finished_tasks=0,
                    completed=0,
                    failed=0,
                    started=started,
                    message=(
                        f"output_audit_leaves={audited_leaves}/"
                        f"{len(rank_leaf_tasks)}"
                    ),
                )
                audit_last_update = now
    except BaseException as error:
        _update_rank_status(
            status_path,
            plan_id=plan_id,
            rank=rank,
            state="crashed",
            task_count=len(tasks),
            finished_tasks=0,
            completed=0,
            failed=0,
            started=started,
            message=f"{type(error).__name__}: {error}",
        )
        raise

    pending_tasks = []
    for task in tasks:
        paths = _chunk_paths(_PREPARED_LEAVES[task.leaf_id], task)
        if _chunk_marker_matches(
            task,
            plan_id=plan_id,
            fast_path_mode=fast_path_mode,
            paths=paths,
        ):
            totals["completed"] += task.size
            totals["skipped"] += task.size
        else:
            pending_tasks.append(task)

    prewarm = None
    finished_tasks = len(tasks) - len(pending_tasks)
    if pending_tasks:
        try:
            prewarm = _prewarm_parent()
        except BaseException as error:
            _update_rank_status(
                status_path,
                plan_id=plan_id,
                rank=rank,
                state="crashed",
                task_count=len(tasks),
                finished_tasks=finished_tasks,
                completed=totals["completed"],
                failed=totals["failed"],
                started=started,
                message=f"{type(error).__name__}: {error}",
            )
            raise
        _update_rank_status(
            status_path,
            plan_id=plan_id,
            rank=rank,
            state="running",
            task_count=len(tasks),
            finished_tasks=finished_tasks,
            completed=totals["completed"],
            failed=totals["failed"],
            started=started,
            message=f"prewarm={prewarm['elapsed_seconds']:.3f}s",
        )
        invocations = [
            (
                task,
                plan_id,
                compresslevel,
                fast_path_mode,
                durable_output,
                max_files,
            )
            for task in pending_tasks
        ]
        try:
            results = _run_directory_affine_tasks(
                invocations,
                workers=workers,
                tail_max_lanes_per_leaf=tail_max_lanes_per_leaf,
            )
            for result in results:
                if result.get("status") == "heartbeat":
                    _update_rank_status(
                        status_path,
                        plan_id=plan_id,
                        rank=rank,
                        state="running",
                        task_count=len(tasks),
                        finished_tasks=finished_tasks,
                        completed=totals["completed"],
                        failed=totals["failed"],
                        started=started,
                        message=(
                            f"active_tasks={result.get('active_tasks', 0)},"
                            f"queued_tasks={result.get('queued_tasks', 0)}"
                        ),
                    )
                    continue
                finished_tasks += 1
                totals["completed"] += int(result.get("completed", 0))
                totals["failed"] += int(result.get("failed", 0))
                totals["skipped"] += int(result.get("skipped", 0))
                _update_rank_status(
                    status_path,
                    plan_id=plan_id,
                    rank=rank,
                    state="running",
                    task_count=len(tasks),
                    finished_tasks=finished_tasks,
                    completed=totals["completed"],
                    failed=totals["failed"],
                    started=started,
                    message=(
                        None
                        if result.get("status") != "crashed"
                        else str(result.get("error", "chunk crashed"))
                    ),
                )
        except BaseException as error:
            _update_rank_status(
                status_path,
                plan_id=plan_id,
                rank=rank,
                state="crashed",
                task_count=len(tasks),
                finished_tasks=finished_tasks,
                completed=totals["completed"],
                failed=totals["failed"],
                started=started,
                message=f"{type(error).__name__}: {error}",
            )
            raise

    complete_leaves = 0
    try:
        for leaf_id, leaf_tasks in rank_leaf_tasks.items():
            prepared_leaf = _PREPARED_LEAVES[leaf_id]
            if _leaf_marker_matches(
                prepared_leaf,
                plan_id=plan_id,
                fast_path_mode=fast_path_mode,
            ) or _finalize_leaf(
                leaf_tasks,
                prepared_leaf,
                plan_id=plan_id,
                fast_path_mode=fast_path_mode,
                compresslevel=compresslevel,
            ):
                complete_leaves += 1
    except BaseException as error:
        _update_rank_status(
            status_path,
            plan_id=plan_id,
            rank=rank,
            state="crashed",
            task_count=len(tasks),
            finished_tasks=finished_tasks,
            completed=totals["completed"],
            failed=totals["failed"],
            started=started,
            message=f"{type(error).__name__}: {error}",
        )
        raise

    incomplete_leaves = len(rank_leaf_tasks) - complete_leaves
    final_state = (
        "complete"
        if incomplete_leaves == 0 and totals["failed"] == 0
        else "incomplete"
    )
    _update_rank_status(
        status_path,
        plan_id=plan_id,
        rank=rank,
        state=final_state,
        task_count=len(tasks),
        finished_tasks=finished_tasks,
        completed=totals["completed"],
        failed=totals["failed"],
        started=started,
        message=f"incomplete_leaves={incomplete_leaves}",
    )
    return {
        "rank": rank,
        "state": final_state,
        "task_count": len(tasks),
        "leaf_count": len(rank_leaf_tasks),
        "complete_leaves": complete_leaves,
        "incomplete_leaves": incomplete_leaves,
        "completed": totals["completed"],
        "failed": totals["failed"],
        "skipped": totals["skipped"],
        "elapsed_seconds": time.monotonic() - started,
        "prewarm": prewarm,
        "tail_max_lanes_per_leaf": tail_max_lanes_per_leaf,
    }


def run_rank(
    plan_dir: Path,
    *,
    rank: int,
    workers: int,
    compresslevel: int = 3,
    protenix_root_dir: Optional[Path] = None,
    fast_path_mode: str = "auto",
    durable_output: bool = False,
    max_files: Optional[int] = None,
    tail_max_lanes_per_leaf: int = 0,
) -> dict[str, Any]:
    resolved_plan_dir = plan_dir.expanduser().resolve()
    with _plan_activity_lock(resolved_plan_dir, exclusive=False):
        return _run_rank_impl(
            resolved_plan_dir,
            rank=rank,
            workers=workers,
            compresslevel=compresslevel,
            protenix_root_dir=protenix_root_dir,
            fast_path_mode=fast_path_mode,
            durable_output=durable_output,
            max_files=max_files,
            tail_max_lanes_per_leaf=tail_max_lanes_per_leaf,
        )


def read_status(plan_dir: Path) -> dict[str, Any]:
    plan = _load_plan(plan_dir)
    rows = []
    states: dict[str, int] = {}
    for rank in range(int(plan["world_size"])):
        path = plan_dir / "status" / f"rank-{rank:03d}.json"
        if path.is_file():
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("plan_id") != plan["plan_id"]:
                    row = {"rank": rank, "state": "stale"}
            except (OSError, ValueError):
                row = {"rank": rank, "state": "invalid"}
        else:
            row = {"rank": rank, "state": "not_started"}
        state = str(row.get("state", "unknown"))
        states[state] = states.get(state, 0) + 1
        rows.append(row)
    return {
        "plan_id": plan["plan_id"],
        "world_size": plan["world_size"],
        "states": states,
        "completed": sum(int(row.get("completed", 0)) for row in rows),
        "failed": sum(int(row.get("failed", 0)) for row in rows),
        "ranks": rows,
    }


def _write_merged_indices(
    index_paths: Sequence[Path],
    output_path: Path,
    *,
    compresslevel: int = 3,
    expected_row_count: Optional[int] = None,
) -> int:
    return _merge_chunk_indices(
        index_paths,
        output_path,
        compresslevel=compresslevel,
        expected_row_count=expected_row_count,
    )


def _validate_finalize_output_paths(
    plan_dir: Path,
    *,
    catalog_path: Path,
    merged_indices_path: Optional[Path],
) -> tuple[Path, Optional[Path]]:
    plan_dir = plan_dir.expanduser().resolve(strict=True)
    catalog_path = catalog_path.expanduser().resolve()
    outputs = [catalog_path]
    if merged_indices_path is not None:
        merged_indices_path = merged_indices_path.expanduser().resolve()
        outputs.append(merged_indices_path)
    for output_path in outputs:
        try:
            output_path.relative_to(plan_dir)
        except ValueError as error:
            raise ValueError(
                "Finalize audit outputs must stay inside the immutable "
                f"plan directory {plan_dir}: {output_path}"
            ) from error
        if output_path.name == "distillation_index.csv.gz":
            raise ValueError(
                "Refusing to use the authoritative training catalog name "
                f"for a conversion audit artifact: {output_path}"
            )
    if catalog_path.suffix != ".csv":
        raise ValueError(
            f"Leaf audit catalog must end in .csv: {catalog_path}"
        )
    if (
        merged_indices_path is not None
        and not merged_indices_path.name.endswith(".csv.gz")
    ):
        raise ValueError(
            "Merged conversion audit index must end in .csv.gz: "
            f"{merged_indices_path}"
        )
    if (
        merged_indices_path is not None
        and catalog_path == merged_indices_path
    ):
        raise ValueError("Finalize audit outputs must use distinct paths")
    return catalog_path, merged_indices_path


def _finalize_plan_impl(
    plan_dir: Path,
    *,
    catalog_path: Path,
    merged_indices_path: Optional[Path] = None,
    compresslevel: int = 3,
    fast_path_mode: str = "auto",
) -> dict[str, Any]:
    if not 0 <= compresslevel <= 9:
        raise ValueError(
            f"compresslevel must be in [0, 9], got {compresslevel}"
        )
    if fast_path_mode not in {"off", "auto", "require"}:
        raise ValueError(
            f"Unsupported fast path mode: {fast_path_mode!r}"
        )
    plan_dir = plan_dir.expanduser().resolve(strict=True)
    catalog_path, merged_indices_path = _validate_finalize_output_paths(
        plan_dir,
        catalog_path=catalog_path,
        merged_indices_path=merged_indices_path,
    )
    plan = _load_plan(plan_dir)
    finalize_success_path = plan_dir / FINALIZE_SUCCESS_NAME
    try:
        finalize_success_path.unlink()
    except FileNotFoundError:
        pass
    roots = _roots_from_plan(plan)
    rank_tasks = _load_and_validate_all_rank_tasks(
        plan_dir,
        plan=plan,
    )
    all_tasks = [
        task
        for tasks in rank_tasks
        for task in tasks
    ]
    prepared: dict[int, PreparedLeaf] = {}
    for task in all_tasks:
        if task.leaf_id in prepared:
            continue
        source_root = roots[task.root_id]
        root_path = Path(source_root.path).resolve()
        leaf_dir = (root_path / task.leaf_relpath).resolve()
        try:
            leaf_dir.relative_to(root_path)
        except ValueError as error:
            raise ValueError(
                f"Leaf path escapes source root: {task.leaf_relpath}"
            ) from error
        prepared[task.leaf_id] = PreparedLeaf(
            job=_leaf_job_from_task(task),
            source_root=source_root,
            leaf_dir=leaf_dir,
            cif_names=(),
        )
    rows = []
    incomplete = []
    index_paths = []
    complete_cifs = 0
    complete_tasks = 0
    for leaf_id, leaf_tasks in sorted(_leaf_tasks(all_tasks).items()):
        leaf = prepared[leaf_id]
        chunks_complete = all(
            _chunk_marker_matches(
                task,
                plan_id=str(plan["plan_id"]),
                fast_path_mode=fast_path_mode,
                paths=_chunk_paths(leaf, task),
            )
            for task in leaf_tasks
        )
        if (
            not chunks_complete
            or not _leaf_marker_matches(
                leaf,
                plan_id=str(plan["plan_id"]),
                fast_path_mode=fast_path_mode,
                verify_outputs=True,
            )
        ):
            incomplete.append(
                f"{leaf.source_root.root_id}:{leaf.job.leaf_relpath}"
            )
            continue
        output_dir = leaf.leaf_dir / BIOASSEMBLY_DIRNAME
        marker = _read_json_marker(output_dir / LEAF_SUCCESS_NAME) or {}
        index_path = output_dir / LEAF_INDEX_NAME
        index_paths.append(index_path)
        complete_cifs += leaf.job.cif_count
        complete_tasks += len(leaf_tasks)
        rows.append(
            {
                "artifact_role": LEAF_AUDIT_ROLE,
                "plan_id": plan["plan_id"],
                "leaf_id": leaf_id,
                "source_root_id": leaf.source_root.root_id,
                "source_db": leaf.source_root.source_db,
                "leaf_relpath": leaf.job.leaf_relpath,
                "cif_names_sha256": leaf.job.cif_names_sha256,
                "index_relpath": index_path.relative_to(
                    Path(leaf.source_root.path)
                ).as_posix(),
                "cif_count": leaf.job.cif_count,
                "index_rows": marker.get("index_rows", -1),
            }
        )

    result: dict[str, Any] = {
        "plan_id": plan["plan_id"],
        "fast_path_mode": fast_path_mode,
        "expected_tasks": int(plan["task_count"]),
        "complete_tasks": complete_tasks,
        "expected_leaves": int(plan["leaf_count"]),
        "complete_leaves": len(rows),
        "incomplete_leaves": len(incomplete),
        "expected_cifs": int(plan["cif_count"]),
        "complete_cifs": complete_cifs,
        "catalog_path": str(catalog_path),
        "first_incomplete": incomplete[:20],
    }
    if (
        incomplete
        or complete_tasks != int(plan["task_count"])
        or len(rows) != int(plan["leaf_count"])
        or complete_cifs != int(plan["cif_count"])
    ):
        raise RuntimeError(json.dumps(result, sort_keys=True))

    fields = [
        "artifact_role",
        "plan_id",
        "leaf_id",
        "source_root_id",
        "source_db",
        "leaf_relpath",
        "cif_names_sha256",
        "index_relpath",
        "cif_count",
        "index_rows",
    ]
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = catalog_path.with_name(
        f".{catalog_path.name}.tmp.{socket.gethostname()}.{os.getpid()}"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, catalog_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    if merged_indices_path is not None:
        result["merged_index_rows"] = _write_merged_indices(
            index_paths,
            merged_indices_path,
            compresslevel=compresslevel,
            expected_row_count=int(plan["cif_count"]),
        )
        result["merged_indices_path"] = str(merged_indices_path)
    finalize_marker: dict[str, Any] = {
        "artifact_role": "distillation_bioassembly_finalize",
        "plan_id": plan["plan_id"],
        "fast_path_mode": fast_path_mode,
        "complete_tasks": complete_tasks,
        "complete_leaves": len(rows),
        "complete_cifs": complete_cifs,
        "failure_count": 0,
        "merged_audit_present": merged_indices_path is not None,
        "leaf_audit_relpath": catalog_path.relative_to(
            plan_dir
        ).as_posix(),
        **_file_marker_fields(catalog_path, prefix="leaf_audit"),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if merged_indices_path is not None:
        finalize_marker.update(
            {
                "merged_audit_relpath": merged_indices_path.relative_to(
                    plan_dir
                ).as_posix(),
                "merged_audit_rows": result["merged_index_rows"],
                **_file_marker_fields(
                    merged_indices_path,
                    prefix="merged_audit",
                ),
            }
        )
    _atomic_json_dump(finalize_marker, finalize_success_path)
    result["finalize_success_path"] = str(finalize_success_path)
    return result


def finalize_plan(
    plan_dir: Path,
    *,
    catalog_path: Path,
    merged_indices_path: Optional[Path] = None,
    compresslevel: int = 3,
    fast_path_mode: str = "auto",
) -> dict[str, Any]:
    resolved_plan_dir = plan_dir.expanduser().resolve(strict=True)
    with _plan_activity_lock(resolved_plan_dir, exclusive=True):
        return _finalize_plan_impl(
            resolved_plan_dir,
            catalog_path=catalog_path,
            merged_indices_path=merged_indices_path,
            compresslevel=compresslevel,
            fast_path_mode=fast_path_mode,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Inventory CIF names and build immutable rank plans",
    )
    plan_parser.add_argument(
        "--roots-json",
        type=Path,
        default=None,
        help="Existing roots.json containing root_id/source_db/path records",
    )
    plan_parser.add_argument(
        "--source-root",
        action="append",
        type=_parse_source_root,
        default=[],
        help="ROOT_ID,SOURCE_DB,PATH; repeat for multiple roots",
    )
    plan_parser.add_argument(
        "--catalog-manifest",
        type=Path,
        default=None,
        help=(
            "Catalog manifest to bind into plan provenance; defaults to "
            "manifest.json beside --roots-json when present"
        ),
    )
    plan_parser.add_argument("--world-size", type=int, default=16)
    plan_parser.add_argument("--workers-per-rank", type=int, default=128)
    plan_parser.add_argument(
        "--max-files-per-task",
        type=int,
        default=10_000,
    )
    plan_parser.add_argument(
        "--scan-workers",
        type=int,
        default=64,
        help="Maximum concurrent directory scans during plan inventory",
    )
    plan_parser.add_argument("--plan-dir", type=Path, required=True)
    plan_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run one rank")
    run_parser.add_argument("--plan-dir", type=Path, required=True)
    run_parser.add_argument("--rank", type=int, required=True)
    run_parser.add_argument("--workers", type=int, default=128)
    run_parser.add_argument("--compresslevel", type=int, default=3)
    fast_group = run_parser.add_mutually_exclusive_group()
    fast_group.add_argument(
        "--disable-fast-path",
        dest="fast_path_mode",
        action="store_const",
        const="off",
    )
    fast_group.add_argument(
        "--require-fast-path",
        dest="fast_path_mode",
        action="store_const",
        const="require",
    )
    run_parser.set_defaults(fast_path_mode="auto")
    run_parser.add_argument("--protenix-root-dir", type=Path, default=None)
    run_parser.add_argument("--durable-output", action="store_true")
    run_parser.add_argument(
        "--tail-max-lanes-per-leaf",
        type=int,
        default=0,
        help=(
            "Only used after all remaining leaves already have an active "
            "worker; 0 allows all planned chunks, 1 forbids same-leaf "
            "parallelism"
        ),
    )
    run_parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Single-worker, single-chunk correctness pilot only",
    )

    status_parser = subparsers.add_parser(
        "status",
        help="Read small rank heartbeat files",
    )
    status_parser.add_argument("--plan-dir", type=Path, required=True)

    finalize_parser = subparsers.add_parser(
        "finalize",
        help="Validate leaf markers and write a relative-path catalog",
    )
    finalize_parser.add_argument("--plan-dir", type=Path, required=True)
    finalize_parser.add_argument("--catalog", type=Path, required=True)
    finalize_parser.add_argument("--merged-indices", type=Path, default=None)
    finalize_parser.add_argument("--compresslevel", type=int, default=3)
    finalize_fast_group = finalize_parser.add_mutually_exclusive_group()
    finalize_fast_group.add_argument(
        "--disable-fast-path",
        dest="fast_path_mode",
        action="store_const",
        const="off",
    )
    finalize_fast_group.add_argument(
        "--require-fast-path",
        dest="fast_path_mode",
        action="store_const",
        const="require",
    )
    finalize_parser.set_defaults(fast_path_mode="auto")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            roots = list(args.source_root)
            if args.roots_json is not None:
                roots.extend(_load_roots_json(args.roots_json))
            result = build_plan(
                roots,
                world_size=args.world_size,
                workers_per_rank=args.workers_per_rank,
                max_files_per_task=args.max_files_per_task,
                plan_dir=args.plan_dir,
                scan_workers=args.scan_workers,
                force=args.force,
                roots_json_path=args.roots_json,
                catalog_manifest_path=args.catalog_manifest,
            )
        elif args.command == "run":
            result = run_rank(
                args.plan_dir,
                rank=args.rank,
                workers=args.workers,
                compresslevel=args.compresslevel,
                protenix_root_dir=args.protenix_root_dir,
                fast_path_mode=args.fast_path_mode,
                durable_output=args.durable_output,
                max_files=args.max_files,
                tail_max_lanes_per_leaf=args.tail_max_lanes_per_leaf,
            )
        elif args.command == "status":
            result = read_status(args.plan_dir)
        elif args.command == "finalize":
            result = finalize_plan(
                args.plan_dir,
                catalog_path=args.catalog,
                merged_indices_path=args.merged_indices,
                compresslevel=args.compresslevel,
                fast_path_mode=args.fast_path_mode,
            )
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
        print(
            json.dumps(
                result,
                default=_json_default,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        if (
            args.command == "run"
            and result.get("state") != "complete"
            and args.max_files is None
        ):
            return 3
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
