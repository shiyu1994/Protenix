#!/usr/bin/env python3
"""Build sequence-keyed, preprocessed template feature caches in parallel."""

from __future__ import annotations

# Set all common native math runtimes to one thread before NumPy/Biopython are
# imported.  Parallelism is supplied exclusively by worker processes.
import os

for _thread_environment_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_thread_environment_variable] = "1"

import argparse
import collections
import json
import multiprocessing
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

# Make direct ``python scripts/template_cache/...py`` invocation independent
# of the caller's PYTHONPATH.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from protenix.data.template.sequence_template_cache import (
    DEFAULT_CACHE_FILENAME,
    DEFAULT_HIT_FILENAME,
    MAX_ALIGNMENT_IDENTITY,
    MAX_VALID_TEMPLATES,
    POLICY_VERSION,
    SCHEMA_VERSION,
    TOP_K_QUALITY,
    SequenceTemplateCacheBuilder,
    SequenceTemplateCacheConfig,
    normalize_relative_directory,
    sequence_sha256,
    sequence_shard,
)


_WORKER_BUILDER: Optional[SequenceTemplateCacheBuilder] = None
_WORKER_RESUME = False


def _initialize_worker(
    config: SequenceTemplateCacheConfig,
    resume: bool,
) -> None:
    global _WORKER_BUILDER, _WORKER_RESUME
    _WORKER_BUILDER = SequenceTemplateCacheBuilder(config)
    _WORKER_RESUME = resume


def _build_worker(task: Tuple[str, str]) -> Dict[str, Any]:
    sequence, relative_directory = task
    started_at = time.time()
    try:
        if _WORKER_BUILDER is None:
            raise RuntimeError("template cache worker was not initialized")
        return _WORKER_BUILDER.build(
            sequence=sequence,
            relative_directory=relative_directory,
            resume=_WORKER_RESUME,
        )
    except BaseException as exc:
        return {
            "status": "failed",
            "sequence_sha256": sequence_sha256(sequence),
            "source_relative_dir": relative_directory,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started_at,
        }


def _load_sequence_mapping(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict):
        raise TypeError("sequence mapping must be a JSON object")

    normalized = {}
    seen_directories = {}
    for sequence, relative_directory in mapping.items():
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("every sequence mapping key must be a non-empty string")
        relative_directory = normalize_relative_directory(relative_directory)
        other_sequence = seen_directories.get(relative_directory)
        if other_sequence is not None and other_sequence != sequence:
            raise ValueError(
                "multiple sequences map to the same relative directory: "
                f"{relative_directory!r}"
            )
        seen_directories[relative_directory] = sequence
        normalized[sequence] = relative_directory
    return normalized


def _iter_tasks(
    mapping: Mapping[str, str],
    *,
    shard_id: int,
    num_shards: int,
    limit: Optional[int],
    requested_sequences: Sequence[str],
) -> Iterator[Tuple[str, str]]:
    if requested_sequences:
        seen = set()
        selected_items = []
        for sequence in requested_sequences:
            if sequence in seen:
                continue
            seen.add(sequence)
            if sequence not in mapping:
                raise KeyError(
                    f"requested sequence is absent from mapping: {sequence!r}"
                )
            selected_items.append((sequence, mapping[sequence]))
    else:
        selected_items = (
            (sequence, relative_directory)
            for sequence, relative_directory in mapping.items()
            if sequence_shard(sequence, num_shards) == shard_id
        )

    emitted = 0
    for task in selected_items:
        if limit is not None and emitted >= limit:
            break
        yield task
        emitted += 1


def _atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _default_metadata_paths(
    cache_root: str,
    shard_id: int,
    num_shards: int,
) -> Tuple[Path, Path]:
    metadata_directory = Path(cache_root) / "_manifests"
    suffix = f"shard-{shard_id:05d}-of-{num_shards:05d}"
    return (
        metadata_directory / f"manifest.{suffix}.jsonl",
        metadata_directory / f"stats.{suffix}.json",
    )


def _stats_payload(
    *,
    args: argparse.Namespace,
    counters: Mapping[str, int],
    completed: int,
    total: int,
    started_at: float,
) -> Dict[str, Any]:
    elapsed_seconds = time.time() - started_at
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "workers": args.workers,
        "resume": args.resume,
        "completed_tasks": completed,
        "total_tasks": total,
        "remaining_tasks": max(0, total - completed),
        "status_counts": dict(sorted(counters.items())),
        "elapsed_seconds": elapsed_seconds,
        "sequences_per_second": (
            completed / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
        "updated_at_unix": time.time(),
    }


def _run_tasks(
    *,
    args: argparse.Namespace,
    config: SequenceTemplateCacheConfig,
    tasks: Sequence[Tuple[str, str]],
    manifest_path: Path,
    stats_path: Path,
) -> int:
    started_at = time.time()
    counters: collections.Counter[str] = collections.Counter()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_mode = "a" if args.resume else "w"

    def consume_results() -> Iterable[Dict[str, Any]]:
        if args.workers == 1:
            _initialize_worker(config, args.resume)
            return map(_build_worker, tasks)

        context = multiprocessing.get_context(args.start_method)
        pool = context.Pool(
            processes=args.workers,
            initializer=_initialize_worker,
            initargs=(config, args.resume),
            maxtasksperchild=(
                args.max_tasks_per_child
                if args.max_tasks_per_child > 0
                else None
            ),
        )

        def result_iterator() -> Iterator[Dict[str, Any]]:
            try:
                yield from pool.imap_unordered(
                    _build_worker,
                    tasks,
                    chunksize=1,
                )
                pool.close()
            except BaseException:
                pool.terminate()
                raise
            finally:
                pool.join()

        return result_iterator()

    completed = 0
    with open(manifest_path, manifest_mode, encoding="utf-8") as manifest_handle:
        for completed, result in enumerate(consume_results(), start=1):
            status = str(result.get("status", "unknown"))
            counters[status] += 1
            manifest_handle.write(
                json.dumps(result, sort_keys=True, default=str) + "\n"
            )
            manifest_handle.flush()

            if completed % args.progress_every == 0 or completed == len(tasks):
                stats = _stats_payload(
                    args=args,
                    counters=counters,
                    completed=completed,
                    total=len(tasks),
                    started_at=started_at,
                )
                _atomic_json_dump(stats, stats_path)
                print(json.dumps(stats, sort_keys=True), flush=True)

    if not tasks:
        _atomic_json_dump(
            _stats_payload(
                args=args,
                counters=counters,
                completed=0,
                total=0,
                started_at=started_at,
            ),
            stats_path,
        )

    return 1 if counters.get("failed", 0) else 0


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one deterministic top60/uniform20/id<=0.90 template "
            "feature pickle per mapped protein sequence."
        )
    )
    parser.add_argument("--sequence-mapping", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--mmcif-dir", required=True)
    parser.add_argument("--kalign-binary-path", default="/usr/bin/kalign")
    parser.add_argument("--obsolete-pdbs-path", default="")
    parser.add_argument("--parsed-mmcif-cache-dir", default=None)
    parser.add_argument("--hit-filename", default=DEFAULT_HIT_FILENAME)
    parser.add_argument("--cache-filename", default=DEFAULT_CACHE_FILENAME)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help=(
            "Build exactly this sequence (repeatable); explicit sequences "
            "bypass hash-shard selection."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--start-method",
        choices=("fork", "spawn", "forkserver"),
        default="fork",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=0,
        help=(
            "Recycle workers after N sequences; 0 (default) keeps workers "
            "resident for maximum throughput."
        ),
    )
    parser.add_argument("--gzip-compresslevel", type=int, default=1)
    args = parser.parse_args(argv)

    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard-id must satisfy 0 <= shard-id < num-shards")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if args.max_tasks_per_child < 0:
        parser.error("--max-tasks-per-child must be non-negative")
    if not 0 <= args.gzip_compresslevel <= 9:
        parser.error("--gzip-compresslevel must be in [0, 9]")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    for required_file in (
        args.sequence_mapping,
        args.kalign_binary_path,
    ):
        if not Path(required_file).is_file():
            raise FileNotFoundError(required_file)
    for required_directory in (
        args.source_root,
        args.mmcif_dir,
    ):
        if not Path(required_directory).is_dir():
            raise NotADirectoryError(required_directory)
    if args.obsolete_pdbs_path and not Path(args.obsolete_pdbs_path).is_file():
        raise FileNotFoundError(args.obsolete_pdbs_path)

    mapping = _load_sequence_mapping(args.sequence_mapping)
    tasks = list(
        _iter_tasks(
            mapping,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            limit=args.limit,
            requested_sequences=args.sequence,
        )
    )
    default_manifest, default_stats = _default_metadata_paths(
        args.cache_root,
        args.shard_id,
        args.num_shards,
    )
    manifest_path = Path(args.manifest) if args.manifest else default_manifest
    stats_path = Path(args.stats) if args.stats else default_stats

    config = SequenceTemplateCacheConfig(
        source_root=args.source_root,
        cache_root=args.cache_root,
        mmcif_dir=args.mmcif_dir,
        kalign_binary_path=args.kalign_binary_path,
        obsolete_pdbs_path=args.obsolete_pdbs_path,
        parsed_mmcif_cache_dir=args.parsed_mmcif_cache_dir,
        hit_filename=args.hit_filename,
        cache_filename=args.cache_filename,
        top_k_quality=TOP_K_QUALITY,
        max_valid_templates=MAX_VALID_TEMPLATES,
        max_alignment_identity=MAX_ALIGNMENT_IDENTITY,
        gzip_compresslevel=args.gzip_compresslevel,
    )

    print(
        json.dumps(
            {
                "event": "start",
                "schema_version": SCHEMA_VERSION,
                "policy_version": POLICY_VERSION,
                "mapping_sequences": len(mapping),
                "selected_tasks": len(tasks),
                "workers": args.workers,
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "manifest": str(manifest_path),
                "stats": str(stats_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return _run_tasks(
        args=args,
        config=config,
        tasks=tasks,
        manifest_path=manifest_path,
        stats_path=stats_path,
    )


if __name__ == "__main__":
    sys.exit(main())
