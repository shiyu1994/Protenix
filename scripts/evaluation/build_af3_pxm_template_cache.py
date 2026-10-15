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

"""Build the locked PXM evaluation template cache in parallel."""

from __future__ import annotations

# Worker processes, not native math runtimes, own the parallelism.
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
from typing import Any, Iterator, Optional, Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from protenix.data.template.evaluation_template_cache import (
    EVALUATION_CACHE_CUTOFF_TEXT,
    EVALUATION_CACHE_MANIFEST_FILENAME,
    EVALUATION_CACHE_MAX_TEMPLATES,
    EVALUATION_CACHE_POLICY_VERSION,
    EVALUATION_CACHE_SCHEMA_VERSION,
    EVALUATION_CACHE_STATS_FILENAME,
    EVALUATION_CACHE_SUCCESS_FILENAME,
    EvaluationTemplateCacheBuilder,
    EvaluationTemplateCacheConfig,
    atomic_json_dump,
    file_sha256,
    load_sequence_sources,
    sequence_sha256,
    sequence_sources_sha256,
)


_WORKER_BUILDER: Optional[EvaluationTemplateCacheBuilder] = None
_WORKER_RESUME = False


def _initialize_worker(
    config: EvaluationTemplateCacheConfig,
    resume: bool,
) -> None:
    global _WORKER_BUILDER, _WORKER_RESUME
    _WORKER_BUILDER = EvaluationTemplateCacheBuilder(config)
    _WORKER_RESUME = bool(resume)


def _build_worker(task: tuple[str, str]) -> dict[str, Any]:
    sequence, source_hit_path = task
    started_at = time.time()
    try:
        if _WORKER_BUILDER is None:
            raise RuntimeError("evaluation cache worker was not initialized")
        return _WORKER_BUILDER.build(
            sequence=sequence,
            source_hit_path=source_hit_path,
            resume=_WORKER_RESUME,
        )
    except BaseException as exc:
        return {
            "status": "failed",
            "sequence_sha256": sequence_sha256(sequence),
            "source_hit_path": source_hit_path,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started_at,
        }


def _stats_payload(
    *,
    counters: collections.Counter[str],
    completed: int,
    total: int,
    workers: int,
    resume: bool,
    started_at: float,
) -> dict[str, Any]:
    elapsed_seconds = time.time() - started_at
    status_counts = dict(sorted(counters.items()))
    status_counts.setdefault("failed", 0)
    return {
        "schema_version": EVALUATION_CACHE_SCHEMA_VERSION,
        "policy_version": EVALUATION_CACHE_POLICY_VERSION,
        "cutoff_date": EVALUATION_CACHE_CUTOFF_TEXT,
        "max_templates": EVALUATION_CACHE_MAX_TEMPLATES,
        "workers": workers,
        "resume": resume,
        "completed_tasks": completed,
        "total_tasks": total,
        "remaining_tasks": max(0, total - completed),
        "failed_tasks": int(status_counts["failed"]),
        "missing_artifacts": 0,
        "status_counts": status_counts,
        "elapsed_seconds": elapsed_seconds,
        "sequences_per_second": (
            completed / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
        "updated_at_unix": time.time(),
    }


def _result_iterator(
    *,
    tasks: Sequence[tuple[str, str]],
    config: EvaluationTemplateCacheConfig,
    workers: int,
    resume: bool,
    start_method: str,
    max_tasks_per_child: int,
) -> Iterator[dict[str, Any]]:
    if workers == 1:
        _initialize_worker(config, resume)
        yield from map(_build_worker, tasks)
        return

    context = multiprocessing.get_context(start_method)
    pool = context.Pool(
        processes=workers,
        initializer=_initialize_worker,
        initargs=(config, resume),
        maxtasksperchild=(
            max_tasks_per_child if max_tasks_per_child > 0 else None
        ),
    )
    try:
        yield from pool.imap_unordered(_build_worker, tasks, chunksize=1)
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a content-addressed, date-locked, top-four PXM evaluation "
            "template cache."
        )
    )
    parser.add_argument("--sequence-sources", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--mmcif-dir", required=True)
    parser.add_argument("--kalign-binary-path", required=True)
    parser.add_argument("--release-dates-path", required=True)
    parser.add_argument("--obsolete-pdbs-path", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--start-method",
        choices=("fork", "spawn", "forkserver"),
        default="fork",
    )
    parser.add_argument("--max-tasks-per-child", type=int, default=0)
    parser.add_argument("--gzip-compresslevel", type=int, default=1)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--success", default=None)
    args = parser.parse_args(argv)

    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if args.max_tasks_per_child < 0:
        parser.error("--max-tasks-per-child must be non-negative")
    if not 0 <= args.gzip_compresslevel <= 9:
        parser.error("--gzip-compresslevel must be in [0, 9]")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    sequence_sources_path = Path(args.sequence_sources)
    cache_root = Path(args.cache_root)
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else cache_root / EVALUATION_CACHE_MANIFEST_FILENAME
    )
    stats_path = (
        Path(args.stats)
        if args.stats
        else cache_root / EVALUATION_CACHE_STATS_FILENAME
    )
    success_path = (
        Path(args.success)
        if args.success
        else cache_root / EVALUATION_CACHE_SUCCESS_FILENAME
    )

    for required_file in (
        sequence_sources_path,
        Path(args.kalign_binary_path),
        Path(args.release_dates_path),
        Path(args.obsolete_pdbs_path),
    ):
        if not required_file.is_file():
            raise FileNotFoundError(required_file)
    if not Path(args.mmcif_dir).is_dir():
        raise NotADirectoryError(args.mmcif_dir)

    sequence_sources = load_sequence_sources(sequence_sources_path)
    tasks = sorted(
        sequence_sources.items(),
        key=lambda item: sequence_sha256(item[0]),
    )
    config = EvaluationTemplateCacheConfig(
        cache_root=str(cache_root),
        mmcif_dir=args.mmcif_dir,
        kalign_binary_path=args.kalign_binary_path,
        release_dates_path=args.release_dates_path,
        obsolete_pdbs_path=args.obsolete_pdbs_path,
        gzip_compresslevel=args.gzip_compresslevel,
    )

    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    success_path.parent.mkdir(parents=True, exist_ok=True)
    # A previous success marker must never survive an incomplete new attempt.
    try:
        success_path.unlink()
    except FileNotFoundError:
        pass

    started_at = time.time()
    counters: collections.Counter[str] = collections.Counter()
    completed = 0
    results: list[dict[str, Any]] = []
    print(
        json.dumps(
            {
                "event": "start",
                "schema_version": EVALUATION_CACHE_SCHEMA_VERSION,
                "policy_version": EVALUATION_CACHE_POLICY_VERSION,
                "cutoff_date": EVALUATION_CACHE_CUTOFF_TEXT,
                "max_templates": EVALUATION_CACHE_MAX_TEMPLATES,
                "sequence_count": len(tasks),
                "workers": args.workers,
                "cache_root": str(cache_root),
                "manifest": str(manifest_path),
                "stats": str(stats_path),
                "success": str(success_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    with open(manifest_path, "w", encoding="utf-8") as manifest_handle:
        for completed, result in enumerate(
            _result_iterator(
                tasks=tasks,
                config=config,
                workers=args.workers,
                resume=args.resume,
                start_method=args.start_method,
                max_tasks_per_child=args.max_tasks_per_child,
            ),
            start=1,
        ):
            results.append(result)
            counters[str(result.get("status", "unknown"))] += 1
            manifest_handle.write(
                json.dumps(result, sort_keys=True, default=str) + "\n"
            )
            manifest_handle.flush()
            if completed % args.progress_every == 0 or completed == len(tasks):
                stats = _stats_payload(
                    counters=counters,
                    completed=completed,
                    total=len(tasks),
                    workers=args.workers,
                    resume=args.resume,
                    started_at=started_at,
                )
                atomic_json_dump(stats, stats_path)
                print(json.dumps(stats, sort_keys=True), flush=True)

    final_stats = _stats_payload(
        counters=counters,
        completed=completed,
        total=len(tasks),
        workers=args.workers,
        resume=args.resume,
        started_at=started_at,
    )
    missing_artifacts = sum(
        1
        for result in results
        if result.get("status") != "failed"
        and not Path(str(result.get("output_path", ""))).is_file()
    )
    final_stats["missing_artifacts"] = missing_artifacts
    atomic_json_dump(final_stats, stats_path)

    failed_count = int(counters.get("failed", 0))
    if completed != len(tasks) or failed_count or missing_artifacts:
        print(
            json.dumps(
                {
                    "event": "failed",
                    "completed_tasks": completed,
                    "total_tasks": len(tasks),
                    "failed_count": failed_count,
                    "missing_count": missing_artifacts,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1

    success = {
        "schema_version": EVALUATION_CACHE_SCHEMA_VERSION,
        "policy_version": EVALUATION_CACHE_POLICY_VERSION,
        "cutoff_date": EVALUATION_CACHE_CUTOFF_TEXT,
        "max_templates": EVALUATION_CACHE_MAX_TEMPLATES,
        "sequence_count": len(tasks),
        "sequence_sources_path": str(sequence_sources_path.resolve()),
        "sequence_sources_file_sha256": file_sha256(sequence_sources_path),
        "sequence_sources_sha256": sequence_sources_sha256(sequence_sources),
        "cache_root": str(cache_root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "stats_path": str(stats_path.resolve()),
        "stats_sha256": file_sha256(stats_path),
        "status_counts": dict(sorted(counters.items())),
        "failed_count": 0,
        "missing_count": 0,
        "completed_at_unix": time.time(),
    }
    atomic_json_dump(success, success_path)
    print(
        json.dumps(
            {
                "event": "success",
                "sequence_count": len(tasks),
                "success": str(success_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
