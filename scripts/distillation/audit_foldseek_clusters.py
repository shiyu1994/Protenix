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

"""Audit a Foldseek representative-member TSV with bounded Python memory.

Foldseek names a loose structure input by its filename after removing the
structure suffix.  This program derives those expected IDs from a one-path-per-
line manifest, streams both inputs into scratch files, and delegates all
large-set operations to an external GNU ``sort``.  Python memory use is
therefore independent of the number of proteins.

The scratch parent is mandatory and must resolve below ``/mnt`` or
``/dev/shm`` on a filesystem other than ``/``.  The program never falls back
to ``/tmp`` or the process-wide default temporary directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterator


SCHEMA_VERSION = 1
ARTIFACT_ROLE = "foldseek_representative_member_audit"
STRUCTURE_SUFFIX = re.compile(r"\.(?:cif|mmcif|pdb)(?:\.gz)?$", re.IGNORECASE)
SORT_MEMORY_PATTERN = re.compile(r"^[1-9][0-9]*(?:[KMGTP](?:i?B)?)?$")
SAFE_TEMP_PREFIXES = (Path("/mnt"), Path("/dev/shm"))


class FoldseekAuditError(RuntimeError):
    """Raised when the audit cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class RunStatistics:
    """Run-length statistics for one sorted, newline-delimited ID file."""

    rows: int
    unique_values: int
    repeated_values: int
    repeated_excess_rows: int
    max_run_size: int
    max_run_values: list[str]
    repeated_value_samples: list[str]


@dataclass(frozen=True, slots=True)
class DifferenceStatistics:
    """Set difference counts and bounded samples for two sorted ID files."""

    left_only: int
    right_only: int
    left_only_samples: list[str]
    right_only_samples: list[str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_below(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_tmp_parent(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise FoldseekAuditError(
            f"Scratch directory must already exist and be resolvable: {candidate}"
        ) from exc
    if not resolved.is_dir():
        raise FoldseekAuditError(f"Scratch path is not a directory: {resolved}")
    if not any(
        resolved == prefix or _is_below(resolved, prefix)
        for prefix in SAFE_TEMP_PREFIXES
    ):
        raise FoldseekAuditError(
            f"Scratch directory must resolve below /mnt or /dev/shm, got: {resolved}"
        )
    try:
        root_device = Path("/").stat().st_dev
        scratch_device = resolved.stat().st_dev
    except OSError as exc:
        raise FoldseekAuditError(
            f"Cannot inspect scratch filesystem for {resolved}: {exc}"
        ) from exc
    if scratch_device == root_device:
        raise FoldseekAuditError(
            f"Scratch directory resolves to the root filesystem: {resolved}"
        )
    if not os.access(resolved, os.W_OK | os.X_OK):
        raise FoldseekAuditError(f"Scratch directory is not writable: {resolved}")
    return resolved


def _normalize_input(path: Path | str, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise FoldseekAuditError(f"Cannot resolve {label}: {candidate}") from exc
    if not resolved.is_file():
        raise FoldseekAuditError(f"{label} is not a regular file: {resolved}")
    return resolved


def _normalize_report(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise FoldseekAuditError(
            f"Cannot resolve report parent directory: {candidate.parent}"
        ) from exc
    if not parent.is_dir():
        raise FoldseekAuditError(f"Report parent is not a directory: {parent}")
    return parent / candidate.name


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _line_body(raw_line: bytes, *, path: Path, line_number: int) -> str:
    body = raw_line
    if body.endswith(b"\n"):
        body = body[:-1]
        if body.endswith(b"\r"):
            body = body[:-1]
    try:
        value = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FoldseekAuditError(
            f"{path}: line {line_number} is not valid UTF-8"
        ) from exc
    if "\r" in value or "\n" in value or "\0" in value:
        raise FoldseekAuditError(
            f"{path}: line {line_number} contains a forbidden control character"
        )
    return value


def foldseek_id_from_manifest_path(
    manifest_path: str,
    *,
    line_number: int = 0,
) -> str:
    """Return Foldseek's loose-file identifier for one manifest path."""

    context = f"manifest line {line_number}" if line_number else "manifest path"
    if not manifest_path or "\t" in manifest_path:
        raise FoldseekAuditError(f"{context} must contain exactly one non-empty path")
    relative = PurePosixPath(manifest_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise FoldseekAuditError(f"{context} is unsafe: {manifest_path!r}")
    filename = relative.name
    match = STRUCTURE_SUFFIX.search(filename)
    if match is None or match.start() == 0:
        raise FoldseekAuditError(
            f"{context} does not end in .cif, .mmcif, .pdb, or a gzipped variant: "
            f"{manifest_path!r}"
        )
    return filename[: match.start()]


def _materialize_manifest_ids(
    source: Path,
    destination: Path,
    *,
    progress_every: int,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    rows = 0
    before = source.stat()
    with source.open("rb") as input_stream, destination.open(
        "w", encoding="utf-8", buffering=1024 * 1024
    ) as output_stream:
        for rows, raw_line in enumerate(input_stream, start=1):
            digest.update(raw_line)
            manifest_path = _line_body(
                raw_line,
                path=source,
                line_number=rows,
            )
            identifier = foldseek_id_from_manifest_path(
                manifest_path,
                line_number=rows,
            )
            output_stream.write(identifier)
            output_stream.write("\n")
            if progress_every and rows % progress_every == 0:
                print(f"[audit] streamed {rows:,} manifest rows", file=sys.stderr)
    after = source.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise FoldseekAuditError(f"Manifest changed while it was being read: {source}")
    if rows == 0:
        raise FoldseekAuditError(f"Manifest is empty: {source}")
    return rows, digest.hexdigest()


def _materialize_cluster_columns(
    source: Path,
    members_path: Path,
    representatives_path: Path,
    self_representatives_path: Path,
    *,
    progress_every: int,
) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    rows = 0
    self_edge_rows = 0
    before = source.stat()
    with (
        source.open("rb") as input_stream,
        members_path.open("w", encoding="utf-8", buffering=1024 * 1024) as members,
        representatives_path.open(
            "w", encoding="utf-8", buffering=1024 * 1024
        ) as representatives,
        self_representatives_path.open(
            "w", encoding="utf-8", buffering=1024 * 1024
        ) as self_representatives,
    ):
        for rows, raw_line in enumerate(input_stream, start=1):
            digest.update(raw_line)
            value = _line_body(raw_line, path=source, line_number=rows)
            fields = value.split("\t")
            if len(fields) != 2 or not fields[0] or not fields[1]:
                raise FoldseekAuditError(
                    f"{source}: line {rows} must contain exactly two non-empty "
                    "tab-separated fields"
                )
            representative, member = fields
            representatives.write(representative)
            representatives.write("\n")
            members.write(member)
            members.write("\n")
            if representative == member:
                self_representatives.write(representative)
                self_representatives.write("\n")
                self_edge_rows += 1
            if progress_every and rows % progress_every == 0:
                print(f"[audit] streamed {rows:,} cluster rows", file=sys.stderr)
    after = source.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise FoldseekAuditError(f"Cluster TSV changed while it was being read: {source}")
    if rows == 0:
        raise FoldseekAuditError(f"Cluster TSV is empty: {source}")
    return rows, self_edge_rows, digest.hexdigest()


def _external_sort(
    path: Path,
    *,
    sort_binary: str,
    scratch_directory: Path,
    parallel: int,
    memory: str,
) -> None:
    command = [
        sort_binary,
        "--temporary-directory",
        str(scratch_directory),
        f"--parallel={parallel}",
        f"--buffer-size={memory}",
        "--output",
        str(path),
        str(path),
    ]
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    environment["TMPDIR"] = str(scratch_directory)
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        raise FoldseekAuditError(
            f"External sort failed for {path} with exit code "
            f"{completed.returncode}: {detail}"
        )


def _iter_sorted_values(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith("\n"):
                value = line
            else:
                value = line[:-1]
            if not value:
                raise FoldseekAuditError(
                    f"Internal sorted file {path} has an empty line at {line_number}"
                )
            yield value


def _iter_unique_sorted_values(path: Path) -> Iterator[str]:
    previous: str | None = None
    for value in _iter_sorted_values(path):
        if value != previous:
            yield value
            previous = value


def _scan_runs(path: Path, *, sample_limit: int) -> RunStatistics:
    rows = 0
    unique_values = 0
    repeated_values = 0
    repeated_excess_rows = 0
    max_run_size = 0
    max_run_values: list[str] = []
    repeated_samples: list[str] = []
    previous: str | None = None
    run_size = 0

    def finish_run(value: str | None, size: int) -> None:
        nonlocal unique_values
        nonlocal repeated_values
        nonlocal repeated_excess_rows
        nonlocal max_run_size
        nonlocal max_run_values
        if value is None:
            return
        unique_values += 1
        if size > 1:
            repeated_values += 1
            repeated_excess_rows += size - 1
            if len(repeated_samples) < sample_limit:
                repeated_samples.append(value)
        if size > max_run_size:
            max_run_size = size
            max_run_values = [value] if sample_limit else []
        elif size == max_run_size and len(max_run_values) < sample_limit:
            max_run_values.append(value)

    for value in _iter_sorted_values(path):
        rows += 1
        if previous is None:
            previous = value
            run_size = 1
        elif value == previous:
            run_size += 1
        else:
            finish_run(previous, run_size)
            previous = value
            run_size = 1
    finish_run(previous, run_size)
    return RunStatistics(
        rows=rows,
        unique_values=unique_values,
        repeated_values=repeated_values,
        repeated_excess_rows=repeated_excess_rows,
        max_run_size=max_run_size,
        max_run_values=max_run_values,
        repeated_value_samples=repeated_samples,
    )


def _next_or_none(iterator: Iterator[str]) -> str | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _compare_unique_sets(
    left_path: Path,
    right_path: Path,
    *,
    sample_limit: int,
) -> DifferenceStatistics:
    left_iterator = _iter_unique_sorted_values(left_path)
    right_iterator = _iter_unique_sorted_values(right_path)
    left = _next_or_none(left_iterator)
    right = _next_or_none(right_iterator)
    left_only = 0
    right_only = 0
    left_samples: list[str] = []
    right_samples: list[str] = []

    while left is not None or right is not None:
        if right is None or (left is not None and left < right):
            left_only += 1
            if len(left_samples) < sample_limit:
                left_samples.append(left)
            left = _next_or_none(left_iterator)
        elif left is None or right < left:
            right_only += 1
            if len(right_samples) < sample_limit:
                right_samples.append(right)
            right = _next_or_none(right_iterator)
        else:
            left = _next_or_none(left_iterator)
            right = _next_or_none(right_iterator)
    return DifferenceStatistics(
        left_only=left_only,
        right_only=right_only,
        left_only_samples=left_samples,
        right_only_samples=right_samples,
    )


def _report_base(
    *,
    manifest: Path,
    clusters: Path,
    tmp_parent: Path | None,
    sort_parallel: int,
    sort_memory: str,
    sample_limit: int,
    started_at: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": ARTIFACT_ROLE,
        "status": "running",
        "started_at_utc": started_at,
        "inputs": {
            "manifest_paths_tsv": str(manifest),
            "clusters_tsv": str(clusters),
        },
        "parameters": {
            "tmp_parent": str(tmp_parent) if tmp_parent is not None else None,
            "sort_parallel": sort_parallel,
            "sort_memory": sort_memory,
            "sample_limit": sample_limit,
        },
    }


def audit_foldseek_clusters(
    *,
    manifest_paths_tsv: Path | str,
    clusters_tsv: Path | str,
    report_json: Path | str,
    tmp_dir: Path | str,
    sort_parallel: int = 8,
    sort_memory: str = "4G",
    sample_limit: int = 20,
    progress_every: int = 1_000_000,
    min_free_gb: float = 4.0,
    keep_tmp_on_failure: bool = False,
) -> dict[str, object]:
    """Run the complete audit and atomically publish its JSON report.

    Data-quality failures return a report whose status is ``"failed"``.
    Operational or malformed-input failures publish an ``"error"`` report and
    raise :class:`FoldseekAuditError`.
    """

    started_monotonic = time.monotonic()
    started_at = _utc_now()
    report_path = _normalize_report(report_json)
    manifest = _normalize_input(manifest_paths_tsv, label="manifest paths TSV")
    clusters = _normalize_input(clusters_tsv, label="clusters TSV")
    if report_path in {manifest, clusters}:
        raise FoldseekAuditError("Report path must not overwrite either input file")
    if sort_parallel < 1:
        raise FoldseekAuditError("sort_parallel must be at least 1")
    if not SORT_MEMORY_PATTERN.fullmatch(sort_memory):
        raise FoldseekAuditError(
            "sort_memory must be a positive GNU sort size such as 512M or 4G"
        )
    if sample_limit < 0:
        raise FoldseekAuditError("sample_limit must be non-negative")
    if progress_every < 0:
        raise FoldseekAuditError("progress_every must be non-negative")
    if min_free_gb < 0:
        raise FoldseekAuditError("min_free_gb must be non-negative")

    tmp_parent: Path | None = None
    run_tmp: Path | None = None
    report = _report_base(
        manifest=manifest,
        clusters=clusters,
        tmp_parent=None,
        sort_parallel=sort_parallel,
        sort_memory=sort_memory,
        sample_limit=sample_limit,
        started_at=started_at,
    )
    failed = False
    try:
        tmp_parent = _validate_tmp_parent(tmp_dir)
        report["parameters"]["tmp_parent"] = str(tmp_parent)  # type: ignore[index]
        available_bytes = shutil.disk_usage(tmp_parent).free
        required_bytes = int(min_free_gb * 1024**3)
        report["scratch"] = {
            "available_bytes_at_start": available_bytes,
            "minimum_free_bytes": required_bytes,
        }
        if available_bytes < required_bytes:
            raise FoldseekAuditError(
                f"Scratch filesystem has {available_bytes / 1024**3:.2f} GiB free; "
                f"{min_free_gb:.2f} GiB is required"
            )
        sort_binary = shutil.which("sort")
        if sort_binary is None:
            raise FoldseekAuditError("GNU sort is not available on PATH")
        run_tmp = Path(
            tempfile.mkdtemp(prefix="foldseek-cluster-audit-", dir=tmp_parent)
        )
        manifest_ids = run_tmp / "manifest.ids"
        members = run_tmp / "members.ids"
        representatives = run_tmp / "representatives.ids"
        self_representatives = run_tmp / "self-representatives.ids"

        print("[audit] materializing manifest IDs", file=sys.stderr)
        manifest_rows, manifest_sha256 = _materialize_manifest_ids(
            manifest,
            manifest_ids,
            progress_every=progress_every,
        )
        print("[audit] materializing cluster columns", file=sys.stderr)
        cluster_rows, self_edge_rows, clusters_sha256 = _materialize_cluster_columns(
            clusters,
            members,
            representatives,
            self_representatives,
            progress_every=progress_every,
        )

        for label, path in (
            ("manifest IDs", manifest_ids),
            ("members", members),
            ("representatives", representatives),
            ("self representatives", self_representatives),
        ):
            print(f"[audit] external sort: {label}", file=sys.stderr)
            _external_sort(
                path,
                sort_binary=sort_binary,
                scratch_directory=run_tmp,
                parallel=sort_parallel,
                memory=sort_memory,
            )

        manifest_runs = _scan_runs(manifest_ids, sample_limit=sample_limit)
        member_runs = _scan_runs(members, sample_limit=sample_limit)
        representative_runs = _scan_runs(
            representatives,
            sample_limit=sample_limit,
        )
        self_runs = _scan_runs(
            self_representatives,
            sample_limit=sample_limit,
        )
        expected_vs_members = _compare_unique_sets(
            manifest_ids,
            members,
            sample_limit=sample_limit,
        )
        expected_vs_representatives = _compare_unique_sets(
            manifest_ids,
            representatives,
            sample_limit=sample_limit,
        )
        members_vs_representatives = _compare_unique_sets(
            members,
            representatives,
            sample_limit=sample_limit,
        )
        representatives_vs_self = _compare_unique_sets(
            representatives,
            self_representatives,
            sample_limit=sample_limit,
        )

        counts = {
            "manifest_rows": manifest_rows,
            "manifest_unique_ids": manifest_runs.unique_values,
            "cluster_rows": cluster_rows,
            "cluster_count": representative_runs.unique_values,
            "member_unique_ids": member_runs.unique_values,
            "maximum_cluster_size": representative_runs.max_run_size,
            "self_edge_rows": self_edge_rows,
            "representatives_with_self_edge": self_runs.unique_values,
        }
        violations = {
            "manifest_duplicate_ids": manifest_runs.repeated_values,
            "manifest_duplicate_excess_rows": manifest_runs.repeated_excess_rows,
            "duplicate_member_ids": member_runs.repeated_values,
            "duplicate_member_excess_rows": member_runs.repeated_excess_rows,
            "missing_members": expected_vs_members.left_only,
            "extra_members": expected_vs_members.right_only,
            "representatives_not_in_manifest": expected_vs_representatives.right_only,
            "representatives_not_observed_as_members": (
                members_vs_representatives.right_only
            ),
            "representatives_without_self_edge": representatives_vs_self.left_only,
            "duplicate_self_edge_representatives": self_runs.repeated_values,
            "duplicate_self_edge_excess_rows": self_runs.repeated_excess_rows,
        }
        checks = {
            "row_count_matches_manifest": cluster_rows == manifest_rows,
            "manifest_ids_are_unique": manifest_runs.repeated_values == 0,
            "members_are_unique": member_runs.repeated_values == 0,
            "member_set_matches_manifest": (
                expected_vs_members.left_only == 0
                and expected_vs_members.right_only == 0
            ),
            "all_representatives_are_manifest_ids": (
                expected_vs_representatives.right_only == 0
            ),
            "all_representatives_are_observed_as_members": (
                members_vs_representatives.right_only == 0
            ),
            "each_representative_has_exactly_one_self_edge": (
                representatives_vs_self.left_only == 0
                and self_runs.repeated_values == 0
            ),
        }
        samples = {
            "manifest_duplicate_ids": manifest_runs.repeated_value_samples,
            "duplicate_member_ids": member_runs.repeated_value_samples,
            "missing_members": expected_vs_members.left_only_samples,
            "extra_members": expected_vs_members.right_only_samples,
            "representatives_not_in_manifest": (
                expected_vs_representatives.right_only_samples
            ),
            "representatives_not_observed_as_members": (
                members_vs_representatives.right_only_samples
            ),
            "representatives_without_self_edge": (
                representatives_vs_self.left_only_samples
            ),
            "duplicate_self_edge_representatives": (
                self_runs.repeated_value_samples
            ),
            "maximum_cluster_representatives": representative_runs.max_run_values,
        }
        failed = not all(checks.values())
        report.update(
            {
                "status": "failed" if failed else "passed",
                "completed_at_utc": _utc_now(),
                "elapsed_seconds": round(time.monotonic() - started_monotonic, 6),
                "inputs": {
                    "manifest_paths_tsv": str(manifest),
                    "manifest_paths_tsv_size_bytes": manifest.stat().st_size,
                    "manifest_paths_tsv_sha256": manifest_sha256,
                    "clusters_tsv": str(clusters),
                    "clusters_tsv_size_bytes": clusters.stat().st_size,
                    "clusters_tsv_sha256": clusters_sha256,
                },
                "counts": counts,
                "violations": violations,
                "checks": checks,
                "samples": samples,
            }
        )
        _atomic_write_json(report_path, report)
        return report
    except FoldseekAuditError as exc:
        failed = True
        report.update(
            {
                "status": "error",
                "completed_at_utc": _utc_now(),
                "elapsed_seconds": round(time.monotonic() - started_monotonic, 6),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        _atomic_write_json(report_path, report)
        raise
    finally:
        if run_tmp is not None and run_tmp.exists():
            if failed and keep_tmp_on_failure:
                print(f"[audit] retained scratch directory: {run_tmp}", file=sys.stderr)
                report.setdefault("scratch", {})["retained_path"] = str(run_tmp)  # type: ignore[index]
                if report.get("status") in {"failed", "error"}:
                    _atomic_write_json(report_path, report)
            else:
                shutil.rmtree(run_tmp)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a Foldseek representative-member TSV using streaming I/O "
            "and external sort."
        )
    )
    parser.add_argument(
        "--manifest-paths-tsv",
        "--paths-tsv",
        dest="manifest_paths_tsv",
        type=Path,
        required=True,
        help="One relative structure path per line; no header.",
    )
    parser.add_argument(
        "--clusters-tsv",
        type=Path,
        required=True,
        help="Foldseek createtsv output: representative TAB member.",
    )
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        required=True,
        help="Existing scratch parent below /mnt or /dev/shm, not on rootfs.",
    )
    parser.add_argument("--sort-parallel", type=int, default=8)
    parser.add_argument("--sort-memory", default="4G")
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument("--min-free-gb", type=float, default=4.0)
    parser.add_argument(
        "--keep-tmp-on-failure",
        action="store_true",
        help="Retain the unique audit scratch directory when validation fails.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = audit_foldseek_clusters(
            manifest_paths_tsv=args.manifest_paths_tsv,
            clusters_tsv=args.clusters_tsv,
            report_json=args.report_json,
            tmp_dir=args.tmp_dir,
            sort_parallel=args.sort_parallel,
            sort_memory=args.sort_memory,
            sample_limit=args.sample_limit,
            progress_every=args.progress_every,
            min_free_gb=args.min_free_gb,
            keep_tmp_on_failure=args.keep_tmp_on_failure,
        )
    except FoldseekAuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    counts = report["counts"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "manifest_rows": counts["manifest_rows"],
                "cluster_rows": counts["cluster_rows"],
                "cluster_count": counts["cluster_count"],
                "maximum_cluster_size": counts["maximum_cluster_size"],
                "report_json": str(args.report_json),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
