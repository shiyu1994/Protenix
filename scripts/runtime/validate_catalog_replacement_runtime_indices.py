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

"""Validate the atomically published residual/ge200 runtime-index pair.

The default validation is intentionally light: it reads metadata, opens the
runtime mmaps, probes cluster boundaries, and stats a bounded set of source
files.  ``--verify-checksums`` additionally hashes every generated catalog
artifact listed in ``checksums.sha256``; it never scans the original large
input TSVs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protenix.data.pipeline.distillation_runtime_index import (  # noqa: E402
    DistillationRuntimeIndex,
)


DEFAULT_RESIDUAL_RUNTIME = Path(
    "/mnt/project_abd_hs/klp/databases/"
    "afdb_mgnify_legacy_residual_not_ge200_runtime_index_20260804"
)
DEFAULT_GE200_RUNTIME = Path(
    "/mnt/project_abd_hs/klp/databases/"
    "all_sources_ge200_structural_runtime_index_20260804"
)


class CatalogValidationError(RuntimeError):
    """A published catalog invariant was violated."""


@dataclass(frozen=True)
class ExpectedCatalogCounts:
    legacy_input_rows: int
    overlap_rows: int
    residual_rows: int
    ge200_rows: int
    target_rows: int
    ge200_clusters: int
    residual_short_rows: int
    residual_long_rows: int
    ambiguity_records: int


PRODUCTION_COUNTS = ExpectedCatalogCounts(
    legacy_input_rows=33_498_651,
    overlap_rows=18_175_512,
    residual_rows=15_323_139,
    ge200_rows=25_417_250,
    target_rows=40_740_389,
    ge200_clusters=9_050_355,
    residual_short_rows=15_313_927,
    residual_long_rows=9_212,
    ambiguity_records=142,
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(document: object) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _read_stable(path: Path) -> bytes:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(payload) != after.st_size:
        raise CatalogValidationError(f"file changed while being read: {path}")
    return payload


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogValidationError(f"invalid JSON in {label}: {exc}") from exc
    if not isinstance(document, dict):
        raise CatalogValidationError(f"{label} must contain a JSON object")
    return document


def _validate_checksum_listing(root: Path, *, verify_payloads: bool) -> int:
    checksum_path = root / "checksums.sha256"
    payload = _read_stable(checksum_path).decode("utf-8")
    records: dict[str, str] = {}
    for line_number, line in enumerate(payload.splitlines(), start=1):
        digest, separator, relative_text = line.partition("  ")
        relative = Path(relative_text)
        if (
            separator != "  "
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or not relative_text
            or relative.is_absolute()
            or ".." in relative.parts
            or "." in relative.parts
        ):
            raise CatalogValidationError(
                f"invalid checksums.sha256 line {line_number} in {root}"
            )
        if relative_text in records:
            raise CatalogValidationError(
                f"duplicate checksum path {relative_text!r} in {root}"
            )
        records[relative_text] = digest

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    }
    if set(records) != actual_files:
        missing = sorted(actual_files - set(records))
        extra = sorted(set(records) - actual_files)
        raise CatalogValidationError(
            f"checksum listing differs in {root}: missing={missing} extra={extra}"
        )
    if verify_payloads:
        for relative_text, expected in records.items():
            path = (root / relative_text).resolve(strict=True)
            if not path.is_relative_to(root):
                raise CatalogValidationError(f"checksum path escapes {root}: {path}")
            observed = _sha256_file(path)
            if observed != expected:
                raise CatalogValidationError(
                    f"checksum mismatch for {path}: {observed} != {expected}"
                )
    return len(records)


def _validate_runtime(
    root: Path,
    *,
    role: str,
    expected_rows: int,
    expected_clusters: int | None,
    expected_root_ids: Sequence[str],
    verify_checksums: bool,
    check_source_paths: bool,
    require_all_roots_used: bool,
) -> tuple[dict[str, Any], bytes, bytes, dict[str, Any]]:
    root = root.expanduser().resolve(strict=True)
    required = {
        "manifest": root / "manifest.json",
        "success": root / "_SUCCESS.json",
        "pair": root / "target_catalog_pair.json",
        "audit": root / "build_audit.json",
        "checksums": root / "checksums.sha256",
    }
    for label, path in required.items():
        if not path.is_file():
            raise CatalogValidationError(f"{role} lacks {label}: {path}")

    manifest_payload = _read_stable(required["manifest"])
    success_payload = _read_stable(required["success"])
    pair_payload = _read_stable(required["pair"])
    audit_payload = _read_stable(required["audit"])
    manifest = _json_object(manifest_payload, label=f"{role} manifest")
    success = _json_object(success_payload, label=f"{role} success")
    target = manifest.get("target_catalog")
    if not isinstance(target, Mapping) or target.get("role") != role:
        raise CatalogValidationError(f"{role} target_catalog role mismatch")
    if int(manifest.get("row_count", -1)) != expected_rows:
        raise CatalogValidationError(
            f"{role} row_count differs: {manifest.get('row_count')} != {expected_rows}"
        )
    clusters = int(manifest.get("cluster_count", -1))
    if expected_clusters is not None and clusters != expected_clusters:
        raise CatalogValidationError(
            f"{role} cluster_count differs: {clusters} != {expected_clusters}"
        )
    if (
        int(success.get("schema_version", -1)) != 1
        or success.get("status") != "complete"
        or success.get("manifest_sha256") != _sha256_bytes(manifest_payload)
        or int(success.get("row_count", -1)) != expected_rows
        or int(success.get("cluster_count", -1)) != clusters
    ):
        raise CatalogValidationError(f"{role} _SUCCESS binding mismatch")
    receipt = target.get("pair_receipt")
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise CatalogValidationError(f"{role} pair_receipt record is invalid")
    if (
        receipt.get("path") != "target_catalog_pair.json"
        or int(receipt.get("size_bytes", -1)) != len(pair_payload)
        or receipt.get("sha256") != _sha256_bytes(pair_payload)
    ):
        raise CatalogValidationError(f"{role} pair_receipt binding mismatch")

    listed_files = _validate_checksum_listing(
        root, verify_payloads=verify_checksums
    )
    sampled: list[dict[str, Any]] = []
    unused_roots: list[str] = []
    with DistillationRuntimeIndex(root) as runtime:
        if runtime.row_count != expected_rows or runtime.cluster_count != clusters:
            raise CatalogValidationError(f"{role} runtime/manifest counts mismatch")
        if list(runtime.root_ids) != list(expected_root_ids):
            raise CatalogValidationError(
                f"{role} root_ids differ: {runtime.root_ids} != {expected_root_ids}"
            )
        if not runtime.has_explicit_paths or not runtime.has_global_ids:
            raise CatalogValidationError(
                f"{role} lacks row-aligned explicit paths or global IDs"
            )
        if runtime.get_member_row(0, 0) != 0:
            raise CatalogValidationError(f"{role} first cluster does not start at row 0")
        last_ordinal = runtime.cluster_count - 1
        last_size = runtime.get_cluster_size(last_ordinal)
        if runtime.get_member_row(last_ordinal, last_size - 1) != runtime.row_count - 1:
            raise CatalogValidationError(
                f"{role} final cluster does not end at the final row"
            )

        roots_doc = _json_object(
            _read_stable(runtime.roots_path), label=f"{role} roots"
        )
        prefixes = {
            str(record["root_id"]): str(record["source_prefix"])
            for record in roots_doc.get("roots", [])
        }
        sample_rows = {0, runtime.row_count // 2, runtime.row_count - 1}
        root_ordinals = runtime._map("source_root_ids")
        for ordinal, root_id in enumerate(runtime.root_ids):
            row = root_ordinals.find(bytes((ordinal,)))
            if row < 0:
                unused_roots.append(root_id)
            else:
                sample_rows.add(row)
        if require_all_roots_used and unused_roots:
            raise CatalogValidationError(
                f"{role} declared roots have no rows: {unused_roots}"
            )
        for row in sorted(sample_rows):
            paths = runtime.resolve_paths(row)
            prefix = prefixes.get(paths.root_id)
            if not prefix or not paths.global_id or not paths.global_id.startswith(prefix):
                raise CatalogValidationError(
                    f"{role} row {row} global-ID/root-prefix mismatch"
                )
            runtime.get_row_cluster(row)
            if check_source_paths:
                exists: dict[str, bool | None] = {
                    "cif": paths.cif_path.is_file(),
                    "a3m": paths.a3m_path.is_file(),
                    "pkl": paths.bioassembly_path.is_file(),
                }
                if not all(exists.values()):
                    raise CatalogValidationError(
                        f"{role} row {row} has missing source paths: {exists}"
                    )
            else:
                exists = {"cif": None, "a3m": None, "pkl": None}
            sampled.append(
                {
                    "row": row,
                    "root_id": paths.root_id,
                    "global_id": paths.global_id,
                    "cif_path": str(paths.cif_path),
                    "a3m_path": str(paths.a3m_path),
                    "bioassembly_path": str(paths.bioassembly_path),
                    "source_paths_exist": exists,
                }
            )

    return (
        {
            "runtime_dir": str(root),
            "row_count": expected_rows,
            "cluster_count": clusters,
            "root_ids": list(expected_root_ids),
            "unused_root_ids": unused_roots,
            "checksum_entries": listed_files,
            "checksum_payloads_verified": verify_checksums,
            "sampled_rows": sampled,
        },
        pair_payload,
        audit_payload,
        manifest,
    )


def validate_catalog_pair(
    *,
    residual_runtime_dir: Path,
    ge200_runtime_dir: Path,
    expected: ExpectedCatalogCounts = PRODUCTION_COUNTS,
    verify_checksums: bool = False,
    check_source_paths: bool = True,
    require_all_roots_used: bool = True,
) -> dict[str, Any]:
    residual, residual_pair, residual_audit, residual_manifest = _validate_runtime(
        residual_runtime_dir,
        role="legacy_residual",
        expected_rows=expected.residual_rows,
        expected_clusters=None,
        expected_root_ids=("0", "1", "2"),
        verify_checksums=verify_checksums,
        check_source_paths=check_source_paths,
        require_all_roots_used=require_all_roots_used,
    )
    ge200, ge200_pair, ge200_audit, ge200_manifest = _validate_runtime(
        ge200_runtime_dir,
        role="ge200_structural",
        expected_rows=expected.ge200_rows,
        expected_clusters=expected.ge200_clusters,
        expected_root_ids=("0", "1", "2", "3", "4", "5"),
        verify_checksums=verify_checksums,
        check_source_paths=check_source_paths,
        require_all_roots_used=require_all_roots_used,
    )
    if residual_pair != ge200_pair:
        raise CatalogValidationError("residual/ge200 pair receipts are not byte-identical")
    if residual_audit != ge200_audit:
        raise CatalogValidationError("residual/ge200 build audits are not byte-identical")

    pair = _json_object(residual_pair, label="target catalog pair")
    if pair.get("format") != "protenix.target_distillation_catalog_pair.v1":
        raise CatalogValidationError("target catalog pair format mismatch")
    pair_basis = {key: value for key, value in pair.items() if key != "pair_id"}
    if pair.get("pair_id") != _canonical_sha256(pair_basis):
        raise CatalogValidationError("target catalog pair_id is not canonical")
    expected_pair_counts = {
        "legacy_input_rows": expected.legacy_input_rows,
        "ge200_afq_mgn_overlap_rows": expected.overlap_rows,
        "residual_rows": expected.residual_rows,
        "ge200_rows": expected.ge200_rows,
        "target_rows": expected.target_rows,
    }
    if pair.get("counts") != expected_pair_counts:
        raise CatalogValidationError(
            f"target pair counts differ: {pair.get('counts')} != {expected_pair_counts}"
        )
    expected_target_counts = {
        "legacy_residual": {
            "row_count": expected.residual_rows,
            "cluster_count": int(residual_manifest["cluster_count"]),
            "target_union_row_count": expected.target_rows,
        },
        "ge200_structural": {
            "row_count": expected.ge200_rows,
            "cluster_count": expected.ge200_clusters,
            "afq_mgn_overlap_rows": expected.overlap_rows,
            "target_union_row_count": expected.target_rows,
        },
    }
    children = pair.get("children")
    if not isinstance(children, Mapping):
        raise CatalogValidationError("target pair has no children mapping")
    for role, manifest in (
        ("legacy_residual", residual_manifest),
        ("ge200_structural", ge200_manifest),
    ):
        target_catalog = manifest["target_catalog"]
        if (
            target_catalog.get("pair_id") != pair["pair_id"]
            or target_catalog.get("counts") != expected_target_counts[role]
        ):
            raise CatalogValidationError(
                f"target_catalog identity/counts differ for {role}"
            )
        child = children.get(role)
        if not isinstance(child, Mapping):
            raise CatalogValidationError(f"target pair lacks child {role}")
        if (
            int(child.get("row_count", -1)) != int(manifest["row_count"])
            or int(child.get("cluster_count", -1))
            != int(manifest["cluster_count"])
            or child.get("lookup_sha256")
            != manifest["artifacts"]["member_to_cluster_lookup"]["sha256"]
        ):
            raise CatalogValidationError(f"target pair child binding differs for {role}")
    if children["ge200_structural"].get("ambiguity_sha256") != ge200_manifest[
        "artifacts"
    ]["ambiguity_alternates"]["sha256"]:
        raise CatalogValidationError("ge200 ambiguity binding differs")
    if int(
        ge200_manifest["artifacts"]["ambiguity_alternates"].get("count", -1)
    ) != expected.ambiguity_records:
        raise CatalogValidationError("ge200 ambiguity record count differs")

    audit = _json_object(residual_audit, label="build audit")
    if (
        audit.get("status") != "complete"
        or audit.get("pair_id") != pair["pair_id"]
        or audit.get("counts") != expected_pair_counts
        or int(audit.get("ambiguity_alternate_records", -1))
        != expected.ambiguity_records
    ):
        raise CatalogValidationError("build audit identity/counts mismatch")
    invariants = audit.get("invariants")
    if not isinstance(invariants, Mapping) or not invariants or not all(
        value is True for value in invariants.values()
    ):
        raise CatalogValidationError("build audit invariants are not all true")
    residual_partitions = audit.get("source_partition_counts", {}).get("residual")
    if residual_partitions != {
        "long_gt200": expected.residual_long_rows,
        "short_le200": expected.residual_short_rows,
    }:
        raise CatalogValidationError(
            f"residual partition counts differ: {residual_partitions}"
        )

    return {
        "status": "complete",
        "pair_id": pair["pair_id"],
        "pair_receipt_sha256": _sha256_bytes(residual_pair),
        "counts": expected_pair_counts,
        "ambiguity_alternate_records": expected.ambiguity_records,
        "residual": residual,
        "ge200": ge200,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--residual-runtime-dir", type=Path, default=DEFAULT_RESIDUAL_RUNTIME
    )
    parser.add_argument("--ge200-runtime-dir", type=Path, default=DEFAULT_GE200_RUNTIME)
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Hash every generated runtime artifact (more I/O, no source TSV scan).",
    )
    parser.add_argument(
        "--skip-source-path-checks",
        action="store_true",
        help="Resolve sampled CIF/A3M/PKL paths but do not stat the source files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = validate_catalog_pair(
        residual_runtime_dir=args.residual_runtime_dir,
        ge200_runtime_dir=args.ge200_runtime_dir,
        verify_checksums=args.verify_checksums,
        check_source_paths=not args.skip_source_path_checks,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
