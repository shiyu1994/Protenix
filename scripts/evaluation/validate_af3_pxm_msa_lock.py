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
"""Recompute and fail-closed validate every PXM evaluation MSA hash."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.evaluation.build_af3_pxm_msa_lock import (
    DEFAULT_OFFICIAL_ROOT,
    DEFAULT_PXM_INPUT_ROOT,
    DEFAULT_PXM_MANIFEST_ROOT,
    MSA_LOCK_MANIFEST_FILENAME,
    MSA_LOCK_POLICY_VERSION,
    MSA_LOCK_SCHEMA_VERSION,
    MSA_LOCK_SUCCESS_FILENAME,
    build_lock_manifest,
    file_sha256,
    read_pdb_ids,
)


def _load_json_object(path: str | Path, *, field_name: str) -> Mapping[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{field_name} must be a JSON object: {path}")
    return payload


def validate_msa_lock(
    *,
    pdb_list: str | Path,
    bioassembly_dir: str | Path,
    protein_sequence_sources_path: str | Path,
    rna_sequence_mapping_path: str | Path,
    rna_msa_root: str | Path,
    manifest_path: str | Path,
    success_path: str | Path,
    requested_pdb_list: str | Path | None = None,
) -> dict[str, Any]:
    """Validate metadata, exact coverage, queries, sizes, and every file hash."""

    manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    success_path = Path(success_path).expanduser().resolve(strict=True)
    success = _load_json_object(success_path, field_name="MSA success marker")
    manifest = _load_json_object(manifest_path, field_name="MSA lock manifest")

    fixed_expectations = {
        "schema_version": MSA_LOCK_SCHEMA_VERSION,
        "policy_version": MSA_LOCK_POLICY_VERSION,
    }
    for field_name, expected in fixed_expectations.items():
        if success.get(field_name) != expected:
            raise ValueError(
                f"_MSA_SUCCESS {field_name} mismatch: "
                f"{success.get(field_name)!r} != {expected!r}"
            )
        if manifest.get(field_name) != expected:
            raise ValueError(
                f"MSA manifest {field_name} mismatch: "
                f"{manifest.get(field_name)!r} != {expected!r}"
            )
    if success.get("status") != "complete":
        raise ValueError("_MSA_SUCCESS status must be 'complete'")
    if (
        Path(str(success.get("manifest_path", ""))).resolve(strict=False)
        != manifest_path
    ):
        raise ValueError("_MSA_SUCCESS manifest_path mismatch")
    if success.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("_MSA_SUCCESS manifest_sha256 mismatch")

    # This is intentionally a full rebuild, not a trust in stored metadata:
    # every A3M is parsed and hashed again, and target/polymer coverage is
    # recomputed from the requested PDB and bioassembly inputs.
    recomputed = build_lock_manifest(
        pdb_list=pdb_list,
        bioassembly_dir=bioassembly_dir,
        protein_sequence_sources_path=protein_sequence_sources_path,
        rna_sequence_mapping_path=rna_sequence_mapping_path,
        rna_msa_root=rna_msa_root,
    )
    if manifest != recomputed:
        stored_counts = manifest.get("counts")
        recomputed_counts = recomputed.get("counts")
        if stored_counts != recomputed_counts:
            raise ValueError(
                "MSA lock target or unique-sequence coverage mismatch: "
                f"stored={stored_counts!r}, recomputed={recomputed_counts!r}"
            )
        raise ValueError(
            "MSA lock manifest differs from fully recomputed paths, sizes, "
            "queries, or SHA256 hashes"
        )
    if success.get("inputs") != recomputed["inputs"]:
        raise ValueError("_MSA_SUCCESS input fingerprints mismatch")
    if success.get("counts") != recomputed["counts"]:
        raise ValueError("_MSA_SUCCESS coverage counts mismatch")

    requested_target_count = recomputed["counts"]["target_count"]
    if requested_pdb_list is not None:
        requested_target_ids = read_pdb_ids(requested_pdb_list)
        locked_target_ids = set(recomputed["target_ids"])
        unlocked_target_ids = sorted(
            set(requested_target_ids).difference(locked_target_ids)
        )
        if unlocked_target_ids:
            raise ValueError(
                "requested PDB list contains targets outside the full MSA "
                f"lock: count={len(unlocked_target_ids)}, "
                f"examples={unlocked_target_ids[:10]!r}"
            )
        requested_target_count = len(requested_target_ids)

    return {
        "status": "valid",
        "schema_version": MSA_LOCK_SCHEMA_VERSION,
        "policy_version": MSA_LOCK_POLICY_VERSION,
        "manifest_path": str(manifest_path),
        "success_path": str(success_path),
        "manifest_sha256": file_sha256(manifest_path),
        "requested_target_count": requested_target_count,
        **recomputed["counts"],
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fully validate the locked PXM evaluation MSA collection."
    )
    parser.add_argument(
        "--pdb-list",
        default=str(DEFAULT_PXM_MANIFEST_ROOT / "combined_le1024.txt"),
    )
    parser.add_argument(
        "--bioassembly-dir",
        default=str(DEFAULT_OFFICIAL_ROOT / "recentPDB_bioassembly"),
    )
    parser.add_argument(
        "--protein-sequence-sources",
        default=str(
            DEFAULT_PXM_INPUT_ROOT
            / "manifests/protein_sequence_sources.json"
        ),
    )
    parser.add_argument(
        "--rna-sequence-mapping",
        default=str(
            DEFAULT_OFFICIAL_ROOT
            / "rna_msa/rna_sequence_to_pdb_chains.json"
        ),
    )
    parser.add_argument(
        "--rna-msa-root",
        default=str(DEFAULT_OFFICIAL_ROOT / "rna_msa/msas"),
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_PXM_INPUT_ROOT / "msa_lock"),
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--success", default=None)
    parser.add_argument(
        "--requested-pdb-list",
        default=None,
        help=(
            "Optional actual evaluation target list. The lock is always "
            "recomputed from --pdb-list (the full combined manifest); every "
            "requested target must be a member of that full lock."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    output_root = Path(args.output_root)
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else output_root / MSA_LOCK_MANIFEST_FILENAME
    )
    success_path = (
        Path(args.success)
        if args.success
        else output_root / MSA_LOCK_SUCCESS_FILENAME
    )
    summary = validate_msa_lock(
        pdb_list=args.pdb_list,
        bioassembly_dir=args.bioassembly_dir,
        protein_sequence_sources_path=args.protein_sequence_sources,
        rna_sequence_mapping_path=args.rna_sequence_mapping,
        rna_msa_root=args.rna_msa_root,
        manifest_path=manifest_path,
        success_path=success_path,
        requested_pdb_list=args.requested_pdb_list,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
