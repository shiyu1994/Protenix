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

"""Fail-closed full validator for the locked PXM template cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from protenix.data.template.evaluation_template_cache import (
    EVALUATION_CACHE_MANIFEST_FILENAME,
    EVALUATION_CACHE_STATS_FILENAME,
    EVALUATION_CACHE_SUCCESS_FILENAME,
    file_sha256,
    load_sequence_sources,
    validate_cache_collection,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate complete sequence coverage and every content-addressed "
            "PXM evaluation template artifact."
        )
    )
    parser.add_argument("--sequence-sources", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--success", default=None)
    return parser.parse_args(argv)


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

    sequence_sources = load_sequence_sources(sequence_sources_path)
    if not success_path.is_file():
        raise FileNotFoundError(success_path)
    with open(success_path, "r", encoding="utf-8") as handle:
        success = json.load(handle)
    expected_input_hash = file_sha256(sequence_sources_path)
    if success.get("sequence_sources_file_sha256") != expected_input_hash:
        raise ValueError(
            "_SUCCESS sequence_sources_file_sha256 does not match the "
            "requested input manifest"
        )
    if int(success.get("failed_count", -1)) != 0:
        raise ValueError("_SUCCESS reports failed cache builds")
    if int(success.get("missing_count", -1)) != 0:
        raise ValueError("_SUCCESS reports missing cache artifacts")

    summary = validate_cache_collection(
        sequence_sources=sequence_sources,
        cache_root=cache_root,
        manifest_path=manifest_path,
        stats_path=stats_path,
        success_path=success_path,
    )
    summary["sequence_sources_file_sha256"] = expected_input_hash
    summary["cache_root"] = str(cache_root.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
