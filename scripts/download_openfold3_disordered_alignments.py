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

"""Download alignment representatives required by the OpenFold3 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_BASE_URL = (
    "https://openfold3-data.s3.amazonaws.com/"
    "pdb_training_set/alignment_arrays"
)
REQUIRED_PAYLOAD_FIELDS = frozenset(
    {"msa", "deletion_matrix", "metadata"}
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-json",
        type=Path,
        required=True,
        help="Official disordered_cache_with_templates.json.",
    )
    parser.add_argument(
        "--alignment-dir",
        type=Path,
        action="append",
        required=True,
        help=(
            "Existing alignment lookup directory. Repeat for supplemental "
            "directories; the last directory is also the download target."
        ),
    )
    parser.add_argument(
        "--representative-list",
        type=Path,
        default=None,
        help=(
            "Optional newline-delimited subset of representative IDs. Every "
            "entry must occur in the official cache. This is useful for "
            "downloading an exported missing set on another machine."
        ),
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Atomic completion report; defaults beside the cache.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inventory missing representatives without downloading.",
    )
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help=(
            "Load and validate every existing selected NPZ. Use with "
            "--dry-run after copying externally downloaded files."
        ),
    )
    parser.add_argument(
        "--export-missing-prefix",
        type=Path,
        default=None,
        help=(
            "Write the current missing set as <prefix>.ids.txt, "
            "<prefix>.urls.txt, <prefix>.tsv, and <prefix>.json. This is "
            "an offline inventory operation and does not issue HTTP requests."
        ),
    )
    return parser.parse_args()


def _load_representatives(cache_path: Path) -> tuple[str, ...]:
    with cache_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    structure_data = document.get("structure_data")
    if not isinstance(structure_data, dict) or not structure_data:
        raise ValueError(
            f"{cache_path} does not contain non-empty structure_data"
        )

    representatives = set()
    for pdb_id, entry in structure_data.items():
        chains = entry.get("chains") if isinstance(entry, dict) else None
        if not isinstance(chains, dict):
            raise ValueError(f"{cache_path}: invalid chains for {pdb_id}")
        for chain in chains.values():
            representative = chain.get("alignment_representative_id")
            if representative:
                representative = str(representative)
                if (
                    "/" in representative
                    or "\\" in representative
                    or "\0" in representative
                    or "_" not in representative
                ):
                    raise ValueError(
                        "Unsafe alignment representative ID: "
                        f"{representative!r}"
                    )
                representatives.add(representative)
    return tuple(sorted(representatives))


def _candidate_names(representative: str) -> Iterable[str]:
    pdb_id, label_asym_id = representative.rsplit("_", 1)
    for candidate in dict.fromkeys(
        (pdb_id, pdb_id.lower(), pdb_id.upper())
    ):
        yield f"{candidate}_{label_asym_id}.npz"


def _select_representatives(
    official_representatives: tuple[str, ...],
    representative_list: Path | None,
) -> tuple[str, ...]:
    if representative_list is None:
        return official_representatives
    list_path = representative_list.expanduser().resolve(strict=True)
    requested = [
        line.strip()
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not requested:
        raise ValueError(f"{list_path} contains no representative IDs")
    if len(requested) != len(set(requested)):
        raise ValueError(f"{list_path} contains duplicate representative IDs")
    official = set(official_representatives)
    unknown = sorted(set(requested) - official)
    if unknown:
        preview = ", ".join(unknown[:5])
        raise ValueError(
            f"{list_path} contains {len(unknown)} IDs absent from the "
            f"official cache: {preview}"
        )
    return tuple(requested)


def _find_existing(
    representative: str,
    directories: tuple[Path, ...],
) -> Path | None:
    for directory in directories:
        for name in _candidate_names(representative):
            path = directory / name
            if path.is_file():
                return path
    return None


def _validate_alignment(path: Path) -> None:
    with np.load(path, allow_pickle=True) as archive:
        if not archive.files:
            raise ValueError(f"{path} contains no alignment sources")
        query_rows: list[np.ndarray] = []
        for source in archive.files:
            value = archive[source]
            if value.shape != () or value.dtype != object:
                raise ValueError(
                    f"{path}:{source} must be a scalar object array"
                )
            payload = value.item()
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{source} must contain a dictionary")
            missing = REQUIRED_PAYLOAD_FIELDS - set(payload)
            if missing:
                raise ValueError(
                    f"{path}:{source} is missing {sorted(missing)}"
                )
            msa = np.asarray(payload["msa"])
            deletion_matrix = np.asarray(payload["deletion_matrix"])
            metadata = np.asarray(payload["metadata"])
            if (
                msa.ndim != 2
                or msa.shape[0] == 0
                or deletion_matrix.shape != msa.shape
                or metadata.shape != (msa.shape[0],)
            ):
                raise ValueError(
                    f"{path}:{source} has inconsistent alignment shapes"
                )
            query_rows.append(np.char.upper(msa[0].astype("<U1")))
        reference_query = query_rows[0]
        if any(
            query.shape != reference_query.shape
            or not np.array_equal(query, reference_query)
            for query in query_rows[1:]
        ):
            raise ValueError(
                f"{path} alignment sources have different query rows"
            )


def _download_one(
    representative: str,
    *,
    output_dir: Path,
    base_url: str,
    retries: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    filename = f"{representative}.npz"
    output_path = output_dir / filename
    url = (
        base_url.rstrip("/")
        + "/"
        + urllib.parse.quote(filename, safe="")
    )
    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{filename}.",
                suffix=".partial",
                dir=output_dir,
                delete=False,
            ) as output_handle:
                temporary_path = Path(output_handle.name)
                with urllib.request.urlopen(
                    url,
                    timeout=timeout_seconds,
                ) as response:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        output_handle.write(chunk)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            _validate_alignment(temporary_path)
            size = temporary_path.stat().st_size
            os.replace(temporary_path, output_path)
            return {
                "representative_id": representative,
                "path": str(output_path),
                "bytes": size,
                "attempts": attempt,
            }
        except (
            OSError,
            ValueError,
            TypeError,
            urllib.error.URLError,
            socket.timeout,
        ) as error:
            last_error = error
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(
        f"Failed to download {representative} after {retries} attempts: "
        f"{last_error}"
    )


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _export_missing_manifests(
    prefix: Path,
    *,
    missing: list[str],
    base_url: str,
) -> dict[str, Any]:
    prefix = prefix.expanduser().resolve()
    ids_path = Path(f"{prefix}.ids.txt")
    urls_path = Path(f"{prefix}.urls.txt")
    tsv_path = Path(f"{prefix}.tsv")
    json_path = Path(f"{prefix}.json")

    records = []
    for representative in missing:
        filename = f"{representative}.npz"
        url = (
            base_url.rstrip("/")
            + "/"
            + urllib.parse.quote(filename, safe="")
        )
        records.append((representative, filename, url))

    ids_content = "".join(f"{record[0]}\n" for record in records)
    urls_content = "".join(f"{record[2]}\n" for record in records)
    tsv_content = "representative_id\tfilename\turl\n" + "".join(
        "\t".join(record) + "\n" for record in records
    )
    _atomic_write_text(ids_path, ids_content)
    _atomic_write_text(urls_path, urls_content)
    _atomic_write_text(tsv_path, tsv_content)

    manifest = {
        "base_url": base_url,
        "missing_count": len(records),
        "ids_path": str(ids_path),
        "ids_sha256": hashlib.sha256(ids_content.encode()).hexdigest(),
        "urls_path": str(urls_path),
        "urls_sha256": hashlib.sha256(urls_content.encode()).hexdigest(),
        "tsv_path": str(tsv_path),
        "tsv_sha256": hashlib.sha256(tsv_content.encode()).hexdigest(),
    }
    _atomic_write_json(json_path, manifest)
    manifest["json_path"] = str(json_path)
    return manifest


def main() -> None:
    args = _parse_args()
    if args.workers <= 0 or args.retries <= 0:
        raise ValueError("--workers and --retries must be positive")
    if args.validate_existing and not args.dry_run:
        raise ValueError("--validate-existing must be used with --dry-run")
    cache_path = args.cache_json.expanduser().resolve(strict=True)
    directories = tuple(
        path.expanduser().resolve(strict=True)
        for path in args.alignment_dir
    )
    output_dir = directories[-1]
    representatives = _select_representatives(
        _load_representatives(cache_path),
        args.representative_list,
    )
    existing = {
        representative: _find_existing(representative, directories)
        for representative in representatives
    }
    validation_failures: list[dict[str, str]] = []
    validated_count = 0
    validation_started_at = time.time()
    if args.validate_existing:
        validation_items = [
            (representative, path)
            for representative, path in existing.items()
            if path is not None
        ]
        # NPZ decompression, object unpickling, and schema checks are
        # CPU-heavy and spend material time under the GIL.  Processes scale
        # this one-time full-corpus validation across the host's cores;
        # downloads below remain threads because they are network-bound.
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending = {
                executor.submit(_validate_alignment, path): (
                    representative,
                    path,
                )
                for representative, path in validation_items
            }
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    representative, path = pending.pop(future)
                    validated_count += 1
                    try:
                        future.result()
                    except Exception as error:
                        validation_failures.append(
                            {
                                "representative_id": representative,
                                "path": str(path),
                                "error": str(error),
                            }
                        )
                    if (
                        validated_count % 250 == 0
                        or not pending
                    ):
                        print(
                            "validated_existing="
                            f"{validated_count}/{len(validation_items)} "
                            f"failed={len(validation_failures)}",
                            flush=True,
                        )
    validation_elapsed_seconds = (
        time.time() - validation_started_at
        if args.validate_existing
        else 0.0
    )
    missing = [
        representative
        for representative, path in existing.items()
        if path is None
    ]
    print(
        "OpenFold3 alignment representatives: "
        f"required={len(representatives)} "
        f"existing={len(representatives) - len(missing)} "
        f"missing={len(missing)}",
        flush=True,
    )
    if args.export_missing_prefix is not None:
        export_manifest = _export_missing_manifests(
            args.export_missing_prefix,
            missing=missing,
            base_url=args.base_url,
        )
        print(
            "exported_missing_manifests="
            f"{export_manifest['json_path']} "
            f"count={export_manifest['missing_count']}",
            flush=True,
        )

    started_at = time.time()
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    if not args.dry_run and missing:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            pending = {
                executor.submit(
                    _download_one,
                    representative,
                    output_dir=output_dir,
                    base_url=args.base_url,
                    retries=args.retries,
                    timeout_seconds=args.timeout_seconds,
                ): representative
                for representative in missing
            }
            finished_count = 0
            while pending:
                done, _ = wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    representative = pending.pop(future)
                    finished_count += 1
                    try:
                        completed.append(future.result())
                    except BaseException as error:
                        failures.append(
                            {
                                "representative_id": representative,
                                "error": str(error),
                            }
                        )
                    if finished_count % 25 == 0 or not pending:
                        downloaded_bytes = sum(
                            item["bytes"] for item in completed
                        )
                        print(
                            f"finished={finished_count}/{len(missing)} "
                            f"success={len(completed)} "
                            f"failed={len(failures)} "
                            f"bytes={downloaded_bytes}",
                            flush=True,
                        )

    unresolved = [
        representative
        for representative in representatives
        if _find_existing(representative, directories) is None
    ]
    report_path = args.report_json or (
        cache_path.parent / "alignment_download_report.json"
    )
    report = {
        "cache_json": str(cache_path),
        "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "base_url": args.base_url,
        "required_representatives": len(representatives),
        "validate_existing": bool(args.validate_existing),
        "validated_existing": validated_count,
        "validation_workers": args.workers,
        "validation_elapsed_seconds": validation_elapsed_seconds,
        "initially_missing": len(missing),
        "downloaded": len(completed),
        "downloaded_bytes": sum(item["bytes"] for item in completed),
        "failures": sorted(
            failures,
            key=lambda item: item["representative_id"],
        ),
        "existing_validation_failures": sorted(
            validation_failures,
            key=lambda item: item["representative_id"],
        ),
        "unresolved_representatives": unresolved,
        "dry_run": bool(args.dry_run),
        "elapsed_seconds": time.time() - started_at,
    }
    _atomic_write_json(report_path, report)
    print(f"report={report_path}", flush=True)
    if unresolved and not args.dry_run:
        raise SystemExit(
            f"{len(unresolved)} required alignment representatives remain missing"
        )
    if validation_failures:
        raise SystemExit(
            f"{len(validation_failures)} existing alignment archives failed validation"
        )


if __name__ == "__main__":
    main()
