#!/usr/bin/env python3
"""Build a deterministic exact-crop benchmark whose real MSA depth is sufficient.

The source rows are filtered and ordered with the same ``top_tokens`` logic as
``build_long_protein_benchmark.py``.  PDB- and cluster-uniqueness is applied to
that ordered candidate stream first.  Candidates are then audited in order
until ``max_rows`` assemblies pass this stricter condition:

* every protein entity in the complete biological assembly has an effective
  depth of at least ``target_depth``;
* effective depth combines ``pairing.a3m`` and ``non_pairing.a3m`` as unpaired
  rows and deduplicates A3M rows after removing lower-case insertions.

The result contains the selected index, complete per-entity MSA evidence, and
an audit trace that also records rejected candidates.  Validation is artifact
based and does not rescan the source MSA database.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

try:
    from scripts.benchmark.audit_msa_depth import (
        audit_index_row,
        sha256_file,
        summarize_records,
        write_audit_outputs,
    )
    from scripts.benchmark.build_long_protein_benchmark import (
        DEFAULT_PROTENIX_ROOT,
        DEFAULT_SOURCE_INDEX,
        filter_long_protein_context,
        select_benchmark_rows,
        summarize_selection,
    )
except ModuleNotFoundError:  # Direct ``python scripts/benchmark/...py`` execution.
    from audit_msa_depth import (  # type: ignore[no-redef]
        audit_index_row,
        sha256_file,
        summarize_records,
        write_audit_outputs,
    )
    from build_long_protein_benchmark import (  # type: ignore[no-redef]
        DEFAULT_PROTENIX_ROOT,
        DEFAULT_SOURCE_INDEX,
        filter_long_protein_context,
        select_benchmark_rows,
        summarize_selection,
    )


DEFAULT_OUTPUT_DIR = Path("data/benchmark_long_protein_context_1024_800_msa2048")
DEFAULT_MAPPING = DEFAULT_PROTENIX_ROOT / "common/seq_to_pdb_index.json"
DEFAULT_BIOASSEMBLY_DIR = DEFAULT_PROTENIX_ROOT / "mmcif_bioassembly"
DEFAULT_MSA_ROOT = DEFAULT_PROTENIX_ROOT / "mmcif_msa_template"
AUDIT_SEMANTICS = "pairing_and_non_pairing_as_unpaired_deduplicated"
TRACE_FIELDS = (
    "candidate_rank",
    "accepted_row_index",
    "status",
    "pdb_id",
    "cluster_id",
    "num_tokens",
    "protein_entity_count",
    "min_effective_depth",
    "max_effective_depth",
    "missing_mapping_entity_count",
    "missing_a3m_entity_count",
    "invalid_a3m_entity_count",
)
READY_MARKER_FIELDS = (
    "schema_version",
    "source_index",
    "source_index_sha256",
    "mapping",
    "mapping_sha256",
    "output_index",
    "index_sha256",
    "msa_depth_json_sha256",
    "msa_depth_csv_sha256",
    "msa_depth_candidates_csv_sha256",
    "audit_semantics",
    "target_depth",
    "max_msa_size",
    "max_rows",
    "filter_min_num_tokens",
    "filter_max_num_tokens",
    "selection",
    "seed",
    "rows",
)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv_atomic(path: Path, fieldnames: Sequence[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_ready_marker(path: Path, summary: Mapping[str, Any]) -> None:
    """Publish a synchronization marker only after every artifact is durable."""
    payload = {field: summary[field] for field in READY_MARKER_FIELDS}
    _write_text_atomic(
        path.resolve(), json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict):
        raise TypeError(f"MSA mapping must be a JSON object: {path}")
    return mapping


def _candidate_stream(
    source: pd.DataFrame,
    *,
    min_num_tokens: int,
    max_num_tokens: int | None,
    selection: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = filter_long_protein_context(
        source,
        min_num_tokens=min_num_tokens,
        max_num_tokens=max_num_tokens,
    )
    # Requesting all rows preserves the existing sort -> cluster dedup -> PDB
    # dedup semantics while producing the full ordered stream for depth audit.
    ordered_unique = select_benchmark_rows(
        filtered,
        max_rows=len(filtered),
        selection=selection,
        seed=seed,
    )
    return filtered, ordered_unique


def _trace_row(
    *,
    candidate_rank: int,
    row: Mapping[str, Any],
    status: str,
    record: Mapping[str, Any] | None,
    accepted_row_index: int | None,
) -> dict[str, Any]:
    result = {
        "candidate_rank": candidate_rank,
        "accepted_row_index": (
            accepted_row_index if accepted_row_index is not None else ""
        ),
        "status": status,
        "pdb_id": str(row["pdb_id"]).strip().lower(),
        "cluster_id": str(row["cluster_id"]),
        "num_tokens": int(row["num_tokens"]),
        "protein_entity_count": 0,
        "min_effective_depth": 0,
        "max_effective_depth": 0,
        "missing_mapping_entity_count": 0,
        "missing_a3m_entity_count": 0,
        "invalid_a3m_entity_count": 0,
    }
    if record is not None:
        for field in (
            "protein_entity_count",
            "min_effective_depth",
            "max_effective_depth",
            "missing_mapping_entity_count",
            "missing_a3m_entity_count",
            "invalid_a3m_entity_count",
        ):
            result[field] = int(record[field])
    return result


def select_msa_qualified_rows(
    *,
    candidates: pd.DataFrame,
    sequence_to_msa_dir: Mapping[str, Any],
    bioassembly_dir: Path,
    msa_root: Path,
    max_rows: int,
    target_depth: int,
    max_msa_size: int,
    progress_every: int = 25,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit an ordered unique stream and return exactly ``max_rows`` passes."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if target_depth <= 0:
        raise ValueError("target_depth must be positive")
    if max_msa_size < target_depth:
        raise ValueError("max_msa_size must be >= target_depth")

    selected_rows: list[pd.Series] = []
    accepted_records: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for candidate_rank, (_, row) in enumerate(candidates.iterrows()):
        try:
            record = audit_index_row(
                row_index=len(selected_rows),
                row=row,
                bioassembly_dir=bioassembly_dir,
                sequence_to_msa_dir=sequence_to_msa_dir,
                msa_root=msa_root,
                target_depth=target_depth,
                max_msa_size=max_msa_size,
            )
        except FileNotFoundError:
            trace.append(
                _trace_row(
                    candidate_rank=candidate_rank,
                    row=row,
                    status="rejected_missing_bioassembly",
                    record=None,
                    accepted_row_index=None,
                )
            )
            continue

        if not record["protein_entity_count"]:
            status = "rejected_no_protein_entity"
        elif not record["all_entities_reach_target"]:
            status = "rejected_msa_depth"
        else:
            status = "accepted"

        accepted_index = len(selected_rows) if status == "accepted" else None
        trace.append(
            _trace_row(
                candidate_rank=candidate_rank,
                row=row,
                status=status,
                record=record,
                accepted_row_index=accepted_index,
            )
        )
        if status != "accepted":
            continue

        selected_rows.append(row.copy())
        accepted_records.append(record)
        if progress_every > 0 and len(selected_rows) % progress_every == 0:
            print(
                f"MSA-qualified rows: {len(selected_rows)}/{max_rows}; "
                f"audited candidates: {len(trace)}",
                flush=True,
            )
        if len(selected_rows) == max_rows:
            break

    if len(selected_rows) != max_rows:
        raise ValueError(
            f"only {len(selected_rows)} of {max_rows} required rows passed "
            f"MSA depth {target_depth} after auditing {len(trace)} candidates"
        )

    selected = pd.DataFrame(selected_rows, columns=candidates.columns).reset_index(
        drop=True
    )
    # The candidate stream is already unique.  Keep an explicit invariant at
    # the final boundary because benchmark replacement=False depends on it.
    if not selected["pdb_id"].is_unique or not selected["cluster_id"].is_unique:
        raise AssertionError("MSA-qualified index lost PDB/cluster uniqueness")
    return selected, accepted_records, trace


def _build_audit_report(
    *,
    index_path: Path,
    mapping_path: Path,
    bioassembly_dir: Path,
    msa_root: Path,
    target_depth: int,
    max_msa_size: int,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "audit_semantics": AUDIT_SEMANTICS,
        "interpretation": (
            "All assembly protein entities meet target source capacity; "
            "stochastic-crop runtime N_msa remains authoritative."
        ),
        "inputs": {
            "index": str(index_path),
            "index_sha256": sha256_file(index_path),
            "mapping": str(mapping_path),
            "bioassembly_dir": str(bioassembly_dir),
            "msa_root": str(msa_root),
            "limit": None,
            "target_depth": target_depth,
            "max_msa_size": max_msa_size,
        },
        "summary": summarize_records(records, target_depth),
        "records": records,
    }


def build_msa_aware_benchmark_index(
    *,
    source_index: Path,
    output_dir: Path,
    mapping_path: Path,
    bioassembly_dir: Path,
    msa_root: Path,
    max_rows: int = 800,
    min_num_tokens: int = 1025,
    max_num_tokens: int | None = 1536,
    selection: str = "top_tokens",
    seed: int = 42,
    target_depth: int = 2048,
    max_msa_size: int = 16384,
    progress_every: int = 25,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build and atomically publish the index plus complete MSA evidence."""
    source_index = source_index.resolve()
    output_dir = output_dir.resolve()
    mapping_path = mapping_path.resolve()
    bioassembly_dir = bioassembly_dir.resolve()
    msa_root = msa_root.resolve()

    source = pd.read_csv(source_index)
    filtered, candidates = _candidate_stream(
        source,
        min_num_tokens=min_num_tokens,
        max_num_tokens=max_num_tokens,
        selection=selection,
        seed=seed,
    )
    sequence_to_msa_dir = _load_mapping(mapping_path)
    selected, records, trace = select_msa_qualified_rows(
        candidates=candidates,
        sequence_to_msa_dir=sequence_to_msa_dir,
        bioassembly_dir=bioassembly_dir,
        msa_root=msa_root,
        max_rows=max_rows,
        target_depth=target_depth,
        max_msa_size=max_msa_size,
        progress_every=progress_every,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "indices.csv"
    audit_json_path = output_dir / "msa_depth.json"
    audit_csv_path = output_dir / "msa_depth.csv"
    trace_path = output_dir / "msa_depth_candidates.csv"
    summary_path = output_dir / "summary.json"

    temporary_index = output_dir / f".indices.csv.{os.getpid()}.tmp"
    selected.to_csv(temporary_index, index=False)
    os.replace(temporary_index, index_path)
    audit_report = _build_audit_report(
        index_path=index_path,
        mapping_path=mapping_path,
        bioassembly_dir=bioassembly_dir,
        msa_root=msa_root,
        target_depth=target_depth,
        max_msa_size=max_msa_size,
        records=records,
    )
    write_audit_outputs(audit_report, audit_json_path, audit_csv_path)
    _write_csv_atomic(trace_path, TRACE_FIELDS, trace)

    statuses = pd.Series([entry["status"] for entry in trace]).value_counts()
    entity_depths = [
        int(entity["effective_depth"])
        for record in records
        for entity in record["entities"]
    ]
    summary = {
        "schema_version": 1,
        "source_index": str(source_index),
        "source_index_sha256": sha256_file(source_index),
        "mapping": str(mapping_path),
        "mapping_sha256": sha256_file(mapping_path),
        "bioassembly_dir": str(bioassembly_dir),
        "msa_root": str(msa_root),
        "output_index": str(index_path),
        "index_sha256": sha256_file(index_path),
        "msa_depth_json": str(audit_json_path),
        "msa_depth_json_sha256": sha256_file(audit_json_path),
        "msa_depth_csv": str(audit_csv_path),
        "msa_depth_csv_sha256": sha256_file(audit_csv_path),
        "msa_depth_candidates_csv": str(trace_path),
        "msa_depth_candidates_csv_sha256": sha256_file(trace_path),
        "audit_semantics": AUDIT_SEMANTICS,
        "target_depth": int(target_depth),
        "max_msa_size": int(max_msa_size),
        "max_rows": int(max_rows),
        "filter_min_num_tokens": int(min_num_tokens),
        "filter_max_num_tokens": (
            int(max_num_tokens) if max_num_tokens is not None else None
        ),
        "selection": selection,
        "seed": int(seed),
        "candidate_rows": int(len(filtered)),
        "unique_candidate_rows": int(len(candidates)),
        "audited_candidate_rows": int(len(trace)),
        "candidate_status_counts": {
            str(key): int(value) for key, value in statuses.items()
        },
        "accepted_protein_entities": int(len(entity_depths)),
        "accepted_min_effective_depth": min(entity_depths),
        "accepted_max_effective_depth": max(entity_depths),
        "accepted_mean_effective_depth": float(sum(entity_depths) / len(entity_depths)),
        **summarize_selection(selected),
    }
    _write_text_atomic(
        summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return selected, summary


def _require_summary_setting(
    summary: Mapping[str, Any], key: str, expected: Any
) -> None:
    if summary.get(key) != expected:
        raise ValueError(
            f"benchmark summary field {key} is stale: "
            f"{summary.get(key)!r} != {expected!r}"
        )


def validate_msa_aware_benchmark_index(
    *,
    source_index: Path,
    output_dir: Path,
    mapping_path: Path,
    bioassembly_dir: Path,
    msa_root: Path,
    max_rows: int = 800,
    min_num_tokens: int = 1025,
    max_num_tokens: int | None = 1536,
    selection: str = "top_tokens",
    seed: int = 42,
    target_depth: int = 2048,
    max_msa_size: int = 16384,
    ready_marker: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate locked artifacts without rereading all source A3M files."""
    source_index = source_index.resolve()
    output_dir = output_dir.resolve()
    mapping_path = mapping_path.resolve()
    bioassembly_dir = bioassembly_dir.resolve()
    msa_root = msa_root.resolve()
    paths = {
        "index": output_dir / "indices.csv",
        "summary": output_dir / "summary.json",
        "audit_json": output_dir / "msa_depth.json",
        "audit_csv": output_dir / "msa_depth.csv",
        "trace": output_dir / "msa_depth_candidates.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing MSA benchmark artifacts: {missing}")

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    expected_settings = {
        "schema_version": 1,
        "source_index": str(source_index),
        "mapping": str(mapping_path),
        "bioassembly_dir": str(bioassembly_dir),
        "msa_root": str(msa_root),
        "output_index": str(paths["index"]),
        "audit_semantics": AUDIT_SEMANTICS,
        "target_depth": int(target_depth),
        "max_msa_size": int(max_msa_size),
        "max_rows": int(max_rows),
        "filter_min_num_tokens": int(min_num_tokens),
        "filter_max_num_tokens": (
            int(max_num_tokens) if max_num_tokens is not None else None
        ),
        "selection": selection,
        "seed": int(seed),
        "rows": int(max_rows),
        "pdbs": int(max_rows),
        "clusters": int(max_rows),
    }
    for key, expected in expected_settings.items():
        _require_summary_setting(summary, key, expected)

    hash_fields = {
        "index_sha256": paths["index"],
        "msa_depth_json_sha256": paths["audit_json"],
        "msa_depth_csv_sha256": paths["audit_csv"],
        "msa_depth_candidates_csv_sha256": paths["trace"],
    }
    for field, path in hash_fields.items():
        _require_summary_setting(summary, field, sha256_file(path))

    selected = pd.read_csv(paths["index"])
    if len(selected) != max_rows:
        raise ValueError(f"expected {max_rows} index rows, got {len(selected)}")
    if not selected["pdb_id"].is_unique or not selected["cluster_id"].is_unique:
        raise ValueError("MSA-aware benchmark must be PDB- and cluster-unique")
    filtered = filter_long_protein_context(
        selected,
        min_num_tokens=min_num_tokens,
        max_num_tokens=max_num_tokens,
    )
    if len(filtered) != len(selected):
        raise ValueError("index contains rows outside the protein/token filter")

    audit = json.loads(paths["audit_json"].read_text(encoding="utf-8"))
    if audit.get("audit_semantics") != AUDIT_SEMANTICS:
        raise ValueError("MSA audit semantics mismatch")
    audit_inputs = audit.get("inputs", {})
    if audit_inputs.get("index_sha256") != summary["index_sha256"]:
        raise ValueError("MSA audit refers to a different index SHA256")
    if audit_inputs.get("target_depth") != target_depth:
        raise ValueError("MSA audit target depth mismatch")
    records = audit.get("records", [])
    if len(records) != max_rows:
        raise ValueError(f"expected {max_rows} MSA records, got {len(records)}")
    if [record["pdb_id"] for record in records] != [
        str(value).lower() for value in selected["pdb_id"]
    ]:
        raise ValueError("MSA audit PDB order differs from indices.csv")
    for record in records:
        if not record.get("protein_entity_count"):
            raise ValueError(f"{record.get('pdb_id')} has no protein entity")
        if not record.get("all_entities_reach_target"):
            raise ValueError(
                f"{record.get('pdb_id')} has an assembly protein below target depth"
            )
        if any(
            int(entity.get("effective_depth", 0)) < target_depth
            for entity in record.get("entities", [])
        ):
            raise ValueError(f"{record.get('pdb_id')} entity depth evidence is stale")

    with paths["audit_csv"].open("r", encoding="utf-8", newline="") as handle:
        audit_csv_rows = list(csv.DictReader(handle))
    if len(audit_csv_rows) != max_rows:
        raise ValueError("msa_depth.csv row count mismatch")
    with paths["trace"].open("r", encoding="utf-8", newline="") as handle:
        trace_rows = list(csv.DictReader(handle))
    if len(trace_rows) != summary.get("audited_candidate_rows"):
        raise ValueError("MSA candidate trace row count mismatch")
    if sum(row["status"] == "accepted" for row in trace_rows) != max_rows:
        raise ValueError("MSA candidate trace accepted count mismatch")
    if ready_marker is not None:
        ready_marker = ready_marker.resolve()
        if not ready_marker.is_file():
            raise FileNotFoundError(f"missing benchmark ready marker: {ready_marker}")
        marker = json.loads(ready_marker.read_text(encoding="utf-8"))
        for field in READY_MARKER_FIELDS:
            if marker.get(field) != summary.get(field):
                raise ValueError(f"benchmark ready marker field {field} is stale")
    return selected, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--bioassembly-dir", type=Path, default=DEFAULT_BIOASSEMBLY_DIR)
    parser.add_argument("--msa-root", type=Path, default=DEFAULT_MSA_ROOT)
    parser.add_argument("--max-rows", type=int, default=800)
    parser.add_argument("--min-num-tokens", type=int, default=1025)
    parser.add_argument("--max-num-tokens", type=int, default=1536)
    parser.add_argument(
        "--selection", choices=("top_tokens", "random"), default="top_tokens"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-depth", type=int, default=2048)
    parser.add_argument("--max-msa-size", type=int, default=16384)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--ready-marker", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-mapping-sha256")
    parser.add_argument("--expected-index-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    common = {
        "source_index": args.source_index,
        "output_dir": args.output_dir,
        "mapping_path": args.mapping,
        "bioassembly_dir": args.bioassembly_dir,
        "msa_root": args.msa_root,
        "max_rows": args.max_rows,
        "min_num_tokens": args.min_num_tokens,
        "max_num_tokens": args.max_num_tokens,
        "selection": args.selection,
        "seed": args.seed,
        "target_depth": args.target_depth,
        "max_msa_size": args.max_msa_size,
    }
    if args.validate_only:
        # A locked dataset may predate this particular benchmark run.  Validate
        # all durable artifacts first, then publish the run-local marker rather
        # than rescanning source A3Ms merely to create a synchronization file.
        marker_to_validate = (
            args.ready_marker
            if args.ready_marker is not None and args.ready_marker.is_file()
            else None
        )
        selected, summary = validate_msa_aware_benchmark_index(
            **common, ready_marker=marker_to_validate
        )
        if args.ready_marker is not None and marker_to_validate is None:
            write_ready_marker(args.ready_marker, summary)
    else:
        selected, summary = build_msa_aware_benchmark_index(
            **common, progress_every=args.progress_every
        )
        if args.ready_marker is not None:
            write_ready_marker(args.ready_marker, summary)
    expected_hashes = {
        "source_index_sha256": args.expected_source_sha256,
        "mapping_sha256": args.expected_mapping_sha256,
        "index_sha256": args.expected_index_sha256,
    }
    for field, expected in expected_hashes.items():
        if expected is not None and summary.get(field) != expected:
            raise ValueError(
                f"{field} differs from locked benchmark: "
                f"{summary.get(field)} != {expected}"
            )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(
        selected[["pdb_id", "type", "num_tokens", "cluster_id"]]
        .head(20)
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
