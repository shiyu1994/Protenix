#!/usr/bin/env python
"""Build a deterministic long-protein index for exact 768-token benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROTENIX_ROOT = Path(
    os.environ.get(
        "PROTENIX_ROOT_DIR",
        "/mnt/project_abd/public_databases/protenix.20260101",
    )
)
DEFAULT_SOURCE_INDEX = (
    DEFAULT_PROTENIX_ROOT
    / "indices"
    / "indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz"
)
DEFAULT_OUTPUT_DIR = Path("data/benchmark_long_protein_context_768")
REQUIRED_COLUMNS = {
    "pdb_id",
    "mol_1_type",
    "mol_2_type",
    "type",
    "num_tokens",
    "cluster_id",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _has_protein_context(df: pd.DataFrame) -> pd.Series:
    return df["mol_1_type"].eq("prot") | df["mol_2_type"].eq("prot")


def _has_ions(df: pd.DataFrame) -> pd.Series:
    return df["mol_1_type"].eq("ions") | df["mol_2_type"].eq("ions")


def filter_long_protein_context(
    indices: pd.DataFrame,
    min_num_tokens: int,
    max_num_tokens: int | None,
) -> pd.DataFrame:
    """Keep protein-containing rows that can supply the requested exact crop."""
    missing = sorted(REQUIRED_COLUMNS.difference(indices.columns))
    if missing:
        raise ValueError(f"source index is missing required columns: {missing}")

    num_tokens = indices["num_tokens"].astype(int)
    valid = _has_protein_context(indices) & (num_tokens >= min_num_tokens)
    valid &= ~_has_ions(indices)
    if max_num_tokens is not None:
        valid &= num_tokens <= max_num_tokens

    filtered = indices.loc[valid].copy()
    filtered["num_tokens"] = filtered["num_tokens"].astype(int)
    return filtered.reset_index(drop=True)


def select_benchmark_rows(
    candidates: pd.DataFrame,
    max_rows: int,
    selection: str,
    seed: int,
) -> pd.DataFrame:
    """Select a deterministic, cluster-unique subset for benchmark runs."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if selection not in {"top_tokens", "random"}:
        raise ValueError("selection must be one of: top_tokens, random")

    if selection == "top_tokens":
        sort_columns = ["num_tokens", "pdb_id"]
        ascending = [False, True]
        for optional_column in ("chain_1_id", "chain_2_id"):
            if optional_column in candidates.columns:
                sort_columns.append(optional_column)
                ascending.append(True)
        selected = candidates.sort_values(
            by=sort_columns,
            ascending=ascending,
            na_position="last",
        )
    else:
        selected = candidates.sample(frac=1.0, random_state=seed)

    selected = selected.drop_duplicates(subset=["cluster_id"], keep="first")
    selected = selected.drop_duplicates(subset=["pdb_id"], keep="first")
    return selected.head(max_rows).reset_index(drop=True)


def summarize_selection(selected: pd.DataFrame) -> dict[str, Any]:
    num_tokens = selected["num_tokens"].astype(int)
    return {
        "rows": int(len(selected)),
        "pdbs": int(selected["pdb_id"].nunique()),
        "clusters": int(selected["cluster_id"].nunique()),
        "min_num_tokens": int(num_tokens.min()) if len(selected) else 0,
        "max_num_tokens": int(num_tokens.max()) if len(selected) else 0,
        "mean_num_tokens": float(num_tokens.mean()) if len(selected) else 0.0,
        "type_counts": selected["type"].astype(str).value_counts().to_dict(),
        "mol_1_type_counts": (
            selected["mol_1_type"].astype(str).value_counts().to_dict()
        ),
        "mol_2_type_counts": (
            selected["mol_2_type"].astype(str).value_counts().to_dict()
        ),
    }


def build_benchmark_index(
    source_index: Path,
    output_dir: Path,
    max_rows: int,
    min_num_tokens: int,
    max_num_tokens: int | None,
    selection: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_index = source_index.resolve()
    output_dir = output_dir.resolve()
    indices = pd.read_csv(source_index)
    candidates = filter_long_protein_context(
        indices,
        min_num_tokens=min_num_tokens,
        max_num_tokens=max_num_tokens,
    )
    selected = select_benchmark_rows(
        candidates,
        max_rows=max_rows,
        selection=selection,
        seed=seed,
    )
    if selected.empty:
        raise ValueError("benchmark selection is empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "indices.csv"
    summary_path = output_dir / "summary.json"
    temporary_index = output_dir / f".indices.csv.{os.getpid()}.tmp"
    temporary_summary = output_dir / f".summary.json.{os.getpid()}.tmp"
    selected.to_csv(temporary_index, index=False)
    index_sha256 = sha256_file(temporary_index)

    summary = {
        "schema_version": 1,
        "source_index": str(source_index),
        "source_index_sha256": sha256_file(source_index),
        "output_index": str(index_path),
        "index_sha256": index_sha256,
        "max_rows": int(max_rows),
        "filter_min_num_tokens": int(min_num_tokens),
        "filter_max_num_tokens": (
            int(max_num_tokens) if max_num_tokens is not None else None
        ),
        "selection": selection,
        "seed": int(seed),
        "candidate_rows": int(len(candidates)),
        **summarize_selection(selected),
    }
    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_index, index_path)
    os.replace(temporary_summary, summary_path)
    return selected, summary


def write_ready_marker(path: Path, summary: dict[str, Any]) -> None:
    """Publish a per-run marker only after the index and summary are durable."""
    fields = (
        "schema_version",
        "source_index",
        "source_index_sha256",
        "output_index",
        "index_sha256",
        "max_rows",
        "filter_min_num_tokens",
        "filter_max_num_tokens",
        "selection",
        "seed",
        "rows",
    )
    _write_json_atomic(path.resolve(), {key: summary[key] for key in fields})


def validate_benchmark_index(
    *,
    source_index: Path,
    output_dir: Path,
    max_rows: int,
    min_num_tokens: int,
    max_num_tokens: int | None,
    selection: str,
    seed: int,
    ready_marker: Path | None = None,
    require_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reject stale or partially written benchmark index artifacts."""
    source_index = source_index.resolve()
    output_dir = output_dir.resolve()
    index_path = output_dir / "indices.csv"
    summary_path = output_dir / "summary.json"
    if not index_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"missing benchmark index artifacts in {output_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "source_index": str(source_index),
        "output_index": str(index_path),
        "max_rows": int(max_rows),
        "filter_min_num_tokens": int(min_num_tokens),
        "filter_max_num_tokens": (
            int(max_num_tokens) if max_num_tokens is not None else None
        ),
        "selection": selection,
        "seed": int(seed),
    }
    mismatches = {
        key: (summary.get(key), expected_value)
        for key, expected_value in expected.items()
        if summary.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"benchmark summary settings are stale: {mismatches}")

    actual_index_sha256 = sha256_file(index_path)
    if summary.get("index_sha256") != actual_index_sha256:
        raise ValueError("benchmark index SHA256 does not match summary.json")
    if not summary.get("source_index_sha256"):
        raise ValueError("benchmark summary is missing source_index_sha256")

    selected = pd.read_csv(index_path)
    filtered = filter_long_protein_context(
        selected,
        min_num_tokens=min_num_tokens,
        max_num_tokens=max_num_tokens,
    )
    if len(filtered) != len(selected):
        raise ValueError("benchmark index contains rows outside the protein/token filter")
    if selected.empty or len(selected) > max_rows:
        raise ValueError("benchmark index row count is empty or exceeds max_rows")
    if require_rows is not None and len(selected) != require_rows:
        raise ValueError(
            f"benchmark index requires exactly {require_rows} rows, got {len(selected)}"
        )
    if not selected["cluster_id"].is_unique or not selected["pdb_id"].is_unique:
        raise ValueError("benchmark index must be cluster-unique and PDB-unique")

    observed_selection = summarize_selection(selected)
    for key in ("rows", "pdbs", "clusters", "min_num_tokens", "max_num_tokens"):
        if summary.get(key) != observed_selection[key]:
            raise ValueError(f"benchmark summary field {key} does not match indices.csv")

    if ready_marker is not None:
        ready_marker = ready_marker.resolve()
        if not ready_marker.is_file():
            raise FileNotFoundError(f"missing benchmark ready marker: {ready_marker}")
        marker = json.loads(ready_marker.read_text(encoding="utf-8"))
        for key, expected_value in {
            **expected,
            "source_index_sha256": summary["source_index_sha256"],
            "index_sha256": actual_index_sha256,
            "rows": len(selected),
        }.items():
            if marker.get(key) != expected_value:
                raise ValueError(f"benchmark ready marker field {key} is stale")

    return selected, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rows", type=int, default=128)
    parser.add_argument("--min-num-tokens", type=int, default=769)
    parser.add_argument("--max-num-tokens", type=int, default=1536)
    parser.add_argument(
        "--selection",
        choices=("top_tokens", "random"),
        default="top_tokens",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ready-marker", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--require-rows", type=int)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-index-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    common_args = {
        "source_index": args.source_index,
        "output_dir": args.output_dir,
        "max_rows": args.max_rows,
        "min_num_tokens": args.min_num_tokens,
        "max_num_tokens": args.max_num_tokens,
        "selection": args.selection,
        "seed": args.seed,
    }
    if args.validate_only:
        selected, summary = validate_benchmark_index(
            **common_args,
            ready_marker=args.ready_marker,
            require_rows=args.require_rows,
        )
    else:
        selected, summary = build_benchmark_index(**common_args)
        if args.require_rows is not None and len(selected) != args.require_rows:
            raise ValueError(
                f"benchmark index requires exactly {args.require_rows} rows, "
                f"got {len(selected)}"
            )
        if args.ready_marker is not None:
            write_ready_marker(args.ready_marker, summary)
    if (
        args.expected_source_sha256
        and summary.get("source_index_sha256") != args.expected_source_sha256
    ):
        raise ValueError("source index SHA256 differs from the locked benchmark")
    if (
        args.expected_index_sha256
        and summary.get("index_sha256") != args.expected_index_sha256
    ):
        raise ValueError("generated index SHA256 differs from the locked benchmark")
    print(json.dumps(summary, indent=2, sort_keys=True))
    display_columns = [
        "pdb_id",
        "type",
        "mol_1_type",
        "mol_2_type",
        "num_tokens",
    ]
    print(selected[display_columns].head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
