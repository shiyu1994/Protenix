#!/usr/bin/env python3
"""Compute top-k DockQ metrics for SAb23H2 per-model DockQ CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path("/mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2")
DEFAULT_PER_MODEL_CSVS = [
    BASE_DIR / "dockq_eval_immune_interfaces" / "per_model_dockq.csv",
    BASE_DIR / "dockq_eval_immune_interfaces_20260525_999" / "per_model_dockq.csv",
    BASE_DIR / "dockq_eval_immune_interfaces_20260531" / "per_model_dockq.csv",
    BASE_DIR / "dockq_eval_immune_interfaces_raw999" / "per_model_dockq.csv",
]
DEFAULT_OUT_DIR = BASE_DIR / "topk_dockq_analysis"
DEFAULT_K_VALUES = [1, 3, 5, 10, 25]
SUCCESS_THRESHOLD = 0.23


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def ranked_by_score(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["ranking_score"]),
            float(row["immune_other_dockq"]),
            -int(row.get("seed", 0)),
            -int(row.get("sample", 0)),
        ),
        reverse=True,
    )


def topk_success(rows: list[dict[str, object]], k: int, threshold: float) -> bool:
    ranked = ranked_by_score(rows)
    return any(float(row["immune_other_dockq"]) > threshold for row in ranked[:k])


def topk_best_dockq(rows: list[dict[str, object]], k: int) -> float:
    ranked = ranked_by_score(rows)
    return max(float(row["immune_other_dockq"]) for row in ranked[:k])


def read_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows_by_key: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("dockq_status") != "OK":
                    continue
                dockq = safe_float(row.get("immune_other_dockq"))
                ranking_score = safe_float(row.get("ranking_score"))
                if dockq is None or ranking_score is None:
                    continue
                key = (
                    row.get("run_label", ""),
                    row.get("target", ""),
                    row.get("seed", ""),
                    row.get("sample", ""),
                )
                clean = dict(row)
                clean["immune_other_dockq"] = dockq
                clean["ranking_score"] = ranking_score
                rows_by_key[key] = clean
    return list(rows_by_key.values())


def summarize(
    rows: list[dict[str, object]],
    k_values: list[int],
    threshold: float,
    expected_models: int,
    mapping_ok_only: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if mapping_ok_only and row.get("mapping_status") != "OK":
            continue
        groups[(str(row["run_label"]), str(row["target"]))].append(row)

    per_target: list[dict[str, object]] = []
    for (run_label, target), group in sorted(groups.items()):
        if len(group) != expected_models:
            continue
        oracle = max(float(row["immune_other_dockq"]) for row in group)
        for k in k_values:
            best = topk_best_dockq(group, k)
            per_target.append(
                {
                    "filter": "mapping_ok" if mapping_ok_only else "all_scoreable",
                    "run_label": run_label,
                    "target": target,
                    "n_models": len(group),
                    "k": k,
                    "topk_best_dockq": best,
                    "topk_success": int(best > threshold),
                    "oracle_best_dockq": oracle,
                    "oracle_success": int(oracle > threshold),
                    "gap_to_oracle": oracle - best,
                }
            )

    by_run_k: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in per_target:
        by_run_k[(str(row["run_label"]), int(row["k"]))].append(row)

    summary: list[dict[str, object]] = []
    for (run_label, k), group in sorted(by_run_k.items()):
        topk = [float(row["topk_best_dockq"]) for row in group]
        oracle = [float(row["oracle_best_dockq"]) for row in group]
        gaps = [float(row["gap_to_oracle"]) for row in group]
        summary.append(
            {
                "filter": "mapping_ok" if mapping_ok_only else "all_scoreable",
                "run_label": run_label,
                "k": k,
                "n_targets": len(group),
                "topk_best_mean_dockq": statistics.fmean(topk),
                "topk_best_median_dockq": statistics.median(topk),
                "topk_success_rate": sum(int(row["topk_success"]) for row in group)
                / len(group),
                "oracle_best_mean_dockq": statistics.fmean(oracle),
                "oracle_success_rate": sum(int(row["oracle_success"]) for row in group)
                / len(group),
                "mean_gap_to_oracle": statistics.fmean(gaps),
                "median_gap_to_oracle": statistics.median(gaps),
            }
        )
    return per_target, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean = {
                key: f"{value:.6f}" if isinstance(value, float) else value
                for key, value in row.items()
            }
            writer.writerow(clean)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-model-csv",
        nargs="*",
        type=Path,
        default=DEFAULT_PER_MODEL_CSVS,
        help="One or more per_model_dockq.csv files.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--k-values", default="1,3,5,10,25")
    parser.add_argument("--threshold", type=float, default=SUCCESS_THRESHOLD)
    parser.add_argument("--expected-models", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    k_values = [int(x) for x in args.k_values.split(",") if x]
    rows = read_rows(args.per_model_csv)
    all_per_target: list[dict[str, object]] = []
    all_summary: list[dict[str, object]] = []
    for mapping_ok_only in (False, True):
        per_target, summary = summarize(
            rows,
            k_values,
            args.threshold,
            args.expected_models,
            mapping_ok_only,
        )
        all_per_target.extend(per_target)
        all_summary.extend(summary)

    write_csv(args.out_dir / "topk_per_target.csv", all_per_target)
    write_csv(args.out_dir / "topk_summary.csv", all_summary)
    print(f"rows_read={len(rows)}")
    print(f"per_target_rows={len(all_per_target)}")
    print(f"summary_rows={len(all_summary)}")
    print(f"out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
