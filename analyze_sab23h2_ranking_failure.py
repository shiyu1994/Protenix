#!/usr/bin/env python3
"""Analyze SAb23H2 ranking failures and simple confidence-based reranking.

This script consumes per_model_dockq.csv files produced by
eval_sab23h2_dockq.py. For each checkpoint/target, it compares the model picked
by Protenix ranking_score against the oracle DockQ-best model among the 25
samples, then tries alternative selectors derived from ptm/iptm/chain_pair_iptm.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


BASE_DIR = Path("/mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2")
DEFAULT_EVAL_DIRS = [
    BASE_DIR / "dockq_eval_immune_interfaces",
    BASE_DIR / "dockq_eval_immune_interfaces_20260525_999",
    BASE_DIR / "dockq_eval_immune_interfaces_20260531",
    BASE_DIR / "dockq_eval_immune_interfaces_raw999",
]
DEFAULT_OUT_DIR = BASE_DIR / "ranking_failure_reranking_analysis"
SUCCESS_THRESHOLD = 0.23
EXPECTED_MODELS_PER_TARGET = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dirs",
        nargs="*",
        default=[str(p) for p in DEFAULT_EVAL_DIRS],
        help="DockQ eval directories containing per_model_dockq.csv.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory for analysis CSV outputs.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=SUCCESS_THRESHOLD,
        help="DockQ threshold for success-rate summaries.",
    )
    parser.add_argument(
        "--expected-models",
        type=int,
        default=EXPECTED_MODELS_PER_TARGET,
        help="Require this many scoreable models per run/target group.",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return x


def parse_json(value: str, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def finite(values: list[float | None]) -> list[float]:
    return [x for x in values if x is not None and not math.isnan(x)]


def round_or_blank(value: float | None, ndigits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{ndigits}f}"


def input_indices_for_roles(row: dict[str, str]) -> tuple[list[int], list[int]]:
    roles = parse_json(row.get("model_chain_roles", ""), [])
    immune: list[int] = []
    other: list[int] = []
    for item in roles:
        try:
            idx = int(item["input_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if item.get("role") == "immune":
            immune.append(idx)
        elif item.get("role") == "other":
            other.append(idx)
    return sorted(set(immune)), sorted(set(other))


def add_confidence_features(row: dict[str, str]) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for key in ("ranking_score", "iptm", "ptm", "immune_other_dockq"):
        features[key] = safe_float(row.get(key))

    chain_iptm = finite([safe_float(x) for x in parse_json(row.get("chain_iptm", ""), [])])
    features["chain_iptm_mean"] = mean(chain_iptm)
    features["chain_iptm_min"] = min(chain_iptm) if chain_iptm else None

    matrix = parse_json(row.get("chain_pair_iptm", ""), [])
    pair_values: list[float] = []
    for i, values in enumerate(matrix):
        if not isinstance(values, list):
            continue
        for j, value in enumerate(values):
            if i == j:
                continue
            parsed = safe_float(value)
            if parsed is not None:
                pair_values.append(parsed)
    features["chain_pair_iptm_mean"] = mean(pair_values)

    immune_indices, other_indices = input_indices_for_roles(row)
    immune_other_values: list[float] = []
    for i in immune_indices:
        for j in other_indices:
            try:
                parsed = safe_float(matrix[i][j])
            except (IndexError, TypeError):
                parsed = None
            if parsed is not None:
                immune_other_values.append(parsed)
    features["immune_other_chain_pair_iptm_mean"] = mean(immune_other_values)
    features["immune_other_chain_pair_iptm_min"] = (
        min(immune_other_values) if immune_other_values else None
    )
    features["immune_other_chain_pair_iptm_max"] = (
        max(immune_other_values) if immune_other_values else None
    )
    return features


SELECTORS: dict[str, Callable[[dict[str, Any]], float | None]] = {
    "ranking_score": lambda row: row["features"].get("ranking_score"),
    "iptm": lambda row: row["features"].get("iptm"),
    "ptm": lambda row: row["features"].get("ptm"),
    "chain_iptm_mean": lambda row: row["features"].get("chain_iptm_mean"),
    "chain_iptm_min": lambda row: row["features"].get("chain_iptm_min"),
    "chain_pair_iptm_mean": lambda row: row["features"].get("chain_pair_iptm_mean"),
    "immune_other_chain_pair_iptm_mean": lambda row: row["features"].get(
        "immune_other_chain_pair_iptm_mean"
    ),
    "immune_other_chain_pair_iptm_min": lambda row: row["features"].get(
        "immune_other_chain_pair_iptm_min"
    ),
    "immune_other_chain_pair_iptm_max": lambda row: row["features"].get(
        "immune_other_chain_pair_iptm_max"
    ),
}


def read_rows(eval_dirs: list[Path]) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for eval_dir in eval_dirs:
        csv_path = eval_dir / "per_model_dockq.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                if raw.get("dockq_status") != "OK":
                    continue
                dockq = safe_float(raw.get("immune_other_dockq"))
                if dockq is None:
                    continue
                key = (
                    raw.get("run_label", ""),
                    raw.get("target", ""),
                    raw.get("seed", ""),
                    raw.get("sample", ""),
                )
                row = dict(raw)
                row["source_eval_dir"] = str(eval_dir)
                row["features"] = add_confidence_features(raw)
                rows_by_key[key] = row
    return list(rows_by_key.values())


def model_id(row: dict[str, Any]) -> str:
    return f"seed_{row.get('seed')}_sample_{row.get('sample')}"


def pick_best(
    rows: list[dict[str, Any]],
    selector_name: str,
    selector: Callable[[dict[str, Any]], float | None],
) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        value = selector(row)
        if value is None:
            continue
        tie_ranking = row["features"].get("ranking_score")
        tie_dockq = row["features"].get("immune_other_dockq")
        candidates.append(
            (
                value,
                tie_ranking if tie_ranking is not None else float("-inf"),
                tie_dockq if tie_dockq is not None else float("-inf"),
                -int(row.get("seed") or 0),
                -int(row.get("sample") or 0),
                row,
            )
        )
    if not candidates:
        return None
    picked = max(candidates)
    picked[-1]["selected_by_value"] = picked[0]
    picked[-1]["selected_by"] = selector_name
    return picked[-1]


def analyze_filter(
    rows: list[dict[str, Any]],
    filter_name: str,
    success_threshold: float,
    expected_models: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if filter_name == "mapping_ok" and row.get("mapping_status") != "OK":
            continue
        groups[(row["run_label"], row["target"])].append(row)

    per_target: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for (run_label, target), group in sorted(groups.items()):
        if len(group) != expected_models:
            continue
        oracle = max(
            group,
            key=lambda row: (
                row["features"]["immune_other_dockq"],
                row["features"].get("ranking_score") or float("-inf"),
            ),
        )
        rank_top = pick_best(group, "ranking_score", SELECTORS["ranking_score"])
        if rank_top is None:
            continue
        oracle_dockq = oracle["features"]["immune_other_dockq"]
        rank_dockq = rank_top["features"]["immune_other_dockq"]
        rank_gap = oracle_dockq - rank_dockq
        gap_rows.append(
            {
                "filter": filter_name,
                "run_label": run_label,
                "target": target,
                "ranking_top_model": model_id(rank_top),
                "ranking_top_dockq": rank_dockq,
                "ranking_top_score": rank_top["features"].get("ranking_score"),
                "oracle_model": model_id(oracle),
                "oracle_dockq": oracle_dockq,
                "oracle_ranking_score": oracle["features"].get("ranking_score"),
                "gap": rank_gap,
                "ranking_success": int(rank_dockq > success_threshold),
                "oracle_success": int(oracle_dockq > success_threshold),
            }
        )
        for selector_name, selector in SELECTORS.items():
            selected = pick_best(group, selector_name, selector)
            if selected is None:
                continue
            selected_dockq = selected["features"]["immune_other_dockq"]
            per_target.append(
                {
                    "filter": filter_name,
                    "run_label": run_label,
                    "target": target,
                    "n_models": len(group),
                    "selector": selector_name,
                    "selected_model": model_id(selected),
                    "selector_value": selected.get("selected_by_value"),
                    "selected_dockq": selected_dockq,
                    "selected_success": int(selected_dockq > success_threshold),
                    "oracle_model": model_id(oracle),
                    "oracle_dockq": oracle_dockq,
                    "oracle_success": int(oracle_dockq > success_threshold),
                    "gap_to_oracle": oracle_dockq - selected_dockq,
                    "selected_exact_oracle_model": int(model_id(selected) == model_id(oracle)),
                    "selected_ties_oracle_dockq": int(abs(selected_dockq - oracle_dockq) < 1e-9),
                    "ranking_top_model": model_id(rank_top),
                    "ranking_top_dockq": rank_dockq,
                    "delta_vs_ranking_top": selected_dockq - rank_dockq,
                }
            )

    summary: list[dict[str, Any]] = []
    by_run_selector: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_target:
        by_run_selector[(row["run_label"], row["selector"])].append(row)
    for (run_label, selector), values in sorted(by_run_selector.items()):
        selected = [float(v["selected_dockq"]) for v in values]
        oracle = [float(v["oracle_dockq"]) for v in values]
        gaps = [float(v["gap_to_oracle"]) for v in values]
        deltas = [float(v["delta_vs_ranking_top"]) for v in values]
        summary.append(
            {
                "filter": filter_name,
                "run_label": run_label,
                "selector": selector,
                "n_targets": len(values),
                "selected_mean_dockq": mean(selected),
                "selected_median_dockq": statistics.median(selected),
                "selected_success_rate": sum(v["selected_success"] for v in values)
                / len(values),
                "oracle_best_mean_dockq": mean(oracle),
                "oracle_best_success_rate": sum(v["oracle_success"] for v in values)
                / len(values),
                "mean_gap_to_oracle": mean(gaps),
                "median_gap_to_oracle": statistics.median(gaps),
                "exact_oracle_model_rate": sum(v["selected_exact_oracle_model"] for v in values)
                / len(values),
                "ties_oracle_dockq_rate": sum(v["selected_ties_oracle_dockq"] for v in values)
                / len(values),
                "mean_delta_vs_ranking_top": mean(deltas),
                "targets_better_than_ranking_top": sum(1 for x in deltas if x > 1e-9),
                "targets_worse_than_ranking_top": sum(1 for x in deltas if x < -1e-9),
            }
        )
    return per_target, summary, gap_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean = dict(row)
            for key, value in list(clean.items()):
                if isinstance(value, float):
                    clean[key] = round_or_blank(value)
            writer.writerow(clean)


def summarize_ranking_failures(
    gaps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in gaps:
        grouped[(row["filter"], row["run_label"])].append(row)

    rows: list[dict[str, Any]] = []
    for (filter_name, run_label), values in sorted(grouped.items()):
        ranking = [float(v["ranking_top_dockq"]) for v in values]
        oracle = [float(v["oracle_dockq"]) for v in values]
        gap = [float(v["gap"]) for v in values]
        rows.append(
            {
                "filter": filter_name,
                "run_label": run_label,
                "n_targets": len(values),
                "ranking_top_mean_dockq": mean(ranking),
                "ranking_top_success_rate": sum(int(v["ranking_success"]) for v in values)
                / len(values),
                "oracle_best_mean_dockq": mean(oracle),
                "oracle_best_success_rate": sum(int(v["oracle_success"]) for v in values)
                / len(values),
                "mean_gap": mean(gap),
                "median_gap": statistics.median(gap),
                "gap_gt_0.05_targets": sum(1 for x in gap if x > 0.05),
                "gap_gt_0.10_targets": sum(1 for x in gap if x > 0.10),
                "ranking_failed_oracle_succeeded_targets": sum(
                    1
                    for v in values
                    if int(v["ranking_success"]) == 0 and int(v["oracle_success"]) == 1
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    eval_dirs = [Path(p) for p in args.eval_dirs]
    out_dir = Path(args.out_dir)
    rows = read_rows(eval_dirs)
    per_target_all: list[dict[str, Any]] = []
    summary_all: list[dict[str, Any]] = []
    gaps_all: list[dict[str, Any]] = []
    for filter_name in ("all_scoreable", "mapping_ok"):
        per_target, summary, gaps = analyze_filter(
            rows,
            filter_name,
            args.success_threshold,
            args.expected_models,
        )
        per_target_all.extend(per_target)
        summary_all.extend(summary)
        gaps_all.extend(gaps)

    summary_fields = [
        "filter",
        "run_label",
        "selector",
        "n_targets",
        "selected_mean_dockq",
        "selected_median_dockq",
        "selected_success_rate",
        "oracle_best_mean_dockq",
        "oracle_best_success_rate",
        "mean_gap_to_oracle",
        "median_gap_to_oracle",
        "exact_oracle_model_rate",
        "ties_oracle_dockq_rate",
        "mean_delta_vs_ranking_top",
        "targets_better_than_ranking_top",
        "targets_worse_than_ranking_top",
    ]
    per_target_fields = [
        "filter",
        "run_label",
        "target",
        "n_models",
        "selector",
        "selected_model",
        "selector_value",
        "selected_dockq",
        "selected_success",
        "oracle_model",
        "oracle_dockq",
        "oracle_success",
        "gap_to_oracle",
        "selected_exact_oracle_model",
        "selected_ties_oracle_dockq",
        "ranking_top_model",
        "ranking_top_dockq",
        "delta_vs_ranking_top",
    ]
    gap_fields = [
        "filter",
        "run_label",
        "target",
        "ranking_top_model",
        "ranking_top_dockq",
        "ranking_top_score",
        "oracle_model",
        "oracle_dockq",
        "oracle_ranking_score",
        "gap",
        "ranking_success",
        "oracle_success",
    ]
    ranking_failure_fields = [
        "filter",
        "run_label",
        "n_targets",
        "ranking_top_mean_dockq",
        "ranking_top_success_rate",
        "oracle_best_mean_dockq",
        "oracle_best_success_rate",
        "mean_gap",
        "median_gap",
        "gap_gt_0.05_targets",
        "gap_gt_0.10_targets",
        "ranking_failed_oracle_succeeded_targets",
    ]

    write_csv(out_dir / "selector_summary.csv", summary_all, summary_fields)
    write_csv(out_dir / "per_target_selector_results.csv", per_target_all, per_target_fields)
    write_csv(out_dir / "ranking_top_oracle_gaps.csv", gaps_all, gap_fields)
    write_csv(
        out_dir / "ranking_failure_summary.csv",
        summarize_ranking_failures(gaps_all),
        ranking_failure_fields,
    )

    high_gap = sorted(gaps_all, key=lambda row: float(row["gap"]), reverse=True)[:200]
    write_csv(out_dir / "largest_ranking_failures.csv", high_gap, gap_fields)

    print(f"rows_read={len(rows)}")
    print(f"summary_rows={len(summary_all)}")
    print(f"per_target_rows={len(per_target_all)}")
    print(f"gap_rows={len(gaps_all)}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
