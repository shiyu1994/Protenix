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

"""Summarize v9 runtime windows from a training ``train_metrics.jsonl``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REQUIRED_TIMING_FIELDS = {
    "data_wait_seconds": ("p50", "p95", "p99", "max", "max_rank"),
    "data_process_seconds": ("p95", "max"),
    "loss_health_sync_seconds": ("p95", "max"),
    "step_wall_seconds": ("p95", "p99", "max"),
}
REQUIRED_HEALTH_FIELDS = (
    "grad_clip_count",
    "loss_nan_count",
    "nonfinite_grad_count",
    "optimizer_skipped_count",
)
OUTLIER_IDENTITY_FIELDS = (
    "rank",
    "step",
    "global_step",
    "dataset_name",
    "data_id",
    "pdb_id",
    "sample_global_position",
    "chain_id",
)
OUTLIER_DETAIL_FIELDS = (
    *OUTLIER_IDENTITY_FIELDS,
    "n_token",
    "n_atom",
    "N_cycle",
    "N_msa",
    "N_msa_prot_pair",
    "N_msa_prot_unpair",
    "N_msa_rna_pair",
    "N_msa_rna_unpair",
    "N_msa_virtual_segment",
    "msa_virtual_segment_shuffle_applied",
    "N_msa_shuffle_group",
    "msa_shuffle_requested_variant_code",
    "msa_shuffle_variant_eligible",
    "msa_shuffle_variant_applied",
    "msa_shuffle_whole_physical_chain_applied",
    "N_msa_crop_protein_chain",
    "N_msa_query_only_protein_chain",
    "N_msa_retained_protein_chain",
    "msa_partial_query_only_applied",
    "msa_global_query_only_applied",
    "N_prot_token",
    "N_lig_token",
    "N_rna_token",
    "N_dna_token",
    "msa_mode",
    "query_only_msa",
    "unpaired_msa",
    "N_template_valid",
    "template_use_requested",
    "sample_sampler_epoch",
)


class MetricsSummaryError(ValueError):
    """Raised when a metrics file does not satisfy the v9 summary contract."""


def _required_mapping(
    value: Any,
    *,
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MetricsSummaryError(f"{context} must be a JSON object")
    return value


def _required_finite_number(
    mapping: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> float | int:
    if key not in mapping:
        raise MetricsSummaryError(f"{context} is missing required field {key!r}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetricsSummaryError(f"{context}.{key} must be a finite number")
    if not math.isfinite(float(value)):
        raise MetricsSummaryError(f"{context}.{key} must be a finite number")
    return value


def _required_nonnegative_int(
    mapping: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> int:
    value = _required_finite_number(mapping, key, context=context)
    if not isinstance(value, int) or value < 0:
        raise MetricsSummaryError(f"{context}.{key} must be a non-negative integer")
    return value


def _required_int(
    mapping: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> int:
    value = _required_finite_number(mapping, key, context=context)
    if not isinstance(value, int):
        raise MetricsSummaryError(f"{context}.{key} must be an integer")
    return value


def _json_identity_value(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise MetricsSummaryError(
            f"top outlier identity is not JSON serializable: {exc}"
        ) from exc


def _merge_top_outliers(
    top_outliers: Mapping[str, Any],
    *,
    context: str,
) -> list[dict[str, Any]]:
    """Merge one sample's entries across runtime-outlier categories."""
    merged: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}

    for metric_name, raw_entries in top_outliers.items():
        if not isinstance(metric_name, str):
            raise MetricsSummaryError(f"{context} contains a non-string metric name")
        if isinstance(raw_entries, (str, bytes)) or not isinstance(
            raw_entries, Sequence
        ):
            raise MetricsSummaryError(f"{context}.{metric_name} must be a JSON array")

        for index, raw_entry in enumerate(raw_entries):
            entry_context = f"{context}.{metric_name}[{index}]"
            entry = _required_mapping(raw_entry, context=entry_context)
            metric_value = _required_finite_number(
                entry,
                metric_name,
                context=entry_context,
            )
            rank = _required_int(entry, "rank", context=entry_context)
            step = _required_int(entry, "step", context=entry_context)
            global_step = _required_int(
                entry,
                "global_step",
                context=entry_context,
            )

            identity_values = {
                field: entry[field]
                for field in OUTLIER_IDENTITY_FIELDS
                if field in entry
            }
            identity_values.update(
                {"rank": rank, "step": step, "global_step": global_step}
            )
            identity = tuple(
                (field, _json_identity_value(identity_values[field]))
                for field in OUTLIER_IDENTITY_FIELDS
                if field in identity_values
            )

            if identity not in merged:
                merged[identity] = {
                    field: entry[field]
                    for field in OUTLIER_DETAIL_FIELDS
                    if field in entry
                }
                merged[identity].update(
                    {"rank": rank, "step": step, "global_step": global_step}
                )
                merged[identity]["metrics"] = {}
            metrics = merged[identity]["metrics"]
            metrics[metric_name] = max(
                metric_value,
                metrics.get(metric_name, metric_value),
            )

    return list(merged.values())


def summarize_train_metrics_record(
    record: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """Validate and compact one ``train_metrics`` JSONL record."""
    event = record.get("event")
    if event != "train_metrics":
        raise MetricsSummaryError(
            f"{context}.event must be 'train_metrics', got {event!r}"
        )
    global_step = _required_int(record, "global_step", context=context)
    runtime = _required_mapping(
        record.get("runtime_window"),
        context=f"{context}.runtime_window",
    )
    timings = _required_mapping(
        runtime.get("timings"),
        context=f"{context}.runtime_window.timings",
    )
    grad_health = _required_mapping(
        runtime.get("grad_health"),
        context=f"{context}.runtime_window.grad_health",
    )
    raw_top_outliers = _required_mapping(
        runtime.get("top_outliers"),
        context=f"{context}.runtime_window.top_outliers",
    )

    compact_timings: dict[str, dict[str, Any]] = {}
    for timing_name, fields in REQUIRED_TIMING_FIELDS.items():
        timing = _required_mapping(
            timings.get(timing_name),
            context=f"{context}.runtime_window.timings.{timing_name}",
        )
        compact_timing = {}
        for field in fields:
            if field == "max_rank":
                compact_timing[field] = _required_int(
                    timing,
                    field,
                    context=(
                        f"{context}.runtime_window.timings.{timing_name}"
                    ),
                )
            else:
                compact_timing[field] = _required_finite_number(
                    timing,
                    field,
                    context=(
                        f"{context}.runtime_window.timings.{timing_name}"
                    ),
                )
        compact_timings[timing_name] = compact_timing

    health = {
        field: _required_nonnegative_int(
            grad_health,
            field,
            context=f"{context}.runtime_window.grad_health",
        )
        for field in REQUIRED_HEALTH_FIELDS
    }

    return {
        "global_step": global_step,
        "end_to_end_elapsed_seconds": _required_finite_number(
            runtime,
            "end_to_end_elapsed_seconds",
            context=f"{context}.runtime_window",
        ),
        "end_to_end_samples_per_sec": _required_finite_number(
            runtime,
            "end_to_end_samples_per_sec",
            context=f"{context}.runtime_window",
        ),
        "timings": compact_timings,
        "health": health,
        "top_outliers": _merge_top_outliers(
            raw_top_outliers,
            context=f"{context}.runtime_window.top_outliers",
        ),
    }


def load_summary(path: Path) -> dict[str, Any]:
    """Load a JSONL metrics file and return a validated compact summary."""
    if not path.exists():
        raise MetricsSummaryError(f"metrics file does not exist: {path}")
    if not path.is_file():
        raise MetricsSummaryError(f"metrics path is not a regular file: {path}")

    windows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MetricsSummaryError(f"could not read metrics file {path}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        context = f"{path}:line {line_number}"
        try:
            raw_record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MetricsSummaryError(
                f"{context} contains malformed JSON at column {exc.colno}: {exc.msg}"
            ) from exc
        record = _required_mapping(raw_record, context=context)
        windows.append(summarize_train_metrics_record(record, context=context))

    if not windows:
        raise MetricsSummaryError(
            f"metrics file contains no train_metrics records: {path}"
        )
    return {
        "source": str(path.resolve()),
        "window_count": len(windows),
        "windows": windows,
    }


def _format_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render a compact window table and merged outlier table."""
    lines = [
        "# v9 training runtime summary",
        "",
        f"Source: `{summary['source']}`",
        "",
        f"Runtime windows: {summary['window_count']}",
        "",
        (
            "| step | E2E s | samples/s | data wait p50/p95/p99/max (rank) "
            "| data process p95/max | loss sync p95/max "
            "| step wall p95/p99/max | clip/NaN/nonfinite/skip |"
        ),
        (
            "| ---: | ---: | ---: | --- | --- | --- | --- "
            "| --- |"
        ),
    ]
    for window in summary["windows"]:
        timings = window["timings"]
        data_wait = timings["data_wait_seconds"]
        data_process = timings["data_process_seconds"]
        loss_sync = timings["loss_health_sync_seconds"]
        step_wall = timings["step_wall_seconds"]
        health = window["health"]
        lines.append(
            "| {step} | {elapsed} | {throughput} | {wait} | {process} "
            "| {sync} | {wall} | {health} |".format(
                step=window["global_step"],
                elapsed=_format_number(window["end_to_end_elapsed_seconds"]),
                throughput=_format_number(window["end_to_end_samples_per_sec"]),
                wait="/".join(
                    _format_number(data_wait[field])
                    for field in ("p50", "p95", "p99", "max")
                )
                + f" (r{data_wait['max_rank']})",
                process="/".join(
                    _format_number(data_process[field])
                    for field in ("p95", "max")
                ),
                sync="/".join(
                    _format_number(loss_sync[field])
                    for field in ("p95", "max")
                ),
                wall="/".join(
                    _format_number(step_wall[field])
                    for field in ("p95", "p99", "max")
                ),
                health="/".join(
                    str(health[field]) for field in REQUIRED_HEALTH_FIELDS
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Merged top outlier samples",
            "",
            "| window step | sample step | rank | dataset | PDB/data ID | tokens/atoms | slow metrics (s) |",
            "| ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    outlier_count = 0
    for window in summary["windows"]:
        for outlier in window["top_outliers"]:
            outlier_count += 1
            pdb_or_data = outlier.get("pdb_id") or outlier.get("data_id") or "-"
            size = f"{outlier.get('n_token', '-')}/{outlier.get('n_atom', '-')}"
            metrics = ", ".join(
                f"{name}={_format_number(value)}"
                for name, value in outlier["metrics"].items()
            )
            lines.append(
                "| {window_step} | {sample_step} | {rank} | {dataset} "
                "| {sample} | {size} | {metrics} |".format(
                    window_step=window["global_step"],
                    sample_step=outlier["step"],
                    rank=outlier["rank"],
                    dataset=_escape_markdown(outlier.get("dataset_name", "-")),
                    sample=_escape_markdown(pdb_or_data),
                    size=size,
                    metrics=_escape_markdown(metrics),
                )
            )
    if outlier_count == 0:
        lines.append("| - | - | - | - | - | - | none |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize v9 runtime windows from logs/train_metrics.jsonl. "
            "The input file is never modified."
        )
    )
    parser.add_argument("metrics_path", type=Path, help="path to train_metrics.jsonl")
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="markdown",
        help="stdout format (default: markdown)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = load_summary(args.metrics_path)
    except MetricsSummaryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(render_markdown(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
