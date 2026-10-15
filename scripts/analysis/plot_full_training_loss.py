#!/usr/bin/env python3
"""Plot the audited main-line training loss from random initialization.

The training history spans several output directories and one intentional
params-only step reset.  This script encodes the checkpoint ancestry and the
exact non-overlapping slices, rather than globbing every run directory (which
would mix failed restarts and experimental branches into the curve).

Each JSONL record is already a mean over 25 optimizer updates.  The plot keeps
that series visible and overlays:

* a trailing 500-update arithmetic mean (20 records), and
* an EWMA with a 2,000-update half-life (80 records).

Both smoothers reset at genuine training/data-contract changes.  They do not
reset at the step-47199 same-contract full-state resume.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, ScalarFormatter


DEFAULT_OUTPUT_DIR = Path(
    "/mnt/project_abd/klp/proteinx_output/training_monitor_plots"
)


@dataclass(frozen=True)
class LineageSlice:
    name: str
    path: Path
    min_step: int | None
    max_step: int | None
    cumulative_offset: int
    smooth_group: str
    description: str


LINEAGE: tuple[LineageSlice, ...] = (
    LineageSlice(
        name="v8_scratch",
        path=Path(
            "/mnt/project_abd/klp/proteinx_output/"
            "protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_msamix_q20_u30_"
            "p50_50k_v8_20260724_155026/logs/train_metrics.jsonl"
        ),
        min_step=None,
        max_step=999,
        cumulative_offset=0,
        smooth_group="scratch_v8",
        description="random initialization; keep through checkpoint step 999",
    ),
    LineageSlice(
        name="v9_templatecache",
        path=Path(
            "/mnt/project_abd/klp/proteinx_output/"
            "protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_"
            "straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_"
            "ckpt200_20260726_061627/logs/train_metrics.jsonl"
        ),
        min_step=None,
        max_step=9999,
        cumulative_offset=1000,
        smooth_group="v9_templatecache",
        description="params-only start from v8 step 999; local step reset",
    ),
    LineageSlice(
        name="v11_query_pin",
        path=Path(
            "/mnt/project_abd/klp/proteinx_output/"
            "protenix_v4_crop384_db48_disordered_v4_v10_to_v11_"
            "msaquerypinle1024_from_s9999_v9_straggler_20260730_104039/"
            "logs/train_metrics.jsonl"
        ),
        min_step=10000,
        max_step=21999,
        cumulative_offset=1000,
        smooth_group="v11_query_pin",
        description="query sequence pinning for sampled MSA <= 1024",
    ),
    LineageSlice(
        name="residual_ge200_softcluster_g02",
        path=Path(
            "/mnt/project_abd/klp/proteinx_output/"
            "protenix_v4_step21999_residual_ge200_softcluster_g02_v9_"
            "straggler_20260804_140441/logs/train_metrics.jsonl"
        ),
        min_step=22000,
        max_step=27799,
        cumulative_offset=1000,
        smooth_group="residual_ge200_softcluster_g02",
        description="residual ge200 catalog plus soft-cluster gamma=0.2",
    ),
    LineageSlice(
        name="msaindependent_only",
        path=Path(
            "/mnt/project_abd/klp/proteinx_output/"
            "protenix_v4_step27799_msaindependent_only_softcluster_g02_v9_"
            "straggler_20260807_053504/logs/train_metrics.jsonl"
        ),
        min_step=27800,
        max_step=47199,
        cumulative_offset=1000,
        smooth_group="msaindependent_only",
        description="independent paired/unpaired source budgets for weightedPDB",
    ),
    LineageSlice(
        name="msaindependent_only_resume",
        path=Path(
            "/mnt/project_abd/klp/proteinx_output/"
            "protenix_v4_from47199_msaindependent_only_softcluster_g02_v9_"
            "straggler_20260814_064306/logs/train_metrics.jsonl"
        ),
        min_step=47200,
        max_step=None,
        cumulative_offset=1000,
        # Same training contract and a full-state resume: keep one smoother.
        smooth_group="msaindependent_only",
        description="same-contract full-state resume from step 47199",
    ),
)


PHASE_BOUNDARIES: tuple[tuple[int, str, bool], ...] = (
    (1000, "v8 -> v9\nparams-only reset", True),
    (11000, "query-pin\nmigration", True),
    (23000, "ge200 + soft-cluster\ngamma=0.2", True),
    (28800, "independent paired /\nunpaired budgets", True),
    (48200, "same-contract\nfull-state resume", False),
)


def _scalar_lr(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        return float(value[0]) if value else math.nan
    if value is None:
        return math.nan
    return float(value)


def _iter_selected_records(spec: LineageSlice) -> Iterable[dict[str, Any]]:
    if not spec.path.is_file():
        raise FileNotFoundError(f"Missing lineage metrics file: {spec.path}")
    with spec.path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            step = int(record["step"])
            if spec.min_step is not None and step < spec.min_step:
                continue
            if spec.max_step is not None and step > spec.max_step:
                continue
            try:
                loss = float(record["metrics"]["train/loss.avg"])
            except KeyError as exc:
                raise KeyError(
                    f"Missing train/loss.avg in {spec.path}:{line_number}"
                ) from exc
            yield {
                "stage": spec.name,
                "smooth_group": spec.smooth_group,
                "source_step": step,
                # step is zero based and denotes the end of the 25-update window.
                "cumulative_update": step + spec.cumulative_offset + 1,
                "window_start_update": step + spec.cumulative_offset - 23,
                "window_end_update": step + spec.cumulative_offset + 1,
                "train_loss_avg_25_updates": loss,
                "lr": _scalar_lr(record.get("lr")),
                "grad_norm": float(record.get("grad_norm", math.nan)),
                "loss_nan": bool(record.get("loss_nan", False)),
                "global_step": record.get("global_step"),
                "source_file": str(spec.path),
            }


def load_lineage() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    source_summary: list[dict[str, Any]] = []
    for spec in LINEAGE:
        selected = list(_iter_selected_records(spec))
        if not selected:
            raise RuntimeError(f"Lineage slice selected no records: {spec.name}")
        steps = np.asarray([row["source_step"] for row in selected], dtype=np.int64)
        if len(steps) > 1 and not np.all(np.diff(steps) == 25):
            raise RuntimeError(f"Non-25-step gap inside lineage slice {spec.name}")
        rows.extend(selected)
        stat = spec.path.stat()
        source_summary.append(
            {
                "name": spec.name,
                "description": spec.description,
                "source_file": str(spec.path),
                "configured_min_step": spec.min_step,
                "configured_max_step": spec.max_step,
                "selected_first_step": int(steps[0]),
                "selected_last_step": int(steps[-1]),
                "selected_records": int(len(steps)),
                "file_size_bytes": stat.st_size,
                "file_mtime_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )

    frame = pd.DataFrame(rows).sort_values("cumulative_update").reset_index(drop=True)
    if frame["cumulative_update"].duplicated().any():
        duplicate = frame.loc[
            frame["cumulative_update"].duplicated(False), "cumulative_update"
        ].tolist()
        raise RuntimeError(f"Duplicate cumulative updates in lineage: {duplicate}")
    gaps = frame["cumulative_update"].diff().dropna()
    if not (gaps == 25).all():
        bad = frame.loc[gaps.index[gaps != 25], ["stage", "cumulative_update"]]
        raise RuntimeError(f"Combined lineage has a non-25-update gap:\n{bad}")
    if not np.isfinite(frame["train_loss_avg_25_updates"]).all():
        raise RuntimeError("Lineage contains a non-finite training loss")
    if (frame["train_loss_avg_25_updates"] <= 0).any():
        raise RuntimeError("Lineage contains a non-positive training loss")

    # A 500-update trailing mean and a 2,000-update half-life EWMA.  Grouping
    # prevents either curve from blending across real contract changes.
    frame["loss_trailing_mean_500_updates"] = np.nan
    frame["loss_ewma_halflife_2000_updates"] = np.nan
    for _, index in frame.groupby("smooth_group", sort=False).groups.items():
        values = frame.loc[index, "train_loss_avg_25_updates"]
        frame.loc[index, "loss_trailing_mean_500_updates"] = (
            values.rolling(window=20, min_periods=1).mean().to_numpy()
        )
        frame.loc[index, "loss_ewma_halflife_2000_updates"] = (
            values.ewm(halflife=80, adjust=False).mean().to_numpy()
        )
    return frame, source_summary


def _thousands(value: float, _: int) -> str:
    return f"{int(value):,}"


def _plot_lines(ax: plt.Axes, frame: pd.DataFrame, include_legend: bool) -> None:
    # Split at smoothing groups so no line visually bridges a contract reset.
    first_group = True
    for _, group in frame.groupby("smooth_group", sort=False):
        ax.plot(
            group["cumulative_update"],
            group["train_loss_avg_25_updates"],
            color="#94a3b8",
            alpha=0.34,
            linewidth=0.65,
            label="25-update window mean" if first_group else None,
            zorder=1,
        )
        ax.plot(
            group["cumulative_update"],
            group["loss_trailing_mean_500_updates"],
            color="#2563eb",
            linewidth=2.0,
            label="Trailing 500-update mean" if first_group else None,
            zorder=3,
        )
        ax.plot(
            group["cumulative_update"],
            group["loss_ewma_halflife_2000_updates"],
            color="#c2410c",
            linewidth=1.7,
            alpha=0.92,
            label="EWMA (2,000-update half-life)" if first_group else None,
            zorder=2,
        )
        first_group = False
    if include_legend:
        ax.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=9)


def _decorate_axes(ax: plt.Axes, x_min: int, x_max: int, annotate: bool) -> None:
    for boundary, label, contract_change in PHASE_BOUNDARIES:
        if x_min <= boundary <= x_max:
            ax.axvline(
                boundary,
                color="#7c3aed" if contract_change else "#0f766e",
                linestyle="--" if contract_change else ":",
                linewidth=0.9,
                alpha=0.65,
                zorder=0,
            )
            if annotate:
                ax.text(
                    boundary,
                    0.98,
                    label,
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=7.5,
                    color="#4c1d95" if contract_change else "#115e59",
                    bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
                )
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.grid(axis="y", color="#cbd5e1", alpha=0.55, linewidth=0.7)
    ax.grid(axis="x", color="#e2e8f0", alpha=0.34, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Training loss")


def make_plot(frame: pd.DataFrame, generated_at: datetime) -> plt.Figure:
    latest = frame.iloc[-1]
    latest_update = int(latest["cumulative_update"])
    latest_step = int(latest["source_step"])

    fig = plt.figure(figsize=(15.2, 12.2), constrained_layout=False)
    grid = fig.add_gridspec(3, 1, height_ratios=(1.12, 1.0, 1.0), hspace=0.30)
    axes = [fig.add_subplot(grid[row, 0]) for row in range(3)]

    # Panel 1: the large random-initialization transient is legible on log y.
    _plot_lines(axes[0], frame, include_legend=True)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(ScalarFormatter())
    axes[0].yaxis.set_minor_formatter(ScalarFormatter())
    axes[0].tick_params(axis="y", which="minor", labelsize=7)
    _decorate_axes(
        axes[0], int(frame["cumulative_update"].min()), latest_update, annotate=True
    )
    axes[0].set_title(
        "A. Full history (log y-axis preserves the random-initialization transient)",
        loc="left",
        fontsize=11,
        fontweight="semibold",
    )

    # Panel 2: current-v4 stage, after the first 1,000 scratch updates.
    post_scratch = frame.loc[frame["cumulative_update"] > 1000]
    _plot_lines(axes[1], post_scratch, include_legend=False)
    _decorate_axes(axes[1], 1000, latest_update, annotate=False)
    y_low = max(0.0, float(post_scratch["train_loss_avg_25_updates"].min()) - 0.08)
    y_high = float(post_scratch["train_loss_avg_25_updates"].max()) + 0.08
    axes[1].set_ylim(y_low, y_high)
    axes[1].set_title(
        "B. After the first 1,000 scratch updates (linear y-axis)",
        loc="left",
        fontsize=11,
        fontweight="semibold",
    )

    # Panel 3: late-stage convergence is otherwise compressed in the full plot.
    recent_min = max(int(frame["cumulative_update"].min()), latest_update - 20000)
    recent = frame.loc[frame["cumulative_update"] >= recent_min]
    _plot_lines(axes[2], recent, include_legend=False)
    _decorate_axes(axes[2], recent_min, latest_update, annotate=False)
    recent_low = float(recent["train_loss_avg_25_updates"].quantile(0.003)) - 0.015
    recent_high = float(recent["train_loss_avg_25_updates"].quantile(0.997)) + 0.015
    axes[2].set_ylim(recent_low, recent_high)
    axes[2].set_title(
        "C. Most recent 20,000 updates (linear y-axis)",
        loc="left",
        fontsize=11,
        fontweight="semibold",
    )
    axes[2].set_xlabel("Cumulative optimizer updates from random initialization")

    latest_500 = float(latest["loss_trailing_mean_500_updates"])
    latest_ewma = float(latest["loss_ewma_halflife_2000_updates"])
    axes[2].scatter(
        [latest_update], [latest_500], s=28, color="#1d4ed8", zorder=5, clip_on=False
    )
    axes[2].annotate(
        f"latest 500-update mean = {latest_500:.4f}",
        xy=(latest_update, latest_500),
        xytext=(-8, 12),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        color="#1d4ed8",
    )

    fig.suptitle(
        "Protenix training loss — audited full ancestor lineage",
        x=0.07,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.07,
        0.956,
        (
            "Random initialization to current training state; failed restarts, "
            "experimental branches, and overlapped parent tails are excluded."
        ),
        ha="left",
        fontsize=10,
        color="#475569",
    )
    fig.text(
        0.07,
        0.018,
        (
            f"Snapshot: source step {latest_step:,} = {latest_update:,} cumulative "
            f"optimizer updates | {len(frame):,} logged 25-update windows | "
            f"latest loss: raw-window {float(latest['train_loss_avg_25_updates']):.4f}, "
            f"500-update mean {latest_500:.4f}, EWMA {latest_ewma:.4f} | "
            f"generated {generated_at.strftime('%Y-%m-%d %H:%M UTC')}"
        ),
        ha="left",
        fontsize=8.3,
        color="#475569",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.925, bottom=0.065)
    return fig


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _write_json_atomic(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc)
    frame, source_summary = load_lineage()
    latest = frame.iloc[-1]
    latest_step = int(latest["source_step"])
    latest_update = int(latest["cumulative_update"])
    stem = (
        f"protenix_full_lineage_loss_through_step{latest_step}_cum{latest_update}"
    )

    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    csv_path = output_dir / f"{stem}.csv"
    manifest_path = output_dir / f"{stem}.manifest.json"

    figure = make_plot(frame, generated_at)
    figure.savefig(png_path, dpi=180, facecolor="white")
    figure.savefig(svg_path, facecolor="white")
    plt.close(figure)
    frame.to_csv(csv_path, index=False, float_format="%.10g")

    manifest = {
        "generated_at_utc": generated_at.isoformat(),
        "definition": (
            "Audited checkpoint ancestry from random initialization; each raw loss "
            "point is the logged mean over 25 optimizer updates."
        ),
        "latest_source_step": latest_step,
        "latest_cumulative_optimizer_updates": latest_update,
        "metric_records": int(len(frame)),
        "first_cumulative_update": int(frame.iloc[0]["cumulative_update"]),
        "last_cumulative_update": latest_update,
        "smoothing": {
            "short": "trailing arithmetic mean over 20 records = 500 updates",
            "long": "EWMA half-life 80 records = 2,000 updates",
            "reset_groups": frame["smooth_group"].drop_duplicates().tolist(),
            "same_contract_step47199_resume_resets_smoother": False,
        },
        "latest_values": {
            "train_loss_avg_25_updates": float(
                latest["train_loss_avg_25_updates"]
            ),
            "loss_trailing_mean_500_updates": float(
                latest["loss_trailing_mean_500_updates"]
            ),
            "loss_ewma_halflife_2000_updates": float(
                latest["loss_ewma_halflife_2000_updates"]
            ),
        },
        "phase_boundaries_cumulative_updates": [
            {
                "cumulative_update": update,
                "label": label.replace("\n", " "),
                "smoothing_reset": reset,
            }
            for update, label, reset in PHASE_BOUNDARIES
        ],
        "lineage_sources": source_summary,
        "explicitly_excluded": [
            "three dummytemplatefix1 short runs",
            "step27799 pdbmember LR rewarm experimental branch",
            "step27799 msaindependent failed short restart at 20260814_044445",
            "all parent-run records after each selected migration checkpoint",
            "v8 metric step 1024 after selected checkpoint step 999",
        ],
        "artifacts": {
            "png": str(png_path),
            "svg": str(svg_path),
            "csv": str(csv_path),
            "manifest": str(manifest_path),
        },
    }
    _write_json_atomic(manifest, manifest_path)

    latest_files = {
        png_path: output_dir / "protenix_full_lineage_loss_latest.png",
        svg_path: output_dir / "protenix_full_lineage_loss_latest.svg",
        csv_path: output_dir / "protenix_full_lineage_loss_latest.csv",
        manifest_path: output_dir / "protenix_full_lineage_loss_latest.manifest.json",
    }
    for source, destination in latest_files.items():
        _atomic_copy(source, destination)

    print(
        json.dumps(
            {
                "png": str(png_path),
                "latest_png": str(latest_files[png_path]),
                "latest_source_step": latest_step,
                "latest_cumulative_optimizer_updates": latest_update,
                "metric_records": len(frame),
                "latest_500_update_mean": float(
                    latest["loss_trailing_mean_500_updates"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
