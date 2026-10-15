#!/usr/bin/env python3
"""Strictly compare one passed DDP profile with one passed FSDP2 profile."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from scripts.benchmark.summarize_aistudio_benchmark import build_summary
except ModuleNotFoundError:  # Direct execution via scripts/benchmark/*.py.
    from summarize_aistudio_benchmark import build_summary


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalized_command(command: Any) -> list[Any]:
    if not isinstance(command, list):
        return []
    normalized = list(command)
    ignored_values = {
        "--distributed_strategy": "<STRATEGY>",
        "--run_name": "<RUN_NAME>",
        "--base_dir": "<BASE_DIR>",
    }
    for index, token in enumerate(normalized[:-1]):
        if token in ignored_values:
            normalized[index + 1] = ignored_values[token]
    return normalized


def _manifest_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    benchmark = copy.deepcopy(manifest.get("benchmark") or {})
    benchmark.pop("variant", None)
    distributed = copy.deepcopy(manifest.get("distributed") or {})
    for key in ("rank", "node_rank", "strategy"):
        distributed.pop(key, None)
    environment = manifest.get("environment") or {}
    return {
        "project": manifest.get("project"),
        "model_name": manifest.get("model_name"),
        "seed": manifest.get("seed"),
        "cwd": manifest.get("cwd"),
        "training": manifest.get("training"),
        "model": manifest.get("model"),
        "data": manifest.get("data"),
        "initial_checkpoint": manifest.get("initial_checkpoint"),
        "benchmark": benchmark,
        "distributed": distributed,
        "environment": {
            key: environment.get(key)
            for key in (
                "python_version",
                "python_executable",
                "torch_version",
                "cuda_version",
                "protenix_root_dir",
                "module_locations",
                "cc",
                "cxx",
                "torch_lib_dir",
                "torch_multiprocessing_sharing_strategy",
            )
        },
        "git": manifest.get("git"),
        "code_sha256": manifest.get("code_sha256"),
        "command": _normalized_command(manifest.get("command")),
    }


def _rank_workload_signature(runner: dict[str, Any]) -> list[dict[str, Any]]:
    keys = (
        "rank",
        "local_rank",
        "node_rank",
        "sample_count",
        "token_count",
        "atom_count",
        "min_token_count",
        "max_token_count",
        "workload_sha256",
        "gpu_name",
        "loss_nonfinite_count",
        "gradient_nonfinite_count",
        "optimizer_skipped_count",
    )
    return [
        {key: record.get(key) for key in keys}
        for record in sorted(
            runner.get("rank_records") or [], key=lambda record: record.get("rank", -1)
        )
    ]


def _runner_signature(runner: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "benchmark_dataset_name",
        "train_crop_size",
        "world_size",
        "warmup_steps",
        "measure_steps",
        "total_samples",
        "total_tokens",
        "total_atoms",
        "topology",
        "model_parameters",
        "ema",
        "skip_final_checkpoint",
        "load_params_only",
        "initial_checkpoint",
        "optimizer_sharding",
        "fsdp",
        "iters_to_accumulate",
        "train_sampler_replacement",
        "run_sample_coverage",
        "max_loss_nonfinite_count_per_rank",
        "max_gradient_nonfinite_count_per_rank",
        "max_optimizer_skipped_count_per_rank",
    )
    signature = {key: runner.get(key) for key in keys}
    signature["rank_workloads"] = _rank_workload_signature(runner)
    return signature


def _hardware_signature(summary: dict[str, Any]) -> dict[str, Any]:
    measured = (summary.get("nvml_device") or {}).get("measurement_window") or {}
    gpu_inventory = sorted(
        (
            {
                key: record.get(key)
                for key in (
                    "node_rank",
                    "hostname",
                    "gpu_index",
                    "gpu_uuid",
                    "gpu_name",
                    "driver_version",
                    "memory_total_mib",
                )
            }
            for record in measured.get("per_gpu") or []
        ),
        key=lambda record: (
            record.get("node_rank", -1),
            record.get("gpu_index", -1),
            str(record.get("gpu_uuid", "")),
        ),
    )
    return {
        "nodes_expected": (summary.get("nodes") or {}).get("expected"),
        "nodes_observed": (summary.get("nodes") or {}).get("observed"),
        "measured_gpu_count": measured.get("gpu_count"),
        "gpu_inventory": gpu_inventory,
    }


def _node_evidence_signature(summary: dict[str, Any]) -> dict[str, Any]:
    evidence = summary.get("node_evidence") or {}
    code_tree = evidence.get("code_tree") or {}
    artifacts = evidence.get("artifacts") or {}
    return {
        "code_tree_mapping_sha256": code_tree.get("mapping_sha256"),
        "code_tree_entry_count": code_tree.get("entry_count"),
        "code_tree_mapping": code_tree.get("mapping"),
        "artifact_mapping_sha256": artifacts.get("mapping_sha256"),
        "artifact_mapping": artifacts.get("mapping"),
    }


def _diff(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            child_path = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                differences.append(
                    {
                        "path": child_path,
                        "ddp": left.get(key),
                        "fsdp2": right.get(key),
                    }
                )
            else:
                differences.extend(_diff(left[key], right[key], child_path))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = []
        for index in range(max(len(left), len(right))):
            child_path = f"{path}[{index}]"
            if index >= len(left) or index >= len(right):
                differences.append(
                    {
                        "path": child_path,
                        "ddp": left[index] if index < len(left) else None,
                        "fsdp2": right[index] if index < len(right) else None,
                    }
                )
            else:
                differences.extend(_diff(left[index], right[index], child_path))
        return differences
    if left != right:
        return [{"path": path, "ddp": left, "fsdp2": right}]
    return []


def _metric(runner: dict[str, Any], key: str) -> float | None:
    value = runner.get(key)
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value > 0
    ):
        return float(value)
    return None


def _summary_shape_errors(
    label: str,
    summary: dict[str, Any],
    *,
    expected_variant: str,
    expected_strategy: str,
) -> list[dict[str, Any]]:
    """Reject sparse/self-declared summaries before comparing missing values."""
    errors: list[dict[str, Any]] = []

    def require(condition: bool, path: str, observed: Any) -> None:
        if not condition:
            errors.append(
                {
                    "path": f"{label}.{path}",
                    "ddp": observed if label == "ddp" else None,
                    "fsdp2": observed if label == "fsdp2" else None,
                    "category": "single_run_shape",
                }
            )

    runner = summary.get("runner")
    manifest = summary.get("run_manifest")
    nodes = summary.get("nodes")
    measured = (summary.get("nvml_device") or {}).get("measurement_window")
    node_evidence = summary.get("node_evidence")
    require(summary.get("schema_version") == 1, "schema_version", summary.get("schema_version"))
    require(
        isinstance(summary.get("benchmark_id"), str)
        and bool(summary.get("benchmark_id")),
        "benchmark_id",
        summary.get("benchmark_id"),
    )
    require(isinstance(runner, dict), "runner", type(runner).__name__)
    require(isinstance(manifest, dict), "run_manifest", type(manifest).__name__)
    require(isinstance(nodes, dict), "nodes", type(nodes).__name__)
    require(isinstance(measured, dict), "nvml_device.measurement_window", type(measured).__name__)
    require(isinstance(node_evidence, dict), "node_evidence", type(node_evidence).__name__)
    if isinstance(node_evidence, dict):
        for evidence_key in ("code_tree", "artifacts"):
            evidence_value = node_evidence.get(evidence_key)
            require(
                isinstance(evidence_value, dict)
                and isinstance(evidence_value.get("mapping_sha256"), str)
                and len(evidence_value["mapping_sha256"]) == 64
                and isinstance(evidence_value.get("mapping"), dict)
                and bool(evidence_value["mapping"]),
                f"node_evidence.{evidence_key}",
                evidence_value,
            )
    if not isinstance(runner, dict) or not isinstance(manifest, dict):
        return errors

    required_runner_keys = {
        "schema_version",
        "benchmark_variant",
        "distributed_strategy",
        "world_size",
        "warmup_steps",
        "measure_steps",
        "rank_records",
        "model_parameters",
        "ema",
        "initial_checkpoint",
        "seconds_per_step",
        "tokens_per_sec",
        "max_allocated_bytes",
    }
    require(
        required_runner_keys.issubset(runner),
        "runner.required_keys",
        sorted(required_runner_keys - set(runner)),
    )
    require(runner.get("schema_version") == 1, "runner.schema_version", runner.get("schema_version"))
    require(
        runner.get("benchmark_variant") == expected_variant,
        "runner.benchmark_variant",
        runner.get("benchmark_variant"),
    )
    require(
        runner.get("distributed_strategy") == expected_strategy,
        "runner.distributed_strategy",
        runner.get("distributed_strategy"),
    )
    world_size = runner.get("world_size")
    rank_records = runner.get("rank_records")
    require(
        isinstance(world_size, int) and not isinstance(world_size, bool) and world_size > 0,
        "runner.world_size",
        world_size,
    )
    require(
        isinstance(rank_records, list)
        and isinstance(world_size, int)
        and len(rank_records) == world_size,
        "runner.rank_records",
        len(rank_records) if isinstance(rank_records, list) else None,
    )
    if isinstance(rank_records, list):
        for index, record in enumerate(rank_records):
            require(
                isinstance(record, dict)
                and isinstance(record.get("workload_sha256"), str)
                and len(record["workload_sha256"]) == 64,
                f"runner.rank_records[{index}].workload_sha256",
                record.get("workload_sha256") if isinstance(record, dict) else None,
            )
    for key in (
        "project",
        "model_name",
        "seed",
        "cwd",
        "training",
        "model",
        "data",
        "initial_checkpoint",
        "benchmark",
        "distributed",
        "environment",
        "git",
        "code_sha256",
        "command",
    ):
        require(key in manifest and manifest.get(key) is not None, f"run_manifest.{key}", manifest.get(key))
    require(
        isinstance(manifest.get("code_sha256"), dict)
        and bool(manifest.get("code_sha256")),
        "run_manifest.code_sha256",
        manifest.get("code_sha256"),
    )
    return errors


def _absolute_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    runner = summary.get("runner") or {}
    measured_nvml = (summary.get("nvml_device") or {}).get("measurement_window") or {}
    return {
        "seconds_per_step": _metric(runner, "seconds_per_step"),
        "samples_per_sec": _metric(runner, "samples_per_sec"),
        "tokens_per_sec": _metric(runner, "tokens_per_sec"),
        "atoms_per_sec": _metric(runner, "atoms_per_sec"),
        "pytorch_max_allocated_bytes": runner.get("max_allocated_bytes"),
        "pytorch_max_reserved_bytes": runner.get("max_reserved_bytes"),
        "nvml_hottest_gpu_peak_mib": measured_nvml.get("hottest_gpu_peak_mib"),
        "nvml_max_time_bucket_sum_mib": measured_nvml.get(
            "max_time_bucket_sum_mib"
        ),
        "rank_elapsed_imbalance": runner.get("rank_elapsed_imbalance"),
        "launcher_wall_seconds": (summary.get("nodes") or {}).get(
            "max_launcher_wall_seconds"
        ),
    }


def compare_summaries(
    ddp_summary: dict[str, Any], fsdp_summary: dict[str, Any]
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for label, summary, variant, strategy in (
        ("ddp", ddp_summary, "ddp_full", "ddp"),
        ("fsdp2", fsdp_summary, "fsdp2_full", "fsdp2"),
    ):
        mismatches.extend(
            _summary_shape_errors(
                label,
                summary,
                expected_variant=variant,
                expected_strategy=strategy,
            )
        )
        runner = summary.get("runner") or {}
        if summary.get("status") != "passed" or summary.get("validation_errors"):
            mismatches.append(
                {
                    "path": f"{label}.single_run_validation",
                    "ddp": ddp_summary.get("validation_errors") if label == "ddp" else None,
                    "fsdp2": fsdp_summary.get("validation_errors") if label == "fsdp2" else None,
                }
            )
        if runner.get("benchmark_variant") != variant:
            mismatches.append(
                {
                    "path": f"{label}.benchmark_variant",
                    "ddp": runner.get("benchmark_variant") if label == "ddp" else None,
                    "fsdp2": runner.get("benchmark_variant") if label == "fsdp2" else None,
                }
            )
        if runner.get("distributed_strategy") != strategy:
            mismatches.append(
                {
                    "path": f"{label}.distributed_strategy",
                    "ddp": runner.get("distributed_strategy") if label == "ddp" else None,
                    "fsdp2": runner.get("distributed_strategy") if label == "fsdp2" else None,
                }
            )
    if (ddp_summary.get("runner") or {}).get("resolved_fsdp_mesh") is not None:
        mismatches.append(
            {"path": "ddp.resolved_fsdp_mesh", "ddp": "non-null", "fsdp2": None}
        )
    if not (fsdp_summary.get("runner") or {}).get("resolved_fsdp_mesh"):
        mismatches.append(
            {"path": "fsdp2.resolved_fsdp_mesh", "ddp": None, "fsdp2": None}
        )

    comparable = {
        "manifest": (
            _manifest_signature(ddp_summary.get("run_manifest") or {}),
            _manifest_signature(fsdp_summary.get("run_manifest") or {}),
        ),
        "runner": (
            _runner_signature(ddp_summary.get("runner") or {}),
            _runner_signature(fsdp_summary.get("runner") or {}),
        ),
        "hardware": (
            _hardware_signature(ddp_summary),
            _hardware_signature(fsdp_summary),
        ),
        "node_evidence": (
            _node_evidence_signature(ddp_summary),
            _node_evidence_signature(fsdp_summary),
        ),
    }
    for category, (ddp_value, fsdp_value) in comparable.items():
        for mismatch in _diff(ddp_value, fsdp_value, category):
            mismatch["category"] = category
            mismatches.append(mismatch)

    canonical = {
        "ddp_benchmark_id": ddp_summary.get("benchmark_id"),
        "fsdp2_benchmark_id": fsdp_summary.get("benchmark_id"),
        "ddp": {key: values[0] for key, values in comparable.items()},
        "fsdp2": {key: values[1] for key, values in comparable.items()},
    }
    comparison_signature = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    ddp_metrics = _absolute_metrics(ddp_summary)
    fsdp_metrics = _absolute_metrics(fsdp_summary)
    relative: dict[str, Any] | None = None
    if not mismatches:
        ddp_step = ddp_metrics["seconds_per_step"]
        fsdp_step = fsdp_metrics["seconds_per_step"]
        ddp_tokens = ddp_metrics["tokens_per_sec"]
        fsdp_tokens = fsdp_metrics["tokens_per_sec"]
        ddp_memory = ddp_metrics["pytorch_max_allocated_bytes"]
        fsdp_memory = fsdp_metrics["pytorch_max_allocated_bytes"]
        if all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and value > 0
            for value in (ddp_step, fsdp_step, ddp_tokens, fsdp_tokens, ddp_memory, fsdp_memory)
        ):
            relative = {
                "step_time_overhead_percent": (fsdp_step / ddp_step - 1.0) * 100.0,
                "token_throughput_ratio": fsdp_tokens / ddp_tokens,
                "allocated_memory_saving_percent": (
                    (ddp_memory - fsdp_memory) / ddp_memory * 100.0
                ),
                "allocated_memory_saving_gib": (
                    (ddp_memory - fsdp_memory) / (1024**3)
                ),
            }
        else:
            mismatches.append(
                {
                    "path": "performance.required_metrics",
                    "ddp": ddp_metrics,
                    "fsdp2": fsdp_metrics,
                    "category": "performance",
                }
            )

    return {
        "schema_version": 1,
        "paired_status": "passed" if not mismatches else "failed",
        "ddp_benchmark_id": ddp_summary.get("benchmark_id"),
        "fsdp2_benchmark_id": fsdp_summary.get("benchmark_id"),
        "comparison_signature_sha256": comparison_signature,
        "mismatches": mismatches,
        "absolute_metrics": {"ddp": ddp_metrics, "fsdp2": fsdp_metrics},
        "relative_metrics": relative if not mismatches else None,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Paired DDP/FSDP2 benchmark comparison",
        "",
        f"- Status: **{result['paired_status']}**",
        f"- DDP: {result.get('ddp_benchmark_id')}",
        f"- FSDP2: {result.get('fsdp2_benchmark_id')}",
        f"- Comparison signature: `{result['comparison_signature_sha256']}`",
        "",
        "## Absolute metrics",
        "",
        "```json",
        json.dumps(result["absolute_metrics"], indent=2, sort_keys=True),
        "```",
    ]
    if result.get("relative_metrics") is not None:
        lines.extend(
            [
                "",
                "## Relative metrics (DDP baseline)",
                "",
                "```json",
                json.dumps(result["relative_metrics"], indent=2, sort_keys=True),
                "```",
            ]
        )
    if result["mismatches"]:
        lines.extend(["", "## Mismatches", ""])
        lines.extend(
            f"- `{item['path']}`: DDP={item.get('ddp')!r}, FSDP2={item.get('fsdp2')!r}"
            for item in result["mismatches"]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddp-profile", type=Path, required=True)
    parser.add_argument("--fsdp2-profile", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    ddp_path = args.ddp_profile / "summary.json"
    fsdp_path = args.fsdp2_profile / "summary.json"
    # Rebuild both summaries from raw profile artifacts. This prevents a stale
    # or hand-edited summary.json from self-declaring validation success.
    if not ddp_path.is_file() or not fsdp_path.is_file():
        missing = [str(path) for path in (ddp_path, fsdp_path) if not path.is_file()]
        raise SystemExit("Missing single-run summary artifact(s): " + ", ".join(missing))
    result = compare_summaries(
        build_summary(args.ddp_profile),
        build_summary(args.fsdp2_profile),
    )
    output_json = args.output_json or args.fsdp2_profile / "paired_comparison.json"
    output_md = args.output_md or args.fsdp2_profile / "paired_comparison.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_md.write_text(render_markdown(result), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    if result["paired_status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
