#!/usr/bin/env python3
"""Strictly compare two FSDP2 implementation benchmark profiles."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

try:
    from scripts.benchmark.summarize_aistudio_benchmark import build_summary
except ModuleNotFoundError:  # Direct execution via scripts/benchmark/*.py.
    from summarize_aistudio_benchmark import build_summary


IMPLEMENTATION_SPECS = {
    "local_rearrange_v1": {
        "name": "local_rearrange_v1",
        "changed_path": "protenix/model/modules/primitives.py",
        "baseline_sha256": (
            "ef07426b3a3d0bdc694b1eabbee3dfafaf3dfe679cb45d976c44114e05d0f4b1"
        ),
        "candidate_sha256": (
            "987f3a803d8cf5d669617c6d0c874f4919426d26352963dcc3cf0121e6257181"
        ),
    }
}


def _diff(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                differences.append(
                    {
                        "path": child,
                        "baseline": left.get(key),
                        "candidate": right.get(key),
                    }
                )
            else:
                differences.extend(_diff(left[key], right[key], child))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = []
        for index in range(max(len(left), len(right))):
            child = f"{path}[{index}]"
            if index >= len(left) or index >= len(right):
                differences.append(
                    {
                        "path": child,
                        "baseline": left[index] if index < len(left) else None,
                        "candidate": right[index] if index < len(right) else None,
                    }
                )
            else:
                differences.extend(_diff(left[index], right[index], child))
        return differences
    if left != right:
        return [{"path": path, "baseline": left, "candidate": right}]
    return []


def _normalize_checkout_path(value: Any, checkout: Path) -> Any:
    if not isinstance(value, str) or not value:
        return value
    try:
        relative = Path(value).resolve().relative_to(checkout)
    except (OSError, ValueError):
        return value
    return str(Path("<CHECKOUT>") / relative)


def _normalized_command(command: Any) -> list[Any]:
    if not isinstance(command, list):
        return []
    normalized = list(command)
    for index, token in enumerate(normalized[:-1]):
        if token == "--run_name":
            normalized[index + 1] = "<RUN_NAME>"
        elif token == "--base_dir":
            normalized[index + 1] = "<BASE_DIR>"
    return normalized


def canonicalize_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    checkout = Path(str(manifest.get("cwd", "."))).resolve()
    raw_distributed = manifest.get("distributed")
    distributed = (
        copy.deepcopy(dict(raw_distributed))
        if isinstance(raw_distributed, Mapping)
        else {}
    )
    raw_environment = manifest.get("environment")
    environment = raw_environment if isinstance(raw_environment, Mapping) else {}
    raw_module_locations = environment.get("module_locations")
    module_location_mapping = (
        raw_module_locations if isinstance(raw_module_locations, Mapping) else {}
    )
    module_locations = {
        name: _normalize_checkout_path(path, checkout)
        for name, path in module_location_mapping.items()
    }
    raw_git = manifest.get("git")
    git = raw_git if isinstance(raw_git, Mapping) else {}
    return {
        "project": manifest.get("project"),
        "model_name": manifest.get("model_name"),
        "seed": manifest.get("seed"),
        "cwd": "<CHECKOUT>",
        "training": manifest.get("training"),
        "model": manifest.get("model"),
        "data": manifest.get("data"),
        "initial_checkpoint": manifest.get("initial_checkpoint"),
        "benchmark": manifest.get("benchmark"),
        "distributed": distributed,
        "environment": {
            key: environment.get(key)
            for key in (
                "hostname",
                "python_version",
                "python_executable",
                "torch_version",
                "cuda_available",
                "cuda_version",
                "cuda_visible_devices",
                "protenix_root_dir",
                "cc",
                "cxx",
                "torch_lib_dir",
                "ld_library_path",
                "torch_multiprocessing_sharing_strategy",
            )
        }
        | {"module_locations": module_locations},
        "git": {
            "commit": git.get("commit"),
            "branch": git.get("branch"),
            "dirty": git.get("dirty"),
        },
        "command": _normalized_command(manifest.get("command")),
    }


def canonicalize_runner(runner: Mapping[str, Any]) -> dict[str, Any]:
    """Keep every runner setting/workload field while dropping run-local metrics."""
    result = copy.deepcopy(dict(runner))
    for key in (
        "run_name",
        "run_dir",
        "started_at",
        "started_at_epoch",
        "completed_at",
        "completed_at_epoch",
        "elapsed_seconds",
        "seconds_per_step",
        "samples_per_sec",
        "tokens_per_sec",
        "atoms_per_sec",
        "slowest_rank",
        "rank_elapsed_imbalance",
        "max_allocated_bytes",
        "max_reserved_bytes",
        "phase_cuda_memory",
    ):
        result.pop(key, None)
    rank_records = []
    for raw_record in result.get("rank_records") or []:
        record = copy.deepcopy(dict(raw_record))
        for key in (
            "elapsed_seconds",
            "max_allocated_bytes",
            "max_reserved_bytes",
            "phase_cuda_memory",
        ):
            record.pop(key, None)
        rank_records.append(record)
    rank_records.sort(key=lambda record: _integer_or(record.get("rank"), -1))
    result["rank_records"] = rank_records
    return result


def _integer_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hardware_signature(summary: Mapping[str, Any]) -> dict[str, Any]:
    measured = (summary.get("nvml_device") or {}).get("measurement_window") or {}
    per_gpu = [
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
    ]
    per_gpu.sort(
        key=lambda record: (
            _integer_or(record.get("node_rank"), -1),
            _integer_or(record.get("gpu_index"), -1),
            str(record.get("gpu_uuid", "")),
        )
    )
    return {
        "nodes_expected": (summary.get("nodes") or {}).get("expected"),
        "nodes_observed": (summary.get("nodes") or {}).get("observed"),
        "gpu_count": measured.get("gpu_count"),
        "per_gpu": per_gpu,
    }


def validate_code_delta(
    baseline_mapping: Mapping[str, Any],
    candidate_mapping: Mapping[str, Any],
    spec: Mapping[str, str],
    *,
    prefix: str,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if set(baseline_mapping) != set(candidate_mapping):
        errors.append(
            {
                "path": f"{prefix}.file_set",
                "baseline": sorted(baseline_mapping),
                "candidate": sorted(candidate_mapping),
                "category": "code_evidence",
            }
        )
    changed = {
        path
        for path in set(baseline_mapping) | set(candidate_mapping)
        if baseline_mapping.get(path) != candidate_mapping.get(path)
    }
    expected_path = spec["changed_path"]
    if changed != {expected_path}:
        errors.append(
            {
                "path": f"{prefix}.changed_paths",
                "baseline": [expected_path],
                "candidate": sorted(changed),
                "category": "code_evidence",
            }
        )
    if baseline_mapping.get(expected_path) != spec["baseline_sha256"]:
        errors.append(
            {
                "path": f"{prefix}.{expected_path}.baseline_sha256",
                "baseline": spec["baseline_sha256"],
                "candidate": baseline_mapping.get(expected_path),
                "category": "code_evidence",
            }
        )
    if candidate_mapping.get(expected_path) != spec["candidate_sha256"]:
        errors.append(
            {
                "path": f"{prefix}.{expected_path}.candidate_sha256",
                "baseline": spec["candidate_sha256"],
                "candidate": candidate_mapping.get(expected_path),
                "category": "code_evidence",
            }
        )
    return errors


def _observed_code_delta(
    baseline_mapping: Mapping[str, Any],
    candidate_mapping: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "baseline_sha256": baseline_mapping.get(path),
            "candidate_sha256": candidate_mapping.get(path),
        }
        for path in sorted(set(baseline_mapping) | set(candidate_mapping))
        if baseline_mapping.get(path) != candidate_mapping.get(path)
    ]


def _evidence_signature(
    summary: Mapping[str, Any], evidence_name: str
) -> dict[str, Any]:
    evidence = (summary.get("node_evidence") or {}).get(evidence_name) or {}
    return {
        "entry_count": evidence.get("entry_count"),
        "mapping_sha256": evidence.get("mapping_sha256"),
        "mapping": evidence.get("mapping"),
    }


def _metric(runner: Mapping[str, Any], key: str) -> float | None:
    value = runner.get(key)
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    ):
        return float(value)
    return None


def absolute_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    raw_runner = summary.get("runner")
    runner = raw_runner if isinstance(raw_runner, Mapping) else {}
    raw_nvml = summary.get("nvml_device")
    nvml_device = raw_nvml if isinstance(raw_nvml, Mapping) else {}
    raw_window = nvml_device.get("measurement_window")
    nvml = raw_window if isinstance(raw_window, Mapping) else {}
    return {
        "seconds_per_step": _metric(runner, "seconds_per_step"),
        "samples_per_sec": _metric(runner, "samples_per_sec"),
        "tokens_per_sec": _metric(runner, "tokens_per_sec"),
        "atoms_per_sec": _metric(runner, "atoms_per_sec"),
        "pytorch_max_allocated_bytes": runner.get("max_allocated_bytes"),
        "pytorch_max_reserved_bytes": runner.get("max_reserved_bytes"),
        "nvml_hottest_gpu_peak_mib": nvml.get("hottest_gpu_peak_mib"),
        "phase_cuda_memory": runner.get("phase_cuda_memory"),
    }


def _positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def relative_metrics(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any] | None:
    baseline_metrics = absolute_metrics(baseline)
    candidate_metrics = absolute_metrics(candidate)
    required_memory = (
        baseline_metrics["pytorch_max_allocated_bytes"],
        candidate_metrics["pytorch_max_allocated_bytes"],
        baseline_metrics["pytorch_max_reserved_bytes"],
        candidate_metrics["pytorch_max_reserved_bytes"],
        baseline_metrics["nvml_hottest_gpu_peak_mib"],
        candidate_metrics["nvml_hottest_gpu_peak_mib"],
    )
    if not all(_positive(value) for value in required_memory):
        return None
    (
        base_alloc,
        cand_alloc,
        base_reserved,
        cand_reserved,
        base_nvml,
        cand_nvml,
    ) = required_memory
    phase_diagnostic = bool(
        baseline_metrics.get("phase_cuda_memory")
        or candidate_metrics.get("phase_cuda_memory")
    )
    result: dict[str, Any] = {
        "timing_comparable": not phase_diagnostic,
        "allocated_memory_saving_percent": (base_alloc - cand_alloc)
        / base_alloc
        * 100.0,
        "allocated_memory_saving_gib": (base_alloc - cand_alloc) / (1024**3),
        "reserved_memory_saving_percent": (base_reserved - cand_reserved)
        / base_reserved
        * 100.0,
        "reserved_memory_saving_gib": (base_reserved - cand_reserved) / (1024**3),
        "nvml_memory_saving_percent": (base_nvml - cand_nvml)
        / base_nvml
        * 100.0,
        "nvml_memory_saving_gib": (base_nvml - cand_nvml) / 1024.0,
    }
    if not phase_diagnostic:
        base_step = baseline_metrics["seconds_per_step"]
        cand_step = candidate_metrics["seconds_per_step"]
        base_tokens = baseline_metrics["tokens_per_sec"]
        cand_tokens = candidate_metrics["tokens_per_sec"]
        if not all(_positive(v) for v in (base_step, cand_step, base_tokens, cand_tokens)):
            return None
        result.update(
            {
                "step_time_change_percent": (cand_step / base_step - 1.0) * 100.0,
                "speedup_ratio": base_step / cand_step,
                "token_throughput_ratio": cand_tokens / base_tokens,
                "token_throughput_change_percent": (
                    cand_tokens / base_tokens - 1.0
                )
                * 100.0,
            }
        )
    return result


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping_sha256(mapping: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(mapping), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _shape_errors(label: str, summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Reject sparse or internally inconsistent self-declared summaries."""
    errors: list[dict[str, Any]] = []
    runner = summary.get("runner")
    manifest = summary.get("run_manifest")
    nodes = summary.get("nodes")
    raw_nvml_device = summary.get("nvml_device")
    nvml_device = (
        raw_nvml_device if isinstance(raw_nvml_device, Mapping) else {}
    )
    measured = nvml_device.get("measurement_window")
    node_evidence = summary.get("node_evidence")

    def require(condition: bool, path: str, value: Any) -> None:
        if not condition:
            errors.append(
                {"path": f"{label}.{path}", label: value, "category": "shape"}
            )

    require(
        summary.get("schema_version") == 1,
        "schema_version",
        summary.get("schema_version"),
    )
    require(
        isinstance(summary.get("benchmark_id"), str)
        and bool(summary.get("benchmark_id")),
        "benchmark_id",
        summary.get("benchmark_id"),
    )
    require(summary.get("status") == "passed", "status", summary.get("status"))
    require(
        not summary.get("validation_errors"),
        "validation_errors",
        summary.get("validation_errors"),
    )
    require(isinstance(runner, Mapping), "runner", type(runner).__name__)
    require(isinstance(manifest, Mapping), "run_manifest", type(manifest).__name__)
    require(isinstance(nodes, Mapping), "nodes", type(nodes).__name__)
    require(
        isinstance(measured, Mapping),
        "nvml_device.measurement_window",
        type(measured).__name__,
    )
    require(
        isinstance(node_evidence, Mapping),
        "node_evidence",
        type(node_evidence).__name__,
    )

    world_size: int | None = None
    if isinstance(runner, Mapping):
        required_runner_keys = {
            "schema_version",
            "benchmark_variant",
            "benchmark_workload_profile",
            "distributed_strategy",
            "resolved_fsdp_mesh",
            "world_size",
            "rank_records",
            "seconds_per_step",
            "tokens_per_sec",
            "max_allocated_bytes",
            "max_reserved_bytes",
        }
        require(
            required_runner_keys.issubset(runner),
            "runner.required_keys",
            sorted(required_runner_keys - set(runner)),
        )
        require(
            runner.get("schema_version") == 1,
            "runner.schema_version",
            runner.get("schema_version"),
        )
        require(
            runner.get("benchmark_variant") == "fsdp2_full",
            "runner.benchmark_variant",
            runner.get("benchmark_variant"),
        )
        require(
            runner.get("distributed_strategy") == "fsdp2",
            "runner.distributed_strategy",
            runner.get("distributed_strategy"),
        )
        require(
            bool(runner.get("resolved_fsdp_mesh")),
            "runner.resolved_fsdp_mesh",
            runner.get("resolved_fsdp_mesh"),
        )
        raw_world_size = runner.get("world_size")
        if (
            isinstance(raw_world_size, int)
            and not isinstance(raw_world_size, bool)
            and raw_world_size > 0
        ):
            world_size = raw_world_size
        require(world_size is not None, "runner.world_size", raw_world_size)
        rank_records = runner.get("rank_records")
        require(
            isinstance(rank_records, list)
            and world_size is not None
            and len(rank_records) == world_size,
            "runner.rank_records",
            rank_records,
        )
        if isinstance(rank_records, list):
            ranks = []
            for index, record in enumerate(rank_records):
                require(
                    isinstance(record, Mapping),
                    f"runner.rank_records[{index}]",
                    record,
                )
                if not isinstance(record, Mapping):
                    continue
                rank = record.get("rank")
                ranks.append(
                    rank
                    if isinstance(rank, int) and not isinstance(rank, bool)
                    else None
                )
                require(
                    _is_sha256(record.get("workload_sha256")),
                    f"runner.rank_records[{index}].workload_sha256",
                    record.get("workload_sha256"),
                )
            if world_size is not None:
                require(
                    set(ranks) == set(range(world_size)),
                    "runner.rank_records.ranks",
                    ranks,
                )

    if isinstance(manifest, Mapping):
        required_manifest_keys = {
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
        }
        require(
            required_manifest_keys.issubset(manifest),
            "run_manifest.required_keys",
            sorted(required_manifest_keys - set(manifest)),
        )
        for key in ("training", "model", "data", "distributed", "environment", "git"):
            require(
                isinstance(manifest.get(key), Mapping),
                f"run_manifest.{key}",
                manifest.get(key),
            )
        require(
            isinstance(manifest.get("cwd"), str) and bool(manifest.get("cwd")),
            "run_manifest.cwd",
            manifest.get("cwd"),
        )
        require(
            isinstance(manifest.get("command"), list)
            and bool(manifest.get("command")),
            "run_manifest.command",
            manifest.get("command"),
        )
        manifest_code = manifest.get("code_sha256")
        require(
            isinstance(manifest_code, Mapping)
            and bool(manifest_code)
            and all(_is_sha256(value) for value in manifest_code.values()),
            "run_manifest.code_sha256",
            manifest_code,
        )

    if isinstance(nodes, Mapping):
        expected_nodes = nodes.get("expected")
        observed_nodes = nodes.get("observed")
        require(
            isinstance(expected_nodes, int)
            and not isinstance(expected_nodes, bool)
            and expected_nodes > 0
            and observed_nodes == expected_nodes,
            "nodes",
            {"expected": expected_nodes, "observed": observed_nodes},
        )

    if isinstance(measured, Mapping) and world_size is not None:
        per_gpu = measured.get("per_gpu")
        require(
            measured.get("gpu_count") == world_size
            and isinstance(per_gpu, list)
            and len(per_gpu) == world_size,
            "nvml_device.measurement_window.gpu_inventory",
            {
                "gpu_count": measured.get("gpu_count"),
                "per_gpu": per_gpu,
            },
        )
        if isinstance(per_gpu, list):
            for index, record in enumerate(per_gpu):
                require(
                    isinstance(record, Mapping)
                    and isinstance(record.get("gpu_uuid"), str)
                    and bool(record.get("gpu_uuid"))
                    and isinstance(record.get("gpu_name"), str)
                    and bool(record.get("gpu_name")),
                    f"nvml_device.measurement_window.per_gpu[{index}]",
                    record,
                )

    if isinstance(node_evidence, Mapping):
        for name in ("code_tree", "artifacts"):
            evidence = node_evidence.get(name)
            mapping = evidence.get("mapping") if isinstance(evidence, Mapping) else None
            valid_mapping = (
                isinstance(mapping, Mapping)
                and bool(mapping)
                and all(_is_sha256(value) for value in mapping.values())
            )
            require(valid_mapping, f"node_evidence.{name}.mapping", mapping)
            if valid_mapping:
                require(
                    evidence.get("entry_count") == len(mapping),
                    f"node_evidence.{name}.entry_count",
                    evidence.get("entry_count"),
                )
                require(
                    evidence.get("mapping_sha256") == _mapping_sha256(mapping),
                    f"node_evidence.{name}.mapping_sha256",
                    evidence.get("mapping_sha256"),
                )
    return errors


def compare_implementation_summaries(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    spec: Mapping[str, str],
) -> dict[str, Any]:
    required_spec_keys = {"name", "changed_path", "baseline_sha256", "candidate_sha256"}
    if not required_spec_keys.issubset(spec) or not all(
        _is_sha256(spec.get(key))
        for key in ("baseline_sha256", "candidate_sha256")
    ):
        raise ValueError("invalid implementation comparison spec")

    mismatches = _shape_errors("baseline", baseline) + _shape_errors(
        "candidate", candidate
    )
    comparable: dict[str, tuple[Any, Any]] = {}
    observed_code_delta: dict[str, Any] = {
        "code_tree": [],
        "manifest_code_sha256": [],
    }
    if not mismatches:
        comparable = {
            "manifest": (
                canonicalize_manifest(baseline["run_manifest"]),
                canonicalize_manifest(candidate["run_manifest"]),
            ),
            "runner": (
                canonicalize_runner(baseline["runner"]),
                canonicalize_runner(candidate["runner"]),
            ),
            "hardware": (
                _hardware_signature(baseline),
                _hardware_signature(candidate),
            ),
            "artifacts": (
                _evidence_signature(baseline, "artifacts"),
                _evidence_signature(candidate, "artifacts"),
            ),
        }
        for category, (left, right) in comparable.items():
            for mismatch in _diff(left, right, category):
                mismatch["category"] = category
                mismatches.append(mismatch)

        baseline_code_tree = (
            baseline["node_evidence"].get("code_tree") or {}
        ).get("mapping") or {}
        candidate_code_tree = (
            candidate["node_evidence"].get("code_tree") or {}
        ).get("mapping") or {}
        observed_code_delta["code_tree"] = _observed_code_delta(
            baseline_code_tree, candidate_code_tree
        )
        mismatches.extend(
            validate_code_delta(
                baseline_code_tree,
                candidate_code_tree,
                spec,
                prefix="node_evidence.code_tree",
            )
        )
        baseline_manifest_code = baseline["run_manifest"].get("code_sha256") or {}
        candidate_manifest_code = candidate["run_manifest"].get("code_sha256") or {}
        observed_code_delta["manifest_code_sha256"] = _observed_code_delta(
            baseline_manifest_code, candidate_manifest_code
        )
        mismatches.extend(
            validate_code_delta(
                baseline_manifest_code,
                candidate_manifest_code,
                spec,
                prefix="manifest.code_sha256",
            )
        )
        for label, manifest_code, tree_code in (
            ("baseline", baseline_manifest_code, baseline_code_tree),
            ("candidate", candidate_manifest_code, candidate_code_tree),
        ):
            for path, digest in manifest_code.items():
                if tree_code.get(path) != digest:
                    mismatches.append(
                        {
                            "path": f"{label}.manifest_vs_code_tree.{path}",
                            "baseline": digest if label == "baseline" else None,
                            "candidate": digest if label == "candidate" else None,
                            "category": "code_evidence",
                        }
                    )

    absolute = {
        "baseline": absolute_metrics(baseline),
        "candidate": absolute_metrics(candidate),
    }
    relative = None if mismatches else relative_metrics(baseline, candidate)
    if not mismatches and relative is None:
        mismatches.append(
            {
                "path": "performance.required_metrics",
                "baseline": absolute["baseline"],
                "candidate": absolute["candidate"],
                "category": "performance",
            }
        )
    signature_payload = {
        "baseline_id": baseline.get("benchmark_id"),
        "candidate_id": candidate.get("benchmark_id"),
        "spec": dict(spec),
        "comparable_evidence": comparable,
        "observed_code_delta": observed_code_delta,
        "absolute_metrics": absolute,
        "mismatches": mismatches,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "comparison_type": "fsdp2_implementation_ab",
        "status": "passed" if not mismatches else "failed",
        "baseline_benchmark_id": baseline.get("benchmark_id"),
        "candidate_benchmark_id": candidate.get("benchmark_id"),
        "implementation_spec": dict(spec),
        "observed_code_delta": observed_code_delta,
        "comparison_signature_sha256": signature,
        "mismatches": mismatches,
        "absolute_metrics": absolute,
        "relative_metrics": relative if not mismatches else None,
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# FSDP2 implementation A/B benchmark",
        "",
        f"- Status: **{result['status']}**",
        f"- Baseline: {result.get('baseline_benchmark_id')}",
        f"- Candidate: {result.get('candidate_benchmark_id')}",
        f"- Signature: `{result['comparison_signature_sha256']}`",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(
            {
                "absolute": result["absolute_metrics"],
                "relative": result.get("relative_metrics"),
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
    ]
    if result["mismatches"]:
        lines.extend(["", "## Mismatches", ""])
        lines.extend(
            f"- `{item['path']}`: baseline={item.get('baseline')!r}, candidate={item.get('candidate')!r}"
            for item in result["mismatches"]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument(
        "--experiment", choices=sorted(IMPLEMENTATION_SPECS), required=True
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    for profile in (args.baseline_profile, args.candidate_profile):
        if not (profile / "summary.json").is_file():
            raise SystemExit(f"Missing single-run summary artifact: {profile / 'summary.json'}")
    result = compare_implementation_summaries(
        build_summary(args.baseline_profile),
        build_summary(args.candidate_profile),
        spec=IMPLEMENTATION_SPECS[args.experiment],
    )
    output_json = args.output_json or args.candidate_profile / "implementation_comparison.json"
    output_md = args.output_md or args.candidate_profile / "implementation_comparison.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
