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

"""Validate one completed locked AF3-like PXM evaluation run.

The queue calls this program before publishing a done marker.  It deliberately
reads the structured runner artifacts instead of treating a zero exit status or
an existing output directory as proof of a complete evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPECTED_TEST_NAME = "recentPDB_1536_sample384_0925"
EXPECTED_EMA_SUFFIX = "ema0.999_"
REQUIRED_CATEGORIES = (
    "intra_prot",
    "prot_prot",
    "ligand_prot",
    "intra_dna",
    "dna_prot",
    "intra_rna",
    "rna_prot",
)
REQUIRED_METRICS = ("top1_pct", "best5_pct", "top1_best5_pct")
REQUIRED_AGGREGATES = ("avg", "median", "pct90", "min", "max")
FORBIDDEN_LOG_TEXT = (
    "traceback",
    "cuda out of memory",
    "nan/inf",
    "template/msa parser error",
    "missing target",
    "random replacement",
)
VALIDATION_CONTRACT = "af3_pxm_full_combined_formal_every600_v1"


class ResultValidationError(ValueError):
    """Raised when an evaluation artifact violates the locked result contract."""


def _read_json(path: Path, description: str) -> Mapping[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ResultValidationError(f"{description} is missing or empty: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultValidationError(f"{description} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ResultValidationError(f"{description} must contain a JSON object: {path}")
    return value


def _iter_jsonl(path: Path, description: str) -> Iterable[tuple[int, Mapping[str, Any]]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ResultValidationError(f"{description} is missing or empty: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ResultValidationError(
                        f"{description} has a blank record at line {line_number}: {path}"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ResultValidationError(
                        f"{description} has invalid JSON at line {line_number}: {path}: {exc}"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise ResultValidationError(
                        f"{description} record {line_number} must be a JSON object: {path}"
                    )
                yield line_number, value
    except OSError as exc:
        raise ResultValidationError(f"failed to read {description}: {path}: {exc}") from exc


def _manifest_targets(path: Path) -> list[str]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ResultValidationError(f"evaluation target manifest is missing or empty: {path}")
    targets = [
        line.strip().split()[0].lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(targets) != len(set(targets)):
        raise ResultValidationError(f"evaluation target manifest contains duplicate IDs: {path}")
    return targets


def _single_string(value: Any, description: str) -> str:
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ResultValidationError(f"{description} must resolve to one string; got {value!r}")
        value = value[0]
    if not isinstance(value, str) or not value.strip():
        raise ResultValidationError(f"{description} must be a non-empty string; got {value!r}")
    return value.strip()


def _finite_number(value: Any, description: str) -> float:
    if isinstance(value, bool):
        raise ResultValidationError(f"{description} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ResultValidationError(f"{description} must be numeric; got {value!r}") from exc
    if not math.isfinite(result):
        raise ResultValidationError(f"{description} is not finite: {value!r}")
    return result


def _scan_log(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ResultValidationError(f"evaluator log is missing or empty: {path}")
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                folded = line.casefold()
                for forbidden in FORBIDDEN_LOG_TEXT:
                    if forbidden in folded:
                        raise ResultValidationError(
                            f"evaluator log contains forbidden text {forbidden!r} "
                            f"at line {line_number}: {path}"
                        )
    except OSError as exc:
        raise ResultValidationError(f"failed to read evaluator log: {path}: {exc}") from exc


def _validate_checkpoint_identity(checkpoint_path: Path, completed_updates: int) -> int:
    match = re.fullmatch(r"(\d+)_ema_0\.999\.pt", checkpoint_path.name)
    if match is None:
        raise ResultValidationError(
            "checkpoint_path must end in <zero_based_step>_ema_0.999.pt; "
            f"got {checkpoint_path}"
        )
    checkpoint_step = int(match.group(1))
    if checkpoint_step + 1 != completed_updates:
        raise ResultValidationError(
            f"checkpoint/update mismatch: step={checkpoint_step}, "
            f"completed_updates={completed_updates}"
        )
    return checkpoint_step


def validate_run(
    *,
    run_dir: Path,
    log_path: Path,
    pdb_list: Path,
    checkpoint_path: Path,
    completed_updates: int,
    scope: str,
    subset: str,
    mode: str,
) -> dict[str, Any]:
    """Validate artifacts and return the complete done-marker payload."""
    checkpoint_step = _validate_checkpoint_identity(checkpoint_path, completed_updates)
    expected_targets = _manifest_targets(pdb_list)
    expected_count = len(expected_targets)

    status_path = run_dir / "manifest" / "run_status.json"
    status = _read_json(status_path, "run status")
    if status.get("status") != "completed":
        raise ResultValidationError(
            f"run status must be 'completed'; got {status.get('status')!r}: {status_path}"
        )
    if status.get("exit_reason") != "eval_only_completed":
        raise ResultValidationError(
            "run exit_reason must be 'eval_only_completed'; "
            f"got {status.get('exit_reason')!r}: {status_path}"
        )
    recorded_run_dir = status.get("run_dir")
    if recorded_run_dir is not None and os.path.abspath(str(recorded_run_dir)) != os.path.abspath(run_dir):
        raise ResultValidationError(
            f"run_status run_dir mismatch: {recorded_run_dir!r} != {str(run_dir)!r}"
        )

    metrics_path = run_dir / "logs" / "eval_metrics.jsonl"
    metric_records = list(_iter_jsonl(metrics_path, "evaluation metrics"))
    _, metric_record = metric_records[-1]
    if metric_record.get("event") != "eval_metrics":
        raise ResultValidationError("last evaluation metrics record has the wrong event")
    if metric_record.get("test_name") != EXPECTED_TEST_NAME:
        raise ResultValidationError(
            f"last metrics test_name must be {EXPECTED_TEST_NAME!r}; "
            f"got {metric_record.get('test_name')!r}"
        )
    if metric_record.get("ema_suffix") != EXPECTED_EMA_SUFFIX or metric_record.get("is_ema") is not True:
        raise ResultValidationError("last evaluation metrics record is not EMA 0.999")
    for count_key in ("sample_count", "num_batches"):
        if metric_record.get(count_key) != expected_count:
            raise ResultValidationError(
                f"last metrics {count_key} must be {expected_count}; "
                f"got {metric_record.get(count_key)!r}"
            )
    eval_seconds = _finite_number(metric_record.get("eval_seconds"), "eval_seconds")
    if eval_seconds < 0:
        raise ResultValidationError(f"eval_seconds must be non-negative; got {eval_seconds}")

    metrics = metric_record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ResultValidationError("last evaluation metrics record has no metrics object")
    metric_summary: dict[str, dict[str, dict[str, float]]] = {}
    metric_prefix = f"{EXPECTED_TEST_NAME}/{EXPECTED_EMA_SUFFIX}lddt/af3_proxy"
    for category in REQUIRED_CATEGORIES:
        category_summary: dict[str, dict[str, float]] = {}
        for metric_name in REQUIRED_METRICS:
            aggregate_summary = {}
            for aggregate in REQUIRED_AGGREGATES:
                key = f"{metric_prefix}/{category}/{metric_name}.{aggregate}"
                if key not in metrics:
                    raise ResultValidationError(f"required metric is missing: {key}")
                aggregate_summary[aggregate] = _finite_number(metrics[key], key)
            category_summary[metric_name] = aggregate_summary
        metric_summary[category] = category_summary

    samples_path = run_dir / "logs" / "eval_samples_rank0.jsonl"
    sample_ids = []
    for line_number, sample_record in _iter_jsonl(samples_path, "evaluation samples"):
        if sample_record.get("event") != "eval_sample":
            raise ResultValidationError(f"sample record {line_number} has the wrong event")
        if sample_record.get("is_ema") is not True or sample_record.get("ema_suffix") != EXPECTED_EMA_SUFFIX:
            raise ResultValidationError(f"sample record {line_number} is not EMA 0.999")
        sample = sample_record.get("sample")
        if not isinstance(sample, Mapping):
            raise ResultValidationError(f"sample record {line_number} has no sample object")
        sample_ids.append(_single_string(sample.get("pdb_id"), f"sample record {line_number} pdb_id").lower())
    if len(sample_ids) != expected_count:
        raise ResultValidationError(
            f"evaluation sample record count must be {expected_count}; got {len(sample_ids)}"
        )
    if len(sample_ids) != len(set(sample_ids)):
        raise ResultValidationError("evaluation sample records contain duplicate pdb_id values")
    if set(sample_ids) != set(expected_targets):
        missing = sorted(set(expected_targets) - set(sample_ids))
        unexpected = sorted(set(sample_ids) - set(expected_targets))
        raise ResultValidationError(
            "evaluation sample IDs do not exactly match the frozen manifest; "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    _scan_log(log_path)
    finished_at = status.get("updated_at")
    if not isinstance(finished_at, str) or not finished_at:
        finished_at = datetime.now(timezone.utc).isoformat()
    return {
        "validation_contract": VALIDATION_CONTRACT,
        "completed_updates": completed_updates,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "scope": scope,
        "subset": subset,
        "mode": mode,
        "run_dir": os.path.abspath(run_dir),
        "log_path": os.path.abspath(log_path),
        "pdb_list": os.path.abspath(pdb_list),
        "sample_count": expected_count,
        "num_batches": expected_count,
        "is_ema": True,
        "eval_seconds": eval_seconds,
        "finished_at": finished_at,
        "metrics": metric_summary,
    }


def _same_path(left: Any, right: Path) -> bool:
    return isinstance(left, str) and os.path.abspath(left) == os.path.abspath(right)


def validate_done_marker(
    *,
    marker_path: Path,
    pdb_list: Path,
    checkpoint_path: Path,
    completed_updates: int,
    scope: str,
    subset: str,
    mode: str,
) -> dict[str, Any]:
    """Revalidate an existing done marker and all artifacts it references."""
    marker = _read_json(marker_path, "done marker")
    expected_scalars = {
        "validation_contract": VALIDATION_CONTRACT,
        "completed_updates": completed_updates,
        "checkpoint_step": completed_updates - 1,
        "scope": scope,
        "subset": subset,
        "mode": mode,
        "is_ema": True,
    }
    for key, expected in expected_scalars.items():
        if marker.get(key) != expected:
            raise ResultValidationError(
                f"done marker {key} mismatch: {marker.get(key)!r} != {expected!r}"
            )
    if not _same_path(marker.get("checkpoint_path"), checkpoint_path):
        raise ResultValidationError("done marker checkpoint_path mismatch")
    if not _same_path(marker.get("pdb_list"), pdb_list):
        raise ResultValidationError("done marker pdb_list mismatch")
    try:
        run_dir = Path(_single_string(marker.get("run_dir"), "done marker run_dir"))
        log_path = Path(_single_string(marker.get("log_path"), "done marker log_path"))
    except TypeError as exc:
        raise ResultValidationError(str(exc)) from exc
    validated = validate_run(
        run_dir=run_dir,
        log_path=log_path,
        pdb_list=pdb_list,
        checkpoint_path=checkpoint_path,
        completed_updates=completed_updates,
        scope=scope,
        subset=subset,
        mode=mode,
    )
    for key in ("sample_count", "num_batches", "eval_seconds"):
        if marker.get(key) != validated.get(key):
            raise ResultValidationError(
                f"done marker {key} does not match revalidated artifacts"
            )
    if marker.get("metrics") != validated.get("metrics"):
        raise ResultValidationError("done marker metrics do not match revalidated artifacts")
    return validated


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--done-marker", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--pdb-list", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--completed-updates", required=True, type=int)
    parser.add_argument("--scope", default="full")
    parser.add_argument("--subset", default="combined")
    parser.add_argument("--mode", default="formal")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.completed_updates <= 0:
        raise ResultValidationError("--completed-updates must be positive")
    if args.done_marker is not None:
        if args.log_path is not None:
            raise ResultValidationError("--log-path cannot be used with --done-marker")
        summary = validate_done_marker(
            marker_path=args.done_marker,
            pdb_list=args.pdb_list,
            checkpoint_path=args.checkpoint_path,
            completed_updates=args.completed_updates,
            scope=args.scope,
            subset=args.subset,
            mode=args.mode,
        )
    else:
        if args.log_path is None:
            raise ResultValidationError("--log-path is required with --run-dir")
        summary = validate_run(
            run_dir=args.run_dir,
            log_path=args.log_path,
            pdb_list=args.pdb_list,
            checkpoint_path=args.checkpoint_path,
            completed_updates=args.completed_updates,
            scope=args.scope,
            subset=args.subset,
            mode=args.mode,
        )
    if args.output is not None:
        _atomic_write_json(args.output, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResultValidationError as exc:
        raise SystemExit(f"result validation failed: {exc}") from exc
