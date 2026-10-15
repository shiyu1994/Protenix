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

"""Focused tests for the every-600 AF3-like PXM evaluation queue."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.evaluation.validate_af3_pxm_eval_result import (
    EXPECTED_EMA_SUFFIX,
    EXPECTED_TEST_NAME,
    REQUIRED_AGGREGATES,
    REQUIRED_CATEGORIES,
    REQUIRED_METRICS,
    ResultValidationError,
    validate_run,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
QUEUE = REPOSITORY_ROOT / "run_af3_pxm_ema_eval_queue.sh"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _metric_values(value: float = 50.0) -> dict[str, float]:
    prefix = f"{EXPECTED_TEST_NAME}/{EXPECTED_EMA_SUFFIX}lddt/af3_proxy"
    return {
        f"{prefix}/{category}/{metric}.{aggregate}": value
        for category in REQUIRED_CATEGORIES
        for metric in REQUIRED_METRICS
        for aggregate in REQUIRED_AGGREGATES
    }


def _make_result_fixture(root: Path, targets: list[str]) -> tuple[Path, Path]:
    run_dir = root / "run"
    _write_json(
        run_dir / "manifest" / "run_status.json",
        {
            "status": "completed",
            "exit_reason": "eval_only_completed",
            "updated_at": "2026-07-29T00:00:00+00:00",
        },
    )
    _append_jsonl(
        run_dir / "logs" / "eval_metrics.jsonl",
        [
            {
                "event": "eval_metrics",
                "test_name": EXPECTED_TEST_NAME,
                "ema_suffix": EXPECTED_EMA_SUFFIX,
                "is_ema": True,
                "sample_count": len(targets),
                "num_batches": len(targets),
                "eval_seconds": 12.5,
                "metrics": _metric_values(),
            }
        ],
    )
    _append_jsonl(
        run_dir / "logs" / "eval_samples_rank0.jsonl",
        [
            {
                "event": "eval_sample",
                "ema_suffix": EXPECTED_EMA_SUFFIX,
                "is_ema": True,
                "sample": {"pdb_id": target},
            }
            for target in targets
        ],
    )
    evaluator_log = root / "evaluator.log"
    evaluator_log.write_text("evaluation completed normally\n", encoding="utf-8")
    return run_dir, evaluator_log


def test_result_validator_accepts_complete_exact_target_run(tmp_path: Path) -> None:
    targets = ["7abc", "8def", "9ghi"]
    pdb_list = tmp_path / "combined_le1024.txt"
    pdb_list.write_text("\n".join(targets) + "\n", encoding="utf-8")
    run_dir, evaluator_log = _make_result_fixture(tmp_path, targets)
    checkpoint = tmp_path / "599_ema_0.999.pt"

    summary = validate_run(
        run_dir=run_dir,
        log_path=evaluator_log,
        pdb_list=pdb_list,
        checkpoint_path=checkpoint,
        completed_updates=600,
        scope="full",
        subset="combined",
        mode="formal",
    )

    assert summary["sample_count"] == 3
    assert summary["checkpoint_step"] == 599
    assert summary["is_ema"] is True
    assert set(summary["metrics"]) == set(REQUIRED_CATEGORIES)


def test_result_validator_rejects_nonfinite_metric_and_wrong_coverage(
    tmp_path: Path,
) -> None:
    targets = ["7abc", "8def"]
    pdb_list = tmp_path / "combined_le1024.txt"
    pdb_list.write_text("\n".join(targets) + "\n", encoding="utf-8")
    run_dir, evaluator_log = _make_result_fixture(tmp_path, targets)
    checkpoint = tmp_path / "599_ema_0.999.pt"
    metrics_path = run_dir / "logs" / "eval_metrics.jsonl"
    metrics_record = json.loads(metrics_path.read_text(encoding="utf-8"))
    metric_key = (
        f"{EXPECTED_TEST_NAME}/{EXPECTED_EMA_SUFFIX}lddt/af3_proxy/"
        "intra_prot/top1_pct.avg"
    )
    metrics_record["metrics"][metric_key] = float("nan")
    _append_jsonl(metrics_path, [metrics_record])

    with pytest.raises(ResultValidationError, match="not finite"):
        validate_run(
            run_dir=run_dir,
            log_path=evaluator_log,
            pdb_list=pdb_list,
            checkpoint_path=checkpoint,
            completed_updates=600,
            scope="full",
            subset="combined",
            mode="formal",
        )

    metrics_record["metrics"][metric_key] = 50.0
    _append_jsonl(metrics_path, [metrics_record])
    sample_path = run_dir / "logs" / "eval_samples_rank0.jsonl"
    sample_records = [
        {
            "event": "eval_sample",
            "ema_suffix": EXPECTED_EMA_SUFFIX,
            "is_ema": True,
            "sample": {"pdb_id": "7abc"},
        },
        {
            "event": "eval_sample",
            "ema_suffix": EXPECTED_EMA_SUFFIX,
            "is_ema": True,
            "sample": {"pdb_id": "7abc"},
        },
    ]
    _append_jsonl(sample_path, sample_records)
    with pytest.raises(ResultValidationError, match="duplicate pdb_id"):
        validate_run(
            run_dir=run_dir,
            log_path=evaluator_log,
            pdb_list=pdb_list,
            checkpoint_path=checkpoint,
            completed_updates=600,
            scope="full",
            subset="combined",
            mode="formal",
        )


def _queue_environment(
    *,
    train_run_dir: Path,
    manifest_dir: Path,
    eval_base_dir: Path,
    **updates: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "TRAIN_RUN_DIR": str(train_run_dir),
            "PXM_MANIFEST_DIR": str(manifest_dir),
            "EVAL_BASE_DIR": str(eval_base_dir),
            "QUEUE_PYTHON": sys.executable,
            "EVAL_GPUS": "2,3",
            "MASTER_PORT_BASE": "31000",
            "POLL_SECONDS": "60",
            "DISPATCH_POLL_SECONDS": "1",
            "RETRY_DELAY_SECONDS": "0",
            "WAIT_FOR_CHECKPOINTS": "false",
        }
    )
    env.update({key: str(value) for key, value in updates.items()})
    return env


def _make_training_fixture(tmp_path: Path, steps: list[int]) -> tuple[Path, Path]:
    train_run_dir = tmp_path / "source_run"
    checkpoint_dir = train_run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for step in steps:
        (checkpoint_dir / f"{step}_ema_0.999.pt").write_text(
            "checkpoint\n", encoding="utf-8"
        )
    _write_json(
        train_run_dir / "manifest" / "run_status.json",
        {"status": "running"},
    )
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "combined_le1024.txt").write_text(
        "7abc\n8def\n", encoding="utf-8"
    )
    return train_run_dir, manifest_dir


def test_queue_dry_run_discovers_numeric_600_grid_and_assigns_lanes(
    tmp_path: Path,
) -> None:
    train_run_dir, manifest_dir = _make_training_fixture(
        tmp_path,
        [599, 600, 1199, 1799],
    )
    result = subprocess.run(
        ["bash", str(QUEUE)],
        cwd=REPOSITORY_ROOT,
        env=_queue_environment(
            train_run_dir=train_run_dir,
            manifest_dir=manifest_dir,
            eval_base_dir=tmp_path / "eval",
            DRY_RUN="1",
            PRIORITIZE_LATEST="true",
        ),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert "planned_eval update=600 " in result.stdout
    assert "planned_eval update=1200 " in result.stdout
    assert "planned_eval update=1800 " in result.stdout
    assert "update=601 " not in result.stdout
    planned_lines = [
        line for line in result.stdout.splitlines() if line.startswith("planned_eval")
    ]
    assert "update=1800 " in planned_lines[0]
    assert "gpu=2 master_port=31000" in planned_lines[0]
    assert "update=600 " in planned_lines[1]
    assert "gpu=3 master_port=31001" in planned_lines[1]
    assert "update=1200 " in planned_lines[2]
    assert result.stdout.count("planned_eval") == 3


def test_queue_min_update_skips_earlier_grid_points(tmp_path: Path) -> None:
    train_run_dir, manifest_dir = _make_training_fixture(
        tmp_path,
        [599, 9599, 10199, 10799],
    )
    result = subprocess.run(
        ["bash", str(QUEUE)],
        cwd=REPOSITORY_ROOT,
        env=_queue_environment(
            train_run_dir=train_run_dir,
            manifest_dir=manifest_dir,
            eval_base_dir=tmp_path / "eval",
            DRY_RUN="1",
            EVAL_MIN_UPDATE="10200",
        ),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert "min_update=10200" in result.stdout
    assert "planned_eval update=10200 " in result.stdout
    assert "planned_eval update=10800 " in result.stdout
    assert "planned_eval update=600 " not in result.stdout
    assert "planned_eval update=9600 " not in result.stdout
    assert result.stdout.count("planned_eval") == 2


def test_parallel_queue_retries_then_publishes_shared_done_markers(
    tmp_path: Path,
) -> None:
    train_run_dir, manifest_dir = _make_training_fixture(
        tmp_path,
        [599, 1199, 1799, 2399],
    )
    template_dir, _ = _make_result_fixture(
        tmp_path / "template",
        ["7abc", "8def"],
    )
    fake_evaluator = tmp_path / "fake_evaluator.sh"
    fake_evaluator.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
step_name="$(basename "${EMA_CHECKPOINT}")"
failure_claim="${EVAL_BASE_DIR}/fake_fail_once/${step_name}"
if mkdir "${failure_claim}" 2>/dev/null; then
    echo "intentional first-attempt failure"
    exit 7
fi
run_dir="${EVAL_BASE_DIR}/${RUN_NAME}"
mkdir -p "${run_dir}"
cp -R "${FAKE_RESULT_TEMPLATE}/." "${run_dir}/"
echo "2026-07-30 09:00:00,000 [runner/train.py:6834] INFO root: Using run name: ${RUN_NAME}, run dir: ${run_dir}, checkpoint_dir: ${run_dir}/checkpoints, \
prediction_dir: ${run_dir}/predictions"
""",
        encoding="utf-8",
    )
    fake_evaluator.chmod(0o755)
    eval_base_dir = tmp_path / "eval"
    (eval_base_dir / "fake_fail_once").mkdir(parents=True)
    result = subprocess.run(
        ["bash", str(QUEUE)],
        cwd=REPOSITORY_ROOT,
        env=_queue_environment(
            train_run_dir=train_run_dir,
            manifest_dir=manifest_dir,
            eval_base_dir=eval_base_dir,
            DRY_RUN="0",
            EVALUATOR_SCRIPT=str(fake_evaluator),
            FAKE_RESULT_TEMPLATE=str(template_dir),
            MAX_RETRIES="1",
        ),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    state_dir = (
        eval_base_dir
        / "queue_state"
        / train_run_dir.name
        / "full_combined_formal_every600"
    )
    for update in (600, 1200, 1800, 2400):
        done = json.loads(
            (state_dir / f"update_{update}.done.json").read_text(encoding="utf-8")
        )
        first_attempt = json.loads(
            (state_dir / f"update_{update}.attempt_1.json").read_text(
                encoding="utf-8"
            )
        )
        second_attempt = json.loads(
            (state_dir / f"update_{update}.attempt_2.json").read_text(
                encoding="utf-8"
            )
        )
        assert done["completed_updates"] == update
        assert first_attempt["status"] == "evaluator_failed"
        assert second_attempt["status"] == "completed"
    assert {json.loads(path.read_text())["gpu"] for path in state_dir.glob("*.attempt_2.json")} == {
        "2",
        "3",
    }
    assert "queue_snapshot_drained status=running" in result.stdout


def test_queue_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(QUEUE)], check=True)
