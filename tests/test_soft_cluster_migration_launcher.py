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

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "migrate_crop384_db48_ddp_disordered_v4_to_softcluster_g02.sh"
TARGET_CONTRACT = (
    "crop384_db48_ddp_disordered_v4_ws128_acc4_softcluster_g02_"
    "checkpointbound_wave_schedule_v1"
)


def _env(source_run: Path, schedule_dir: Path, **overrides: str) -> dict[str, str]:
    env = {
        key: os.environ[key]
        for key in ("HOME", "PATH", "LANG", "LC_ALL")
        if key in os.environ
    }
    env.update(
        {
            "DRY_RUN": "1",
            "SKIP_PATH_CHECK": "1",
            "MLP_WORKER_NUM": "16",
            "MLP_ROLE_INDEX": "0",
            "MLP_WORKER_0_HOST": "worker-0",
            "MLP_WORKER_0_PORT": "23456",
            "NPROC_PER_NODE": "8",
            "RUN_NAME": "disordered_v4_softcluster_g02_v9_straggler_test",
            "BASE_DIR": str(source_run.parent / "output"),
            "RESUME_RUN_DIR": str(source_run),
            "RESUME_STEP": "47999",
            "SOFT_CLUSTER_SCHEDULE_DIR": str(schedule_dir),
        }
    )
    env.update(overrides)
    return env


def _run(
    source_run: Path,
    schedule_dir: Path,
    *,
    check: bool = True,
    **overrides: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO,
        env=_env(source_run, schedule_dir, **overrides),
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _command(result: subprocess.CompletedProcess[str]) -> list[str]:
    line = next(
        line for line in result.stdout.splitlines() if line.startswith("torchrun ")
    )
    return shlex.split(line)


def _arg(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_launcher_passes_schedule_and_preserves_full_state_resume(tmp_path: Path):
    source_run = tmp_path / "source_run"
    schedule_dir = tmp_path / "schedule"
    source_run.mkdir()
    schedule_dir.mkdir()

    result = _run(source_run, schedule_dir)
    command = _command(result)

    assert _arg(command, "--data.train_sampler.soft_cluster_schedule_manifest") == str(
        schedule_dir.resolve()
    )
    assert _arg(command, "--data.train_sampler.soft_cluster_gamma") == "0.2"
    assert _arg(command, "--load_params_only") == "false"
    assert _arg(command, "--skip_load_optimizer") == "false"
    assert _arg(command, "--skip_load_scheduler") == "false"
    assert _arg(command, "--skip_load_step") == "false"
    assert _arg(command, "--max_steps") == "50000"
    assert _arg(command, "--nnodes") == "16"
    assert _arg(command, "--nproc_per_node") == "8"
    assert f"target_training_contract={TARGET_CONTRACT}" in result.stdout
    assert f"contract={TARGET_CONTRACT}" in result.stdout
    assert "policy_delta=Short/Long_cluster_weight_gamma_power_exposure" in result.stdout


def test_launcher_preserves_explicit_checkpoint_file_selection(tmp_path: Path):
    source_run = tmp_path / "source_run"
    schedule_dir = tmp_path / "schedule"
    source_run.mkdir()
    schedule_dir.mkdir()
    raw = source_run / "selected" / "raw.pt"
    ema = source_run / "selected" / "selected_ema_0.999.pt"

    result = _run(
        source_run,
        schedule_dir,
        RESUME_RAW_CHECKPOINT=str(raw),
        RESUME_EMA_CHECKPOINT=str(ema),
    )
    command = _command(result)

    assert _arg(command, "--load_checkpoint_path") == str(raw.resolve())
    assert _arg(command, "--load_ema_checkpoint_path") == str(ema.resolve())
    assert f"source_raw_checkpoint={raw.resolve()}" in result.stdout
    assert f"source_ema_checkpoint={ema.resolve()}" in result.stdout


def test_launcher_rejects_missing_schedule_or_inherited_internal_controls(
    tmp_path: Path,
):
    source_run = tmp_path / "source_run"
    schedule_dir = tmp_path / "schedule"
    source_run.mkdir()
    schedule_dir.mkdir()

    missing = _run(
        source_run,
        schedule_dir,
        check=False,
        SOFT_CLUSTER_SCHEDULE_DIR="",
    )
    assert missing.returncode == 2
    assert "SOFT_CLUSTER_SCHEDULE_DIR is required" in missing.stderr
    assert "torchrun " not in missing.stdout

    inherited = _run(
        source_run,
        schedule_dir,
        check=False,
        PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION="future_policy",
    )
    assert inherited.returncode == 2
    assert "rejects inherited PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION" in inherited.stderr
    assert "torchrun " not in inherited.stdout
