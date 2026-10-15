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
LAUNCHER = REPO / "migrate_step21999_to_residual_ge200_softcluster_g02.sh"
HISTORICAL_LAUNCHER = (
    REPO / "migrate_step21399_to_residual_ge200_softcluster_g02.sh"
)
SOURCE_RUN = (
    "/mnt/project_abd/klp/proteinx_output/"
    "protenix_v4_crop384_db48_disordered_v4_v10_to_v11_"
    "msaquerypinle1024_from_s9999_v9_straggler_20260730_104039"
)
SCHEDULE = (
    "/mnt/project_abd/klp/proteinx_output/soft_cluster_schedules/"
    "step21999_residual_ge200_g02_v2"
)


def _base_env(tmp_path: Path) -> dict[str, str]:
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
            "RUN_NAME": (
                "protenix_v4_step21999_residual_ge200_"
                "softcluster_g02_v9_straggler_test"
            ),
            "BASE_DIR": str(tmp_path / "output"),
        }
    )
    return env


def _run(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    env = _base_env(tmp_path)
    env.update(overrides)
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_step21999_launcher_locks_full_state_boundary_and_target(tmp_path: Path):
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "source_step=21999 target_start_step=22000 target_updates=50000" in (
        result.stdout
    )
    assert f"schedule={SCHEDULE}" in result.stdout
    assert "topology=16x8 world_size=128 accumulation=4" in result.stdout
    assert "schedule_validation=skipped_only_for_explicit_dry_run" in result.stdout

    command_line = next(
        line for line in result.stdout.splitlines() if line.startswith("torchrun ")
    )
    command = shlex.split(command_line)

    def arg(name: str) -> str:
        return command[command.index(name) + 1]

    assert arg("--load_checkpoint_path") == f"{SOURCE_RUN}/checkpoints/21999.pt"
    assert arg("--load_ema_checkpoint_path") == (
        f"{SOURCE_RUN}/checkpoints/21999_ema_0.999.pt"
    )
    assert arg("--load_params_only") == "false"
    assert arg("--skip_load_optimizer") == "false"
    assert arg("--skip_load_scheduler") == "false"
    assert arg("--skip_load_step") == "false"
    assert arg("--max_steps") == "50000"
    assert arg("--data.train_sampler.soft_cluster_gamma") == "0.2"


def test_step21999_launcher_rejects_step_and_checkpoint_overrides(tmp_path: Path):
    wrong_step = _run(tmp_path, RESUME_STEP="21399")
    assert wrong_step.returncode == 2
    assert "locked to source step 21999" in wrong_step.stderr

    wrong_raw = _run(
        tmp_path,
        RESUME_RAW_CHECKPOINT=f"{SOURCE_RUN}/checkpoints/21399.pt",
    )
    assert wrong_raw.returncode == 2
    assert "locked step-21999 raw file" in wrong_raw.stderr

    wrong_ema = _run(
        tmp_path,
        RESUME_EMA_CHECKPOINT=(
            f"{SOURCE_RUN}/checkpoints/21399_ema_0.999.pt"
        ),
    )
    assert wrong_ema.returncode == 2
    assert "locked step-21999 EMA file" in wrong_ema.stderr


def test_step21999_launcher_rejects_schedule_and_runtime_catalog_drift(
    tmp_path: Path,
):
    wrong_schedule = _run(
        tmp_path,
        SOFT_CLUSTER_SCHEDULE_DIR=(
            "/mnt/project_abd/klp/proteinx_output/soft_cluster_schedules/"
            "step21399_residual_ge200_g02_v2"
        ),
    )
    assert wrong_schedule.returncode == 2
    assert "locked to schedule" in wrong_schedule.stderr

    wrong_runtime = _run(
        tmp_path,
        PROTENIX_GE200_STRUCTURAL_RUNTIME_INDEX_DIR="/tmp/not-the-formal-ge200",
    )
    assert wrong_runtime.returncode == 2
    assert "published ge200 runtime catalog" in wrong_runtime.stderr


def test_step21999_launcher_keeps_historical_launcher_independent():
    new_text = LAUNCHER.read_text(encoding="utf-8")
    historical_text = HISTORICAL_LAUNCHER.read_text(encoding="utf-8")
    assert "--source-step 21999" in new_text
    assert "step21999_residual_ge200_g02_v2" in new_text
    assert "source_raw_sha256=" not in new_text
    assert "source_ema_sha256=" not in new_text
    assert "--source-step 21399" in historical_text
    assert "source_step=21399 target_start_step=21400" in historical_text


def test_step21999_launcher_requires_explicit_run_identity(tmp_path: Path):
    result = _run(
        tmp_path,
        RUN_NAME="protenix_v4_residual_ge200_softcluster_g02_v9_straggler_test",
    )
    assert result.returncode == 2
    assert "RUN_NAME must contain step21999" in result.stderr


def test_step21999_launcher_never_allows_path_bypass_for_real_training(
    tmp_path: Path,
):
    result = _run(tmp_path, DRY_RUN="0", SKIP_PATH_CHECK="1")
    assert result.returncode == 2
    assert "allowed only with DRY_RUN=1" in result.stderr
