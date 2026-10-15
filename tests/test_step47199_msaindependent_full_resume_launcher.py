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
LAUNCHER = REPO / "resume_step47199_msaindependent_only.sh"
SOURCE_RUN = (
    "/mnt/project_abd/klp/proteinx_output/"
    "protenix_v4_step27799_msaindependent_only_softcluster_g02_"
    "v9_straggler_20260807_053504"
)
SCHEDULE = (
    "/mnt/project_abd/klp/proteinx_output/soft_cluster_schedules/"
    "step21999_residual_ge200_g02_v2"
)
TRAINING_CONTRACT = (
    "crop384_db48_ddp_disordered_v5_ws128_acc4_catalogreplacement_"
    "residual_ge200_20260804_softcluster_g02_checkpointbound_step21999_"
    "wave_schedule_v2_weightedpdb_pairing_nonpairing_independent10k_v1"
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
            "RUN_NAME": (
                "protenix_v4_from47199_msaindependent_only_"
                "softcluster_g02_v9_straggler_test"
            ),
            "BASE_DIR": str(tmp_path / "output"),
            "RESUME_NNODES": "16",
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


def _command(result: subprocess.CompletedProcess[str]) -> list[str]:
    line = next(
        line for line in result.stdout.splitlines() if line.startswith("torchrun ")
    )
    return shlex.split(line)


def _arg(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_launcher_expands_same_contract_full_state_resume(tmp_path: Path):
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert (
        "source_step=47199 next_trainer_step=47200 "
        "next_global_step=188800"
    ) in result.stdout
    assert "resume_mode=same_contract contract_migration=none" in result.stdout
    assert "remaining_updates=2800 checkpoint_interval=200" in result.stdout
    assert "cursor=188800 target=200000 unchanged=true" in result.stdout
    assert f"training_contract={TRAINING_CONTRACT}" in result.stdout

    command = _command(result)
    assert _arg(command, "--load_checkpoint_path") == (
        f"{SOURCE_RUN}/checkpoints/47199.pt"
    )
    assert _arg(command, "--load_ema_checkpoint_path") == (
        f"{SOURCE_RUN}/checkpoints/47199_ema_0.999.pt"
    )
    assert _arg(command, "--load_params_only") == "false"
    assert _arg(command, "--load_strict") == "true"
    assert _arg(command, "--skip_load_optimizer") == "false"
    assert _arg(command, "--skip_load_scheduler") == "false"
    assert _arg(command, "--skip_load_step") == "false"
    assert _arg(command, "--load_step_for_scheduler") == "false"
    assert _arg(command, "--distributed_strategy") == "ddp"
    assert _arg(command, "--iters_to_accumulate") == "4"
    assert _arg(command, "--n_blocks") == "64"
    assert _arg(command, "--max_steps") == "50000"
    assert _arg(command, "--checkpoint_interval") == "200"
    assert _arg(command, "--data.train_sampler.train_sample_weights") == (
        "0.49,0.005,0.505,0.02"
    )
    assert _arg(
        command,
        "--data.train_sampler.soft_cluster_schedule_manifest",
    ) == SCHEDULE
    assert _arg(command, "--data.train_sampler.soft_cluster_gamma") == "0.2"
    assert _arg(
        command,
        "--data.msa.independent_pairing_non_pairing_budget",
    ) == "true"
    assert _arg(command, "--data.msa.max_input_paired_msa_depth") == "10000"
    assert _arg(command, "--data.msa.max_input_unpaired_msa_depth") == "10000"
    assert _arg(command, "--data.msa.max_size.train") == "16384"
    assert _arg(command, "--data.msa.sample_cutoff.train") == "2048"
    assert _arg(command, "--model.msa_module.msa_chunk_size") == "2048"
    assert _arg(command, "--lr") == "0.001"


def test_launcher_binds_the_audited_checkpoint_pair_and_contract():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "858ef830befb2e812b3eef1f7ec0f4a8e7db962e18a894975f95af87f9c8ce6f" in text
    assert "1a29bf7b121dab71533d783c761bf850db76431605fc8424a01dc16156ed8730" in text
    assert "bd6a4a8c0ca5f5f2c50384ef0b88af9edd249b525916273b14e96a9468ef70d5" in text
    assert "ccf4fcfd799521fbc7cd0be1c6cc0eea744cf18fe87ab906c24bc485486dce00" in text
    assert TRAINING_CONTRACT in text.replace('"\\\n"', "")
    assert "PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION=" not in text


def test_launcher_rejects_old_migration_and_source_overrides(tmp_path: Path):
    for variable_name, value in (
        ("RESUME_STEP", "27799"),
        ("RESUME_RUN_DIR", "/tmp/wrong"),
        (
            "PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION",
            "weightedpdb_independent_pairing_nonpairing_full_state_v1",
        ),
        ("PROTENIX_SOFT_CLUSTER_SCHEDULE_DIR", "/tmp/wrong"),
        ("PROTENIX_OPENFOLD3_DISORDERED_ROOT_DIR", "/tmp/wrong"),
        ("INDEPENDENT_PAIRING_NON_PAIRING_BUDGET", "false"),
    ):
        result = _run(tmp_path, **{variable_name: value})
        assert result.returncode == 2
        assert f"rejects inherited {variable_name}" in result.stderr


def test_launcher_rejects_ambiguous_run_name(tmp_path: Path):
    result = _run(
        tmp_path,
        RUN_NAME="protenix_v4_msaindependent_only_softcluster_g02_v9_straggler",
    )
    assert result.returncode == 2
    assert "RUN_NAME must contain from47199" in result.stderr


def test_launcher_never_allows_path_bypass_for_real_training(tmp_path: Path):
    result = _run(tmp_path, DRY_RUN="0", SKIP_PATH_CHECK="1")
    assert result.returncode == 2
    assert "allowed only with DRY_RUN=1" in result.stderr
