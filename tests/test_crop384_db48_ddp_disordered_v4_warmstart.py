"""Contracts for the v8-step999 -> disordered-v4 params-only stage init."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "run_crop384_db48_ddp_disordered_v4_from_v8_step999.sh"
RESUME_LAUNCHER = REPO / "resume_crop384_db48_ddp_disordered_v4.sh"
RAW = (
    "/mnt/project_abd/klp/proteinx_output/"
    "protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_"
    "msamix_q20_u30_p50_50k_v8_20260724_155026/checkpoints/999.pt"
)
EMA = RAW.removesuffix("999.pt") + "999_ema_0.999.pt"


def _env() -> dict[str, str]:
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
                "protenix_v4_crop384_db48_disordered_v4_"
                "from_v8s999_50k_v9_straggler_test"
            ),
            "BASE_DIR": "/tmp/protenix-v4-warmstart-test",
        }
    )
    return env


def _run(env: dict[str, str], *, check: bool = True):
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO,
        env=env,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _command(result) -> list[str]:
    line = next(
        line for line in result.stdout.splitlines() if line.startswith("torchrun ")
    )
    return shlex.split(line)


def _arg(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_v4_stage_init_locks_source_pair_weights_and_fresh_state():
    result = _run(_env())
    command = _command(result)
    assert _arg(command, "--load_checkpoint_path") == RAW
    assert _arg(command, "--load_ema_checkpoint_path") == EMA
    assert _arg(command, "--load_params_only") == "true"
    assert _arg(command, "--load_strict") == "true"
    assert _arg(command, "--skip_load_optimizer") == "true"
    assert _arg(command, "--skip_load_scheduler") == "true"
    assert _arg(command, "--skip_load_step") == "true"
    assert _arg(command, "--load_step_for_scheduler") == "false"
    assert _arg(command, "--max_steps") == "50000"
    assert _arg(command, "--data.train_sampler.train_sample_weights") == (
        "0.49,0.005,0.505,0.02"
    )
    assert "data_profile=weighted_pdb_afdb_mgnify_short_long_disordered_v4" in (
        result.stdout
    )
    assert "target_start_step=0 target_start_global_step=0" in result.stdout
    assert (
        "template_cache=/mnt/project_abd/klp/databases/"
        "protenix_sequence_template_cache_top60_uniform20_id90_v1"
        in result.stdout
    )
    assert "template_cache_mode/error_policy=required/chain_dummy" in result.stdout
    assert _arg(command, "--checkpoint_interval") == "200"
    assert _arg(command, "--data.msa.max_input_paired_msa_depth") == "10000"
    assert _arg(command, "--data.msa.max_input_unpaired_msa_depth") == "10000"
    for dataset_name in (
        "afdb_mgnify_short_distillation_v20260725",
        "afdb_mgnify_long_distillation_v20260725",
    ):
        assert _arg(
            command,
            f"--data.{dataset_name}.msa.max_input_unpaired_msa_depth",
        ) == "16384"
    assert "msaraw_p10000_u10000_distillu16384_final16384" in result.stdout
    assert (
        "MSA raw_prefix_cap_train AFDB/MGnify short/long "
        "direct-unpaired=16384 paired-source=10000(unused)"
    ) in result.stdout


def test_v4_stage_init_rejects_weight_override():
    env = _env()
    env["PROTENIX_PDB_SAMPLE_WEIGHT"] = "0.5"
    result = _run(env, check=False)
    assert result.returncode == 2
    assert "rejects inherited PROTENIX_PDB_SAMPLE_WEIGHT" in result.stderr


def test_v4_stage_init_rejects_template_cache_override():
    env = _env()
    env["PROTENIX_SEQUENCE_TEMPLATE_CACHE_ROOT"] = "/tmp/not-the-locked-cache"
    result = _run(env, check=False)
    assert result.returncode == 2
    assert (
        "rejects inherited PROTENIX_SEQUENCE_TEMPLATE_CACHE_ROOT"
        in result.stderr
    )


def test_v4_stage_init_requires_lineage_in_run_name():
    env = _env()
    env["RUN_NAME"] = "disordered_v4_v9_straggler"
    result = _run(env, check=False)
    assert result.returncode == 2
    assert "RUN_NAME must contain from_v8s999" in result.stderr


def test_v4_stage_init_smoke_still_loads_pair_and_starts_at_zero():
    env = _env()
    env["PROTENIX_SMOKE_TEST"] = "1"
    env["PROTENIX_SMOKE_UPDATES"] = "1"
    result = _run(env)
    command = _command(result)
    assert _arg(command, "--max_steps") == "1"
    assert _arg(command, "--skip_final_checkpoint") == "true"
    assert "launch_mode=params_only_warmstart_smoke" in result.stdout


def test_v4_checkpoints_have_a_same_contract_strict_resume_entry():
    env = _env()
    env.update(
        {
            "RUN_NAME": "protenix_disordered_v4_resume_v9_straggler",
            "RESUME_RUN_DIR": "/tmp/v4-source",
            "RESUME_STEP": "999",
        }
    )
    result = subprocess.run(
        ["bash", str(RESUME_LAUNCHER)],
        cwd=REPO,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    command = _command(result)
    assert _arg(command, "--load_params_only") == "false"
    assert _arg(command, "--data.train_sampler.train_sample_weights") == (
        "0.49,0.005,0.505,0.02"
    )
    assert "data_profile=weighted_pdb_afdb_mgnify_short_long_disordered_v4" in (
        result.stdout
    )
    assert _arg(command, "--checkpoint_interval") == "200"
    assert "template_cache_mode/error_policy=required/chain_dummy" in result.stdout
    assert "msaraw_p10000_u10000_distillu16384_final16384" in result.stdout
    for dataset_name in (
        "afdb_mgnify_short_distillation_v20260725",
        "afdb_mgnify_long_distillation_v20260725",
    ):
        assert _arg(
            command,
            f"--data.{dataset_name}.msa.max_input_unpaired_msa_depth",
        ) == "16384"
