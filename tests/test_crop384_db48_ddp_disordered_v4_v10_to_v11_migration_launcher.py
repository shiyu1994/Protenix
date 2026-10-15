"""Launcher contract for the disordered-v4 v10 -> query-pin v11 migration."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "migrate_crop384_db48_ddp_disordered_v4_v10_to_v11.sh"
SOURCE_RUN = (
    "/mnt/project_abd/klp/proteinx_output/"
    "protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_"
    "v9_straggler_templatecache_chaindummy_msabase10ku10k_"
    "distillu16384_ckpt200_20260726_061627"
)
SOURCE_CONTRACT = (
    "crop384_db48_ddp_scratch_50k_ws128_acc4_data_"
    "weightedpdb_afdb_mgnify_short_long_disordered_v4_"
    "pdb490_short005_long505_dis020_"
    "afq_mgn_cluster_replen_le200_gt200_v1_"
    "proteinonlydiffusion_v1_resolvedproteincrop_minloss4_v2_"
    "allmsashuffle_multichain_seg50_whole50_gap24_min9_v2_"
    "partialquery_max2_keep1_v1_arch_pf64_cs384_dct768_"
    "pfckpt8_c48_msamix_q10_u30_p60_"
    "msaraw_p10000_u10000_distillu16384_final16384_"
    "templatecache_top60u20id90_chaindummy_ckpt200_"
    "dlpersist1_prefetch2_deferredhealth_v10"
)
TARGET_CONTRACT = SOURCE_CONTRACT.removesuffix(
    "templatecache_top60u20id90_chaindummy_ckpt200_"
    "dlpersist1_prefetch2_deferredhealth_v10"
) + (
    "msaquerypinle1024_templatecache_top60u20id90_"
    "chaindummy_ckpt200_dlpersist1_prefetch2_deferredhealth_v11"
)
SOURCE_DATA_SHA256 = (
    "cffda1e6694a9f293c8ed213d1b33e188d828d10a4d0eebdf66b2f428e75756c"
)
TARGET_DATA_SHA256 = (
    "6c98c5cab75197b7e4d41b7ce5287af65ca3f750a707ec69a1bdaf29ed24b8de"
)
ARCHITECTURE_SHA256 = (
    "cd1cbb658e352f0d56a2dcd65c5e0e95ae2e29c321f4fd0da9f79c6145edc34c"
)
MIGRATION_MODE = "msa_query_pin_v10_to_v11_full_state_v1"


def _env(**overrides: str) -> dict[str, str]:
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
                "protenix_disordered_v4_v10_to_v11_"
                "msaquerypinle1024_v9_straggler_test"
            ),
            "BASE_DIR": "/tmp/protenix-v10-to-v11-migration-test",
        }
    )
    env.update(overrides)
    return env


def _run(
    *,
    check: bool = True,
    arguments: list[str] | None = None,
    **overrides: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *(arguments or [])],
        cwd=REPO,
        env=_env(**overrides),
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


def test_default_migration_is_full_state_and_locks_both_contracts():
    result = _run()
    command = _command(result)

    assert _arg(command, "--load_checkpoint_path") == (
        f"{SOURCE_RUN}/checkpoints/9999.pt"
    )
    assert _arg(command, "--load_ema_checkpoint_path") == (
        f"{SOURCE_RUN}/checkpoints/9999_ema_0.999.pt"
    )
    assert _arg(command, "--load_params_only") == "false"
    assert _arg(command, "--load_strict") == "true"
    assert _arg(command, "--skip_load_optimizer") == "false"
    assert _arg(command, "--skip_load_scheduler") == "false"
    assert _arg(command, "--skip_load_step") == "false"
    assert _arg(command, "--load_step_for_scheduler") == "false"
    assert _arg(command, "--max_steps") == "50000"
    assert _arg(command, "--checkpoint_interval") == "200"
    assert _arg(command, "--data.msa.query_inclusion_max_sample_size.train") == (
        "1024"
    )
    assert _arg(command, "--data.msa.query_inclusion_max_sample_size.test") == (
        "1024"
    )
    assert _arg(command, "--nnodes") == "16"
    assert _arg(command, "--nproc_per_node") == "8"

    assert f"migration={MIGRATION_MODE}" in result.stdout
    assert "trajectory_exact=false fail_closed=true" in result.stdout
    assert f"source_run={SOURCE_RUN}" in result.stdout
    assert "source_step=9999 target_start_step=10000" in result.stdout
    assert f"source_training_contract={SOURCE_CONTRACT}" in result.stdout
    assert f"source_data_contract_sha256={SOURCE_DATA_SHA256}" in result.stdout
    assert f"target_training_contract={TARGET_CONTRACT}" in result.stdout
    assert f"target_data_contract_sha256={TARGET_DATA_SHA256}" in result.stdout
    assert f"model_architecture_sha256={ARCHITECTURE_SHA256}" in result.stdout
    assert (
        "restored=model+ema+optimizer+scheduler+step+scaler+rank_rng+"
        "sampler_cursor"
    ) in result.stdout
    assert f"contract={TARGET_CONTRACT}" in result.stdout


def test_explicit_complete_200_update_pair_is_allowed():
    result = _run(RESUME_STEP="9799")
    command = _command(result)

    assert _arg(command, "--load_checkpoint_path").endswith(
        "/checkpoints/9799.pt"
    )
    assert _arg(command, "--load_ema_checkpoint_path").endswith(
        "/checkpoints/9799_ema_0.999.pt"
    )
    assert "source_step=9799 target_start_step=9800" in result.stdout


def test_smoke_keeps_full_state_and_disables_checkpoint_writes():
    result = _run(RESUME_SMOKE_STEPS="2")
    command = _command(result)

    assert _arg(command, "--max_steps") == "10002"
    assert _arg(command, "--checkpoint_interval") == "-1"
    assert _arg(command, "--skip_final_checkpoint") == "true"
    assert _arg(command, "--load_params_only") == "false"


def test_wrong_source_step_or_run_is_rejected_before_torchrun():
    cases = (
        (
            {"RESUME_STEP": "9998"},
            "step+1 divisible by 200",
        ),
        (
            {"RESUME_RUN_DIR": "/tmp/not-the-locked-v10-run"},
            "locks RESUME_RUN_DIR",
        ),
    )
    for overrides, message in cases:
        result = _run(check=False, **overrides)
        assert result.returncode == 2
        assert message in result.stderr
        assert "torchrun " not in result.stdout


def test_inherited_migration_or_checkpoint_controls_are_rejected():
    for variable_name, value in (
        ("PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION", "future_migration"),
        ("PROTENIX_TRAINING_CONTRACT_ID", "future_contract"),
        ("RESUME_RAW_CHECKPOINT", "/tmp/other.pt"),
        ("LOAD_CHECKPOINT_PATH", "/tmp/other.pt"),
    ):
        result = _run(check=False, **{variable_name: value})
        assert result.returncode == 2
        assert f"rejects inherited {variable_name}" in result.stderr
        assert "torchrun " not in result.stdout


def test_run_name_must_declare_the_contract_boundary():
    result = _run(
        check=False,
        RUN_NAME="protenix_disordered_v4_v9_straggler_test",
    )

    assert result.returncode == 2
    assert "RUN_NAME must contain v10_to_v11" in result.stderr
    assert "torchrun " not in result.stdout


def test_migration_still_requires_the_original_16_by_8_topology():
    result = _run(
        check=False,
        MLP_WORKER_NUM="1",
        NPROC_PER_NODE="8",
    )

    assert result.returncode == 2
    assert "requires topology=16x8" in result.stderr
    assert "torchrun " not in result.stdout


def test_migration_uses_locked_node_count_when_platform_count_is_absent():
    result = _run(MLP_WORKER_NUM="")
    command = _command(result)

    assert _arg(command, "--nnodes") == "16"
