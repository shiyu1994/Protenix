from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "migrate_step27799_to_weightedpdb_msa_independent.sh"
SOURCE_RUN = (
    "/mnt/project_abd/klp/proteinx_output/"
    "protenix_v4_step21999_residual_ge200_softcluster_g02_"
    "v9_straggler_20260804_140441"
)
SCHEDULE = (
    "/mnt/project_abd/klp/proteinx_output/soft_cluster_schedules/"
    "step21999_residual_ge200_g02_v2"
)
SOURCE_CONTRACT = (
    "crop384_db48_ddp_disordered_v5_ws128_acc4_catalogreplacement_"
    "residual_ge200_20260804_softcluster_g02_checkpointbound_step21999_"
    "wave_schedule_v2"
)
TARGET_CONTRACT = (
    SOURCE_CONTRACT
    + "_weightedpdb_pairing_nonpairing_independent10k_v1"
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
                "protenix_v4_step27799_msaindependent_only_"
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


def _command(result: subprocess.CompletedProcess[str]) -> list[str]:
    line = next(
        line for line in result.stdout.splitlines() if line.startswith("torchrun ")
    )
    return shlex.split(line)


def _arg(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_launcher_locks_step27799_and_changes_only_weightedpdb_msa(tmp_path: Path):
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "source_step=27799 next_trainer_step=27800" in result.stdout
    assert (
        "source_scheduler_step=27800 source_lr=0.00095 "
        "restored_unchanged=true"
    ) in result.stdout
    assert (
        f"schedule={SCHEDULE} gamma=0.2 cursor_continuation=true "
        "unchanged=true"
    ) in result.stdout
    assert "PDB_member_schedule=parent_exact unchanged=true" in result.stdout
    assert (
        "weightedPDB_MSA=pairing.a3m_top10000+"
        "genuine_non_pairing.a3m_top10000 independent=true final<=16384"
        in result.stdout
    )
    assert "schedule_validation=skipped_only_for_explicit_dry_run" in result.stdout
    assert f"target_contract={TARGET_CONTRACT}" in result.stdout
    assert (
        "only_allowed_delta="
        "weightedpdb_independent_pairing_nonpairing_top10k_only_v1"
        in result.stdout
    )

    command = _command(result)
    assert _arg(command, "--load_checkpoint_path") == (
        f"{SOURCE_RUN}/checkpoints/27799.pt"
    )
    assert _arg(command, "--load_ema_checkpoint_path") == (
        f"{SOURCE_RUN}/checkpoints/27799_ema_0.999.pt"
    )
    assert _arg(command, "--load_params_only") == "false"
    assert _arg(command, "--skip_load_optimizer") == "false"
    assert _arg(command, "--skip_load_scheduler") == "false"
    assert _arg(command, "--skip_load_step") == "false"
    assert _arg(command, "--max_steps") == "50000"
    assert _arg(command, "--data.train_sampler.soft_cluster_gamma") == "0.2"
    assert _arg(
        command,
        "--data.train_sampler.soft_cluster_schedule_manifest",
    ) == SCHEDULE
    assert _arg(
        command,
        "--data.msa.independent_pairing_non_pairing_budget",
    ) == "true"
    assert _arg(command, "--lr") == "0.001"
    assert "0.0013" not in command


def test_launcher_binds_checkpoint_pair_mode_contract_and_parent_schedule():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "4cefb9af5088a4671b4287149c95efcd72cac37449136fe37b4e4d060754f94a" in text
    assert "b3ad325bde97f2b249a03c4004a2a00d704e92a3aa0be925b0705e1a23ee67c8" in text
    assert "d40d64151cc8a8ff17458c310ca0a53f40047d3b5d4adf5577db3d653ae1c8f2" in text
    assert "weightedpdb_independent_pairing_nonpairing_full_state_v1" in text
    assert "weightedpdb_independent_pairing_nonpairing_top10k_only_v1" in text
    assert "step21999_residual_ge200_g02_v2" in text
    assert "step27799_pdb_member_horizon_to50000_v1" not in text
    assert "0.0013" not in text
    assert "checkpointbound_step21999_wave_schedule_v2" in text
    assert "_weightedpdb_pairing_nonpairing_independent10k_v1" in text


def test_launcher_rejects_identity_and_state_overrides(tmp_path: Path):
    bad_name = _run(
        tmp_path,
        RUN_NAME="protenix_v4_step27799_softcluster_g02_v9_straggler_test",
    )
    assert bad_name.returncode == 2
    assert "RUN_NAME must contain msaindependent_only" in bad_name.stderr

    for variable_name, value in (
        ("RESUME_STEP", "26999"),
        ("PROTENIX_SOFT_CLUSTER_SCHEDULE_DIR", "/tmp/x"),
        ("INDEPENDENT_PAIRING_NON_PAIRING_BUDGET", "false"),
        ("RESUME_SMOKE_STEPS", "1"),
        ("CHECK_ONLY", "1"),
    ):
        result = _run(tmp_path, **{variable_name: value})
        assert result.returncode == 2
        assert f"rejects inherited {variable_name}" in result.stderr


def test_launcher_never_allows_path_bypass_for_real_training(tmp_path: Path):
    result = _run(tmp_path, DRY_RUN="0", SKIP_PATH_CHECK="1")
    assert result.returncode == 2
    assert "allowed only with DRY_RUN=1" in result.stderr
