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

import copy
import os
import shlex
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_model_type import model_configs
from protenix.config.config import parse_configs


REPO_DIR = Path(__file__).resolve().parents[1]
SCRATCH_LAUNCHER = REPO_DIR / "run_crop384_db48_ddp_distill_50k_v1.sh"
OLD_SCRATCH_LAUNCHER = REPO_DIR / "run_crop384_db48_ddp_scratch_50k.sh"
RESUME_LAUNCHER = REPO_DIR / "resume_crop384_db48_ddp_distill_50k_v1.sh"
PROFILE_REGISTRY = REPO_DIR / "configs" / "crop384_db48_data_profiles.sh"
DATASET_NAME = "afdb_mgnify_distillation_v20260725"
PROFILE_NAME = "weighted_pdb_afdb_mgnify_distill_v1"
RUNTIME_DIR = "/tmp/protenix-distillation-runtime-index-test"


def _clean_environment() -> dict[str, str]:
    env = os.environ.copy()
    for variable_name in (
        "CHECKPOINT",
        "checkpoint",
        "CHECKPOINT_PATH",
        "checkpoint_path",
        "LOAD_CHECKPOINT_PATH",
        "load_checkpoint_path",
        "LOAD_EMA_CHECKPOINT_PATH",
        "load_ema_checkpoint_path",
        "RESUME_RUN_DIR",
        "RESUME_STEP",
        "RESUME_RAW_CHECKPOINT",
        "RESUME_EMA_CHECKPOINT",
        "PROTENIX_RESUME_RAW_CHECKPOINT",
        "PROTENIX_RESUME_EMA_CHECKPOINT",
        "PROTENIX_RESUME_CHECKPOINT_STEP",
        "PROTENIX_RESUME_TARGET_MAX_STEPS",
        "PROTENIX_RESUME_SMOKE_STEPS",
        "PROTENIX_CROP384_STRICT_RESUME_INTERNAL",
        "PROTENIX_FULL_RESUME",
        "PROTENIX_STRICT_FULL_RESUME",
        "PROTENIX_LOCKED_DATA_PROFILE",
        "PROTENIX_TRAINING_DATA_PROFILE",
        "PROTENIX_EXTRA_TRAIN_DATA_REQUIRED_PATHS",
        "PROTENIX_PDB_SAMPLE_WEIGHT",
        "PROTENIX_DISTILLATION_SAMPLE_WEIGHT",
        "PROTENIX_DISTILLATION_RUNTIME_INDEX_DIR",
        "PROTENIX_SMOKE_TEST",
        "PROTENIX_SMOKE_UPDATES",
        "PROTENIX_CHECKPOINT_SOURCE_SMOKE",
        "PROTENIX_STRAGGLER_BENCHMARK_UPDATES",
        "MAX_STEPS",
        "max_steps",
        "NNODES",
        "nnodes",
        "NODE_RANK",
        "node_rank",
        "NPROC_PER_NODE",
        "nproc_per_node",
        "WORLD_SIZE",
        "RANK",
        "MLP_WORKER_NUM",
        "MLP_ROLE_INDEX",
        "MLP_WORKER_0_HOST",
        "MLP_WORKER_0_PORT",
        "SLURM_NNODES",
        "SLURM_NODEID",
        "MASTER_ADDR",
        "master_addr",
        "MASTER_PORT",
        "master_port",
        "RESUME_NNODES",
        "RESUME_NPROC_PER_NODE",
        "RESUME_NODE_RANK",
        "RESUME_MASTER_ADDR",
        "RESUME_MASTER_PORT",
    ):
        env.pop(variable_name, None)
    return env


def _scratch_environment(*, distillation: bool = True) -> dict[str, str]:
    env = _clean_environment()
    run_marker = "distill_v1_" if distillation else ""
    env.update(
        {
            "DRY_RUN": "1",
            "SKIP_PATH_CHECK": "1",
            "MLP_WORKER_NUM": "16",
            "MLP_ROLE_INDEX": "0",
            "MLP_WORKER_0_HOST": "scratch-worker-0",
            "MLP_WORKER_0_PORT": "23456",
            "NPROC_PER_NODE": "8",
            "RUN_NAME": f"crop384_db48_ddp_{run_marker}v9_straggler_test",
            "BASE_DIR": "/tmp/protenix_crop384_db48_ddp_distillation_test",
            "PROTENIX_DISTILLATION_RUNTIME_INDEX_DIR": RUNTIME_DIR,
        }
    )
    return env


def _resume_environment() -> dict[str, str]:
    env = _clean_environment()
    env.update(
        {
            "DRY_RUN": "1",
            "SKIP_PATH_CHECK": "1",
            "RUN_NAME": "crop384_distill_v1_strict_resume_v9_straggler_test",
            "BASE_DIR": "/tmp/protenix_crop384_distillation_resume_test",
            "RESUME_RUN_DIR": "/tmp/crop384_distillation_source_run",
            "RESUME_STEP": "4999",
            "RESUME_NNODES": "16",
            "RESUME_NPROC_PER_NODE": "8",
            "RESUME_NODE_RANK": "0",
            "RESUME_MASTER_ADDR": "resume-worker-0",
            "RESUME_MASTER_PORT": "29641",
            "PROTENIX_DISTILLATION_RUNTIME_INDEX_DIR": RUNTIME_DIR,
        }
    )
    return env


def _run(
    launcher: Path,
    env: dict[str, str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(launcher)],
        cwd=REPO_DIR,
        env=env,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _command(result: subprocess.CompletedProcess[str]) -> list[str]:
    command_line = next(
        line for line in result.stdout.splitlines() if line.startswith("torchrun ")
    )
    return shlex.split(command_line)


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _deep_update(target: dict, update: Mapping) -> None:
    for key, value in update.items():
        if (
            isinstance(value, Mapping)
            and key in target
            and isinstance(target[key], Mapping)
        ):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _resolved_config(command: list[str]):
    train_entry_index = command.index("./runner/train.py")
    train_args = command[train_entry_index + 1 :]
    base = copy.deepcopy({**configs_base, **{"data": data_configs}})
    _deep_update(base, copy.deepcopy(model_configs["protenix-v2"]))
    return parse_configs(
        base,
        arg_str=" ".join(shlex.quote(argument) for argument in train_args),
        fill_required_with_null=True,
    )


def test_historical_scratch_wrapper_keeps_the_pdb_only_default():
    result = _run(
        OLD_SCRATCH_LAUNCHER,
        _scratch_environment(distillation=False),
    )
    command = _command(result)

    assert _argument(command, "--data.train_sets") == (
        "weightedPDB_before210930_v20260101"
    )
    assert _argument(command, "--data.train_sampler.train_sample_weights") == "1.0"
    assert (
        f"--data.{DATASET_NAME}.base_info.runtime_index_dir" not in command
    )
    assert "data_profile=weighted_pdb_only_v1" in result.stdout
    assert "data_weightedpdb_only_v1_arch" in result.stdout


def test_distillation_scratch_wrapper_resolves_mixed_profile_and_default_weights():
    result = _run(SCRATCH_LAUNCHER, _scratch_environment())
    command = _command(result)
    config = _resolved_config(command)

    assert _argument(command, "--data.train_sets") == (
        "weightedPDB_before210930_v20260101,"
        "afdb_mgnify_distillation_v20260725"
    )
    assert _argument(command, "--data.train_sampler.train_sample_weights") == (
        "0.5,0.5"
    )
    assert _argument(
        command,
        f"--data.{DATASET_NAME}.base_info.runtime_index_dir",
    ) == RUNTIME_DIR
    assert _argument(
        command,
        f"--data.{DATASET_NAME}.base_info.source_roots_fpath",
    ) == f"{RUNTIME_DIR}/roots.json"
    assert _argument(
        command,
        f"--data.{DATASET_NAME}.cropping_configs.max_nuc_tokens",
    ) == "128"
    assert f"data_profile={PROFILE_NAME}" in result.stdout
    assert f"distillation_runtime_index={RUNTIME_DIR}" in result.stdout
    assert "data_weightedpdb_afdb_mgnify_distill_v1_arch" in result.stdout
    assert list(config.data.train_sets) == [
        "weightedPDB_before210930_v20260101",
        DATASET_NAME,
    ]
    assert list(config.data.train_sampler.train_sample_weights) == [0.5, 0.5]
    distillation_config = config.data[DATASET_NAME]
    assert distillation_config.base_info.runtime_index_dir == RUNTIME_DIR
    assert distillation_config.base_info.source_roots_fpath == (
        f"{RUNTIME_DIR}/roots.json"
    )
    assert distillation_config.base_info.is_distillation is True
    assert distillation_config.sampler_configs.sampler_type == "clustered"
    assert (
        distillation_config.augmentation.query_only_msa_probability
        == pytest.approx(0.2)
    )
    assert (
        distillation_config.augmentation.pair_msa_as_unpaired_probability
        == pytest.approx(0.0)
    )
    assert (
        distillation_config.augmentation.template_use_probability
        == pytest.approx(0.0)
    )
    assert distillation_config.template.enable_prot_template is False


def test_distillation_profile_accepts_custom_positive_decimal_outer_weights():
    env = _scratch_environment()
    env.update(
        {
            "PROTENIX_PDB_SAMPLE_WEIGHT": "0.75",
            "PROTENIX_DISTILLATION_SAMPLE_WEIGHT": "0.25",
        }
    )
    result = _run(SCRATCH_LAUNCHER, env)

    assert _argument(
        _command(result),
        "--data.train_sampler.train_sample_weights",
    ) == "0.75,0.25"
    assert "weights=0.75,0.25" in result.stdout


@pytest.mark.parametrize(
    ("variable_name", "value"),
    (
        ("PROTENIX_PDB_SAMPLE_WEIGHT", "0"),
        ("PROTENIX_DISTILLATION_SAMPLE_WEIGHT", "0.0"),
        ("PROTENIX_PDB_SAMPLE_WEIGHT", "-0.5"),
        ("PROTENIX_DISTILLATION_SAMPLE_WEIGHT", ".5"),
        ("PROTENIX_PDB_SAMPLE_WEIGHT", "1e-1"),
        ("PROTENIX_DISTILLATION_SAMPLE_WEIGHT", "nan"),
    ),
)
def test_distillation_profile_rejects_non_positive_or_non_decimal_weights(
    variable_name: str,
    value: str,
):
    env = _scratch_environment()
    env[variable_name] = value

    result = _run(SCRATCH_LAUNCHER, env, check=False)

    assert result.returncode == 2
    assert f"{variable_name} must be a positive decimal number" in result.stderr
    assert "torchrun " not in result.stdout


def test_distillation_profile_declares_every_runtime_index_artifact():
    env = _clean_environment()
    env["PROTENIX_DISTILLATION_RUNTIME_INDEX_DIR"] = RUNTIME_DIR
    script = (
        'source "$1"\n'
        f"resolve_crop384_db48_data_profile {PROFILE_NAME}\n"
        'printf "%s\\n" "${CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS}"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script, "bash", str(PROFILE_REGISTRY)],
        cwd=REPO_DIR,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.stdout.strip().split(":") == [
        RUNTIME_DIR,
        f"{RUNTIME_DIR}/manifest.json",
        f"{RUNTIME_DIR}/roots.json",
        f"{RUNTIME_DIR}/row_offsets.u64",
        f"{RUNTIME_DIR}/source_root_ids.u8",
        f"{RUNTIME_DIR}/cif_relpaths.bin",
        f"{RUNTIME_DIR}/sampling",
        f"{RUNTIME_DIR}/sampling/cluster_ids.u32",
        f"{RUNTIME_DIR}/sampling/first_row_ids.u64",
        f"{RUNTIME_DIR}/sampling/cluster_sizes.u32",
    ]


def test_distillation_resume_wrapper_keeps_profile_weights_and_contract():
    env = _resume_environment()
    env.update(
        {
            "PROTENIX_PDB_SAMPLE_WEIGHT": "0.6",
            "PROTENIX_DISTILLATION_SAMPLE_WEIGHT": "0.4",
        }
    )
    result = _run(RESUME_LAUNCHER, env)
    command = _command(result)

    assert _argument(command, "--data.train_sets") == (
        "weightedPDB_before210930_v20260101,"
        "afdb_mgnify_distillation_v20260725"
    )
    assert _argument(command, "--data.train_sampler.train_sample_weights") == (
        "0.6,0.4"
    )
    assert _argument(
        command,
        f"--data.{DATASET_NAME}.base_info.runtime_index_dir",
    ) == RUNTIME_DIR
    assert _argument(
        command,
        f"--data.{DATASET_NAME}.base_info.source_roots_fpath",
    ) == f"{RUNTIME_DIR}/roots.json"
    assert f"data_profile={PROFILE_NAME}" in result.stdout
    assert "weights=0.6,0.4" in result.stdout
    assert "data_weightedpdb_afdb_mgnify_distill_v1_arch" in result.stdout
    assert "fail_closed=true" in result.stdout


@pytest.mark.parametrize("launcher", (SCRATCH_LAUNCHER, RESUME_LAUNCHER))
def test_distillation_wrappers_require_an_unambiguous_run_name(launcher: Path):
    env = (
        _scratch_environment()
        if launcher == SCRATCH_LAUNCHER
        else _resume_environment()
    )
    env["RUN_NAME"] = "crop384_v9_straggler_without_profile_marker"

    result = _run(launcher, env, check=False)

    assert result.returncode == 2
    assert "RUN_NAME must contain distill_v1" in result.stderr
    assert "torchrun " not in result.stdout
