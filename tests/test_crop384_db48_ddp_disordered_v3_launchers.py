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

"""Contracts for the four-source crop384 disordered-v3 launcher."""

from __future__ import annotations

import copy
import os
import shlex
import subprocess
from collections.abc import Mapping
from pathlib import Path

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_model_type import model_configs
from protenix.config.config import parse_configs


REPO_DIR = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_DIR / "run_crop384_db48_ddp_disordered_v3.sh"
RESUME_LAUNCHER = REPO_DIR / "resume_crop384_db48_ddp_disordered_v3.sh"
PROFILE = "weighted_pdb_afdb_mgnify_short_long_disordered_v3"
PDB = "weightedPDB_before210930_v20260101"
SHORT = "afdb_mgnify_short_distillation_v20260725"
LONG = "afdb_mgnify_long_distillation_v20260725"
DISORDERED = "openfold3_disordered_v20260725"
TRAIN_SETS = f"{PDB},{SHORT},{LONG},{DISORDERED}"


def _environment() -> dict[str, str]:
    env = {
        name: os.environ[name]
        for name in ("HOME", "PATH", "LANG", "LC_ALL")
        if name in os.environ
    }
    env.update(
        {
            "DRY_RUN": "1",
            "SKIP_PATH_CHECK": "1",
            "BASE_DIR": "/tmp/protenix-disordered-v3-launcher-test",
            "RUN_NAME": (
                "crop384_db48_ddp_disordered_v3_v9_straggler_test"
            ),
            "MLP_WORKER_NUM": "16",
            "MLP_ROLE_INDEX": "0",
            "MLP_WORKER_0_HOST": "worker-0",
            "MLP_WORKER_0_PORT": "23456",
            "NPROC_PER_NODE": "8",
            "PROTENIX_SHORT_DISTILLATION_RUNTIME_INDEX_DIR": (
                "/tmp/protenix-short-runtime-test"
            ),
            "PROTENIX_LONG_DISTILLATION_RUNTIME_INDEX_DIR": (
                "/tmp/protenix-long-runtime-test"
            ),
            "PROTENIX_OPENFOLD3_DISORDERED_ROOT_DIR": (
                "/tmp/protenix-disordered-v3-test"
            ),
            # The locked wrapper must replace both stale selectors.
            "PROTENIX_LOCKED_DATA_PROFILE": "weighted_pdb_only_v1",
            "PROTENIX_TRAINING_DATA_PROFILE": (
                "weighted_pdb_afdb_mgnify_disordered_v2"
            ),
        }
    )
    return env


def _run(
    env: dict[str, str],
    *,
    launcher: Path = LAUNCHER,
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
        line
        for line in result.stdout.splitlines()
        if line.startswith("torchrun ")
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
        arg_str=" ".join(shlex.quote(value) for value in train_args),
        fill_required_with_null=True,
    )


def test_v3_dataset_registration_uses_distinct_runtime_partitions():
    short = data_configs[SHORT]
    long = data_configs[LONG]

    assert short is not long
    assert short["base_info"]["is_distillation"] is True
    assert long["base_info"]["is_distillation"] is True
    assert short["base_info"]["require_resolved_protein_crop"] is True
    assert long["base_info"]["require_resolved_protein_crop"] is True
    assert short["base_info"]["min_diffusion_supervised_tokens"] == 4
    assert long["base_info"]["min_diffusion_supervised_tokens"] == 4
    assert (
        short["base_info"]["runtime_index_dir"]
        != long["base_info"]["runtime_index_dir"]
    )
    for dataset in (short, long):
        assert dataset["sampler_configs"]["sampler_type"] == "clustered"
        assert dataset["msa"]["max_input_unpaired_msa_depth"] == 16384
        assert "max_input_paired_msa_depth" not in dataset["msa"]
        assert dataset["augmentation"]["query_only_msa_probability"] == 0.1
        assert (
            dataset["augmentation"][
                "pair_msa_as_unpaired_probability"
            ]
            == 0.3
        )


def test_v3_scratch_launcher_locks_four_source_contract():
    result = _run(_environment())
    command = _command(result)

    assert _argument(command, "--data.train_sets") == TRAIN_SETS
    assert (
        _argument(
            command,
            "--data.train_sampler.train_sample_weights",
        )
        == "0.5,0.005,0.495,0.02"
    )
    assert _argument(
        command,
        "--data.query_only_msa_probability",
    ) == "0.1"
    assert _argument(
        command,
        "--data.pair_msa_as_unpaired_probability",
    ) == "0.3"
    assert _argument(
        command,
        "--data.msa.enable_multichain_msa_shuffle_variant_sampling",
    ) == "true"
    assert _argument(
        command,
        "--data.msa.multichain_msa_shuffle_segment_probability",
    ) == "0.5"
    for dataset_name in (SHORT, LONG, DISORDERED):
        assert _argument(
            command,
            (
                f"--data.{dataset_name}.augmentation."
                "query_only_msa_probability"
            ),
        ) == "0.1"
        assert _argument(
            command,
            (
                f"--data.{dataset_name}.augmentation."
                "pair_msa_as_unpaired_probability"
            ),
        ) == "0.3"
    assert f"data_profile={PROFILE}" in result.stdout
    assert "weights=0.5,0.005,0.495,0.02" in result.stdout
    assert (
        "MSA mode_probabilities=query_only:0.1 "
        "unpaired_shuffle:0.3 species_paired:0.6"
    ) in result.stdout
    assert "multichain_segment50_whole_physical_chain50" in result.stdout
    assert "msamix_q10_u30_p60" in result.stdout
    assert "msaraw_p10000_u10000_distillu16384_final16384" in result.stdout
    assert (
        "MSA raw_prefix_cap_train AFDB/MGnify short/long "
        "direct-unpaired=16384 paired-source=10000(unused)"
    ) in result.stdout
    for dataset_name in (SHORT, LONG):
        assert _argument(
            command,
            f"--data.{dataset_name}.msa.max_input_unpaired_msa_depth",
        ) == "16384"

    resolved = _resolved_config(command)
    assert list(resolved.data.train_sets) == TRAIN_SETS.split(",")
    assert list(resolved.data.train_sampler.train_sample_weights) == [
        0.5,
        0.005,
        0.495,
        0.02,
    ]
    assert resolved.data.msa.enable_virtual_segment_msa_shuffle is True
    assert (
        resolved.data.msa.enable_multichain_msa_shuffle_variant_sampling
        is True
    )
    assert (
        resolved.data.msa.multichain_msa_shuffle_segment_probability
        == 0.5
    )
    assert resolved.data.msa.enable_crop_aware_partial_query_only is True
    assert resolved.data.msa.partial_query_only_max_chains == 2


def test_v3_resume_launcher_locks_the_same_four_source_contract():
    env = _environment()
    env.update(
        {
            "RUN_NAME": (
                "crop384_db48_ddp_disordered_v3_v9_straggler_resume_test"
            ),
            "RESUME_RUN_DIR": "/tmp/protenix-disordered-v3-source",
            "RESUME_STEP": "4999",
            "RESUME_NNODES": "16",
            "RESUME_NPROC_PER_NODE": "8",
            "RESUME_NODE_RANK": "0",
            "RESUME_MASTER_ADDR": "resume-worker-0",
            "RESUME_MASTER_PORT": "29641",
        }
    )
    result = _run(env, launcher=RESUME_LAUNCHER)
    command = _command(result)

    assert _argument(command, "--data.train_sets") == TRAIN_SETS
    assert _argument(
        command,
        "--data.train_sampler.train_sample_weights",
    ) == "0.5,0.005,0.495,0.02"
    assert _argument(
        command,
        "--load_checkpoint_path",
    ) == "/tmp/protenix-disordered-v3-source/checkpoints/4999.pt"
    assert _argument(
        command,
        "--load_ema_checkpoint_path",
    ).endswith("/checkpoints/4999_ema_0.999.pt")
    assert command.count("--load_params_only") == 1
    assert _argument(command, "--load_params_only") == "false"
    assert _argument(command, "--skip_load_optimizer") == "false"
    assert _argument(command, "--skip_load_scheduler") == "false"
    assert _argument(command, "--skip_load_step") == "false"
    assert "resume=full checkpoint_step=4999 next_step=5000" in result.stdout
    assert "fail_closed=true" in result.stdout
    assert f"data_profile={PROFILE}" in result.stdout
    assert "msamix_q10_u30_p60" in result.stdout
    assert "msaraw_p10000_u10000_distillu16384_final16384" in result.stdout
    for dataset_name in (SHORT, LONG):
        assert _argument(
            command,
            f"--data.{dataset_name}.msa.max_input_unpaired_msa_depth",
        ) == "16384"


def test_v3_launcher_requires_versioned_run_name():
    env = _environment()
    env["RUN_NAME"] = "crop384_db48_v9_straggler_wrong_profile"
    result = _run(env, check=False)

    assert result.returncode == 2
    assert "RUN_NAME must contain disordered_v3" in result.stderr
