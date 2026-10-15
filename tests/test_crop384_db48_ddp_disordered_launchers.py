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

"""Contracts for the crop384 OpenFold3-disordered training wrappers."""

from __future__ import annotations

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
SCRATCH_LAUNCHER = REPO_DIR / "run_crop384_db48_ddp_disordered_v2.sh"
RESUME_LAUNCHER = REPO_DIR / "resume_crop384_db48_ddp_disordered_v2.sh"
PROFILE_NAME = "weighted_pdb_afdb_mgnify_disordered_v2"
PDB_DATASET = "weightedPDB_before210930_v20260101"
PREDICTED_DATASET = "afdb_mgnify_distillation_v20260725"
DISORDERED_DATASET = "openfold3_disordered_v20260725"
TRAIN_SETS = f"{PDB_DATASET},{PREDICTED_DATASET},{DISORDERED_DATASET}"
RUNTIME_INDEX_DIR = "/tmp/protenix-disordered-runtime-index-test"
DISORDERED_ROOT = "/tmp/protenix-openfold3-disordered-test"


def _minimal_environment() -> dict[str, str]:
    """Build a deterministic shell environment without stale training knobs."""
    env = {
        name: os.environ[name]
        for name in ("HOME", "PATH", "LANG", "LC_ALL")
        if name in os.environ
    }
    env.update(
        {
            "DRY_RUN": "1",
            "SKIP_PATH_CHECK": "1",
            "BASE_DIR": "/tmp/protenix_crop384_disordered_launcher_test",
            "PROTENIX_DISTILLATION_RUNTIME_INDEX_DIR": RUNTIME_INDEX_DIR,
            "PROTENIX_OPENFOLD3_DISORDERED_ROOT_DIR": DISORDERED_ROOT,
            # Both wrappers must replace stale profile selections.
            "PROTENIX_LOCKED_DATA_PROFILE": "weighted_pdb_only_v1",
            "PROTENIX_TRAINING_DATA_PROFILE": (
                "weighted_pdb_afdb_mgnify_distill_v1"
            ),
        }
    )
    return env


def _scratch_environment() -> dict[str, str]:
    env = _minimal_environment()
    env.update(
        {
            "MLP_WORKER_NUM": "16",
            "MLP_ROLE_INDEX": "0",
            "MLP_WORKER_0_HOST": "scratch-worker-0",
            "MLP_WORKER_0_PORT": "23456",
            "NPROC_PER_NODE": "8",
            "RUN_NAME": (
                "crop384_db48_ddp_disordered_v2_v9_straggler_test"
            ),
            "run_name": "stale_lowercase_run_name",
        }
    )
    return env


def _resume_environment() -> dict[str, str]:
    env = _minimal_environment()
    env.update(
        {
            "RUN_NAME": (
                "crop384_db48_ddp_disordered_v2_v9_straggler_resume_test"
            ),
            "run_name": "stale_lowercase_resume_name",
            "RESUME_RUN_DIR": "/tmp/protenix_disordered_source_run",
            "RESUME_STEP": "4999",
            "RESUME_NNODES": "16",
            "RESUME_NPROC_PER_NODE": "8",
            "RESUME_NODE_RANK": "0",
            "RESUME_MASTER_ADDR": "resume-worker-0",
            "RESUME_MASTER_PORT": "29641",
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
        arg_str=" ".join(shlex.quote(argument) for argument in train_args),
        fill_required_with_null=True,
    )


def _assert_disordered_profile_command(
    result: subprocess.CompletedProcess[str],
    *,
    expected_run_name: str,
    expected_weights: str,
    expected_disable_non_protein_diffusion_weights: str = "true",
) -> None:
    command = _command(result)
    assert _argument(command, "--run_name") == expected_run_name
    assert _argument(command, "--data.train_sets") == TRAIN_SETS
    assert (
        _argument(
            command,
            "--data.train_sampler.train_sample_weights",
        )
        == expected_weights
    )
    assert _argument(
        command,
        f"--data.{PREDICTED_DATASET}.base_info.runtime_index_dir",
    ) == RUNTIME_INDEX_DIR
    assert _argument(
        command,
        "--data.msa.enable_virtual_segment_msa_shuffle",
    ) == "true"
    assert _argument(
        command,
        "--data.msa.virtual_segment_residue_gap",
    ) == "24"
    assert _argument(
        command,
        "--data.msa.virtual_segment_min_length",
    ) == "9"
    assert _argument(
        command,
        "--data.msa.enable_crop_aware_partial_query_only",
    ) == "true"
    assert _argument(
        command,
        "--data.msa.partial_query_only_max_chains",
    ) == "2"
    for dataset_name in (PREDICTED_DATASET, DISORDERED_DATASET):
        assert _argument(
            command,
            (
                f"--data.{dataset_name}.augmentation."
                "pair_msa_as_unpaired_probability"
            ),
        ) == "0.3"
    assert _argument(
        command,
        (
            f"--data.{PREDICTED_DATASET}.base_info."
            "require_resolved_protein_crop"
        ),
    ) == "true"
    assert _argument(
        command,
        f"--data.{DISORDERED_DATASET}.base_info.indices_fpath",
    ) == f"{DISORDERED_ROOT}/protenix/indices.csv.gz"
    assert _argument(
        command,
        f"--data.{DISORDERED_DATASET}.base_info.dataset_success_fpath",
    ) == f"{DISORDERED_ROOT}/protenix/_SUCCESS.json"
    assert _argument(
        command,
        f"--data.{DISORDERED_DATASET}.base_info.dataset_manifest_fpath",
    ) == f"{DISORDERED_ROOT}/protenix/manifest.json"
    assert _argument(
        command,
        f"--data.{DISORDERED_DATASET}.base_info.mmcif_dir",
    ) == f"{DISORDERED_ROOT}/structures/structures"
    assert _argument(
        command,
        f"--data.{DISORDERED_DATASET}.base_info.bioassembly_dict_dir",
    ) == f"{DISORDERED_ROOT}/structures/structures"
    assert _argument(
        command,
        (
            f"--data.{DISORDERED_DATASET}.base_info."
            "disable_non_protein_diffusion_weights"
        ),
    ) == expected_disable_non_protein_diffusion_weights
    assert _argument(
        command,
        (
            f"--data.{DISORDERED_DATASET}.base_info."
            "require_resolved_protein_crop"
        ),
    ) == "true"
    for dataset_name in (
        "weightedPDB_before210930_v20260101",
        PREDICTED_DATASET,
        DISORDERED_DATASET,
    ):
        assert _argument(
            command,
            (
                f"--data.{dataset_name}.base_info."
                "min_diffusion_supervised_tokens"
            ),
        ) == "4"
    assert f"data_profile={PROFILE_NAME}" in result.stdout
    assert f"weights={expected_weights}" in result.stdout
    assert (
        "MSA unpaired_shuffle_branch=all_protein_rows "
        "crop_segments_gap24_min9 query_fixed occurrence_seeded"
    ) in result.stdout
    assert (
        "MSA query_only_branch=monomer_global "
        "complex_crop_physical_protein_asym_partial_max2_keep1 "
        "occurrence_seeded_given_crop"
    ) in result.stdout
    expected_mask_contract = (
        "proteinonlydiffusion_v1"
        if expected_disable_non_protein_diffusion_weights == "true"
        else "allmoldiffusion_v1"
    )
    assert (
        "data_weightedpdb_afdb_mgnify_disordered_v2_"
        f"{expected_mask_contract}_resolvedproteincrop_minloss4_v2_"
        "allmsasegshuffle_gap24_min9_v1_"
        "partialquery_max2_keep1_v1_arch"
    ) in result.stdout


def test_disordered_dataset_registration_locks_training_semantics():
    dataset = data_configs[DISORDERED_DATASET]
    base_info = dataset["base_info"]
    msa = dataset["msa"]
    configured_root = Path(
        os.environ.get(
            "PROTENIX_OPENFOLD3_DISORDERED_ROOT_DIR",
            "/mnt/project_abd_hs/klp/databases/pdb_disordered",
        )
    )

    assert base_info["bioassembly_relpath_column"] == "bioassembly_relpath"
    assert base_info["is_distillation"] is True
    assert base_info["disable_non_protein_diffusion_weights"] is True
    assert base_info["require_resolved_protein_crop"] is True
    assert base_info["min_diffusion_supervised_tokens"] == 4
    assert msa["openfold3_alignment_npz_dirs"].value == [
        str(configured_root / "alignment_arrays"),
        str(configured_root / "alignment_arrays" / "alignment_arrays"),
    ]
    assert msa["enable_prot_msa"] is True
    assert msa["enable_rna_msa"] is True


def test_disordered_scratch_wrapper_uses_default_two_percent_profile():
    env = _scratch_environment()
    result = _run(SCRATCH_LAUNCHER, env)

    _assert_disordered_profile_command(
        result,
        expected_run_name=env["RUN_NAME"],
        expected_weights="0.5,0.5,0.02",
    )
    resolved = _resolved_config(_command(result))
    assert (
        resolved.data[PREDICTED_DATASET].base_info
        .require_resolved_protein_crop
        is True
    )
    assert resolved.data.msa.enable_virtual_segment_msa_shuffle is True
    assert resolved.data.msa.virtual_segment_residue_gap == 24
    assert resolved.data.msa.virtual_segment_min_length == 9
    assert resolved.data.msa.enable_crop_aware_partial_query_only is True
    assert resolved.data.msa.partial_query_only_max_chains == 2
    assert (
        resolved.data[PREDICTED_DATASET].augmentation
        .pair_msa_as_unpaired_probability
        == 0.3
    )
    assert (
        resolved.data[DISORDERED_DATASET].augmentation
        .pair_msa_as_unpaired_probability
        == 0.3
    )
    assert (
        resolved.data[DISORDERED_DATASET].base_info.dataset_success_fpath
        == f"{DISORDERED_ROOT}/protenix/_SUCCESS.json"
    )
    assert "initialization=random" in result.stdout


def test_disordered_resume_wrapper_keeps_default_two_percent_profile():
    env = _resume_environment()
    result = _run(RESUME_LAUNCHER, env)

    _assert_disordered_profile_command(
        result,
        expected_run_name=env["RUN_NAME"],
        expected_weights="0.5,0.5,0.02",
    )
    assert "resume=full checkpoint_step=4999 next_step=5000" in result.stdout
    assert "fail_closed=true" in result.stdout


def test_disordered_profile_accepts_arbitrary_positive_outer_masses():
    env = _scratch_environment()
    env.update(
        {
            "PROTENIX_PDB_SAMPLE_WEIGHT": "1.25",
            "PROTENIX_DISTILLATION_SAMPLE_WEIGHT": "2",
            "PROTENIX_DISORDERED_SAMPLE_WEIGHT": "4.5",
        }
    )
    result = _run(SCRATCH_LAUNCHER, env)

    _assert_disordered_profile_command(
        result,
        expected_run_name=env["RUN_NAME"],
        expected_weights="1.25,2,4.5",
    )


def test_disordered_initial_launcher_rejects_all_molecule_diffusion():
    env = _scratch_environment()
    env["PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS"] = "false"
    result = _run(SCRATCH_LAUNCHER, env, check=False)

    assert result.returncode == 2
    assert (
        "initial-training launcher requires "
        "PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS=true"
        in result.stderr
    )
    assert "torchrun " not in result.stdout


def test_disordered_profile_rejects_invalid_diffusion_mask_switch():
    env = _scratch_environment()
    env["PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS"] = "yes"

    result = _run(SCRATCH_LAUNCHER, env, check=False)

    assert result.returncode == 2
    assert (
        "PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS "
        "must be 'true' or 'false'"
    ) in result.stderr
    assert "torchrun " not in result.stdout


@pytest.mark.parametrize(
    ("launcher", "environment_factory"),
    (
        (SCRATCH_LAUNCHER, _scratch_environment),
        (RESUME_LAUNCHER, _resume_environment),
    ),
)
def test_disordered_wrappers_require_profile_marker_in_run_name(
    launcher: Path,
    environment_factory,
):
    env = environment_factory()
    env["RUN_NAME"] = "crop384_db48_ddp_v9_straggler_without_marker"

    result = _run(launcher, env, check=False)

    assert result.returncode == 2
    assert "RUN_NAME must contain disordered_v2" in result.stderr
    assert "torchrun " not in result.stdout
