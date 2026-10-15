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

"""CPU-only launcher expansion tests; no checkpoint tensors or GPUs are read."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SOURCE_REPO = REPO.parent / (
    "Protenix_train2_templatecache_top60_uniform20_id90_softcluster_g02_"
    "msaindependent_only"
)
SOURCE_RUN = (
    "/mnt/project_abd/klp/output/initial_checkpoints/"
    "stage3_nopeptide_v2_from47999"
)
RUN_NAME = "protenix_v4_crop640_stage3_16x8_acc4_nopeptide_v2_from67999"
SCHEDULE = (
    "/mnt/project_abd/klp/output/soft_cluster_schedules/"
    "stage3_nopeptide_v2_fresh67999_to169999_g02"
)


def _run(
    output_dir: Path,
    *,
    source: bool = False,
    **overrides: str,
) -> subprocess.CompletedProcess[str]:
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
            "BASE_DIR": str(output_dir),
        }
    )
    repo = SOURCE_REPO if source else REPO
    launcher = "resume_step47199_msaindependent_only.sh" if source else "train.sh"
    if source:
        env["RUN_NAME"] = (
            "protenix_v4_from47199_msaindependent_only_"
            "softcluster_g02_v9_straggler_test"
        )
    env.update(overrides)
    result = subprocess.run(
        ["bash", str(repo / launcher)],
        cwd=repo,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _command(result: subprocess.CompletedProcess[str]) -> list[str]:
    lines = [line for line in result.stdout.splitlines() if line.startswith("torchrun ")]
    assert len(lines) == 1, result.stdout
    return shlex.split(lines[0])


def _arg(command: list[str], name: str) -> str:
    values = [command[i + 1] for i, token in enumerate(command) if token == name]
    assert values, name
    assert len(set(values)) == 1, (name, values)
    return values[-1]


def _training_args(command: list[str]) -> dict[str, tuple[str, ...]]:
    tokens = command[command.index("./runner/train.py") + 1 :]
    assert len(tokens) % 2 == 0
    args: dict[str, tuple[str, ...]] = {}
    for option, value in zip(tokens[::2], tokens[1::2]):
        assert option.startswith("--"), option
        args[option] = (*args.get(option, ()), value)
    return args


@pytest.fixture(scope="module")
def default_launch(tmp_path_factory: pytest.TempPathFactory):
    return _run(tmp_path_factory.mktemp("crop640_launcher"))


def test_default_crop640_uses_requested_68999_checkpoint_and_chunk_settings(default_launch):
    command = _command(default_launch)
    assert "topology=16x8 world_size=128" in default_launch.stdout
    for option, value in {
        "--train_crop_size": "640",
        "--iters_to_accumulate": "4",
        "--distributed_strategy": "ddp",
        "--data.epoch_size": "10240",
        "--diffusion_batch_size": "48",
        "--diffusion_chunk_size": "48",
        "--ema_shadow_device": "model",
        "--ema_decay": "0.999",
        "--model.pairformer.blocks_per_ckpt": "4",
        "--model.msa_module.blocks_per_ckpt": "1",
        "--model.diffusion_module.blocks_per_ckpt": "1",
        "--model.pairformer.train_pair_transition_chunk_size": "128",
        "--model.diffusion_module.train_pair_transition_chunk_size": "128",
        "--model.pairformer.train_pair_transition_checkpoint_chunks": "true",
        "--model.diffusion_module.train_pair_transition_checkpoint_chunks": "true",
        "--model.diffusion_module.use_fine_grained_checkpoint": "true",
        "--model.pairformer.checkpoint_cpu_offload_every_n": "null",
        "--model.msa_module.checkpoint_cpu_offload_every_n": "null",
        "--enable_diffusion_shared_vars_cache": "true",
        "--data.num_dl_workers": "8",
        "--data.train_dl_prefetch_factor": "2",
    }.items():
        assert _arg(command, option) == value


def test_default_47999_loads_full_state_without_resetting_scheduler(default_launch):
    command = _command(default_launch)
    assert "resume=full checkpoint=67999" in default_launch.stdout
    assert "new_data_start_step=68000" in default_launch.stdout
    assert "state=model+optimizer+scheduler+step+scaler+rank_rng+matched_ema" in default_launch.stdout
    assert "sampler=new schedule starts fresh once; matching schedule restores its saved cursor" in default_launch.stdout
    for option, value in {
        "--load_checkpoint_path": f"{SOURCE_RUN}/checkpoints/67999.pt",
        "--load_ema_checkpoint_path": f"{SOURCE_RUN}/checkpoints/67999_ema_0.999.pt",
        "--load_params_only": "false",
        "--load_strict": "true",
        "--skip_load_optimizer": "false",
        "--skip_load_scheduler": "false",
        "--skip_load_step": "false",
        "--load_step_for_scheduler": "false",
        "--lr": "0.001",
        "--lr_scheduler": "af3",
        "--warmup_steps": "1000",
        "--decay_every_n_steps": "25000",
        "--af3_lr_scheduler.decay_factor": "0.95",
        "--max_steps": "170000",
        "--checkpoint_interval": "50",
        "--data.train_sampler.soft_cluster_schedule_manifest": SCHEDULE,
        "--data.train_sampler.soft_cluster_gamma": "0.2",
    }.items():
        assert _arg(command, option) == value


def test_bash_source_run_and_step_select_an_unbound_pair(tmp_path: Path):
    source_run = tmp_path / "another_source"
    result = _run(tmp_path / "output", RESUME_RUN_DIR=str(source_run), RESUME_STEP="68199")
    command = _command(result)
    assert not source_run.exists()
    assert "checkpoint=68199" in result.stdout
    assert _arg(command, "--load_checkpoint_path") == str(source_run / "checkpoints/68199.pt")
    assert _arg(command, "--load_ema_checkpoint_path") == str(source_run / "checkpoints/68199_ema_0.999.pt")


def test_bash_explicit_raw_and_ema_paths_are_not_replaced(tmp_path: Path):
    raw = tmp_path / "selected_raw" / "68399.pt"
    ema = tmp_path / "selected_ema" / "68399_ema_0.999.pt"
    command = _command(
        _run(
            tmp_path / "output",
            RESUME_STEP="68399",
            RESUME_RUN_DIR=str(tmp_path / "different_run"),
            RESUME_RAW_CHECKPOINT=str(raw),
            RESUME_EMA_CHECKPOINT=str(ema),
        )
    )
    assert not raw.exists() and not ema.exists()
    assert _arg(command, "--load_checkpoint_path") == str(raw)
    assert _arg(command, "--load_ema_checkpoint_path") == str(ema)


@pytest.mark.parametrize("crop_size", ["384", "512"])
def test_bash_crop_override_reaches_final_training_cli(tmp_path: Path, crop_size: str):
    result = _run(tmp_path / "output", TRAIN_CROP_SIZE=crop_size)
    assert _arg(_command(result), "--train_crop_size") == crop_size
    assert f"crop={crop_size} accumulation=4" in result.stdout


def test_custom_run_name_does_not_require_the_old_v9_tag(tmp_path: Path):
    result = _run(tmp_path / "output", RUN_NAME="my_crop640_experiment")
    assert _arg(_command(result), "--run_name") == "my_crop640_experiment"


def test_final_scientific_cli_preserves_47999_settings_outside_authorized_changes(
    tmp_path: Path, default_launch
):
    if not SOURCE_REPO.is_dir():
        pytest.skip("historical producer repository is not mounted")
    source = _run(tmp_path / "source_output", source=True)
    old = _training_args(_command(source))
    new = _training_args(_command(default_launch))
    changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    allowed = {
        "--train_crop_size",
        "--run_name",
        "--base_dir",
        "--load_checkpoint_path",
        "--load_ema_checkpoint_path",
        "--max_steps",
        "--checkpoint_interval",
        "--data.train_sets",
        "--data.train_sampler.train_sample_weights",
        "--data.train_sampler.soft_cluster_schedule_manifest",
        "--model.pairformer.blocks_per_ckpt",
        "--model.msa_module.blocks_per_ckpt",
        "--model.diffusion_module.blocks_per_ckpt",
        "--model.pairformer.train_pair_transition_chunk_size",
        "--model.diffusion_module.train_pair_transition_chunk_size",
        "--model.pairformer.train_pair_transition_checkpoint_chunks",
        "--model.diffusion_module.train_pair_transition_checkpoint_chunks",
    }
    allowed.update(
        key for key in new
        if key.startswith(("--data.sabdab-train.", "--data.distill-sabdab."))
    )
    assert changed == allowed, {key: (old.get(key), new.get(key)) for key in changed}
    assert set(old["--train_crop_size"]) == {"384"}
    assert set(new["--train_crop_size"]) == {"640"}


def test_sabdab_mixture_is_normalized_and_preserves_old_child_ratios(default_launch):
    command = _command(default_launch)
    sets = _arg(command, "--data.train_sets").split(",")
    weights = [float(value) for value in _arg(command, "--data.train_sampler.train_sample_weights").split(",")]
    assert sets[-2:] == ["sabdab-train", "distill-sabdab"]
    assert len(sets) == 6
    assert sum(weights) == pytest.approx(1.0)
    assert weights == pytest.approx([value / 1.05 for value in [0.49, 0.005, 0.505, 0.02, 0.02, 0.01]])
    assert "release_le_20210930" in _arg(command, "--data.sabdab-train.base_info.indices_fpath")
    assert _arg(command, "--data.distill-sabdab.base_info.indices_fpath").endswith("distill-sabdab_exact_antibody_interface_v1.csv")
    for dataset in sets[-2:]:
        assert _arg(command, f"--data.{dataset}.cropping_configs.max_nuc_tokens") == "256"
    for dataset in sets[:4]:
        assert _arg(command, f"--data.{dataset}.cropping_configs.max_nuc_tokens") == "128"


def _checkpoint_pair(run_dir: Path, step: int, *, ema: bool = True) -> None:
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    (checkpoints / f"{step}.pt").write_bytes(b"raw-fixture")
    if ema:
        (checkpoints / f"{step}_ema_0.999.pt").write_bytes(b"ema-fixture")


def test_restart_selects_latest_complete_pair_in_own_run(tmp_path: Path):
    output = tmp_path / "output"
    first_run = output / f"{RUN_NAME}_20260910_100000"
    second_run = output / f"{RUN_NAME}_20260910_120000"
    other_run = output / "unrelated_run_20260910_130000"
    _checkpoint_pair(first_run, 68199)
    _checkpoint_pair(second_run, 68399)
    _checkpoint_pair(second_run, 68599, ema=False)
    _checkpoint_pair(other_run, 99999)
    result = _run(output)
    command = _command(result)
    assert _arg(command, "--load_checkpoint_path") == str(second_run / "checkpoints/68399.pt")
    assert "checkpoint=68399" in result.stdout


def test_explicit_bash_pair_wins_over_own_latest(tmp_path: Path):
    output = tmp_path / "output"
    _checkpoint_pair(output / f"{RUN_NAME}_20260910_100000", 68799)
    selected = tmp_path / "selected"
    result = _run(output, RESUME_RUN_DIR=str(selected), RESUME_STEP="68199")
    assert _arg(_command(result), "--load_checkpoint_path") == str(selected / "checkpoints/68199.pt")


def test_bash_start_step_zero_and_target_budget_are_not_bound_to_filename(tmp_path: Path):
    result = _run(tmp_path / "output", START_STEP="0", RESUME_STEP="47999", MAX_STEPS="100")
    command = _command(result)
    assert "checkpoint=47999" in result.stdout
    assert "new_data_start_step=0" in result.stdout
    assert _arg(command, "--max_steps") == "100"
    assert _arg(command, "--load_checkpoint_path").endswith("/checkpoints/47999.pt")


def test_smoke_length_is_deferred_until_trainer_resolves_start_step(tmp_path: Path):
    result = _run(tmp_path / "output", RESUME_SMOKE_STEPS="2")
    command = _command(result)
    assert _arg(command, "--max_steps") == "170000"
    assert _arg(command, "--skip_load_optimizer") == "false"
    assert _arg(command, "--checkpoint_interval") == "-1"
    assert _arg(command, "--skip_final_checkpoint") == "true"
    assert "smoke_updates=2" in result.stdout


def test_inherited_expected_step_does_not_bind_filename_to_payload(tmp_path: Path):
    result = _run(
        tmp_path / "output",
        RESUME_STEP="67999",
        PROTENIX_RESUME_CHECKPOINT_STEP="12345",
    )
    command = _command(result)
    assert _arg(command, "--load_checkpoint_path").endswith("/67999.pt")
    assert _arg(command, "--load_strict") == "true"
    assert _arg(command, "--skip_load_step") == "false"
    assert "checkpoint_step=12345" not in result.stdout


def test_bash_sabdab_weights_and_paths_are_honored(tmp_path: Path):
    result = _run(
        tmp_path / "output",
        PROTENIX_SABDAB_TRAIN_SAMPLE_WEIGHT="0.03",
        PROTENIX_DISTILL_SABDAB_SAMPLE_WEIGHT="0.02",
        SABDAB_ROOT_DIR="/chosen/sabdab",
        SABDAB_TRAIN_INDICES_FPATH="/chosen/train.csv",
        DISTILL_SABDAB_INDICES_FPATH="/chosen/distill.csv",
    )
    command = _command(result)
    weights = [float(value) for value in _arg(command, "--data.train_sampler.train_sample_weights").split(",")]
    assert weights == pytest.approx([value / 1.07 for value in [0.49, 0.005, 0.505, 0.02, 0.03, 0.02]])
    assert _arg(command, "--data.sabdab-train.base_info.indices_fpath") == "/chosen/train.csv"
    assert _arg(command, "--data.distill-sabdab.base_info.indices_fpath") == "/chosen/distill.csv"
    assert _arg(command, "--data.sabdab-train.msa.prot_msadir_raw_paths") == "/chosen/sabdab/mmcif_msa_template"
