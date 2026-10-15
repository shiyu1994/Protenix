from __future__ import annotations

import copy
import os
import shlex
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_model_type import model_configs
from protenix.config.config import parse_configs
from scripts.runtime.validate_crop384_db48_ema_checkpoint import (
    EXPECTED_TRAINING_CONTRACT_ID,
    validate_checkpoint,
)


REPO_DIR = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_DIR / "eval_crop384_db48_ema_checkpoint.sh"
QUEUE = REPO_DIR / "run_crop384_db48_ema_eval_queue.sh"


def _deep_update(target, update):
    for key, value in update.items():
        if (
            isinstance(value, Mapping)
            and key in target
            and isinstance(target[key], Mapping)
        ):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _resolved_config(arguments):
    base = copy.deepcopy({**configs_base, **{"data": data_configs}})
    _deep_update(base, copy.deepcopy(model_configs["protenix-v2"]))
    return parse_configs(
        base,
        arg_str=" ".join(shlex.quote(argument) for argument in arguments),
        fill_required_with_null=True,
    )


def _environment(**updates):
    env = os.environ.copy()
    for variable_name in (
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
        "MASTER_ADDR",
        "master_addr",
        "MASTER_PORT",
        "master_port",
        "CHECKPOINT_PATH",
        "checkpoint_path",
        "LOAD_CHECKPOINT_PATH",
        "load_checkpoint_path",
        "LOAD_EMA_CHECKPOINT_PATH",
        "load_ema_checkpoint_path",
        "EMA_CHECKPOINT",
        "EMA_CHECKPOINT_PATH",
        "PROTENIX_FULL_RESUME",
        "PROTENIX_STRICT_FULL_RESUME",
    ):
        env.pop(variable_name, None)
    env.update(
        {
            "DRY_RUN": "1",
            "SKIP_PATH_CHECK": "1",
            "CUDA_VISIBLE_DEVICES": "7",
            "EMA_CHECKPOINT": "/tmp/source/checkpoints/4999_ema_0.999.pt",
            "RUN_NAME": "crop384_ema_eval_test",
            "EVAL_BASE_DIR": "/tmp/crop384_ema_eval_test",
        }
    )
    env.update({key: str(value) for key, value in updates.items()})
    return env


def test_ema_evaluator_is_single_gpu_fixed_and_architecture_matched():
    result = subprocess.run(
        ["bash", str(EVALUATOR)],
        cwd=REPO_DIR,
        env=_environment(),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    command_line = next(
        line for line in result.stdout.splitlines() if line.startswith("torchrun ")
    )
    command = shlex.split(command_line)
    train_index = command.index("./runner/train.py")
    cfg = _resolved_config(command[train_index + 1 :])

    assert command[command.index("--nnodes") + 1] == "1"
    assert command[command.index("--nproc_per_node") + 1] == "1"
    assert cfg.load_checkpoint_path == ""
    assert (
        cfg.load_ema_checkpoint_path
        == "/tmp/source/checkpoints/4999_ema_0.999.pt"
    )
    assert cfg.load_params_only is True
    assert cfg.load_strict is True
    assert cfg.eval_only is True
    assert cfg.eval_ema_only is True
    assert cfg.eval_interval == 0
    assert cfg.eval_at_end is False
    assert cfg.checkpoint_interval == -1
    assert cfg.skip_final_checkpoint is True
    assert cfg.log_interval == 25
    assert cfg.warmup_steps == 1000
    assert cfg.decay_every_n_steps == 25000
    assert cfg.af3_lr_scheduler.decay_factor == pytest.approx(0.95)
    assert cfg.deterministic_seed is True
    assert cfg.mc_dropout_apply_rate == 0
    assert cfg.data.test_ref_pos_augment is False
    assert list(cfg.data.test_sets) == ["posebusters_0925"]
    assert cfg.data.posebusters_0925.base_info.max_n_token == 512
    assert cfg.sample_diffusion.N_sample == 5
    assert cfg.sample_diffusion.N_step == 20
    assert cfg.model.N_cycle == 4
    assert cfg.n_blocks == 64
    assert cfg.c_s == 384
    assert cfg.c_z == 256
    assert cfg.model.msa_module.c_m == 128
    assert cfg.model.diffusion_module.c_token == 768
    assert cfg.diffusion_batch_size == 48
    assert cfg.diffusion_chunk_size == 48
    assert list(cfg.data.train_sets) == [
        "weightedPDB_before210930_v20260101"
    ]
    assert list(cfg.data.train_sampler.train_sample_weights) == [1.0]
    assert "checkpoint_step=4999 completed_updates=5000" in result.stdout
    assert f"contract={EXPECTED_TRAINING_CONTRACT_ID}" in result.stdout


@pytest.mark.parametrize(
    ("environment_update", "message"),
    [
        (
            {"EMA_CHECKPOINT": "/tmp/3999_ema_0.999.pt"},
            "locked to updates",
        ),
        (
            {"CUDA_VISIBLE_DEVICES": "0,1"},
            "must expose exactly one GPU",
        ),
    ],
)
def test_ema_evaluator_rejects_nonfixed_or_multigpu_inputs(
    environment_update,
    message,
):
    result = subprocess.run(
        ["bash", str(EVALUATOR)],
        cwd=REPO_DIR,
        env=_environment(**environment_update),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 2
    assert message in result.stderr
    assert "torchrun " not in result.stdout


def test_eval_queue_maps_optimizer_updates_to_zero_based_ema_files():
    result = subprocess.run(
        ["bash", str(QUEUE)],
        cwd=REPO_DIR,
        env=_environment(
            TRAIN_RUN_DIR="/tmp/formal_run",
            EVAL_UPDATES="5000,50000",
        ),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "checkpoints/4999_ema_0.999.pt" in result.stdout
    assert "checkpoints/49999_ema_0.999.pt" in result.stdout
    assert result.stdout.count("planned_ema_eval") == 2


def _checkpoint_payload(*, contract_id=EXPECTED_TRAINING_CONTRACT_ID):
    return {
        "format_version": 2,
        "checkpoint_kind": "ema",
        "model": {"module.test": torch.ones(1)},
        "optimizer": None,
        "scheduler": None,
        "grad_scaler": None,
        "step": 4999,
        "resume_contract": {
            "training_contract_id": contract_id,
            "run_id": "formal_run",
            "checkpoint_pair_id": "a" * 64,
            "step": 4999,
            "next_step": 5000,
            "iters_to_accumulate": 4,
            "world_size": 128,
            "distributed_strategy": "ddp",
            "dtype": "bf16",
            "ema_decay": 0.999,
            "data_profile_id": "weighted_pdb_only_v1",
            "model_architecture_contract": {
                "global": {"c_s": 384, "c_z": 256},
                "pairformer": {"n_blocks": 64, "c_s": 384, "c_z": 256},
                "diffusion_module": {
                    "c_token": 768,
                    "c_s": 384,
                    "c_z": 256,
                },
                "msa_module": {"c_m": 128, "c_z": 256},
            },
        },
    }


def test_checkpoint_preflight_accepts_v8_and_rejects_v7(tmp_path):
    checkpoint_path = tmp_path / "4999_ema_0.999.pt"
    torch.save(_checkpoint_payload(), checkpoint_path)
    summary = validate_checkpoint(checkpoint_path)
    assert summary["completed_updates"] == 5000
    assert summary["training_contract_id"] == EXPECTED_TRAINING_CONTRACT_ID

    torch.save(
        _checkpoint_payload(contract_id=EXPECTED_TRAINING_CONTRACT_ID[:-1] + "7"),
        checkpoint_path,
    )
    with pytest.raises(ValueError, match="training contract mismatch"):
        validate_checkpoint(checkpoint_path)
