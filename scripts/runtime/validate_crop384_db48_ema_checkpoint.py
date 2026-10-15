#!/usr/bin/env python3
"""Fail-closed preflight for the crop384/DB48 v8 EMA evaluation queue."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path

import torch


EXPECTED_TRAINING_CONTRACT_ID = (
    "crop384_db48_ddp_scratch_50k_ws128_acc4_data_"
    "weightedpdb_only_v1_arch_pf64_cs384_dct768_pfckpt8_c48_"
    "msamix_q20_u30_p50_v8"
)
EMA_FILENAME_PATTERN = re.compile(r"^(?P<step>[0-9]+)_ema_0[.]999[.]pt$")


def _load_checkpoint(path: Path) -> Mapping:
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except (TypeError, RuntimeError):
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    return checkpoint


def validate_checkpoint(path: Path) -> dict[str, object]:
    match = EMA_FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(
            "EMA checkpoint name must be <zero_based_step>_ema_0.999.pt"
        )
    filename_step = int(match.group("step"))
    completed_updates = filename_step + 1
    if completed_updates % 5000 != 0 or not 5000 <= completed_updates <= 50000:
        raise ValueError(
            "fixed validation accepts only updates 5000,10000,...,50000 "
            f"(got update {completed_updates})"
        )

    checkpoint = _load_checkpoint(path)
    if checkpoint.get("format_version") != 2:
        raise ValueError("EMA checkpoint must use format_version=2")
    if checkpoint.get("checkpoint_kind") != "ema":
        raise ValueError("checkpoint_kind must be 'ema'")
    if checkpoint.get("step") != filename_step:
        raise ValueError(
            "checkpoint step does not match its filename: "
            f"payload={checkpoint.get('step')!r}, filename={filename_step}"
        )
    model_state = checkpoint.get("model")
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("EMA checkpoint model state must be a non-empty mapping")

    contract = checkpoint.get("resume_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("EMA checkpoint is missing resume_contract")
    if contract.get("training_contract_id") != EXPECTED_TRAINING_CONTRACT_ID:
        raise ValueError(
            "training contract mismatch: "
            f"checkpoint={contract.get('training_contract_id')!r}, "
            f"expected={EXPECTED_TRAINING_CONTRACT_ID!r}"
        )

    scalar_expectations = {
        "step": filename_step,
        "next_step": completed_updates,
        "iters_to_accumulate": 4,
        "world_size": 128,
        "distributed_strategy": "ddp",
        "dtype": "bf16",
        "ema_decay": 0.999,
        "data_profile_id": "weighted_pdb_only_v1",
    }
    mismatches = {
        key: {"checkpoint": contract.get(key), "expected": expected}
        for key, expected in scalar_expectations.items()
        if contract.get(key) != expected
    }
    architecture = contract.get("model_architecture_contract")
    if not isinstance(architecture, Mapping):
        mismatches["model_architecture_contract"] = {
            "checkpoint": type(architecture).__name__,
            "expected": "mapping",
        }
    else:
        global_architecture = architecture.get("global", {})
        pairformer = architecture.get("pairformer", {})
        diffusion = architecture.get("diffusion_module", {})
        msa = architecture.get("msa_module", {})
        architecture_expectations = {
            "global.c_s": (global_architecture, "c_s", 384),
            "global.c_z": (global_architecture, "c_z", 256),
            "pairformer.n_blocks": (pairformer, "n_blocks", 64),
            "pairformer.c_s": (pairformer, "c_s", 384),
            "pairformer.c_z": (pairformer, "c_z", 256),
            "diffusion_module.c_token": (diffusion, "c_token", 768),
            "diffusion_module.c_s": (diffusion, "c_s", 384),
            "diffusion_module.c_z": (diffusion, "c_z", 256),
            "msa_module.c_m": (msa, "c_m", 128),
            "msa_module.c_z": (msa, "c_z", 256),
        }
        for label, (section, key, expected) in architecture_expectations.items():
            observed = section.get(key) if isinstance(section, Mapping) else None
            if observed != expected:
                mismatches[label] = {
                    "checkpoint": observed,
                    "expected": expected,
                }
    if mismatches:
        raise ValueError(
            "EMA checkpoint does not match the crop384/DB48 v8 contract: "
            + json.dumps(mismatches, sort_keys=True)
        )

    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_step": filename_step,
        "completed_updates": completed_updates,
        "training_contract_id": EXPECTED_TRAINING_CONTRACT_ID,
        "checkpoint_pair_id": contract.get("checkpoint_pair_id"),
        "run_id": contract.get("run_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    print(json.dumps(validate_checkpoint(args.checkpoint), sort_keys=True))


if __name__ == "__main__":
    main()
