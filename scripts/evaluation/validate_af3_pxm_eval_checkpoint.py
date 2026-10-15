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

"""Fail-closed architecture check for AF3-like EMA evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import re

import torch


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


def _check_architecture(contract: Mapping) -> None:
    architecture = contract.get("model_architecture_contract")
    if not isinstance(architecture, Mapping):
        raise ValueError("EMA checkpoint is missing model_architecture_contract")

    expected = {
        "global": {
            "c_s": 384,
            "c_z": 256,
            "max_atoms_per_token": 24,
        },
        "pairformer": {
            "n_blocks": 64,
            "c_s": 384,
            "c_z": 256,
        },
        "diffusion_module": {
            "c_token": 768,
            "c_s": 384,
            "c_z": 256,
        },
        "msa_module": {
            "n_blocks": 4,
            "c_m": 128,
            "c_z": 256,
        },
    }
    mismatches = {}
    for section_name, section_expected in expected.items():
        section = architecture.get(section_name)
        for key, expected_value in section_expected.items():
            observed = section.get(key) if isinstance(section, Mapping) else None
            if observed != expected_value:
                mismatches[f"{section_name}.{key}"] = {
                    "checkpoint": observed,
                    "expected": expected_value,
                }
    if mismatches:
        raise ValueError(
            "EMA checkpoint architecture does not match PF64/cz256/cm128/"
            "diffusion-cs384: "
            + json.dumps(mismatches, sort_keys=True)
        )


def validate_checkpoint(path: Path) -> dict[str, object]:
    match = EMA_FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(
            "EMA checkpoint name must be <zero_based_step>_ema_0.999.pt"
        )
    filename_step = int(match.group("step"))
    completed_updates = filename_step + 1

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
    scalar_expectations = {
        "step": filename_step,
        "next_step": completed_updates,
        "distributed_strategy": "ddp",
        "dtype": "bf16",
        "ema_decay": 0.999,
    }
    mismatches = {
        key: {"checkpoint": contract.get(key), "expected": expected}
        for key, expected in scalar_expectations.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "EMA checkpoint metadata mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    _check_architecture(contract)

    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_step": filename_step,
        "completed_updates": completed_updates,
        "training_contract_id": contract.get("training_contract_id"),
        "data_profile_id": contract.get("data_profile_id"),
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
