#!/usr/bin/env python3
"""Validate the immutable v8 step-999 raw/EMA source pair for v4 stage init."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import FakeTensorMode


EXPECTED = {
    "step": 999,
    "raw_name": "999.pt",
    "ema_name": "999_ema_0.999.pt",
    "raw_sha256": "b92361c83de2bf3d300c082bb84a0b797f5b2d92837c8deead7a4f1b3da17ca8",
    "ema_sha256": "a4ecad60c39cad6072e0e1093d6f1394e66142ef1b24f03b931f048a48f7b5f7",
    "raw_size": 6_493_867_817,
    "ema_size": 2_164_192_333,
    "pair_id": "1220d08b9f62db65364a5e936e2e7d22577c72e71fcca4ae2a261f0045fb9001",
    "run_id": (
        "protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_"
        "msamix_q20_u30_p50_50k_v8_20260724_155026"
    ),
    "training_contract_id": (
        "crop384_db48_ddp_scratch_50k_ws128_acc4_data_"
        "weightedpdb_only_v1_arch_pf64_cs384_dct768_pfckpt8_c48_"
        "msamix_q20_u30_p50_v8"
    ),
    "data_contract_sha256": (
        "835a4089f4b20b138c310be537b95dedcdd0ca65cb7a8fd39eddc460947c3a08"
    ),
    "architecture_sha256": (
        "cd1cbb658e352f0d56a2dcd65c5e0e95ae2e29c321f4fd0da9f79c6145edc34c"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_metadata(path: Path) -> Mapping:
    with FakeTensorMode():
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{path}: checkpoint payload must be a mapping")
    return checkpoint


def _validate_one(
    path: Path,
    *,
    kind: str,
    expected_sha256: str,
    verify_sha256: bool,
) -> Mapping:
    expected_name = EXPECTED[f"{kind}_name"]
    if path.name != expected_name:
        raise ValueError(f"{kind} filename must be {expected_name}, got {path.name}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing non-empty {kind} checkpoint: {path}")
    if path.stat().st_size != EXPECTED[f"{kind}_size"]:
        raise ValueError(f"{kind} checkpoint size does not match the receipt")
    if verify_sha256:
        observed_sha256 = _sha256(path)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"{kind} SHA256 mismatch: expected {expected_sha256}, "
                f"got {observed_sha256}"
            )

    checkpoint = _load_metadata(path)
    if checkpoint.get("format_version") != 2:
        raise ValueError(f"{kind} checkpoint format_version must be 2")
    if checkpoint.get("checkpoint_kind") != kind:
        raise ValueError(f"{kind} checkpoint_kind must be {kind!r}")
    if checkpoint.get("step") != EXPECTED["step"]:
        raise ValueError(f"{kind} checkpoint step must be {EXPECTED['step']}")
    model = checkpoint.get("model")
    if not isinstance(model, Mapping) or len(model) != 5086:
        raise ValueError(f"{kind} checkpoint must contain exactly 5086 model keys")
    contract = checkpoint.get("resume_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{kind} checkpoint is missing resume_contract")

    required = {
        "schema_version": 3,
        "step": EXPECTED["step"],
        "global_step": 3999,
        "next_step": 1000,
        "next_global_step": 4000,
        "iters_to_accumulate": 4,
        "world_size": 128,
        "distributed_strategy": "ddp",
        "dtype": "bf16",
        "ema_decay": 0.999,
        "data_profile_id": "weighted_pdb_only_v1",
        "data_contract_sha256": EXPECTED["data_contract_sha256"],
        "model_architecture_sha256": EXPECTED["architecture_sha256"],
        "training_contract_id": EXPECTED["training_contract_id"],
        "checkpoint_pair_id": EXPECTED["pair_id"],
        "run_id": EXPECTED["run_id"],
    }
    mismatches = {
        key: {"expected": value, "observed": contract.get(key)}
        for key, value in required.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"{kind} source contract mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {"checkpoint": checkpoint, "model_keys": tuple(model)}


def validate_pair(
    raw_path: Path,
    ema_path: Path,
    *,
    verify_sha256: bool = False,
) -> dict[str, object]:
    if raw_path.resolve() == ema_path.resolve():
        raise ValueError("raw and EMA checkpoint paths must be distinct")
    raw = _validate_one(
        raw_path,
        kind="raw",
        expected_sha256=EXPECTED["raw_sha256"],
        verify_sha256=verify_sha256,
    )
    ema = _validate_one(
        ema_path,
        kind="ema",
        expected_sha256=EXPECTED["ema_sha256"],
        verify_sha256=verify_sha256,
    )
    if raw["model_keys"] != ema["model_keys"]:
        raise ValueError("raw and EMA model key order differs")
    if not isinstance(raw["checkpoint"].get("optimizer"), Mapping):
        raise ValueError("raw source checkpoint is missing optimizer state")
    if raw["checkpoint"].get("scheduler") is None:
        raise ValueError("raw source checkpoint is missing scheduler state")
    return {
        "mode": "params_only_source_pair",
        "source_step": EXPECTED["step"],
        "source_run_id": EXPECTED["run_id"],
        "checkpoint_pair_id": EXPECTED["pair_id"],
        "architecture_sha256": EXPECTED["architecture_sha256"],
        "raw": {
            "path": str(raw_path.resolve()),
            "sha256": EXPECTED["raw_sha256"],
            "kind": "raw",
        },
        "ema": {
            "path": str(ema_path.resolve()),
            "sha256": EXPECTED["ema_sha256"],
            "kind": "ema",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--ema", required=True, type=Path)
    parser.add_argument("--verify-sha256", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            validate_pair(
                args.raw,
                args.ema,
                verify_sha256=args.verify_sha256,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
