#!/usr/bin/env python3
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

"""Atomically derive the locked residual/ge200 target config from a source run.

This is an offline, fail-closed preparation step.  It never opens a checkpoint,
constructs a dataset, or starts training.  The only accepted config delta is the
in-place Short/Long child replacement plus four runtime-address fields on each
replacement child.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml


FORMAT = "protenix.catalog_replacement_target_config.v1"
DEFAULT_SOURCE_CHECKPOINT_STEP = 21_399
SOURCE_RUN_DIR = Path(
    "/mnt/project_abd/klp/proteinx_output/"
    "protenix_v4_crop384_db48_disordered_v4_v10_to_v11_"
    "msaquerypinle1024_from_s9999_v9_straggler_20260730_104039"
)
SOURCE_CONFIG_SHA256 = (
    "64bfe654442970bc7b90f99e39407f6bc5ec1ea0ad09001df2802bef84aa32e4"
)
RESIDUAL_RUNTIME_INDEX_DIR = Path(
    "/mnt/project_abd_hs/klp/databases/"
    "afdb_mgnify_legacy_residual_not_ge200_runtime_index_20260804"
)
GE200_RUNTIME_INDEX_DIR = Path(
    "/mnt/project_abd_hs/klp/databases/"
    "all_sources_ge200_structural_runtime_index_20260804"
)

SOURCE_TRAIN_SETS = [
    "weightedPDB_before210930_v20260101",
    "afdb_mgnify_short_distillation_v20260725",
    "afdb_mgnify_long_distillation_v20260725",
    "openfold3_disordered_v20260725",
]
TARGET_TRAIN_SETS = [
    "weightedPDB_before210930_v20260101",
    "afdb_mgnify_legacy_residual_distillation_v20260804",
    "all_sources_ge200_structural_distillation_v20260804",
    "openfold3_disordered_v20260725",
]
TRAIN_SAMPLE_WEIGHTS = [0.49, 0.005, 0.505, 0.02]
SOURCE_TO_TARGET = {
    SOURCE_TRAIN_SETS[1]: TARGET_TRAIN_SETS[1],
    SOURCE_TRAIN_SETS[2]: TARGET_TRAIN_SETS[2],
}
TARGET_ROLES = {
    TARGET_TRAIN_SETS[1]: "legacy_residual",
    TARGET_TRAIN_SETS[2]: "ge200_structural",
}
RUNTIME_ADDRESS_FIELDS = (
    "runtime_index_dir",
    "source_roots_fpath",
    "mmcif_dir",
    "bioassembly_dict_dir",
)


class TargetConfigError(ValueError):
    """The source config or target runtime pair violates the locked contract."""


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_stable(path: Path) -> bytes:
    path = path.resolve(strict=True)
    before = path.stat()
    with path.open("rb") as handle:
        descriptor_before = os.fstat(handle.fileno())
        payload = handle.read()
        descriptor_after = os.fstat(handle.fileno())
    after = path.stat()
    identities = {
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        for item in (before, descriptor_before, descriptor_after, after)
    }
    if len(identities) != 1:
        raise RuntimeError(f"File changed while it was being read: {path}")
    return payload


def _write_exclusive_fsync(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write while publishing {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _atomic_output_directory(output_dir: Path) -> Iterator[Path]:
    output_dir = output_dir.expanduser().resolve(strict=False)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output_dir):
        raise FileExistsError(f"Refusing to overwrite output: {output_dir}")
    lock = output_dir.with_name(f".{output_dir.name}.prepare.lock")
    lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    stage = output_dir.with_name(
        f".{output_dir.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        stage.mkdir(mode=0o755)
        yield stage
        if os.path.lexists(output_dir):
            raise FileExistsError(f"Output appeared before publication: {output_dir}")
        directory_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(stage, output_dir)
        parent_fd = os.open(output_dir.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        os.close(lock_fd)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TargetConfigError(f"{label} must be a mapping")
    return value


def _effective_sampler_seed(config: Mapping[str, Any]) -> int:
    data = _require_mapping(config.get("data"), label="data")
    sampler = _require_mapping(data.get("train_sampler"), label="data.train_sampler")
    sampler_seed = int(sampler.get("seed", -1))
    return int(config.get("seed", 42)) if sampler_seed < 0 else sampler_seed


def _validate_locked_training_contract(config: Mapping[str, Any]) -> None:
    data = _require_mapping(config.get("data"), label="data")
    sampler = _require_mapping(data.get("train_sampler"), label="data.train_sampler")
    checks = {
        "data.train_sets": (list(data.get("train_sets", [])), SOURCE_TRAIN_SETS),
        "data.train_sampler.train_sample_weights": (
            [float(value) for value in sampler.get("train_sample_weights", [])],
            TRAIN_SAMPLE_WEIGHTS,
        ),
        "data.epoch_size": (int(data.get("epoch_size", -1)), 10240),
        "effective sampler seed": (_effective_sampler_seed(config), 42),
        "iters_to_accumulate": (int(config.get("iters_to_accumulate", -1)), 4),
        "max_steps": (int(config.get("max_steps", -1)), 50000),
        "data.train_sampler.replacement": (bool(sampler.get("replacement", True)), True),
    }
    failures = [
        f"{name}: got {actual!r}, expected {expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if failures:
        raise TargetConfigError("Locked source contract mismatch: " + "; ".join(failures))
    for source, target in SOURCE_TO_TARGET.items():
        if source not in data:
            raise TargetConfigError(f"Source config has no child {source!r}")
        if target in data:
            raise TargetConfigError(f"Source config unexpectedly contains target child {target!r}")


def _load_target_runtime_binding(runtime_dir: Path, *, role: str) -> dict[str, Any]:
    runtime_dir = runtime_dir.expanduser().resolve(strict=True)
    required_paths = {
        "manifest": runtime_dir / "manifest.json",
        "pair_receipt": runtime_dir / "target_catalog_pair.json",
        "roots": runtime_dir / "roots.json",
        "success": runtime_dir / "_SUCCESS.json",
    }
    payloads: dict[str, bytes] = {}
    for label, path in required_paths.items():
        if not path.is_file():
            raise TargetConfigError(f"Target runtime {role} lacks {label}: {path}")
        payloads[label] = _read_stable(path)
    manifest = json.loads(payloads["manifest"])
    target_catalog = _require_mapping(
        _require_mapping(manifest, label=f"{role} manifest").get("target_catalog"),
        label=f"{role}.target_catalog",
    )
    if int(target_catalog.get("schema_version", -1)) != 1:
        raise TargetConfigError(f"{role} target_catalog schema_version must be 1")
    if target_catalog.get("role") != role:
        raise TargetConfigError(
            f"Target runtime role mismatch: got {target_catalog.get('role')!r}, "
            f"expected {role!r}"
        )
    pair_record = _require_mapping(
        target_catalog.get("pair_receipt"), label=f"{role}.pair_receipt"
    )
    if set(pair_record) != {"path", "size_bytes", "sha256"}:
        raise TargetConfigError(f"{role} pair_receipt record has unexpected fields")
    if pair_record.get("path") != "target_catalog_pair.json":
        raise TargetConfigError(f"{role} pair_receipt must be target_catalog_pair.json")
    pair_payload = payloads["pair_receipt"]
    pair_sha = _sha256_bytes(pair_payload)
    if int(pair_record.get("size_bytes", -1)) != len(pair_payload):
        raise TargetConfigError(f"{role} pair_receipt size mismatch")
    if str(pair_record.get("sha256", "")).lower() != pair_sha:
        raise TargetConfigError(f"{role} pair_receipt SHA-256 mismatch")
    pair = _require_mapping(json.loads(pair_payload), label=f"{role} pair document")
    if pair.get("format") != "protenix.target_distillation_catalog_pair.v1":
        raise TargetConfigError(f"{role} pair document format mismatch")
    if pair.get("pair_id") != target_catalog.get("pair_id"):
        raise TargetConfigError(f"{role} manifest/pair pair_id mismatch")
    children = _require_mapping(pair.get("children"), label=f"{role} pair children")
    child = _require_mapping(children.get(role), label=f"{role} pair child")
    if int(child.get("row_count", -1)) != int(manifest.get("row_count", -2)):
        raise TargetConfigError(f"{role} pair/manifest row_count mismatch")
    if int(child.get("cluster_count", -1)) != int(manifest.get("cluster_count", -2)):
        raise TargetConfigError(f"{role} pair/manifest cluster_count mismatch")
    success = _require_mapping(json.loads(payloads["success"]), label=f"{role} success")
    if success.get("status") != "complete":
        raise TargetConfigError(f"{role} runtime is not complete")
    manifest_sha = _sha256_bytes(payloads["manifest"])
    if success.get("manifest_sha256") != manifest_sha:
        raise TargetConfigError(f"{role} _SUCCESS manifest SHA-256 mismatch")
    return {
        "role": role,
        "runtime_index_dir": str(runtime_dir),
        "manifest_path": str(required_paths["manifest"]),
        "manifest_size_bytes": len(payloads["manifest"]),
        "manifest_sha256": manifest_sha,
        "pair_receipt_path": str(required_paths["pair_receipt"]),
        "pair_receipt_size_bytes": len(pair_payload),
        "pair_receipt_sha256": pair_sha,
        "pair_id": str(pair.get("pair_id")),
        "roots_path": str(required_paths["roots"]),
        "roots_sha256": _sha256_bytes(payloads["roots"]),
        "row_count": int(manifest["row_count"]),
        "cluster_count": int(manifest["cluster_count"]),
    }


def _derive_target_config(
    source: Mapping[str, Any],
    *,
    residual_runtime_dir: Path,
    ge200_runtime_dir: Path,
) -> dict[str, Any]:
    _validate_locked_training_contract(source)
    target = copy.deepcopy(dict(source))
    data = target["data"]
    runtime_by_source = {
        SOURCE_TRAIN_SETS[1]: residual_runtime_dir.resolve(strict=True),
        SOURCE_TRAIN_SETS[2]: ge200_runtime_dir.resolve(strict=True),
    }
    for source_name, target_name in SOURCE_TO_TARGET.items():
        child = data.pop(source_name)
        runtime_dir = runtime_by_source[source_name]
        base_info = child["base_info"]
        base_info["runtime_index_dir"] = str(runtime_dir)
        base_info["source_roots_fpath"] = str(runtime_dir / "roots.json")
        base_info["mmcif_dir"] = str(runtime_dir)
        base_info["bioassembly_dict_dir"] = str(runtime_dir)
        data[target_name] = child
    data["train_sets"] = list(TARGET_TRAIN_SETS)
    _validate_target_delta(source, target)
    return target


def _validate_target_delta(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> None:
    restored = copy.deepcopy(dict(target))
    restored_data = restored["data"]
    for source_name, target_name in SOURCE_TO_TARGET.items():
        child = restored_data.pop(target_name)
        source_base = source["data"][source_name]["base_info"]
        for field in RUNTIME_ADDRESS_FIELDS:
            child["base_info"][field] = copy.deepcopy(source_base[field])
        restored_data[source_name] = child
    restored_data["train_sets"] = list(SOURCE_TRAIN_SETS)
    if restored != source:
        raise TargetConfigError(
            "Derived target config changes fields outside the locked catalog replacement"
        )
    _validate_locked_training_contract(restored)
    target_data = target["data"]
    if list(target_data["train_sets"]) != TARGET_TRAIN_SETS:
        raise TargetConfigError("Derived target train_sets differ from the locked order")
    if int(target.get("max_steps", -1)) != 50000:
        raise TargetConfigError("Derived target max_steps must remain 50000")


def prepare_target_config(
    *,
    source_run_dir: Path,
    source_checkpoint_step: int = DEFAULT_SOURCE_CHECKPOINT_STEP,
    residual_runtime_index_dir: Path,
    ge200_runtime_index_dir: Path,
    output_dir: Path,
    enforce_production_source_hash: bool = True,
) -> Path:
    if isinstance(source_checkpoint_step, bool) or not isinstance(
        source_checkpoint_step, int
    ):
        raise TargetConfigError("source_checkpoint_step must be an integer")
    if source_checkpoint_step < 0:
        raise TargetConfigError("source_checkpoint_step must be non-negative")

    source_run_dir = source_run_dir.expanduser().resolve(strict=True)
    source_config_path = (source_run_dir / "config.yaml").resolve(strict=True)
    source_payload = _read_stable(source_config_path)
    source_sha = _sha256_bytes(source_payload)
    if enforce_production_source_hash and source_run_dir == SOURCE_RUN_DIR:
        if source_sha != SOURCE_CONFIG_SHA256:
            raise TargetConfigError(
                "Locked production source config SHA-256 mismatch: "
                f"got {source_sha}, expected {SOURCE_CONFIG_SHA256}"
            )
    source = yaml.safe_load(source_payload)
    if not isinstance(source, Mapping):
        raise TargetConfigError("Source config.yaml must contain a mapping")
    target_updates = int(source.get("max_steps", -1))
    if source_checkpoint_step + 1 >= target_updates:
        raise TargetConfigError(
            "source checkpoint leaves no future optimizer updates: "
            f"completed={source_checkpoint_step + 1}, target={target_updates}"
        )

    bindings = {
        TARGET_TRAIN_SETS[1]: _load_target_runtime_binding(
            residual_runtime_index_dir, role=TARGET_ROLES[TARGET_TRAIN_SETS[1]]
        ),
        TARGET_TRAIN_SETS[2]: _load_target_runtime_binding(
            ge200_runtime_index_dir, role=TARGET_ROLES[TARGET_TRAIN_SETS[2]]
        ),
    }
    residual_pair = bindings[TARGET_TRAIN_SETS[1]]["pair_receipt_sha256"]
    ge200_pair = bindings[TARGET_TRAIN_SETS[2]]["pair_receipt_sha256"]
    pair_ids = {binding["pair_id"] for binding in bindings.values()}
    if residual_pair != ge200_pair or len(pair_ids) != 1:
        raise TargetConfigError(
            "Residual/ge200 runtime indices do not share one byte-identical pair receipt"
        )

    target = _derive_target_config(
        source,
        residual_runtime_dir=residual_runtime_index_dir,
        ge200_runtime_dir=ge200_runtime_index_dir,
    )
    target_payload = yaml.safe_dump(
        target,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).encode("utf-8")
    target_sha = _sha256_bytes(target_payload)
    output_dir = output_dir.expanduser().resolve(strict=False)
    with _atomic_output_directory(output_dir) as stage:
        config_path = stage / "target_config.yaml"
        _write_exclusive_fsync(config_path, target_payload)
        receipt = {
            "schema_version": 1,
            "format": FORMAT,
            "status": "complete",
            "source": {
                "run_dir": str(source_run_dir),
                "config_path": str(source_config_path),
                "config_size_bytes": len(source_payload),
                "config_sha256": source_sha,
                "checkpoint_step": source_checkpoint_step,
                "next_optimizer_update": source_checkpoint_step + 1,
            },
            "target": {
                "config_path": "target_config.yaml",
                "config_size_bytes": len(target_payload),
                "config_sha256": target_sha,
                "max_steps": 50000,
                "train_sets": TARGET_TRAIN_SETS,
                "train_sample_weights": TRAIN_SAMPLE_WEIGHTS,
                "epoch_size": 10240,
                "seed": 42,
                "iters_to_accumulate": 4,
            },
            "allowed_delta": {
                "child_replacements": SOURCE_TO_TARGET,
                "replacement_child_runtime_address_fields": list(
                    RUNTIME_ADDRESS_FIELDS
                ),
            },
            "target_runtime_bindings": bindings,
            "pair_id": next(iter(pair_ids)),
            "pair_receipt_sha256": residual_pair,
        }
        receipt_payload = _json_bytes(receipt)
        _write_exclusive_fsync(stage / "receipt.json", receipt_payload)
        success = {
            "schema_version": 1,
            "status": "complete",
            "target_config_sha256": target_sha,
            "receipt_sha256": _sha256_bytes(receipt_payload),
        }
        _write_exclusive_fsync(stage / "_SUCCESS.json", _json_bytes(success))
    return output_dir / "target_config.yaml"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Atomically derive the locked residual/ge200 target config from the "
            "selected source checkpoint boundary and its source run config."
        )
    )
    parser.add_argument("--source-run-dir", type=Path, default=SOURCE_RUN_DIR)
    parser.add_argument(
        "--source-checkpoint-step",
        type=int,
        default=DEFAULT_SOURCE_CHECKPOINT_STEP,
        help=(
            "Source checkpoint step recorded in the receipt; defaults to "
            f"{DEFAULT_SOURCE_CHECKPOINT_STEP} for backward compatibility."
        ),
    )
    parser.add_argument(
        "--residual-runtime-index-dir",
        type=Path,
        default=RESIDUAL_RUNTIME_INDEX_DIR,
    )
    parser.add_argument(
        "--ge200-runtime-index-dir",
        type=Path,
        default=GE200_RUNTIME_INDEX_DIR,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    target_path = prepare_target_config(
        source_run_dir=args.source_run_dir,
        source_checkpoint_step=args.source_checkpoint_step,
        residual_runtime_index_dir=args.residual_runtime_index_dir,
        ge200_runtime_index_dir=args.ge200_runtime_index_dir,
        output_dir=args.output_dir,
    )
    print(target_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
