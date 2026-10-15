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

"""Validate the locked crop384 soft-cluster migration before torchrun."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protenix.data.pipeline.soft_cluster_schedule import (  # noqa: E402
    load_soft_cluster_schedule_manifest,
)


EXPECTED_TRAIN_SETS = [
    "weightedPDB_before210930_v20260101",
    "afdb_mgnify_short_distillation_v20260725",
    "afdb_mgnify_long_distillation_v20260725",
    "openfold3_disordered_v20260725",
]
EXPECTED_MASSES = [0.49, 0.005, 0.505, 0.02]
EXPECTED_SOFT_DATASETS = EXPECTED_TRAIN_SETS[1:3]
EXPECTED_TARGET_TRAIN_SETS_V2 = [
    "weightedPDB_before210930_v20260101",
    "afdb_mgnify_legacy_residual_distillation_v20260804",
    "all_sources_ge200_structural_distillation_v20260804",
    "openfold3_disordered_v20260725",
]
EXPECTED_TARGET_SOFT_DATASETS_V2 = EXPECTED_TARGET_TRAIN_SETS_V2[1:3]
CATALOG_REPLACEMENT_FORMAT = "protenix.catalog_replacement.v1"
CATALOG_REPLACEMENT_ALLOWED_DELTA = (
    "legacy_short_long_to_legacy_residual_ge200_v1"
)
LOCKED_SOURCE_INVARIANTS_V2 = {
    "source_data_contract_sha256": (
        "6c98c5cab75197b7e4d41b7ce5287af65ca3f750a707ec69a1bdaf29ed24b8de"
    ),
    "model_architecture_sha256": (
        "cd1cbb658e352f0d56a2dcd65c5e0e95ae2e29c321f4fd0da9f79c6145edc34c"
    ),
}
SAME_SCHEDULE_CONTINUATION_MIGRATION_MODE = "same-schedule-continuation-v1"
SAME_SCHEDULE_PARENT_RUN_ID = (
    "protenix_v4_crop384_db48_disordered_v4_v10_to_v11_"
    "msaquerypinle1024_from_s9999_v9_straggler_20260730_104039"
)
SAME_SCHEDULE_PARENT_MANIFEST_SHA256 = (
    "3b272882b12d2d7e35e37df1a7ef2524edab7fd8a9ab04d76d77f2a9ec596dea"
)
SAME_SCHEDULE_PARENT_ROWS_SHA256 = (
    "b6571b197471e38be0d1a4ac580663c666dd680bf4acfdb0a051553c538e758f"
)
SAME_SCHEDULE_PARENT_DATASET_BINDING_SHA256 = (
    "faee230baca09fc06cef26befdd5b46bddad4a7f67e9e8fed8c2a3a59a2275c6"
)
SAME_SCHEDULE_PARENT_SOURCE_BINDING_SHA256 = (
    "eacc8d1323f3a0a2f08ea52ebab0c292334e0d48bcc07505cf21df4a31de2194"
)
MSA_INDEPENDENT_SOURCE_RUN_ID = (
    "protenix_v4_step21999_residual_ge200_softcluster_g02_"
    "v9_straggler_20260804_140441"
)
MSA_INDEPENDENT_SOURCE_STEP = 27_799
MSA_INDEPENDENT_SOURCE_COMPLETED_UPDATES = 27_800
MSA_INDEPENDENT_SOURCE_RAW_SHA256 = (
    "4cefb9af5088a4671b4287149c95efcd72cac37449136fe37b4e4d060754f94a"
)
MSA_INDEPENDENT_SOURCE_EMA_SHA256 = (
    "b3ad325bde97f2b249a03c4004a2a00d704e92a3aa0be925b0705e1a23ee67c8"
)
MSA_INDEPENDENT_SOURCE_PAIR_ID = (
    "d40d64151cc8a8ff17458c310ca0a53f40047d3b5d4adf5577db3d653ae1c8f2"
)
MSA_INDEPENDENT_SOURCE_TRAINING_CONTRACT_ID = (
    "crop384_db48_ddp_disordered_v5_ws128_acc4_catalogreplacement_"
    "residual_ge200_20260804_softcluster_g02_checkpointbound_step21999_"
    "wave_schedule_v2"
)
MSA_INDEPENDENT_SOURCE_DATA_CONTRACT_SHA256 = (
    "bbb8d9b0a987e2aed17743df24024daa832d7b66d05ab45783a11a0f03a3f42c"
)


def _sha256_stable_file(path: Path) -> str:
    """Hash one immutable checkpoint and reject concurrent replacement/writes."""

    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        descriptor_before = os.fstat(handle.fileno())
        while True:
            chunk = handle.read(64 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        descriptor_after = os.fstat(handle.fileno())
    after = path.stat()
    identities = {
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        for item in (before, descriptor_before, descriptor_after, after)
    }
    if len(identities) != 1:
        raise RuntimeError(f"checkpoint changed while hashing: {path}")
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--source-step", type=int, required=True)
    parser.add_argument("--source-raw-checkpoint", type=Path, required=True)
    parser.add_argument("--source-ema-checkpoint", type=Path, required=True)
    parser.add_argument("--target-updates", type=int, default=50_000)
    parser.add_argument(
        "--migration-mode",
        choices=(
            "auto",
            "schedule-only-v1",
            "catalog-replacement-v2",
            SAME_SCHEDULE_CONTINUATION_MIGRATION_MODE,
        ),
        default="auto",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_run_dir = args.source_run_dir.expanduser().resolve(strict=True)
    if args.source_step < 0:
        raise ValueError("--source-step must be non-negative")
    if args.target_updates <= args.source_step + 1:
        raise ValueError("--target-updates must be after the source checkpoint")

    expected_raw = args.source_raw_checkpoint.expanduser().resolve(strict=True)
    expected_ema = args.source_ema_checkpoint.expanduser().resolve(strict=True)
    for label, path in (("raw", expected_raw), ("EMA", expected_ema)):
        if not path.is_file() or not path.is_relative_to(source_run_dir):
            raise ValueError(
                f"source {label} checkpoint must be a file inside source-run-dir"
            )

    is_same_schedule_continuation = (
        args.migration_mode == SAME_SCHEDULE_CONTINUATION_MIGRATION_MODE
    )
    schedule_expectations = {
        "world_size": 128,
        "epoch_size": 10_240,
        "gamma": 0.2,
        "seed": 42,
    }
    if not is_same_schedule_continuation:
        schedule_expectations["source_checkpoint"] = {
            "checkpoint_step": args.source_step,
            "raw_checkpoint_path": str(expected_raw),
            "ema_checkpoint_path": str(expected_ema),
        }
    info = load_soft_cluster_schedule_manifest(
        args.schedule,
        expected=schedule_expectations,
    )
    source = info.source_checkpoint
    if source["mode"] != "checkpoint_migration":
        raise ValueError("formal migration requires a checkpoint-bound schedule")
    if not is_same_schedule_continuation:
        if int(source["checkpoint_step"]) != args.source_step:
            raise ValueError(
                "schedule/source step mismatch: "
                f"schedule={source['checkpoint_step']} requested={args.source_step}"
            )
        if source["source_run_id"] != source_run_dir.name:
            raise ValueError(
                "schedule/source run mismatch: "
                f"schedule={source['source_run_id']!r} "
                f"requested={source_run_dir.name!r}"
            )

        if source["raw_checkpoint_path"] != str(expected_raw):
            raise ValueError(
                "manifest raw checkpoint path does not match the selected file"
            )
        if source["ema_checkpoint_path"] != str(expected_ema):
            raise ValueError(
                "manifest EMA checkpoint path does not match the selected file"
            )

    accumulation = int(source["iters_to_accumulate"])
    if accumulation != 4:
        raise ValueError(f"formal migration requires accumulation=4, got {accumulation}")
    if info.origin_consumed_global_batches != (
        int(source["checkpoint_step"]) + 1
    ) * accumulation:
        raise ValueError("manifest origin is inconsistent with its anchor step")
    selected_consumed_global_batches = (args.source_step + 1) * accumulation
    if is_same_schedule_continuation and not (
        info.origin_consumed_global_batches
        <= selected_consumed_global_batches
        < info.target_consumed_global_batches
    ):
        raise ValueError(
            "continuation checkpoint cursor is outside the authenticated schedule"
        )
    if info.target_consumed_global_batches != args.target_updates * accumulation:
        raise ValueError(
            "manifest target does not match the absolute update target: "
            f"schedule={info.target_consumed_global_batches / accumulation:g}, "
            f"requested={args.target_updates}"
        )

    manifest_schema = int(info.schema_version)
    migration_mode = args.migration_mode
    if migration_mode == "auto":
        migration_mode = (
            "catalog-replacement-v2"
            if manifest_schema == 2
            else "schedule-only-v1"
        )
    expected_schema = (
        2
        if migration_mode
        in {
            "catalog-replacement-v2",
            SAME_SCHEDULE_CONTINUATION_MIGRATION_MODE,
        }
        else 1
    )
    if manifest_schema != expected_schema:
        raise ValueError(
            f"migration mode {migration_mode} requires schedule schema "
            f"{expected_schema}, got {manifest_schema}"
        )

    dataset = info.dataset
    expected_train_sets = (
        EXPECTED_TARGET_TRAIN_SETS_V2
        if migration_mode
        in {
            "catalog-replacement-v2",
            SAME_SCHEDULE_CONTINUATION_MIGRATION_MODE,
        }
        else EXPECTED_TRAIN_SETS
    )
    expected_soft_datasets = (
        EXPECTED_TARGET_SOFT_DATASETS_V2
        if migration_mode
        in {
            "catalog-replacement-v2",
            SAME_SCHEDULE_CONTINUATION_MIGRATION_MODE,
        }
        else EXPECTED_SOFT_DATASETS
    )
    if dataset["train_sets"] != expected_train_sets:
        raise ValueError("manifest train_sets do not match the locked profile")
    if dataset["dataset_masses"] != EXPECTED_MASSES:
        raise ValueError("manifest dataset masses do not match the locked profile")
    if info.algorithm["soft_cluster_datasets"] != expected_soft_datasets:
        raise ValueError(
            "manifest does not apply soft sampling to the exact target datasets"
        )

    catalog_summary = None
    checkpoint_hashes = None
    selected_source = source
    if migration_mode in {
        "catalog-replacement-v2",
        SAME_SCHEDULE_CONTINUATION_MIGRATION_MODE,
    }:
        for field, expected_value in LOCKED_SOURCE_INVARIANTS_V2.items():
            if source.get(field) != expected_value:
                raise ValueError(
                    f"catalog replacement source invariant mismatch for {field}"
                )
        if is_same_schedule_continuation:
            locked_schedule_fields = {
                "manifest_sha256": (
                    info.manifest_sha256,
                    SAME_SCHEDULE_PARENT_MANIFEST_SHA256,
                ),
                "schedule_sha256": (
                    info.schedule_sha256,
                    SAME_SCHEDULE_PARENT_ROWS_SHA256,
                ),
                "source_checkpoint_binding_sha256": (
                    info.source_checkpoint_binding_sha256,
                    SAME_SCHEDULE_PARENT_SOURCE_BINDING_SHA256,
                ),
            }
            for field, (actual_value, expected_value) in (
                locked_schedule_fields.items()
            ):
                if actual_value != expected_value:
                    raise ValueError(
                        f"same-schedule continuation {field} mismatch: "
                        f"actual={actual_value}, expected={expected_value}"
                    )
            if source.get("source_run_id") != SAME_SCHEDULE_PARENT_RUN_ID:
                raise ValueError(
                    "same-schedule continuation anchor run identity mismatch"
                )
            if source_run_dir.name != MSA_INDEPENDENT_SOURCE_RUN_ID:
                raise ValueError(
                    "MSA-only continuation source run is not the locked run"
                )
            if args.source_step != MSA_INDEPENDENT_SOURCE_STEP:
                raise ValueError(
                    "MSA-only continuation is locked to source step "
                    f"{MSA_INDEPENDENT_SOURCE_STEP}"
                )
            locked_raw_path = (
                source_run_dir
                / "checkpoints"
                / f"{MSA_INDEPENDENT_SOURCE_STEP}.pt"
            )
            locked_ema_path = (
                source_run_dir
                / "checkpoints"
                / f"{MSA_INDEPENDENT_SOURCE_STEP}_ema_0.999.pt"
            )
            if expected_raw != locked_raw_path:
                raise ValueError(
                    "MSA-only continuation raw checkpoint path is not locked"
                )
            if expected_ema != locked_ema_path:
                raise ValueError(
                    "MSA-only continuation EMA checkpoint path is not locked"
                )
            selected_source = {
                "checkpoint_step": MSA_INDEPENDENT_SOURCE_STEP,
                "completed_updates": MSA_INDEPENDENT_SOURCE_COMPLETED_UPDATES,
                "next_global_step": selected_consumed_global_batches,
                "origin_consumed_global_batches_per_rank": (
                    selected_consumed_global_batches
                ),
                "iters_to_accumulate": 4,
                "raw_checkpoint_sha256": MSA_INDEPENDENT_SOURCE_RAW_SHA256,
                "ema_checkpoint_sha256": MSA_INDEPENDENT_SOURCE_EMA_SHA256,
                "checkpoint_pair_id": MSA_INDEPENDENT_SOURCE_PAIR_ID,
                "source_training_contract_id": (
                    MSA_INDEPENDENT_SOURCE_TRAINING_CONTRACT_ID
                ),
                "source_data_contract_sha256": (
                    MSA_INDEPENDENT_SOURCE_DATA_CONTRACT_SHA256
                ),
                "source_run_id": MSA_INDEPENDENT_SOURCE_RUN_ID,
                "model_architecture_sha256": LOCKED_SOURCE_INVARIANTS_V2[
                    "model_architecture_sha256"
                ],
            }
        checkpoint_hashes = {}
        for label, path, source_field in (
            ("raw", expected_raw, "raw_checkpoint_sha256"),
            ("EMA", expected_ema, "ema_checkpoint_sha256"),
        ):
            actual_sha256 = _sha256_stable_file(path)
            expected_sha256 = selected_source.get(source_field)
            if actual_sha256 != expected_sha256:
                checkpoint_binding_label = (
                    "locked step-27799 boundary"
                    if is_same_schedule_continuation
                    else "validated schedule binding"
                )
                raise ValueError(
                    f"source {label} checkpoint SHA-256 differs from the "
                    f"{checkpoint_binding_label}: "
                    f"actual={actual_sha256}, expected={expected_sha256}"
                )
            checkpoint_hashes[source_field] = actual_sha256
        catalog = info.manifest.get("catalog_migration")
        if not isinstance(catalog, dict):
            raise ValueError("schema-v2 manifest lacks catalog_migration")
        if catalog.get("format") != CATALOG_REPLACEMENT_FORMAT:
            raise ValueError("catalog migration format mismatch")
        if catalog.get("allowed_delta") != CATALOG_REPLACEMENT_ALLOWED_DELTA:
            raise ValueError("catalog migration allowed_delta mismatch")
        source_dataset = catalog.get("source_dataset", {}).get("binding", {})
        target_dataset = catalog.get("target_dataset", {}).get("binding", {})
        if source_dataset.get("train_sets") != EXPECTED_TRAIN_SETS:
            raise ValueError("catalog migration source train_sets mismatch")
        if source_dataset.get("dataset_masses") != EXPECTED_MASSES:
            raise ValueError("catalog migration source masses mismatch")
        if target_dataset != dataset:
            raise ValueError(
                "catalog migration target dataset does not equal manifest.dataset"
            )
        source_run = info.manifest.get("source_run")
        if not isinstance(source_run, dict):
            raise ValueError("catalog replacement lacks source_run binding")
        manifest_source_run = Path(source_run.get("run_dir", "")).resolve()
        if is_same_schedule_continuation:
            if manifest_source_run.name != SAME_SCHEDULE_PARENT_RUN_ID:
                raise ValueError(
                    "same-schedule continuation parent source_run mismatch"
                )
        elif manifest_source_run != source_run_dir:
            raise ValueError("catalog replacement source_run directory mismatch")
        target_dataset_binding_sha256 = catalog["target_dataset"][
            "binding_sha256"
        ]
        if (
            is_same_schedule_continuation
            and target_dataset_binding_sha256
            != SAME_SCHEDULE_PARENT_DATASET_BINDING_SHA256
        ):
            raise ValueError(
                "same-schedule continuation dataset binding mismatch"
            )
        catalog_summary = {
            "format": catalog["format"],
            "allowed_delta": catalog["allowed_delta"],
            "source_dataset_binding_sha256": catalog["source_dataset"][
                "binding_sha256"
            ],
            "target_dataset_binding_sha256": catalog["target_dataset"][
                "binding_sha256"
            ],
            "crosswalk_format": catalog["crosswalk"]["format"],
            "exposure_policy": catalog["crosswalk"]["exposure_policy"],
            "ambiguous_alternate_policy": catalog["crosswalk"][
                "ambiguous_alternate_policy"
            ],
        }

    print(
        json.dumps(
            {
                "manifest_path": str(info.manifest_path),
                "manifest_sha256": info.manifest_sha256,
                "source_run_id": selected_source["source_run_id"],
                "source_checkpoint_step": selected_source["checkpoint_step"],
                "source_checkpoint_pair_id": selected_source[
                    "checkpoint_pair_id"
                ],
                "source_completed_updates": selected_source[
                    "completed_updates"
                ],
                "origin_consumed_global_batches_per_rank": (
                    info.origin_consumed_global_batches
                ),
                "selected_consumed_global_batches_per_rank": (
                    selected_consumed_global_batches
                ),
                "target_updates": args.target_updates,
                "schedule_rows": info.row_count,
                "schedule_sha256": info.schedule_sha256,
                "source_checkpoint_binding_sha256": (
                    info.source_checkpoint_binding_sha256
                ),
                "source_checkpoint_hashes": checkpoint_hashes,
                "gamma": info.gamma,
                "migration_mode": migration_mode,
                "catalog_migration": catalog_summary,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
