#!/usr/bin/env bash
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

# Independent one-way full-state migration from the locked step-21999 legacy
# Short/Long checkpoint into the already published residual + ge200 catalogs.
# The raw/EMA identities are never guessed here: the immutable schema-v2
# schedule binds them and the validator hashes the real files before torchrun.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dry_run="${DRY_RUN:-0}"
skip_path_check="${SKIP_PATH_CHECK:-0}"
if [[ ! "${dry_run}" =~ ^[01]$ ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 2
fi
if [[ ! "${skip_path_check}" =~ ^[01]$ ]]; then
    echo "SKIP_PATH_CHECK must be 0 or 1." >&2
    exit 2
fi
if [ "${skip_path_check}" = "1" ] && [ "${dry_run}" != "1" ]; then
    echo "SKIP_PATH_CHECK=1 is allowed only with DRY_RUN=1." >&2
    exit 2
fi
if (( $# != 0 )); then
    echo "This locked step-21999 catalog-replacement launcher accepts no arguments." >&2
    exit 2
fi

resolved_run_name="${RUN_NAME:-${run_name:-}}"
if [ -z "${resolved_run_name}" ]; then
    echo "RUN_NAME is required." >&2
    exit 2
fi
for required_tag in step21999 v9_straggler residual_ge200 softcluster_g02; do
    if [[ "${resolved_run_name}" != *"${required_tag}"* ]]; then
        echo "RUN_NAME must contain ${required_tag}." >&2
        exit 2
    fi
done

locked_source_run_dir="/mnt/project_abd/klp/proteinx_output/"\
"protenix_v4_crop384_db48_disordered_v4_v10_to_v11_"\
"msaquerypinle1024_from_s9999_v9_straggler_20260730_104039"
source_run_dir="$(realpath -m "${RESUME_RUN_DIR:-${locked_source_run_dir}}")"
if [ "${source_run_dir}" != "${locked_source_run_dir}" ]; then
    echo "Step-21999 catalog migration is locked to ${locked_source_run_dir}." >&2
    exit 2
fi
source_step="${RESUME_STEP:-21999}"
if [ "${source_step}" != "21999" ]; then
    echo "Catalog migration is locked to source step 21999." >&2
    exit 2
fi

locked_schedule_dir="/mnt/project_abd/klp/proteinx_output/soft_cluster_schedules/step21999_residual_ge200_g02_v2"
schedule_dir="$(realpath -m "${SOFT_CLUSTER_SCHEDULE_DIR:-${locked_schedule_dir}}")"
if [ "${schedule_dir}" != "${locked_schedule_dir}" ]; then
    echo "Step-21999 catalog migration is locked to schedule ${locked_schedule_dir}." >&2
    exit 2
fi

source_raw_checkpoint="$(realpath -m "${source_run_dir}/checkpoints/21999.pt")"
source_ema_checkpoint="$(realpath -m "${source_run_dir}/checkpoints/21999_ema_0.999.pt")"
if [ -n "${RESUME_RAW_CHECKPOINT:-}" ] \
    && [ "$(realpath -m "${RESUME_RAW_CHECKPOINT}")" != "${source_raw_checkpoint}" ]; then
    echo "RESUME_RAW_CHECKPOINT must be the locked step-21999 raw file." >&2
    exit 2
fi
if [ -n "${RESUME_EMA_CHECKPOINT:-}" ] \
    && [ "$(realpath -m "${RESUME_EMA_CHECKPOINT}")" != "${source_ema_checkpoint}" ]; then
    echo "RESUME_EMA_CHECKPOINT must be the locked step-21999 EMA file." >&2
    exit 2
fi

for variable_name in \
    PROTENIX_SOFT_CLUSTER_SCHEDULE_DIR \
    PROTENIX_SOFT_CLUSTER_TARGET_TRAINING_CONTRACT_ID \
    PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION \
    PROTENIX_CROP384_STRICT_RESUME_INTERNAL \
    PROTENIX_FULL_RESUME PROTENIX_STRICT_FULL_RESUME \
    PROTENIX_TRAINING_CONTRACT_ID PROTENIX_TRAINING_DATA_PROFILE \
    PROTENIX_LOCKED_DATA_PROFILE; do
    if [ -n "${!variable_name-}" ]; then
        echo "Locked step-21999 catalog migration rejects inherited ${variable_name}." >&2
        exit 2
    fi
done

locked_residual_runtime_index_dir="/mnt/project_abd_hs/klp/databases/"\
"afdb_mgnify_legacy_residual_not_ge200_runtime_index_20260804"
locked_ge200_runtime_index_dir="/mnt/project_abd_hs/klp/databases/"\
"all_sources_ge200_structural_runtime_index_20260804"
residual_runtime_index_dir="$(
    realpath -m "${PROTENIX_LEGACY_RESIDUAL_RUNTIME_INDEX_DIR:-${locked_residual_runtime_index_dir}}"
)"
ge200_runtime_index_dir="$(
    realpath -m "${PROTENIX_GE200_STRUCTURAL_RUNTIME_INDEX_DIR:-${locked_ge200_runtime_index_dir}}"
)"
if [ "${residual_runtime_index_dir}" != "${locked_residual_runtime_index_dir}" ]; then
    echo "Step-21999 migration requires the published residual runtime catalog." >&2
    exit 2
fi
if [ "${ge200_runtime_index_dir}" != "${locked_ge200_runtime_index_dir}" ]; then
    echo "Step-21999 migration requires the published ge200 runtime catalog." >&2
    exit 2
fi

schedule_validation=""
if [ "${dry_run}" != "1" ]; then
    schedule_validation="$(
        /mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
            "${repo_dir}/scripts/runtime/validate_soft_cluster_migration_request.py" \
            --schedule "${schedule_dir}" \
            --source-run-dir "${source_run_dir}" \
            --source-step 21999 \
            --source-raw-checkpoint "${source_raw_checkpoint}" \
            --source-ema-checkpoint "${source_ema_checkpoint}" \
            --target-updates 50000 \
            --migration-mode catalog-replacement-v2
    )"
fi

# shellcheck disable=SC1091
source "${repo_dir}/scripts/runtime/sequence_template_cache_contract.sh"
lock_and_validate_sequence_template_cache

target_training_contract_id="crop384_db48_ddp_disordered_v5_ws128_acc4_"\
"catalogreplacement_residual_ge200_20260804_softcluster_g02_"\
"checkpointbound_step21999_wave_schedule_v2"
export RUN_NAME="${resolved_run_name}"
export run_name="${resolved_run_name}"
export RESUME_RUN_DIR="${source_run_dir}"
export RESUME_STEP="21999"
export RESUME_RAW_CHECKPOINT="${source_raw_checkpoint}"
export RESUME_EMA_CHECKPOINT="${source_ema_checkpoint}"
export PROTENIX_LEGACY_RESIDUAL_RUNTIME_INDEX_DIR="${residual_runtime_index_dir}"
export PROTENIX_GE200_STRUCTURAL_RUNTIME_INDEX_DIR="${ge200_runtime_index_dir}"
export PROTENIX_SOFT_CLUSTER_SCHEDULE_DIR="${schedule_dir}"
export PROTENIX_SOFT_CLUSTER_TARGET_TRAINING_CONTRACT_ID="${target_training_contract_id}"
export PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION="soft_cluster_catalog_replacement_full_state_v2"
export PROTENIX_LOCKED_DATA_PROFILE="weighted_pdb_legacy_residual_ge200_disordered_v5"
export PROTENIX_TRAINING_DATA_PROFILE="${PROTENIX_LOCKED_DATA_PROFILE}"
export PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS=true

echo "Locked step-21999 catalog replacement migration:"
echo "  source_run=${source_run_dir}"
echo "  source_step=21999 target_start_step=22000 target_updates=50000"
echo "  source_raw_checkpoint=${source_raw_checkpoint}"
echo "  source_ema_checkpoint=${source_ema_checkpoint}"
echo "  schedule=${schedule_dir}"
if [ -n "${schedule_validation}" ]; then
    echo "  schedule_validation=${schedule_validation}"
else
    echo "  schedule_validation=skipped_only_for_explicit_dry_run"
fi
echo "  residual_runtime_index=${residual_runtime_index_dir}"
echo "  ge200_runtime_index=${ge200_runtime_index_dir}"
echo "  target_training_contract=${target_training_contract_id}"
echo "  topology=16x8 world_size=128 accumulation=4 epoch_size=10240 seed=42"
echo "  restored=raw_model+matched_ema+optimizer+scheduler+step+grad_scaler+128_rank_rng+sampler_cursor"
echo "  allowed_delta=legacy_short_long_to_legacy_residual_ge200_v1"
echo "  policy=occurrence_sum_v1 gamma=0.2 ambiguous_alternate=seen_floor_1_v1"
echo "  training_started_by_this_validation=false"

exec bash "${repo_dir}/resume_crop384_db48_ddp_scratch_200k.sh"
