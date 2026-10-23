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

# One-way full-state migration from the locked step-21399 legacy Short/Long
# checkpoint into the residual + all-source-ge200 catalog and its immutable
# gamma=0.2 future schedule. This launcher never builds either data artifact.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 0 )); then
    echo "This locked catalog-replacement launcher accepts no arguments." >&2
    exit 2
fi

resolved_run_name="${RUN_NAME:-${run_name:-}}"
if [ -z "${resolved_run_name}" ]; then
    echo "RUN_NAME is required." >&2
    exit 2
fi
for required_tag in v9_straggler residual_ge200 softcluster_g02; do
    if [[ "${resolved_run_name}" != *"${required_tag}"* ]]; then
        echo "RUN_NAME must contain ${required_tag}." >&2
        exit 2
    fi
done

locked_source_run_dir="/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_v10_to_v11_msaquerypinle1024_from_s9999_v9_straggler_20260730_104039"
source_run_dir="$(realpath -m "${RESUME_RUN_DIR:-${locked_source_run_dir}}")"
if [ "${source_run_dir}" != "${locked_source_run_dir}" ]; then
    echo "Catalog migration is locked to ${locked_source_run_dir}." >&2
    exit 2
fi
source_step="${RESUME_STEP:-21399}"
if [ "${source_step}" != "21399" ]; then
    echo "Catalog migration is locked to source step 21399." >&2
    exit 2
fi
schedule_dir="${SOFT_CLUSTER_SCHEDULE_DIR:-}"
if [ -z "${schedule_dir}" ]; then
    echo "SOFT_CLUSTER_SCHEDULE_DIR is required." >&2
    exit 2
fi
schedule_dir="$(realpath -m "${schedule_dir}")"

source_raw_checkpoint="$(realpath -m "${source_run_dir}/checkpoints/21399.pt")"
source_ema_checkpoint="$(realpath -m "${source_run_dir}/checkpoints/21399_ema_0.999.pt")"
if [ -n "${RESUME_RAW_CHECKPOINT:-}" ] \
    && [ "$(realpath -m "${RESUME_RAW_CHECKPOINT}")" != "${source_raw_checkpoint}" ]; then
    echo "RESUME_RAW_CHECKPOINT must be the locked step-21399 raw file." >&2
    exit 2
fi
if [ -n "${RESUME_EMA_CHECKPOINT:-}" ] \
    && [ "$(realpath -m "${RESUME_EMA_CHECKPOINT}")" != "${source_ema_checkpoint}" ]; then
    echo "RESUME_EMA_CHECKPOINT must be the locked step-21399 EMA file." >&2
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
        echo "Locked catalog migration rejects inherited ${variable_name}." >&2
        exit 2
    fi
done

residual_runtime_index_dir="${PROTENIX_LEGACY_RESIDUAL_RUNTIME_INDEX_DIR:-/mnt/project_abd_hs/klp/databases/afdb_mgnify_legacy_residual_not_ge200_runtime_index_20260804}"
ge200_runtime_index_dir="${PROTENIX_GE200_STRUCTURAL_RUNTIME_INDEX_DIR:-/mnt/project_abd_hs/klp/databases/all_sources_ge200_structural_runtime_index_20260804}"
residual_runtime_index_dir="$(realpath -m "${residual_runtime_index_dir}")"
ge200_runtime_index_dir="$(realpath -m "${ge200_runtime_index_dir}")"

if [ "${DRY_RUN:-0}" != "1" ] && [ "${SKIP_PATH_CHECK:-0}" != "1" ]; then
    /mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
        "${repo_dir}/scripts/runtime/validate_soft_cluster_migration_request.py" \
        --schedule "${schedule_dir}" \
        --source-run-dir "${source_run_dir}" \
        --source-step 21399 \
        --source-raw-checkpoint "${source_raw_checkpoint}" \
        --source-ema-checkpoint "${source_ema_checkpoint}" \
        --target-updates 50000 \
        --migration-mode catalog-replacement-v2
fi

# shellcheck disable=SC1091
source "${repo_dir}/scripts/runtime/sequence_template_cache_contract.sh"
lock_and_validate_sequence_template_cache

target_training_contract_id="crop384_db48_ddp_disordered_v5_ws128_acc4_catalogreplacement_residual_ge200_20260804_softcluster_g02_checkpointbound_wave_schedule_v2"
export RUN_NAME="${resolved_run_name}"
export run_name="${resolved_run_name}"
export RESUME_RUN_DIR="${source_run_dir}"
export RESUME_STEP="21399"
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

echo "Locked step-21399 catalog replacement migration:"
echo "  source_run=${source_run_dir}"
echo "  source_step=21399 target_start_step=21400 target_updates=50000"
echo "  source_raw_checkpoint=${source_raw_checkpoint}"
echo "  source_ema_checkpoint=${source_ema_checkpoint}"
echo "  source_raw_sha256=9250ec1f5de4846904580af8614a324932821e3eadbb58b093075aa4eddf5d4a"
echo "  source_ema_sha256=74a9e24f3e377b278e0db214e3986676b43edafc5b39f4e1a5b5eb2c9c65288c"
echo "  source_pair_id=dadd29c0819c0911a43c9b7490a3c60679d073dc21b4ad4cb33117c0323d29c4"
echo "  schedule=${schedule_dir}"
echo "  residual_runtime_index=${residual_runtime_index_dir}"
echo "  ge200_runtime_index=${ge200_runtime_index_dir}"
echo "  target_training_contract=${target_training_contract_id}"
echo "  topology=16x8 world_size=128 accumulation=4 epoch_size=10240 seed=42"
echo "  restored=raw_model+matched_ema+optimizer+scheduler+step+grad_scaler+128_rank_rng+sampler_cursor"
echo "  allowed_delta=legacy_short_long_to_legacy_residual_ge200_v1"
echo "  policy=occurrence_sum_v1 gamma=0.2 ambiguous_alternate=seen_floor_1_v1"
echo "  training_started_by_this_validation=false"

exec bash "${repo_dir}/resume_crop384_db48_ddp_scratch_200k.sh"
