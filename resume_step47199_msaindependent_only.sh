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

# Same-contract strict full-state continuation from the last complete
# checkpoint pair produced before the 2026-08-13 interruption.  This is not a
# second MSA migration: the source checkpoint already belongs to the
# weightedPDB independent pairing/non-pairing contract.
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
    echo "This locked step-47199 full-resume launcher accepts no arguments." >&2
    exit 2
fi

resolved_run_name="${RUN_NAME:-${run_name:-}}"
if [ -z "${resolved_run_name}" ]; then
    echo "RUN_NAME is required." >&2
    exit 2
fi
for required_tag in from47199 msaindependent_only softcluster_g02 v9_straggler; do
    if [[ "${resolved_run_name}" != *"${required_tag}"* ]]; then
        echo "RUN_NAME must contain ${required_tag}." >&2
        exit 2
    fi
done

source_run_dir="/mnt/project_abd/klp/proteinx_output/"\
"protenix_v4_step27799_msaindependent_only_softcluster_g02_v9_straggler_"\
"20260807_053504"
source_step=47199
source_next_step=47200
source_next_global_step=188800
source_scheduler_step=47200
source_raw_checkpoint="${source_run_dir}/checkpoints/${source_step}.pt"
source_ema_checkpoint="${source_run_dir}/checkpoints/${source_step}_ema_0.999.pt"
source_raw_size=6494003469
source_ema_size=2164205909
source_raw_sha256="858ef830befb2e812b3eef1f7ec0f4a8e7db962e18a894975f95af87f9c8ce6f"
source_ema_sha256="1a29bf7b121dab71533d783c761bf850db76431605fc8424a01dc16156ed8730"
source_pair_id="bd6a4a8c0ca5f5f2c50384ef0b88af9edd249b525916273b14e96a9468ef70d5"
training_contract_id="crop384_db48_ddp_disordered_v5_ws128_acc4_"\
"catalogreplacement_residual_ge200_20260804_softcluster_g02_"\
"checkpointbound_step21999_wave_schedule_v2_"\
"weightedpdb_pairing_nonpairing_independent10k_v1"
data_contract_sha256="ccf4fcfd799521fbc7cd0be1c6cc0eea744cf18fe87ab906c24bc485486dce00"
model_architecture_sha256="cd1cbb658e352f0d56a2dcd65c5e0e95ae2e29c321f4fd0da9f79c6145edc34c"
locked_data_profile="weighted_pdb_legacy_residual_ge200_disordered_v5"
schedule_dir="/mnt/project_abd/klp/proteinx_output/soft_cluster_schedules/"\
"step21999_residual_ge200_g02_v2"
residual_runtime_index_dir="/mnt/project_abd_hs/klp/databases/"\
"afdb_mgnify_legacy_residual_not_ge200_runtime_index_20260804"
ge200_runtime_index_dir="/mnt/project_abd_hs/klp/databases/"\
"all_sources_ge200_structural_runtime_index_20260804"

# Prevent a reused shell from silently turning this same-contract continuation
# into the old step-27799 migration or changing its data/sampler identity.
for variable_name in \
    RESUME_RUN_DIR RESUME_STEP RESUME_RAW_CHECKPOINT RESUME_EMA_CHECKPOINT \
    SOFT_CLUSTER_SCHEDULE_DIR PROTENIX_SOFT_CLUSTER_SCHEDULE_DIR \
    PROTENIX_SOFT_CLUSTER_TARGET_TRAINING_CONTRACT_ID \
    PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION \
    PROTENIX_CROP384_STRICT_RESUME_INTERNAL \
    PROTENIX_FULL_RESUME PROTENIX_STRICT_FULL_RESUME \
    RESUME_SMOKE_STEPS PROTENIX_RESUME_SMOKE_STEPS CHECK_ONLY \
    PROTENIX_TRAINING_CONTRACT_ID PROTENIX_TRAINING_DATA_PROFILE \
    PROTENIX_LOCKED_DATA_PROFILE \
    PROTENIX_LEGACY_RESIDUAL_RUNTIME_INDEX_DIR \
    PROTENIX_GE200_STRUCTURAL_RUNTIME_INDEX_DIR \
    PROTENIX_OPENFOLD3_DISORDERED_ROOT_DIR \
    PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS \
    INDEPENDENT_PAIRING_NON_PAIRING_BUDGET; do
    if [ -n "${!variable_name-}" ]; then
        echo "Locked step-47199 full resume rejects inherited ${variable_name}." >&2
        exit 2
    fi
done

if [ "${dry_run}" != "1" ]; then
    for required_path in \
        "${source_run_dir}/manifest/run_manifest.json" \
        "${source_raw_checkpoint}" \
        "${source_ema_checkpoint}" \
        "${schedule_dir}/manifest.json" \
        "${residual_runtime_index_dir}" \
        "${ge200_runtime_index_dir}"; do
        if [ ! -e "${required_path}" ]; then
            echo "Required locked artifact is missing: ${required_path}" >&2
            exit 2
        fi
    done

    actual_raw_size="$(stat -c '%s' "${source_raw_checkpoint}")"
    actual_ema_size="$(stat -c '%s' "${source_ema_checkpoint}")"
    if [ "${actual_raw_size}" != "${source_raw_size}" ]; then
        echo "step-47199 raw checkpoint size mismatch: actual=${actual_raw_size}, expected=${source_raw_size}." >&2
        exit 2
    fi
    if [ "${actual_ema_size}" != "${source_ema_size}" ]; then
        echo "step-47199 EMA checkpoint size mismatch: actual=${actual_ema_size}, expected=${source_ema_size}." >&2
        exit 2
    fi

    actual_raw_sha256="$(sha256sum "${source_raw_checkpoint}" | awk '{print $1}')"
    actual_ema_sha256="$(sha256sum "${source_ema_checkpoint}" | awk '{print $1}')"
    if [ "${actual_raw_sha256}" != "${source_raw_sha256}" ]; then
        echo "step-47199 raw checkpoint SHA-256 mismatch: actual=${actual_raw_sha256}, expected=${source_raw_sha256}." >&2
        exit 2
    fi
    if [ "${actual_ema_sha256}" != "${source_ema_sha256}" ]; then
        echo "step-47199 EMA checkpoint SHA-256 mismatch: actual=${actual_ema_sha256}, expected=${source_ema_sha256}." >&2
        exit 2
    fi
fi

# Keep the exact template-cache contract used to produce the source pair.
# shellcheck disable=SC1091
source "${repo_dir}/scripts/runtime/sequence_template_cache_contract.sh"
lock_and_validate_sequence_template_cache

export RUN_NAME="${resolved_run_name}"
export run_name="${resolved_run_name}"
export RESUME_RUN_DIR="${source_run_dir}"
export RESUME_STEP="${source_step}"
export RESUME_RAW_CHECKPOINT="${source_raw_checkpoint}"
export RESUME_EMA_CHECKPOINT="${source_ema_checkpoint}"
export PROTENIX_LOCKED_DATA_PROFILE="${locked_data_profile}"
export PROTENIX_TRAINING_DATA_PROFILE="${locked_data_profile}"
export PROTENIX_LEGACY_RESIDUAL_RUNTIME_INDEX_DIR="${residual_runtime_index_dir}"
export PROTENIX_GE200_STRUCTURAL_RUNTIME_INDEX_DIR="${ge200_runtime_index_dir}"
export PROTENIX_OPENFOLD3_DISORDERED_ROOT_DIR="/mnt/project_abd_hs/klp/databases/pdb_disordered"
export PROTENIX_SOFT_CLUSTER_SCHEDULE_DIR="${schedule_dir}"
export PROTENIX_SOFT_CLUSTER_TARGET_TRAINING_CONTRACT_ID="${training_contract_id}"
export PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS=true
export INDEPENDENT_PAIRING_NON_PAIRING_BUDGET=true

echo "Locked step-47199 same-contract strict full-state resume:"
echo "  source_run=${source_run_dir}"
echo "  source_step=${source_step} next_trainer_step=${source_next_step} next_global_step=${source_next_global_step}"
echo "  source_scheduler_step=${source_scheduler_step} source_lr=0.00095 restored_unchanged=true"
echo "  raw=${source_raw_checkpoint} sha256=${source_raw_sha256}"
echo "  ema=${source_ema_checkpoint} sha256=${source_ema_sha256}"
echo "  source_pair_id=${source_pair_id}"
echo "  data_contract_sha256=${data_contract_sha256}"
echo "  model_architecture_sha256=${model_architecture_sha256}"
echo "  resume_mode=same_contract contract_migration=none"
echo "  state=model+optimizer+scheduler+step+scaler+128_rank_rng+sampler_cursor+matched_ema"
echo "  data_worker_rng=not_checkpointed"
echo "  schedule=${schedule_dir} gamma=0.2 cursor=188800 target=200000 unchanged=true"
echo "  weightedPDB_MSA=pairing.a3m_top10000+genuine_non_pairing.a3m_top10000 independent=true final<=16384"
echo "  training_contract=${training_contract_id}"
echo "  target_max_steps=50000 remaining_updates=2800 checkpoint_interval=200"

exec bash "${repo_dir}/resume_crop384_db48_ddp_scratch_200k.sh"
