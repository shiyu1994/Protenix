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

# One-way, full-state, single-variable migration from the immutable step-27799
# checkpoint pair.  The only training-contract delta is weightedPDB's
# independent 10k pairing / 10k genuine-non-pairing source budget.
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
    echo "This locked step-27799 MSA-only migration accepts no arguments." >&2
    exit 2
fi

resolved_run_name="${RUN_NAME:-${run_name:-}}"
if [ -z "${resolved_run_name}" ]; then
    echo "RUN_NAME is required." >&2
    exit 2
fi
for required_tag in step27799 msaindependent_only softcluster_g02 v9_straggler; do
    if [[ "${resolved_run_name}" != *"${required_tag}"* ]]; then
        echo "RUN_NAME must contain ${required_tag}." >&2
        exit 2
    fi
done

source_run_dir="/mnt/project_abd/klp/proteinx_output/"\
"protenix_v4_step21999_residual_ge200_softcluster_g02_v9_straggler_"\
"20260804_140441"
source_step=27799
source_scheduler_step=27800
source_raw_checkpoint="${source_run_dir}/checkpoints/${source_step}.pt"
source_ema_checkpoint="${source_run_dir}/checkpoints/${source_step}_ema_0.999.pt"
source_raw_sha256="4cefb9af5088a4671b4287149c95efcd72cac37449136fe37b4e4d060754f94a"
source_ema_sha256="b3ad325bde97f2b249a03c4004a2a00d704e92a3aa0be925b0705e1a23ee67c8"
source_pair_id="d40d64151cc8a8ff17458c310ca0a53f40047d3b5d4adf5577db3d653ae1c8f2"
source_training_contract_id="crop384_db48_ddp_disordered_v5_ws128_acc4_"\
"catalogreplacement_residual_ge200_20260804_softcluster_g02_"\
"checkpointbound_step21999_wave_schedule_v2"
target_training_contract_id="${source_training_contract_id}"\
"_weightedpdb_pairing_nonpairing_independent10k_v1"

schedule_dir="/mnt/project_abd/klp/proteinx_output/soft_cluster_schedules/"\
"step21999_residual_ge200_g02_v2"
residual_runtime_index_dir="/mnt/project_abd_hs/klp/databases/"\
"afdb_mgnify_legacy_residual_not_ge200_runtime_index_20260804"
ge200_runtime_index_dir="/mnt/project_abd_hs/klp/databases/"\
"all_sources_ge200_structural_runtime_index_20260804"

# Reject stale parent-shell state.  The entry point owns every identity and
# semantic switch involved in this migration.
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
    INDEPENDENT_PAIRING_NON_PAIRING_BUDGET; do
    if [ -n "${!variable_name-}" ]; then
        echo "Locked step-27799 MSA-only migration rejects inherited ${variable_name}." >&2
        exit 2
    fi
done

if [ "${dry_run}" != "1" ] && [ "${skip_path_check}" != "1" ]; then
    for required_path in \
        "${source_raw_checkpoint}" \
        "${source_ema_checkpoint}" \
        "${schedule_dir}" \
        "${residual_runtime_index_dir}" \
        "${ge200_runtime_index_dir}"; do
        if [ ! -e "${required_path}" ]; then
            echo "Required locked artifact is missing: ${required_path}" >&2
            exit 2
        fi
    done
fi

schedule_validation=""
if [ "${dry_run}" != "1" ]; then
    schedule_validation="$(
        /mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
            "${repo_dir}/scripts/runtime/validate_soft_cluster_migration_request.py" \
            --schedule "${schedule_dir}" \
            --source-run-dir "${source_run_dir}" \
            --source-step "${source_step}" \
            --source-raw-checkpoint "${source_raw_checkpoint}" \
            --source-ema-checkpoint "${source_ema_checkpoint}" \
            --target-updates 50000 \
            --migration-mode same-schedule-continuation-v1
    )"
fi

# shellcheck disable=SC1091
source "${repo_dir}/scripts/runtime/sequence_template_cache_contract.sh"
lock_and_validate_sequence_template_cache

export RUN_NAME="${resolved_run_name}"
export run_name="${resolved_run_name}"
export RESUME_RUN_DIR="${source_run_dir}"
export RESUME_STEP="${source_step}"
export RESUME_RAW_CHECKPOINT="${source_raw_checkpoint}"
export RESUME_EMA_CHECKPOINT="${source_ema_checkpoint}"
export PROTENIX_LEGACY_RESIDUAL_RUNTIME_INDEX_DIR="${residual_runtime_index_dir}"
export PROTENIX_GE200_STRUCTURAL_RUNTIME_INDEX_DIR="${ge200_runtime_index_dir}"
export PROTENIX_SOFT_CLUSTER_SCHEDULE_DIR="${schedule_dir}"
export PROTENIX_SOFT_CLUSTER_TARGET_TRAINING_CONTRACT_ID="${target_training_contract_id}"
export PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION="weightedpdb_independent_pairing_nonpairing_full_state_v1"
export PROTENIX_LOCKED_DATA_PROFILE="weighted_pdb_legacy_residual_ge200_disordered_v5"
export PROTENIX_TRAINING_DATA_PROFILE="${PROTENIX_LOCKED_DATA_PROFILE}"
export PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS=true
export INDEPENDENT_PAIRING_NON_PAIRING_BUDGET=true

echo "Locked step-27799 weightedPDB MSA-only migration:"
echo "  source_run=${source_run_dir}"
echo "  source_step=${source_step} next_trainer_step=$((source_step + 1))"
echo "  source_scheduler_step=${source_scheduler_step} source_lr=0.00095 restored_unchanged=true"
echo "  raw=${source_raw_checkpoint} sha256=${source_raw_sha256}"
echo "  ema=${source_ema_checkpoint} sha256=${source_ema_sha256}"
echo "  source_pair_id=${source_pair_id}"
echo "  source_contract=${source_training_contract_id}"
echo "  schedule=${schedule_dir} gamma=0.2 cursor_continuation=true unchanged=true"
echo "  PDB_member_schedule=parent_exact unchanged=true"
echo "  weightedPDB_MSA=pairing.a3m_top10000+genuine_non_pairing.a3m_top10000 independent=true final<=16384"
echo "  MSA_dedup=paired_block:false unpaired_merge:true inference:unchanged"
if [ -n "${schedule_validation}" ]; then
    echo "  schedule_validation=${schedule_validation}"
else
    echo "  schedule_validation=skipped_only_for_explicit_dry_run"
fi
echo "  data_profile=${PROTENIX_LOCKED_DATA_PROFILE} outer_masses=unchanged"
echo "  target_contract=${target_training_contract_id}"
echo "  target_max_steps=50000 checkpoint_interval=200"
echo "  only_allowed_delta=weightedpdb_independent_pairing_nonpairing_top10k_only_v1"
echo "  rollback=restart_parent_repo_from_step27799_raw_and_matched_ema"

exec bash "${repo_dir}/resume_crop384_db48_ddp_scratch_200k.sh"
