#!/usr/bin/env bash
# Copyright 2024 ByteDance and/or its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Continue the 47999 code with the enlarged crop and the SAbDab mixture.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"

export RUN_NAME="${RUN_NAME:-protenix_v4_crop640_stage3_16x8_acc4_nopeptide_v2_from67999}"
export BASE_DIR="${BASE_DIR:-/mnt/project_abd/klp/output}"
export TRAIN_CROP_SIZE="${TRAIN_CROP_SIZE:-640}"
export RESUME_NNODES="${RESUME_NNODES:-${NNODES:-16}}"
export RESUME_NPROC_PER_NODE="${RESUME_NPROC_PER_NODE:-${NPROC_PER_NODE:-8}}"
export START_STEP="${START_STEP:-68000}"
export MAX_STEPS="${MAX_STEPS:-170000}"

# Explicit Bash checkpoint settings win. Otherwise resume this branch's latest
# completed raw/EMA pair; the initial renamed pair is used only on first launch.
if [[ -z "${RESUME_RUN_DIR:-}${RESUME_STEP:-}${RESUME_RAW_CHECKPOINT:-}${RESUME_EMA_CHECKPOINT:-}" ]]; then
    for checkpoint in "${BASE_DIR}/${RUN_NAME}"*/checkpoints/*.pt; do
        checkpoint_name="${checkpoint##*/}"
        [[ "${checkpoint_name}" =~ ^([0-9]+)\.pt$ ]] || continue
        checkpoint_step="${BASH_REMATCH[1]}"
        [[ -s "${checkpoint}" && -s "${checkpoint%.pt}_ema_0.999.pt" ]] || continue
        if (( checkpoint_step > ${RESUME_STEP:-0} )); then
            export RESUME_STEP="${checkpoint_step}"
            export RESUME_RUN_DIR="${checkpoint%/checkpoints/*}"
        fi
    done
fi
export RESUME_RUN_DIR="${RESUME_RUN_DIR:-/mnt/project_abd/klp/output/initial_checkpoints/stage3_nopeptide_v2_from47999}"
export RESUME_STEP="${RESUME_STEP:-67999}"

# Retain the original four dataset definitions and soft-cluster algorithm;
# only the new data stage starts with zero exposure. Ordinary restarts restore
# the current schedule cursor, as checked by the trainer.
export PROTENIX_LOCKED_DATA_PROFILE=weighted_pdb_legacy_residual_ge200_disordered_v5
export PROTENIX_TRAINING_DATA_PROFILE="${PROTENIX_LOCKED_DATA_PROFILE}"
export PROTENIX_LEGACY_RESIDUAL_RUNTIME_INDEX_DIR=/mnt/project_abd_hs/klp/databases/afdb_mgnify_legacy_residual_not_ge200_runtime_index_20260804
export PROTENIX_GE200_STRUCTURAL_RUNTIME_INDEX_DIR=/mnt/project_abd_hs/klp/databases/all_sources_ge200_structural_runtime_index_20260804
export PROTENIX_OPENFOLD3_DISORDERED_ROOT_DIR=/mnt/project_abd_hs/klp/databases/pdb_disordered
export PROTENIX_SOFT_CLUSTER_SCHEDULE_DIR="${PROTENIX_SOFT_CLUSTER_SCHEDULE_DIR:-/mnt/project_abd/klp/output/soft_cluster_schedules/stage3_nopeptide_v2_fresh67999_to169999_g02}"
export PROTENIX_RESET_DATA_SAMPLER="${PROTENIX_RESET_DATA_SAMPLER:-1}"
# Historical identifier retained for the unchanged full-state resume format.
# The actual crop is recorded separately in config.yaml and run_manifest.json.
export PROTENIX_SOFT_CLUSTER_TARGET_TRAINING_CONTRACT_ID=crop384_db48_ddp_disordered_v5_ws128_acc4_catalogreplacement_residual_ge200_20260804_softcluster_g02_checkpointbound_step21999_wave_schedule_v2_weightedpdb_pairing_nonpairing_independent10k_v1
export PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS=true
export INDEPENDENT_PAIRING_NON_PAIRING_BUDGET=true

source "${repo_dir}/scripts/runtime/sequence_template_cache_contract.sh"
lock_and_validate_sequence_template_cache

echo "47999-code SAbDab continuation: crop=${TRAIN_CROP_SIZE} accumulation=4"
echo "  source_run=${RESUME_RUN_DIR} checkpoint=${RESUME_STEP}"
echo "  new_data_start_step=${START_STEP}; applied only when entering the new data schedule"
echo "  restore=raw+EMA+Adam+scheduler+step+rank_RNG; fresh data schedule only on first transition"
exec bash "${repo_dir}/resume_crop384_db48_ddp_scratch_200k.sh" "$@"
