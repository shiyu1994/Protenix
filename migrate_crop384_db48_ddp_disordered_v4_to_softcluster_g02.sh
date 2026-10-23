#!/usr/bin/env bash
# One-way full-state migration from an explicitly selected legacy sampler
# checkpoint to the immutable gamma=0.2 soft-cluster schedule contract.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 0 )); then
    echo "This locked soft-cluster migration launcher accepts no arguments." >&2
    exit 2
fi

resolved_run_name="${RUN_NAME:-${run_name:-}}"
if [ -z "${resolved_run_name}" ]; then
    echo "RUN_NAME is required." >&2
    exit 2
fi
for required_tag in disordered_v4 softcluster_g02 v9_straggler; do
    if [[ "${resolved_run_name}" != *"${required_tag}"* ]]; then
        echo "RUN_NAME must contain ${required_tag}." >&2
        exit 2
    fi
done

source_run_dir="${RESUME_RUN_DIR:-}"
source_step="${RESUME_STEP:-}"
schedule_dir="${SOFT_CLUSTER_SCHEDULE_DIR:-}"
if [ -z "${source_run_dir}" ]; then
    echo "RESUME_RUN_DIR is required; the launcher never guesses history." >&2
    exit 2
fi
if [[ ! "${source_step}" =~ ^[0-9]+$ ]]; then
    echo "RESUME_STEP must be a non-negative integer." >&2
    exit 2
fi
if [ -z "${schedule_dir}" ]; then
    echo "SOFT_CLUSTER_SCHEDULE_DIR is required." >&2
    exit 2
fi

for variable_name in \
    PROTENIX_SOFT_CLUSTER_SCHEDULE_DIR \
    PROTENIX_SOFT_CLUSTER_TARGET_TRAINING_CONTRACT_ID \
    PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION \
    PROTENIX_CROP384_STRICT_RESUME_INTERNAL \
    PROTENIX_FULL_RESUME PROTENIX_STRICT_FULL_RESUME \
    PROTENIX_TRAINING_CONTRACT_ID; do
    if [ -n "${!variable_name-}" ]; then
        echo "Locked soft-cluster migration rejects inherited ${variable_name}." >&2
        exit 2
    fi
done

source_run_dir="$(realpath "${source_run_dir}")"
schedule_dir="$(realpath "${schedule_dir}")"
source_raw_checkpoint="${RESUME_RAW_CHECKPOINT:-${source_run_dir}/checkpoints/${source_step}.pt}"
source_ema_checkpoint="${RESUME_EMA_CHECKPOINT:-${source_run_dir}/checkpoints/${source_step}_ema_0.999.pt}"
source_raw_checkpoint="$(realpath -m "${source_raw_checkpoint}")"
source_ema_checkpoint="$(realpath -m "${source_ema_checkpoint}")"
if [ "${DRY_RUN:-0}" != "1" ] && [ "${SKIP_PATH_CHECK:-0}" != "1" ]; then
    /mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
        "${repo_dir}/scripts/runtime/validate_soft_cluster_migration_request.py" \
        --schedule "${schedule_dir}" \
        --source-run-dir "${source_run_dir}" \
        --source-step "${source_step}" \
        --source-raw-checkpoint "${source_raw_checkpoint}" \
        --source-ema-checkpoint "${source_ema_checkpoint}" \
        --target-updates 50000
fi

target_training_contract_id="crop384_db48_ddp_disordered_v4_ws128_acc4_softcluster_g02_checkpointbound_wave_schedule_v1"
export RUN_NAME="${resolved_run_name}"
export run_name="${resolved_run_name}"
export RESUME_RUN_DIR="${source_run_dir}"
export RESUME_STEP="${source_step}"
export RESUME_RAW_CHECKPOINT="${source_raw_checkpoint}"
export RESUME_EMA_CHECKPOINT="${source_ema_checkpoint}"
export PROTENIX_SOFT_CLUSTER_SCHEDULE_DIR="${schedule_dir}"
export PROTENIX_SOFT_CLUSTER_TARGET_TRAINING_CONTRACT_ID="${target_training_contract_id}"
export PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION="soft_cluster_schedule_manifest_full_state_v1"

echo "Locked disordered-v4 -> softcluster gamma=0.2 migration:"
echo "  source_run=${source_run_dir}"
echo "  source_step=${source_step} target_start_step=$((source_step + 1))"
echo "  source_raw_checkpoint=${source_raw_checkpoint}"
echo "  source_ema_checkpoint=${source_ema_checkpoint}"
echo "  schedule=${schedule_dir}"
echo "  target_training_contract=${target_training_contract_id}"
echo "  target_updates=50000 topology=16x8 accumulation=4"
echo "  restored=model+ema+optimizer+scheduler+step+scaler+rank_rng+sampler_cursor"
echo "  policy_delta=Short/Long_cluster_weight_gamma_power_exposure gamma=0.2"
echo "  training_started_by_this_validation=false"

exec bash "${repo_dir}/resume_crop384_db48_ddp_disordered_v4.sh"
