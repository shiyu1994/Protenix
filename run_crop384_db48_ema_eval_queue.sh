#!/usr/bin/env bash
# Wait for and evaluate every 5k-update EMA snapshot on one separate GPU.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 0 )); then
    echo "This queue accepts environment variables only." >&2
    exit 2
fi

train_run_dir="${TRAIN_RUN_DIR:-}"
if [ -z "${train_run_dir}" ]; then
    echo "TRAIN_RUN_DIR is required." >&2
    exit 2
fi
if [ "${DRY_RUN:-0}" != "1" ] && [ ! -d "${train_run_dir}" ]; then
    echo "TRAIN_RUN_DIR does not exist: ${train_run_dir}" >&2
    exit 2
fi
if [ "${DRY_RUN:-0}" != "1" ]; then
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "Set CUDA_VISIBLE_DEVICES to the one GPU allocated to this independent queue." >&2
        exit 2
    fi
    if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
        echo "EMA evaluation is single-GPU; CUDA_VISIBLE_DEVICES must expose exactly one GPU." >&2
        exit 2
    fi
fi
updates_csv="${EVAL_UPDATES:-5000,10000,15000,20000,25000,30000,35000,40000,45000,50000}"
poll_seconds="${POLL_SECONDS:-60}"
if [[ ! "${poll_seconds}" =~ ^[1-9][0-9]*$ ]] || (( poll_seconds > 3600 )); then
    echo "POLL_SECONDS must be an integer in [1, 3600]." >&2
    exit 2
fi
wait_for_checkpoints="${WAIT_FOR_CHECKPOINTS:-true}"
if [ "${wait_for_checkpoints}" != "true" ] \
    && [ "${wait_for_checkpoints}" != "false" ]; then
    echo "WAIT_FOR_CHECKPOINTS must be true or false." >&2
    exit 2
fi

source_label="$(basename "${train_run_dir}")"
if [[ ! "${source_label}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "TRAIN_RUN_DIR basename may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
fi
eval_base_dir="${EVAL_BASE_DIR:-/mnt/project_abd/klp/protenix_eval_outputs/crop384_db48_v8_ema}"
queue_state_dir="${QUEUE_STATE_DIR:-${eval_base_dir}/queue_state/${source_label}}"
queue_log_dir="${QUEUE_LOG_DIR:-${eval_base_dir}/queue_logs/${source_label}}"
if [ "${DRY_RUN:-0}" != "1" ]; then
    mkdir -p "${queue_state_dir}" "${queue_log_dir}"
fi

IFS=',' read -r -a updates <<< "${updates_csv}"
for update in "${updates[@]}"; do
    if [[ ! "${update}" =~ ^[1-9][0-9]*$ ]] \
        || (( update < 5000 || update > 50000 || update % 5000 != 0 )); then
        echo "EVAL_UPDATES entries must be in 5000,10000,...,50000; got ${update}." >&2
        exit 2
    fi
    checkpoint_step=$((update - 1))
    checkpoint_path="${train_run_dir}/checkpoints/${checkpoint_step}_ema_0.999.pt"
    done_marker="${queue_state_dir}/update_${update}.done"
    eval_run_name="${source_label}_ema_eval_u${update}"
    eval_log="${queue_log_dir}/update_${update}.log"

    if [ -s "${done_marker}" ]; then
        echo "skip_completed update=${update} marker=${done_marker}"
        continue
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "planned_ema_eval update=${update} checkpoint=${checkpoint_path}"
        printf 'CUDA_VISIBLE_DEVICES=%q EMA_CHECKPOINT=%q RUN_NAME=%q EVAL_BASE_DIR=%q bash %q\n' \
            "${CUDA_VISIBLE_DEVICES:-<set-one-independent-gpu>}" \
            "${checkpoint_path}" \
            "${eval_run_name}" \
            "${eval_base_dir}" \
            "${repo_dir}/eval_crop384_db48_ema_checkpoint.sh"
        continue
    fi

    while [ ! -s "${checkpoint_path}" ]; do
        if [ "${wait_for_checkpoints}" = "false" ]; then
            echo "Missing EMA checkpoint: ${checkpoint_path}" >&2
            exit 2
        fi
        status_path="${train_run_dir}/manifest/run_status.json"
        if [ -s "${status_path}" ] \
            && rg -q '"status"[[:space:]]*:[[:space:]]*"(failed|completed)"' "${status_path}"; then
            echo "Training ended before required EMA checkpoint appeared: ${checkpoint_path}" >&2
            exit 2
        fi
        echo "waiting update=${update} checkpoint=${checkpoint_path} poll_seconds=${poll_seconds}"
        sleep "${poll_seconds}"
    done

    echo "starting_ema_eval update=${update} checkpoint=${checkpoint_path} log=${eval_log}"
    EMA_CHECKPOINT="${checkpoint_path}" \
    RUN_NAME="${eval_run_name}" \
    EVAL_BASE_DIR="${eval_base_dir}" \
    bash "${repo_dir}/eval_crop384_db48_ema_checkpoint.sh" \
        > "${eval_log}" 2>&1
    printf 'completed_updates=%s\ncheckpoint=%s\nlog=%s\n' \
        "${update}" "${checkpoint_path}" "${eval_log}" > "${done_marker}"
    echo "completed_ema_eval update=${update} marker=${done_marker}"
done
