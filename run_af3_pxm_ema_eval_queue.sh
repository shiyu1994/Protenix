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

# Continuously evaluate every 600-update EMA checkpoint on one or more
# independent GPUs. Each evaluator remains strictly single-GPU.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 0 )); then
    echo "This queue accepts environment variables only." >&2
    exit 2
fi

dry_run="${DRY_RUN:-0}"
if [ "${dry_run}" != "0" ] && [ "${dry_run}" != "1" ]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 2
fi

train_run_dir="${TRAIN_RUN_DIR:-}"
if [ -z "${train_run_dir}" ]; then
    echo "TRAIN_RUN_DIR is required." >&2
    exit 2
fi
if [ ! -d "${train_run_dir}/checkpoints" ]; then
    echo "Training checkpoint directory does not exist: ${train_run_dir}/checkpoints" >&2
    exit 2
fi

eval_every_updates="${EVAL_EVERY_UPDATES:-600}"
if [ "${eval_every_updates}" != "600" ]; then
    echo "This locked queue requires EVAL_EVERY_UPDATES=600." >&2
    exit 2
fi
include_final="${INCLUDE_FINAL:-true}"
if [ "${include_final}" != "true" ] && [ "${include_final}" != "false" ]; then
    echo "INCLUDE_FINAL must be true or false." >&2
    exit 2
fi
final_update="${FINAL_UPDATE:-50000}"
if [[ ! "${final_update}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FINAL_UPDATE must be a positive integer." >&2
    exit 2
fi
min_update="${EVAL_MIN_UPDATE:-600}"
if [[ ! "${min_update}" =~ ^[1-9][0-9]*$ ]] \
    || (( min_update % eval_every_updates != 0 )); then
    echo "EVAL_MIN_UPDATE must be a positive multiple of 600." >&2
    exit 2
fi
if (( min_update > final_update )); then
    echo "EVAL_MIN_UPDATE cannot exceed FINAL_UPDATE." >&2
    exit 2
fi

eval_scope="${EVAL_SCOPE:-full}"
eval_subset="${EVAL_SUBSET:-combined}"
eval_mode="${EVAL_MODE:-formal}"
if [ "${eval_scope}/${eval_subset}/${eval_mode}" != "full/combined/formal" ]; then
    echo "This queue requires EVAL_SCOPE=full, EVAL_SUBSET=combined, EVAL_MODE=formal." >&2
    exit 2
fi

poll_seconds="${POLL_SECONDS:-60}"
if [[ ! "${poll_seconds}" =~ ^[0-9]+$ ]] \
    || (( poll_seconds < 60 || poll_seconds > 300 )); then
    echo "POLL_SECONDS must be an integer in [60, 300]." >&2
    exit 2
fi
dispatch_poll_seconds="${DISPATCH_POLL_SECONDS:-5}"
if [[ ! "${dispatch_poll_seconds}" =~ ^[0-9]+$ ]] \
    || (( dispatch_poll_seconds < 1 || dispatch_poll_seconds > 60 )); then
    echo "DISPATCH_POLL_SECONDS must be an integer in [1, 60]." >&2
    exit 2
fi
max_retries="${MAX_RETRIES:-1}"
if [[ ! "${max_retries}" =~ ^[0-9]+$ ]] || (( max_retries > 20 )); then
    echo "MAX_RETRIES must be an integer in [0, 20]." >&2
    exit 2
fi
max_attempts=$((max_retries + 1))
retry_delay_seconds="${RETRY_DELAY_SECONDS:-300}"
if [[ ! "${retry_delay_seconds}" =~ ^[0-9]+$ ]] \
    || (( retry_delay_seconds > 86400 )); then
    echo "RETRY_DELAY_SECONDS must be an integer in [0, 86400]." >&2
    exit 2
fi
wait_for_checkpoints="${WAIT_FOR_CHECKPOINTS:-true}"
if [ "${wait_for_checkpoints}" != "true" ] \
    && [ "${wait_for_checkpoints}" != "false" ]; then
    echo "WAIT_FOR_CHECKPOINTS must be true or false." >&2
    exit 2
fi
prioritize_latest="${PRIORITIZE_LATEST:-false}"
if [ "${prioritize_latest}" != "true" ] \
    && [ "${prioritize_latest}" != "false" ]; then
    echo "PRIORITIZE_LATEST must be true or false." >&2
    exit 2
fi

gpu_csv="${EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [ -z "${gpu_csv}" ]; then
    if [ "${dry_run}" = "1" ]; then
        gpu_csv="0"
    else
        echo "Set EVAL_GPUS or CUDA_VISIBLE_DEVICES to one or more comma-separated GPUs." >&2
        exit 2
    fi
fi
IFS=',' read -r -a eval_gpus <<< "${gpu_csv}"
if (( ${#eval_gpus[@]} == 0 )); then
    echo "No evaluator GPUs were configured." >&2
    exit 2
fi
declare -A seen_gpus=()
for gpu in "${eval_gpus[@]}"; do
    if [ -z "${gpu}" ] || [[ ! "${gpu}" =~ ^[A-Za-z0-9._:-]+$ ]]; then
        echo "Invalid GPU identifier in EVAL_GPUS/CUDA_VISIBLE_DEVICES: '${gpu}'." >&2
        exit 2
    fi
    if [[ -n "${seen_gpus[${gpu}]:-}" ]]; then
        echo "GPU identifier is repeated: ${gpu}" >&2
        exit 2
    fi
    seen_gpus["${gpu}"]=1
done

master_port_base="${MASTER_PORT_BASE:-${MASTER_PORT:-29681}}"
if [[ ! "${master_port_base}" =~ ^[0-9]+$ ]] \
    || (( master_port_base < 1024 || master_port_base + ${#eval_gpus[@]} - 1 > 65535 )); then
    echo "MASTER_PORT_BASE/MASTER_PORT does not leave one valid unique port per GPU." >&2
    exit 2
fi

source_label="$(basename "${train_run_dir}")"
if [[ ! "${source_label}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "TRAIN_RUN_DIR basename may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
fi
eval_base_dir="${EVAL_BASE_DIR:-/mnt/project_abd/klp/protenix_eval_outputs/af3_pxm_every600}"
state_namespace="full_combined_formal_every600"
queue_state_dir="${QUEUE_STATE_DIR:-${eval_base_dir}/queue_state/${source_label}/${state_namespace}}"
queue_log_dir="${QUEUE_LOG_DIR:-${eval_base_dir}/queue_logs/${source_label}/${state_namespace}}"
claim_root="${queue_state_dir}/claims"

manifest_dir="${PXM_MANIFEST_DIR:-/mnt/project_abd/klp/proteinx_eval_data/manifests/pxm_legacy_af3_like}"
pdb_list="${PDB_LIST:-${manifest_dir}/combined_le1024.txt}"
if [ ! -s "${pdb_list}" ]; then
    echo "Full combined PXM manifest is missing or empty: ${pdb_list}" >&2
    exit 2
fi

queue_python="${QUEUE_PYTHON:-${VALIDATOR_PYTHON:-/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python}}"
if [ ! -x "${queue_python}" ]; then
    echo "Queue validator Python is not executable: ${queue_python}" >&2
    exit 2
fi
result_validator="${repo_dir}/scripts/evaluation/validate_af3_pxm_eval_result.py"
evaluator_script="${EVALUATOR_SCRIPT:-${repo_dir}/eval_af3_pxm_legacy_ema_checkpoint.sh}"
if [ ! -f "${result_validator}" ]; then
    echo "Result validator is missing: ${result_validator}" >&2
    exit 2
fi
if [ ! -f "${evaluator_script}" ]; then
    echo "Evaluator script is missing: ${evaluator_script}" >&2
    exit 2
fi

read_training_status() {
    "${queue_python}" - "${train_run_dir}/manifest/run_status.json" <<'PY'
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("unknown")
    raise SystemExit(0)
for attempt in range(3):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        status = value.get("status", "unknown")
        print(status if status in {"running", "completed", "failed"} else "unknown")
        raise SystemExit(0)
    except (OSError, json.JSONDecodeError):
        if attempt != 2:
            time.sleep(0.2)
print("unknown")
PY
}

discover_checkpoints() {
    local training_status="$1"
    local checkpoint_path basename checkpoint_digits checkpoint_step completed_updates
    local -a records=()
    shopt -s nullglob
    for checkpoint_path in "${train_run_dir}"/checkpoints/*_ema_0.999.pt; do
        [ -s "${checkpoint_path}" ] || continue
        basename="$(basename "${checkpoint_path}")"
        if [[ ! "${basename}" =~ ^([0-9]+)_ema_0[.]999[.]pt$ ]]; then
            continue
        fi
        checkpoint_digits="${BASH_REMATCH[1]}"
        checkpoint_step=$((10#${checkpoint_digits}))
        completed_updates=$((checkpoint_step + 1))
        if (( completed_updates >= min_update \
            && completed_updates % eval_every_updates == 0 )); then
            records+=("${completed_updates}"$'\t'"${checkpoint_path}")
        fi
    done
    if [ "${training_status}" = "completed" ] && [ "${include_final}" = "true" ]; then
        checkpoint_step=$((final_update - 1))
        checkpoint_path="${train_run_dir}/checkpoints/${checkpoint_step}_ema_0.999.pt"
        if [ -s "${checkpoint_path}" ] \
            && (( final_update >= min_update \
                && final_update % eval_every_updates != 0 )); then
            records+=("${final_update}"$'\t'"${checkpoint_path}")
        fi
    fi
    if (( ${#records[@]} == 0 )); then
        return
    fi
    if [ "${prioritize_latest}" = "true" ] && (( ${#records[@]} > 1 )); then
        mapfile -t records < <(printf '%s\n' "${records[@]}" | sort -n -k1,1)
        printf '%s\n' "${records[-1]}"
        printf '%s\n' "${records[@]:0:${#records[@]}-1}"
        return
    fi
    printf '%s\n' "${records[@]}" | sort -n -k1,1
}

extract_run_dir() {
    local log_path="$1"
    sed -n \
        's/.*Using run name: [^,]*, run dir: \([^,]*\), checkpoint_dir:.*/\1/p' \
        "${log_path}" \
        | tail -n 1
}

latest_attempt_number() {
    local update="$1"
    local path basename attempt max_attempt=0
    shopt -s nullglob
    for path in \
        "${queue_log_dir}/update_${update}.attempt_"*.log \
        "${queue_state_dir}/update_${update}.attempt_"*.json; do
        basename="$(basename "${path}")"
        if [[ "${basename}" =~ ^update_${update}[.]attempt_([0-9]+)[.](log|json)$ ]]; then
            attempt=$((10#${BASH_REMATCH[1]}))
            if (( attempt > max_attempt )); then
                max_attempt="${attempt}"
            fi
        fi
    done
    printf '%s\n' "${max_attempt}"
}

write_attempt_record() {
    local output_path="$1"
    local update="$2"
    local checkpoint_path="$3"
    local gpu="$4"
    local port="$5"
    local attempt="$6"
    local status="$7"
    local exit_code="$8"
    local run_dir="$9"
    local log_path="${10}"
    local validation_log="${11}"
    local started_at="${12}"
    "${queue_python}" - \
        "${output_path}" \
        "${update}" \
        "${checkpoint_path}" \
        "${gpu}" \
        "${port}" \
        "${attempt}" \
        "${status}" \
        "${exit_code}" \
        "${run_dir}" \
        "${log_path}" \
        "${validation_log}" \
        "${started_at}" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

(
    output_path,
    update,
    checkpoint_path,
    gpu,
    port,
    attempt,
    status,
    exit_code,
    run_dir,
    log_path,
    validation_log,
    started_at,
) = sys.argv[1:]
record = {
    "event": "af3_pxm_eval_attempt",
    "completed_updates": int(update),
    "checkpoint_path": os.path.abspath(checkpoint_path),
    "gpu": gpu,
    "master_port": int(port),
    "attempt": int(attempt),
    "status": status,
    "evaluator_exit_code": int(exit_code),
    "run_dir": os.path.abspath(run_dir) if run_dir else None,
    "log_path": os.path.abspath(log_path),
    "validation_log": os.path.abspath(validation_log),
    "started_at": started_at,
    "finished_at": datetime.now(timezone.utc).isoformat(),
}
destination = Path(output_path)
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, destination)
except BaseException:
    try:
        os.unlink(temporary_name)
    except FileNotFoundError:
        pass
    raise
PY
}

declare -A validated_done=()
done_is_valid() {
    local update="$1"
    local checkpoint_path="$2"
    local done_marker="${queue_state_dir}/update_${update}.done.json"
    local validation_log="${queue_state_dir}/update_${update}.done_revalidation.log"
    if [[ -n "${validated_done[${update}]:-}" ]]; then
        return 0
    fi
    if [ ! -s "${done_marker}" ]; then
        return 1
    fi
    if "${queue_python}" "${result_validator}" \
        --done-marker "${done_marker}" \
        --pdb-list "${pdb_list}" \
        --checkpoint-path "${checkpoint_path}" \
        --completed-updates "${update}" \
        --scope "${eval_scope}" \
        --subset "${eval_subset}" \
        --mode "${eval_mode}" \
        > /dev/null 2> "${validation_log}"; then
        validated_done["${update}"]=1
        return 0
    fi
    echo "invalid_done_marker update=${update} marker=${done_marker} details=${validation_log}" >&2
    return 1
}

evaluate_update() {
    local update="$1"
    local checkpoint_path="$2"
    local gpu="$3"
    local master_port="$4"
    local claim_dir="$5"
    local prior_attempt attempt eval_log validation_log attempt_record
    local eval_run_name started_at evaluator_exit run_dir attempt_status

    prior_attempt="$(latest_attempt_number "${update}")"
    attempt=$((prior_attempt + 1))
    if (( attempt > max_attempts )); then
        echo "retry_budget_exhausted update=${update} attempts=${prior_attempt}" >&2
        return 1
    fi

    while (( attempt <= max_attempts )); do
        eval_log="${queue_log_dir}/update_${update}.attempt_${attempt}.log"
        validation_log="${queue_log_dir}/update_${update}.attempt_${attempt}.validation.log"
        attempt_record="${queue_state_dir}/update_${update}.attempt_${attempt}.json"
        eval_run_name="af3_pxm_full_combined_formal_u${update}_a${attempt}"
        started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "starting_eval update=${update} attempt=${attempt}/${max_attempts} gpu=${gpu} master_port=${master_port}"
        if CUDA_VISIBLE_DEVICES="${gpu}" \
            MASTER_PORT="${master_port}" \
            EMA_CHECKPOINT="${checkpoint_path}" \
            EVAL_SCOPE="${eval_scope}" \
            EVAL_SUBSET="${eval_subset}" \
            EVAL_MODE="${eval_mode}" \
            EVAL_BASE_DIR="${eval_base_dir}" \
            RUN_NAME="${eval_run_name}" \
            bash "${evaluator_script}" > "${eval_log}" 2>&1; then
            evaluator_exit=0
        else
            evaluator_exit=$?
        fi
        run_dir="$(extract_run_dir "${eval_log}")"
        attempt_status="evaluator_failed"
        if (( evaluator_exit == 0 )) && [ -n "${run_dir}" ]; then
            if "${queue_python}" "${result_validator}" \
                --run-dir "${run_dir}" \
                --log-path "${eval_log}" \
                --pdb-list "${pdb_list}" \
                --checkpoint-path "${checkpoint_path}" \
                --completed-updates "${update}" \
                --scope "${eval_scope}" \
                --subset "${eval_subset}" \
                --mode "${eval_mode}" \
                --output "${queue_state_dir}/update_${update}.done.json" \
                > /dev/null 2> "${validation_log}"; then
                attempt_status="completed"
                write_attempt_record \
                    "${attempt_record}" "${update}" "${checkpoint_path}" \
                    "${gpu}" "${master_port}" "${attempt}" "${attempt_status}" \
                    "${evaluator_exit}" "${run_dir}" "${eval_log}" \
                    "${validation_log}" "${started_at}"
                echo "completed_eval update=${update} attempt=${attempt} run_dir=${run_dir}"
                return 0
            fi
            attempt_status="validation_failed"
        elif (( evaluator_exit == 0 )); then
            attempt_status="run_dir_missing"
            printf 'Evaluator log did not contain the structured run directory line.\n' \
                > "${validation_log}"
        else
            printf 'Evaluator exited with status %s.\n' "${evaluator_exit}" \
                > "${validation_log}"
        fi
        write_attempt_record \
            "${attempt_record}" "${update}" "${checkpoint_path}" \
            "${gpu}" "${master_port}" "${attempt}" "${attempt_status}" \
            "${evaluator_exit}" "${run_dir}" "${eval_log}" \
            "${validation_log}" "${started_at}"
        echo "failed_eval update=${update} attempt=${attempt} status=${attempt_status} details=${validation_log}" >&2
        if (( attempt >= max_attempts )); then
            return 1
        fi
        echo "retry_wait update=${update} seconds=${retry_delay_seconds}"
        sleep "${retry_delay_seconds}"
        attempt=$((attempt + 1))
    done
    return 1
}

training_status="$(read_training_status)"
if [ "${dry_run}" = "1" ]; then
    echo "queue_dry_run source_run=${source_label} status=${training_status} namespace=${state_namespace}"
    echo "queue_dry_run gpus=${gpu_csv} master_port_base=${master_port_base} max_parallel=${#eval_gpus[@]} min_update=${min_update} prioritize_latest=${prioritize_latest}"
    planned_count=0
    lane_index=0
    while IFS=$'\t' read -r update checkpoint_path; do
        [ -n "${update}" ] || continue
        done_marker="${queue_state_dir}/update_${update}.done.json"
        if [ -s "${done_marker}" ]; then
            echo "existing_done_candidate update=${update} marker=${done_marker}"
            continue
        fi
        gpu="${eval_gpus[${lane_index}]}"
        master_port=$((master_port_base + lane_index))
        echo "planned_eval update=${update} checkpoint=${checkpoint_path} gpu=${gpu} master_port=${master_port}"
        printf 'CUDA_VISIBLE_DEVICES=%q MASTER_PORT=%q EMA_CHECKPOINT=%q EVAL_SCOPE=full EVAL_SUBSET=combined EVAL_MODE=formal EVAL_BASE_DIR=%q RUN_NAME=%q bash %q\n' \
            "${gpu}" \
            "${master_port}" \
            "${checkpoint_path}" \
            "${eval_base_dir}" \
            "af3_pxm_full_combined_formal_u${update}_a1" \
            "${evaluator_script}"
        planned_count=$((planned_count + 1))
        lane_index=$(((lane_index + 1) % ${#eval_gpus[@]}))
    done < <(discover_checkpoints "${training_status}")
    echo "queue_dry_run_complete planned=${planned_count}"
    exit 0
fi

mkdir -p "${queue_state_dir}" "${queue_log_dir}" "${claim_root}"
exec 9> "${queue_state_dir}/queue.lock"
if ! flock -n 9; then
    echo "Another AF3 PXM queue already owns ${queue_state_dir}/queue.lock." >&2
    exit 2
fi

# The exclusive dispatcher lock proves that any claim left by an earlier
# dispatcher is stale. Claims are empty directories, so rmdir is sufficient.
shopt -s nullglob
for stale_claim in "${claim_root}"/update_*.claim; do
    rmdir "${stale_claim}" 2>/dev/null || true
done

declare -a lane_pids=()
declare -a lane_updates=()
declare -a lane_worker_results=()
for ((lane_index = 0; lane_index < ${#eval_gpus[@]}; lane_index++)); do
    lane_pids[${lane_index}]=""
    lane_updates[${lane_index}]=""
    lane_worker_results[${lane_index}]=""
done

active_lane_count() {
    local lane count=0
    for ((lane = 0; lane < ${#eval_gpus[@]}; lane++)); do
        if [ -n "${lane_pids[${lane}]}" ]; then
            count=$((count + 1))
        fi
    done
    printf '%s\n' "${count}"
}

failure_seen=0
while true; do
    for ((lane_index = 0; lane_index < ${#eval_gpus[@]}; lane_index++)); do
        worker_result="${lane_worker_results[${lane_index}]}"
        if [ -z "${worker_result}" ] || [ ! -s "${worker_result}" ]; then
            continue
        fi
        pid="${lane_pids[${lane_index}]}"
        update="${lane_updates[${lane_index}]}"
        worker_exit="$(tr -d '[:space:]' < "${worker_result}")"
        if wait "${pid}"; then
            wait_exit=0
        else
            wait_exit=$?
        fi
        if [ "${worker_exit}" != "0" ] || (( wait_exit != 0 )); then
            failure_seen=1
            echo "queue_update_failed update=${update} gpu=${eval_gpus[${lane_index}]}" >&2
        else
            validated_done["${update}"]=1
        fi
        lane_pids[${lane_index}]=""
        lane_updates[${lane_index}]=""
        lane_worker_results[${lane_index}]=""
        rm -f "${worker_result}"
    done

    active_count="$(active_lane_count)"
    if (( failure_seen != 0 )); then
        if (( active_count == 0 )); then
            echo "queue_stopped_after_failure" >&2
            exit 1
        fi
        sleep "${dispatch_poll_seconds}"
        continue
    fi

    training_status="$(read_training_status)"
    pending_count=0
    while IFS=$'\t' read -r update checkpoint_path; do
        [ -n "${update}" ] || continue
        if done_is_valid "${update}" "${checkpoint_path}"; then
            continue
        fi
        claim_dir="${claim_root}/update_${update}.claim"
        if [ -d "${claim_dir}" ]; then
            pending_count=$((pending_count + 1))
            continue
        fi
        free_lane=-1
        for ((lane_index = 0; lane_index < ${#eval_gpus[@]}; lane_index++)); do
            if [ -z "${lane_pids[${lane_index}]}" ]; then
                free_lane="${lane_index}"
                break
            fi
        done
        if (( free_lane < 0 )); then
            pending_count=$((pending_count + 1))
            continue
        fi
        if ! mkdir "${claim_dir}" 2>/dev/null; then
            pending_count=$((pending_count + 1))
            continue
        fi
        gpu="${eval_gpus[${free_lane}]}"
        master_port=$((master_port_base + free_lane))
        worker_result="${queue_state_dir}/update_${update}.worker_result"
        rm -f "${worker_result}"
        (
            if evaluate_update \
                "${update}" "${checkpoint_path}" "${gpu}" \
                "${master_port}" "${claim_dir}"; then
                worker_exit=0
            else
                worker_exit=$?
            fi
            rmdir "${claim_dir}" 2>/dev/null || true
            worker_result_tmp="${worker_result}.tmp.$$"
            printf '%s\n' "${worker_exit}" > "${worker_result_tmp}"
            mv -f "${worker_result_tmp}" "${worker_result}"
            exit "${worker_exit}"
        ) &
        lane_pids[${free_lane}]=$!
        lane_updates[${free_lane}]="${update}"
        lane_worker_results[${free_lane}]="${worker_result}"
        active_count=$((active_count + 1))
        echo "claimed_update update=${update} gpu=${gpu} master_port=${master_port} pid=${lane_pids[${free_lane}]}"
    done < <(discover_checkpoints "${training_status}")

    active_count="$(active_lane_count)"
    if [ "${training_status}" = "completed" ] \
        && [ "${include_final}" = "true" ] \
        && [ ! -s "${train_run_dir}/checkpoints/$((final_update - 1))_ema_0.999.pt" ]; then
        final_missing=1
    else
        final_missing=0
    fi

    if (( active_count == 0 && pending_count == 0 )); then
        case "${training_status}" in
            completed)
                if (( final_missing != 0 )); then
                    echo "Training completed but final EMA checkpoint u${final_update} is missing." >&2
                    exit 1
                fi
                echo "queue_completed source_run=${source_label}"
                exit 0
                ;;
            failed)
                echo "Training failed; all formally published grid checkpoints were drained." >&2
                exit 1
                ;;
            running|unknown)
                if [ "${wait_for_checkpoints}" = "false" ]; then
                    echo "queue_snapshot_drained status=${training_status}"
                    exit 0
                fi
                echo "waiting_for_checkpoints status=${training_status} poll_seconds=${poll_seconds}"
                sleep "${poll_seconds}"
                ;;
        esac
    else
        sleep "${dispatch_poll_seconds}"
    fi
done
