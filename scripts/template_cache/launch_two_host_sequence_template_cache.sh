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

# Launch the deterministic sequence-template cache builder as two hash shards:
# shard 0 on the current h1003 shell and shard 1 over SSH on h1004.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/../.." && pwd)"

python_bin="${PYTHON_BIN:-/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python}"
data_root="${PROTENIX_DATA_ROOT:-/mnt/project_abd/public_databases/protenix.20260101}"
sequence_mapping="${SEQUENCE_MAPPING:-${data_root}/common/seq_to_pdb_index.json}"
source_root="${SOURCE_ROOT:-${data_root}/mmcif_msa_template}"
mmcif_dir="${MMCIF_DIR:-${data_root}/mmcif}"
obsolete_pdbs_path="${OBSOLETE_PDBS_PATH:-${data_root}/common/obsolete_to_successor.json}"
kalign_binary_path="${KALIGN_BINARY_PATH:-/mnt/project_abd/klp/miniconda3/envs/protenix/bin/kalign}"
cache_root="${CACHE_ROOT:-/mnt/project_abd/klp/databases/protenix_sequence_template_cache_top60_uniform20_id90_v1}"
smoke_cache_root="${SMOKE_CACHE_ROOT:-${cache_root}_smoke}"
remote_host="${REMOTE_HOST:-h1004}"
workers="${WORKERS:-128}"
smoke_workers="${SMOKE_WORKERS:-4}"
smoke_limit="${SMOKE_LIMIT:-4}"
progress_every="${PROGRESS_EVERY:-25}"
max_tasks_per_child="${MAX_TASKS_PER_CHILD:-0}"
builder="${repo_dir}/scripts/template_cache/build_sequence_template_cache.py"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/template_cache/launch_two_host_sequence_template_cache.sh preflight
  bash scripts/template_cache/launch_two_host_sequence_template_cache.sh smoke
  bash scripts/template_cache/launch_two_host_sequence_template_cache.sh start
  bash scripts/template_cache/launch_two_host_sequence_template_cache.sh status
  bash scripts/template_cache/launch_two_host_sequence_template_cache.sh stop

Actions:
  preflight  Validate both hosts and all required shared inputs.
  smoke      Foreground two-host smoke: 4 workers and 4 sequences per shard by default.
             Outputs go to an independent *_smoke cache root.
  start      Background full build: 128 workers per host by default, with --resume.
  status     Report PIDs, per-shard stats JSON, and recent log lines.
  stop       Send TERM to the recorded shard processes.

All paths/counts can be overridden with the uppercase environment variables
defined near the top of this script.
EOF
}

shell_quote() {
    printf '%q' "$1"
}

check_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${name} must be a positive integer, got ${value@Q}" >&2
        exit 2
    fi
}

preflight_local() {
    for file in \
        "${python_bin}" \
        "${kalign_binary_path}" \
        "${sequence_mapping}" \
        "${obsolete_pdbs_path}" \
        "${builder}"; do
        if [[ ! -f "${file}" ]]; then
            echo "Missing required file on local host: ${file}" >&2
            return 1
        fi
    done
    for directory in "${source_root}" "${mmcif_dir}"; do
        if [[ ! -d "${directory}" ]]; then
            echo "Missing required directory on local host: ${directory}" >&2
            return 1
        fi
    done
}

preflight_remote() {
    local command
    printf -v command \
        'test -f %s && test -f %s && test -f %s && test -f %s && test -f %s && test -d %s && test -d %s' \
        "$(shell_quote "${python_bin}")" \
        "$(shell_quote "${kalign_binary_path}")" \
        "$(shell_quote "${sequence_mapping}")" \
        "$(shell_quote "${obsolete_pdbs_path}")" \
        "$(shell_quote "${builder}")" \
        "$(shell_quote "${source_root}")" \
        "$(shell_quote "${mmcif_dir}")"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${remote_host}" "${command}"
}

preflight() {
    check_positive_integer WORKERS "${workers}"
    check_positive_integer SMOKE_WORKERS "${smoke_workers}"
    check_positive_integer SMOKE_LIMIT "${smoke_limit}"
    check_positive_integer PROGRESS_EVERY "${progress_every}"
    preflight_local
    preflight_remote
    echo "Preflight passed: local shard 0 + ${remote_host} shard 1."
}

builder_command() {
    local shard_id="$1"
    local selected_workers="$2"
    local selected_cache_root="$3"
    local selected_limit="${4:-}"
    local -a command=(
        "${python_bin}"
        "${builder}"
        --sequence-mapping "${sequence_mapping}"
        --source-root "${source_root}"
        --cache-root "${selected_cache_root}"
        --mmcif-dir "${mmcif_dir}"
        --kalign-binary-path "${kalign_binary_path}"
        --obsolete-pdbs-path "${obsolete_pdbs_path}"
        --workers "${selected_workers}"
        --shard-id "${shard_id}"
        --num-shards 2
        --progress-every "${progress_every}"
        --max-tasks-per-child "${max_tasks_per_child}"
        --resume
    )
    if [[ -n "${selected_limit}" ]]; then
        command+=(--limit "${selected_limit}")
    fi
    printf '%q ' "${command[@]}"
}

pid_path() {
    local selected_cache_root="$1"
    local shard_id="$2"
    printf '%s/_manifests/shard-%05d-of-00002.pid\n' \
        "${selected_cache_root}" "${shard_id}"
}

log_path() {
    local selected_cache_root="$1"
    local shard_id="$2"
    printf '%s/_logs/shard-%05d-of-00002.log\n' \
        "${selected_cache_root}" "${shard_id}"
}

stats_path() {
    local selected_cache_root="$1"
    local shard_id="$2"
    printf '%s/_manifests/stats.shard-%05d-of-00002.json\n' \
        "${selected_cache_root}" "${shard_id}"
}

is_local_running() {
    local pid_file="$1"
    [[ -s "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null
}

is_remote_running() {
    local pid_file="$1"
    [[ -s "${pid_file}" ]] || return 1
    local pid
    pid="$(cat "${pid_file}")"
    ssh -o BatchMode=yes -o ConnectTimeout=5 "${remote_host}" \
        "kill -0 $(shell_quote "${pid}") 2>/dev/null"
}

run_smoke() {
    preflight
    mkdir -p \
        "${smoke_cache_root}/_logs" \
        "${smoke_cache_root}/_manifests"
    local log0 log1 command0 command1
    log0="$(log_path "${smoke_cache_root}" 0)"
    log1="$(log_path "${smoke_cache_root}" 1)"
    command0="$(builder_command 0 "${smoke_workers}" "${smoke_cache_root}" "${smoke_limit}")"
    command1="$(builder_command 1 "${smoke_workers}" "${smoke_cache_root}" "${smoke_limit}")"

    echo "Starting smoke shard 0 locally: ${log0}"
    (
        cd "${repo_dir}"
        bash -lc "exec ${command0}"
    ) >"${log0}" 2>&1 &
    local local_pid=$!
    echo "${local_pid}" >"$(pid_path "${smoke_cache_root}" 0)"

    echo "Starting smoke shard 1 on ${remote_host}: ${log1}"
    ssh "${remote_host}" \
        "cd $(shell_quote "${repo_dir}") && exec ${command1}" \
        >"${log1}" 2>&1 &
    local ssh_pid=$!
    echo "${ssh_pid}" >"$(pid_path "${smoke_cache_root}" 1)"

    local local_status=0
    local remote_status=0
    wait "${local_pid}" || local_status=$?
    wait "${ssh_pid}" || remote_status=$?
    if (( local_status != 0 || remote_status != 0 )); then
        echo "Smoke failed: shard0=${local_status}, shard1=${remote_status}" >&2
        tail -n 30 "${log0}" "${log1}" >&2 || true
        return 1
    fi
    echo "Smoke passed. Artifacts: ${smoke_cache_root}"
}

start_full() {
    preflight
    mkdir -p "${cache_root}/_logs" "${cache_root}/_manifests"
    local pid0 pid1 log0 log1 command0 command1
    pid0="$(pid_path "${cache_root}" 0)"
    pid1="$(pid_path "${cache_root}" 1)"
    log0="$(log_path "${cache_root}" 0)"
    log1="$(log_path "${cache_root}" 1)"
    if is_local_running "${pid0}"; then
        echo "Local shard 0 is already running with PID $(cat "${pid0}")." >&2
        exit 1
    fi
    if is_remote_running "${pid1}"; then
        echo "Remote shard 1 is already running with PID $(cat "${pid1}")." >&2
        exit 1
    fi
    command0="$(builder_command 0 "${workers}" "${cache_root}")"
    command1="$(builder_command 1 "${workers}" "${cache_root}")"

    local local_command
    printf -v local_command \
        'echo $$ >%s; cd %s || exit 1; exec %s >>%s 2>&1 </dev/null' \
        "$(shell_quote "${pid0}")" \
        "$(shell_quote "${repo_dir}")" \
        "${command0}" \
        "$(shell_quote "${log0}")"
    setsid -f bash -lc "${local_command}"

    local remote_command
    printf -v remote_command \
        'cd %s || exit 1; mkdir -p %s %s; nohup bash -lc %s >>%s 2>&1 </dev/null & child=$!; echo "$child" >%s' \
        "$(shell_quote "${repo_dir}")" \
        "$(shell_quote "${cache_root}/_logs")" \
        "$(shell_quote "${cache_root}/_manifests")" \
        "$(shell_quote "exec ${command1}")" \
        "$(shell_quote "${log1}")" \
        "$(shell_quote "${pid1}")"
    ssh "${remote_host}" "${remote_command}"

    echo "Full build started:"
    echo "  shard 0 local PID $(cat "${pid0}"), log ${log0}"
    echo "  shard 1 ${remote_host} PID $(cat "${pid1}"), log ${log1}"
}

show_one_status() {
    local shard_id="$1"
    local host_label="$2"
    local pid_file log_file stats_file
    pid_file="$(pid_path "${cache_root}" "${shard_id}")"
    log_file="$(log_path "${cache_root}" "${shard_id}")"
    stats_file="$(stats_path "${cache_root}" "${shard_id}")"
    echo "===== shard ${shard_id} (${host_label}) ====="
    if [[ -s "${pid_file}" ]]; then
        echo "pid=$(cat "${pid_file}")"
    else
        echo "pid=missing"
    fi
    if [[ -f "${stats_file}" ]]; then
        "${python_bin}" -m json.tool "${stats_file}" || true
    else
        echo "stats=pending (${stats_file})"
    fi
    if [[ -f "${log_file}" ]]; then
        tail -n 8 "${log_file}"
    else
        echo "log=pending (${log_file})"
    fi
}

show_status() {
    local pid0 pid1
    pid0="$(pid_path "${cache_root}" 0)"
    pid1="$(pid_path "${cache_root}" 1)"
    if is_local_running "${pid0}"; then
        echo "shard0_process=running"
    else
        echo "shard0_process=not_running"
    fi
    if is_remote_running "${pid1}"; then
        echo "shard1_process=running"
    else
        echo "shard1_process=not_running_or_unreachable"
    fi
    show_one_status 0 "local/h1003"
    show_one_status 1 "${remote_host}"
}

stop_full() {
    local pid0 pid1
    pid0="$(pid_path "${cache_root}" 0)"
    pid1="$(pid_path "${cache_root}" 1)"
    if is_local_running "${pid0}"; then
        kill -TERM "$(cat "${pid0}")"
        echo "Sent TERM to local shard 0."
    fi
    if is_remote_running "${pid1}"; then
        ssh "${remote_host}" "kill -TERM $(shell_quote "$(cat "${pid1}")")"
        echo "Sent TERM to remote shard 1."
    fi
}

action="${1:-}"
case "${action}" in
    preflight)
        preflight
        ;;
    smoke)
        run_smoke
        ;;
    start)
        start_full
        ;;
    status)
        show_status
        ;;
    stop)
        stop_full
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
