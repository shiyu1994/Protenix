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
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_profile_registry="${repo_dir}/configs/crop384_db48_data_profiles.sh"
# shellcheck disable=SC1090
source "${data_profile_registry}"
locked_data_profile="${PROTENIX_LOCKED_DATA_PROFILE:-weighted_pdb_only_v1}"
if [ -n "${PROTENIX_TRAINING_DATA_PROFILE:-}" ] \
    && [ "${PROTENIX_TRAINING_DATA_PROFILE}" != "${locked_data_profile}" ]; then
    echo "This resume launcher locks PROTENIX_TRAINING_DATA_PROFILE=${locked_data_profile}; got ${PROTENIX_TRAINING_DATA_PROFILE}." >&2
    exit 2
fi
resolve_crop384_db48_data_profile "${locked_data_profile}"
export PROTENIX_TRAINING_DATA_PROFILE="${CROP384_DATA_PROFILE_NAME}"

# Fail before checkpoint inspection if this node cannot provide the exact
# CUDA/PyTorch extension toolchain used by the training contract.
# shellcheck disable=SC1091
source "${repo_dir}/scripts/runtime/cuda126_env.sh"

if (( $# != 0 )); then
    echo "This locked strict-resume launcher accepts no command-line overrides." >&2
    exit 2
fi

resolved_run_name="${RUN_NAME:-${run_name:-}}"
if [ -z "${resolved_run_name}" ]; then
    echo "RUN_NAME is required and must name a new resumed output run." >&2
    exit 2
fi
if [[ ! "${resolved_run_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
fi
source_run_dir="${RESUME_RUN_DIR:-}"
resume_step="${RESUME_STEP:-}"
if [ -z "${source_run_dir}" ]; then
    echo "RESUME_RUN_DIR is required; strict resume never guesses a source run." >&2
    exit 2
fi
if [[ ! "${resume_step}" =~ ^[0-9]+$ ]]; then
    echo "RESUME_STEP must be a non-negative integer." >&2
    exit 2
fi

if [[ -n "${START_STEP:-}" && ! "${START_STEP}" =~ ^[0-9]+$ ]]; then
    echo "START_STEP must be a non-negative integer." >&2
    exit 2
fi
target_max_steps="${MAX_STEPS:-170000}"
if [[ ! "${target_max_steps}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_STEPS must be a positive integer." >&2
    exit 2
fi

resume_raw_checkpoint="${RESUME_RAW_CHECKPOINT:-${source_run_dir}/checkpoints/${resume_step}.pt}"
default_ema_checkpoint="${source_run_dir}/checkpoints/${resume_step}_ema_0.999.pt"
resume_ema_checkpoint="${RESUME_EMA_CHECKPOINT:-${default_ema_checkpoint}}"
if [[ "$(basename "${resume_raw_checkpoint}")" == *"_ema_"* ]]; then
    echo "RESUME_RAW_CHECKPOINT must not point to an EMA checkpoint." >&2
    exit 2
fi
if [[ "$(basename "${resume_ema_checkpoint}")" != *"_ema_0.999.pt" ]]; then
    echo "RESUME_EMA_CHECKPOINT must identify the matching EMA=0.999 snapshot." >&2
    exit 2
fi
if [ "$(realpath -m "${resume_raw_checkpoint}")" = "$(realpath -m "${resume_ema_checkpoint}")" ]; then
    echo "Raw and EMA checkpoint paths must be distinct." >&2
    exit 2
fi

if [ "${DRY_RUN:-0}" != "1" ] && [ "${SKIP_PATH_CHECK:-0}" != "1" ]; then
    for checkpoint_path in "${resume_raw_checkpoint}" "${resume_ema_checkpoint}"; do
        if [ ! -s "${checkpoint_path}" ]; then
            echo "Required non-empty checkpoint does not exist: ${checkpoint_path}" >&2
            exit 2
        fi
    done
fi

resume_smoke_steps="${RESUME_SMOKE_STEPS:-0}"
if [[ ! "${resume_smoke_steps}" =~ ^[0-9]+$ ]]; then
    echo "RESUME_SMOKE_STEPS must be a non-negative integer." >&2
    exit 2
fi
# Smoke length is applied by the trainer after reading the actual checkpoint
# state and performing any first-launch rebase, not inferred from its filename.

# Ignore stale generic torchrun variables and resolve the active topology from
# explicit resume settings plus AIStudio/SLURM coordinates. The runner then
# compares the resulting world size to the raw checkpoint contract and every
# saved rank-state entry before applying optimizer/model state.
# This launcher is already locked to the checkpoint's 16x8 topology.  Some
# AIStudio jobs do not publish MLP_WORKER_NUM on every worker, so use the
# locked node count as the final fallback and still reject every conflicting
# explicit platform value below.
resume_nnodes="${RESUME_NNODES:-${MLP_WORKER_NUM:-${SLURM_NNODES:-16}}}"
resume_nproc_per_node="${RESUME_NPROC_PER_NODE:-8}"
if [[ ! "${resume_nnodes}" =~ ^[1-9][0-9]*$ ]]; then
    echo "The 128-GPU resume contract requires a positive node count." >&2
    exit 2
fi
if [[ ! "${resume_nproc_per_node}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RESUME_NPROC_PER_NODE must be a positive integer." >&2
    exit 2
fi
if [ -n "${MLP_WORKER_NUM:-}" ] \
    && [ "${MLP_WORKER_NUM}" != "${resume_nnodes}" ]; then
    echo "RESUME_NNODES disagrees with AIStudio MLP_WORKER_NUM." >&2
    exit 2
fi
if [ -n "${SLURM_NNODES:-}" ] \
    && [ "${SLURM_NNODES}" != "${resume_nnodes}" ]; then
    echo "RESUME_NNODES disagrees with SLURM_NNODES." >&2
    exit 2
fi
resume_world_size=$((resume_nnodes * resume_nproc_per_node))
if (( resume_nnodes != 16 \
    || resume_nproc_per_node != 8 \
    || resume_world_size != 128 )); then
    echo "Strict resume requires topology=16x8 and world_size=128; resolved ${resume_nnodes}x${resume_nproc_per_node}=${resume_world_size}." >&2
    exit 2
fi

if [ "${resume_nnodes}" = "1" ]; then
    resume_node_rank="${RESUME_NODE_RANK:-0}"
    resume_node_rank_source="RESUME_NODE_RANK_or_single_node_default"
    resume_master_addr="${RESUME_MASTER_ADDR:-127.0.0.1}"
    resume_master_addr_source="RESUME_MASTER_ADDR_or_single_node_default"
else
    # AIStudio normally provides MLP_ROLE_INDEX and MLP_WORKER_0_HOST.  In
    # practice either variable can be absent on an individual worker even
    # though the worker hostname remains stable as "...-worker-N".  Accept
    # that exact suffix only as a last-resort coordinate; never turn an
    # arbitrary hostname into rank 0.
    resume_hostname="${HOSTNAME:-$(hostname)}"
    resume_hostname_rank=""
    resume_hostname_master_addr=""
    if [[ "${resume_hostname}" =~ ^(.+)-worker-([0-9]+)$ ]]; then
        resume_hostname_master_addr="${BASH_REMATCH[1]}-worker-0"
        resume_hostname_rank=$((10#${BASH_REMATCH[2]}))
    fi

    if [ -n "${MLP_ROLE_INDEX:-}" ]; then
        resume_node_rank="${MLP_ROLE_INDEX}"
        resume_node_rank_source="MLP_ROLE_INDEX"
    elif [ -n "${SLURM_NODEID:-}" ]; then
        resume_node_rank="${SLURM_NODEID}"
        resume_node_rank_source="SLURM_NODEID"
    elif [ -n "${RESUME_NODE_RANK:-}" ]; then
        resume_node_rank="${RESUME_NODE_RANK}"
        resume_node_rank_source="RESUME_NODE_RANK"
    else
        resume_node_rank="${resume_hostname_rank}"
        resume_node_rank_source="hostname_suffix"
    fi

    if [ -n "${MLP_WORKER_0_HOST:-}" ]; then
        resume_master_addr="${MLP_WORKER_0_HOST}"
        resume_master_addr_source="MLP_WORKER_0_HOST"
    elif [ -n "${RESUME_MASTER_ADDR:-}" ]; then
        resume_master_addr="${RESUME_MASTER_ADDR}"
        resume_master_addr_source="RESUME_MASTER_ADDR"
    else
        resume_master_addr="${resume_hostname_master_addr}"
        resume_master_addr_source="hostname_worker_0"
    fi
    if [[ ! "${resume_node_rank}" =~ ^[0-9]+$ ]] \
        || (( resume_node_rank >= resume_nnodes )); then
        echo "Multi-node resume requires a valid explicit node rank or an exact hostname suffix '-worker-N'." >&2
        exit 2
    fi
    if [ -z "${resume_master_addr}" ] \
        || [ "${resume_master_addr}" = "127.0.0.1" ]; then
        echo "Multi-node resume requires a master address or an exact hostname suffix '-worker-N'." >&2
        exit 2
    fi
fi
if [[ ! "${resume_node_rank}" =~ ^[0-9]+$ ]] \
    || (( resume_node_rank >= resume_nnodes )); then
    echo "Resolved resume node rank is outside the active topology." >&2
    exit 2
fi

resolved_master_port="${MLP_WORKER_0_PORT:-${RESUME_MASTER_PORT:-${MASTER_PORT:-29500}}}"
if [[ ! "${resolved_master_port}" =~ ^[1-9][0-9]*$ ]] \
    || (( resolved_master_port > 65535 )); then
    echo "RESUME_MASTER_PORT must be in [1, 65535]." >&2
    exit 2
fi

export RUN_NAME="${resolved_run_name}"
export run_name="${resolved_run_name}"
export NNODES="${resume_nnodes}"
export nnodes="${resume_nnodes}"
export NODE_RANK="${resume_node_rank}"
export node_rank="${resume_node_rank}"
export NPROC_PER_NODE="${resume_nproc_per_node}"
export nproc_per_node="${resume_nproc_per_node}"
export MASTER_ADDR="${resume_master_addr}"
export master_addr="${MASTER_ADDR}"
export MASTER_PORT="${resolved_master_port}"
export master_port="${resolved_master_port}"

soft_cluster_schedule_dir="${PROTENIX_SOFT_CLUSTER_SCHEDULE_DIR:-}"
soft_cluster_target_contract="${PROTENIX_SOFT_CLUSTER_TARGET_TRAINING_CONTRACT_ID:-}"
if [ -n "${soft_cluster_schedule_dir}" ]; then
    if [ -z "${soft_cluster_target_contract}" ]; then
        echo "Soft-cluster resume requires PROTENIX_SOFT_CLUSTER_TARGET_TRAINING_CONTRACT_ID." >&2
        exit 2
    fi
elif [ -n "${soft_cluster_target_contract}" ]; then
    echo "PROTENIX_SOFT_CLUSTER_TARGET_TRAINING_CONTRACT_ID requires PROTENIX_SOFT_CLUSTER_SCHEDULE_DIR." >&2
    exit 2
fi

export PROTENIX_CROP384_STRICT_RESUME_INTERNAL=1
export PROTENIX_FULL_RESUME=1
export PROTENIX_STRICT_FULL_RESUME=1
default_training_contract_id="crop384_db48_ddp_scratch_50k_ws128_acc4_data_${CROP384_DATA_PROFILE_CONTRACT_TAG}_arch_pf64_cs384_dct768_pfckpt8_c48_msamix_${CROP384_DATA_PROFILE_MSA_MIX_CONTRACT_TAG}_msaraw_${CROP384_DATA_PROFILE_MSA_RAW_CONTRACT_TAG}_msaquerypinle1024_templatecache_top60u20id90_chaindummy_ckpt200_dlpersist1_prefetch2_deferredhealth_v11"
export PROTENIX_TRAINING_CONTRACT_ID="${soft_cluster_target_contract:-${default_training_contract_id}}"
export PROTENIX_RESUME_RAW_CHECKPOINT="${resume_raw_checkpoint}"
export PROTENIX_RESUME_EMA_CHECKPOINT="${resume_ema_checkpoint}"
# RESUME_STEP selects a filename, not the payload's counter. The first source
# pair is a byte-copy under a new name; validate its raw/EMA state internally.
unset PROTENIX_RESUME_CHECKPOINT_STEP
export PROTENIX_RESUME_TARGET_MAX_STEPS="${target_max_steps}"
export PROTENIX_RESUME_SMOKE_STEPS="${resume_smoke_steps}"
export PROTENIX_SMOKE_TEST=0
export PROTENIX_STRAGGLER_BENCHMARK_UPDATES=0

echo "Locked crop384/db48 DDP strict full-state resume:"
echo "  source_run=${source_run_dir}"
echo "  resume=full checkpoint=${resume_step}"
echo "  raw_checkpoint=${resume_raw_checkpoint}"
echo "  ema_checkpoint=${resume_ema_checkpoint}"
echo "  state=model+optimizer+scheduler+step+scaler+rank_rng+matched_ema"
echo "  sampler=new schedule starts fresh once; matching schedule restores its saved cursor"
echo "  data_worker_rng=not_checkpointed num_dl_workers=8 persistent_workers=true prefetch_factor=2"
echo "  topology=${NNODES}x${NPROC_PER_NODE} world_size=$((NNODES * NPROC_PER_NODE))"
echo "  node_rank=${NODE_RANK} source=${resume_node_rank_source} "\
"master=${MASTER_ADDR}:${MASTER_PORT} master_source=${resume_master_addr_source}"
echo "  ending_checkpoint=$((target_max_steps - 1)) smoke_updates=${resume_smoke_steps}"
echo "  data_profile=${PROTENIX_TRAINING_DATA_PROFILE} "\
"train_sets=${CROP384_DATA_PROFILE_TRAIN_SETS} "\
"weights=${CROP384_DATA_PROFILE_TRAIN_SAMPLE_WEIGHTS}"
if [ -n "${CROP384_DATA_PROFILE_RUNTIME_INDEX_DIR}" ]; then
    echo "  distillation_runtime_index=${CROP384_DATA_PROFILE_RUNTIME_INDEX_DIR}"
fi
echo "  contract=${PROTENIX_TRAINING_CONTRACT_ID} fail_closed=true"
if [ -n "${soft_cluster_schedule_dir}" ]; then
    echo "  soft_cluster_schedule=${soft_cluster_schedule_dir} gamma=0.2"
fi

exec bash "${repo_dir}/run_crop384_db48_ddp_scratch_200k.sh"
