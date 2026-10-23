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
#
# Resume the crop1024/nuc128/db40/species-paired/local-attention experiment
# from a matched raw + EMA checkpoint pair.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if (( $# != 0 )); then
    echo "This resume launcher accepts no positional arguments; use environment variables." >&2
    exit 2
fi

if [ -z "${RUN_NAME:-}" ]; then
    echo "RUN_NAME is required and must identify a new resume output run." >&2
    exit 2
fi

default_source_run="/mnt/project_abd/klp/proteinx_output/immune_heavy_old_content_crop1024_nuc128_db40_pairedmsa_localattention_fsdp2_20260722_032221"
source_run_dir="${RESUME_RUN_DIR:-${default_source_run}}"
resume_step="${RESUME_STEP:-499}"
if [[ ! "${resume_step}" =~ ^[0-9]+$ ]]; then
    echo "RESUME_STEP must be a non-negative integer." >&2
    exit 2
fi

resume_raw_checkpoint="${RESUME_RAW_CHECKPOINT:-${source_run_dir}/checkpoints/${resume_step}.pt}"
resume_ema_checkpoint="${RESUME_EMA_CHECKPOINT:-${source_run_dir}/checkpoints/${resume_step}_ema_0.999.pt}"
resume_smoke_steps="${RESUME_SMOKE_STEPS:-0}"
if [[ ! "${resume_smoke_steps}" =~ ^[0-9]+$ ]]; then
    echo "RESUME_SMOKE_STEPS must be a non-negative integer." >&2
    exit 2
fi

export PROTENIX_FULL_RESUME=1
export PROTENIX_RESUME_CHECKPOINT_STEP="${resume_step}"
export PROTENIX_RESUME_RAW_CHECKPOINT="${resume_raw_checkpoint}"
export PROTENIX_RESUME_EMA_CHECKPOINT="${resume_ema_checkpoint}"
export PROTENIX_RESUME_SMOKE_STEPS="${resume_smoke_steps}"
export PROTENIX_SMOKE_TEST=0

# The formal schedule is an absolute total-step target, not another 2,000
# updates. A checkpoint at step 499 therefore continues with step 500 and ends
# when the global optimizer-step counter reaches 2,000.
export MAX_STEPS="${RESUME_TOTAL_MAX_STEPS:-2000}"
export max_steps="${MAX_STEPS}"

# Avoid the torch_shm_manager/file_system failure seen after step 534 and reduce
# the four-node worker count from 512 to 256 by default. Use a resume-specific
# override so stale generic settings from a reused shell cannot silently put
# this recovery back on file_system.
export PROTENIX_MP_SHARING_STRATEGY="${RESUME_MP_SHARING_STRATEGY:-file_descriptor}"
export PROTENIX_RESUME_NUM_DL_WORKERS="${RESUME_NUM_DL_WORKERS:-8}"

# The source run used 4 AIStudio workers with 8 GPUs each. Ignore stale generic
# torchrun variables from a reused shell and resolve topology from explicit
# resume settings plus AIStudio's current MLP variables. A local lifecycle
# smoke may deliberately set RESUME_NNODES=1; formal exact continuation uses 4.
resume_nnodes="${RESUME_NNODES:-${MLP_WORKER_NUM:-4}}"
resume_nproc_per_node="${RESUME_NPROC_PER_NODE:-8}"
if [[ ! "${resume_nnodes}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RESUME_NNODES must be a positive integer." >&2
    exit 2
fi
if [[ ! "${resume_nproc_per_node}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RESUME_NPROC_PER_NODE must be a positive integer." >&2
    exit 2
fi
if [ "${resume_smoke_steps}" = "0" ] && [ "${resume_nproc_per_node}" != "8" ]; then
    echo "Formal resume requires RESUME_NPROC_PER_NODE=8." >&2
    exit 2
fi
if [ "${resume_smoke_steps}" != "0" ] && [ "${resume_nnodes}" != "1" ] \
    && [ "${resume_nproc_per_node}" != "8" ]; then
    echo "Multi-node resume smoke requires RESUME_NPROC_PER_NODE=8." >&2
    exit 2
fi
if [ -n "${MLP_WORKER_NUM:-}" ] && [ "${MLP_WORKER_NUM}" != "${resume_nnodes}" ]; then
    echo "RESUME_NNODES=${resume_nnodes} disagrees with AIStudio MLP_WORKER_NUM=${MLP_WORKER_NUM}." >&2
    exit 2
fi

if [ "${resume_nnodes}" = "1" ]; then
    resume_node_rank=0
    resume_master_addr="${RESUME_MASTER_ADDR:-127.0.0.1}"
else
    resume_node_rank="${MLP_ROLE_INDEX:-${RESUME_NODE_RANK:-}}"
    resume_master_addr="${MLP_WORKER_0_HOST:-${RESUME_MASTER_ADDR:-}}"
    if [[ ! "${resume_node_rank}" =~ ^[0-9]+$ ]] \
        || [ "${resume_node_rank}" -ge "${resume_nnodes}" ]; then
        echo "Multi-node resume requires a valid AIStudio MLP_ROLE_INDEX or RESUME_NODE_RANK." >&2
        exit 2
    fi
    if [ -z "${resume_master_addr}" ] || [ "${resume_master_addr}" = "127.0.0.1" ]; then
        echo "Multi-node resume requires MLP_WORKER_0_HOST or RESUME_MASTER_ADDR." >&2
        exit 2
    fi
fi
resume_master_port="${MLP_WORKER_0_PORT:-${RESUME_MASTER_PORT:-29500}}"
if [[ ! "${resume_master_port}" =~ ^[1-9][0-9]*$ ]] \
    || [ "${resume_master_port}" -gt 65535 ]; then
    echo "Resolved resume master port must be in [1, 65535]." >&2
    exit 2
fi

unset nproc_per_node nnodes node_rank master_addr master_port
export NPROC_PER_NODE="${resume_nproc_per_node}"
export NNODES="${resume_nnodes}"
export NODE_RANK="${resume_node_rank}"
export MASTER_ADDR="${resume_master_addr}"
export MASTER_PORT="${resume_master_port}"

# Known digests for the default failed run. A custom source may provide its own
# values. Full hashing is opt-in because reading both multi-GB files on every
# AIStudio node is expensive; the runner still validates checkpoint structure,
# kind, model, optimizer, scheduler, and step during every full resume.
if [ "${resume_raw_checkpoint}" = "${default_source_run}/checkpoints/499.pt" ]; then
    default_raw_sha256="3681c271febfad4336e444b4ba00c275f754c557e75325f27dfefabdb9aa8416"
else
    default_raw_sha256=""
fi
if [ "${resume_ema_checkpoint}" = "${default_source_run}/checkpoints/499_ema_0.999.pt" ]; then
    default_ema_sha256="ded9c659d46bd859fbe6c0ab7fbbf8fafd0e81e5d97a07ed869336eaa4adafbd"
else
    default_ema_sha256=""
fi
export PROTENIX_RESUME_RAW_SHA256="${RESUME_RAW_SHA256:-${default_raw_sha256}}"
export PROTENIX_RESUME_EMA_SHA256="${RESUME_EMA_SHA256:-${default_ema_sha256}}"
export PROTENIX_VERIFY_RESUME_SHA256="${RESUME_VERIFY_SHA256:-0}"

echo "Locked resume topology: ${NNODES} node(s) x ${NPROC_PER_NODE} GPU(s), node_rank=${NODE_RANK}"
exec bash "${repo_dir}/run_immune_heavy_old_content_crop1024_nuc128_db40_fsdp2_finetune.sh"
