#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"

# AIStudio Pytorch-DDP exports WORLD_SIZE/RANK. finetune_demo.sh expects
# NNODES/NODE_RANK or older MLP/SLURM names before building torchrun args.
export NNODES="${NNODES:-${WORLD_SIZE:-${MLP_WORKER_NUM:-${SLURM_NNODES:-1}}}}"
export NODE_RANK="${NODE_RANK:-${RANK:-${MLP_ROLE_INDEX:-${SLURM_NODEID:-0}}}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-gpu}"
export eval_interval=0
export EVAL_AT_END=false
export eval_at_end=false
export EVAL_ONLY=false
export eval_only=false

if [ -z "${MASTER_ADDR:-}" ]; then
    if [ -n "${MLP_WORKER_0_HOST:-}" ]; then
        export MASTER_ADDR="${MLP_WORKER_0_HOST}"
    elif [ "${NNODES}" = "1" ]; then
        export MASTER_ADDR="127.0.0.1"
    else
        echo "Multi-node training requires MASTER_ADDR from AIStudio Pytorch-DDP." >&2
        exit 1
    fi
fi

echo "Launching Protenix finetune:"
echo "  repo_dir=${repo_dir}"
echo "  NNODES=${NNODES}"
echo "  NODE_RANK=${NODE_RANK}"
echo "  MASTER_ADDR=${MASTER_ADDR}"
echo "  MASTER_PORT=${MASTER_PORT}"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  eval_interval=${eval_interval}"
echo "  eval_at_end=${eval_at_end}"

exec bash "${repo_dir}/finetune_demo2.sh" "$@"
