#!/usr/bin/env bash
set -euo pipefail

repo_dir="/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train"
base_checkpoint="/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/checkpoints/999_ema_0.999.pt"

experiment_name="${EXPERIMENT_NAME:?Set EXPERIMENT_NAME, e.g. sampling_conservative}"
train_sample_weights="${TRAIN_SAMPLE_WEIGHTS:?Set TRAIN_SAMPLE_WEIGHTS, e.g. 0.4,0.4,0.2}"
train_sets="${TRAIN_SETS:-weightedPDB_before210930_v20260101,sabdab-train,distill-sabdab}"
lr="${LR:-0.000003}"
max_steps="${MAX_STEPS:-2000}"
checkpoint_interval="${CHECKPOINT_INTERVAL:-250}"
timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
run_name="${RUN_NAME:-protenix_sampling_${experiment_name}_from20260531_999ema_lr${lr}_${timestamp}}"

if [ ! -f "${base_checkpoint}" ]; then
  echo "Missing base checkpoint: ${base_checkpoint}" >&2
  exit 1
fi

cd "${repo_dir}"

export LOAD_CHECKPOINT_PATH="${base_checkpoint}"
export LOAD_EMA_CHECKPOINT_PATH="${base_checkpoint}"
export RUN_NAME="${run_name}"
export run_name="${run_name}"
export train_sets="${train_sets}"
export train_sample_weights="${train_sample_weights}"
export lr="${lr}"
export max_steps="${max_steps}"
export checkpoint_interval="${checkpoint_interval}"
export eval_interval=0
export EVAL_AT_END=false
export eval_at_end=false
export EVAL_ONLY=false
export eval_only=false

# AIStudio Pytorch-DDP exports WORLD_SIZE/RANK. finetune_demo.sh expects
# NNODES/NODE_RANK or older MLP/SLURM names before building torchrun args.
export NNODES="${NNODES:-${WORLD_SIZE:-${MLP_WORKER_NUM:-${SLURM_NNODES:-1}}}}"
export NODE_RANK="${NODE_RANK:-${RANK:-${MLP_ROLE_INDEX:-${SLURM_NODEID:-0}}}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-gpu}"

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

echo "Launching sampling finetune"
echo "  experiment_name=${experiment_name}"
echo "  run_name=${run_name}"
echo "  base_checkpoint=${base_checkpoint}"
echo "  train_sets=${train_sets}"
echo "  train_sample_weights=${train_sample_weights}"
echo "  lr=${lr}"
echo "  max_steps=${max_steps}"
echo "  checkpoint_interval=${checkpoint_interval}"
echo "  NNODES=${NNODES}"
echo "  NODE_RANK=${NODE_RANK}"
echo "  MASTER_ADDR=${MASTER_ADDR}"
echo "  MASTER_PORT=${MASTER_PORT}"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  eval_interval=${eval_interval}"
echo "  eval_at_end=${eval_at_end}"

exec bash "${repo_dir}/finetune_demo.sh"
