#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"

# Human step 31 is zero-based log step 30.  The locked sampler seed places
# 7d1a on rank5 at that final step while the global model/crop seed stays 42.
: "${BENCHMARK_ID:?Set BENCHMARK_ID to a new unique value}"

export BENCHMARK_VARIANT=fsdp2_full
export BENCHMARK_WORKLOAD_PROFILE=crop1024_800_msa2048_offload_nuc128_db32_31step_7d1a
export BENCHMARK_OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-/mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2_msa2048_offload_crop1024_nuc128_db32_31step_7d1a}"
export BENCHMARK_DATASET_DIR="${BENCHMARK_DATASET_DIR:-${repo_dir}/data/benchmark_long_protein_context_1024_800_msa2048}"
export BENCHMARK_TRAIN_SAMPLER_SEED=77
export BENCHMARK_DIFFUSION_BATCH_SIZE=32
export BENCHMARK_MAX_NUC_TOKENS=128
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

exec bash scripts/benchmark/run_aistudio_benchmark.sh
