#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"

# The sampled MSA depth is fixed at 2048. A chunk size of 2048 therefore
# processes all MSA rows in one block and disables row-wise chunk splitting.
: "${BENCHMARK_ID:?Set BENCHMARK_ID to a new unique value}"

export BENCHMARK_VARIANT=fsdp2_full
export BENCHMARK_WORKLOAD_PROFILE=crop1024_800_msa2048_offload_nuc256_db32_chunk2048_10step
export BENCHMARK_OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-/mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2_msa2048_offload_crop1024_nuc256_db32_chunk2048_10step}"
export BENCHMARK_DATASET_DIR="${BENCHMARK_DATASET_DIR:-${repo_dir}/data/benchmark_long_protein_context_1024_800_msa2048}"
export BENCHMARK_WARMUP_STEPS=2
export BENCHMARK_MEASURE_STEPS=8
export BENCHMARK_TRAIN_SAMPLER_SEED=77
export BENCHMARK_MSA_CHUNK_SIZE=2048
export BENCHMARK_DIFFUSION_BATCH_SIZE=32
export BENCHMARK_MAX_NUC_TOKENS=256
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

exec bash scripts/benchmark/run_aistudio_benchmark.sh
