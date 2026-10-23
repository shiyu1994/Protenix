#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"

# Each execution must use a fresh ID because benchmark artifacts are immutable.
: "${BENCHMARK_ID:?Set BENCHMARK_ID to a new unique value}"

export BENCHMARK_VARIANT=fsdp2_full
export BENCHMARK_WORKLOAD_PROFILE=crop960_800_msa2048_offload
export BENCHMARK_OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-/mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2_msa2048_offload_crop960}"
# Reuse, and revalidate, the existing strict 800-row MSA2048 artifact bundle.
export BENCHMARK_DATASET_DIR="${BENCHMARK_DATASET_DIR:-${repo_dir}/data/benchmark_long_protein_context_1024_800_msa2048}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

exec bash scripts/benchmark/run_aistudio_benchmark.sh
