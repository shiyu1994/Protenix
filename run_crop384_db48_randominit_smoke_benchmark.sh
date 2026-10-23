#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"

: "${BENCHMARK_ID:?Set BENCHMARK_ID to a new unique value}"

# DDP is the speed-oriented default. Set BENCHMARK_VARIANT=fsdp2_full to run
# the otherwise identical memory/reference variant.
export BENCHMARK_VARIANT="${BENCHMARK_VARIANT:-ddp_full}"
export BENCHMARK_WORKLOAD_PROFILE=crop384_800_msa2048_randominit_nuc128_db48_paired_chunk2048_smoke
default_output_root="/mnt/project_abd/klp/proteinx_benchmark/crop384_db48_randominit_fast"
export BENCHMARK_OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-${default_output_root}}"
audited_dataset_repo="${repo_dir%/*}/Protenix_train2_fsdp2_msa2048_offload"
audited_dataset_dir="${audited_dataset_repo}/data/benchmark_long_protein_context_1024_800_msa2048"
export BENCHMARK_DATASET_DIR="${BENCHMARK_DATASET_DIR:-${audited_dataset_dir}}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

exec bash scripts/benchmark/run_aistudio_benchmark.sh
