#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"

# Each execution must use a fresh ID because benchmark artifacts are immutable.
: "${BENCHMARK_ID:?Set BENCHMARK_ID to a new unique value}"

phase="${MSA2048_BENCHMARK_PHASE:-smoke}"
case "${phase}" in
    smoke)
        workload_profile=crop1024_800_msa2048_offload_smoke
        ;;
    matched)
        workload_profile=crop1024_800_msa2048_offload_matched
        ;;
    full)
        workload_profile=crop1024_800_msa2048_offload
        ;;
    *)
        echo "MSA2048_BENCHMARK_PHASE must be smoke, matched, or full" >&2
        exit 2
        ;;
esac

export BENCHMARK_VARIANT=fsdp2_full
export BENCHMARK_WORKLOAD_PROFILE="${workload_profile}"
export BENCHMARK_OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-/mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2_msa2048_offload_crop1024}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

exec bash scripts/benchmark/run_aistudio_benchmark.sh
