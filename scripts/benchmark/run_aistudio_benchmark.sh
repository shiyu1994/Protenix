#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_dir}"

# The outer benchmark process must record and use the same pinned CUDA 12.6
# toolchain as the training process.
# shellcheck disable=SC1091
source "${repo_dir}/scripts/runtime/cuda126_env.sh"

: "${BENCHMARK_ID:?Set one unique BENCHMARK_ID and use it on every AIStudio node}"
if [[ ! "${BENCHMARK_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid BENCHMARK_ID=${BENCHMARK_ID}" >&2
    exit 2
fi
if (( $# != 0 )); then
    echo "run_aistudio_benchmark.sh accepts no positional CLI overrides" >&2
    exit 2
fi

# This outer launcher does not inherit the Conda activation performed inside
# finetune_demo.sh. Pin post-training validation to the exact training
# environment instead of whichever Python happened to launch this shell.
benchmark_conda_activate="${CONDA_ACTIVATE:-/mnt/project_abd/klp/miniconda3/bin/activate}"
benchmark_conda_root="${benchmark_conda_activate%/bin/activate}"
benchmark_conda_env="${CONDA_ENV:-protenix}"
benchmark_python="${BENCHMARK_PYTHON:-${benchmark_conda_root}/envs/${benchmark_conda_env}/bin/python}"
if [ ! -x "${benchmark_python}" ]; then
    echo "Missing benchmark Python: ${benchmark_python}" >&2
    exit 2
fi

node_rank="${NODE_RANK:-${MLP_ROLE_INDEX:-${SLURM_NODEID:-0}}}"
expected_nodes="${NNODES:-${MLP_WORKER_NUM:-${SLURM_NNODES:-1}}}"
variant="${BENCHMARK_VARIANT:-ddp_full}"
workload_profile="${BENCHMARK_WORKLOAD_PROFILE:-legacy_msa32}"
case "${workload_profile}" in
    legacy_msa32)
        profile_default_crop_size=768
        profile_default_dataset_rows=128
        profile_default_warmup_steps=10
        profile_default_measure_steps=20
        profile_msa_min_size=16
        profile_msa_sample_cutoff=32
        profile_msa_max_size=16384
        profile_msa_chunk_size=""
        profile_pairformer_cpu_offload_every_n=""
        profile_msa_cpu_offload_every_n=""
        profile_uses_msa_aware_dataset=false
        profile_diffusion_batch_size=24
        profile_max_nuc_tokens=-1
        profile_exact_token_count=true
        profile_train_sampler_seed=-1
        profile_pair_msa_as_unpaired_only=true
        profile_shuffle_pair_msa=true
        profile_checkpoint_kind=official
        profile_lr=0.00001
        profile_phase_memory=false
        ;;
    crop384_800_msa2048_randominit_nuc128_db48_paired_chunk2048_smoke|crop384_800_msa2048_randominit_nuc128_db48_paired_chunk2048_matched)
        profile_default_crop_size=384
        profile_default_dataset_rows=800
        if [[ "${workload_profile}" == *_smoke ]]; then
            profile_default_warmup_steps=0
            profile_default_measure_steps=1
            profile_phase_memory=true
        else
            profile_default_warmup_steps=10
            profile_default_measure_steps=20
            profile_phase_memory=false
        fi
        profile_msa_min_size=2048
        profile_msa_sample_cutoff=2048
        profile_msa_max_size=16384
        profile_msa_chunk_size=2048
        profile_pairformer_cpu_offload_every_n=2
        profile_msa_cpu_offload_every_n=1
        profile_uses_msa_aware_dataset=true
        profile_diffusion_batch_size=48
        profile_max_nuc_tokens=128
        profile_exact_token_count=false
        profile_train_sampler_seed=77
        profile_pair_msa_as_unpaired_only=false
        profile_shuffle_pair_msa=false
        profile_checkpoint_kind=random_init
        profile_lr=0.001
        ;;
    crop1024_800_msa2048_offload_smoke|crop1024_800_msa2048_offload_matched|crop1024_800_msa2048_offload|crop1024_800_msa2048_offload_nuc128_db32_31step_7d1a|crop1024_800_msa2048_offload_nuc256_db32_31step_7d1a|crop1024_800_msa2048_offload_nuc256_db32_chunk256_10step|crop1024_800_msa2048_offload_nuc256_db32_chunk2048_10step|crop1024_800_msa2048_msaoffload_no_pairformer_offload_nuc128_db32_31step_7d1a|crop960_800_msa2048_offload|crop960_800_msa2048_offload_nuc128_db32|crop1024_800_msa2048_offload_nuc128_db40_paired_chunk2048_smoke|crop1024_800_msa2048_offload_nuc128_db40_paired_chunk2048_matched)
        if [[ "${workload_profile}" == crop960_* ]]; then
            profile_default_crop_size=960
        else
            profile_default_crop_size=1024
        fi
        profile_default_dataset_rows=800
        case "${workload_profile}" in
            crop1024_800_msa2048_offload_smoke)
                profile_default_warmup_steps=0
                profile_default_measure_steps=1
                ;;
            crop1024_800_msa2048_offload_nuc128_db40_paired_chunk2048_smoke)
                profile_default_warmup_steps=0
                profile_default_measure_steps=1
                ;;
            crop1024_800_msa2048_offload_nuc128_db40_paired_chunk2048_matched)
                profile_default_warmup_steps=10
                profile_default_measure_steps=20
                ;;
            crop1024_800_msa2048_offload_matched)
                profile_default_warmup_steps=10
                profile_default_measure_steps=20
                ;;
            crop1024_800_msa2048_offload_nuc256_db32_chunk256_10step|crop1024_800_msa2048_offload_nuc256_db32_chunk2048_10step)
                profile_default_warmup_steps=2
                profile_default_measure_steps=8
                ;;
            crop1024_800_msa2048_offload_nuc128_db32_31step_7d1a|crop1024_800_msa2048_offload_nuc256_db32_31step_7d1a|crop1024_800_msa2048_msaoffload_no_pairformer_offload_nuc128_db32_31step_7d1a)
                profile_default_warmup_steps=10
                profile_default_measure_steps=21
                ;;
            crop1024_800_msa2048_offload|crop960_800_msa2048_offload|crop960_800_msa2048_offload_nuc128_db32)
                profile_default_warmup_steps=10
                profile_default_measure_steps=90
                ;;
        esac
        profile_msa_min_size=2048
        profile_msa_sample_cutoff=2048
        profile_msa_max_size=16384
        profile_msa_chunk_size=128
        if [ "${workload_profile}" = "crop1024_800_msa2048_offload_nuc256_db32_chunk256_10step" ]; then
            profile_msa_chunk_size=256
        elif [ "${workload_profile}" = "crop1024_800_msa2048_offload_nuc256_db32_chunk2048_10step" ]; then
            profile_msa_chunk_size=2048
        elif [[ "${workload_profile}" == crop1024_800_msa2048_offload_nuc128_db40_paired_chunk2048_* ]]; then
            profile_msa_chunk_size=2048
        fi
        profile_pairformer_cpu_offload_every_n=2
        profile_msa_cpu_offload_every_n=1
        if [ "${workload_profile}" = "crop1024_800_msa2048_msaoffload_no_pairformer_offload_nuc128_db32_31step_7d1a" ]; then
            profile_pairformer_cpu_offload_every_n=""
        fi
        profile_uses_msa_aware_dataset=true
        profile_diffusion_batch_size=24
        profile_max_nuc_tokens=-1
        profile_exact_token_count=true
        profile_train_sampler_seed=-1
        profile_pair_msa_as_unpaired_only=true
        profile_shuffle_pair_msa=true
        profile_checkpoint_kind=official
        profile_lr=0.00001
        profile_phase_memory=false
        if [ "${workload_profile}" = "crop960_800_msa2048_offload_nuc128_db32" ] || \
           [ "${workload_profile}" = "crop1024_800_msa2048_offload_nuc128_db32_31step_7d1a" ] || \
           [ "${workload_profile}" = "crop1024_800_msa2048_msaoffload_no_pairformer_offload_nuc128_db32_31step_7d1a" ]; then
            profile_diffusion_batch_size=32
            profile_max_nuc_tokens=128
            profile_exact_token_count=false
        fi
        if [ "${workload_profile}" = "crop1024_800_msa2048_offload_nuc256_db32_31step_7d1a" ] || \
           [ "${workload_profile}" = "crop1024_800_msa2048_offload_nuc256_db32_chunk256_10step" ] || \
           [ "${workload_profile}" = "crop1024_800_msa2048_offload_nuc256_db32_chunk2048_10step" ]; then
            profile_diffusion_batch_size=32
            profile_max_nuc_tokens=256
            profile_exact_token_count=false
        fi
        if [ "${workload_profile}" = "crop1024_800_msa2048_offload_nuc128_db32_31step_7d1a" ] || \
           [ "${workload_profile}" = "crop1024_800_msa2048_offload_nuc256_db32_31step_7d1a" ] || \
           [ "${workload_profile}" = "crop1024_800_msa2048_offload_nuc256_db32_chunk256_10step" ] || \
           [ "${workload_profile}" = "crop1024_800_msa2048_offload_nuc256_db32_chunk2048_10step" ] || \
           [ "${workload_profile}" = "crop1024_800_msa2048_msaoffload_no_pairformer_offload_nuc128_db32_31step_7d1a" ]; then
            profile_train_sampler_seed=77
        fi
        if [[ "${workload_profile}" == crop1024_800_msa2048_offload_nuc128_db40_paired_chunk2048_* ]]; then
            profile_diffusion_batch_size=40
            profile_max_nuc_tokens=128
            profile_exact_token_count=false
            profile_train_sampler_seed=77
            profile_pair_msa_as_unpaired_only=false
            profile_shuffle_pair_msa=false
            profile_checkpoint_kind=immune_heavy_999_ema
            profile_lr=0.000003
            if [[ "${workload_profile}" == *_smoke ]]; then
                profile_phase_memory=true
            fi
        fi
        ;;
    *)
        echo "Unsupported BENCHMARK_WORKLOAD_PROFILE=${workload_profile}" >&2
        exit 2
        ;;
esac
export BENCHMARK_WORKLOAD_PROFILE="${workload_profile}"
legacy_crop_size="${BENCHMARK_TRAIN_CROP_SIZE:-}"
benchmark_crop_size="${BENCHMARK_CROP_SIZE:-${legacy_crop_size:-${profile_default_crop_size}}}"
if [[ -n "${BENCHMARK_CROP_SIZE:-}" && -n "${legacy_crop_size}" && \
      "${BENCHMARK_CROP_SIZE}" != "${legacy_crop_size}" ]]; then
    echo "BENCHMARK_CROP_SIZE and BENCHMARK_TRAIN_CROP_SIZE disagree" >&2
    exit 2
fi
legacy_dataset_rows="${BENCHMARK_MAX_ROWS:-}"
benchmark_dataset_rows="${BENCHMARK_DATASET_ROWS:-${legacy_dataset_rows:-${profile_default_dataset_rows}}}"
if [[ -n "${BENCHMARK_DATASET_ROWS:-}" && -n "${legacy_dataset_rows}" && \
      "${BENCHMARK_DATASET_ROWS}" != "${legacy_dataset_rows}" ]]; then
    echo "BENCHMARK_DATASET_ROWS and BENCHMARK_MAX_ROWS disagree" >&2
    exit 2
fi
case "${benchmark_crop_size}:${benchmark_dataset_rows}" in
    384:800)
        if [ "${profile_uses_msa_aware_dataset}" != "true" ]; then
            echo "The crop384:800 profile requires the strict MSA2048 workload" >&2
            exit 2
        fi
        benchmark_dataset_name="benchmark_long_protein_context_384_800_msa2048"
        benchmark_epoch_size=800
        benchmark_sampler_replacement=false
        benchmark_random_sample_if_failed=false
        ;;
    768:128)
        benchmark_dataset_name="benchmark_long_protein_context_768"
        benchmark_epoch_size=10000
        benchmark_sampler_replacement=true
        benchmark_random_sample_if_failed=true
        ;;
    1024:128)
        benchmark_dataset_name="benchmark_long_protein_context_1024"
        benchmark_epoch_size=10000
        benchmark_sampler_replacement=true
        benchmark_random_sample_if_failed=true
        ;;
    1024:800)
        if [ "${profile_uses_msa_aware_dataset}" = "true" ]; then
            benchmark_dataset_name="benchmark_long_protein_context_1024_800_msa2048"
        else
            benchmark_dataset_name="benchmark_long_protein_context_1024_800"
        fi
        benchmark_epoch_size=800
        benchmark_sampler_replacement=false
        benchmark_random_sample_if_failed=false
        ;;
    960:800)
        if [ "${profile_uses_msa_aware_dataset}" != "true" ]; then
            echo "The crop960:800 profile requires the strict MSA2048 workload" >&2
            exit 2
        fi
        benchmark_dataset_name="benchmark_long_protein_context_960_800_msa2048"
        benchmark_epoch_size=800
        benchmark_sampler_replacement=false
        benchmark_random_sample_if_failed=false
        ;;
    *)
        echo "Supported exact-crop dataset profiles are 384:800, 768:128, 960:800, 1024:128, and 1024:800" >&2
        exit 2
        ;;
esac
resolved_warmup_steps="${BENCHMARK_WARMUP_STEPS:-${profile_default_warmup_steps}}"
resolved_measure_steps="${BENCHMARK_MEASURE_STEPS:-${profile_default_measure_steps}}"
resolved_msa_min_size="${BENCHMARK_MSA_MIN_SIZE_TRAIN:-${profile_msa_min_size}}"
resolved_msa_sample_cutoff="${BENCHMARK_MSA_SAMPLE_CUTOFF_TRAIN:-${profile_msa_sample_cutoff}}"
resolved_msa_max_size="${BENCHMARK_MSA_MAX_SIZE_TRAIN:-${profile_msa_max_size}}"
resolved_msa_chunk_size="${BENCHMARK_MSA_CHUNK_SIZE:-${profile_msa_chunk_size}}"
resolved_pairformer_cpu_offload_every_n="${BENCHMARK_PAIRFORMER_CHECKPOINT_CPU_OFFLOAD_EVERY_N:-${profile_pairformer_cpu_offload_every_n}}"
resolved_msa_cpu_offload_every_n="${BENCHMARK_MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N:-${profile_msa_cpu_offload_every_n}}"
resolved_diffusion_batch_size="${BENCHMARK_DIFFUSION_BATCH_SIZE:-${profile_diffusion_batch_size}}"
resolved_max_nuc_tokens="${BENCHMARK_MAX_NUC_TOKENS:-${profile_max_nuc_tokens}}"
resolved_train_sampler_seed="${BENCHMARK_TRAIN_SAMPLER_SEED:-${profile_train_sampler_seed}}"
resolved_pair_msa_as_unpaired_only="${BENCHMARK_PAIR_MSA_AS_UNPAIRED_ONLY:-${profile_pair_msa_as_unpaired_only}}"
resolved_shuffle_pair_msa="${BENCHMARK_SHUFFLE_PAIR_MSA:-${profile_shuffle_pair_msa}}"
resolved_lr="${BENCHMARK_LR:-${profile_lr}}"
resolved_phase_memory="${PROTENIX_BENCHMARK_PHASE_MEMORY:-${profile_phase_memory}}"
official_checkpoint="/mnt/project_abd/public_databases/protenix.20260101/checkpoint/protenix-v2.pt"
immune_heavy_checkpoint="/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/checkpoints/999_ema_0.999.pt"
case "${profile_checkpoint_kind}" in
    random_init) profile_checkpoint="" ;;
    official) profile_checkpoint="${official_checkpoint}" ;;
    immune_heavy_999_ema) profile_checkpoint="${immune_heavy_checkpoint}" ;;
    *) echo "Unsupported checkpoint profile ${profile_checkpoint_kind}" >&2; exit 2 ;;
esac
if [ "${profile_checkpoint_kind}" = "random_init" ]; then
    for checkpoint_var in \
        BENCHMARK_LOAD_CHECKPOINT_PATH BENCHMARK_LOAD_EMA_CHECKPOINT_PATH \
        LOAD_CHECKPOINT_PATH LOAD_EMA_CHECKPOINT_PATH \
        load_checkpoint_path load_ema_checkpoint_path; do
        if [ -n "${!checkpoint_var:-}" ]; then
            echo "${workload_profile} requires random initialization; inherited ${checkpoint_var} is forbidden" >&2
            exit 2
        fi
    done
fi
resolved_checkpoint="${BENCHMARK_LOAD_CHECKPOINT_PATH:-${profile_checkpoint}}"
resolved_ema_checkpoint="${BENCHMARK_LOAD_EMA_CHECKPOINT_PATH:-${resolved_checkpoint}}"
if [ "${workload_profile}" != "legacy_msa32" ] && \
   { [ "${benchmark_crop_size}" != "${profile_default_crop_size}" ] || \
     [ "${benchmark_dataset_rows}" != "800" ] || \
     [ "${resolved_warmup_steps}" != "${profile_default_warmup_steps}" ] || \
     [ "${resolved_measure_steps}" != "${profile_default_measure_steps}" ] || \
     [ "${resolved_msa_min_size}" != "2048" ] || \
     [ "${resolved_msa_sample_cutoff}" != "2048" ] || \
     [ "${resolved_msa_max_size}" != "16384" ] || \
     [ "${resolved_msa_chunk_size}" != "${profile_msa_chunk_size}" ] || \
     [ "${resolved_pairformer_cpu_offload_every_n}" != "${profile_pairformer_cpu_offload_every_n}" ] || \
     [ "${resolved_msa_cpu_offload_every_n}" != "${profile_msa_cpu_offload_every_n}" ] || \
     [ "${resolved_diffusion_batch_size}" != "${profile_diffusion_batch_size}" ] || \
     [ "${resolved_max_nuc_tokens}" != "${profile_max_nuc_tokens}" ] || \
     [ "${resolved_train_sampler_seed}" != "${profile_train_sampler_seed}" ] || \
     [ "${resolved_pair_msa_as_unpaired_only}" != "${profile_pair_msa_as_unpaired_only}" ] || \
     [ "${resolved_shuffle_pair_msa}" != "${profile_shuffle_pair_msa}" ] || \
     [ "${resolved_lr}" != "${profile_lr}" ] || \
     [ "${resolved_phase_memory}" != "${profile_phase_memory}" ] || \
     [ "${resolved_checkpoint}" != "${profile_checkpoint}" ] || \
     [ "${resolved_ema_checkpoint}" != "${profile_checkpoint}" ]; }; then
    echo "${workload_profile} settings differ from the locked profile" >&2
    exit 2
fi
export BENCHMARK_CROP_SIZE="${benchmark_crop_size}"
export BENCHMARK_TRAIN_CROP_SIZE="${benchmark_crop_size}"
export BENCHMARK_DATASET_ROWS="${benchmark_dataset_rows}"
export BENCHMARK_DATASET_NAME="${benchmark_dataset_name}"
if [ "${benchmark_dataset_name}" = "benchmark_long_protein_context_384_800_msa2048" ]; then
    profile_default_dataset_dir="${repo_dir}/data/benchmark_long_protein_context_1024_800_msa2048"
elif [[ "${workload_profile}" == crop1024_800_msa2048_offload_nuc128_db40_paired_chunk2048_* ]]; then
    profile_default_dataset_dir="/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_msa2048_offload/data/benchmark_long_protein_context_1024_800_msa2048"
elif [ "${benchmark_dataset_name}" = "benchmark_long_protein_context_960_800_msa2048" ]; then
    profile_default_dataset_dir="${repo_dir}/data/benchmark_long_protein_context_1024_800_msa2048"
else
    profile_default_dataset_dir="${repo_dir}/data/${benchmark_dataset_name}"
fi
export BENCHMARK_DATASET_DIR="${BENCHMARK_DATASET_DIR:-${profile_default_dataset_dir}}"
export BENCHMARK_WARMUP_STEPS="${resolved_warmup_steps}"
export BENCHMARK_MEASURE_STEPS="${resolved_measure_steps}"
export BENCHMARK_MSA_MIN_SIZE_TRAIN="${resolved_msa_min_size}"
export BENCHMARK_MSA_SAMPLE_CUTOFF_TRAIN="${resolved_msa_sample_cutoff}"
export BENCHMARK_MSA_MAX_SIZE_TRAIN="${resolved_msa_max_size}"
export BENCHMARK_MSA_CHUNK_SIZE="${resolved_msa_chunk_size}"
export BENCHMARK_PAIRFORMER_CHECKPOINT_CPU_OFFLOAD_EVERY_N="${resolved_pairformer_cpu_offload_every_n}"
export BENCHMARK_MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N="${resolved_msa_cpu_offload_every_n}"
export BENCHMARK_DIFFUSION_BATCH_SIZE="${resolved_diffusion_batch_size}"
export BENCHMARK_MAX_NUC_TOKENS="${resolved_max_nuc_tokens}"
export BENCHMARK_TRAIN_SAMPLER_SEED="${resolved_train_sampler_seed}"
export BENCHMARK_PAIR_MSA_AS_UNPAIRED_ONLY="${resolved_pair_msa_as_unpaired_only}"
export BENCHMARK_SHUFFLE_PAIR_MSA="${resolved_shuffle_pair_msa}"
export BENCHMARK_LR="${resolved_lr}"
export PROTENIX_BENCHMARK_PHASE_MEMORY="${resolved_phase_memory}"
export BENCHMARK_LOAD_CHECKPOINT_PATH="${resolved_checkpoint}"
export BENCHMARK_LOAD_EMA_CHECKPOINT_PATH="${resolved_ema_checkpoint}"
if [[ ! "${node_rank}" =~ ^[0-9]+$ ]] || \
   [[ ! "${expected_nodes}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NODE_RANK and node count must be non-negative/positive integers" >&2
    exit 2
fi
if (( node_rank >= expected_nodes )); then
    echo "NODE_RANK=${node_rank} must be smaller than node count ${expected_nodes}" >&2
    exit 2
fi
export NNODES="${expected_nodes}"
export NODE_RANK="${node_rank}"
hostname_value="$(hostname)"
hostname_tag="${hostname_value//[^A-Za-z0-9._-]/_}"
output_root="${BENCHMARK_OUTPUT_ROOT:-${repo_dir}/test_outputs/aistudio_benchmark}"
profile_root="${output_root}/${BENCHMARK_ID}"
nodes_root="${profile_root}/nodes"
node_dir="${nodes_root}/node_${node_rank}_${hostname_tag}"
poll_seconds="${BENCHMARK_POLL_SECONDS:-0.5}"
dry_run="${DRY_RUN:-0}"
if [[ ! "${poll_seconds}" =~ ^(0\.[0-9]*[1-9][0-9]*|[1-9][0-9]*(\.[0-9]+)?)$ ]]; then
    echo "BENCHMARK_POLL_SECONDS must be a positive number" >&2
    exit 2
fi

mkdir -p "${nodes_root}"
if ! mkdir "${node_dir}" 2>/dev/null; then
    echo "Benchmark node output already exists: ${node_dir}" >&2
    echo "Use a new BENCHMARK_ID; existing results are never overwritten." >&2
    exit 2
fi

export BENCHMARK_PROFILE_ROOT="${profile_root}"
export BASE_DIR="${BENCHMARK_TRAIN_BASE_DIR:-${profile_root}/train}"
# Triton's default cache is under a shared home directory in AIStudio. Eight
# ranks compiling the same kernels there can hit filesystem EBUSY errors. Keep
# every benchmark job/rank on a node-local cache; runner/train.py appends rank_N.
export PROTENIX_TRITON_CACHE_ROOT="${BENCHMARK_TRITON_CACHE_ROOT:-/tmp/protenix_triton_cache/${BENCHMARK_ID}/node_${node_rank}}"
# FusedLayerNorm is a torch C++/CUDA extension. Keep its build products off the
# shared source tree and let all local ranks reuse one node-local, ABI-specific
# cache after finetune_demo.sh has precompiled it in a single process.
export TORCH_EXTENSIONS_DIR="${BENCHMARK_TORCH_EXTENSIONS_DIR:-/tmp/protenix_torch_extensions/${BENCHMARK_ID}/node_${node_rank}/${PROTENIX_LAYER_NORM_CACHE_FINGERPRINT}}"
# Formal matched results cannot override the audited CUDA host toolchain.
if [ -n "${BENCHMARK_CC:-}" ] \
    && [ "$(readlink -m "${BENCHMARK_CC}")" != "${PROTENIX_CUDA_HOST_CC}" ]; then
    echo "BENCHMARK_CC must be ${PROTENIX_CUDA_HOST_CC}" >&2
    exit 2
fi
if [ -n "${BENCHMARK_CXX:-}" ] \
    && [ "$(readlink -m "${BENCHMARK_CXX}")" != "${PROTENIX_CUDA_HOST_CXX}" ]; then
    echo "BENCHMARK_CXX must be ${PROTENIX_CUDA_HOST_CXX}" >&2
    exit 2
fi
export CC="${PROTENIX_CUDA_HOST_CC}"
export CXX="${PROTENIX_CUDA_HOST_CXX}"
if [ ! -x "${CC}" ] || [ ! -x "${CXX}" ]; then
    echo "Missing benchmark CUDA host compiler: CC=${CC}, CXX=${CXX}" >&2
    exit 2
fi
# Triton also compiles a small C launcher through Python tempfile. AIStudio's
# inherited TMPDIR may point at shared storage with unreliable directory
# removal semantics, so all job-local temporary files stay on node-local /tmp.
# Keep this path short: torch_shm_manager appends a random directory and Unix
# socket name, and Linux AF_UNIX paths are limited to roughly 107 bytes.  A
# descriptive BENCHMARK_ID can otherwise make every DataLoader queue fail with
# `torch_shm_manager ... Invalid argument` before the first training step.
runtime_tmp_tag="$(printf '%s' "${BENCHMARK_ID}" | sha256sum | cut -c1-12)"
runtime_tmpdir="${BENCHMARK_RUNTIME_TMPDIR:-/tmp/ptx_${runtime_tmp_tag}_${node_rank}}"
mkdir -p "${runtime_tmpdir}"
runtime_tmpdir="$(readlink -f "${runtime_tmpdir}")"
if (( ${#runtime_tmpdir} > 60 )); then
    echo "BENCHMARK_RUNTIME_TMPDIR is too long for PyTorch multiprocessing: ${runtime_tmpdir}" >&2
    echo "Use a node-local path of at most 60 characters." >&2
    exit 2
fi
mkdir -p "${TORCH_EXTENSIONS_DIR}"
export TMPDIR="${runtime_tmpdir}"
export TEMP="${runtime_tmpdir}"
export TMP="${runtime_tmpdir}"

launcher_log="${node_dir}/launcher.log"
nvml_csv="${node_dir}/nvml.csv"
compute_apps_csv="${node_dir}/compute_apps.csv"
preflight_compute_apps_csv="${node_dir}/preflight_compute_apps.csv"
system_csv="${node_dir}/system.csv"
metadata_path="${node_dir}/metadata.env"
topology_path="${node_dir}/topology.txt"
status_path="${node_dir}/node_status.json"
artifact_hashes_path="${node_dir}/artifact_hashes.sha256"
code_tree_hashes_path="${node_dir}/code_tree.sha256"

start_epoch="$(date +%s)"
start_iso="$(date --iso-8601=seconds)"

{
    printf 'BENCHMARK_ID=%q\n' "${BENCHMARK_ID}"
    printf 'BENCHMARK_VARIANT=%q\n' "${variant}"
    printf 'BENCHMARK_WORKLOAD_PROFILE=%q\n' "${workload_profile}"
    printf 'NODE_RANK=%q\n' "${node_rank}"
    printf 'EXPECTED_NODES=%q\n' "${expected_nodes}"
    printf 'HOSTNAME=%q\n' "${hostname_value}"
    printf 'MASTER_ADDR=%q\n' "${MASTER_ADDR:-}"
    printf 'MASTER_PORT=%q\n' "${MASTER_PORT:-}"
    printf 'NPROC_PER_NODE=%q\n' "${NPROC_PER_NODE:-gpu}"
    printf 'CONDA_ENV=%q\n' "${CONDA_ENV:-protenix}"
    printf 'BENCHMARK_PYTHON=%q\n' "${benchmark_python}"
    printf 'CUDA_VISIBLE_DEVICES=%q\n' "${CUDA_VISIBLE_DEVICES:-}"
    printf 'BENCHMARK_WARMUP_STEPS=%q\n' "${resolved_warmup_steps}"
    printf 'BENCHMARK_MEASURE_STEPS=%q\n' "${resolved_measure_steps}"
    printf 'BENCHMARK_SKIP_FINAL_CHECKPOINT=%q\n' "${BENCHMARK_SKIP_FINAL_CHECKPOINT:-true}"
    printf 'BENCHMARK_FSDP_MESH=%q\n' "${BENCHMARK_FSDP_MESH:-auto}"
    printf 'BENCHMARK_EMA_SHADOW_DEVICE=%q\n' "${BENCHMARK_EMA_SHADOW_DEVICE:-cpu}"
    printf 'BENCHMARK_EXPECTED_TORCH_CUDA=%q\n' "${BENCHMARK_EXPECTED_TORCH_CUDA:-12.6}"
    printf 'BENCHMARK_EXACT_CROP=%q\n' "${BENCHMARK_EXACT_CROP:-true}"
    printf 'BENCHMARK_MODEL_C_Z=%q\n' "${BENCHMARK_MODEL_C_Z:-256}"
    printf 'BENCHMARK_MODEL_C_M=%q\n' "${BENCHMARK_MODEL_C_M:-128}"
    printf 'BENCHMARK_CROP_SIZE=%q\n' "${benchmark_crop_size}"
    printf 'BENCHMARK_TRAIN_CROP_SIZE=%q\n' "${benchmark_crop_size}"
    printf 'BENCHMARK_DATASET_NAME=%q\n' "${benchmark_dataset_name}"
    printf 'BENCHMARK_DATASET_ROWS=%q\n' "${benchmark_dataset_rows}"
    printf 'BENCHMARK_DATASET_DIR=%q\n' "${BENCHMARK_DATASET_DIR}"
    printf 'BENCHMARK_DIFFUSION_BATCH_SIZE=%q\n' "${resolved_diffusion_batch_size}"
    printf 'BENCHMARK_MAX_NUC_TOKENS=%q\n' "${resolved_max_nuc_tokens}"
    printf 'BENCHMARK_EXACT_TOKEN_COUNT=%q\n' "${profile_exact_token_count}"
    printf 'BENCHMARK_DIFFUSION_CHUNK_SIZE=%q\n' "${BENCHMARK_DIFFUSION_CHUNK_SIZE:-4}"
    printf 'BENCHMARK_NUM_DL_WORKERS=%q\n' "${BENCHMARK_NUM_DL_WORKERS:-4}"
    printf 'BENCHMARK_EPOCH_SIZE=%q\n' "${BENCHMARK_EPOCH_SIZE:-${benchmark_epoch_size}}"
    printf 'BENCHMARK_TRAIN_SAMPLER_REPLACEMENT=%q\n' "${BENCHMARK_TRAIN_SAMPLER_REPLACEMENT:-${benchmark_sampler_replacement}}"
    printf 'BENCHMARK_TRAIN_SAMPLER_SEED=%q\n' "${resolved_train_sampler_seed}"
    printf 'BENCHMARK_PAIR_MSA_AS_UNPAIRED_ONLY=%q\n' "${resolved_pair_msa_as_unpaired_only}"
    printf 'BENCHMARK_SHUFFLE_PAIR_MSA=%q\n' "${resolved_shuffle_pair_msa}"
    printf 'BENCHMARK_LR=%q\n' "${resolved_lr}"
    printf 'PROTENIX_BENCHMARK_PHASE_MEMORY=%q\n' "${resolved_phase_memory}"
    printf 'BENCHMARK_LOAD_CHECKPOINT_PATH=%q\n' "${resolved_checkpoint}"
    printf 'BENCHMARK_LOAD_EMA_CHECKPOINT_PATH=%q\n' "${resolved_ema_checkpoint}"
    printf 'BENCHMARK_RANDOM_SAMPLE_IF_FAILED=%q\n' "${BENCHMARK_RANDOM_SAMPLE_IF_FAILED:-${benchmark_random_sample_if_failed}}"
    printf 'BENCHMARK_PAIRFORMER_BLOCKS_PER_CKPT=%q\n' "${BENCHMARK_PAIRFORMER_BLOCKS_PER_CKPT:-2}"
    printf 'BENCHMARK_PAIR_TRANSITION_CHUNK_SIZE=%q\n' "${BENCHMARK_PAIR_TRANSITION_CHUNK_SIZE:-128}"
    printf 'BENCHMARK_PAIR_TRANSITION_CHECKPOINT_CHUNKS=%q\n' "${BENCHMARK_PAIR_TRANSITION_CHECKPOINT_CHUNKS:-true}"
    printf 'BENCHMARK_MSA_MIN_SIZE_TRAIN=%q\n' "${resolved_msa_min_size}"
    printf 'BENCHMARK_MSA_SAMPLE_CUTOFF_TRAIN=%q\n' "${resolved_msa_sample_cutoff}"
    printf 'BENCHMARK_MSA_MAX_SIZE_TRAIN=%q\n' "${resolved_msa_max_size}"
    printf 'BENCHMARK_MSA_CHUNK_SIZE=%q\n' "${resolved_msa_chunk_size}"
    printf 'BENCHMARK_PAIRFORMER_CHECKPOINT_CPU_OFFLOAD_EVERY_N=%q\n' "${resolved_pairformer_cpu_offload_every_n}"
    printf 'BENCHMARK_MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N=%q\n' "${resolved_msa_cpu_offload_every_n}"
    printf 'PROTENIX_TRITON_CACHE_ROOT=%q\n' "${PROTENIX_TRITON_CACHE_ROOT}"
    printf 'TORCH_EXTENSIONS_DIR=%q\n' "${TORCH_EXTENSIONS_DIR}"
    printf 'CUDA_HOME=%q\n' "${CUDA_HOME}"
    printf 'CUDACXX=%q\n' "${CUDACXX}"
    printf 'PROTENIX_CUDA_TOOLCHAIN_FINGERPRINT=%q\n' "${PROTENIX_CUDA_TOOLCHAIN_FINGERPRINT}"
    printf 'PROTENIX_LAYER_NORM_CACHE_FINGERPRINT=%q\n' "${PROTENIX_LAYER_NORM_CACHE_FINGERPRINT}"
    printf 'CC=%q\n' "${CC}"
    printf 'CXX=%q\n' "${CXX}"
    printf 'TMPDIR=%q\n' "${TMPDIR}"
    printf 'DRY_RUN=%q\n' "${dry_run}"
    printf 'STARTED_AT=%q\n' "${start_iso}"
} > "${metadata_path}"

{
    uname -a
    "${CC}" --version 2>&1 || true
    "${CXX}" --version 2>&1 || true
    nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total \
        --format=csv,noheader,nounits 2>&1 || true
    git rev-parse HEAD 2>&1 || true
    git status --short 2>&1 || true
    sha256sum benchmark.sh finetune_submit.sh finetune_demo.sh \
        configs/configs_base.py configs/configs_data.py runner/train.py \
        runner/ema.py protenix/model/protenix.py \
        protenix/model/modules/primitives.py \
        protenix/model/modules/pairformer.py \
        protenix/model/modules/diffusion.py \
        protenix/model/layer_norm/cache_provenance.py \
        protenix/model/layer_norm/layer_norm.py \
        protenix/model/layer_norm/torch_ext_compile.py \
        protenix/model/layer_norm/kernel/layer_norm_cuda.cpp \
        protenix/model/layer_norm/kernel/layer_norm_cuda_kernel.cu \
        scripts/runtime/cuda126_env.sh \
        protenix/utils/fsdp.py \
        scripts/benchmark/build_long_protein_benchmark.py \
        scripts/benchmark/build_msa_aware_long_protein_benchmark.py \
        scripts/benchmark/audit_msa_depth.py \
        scripts/benchmark/compare_aistudio_benchmarks.py \
        scripts/benchmark/compare_implementation_benchmarks.py \
        scripts/benchmark/summarize_aistudio_benchmark.py 2>&1 || true
    metadata_checkpoint="${BENCHMARK_LOAD_CHECKPOINT_PATH-${LOAD_CHECKPOINT_PATH-}}"
    if [ -n "${metadata_checkpoint}" ]; then
        stat "${metadata_checkpoint}" 2>&1 || true
    else
        printf '%s\n' "initialization=random (no checkpoint)"
    fi
} > "${topology_path}"

# Hash every source/config file that can participate in training, including
# untracked files in this deliberately dirty research checkout. The summary
# requires this mapping to match across nodes and the paired comparator requires
# it to match between DDP and FSDP2 runs.
{
    find configs protenix runner scripts/benchmark scripts/runtime -type f \
        \( -name '*.py' -o -name '*.sh' -o -name '*.cu' -o -name '*.cuh' \
           -o -name '*.cpp' -o -name '*.cc' -o -name '*.c' -o -name '*.h' \
           -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' \) \
        ! -path '*/__pycache__/*' -print0
    printf '%s\0' benchmark.sh finetune_submit.sh finetune_demo.sh
} | LC_ALL=C sort -z | xargs -0 sha256sum > "${code_tree_hashes_path}"

printf 'epoch_seconds,node_rank,hostname,gpu_index,gpu_uuid,gpu_name,driver_version,memory_used_mib,memory_total_mib,utilization_gpu_pct\n' > "${nvml_csv}"
printf 'epoch_seconds,node_rank,hostname,pid,gpu_uuid,used_memory_mib,process_name,is_benchmark_process,process_role\n' > "${compute_apps_csv}"
printf 'node_rank,hostname,pid,gpu_uuid,used_memory_mib,process_name\n' > "${preflight_compute_apps_csv}"
nvidia-smi \
    --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader,nounits 2>/dev/null | \
while IFS= read -r row; do
    [ -n "${row}" ] || continue
    printf '%s,%s,%s\n' "${node_rank}" "${hostname_tag}" "${row}"
done >> "${preflight_compute_apps_csv}"
if [ "$(wc -l < "${preflight_compute_apps_csv}")" -ne 1 ]; then
    echo "Benchmark requires GPUs with no pre-existing compute processes:" >&2
    tail -n +2 "${preflight_compute_apps_csv}" >&2
    exit 1
fi
printf 'epoch_seconds,node_rank,hostname,mem_available_kib,load_1m\n' > "${system_csv}"

monitor_pid=""
benchmark_root_pid="$$"
is_descendant_process() {
    local child_pid="$1"
    local ancestor_pid="$2"
    local parent_pid
    while [[ "${child_pid}" =~ ^[0-9]+$ ]] && (( child_pid > 1 )); do
        if [ "${child_pid}" = "${ancestor_pid}" ]; then
            return 0
        fi
        parent_pid="$(awk '/^PPid:/ {print $2}' "/proc/${child_pid}/status" 2>/dev/null || true)"
        if [[ ! "${parent_pid}" =~ ^[0-9]+$ ]] || [ "${parent_pid}" = "${child_pid}" ]; then
            break
        fi
        child_pid="${parent_pid}"
    done
    return 1
}
monitor_node() {
    while true; do
        sample_epoch="$(date +%s.%N)"
        nvidia-smi \
            --query-gpu=index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null | \
        while IFS= read -r row; do
            printf '%s,%s,%s,%s\n' \
                "${sample_epoch}" "${node_rank}" "${hostname_tag}" "${row}"
        done >> "${nvml_csv}"

        nvidia-smi \
            --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
            --format=csv,noheader,nounits 2>/dev/null | \
        while IFS= read -r row; do
            [ -n "${row}" ] || continue
            IFS=',' read -r process_pid process_gpu process_memory process_name <<< "${row}"
            process_pid="${process_pid//[[:space:]]/}"
            is_benchmark=0
            process_role=external
            if is_descendant_process "${process_pid}" "${benchmark_root_pid}"; then
                is_benchmark=1
                process_role=benchmark_descendant
            elif [ ! -r "/proc/${process_pid}/status" ] && \
                 [[ "${process_name}" == *"[Not Found]"* ]]; then
                # AIStudio's nvidia-smi may expose host-namespace PIDs while
                # /proc exposes container PIDs. Keep this as unresolved
                # evidence. The summary selects the earliest context that
                # survives into the measurement window as the rank, but never
                # silently accepts a later unresolved context as an auxiliary
                # process: an external job can start after the empty-GPU
                # preflight.
                is_benchmark=1
                process_role=namespace_unresolved
            fi
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "${sample_epoch}" "${node_rank}" "${hostname_tag}" \
                "${process_pid}" "${process_gpu}" "${process_memory}" \
                "${process_name}" "${is_benchmark}" "${process_role}"
        done >> "${compute_apps_csv}"

        mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
        load_1m="$(awk '{print $1}' /proc/loadavg)"
        printf '%s,%s,%s,%s,%s\n' \
            "${sample_epoch}" "${node_rank}" "${hostname_tag}" \
            "${mem_available:-0}" "${load_1m:-0}" >> "${system_csv}"
        sleep "${poll_seconds}"
    done
}

stop_monitor() {
    if [ -n "${monitor_pid}" ]; then
        kill "${monitor_pid}" 2>/dev/null || true
        wait "${monitor_pid}" 2>/dev/null || true
        monitor_pid=""
    fi
}

handle_signal() {
    stop_monitor
    exit 130
}

trap stop_monitor EXIT
trap handle_signal INT TERM
monitor_node &
monitor_pid=$!

set +e
bash "${repo_dir}/benchmark.sh" 2>&1 | tee "${launcher_log}"
pipeline_status=("${PIPESTATUS[@]}")
training_status="${pipeline_status[0]}"
tee_status="${pipeline_status[1]}"
set -e
benchmark_status="${training_status}"
if [ "${benchmark_status}" -eq 0 ] && [ "${tee_status}" -ne 0 ]; then
    benchmark_status="${tee_status}"
fi

if [ "${dry_run}" = "1" ]; then
    : > "${artifact_hashes_path}"
else
    {
        benchmark_dataset_dir="${BENCHMARK_DATASET_DIR:-${PROTENIX_BENCHMARK_DIR:-${repo_dir}/data/${benchmark_dataset_name}}}"
        metadata_checkpoint="${BENCHMARK_LOAD_CHECKPOINT_PATH-${LOAD_CHECKPOINT_PATH-}}"
        metadata_source_index="${BENCHMARK_SOURCE_INDEX:-/mnt/project_abd/public_databases/protenix.20260101/indices/indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz}"
        artifact_paths=(
            "${metadata_source_index}" \
            "${benchmark_dataset_dir}/indices.csv" \
            "${benchmark_dataset_dir}/summary.json"
        )
        if [ -n "${metadata_checkpoint}" ]; then
            artifact_paths=("${metadata_checkpoint}" "${artifact_paths[@]}")
        fi
        if [ "${profile_uses_msa_aware_dataset}" = "true" ]; then
            artifact_paths+=(
                "/mnt/project_abd/public_databases/protenix.20260101/common/seq_to_pdb_index.json"
                "${benchmark_dataset_dir}/msa_depth.json"
                "${benchmark_dataset_dir}/msa_depth.csv"
                "${benchmark_dataset_dir}/msa_depth_candidates.csv"
            )
        fi
        sha256sum "${artifact_paths[@]}" 2>&1 || true
    } > "${artifact_hashes_path}"
fi

stop_monitor
end_epoch="$(date +%s)"
end_iso="$(date --iso-8601=seconds)"
wall_seconds=$((end_epoch - start_epoch))

cat_status="completed"
if [ "${dry_run}" = "1" ] && [ "${benchmark_status}" -eq 0 ]; then
    cat_status="dry_run"
elif [ "${benchmark_status}" -ne 0 ]; then
    cat_status="failed"
fi
printf '{\n  "benchmark_id": "%s",\n  "benchmark_variant": "%s",\n  "workload_profile": "%s",\n  "node_rank": %s,\n  "hostname": "%s",\n  "expected_nodes": %s,\n  "poll_seconds": %s,\n  "status": "%s",\n  "exit_code": %s,\n  "training_exit_code": %s,\n  "tee_exit_code": %s,\n  "started_at": "%s",\n  "completed_at": "%s",\n  "wall_seconds": %s\n}\n' \
    "${BENCHMARK_ID}" "${variant}" "${workload_profile}" "${node_rank}" "${hostname_tag}" \
    "${expected_nodes}" "${poll_seconds}" "${cat_status}" \
    "${benchmark_status}" "${training_status}" "${tee_status}" \
    "${start_iso}" "${end_iso}" "${wall_seconds}" > "${status_path}"

final_status="${benchmark_status}"
if [ "${dry_run}" != "1" ] && [ "${expected_nodes}" = "1" ] && \
   [ "${benchmark_status}" -eq 0 ] && \
   [ -f "${repo_dir}/scripts/benchmark/summarize_aistudio_benchmark.py" ]; then
    set +e
    "${benchmark_python}" "${repo_dir}/scripts/benchmark/summarize_aistudio_benchmark.py" \
        --profile-root "${profile_root}"
    summary_status=$?
    set -e
    if [ "${summary_status}" -ne 0 ]; then
        echo "Training completed, but benchmark result validation failed." >&2
        final_status="${summary_status}"
    fi
fi

exit "${final_status}"
