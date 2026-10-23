#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"

: "${BENCHMARK_ID:?Set BENCHMARK_ID to a unique value shared by every node}"
if [[ ! "${BENCHMARK_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "BENCHMARK_ID may contain only letters, digits, dot, underscore, and dash" >&2
    exit 2
fi
if (( $# != 0 )); then
    echo "benchmark.sh accepts no positional CLI overrides; use the documented BENCHMARK_* variables" >&2
    exit 2
fi
export DRY_RUN="${DRY_RUN:-0}"
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1" >&2
    exit 2
fi

variant="${BENCHMARK_VARIANT:-ddp_full}"
case "${variant}" in
    ddp_full)
        strategy=ddp
        ;;
    fsdp2_full)
        strategy=fsdp2
        ;;
    *)
        echo "Unsupported BENCHMARK_VARIANT=${variant}; use ddp_full or fsdp2_full" >&2
        exit 2
        ;;
esac
export BENCHMARK_VARIANT="${variant}"

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
            # The sampled MSA depth is exactly 2048, so this is one full MSA
            # row block and therefore disables row-wise chunk splitting.
            profile_msa_chunk_size=2048
        elif [[ "${workload_profile}" == crop1024_800_msa2048_offload_nuc128_db40_paired_chunk2048_* ]]; then
            profile_msa_chunk_size=2048
        fi
        profile_pairformer_cpu_offload_every_n=2
        profile_msa_cpu_offload_every_n=1
        if [ "${workload_profile}" = "crop1024_800_msa2048_msaoffload_no_pairformer_offload_nuc128_db32_31step_7d1a" ]; then
            # Matched ablation: keep Pairformer activation checkpointing, but
            # retain its checkpoint inputs on GPU instead of using save_on_cpu.
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
            # With the audited 800-row weighted/no-replacement index, sampler
            # seed 77 places 7d1a on rank5 at zero-based step30 (human step31).
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

warmup_steps_count="${BENCHMARK_WARMUP_STEPS:-${profile_default_warmup_steps}}"
measure_steps_count="${BENCHMARK_MEASURE_STEPS:-${profile_default_measure_steps}}"
if [[ ! "${warmup_steps_count}" =~ ^[0-9]+$ ]] || \
   [[ ! "${measure_steps_count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Benchmark warmup must be >=0 and measurement steps must be >0" >&2
    exit 2
fi
benchmark_total_steps=$((warmup_steps_count + measure_steps_count))

profile_root="${BENCHMARK_PROFILE_ROOT:-${BENCHMARK_OUTPUT_ROOT:-${repo_dir}/test_outputs/aistudio_benchmark}/${BENCHMARK_ID}}"
export BENCHMARK_PROFILE_ROOT="${profile_root}"
export RUN_NAME="${BENCHMARK_RUN_NAME:-${BENCHMARK_ID}}"
export BASE_DIR="${BENCHMARK_TRAIN_BASE_DIR:-${profile_root}/train}"
export run_name="${RUN_NAME}"
export base_dir="${BASE_DIR}"

# finetune_demo.sh supports legacy lower-case launcher inputs. Resolve and
# overwrite them here so a reused AIStudio login shell cannot silently alter
# topology, output paths, or turn a benchmark into a path-check-only command.
export NPROC_PER_NODE="${BENCHMARK_NPROC_PER_NODE:-${NPROC_PER_NODE:-gpu}}"
export NNODES="${NNODES:-${MLP_WORKER_NUM:-${SLURM_NNODES:-1}}}"
export NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-${SLURM_NODEID:-0}}}"
export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"
export nproc_per_node="${NPROC_PER_NODE}"
export nnodes="${NNODES}"
export node_rank="${NODE_RANK}"
export master_addr="${MASTER_ADDR}"
export master_port="${MASTER_PORT}"
export CHECK_ONLY=0
export SKIP_PATH_CHECK=0
unset PROTENIX_LAYER_NORM_BUILD_DIR
unset PROTENIX_CUDA_ARCH_LIST

# Both variants train the complete model. The resolved values below are
# deliberately written to both legacy lower-case and upper-case inputs so an
# inherited shell variable cannot make the two runs silently differ.
official_root="/mnt/project_abd/public_databases/protenix.20260101"
official_checkpoint="${official_root}/checkpoint/protenix-v2.pt"
official_checkpoint_sha256="8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599"
immune_heavy_checkpoint="/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/checkpoints/999_ema_0.999.pt"
immune_heavy_checkpoint_sha256="9e0df397d9deb18ea46921c678a62afc4a587241aacd01acd27887729e3a0c3e"
official_source_index="${official_root}/indices/indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz"
official_source_index_sha256="e9fc812d2250c8d19a995dbd38991ac05204d2ace5e1e4878d4760e708ad2bc1"
official_benchmark_index_128_sha256="6bdf93c0e8f5a6a527c50112516a2f3c1cd27d4c0bfc825b5a7465fdadf5fd6a"
official_benchmark_index_800_sha256="f49cf926cb21dd271151700042b23ce787a8b4ae8783f9a74a4141a457491991"
official_benchmark_index_msa2048_sha256="dce9db1758566a67f626959ba3e1cb5b56f8f693d91f2c37a44cec99ca2c54f1"
official_msa_mapping_sha256="cb5fb9aba9372dab1bbb0ed4322c3420fa3e606b597b4857c6bb577f94687ea9"
case "${profile_checkpoint_kind}" in
    random_init)
        profile_checkpoint=""
        profile_checkpoint_sha256=""
        ;;
    official)
        profile_checkpoint="${official_checkpoint}"
        profile_checkpoint_sha256="${official_checkpoint_sha256}"
        ;;
    immune_heavy_999_ema)
        profile_checkpoint="${immune_heavy_checkpoint}"
        profile_checkpoint_sha256="${immune_heavy_checkpoint_sha256}"
        ;;
    *)
        echo "Unsupported checkpoint profile ${profile_checkpoint_kind}" >&2
        exit 2
        ;;
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
if [[ "${resolved_checkpoint}" != "${profile_checkpoint}" || \
      "${resolved_ema_checkpoint}" != "${profile_checkpoint}" ]]; then
    if [ "${profile_checkpoint_kind}" = "random_init" ]; then
        echo "${workload_profile} requires random initialization; inherited raw/EMA checkpoints are forbidden" >&2
    else
        echo "${workload_profile} requires ${profile_checkpoint} for raw and EMA initialization" >&2
    fi
    exit 2
fi
export LOAD_CHECKPOINT_PATH="${resolved_checkpoint}"
export LOAD_EMA_CHECKPOINT_PATH="${resolved_ema_checkpoint}"
export load_checkpoint_path="${resolved_checkpoint}"
export load_ema_checkpoint_path="${resolved_ema_checkpoint}"
export LOAD_PARAMS_ONLY=true
export load_params_only=true
export BENCHMARK_LOAD_CHECKPOINT_PATH="${resolved_checkpoint}"
export BENCHMARK_LOAD_EMA_CHECKPOINT_PATH="${resolved_ema_checkpoint}"
export EXPECTED_CHECKPOINT_SHA256="${profile_checkpoint_sha256}"

export DISTRIBUTED_STRATEGY="${strategy}"
export distributed_strategy="${strategy}"
export OPTIMIZER_SHARDING=none
export optimizer_sharding=none
resolved_ema_device="${BENCHMARK_EMA_SHADOW_DEVICE:-cpu}"
if [[ "${resolved_ema_device}" != "cpu" ]]; then
    echo "Matched benchmark requires BENCHMARK_EMA_SHADOW_DEVICE=cpu" >&2
    exit 2
fi
export EMA_SHADOW_DEVICE="${resolved_ema_device}"
export ema_shadow_device="${resolved_ema_device}"
export FSDP_MESH="${BENCHMARK_FSDP_MESH:-auto}"
export fsdp_mesh="${FSDP_MESH}"
export FSDP_PARAM_DTYPE="${BENCHMARK_FSDP_PARAM_DTYPE:-fp32}"
export fsdp_param_dtype="${FSDP_PARAM_DTYPE}"
export FSDP_REDUCE_DTYPE="${BENCHMARK_FSDP_REDUCE_DTYPE:-fp32}"
export fsdp_reduce_dtype="${FSDP_REDUCE_DTYPE}"
export FSDP_RESHARD_AFTER_FORWARD="${BENCHMARK_FSDP_RESHARD_AFTER_FORWARD:-true}"
export fsdp_reshard_after_forward="${FSDP_RESHARD_AFTER_FORWARD}"
export EXPECTED_TORCH_CUDA="${BENCHMARK_EXPECTED_TORCH_CUDA:-12.6}"
case "${FSDP_MESH}" in
    auto|global|hybrid) ;;
    *)
        echo "BENCHMARK_FSDP_MESH must be auto, global, or hybrid" >&2
        exit 2
        ;;
esac
if [[ "${FSDP_PARAM_DTYPE}" != "fp32" || \
      "${FSDP_REDUCE_DTYPE}" != "fp32" || \
      "${FSDP_RESHARD_AFTER_FORWARD}" != "true" || \
      "${EXPECTED_TORCH_CUDA}" != "12.6" ]]; then
    echo "Matched benchmark requires FP32 FSDP params/reductions, reshard_after_forward=true, and PyTorch CUDA 12.6" >&2
    exit 2
fi

# Fixed-shape benchmark settings ported selectively from the fast-training
# repository. They apply identically to ddp_full and fsdp2_full; only the
# distributed strategy differs between the two variants.
export BENCHMARK_EXACT_CROP="${BENCHMARK_EXACT_CROP:-true}"
if [[ "${BENCHMARK_EXACT_CROP}" != "true" ]]; then
    echo "Matched benchmark requires BENCHMARK_EXACT_CROP=true" >&2
    exit 2
fi
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
case "${benchmark_crop_size}" in
    384|768|960|1024) ;;
    *)
        echo "BENCHMARK_CROP_SIZE must be 384, 768, 960, or 1024" >&2
        exit 2
        ;;
esac
case "${benchmark_crop_size}:${benchmark_dataset_rows}" in
    384:800)
        if [ "${profile_uses_msa_aware_dataset}" != "true" ]; then
            echo "The crop384:800 profile requires the strict MSA2048 workload" >&2
            exit 2
        fi
        benchmark_dataset_name="benchmark_long_protein_context_384_800_msa2048"
        # Reuse the hash-audited crop1024 artifact. Its 1025-token source
        # threshold is stricter than exact crop384 requires.
        benchmark_min_num_tokens=1025
        benchmark_max_num_tokens=1536
        benchmark_epoch_size=800
        benchmark_sampler_replacement=false
        benchmark_profile_random_sample_if_failed=false
        official_benchmark_index_sha256="${official_benchmark_index_msa2048_sha256}"
        ;;
    768:128)
        benchmark_dataset_name="benchmark_long_protein_context_768"
        benchmark_min_num_tokens=769
        benchmark_max_num_tokens=1536
        benchmark_epoch_size=10000
        benchmark_sampler_replacement=true
        benchmark_profile_random_sample_if_failed=true
        official_benchmark_index_sha256="${official_benchmark_index_128_sha256}"
        ;;
    1024:128)
        benchmark_dataset_name="benchmark_long_protein_context_1024"
        benchmark_min_num_tokens=1025
        benchmark_max_num_tokens=1536
        benchmark_epoch_size=10000
        benchmark_sampler_replacement=true
        benchmark_profile_random_sample_if_failed=true
        official_benchmark_index_sha256="${official_benchmark_index_128_sha256}"
        ;;
    1024:800)
        if [ "${profile_uses_msa_aware_dataset}" = "true" ]; then
            benchmark_dataset_name="benchmark_long_protein_context_1024_800_msa2048"
            official_benchmark_index_sha256="${official_benchmark_index_msa2048_sha256}"
        else
            benchmark_dataset_name="benchmark_long_protein_context_1024_800"
            official_benchmark_index_sha256="${official_benchmark_index_800_sha256}"
        fi
        benchmark_min_num_tokens=1025
        benchmark_max_num_tokens=1536
        benchmark_epoch_size=800
        benchmark_sampler_replacement=false
        benchmark_profile_random_sample_if_failed=false
        ;;
    960:800)
        if [ "${profile_uses_msa_aware_dataset}" != "true" ]; then
            echo "The crop960:800 profile requires the strict MSA2048 workload" >&2
            exit 2
        fi
        benchmark_dataset_name="benchmark_long_protein_context_960_800_msa2048"
        # Reuse the existing audited artifact exactly as built.  Its source
        # filter (1025) is stricter than required for an exact crop of 960.
        benchmark_min_num_tokens=1025
        benchmark_max_num_tokens=1536
        benchmark_epoch_size=800
        benchmark_sampler_replacement=false
        benchmark_profile_random_sample_if_failed=false
        official_benchmark_index_sha256="${official_benchmark_index_msa2048_sha256}"
        ;;
    *)
        echo "Supported exact-crop dataset profiles are 384:800, 768:128, 960:800, 1024:128, and 1024:800" >&2
        exit 2
        ;;
esac
export BENCHMARK_CROP_SIZE="${benchmark_crop_size}"
export BENCHMARK_TRAIN_CROP_SIZE="${benchmark_crop_size}"
export BENCHMARK_DATASET_ROWS="${benchmark_dataset_rows}"
export BENCHMARK_DATASET_NAME="${benchmark_dataset_name}"
if [ "${benchmark_dataset_name}" = "benchmark_long_protein_context_384_800_msa2048" ]; then
    default_benchmark_dataset_dir="${repo_dir}/data/benchmark_long_protein_context_1024_800_msa2048"
elif [[ "${workload_profile}" == crop1024_800_msa2048_offload_nuc128_db40_paired_chunk2048_* ]]; then
    default_benchmark_dataset_dir="/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_msa2048_offload/data/benchmark_long_protein_context_1024_800_msa2048"
elif [ "${benchmark_dataset_name}" = "benchmark_long_protein_context_960_800_msa2048" ]; then
    default_benchmark_dataset_dir="${repo_dir}/data/benchmark_long_protein_context_1024_800_msa2048"
else
    default_benchmark_dataset_dir="${repo_dir}/data/${benchmark_dataset_name}"
fi
legacy_benchmark_dataset_dir="${PROTENIX_BENCHMARK_DIR:-}"
benchmark_dataset_dir="${BENCHMARK_DATASET_DIR:-${legacy_benchmark_dataset_dir:-${default_benchmark_dataset_dir}}}"
if [[ -n "${BENCHMARK_DATASET_DIR:-}" && -n "${legacy_benchmark_dataset_dir}" && \
      "${BENCHMARK_DATASET_DIR}" != "${legacy_benchmark_dataset_dir}" ]]; then
    echo "BENCHMARK_DATASET_DIR and PROTENIX_BENCHMARK_DIR disagree" >&2
    exit 2
fi
if [[ -z "${BENCHMARK_DATASET_DIR:-}" && -n "${legacy_benchmark_dataset_dir}" && \
      "$(basename "${legacy_benchmark_dataset_dir}")" != "${benchmark_dataset_name}" ]]; then
    echo "Inherited PROTENIX_BENCHMARK_DIR does not match ${benchmark_dataset_name}; use BENCHMARK_DATASET_DIR for a custom path" >&2
    exit 2
fi
export BENCHMARK_DATASET_DIR="${benchmark_dataset_dir}"
export PROTENIX_BENCHMARK_DIR="${benchmark_dataset_dir}"
export PAIRFORMER_BLOCKS_PER_CKPT="${BENCHMARK_PAIRFORMER_BLOCKS_PER_CKPT:-2}"
export PAIR_TRANSITION_CHUNK_SIZE="${BENCHMARK_PAIR_TRANSITION_CHUNK_SIZE:-128}"
export PAIR_TRANSITION_CHECKPOINT_CHUNKS="${BENCHMARK_PAIR_TRANSITION_CHECKPOINT_CHUNKS:-true}"
export MSA_CHUNK_SIZE="${BENCHMARK_MSA_CHUNK_SIZE:-${profile_msa_chunk_size}}"
export PAIRFORMER_CHECKPOINT_CPU_OFFLOAD_EVERY_N="${BENCHMARK_PAIRFORMER_CHECKPOINT_CPU_OFFLOAD_EVERY_N:-${profile_pairformer_cpu_offload_every_n}}"
export MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N="${BENCHMARK_MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N:-${profile_msa_cpu_offload_every_n}}"
export MODEL_C_Z="${BENCHMARK_MODEL_C_Z:-256}"
export MODEL_C_M="${BENCHMARK_MODEL_C_M:-128}"
# finetune_demo.sh intentionally supports both old lower-case variables and the
# newer upper-case interface. Overwrite both so a reused login shell cannot
# silently change the command while benchmark.sh prints the target values.
export benchmark_exact_crop="${BENCHMARK_EXACT_CROP}"
export pairformer_blocks_per_ckpt="${PAIRFORMER_BLOCKS_PER_CKPT}"
export pair_transition_chunk_size="${PAIR_TRANSITION_CHUNK_SIZE}"
export pair_transition_checkpoint_chunks="${PAIR_TRANSITION_CHECKPOINT_CHUNKS}"
export msa_chunk_size="${MSA_CHUNK_SIZE}"
export pairformer_checkpoint_cpu_offload_every_n="${PAIRFORMER_CHECKPOINT_CPU_OFFLOAD_EVERY_N}"
export msa_checkpoint_cpu_offload_every_n="${MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N}"
export model_c_z="${MODEL_C_Z}"
export model_c_m="${MODEL_C_M}"

# Lock the matched dataset source. PROTENIX_BENCHMARK_DIR itself remains
# configurable so the generated small index can live on a chosen shared path.
export PROTENIX_ROOT_DIR="${official_root}"
export WEIGHTED_MMCIF_DIR="${official_root}/mmcif"
export WEIGHTED_BIOASSEMBLY_DIR="${official_root}/mmcif_bioassembly"
export BENCHMARK_SOURCE_INDEX="${official_source_index}"
export BENCHMARK_MAX_ROWS="${benchmark_dataset_rows}"
export BENCHMARK_MIN_NUM_TOKENS="${BENCHMARK_MIN_NUM_TOKENS:-${benchmark_min_num_tokens}}"
export BENCHMARK_MAX_NUM_TOKENS="${BENCHMARK_MAX_NUM_TOKENS:-${benchmark_max_num_tokens}}"
export BENCHMARK_SELECTION="${BENCHMARK_SELECTION:-top_tokens}"
export BENCHMARK_CROP_METHOD_WEIGHTS="${BENCHMARK_CROP_METHOD_WEIGHTS:-0,1,0}"
export BENCHMARK_MAX_NUC_TOKENS="${BENCHMARK_MAX_NUC_TOKENS:-${profile_max_nuc_tokens}}"
export benchmark_max_nuc_tokens="${BENCHMARK_MAX_NUC_TOKENS}"

export PAIR_MSA_AS_UNPAIRED_ONLY="${BENCHMARK_PAIR_MSA_AS_UNPAIRED_ONLY:-${profile_pair_msa_as_unpaired_only}}"
export SHUFFLE_PAIR_MSA_WHEN_UNPAIRED_ONLY="${BENCHMARK_SHUFFLE_PAIR_MSA:-${profile_shuffle_pair_msa}}"
export MSA_MIN_SIZE_TRAIN="${BENCHMARK_MSA_MIN_SIZE_TRAIN:-${profile_msa_min_size}}"
export MSA_SAMPLE_CUTOFF_TRAIN="${BENCHMARK_MSA_SAMPLE_CUTOFF_TRAIN:-${profile_msa_sample_cutoff}}"
export pair_msa_as_unpaired_only="${PAIR_MSA_AS_UNPAIRED_ONLY}"
export shuffle_pair_msa_when_unpaired_only="${SHUFFLE_PAIR_MSA_WHEN_UNPAIRED_ONLY}"
export msa_min_size_train="${MSA_MIN_SIZE_TRAIN}"
export msa_sample_cutoff_train="${MSA_SAMPLE_CUTOFF_TRAIN}"

export seed="${BENCHMARK_SEED:-42}"
export model_name="${BENCHMARK_MODEL_NAME:-protenix-v2}"
if [[ "${model_name}" != "protenix-v2" ]]; then
    echo "This benchmark harness supports only model_name=protenix-v2" >&2
    exit 2
fi
export dtype="${BENCHMARK_DTYPE:-bf16}"
if [[ "${dtype}" != "bf16" ]]; then
    echo "This benchmark harness currently supports only BENCHMARK_DTYPE=bf16" >&2
    exit 2
fi
export train_crop_size="${benchmark_crop_size}"
export diffusion_batch_size="${BENCHMARK_DIFFUSION_BATCH_SIZE:-${profile_diffusion_batch_size}}"
export diffusion_chunk_size="${BENCHMARK_DIFFUSION_CHUNK_SIZE:-4}"
export N_cycle="${BENCHMARK_N_CYCLE:-4}"
export N_step="${BENCHMARK_N_STEP:-20}"
export iters_to_accumulate="${BENCHMARK_ITERS_TO_ACCUMULATE:-1}"
export ema_decay="${BENCHMARK_EMA_DECAY:-0.999}"
export warmup_steps="${BENCHMARK_LR_WARMUP_STEPS:-2000}"
export lr="${BENCHMARK_LR:-${profile_lr}}"
export msa_max_size_train="${BENCHMARK_MSA_MAX_SIZE_TRAIN:-${profile_msa_max_size}}"
export triangle_attention="${BENCHMARK_TRIANGLE_ATTENTION:-cuequivariance}"
export triangle_multiplicative="${BENCHMARK_TRIANGLE_MULTIPLICATIVE:-cuequivariance}"
export train_sets="${benchmark_dataset_name}"
export train_sample_weights="1.0"
export train_sampler_seed="${BENCHMARK_TRAIN_SAMPLER_SEED:-${profile_train_sampler_seed}}"
export train_sampler_replacement="${BENCHMARK_TRAIN_SAMPLER_REPLACEMENT:-${benchmark_sampler_replacement}}"
export test_sets="${benchmark_dataset_name}"
export num_dl_workers="${BENCHMARK_NUM_DL_WORKERS:-4}"
export epoch_size="${BENCHMARK_EPOCH_SIZE:-${benchmark_epoch_size}}"
export benchmark_random_sample_if_failed="${BENCHMARK_RANDOM_SAMPLE_IF_FAILED:-${benchmark_profile_random_sample_if_failed}}"
export use_wandb=false
export eval_at_end=false
export eval_interval=0
export checkpoint_interval=-1
export empty_cache_interval=0
export log_interval="${benchmark_total_steps}"

export PROTENIX_BENCHMARK_PHASE_MEMORY="${PROTENIX_BENCHMARK_PHASE_MEMORY:-${profile_phase_memory}}"
if [[ "${PROTENIX_BENCHMARK_PHASE_MEMORY}" != "true" && "${PROTENIX_BENCHMARK_PHASE_MEMORY}" != "false" ]]; then
    echo "PROTENIX_BENCHMARK_PHASE_MEMORY must be true or false" >&2
    exit 2
fi

export BENCHMARK_WARMUP_STEPS="${warmup_steps_count}"
export BENCHMARK_MEASURE_STEPS="${measure_steps_count}"
export benchmark_warmup_steps="${warmup_steps_count}"
export benchmark_measure_steps="${measure_steps_count}"
export max_steps="${benchmark_total_steps}"

if [ "${workload_profile}" != "legacy_msa32" ] && \
   { [ "${benchmark_crop_size}" != "${profile_default_crop_size}" ] || \
     [ "${benchmark_dataset_rows}" != "800" ] || \
     [ "${warmup_steps_count}" != "${profile_default_warmup_steps}" ] || \
     [ "${measure_steps_count}" != "${profile_default_measure_steps}" ]; }; then
    echo "${workload_profile} requires crop=${profile_default_crop_size}, dataset_rows=800, and exactly ${profile_default_warmup_steps}+${profile_default_measure_steps} steps" >&2
    exit 2
fi

if [ "${MODEL_C_Z}" != "256" ] || [ "${MODEL_C_M}" != "128" ] || \
   [ "${train_crop_size}" != "${benchmark_crop_size}" ] || \
   [ "${diffusion_batch_size}" != "${profile_diffusion_batch_size}" ] || \
   [ "${diffusion_chunk_size}" != "4" ] || \
   [ "${PAIRFORMER_BLOCKS_PER_CKPT}" != "2" ] || \
   [ "${PAIR_TRANSITION_CHUNK_SIZE}" != "128" ] || \
   [ "${PAIR_TRANSITION_CHECKPOINT_CHUNKS}" != "true" ] || \
   [ "${MSA_CHUNK_SIZE}" != "${profile_msa_chunk_size}" ] || \
   [ "${PAIRFORMER_CHECKPOINT_CPU_OFFLOAD_EVERY_N}" != "${profile_pairformer_cpu_offload_every_n}" ] || \
   [ "${MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N}" != "${profile_msa_cpu_offload_every_n}" ]; then
    echo "Matched benchmark requires the model/checkpoint/offload settings from BENCHMARK_WORKLOAD_PROFILE=${workload_profile}" >&2
    exit 2
fi
if [ "${BENCHMARK_MAX_NUC_TOKENS}" != "${profile_max_nuc_tokens}" ]; then
    echo "${workload_profile} requires max_nuc_tokens=${profile_max_nuc_tokens}" >&2
    exit 2
fi
if [ "${seed}" != "42" ] || [ "${N_cycle}" != "4" ] || \
   [ "${N_step}" != "20" ] || [ "${iters_to_accumulate}" != "1" ] || \
   [ "${num_dl_workers}" != "4" ] || [ "${epoch_size}" != "${benchmark_epoch_size}" ] || \
   [ "${train_sampler_replacement}" != "${benchmark_sampler_replacement}" ] || \
   [ "${benchmark_random_sample_if_failed}" != "${benchmark_profile_random_sample_if_failed}" ] || \
   [ "${ema_decay}" != "0.999" ] || [ "${warmup_steps}" != "2000" ] || \
   [ "${lr}" != "${profile_lr}" ] || [ "${msa_max_size_train}" != "${profile_msa_max_size}" ] || \
   [ "${MSA_MIN_SIZE_TRAIN}" != "${profile_msa_min_size}" ] || \
   [ "${MSA_SAMPLE_CUTOFF_TRAIN}" != "${profile_msa_sample_cutoff}" ] || \
   [ "${PAIR_MSA_AS_UNPAIRED_ONLY}" != "${profile_pair_msa_as_unpaired_only}" ] || \
   [ "${SHUFFLE_PAIR_MSA_WHEN_UNPAIRED_ONLY}" != "${profile_shuffle_pair_msa}" ] || \
   [ "${lr}" != "${profile_lr}" ] || \
   [ "${PROTENIX_BENCHMARK_PHASE_MEMORY}" != "${profile_phase_memory}" ] || \
   [ "${train_sampler_seed}" != "${profile_train_sampler_seed}" ] || \
   [ "${triangle_attention}" != "cuequivariance" ] || \
   [ "${triangle_multiplicative}" != "cuequivariance" ]; then
    echo "Matched benchmark training/MSA/operator settings differ from BENCHMARK_WORKLOAD_PROFILE=${workload_profile}" >&2
    exit 2
fi
if [ "${BENCHMARK_MAX_ROWS}" != "${benchmark_dataset_rows}" ] || \
   [ "${BENCHMARK_MIN_NUM_TOKENS}" != "${benchmark_min_num_tokens}" ] || \
   [ "${BENCHMARK_MAX_NUM_TOKENS}" != "${benchmark_max_num_tokens}" ] || \
   [ "${BENCHMARK_SELECTION}" != "top_tokens" ] || \
   [ "${BENCHMARK_CROP_METHOD_WEIGHTS}" != "0,1,0" ] || \
   [ "${BENCHMARK_SOURCE_INDEX}" != "${official_source_index}" ]; then
    echo "Matched benchmark dataset selection differs from the locked ${benchmark_dataset_rows}-row exact-crop profile" >&2
    exit 2
fi
if [ "${benchmark_dataset_rows}" = "800" ]; then
    required_coverage_steps=100
    if [ "${workload_profile}" != "legacy_msa32" ]; then
        required_coverage_steps=$((profile_default_warmup_steps + profile_default_measure_steps))
    fi
    if [ "${benchmark_total_steps}" != "${required_coverage_steps}" ]; then
        echo "The selected 800-row workload profile requires exactly ${required_coverage_steps} total training steps" >&2
        exit 2
    fi
    if [[ ! "${NNODES}" =~ ^[1-9][0-9]*$ ]] || \
       [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || \
       (( NNODES * NPROC_PER_NODE != 8 )); then
        echo "The 800-row coverage profile requires exactly 8 ranks and an explicit numeric NPROC_PER_NODE" >&2
        exit 2
    fi
fi

skip_final="${BENCHMARK_SKIP_FINAL_CHECKPOINT:-true}"
if [[ "${skip_final}" != "true" && "${skip_final}" != "false" ]]; then
    echo "BENCHMARK_SKIP_FINAL_CHECKPOINT must be true or false" >&2
    exit 2
fi
if [[ "${skip_final}" != "true" ]]; then
    echo "Matched benchmark requires BENCHMARK_SKIP_FINAL_CHECKPOINT=true" >&2
    exit 2
fi
export SKIP_FINAL_CHECKPOINT="${skip_final}"
export skip_final_checkpoint="${skip_final}"

if [ "${DRY_RUN}" != "1" ]; then
    benchmark_index="${PROTENIX_BENCHMARK_DIR}/indices.csv"
    benchmark_summary="${PROTENIX_BENCHMARK_DIR}/summary.json"
    benchmark_node_rank="${NODE_RANK:-${RANK:-0}}"
    if [[ ! "${benchmark_node_rank}" =~ ^[0-9]+$ ]]; then
        echo "NODE_RANK/RANK must be a non-negative integer" >&2
        exit 2
    fi
    conda_activate="${CONDA_ACTIVATE:-/mnt/project_abd/klp/miniconda3/bin/activate}"
    conda_root="${conda_activate%/bin/activate}"
    # Index generation is CPU-only and intentionally uses the existing pinned
    # data environment; the training runtime is selected independently by
    # CONDA_ENV and must pass the cu126 preflight in finetune_demo.sh.
    benchmark_python="${BENCHMARK_INDEX_PYTHON:-${conda_root}/envs/protenix/bin/python}"
    dataset_ready_marker="${profile_root}/benchmark_dataset_ready.json"
    if [ "${profile_uses_msa_aware_dataset}" = "true" ]; then
        benchmark_builder="scripts/benchmark/build_msa_aware_long_protein_benchmark.py"
        benchmark_builder_args=(
            --source-index "${BENCHMARK_SOURCE_INDEX}"
            --output-dir "${PROTENIX_BENCHMARK_DIR}"
            --mapping "${official_root}/common/seq_to_pdb_index.json"
            --bioassembly-dir "${official_root}/mmcif_bioassembly"
            --msa-root "${official_root}/mmcif_msa_template"
            --max-rows "${BENCHMARK_MAX_ROWS}"
            --min-num-tokens "${BENCHMARK_MIN_NUM_TOKENS}"
            --max-num-tokens "${BENCHMARK_MAX_NUM_TOKENS}"
            --selection "${BENCHMARK_SELECTION}"
            --seed "${seed}"
            --target-depth 2048
            --max-msa-size "${msa_max_size_train}"
            --expected-source-sha256 "${official_source_index_sha256}"
            --expected-mapping-sha256 "${official_msa_mapping_sha256}"
            --expected-index-sha256 "${official_benchmark_index_sha256}"
            --ready-marker "${dataset_ready_marker}"
        )
    else
        benchmark_builder="scripts/benchmark/build_long_protein_benchmark.py"
        benchmark_builder_args=(
            --source-index "${BENCHMARK_SOURCE_INDEX}"
            --output-dir "${PROTENIX_BENCHMARK_DIR}"
            --max-rows "${BENCHMARK_MAX_ROWS}"
            --min-num-tokens "${BENCHMARK_MIN_NUM_TOKENS}"
            --max-num-tokens "${BENCHMARK_MAX_NUM_TOKENS}"
            --selection "${BENCHMARK_SELECTION}"
            --seed "${seed}"
            --require-rows "${BENCHMARK_MAX_ROWS}"
            --expected-source-sha256 "${official_source_index_sha256}"
            --expected-index-sha256 "${official_benchmark_index_sha256}"
            --ready-marker "${dataset_ready_marker}"
        )
    fi
    mkdir -p "${profile_root}"
    if [ "${benchmark_node_rank}" = "0" ]; then
        if [ ! -x "${benchmark_python}" ]; then
            echo "Missing benchmark index Python: ${benchmark_python}" >&2
            exit 1
        fi
        if [ -n "${resolved_checkpoint}" ]; then
            observed_checkpoint_sha256="$(sha256sum "${resolved_checkpoint}" | awk '{print $1}')"
            if [ "${observed_checkpoint_sha256}" != "${profile_checkpoint_sha256}" ]; then
                echo "Initial checkpoint SHA256 mismatch for ${resolved_checkpoint}" >&2
                exit 1
            fi
        fi
        if [ "${profile_uses_msa_aware_dataset}" = "true" ]; then
            "${benchmark_python}" "${benchmark_builder}" --validate-only \
                "${benchmark_builder_args[@]}"
        else
            "${benchmark_python}" "${benchmark_builder}" \
                "${benchmark_builder_args[@]}"
        fi
    fi
    wait_seconds="${BENCHMARK_INDEX_WAIT_SECONDS:-600}"
    if [[ ! "${wait_seconds}" =~ ^[0-9]+$ ]]; then
        echo "BENCHMARK_INDEX_WAIT_SECONDS must be a non-negative integer" >&2
        exit 2
    fi
    waited=0
    while { [ ! -f "${benchmark_index}" ] || [ ! -f "${benchmark_summary}" ] || [ ! -f "${dataset_ready_marker}" ]; } && (( waited < wait_seconds )); do
        sleep 1
        waited=$((waited + 1))
    done
    if [ ! -f "${benchmark_index}" ] || [ ! -f "${benchmark_summary}" ] || [ ! -f "${dataset_ready_marker}" ]; then
        echo "Timed out waiting for exact-crop benchmark index in ${PROTENIX_BENCHMARK_DIR}" >&2
        exit 1
    fi
    "${benchmark_python}" "${benchmark_builder}" --validate-only \
        "${benchmark_builder_args[@]}"
fi

echo "Benchmark configuration:"
echo "  BENCHMARK_ID=${BENCHMARK_ID}"
echo "  BENCHMARK_VARIANT=${variant}"
echo "  BENCHMARK_WORKLOAD_PROFILE=${workload_profile}"
echo "  distributed_strategy=${distributed_strategy}"
echo "  optimizer_sharding=${optimizer_sharding}"
echo "  ema_shadow_device=${ema_shadow_device}"
echo "  expected_torch_cuda=${EXPECTED_TORCH_CUDA}"
echo "  fsdp_mesh=${fsdp_mesh}"
echo "  exact_crop=${BENCHMARK_EXACT_CROP} crop_size=${train_crop_size}"
echo "  exact_token_count=${profile_exact_token_count} max_nuc_tokens=${BENCHMARK_MAX_NUC_TOKENS}"
echo "  benchmark_dataset=${benchmark_dataset_name} source_token_range=${BENCHMARK_MIN_NUM_TOKENS}-${BENCHMARK_MAX_NUM_TOKENS}"
echo "  dataset_rows/epoch/replacement=${BENCHMARK_MAX_ROWS}/${epoch_size}/${train_sampler_replacement}"
echo "  random_sample_if_failed=${benchmark_random_sample_if_failed}"
echo "  c_z/c_m=${MODEL_C_Z}/${MODEL_C_M}"
echo "  diffusion_batch_size=${diffusion_batch_size}"
echo "  diffusion_chunk_size=${diffusion_chunk_size}"
echo "  pairformer_blocks_per_ckpt=${PAIRFORMER_BLOCKS_PER_CKPT}"
echo "  msa_chunk_size=${MSA_CHUNK_SIZE:-model_default}"
echo "  pairformer_checkpoint_cpu_offload_every_n=${PAIRFORMER_CHECKPOINT_CPU_OFFLOAD_EVERY_N:-disabled}"
echo "  msa_checkpoint_cpu_offload_every_n=${MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N:-disabled}"
echo "  pair_transition_chunk/checkpoint=${PAIR_TRANSITION_CHUNK_SIZE}/${PAIR_TRANSITION_CHECKPOINT_CHUNKS}"
echo "  benchmark_dataset_dir=${PROTENIX_BENCHMARK_DIR}"
echo "  warmup/measure/max=${benchmark_warmup_steps}/${benchmark_measure_steps}/${max_steps}"
echo "  load_checkpoint_path=${load_checkpoint_path}"
echo "  load_ema_checkpoint_path=${load_ema_checkpoint_path}"
echo "  train_sample_weights=${train_sample_weights}"
echo "  train_sampler_seed=${train_sampler_seed} (global_seed=${seed})"
echo "  MSA min/cutoff=${msa_min_size_train}/${msa_sample_cutoff_train}"
echo "  pair_msa_as_unpaired_only/shuffle=${PAIR_MSA_AS_UNPAIRED_ONLY}/${SHUFFLE_PAIR_MSA_WHEN_UNPAIRED_ONLY}"
echo "  lr=${lr} phase_memory=${PROTENIX_BENCHMARK_PHASE_MEMORY}"
echo "  BASE_DIR=${BASE_DIR}"

exec bash "${repo_dir}/finetune_submit.sh"
