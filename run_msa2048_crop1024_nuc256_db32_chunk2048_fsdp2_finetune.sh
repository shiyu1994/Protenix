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
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"

if (( $# != 0 )); then
    echo "This locked launcher accepts no positional CLI overrides." >&2
    echo "Use RUN_NAME and BASE_DIR for operational settings; the general profile also permits MAX_STEPS." >&2
    exit 2
fi

# This is the production-training counterpart of the validated
# crop1024/MSA2048/chunk2048 benchmark.  Clear benchmark-only state first so a
# reused AIStudio shell cannot silently select the 800-row benchmark dataset,
# its 2+8 timing window, or its no-checkpoint behavior.
while IFS= read -r variable_name; do
    unset "${variable_name}"
done < <(compgen -A variable BENCHMARK_ || true)
unset PROTENIX_BENCHMARK_DIR

# Keep the validated current workload as the default.  The dedicated
# run_immune_heavy_old_content_fsdp2_finetune.sh wrapper selects the second
# profile so training-content changes stay separate from FSDP2/runtime changes.
content_profile="${PROTENIX_TRAIN_CONTENT_PROFILE:-general_msa2048}"
official_root="/mnt/project_abd/public_databases/protenix.20260101"
case "${content_profile}" in
    general_msa2048)
        profile_description="general training from official PX v2"
        initial_checkpoint="${official_root}/checkpoint/protenix-v2.pt"
        initial_checkpoint_sha256="8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599"
        profile_train_sample_weights="0.5,0.3,0.2"
        profile_lr="0.00001"
        profile_max_steps="20000"
        profile_warmup_steps="2000"
        profile_checkpoint_interval="1000"
        profile_msa_max_size_train="16384"
        profile_msa_min_size_train="2048"
        profile_msa_sample_cutoff_train="2048"
        ;;
    immune_heavy_old_content)
        profile_description="old immune-heavy content on the current FSDP2 engine"
        initial_checkpoint="/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/checkpoints/999_ema_0.999.pt"
        initial_checkpoint_sha256="9e0df397d9deb18ea46921c678a62afc4a587241aacd01acd27887729e3a0c3e"
        profile_train_sample_weights="0.25,0.5,0.25"
        profile_lr="0.000003"
        profile_max_steps="2000"
        profile_warmup_steps="2000"
        profile_checkpoint_interval="250"
        profile_msa_max_size_train="16384"
        profile_msa_min_size_train="1"
        profile_msa_sample_cutoff_train="2048"
        ;;
    *)
        echo "Unsupported PROTENIX_TRAIN_CONTENT_PROFILE=${content_profile}" >&2
        exit 2
        ;;
esac

# Keep diffusion-batch experiments explicit and auditable. Public wrappers
# overwrite this profile so values inherited from a reused AIStudio shell
# cannot mutate their locked workload.
diffusion_batch_profile="${PROTENIX_DIFFUSION_BATCH_PROFILE:-32}"
case "${diffusion_batch_profile}" in
    32 | 40 | 48)
        profile_diffusion_batch_size="${diffusion_batch_profile}"
        ;;
    *)
        echo "Unsupported PROTENIX_DIFFUSION_BATCH_PROFILE=${diffusion_batch_profile}; expected 32, 40, or 48." >&2
        exit 2
        ;;
esac

# Keep the nucleic-acid crop cap explicit for each public experiment wrapper.
# The default preserves the validated nuc256 launchers copied into this repo;
# the dedicated db40 wrapper below pins this profile to 128.
max_nuc_tokens_profile="${PROTENIX_MAX_NUC_TOKENS_PROFILE:-256}"
case "${max_nuc_tokens_profile}" in
    128 | 256)
        profile_max_nuc_tokens="${max_nuc_tokens_profile}"
        ;;
    *)
        echo "Unsupported PROTENIX_MAX_NUC_TOKENS_PROFILE=${max_nuc_tokens_profile}; expected 128 or 256." >&2
        exit 2
        ;;
esac

# Keep paired-MSA semantics explicit per public experiment. The historical
# nuc256 launchers preserve their paired-as-unpaired shuffle behavior, while
# the nuc128/db40 wrapper selects normal species pairing.
pair_msa_profile="${PROTENIX_PAIR_MSA_PROFILE:-unpaired_shuffle}"
case "${pair_msa_profile}" in
    unpaired_shuffle)
        profile_pair_msa_as_unpaired_only=true
        profile_shuffle_pair_msa_when_unpaired_only=true
        ;;
    species_paired)
        profile_pair_msa_as_unpaired_only=false
        profile_shuffle_pair_msa_when_unpaired_only=false
        ;;
    *)
        echo "Unsupported PROTENIX_PAIR_MSA_PROFILE=${pair_msa_profile}; expected unpaired_shuffle or species_paired." >&2
        exit 2
        ;;
esac

resolved_run_name="${RUN_NAME:-${run_name:-}}"
if [ -z "${resolved_run_name}" ]; then
    echo "RUN_NAME is required for formal training." >&2
    exit 2
fi
if [[ ! "${resolved_run_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
fi
resolved_base_dir="${BASE_DIR:-${base_dir:-/mnt/project_abd/klp/proteinx_output}}"
smoke_test="${PROTENIX_SMOKE_TEST:-0}"
if [ "${smoke_test}" != "0" ] && [ "${smoke_test}" != "1" ]; then
    echo "PROTENIX_SMOKE_TEST must be 0 or 1." >&2
    exit 2
fi
full_resume="${PROTENIX_FULL_RESUME:-0}"
if [ "${full_resume}" != "0" ] && [ "${full_resume}" != "1" ]; then
    echo "PROTENIX_FULL_RESUME must be 0 or 1." >&2
    exit 2
fi

resume_smoke_steps="${PROTENIX_RESUME_SMOKE_STEPS:-0}"
if [[ ! "${resume_smoke_steps}" =~ ^[0-9]+$ ]]; then
    echo "PROTENIX_RESUME_SMOKE_STEPS must be a non-negative integer." >&2
    exit 2
fi
if [ "${full_resume}" = "0" ] && [ "${resume_smoke_steps}" != "0" ]; then
    echo "PROTENIX_RESUME_SMOKE_STEPS requires PROTENIX_FULL_RESUME=1." >&2
    exit 2
fi
if [ "${full_resume}" = "1" ] && [ "${smoke_test}" = "1" ]; then
    echo "Full resume smoke must use PROTENIX_RESUME_SMOKE_STEPS, not PROTENIX_SMOKE_TEST." >&2
    exit 2
fi

if [ "${full_resume}" = "1" ]; then
    resume_checkpoint_step="${PROTENIX_RESUME_CHECKPOINT_STEP:-}"
    if [[ ! "${resume_checkpoint_step}" =~ ^[0-9]+$ ]]; then
        echo "PROTENIX_RESUME_CHECKPOINT_STEP is required and must be a non-negative integer." >&2
        exit 2
    fi
    next_resume_step=$((resume_checkpoint_step + 1))
    if [ "${resume_smoke_steps}" != "0" ]; then
        # max_steps is an absolute total-step limit. A checkpoint named N.pt
        # resumes at N+1, so K smoke updates end at N+1+K.
        resolved_max_steps=$((next_resume_step + resume_smoke_steps))
    else
        resolved_max_steps="${MAX_STEPS:-${max_steps:-${profile_max_steps}}}"
    fi
    if [[ ! "${resolved_max_steps}" =~ ^[1-9][0-9]*$ ]] \
        || [ "${resolved_max_steps}" -le "${next_resume_step}" ]; then
        echo "Full resume MAX_STEPS must be greater than checkpoint_step+1 (${next_resume_step})." >&2
        exit 2
    fi
elif [ "${content_profile}" = "immune_heavy_old_content" ]; then
    # The formal old-content trial is intentionally locked to the historical
    # 2,000-step schedule.  A separate one-step mode exercises checkpoint
    # loading, forward, backward, and optimizer update without writing a large
    # checkpoint; it cannot silently turn into a short formal run.
    if [ "${smoke_test}" = "1" ]; then
        resolved_max_steps=1
    else
        if [ -n "${MAX_STEPS:-}" ] && [ "${MAX_STEPS}" != "${profile_max_steps}" ]; then
            echo "immune_heavy_old_content locks MAX_STEPS=${profile_max_steps}; unset MAX_STEPS or use PROTENIX_SMOKE_TEST=1." >&2
            exit 2
        fi
        if [ -n "${max_steps:-}" ] && [ "${max_steps}" != "${profile_max_steps}" ]; then
            echo "immune_heavy_old_content locks max_steps=${profile_max_steps}; unset max_steps or use PROTENIX_SMOKE_TEST=1." >&2
            exit 2
        fi
        resolved_max_steps="${profile_max_steps}"
    fi
else
    resolved_max_steps="${MAX_STEPS:-${max_steps:-${profile_max_steps}}}"
fi
if [[ ! "${resolved_max_steps}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_STEPS must be a positive integer." >&2
    exit 2
fi

export RUN_NAME="${resolved_run_name}"
export run_name="${resolved_run_name}"
export BASE_DIR="${resolved_base_dir}"
export base_dir="${resolved_base_dir}"
export MAX_STEPS="${resolved_max_steps}"
export max_steps="${resolved_max_steps}"

# A fresh fine-tune initializes raw and EMA model parameters from the profile
# checkpoint. A full resume instead loads live/raw state from an explicit raw
# checkpoint and EMA shadow parameters from its explicit EMA companion.
export PROTENIX_ROOT_DIR="${official_root}"
if [ "${full_resume}" = "1" ]; then
    resume_raw_checkpoint="${PROTENIX_RESUME_RAW_CHECKPOINT:-}"
    resume_ema_checkpoint="${PROTENIX_RESUME_EMA_CHECKPOINT:-}"
    if [ -z "${resume_raw_checkpoint}" ] || [ -z "${resume_ema_checkpoint}" ]; then
        echo "Full resume requires PROTENIX_RESUME_RAW_CHECKPOINT and PROTENIX_RESUME_EMA_CHECKPOINT." >&2
        exit 2
    fi
    if [ "${resume_raw_checkpoint}" = "${resume_ema_checkpoint}" ]; then
        echo "Raw and EMA resume checkpoints must be different files." >&2
        exit 2
    fi
    if [[ "$(basename "${resume_raw_checkpoint}")" == *_ema_* ]]; then
        echo "PROTENIX_RESUME_RAW_CHECKPOINT must not point to an EMA checkpoint." >&2
        exit 2
    fi
    expected_raw_basename="${resume_checkpoint_step}.pt"
    expected_ema_basename="${resume_checkpoint_step}_ema_0.999.pt"
    if [ "$(basename "${resume_raw_checkpoint}")" != "${expected_raw_basename}" ]; then
        echo "Raw checkpoint basename must be ${expected_raw_basename}." >&2
        exit 2
    fi
    if [ "$(basename "${resume_ema_checkpoint}")" != "${expected_ema_basename}" ]; then
        echo "EMA checkpoint basename must be ${expected_ema_basename}." >&2
        exit 2
    fi

    export CHECKPOINT_PATH="${resume_raw_checkpoint}"
    export checkpoint_path="${resume_raw_checkpoint}"
    export LOAD_CHECKPOINT_PATH="${resume_raw_checkpoint}"
    export load_checkpoint_path="${resume_raw_checkpoint}"
    export LOAD_EMA_CHECKPOINT_PATH="${resume_ema_checkpoint}"
    export load_ema_checkpoint_path="${resume_ema_checkpoint}"
    export LOAD_PARAMS_ONLY=false
    export load_params_only=false
    export EXPECTED_CHECKPOINT_SHA256="${PROTENIX_RESUME_RAW_SHA256:-}"

    verify_resume_sha256="${PROTENIX_VERIFY_RESUME_SHA256:-0}"
    if [ "${verify_resume_sha256}" != "0" ] && [ "${verify_resume_sha256}" != "1" ]; then
        echo "PROTENIX_VERIFY_RESUME_SHA256 must be 0 or 1." >&2
        exit 2
    fi
    if [ "${DRY_RUN:-0}" != "1" ]; then
        for checkpoint_file in "${resume_raw_checkpoint}" "${resume_ema_checkpoint}"; do
            if [ ! -f "${checkpoint_file}" ]; then
                echo "Missing resume checkpoint: ${checkpoint_file}" >&2
                exit 1
            fi
        done
        if [ "${verify_resume_sha256}" = "1" ]; then
            expected_raw_sha256="${PROTENIX_RESUME_RAW_SHA256:-}"
            expected_ema_sha256="${PROTENIX_RESUME_EMA_SHA256:-}"
            if [ -z "${expected_raw_sha256}" ] || [ -z "${expected_ema_sha256}" ]; then
                echo "Strict resume SHA256 verification requires both expected digests." >&2
                exit 2
            fi
            observed_raw_sha256="$(sha256sum "${resume_raw_checkpoint}" | awk '{print $1}')"
            observed_ema_sha256="$(sha256sum "${resume_ema_checkpoint}" | awk '{print $1}')"
            if [ "${observed_raw_sha256}" != "${expected_raw_sha256}" ]; then
                echo "Raw resume checkpoint SHA256 mismatch." >&2
                echo "Expected: ${expected_raw_sha256}" >&2
                echo "Observed: ${observed_raw_sha256}" >&2
                exit 1
            fi
            if [ "${observed_ema_sha256}" != "${expected_ema_sha256}" ]; then
                echo "EMA resume checkpoint SHA256 mismatch." >&2
                echo "Expected: ${expected_ema_sha256}" >&2
                echo "Observed: ${observed_ema_sha256}" >&2
                exit 1
            fi
        fi
    fi
else
    export CHECKPOINT_PATH="${initial_checkpoint}"
    export checkpoint_path="${initial_checkpoint}"
    export LOAD_CHECKPOINT_PATH="${initial_checkpoint}"
    export load_checkpoint_path="${initial_checkpoint}"
    export LOAD_EMA_CHECKPOINT_PATH="${initial_checkpoint}"
    export load_ema_checkpoint_path="${initial_checkpoint}"
    export LOAD_PARAMS_ONLY=true
    export load_params_only=true
    export EXPECTED_CHECKPOINT_SHA256="${initial_checkpoint_sha256}"

    if [ "${DRY_RUN:-0}" != "1" ]; then
        if [ ! -f "${initial_checkpoint}" ]; then
            echo "Missing profile checkpoint: ${initial_checkpoint}" >&2
            exit 1
        fi
        observed_checkpoint_sha256="$(sha256sum "${initial_checkpoint}" | awk '{print $1}')"
        if [ "${observed_checkpoint_sha256}" != "${initial_checkpoint_sha256}" ]; then
            echo "Profile checkpoint SHA256 mismatch." >&2
            echo "Expected: ${initial_checkpoint_sha256}" >&2
            echo "Observed: ${observed_checkpoint_sha256}" >&2
            exit 1
        fi
    fi
fi

# Lock the formal dataset mixture.  This intentionally does not use the
# benchmark_long_protein_context_* index that was created only for profiling.
weighted_set="weightedPDB_before210930_v20260101"
sabdab_root="/mnt/project_abd/zhe/protenix_copy/vepfs-mlp2/project-abd/zhujianwei/public_databases/protenix"
export WEIGHTED_TRAIN_SET="${weighted_set}"
export weighted_train_set="${weighted_set}"
export WEIGHTED_MMCIF_DIR="${official_root}/mmcif"
export WEIGHTED_BIOASSEMBLY_DIR="${official_root}/mmcif_bioassembly"
export WEIGHTED_INDICES_FPATH="${official_root}/indices/indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz"
export SABDAB_ROOT_DIR="${sabdab_root}"
export sabdab_root_dir="${sabdab_root}"
export SABDAB_TRAIN_MMCIF_DIR="${sabdab_root}/sabdab-train_mmcif"
export SABDAB_TRAIN_BIOASSEMBLY_DIR="${sabdab_root}/sabdab-train_bioassembly"
export SABDAB_TRAIN_INDICES_FPATH="${sabdab_root}/indices/sabdab-train.csv"
export SABDAB_TRAIN_SEQ_INDEX_JSON="${sabdab_root}/sabdab-train.json"
export DISTILL_SABDAB_MMCIF_DIR="${sabdab_root}/distill-sabdab_cif"
export DISTILL_SABDAB_BIOASSEMBLY_DIR="${sabdab_root}/distill-sabdab_bioassembly"
export DISTILL_SABDAB_INDICES_FPATH="${sabdab_root}/indices/distill-sabdab.csv"
export DISTILL_SABDAB_SEQ_INDEX_JSON="${sabdab_root}/sab-20260416.json"
export SABDAB_MSA_TEMPLATE_DIR="${sabdab_root}/mmcif_msa_template"
export POSEBUSTERS_MMCIF_DIR="${official_root}/posebusters_mmcif"
export POSEBUSTERS_BIOASSEMBLY_DIR="${official_root}/posebusters_bioassembly"
export POSEBUSTERS_INDICES_FPATH="${official_root}/indices/posebusters_indices_mainchain_interface.csv"
export train_sets="${weighted_set},sabdab-train,distill-sabdab"
export train_sample_weights="${profile_train_sample_weights}"
export TRAIN_SAMPLER_SEED=-1
export train_sampler_seed=-1
export TRAIN_SAMPLER_REPLACEMENT=true
export train_sampler_replacement=true
export EPOCH_SIZE=10000
export epoch_size=10000
if [ "${full_resume}" = "1" ]; then
    # The failed run used 16 workers/rank (512 workers across four nodes) and
    # died in torch_shm_manager IPC. Resume with a lower, explicit default.
    resolved_num_dl_workers="${PROTENIX_RESUME_NUM_DL_WORKERS:-8}"
else
    resolved_num_dl_workers=16
fi
if [[ ! "${resolved_num_dl_workers}" =~ ^[0-9]+$ ]]; then
    echo "Resolved data-loader worker count must be a non-negative integer." >&2
    exit 2
fi
export NUM_DL_WORKERS="${resolved_num_dl_workers}"
export num_dl_workers="${resolved_num_dl_workers}"

# Model shape and compute workload selected by the completed benchmark series.
export MODEL_NAME=protenix-v2
export model_name=protenix-v2
export MODEL_C_Z=256
export model_c_z=256
export MODEL_C_M=128
export model_c_m=128
export train_crop_size=1024
export diffusion_batch_size="${profile_diffusion_batch_size}"
export DIFFUSION_CHUNK_SIZE=4
export diffusion_chunk_size=4
export dtype=bf16
export N_cycle=4
export N_step=20
export seed=42
export iters_to_accumulate=1
export lr="${profile_lr}"
export warmup_steps="${profile_warmup_steps}"
export triangle_attention=cuequivariance
export triangle_multiplicative=cuequivariance

# FSDP2 full-shard and rank-local CPU EMA.  The CPU offload settings below are
# activation-checkpoint input offload; model parameters remain FSDP2-managed.
export DISTRIBUTED_STRATEGY=fsdp2
export distributed_strategy=fsdp2
export OPTIMIZER_SHARDING=none
export optimizer_sharding=none
export FSDP_MESH=auto
export fsdp_mesh=auto
export FSDP_PARAM_DTYPE=fp32
export fsdp_param_dtype=fp32
export FSDP_REDUCE_DTYPE=fp32
export fsdp_reduce_dtype=fp32
export FSDP_RESHARD_AFTER_FORWARD=true
export fsdp_reshard_after_forward=true
export EMA_SHADOW_DEVICE=cpu
export ema_shadow_device=cpu
export ema_decay=0.999

export PAIRFORMER_BLOCKS_PER_CKPT=2
export pairformer_blocks_per_ckpt=2
export PAIRFORMER_CHECKPOINT_CPU_OFFLOAD_EVERY_N=2
export pairformer_checkpoint_cpu_offload_every_n=2
export MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N=1
export msa_checkpoint_cpu_offload_every_n=1
export PAIR_TRANSITION_CHUNK_SIZE=128
export pair_transition_chunk_size=128
export PAIR_TRANSITION_CHECKPOINT_CHUNKS=true
export pair_transition_checkpoint_chunks=true

# Both profiles keep the current 16,384-row featurizer pool and 2,048-row
# runtime compute height (one chunk). The general profile uses exactly 2,048
# rows whenever available; the old-content profile independently draws each
# recycle's real row count uniformly from 1..min(actual_depth, 2,048).
export MSA_MIN_SIZE_TRAIN="${profile_msa_min_size_train}"
export msa_min_size_train="${profile_msa_min_size_train}"
export MSA_SAMPLE_CUTOFF_TRAIN="${profile_msa_sample_cutoff_train}"
export msa_sample_cutoff_train="${profile_msa_sample_cutoff_train}"
export msa_max_size_train="${profile_msa_max_size_train}"
export MSA_CHUNK_SIZE=2048
export msa_chunk_size=2048
export PAIR_MSA_AS_UNPAIRED_ONLY="${profile_pair_msa_as_unpaired_only}"
export pair_msa_as_unpaired_only="${profile_pair_msa_as_unpaired_only}"
export SHUFFLE_PAIR_MSA_WHEN_UNPAIRED_ONLY="${profile_shuffle_pair_msa_when_unpaired_only}"
export shuffle_pair_msa_when_unpaired_only="${profile_shuffle_pair_msa_when_unpaired_only}"

# Production lifecycle: no benchmark timing window, periodic raw+EMA
# checkpoints, a final checkpoint, and no periodic evaluation.
export BENCHMARK_EXACT_CROP=false
export benchmark_exact_crop=false
export BENCHMARK_WARMUP_STEPS=-1
export benchmark_warmup_steps=-1
export BENCHMARK_MEASURE_STEPS=-1
export benchmark_measure_steps=-1
if [ "${smoke_test}" = "1" ] || [ "${resume_smoke_steps}" != "0" ]; then
    export SKIP_FINAL_CHECKPOINT=true
    export skip_final_checkpoint=true
    export checkpoint_interval=-1
else
    export SKIP_FINAL_CHECKPOINT=false
    export skip_final_checkpoint=false
    export checkpoint_interval="${profile_checkpoint_interval}"
fi
export log_interval=50
export empty_cache_interval=0
export use_wandb=false
export project=protenix
export test_sets=posebusters_0925
export eval_interval=0
export EVAL_AT_END=false
export eval_at_end=false
export EVAL_ONLY=false
export eval_only=false
export eval_ema_only=false
export CHECK_ONLY=0
export SKIP_PATH_CHECK=0

# Keep CUDA extension and Triton compilation in short node-local paths.  The
# selected PyTorch wheel embeds CUDA 12.6 and is supported on the CUDA 12.9
# driver/toolkit currently installed on AIStudio.
runtime_tag="$(printf '%s' "${resolved_run_name}" | sha256sum | cut -c1-12)"
node_rank="${NODE_RANK:-${RANK:-${MLP_ROLE_INDEX:-0}}}"
export EXPECTED_TORCH_CUDA=12.6

# CUDA extension compilation needs a matched C/C++ compiler pair. AIStudio
# images do not all install the same versioned GCC paths, so never assume that
# gcc-13 exists. An explicit override must provide both executables; otherwise
# select the newest installed system pair supported by the current images.
requested_cuda_cc="${PROTENIX_CUDA_HOST_CC:-}"
requested_cuda_cxx="${PROTENIX_CUDA_HOST_CXX:-}"
if { [ -n "${requested_cuda_cc}" ] && [ -z "${requested_cuda_cxx}" ]; } \
    || { [ -z "${requested_cuda_cc}" ] && [ -n "${requested_cuda_cxx}" ]; }; then
    echo "PROTENIX_CUDA_HOST_CC and PROTENIX_CUDA_HOST_CXX must be set together." >&2
    exit 2
fi
if [ -z "${requested_cuda_cc}" ]; then
    for compiler_version in 13 12 11; do
        candidate_cc="/usr/bin/gcc-${compiler_version}"
        candidate_cxx="/usr/bin/g++-${compiler_version}"
        if [ -x "${candidate_cc}" ] && [ -x "${candidate_cxx}" ]; then
            requested_cuda_cc="${candidate_cc}"
            requested_cuda_cxx="${candidate_cxx}"
            break
        fi
    done
fi
if [ -z "${requested_cuda_cc}" ]; then
    requested_cuda_cc="$(command -v gcc || true)"
    requested_cuda_cxx="$(command -v g++ || true)"
fi
if [ -z "${requested_cuda_cc}" ] || [ ! -x "${requested_cuda_cc}" ]; then
    echo "No executable CUDA host C compiler found: '${requested_cuda_cc}'." >&2
    exit 2
fi
if [ -z "${requested_cuda_cxx}" ] || [ ! -x "${requested_cuda_cxx}" ]; then
    echo "No executable CUDA host C++ compiler found: '${requested_cuda_cxx}'." >&2
    exit 2
fi
export PROTENIX_CUDA_HOST_CC="${requested_cuda_cc}"
export PROTENIX_CUDA_HOST_CXX="${requested_cuda_cxx}"
export PROTENIX_TRITON_CACHE_ROOT="${PROTENIX_TRITON_CACHE_ROOT:-/tmp/protenix_triton_cache/train_${runtime_tag}/node_${node_rank}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/protenix_torch_extensions/train_${runtime_tag}/node_${node_rank}}"
runtime_tmpdir="${PROTENIX_RUNTIME_TMPDIR:-/tmp/ptx_train_${runtime_tag}_${node_rank}}"
mkdir -p "${runtime_tmpdir}" "${TORCH_EXTENSIONS_DIR}"
export TMPDIR="${runtime_tmpdir}"
export TEMP="${runtime_tmpdir}"
export TMP="${runtime_tmpdir}"

echo "Locked Protenix formal training profile:"
echo "  content_profile=${content_profile} (${profile_description})"
echo "  run=${RUN_NAME} base_dir=${BASE_DIR} max_steps=${max_steps}"
if [ "${full_resume}" = "1" ]; then
    echo "  resume=full checkpoint_step=${resume_checkpoint_step} next_step=${next_resume_step}"
    echo "  raw=${resume_raw_checkpoint} ema=${resume_ema_checkpoint}"
else
    echo "  init=${initial_checkpoint} sha256=${initial_checkpoint_sha256}"
fi
echo "  data=${train_sets} weights=${train_sample_weights}"
echo "  num_dl_workers=${resolved_num_dl_workers} mp_sharing_strategy=${PROTENIX_MP_SHARING_STRATEGY:-file_system}"
echo "  crop=1024 max_nuc_tokens=${profile_max_nuc_tokens} diffusion_batch/chunk=${profile_diffusion_batch_size}/4"
echo "  MSA min/cutoff/pool/chunk=${profile_msa_min_size_train}/${profile_msa_sample_cutoff_train}/${profile_msa_max_size_train}/2048"
echo "  pair_msa_profile=${pair_msa_profile} pair_as_unpaired=${profile_pair_msa_as_unpaired_only} shuffle_paired_hits=${profile_shuffle_pair_msa_when_unpaired_only}"
echo "  c_z/c_m=256/128 FSDP2=full-shard EMA=cpu"
echo "  cuda_host_compiler=${PROTENIX_CUDA_HOST_CC} / ${PROTENIX_CUDA_HOST_CXX}"
echo "  train_scope=full_model (expected 464,441,841 trainable + 590 fixed constants)"
echo "  Pairformer ckpt/offload=2/2 MSA offload=1 Transition chunk/checkpoint=128/true"
echo "  lr/warmup=${profile_lr}/${profile_warmup_steps} checkpoint_interval=${checkpoint_interval} final_checkpoint=$([ "${skip_final_checkpoint}" = false ] && echo true || echo false)"

resume_state_args=()
if [ "${full_resume}" = "1" ]; then
    # Pin every full-state switch on the command line so an inherited shell or
    # future config-default change cannot silently turn this into params-only
    # initialization or a partial optimizer/scheduler resume.
    resume_state_args=(
        --skip_load_optimizer false
        --skip_load_scheduler false
        --skip_load_step false
        --load_step_for_scheduler true
    )
fi

# max_nuc_tokens and the two SAbDab pair-MSA values are dataset-local configs.
# Pass the selected pair mode at both levels so no dataset-local default can
# silently override the locked experiment profile.
exec bash "${repo_dir}/finetune_submit.sh" \
    "${resume_state_args[@]}" \
    --data.${weighted_set}.cropping_configs.max_nuc_tokens "${profile_max_nuc_tokens}" \
    --data.sabdab-train.cropping_configs.max_nuc_tokens "${profile_max_nuc_tokens}" \
    --data.distill-sabdab.cropping_configs.max_nuc_tokens "${profile_max_nuc_tokens}" \
    --data.sabdab-train.msa.pair_msa_as_unpaired_only "${profile_pair_msa_as_unpaired_only}" \
    --data.distill-sabdab.msa.pair_msa_as_unpaired_only "${profile_pair_msa_as_unpaired_only}" \
    --model.msa_module.blocks_per_ckpt 1 \
    --model.msa_module.msa_max_size 2048 \
    --model.diffusion_module.blocks_per_ckpt 1 \
    --model.diffusion_module.use_fine_grained_checkpoint true \
    --max_atoms_per_token 24 \
    --enable_tf32 false \
    --train_confidence_only false \
    --load_strict true \
    --lr_scheduler af3 \
    --grad_clip_norm 10 \
    --decay_every_n_steps 50000 \
    --adam.beta1 0.9 \
    --adam.beta2 0.95 \
    --adam.weight_decay 0.00000001 \
    --adam.use_adamw false
