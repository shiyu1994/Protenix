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

# Speed trials must remain random-init comparisons. Reject every checkpoint
# source spelling used by the training, resume, and benchmark launchers.
checkpoint_source_env_vars=(
    CHECKPOINT
    checkpoint
    CHECKPOINT_PATH
    checkpoint_path
    LOAD_CHECKPOINT_PATH
    load_checkpoint_path
    LOAD_EMA_CHECKPOINT_PATH
    load_ema_checkpoint_path
    RAW_CHECKPOINT
    raw_checkpoint
    RAW_CHECKPOINT_PATH
    raw_checkpoint_path
    EMA_CHECKPOINT
    ema_checkpoint
    EMA_CHECKPOINT_PATH
    ema_checkpoint_path
    INITIAL_CHECKPOINT
    initial_checkpoint
    INITIAL_CHECKPOINT_PATH
    initial_checkpoint_path
    RESUME_CHECKPOINT
    resume_checkpoint
    RESUME_CHECKPOINT_PATH
    resume_checkpoint_path
    RESUME_RAW_CHECKPOINT
    RESUME_EMA_CHECKPOINT
    RESUME_RUN_DIR
    RESUME_STEP
    PROTENIX_RESUME_RAW_CHECKPOINT
    PROTENIX_RESUME_EMA_CHECKPOINT
    PROTENIX_RESUME_CHECKPOINT_STEP
    PROTENIX_RESUME_RUN_DIR
    PROTENIX_RESUME_STEP
    BENCHMARK_LOAD_CHECKPOINT_PATH
    BENCHMARK_LOAD_EMA_CHECKPOINT_PATH
    EXPECTED_CHECKPOINT_SHA256
)
for variable_name in "${checkpoint_source_env_vars[@]}"; do
    if [ -n "${!variable_name-}" ]; then
        echo "Random-init speed trial rejects checkpoint environment variable ${variable_name}." >&2
        exit 2
    fi
done
if [ "${PROTENIX_FULL_RESUME:-0}" != "0" ] \
    || [ "${PROTENIX_RESUME_SMOKE_STEPS:-0}" != "0" ]; then
    echo "Random-init speed trial rejects all resume modes." >&2
    exit 2
fi

for argument in "$@"; do
    case "${argument}" in
        --checkpoint_path | --checkpoint_path=* | --checkpoint-path | --checkpoint-path=* \
            | --load_checkpoint_path | --load_checkpoint_path=* \
            | --load-checkpoint-path | --load-checkpoint-path=* \
            | --load_ema_checkpoint_path | --load_ema_checkpoint_path=* \
            | --load-ema-checkpoint-path | --load-ema-checkpoint-path=* \
            | --raw_checkpoint | --raw_checkpoint=* | --raw-checkpoint | --raw-checkpoint=* \
            | --ema_checkpoint | --ema_checkpoint=* | --ema-checkpoint | --ema-checkpoint=* \
            | --load_params_only | --load_params_only=* \
            | --skip_load_optimizer | --skip_load_optimizer=* \
            | --skip_load_scheduler | --skip_load_scheduler=* \
            | --skip_load_step | --skip_load_step=* \
            | --load_step_for_scheduler | --load_step_for_scheduler=*)
            echo "Random-init speed trial rejects checkpoint/resume parameter ${argument}." >&2
            exit 2
            ;;
    esac
done
if [ "$#" -ne 0 ]; then
    echo "This locked speed-trial launcher does not accept command-line overrides: $*" >&2
    exit 2
fi

speed_profile="${SPEED_PROFILE:-stable}"
case "${speed_profile}" in
    stable | lowrisk | cache | chunk8 | chunk12 | chunk16 | chunk24 | chunk48 \
        | grouped | transition | pf16 | pf18 | pf19 | pf20 | pf21 | pf22 \
        | pf23 | pf24 | pf48 \
        | trunk_nockpt | all_nockpt)
        ;;
    *)
        echo "Unsupported SPEED_PROFILE=${speed_profile}." >&2
        echo "Expected stable, lowrisk, cache, chunk8/12/16/24/48, grouped," >&2
        echo "transition, pf16, pf18, pf19, pf20, pf21, pf22, pf23, pf24, pf48," >&2
        echo "trunk_nockpt, or all_nockpt." >&2
        exit 2
        ;;
esac

# Start from the current stable crop384 implementation. Every non-stable
# profile below is cumulative so a result can be attributed to one documented
# delta from the preceding layer.
resolved_bucket_view=false
resolved_adam_fused=false
resolved_ema_device=cpu
resolved_shared_vars_cache=false
resolved_diffusion_chunk=4
resolved_pairformer_blocks=2
resolved_msa_blocks=1
resolved_diffusion_blocks=1
resolved_fine_checkpoint=true
resolved_pairformer_offload=2
resolved_msa_offload=1
resolved_transition_chunk=128
resolved_transition_checkpoint=true

if [ "${speed_profile}" != "stable" ]; then
    # lowrisk
    resolved_bucket_view=true
    resolved_adam_fused=true
    resolved_ema_device=model
    resolved_pairformer_offload=null
    resolved_msa_offload=null
    resolved_pairformer_blocks=4
    resolved_msa_blocks=4
fi

case "${speed_profile}" in
    cache | chunk8 | chunk12 | chunk16 | chunk24 | chunk48 | grouped | transition \
        | pf16 | pf18 | pf19 | pf20 | pf21 | pf22 | pf23 | pf24 | pf48 \
        | trunk_nockpt | all_nockpt)
        resolved_shared_vars_cache=true
        ;;
esac

case "${speed_profile}" in
    chunk8 | chunk12 | chunk16 | chunk24 | chunk48)
        resolved_diffusion_chunk="${speed_profile#chunk}"
        ;;
    grouped | transition | pf16 | pf18 | pf19 | pf20 | pf21 | pf22 \
        | pf23 | pf24 | pf48 \
        | trunk_nockpt | all_nockpt)
        if [ "${speed_profile}" = "all_nockpt" ]; then
            default_trial_diffusion_chunk=4
        else
            default_trial_diffusion_chunk=24
        fi
        resolved_diffusion_chunk="${TRIAL_DIFFUSION_CHUNK:-${default_trial_diffusion_chunk}}"
        case "${resolved_diffusion_chunk}" in
            4 | 8 | 12 | 16 | 24 | 48)
                ;;
            *)
                echo "TRIAL_DIFFUSION_CHUNK must be one of 4, 8, 12, 16, 24, or 48." >&2
                exit 2
                ;;
        esac
        resolved_pairformer_blocks=8
        resolved_msa_blocks=4
        resolved_diffusion_blocks=3
        ;;
    *)
        if [ -n "${TRIAL_DIFFUSION_CHUNK:-}" ]; then
            echo "TRIAL_DIFFUSION_CHUNK is only valid for grouped, transition," >&2
            echo "pf16, pf18, pf19, pf20, pf21, pf22, pf23, pf24, pf48," >&2
            echo "trunk_nockpt," >&2
            echo "or all_nockpt." >&2
            exit 2
        fi
        ;;
esac

case "${speed_profile}" in
    transition | pf16 | pf18 | pf19 | pf20 | pf21 | pf22 | pf23 | pf24 | pf48 \
        | trunk_nockpt | all_nockpt)
        resolved_transition_chunk=null
        resolved_transition_checkpoint=false
        ;;
esac

case "${speed_profile}" in
    pf16)
        resolved_pairformer_blocks=16
        ;;
    pf18)
        resolved_pairformer_blocks=18
        ;;
    pf19)
        resolved_pairformer_blocks=19
        ;;
    pf20)
        resolved_pairformer_blocks=20
        ;;
    pf21)
        resolved_pairformer_blocks=21
        ;;
    pf22)
        resolved_pairformer_blocks=22
        ;;
    pf23)
        resolved_pairformer_blocks=23
        ;;
    pf24)
        resolved_pairformer_blocks=24
        ;;
    pf48)
        resolved_pairformer_blocks=48
        ;;
esac
if [ "${speed_profile}" = "trunk_nockpt" ] \
    || [ "${speed_profile}" = "all_nockpt" ]; then
    resolved_pairformer_blocks=null
    resolved_msa_blocks=null
fi
if [ "${speed_profile}" = "all_nockpt" ]; then
    resolved_diffusion_blocks=null
    resolved_fine_checkpoint=false
fi

speed_screen="${SPEED_SCREEN:-0}"
if [ "${speed_screen}" != "0" ] && [ "${speed_screen}" != "1" ]; then
    echo "SPEED_SCREEN must be 0 or 1." >&2
    exit 2
fi
speed_nuc_stress="${SPEED_NUC_STRESS:-0}"
if [ "${speed_nuc_stress}" != "0" ] && [ "${speed_nuc_stress}" != "1" ]; then
    echo "SPEED_NUC_STRESS must be 0 or 1." >&2
    exit 2
fi
if [ "${speed_screen}" = "1" ] && [ "${speed_nuc_stress}" = "1" ]; then
    echo "SPEED_SCREEN=1 and SPEED_NUC_STRESS=1 are mutually exclusive." >&2
    exit 2
fi
if [ "${speed_nuc_stress}" = "1" ]; then
    # With the audited 800-row no-replacement dataset and sampler seed 77,
    # rank 5 receives 7d1a at zero-based step 30 (human step 31). Measure only
    # that final step so the candidate profile is tested on the known NA case.
    resolved_warmup_steps=30
    resolved_measure_steps=1
    resolved_max_steps=31
elif [ "${speed_screen}" = "1" ]; then
    resolved_warmup_steps=0
    resolved_measure_steps=1
    resolved_max_steps=1
else
    resolved_warmup_steps=10
    resolved_measure_steps=20
    resolved_max_steps=30
fi

trial_id="${TRIAL_ID:-trial0}"
if [[ ! "${trial_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "TRIAL_ID may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
fi
resolved_run_name="${RUN_NAME:-crop384_db48_ddp_speed_${speed_profile}_${trial_id}}"
if [[ ! "${resolved_run_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
fi
resolved_base_dir="${BASE_DIR:-${repo_dir}/test_outputs/crop384_db48_ddp_speed_trials}"

# This experiment is intentionally single-node, eight-rank DDP. Reject stale
# multi-node coordinates rather than silently benchmarking a different global
# target batch.
for node_count_variable in NNODES nnodes MLP_WORKER_NUM SLURM_NNODES; do
    if [ -n "${!node_count_variable-}" ] && [ "${!node_count_variable}" != "1" ]; then
        echo "Speed trial requires one node; ${node_count_variable}=${!node_count_variable}." >&2
        exit 2
    fi
done
for node_rank_variable in NODE_RANK node_rank MLP_ROLE_INDEX SLURM_NODEID; do
    if [ -n "${!node_rank_variable-}" ] && [ "${!node_rank_variable}" != "0" ]; then
        echo "Speed trial requires node rank 0; ${node_rank_variable}=${!node_rank_variable}." >&2
        exit 2
    fi
done
for local_size_variable in NPROC_PER_NODE nproc_per_node; do
    if [ -n "${!local_size_variable-}" ] && [ "${!local_size_variable}" != "8" ]; then
        echo "Speed trial requires 8 local ranks; ${local_size_variable}=${!local_size_variable}." >&2
        exit 2
    fi
done
master_port="${MASTER_PORT:-29500}"
if [[ ! "${master_port}" =~ ^[1-9][0-9]*$ ]] \
    || [ "${master_port}" -gt 65535 ]; then
    echo "MASTER_PORT must be an integer in 1..65535." >&2
    exit 2
fi

export NNODES=1
export nnodes=1
export NODE_RANK=0
export node_rank=0
export NPROC_PER_NODE=8
export nproc_per_node=8
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${master_port}"
export master_addr="${MASTER_ADDR}"
export master_port="${MASTER_PORT}"

official_root="/mnt/project_abd/public_databases/protenix.20260101"
benchmark_dataset_name="benchmark_long_protein_context_384_800_msa2048"
audited_dataset_repo="${repo_dir%/*}/Protenix_train2_fsdp2_msa2048_offload"
default_dataset_dir="${audited_dataset_repo}/data/benchmark_long_protein_context_1024_800_msa2048"
benchmark_dataset_dir="${BENCHMARK_DATASET_DIR:-${default_dataset_dir}}"

export RUN_NAME="${resolved_run_name}"
export run_name="${resolved_run_name}"
export BASE_DIR="${resolved_base_dir}"
export base_dir="${resolved_base_dir}"
export PROTENIX_ROOT_DIR="${official_root}"

# A set-but-empty lower-case value suppresses finetune_demo.sh's pretrained
# checkpoint fallback. The runner then retains its empty load-path defaults.
export CHECKPOINT_PATH=
export checkpoint_path=
export LOAD_CHECKPOINT_PATH=
export load_checkpoint_path=
export LOAD_EMA_CHECKPOINT_PATH=
export load_ema_checkpoint_path=
export EXPECTED_CHECKPOINT_SHA256=

export BENCHMARK_VARIANT="ddp_speed_${speed_profile}"
export BENCHMARK_WORKLOAD_PROFILE="crop384_db48_ddp_speed_trial"
export BENCHMARK_EXACT_CROP=true
export benchmark_exact_crop=true
export BENCHMARK_CROP_SIZE=384
export BENCHMARK_TRAIN_CROP_SIZE=384
export BENCHMARK_DATASET_NAME="${benchmark_dataset_name}"
export BENCHMARK_DATASET_DIR="${benchmark_dataset_dir}"
export BENCHMARK_MIN_NUM_TOKENS=1025
export BENCHMARK_RANDOM_SAMPLE_IF_FAILED=false
export BENCHMARK_CROP_METHOD_WEIGHTS="0,1,0"
export BENCHMARK_MAX_NUC_TOKENS=128
export BENCHMARK_WARMUP_STEPS="${resolved_warmup_steps}"
export benchmark_warmup_steps="${resolved_warmup_steps}"
export BENCHMARK_MEASURE_STEPS="${resolved_measure_steps}"
export benchmark_measure_steps="${resolved_measure_steps}"
export PROTENIX_BENCHMARK_PHASE_MEMORY=false

export MODEL_NAME=protenix-v2
export model_name=protenix-v2
export MODEL_C_Z=256
export model_c_z=256
export MODEL_C_M=128
export model_c_m=128
export train_crop_size=384
export diffusion_batch_size=48
export DIFFUSION_CHUNK_SIZE="${resolved_diffusion_chunk}"
export diffusion_chunk_size="${resolved_diffusion_chunk}"
export dtype=bf16
export N_cycle=4
export N_step=20
export seed=42
export iters_to_accumulate=1
export lr=0.001
export warmup_steps=2000
export MAX_STEPS="${resolved_max_steps}"
export max_steps="${resolved_max_steps}"
export DISTRIBUTED_STRATEGY=ddp
export distributed_strategy=ddp
export OPTIMIZER_SHARDING=none
export optimizer_sharding=none
export EMA_SHADOW_DEVICE="${resolved_ema_device}"
export ema_shadow_device="${resolved_ema_device}"
export ema_decay=0.999
export triangle_attention=cuequivariance
export triangle_multiplicative=cuequivariance

export train_sets="${benchmark_dataset_name}"
export train_sample_weights=1.0
export TRAIN_SAMPLER_SEED=77
export train_sampler_seed=77
export TRAIN_SAMPLER_REPLACEMENT=false
export train_sampler_replacement=false
export EPOCH_SIZE=800
export epoch_size=800
export NUM_DL_WORKERS=8
export num_dl_workers=8
export PROTENIX_MP_SHARING_STRATEGY=file_descriptor
export MSA_MIN_SIZE_TRAIN=2048
export msa_min_size_train=2048
export MSA_SAMPLE_CUTOFF_TRAIN=2048
export msa_sample_cutoff_train=2048
export msa_max_size_train=16384
export MSA_CHUNK_SIZE=2048
export msa_chunk_size=2048
export PAIR_MSA_AS_UNPAIRED_ONLY=false
export pair_msa_as_unpaired_only=false
export SHUFFLE_PAIR_MSA_WHEN_UNPAIRED_ONLY=false
export shuffle_pair_msa_when_unpaired_only=false

# Prevent inherited activation settings from being appended ahead of the
# profile's final explicit values.
unset PAIRFORMER_CHECKPOINT_CPU_OFFLOAD_EVERY_N
unset pairformer_checkpoint_cpu_offload_every_n
unset MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N
unset msa_checkpoint_cpu_offload_every_n
export PAIRFORMER_BLOCKS_PER_CKPT="${resolved_pairformer_blocks}"
export pairformer_blocks_per_ckpt="${resolved_pairformer_blocks}"
export PAIR_TRANSITION_CHUNK_SIZE="${resolved_transition_chunk}"
export pair_transition_chunk_size="${resolved_transition_chunk}"
export PAIR_TRANSITION_CHECKPOINT_CHUNKS="${resolved_transition_checkpoint}"
export pair_transition_checkpoint_chunks="${resolved_transition_checkpoint}"

export SKIP_FINAL_CHECKPOINT=true
export skip_final_checkpoint=true
export checkpoint_interval=-1
export log_interval=10
export empty_cache_interval=0
export use_wandb=false
export project=protenix
export eval_interval=0
export EVAL_AT_END=false
export eval_at_end=false
export EVAL_ONLY=false
export eval_only=false
export eval_ema_only=false

# Speed trials use the identical fail-closed toolchain as production so a
# result can never be labeled cu126 after compiling an extension with cu124.
# shellcheck disable=SC1091
source "${repo_dir}/scripts/runtime/cuda126_env.sh"

runtime_key="${resolved_run_name}_${master_port}"
runtime_tag="$(printf '%s' "${runtime_key}" | sha256sum | cut -c1-12)"
# Reuse only a cache with the same toolchain and LayerNorm source fingerprint.
default_triton_cache="/tmp/protenix_triton_cache/crop384_db48_ddp_fast"
default_torch_extensions="/tmp/protenix_torch_extensions/crop384_db48_ddp_fast/${PROTENIX_LAYER_NORM_CACHE_FINGERPRINT}"
export PROTENIX_TRITON_CACHE_ROOT="${PROTENIX_TRITON_CACHE_ROOT:-${default_triton_cache}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${default_torch_extensions}}"
runtime_tmpdir="${PROTENIX_RUNTIME_TMPDIR:-/tmp/ptx_speed_${runtime_tag}}"
mkdir -p "${runtime_tmpdir}" "${TORCH_EXTENSIONS_DIR}"
export TMPDIR="${runtime_tmpdir}"
export TEMP="${runtime_tmpdir}"
export TMP="${runtime_tmpdir}"

echo "Locked crop384/db48 DDP speed trial:"
echo "  profile=${speed_profile} trial_id=${trial_id}"
echo "  initialization=random raw_checkpoint=<empty> ema_checkpoint=<empty>"
echo "  topology=single_node/8_rank master=${MASTER_ADDR}:${MASTER_PORT}"
echo "  run=${RUN_NAME} base_dir=${BASE_DIR}"
echo "  benchmark=${resolved_warmup_steps}_warmup+${resolved_measure_steps}_measured"
echo "  max_steps=${resolved_max_steps} screen=${speed_screen} nuc_stress=${speed_nuc_stress} phase_memory=false"
if [ "${speed_nuc_stress}" = "1" ]; then
    echo "  nuc_stress_target=7d1a rank=5 human_step=31 zero_based_step=30 sampler_seed=77"
fi
echo "  runner_artifact=benchmark/benchmark_result.json under the timestamped run directory"
echo "  dataset=${benchmark_dataset_name} source_dir=${BENCHMARK_DATASET_DIR}"
echo "  crop=384 exact_crop=true nuc128 paired_msa=true diffusion_batch=48"
echo "  MSA depth/compute/chunk=2048/2048/2048 pool=16384"
echo "  ddp.gradient_as_bucket_view=${resolved_bucket_view}"
echo "  Adam use_adamw=false fused=${resolved_adam_fused} EMA_device=${resolved_ema_device}"
echo "  enable_diffusion_shared_vars_cache=${resolved_shared_vars_cache}"
echo "  diffusion_chunk=${resolved_diffusion_chunk}"
echo "  activation_groups PF/MSA/Diff=${resolved_pairformer_blocks}/${resolved_msa_blocks}/${resolved_diffusion_blocks}"
echo "  diffusion_fine_checkpoint=${resolved_fine_checkpoint}"
echo "  checkpoint_input_offload PF/MSA=${resolved_pairformer_offload}/${resolved_msa_offload}"
echo "  transition chunk/checkpoint=${resolved_transition_chunk}/${resolved_transition_checkpoint}"
echo "  checkpoint_write=false eval=false lr/warmup=0.001/2000"

exec bash "${repo_dir}/finetune_submit.sh" \
    --model_name protenix-v2 \
    --train_crop_size 384 \
    --diffusion_batch_size 48 \
    --diffusion_chunk_size "${resolved_diffusion_chunk}" \
    --distributed_strategy ddp \
    --ddp.gradient_as_bucket_view "${resolved_bucket_view}" \
    --optimizer_sharding none \
    --max_steps "${resolved_max_steps}" \
    --warmup_steps 2000 \
    --lr 0.001 \
    --lr_scheduler af3 \
    --decay_every_n_steps 50000 \
    --grad_clip_norm 10 \
    --adam.beta1 0.9 \
    --adam.beta2 0.95 \
    --adam.weight_decay 0.00000001 \
    --adam.use_adamw false \
    --adam.fused "${resolved_adam_fused}" \
    --ema_decay 0.999 \
    --ema_shadow_device "${resolved_ema_device}" \
    --model.N_cycle 4 \
    --sample_diffusion.N_step 20 \
    --iters_to_accumulate 1 \
    --data.train_sets "${benchmark_dataset_name}" \
    --data.train_sampler.train_sample_weights 1.0 \
    --data.train_sampler.seed 77 \
    --data.train_sampler.replacement false \
    --data.epoch_size 800 \
    --data.msa.min_size.train 2048 \
    --data.msa.max_size.train 16384 \
    --data.msa.sample_cutoff.train 2048 \
    --data.msa.pair_msa_as_unpaired_only false \
    --data.msa.shuffle_pair_msa_when_unpaired_only false \
    --model.msa_module.msa_chunk_size 2048 \
    --model.msa_module.msa_max_size 2048 \
    --model.pairformer.blocks_per_ckpt "${resolved_pairformer_blocks}" \
    --model.msa_module.blocks_per_ckpt "${resolved_msa_blocks}" \
    --model.diffusion_module.blocks_per_ckpt "${resolved_diffusion_blocks}" \
    --model.diffusion_module.use_fine_grained_checkpoint "${resolved_fine_checkpoint}" \
    --model.pairformer.checkpoint_cpu_offload_every_n "${resolved_pairformer_offload}" \
    --model.msa_module.checkpoint_cpu_offload_every_n "${resolved_msa_offload}" \
    --model.pairformer.train_pair_transition_chunk_size "${resolved_transition_chunk}" \
    --model.diffusion_module.train_pair_transition_chunk_size "${resolved_transition_chunk}" \
    --model.pairformer.train_pair_transition_checkpoint_chunks "${resolved_transition_checkpoint}" \
    --model.diffusion_module.train_pair_transition_checkpoint_chunks "${resolved_transition_checkpoint}" \
    --enable_diffusion_shared_vars_cache "${resolved_shared_vars_cache}" \
    --loss.weight.alpha_confidence 0.0001 \
    --loss.weight.alpha_pae 0 \
    --loss.weight.alpha_except_pae 1 \
    --loss.weight.alpha_diffusion 4 \
    --loss.weight.alpha_distogram 0.03 \
    --loss.weight.alpha_bond 0 \
    --loss.weight.smooth_lddt 1 \
    --train_confidence_only false \
    --max_atoms_per_token 24 \
    --enable_tf32 false \
    --benchmark_warmup_steps "${resolved_warmup_steps}" \
    --benchmark_measure_steps "${resolved_measure_steps}" \
    --checkpoint_interval -1 \
    --skip_final_checkpoint true \
    --eval_interval 0 \
    --eval_at_end false
