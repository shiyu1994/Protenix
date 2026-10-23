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

strict_resume_internal="${PROTENIX_CROP384_STRICT_RESUME_INTERNAL:-0}"
warmstart_internal="${PROTENIX_CROP384_WARMSTART_INTERNAL:-0}"
if [ "${strict_resume_internal}" != "0" ] && [ "${strict_resume_internal}" != "1" ]; then
    echo "PROTENIX_CROP384_STRICT_RESUME_INTERNAL must be 0 or 1." >&2
    exit 2
fi
if [ "${warmstart_internal}" != "0" ] && [ "${warmstart_internal}" != "1" ]; then
    echo "PROTENIX_CROP384_WARMSTART_INTERNAL must be 0 or 1." >&2
    exit 2
fi
if [ "${strict_resume_internal}" = "1" ] && [ "${warmstart_internal}" = "1" ]; then
    echo "Strict resume and params-only warm-start modes are mutually exclusive." >&2
    exit 2
fi

# Direct use of this entry point is intentionally random initialization. The
# dedicated strict-resume wrapper is the only supported caller that may set the
# internal mode and checkpoint pair.
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
    PROTENIX_SOFT_CLUSTER_SCHEDULE_DIR
    PROTENIX_SOFT_CLUSTER_TARGET_TRAINING_CONTRACT_ID
    BENCHMARK_LOAD_CHECKPOINT_PATH
    BENCHMARK_LOAD_EMA_CHECKPOINT_PATH
    EXPECTED_CHECKPOINT_SHA256
)
if [ "${strict_resume_internal}" = "0" ] && [ "${warmstart_internal}" = "0" ]; then
    for variable_name in "${checkpoint_source_env_vars[@]}"; do
        if [ -n "${!variable_name-}" ]; then
            echo "Random initialization rejects non-empty checkpoint environment variable ${variable_name}." >&2
            exit 2
        fi
    done
    if [ "${PROTENIX_FULL_RESUME:-0}" != "0" ]; then
        echo "Random initialization rejects PROTENIX_FULL_RESUME=${PROTENIX_FULL_RESUME}." >&2
        exit 2
    fi
    if [ "${PROTENIX_STRICT_FULL_RESUME:-0}" != "0" ]; then
        echo "Random initialization rejects PROTENIX_STRICT_FULL_RESUME." >&2
        exit 2
    fi
    if [ "${PROTENIX_RESUME_SMOKE_STEPS:-0}" != "0" ]; then
        echo "Random initialization rejects PROTENIX_RESUME_SMOKE_STEPS." >&2
        exit 2
    fi
elif [ "${strict_resume_internal}" = "1" ]; then
    if [ "${PROTENIX_FULL_RESUME:-0}" != "1" ] \
        || [ "${PROTENIX_STRICT_FULL_RESUME:-0}" != "1" ]; then
        echo "Internal resume mode requires both strict/full resume flags." >&2
        exit 2
    fi
    for required_resume_variable in \
        PROTENIX_RESUME_RAW_CHECKPOINT \
        PROTENIX_RESUME_EMA_CHECKPOINT \
        PROTENIX_RESUME_TARGET_MAX_STEPS; do
        if [ -z "${!required_resume_variable-}" ]; then
            echo "Internal resume mode requires ${required_resume_variable}." >&2
            exit 2
        fi
    done
else
    if [ "${PROTENIX_FULL_RESUME:-0}" != "0" ] \
        || [ "${PROTENIX_STRICT_FULL_RESUME:-0}" != "0" ]; then
        echo "Params-only warm-start rejects strict/full resume flags." >&2
        exit 2
    fi
    for required_warmstart_variable in \
        PROTENIX_WARMSTART_RAW_CHECKPOINT \
        PROTENIX_WARMSTART_EMA_CHECKPOINT \
        PROTENIX_WARMSTART_CHECKPOINT_STEP \
        PROTENIX_WARMSTART_RAW_SHA256 \
        PROTENIX_WARMSTART_EMA_SHA256 \
        PROTENIX_WARMSTART_SOURCE_TRAINING_CONTRACT_ID \
        PROTENIX_WARMSTART_SOURCE_DATA_CONTRACT_SHA256 \
        PROTENIX_WARMSTART_SOURCE_MODEL_ARCHITECTURE_SHA256 \
        PROTENIX_WARMSTART_SOURCE_CHECKPOINT_PAIR_ID; do
        if [ -z "${!required_warmstart_variable-}" ]; then
            echo "Internal warm-start mode requires ${required_warmstart_variable}." >&2
            exit 2
        fi
    done
    if [[ ! "${PROTENIX_WARMSTART_CHECKPOINT_STEP}" =~ ^[0-9]+$ ]]; then
        echo "PROTENIX_WARMSTART_CHECKPOINT_STEP must be a non-negative integer." >&2
        exit 2
    fi
fi

# A locked production launcher does not accept scientific command-line
# overrides. Keep a checkpoint-specific error so accidental resume commands are
# immediately recognizable.
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
            echo "Locked training rejects checkpoint/resume command-line parameter ${argument}." >&2
            exit 2
            ;;
    esac
done
if [ "$#" -ne 0 ]; then
    echo "This locked scratch launcher does not accept command-line overrides: $*" >&2
    exit 2
fi

resolved_run_name="${RUN_NAME:-${run_name:-}}"
if [ -z "${resolved_run_name}" ]; then
    echo "RUN_NAME is required for formal scratch training." >&2
    exit 2
fi
if [[ ! "${resolved_run_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
fi
resolved_base_dir="${BASE_DIR:-${base_dir:-/mnt/project_abd/klp/proteinx_output}}"

independent_pairing_non_pairing_budget="${INDEPENDENT_PAIRING_NON_PAIRING_BUDGET:-false}"
if [ "${independent_pairing_non_pairing_budget}" != "false" ] \
    && [ "${independent_pairing_non_pairing_budget}" != "true" ]; then
    echo "INDEPENDENT_PAIRING_NON_PAIRING_BUDGET must be true or false." >&2
    exit 2
fi

if [ "${strict_resume_internal}" = "1" ]; then
    if [ "${PROTENIX_STRAGGLER_BENCHMARK_UPDATES:-0}" != "0" ]; then
        echo "Strict resume rejects PROTENIX_STRAGGLER_BENCHMARK_UPDATES." >&2
        exit 2
    fi
    resolved_launch_mode=strict_full_resume
    resume_smoke_steps="${PROTENIX_RESUME_SMOKE_STEPS:-0}"
    if [[ ! "${resume_smoke_steps}" =~ ^[0-9]+$ ]]; then
        echo "PROTENIX_RESUME_SMOKE_STEPS must be a non-negative integer." >&2
        exit 2
    fi
    resolved_max_steps="${PROTENIX_RESUME_TARGET_MAX_STEPS}"
    if [[ ! "${resolved_max_steps}" =~ ^[1-9][0-9]*$ ]]; then
        echo "PROTENIX_RESUME_TARGET_MAX_STEPS must be a positive integer." >&2
        exit 2
    fi
    if [ "${resume_smoke_steps}" = "0" ]; then
        resolved_checkpoint_interval=50
        resolved_skip_final_checkpoint=false
    else
        resolved_checkpoint_interval=-1
        resolved_skip_final_checkpoint=true
    fi
else
    smoke_test="${PROTENIX_SMOKE_TEST:-0}"
    smoke_updates="${PROTENIX_SMOKE_UPDATES:-1}"
    checkpoint_source_smoke="${PROTENIX_CHECKPOINT_SOURCE_SMOKE:-0}"
    straggler_benchmark_updates="${PROTENIX_STRAGGLER_BENCHMARK_UPDATES:-0}"
    if [ "${warmstart_internal}" = "1" ] \
        && { [ "${checkpoint_source_smoke}" != "0" ] \
            || [ "${straggler_benchmark_updates}" != "0" ]; }; then
        echo "Params-only warm-start supports only production or PROTENIX_SMOKE_TEST=1." >&2
        exit 2
    fi
    if [ "${smoke_test}" != "0" ] && [ "${smoke_test}" != "1" ]; then
        echo "PROTENIX_SMOKE_TEST must be 0 or 1." >&2
        exit 2
    fi
    if [ "${checkpoint_source_smoke}" != "0" ] \
        && [ "${checkpoint_source_smoke}" != "1" ]; then
        echo "PROTENIX_CHECKPOINT_SOURCE_SMOKE must be 0 or 1." >&2
        exit 2
    fi
    if [ "${straggler_benchmark_updates}" != "0" ] \
        && [ "${straggler_benchmark_updates}" != "100" ] \
        && [ "${straggler_benchmark_updates}" != "250" ]; then
        echo "PROTENIX_STRAGGLER_BENCHMARK_UPDATES must be 0, 100, or 250." >&2
        exit 2
    fi
    enabled_special_modes=0
    if [ "${smoke_test}" = "1" ]; then
        enabled_special_modes=$((enabled_special_modes + 1))
    fi
    if [ "${checkpoint_source_smoke}" = "1" ]; then
        enabled_special_modes=$((enabled_special_modes + 1))
    fi
    if [ "${straggler_benchmark_updates}" != "0" ]; then
        enabled_special_modes=$((enabled_special_modes + 1))
    fi
    if (( enabled_special_modes > 1 )); then
        echo "PROTENIX_SMOKE_TEST, PROTENIX_CHECKPOINT_SOURCE_SMOKE, and PROTENIX_STRAGGLER_BENCHMARK_UPDATES are mutually exclusive." >&2
        exit 2
    fi
    if [[ ! "${smoke_updates}" =~ ^[1-9][0-9]*$ ]] \
        || (( smoke_updates > 100 )); then
        echo "PROTENIX_SMOKE_UPDATES must be an integer in [1, 100]." >&2
        exit 2
    fi
    if [ "${smoke_test}" = "0" ] && [ "${smoke_updates}" != "1" ]; then
        echo "PROTENIX_SMOKE_UPDATES requires PROTENIX_SMOKE_TEST=1." >&2
        exit 2
    fi
    if [ "${checkpoint_source_smoke}" = "1" ]; then
        resolved_launch_mode=checkpoint_source_smoke
        resolved_max_steps=1
        resolved_checkpoint_interval=1
        resolved_skip_final_checkpoint=false
    elif [ "${straggler_benchmark_updates}" != "0" ]; then
        for max_steps_variable in MAX_STEPS max_steps; do
            if [ -n "${!max_steps_variable-}" ] \
                && [ "${!max_steps_variable}" != "${straggler_benchmark_updates}" ]; then
                echo "v9 straggler benchmark locks ${max_steps_variable}=${straggler_benchmark_updates}." >&2
                exit 2
            fi
        done
        resolved_launch_mode=v9_straggler_benchmark_no_checkpoint
        resolved_max_steps="${straggler_benchmark_updates}"
        resolved_checkpoint_interval=-1
        resolved_skip_final_checkpoint=true
    elif [ "${smoke_test}" = "1" ]; then
        if [ "${warmstart_internal}" = "1" ]; then
            resolved_launch_mode=params_only_warmstart_smoke
        else
            resolved_launch_mode=smoke_no_checkpoint
        fi
        resolved_max_steps="${smoke_updates}"
        resolved_checkpoint_interval=-1
        resolved_skip_final_checkpoint=true
    else
        if [ "${warmstart_internal}" = "1" ]; then
            resolved_launch_mode=params_only_warmstart
        else
            resolved_launch_mode=production
        fi
        for max_steps_variable in MAX_STEPS max_steps; do
            if [ -n "${!max_steps_variable-}" ] && [ "${!max_steps_variable}" != "50000" ]; then
                echo "Scratch production locks ${max_steps_variable}=50000; unset it or use PROTENIX_SMOKE_TEST=1." >&2
                exit 2
            fi
        done
        resolved_max_steps=50000
        resolved_checkpoint_interval=200
        resolved_skip_final_checkpoint=false
    fi
fi

# One rank consumes one target per microstep. Require exactly 128 ranks so four
# accumulated microsteps always form an effective target batch of 512.
# AIStudio normally supplies this as 16 workers with 8 H100s per worker.
expected_nnodes=16
expected_nproc_per_node=8
expected_world_size=128
resolved_nnodes="${NNODES:-${MLP_WORKER_NUM:-${SLURM_NNODES:-${nnodes:-${WORLD_SIZE:-}}}}}"
resolved_nproc_per_node="${NPROC_PER_NODE:-${nproc_per_node:-8}}"
if [[ ! "${resolved_nnodes}" =~ ^[1-9][0-9]*$ ]]; then
    echo "The 128-GPU contract requires NNODES or AIStudio MLP_WORKER_NUM." >&2
    exit 2
fi
if [[ ! "${resolved_nproc_per_node}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer." >&2
    exit 2
fi
for topology_pair in \
    "MLP_WORKER_NUM:${MLP_WORKER_NUM:-}" \
    "SLURM_NNODES:${SLURM_NNODES:-}"; do
    topology_name="${topology_pair%%:*}"
    topology_value="${topology_pair#*:}"
    if [ -n "${topology_value}" ] \
        && [ "${topology_value}" != "${resolved_nnodes}" ]; then
        echo "${topology_name}=${topology_value} disagrees with NNODES=${resolved_nnodes}." >&2
        exit 2
    fi
done
resolved_world_size=$((resolved_nnodes * resolved_nproc_per_node))
if (( resolved_nnodes != expected_nnodes \
    || resolved_nproc_per_node != expected_nproc_per_node \
    || resolved_world_size != expected_world_size )); then
    echo "This launcher requires topology=16x8 and world_size=128; resolved ${resolved_nnodes}x${resolved_nproc_per_node}=${resolved_world_size}." >&2
    exit 2
fi

if (( resolved_nnodes == 1 )); then
    resolved_node_rank="${NODE_RANK:-${node_rank:-0}}"
    resolved_master_addr="${MASTER_ADDR:-${master_addr:-127.0.0.1}}"
else
    resolved_node_rank="${NODE_RANK:-${MLP_ROLE_INDEX:-${SLURM_NODEID:-${node_rank:-${RANK:-}}}}}"
    resolved_master_addr="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-${master_addr:-}}}"
    if [[ ! "${resolved_node_rank}" =~ ^[0-9]+$ ]] \
        || (( resolved_node_rank >= resolved_nnodes )); then
        echo "Multi-node training requires a valid NODE_RANK or AIStudio MLP_ROLE_INDEX." >&2
        exit 2
    fi
    if [ -z "${resolved_master_addr}" ] \
        || [ "${resolved_master_addr}" = "127.0.0.1" ]; then
        echo "Multi-node training requires MASTER_ADDR or AIStudio MLP_WORKER_0_HOST." >&2
        exit 2
    fi
fi
if [ -n "${MLP_ROLE_INDEX:-}" ] \
    && [ "${MLP_ROLE_INDEX}" != "${resolved_node_rank}" ]; then
    echo "MLP_ROLE_INDEX=${MLP_ROLE_INDEX} disagrees with NODE_RANK=${resolved_node_rank}." >&2
    exit 2
fi
if [ -n "${MLP_WORKER_0_HOST:-}" ] \
    && [ "${MLP_WORKER_0_HOST}" != "${resolved_master_addr}" ]; then
    echo "MLP_WORKER_0_HOST disagrees with MASTER_ADDR." >&2
    exit 2
fi
if [[ ! "${resolved_node_rank}" =~ ^[0-9]+$ ]] \
    || (( resolved_node_rank >= resolved_nnodes )); then
    echo "Resolved node rank is outside the active topology." >&2
    exit 2
fi
resolved_master_port="${MASTER_PORT:-${MLP_WORKER_0_PORT:-${master_port:-29500}}}"
if [ -n "${MLP_WORKER_0_PORT:-}" ] \
    && [ "${MLP_WORKER_0_PORT}" != "${resolved_master_port}" ]; then
    echo "MLP_WORKER_0_PORT disagrees with MASTER_PORT." >&2
    exit 2
fi
if [[ ! "${resolved_master_port}" =~ ^[1-9][0-9]*$ ]] \
    || (( resolved_master_port > 65535 )); then
    echo "MASTER_PORT must be in [1, 65535]." >&2
    exit 2
fi

export NNODES="${resolved_nnodes}"
export nnodes="${resolved_nnodes}"
export NODE_RANK="${resolved_node_rank}"
export node_rank="${resolved_node_rank}"
export NPROC_PER_NODE="${resolved_nproc_per_node}"
export nproc_per_node="${resolved_nproc_per_node}"
export MASTER_ADDR="${resolved_master_addr}"
export master_addr="${resolved_master_addr}"
export MASTER_PORT="${resolved_master_port}"
export master_port="${resolved_master_port}"
export PROTENIX_EXPECTED_WORLD_SIZE="${expected_world_size}"
export PROTENIX_EXPECTED_LOCAL_WORLD_SIZE="${expected_nproc_per_node}"
export PROTENIX_EXPECTED_NUM_NODES="${expected_nnodes}"

official_root="/mnt/project_abd/public_databases/protenix.20260101"
weighted_set="weightedPDB_before210930_v20260101"
data_profile_registry="${repo_dir}/configs/crop384_db48_data_profiles.sh"
# shellcheck disable=SC1090
source "${data_profile_registry}"
locked_data_profile="${PROTENIX_LOCKED_DATA_PROFILE:-weighted_pdb_only_v1}"
if [ -n "${PROTENIX_TRAINING_DATA_PROFILE:-}" ] \
    && [ "${PROTENIX_TRAINING_DATA_PROFILE}" != "${locked_data_profile}" ]; then
    echo "This launcher locks PROTENIX_TRAINING_DATA_PROFILE=${locked_data_profile}; got ${PROTENIX_TRAINING_DATA_PROFILE}." >&2
    exit 2
fi
resolve_crop384_db48_data_profile "${locked_data_profile}"
export PROTENIX_TRAINING_DATA_PROFILE="${CROP384_DATA_PROFILE_NAME}"
export PROTENIX_EXTRA_TRAIN_DATA_REQUIRED_PATHS="${CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS}"

export RUN_NAME="${resolved_run_name}"
export run_name="${resolved_run_name}"
export BASE_DIR="${resolved_base_dir}"
export base_dir="${resolved_base_dir}"
export PROTENIX_ROOT_DIR="${official_root}"
soft_cluster_schedule_dir="${PROTENIX_SOFT_CLUSTER_SCHEDULE_DIR:-}"
soft_cluster_target_contract="${PROTENIX_SOFT_CLUSTER_TARGET_TRAINING_CONTRACT_ID:-}"
if [ -n "${soft_cluster_schedule_dir}" ]; then
    if [ "${strict_resume_internal}" != "1" ]; then
        echo "Soft-cluster schedule is only supported by strict full-state resume." >&2
        exit 2
    fi
    if [ -z "${soft_cluster_target_contract}" ]; then
        echo "Soft-cluster schedule requires PROTENIX_SOFT_CLUSTER_TARGET_TRAINING_CONTRACT_ID." >&2
        exit 2
    fi
elif [ -n "${soft_cluster_target_contract}" ]; then
    echo "PROTENIX_SOFT_CLUSTER_TARGET_TRAINING_CONTRACT_ID requires PROTENIX_SOFT_CLUSTER_SCHEDULE_DIR." >&2
    exit 2
fi
default_training_contract_id="crop384_db48_ddp_scratch_50k_ws128_acc4_data_${CROP384_DATA_PROFILE_CONTRACT_TAG}_arch_pf64_cs384_dct768_pfckpt8_c48_msamix_${CROP384_DATA_PROFILE_MSA_MIX_CONTRACT_TAG}_msaraw_${CROP384_DATA_PROFILE_MSA_RAW_CONTRACT_TAG}_msaquerypinle1024_templatecache_top60u20id90_chaindummy_ckpt200_dlpersist1_prefetch2_deferredhealth_v11"
export PROTENIX_TRAINING_CONTRACT_ID="${soft_cluster_target_contract:-${default_training_contract_id}}"

# Set both lower-case variables because finetune_demo.sh treats a set lower-case
# value as authoritative. Scratch mode uses empty values to bypass the official
# fine-tuning checkpoint; strict mode receives its validated raw/EMA pair from
# the dedicated wrapper.
if [ "${strict_resume_internal}" = "1" ]; then
    export CHECKPOINT_PATH="${PROTENIX_RESUME_RAW_CHECKPOINT}"
    export checkpoint_path="${PROTENIX_RESUME_RAW_CHECKPOINT}"
    export LOAD_CHECKPOINT_PATH="${PROTENIX_RESUME_RAW_CHECKPOINT}"
    export load_checkpoint_path="${PROTENIX_RESUME_RAW_CHECKPOINT}"
    export LOAD_EMA_CHECKPOINT_PATH="${PROTENIX_RESUME_EMA_CHECKPOINT}"
    export load_ema_checkpoint_path="${PROTENIX_RESUME_EMA_CHECKPOINT}"
elif [ "${warmstart_internal}" = "1" ]; then
    export CHECKPOINT_PATH="${PROTENIX_WARMSTART_RAW_CHECKPOINT}"
    export checkpoint_path="${PROTENIX_WARMSTART_RAW_CHECKPOINT}"
    export LOAD_CHECKPOINT_PATH="${PROTENIX_WARMSTART_RAW_CHECKPOINT}"
    export load_checkpoint_path="${PROTENIX_WARMSTART_RAW_CHECKPOINT}"
    export LOAD_EMA_CHECKPOINT_PATH="${PROTENIX_WARMSTART_EMA_CHECKPOINT}"
    export load_ema_checkpoint_path="${PROTENIX_WARMSTART_EMA_CHECKPOINT}"
else
    export CHECKPOINT_PATH=
    export checkpoint_path=
    export LOAD_CHECKPOINT_PATH=
    export load_checkpoint_path=
    export LOAD_EMA_CHECKPOINT_PATH=
    export load_ema_checkpoint_path=
fi
export EXPECTED_CHECKPOINT_SHA256=

# The selected immutable profile owns the ordered datasets, their outer masses,
# and any additional runtime-index requirements. Dedicated wrappers select
# non-default profiles without changing the historical PDB-only default.
export WEIGHTED_TRAIN_SET="${weighted_set}"
export weighted_train_set="${weighted_set}"
export WEIGHTED_MMCIF_DIR="${official_root}/mmcif"
export WEIGHTED_BIOASSEMBLY_DIR="${official_root}/mmcif_bioassembly"
export WEIGHTED_INDICES_FPATH="${official_root}/indices/indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz"
export POSEBUSTERS_MMCIF_DIR="${official_root}/posebusters_mmcif"
export POSEBUSTERS_BIOASSEMBLY_DIR="${official_root}/posebusters_bioassembly"
export POSEBUSTERS_INDICES_FPATH="${official_root}/indices/posebusters_indices_mainchain_interface.csv"
export train_sets="${CROP384_DATA_PROFILE_TRAIN_SETS}"
export train_sample_weights="${CROP384_DATA_PROFILE_TRAIN_SAMPLE_WEIGHTS}"
export TRAIN_SAMPLER_SEED=-1
export train_sampler_seed=-1
export TRAIN_SAMPLER_REPLACEMENT=true
export train_sampler_replacement=true
export EPOCH_SIZE=10240
export epoch_size=10240
export NUM_DL_WORKERS=8
export num_dl_workers=8
export PROTENIX_MP_SHARING_STRATEGY=file_descriptor

# Lock the initial-training scientific workload.
export MODEL_NAME=protenix-v2
export model_name=protenix-v2
export MODEL_C_Z=256
export model_c_z=256
export MODEL_C_M=128
export model_c_m=128
export train_crop_size="${TRAIN_CROP_SIZE:-384}"
export diffusion_batch_size=48
export DIFFUSION_CHUNK_SIZE=48
export diffusion_chunk_size=48
export dtype=bf16
export N_cycle=4
export N_step=20
export seed=42
export iters_to_accumulate=4
export lr=0.001
export warmup_steps=1000
export MAX_STEPS="${resolved_max_steps}"
export max_steps="${resolved_max_steps}"
export DISTRIBUTED_STRATEGY=ddp
export distributed_strategy=ddp
export OPTIMIZER_SHARDING=none
export optimizer_sharding=none
export EMA_SHADOW_DEVICE=model
export ema_shadow_device=model
export ema_decay=0.999
export triangle_attention=cuequivariance
export triangle_multiplicative=cuequivariance

# Match the 68999 crop640 checkpoint grouping (PF/MSA/Diff=4/1/1) and
# checkpointed 128-row pair transitions. Keep CPU offload, GPU EMA and
# diffusion shared-cache settings unchanged. Clear inherited offload values.
unset PAIRFORMER_CHECKPOINT_CPU_OFFLOAD_EVERY_N
unset pairformer_checkpoint_cpu_offload_every_n
unset MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N
unset msa_checkpoint_cpu_offload_every_n
export PAIRFORMER_BLOCKS_PER_CKPT=4
export pairformer_blocks_per_ckpt=4
export PAIR_TRANSITION_CHUNK_SIZE=128
export pair_transition_chunk_size=128
export PAIR_TRANSITION_CHECKPOINT_CHUNKS=true
export pair_transition_checkpoint_chunks=true

# The featurizer assembles a pool of at most 16,384 rows. Every recycle
# independently samples K~UniformInteger[1,min(actual depth,2,048)] and then
# selects K rows without replacement from the complete assembled pool. When
# K<=1,024, row zero is retained as the canonical global query by replacing
# one sampled slot rather than appending a row. The sampled features and OPM
# therefore still use exactly K rows; only selected MSAStack operations are
# temporarily zero-padded to 2,048 and run as one 2,048-row chunk.
export MSA_MIN_SIZE_TRAIN=1
export msa_min_size_train=1
export MSA_SAMPLE_CUTOFF_TRAIN=2048
export msa_sample_cutoff_train=2048
export MSA_QUERY_INCLUSION_MAX_SAMPLE_SIZE_TRAIN=1024
export MSA_QUERY_INCLUSION_MAX_SAMPLE_SIZE_TEST=1024
export msa_max_size_train=16384
export MSA_CHUNK_SIZE=2048
export msa_chunk_size=2048
# These are the configured/default and 50% normal-branch values. The 30%
# occurrence branch passes legacy true/true as local featurizer overrides.
export PAIR_MSA_AS_UNPAIRED_ONLY=false
export pair_msa_as_unpaired_only=false
export SHUFFLE_PAIR_MSA_WHEN_UNPAIRED_ONLY=false
export shuffle_pair_msa_when_unpaired_only=false

export BENCHMARK_EXACT_CROP=false
export benchmark_exact_crop=false
export BENCHMARK_WARMUP_STEPS=-1
export benchmark_warmup_steps=-1
export BENCHMARK_MEASURE_STEPS=-1
export benchmark_measure_steps=-1
export SKIP_FINAL_CHECKPOINT="${resolved_skip_final_checkpoint}"
export skip_final_checkpoint="${resolved_skip_final_checkpoint}"
export checkpoint_interval="${resolved_checkpoint_interval}"
export log_interval=25
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

# This production profile is tied to the audited cu126 wheel and the independent
# CUDA 12.6.85/GCC 12.4 toolchain. Never fall back to /usr/local/cuda or a
# compiler inherited from an AIStudio shell.
# shellcheck disable=SC1091
source "${repo_dir}/scripts/runtime/cuda126_env.sh"

node_rank_for_cache="${NODE_RANK:-${RANK:-${MLP_ROLE_INDEX:-0}}}"
runtime_tag="$(printf '%s' "${resolved_run_name}" | sha256sum | cut -c1-12)"
# Cache only across runs with the exact same CUDA/PyTorch toolchain and
# LayerNorm sources. The module also verifies an atomic provenance sidecar.
default_triton_cache="/tmp/protenix_triton_cache/crop384_db48_ddp_fast/node_${node_rank_for_cache}"
default_torch_extensions="/tmp/protenix_torch_extensions/crop384_db48_ddp_fast/${PROTENIX_LAYER_NORM_CACHE_FINGERPRINT}/node_${node_rank_for_cache}"
export PROTENIX_TRITON_CACHE_ROOT="${PROTENIX_TRITON_CACHE_ROOT:-${default_triton_cache}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${default_torch_extensions}}"
runtime_tmpdir="${PROTENIX_RUNTIME_TMPDIR:-/tmp/ptx_train_${runtime_tag}_${node_rank_for_cache}}"
mkdir -p "${runtime_tmpdir}" "${TORCH_EXTENSIONS_DIR}"
export TMPDIR="${runtime_tmpdir}"
export TEMP="${runtime_tmpdir}"
export TMP="${runtime_tmpdir}"

echo "Locked Protenix crop384/db48 DDP scratch profile:"
echo "  launch_mode=${resolved_launch_mode}"
if [ "${strict_resume_internal}" = "1" ]; then
    echo "  initialization=strict_full_resume"
    echo "  raw_checkpoint=${load_checkpoint_path}"
    echo "  ema_checkpoint=${load_ema_checkpoint_path}"
    echo "  checkpoint=${RESUME_STEP:-selected_by_path}"
    echo "  state=model+optimizer+scheduler+step+scaler+rank_rng+matched_ema"
    echo "  sampler=new schedule starts fresh once; matching schedule restores its saved cursor"
    echo "  data_worker_rng=not_checkpointed num_dl_workers=8"
elif [ "${warmstart_internal}" = "1" ]; then
    echo "  initialization=params_only_source_pair"
    echo "  raw_checkpoint=${load_checkpoint_path}"
    echo "  ema_checkpoint=${load_ema_checkpoint_path}"
    echo "  source_checkpoint_step=${PROTENIX_WARMSTART_CHECKPOINT_STEP}"
    echo "  imported=live_model:raw ema_shadow:ema"
    echo "  reset=optimizer+scheduler+step+scaler+rank_rng+sampler_cursor"
    echo "  target_start_step=0 target_start_global_step=0"
else
    echo "  initialization=random raw_checkpoint=<empty> ema_checkpoint=<empty>"
fi
echo "  run=${RUN_NAME} base_dir=${BASE_DIR} ending_checkpoint=$((resolved_max_steps - 1))"
echo "  model=protenix-v2 pairformer_blocks=64 c_s=384 diffusion_c_token=768"
echo "  parameters=529902655 trainable=529902065 diffusion_and_conditioning=unchanged"
echo "  crop=${train_crop_size} diffusion_batch/chunk=48/48 N_cycle/N_step=4/20"
echo "  distributed_strategy=ddp optimizer_sharding=none ddp_bucket_view=true EMA=model/0.999"
echo "  topology=${NNODES}x${NPROC_PER_NODE} world_size=${resolved_world_size}"
echo "  target_batch=128 targets/microstep x accumulation=4 = 512 targets/update"
echo "  ending_checkpoint=$((resolved_max_steps - 1))"
echo "  optimizer=Adam fused=true beta=0.9/0.95 weight_decay=1e-8 lr/warmup=0.001/1000 scheduler=af3 decay_every=25000 factor=0.95"
echo "  ddp_accumulation=first_backward_sync_once then steady_state_no_syncx3+sync"
echo "  data_profile=${PROTENIX_TRAINING_DATA_PROFILE} train_sets=${train_sets} weights=${train_sample_weights}"
if [ -n "${CROP384_DATA_PROFILE_RUNTIME_INDEX_DIR}" ]; then
    echo "  distillation_runtime_index=${CROP384_DATA_PROFILE_RUNTIME_INDEX_DIR}"
fi
echo "  contract=${PROTENIX_TRAINING_CONTRACT_ID}"
echo "  augmentation_gates=msa_categorical_per_target_occurrence template_independent decision=sample_occurrence_sha256_v1"
echo "  msa_mode_probabilities=query_only/unpaired_shuffle/species_paired=${CROP384_DATA_PROFILE_QUERY_ONLY_MSA_PROBABILITY}/${CROP384_DATA_PROFILE_PAIR_MSA_AS_UNPAIRED_PROBABILITY}/${CROP384_DATA_PROFILE_SPECIES_PAIRED_MSA_PROBABILITY}"
echo "  template_use_probability=0.2 template_none_probability=0.8 legacy_template_dropout_rate=0.0"
if [ -n "${PROTENIX_SEQUENCE_TEMPLATE_CACHE_ROOT:-}" ]; then
    echo "  template_cache=${PROTENIX_SEQUENCE_TEMPLATE_CACHE_ROOT}"
    echo "  template_cache_mode/error_policy=${PROTENIX_SEQUENCE_TEMPLATE_CACHE_MODE:-<unset>}/${PROTENIX_SEQUENCE_TEMPLATE_CACHE_ERROR_POLICY:-<unset>}"
fi
if [ -n "${soft_cluster_schedule_dir}" ]; then
    echo "  sampler=soft_cluster_schedule gamma=0.2 manifest=${soft_cluster_schedule_dir}"
    echo "  sampler_outer_and_weighted_children=legacy_with_replacement_preserved"
else
    echo "  sampler=with_replacement epoch_size=10240 local_microsteps/epoch=80 updates/epoch=20"
fi
echo "  train_dataloader=num_workers=8 persistent_workers=true prefetch_factor=2 pin_memory=false"
echo "  loss_health_sync=local_each_microstep global_all_reduce=once_at_update_boundary"
echo "  max_nuc_tokens=128"
if [ "${independent_pairing_non_pairing_budget}" = "true" ]; then
    echo "  MSA weightedPDB_sources=pairing.a3m_top10000+genuine_non_pairing.a3m_top10000 independent=true paired_block_dedup=false unpaired_merge_dedup=true"
else
    echo "  MSA weightedPDB_sources=legacy_shared_pairing_non_pairing_scan independent=false"
fi
if [ -n "${CROP384_DATA_PROFILE_DISTILL_UNPAIRED_MSA_DEPTH}" ]; then
    echo "  MSA raw_prefix_cap_train AFDB/MGnify short/long direct-unpaired=${CROP384_DATA_PROFILE_DISTILL_UNPAIRED_MSA_DEPTH} paired-source=10000(unused)"
fi
echo "  MSA assembled_pool<=16384 K~UniformInteger[1,min(actual_depth,2048)] base_draw=randperm(full_pool)"
echo "  MSA query_row_policy=pin_index0_when_K_le_1024 train/test=1024/1024 final_depth_remains_K"
echo "  MSA full_branch_features/OPM=K temporary_pad/chunk=2048/2048"
echo "  MSA query_only_branch=${CROP384_DATA_PROFILE_QUERY_ONLY_DESCRIPTION}"
echo "  MSA unpaired_shuffle_branch=${CROP384_DATA_PROFILE_MSA_UNPAIRED_DESCRIPTION}"
echo "  MSA mode_probabilities=query_only:${CROP384_DATA_PROFILE_QUERY_ONLY_MSA_PROBABILITY} unpaired_shuffle:${CROP384_DATA_PROFILE_PAIR_MSA_AS_UNPAIRED_PROBABILITY} species_paired:${CROP384_DATA_PROFILE_SPECIES_PAIRED_MSA_PROBABILITY}"
echo "  MSA species_paired_branch=false/false normal_protenix_flow"
echo "  speed_profile=PF64/PFckpt4+c48 shared_diffusion_cache=true activation_groups=4/1/1"
echo "  checkpoint_input_offload PF/MSA=null/null diffusion_fine_checkpoint=true"
echo "  transition_chunk/checkpoint=128/true"
if [ "${skip_final_checkpoint}" = false ]; then
    final_checkpoint=true
else
    final_checkpoint=false
fi
echo "  checkpoint_interval=${checkpoint_interval} final_checkpoint=${final_checkpoint}"
echo "  train_log_interval=25 inline_eval=false external_ema_eval_every=5000_updates"
echo "  cuda_host_compiler=${PROTENIX_CUDA_HOST_CC} / ${PROTENIX_CUDA_HOST_CXX}"

# Scratch leaves these arguments absent; the two checkpoint modes make every
# state-load choice explicit.
initialization_train_args=()
if [ "${strict_resume_internal}" = "1" ]; then
    # finetune_demo.sh materializes this option before appending passthrough
    # arguments. Set its input explicitly so the final command contains one
    # unambiguous full-state value instead of both true and false.
    export LOAD_PARAMS_ONLY=false
    initialization_train_args=(
        --load_checkpoint_path "${load_checkpoint_path}"
        --load_ema_checkpoint_path "${load_ema_checkpoint_path}"
        --load_strict true
        --skip_load_optimizer false
        --skip_load_scheduler false
        --skip_load_step false
        --load_step_for_scheduler false
    )
elif [ "${warmstart_internal}" = "1" ]; then
    export LOAD_PARAMS_ONLY=true
    initialization_train_args=(
        --load_strict true
        --skip_load_optimizer true
        --skip_load_scheduler true
        --skip_load_step true
        --load_step_for_scheduler false
    )
fi
data_profile_train_args=(
    --data.train_sets "${train_sets}"
    --data.train_sampler.train_sample_weights "${train_sample_weights}"
    "${CROP384_DATA_PROFILE_ARGS[@]}"
)
if [ -n "${soft_cluster_schedule_dir}" ]; then
    data_profile_train_args+=(
        --data.train_sampler.soft_cluster_schedule_manifest "${soft_cluster_schedule_dir}"
        --data.train_sampler.soft_cluster_gamma 0.2
    )
fi

exec bash "${repo_dir}/finetune_submit.sh" \
    "${initialization_train_args[@]}" \
    --model_name protenix-v2 \
    --n_blocks 64 \
    --train_crop_size "${train_crop_size}" \
    --diffusion_batch_size 48 \
    --diffusion_chunk_size 48 \
    --distributed_strategy ddp \
    --ddp.gradient_as_bucket_view true \
    --optimizer_sharding none \
    --max_steps "${resolved_max_steps}" \
    --warmup_steps 1000 \
    --lr 0.001 \
    --lr_scheduler af3 \
    --decay_every_n_steps 25000 \
    --af3_lr_scheduler.decay_factor 0.95 \
    --grad_clip_norm 10 \
    --adam.beta1 0.9 \
    --adam.beta2 0.95 \
    --adam.weight_decay 0.00000001 \
    --adam.use_adamw false \
    --adam.fused true \
    --ema_decay 0.999 \
    --ema_shadow_device model \
    --model.N_cycle 4 \
    --sample_diffusion.N_step 20 \
    --iters_to_accumulate 4 \
    --data.epoch_size 10240 \
    --data.num_dl_workers 8 \
    --data.train_dl_persistent_workers true \
    --data.train_dl_prefetch_factor 2 \
    "${data_profile_train_args[@]}" \
    --data.query_only_msa_probability "${CROP384_DATA_PROFILE_QUERY_ONLY_MSA_PROBABILITY}" \
    --data.pair_msa_as_unpaired_probability "${CROP384_DATA_PROFILE_PAIR_MSA_AS_UNPAIRED_PROBABILITY}" \
    --data.template_use_probability 0.2 \
    --data.augmentation_decision_version sample_occurrence_sha256_v1 \
    --data.template.template_dropout_rate 0.0 \
    --data.msa.min_size.train 1 \
    --data.msa.max_size.train 16384 \
    --data.msa.sample_cutoff.train 2048 \
    --data.msa.query_inclusion_max_sample_size.train "${MSA_QUERY_INCLUSION_MAX_SAMPLE_SIZE_TRAIN}" \
    --data.msa.query_inclusion_max_sample_size.test "${MSA_QUERY_INCLUSION_MAX_SAMPLE_SIZE_TEST}" \
    --data.msa.max_input_paired_msa_depth 10000 \
    --data.msa.max_input_unpaired_msa_depth 10000 \
    --data.msa.independent_pairing_non_pairing_budget "${independent_pairing_non_pairing_budget}" \
    --data.msa.pair_msa_as_unpaired_only false \
    --data.msa.shuffle_pair_msa_when_unpaired_only false \
    --model.msa_module.msa_chunk_size 2048 \
    --model.msa_module.msa_max_size 2048 \
    --model.pairformer.blocks_per_ckpt 4 \
    --model.msa_module.blocks_per_ckpt 1 \
    --model.diffusion_module.blocks_per_ckpt 1 \
    --model.diffusion_module.use_fine_grained_checkpoint true \
    --model.pairformer.checkpoint_cpu_offload_every_n null \
    --model.msa_module.checkpoint_cpu_offload_every_n null \
    --model.pairformer.train_pair_transition_chunk_size 128 \
    --model.diffusion_module.train_pair_transition_chunk_size 128 \
    --model.pairformer.train_pair_transition_checkpoint_chunks true \
    --model.diffusion_module.train_pair_transition_checkpoint_chunks true \
    --enable_diffusion_shared_vars_cache true \
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
    --benchmark_warmup_steps -1 \
    --benchmark_measure_steps -1 \
    --checkpoint_interval "${resolved_checkpoint_interval}" \
    --skip_final_checkpoint "${resolved_skip_final_checkpoint}" \
    --log_interval 25 \
    --eval_interval 0 \
    --eval_at_end false \
    --eval_ema_only false
