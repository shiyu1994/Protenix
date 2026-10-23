#!/usr/bin/env bash
# Evaluate one formal crop384/DB48 v8 EMA checkpoint on one isolated GPU.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 0 )); then
    echo "This evaluator accepts environment variables only." >&2
    exit 2
fi

ema_checkpoint="${EMA_CHECKPOINT:-${EMA_CHECKPOINT_PATH:-}}"
if [ -z "${ema_checkpoint}" ]; then
    echo "EMA_CHECKPOINT is required." >&2
    exit 2
fi
ema_basename="$(basename "${ema_checkpoint}")"
if [[ ! "${ema_basename}" =~ ^([0-9]+)_ema_0[.]999[.]pt$ ]]; then
    echo "EMA_CHECKPOINT must end in <zero_based_step>_ema_0.999.pt." >&2
    exit 2
fi
checkpoint_step="${BASH_REMATCH[1]}"
completed_updates=$((checkpoint_step + 1))
if (( completed_updates < 5000 \
    || completed_updates > 50000 \
    || completed_updates % 5000 != 0 )); then
    echo "EMA evaluation is locked to updates 5000,10000,...,50000; got ${completed_updates}." >&2
    exit 2
fi

visible_devices="${CUDA_VISIBLE_DEVICES:-}"
if [ -z "${visible_devices}" ]; then
    echo "Set CUDA_VISIBLE_DEVICES to the one GPU allocated to this independent evaluator." >&2
    exit 2
fi
if [[ "${visible_devices}" == *,* ]]; then
    echo "EMA evaluation is single-GPU; CUDA_VISIBLE_DEVICES must expose exactly one GPU." >&2
    exit 2
fi

if [ "${DRY_RUN:-0}" != "1" ]; then
    if [ ! -s "${ema_checkpoint}" ]; then
        echo "EMA checkpoint is missing or empty: ${ema_checkpoint}" >&2
        exit 2
    fi
    validator_python="${VALIDATOR_PYTHON:-/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python}"
    if [ ! -x "${validator_python}" ]; then
        echo "Checkpoint validator Python is not executable: ${validator_python}" >&2
        exit 2
    fi
    "${validator_python}" \
        "${repo_dir}/scripts/runtime/validate_crop384_db48_ema_checkpoint.py" \
        "${ema_checkpoint}"
fi

official_root="/mnt/project_abd/public_databases/protenix.20260101"
weighted_set="weightedPDB_before210930_v20260101"
data_profile_registry="${repo_dir}/configs/crop384_db48_data_profiles.sh"
# shellcheck disable=SC1090
source "${data_profile_registry}"
resolve_crop384_db48_data_profile weighted_pdb_only_v1

resolved_run_name="${RUN_NAME:-crop384_db48_v8_ema_eval_u${completed_updates}}"
if [[ ! "${resolved_run_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
fi

# Explicit single-node/single-rank topology plus a one-visible-GPU runtime
# assertion prevents this helper from accidentally joining or occupying the
# 128-GPU training allocation.
export NNODES=1
export nnodes=1
export NODE_RANK=0
export node_rank=0
export NPROC_PER_NODE=1
export nproc_per_node=1
export MASTER_ADDR=127.0.0.1
export master_addr=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29680}"
export master_port="${MASTER_PORT}"
export PROTENIX_EXPECTED_WORLD_SIZE=1
export PROTENIX_EXPECTED_LOCAL_WORLD_SIZE=1
export PROTENIX_EXPECTED_NUM_NODES=1

export RUN_NAME="${resolved_run_name}"
export run_name="${resolved_run_name}"
export BASE_DIR="${EVAL_BASE_DIR:-/mnt/project_abd/klp/protenix_eval_outputs/crop384_db48_v8_ema}"
export base_dir="${BASE_DIR}"
export PROTENIX_ROOT_DIR="${official_root}"
export PROTENIX_TRAINING_CONTRACT_ID="crop384_db48_ddp_scratch_50k_ws128_acc4_data_${CROP384_DATA_PROFILE_CONTRACT_TAG}_arch_pf64_cs384_dct768_pfckpt8_c48_msamix_q20_u30_p50_v8"

# Lower-case empty values are deliberate: finetune_demo.sh otherwise falls
# back to the official raw checkpoint. Only the selected EMA snapshot is loaded.
export CHECKPOINT_PATH=
export checkpoint_path=
export LOAD_CHECKPOINT_PATH=
export load_checkpoint_path=
export LOAD_EMA_CHECKPOINT_PATH="${ema_checkpoint}"
export load_ema_checkpoint_path="${ema_checkpoint}"
export LOAD_PARAMS_ONLY=true
export load_params_only=true
export EVAL_ONLY=true
export eval_only=true
export EVAL_EMA_ONLY=true
export eval_ema_only=true
export EVAL_AT_END=false
export eval_at_end=false
export eval_interval=0
export checkpoint_interval=-1
export SKIP_FINAL_CHECKPOINT=true
export skip_final_checkpoint=true

export MODEL_NAME=protenix-v2
export model_name=protenix-v2
export MODEL_C_Z=256
export model_c_z=256
export MODEL_C_M=128
export model_c_m=128
export train_crop_size=384
export diffusion_batch_size=48
export DIFFUSION_CHUNK_SIZE=48
export diffusion_chunk_size=48
export dtype=bf16
export N_cycle=4
export N_step=20
export seed=42
export DISTRIBUTED_STRATEGY=ddp
export distributed_strategy=ddp
export OPTIMIZER_SHARDING=none
export optimizer_sharding=none
export EMA_SHADOW_DEVICE=model
export ema_shadow_device=model
export ema_decay=0.999
export triangle_attention=cuequivariance
export triangle_multiplicative=cuequivariance

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
export test_sets=posebusters_0925
export posebusters_max_n_token=512
export NUM_DL_WORKERS="${EVAL_NUM_DL_WORKERS:-8}"
export num_dl_workers="${NUM_DL_WORKERS}"
export EPOCH_SIZE=10240
export epoch_size=10240
export PROTENIX_MP_SHARING_STRATEGY=file_descriptor

export PAIRFORMER_BLOCKS_PER_CKPT=8
export pairformer_blocks_per_ckpt=8
export MSA_CHUNK_SIZE=2048
export msa_chunk_size=2048
export MSA_MIN_SIZE_TRAIN=1
export msa_min_size_train=1
export MSA_SAMPLE_CUTOFF_TRAIN=2048
export msa_sample_cutoff_train=2048
export msa_max_size_train=16384
export PAIR_MSA_AS_UNPAIRED_ONLY=false
export pair_msa_as_unpaired_only=false
export SHUFFLE_PAIR_MSA_WHEN_UNPAIRED_ONLY=false
export shuffle_pair_msa_when_unpaired_only=false
unset PAIRFORMER_CHECKPOINT_CPU_OFFLOAD_EVERY_N
unset pairformer_checkpoint_cpu_offload_every_n
unset MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N
unset msa_checkpoint_cpu_offload_every_n

echo "Locked crop384/DB48 v8 independent EMA evaluation:"
echo "  checkpoint=${ema_checkpoint}"
echo "  checkpoint_step=${checkpoint_step} completed_updates=${completed_updates}"
echo "  topology=1x1 visible_gpu=${CUDA_VISIBLE_DEVICES}"
echo "  eval_only=true ema_only=true deterministic_seed=true mc_dropout=0 ref_pos_augment=false"
echo "  test_sets=posebusters_0925 max_n_token=512 N_sample/N_step=5/20"
echo "  training_job_inline_eval=false; this process uses an independent GPU"
echo "  contract=${PROTENIX_TRAINING_CONTRACT_ID}"

data_profile_args=("${CROP384_DATA_PROFILE_ARGS[@]}")
exec bash "${repo_dir}/finetune_demo.sh" \
    --model_name protenix-v2 \
    --n_blocks 64 \
    --c_z 256 \
    --model.relative_position_encoding.c_z 256 \
    --model.template_embedder.c_z 256 \
    --model.msa_module.c_z 256 \
    --model.msa_module.c_m 128 \
    --model.constraint_embedder.c_constraint_z 256 \
    --model.pairformer.c_z 256 \
    --model.diffusion_module.c_z 256 \
    --model.confidence_head.c_z 256 \
    --model.distogram_head.c_z 256 \
    --train_crop_size 384 \
    --diffusion_batch_size 48 \
    --diffusion_chunk_size 48 \
    --distributed_strategy ddp \
    --ddp.gradient_as_bucket_view true \
    --optimizer_sharding none \
    --max_steps 1 \
    --warmup_steps 1000 \
    --lr 0.001 \
    --lr_scheduler af3 \
    --decay_every_n_steps 25000 \
    --af3_lr_scheduler.decay_factor 0.95 \
    --ema_decay 0.999 \
    --ema_shadow_device model \
    --model.N_cycle 4 \
    --sample_diffusion.N_step 20 \
    --sample_diffusion.N_sample 5 \
    --seed 42 \
    --deterministic_seed true \
    --mc_dropout_apply_rate 0 \
    --iters_to_accumulate 4 \
    --data.epoch_size 10240 \
    --data.num_dl_workers "${NUM_DL_WORKERS}" \
    --data.train_sets "${CROP384_DATA_PROFILE_TRAIN_SETS}" \
    --data.train_sampler.train_sample_weights "${CROP384_DATA_PROFILE_TRAIN_SAMPLE_WEIGHTS}" \
    "${data_profile_args[@]}" \
    --data.test_sets posebusters_0925 \
    --data.posebusters_0925.base_info.max_n_token 512 \
    --data.test_ref_pos_augment false \
    --data.query_only_msa_probability 0.2 \
    --data.pair_msa_as_unpaired_probability 0.3 \
    --data.template_use_probability 0.2 \
    --data.augmentation_decision_version sample_occurrence_sha256_v1 \
    --data.template.template_dropout_rate 0.0 \
    --data.msa.min_size.train 1 \
    --data.msa.max_size.train 16384 \
    --data.msa.sample_cutoff.train 2048 \
    --data.msa.pair_msa_as_unpaired_only false \
    --data.msa.shuffle_pair_msa_when_unpaired_only false \
    --model.msa_module.msa_chunk_size 2048 \
    --model.msa_module.msa_max_size 2048 \
    --model.pairformer.blocks_per_ckpt 8 \
    --model.msa_module.blocks_per_ckpt 4 \
    --model.diffusion_module.blocks_per_ckpt 3 \
    --model.diffusion_module.use_fine_grained_checkpoint true \
    --model.pairformer.checkpoint_cpu_offload_every_n null \
    --model.msa_module.checkpoint_cpu_offload_every_n null \
    --model.pairformer.train_pair_transition_chunk_size null \
    --model.diffusion_module.train_pair_transition_chunk_size null \
    --model.pairformer.train_pair_transition_checkpoint_chunks false \
    --model.diffusion_module.train_pair_transition_checkpoint_chunks false \
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
    --load_ema_checkpoint_path "${ema_checkpoint}" \
    --load_params_only true \
    --load_strict true \
    --eval_only true \
    --eval_ema_only true \
    --eval_interval 0 \
    --eval_at_end false \
    --log_interval 25 \
    --checkpoint_interval -1 \
    --skip_final_checkpoint true
