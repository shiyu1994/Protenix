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

# Evaluate one PF64/cz256/cm128/DB48 EMA checkpoint on an AF3-like,
# PXM-Legacy-derived held-out subset using one isolated GPU.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Pin every preflight validator and the eventual evaluator to this checkout,
# even when an AIStudio shell inherited PYTHONPATH from an older experiment.
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
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

eval_mode="${EVAL_MODE:-quick}"
case "${eval_mode}" in
    quick)
        diffusion_steps=20
        ;;
    formal)
        diffusion_steps=200
        ;;
    *)
        echo "EVAL_MODE must be quick or formal; got '${eval_mode}'." >&2
        exit 2
        ;;
esac

eval_scope="${EVAL_SCOPE:-smoke}"
eval_subset="${EVAL_SUBSET:-combined}"
manifest_dir="${PXM_MANIFEST_DIR:-/mnt/project_abd/klp/proteinx_eval_data/manifests/pxm_legacy_af3_like}"
case "${eval_scope}" in
    smoke)
        if [ "${eval_subset}" != "combined" ]; then
            echo "EVAL_SCOPE=smoke requires EVAL_SUBSET=combined." >&2
            exit 2
        fi
        pdb_list="${PDB_LIST:-${manifest_dir}/combined_le1024_smoke2pertype.txt}"
        ;;
    full)
        case "${eval_subset}" in
            combined|recentpdb|rna_protein|dna_protein)
                ;;
            *)
                echo "EVAL_SUBSET must be combined, recentpdb, rna_protein, or dna_protein." >&2
                exit 2
                ;;
        esac
        pdb_list="${PDB_LIST:-${manifest_dir}/${eval_subset}_le1024.txt}"
        ;;
    custom)
        pdb_list="${PDB_LIST:-}"
        if [ -z "${pdb_list}" ]; then
            echo "EVAL_SCOPE=custom requires PDB_LIST." >&2
            exit 2
        fi
        ;;
    *)
        echo "EVAL_SCOPE must be smoke, full, or custom; got '${eval_scope}'." >&2
        exit 2
        ;;
esac
if [ ! -s "${pdb_list}" ]; then
    echo "Evaluation PDB manifest is missing or empty: ${pdb_list}" >&2
    exit 2
fi
target_count="$(awk 'NF {count += 1} END {print count + 0}' "${pdb_list}")"

eval_max_n_token="${EVAL_MAX_N_TOKEN:-1024}"
if [[ ! "${eval_max_n_token}" =~ ^-?[0-9]+$ ]] \
    || (( eval_max_n_token == 0 || eval_max_n_token < -1 )); then
    echo "EVAL_MAX_N_TOKEN must be -1 (unbounded) or a positive integer." >&2
    exit 2
fi
eval_sample_diffusion_chunk_size="${EVAL_SAMPLE_DIFFUSION_CHUNK_SIZE:-5}"
if [[ ! "${eval_sample_diffusion_chunk_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVAL_SAMPLE_DIFFUSION_CHUNK_SIZE must be a positive integer." >&2
    exit 2
fi

visible_devices="${CUDA_VISIBLE_DEVICES:-}"
if [ -z "${visible_devices}" ]; then
    echo "Set CUDA_VISIBLE_DEVICES to the one GPU allocated to this evaluator." >&2
    exit 2
fi
if [[ "${visible_devices}" == *,* ]]; then
    echo "Evaluation is single-GPU; CUDA_VISIBLE_DEVICES must expose one GPU." >&2
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
        "${repo_dir}/scripts/evaluation/validate_af3_pxm_eval_checkpoint.py" \
        "${ema_checkpoint}"

    # Refuse to share a physical GPU with training unless the caller explicitly
    # raises the threshold. CUDA_VISIBLE_DEVICES is numeric in AIStudio jobs.
    if [[ "${visible_devices}" =~ ^[0-9]+$ ]] && command -v nvidia-smi >/dev/null 2>&1; then
        existing_gpu_mib="$(
            nvidia-smi \
                --id="${visible_devices}" \
                --query-gpu=memory.used \
                --format=csv,noheader,nounits \
                | tr -d '[:space:]'
        )"
        max_existing_gpu_mib="${EVAL_MAX_PREEXISTING_GPU_MIB:-2048}"
        if [[ ! "${max_existing_gpu_mib}" =~ ^[0-9]+$ ]]; then
            echo "EVAL_MAX_PREEXISTING_GPU_MIB must be a non-negative integer." >&2
            exit 2
        fi
        if (( existing_gpu_mib > max_existing_gpu_mib )); then
            echo "GPU ${visible_devices} already uses ${existing_gpu_mib} MiB; evaluator limit is ${max_existing_gpu_mib} MiB." >&2
            echo "Use a genuinely isolated GPU; do not share a training GPU." >&2
            exit 2
        fi
    fi
fi

official_root="/mnt/project_abd/public_databases/protenix.20260101"
weighted_set="weightedPDB_before210930_v20260101"
pxm_eval_input_root="${PXM_EVAL_INPUT_ROOT:-/mnt/project_abd/klp/proteinx_eval_data/cache/pxm_legacy_af3_like_cutoff20210930_v1}"
pxm_sequence_sources="${PXM_SEQUENCE_SOURCES:-${pxm_eval_input_root}/manifests/protein_sequence_sources.json}"
pxm_input_success="${PXM_INPUT_SUCCESS:-${pxm_eval_input_root}/manifests/_INPUT_SUCCESS.json}"
pxm_template_cache_root="${PXM_TEMPLATE_CACHE_ROOT:-${pxm_eval_input_root}/template_features}"
pxm_template_cache_success="${pxm_template_cache_root}/_SUCCESS.json"
pxm_msa_overlay_root="${PXM_MSA_OVERLAY_ROOT:-${pxm_eval_input_root}/msa_template}"
pxm_msa_overlay_mapping="${PXM_MSA_OVERLAY_MAPPING:-${pxm_msa_overlay_root}/seq_to_msa_template_dir.json}"
pxm_msa_lock_root="${PXM_MSA_LOCK_ROOT:-${pxm_eval_input_root}/msa_lock}"
pxm_msa_lock_manifest="${PXM_MSA_LOCK_MANIFEST:-${pxm_msa_lock_root}/msa_manifest.json}"
pxm_msa_lock_success="${PXM_MSA_LOCK_SUCCESS:-${pxm_msa_lock_root}/_MSA_SUCCESS.json}"
pxm_full_pdb_list="${PXM_FULL_PDB_LIST:-${manifest_dir}/combined_le1024.txt}"

for required_eval_input in \
    "${pxm_sequence_sources}" \
    "${pxm_input_success}" \
    "${pxm_template_cache_success}" \
    "${pxm_msa_overlay_mapping}" \
    "${pxm_msa_lock_manifest}" \
    "${pxm_msa_lock_success}" \
    "${pxm_full_pdb_list}"; do
    if [ ! -s "${required_eval_input}" ]; then
        echo "Locked PXM evaluation input is missing or empty: ${required_eval_input}" >&2
        exit 2
    fi
done

validator_python="${VALIDATOR_PYTHON:-/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python}"
if [ ! -x "${validator_python}" ]; then
    echo "Evaluation validator Python is not executable: ${validator_python}" >&2
    exit 2
fi
"${validator_python}" \
    "${repo_dir}/scripts/evaluation/validate_af3_pxm_template_cache.py" \
    --sequence-sources "${pxm_sequence_sources}" \
    --cache-root "${pxm_template_cache_root}"
"${validator_python}" \
    "${repo_dir}/scripts/evaluation/validate_af3_pxm_msa_lock.py" \
    --pdb-list "${pxm_full_pdb_list}" \
    --requested-pdb-list "${pdb_list}" \
    --bioassembly-dir "${official_root}/recentPDB_bioassembly" \
    --protein-sequence-sources "${pxm_sequence_sources}" \
    --rna-sequence-mapping "${official_root}/rna_msa/rna_sequence_to_pdb_chains.json" \
    --rna-msa-root "${official_root}/rna_msa/msas" \
    --manifest "${pxm_msa_lock_manifest}" \
    --success "${pxm_msa_lock_success}"

data_profile_registry="${repo_dir}/configs/crop384_db48_data_profiles.sh"
# shellcheck disable=SC1090
source "${data_profile_registry}"
# Evaluation only needs a valid, lightweight train-profile declaration.
resolve_crop384_db48_data_profile weighted_pdb_only_v1

default_run_name="af3_pxm_${eval_scope}_${eval_subset}_${eval_mode}_u${completed_updates}"
resolved_run_name="${RUN_NAME:-${default_run_name}}"
if [[ ! "${resolved_run_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
fi

# Lock this helper to one process on one node so it cannot join the 128-GPU
# training rendezvous.
export NNODES=1
export nnodes=1
export NODE_RANK=0
export node_rank=0
export NPROC_PER_NODE=1
export nproc_per_node=1
export MASTER_ADDR=127.0.0.1
export master_addr=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29681}"
export master_port="${MASTER_PORT}"
export PROTENIX_EXPECTED_WORLD_SIZE=1
export PROTENIX_EXPECTED_LOCAL_WORLD_SIZE=1
export PROTENIX_EXPECTED_NUM_NODES=1

export RUN_NAME="${resolved_run_name}"
export run_name="${resolved_run_name}"
export BASE_DIR="${EVAL_BASE_DIR:-/mnt/project_abd/klp/protenix_eval_outputs/af3_pxm_legacy_proxy}"
export base_dir="${BASE_DIR}"
export PROTENIX_ROOT_DIR="${official_root}"
export PROTENIX_TRAINING_CONTRACT_ID="eval_only_af3_pxm_legacy_pf64_cz256_cm128_db48"
# Evaluation and training template caches are intentionally independent.
# Pin the test-stage cache here so a reused AIStudio shell cannot leak the
# no-date top60/uniform20 training policy into checkpoint evaluation.
export PROTENIX_EVAL_SEQUENCE_TEMPLATE_CACHE_ROOT="${pxm_template_cache_root}"
export PROTENIX_EVAL_SEQUENCE_TEMPLATE_CACHE_MODE=required
export PROTENIX_EVAL_SEQUENCE_TEMPLATE_CACHE_FILENAME=template_features.pkl.gz
export PROTENIX_EVAL_SEQUENCE_TEMPLATE_CACHE_POLICY=pxm_eval_top4_maxdate_2021_09_30_v1
export PROTENIX_SEQUENCE_TEMPLATE_CACHE_MODE=online
export PROTENIX_SEQUENCE_TEMPLATE_CACHE_ROOT=
export PROTENIX_SEQUENCE_TEMPLATE_CACHE_ERROR_POLICY=raise

# Never inherit the official raw-checkpoint fallback. The selected EMA payload
# is loaded as model parameters only.
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
export N_step="${diffusion_steps}"
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
export train_sets="${CROP384_DATA_PROFILE_TRAIN_SETS}"
export train_sample_weights="${CROP384_DATA_PROFILE_TRAIN_SAMPLE_WEIGHTS}"
export test_sets=recentPDB_1536_sample384_0925
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
export PROTENIX_MSA_SAMPLE_DEBUG_LOG_LIMIT="${PROTENIX_MSA_SAMPLE_DEBUG_LOG_LIMIT:-2}"
unset PAIRFORMER_CHECKPOINT_CPU_OFFLOAD_EVERY_N
unset pairformer_checkpoint_cpu_offload_every_n
unset MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N
unset msa_checkpoint_cpu_offload_every_n

echo "Locked AF3-like PXM-Legacy EMA evaluation:"
echo "  checkpoint=${ema_checkpoint}"
echo "  checkpoint_step=${checkpoint_step} completed_updates=${completed_updates}"
echo "  scope/subset/mode=${eval_scope}/${eval_subset}/${eval_mode}"
echo "  manifest=${pdb_list} targets=${target_count} max_tokens=${eval_max_n_token}"
echo "  topology=1x1 visible_gpu=${CUDA_VISIBLE_DEVICES}"
echo "  model=PF64 c_z=256 c_m=128 diffusion_batch/chunk=48/48 sample_chunk=${eval_sample_diffusion_chunk_size}"
echo "  N_sample/N_step=5/${diffusion_steps} N_cycle=4"
echo "  deterministic_seed=true mc_dropout=0 ref_pos_augment=false"
echo "  templates=required-eval-cache max_template_date=2021-09-30 max_hits=4"
echo "  template_cache=${pxm_template_cache_root} (fully validated)"
echo "  protein_msa=official+evaluation-overlay overlay=${pxm_msa_overlay_root}"
echo "  msa_lock=${pxm_msa_lock_manifest} (all requested files fully rehashed)"
echo "  MSA=input<=16384 K~Uniform[1,min(depth,16384)] chunk=2048 deterministic seed"
echo "  suite=7-category AF3-like proxy (no intra_ligand scoring row)"
echo "  metric=AF3-style global-gPDE-top1 plus per-entry best-of-5 LDDT"
echo "  note=AF3 internal 1220-target Fig2d IDs are not public; this is a reproducible proxy"

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
    --sample_diffusion.N_step "${diffusion_steps}" \
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
    --data.test_sets recentPDB_1536_sample384_0925 \
    --data.recentPDB_1536_sample384_0925.base_info.pdb_list "${pdb_list}" \
    --data.recentPDB_1536_sample384_0925.base_info.max_n_token "${eval_max_n_token}" \
    --data.msa.prot_seq_or_filename_to_msadir_jsons "${official_root}/common/seq_to_pdb_index.json,${pxm_msa_overlay_mapping}" \
    --data.msa.prot_msadir_raw_paths "${official_root}/mmcif_msa_template,${pxm_msa_overlay_root}" \
    --data.msa.prot_pairing_dbs pairing,pairing \
    --data.msa.prot_non_pairing_dbs pairing-non_pairing,pairing-non_pairing \
    --data.msa.prot_indexing_methods sequence,sequence \
    --data.template.evaluation_sequence_template_cache_root "${pxm_template_cache_root}" \
    --data.template.evaluation_sequence_template_cache_mode required \
    --data.template.evaluation_sequence_template_cache_filename template_features.pkl.gz \
    --data.template.evaluation_sequence_template_cache_policy pxm_eval_top4_maxdate_2021_09_30_v1 \
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
    --infer_setting.dynamic_chunk_size true \
    --infer_setting.sample_diffusion_chunk_size "${eval_sample_diffusion_chunk_size}" \
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
