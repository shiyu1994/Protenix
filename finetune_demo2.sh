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
export PYTHONPATH="${PYTHONPATH:-}:${repo_dir}"

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/mnt/project_abd/klp/miniconda3/bin/activate}"
CONDA_ENV="${CONDA_ENV:-protenix}"
if [ ! -f "${CONDA_ACTIVATE}" ]; then
    echo "Missing conda activate script: ${CONDA_ACTIVATE}" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${CONDA_ACTIVATE}" "${CONDA_ENV}"
CONDA_ROOT="${CONDA_ACTIVATE%/bin/activate}"
CONDA_ENV_BIN="${CONDA_ROOT}/envs/${CONDA_ENV}/bin"
if [ ! -d "${CONDA_ENV_BIN}" ]; then
    echo "Missing conda env bin directory: ${CONDA_ENV_BIN}" >&2
    exit 1
fi
export PATH="${CONDA_ENV_BIN}:${PATH}"
hash -r

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_TIMEOUT_SECOND="${NCCL_TIMEOUT_SECOND:-1800}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
default_2026_root="/mnt/project_abd/public_databases/protenix.20260101"
local_2026_root="${repo_dir}/../../../../../public_databases/protenix.20260101"
if [ -z "${PROTENIX_ROOT_DIR:-}" ]; then
    if [ -d "${default_2026_root}" ]; then
        export PROTENIX_ROOT_DIR="${default_2026_root}"
    elif [ -d "${local_2026_root}" ]; then
        export PROTENIX_ROOT_DIR="${local_2026_root}"
    else
        export PROTENIX_ROOT_DIR="${default_2026_root}"
    fi
fi

checkpoint_path="${checkpoint_path:-${CHECKPOINT_PATH:-${PROTENIX_ROOT_DIR}/checkpoint/protenix-v2.pt}}"
load_checkpoint_path="${load_checkpoint_path-${LOAD_CHECKPOINT_PATH-${checkpoint_path}}}"
load_ema_checkpoint_path="${load_ema_checkpoint_path-${LOAD_EMA_CHECKPOINT_PATH-${checkpoint_path}}}"
base_dir="${base_dir:-${BASE_DIR:-/mnt/project_abd/klp/proteinx_output}}"
run_name="${run_name:-${RUN_NAME:-protenix_finetune_20260101_weightedpdb_lr1e-5}}"
model_name="${model_name:-${MODEL_NAME:-protenix-v2}}"

nproc_per_node="${nproc_per_node:-${NPROC_PER_NODE:-gpu}}"
nnodes="${nnodes:-${NNODES:-${MLP_WORKER_NUM:-${SLURM_NNODES:-1}}}}"
node_rank="${node_rank:-${NODE_RANK:-${MLP_ROLE_INDEX:-${SLURM_NODEID:-0}}}}"
master_addr="${master_addr:-${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}}"
master_port="${master_port:-${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}}"
if [ "${nnodes}" != "1" ] && [ "${master_addr}" = "127.0.0.1" ]; then
    echo "Multi-node training requires MASTER_ADDR or MLP_WORKER_0_HOST." >&2
    exit 1
fi

weighted_train_set="${weighted_train_set:-${WEIGHTED_TRAIN_SET:-weightedPDB_before210930_v20260101}}"
weighted_indices_default="${PROTENIX_ROOT_DIR}/indices/indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz"
if [ "${weighted_train_set}" = "weightedPDB_before250701_v20260101" ]; then
    weighted_indices_default="${PROTENIX_ROOT_DIR}/indices/indices_20260107-20chains_before_2025-07-01_res4.5.csv.gz"
fi
weighted_mmcif_dir="${WEIGHTED_MMCIF_DIR:-${PROTENIX_ROOT_DIR}/mmcif}"
weighted_bioassembly_dir="${WEIGHTED_BIOASSEMBLY_DIR:-${PROTENIX_ROOT_DIR}/mmcif_bioassembly}"
weighted_indices_fpath="${WEIGHTED_INDICES_FPATH:-${weighted_indices_default}}"

sabdab_root_dir="${sabdab_root_dir:-${SABDAB_ROOT_DIR:-/mnt/project_abd/zhe/protenix_copy/vepfs-mlp2/project-abd/zhujianwei/public_databases/protenix}}"
export SABDAB_ROOT_DIR="${sabdab_root_dir}"
sabdab_train_mmcif_dir="${SABDAB_TRAIN_MMCIF_DIR:-${sabdab_root_dir}/sabdab-train_mmcif}"
sabdab_train_bioassembly_dir="${SABDAB_TRAIN_BIOASSEMBLY_DIR:-${sabdab_root_dir}/sabdab-train_bioassembly}"
sabdab_train_indices_fpath="${SABDAB_TRAIN_INDICES_FPATH:-${sabdab_root_dir}/indices/sabdab-train.csv}"
sabdab_train_seq_index_json="${SABDAB_TRAIN_SEQ_INDEX_JSON:-${sabdab_root_dir}/sabdab-train.json}"
distill_sabdab_mmcif_dir="${DISTILL_SABDAB_MMCIF_DIR:-${sabdab_root_dir}/distill-sabdab_cif}"
distill_sabdab_bioassembly_dir="${DISTILL_SABDAB_BIOASSEMBLY_DIR:-${sabdab_root_dir}/distill-sabdab_bioassembly}"
distill_sabdab_indices_fpath="${DISTILL_SABDAB_INDICES_FPATH:-${sabdab_root_dir}/indices/distill-sabdab.csv}"
distill_sabdab_seq_index_json="${DISTILL_SABDAB_SEQ_INDEX_JSON:-${sabdab_root_dir}/sab-20260416.json}"
sabdab_msa_template_dir="${SABDAB_MSA_TEMPLATE_DIR:-${sabdab_root_dir}/mmcif_msa_template}"

posebusters_mmcif_dir="${POSEBUSTERS_MMCIF_DIR:-${PROTENIX_ROOT_DIR}/posebusters_mmcif}"
posebusters_bioassembly_dir="${POSEBUSTERS_BIOASSEMBLY_DIR:-${PROTENIX_ROOT_DIR}/posebusters_bioassembly}"
posebusters_indices_fpath="${POSEBUSTERS_INDICES_FPATH:-${PROTENIX_ROOT_DIR}/indices/posebusters_indices_mainchain_interface.csv}"
default_train_sets="${weighted_train_set},sabdab-train"
default_train_sample_weights="0.7,0.3"
required_paths=(
    "${weighted_mmcif_dir}"
    "${weighted_bioassembly_dir}"
    "${weighted_indices_fpath}"
    "${sabdab_train_mmcif_dir}"
    "${sabdab_train_bioassembly_dir}"
    "${sabdab_train_indices_fpath}"
    "${sabdab_train_seq_index_json}"
    "${distill_sabdab_mmcif_dir}"
    "${distill_sabdab_bioassembly_dir}"
    "${distill_sabdab_indices_fpath}"
    "${distill_sabdab_seq_index_json}"
    "${sabdab_msa_template_dir}"
    "${posebusters_mmcif_dir}"
    "${posebusters_bioassembly_dir}"
    "${posebusters_indices_fpath}"
)
if [ -n "${load_checkpoint_path}" ]; then
    required_paths+=("${load_checkpoint_path}")
fi
if [ -n "${load_ema_checkpoint_path}" ]; then
    required_paths+=("${load_ema_checkpoint_path}")
fi

if [ "${DRY_RUN:-0}" != "1" ] && [ "${SKIP_PATH_CHECK:-0}" != "1" ]; then
    missing=0
    for path in "${required_paths[@]}"; do
        if [ ! -e "${path}" ]; then
            echo "Missing required path: ${path}" >&2
            missing=1
        fi
    done
    if [ "${missing}" != "0" ]; then
        exit 1
    fi
fi

if [ "${CHECK_ONLY:-0}" = "1" ]; then
    echo "finetune_demo.sh path check passed for PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR}"
    exit 0
fi

torchrun_args=(
    --nproc_per_node "${nproc_per_node}"
    --nnodes "${nnodes}"
    --node_rank "${node_rank}"
    --master_addr "${master_addr}"
    --master_port "${master_port}"
)
train_args=(
    --model_name "${model_name}"
    --run_name "${run_name}"
    --seed "${seed:-42}"
    --base_dir "${base_dir}"
    --dtype "${dtype:-bf16}"
    --project "${project:-protenix}"
    --use_wandb "${use_wandb:-false}"
    --diffusion_batch_size "${diffusion_batch_size:-48}"
    --eval_interval "${eval_interval:-400}"
    --eval_at_end "${eval_at_end:-${EVAL_AT_END:-true}}"
    --eval_only "${eval_only:-${EVAL_ONLY:-false}}"
    --eval_ema_only "${eval_ema_only:-${EVAL_EMA_ONLY:-false}}"
    --eval_worst_sample_count "${eval_worst_sample_count:-${EVAL_WORST_SAMPLE_COUNT:-20}}"
    --log_interval "${log_interval:-50}"
    --checkpoint_interval "${checkpoint_interval:-1000}"
    --empty_cache_interval "${empty_cache_interval:-${EMPTY_CACHE_INTERVAL:-0}}"
    --ema_decay "${ema_decay:-0.999}"
    --train_crop_size "${train_crop_size:-768}"
    --max_steps "${max_steps:-20000}"
    --warmup_steps "${warmup_steps:-2000}"
    --lr "${lr:-0.00001}"
    --model.N_cycle "${N_cycle:-4}"
    --sample_diffusion.N_step "${N_step:-20}"
    --triangle_attention "${triangle_attention:-cuequivariance}"
    --triangle_multiplicative "${triangle_multiplicative:-cuequivariance}"
    --data.train_sets "${train_sets:-${default_train_sets}}"
    --data.train_sampler.train_sample_weights "${train_sample_weights:-${default_train_sample_weights}}"
    --data.test_sets "${test_sets:-posebusters_0925}"
    --data.posebusters_0925.base_info.max_n_token "${posebusters_max_n_token:-512}"
    --data.msa.max_size.train "${msa_max_size_train:-4096}"
    --iters_to_accumulate "${iters_to_accumulate:-1}"
)
if [ -n "${load_checkpoint_path}" ]; then
    train_args+=(--load_checkpoint_path "${load_checkpoint_path}")
fi
if [ -n "${load_ema_checkpoint_path}" ]; then
    train_args+=(--load_ema_checkpoint_path "${load_ema_checkpoint_path}")
fi
train_args+=(
    --data.${weighted_train_set}.base_info.mmcif_dir "${weighted_mmcif_dir}"
    --data.${weighted_train_set}.base_info.bioassembly_dict_dir "${weighted_bioassembly_dir}"
    --data.${weighted_train_set}.base_info.indices_fpath "${weighted_indices_fpath}"
    --data.sabdab-train.base_info.mmcif_dir "${sabdab_train_mmcif_dir}"
    --data.sabdab-train.base_info.bioassembly_dict_dir "${sabdab_train_bioassembly_dir}"
    --data.sabdab-train.base_info.indices_fpath "${sabdab_train_indices_fpath}"
    --data.sabdab-train.msa.prot_seq_or_filename_to_msadir_jsons "${sabdab_train_seq_index_json}"
    --data.sabdab-train.msa.prot_msadir_raw_paths "${sabdab_msa_template_dir}"
    --data.sabdab-train.template.prot_seq_or_filename_to_templatedir_jsons "${sabdab_train_seq_index_json}"
    --data.sabdab-train.template.prot_template_raw_paths "${sabdab_msa_template_dir}"
    --data.distill-sabdab.base_info.mmcif_dir "${distill_sabdab_mmcif_dir}"
    --data.distill-sabdab.base_info.bioassembly_dict_dir "${distill_sabdab_bioassembly_dir}"
    --data.distill-sabdab.base_info.indices_fpath "${distill_sabdab_indices_fpath}"
    --data.distill-sabdab.msa.prot_seq_or_filename_to_msadir_jsons "${distill_sabdab_seq_index_json}"
    --data.distill-sabdab.msa.prot_msadir_raw_paths "${sabdab_msa_template_dir}"
    --data.distill-sabdab.template.prot_seq_or_filename_to_templatedir_jsons "${distill_sabdab_seq_index_json}"
    --data.distill-sabdab.template.prot_template_raw_paths "${sabdab_msa_template_dir}"
    --data.posebusters_0925.base_info.mmcif_dir "${posebusters_mmcif_dir}"
    --data.posebusters_0925.base_info.bioassembly_dict_dir "${posebusters_bioassembly_dir}"
    --data.posebusters_0925.base_info.indices_fpath "${posebusters_indices_fpath}"
)
cmd=(torchrun "${torchrun_args[@]}" ./runner/train.py "${train_args[@]}")

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf "%q " "${cmd[@]}"
    echo
    exit 0
fi

"${cmd[@]}"
