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

usage() {
    cat <<'HELP'
Usage: bash finetune_peptide.sh [all|prepare|train] [--dry-run] [--runner.option value ...]

train (default): verify the prepared cache, then finetune without reconverting CIFs.
all:             prepare/reuse the CIF cache, then finetune.
prepare:         prepare the entire CIF directory without allocating a GPU.
--dry-run:       print commands without preprocessing or training.

Standalone CPU preprocessing: bash preprocess_peptide.sh
Then use this launcher's train mode with the same data settings.

Main environment variables:
  DATA_DIR                 Converted peptide CIF directory (defaults to your supplied data)
  PREPARED_DIR             Cache directory (default: ../peptide_finetune_data)
  OUTPUT_DIR               Training output root (default: /mnt/project_abd/shiyu/peptide_finetune_output)
  PYTHON_BIN               Python interpreter (defaults to the working protenix Conda env)
  PROTENIX_ROOT_DIR        Shared CCD/checkpoint root
  ORIGINAL_MMCIF_DIR       Original PDB CIFs (default: PROTENIX_ROOT_DIR/mmcif)
  PREP_REUSE_DIR           Optional existing cache to repair into PREPARED_DIR
  CHECKPOINT_PATH         Initial weights (default: official protenix-v2.pt)
  RESUME_CHECKPOINT        Resume a raw checkpoint from THIS launcher, including optimizer
  CUDA_VISIBLE_DEVICES     GPUs to expose (default: 0,1,2,3,4,5,6,7)
  NPROC_PER_NODE           Number of training processes (default: 8; single node)
  PREP_WORKERS             Preprocessing workers (default: 4)
  CIF_IDS                  Optional comma-separated CIF stems; unset means all files
  SKIP_PREP_ERRORS         Set 1 to explicitly exclude failed CIFs (default: 0)
  MAX_STEPS                Total optimizer updates (default: 100000)
  LR / WARMUP_STEPS        Learning rate / warmup updates (default: 1e-5 / 200)
  CROP_SIZE                Maximum crop tokens (default: 384)
  COMPLETE_LIGAND_CROPS     Require whole ligands in crops (default: false)
  DIFFUSION_BATCH_SIZE     Noise samples per structure (default: 4)
  N_CYCLE                  Maximum recycling iterations (default: 4)
  MINI_ROLLOUT_STEPS       Confidence rollout steps (default: 20)
  ACCUMULATION_STEPS       Microbatches per optimizer update (default: 1)
  CHECKPOINT_INTERVAL      Save every N updates (default: 500; final checkpoint always saved)
  LOG_INTERVAL             Log every N updates (default: 10)
  NUM_WORKERS              Training loader workers per GPU (default: 2)
  RUN_NAME                 Output name prefix (default: peptide_ligand_v2_finetune)

MSA/templates and evaluation are disabled by default. Extra trainer arguments can
enable protein MSAs; see docs/peptide_msa_finetuning.md for the recorded MSA run.
Unknown resolution masks confidence supervision. Training samples converted-peptide
interfaces with proteins or nucleic acids. See docs/peptide_finetuning.md for
preprocessing, resume, and smoke examples.
HELP
}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode=train
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if [[ ${1:-} == all || ${1:-} == prepare || ${1:-} == train ]]; then
    mode=$1
    shift
fi
dry_run="${DRY_RUN:-0}"
if [[ ${1:-} == --dry-run ]]; then dry_run=1; shift; fi

data_dir="${DATA_DIR:-/root/shiyu/pdb_peptide_to_lig_20190930_strict_filtered_nan0.7_mmcif_ligand_peptide}"
prepared_dir="${PREPARED_DIR:-${repo_dir}/../peptide_finetune_data}"
output_dir="${OUTPUT_DIR:-/mnt/project_abd/shiyu/peptide_finetune_output}"
python_bin="${PYTHON_BIN:-/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/mnt/project_abd/public_databases/protenix.20260101}"
checkpoint="${RESUME_CHECKPOINT:-${CHECKPOINT_PATH:-${PROTENIX_ROOT_DIR}/checkpoint/protenix-v2.pt}}"
export PYTHONPATH="${repo_dir}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export LAYERNORM_TYPE=torch
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PROTENIX_MP_SHARING_STRATEGY=file_system

prep=("${python_bin}" "${repo_dir}/scripts/prepare_peptide_training_data.py"
    --input-dir "${data_dir}" --output-dir "${prepared_dir}"
    --original-mmcif-dir "${ORIGINAL_MMCIF_DIR:-${PROTENIX_ROOT_DIR}/mmcif}"
    --workers "${PREP_WORKERS:-4}" --seed "${SEED:-42}")
if [[ -n ${PREP_REUSE_DIR:-} ]]; then prep+=(--reuse-prepared-dir "${PREP_REUSE_DIR}"); fi
if [[ -n ${CIF_IDS:-} ]]; then prep+=(--ids "${CIF_IDS}"); fi
if [[ ${SKIP_PREP_ERRORS:-0} == 1 ]]; then prep+=(--skip-errors); fi
if [[ ${mode} == train ]]; then prep+=(--check); fi

# Reuse a standard dataset configuration and replace every structure/index path.
# Uniform interface sampling avoids treating missing PDB cluster IDs as clusters.
dataset=weightedPDB_before2109_wopb_nometalc_0925
params_only=true
if [[ -n ${RESUME_CHECKPOINT:-} ]]; then params_only=false; fi
train=("${python_bin}" -m torch.distributed.run --standalone --nnodes=1
    --nproc_per_node "${NPROC_PER_NODE:-8}" "${repo_dir}/runner/train.py"
    --project peptide_ligand_finetune --run_name "${RUN_NAME:-peptide_ligand_v2_finetune}"
    --base_dir "${output_dir}" --model_name protenix-v2
    --load_checkpoint_path "${checkpoint}" --load_params_only "${params_only}" --load_strict true
    --use_wandb false --dtype bf16 --seed "${SEED:-42}"
    --max_steps "${MAX_STEPS:-100000}" --lr "${LR:-0.00001}" --warmup_steps "${WARMUP_STEPS:-200}"
    --iters_to_accumulate "${ACCUMULATION_STEPS:-1}" --ema_decay -1
    --log_interval "${LOG_INTERVAL:-10}" --checkpoint_interval "${CHECKPOINT_INTERVAL:-500}"
    --eval_interval 0 --eval_first false --eval_at_end false
    --train_crop_size "${CROP_SIZE:-384}" --diffusion_batch_size "${DIFFUSION_BATCH_SIZE:-4}"
    --diffusion_chunk_size 1 --model.N_cycle "${N_CYCLE:-4}"
    --sample_diffusion.N_step_mini_rollout "${MINI_ROLLOUT_STEPS:-20}"
    --sample_diffusion.N_sample_mini_rollout 1
    --triangle_attention torch --triangle_multiplicative torch
    --data.num_dl_workers "${NUM_WORKERS:-2}" --data.epoch_size "${EPOCH_SIZE:-10000}"
    --data.train_sets "${dataset}" --data.test_sets "${dataset}"
    --data.train_sampler.train_sample_weights 1.0
    --data.msa.enable_prot_msa false --data.msa.enable_rna_msa false
    --data.template.enable_prot_template false --data.template_use_probability 0.0
    --data."${dataset}".base_info.mmcif_dir "${data_dir}"
    --data."${dataset}".base_info.bioassembly_dict_dir "${prepared_dir}/bioassembly"
    --data."${dataset}".base_info.indices_fpath "${prepared_dir}/indices.csv"
    --data."${dataset}".base_info.random_sample_if_failed false
    --data."${dataset}".sampler_configs.sampler_type uniform
    --data."${dataset}".cropping_configs.method_weights 0.0,0.0,1.0
    --data."${dataset}".cropping_configs.spatial_crop_complete_lig "${COMPLETE_LIGAND_CROPS:-false}"
    "$@")

printf 'Preprocessing command: '; printf '%q ' "${prep[@]}"; printf '\n'
if [[ ${mode} != prepare ]]; then
    printf 'Training command: '; printf '%q ' "${train[@]}"; printf '\n'
fi
if [[ ${dry_run} == 1 ]]; then exit 0; fi
if [[ ! -x ${python_bin} ]]; then echo "Missing executable PYTHON_BIN: ${python_bin}" >&2; exit 1; fi
if [[ ${mode} != prepare && ! -f ${checkpoint} ]]; then echo "Missing checkpoint: ${checkpoint}" >&2; exit 1; fi
# Prior production-launcher contracts must not leak into this independent run.
if [[ -n ${PROTENIX_FULL_RESUME:-} || -n ${PROTENIX_INITIALIZATION_MODE:-} ]]; then
    echo 'Unset PROTENIX_FULL_RESUME and PROTENIX_INITIALIZATION_MODE before using this launcher.' >&2
    exit 1
fi
python_env_bin="$(dirname "${python_bin}")"
export PATH="${python_env_bin}:${PATH}"
torch_lib="$("${python_bin}" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="${torch_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${repo_dir}"
"${prep[@]}"
if [[ ${mode} == prepare ]]; then exit 0; fi
mkdir -p "${output_dir}"
log_file="${output_dir}/launcher_$(date -u +%Y%m%d_%H%M%S)_$$.log"
printf 'Console log: %s\n' "${log_file}"
"${train[@]}" 2>&1 | tee "${log_file}"
