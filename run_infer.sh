#!/usr/bin/env bash
set -euo pipefail

test_run="${test_run:-${TEST_RUN:-99}}"

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/mnt/project_abd/klp/miniconda3/bin/activate}"
CONDA_ENV="${CONDA_ENV:-protenix}"

export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/mnt/project_abd/public_databases/protenix.20260101}"
load_checkpoint_dir="${load_checkpoint_dir:-${PROTENIX_ROOT_DIR}/checkpoint/protenix-v2.pt}"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

N_sample="${N_sample:-1}"
N_step="${N_step:-5}"
N_cycle="${N_cycle:-1}"
seed="${seed:-101}"
input_json_path="${input_json_path:-./examples/examples_with_template/example_9fm7.json}"
dump_dir="${dump_dir:-./test_outputs/run_infer/${test_run}}"
model_name="${model_name:-protenix-v2}"
nproc_per_node="${nproc_per_node:-1}"
master_port="${master_port:-23333}"

if [ ! -f "${CONDA_ACTIVATE}" ]; then
    echo "Missing conda activate script: ${CONDA_ACTIVATE}" >&2
    exit 1
fi

if [ ! -f "${load_checkpoint_dir}" ]; then
    echo "Missing checkpoint: ${load_checkpoint_dir}" >&2
    exit 1
fi

if [ ! -f "${input_json_path}" ]; then
    echo "Missing input JSON: ${input_json_path}" >&2
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

torchrun_args=(
    --nproc_per_node "${nproc_per_node}"
    --master_port "${master_port}"
    runner/inference.py
    --model_name "${model_name}"
    --seeds "${seed}"
    --dump_dir "${dump_dir}"
    --input_json_path "${input_json_path}"
    --model.N_cycle "${N_cycle}"
    --sample_diffusion.N_sample "${N_sample}"
    --sample_diffusion.N_step "${N_step}"
    --triangle_attention cuequivariance
    --triangle_multiplicative cuequivariance
    --use_template true
    --load_checkpoint_dir "${load_checkpoint_dir}"
)

torchrun "${torchrun_args[@]}"
