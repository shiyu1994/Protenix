#!/usr/bin/env bash
set -euo pipefail

test_run="${test_run:-${TEST_RUN:-899_ema_0.999}}"

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/mnt/project_abd/klp/miniconda3/bin/activate}"
CONDA_ENV="${CONDA_ENV:-protenix}"

export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/mnt/project_abd/public_databases/protenix.20260101}"
checkpoint_dir="${checkpoint_dir:-${CHECKPOINT_DIR:-/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260521_033600/checkpoints}}"
if [ -n "${load_checkpoint_dir:-${LOAD_CHECKPOINT_DIR:-}}" ]; then
    load_checkpoint_dir="${load_checkpoint_dir:-${LOAD_CHECKPOINT_DIR}}"
elif [[ "${test_run}" == /* ]]; then
    load_checkpoint_dir="${test_run}"
elif [[ "${test_run}" == *.pt ]]; then
    load_checkpoint_dir="${checkpoint_dir}/${test_run}"
else
    load_checkpoint_dir="${checkpoint_dir}/${test_run}.pt"
fi

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

N_sample="${N_sample:-${N_SAMPLE:-5}}"
N_step="${N_step:-${N_STEP:-200}}"
N_cycle="${N_cycle:-${N_CYCLE:-10}}"
seed="${seed:-${SEED:-101,102,103,104,105}}"
input_json_path="${input_json_path:-${INPUT_JSON_PATH:-./examples/examples_with_template/example_9fm7.json}}"
dump_dir="${dump_dir:-${DUMP_DIR:-/mnt/project_abd/klp/proteinx_infer_outputs/${test_run}}}"
model_name="${model_name:-${MODEL_NAME:-protenix-v2}}"
nproc_per_node="${nproc_per_node:-${NPROC_PER_NODE:-gpu}}"
master_port="${master_port:-${MASTER_PORT:-23333}}"

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

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf 'python=%s\n' "$(command -v python)"
    printf 'torchrun=%s\n' "$(command -v torchrun)"
    printf 'PROTENIX_ROOT_DIR=%s\n' "${PROTENIX_ROOT_DIR}"
    printf 'load_checkpoint_dir=%s\n' "${load_checkpoint_dir}"
    printf 'input_json_path=%s\n' "${input_json_path}"
    printf 'dump_dir=%s\n' "${dump_dir}"
    printf 'torchrun'
    printf ' %q' "${torchrun_args[@]}"
    printf '\n'
    exit 0
fi

torchrun "${torchrun_args[@]}"
