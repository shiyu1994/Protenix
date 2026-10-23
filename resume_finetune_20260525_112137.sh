#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"

run_dir="/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260525_112137"
resume_step="${RESUME_STEP:-7999}"
default_checkpoint="${run_dir}/checkpoints/${resume_step}.pt"
default_ema_checkpoint="${run_dir}/checkpoints/${resume_step}_ema_0.999.pt"

if [ -z "${LOAD_CHECKPOINT_PATH+x}" ]; then
    export LOAD_CHECKPOINT_PATH="${default_checkpoint}"
fi
if [ -z "${LOAD_EMA_CHECKPOINT_PATH+x}" ]; then
    export LOAD_EMA_CHECKPOINT_PATH="${default_ema_checkpoint}"
fi
if [ -z "${run_name+x}" ] && [ -z "${RUN_NAME+x}" ]; then
    export run_name="protenix_finetune_20260101_weightedpdb_lr1e-5_resume${resume_step}"
fi

# runner/train.py restores optimizer, scheduler, and step from LOAD_CHECKPOINT_PATH
# unless skip_load_* flags are explicitly set. EMA weights initialize the EMA shadow.
echo "Resuming Protenix finetune:"
echo "  LOAD_CHECKPOINT_PATH=${LOAD_CHECKPOINT_PATH}"
echo "  LOAD_EMA_CHECKPOINT_PATH=${LOAD_EMA_CHECKPOINT_PATH}"
echo "  run_name=${run_name:-${RUN_NAME:-}}"

exec bash "${repo_dir}/finetune_submit.sh" "$@"
