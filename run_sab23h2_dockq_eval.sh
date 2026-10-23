#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train"
PYTHON_BIN="/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python"
OUT_DIR="/mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/dockq_eval_immune_interfaces"
LOG_DIR="${OUT_DIR}/run_logs"
WORKERS="${WORKERS:-32}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/dockq_eval_$(date +%Y%m%d_%H%M%S).log"

cd "${ROOT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "started_at=$(date --iso-8601=seconds)"
echo "workers=${WORKERS}"
echo "out_dir=${OUT_DIR}"
echo "log_file=${LOG_FILE}"

"${PYTHON_BIN}" eval_sab23h2_dockq.py \
  --num-workers "${WORKERS}" \
  --out-dir "${OUT_DIR}"

echo "finished_at=$(date --iso-8601=seconds)"
