#!/usr/bin/env bash
set -euo pipefail

repo_dir="${REPO_DIR:-/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2}"
benchmark_dir="/mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2"
ckpt_dir="/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260611_100747/checkpoints"
queue_log_dir="${benchmark_dir}/sab23h2_checkpoint_queue_20260611_ema_logs"
mkdir -p "${queue_log_dir}"

count_completed_outputs() {
  local dump_dir="$1"
  if [ ! -d "${dump_dir}" ]; then
    printf '0\n'
    return
  fi
  find "${dump_dir}" -mindepth 2 -maxdepth 2 -type d -name 'seed_*' 2>/dev/null | wc -l
}

declare -a runs=(
  "px_output_finetune_20260611_999_ema_0.999_template_5seeds|${ckpt_dir}/999_ema_0.999.pt|25101"
  "px_output_finetune_20260611_1999_ema_0.999_template_5seeds|${ckpt_dir}/1999_ema_0.999.pt|25102"
  "px_output_finetune_20260611_2999_ema_0.999_template_5seeds|${ckpt_dir}/2999_ema_0.999.pt|25103"
  "px_output_finetune_20260611_3999_ema_0.999_template_5seeds|${ckpt_dir}/3999_ema_0.999.pt|25104"
  "px_output_finetune_20260611_4999_ema_0.999_template_5seeds|${ckpt_dir}/4999_ema_0.999.pt|25105"
  "px_output_finetune_20260611_5999_ema_0.999_template_5seeds|${ckpt_dir}/5999_ema_0.999.pt|25106"
  "px_output_finetune_20260611_6999_ema_0.999_template_5seeds|${ckpt_dir}/6999_ema_0.999.pt|25107"
  "px_output_finetune_20260611_7999_ema_0.999_template_5seeds|${ckpt_dir}/7999_ema_0.999.pt|25108"
  "px_output_finetune_20260611_8999_ema_0.999_template_5seeds|${ckpt_dir}/8999_ema_0.999.pt|25109"
  "px_output_finetune_20260611_9999_ema_0.999_template_5seeds|${ckpt_dir}/9999_ema_0.999.pt|25110"
  "px_output_finetune_20260611_10999_ema_0.999_template_5seeds|${ckpt_dir}/10999_ema_0.999.pt|25111"
  "px_output_finetune_20260611_11999_ema_0.999_template_5seeds|${ckpt_dir}/11999_ema_0.999.pt|25112"
  "px_output_finetune_20260611_12999_ema_0.999_template_5seeds|${ckpt_dir}/12999_ema_0.999.pt|25113"
  "px_output_finetune_20260611_13999_ema_0.999_template_5seeds|${ckpt_dir}/13999_ema_0.999.pt|25114"
  "px_output_finetune_20260611_14999_ema_0.999_template_5seeds|${ckpt_dir}/14999_ema_0.999.pt|25115"
)

printf 'queue_started_at=%s\n' "$(date -Is)" | tee -a "${queue_log_dir}/queue.log"
for item in "${runs[@]}"; do
  IFS='|' read -r run_name checkpoint port <<< "${item}"
  dump_dir="${benchmark_dir}/${run_name}"
  complete_count="$(count_completed_outputs "${dump_dir}")"

  if [ "${complete_count}" -ge 435 ]; then
    printf 'run_skipped_at=%s run_name=%s reason=complete count=%s\n' \
      "$(date -Is)" "${run_name}" "${complete_count}" | tee -a "${queue_log_dir}/queue.log"
    continue
  fi

  if [ ! -f "${checkpoint}" ]; then
    printf 'run_failed_at=%s run_name=%s reason=missing_checkpoint checkpoint=%s\n' \
      "$(date -Is)" "${run_name}" "${checkpoint}" | tee -a "${queue_log_dir}/queue.log"
    exit 1
  fi

  mkdir -p "${dump_dir}/run_logs"
  log_path="${dump_dir}/run_logs/inference_$(date +%Y%m%d_%H%M%S)_queue.log"

  printf '\nrun_started_at=%s run_name=%s checkpoint=%s log=%s existing_count=%s\n' \
    "$(date -Is)" "${run_name}" "${checkpoint}" "${log_path}" "${complete_count}" | tee -a "${queue_log_dir}/queue.log"

  REPO_DIR="${repo_dir}" \
  RUN_NAME="${run_name}" \
  CHECKPOINT_PATH="${checkpoint}" \
  MASTER_PORT="${port}" \
  TORCHRUN_STANDALONE=true \
  "${repo_dir}/run_sab23h2_infer_checkpoint.sh" > "${log_path}" 2>&1

  complete_count="$(count_completed_outputs "${dump_dir}")"
  printf 'run_finished_at=%s run_name=%s status=ok count=%s\n' \
    "$(date -Is)" "${run_name}" "${complete_count}" | tee -a "${queue_log_dir}/queue.log"
done

printf 'queue_finished_at=%s\n' "$(date -Is)" | tee -a "${queue_log_dir}/queue.log"
