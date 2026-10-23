#!/usr/bin/env bash
set -euo pipefail

repo_dir="/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train"
benchmark_dir="/mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2"
ckpt_20260525="/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260525_112137/checkpoints"
ckpt_20260531="/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/checkpoints"
queue_log_dir="${benchmark_dir}/sab23h2_raw999_queue_logs"
mkdir -p "${queue_log_dir}"

declare -a runs=(
  "px_output_finetune_20260525_999_raw_template_5seeds|${ckpt_20260525}/999.pt|23361"
  "px_output_finetune_20260531_999_raw_template_5seeds|${ckpt_20260531}/999.pt|23362"
)

printf 'queue_started_at=%s\n' "$(date -Is)" | tee -a "${queue_log_dir}/queue.log"
for item in "${runs[@]}"; do
  IFS='|' read -r run_name checkpoint port <<< "${item}"
  dump_dir="${benchmark_dir}/${run_name}"
  log_path="${dump_dir}/run_logs/inference_$(date +%Y%m%d_%H%M%S)_queue.log"
  mkdir -p "${dump_dir}/run_logs"

  printf '\nrun_started_at=%s run_name=%s checkpoint=%s log=%s\n' \
    "$(date -Is)" "${run_name}" "${checkpoint}" "${log_path}" | tee -a "${queue_log_dir}/queue.log"

  RUN_NAME="${run_name}" \
  CHECKPOINT_PATH="${checkpoint}" \
  MASTER_PORT="${port}" \
  "${repo_dir}/run_sab23h2_infer_checkpoint.sh" > "${log_path}" 2>&1

  printf 'run_finished_at=%s run_name=%s status=ok\n' "$(date -Is)" "${run_name}" \
    | tee -a "${queue_log_dir}/queue.log"
done

printf 'queue_finished_at=%s\n' "$(date -Is)" | tee -a "${queue_log_dir}/queue.log"
