#!/usr/bin/env bash
set -euo pipefail

repo_dir="/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train"
log_dir="/mnt/project_abd/klp/proteinx_output/sab23h2_sampling_finetune_logs"
mkdir -p "${log_dir}"

declare -a runs=(
  "sampling_conservative|0.4,0.4,0.2|29651"
  "sampling_immune_heavy|0.25,0.5,0.25|29652"
)

queue_log="${log_dir}/queue_$(date +%Y%m%d_%H%M%S).log"
printf 'queue_started_at=%s\n' "$(date -Is)" | tee -a "${queue_log}"

run_index=0
for item in "${runs[@]}"; do
  IFS='|' read -r experiment_name weights port <<< "${item}"
  run_timestamp="$(date +%Y%m%d_%H%M%S)"
  run_log="${log_dir}/${experiment_name}_${run_timestamp}.log"
  if [ -n "${MASTER_PORT:-}" ]; then
    run_master_port="$((MASTER_PORT + run_index))"
  else
    run_master_port="${port}"
  fi
  printf '\nrun_started_at=%s experiment=%s weights=%s log=%s\n' \
    "$(date -Is)" "${experiment_name}" "${weights}" "${run_log}" | tee -a "${queue_log}"

  EXPERIMENT_NAME="${experiment_name}" \
  TRAIN_SAMPLE_WEIGHTS="${weights}" \
  MASTER_PORT="${run_master_port}" \
  RUN_TIMESTAMP="${run_timestamp}" \
  "${repo_dir}/run_sab23h2_sampling_finetune.sh" > "${run_log}" 2>&1

  printf 'run_finished_at=%s experiment=%s status=ok\n' \
    "$(date -Is)" "${experiment_name}" | tee -a "${queue_log}"
  run_index="$((run_index + 1))"
done

printf 'queue_finished_at=%s\n' "$(date -Is)" | tee -a "${queue_log}"
