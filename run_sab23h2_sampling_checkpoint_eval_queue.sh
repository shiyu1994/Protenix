#!/usr/bin/env bash
set -euo pipefail

repo_dir="/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train"
benchmark_dir="/mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2"
python_bin="/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python"
steps_csv="${STEPS:-249,499,749,999,1249,1499,1749,1999}"
train_run_dirs_csv="${TRAIN_RUN_DIRS:?Set TRAIN_RUN_DIRS as label=/path,label2=/path2 or /path,/path2}"
out_dir="${DOCKQ_OUT_DIR:-${benchmark_dir}/dockq_eval_immune_interfaces_sampling_phase_a}"
topk_out_dir="${TOPK_OUT_DIR:-${benchmark_dir}/topk_dockq_analysis_sampling_phase_a}"
log_dir="${benchmark_dir}/sampling_checkpoint_eval_logs"
mkdir -p "${log_dir}"

IFS=',' read -r -a steps <<< "${steps_csv}"
IFS=',' read -r -a train_entries <<< "${train_run_dirs_csv}"

extra_runs_json="${log_dir}/extra_runs_$(date +%Y%m%d_%H%M%S).json"
run_labels=()
run_dirs=()

json_body="{"
first_json=1
port_base="${MASTER_PORT_BASE:-24400}"
port_offset=0

for entry in "${train_entries[@]}"; do
  if [[ "${entry}" == *=* ]]; then
    experiment_label="${entry%%=*}"
    train_run_dir="${entry#*=}"
  else
    train_run_dir="${entry}"
    experiment_label="$(basename "${train_run_dir}")"
  fi

  for step in "${steps[@]}"; do
    checkpoint_path="${train_run_dir}/checkpoints/${step}_ema_0.999.pt"
    if [ ! -f "${checkpoint_path}" ]; then
      printf 'skip_missing_checkpoint label=%s step=%s path=%s\n' \
        "${experiment_label}" "${step}" "${checkpoint_path}"
      continue
    fi

    run_label="${experiment_label}_${step}_ema"
    run_dir="px_output_${run_label}_0.999_template_5seeds"
    run_labels+=("${run_label}")
    run_dirs+=("${run_dir}")

    if [ "${first_json}" = "0" ]; then
      json_body+=","
    fi
    first_json=0
    json_body+="\"${run_label}\":\"${run_dir}\""

    infer_log="${log_dir}/infer_${run_label}_$(date +%Y%m%d_%H%M%S).log"
    master_port="$((port_base + port_offset))"
    port_offset="$((port_offset + 1))"

    printf 'planned_inference label=%s run_dir=%s checkpoint=%s log=%s\n' \
      "${run_label}" "${run_dir}" "${checkpoint_path}" "${infer_log}"
    if [ "${DRY_RUN:-0}" = "1" ]; then
      continue
    fi

    RUN_NAME="${run_dir}" \
    CHECKPOINT_PATH="${checkpoint_path}" \
    MASTER_PORT="${master_port}" \
    "${repo_dir}/run_sab23h2_infer_checkpoint.sh" > "${infer_log}" 2>&1
  done
done

json_body+="}"
printf '%s\n' "${json_body}" > "${extra_runs_json}"

if [ "${#run_labels[@]}" = "0" ]; then
  echo "No checkpoints found for evaluation." >&2
  exit 1
fi

labels_csv="$(IFS=','; echo "${run_labels[*]}")"
per_model_csvs=()

printf 'extra_runs_json=%s\n' "${extra_runs_json}"
printf 'run_labels=%s\n' "${labels_csv}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'dry_run_dockq command=%s eval_sab23h2_dockq_with_extra_runs.py --extra-runs-json %s --runs %s --out-dir %s --num-workers 32\n' \
    "${python_bin}" "${extra_runs_json}" "${labels_csv}" "${out_dir}"
  exit 0
fi

"${python_bin}" "${repo_dir}/eval_sab23h2_dockq_with_extra_runs.py" \
  --extra-runs-json "${extra_runs_json}" \
  --runs "${labels_csv}" \
  --out-dir "${out_dir}" \
  --num-workers "${DOCKQ_NUM_WORKERS:-32}"

python "${repo_dir}/analyze_sab23h2_ranking_failure.py"

per_model_csvs+=("${out_dir}/per_model_dockq.csv")
python "${repo_dir}/analyze_sab23h2_topk.py" \
  --per-model-csv "${per_model_csvs[@]}" \
  --out-dir "${topk_out_dir}"
