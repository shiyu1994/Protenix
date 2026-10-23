#!/usr/bin/env bash
set -euo pipefail

repo_dir="/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train"
out_dir="/mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/dockq_eval_immune_interfaces_raw999"
python_bin="/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python"

cd "${repo_dir}"
"${python_bin}" eval_sab23h2_dockq.py \
  --runs 20260525_999_raw,20260531_999_raw \
  --out-dir "${out_dir}" \
  --num-workers 32

python analyze_sab23h2_ranking_failure.py
