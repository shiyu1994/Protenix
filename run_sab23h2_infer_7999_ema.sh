#!/usr/bin/env bash
set -euo pipefail

repo_dir="/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train"
benchmark_dir="/mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2"
dump_dir="${benchmark_dir}/px_output_finetune_7999_ema_0.999_template_5seeds"
input_json="${dump_dir}/input_jsons/sab23h2_final_updated.json"
checkpoint="/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260525_112137/checkpoints/7999_ema_0.999.pt"

source /mnt/project_abd/klp/miniconda3/bin/activate protenix
export PATH="/mnt/project_abd/klp/miniconda3/envs/protenix/bin:${PATH}"
hash -r

export PROTENIX_ROOT_DIR="/mnt/project_abd/public_databases/protenix.20260101"
export PYTHONPATH="${repo_dir}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

cd "${benchmark_dir}"

printf 'started_at=%s\n' "$(date -Is)"
printf 'python=%s\n' "$(command -v python)"
printf 'torchrun=%s\n' "$(command -v torchrun)"
printf 'repo_dir=%s\n' "${repo_dir}"
printf 'checkpoint=%s\n' "${checkpoint}"
printf 'input_json=%s\n' "${input_json}"
printf 'dump_dir=%s\n' "${dump_dir}"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"

torchrun --nproc_per_node 8 --master_port "${MASTER_PORT:-23346}" \
  "${repo_dir}/runner/inference.py" \
  --model_name protenix-v2 \
  --seeds 31415,31416,31417,31418,31419 \
  --dump_dir "${dump_dir}" \
  --input_json_path "${input_json}" \
  --model.N_cycle 10 \
  --sample_diffusion.N_sample 5 \
  --sample_diffusion.N_step 200 \
  --triangle_attention cuequivariance \
  --triangle_multiplicative cuequivariance \
  --use_msa true \
  --msa_pair_as_unpair true \
  --use_template true \
  --use_rna_msa false \
  --load_checkpoint_dir "${checkpoint}"

printf 'finished_at=%s\n' "$(date -Is)"
