#!/usr/bin/env bash
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail

# Use a new --output-dir, and supply --checkpoint explicitly for finetuned weights.
# All evaluate_peptide.sh options are accepted; later options override defaults.
# prepare is CPU-only; all launches predictions and scores them.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
stage=all
case "${1:-}" in
    all|prepare|predict|score|watch) stage=$1; shift ;;
esac
export PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/proteinix/bin/python}"
exec bash "$repo_dir/evaluate_peptide.sh" "$stage" \
    --output-dir /mnt/project_abd/shiyu/peptide_eval_training_aligned \
    --msa-map "${TEST_MSA_MAP:-/root/shiyu/peptide_pose_eval_v2_msa_map.json}" \
    --gpus "${GPUS:-0,1,2,3,4,5,6,7}" \
    --seeds 101,102,103,104,105 --samples 5 --steps 200 --cycles 10 --chunk-size 32 \
    "$@" --peptide-representation training-aligned --reference-seed "${REFERENCE_SEED:-42}"
