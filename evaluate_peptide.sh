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

# Usage: bash evaluate_peptide.sh [all|prepare|predict|score|watch] [--help] [options]
# Defaults: supplied 51-CIF test directory, official v2 weights, one GPU,
# five samples at seed 101, 200 diffusion steps, 10 recycles, no MSA/templates.
# Parallel evaluation: bash evaluate_peptide.sh all --gpus 0,1,2,3,4,5,6,7
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/mnt/project_abd/public_databases/protenix.20260101}"
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ ${1:-} == prepare || ${1:-} == score || ${1:-} == watch ]]; then export CUDA_VISIBLE_DEVICES=""; fi
export LAYERNORM_TYPE=torch
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ ! -x ${python_bin} ]]; then echo "Missing executable PYTHON_BIN: ${python_bin}" >&2; exit 1; fi
export PATH="$(dirname "${python_bin}"):${PATH}"
torch_lib="$("${python_bin}" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="${torch_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${repo_dir}"
exec "${python_bin}" -m scripts.evaluation.evaluate_peptide_pose "$@"
