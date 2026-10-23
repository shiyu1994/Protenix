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

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep all old immune-heavy crop1024 settings identical to the validated db32
# entry point while locking only the per-target diffusion batch to 48.
export PROTENIX_TRAIN_CONTENT_PROFILE=immune_heavy_old_content
export PROTENIX_DIFFUSION_BATCH_PROFILE=48
export PROTENIX_MAX_NUC_TOKENS_PROFILE=256
export PROTENIX_PAIR_MSA_PROFILE=unpaired_shuffle
exec bash "${repo_dir}/run_msa2048_crop1024_nuc256_db32_chunk2048_fsdp2_finetune.sh" "$@"
