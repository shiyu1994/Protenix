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

# This experiment retains the validated old immune-heavy crop1024 workload,
# reduces the nucleic-acid crop cap from 256 to 128, uses 40 diffusion samples
# per target, and restores normal species-paired MSA without paired-hit
# shuffling. These exports deliberately overwrite inherited values from reused
# AIStudio shells.
export PROTENIX_TRAIN_CONTENT_PROFILE=immune_heavy_old_content
export PROTENIX_DIFFUSION_BATCH_PROFILE=40
export PROTENIX_MAX_NUC_TOKENS_PROFILE=128
export PROTENIX_PAIR_MSA_PROFILE=species_paired
exec bash "${repo_dir}/run_msa2048_crop1024_nuc256_db32_chunk2048_fsdp2_finetune.sh" "$@"
