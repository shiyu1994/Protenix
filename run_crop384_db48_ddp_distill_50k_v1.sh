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
resolved_run_name="${RUN_NAME:-${run_name:-}}"
if [ -z "${resolved_run_name}" ]; then
    echo "RUN_NAME is required for the distillation-v1 training launcher." >&2
    exit 2
fi
if [[ "${resolved_run_name}" != *"distill_v1"* ]]; then
    echo "RUN_NAME must contain distill_v1 for the distillation-v1 training launcher." >&2
    exit 2
fi

export PROTENIX_LOCKED_DATA_PROFILE="weighted_pdb_afdb_mgnify_distill_v1"
export PROTENIX_TRAINING_DATA_PROFILE="${PROTENIX_LOCKED_DATA_PROFILE}"

exec bash "${repo_dir}/run_crop384_db48_ddp_scratch_200k.sh" "$@"
