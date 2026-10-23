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
    echo "RUN_NAME is required for the disordered-v2 training launcher." >&2
    exit 2
fi
if [[ "${resolved_run_name}" != *"disordered_v2"* ]]; then
    echo "RUN_NAME must contain disordered_v2 for this launcher." >&2
    exit 2
fi
resolved_diffusion_mask="${PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS:-true}"
if [ "${resolved_diffusion_mask}" != "true" ] && [ "${resolved_diffusion_mask}" != "false" ]; then
    echo "PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS must be 'true' or 'false'; got '${resolved_diffusion_mask}'." >&2
    exit 2
fi
if [ "${resolved_diffusion_mask}" != "true" ]; then
    echo "The disordered-v2 initial-training launcher requires PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS=true." >&2
    echo "Create a separately named fine-tuning profile/contract to use all-molecule diffusion supervision." >&2
    exit 2
fi

export PROTENIX_LOCKED_DATA_PROFILE="weighted_pdb_afdb_mgnify_disordered_v2"
export PROTENIX_TRAINING_DATA_PROFILE="${PROTENIX_LOCKED_DATA_PROFILE}"
export PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS="true"

exec bash "${repo_dir}/run_crop384_db48_ddp_scratch_200k.sh" "$@"
