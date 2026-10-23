#!/usr/bin/env bash
# Strict full-state resume for checkpoints produced by the disordered-v4 stage.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${repo_dir}/scripts/runtime/sequence_template_cache_contract.sh"
resolved_run_name="${RUN_NAME:-${run_name:-}}"
if [ -z "${resolved_run_name}" ]; then
    echo "RUN_NAME is required for the disordered-v4 resume launcher." >&2
    exit 2
fi
if [[ "${resolved_run_name}" != *"disordered_v4"* ]]; then
    echo "RUN_NAME must contain disordered_v4." >&2
    exit 2
fi

lock_and_validate_sequence_template_cache

export PROTENIX_LOCKED_DATA_PROFILE="weighted_pdb_afdb_mgnify_short_long_disordered_v4"
export PROTENIX_TRAINING_DATA_PROFILE="${PROTENIX_LOCKED_DATA_PROFILE}"
export PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS=true

exec bash "${repo_dir}/resume_crop384_db48_ddp_scratch_200k.sh" "$@"
