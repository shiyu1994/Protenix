#!/usr/bin/env bash
# Params-only stage initialization of disordered-v4 from the immutable v8
# step-999 raw/EMA source pair.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${repo_dir}/scripts/runtime/sequence_template_cache_contract.sh"
if (( $# != 0 )); then
    echo "This locked v4 stage-init launcher accepts no command-line arguments." >&2
    exit 2
fi

resolved_run_name="${RUN_NAME:-${run_name:-}}"
if [ -z "${resolved_run_name}" ]; then
    echo "RUN_NAME is required." >&2
    exit 2
fi
for required_tag in disordered_v4 from_v8s999 v9_straggler; do
    if [[ "${resolved_run_name}" != *"${required_tag}"* ]]; then
        echo "RUN_NAME must contain ${required_tag}." >&2
        exit 2
    fi
done

lock_and_validate_sequence_template_cache

for variable_name in \
    PROTENIX_PDB_SAMPLE_WEIGHT \
    PROTENIX_SHORT_DISTILLATION_SAMPLE_WEIGHT \
    PROTENIX_LONG_DISTILLATION_SAMPLE_WEIGHT \
    PROTENIX_DISORDERED_SAMPLE_WEIGHT \
    CHECKPOINT CHECKPOINT_PATH LOAD_CHECKPOINT_PATH LOAD_EMA_CHECKPOINT_PATH \
    RESUME_RUN_DIR RESUME_STEP RESUME_RAW_CHECKPOINT RESUME_EMA_CHECKPOINT \
    PROTENIX_FULL_RESUME PROTENIX_STRICT_FULL_RESUME \
    PROTENIX_CROP384_STRICT_RESUME_INTERNAL \
    PROTENIX_CROP384_WARMSTART_INTERNAL; do
    if [ -n "${!variable_name-}" ]; then
        echo "Locked v4 stage-init rejects inherited ${variable_name}." >&2
        exit 2
    fi
done

source_run_dir="/mnt/project_abd/klp/proteinx_output/protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_msamix_q20_u30_p50_50k_v8_20260724_155026"
raw_checkpoint="${source_run_dir}/checkpoints/999.pt"
ema_checkpoint="${source_run_dir}/checkpoints/999_ema_0.999.pt"
raw_sha256="b92361c83de2bf3d300c082bb84a0b797f5b2d92837c8deead7a4f1b3da17ca8"
ema_sha256="a4ecad60c39cad6072e0e1093d6f1394e66142ef1b24f03b931f048a48f7b5f7"
source_training_contract_id="crop384_db48_ddp_scratch_50k_ws128_acc4_data_weightedpdb_only_v1_arch_pf64_cs384_dct768_pfckpt8_c48_msamix_q20_u30_p50_v8"
source_data_contract_sha256="835a4089f4b20b138c310be537b95dedcdd0ca65cb7a8fd39eddc460947c3a08"
source_architecture_sha256="cd1cbb658e352f0d56a2dcd65c5e0e95ae2e29c321f4fd0da9f79c6145edc34c"
source_pair_id="1220d08b9f62db65364a5e936e2e7d22577c72e71fcca4ae2a261f0045fb9001"

verify_full_sha256="${PROTENIX_WARMSTART_VERIFY_FULL_SHA256:-0}"
if [ "${verify_full_sha256}" != "0" ] && [ "${verify_full_sha256}" != "1" ]; then
    echo "PROTENIX_WARMSTART_VERIFY_FULL_SHA256 must be 0 or 1." >&2
    exit 2
fi
if [ "${DRY_RUN:-0}" != "1" ] && [ "${SKIP_PATH_CHECK:-0}" != "1" ]; then
    validator_args=(
        --raw "${raw_checkpoint}"
        --ema "${ema_checkpoint}"
    )
    if [ "${verify_full_sha256}" = "1" ]; then
        validator_args+=(--verify-sha256)
    fi
    /mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
        "${repo_dir}/scripts/runtime/validate_v8_step999_warmstart_pair.py" \
        "${validator_args[@]}"
fi

export PROTENIX_LOCKED_DATA_PROFILE="weighted_pdb_afdb_mgnify_short_long_disordered_v4"
export PROTENIX_TRAINING_DATA_PROFILE="${PROTENIX_LOCKED_DATA_PROFILE}"
export PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS=true
export PROTENIX_CROP384_WARMSTART_INTERNAL=1
export PROTENIX_INITIALIZATION_MODE=params_only_source_pair
export PROTENIX_WARMSTART_RAW_CHECKPOINT="${raw_checkpoint}"
export PROTENIX_WARMSTART_EMA_CHECKPOINT="${ema_checkpoint}"
export PROTENIX_WARMSTART_CHECKPOINT_STEP=999
export PROTENIX_WARMSTART_RAW_SHA256="${raw_sha256}"
export PROTENIX_WARMSTART_EMA_SHA256="${ema_sha256}"
export PROTENIX_WARMSTART_SOURCE_RUN_ID="$(basename "${source_run_dir}")"
export PROTENIX_WARMSTART_SOURCE_TRAINING_CONTRACT_ID="${source_training_contract_id}"
export PROTENIX_WARMSTART_SOURCE_DATA_CONTRACT_SHA256="${source_data_contract_sha256}"
export PROTENIX_WARMSTART_SOURCE_MODEL_ARCHITECTURE_SHA256="${source_architecture_sha256}"
export PROTENIX_WARMSTART_SOURCE_CHECKPOINT_PAIR_ID="${source_pair_id}"

echo "Locked disordered-v4 params-only stage initialization:"
echo "  source_run=${source_run_dir}"
echo "  source_step=999 raw=${raw_checkpoint}"
echo "  source_ema=${ema_checkpoint}"
echo "  imported=live_model:raw ema_shadow:ema"
echo "  reset=optimizer+scheduler+step+scaler+rank_rng+sampler_cursor"
echo "  target_weights=0.49,0.005,0.505,0.02"
echo "  template_cache=${PROTENIX_SEQUENCE_TEMPLATE_CACHE_ROOT}"
echo "  template_cache_mode/error_policy=${PROTENIX_SEQUENCE_TEMPLATE_CACHE_MODE}/${PROTENIX_SEQUENCE_TEMPLATE_CACHE_ERROR_POLICY}"

exec bash "${repo_dir}/run_crop384_db48_ddp_scratch_200k.sh"
