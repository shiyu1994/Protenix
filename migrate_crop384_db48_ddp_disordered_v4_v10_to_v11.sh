#!/usr/bin/env bash
# One-way full-state migration of the active disordered-v4 v10 run to the
# query-pin v11 contract. This is an explicit policy-stage boundary, not a
# trajectory-exact same-contract resume.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${repo_dir}/scripts/runtime/sequence_template_cache_contract.sh"

if (( $# != 0 )); then
    echo "This locked v10-to-v11 migration launcher accepts no command-line arguments." >&2
    exit 2
fi

resolved_run_name="${RUN_NAME:-${run_name:-}}"
if [ -z "${resolved_run_name}" ]; then
    echo "RUN_NAME is required for the disordered-v4 v10-to-v11 migration." >&2
    exit 2
fi
for required_tag in disordered_v4 v10_to_v11 msaquerypinle1024 v9_straggler; do
    if [[ "${resolved_run_name}" != *"${required_tag}"* ]]; then
        echo "RUN_NAME must contain ${required_tag}." >&2
        exit 2
    fi
done

lock_and_validate_sequence_template_cache

source_run_dir="/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200_20260726_061627"
source_run_id="$(basename "${source_run_dir}")"
source_training_contract_id="crop384_db48_ddp_scratch_50k_ws128_acc4_data_weightedpdb_afdb_mgnify_short_long_disordered_v4_pdb490_short005_long505_dis020_afq_mgn_cluster_replen_le200_gt200_v1_proteinonlydiffusion_v1_resolvedproteincrop_minloss4_v2_allmsashuffle_multichain_seg50_whole50_gap24_min9_v2_partialquery_max2_keep1_v1_arch_pf64_cs384_dct768_pfckpt8_c48_msamix_q10_u30_p60_msaraw_p10000_u10000_distillu16384_final16384_templatecache_top60u20id90_chaindummy_ckpt200_dlpersist1_prefetch2_deferredhealth_v10"
source_data_contract_sha256="cffda1e6694a9f293c8ed213d1b33e188d828d10a4d0eebdf66b2f428e75756c"
model_architecture_sha256="cd1cbb658e352f0d56a2dcd65c5e0e95ae2e29c321f4fd0da9f79c6145edc34c"
target_training_contract_id="crop384_db48_ddp_scratch_50k_ws128_acc4_data_weightedpdb_afdb_mgnify_short_long_disordered_v4_pdb490_short005_long505_dis020_afq_mgn_cluster_replen_le200_gt200_v1_proteinonlydiffusion_v1_resolvedproteincrop_minloss4_v2_allmsashuffle_multichain_seg50_whole50_gap24_min9_v2_partialquery_max2_keep1_v1_arch_pf64_cs384_dct768_pfckpt8_c48_msamix_q10_u30_p60_msaraw_p10000_u10000_distillu16384_final16384_msaquerypinle1024_templatecache_top60u20id90_chaindummy_ckpt200_dlpersist1_prefetch2_deferredhealth_v11"
target_data_contract_sha256="6c98c5cab75197b7e4d41b7ce5287af65ca3f750a707ec69a1bdaf29ed24b8de"
migration_mode="msa_query_pin_v10_to_v11_full_state_v1"

for variable_name in \
    CHECKPOINT checkpoint CHECKPOINT_PATH checkpoint_path \
    LOAD_CHECKPOINT_PATH load_checkpoint_path \
    LOAD_EMA_CHECKPOINT_PATH load_ema_checkpoint_path \
    RESUME_RAW_CHECKPOINT RESUME_EMA_CHECKPOINT \
    PROTENIX_RESUME_RAW_CHECKPOINT PROTENIX_RESUME_EMA_CHECKPOINT \
    PROTENIX_RESUME_CHECKPOINT_STEP PROTENIX_RESUME_TARGET_MAX_STEPS \
    PROTENIX_CROP384_STRICT_RESUME_INTERNAL \
    PROTENIX_CROP384_WARMSTART_INTERNAL \
    PROTENIX_FULL_RESUME PROTENIX_STRICT_FULL_RESUME \
    PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION \
    PROTENIX_MIGRATION_SOURCE_TRAINING_CONTRACT_ID \
    PROTENIX_MIGRATION_SOURCE_DATA_CONTRACT_SHA256 \
    PROTENIX_MIGRATION_SOURCE_MODEL_ARCHITECTURE_SHA256 \
    PROTENIX_MIGRATION_SOURCE_RUN_ID \
    PROTENIX_MIGRATION_TARGET_TRAINING_CONTRACT_ID \
    PROTENIX_MIGRATION_TARGET_DATA_CONTRACT_SHA256 \
    PROTENIX_TRAINING_CONTRACT_ID PROTENIX_TRAINING_DATA_PROFILE \
    PROTENIX_LOCKED_DATA_PROFILE; do
    if [ -n "${!variable_name-}" ]; then
        echo "Locked v10-to-v11 migration rejects inherited ${variable_name}." >&2
        exit 2
    fi
done

if [ -n "${RESUME_RUN_DIR:-}" ] \
    && [ "$(realpath -m "${RESUME_RUN_DIR}")" != "${source_run_dir}" ]; then
    echo "This migration locks RESUME_RUN_DIR=${source_run_dir}." >&2
    exit 2
fi

resume_step="${RESUME_STEP:-9999}"
if [[ ! "${resume_step}" =~ ^[0-9]+$ ]]; then
    echo "RESUME_STEP must be a non-negative integer." >&2
    exit 2
fi
next_step=$((resume_step + 1))
if (( next_step % 200 != 0 )); then
    echo "RESUME_STEP must identify a complete 200-update checkpoint pair (step+1 divisible by 200)." >&2
    exit 2
fi

export RESUME_RUN_DIR="${source_run_dir}"
export RESUME_STEP="${resume_step}"
export PROTENIX_LOCKED_DATA_PROFILE="weighted_pdb_afdb_mgnify_short_long_disordered_v4"
export PROTENIX_TRAINING_DATA_PROFILE="${PROTENIX_LOCKED_DATA_PROFILE}"
export PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS=true
export PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION="${migration_mode}"

echo "Locked disordered-v4 v10-to-v11 full-state migration:"
echo "  migration=${migration_mode} trajectory_exact=false fail_closed=true"
echo "  source_run=${source_run_dir}"
echo "  source_run_id=${source_run_id}"
echo "  source_step=${resume_step} target_start_step=${next_step}"
echo "  source_training_contract=${source_training_contract_id}"
echo "  source_data_contract_sha256=${source_data_contract_sha256}"
echo "  target_training_contract=${target_training_contract_id}"
echo "  target_data_contract_sha256=${target_data_contract_sha256}"
echo "  model_architecture_sha256=${model_architecture_sha256} unchanged=true"
echo "  restored=model+ema+optimizer+scheduler+step+scaler+rank_rng+sampler_cursor"
echo "  policy_change=MSA_query_row_pin_index0_when_K_le_1024 depth_remains_K"
echo "  absolute_target_updates=50000 checkpoint_interval=200 topology=16x8"

exec bash "${repo_dir}/resume_crop384_db48_ddp_scratch_50k.sh"
