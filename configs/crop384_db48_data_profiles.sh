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

# Named data profiles for the locked crop384/DB48 initial-training launcher.
#
# Keep dataset definitions and paths in configs/configs_data.py. A profile only
# selects registered dataset names, their outer sampler masses, and any
# profile-specific CLI overrides. This independent v2 branch adds SAbDab to
# the legacy four-child profile at its explicitly authorized fresh-data boundary.
_crop384_require_positive_decimal() {
    local variable_name="$1"
    local value="$2"

    if [[ ! "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
        || [[ "${value}" =~ ^0+([.]0+)?$ ]]; then
        echo "${variable_name} must be a positive decimal number; got '${value}'." >&2
        return 2
    fi
}

_crop384_require_boolean() {
    local variable_name="$1"
    local value="$2"

    if [[ "${value}" != "true" && "${value}" != "false" ]]; then
        echo "${variable_name} must be 'true' or 'false'; got '${value}'." >&2
        return 2
    fi
}

resolve_crop384_db48_data_profile() {
    local requested_profile="$1"
    CROP384_DATA_PROFILE_QUERY_ONLY_DESCRIPTION="one_global_target_row protein_and_rna_homolog_io_skipped"
    CROP384_DATA_PROFILE_MSA_UNPAIRED_DESCRIPTION="legacy_true/true paired_hits_to_chain_unpaired_pool"
    # Historical launchers used these values directly in the shared core
    # command and contract ID.  Profiles may override them, but keeping the
    # defaults here preserves every existing v1/v2 launch contract.
    CROP384_DATA_PROFILE_QUERY_ONLY_MSA_PROBABILITY="0.2"
    CROP384_DATA_PROFILE_PAIR_MSA_AS_UNPAIRED_PROBABILITY="0.3"
    CROP384_DATA_PROFILE_SPECIES_PAIRED_MSA_PROBABILITY="0.5"
    CROP384_DATA_PROFILE_MSA_MIX_CONTRACT_TAG="q20_u30_p50"
    CROP384_DATA_PROFILE_MSA_RAW_CONTRACT_TAG="p10000_u10000_final16384"
    CROP384_DATA_PROFILE_DISTILL_UNPAIRED_MSA_DEPTH=""

    case "${requested_profile}" in
        weighted_pdb_only_v1)
            CROP384_DATA_PROFILE_NAME="weighted_pdb_only_v1"
            CROP384_DATA_PROFILE_CONTRACT_TAG="weightedpdb_only_v1"
            CROP384_DATA_PROFILE_TRAIN_SETS="weightedPDB_before210930_v20260101"
            CROP384_DATA_PROFILE_TRAIN_SAMPLE_WEIGHTS="1.0"
            CROP384_DATA_PROFILE_RUNTIME_INDEX_DIR=""
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS=""
            CROP384_DATA_PROFILE_ARGS=(
                --data.weightedPDB_before210930_v20260101.cropping_configs.max_nuc_tokens
                128
            )
            ;;
        weighted_pdb_afdb_mgnify_distill_v1)
            local pdb_sample_weight="${PROTENIX_PDB_SAMPLE_WEIGHT:-0.5}"
            local distillation_sample_weight="${PROTENIX_DISTILLATION_SAMPLE_WEIGHT:-0.5}"
            local default_runtime_index_dir="/mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_runtime_index_20260725"
            local runtime_index_dir="${PROTENIX_DISTILLATION_RUNTIME_INDEX_DIR:-${default_runtime_index_dir}}"

            _crop384_require_positive_decimal \
                PROTENIX_PDB_SAMPLE_WEIGHT \
                "${pdb_sample_weight}" || return
            _crop384_require_positive_decimal \
                PROTENIX_DISTILLATION_SAMPLE_WEIGHT \
                "${distillation_sample_weight}" || return

            CROP384_DATA_PROFILE_NAME="weighted_pdb_afdb_mgnify_distill_v1"
            CROP384_DATA_PROFILE_CONTRACT_TAG="weightedpdb_afdb_mgnify_distill_v1"
            CROP384_DATA_PROFILE_TRAIN_SETS="weightedPDB_before210930_v20260101,afdb_mgnify_distillation_v20260725"
            CROP384_DATA_PROFILE_TRAIN_SAMPLE_WEIGHTS="${pdb_sample_weight},${distillation_sample_weight}"
            CROP384_DATA_PROFILE_RUNTIME_INDEX_DIR="${runtime_index_dir}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS="${runtime_index_dir}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${runtime_index_dir}/manifest.json"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${runtime_index_dir}/roots.json"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${runtime_index_dir}/row_offsets.u64"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${runtime_index_dir}/source_root_ids.u8"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${runtime_index_dir}/cif_relpaths.bin"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${runtime_index_dir}/sampling"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${runtime_index_dir}/sampling/cluster_ids.u32"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${runtime_index_dir}/sampling/first_row_ids.u64"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${runtime_index_dir}/sampling/cluster_sizes.u32"
            CROP384_DATA_PROFILE_ARGS=(
                --data.weightedPDB_before210930_v20260101.cropping_configs.max_nuc_tokens
                128
                --data.afdb_mgnify_distillation_v20260725.base_info.runtime_index_dir
                "${runtime_index_dir}"
                --data.afdb_mgnify_distillation_v20260725.base_info.source_roots_fpath
                "${runtime_index_dir}/roots.json"
                --data.afdb_mgnify_distillation_v20260725.cropping_configs.max_nuc_tokens
                128
            )
            ;;
        weighted_pdb_afdb_mgnify_disordered_v2)
            local pdb_sample_weight="${PROTENIX_PDB_SAMPLE_WEIGHT:-0.5}"
            local distillation_sample_weight="${PROTENIX_DISTILLATION_SAMPLE_WEIGHT:-0.5}"
            local disordered_sample_weight="${PROTENIX_DISORDERED_SAMPLE_WEIGHT:-0.02}"
            local default_runtime_index_dir="/mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_runtime_index_20260725"
            local runtime_index_dir="${PROTENIX_DISTILLATION_RUNTIME_INDEX_DIR:-${default_runtime_index_dir}}"
            local default_disordered_root="/mnt/project_abd_hs/klp/databases/pdb_disordered"
            local disordered_root="${PROTENIX_OPENFOLD3_DISORDERED_ROOT_DIR:-${default_disordered_root}}"
            local disordered_index="${disordered_root}/protenix/indices.csv.gz"
            local disordered_success="${disordered_root}/protenix/_SUCCESS.json"
            local disordered_structure_dir="${disordered_root}/structures/structures"
            local disordered_alignment_dir="${disordered_root}/alignment_arrays"
            local disordered_alignment_main="${disordered_alignment_dir}/alignment_arrays"
            local disable_non_protein_diffusion_weights
            disable_non_protein_diffusion_weights="${PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS:-true}"

            _crop384_require_positive_decimal \
                PROTENIX_PDB_SAMPLE_WEIGHT \
                "${pdb_sample_weight}" || return
            _crop384_require_positive_decimal \
                PROTENIX_DISTILLATION_SAMPLE_WEIGHT \
                "${distillation_sample_weight}" || return
            _crop384_require_positive_decimal \
                PROTENIX_DISORDERED_SAMPLE_WEIGHT \
                "${disordered_sample_weight}" || return
            _crop384_require_boolean \
                PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS \
                "${disable_non_protein_diffusion_weights}" || return

            local diffusion_mask_contract_tag="allmoldiffusion_v1"
            if [[ "${disable_non_protein_diffusion_weights}" == "true" ]]; then
                diffusion_mask_contract_tag="proteinonlydiffusion_v1"
            fi
            local protein_crop_contract_tag="resolvedproteincrop_minloss4_v2"
            local msa_segment_contract_tag="allmsasegshuffle_gap24_min9_v1"
            local query_only_contract_tag="partialquery_max2_keep1_v1"
            CROP384_DATA_PROFILE_QUERY_ONLY_DESCRIPTION="monomer_global complex_crop_physical_protein_asym_partial_max2_keep1 occurrence_seeded_given_crop"
            CROP384_DATA_PROFILE_MSA_UNPAIRED_DESCRIPTION="all_protein_rows crop_segments_gap24_min9 query_fixed occurrence_seeded"

            export PROTENIX_OPENFOLD3_DISORDERED_ROOT_DIR="${disordered_root}"
            CROP384_DATA_PROFILE_NAME="weighted_pdb_afdb_mgnify_disordered_v2"
            CROP384_DATA_PROFILE_CONTRACT_TAG="weightedpdb_afdb_mgnify_disordered_v2_${diffusion_mask_contract_tag}_${protein_crop_contract_tag}_${msa_segment_contract_tag}_${query_only_contract_tag}"
            CROP384_DATA_PROFILE_TRAIN_SETS="weightedPDB_before210930_v20260101,afdb_mgnify_distillation_v20260725,openfold3_disordered_v20260725"
            CROP384_DATA_PROFILE_TRAIN_SAMPLE_WEIGHTS="${pdb_sample_weight},${distillation_sample_weight},${disordered_sample_weight}"
            CROP384_DATA_PROFILE_RUNTIME_INDEX_DIR="${runtime_index_dir}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS="${runtime_index_dir}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${runtime_index_dir}/manifest.json"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${runtime_index_dir}/roots.json"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${runtime_index_dir}/row_offsets.u64"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${runtime_index_dir}/source_root_ids.u8"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${runtime_index_dir}/cif_relpaths.bin"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${runtime_index_dir}/sampling"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${runtime_index_dir}/sampling/cluster_ids.u32"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${runtime_index_dir}/sampling/first_row_ids.u64"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${runtime_index_dir}/sampling/cluster_sizes.u32"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${disordered_index}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${disordered_success}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${disordered_root}/protenix/manifest.json"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${disordered_structure_dir}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${disordered_alignment_dir}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${disordered_alignment_main}"
            CROP384_DATA_PROFILE_ARGS=(
                --data.msa.enable_virtual_segment_msa_shuffle
                true
                --data.msa.virtual_segment_residue_gap
                24
                --data.msa.virtual_segment_min_length
                9
                --data.msa.enable_crop_aware_partial_query_only
                true
                --data.msa.partial_query_only_max_chains
                2
                --data.weightedPDB_before210930_v20260101.base_info.min_diffusion_supervised_tokens
                4
                --data.weightedPDB_before210930_v20260101.cropping_configs.max_nuc_tokens
                128
                --data.afdb_mgnify_distillation_v20260725.base_info.min_diffusion_supervised_tokens
                4
                --data.afdb_mgnify_distillation_v20260725.base_info.require_resolved_protein_crop
                true
                --data.afdb_mgnify_distillation_v20260725.base_info.runtime_index_dir
                "${runtime_index_dir}"
                --data.afdb_mgnify_distillation_v20260725.base_info.source_roots_fpath
                "${runtime_index_dir}/roots.json"
                --data.afdb_mgnify_distillation_v20260725.cropping_configs.max_nuc_tokens
                128
                --data.afdb_mgnify_distillation_v20260725.augmentation.pair_msa_as_unpaired_probability
                0.3
                --data.openfold3_disordered_v20260725.base_info.indices_fpath
                "${disordered_index}"
                --data.openfold3_disordered_v20260725.base_info.dataset_success_fpath
                "${disordered_success}"
                --data.openfold3_disordered_v20260725.base_info.dataset_manifest_fpath
                "${disordered_root}/protenix/manifest.json"
                --data.openfold3_disordered_v20260725.base_info.mmcif_dir
                "${disordered_structure_dir}"
                --data.openfold3_disordered_v20260725.base_info.bioassembly_dict_dir
                "${disordered_structure_dir}"
                --data.openfold3_disordered_v20260725.base_info.disable_non_protein_diffusion_weights
                "${disable_non_protein_diffusion_weights}"
                --data.openfold3_disordered_v20260725.base_info.require_resolved_protein_crop
                true
                --data.openfold3_disordered_v20260725.base_info.min_diffusion_supervised_tokens
                4
                --data.openfold3_disordered_v20260725.cropping_configs.max_nuc_tokens
                128
                --data.openfold3_disordered_v20260725.augmentation.pair_msa_as_unpaired_probability
                0.3
            )
            ;;
        weighted_pdb_afdb_mgnify_short_long_disordered_v3|\
        weighted_pdb_afdb_mgnify_short_long_disordered_v4)
            local profile_name="weighted_pdb_afdb_mgnify_short_long_disordered_v3"
            local profile_contract_prefix="weightedpdb_afdb_mgnify_short_long_disordered_v3"
            local default_pdb_sample_weight="0.5"
            local default_short_sample_weight="0.005"
            local default_long_sample_weight="0.495"
            local default_disordered_sample_weight="0.02"
            if [[ "${requested_profile}" == \
                "weighted_pdb_afdb_mgnify_short_long_disordered_v4" ]]; then
                profile_name="${requested_profile}"
                profile_contract_prefix="weightedpdb_afdb_mgnify_short_long_disordered_v4_pdb490_short005_long505_dis020"
                default_pdb_sample_weight="0.49"
                default_long_sample_weight="0.505"
            fi
            local pdb_sample_weight="${PROTENIX_PDB_SAMPLE_WEIGHT:-${default_pdb_sample_weight}}"
            local short_sample_weight="${PROTENIX_SHORT_DISTILLATION_SAMPLE_WEIGHT:-${default_short_sample_weight}}"
            local long_sample_weight="${PROTENIX_LONG_DISTILLATION_SAMPLE_WEIGHT:-${default_long_sample_weight}}"
            local disordered_sample_weight="${PROTENIX_DISORDERED_SAMPLE_WEIGHT:-${default_disordered_sample_weight}}"
            local default_short_runtime_index_dir="/mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_short_le200_runtime_index_20260725"
            local default_long_runtime_index_dir="/mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_long_gt200_runtime_index_20260725"
            local short_runtime_index_dir="${PROTENIX_SHORT_DISTILLATION_RUNTIME_INDEX_DIR:-${default_short_runtime_index_dir}}"
            local long_runtime_index_dir="${PROTENIX_LONG_DISTILLATION_RUNTIME_INDEX_DIR:-${default_long_runtime_index_dir}}"
            local default_disordered_root="/mnt/project_abd_hs/klp/databases/pdb_disordered"
            local disordered_root="${PROTENIX_OPENFOLD3_DISORDERED_ROOT_DIR:-${default_disordered_root}}"
            local disordered_index="${disordered_root}/protenix/indices.csv.gz"
            local disordered_success="${disordered_root}/protenix/_SUCCESS.json"
            local disordered_structure_dir="${disordered_root}/structures/structures"
            local disordered_alignment_dir="${disordered_root}/alignment_arrays"
            local disordered_alignment_main="${disordered_alignment_dir}/alignment_arrays"
            local disable_non_protein_diffusion_weights
            disable_non_protein_diffusion_weights="${PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS:-true}"

            _crop384_require_positive_decimal \
                PROTENIX_PDB_SAMPLE_WEIGHT \
                "${pdb_sample_weight}" || return
            _crop384_require_positive_decimal \
                PROTENIX_SHORT_DISTILLATION_SAMPLE_WEIGHT \
                "${short_sample_weight}" || return
            _crop384_require_positive_decimal \
                PROTENIX_LONG_DISTILLATION_SAMPLE_WEIGHT \
                "${long_sample_weight}" || return
            _crop384_require_positive_decimal \
                PROTENIX_DISORDERED_SAMPLE_WEIGHT \
                "${disordered_sample_weight}" || return
            _crop384_require_boolean \
                PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS \
                "${disable_non_protein_diffusion_weights}" || return
            if [[ "${requested_profile}" == \
                "weighted_pdb_afdb_mgnify_short_long_disordered_v4" ]] \
                && [[ "${pdb_sample_weight},${short_sample_weight},${long_sample_weight},${disordered_sample_weight}" \
                    != "0.49,0.005,0.505,0.02" ]]; then
                echo "${requested_profile} locks raw weights=0.49,0.005,0.505,0.02." >&2
                return 2
            fi

            local diffusion_mask_contract_tag="allmoldiffusion_v1"
            if [[ "${disable_non_protein_diffusion_weights}" == "true" ]]; then
                diffusion_mask_contract_tag="proteinonlydiffusion_v1"
            fi
            local length_partition_contract_tag="afq_mgn_cluster_replen_le200_gt200_v1"
            local protein_crop_contract_tag="resolvedproteincrop_minloss4_v2"
            local msa_segment_contract_tag="allmsashuffle_multichain_seg50_whole50_gap24_min9_v2"
            local query_only_contract_tag="partialquery_max2_keep1_v1"

            CROP384_DATA_PROFILE_QUERY_ONLY_MSA_PROBABILITY="0.1"
            CROP384_DATA_PROFILE_PAIR_MSA_AS_UNPAIRED_PROBABILITY="0.3"
            CROP384_DATA_PROFILE_SPECIES_PAIRED_MSA_PROBABILITY="0.6"
            CROP384_DATA_PROFILE_MSA_MIX_CONTRACT_TAG="q10_u30_p60"
            CROP384_DATA_PROFILE_MSA_RAW_CONTRACT_TAG="p10000_u10000_distillu16384_final16384"
            CROP384_DATA_PROFILE_DISTILL_UNPAIRED_MSA_DEPTH="16384"
            CROP384_DATA_PROFILE_QUERY_ONLY_DESCRIPTION="monomer_global complex_crop_physical_protein_asym_partial_max2_keep1 occurrence_seeded_given_crop"
            CROP384_DATA_PROFILE_MSA_UNPAIRED_DESCRIPTION="all_protein_rows crop_visible_multichain_segment50_whole_physical_chain50 monomer_segment gap24_min9 query_fixed occurrence_seeded"

            export PROTENIX_OPENFOLD3_DISORDERED_ROOT_DIR="${disordered_root}"
            export PROTENIX_SHORT_DISTILLATION_RUNTIME_INDEX_DIR="${short_runtime_index_dir}"
            export PROTENIX_LONG_DISTILLATION_RUNTIME_INDEX_DIR="${long_runtime_index_dir}"
            CROP384_DATA_PROFILE_NAME="${profile_name}"
            CROP384_DATA_PROFILE_CONTRACT_TAG="${profile_contract_prefix}_${length_partition_contract_tag}_${diffusion_mask_contract_tag}_${protein_crop_contract_tag}_${msa_segment_contract_tag}_${query_only_contract_tag}"
            CROP384_DATA_PROFILE_TRAIN_SETS="weightedPDB_before210930_v20260101,afdb_mgnify_short_distillation_v20260725,afdb_mgnify_long_distillation_v20260725,openfold3_disordered_v20260725"
            CROP384_DATA_PROFILE_TRAIN_SAMPLE_WEIGHTS="${pdb_sample_weight},${short_sample_weight},${long_sample_weight},${disordered_sample_weight}"
            CROP384_DATA_PROFILE_RUNTIME_INDEX_DIR="${short_runtime_index_dir},${long_runtime_index_dir}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS="${short_runtime_index_dir}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${short_runtime_index_dir}/manifest.json"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${short_runtime_index_dir}/partition_pair.json"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${short_runtime_index_dir}/roots.json"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${short_runtime_index_dir}/row_offsets.u64"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${short_runtime_index_dir}/source_root_ids.u8"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${short_runtime_index_dir}/cif_relpaths.bin"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${short_runtime_index_dir}/sampling/cluster_ids.u32"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${short_runtime_index_dir}/sampling/first_row_ids.u64"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${short_runtime_index_dir}/sampling/cluster_sizes.u32"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${long_runtime_index_dir}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${long_runtime_index_dir}/manifest.json"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${long_runtime_index_dir}/partition_pair.json"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${long_runtime_index_dir}/roots.json"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${long_runtime_index_dir}/row_offsets.u64"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${long_runtime_index_dir}/source_root_ids.u8"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${long_runtime_index_dir}/cif_relpaths.bin"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${long_runtime_index_dir}/sampling/cluster_ids.u32"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${long_runtime_index_dir}/sampling/first_row_ids.u64"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${long_runtime_index_dir}/sampling/cluster_sizes.u32"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${disordered_index}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${disordered_success}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${disordered_root}/protenix/manifest.json"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${disordered_structure_dir}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${disordered_alignment_dir}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${disordered_alignment_main}"
            CROP384_DATA_PROFILE_ARGS=(
                --data.msa.enable_virtual_segment_msa_shuffle
                true
                --data.msa.virtual_segment_residue_gap
                24
                --data.msa.virtual_segment_min_length
                9
                --data.msa.enable_multichain_msa_shuffle_variant_sampling
                true
                --data.msa.multichain_msa_shuffle_segment_probability
                0.5
                --data.msa.enable_crop_aware_partial_query_only
                true
                --data.msa.partial_query_only_max_chains
                2
                --data.weightedPDB_before210930_v20260101.base_info.min_diffusion_supervised_tokens
                4
                --data.weightedPDB_before210930_v20260101.cropping_configs.max_nuc_tokens
                128
                --data.afdb_mgnify_short_distillation_v20260725.base_info.runtime_index_dir
                "${short_runtime_index_dir}"
                --data.afdb_mgnify_short_distillation_v20260725.base_info.source_roots_fpath
                "${short_runtime_index_dir}/roots.json"
                --data.afdb_mgnify_short_distillation_v20260725.cropping_configs.max_nuc_tokens
                128
                --data.afdb_mgnify_short_distillation_v20260725.msa.max_input_unpaired_msa_depth
                16384
                --data.afdb_mgnify_short_distillation_v20260725.augmentation.query_only_msa_probability
                0.1
                --data.afdb_mgnify_short_distillation_v20260725.augmentation.pair_msa_as_unpaired_probability
                0.3
                --data.afdb_mgnify_long_distillation_v20260725.base_info.runtime_index_dir
                "${long_runtime_index_dir}"
                --data.afdb_mgnify_long_distillation_v20260725.base_info.source_roots_fpath
                "${long_runtime_index_dir}/roots.json"
                --data.afdb_mgnify_long_distillation_v20260725.cropping_configs.max_nuc_tokens
                128
                --data.afdb_mgnify_long_distillation_v20260725.msa.max_input_unpaired_msa_depth
                16384
                --data.afdb_mgnify_long_distillation_v20260725.augmentation.query_only_msa_probability
                0.1
                --data.afdb_mgnify_long_distillation_v20260725.augmentation.pair_msa_as_unpaired_probability
                0.3
                --data.openfold3_disordered_v20260725.base_info.indices_fpath
                "${disordered_index}"
                --data.openfold3_disordered_v20260725.base_info.dataset_success_fpath
                "${disordered_success}"
                --data.openfold3_disordered_v20260725.base_info.dataset_manifest_fpath
                "${disordered_root}/protenix/manifest.json"
                --data.openfold3_disordered_v20260725.base_info.mmcif_dir
                "${disordered_structure_dir}"
                --data.openfold3_disordered_v20260725.base_info.bioassembly_dict_dir
                "${disordered_structure_dir}"
                --data.openfold3_disordered_v20260725.base_info.disable_non_protein_diffusion_weights
                "${disable_non_protein_diffusion_weights}"
                --data.openfold3_disordered_v20260725.base_info.require_resolved_protein_crop
                true
                --data.openfold3_disordered_v20260725.base_info.min_diffusion_supervised_tokens
                4
                --data.openfold3_disordered_v20260725.cropping_configs.max_nuc_tokens
                128
                --data.openfold3_disordered_v20260725.augmentation.query_only_msa_probability
                0.1
                --data.openfold3_disordered_v20260725.augmentation.pair_msa_as_unpaired_probability
                0.3
            )
            ;;
        weighted_pdb_legacy_residual_ge200_disordered_v5)
            local residual_runtime_index_dir
            residual_runtime_index_dir="${PROTENIX_LEGACY_RESIDUAL_RUNTIME_INDEX_DIR:-/mnt/project_abd_hs/klp/databases/afdb_mgnify_legacy_residual_not_ge200_runtime_index_20260804}"
            local ge200_runtime_index_dir
            ge200_runtime_index_dir="${PROTENIX_GE200_STRUCTURAL_RUNTIME_INDEX_DIR:-/mnt/project_abd_hs/klp/databases/all_sources_ge200_structural_runtime_index_20260804}"
            local default_disordered_root="/mnt/project_abd_hs/klp/databases/pdb_disordered"
            local disordered_root="${PROTENIX_OPENFOLD3_DISORDERED_ROOT_DIR:-${default_disordered_root}}"
            local disordered_index="${disordered_root}/protenix/indices.csv.gz"
            local disordered_success="${disordered_root}/protenix/_SUCCESS.json"
            local disordered_structure_dir="${disordered_root}/structures/structures"
            local disordered_alignment_dir="${disordered_root}/alignment_arrays"
            local disordered_alignment_main="${disordered_alignment_dir}/alignment_arrays"
            local disable_non_protein_diffusion_weights
            disable_non_protein_diffusion_weights="${PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS:-true}"
            _crop384_require_boolean \
                PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS \
                "${disable_non_protein_diffusion_weights}" || return

            CROP384_DATA_PROFILE_QUERY_ONLY_MSA_PROBABILITY="0.1"
            CROP384_DATA_PROFILE_PAIR_MSA_AS_UNPAIRED_PROBABILITY="0.3"
            CROP384_DATA_PROFILE_SPECIES_PAIRED_MSA_PROBABILITY="0.6"
            CROP384_DATA_PROFILE_MSA_MIX_CONTRACT_TAG="q10_u30_p60"
            CROP384_DATA_PROFILE_MSA_RAW_CONTRACT_TAG="p10000_u10000_distillu16384_final16384"
            CROP384_DATA_PROFILE_DISTILL_UNPAIRED_MSA_DEPTH="16384"
            CROP384_DATA_PROFILE_QUERY_ONLY_DESCRIPTION="monomer_global complex_crop_physical_protein_asym_partial_max2_keep1 occurrence_seeded_given_crop"
            CROP384_DATA_PROFILE_MSA_UNPAIRED_DESCRIPTION="all_protein_rows crop_visible_multichain_segment50_whole_physical_chain50 monomer_segment gap24_min9 query_fixed occurrence_seeded"

            export PROTENIX_OPENFOLD3_DISORDERED_ROOT_DIR="${disordered_root}"
            export PROTENIX_LEGACY_RESIDUAL_RUNTIME_INDEX_DIR="${residual_runtime_index_dir}"
            export PROTENIX_GE200_STRUCTURAL_RUNTIME_INDEX_DIR="${ge200_runtime_index_dir}"
            CROP384_DATA_PROFILE_NAME="weighted_pdb_legacy_residual_ge200_disordered_v5"
            CROP384_DATA_PROFILE_CONTRACT_TAG="weightedpdb_legacyresidual_ge200structural_disordered_v5_pdb490_residual005_ge200505_dis020_catalogreplacement20260804_v1_proteinonlydiffusion_v1_resolvedproteincrop_minloss4_v2_allmsashuffle_multichain_seg50_whole50_gap24_min9_v2_partialquery_max2_keep1_v1"
            local sabdab_root="${SABDAB_ROOT_DIR:-/mnt/project_abd/zhe/protenix_copy/vepfs-mlp2/project-abd/zhujianwei/public_databases/protenix}"
            local sabdab_index="${SABDAB_TRAIN_INDICES_FPATH:-/mnt/project_abd/klp/databases/protenix_sabdab_exact_antibody_interface_release_le_20210930_v2_20260824/sabdab-train_exact_antibody_interface_release_le_20210930_v2.csv}"
            local distill_sabdab_index="${DISTILL_SABDAB_INDICES_FPATH:-/mnt/project_abd/klp/databases/protenix_sabdab_exact_antibody_interface_v1_20260821/distill-sabdab_exact_antibody_interface_v1.csv}"
            local sabdab_weight="${PROTENIX_SABDAB_TRAIN_SAMPLE_WEIGHT:-0.02}"
            local distill_sabdab_weight="${PROTENIX_DISTILL_SABDAB_SAMPLE_WEIGHT:-0.01}"
            export SABDAB_ROOT_DIR="${sabdab_root}"
            export SABDAB_TRAIN_INDICES_FPATH="${sabdab_index}"
            export DISTILL_SABDAB_INDICES_FPATH="${distill_sabdab_index}"
            _crop384_require_positive_decimal PROTENIX_SABDAB_TRAIN_SAMPLE_WEIGHT "${sabdab_weight}" || return
            _crop384_require_positive_decimal PROTENIX_DISTILL_SABDAB_SAMPLE_WEIGHT "${distill_sabdab_weight}" || return
            CROP384_DATA_PROFILE_TRAIN_SETS="weightedPDB_before210930_v20260101,afdb_mgnify_legacy_residual_distillation_v20260804,all_sources_ge200_structural_distillation_v20260804,openfold3_disordered_v20260725,sabdab-train,distill-sabdab"
            CROP384_DATA_PROFILE_TRAIN_SAMPLE_WEIGHTS="$(awk -v weights="0.49,0.005,0.505,0.02,${sabdab_weight},${distill_sabdab_weight}" 'BEGIN {
                n = split(weights, values, ","); total = 0;
                for (i = 1; i <= n; i++) total += values[i];
                for (i = 1; i <= n; i++) printf "%s%.17g", (i == 1 ? "" : ","), values[i] / total;
            }')"
            CROP384_DATA_PROFILE_RUNTIME_INDEX_DIR="${residual_runtime_index_dir},${ge200_runtime_index_dir}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS="${residual_runtime_index_dir}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${residual_runtime_index_dir}/manifest.json"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${residual_runtime_index_dir}/target_catalog_pair.json"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${residual_runtime_index_dir}/roots.json"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${residual_runtime_index_dir}/row_offsets.u64"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${residual_runtime_index_dir}/source_root_ids.u8"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${residual_runtime_index_dir}/cif_relpaths.bin"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${residual_runtime_index_dir}/sampling/cluster_ids.u32"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${residual_runtime_index_dir}/sampling/first_row_ids.u64"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${residual_runtime_index_dir}/sampling/cluster_sizes.u32"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${ge200_runtime_index_dir}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${ge200_runtime_index_dir}/manifest.json"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${ge200_runtime_index_dir}/target_catalog_pair.json"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${ge200_runtime_index_dir}/roots.json"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${ge200_runtime_index_dir}/row_offsets.u64"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${ge200_runtime_index_dir}/source_root_ids.u8"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${ge200_runtime_index_dir}/cif_relpaths.bin"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${ge200_runtime_index_dir}/sampling/cluster_ids.u32"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${ge200_runtime_index_dir}/sampling/first_row_ids.u64"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${ge200_runtime_index_dir}/sampling/cluster_sizes.u32"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${disordered_index}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${disordered_success}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${disordered_root}/protenix/manifest.json"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${disordered_structure_dir}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${disordered_alignment_dir}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${disordered_alignment_main}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${sabdab_index}:${distill_sabdab_index}"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${sabdab_root}/sabdab-train_mmcif:${sabdab_root}/sabdab-train_bioassembly"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${sabdab_root}/distill-sabdab_cif:${sabdab_root}/distill-sabdab_bioassembly"
            CROP384_DATA_PROFILE_EXTRA_REQUIRED_PATHS+=":${sabdab_root}/sabdab-train.json:${sabdab_root}/sab-20260416.json:${sabdab_root}/mmcif_msa_template"
            CROP384_DATA_PROFILE_ARGS=(
                --data.msa.enable_virtual_segment_msa_shuffle true
                --data.msa.virtual_segment_residue_gap 24
                --data.msa.virtual_segment_min_length 9
                --data.msa.enable_multichain_msa_shuffle_variant_sampling true
                --data.msa.multichain_msa_shuffle_segment_probability 0.5
                --data.msa.enable_crop_aware_partial_query_only true
                --data.msa.partial_query_only_max_chains 2
                --data.weightedPDB_before210930_v20260101.base_info.min_diffusion_supervised_tokens 4
                --data.weightedPDB_before210930_v20260101.cropping_configs.max_nuc_tokens 128
                --data.afdb_mgnify_legacy_residual_distillation_v20260804.base_info.runtime_index_dir "${residual_runtime_index_dir}"
                --data.afdb_mgnify_legacy_residual_distillation_v20260804.base_info.source_roots_fpath "${residual_runtime_index_dir}/roots.json"
                --data.afdb_mgnify_legacy_residual_distillation_v20260804.cropping_configs.max_nuc_tokens 128
                --data.afdb_mgnify_legacy_residual_distillation_v20260804.msa.max_input_unpaired_msa_depth 16384
                --data.afdb_mgnify_legacy_residual_distillation_v20260804.augmentation.query_only_msa_probability 0.1
                --data.afdb_mgnify_legacy_residual_distillation_v20260804.augmentation.pair_msa_as_unpaired_probability 0.3
                --data.all_sources_ge200_structural_distillation_v20260804.base_info.runtime_index_dir "${ge200_runtime_index_dir}"
                --data.all_sources_ge200_structural_distillation_v20260804.base_info.source_roots_fpath "${ge200_runtime_index_dir}/roots.json"
                --data.all_sources_ge200_structural_distillation_v20260804.cropping_configs.max_nuc_tokens 128
                --data.all_sources_ge200_structural_distillation_v20260804.msa.max_input_unpaired_msa_depth 16384
                --data.all_sources_ge200_structural_distillation_v20260804.augmentation.query_only_msa_probability 0.1
                --data.all_sources_ge200_structural_distillation_v20260804.augmentation.pair_msa_as_unpaired_probability 0.3
                --data.openfold3_disordered_v20260725.base_info.indices_fpath "${disordered_index}"
                --data.openfold3_disordered_v20260725.base_info.dataset_success_fpath "${disordered_success}"
                --data.openfold3_disordered_v20260725.base_info.dataset_manifest_fpath "${disordered_root}/protenix/manifest.json"
                --data.openfold3_disordered_v20260725.base_info.mmcif_dir "${disordered_structure_dir}"
                --data.openfold3_disordered_v20260725.base_info.bioassembly_dict_dir "${disordered_structure_dir}"
                --data.openfold3_disordered_v20260725.base_info.disable_non_protein_diffusion_weights "${disable_non_protein_diffusion_weights}"
                --data.openfold3_disordered_v20260725.base_info.require_resolved_protein_crop true
                --data.openfold3_disordered_v20260725.base_info.min_diffusion_supervised_tokens 4
                --data.openfold3_disordered_v20260725.cropping_configs.max_nuc_tokens 128
                --data.openfold3_disordered_v20260725.augmentation.query_only_msa_probability 0.1
                --data.openfold3_disordered_v20260725.augmentation.pair_msa_as_unpaired_probability 0.3
                --data.sabdab-train.base_info.indices_fpath "${sabdab_index}"
                --data.sabdab-train.base_info.mmcif_dir "${sabdab_root}/sabdab-train_mmcif"
                --data.sabdab-train.base_info.bioassembly_dict_dir "${sabdab_root}/sabdab-train_bioassembly"
                --data.sabdab-train.msa.prot_seq_or_filename_to_msadir_jsons "${sabdab_root}/sabdab-train.json"
                --data.sabdab-train.msa.prot_msadir_raw_paths "${sabdab_root}/mmcif_msa_template"
                --data.sabdab-train.template.prot_seq_or_filename_to_templatedir_jsons "${sabdab_root}/sabdab-train.json"
                --data.sabdab-train.template.prot_template_raw_paths "${sabdab_root}/mmcif_msa_template"
                --data.sabdab-train.cropping_configs.max_nuc_tokens 256
                --data.distill-sabdab.base_info.indices_fpath "${distill_sabdab_index}"
                --data.distill-sabdab.base_info.mmcif_dir "${sabdab_root}/distill-sabdab_cif"
                --data.distill-sabdab.base_info.bioassembly_dict_dir "${sabdab_root}/distill-sabdab_bioassembly"
                --data.distill-sabdab.msa.prot_seq_or_filename_to_msadir_jsons "${sabdab_root}/sab-20260416.json"
                --data.distill-sabdab.msa.prot_msadir_raw_paths "${sabdab_root}/mmcif_msa_template"
                --data.distill-sabdab.template.prot_seq_or_filename_to_templatedir_jsons "${sabdab_root}/sab-20260416.json"
                --data.distill-sabdab.template.prot_template_raw_paths "${sabdab_root}/mmcif_msa_template"
                --data.distill-sabdab.cropping_configs.max_nuc_tokens 256
            )
            ;;
        *)
            echo "Unknown crop384/DB48 data profile '${requested_profile}'." >&2
            echo \
                "Available profiles: weighted_pdb_only_v1, weighted_pdb_afdb_mgnify_distill_v1, weighted_pdb_afdb_mgnify_disordered_v2, weighted_pdb_afdb_mgnify_short_long_disordered_v3, weighted_pdb_afdb_mgnify_short_long_disordered_v4, weighted_pdb_legacy_residual_ge200_disordered_v5" \
                >&2
            return 2
            ;;
    esac
}
