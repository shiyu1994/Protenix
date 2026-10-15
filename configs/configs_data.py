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

# pylint: disable=C0114,C0301
import os
from copy import deepcopy
from pathlib import Path

from protenix.config.extend_types import GlobalConfigValue, ListValue

PROTENIX_ROOT_DIR = os.environ.get("PROTENIX_ROOT_DIR", str(Path.home()))
SABDAB_ROOT_DIR = os.environ.get("SABDAB_ROOT_DIR", PROTENIX_ROOT_DIR)
SEQUENCE_TEMPLATE_CACHE_ROOT = os.environ.get(
    "PROTENIX_SEQUENCE_TEMPLATE_CACHE_ROOT",
    "",
)
SEQUENCE_TEMPLATE_CACHE_MODE = os.environ.get(
    "PROTENIX_SEQUENCE_TEMPLATE_CACHE_MODE",
    "online",
)
SEQUENCE_TEMPLATE_CACHE_ERROR_POLICY = os.environ.get(
    "PROTENIX_SEQUENCE_TEMPLATE_CACHE_ERROR_POLICY",
    "raise",
)
SEQUENCE_TEMPLATE_CACHE_FILENAME = os.environ.get(
    "PROTENIX_SEQUENCE_TEMPLATE_CACHE_FILENAME",
    "template_features.pkl.gz",
)
EVALUATION_SEQUENCE_TEMPLATE_CACHE_ROOT = os.environ.get(
    "PROTENIX_EVAL_SEQUENCE_TEMPLATE_CACHE_ROOT",
    "",
)
EVALUATION_SEQUENCE_TEMPLATE_CACHE_MODE = os.environ.get(
    "PROTENIX_EVAL_SEQUENCE_TEMPLATE_CACHE_MODE",
    "online",
)
EVALUATION_SEQUENCE_TEMPLATE_CACHE_FILENAME = os.environ.get(
    "PROTENIX_EVAL_SEQUENCE_TEMPLATE_CACHE_FILENAME",
    "template_features.pkl.gz",
)
EVALUATION_SEQUENCE_TEMPLATE_CACHE_POLICY = os.environ.get(
    "PROTENIX_EVAL_SEQUENCE_TEMPLATE_CACHE_POLICY",
    "pxm_eval_top4_maxdate_2021_09_30_v1",
)
AFDB_MGNIFY_DISTILLATION_CATALOG_DIR = os.environ.get(
    "PROTENIX_DISTILLATION_CATALOG_DIR",
    "/mnt/project_abd_hs/klp/databases/"
    "afdb_mgnify_distillation_catalog_20260724",
)
AFDB_MGNIFY_DISTILLATION_RUNTIME_INDEX_DIR = os.environ.get(
    "PROTENIX_DISTILLATION_RUNTIME_INDEX_DIR",
    "/mnt/project_abd_hs/klp/databases/"
    "afdb_mgnify_distillation_runtime_index_20260725",
)
AFDB_MGNIFY_SHORT_DISTILLATION_RUNTIME_INDEX_DIR = os.environ.get(
    "PROTENIX_SHORT_DISTILLATION_RUNTIME_INDEX_DIR",
    "/mnt/project_abd_hs/klp/databases/"
    "afdb_mgnify_distillation_short_le200_runtime_index_20260725",
)
AFDB_MGNIFY_LONG_DISTILLATION_RUNTIME_INDEX_DIR = os.environ.get(
    "PROTENIX_LONG_DISTILLATION_RUNTIME_INDEX_DIR",
    "/mnt/project_abd_hs/klp/databases/"
    "afdb_mgnify_distillation_long_gt200_runtime_index_20260725",
)
AFDB_MGNIFY_LEGACY_RESIDUAL_RUNTIME_INDEX_DIR = os.environ.get(
    "PROTENIX_LEGACY_RESIDUAL_RUNTIME_INDEX_DIR",
    "/mnt/project_abd_hs/klp/databases/"
    "afdb_mgnify_legacy_residual_not_ge200_runtime_index_20260804",
)
ALL_SOURCES_GE200_STRUCTURAL_RUNTIME_INDEX_DIR = os.environ.get(
    "PROTENIX_GE200_STRUCTURAL_RUNTIME_INDEX_DIR",
    "/mnt/project_abd_hs/klp/databases/"
    "all_sources_ge200_structural_runtime_index_20260804",
)
OPENFOLD3_DISORDERED_ROOT_DIR = os.environ.get(
    "PROTENIX_OPENFOLD3_DISORDERED_ROOT_DIR",
    "/mnt/project_abd_hs/klp/databases/pdb_disordered",
)
OPENFOLD3_DISORDERED_STRUCTURE_DIR = os.path.join(
    OPENFOLD3_DISORDERED_ROOT_DIR,
    "structures",
    "structures",
)
OPENFOLD3_DISORDERED_ALIGNMENT_DIRS = (
    os.path.join(
        OPENFOLD3_DISORDERED_ROOT_DIR,
        "alignment_arrays",
    ),
    os.path.join(
        OPENFOLD3_DISORDERED_ROOT_DIR,
        "alignment_arrays",
        "alignment_arrays",
    ),
)
OPENFOLD3_DISORDERED_PROTENIX_DIR = os.path.join(
    OPENFOLD3_DISORDERED_ROOT_DIR,
    "protenix",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_LONG_PROTEIN_CONTEXT_768_DIR = os.environ.get(
    "PROTENIX_BENCHMARK_DIR",
    str(REPO_ROOT / "data" / "benchmark_long_protein_context_768"),
)
BENCHMARK_LONG_PROTEIN_CONTEXT_1024_DIR = os.environ.get(
    "PROTENIX_BENCHMARK_DIR",
    str(REPO_ROOT / "data" / "benchmark_long_protein_context_1024"),
)
BENCHMARK_LONG_PROTEIN_CONTEXT_1024_800_DIR = os.environ.get(
    "PROTENIX_BENCHMARK_DIR",
    str(REPO_ROOT / "data" / "benchmark_long_protein_context_1024_800"),
)
BENCHMARK_LONG_PROTEIN_CONTEXT_1024_800_MSA2048_DIR = os.environ.get(
    "PROTENIX_BENCHMARK_DIR",
    str(
        REPO_ROOT
        / "data"
        / "benchmark_long_protein_context_1024_800_msa2048"
    ),
)
# Crop384 reuses the exact, hash-audited 800-row MSA2048 artifact selected for
# crop1024.  Every source row is substantially longer than 384 tokens; the
# distinct logical key keeps resolved configs and benchmark evidence explicit.
BENCHMARK_LONG_PROTEIN_CONTEXT_384_800_MSA2048_DIR = os.environ.get(
    "PROTENIX_BENCHMARK_DIR",
    str(
        REPO_ROOT
        / "data"
        / "benchmark_long_protein_context_1024_800_msa2048"
    ),
)
# Crop960 deliberately reuses the already audited 800-row MSA2048 artifacts.
# Only the runtime crop shape changes; keeping a distinct dataset key makes the
# resolved config and benchmark summaries unambiguous without copying data.
BENCHMARK_LONG_PROTEIN_CONTEXT_960_800_MSA2048_DIR = os.environ.get(
    "PROTENIX_BENCHMARK_DIR",
    str(
        REPO_ROOT
        / "data"
        / "benchmark_long_protein_context_1024_800_msa2048"
    ),
)

default_test_configs = {
    "sampler_configs": {
        "sampler_type": "uniform",
    },
    "cropping_configs": {
        "method_weights": [
            0.0,  # ContiguousCropping
            0.0,  # SpatialCropping
            1.0,  # SpatialInterfaceCropping
        ],
        "crop_size": -1,
    },
    "lig_atom_rename": GlobalConfigValue("test_lig_atom_rename"),
    "shuffle_mols": GlobalConfigValue("test_shuffle_mols"),
    "shuffle_sym_ids": GlobalConfigValue("test_shuffle_sym_ids"),
    "constraint": {
        "enable": False,
        "fix_seed": False,  # True means use use the same contact in each evaluation.
    },
}

default_weighted_pdb_configs = {
    "sampler_configs": {
        "sampler_type": "weighted",
        "beta_dict": {
            "chain": 0.5,
            "interface": 1,
        },
        "alpha_dict": {
            "prot": 3,
            "nuc": 3,
            "ligand": 1,
        },
        "force_recompute_weight": True,
    },
    "cropping_configs": {
        "method_weights": ListValue([0.2, 0.4, 0.4]),
        "crop_size": GlobalConfigValue("train_crop_size"),
        "max_nuc_tokens": 128,
        # Keep the Dataset.crop defaults explicit so controlled benchmarks can
        # override them from the CLI and guarantee a fixed token shape.
        "contiguous_crop_complete_lig": True,
        "spatial_crop_complete_lig": True,
        "drop_last": True,
        "remove_metal": True,
    },
    "sample_weight": 0.5,
    "limits": -1,
    "lig_atom_rename": GlobalConfigValue("train_lig_atom_rename"),
    "shuffle_mols": GlobalConfigValue("train_shuffle_mols"),
    "shuffle_sym_ids": GlobalConfigValue("train_shuffle_sym_ids"),
    # If enabled, the training settings for different constraint types,
    # providing the model a certain proportion of constraints
    # that meet specific conditions.
    "constraint": {
        "enable": False,
        "fix_seed": False,
        "pocket": {
            "prob": 0.0,
            "size": 1 / 3,
            "spec_binder_chain": False,
            "max_distance_range": {"PP": ListValue([6, 20]), "LP": ListValue([6, 20])},
            "group": "complex",
            "distance_type": "center_atom",
        },
        "contact": {
            "prob": 0.0,
            "size": 1 / 3,
            "max_distance_range": {
                "PP": ListValue([6, 30]),
                "PL": ListValue([4, 10]),
            },
            "group": "complex",
            "distance_type": "center_atom",
        },
        "substructure": {
            "prob": 0.0,
            "size": 0.8,
            "mol_type_pairs": {
                "PP": 15,
                "PL": 10,
                "LP": 10,
            },
            "feature_type": "one_hot",
            "ratios": {
                "full": [
                    0.0,
                    0.5,
                    1.0,
                ],  # ratio options of full chain substructure constraint
                "partial": 0.3,  # ratio of partial chain substructure constraint
            },
            "coord_noise_scale": 0.05,
            "spec_asym_id": False,
        },
        "contact_atom": {
            "prob": 0.0,
            "size": 1 / 3,
            "max_distance_range": {
                "PP": ListValue([2, 12]),
                "PL": ListValue([2, 8]),
            },
            "min_distance": -1,
            "group": "complex",
            "distance_type": "atom",
            "feature_type": "continuous",
        },
    },
}


data_configs = {
    "num_dl_workers": 16,
    # Reuse training workers across the short sampler epochs. This avoids
    # repeatedly constructing every worker process at each epoch boundary.
    # PyTorch requires this to be False when num_dl_workers == 0; the training
    # dataloader applies that compatibility fallback automatically.
    "train_dl_persistent_workers": True,
    # Number of batches prefetched by each training worker. This setting is
    # ignored when num_dl_workers == 0. Evaluation loaders intentionally keep
    # their historical behavior.
    "train_dl_prefetch_factor": 2,
    "epoch_size": 10000,
    # Target-occurrence augmentation decisions are derived from the resumable
    # sampler position rather than DataLoader worker RNG. Individual dataset
    # configs may override any probability when a future data profile needs
    # different treatment. Query-only and unpaired-shuffle are mutually
    # exclusive; the remaining probability uses normal species pairing.
    "query_only_msa_probability": 0.0,
    "pair_msa_as_unpaired_probability": 0.0,
    "template_use_probability": 1.0,
    "augmentation_decision_version": "sample_occurrence_sha256_v1",
    "train_ref_pos_augment": True,
    "test_ref_pos_augment": True,
    "train_sets": ListValue(["weightedPDB_before2109_wopb_nometalc_0925"]),
    "train_sampler": {
        "train_sample_weights": ListValue([1.0]),
        "sampler_type": "weighted",
        # A negative value inherits the global training seed.  Controlled
        # benchmarks may override only the sample order without changing
        # model, crop, or diffusion RNG streams.
        "seed": -1,
        # Preserve the official training behavior by default. Controlled
        # benchmark profiles may disable replacement to audit one exact epoch.
        "replacement": True,
        # Empty keeps the historical IID-with-replacement sampler byte-for-byte
        # unchanged.  A non-empty path activates the checkpoint-bound,
        # precomputed soft-cluster schedule.  The schedule manifest is the
        # authority for its source checkpoint, topology, cursor boundary and
        # immutable row-array hashes; the runtime fails closed on any mismatch.
        "soft_cluster_schedule_manifest": "",
        # Kept in typed configuration so launchers state the scientific policy
        # explicitly.  The runtime also requires this value to match the
        # manifest, rather than silently trusting either side.
        "soft_cluster_gamma": 0.2,
    },
    "test_sets": ListValue(["recentPDB_1536_sample384_0925"]),
    # NOTE:
    # `weightedPDB_before2109_wopb_nometalc_0925` is compatible with the `2024.05.22` data version
    # downloaded via `scripts/database/download_protenix_data.sh`.
    # `weightedPDB_before250701_v20260101` and `weightedPDB_before210930_v20260101` are compatible
    # with the `2026.01.01` data version.
    "afdb_mgnify_distillation_v20260725": {
        "base_info": {
            # The authority CSV is retained as provenance. Runtime sampling
            # uses the compact mmap directory and never loads this 33.5M-row
            # table into pandas.
            "indices_fpath": os.path.join(
                AFDB_MGNIFY_DISTILLATION_CATALOG_DIR,
                "distillation_index.csv.gz",
            ),
            "runtime_index_dir": AFDB_MGNIFY_DISTILLATION_RUNTIME_INDEX_DIR,
            "source_roots_fpath": os.path.join(
                AFDB_MGNIFY_DISTILLATION_RUNTIME_INDEX_DIR,
                "roots.json",
            ),
            "mmcif_dir": AFDB_MGNIFY_DISTILLATION_RUNTIME_INDEX_DIR,
            # Kept non-empty for the typed config parser. Runtime-index samples
            # resolve their nested PKL paths from source_roots instead.
            "bioassembly_dict_dir": AFDB_MGNIFY_DISTILLATION_RUNTIME_INDEX_DIR,
            "pdb_list": "",
            "random_sample_if_failed": False,
            "max_n_token": -1,
            "use_reference_chains_only": False,
            "is_distillation": True,
            # Historical two-source profiles leave this at zero. The current
            # disordered-v2 three-source profile overrides it to four.
            "min_diffusion_supervised_tokens": 0,
            # Registered as false so historical profiles retain their
            # behavior. The current disordered-v2 profile enables it to
            # constructively repair AFDB/MGnify crops to four protein tokens.
            "require_resolved_protein_crop": False,
            "exclusion": {},
            "remove_pure_nuc": False,
        },
        "sampler_configs": {
            "sampler_type": "clustered",
        },
        "cropping_configs": deepcopy(
            default_weighted_pdb_configs["cropping_configs"]
        ),
        "msa": {
            # Empty-string sentinels keep the typed CLI schema valid without
            # loading a 33.5M-entry JSON map. MSAFeaturizer treats an empty
            # mapping path as an empty lookup table; runtime samples always
            # supply their direct sibling A3M path.
            "prot_seq_or_filename_to_msadir_jsons": ListValue([""]),
            "prot_msadir_raw_paths": ListValue([""]),
            "prot_pairing_dbs": ListValue([""]),
            "prot_non_pairing_dbs": ListValue([""]),
            "prot_indexing_methods": ListValue(["pdb_id"]),
            "rna_seq_or_filename_to_msadir_jsons": ListValue([""]),
            "rna_msadir_raw_paths": ListValue([""]),
            "rna_indexing_methods": ListValue(["sequence"]),
            "enable_prot_msa": True,
            "enable_rna_msa": False,
            "pair_msa_as_unpaired_only": False,
            "shuffle_pair_msa_when_unpaired_only": False,
        },
        "template": {
            "enable_prot_template": False,
            # Keep TemplateEmbedder in the static DDP graph even though this
            # source has no template database. The emitted four-slot dummy
            # representation is identical to weightedPDB template dropout.
            "emit_dummy_template_when_disabled": True,
        },
        "augmentation": {
            "query_only_msa_probability": 0.2,
            "pair_msa_as_unpaired_probability": 0.0,
            "template_use_probability": 0.0,
        },
        "lig_atom_rename": False,
        "shuffle_mols": False,
        "shuffle_sym_ids": False,
        "constraint": {"enable": False},
    },
    "openfold3_disordered_v20260725": {
        "base_info": {
            "indices_fpath": os.path.join(
                OPENFOLD3_DISORDERED_PROTENIX_DIR,
                "indices.csv.gz",
            ),
            "dataset_success_fpath": os.path.join(
                OPENFOLD3_DISORDERED_PROTENIX_DIR,
                "_SUCCESS.json",
            ),
            "dataset_manifest_fpath": os.path.join(
                OPENFOLD3_DISORDERED_PROTENIX_DIR,
                "manifest.json",
            ),
            "mmcif_dir": OPENFOLD3_DISORDERED_STRUCTURE_DIR,
            "bioassembly_dict_dir": OPENFOLD3_DISORDERED_STRUCTURE_DIR,
            "bioassembly_relpath_column": "bioassembly_relpath",
            "pdb_list": "",
            "random_sample_if_failed": False,
            "max_n_token": -1,
            "use_reference_chains_only": False,
            "is_distillation": True,
            # Match the AF3/OpenFold3 initial-training treatment of the
            # disordered hybrid structures. Protein coordinates come from
            # AF-Multimer, whereas non-protein coordinates were aligned in
            # from the experimental PDB structure.
            "disable_non_protein_diffusion_weights": True,
            # This is an explicit Protenix data-quality constraint, separate
            # from the OpenFold3 loss flag above. Together with the minimum
            # below, every crop must retain at least four tokens containing a
            # resolved protein atom.
            "require_resolved_protein_crop": True,
            "min_diffusion_supervised_tokens": 4,
            "exclusion": {},
            "remove_pure_nuc": False,
        },
        "sampler_configs": {
            "sampler_type": "weighted",
            "beta_dict": {
                "chain": 0.5,
                "interface": 1,
            },
            "alpha_dict": {
                "prot": 3,
                "nuc": 3,
                "ligand": 1,
            },
            # The cache builder writes weights using the official OpenFold3
            # cluster_size values. Recomputing from this 28,567-entry subset
            # would silently change the cluster masses.
            "force_recompute_weight": False,
        },
        "cropping_configs": deepcopy(
            default_weighted_pdb_configs["cropping_configs"]
        ),
        "msa": {
            "prot_seq_or_filename_to_msadir_jsons": ListValue([""]),
            "prot_msadir_raw_paths": ListValue([""]),
            "prot_pairing_dbs": ListValue([""]),
            "prot_non_pairing_dbs": ListValue([""]),
            "prot_indexing_methods": ListValue(["pdb_id"]),
            "rna_seq_or_filename_to_msadir_jsons": ListValue([""]),
            "rna_msadir_raw_paths": ListValue([""]),
            "rna_indexing_methods": ListValue(["sequence"]),
            "openfold3_alignment_npz_dirs": ListValue(
                list(OPENFOLD3_DISORDERED_ALIGNMENT_DIRS)
            ),
            "enable_prot_msa": True,
            "enable_rna_msa": True,
            "pair_msa_as_unpaired_only": False,
            "shuffle_pair_msa_when_unpaired_only": False,
        },
        "template": {
            "enable_prot_template": False,
            # See the AFDB/MGnify source above. Missing template coordinates
            # are represented by fixed fully masked features, not by skipping
            # TemplateEmbedder.
            "emit_dummy_template_when_disabled": True,
        },
        "augmentation": {
            "query_only_msa_probability": 0.2,
            "pair_msa_as_unpaired_probability": 0.0,
            "template_use_probability": 0.0,
        },
        "lig_atom_rename": False,
        "shuffle_mols": False,
        "shuffle_sym_ids": False,
        "constraint": {"enable": False},
    },
    "distill-sabdab": {
        "base_info": {
            "mmcif_dir": os.path.join(SABDAB_ROOT_DIR, "distill-sabdab_cif"),
            "bioassembly_dict_dir": os.path.join(
                SABDAB_ROOT_DIR, "distill-sabdab_bioassembly"
            ),
            "indices_fpath": os.path.join(
                SABDAB_ROOT_DIR,
                "indices/distill-sabdab.csv",
            ),
            "pdb_list": "",
            "random_sample_if_failed": True,
            "max_n_token": -1,  # can be used for removing data with too many tokens.
            "use_reference_chains_only": False,
            "exclusion": {  # do not sample the data based on ions.
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
            "remove_pure_nuc": True,
        },
        "msa": {
            "prot_seq_or_filename_to_msadir_jsons": ListValue(
                [os.path.join(SABDAB_ROOT_DIR, "sab-20260416.json")]
            ),
            "prot_msadir_raw_paths": ListValue(
                [os.path.join(SABDAB_ROOT_DIR, "mmcif_msa_template")]
            ),
            "prot_indexing_methods": ListValue(["sequence"]),
            "pair_msa_as_unpaired_only": False,
        },
        "template": {
            "prot_template_raw_paths": ListValue(
                [os.path.join(SABDAB_ROOT_DIR, "mmcif_msa_template")]
            ),
            "prot_seq_or_filename_to_templatedir_jsons": ListValue(
                [os.path.join(SABDAB_ROOT_DIR, "sab-20260416.json")]
            ),
            "prot_indexing_methods": ListValue(["sequence"]),
        },
        **deepcopy(default_weighted_pdb_configs),
    },
    "sabdab-train": {
        "base_info": {
            "mmcif_dir": os.path.join(SABDAB_ROOT_DIR, "sabdab-train_mmcif"),
            "bioassembly_dict_dir": os.path.join(
                SABDAB_ROOT_DIR, "sabdab-train_bioassembly"
            ),
            "indices_fpath": os.path.join(
                SABDAB_ROOT_DIR,
                "indices/sabdab-train.csv",
            ),
            "pdb_list": "",
            "random_sample_if_failed": True,
            "max_n_token": -1,  # can be used for removing data with too many tokens.
            "use_reference_chains_only": False,
            "exclusion": {  # do not sample the data based on ions.
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
            "remove_pure_nuc": True,
        },
        "msa": {
            "prot_seq_or_filename_to_msadir_jsons": ListValue(
                [os.path.join(SABDAB_ROOT_DIR, "sabdab-train.json")]
            ),
            "prot_msadir_raw_paths": ListValue(
                [os.path.join(SABDAB_ROOT_DIR, "mmcif_msa_template")]
            ),
            "prot_indexing_methods": ListValue(["sequence"]),
            "pair_msa_as_unpaired_only": False,
        },
        "template": {
            "prot_template_raw_paths": ListValue(
                [os.path.join(SABDAB_ROOT_DIR, "mmcif_msa_template")]
            ),
            "prot_seq_or_filename_to_templatedir_jsons": ListValue(
                [os.path.join(SABDAB_ROOT_DIR, "sabdab-train.json")]
            ),
            "prot_indexing_methods": ListValue(["sequence"]),
        },
        **deepcopy(default_weighted_pdb_configs),
    },
    "weightedPDB_before2109_wopb_nometalc_0925": {
        "base_info": {
            "mmcif_dir": os.path.join(PROTENIX_ROOT_DIR, "mmcif"),
            "bioassembly_dict_dir": os.path.join(
                PROTENIX_ROOT_DIR, "mmcif_bioassembly"
            ),
            "indices_fpath": os.path.join(
                PROTENIX_ROOT_DIR,
                "indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz",
            ),
            "pdb_list": "",
            "random_sample_if_failed": True,
            "max_n_token": -1,  # can be used for removing data with too many tokens.
            "use_reference_chains_only": False,
            "exclusion": {  # do not sample the data based on ions.
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
            "remove_pure_nuc": True,
        },
        **deepcopy(default_weighted_pdb_configs),
    },
    # Shape-controlled benchmark dataset. The generated index contains only
    # protein-context examples longer than 768 tokens; benchmark.sh combines
    # it with spatial-crop overrides so every measured sample is exactly 768.
    "benchmark_long_protein_context_768": {
        "base_info": {
            "mmcif_dir": os.path.join(PROTENIX_ROOT_DIR, "mmcif"),
            "bioassembly_dict_dir": os.path.join(
                PROTENIX_ROOT_DIR, "mmcif_bioassembly"
            ),
            "indices_fpath": os.path.join(
                BENCHMARK_LONG_PROTEIN_CONTEXT_768_DIR,
                "indices.csv",
            ),
            "pdb_list": "",
            "random_sample_if_failed": True,
            "max_n_token": -1,
            "min_n_token": 769,
            "use_reference_chains_only": False,
            "exclusion": {
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
            "remove_pure_nuc": True,
        },
        **deepcopy(default_weighted_pdb_configs),
    },
    # The crop-1024 profile deliberately selects the same long-PDB population
    # as the crop-768 profile; only the minimum source length and resolved crop
    # size differ. benchmark.sh supplies the profile-specific index path.
    "benchmark_long_protein_context_1024": {
        "base_info": {
            "mmcif_dir": os.path.join(PROTENIX_ROOT_DIR, "mmcif"),
            "bioassembly_dict_dir": os.path.join(
                PROTENIX_ROOT_DIR, "mmcif_bioassembly"
            ),
            "indices_fpath": os.path.join(
                BENCHMARK_LONG_PROTEIN_CONTEXT_1024_DIR,
                "indices.csv",
            ),
            "pdb_list": "",
            "random_sample_if_failed": True,
            "max_n_token": -1,
            "min_n_token": 1025,
            "use_reference_chains_only": False,
            "exclusion": {
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
            "remove_pure_nuc": True,
        },
        **deepcopy(default_weighted_pdb_configs),
    },
    # Eight-rank, 100-step coverage benchmark. The generated index is PDB- and
    # cluster-unique; the launcher pairs it with epoch_size=800 and sampling
    # without replacement so every source row is consumed exactly once.
    "benchmark_long_protein_context_1024_800": {
        "base_info": {
            "mmcif_dir": os.path.join(PROTENIX_ROOT_DIR, "mmcif"),
            "bioassembly_dict_dir": os.path.join(
                PROTENIX_ROOT_DIR, "mmcif_bioassembly"
            ),
            "indices_fpath": os.path.join(
                BENCHMARK_LONG_PROTEIN_CONTEXT_1024_800_DIR,
                "indices.csv",
            ),
            "pdb_list": "",
            # A fallback sample would invalidate the exact 800-sample coverage
            # claim, so this profile fails loudly on a bad source row.
            "random_sample_if_failed": False,
            "max_n_token": -1,
            "min_n_token": 1025,
            "use_reference_chains_only": False,
            "exclusion": {
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
            "remove_pure_nuc": True,
        },
        **deepcopy(default_weighted_pdb_configs),
    },
    # Strict MSA2048 variant of the 800-row crop-1024 benchmark. Every
    # biological-assembly protein entity has at least 2048 effective A3M rows
    # after pairing/non-pairing merge and deduplication. Runtime trace remains
    # authoritative because a stochastic crop chooses the active chains.
    "benchmark_long_protein_context_1024_800_msa2048": {
        "base_info": {
            "mmcif_dir": os.path.join(PROTENIX_ROOT_DIR, "mmcif"),
            "bioassembly_dict_dir": os.path.join(
                PROTENIX_ROOT_DIR, "mmcif_bioassembly"
            ),
            "indices_fpath": os.path.join(
                BENCHMARK_LONG_PROTEIN_CONTEXT_1024_800_MSA2048_DIR,
                "indices.csv",
            ),
            "pdb_list": "",
            "random_sample_if_failed": False,
            "max_n_token": -1,
            "min_n_token": 1025,
            "use_reference_chains_only": False,
            "exclusion": {
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
            "remove_pure_nuc": True,
        },
        **deepcopy(default_weighted_pdb_configs),
    },
    # Exact-crop-384 view of the same strictly audited 800-row MSA2048 source
    # population.  Its min_n_token=1025 selection is intentionally retained:
    # it is stricter than crop384 requires and preserves the durable index hash.
    "benchmark_long_protein_context_384_800_msa2048": {
        "base_info": {
            "mmcif_dir": os.path.join(PROTENIX_ROOT_DIR, "mmcif"),
            "bioassembly_dict_dir": os.path.join(
                PROTENIX_ROOT_DIR, "mmcif_bioassembly"
            ),
            "indices_fpath": os.path.join(
                BENCHMARK_LONG_PROTEIN_CONTEXT_384_800_MSA2048_DIR,
                "indices.csv",
            ),
            "pdb_list": "",
            "random_sample_if_failed": False,
            "max_n_token": -1,
            "min_n_token": 1025,
            "use_reference_chains_only": False,
            "exclusion": {
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
            "remove_pure_nuc": True,
        },
        **deepcopy(default_weighted_pdb_configs),
    },
    # Exact-crop-960 view of the same strictly audited 800-row MSA2048 source
    # population.  The durable artifact was selected with min_n_token=1025,
    # which is stricter than crop960 requires and is retained for hash-stable
    # validation.
    "benchmark_long_protein_context_960_800_msa2048": {
        "base_info": {
            "mmcif_dir": os.path.join(PROTENIX_ROOT_DIR, "mmcif"),
            "bioassembly_dict_dir": os.path.join(
                PROTENIX_ROOT_DIR, "mmcif_bioassembly"
            ),
            "indices_fpath": os.path.join(
                BENCHMARK_LONG_PROTEIN_CONTEXT_960_800_MSA2048_DIR,
                "indices.csv",
            ),
            "pdb_list": "",
            "random_sample_if_failed": False,
            "max_n_token": -1,
            "min_n_token": 1025,
            "use_reference_chains_only": False,
            "exclusion": {
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
            "remove_pure_nuc": True,
        },
        **deepcopy(default_weighted_pdb_configs),
    },
    "weightedPDB_before250701_v20260101": {
        "base_info": {
            "mmcif_dir": os.path.join(PROTENIX_ROOT_DIR, "mmcif"),
            "bioassembly_dict_dir": os.path.join(
                PROTENIX_ROOT_DIR, "mmcif_bioassembly"
            ),
            "indices_fpath": os.path.join(
                PROTENIX_ROOT_DIR,
                "indices/indices_20260107-20chains_before_2025-07-01_res4.5.csv.gz",
            ),
            "pdb_list": "",
            "random_sample_if_failed": True,
            "max_n_token": -1,  # can be used for removing data with too many tokens.
            "use_reference_chains_only": False,
            "exclusion": {  # do not sample the data based on ions.
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
            "remove_pure_nuc": True,
        },
        **deepcopy(default_weighted_pdb_configs),
    },
    "weightedPDB_before210930_v20260101": {
        "base_info": {
            "mmcif_dir": os.path.join(PROTENIX_ROOT_DIR, "mmcif"),
            "bioassembly_dict_dir": os.path.join(
                PROTENIX_ROOT_DIR, "mmcif_bioassembly"
            ),
            "indices_fpath": os.path.join(
                PROTENIX_ROOT_DIR,
                "indices/indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz",
            ),
            "pdb_list": "",
            "random_sample_if_failed": True,
            "max_n_token": -1,  # can be used for removing data with too many tokens.
            "use_reference_chains_only": False,
            # Kept at zero for immutable historical profiles; disordered-v2
            # overrides the value to four on its training command.
            "min_diffusion_supervised_tokens": 0,
            "exclusion": {  # do not sample the data based on ions.
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
            "remove_pure_nuc": True,
        },
        **deepcopy(default_weighted_pdb_configs),
    },
    "recentPDB_1536_sample384_0925": {
        "base_info": {
            "mmcif_dir": os.path.join(PROTENIX_ROOT_DIR, "mmcif"),
            "bioassembly_dict_dir": os.path.join(
                PROTENIX_ROOT_DIR, "recentPDB_bioassembly"
            ),
            "indices_fpath": os.path.join(
                PROTENIX_ROOT_DIR, "indices/recentPDB_low_homology_maxtoken1536.csv"
            ),
            "pdb_list": os.path.join(
                PROTENIX_ROOT_DIR,
                "indices/recentPDB_low_homology_maxtoken1024_sample384_pdb_id.txt",
            ),
            "max_n_token": GlobalConfigValue("test_max_n_token"),  # filter data
            "sort_by_n_token": False,
            "group_by_pdb_id": True,
            "find_eval_chain_interface": True,
        },
        **deepcopy(default_test_configs),
    },
    "posebusters_0925": {
        "base_info": {
            "mmcif_dir": os.path.join(PROTENIX_ROOT_DIR, "posebusters_mmcif"),
            "bioassembly_dict_dir": os.path.join(
                PROTENIX_ROOT_DIR, "posebusters_bioassembly"
            ),
            "indices_fpath": os.path.join(
                PROTENIX_ROOT_DIR, "indices/posebusters_indices_mainchain_interface.csv"
            ),
            "pdb_list": "",
            "find_pocket": True,
            "find_all_pockets": False,
            "max_n_token": GlobalConfigValue("test_max_n_token"),  # filter data
        },
        **deepcopy(default_test_configs),
    },
    "msa": {
        "enable_prot_msa": True,
        "prot_seq_or_filename_to_msadir_jsons": ListValue(
            [os.path.join(PROTENIX_ROOT_DIR, "common/seq_to_pdb_index.json")]
        ),
        "prot_msadir_raw_paths": ListValue(
            [os.path.join(PROTENIX_ROOT_DIR, "mmcif_msa_template")]
        ),
        "prot_pairing_dbs": ListValue(["pairing"]),
        "prot_non_pairing_dbs": ListValue(
            ["pairing-non_pairing"]
        ),  # Separated by "-", "pairing-non_pairing" means both pairing and non_pairing are used as non_pairing,
        # with pairing used first.
        "pair_msa_as_unpaired_only": False,
        "shuffle_pair_msa_when_unpaired_only": False,
        # In unpaired-shuffle mode, independently permute all non-query
        # protein MSA rows across crop-disconnected canonical residue groups.
        # Keep this disabled by default so historical profiles retain their
        # exact data semantics.
        "enable_virtual_segment_msa_shuffle": False,
        "virtual_segment_residue_gap": 24,
        "virtual_segment_min_length": 9,
        # New profiles may choose crop-segment versus whole-physical-chain
        # all-row shuffling per multi-protein-chain occurrence. Disabled by
        # default so every historical profile retains its exact MSA policy.
        "enable_multichain_msa_shuffle_variant_sampling": False,
        "multichain_msa_shuffle_segment_probability": 0.5,
        # Query-only remains global for monomers. When enabled, a crop with
        # multiple physical protein chains makes only 1..max chains
        # query-only and always retains at least one normal protein MSA.
        "enable_crop_aware_partial_query_only": False,
        "partial_query_only_max_chains": 2,
        "prot_indexing_methods": ListValue(["sequence"]),
        "enable_rna_msa": True,  # enable rna msa
        "rna_seq_or_filename_to_msadir_jsons": ListValue(
            [os.path.join(PROTENIX_ROOT_DIR, "rna_msa/rna_sequence_to_pdb_chains.json")]
        ),
        "rna_msadir_raw_paths": ListValue(
            [os.path.join(PROTENIX_ROOT_DIR, "rna_msa/msas")]
        ),
        "rna_indexing_methods": ListValue(["sequence"]),
        "min_size": {
            "train": 1,
            "test": 1,
        },
        "max_size": {
            "train": 16384,
            "test": 16384,
        },
        # Training reads paired and unpaired raw candidates into two
        # independent pools. Unused capacity on one side never transfers to
        # the other; FeatureAssemblyLine applies max_size=16384 only after the
        # two bounded pools have been assembled.
        "max_input_paired_msa_depth": 10000,
        "max_input_unpaired_msa_depth": 10000,
        # Historical behavior charges the paired-source copy and genuine
        # non-pairing sources to one shared unpaired budget. A locked
        # checkpoint migration may opt into independent source budgets while
        # retaining the same paired/unpaired assembly and final max_size.
        "independent_pairing_non_pairing_budget": False,
        "sample_cutoff": {
            "train": 16384,
            "test": 16384,
        },
        # Row zero is the canonical global query. Preserve it whenever the
        # model-side sampled MSA depth K is at most 1,024, while keeping the
        # final depth exactly K. The same policy applies to train and inference.
        "query_inclusion_max_sample_size": {
            "train": 1024,
            "test": 1024,
        },
    },
    "template": {
        "enable_prot_template": True,
        "template_dropout_rate": 0.0,
        # Offline cache modes: required=cache-only,
        # online=legacy mmCIF+Kalign processing, disabled=no real templates.
        # In required mode, error_policy=raise preserves fail-fast behavior and
        # chain_dummy converts an unusable unique-sequence artifact into dummy
        # features only for chains sharing that sequence. The cache path is:
        # root/<seq_to_pdb_index relative directory>/<filename>.
        "sequence_template_cache_root": SEQUENCE_TEMPLATE_CACHE_ROOT,
        "sequence_template_cache_mode": SEQUENCE_TEMPLATE_CACHE_MODE,
        "sequence_template_cache_error_policy": (
            SEQUENCE_TEMPLATE_CACHE_ERROR_POLICY
        ),
        "sequence_template_cache_filename": SEQUENCE_TEMPLATE_CACHE_FILENAME,
        # Evaluation cache settings are intentionally separate from the
        # training cache contract.  Non-training datasets use this cache only
        # when mode is explicitly set to ``required``.
        "evaluation_sequence_template_cache_root": (
            EVALUATION_SEQUENCE_TEMPLATE_CACHE_ROOT
        ),
        "evaluation_sequence_template_cache_mode": (
            EVALUATION_SEQUENCE_TEMPLATE_CACHE_MODE
        ),
        "evaluation_sequence_template_cache_filename": (
            EVALUATION_SEQUENCE_TEMPLATE_CACHE_FILENAME
        ),
        "evaluation_sequence_template_cache_policy": (
            EVALUATION_SEQUENCE_TEMPLATE_CACHE_POLICY
        ),
        "prot_template_mmcif_dir": os.path.join(PROTENIX_ROOT_DIR, "mmcif"),
        "prot_template_cache_dir": "",
        "prot_template_raw_paths": ListValue(
            [os.path.join(PROTENIX_ROOT_DIR, "mmcif_msa_template")]
        ),
        "prot_seq_or_filename_to_templatedir_jsons": ListValue(
            [os.path.join(PROTENIX_ROOT_DIR, "common/seq_to_pdb_index.json")]
        ),
        "prot_indexing_methods": ListValue(["sequence"]),
        "release_dates_path": os.path.join(
            PROTENIX_ROOT_DIR, "common/release_date_cache.json"
        ),
        "obsolete_pdbs_path": os.path.join(
            PROTENIX_ROOT_DIR, "common/obsolete_to_successor.json"
        ),
        "max_template_date": "2021-09-30",
        "kalign_binary_path": "/usr/bin/kalign",  # apt-get install kalign
    },
    "ccd_components_file": os.path.join(PROTENIX_ROOT_DIR, "common/components.cif"),
    "ccd_components_rdkit_mol_file": os.path.join(
        PROTENIX_ROOT_DIR, "common/components.cif.rdkit_mol.pkl"
    ),
    "obsolete_release_data_csv": os.path.join(
        PROTENIX_ROOT_DIR, "common/obsolete_release_date.csv"
    ),
    "pdb_cluster_file": os.path.join(
        PROTENIX_ROOT_DIR, "common/clusters-by-entity-40.txt"
    ),
}


def _partitioned_afdb_mgnify_distillation_config(
    runtime_index_dir: str,
) -> dict:
    """Return one representative-length partition of the predicted library.

    The two partitions retain the original AFQ/MGN cluster IDs and compact
    cluster-uniform/member-uniform sampler.  Their independently verified
    runtime manifests are the authority for membership; the full catalog path
    remains only as shared provenance and is never loaded by training workers.
    """

    config = deepcopy(
        data_configs["afdb_mgnify_distillation_v20260725"]
    )
    base_info = config["base_info"]
    base_info["runtime_index_dir"] = runtime_index_dir
    base_info["source_roots_fpath"] = os.path.join(
        runtime_index_dir,
        "roots.json",
    )
    base_info["mmcif_dir"] = runtime_index_dir
    base_info["bioassembly_dict_dir"] = runtime_index_dir
    base_info["require_resolved_protein_crop"] = True
    base_info["min_diffusion_supervised_tokens"] = 4
    # Each AFDB/MGnify predicted monomer has one direct sibling A3M, which is
    # consumed exclusively as protein unpaired MSA.  Let that sole raw pool
    # use the complete 16,384-row assembled-pool budget; paired remains on the
    # global 10,000-row limit but has no homolog source for these datasets.
    config["msa"]["max_input_unpaired_msa_depth"] = 16384
    config["augmentation"]["query_only_msa_probability"] = 0.1
    config["augmentation"]["pair_msa_as_unpaired_probability"] = 0.3
    return config


data_configs["afdb_mgnify_short_distillation_v20260725"] = (
    _partitioned_afdb_mgnify_distillation_config(
        AFDB_MGNIFY_SHORT_DISTILLATION_RUNTIME_INDEX_DIR,
    )
)
data_configs["afdb_mgnify_long_distillation_v20260725"] = (
    _partitioned_afdb_mgnify_distillation_config(
        AFDB_MGNIFY_LONG_DISTILLATION_RUNTIME_INDEX_DIR,
    )
)
data_configs["afdb_mgnify_legacy_residual_distillation_v20260804"] = (
    _partitioned_afdb_mgnify_distillation_config(
        AFDB_MGNIFY_LEGACY_RESIDUAL_RUNTIME_INDEX_DIR,
    )
)
data_configs["all_sources_ge200_structural_distillation_v20260804"] = (
    _partitioned_afdb_mgnify_distillation_config(
        ALL_SOURCES_GE200_STRUCTURAL_RUNTIME_INDEX_DIR,
    )
)
