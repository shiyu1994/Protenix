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

import dataclasses
import gzip
import hashlib
import os
import pickle
import random
import time
from datetime import datetime, timedelta
from os.path import exists as opexists, join as opjoin
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
from biotite.structure import AtomArray
from typing_extensions import Self, TypeAlias

from protenix.data.constants import (
    DNA_CHAIN,
    LIGAND_CHAIN_TYPES,
    PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37,
    PROTEIN_CHAIN,
    RNA_CHAIN,
)
from protenix.data.msa.msa_utils import map_to_standard
from protenix.data.template.evaluation_template_cache import (
    EVALUATION_CACHE_CUTOFF_TEXT,
    EVALUATION_CACHE_MAX_TEMPLATES,
    EVALUATION_CACHE_POLICY_VERSION,
    EVALUATION_CACHE_SCHEMA_VERSION,
)
from protenix.data.template.template_parser import HHRParser, HmmsearchA3MParser
from protenix.data.template.template_utils import (
    DAYS_BEFORE_QUERY_DATE,
    DistogramFeaturesConfig,
    TEMPLATE_FEATURES,
    TemplateFeatures,
    TemplateHitFeaturizer,
    _sample_template_count,
)
from protenix.data.utils import pad_to
from protenix.utils.file_io import load_json_cached
from protenix.utils.logger import get_logger

logger = get_logger(__name__)

BatchDict: TypeAlias = dict[str, np.ndarray]
FeatureDict: TypeAlias = Mapping[str, np.ndarray]

TEMPLATE_USAGE_FORCE_NO_TEMPLATE = "force_no_template"
TEMPLATE_USAGE_USE_REAL_TEMPLATE = "use_real_template"
TemplateUsageDecision: TypeAlias = Literal[
    "force_no_template",
    "use_real_template",
]
_VALID_TEMPLATE_USAGE_DECISIONS = frozenset(
    {
        TEMPLATE_USAGE_FORCE_NO_TEMPLATE,
        TEMPLATE_USAGE_USE_REAL_TEMPLATE,
    }
)

SEQUENCE_TEMPLATE_CACHE_SCHEMA_VERSION = 1
SEQUENCE_TEMPLATE_CACHE_POLICY_VERSION = "top60_uniform20_identity_le_0.90_v1"
SEQUENCE_TEMPLATE_CACHE_MODES = frozenset({"required", "online", "disabled"})
SEQUENCE_TEMPLATE_CACHE_ERROR_POLICIES = frozenset({"raise", "chain_dummy"})
EVALUATION_SEQUENCE_TEMPLATE_CACHE_SCHEMA_VERSION = EVALUATION_CACHE_SCHEMA_VERSION
EVALUATION_SEQUENCE_TEMPLATE_CACHE_POLICY_VERSION = EVALUATION_CACHE_POLICY_VERSION
EVALUATION_SEQUENCE_TEMPLATE_CACHE_MAX_DATE = EVALUATION_CACHE_CUTOFF_TEXT
EVALUATION_SEQUENCE_TEMPLATE_CACHE_MODES = frozenset({"required", "online"})
EVALUATION_SEQUENCE_TEMPLATE_CACHE_EFFECTIVE_MODE = "evaluation_required"
EVALUATION_SEQUENCE_TEMPLATE_CACHE_MAX_TEMPLATES = EVALUATION_CACHE_MAX_TEMPLATES
SEQUENCE_TEMPLATE_CACHE_ERROR_REASONS = (
    "mapping_missing",
    "unsafe_path",
    "missing",
    "unreadable",
    "invalid_payload",
    "unknown",
)
_RAW_TEMPLATE_FEATURE_KEYS = frozenset(
    {
        "template_aatype",
        "template_all_atom_masks",
        "template_all_atom_positions",
        "template_domain_names",
        "template_release_date",
        "template_sequence",
    }
)


def _unwrap_cached_scalar(value: Any, *, field_name: str) -> Any:
    """Return a cache scalar while rejecting sequence-like containers."""
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(
                f"{field_name} must be scalar, got shape {value.shape}"
            )
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple, dict, set)):
        raise TypeError(
            f"{field_name} must be scalar, got {type(value).__name__}"
        )
    return value


def _validate_cached_template_features(
    features: Mapping[str, Any],
    *,
    query_sequence: str,
    candidate_index: int,
) -> Dict[str, Any]:
    """Validate and canonicalize one cached atom37 template feature record."""
    prefix = f"candidate {candidate_index}"
    query_length = len(query_sequence)

    aatype = features["template_aatype"]
    if not isinstance(aatype, np.ndarray):
        raise TypeError(f"{prefix} template_aatype must be an ndarray")
    if aatype.shape != (query_length,):
        raise ValueError(
            f"{prefix} template_aatype shape {aatype.shape} != "
            f"({query_length},)"
        )
    if not np.issubdtype(aatype.dtype, np.integer):
        raise TypeError(
            f"{prefix} template_aatype dtype {aatype.dtype} is not integer"
        )
    if np.any(aatype < 0) or np.any(
        aatype >= PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37.shape[0]
    ):
        raise ValueError(f"{prefix} template_aatype contains invalid residue ids")

    atom_masks = features["template_all_atom_masks"]
    expected_mask_shape = (query_length, 37)
    if not isinstance(atom_masks, np.ndarray):
        raise TypeError(
            f"{prefix} template_all_atom_masks must be an ndarray"
        )
    if atom_masks.shape != expected_mask_shape:
        raise ValueError(
            f"{prefix} template_all_atom_masks shape {atom_masks.shape} != "
            f"{expected_mask_shape}"
        )
    if not (
        np.issubdtype(atom_masks.dtype, np.number)
        and not np.issubdtype(atom_masks.dtype, np.complexfloating)
    ):
        raise TypeError(
            f"{prefix} template_all_atom_masks dtype {atom_masks.dtype} "
            "is not real numeric"
        )
    if not np.isfinite(atom_masks).all():
        raise ValueError(
            f"{prefix} template_all_atom_masks contains non-finite values"
        )
    if np.any(atom_masks < 0) or np.any(atom_masks > 1):
        raise ValueError(
            f"{prefix} template_all_atom_masks contains values outside [0, 1]"
        )

    atom_positions = features["template_all_atom_positions"]
    expected_position_shape = (query_length, 37, 3)
    if not isinstance(atom_positions, np.ndarray):
        raise TypeError(
            f"{prefix} template_all_atom_positions must be an ndarray"
        )
    if atom_positions.shape != expected_position_shape:
        raise ValueError(
            f"{prefix} template_all_atom_positions shape "
            f"{atom_positions.shape} != {expected_position_shape}"
        )
    if not (
        np.issubdtype(atom_positions.dtype, np.number)
        and not np.issubdtype(atom_positions.dtype, np.complexfloating)
    ):
        raise TypeError(
            f"{prefix} template_all_atom_positions dtype "
            f"{atom_positions.dtype} is not real numeric"
        )
    if not np.isfinite(atom_positions).all():
        raise ValueError(
            f"{prefix} template_all_atom_positions contains non-finite values"
        )

    domain_name = _unwrap_cached_scalar(
        features["template_domain_names"],
        field_name=f"{prefix} template_domain_names",
    )
    if not isinstance(domain_name, (str, bytes, bytearray)):
        raise TypeError(
            f"{prefix} template_domain_names must contain text or bytes"
        )
    if isinstance(domain_name, str):
        domain_name = domain_name.encode("utf-8")
    else:
        domain_name = bytes(domain_name)
    if not domain_name:
        raise ValueError(f"{prefix} template_domain_names cannot be empty")

    release_date = _unwrap_cached_scalar(
        features["template_release_date"],
        field_name=f"{prefix} template_release_date",
    )
    if not isinstance(release_date, (str, bytes, bytearray)):
        raise TypeError(
            f"{prefix} template_release_date must contain text or bytes"
        )
    if isinstance(release_date, str):
        release_date_text = release_date
    else:
        release_date_text = bytes(release_date).decode("utf-8")
    datetime.fromisoformat(release_date_text)

    template_sequence = _unwrap_cached_scalar(
        features["template_sequence"],
        field_name=f"{prefix} template_sequence",
    )
    if not isinstance(template_sequence, (str, bytes, bytearray)):
        raise TypeError(
            f"{prefix} template_sequence must contain text or bytes"
        )
    if isinstance(template_sequence, str):
        template_sequence_text = template_sequence
    else:
        template_sequence_text = bytes(template_sequence).decode("utf-8")
    if len(template_sequence_text) != query_length:
        raise ValueError(
            f"{prefix} template_sequence length "
            f"{len(template_sequence_text)} != {query_length}"
        )

    canonical = dict(features)
    canonical["template_domain_names"] = np.array(
        domain_name,
        dtype=object,
    )
    canonical["template_release_date"] = np.array(
        release_date_text.encode("utf-8"),
        dtype=object,
    )
    canonical["template_sequence"] = template_sequence_text.encode("utf-8")
    return canonical


class SequenceTemplateCacheError(RuntimeError):
    """Raised when a required per-sequence template cache is unusable."""

    def __init__(
        self,
        message: str,
        *,
        reason: str = "unknown",
        cache_path: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.cache_path = cache_path


class TemplateSourceManager:
    """
    Manages template data retrieval and loading from multiple sources.

    Args:
        raw_paths: List of base paths for template storage.
        indexing_methods: List of indexing methods (e.g., 'sequence' or 'pdb_id').
        mappings: Dictionary mapping source index to its respective lookup table.
        enabled: Whether template loading is enabled.
    """

    def __init__(
        self,
        raw_paths: Sequence[str],
        indexing_methods: Sequence[str],
        mappings: Dict[int, Dict[str, Any]],
        enabled: bool = True,
    ) -> None:
        self.raw_paths = raw_paths
        self.indexing_methods = indexing_methods
        self.mappings = mappings
        self.enabled = enabled

    def fetch_template_paths(
        self, pdb_id: str, query_sequence: str, chain_entity_type: str
    ) -> List[str]:
        """
        Fetches template file paths from the configured sources.

        Args:
            pdb_id: PDB identifier of the query.
            query_sequence: Query sequence.
            chain_entity_type: Type of the chain (e.g., PROTEIN_CHAIN).

        Returns:
            A list of paths to found template files (.a3m or .hhr).
        """
        if not self.enabled or chain_entity_type != PROTEIN_CHAIN:
            return []

        template_paths = []
        for i, (path, method) in enumerate(zip(self.raw_paths, self.indexing_methods)):
            mapping = self.mappings.get(i, {})
            key = pdb_id if method == "pdb_id" else query_sequence

            if key not in mapping:
                continue

            dir_path = opjoin(path, str(mapping[key]))
            # Check for multiple possible template filenames
            possible_subpaths = [
                "hmmsearch.a3m",
                "concat.hhr",
            ]
            for subpath in possible_subpaths:
                full_path = opjoin(dir_path, subpath)
                if opexists(full_path):
                    template_paths.append(full_path)
        return template_paths

    def fetch_sequence_relative_dir(self, query_sequence: str) -> Optional[str]:
        """Return the sequence-indexed source directory used by the cache builder."""
        for i, method in enumerate(self.indexing_methods):
            if method != "sequence":
                continue
            mapping = self.mappings.get(i, {})
            if query_sequence in mapping:
                return str(mapping[query_sequence])
        return None


class TemplateFeatureAssemblyLine:
    """
    Orchestrates the conversion of raw templates into finalized Protenix features.

    Args:
        max_templates: Maximum number of templates to include in the features.
    """

    def __init__(self, max_templates: int = 4) -> None:
        self.max_templates = max_templates

    def assemble(
        self,
        bioassembly: Mapping[int, Mapping[str, Any]],
        standard_token_idxs: np.ndarray,
    ) -> "Templates":
        """
        Executes the complete feature assembly pipeline.

        Args:
            bioassembly: Mapping of asymmetric IDs to chain information.
            standard_token_idxs: Array of standardized residue indices.

        Returns:
            An assembled Templates object.
        """
        np_chains_list = []
        polymer_entity_features = {True: {}, False: {}}
        # Identify entities where template features can be safely copied (same sequence)
        safe_entity_ids = get_safe_entity_id_for_template_copy(bioassembly)

        for asym_id, info in bioassembly.items():
            chain_id = info["chain_id"]
            entity_id = info["entity_id"]
            chain_type = info["chain_entity_type"]
            num_tokens = len(info["sequence"])

            # Templates are currently only supported for protein chains with sufficient length
            skip_chain = chain_type != PROTEIN_CHAIN or num_tokens <= 4

            if (entity_id not in polymer_entity_features[skip_chain]) or (
                entity_id not in safe_entity_ids
            ):
                templates = info["templates"]
                if skip_chain or not templates:
                    template_features = TemplateFeatures.empty_template_features(
                        num_tokens
                    )
                else:
                    # Package and fix template features
                    template_features = TemplateFeatures.package_template_features(
                        hit_features=templates
                    )
                    template_features = TemplateFeatures.fix_template_features(
                        template_features=template_features,
                        num_res=num_tokens,
                    )
                # Reduce to requested maximum number of templates
                template_features = _reduce_template_features(
                    template_features, self.max_templates
                )
                if entity_id in safe_entity_ids:
                    polymer_entity_features[skip_chain][entity_id] = template_features

            if entity_id in safe_entity_ids:
                feats = polymer_entity_features[skip_chain][entity_id].copy()
            else:
                feats = template_features

            feats["chain_id"] = chain_id
            np_chains_list.append(feats)

        # Pad the number of templates to max_templates for each chain to allow concatenation
        for chain in np_chains_list:
            chain["template_aatype"] = pad_to(
                chain["template_aatype"], (self.max_templates, None)
            )
            chain["template_atom_positions"] = pad_to(
                chain["template_atom_positions"],
                (self.max_templates, None, None, None),
            )
            chain["template_atom_mask"] = pad_to(
                chain["template_atom_mask"], (self.max_templates, None, None)
            )

        # Concatenate features along the residue dimension
        merged_example = {
            ft: np.concatenate([c[ft] for c in np_chains_list], axis=1)
            for ft in np_chains_list[0]
            if ft in TEMPLATE_FEATURES
        }

        # Crop/index merged features using standard token indices
        for feature_name, v in merged_example.items():
            merged_example[feature_name] = v[
                : self.max_templates, standard_token_idxs, ...
            ]

        return Templates(
            aatype=merged_example["template_aatype"],
            atom_positions=merged_example["template_atom_positions"],
            atom_mask=merged_example["template_atom_mask"].astype(bool),
        )


class TemplateFeaturizer:
    """
    Main class for template featurization in training and evaluation.

    Args:
        dataset_name: Name of the dataset.
        prot_template_mmcif_dir: Directory containing template mmCIF files.
        prot_template_cache_dir: Directory for cached template data.
        prot_template_raw_paths: Base paths for protein template storage.
        prot_seq_or_filename_to_templatedir_jsons: JSON files mapping sequences/PDB IDs to template directories.
        prot_indexing_methods: Indexing methods for each source.
        enable_prot_template: Whether to enable protein templates.
        template_dropout_rate: Dropout rate for templates during training.
        sequence_template_cache_root: Root containing per-sequence cache files.
        sequence_template_cache_mode: ``required`` reads only the offline cache,
            ``online`` retains legacy online featurization, and ``disabled``
            explicitly yields no real templates.
        sequence_template_cache_error_policy: ``raise`` preserves fail-fast
            behavior. ``chain_dummy`` converts an unusable required-cache
            artifact into fixed dummy features for every physical chain that
            shares the affected unique query sequence; other sequences in the
            same target retain their templates.
        sequence_template_cache_filename: Cache filename below the sequence's
            source-relative directory.
        evaluation_sequence_template_cache_root: Root of immutable,
            SHA256-keyed evaluation template artifacts.
        evaluation_sequence_template_cache_mode: ``required`` opts a
            non-training dataset into the evaluation cache. ``online`` keeps
            the historical date-filtered implementation.
        evaluation_sequence_template_cache_filename: Cache filename below
            ``root/<sha256[:2]>/<sha256>/``.
        evaluation_sequence_template_cache_policy: Required immutable
            evaluation-cache policy identifier.
        stage: 'train' or 'eval'.
        kalign_binary_path: Path to kalign binary.
        release_dates_path: Path to PDB release dates JSON.
        obsolete_pdbs_path: Path to obsolete PDBs mapping JSON.
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        dataset_name: str = "",
        prot_template_mmcif_dir: str = "",
        prot_template_cache_dir: str = None,
        prot_template_raw_paths: Sequence[str] = [""],
        prot_seq_or_filename_to_templatedir_jsons: Sequence[str] = [""],
        prot_indexing_methods: Sequence[str] = ["sequence"],
        enable_prot_template: bool = True,
        template_dropout_rate: float = 0.0,
        sequence_template_cache_root: str = "",
        sequence_template_cache_mode: str = "online",
        sequence_template_cache_error_policy: str = "raise",
        sequence_template_cache_filename: str = "template_features.pkl.gz",
        evaluation_sequence_template_cache_root: str = "",
        evaluation_sequence_template_cache_mode: str = "online",
        evaluation_sequence_template_cache_filename: str = (
            "template_features.pkl.gz"
        ),
        evaluation_sequence_template_cache_policy: str = (
            EVALUATION_SEQUENCE_TEMPLATE_CACHE_POLICY_VERSION
        ),
        stage: str = "train",
        kalign_binary_path: str = "",
        release_dates_path: str = "",
        obsolete_pdbs_path: str = "",
        **kwargs,
    ) -> None:
        try:
            template_dropout_rate = float(template_dropout_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError("template_dropout_rate must be a finite number") from exc
        if (
            not np.isfinite(template_dropout_rate)
            or not 0.0 <= template_dropout_rate <= 1.0
        ):
            raise ValueError("template_dropout_rate must be in [0, 1]")
        if not isinstance(sequence_template_cache_mode, str):
            raise TypeError("sequence_template_cache_mode must be a string")
        sequence_template_cache_mode = sequence_template_cache_mode.strip().lower()
        if sequence_template_cache_mode not in SEQUENCE_TEMPLATE_CACHE_MODES:
            allowed = ", ".join(sorted(SEQUENCE_TEMPLATE_CACHE_MODES))
            raise ValueError(
                "sequence_template_cache_mode must be one of "
                f"{{{allowed}}}, got {sequence_template_cache_mode!r}"
            )
        if not isinstance(sequence_template_cache_error_policy, str):
            raise TypeError(
                "sequence_template_cache_error_policy must be a string"
            )
        sequence_template_cache_error_policy = (
            sequence_template_cache_error_policy.strip().lower()
        )
        if (
            sequence_template_cache_error_policy
            not in SEQUENCE_TEMPLATE_CACHE_ERROR_POLICIES
        ):
            allowed = ", ".join(
                sorted(SEQUENCE_TEMPLATE_CACHE_ERROR_POLICIES)
            )
            raise ValueError(
                "sequence_template_cache_error_policy must be one of "
                f"{{{allowed}}}, got "
                f"{sequence_template_cache_error_policy!r}"
            )
        if not isinstance(sequence_template_cache_root, str):
            raise TypeError("sequence_template_cache_root must be a string")
        if not isinstance(sequence_template_cache_filename, str):
            raise TypeError("sequence_template_cache_filename must be a string")
        sequence_template_cache_filename = sequence_template_cache_filename.strip()
        if (
            not sequence_template_cache_filename
            or os.path.basename(sequence_template_cache_filename)
            != sequence_template_cache_filename
        ):
            raise ValueError(
                "sequence_template_cache_filename must be a non-empty basename"
            )
        if (
            stage == "train"
            and sequence_template_cache_mode == "required"
            and not sequence_template_cache_root.strip()
        ):
            raise ValueError(
                "sequence_template_cache_root is required when "
                "sequence_template_cache_mode='required'"
            )

        # Evaluation caching is an explicit, independent contract.  Do not
        # validate or otherwise consult these fields for training so existing
        # training cache behavior remains unchanged.
        resolved_evaluation_cache_mode = "online"
        resolved_evaluation_cache_root = ""
        resolved_evaluation_cache_filename = "template_features.pkl.gz"
        resolved_evaluation_cache_policy = (
            EVALUATION_SEQUENCE_TEMPLATE_CACHE_POLICY_VERSION
        )
        if stage != "train":
            if not isinstance(evaluation_sequence_template_cache_mode, str):
                raise TypeError(
                    "evaluation_sequence_template_cache_mode must be a string"
                )
            resolved_evaluation_cache_mode = (
                evaluation_sequence_template_cache_mode.strip().lower()
            )
            if (
                resolved_evaluation_cache_mode
                not in EVALUATION_SEQUENCE_TEMPLATE_CACHE_MODES
            ):
                allowed = ", ".join(
                    sorted(EVALUATION_SEQUENCE_TEMPLATE_CACHE_MODES)
                )
                raise ValueError(
                    "evaluation_sequence_template_cache_mode must be one of "
                    f"{{{allowed}}}, got {resolved_evaluation_cache_mode!r}"
                )
            if resolved_evaluation_cache_mode == "required":
                if not isinstance(evaluation_sequence_template_cache_root, str):
                    raise TypeError(
                        "evaluation_sequence_template_cache_root must be a string"
                    )
                if not evaluation_sequence_template_cache_root.strip():
                    raise ValueError(
                        "evaluation_sequence_template_cache_root is required "
                        "when evaluation_sequence_template_cache_mode='required'"
                    )
                if not isinstance(
                    evaluation_sequence_template_cache_filename,
                    str,
                ):
                    raise TypeError(
                        "evaluation_sequence_template_cache_filename must be "
                        "a string"
                    )
                resolved_evaluation_cache_filename = (
                    evaluation_sequence_template_cache_filename.strip()
                )
                if (
                    not resolved_evaluation_cache_filename
                    or os.path.basename(resolved_evaluation_cache_filename)
                    != resolved_evaluation_cache_filename
                ):
                    raise ValueError(
                        "evaluation_sequence_template_cache_filename must be "
                        "a non-empty basename"
                    )
                if not isinstance(
                    evaluation_sequence_template_cache_policy,
                    str,
                ):
                    raise TypeError(
                        "evaluation_sequence_template_cache_policy must be "
                        "a string"
                    )
                resolved_evaluation_cache_policy = (
                    evaluation_sequence_template_cache_policy.strip()
                )
                if (
                    resolved_evaluation_cache_policy
                    != EVALUATION_SEQUENCE_TEMPLATE_CACHE_POLICY_VERSION
                ):
                    raise ValueError(
                        "evaluation_sequence_template_cache_policy must be "
                        f"{EVALUATION_SEQUENCE_TEMPLATE_CACHE_POLICY_VERSION!r}, "
                        f"got {resolved_evaluation_cache_policy!r}"
                    )
                resolved_evaluation_cache_root = os.path.abspath(
                    os.path.expanduser(
                        evaluation_sequence_template_cache_root.strip()
                    )
                )

        # Validate inputs
        assert len(prot_template_raw_paths) == len(
            prot_seq_or_filename_to_templatedir_jsons
        ), "Mismatch between raw paths and JSON mapping files."
        assert len(prot_template_raw_paths) == len(
            prot_indexing_methods
        ), "Mismatch between raw paths and indexing methods."

        # Currently, multiple source training strategy is not determined
        assert (
            len(prot_template_raw_paths) == 1
        ), "Only single source template supported."

        if stage != "train":
            max_template_date = "2021-09-30"
        else:
            if ("distillation" in dataset_name) or ("openprotein" in dataset_name):
                max_template_date = "2018-04-30"
            else:
                max_template_date = "3000-01-01"

        self.stage = stage
        self.max_template_date = datetime.strptime(max_template_date, "%Y-%m-%d")
        self.dataset_name = dataset_name
        self.enable_prot_template = enable_prot_template
        self.template_dropout_rate = template_dropout_rate
        self.sequence_template_cache_root = (
            os.path.abspath(os.path.expanduser(sequence_template_cache_root))
            if sequence_template_cache_root.strip()
            else ""
        )
        self.configured_sequence_template_cache_mode = (
            sequence_template_cache_mode
        )
        self.sequence_template_cache_error_policy = (
            sequence_template_cache_error_policy
        )
        self.evaluation_sequence_template_cache_root = (
            resolved_evaluation_cache_root
        )
        self.evaluation_sequence_template_cache_mode = (
            resolved_evaluation_cache_mode
        )
        self.evaluation_sequence_template_cache_filename = (
            resolved_evaluation_cache_filename
        )
        self.evaluation_sequence_template_cache_policy = (
            resolved_evaluation_cache_policy
        )
        # Training retains its exact historical mode. Non-training stages use
        # the SHA256 evaluation cache only after an explicit required opt-in.
        if stage == "train":
            self.sequence_template_cache_mode = sequence_template_cache_mode
        elif resolved_evaluation_cache_mode == "required":
            self.sequence_template_cache_mode = (
                EVALUATION_SEQUENCE_TEMPLATE_CACHE_EFFECTIVE_MODE
            )
        else:
            self.sequence_template_cache_mode = "online"
        self.sequence_template_cache_filename = sequence_template_cache_filename

        # Initialize source manager
        mappings = {
            i: load_json_cached(p)
            for i, p in enumerate(prot_seq_or_filename_to_templatedir_jsons)
        }
        self.source_mgr = TemplateSourceManager(
            raw_paths=prot_template_raw_paths,
            indexing_methods=prot_indexing_methods,
            mappings=mappings,
            enabled=enable_prot_template,
        )

        # Cache-required modes never initialize the expensive online
        # release-date/mmCIF machinery. This also makes accidental fallback
        # impossible.
        self.online_featurizer: Optional[TemplateHitFeaturizer] = None
        if self.sequence_template_cache_mode == "online":
            assert len(kalign_binary_path) > 0, "kalign_binary_path must be provided."
            assert len(release_dates_path) > 0, "release_dates_path must be provided."
            assert len(obsolete_pdbs_path) > 0, "obsolete_pdbs_path must be provided."
            self.online_featurizer = TemplateHitFeaturizer(
                mmcif_dir=prot_template_mmcif_dir,
                template_cache_dir=prot_template_cache_dir,
                max_hits=None if self.stage == "train" else 4,
                kalign_binary_path=kalign_binary_path,
                max_template_date=max_template_date,
                release_dates_path=release_dates_path,
                obsolete_pdbs_path=obsolete_pdbs_path,
                _shuffle_top_k_prefiltered=20 if self.stage == "train" else None,
            )

        self._last_profile: Dict[str, Any] = {
            "elapsed_seconds": 0.0,
            "raw_templates": 0,
            "selected_templates": 0,
            "valid_template_count": 0,
            "has_real_template": False,
            "drop_template_applied": False,
            "template_usage_decision": TEMPLATE_USAGE_USE_REAL_TEMPLATE,
            "template_usage_decision_source": "default",
            "sequence_template_cache_mode": self.sequence_template_cache_mode,
            "sequence_template_cache_error_policy": (
                self.sequence_template_cache_error_policy
            ),
            "sequence_template_cache_load_seconds": 0.0,
            "sequence_template_cache_files_loaded": 0,
            "sequence_template_cache_candidates": 0,
            "sequence_template_cache_chain_dummy_fallback_unique_sequence_count": 0,
            "sequence_template_cache_chain_dummy_fallback_affected_chain_count": 0,
            "sequence_template_cache_chain_dummy_fallback_reason_counts": {},
            "sequence_template_cache_chain_dummy_fallbacks": [],
            "chain_profiles": [],
        }

    def _resolve_template_usage_decision(
        self,
        template_usage_decision: Optional[TemplateUsageDecision],
    ) -> Tuple[TemplateUsageDecision, str]:
        """Resolve one occurrence's template policy.

        Passing an explicit decision never consumes the process-global NumPy
        RNG.  Omitting it retains the historical training dropout behavior.
        """
        if template_usage_decision is not None:
            if (
                not isinstance(template_usage_decision, str)
                or template_usage_decision not in _VALID_TEMPLATE_USAGE_DECISIONS
            ):
                allowed = ", ".join(sorted(_VALID_TEMPLATE_USAGE_DECISIONS))
                raise ValueError(
                    "template_usage_decision must be one of "
                    f"{{{allowed}}}, got {template_usage_decision!r}"
                )
            return template_usage_decision, "explicit"

        if self.stage == "train" and self.template_dropout_rate > 0.0:
            drop_template = np.random.uniform(0, 1) < self.template_dropout_rate
            return (
                TEMPLATE_USAGE_FORCE_NO_TEMPLATE
                if drop_template
                else TEMPLATE_USAGE_USE_REAL_TEMPLATE,
                "legacy_dropout_rate",
            )

        return TEMPLATE_USAGE_USE_REAL_TEMPLATE, "default"

    @staticmethod
    def _sequence_sha256(sequence: str) -> str:
        return hashlib.sha256(sequence.encode("utf-8")).hexdigest()

    def _evaluation_sequence_template_cache_path(
        self,
        query_sequence: str,
    ) -> str:
        """Resolve an immutable evaluation artifact by exact sequence hash."""
        digest = self._sequence_sha256(query_sequence)
        cache_path = os.path.abspath(
            os.path.join(
                self.evaluation_sequence_template_cache_root,
                digest[:2],
                digest,
                self.evaluation_sequence_template_cache_filename,
            )
        )
        if (
            os.path.commonpath(
                [self.evaluation_sequence_template_cache_root, cache_path]
            )
            != self.evaluation_sequence_template_cache_root
        ):
            raise SequenceTemplateCacheError(
                f"evaluation template cache path escapes configured root: "
                f"{cache_path}",
                reason="unsafe_path",
                cache_path=cache_path,
            )
        return cache_path

    def _load_evaluation_template_candidates(
        self,
        query_sequence: str,
    ) -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
        """Load one fixed, date-filtered evaluation template artifact.

        Unlike the training cache, evaluation returns every cached candidate in
        file order. It deliberately performs no count sampling or permutation.
        """
        cache_path = self._evaluation_sequence_template_cache_path(
            query_sequence
        )
        load_start = time.perf_counter()
        try:
            opener = gzip.open if cache_path.endswith(".gz") else open
            with opener(cache_path, "rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            elapsed = time.perf_counter() - load_start
            reason = (
                "missing"
                if isinstance(exc, FileNotFoundError)
                else "unreadable"
            )
            raise SequenceTemplateCacheError(
                "failed to load required evaluation sequence template cache "
                f"{cache_path!r} after {elapsed:.3f}s: {exc}",
                reason=reason,
                cache_path=cache_path,
            ) from exc
        load_seconds = time.perf_counter() - load_start

        try:
            if not isinstance(payload, Mapping):
                raise TypeError(
                    f"payload must be a mapping, got {type(payload).__name__}"
                )
            if (
                payload.get("schema_version")
                != EVALUATION_SEQUENCE_TEMPLATE_CACHE_SCHEMA_VERSION
            ):
                raise ValueError(
                    "schema_version mismatch: "
                    f"{payload.get('schema_version')!r} != "
                    f"{EVALUATION_SEQUENCE_TEMPLATE_CACHE_SCHEMA_VERSION!r}"
                )
            if (
                payload.get("policy_version")
                != self.evaluation_sequence_template_cache_policy
            ):
                raise ValueError(
                    "policy_version mismatch: "
                    f"{payload.get('policy_version')!r} != "
                    f"{self.evaluation_sequence_template_cache_policy!r}"
                )
            if (
                payload.get("cutoff_date")
                != EVALUATION_SEQUENCE_TEMPLATE_CACHE_MAX_DATE
            ):
                raise ValueError(
                    "cutoff_date mismatch: "
                    f"{payload.get('cutoff_date')!r} != "
                    f"{EVALUATION_SEQUENCE_TEMPLATE_CACHE_MAX_DATE!r}"
                )
            if (
                payload.get("max_templates")
                != EVALUATION_SEQUENCE_TEMPLATE_CACHE_MAX_TEMPLATES
            ):
                raise ValueError(
                    "max_templates mismatch: "
                    f"{payload.get('max_templates')!r} != "
                    f"{EVALUATION_SEQUENCE_TEMPLATE_CACHE_MAX_TEMPLATES!r}"
                )

            expected_sha256 = self._sequence_sha256(query_sequence)
            if payload.get("sequence_sha256") != expected_sha256:
                raise ValueError(
                    "sequence_sha256 does not match requested sequence"
                )
            if payload.get("sequence") != query_sequence:
                raise ValueError("payload sequence does not match requested sequence")

            status = payload.get("status")
            if status not in {"complete", "complete_empty"}:
                raise ValueError(
                    "status must be 'complete' or 'complete_empty', "
                    f"got {status!r}"
                )
            candidate_records = payload.get("candidates")
            if not isinstance(candidate_records, list):
                raise TypeError("candidates must be a list")
            if (
                len(candidate_records)
                > EVALUATION_SEQUENCE_TEMPLATE_CACHE_MAX_TEMPLATES
            ):
                raise ValueError(
                    f"evaluation cache contains {len(candidate_records)} "
                    "candidates; maximum is "
                    f"{EVALUATION_SEQUENCE_TEMPLATE_CACHE_MAX_TEMPLATES}"
                )
            if status == "complete" and not candidate_records:
                raise ValueError(
                    "complete evaluation cache must contain at least one candidate"
                )
            if status == "complete_empty" and candidate_records:
                raise ValueError(
                    "complete_empty evaluation cache must not contain candidates"
                )
            counts = payload.get("counts")
            if not isinstance(counts, Mapping):
                raise TypeError("evaluation cache counts must be a mapping")
            raw_hit_count = counts.get("raw_hit_count")
            if (
                not isinstance(raw_hit_count, int)
                or isinstance(raw_hit_count, bool)
                or raw_hit_count < len(candidate_records)
            ):
                raise ValueError(
                    "counts.raw_hit_count must be an integer greater than or "
                    "equal to the cached candidate count"
                )

            cutoff = datetime.strptime(
                EVALUATION_SEQUENCE_TEMPLATE_CACHE_MAX_DATE,
                "%Y-%m-%d",
            )
            candidates: List[Dict[str, Any]] = []
            for index, candidate in enumerate(candidate_records):
                if not isinstance(candidate, Mapping):
                    raise TypeError(f"candidate {index} must be a mapping")
                features = candidate.get("features")
                if not isinstance(features, Mapping):
                    raise TypeError(
                        f"candidate {index} features must be a mapping"
                    )
                missing_keys = _RAW_TEMPLATE_FEATURE_KEYS.difference(features)
                if missing_keys:
                    raise ValueError(
                        f"candidate {index} is missing raw feature keys: "
                        + ", ".join(sorted(missing_keys))
                    )
                canonical = _validate_cached_template_features(
                    features,
                    query_sequence=query_sequence,
                    candidate_index=index,
                )
                release_date_scalar = _unwrap_cached_scalar(
                    canonical["template_release_date"],
                    field_name=(
                        f"candidate {index} template_release_date"
                    ),
                )
                if isinstance(release_date_scalar, str):
                    release_date_text = release_date_scalar
                else:
                    release_date_text = bytes(release_date_scalar).decode(
                        "utf-8"
                    )
                release_date = datetime.strptime(
                    release_date_text,
                    "%Y-%m-%d",
                )
                if release_date > cutoff:
                    raise ValueError(
                        f"candidate {index} release date "
                        f"{release_date_text!r} exceeds cutoff "
                        f"{EVALUATION_SEQUENCE_TEMPLATE_CACHE_MAX_DATE!r}"
                    )
                candidates.append(canonical)

            if candidates:
                packaged = TemplateFeatures.package_template_features(
                    hit_features=candidates
                )
                TemplateFeatures.fix_template_features(
                    template_features=packaged,
                    num_res=len(query_sequence),
                )
        except Exception as exc:
            raise SequenceTemplateCacheError(
                "invalid required evaluation sequence template cache "
                f"{cache_path!r}: {exc}",
                reason="invalid_payload",
                cache_path=cache_path,
            ) from exc

        num_candidates = len(candidates)
        time_track: Dict[str, Any] = {
            "template_source": "evaluation_sequence_cache",
            "sequence_template_cache_path": cache_path,
            "sequence_template_cache_status": payload["status"],
            "sequence_template_cache_policy": payload["policy_version"],
            "sequence_template_cache_cutoff_date": payload["cutoff_date"],
            "sequence_template_cache_max_templates": payload["max_templates"],
            "sequence_template_cache_load_seconds": load_seconds,
            "sequence_template_cache_candidates": num_candidates,
            "sequence_template_cache_selected": num_candidates,
        }
        return candidates, raw_hit_count, time_track

    def _sequence_template_cache_path(self, query_sequence: str) -> str:
        relative_dir = self.source_mgr.fetch_sequence_relative_dir(query_sequence)
        if relative_dir is None:
            raise SequenceTemplateCacheError(
                "required sequence template cache cannot resolve the query "
                f"sequence in a sequence-indexed source mapping "
                f"(sha256={self._sequence_sha256(query_sequence)})",
                reason="mapping_missing",
            )
        normalized = os.path.normpath(relative_dir)
        if (
            os.path.isabs(relative_dir)
            or normalized == os.pardir
            or normalized.startswith(os.pardir + os.sep)
        ):
            raise SequenceTemplateCacheError(
                "unsafe source-relative template cache directory "
                f"{relative_dir!r}",
                reason="unsafe_path",
            )
        cache_path = os.path.abspath(
            os.path.join(
                self.sequence_template_cache_root,
                normalized,
                self.sequence_template_cache_filename,
            )
        )
        root_prefix = self.sequence_template_cache_root.rstrip(os.sep) + os.sep
        if not cache_path.startswith(root_prefix):
            raise SequenceTemplateCacheError(
                f"template cache path escapes configured root: {cache_path}",
                reason="unsafe_path",
                cache_path=cache_path,
            )
        return cache_path

    def _load_cached_template_candidates(
        self,
        query_sequence: str,
        *,
        minimum_template_count: int,
    ) -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
        """Load and sample one occurrence from an offline sequence cache."""
        cache_path = self._sequence_template_cache_path(query_sequence)
        load_start = time.perf_counter()
        try:
            opener = gzip.open if cache_path.endswith(".gz") else open
            with opener(cache_path, "rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            elapsed = time.perf_counter() - load_start
            reason = (
                "missing"
                if isinstance(exc, FileNotFoundError)
                else "unreadable"
            )
            raise SequenceTemplateCacheError(
                "failed to load required sequence template cache "
                f"{cache_path!r} after {elapsed:.3f}s: {exc}",
                reason=reason,
                cache_path=cache_path,
            ) from exc
        load_seconds = time.perf_counter() - load_start

        try:
            if not isinstance(payload, Mapping):
                raise TypeError(
                    f"payload must be a mapping, got {type(payload).__name__}"
                )
            schema_version = payload.get("schema_version")
            if schema_version != SEQUENCE_TEMPLATE_CACHE_SCHEMA_VERSION:
                raise ValueError(
                    "schema_version mismatch: "
                    f"{schema_version!r} != "
                    f"{SEQUENCE_TEMPLATE_CACHE_SCHEMA_VERSION}"
                )
            policy_version = payload.get(
                "policy_version",
                payload.get("policy"),
            )
            if isinstance(policy_version, Mapping):
                policy_version = policy_version.get("version")
            if policy_version != SEQUENCE_TEMPLATE_CACHE_POLICY_VERSION:
                raise ValueError(
                    "policy version mismatch: "
                    f"{policy_version!r} != "
                    f"{SEQUENCE_TEMPLATE_CACHE_POLICY_VERSION!r}"
                )

            expected_sha256 = self._sequence_sha256(query_sequence)
            if payload.get("sequence_sha256") != expected_sha256:
                raise ValueError(
                    "sequence_sha256 does not match requested sequence"
                )
            payload_sequence = payload.get("sequence")
            if payload_sequence is not None and payload_sequence != query_sequence:
                raise ValueError("payload sequence does not match requested sequence")

            status = payload.get("status")
            if status not in {"complete", "complete_empty"}:
                raise ValueError(
                    "status must be 'complete' or 'complete_empty', "
                    f"got {status!r}"
                )
            candidate_records = payload.get("candidates")
            if not isinstance(candidate_records, list):
                raise TypeError("candidates must be a list")
            if len(candidate_records) > 20:
                raise ValueError(
                    f"cache contains {len(candidate_records)} candidates; maximum is 20"
                )
            if status == "complete" and not candidate_records:
                raise ValueError(
                    "complete cache must contain at least one candidate"
                )
            if status == "complete_empty" and candidate_records:
                raise ValueError(
                    "complete_empty cache must not contain candidates"
                )

            candidates: List[Dict[str, Any]] = []
            for index, candidate in enumerate(candidate_records):
                if not isinstance(candidate, Mapping):
                    raise TypeError(f"candidate {index} must be a mapping")
                identity = candidate.get("alignment_identity")
                if identity is None:
                    raise ValueError(
                        f"candidate {index} is missing alignment_identity"
                    )
                identity = float(identity)
                if not np.isfinite(identity) or not 0.0 <= identity <= 0.90 + 1e-8:
                    raise ValueError(
                        f"candidate {index} identity {identity!r} is outside "
                        "the allowed [0.0, 0.90] range"
                    )
                features = candidate.get("features", candidate)
                if not isinstance(features, Mapping):
                    raise TypeError(
                        f"candidate {index} features must be a mapping"
                    )
                missing_keys = _RAW_TEMPLATE_FEATURE_KEYS.difference(features)
                if missing_keys:
                    raise ValueError(
                        f"candidate {index} is missing raw feature keys: "
                        + ", ".join(sorted(missing_keys))
                    )
                candidates.append(
                    _validate_cached_template_features(
                        features,
                        query_sequence=query_sequence,
                        candidate_index=index,
                    )
                )
        except Exception as exc:
            raise SequenceTemplateCacheError(
                f"invalid required sequence template cache {cache_path!r}: {exc}",
                reason="invalid_payload",
                cache_path=cache_path,
            ) from exc

        num_candidates = len(candidates)
        max_to_collect = _sample_template_count(
            num_candidates=num_candidates,
            max_hits=None,
            minimum_template_count=minimum_template_count,
        )
        if num_candidates:
            permutation = np.random.permutation(num_candidates)
            selected = [
                candidates[int(index)]
                for index in permutation[:max_to_collect]
            ]
        else:
            selected = []
        if selected:
            try:
                packaged = TemplateFeatures.package_template_features(
                    hit_features=selected
                )
                TemplateFeatures.fix_template_features(
                    template_features=packaged,
                    num_res=len(query_sequence),
                )
            except Exception as exc:
                raise SequenceTemplateCacheError(
                    "selected features in required sequence template cache "
                    f"{cache_path!r} cannot be assembled: {exc}",
                    reason="invalid_payload",
                    cache_path=cache_path,
                ) from exc

        time_track: Dict[str, Any] = {
            "template_source": "sequence_cache",
            "sequence_template_cache_path": cache_path,
            "sequence_template_cache_status": payload["status"],
            "sequence_template_cache_load_seconds": load_seconds,
            "sequence_template_cache_candidates": num_candidates,
            "sequence_template_cache_selected": len(selected),
        }
        return selected, num_candidates, time_track

    def get_template(
        self,
        pdb_id: str,
        query_sequence: str,
        sequence_uid: str,
        query_release_date: Optional[datetime],
        chain_entity_type: str,
        *,
        minimum_template_count: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
        """
        Fetches and processes templates for a single sequence.

        Args:
            pdb_id: PDB identifier.
            query_sequence: Query sequence string.
            sequence_uid: Unique identifier for the sequence.
            query_release_date: Release date of the query structure.
            chain_entity_type: Type of the chain.

        Returns:
            A tuple of (template_features_list, raw_hit_count, timing_track).
        """
        if not self.enable_prot_template or chain_entity_type != PROTEIN_CHAIN:
            return [], 0, {}

        if self.sequence_template_cache_mode == "disabled":
            return (
                [],
                0,
                {
                    "template_source": "disabled",
                    "sequence_template_cache_candidates": 0,
                    "sequence_template_cache_selected": 0,
                },
            )
        if (
            self.sequence_template_cache_mode
            == EVALUATION_SEQUENCE_TEMPLATE_CACHE_EFFECTIVE_MODE
        ):
            return self._load_evaluation_template_candidates(query_sequence)
        if self.sequence_template_cache_mode == "required":
            # Cached features are global per sequence by design. In particular,
            # no query release-date cutoff is applied on this path.
            return self._load_cached_template_candidates(
                query_sequence,
                minimum_template_count=minimum_template_count,
            )

        # Fetch template paths from sources
        template_paths = self.source_mgr.fetch_template_paths(
            pdb_id, query_sequence, chain_entity_type
        )

        hmmsearched_a3m, hmmsearched_hhr = "", ""
        for path in template_paths:
            with open(path, "r") as f:
                content = f.read()
            if not content:
                continue
            if path.endswith(".a3m"):
                hmmsearched_a3m += content
            elif path.endswith(".hhr"):
                hmmsearched_hhr += content

        # Determine cutoff date for filtering templates
        if query_release_date is None:
            cutoff_date = self.max_template_date
        else:
            cutoff_date = min(
                self.max_template_date,
                query_release_date - timedelta(days=DAYS_BEFORE_QUERY_DATE),
            )

        # Parse hits from a3m or hhr content
        if hmmsearched_hhr:
            hits = HHRParser.parse(hhr_string=hmmsearched_hhr)
        else:
            hits = HmmsearchA3MParser.parse(
                query_seq=query_sequence, a3m_str=hmmsearched_a3m, skip_first=False
            )

        raw_hit_count = len(hits)
        if raw_hit_count == 0:
            return [], 0, {}

        # Process hits into features using the online featurizer
        get_templates_kwargs = {}
        if minimum_template_count:
            get_templates_kwargs["minimum_template_count"] = minimum_template_count
        if self.online_featurizer is None:
            raise RuntimeError("online template featurizer is not initialized")
        result, time_track = self.online_featurizer.get_templates(
            sequence_uid=sequence_uid,
            query_sequence=query_sequence,
            hits=hits,
            max_template_date=cutoff_date,
            **get_templates_kwargs,
        )
        return result.features, raw_hit_count, time_track

    def make_template_features(
        self,
        bioassembly_dict: Dict[str, Any],
        selected_token_indices: Optional[np.ndarray],
        entity_to_asym_id_int: Mapping[str, Sequence[int]],
        *,
        template_usage_decision: Optional[TemplateUsageDecision] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Main method to generate template features for a bioassembly.

        Args:
            bioassembly_dict: Dictionary containing bioassembly data.
            selected_token_indices: Indices of selected tokens (for cropping).
            entity_to_asym_id_int: Mapping from entity ID to asymmetric IDs.
            template_usage_decision: Optional caller-controlled decision for
                this occurrence. ``force_no_template`` produces the existing
                fixed-shape dummy template representation.

        Returns:
            Dictionary of processed template features.
        """
        profile_start = time.time()
        (
            resolved_template_usage_decision,
            template_usage_decision_source,
        ) = self._resolve_template_usage_decision(template_usage_decision)
        drop_template = (
            resolved_template_usage_decision == TEMPLATE_USAGE_FORCE_NO_TEMPLATE
        )
        require_at_least_one_template = (
            template_usage_decision_source == "explicit"
            and resolved_template_usage_decision
            == TEMPLATE_USAGE_USE_REAL_TEMPLATE
        )

        atom_array = bioassembly_dict["atom_array"]
        token_array = bioassembly_dict["token_array"]

        # Identify selected asymmetric IDs
        centre_atom_indices = (
            token_array[selected_token_indices]
            if selected_token_indices is not None
            else token_array
        ).get_annotation("centre_atom_index")
        selected_asym_ids = set(atom_array[centre_atom_indices].asym_id_int)

        # Build mapping from asym_id to entity_id and sequence
        asym_to_entity_id: Dict[int, str] = {}
        for eid, asyms in entity_to_asym_id_int.items():
            for aid in asyms:
                if aid in selected_asym_ids:
                    asym_to_entity_id[aid] = eid

        entity_id_to_sequence = {
            eid: bioassembly_dict["sequences"].get(eid)
            for eid in asym_to_entity_id.values()
        }

        # Patch sequences for ligands if missing
        asym_id_to_sequence = {}
        for aid, eid in asym_to_entity_id.items():
            seq = entity_id_to_sequence.get(eid)
            if seq is None:
                ligand_mask = atom_array.asym_id_int == aid
                assert atom_array.is_ligand[ligand_mask].all(), "Unknown molecule type"
                seq = "X" * ligand_mask.sum()
            asym_id_to_sequence[aid] = seq

        poly_types_mapping = {
            "polypeptide(L)": PROTEIN_CHAIN,
            "polyribonucleotide": RNA_CHAIN,
            "polydeoxyribonucleotide": DNA_CHAIN,
        }
        entity_poly_type = bioassembly_dict.get("entity_poly_type", {})

        query_release_date = None
        if self.stage == "train" and "release_date" in bioassembly_dict:
            query_release_date = datetime.strptime(
                bioassembly_dict["release_date"], "%Y-%m-%d"
            )

        template_meta_infos = {}
        seq_to_templates_cache: Dict[
            str,
            Tuple[List[Dict[str, Any]], int, Dict[str, Any]],
        ] = {}
        sequence_template_cache_mode = getattr(
            self,
            "sequence_template_cache_mode",
            "online",
        )
        configured_sequence_template_cache_mode = getattr(
            self,
            "configured_sequence_template_cache_mode",
            sequence_template_cache_mode,
        )
        profile = {
            "elapsed_seconds": 0.0,
            "raw_templates": 0,
            "selected_templates": 0,
            "valid_template_count": 0,
            "has_real_template": False,
            "drop_template_applied": drop_template,
            "template_usage_decision": resolved_template_usage_decision,
            "template_usage_decision_source": template_usage_decision_source,
            "sequence_template_cache_mode": sequence_template_cache_mode,
            "sequence_template_cache_error_policy": getattr(
                self,
                "sequence_template_cache_error_policy",
                "raise",
            ),
            "configured_sequence_template_cache_mode": (
                configured_sequence_template_cache_mode
            ),
            "sequence_template_cache_load_seconds": 0.0,
            "sequence_template_cache_files_loaded": 0,
            "sequence_template_cache_candidates": 0,
            "sequence_template_cache_selected": 0,
            "sequence_template_cache_reuses": 0,
            "sequence_template_cache_chain_dummy_fallback_unique_sequence_count": 0,
            "sequence_template_cache_chain_dummy_fallback_affected_chain_count": 0,
            "sequence_template_cache_chain_dummy_fallback_reason_counts": {},
            "sequence_template_cache_chain_dummy_fallbacks": [],
            "chain_profiles": [],
        }

        pdb_id = bioassembly_dict.get("pdb_id", "unknown")
        cache_error_policy = getattr(
            self,
            "sequence_template_cache_error_policy",
            "raise",
        )
        cache_fallback_profiles_by_sequence: Dict[str, Dict[str, Any]] = {}

        for aid, sequence in asym_id_to_sequence.items():
            chain_start = time.time()
            sequence_cache_reused = False
            eid = asym_to_entity_id[aid]
            chain_type = poly_types_mapping.get(
                entity_poly_type.get(eid, "non-polymer"), LIGAND_CHAIN_TYPES
            )

            if drop_template or chain_type != PROTEIN_CHAIN:
                templates, raw_hit_count, time_track = [], 0, {}
            else:
                if sequence in seq_to_templates_cache:
                    templates, raw_hit_count, time_track = seq_to_templates_cache[
                        sequence
                    ]
                    sequence_cache_reused = True
                    time_track = dict(time_track)
                    if time_track.get("template_source") in {
                        "sequence_cache",
                        "evaluation_sequence_cache",
                        "sequence_cache_chain_dummy",
                    }:
                        time_track["sequence_template_cache_reused"] = True
                        time_track["sequence_template_cache_load_seconds"] = 0.0
                else:
                    get_template_kwargs = {}
                    if require_at_least_one_template:
                        get_template_kwargs["minimum_template_count"] = 1
                    cache_load_rng_state = None
                    if (
                        sequence_template_cache_mode == "required"
                        and cache_error_policy == "chain_dummy"
                    ):
                        cache_load_rng_state = (
                            random.getstate(),
                            np.random.get_state(),
                        )
                    try:
                        templates, raw_hit_count, time_track = self.get_template(
                            pdb_id=pdb_id,
                            query_sequence=sequence,
                            sequence_uid=f"{pdb_id}_{aid}",
                            query_release_date=query_release_date,
                            chain_entity_type=chain_type,
                            **get_template_kwargs,
                        )
                    except SequenceTemplateCacheError as exc:
                        if (
                            sequence_template_cache_mode != "required"
                            or cache_error_policy != "chain_dummy"
                        ):
                            raise
                        if cache_load_rng_state is None:
                            raise RuntimeError(
                                "required-cache chain-dummy fallback is "
                                "missing its RNG snapshot"
                            ) from exc
                        python_rng_state, numpy_rng_state = (
                            cache_load_rng_state
                        )
                        random.setstate(python_rng_state)
                        np.random.set_state(numpy_rng_state)

                        query_sha256 = self._sequence_sha256(sequence)
                        reason = (
                            exc.reason
                            if exc.reason
                            in SEQUENCE_TEMPLATE_CACHE_ERROR_REASONS
                            else "unknown"
                        )
                        templates, raw_hit_count = [], 0
                        time_track = {
                            "template_source": (
                                "sequence_cache_chain_dummy"
                            ),
                            "sequence_template_cache_candidates": 0,
                            "sequence_template_cache_selected": 0,
                            "sequence_template_cache_error_reason": reason,
                            "sequence_template_cache_error_query_sha256": (
                                query_sha256
                            ),
                            "sequence_template_cache_error_path": (
                                exc.cache_path
                            ),
                        }
                        logger.warning(
                            "Required sequence-template cache is unusable; "
                            "using fixed dummy templates for all chains that "
                            "share this query sequence: "
                            f"pdb_id={pdb_id!r}, query_sha256={query_sha256}, "
                            f"reason={reason!r}, cache_path={exc.cache_path!r}"
                        )
                    seq_to_templates_cache[sequence] = (
                        templates,
                        raw_hit_count,
                        time_track,
                    )

            chain_id = atom_array.chain_id[atom_array.asym_id_int == aid][0]
            if chain_type == PROTEIN_CHAIN:
                num_sel = len(templates) if isinstance(templates, list) else 0
                profile["raw_templates"] += raw_hit_count
                profile["selected_templates"] += num_sel
                template_source = time_track.get("template_source")
                if template_source in {
                    "sequence_cache",
                    "evaluation_sequence_cache",
                }:
                    if sequence_cache_reused:
                        profile["sequence_template_cache_reuses"] += 1
                    else:
                        profile["sequence_template_cache_files_loaded"] += 1
                        profile["sequence_template_cache_load_seconds"] += float(
                            time_track.get(
                                "sequence_template_cache_load_seconds",
                                0.0,
                            )
                        )
                        profile["sequence_template_cache_candidates"] += int(
                            time_track.get(
                                "sequence_template_cache_candidates",
                                0,
                            )
                        )
                        profile["sequence_template_cache_selected"] += int(
                            time_track.get(
                                "sequence_template_cache_selected",
                                0,
                            )
                        )
                elif template_source == "sequence_cache_chain_dummy":
                    fallback_profile = (
                        cache_fallback_profiles_by_sequence.get(sequence)
                    )
                    if fallback_profile is None:
                        reason = time_track.get(
                            "sequence_template_cache_error_reason",
                            "unknown",
                        )
                        fallback_profile = {
                            "query_sha256": time_track.get(
                                "sequence_template_cache_error_query_sha256",
                                self._sequence_sha256(sequence),
                            ),
                            "reason": reason,
                            "cache_path": time_track.get(
                                "sequence_template_cache_error_path"
                            ),
                            "affected_chain_count": 0,
                        }
                        cache_fallback_profiles_by_sequence[sequence] = (
                            fallback_profile
                        )
                        profile[
                            "sequence_template_cache_chain_dummy_fallback_unique_sequence_count"
                        ] += 1
                        reason_counts = profile[
                            "sequence_template_cache_chain_dummy_fallback_reason_counts"
                        ]
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    fallback_profile["affected_chain_count"] += 1
                    profile[
                        "sequence_template_cache_chain_dummy_fallback_affected_chain_count"
                    ] += 1
                profile["chain_profiles"].append(
                    {
                        "asym_id": aid,
                        "chain_id": chain_id,
                        "elapsed_seconds": time.time() - chain_start,
                        "raw_templates": raw_hit_count,
                        "selected_templates": num_sel,
                        "chain_entity_type": chain_type,
                        "time_track": time_track,
                    }
                )

            template_meta_infos[aid] = {
                "entity_id": eid,
                "chain_id": chain_id,
                "sequence": sequence,
                "chain_entity_type": chain_type,
                "templates": templates,
            }

        if not get_safe_entity_id_for_template_copy(template_meta_infos):
            logger.warning(
                f"PDB {pdb_id} has inconsistent sequences within same entity."
            )

        # Coordinate mapping to standardized indices
        ca = atom_array[centre_atom_indices]
        std_idxs = map_to_standard(ca.asym_id_int, ca.res_id, template_meta_infos)

        # Assemble final features
        max_t = 4 if self.stage != "train" else 4  # Default to 4
        assembly_line = TemplateFeatureAssemblyLine(max_templates=max_t)
        template_features = assembly_line.assemble(
            template_meta_infos, std_idxs
        ).as_protenix_dict()

        valid_template_slots = np.any(
            template_features["template_atom_mask"], axis=(1, 2)
        )
        profile["elapsed_seconds"] = time.time() - profile_start
        profile["valid_template_count"] = int(valid_template_slots.sum())
        profile["has_real_template"] = bool(valid_template_slots.any())
        profile["sequence_template_cache_chain_dummy_fallbacks"] = list(
            cache_fallback_profiles_by_sequence.values()
        )
        self._last_profile = profile
        return template_features

    def __call__(
        self,
        bioassembly_dict: Dict[str, Any],
        selected_indices: np.ndarray,
        entity_to_asym_id_int: Mapping[str, Sequence[int]],
        *,
        template_usage_decision: Optional[TemplateUsageDecision] = None,
    ) -> Dict[str, np.ndarray]:
        """Call method proxying to make_template_features."""
        return self.make_template_features(
            bioassembly_dict,
            selected_indices,
            entity_to_asym_id_int,
            template_usage_decision=template_usage_decision,
        )

    def get_last_profile(self) -> Dict[str, Any]:
        """Returns profiling information for the last call."""
        return self._last_profile


@dataclasses.dataclass(frozen=True)
class Templates:
    """Dataclass containing template features."""

    # aatype: [num_templates, num_res]
    aatype: np.ndarray
    # atom_positions: [num_templates, num_res, 24, 3]
    atom_positions: np.ndarray
    # atom_mask: [num_templates, num_res, 24]
    atom_mask: np.ndarray

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        """Construct instance from a data dictionary."""
        return cls(
            aatype=batch["template_aatype"],
            atom_positions=batch["template_atom_positions"],
            atom_mask=batch["template_atom_mask"],
        )

    def as_data_dict(self) -> BatchDict:
        """Convert to a standard data dictionary."""
        return {
            "template_aatype": self.aatype,
            "template_atom_positions": self.atom_positions,
            "template_atom_mask": self.atom_mask,
        }

    # Shared config instance to avoid repeated object creation
    _DGRAM_CONFIG = DistogramFeaturesConfig(
        min_bin=3.25, max_bin=50.75, num_bins=39
    )

    def as_protenix_dict(self) -> BatchDict:
        """Compute additional features and return as Protenix dictionary."""
        features = self.as_data_dict()
        num_templates = self.aatype.shape[0]
        num_res = self.aatype.shape[1]

        # Pre-allocate output arrays instead of list append + stack
        all_pb_masks = np.empty(
            (num_templates, num_res, num_res), dtype=np.float32
        )
        all_dgrams = np.empty(
            (num_templates, num_res, num_res, 39), dtype=np.float32
        )
        all_unit_vectors = np.empty(
            (num_templates, num_res, num_res, 3), dtype=np.float32
        )
        all_bb_masks = np.empty(
            (num_templates, num_res, num_res), dtype=np.float32
        )

        config = Templates._DGRAM_CONFIG
        is_lig = getattr(self, "is_ligand", None)
        for i in range(num_templates):
            aatype = self.aatype[i]
            mask = self.atom_mask[i]
            pos = self.atom_positions[i] * mask[..., None]

            pb_pos, pb_mask = TemplateFeatures.pseudo_beta_fn(
                aatype, pos, mask, is_ligand=is_lig
            )
            pb_mask_2d = pb_mask[:, None] * pb_mask[None, :]

            dgram = TemplateFeatures.dgram_from_positions(
                pb_pos, config=config
            )
            all_dgrams[i] = dgram * pb_mask_2d[..., None]
            all_pb_masks[i] = pb_mask_2d

            uv, bb_mask_2d = TemplateFeatures.compute_template_unit_vector(
                aatype, pos, mask
            )
            all_unit_vectors[i] = uv * bb_mask_2d[..., None]
            all_bb_masks[i] = bb_mask_2d

        features.update(
            {
                "template_pseudo_beta_mask": all_pb_masks,
                "template_distogram": all_dgrams,
                "template_unit_vector": all_unit_vectors,
                "template_backbone_frame_mask": all_bb_masks,
            }
        )
        return features


def make_fixed_dummy_template_features(
    num_res: int,
    max_templates: int = 4,
) -> BatchDict:
    """Create the same fixed-shape no-template features as template dropout.

    The current TemplateEmbedder is selected by feature presence rather than
    by a separate mask.  Training datasets without a template database must
    therefore still emit this fully masked representation when DDP uses a
    static graph.
    """
    if isinstance(num_res, bool) or not isinstance(num_res, (int, np.integer)):
        raise TypeError(f"num_res must be an integer, got {type(num_res).__name__}")
    if num_res < 1:
        raise ValueError(f"num_res must be positive, got {num_res}")
    if (
        isinstance(max_templates, bool)
        or not isinstance(max_templates, (int, np.integer))
    ):
        raise TypeError(
            "max_templates must be an integer, "
            f"got {type(max_templates).__name__}"
        )
    if max_templates < 1:
        raise ValueError(
            f"max_templates must be positive, got {max_templates}"
        )

    empty = TemplateFeatures.empty_template_features(num_res=int(num_res))
    templates = Templates(
        aatype=pad_to(
            empty["template_aatype"],
            (int(max_templates), None),
        ),
        atom_positions=pad_to(
            empty["template_atom_positions"],
            (int(max_templates), None, None, None),
        ),
        atom_mask=pad_to(
            empty["template_atom_mask"],
            (int(max_templates), None, None),
        ).astype(bool),
    )
    return templates.as_protenix_dict()


class DummyTemplateFeaturizer:
    """Emit fixed-shape dummy template features without loading template DBs."""

    def __init__(self, max_templates: int = 4) -> None:
        if (
            isinstance(max_templates, bool)
            or not isinstance(max_templates, (int, np.integer))
        ):
            raise TypeError(
                "max_templates must be an integer, "
                f"got {type(max_templates).__name__}"
            )
        if max_templates < 1:
            raise ValueError(
                f"max_templates must be positive, got {max_templates}"
            )
        self.max_templates = int(max_templates)
        self._last_profile: Dict[str, Any] = {}

    def __call__(
        self,
        bioassembly_dict: Dict[str, Any],
        selected_indices: Optional[np.ndarray],
        entity_to_asym_id_int: Mapping[str, Sequence[int]],
        *,
        template_usage_decision: Optional[TemplateUsageDecision] = None,
    ) -> BatchDict:
        del entity_to_asym_id_int, template_usage_decision
        profile_start = time.perf_counter()
        token_array = bioassembly_dict["token_array"]
        selected_token_array = (
            token_array
            if selected_indices is None
            else token_array[selected_indices]
        )
        num_res = len(selected_token_array)
        features = make_fixed_dummy_template_features(
            num_res=num_res,
            max_templates=self.max_templates,
        )
        self._last_profile = {
            "elapsed_seconds": time.perf_counter() - profile_start,
            "raw_templates": 0,
            "selected_templates": 0,
            "valid_template_count": 0,
            "has_real_template": False,
            "drop_template_applied": True,
            "template_usage_decision": TEMPLATE_USAGE_FORCE_NO_TEMPLATE,
            "template_usage_decision_source": "template_loading_disabled",
            "sequence_template_cache_chain_dummy_fallback_unique_sequence_count": 0,
            "sequence_template_cache_chain_dummy_fallback_affected_chain_count": 0,
            "sequence_template_cache_chain_dummy_fallback_reason_counts": {},
            "sequence_template_cache_chain_dummy_fallbacks": [],
            "chain_profiles": [],
        }
        return features

    def get_last_profile(self) -> Dict[str, Any]:
        """Return profiling metadata compatible with TemplateFeaturizer."""
        return self._last_profile


def _reduce_template_features(
    template_features: FeatureDict, max_templates: int
) -> FeatureDict:
    """Reduces templates to the requested maximum number."""
    num_t = template_features["template_aatype"].shape[0]
    keep_mask = np.arange(num_t) < max_templates
    fields = TEMPLATE_FEATURES + ("template_release_timestamp",)
    return {k: v[keep_mask] for k, v in template_features.items() if k in fields}


def get_safe_entity_id_for_template_copy(
    bioassembly: Mapping[int, Mapping[str, Any]],
) -> List[str]:
    """Identifies entity IDs that have consistent sequences across all chains."""
    eid_to_seqs = {}
    for aid, info in bioassembly.items():
        eid = info["entity_id"]
        seq = info["sequence"]
        eid_to_seqs.setdefault(eid, set()).add(seq)
    return [eid for eid, seqs in eid_to_seqs.items() if len(seqs) == 1]


class InferenceTemplateFeaturizer:
    """Simplified featurizer for inference, leveraging the same assembly logic."""

    @staticmethod
    def make_template_feature(
        bioassembly: Sequence[Mapping[str, Any]],
        atom_array: AtomArray,
        use_template: bool = True,
        online_template_featurizer: Optional[TemplateHitFeaturizer] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generates template features during inference.

        Args:
            bioassembly: List of entity information from the input JSON.
            atom_array: Parsed atom structure.
            use_template: Whether to use templates.
            online_template_featurizer: Featurizer for processing template hits.

        Returns:
            Dictionary of template features.
        """
        logger.info("Calling InferenceTemplateFeaturizer.make_template_feature")
        template_meta_infos = {}
        curr_asym_id = 0

        for eid, info in enumerate(bioassembly):
            seq, count, ctype, t_path = "", 0, LIGAND_CHAIN_TYPES, ""

            if "proteinChain" in info:
                c = info["proteinChain"]
                seq, count, ctype, t_path = (
                    c["sequence"],
                    c["count"],
                    PROTEIN_CHAIN,
                    c.get("templatesPath", ""),
                )
            elif "rnaSequence" in info:
                c = info["rnaSequence"]
                seq, count, ctype = c["sequence"], c["count"], RNA_CHAIN
            elif "dnaSequence" in info:
                c = info["dnaSequence"]
                seq, count, ctype = c["sequence"], c["count"], DNA_CHAIN
            elif "ligand" in info:
                count, ctype = info["ligand"]["count"], LIGAND_CHAIN_TYPES
                seq = "X" * (atom_array.asym_id_int == curr_asym_id).sum()

            templates = []
            if t_path and use_template and online_template_featurizer:
                assert ctype == PROTEIN_CHAIN, "Only protein templates are supported."
                with open(t_path, "r") as f:
                    content = f.read()

                if t_path.endswith(".hhr"):
                    hits = HHRParser.parse(hhr_string=content)
                elif t_path.endswith(".a3m"):
                    hits = HmmsearchA3MParser.parse(
                        query_seq=seq, a3m_str=content, skip_first=False
                    )
                else:
                    raise ValueError(f"Unsupported template format: {t_path}")

                result, _ = online_template_featurizer.get_templates(
                    sequence_uid=seq,
                    query_sequence=seq,
                    hits=hits,
                    max_template_date=None,
                )
                templates = result.features
                logger.info(f"Found {len(templates)} templates for sequence {seq}")

            for i in range(count):
                aid = curr_asym_id + i
                template_meta_infos[aid] = {
                    "entity_id": eid,
                    "chain_id": atom_array.chain_id[atom_array.asym_id_int == aid][0],
                    "sequence": seq,
                    "chain_entity_type": ctype,
                    "templates": templates,
                }
            curr_asym_id += count

        # Coordinate mapping
        ca = atom_array[atom_array.centre_atom_mask.astype(bool)]
        std_idxs = map_to_standard(ca.asym_id_int, ca.res_id, template_meta_infos)

        # Assemble features
        return (
            TemplateFeatureAssemblyLine(max_templates=4)
            .assemble(template_meta_infos, std_idxs)
            .as_protenix_dict()
        )
