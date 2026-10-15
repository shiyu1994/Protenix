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
from os.path import exists as opexists, getsize as opgetsize, join as opjoin
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from biotite.structure import AtomArray

from protenix.data.constants import (
    DNA_CHAIN,
    LIGAND_CHAIN_TYPES,
    PROTEIN_CHAIN,
    RNA_CHAIN,
    STANDARD_POLYMER_CHAIN_TYPES,
    STD_RESIDUES_WITH_GAP,
)
from protenix.data.msa.msa_utils import (
    map_to_standard,
    MSA_GAP_IDX,
    MSAPairingEngine,
    NUM_SEQ_NUM_RES_MSA_FEATURES,
    RawMsa,
)
from protenix.utils.file_io import load_json_cached
from protenix.utils.logger import get_logger

logger = get_logger(__name__)

VIRTUAL_SEGMENT_MSA_SHUFFLE_POLICY_VERSION = (
    "all_protein_msa_crop_segments_nearest_eligible_v1"
)
PAIRING_NON_PAIRING_BUDGET_POLICY_VERSION = (
    "pairing_copy_and_genuine_non_pairing_independent_top_n_v1"
)
PARTIAL_QUERY_ONLY_MSA_POLICY_VERSION = (
    "crop_selected_physical_protein_asym_uniform_1_to_max_retain_one_v1"
)
MULTICHAIN_MSA_SHUFFLE_VARIANT_POLICY_VERSION = (
    "crop_visible_physical_protein_asym_n1_segment_"
    "nge2_bernoulli_segment_or_whole_all_rows_v1"
)
MSA_SHUFFLE_VARIANT_NONE = "none"
MSA_SHUFFLE_VARIANT_SEGMENT = "segment"
MSA_SHUFFLE_VARIANT_WHOLE_PHYSICAL_CHAIN = "whole_physical_chain"
MSA_SHUFFLE_VARIANT_CODES = {
    MSA_SHUFFLE_VARIANT_NONE: 0,
    MSA_SHUFFLE_VARIANT_SEGMENT: 1,
    MSA_SHUFFLE_VARIANT_WHOLE_PHYSICAL_CHAIN: 2,
}


def _has_paired_copy_and_genuine_non_pairing_sources(
    paired_dbs: Sequence[str],
    non_pairing_dbs: Sequence[str],
) -> bool:
    """Return whether one source entry requests both MSA source classes."""
    for paired_db, non_pairing_db in zip(paired_dbs, non_pairing_dbs):
        paired_db = str(paired_db)
        sources = [
            source
            for source in str(non_pairing_db).split("-")
            if source
        ]
        if (
            paired_db
            and paired_db in sources
            and any(source != paired_db for source in sources)
        ):
            return True
    return False


# Transient bioassembly context populated by row-addressed distillation
# datasets.  The path is deliberately not part of the serialized bioassembly:
# it is derived from ``source_root_id + cif_relpath`` for each sampled row.
DIRECT_PROTEIN_UNPAIRED_MSA_PATH = "direct_protein_unpaired_msa_path"

_OPENFOLD3_PROTEIN_UNPAIRED_SOURCES = (
    "uniref90_hits",
    "cfdb_hits",
    "mgnify_hits",
)
_OPENFOLD3_PROTEIN_PAIRED_SOURCES = ("uniprot_hits",)
_OPENFOLD3_RNA_UNPAIRED_SOURCES = (
    "nucleotide_collection_hits",
    "rfam_hits",
    "rnacentral_hits",
)
_OPENFOLD3_QUERY_MISMATCH_MESSAGE = (
    "alignment query row does not match the canonical sequence"
)
_OPENFOLD3_RUNTIME_RECOVERY_DATASET = "openfold3_disordered_v20260725"


@dataclasses.dataclass(frozen=True)
class _OpenFold3RecoveryCandidate:
    """One declared representative loaded for mismatch-only recovery."""

    representative_id: str
    canonical_query: Optional[str]
    unpaired_msas: tuple[RawMsa, ...]
    paired_msas: tuple[RawMsa, ...]
    error: Optional[str] = None


def _strict_positive_int(value: Any, name: str) -> int:
    """Return a positive integer without accepting booleans or floats."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be a positive integer, got {value!r}")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def select_query_only_protein_asym_ids(
    *,
    protein_asym_ids: Sequence[int],
    max_query_only_chains: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Select crop-visible physical protein chains for partial query-only MSA.

    A multi-chain crop always retains at least one protein chain with its
    normal MSA. The selected count is uniform on
    ``[1, min(max_query_only_chains, N - 1)]`` and selection is without
    replacement. A single protein chain is returned as the sole selection;
    its caller applies the historical one-row global query-only transform.
    """
    max_query_only_chains = _strict_positive_int(
        max_query_only_chains,
        "max_query_only_chains",
    )
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    raw_asym_ids = np.asarray(protein_asym_ids)
    if raw_asym_ids.ndim != 1:
        raise ValueError("protein_asym_ids must be a rank-1 sequence")
    if raw_asym_ids.size == 0:
        return ()
    if not np.issubdtype(raw_asym_ids.dtype, np.integer):
        raise TypeError("protein_asym_ids must contain integers")
    asym_ids = np.unique(raw_asym_ids.astype(np.int64, copy=False))
    if asym_ids.size == 1:
        return (int(asym_ids[0]),)

    max_selected = min(max_query_only_chains, int(asym_ids.size) - 1)
    selected_count = int(rng.integers(1, max_selected + 1))
    selected = rng.choice(
        asym_ids,
        size=selected_count,
        replace=False,
    )
    return tuple(sorted(int(asym_id) for asym_id in selected))


def select_multichain_msa_shuffle_variant(
    *,
    protein_asym_ids: Sequence[int],
    sampling_enabled: bool,
    segment_probability: float,
    variant_seed: Optional[int],
) -> tuple[str, bool]:
    """Choose the all-row shuffle grouping for a crop.

    A single crop-visible physical protein chain always requests segment
    grouping. With at least two physical protein chains, the opt-in policy
    draws between segment and whole-physical-chain grouping from an independent
    occurrence seed. Disabling the policy preserves the historical static
    all-segment behavior.

    Returns:
        A ``(requested_variant, variant_eligible)`` pair. Eligibility denotes
        participation in the multi-chain Bernoulli gate, not whether the
        requested shuffle later has enough homolog rows to be applied.
    """
    if not isinstance(sampling_enabled, (bool, np.bool_)):
        raise TypeError("sampling_enabled must be a boolean")
    if isinstance(segment_probability, (bool, np.bool_)):
        raise TypeError("segment_probability must be a real number")
    try:
        segment_probability = float(segment_probability)
    except (TypeError, ValueError) as error:
        raise TypeError("segment_probability must be a real number") from error
    if not np.isfinite(segment_probability) or not (
        0.0 <= segment_probability <= 1.0
    ):
        raise ValueError(
            "segment_probability must be finite and in [0, 1], got "
            f"{segment_probability!r}"
        )

    raw_asym_ids = np.asarray(protein_asym_ids)
    if raw_asym_ids.ndim != 1:
        raise ValueError("protein_asym_ids must be a rank-1 sequence")
    if raw_asym_ids.size and not np.issubdtype(
        raw_asym_ids.dtype,
        np.integer,
    ):
        raise TypeError("protein_asym_ids must contain integers")
    protein_chain_count = int(
        np.unique(raw_asym_ids.astype(np.int64, copy=False)).size
    )
    if protein_chain_count == 0:
        return MSA_SHUFFLE_VARIANT_NONE, False
    if protein_chain_count == 1 or not bool(sampling_enabled):
        return MSA_SHUFFLE_VARIANT_SEGMENT, False

    if segment_probability <= 0.0:
        return MSA_SHUFFLE_VARIANT_WHOLE_PHYSICAL_CHAIN, True
    if segment_probability >= 1.0:
        return MSA_SHUFFLE_VARIANT_SEGMENT, True
    if isinstance(variant_seed, (bool, np.bool_)) or not isinstance(
        variant_seed,
        (int, np.integer),
    ):
        raise TypeError(
            "variant_seed must be an unsigned 64-bit integer for a "
            "non-trivial multi-chain variant decision"
        )
    variant_seed = int(variant_seed)
    if not 0 <= variant_seed < (1 << 64):
        raise ValueError(
            "variant_seed must be in [0, 2**64), got "
            f"{variant_seed}"
        )
    threshold = int(segment_probability * (1 << 64))
    if variant_seed < threshold:
        return MSA_SHUFFLE_VARIANT_SEGMENT, True
    return MSA_SHUFFLE_VARIANT_WHOLE_PHYSICAL_CHAIN, True


def build_virtual_msa_segment_groups(
    *,
    asym_ids: np.ndarray,
    res_ids: np.ndarray,
    protein_sequence_lengths: Mapping[int, int],
    residue_gap: int = 24,
    min_segment_length: int = 9,
) -> Dict[int, tuple[np.ndarray, ...]]:
    """Build crop-aware canonical MSA column groups for protein chains.

    A raw segment starts when adjacent selected canonical residue IDs differ by
    at least ``residue_gap``. Raw segments shorter than
    ``min_segment_length`` residues share the row permutation of the nearest
    eligible segment in the same physical chain; ties go to the left. If a
    chain has no eligible segment, all of its selected residues form one group.

    The input may contain repeated ``(asym_id, res_id)`` pairs from atom-level
    tokenization of a modified residue. Stable residue de-duplication ensures
    that such a residue counts once and maps to one canonical MSA column.
    Returned columns are zero-based within each physical chain.
    """
    residue_gap = _strict_positive_int(residue_gap, "residue_gap")
    min_segment_length = _strict_positive_int(
        min_segment_length,
        "min_segment_length",
    )
    asym_ids = np.asarray(asym_ids)
    res_ids = np.asarray(res_ids)
    if asym_ids.ndim != 1 or res_ids.ndim != 1:
        raise ValueError("asym_ids and res_ids must be rank-1 arrays")
    if asym_ids.shape != res_ids.shape:
        raise ValueError(
            "asym_ids and res_ids must have identical shapes: "
            f"{asym_ids.shape} != {res_ids.shape}"
        )

    groups_by_asym: Dict[int, tuple[np.ndarray, ...]] = {}
    for raw_asym_id, raw_sequence_length in protein_sequence_lengths.items():
        asym_id = int(raw_asym_id)
        sequence_length = _strict_positive_int(
            raw_sequence_length,
            f"protein_sequence_lengths[{asym_id}]",
        )
        chain_res_ids = np.unique(
            np.asarray(res_ids[asym_ids == asym_id], dtype=np.int64)
        )
        if chain_res_ids.size == 0:
            continue
        if np.any(chain_res_ids < 1) or np.any(
            chain_res_ids > sequence_length
        ):
            raise ValueError(
                f"protein asym_id={asym_id} has crop residue IDs outside "
                f"[1, {sequence_length}]: {chain_res_ids.tolist()}"
            )

        split_points = np.flatnonzero(
            np.diff(chain_res_ids) >= residue_gap
        ) + 1
        raw_segments = [
            segment
            for segment in np.split(chain_res_ids, split_points)
            if segment.size
        ]
        eligible_indices = [
            index
            for index, segment in enumerate(raw_segments)
            if segment.size >= min_segment_length
        ]
        if not eligible_indices:
            groups_by_asym[asym_id] = (
                np.concatenate(raw_segments).astype(np.int64) - 1,
            )
            continue

        assigned = {
            index: [raw_segments[index]]
            for index in eligible_indices
        }
        for index, segment in enumerate(raw_segments):
            if index in assigned:
                continue

            def sequence_distance(eligible_index: int) -> tuple[int, int]:
                eligible = raw_segments[eligible_index]
                if segment[-1] < eligible[0]:
                    distance = int(eligible[0] - segment[-1])
                elif eligible[-1] < segment[0]:
                    distance = int(segment[0] - eligible[-1])
                else:
                    distance = 0
                # The raw segment index makes equal-distance ties go left.
                return distance, eligible_index

            nearest_eligible = min(
                eligible_indices,
                key=sequence_distance,
            )
            assigned[nearest_eligible].append(segment)

        groups = []
        for eligible_index in eligible_indices:
            residues = np.unique(
                np.concatenate(assigned[eligible_index])
            )
            groups.append(residues.astype(np.int64) - 1)
        groups.sort(key=lambda columns: int(columns[0]))
        groups_by_asym[asym_id] = tuple(groups)
    return groups_by_asym


def build_whole_physical_chain_msa_groups(
    *,
    asym_ids: np.ndarray,
    res_ids: np.ndarray,
    protein_sequence_lengths: Mapping[int, int],
) -> Dict[int, tuple[np.ndarray, ...]]:
    """Build one canonical MSA-column group per crop-visible protein chain.

    Physical ``asym_id`` values remain separate even when homomer copies share
    a sequence. Repeated residue IDs from atom-level modified-residue tokens
    are de-duplicated before converting one-based residue IDs to zero-based MSA
    columns.
    """
    asym_ids = np.asarray(asym_ids)
    res_ids = np.asarray(res_ids)
    if asym_ids.ndim != 1 or res_ids.ndim != 1:
        raise ValueError("asym_ids and res_ids must be rank-1 arrays")
    if asym_ids.shape != res_ids.shape:
        raise ValueError(
            "asym_ids and res_ids must have identical shapes: "
            f"{asym_ids.shape} != {res_ids.shape}"
        )

    groups_by_asym: Dict[int, tuple[np.ndarray, ...]] = {}
    for raw_asym_id, raw_sequence_length in protein_sequence_lengths.items():
        asym_id = int(raw_asym_id)
        sequence_length = _strict_positive_int(
            raw_sequence_length,
            f"protein_sequence_lengths[{asym_id}]",
        )
        chain_res_ids = np.unique(
            np.asarray(res_ids[asym_ids == asym_id], dtype=np.int64)
        )
        if chain_res_ids.size == 0:
            continue
        if np.any(chain_res_ids < 1) or np.any(
            chain_res_ids > sequence_length
        ):
            raise ValueError(
                f"protein asym_id={asym_id} has crop residue IDs outside "
                f"[1, {sequence_length}]: {chain_res_ids.tolist()}"
            )
        groups_by_asym[asym_id] = (
            chain_res_ids.astype(np.int64, copy=False) - 1,
        )
    return groups_by_asym


def shuffle_msa_feature_rows_by_segment(
    *,
    msa: np.ndarray,
    deletion_matrix: np.ndarray,
    segment_columns: Sequence[np.ndarray],
    rng: np.random.Generator,
    copy: bool = True,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Independently shuffle non-query rows for canonical column groups.

    ``copy=False`` mutates the two owned input arrays and is intended for the
    assembly hot path, where copying only selected segment columns avoids a
    depth-by-full-sequence clone.
    """
    msa = np.asarray(msa)
    deletion_matrix = np.asarray(deletion_matrix)
    if msa.ndim != 2 or deletion_matrix.ndim != 2:
        raise ValueError("msa and deletion_matrix must be rank-2 arrays")
    if msa.shape != deletion_matrix.shape:
        raise ValueError(
            "msa and deletion_matrix must have identical shapes: "
            f"{msa.shape} != {deletion_matrix.shape}"
        )
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not isinstance(copy, (bool, np.bool_)):
        raise TypeError("copy must be a boolean")

    validated_columns: list[np.ndarray] = []
    occupied_columns: set[int] = set()
    for raw_columns in segment_columns:
        columns = np.asarray(raw_columns, dtype=np.int64)
        if columns.ndim != 1 or columns.size == 0:
            raise ValueError("each segment must contain canonical columns")
        if np.any(columns < 0) or np.any(columns >= msa.shape[1]):
            raise ValueError(
                "segment columns are outside the canonical MSA width "
                f"{msa.shape[1]}: {columns.tolist()}"
            )
        if np.unique(columns).size != columns.size:
            raise ValueError("segment columns must be unique")
        overlap = occupied_columns.intersection(columns.tolist())
        if overlap:
            raise ValueError(
                "virtual MSA segments must be disjoint; duplicate columns: "
                f"{sorted(overlap)}"
            )
        occupied_columns.update(columns.tolist())
        validated_columns.append(columns)

    shuffled_msa = msa.copy() if copy else msa
    shuffled_deletion = (
        deletion_matrix.copy() if copy else deletion_matrix
    )
    attempted = False
    for columns in validated_columns:
        if msa.shape[0] <= 2:
            continue
        order = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                rng.permutation(np.arange(1, msa.shape[0], dtype=np.int64)),
            )
        )
        # Only copy crop-selected segment columns. In the assembly hot path
        # this avoids cloning depth x full_sequence_length arrays merely to
        # augment a crop of at most a few hundred residues.
        source_segment_msa = shuffled_msa[:, columns].copy()
        source_segment_deletion = shuffled_deletion[:, columns].copy()
        shuffled_msa[:, columns] = source_segment_msa[order]
        shuffled_deletion[:, columns] = source_segment_deletion[order]
        attempted = True
    return shuffled_msa, shuffled_deletion, attempted


def make_query_only_msa_features(
    msa_features: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Return an MSA feature dictionary containing only the global query row.

    The caller owns the occurrence-level mode decision.  This pure transform
    keeps row zero across the complete cropped target, synchronizes every
    row-dependent feature, and recomputes aggregate features so homolog
    statistics cannot leak into query-only samples.
    """
    if "msa" not in msa_features:
        return dict(msa_features)

    msa = np.asarray(msa_features["msa"])
    if msa.ndim != 2 or msa.shape[0] < 1:
        raise ValueError(
            "Query-only MSA requires a non-empty rank-2 'msa' array, "
            f"got shape {msa.shape}."
        )

    query_msa = msa[:1].copy()
    query_only = dict(msa_features)

    for name in (
        "msa",
        "msa_mask",
        "deletion_matrix",
        "has_deletion",
        "deletion_value",
    ):
        if name not in msa_features:
            continue
        value = np.asarray(msa_features[name])
        if value.ndim < 2 or value.shape[0] != msa.shape[0]:
            raise ValueError(
                f"MSA feature '{name}' must share msa depth {msa.shape[0]}, "
                f"got shape {value.shape}."
            )
        query_only[name] = value[:1].copy()

    profile = msa_features.get("profile")
    num_profile_classes = (
        int(np.asarray(profile).shape[-1])
        if profile is not None
        else len(STD_RESIDUES_WITH_GAP)
    )
    profile_dtype = (
        np.asarray(profile).dtype if profile is not None else np.dtype(np.float32)
    )
    query_only["profile"] = (
        query_msa[0, :, None] == np.arange(num_profile_classes)[None, :]
    ).astype(profile_dtype, copy=False)

    deletion_matrix = msa_features.get("deletion_matrix")
    if deletion_matrix is not None:
        deletion_mean_dtype = (
            np.asarray(msa_features["deletion_mean"]).dtype
            if "deletion_mean" in msa_features
            else np.dtype(np.float32)
        )
        query_only["deletion_mean"] = np.asarray(deletion_matrix)[0].astype(
            deletion_mean_dtype, copy=True
        )
    elif "deletion_mean" in msa_features:
        deletion_mean = np.asarray(msa_features["deletion_mean"])
        if "deletion_value" in query_only:
            # Invert deletion_value = 2/pi * atan(deletion_matrix / 3).
            value = np.asarray(query_only["deletion_value"])[0].astype(
                np.float64, copy=False
            )
            recovered = 3.0 * np.tan(
                np.clip(value, 0.0, 1.0 - 1e-7) * np.pi / 2
            )
            query_only["deletion_mean"] = np.rint(recovered).astype(
                deletion_mean.dtype, copy=False
            )
        else:
            query_only["deletion_mean"] = np.zeros_like(deletion_mean)

    alignment_count_names = (
        "prot_unpaired_num_alignments",
        "prot_paired_num_alignments",
        "rna_unpaired_num_alignments",
        "prot_pair_num_alignments",
        "prot_unpair_num_alignments",
        "rna_pair_num_alignments",
        "rna_unpair_num_alignments",
    )
    for name in alignment_count_names:
        if name in msa_features:
            value = np.asarray(msa_features[name])
            query_only[name] = np.asarray(1, dtype=value.dtype)

    return query_only


def apply_partial_query_only_msa_features(
    msa_features: Mapping[str, np.ndarray],
    *,
    token_asym_ids: np.ndarray,
    query_only_asym_ids: Sequence[int],
) -> Dict[str, np.ndarray]:
    """Remove non-query homolog information from selected protein chains.

    The global query row and all unselected chain columns remain unchanged.
    Selected-chain columns become gaps below row zero, while deletion-derived
    and profile features are recomputed from the authoritative query row.
    """
    result = dict(msa_features)
    if "msa" not in result:
        return result
    msa = np.asarray(result["msa"])
    token_asym_ids = np.asarray(token_asym_ids)
    if msa.ndim != 2 or msa.shape[0] < 1:
        raise ValueError(
            "Partial query-only MSA requires a non-empty rank-2 'msa' array, "
            f"got shape {msa.shape}."
        )
    if token_asym_ids.ndim != 1 or token_asym_ids.shape[0] != msa.shape[1]:
        raise ValueError(
            "token_asym_ids must match the MSA residue width: "
            f"{token_asym_ids.shape} versus {msa.shape[1]}"
        )
    selected_asym_ids = tuple(
        sorted({int(asym_id) for asym_id in query_only_asym_ids})
    )
    selected_columns = np.isin(token_asym_ids, selected_asym_ids)
    if not selected_asym_ids or not np.any(selected_columns):
        return result

    result["msa"] = msa.copy()
    result["msa"][1:, selected_columns] = MSA_GAP_IDX

    if "deletion_matrix" in result:
        deletion_matrix = np.asarray(result["deletion_matrix"])
        if deletion_matrix.shape != msa.shape:
            raise ValueError(
                "deletion_matrix must match msa shape for partial "
                f"query-only: {deletion_matrix.shape} != {msa.shape}"
            )
        result["deletion_matrix"] = deletion_matrix.copy()
        result["deletion_matrix"][1:, selected_columns] = 0

    for name in ("has_deletion", "deletion_value"):
        if name not in result:
            continue
        value = np.asarray(result[name])
        if value.shape != msa.shape:
            raise ValueError(
                f"{name} must match msa shape for partial query-only: "
                f"{value.shape} != {msa.shape}"
            )
        result[name] = value.copy()
        result[name][1:, selected_columns] = 0

    if "profile" in result:
        profile = np.asarray(result["profile"])
        if profile.ndim != 2 or profile.shape[0] != msa.shape[1]:
            raise ValueError(
                "profile must share the MSA residue width for partial "
                f"query-only: {profile.shape}"
            )
        result["profile"] = profile.copy()
        result["profile"][selected_columns] = (
            result["msa"][0, selected_columns, None]
            == np.arange(profile.shape[-1])[None, :]
        ).astype(profile.dtype, copy=False)

    if "deletion_mean" in result:
        deletion_mean = np.asarray(result["deletion_mean"])
        if deletion_mean.ndim != 1 or deletion_mean.shape[0] != msa.shape[1]:
            raise ValueError(
                "deletion_mean must share the MSA residue width for partial "
                f"query-only: {deletion_mean.shape}"
            )
        result["deletion_mean"] = deletion_mean.copy()
        if "deletion_matrix" in result:
            result["deletion_mean"][selected_columns] = result[
                "deletion_matrix"
            ][0, selected_columns].astype(
                deletion_mean.dtype,
                copy=False,
            )
        else:
            result["deletion_mean"][selected_columns] = 0
    return result


def _shuffle_a3m_hits_preserving_query(query: str, ctype: str, a3m: str) -> str:
    """Shuffle A3M hits while keeping the first row fixed."""
    raw = RawMsa.from_a3m(query, ctype, a3m, dedup=False)
    if len(raw.seqs) <= 2:
        return raw.to_a3m()
    hit_indices = np.arange(1, len(raw.seqs))
    np.random.shuffle(hit_indices)
    order = [0] + hit_indices.tolist()
    return RawMsa(
        query,
        ctype,
        [raw.seqs[i] for i in order],
        [raw.descs[i] for i in order],
        deduplicate=False,
    ).to_a3m()


def compute_msa_profile(
    msa: np.ndarray,
    num_classes: int = len(STD_RESIDUES_WITH_GAP),
    max_chunk_cells: int = 4_000_000,
) -> np.ndarray:
    """Compute an exact MSA profile without an ``N x L x C`` temporary."""
    msa = np.asarray(msa)
    if msa.ndim != 2 or msa.shape[0] < 1:
        raise ValueError(
            f"msa must be a non-empty rank-2 array, got {msa.shape}"
        )
    num_classes = _strict_positive_int(num_classes, "num_classes")
    max_chunk_cells = _strict_positive_int(
        max_chunk_cells,
        "max_chunk_cells",
    )
    if not np.issubdtype(msa.dtype, np.integer):
        raise TypeError(f"msa must contain integers, got {msa.dtype}")
    if np.any(msa < 0) or np.any(msa >= num_classes):
        raise ValueError(
            f"msa values must be in [0, {num_classes}), got "
            f"[{int(msa.min())}, {int(msa.max())}]"
        )

    depth, width = msa.shape
    counts = np.zeros((width, num_classes), dtype=np.int64)
    column_offsets = (
        np.arange(width, dtype=np.int64) * num_classes
    )
    chunk_rows = max(1, max_chunk_cells // max(1, width))
    for start in range(0, depth, chunk_rows):
        encoded = (
            msa[start : start + chunk_rows].astype(
                np.int64,
                copy=False,
            )
            + column_offsets
        )
        counts += np.bincount(
            encoded.reshape(-1),
            minlength=width * num_classes,
        ).reshape(width, num_classes)
    return (counts / depth).astype(np.float32)


class MSASourceManager:
    """
    Manages MSA data retrieval and loading from multiple sources.

    Args:
        raw_paths: List of base paths for MSA storage.
        methods: List of indexing methods (e.g., 'sequence' or 'pdb_id').
        mappings: Dictionary mapping source index to its respective lookup table.
        enabled: Whether MSA loading is enabled.
    """

    def __init__(
        self,
        raw_paths: Sequence[str],
        methods: Sequence[str],
        mappings: Dict[int, Dict[Any, Any]],
        enabled: bool,
    ) -> None:
        self.raw_paths = raw_paths
        self.methods = methods
        self.mappings = mappings
        self.enabled = enabled

    def fetch_msas(
        self,
        sequence: str,
        pdb_id: str,
        chain_type: str,
        p_dbs: Optional[Sequence[str]] = None,
        np_dbs: Optional[Sequence[str]] = None,
        depth_limit: Optional[int] = None,
        paired_depth_limit: Optional[int] = None,
        unpaired_depth_limit: Optional[int] = None,
        independent_pairing_non_pairing_budget: bool = False,
    ) -> Tuple[List[RawMsa], List[RawMsa]]:
        """
        Fetches MSAs from the configured paths based on chain type and indexing methods.

        Args:
            sequence: Query sequence.
            pdb_id: PDB identifier.
            chain_type: Type of the chain (e.g., PROTEIN_CHAIN, RNA_CHAIN).
            p_dbs: Databases for paired MSA search.
            np_dbs: Databases for unpaired MSA search.
            depth_limit: Legacy shared cap for both candidate pools.
            paired_depth_limit: Independent cap for paired raw rows. When
                omitted, ``depth_limit`` is used for backward compatibility.
            unpaired_depth_limit: Independent cap for unpaired raw rows. When
                omitted, ``depth_limit`` is used for backward compatibility.
            independent_pairing_non_pairing_budget: Reuse a paired source that
                is also named in ``np_dbs`` without charging it to the genuine
                non-pairing budget. This gives the paired-source copy and
                genuine non-pairing sources independent top-N budgets.

        Returns:
            A tuple of (unpaired_msas, paired_msas).
        """
        if not self.enabled:
            return [], []
        if not isinstance(
            independent_pairing_non_pairing_budget,
            (bool, np.bool_),
        ):
            raise TypeError(
                "independent_pairing_non_pairing_budget must be a boolean"
            )
        independent_pairing_non_pairing_budget = bool(
            independent_pairing_non_pairing_budget
        )
        unpaired, paired = [], []
        legacy_depth_limit = (
            None
            if depth_limit is None
            else _strict_positive_int(depth_limit, "depth_limit")
        )
        unpaired_remaining = (
            legacy_depth_limit
            if unpaired_depth_limit is None
            else _strict_positive_int(
                unpaired_depth_limit,
                "unpaired_depth_limit",
            )
        )
        paired_remaining = (
            legacy_depth_limit
            if paired_depth_limit is None
            else _strict_positive_int(
                paired_depth_limit,
                "paired_depth_limit",
            )
        )

        # RNA-specific loading logic
        if chain_type == RNA_CHAIN:
            for path, method, m_key in zip(self.raw_paths, self.methods, self.mappings):
                if unpaired_remaining is not None and unpaired_remaining <= 0:
                    break
                if method != "sequence":
                    continue
                mapping = self.mappings[m_key]
                if sequence not in mapping:
                    continue
                eid = str(mapping[sequence][0])
                fpath = opjoin(path, eid, f"{eid}_all.a3m")
                if opexists(fpath) and opgetsize(fpath) > 0:
                    source_limit = (
                        min(30000, unpaired_remaining)
                        if unpaired_remaining is not None
                        else 30000
                    )
                    raw_msa = RawMsa.from_a3m_file(
                        sequence,
                        RNA_CHAIN,
                        fpath,
                        depth_limit=source_limit,
                        dedup=False,
                    )
                    unpaired.append(raw_msa)
                    if unpaired_remaining is not None:
                        unpaired_remaining -= raw_msa.depth
            return unpaired, []

        # Protein-specific loading logic
        if chain_type == PROTEIN_CHAIN:
            p_dbs = p_dbs or []
            np_dbs = np_dbs or []
            for p_db, np_db, path, method, m_key in zip(
                p_dbs, np_dbs, self.raw_paths, self.methods, self.mappings
            ):
                key = sequence if method == "sequence" else str(pdb_id)
                mapping = self.mappings[m_key]
                if key not in mapping:
                    continue
                dir_path = opjoin(path, str(mapping[key]))
                paired_msa_for_unpaired = None

                if p_db:
                    for pat in [f"{p_db}.a3m"]:
                        if paired_remaining is not None and paired_remaining <= 0:
                            break
                        fpath = opjoin(dir_path, pat)
                        if opexists(fpath) and opgetsize(fpath) > 0:
                            raw_msa = RawMsa.from_a3m_file(
                                sequence,
                                PROTEIN_CHAIN,
                                fpath,
                                depth_limit=paired_remaining,
                                dedup=False,
                            )
                            paired.append(raw_msa)
                            paired_msa_for_unpaired = raw_msa
                            if paired_remaining is not None:
                                paired_remaining -= raw_msa.depth
                            break
                if np_db:
                    for db in np_db.split("-"):
                        if (
                            independent_pairing_non_pairing_budget
                            and db == p_db
                        ):
                            # Reuse the bounded paired-source object in the
                            # unpaired candidate pool without charging it to
                            # the genuine non-pairing source budget.
                            if paired_msa_for_unpaired is not None:
                                unpaired.append(paired_msa_for_unpaired)
                            continue
                        if (
                            unpaired_remaining is not None
                            and unpaired_remaining <= 0
                        ):
                            if independent_pairing_non_pairing_budget:
                                # A later source may be the independently
                                # budgeted paired-source copy.
                                continue
                            break
                        for pat in [f"{db}.a3m"]:
                            fpath = opjoin(dir_path, pat)
                            if opexists(fpath) and opgetsize(fpath) > 0:
                                raw_msa = RawMsa.from_a3m_file(
                                    sequence,
                                    PROTEIN_CHAIN,
                                    fpath,
                                    depth_limit=unpaired_remaining,
                                    dedup=False,
                                )
                                unpaired.append(raw_msa)
                                if unpaired_remaining is not None:
                                    unpaired_remaining -= raw_msa.depth
                                break
        return unpaired, paired


class FeatureAssemblyLine:
    """
    Orchestrates the conversion of Raw MSAs into finalized Protenix features.

    Args:
        max_msa_size: Maximum number of sequences allowed in the final MSA.
        max_paired_per_species: Maximum number of paired sequences per species.
        pair_msa_as_unpaired_only: Whether to disable species pairing and use
            paired MSA rows as unpaired rows.
        shuffle_pair_msa_when_unpaired_only: Whether to shuffle paired MSA hit
            rows before moving them into unpaired MSA rows.
        enable_virtual_segment_msa_shuffle: Whether to replace the legacy
            paired-hit row shuffle with crop-segment permutations across all
            protein MSA rows.
        max_input_msa_depth: Legacy shared raw-pool cap.
        max_input_paired_msa_depth: Independent paired raw-pool cap.
        max_input_unpaired_msa_depth: Independent unpaired raw-pool cap.
    """

    def __init__(
        self,
        max_msa_size: int = 16384,
        max_paired_per_species: int = 600,
        pair_msa_as_unpaired_only: bool = False,
        shuffle_pair_msa_when_unpaired_only: bool = False,
        enable_virtual_segment_msa_shuffle: bool = False,
        max_input_msa_depth: Optional[int] = None,
        max_input_paired_msa_depth: Optional[int] = None,
        max_input_unpaired_msa_depth: Optional[int] = None,
    ) -> None:
        self.max_size = max_msa_size
        self.max_paired_per_sp = max_paired_per_species
        self.pair_msa_as_unpaired_only = pair_msa_as_unpaired_only
        self.shuffle_pair_msa_when_unpaired_only = shuffle_pair_msa_when_unpaired_only
        self.enable_virtual_segment_msa_shuffle = bool(
            enable_virtual_segment_msa_shuffle
        )
        legacy_depth_limit = (
            None
            if max_input_msa_depth is None
            else _strict_positive_int(
                max_input_msa_depth,
                "max_input_msa_depth",
            )
        )
        self.max_input_paired_msa_depth = (
            legacy_depth_limit
            if max_input_paired_msa_depth is None
            else _strict_positive_int(
                max_input_paired_msa_depth,
                "max_input_paired_msa_depth",
            )
        )
        self.max_input_unpaired_msa_depth = (
            legacy_depth_limit
            if max_input_unpaired_msa_depth is None
            else _strict_positive_int(
                max_input_unpaired_msa_depth,
                "max_input_unpaired_msa_depth",
            )
        )
        # Retain the historical public attribute for callers that only inspect
        # it. Internal code below always uses the side-specific limits.
        self.max_input_msa_depth = (
            self.max_input_paired_msa_depth
            if self.max_input_paired_msa_depth
            == self.max_input_unpaired_msa_depth
            else None
        )
        self.last_virtual_segment_count = 0
        self.last_virtual_segment_shuffle_applied = False

    def assemble(
        self,
        bioassembly: Mapping[int, Mapping[str, Any]],
        std_idxs: np.ndarray,
        *,
        virtual_segment_groups: Optional[
            Mapping[int, Sequence[np.ndarray]]
        ] = None,
        virtual_segment_shuffle_seed: Optional[int] = None,
    ) -> "MSAFeat":
        """
        Executes the complete feature assembly pipeline.

        Args:
            bioassembly: Mapping of asymmetric IDs to chain information.
            std_idxs: Array of standardized residue indices.
            virtual_segment_groups: Canonical residue columns grouped by
                crop-disconnected protein segment for each physical chain.
            virtual_segment_shuffle_seed: Optional deterministic seed for all
                segment permutations in this sampled occurrence.

        Returns:
            An assembled MSAFeat object.
        """
        self.last_virtual_segment_count = sum(
            len(groups)
            for groups in (virtual_segment_groups or {}).values()
        )
        self.last_virtual_segment_shuffle_applied = False
        virtual_shuffle_active = (
            self.enable_virtual_segment_msa_shuffle
            and self.pair_msa_as_unpaired_only
            and self.shuffle_pair_msa_when_unpaired_only
            and self.last_virtual_segment_count >= 2
        )
        virtual_shuffle_rng = (
            np.random.default_rng(virtual_segment_shuffle_seed)
            if virtual_shuffle_active
            else None
        )

        # 1. Base featurization
        unique_prot_seqs = {
            v["sequence"]
            for v in bioassembly.values()
            if v["chain_entity_type"] == PROTEIN_CHAIN
        }
        need_pairing = len(unique_prot_seqs) > 1
        active_chain_ids = {v["chain_id"] for v in bioassembly.values()}

        raw_chains = []
        for aid, info in bioassembly.items():
            ctype, seq = info["chain_entity_type"], info["sequence"]
            skip = ctype not in STANDARD_POLYMER_CHAIN_TYPES or len(seq) <= 4

            if ctype in STANDARD_POLYMER_CHAIN_TYPES:
                unpaired_msa = info["unpaired_msa"]
                paired_msa = info["paired_msa"]

                def as_raw_msa(
                    value: Any,
                    *,
                    enabled: bool,
                    deduplicate: bool,
                    depth_limit: Optional[int],
                ) -> RawMsa:
                    if not enabled:
                        return RawMsa(
                            seq,
                            ctype,
                            [],
                            [],
                            deduplicate=False,
                        )
                    if isinstance(value, RawMsa):
                        if value.query != seq or value.chain_type != ctype:
                            raise ValueError(
                                "Precomputed MSA query or chain type does not "
                                "match the target chain"
                            )
                        return RawMsa.merge(
                            [value],
                            deduplicate=deduplicate,
                            depth_limit=depth_limit,
                        )
                    return RawMsa.from_a3m(
                        seq,
                        ctype,
                        value or "",
                        depth_limit=depth_limit,
                        dedup=deduplicate,
                    )

                up_msa = as_raw_msa(
                    unpaired_msa,
                    enabled=(
                        not skip
                        and ctype in [PROTEIN_CHAIN, RNA_CHAIN]
                    ),
                    deduplicate=True,
                    depth_limit=(
                        self.max_size
                        if info.get(
                            "unpaired_msa_contains_paired_source_rows",
                            False,
                        )
                        else self.max_input_unpaired_msa_depth
                    ),
                )
                p_msa = as_raw_msa(
                    paired_msa,
                    enabled=(
                        not skip
                        and ctype == PROTEIN_CHAIN
                        and (
                            need_pairing
                            or self.pair_msa_as_unpaired_only
                        )
                    ),
                    deduplicate=False,
                    depth_limit=self.max_input_paired_msa_depth,
                )
                if (
                    self.pair_msa_as_unpaired_only
                    and ctype == PROTEIN_CHAIN
                    and not skip
                ):
                    if (
                        self.shuffle_pair_msa_when_unpaired_only
                        and not self.enable_virtual_segment_msa_shuffle
                    ):
                        p_msa = p_msa.shuffled_hits_preserving_query()
                    unpaired_already_contains_paired = bool(
                        info.get(
                            "unpaired_msa_uses_independent_pairing_source_budget",
                            False,
                        )
                    )
                    up_msa = RawMsa.merge(
                        [p_msa, up_msa],
                        deduplicate=True,
                        # When the unpaired pool already carries the bounded
                        # paired-source copy, deduplicate before applying the
                        # final cap so duplicate paired rows cannot crowd out
                        # genuine non-pairing rows.
                        depth_limit=(
                            None
                            if unpaired_already_contains_paired
                            else self.max_size
                        ),
                    )
                    if unpaired_already_contains_paired:
                        up_msa = RawMsa.merge(
                            [up_msa],
                            deduplicate=False,
                            depth_limit=self.max_size,
                        )
                    p_msa = RawMsa(
                        seq,
                        ctype,
                        [],
                        [],
                        deduplicate=False,
                    )
            else:
                up_msa = p_msa = RawMsa(
                    seq, PROTEIN_CHAIN, [], [], deduplicate=False
                )  # Ligand placeholders

            u_f, p_f = up_msa.featurize(), p_msa.featurize()
            chain_feat = dict(u_f)
            chain_feat.update({f"{k}_all_seq": v for k, v in p_f.items()})
            chain_feat.update(
                {
                    "asym_id": np.full(len(seq), aid),
                    "chain_id": info["chain_id"],
                    "entity_id": info["entity_id"],
                }
            )
            if (
                virtual_shuffle_rng is not None
                and ctype == PROTEIN_CHAIN
                and aid in virtual_segment_groups
            ):
                (
                    chain_feat["msa"],
                    chain_feat["deletion_matrix"],
                    shuffle_attempted,
                ) = shuffle_msa_feature_rows_by_segment(
                    msa=chain_feat["msa"],
                    deletion_matrix=chain_feat["deletion_matrix"],
                    segment_columns=virtual_segment_groups[aid],
                    rng=virtual_shuffle_rng,
                    copy=False,
                )
                self.last_virtual_segment_shuffle_applied |= (
                    shuffle_attempted
                )
            # Compute Profile
            msa = chain_feat["msa"]
            prof = compute_msa_profile(msa)
            chain_feat.update(
                {
                    "profile": prof,
                    "deletion_mean": np.mean(chain_feat["deletion_matrix"], axis=0),
                }
            )
            raw_chains.append(chain_feat)

        # 2. Pairing and cleanup
        max_p = self.max_size // 2
        if need_pairing:
            if not self.pair_msa_as_unpaired_only:
                raw_chains = MSAPairingEngine.pair_chains_by_species(
                    raw_chains, max_p, active_chain_ids, self.max_paired_per_sp
                )
            raw_chains = MSAPairingEngine.cleanup_unpaired_features(raw_chains)

        # 3. Filter all-gap rows
        nonempty_asyms = [
            c["asym_id"][0] for c in raw_chains if c["chain_id"] in active_chain_ids
        ]
        if "msa_all_seq" in raw_chains[0]:
            raw_chains = MSAPairingEngine.filter_all_gapped_rows(
                raw_chains, nonempty_asyms
            )

        # 4. Cropping and merging
        cropped = []
        for c in raw_chains:
            p_msa = c.get("msa_all_seq")
            ps = min(p_msa.shape[0], max_p) if p_msa is not None else 0
            us = max(0, min(c["msa"].shape[0], self.max_size - ps))

            cr = {
                "asym_id": c["asym_id"],
                "chain_id": c["chain_id"],
                "profile": c["profile"],
                "deletion_mean": c["deletion_mean"],
            }
            for k in NUM_SEQ_NUM_RES_MSA_FEATURES:
                if k in c:
                    cr[k] = c[k][:us]
                if f"{k}_all_seq" in c:
                    cr[f"{k}_all_seq"] = c[f"{k}_all_seq"][:ps]
            cropped.append(cr)

        merged = {"asym_id": np.concatenate([c["asym_id"] for c in cropped])}
        for base in NUM_SEQ_NUM_RES_MSA_FEATURES:
            for f in [base, f"{base}_all_seq"]:
                if f in cropped[0]:
                    merged[f] = MSAPairingEngine.merge_chain_features(cropped, f)
        for f in ["profile", "deletion_mean"]:
            merged[f] = np.concatenate([c[f] for c in cropped])

        # 5. Depth tracking
        active_set = set(nonempty_asyms)
        max_u = max([len(c["msa"]) for c in cropped if c["asym_id"][0] in active_set])
        rna_u = max(
            [1]
            + [
                len(c["msa"])
                for c in cropped
                if bioassembly[c["asym_id"][0]]["chain_entity_type"] == RNA_CHAIN
            ]
        )
        prot_u = max(
            [1]
            + [
                len(c["msa"])
                for c in cropped
                if bioassembly[c["asym_id"][0]]["chain_entity_type"] == PROTEIN_CHAIN
            ]
        )

        merged["msa"] = merged["msa"][:max_u]
        prot_p = 1
        if "msa_all_seq" in merged:
            max_p_actual = max(
                [
                    len(c["msa_all_seq"])
                    for c in cropped
                    if c["asym_id"][0] in active_set
                ]
            )
            merged["msa_all_seq"] = merged["msa_all_seq"][:max_p_actual]
            prot_p = max_p_actual

        # 6. Final integration and coordinate mapping
        for k in NUM_SEQ_NUM_RES_MSA_FEATURES:
            if k in merged and f"{k}_all_seq" in merged:
                merged[k] = np.concatenate([merged[f"{k}_all_seq"], merged[k]], axis=0)

        # Forward compatibility patch for non-protein entities
        for aid in [
            aid
            for aid, info in bioassembly.items()
            if info["chain_entity_type"] != PROTEIN_CHAIN
        ]:
            cols = np.where(merged["asym_id"] == aid)[0]
            if cols.size > 0:
                gap_mask = np.all(merged["msa"][:, cols] == MSA_GAP_IDX, axis=1)
                merged["msa"][np.ix_(np.where(gap_mask)[0], cols)] = merged["msa"][
                    0, cols
                ]

        for f in NUM_SEQ_NUM_RES_MSA_FEATURES:
            if f in merged:
                merged[f] = merged[f][:, std_idxs].copy()
        for f in ["profile", "deletion_mean"]:
            merged[f] = merged[f][std_idxs]

        def to_i8(x: np.ndarray) -> np.ndarray:
            return np.clip(x, -128, 127).astype(np.int8)

        return MSAFeat(
            rows=to_i8(merged["msa"]),
            mask=np.ones_like(merged["msa"], dtype=bool),
            deletion_matrix=to_i8(merged["deletion_matrix"]),
            profile=merged["profile"],
            deletion_mean=merged["deletion_mean"],
            prot_unpaired_num_alignments=np.array(prot_u, dtype=np.int32),
            prot_paired_num_alignments=np.array(prot_p, dtype=np.int32),
            rna_unpaired_num_alignments=np.array(rna_u, dtype=np.int32),
        )


class MSAFeaturizer:
    """
    Main entry point for MSA featurization, coordinating source management and assembly.

    Args:
        dataset_name: Name of the dataset.
        prot_seq_or_filename_to_msadir_jsons: JSON maps for protein MSA lookups.
        prot_msadir_raw_paths: Base paths for protein MSAs.
        rna_seq_or_filename_to_msadir_jsons: JSON maps for RNA MSA lookups.
        rna_msadir_raw_paths: Base paths for RNA MSAs.
        prot_pairing_dbs: List of databases for protein pairing.
        prot_non_pairing_dbs: List of databases for protein non-pairing.
        prot_indexing_methods: Methods for protein MSA indexing.
        rna_indexing_methods: Methods for RNA MSA indexing.
        enable_prot_msa: Whether to enable protein MSA processing.
        enable_rna_msa: Whether to enable RNA MSA processing.
        max_msa_size: Maximum number of sequences allowed in the assembled MSA.
        pair_msa_as_unpaired_only: Whether to disable protein MSA species pairing
            and use paired MSA rows only as unpaired rows.
        shuffle_pair_msa_when_unpaired_only: Whether to shuffle paired MSA hit
            rows before using them as unpaired rows.
        enable_virtual_segment_msa_shuffle: Whether unpaired-shuffle mode
            independently permutes all protein MSA rows by crop segment.
        virtual_segment_residue_gap: Minimum adjacent canonical residue-index
            delta that begins a raw segment.
        virtual_segment_min_length: Minimum unique-residue count for an
            independently permuted segment.
        enable_multichain_msa_shuffle_variant_sampling: Whether a crop with at
            least two physical protein chains independently chooses segment or
            whole-physical-chain grouping.
        multichain_msa_shuffle_segment_probability: Segment-grouping
            probability for eligible multi-chain crops.
        enable_crop_aware_partial_query_only: Whether a query-only occurrence
            with multiple crop-visible protein chains drops homologs for only
            a bounded subset of physical chains.
        partial_query_only_max_chains: Maximum physical protein chains made
            query-only in a multi-chain crop.
        max_input_msa_depth: Legacy shared training-only prefix cap for paired
            and unpaired raw candidate pools.
        max_input_paired_msa_depth: Independent training-only prefix cap for
            the paired raw candidate pool before deduplication.
        max_input_unpaired_msa_depth: Independent training-only prefix cap for
            the unpaired raw candidate pool before deduplication.
        independent_pairing_non_pairing_budget: Whether a paired source also
            listed as an unpaired source has its own top-N budget, independent
            from genuine non-pairing sources.
    """

    def __init__(
        self,
        dataset_name: str = "",
        prot_seq_or_filename_to_msadir_jsons: Sequence[str] = [""],
        prot_msadir_raw_paths: Sequence[str] = [""],
        rna_seq_or_filename_to_msadir_jsons: Sequence[str] = [""],
        rna_msadir_raw_paths: Sequence[str] = [""],
        prot_pairing_dbs: Sequence[str] = [""],
        prot_non_pairing_dbs: Sequence[str] = [""],
        prot_indexing_methods: Sequence[str] = ["sequence"],
        rna_indexing_methods: Sequence[str] = ["sequence"],
        enable_prot_msa: bool = True,
        enable_rna_msa: bool = True,
        max_msa_size: int = 16384,
        pair_msa_as_unpaired_only: bool = False,
        shuffle_pair_msa_when_unpaired_only: bool = False,
        enable_virtual_segment_msa_shuffle: bool = False,
        virtual_segment_residue_gap: int = 24,
        virtual_segment_min_length: int = 9,
        enable_crop_aware_partial_query_only: bool = False,
        partial_query_only_max_chains: int = 2,
        openfold3_alignment_npz_dirs: Sequence[str] = (),
        enable_multichain_msa_shuffle_variant_sampling: bool = False,
        multichain_msa_shuffle_segment_probability: float = 0.5,
        max_input_msa_depth: Optional[int] = None,
        max_input_paired_msa_depth: Optional[int] = None,
        max_input_unpaired_msa_depth: Optional[int] = None,
        independent_pairing_non_pairing_budget: bool = False,
    ) -> None:
        self.dataset_name = dataset_name
        super().__init__()
        # Initialize source managers for protein and RNA
        self.prot_mgr = MSASourceManager(
            prot_msadir_raw_paths,
            prot_indexing_methods,
            {
                i: load_json_cached(p) if p else {}
                for i, p in enumerate(prot_seq_or_filename_to_msadir_jsons)
            },
            enable_prot_msa,
        )
        self.rna_mgr = MSASourceManager(
            rna_msadir_raw_paths,
            rna_indexing_methods,
            {
                i: load_json_cached(p) if p else {}
                for i, p in enumerate(rna_seq_or_filename_to_msadir_jsons)
            },
            enable_rna_msa,
        )
        self.prot_p_dbs = prot_pairing_dbs
        self.prot_np_dbs = prot_non_pairing_dbs
        if not isinstance(
            independent_pairing_non_pairing_budget,
            (bool, np.bool_),
        ):
            raise TypeError(
                "independent_pairing_non_pairing_budget must be a boolean"
            )
        self.independent_pairing_non_pairing_budget = bool(
            independent_pairing_non_pairing_budget
            and _has_paired_copy_and_genuine_non_pairing_sources(
                self.prot_p_dbs,
                self.prot_np_dbs,
            )
        )
        self.max_msa_size = max_msa_size
        legacy_depth_limit = (
            None
            if max_input_msa_depth is None
            else _strict_positive_int(
                max_input_msa_depth,
                "max_input_msa_depth",
            )
        )
        self.max_input_paired_msa_depth = (
            legacy_depth_limit
            if max_input_paired_msa_depth is None
            else _strict_positive_int(
                max_input_paired_msa_depth,
                "max_input_paired_msa_depth",
            )
        )
        self.max_input_unpaired_msa_depth = (
            legacy_depth_limit
            if max_input_unpaired_msa_depth is None
            else _strict_positive_int(
                max_input_unpaired_msa_depth,
                "max_input_unpaired_msa_depth",
            )
        )
        # Backward-compatible introspection only. Runtime code always uses the
        # independent paired/unpaired limits above.
        self.max_input_msa_depth = (
            self.max_input_paired_msa_depth
            if self.max_input_paired_msa_depth
            == self.max_input_unpaired_msa_depth
            else None
        )
        self.pair_msa_as_unpaired_only = pair_msa_as_unpaired_only
        self.shuffle_pair_msa_when_unpaired_only = (
            shuffle_pair_msa_when_unpaired_only
        )
        if not isinstance(enable_virtual_segment_msa_shuffle, (bool, np.bool_)):
            raise TypeError(
                "enable_virtual_segment_msa_shuffle must be a boolean"
            )
        self.enable_virtual_segment_msa_shuffle = bool(
            enable_virtual_segment_msa_shuffle
        )
        self.virtual_segment_residue_gap = _strict_positive_int(
            virtual_segment_residue_gap,
            "virtual_segment_residue_gap",
        )
        self.virtual_segment_min_length = _strict_positive_int(
            virtual_segment_min_length,
            "virtual_segment_min_length",
        )
        if not isinstance(
            enable_multichain_msa_shuffle_variant_sampling,
            (bool, np.bool_),
        ):
            raise TypeError(
                "enable_multichain_msa_shuffle_variant_sampling must be a "
                "boolean"
            )
        self.enable_multichain_msa_shuffle_variant_sampling = bool(
            enable_multichain_msa_shuffle_variant_sampling
        )
        if isinstance(
            multichain_msa_shuffle_segment_probability,
            (bool, np.bool_),
        ):
            raise TypeError(
                "multichain_msa_shuffle_segment_probability must be a real "
                "number"
            )
        try:
            multichain_msa_shuffle_segment_probability = float(
                multichain_msa_shuffle_segment_probability
            )
        except (TypeError, ValueError) as error:
            raise TypeError(
                "multichain_msa_shuffle_segment_probability must be a real "
                "number"
            ) from error
        if not np.isfinite(
            multichain_msa_shuffle_segment_probability
        ) or not (
            0.0
            <= multichain_msa_shuffle_segment_probability
            <= 1.0
        ):
            raise ValueError(
                "multichain_msa_shuffle_segment_probability must be finite "
                "and in [0, 1], got "
                f"{multichain_msa_shuffle_segment_probability!r}"
            )
        if (
            self.enable_multichain_msa_shuffle_variant_sampling
            and not self.enable_virtual_segment_msa_shuffle
        ):
            raise ValueError(
                "enable_multichain_msa_shuffle_variant_sampling requires "
                "enable_virtual_segment_msa_shuffle"
            )
        self.multichain_msa_shuffle_segment_probability = (
            multichain_msa_shuffle_segment_probability
        )
        if not isinstance(
            enable_crop_aware_partial_query_only,
            (bool, np.bool_),
        ):
            raise TypeError(
                "enable_crop_aware_partial_query_only must be a boolean"
            )
        self.enable_crop_aware_partial_query_only = bool(
            enable_crop_aware_partial_query_only
        )
        self.partial_query_only_max_chains = _strict_positive_int(
            partial_query_only_max_chains,
            "partial_query_only_max_chains",
        )
        self.openfold3_alignment_npz_dirs = tuple(
            Path(path).expanduser()
            for path in openfold3_alignment_npz_dirs
            if str(path).strip()
        )
        missing_alignment_dirs = [
            path
            for path in self.openfold3_alignment_npz_dirs
            if not path.is_dir()
        ]
        if missing_alignment_dirs:
            raise FileNotFoundError(
                "OpenFold3 alignment NPZ directories do not exist: "
                + ", ".join(str(path) for path in missing_alignment_dirs)
            )
        self._profile: Dict[str, Any] = {}
        logger.info(f"MSAFeaturizer for {dataset_name} initialized.")

    def set_last_profile(self, p: Dict[str, Any]) -> None:
        """Sets the internal profile state."""
        self._profile = p

    def get_last_profile(self) -> Dict[str, Any]:
        """Returns the internal profile state."""
        return self._profile

    def _resolve_openfold3_alignment_path(
        self,
        pdb_id: str,
        label_asym_id: str,
    ) -> Optional[Path]:
        """Resolve an exact, case-sensitive chain alignment archive."""
        if (
            not pdb_id
            or not label_asym_id
            or any(char in pdb_id for char in ("/", "\\", "\0"))
            or any(char in label_asym_id for char in ("/", "\\", "\0"))
        ):
            raise ValueError(
                "Unsafe OpenFold3 alignment key: "
                f"{pdb_id!r}, {label_asym_id!r}"
            )
        pdb_candidates = tuple(dict.fromkeys((pdb_id, pdb_id.lower(), pdb_id.upper())))
        for directory in getattr(self, "openfold3_alignment_npz_dirs", ()):
            for candidate in pdb_candidates:
                path = directory / f"{candidate}_{label_asym_id}.npz"
                if path.is_file():
                    return path
        return None

    @staticmethod
    def _raw_msa_from_openfold3_payload(
        *,
        query: str,
        chain_type: str,
        payload: Any,
        source_name: str,
        path: Path,
        depth_limit: Optional[int] = None,
    ) -> RawMsa:
        """Validate one trusted OpenFold3 alignment source."""
        if not isinstance(payload, dict):
            raise TypeError(
                f"{path}:{source_name} must contain a dictionary"
            )
        missing = {
            "msa",
            "deletion_matrix",
            "metadata",
        } - set(payload)
        if missing:
            raise ValueError(
                f"{path}:{source_name} is missing {sorted(missing)}"
            )
        return RawMsa.from_aligned_arrays(
            query,
            chain_type,
            np.asarray(payload["msa"]),
            np.asarray(payload["deletion_matrix"]),
            np.asarray(payload["metadata"]),
            dedup=False,
            depth_limit=depth_limit,
        )

    def _load_openfold3_chain_msas(
        self,
        *,
        pdb_id: str,
        label_asym_id: str,
        query: str,
        chain_type: str,
        depth_limit: Optional[int] = None,
        paired_depth_limit: Optional[int] = None,
        unpaired_depth_limit: Optional[int] = None,
    ) -> tuple[list[RawMsa], list[RawMsa]]:
        """Load per-chain OpenFold3 arrays without an A3M round trip."""
        path = self._resolve_openfold3_alignment_path(
            pdb_id,
            label_asym_id,
        )
        if path is None:
            return [], []

        if chain_type == PROTEIN_CHAIN:
            unpaired_sources = _OPENFOLD3_PROTEIN_UNPAIRED_SOURCES
            paired_sources = _OPENFOLD3_PROTEIN_PAIRED_SOURCES
        elif chain_type == RNA_CHAIN:
            unpaired_sources = _OPENFOLD3_RNA_UNPAIRED_SOURCES
            paired_sources = ()
        else:
            return [], []

        unpaired_msas: list[RawMsa] = []
        paired_msas: list[RawMsa] = []
        legacy_depth_limit = (
            None
            if depth_limit is None
            else _strict_positive_int(depth_limit, "depth_limit")
        )
        unpaired_remaining = (
            legacy_depth_limit
            if unpaired_depth_limit is None
            else _strict_positive_int(
                unpaired_depth_limit,
                "unpaired_depth_limit",
            )
        )
        paired_remaining = (
            legacy_depth_limit
            if paired_depth_limit is None
            else _strict_positive_int(
                paired_depth_limit,
                "paired_depth_limit",
            )
        )
        with np.load(path, allow_pickle=True) as archive:
            for source_name, destination in (
                *((name, unpaired_msas) for name in unpaired_sources),
                *((name, paired_msas) for name in paired_sources),
            ):
                remaining = (
                    unpaired_remaining
                    if destination is unpaired_msas
                    else paired_remaining
                )
                if remaining is not None and remaining <= 0:
                    continue
                if source_name not in archive.files:
                    continue
                source_value = archive[source_name]
                if source_value.shape != () or source_value.dtype != object:
                    raise ValueError(
                        f"{path}:{source_name} must be a scalar object array"
                    )
                raw_msa = self._raw_msa_from_openfold3_payload(
                    query=query,
                    chain_type=chain_type,
                    payload=source_value.item(),
                    source_name=source_name,
                    path=path,
                    depth_limit=remaining,
                )
                destination.append(raw_msa)
                if destination is unpaired_msas:
                    if unpaired_remaining is not None:
                        unpaired_remaining -= raw_msa.depth
                elif paired_remaining is not None:
                    paired_remaining -= raw_msa.depth
        return unpaired_msas, paired_msas

    def _load_openfold3_recovery_candidate(
        self,
        *,
        representative_id: str,
        chain_type: str,
        depth_limit: Optional[int] = None,
        paired_depth_limit: Optional[int] = None,
        unpaired_depth_limit: Optional[int] = None,
    ) -> _OpenFold3RecoveryCandidate:
        """Load one declared representative using its authoritative query.

        This helper is called only after the normal loader reports an exact
        canonical-query mismatch. Recovery obeys the same input-depth cap as
        the normal OpenFold3 path.
        """
        try:
            pdb_id, label_asym_id = representative_id.rsplit("_", 1)
        except ValueError as error:
            raise ValueError(
                "Invalid OpenFold3 alignment representative ID: "
                f"{representative_id!r}"
            ) from error
        path = self._resolve_openfold3_alignment_path(
            pdb_id,
            label_asym_id,
        )
        if path is None:
            raise FileNotFoundError(
                "OpenFold3 alignment archive is missing for declared "
                f"representative {representative_id!r}"
            )

        if chain_type == PROTEIN_CHAIN:
            unpaired_sources = _OPENFOLD3_PROTEIN_UNPAIRED_SOURCES
            paired_sources = _OPENFOLD3_PROTEIN_PAIRED_SOURCES
        elif chain_type == RNA_CHAIN:
            unpaired_sources = _OPENFOLD3_RNA_UNPAIRED_SOURCES
            paired_sources = ()
        else:
            raise ValueError(
                "OpenFold3 representative recovery only supports protein "
                f"and RNA chains, got {chain_type!r}"
            )

        canonical_query: Optional[str] = None
        source_count = 0
        unpaired_msas: list[RawMsa] = []
        paired_msas: list[RawMsa] = []
        legacy_depth_limit = (
            None
            if depth_limit is None
            else _strict_positive_int(depth_limit, "depth_limit")
        )
        unpaired_remaining = (
            legacy_depth_limit
            if unpaired_depth_limit is None
            else _strict_positive_int(
                unpaired_depth_limit,
                "unpaired_depth_limit",
            )
        )
        paired_remaining = (
            legacy_depth_limit
            if paired_depth_limit is None
            else _strict_positive_int(
                paired_depth_limit,
                "paired_depth_limit",
            )
        )
        with np.load(path, allow_pickle=True) as archive:
            for source_name, destination in (
                *((name, unpaired_msas) for name in unpaired_sources),
                *((name, paired_msas) for name in paired_sources),
            ):
                if source_name not in archive.files:
                    continue
                source_count += 1
                source_value = archive[source_name]
                if source_value.shape != () or source_value.dtype != object:
                    raise ValueError(
                        f"{path}:{source_name} must be a scalar object array"
                    )
                payload = source_value.item()
                if not isinstance(payload, dict):
                    raise TypeError(
                        f"{path}:{source_name} must contain a dictionary"
                    )
                missing = {
                    "msa",
                    "deletion_matrix",
                    "metadata",
                } - set(payload)
                if missing:
                    raise ValueError(
                        f"{path}:{source_name} is missing {sorted(missing)}"
                    )
                msa = np.asarray(payload["msa"])
                if msa.ndim != 2 or msa.shape[0] == 0:
                    raise ValueError(
                        f"{path}:{source_name} has invalid MSA shape "
                        f"{msa.shape}"
                    )
                if msa.dtype.kind not in {"S", "U"}:
                    raise TypeError(
                        f"{path}:{source_name} MSA must contain characters, "
                        f"got {msa.dtype}"
                    )
                source_query = "".join(
                    msa[0].astype("<U1", copy=False).tolist()
                )
                if canonical_query is None:
                    canonical_query = source_query
                elif source_query != canonical_query:
                    raise ValueError(
                        "OpenFold3 alignment sources do not share one exact "
                        f"canonical query: {path}"
                    )

                remaining = (
                    unpaired_remaining
                    if destination is unpaired_msas
                    else paired_remaining
                )
                if remaining is not None and remaining <= 0:
                    continue
                raw_msa = self._raw_msa_from_openfold3_payload(
                    query=canonical_query,
                    chain_type=chain_type,
                    payload=payload,
                    source_name=source_name,
                    path=path,
                    depth_limit=remaining,
                )
                destination.append(raw_msa)
                if destination is unpaired_msas:
                    if unpaired_remaining is not None:
                        unpaired_remaining -= raw_msa.depth
                elif paired_remaining is not None:
                    paired_remaining -= raw_msa.depth

        if source_count == 0 or canonical_query is None:
            raise ValueError(
                f"{path} has no supported MSA sources for {chain_type!r}"
            )
        return _OpenFold3RecoveryCandidate(
            representative_id=representative_id,
            canonical_query=canonical_query,
            unpaired_msas=tuple(unpaired_msas),
            paired_msas=tuple(paired_msas),
        )

    def make_msa_features(
        self,
        bioassembly_dict: Dict[str, Any],
        selected_indices: Optional[np.ndarray],
        entity_to_asym_id_int: Mapping[str, Sequence[int]],
        query_only: bool = False,
        pair_msa_as_unpaired_only: Optional[bool] = None,
        shuffle_pair_msa_when_unpaired_only: Optional[bool] = None,
        virtual_segment_shuffle_seed: Optional[int] = None,
        query_only_chain_selection_seed: Optional[int] = None,
        msa_shuffle_variant_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Processes bioassembly information into a dictionary of MSA features.

        Args:
            bioassembly_dict: Dictionary containing biological assembly data.
            selected_indices: Optional array of indices to select from the token array.
            entity_to_asym_id_int: Mapping from entity ID to asymmetric IDs.
            query_only: Whether this sampled occurrence should use only the
                authoritative global query row. The caller makes this decision.
            pair_msa_as_unpaired_only: Optional per-occurrence override of the
                instance's species-pairing policy.
            shuffle_pair_msa_when_unpaired_only: Optional per-occurrence
                override of the paired-hit shuffle policy.
            virtual_segment_shuffle_seed: Optional deterministic seed for
                crop-segment row permutations.
            query_only_chain_selection_seed: Optional deterministic seed for
                choosing physical protein chains in a partial query-only
                occurrence. Direct calls without an occurrence use seed zero.
            msa_shuffle_variant_seed: Independent deterministic seed for the
                eligible multi-chain segment-versus-whole grouping decision.

        Returns:
            A dictionary containing processed MSA features.
        """
        atom_array, token_array = (
            bioassembly_dict["atom_array"],
            bioassembly_dict["token_array"],
        )
        direct_protein_unpaired_msa_path = bioassembly_dict.get(
            DIRECT_PROTEIN_UNPAIRED_MSA_PATH
        )
        max_input_paired_msa_depth = getattr(
            self,
            "max_input_paired_msa_depth",
            getattr(self, "max_input_msa_depth", None),
        )
        max_input_unpaired_msa_depth = getattr(
            self,
            "max_input_unpaired_msa_depth",
            getattr(self, "max_input_msa_depth", None),
        )
        independent_pairing_non_pairing_budget = bool(
            getattr(
                self,
                "independent_pairing_non_pairing_budget",
                False,
            )
        )
        direct_protein_unpaired_msas: Dict[str, RawMsa] = {}
        alignment_representatives = bioassembly_dict.get(
            "alignment_representative_id_by_chain_id"
        )
        sequences_by_chain_id = bioassembly_dict.get(
            "sequences_by_chain_id",
            {},
        )
        if (
            alignment_representatives is not None
            and not isinstance(alignment_representatives, Mapping)
        ):
            raise TypeError(
                "alignment_representative_id_by_chain_id must be a mapping"
            )
        if not isinstance(sequences_by_chain_id, Mapping):
            raise TypeError("sequences_by_chain_id must be a mapping")
        openfold3_chain_msas: Dict[
            tuple[str, ...],
            tuple[list[RawMsa], list[RawMsa]],
        ] = {}
        openfold3_alignment_npz_dirs = tuple(
            getattr(self, "openfold3_alignment_npz_dirs", ())
        )
        openfold3_runtime_recovery_enabled = bool(
            getattr(self, "dataset_name", "")
            == _OPENFOLD3_RUNTIME_RECOVERY_DATASET
            and openfold3_alignment_npz_dirs
        )
        openfold3_explicit_query_only_chain_ids: set[str] = set()
        if openfold3_runtime_recovery_enabled:
            raw_explicit_query_only_chain_ids = bioassembly_dict.get(
                "openfold3_query_only_chain_ids",
                (),
            )
            if isinstance(raw_explicit_query_only_chain_ids, (str, bytes)):
                raise TypeError(
                    "openfold3_query_only_chain_ids must be a sequence of "
                    "physical chain IDs"
                )
            try:
                openfold3_explicit_query_only_chain_ids = {
                    str(chain_id)
                    for chain_id in raw_explicit_query_only_chain_ids
                }
            except TypeError as error:
                raise TypeError(
                    "openfold3_query_only_chain_ids must be a sequence of "
                    "physical chain IDs"
                ) from error
            if "" in openfold3_explicit_query_only_chain_ids:
                raise ValueError(
                    "openfold3_query_only_chain_ids cannot contain an empty "
                    "chain ID"
                )
            unknown_explicit_query_only_chain_ids = (
                openfold3_explicit_query_only_chain_ids
                - {str(chain_id) for chain_id in sequences_by_chain_id}
            )
            if unknown_explicit_query_only_chain_ids:
                raise ValueError(
                    "openfold3_query_only_chain_ids contains chain IDs absent "
                    "from sequences_by_chain_id: "
                    f"{sorted(unknown_explicit_query_only_chain_ids)!r}"
                )
        declared_openfold3_representatives = tuple(
            sorted(
                {
                    str(representative_id)
                    for representative_id in (
                        alignment_representatives.values()
                        if alignment_representatives is not None
                        else ()
                    )
                    if representative_id
                }
            )
        )
        openfold3_recovery_candidates: Dict[
            tuple[str, str],
            _OpenFold3RecoveryCandidate,
        ] = {}
        openfold3_representative_query_mismatch_count = 0
        openfold3_representative_remap_count = 0
        openfold3_representative_query_only_fallback_count = 0
        openfold3_explicit_query_only_chain_count = 0
        openfold3_recovery_candidate_load_count = 0
        openfold3_recovery_candidate_cache_hit_count = 0
        openfold3_remapped_chain_ids: list[str] = []
        openfold3_representative_remaps: list[dict[str, str]] = []
        openfold3_query_only_fallback_chain_ids: list[str] = []
        openfold3_recovery_candidate_errors: dict[str, str] = {}

        def get_openfold3_recovery_candidate(
            representative_id: str,
            chain_type: str,
        ) -> _OpenFold3RecoveryCandidate:
            nonlocal openfold3_recovery_candidate_load_count
            nonlocal openfold3_recovery_candidate_cache_hit_count
            key = (representative_id, chain_type)
            if key in openfold3_recovery_candidates:
                openfold3_recovery_candidate_cache_hit_count += 1
                return openfold3_recovery_candidates[key]
            openfold3_recovery_candidate_load_count += 1
            try:
                candidate = self._load_openfold3_recovery_candidate(
                    representative_id=representative_id,
                    chain_type=chain_type,
                    paired_depth_limit=max_input_paired_msa_depth,
                    unpaired_depth_limit=max_input_unpaired_msa_depth,
                )
            except Exception as error:
                error_message = f"{type(error).__name__}: {error}"
                candidate = _OpenFold3RecoveryCandidate(
                    representative_id=representative_id,
                    canonical_query=None,
                    unpaired_msas=(),
                    paired_msas=(),
                    error=error_message,
                )
                openfold3_recovery_candidate_errors[
                    f"{representative_id}:{chain_type}"
                ] = error_message
            openfold3_recovery_candidates[key] = candidate
            return candidate

        sel_tokens = (
            token_array[selected_indices]
            if selected_indices is not None
            else token_array
        )
        sel_asyms = set(
            atom_array[sel_tokens.get_annotation("centre_atom_index")].asym_id_int
        )

        # 1. Resolve metadata and fetch MSAs
        meta = {}
        poly_map = {
            "polypeptide(L)": PROTEIN_CHAIN,
            "polyribonucleotide": RNA_CHAIN,
            "polydeoxyribonucleotide": DNA_CHAIN,
        }
        crop_protein_asym_ids = sorted(
            {
                int(aid)
                for eid, asyms in entity_to_asym_id_int.items()
                if poly_map.get(
                    bioassembly_dict["entity_poly_type"].get(
                        eid,
                        "non-polymer",
                    ),
                    LIGAND_CHAIN_TYPES,
                )
                == PROTEIN_CHAIN
                for aid in asyms
                if aid in sel_asyms
            }
        )
        partial_query_only_enabled = bool(
            getattr(
                self,
                "enable_crop_aware_partial_query_only",
                False,
            )
        )
        partial_query_only_max_chains = _strict_positive_int(
            getattr(self, "partial_query_only_max_chains", 2),
            "partial_query_only_max_chains",
        )
        partial_query_only_applied = bool(
            query_only
            and partial_query_only_enabled
            and len(crop_protein_asym_ids) >= 2
        )
        global_query_only_applied = bool(
            query_only and not partial_query_only_applied
        )
        if partial_query_only_applied:
            selection_rng = np.random.default_rng(
                0
                if query_only_chain_selection_seed is None
                else query_only_chain_selection_seed
            )
            query_only_protein_asym_ids = (
                select_query_only_protein_asym_ids(
                    protein_asym_ids=crop_protein_asym_ids,
                    max_query_only_chains=partial_query_only_max_chains,
                    rng=selection_rng,
                )
            )
        else:
            query_only_protein_asym_ids = tuple(crop_protein_asym_ids)
        query_only_protein_asym_id_set = set(
            query_only_protein_asym_ids
        )
        for eid, asyms in entity_to_asym_id_int.items():
            for aid in [a for a in asyms if a in sel_asyms]:
                aid_mask = atom_array.asym_id_int == aid
                chain_ids = np.unique(atom_array.chain_id[aid_mask])
                if chain_ids.size != 1:
                    raise ValueError(
                        "Each assembly asym_id_int must map to exactly one "
                        "chain_id"
                    )
                chain_id = str(chain_ids[0])
                seq = sequences_by_chain_id.get(chain_id) or (
                    bioassembly_dict["sequences"].get(eid)
                ) or (
                    "X" * (atom_array.asym_id_int == aid).sum()
                )
                ctype = poly_map.get(
                    bioassembly_dict["entity_poly_type"].get(eid, "non-polymer"),
                    LIGAND_CHAIN_TYPES,
                )
                explicit_openfold3_query_only = bool(
                    chain_id in openfold3_explicit_query_only_chain_ids
                )
                if (
                    explicit_openfold3_query_only
                    and ctype not in (PROTEIN_CHAIN, RNA_CHAIN)
                ):
                    raise ValueError(
                        "openfold3_query_only_chain_ids can only mark protein "
                        f"or RNA chains, got chain {chain_id!r} with "
                        f"type {ctype!r}"
                    )
                if explicit_openfold3_query_only:
                    openfold3_explicit_query_only_chain_count += 1

                up_msas, p_msas = [], []
                paired_sources_moved_to_unpaired = False
                independent_pairing_copy_in_unpaired = False
                # Query-only samples must be derived from the authoritative
                # target sequences. Avoiding homolog I/O also prevents a
                # malformed A3M row zero from becoming the retained query.
                skip_homolog_io = bool(
                    explicit_openfold3_query_only
                    or global_query_only_applied
                    or (
                        partial_query_only_applied
                        and ctype == PROTEIN_CHAIN
                        and int(aid) in query_only_protein_asym_id_set
                    )
                )
                if not skip_homolog_io:
                    if openfold3_alignment_npz_dirs:
                        label_asym_ids = np.unique(
                            atom_array.label_asym_id[aid_mask]
                        )
                        if label_asym_ids.size != 1:
                            raise ValueError(
                                "Each assembly chain must map to exactly one "
                                "label_asym_id"
                            )
                        label_asym_id = str(label_asym_ids[0])
                        alignment_pdb_id = str(bioassembly_dict["pdb_id"])
                        alignment_label_asym_id = label_asym_id
                        representative_id = (
                            f"{alignment_pdb_id}_{alignment_label_asym_id}"
                        )
                        if alignment_representatives is not None:
                            representative_id = alignment_representatives.get(
                                chain_id
                            )
                            if ctype in (PROTEIN_CHAIN, RNA_CHAIN):
                                if not representative_id:
                                    raise ValueError(
                                        "Missing alignment representative for "
                                        f"polymer chain {chain_id!r}"
                                    )
                                try:
                                    (
                                        alignment_pdb_id,
                                        alignment_label_asym_id,
                                    ) = str(representative_id).rsplit("_", 1)
                                except ValueError as error:
                                    raise ValueError(
                                        "Invalid alignment representative ID: "
                                        f"{representative_id!r}"
                                    ) from error
                        openfold3_key = (
                            alignment_pdb_id,
                            alignment_label_asym_id,
                            ctype,
                        )
                    if (
                        openfold3_alignment_npz_dirs
                        and (
                            (ctype == PROTEIN_CHAIN and self.prot_mgr.enabled)
                            or (ctype == RNA_CHAIN and self.rna_mgr.enabled)
                        )
                    ):
                        query_mismatch = False
                        openfold3_cache_key = (
                            (*openfold3_key, seq)
                            if openfold3_runtime_recovery_enabled
                            else openfold3_key
                        )
                        known_recovery_candidate = (
                            openfold3_recovery_candidates.get(
                                (str(representative_id), ctype)
                            )
                            if openfold3_runtime_recovery_enabled
                            else None
                        )
                        if (
                            known_recovery_candidate is not None
                            and known_recovery_candidate.error is None
                            and known_recovery_candidate.canonical_query
                            is not None
                        ):
                            if known_recovery_candidate.canonical_query == seq:
                                up_msas = list(
                                    known_recovery_candidate.unpaired_msas
                                )
                                p_msas = list(
                                    known_recovery_candidate.paired_msas
                                )
                            else:
                                query_mismatch = True
                        else:
                            try:
                                if (
                                    openfold3_cache_key
                                    not in openfold3_chain_msas
                                ):
                                    openfold3_chain_msas[
                                        openfold3_cache_key
                                    ] = (
                                        self._load_openfold3_chain_msas(
                                            pdb_id=openfold3_key[0],
                                            label_asym_id=openfold3_key[1],
                                            query=seq,
                                            chain_type=ctype,
                                            paired_depth_limit=(
                                                max_input_paired_msa_depth
                                            ),
                                            unpaired_depth_limit=(
                                                max_input_unpaired_msa_depth
                                            ),
                                        )
                                    )
                                up_msas, p_msas = openfold3_chain_msas[
                                    openfold3_cache_key
                                ]
                            except ValueError as error:
                                if (
                                    not openfold3_runtime_recovery_enabled
                                    or str(error)
                                    != _OPENFOLD3_QUERY_MISMATCH_MESSAGE
                                ):
                                    raise
                                query_mismatch = True

                        if query_mismatch:
                            openfold3_representative_query_mismatch_count += 1
                            exact_matches: list[
                                _OpenFold3RecoveryCandidate
                            ] = []
                            candidate_errors: dict[str, str] = {}
                            for declared_representative_id in (
                                declared_openfold3_representatives
                            ):
                                candidate = (
                                    get_openfold3_recovery_candidate(
                                        declared_representative_id,
                                        ctype,
                                    )
                                )
                                if candidate.error is not None:
                                    candidate_errors[
                                        declared_representative_id
                                    ] = candidate.error
                                elif candidate.canonical_query == seq:
                                    exact_matches.append(candidate)

                            if len(exact_matches) == 1:
                                match = exact_matches[0]
                                up_msas = list(match.unpaired_msas)
                                p_msas = list(match.paired_msas)
                                openfold3_representative_remap_count += 1
                                openfold3_remapped_chain_ids.append(chain_id)
                                openfold3_representative_remaps.append(
                                    {
                                        "chain_id": chain_id,
                                        "from": str(representative_id),
                                        "to": match.representative_id,
                                    }
                                )
                                logger.warning(
                                    "Recovered OpenFold3 alignment "
                                    "representative query mismatch by exact "
                                    "canonical-query remapping: "
                                    f"dataset={self.dataset_name!r}, "
                                    f"pdb_id={bioassembly_dict['pdb_id']!r}, "
                                    f"chain_id={chain_id!r}, "
                                    "configured_representative="
                                    f"{str(representative_id)!r}, "
                                    "remapped_representative="
                                    f"{match.representative_id!r}"
                                )
                            else:
                                up_msas, p_msas = [], []
                                (
                                    openfold3_representative_query_only_fallback_count
                                ) += 1
                                openfold3_query_only_fallback_chain_ids.append(
                                    chain_id
                                )
                                logger.warning(
                                    "OpenFold3 alignment representative query "
                                    "mismatch has no unique exact declared "
                                    "canonical-query match; using canonical "
                                    "query-only MSA for this chain: "
                                    f"dataset={self.dataset_name!r}, "
                                    f"pdb_id={bioassembly_dict['pdb_id']!r}, "
                                    f"chain_id={chain_id!r}, "
                                    "configured_representative="
                                    f"{str(representative_id)!r}, "
                                    "declared_representatives="
                                    f"{list(declared_openfold3_representatives)!r}, "
                                    "exact_matches="
                                    f"{[candidate.representative_id for candidate in exact_matches]!r}, "
                                    f"candidate_errors={candidate_errors!r}"
                                )
                    elif ctype == RNA_CHAIN:
                        up_msas, _ = self.rna_mgr.fetch_msas(
                            seq,
                            "",
                            ctype,
                            unpaired_depth_limit=(
                                max_input_unpaired_msa_depth
                            ),
                        )
                    elif ctype == PROTEIN_CHAIN:
                        if (
                            direct_protein_unpaired_msa_path is not None
                            and self.prot_mgr.enabled
                        ):
                            if seq not in direct_protein_unpaired_msas:
                                direct_protein_unpaired_msas[seq] = (
                                    RawMsa.from_a3m_file(
                                        seq,
                                        PROTEIN_CHAIN,
                                        direct_protein_unpaired_msa_path,
                                        depth_limit=(
                                            max_input_unpaired_msa_depth
                                        ),
                                        dedup=False,
                                    )
                                )
                            up_msas = [direct_protein_unpaired_msas[seq]]
                        else:
                            up_msas, p_msas = self.prot_mgr.fetch_msas(
                                seq,
                                bioassembly_dict["pdb_id"],
                                ctype,
                                self.prot_p_dbs,
                                self.prot_np_dbs,
                                paired_depth_limit=(
                                    max_input_paired_msa_depth
                                ),
                                unpaired_depth_limit=(
                                    max_input_unpaired_msa_depth
                                ),
                                independent_pairing_non_pairing_budget=(
                                    independent_pairing_non_pairing_budget
                                ),
                            )
                            independent_pairing_copy_in_unpaired = any(
                                unpaired_msa is paired_msa
                                for unpaired_msa in up_msas
                                for paired_msa in p_msas
                            )
                            paired_sources_moved_to_unpaired = (
                                independent_pairing_copy_in_unpaired
                            )
                if (
                    partial_query_only_applied
                    and ctype == PROTEIN_CHAIN
                    and int(aid) not in query_only_protein_asym_id_set
                    and (
                        len(crop_protein_asym_ids)
                        - len(query_only_protein_asym_ids)
                        == 1
                    )
                    and p_msas
                ):
                    # Species pairing requires the same species in at least
                    # two chains. If only one protein chain retains MSA, keep
                    # all of its paired-source hits by treating them as
                    # ordinary unpaired rows rather than silently dropping
                    # them during cross-chain pairing.
                    up_msas = [*p_msas, *up_msas]
                    p_msas = []
                    paired_sources_moved_to_unpaired = True

                meta[aid] = {
                    "entity_id": eid,
                    "chain_id": chain_id,
                    "sequence": seq,
                    "chain_entity_type": ctype,
                    "unpaired_msa_contains_paired_source_rows": (
                        paired_sources_moved_to_unpaired
                    ),
                    "unpaired_msa_uses_independent_pairing_source_budget": (
                        independent_pairing_copy_in_unpaired
                    ),
                    "paired_msa": (
                        RawMsa.merge(
                            p_msas,
                            deduplicate=False,
                            depth_limit=max_input_paired_msa_depth,
                        )
                        if p_msas
                        else ""
                    ),
                    "unpaired_msa": (
                        RawMsa.merge(
                            up_msas,
                            deduplicate=True,
                            # Independently capped paired-source and genuine
                            # non-pairing pools are deduplicated together before
                            # FeatureAssemblyLine applies the final target
                            # budget. The historical partial-query-only move
                            # retains its existing pre-assembly max-size cap.
                            depth_limit=(
                                None
                                if independent_pairing_copy_in_unpaired
                                else (
                                    self.max_msa_size
                                    if paired_sources_moved_to_unpaired
                                    else max_input_unpaired_msa_depth
                                )
                            ),
                        )
                        if up_msas
                        else ""
                    ),
                }

        # Every crop-visible chain has now reused its representative/source
        # payload and owns the bounded RawMsa objects stored in ``meta``.
        # Release source-level copies before numerical assembly; otherwise
        # OpenFold scalar-object payload prefixes and their merged copies stay
        # live together through the highest-memory phase.
        openfold3_chain_msas.clear()
        openfold3_recovery_candidates.clear()
        direct_protein_unpaired_msas.clear()
        up_msas, p_msas = [], []

        # 2. Map coordinates and assemble features
        ca = atom_array[sel_tokens.get_annotation("centre_atom_index")]
        std_idxs = map_to_standard(ca.asym_id_int, ca.res_id, meta)

        resolved_pair_as_unpaired = (
            self.pair_msa_as_unpaired_only
            if pair_msa_as_unpaired_only is None
            else bool(pair_msa_as_unpaired_only)
        )
        resolved_shuffle = (
            self.shuffle_pair_msa_when_unpaired_only
            if shuffle_pair_msa_when_unpaired_only is None
            else bool(shuffle_pair_msa_when_unpaired_only)
        )
        if query_only:
            # Query-only always preserves normal species pairing on retained
            # chains. Keep this invariant even for direct featurizer callers
            # that bypass BaseSingleDataset's occurrence-mode normalization.
            resolved_pair_as_unpaired = False
            resolved_shuffle = False
        virtual_segment_shuffle_enabled = bool(
            getattr(
                self,
                "enable_virtual_segment_msa_shuffle",
                False,
            )
        )
        virtual_segment_residue_gap = _strict_positive_int(
            getattr(self, "virtual_segment_residue_gap", 24),
            "virtual_segment_residue_gap",
        )
        virtual_segment_min_length = _strict_positive_int(
            getattr(self, "virtual_segment_min_length", 9),
            "virtual_segment_min_length",
        )
        variant_sampling_enabled = bool(
            getattr(
                self,
                "enable_multichain_msa_shuffle_variant_sampling",
                False,
            )
        )
        segment_probability = float(
            getattr(
                self,
                "multichain_msa_shuffle_segment_probability",
                0.5,
            )
        )
        msa_shuffle_requested_variant = MSA_SHUFFLE_VARIANT_NONE
        msa_shuffle_variant_eligible = False
        msa_shuffle_groups: Dict[int, tuple[np.ndarray, ...]] = {}
        msa_shuffle_mode_active = bool(
            virtual_segment_shuffle_enabled
            and resolved_pair_as_unpaired
            and resolved_shuffle
        )
        if msa_shuffle_mode_active:
            protein_sequence_lengths = {
                int(aid): len(info["sequence"])
                for aid, info in meta.items()
                if info["chain_entity_type"] == PROTEIN_CHAIN
            }
            (
                msa_shuffle_requested_variant,
                msa_shuffle_variant_eligible,
            ) = select_multichain_msa_shuffle_variant(
                protein_asym_ids=crop_protein_asym_ids,
                sampling_enabled=variant_sampling_enabled,
                segment_probability=segment_probability,
                variant_seed=msa_shuffle_variant_seed,
            )
            if (
                msa_shuffle_requested_variant
                == MSA_SHUFFLE_VARIANT_SEGMENT
            ):
                msa_shuffle_groups = build_virtual_msa_segment_groups(
                    asym_ids=ca.asym_id_int,
                    res_ids=ca.res_id,
                    protein_sequence_lengths=protein_sequence_lengths,
                    residue_gap=virtual_segment_residue_gap,
                    min_segment_length=virtual_segment_min_length,
                )
            elif (
                msa_shuffle_requested_variant
                == MSA_SHUFFLE_VARIANT_WHOLE_PHYSICAL_CHAIN
            ):
                msa_shuffle_groups = (
                    build_whole_physical_chain_msa_groups(
                        asym_ids=ca.asym_id_int,
                        res_ids=ca.res_id,
                        protein_sequence_lengths=protein_sequence_lengths,
                    )
                )
        assembly_line_kwargs = {
            "max_msa_size": self.max_msa_size,
            "pair_msa_as_unpaired_only": resolved_pair_as_unpaired,
            "shuffle_pair_msa_when_unpaired_only": resolved_shuffle,
        }
        if max_input_paired_msa_depth is not None:
            assembly_line_kwargs[
                "max_input_paired_msa_depth"
            ] = max_input_paired_msa_depth
        if max_input_unpaired_msa_depth is not None:
            assembly_line_kwargs[
                "max_input_unpaired_msa_depth"
            ] = max_input_unpaired_msa_depth
        if virtual_segment_shuffle_enabled:
            assembly_line_kwargs[
                "enable_virtual_segment_msa_shuffle"
            ] = True
        assembly_line = FeatureAssemblyLine(
            **assembly_line_kwargs,
        )
        res = assembly_line.assemble(
            meta,
            std_idxs,
            virtual_segment_groups=msa_shuffle_groups,
            virtual_segment_shuffle_seed=virtual_segment_shuffle_seed,
        ).to_dict()
        msa_shuffle_variant_applied = bool(
            assembly_line.last_virtual_segment_shuffle_applied
        )
        segment_variant_requested = (
            msa_shuffle_requested_variant
            == MSA_SHUFFLE_VARIANT_SEGMENT
        )
        self._profile = {
            "max_input_paired_msa_depth": (
                max_input_paired_msa_depth
            ),
            "max_input_unpaired_msa_depth": (
                max_input_unpaired_msa_depth
            ),
            "max_msa_size": self.max_msa_size,
            "independent_pairing_non_pairing_budget": (
                independent_pairing_non_pairing_budget
            ),
            "pairing_non_pairing_budget_policy": (
                PAIRING_NON_PAIRING_BUDGET_POLICY_VERSION
                if independent_pairing_non_pairing_budget
                else "shared_unpaired_source_budget_v1"
            ),
            "query_only_requested": bool(query_only),
            "query_only_global_applied": global_query_only_applied,
            "query_only_partial_applied": partial_query_only_applied,
            "query_only_policy": (
                PARTIAL_QUERY_ONLY_MSA_POLICY_VERSION
                if partial_query_only_enabled
                else "global_query_only_v1"
            ),
            "query_only_crop_protein_chain_count": len(
                crop_protein_asym_ids
            ),
            "query_only_selected_protein_chain_count": (
                len(query_only_protein_asym_ids) if query_only else 0
            ),
            "query_only_retained_protein_chain_count": (
                len(crop_protein_asym_ids)
                - len(query_only_protein_asym_ids)
                if partial_query_only_applied
                else 0
            ),
            "query_only_selected_protein_asym_ids": (
                list(query_only_protein_asym_ids) if query_only else []
            ),
            "partial_query_only_max_chains": (
                partial_query_only_max_chains
            ),
            "openfold3_runtime_recovery_enabled": (
                openfold3_runtime_recovery_enabled
            ),
            "openfold3_representative_query_mismatch_count": (
                openfold3_representative_query_mismatch_count
            ),
            "openfold3_representative_remap_count": (
                openfold3_representative_remap_count
            ),
            "openfold3_representative_query_only_fallback_count": (
                openfold3_representative_query_only_fallback_count
            ),
            "openfold3_explicit_query_only_chain_count": (
                openfold3_explicit_query_only_chain_count
            ),
            "openfold3_explicit_query_only_chain_ids": sorted(
                openfold3_explicit_query_only_chain_ids
            ),
            "openfold3_recovery_candidate_load_count": (
                openfold3_recovery_candidate_load_count
            ),
            "openfold3_recovery_candidate_cache_hit_count": (
                openfold3_recovery_candidate_cache_hit_count
            ),
            "openfold3_remapped_chain_ids": (
                openfold3_remapped_chain_ids
            ),
            "openfold3_representative_remaps": (
                openfold3_representative_remaps
            ),
            "openfold3_query_only_fallback_chain_ids": (
                openfold3_query_only_fallback_chain_ids
            ),
            "openfold3_recovery_candidate_errors": (
                openfold3_recovery_candidate_errors
            ),
            "virtual_segment_count": (
                assembly_line.last_virtual_segment_count
                if segment_variant_requested
                else 0
            ),
            "virtual_segment_shuffle_applied": (
                msa_shuffle_variant_applied
                and segment_variant_requested
            ),
            "virtual_segment_residue_gap": (
                virtual_segment_residue_gap
            ),
            "virtual_segment_min_length": (
                virtual_segment_min_length
            ),
            "virtual_segment_shuffle_policy": (
                VIRTUAL_SEGMENT_MSA_SHUFFLE_POLICY_VERSION
                if virtual_segment_shuffle_enabled
                else None
            ),
            "msa_shuffle_variant_policy": (
                MULTICHAIN_MSA_SHUFFLE_VARIANT_POLICY_VERSION
                if variant_sampling_enabled
                else None
            ),
            "msa_shuffle_variant_sampling_enabled": (
                variant_sampling_enabled
            ),
            "msa_shuffle_segment_probability": segment_probability,
            "msa_shuffle_crop_protein_chain_count": len(
                crop_protein_asym_ids
            ),
            "msa_shuffle_requested_variant": (
                msa_shuffle_requested_variant
            ),
            "msa_shuffle_requested_variant_code": (
                MSA_SHUFFLE_VARIANT_CODES[
                    msa_shuffle_requested_variant
                ]
            ),
            "msa_shuffle_variant_eligible": (
                msa_shuffle_variant_eligible
            ),
            "msa_shuffle_variant_applied": (
                msa_shuffle_variant_applied
            ),
            "msa_shuffle_group_count": (
                assembly_line.last_virtual_segment_count
            ),
            "msa_shuffle_whole_physical_chain_applied": bool(
                msa_shuffle_variant_applied
                and msa_shuffle_requested_variant
                == MSA_SHUFFLE_VARIANT_WHOLE_PHYSICAL_CHAIN
            ),
        }
        if partial_query_only_applied:
            res = apply_partial_query_only_msa_features(
                res,
                token_asym_ids=ca.asym_id_int,
                query_only_asym_ids=query_only_protein_asym_ids,
            )
        elif global_query_only_applied:
            res = make_query_only_msa_features(res)
        keep = {
            "msa",
            "has_deletion",
            "deletion_value",
            "deletion_mean",
            "profile",
            "prot_pair_num_alignments",
            "prot_unpair_num_alignments",
            "rna_pair_num_alignments",
            "rna_unpair_num_alignments",
        }
        return {k: v for k, v in res.items() if k in keep}

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convenience method to call make_msa_features."""
        return self.make_msa_features(*args, **kwargs)


@dataclasses.dataclass(frozen=True)
class MSAFeat:
    """Container for finalized numerical MSA features."""

    rows: np.ndarray
    mask: np.ndarray
    deletion_matrix: np.ndarray
    profile: np.ndarray
    deletion_mean: np.ndarray
    prot_unpaired_num_alignments: np.ndarray
    prot_paired_num_alignments: np.ndarray
    rna_unpaired_num_alignments: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Converts the MSA object into a standard Protenix data dictionary."""
        return {
            "msa": self.rows,
            "msa_mask": self.mask,
            "deletion_matrix": self.deletion_matrix,
            "deletion_value": (np.arctan(self.deletion_matrix / 3.0) * (2.0 / np.pi)),
            "has_deletion": np.clip(self.deletion_matrix, 0.0, 1.0),
            "profile": self.profile,
            "deletion_mean": self.deletion_mean,
            "prot_unpaired_num_alignments": self.prot_unpaired_num_alignments,
            "prot_paired_num_alignments": self.prot_paired_num_alignments,
            "rna_unpaired_num_alignments": self.rna_unpaired_num_alignments,
            "rna_pair_num_alignments": np.asarray(1, dtype=np.int32),
            "prot_pair_num_alignments": self.prot_paired_num_alignments,
            "prot_unpair_num_alignments": self.prot_unpaired_num_alignments,
            "rna_unpair_num_alignments": self.rna_unpaired_num_alignments,
        }


def ensure_ends_with_newline(s: Optional[str]) -> Optional[str]:
    """
    Ensure the given string ends with a newline character.

    If the string is non-empty and does not already end with '\\n',
    append '\\n'. Empty strings are returned unchanged.

    Args:
        s (str): Input string.

    Returns:
        str: The input string guaranteed to end with '\\n' when non-empty.
    """
    if not s:
        return s
    if not s.endswith("\n"):
        s += "\n"
    return s


class InferenceMSAFeaturizer:
    """Specialized featurizer for inference scenarios, leveraging the unified assembly line."""

    @staticmethod
    def make_msa_feature(
        bioassembly: Sequence[Dict[str, Any]],
        atom_array: AtomArray,
        msa_pair_as_unpair: bool = False,
        use_rna_msa: bool = True,
    ) -> Dict[str, Any]:
        """
        Prepares MSA features during inference from bioassembly structure.

        Args:
            bioassembly: List of entities in the biological assembly.
            atom_array: Structural data array.
            msa_pair_as_unpair: Whether to treat paired MSA as unpaired.
            use_rna_msa: Whether to use MSA for RNA chains.

        Returns:
            Dictionary of processed MSA features.
        """
        meta, curr_aid = {}, 0
        for eid, info in enumerate(bioassembly):
            seq, count, ctype, u_a3m, p_a3m = "", 0, LIGAND_CHAIN_TYPES, None, None
            if "proteinChain" in info:
                c = info["proteinChain"]
                seq, count, ctype, u_a3m, p_a3m = (
                    c["sequence"],
                    c["count"],
                    PROTEIN_CHAIN,
                    c.get("unpairedMsa"),
                    c.get("pairedMsa"),
                )
                if u_a3m is None and c.get("unpairedMsaPath"):
                    with open(c["unpairedMsaPath"]) as f:
                        u_a3m = f.read()
                if p_a3m is None and c.get("pairedMsaPath"):
                    with open(c["pairedMsaPath"]) as f:
                        p_a3m = f.read()
                if u_a3m is None and (p_a3m is None):
                    if c.get("msa"):
                        msa_dir = c["msa"].get("precomputed_msa_dir")
                        if msa_dir and opexists(msa_dir):
                            logger.warning(
                                "Use the old msa json format, change to pairedMsaPath/unpairedMsaPath field for future use."
                            )
                            if opexists(opjoin(msa_dir, "pairing.a3m")):
                                with open(opjoin(msa_dir, "pairing.a3m")) as f:
                                    p_a3m = f.read()
                            if opexists(opjoin(msa_dir, "non_pairing.a3m")):
                                with open(opjoin(msa_dir, "non_pairing.a3m")) as f:
                                    u_a3m = f.read()

            elif "rnaSequence" in info:
                c = info["rnaSequence"]
                seq, count, ctype = c["sequence"], c["count"], RNA_CHAIN
                if use_rna_msa:
                    u_a3m = c.get("unpairedMsa")
                    if u_a3m is None and c.get("unpairedMsaPath"):
                        with open(c["unpairedMsaPath"]) as f:
                            u_a3m = f.read()
            elif "dnaSequence" in info:
                c = info["dnaSequence"]
                seq, count, ctype = c["sequence"], c["count"], DNA_CHAIN
            elif "ligand" in info:
                count, ctype, seq = (
                    info["ligand"]["count"],
                    LIGAND_CHAIN_TYPES,
                    "X" * (atom_array.asym_id_int == curr_aid).sum(),
                )

            p_a3m = ensure_ends_with_newline(p_a3m)
            u_a3m = ensure_ends_with_newline(u_a3m)

            if msa_pair_as_unpair and p_a3m:
                u_a3m = RawMsa.from_a3m(
                    seq, ctype, p_a3m + (u_a3m or ""), dedup=True
                ).to_a3m()

            for c_idx in range(count):
                aid = curr_aid + c_idx
                meta[aid] = {
                    "entity_id": eid,
                    "chain_id": atom_array.chain_id[atom_array.asym_id_int == aid][0],
                    "sequence": seq,
                    "paired_msa": p_a3m or "",
                    "unpaired_msa": u_a3m or "",
                    "chain_entity_type": ctype,
                }
            curr_aid += count

        ca = atom_array[atom_array.centre_atom_mask.astype(bool)]
        std_idxs = map_to_standard(ca.asym_id_int, ca.res_id, meta)
        return FeatureAssemblyLine().assemble(meta, std_idxs).to_dict()
