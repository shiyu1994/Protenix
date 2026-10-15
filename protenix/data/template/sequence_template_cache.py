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

"""Offline, sequence-keyed protein template feature cache construction.

The cache policy implemented here is deliberately independent of the query
target's release date.  A query sequence therefore has one globally fixed
candidate pool:

1. Apply the historical non-date prefilters and de-duplicate hit sequences.
2. Reject alignments whose non-gap sequence identity is greater than 0.90.
3. Sort by ``sum_probs`` (the source order is the stable tie-breaker).
4. Keep the quality top 60 and deterministically shuffle that set.
5. Process candidates in shuffled order until 20 valid, distinct templates
   have been collected.  Failed candidates are replaced only from the same
   top-60 set.

One gzip-compressed pickle is written atomically for each query sequence.  Its
relative directory mirrors the existing ``seq_to_pdb_index.json`` mapping.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import os
import pickle
import random
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from protenix.data.template.template_parser import (
    HHRParser,
    HmmsearchA3MParser,
    TemplateHit,
    get_pdb_id_and_chain,
)
from protenix.data.template.template_utils import TemplateHitProcessor
from protenix.data.tools.common import parse_fasta


SCHEMA_VERSION = 1
POLICY_VERSION = "top60_uniform20_identity_le_0.90_v1"
DEFAULT_CACHE_FILENAME = "template_features.pkl.gz"
DEFAULT_HIT_FILENAME = "hmmsearch.a3m"
TOP_K_QUALITY = 60
MAX_VALID_TEMPLATES = 20
MAX_ALIGNMENT_IDENTITY = 0.90

# ``TemplateHitProcessor`` currently requires a datetime even when callers do
# not want a date restriction.  This value admits every valid mmCIF date,
# including its internal 9999-12-31 fallback.
_NO_DATE_CUTOFF = datetime.max


@dataclasses.dataclass(frozen=True)
class SequenceTemplateCacheConfig:
    """Configuration shared by all sequence cache workers."""

    source_root: str
    cache_root: str
    mmcif_dir: str
    kalign_binary_path: str
    obsolete_pdbs_path: str = ""
    parsed_mmcif_cache_dir: Optional[str] = None
    hit_filename: str = DEFAULT_HIT_FILENAME
    cache_filename: str = DEFAULT_CACHE_FILENAME
    top_k_quality: int = TOP_K_QUALITY
    max_valid_templates: int = MAX_VALID_TEMPLATES
    max_alignment_identity: float = MAX_ALIGNMENT_IDENTITY
    gzip_compresslevel: int = 1

    def __post_init__(self) -> None:
        if self.top_k_quality <= 0:
            raise ValueError("top_k_quality must be positive")
        if self.max_valid_templates <= 0:
            raise ValueError("max_valid_templates must be positive")
        if self.max_valid_templates > self.top_k_quality:
            raise ValueError("max_valid_templates cannot exceed top_k_quality")
        if not 0.0 <= self.max_alignment_identity <= 1.0:
            raise ValueError("max_alignment_identity must be in [0, 1]")
        if not 0 <= self.gzip_compresslevel <= 9:
            raise ValueError("gzip_compresslevel must be in [0, 9]")


@dataclasses.dataclass(frozen=True)
class ParsedHit:
    """A parsed search hit and its search-alignment sequence identity."""

    hit: TemplateHit
    alignment_identity: float


def sequence_sha256(sequence: str) -> str:
    """Return the canonical hexadecimal digest used by cache files and shards."""

    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def deterministic_seed(sequence: str) -> int:
    """Derive the fixed policy RNG seed requested by the cache contract."""

    digest = hashlib.sha256(
        (sequence + POLICY_VERSION).encode("utf-8")
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def sequence_shard(sequence: str, num_shards: int) -> int:
    """Assign a sequence to a stable hash shard."""

    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    digest = hashlib.sha256(sequence.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % num_shards


def normalize_relative_directory(relative_directory: str) -> str:
    """Validate a mapping value before using it beneath source/cache roots."""

    path = Path(str(relative_directory))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "sequence mapping directories must be relative and may not contain '..': "
            f"{relative_directory!r}"
        )
    normalized = str(path)
    if normalized in ("", "."):
        raise ValueError("sequence mapping directory cannot be empty")
    return normalized


def cache_path_for_relative_directory(
    cache_root: str,
    relative_directory: str,
    cache_filename: str = DEFAULT_CACHE_FILENAME,
) -> Path:
    """Resolve one sequence's cache path."""

    relative_directory = normalize_relative_directory(relative_directory)
    return Path(cache_root) / relative_directory / cache_filename


def _identity_from_aligned_strings(
    aligned_query: str,
    aligned_template: str,
) -> float:
    """Compute identity over columns where both aligned residues are non-gap."""

    if len(aligned_query) != len(aligned_template):
        raise ValueError(
            "aligned query/template lengths differ: "
            f"{len(aligned_query)} != {len(aligned_template)}"
        )

    matches = 0
    nongap_columns = 0
    for query_residue, template_residue in zip(aligned_query, aligned_template):
        if query_residue == "-" or template_residue == "-":
            continue
        nongap_columns += 1
        matches += query_residue.upper() == template_residue.upper()

    if nongap_columns == 0:
        raise ValueError("alignment has no columns with two non-gap residues")
    return matches / nongap_columns


def aligned_non_gap_identity(hit: TemplateHit) -> float:
    """Compute standard aligned non-gap identity for HHR-like hits.

    For hmmsearch A3M input, use :func:`parse_hits_with_identities`, which
    removes lowercase insertion residues before calling the same primitive.
    """

    return _identity_from_aligned_strings(hit.query, hit.hit_sequence)


def _parse_a3m_hits_with_identities(
    *,
    query_sequence: str,
    content: str,
) -> List[ParsedHit]:
    """Parse hmmsearch A3M while retaining lowercase insertion semantics."""

    hits = HmmsearchA3MParser.parse(
        query_seq=query_sequence,
        a3m_str=content,
        skip_first=False,
    )
    raw_sequences, raw_descriptions = parse_fasta(content)
    protein_aligned_sequences = [
        sequence
        for sequence, description in zip(raw_sequences, raw_descriptions)
        if "mol:protein" in description
    ]

    if len(hits) != len(protein_aligned_sequences):
        raise ValueError(
            "hmmsearch parser/raw A3M protein-hit counts differ: "
            f"{len(hits)} != {len(protein_aligned_sequences)}"
        )

    parsed_hits = []
    for hit, raw_aligned_template in zip(hits, protein_aligned_sequences):
        # Lowercase A3M characters are insertions relative to the query and do
        # not occupy an alignment column.
        aligned_template = "".join(
            residue
            for residue in raw_aligned_template
            if not residue.islower()
        )
        identity = _identity_from_aligned_strings(
            query_sequence,
            aligned_template,
        )
        parsed_hits.append(ParsedHit(hit=hit, alignment_identity=identity))
    return parsed_hits


def parse_hits_with_identities(
    *,
    query_sequence: str,
    content: str,
    filename: str,
) -> List[ParsedHit]:
    """Parse an A3M/HHR search artifact and calculate hit identities."""

    lower_filename = filename.lower()
    if lower_filename.endswith(".hhr"):
        return [
            ParsedHit(hit=hit, alignment_identity=aligned_non_gap_identity(hit))
            for hit in HHRParser.parse(hhr_string=content)
        ]
    if lower_filename.endswith(".a3m"):
        return _parse_a3m_hits_with_identities(
            query_sequence=query_sequence,
            content=content,
        )
    raise ValueError(f"unsupported template hit file: {filename!r}")


def _passes_non_date_prefilter(
    query_sequence: str,
    hit: TemplateHit,
) -> Tuple[bool, Optional[str]]:
    """Mirror the historical fast prefilter while deliberately omitting date."""

    try:
        get_pdb_id_and_chain(hit)
    except ValueError as exc:
        return False, str(exc)

    if not query_sequence:
        return False, "query sequence is empty"

    align_ratio = hit.aligned_cols / len(query_sequence)
    if align_ratio <= 0.1:
        return False, f"align ratio {align_ratio:.6f} <= 0.1"

    template_sequence = hit.hit_sequence.replace("-", "")
    length_ratio = len(template_sequence) / len(query_sequence)
    if template_sequence in query_sequence and length_ratio > 0.95:
        return False, "hit is a >95% length subsequence duplicate"
    if len(template_sequence) < 10:
        return False, "template sequence is shorter than 10 residues"
    return True, None


def select_top_quality_hits(
    *,
    query_sequence: str,
    parsed_hits: Sequence[ParsedHit],
    top_k_quality: int = TOP_K_QUALITY,
    max_alignment_identity: float = MAX_ALIGNMENT_IDENTITY,
    max_rejection_details: int = 64,
) -> Tuple[List[ParsedHit], Dict[str, int], List[Dict[str, Any]]]:
    """Prefilter, de-duplicate, identity-filter, and quality-rank search hits."""

    counts = {
        "raw_hit_count": len(parsed_hits),
        "prefiltered_hit_count": 0,
        "prefilter_rejected_count": 0,
        "deduplicated_hit_count": 0,
        "duplicate_rejected_count": 0,
        "identity_eligible_hit_count": 0,
        "identity_rejected_count": 0,
        "top_quality_count": 0,
    }
    rejections: List[Dict[str, Any]] = []

    def record_rejection(detail: Dict[str, Any]) -> None:
        if len(rejections) < max_rejection_details:
            rejections.append(detail)

    prefiltered: List[ParsedHit] = []
    for parsed_hit in parsed_hits:
        valid, reason = _passes_non_date_prefilter(
            query_sequence,
            parsed_hit.hit,
        )
        if valid:
            prefiltered.append(parsed_hit)
        else:
            counts["prefilter_rejected_count"] += 1
            record_rejection(
                {
                    "name": parsed_hit.hit.name,
                    "stage": "prefilter",
                    "reason": reason,
                }
            )
    counts["prefiltered_hit_count"] = len(prefiltered)

    # The source search result is already quality ordered.  Keeping the first
    # instance here matches the historical de-duplication behavior.
    deduplicated: List[ParsedHit] = []
    seen_hit_sequences = set()
    for parsed_hit in prefiltered:
        hit_sequence = parsed_hit.hit.hit_sequence.replace("-", "")
        if hit_sequence in seen_hit_sequences:
            counts["duplicate_rejected_count"] += 1
            continue
        seen_hit_sequences.add(hit_sequence)
        deduplicated.append(parsed_hit)
    counts["deduplicated_hit_count"] = len(deduplicated)

    identity_eligible = []
    for parsed_hit in deduplicated:
        if parsed_hit.alignment_identity > max_alignment_identity:
            counts["identity_rejected_count"] += 1
            record_rejection(
                {
                    "name": parsed_hit.hit.name,
                    "stage": "identity",
                    "reason": (
                        f"aligned non-gap identity "
                        f"{parsed_hit.alignment_identity:.8f} > "
                        f"{max_alignment_identity:.8f}"
                    ),
                    "alignment_identity": parsed_hit.alignment_identity,
                }
            )
            continue
        identity_eligible.append(parsed_hit)
    counts["identity_eligible_hit_count"] = len(identity_eligible)

    # Python's sort is stable, so source order is the deterministic tie-breaker
    # for hmmsearch A3M hits whose parser does not expose sum_probs.
    identity_eligible.sort(
        key=lambda item: (
            item.hit.sum_probs if item.hit.sum_probs is not None else 0.0
        ),
        reverse=True,
    )
    selected = identity_eligible[:top_k_quality]
    counts["top_quality_count"] = len(selected)
    return selected, counts, rejections


def deterministic_permutation(
    sequence: str,
    hits: Sequence[ParsedHit],
) -> List[ParsedHit]:
    """Return the SHA256-seeded fixed random ordering of the top-quality set."""

    indices = list(range(len(hits)))
    random.Random(deterministic_seed(sequence)).shuffle(indices)
    return [hits[index] for index in indices]


def _load_obsolete_mapping(path: str) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict):
        raise TypeError("obsolete PDB mapping must be a JSON object")
    return {str(key).lower(): str(value).lower() for key, value in mapping.items()}


def _atomic_pickle_gzip_dump(
    payload: Mapping[str, Any],
    output_path: Path,
    *,
    compresslevel: int,
) -> None:
    """Write a gzip pickle in the destination directory then atomically rename."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    try:
        with os.fdopen(file_descriptor, "wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=compresslevel,
                fileobj=raw_handle,
                mtime=0,
            ) as gzip_handle:
                pickle.dump(
                    dict(payload),
                    gzip_handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def load_sequence_template_cache(path: str | Path) -> Dict[str, Any]:
    """Load one cache artifact (primarily for validation and tooling)."""

    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"template cache payload must be a dict: {path}")
    return payload


class SequenceTemplateCacheBuilder:
    """Build the globally fixed offline template pool for query sequences."""

    def __init__(
        self,
        config: SequenceTemplateCacheConfig,
        *,
        hit_processor: Optional[TemplateHitProcessor] = None,
    ) -> None:
        self.config = config
        self.obsolete_pdbs = _load_obsolete_mapping(config.obsolete_pdbs_path)
        self.hit_processor = hit_processor or TemplateHitProcessor(
            mmcif_dir=config.mmcif_dir,
            template_cache_dir=config.parsed_mmcif_cache_dir,
            kalign_binary_path=config.kalign_binary_path,
            _zero_center_positions=True,
            fetch_remote=False,
        )

    def source_path(self, relative_directory: str) -> Path:
        relative_directory = normalize_relative_directory(relative_directory)
        return (
            Path(self.config.source_root)
            / relative_directory
            / self.config.hit_filename
        )

    def output_path(self, relative_directory: str) -> Path:
        return cache_path_for_relative_directory(
            self.config.cache_root,
            relative_directory,
            self.config.cache_filename,
        )

    def build(
        self,
        *,
        sequence: str,
        relative_directory: str,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """Build one sequence cache and return a compact manifest record."""

        started_at = time.time()
        relative_directory = normalize_relative_directory(relative_directory)
        output_path = self.output_path(relative_directory)
        digest = sequence_sha256(sequence)

        if resume and output_path.is_file():
            try:
                existing_payload = load_sequence_template_cache(output_path)
                validate_cache_payload(
                    existing_payload,
                    expected_sequence=sequence,
                )
                if (
                    existing_payload.get("source_relative_dir")
                    != relative_directory
                ):
                    raise ValueError(
                        "existing cache source_relative_dir does not match mapping"
                    )
                return {
                    "status": "skipped_existing",
                    "sequence_sha256": digest,
                    "source_relative_dir": relative_directory,
                    "output_path": str(output_path),
                    "existing_cache_status": existing_payload["status"],
                    "elapsed_seconds": time.time() - started_at,
                }
            except (EOFError, OSError, pickle.PickleError, TypeError, ValueError):
                # Atomic writes make partial artifacts unlikely, but an old or
                # corrupt cache must never be silently accepted by --resume.
                # Rebuild it in place using the current policy.
                pass

        source_path = self.source_path(relative_directory)
        source_missing = not source_path.exists()
        if source_missing:
            # A missing hmmsearch result is a legitimate N=0 query, matching
            # the online template path's dummy-template behavior.
            content = ""
        else:
            with open(source_path, "r", encoding="utf-8") as handle:
                content = handle.read()
        source_empty = not content.strip()

        parsed_hits = parse_hits_with_identities(
            query_sequence=sequence,
            content=content,
            filename=self.config.hit_filename,
        )
        top_hits, counts, rejections = select_top_quality_hits(
            query_sequence=sequence,
            parsed_hits=parsed_hits,
            top_k_quality=self.config.top_k_quality,
            max_alignment_identity=self.config.max_alignment_identity,
        )
        shuffled_hits = deterministic_permutation(sequence, top_hits)

        candidates: List[Dict[str, Any]] = []
        process_failures: List[Dict[str, Any]] = []
        already_seen_template_sequences = set()
        rank_by_hit_identity = {
            id(parsed_hit): rank
            for rank, parsed_hit in enumerate(top_hits, start=1)
        }

        attempted_count = 0
        postprocess_identity_rejected_count = 0
        for sample_order, parsed_hit in enumerate(shuffled_hits, start=1):
            if len(candidates) >= self.config.max_valid_templates:
                break
            attempted_count += 1
            hit = parsed_hit.hit
            result, timing = self.hit_processor.process(
                sequence,
                hit,
                _NO_DATE_CUTOFF,
                {},
                self.obsolete_pdbs,
                False,
            )

            if result.features is None:
                process_failures.append(
                    {
                        "name": hit.name,
                        "sample_order": sample_order,
                        "rank_in_top60": rank_by_hit_identity[id(parsed_hit)],
                        "alignment_identity": parsed_hit.alignment_identity,
                        "error": result.error,
                        "warning": result.warning,
                        "timing_seconds": dict(timing),
                    }
                )
                continue

            template_sequence = result.features["template_sequence"]
            if isinstance(template_sequence, np.ndarray):
                template_sequence = template_sequence.item()
            if isinstance(template_sequence, str):
                template_sequence_string = template_sequence
                template_sequence_key = template_sequence.encode("utf-8")
            else:
                template_sequence_key = bytes(template_sequence)
                template_sequence_string = template_sequence_key.decode("utf-8")

            try:
                postprocess_identity = _identity_from_aligned_strings(
                    sequence,
                    template_sequence_string,
                )
            except ValueError as exc:
                process_failures.append(
                    {
                        "name": hit.name,
                        "sample_order": sample_order,
                        "rank_in_top60": rank_by_hit_identity[id(parsed_hit)],
                        "search_alignment_identity": (
                            parsed_hit.alignment_identity
                        ),
                        "error": (
                            "cannot compute post-Kalign sequence identity: "
                            f"{exc}"
                        ),
                        "warning": result.warning,
                        "timing_seconds": dict(timing),
                    }
                )
                continue
            if postprocess_identity > self.config.max_alignment_identity:
                postprocess_identity_rejected_count += 1
                process_failures.append(
                    {
                        "name": hit.name,
                        "sample_order": sample_order,
                        "rank_in_top60": rank_by_hit_identity[id(parsed_hit)],
                        "search_alignment_identity": (
                            parsed_hit.alignment_identity
                        ),
                        "alignment_identity": postprocess_identity,
                        "error": (
                            "post-Kalign aligned non-gap identity "
                            f"{postprocess_identity:.8f} > "
                            f"{self.config.max_alignment_identity:.8f}"
                        ),
                        "warning": result.warning,
                        "timing_seconds": dict(timing),
                    }
                )
                continue

            if template_sequence_key in already_seen_template_sequences:
                process_failures.append(
                    {
                        "name": hit.name,
                        "sample_order": sample_order,
                        "rank_in_top60": rank_by_hit_identity[id(parsed_hit)],
                        "search_alignment_identity": (
                            parsed_hit.alignment_identity
                        ),
                        "alignment_identity": postprocess_identity,
                        "error": "duplicate processed template_sequence",
                        "warning": result.warning,
                        "timing_seconds": dict(timing),
                    }
                )
                continue
            already_seen_template_sequences.add(template_sequence_key)

            processed_hit = result.hit or hit
            pdb_id, chain_id = get_pdb_id_and_chain(processed_hit)
            candidates.append(
                {
                    "rank_in_top60": rank_by_hit_identity[id(parsed_hit)],
                    "sample_order": sample_order,
                    "name": hit.name,
                    "pdb_id": self.obsolete_pdbs.get(pdb_id, pdb_id),
                    "chain_id": chain_id,
                    "sum_probs": float(
                        hit.sum_probs if hit.sum_probs is not None else 0.0
                    ),
                    "search_alignment_identity": float(
                        parsed_hit.alignment_identity
                    ),
                    "alignment_identity": float(postprocess_identity),
                    "features": dict(result.features),
                }
            )

        counts.update(
            {
                "attempted_top60_count": attempted_count,
                "process_failed_count": len(process_failures),
                "postprocess_identity_rejected_count": (
                    postprocess_identity_rejected_count
                ),
                "valid_cached_count": len(candidates),
            }
        )
        status = "complete" if candidates else "complete_empty"
        completed_at = time.time()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "policy_version": POLICY_VERSION,
            "status": status,
            "sequence": sequence,
            "sequence_sha256": digest,
            "source_relative_dir": relative_directory,
            "source_hit_path": str(source_path),
            "source": {
                "missing": source_missing,
                "empty": source_empty,
            },
            "counts": counts,
            "candidates": candidates,
            "prefilter_rejections": rejections,
            "process_failures": process_failures,
            "build": {
                "created_at_unix": completed_at,
                "elapsed_seconds": completed_at - started_at,
                "top_k_quality": self.config.top_k_quality,
                "max_valid_templates": self.config.max_valid_templates,
                "max_alignment_identity": self.config.max_alignment_identity,
                "date_filter_enabled": False,
                "deterministic_seed_sha256": hashlib.sha256(
                    (sequence + POLICY_VERSION).encode("utf-8")
                ).hexdigest(),
            },
        }
        _atomic_pickle_gzip_dump(
            payload,
            output_path,
            compresslevel=self.config.gzip_compresslevel,
        )

        return {
            "status": status,
            "sequence_sha256": digest,
            "source_relative_dir": relative_directory,
            "output_path": str(output_path),
            "source_missing": source_missing,
            "source_empty": source_empty,
            "counts": counts,
            "elapsed_seconds": time.time() - started_at,
        }


def validate_cache_payload(
    payload: Mapping[str, Any],
    *,
    expected_sequence: Optional[str] = None,
) -> None:
    """Raise a descriptive error when a cache artifact violates the schema."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: {payload.get('schema_version')!r} "
            f"!= {SCHEMA_VERSION!r}"
        )
    if payload.get("policy_version") != POLICY_VERSION:
        raise ValueError(
            f"policy_version mismatch: {payload.get('policy_version')!r} "
            f"!= {POLICY_VERSION!r}"
        )
    if payload.get("status") not in {"complete", "complete_empty"}:
        raise ValueError(f"invalid cache status: {payload.get('status')!r}")
    if not isinstance(payload.get("candidates"), list):
        raise TypeError("cache candidates must be a list")
    if len(payload["candidates"]) > MAX_VALID_TEMPLATES:
        raise ValueError("cache contains more than 20 candidates")
    for candidate in payload["candidates"]:
        identity = candidate.get("alignment_identity")
        if identity is None or not 0.0 <= float(identity) <= MAX_ALIGNMENT_IDENTITY:
            raise ValueError(
                "cache candidate post-Kalign alignment_identity must be in "
                f"[0, {MAX_ALIGNMENT_IDENTITY}]"
            )
    if expected_sequence is not None:
        if payload.get("sequence") != expected_sequence:
            raise ValueError("cache sequence does not match expected sequence")
        if payload.get("sequence_sha256") != sequence_sha256(expected_sequence):
            raise ValueError("cache sequence_sha256 does not match expected sequence")
