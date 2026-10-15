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

"""Locked, content-addressed template caches for PXM checkpoint evaluation.

This module deliberately does not share the training template-cache policy.
It materializes the exact deterministic template selection used by the current
evaluation online path: a fixed 2021-09-30 cutoff, no top-k shuffling, and at
most four successfully processed templates.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import os
import pickle
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from protenix.data.constants import PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37
from protenix.data.template.template_parser import (
    HHRParser,
    HmmsearchA3MParser,
    TemplateHit,
    get_pdb_id_and_chain,
)
from protenix.data.template.template_utils import TemplateHitFeaturizer


EVALUATION_CACHE_SCHEMA_VERSION = 1
EVALUATION_CACHE_POLICY_VERSION = "pxm_eval_top4_maxdate_2021_09_30_v1"
EVALUATION_CACHE_CUTOFF_TEXT = "2021-09-30"
EVALUATION_CACHE_CUTOFF = datetime.strptime(
    EVALUATION_CACHE_CUTOFF_TEXT,
    "%Y-%m-%d",
)
EVALUATION_CACHE_MAX_TEMPLATES = 4
EVALUATION_CACHE_FILENAME = "template_features.pkl.gz"
EVALUATION_CACHE_SUCCESS_FILENAME = "_SUCCESS.json"
EVALUATION_CACHE_MANIFEST_FILENAME = "_manifests/build_manifest.jsonl"
EVALUATION_CACHE_STATS_FILENAME = "_manifests/build_stats.json"

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


@dataclasses.dataclass(frozen=True)
class EvaluationTemplateCacheConfig:
    """Immutable configuration for PXM evaluation-cache workers."""

    cache_root: str
    mmcif_dir: str
    kalign_binary_path: str
    release_dates_path: str
    obsolete_pdbs_path: str
    gzip_compresslevel: int = 1

    def __post_init__(self) -> None:
        if not str(self.cache_root).strip():
            raise ValueError("cache_root must be non-empty")
        if not str(self.mmcif_dir).strip():
            raise ValueError("mmcif_dir must be non-empty")
        if not str(self.kalign_binary_path).strip():
            raise ValueError("kalign_binary_path must be non-empty")
        if not str(self.release_dates_path).strip():
            raise ValueError("release_dates_path must be non-empty")
        if not str(self.obsolete_pdbs_path).strip():
            raise ValueError("obsolete_pdbs_path must be non-empty")
        if not 0 <= int(self.gzip_compresslevel) <= 9:
            raise ValueError("gzip_compresslevel must be in [0, 9]")


@dataclasses.dataclass(frozen=True)
class LoadedTemplateHits:
    """Parsed hits plus reproducible source-artifact provenance."""

    hits: tuple[TemplateHit, ...]
    source: Mapping[str, Any]


def sequence_sha256(sequence: str) -> str:
    """Return the digest used as both cache identity and directory name."""

    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def evaluation_cache_path(
    cache_root: str | Path,
    sequence: str,
    *,
    filename: str = EVALUATION_CACHE_FILENAME,
) -> Path:
    """Return ``root/<sha[:2]>/<sha>/<filename>`` for one query sequence."""

    digest = sequence_sha256(sequence)
    return Path(cache_root) / digest[:2] / digest / filename


def file_sha256(path: str | Path) -> str:
    """Hash one file without loading it fully into memory."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sources_sha256(sequence_sources: Mapping[str, str]) -> str:
    """Hash a sequence-to-source mapping independent of input file formatting."""

    canonical_records = [
        {
            "sequence": sequence,
            "source_hit_path": str(sequence_sources[sequence]),
        }
        for sequence in sorted(sequence_sources)
    ]
    encoded = json.dumps(
        canonical_records,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_sequence_source_record(
    sequence: Any,
    source_hit_path: Any,
) -> tuple[str, str]:
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("sequence must be a non-empty string")
    if sequence != sequence.strip() or any(character.isspace() for character in sequence):
        raise ValueError("sequence may not contain leading/trailing or embedded whitespace")
    if not isinstance(source_hit_path, str) or not source_hit_path.strip():
        raise ValueError(
            f"source_hit_path for sequence {sequence_sha256(sequence)} "
            "must be a non-empty string"
        )
    normalized_source = os.path.abspath(
        os.path.expanduser(source_hit_path.strip())
    )
    return sequence, normalized_source


def load_sequence_sources(path: str | Path) -> dict[str, str]:
    """Load JSON/JSONL records mapping query sequences to hit paths.

    Supported JSON forms are ``{sequence: source_hit_path}``, a list of
    ``{"sequence", "source_hit_path"}`` records, or an object containing such
    a list under ``records``. JSONL accepts one record object per non-empty
    line.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    records: list[tuple[Any, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    raise TypeError(
                        f"JSONL line {line_number} must contain an object"
                    )
                records.append(
                    (
                        record.get("sequence"),
                        record.get("source_hit_path"),
                    )
                )
    else:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, Mapping) and "records" in payload:
            payload = payload["records"]
        if isinstance(payload, Mapping):
            records.extend(payload.items())
        elif isinstance(payload, list):
            for index, record in enumerate(payload):
                if not isinstance(record, Mapping):
                    raise TypeError(f"JSON record {index} must be an object")
                records.append(
                    (
                        record.get("sequence"),
                        record.get("source_hit_path"),
                    )
                )
        else:
            raise TypeError(
                "sequence-sources JSON must be an object or a list of records"
            )

    sequence_sources: dict[str, str] = {}
    for sequence_value, source_value in records:
        sequence, source_path = _validate_sequence_source_record(
            sequence_value,
            source_value,
        )
        previous = sequence_sources.get(sequence)
        if previous is not None and previous != source_path:
            raise ValueError(
                "duplicate sequence maps to different source paths: "
                f"sha256={sequence_sha256(sequence)}"
            )
        sequence_sources[sequence] = source_path
    if not sequence_sources:
        raise ValueError("sequence-sources manifest contains no records")
    return sequence_sources


def _read_optional_text(path: Path) -> tuple[bool, str, Optional[str], int]:
    if not path.is_file():
        return False, "", None, 0
    raw_content = path.read_bytes()
    try:
        content = raw_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"template hit artifact is not UTF-8: {path}") from exc
    return True, content, hashlib.sha256(raw_content).hexdigest(), len(raw_content)


def _candidate_hit_paths(source_hit_path: str | Path) -> tuple[Path, Path]:
    requested = Path(source_hit_path)
    if requested.is_dir():
        return requested / "hmmsearch.a3m", requested / "concat.hhr"

    lower_name = requested.name.lower()
    if lower_name.endswith(".a3m"):
        return requested, requested.parent / "concat.hhr"
    if lower_name.endswith(".hhr"):
        return requested.parent / "hmmsearch.a3m", requested
    if requested.exists():
        raise ValueError(
            "source_hit_path must be a directory or an .a3m/.hhr file: "
            f"{requested}"
        )

    # A missing path with no known suffix is still a missing source, not a
    # malformed-file error. This keeps failure classification unambiguous.
    return requested / "hmmsearch.a3m", requested / "concat.hhr"


def load_template_hits(
    *,
    query_sequence: str,
    source_hit_path: str | Path,
) -> LoadedTemplateHits:
    """Load hits with the exact current eval precedence: non-empty HHR, else A3M."""

    a3m_path, hhr_path = _candidate_hit_paths(source_hit_path)
    a3m_exists, a3m_content, a3m_sha256, a3m_size = _read_optional_text(a3m_path)
    hhr_exists, hhr_content, hhr_sha256, hhr_size = _read_optional_text(hhr_path)
    if not a3m_exists and not hhr_exists:
        raise FileNotFoundError(
            "neither evaluation template hit artifact exists: "
            f"{a3m_path}, {hhr_path}"
        )

    if hhr_content:
        selected_format = "hhr"
        hits = HHRParser.parse(hhr_string=hhr_content)
    else:
        selected_format = "a3m"
        hits = HmmsearchA3MParser.parse(
            query_seq=query_sequence,
            a3m_str=a3m_content,
            skip_first=False,
        )

    artifacts = {
        "a3m": {
            "path": str(a3m_path.resolve(strict=False)),
            "exists": a3m_exists,
            "size_bytes": a3m_size,
            "sha256": a3m_sha256,
        },
        "hhr": {
            "path": str(hhr_path.resolve(strict=False)),
            "exists": hhr_exists,
            "size_bytes": hhr_size,
            "sha256": hhr_sha256,
        },
    }
    fingerprint_material = {
        "selected_format": selected_format,
        "a3m": {
            "exists": a3m_exists,
            "size_bytes": a3m_size,
            "sha256": a3m_sha256,
        },
        "hhr": {
            "exists": hhr_exists,
            "size_bytes": hhr_size,
            "sha256": hhr_sha256,
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_material,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    source = {
        "requested_path": str(Path(source_hit_path).resolve(strict=False)),
        "selected_format": selected_format,
        "artifact_exists_count": int(a3m_exists) + int(hhr_exists),
        "artifacts": artifacts,
        "fingerprint_sha256": fingerprint,
    }
    return LoadedTemplateHits(hits=tuple(hits), source=source)


def _atomic_pickle_gzip_dump(
    payload: Mapping[str, Any],
    output_path: Path,
    *,
    compresslevel: int,
) -> None:
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


def atomic_json_dump(payload: Mapping[str, Any], output_path: str | Path) -> None:
    """Atomically write a stable, human-readable JSON object."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def load_evaluation_template_cache(path: str | Path) -> dict[str, Any]:
    """Load one gzip-compressed evaluation cache artifact."""

    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"evaluation template cache must contain a dict: {path}")
    return payload


def _unwrap_scalar(value: Any, *, field_name: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(f"{field_name} must be scalar, got {value.shape}")
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, dict, set)):
        raise TypeError(f"{field_name} must be scalar")
    return value


def _decode_text_scalar(value: Any, *, field_name: str) -> str:
    value = _unwrap_scalar(value, field_name=field_name)
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8")
    raise TypeError(f"{field_name} must contain text or bytes")


def _validate_candidate_features(
    features: Mapping[str, Any],
    *,
    query_sequence: str,
    candidate_index: int,
) -> None:
    prefix = f"candidate {candidate_index}"
    missing_keys = _RAW_TEMPLATE_FEATURE_KEYS.difference(features)
    if missing_keys:
        raise ValueError(
            f"{prefix} is missing raw feature keys: "
            + ", ".join(sorted(missing_keys))
        )

    query_length = len(query_sequence)
    aatype = features["template_aatype"]
    if not isinstance(aatype, np.ndarray) or aatype.shape != (query_length,):
        raise ValueError(
            f"{prefix} template_aatype must have shape ({query_length},)"
        )
    if not np.issubdtype(aatype.dtype, np.integer):
        raise TypeError(f"{prefix} template_aatype must use an integer dtype")
    if np.any(aatype < 0) or np.any(
        aatype >= PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37.shape[0]
    ):
        raise ValueError(f"{prefix} template_aatype contains invalid values")

    atom_masks = features["template_all_atom_masks"]
    if (
        not isinstance(atom_masks, np.ndarray)
        or atom_masks.shape != (query_length, 37)
    ):
        raise ValueError(
            f"{prefix} template_all_atom_masks must have shape "
            f"({query_length}, 37)"
        )
    if not (
        np.issubdtype(atom_masks.dtype, np.number)
        and not np.issubdtype(atom_masks.dtype, np.complexfloating)
    ):
        raise TypeError(f"{prefix} template_all_atom_masks must be real numeric")
    if not np.isfinite(atom_masks).all():
        raise ValueError(f"{prefix} template_all_atom_masks contains non-finite values")
    if np.any(atom_masks < 0) or np.any(atom_masks > 1):
        raise ValueError(f"{prefix} template_all_atom_masks is outside [0, 1]")
    if not np.any(atom_masks > 0):
        raise ValueError(f"{prefix} template_all_atom_masks is entirely zero")

    atom_positions = features["template_all_atom_positions"]
    if (
        not isinstance(atom_positions, np.ndarray)
        or atom_positions.shape != (query_length, 37, 3)
    ):
        raise ValueError(
            f"{prefix} template_all_atom_positions must have shape "
            f"({query_length}, 37, 3)"
        )
    if not (
        np.issubdtype(atom_positions.dtype, np.number)
        and not np.issubdtype(atom_positions.dtype, np.complexfloating)
    ):
        raise TypeError(
            f"{prefix} template_all_atom_positions must be real numeric"
        )
    if not np.isfinite(atom_positions).all():
        raise ValueError(
            f"{prefix} template_all_atom_positions contains non-finite values"
        )

    domain_name = _decode_text_scalar(
        features["template_domain_names"],
        field_name=f"{prefix} template_domain_names",
    )
    if not domain_name:
        raise ValueError(f"{prefix} template_domain_names cannot be empty")

    release_date_text = _decode_text_scalar(
        features["template_release_date"],
        field_name=f"{prefix} template_release_date",
    )
    release_date = datetime.strptime(release_date_text, "%Y-%m-%d")
    if release_date > EVALUATION_CACHE_CUTOFF:
        raise ValueError(
            f"{prefix} release date {release_date_text} exceeds "
            f"{EVALUATION_CACHE_CUTOFF_TEXT}"
        )

    template_sequence = _decode_text_scalar(
        features["template_sequence"],
        field_name=f"{prefix} template_sequence",
    )
    if len(template_sequence) != query_length:
        raise ValueError(
            f"{prefix} template_sequence length {len(template_sequence)} "
            f"!= {query_length}"
        )


def validate_evaluation_cache_payload(
    payload: Mapping[str, Any],
    *,
    expected_sequence: Optional[str] = None,
) -> None:
    """Fail closed if one cache payload violates the locked evaluation contract."""

    if payload.get("schema_version") != EVALUATION_CACHE_SCHEMA_VERSION:
        raise ValueError("evaluation cache schema_version mismatch")
    if payload.get("policy_version") != EVALUATION_CACHE_POLICY_VERSION:
        raise ValueError("evaluation cache policy_version mismatch")
    if payload.get("cutoff_date") != EVALUATION_CACHE_CUTOFF_TEXT:
        raise ValueError("evaluation cache cutoff_date mismatch")
    if payload.get("max_templates") != EVALUATION_CACHE_MAX_TEMPLATES:
        raise ValueError("evaluation cache max_templates mismatch")

    sequence = payload.get("sequence")
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("evaluation cache sequence must be non-empty")
    if payload.get("sequence_sha256") != sequence_sha256(sequence):
        raise ValueError("evaluation cache sequence_sha256 mismatch")
    if expected_sequence is not None and sequence != expected_sequence:
        raise ValueError("evaluation cache sequence does not match expected sequence")

    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise TypeError("evaluation cache source must be a mapping")
    if int(source.get("artifact_exists_count", 0)) <= 0:
        raise ValueError("evaluation cache records no existing source artifact")
    fingerprint = source.get("fingerprint_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("evaluation cache source fingerprint is invalid")

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise TypeError("evaluation cache candidates must be a list")
    if len(candidates) > EVALUATION_CACHE_MAX_TEMPLATES:
        raise ValueError("evaluation cache contains more than four templates")
    status = payload.get("status")
    if status not in {"complete", "complete_empty"}:
        raise ValueError(f"invalid evaluation cache status: {status!r}")
    if status == "complete" and not candidates:
        raise ValueError("complete evaluation cache has no candidates")
    if status == "complete_empty" and candidates:
        raise ValueError("complete_empty evaluation cache has candidates")

    counts = payload.get("counts")
    if not isinstance(counts, Mapping):
        raise TypeError("evaluation cache counts must be a mapping")
    raw_hit_count = counts.get("raw_hit_count")
    valid_cached_count = counts.get("valid_cached_count")
    if (
        not isinstance(raw_hit_count, int)
        or isinstance(raw_hit_count, bool)
        or raw_hit_count < len(candidates)
    ):
        raise ValueError(
            "evaluation cache raw_hit_count must be an integer greater than "
            "or equal to the candidate count"
        )
    if valid_cached_count != len(candidates):
        raise ValueError(
            "evaluation cache valid_cached_count does not match candidates"
        )

    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise TypeError(f"candidate {index} must be a mapping")
        features = candidate.get("features")
        if not isinstance(features, Mapping):
            raise TypeError(f"candidate {index} features must be a mapping")
        _validate_candidate_features(
            features,
            query_sequence=sequence,
            candidate_index=index,
        )


class EvaluationTemplateCacheBuilder:
    """Build exact top-four, date-locked PXM template features."""

    def __init__(
        self,
        config: EvaluationTemplateCacheConfig,
        *,
        online_featurizer: Optional[TemplateHitFeaturizer] = None,
    ) -> None:
        self.config = config
        self.online_featurizer = online_featurizer or TemplateHitFeaturizer(
            mmcif_dir=config.mmcif_dir,
            # Keep this unset to match the current PXM evaluator, which parses
            # raw mmCIF and cannot silently lose a hit through a missing parsed
            # mmCIF cache entry.
            template_cache_dir=None,
            max_hits=EVALUATION_CACHE_MAX_TEMPLATES,
            kalign_binary_path=config.kalign_binary_path,
            max_template_date=EVALUATION_CACHE_CUTOFF_TEXT,
            release_dates_path=config.release_dates_path,
            obsolete_pdbs_path=config.obsolete_pdbs_path,
            strict_error_check=False,
            _shuffle_top_k_prefiltered=None,
            _zero_center_positions=True,
            _max_template_candidates_num=None,
            fetch_remote=False,
        )

    def output_path(self, sequence: str) -> Path:
        return evaluation_cache_path(self.config.cache_root, sequence)

    def build(
        self,
        *,
        sequence: str,
        source_hit_path: str,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Build one artifact and return its manifest record."""

        started_at = time.time()
        loaded_hits = load_template_hits(
            query_sequence=sequence,
            source_hit_path=source_hit_path,
        )
        output_path = self.output_path(sequence)
        digest = sequence_sha256(sequence)

        if resume and output_path.is_file():
            try:
                existing = load_evaluation_template_cache(output_path)
                validate_evaluation_cache_payload(
                    existing,
                    expected_sequence=sequence,
                )
                if (
                    existing["source"]["fingerprint_sha256"]
                    == loaded_hits.source["fingerprint_sha256"]
                ):
                    return {
                        "status": "skipped_existing",
                        "sequence_sha256": digest,
                        "source_hit_path": str(source_hit_path),
                        "source_fingerprint_sha256": loaded_hits.source[
                            "fingerprint_sha256"
                        ],
                        "output_path": str(output_path),
                        "output_sha256": file_sha256(output_path),
                        "cache_status": existing["status"],
                        "candidate_count": len(existing["candidates"]),
                        "elapsed_seconds": time.time() - started_at,
                    }
            except (
                EOFError,
                OSError,
                pickle.PickleError,
                TypeError,
                ValueError,
            ):
                pass

        search_result, timing = self.online_featurizer.get_templates(
            sequence_uid=f"pxm_eval_{digest[:16]}",
            query_sequence=sequence,
            hits=loaded_hits.hits,
            max_template_date=EVALUATION_CACHE_CUTOFF,
        )
        features = list(search_result.features)
        final_hits = list(search_result.hits)
        if len(features) != len(final_hits):
            raise ValueError(
                "TemplateHitFeaturizer returned different feature/hit counts: "
                f"{len(features)} != {len(final_hits)}"
            )
        if len(features) > EVALUATION_CACHE_MAX_TEMPLATES:
            raise ValueError(
                "TemplateHitFeaturizer returned more than four templates"
            )

        candidates: list[dict[str, Any]] = []
        for rank, (feature, hit) in enumerate(
            zip(features, final_hits),
            start=1,
        ):
            pdb_id, chain_id = get_pdb_id_and_chain(hit)
            candidates.append(
                {
                    "rank": rank,
                    "name": hit.name,
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "sum_probs": float(
                        hit.sum_probs if hit.sum_probs is not None else 0.0
                    ),
                    "features": dict(feature),
                }
            )

        status = "complete" if candidates else "complete_empty"
        completed_at = time.time()
        payload: dict[str, Any] = {
            "schema_version": EVALUATION_CACHE_SCHEMA_VERSION,
            "policy_version": EVALUATION_CACHE_POLICY_VERSION,
            "cutoff_date": EVALUATION_CACHE_CUTOFF_TEXT,
            "max_templates": EVALUATION_CACHE_MAX_TEMPLATES,
            "status": status,
            "sequence": sequence,
            "sequence_sha256": digest,
            "source": dict(loaded_hits.source),
            "counts": {
                "raw_hit_count": len(loaded_hits.hits),
                "valid_cached_count": len(candidates),
                "error_count": len(search_result.errors),
                "warning_count": len(search_result.warnings),
            },
            "candidates": candidates,
            "errors": list(search_result.errors),
            "warnings": list(search_result.warnings),
            "build": {
                "created_at_unix": completed_at,
                "elapsed_seconds": completed_at - started_at,
                "online_flow": "TemplateHitFeaturizer.get_templates",
                "raw_mmcif": True,
                "fetch_remote": False,
                "strict_error_check": False,
                "shuffle_top_k_prefiltered": None,
                "timing_seconds": dict(timing),
            },
        }
        validate_evaluation_cache_payload(
            payload,
            expected_sequence=sequence,
        )
        _atomic_pickle_gzip_dump(
            payload,
            output_path,
            compresslevel=self.config.gzip_compresslevel,
        )
        return {
            "status": status,
            "sequence_sha256": digest,
            "source_hit_path": str(source_hit_path),
            "source_fingerprint_sha256": loaded_hits.source[
                "fingerprint_sha256"
            ],
            "output_path": str(output_path),
            "output_sha256": file_sha256(output_path),
            "cache_status": status,
            "candidate_count": len(candidates),
            "raw_hit_count": len(loaded_hits.hits),
            "error_count": len(search_result.errors),
            "warning_count": len(search_result.warnings),
            "elapsed_seconds": time.time() - started_at,
        }


def validate_cache_collection(
    *,
    sequence_sources: Mapping[str, str],
    cache_root: str | Path,
    manifest_path: str | Path,
    stats_path: str | Path,
    success_path: str | Path,
) -> dict[str, Any]:
    """Validate success metadata, complete coverage, hashes, and every payload."""

    cache_root = Path(cache_root)
    manifest_path = Path(manifest_path)
    stats_path = Path(stats_path)
    success_path = Path(success_path)
    for required_file in (manifest_path, stats_path, success_path):
        if not required_file.is_file():
            raise FileNotFoundError(required_file)

    with open(success_path, "r", encoding="utf-8") as handle:
        success = json.load(handle)
    with open(stats_path, "r", encoding="utf-8") as handle:
        stats = json.load(handle)
    if not isinstance(success, Mapping) or not isinstance(stats, Mapping):
        raise TypeError("success and stats payloads must be mappings")

    expected_count = len(sequence_sources)
    expected_sources_sha256 = sequence_sources_sha256(sequence_sources)
    fixed_expectations = {
        "schema_version": EVALUATION_CACHE_SCHEMA_VERSION,
        "policy_version": EVALUATION_CACHE_POLICY_VERSION,
        "cutoff_date": EVALUATION_CACHE_CUTOFF_TEXT,
        "max_templates": EVALUATION_CACHE_MAX_TEMPLATES,
        "sequence_count": expected_count,
        "sequence_sources_sha256": expected_sources_sha256,
    }
    for key, expected in fixed_expectations.items():
        if success.get(key) != expected:
            raise ValueError(
                f"_SUCCESS {key} mismatch: {success.get(key)!r} != {expected!r}"
            )
    if success.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("_SUCCESS manifest_sha256 mismatch")
    if success.get("stats_sha256") != file_sha256(stats_path):
        raise ValueError("_SUCCESS stats_sha256 mismatch")

    if int(stats.get("total_tasks", -1)) != expected_count:
        raise ValueError("stats total_tasks does not match sequence coverage")
    if int(stats.get("completed_tasks", -1)) != expected_count:
        raise ValueError("stats completed_tasks does not match sequence coverage")
    if int(stats.get("remaining_tasks", -1)) != 0:
        raise ValueError("stats reports remaining tasks")
    status_counts = stats.get("status_counts")
    if not isinstance(status_counts, Mapping):
        raise TypeError("stats status_counts must be a mapping")
    if int(status_counts.get("failed", 0)) != 0:
        raise ValueError("stats reports failed cache builds")

    manifest_records: dict[str, Mapping[str, Any]] = {}
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise TypeError(f"manifest line {line_number} is not an object")
            digest = record.get("sequence_sha256")
            if not isinstance(digest, str):
                raise ValueError(
                    f"manifest line {line_number} lacks sequence_sha256"
                )
            if digest in manifest_records:
                raise ValueError(f"duplicate manifest sequence_sha256: {digest}")
            if record.get("status") not in {
                "complete",
                "complete_empty",
                "skipped_existing",
            }:
                raise ValueError(
                    f"manifest line {line_number} has unsuccessful status"
                )
            manifest_records[digest] = record
    if len(manifest_records) != expected_count:
        raise ValueError(
            "manifest record count does not match expected sequence coverage"
        )

    complete_count = 0
    complete_empty_count = 0
    missing_count = 0
    for sequence in sequence_sources:
        digest = sequence_sha256(sequence)
        record = manifest_records.get(digest)
        if record is None:
            missing_count += 1
            continue
        cache_path = evaluation_cache_path(cache_root, sequence)
        if Path(record.get("output_path", "")).resolve(strict=False) != (
            cache_path.resolve(strict=False)
        ):
            raise ValueError(f"manifest output path mismatch for {digest}")
        if not cache_path.is_file():
            missing_count += 1
            continue
        if record.get("output_sha256") != file_sha256(cache_path):
            raise ValueError(f"cache artifact hash mismatch for {digest}")
        payload = load_evaluation_template_cache(cache_path)
        validate_evaluation_cache_payload(
            payload,
            expected_sequence=sequence,
        )
        if payload["status"] == "complete":
            complete_count += 1
        else:
            complete_empty_count += 1
    if missing_count:
        raise ValueError(f"evaluation cache has {missing_count} missing artifacts")

    return {
        "status": "valid",
        "schema_version": EVALUATION_CACHE_SCHEMA_VERSION,
        "policy_version": EVALUATION_CACHE_POLICY_VERSION,
        "cutoff_date": EVALUATION_CACHE_CUTOFF_TEXT,
        "max_templates": EVALUATION_CACHE_MAX_TEMPLATES,
        "sequence_count": expected_count,
        "complete_count": complete_count,
        "complete_empty_count": complete_empty_count,
        "failed_count": 0,
        "missing_count": 0,
        "sequence_sources_sha256": expected_sources_sha256,
        "manifest_sha256": file_sha256(manifest_path),
        "stats_sha256": file_sha256(stats_path),
        "success_sha256": file_sha256(success_path),
    }
