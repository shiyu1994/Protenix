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

"""Read-only audit of OpenFold3-disordered alignment representatives.

The CLI is deliberately fail-closed to the
``openfold3_disordered_v20260725`` dataset.  It accepts only the dataset root,
then validates the dataset manifest, success marker, and index hashes before
deriving the bioassembly and alignment paths declared by that manifest.  It
does not accept arbitrary index, bioassembly, or alignment-directory
arguments.

For each indexed protein/RNA chain, the audit compares the chain's canonical
query with the query stored in its currently declared representative NPZ.
Chains listed in ``openfold3_query_only_chain_ids`` are classified without
opening any representative NPZ after their chain annotations and canonical
query have been validated.
Only a current-query mismatch triggers loading the other representatives
declared by the same bioassembly, matching the runtime recovery path.  Records
use one of these classifications:

* ``explicit_query_only``: the bioassembly explicitly marks this valid chain
  as query-only, so no representative archive is inspected;
* ``exact_current``: the current representative has the exact canonical query;
* ``remappable_unique``: exactly one declared representative matches;
* ``missing``: a required representative archive is absent;
* ``ambiguous``: more than one declared representative matches;
* ``no_exact``: all inspected representatives are valid but none matches;
* ``corrupt``: a pickle, mapping, or relevant NPZ payload is malformed.

Results and the summary are published with ``os.replace()`` only after the
selected shard is complete.  A separate append-only state JSONL stores one
completed structure per line, allowing an interrupted run to resume without
publishing a partial result.

Two-host example (write to distinct output files)::

    python scripts/audit_openfold3_alignment_representatives.py \
      --dataset-root /mnt/project_abd_hs/klp/databases/pdb_disordered \
      --output-jsonl /path/audit.shard0.jsonl \
      --summary-json /path/audit.shard0.summary.json \
      --shard-count 2 --shard-index 0 --workers 128 --resume

    python scripts/audit_openfold3_alignment_representatives.py \
      --dataset-root /mnt/project_abd_hs/klp/databases/pdb_disordered \
      --output-jsonl /path/audit.shard1.jsonl \
      --summary-json /path/audit.shard1.summary.json \
      --shard-count 2 --shard-index 1 --workers 128 --resume
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gzip
import hashlib
import json
import multiprocessing
import os
import pickle
import sys
import tempfile
import time
import traceback
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Optional

import numpy as np


DATASET_NAME = "openfold3_disordered_v20260725"
MANIFEST_ROLE = "openfold3_disordered_protenix_dataset"
SUCCESS_ROLE = "openfold3_disordered_protenix_dataset_success"
STATE_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 2
CLASSIFICATIONS = (
    "explicit_query_only",
    "exact_current",
    "remappable_unique",
    "missing",
    "ambiguous",
    "no_exact",
    "corrupt",
)
PROTEIN_CHAIN = "polypeptide(L)"
RNA_CHAIN = "polyribonucleotide"
SUPPORTED_POLYMER_TYPES = frozenset((PROTEIN_CHAIN, RNA_CHAIN))
PROTEIN_SOURCES = (
    "uniref90_hits",
    "cfdb_hits",
    "mgnify_hits",
    "uniprot_hits",
)
RNA_SOURCES = (
    "nucleotide_collection_hits",
    "rfam_hits",
    "rnacentral_hits",
)

_WORKER_ALIGNMENT_DIRS: tuple[Path, ...] = ()
_WORKER_REPRESENTATIVE_CACHE: dict[
    tuple[str, str], "RepresentativeQuery"
] = {}


class AuditError(RuntimeError):
    """Base class for fail-closed audit failures."""


class AuditConfigurationError(AuditError):
    """Raised when the requested root is not the declared disorder dataset."""


class AuditDataError(AuditError):
    """Raised for an invalid audit input or state artifact."""


@dataclasses.dataclass(frozen=True)
class DatasetContract:
    """Validated, immutable inputs for one disorder audit."""

    dataset_root: Path
    protenix_root: Path
    structures_root: Path
    alignment_dirs: tuple[Path, ...]
    index_path: Path
    index_sha256: str
    index_row_count: int
    manifest_path: Path
    manifest_sha256: str
    success_path: Path
    success_sha256: str
    declared_pickle_count: int


@dataclasses.dataclass(frozen=True)
class AuditTask:
    """One indexed bioassembly assigned to a shard."""

    ordinal: int
    pdb_id: str
    bioassembly_relpath: str
    bioassembly_path: str
    task_id: str
    shard_index: int
    shard_count: int


@dataclasses.dataclass(frozen=True)
class RepresentativeQuery:
    """Small result retained after releasing a potentially large NPZ."""

    representative_id: str
    status: str
    path: Optional[str]
    query: Optional[str]
    query_length: Optional[int]
    query_sha256: Optional[str]
    source_names: tuple[str, ...]
    error_type: Optional[str] = None
    error_message: Optional[str] = None


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _short_error(error: BaseException, limit: int = 1000) -> str:
    message = " ".join(str(error).split())
    return message[:limit]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditConfigurationError(
            f"cannot read {label} JSON {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise AuditConfigurationError(f"{label} must be a JSON object: {path}")
    return value


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise AuditConfigurationError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise AuditConfigurationError(f"{label} must be an integer") from error
    if result < minimum:
        raise AuditConfigurationError(
            f"{label} must be >= {minimum}, got {result}"
        )
    return result


def validate_disordered_dataset_root(dataset_root: Path) -> DatasetContract:
    """Validate the exact disorder dataset contract and derive all inputs."""
    root = dataset_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise AuditConfigurationError(f"dataset root is not a directory: {root}")

    protenix_root = root / "protenix"
    manifest_path = protenix_root / "manifest.json"
    success_path = protenix_root / "_SUCCESS.json"
    index_path = protenix_root / "indices.csv.gz"
    for label, path in (
        ("manifest", manifest_path),
        ("success marker", success_path),
        ("index", index_path),
    ):
        if not path.is_file():
            raise AuditConfigurationError(
                f"OpenFold3-disordered {label} is missing: {path}"
            )

    manifest = _load_json_object(manifest_path, label="dataset manifest")
    success = _load_json_object(success_path, label="dataset success marker")
    if manifest.get("artifact_role") != MANIFEST_ROLE:
        raise AuditConfigurationError(
            "dataset manifest artifact_role is not OpenFold3 disordered: "
            f"{manifest.get('artifact_role')!r}"
        )
    if success.get("artifact_role") != SUCCESS_ROLE:
        raise AuditConfigurationError(
            "dataset success artifact_role is not OpenFold3 disordered: "
            f"{success.get('artifact_role')!r}"
        )
    if manifest.get("status") != "complete" or not bool(
        manifest.get("success_eligible")
    ):
        raise AuditConfigurationError(
            "OpenFold3-disordered manifest is not complete/success-eligible"
        )
    cache = manifest.get("cache")
    if not isinstance(cache, Mapping) or (
        cache.get("cache_name") != "disordered-pdb"
        or cache.get("cache_type") != "ClusteredDatasetCache"
    ):
        raise AuditConfigurationError(
            "manifest cache identity is not the disordered-pdb clustered cache"
        )

    manifest_sha256 = _sha256_file(manifest_path)
    if success.get("manifest") != "manifest.json":
        raise AuditConfigurationError(
            "success marker must reference manifest.json"
        )
    if success.get("manifest_sha256") != manifest_sha256:
        raise AuditConfigurationError(
            "success marker manifest_sha256 does not match manifest.json"
        )

    parameters = manifest.get("parameters")
    artifacts = manifest.get("artifacts")
    bioassemblies = manifest.get("bioassemblies")
    if not isinstance(parameters, Mapping) or not isinstance(
        artifacts, Mapping
    ) or not isinstance(bioassemblies, Mapping):
        raise AuditConfigurationError(
            "manifest requires parameters, artifacts, and bioassemblies objects"
        )

    expected_structures_root = (root / "structures" / "structures").resolve()
    try:
        declared_structures_root = Path(
            str(parameters["structures_root"])
        ).expanduser().resolve(strict=True)
    except (KeyError, OSError) as error:
        raise AuditConfigurationError(
            "manifest structures_root is missing or invalid"
        ) from error
    if declared_structures_root != expected_structures_root:
        raise AuditConfigurationError(
            "manifest structures_root escapes the disorder dataset root: "
            f"{declared_structures_root} != {expected_structures_root}"
        )

    raw_alignment_dirs = parameters.get("alignment_dirs")
    if not isinstance(raw_alignment_dirs, list) or not raw_alignment_dirs:
        raise AuditConfigurationError(
            "manifest parameters.alignment_dirs must be a non-empty list"
        )
    alignment_base = (root / "alignment_arrays").resolve()
    alignment_dirs: list[Path] = []
    for raw_path in raw_alignment_dirs:
        try:
            path = Path(str(raw_path)).expanduser().resolve(strict=True)
        except OSError as error:
            raise AuditConfigurationError(
                f"manifest alignment directory is invalid: {raw_path!r}"
            ) from error
        if not path.is_dir() or not _is_relative_to(path, alignment_base):
            raise AuditConfigurationError(
                "manifest alignment directory is outside disorder alignment "
                f"storage: {path}"
            )
        if path not in alignment_dirs:
            alignment_dirs.append(path)

    indices_artifact = artifacts.get("indices")
    if not isinstance(indices_artifact, Mapping):
        raise AuditConfigurationError(
            "manifest artifacts.indices must be an object"
        )
    declared_index_path = Path(
        str(indices_artifact.get("path", ""))
    ).expanduser().resolve()
    if declared_index_path != index_path.resolve():
        raise AuditConfigurationError(
            "manifest index path is not the disorder index: "
            f"{declared_index_path} != {index_path.resolve()}"
        )
    index_sha256 = _sha256_file(index_path)
    if indices_artifact.get("sha256") != index_sha256:
        raise AuditConfigurationError(
            "disorder index SHA256 does not match the manifest"
        )

    index_row_count = _require_int(
        indices_artifact.get("rows"),
        label="manifest index row count",
        minimum=1,
    )
    if _require_int(
        success.get("index_row_count"),
        label="success index_row_count",
        minimum=1,
    ) != index_row_count:
        raise AuditConfigurationError(
            "manifest and success marker disagree on index row count"
        )
    declared_pickle_count = _require_int(
        bioassemblies.get("count"),
        label="manifest bioassembly count",
        minimum=1,
    )
    if _require_int(
        success.get("pickle_count"),
        label="success pickle_count",
        minimum=1,
    ) != declared_pickle_count:
        raise AuditConfigurationError(
            "manifest and success marker disagree on pickle count"
        )

    return DatasetContract(
        dataset_root=root,
        protenix_root=protenix_root,
        structures_root=expected_structures_root,
        alignment_dirs=tuple(alignment_dirs),
        index_path=index_path,
        index_sha256=index_sha256,
        index_row_count=index_row_count,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        success_path=success_path,
        success_sha256=_sha256_file(success_path),
        declared_pickle_count=declared_pickle_count,
    )


def _validate_bioassembly_relpath(
    value: str,
    *,
    pdb_id: str,
) -> str:
    relpath = PurePosixPath(value)
    expected = PurePosixPath(
        pdb_id,
        "bioassembly",
        f"{pdb_id}.pkl.gz",
    )
    if (
        not value
        or relpath.is_absolute()
        or ".." in relpath.parts
        or relpath != expected
    ):
        raise AuditConfigurationError(
            f"unsafe/non-disorder bioassembly_relpath for {pdb_id!r}: {value!r}"
        )
    return relpath.as_posix()


def read_disordered_inventory(contract: DatasetContract) -> list[tuple[str, str]]:
    """Read unique bioassemblies from the validated disorder index."""
    by_pdb_id: dict[str, str] = {}
    row_count = 0
    try:
        with gzip.open(
            contract.index_path,
            "rt",
            encoding="utf-8",
            newline="",
        ) as stream:
            reader = csv.DictReader(stream)
            required = {"pdb_id", "bioassembly_relpath"}
            if reader.fieldnames is None or not required.issubset(
                reader.fieldnames
            ):
                raise AuditConfigurationError(
                    "disorder index is missing pdb_id/bioassembly_relpath"
                )
            for row_count, row in enumerate(reader, start=1):
                pdb_id = str(row.get("pdb_id", "")).strip()
                if (
                    not pdb_id
                    or "/" in pdb_id
                    or "\\" in pdb_id
                    or "\0" in pdb_id
                ):
                    raise AuditConfigurationError(
                        f"invalid PDB ID at disorder index row {row_count}"
                    )
                relpath = _validate_bioassembly_relpath(
                    str(row.get("bioassembly_relpath", "")).strip(),
                    pdb_id=pdb_id,
                )
                previous = by_pdb_id.setdefault(pdb_id, relpath)
                if previous != relpath:
                    raise AuditConfigurationError(
                        f"disorder index maps {pdb_id!r} to multiple pickles"
                    )
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise AuditConfigurationError(
            f"cannot read validated disorder index: {error}"
        ) from error

    if row_count != contract.index_row_count:
        raise AuditConfigurationError(
            "actual disorder index row count does not match manifest: "
            f"{row_count} != {contract.index_row_count}"
        )
    if len(by_pdb_id) != contract.declared_pickle_count:
        raise AuditConfigurationError(
            "unique disorder index bioassembly count does not match manifest: "
            f"{len(by_pdb_id)} != {contract.declared_pickle_count}"
        )
    return sorted(by_pdb_id.items(), key=lambda item: item[1])


def build_tasks(
    contract: DatasetContract,
    inventory: Sequence[tuple[str, str]],
    *,
    shard_index: int,
    shard_count: int,
) -> list[AuditTask]:
    """Assign stable ordinal shards after validating shard arguments."""
    if shard_count < 1:
        raise AuditConfigurationError("shard_count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise AuditConfigurationError(
            f"shard_index must be in [0, {shard_count}), got {shard_index}"
        )
    tasks = []
    for ordinal, (pdb_id, relpath) in enumerate(inventory):
        if ordinal % shard_count != shard_index:
            continue
        task_id = _sha256_text(
            "\0".join(
                (
                    DATASET_NAME,
                    contract.index_sha256,
                    str(shard_count),
                    str(shard_index),
                    str(ordinal),
                    relpath,
                )
            )
        )
        tasks.append(
            AuditTask(
                ordinal=ordinal,
                pdb_id=pdb_id,
                bioassembly_relpath=relpath,
                bioassembly_path=str(contract.structures_root / relpath),
                task_id=task_id,
                shard_index=shard_index,
                shard_count=shard_count,
            )
        )
    return tasks


def _alignment_candidate_paths(
    representative_id: str,
    alignment_dirs: Sequence[Path],
) -> Iterator[Path]:
    try:
        pdb_id, label_asym_id = representative_id.rsplit("_", 1)
    except ValueError as error:
        raise AuditDataError(
            f"invalid representative ID {representative_id!r}"
        ) from error
    if (
        not pdb_id
        or not label_asym_id
        or any(char in pdb_id for char in ("/", "\\", "\0"))
        or any(char in label_asym_id for char in ("/", "\\", "\0"))
    ):
        raise AuditDataError(
            f"unsafe representative ID {representative_id!r}"
        )
    for directory in alignment_dirs:
        for candidate in dict.fromkeys((pdb_id, pdb_id.lower(), pdb_id.upper())):
            yield directory / f"{candidate}_{label_asym_id}.npz"


def _find_alignment_path(
    representative_id: str,
    alignment_dirs: Sequence[Path],
) -> Optional[Path]:
    for path in _alignment_candidate_paths(
        representative_id,
        alignment_dirs,
    ):
        if path.is_file():
            return path
    return None


def _query_from_msa(msa: np.ndarray, *, source_name: str) -> str:
    if msa.ndim != 2 or msa.shape[0] == 0 or msa.shape[1] == 0:
        raise ValueError(
            f"{source_name} MSA must be non-empty rank 2, got {msa.shape}"
        )
    if msa.dtype.kind not in {"S", "U"}:
        raise TypeError(
            f"{source_name} MSA must contain byte/Unicode characters, "
            f"got {msa.dtype}"
        )
    if msa.dtype.kind == "S" and msa.dtype.itemsize != 1:
        raise ValueError(
            f"{source_name} byte-string MSA entries must have width 1"
        )
    if np.any(np.char.str_len(msa) != 1):
        raise ValueError(
            f"{source_name} MSA entries must each contain one character"
        )
    return "".join(msa[0].astype("<U1", copy=False).tolist())


def _load_representative_query_uncached(
    representative_id: str,
    polymer_type: str,
    alignment_dirs: Sequence[Path],
) -> RepresentativeQuery:
    """Load/validate only the canonical query, releasing full arrays promptly."""
    try:
        path = _find_alignment_path(representative_id, alignment_dirs)
    except Exception as error:
        return RepresentativeQuery(
            representative_id=representative_id,
            status="corrupt",
            path=None,
            query=None,
            query_length=None,
            query_sha256=None,
            source_names=(),
            error_type=type(error).__name__,
            error_message=_short_error(error),
        )
    if path is None:
        return RepresentativeQuery(
            representative_id=representative_id,
            status="missing",
            path=None,
            query=None,
            query_length=None,
            query_sha256=None,
            source_names=(),
            error_type="FileNotFoundError",
            error_message="declared representative NPZ is missing",
        )
    source_names = (
        PROTEIN_SOURCES if polymer_type == PROTEIN_CHAIN else RNA_SOURCES
    )
    queries: set[str] = set()
    inspected_sources: list[str] = []
    try:
        with np.load(path, allow_pickle=True) as archive:
            for source_name in source_names:
                if source_name not in archive.files:
                    continue
                source_value = archive[source_name]
                if source_value.shape != () or source_value.dtype != object:
                    raise ValueError(
                        f"{source_name} must be a scalar object array"
                    )
                payload = source_value.item()
                if not isinstance(payload, dict):
                    raise TypeError(
                        f"{source_name} must contain a dictionary"
                    )
                missing = {
                    "msa",
                    "deletion_matrix",
                    "metadata",
                } - set(payload)
                if missing:
                    raise ValueError(
                        f"{source_name} is missing {sorted(missing)}"
                    )
                msa = np.asarray(payload["msa"])
                deletion = np.asarray(payload["deletion_matrix"])
                query = _query_from_msa(msa, source_name=source_name)
                if deletion.shape != msa.shape:
                    raise ValueError(
                        f"{source_name} deletion matrix shape "
                        f"{deletion.shape} != MSA shape {msa.shape}"
                    )
                if not np.issubdtype(deletion.dtype, np.integer):
                    raise TypeError(
                        f"{source_name} deletion matrix must be integer"
                    )
                if np.any(deletion < 0):
                    raise ValueError(
                        f"{source_name} deletion matrix contains negatives"
                    )
                metadata = np.asarray(payload["metadata"]).tolist()
                if not isinstance(metadata, list) or len(metadata) != msa.shape[0]:
                    raise ValueError(
                        f"{source_name} metadata depth does not match MSA"
                    )
                queries.add(query)
                inspected_sources.append(source_name)
        if not inspected_sources:
            raise ValueError(
                f"archive has no supported sources for {polymer_type!r}"
            )
        if len(queries) != 1:
            raise ValueError(
                "alignment sources do not share one exact canonical query"
            )
        query = queries.pop()
        if not query:
            raise ValueError("canonical query is empty")
        return RepresentativeQuery(
            representative_id=representative_id,
            status="valid",
            path=str(path),
            query=query,
            query_length=len(query),
            query_sha256=_sha256_text(query),
            source_names=tuple(inspected_sources),
        )
    except Exception as error:
        return RepresentativeQuery(
            representative_id=representative_id,
            status="corrupt",
            path=str(path),
            query=None,
            query_length=None,
            query_sha256=None,
            source_names=tuple(inspected_sources),
            error_type=type(error).__name__,
            error_message=_short_error(error),
        )


def _init_worker(alignment_dirs: Sequence[str]) -> None:
    global _WORKER_ALIGNMENT_DIRS
    global _WORKER_REPRESENTATIVE_CACHE
    _WORKER_ALIGNMENT_DIRS = tuple(Path(path) for path in alignment_dirs)
    _WORKER_REPRESENTATIVE_CACHE = {}


def _load_representative_query(
    representative_id: str,
    polymer_type: str,
) -> RepresentativeQuery:
    key = (representative_id, polymer_type)
    value = _WORKER_REPRESENTATIVE_CACHE.get(key)
    if value is None:
        value = _load_representative_query_uncached(
            representative_id,
            polymer_type,
            _WORKER_ALIGNMENT_DIRS,
        )
        _WORKER_REPRESENTATIVE_CACHE[key] = value
    return value


def _query_only_chain_ids(bioassembly: Mapping[str, Any]) -> frozenset[str]:
    raw_chain_ids = bioassembly.get("openfold3_query_only_chain_ids")
    if raw_chain_ids is None:
        return frozenset()
    if isinstance(raw_chain_ids, np.ndarray):
        if raw_chain_ids.ndim != 1:
            raise AuditDataError(
                "openfold3_query_only_chain_ids ndarray must be one-dimensional"
            )
        values = raw_chain_ids.tolist()
    elif isinstance(raw_chain_ids, (list, tuple, set, frozenset)):
        values = list(raw_chain_ids)
    else:
        raise AuditDataError(
            "openfold3_query_only_chain_ids must be a one-dimensional "
            "collection of chain IDs"
        )

    chain_ids: set[str] = set()
    for value in values:
        if isinstance(value, (str, int, np.str_, np.integer)):
            chain_id = str(value)
        else:
            raise AuditDataError(
                "openfold3_query_only_chain_ids contains a non-scalar chain ID"
            )
        if not chain_id:
            raise AuditDataError(
                "openfold3_query_only_chain_ids contains an empty chain ID"
            )
        chain_ids.add(chain_id)
    return frozenset(chain_ids)


def _chain_contexts(bioassembly: Mapping[str, Any]) -> list[dict[str, Any]]:
    sequences_by_chain_id = bioassembly.get("sequences_by_chain_id")
    entity_sequences = bioassembly.get("sequences")
    entity_poly_type = bioassembly.get("entity_poly_type")
    representatives = bioassembly.get(
        "alignment_representative_id_by_chain_id"
    )
    atom_array = bioassembly.get("atom_array")
    if not isinstance(sequences_by_chain_id, Mapping):
        raise AuditDataError("sequences_by_chain_id must be a mapping")
    if not isinstance(entity_sequences, Mapping):
        raise AuditDataError("sequences must be a mapping")
    if not isinstance(entity_poly_type, Mapping):
        raise AuditDataError("entity_poly_type must be a mapping")
    if not isinstance(representatives, Mapping):
        raise AuditDataError(
            "alignment_representative_id_by_chain_id must be a mapping"
        )
    if atom_array is None:
        raise AuditDataError("atom_array is missing")
    explicit_query_only_chain_ids = _query_only_chain_ids(bioassembly)

    try:
        atom_chain_ids = np.asarray(atom_array.chain_id).astype(str)
        label_entity_ids = np.asarray(atom_array.label_entity_id).astype(str)
        label_asym_ids = np.asarray(atom_array.label_asym_id).astype(str)
    except Exception as error:
        raise AuditDataError(
            f"atom_array lacks chain/entity/asym annotations: {error}"
        ) from error
    if not (
        atom_chain_ids.ndim
        == label_entity_ids.ndim
        == label_asym_ids.ndim
        == 1
    ) or not (
        len(atom_chain_ids)
        == len(label_entity_ids)
        == len(label_asym_ids)
    ):
        raise AuditDataError("atom_array chain annotations have invalid shapes")

    contexts: list[dict[str, Any]] = []
    for raw_chain_id, raw_sequence in sorted(
        sequences_by_chain_id.items(),
        key=lambda item: str(item[0]),
    ):
        chain_id = str(raw_chain_id)
        sequence = str(raw_sequence)
        chain_indices = np.flatnonzero(atom_chain_ids == chain_id)
        if chain_indices.size == 0:
            raise AuditDataError(
                f"canonical chain {chain_id!r} is absent from atom_array"
            )
        entity_values = set(label_entity_ids[chain_indices].tolist())
        asym_values = set(label_asym_ids[chain_indices].tolist())
        if len(entity_values) != 1 or len(asym_values) != 1:
            raise AuditDataError(
                f"chain {chain_id!r} has non-unique entity/asym annotations"
            )
        entity_id = entity_values.pop()
        polymer_type = str(entity_poly_type.get(entity_id, ""))
        if polymer_type not in SUPPORTED_POLYMER_TYPES:
            continue
        if not sequence:
            sequence = str(entity_sequences.get(entity_id, ""))
        if not sequence:
            raise AuditDataError(
                f"chain {chain_id!r} has no canonical sequence"
            )
        representative_value = representatives.get(chain_id)
        representative_id = (
            "" if representative_value is None else str(representative_value)
        )
        contexts.append(
            {
                "chain_id": chain_id,
                "entity_id": entity_id,
                "label_asym_id": asym_values.pop(),
                "polymer_type": polymer_type,
                "canonical_query": sequence,
                "current_representative_id": representative_id,
                "explicit_query_only": (
                    chain_id in explicit_query_only_chain_ids
                ),
            }
        )
    audited_chain_ids = {context["chain_id"] for context in contexts}
    invalid_explicit_chain_ids = (
        explicit_query_only_chain_ids - audited_chain_ids
    )
    if invalid_explicit_chain_ids:
        raise AuditDataError(
            "openfold3_query_only_chain_ids references chains that are not "
            "valid audited protein/RNA chains: "
            f"{sorted(invalid_explicit_chain_ids)!r}"
        )
    return contexts


def _representative_public(
    value: RepresentativeQuery,
    *,
    alignment_dirs: Sequence[Path],
) -> dict[str, Any]:
    relative_path = None
    if value.path is not None:
        path = Path(value.path)
        for ordinal, directory in enumerate(alignment_dirs):
            try:
                relpath = path.relative_to(directory)
            except ValueError:
                continue
            relative_path = {
                "alignment_dir_ordinal": ordinal,
                "relative_path": relpath.as_posix(),
            }
            break
    return {
        "status": value.status,
        "path": relative_path,
        "query_length": value.query_length,
        "query_sha256": value.query_sha256,
        "source_names": list(value.source_names),
        "error_type": value.error_type,
        "error_message": value.error_message,
    }


def _classify_chain(
    *,
    task: AuditTask,
    context: Mapping[str, Any],
    declared_representatives: Sequence[str],
) -> dict[str, Any]:
    canonical_query = context["canonical_query"]
    current_id = context["current_representative_id"]
    base = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "record_type": "chain",
        "shard_index": task.shard_index,
        "shard_count": task.shard_count,
        "structure_ordinal": task.ordinal,
        "bioassembly_relpath": task.bioassembly_relpath,
        "pdb_id": task.pdb_id,
        "chain_id": context["chain_id"],
        "entity_id": context["entity_id"],
        "label_asym_id": context["label_asym_id"],
        "polymer_type": context["polymer_type"],
        "canonical_query_length": len(canonical_query),
        "canonical_query_sha256": _sha256_text(canonical_query),
        "current_representative_id": current_id or None,
        "declared_representative_count": len(declared_representatives),
        "matching_representative_ids": [],
        "proposed_representative_id": None,
        "candidate_errors": {},
        "current_representative": None,
        "explicit_query_only": bool(context["explicit_query_only"]),
    }
    if context["explicit_query_only"]:
        base.update(
            classification="explicit_query_only",
            reason=(
                "valid canonical chain is explicitly marked query-only; "
                "representative NPZ inspection was skipped"
            ),
        )
        return base
    if not current_id:
        base.update(
            classification="missing",
            reason="polymer chain has no declared representative ID",
        )
        return base

    current = _load_representative_query(
        current_id,
        context["polymer_type"],
    )
    base["current_representative"] = _representative_public(
        current,
        alignment_dirs=_WORKER_ALIGNMENT_DIRS,
    )
    if current.status == "missing":
        base.update(
            classification="missing",
            reason="current declared representative NPZ is missing",
        )
        return base
    if current.status != "valid":
        base.update(
            classification="corrupt",
            reason="current declared representative NPZ is corrupt",
        )
        return base
    if current.query == canonical_query:
        base["matching_representative_ids"] = [current_id]
        base.update(
            classification="exact_current",
            reason="current representative query exactly matches canonical query",
        )
        return base

    matches: list[str] = []
    candidate_errors: dict[str, dict[str, Optional[str]]] = {}
    statuses: Counter[str] = Counter()
    for representative_id in declared_representatives:
        candidate = _load_representative_query(
            representative_id,
            context["polymer_type"],
        )
        statuses[candidate.status] += 1
        if candidate.status == "valid" and candidate.query == canonical_query:
            matches.append(representative_id)
        elif candidate.status != "valid":
            candidate_errors[representative_id] = {
                "status": candidate.status,
                "error_type": candidate.error_type,
                "error_message": candidate.error_message,
            }
    matches.sort()
    base["matching_representative_ids"] = matches
    base["candidate_errors"] = candidate_errors
    if len(matches) == 1:
        base["proposed_representative_id"] = matches[0]
        base.update(
            classification="remappable_unique",
            reason="one declared representative exactly matches canonical query",
        )
    elif len(matches) > 1:
        base.update(
            classification="ambiguous",
            reason="multiple declared representatives exactly match canonical query",
        )
    elif statuses["corrupt"]:
        base.update(
            classification="corrupt",
            reason=(
                "no exact match and at least one declared candidate is corrupt"
            ),
        )
    elif statuses["missing"]:
        base.update(
            classification="missing",
            reason=(
                "no exact match and at least one declared candidate is missing"
            ),
        )
    else:
        base.update(
            classification="no_exact",
            reason="no declared representative exactly matches canonical query",
        )
    return base


def _corrupt_structure_record(
    task: AuditTask,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "record_type": "structure",
        "shard_index": task.shard_index,
        "shard_count": task.shard_count,
        "structure_ordinal": task.ordinal,
        "bioassembly_relpath": task.bioassembly_relpath,
        "pdb_id": task.pdb_id,
        "chain_id": None,
        "classification": "corrupt",
        "reason": "bioassembly pickle or structure metadata is corrupt",
        "error_type": type(error).__name__,
        "error_message": _short_error(error),
    }


def _audit_structure_task(task: AuditTask) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with gzip.open(task.bioassembly_path, "rb") as stream:
            bioassembly = pickle.load(stream)
        if not isinstance(bioassembly, Mapping):
            raise AuditDataError("bioassembly pickle must contain a mapping")
        if str(bioassembly.get("pdb_id", "")) != task.pdb_id:
            raise AuditDataError(
                f"bioassembly pdb_id {bioassembly.get('pdb_id')!r} "
                f"!= index pdb_id {task.pdb_id!r}"
            )
        contexts = _chain_contexts(bioassembly)
        declared_representatives_by_polymer_type: dict[str, tuple[str, ...]] = {}
        for polymer_type in SUPPORTED_POLYMER_TYPES:
            declared_representatives_by_polymer_type[polymer_type] = tuple(
                sorted(
                    {
                        context["current_representative_id"]
                        for context in contexts
                        if context["polymer_type"] == polymer_type
                        and not context["explicit_query_only"]
                        and context["current_representative_id"]
                    }
                )
            )
        records = [
            _classify_chain(
                task=task,
                context=context,
                declared_representatives=(
                    declared_representatives_by_polymer_type[
                        context["polymer_type"]
                    ]
                ),
            )
            for context in contexts
        ]
    except Exception as error:
        records = [_corrupt_structure_record(task, error)]
    return {
        "event": "structure_result",
        "schema_version": STATE_SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "task_id": task.task_id,
        "structure_ordinal": task.ordinal,
        "bioassembly_relpath": task.bioassembly_relpath,
        "pdb_id": task.pdb_id,
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }


def _atomic_json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_jsonl_dump(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for value in values:
                stream.write(
                    json.dumps(value, sort_keys=True, separators=(",", ":"))
                )
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _output_paths_outside_dataset(
    contract: DatasetContract,
    paths: Sequence[Path],
) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for path in paths:
        candidate = path.expanduser().absolute()
        if _is_relative_to(candidate.resolve(strict=False), contract.dataset_root):
            raise AuditConfigurationError(
                "audit outputs must be outside the read-only dataset root: "
                f"{candidate}"
            )
        resolved.append(candidate)
    if len(set(resolved)) != len(resolved):
        raise AuditConfigurationError("audit output paths must be distinct")
    return tuple(resolved)


def _inventory_sha256(tasks: Sequence[AuditTask]) -> str:
    digest = hashlib.sha256()
    for task in tasks:
        digest.update(
            _canonical_json_bytes(
                {
                    "ordinal": task.ordinal,
                    "pdb_id": task.pdb_id,
                    "bioassembly_relpath": task.bioassembly_relpath,
                    "task_id": task.task_id,
                }
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _run_contract(
    dataset: DatasetContract,
    tasks: Sequence[AuditTask],
    *,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "dataset_root": str(dataset.dataset_root),
        "manifest_sha256": dataset.manifest_sha256,
        "success_sha256": dataset.success_sha256,
        "index_sha256": dataset.index_sha256,
        "index_row_count": dataset.index_row_count,
        "declared_pickle_count": dataset.declared_pickle_count,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected_structure_count": len(tasks),
        "selected_inventory_sha256": _inventory_sha256(tasks),
        "sharding": "sorted_bioassembly_relpath_ordinal_mod_v1",
        "comparison": (
            "exact_canonical_query_string_same_polymer_candidates_"
            "explicit_query_only_v3"
        ),
    }


def _state_header(run_contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event": "audit_state_header",
        "schema_version": STATE_SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "run_contract": dict(run_contract),
        "run_contract_sha256": _sha256_bytes(
            _canonical_json_bytes(run_contract)
        ),
    }


def _iter_state_events(
    path: Path,
    *,
    tolerate_truncated_tail: bool,
) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        line_number = 0
        while True:
            line = stream.readline()
            if not line:
                return
            line_number += 1
            terminated = line.endswith(b"\n")
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                if tolerate_truncated_tail and not terminated and not stream.read(1):
                    return
                raise AuditDataError(
                    f"invalid state JSONL at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise AuditDataError(
                    f"state event must be an object at {path}:{line_number}"
                )
            yield value


def _read_state(
    state_path: Path,
    *,
    expected_header: Mapping[str, Any],
) -> set[str]:
    _repair_truncated_state_tail(state_path)
    events = _iter_state_events(
        state_path,
        tolerate_truncated_tail=False,
    )
    try:
        header = next(events)
    except StopIteration as error:
        raise AuditDataError(f"state journal is empty: {state_path}") from error
    if header != expected_header:
        raise AuditDataError(
            "state journal run contract differs from this audit invocation"
        )
    completed: set[str] = set()
    for event in events:
        if event.get("event") != "structure_result":
            raise AuditDataError(
                f"unexpected state event {event.get('event')!r}"
            )
        task_id = str(event.get("task_id", ""))
        if not task_id or task_id in completed:
            raise AuditDataError(
                f"missing or duplicate task_id in state journal: {task_id!r}"
            )
        records = event.get("records")
        if not isinstance(records, list):
            raise AuditDataError(
                f"state result {task_id!r} has invalid records"
            )
        completed.add(task_id)
    return completed


def _repair_truncated_state_tail(state_path: Path) -> None:
    """Remove only an invalid, unterminated final journal line.

    A process can be killed between writing a structure result and its newline.
    Earlier malformed lines remain fatal; silently skipping those could make a
    genuinely corrupt journal look resumable.
    """
    with state_path.open("r+b") as stream:
        line_number = 0
        while True:
            line_offset = stream.tell()
            line = stream.readline()
            if not line:
                return
            line_number += 1
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                if not line.endswith(b"\n") and not stream.read(1):
                    stream.seek(line_offset)
                    stream.truncate()
                    stream.flush()
                    os.fsync(stream.fileno())
                    return
                raise AuditDataError(
                    f"invalid state JSONL at {state_path}:{line_number}: "
                    f"{error}"
                ) from error
            if not isinstance(value, dict):
                raise AuditDataError(
                    "state event must be an object at "
                    f"{state_path}:{line_number}"
                )


def _initialize_state(
    state_path: Path,
    *,
    header: Mapping[str, Any],
) -> None:
    _atomic_jsonl_dump(state_path, [header])


def _iter_task_results(
    tasks: Sequence[AuditTask],
    *,
    workers: int,
    chunksize: int,
    alignment_dirs: Sequence[Path],
) -> Iterator[dict[str, Any]]:
    alignment_args = tuple(str(path) for path in alignment_dirs)
    if workers == 1:
        _init_worker(alignment_args)
        for task in tasks:
            yield _audit_structure_task(task)
        return
    context = multiprocessing.get_context("fork")
    with context.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(alignment_args,),
    ) as pool:
        yield from pool.imap_unordered(
            _audit_structure_task,
            tasks,
            chunksize=chunksize,
        )


def _publish_results_from_state(
    *,
    state_path: Path,
    output_jsonl: Path,
    summary_json: Path,
    expected_header: Mapping[str, Any],
    expected_task_ids: set[str],
    run_contract: Mapping[str, Any],
    workers: int,
    chunksize: int,
    checkpoint_every: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    completed: set[str] = set()
    classification_counts: Counter[str] = Counter()
    record_type_counts: Counter[str] = Counter()
    chain_record_count = 0
    structure_record_count = 0

    def iter_records() -> Iterator[dict[str, Any]]:
        nonlocal chain_record_count
        nonlocal structure_record_count
        events = _iter_state_events(
            state_path,
            tolerate_truncated_tail=False,
        )
        header = next(events)
        if header != expected_header:
            raise AuditDataError("state header changed during audit")
        for event in events:
            task_id = str(event.get("task_id", ""))
            if task_id in completed:
                raise AuditDataError(
                    f"duplicate completed task in state: {task_id!r}"
                )
            completed.add(task_id)
            for record in event["records"]:
                classification = str(record.get("classification", ""))
                if classification not in CLASSIFICATIONS:
                    raise AuditDataError(
                        f"unknown classification {classification!r}"
                    )
                record_type = str(record.get("record_type", ""))
                classification_counts[classification] += 1
                record_type_counts[record_type] += 1
                if record_type == "chain":
                    chain_record_count += 1
                elif record_type == "structure":
                    structure_record_count += 1
                yield record

    _atomic_jsonl_dump(output_jsonl, iter_records())
    if completed != expected_task_ids:
        output_jsonl.unlink(missing_ok=True)
        missing = len(expected_task_ids - completed)
        unexpected = len(completed - expected_task_ids)
        raise AuditDataError(
            "state journal does not exactly cover selected shard: "
            f"missing={missing}, unexpected={unexpected}"
        )

    summary = {
        "event": "audit_summary",
        "schema_version": RESULT_SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "status": "complete",
        "read_only_source_dataset": True,
        "run_contract": dict(run_contract),
        "selected_structure_count": len(expected_task_ids),
        "completed_structure_count": len(completed),
        "chain_record_count": chain_record_count,
        "structure_record_count": structure_record_count,
        "classification_counts": {
            name: classification_counts.get(name, 0)
            for name in CLASSIFICATIONS
        },
        "record_type_counts": dict(sorted(record_type_counts.items())),
        "all_exact_current": (
            chain_record_count > 0
            and classification_counts["exact_current"] == chain_record_count
            and structure_record_count == 0
        ),
        "parameters": {
            "workers": workers,
            "chunksize": chunksize,
            "checkpoint_every": checkpoint_every,
        },
        "outputs": {
            "result_jsonl": str(output_jsonl),
            "state_jsonl": str(state_path),
            "summary_json": str(summary_json),
        },
        "elapsed_seconds": elapsed_seconds,
    }
    _atomic_json_dump(summary_json, summary)
    return summary


def run_audit(
    *,
    dataset_root: Path,
    output_jsonl: Path,
    summary_json: Path,
    state_jsonl: Optional[Path] = None,
    workers: int = 1,
    chunksize: int = 1,
    checkpoint_every: int = 32,
    shard_index: int = 0,
    shard_count: int = 1,
    resume: bool = False,
) -> dict[str, Any]:
    """Run one read-only shard and atomically publish its audit artifacts."""
    started = time.perf_counter()
    if workers < 1:
        raise AuditConfigurationError("workers must be >= 1")
    if chunksize < 1:
        raise AuditConfigurationError("chunksize must be >= 1")
    if checkpoint_every < 1:
        raise AuditConfigurationError("checkpoint_every must be >= 1")

    dataset = validate_disordered_dataset_root(dataset_root)
    raw_state_path = (
        state_jsonl
        if state_jsonl is not None
        else Path(f"{output_jsonl}.state.jsonl")
    )
    output_jsonl, summary_json, state_path = _output_paths_outside_dataset(
        dataset,
        (output_jsonl, summary_json, raw_state_path),
    )
    inventory = read_disordered_inventory(dataset)
    tasks = build_tasks(
        dataset,
        inventory,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    if not tasks:
        raise AuditConfigurationError(
            "selected shard contains no disorder bioassemblies"
        )
    run_contract = _run_contract(
        dataset,
        tasks,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    header = _state_header(run_contract)

    if state_path.exists():
        if not resume:
            raise AuditConfigurationError(
                f"state journal exists; pass --resume or remove it: {state_path}"
            )
        completed = _read_state(state_path, expected_header=header)
    else:
        _initialize_state(state_path, header=header)
        completed = set()

    task_by_id = {task.task_id: task for task in tasks}
    if not completed.issubset(task_by_id):
        raise AuditDataError(
            "state journal contains tasks outside the selected shard"
        )
    pending = [task for task in tasks if task.task_id not in completed]
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("a", encoding="utf-8", buffering=1024 * 1024) as stream:
        since_checkpoint = 0
        try:
            for result in _iter_task_results(
                pending,
                workers=workers,
                chunksize=chunksize,
                alignment_dirs=dataset.alignment_dirs,
            ):
                stream.write(
                    json.dumps(
                        result,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                stream.write("\n")
                since_checkpoint += 1
                if since_checkpoint >= checkpoint_every:
                    stream.flush()
                    os.fsync(stream.fileno())
                    since_checkpoint = 0
        finally:
            stream.flush()
            os.fsync(stream.fileno())

    return _publish_results_from_state(
        state_path=state_path,
        output_jsonl=output_jsonl,
        summary_json=summary_json,
        expected_header=header,
        expected_task_ids=set(task_by_id),
        run_contract=run_contract,
        workers=workers,
        chunksize=chunksize,
        checkpoint_every=checkpoint_every,
        elapsed_seconds=time.perf_counter() - started,
    )


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only, shardable audit restricted to "
            "openfold3_disordered_v20260725"
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "OpenFold3-disordered dataset root containing protenix/, "
            "structures/, and alignment_arrays/; arbitrary dataset inputs "
            "are rejected"
        ),
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument(
        "--state-jsonl",
        type=Path,
        help="resumable state journal (default: OUTPUT.state.jsonl)",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=max(1, min(128, os.cpu_count() or 1)),
    )
    parser.add_argument("--chunksize", type=_positive_int, default=1)
    parser.add_argument(
        "--checkpoint-every",
        type=_positive_int,
        default=32,
        help="fsync state after this many completed structures",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=_positive_int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed structures in the matching state journal",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_audit(
            dataset_root=args.dataset_root,
            output_jsonl=args.output_jsonl,
            summary_json=args.summary_json,
            state_jsonl=args.state_jsonl,
            workers=args.workers,
            chunksize=args.chunksize,
            checkpoint_every=args.checkpoint_every,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            resume=args.resume,
        )
    except (AuditError, OSError) as error:
        print(
            f"OpenFold3-disordered representative audit failed: {error}",
            file=sys.stderr,
        )
        return 2
    except Exception:
        traceback.print_exc()
        return 3
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
