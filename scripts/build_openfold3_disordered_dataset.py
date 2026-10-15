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

"""Build the Protenix view of the OpenFold3 disordered PDB dataset.

The official metadata cache is the inventory authority.  For every cache entry
this program converts ``<structures-root>/<pdb>/<pdb>.npz`` into
``<structures-root>/<pdb>/bioassembly/<pdb>.pkl.gz``.  Per-entry state stored
beside the pickle makes interrupted runs resumable without reparsing completed
structures.

The training index and audit artifacts are written below the dataset-level
``protenix`` directory.  ``_SUCCESS.json`` is published only after every
official entry, pickle, index row, and required alignment representative has
been validated.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import multiprocessing
import os
import pickle
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    sys.path.remove(str(REPO_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 3
EXPECTED_CACHE_ENTRY_COUNT = 28_567
STATE_SCHEMA_VERSION = 3

STANDARD_INDEX_FIELDS = [
    "entity_1_id",
    "chain_1_id",
    "mol_1_type",
    "cluster_1_id",
    "entity_2_id",
    "chain_2_id",
    "mol_2_type",
    "cluster_2_id",
    "cluster_id",
    "pdb_id",
    "release_date",
    "num_tokens",
    "num_prot_chains",
    "resolution",
    "type",
    "mol_type_group",
    "sub_mol_1_type",
    "sub_mol_2_type",
    "eval_type",
]
REQUIRED_INDEX_FIELDS = {
    "entity_1_id",
    "chain_1_id",
    "mol_1_type",
    "cluster_id",
    "pdb_id",
    "num_tokens",
    "type",
}
BUILDER_INDEX_FIELDS = [
    "cluster_size",
    "weights",
    "bioassembly_relpath",
]
FAILURE_FIELDS = [
    "pdb_id",
    "stage",
    "source_path",
    "error_type",
    "message",
]
ALIGNMENT_FIELDS = [
    "representative_id",
    "representative_pdb_id",
    "label_asym_id",
    "status",
    "alignment_dir_ordinal",
    "filename",
    "path",
    "size_bytes",
    "mtime_ns",
    "consumer_structure_count",
    "reference_count",
]

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9]+$")
_MOL_ALPHA = {
    "prot": 3.0,
    "protein": 3.0,
    "nuc": 3.0,
    "nucleic": 3.0,
    "nucleic_acid": 3.0,
    "dna": 3.0,
    "rna": 3.0,
    "ligand": 1.0,
    "lig": 1.0,
    "ion": 1.0,
    "ions": 1.0,
}


class DatasetBuildError(RuntimeError):
    """Raised when a dataset-level invariant is violated."""


def _json_normalize(value: Any) -> Any:
    """Convert NumPy-like scalars and containers into JSON values."""

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_normalize(item())
        except ValueError:
            pass
    if isinstance(value, Mapping):
        return {
            str(key): _json_normalize(item_value)
            for key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item_value) for item_value in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_normalize(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb", buffering=8 * 1024 * 1024) as stream:
        while block := stream.read(8 * 1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _source_stat_unchanged(
    before: os.stat_result,
    after: os.stat_result,
) -> bool:
    """Ignore read-side atime updates while detecting source replacement."""

    return (
        before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )


def _artifact_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _atomic_bytes(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _atomic_json(value: Any, path: Path) -> None:
    payload = json.dumps(
        _json_normalize(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    _atomic_bytes(payload + b"\n", path)


def _deterministic_gzip(payload: bytes, *, compresslevel: int) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(
        fileobj=output,
        mode="wb",
        compresslevel=compresslevel,
        filename="",
        mtime=0,
    ) as stream:
        stream.write(payload)
    return output.getvalue()


def _atomic_gzip_pickle(
    value: Any,
    path: Path,
    *,
    compresslevel: int,
) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_bytes(
        _deterministic_gzip(payload, compresslevel=compresslevel),
        path,
    )


def _csv_value(value: Any) -> Any:
    value = _json_normalize(value)
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return value


def _csv_payload(
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
    *,
    delimiter: str,
) -> bytes:
    raw = io.StringIO(newline="")
    writer = csv.DictWriter(
        raw,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
        quoting=csv.QUOTE_NONNUMERIC,
        delimiter=delimiter,
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {field: _csv_value(row.get(field, "")) for field in fieldnames}
        )
    return raw.getvalue().encode("utf-8")


def _atomic_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    fieldnames: Sequence[str],
    *,
    compresslevel: int,
    delimiter: str = ",",
) -> None:
    payload = _csv_payload(rows, fieldnames, delimiter=delimiter)
    if path.suffix == ".gz":
        payload = _deterministic_gzip(
            payload,
            compresslevel=compresslevel,
        )
    _atomic_bytes(payload, path)


def _validate_component(value: object, *, label: str) -> str:
    text = str(value)
    if not text or not _SAFE_COMPONENT.fullmatch(text):
        raise DatasetBuildError(f"Unsafe {label}: {text!r}")
    return text


def _load_cache(
    cache_path: Path,
    *,
    expected_entry_count: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    cache_path = cache_path.expanduser().resolve(strict=True)
    raw_sha256 = _sha256_file(cache_path)
    try:
        with cache_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetBuildError(f"Cannot read cache {cache_path}: {error}") from error

    if not isinstance(document, dict):
        raise DatasetBuildError(f"{cache_path} must contain a JSON object")
    structure_data = document.get("structure_data")
    if not isinstance(structure_data, dict) or not structure_data:
        raise DatasetBuildError(
            f"{cache_path} does not contain non-empty structure_data"
        )
    if len(structure_data) != expected_entry_count:
        raise DatasetBuildError(
            f"{cache_path} has {len(structure_data)} structure_data entries; "
            f"expected {expected_entry_count}"
        )

    normalized: dict[str, dict[str, Any]] = {}
    for raw_pdb_id, raw_entry in structure_data.items():
        pdb_id = _validate_component(raw_pdb_id, label="PDB ID")
        if not isinstance(raw_entry, dict):
            raise DatasetBuildError(
                f"{cache_path}: structure_data[{pdb_id!r}] must be an object"
            )
        chains = raw_entry.get("chains")
        interfaces = raw_entry.get("interfaces")
        if not isinstance(chains, dict) or not isinstance(interfaces, dict):
            raise DatasetBuildError(
                f"{cache_path}: {pdb_id} requires chains and interfaces objects"
            )
        if pdb_id in normalized:
            raise DatasetBuildError(f"Duplicate PDB ID in cache: {pdb_id}")
        normalized[pdb_id] = raw_entry

    cache_info = {
        "path": str(cache_path),
        "size_bytes": cache_path.stat().st_size,
        "sha256": raw_sha256,
        "entry_count": len(normalized),
        "cache_type": document.get("_type"),
        "cache_name": document.get("name"),
    }
    return normalized, cache_info


def _entry_representatives(
    pdb_id: str,
    entry: Mapping[str, Any],
) -> dict[str, str]:
    chains = entry.get("chains")
    if not isinstance(chains, Mapping):
        raise DatasetBuildError(f"{pdb_id}: cache entry has invalid chains")
    representatives: dict[str, str] = {}
    for raw_chain_id, chain in chains.items():
        chain_id = _validate_component(raw_chain_id, label="cache chain ID")
        if not isinstance(chain, Mapping):
            raise DatasetBuildError(
                f"{pdb_id}: chain {chain_id!r} metadata is not an object"
            )
        representative = chain.get("alignment_representative_id")
        if representative is None or str(representative) == "":
            continue
        representative = str(representative)
        try:
            representative_pdb_id, label_asym_id = representative.rsplit(
                "_",
                1,
            )
        except ValueError as error:
            raise DatasetBuildError(
                f"{pdb_id}: invalid alignment representative "
                f"{representative!r}"
            ) from error
        _validate_component(
            representative_pdb_id,
            label="alignment representative PDB ID",
        )
        _validate_component(
            label_asym_id,
            label="alignment representative label_asym_id",
        )
        representatives[chain_id] = representative
    return representatives


def _alignment_candidate_names(representative: str) -> tuple[str, ...]:
    try:
        pdb_id, label_asym_id = representative.rsplit("_", 1)
    except ValueError as error:
        raise DatasetBuildError(
            f"Invalid alignment representative ID: {representative!r}"
        ) from error
    _validate_component(pdb_id, label="alignment representative PDB ID")
    _validate_component(
        label_asym_id,
        label="alignment representative label_asym_id",
    )
    pdb_candidates = tuple(dict.fromkeys((pdb_id, pdb_id.lower(), pdb_id.upper())))
    return tuple(
        f"{candidate}_{label_asym_id}.npz"
        for candidate in pdb_candidates
    )


def _find_alignment_path(
    representative: str,
    alignment_dirs: Sequence[Path],
) -> Path | None:
    for directory in alignment_dirs:
        for filename in _alignment_candidate_names(representative):
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    return None


def _load_entry_alignment_queries(
    *,
    pdb_id: str,
    cache_entry: Mapping[str, Any],
    alignment_dirs: Sequence[Path],
) -> dict[str, str]:
    """Load exact query rows used to validate and repair chain mappings."""
    representatives = sorted(
        set(_entry_representatives(pdb_id, cache_entry).values())
    )
    queries: dict[str, str] = {}
    for representative in representatives:
        path = _find_alignment_path(representative, alignment_dirs)
        if path is None:
            raise DatasetBuildError(
                f"{pdb_id}: cannot remap chain sequences because alignment "
                f"{representative!r} is missing"
            )
        source_queries = set()
        try:
            with np.load(path, allow_pickle=True) as archive:
                for source in archive.files:
                    payload = archive[source].item()
                    msa = np.asarray(payload["msa"])
                    if msa.ndim != 2 or msa.shape[0] == 0:
                        raise ValueError(
                            f"{source} MSA has invalid shape {msa.shape}"
                        )
                    source_queries.add(
                        "".join(np.asarray(msa[0]).astype(str).tolist())
                    )
        except (OSError, ValueError, KeyError, AttributeError) as error:
            raise DatasetBuildError(
                f"{pdb_id}: invalid alignment archive for "
                f"{representative!r}: {path}: {error}"
            ) from error
        if len(source_queries) != 1 or not next(iter(source_queries), ""):
            raise DatasetBuildError(
                f"{pdb_id}: alignment sources for {representative!r} do not "
                "share one non-empty query"
            )
        queries[representative] = source_queries.pop()
    return queries


def _normalize_alignment_dirs(
    alignment_dirs: Sequence[Path],
) -> tuple[Path, ...]:
    directories: list[Path] = []
    seen: set[Path] = set()
    for directory in alignment_dirs:
        resolved = directory.expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise DatasetBuildError(
                f"Alignment lookup path is not a directory: {resolved}"
            )
        if resolved not in seen:
            seen.add(resolved)
            directories.append(resolved)
    if not directories:
        raise DatasetBuildError("At least one --alignment-dir is required")
    return tuple(directories)


def _alignment_inventory(
    selected_entries: Mapping[str, Mapping[str, Any]],
    alignment_dirs: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    structure_consumers: dict[str, set[str]] = defaultdict(set)
    reference_counts: Counter[str] = Counter()
    for pdb_id in sorted(selected_entries):
        for representative in _entry_representatives(
            pdb_id,
            selected_entries[pdb_id],
        ).values():
            structure_consumers[representative].add(pdb_id)
            reference_counts[representative] += 1

    inventory: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for representative in sorted(reference_counts):
        representative_pdb_id, label_asym_id = representative.rsplit("_", 1)
        found_path: Path | None = None
        found_ordinal: int | None = None
        found_filename = ""
        for ordinal, directory in enumerate(alignment_dirs):
            found_path = _find_alignment_path(representative, (directory,))
            if found_path is not None:
                found_ordinal = ordinal
                found_filename = found_path.name
                break

        if found_path is None:
            record = {
                "representative_id": representative,
                "representative_pdb_id": representative_pdb_id,
                "label_asym_id": label_asym_id,
                "status": "missing",
                "alignment_dir_ordinal": "",
                "filename": "",
                "path": "",
                "size_bytes": "",
                "mtime_ns": "",
                "consumer_structure_count": len(
                    structure_consumers[representative]
                ),
                "reference_count": reference_counts[representative],
            }
            failures.append(
                {
                    "pdb_id": ",".join(
                        sorted(structure_consumers[representative])
                    ),
                    "stage": "alignment",
                    "source_path": representative,
                    "error_type": "MissingAlignmentRepresentative",
                    "message": (
                        "No exact case-sensitive chain alignment was found "
                        "in the configured lookup directories"
                    ),
                }
            )
        else:
            stat = found_path.stat()
            record = {
                "representative_id": representative,
                "representative_pdb_id": representative_pdb_id,
                "label_asym_id": label_asym_id,
                "status": "resolved",
                "alignment_dir_ordinal": found_ordinal,
                "filename": found_filename,
                "path": str(found_path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "consumer_structure_count": len(
                    structure_consumers[representative]
                ),
                "reference_count": reference_counts[representative],
            }
        inventory.append(record)
    return inventory, failures


def _validate_bioassembly(
    value: Any,
    *,
    pdb_id: str,
    expected_representatives: Mapping[str, str],
) -> None:
    if not isinstance(value, dict):
        raise DatasetBuildError(
            f"{pdb_id}: bioassembly must be a dictionary"
        )
    if str(value.get("pdb_id", "")) != pdb_id:
        raise DatasetBuildError(
            f"{pdb_id}: bioassembly pdb_id is {value.get('pdb_id')!r}"
        )
    atom_array = value.get("atom_array")
    token_array = value.get("token_array")
    if atom_array is None or len(atom_array) == 0:
        raise DatasetBuildError(f"{pdb_id}: bioassembly atom_array is empty")
    if token_array is None or len(token_array) == 0:
        raise DatasetBuildError(f"{pdb_id}: bioassembly token_array is empty")
    try:
        num_tokens = int(value.get("num_tokens"))
    except (TypeError, ValueError) as error:
        raise DatasetBuildError(
            f"{pdb_id}: bioassembly num_tokens is invalid"
        ) from error
    if num_tokens != len(token_array):
        raise DatasetBuildError(
            f"{pdb_id}: num_tokens={num_tokens} but token_array has "
            f"{len(token_array)} tokens"
        )

    actual_representatives = value.get(
        "alignment_representative_id_by_chain_id"
    )
    actual_chain_ids = {
        str(chain_id) for chain_id in np.unique(atom_array.chain_id)
    }
    retained_expected_representatives = {
        chain_id: representative
        for chain_id, representative in expected_representatives.items()
        if chain_id in actual_chain_ids
    }
    if retained_expected_representatives:
        if not isinstance(actual_representatives, Mapping):
            raise DatasetBuildError(
                f"{pdb_id}: bioassembly is missing alignment representatives"
            )
        allowed_representatives = set(expected_representatives.values())
        for chain_id in retained_expected_representatives:
            representative = str(actual_representatives.get(chain_id, ""))
            if representative not in allowed_representatives:
                raise DatasetBuildError(
                    f"{pdb_id}: chain {chain_id!r} alignment representative "
                    f"{representative!r} is not declared by the cache entry"
                )


def _load_and_validate_pickle(
    path: Path,
    *,
    pdb_id: str,
    expected_representatives: Mapping[str, str],
) -> dict[str, Any]:
    with gzip.open(path, "rb") as stream:
        value = pickle.load(stream)
    _validate_bioassembly(
        value,
        pdb_id=pdb_id,
        expected_representatives=expected_representatives,
    )
    return value


def _build_with_core(
    source_path: Path,
    *,
    pdb_id: str,
    cache_entry: Mapping[str, Any],
    alignment_dirs: Sequence[Path] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Thin adapter around the OpenFold3 disordered core converter."""

    from protenix.data.core.openfold3_disordered import (
        OpenFold3DisorderedError,
        build_openfold3_disordered_data,
    )

    canonical_queries = None
    if alignment_dirs:
        canonical_queries = _load_entry_alignment_queries(
            pdb_id=pdb_id,
            cache_entry=cache_entry,
            alignment_dirs=alignment_dirs,
        )
    build_kwargs = {
        "pdb_id": pdb_id,
        "cache_entry": cache_entry,
        "ccd_mols": None,
    }
    if canonical_queries is not None:
        build_kwargs["canonical_sequences_by_representative"] = (
            canonical_queries
        )
    try:
        result = build_openfold3_disordered_data(
            source_path,
            **build_kwargs,
        )
    except OpenFold3DisorderedError as error:
        if (
            canonical_queries is not None
            or "inconsistent sequences across chain copies" not in str(error)
        ):
            raise
        # Preserve the legacy no-alignment-directory behavior.  Production
        # dataset builds provide alignment_dirs and therefore load canonical
        # queries before the first core conversion attempt above.
        canonical_queries = _load_entry_alignment_queries(
            pdb_id=pdb_id,
            cache_entry=cache_entry,
            alignment_dirs=alignment_dirs,
        )
        result = build_openfold3_disordered_data(
            source_path,
            pdb_id=pdb_id,
            cache_entry=cache_entry,
            canonical_sequences_by_representative=canonical_queries,
            ccd_mols=None,
        )
    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or not isinstance(result[0], list)
        or not isinstance(result[1], dict)
    ):
        raise DatasetBuildError(
            f"{pdb_id}: core converter returned an invalid result"
        )
    return result


def _resume_result(
    *,
    pdb_id: str,
    source_path: Path,
    source_sha256: str,
    cache_entry_sha256: str,
    output_path: Path,
    state_path: Path,
    expected_representatives: Mapping[str, str],
    structures_root: Path,
) -> dict[str, Any] | None:
    if not output_path.is_file() or not state_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != STATE_SCHEMA_VERSION
        or state.get("pdb_id") != pdb_id
        or state.get("source_sha256") != source_sha256
        or state.get("cache_entry_sha256") != cache_entry_sha256
    ):
        return None
    index_rows = state.get("index_rows")
    if not isinstance(index_rows, list) or not index_rows:
        return None

    _load_and_validate_pickle(
        output_path,
        pdb_id=pdb_id,
        expected_representatives=expected_representatives,
    )
    output_sha256 = _sha256_file(output_path)
    if state.get("output_sha256") != output_sha256:
        return None
    return {
        "status": "resumed",
        "pdb_id": pdb_id,
        "source_path": str(source_path),
        "source_relpath": source_path.relative_to(structures_root).as_posix(),
        "source_size_bytes": source_path.stat().st_size,
        "source_sha256": source_sha256,
        "output_path": str(output_path),
        "output_relpath": output_path.relative_to(structures_root).as_posix(),
        "output_size_bytes": output_path.stat().st_size,
        "output_sha256": output_sha256,
        "index_rows": index_rows,
    }


def _convert_entry(task: Mapping[str, Any]) -> dict[str, Any]:
    pdb_id = str(task["pdb_id"])
    structures_root = Path(str(task["structures_root"]))
    source_path = structures_root / pdb_id / f"{pdb_id}.npz"
    output_path = (
        structures_root
        / pdb_id
        / "bioassembly"
        / f"{pdb_id}.pkl.gz"
    )
    state_path = output_path.with_name(f"{pdb_id}.build.json")
    source_sha256 = ""
    try:
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Missing official structure archive: {source_path}"
            )
        source_stat = source_path.stat()
        source_sha256 = _sha256_file(source_path)
        cache_entry = task["cache_entry"]
        cache_entry_sha256 = _sha256_bytes(
            _canonical_json_bytes(cache_entry)
        )
        expected_representatives = _entry_representatives(
            pdb_id,
            cache_entry,
        )

        if bool(task["resume"]):
            resumed = _resume_result(
                pdb_id=pdb_id,
                source_path=source_path,
                source_sha256=source_sha256,
                cache_entry_sha256=cache_entry_sha256,
                output_path=output_path,
                state_path=state_path,
                expected_representatives=expected_representatives,
                structures_root=structures_root,
            )
            if resumed is not None:
                return resumed

        index_rows, bioassembly = _build_with_core(
            source_path,
            pdb_id=pdb_id,
            cache_entry=cache_entry,
            alignment_dirs=tuple(
                Path(path) for path in task["alignment_dirs"]
            ),
        )
        if not index_rows or not all(
            isinstance(row, Mapping) for row in index_rows
        ):
            raise DatasetBuildError(
                f"{pdb_id}: core converter returned no index rows"
            )

        bioassembly["pdb_id"] = pdb_id
        bioassembly["source_npz_relpath"] = source_path.relative_to(
            structures_root
        ).as_posix()
        bioassembly["openfold3_cache_entry_sha256"] = cache_entry_sha256
        _validate_bioassembly(
            bioassembly,
            pdb_id=pdb_id,
            expected_representatives=expected_representatives,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_gzip_pickle(
            bioassembly,
            output_path,
            compresslevel=int(task["compresslevel"]),
        )
        _load_and_validate_pickle(
            output_path,
            pdb_id=pdb_id,
            expected_representatives=expected_representatives,
        )
        output_sha256 = _sha256_file(output_path)
        if not _source_stat_unchanged(source_stat, source_path.stat()):
            raise DatasetBuildError(
                f"{pdb_id}: source archive changed during conversion"
            )

        normalized_rows = [
            _json_normalize(dict(row)) for row in index_rows
        ]
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "pdb_id": pdb_id,
            "source_relpath": source_path.relative_to(
                structures_root
            ).as_posix(),
            "source_size_bytes": source_stat.st_size,
            "source_sha256": source_sha256,
            "cache_entry_sha256": cache_entry_sha256,
            "output_relpath": output_path.relative_to(
                structures_root
            ).as_posix(),
            "output_size_bytes": output_path.stat().st_size,
            "output_sha256": output_sha256,
            "index_rows": normalized_rows,
        }
        _atomic_json(state, state_path)
        return {
            "status": "converted",
            "pdb_id": pdb_id,
            "source_path": str(source_path),
            "source_relpath": state["source_relpath"],
            "source_size_bytes": source_stat.st_size,
            "source_sha256": source_sha256,
            "output_path": str(output_path),
            "output_relpath": state["output_relpath"],
            "output_size_bytes": state["output_size_bytes"],
            "output_sha256": output_sha256,
            "index_rows": normalized_rows,
        }
    except Exception as error:
        return {
            "status": "failed",
            "pdb_id": pdb_id,
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _run_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    fail_fast: bool,
) -> tuple[list[dict[str, Any]], int]:
    if workers == 1:
        results = []
        for task in tasks:
            result = _convert_entry(task)
            results.append(result)
            if fail_fast and result["status"] == "failed":
                break
        return results, len(tasks) - len(results)

    results: list[dict[str, Any]] = []
    next_task = 0
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("fork"),
    )
    futures: dict[Any, int] = {}
    stop = False
    try:
        window = max(workers, workers * 2)
        while next_task < len(tasks) and len(futures) < window:
            future = executor.submit(_convert_entry, tasks[next_task])
            futures[future] = next_task
            next_task += 1

        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=lambda item: futures[item]):
                futures.pop(future)
                result = future.result()
                results.append(result)
                if result["status"] == "failed":
                    print(
                        "conversion_failure "
                        f"pdb_id={result.get('pdb_id', '')} "
                        f"error_type={result.get('error_type', '')} "
                        f"error={result.get('error', '')}",
                        flush=True,
                    )
                if len(results) % 250 == 0 or len(results) == len(tasks):
                    successful = sum(
                        item["status"] != "failed" for item in results
                    )
                    print(
                        "conversion_progress="
                        f"{len(results)}/{len(tasks)} "
                        f"successful={successful} "
                        f"failed={len(results) - successful}",
                        flush=True,
                    )
                if fail_fast and result["status"] == "failed":
                    stop = True
            if stop:
                for future in futures:
                    future.cancel()
                break
            while next_task < len(tasks) and len(futures) < window:
                future = executor.submit(_convert_entry, tasks[next_task])
                futures[future] = next_task
                next_task += 1
    finally:
        executor.shutdown(wait=True, cancel_futures=stop)

    attempted_ids = {str(result["pdb_id"]) for result in results}
    # Running tasks may have completed during fail-fast shutdown. Their files
    # remain resumable, but only collected results enter this partial manifest.
    not_attempted = len(tasks) - len(attempted_ids)
    return results, not_attempted


def _prewarm_ccd_for_fork() -> dict[str, Any]:
    """Load CCD assets once so forked workers share them copy-on-write."""
    from protenix.data.core import ccd

    components_path = Path(str(ccd.COMPONENTS_FILE)).expanduser().resolve(
        strict=False
    )
    rdkit_path = Path(ccd.RKDIT_MOL_PKL).expanduser().resolve(strict=False)
    missing = [
        str(path)
        for path in (components_path, rdkit_path)
        if not path.is_file()
    ]
    if missing:
        raise DatasetBuildError(
            "CCD preflight failed; set PROTENIX_ROOT_DIR to a complete "
            f"Protenix data root. Missing: {missing}"
        )

    started = time.monotonic()
    ccd.get_component_rdkit_mol("ALA")
    ccd.biotite_load_ccd_cif()
    return {
        "components_file": str(components_path),
        "components_size_bytes": components_path.stat().st_size,
        "rdkit_mol_file": str(rdkit_path),
        "rdkit_mol_size_bytes": rdkit_path.stat().st_size,
        "preload_seconds": time.monotonic() - started,
        "process_start_method": "fork",
    }


def _cache_chain_metadata(
    entry: Mapping[str, Any],
    *,
    chain_id: str,
    entity_id: str,
) -> Mapping[str, Any]:
    chains = entry["chains"]
    direct = chains.get(chain_id)
    if isinstance(direct, Mapping):
        return direct
    candidates = [
        chain
        for chain in chains.values()
        if isinstance(chain, Mapping)
        and str(chain.get("entity_id", "")) == entity_id
    ]
    cluster_pairs = {
        (
            str(chain.get("cluster_id", "")),
            int(chain.get("cluster_size", 0)),
        )
        for chain in candidates
    }
    if len(cluster_pairs) == 1 and candidates:
        return candidates[0]
    raise DatasetBuildError(
        f"Cannot map chain {chain_id!r}, entity {entity_id!r} "
        "to official cache metadata"
    )


def _cache_interface_metadata(
    entry: Mapping[str, Any],
    *,
    chain_1_id: str,
    chain_2_id: str,
    cluster_id: str,
) -> Mapping[str, Any]:
    interfaces = entry["interfaces"]
    for key in (
        f"{chain_1_id}_{chain_2_id}",
        f"{chain_2_id}_{chain_1_id}",
    ):
        direct = interfaces.get(key)
        if isinstance(direct, Mapping):
            return direct
    candidates = [
        interface
        for interface in interfaces.values()
        if isinstance(interface, Mapping)
        and str(interface.get("cluster_id", "")) == cluster_id
    ]
    sizes = {int(interface.get("cluster_size", 0)) for interface in candidates}
    if len(sizes) == 1 and candidates:
        return candidates[0]
    raise DatasetBuildError(
        f"Cannot map interface {chain_1_id!r}/{chain_2_id!r} "
        "to official cache metadata"
    )


def _official_cluster_metadata(
    row: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> tuple[str, int]:
    row_type = str(row.get("type", ""))
    chain_1_id = str(row.get("chain_1_id", ""))
    chain_2_id = str(row.get("chain_2_id", ""))
    cluster_id = str(row.get("cluster_id", ""))
    if row_type == "chain":
        metadata = _cache_chain_metadata(
            entry,
            chain_id=chain_1_id,
            entity_id=str(row.get("entity_1_id", "")),
        )
    elif row_type == "interface":
        metadata = _cache_interface_metadata(
            entry,
            chain_1_id=chain_1_id,
            chain_2_id=chain_2_id,
            cluster_id=cluster_id,
        )
    else:
        raise DatasetBuildError(f"Unsupported index row type {row_type!r}")

    official_cluster_id = str(metadata.get("cluster_id", ""))
    try:
        cluster_size = int(metadata.get("cluster_size"))
    except (TypeError, ValueError) as error:
        raise DatasetBuildError(
            f"Invalid official cluster_size for row {row!r}"
        ) from error
    if not official_cluster_id or cluster_size <= 0:
        raise DatasetBuildError(
            f"Invalid official cluster metadata for row {row!r}"
        )
    if cluster_id and cluster_id != official_cluster_id:
        raise DatasetBuildError(
            f"Core cluster_id {cluster_id!r} does not match official "
            f"cache value {official_cluster_id!r}"
        )
    return official_cluster_id, cluster_size


def _row_occurrence_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    entity_1 = str(row.get("entity_1_id", "") or "")
    entity_2 = str(row.get("entity_2_id", "") or "")
    low, high = sorted((entity_1, entity_2))
    assembly_id = str(row.get("assembly_id", "") or "")
    return assembly_id, low, high


def _mol_alpha(value: Any) -> float:
    normalized = str(value or "").strip().lower()
    if normalized not in _MOL_ALPHA:
        raise DatasetBuildError(
            f"Unsupported molecule type for weighting: {value!r}"
        )
    return _MOL_ALPHA[normalized]


def _weight_index_rows(
    *,
    pdb_id: str,
    raw_rows: Sequence[Mapping[str, Any]],
    cache_entry: Mapping[str, Any],
    bioassembly_relpath: str,
) -> list[dict[str, Any]]:
    if not raw_rows:
        raise DatasetBuildError(f"{pdb_id}: no index rows to weight")
    rows = [dict(row) for row in raw_rows]
    occurrence_counts = Counter(_row_occurrence_key(row) for row in rows)
    weighted: list[dict[str, Any]] = []
    for row in rows:
        missing = REQUIRED_INDEX_FIELDS - set(row)
        if missing:
            raise DatasetBuildError(
                f"{pdb_id}: index row is missing {sorted(missing)}"
            )
        if str(row.get("pdb_id", pdb_id)) not in ("", pdb_id):
            raise DatasetBuildError(
                f"{pdb_id}: core index row has a different pdb_id"
            )
        row["pdb_id"] = pdb_id
        official_cluster_id, cluster_size = _official_cluster_metadata(
            row,
            cache_entry,
        )
        row["cluster_id"] = official_cluster_id
        row["cluster_size"] = cluster_size

        row_type = str(row["type"])
        beta = 0.5 if row_type == "chain" else 1.0
        alpha_sum = _mol_alpha(row["mol_1_type"])
        if row_type == "interface":
            alpha_sum += _mol_alpha(row.get("mol_2_type"))
        occurrence_count = occurrence_counts[_row_occurrence_key(row)]
        weight = beta * alpha_sum / cluster_size / occurrence_count
        if not math.isfinite(weight) or weight <= 0:
            raise DatasetBuildError(
                f"{pdb_id}: non-positive or non-finite weight {weight}"
            )
        row["weights"] = weight
        row["bioassembly_relpath"] = bioassembly_relpath
        weighted.append(row)

    return sorted(
        weighted,
        key=lambda row: (
            str(row.get("type", "")),
            str(row.get("chain_1_id", "")),
            str(row.get("chain_2_id", "")),
            str(row.get("entity_1_id", "")),
            str(row.get("entity_2_id", "")),
            str(row.get("cluster_id", "")),
            _canonical_json_bytes(row),
        ),
    )


def _index_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    observed = {str(key) for row in rows for key in row}
    ordered = []
    for field in STANDARD_INDEX_FIELDS + BUILDER_INDEX_FIELDS:
        if field in observed and field not in ordered:
            ordered.append(field)
    ordered.extend(sorted(observed - set(ordered)))
    if not ordered:
        ordered = STANDARD_INDEX_FIELDS + BUILDER_INDEX_FIELDS
    return ordered


def _aggregate_content_hash(
    records: Sequence[Mapping[str, Any]],
    *,
    id_field: str,
    hash_field: str,
) -> str:
    hasher = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item[id_field])):
        hasher.update(str(record[id_field]).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(record[hash_field]).encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _default_output_dir(structures_root: Path) -> Path:
    if (
        structures_root.name == "structures"
        and structures_root.parent.name == "structures"
    ):
        return structures_root.parent.parent / "protenix"
    return structures_root.parent / "protenix"


def build_dataset(
    *,
    cache_json: Path,
    structures_root: Path,
    alignment_dirs: Sequence[Path],
    output_dir: Path | None = None,
    workers: int = 1,
    resume: bool = False,
    limit: int | None = None,
    fail_fast: bool = False,
    allow_missing_alignments: bool = False,
    expected_entry_count: int = EXPECTED_CACHE_ENTRY_COUNT,
    compresslevel: int = 3,
) -> dict[str, Any]:
    """Build all selected structures and publish deterministic aggregates."""

    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if expected_entry_count <= 0:
        raise ValueError(
            "expected_entry_count must be positive, "
            f"got {expected_entry_count}"
        )
    if not 0 <= compresslevel <= 9:
        raise ValueError(
            f"compresslevel must be in [0, 9], got {compresslevel}"
        )

    structures_root = structures_root.expanduser().resolve(strict=True)
    if not structures_root.is_dir():
        raise DatasetBuildError(
            f"Structures root is not a directory: {structures_root}"
        )
    cache, cache_info = _load_cache(
        cache_json,
        expected_entry_count=expected_entry_count,
    )
    normalized_alignment_dirs = _normalize_alignment_dirs(alignment_dirs)
    output_dir = (
        output_dir.expanduser().resolve(strict=False)
        if output_dir is not None
        else _default_output_dir(structures_root)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    success_path = output_dir / "_SUCCESS.json"
    try:
        success_path.unlink()
    except FileNotFoundError:
        pass

    selected_ids = sorted(cache)
    if limit is not None:
        selected_ids = selected_ids[:limit]
    selected_entries = {pdb_id: cache[pdb_id] for pdb_id in selected_ids}

    alignment_rows, alignment_failures = _alignment_inventory(
        selected_entries,
        normalized_alignment_dirs,
    )
    ccd_prewarm = None
    if workers > 1:
        ccd_prewarm = _prewarm_ccd_for_fork()
        print(
            "ccd_prewarm_seconds="
            f"{ccd_prewarm['preload_seconds']:.3f} "
            f"start_method={ccd_prewarm['process_start_method']}",
            flush=True,
        )
    tasks = [
        {
            "pdb_id": pdb_id,
            "cache_entry": selected_entries[pdb_id],
            "structures_root": str(structures_root),
            "alignment_dirs": [
                str(path) for path in normalized_alignment_dirs
            ],
            "resume": resume,
            "compresslevel": compresslevel,
        }
        for pdb_id in selected_ids
    ]
    results, not_attempted_count = _run_tasks(
        tasks,
        workers=workers,
        fail_fast=fail_fast,
    )
    results.sort(key=lambda result: str(result["pdb_id"]))

    failures: list[dict[str, Any]] = list(alignment_failures)
    successful_results = []
    for result in results:
        if result["status"] == "failed":
            failures.append(
                {
                    "pdb_id": result["pdb_id"],
                    "stage": "conversion",
                    "source_path": result.get("source_path", ""),
                    "error_type": result.get("error_type", "UnknownError"),
                    "message": result.get("error", ""),
                }
            )
        else:
            successful_results.append(result)

    index_rows: list[dict[str, Any]] = []
    index_pdb_ids: set[str] = set()
    weight_failure_pdb_ids: set[str] = set()
    for result in successful_results:
        pdb_id = str(result["pdb_id"])
        try:
            weighted_rows = _weight_index_rows(
                pdb_id=pdb_id,
                raw_rows=result["index_rows"],
                cache_entry=selected_entries[pdb_id],
                bioassembly_relpath=str(result["output_relpath"]),
            )
        except Exception as error:
            weight_failure_pdb_ids.add(pdb_id)
            failures.append(
                {
                    "pdb_id": pdb_id,
                    "stage": "index_weighting",
                    "source_path": result.get("source_path", ""),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            continue
        index_rows.extend(weighted_rows)
        index_pdb_ids.add(pdb_id)
    index_rows.sort(
        key=lambda row: (
            str(row["pdb_id"]),
            str(row.get("type", "")),
            str(row.get("chain_1_id", "")),
            str(row.get("chain_2_id", "")),
            str(row.get("entity_1_id", "")),
            str(row.get("entity_2_id", "")),
            _canonical_json_bytes(row),
        )
    )
    failures.sort(
        key=lambda row: (
            str(row.get("pdb_id", "")),
            str(row.get("stage", "")),
            str(row.get("source_path", "")),
            str(row.get("error_type", "")),
            str(row.get("message", "")),
        )
    )

    indices_path = output_dir / "indices.csv.gz"
    failures_path = output_dir / "failures.tsv"
    alignments_path = output_dir / "alignment_representatives.tsv.gz"
    _atomic_csv(
        index_rows,
        indices_path,
        _index_fieldnames(index_rows),
        compresslevel=compresslevel,
    )
    _atomic_csv(
        failures,
        failures_path,
        FAILURE_FIELDS,
        compresslevel=compresslevel,
        delimiter="\t",
    )
    _atomic_csv(
        alignment_rows,
        alignments_path,
        ALIGNMENT_FIELDS,
        compresslevel=compresslevel,
        delimiter="\t",
    )

    missing_alignment_count = sum(
        row["status"] == "missing" for row in alignment_rows
    )
    converted_count = sum(
        result["status"] == "converted"
        for result in successful_results
    )
    resumed_count = sum(
        result["status"] == "resumed"
        for result in successful_results
    )
    full_inventory_selected = (
        limit is None and len(selected_ids) == expected_entry_count
    )
    all_entries_valid = (
        len(successful_results) == len(selected_ids)
        and not_attempted_count == 0
        and not weight_failure_pdb_ids
        and index_pdb_ids == set(selected_ids)
    )
    success_eligible = (
        full_inventory_selected
        and all_entries_valid
        and missing_alignment_count == 0
    )

    source_records = [
        result
        for result in successful_results
        if result.get("source_sha256")
    ]
    output_records = [
        result
        for result in successful_results
        if result.get("output_sha256")
    ]
    alignment_inventory_sha256 = _sha256_bytes(
        _canonical_json_bytes(alignment_rows)
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "openfold3_disordered_protenix_dataset",
        "status": (
            "complete"
            if success_eligible
            else "partial" if limit is not None else "failed"
        ),
        "parameters": {
            "structures_root": str(structures_root),
            "output_dir": str(output_dir),
            "alignment_dirs": [
                str(path) for path in normalized_alignment_dirs
            ],
            "workers": workers,
            "resume": resume,
            "limit": limit,
            "fail_fast": fail_fast,
            "allow_missing_alignments": allow_missing_alignments,
            "expected_entry_count": expected_entry_count,
            "compresslevel": compresslevel,
        },
        "cache": cache_info,
        "ccd_prewarm": ccd_prewarm,
        "counts": {
            "cache_entries": len(cache),
            "selected_entries": len(selected_ids),
            "attempted_entries": len(results),
            "not_attempted_entries": not_attempted_count,
            "converted_entries": converted_count,
            "resumed_entries": resumed_count,
            "validated_pickles": len(successful_results),
            "index_covered_entries": len(index_pdb_ids),
            "index_rows": len(index_rows),
            "conversion_failures": sum(
                failure["stage"] == "conversion"
                for failure in failures
            ),
            "index_weighting_failures": len(weight_failure_pdb_ids),
            "alignment_representatives": len(alignment_rows),
            "resolved_alignment_representatives": (
                len(alignment_rows) - missing_alignment_count
            ),
            "missing_alignment_representatives": missing_alignment_count,
        },
        "sources": {
            "count": len(source_records),
            "total_bytes": sum(
                int(record.get("source_size_bytes", 0))
                for record in source_records
            ),
            "content_manifest_sha256": _aggregate_content_hash(
                source_records,
                id_field="pdb_id",
                hash_field="source_sha256",
            ),
        },
        "bioassemblies": {
            "count": len(output_records),
            "total_bytes": sum(
                int(record.get("output_size_bytes", 0))
                for record in output_records
            ),
            "content_manifest_sha256": _aggregate_content_hash(
                output_records,
                id_field="pdb_id",
                hash_field="output_sha256",
            ),
        },
        "alignments": {
            "inventory_sha256": alignment_inventory_sha256,
            "case_sensitive_chain_ids": True,
            "pdb_id_candidates": ["exact", "lower", "upper"],
            "missing_allowed_for_debug": allow_missing_alignments,
        },
        "artifacts": {
            "indices": _artifact_record(
                indices_path,
                rows=len(index_rows),
            ),
            "failures": _artifact_record(
                failures_path,
                rows=len(failures),
            ),
            "alignment_representatives": _artifact_record(
                alignments_path,
                rows=len(alignment_rows),
            ),
        },
        "success_eligible": success_eligible,
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_json(manifest, manifest_path)
    manifest_sha256 = _sha256_file(manifest_path)
    if success_eligible:
        _atomic_json(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_role": (
                    "openfold3_disordered_protenix_dataset_success"
                ),
                "manifest": manifest_path.name,
                "manifest_sha256": manifest_sha256,
                "cache_sha256": cache_info["sha256"],
                "entry_count": len(selected_ids),
                "pickle_count": len(successful_results),
                "index_row_count": len(index_rows),
                "alignment_representative_count": len(alignment_rows),
            },
            success_path,
        )

    summary = {
        "status": manifest["status"],
        "success_marker": str(success_path) if success_eligible else None,
        "manifest": str(manifest_path),
        "counts": manifest["counts"],
        "missing_alignments_allowed": allow_missing_alignments,
        "exit_ok": (
            not any(
                failure["stage"] != "alignment" for failure in failures
            )
            and (
                missing_alignment_count == 0
                or allow_missing_alignments
            )
        ),
    }
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-json",
        type=Path,
        required=True,
        help="Official cache JSON containing top-level structure_data.",
    )
    parser.add_argument(
        "--structures-root",
        type=Path,
        required=True,
        help="Directory containing <pdb>/<pdb>.npz.",
    )
    parser.add_argument(
        "--alignment-dir",
        type=Path,
        action="append",
        required=True,
        help=(
            "Case-sensitive alignment lookup directory. Repeat for "
            "supplemental directories; first match wins."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Dataset-level aggregate directory. Defaults to "
            "<dataset-root>/protenix for a structures/structures root."
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--allow-missing-alignments",
        action="store_true",
        help=(
            "Debug only: return success despite missing representatives. "
            "A dataset _SUCCESS.json is still never published when missing."
        ),
    )
    parser.add_argument(
        "--expected-entry-count",
        type=int,
        default=EXPECTED_CACHE_ENTRY_COUNT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--compresslevel", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        summary = build_dataset(
            cache_json=args.cache_json,
            structures_root=args.structures_root,
            alignment_dirs=args.alignment_dir,
            output_dir=args.output_dir,
            workers=args.workers,
            resume=args.resume,
            limit=args.limit,
            fail_fast=args.fail_fast,
            allow_missing_alignments=args.allow_missing_alignments,
            expected_entry_count=args.expected_entry_count,
            compresslevel=args.compresslevel,
        )
    except (DatasetBuildError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["exit_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
