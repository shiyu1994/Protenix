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

"""Repair audited OpenFold3-disordered representative mismatches offline.

This tool is deliberately limited to the ``remappable_unique`` records emitted
by :mod:`audit_openfold3_alignment_representatives`.  It changes only the
audited entries of ``alignment_representative_id_by_chain_id`` and publishes
updated per-entry state, manifest, and success-marker hashes.  Atom/token
features, canonical sequences, raw training index rows, and alignment archives
remain byte-for-byte semantically unchanged.

The operation is transactional at file granularity:

* all inputs and replacement payloads are validated before the success marker
  is withdrawn;
* the original mutable files are copied to an external backup directory;
* each replacement is written through a same-directory temporary file and
  ``os.replace()``;
* on failure, every replaced file is restored from the backup and the old
  success marker is published last.

The training index and alignment archives are never modified.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import gzip
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_openfold3_alignment_representatives as audit  # noqa: E402
import build_openfold3_disordered_dataset as builder  # noqa: E402


REPAIR_ID = "openfold3_disordered_exact_representative_remap_v1"
EXPECTED_DATASET = "openfold3_disordered_v20260725"


class RepairError(RuntimeError):
    """Raised when a repair precondition or verification fails."""


@dataclass(frozen=True)
class RepairRecord:
    """One exact representative reassignment proved by the read-only audit."""

    pdb_id: str
    bioassembly_relpath: str
    chain_id: str
    polymer_type: str
    canonical_query_length: int
    canonical_query_sha256: str
    current_representative_id: str
    proposed_representative_id: str


@dataclass(frozen=True)
class PatchedEntry:
    """Validated replacement payloads for one affected structure."""

    pdb_id: str
    bioassembly_path: Path
    state_path: Path
    bioassembly_payload: bytes
    state: dict[str, Any]
    old_output_sha256: str
    new_output_sha256: str
    old_output_size_bytes: int
    new_output_size_bytes: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=8 * 1024 * 1024) as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_payload(value: Any) -> bytes:
    return (
        json.dumps(
            builder._json_normalize(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RepairError(f"cannot load {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RepairError(f"{label} must be a JSON object: {path}")
    return value


def _audit_summary_path(result_path: Path) -> Path:
    suffix = ".jsonl"
    if not result_path.name.endswith(suffix):
        raise RepairError(f"audit result must end in {suffix}: {result_path}")
    return result_path.with_name(
        result_path.name[: -len(suffix)] + ".summary.json"
    )


def _validate_audit_summaries(
    *,
    result_paths: Sequence[Path],
    contract: audit.DatasetContract,
) -> dict[Path, dict[str, Any]]:
    summaries: dict[Path, dict[str, Any]] = {}
    shard_pairs: set[tuple[int, int]] = set()
    for result_path in result_paths:
        if audit._is_relative_to(result_path, contract.dataset_root):
            raise RepairError(
                f"audit result must be outside the dataset root: {result_path}"
            )
        summary_path = _audit_summary_path(result_path)
        if audit._is_relative_to(summary_path, contract.dataset_root):
            raise RepairError(
                f"audit summary must be outside the dataset root: {summary_path}"
            )
        summary = _load_json(summary_path, label="audit summary")
        if summary.get("status") != "complete":
            raise RepairError(f"audit summary is not complete: {summary_path}")
        if summary.get("dataset") != EXPECTED_DATASET:
            raise RepairError(f"unexpected audit dataset: {summary_path}")
        if int(summary.get("schema_version", -1)) != 2:
            raise RepairError(f"audit summary is not schema v2: {summary_path}")
        run_contract = summary.get("run_contract")
        if not isinstance(run_contract, Mapping):
            raise RepairError(
                f"audit summary has no run contract: {summary_path}"
            )
        if (
            run_contract.get("dataset") != EXPECTED_DATASET
            or int(run_contract.get("schema_version", -1)) != 1
            or Path(str(run_contract.get("dataset_root", "")))
            .expanduser()
            .resolve(strict=False)
            != contract.dataset_root
            or int(run_contract.get("declared_pickle_count", -1))
            != contract.declared_pickle_count
            or int(run_contract.get("index_row_count", -1))
            != contract.index_row_count
        ):
            raise RepairError(
                f"audit run contract does not match the disorder dataset: "
                f"{summary_path}"
            )
        expected_identities = {
            "manifest_sha256": contract.manifest_sha256,
            "index_sha256": contract.index_sha256,
            "success_sha256": contract.success_sha256,
        }
        for field, expected in expected_identities.items():
            if str(run_contract.get(field, "")) != expected:
                raise RepairError(
                    f"audit {field} no longer matches the dataset: "
                    f"{summary_path}"
                )
        if (
            run_contract.get("comparison")
            != "exact_canonical_query_string_same_polymer_candidates_v2"
        ):
            raise RepairError(
                f"audit comparison contract is unsupported: {summary_path}"
            )
        shard_index = int(run_contract.get("shard_index", -1))
        shard_count = int(run_contract.get("shard_count", -1))
        try:
            selected_structure_count = int(
                summary["selected_structure_count"]
            )
            completed_structure_count = int(
                summary["completed_structure_count"]
            )
            contract_selected_count = int(
                run_contract["selected_structure_count"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RepairError(
                f"audit summary has invalid structure counts: {summary_path}"
            ) from error
        if (
            selected_structure_count <= 0
            or completed_structure_count != selected_structure_count
            or contract_selected_count != selected_structure_count
            or not isinstance(summary.get("classification_counts"), Mapping)
            or not isinstance(summary.get("record_type_counts"), Mapping)
        ):
            raise RepairError(
                f"audit summary does not prove complete shard coverage: "
                f"{summary_path}"
            )
        outputs = summary.get("outputs")
        if (
            not isinstance(outputs, Mapping)
            or Path(str(outputs.get("result_jsonl", "")))
            .expanduser()
            .resolve(strict=False)
            != result_path
            or Path(str(outputs.get("summary_json", "")))
            .expanduser()
            .resolve(strict=False)
            != summary_path
        ):
            raise RepairError(
                f"audit summary output paths do not match: {summary_path}"
            )
        if (shard_index, shard_count) in shard_pairs:
            raise RepairError(
                f"duplicate audit shard ({shard_index}, {shard_count})"
            )
        shard_pairs.add((shard_index, shard_count))
        summaries[result_path] = summary

    shard_counts = {count for _, count in shard_pairs}
    if len(shard_counts) != 1:
        raise RepairError("audit summaries disagree on shard_count")
    shard_count = shard_counts.pop()
    if shard_count <= 0 or shard_pairs != {
        (index, shard_count) for index in range(shard_count)
    }:
        raise RepairError(
            "audit inputs must contain every shard exactly once; got "
            f"{sorted(shard_pairs)}"
        )
    return summaries


def _load_repair_records(
    result_paths: Sequence[Path],
    *,
    summaries: Mapping[Path, Mapping[str, Any]],
    inventory_by_pdb_id: Mapping[str, str],
    expected_repair_count: int,
) -> list[RepairRecord]:
    records: list[RepairRecord] = []
    seen: set[tuple[str, str]] = set()
    for result_path in result_paths:
        summary = summaries[result_path]
        run_contract = summary["run_contract"]
        expected_shard_index = int(run_contract["shard_index"])
        expected_shard_count = int(run_contract["shard_count"])
        classification_counts: dict[str, int] = {
            name: 0 for name in audit.CLASSIFICATIONS
        }
        record_type_counts: dict[str, int] = {}
        with result_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RepairError(
                        f"invalid JSON at {result_path}:{line_number}: {error}"
                    ) from error
                if not isinstance(raw, Mapping):
                    raise RepairError(
                        f"audit record must be an object at "
                        f"{result_path}:{line_number}"
                    )
                if (
                    raw.get("dataset") != EXPECTED_DATASET
                    or int(raw.get("schema_version", -1)) != 2
                    or int(raw.get("shard_index", -1))
                    != expected_shard_index
                    or int(raw.get("shard_count", -1))
                    != expected_shard_count
                ):
                    raise RepairError(
                        f"audit record contract mismatch at "
                        f"{result_path}:{line_number}"
                    )
                classification = str(raw.get("classification", ""))
                if classification not in classification_counts:
                    raise RepairError(
                        f"unknown audit classification at "
                        f"{result_path}:{line_number}: {classification!r}"
                    )
                classification_counts[classification] += 1
                record_type = str(raw.get("record_type", ""))
                if record_type not in {"chain", "structure"}:
                    raise RepairError(
                        f"unknown audit record type at "
                        f"{result_path}:{line_number}: {record_type!r}"
                    )
                record_type_counts[record_type] = (
                    record_type_counts.get(record_type, 0) + 1
                )
                if classification != "remappable_unique":
                    continue
                if record_type != "chain":
                    raise RepairError(
                        "remappable_unique must be a chain record at "
                        f"{result_path}:{line_number}"
                    )
                matching = raw.get("matching_representative_ids")
                proposed = str(raw.get("proposed_representative_id", ""))
                if (
                    not isinstance(matching, list)
                    or len(matching) != 1
                    or str(matching[0]) != proposed
                    or not proposed
                ):
                    raise RepairError(
                        "remappable_unique record lacks exactly one proposed "
                        f"representative at {result_path}:{line_number}"
                    )
                pdb_id = str(raw["pdb_id"])
                bioassembly_relpath = str(raw["bioassembly_relpath"])
                expected_relpath = inventory_by_pdb_id.get(pdb_id)
                if expected_relpath != bioassembly_relpath:
                    raise RepairError(
                        "audit record is not in the published disorder "
                        f"inventory at {result_path}:{line_number}: "
                        f"{pdb_id!r}, {bioassembly_relpath!r}"
                    )
                record = RepairRecord(
                    pdb_id=pdb_id,
                    bioassembly_relpath=bioassembly_relpath,
                    chain_id=str(raw["chain_id"]),
                    polymer_type=str(raw["polymer_type"]),
                    canonical_query_length=int(raw["canonical_query_length"]),
                    canonical_query_sha256=str(
                        raw["canonical_query_sha256"]
                    ),
                    current_representative_id=str(
                        raw["current_representative_id"]
                    ),
                    proposed_representative_id=proposed,
                )
                key = (record.bioassembly_relpath, record.chain_id)
                if key in seen:
                    raise RepairError(f"duplicate audit repair record: {key}")
                seen.add(key)
                records.append(record)

        expected_classification_counts = {
            name: int(summary["classification_counts"].get(name, 0))
            for name in audit.CLASSIFICATIONS
        }
        if classification_counts != expected_classification_counts:
            raise RepairError(
                f"audit result classification counts do not match its "
                f"summary: {result_path}"
            )
        expected_record_type_counts = {
            str(name): int(count)
            for name, count in summary.get("record_type_counts", {}).items()
        }
        if record_type_counts != expected_record_type_counts:
            raise RepairError(
                f"audit result record-type counts do not match its summary: "
                f"{result_path}"
            )
        if (
            record_type_counts.get("chain", 0)
            != int(summary.get("chain_record_count", -1))
            or record_type_counts.get("structure", 0)
            != int(summary.get("structure_record_count", -1))
        ):
            raise RepairError(
                f"audit result record counts do not match its summary: "
                f"{result_path}"
            )

    records.sort(
        key=lambda item: (
            item.bioassembly_relpath,
            item.chain_id,
        )
    )
    if len(records) != expected_repair_count:
        raise RepairError(
            f"expected {expected_repair_count} exact repairs, got "
            f"{len(records)}"
        )
    return records


def _read_state(
    structures_root: Path,
    pdb_id: str,
) -> tuple[str, dict[str, Any], int]:
    state_path = (
        structures_root
        / pdb_id
        / "bioassembly"
        / f"{pdb_id}.build.json"
    )
    payload = state_path.read_bytes()
    state = json.loads(payload)
    if not isinstance(state, dict) or str(state.get("pdb_id", "")) != pdb_id:
        raise RepairError(f"invalid per-entry build state: {state_path}")
    return pdb_id, state, len(payload)


def _load_all_states(
    *,
    structures_root: Path,
    pdb_ids: Sequence[str],
    workers: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = list(
            executor.map(
                lambda pdb_id: _read_state(structures_root, pdb_id),
                pdb_ids,
                chunksize=8,
            )
        )
    states = {pdb_id: state for pdb_id, state, _ in values}
    return states, sum(size for _, _, size in values)


def _state_inventory(
    states: Mapping[str, Mapping[str, Any]],
) -> tuple[int, str]:
    records = [
        {
            "pdb_id": pdb_id,
            "output_size_bytes": int(state["output_size_bytes"]),
            "output_sha256": str(state["output_sha256"]),
        }
        for pdb_id, state in states.items()
    ]
    total_bytes = sum(record["output_size_bytes"] for record in records)
    aggregate = builder._aggregate_content_hash(
        records,
        id_field="pdb_id",
        hash_field="output_sha256",
    )
    return total_bytes, aggregate


def _validate_pre_repair_inventory(
    *,
    manifest: Mapping[str, Any],
    states: Mapping[str, Mapping[str, Any]],
    expected_pickle_count: int,
) -> None:
    if len(states) != expected_pickle_count:
        raise RepairError(
            f"expected {expected_pickle_count} build states, got {len(states)}"
        )
    total_bytes, aggregate = _state_inventory(states)
    bioassemblies = manifest.get("bioassemblies")
    if not isinstance(bioassemblies, Mapping):
        raise RepairError("dataset manifest has no bioassemblies object")
    expected = (
        int(bioassemblies.get("total_bytes", -1)),
        str(bioassemblies.get("content_manifest_sha256", "")),
    )
    actual = (total_bytes, aggregate)
    if actual != expected:
        raise RepairError(
            "per-entry build states do not match the published bioassembly "
            f"inventory: actual={actual}, manifest={expected}"
        )


def _group_records(
    records: Sequence[RepairRecord],
) -> dict[str, list[RepairRecord]]:
    grouped: dict[str, list[RepairRecord]] = {}
    for record in records:
        grouped.setdefault(record.pdb_id, []).append(record)
    return grouped


def _validate_record_against_current_data(
    *,
    record: RepairRecord,
    bioassembly: Mapping[str, Any],
    alignment_dirs: Sequence[Path],
    cache_entry: Mapping[str, Any],
) -> None:
    representatives = bioassembly.get(
        "alignment_representative_id_by_chain_id"
    )
    sequences = bioassembly.get("sequences_by_chain_id")
    if not isinstance(representatives, Mapping) or not isinstance(
        sequences, Mapping
    ):
        raise RepairError(
            f"{record.pdb_id}: missing representative/sequence mappings"
        )
    current = str(representatives.get(record.chain_id, ""))
    if current != record.current_representative_id:
        raise RepairError(
            f"{record.pdb_id} chain {record.chain_id}: current "
            f"representative {current!r} != audited "
            f"{record.current_representative_id!r}"
        )
    if record.polymer_type not in audit.SUPPORTED_POLYMER_TYPES:
        raise RepairError(
            f"{record.pdb_id} chain {record.chain_id}: unsupported audited "
            f"polymer type {record.polymer_type!r}"
        )
    cache_chains = cache_entry.get("chains")
    if not isinstance(cache_chains, Mapping):
        raise RepairError(f"{record.pdb_id}: invalid cache chain mapping")
    expected_molecule_type = {
        audit.PROTEIN_CHAIN: "PROTEIN",
        audit.RNA_CHAIN: "RNA",
    }[record.polymer_type]
    audited_chain = cache_chains.get(record.chain_id)
    if (
        not isinstance(audited_chain, Mapping)
        or str(audited_chain.get("molecule_type", ""))
        != expected_molecule_type
    ):
        raise RepairError(
            f"{record.pdb_id} chain {record.chain_id}: audited polymer type "
            "does not match the official cache entry"
        )
    declared_representatives_by_chain_id = (
        builder._entry_representatives(record.pdb_id, cache_entry)
    )
    declared_current = str(
        declared_representatives_by_chain_id.get(record.chain_id, "")
    )
    if declared_current != record.current_representative_id:
        raise RepairError(
            f"{record.pdb_id} chain {record.chain_id}: audited current "
            "representative is not the representative declared by the "
            "official cache entry"
        )
    same_polymer_representatives = sorted(
        {
            str(chain.get("alignment_representative_id"))
            for chain in cache_chains.values()
            if isinstance(chain, Mapping)
            and str(chain.get("molecule_type", ""))
            == expected_molecule_type
            and chain.get("alignment_representative_id") not in (None, "")
        }
    )
    if record.proposed_representative_id not in same_polymer_representatives:
        raise RepairError(
            f"{record.pdb_id} chain {record.chain_id}: proposed "
            f"representative {record.proposed_representative_id!r} is not "
            "declared for the same polymer type by the same structure's "
            "official cache entry"
        )
    canonical = str(sequences.get(record.chain_id, ""))
    if (
        len(canonical) != record.canonical_query_length
        or _sha256_bytes(canonical.encode("utf-8"))
        != record.canonical_query_sha256
    ):
        raise RepairError(
            f"{record.pdb_id} chain {record.chain_id}: canonical query "
            "changed since the audit"
        )
    candidate_queries = {
        representative_id: audit._load_representative_query_uncached(
            representative_id,
            record.polymer_type,
            alignment_dirs,
        )
        for representative_id in same_polymer_representatives
    }
    current_query = candidate_queries[record.current_representative_id]
    if (
        current_query.status != "valid"
        or current_query.query_sha256 == record.canonical_query_sha256
    ):
        raise RepairError(
            f"{record.pdb_id} chain {record.chain_id}: current "
            "representative is no longer a valid mismatching query"
        )
    exact_matches = sorted(
        representative_id
        for representative_id, candidate in candidate_queries.items()
        if candidate.status == "valid"
        and candidate.query_length == record.canonical_query_length
        and candidate.query_sha256 == record.canonical_query_sha256
    )
    if exact_matches != [record.proposed_representative_id]:
        raise RepairError(
            f"{record.pdb_id} chain {record.chain_id}: proposed "
            f"representative is no longer the unique exact same-polymer "
            f"match; current matches are {exact_matches}"
        )


def _patch_entry(
    *,
    pdb_id: str,
    records: Sequence[RepairRecord],
    structures_root: Path,
    alignment_dirs: Sequence[Path],
    cache_entry: Mapping[str, Any],
    old_state: Mapping[str, Any],
    compresslevel: int,
) -> PatchedEntry:
    source_path = structures_root / pdb_id / f"{pdb_id}.npz"
    bioassembly_path = (
        structures_root
        / pdb_id
        / "bioassembly"
        / f"{pdb_id}.pkl.gz"
    )
    state_path = bioassembly_path.with_name(f"{pdb_id}.build.json")
    old_output_sha256 = _sha256_file(bioassembly_path)
    old_output_size_bytes = bioassembly_path.stat().st_size
    if (
        old_output_sha256 != str(old_state.get("output_sha256", ""))
        or old_output_size_bytes
        != int(old_state.get("output_size_bytes", -1))
    ):
        raise RepairError(
            f"{pdb_id}: current bioassembly does not match its build state"
        )

    with gzip.open(bioassembly_path, "rb") as stream:
        current_bioassembly = pickle.load(stream)
    if not isinstance(current_bioassembly, dict):
        raise RepairError(f"{pdb_id}: bioassembly payload is not a dict")
    for record in records:
        _validate_record_against_current_data(
            record=record,
            bioassembly=current_bioassembly,
            alignment_dirs=alignment_dirs,
            cache_entry=cache_entry,
        )

    source_sha256 = _sha256_file(source_path)
    cache_entry_sha256 = _sha256_bytes(
        builder._canonical_json_bytes(cache_entry)
    )
    if source_sha256 != str(old_state.get("source_sha256", "")):
        raise RepairError(f"{pdb_id}: source NPZ changed since the build")
    if cache_entry_sha256 != str(old_state.get("cache_entry_sha256", "")):
        raise RepairError(f"{pdb_id}: cache entry changed since the build")

    # Construct the replacement by sharing every non-mapping value with the
    # validated source object and copying only the representative mapping.
    # This makes "mapping-only" a property of the construction rather than a
    # convention that later code could accidentally violate.
    current_representatives = current_bioassembly[
        "alignment_representative_id_by_chain_id"
    ]
    if not isinstance(current_representatives, dict):
        raise RepairError(
            f"{pdb_id}: representative mapping must be a plain dict"
        )
    patched = dict(current_bioassembly)
    patched_representatives = dict(current_representatives)
    patched["alignment_representative_id_by_chain_id"] = (
        patched_representatives
    )
    for key, value in current_bioassembly.items():
        if (
            key != "alignment_representative_id_by_chain_id"
            and patched[key] is not value
        ):
            raise RepairError(
                f"{pdb_id}: mapping-only construction changed {key!r}"
            )
    for record in records:
        patched_representatives[
            record.chain_id
        ] = record.proposed_representative_id
        actual = str(patched_representatives.get(record.chain_id, ""))
        if actual != record.proposed_representative_id:
            raise RepairError(
                f"{pdb_id} chain {record.chain_id}: patch produced "
                f"{actual!r}, expected {record.proposed_representative_id!r}"
            )
        canonical = str(
            patched["sequences_by_chain_id"].get(record.chain_id, "")
        )
        if _sha256_bytes(canonical.encode()) != record.canonical_query_sha256:
            raise RepairError(
                f"{pdb_id} chain {record.chain_id}: canonical query "
                "does not match the audit"
            )

    pickle_payload = pickle.dumps(patched, protocol=pickle.HIGHEST_PROTOCOL)
    bioassembly_payload = builder._deterministic_gzip(
        pickle_payload,
        compresslevel=compresslevel,
    )
    new_output_sha256 = _sha256_bytes(bioassembly_payload)
    new_state = copy.deepcopy(dict(old_state))
    new_state["output_size_bytes"] = len(bioassembly_payload)
    new_state["output_sha256"] = new_output_sha256
    return PatchedEntry(
        pdb_id=pdb_id,
        bioassembly_path=bioassembly_path,
        state_path=state_path,
        bioassembly_payload=bioassembly_payload,
        state=new_state,
        old_output_sha256=old_output_sha256,
        new_output_sha256=new_output_sha256,
        old_output_size_bytes=old_output_size_bytes,
        new_output_size_bytes=len(bioassembly_payload),
    )


def _backup_path(backup_dir: Path, source: Path, dataset_root: Path) -> Path:
    try:
        relative = source.relative_to(dataset_root)
    except ValueError as error:
        raise RepairError(f"backup source escapes dataset root: {source}") from error
    return backup_dir / "dataset_before" / relative


def _copy_backup(
    *,
    backup_dir: Path,
    source: Path,
    dataset_root: Path,
) -> Path:
    destination = _backup_path(backup_dir, source, dataset_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if _sha256_file(source) != _sha256_file(destination):
        raise RepairError(f"backup verification failed: {source}")
    return destination


def _restore_from_backup(
    *,
    backup_dir: Path,
    paths: Sequence[Path],
    dataset_root: Path,
) -> None:
    for path in paths:
        backup = _backup_path(backup_dir, path, dataset_root)
        if backup.is_file():
            builder._atomic_bytes(backup.read_bytes(), path)


def _write_receipt(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    builder._atomic_bytes(_canonical_json_payload(dict(value)), path)


def _repair_dataset_with_contract(
    *,
    contract: audit.DatasetContract,
    audit_results: Sequence[Path],
    backup_dir: Path,
    expected_repair_count: int,
    state_read_workers: int,
    apply: bool,
    started: float,
) -> dict[str, Any]:
    result_paths = [
        path.expanduser().resolve(strict=True) for path in audit_results
    ]
    summaries = _validate_audit_summaries(
        result_paths=result_paths,
        contract=contract,
    )
    inventory_by_pdb_id = dict(audit.read_disordered_inventory(contract))
    records = _load_repair_records(
        result_paths,
        summaries=summaries,
        inventory_by_pdb_id=inventory_by_pdb_id,
        expected_repair_count=expected_repair_count,
    )
    grouped = _group_records(records)

    manifest = _load_json(contract.manifest_path, label="dataset manifest")
    success = _load_json(contract.success_path, label="dataset success marker")
    cache_path = Path(str(manifest["cache"]["path"])).resolve(strict=True)
    cache, cache_info = builder._load_cache(
        cache_path,
        expected_entry_count=contract.declared_pickle_count,
    )
    if cache_info["sha256"] != str(manifest["cache"]["sha256"]):
        raise RepairError("official cache changed since dataset publication")
    states, state_bytes = _load_all_states(
        structures_root=contract.structures_root,
        pdb_ids=sorted(cache),
        workers=state_read_workers,
    )
    _validate_pre_repair_inventory(
        manifest=manifest,
        states=states,
        expected_pickle_count=contract.declared_pickle_count,
    )

    compresslevel = int(manifest["parameters"]["compresslevel"])
    patched_entries = [
        _patch_entry(
            pdb_id=pdb_id,
            records=grouped[pdb_id],
            structures_root=contract.structures_root,
            alignment_dirs=contract.alignment_dirs,
            cache_entry=cache[pdb_id],
            old_state=states[pdb_id],
            compresslevel=compresslevel,
        )
        for pdb_id in sorted(grouped)
    ]
    for entry in patched_entries:
        states[entry.pdb_id] = entry.state
    new_total_bytes, new_aggregate = _state_inventory(states)

    audit_artifacts = [
        {
            "path": str(path),
            "sha256": _sha256_file(path),
            "summary_path": str(_audit_summary_path(path)),
            "summary_sha256": _sha256_file(_audit_summary_path(path)),
        }
        for path in result_paths
    ]
    repair_manifest_record = {
        "repair_id": REPAIR_ID,
        "dataset": EXPECTED_DATASET,
        "method": "targeted_mapping_only_exact_canonical_query_patch",
        "affected_structure_count": len(patched_entries),
        "affected_chain_count": len(records),
        "pdb_ids": sorted(grouped),
        "audit_artifacts": audit_artifacts,
    }
    new_manifest = copy.deepcopy(manifest)
    repair_history = new_manifest.setdefault("repair_history", [])
    if not isinstance(repair_history, list):
        raise RepairError("manifest repair_history is not a list")
    if any(
        isinstance(item, Mapping) and item.get("repair_id") == REPAIR_ID
        for item in repair_history
    ):
        raise RepairError(f"dataset already declares repair {REPAIR_ID}")
    repair_history.append(repair_manifest_record)
    new_manifest["bioassemblies"]["total_bytes"] = new_total_bytes
    new_manifest["bioassemblies"][
        "content_manifest_sha256"
    ] = new_aggregate
    new_manifest_payload = _canonical_json_payload(new_manifest)
    new_manifest_sha256 = _sha256_bytes(new_manifest_payload)
    new_success = copy.deepcopy(success)
    new_success["manifest_sha256"] = new_manifest_sha256
    new_success_payload = _canonical_json_payload(new_success)

    plan = {
        "status": "validated" if not apply else "applying",
        "repair_id": REPAIR_ID,
        "dataset_root": str(contract.dataset_root),
        "expected_dataset": EXPECTED_DATASET,
        "affected_structure_count": len(patched_entries),
        "affected_chain_count": len(records),
        "pdb_ids": sorted(grouped),
        "records": [record.__dict__ for record in records],
        "audit_artifacts": audit_artifacts,
        "before": {
            "manifest_sha256": contract.manifest_sha256,
            "success_sha256": contract.success_sha256,
            "bioassembly_total_bytes": int(
                manifest["bioassemblies"]["total_bytes"]
            ),
            "bioassembly_content_manifest_sha256": str(
                manifest["bioassemblies"]["content_manifest_sha256"]
            ),
        },
        "after": {
            "manifest_sha256": new_manifest_sha256,
            "success_sha256": _sha256_bytes(new_success_payload),
            "bioassembly_total_bytes": new_total_bytes,
            "bioassembly_content_manifest_sha256": new_aggregate,
        },
        "state_inventory_bytes_read": state_bytes,
        "entries": [
            {
                "pdb_id": entry.pdb_id,
                "old_output_sha256": entry.old_output_sha256,
                "new_output_sha256": entry.new_output_sha256,
                "old_output_size_bytes": entry.old_output_size_bytes,
                "new_output_size_bytes": entry.new_output_size_bytes,
            }
            for entry in patched_entries
        ],
    }
    if not apply:
        return {
            **plan,
            "status": "dry_run_complete",
            "elapsed_seconds": time.time() - started,
        }

    backup_dir = backup_dir.expanduser().resolve(strict=False)
    if audit._is_relative_to(backup_dir, contract.dataset_root):
        raise RepairError("backup directory must be outside the dataset root")
    if backup_dir.exists():
        raise RepairError(f"backup directory already exists: {backup_dir}")
    backup_dir.mkdir(parents=True)
    _write_receipt(backup_dir / "repair_plan.json", plan)

    mutable_paths = [
        path
        for entry in patched_entries
        for path in (entry.bioassembly_path, entry.state_path)
    ] + [contract.manifest_path, contract.success_path]
    for path in mutable_paths:
        _copy_backup(
            backup_dir=backup_dir,
            source=path,
            dataset_root=contract.dataset_root,
        )

    non_success_mutable_paths = [
        path for path in mutable_paths if path != contract.success_path
    ]
    try:
        # An absent success marker is the public indication that publication is
        # in progress.  It remains absent until all payloads and hashes verify.
        contract.success_path.unlink()
        for entry in patched_entries:
            builder._atomic_bytes(
                entry.bioassembly_payload,
                entry.bioassembly_path,
            )
            builder._atomic_bytes(
                _canonical_json_payload(entry.state),
                entry.state_path,
            )
        builder._atomic_bytes(
            new_manifest_payload,
            contract.manifest_path,
        )

        # Verify mapping-only outputs and their state/manifest hashes before
        # making the dataset success-eligible again.
        if _sha256_file(contract.manifest_path) != new_manifest_sha256:
            raise RepairError(
                "staged manifest SHA does not match the repair plan"
            )
        for entry in patched_entries:
            if (
                _sha256_file(entry.bioassembly_path)
                != entry.new_output_sha256
                or _sha256_file(entry.state_path)
                != _sha256_bytes(_canonical_json_payload(entry.state))
            ):
                raise RepairError(
                    f"staged output verification failed for {entry.pdb_id}"
                )
            with gzip.open(entry.bioassembly_path, "rb") as stream:
                patched = pickle.load(stream)
            for record in grouped[entry.pdb_id]:
                actual = str(
                    patched[
                        "alignment_representative_id_by_chain_id"
                    ].get(record.chain_id, "")
                )
                if actual != record.proposed_representative_id:
                    raise RepairError(
                        f"staged mapping verification failed for "
                        f"{entry.pdb_id} chain {record.chain_id}"
                    )

        # Publish the success marker last, then re-run the full public dataset
        # contract while the exclusive repair lock is still held.
        builder._atomic_bytes(
            new_success_payload,
            contract.success_path,
        )
        post_contract = audit.validate_disordered_dataset_root(
            contract.dataset_root
        )
        if (
            post_contract.manifest_sha256 != new_manifest_sha256
            or post_contract.success_sha256
            != _sha256_bytes(new_success_payload)
        ):
            raise RepairError(
                "published manifest/success SHA does not match the repair plan"
            )

        receipt = {
            **plan,
            "status": "complete",
            "backup_dir": str(backup_dir),
            "completed_unix_time": time.time(),
            "elapsed_seconds": time.time() - started,
        }
        _write_receipt(backup_dir / "repair_receipt.json", receipt)
    except BaseException as repair_error:
        try:
            # Withdraw any newly published marker first.  Restore every
            # mutable payload from verified backups even if the interrupted
            # write was not yet recorded locally, and restore old _SUCCESS
            # strictly last.
            contract.success_path.unlink(missing_ok=True)
            _restore_from_backup(
                backup_dir=backup_dir,
                paths=list(reversed(non_success_mutable_paths)),
                dataset_root=contract.dataset_root,
            )
            _restore_from_backup(
                backup_dir=backup_dir,
                paths=[contract.success_path],
                dataset_root=contract.dataset_root,
            )
            restored_contract = audit.validate_disordered_dataset_root(
                contract.dataset_root
            )
            if (
                restored_contract.manifest_sha256
                != contract.manifest_sha256
                or restored_contract.success_sha256
                != contract.success_sha256
            ):
                raise RepairError(
                    "rollback restored a dataset with unexpected "
                    "manifest/success hashes"
                )
            for entry in patched_entries:
                if (
                    _sha256_file(entry.bioassembly_path)
                    != entry.old_output_sha256
                ):
                    raise RepairError(
                        f"rollback did not restore {entry.pdb_id}"
                    )
        except BaseException as rollback_error:
            raise RepairError(
                "representative repair failed and automatic rollback also "
                f"failed; original error was {type(repair_error).__name__}: "
                f"{repair_error}"
            ) from rollback_error
        raise
    return receipt


def repair_dataset(
    *,
    dataset_root: Path,
    audit_results: Sequence[Path],
    backup_dir: Path,
    expected_repair_count: int,
    state_read_workers: int,
    apply: bool,
) -> dict[str, Any]:
    """Validate or atomically publish an offline mapping-only repair."""

    started = time.time()
    initial_contract = audit.validate_disordered_dataset_root(dataset_root)
    if not apply:
        return _repair_dataset_with_contract(
            contract=initial_contract,
            audit_results=audit_results,
            backup_dir=backup_dir,
            expected_repair_count=expected_repair_count,
            state_read_workers=state_read_workers,
            apply=False,
            started=started,
        )

    resolved_backup_dir = backup_dir.expanduser().resolve(strict=False)
    if audit._is_relative_to(
        resolved_backup_dir,
        initial_contract.dataset_root,
    ):
        raise RepairError("backup directory must be outside the dataset root")
    if resolved_backup_dir.exists():
        raise RepairError(
            f"backup directory already exists: {resolved_backup_dir}"
        )

    # Acquire the lock before re-reading any repair input.  The second contract
    # validation closes the race between the initial root lookup and lock
    # acquisition; stale audit summaries then fail against the fresh hashes.
    lock_path = (
        initial_contract.protenix_root / ".representative_repair.lock"
    )
    with lock_path.open("a+b") as lock_stream:
        try:
            fcntl.flock(
                lock_stream.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            raise RepairError(
                "another representative repair holds the lock"
            ) from error
        locked_contract = audit.validate_disordered_dataset_root(
            initial_contract.dataset_root
        )
        return _repair_dataset_with_contract(
            contract=locked_contract,
            audit_results=audit_results,
            backup_dir=resolved_backup_dir,
            expected_repair_count=expected_repair_count,
            state_read_workers=state_read_workers,
            apply=True,
            started=started,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-repair exact OpenFold3-disordered representative "
            "remaps proved by a complete v2 audit."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--audit-result",
        type=Path,
        action="append",
        required=True,
        help="Complete audit shard JSONL; pass once per shard.",
    )
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--expected-repair-count", type=int, default=20)
    parser.add_argument("--state-read-workers", type=int, default=128)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Publish the repair. Without this flag, perform a dry run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.expected_repair_count <= 0:
        raise RepairError("--expected-repair-count must be positive")
    if args.state_read_workers <= 0:
        raise RepairError("--state-read-workers must be positive")
    result = repair_dataset(
        dataset_root=args.dataset_root,
        audit_results=args.audit_result,
        backup_dir=args.backup_dir,
        expected_repair_count=args.expected_repair_count,
        state_read_workers=args.state_read_workers,
        apply=args.apply,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
