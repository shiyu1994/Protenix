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

"""Persist audited OpenFold3-disordered ``no_exact`` query-only chains.

The post-representative-repair v3 audit contains exactly two ``no_exact``
chains.  This tool proves that the complete audit still matches the currently
published dataset, independently rechecks that each chain still has no exact
same-polymer representative query, and adds those chain IDs to the per-PKL
``openfold3_query_only_chain_ids`` list.

Only that top-level marker is changed.  Structure/token features, canonical
sequences, representative mappings, the official cache, index, source NPZs,
and alignment archives remain unchanged.

Apply mode is fail-closed and transactional:

* an exclusive dataset repair lock covers validation, backup, publication,
  verification, and receipt creation;
* all mutable inputs are copied to a verified external backup first;
* the old success marker is withdrawn before the first dataset write;
* PKLs, build states, and the manifest are atomically replaced and verified;
* ``_SUCCESS.json`` is published last;
* any failure restores every original payload and restores old ``_SUCCESS``
  strictly last.

Running the tool again is rejected by both manifest repair provenance and
per-PKL marker checks.  Without ``--apply`` it performs a read-only dry run.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import gzip
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_openfold3_alignment_representatives as audit  # noqa: E402
import build_openfold3_disordered_dataset as builder  # noqa: E402
import repair_openfold3_alignment_representatives as repair  # noqa: E402


REPAIR_ID = "openfold3_disordered_no_exact_query_only_marker_v1"
EXPECTED_DATASET = repair.EXPECTED_DATASET
EXPECTED_NO_EXACT_COUNT = 2
EXPECTED_AFFECTED_STRUCTURE_COUNT = 2
MARKER_FIELD = "openfold3_query_only_chain_ids"


class QueryOnlyMarkerError(RuntimeError):
    """Raised when a marker precondition or transaction fails."""


@dataclass(frozen=True)
class NoExactRecord:
    """One chain proved to have no exact same-polymer representative."""

    pdb_id: str
    bioassembly_relpath: str
    chain_id: str
    polymer_type: str
    canonical_query_length: int
    canonical_query_sha256: str
    current_representative_id: str


@dataclass(frozen=True)
class MarkedEntry:
    """One validated PKL/state replacement pair."""

    pdb_id: str
    bioassembly_path: Path
    state_path: Path
    bioassembly_payload: bytes
    state: dict[str, Any]
    old_output_sha256: str
    new_output_sha256: str
    old_output_size_bytes: int
    new_output_size_bytes: int
    old_state_sha256: str
    old_marker_chain_ids: tuple[str, ...]
    new_marker_chain_ids: tuple[str, ...]


def _require_prior_representative_repair(
    manifest: Mapping[str, Any],
) -> None:
    history = manifest.get("repair_history")
    if not isinstance(history, list):
        raise QueryOnlyMarkerError(
            "manifest has no representative repair history"
        )
    prior = [
        item
        for item in history
        if isinstance(item, Mapping)
        and item.get("repair_id") == repair.REPAIR_ID
    ]
    if len(prior) != 1:
        raise QueryOnlyMarkerError(
            "manifest must contain exactly one completed representative "
            f"repair {repair.REPAIR_ID!r}"
        )
    if (
        int(prior[0].get("affected_chain_count", -1)) != 20
        or int(prior[0].get("affected_structure_count", -1)) != 9
    ):
        raise QueryOnlyMarkerError(
            "manifest representative repair provenance has unexpected counts"
        )


def _load_no_exact_records(
    *,
    result_paths: Sequence[Path],
    summaries: Mapping[Path, Mapping[str, Any]],
    inventory_by_pdb_id: Mapping[str, str],
) -> list[NoExactRecord]:
    # Reuse the hardened full-result scanner.  A post-repair v3 audit must
    # contain zero remaining remappable_unique records; this also verifies all
    # classification and record-type counts against every complete summary.
    repair._load_repair_records(
        result_paths,
        summaries=summaries,
        inventory_by_pdb_id=inventory_by_pdb_id,
        expected_repair_count=0,
    )

    records: list[NoExactRecord] = []
    seen: set[tuple[str, str]] = set()
    for result_path in result_paths:
        summary = summaries[result_path]
        run_contract = summary["run_contract"]
        expected_shard_index = int(run_contract["shard_index"])
        expected_shard_count = int(run_contract["shard_count"])
        with result_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw.get("classification") != "no_exact":
                    continue
                if (
                    raw.get("dataset") != EXPECTED_DATASET
                    or int(raw.get("schema_version", -1)) != 2
                    or raw.get("record_type") != "chain"
                    or int(raw.get("shard_index", -1))
                    != expected_shard_index
                    or int(raw.get("shard_count", -1))
                    != expected_shard_count
                ):
                    raise QueryOnlyMarkerError(
                        f"no_exact audit record contract mismatch at "
                        f"{result_path}:{line_number}"
                    )
                if (
                    raw.get("matching_representative_ids") != []
                    or raw.get("proposed_representative_id") not in (None, "")
                    or not isinstance(raw.get("current_representative"), Mapping)
                    or raw["current_representative"].get("status") != "valid"
                    or raw.get("candidate_errors") != {}
                ):
                    raise QueryOnlyMarkerError(
                        f"no_exact record is not a clean zero-match proof at "
                        f"{result_path}:{line_number}"
                    )
                pdb_id = str(raw.get("pdb_id", ""))
                relpath = str(raw.get("bioassembly_relpath", ""))
                if inventory_by_pdb_id.get(pdb_id) != relpath:
                    raise QueryOnlyMarkerError(
                        f"no_exact record is outside the published inventory "
                        f"at {result_path}:{line_number}"
                    )
                record = NoExactRecord(
                    pdb_id=pdb_id,
                    bioassembly_relpath=relpath,
                    chain_id=str(raw.get("chain_id", "")),
                    polymer_type=str(raw.get("polymer_type", "")),
                    canonical_query_length=int(
                        raw.get("canonical_query_length", -1)
                    ),
                    canonical_query_sha256=str(
                        raw.get("canonical_query_sha256", "")
                    ),
                    current_representative_id=str(
                        raw.get("current_representative_id", "")
                    ),
                )
                if (
                    not record.chain_id
                    or record.polymer_type not in audit.SUPPORTED_POLYMER_TYPES
                    or record.canonical_query_length <= 0
                    or len(record.canonical_query_sha256) != 64
                    or not record.current_representative_id
                ):
                    raise QueryOnlyMarkerError(
                        f"invalid no_exact identity at "
                        f"{result_path}:{line_number}"
                    )
                key = (record.bioassembly_relpath, record.chain_id)
                if key in seen:
                    raise QueryOnlyMarkerError(
                        f"duplicate no_exact audit record: {key}"
                    )
                seen.add(key)
                records.append(record)

    records.sort(
        key=lambda item: (item.bioassembly_relpath, item.chain_id)
    )
    if len(records) != EXPECTED_NO_EXACT_COUNT:
        raise QueryOnlyMarkerError(
            f"expected exactly {EXPECTED_NO_EXACT_COUNT} no_exact chains, "
            f"got {len(records)}"
        )
    if len({record.pdb_id for record in records}) != (
        EXPECTED_AFFECTED_STRUCTURE_COUNT
    ):
        raise QueryOnlyMarkerError(
            "the two no_exact chains must belong to two distinct structures"
        )
    return records


def _group_records(
    records: Sequence[NoExactRecord],
) -> dict[str, list[NoExactRecord]]:
    grouped: dict[str, list[NoExactRecord]] = {}
    for record in records:
        grouped.setdefault(record.pdb_id, []).append(record)
    return grouped


def _audit_artifact_records(
    result_paths: Sequence[Path],
) -> list[dict[str, str]]:
    return [
        {
            "path": str(path),
            "sha256": repair._sha256_file(path),
            "summary_path": str(repair._audit_summary_path(path)),
            "summary_sha256": repair._sha256_file(
                repair._audit_summary_path(path)
            ),
        }
        for path in result_paths
    ]


def _verify_audit_artifacts_unchanged(
    artifacts: Sequence[Mapping[str, str]],
) -> None:
    for artifact in artifacts:
        for path_field, hash_field in (
            ("path", "sha256"),
            ("summary_path", "summary_sha256"),
        ):
            path = Path(artifact[path_field])
            if repair._sha256_file(path) != artifact[hash_field]:
                raise QueryOnlyMarkerError(
                    f"audit artifact changed during marker validation: {path}"
                )


def _same_polymer_representatives(
    *,
    record: NoExactRecord,
    cache_entry: Mapping[str, Any],
) -> list[str]:
    cache_chains = cache_entry.get("chains")
    if not isinstance(cache_chains, Mapping):
        raise QueryOnlyMarkerError(
            f"{record.pdb_id}: invalid official cache chain mapping"
        )
    expected_molecule_type = {
        audit.PROTEIN_CHAIN: "PROTEIN",
        audit.RNA_CHAIN: "RNA",
    }[record.polymer_type]
    current_chain = cache_chains.get(record.chain_id)
    if (
        not isinstance(current_chain, Mapping)
        or str(current_chain.get("molecule_type", ""))
        != expected_molecule_type
        or str(current_chain.get("alignment_representative_id", ""))
        != record.current_representative_id
    ):
        raise QueryOnlyMarkerError(
            f"{record.pdb_id} chain {record.chain_id}: audit identity does "
            "not match the official cache entry"
        )
    representatives = sorted(
        {
            str(chain.get("alignment_representative_id"))
            for chain in cache_chains.values()
            if isinstance(chain, Mapping)
            and str(chain.get("molecule_type", ""))
            == expected_molecule_type
            and chain.get("alignment_representative_id") not in (None, "")
        }
    )
    if record.current_representative_id not in representatives:
        raise QueryOnlyMarkerError(
            f"{record.pdb_id} chain {record.chain_id}: current "
            "representative is not in the same-polymer candidate set"
        )
    return representatives


def _validate_still_no_exact(
    *,
    record: NoExactRecord,
    bioassembly: Mapping[str, Any],
    cache_entry: Mapping[str, Any],
    alignment_dirs: Sequence[Path],
) -> None:
    representatives = bioassembly.get(
        "alignment_representative_id_by_chain_id"
    )
    sequences = bioassembly.get("sequences_by_chain_id")
    if not isinstance(representatives, Mapping) or not isinstance(
        sequences,
        Mapping,
    ):
        raise QueryOnlyMarkerError(
            f"{record.pdb_id}: missing representative/sequence mappings"
        )
    if (
        str(representatives.get(record.chain_id, ""))
        != record.current_representative_id
    ):
        raise QueryOnlyMarkerError(
            f"{record.pdb_id} chain {record.chain_id}: current "
            "representative changed since the audit"
        )
    canonical = str(sequences.get(record.chain_id, ""))
    if (
        len(canonical) != record.canonical_query_length
        or repair._sha256_bytes(canonical.encode("utf-8"))
        != record.canonical_query_sha256
    ):
        raise QueryOnlyMarkerError(
            f"{record.pdb_id} chain {record.chain_id}: canonical query "
            "changed since the audit"
        )

    same_polymer = _same_polymer_representatives(
        record=record,
        cache_entry=cache_entry,
    )
    candidate_queries = {
        representative_id: audit._load_representative_query_uncached(
            representative_id,
            record.polymer_type,
            alignment_dirs,
        )
        for representative_id in same_polymer
    }
    invalid = sorted(
        representative_id
        for representative_id, candidate in candidate_queries.items()
        if candidate.status != "valid"
    )
    exact = sorted(
        representative_id
        for representative_id, candidate in candidate_queries.items()
        if candidate.status == "valid"
        and candidate.query_length == record.canonical_query_length
        and candidate.query_sha256 == record.canonical_query_sha256
    )
    if invalid or exact:
        raise QueryOnlyMarkerError(
            f"{record.pdb_id} chain {record.chain_id}: no_exact proof no "
            f"longer holds; invalid={invalid}, exact={exact}"
        )


def _validated_existing_markers(
    *,
    pdb_id: str,
    bioassembly: Mapping[str, Any],
) -> tuple[str, ...]:
    if MARKER_FIELD not in bioassembly:
        return ()
    raw = bioassembly[MARKER_FIELD]
    if not isinstance(raw, list) or not all(
        isinstance(chain_id, str) and chain_id for chain_id in raw
    ):
        raise QueryOnlyMarkerError(
            f"{pdb_id}: existing {MARKER_FIELD} must be a list of chain IDs"
        )
    if len(raw) != len(set(raw)):
        raise QueryOnlyMarkerError(
            f"{pdb_id}: existing {MARKER_FIELD} contains duplicates"
        )
    sequences = bioassembly.get("sequences_by_chain_id")
    if not isinstance(sequences, Mapping) or not set(raw).issubset(sequences):
        raise QueryOnlyMarkerError(
            f"{pdb_id}: existing {MARKER_FIELD} contains unknown chain IDs"
        )
    return tuple(sorted(raw))


def _mark_entry(
    *,
    pdb_id: str,
    records: Sequence[NoExactRecord],
    structures_root: Path,
    alignment_dirs: Sequence[Path],
    cache_entry: Mapping[str, Any],
    old_state: Mapping[str, Any],
    compresslevel: int,
) -> MarkedEntry:
    source_path = structures_root / pdb_id / f"{pdb_id}.npz"
    bioassembly_path = (
        structures_root
        / pdb_id
        / "bioassembly"
        / f"{pdb_id}.pkl.gz"
    )
    state_path = bioassembly_path.with_name(f"{pdb_id}.build.json")
    old_output_sha256 = repair._sha256_file(bioassembly_path)
    old_output_size_bytes = bioassembly_path.stat().st_size
    if (
        old_output_sha256 != str(old_state.get("output_sha256", ""))
        or old_output_size_bytes
        != int(old_state.get("output_size_bytes", -1))
    ):
        raise QueryOnlyMarkerError(
            f"{pdb_id}: current bioassembly does not match its build state"
        )
    current_state = repair._load_json(
        state_path,
        label=f"{pdb_id} build state",
    )
    if current_state != old_state:
        raise QueryOnlyMarkerError(
            f"{pdb_id}: build state changed during validation"
        )
    old_state_sha256 = repair._sha256_file(state_path)

    with gzip.open(bioassembly_path, "rb") as stream:
        current_bioassembly = pickle.load(stream)
    if not isinstance(current_bioassembly, dict):
        raise QueryOnlyMarkerError(
            f"{pdb_id}: bioassembly payload is not a dict"
        )
    for record in records:
        _validate_still_no_exact(
            record=record,
            bioassembly=current_bioassembly,
            cache_entry=cache_entry,
            alignment_dirs=alignment_dirs,
        )

    source_sha256 = repair._sha256_file(source_path)
    cache_entry_sha256 = repair._sha256_bytes(
        builder._canonical_json_bytes(cache_entry)
    )
    if source_sha256 != str(old_state.get("source_sha256", "")):
        raise QueryOnlyMarkerError(
            f"{pdb_id}: source NPZ changed since the build"
        )
    if cache_entry_sha256 != str(
        old_state.get("cache_entry_sha256", "")
    ):
        raise QueryOnlyMarkerError(
            f"{pdb_id}: official cache entry changed since the build"
        )

    old_marker_chain_ids = _validated_existing_markers(
        pdb_id=pdb_id,
        bioassembly=current_bioassembly,
    )
    audited_chain_ids = {record.chain_id for record in records}
    already_marked = audited_chain_ids.intersection(old_marker_chain_ids)
    if already_marked:
        raise QueryOnlyMarkerError(
            f"{pdb_id}: audited chains are already query-only marked: "
            f"{sorted(already_marked)}"
        )
    new_marker_chain_ids = tuple(
        sorted(set(old_marker_chain_ids).union(audited_chain_ids))
    )

    # Marker-only by construction: every existing top-level value is shared
    # with the validated source object and only the marker list is replaced.
    patched = dict(current_bioassembly)
    patched[MARKER_FIELD] = list(new_marker_chain_ids)
    for key, value in current_bioassembly.items():
        if key != MARKER_FIELD and patched[key] is not value:
            raise QueryOnlyMarkerError(
                f"{pdb_id}: marker-only construction changed {key!r}"
            )

    bioassembly_payload = builder._deterministic_gzip(
        pickle.dumps(patched, protocol=pickle.HIGHEST_PROTOCOL),
        compresslevel=compresslevel,
    )
    new_output_sha256 = repair._sha256_bytes(bioassembly_payload)
    new_state = copy.deepcopy(dict(old_state))
    new_state["output_size_bytes"] = len(bioassembly_payload)
    new_state["output_sha256"] = new_output_sha256
    return MarkedEntry(
        pdb_id=pdb_id,
        bioassembly_path=bioassembly_path,
        state_path=state_path,
        bioassembly_payload=bioassembly_payload,
        state=new_state,
        old_output_sha256=old_output_sha256,
        new_output_sha256=new_output_sha256,
        old_output_size_bytes=old_output_size_bytes,
        new_output_size_bytes=len(bioassembly_payload),
        old_state_sha256=old_state_sha256,
        old_marker_chain_ids=old_marker_chain_ids,
        new_marker_chain_ids=new_marker_chain_ids,
    )


def _mark_with_contract(
    *,
    contract: audit.DatasetContract,
    audit_results: Sequence[Path],
    backup_dir: Path,
    state_read_workers: int,
    apply: bool,
    started: float,
) -> dict[str, Any]:
    result_paths = [
        path.expanduser().resolve(strict=True) for path in audit_results
    ]
    audit_artifacts = _audit_artifact_records(result_paths)
    summaries = repair._validate_audit_summaries(
        result_paths=result_paths,
        contract=contract,
    )
    inventory_by_pdb_id = dict(audit.read_disordered_inventory(contract))
    records = _load_no_exact_records(
        result_paths=result_paths,
        summaries=summaries,
        inventory_by_pdb_id=inventory_by_pdb_id,
    )
    _verify_audit_artifacts_unchanged(audit_artifacts)
    grouped = _group_records(records)

    manifest = repair._load_json(
        contract.manifest_path,
        label="dataset manifest",
    )
    success = repair._load_json(
        contract.success_path,
        label="dataset success marker",
    )
    _require_prior_representative_repair(manifest)
    repair_history = manifest.get("repair_history", [])
    if any(
        isinstance(item, Mapping) and item.get("repair_id") == REPAIR_ID
        for item in repair_history
    ):
        raise QueryOnlyMarkerError(
            f"dataset already declares marker repair {REPAIR_ID}"
        )

    cache_path = Path(str(manifest["cache"]["path"])).resolve(strict=True)
    cache, cache_info = builder._load_cache(
        cache_path,
        expected_entry_count=contract.declared_pickle_count,
    )
    if cache_info["sha256"] != str(manifest["cache"]["sha256"]):
        raise QueryOnlyMarkerError(
            "official cache changed since dataset publication"
        )
    states, state_bytes = repair._load_all_states(
        structures_root=contract.structures_root,
        pdb_ids=sorted(cache),
        workers=state_read_workers,
    )
    repair._validate_pre_repair_inventory(
        manifest=manifest,
        states=states,
        expected_pickle_count=contract.declared_pickle_count,
    )

    compresslevel = int(manifest["parameters"]["compresslevel"])
    marked_entries = [
        _mark_entry(
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
    for entry in marked_entries:
        states[entry.pdb_id] = entry.state
    new_total_bytes, new_aggregate = repair._state_inventory(states)

    new_manifest = copy.deepcopy(manifest)
    new_manifest["repair_history"].append(
        {
            "repair_id": REPAIR_ID,
            "dataset": EXPECTED_DATASET,
            "method": "targeted_no_exact_query_only_marker_patch",
            "marker_field": MARKER_FIELD,
            "affected_structure_count": len(marked_entries),
            "affected_chain_count": len(records),
            "pdb_ids": sorted(grouped),
            "audit_artifacts": audit_artifacts,
        }
    )
    new_manifest["bioassemblies"]["total_bytes"] = new_total_bytes
    new_manifest["bioassemblies"][
        "content_manifest_sha256"
    ] = new_aggregate
    new_manifest_payload = repair._canonical_json_payload(new_manifest)
    new_manifest_sha256 = repair._sha256_bytes(new_manifest_payload)
    new_success = copy.deepcopy(success)
    new_success["manifest_sha256"] = new_manifest_sha256
    new_success_payload = repair._canonical_json_payload(new_success)
    new_success_sha256 = repair._sha256_bytes(new_success_payload)

    plan = {
        "status": "validated" if not apply else "applying",
        "repair_id": REPAIR_ID,
        "dataset_root": str(contract.dataset_root),
        "expected_dataset": EXPECTED_DATASET,
        "marker_field": MARKER_FIELD,
        "affected_structure_count": len(marked_entries),
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
            "success_sha256": new_success_sha256,
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
                "old_state_sha256": entry.old_state_sha256,
                "old_marker_chain_ids": list(entry.old_marker_chain_ids),
                "new_marker_chain_ids": list(entry.new_marker_chain_ids),
            }
            for entry in marked_entries
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
        raise QueryOnlyMarkerError(
            "backup directory must be outside the dataset root"
        )
    if backup_dir.exists():
        raise QueryOnlyMarkerError(
            f"backup directory already exists: {backup_dir}"
        )
    backup_dir.mkdir(parents=True)
    repair._write_receipt(backup_dir / "repair_plan.json", plan)

    mutable_paths = [
        path
        for entry in marked_entries
        for path in (entry.bioassembly_path, entry.state_path)
    ] + [contract.manifest_path, contract.success_path]
    expected_mutable_hashes = {
        contract.manifest_path: contract.manifest_sha256,
        contract.success_path: contract.success_sha256,
    }
    for entry in marked_entries:
        expected_mutable_hashes[entry.bioassembly_path] = (
            entry.old_output_sha256
        )
        expected_mutable_hashes[entry.state_path] = entry.old_state_sha256
    for path in mutable_paths:
        repair._copy_backup(
            backup_dir=backup_dir,
            source=path,
            dataset_root=contract.dataset_root,
        )
    for path, expected_sha256 in expected_mutable_hashes.items():
        backup_path = repair._backup_path(
            backup_dir,
            path,
            contract.dataset_root,
        )
        if (
            repair._sha256_file(path) != expected_sha256
            or repair._sha256_file(backup_path) != expected_sha256
        ):
            raise QueryOnlyMarkerError(
                f"mutable input changed before publication: {path}"
            )
    _verify_audit_artifacts_unchanged(audit_artifacts)
    non_success_paths = [
        path for path in mutable_paths if path != contract.success_path
    ]

    try:
        contract.success_path.unlink()
        for entry in marked_entries:
            builder._atomic_bytes(
                entry.bioassembly_payload,
                entry.bioassembly_path,
            )
            builder._atomic_bytes(
                repair._canonical_json_payload(entry.state),
                entry.state_path,
            )
        builder._atomic_bytes(
            new_manifest_payload,
            contract.manifest_path,
        )

        if (
            repair._sha256_file(contract.manifest_path)
            != new_manifest_sha256
        ):
            raise QueryOnlyMarkerError(
                "staged manifest SHA does not match the marker plan"
            )
        for entry in marked_entries:
            if (
                repair._sha256_file(entry.bioassembly_path)
                != entry.new_output_sha256
                or repair._sha256_file(entry.state_path)
                != repair._sha256_bytes(
                    repair._canonical_json_payload(entry.state)
                )
            ):
                raise QueryOnlyMarkerError(
                    f"staged output verification failed for {entry.pdb_id}"
                )
            with gzip.open(entry.bioassembly_path, "rb") as stream:
                marked = pickle.load(stream)
            if tuple(marked.get(MARKER_FIELD, ())) != (
                entry.new_marker_chain_ids
            ):
                raise QueryOnlyMarkerError(
                    f"staged marker verification failed for {entry.pdb_id}"
                )
        _verify_audit_artifacts_unchanged(audit_artifacts)

        builder._atomic_bytes(
            new_success_payload,
            contract.success_path,
        )
        post_contract = audit.validate_disordered_dataset_root(
            contract.dataset_root
        )
        if (
            post_contract.manifest_sha256 != new_manifest_sha256
            or post_contract.success_sha256 != new_success_sha256
        ):
            raise QueryOnlyMarkerError(
                "published manifest/success SHA does not match marker plan"
            )
        receipt = {
            **plan,
            "status": "complete",
            "backup_dir": str(backup_dir),
            "completed_unix_time": time.time(),
            "elapsed_seconds": time.time() - started,
        }
        repair._write_receipt(
            backup_dir / "repair_receipt.json",
            receipt,
        )
    except BaseException as marker_error:
        try:
            contract.success_path.unlink(missing_ok=True)
            repair._restore_from_backup(
                backup_dir=backup_dir,
                paths=list(reversed(non_success_paths)),
                dataset_root=contract.dataset_root,
            )
            repair._restore_from_backup(
                backup_dir=backup_dir,
                paths=[contract.success_path],
                dataset_root=contract.dataset_root,
            )
            restored = audit.validate_disordered_dataset_root(
                contract.dataset_root
            )
            if (
                restored.manifest_sha256 != contract.manifest_sha256
                or restored.success_sha256 != contract.success_sha256
            ):
                raise QueryOnlyMarkerError(
                    "rollback restored unexpected manifest/success hashes"
                )
            for path, expected_sha256 in expected_mutable_hashes.items():
                if repair._sha256_file(path) != expected_sha256:
                    raise QueryOnlyMarkerError(
                        f"rollback did not restore {path}"
                    )
        except BaseException as rollback_error:
            raise QueryOnlyMarkerError(
                "query-only marker transaction failed and automatic rollback "
                f"also failed; original error was "
                f"{type(marker_error).__name__}: {marker_error}"
            ) from rollback_error
        raise
    return receipt


def mark_dataset(
    *,
    dataset_root: Path,
    audit_results: Sequence[Path],
    backup_dir: Path,
    state_read_workers: int,
    apply: bool,
) -> dict[str, Any]:
    """Validate or atomically publish the two audited query-only markers."""

    started = time.time()
    initial_contract = audit.validate_disordered_dataset_root(dataset_root)
    if not apply:
        return _mark_with_contract(
            contract=initial_contract,
            audit_results=audit_results,
            backup_dir=backup_dir,
            state_read_workers=state_read_workers,
            apply=False,
            started=started,
        )

    resolved_backup_dir = backup_dir.expanduser().resolve(strict=False)
    if audit._is_relative_to(
        resolved_backup_dir,
        initial_contract.dataset_root,
    ):
        raise QueryOnlyMarkerError(
            "backup directory must be outside the dataset root"
        )
    if resolved_backup_dir.exists():
        raise QueryOnlyMarkerError(
            f"backup directory already exists: {resolved_backup_dir}"
        )

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
            raise QueryOnlyMarkerError(
                "another OpenFold3 offline repair holds the lock"
            ) from error
        locked_contract = audit.validate_disordered_dataset_root(
            initial_contract.dataset_root
        )
        return _mark_with_contract(
            contract=locked_contract,
            audit_results=audit_results,
            backup_dir=resolved_backup_dir,
            state_read_workers=state_read_workers,
            apply=True,
            started=started,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Persist exactly two post-repair OpenFold3-disordered no_exact "
            "chains as offline query-only PKL markers."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--audit-result",
        type=Path,
        action="append",
        required=True,
        help="Complete post-repair audit shard JSONL; pass once per shard.",
    )
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument(
        "--report-json",
        type=Path,
        help=(
            "Atomically save the dry-run result outside the dataset root. "
            "The destination must not already exist."
        ),
    )
    parser.add_argument("--state-read-workers", type=int, default=128)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Publish markers. Without this flag, perform a dry run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.state_read_workers <= 0:
        raise QueryOnlyMarkerError(
            "--state-read-workers must be positive"
        )
    report_path: Path | None = None
    if args.report_json is not None:
        if args.apply:
            raise QueryOnlyMarkerError(
                "--report-json is only supported for a dry run"
            )
        report_path = args.report_json.expanduser().resolve(strict=False)
        dataset_root = args.dataset_root.expanduser().resolve(strict=True)
        if audit._is_relative_to(report_path, dataset_root):
            raise QueryOnlyMarkerError(
                "dry-run report must be outside the dataset root"
            )
        if report_path.exists():
            raise QueryOnlyMarkerError(
                f"dry-run report already exists: {report_path}"
            )
    result = mark_dataset(
        dataset_root=args.dataset_root,
        audit_results=args.audit_result,
        backup_dir=args.backup_dir,
        state_read_workers=args.state_read_workers,
        apply=args.apply,
    )
    if report_path is not None:
        repair._write_receipt(report_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
