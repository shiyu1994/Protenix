#!/usr/bin/env python3
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
"""Build a fail-closed lock manifest for all PXM evaluation MSAs.

The lock follows the current Protenix loader semantics:

* ``polypeptide(L)`` entities require ``pairing.a3m`` and
  ``non_pairing.a3m`` from the sequence-source manifest;
* exact ``polyribonucleotide`` entities use the first RNA mapping entry and
  require ``<eid>/<eid>_all.a3m``;
* DNA and DNA/RNA-hybrid entity types do not consume an RNA MSA.

Records are de-duplicated by exact query sequence.  The manifest itself is
deterministic; the success marker binds its hash and is written only after the
complete collection has passed validation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional


MSA_LOCK_SCHEMA_VERSION = 1
MSA_LOCK_POLICY_VERSION = "pxm_eval_exact_sequence_msa_v1"
MSA_LOCK_MANIFEST_FILENAME = "msa_manifest.json"
MSA_LOCK_SUCCESS_FILENAME = "_MSA_SUCCESS.json"

PROTEIN_POLY_TYPE = "polypeptide(L)"
RNA_POLY_TYPE = "polyribonucleotide"

DEFAULT_OFFICIAL_ROOT = Path(
    "/mnt/project_abd/public_databases/protenix.20260101"
)
DEFAULT_PXM_MANIFEST_ROOT = Path(
    "/mnt/project_abd/klp/proteinx_eval_data/manifests/"
    "pxm_legacy_af3_like"
)
DEFAULT_PXM_INPUT_ROOT = Path(
    "/mnt/project_abd/klp/proteinx_eval_data/cache/"
    "pxm_legacy_af3_like_cutoff20210930_v1"
)

_PDB_ID_PATTERN = re.compile(r"^[A-Za-z0-9]{4}$")


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA256 digest for one file."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json_dump(payload: Any, output_path: str | Path) -> None:
    """Atomically replace one JSON file after flushing its contents."""

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


def _load_json_mapping(path: str | Path, *, field_name: str) -> Mapping[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{field_name} must contain a JSON object: {path}")
    return payload


def load_protein_sequence_sources(path: str | Path) -> dict[str, str]:
    """Load exact sequence-to-directory sources used by PXM protein MSAs."""

    payload = _load_json_mapping(
        path,
        field_name="protein sequence sources",
    )
    sources: dict[str, str] = {}
    for sequence, source_directory in payload.items():
        if (
            not isinstance(sequence, str)
            or not sequence
            or sequence != sequence.strip()
            or any(character.isspace() for character in sequence)
        ):
            raise ValueError("protein sequence sources contain an invalid sequence")
        if not isinstance(source_directory, str) or not source_directory.strip():
            raise ValueError(
                "protein sequence sources contain an invalid source directory "
                f"for sha256={sequence_sha256(sequence)}"
            )
        sources[sequence] = str(
            Path(source_directory).expanduser().resolve(strict=False)
        )
    if not sources:
        raise ValueError("protein sequence sources contain no records")
    return sources


def load_rna_sequence_mapping(path: str | Path) -> Mapping[str, Any]:
    """Load and minimally validate the mapping consumed by MSASourceManager."""

    payload = _load_json_mapping(path, field_name="RNA sequence mapping")
    for sequence, entity_ids in payload.items():
        if (
            not isinstance(sequence, str)
            or not sequence
            or sequence != sequence.strip()
            or any(character.isspace() for character in sequence)
        ):
            raise ValueError("RNA sequence mapping contains an invalid sequence")
        if (
            not isinstance(entity_ids, list)
            or not entity_ids
            or not isinstance(entity_ids[0], (str, int))
            or not str(entity_ids[0]).strip()
        ):
            raise ValueError(
                "RNA sequence mapping must provide a non-empty list whose first "
                f"entry is an eid: sha256={sequence_sha256(sequence)}"
            )
    return payload


def read_pdb_ids(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    pdb_ids: list[str] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        value = raw_line.strip()
        if not value:
            continue
        if not _PDB_ID_PATTERN.fullmatch(value):
            raise ValueError(
                f"invalid PDB ID on line {line_number} of {path}: {value!r}"
            )
        pdb_ids.append(value.lower())
    if not pdb_ids:
        raise ValueError(f"PDB manifest is empty: {path}")
    if len(pdb_ids) != len(set(pdb_ids)):
        raise ValueError(f"PDB manifest contains duplicate IDs: {path}")
    return pdb_ids


def _load_bioassembly(
    bioassembly_dir: str | Path,
    pdb_id: str,
) -> tuple[Mapping[str, Any], Path]:
    directory = Path(bioassembly_dir)
    gzip_path = directory / f"{pdb_id}.pkl.gz"
    pickle_path = directory / f"{pdb_id}.pkl"
    if gzip_path.is_file():
        with gzip.open(gzip_path, "rb") as handle:
            payload = pickle.load(handle)
        selected_path = gzip_path
    elif pickle_path.is_file():
        with open(pickle_path, "rb") as handle:
            payload = pickle.load(handle)
        selected_path = pickle_path
    else:
        raise FileNotFoundError(
            f"bioassembly not found for {pdb_id}: "
            f"{gzip_path} or {pickle_path}"
        )
    if not isinstance(payload, Mapping):
        raise TypeError(f"bioassembly for {pdb_id} must be a mapping")
    return payload, selected_path.resolve(strict=True)


def collect_polymer_occurrences(
    *,
    pdb_ids: Sequence[str],
    bioassembly_dir: str | Path,
) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, list[dict[str, str]]],
]:
    """Collect exact protein/RNA entities and de-duplicate by query sequence."""

    protein_occurrences: dict[str, list[dict[str, str]]] = {}
    rna_occurrences: dict[str, list[dict[str, str]]] = {}
    for pdb_id in pdb_ids:
        payload, _ = _load_bioassembly(bioassembly_dir, pdb_id)
        sequences = payload.get("sequences")
        entity_poly_type = payload.get("entity_poly_type")
        if not isinstance(sequences, Mapping):
            raise TypeError(f"{pdb_id} bioassembly has invalid sequences")
        if not isinstance(entity_poly_type, Mapping):
            raise TypeError(f"{pdb_id} bioassembly has invalid entity_poly_type")

        for entity_id, polymer_type in entity_poly_type.items():
            if polymer_type == PROTEIN_POLY_TYPE:
                destination = protein_occurrences
            elif polymer_type == RNA_POLY_TYPE:
                destination = rna_occurrences
            else:
                # This intentionally includes DNA and DNA/RNA hybrids.
                continue
            sequence = sequences.get(entity_id)
            if not isinstance(sequence, str) or not sequence:
                raise ValueError(
                    f"{pdb_id} {polymer_type} entity {entity_id!r} "
                    "has no exact sequence"
                )
            if sequence != sequence.strip() or any(
                character.isspace() for character in sequence
            ):
                raise ValueError(
                    f"{pdb_id} entity {entity_id!r} has an invalid sequence"
                )
            destination.setdefault(sequence, []).append(
                {
                    "pdb_id": pdb_id,
                    "entity_id": str(entity_id),
                }
            )

    for occurrences in (protein_occurrences, rna_occurrences):
        for sequence in occurrences:
            occurrences[sequence].sort(
                key=lambda record: (
                    record["pdb_id"],
                    record["entity_id"],
                )
            )
    return protein_occurrences, rna_occurrences


def _aligned_a3m_sequence(sequence: str) -> str:
    """Remove A3M insertion columns while retaining aligned deletion gaps."""

    return "".join(
        character
        for character in sequence
        if not character.islower() and character != "."
    )


def _ungapped_a3m_sequence(sequence: str) -> str:
    return _aligned_a3m_sequence(sequence).replace("-", "")


def inspect_a3m(path: str | Path, *, expected_query: str) -> dict[str, Any]:
    """Hash and structurally inspect one A3M without loading it into memory."""

    requested_path = Path(path)
    if not requested_path.is_file():
        raise FileNotFoundError(requested_path)
    resolved_path = requested_path.resolve(strict=True)
    stat_before = resolved_path.stat()
    if stat_before.st_size <= 0:
        raise ValueError(f"A3M is empty: {resolved_path}")

    digest = hashlib.sha256()
    streamed_size = 0
    record_depth = 0
    current_sequence_parts: list[str] = []
    first_sequence: Optional[str] = None
    aligned_width_min: Optional[int] = None
    aligned_width_max: Optional[int] = None
    aligned_width_mismatch_count = 0
    saw_header = False

    def finish_record() -> None:
        nonlocal first_sequence
        nonlocal aligned_width_min
        nonlocal aligned_width_max
        nonlocal aligned_width_mismatch_count
        if not current_sequence_parts:
            raise ValueError(
                f"A3M record {record_depth} has no sequence: {resolved_path}"
            )
        record_sequence = "".join(current_sequence_parts)
        aligned_sequence = _aligned_a3m_sequence(record_sequence)
        aligned_width = len(aligned_sequence)
        aligned_width_min = (
            aligned_width
            if aligned_width_min is None
            else min(aligned_width_min, aligned_width)
        )
        aligned_width_max = (
            aligned_width
            if aligned_width_max is None
            else max(aligned_width_max, aligned_width)
        )
        if aligned_width != len(expected_query):
            aligned_width_mismatch_count += 1
        if first_sequence is None:
            first_sequence = record_sequence

    with open(resolved_path, "rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            streamed_size += len(raw_line)
            try:
                line = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"A3M is not UTF-8 at line {line_number}: {resolved_path}"
                ) from exc
            if not line:
                continue
            if line.startswith(">"):
                if len(line) == 1:
                    raise ValueError(
                        f"A3M has an empty header at line {line_number}: "
                        f"{resolved_path}"
                    )
                if saw_header:
                    finish_record()
                saw_header = True
                record_depth += 1
                current_sequence_parts = []
                continue
            if not saw_header:
                raise ValueError(
                    f"A3M sequence appears before its first header at line "
                    f"{line_number}: {resolved_path}"
                )
            if any(character.isspace() for character in line):
                raise ValueError(
                    f"A3M sequence contains embedded whitespace at line "
                    f"{line_number}: {resolved_path}"
                )
            current_sequence_parts.append(line)

    if not saw_header:
        raise ValueError(f"A3M contains no records: {resolved_path}")
    finish_record()
    stat_after = resolved_path.stat()
    if (
        stat_before.st_size != stat_after.st_size
        or stat_before.st_mtime_ns != stat_after.st_mtime_ns
        or stat_before.st_ino != stat_after.st_ino
        or streamed_size != stat_after.st_size
    ):
        raise RuntimeError(f"A3M changed while it was being locked: {resolved_path}")
    assert first_sequence is not None
    first_query_exact = (
        _ungapped_a3m_sequence(first_sequence) == expected_query
    )
    if not first_query_exact:
        raise ValueError(
            "A3M first query does not match its exact sequence: "
            f"path={resolved_path}, expected_sha256="
            f"{sequence_sha256(expected_query)}"
        )
    if aligned_width_mismatch_count:
        raise ValueError(
            f"A3M has {aligned_width_mismatch_count} records whose aligned "
            f"width differs from query length {len(expected_query)}: "
            f"{resolved_path}"
        )

    return {
        "path": str(resolved_path),
        "size_bytes": streamed_size,
        "sha256": digest.hexdigest(),
        "record_depth": record_depth,
        "first_query_exact": True,
        "first_query_sha256": sequence_sha256(
            _ungapped_a3m_sequence(first_sequence)
        ),
        "aligned_width_min": aligned_width_min,
        "aligned_width_max": aligned_width_max,
        "aligned_width_mismatch_count": 0,
    }


def _sorted_sequences(
    occurrences: Mapping[str, Sequence[Mapping[str, str]]],
) -> list[str]:
    return sorted(
        occurrences,
        key=lambda sequence: (sequence_sha256(sequence), sequence),
    )


def build_lock_manifest(
    *,
    pdb_list: str | Path,
    bioassembly_dir: str | Path,
    protein_sequence_sources_path: str | Path,
    rna_sequence_mapping_path: str | Path,
    rna_msa_root: str | Path,
) -> dict[str, Any]:
    """Build the deterministic, fully hashed MSA lock payload."""

    pdb_list = Path(pdb_list).expanduser().resolve(strict=True)
    bioassembly_dir = Path(bioassembly_dir).expanduser().resolve(strict=True)
    if not bioassembly_dir.is_dir():
        raise NotADirectoryError(bioassembly_dir)
    protein_sequence_sources_path = (
        Path(protein_sequence_sources_path).expanduser().resolve(strict=True)
    )
    rna_sequence_mapping_path = (
        Path(rna_sequence_mapping_path).expanduser().resolve(strict=True)
    )
    rna_msa_root = Path(rna_msa_root).expanduser().resolve(strict=True)
    if not rna_msa_root.is_dir():
        raise NotADirectoryError(rna_msa_root)

    pdb_ids = read_pdb_ids(pdb_list)
    protein_occurrences, rna_occurrences = collect_polymer_occurrences(
        pdb_ids=pdb_ids,
        bioassembly_dir=bioassembly_dir,
    )
    protein_sources = load_protein_sequence_sources(
        protein_sequence_sources_path
    )
    expected_protein_sequences = set(protein_occurrences)
    configured_protein_sequences = set(protein_sources)
    missing_proteins = expected_protein_sequences - configured_protein_sequences
    extra_proteins = configured_protein_sequences - expected_protein_sequences
    if missing_proteins or extra_proteins:
        raise ValueError(
            "protein sequence-source coverage is not exact: "
            f"missing={len(missing_proteins)}, extra={len(extra_proteins)}"
        )

    rna_mapping = load_rna_sequence_mapping(rna_sequence_mapping_path)
    missing_rna = set(rna_occurrences).difference(rna_mapping)
    if missing_rna:
        raise ValueError(
            f"RNA sequence mapping misses {len(missing_rna)} exact sequences"
        )

    records: list[dict[str, Any]] = []
    for sequence in _sorted_sequences(protein_occurrences):
        source_directory = Path(protein_sources[sequence])
        if not source_directory.is_dir():
            raise NotADirectoryError(source_directory)
        records.append(
            {
                "polymer_type": "protein",
                "sequence": sequence,
                "sequence_sha256": sequence_sha256(sequence),
                "sequence_length": len(sequence),
                "occurrence_count": len(protein_occurrences[sequence]),
                "occurrences": protein_occurrences[sequence],
                "source_directory": str(source_directory.resolve(strict=True)),
                "files": {
                    "pairing": inspect_a3m(
                        source_directory / "pairing.a3m",
                        expected_query=sequence,
                    ),
                    "non_pairing": inspect_a3m(
                        source_directory / "non_pairing.a3m",
                        expected_query=sequence,
                    ),
                },
            }
        )

    for sequence in _sorted_sequences(rna_occurrences):
        mapping_entries = rna_mapping[sequence]
        # load_rna_sequence_mapping has already enforced the exact loader shape.
        eid = str(mapping_entries[0])
        msa_path = rna_msa_root / eid / f"{eid}_all.a3m"
        records.append(
            {
                "polymer_type": "rna",
                "sequence": sequence,
                "sequence_sha256": sequence_sha256(sequence),
                "sequence_length": len(sequence),
                "occurrence_count": len(rna_occurrences[sequence]),
                "occurrences": rna_occurrences[sequence],
                "rna_mapping_first_eid": eid,
                "files": {
                    "unpaired": inspect_a3m(
                        msa_path,
                        expected_query=sequence,
                    )
                },
            }
        )

    records.sort(
        key=lambda record: (
            record["polymer_type"],
            record["sequence_sha256"],
            record["sequence"],
        )
    )
    counts = {
        "target_count": len(pdb_ids),
        "protein_entity_occurrence_count": sum(
            len(items) for items in protein_occurrences.values()
        ),
        "unique_protein_sequence_count": len(protein_occurrences),
        "rna_entity_occurrence_count": sum(
            len(items) for items in rna_occurrences.values()
        ),
        "unique_rna_sequence_count": len(rna_occurrences),
        "protein_file_count": 2 * len(protein_occurrences),
        "rna_file_count": len(rna_occurrences),
        "file_count": 2 * len(protein_occurrences) + len(rna_occurrences),
    }
    return {
        "schema_version": MSA_LOCK_SCHEMA_VERSION,
        "policy_version": MSA_LOCK_POLICY_VERSION,
        "target_ids": pdb_ids,
        "inputs": {
            "pdb_list": str(pdb_list),
            "pdb_list_sha256": file_sha256(pdb_list),
            "target_ids_sha256": canonical_json_sha256(pdb_ids),
            "bioassembly_dir": str(bioassembly_dir),
            "protein_sequence_sources": str(protein_sequence_sources_path),
            "protein_sequence_sources_sha256": file_sha256(
                protein_sequence_sources_path
            ),
            "rna_sequence_mapping": str(rna_sequence_mapping_path),
            "rna_sequence_mapping_sha256": file_sha256(
                rna_sequence_mapping_path
            ),
            "rna_msa_root": str(rna_msa_root),
        },
        "counts": counts,
        "records": records,
    }


def build_msa_lock(
    *,
    pdb_list: str | Path,
    bioassembly_dir: str | Path,
    protein_sequence_sources_path: str | Path,
    rna_sequence_mapping_path: str | Path,
    rna_msa_root: str | Path,
    manifest_path: str | Path,
    success_path: str | Path,
) -> dict[str, Any]:
    """Build and atomically publish the manifest and success marker."""

    manifest_path = Path(manifest_path).expanduser().resolve(strict=False)
    success_path = Path(success_path).expanduser().resolve(strict=False)
    if manifest_path == success_path:
        raise ValueError("manifest_path and success_path must differ")
    success_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        success_path.unlink()
    except FileNotFoundError:
        pass

    manifest = build_lock_manifest(
        pdb_list=pdb_list,
        bioassembly_dir=bioassembly_dir,
        protein_sequence_sources_path=protein_sequence_sources_path,
        rna_sequence_mapping_path=rna_sequence_mapping_path,
        rna_msa_root=rna_msa_root,
    )
    atomic_json_dump(manifest, manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        persisted_manifest = json.load(handle)
    if persisted_manifest != manifest:
        raise RuntimeError("persisted MSA lock manifest failed round-trip validation")

    success = {
        "schema_version": MSA_LOCK_SCHEMA_VERSION,
        "policy_version": MSA_LOCK_POLICY_VERSION,
        "status": "complete",
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "inputs": dict(manifest["inputs"]),
        "counts": dict(manifest["counts"]),
        "completed_at_unix": time.time(),
    }
    atomic_json_dump(success, success_path)
    return {
        "status": "complete",
        "manifest_path": str(manifest_path),
        "success_path": str(success_path),
        "manifest_sha256": success["manifest_sha256"],
        **manifest["counts"],
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the complete, hash-locked PXM evaluation MSA inventory."
    )
    parser.add_argument(
        "--pdb-list",
        default=str(DEFAULT_PXM_MANIFEST_ROOT / "combined_le1024.txt"),
    )
    parser.add_argument(
        "--bioassembly-dir",
        default=str(DEFAULT_OFFICIAL_ROOT / "recentPDB_bioassembly"),
    )
    parser.add_argument(
        "--protein-sequence-sources",
        default=str(
            DEFAULT_PXM_INPUT_ROOT
            / "manifests/protein_sequence_sources.json"
        ),
    )
    parser.add_argument(
        "--rna-sequence-mapping",
        default=str(
            DEFAULT_OFFICIAL_ROOT
            / "rna_msa/rna_sequence_to_pdb_chains.json"
        ),
    )
    parser.add_argument(
        "--rna-msa-root",
        default=str(DEFAULT_OFFICIAL_ROOT / "rna_msa/msas"),
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_PXM_INPUT_ROOT / "msa_lock"),
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--success", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    output_root = Path(args.output_root)
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else output_root / MSA_LOCK_MANIFEST_FILENAME
    )
    success_path = (
        Path(args.success)
        if args.success
        else output_root / MSA_LOCK_SUCCESS_FILENAME
    )
    summary = build_msa_lock(
        pdb_list=args.pdb_list,
        bioassembly_dir=args.bioassembly_dir,
        protein_sequence_sources_path=args.protein_sequence_sources,
        rna_sequence_mapping_path=args.rna_sequence_mapping,
        rna_msa_root=args.rna_msa_root,
        manifest_path=manifest_path,
        success_path=success_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
