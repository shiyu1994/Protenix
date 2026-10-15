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

"""Prepare the immutable protein-input inventory for the PXM-Legacy proxy.

The generated sequence-source manifest is consumed by
``build_af3_pxm_template_cache.py``.  Existing public Protenix MSA/template
artifacts are referenced in place; genuinely missing sequences (currently the
186-residue protein in 7tdw) are resolved only through an evaluation-owned
overlay, so this tool never mutates the public database.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


PROTEIN_POLY_TYPE = "polypeptide(L)"
DEFAULT_EVAL_ROOT = Path(
    "/mnt/project_abd/klp/proteinx_eval_data/cache/"
    "pxm_legacy_af3_like_cutoff20210930_v1"
)
DEFAULT_MANIFEST = Path(
    "/mnt/project_abd/klp/proteinx_eval_data/manifests/"
    "pxm_legacy_af3_like/combined_le1024.txt"
)
DEFAULT_OFFICIAL_ROOT = Path(
    "/mnt/project_abd/public_databases/protenix.20260101"
)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _load_json_mapping(path: Path, *, optional: bool = False) -> dict[str, str]:
    if optional and not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"mapping must be a JSON object: {path}")
    mapping: dict[str, str] = {}
    for sequence, relative_directory in payload.items():
        if not isinstance(sequence, str) or not sequence:
            raise ValueError(f"invalid sequence key in {path}")
        if not isinstance(relative_directory, str) or not relative_directory:
            raise ValueError(
                f"invalid relative directory for sequence in {path}"
            )
        relative_path = Path(relative_directory)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"unsafe relative directory {relative_directory!r} in {path}"
            )
        mapping[sequence] = relative_directory
    return mapping


def _read_pdb_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    pdb_ids = [
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not pdb_ids:
        raise ValueError(f"PDB manifest is empty: {path}")
    if len(pdb_ids) != len(set(pdb_ids)):
        raise ValueError(f"PDB manifest contains duplicate IDs: {path}")
    return pdb_ids


def _load_bioassembly(directory: Path, pdb_id: str) -> Mapping[str, Any]:
    gzip_path = directory / f"{pdb_id}.pkl.gz"
    pickle_path = directory / f"{pdb_id}.pkl"
    if gzip_path.is_file():
        with gzip.open(gzip_path, "rb") as handle:
            payload = pickle.load(handle)
    elif pickle_path.is_file():
        with pickle_path.open("rb") as handle:
            payload = pickle.load(handle)
    else:
        raise FileNotFoundError(
            f"bioassembly not found for {pdb_id}: "
            f"{gzip_path} or {pickle_path}"
        )
    if not isinstance(payload, Mapping):
        raise TypeError(f"bioassembly for {pdb_id} must be a mapping")
    return payload


def _protein_occurrences(
    *,
    pdb_ids: Iterable[str],
    bioassembly_dir: Path,
) -> dict[str, list[dict[str, str]]]:
    occurrences: dict[str, list[dict[str, str]]] = {}
    for pdb_id in pdb_ids:
        payload = _load_bioassembly(bioassembly_dir, pdb_id)
        sequences = payload.get("sequences")
        entity_poly_type = payload.get("entity_poly_type")
        if not isinstance(sequences, Mapping):
            raise TypeError(f"{pdb_id} bioassembly has invalid sequences")
        if not isinstance(entity_poly_type, Mapping):
            raise TypeError(
                f"{pdb_id} bioassembly has invalid entity_poly_type"
            )
        for entity_id, poly_type in entity_poly_type.items():
            if poly_type != PROTEIN_POLY_TYPE:
                continue
            sequence = sequences.get(entity_id)
            if not isinstance(sequence, str) or not sequence:
                raise ValueError(
                    f"{pdb_id} protein entity {entity_id} has no sequence"
                )
            occurrences.setdefault(sequence, []).append(
                {"pdb_id": pdb_id, "entity_id": str(entity_id)}
            )
    return occurrences


def _has_template_search_artifact(directory: Path) -> bool:
    return any(
        path.is_file()
        for path in (
            directory / "concat.hhr",
            directory / "hmmsearch.a3m",
        )
    )


def _read_first_a3m_sequence(path: Path) -> str:
    residues: list[str] = []
    saw_header = False
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if saw_header and residues:
                    break
                saw_header = True
                continue
            if not saw_header:
                raise ValueError(f"A3M sequence appears before its header: {path}")
            residues.append(line)
    return "".join(
        residue
        for residue in "".join(residues)
        if not residue.islower() and residue not in {"-", "."}
    )


def _resolve_sources(
    *,
    occurrences: Mapping[str, list[dict[str, str]]],
    official_mapping: Mapping[str, str],
    official_raw_root: Path,
    overlay_mapping: Mapping[str, str],
    overlay_raw_root: Path,
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    resolved: dict[str, Path] = {}
    inventory: list[dict[str, Any]] = []
    for sequence in sorted(occurrences, key=lambda item: hashlib.sha256(item.encode()).hexdigest()):
        source_name = ""
        relative_directory = ""
        source_directory: Path | None = None
        if sequence in official_mapping:
            source_name = "official"
            relative_directory = official_mapping[sequence]
            source_directory = official_raw_root / relative_directory
        elif sequence in overlay_mapping:
            source_name = "evaluation_overlay"
            relative_directory = overlay_mapping[sequence]
            source_directory = overlay_raw_root / relative_directory

        reasons: list[str] = []
        if source_directory is None:
            reasons.append("sequence_mapping_missing")
        else:
            if not _has_template_search_artifact(source_directory):
                reasons.append("template_search_artifact_missing")
            for msa_filename in ("pairing.a3m", "non_pairing.a3m"):
                msa_path = source_directory / msa_filename
                if not msa_path.is_file():
                    reasons.append(f"{msa_filename}_missing")
                elif msa_path.stat().st_size == 0:
                    reasons.append(f"{msa_filename}_empty")
                else:
                    try:
                        first_sequence = _read_first_a3m_sequence(msa_path)
                    except (OSError, UnicodeError, ValueError):
                        reasons.append(f"{msa_filename}_unreadable")
                    else:
                        if first_sequence != sequence:
                            reasons.append(f"{msa_filename}_query_mismatch")

        digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        record: dict[str, Any] = {
            "sequence_sha256": digest,
            "sequence_length": len(sequence),
            "source": source_name or None,
            "source_relative_dir": relative_directory or None,
            "source_directory": (
                str(source_directory) if source_directory is not None else None
            ),
            "occurrence_count": len(occurrences[sequence]),
            "occurrences": occurrences[sequence],
            "status": "complete" if not reasons else "missing",
            "reasons": reasons,
        }
        inventory.append(record)
        if not reasons and source_directory is not None:
            resolved[sequence] = source_directory
    return resolved, inventory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb-list", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--bioassembly-dir",
        type=Path,
        default=DEFAULT_OFFICIAL_ROOT / "recentPDB_bioassembly",
    )
    parser.add_argument(
        "--official-sequence-mapping",
        type=Path,
        default=DEFAULT_OFFICIAL_ROOT / "common/seq_to_pdb_index.json",
    )
    parser.add_argument(
        "--official-raw-root",
        type=Path,
        default=DEFAULT_OFFICIAL_ROOT / "mmcif_msa_template",
    )
    parser.add_argument(
        "--overlay-sequence-mapping",
        type=Path,
        default=DEFAULT_EVAL_ROOT / "msa_template/seq_to_msa_template_dir.json",
    )
    parser.add_argument(
        "--overlay-raw-root",
        type=Path,
        default=DEFAULT_EVAL_ROOT / "msa_template",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_EVAL_ROOT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pdb_ids = _read_pdb_ids(args.pdb_list)
    occurrences = _protein_occurrences(
        pdb_ids=pdb_ids,
        bioassembly_dir=args.bioassembly_dir,
    )
    official_mapping = _load_json_mapping(args.official_sequence_mapping)
    overlay_mapping = _load_json_mapping(
        args.overlay_sequence_mapping,
        optional=True,
    )
    resolved, inventory = _resolve_sources(
        occurrences=occurrences,
        official_mapping=official_mapping,
        official_raw_root=args.official_raw_root,
        overlay_mapping=overlay_mapping,
        overlay_raw_root=args.overlay_raw_root,
    )

    output_manifest = args.output_root / "manifests/protein_sequence_sources.json"
    output_inventory = args.output_root / "manifests/protein_sequence_inventory.json"
    success_path = args.output_root / "manifests/_INPUT_SUCCESS.json"
    missing = [record for record in inventory if record["status"] != "complete"]

    _atomic_write_json(
        output_manifest,
        {sequence: str(path) for sequence, path in resolved.items()},
    )
    _atomic_write_json(
        output_inventory,
        {
            "schema_version": 1,
            "pdb_manifest": str(args.pdb_list),
            "pdb_count": len(pdb_ids),
            "unique_protein_sequence_count": len(occurrences),
            "protein_entity_occurrence_count": sum(
                len(items) for items in occurrences.values()
            ),
            "resolved_sequence_count": len(resolved),
            "missing_sequence_count": len(missing),
            "records": inventory,
        },
    )
    if success_path.exists():
        success_path.unlink()
    if missing:
        missing_summary = [
            {
                "sequence_sha256": record["sequence_sha256"],
                "occurrences": record["occurrences"],
                "reasons": record["reasons"],
            }
            for record in missing
        ]
        raise RuntimeError(
            "evaluation MSA/template input coverage is incomplete: "
            + json.dumps(missing_summary, sort_keys=True)
        )

    _atomic_write_json(
        success_path,
        {
            "schema_version": 1,
            "status": "complete",
            "created_at_unix": time.time(),
            "pdb_manifest": str(args.pdb_list),
            "pdb_count": len(pdb_ids),
            "unique_protein_sequence_count": len(occurrences),
            "protein_entity_occurrence_count": sum(
                len(items) for items in occurrences.values()
            ),
            "resolved_sequence_count": len(resolved),
            "missing_sequence_count": 0,
            "sequence_sources": str(output_manifest),
            "inventory": str(output_inventory),
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "pdb_count": len(pdb_ids),
                "unique_protein_sequence_count": len(occurrences),
                "resolved_sequence_count": len(resolved),
                "sequence_sources": str(output_manifest),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
