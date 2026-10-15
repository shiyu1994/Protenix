#!/usr/bin/env python3
"""Audit the real source-A3M depth of a Protenix benchmark index.

This is a read-only database audit.  For every PDB row it loads the protein
entity sequences from ``mmcif_bioassembly``, resolves them through
``seq_to_pdb_index.json``, and scans ``pairing.a3m`` and
``non_pairing.a3m``.  The effective depth reproduces the relevant
``PAIR_MSA_AS_UNPAIRED_ONLY=true`` semantics: the two sources are combined,
lower-case A3M insertions are ignored for deduplication, and a missing or
invalid alignment falls back to the query row.

The audit reports source capacity, not the exact MSA depth of a stochastic
crop.  A crop can activate a subset of assembly chains; the authoritative
per-step value must still be taken from the runtime ``N_msa`` feature.

Example::

    python scripts/benchmark/audit_msa_depth.py \
      --index data/benchmark_long_protein_context_1024_800/indices.csv \
      --output-json /tmp/msa1024_800.json \
      --output-csv /tmp/msa1024_800.csv \
      --target-depth 2048 --limit 20
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import string
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


DEFAULT_PROTENIX_ROOT = Path(
    os.environ.get(
        "PROTENIX_ROOT_DIR",
        "/mnt/project_abd/public_databases/protenix.20260101",
    )
)
DEFAULT_INDEX = Path("data/benchmark_long_protein_context_1024_800/indices.csv")
LOWERCASE_TRANSLATION = str.maketrans("", "", string.ascii_lowercase)
ROW_CSV_FIELDS = (
    "row_index",
    "pdb_id",
    "num_tokens",
    "protein_entity_count",
    "referenced_protein_entity_count",
    "protein_entity_ids",
    "referenced_protein_entity_ids",
    "min_effective_depth",
    "max_effective_depth",
    "referenced_min_effective_depth",
    "referenced_max_effective_depth",
    "potential_sampled_depth",
    "referenced_potential_sampled_depth",
    "any_entity_reaches_target",
    "all_entities_reach_target",
    "any_referenced_entity_reaches_target",
    "all_referenced_entities_reach_target",
    "missing_mapping_entity_count",
    "missing_a3m_entity_count",
    "invalid_a3m_entity_count",
)


def sha256_file(path: Path) -> str:
    """Return the SHA256 of a file without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_index_rows(index_path: Path, limit: int | None = None) -> list[dict[str, str]]:
    """Read benchmark rows in stable file order."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    with _open_text(index_path) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"benchmark index is empty: {index_path}")
    if "pdb_id" not in rows[0]:
        raise ValueError(f"benchmark index is missing pdb_id: {index_path}")
    return rows[:limit] if limit is not None else rows


def iter_a3m_sequences(path: Path) -> Iterator[str]:
    """Yield sequences from a FASTA/A3M file, including multiline records."""
    sequence_lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sequence_lines:
                    yield "".join(sequence_lines)
                    sequence_lines = []
                continue
            sequence_lines.append(line)
    if sequence_lines:
        yield "".join(sequence_lines)


def canonical_a3m_sequence(sequence: str) -> str:
    """Match ``RawMsa`` deduplication by removing lower-case insertions."""
    return sequence.translate(LOWERCASE_TRANSLATION)


def _scan_a3m(path: Path, query: str) -> dict[str, Any]:
    """Scan one source and return both raw and training-effective rows."""
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": 0,
            "raw_depth": 0,
            "unique_depth": 0,
            "query_length_matches": None,
            "fallback_to_query": False,
            "effective_sequences": set(),
        }

    raw_depth = 0
    first_aligned_length: int | None = None
    unique_sequences: set[str] = set()
    for sequence in iter_a3m_sequences(path):
        canonical = canonical_a3m_sequence(sequence)
        if first_aligned_length is None:
            first_aligned_length = len(canonical)
        raw_depth += 1
        unique_sequences.add(canonical)

    # Empty files are ignored by MSASourceManager.  A non-empty file whose
    # first row has the wrong aligned length is converted to query-only by
    # RawMsa._verify_query().
    query_length_matches = (
        first_aligned_length == len(canonical_a3m_sequence(query))
        if first_aligned_length is not None
        else None
    )
    fallback_to_query = raw_depth > 0 and not query_length_matches
    effective_sequences = (
        {canonical_a3m_sequence(query)} if fallback_to_query else unique_sequences
    )
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "raw_depth": raw_depth,
        "unique_depth": len(unique_sequences),
        "query_length_matches": query_length_matches,
        "fallback_to_query": fallback_to_query,
        "effective_sequences": effective_sequences,
    }


def _public_source_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in stats.items() if key != "effective_sequences"}


def normalize_entity_id(value: Any) -> str | None:
    """Normalize CSV entity IDs such as ``2.0`` to bioassembly key ``2``."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def _mapping_relative_path(value: Any) -> str:
    # This intentionally mirrors MSASourceManager's ``str(mapping[key])``.
    return str(value)


def audit_protein_entity(
    *,
    pdb_id: str,
    entity_id: str,
    sequence: str,
    sequence_to_msa_dir: Mapping[str, Any],
    msa_root: Path,
    target_depth: int,
    max_msa_size: int,
    referenced_by_index: bool,
) -> dict[str, Any]:
    """Audit a single protein entity without writing to the source database."""
    mapping_value = sequence_to_msa_dir.get(sequence)
    if mapping_value is None:
        effective_depth = 1
        return {
            "pdb_id": pdb_id,
            "entity_id": entity_id,
            "sequence_length": len(sequence),
            "referenced_by_index": referenced_by_index,
            "mapping_found": False,
            "msa_relative_dir": None,
            "pairing": None,
            "non_pairing": None,
            "combined_unique_depth": 1,
            "effective_depth": effective_depth,
            "potential_sampled_depth": min(effective_depth, target_depth),
            "reaches_target": effective_depth >= target_depth,
            "missing_all_a3m": True,
            "has_invalid_a3m": False,
        }

    relative_dir = _mapping_relative_path(mapping_value)
    directory = msa_root / relative_dir
    pairing = _scan_a3m(directory / "pairing.a3m", sequence)
    non_pairing = _scan_a3m(directory / "non_pairing.a3m", sequence)
    combined = set(pairing["effective_sequences"])
    combined.update(non_pairing["effective_sequences"])
    if not combined:
        combined.add(canonical_a3m_sequence(sequence))
    combined_unique_depth = len(combined)
    effective_depth = min(combined_unique_depth, max_msa_size)
    missing_all_a3m = not pairing["exists"] and not non_pairing["exists"]
    has_invalid_a3m = bool(
        pairing["fallback_to_query"] or non_pairing["fallback_to_query"]
    )
    return {
        "pdb_id": pdb_id,
        "entity_id": entity_id,
        "sequence_length": len(sequence),
        "referenced_by_index": referenced_by_index,
        "mapping_found": True,
        "msa_relative_dir": relative_dir,
        "pairing": _public_source_stats(pairing),
        "non_pairing": _public_source_stats(non_pairing),
        "combined_unique_depth": combined_unique_depth,
        "effective_depth": effective_depth,
        "potential_sampled_depth": min(effective_depth, target_depth),
        "reaches_target": effective_depth >= target_depth,
        "missing_all_a3m": missing_all_a3m,
        "has_invalid_a3m": has_invalid_a3m,
    }


def _referenced_protein_entities(row: Mapping[str, Any]) -> set[str]:
    entity_ids: set[str] = set()
    for suffix in ("1", "2"):
        if str(row.get(f"mol_{suffix}_type", "")).strip().lower() != "prot":
            continue
        entity_id = normalize_entity_id(row.get(f"entity_{suffix}_id"))
        if entity_id is not None:
            entity_ids.add(entity_id)
    return entity_ids


def _aggregate_depths(entities: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    depths = [int(entity["effective_depth"]) for entity in entities]
    return (min(depths), max(depths)) if depths else (0, 0)


def audit_index_row(
    *,
    row_index: int,
    row: Mapping[str, Any],
    bioassembly_dir: Path,
    sequence_to_msa_dir: Mapping[str, Any],
    msa_root: Path,
    target_depth: int,
    max_msa_size: int,
) -> dict[str, Any]:
    """Audit all protein entities for one benchmark index row."""
    pdb_id = str(row["pdb_id"]).strip().lower()
    bioassembly_path = bioassembly_dir / f"{pdb_id}.pkl.gz"
    if not bioassembly_path.is_file():
        raise FileNotFoundError(f"missing bioassembly for {pdb_id}: {bioassembly_path}")
    with gzip.open(bioassembly_path, "rb") as handle:
        bioassembly = pickle.load(handle)  # nosec B301: trusted local database

    sequences = {
        str(key): str(value) for key, value in bioassembly.get("sequences", {}).items()
    }
    entity_poly_type = {
        str(key): str(value)
        for key, value in bioassembly.get("entity_poly_type", {}).items()
    }
    protein_entity_ids = sorted(
        entity_id
        for entity_id in sequences
        if entity_poly_type.get(entity_id) == "polypeptide(L)"
    )
    referenced_ids = _referenced_protein_entities(row)
    entities = [
        audit_protein_entity(
            pdb_id=pdb_id,
            entity_id=entity_id,
            sequence=sequences[entity_id],
            sequence_to_msa_dir=sequence_to_msa_dir,
            msa_root=msa_root,
            target_depth=target_depth,
            max_msa_size=max_msa_size,
            referenced_by_index=entity_id in referenced_ids,
        )
        for entity_id in protein_entity_ids
    ]
    referenced_entities = [
        entity for entity in entities if entity["referenced_by_index"]
    ]
    missing_referenced_ids = sorted(referenced_ids.difference(protein_entity_ids))
    min_depth, max_depth = _aggregate_depths(entities)
    referenced_min_depth, referenced_max_depth = _aggregate_depths(referenced_entities)

    any_reaches = any(entity["reaches_target"] for entity in entities)
    all_reach = bool(entities) and all(entity["reaches_target"] for entity in entities)
    any_ref_reaches = any(entity["reaches_target"] for entity in referenced_entities)
    all_ref_reach = bool(referenced_entities) and all(
        entity["reaches_target"] for entity in referenced_entities
    )
    return {
        "row_index": row_index,
        "pdb_id": pdb_id,
        "num_tokens": int(float(row["num_tokens"])) if row.get("num_tokens") else None,
        "protein_entity_ids": protein_entity_ids,
        "referenced_protein_entity_ids": sorted(referenced_ids),
        "missing_referenced_protein_entity_ids": missing_referenced_ids,
        "protein_entity_count": len(entities),
        "referenced_protein_entity_count": len(referenced_entities),
        "min_effective_depth": min_depth,
        "max_effective_depth": max_depth,
        "referenced_min_effective_depth": referenced_min_depth,
        "referenced_max_effective_depth": referenced_max_depth,
        # FeatureAssemblyLine uses the maximum unpaired depth over active
        # chains.  These are source-capacity upper bounds for all assembly
        # proteins and for the index-referenced proteins respectively.
        "potential_sampled_depth": min(max_depth, target_depth),
        "referenced_potential_sampled_depth": min(referenced_max_depth, target_depth),
        "any_entity_reaches_target": any_reaches,
        "all_entities_reach_target": all_reach,
        "any_referenced_entity_reaches_target": any_ref_reaches,
        "all_referenced_entities_reach_target": all_ref_reach,
        "missing_mapping_entity_count": sum(
            not entity["mapping_found"] for entity in entities
        ),
        "missing_a3m_entity_count": sum(
            entity["missing_all_a3m"] for entity in entities
        ),
        "invalid_a3m_entity_count": sum(
            entity["has_invalid_a3m"] for entity in entities
        ),
        "entities": entities,
    }


def _count_rows(records: Iterable[Mapping[str, Any]], field: str) -> int:
    return sum(bool(record[field]) for record in records)


def summarize_records(
    records: Sequence[Mapping[str, Any]], target_depth: int
) -> dict[str, Any]:
    """Build row- and entity-level target-depth coverage statistics."""
    entities = [entity for record in records for entity in record["entities"]]
    referenced_entities = [
        entity for entity in entities if entity["referenced_by_index"]
    ]
    return {
        "processed_rows": len(records),
        "unique_pdbs": len({record["pdb_id"] for record in records}),
        "target_depth": target_depth,
        "protein_entities": len(entities),
        "referenced_protein_entities": len(referenced_entities),
        "protein_entities_reaching_target": sum(
            bool(entity["reaches_target"]) for entity in entities
        ),
        "referenced_protein_entities_reaching_target": sum(
            bool(entity["reaches_target"]) for entity in referenced_entities
        ),
        "rows_any_entity_reaching_target": _count_rows(
            records, "any_entity_reaches_target"
        ),
        "rows_all_entities_reaching_target": _count_rows(
            records, "all_entities_reach_target"
        ),
        "rows_any_referenced_entity_reaching_target": _count_rows(
            records, "any_referenced_entity_reaches_target"
        ),
        "rows_all_referenced_entities_reaching_target": _count_rows(
            records, "all_referenced_entities_reach_target"
        ),
        "missing_mapping_entities": sum(
            int(record["missing_mapping_entity_count"]) for record in records
        ),
        "missing_a3m_entities": sum(
            int(record["missing_a3m_entity_count"]) for record in records
        ),
        "invalid_a3m_entities": sum(
            int(record["invalid_a3m_entity_count"]) for record in records
        ),
    }


def audit_msa_depth(
    *,
    index_path: Path,
    mapping_path: Path,
    bioassembly_dir: Path,
    msa_root: Path,
    target_depth: int = 2048,
    max_msa_size: int = 16384,
    limit: int | None = None,
) -> dict[str, Any]:
    """Audit a benchmark index and return a JSON-serializable report."""
    if target_depth <= 0:
        raise ValueError("target_depth must be positive")
    if max_msa_size < target_depth:
        raise ValueError("max_msa_size must be greater than or equal to target_depth")
    index_path = index_path.resolve()
    mapping_path = mapping_path.resolve()
    bioassembly_dir = bioassembly_dir.resolve()
    msa_root = msa_root.resolve()
    rows = read_index_rows(index_path, limit=limit)
    with mapping_path.open("r", encoding="utf-8") as handle:
        sequence_to_msa_dir = json.load(handle)
    if not isinstance(sequence_to_msa_dir, dict):
        raise TypeError(f"MSA mapping must be a JSON object: {mapping_path}")

    records = [
        audit_index_row(
            row_index=row_index,
            row=row,
            bioassembly_dir=bioassembly_dir,
            sequence_to_msa_dir=sequence_to_msa_dir,
            msa_root=msa_root,
            target_depth=target_depth,
            max_msa_size=max_msa_size,
        )
        for row_index, row in enumerate(rows)
    ]
    return {
        "schema_version": 1,
        "audit_semantics": "pairing_and_non_pairing_as_unpaired_deduplicated",
        "interpretation": (
            "Source-capacity audit; stochastic crop runtime N_msa remains authoritative."
        ),
        "inputs": {
            "index": str(index_path),
            "index_sha256": sha256_file(index_path),
            "mapping": str(mapping_path),
            "bioassembly_dir": str(bioassembly_dir),
            "msa_root": str(msa_root),
            "limit": limit,
            "target_depth": target_depth,
            "max_msa_size": max_msa_size,
        },
        "summary": summarize_records(records, target_depth),
        "records": records,
    }


def _csv_row(record: Mapping[str, Any]) -> dict[str, Any]:
    result = {field: record[field] for field in ROW_CSV_FIELDS}
    result["protein_entity_ids"] = ";".join(record["protein_entity_ids"])
    result["referenced_protein_entity_ids"] = ";".join(
        record["referenced_protein_entity_ids"]
    )
    return result


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_audit_outputs(
    report: Mapping[str, Any], json_path: Path, csv_path: Path
) -> None:
    """Write stable JSON plus one aggregate CSV row per benchmark row."""
    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _write_text_atomic(json_path.resolve(), json_text)

    csv_path = csv_path.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.parent / f".{csv_path.name}.{os.getpid()}.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_row(record) for record in report["records"])
    os.replace(temporary, csv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_PROTENIX_ROOT / "common/seq_to_pdb_index.json",
    )
    parser.add_argument(
        "--bioassembly-dir",
        type=Path,
        default=DEFAULT_PROTENIX_ROOT / "mmcif_bioassembly",
    )
    parser.add_argument(
        "--msa-root",
        type=Path,
        default=DEFAULT_PROTENIX_ROOT / "mmcif_msa_template",
    )
    parser.add_argument("--target-depth", type=int, default=2048)
    parser.add_argument("--max-msa-size", type=int, default=16384)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Audit only the first N index rows (default: all rows).",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_msa_depth(
        index_path=args.index,
        mapping_path=args.mapping,
        bioassembly_dir=args.bioassembly_dir,
        msa_root=args.msa_root,
        target_depth=args.target_depth,
        max_msa_size=args.max_msa_size,
        limit=args.limit,
    )
    write_audit_outputs(report, args.output_json, args.output_csv)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
