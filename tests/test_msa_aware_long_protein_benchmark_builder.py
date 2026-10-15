import csv
import gzip
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.benchmark.build_msa_aware_long_protein_benchmark import (
    build_msa_aware_benchmark_index,
    main,
    validate_msa_aware_benchmark_index,
    write_ready_marker,
)


def _write_a3m(path: Path, query: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f">query\n{query}\n>hit-1\n{query[:2]}-{query[3:]}\n"
        f">hit-2\n{query[0]}-{query[2:]}\n",
        encoding="utf-8",
    )


def _write_bioassembly(path: Path, sequences: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        pickle.dump(
            {
                "sequences": sequences,
                "entity_poly_type": {
                    entity_id: "polypeptide(L)" for entity_id in sequences
                },
            },
            handle,
        )


def _source_rows() -> pd.DataFrame:
    specs = [
        ("p001", "c001", 1500),  # missing mapping -> reject
        ("p002", "c002", 1490),  # accept
        ("p003", "c003", 1480),  # second assembly entity is shallow -> reject
        ("p004", "c004", 1470),  # accept
        ("p005", "c002", 1460),  # duplicate cluster -> not in audit stream
        ("p006", "c006", 1450),  # accept and complete the requested set
    ]
    return pd.DataFrame(
        [
            {
                "pdb_id": pdb_id,
                "chain_1_id": "A",
                "chain_2_id": None,
                "entity_1_id": 1,
                "entity_2_id": None,
                "mol_1_type": "prot",
                "mol_2_type": None,
                "type": "chain",
                "num_tokens": num_tokens,
                "cluster_id": cluster_id,
            }
            for pdb_id, cluster_id, num_tokens in specs
        ]
    )


def _database(root: Path):
    source_index = root / "source.csv"
    output_dir = root / "output"
    mapping_path = root / "seq_to_pdb_index.json"
    bioassembly_dir = root / "bioassembly"
    msa_root = root / "msa"
    _source_rows().to_csv(source_index, index=False)

    sequences = {
        "p001": {"1": "AAAA"},
        "p002": {"1": "BBBB"},
        "p003": {"1": "CCCC", "2": "DDDD"},
        "p004": {"1": "EEEE"},
        "p005": {"1": "FFFF"},
        "p006": {"1": "GGGG"},
    }
    for pdb_id, entity_sequences in sequences.items():
        _write_bioassembly(bioassembly_dir / f"{pdb_id}.pkl.gz", entity_sequences)

    mapping = {
        "BBBB": "deep/b",
        "CCCC": "deep/c",
        "EEEE": "deep/e",
        "FFFF": "deep/f",
        "GGGG": "deep/g",
    }
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    for sequence, relative_dir in mapping.items():
        _write_a3m(msa_root / relative_dir / "pairing.a3m", sequence)
    return source_index, output_dir, mapping_path, bioassembly_dir, msa_root


def _common(root: Path):
    source_index, output_dir, mapping_path, bioassembly_dir, msa_root = _database(root)
    return {
        "source_index": source_index,
        "output_dir": output_dir,
        "mapping_path": mapping_path,
        "bioassembly_dir": bioassembly_dir,
        "msa_root": msa_root,
        "max_rows": 3,
        "min_num_tokens": 1025,
        "max_num_tokens": 1536,
        "selection": "top_tokens",
        "seed": 42,
        "target_depth": 3,
        "max_msa_size": 16,
    }


def test_builder_replaces_shallow_candidates_and_requires_every_entity(tmp_path):
    common = _common(tmp_path)
    selected, summary = build_msa_aware_benchmark_index(**common, progress_every=0)

    assert selected["pdb_id"].tolist() == ["p002", "p004", "p006"]
    assert selected["pdb_id"].is_unique
    assert selected["cluster_id"].is_unique
    assert summary["rows"] == 3
    assert summary["audited_candidate_rows"] == 5
    assert summary["candidate_status_counts"] == {
        "accepted": 3,
        "rejected_msa_depth": 2,
    }
    assert summary["accepted_min_effective_depth"] == 3

    report = json.loads(
        (common["output_dir"] / "msa_depth.json").read_text(encoding="utf-8")
    )
    assert [record["pdb_id"] for record in report["records"]] == [
        "p002",
        "p004",
        "p006",
    ]
    assert all(record["all_entities_reach_target"] for record in report["records"])
    with (common["output_dir"] / "msa_depth_candidates.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        trace = list(csv.DictReader(handle))
    assert [row["pdb_id"] for row in trace] == [
        "p001",
        "p002",
        "p003",
        "p004",
        "p006",
    ]
    assert trace[2]["min_effective_depth"] == "1"


def test_validator_locks_all_artifact_hashes_and_depth_evidence(tmp_path):
    common = _common(tmp_path)
    built, built_summary = build_msa_aware_benchmark_index(**common, progress_every=0)
    ready_marker = tmp_path / "run/dataset_ready.json"
    write_ready_marker(ready_marker, built_summary)
    validated, summary = validate_msa_aware_benchmark_index(
        **common, ready_marker=ready_marker
    )
    assert validated["pdb_id"].tolist() == built["pdb_id"].tolist()
    assert summary["index_sha256"] == built_summary["index_sha256"]

    audit_path = common["output_dir"] / "msa_depth.json"
    audit_path.write_text(
        audit_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="msa_depth_json_sha256"):
        validate_msa_aware_benchmark_index(**common, ready_marker=ready_marker)


def test_builder_fails_if_not_enough_qualified_unique_rows(tmp_path):
    common = _common(tmp_path)
    common["max_rows"] = 4
    with pytest.raises(ValueError, match="only 3 of 4 required rows"):
        build_msa_aware_benchmark_index(**common, progress_every=0)


def test_validate_only_publishes_missing_run_local_ready_marker(
    tmp_path, monkeypatch, capsys
):
    common = _common(tmp_path)
    build_msa_aware_benchmark_index(**common, progress_every=0)
    ready_marker = tmp_path / "new-run/dataset_ready.json"
    argv = [
        "build_msa_aware_long_protein_benchmark.py",
        "--source-index",
        str(common["source_index"]),
        "--output-dir",
        str(common["output_dir"]),
        "--mapping",
        str(common["mapping_path"]),
        "--bioassembly-dir",
        str(common["bioassembly_dir"]),
        "--msa-root",
        str(common["msa_root"]),
        "--max-rows",
        str(common["max_rows"]),
        "--min-num-tokens",
        str(common["min_num_tokens"]),
        "--max-num-tokens",
        str(common["max_num_tokens"]),
        "--selection",
        str(common["selection"]),
        "--seed",
        str(common["seed"]),
        "--target-depth",
        str(common["target_depth"]),
        "--max-msa-size",
        str(common["max_msa_size"]),
        "--validate-only",
        "--ready-marker",
        str(ready_marker),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert main() == 0
    capsys.readouterr()
    assert ready_marker.is_file()
    validate_msa_aware_benchmark_index(**common, ready_marker=ready_marker)
