import pandas as pd
import pytest

from scripts.benchmark.build_long_protein_benchmark import (
    build_benchmark_index,
    filter_long_protein_context,
    select_benchmark_rows,
    summarize_selection,
    validate_benchmark_index,
    write_ready_marker,
)


def _toy_indices() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pdb_id": "p001",
                "chain_1_id": "A",
                "chain_2_id": None,
                "mol_1_type": "prot",
                "mol_2_type": None,
                "type": "chain",
                "num_tokens": 900,
                "cluster_id": "cluster_a",
            },
            {
                "pdb_id": "p002",
                "chain_1_id": "A",
                "chain_2_id": "B",
                "mol_1_type": "ligand",
                "mol_2_type": "prot",
                "type": "interface",
                "num_tokens": 1100,
                "cluster_id": "cluster_b",
            },
            {
                "pdb_id": "p003",
                "chain_1_id": "A",
                "chain_2_id": "B",
                "mol_1_type": "prot",
                "mol_2_type": "prot",
                "type": "interface",
                "num_tokens": 1400,
                "cluster_id": "cluster_c",
            },
            {
                "pdb_id": "p004",
                "chain_1_id": "A",
                "chain_2_id": None,
                "mol_1_type": "prot",
                "mol_2_type": None,
                "type": "chain",
                "num_tokens": 768,
                "cluster_id": "cluster_d",
            },
            {
                "pdb_id": "p005",
                "chain_1_id": "A",
                "chain_2_id": None,
                "mol_1_type": "nuc",
                "mol_2_type": None,
                "type": "chain",
                "num_tokens": 900,
                "cluster_id": "cluster_e",
            },
            {
                "pdb_id": "p006",
                "chain_1_id": "A",
                "chain_2_id": "B",
                "mol_1_type": "ligand",
                "mol_2_type": "ligand",
                "type": "interface",
                "num_tokens": 1200,
                "cluster_id": "cluster_f",
            },
            {
                "pdb_id": "p007",
                "chain_1_id": "A",
                "chain_2_id": "B",
                "mol_1_type": "nuc",
                "mol_2_type": "prot",
                "type": "interface",
                "num_tokens": 1300,
                "cluster_id": "cluster_g",
            },
            {
                "pdb_id": "p008",
                "chain_1_id": "A",
                "chain_2_id": None,
                "mol_1_type": "prot",
                "mol_2_type": None,
                "type": "chain",
                "num_tokens": 2000,
                "cluster_id": "cluster_h",
            },
            {
                "pdb_id": "p009",
                "chain_1_id": "A",
                "chain_2_id": "B",
                "mol_1_type": "prot",
                "mol_2_type": "ions",
                "type": "interface",
                "num_tokens": 1500,
                "cluster_id": "cluster_i",
            },
        ]
    )


def test_filter_long_protein_context_keeps_protein_contexts_in_range():
    filtered = filter_long_protein_context(
        _toy_indices(),
        min_num_tokens=769,
        max_num_tokens=1536,
    )

    assert filtered["pdb_id"].tolist() == ["p001", "p002", "p003", "p007"]
    assert (filtered["num_tokens"].astype(int) >= 769).all()
    assert (filtered["num_tokens"].astype(int) <= 1536).all()


def test_crop1024_filter_requires_more_than_1024_source_tokens():
    filtered = filter_long_protein_context(
        _toy_indices(),
        min_num_tokens=1025,
        max_num_tokens=1536,
    )

    assert filtered["pdb_id"].tolist() == ["p002", "p003", "p007"]
    assert (filtered["num_tokens"].astype(int) >= 1025).all()


def test_select_benchmark_rows_is_deterministic_and_cluster_unique():
    candidates = filter_long_protein_context(_toy_indices(), 769, 1536)
    duplicated = pd.concat([candidates, candidates.iloc[[0]].assign(pdb_id="p010")])

    first = select_benchmark_rows(
        duplicated,
        max_rows=3,
        selection="top_tokens",
        seed=17,
    )
    second = select_benchmark_rows(
        duplicated,
        max_rows=3,
        selection="top_tokens",
        seed=99,
    )

    assert first["pdb_id"].tolist() == ["p003", "p007", "p002"]
    assert second["pdb_id"].tolist() == first["pdb_id"].tolist()
    assert first["cluster_id"].is_unique


def test_select_800_rows_is_pdb_and_cluster_unique():
    candidates = pd.DataFrame(
        {
            "pdb_id": [f"p{index:04d}" for index in range(900)],
            "cluster_id": [f"cluster_{index:04d}" for index in range(900)],
            "num_tokens": [1536 - (index % 100) for index in range(900)],
            "chain_1_id": ["A"] * 900,
            "chain_2_id": ["B"] * 900,
        }
    )
    selected = select_benchmark_rows(
        candidates,
        max_rows=800,
        selection="top_tokens",
        seed=42,
    )

    assert len(selected) == 800
    assert selected["pdb_id"].is_unique
    assert selected["cluster_id"].is_unique


def test_summarize_selection_reports_token_range_and_counts():
    selected = select_benchmark_rows(
        filter_long_protein_context(_toy_indices(), 769, 1536),
        max_rows=10,
        selection="top_tokens",
        seed=0,
    )

    summary = summarize_selection(selected)

    assert summary["rows"] == 4
    assert summary["pdbs"] == 4
    assert summary["clusters"] == 4
    assert summary["min_num_tokens"] == 900
    assert summary["max_num_tokens"] == 1400


def test_build_and_validate_records_hashes_and_rejects_tampering(tmp_path):
    source_path = tmp_path / "source.csv"
    output_dir = tmp_path / "benchmark"
    ready_marker = tmp_path / "run/dataset_ready.json"
    _toy_indices().to_csv(source_path, index=False)

    selected, summary = build_benchmark_index(
        source_index=source_path,
        output_dir=output_dir,
        max_rows=4,
        min_num_tokens=769,
        max_num_tokens=1536,
        selection="top_tokens",
        seed=42,
    )
    write_ready_marker(ready_marker, summary)

    validated, validated_summary = validate_benchmark_index(
        source_index=source_path,
        output_dir=output_dir,
        max_rows=4,
        min_num_tokens=769,
        max_num_tokens=1536,
        selection="top_tokens",
        seed=42,
        ready_marker=ready_marker,
        require_rows=4,
    )
    assert validated["pdb_id"].tolist() == selected["pdb_id"].tolist()
    assert len(validated_summary["index_sha256"]) == 64
    assert len(validated_summary["source_index_sha256"]) == 64

    with (output_dir / "indices.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="SHA256"):
        validate_benchmark_index(
            source_index=source_path,
            output_dir=output_dir,
            max_rows=4,
            min_num_tokens=769,
            max_num_tokens=1536,
            selection="top_tokens",
            seed=42,
            ready_marker=ready_marker,
            require_rows=4,
        )
