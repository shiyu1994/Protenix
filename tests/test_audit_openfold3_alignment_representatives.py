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

"""Tests for the read-only OpenFold3-disordered representative audit."""

from __future__ import annotations

import csv
import gzip
import json
import pickle
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from scripts import audit_openfold3_alignment_representatives as audit


class _FakeAtomArray:
    """Minimum picklable annotation surface consumed by the audit."""

    def __init__(
        self,
        *,
        chain_id: list[str],
        label_entity_id: list[str],
        label_asym_id: list[str],
    ) -> None:
        self.chain_id = np.asarray(chain_id)
        self.label_entity_id = np.asarray(label_entity_id)
        self.label_asym_id = np.asarray(label_asym_id)


def _payload(query: str) -> np.ndarray:
    value = np.empty((), dtype=object)
    value[()] = {
        "msa": np.asarray(
            [list(query), list(query)],
            dtype="<U1",
        ),
        "deletion_matrix": np.zeros((2, len(query)), dtype=np.int32),
        "metadata": np.asarray(["query", "hit"], dtype=object),
    }
    return value


def _write_alignment(
    alignment_dir: Path,
    representative_id: str,
    query: str,
) -> None:
    np.savez(
        alignment_dir / f"{representative_id}.npz",
        uniref90_hits=_payload(query),
    )


def _write_rna_alignment(
    alignment_dir: Path,
    representative_id: str,
    query: str,
) -> None:
    np.savez(
        alignment_dir / f"{representative_id}.npz",
        rfam_hits=_payload(query),
    )


def _write_bioassembly(
    structures_root: Path,
    pdb_id: str,
    chains: list[tuple[str, str, str]],
    *,
    polymer_types: list[str] | None = None,
    query_only_chain_ids: list[str] | None = None,
) -> str:
    """Write chains as ``(chain_id, canonical_query, representative_id)``."""
    if polymer_types is not None and len(polymer_types) != len(chains):
        raise ValueError("polymer_types must match chains")
    chain_annotations: list[str] = []
    entity_annotations: list[str] = []
    asym_annotations: list[str] = []
    sequences: dict[str, str] = {}
    sequences_by_chain: dict[str, str] = {}
    entity_poly_type: dict[str, str] = {}
    representatives: dict[str, str] = {}
    for ordinal, (chain_id, sequence, representative_id) in enumerate(
        chains,
        start=1,
    ):
        entity_id = str(ordinal)
        chain_annotations.extend([chain_id] * len(sequence))
        entity_annotations.extend([entity_id] * len(sequence))
        asym_annotations.extend([chain_id] * len(sequence))
        sequences[entity_id] = sequence
        sequences_by_chain[chain_id] = sequence
        entity_poly_type[entity_id] = (
            audit.PROTEIN_CHAIN
            if polymer_types is None
            else polymer_types[ordinal - 1]
        )
        representatives[chain_id] = representative_id
    bioassembly = {
        "pdb_id": pdb_id,
        "atom_array": _FakeAtomArray(
            chain_id=chain_annotations,
            label_entity_id=entity_annotations,
            label_asym_id=asym_annotations,
        ),
        "sequences": sequences,
        "sequences_by_chain_id": sequences_by_chain,
        "entity_poly_type": entity_poly_type,
        "alignment_representative_id_by_chain_id": representatives,
    }
    if query_only_chain_ids is not None:
        bioassembly["openfold3_query_only_chain_ids"] = query_only_chain_ids
    relpath = f"{pdb_id}/bioassembly/{pdb_id}.pkl.gz"
    path = structures_root / relpath
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wb") as stream:
        pickle.dump(bioassembly, stream)
    return relpath


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_contract(
    dataset_root: Path,
    rows: list[tuple[str, str]],
) -> None:
    protenix_root = dataset_root / "protenix"
    protenix_root.mkdir()
    index_path = protenix_root / "indices.csv.gz"
    with gzip.open(
        index_path,
        "wt",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("pdb_id", "bioassembly_relpath"),
        )
        writer.writeheader()
        for pdb_id, relpath in rows:
            writer.writerow(
                {
                    "pdb_id": pdb_id,
                    "bioassembly_relpath": relpath,
                }
            )

    structures_root = dataset_root / "structures" / "structures"
    alignment_root = dataset_root / "alignment_arrays"
    manifest = {
        "schema_version": 3,
        "artifact_role": audit.MANIFEST_ROLE,
        "status": "complete",
        "success_eligible": True,
        "cache": {
            "cache_name": "disordered-pdb",
            "cache_type": "ClusteredDatasetCache",
        },
        "parameters": {
            "structures_root": str(structures_root),
            "alignment_dirs": [str(alignment_root)],
        },
        "artifacts": {
            "indices": {
                "path": str(index_path),
                "rows": len(rows),
                "sha256": _sha256(index_path),
            }
        },
        "bioassemblies": {"count": len(rows)},
    }
    manifest_path = protenix_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    success = {
        "schema_version": 3,
        "artifact_role": audit.SUCCESS_ROLE,
        "manifest": "manifest.json",
        "manifest_sha256": _sha256(manifest_path),
        "index_row_count": len(rows),
        "pickle_count": len(rows),
    }
    (protenix_root / "_SUCCESS.json").write_text(
        json.dumps(success, sort_keys=True),
        encoding="utf-8",
    )


def _build_fixture(base: Path) -> tuple[Path, list[tuple[str, str]]]:
    dataset_root = base / "dataset"
    structures_root = dataset_root / "structures" / "structures"
    alignment_dir = dataset_root / "alignment_arrays"
    structures_root.mkdir(parents=True)
    alignment_dir.mkdir(parents=True)
    rows: list[tuple[str, str]] = []

    def add(
        pdb_id: str,
        chains: list[tuple[str, str, str]],
        alignments: dict[str, str | None],
    ) -> None:
        rows.append(
            (
                pdb_id,
                _write_bioassembly(
                    structures_root,
                    pdb_id,
                    chains,
                ),
            )
        )
        for representative_id, query in alignments.items():
            if query is None:
                continue
            _write_alignment(
                alignment_dir,
                representative_id,
                query,
            )

    add(
        "1exact",
        [("A", "AAA", "1exact_A")],
        {"1exact_A": "AAA"},
    )
    add(
        "2remap",
        [
            ("A", "CCC", "2remap_A"),
            ("B", "CCC", "2remap_B"),
        ],
        {"2remap_A": "AAA", "2remap_B": "CCC"},
    )
    add(
        "3missing",
        [("A", "AAA", "3missing_A")],
        {"3missing_A": None},
    )
    add(
        "4ambigu",
        [
            ("A", "CCC", "4ambigu_A"),
            ("B", "CCC", "4ambigu_B"),
            ("C", "CCC", "4ambigu_C"),
        ],
        {
            "4ambigu_A": "AAA",
            "4ambigu_B": "CCC",
            "4ambigu_C": "CCC",
        },
    )
    add(
        "5noexact",
        [("A", "GGG", "5noexact_A")],
        {"5noexact_A": "AAA"},
    )
    add(
        "6corrupt",
        [("A", "AAA", "6corrupt_A")],
        {},
    )
    (alignment_dir / "6corrupt_A.npz").write_bytes(b"not-an-npz")

    mixed_relpath = _write_bioassembly(
        structures_root,
        "7mixed",
        [
            ("A", "AAA", "7mixed_A"),
            ("B", "CCC", "7mixed_B"),
        ],
        polymer_types=[audit.PROTEIN_CHAIN, audit.RNA_CHAIN],
    )
    _write_alignment(alignment_dir, "7mixed_A", "AAA")
    _write_rna_alignment(alignment_dir, "7mixed_B", "GGG")
    rows.append(("7mixed", mixed_relpath))

    corrupt_relpath = "7badpkl/bioassembly/7badpkl.pkl.gz"
    corrupt_path = structures_root / corrupt_relpath
    corrupt_path.parent.mkdir(parents=True)
    corrupt_path.write_bytes(b"not-a-gzip-pickle")
    rows.append(("7badpkl", corrupt_relpath))

    _publish_contract(dataset_root, rows)
    return dataset_root, rows


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _run(
    dataset_root: Path,
    output_root: Path,
    *,
    name: str = "audit",
    **overrides,
) -> dict:
    arguments = {
        "dataset_root": dataset_root,
        "output_jsonl": output_root / f"{name}.jsonl",
        "summary_json": output_root / f"{name}.summary.json",
        "workers": 1,
        "checkpoint_every": 1,
    }
    arguments.update(overrides)
    return audit.run_audit(**arguments)


def test_classifies_all_outcomes_and_marks_disorder_dataset(tmp_path):
    dataset_root, _ = _build_fixture(tmp_path)
    output_root = tmp_path / "outputs"
    summary = _run(dataset_root, output_root)
    records = _read_jsonl(output_root / "audit.jsonl")
    by_chain = {
        (record["pdb_id"], record["chain_id"]): record
        for record in records
        if record["record_type"] == "chain"
    }

    assert by_chain[("1exact", "A")]["classification"] == "exact_current"
    remap = by_chain[("2remap", "A")]
    assert remap["classification"] == "remappable_unique"
    assert remap["proposed_representative_id"] == "2remap_B"
    assert by_chain[("3missing", "A")]["classification"] == "missing"
    ambiguous = by_chain[("4ambigu", "A")]
    assert ambiguous["classification"] == "ambiguous"
    assert ambiguous["matching_representative_ids"] == [
        "4ambigu_B",
        "4ambigu_C",
    ]
    assert by_chain[("5noexact", "A")]["classification"] == "no_exact"
    assert by_chain[("6corrupt", "A")]["classification"] == "corrupt"
    mixed_rna = by_chain[("7mixed", "B")]
    assert mixed_rna["classification"] == "no_exact"
    assert mixed_rna["declared_representative_count"] == 1
    assert mixed_rna["candidate_errors"] == {}
    corrupt_structure = next(
        record for record in records if record["pdb_id"] == "7badpkl"
    )
    assert corrupt_structure["record_type"] == "structure"
    assert corrupt_structure["classification"] == "corrupt"

    assert summary["dataset"] == audit.DATASET_NAME
    assert summary["schema_version"] == 2
    assert summary["run_contract"]["dataset"] == audit.DATASET_NAME
    assert (
        summary["run_contract"]["comparison"]
        == (
            "exact_canonical_query_string_same_polymer_candidates_"
            "explicit_query_only_v3"
        )
    )
    assert summary["read_only_source_dataset"]
    assert set(summary["classification_counts"]) == set(
        audit.CLASSIFICATIONS
    )
    assert (output_root / "audit.jsonl.state.jsonl").is_file()
    assert not list(output_root.glob("*.tmp"))


def test_explicit_query_only_skips_all_representative_npz_reads(tmp_path):
    dataset_root = tmp_path / "dataset"
    structures_root = dataset_root / "structures" / "structures"
    alignment_dir = dataset_root / "alignment_arrays"
    structures_root.mkdir(parents=True)
    alignment_dir.mkdir(parents=True)
    relpath = _write_bioassembly(
        structures_root,
        "8queryonly",
        [("A", "AAA", "8queryonly_A")],
        query_only_chain_ids=["A"],
    )
    # If the explicit marker were ignored, this archive would classify as
    # corrupt.  The audit must not open it at all.
    (alignment_dir / "8queryonly_A.npz").write_bytes(b"not-an-npz")
    _publish_contract(dataset_root, [("8queryonly", relpath)])

    with mock.patch.object(
        audit,
        "_load_representative_query",
        side_effect=AssertionError("representative NPZ must not be loaded"),
    ) as load_representative:
        summary = _run(dataset_root, tmp_path / "outputs")

    load_representative.assert_not_called()
    records = _read_jsonl(tmp_path / "outputs" / "audit.jsonl")
    assert len(records) == 1
    record = records[0]
    assert record["record_type"] == "chain"
    assert record["classification"] == "explicit_query_only"
    assert record["explicit_query_only"] is True
    assert record["current_representative_id"] == "8queryonly_A"
    assert record["current_representative"] is None
    assert summary["classification_counts"]["explicit_query_only"] == 1
    assert summary["chain_record_count"] == 1
    assert not summary["all_exact_current"]


def test_invalid_explicit_query_only_chain_is_structure_corrupt(tmp_path):
    dataset_root = tmp_path / "dataset"
    structures_root = dataset_root / "structures" / "structures"
    alignment_dir = dataset_root / "alignment_arrays"
    structures_root.mkdir(parents=True)
    alignment_dir.mkdir(parents=True)
    relpath = _write_bioassembly(
        structures_root,
        "9badmarker",
        [("A", "AAA", "9badmarker_A")],
        query_only_chain_ids=["not-a-chain"],
    )
    _publish_contract(dataset_root, [("9badmarker", relpath)])

    summary = _run(dataset_root, tmp_path / "outputs")
    records = _read_jsonl(tmp_path / "outputs" / "audit.jsonl")

    assert len(records) == 1
    assert records[0]["record_type"] == "structure"
    assert records[0]["classification"] == "corrupt"
    assert "openfold3_query_only_chain_ids" in records[0]["error_message"]
    assert summary["classification_counts"]["corrupt"] == 1


def test_ordinal_shards_are_disjoint_and_cover_the_disorder_index(tmp_path):
    dataset_root, rows = _build_fixture(tmp_path)
    output_root = tmp_path / "outputs"
    _run(
        dataset_root,
        output_root,
        name="shard0",
        workers=2,
        shard_count=2,
        shard_index=0,
    )
    _run(
        dataset_root,
        output_root,
        name="shard1",
        workers=2,
        shard_count=2,
        shard_index=1,
    )
    shard_paths = []
    for name in ("shard0", "shard1"):
        shard_paths.append(
            {
                record["bioassembly_relpath"]
                for record in _read_jsonl(output_root / f"{name}.jsonl")
            }
        )
    assert shard_paths[0].isdisjoint(shard_paths[1])
    assert shard_paths[0] | shard_paths[1] == {
        relpath for _, relpath in rows
    }


def test_resume_reuses_complete_structure_state_and_publishes_atomically(
    tmp_path,
):
    dataset_root, rows = _build_fixture(tmp_path)
    output_root = tmp_path / "outputs"
    _run(dataset_root, output_root)
    result_path = output_root / "audit.jsonl"
    summary_path = output_root / "audit.summary.json"
    state_path = output_root / "audit.jsonl.state.jsonl"
    result_path.unlink()
    summary_path.unlink()

    state_lines = state_path.read_text(encoding="utf-8").splitlines()
    assert len(state_lines) == len(rows) + 1
    state_path.write_text(
        "\n".join(state_lines[:-1]) + "\n" + '{"event":"structure_res',
        encoding="utf-8",
    )
    with mock.patch.object(
        audit,
        "_audit_structure_task",
        wraps=audit._audit_structure_task,
    ) as audit_one:
        summary = _run(
            dataset_root,
            output_root,
            resume=True,
        )

    assert audit_one.call_count == 1
    assert summary["completed_structure_count"] == len(rows)
    assert result_path.is_file()
    assert summary_path.is_file()
    assert not list(output_root.glob("*.tmp"))


def test_fail_closed_rejects_non_disorder_role_and_dataset_writes(tmp_path):
    dataset_root, _ = _build_fixture(tmp_path)
    manifest_path = dataset_root / "protenix" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_role"] = "some_other_dataset"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        audit.AuditConfigurationError,
        match="not OpenFold3 disordered",
    ):
        audit.validate_disordered_dataset_root(dataset_root)

    dataset_root, _ = _build_fixture(tmp_path / "second")
    with pytest.raises(
        audit.AuditConfigurationError,
        match="outside the read-only dataset root",
    ):
        audit.run_audit(
            dataset_root=dataset_root,
            output_jsonl=dataset_root / "audit.jsonl",
            summary_json=tmp_path / "summary.json",
            workers=1,
        )
