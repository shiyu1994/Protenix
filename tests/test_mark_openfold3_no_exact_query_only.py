from __future__ import annotations

import csv
import fcntl
import gzip
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import audit_openfold3_alignment_representatives as audit  # noqa: E402
import build_openfold3_disordered_dataset as builder  # noqa: E402
import mark_openfold3_no_exact_query_only as marker  # noqa: E402
import repair_openfold3_alignment_representatives as repair  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_rna_alignment(path: Path, query: str) -> None:
    msa = np.asarray([list(query)], dtype="<U1")
    payload = {
        "msa": msa,
        "deletion_matrix": np.zeros(msa.shape, dtype=np.int32),
        "metadata": np.asarray([{}], dtype=object),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        nucleotide_collection_hits=np.asarray(payload, dtype=object),
    )


def _canonical_gzip_pickle(value: Any) -> bytes:
    return builder._deterministic_gzip(
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL),
        compresslevel=3,
    )


def _make_fixture(tmp_path: Path) -> dict[str, Any]:
    dataset_root = tmp_path / "dataset"
    structures_root = dataset_root / "structures" / "structures"
    alignment_dir = dataset_root / "alignment_arrays"
    protenix_dir = dataset_root / "protenix"
    metadata_dir = dataset_root / "metadata"
    for directory in (
        structures_root,
        alignment_dir,
        protenix_dir,
        metadata_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    cache_entries: dict[str, dict[str, Any]] = {}
    states: dict[str, dict[str, Any]] = {}
    bioassemblies: dict[str, dict[str, Any]] = {}
    bioassembly_paths: dict[str, Path] = {}
    state_paths: dict[str, Path] = {}
    source_paths: list[Path] = []
    alignment_paths: list[Path] = []
    for ordinal, pdb_id in enumerate(("aaaa", "bbbb")):
        structure_dir = structures_root / pdb_id
        bioassembly_dir = structure_dir / "bioassembly"
        bioassembly_dir.mkdir(parents=True)
        source_path = structure_dir / f"{pdb_id}.npz"
        np.savez(source_path, source=np.asarray([ordinal]))
        source_paths.append(source_path)

        rep_a = f"{pdb_id}_A"
        rep_b = f"{pdb_id}_B"
        path_a = alignment_dir / f"{rep_a}.npz"
        path_b = alignment_dir / f"{rep_b}.npz"
        _write_rna_alignment(path_a, "AAAA")
        _write_rna_alignment(path_b, "BBBB")
        alignment_paths.extend((path_a, path_b))
        cache_entry = {
            "chains": {
                "1": {
                    "molecule_type": "RNA",
                    "alignment_representative_id": rep_a,
                },
                "2": {
                    "molecule_type": "RNA",
                    "alignment_representative_id": rep_b,
                },
            },
            "interfaces": {},
        }
        cache_entries[pdb_id] = cache_entry

        bioassembly = {
            "pdb_id": pdb_id,
            "alignment_representative_id_by_chain_id": {
                "1": rep_a,
                "2": rep_b,
            },
            "sequences_by_chain_id": {
                "1": "CCCC",
                "2": "BBBB",
            },
            "sentinel": {
                "array": np.asarray(
                    [[ordinal, ordinal + 1]],
                    dtype=np.int16,
                ),
                "text": f"unchanged-{pdb_id}",
            },
        }
        if pdb_id == "aaaa":
            bioassembly[marker.MARKER_FIELD] = ["2"]
        bioassemblies[pdb_id] = bioassembly
        bioassembly_path = (
            bioassembly_dir / f"{pdb_id}.pkl.gz"
        )
        builder._atomic_bytes(
            _canonical_gzip_pickle(bioassembly),
            bioassembly_path,
        )
        bioassembly_paths[pdb_id] = bioassembly_path

        state = {
            "schema_version": builder.STATE_SCHEMA_VERSION,
            "pdb_id": pdb_id,
            "source_relpath": f"{pdb_id}/{pdb_id}.npz",
            "source_size_bytes": source_path.stat().st_size,
            "source_sha256": _sha256(source_path),
            "cache_entry_sha256": hashlib.sha256(
                builder._canonical_json_bytes(cache_entry)
            ).hexdigest(),
            "output_relpath": (
                f"{pdb_id}/bioassembly/{pdb_id}.pkl.gz"
            ),
            "output_size_bytes": bioassembly_path.stat().st_size,
            "output_sha256": _sha256(bioassembly_path),
            "index_rows": [{"pdb_id": pdb_id}],
        }
        states[pdb_id] = state
        state_path = bioassembly_dir / f"{pdb_id}.build.json"
        _write_json(state_path, state)
        state_paths[pdb_id] = state_path

    cache_path = metadata_dir / "cache.json"
    _write_json(
        cache_path,
        {
            "_type": "ClusteredDatasetCache",
            "name": "disordered-pdb",
            "structure_data": cache_entries,
        },
    )
    index_path = protenix_dir / "indices.csv.gz"
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
        for pdb_id in sorted(cache_entries):
            writer.writerow(
                {
                    "pdb_id": pdb_id,
                    "bioassembly_relpath": (
                        f"{pdb_id}/bioassembly/{pdb_id}.pkl.gz"
                    ),
                }
            )

    total_bytes, aggregate = repair._state_inventory(states)
    manifest = {
        "schema_version": 3,
        "artifact_role": audit.MANIFEST_ROLE,
        "status": "complete",
        "success_eligible": True,
        "cache": {
            "cache_name": "disordered-pdb",
            "cache_type": "ClusteredDatasetCache",
            "entry_count": 2,
            "path": str(cache_path),
            "sha256": _sha256(cache_path),
            "size_bytes": cache_path.stat().st_size,
        },
        "parameters": {
            "structures_root": str(structures_root),
            "alignment_dirs": [str(alignment_dir)],
            "compresslevel": 3,
        },
        "artifacts": {
            "indices": {
                "path": str(index_path),
                "rows": 2,
                "sha256": _sha256(index_path),
                "size_bytes": index_path.stat().st_size,
            }
        },
        "bioassemblies": {
            "count": 2,
            "total_bytes": total_bytes,
            "content_manifest_sha256": aggregate,
        },
        "repair_history": [
            {
                "repair_id": repair.REPAIR_ID,
                "affected_structure_count": 9,
                "affected_chain_count": 20,
            }
        ],
    }
    manifest_path = protenix_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    success = {
        "schema_version": 3,
        "artifact_role": audit.SUCCESS_ROLE,
        "manifest": "manifest.json",
        "manifest_sha256": _sha256(manifest_path),
        "index_row_count": 2,
        "pickle_count": 2,
    }
    success_path = protenix_dir / "_SUCCESS.json"
    _write_json(success_path, success)

    audit_dir = tmp_path / "audit_v3"
    audit_dir.mkdir()
    result_path = audit_dir / "audit.shard0.jsonl"
    result_records: list[dict[str, Any]] = []
    for structure_ordinal, pdb_id in enumerate(sorted(cache_entries)):
        current_id = f"{pdb_id}_A"
        result_records.append(
            {
                "schema_version": 2,
                "dataset": marker.EXPECTED_DATASET,
                "record_type": "chain",
                "shard_index": 0,
                "shard_count": 1,
                "structure_ordinal": structure_ordinal,
                "classification": "no_exact",
                "pdb_id": pdb_id,
                "bioassembly_relpath": (
                    f"{pdb_id}/bioassembly/{pdb_id}.pkl.gz"
                ),
                "chain_id": "1",
                "polymer_type": audit.RNA_CHAIN,
                "canonical_query_length": 4,
                "canonical_query_sha256": hashlib.sha256(
                    b"CCCC"
                ).hexdigest(),
                "current_representative_id": current_id,
                "current_representative": {
                    "status": "valid",
                    "query_length": 4,
                    "query_sha256": hashlib.sha256(
                        b"AAAA"
                    ).hexdigest(),
                },
                "candidate_errors": {},
                "matching_representative_ids": [],
                "proposed_representative_id": None,
            }
        )
        result_records.append(
            {
                "schema_version": 2,
                "dataset": marker.EXPECTED_DATASET,
                "record_type": "chain",
                "shard_index": 0,
                "shard_count": 1,
                "structure_ordinal": structure_ordinal,
                "classification": "exact_current",
                "pdb_id": pdb_id,
                "bioassembly_relpath": (
                    f"{pdb_id}/bioassembly/{pdb_id}.pkl.gz"
                ),
                "chain_id": "2",
            }
        )
    result_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in result_records
        ),
        encoding="utf-8",
    )

    summary_path = audit_dir / "audit.shard0.summary.json"
    contract = audit.validate_disordered_dataset_root(dataset_root)
    classification_counts = {
        name: (
            2 if name in {"no_exact", "exact_current"} else 0
        )
        for name in audit.CLASSIFICATIONS
    }
    summary = {
        "event": "audit_summary",
        "schema_version": 2,
        "dataset": marker.EXPECTED_DATASET,
        "status": "complete",
        "read_only_source_dataset": True,
        "run_contract": {
            "schema_version": 1,
            "dataset": marker.EXPECTED_DATASET,
            "dataset_root": str(dataset_root),
            "declared_pickle_count": 2,
            "index_row_count": 2,
            "manifest_sha256": contract.manifest_sha256,
            "index_sha256": contract.index_sha256,
            "success_sha256": contract.success_sha256,
            "comparison": (
                "exact_canonical_query_string_same_polymer_candidates_v2"
            ),
            "selected_structure_count": 2,
            "shard_index": 0,
            "shard_count": 1,
        },
        "selected_structure_count": 2,
        "completed_structure_count": 2,
        "chain_record_count": 4,
        "structure_record_count": 0,
        "classification_counts": classification_counts,
        "record_type_counts": {"chain": 4},
        "outputs": {
            "result_jsonl": str(result_path),
            "summary_json": str(summary_path),
            "state_jsonl": str(audit_dir / "audit.shard0.state.jsonl"),
        },
    }
    _write_json(summary_path, summary)

    mutable_paths = (
        *bioassembly_paths.values(),
        *state_paths.values(),
        manifest_path,
        success_path,
    )
    immutable_paths = (
        *source_paths,
        *alignment_paths,
        cache_path,
        index_path,
    )
    return {
        "dataset_root": dataset_root,
        "result_path": result_path,
        "summary_path": summary_path,
        "backup_dir": tmp_path / "backup",
        "bioassembly_paths": bioassembly_paths,
        "state_paths": state_paths,
        "manifest_path": manifest_path,
        "success_path": success_path,
        "mutable_paths": mutable_paths,
        "immutable_paths": immutable_paths,
        "original_bioassemblies": bioassemblies,
    }


def _run(
    fixture: dict[str, Any],
    *,
    apply: bool,
    backup_dir: Path | None = None,
) -> dict[str, Any]:
    return marker.mark_dataset(
        dataset_root=fixture["dataset_root"],
        audit_results=[fixture["result_path"]],
        backup_dir=backup_dir or fixture["backup_dir"],
        state_read_workers=1,
        apply=apply,
    )


def _refresh_summary_contract(fixture: dict[str, Any]) -> None:
    contract = audit.validate_disordered_dataset_root(
        fixture["dataset_root"]
    )
    summary = json.loads(
        fixture["summary_path"].read_text(encoding="utf-8")
    )
    summary["run_contract"]["manifest_sha256"] = (
        contract.manifest_sha256
    )
    summary["run_contract"]["success_sha256"] = contract.success_sha256
    _write_json(fixture["summary_path"], summary)


def test_apply_marks_only_audited_chains_and_updates_hashes(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    immutable_hashes = {
        path: _sha256(path) for path in fixture["immutable_paths"]
    }

    result = _run(fixture, apply=True)

    assert result["status"] == "complete"
    assert result["affected_chain_count"] == 2
    expected_markers = {
        "aaaa": ["1", "2"],
        "bbbb": ["1"],
    }
    for pdb_id, path in fixture["bioassembly_paths"].items():
        with gzip.open(path, "rb") as stream:
            marked = pickle.load(stream)
        assert marked[marker.MARKER_FIELD] == expected_markers[pdb_id]
        original = fixture["original_bioassemblies"][pdb_id]
        for key, value in original.items():
            if key == marker.MARKER_FIELD:
                continue
            assert pickle.dumps(marked[key]) == pickle.dumps(value)
        state = json.loads(
            fixture["state_paths"][pdb_id].read_text(encoding="utf-8")
        )
        assert state["output_sha256"] == _sha256(path)
        assert state["output_size_bytes"] == path.stat().st_size

    for path, expected_hash in immutable_hashes.items():
        assert _sha256(path) == expected_hash
    manifest = json.loads(
        fixture["manifest_path"].read_text(encoding="utf-8")
    )
    success = json.loads(
        fixture["success_path"].read_text(encoding="utf-8")
    )
    assert manifest["repair_history"][-1]["repair_id"] == marker.REPAIR_ID
    assert success["manifest_sha256"] == _sha256(
        fixture["manifest_path"]
    )
    assert (fixture["backup_dir"] / "repair_receipt.json").is_file()


def test_post_publish_failure_rolls_back_success_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    before = {path: _sha256(path) for path in fixture["mutable_paths"]}
    original_validate = audit.validate_disordered_dataset_root
    validate_calls = 0

    def fail_post_publish(path: Path) -> audit.DatasetContract:
        nonlocal validate_calls
        validate_calls += 1
        if validate_calls == 3:
            raise marker.QueryOnlyMarkerError(
                "injected post-publish failure"
            )
        return original_validate(path)

    restore_batches: list[list[Path]] = []
    original_restore = repair._restore_from_backup

    def record_restore(**kwargs: Any) -> None:
        restore_batches.append(list(kwargs["paths"]))
        original_restore(**kwargs)

    monkeypatch.setattr(
        audit,
        "validate_disordered_dataset_root",
        fail_post_publish,
    )
    monkeypatch.setattr(repair, "_restore_from_backup", record_restore)

    with pytest.raises(
        marker.QueryOnlyMarkerError,
        match="injected post-publish failure",
    ):
        _run(fixture, apply=True)

    assert validate_calls == 4
    assert restore_batches[-1] == [fixture["success_path"]]
    assert fixture["success_path"] not in restore_batches[0]
    assert {path: _sha256(path) for path in fixture["mutable_paths"]} == before
    assert not (fixture["backup_dir"] / "repair_receipt.json").exists()


def test_second_apply_fails_closed_on_manifest_provenance(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    _run(fixture, apply=True)
    _refresh_summary_contract(fixture)

    with pytest.raises(
        marker.QueryOnlyMarkerError,
        match="already declares marker repair",
    ):
        _run(
            fixture,
            apply=False,
            backup_dir=tmp_path / "unused-backup",
        )


def test_rejects_when_an_exact_candidate_appears(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    exact_path = (
        fixture["dataset_root"]
        / "alignment_arrays"
        / "aaaa_A.npz"
    )
    _write_rna_alignment(exact_path, "CCCC")

    with pytest.raises(
        marker.QueryOnlyMarkerError,
        match="no_exact proof no longer holds",
    ):
        _run(fixture, apply=False)


def test_requires_exactly_two_no_exact_records(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    records = [
        json.loads(line)
        for line in fixture["result_path"].read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    records = [
        record
        for record in records
        if not (
            record["classification"] == "no_exact"
            and record["pdb_id"] == "bbbb"
        )
    ]
    fixture["result_path"].write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    summary = json.loads(
        fixture["summary_path"].read_text(encoding="utf-8")
    )
    summary["chain_record_count"] = 3
    summary["classification_counts"]["no_exact"] = 1
    summary["record_type_counts"]["chain"] = 3
    _write_json(fixture["summary_path"], summary)

    with pytest.raises(
        marker.QueryOnlyMarkerError,
        match="expected exactly 2 no_exact chains",
    ):
        _run(fixture, apply=False)


def test_apply_holds_shared_offline_repair_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    observed = False

    def assert_locked(**kwargs: Any) -> dict[str, Any]:
        nonlocal observed
        lock_path = (
            kwargs["contract"].protenix_root
            / ".representative_repair.lock"
        )
        with lock_path.open("a+b") as competing:
            with pytest.raises(BlockingIOError):
                fcntl.flock(
                    competing.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
        observed = True
        return {"status": "test"}

    monkeypatch.setattr(marker, "_mark_with_contract", assert_locked)
    result = _run(fixture, apply=True)
    assert result == {"status": "test"}
    assert observed
    assert not fixture["backup_dir"].exists()


def test_main_atomically_writes_external_dry_run_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _make_fixture(tmp_path)
    report_path = tmp_path / "reports" / "dry_run.json"

    assert (
        marker.main(
            [
                "--dataset-root",
                str(fixture["dataset_root"]),
                "--audit-result",
                str(fixture["result_path"]),
                "--backup-dir",
                str(fixture["backup_dir"]),
                "--state-read-workers",
                "1",
                "--report-json",
                str(report_path),
            ]
        )
        == 0
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    stdout = json.loads(capsys.readouterr().out)
    assert report["status"] == "dry_run_complete"
    assert report == stdout
    assert not fixture["backup_dir"].exists()
