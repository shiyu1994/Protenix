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

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import struct
from pathlib import Path

import pandas as pd

from protenix.data.pipeline.dataset import BaseSingleDataset
from protenix.data.pipeline.distillation_runtime_index import (
    DistillationRuntimeIndex,
)
from scripts.distillation.build_ge200_residual_runtime_indices import (
    ExpectedCounts,
    build_ge200_residual_runtime_indices,
)
from scripts.distillation.build_runtime_distillation_index import (
    build_runtime_distillation_index,
)
from scripts.runtime.validate_catalog_replacement_runtime_indices import (
    ExpectedCatalogCounts,
    validate_catalog_pair,
)


def _write_legacy_runtime(
    root: Path,
    name: str,
    rows: list[tuple[str, str]],
    clusters: list[tuple[int, int, int]],
) -> Path:
    source = root / f"{name}_source"
    source.mkdir()
    roots = []
    for root_id, prefix in (("0", "AFQ|"), ("1", "AFQ|"), ("2", "MGN|")):
        data_root = root / f"data_root_{root_id}"
        data_root.mkdir(exist_ok=True)
        roots.append(
            {
                "root_id": root_id,
                "path": str(data_root),
                "source_prefix": prefix,
                "source_db": "afdb" if prefix == "AFQ|" else "mgnify",
            }
        )
    (source / "roots.json").write_text(
        json.dumps({"schema_version": 1, "roots": roots}), encoding="utf-8"
    )
    index = source / "index.csv.gz"
    with gzip.open(index, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_id", "source_root_id", "cif_relpath"])
        for row_id, (root_id, relpath) in enumerate(rows):
            writer.writerow([row_id, root_id, relpath])
    sampling = source / "sampling"
    sampling.mkdir()
    for filename, fmt, values in (
        ("cluster_ids.u32", "<I", [item[0] for item in clusters]),
        ("first_row_ids.u64", "<Q", [item[1] for item in clusters]),
        ("cluster_sizes.u32", "<I", [item[2] for item in clusters]),
    ):
        (sampling / filename).write_bytes(
            b"".join(struct.pack(fmt, value) for value in values)
        )
    output = root / f"{name}_runtime"
    build_runtime_distillation_index(
        input_index=index,
        roots_json=source / "roots.json",
        sampling_dir=sampling,
        output_dir=output,
        progress_every_rows=0,
    )
    return output


def _write_ge200_inputs(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    mapping = root / "mapping.tsv.gz"
    mapping_rows = [
        [0, "AFQ", "same", 200, "0/ge/same.cif"],
        [1, "AFU", "other", 220, "3/aa/bb/other.cif.gz"],
        [2, "AFU", "same", 230, "3/aa/bb/same.cif.gz"],
        [3, "MGN", "move_long", 400, "2/mg/move_long.cif"],
        [4, "ESM", "esm_id", 500, "5/es/esm_id.pdb"],
    ]
    with gzip.open(mapping, "wt", encoding="utf-8", newline="") as stream:
        csv.writer(stream, delimiter="\t", lineterminator="\n").writerows(
            mapping_rows
        )

    numeric = root / "numeric.tsv.gz"
    with gzip.open(numeric, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "representative_row_id",
                "member_row_id",
                "representative_global_id",
                "member_global_id",
            ]
        )
        writer.writerows(
            [
                [0, 0, "AFQ|same", "AFQ|same"],
                [1, 1, "AFU|other", "AFU|other"],
                [1, 2, "AFU|other", "AFU|same"],
                [3, 3, "MGN|move_long", "MGN|move_long"],
                [4, 4, "ESM|esm_id", "ESM|esm_id"],
            ]
        )

    ambiguity = root / "ambiguity.tsv.gz"
    fields = [
        "raw_id",
        "occurrence_index",
        "cluster_run_index",
        "cluster_representative_raw_id",
        "assigned_member_row_id",
        "assigned_member_global_id",
        "candidate_row_ids",
        "candidate_global_ids",
        "resolution_rule",
    ]
    with gzip.open(ambiguity, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for occurrence, cluster, row_id, global_id in (
            (0, 0, 0, "AFQ|same"),
            (1, 1, 2, "AFU|same"),
        ):
            writer.writerow(
                {
                    "raw_id": "same",
                    "occurrence_index": occurrence,
                    "cluster_run_index": cluster,
                    "cluster_representative_raw_id": "same" if cluster == 0 else "other",
                    "assigned_member_row_id": row_id,
                    "assigned_member_global_id": global_id,
                    "candidate_row_ids": "0,2",
                    "candidate_global_ids": "AFQ|same,AFU|same",
                    "resolution_rule": "fixture_canonical",
                }
            )

    esm = root / "esm_msa_only.csv.gz"
    with gzip.open(esm, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "pdb_id",
                "source_root_id",
                "cif_relpath",
                "msa_relpath",
                "bioassembly_relpath",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pdb_id": "esm_id",
                "source_root_id": "5",
                # Deliberately differs from the Foldseek .pdb parent so the
                # test proves that the authoritative msa-only paths are used.
                "cif_relpath": "materialized/esm_id.cif",
                "msa_relpath": "materialized/esm_id.a3m",
                "bioassembly_relpath": "materialized/bioassembly/esm_id.pkl.gz",
            }
        )

    afu = root / "afu_bioassembly.csv.gz"
    with gzip.open(afu, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "pdb_id",
                "source_root_id",
                "cif_relpath",
                "msa_relpath",
                "bioassembly_relpath",
            ],
        )
        writer.writeheader()
        for raw_id in ("other", "same"):
            writer.writerow(
                {
                    "pdb_id": raw_id,
                    "source_root_id": "3",
                    "cif_relpath": f"aa/bb/{raw_id}.cif.gz",
                    "msa_relpath": "",
                    "bioassembly_relpath": f"aa/bb/bioassembly/{raw_id}.pkl.gz",
                }
            )
    return numeric, mapping, ambiguity, esm, afu


def _read_lookup(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_builds_disjoint_residual_and_ge200_with_explicit_paths(tmp_path: Path):
    short = _write_legacy_runtime(
        tmp_path,
        "short",
        [
            ("0", "s/lt199_keep.cif"),
            ("0", "s/same.cif"),
            # An exact-200 legacy member absent from ge200 must survive.
            ("2", "s/eq200_missing_keep.cif"),
        ],
        [(10, 0, 2), (11, 2, 1)],
    )
    long = _write_legacy_runtime(
        tmp_path,
        "long",
        [("2", "l/move_long.cif"), ("0", "l/long_missing_keep.cif")],
        [(20, 0, 1), (21, 1, 1)],
    )
    numeric, mapping, ambiguity, esm, afu = _write_ge200_inputs(tmp_path)
    residual_output = tmp_path / "residual"
    ge200_output = tmp_path / "ge200"

    receipt = build_ge200_residual_runtime_indices(
        legacy_short_runtime_dir=short,
        legacy_long_runtime_dir=long,
        ge200_numeric_global_tsv=numeric,
        ge200_mapping_tsv=mapping,
        ge200_ambiguity_tsv=ambiguity,
        esm_msa_only_index=esm,
        afu_bioassembly_index=afu,
        residual_output_dir=residual_output,
        ge200_output_dir=ge200_output,
        expected_counts=ExpectedCounts(
            residual_rows=3,
            ge200_rows=5,
            target_rows=8,
            ge200_clusters=4,
            ge200_afq_mgn_rows=2,
            residual_short_rows=2,
            residual_long_rows=1,
        ),
        expected_numeric_global_sha256=None,
        progress_every_rows=0,
    )

    assert receipt["target_rows"] == 8
    residual = DistillationRuntimeIndex(residual_output)
    ge200 = DistillationRuntimeIndex(ge200_output)
    assert (residual.row_count, residual.cluster_count) == (3, 3)
    assert (ge200.row_count, ge200.cluster_count) == (5, 4)
    assert residual.has_explicit_paths and residual.has_global_ids
    assert ge200.has_explicit_paths and ge200.has_global_ids

    residual_ids = {residual.get_global_id(row) for row in range(len(residual))}
    ge200_ids = {ge200.get_global_id(row) for row in range(len(ge200))}
    assert residual_ids == {
        "AFQ|lt199_keep",
        "MGN|eq200_missing_keep",
        "AFQ|long_missing_keep",
    }
    assert ge200_ids == {
        "AFQ|same",
        "AFU|other",
        "AFU|same",
        "MGN|move_long",
        "ESM|esm_id",
    }
    assert residual_ids.isdisjoint(ge200_ids)

    afu = ge200.resolve_paths(1)
    assert afu.cif_relpath == "aa/bb/other.cif.gz"
    assert afu.a3m_path.name == "other.a3m"
    assert afu.bioassembly_path.name == "other.pkl.gz"
    assert ".cif.a3m" not in str(afu.a3m_path)
    esm_paths = ge200.resolve_paths(4)
    assert esm_paths.cif_relpath == "materialized/esm_id.cif"
    assert esm_paths.a3m_path.name == "esm_id.a3m"
    assert esm_paths.bioassembly_path.name == "esm_id.pkl.gz"

    # Dataset path resolution must use the row-aligned artifacts too.
    dataset = BaseSingleDataset.__new__(BaseSingleDataset)
    dataset.runtime_distillation_index = ge200
    dataset.source_roots = ge200.source_roots
    pkl_path, a3m_path = dataset._resolve_row_addressed_distillation_paths(
        pd.Series({"row_id": 1, "source_root_id": "3", "cif_relpath": afu.cif_relpath})
    )
    assert pkl_path == afu.bioassembly_path
    assert a3m_path == afu.a3m_path

    residual_lookup = _read_lookup(
        residual_output / "lookup/member_to_cluster.tsv.gz"
    )
    assert [row["runtime_row_id"] for row in residual_lookup] == ["0", "1", "2"]
    assert {row["source_partition"] for row in residual_lookup} == {
        "short_le200",
        "long_gt200",
    }
    ambiguity_rows = _read_lookup(
        ge200_output / "lookup/ambiguity_alternates.tsv.gz"
    )
    assert ambiguity_rows == [
        {
            "global_id": "AFQ|same",
            "canonical_cluster_ordinal": "0",
            "alternate_cluster_ordinal": "1",
            "raw_id": "same",
            "resolution_rule": "seen_floor_1_v1:fixture_canonical",
        }
    ]

    residual_manifest = json.loads(
        (residual_output / "manifest.json").read_text(encoding="utf-8")
    )
    ge200_manifest = json.loads(
        (ge200_output / "manifest.json").read_text(encoding="utf-8")
    )
    assert residual_manifest["target_catalog"]["pair_id"] == receipt["pair_id"]
    assert ge200_manifest["target_catalog"]["pair_id"] == receipt["pair_id"]
    assert residual_manifest["target_catalog"]["role"] == "legacy_residual"
    assert ge200_manifest["target_catalog"]["role"] == "ge200_structural"
    for output, manifest in (
        (residual_output, residual_manifest),
        (ge200_output, ge200_manifest),
    ):
        pair_record = manifest["target_catalog"]["pair_receipt"]
        pair_path = output / pair_record["path"]
        assert hashlib.sha256(pair_path.read_bytes()).hexdigest() == pair_record["sha256"]
        assert (output / "build_audit.json").is_file()
        assert (output / "checksums.sha256").is_file()

    validation = validate_catalog_pair(
        residual_runtime_dir=residual_output,
        ge200_runtime_dir=ge200_output,
        expected=ExpectedCatalogCounts(
            legacy_input_rows=5,
            overlap_rows=2,
            residual_rows=3,
            ge200_rows=5,
            target_rows=8,
            ge200_clusters=4,
            residual_short_rows=2,
            residual_long_rows=1,
            ambiguity_records=1,
        ),
        verify_checksums=True,
        check_source_paths=False,
        require_all_roots_used=False,
    )
    assert validation["status"] == "complete"
    assert validation["pair_id"] == receipt["pair_id"]
    assert validation["residual"]["unused_root_ids"] == ["1"]
    assert validation["ge200"]["unused_root_ids"] == ["1", "4"]

    residual.close()
    ge200.close()
