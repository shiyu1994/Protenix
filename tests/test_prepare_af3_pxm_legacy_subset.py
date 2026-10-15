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

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluation"
    / "prepare_af3_pxm_legacy_subset.py"
)
SPEC = importlib.util.spec_from_file_location(
    "prepare_af3_pxm_legacy_subset",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_runtime_evaluable_target_filter_matches_af3_proxy_contract() -> None:
    frame = pd.DataFrame(
        [
            {
                "pdb_id": "valid",
                "num_tokens": 800,
                "eval_type": "intra_prot",
            },
            {
                "pdb_id": "valid",
                "num_tokens": 800,
                "eval_type": "nuc_prot",
            },
            {
                "pdb_id": "hybrid_only",
                "num_tokens": 900,
                "eval_type": "intra_nuc",
            },
            {
                "pdb_id": "hybrid_only",
                "num_tokens": 900,
                "eval_type": "nuc_prot",
            },
            {
                "pdb_id": "antibody",
                "num_tokens": 700,
                "eval_type": "antibody_antigen",
            },
            {
                "pdb_id": "too_long",
                "num_tokens": 1200,
                "eval_type": "rna_prot",
            },
        ]
    )

    selected, above_limit, unsupported = (
        MODULE._select_runtime_evaluable_targets(frame, max_tokens=1024)
    )

    assert set(selected["pdb_id"]) == {"valid", "antibody"}
    # Keep every index row for a retained target. Runtime later selects the
    # scoreable row, while the full target row set remains available.
    assert set(selected["eval_type"]) == {
        "intra_prot",
        "nuc_prot",
        "antibody_antigen",
    }
    assert above_limit == {"too_long"}
    assert unsupported == {"hybrid_only"}


def test_runtime_evaluable_target_filter_can_disable_token_limit() -> None:
    frame = pd.DataFrame(
        [
            {
                "pdb_id": "short",
                "num_tokens": 800,
                "eval_type": "intra_prot",
            },
            {
                "pdb_id": "long",
                "num_tokens": 1800,
                "eval_type": "rna_prot",
            },
            {
                "pdb_id": "unsupported",
                "num_tokens": 2400,
                "eval_type": "nuc_prot",
            },
        ]
    )

    selected, above_limit, unsupported = (
        MODULE._select_runtime_evaluable_targets(frame, max_tokens=None)
    )

    assert set(selected["pdb_id"]) == {"short", "long"}
    assert above_limit == set()
    assert unsupported == {"unsupported"}


def test_token_limit_cli_is_mutually_exclusive() -> None:
    args = MODULE._parse_args(["--no-max-tokens"])
    assert args.max_tokens is None
    assert MODULE._parse_args([]).max_tokens == 1024
    with pytest.raises(SystemExit):
        MODULE._parse_args(["--max-tokens", "1536", "--no-max-tokens"])


def test_no_limit_emits_full_and_added_manifests(tmp_path: Path) -> None:
    pxm_root = tmp_path / "PXM-Legacy"
    list_dir = pxm_root / "supported_data" / "stat_data" / "pdb_ids"
    list_dir.mkdir(parents=True)
    (list_dir / "RecentPDB.txt").write_text(
        "old1\nnew2\n", encoding="utf-8"
    )
    (list_dir / "RNA_Protein.txt").write_text("new2\n", encoding="utf-8")
    (list_dir / "dsDNA_Protein.txt").write_text("", encoding="utf-8")

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "pdb_id": "old1",
                "num_tokens": 800,
                "eval_type": "intra_prot",
            },
            {
                "pdb_id": "new2",
                "num_tokens": 1800,
                "eval_type": "rna_prot",
            },
        ]
    ).to_csv(index_path, index=False)
    mmcif_dir = tmp_path / "mmcif"
    bioassembly_dir = tmp_path / "bioassembly"
    mmcif_dir.mkdir()
    bioassembly_dir.mkdir()
    for pdb_id in ("old1", "new2"):
        (mmcif_dir / f"{pdb_id}.cif").touch()
        (bioassembly_dir / f"{pdb_id}.pkl.gz").touch()

    already_evaluated = tmp_path / "already_evaluated.txt"
    already_evaluated.write_text("OLD1\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    MODULE.main(
        [
            "--pxm-root",
            str(pxm_root),
            "--protenix-index",
            str(index_path),
            "--weighted-pdb-index",
            str(tmp_path / "missing_weighted.csv"),
            "--mmcif-dir",
            str(mmcif_dir),
            "--bioassembly-dir",
            str(bioassembly_dir),
            "--no-max-tokens",
            "--already-evaluated-list",
            str(already_evaluated),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert (output_dir / "combined_alltokens.txt").read_text(
        encoding="utf-8"
    ) == "old1\nnew2\n"
    assert (output_dir / "combined_alltokens_added.txt").read_text(
        encoding="utf-8"
    ) == "new2\n"
    summary = json.loads(
        (output_dir / "summary_alltokens.json").read_text(encoding="utf-8")
    )
    assert summary["max_tokens"] is None
    assert summary["token_limit_enabled"] is False
    assert summary["token_limit_scope"] == "rows_loaded_from_protenix_index"
    assert summary["already_evaluated_pdb_count"] == 1
    combined = summary["subsets"]["combined"]
    assert combined["full_manifest"].endswith("combined_alltokens.txt")
    assert combined["full_pdb_count"] == 2
    assert combined["added_manifest"].endswith(
        "combined_alltokens_added.txt"
    )
    assert combined["added_pdb_count"] == 1


def test_af3_proxy_types_match_metric_implementation() -> None:
    from protenix.data.constants import EvaluationChainInterface
    from protenix.metrics.lddt_metrics import AF3_FIG2D_EVAL_TYPES

    assert frozenset(MODULE.CANONICAL_AF3_TYPES) == AF3_FIG2D_EVAL_TYPES
    assert frozenset(MODULE.RUNTIME_EVALUATION_TYPES) == frozenset(
        EvaluationChainInterface
    )
