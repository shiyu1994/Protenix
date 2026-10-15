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

"""Prepare reproducible PXM-Legacy subsets for Protenix checkpoint evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


CANONICAL_AF3_TYPES = (
    "intra_prot",
    "intra_ligand",
    "intra_dna",
    "intra_rna",
    "prot_prot",
    "ligand_prot",
    "dna_prot",
    "rna_prot",
)

# Must stay aligned with ``protenix.data.constants.EvaluationChainInterface``.
# The two antibody categories are valid runtime evaluation rows even though
# they are not part of the eight AF3 Figure-2d proxy categories above.
RUNTIME_EVALUATION_TYPES = (
    *CANONICAL_AF3_TYPES,
    "antibody_antigen",
    "antibody",
)

PXM_LIST_NAMES = {
    "recentpdb": "RecentPDB.txt",
    "rna_protein": "RNA_Protein.txt",
    "dna_protein": "dsDNA_Protein.txt",
}


def _read_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _write_ids(path: Path, pdb_ids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{pdb_id}\n" for pdb_id in pdb_ids),
        encoding="utf-8",
    )


def _select_smoke_targets(
    index_df: pd.DataFrame,
    *,
    per_type: int,
) -> list[str]:
    """Greedily select short targets covering each available AF3 metric type."""
    selected: list[str] = []
    selected_set: set[str] = set()
    ordered = index_df.sort_values(["num_tokens", "pdb_id"])
    for eval_type in CANONICAL_AF3_TYPES:
        candidates = (
            ordered.loc[ordered["eval_type"] == eval_type, "pdb_id"]
            .drop_duplicates()
            .tolist()
        )
        type_selected = 0
        for pdb_id in candidates:
            if pdb_id not in selected_set:
                selected.append(pdb_id)
                selected_set.add(pdb_id)
                type_selected += 1
            elif pdb_id in selected_set:
                # An already selected target also covers this category.
                type_selected += 1
            if type_selected >= per_type:
                break
    return selected


def _select_runtime_evaluable_targets(
    available_frame: pd.DataFrame,
    *,
    max_tokens: int | None,
) -> tuple[pd.DataFrame, set[str], set[str]]:
    """Keep only targets accepted by the runtime evaluator.

    ``BaseSingleDataset(find_eval_chain_interface=True)`` removes a target
    unless at least one of its index rows has a supported evaluation type.
    Applying that same type set here makes the on-disk manifest agree with the
    number of samples the evaluator will actually iterate.

    Returns:
        selected rows, target IDs above the token limit, and target IDs with
        no runtime-compatible evaluation row. When ``max_tokens`` is
        ``None``, no token-length filter is applied and the above-limit set is
        empty.
    """
    available_ids = set(available_frame["pdb_id"])
    if max_tokens is None:
        within_token_limit = available_frame
        above_token_limit_ids: set[str] = set()
    else:
        within_token_limit = available_frame[
            available_frame["num_tokens"] <= max_tokens
        ]
        above_token_limit_ids = available_ids - set(
            within_token_limit["pdb_id"]
        )
    within_token_ids = set(within_token_limit["pdb_id"])
    evaluable_ids = set(
        within_token_limit.loc[
            within_token_limit["eval_type"].isin(RUNTIME_EVALUATION_TYPES),
            "pdb_id",
        ]
    )
    selected = within_token_limit[
        within_token_limit["pdb_id"].isin(evaluable_ids)
    ]
    return (
        selected,
        above_token_limit_ids,
        within_token_ids - evaluable_ids,
    )


def _token_limit_slug(max_tokens: int | None) -> str:
    return "alltokens" if max_tokens is None else f"le{max_tokens}"


def _parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pxm-root",
        type=Path,
        default=Path("/mnt/project_abd/klp/proteinx_eval_data/PXM-Legacy"),
    )
    parser.add_argument(
        "--protenix-index",
        type=Path,
        default=Path(
            "/mnt/project_abd/public_databases/protenix.20260101/indices/"
            "recentPDB_low_homology_maxtoken1536.csv"
        ),
    )
    parser.add_argument(
        "--weighted-pdb-index",
        type=Path,
        default=Path(
            "/mnt/project_abd/public_databases/protenix.20260101/indices/"
            "indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz"
        ),
    )
    parser.add_argument(
        "--mmcif-dir",
        type=Path,
        default=Path(
            "/mnt/project_abd/public_databases/protenix.20260101/mmcif"
        ),
    )
    parser.add_argument(
        "--bioassembly-dir",
        type=Path,
        default=Path(
            "/mnt/project_abd/public_databases/protenix.20260101/"
            "recentPDB_bioassembly"
        ),
    )
    token_limit_group = parser.add_mutually_exclusive_group()
    token_limit_group.add_argument("--max-tokens", type=int)
    token_limit_group.add_argument(
        "--no-max-tokens",
        action="store_const",
        const=None,
        dest="max_tokens",
        help="Disable token-count filtering.",
    )
    parser.set_defaults(max_tokens=1024)
    parser.add_argument(
        "--already-evaluated-list",
        type=Path,
        help=(
            "Optional newline-delimited PDB list. In addition to each full "
            "manifest, emit an *_added.txt manifest excluding these targets."
        ),
    )
    parser.add_argument("--smoke-per-type", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/mnt/project_abd/klp/proteinx_eval_data/manifests/"
            "pxm_legacy_af3_like"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.smoke_per_type <= 0:
        raise ValueError("--smoke-per-type must be positive")
    already_evaluated_ids = (
        _read_ids(args.already_evaluated_list)
        if args.already_evaluated_list is not None
        else set()
    )
    token_limit_slug = _token_limit_slug(args.max_tokens)

    supported_data = args.pxm_root / "supported_data"
    list_dir = supported_data / "stat_data" / "pdb_ids"
    source_ids = {
        name: _read_ids(list_dir / filename)
        for name, filename in PXM_LIST_NAMES.items()
    }
    source_ids["combined"] = set().union(*source_ids.values())

    index_df = pd.read_csv(args.protenix_index)
    required_columns = {"pdb_id", "num_tokens", "eval_type"}
    missing_columns = required_columns - set(index_df.columns)
    if missing_columns:
        raise ValueError(
            f"{args.protenix_index} is missing columns {sorted(missing_columns)}"
        )
    index_df = index_df.copy()
    index_df["pdb_id"] = index_df["pdb_id"].astype(str).str.lower()
    local_ids = set(index_df["pdb_id"])
    available_local_ids = {
        pdb_id
        for pdb_id in local_ids
        if (args.mmcif_dir / f"{pdb_id}.cif").is_file()
        and (
            (args.bioassembly_dir / f"{pdb_id}.pkl.gz").is_file()
            or (args.bioassembly_dir / f"{pdb_id}.pkl").is_file()
        )
    }

    weighted_pdb_ids: set[str] = set()
    if args.weighted_pdb_index.is_file():
        weighted_df = pd.read_csv(
            args.weighted_pdb_index,
            usecols=lambda column: column in {"pdb_id", "entry_id"},
        )
        weighted_id_column = (
            "pdb_id" if "pdb_id" in weighted_df.columns else "entry_id"
        )
        weighted_pdb_ids = set(
            weighted_df[weighted_id_column].astype(str).str.lower()
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "schema_version": 1,
        "pxm_root": str(args.pxm_root),
        "protenix_index": str(args.protenix_index),
        "weighted_pdb_index": str(args.weighted_pdb_index),
        "mmcif_dir": str(args.mmcif_dir),
        "bioassembly_dir": str(args.bioassembly_dir),
        "max_tokens": args.max_tokens,
        "token_limit_enabled": args.max_tokens is not None,
        "token_limit_scope": "rows_loaded_from_protenix_index",
        "already_evaluated_list": (
            str(args.already_evaluated_list)
            if args.already_evaluated_list is not None
            else None
        ),
        "already_evaluated_pdb_count": len(already_evaluated_ids),
        "source_counts": {name: len(ids) for name, ids in source_ids.items()},
        "subsets": {},
    }

    selected_frames: dict[str, pd.DataFrame] = {}
    for name, pdb_ids in source_ids.items():
        source_frame = index_df[index_df["pdb_id"].isin(pdb_ids)]
        available_frame = source_frame[
            source_frame["pdb_id"].isin(available_local_ids)
        ]
        (
            selected,
            above_token_limit_ids,
            unsupported_eval_type_ids,
        ) = _select_runtime_evaluable_targets(
            available_frame,
            max_tokens=args.max_tokens,
        )
        selected_frames[name] = selected
        ordered_ids = (
            selected[["pdb_id", "num_tokens"]]
            .drop_duplicates()
            .sort_values(["num_tokens", "pdb_id"])["pdb_id"]
            .tolist()
        )
        output_path = args.output_dir / f"{name}_{token_limit_slug}.txt"
        _write_ids(output_path, ordered_ids)
        added_ids = [
            pdb_id
            for pdb_id in ordered_ids
            if pdb_id not in already_evaluated_ids
        ]
        added_path: Path | None = None
        if args.already_evaluated_list is not None:
            added_path = (
                args.output_dir / f"{name}_{token_limit_slug}_added.txt"
            )
            _write_ids(added_path, added_ids)

        selected_id_set = set(ordered_ids)
        summary["subsets"][name] = {
            "manifest": str(output_path),
            "full_manifest": str(output_path),
            "full_pdb_count": len(selected_id_set),
            "added_manifest": str(added_path) if added_path else None,
            "added_pdb_count": (
                len(added_ids)
                if args.already_evaluated_list is not None
                else None
            ),
            "already_evaluated_selected_overlap": len(
                selected_id_set & already_evaluated_ids
            ),
            "source_pdb_count": len(pdb_ids),
            "present_in_local_index": len(pdb_ids & local_ids),
            "missing_from_local_index": len(pdb_ids - local_ids),
            "missing_local_structure_files": len(
                (pdb_ids & local_ids) - available_local_ids
            ),
            "selected_pdb_count": len(selected_id_set),
            "above_token_limit": len(above_token_limit_ids),
            "unsupported_eval_type_pdb_count": len(
                unsupported_eval_type_ids
            ),
            "unsupported_eval_type_pdb_ids": sorted(
                unsupported_eval_type_ids
            ),
            "direct_weighted_pdb_id_overlap": len(
                selected_id_set & weighted_pdb_ids
            ),
            "eval_row_counts": {
                str(eval_type): int(count)
                for eval_type, count in selected[
                    selected["eval_type"].isin(CANONICAL_AF3_TYPES)
                ]["eval_type"]
                .value_counts()
                .items()
            },
            "eval_target_counts": {
                str(eval_type): int(count)
                for eval_type, count in selected[
                    selected["eval_type"].isin(CANONICAL_AF3_TYPES)
                ]
                .groupby("eval_type")["pdb_id"]
                .nunique()
                .items()
            },
        }

    combined_frame = selected_frames["combined"]
    smoke_ids = _select_smoke_targets(
        combined_frame,
        per_type=args.smoke_per_type,
    )
    smoke_path = args.output_dir / (
        f"combined_{token_limit_slug}_smoke"
        f"{args.smoke_per_type}pertype.txt"
    )
    _write_ids(smoke_path, smoke_ids)
    smoke_frame = combined_frame[combined_frame["pdb_id"].isin(smoke_ids)]
    summary["smoke"] = {
        "manifest": str(smoke_path),
        "pdb_count": len(smoke_ids),
        "pdb_ids": smoke_ids,
        "eval_target_counts": {
            str(eval_type): int(count)
            for eval_type, count in smoke_frame[
                smoke_frame["eval_type"].isin(CANONICAL_AF3_TYPES)
            ]
            .groupby("eval_type")["pdb_id"]
            .nunique()
            .items()
        },
    }

    summary_path = args.output_dir / f"summary_{token_limit_slug}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
