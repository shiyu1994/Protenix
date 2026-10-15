#!/usr/bin/env python3
# Copyright 2025 ByteDance and/or its affiliates.
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
"""Summarize the pairing-first training raw-MSA budget for PXM Ab-Ag.

The input is the entity-level source audit produced by
``analyze_pxm22to25_ab_ag_msa_depth.py``.  This script applies the current
training loader's depth-only quota rule:

* the paired raw pool reads at most ``cap`` rows from ``pairing.a3m``;
* the unpaired raw pool first reads ``pairing.a3m`` using the same ``cap``;
* only its remaining raw-row quota may read ``non_pairing.a3m``.

This calculation is exact for source read quotas when both A3M query rows pass
the runtime parser's validation.  It intentionally does not claim to reproduce
post-deduplication, species pairing, cleanup, or final assembled MSA depths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_INPUT = Path(
    "/mnt/project_abd/klp/proteinx_eval_data/PXM-22to25-Ab-Ag/analysis/"
    "msa_depth_protenix20260101/antibody_entity_msa_depth.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/project_abd/klp/proteinx_eval_data/PXM-22to25-Ab-Ag/analysis/"
    "training_msa_budget_pairing_first_10k"
)
DEFAULT_TRAINING_CONFIG = Path(
    "/mnt/project_abd/klp/proteinx_output/"
    "protenix_v4_step21999_residual_ge200_softcluster_g02_v9_straggler_"
    "20260804_140441/config.yaml"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int(row: Mapping[str, str], field: str) -> int:
    value = int(row[field])
    if value < 0:
        raise ValueError(f"{field} must be non-negative, got {value}")
    return value


def simulate_depth_only_budget(
    row: Mapping[str, str],
    cap: int,
) -> dict[str, Any]:
    """Apply the pairing-first raw-row quota to one antibody entity."""

    if cap < 1:
        raise ValueError(f"cap must be positive, got {cap}")
    pairing_raw = _int(row, "pairing_raw_depth")
    nonpairing_raw = _int(row, "non_pairing_raw_depth")
    pairing_unique = _int(row, "pairing_unique_depth")
    nonpairing_unique = _int(row, "non_pairing_unique_depth")
    if pairing_unique > pairing_raw:
        raise ValueError("pairing_unique_depth cannot exceed pairing_raw_depth")
    if nonpairing_unique > nonpairing_raw:
        raise ValueError(
            "non_pairing_unique_depth cannot exceed non_pairing_raw_depth"
        )

    pairing_read = min(pairing_raw, cap)
    remaining = cap - pairing_read
    nonpairing_read = min(nonpairing_raw, remaining)
    nonpairing_not_read = nonpairing_raw - nonpairing_read

    # The source depth includes row zero (the query).  Count homolog hits
    # separately so a one-row read is not described as retaining a genuine
    # non-pairing homolog.
    nonpairing_hits = max(0, nonpairing_raw - 1)
    nonpairing_hits_read = max(0, nonpairing_read - 1)
    nonpairing_hits_not_read = nonpairing_hits - nonpairing_hits_read
    nonpairing_hit_survival = (
        nonpairing_hits_read / nonpairing_hits if nonpairing_hits else 1.0
    )

    pairing_duplicates = max(0, pairing_raw - pairing_unique)
    nonpairing_duplicates = max(0, nonpairing_raw - nonpairing_unique)
    return {
        **row,
        "raw_budget_cap": cap,
        "paired_pool_pairing_rows_read_including_query": pairing_read,
        "unpaired_pool_pairing_rows_read_including_query": pairing_read,
        "unpaired_pool_remaining_after_pairing": remaining,
        "nonpairing_source_rows_read_including_query": nonpairing_read,
        "nonpairing_source_rows_not_read_including_query": nonpairing_not_read,
        "nonpairing_homolog_hits": nonpairing_hits,
        "nonpairing_homolog_hits_read": nonpairing_hits_read,
        "nonpairing_homolog_hits_not_read": nonpairing_hits_not_read,
        "nonpairing_homolog_survival_fraction": nonpairing_hit_survival,
        "nonpairing_homologs_starved": int(
            nonpairing_hits > 0 and nonpairing_hits_read == 0
        ),
        "nonpairing_homologs_fully_read": int(
            nonpairing_hits_read == nonpairing_hits
        ),
        "pairing_full_file_duplicate_rows": pairing_duplicates,
        "pairing_full_file_duplicate_fraction": (
            pairing_duplicates / pairing_raw if pairing_raw else 0.0
        ),
        "nonpairing_full_file_duplicate_rows": nonpairing_duplicates,
        "nonpairing_full_file_duplicate_fraction": (
            nonpairing_duplicates / nonpairing_raw if nonpairing_raw else 0.0
        ),
    }


def _roles(row: Mapping[str, Any]) -> tuple[str, ...]:
    roles = tuple(role for role in str(row.get("roles", "")).split(";") if role)
    return roles or ("unknown",)


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return quota and full-file duplicate summaries for a row collection."""

    if not rows:
        raise ValueError("cannot summarize an empty row collection")
    nonpairing_hits = sum(int(row["nonpairing_homolog_hits"]) for row in rows)
    nonpairing_hits_read = sum(
        int(row["nonpairing_homolog_hits_read"]) for row in rows
    )
    pairing_raw = sum(int(row["pairing_raw_depth"]) for row in rows)
    nonpairing_raw = sum(int(row["non_pairing_raw_depth"]) for row in rows)
    pairing_duplicates = sum(
        int(row["pairing_full_file_duplicate_rows"]) for row in rows
    )
    nonpairing_duplicates = sum(
        int(row["nonpairing_full_file_duplicate_rows"]) for row in rows
    )
    starved = sum(int(row["nonpairing_homologs_starved"]) for row in rows)
    fully_read = sum(
        int(row["nonpairing_homologs_fully_read"]) for row in rows
    )
    return {
        "entity_count": len(rows),
        "pairing_raw_depth_median": statistics.median(
            int(row["pairing_raw_depth"]) for row in rows
        ),
        "nonpairing_raw_depth_median": statistics.median(
            int(row["non_pairing_raw_depth"]) for row in rows
        ),
        "nonpairing_homologs_starved_count": starved,
        "nonpairing_homologs_starved_percent": starved / len(rows) * 100,
        "nonpairing_homologs_fully_read_count": fully_read,
        "nonpairing_homologs_fully_read_percent": fully_read / len(rows) * 100,
        "nonpairing_homolog_hits": nonpairing_hits,
        "nonpairing_homolog_hits_read": nonpairing_hits_read,
        "nonpairing_homolog_hits_read_percent": (
            nonpairing_hits_read / nonpairing_hits * 100
            if nonpairing_hits
            else 100.0
        ),
        "pairing_full_file_duplicate_fraction_percent": (
            pairing_duplicates / pairing_raw * 100 if pairing_raw else 0.0
        ),
        "nonpairing_full_file_duplicate_fraction_percent": (
            nonpairing_duplicates / nonpairing_raw * 100
            if nonpairing_raw
            else 0.0
        ),
    }


def summarize_targets(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_target[str(row["pdb_id"])].append(row)
    target_count = len(by_target)

    def count(predicate: Any) -> int:
        return sum(predicate(target_rows) for target_rows in by_target.values())

    any_starved = count(
        lambda target_rows: any(
            int(row["nonpairing_homologs_starved"]) for row in target_rows
        )
    )
    all_starved = count(
        lambda target_rows: all(
            int(row["nonpairing_homologs_starved"]) for row in target_rows
        )
    )
    any_loss = count(
        lambda target_rows: any(
            int(row["nonpairing_homolog_hits_not_read"]) > 0
            for row in target_rows
        )
    )
    all_fully_read = count(
        lambda target_rows: all(
            int(row["nonpairing_homologs_fully_read"]) for row in target_rows
        )
    )
    return {
        "target_count": target_count,
        "any_antibody_entity_starved_count": any_starved,
        "any_antibody_entity_starved_percent": any_starved / target_count * 100,
        "all_antibody_entities_starved_count": all_starved,
        "all_antibody_entities_starved_percent": all_starved / target_count * 100,
        "any_antibody_entity_loses_nonpairing_count": any_loss,
        "any_antibody_entity_loses_nonpairing_percent": any_loss
        / target_count
        * 100,
        "all_antibody_entities_fully_read_count": all_fully_read,
        "all_antibody_entities_fully_read_percent": all_fully_read
        / target_count
        * 100,
    }


def markdown_table(headers: Sequence[str], body: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def build_report(summary: Mapping[str, Any]) -> str:
    role_order = [
        role
        for role in ("all", "H", "L", "scFv", "unknown")
        if role in summary["by_role"]
    ]
    known_roles = set(role_order)
    role_order.extend(
        sorted(role for role in summary["by_role"] if role not in known_roles)
    )
    role_rows = []
    for role in role_order:
        values = summary["by_role"][role]
        role_rows.append(
            [
                role,
                f'{values["entity_count"]:,}',
                f'{values["pairing_raw_depth_median"]:,.0f}',
                f'{values["nonpairing_raw_depth_median"]:,.0f}',
                f'{values["nonpairing_homologs_starved_count"]:,} '
                f'({values["nonpairing_homologs_starved_percent"]:.1f}%)',
                f'{values["nonpairing_homologs_fully_read_count"]:,} '
                f'({values["nonpairing_homologs_fully_read_percent"]:.1f}%)',
                f'{values["nonpairing_homolog_hits_read_percent"]:.1f}%',
                f'{values["pairing_full_file_duplicate_fraction_percent"]:.2f}%',
                f'{values["nonpairing_full_file_duplicate_fraction_percent"]:.2f}%',
            ]
        )
    targets = summary["targets"]
    return "\n".join(
        [
            "# PXM-22to25-Ab-Ag homolog-I/O 分支 source raw-budget 审计",
            "",
            "## 口径",
            "",
            f'- 输入包含 {targets["target_count"]:,} targets；统计单位是抗体 sequence entity occurrence。',
            f'- 共 {summary["unique_sources"]["entity_count"]:,} 个唯一 MSA '
            "source；同一 source 可在多个 target 中重复。",
            f'- 模拟 raw paired/unpaired pool cap 均为 {summary["raw_budget_cap"]:,}。',
            "- source raw depth 和 10k cap 均包含 row 0 query；"
            "表中的读取/饿死指标只统计 query 之外的 non-pairing raw rows。",
            "- unpaired raw pool 先读取 `pairing.a3m`，剩余预算再读取 `non_pairing.a3m`。",
            "- 本报告是 depth-only source quota 审计；尚未模拟 query fallback、"
            "去重、species pairing、cleanup 或最终 16,384 裁剪。",
            "",
            "## 按抗体角色",
            "",
            markdown_table(
                [
                    "角色",
                    "实体数",
                    "pairing中位数",
                    "non-pairing中位数",
                    "non-pairing非query raw row读取0条",
                    "non-pairing source完整读取",
                    "aggregate raw-row读取率",
                    "pairing全文件重复率",
                    "non-pairing全文件重复率",
                ],
                role_rows,
            ),
            "",
            "## Target 级覆盖",
            "",
            "- 至少一个抗体实体的 non-pairing 非 query raw rows 读取为 0："
            f'{targets["any_antibody_entity_starved_count"]:,}/'
            f'{targets["target_count"]:,} '
            f'({targets["any_antibody_entity_starved_percent"]:.1f}%)。',
            "- 所有抗体实体的 non-pairing 非 query raw rows 读取都为 0："
            f'{targets["all_antibody_entities_starved_count"]:,}/'
            f'{targets["target_count"]:,} '
            f'({targets["all_antibody_entities_starved_percent"]:.1f}%)。',
            "- 至少一个抗体实体未读取全部 non-pairing 非 query raw rows："
            f'{targets["any_antibody_entity_loses_nonpairing_count"]:,}/'
            f'{targets["target_count"]:,} '
            f'({targets["any_antibody_entity_loses_nonpairing_percent"]:.1f}%)。',
            "- 所有抗体实体都读取全部 non-pairing 非 query raw rows："
            f'{targets["all_antibody_entities_fully_read_count"]:,}/'
            f'{targets["target_count"]:,} '
            f'({targets["all_antibody_entities_fully_read_percent"]:.1f}%)。',
            "- 唯一 source 口径下，non-pairing 非 query raw rows 读取为 0："
            f'{summary["unique_sources"]["nonpairing_homologs_starved_count"]:,}/'
            f'{summary["unique_sources"]["entity_count"]:,} '
            f'({summary["unique_sources"]["nonpairing_homologs_starved_percent"]:.1f}%)。',
            "",
            "## 限制",
            "",
            "- 只有 A3M row 0 通过当前 parser 的 feature-equivalent query "
            "校验时，raw-row额度计算才等于运行时行为。",
            "- full-file duplicate fraction 不是 top-10k prefix duplicate fraction。",
            "- raw-row 读取率是按行汇总的 micro-average，不是 entity 平均值、"
            "unique/Neff 或最终模型输入保留率。",
            "- query-only/partial-query-only 分支不会产生这里模拟的 homolog "
            "I/O；本报告不是实际训练 exposure 加权。",
            "- 这些结果不能单独证明 checkpoint 回退；需要与同一 target 的 "
            "delta、final paired fraction 和 species overlap 联表。",
            "",
        ]
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(
    input_csv: Path,
    output_dir: Path,
    cap: int,
    training_config: Path = DEFAULT_TRAINING_CONFIG,
) -> dict[str, Any]:
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError(f"input CSV has no rows: {input_csv}")
    rows = [simulate_depth_only_budget(row, cap) for row in source_rows]
    unique_source_rows = list(
        {str(row["msa_relative_dir"]): row for row in rows}.values()
    )

    by_role: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_role["all"].extend(rows)
    for row in rows:
        for role in _roles(row):
            by_role[role].append(row)

    summary = {
        "schema_version": 3,
        "semantics": "pairing_first_shared_unpaired_raw_budget_depth_only_v3",
        "input_csv": str(input_csv.resolve()),
        "input_sha256": sha256_file(input_csv),
        "raw_budget_cap": cap,
        "raw_depth_and_cap_include_query_row": True,
        "training_config": str(training_config.resolve()),
        "training_config_sha256": sha256_file(training_config),
        "loader_projection": {
            "paired_source_order": ["pairing"],
            "unpaired_source_order": ["pairing", "non_pairing"],
            "paired_raw_cap": cap,
            "unpaired_raw_cap": cap,
        },
        "by_role": {
            role: summarize_rows(role_rows)
            for role, role_rows in sorted(by_role.items())
        },
        "targets": summarize_targets(rows),
        "unique_sources": summarize_rows(unique_source_rows),
        "limitations": [
            "query validation fallback is not simulated",
            "post-read deduplication is not simulated",
            "species pairing and cleanup are not simulated",
            "final assembled 16384-row cropping is not simulated",
            "full-file duplicate rates do not equal top-prefix duplicate rates",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "antibody_entity_training_raw_budget.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(build_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cap", type=int, default=10000)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
    )
    args = parser.parse_args()
    summary = run(
        args.input_csv,
        args.output_dir,
        args.cap,
        args.training_config,
    )
    print(json.dumps(summary["targets"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
