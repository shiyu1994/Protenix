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
"""Audit antibody MSA depths for the 528-target PXM-22to25-Ab-Ag set.

The PXMeter download contains sequences and native structures but no MSA files.
This audit resolves every antibody sequence against the Protenix 20260101 MSA
database, then combines ``pairing.a3m`` and ``non_pairing.a3m`` using the same
sequence-level deduplication rule as Protenix: lower-case A3M insertions are
removed and the query is retained once.

At target level, an antibody containing multiple unique sequence entities is
represented by one query plus the union-equivalent hit counts of its entities::

    antibody_msa_depth = 1 + sum(entity_combined_unique_depth - 1)

Heavy and light chains use different alignment coordinate systems, so they are
not directly sequence-deduplicated against each other.  The target scalar is an
evidence-volume statistic for sampling analysis, not the final model ``N_msa``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import gemmi


DEFAULT_DATASET_ROOT = Path(
    "/mnt/project_abd/klp/proteinx_eval_data/PXM-22to25-Ab-Ag"
)
DEFAULT_PROTENIX_ROOT = Path("/mnt/project_abd/public_databases/protenix.20260101")
LOWERCASE_TRANSLATION = str.maketrans("", "", string.ascii_lowercase)
MISSING_VALUES = {"", ".", "?", "na", "nan", "none", "null"}

# SAbDab2 represents these Fv/scFv chains as synthetic domain IDs (for example,
# B3/B1) that are not PDB auth-chain IDs.  Three Gluebody entries are absent
# from SAbDab2.  Their antibody endpoint is unambiguous in the frozen mmCIF
# entity descriptions and PXMeter's tagged antibody-protein interface.
MANUAL_ANTIBODY_ENTITIES: Mapping[str, set[str]] = {
    "7tzg": {"1"},  # scFvF7
    "7tzh": {"1"},  # scFvF7
    "7yix": {"2"},  # scFv
    "8cz8": {"1", "3"},  # two anti-mesothelin scFvs
    "8rl5": {"1"},  # Gluebody G5-006
    "8rl9": {"3"},  # Gluebody G5-006 on the tagged interface
    "8rlc": {"3"},  # Gluebody GbC4 on the tagged interface
    "8rz1": {"1"},  # R5.034 scFv
    "9gry": {"2"},  # scFv/Fv-MBP fusion
    "9gs3": {"2"},
    "9gs5": {"2"},
    "9gs7": {"2"},
    "9gsl": {"2"},
    "9i20": {"2"},
}
MANUAL_ENTITY_ROLE = {
    (pdb_id, entity_id): ("H" if pdb_id.startswith("8rl") else "scFv")
    for pdb_id, entity_ids in MANUAL_ANTIBODY_ENTITIES.items()
    for entity_id in entity_ids
}

TARGET_FIELDS = (
    "pdb_id",
    "subclasses",
    "antibody_entity_ids",
    "antibody_entity_count",
    "antibody_sequence_count",
    "antibody_sequence_lengths",
    "pairing_raw_depth_sum",
    "non_pairing_raw_depth_sum",
    "raw_depth_sum",
    "pairing_msa_depth",
    "non_pairing_msa_depth",
    "pair_unpair_overlap_depth",
    "pairing_homolog_count",
    "non_pairing_homolog_count",
    "pair_unpair_overlap_homolog_count",
    "chain_combined_depth_sum",
    "antibody_msa_depth",
    "antibody_homolog_count",
    "min_chain_combined_depth",
    "max_chain_combined_depth",
)
ENTITY_FIELDS = (
    "pdb_id",
    "entity_id",
    "label_chain_ids",
    "roles",
    "sequence_length",
    "msa_relative_dir",
    "pairing_raw_depth",
    "pairing_unique_depth",
    "non_pairing_raw_depth",
    "non_pairing_unique_depth",
    "combined_unique_depth",
    "unique_homolog_count",
    "raw_duplicate_rows",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_a3m_sequences(path: Path) -> Iterator[str]:
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
    return sequence.translate(LOWERCASE_TRANSLATION)


def scan_a3m(path: Path, query: str) -> tuple[int, set[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw_depth = 0
    first_length: int | None = None
    unique_sequences: set[str] = set()
    for sequence in iter_a3m_sequences(path):
        canonical = canonical_a3m_sequence(sequence)
        if first_length is None:
            first_length = len(canonical)
        raw_depth += 1
        unique_sequences.add(canonical)
    if raw_depth and first_length != len(query):
        return raw_depth, {query}
    if not unique_sequences:
        unique_sequences.add(query)
    return raw_depth, unique_sequences


def sequence_msa_stats(
    sequence: str,
    sequence_to_msa_dir: Mapping[str, Any],
    msa_root: Path,
) -> dict[str, Any]:
    mapping_value = sequence_to_msa_dir.get(sequence)
    if mapping_value is None:
        raise KeyError(f"sequence is absent from MSA mapping: length={len(sequence)}")
    relative_dir = str(mapping_value)
    query = canonical_a3m_sequence(sequence)
    pair_raw, pair_unique = scan_a3m(msa_root / relative_dir / "pairing.a3m", query)
    unpair_raw, unpair_unique = scan_a3m(
        msa_root / relative_dir / "non_pairing.a3m", query
    )
    combined = pair_unique | unpair_unique
    return {
        "msa_relative_dir": relative_dir,
        "pairing_raw_depth": pair_raw,
        "pairing_unique_depth": len(pair_unique),
        "non_pairing_raw_depth": unpair_raw,
        "non_pairing_unique_depth": len(unpair_unique),
        "combined_unique_depth": len(combined),
        "unique_homolog_count": max(0, len(combined) - 1),
        "raw_duplicate_rows": pair_raw + unpair_raw - len(combined),
    }


def normalize_pdb_id(value: str) -> str:
    text = value.strip().lower()
    if text.startswith("pdb_"):
        return text[-4:]
    return text


def detect_csv_dialect(path: Path) -> csv.Dialect:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(8192)
    return csv.Sniffer().sniff(sample, delimiters=",\t")


def load_sabdab_auth_roles(
    path: Path, targets: set[str]
) -> dict[str, dict[str, set[str]]]:
    roles: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    dialect = detect_csv_dialect(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(f"SAbDab summary has no header: {path}")
        fields = {name.lower(): name for name in reader.fieldnames}
        pdb_field = fields.get("pdb")
        h_field = fields.get("hchain")
        l_field = fields.get("lchain")
        if not pdb_field or not h_field or not l_field:
            raise ValueError(
                "SAbDab summary must contain PDB/pdb, Hchain, and Lchain columns"
            )
        for row in reader:
            pdb_id = normalize_pdb_id(row[pdb_field])
            if pdb_id not in targets:
                continue
            for field, role in ((h_field, "H"), (l_field, "L")):
                chain_id = row[field].strip()
                if chain_id.lower() not in MISSING_VALUES:
                    roles[pdb_id][chain_id].add(role)
    return roles


def cif_chain_maps(cif_path: Path) -> tuple[dict[str, set[str]], dict[str, str]]:
    block = gemmi.cif.read_file(str(cif_path)).sole_block()
    auth_to_labels: dict[str, set[str]] = defaultdict(set)
    for row in block.find(
        ["_pdbx_poly_seq_scheme.asym_id", "_pdbx_poly_seq_scheme.pdb_strand_id"]
    ):
        auth_to_labels[str(row[1])].add(str(row[0]))
    label_to_entity = {
        str(row[0]): str(row[1])
        for row in block.find(["_struct_asym.id", "_struct_asym.entity_id"])
    }
    return auth_to_labels, label_to_entity


def load_tagged_edges(metadata_path: Path) -> dict[str, set[tuple[str, str]]]:
    edges: dict[str, set[tuple[str, str]]] = defaultdict(set)
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if "[antibody-protein]" not in row["subset"]:
                continue
            edge = tuple(sorted((row["entity_id_1"], row["entity_id_2"])))
            edges[row["entry_id"].lower()].add(edge)
    return edges


def load_subclass_memberships(dataset_root: Path) -> dict[str, set[str]]:
    subset_dir = dataset_root / "supported_data/stat_data/pdb_ids/subset"
    memberships: dict[str, set[str]] = defaultdict(set)
    for label, filename in (
        ("HL", "antibody_HL-protein.txt"),
        ("H", "antibody_H-protein.txt"),
        ("scFv", "antibody_scFv-protein.txt"),
    ):
        for pdb_id in (subset_dir / filename).read_text().split():
            memberships[pdb_id.lower()].add(label)
    return memberships


def load_target_entities(
    dataset_root: Path,
    targets: Sequence[str],
    sabdab_auth_roles: Mapping[str, Mapping[str, set[str]]],
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, list[str]]],
    dict[str, dict[str, set[str]]],
    dict[str, set[str]],
]:
    entity_sequences: dict[str, dict[str, str]] = {}
    entity_labels: dict[str, dict[str, list[str]]] = {}
    entity_roles: dict[str, dict[str, set[str]]] = {}
    known_antibody_entities: dict[str, set[str]] = {}

    for pdb_id in targets:
        auth_to_labels, label_to_entity = cif_chain_maps(
            dataset_root / "supported_data/mmcif" / f"{pdb_id}.cif"
        )
        input_data = json.loads(
            (dataset_root / "protenix_inputs" / f"{pdb_id}.json").read_text()
        )[0]
        sequences: dict[str, str] = {}
        labels: dict[str, list[str]] = defaultdict(list)
        for item in input_data["sequences"]:
            if "proteinChain" not in item:
                continue
            protein = item["proteinChain"]
            # PXMeter appends ``.1``, ``.2``, ... when a generated assembly
            # contains repeated copies of one label asym ID.  Those generated
            # IDs are absent from the deposited mmCIF ``struct_asym`` table and
            # retain the base label's entity assignment.
            item_entities = {
                label_to_entity.get(label, label_to_entity[label.split(".", 1)[0]])
                for label in protein["id"]
            }
            if len(item_entities) != 1:
                raise ValueError(f"{pdb_id}: one JSON protein item spans entities")
            entity_id = next(iter(item_entities))
            old_sequence = sequences.setdefault(entity_id, protein["sequence"])
            if old_sequence != protein["sequence"]:
                raise ValueError(f"{pdb_id}: inconsistent sequence for entity {entity_id}")
            labels[entity_id].extend(protein["id"])

        roles: dict[str, set[str]] = defaultdict(set)
        for auth_chain, auth_roles in sabdab_auth_roles.get(pdb_id, {}).items():
            for label_chain in auth_to_labels.get(auth_chain, set()):
                entity_id = label_to_entity[label_chain]
                roles[entity_id].update(auth_roles)
        for entity_id in MANUAL_ANTIBODY_ENTITIES.get(pdb_id, set()):
            roles[entity_id].add(MANUAL_ENTITY_ROLE[(pdb_id, entity_id)])

        entity_sequences[pdb_id] = sequences
        entity_labels[pdb_id] = {key: sorted(set(value)) for key, value in labels.items()}
        entity_roles[pdb_id] = roles
        known_antibody_entities[pdb_id] = set(roles)

    return entity_sequences, entity_labels, entity_roles, known_antibody_entities


def orient_antibody_edges(
    targets: Sequence[str],
    tagged_edges: Mapping[str, set[tuple[str, str]]],
    known_antibody_entities: Mapping[str, set[str]],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for pdb_id in targets:
        antibody_entities: set[str] = set()
        edges = tagged_edges.get(pdb_id, set())
        if not edges:
            raise ValueError(f"{pdb_id}: no tagged antibody-protein interface")
        for edge in edges:
            antibody_endpoints = [
                entity_id
                for entity_id in edge
                if entity_id in known_antibody_entities[pdb_id]
            ]
            if len(antibody_endpoints) != 1:
                raise ValueError(
                    f"{pdb_id}: cannot orient tagged edge {edge}; "
                    f"known antibody entities={known_antibody_entities[pdb_id]}"
                )
            antibody_entities.add(antibody_endpoints[0])
        result[pdb_id] = antibody_entities
    return result


def quantile(values: Sequence[int], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def descriptive_stats(values: Sequence[int]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "sum": sum(values),
        "min": min(values),
        "p05": quantile(values, 0.05),
        "p10": quantile(values, 0.10),
        "p25": quantile(values, 0.25),
        "median": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def histogram(values: Sequence[int]) -> list[dict[str, Any]]:
    # Five-thousand-row bins expose the H-only versus H/L bimodality much more
    # clearly than powers-of-two bins; threshold_cdf() retains the common
    # training-cap values separately.
    boundaries = (5000, 10000, 15000, 20000, 25000, 30000, 40000)
    rows: list[dict[str, Any]] = []
    lower = 1
    for upper in boundaries:
        count = sum(lower <= value <= upper for value in values)
        rows.append(
            {
                "bin": f"{lower:,}-{upper:,}",
                "lower": lower,
                "upper": upper,
                "count": count,
                "percent": count / len(values) * 100,
            }
        )
        lower = upper + 1
    count = sum(value >= lower for value in values)
    rows.append(
        {
            "bin": f">={lower:,}",
            "lower": lower,
            "upper": None,
            "count": count,
            "percent": count / len(values) * 100,
        }
    )
    return rows


def threshold_cdf(values: Sequence[int]) -> list[dict[str, Any]]:
    thresholds = (128, 256, 512, 1024, 2048, 4096, 8192, 10000, 16384, 32768)
    return [
        {
            "threshold": threshold,
            "at_or_below_count": sum(value <= threshold for value in values),
            "at_or_below_percent": sum(value <= threshold for value in values)
            / len(values)
            * 100,
            "at_or_above_count": sum(value >= threshold for value in values),
            "at_or_above_percent": sum(value >= threshold for value in values)
            / len(values)
            * 100,
        }
        for threshold in thresholds
    ]


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: float | int) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.1f}"


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_histogram_svg(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    width, height = 1100, 620
    left, right, top, bottom = 90, 30, 55, 135
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_count = max(int(row["count"]) for row in rows) or 1
    slot = plot_width / len(rows)
    bar_width = slot * 0.72
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.axis{stroke:#4b5563;stroke-width:1}'
        '.grid{stroke:#e5e7eb;stroke-width:1}.bar{fill:#3b82f6}</style>',
        '<text x="550" y="30" text-anchor="middle" font-size="22" font-weight="700">'
        "PXM-22to25-Ab-Ag antibody MSA depth (528 targets)</text>",
    ]
    for tick in range(0, 6):
        count = max_count * tick / 5
        y = top + plot_height - plot_height * tick / 5
        elements.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        elements.append(
            f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="13">{count:.0f}</text>'
        )
    elements.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}"/>',
        ]
    )
    for index, row in enumerate(rows):
        count = int(row["count"])
        bar_height = plot_height * count / max_count
        x = left + index * slot + (slot - bar_width) / 2
        y = top + plot_height - bar_height
        center = x + bar_width / 2
        elements.append(
            f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="3"/>'
        )
        elements.append(
            f'<text x="{center:.1f}" y="{y-8:.1f}" text-anchor="middle" font-size="13">{count}</text>'
        )
        elements.append(
            f'<text x="{center:.1f}" y="{top+plot_height+23}" text-anchor="end" font-size="12" '
            f'transform="rotate(-40 {center:.1f} {top+plot_height+23})">{row["bin"]}</text>'
        )
    elements.append(
        f'<text x="{left+plot_width/2:.1f}" y="{height-18}" text-anchor="middle" font-size="15">'
        "combined antibody MSA depth (query once + unique homolog rows)</text>"
    )
    elements.append(
        f'<text x="20" y="{top+plot_height/2:.1f}" text-anchor="middle" font-size="15" '
        f'transform="rotate(-90 20 {top+plot_height/2:.1f})">target count</text>'
    )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def build_report(
    summary: Mapping[str, Any],
    target_rows: Sequence[Mapping[str, Any]],
) -> str:
    stats = summary["overall"]
    paired_unpaired = summary["paired_unpaired"]
    target_depth_totals = paired_unpaired["target_depth_totals"]
    histogram_rows = summary["histogram"]
    cdf_rows = summary["threshold_cdf"]
    lines = [
        "# PXM-22to25-Ab-Ag 抗体 MSA 深度分布",
        "",
        "## 口径",
        "",
        "- 数据集为 528-target `PXM-22to25-Ab-Ag`；不是旧 PXM-Legacy 的 53 targets / 115 interfaces。",
        "- PXMeter 下载包本身没有 MSA。MSA 来自 Protenix 20260101 的 `pairing.a3m` 和 "
        "`non_pairing.a3m`。",
        "- 每条抗体序列先删除 A3M 小写插入，再对 paired/unpaired 做集合并集，"
        "query 只计一次。",
        "- target 有多条抗体序列时，主指标为 "
        "`1 + Σ(每条序列 combined_unique_depth - 1)`；这表示源 MSA "
        "证据量，不等于模型最终多链拼接后的 `N_msa`。",
        "",
        "## 总体分布",
        "",
        markdown_table(
            ["N", "min", "P5", "P10", "P25", "median", "P75", "P90", "P95", "P99", "max", "mean"],
            [
                [
                    format_number(stats[key])
                    for key in ("n", "min", "p05", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max", "mean")
                ]
            ],
        ),
        "",
        "## Paired / unpaired 拆分",
        "",
        markdown_table(
            [
                "来源",
                "总量",
                "mean",
                "min",
                "P25",
                "median",
                "P75",
                "P90",
                "P95",
                "P99",
                "max",
            ],
            [
                [
                    label,
                    format_number(value["sum"]),
                    format_number(value["mean"]),
                    format_number(value["min"]),
                    format_number(value["p25"]),
                    format_number(value["median"]),
                    format_number(value["p75"]),
                    format_number(value["p90"]),
                    format_number(value["p95"]),
                    format_number(value["p99"]),
                    format_number(value["max"]),
                ]
                for label, value in (
                    (
                        "Paired",
                        paired_unpaired["target_pairing_msa_depth"],
                    ),
                    (
                        "Unpaired",
                        paired_unpaired["target_non_pairing_msa_depth"],
                    ),
                    (
                        "Pair∩Unpair",
                        paired_unpaired["target_pair_unpair_overlap_depth"],
                    ),
                    ("Union", stats),
                )
            ],
        ),
        "",
        "互斥构成：",
        "",
        markdown_table(
            ["部分", "行数", "占 union"],
            [
                [
                    "Pair-only",
                    f'{target_depth_totals["pairing_only"]:,}',
                    f'{target_depth_totals["pairing_only_percent"]:.4f}%',
                ],
                [
                    "Pair∩Unpair",
                    f'{target_depth_totals["pair_unpair_overlap"]:,}',
                    f'{target_depth_totals["pair_unpair_overlap_percent"]:.4f}%',
                ],
                [
                    "Unpair-only",
                    f'{target_depth_totals["non_pairing_only"]:,}',
                    f'{target_depth_totals["non_pairing_only_percent"]:.4f}%',
                ],
            ],
        ),
        "",
        "单条抗体序列（835 个 target occurrence）：",
        "",
        markdown_table(
            ["来源", "P25", "median", "P75", "P90", "max"],
            [
                [
                    label,
                    format_number(value["p25"]),
                    format_number(value["median"]),
                    format_number(value["p75"]),
                    format_number(value["p90"]),
                    format_number(value["max"]),
                ]
                for label, value in (
                    (
                        "Paired",
                        paired_unpaired["chain_pairing_unique_depth"],
                    ),
                    (
                        "Unpaired",
                        paired_unpaired["chain_non_pairing_unique_depth"],
                    ),
                )
            ],
        ),
        "",
        "子类 median（paired / unpaired / intersection / union）：",
        "",
        markdown_table(
            ["子类", "targets", "Paired", "Unpaired", "交集", "Union"],
            [
                [
                    label,
                    value["combined"]["n"],
                    format_number(value["pairing"]["median"]),
                    format_number(value["non_pairing"]["median"]),
                    format_number(value["overlap"]["median"]),
                    format_number(value["combined"]["median"]),
                ]
                for label, value in summary[
                    "paired_unpaired_by_subclass"
                ].items()
            ],
        ),
        "",
        "恒等式：`Paired + Unpaired - Intersection = Union`。",
        "",
        "### 不同使用视角",
        "",
        markdown_table(
            ["口径", "N", "P25", "median", "P75", "P90", "mean", "max"],
            [
                [
                    label,
                    value["n"],
                    format_number(value["p25"]),
                    format_number(value["median"]),
                    format_number(value["p75"]),
                    format_number(value["p90"]),
                    format_number(value["mean"]),
                    format_number(value["max"]),
                ]
                for label, value in summary["source_depth_views"].items()
            ],
        ),
        "",
        f'- raw 两文件共 {summary["raw_total_rows"]:,} 行；union-dedup 后为 '
        f'{summary["deduplicated_chain_rows"]:,} 行，删除重复 '
        f'{summary["raw_duplicate_rows"]:,} 行（{summary["raw_duplicate_percent"]:.2f}%）。',
        "- `单条抗体序列（target occurrence）` 最接近逐链 MSA 采样容量；"
        "`target 最深抗体链` 接近横向 padding 前的容量上界；二者都不是包含抗原和"
        "物种配对后的运行时 `N_msa`。",
        "",
        "![MSA depth histogram](msa_depth_histogram.svg)",
        "",
        "## 主要分桶",
        "",
        markdown_table(
            ["MSA depth", "targets", "占比"],
            [
                [row["bin"], row["count"], f'{row["percent"]:.1f}%']
                for row in histogram_rows
            ],
        ),
        "",
        "## 采样阈值覆盖",
        "",
        markdown_table(
            ["阈值", "≤阈值", "≤占比", "≥阈值", "≥占比"],
            [
                [
                    f'{row["threshold"]:,}',
                    row["at_or_below_count"],
                    f'{row["at_or_below_percent"]:.1f}%',
                    row["at_or_above_count"],
                    f'{row["at_or_above_percent"]:.1f}%',
                ]
                for row in cdf_rows
            ],
        ),
        "",
        "## 子类",
        "",
        markdown_table(
            ["子类", "targets", "median", "P25", "P75", "P90", "mean"],
            [
                [
                    label,
                    value["n"],
                    format_number(value["median"]),
                    format_number(value["p25"]),
                    format_number(value["p75"]),
                    format_number(value["p90"]),
                    format_number(value["mean"]),
                ]
                for label, value in summary["by_subclass"].items()
            ],
        ),
        "",
        "## 最浅 / 最深 targets",
        "",
        markdown_table(
            ["方向", "PDB", "子类", "抗体序列数", "MSA depth"],
            [
                [
                    direction,
                    row["pdb_id"],
                    row["subclasses"],
                    row["antibody_sequence_count"],
                    f'{int(row["antibody_msa_depth"]):,}',
                ]
                for direction, rows in (
                    ("浅", sorted(target_rows, key=lambda item: int(item["antibody_msa_depth"]))[:10]),
                    ("深", sorted(target_rows, key=lambda item: int(item["antibody_msa_depth"]), reverse=True)[:10]),
                )
                for row in rows
            ],
        ),
        "",
        "## 输出",
        "",
        "- `target_msa_depth.csv`: 每个 target 的主指标和链级范围。",
        "- `antibody_entity_msa_depth.csv`: 每个抗体实体的 paired/unpaired raw 与去重深度。",
        "- `summary.json`: 机器可读统计、分桶、阈值 CDF 与输入校验和。",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    protenix_root = args.protenix_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        value.lower()
        for value in (dataset_root / "target_pdb_ids.txt").read_text().split()
    ]
    if len(targets) != 528 or len(set(targets)) != 528:
        raise ValueError(f"expected 528 unique targets, got {len(targets)}")

    sabdab_roles = load_sabdab_auth_roles(args.sabdab_summary, set(targets))
    tagged_edges = load_tagged_edges(
        dataset_root / "supported_data/RecentPDB_low_homology.csv"
    )
    subclasses = load_subclass_memberships(dataset_root)
    (
        entity_sequences,
        entity_labels,
        entity_roles,
        known_antibody_entities,
    ) = load_target_entities(dataset_root, targets, sabdab_roles)
    antibody_entities = orient_antibody_edges(
        targets, tagged_edges, known_antibody_entities
    )

    mapping_path = protenix_root / "common/seq_to_pdb_index.json"
    sequence_to_msa_dir = json.loads(mapping_path.read_text())
    msa_root = protenix_root / "mmcif_msa_template"
    unique_antibody_sequences = {
        entity_sequences[pdb_id][entity_id]
        for pdb_id in targets
        for entity_id in antibody_entities[pdb_id]
    }
    sequence_stats: dict[str, dict[str, Any]] = {}
    for index, sequence in enumerate(sorted(unique_antibody_sequences), start=1):
        sequence_stats[sequence] = sequence_msa_stats(
            sequence, sequence_to_msa_dir, msa_root
        )
        if index % 50 == 0 or index == len(unique_antibody_sequences):
            print(f"scanned {index}/{len(unique_antibody_sequences)} unique antibody sequences", flush=True)

    target_rows: list[dict[str, Any]] = []
    entity_rows: list[dict[str, Any]] = []
    for pdb_id in targets:
        unique_by_sequence: dict[str, list[str]] = defaultdict(list)
        for entity_id in sorted(antibody_entities[pdb_id]):
            unique_by_sequence[entity_sequences[pdb_id][entity_id]].append(entity_id)

        selected_stats: list[dict[str, Any]] = []
        for sequence, entity_ids in sorted(unique_by_sequence.items()):
            stats = sequence_stats[sequence]
            selected_stats.append(stats)
            representative = entity_ids[0]
            roles = sorted(
                set().union(*(entity_roles[pdb_id].get(entity_id, set()) for entity_id in entity_ids))
            )
            entity_rows.append(
                {
                    "pdb_id": pdb_id,
                    "entity_id": ";".join(entity_ids),
                    "label_chain_ids": ";".join(
                        sorted(
                            set().union(
                                *(set(entity_labels[pdb_id][entity_id]) for entity_id in entity_ids)
                            )
                        )
                    ),
                    "roles": ";".join(roles) if roles else "antibody",
                    "sequence_length": len(sequence),
                    **stats,
                }
            )

        pairing_depths = [
            int(stats["pairing_unique_depth"]) for stats in selected_stats
        ]
        non_pairing_depths = [
            int(stats["non_pairing_unique_depth"]) for stats in selected_stats
        ]
        combined_depths = [
            int(stats["combined_unique_depth"]) for stats in selected_stats
        ]
        pairing_homolog_count = sum(depth - 1 for depth in pairing_depths)
        non_pairing_homolog_count = sum(
            depth - 1 for depth in non_pairing_depths
        )
        homolog_count = sum(depth - 1 for depth in combined_depths)
        overlap_homolog_count = (
            pairing_homolog_count + non_pairing_homolog_count - homolog_count
        )
        target_rows.append(
            {
                "pdb_id": pdb_id,
                "subclasses": ";".join(sorted(subclasses[pdb_id])),
                "antibody_entity_ids": ";".join(sorted(antibody_entities[pdb_id])),
                "antibody_entity_count": len(antibody_entities[pdb_id]),
                "antibody_sequence_count": len(unique_by_sequence),
                "antibody_sequence_lengths": ";".join(
                    str(len(sequence)) for sequence in sorted(unique_by_sequence)
                ),
                "pairing_raw_depth_sum": sum(
                    int(stats["pairing_raw_depth"]) for stats in selected_stats
                ),
                "non_pairing_raw_depth_sum": sum(
                    int(stats["non_pairing_raw_depth"]) for stats in selected_stats
                ),
                "raw_depth_sum": sum(
                    int(stats["pairing_raw_depth"])
                    + int(stats["non_pairing_raw_depth"])
                    for stats in selected_stats
                ),
                "pairing_msa_depth": pairing_homolog_count + 1,
                "non_pairing_msa_depth": non_pairing_homolog_count + 1,
                "pair_unpair_overlap_depth": overlap_homolog_count + 1,
                "pairing_homolog_count": pairing_homolog_count,
                "non_pairing_homolog_count": non_pairing_homolog_count,
                "pair_unpair_overlap_homolog_count": overlap_homolog_count,
                "chain_combined_depth_sum": sum(combined_depths),
                "antibody_msa_depth": homolog_count + 1,
                "antibody_homolog_count": homolog_count,
                "min_chain_combined_depth": min(combined_depths),
                "max_chain_combined_depth": max(combined_depths),
            }
        )

    depths = [int(row["antibody_msa_depth"]) for row in target_rows]
    pairing_depths = [int(row["pairing_msa_depth"]) for row in target_rows]
    non_pairing_depths = [
        int(row["non_pairing_msa_depth"]) for row in target_rows
    ]
    overlap_depths = [
        int(row["pair_unpair_overlap_depth"]) for row in target_rows
    ]
    raw_depths = [int(row["raw_depth_sum"]) for row in target_rows]
    max_chain_depths = [int(row["max_chain_combined_depth"]) for row in target_rows]
    entity_occurrence_depths = [
        int(row["combined_unique_depth"]) for row in entity_rows
    ]
    unique_sequence_depths = [
        int(sequence_stats[sequence]["combined_unique_depth"])
        for sequence in unique_antibody_sequences
    ]
    raw_total_rows = sum(int(row["raw_depth_sum"]) for row in target_rows)
    deduplicated_chain_rows = sum(
        int(row["chain_combined_depth_sum"]) for row in target_rows
    )
    raw_duplicate_rows = raw_total_rows - deduplicated_chain_rows
    pairing_homolog_total = sum(
        int(row["pairing_homolog_count"]) for row in target_rows
    )
    non_pairing_homolog_total = sum(
        int(row["non_pairing_homolog_count"]) for row in target_rows
    )
    overlap_homolog_total = sum(
        int(row["pair_unpair_overlap_homolog_count"]) for row in target_rows
    )
    combined_homolog_total = sum(
        int(row["antibody_homolog_count"]) for row in target_rows
    )
    pairing_only_homolog_total = pairing_homolog_total - overlap_homolog_total
    non_pairing_only_homolog_total = (
        non_pairing_homolog_total - overlap_homolog_total
    )
    target_depth_totals = {
        "combined": sum(depths),
        "pairing_including_overlap": sum(pairing_depths),
        "non_pairing_including_overlap": sum(non_pairing_depths),
        "pair_unpair_overlap": sum(overlap_depths),
        "pairing_only": sum(pairing_depths) - sum(overlap_depths),
        "non_pairing_only": sum(non_pairing_depths) - sum(overlap_depths),
    }
    target_depth_totals["pairing_only_percent"] = (
        target_depth_totals["pairing_only"]
        / target_depth_totals["combined"]
        * 100
    )
    target_depth_totals["non_pairing_only_percent"] = (
        target_depth_totals["non_pairing_only"]
        / target_depth_totals["combined"]
        * 100
    )
    target_depth_totals["pair_unpair_overlap_percent"] = (
        target_depth_totals["pair_unpair_overlap"]
        / target_depth_totals["combined"]
        * 100
    )
    summary: dict[str, Any] = {
        "definition": "1 + sum(unique(pairing U non_pairing, lowercase insertions removed) - query)",
        "dataset_root": str(dataset_root),
        "protenix_root": str(protenix_root),
        "target_count": len(targets),
        "antibody_entity_occurrences": sum(
            int(row["antibody_entity_count"]) for row in target_rows
        ),
        "unique_antibody_sequence_count": len(unique_antibody_sequences),
        "overall": descriptive_stats(depths),
        "paired_unpaired": {
            "target_pairing_msa_depth": descriptive_stats(pairing_depths),
            "target_non_pairing_msa_depth": descriptive_stats(
                non_pairing_depths
            ),
            "target_pair_unpair_overlap_depth": descriptive_stats(
                overlap_depths
            ),
            "chain_pairing_unique_depth": descriptive_stats(
                [int(row["pairing_unique_depth"]) for row in entity_rows]
            ),
            "chain_non_pairing_unique_depth": descriptive_stats(
                [int(row["non_pairing_unique_depth"]) for row in entity_rows]
            ),
            "target_depth_totals": target_depth_totals,
            "per_target_share_percent": {
                "pairing": descriptive_stats(
                    [
                        pairing / combined * 100
                        for pairing, combined in zip(pairing_depths, depths)
                    ]
                ),
                "non_pairing": descriptive_stats(
                    [
                        non_pairing / combined * 100
                        for non_pairing, combined in zip(
                            non_pairing_depths, depths
                        )
                    ]
                ),
                "overlap": descriptive_stats(
                    [
                        overlap / combined * 100
                        for overlap, combined in zip(overlap_depths, depths)
                    ]
                ),
            },
            "homolog_evidence": {
                "combined_total": combined_homolog_total,
                "pairing_total_including_overlap": pairing_homolog_total,
                "non_pairing_total_including_overlap": non_pairing_homolog_total,
                "pairing_only": pairing_only_homolog_total,
                "non_pairing_only": non_pairing_only_homolog_total,
                "pair_unpair_overlap": overlap_homolog_total,
                "pairing_only_percent": pairing_only_homolog_total
                / combined_homolog_total
                * 100,
                "non_pairing_only_percent": non_pairing_only_homolog_total
                / combined_homolog_total
                * 100,
                "pair_unpair_overlap_percent": overlap_homolog_total
                / combined_homolog_total
                * 100,
            },
        },
        "source_depth_views": {
            "target 合并抗体": descriptive_stats(depths),
            "target raw paired+unpaired": descriptive_stats(raw_depths),
            "target 最深抗体链": descriptive_stats(max_chain_depths),
            "单条抗体序列（target occurrence）": descriptive_stats(
                entity_occurrence_depths
            ),
            "全局唯一抗体序列": descriptive_stats(unique_sequence_depths),
        },
        "raw_total_rows": raw_total_rows,
        "deduplicated_chain_rows": deduplicated_chain_rows,
        "raw_duplicate_rows": raw_duplicate_rows,
        "raw_duplicate_percent": raw_duplicate_rows / raw_total_rows * 100,
        "histogram": histogram(depths),
        "threshold_cdf": threshold_cdf(depths),
        "by_subclass": {},
        "paired_unpaired_by_subclass": {},
        "antibody_sequence_count_distribution": dict(
            sorted(Counter(int(row["antibody_sequence_count"]) for row in target_rows).items())
        ),
        "checksums": {
            "target_pdb_ids_sha256": sha256_file(dataset_root / "target_pdb_ids.txt"),
            "metadata_sha256": sha256_file(
                dataset_root / "supported_data/RecentPDB_low_homology.csv"
            ),
            "msa_mapping_sha256": sha256_file(mapping_path),
            "sabdab_summary_sha256": sha256_file(args.sabdab_summary),
        },
    }
    for label in ("HL", "H", "scFv"):
        subclass_depths = [
            int(row["antibody_msa_depth"])
            for row in target_rows
            if label in str(row["subclasses"]).split(";")
        ]
        summary["by_subclass"][label] = descriptive_stats(subclass_depths)
        subclass_rows = [
            row
            for row in target_rows
            if label in str(row["subclasses"]).split(";")
        ]
        summary["paired_unpaired_by_subclass"][label] = {
            "pairing": descriptive_stats(
                [int(row["pairing_msa_depth"]) for row in subclass_rows]
            ),
            "non_pairing": descriptive_stats(
                [int(row["non_pairing_msa_depth"]) for row in subclass_rows]
            ),
            "overlap": descriptive_stats(
                [int(row["pair_unpair_overlap_depth"]) for row in subclass_rows]
            ),
            "combined": descriptive_stats(
                [int(row["antibody_msa_depth"]) for row in subclass_rows]
            ),
        }

    write_csv(output_dir / "target_msa_depth.csv", target_rows, TARGET_FIELDS)
    write_csv(output_dir / "antibody_entity_msa_depth.csv", entity_rows, ENTITY_FIELDS)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_histogram_svg(output_dir / "msa_depth_histogram.svg", summary["histogram"])
    (output_dir / "README.md").write_text(
        build_report(summary, target_rows), encoding="utf-8"
    )
    print(f"wrote analysis to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--protenix-root", type=Path, default=DEFAULT_PROTENIX_ROOT)
    parser.add_argument("--sabdab-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
