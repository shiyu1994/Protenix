#!/usr/bin/env python3
"""Check direct PDB ID overlap between SAb23H2 and training index files."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import TextIO


DEFAULT_INPUT_JSON = Path(
    "/mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/"
    "px_output_finetune_7999_ema_0.999_template_5seeds/input_jsons/"
    "sab23h2_final_updated.json"
)
DEFAULT_INDEX_FILES = [
    Path(
        "/mnt/project_abd/public_databases/protenix.20260101/indices/"
        "indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz"
    ),
    Path(
        "/mnt/project_abd/zhe/protenix_copy/vepfs-mlp2/project-abd/"
        "zhujianwei/public_databases/protenix/indices/sabdab-train.csv"
    ),
    Path(
        "/mnt/project_abd/zhe/protenix_copy/vepfs-mlp2/project-abd/"
        "zhujianwei/public_databases/protenix/indices/distill-sabdab.csv"
    ),
]


def target_names_and_pdb_ids(input_json: Path) -> tuple[list[str], set[str]]:
    data = json.loads(input_json.read_text())
    names = [entry["name"] for entry in data]
    pdb_ids = {name.split("_", 1)[0].lower() for name in names}
    return names, pdb_ids


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open(newline="")


def row_pdb_id(row: dict[str, str]) -> str | None:
    for key in ("pdb_id", "pdb", "entry_id", "data_id", "id", "name"):
        value = row.get(key)
        if value:
            return value.split("_", 1)[0].lower()
    values = [value for value in row.values() if value]
    if not values:
        return None
    return values[0].split("_", 1)[0].lower()


def scan_index(path: Path, targets: set[str]) -> list[str]:
    hits: list[str] = []
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pdb_id = row_pdb_id(row)
            if pdb_id in targets:
                hits.append(pdb_id)
    return sorted(set(hits))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--indices", nargs="*", type=Path, default=DEFAULT_INDEX_FILES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_names, targets = target_names_and_pdb_ids(args.input_json)
    print(
        f"target_entries={len(target_names)} unique_pdb_ids={len(targets)} "
        f"input_json={args.input_json}"
    )
    any_hits = False
    for path in args.indices:
        if not path.exists():
            raise FileNotFoundError(f"Missing index file: {path}")
        hits = scan_index(path, targets)
        any_hits = any_hits or bool(hits)
        print(f"index={path} hits={len(hits)}")
        if hits:
            print(",".join(hits))
    return 1 if any_hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
