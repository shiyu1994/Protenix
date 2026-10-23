#!/usr/bin/env python3
"""Run eval_sab23h2_dockq.py with extra run_label -> run_dir mappings.

This wrapper avoids modifying eval_sab23h2_dockq.py when evaluating new
checkpoint output directories.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_wrapper_args(argv: list[str]) -> tuple[Path | None, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--extra-runs-json", type=Path, default=None)
    args, remaining = parser.parse_known_args(argv)
    return args.extra_runs_json, remaining


def load_extra_runs(path: Path | None) -> list[tuple[str, str]]:
    if path is None:
        return []
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"extra-runs-json must be a JSON object: {path}")
    runs: list[tuple[str, str]] = []
    for label, run_dir in data.items():
        if not isinstance(label, str) or not isinstance(run_dir, str):
            raise ValueError("extra-runs-json keys and values must be strings")
        runs.append((label, run_dir))
    return runs


def main() -> int:
    extra_runs_json, remaining = parse_wrapper_args(sys.argv[1:])

    import eval_sab23h2_dockq as dockq_eval

    dockq_eval.RUNS.extend(load_extra_runs(extra_runs_json))
    sys.argv = [sys.argv[0], *remaining]
    return dockq_eval.main()


if __name__ == "__main__":
    raise SystemExit(main())
