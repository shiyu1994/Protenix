# Immune Interface Sampling Finetune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the scripts and analysis needed to run two short sampling-focused finetune experiments from `20260531_999_ema`, then evaluate oracle-best/top-k DockQ on SAb23H2.

**Architecture:** Keep training, inference, DockQ evaluation, and analysis as separate scripts so long-running GPU work can be launched and monitored independently. Reuse the existing `finetune_submit.sh`, `run_sab23h2_infer_checkpoint.sh`, and `eval_sab23h2_dockq.py` patterns, adding only small wrappers and analysis scripts.

**Tech Stack:** Bash, Python 3, Protenix conda environment, torchrun, existing Protenix training/inference scripts, DockQ CSV outputs.

---

## File Structure

- Keep existing training scripts unchanged. New sampling wrappers configure the existing
  `finetune_submit.sh` / `finetune_demo.sh` through environment variables only.
- Create `run_sab23h2_sampling_finetune.sh`: launch one sampling finetune experiment from the confirmed `20260531_999_ema` base checkpoint.
- Create `run_sab23h2_sampling_finetune_queue.sh`: launch `sampling_conservative` and `sampling_immune_heavy` sequentially with logs.
- Create `check_sab23h2_train_overlap.py`: check direct PDB ID overlap between SAb23H2 targets and the selected train index files.
- Create `analyze_sab23h2_topk.py`: compute ranking top-1, top-3, top-5, top-10, and oracle metrics from `per_model_dockq.csv`.
- Create `eval_sab23h2_dockq_with_extra_runs.py`: accept extra dynamic run mappings and delegate to `eval_sab23h2_dockq.py` without modifying it.
- Create `run_sab23h2_sampling_checkpoint_eval_queue.sh`: after training, infer/evaluate selected EMA checkpoints with the existing SAb23H2 pipeline.
- Test with existing `tests/test_sab23h2_dockq_mapping.py` and a new `tests/test_sab23h2_topk_analysis.py`.

## Task 1: Confirm Existing Checkpoint Loading Behavior

**Files:**
- Read-only: `finetune_demo.sh`
- Read-only: `configs/configs_base.py`

- [ ] **Step 1: Confirm no original script patch is required**

Check that `configs/configs_base.py` keeps parameter-only checkpoint loading as
the default:

```bash
rg -n '"load_params_only": True' configs/configs_base.py
```

Expected output:

```text
configs/configs_base.py:43:    "load_params_only": True,
```

This means the sampling wrapper can use `LOAD_CHECKPOINT_PATH` and
`LOAD_EMA_CHECKPOINT_PATH` without modifying `finetune_demo.sh`; optimizer,
scheduler, and previous step are not loaded.

- [ ] **Step 2: Confirm the wrapper passes both checkpoint paths**

Run:

```bash
DRY_RUN=1 SKIP_PATH_CHECK=1 NPROC_PER_NODE=1 \
LOAD_CHECKPOINT_PATH=/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/checkpoints/999_ema_0.999.pt \
LOAD_EMA_CHECKPOINT_PATH=/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/checkpoints/999_ema_0.999.pt \
bash finetune_demo.sh | tr ' ' '\n' | rg 'load_(checkpoint|ema|params)|skip_load|load_step'
```

Expected output contains:

```text
--load_checkpoint_path
/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/checkpoints/999_ema_0.999.pt
--load_ema_checkpoint_path
/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/checkpoints/999_ema_0.999.pt
```

## Task 2: Add Sampling Finetune Launch Scripts

**Files:**
- Create: `run_sab23h2_sampling_finetune.sh`
- Create: `run_sab23h2_sampling_finetune_queue.sh`

- [ ] **Step 1: Create `run_sab23h2_sampling_finetune.sh`**

Use this script content:

```bash
#!/usr/bin/env bash
set -euo pipefail

repo_dir="/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train"
base_checkpoint="/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/checkpoints/999_ema_0.999.pt"

experiment_name="${EXPERIMENT_NAME:?Set EXPERIMENT_NAME, e.g. sampling_conservative}"
train_sample_weights="${TRAIN_SAMPLE_WEIGHTS:?Set TRAIN_SAMPLE_WEIGHTS, e.g. 0.4,0.4,0.2}"
train_sets="${TRAIN_SETS:-weightedPDB_before210930_v20260101,sabdab-train,distill-sabdab}"
lr="${LR:-0.000003}"
max_steps="${MAX_STEPS:-2000}"
checkpoint_interval="${CHECKPOINT_INTERVAL:-250}"
master_port="${MASTER_PORT:-29500}"
timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
run_name="${RUN_NAME:-protenix_sampling_${experiment_name}_from20260531_999ema_lr${lr}_${timestamp}}"

if [ ! -f "${base_checkpoint}" ]; then
  echo "Missing base checkpoint: ${base_checkpoint}" >&2
  exit 1
fi

cd "${repo_dir}"

export LOAD_CHECKPOINT_PATH="${base_checkpoint}"
export LOAD_EMA_CHECKPOINT_PATH="${base_checkpoint}"
export LOAD_PARAMS_ONLY=true
export SKIP_LOAD_STEP=true
export SKIP_LOAD_OPTIMIZER=true
export SKIP_LOAD_SCHEDULER=true
export LOAD_STEP_FOR_SCHEDULER=false
export RUN_NAME="${run_name}"
export run_name="${run_name}"
export train_sets="${train_sets}"
export train_sample_weights="${train_sample_weights}"
export lr="${lr}"
export max_steps="${max_steps}"
export checkpoint_interval="${checkpoint_interval}"
export eval_interval=0
export EVAL_AT_END=false
export eval_at_end=false
export MASTER_PORT="${master_port}"

echo "Launching sampling finetune"
echo "  experiment_name=${experiment_name}"
echo "  run_name=${run_name}"
echo "  base_checkpoint=${base_checkpoint}"
echo "  train_sets=${train_sets}"
echo "  train_sample_weights=${train_sample_weights}"
echo "  lr=${lr}"
echo "  max_steps=${max_steps}"
echo "  checkpoint_interval=${checkpoint_interval}"
echo "  MASTER_PORT=${MASTER_PORT}"

exec bash "${repo_dir}/finetune_submit.sh"
```

- [ ] **Step 2: Create `run_sab23h2_sampling_finetune_queue.sh`**

Use this script content:

```bash
#!/usr/bin/env bash
set -euo pipefail

repo_dir="/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train"
log_dir="/mnt/project_abd/klp/proteinx_output/sab23h2_sampling_finetune_logs"
mkdir -p "${log_dir}"

declare -a runs=(
  "sampling_conservative|0.4,0.4,0.2|29651"
  "sampling_immune_heavy|0.25,0.5,0.25|29652"
)

queue_log="${log_dir}/queue_$(date +%Y%m%d_%H%M%S).log"
printf 'queue_started_at=%s\n' "$(date -Is)" | tee -a "${queue_log}"

for item in "${runs[@]}"; do
  IFS='|' read -r experiment_name weights port <<< "${item}"
  run_timestamp="$(date +%Y%m%d_%H%M%S)"
  run_log="${log_dir}/${experiment_name}_${run_timestamp}.log"
  printf '\nrun_started_at=%s experiment=%s weights=%s log=%s\n' \
    "$(date -Is)" "${experiment_name}" "${weights}" "${run_log}" | tee -a "${queue_log}"

  EXPERIMENT_NAME="${experiment_name}" \
  TRAIN_SAMPLE_WEIGHTS="${weights}" \
  MASTER_PORT="${port}" \
  RUN_TIMESTAMP="${run_timestamp}" \
  "${repo_dir}/run_sab23h2_sampling_finetune.sh" > "${run_log}" 2>&1

  printf 'run_finished_at=%s experiment=%s status=ok\n' \
    "$(date -Is)" "${experiment_name}" | tee -a "${queue_log}"
done

printf 'queue_finished_at=%s\n' "$(date -Is)" | tee -a "${queue_log}"
```

- [ ] **Step 3: Make scripts executable**

Run:

```bash
chmod +x run_sab23h2_sampling_finetune.sh run_sab23h2_sampling_finetune_queue.sh
```

- [ ] **Step 4: Run a dry-run for each experiment**

Run:

```bash
DRY_RUN=1 SKIP_PATH_CHECK=1 NPROC_PER_NODE=1 \
EXPERIMENT_NAME=sampling_conservative TRAIN_SAMPLE_WEIGHTS=0.4,0.4,0.2 \
bash run_sab23h2_sampling_finetune.sh
```

Expected output includes:

```text
experiment_name=sampling_conservative
train_sample_weights=0.4,0.4,0.2
--max_steps 2000
--checkpoint_interval 250
--lr 0.000003
```

Run:

```bash
DRY_RUN=1 SKIP_PATH_CHECK=1 NPROC_PER_NODE=1 \
EXPERIMENT_NAME=sampling_immune_heavy TRAIN_SAMPLE_WEIGHTS=0.25,0.5,0.25 \
bash run_sab23h2_sampling_finetune.sh
```

Expected output includes:

```text
experiment_name=sampling_immune_heavy
train_sample_weights=0.25,0.5,0.25
--max_steps 2000
--checkpoint_interval 250
--lr 0.000003
```

## Task 3: Add Direct PDB ID Overlap Check

**Files:**
- Create: `check_sab23h2_train_overlap.py`

- [ ] **Step 1: Create overlap checker**

Use this implementation:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


DEFAULT_INPUT_JSON = Path(
    "/mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/"
    "px_output_finetune_7999_ema_0.999_template_5seeds/input_jsons/"
    "sab23h2_final_updated.json"
)
DEFAULT_INDEX_FILES = [
    Path("/mnt/project_abd/public_databases/protenix.20260101/indices/indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz"),
    Path("/mnt/project_abd/zhe/protenix_copy/vepfs-mlp2/project-abd/zhujianwei/public_databases/protenix/indices/sabdab-train.csv"),
    Path("/mnt/project_abd/zhe/protenix_copy/vepfs-mlp2/project-abd/zhujianwei/public_databases/protenix/indices/distill-sabdab.csv"),
]


def target_pdb_ids(input_json: Path) -> set[str]:
    data = json.loads(input_json.read_text())
    return {entry["name"].split("_", 1)[0].lower() for entry in data}


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open(newline="")


def row_pdb_id(row: dict[str, str]) -> str | None:
    for key in ("pdb_id", "pdb", "entry_id", "data_id", "id", "name"):
        value = row.get(key)
        if value:
            return value.split("_", 1)[0].lower()
    values = [v for v in row.values() if v]
    if not values:
        return None
    return values[0].split("_", 1)[0].lower()


def scan_index(path: Path, targets: set[str]) -> list[str]:
    hits: list[str] = []
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            for row in reader:
                pdb_id = row_pdb_id(row)
                if pdb_id in targets:
                    hits.append(pdb_id)
    return sorted(set(hits))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--indices", nargs="*", type=Path, default=DEFAULT_INDEX_FILES)
    args = parser.parse_args()

    targets = target_pdb_ids(args.input_json)
    print(f"targets={len(targets)} input_json={args.input_json}")
    total_hits: dict[str, list[str]] = {}
    for path in args.indices:
        hits = scan_index(path, targets)
        total_hits[str(path)] = hits
        print(f"index={path} hits={len(hits)}")
        if hits:
            print(",".join(hits))
    return 1 if any(total_hits.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run direct overlap check**

Run:

```bash
python check_sab23h2_train_overlap.py
```

Expected result for a clean direct-PDB check:

```text
targets=87 input_json=...
index=... hits=0
index=... hits=0
index=... hits=0
```

If the command exits `1`, inspect the printed PDB IDs before launching training.

## Task 4: Add Top-k DockQ Analysis

**Files:**
- Create: `analyze_sab23h2_topk.py`
- Create: `tests/test_sab23h2_topk_analysis.py`

- [ ] **Step 1: Create top-k analysis functions and CLI**

Implement `analyze_sab23h2_topk.py` with functions:

```python
def topk_success(rows: list[dict[str, object]], k: int, threshold: float) -> bool:
    ranked = sorted(rows, key=lambda r: float(r["ranking_score"]), reverse=True)
    return any(float(r["immune_other_dockq"]) > threshold for r in ranked[:k])


def topk_best_dockq(rows: list[dict[str, object]], k: int) -> float:
    ranked = sorted(rows, key=lambda r: float(r["ranking_score"]), reverse=True)
    return max(float(r["immune_other_dockq"]) for r in ranked[:k])
```

The CLI must:

- Read one or more `per_model_dockq.csv` files.
- Keep rows with `dockq_status == "OK"` and numeric `immune_other_dockq`.
- Group by `(run_label, target)`.
- Require exactly 25 scoreable models per group by default.
- Write `topk_summary.csv` and `topk_per_target.csv`.
- Compute k values `1,3,5,10,25`; k=25 is oracle over all samples.

- [ ] **Step 2: Add unit tests**

Create `tests/test_sab23h2_topk_analysis.py` with:

```python
import unittest

from analyze_sab23h2_topk import topk_best_dockq, topk_success


class TopKAnalysisTest(unittest.TestCase):
    def test_topk_success_uses_ranking_order(self):
        rows = [
            {"ranking_score": 0.9, "immune_other_dockq": 0.1},
            {"ranking_score": 0.8, "immune_other_dockq": 0.2},
            {"ranking_score": 0.7, "immune_other_dockq": 0.5},
        ]
        self.assertFalse(topk_success(rows, k=2, threshold=0.23))
        self.assertTrue(topk_success(rows, k=3, threshold=0.23))

    def test_topk_best_dockq_returns_best_within_topk(self):
        rows = [
            {"ranking_score": 0.9, "immune_other_dockq": 0.1},
            {"ranking_score": 0.8, "immune_other_dockq": 0.4},
            {"ranking_score": 0.7, "immune_other_dockq": 0.8},
        ]
        self.assertAlmostEqual(topk_best_dockq(rows, k=2), 0.4)
        self.assertAlmostEqual(topk_best_dockq(rows, k=3), 0.8)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests**

Run:

```bash
python -m unittest tests/test_sab23h2_topk_analysis.py
```

Expected output:

```text
Ran 2 tests
OK
```

- [ ] **Step 4: Run top-k analysis on existing eval outputs**

Run:

```bash
python analyze_sab23h2_topk.py \
  --per-model-csv \
    /mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/dockq_eval_immune_interfaces/per_model_dockq.csv \
    /mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/dockq_eval_immune_interfaces_20260525_999/per_model_dockq.csv \
    /mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/dockq_eval_immune_interfaces_20260531/per_model_dockq.csv \
    /mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/dockq_eval_immune_interfaces_raw999/per_model_dockq.csv \
  --out-dir /mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/topk_dockq_analysis
```

Expected files:

```text
/mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/topk_dockq_analysis/topk_summary.csv
/mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/topk_dockq_analysis/topk_per_target.csv
```

## Task 5: Add Dynamic Run Mapping Wrapper For DockQ Eval

**Files:**
- Create: `eval_sab23h2_dockq_with_extra_runs.py`

- [ ] **Step 1: Create wrapper parser**

The wrapper must parse only `--extra-runs-json`, then pass remaining arguments to
the original `eval_sab23h2_dockq.py`:

```python
def parse_wrapper_args(argv: list[str]) -> tuple[Path | None, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--extra-runs-json", type=Path, default=None)
    args, remaining = parser.parse_known_args(argv)
    return args.extra_runs_json, remaining
```

- [ ] **Step 2: Load JSON mapping and append to imported RUNS**

Use this logic:

```python
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
```

- [ ] **Step 3: Delegate to original main**

```python
def main() -> int:
    extra_runs_json, remaining = parse_wrapper_args(sys.argv[1:])
    import eval_sab23h2_dockq as dockq_eval
    dockq_eval.RUNS.extend(load_extra_runs(extra_runs_json))
    sys.argv = [sys.argv[0], *remaining]
    return dockq_eval.main()
```

- [ ] **Step 4: Dry-run dynamic run mapping**

Run:

```bash
tmp_json=/tmp/sab23h2_extra_runs.json
printf '{"tmp_raw999":"px_output_finetune_20260531_999_raw_template_5seeds"}\n' > "${tmp_json}"
python eval_sab23h2_dockq.py \
  --extra-runs-json "${tmp_json}" \
  --runs tmp_raw999 \
  --out-dir /tmp/sab23h2_dockq_dynamic_smoke \
  --limit 1 \
  --dry-run
```

Expected output:

```text
Jobs: 1
Targets with chain mappings: 1
Output: /tmp/sab23h2_dockq_dynamic_smoke
```

## Task 6: Add Sampling Checkpoint Inference/Eval Queue

**Files:**
- Create: `run_sab23h2_sampling_checkpoint_eval_queue.sh`

- [ ] **Step 1: Create queue script**

The script must accept `TRAIN_RUN_DIRS` and evaluate EMA checkpoints at:

```text
250,500,750,1000,1250,1500,1750,2000
```

For each checkpoint path:

```text
${train_run_dir}/checkpoints/${step}_ema_0.999.pt
```

It should set:

```bash
RUN_NAME="px_output_${experiment_name}_${step}_ema_0.999_template_5seeds"
CHECKPOINT_PATH="${checkpoint_path}"
MASTER_PORT="${port}"
bash run_sab23h2_infer_checkpoint.sh
```

After inference for all selected checkpoints, run:

```bash
/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python eval_sab23h2_dockq.py \
  --extra-runs-json "${extra_runs_json}" \
  --runs "${comma_separated_new_run_labels}" \
  --out-dir "/mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/dockq_eval_immune_interfaces_sampling_phase_a" \
  --num-workers 32
```

Then run:

```bash
python analyze_sab23h2_ranking_failure.py
python analyze_sab23h2_topk.py \
  --per-model-csv \
    /mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/dockq_eval_immune_interfaces_sampling_phase_a/per_model_dockq.csv \
  --out-dir /mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/topk_dockq_analysis_sampling_phase_a
```

- [ ] **Step 2: Add dry-run mode**

When `DRY_RUN=1`, print the planned checkpoint paths, output run labels, and
commands without launching inference or DockQ.

- [ ] **Step 3: Dry-run against a known existing training directory**

Run:

```bash
DRY_RUN=1 TRAIN_RUN_DIRS=/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805 \
bash run_sab23h2_sampling_checkpoint_eval_queue.sh
```

Expected output includes checkpoint paths for existing steps:

```text
999_ema_0.999.pt
2999_ema_0.999.pt
4999_ema_0.999.pt
```

The script must skip missing requested dense steps in dry-run and real-run mode
with an explicit `skip_missing_checkpoint` line.

## Task 7: Verification And Launch Gate

**Files:**
- Verify all files above.

- [ ] **Step 1: Syntax checks**

Run:

```bash
python -m py_compile \
  check_sab23h2_train_overlap.py \
  analyze_sab23h2_topk.py \
  eval_sab23h2_dockq.py \
  analyze_sab23h2_ranking_failure.py
```

Expected: exit code 0.

- [ ] **Step 2: Unit tests**

Run:

```bash
/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python -m unittest \
  tests/test_sab23h2_dockq_mapping.py \
  tests/test_sab23h2_topk_analysis.py
```

Expected:

```text
Ran 5 tests
OK
```

- [ ] **Step 3: Direct overlap gate**

Run:

```bash
python check_sab23h2_train_overlap.py
```

Expected: exit code 0 with `hits=0` for all configured train index files.

- [ ] **Step 4: Training dry-run gate**

Run both dry-runs from Task 2. Do not launch real training until both dry-runs
show the correct base checkpoint, sampling weights, LR, max steps, and checkpoint
interval.

- [ ] **Step 5: Launch training only after gates pass**

Run the two-experiment queue in tmux:

```bash
tmux new-session -d -s sab23h2_sampling_finetune_queue \
  'cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train && ./run_sab23h2_sampling_finetune_queue.sh'
```

Monitor:

```bash
tmux capture-pane -pt sab23h2_sampling_finetune_queue:0 -S -80
```

Expected first lines include:

```text
queue_started_at=
run_started_at=... experiment=sampling_conservative
Launching sampling finetune
```

Do not start checkpoint inference until the training run directories contain
the expected dense EMA checkpoints.
