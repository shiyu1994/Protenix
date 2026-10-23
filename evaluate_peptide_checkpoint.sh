#!/usr/bin/env bash
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
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash evaluate_peptide_checkpoint.sh [all|train|test] [options]

Evaluate one saved checkpoint with receptor MSAs. Default: evaluate 50 training
structures against the official checkpoint, then evaluate the 51 test structures.
The two evaluations run sequentially on the requested GPUs. Training is untouched.

Options:
  --training-run DIR   Training run containing config.yaml and checkpoints/
  --checkpoint FILE    Weights to evaluate (default: latest completed checkpoint)
  --output-dir DIR     Output root; reuse it to resume the same evaluation
  --msa-map FILE       Test receptor MSA map (default: existing peptide MSA map)
  --gpus IDS           Physical GPUs (default: 0,1,2,3,4,5,6,7)
  --count N            Distinct training structures (default: 50)
  --seeds LIST         Comma-separated prediction seeds (default: 101)
  --samples N          Predictions per seed per structure (default: 5)
  --steps N            Diffusion steps (default: 200)
  --cycles N           Recycles (default: 10)
  --chunk-size N       Fixed inference chunk size (default: 32)
  --memory-fraction F  Evaluation CUDA allocator fraction (default: 1.0)
  --dry-run            Print resolved commands; do not prepare or evaluate
  -h, --help           Show this help

Training receptor MSAs follow config.yaml (512 rows in the current run).
Test MSAs use --msa-map. Templates stay disabled. Set --memory-fraction 0.9
when running on otherwise idle GPUs. CUDA context allocations are additional.

The latest checkpoint is selected once when you invoke this script. Resuming an
existing output folder retains its checkpoint. Use a new folder for new weights.
Use a new folder when changing seeds or samples. Top1 and best RMSD are selected
across all seeds: --seeds 101,102,103,104,105 --samples 5 gives 25 predictions.
Outputs: train_subset/comparison.md and test51/scores/summary.json.
HELP
}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/mnt/project_abd/public_databases/protenix.20260101}"
export PYTHONDONTWRITEBYTECODE=1
stage=all
training_run="${TRAINING_RUN:-/mnt/project_abd/shiyu/peptide_finetune_output_1024_offload/peptide_ligand_v2_finetune_20260930_222609}"
checkpoint=""
output_dir=""
msa_map="${TEST_MSA_MAP:-/root/shiyu/peptide_pose_eval_v2_msa_map.json}"
gpus="${GPUS:-0,1,2,3,4,5,6,7}"
count=50
seeds=101
samples=5
steps=200
cycles=10
chunk_size=32
memory_fraction="${PROTENIX_EVAL_GPU_MEMORY_FRACTION:-1.0}"
dry_run=0
if [[ ${1:-} == all || ${1:-} == train || ${1:-} == test ]]; then stage=$1; shift; fi
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=1; shift ;;
        --training-run|--checkpoint|--output-dir|--msa-map|--gpus|--count|--seeds|--samples|--steps|--cycles|--chunk-size|--memory-fraction)
            if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                --training-run) training_run=$2 ;;
                --checkpoint) checkpoint=$2 ;;
                --output-dir) output_dir=$2 ;;
                --msa-map) msa_map=$2 ;;
                --gpus) gpus=$2 ;;
                --count) count=$2 ;;
                --seeds) seeds=$2 ;;
                --samples) samples=$2 ;;
                --steps) steps=$2 ;;
                --cycles) cycles=$2 ;;
                --chunk-size) chunk_size=$2 ;;
                --memory-fraction) memory_fraction=$2 ;;
            esac
            shift 2 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
for value in "$count" "$samples" "$steps" "$cycles" "$chunk_size"; do
    if [[ ! $value =~ ^[1-9][0-9]*$ ]]; then echo "Counts and sampling settings must be positive integers" >&2; exit 2; fi
done

# Resolve paths and a completed checkpoint using CPU-only metadata reads.
resolved="$("$python_bin" -B - "$training_run" "$checkpoint" "$output_dir" "$msa_map" "$gpus" "$memory_fraction" "$count" "$seeds" "$samples" <<'PY'
import datetime
import json
import os
from pathlib import Path
import sys
import yaml

run, checkpoint, output, msa_map, gpus, fraction, count, seeds, samples = sys.argv[1:]
try:
    seeds = [int(value) for value in seeds.split(",")]
except ValueError:
    raise SystemExit("--seeds must be a nonempty comma-separated list of integers")
if len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= 2**32 for seed in seeds):
    raise SystemExit("--seeds must be unique integers in [0, 2**32)")
run = Path(run).resolve(strict=True)
config = yaml.safe_load((run / "config.yaml").read_text())
if not config["data"]["msa"]["enable_prot_msa"]:
    raise SystemExit("The training configuration must enable protein MSAs for this launcher")
gpu_ids = gpus.split(",")
if not all(gpu_ids) or len(gpu_ids) != len(set(gpu_ids)):
    raise SystemExit("--gpus must list distinct GPU IDs")
if not 0 < float(fraction) <= 1:
    raise SystemExit("--memory-fraction must be in (0, 1]")
msa_map = Path(msa_map).resolve(strict=True)
prior = []
if output:
    root = Path(output).resolve()
    train_manifest = root / "train_subset/manifest.json"
    test_config = root / "test51/run_config.json"
    if train_manifest.exists():
        manifest = json.loads(train_manifest.read_text())
        if Path(manifest["training_config"]).resolve() != run / "config.yaml":
            raise SystemExit("Output folder belongs to a different training run")
        if manifest["selection"]["count"] != int(count):
            raise SystemExit("Training subset count differs; use a new output folder")
        prior.append(manifest["after_checkpoint"])
    if test_config.exists():
        prior.append(json.loads(test_config.read_text())["checkpoint"])
    for saved_config in (root / "train_subset/run_config.json", test_config):
        if saved_config.exists():
            saved = json.loads(saved_config.read_text())
            saved_seeds = saved.get("seeds", [saved.get("seed", 101)])
            if saved_seeds != seeds or saved["samples"] != int(samples):
                raise SystemExit("Prediction seeds or samples differ; use a new output folder")
prior = {str(Path(p).resolve()) for p in prior if p}
if len(prior) > 1:
    raise SystemExit("Existing training and test evaluations use different checkpoints")
if checkpoint:
    checkpoint = Path(checkpoint).resolve(strict=True)
    if prior and str(checkpoint) not in prior:
        raise SystemExit("Output folder uses different weights; choose a new output folder")
elif prior:
    checkpoint = Path(prior.pop()).resolve(strict=True)
else:
    raw = (run / "checkpoints/checkpoints.jsonl").read_bytes()
    records = [json.loads(line) for line in raw[:raw.rfind(b"\n") + 1].splitlines() if line.strip()]
    ready = [r for r in records if not r.get("is_ema", False)
             and Path(r["path"]).is_file() and Path(r["path"]).stat().st_size > 0]
    if not ready:
        raise SystemExit("No completed non-EMA checkpoint found")
    checkpoint = Path(max(ready, key=lambda r: r["step"])["path"]).resolve()
if not output:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = run.parent / f"peptide_eval_{stamp}_step{checkpoint.stem}"
before = Path(os.environ["PROTENIX_ROOT_DIR"]) / "checkpoint/protenix-v2.pt"
before = before.resolve(strict=True)
for path in (run, checkpoint, root, msa_map, before):
    if any(c.isspace() for c in str(path)):
        raise SystemExit("Evaluation paths must not contain whitespace")
    print(path)
print(",".join(map(str, seeds)))
print(len(seeds) * int(samples))
PY
)"
mapfile -t paths <<< "$resolved"
training_run=${paths[0]}
checkpoint=${paths[1]}
output_dir=${paths[2]}
msa_map=${paths[3]}
before=${paths[4]}
seeds=${paths[5]}
export PROTENIX_EVAL_GPU_MEMORY_FRACTION="$memory_fraction"
printf 'Checkpoint: %s\nOutput: %s\nGPUs: %s\n' "$checkpoint" "$output_dir" "$gpus"
printf 'Seeds: %s\nSamples per seed: %s\nPredictions per complex per checkpoint: %s\n' "$seeds" "$samples" "${paths[6]}"

run_command() {
    printf 'Command: '; printf '%q ' "$@"; printf '\n'
    if [[ $dry_run == 1 ]]; then return 0; fi
    "$@"
}
common=(--seeds "$seeds" --samples "$samples" --steps "$steps" --cycles "$cycles" --chunk-size "$chunk_size")

evaluate_training() {
    local destination="$output_dir/train_subset"
    if [[ ! -f $destination/manifest.json ]]; then
        run_command bash "$repo_dir/evaluate_peptide_training.sh" prepare \
            --training-config "$training_run/config.yaml" --output-dir "$destination" \
            --after "$checkpoint" --count "$count" --selection-seed 20260923 || return
    fi
    run_command bash "$repo_dir/evaluate_peptide_training.sh" launch \
        --training-config "$training_run/config.yaml" --output-dir "$destination" \
        --before "$before" --after "$checkpoint" --gpus "$gpus" "${common[@]}" || return
    if [[ $dry_run != 1 ]]; then
        "$python_bin" -B - "$destination/exit_status.json" <<'PY'
import json
import sys
from pathlib import Path
status = json.loads(Path(sys.argv[1]).read_text())
if not status["complete"] or any(status["worker_exitcodes"]):
    raise SystemExit("Training-subset evaluation incomplete; inspect worker logs and summary.json")
PY
    fi
}

evaluate_test() {
    run_command bash "$repo_dir/evaluate_peptide.sh" all \
        --checkpoint "$checkpoint" --msa-map "$msa_map" \
        --output-dir "$output_dir/test51" --gpus "$gpus" "${common[@]}"
}

status=0
if [[ $stage == all || $stage == train ]]; then evaluate_training || status=1; fi
if [[ $stage == all || $stage == test ]]; then evaluate_test || status=1; fi
exit "$status"
