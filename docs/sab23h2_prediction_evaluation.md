# SAb23H2 Prediction Evaluation

This document records the local workflow used to evaluate Protenix SAb23H2
predictions with DockQ. The main metric is immune-protein-versus-other-protein
interface DockQ, not whole-complex DockQ.

## Environment

Run DockQ evaluation with the Protenix conda environment. The system Python does
not include all dependencies.

```bash
PYTHON=/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python
REPO=/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train
BENCH=/mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2
NATIVE=/mnt/project_abd/klp/benchmark/SAb23H2/cif.files.native
```

The default DockQ binary is:

```bash
/mnt/project_abd/klp/miniconda3/envs/protenix/bin/DockQ
```

## Inputs

The default input JSON is reused from the earlier 7999 EMA run:

```bash
${BENCH}/px_output_finetune_7999_ema_0.999_template_5seeds/input_jsons/sab23h2_final_updated.json
```

Native reference structures are read from:

```bash
${NATIVE}
```

Each prediction run is expected under:

```bash
${BENCH}/px_output_<run_label>_0.999_template_5seeds
```

Each target should have 25 predictions: 5 seeds times 5 samples.

## One-Step Checkpoint Evaluation

Use this when checkpoints still need prediction, DockQ scoring, and top-k
analysis.

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train

TRAIN_RUN_DIRS='conservative=/mnt/project_abd/klp/proteinx_output/protenix_sampling_sampling_conservative_from20260531_999ema_lr0.000003_20260602_083403_20260602_083442,immune_heavy=/mnt/project_abd/klp/proteinx_output/protenix_sampling_sampling_immune_heavy_from20260531_999ema_lr0.000003_20260603_004205_20260603_004232' \
bash run_sab23h2_sampling_checkpoint_eval_queue.sh
```

The default checkpoint steps are zero-based saved step names:

```bash
249,499,749,999,1249,1499,1749,1999
```

Do not use `250,500,...` unless those files actually exist.

For a dry run:

```bash
DRY_RUN=1 TRAIN_RUN_DIRS='label=/path/to/train_run' \
bash run_sab23h2_sampling_checkpoint_eval_queue.sh
```

For long runs, start the queue in `tmux` so that `torchrun` keeps its parent
process:

```bash
tmux new-session -d -s sab23h2_sampling_eval \
  "cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train && \
   TRAIN_RUN_DIRS='label=/path/to/train_run' bash run_sab23h2_sampling_checkpoint_eval_queue.sh"
```

## DockQ-Only Evaluation

Use the core script when predictions already exist and the run labels are known
by `eval_sab23h2_dockq.py`.

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train

/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python eval_sab23h2_dockq.py \
  --out-dir /mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/dockq_eval_immune_interfaces \
  --num-workers 32
```

Use the wrapper when evaluating newly generated run labels without modifying
`eval_sab23h2_dockq.py`.

Create a JSON file mapping run labels to prediction directories:

```json
{
  "conservative_999_ema": "px_output_conservative_999_ema_0.999_template_5seeds",
  "immune_heavy_999_ema": "px_output_immune_heavy_999_ema_0.999_template_5seeds"
}
```

Then run:

```bash
/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python eval_sab23h2_dockq_with_extra_runs.py \
  --extra-runs-json /path/to/extra_runs.json \
  --runs conservative_999_ema,immune_heavy_999_ema \
  --out-dir /mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/dockq_eval_immune_interfaces_my_eval \
  --num-workers 32
```

Useful flags:

- `--dry-run`: list jobs without running DockQ.
- `--limit N`: run only the first `N` jobs for a smoke test.
- `--force`: recompute DockQ even if raw DockQ JSON already exists.
- `--timeout SEC`: change per-model DockQ timeout.

## Top-k and Oracle Analysis

After DockQ produces `per_model_dockq.csv`, compute ranking top-k and oracle
best summaries:

```bash
python analyze_sab23h2_topk.py \
  --per-model-csv /mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/dockq_eval_immune_interfaces_sampling_phase_a/per_model_dockq.csv \
  --out-dir /mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/topk_dockq_analysis_sampling_phase_a
```

Defaults:

- k values: `1,3,5,10,25`
- success threshold: DockQ `> 0.23`
- expected models per target/run: `25`

`k=1` is the model selected by Protenix `ranking_score`.
`k=25` is the oracle best among all generated samples for that target/run.

## Ranking Failure Analysis

Use this to quantify the gap between ranking top-1 and oracle best, and to test
simple confidence reranking with `ptm`, `iptm`, and chain-pair ipTM features.

```bash
python analyze_sab23h2_ranking_failure.py \
  --eval-dirs /mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/dockq_eval_immune_interfaces_sampling_phase_a \
  --out-dir /mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/ranking_failure_reranking_analysis_sampling_phase_a
```

The default success threshold is DockQ `> 0.23`.

## Chain Mapping Logic

Target names use:

```text
pdb_id_immune_chain_immune_chain_or_NA_other_protein_chain
```

The DockQ script does not assume predicted chain IDs match native chain IDs. It
maps sequences from the input JSON to native CIF chains and model CIF chains,
then checks DockQ's model-to-native mapping before extracting immune-other
interfaces.

Mapping-related columns to inspect:

- `mapping_status`
- `mapping_warnings`
- `native_mapping_warnings`
- `dockq_mapping_checks`
- `immune_native_chains`
- `other_native_chains`
- `immune_model_chains`
- `other_model_chains`
- `best_mapping_str`

For strict reporting, prefer the `mapping_ok` rows in `topk_summary.csv`. For
broader reporting, use `all_scoreable`.

## Output Files

DockQ output directory:

```text
dockq_eval_immune_interfaces_<name>/
```

Important files:

- `per_model_dockq.csv`: one row per predicted model.
- `checkpoint_summary.csv`: one row per run/checkpoint.
- `per_target_checkpoint_summary.csv`: one row per run/checkpoint/target.
- `chain_mapping.csv`: sequence mapping and chain-role diagnostics.
- `run_metadata.json`: run configuration and paths.
- `raw_dockq_json/`: cached raw DockQ outputs.

Top-k output directory:

```text
topk_dockq_analysis_<name>/
```

Important files:

- `topk_summary.csv`: per-run summary for k values.
- `topk_per_target.csv`: per-target top-k and oracle values.

Key `per_model_dockq.csv` columns:

- `run_label`, `target`, `seed`, `sample`
- `ranking_score`, `ptm`, `iptm`
- `dockq_status`, `dockq_error`
- `full_dockq`
- `immune_other_dockq`
- `immune_other_irmsd`, `immune_other_lrmsd`, `immune_other_fnat`
- `n_immune_other_interfaces`, `immune_other_interfaces`

Key `topk_summary.csv` columns:

- `filter`: `all_scoreable` or `mapping_ok`
- `run_label`
- `k`
- `n_targets`
- `topk_best_mean_dockq`
- `topk_success_rate`
- `oracle_best_mean_dockq`
- `oracle_success_rate`
- `mean_gap_to_oracle`

## Current Sampling Phase A Results

The completed sampling phase A run produced:

```text
/mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/dockq_eval_immune_interfaces_sampling_phase_a
/mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2/topk_dockq_analysis_sampling_phase_a
```

Run status:

- 16 evaluated runs: conservative and immune-heavy, each at
  `249/499/749/999/1249/1499/1749/1999` EMA checkpoints.
- `per_model_dockq.csv`: 34,800 rows.
- scoreable OK models: 34,400.
- DockQ errors: 0.
- each prediction output directory contains 4,351 files.

## Troubleshooting

If `ModuleNotFoundError: No module named 'Bio'` appears, the wrong Python was
used. Run DockQ evaluation with:

```bash
/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python
```

If a queue reports only `skip_missing_checkpoint`, check the saved checkpoint
filenames. The dense finetune runs saved `249_ema_0.999.pt`, not
`250_ema_0.999.pt`.

If `torchrun` workers lose their parent process or logs show repeated
`TCPStore` broken-pipe messages, stop the orphan workers and restart inside
`tmux`.

If `n_no_immune_other_interface` is nonzero, the target/model had no DockQ
interface matching immune-vs-other native chain roles. These rows are excluded
from scoreable top-k summaries.
