# Immune Interface Finetune Plan

## Context

We are optimizing Protenix-v2 for immune protein interfaces on the SAb23H2
benchmark. The benchmark evaluates antibody/TCR-to-other-protein interfaces with
DockQ against native CIF structures. Each target is predicted with 5 seeds and 5
samples per seed, giving 25 candidate structures per checkpoint.

Recent SAb23H2 results show:

- `20260531_999_ema` has the strongest current oracle-best sampling result:
  oracle mean DockQ 0.544 and DockQ > 0.23 oracle success 70.9%.
- `20260525_999_ema` is also better than original for oracle-best sampling:
  oracle mean DockQ 0.544 and oracle success 68.6%.
- Raw 999 checkpoints are worse than EMA in both 20260525 and 20260531 runs,
  so EMA improvement is not an accidental checkpoint artifact.
- Ranking top-1 still lags oracle best by about 0.10 DockQ for the 999 EMA
  checkpoints. This means the model often samples a useful structure but does
  not select it as rank 1.

The agreed strategy is sequential:

1. Improve oracle-best sampling first.
2. After a stronger sampling checkpoint is found, improve ranking top-1 /
   confidence selection.

## Phase A: Sampling Finetune

### Objective

Increase the probability that the 25 generated candidates contain a correct
immune-other interface. The primary metric is oracle-best DockQ, not ranking
top-1.

### Base Checkpoint

Use `20260531_999_ema` as the preferred base checkpoint because it currently has
the best oracle success rate on SAb23H2.

Checkpoint path:

```text
/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/checkpoints/999_ema_0.999.pt
```

### Experiments

Run two short sampling experiments with identical training settings except for
dataset sampling weights.

| Experiment | weightedPDB | sabdab-train | distill-sabdab | Purpose |
| --- | ---: | ---: | ---: | --- |
| `sampling_conservative` | 0.4 | 0.4 | 0.2 | Increase immune-interface exposure while retaining general structure ability. |
| `sampling_immune_heavy` | 0.25 | 0.5 | 0.25 | Push immune-interface sampling harder and test whether oracle best improves. |

### Training Schedule

Use short runs with dense checkpointing because previous runs peaked near step
999 and later checkpoints did not clearly improve oracle best.

Recommended schedule:

- LR: `3e-6` initially; `5e-6` is the fallback if training is too conservative.
- Max steps: `2000`.
- Checkpoint interval: `250`.
- Save and evaluate EMA checkpoints.
- Also keep raw checkpoints for selected steps if available, but raw is
  secondary after current raw-vs-EMA comparison.
- Keep `EVAL_AT_END=false` and `eval_interval=0` during training. Benchmark
  evaluation remains decoupled through inference + DockQ scripts.

Evaluation checkpoints:

```text
250, 500, 750, 1000, 1250, 1500, 1750, 2000
```

### Evaluation

For each checkpoint:

- Run SAb23H2 inference on the 87 targets.
- Use the same 5 seeds:
  `31415,31416,31417,31418,31419`.
- Use `N_sample=5`.
- Use the same input JSON and template/MSA configuration as prior runs.
- Evaluate immune-other interfaces with DockQ.

Primary metrics:

- Oracle-best mean DockQ.
- Oracle-best DockQ > 0.23 success rate.
- Top-5 and top-10 success rates.
- Number of targets where any of 25 samples succeeds.

Secondary metrics:

- Ranking top-1 mean DockQ.
- Ranking top-1 DockQ > 0.23 success rate.
- Ranking top-1 vs oracle-best mean gap.
- Number of ranking-failed but oracle-succeeded targets.

### Success Criteria

A sampling experiment is useful if it improves at least one of:

- Oracle-best success above 70.9%.
- Oracle-best mean DockQ above 0.544.
- Top-5 success close to oracle success.

It should not be selected as the ranking/confidence base if:

- Oracle best improves only marginally but ranking gap grows substantially.
- Many targets regress from success to failure compared with `20260531_999_ema`.
- The improvement is driven by obvious benchmark leakage.

## Phase B: Ranking / Confidence Improvement

### Entry Condition

Start Phase B only after Phase A identifies a checkpoint with stronger or
equally strong oracle-best sampling than `20260531_999_ema`.

### Objective

Reduce the gap between ranking top-1 and oracle best. This phase addresses
selection, not generation.

### First Step: Post-hoc Reranker

Before changing model training, build a target-level cross-validated reranker
using existing prediction features:

- `ranking_score`
- `iptm`
- `ptm`
- `chain_iptm_mean/min`
- immune-other `chain_pair_iptm_mean/min/max`
- interface contact count
- interface residue pLDDT if atom confidence is available
- interface PAE/PDE if full confidence output is available
- clash or chain-overlap proxy

The split must be by target, not by sample, to avoid leakage between the 25
samples of the same target.

### Second Step: Confidence Finetune

If the reranker shows a clear learnable signal, run confidence-focused finetune:

- Freeze or mostly freeze the structure-generation backbone.
- Train confidence/ranking-related heads with interface-aware labels.
- Use immune-other interface metrics rather than whole-complex averages.
- Keep SAb23H2 as final evaluation only; do not use its DockQ labels for
  training.

## Leakage And Controls

SAb23H2 must remain a held-out benchmark. Before launching new finetune runs:

- Check train indices for direct PDB ID overlap with SAb23H2 targets.
- Check antigen and immune-chain sequence similarity where practical.
- Record all data roots, index files, sampling weights, base checkpoints, LR,
  max steps, and checkpoint intervals.
- Keep original Protenix-v2 and existing 999 EMA runs as fixed baselines.

## Implementation Tasks

1. Add scripts to launch `sampling_conservative` and `sampling_immune_heavy`
   from `20260531_999_ema`.
2. Add path checks for the selected EMA base checkpoint.
3. Set dense checkpointing at 250-step intervals.
4. Add a checkpoint inference queue for the new sampling experiments.
5. Add top-k DockQ analysis over the 25 samples per target.
6. Reuse existing DockQ immune-interface evaluation for all checkpoints.
7. Generate a compact comparison table with original, current finetunes, new
   sampling checkpoints, top-k, oracle, and ranking-gap metrics.

## Decision Rule

Select the Phase A winner by oracle-best and top-k performance first. Only after
that, use the winner as the base for Phase B ranking/confidence work.
