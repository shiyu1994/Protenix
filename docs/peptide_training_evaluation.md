# Evaluate reconstruction of peptide training examples

## Run both training-subset and test-set evaluation yourself

The combined launcher selects the latest completed checkpoint when invoked,
evaluates 50 training structures, then evaluates the 51 test assemblies with MSAs:

```bash
bash evaluate_peptide_checkpoint.sh --dry-run

nohup bash evaluate_peptide_checkpoint.sh \
  --output-dir /mnt/project_abd/shiyu/peptide_eval_latest_msa \
  --gpus 0,1,2,3,4,5,6,7 \
  > log/evaluate_peptide_latest_msa.log 2>&1 &
```

`--dry-run` reads checkpoint metadata and prints commands without preparing or
evaluating structures. Use `--checkpoint /path/to/18499.pt` to pin weights.
Rerunning with the same output directory retains that directory's checkpoint;
choose another directory to evaluate newer weights. `train` and `test` positional
stages run either dataset separately.

Pass `--seeds` to evaluate multiple prediction seeds in both datasets. `--samples`
is the number of predictions **per seed**. For example, five seeds with five
samples each produce 25 predictions per complex per checkpoint:

```bash
nohup bash evaluate_peptide_checkpoint.sh \
  --seeds 101,102,103,104,105 --samples 5 \
  --output-dir /mnt/project_abd/shiyu/peptide_eval_latest_msa_5seeds \
  --gpus 0,1,2,3,4,5,6,7 \
  > log/evaluate_peptide_latest_msa_5seeds.log 2>&1 &
```

Add `--dry-run` to inspect the commands first. The default remains `--seeds 101
--samples 5`. Seeds must be distinct integers from 0 through 4,294,967,295.
Use the same seed list and sample count to resume, or a new output folder to
change them. Top1 uses the highest confidence across the complete prediction
pool; best-of-25 in the example uses the lowest native peptide RMSD across all
25 predictions. A one-seed baseline has only five candidates and therefore is
not a matched sampling-budget comparison to this run.

Seeds run sequentially within each GPU worker, with five samples per prediction
call in this example. Both checkpoints receive the same training crop, MSA
features, and prediction seeds. Structure-selection and crop seeds are unchanged.
The test stage evaluates the selected checkpoint; run the official checkpoint
separately with the same seeds and samples for a matching test baseline.

The combined launcher runs the two evaluations sequentially on the selected GPUs.
It uses fixed chunk size 32 and a 100% evaluation CUDA allocator limit by default
(unless `PROTENIX_EVAL_GPU_MEMORY_FRACTION` is set). Dynamic chunking is disabled
in the evaluator so `--chunk-size` is honored. The interrupted September 30 run
showed that dynamic chunking could override the requested size and exhaust the
evaluation memory limit. Set `--memory-fraction 0.5` to explicitly limit each
evaluation worker's CUDA allocator to half of its GPU memory.

Outputs are `train_subset/comparison.md`, `train_subset/summary.json`, and
`test51/scores/summary.json` beneath the selected output root. Training-subset
per-structure metrics appear in `train_subset/worker_*.log`; test metrics print in
the main log as each structure finishes. The launcher does not stop training.

## Training-subset protocol

`evaluate_peptide_training.sh` compares the official checkpoint and a finetuned
checkpoint on 50 distinct structures actually sampled during training. The
selection uses seed 20260923, is uniform over seen structures, and does not use
prediction quality or structure size. One actually sampled interface is selected
per structure. The sampler sequence must match every recorded training sample.

This is a **training-crop reconstruction test**. Both checkpoints receive the
same deterministic crop from the saved training configuration, including the
custom peptide ligand references used during training. Native coordinates select
the spatial crop, then are erased before featurization. Inference receives no
coordinate labels or templates. Receptor MSAs follow the saved training
configuration, including its training depth cap (512 for the September 28 run).
Unmatched receptor sequences can still have query-only features; each case's
`crop.msa` records actual total, paired, and unpaired row counts. These are new reproducible crops
from the training policy, not a replay of the exact stochastic training crops.
If a crop omits the indexed peptide or lacks three receptor alignment atoms,
successive fixed seeds are tried, up to 64 attempts. The complete peptide is
required. These retries do not use prediction outcomes or replace structures.

The default prediction settings are seed 101, five samples per seed, 200 diffusion steps,
10 recycles, bf16, and chunk size 64. Model weights are loaded strictly. Each GPU
holds both checkpoints and evaluates its assigned structures sequentially with
identical features and random seeds. The model architecture is unchanged.
The standalone training evaluator accepts `--seeds 101,102,103,104,105`; its
existing `--seed 101` option remains available for single-seed runs. The two
options are mutually exclusive.

Prepare the selection, then launch once with eight GPUs:

```bash
bash evaluate_peptide_training.sh prepare \
  --training-config /path/to/finetuning_run/config.yaml \
  --output-dir /path/to/trainset50_evaluation \
  --after /path/to/finetuning_run/checkpoints/18499.pt \
  --count 50 --selection-seed 20260923

bash evaluate_peptide_training.sh launch \
  --training-config /path/to/finetuning_run/config.yaml \
  --output-dir /path/to/trainset50_evaluation \
  --before /path/to/protenix-v2.pt \
  --after /path/to/finetuning_run/checkpoints/18499.pt \
  --gpus 0,1,2,3,4,5,6,7
```

Use a filesystem with space for prediction coordinates. Rerun the same launch
command to resume; verified completed pairs are reused. Changing checkpoints,
prediction settings, source code, or the manifest requires another output folder.

`prepare --after` reads the completed checkpoint journal entry and reconstructs
training draws only through that checkpoint's zero-based step, using `seed + epoch`
across sampler epochs. It verifies the reconstruction against all recorded samples
up to that step. A frozen `training_samples_snapshot.json` allows the original
training logs to keep growing without invalidating evaluation. Alternatively,
`--checkpoint-step` supplies an explicit zero-based cutoff. Preparation creates a
new selection; use `launch` to resume an existing output directory.

The saved training configuration must currently describe a single dataset with
uniform interface sampling, no gradient accumulation, an epoch size divisible by
the rank count, and disabled RNA MSAs/templates/constraints. Protein MSAs are
supported. Full ligand preservation is checked for each indexed peptide.
Unusable crops and prediction failures are reported without replacing structures.

To share GPUs with ongoing training, set `PROTENIX_EVAL_GPU_MEMORY_FRACTION=0.5`
on the evaluation command. This caps each evaluation worker's PyTorch CUDA
allocator at half the device memory. CUDA context and non-PyTorch allocations
are additional, so leave spare memory. Evaluation shares compute and can slow
training. Use disjoint GPU lists for simultaneous training-subset and test-set
evaluations; each still shares its GPUs with the existing trainer.

The interrupted September 30 evaluation used checkpoint `18499.pt` from
`peptide_ligand_v2_finetune_20260928_064012`. Its output root is
`/mnt/project_abd/shiyu/peptide_checkpoint_eval_20260930_step18499`, with `train50/`
for paired training-crop evaluation and `test51/` for full-assembly evaluation
using `/root/shiyu/peptide_pose_eval_v2_msa_map.json`. Its evaluation processes were
stopped at the user's request. Use a new output directory with the combined
launcher above; the interrupted run's artifacts are not completed results.

## Metrics and artifacts

- Peptide pose RMSD: heavy atoms after aligning the indexed receptor's Cα atoms
  (C4′ for nucleic-acid receptors). Equivalent chain copies and the custom
  peptide's graph automorphisms are handled during scoring.
- Peptide shape RMSD: additionally superpose the peptide, to separate internal
  conformation accuracy from placement accuracy.
- Receptor RMSD: the indexed receptor's alignment atoms within the crop.
- Top1: prediction selected by model confidence across all seeds and samples.
  Best-of-N: prediction selected using native peptide pose RMSD across the same
  pool, where N = number of seeds × samples per seed. This is an oracle metric.

The coordinator refreshes `comparison.md`, `summary.json`, and
`per_interface.csv` every 30 seconds. Per-case JSONs under `cases/` contain all
sample scores with seed and sample IDs, training exposure counts, selected atom identities, feature
fingerprints, and native/predicted coordinate-file hashes. `.npz` files retain
the native crop and both sets of predictions for independent verification.
Coordinates are stored in seed-major order: all samples for the first requested
seed, then the second, and so on. `prediction_seeds` and `prediction_samples`
arrays identify each coordinate entry. The per-interface CSV records the seed
and sample selected for top1 and best for each checkpoint.
`manifest.json` records selection and training-source provenance; `run_config.json`
records checkpoints, code hashes, and inference settings. `processes.json` and
`exit_status.json` identify running processes and final completion.

Compare these results within this experiment. They use training crops and
individual receptor interfaces, whereas `evaluate_peptide.sh` evaluates full
assemblies and aligns all protein chains. A low reconstruction error on these
examples supports memorization under training conditions; it does not establish
held-out generalization or memorization of every training example.
