# Peptide finetuning with receptor MSAs

The active September 28 run uses the existing peptide preprocessing and training
scripts, together with precomputed protein MSAs from the shared Protenix database.
It was already running when this audit began; the audit preserves that job. No new
MSA generation or remapping script was used for this run.

## Scripts and their responsibilities

| Script | Inputs | Outputs and behavior |
|---|---|---|
| [`preprocess_peptide.sh`](../preprocess_peptide.sh) | Converted peptide CIF directory, CCD resources, seed, cache directory | CPU-only wrapper around the converter. Reuses unchanged successful entries. `--check` verifies existing entries; `--dry-run` prints the command. |
| [`scripts/prepare_peptide_training_data.py`](../scripts/prepare_peptide_training_data.py) | CIFs with merged `PEP` residues | Restores required polymer/assembly metadata, preserves observed peptide chemistry and bonds, creates ligand reference conformers, and writes bioassembly caches and peptide-interface indices. |
| [`finetune_peptide.sh`](../finetune_peptide.sh) | Prepared cache, checkpoint, environment variables, extra trainer arguments | `train` checks the cache and launches `runner/train.py` with single-node DDP. `prepare` runs preprocessing only; `all` prepares and trains. |

The converter writes these files under `PREPARED_DIR`:

```text
bioassembly/*.pkl.gz   # Atom/token arrays and sequences consumed by training
indices.csv           # Peptide–protein and peptide–nucleic-acid interfaces
records/*.json        # Per-CIF signatures, output hashes, and chemistry audits
preparation.json      # Preparation counts and status
failures.json         # Rejected source CIFs and reasons
```

Reference conformers use ETKDGv3. The converter discards native coordinates before
generating ligand reference positions, after assigning stereochemistry. Native
structure coordinates remain in the training cache as supervision. Peptides
retain their custom ligand representation.

The cache stores `msa_features=None` and `template_features=None`: the MSA
featurizer loads alignments during training. Enabling MSAs therefore does not
require regenerating the CIF caches.

## Data paths and CPU preparation

Run the examples from the repository root. The source and prepared directories
for this job are:

```text
DATA_DIR=/root/shiyu/pdb_peptide_to_lig_20190930_strict_filtered_nan0.7_mmcif_ligand_peptide
PREPARED_DIR=/root/shiyu/peptide_finetune_data
PROTENIX_ROOT_DIR=/mnt/project_abd/public_databases/protenix.20260101
PYTHON_BIN=/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python
```

Prepare or reuse the cache independently:

```bash
SKIP_PREP_ERRORS=1 PREPARED_DIR=../peptide_finetune_data \
  bash preprocess_peptide.sh
```

Check the existing cache without conversion or training:

```bash
SKIP_PREP_ERRORS=1 PREPARED_DIR=../peptide_finetune_data \
  bash preprocess_peptide.sh --check
```

The source directory contains 8,030 CIFs. This run uses the 5,397 successfully
prepared structures and their 13,311 interface rows; 2,633 rejected CIFs are
excluded through `SKIP_PREP_ERRORS=1`. Use the same data paths, seed, CCD resources,
and skip-errors policy for preprocessing and training. The historical
`preparation.json` records `ready: false` with `skip_errors: false`; the active
launch explicitly checks the cache with `--skip-errors --check` and accepts the
successful entries. This does not mean every source CIF was converted.

## How protein MSAs are loaded

[`get_msa_featurizer()`](../protenix/data/pipeline/dataset.py) passes the active
dataset's MSA configuration to
[`MSAFeaturizer`](../protenix/data/msa/msa_featurizer.py). The source manager looks
up the cached protein sequence by exact sequence identity in:

```text
/mnt/project_abd/public_databases/protenix.20260101/common/seq_to_pdb_index.json
```

Each matching entry identifies a directory beneath:

```text
/mnt/project_abd/public_databases/protenix.20260101/mmcif_msa_template
```

For example, a mapping value `457/84457` resolves to files named `pairing.a3m`
and `non_pairing.a3m` in that directory. The configured paired source is
`pairing`; the unpaired pool includes `pairing` and `non_pairing`.

The active run changes these MSA settings:

```text
data.msa.enable_prot_msa = true
data.msa.max_size.train = 512
```

The 512-row limit applies to the assembled training MSA. Raw paired and unpaired
input limits remain 10,000 each. Protein receptor alignments condition complex
prediction; the ligand peptide has no protein MSA of its own. RNA MSAs and
templates remain disabled.

MSA coverage is incomplete. Unmatched sequences can fall back to query-only
features, and a crop may omit protein chains that have alignments. This run does
not search for missing MSAs. The `--msa-map` argument of `evaluate_peptide.sh`
belongs to the evaluation workflow and is not a training-launcher argument.

## Recorded eight-GPU launch

The active job was launched with the equivalent command below. It is already
running; this command documents how it was started.

```bash
SKIP_PREP_ERRORS=1 PREPARED_DIR=../peptide_finetune_data \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
nohup bash finetune_peptide.sh train \
  --data.msa.enable_prot_msa true \
  --data.msa.max_size.train 512 \
  > log/finetune_peptide_20260928_1438.log 2>&1 &
```

The launcher appends extra trainer arguments after its defaults. Thus the final
`--data.msa.enable_prot_msa true` overrides its earlier `false`. The saved
`config.yaml` confirms the effective value. A command-only check is:

```bash
SKIP_PREP_ERRORS=1 PREPARED_DIR=../peptide_finetune_data \
  bash finetune_peptide.sh train --dry-run \
  --data.msa.enable_prot_msa true \
  --data.msa.max_size.train 512
```

Other settings match the executable defaults in the current launcher:

| Setting | Active value |
|---|---|
| Initial checkpoint | Official `protenix-v2.pt`, strict parameter-only loading |
| GPUs / processes | 0–7 / eight DDP ranks |
| Optimizer updates | 100,000 |
| Learning rate / warmup | 1e-5 / 200 updates |
| Crop size | 384 tokens |
| Complete spatial ligand crops | Disabled |
| Diffusion batch / chunk | 4 / 1 |
| Recycles | Up to 4 |
| Confidence mini-rollout | 20 steps, one sample |
| Gradient accumulation | 1 |
| Precision | bf16 |
| Sampling | Uniform over the prepared interface rows |
| Epoch size / loader workers | 10,000 / 2 per rank |
| Checkpoint / logging interval | 500 / 10 updates |
| RNA MSAs / templates / evaluation / EMA | Disabled |

## Run files and verification

Run directory:

```text
/mnt/project_abd/shiyu/peptide_finetune_output/peptide_ligand_v2_finetune_20260928_064012
```

The run's `config.yaml` is the authoritative saved configuration. Its
`checkpoints/` directory contains weights and `checkpoints.jsonl`; `logs/` contains
`data_samples_rank0.jsonl` through `data_samples_rank7.jsonl`.

Watch the console log:

```bash
tail -f log/finetune_peptide_20260928_1438.log
```

Check device use:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

For runtime MSA evidence, inspect each rank's sample records. `N_msa`,
`N_msa_prot_pair`, and `N_msa_prot_unpair` report actual loaded features. Depths of
256 paired + 256 unpaired rows, or 511 unpaired rows, establish homolog use;
the `query_only_msa` augmentation flag alone does not establish MSA availability.

At the initial audit, all eight ranks had reached logged step 5419, all eight had
recent samples with 512-row MSAs, and all 542 logged metric dictionaries contained
finite values. In the last 50 records per rank, 223/400 samples had more than two
MSA rows. This is an observed sample window, not full-dataset alignment coverage.
Checkpoint `4999.pt` had been saved successfully, and all GPUs were active at
roughly 28 GB each. The dated audit JSON under the run's `audits/` directory records
the final observation and process identities; the training run continues afterward.

The earlier prediction/evaluation scripts are documented separately in
[peptide pose evaluation](peptide_pose_evaluation.md) and
[training-subset evaluation](peptide_training_evaluation.md). They are not needed
to prepare or launch this MSA finetuning run.
