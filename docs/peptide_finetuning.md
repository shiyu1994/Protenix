# Finetune Protenix-v2 on converted peptide CIFs

`preprocess_peptide.sh` prepares the peptide data independently on CPUs, and
`finetune_peptide.sh train` validates its output and launches the existing trainer.
Both scripts default to the supplied directory:

```text
/root/shiyu/pdb_peptide_to_lig_20190930_strict_filtered_nan0.7_mmcif_ligand_peptide
```

For the eight-GPU run with receptor MSAs, see
[peptide MSA finetuning](peptide_msa_finetuning.md). It documents each preparation
script, MSA lookup paths, the recorded launch, and runtime verification.

The standard CIF converter cannot read these files unchanged. They omit polymer
and assembly metadata and represent arbitrary peptides as a `PEP` residue.
`scripts/prepare_peptide_training_data.py` handles this format and saves cached
bioassembly dictionaries and a training index. The input CIFs remain unchanged.

## Run

From the repository root, prepare all CIFs without starting training:

```bash
bash preprocess_peptide.sh
```

The standalone launcher uses the same converter, cache format, and data defaults
as `finetune_peptide.sh prepare`, which remains supported. It hides GPUs from the
preprocessing process and requires the Protenix Python dependencies and CCD
resources, but no model checkpoint. It can be launched from any working directory.

With the default paths, it produces:

```text
/root/shiyu/peptide_finetune_data/
├── bioassembly/*.pkl.gz   # Cached structures for the training loader
├── indices.csv           # Peptide-interface training rows
├── records/*.json        # Per-CIF signatures and chemistry audits
├── preparation.json      # Counts and readiness status
└── failures.json         # Rejected CIFs and reasons
```

Check a prepared dataset without reconverting files or launching training:

```bash
bash preprocess_peptide.sh --check
```

Then start training using that cache:

```bash
bash finetune_peptide.sh train
```

Some source peptides have incomplete or ambiguous chemistry that RDKit cannot
sanitize or embed. Failures are recorded in `../peptide_finetune_data/failures.json`.
By default, any failure stops the launcher before training. To explicitly exclude
failed structures and train on the successfully prepared files:

```bash
SKIP_PREP_ERRORS=1 bash preprocess_peptide.sh
SKIP_PREP_ERRORS=1 bash finetune_peptide.sh train
```

Successful cached structures are reused. The script processes every `.cif` in
`DATA_DIR` unless `CIF_IDS` is set. A source-file hash, converter hash, seed, CCD
resource metadata, and output-file hash are checked before reusing each entry.
Interrupted preparation can be restarted with the same command.

To choose the output directory and CPU worker count, export the settings for both
commands:

```bash
export PREPARED_DIR=/root/shiyu/peptide_finetune_data
export PREP_WORKERS=4
bash preprocess_peptide.sh
bash finetune_peptide.sh train
```

Run with different training settings:

```bash
SKIP_PREP_ERRORS=1 \
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
MAX_STEPS=5000 LR=0.00001 CROP_SIZE=384 \
DIFFUSION_BATCH_SIZE=4 ACCUMULATION_STEPS=4 \
OUTPUT_DIR=/root/shiyu/peptide_finetune_output \
bash finetune_peptide.sh all
```

After preparation, `train` verifies the existing cache and index before launching:

```bash
SKIP_PREP_ERRORS=1 bash finetune_peptide.sh train
```

Use the same `DATA_DIR`, `PREPARED_DIR`, `CIF_IDS`, `SEED`, `PROTENIX_ROOT_DIR`, and
`SKIP_PREP_ERRORS` as during preparation. A subset cache cannot silently stand in
for preparation of the full directory. The default `train` mode fails on missing,
changed, or corrupt cache entries; it does not reconvert them. Explicit `all` mode
combines preprocessing and training.

Print commands without running them:

```bash
bash preprocess_peptide.sh --dry-run
bash preprocess_peptide.sh --help
bash finetune_peptide.sh --dry-run
bash finetune_peptide.sh --help
```

## Using the accepted files after a partial preprocessing failure

A nonzero preprocessing exit does not mean every cached structure is unusable.
`indices.csv` contains only successful structures. With the default strict policy,
any rejected CIF makes `preparation.json` report `ready: false` and prevents training.
To verify and train on the accepted subset without retrying failed conversions:

```bash
SKIP_PREP_ERRORS=1 bash preprocess_peptide.sh --check
SKIP_PREP_ERRORS=1 bash finetune_peptide.sh train
```

Keep `train` in this command, or use the default mode. Running `all` or the
preprocessor without `--check` retries rejected CIFs, which can take hours.
The check still verifies input hashes, output hashes, settings, and the index.
It does not rewrite `preparation.json`; its final `ready` output reflects the
selected exclusion policy. A missing, stale, or corrupt successful entry still
causes a check failure because the expected index no longer matches.

The September 21 cache audit found 5,397 accepted structures / 13,311 interface rows
and 2,633 rejected structures. The accepted cache passed verification without
reconversion. Rejections included 1,681 RDKit aromatic-chemistry failures, 541 atom
names incompatible with CCD references, 281 token/representative-atom assertions,
and 130 other conversion errors. Excluding these entries does not repair their
chemistry or missing-atom metadata; inspect `failures.json` before choosing this
subset for an experiment.

An earlier training run also crashed on accepted structure `6v2d` because a protein
residue lacked backbone atom `C`. The featurizer now marks protein/RNA/DNA frames
invalid when any required backbone atom is absent, preserving the observed atoms
and other supervision. Existing prepared files work with this fix and do not need
to be regenerated.

## Defaults

| Setting | Default |
| --- | --- |
| Model | Official `protenix-v2` checkpoint, strict parameter loading |
| Python | `/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python` |
| Shared data root | `/mnt/project_abd/public_databases/protenix.20260101` |
| Prepared data | `../peptide_finetune_data` relative to the repository |
| Training output | `/mnt/project_abd/shiyu/peptide_finetune_output` |
| GPU/processes | GPUs 0–7, eight processes; single-node DDP |
| Optimizer updates | 100,000 |
| Learning rate/warmup | 1e-5 / 200 updates |
| Crop size | Up to 384 tokens; partial peptide/ligand crops allowed |
| Diffusion batch/chunk | Four noise samples per structure / one |
| Recycles | Up to four |
| Confidence mini-rollout | 20 diffusion steps, one sample |
| Precision/kernels | bf16, PyTorch triangle attention and layer normalization |
| Structure batch | One structure per GPU per microbatch |
| Gradient accumulation | One microbatch per optimizer update |
| Checkpoints/logging | Every 500 / 10 updates; always save the final checkpoint |
| EMA | Disabled |
| Preprocessing/loader workers | Four / two per GPU |
| MSA/templates/evaluation | Disabled |
| Sampling | Uniform over peptide–protein and peptide–nucleic-acid interface rows |

These are conservative starting settings, not tuned hyperparameters. Adjust
`MAX_STEPS`, `CROP_SIZE`, `DIFFUSION_BATCH_SIZE`, and `ACCUMULATION_STEPS` for your
experiment. One structure per GPU means the effective structure batch size is
`NPROC_PER_NODE * ACCUMULATION_STEPS`; diffusion batch size is a separate setting.

`COMPLETE_LIGAND_CROPS=false` keeps nearby peptide atoms even when the entire
peptide does not fit inside a spatial crop. The repository's whole-ligand option
removes an incompletely selected ligand rather than expanding the crop, which can
otherwise leave only protein atoms in a peptide-interface sample. Set
`COMPLETE_LIGAND_CROPS=true` only when that whole-ligand policy is intended.

Set `PYTHON_BIN`, `PROTENIX_ROOT_DIR`, `DATA_DIR`, `PREPARED_DIR`, `OUTPUT_DIR`, and
`CHECKPOINT_PATH` to move the workflow to another machine. No CUDA extension
compilation, `CUTLASS_PATH`, MSA database, or template alignment tools are required
with the default kernels and feature settings.

Extra `runner/train.py` options can be appended as `--key value` pairs:

```bash
SKIP_PREP_ERRORS=1 bash finetune_peptide.sh train --loss.weight.alpha_bond 1.0
```

The repository's trainer reparses arguments by whitespace, so paths and option
values passed to it must not contain spaces. Avoid overriding dataset/checkpoint
paths through extra runner options; use the documented environment variables so
the launcher's preparation checks cover the same files as training.

## Small validation run

Keep subset artifacts separate from full-data preparation:

```bash
CIF_IDS=1a07,1a0n,1a1b,1a1n \
PREPARED_DIR=/root/shiyu/peptide_script_smoke_data \
OUTPUT_DIR=/root/shiyu/peptide_script_smoke_output \
PREP_WORKERS=2 NUM_WORKERS=0 \
MAX_STEPS=2 WARMUP_STEPS=1 LOG_INTERVAL=1 \
CROP_SIZE=256 DIFFUSION_BATCH_SIZE=2 N_CYCLE=1 MINI_ROLLOUT_STEPS=2 \
bash finetune_peptide.sh all
```

## Checkpoints and restart

The trainer creates a timestamped run directory containing `config.yaml`,
`checkpoints/`, `logs/`, `manifest/`, and `errors/`. The console output is also
captured as `launcher_<timestamp>_<pid>.log` under `OUTPUT_DIR`.
Check `manifest/run_status.json` and `logs/runtime_rank0.jsonl` for completion,
finite losses/gradients, and `did_update=true`. Checkpoints contain model weights,
Adam state, scheduler state, and RNG/data-loader state. Each raw v2 checkpoint
is approximately 5.2 GiB; choose an output filesystem and save interval accordingly.

Resume a raw checkpoint produced by this launcher, using the same data, seed,
GPU topology, and training settings. `MAX_STEPS` is the total target update count:

```bash
SKIP_PREP_ERRORS=1 \
RESUME_CHECKPOINT=/absolute/path/to/run/checkpoints/499.pt \
MAX_STEPS=2000 \
bash finetune_peptide.sh train
```

`RESUME_CHECKPOINT` loads optimizer and scheduler state as well as weights.
`CHECKPOINT_PATH` instead starts a new finetuning run from weights only. EMA and
the unrelated production launchers' `PROTENIX_FULL_RESUME` contract are not used.

## Data semantics and limits

The converter reconstructs observed protein/RNA/DNA sequences, retains ordinary
ligands and ions, and restores explicit `struct_conn` bond orders, including
aromatic bonds. Custom peptide atom names are remapped to unique four-character
names for the model. Each peptide gets a reference conformer generated with RDKit
ETKDGv3. Source coordinates establish stereochemistry but are discarded before
reference embedding; they remain the training target. Per-file JSON records
contain the source hash, atom-name mapping, peptide SMILES, and preservation checks.
The unambiguous missing nitrogen hydrogen in a neutral pyrrole/indole ring is
recovered for reference chemistry and recorded in the audit. Ambiguous
multiple-nitrogen aromatic rings still require adequate source chemistry; no
histidine tautomer is chosen automatically.

Only interfaces involving a converted peptide are sampled, although the cached
structure retains surrounding components for cropping. The launcher reuses the
`weightedPDB_before2109_wopb_nometalc_0925` configuration as a schema, overrides all
its structure/index paths, and selects uniform sampling. It does not mix the
original WeightedPDB data into training.

No assembly expansion or missing-atom/residue completion is performed. Unknown
resolution stays unknown (-1), so the existing loss code masks confidence-head
supervision for these structures. Diffusion and distogram supervision remain
enabled. This workflow does not create a validation split or report accuracy.

Files with ambiguous atom identities, unsupported polymer chemistry, missing bond
endpoints, covalently attached custom peptides, failed RDKit sanitization/embedding,
or no peptide/polymer interface are rejected and reported. It never substitutes
native target coordinates for a failed reference conformer. `--skip-errors` (or
`SKIP_PREP_ERRORS=1`) must be explicit to train on the accepted subset.

## Validation performed

The launcher completed two optimizer updates on two H100 GPUs with crop size
384, diffusion batch four, up to four recycles, and a 20-step mini-rollout. Both
ranks reported finite losses and gradients, no skipped updates, and a saved final
checkpoint. Twelve focused tests cover bond orders, independent reference geometry,
aromatic hydrogen recovery, cache invalidation, launcher modes, compatibility
between standalone preprocessing and the training launcher's cache checks, and
complete/incomplete polymer backbone frames. Full input features and labels for
cached `6v2d` now generate successfully with finite input tensors.
The combined peptide and cropped-permutation test suites passed all 17 tests.
Fifty samples from the existing cache also passed the training dataset loader with
finite features and labels, including all 18 `6v2d` interfaces and 32 randomly
selected interface rows. This validates data loading, not a new full training run.

The standalone launcher also prepared `1a07` and `1a0n` in a separate temporary
directory using two CPU workers, producing two bioassembly files and four interface
rows. Its subsequent `--check` completed successfully. This check was launched
from outside the repository to verify path handling.

A varied 11-CIF preprocessing check accepted 10 structures, including ions,
ordinary ligands, peptide–DNA complexes, and a glycosylated complex. `6qns` was
explicitly rejected because its aromatic nitrogen/hydrogen state cannot be
sanitized unambiguously from the converted CIF. This is not a full-corpus or
prediction-accuracy validation.
