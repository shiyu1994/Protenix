# Evaluate protein–peptide binding poses

`evaluate_peptide.sh` prepares coordinate-free model inputs, runs Protenix-v2,
and measures predicted peptide poses after aligning the receptor to the native
complex. It defaults to all 51 CIFs in:

```text
/mnt/project_abd/shiyu/dataset/full_assembly_peptide_ligand_cif/cif
```

This is full-assembly structure prediction: receptor, peptides, and cofactors are
predicted together. It does not hold the receptor fixed as a docking program would.
Native receptor and peptide coordinates are used for scoring, not as templates
or starting poses.

## Run

From this repository, run the complete benchmark with the official v2 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 bash evaluate_peptide.sh all
```

The prepared 51-case inputs are already available in
`/root/shiyu/peptide_pose_eval_v2`. The command verifies and reuses these files.
Defaults are seed 101, five predictions per complex, 200 diffusion steps, ten
recycles, bf16, chunk size 64, and PyTorch attention/layer-normalization kernels.
MSAs and templates are disabled by default; interpret results as a no-MSA baseline.

Run stages separately:

```bash
bash evaluate_peptide.sh prepare
CUDA_VISIBLE_DEVICES=0 bash evaluate_peptide.sh predict
bash evaluate_peptide.sh score
```

To evaluate different complexes concurrently on eight GPUs:

```bash
bash evaluate_peptide.sh all --gpus 0,1,2,3,4,5,6,7 \
  --output-dir /root/shiyu/peptide_pose_eval_v2_8gpu
```

`--gpus` specifies physical GPU IDs or UUIDs and overrides `CUDA_VISIBLE_DEVICES`
for each worker. Each GPU holds one model and predicts its own group of complexes.
Cases are assigned once, balancing their estimated cost by squared token count.
Memory is not pooled across GPUs; each complete complex must fit on one GPU.
`PROTENIX_EVAL_GPU_MEMORY_FRACTION=0.5` optionally limits each evaluation worker's
PyTorch CUDA allocator to half the GPU memory when sharing devices with training.
Leave extra room for CUDA contexts and non-PyTorch allocations. This limit does
not change model sampling settings, and an oversized case is reported as failed.
The evaluator disables dynamic chunk selection so `--chunk-size` is honored.
The combined `evaluate_peptide_checkpoint.sh` launcher uses fixed chunk size 32
and runs training-subset and test-set evaluation sequentially.
Merely exposing multiple GPUs through `CUDA_VISIBLE_DEVICES` keeps the original
single-worker behavior. Launch this command once, without `torchrun`.

The coordinator prepares inputs once, saves each worker's log and case assignment
under `workers/run_*/`, merges progress as cases finish, and writes the same
aggregate `scores/summary.json` and CSV files as a single-GPU run. Failed cases
remain in the full denominator. A worker crash is recorded without discarding
other workers' results. Ctrl+C/SIGTERM stops the spawned workers before the
coordinator releases its output lock.

Rerun the same command after interruption. Verified completed cases are skipped;
unfinished cases can be redistributed using a different `--gpus` list. If fewer
cases remain than GPUs, only that many workers are started. GPU selection applies
to `all` and `predict`; preparation and scoring remain CPU stages. For separate
stages, pass the same inference settings to `predict`:

```bash
bash evaluate_peptide.sh predict --gpus 0,1,2,3,4,5,6,7 \
  --output-dir /root/shiyu/peptide_pose_eval_v2_8gpu
bash evaluate_peptide.sh score --output-dir /root/shiyu/peptide_pose_eval_v2_8gpu
```

Use a new output directory for predictions made with an earlier evaluator code
version, as required by the code-hash checks. Finish or stop an existing evaluation
before assigning its GPU to a new worker.

Metrics now print automatically as each input complex finishes in `all` and
`predict` modes, after its requested seeds and samples are available. Each line
identifies the complex, seed, sample, and peptide chain and reports receptor-aligned
heavy-atom RMSD, backbone RMSD (angstroms), and native-contact recovery (`fnat`).
A following line reports the number of samples scored, the confidence-selected
top1 and oracle-best maximum peptide RMSDs, and top1 receptor Cα RMSD. With
multiple peptides, the maximum is the worst peptide in that sample. In multi-GPU
runs these lines appear in the coordinator's terminal/log while other workers
continue predicting. Failures are printed explicitly, without substituting zero
RMSD. Final CSV/JSON reports reuse live scores after verifying artifact hashes.
Standalone `score` also prints these metrics as each complex is evaluated.

For an evaluation already running with an older launcher, display metrics in
another terminal without restarting inference:

```bash
bash evaluate_peptide.sh watch --output-dir /root/shiyu/peptide_pose_eval_v2
```

For an eight-GPU run, point `--output-dir` at its directory instead. `watch` uses
CPUs, reads saved predictions and manifests, and writes nothing to the evaluation
directory. It works while the prediction process holds the output lock and with
older evaluator runs. Preparation must have produced `manifest.json`, and
prediction must have created `run_config.json`, before monitoring starts.
Already completed cases are printed first; new completed or failed cases are
reported every five seconds by default (`--poll-interval 5`). The monitor stops
when all requested cases have completed or failed. Ctrl+C stops only the monitor;
if inference was interrupted with unfinished cases, use Ctrl+C to stop waiting.
The saved seeds/sample counts are used. Scoring options such as `--contact-cutoff`
and `--no-atom-symmetry` apply as in `score` mode.

Print a single snapshot of currently completed cases and exit:

```bash
bash evaluate_peptide.sh watch --once --output-dir /root/shiyu/peptide_pose_eval_v2
```

Preparation and scoring use CPUs. Prediction requires CUDA, the repository's
dependencies, CCD resources, and a compatible checkpoint. The launcher defaults
to `/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python` and
`PROTENIX_ROOT_DIR=/mnt/project_abd/public_databases/protenix.20260101`.
Override `PYTHON_BIN` and `PROTENIX_ROOT_DIR` when moving to another installation.
Prediction paths must not contain whitespace because the repository's model
configuration parser splits arguments on spaces.

Evaluate a finetuned checkpoint in a separate output directory:

```bash
CUDA_VISIBLE_DEVICES=0 bash evaluate_peptide.sh all \
  --checkpoint /absolute/path/to/checkpoint.pt \
  --output-dir /root/shiyu/peptide_pose_eval_finetuned \
  --seeds 101,102,103 --samples 5
```

Checkpoints must use the trainer's `model` state-dict format and match the v2
architecture. Loading is strict. DDP's `module.` prefix is supported. The script
loads the `model` weights; it does not automatically select EMA weights.

A quick workflow check uses reduced sampling and its own output directory:

```bash
CUDA_VISIBLE_DEVICES=0 bash evaluate_peptide.sh all \
  --ids 8s6o-assembly1 \
  --output-dir /root/shiyu/peptide_pose_eval_quick \
  --samples 2 --steps 10 --cycles 1
```

`--ids` accepts comma-separated CIF stems and applies only to `all`/`prepare`.
`predict`, `score`, and `watch` use the saved manifest. Reduced-step runs
validate the workflow; their accuracy is not a substitute for the full benchmark.
Use `--help` for options and `--dry-run` to print resolved settings without running.

## Representation and chemistry

The test files retain amino-acid residue names and numbers but identify the target
chains through `_foldbench_peptide_ligand`. The evaluator represents those chains
as CCD ligands, with explicit peptide/inter-residue bonds and one token per heavy
atom. Receptors use complete `_entity_poly_seq` sequences, with CCD modifications.
Identical chains share entity identities and retain their explicit chain IDs.
Cofactors, glycans, and ions remain in the complex; water is omitted.

Protenix also supports peptides as protein polymers. A peptide inference entity
can be written as `{"proteinChain": {"sequence": "AGWKT", "count": 1}}`.
Standard amino acids then receive one token per residue; atom-level diffusion
still predicts their heavy atoms. Modified residues may receive atom tokens.
This evaluator deliberately uses the supplied ligand representation. A polymer
comparison requires a separately prepared run with consistent residue identities,
modifications, termini, and bonding. The earlier finetuning dataset's merged `PEP`
residues are a different input format; its existing cache cannot be relabeled to
recover amino-acid tokens. See [peptide finetuning](peptide_finetuning.md).

Preparation checks the actual Protenix atom construction and tokenizer and requires
every observed target-peptide heavy atom to map. These dataset-specific changes
are recorded in each reference JSON's `normalizations`:

- Canonical CCD atom aliases, including doubled terminal apostrophes in SAH.
- Terminal `NXT` amide nitrogen represented by bonded `CCD_NH2`.
- Six spurious intra-residue `C–N` self-links in four structures are omitted
  because they conflict with the CCD `N–CA–C` backbone. These are three PHE
  links in `7yvb`, one LTU link in `8ad9`, and FME links in `8fsq`/`8fsr`.
  CCD intramolecular bonds and the intended terminal caps remain present.
- Incomplete glycan connection identifiers are resolved only for an already
  declared covalent bond when exactly one candidate pair is 0.8–2.3 Å apart.
  This narrow use of native coordinates disambiguates chemical connectivity;
  it does not generate receptor–peptide contacts or distance constraints.

Duplicate covalent connections are deduplicated. Unsupported intra-residue
chemistry is rejected rather than silently overwriting CCD topology. Metal
coordination annotations are recorded in `ignored_connections`, while the ions
remain modeled. Metal contacts are not converted to covalent restraints. MSE
follows the repository's MET/SD normalization.

The largest case, `7yvb-assembly1`, has 2,583 tokens. The evaluator uses the runner
API directly with chunking to retain the entire assembly beyond the standard
CLI's advisory 2,560-token cutoff. Large assemblies require more GPU memory;
the full-size cases have not been GPU-benchmarked by the small smoke test.

## Optional receptor MSAs

Provide a JSON file mapping each distinct receptor sequence to existing MSA files:

```json
{
  "FULL_RECEPTOR_SEQUENCE": {
    "unpairedMsaPath": "/absolute/path/to/unpaired.a3m",
    "pairedMsaPath": "/absolute/path/to/paired.a3m"
  }
}
```

At least one MSA path is required for every receptor sequence in this mode.
Use the actual full sequences from prepared input JSONs. The evaluator does not
run an MSA search and does not use native templates.

```bash
bash evaluate_peptide.sh all --msa-map /absolute/path/to/msa_map.json \
  --output-dir /root/shiyu/peptide_pose_eval_with_msa
```

Use `--seeds 101,102,103,104,105 --samples 5` for 25 predictions per complex.
`--samples` applies separately to every seed; the default is one seed (101)
with five samples. Top1 and best RMSD pool predictions across all seeds, and
`scores/poses.csv` identifies every prediction by its seed and sample. Changing
the seed list or sample count requires a new output directory. Compare official
and finetuned test runs with the same seed list and number of samples.

## Scores and interpretation

The scorer rigidly aligns matched receptor Cα atoms with a proper Kabsch rotation.
It never independently superposes the peptide. At least three noncollinear Cα
atoms and 90% native receptor Cα coverage are required by default.

Equivalent receptor-chain permutations are enumerated and selected by minimum
receptor Cα RMSD. Identical peptide copies receive a one-to-one assignment
minimizing mean squared peptide pose RMSD after that receptor fit. Peptide RMSD
breaks receptor-fit ties within 1e-6 Å. The receptor RMSD is reported separately.
This can differ from a local-interface alignment when the receptor changes shape.

Each target peptide receives:

| Field | Meaning |
| --- | --- |
| `pose_rmsd` | Heavy-atom RMSD in the receptor frame, with allowed CCD atom symmetries |
| `atom_name_rmsd` | Heavy-atom RMSD with literal atom-name correspondence |
| `backbone_pose_rmsd` | RMSD of observed N, CA, C, O atoms in the same receptor frame |
| `fnat` | Fraction of native receptor-residue/peptide-residue contacts recovered |
| `contact_precision` | Fraction of predicted residue contacts also present natively |

CCD symmetry permutations must preserve the observed atom set and fixed external
bond endpoints. Symmetry is applied within residues, with no residue-order
permutations. Use `--no-atom-symmetry` for literal atom-name scoring.
Only observed native heavy atoms are scored; predicted completion atoms have no
native target. Missing target-peptide atoms cause scoring failure. Alternate
locations use greatest occupancy, then a deterministic alternate-ID tie break.

Contacts use any heavy-atom distance ≤5 Å, adjustable with `--contact-cutoff`.
They include receptor proteins only. Contacts are evaluated on receptor atoms
present in both structures; the output reports that coverage. `fnat` is null
when there are no native contacts; precision is zero when none are predicted.

`top1` selects the highest global `ranking_score` across all requested seeds.
`best` is the oracle sample minimizing the largest peptide RMSD in the complex.
Success requires **every peptide** in that sample to meet the threshold, by
default 2 Å or 5 Å (`--thresholds 2,5`). Failed preparation, prediction, or
incomplete sampling remains in the full requested denominator and counts as
failure. This is a pose-RMSD/contact benchmark, not DockQ or a binding-affinity test.

## Outputs and restart

```text
peptide_pose_eval_v2/
  manifest.json                 # All requested cases, hashes, preparation status
  preparation_failures.json
  inputs/<case>.json            # Model input; no native coordinates
  references/<case>.json        # Native targets, atom mapping, chemistry audit
  run_config.json               # Checkpoint, seeds, sampling settings, hashes
  prediction_manifest.json      # Completion/failure and output hashes per case
  workers/run_*/                # Multi-GPU assignments, worker logs and progress
  predictions/<case>/seed_<seed>/predictions/
    <case>_sample_<rank>.cif
    <case>_summary_confidence_sample_<rank>.json
  scores/
    summary.json                # Primary aggregate report
    complexes.csv               # Top1 and oracle result for each requested case
    poses.csv                  # One row per scored sample
    peptides.csv               # One row per peptide per sample
    errors.json
```

Sample indices are confidence ranks within each seed because the runner sorts
its output. The scorer selects top1 across all seeds using the saved confidence
values. In `score` mode, saved run seeds/sample counts take precedence over CLI
sampling defaults.

Rerun the same command after interruption. Completed cases are reused only after
hash verification; a failed case is rerun for all its seeds. Source CIFs, prepared
artifacts, and supplied MSA files are checked. Relevant feature/model code,
checkpoint content, inference settings, and CCD path/size/mtime are recorded for
reuse checks. Use a new output directory when changing inputs, checkpoints, or
inference settings. Keep source files and resources accessible during reruns.
Concurrent writers to one output directory are rejected with a file lock.

Failures produce a nonzero exit. Inspect `preparation_failures.json`,
`prediction_manifest.json`, and `scores/errors.json` as appropriate. Scoring
rejects changed prediction files and cases without a verified completed run.
Reports remain available when some cases fail; their denominator stays explicit.

## Validation

All 51 cases passed real Protenix atom construction/tokenization, with no native
heavy atoms missing from the generated input, including receptors and cofactors.
All 51 also passed the complete inference featurizer with finite input tensors.
Rigidly transformed native structures scored below 1e-6 Å peptide and receptor
RMSD with full native-contact recovery, validating atom/chain mapping and scoring.
These synthetic checks do not measure model accuracy. The validation report is
saved in `/root/shiyu/peptide_pose_eval_v2/validation.json`.
A genuine GPU run on `8s6o-assembly1` loaded the official 464.44M-parameter v2 model
with strict matching and completed two samples at 10 diffusion steps/one recycle,
including prediction CIF parsing and scoring. Its outputs are in
`/root/shiyu/peptide_pose_eval_smoke_20260921`; these are smoke-test results only.

Focused tests cover receptor alignment, peptide displacement, atom symmetry and
bond anchors, hydrogen/permutation indexing, chain swaps, one-to-one assignment,
missing atoms, invalid alignment geometry, confidence selection, failed-case
denominators, changed prediction files, coordinate-free inputs, source self-link
normalization, and explicit ligand-versus-polymer tokenization. They run without
CUDA or a checkpoint and skip when scientific dependencies are unavailable.
The combined evaluation, peptide preprocessing, and cropped-permutation suites
passed all 32 tests. Formatting, lint, docstring, and shell syntax checks passed.

```bash
PROTENIX_ROOT_DIR=/mnt/project_abd/public_databases/protenix.20260101 \
LAYERNORM_TYPE=torch CUDA_VISIBLE_DEVICES='' \
/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python -m pytest -q \
  tests/test_peptide_pose_evaluation.py \
  tests/test_peptide_training_scripts.py tests/test_cropped_atom_permutations.py
```

The multi-GPU path also completed a real two-worker run on GPUs 1 and 2, using
`8fg6-assembly1` and `8s6o-assembly1` with one sample, two diffusion steps, and one
recycle each. Both workers loaded the official v2 checkpoint and exited
successfully; the combined report scored both complexes. Its outputs are in
`/root/shiyu/peptide_pose_eval_multigpu_smoke_20260921`.
The evaluation and parallel-worker suites pass 20 tests, including distribution
of 51 cases across eight workers, independent device selection, worker crashes,
resuming only unfinished cases, and interruption cleanup:

```bash
PROTENIX_ROOT_DIR=/mnt/project_abd/public_databases/protenix.20260101 \
LAYERNORM_TYPE=torch CUDA_VISIBLE_DEVICES='' \
/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python -m pytest -q \
  tests/test_peptide_pose_parallel.py tests/test_peptide_pose_evaluation.py
```

These small GPU checks validate execution and aggregation. The full 51-case
prediction benchmark is a separate experiment.
