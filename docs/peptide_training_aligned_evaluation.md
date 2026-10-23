# Training-aligned peptide evaluation

`evaluate_peptide_training_aligned.sh` evaluates the 51 test complexes with each
peptide collapsed into one custom ligand residue. It uses the training
converter's `make_reference()` (ETKDGv3, reference seed 42), generated four-character
atom names, one residue index and one reference-coordinate group per peptide.
Both old and new modes use one token per peptide heavy atom and ligand/UNK types.

Run with an explicit checkpoint and a **new output directory**:

```bash
bash evaluate_peptide_training_aligned.sh all \
  --checkpoint /path/to/checkpoints/8999.pt \
  --output-dir /mnt/project_abd/shiyu/peptide_eval_training_aligned_8999
```

Defaults match the previous sampling settings: eight GPUs, seeds 101–105, five
samples per seed, 200 diffusion steps, ten recycles, chunk size 32, existing
receptor MSA map, no templates. Override with the usual evaluator flags, e.g.
`--gpus 0,1`. `prepare` is CPU-only; `all` prepares, predicts, and scores. Reuse
the same directory and settings to resume. Reference seed is controlled by
`REFERENCE_SEED` (default 42). Model weights are not changed.

The JSON inputs embed `ligand_reference_overrides`, which the inference feature
builder applies before tokenization. It keeps the observed peptide heavy atoms
in native-reference order. Whole-peptide conformers contain no native pose:
the training routine assigns stereochemistry using the observed coordinates,
discards those coordinates, and independently embeds a new conformer. The
native coordinates remain separately in `references/` for scoring.

An explicit `prediction_atom_mapping` maps generated atom names back to native
residue/atom identities without fitting or altering coordinates. Thus both
heavy-atom scoring and C-alpha scoring use the same native targets as before.

Outputs:

- `manifest.json`, `preparation_failures.json`: input validation and provenance.
- `references/*.json`: atom mappings and per-chain chemistry/embedding audits.
- `predictions/`: sampled CIFs with custom peptide residue/atom names.
- `scores/summary.json`: original global-receptor/heavy-atom metric.
- `scores/epitope_ca_summary.json` and `.csv`: peptide C-alpha RMSD after fitting
  receptor C-alpha atoms within 10 A of native peptide C-alpha atoms. Multiple
  peptides use the union epitope and maximum peptide RMSD. Top uses the maximum
  original model `ranking_score`; best minimizes this RMSD across all 25 poses.
  Thresholds are strictly `<2` and `<5`. Success rates use all requested cases;
  mean/median use only cases with all expected poses scored.

This aligns the **custom ligand peptide representation**, not every aspect of
training. Receptors still use full-sequence inference construction and MSAs;
evaluation uses full assemblies rather than random 1024-token training crops.
Peptide chemistry starts from the existing validated CCD topology and explicit
connections, since the test CIF format differs from the training PEP exports.
Only observed peptide atoms are retained, as in training. No peptide MSA is used.

The `8qci` receptor–peptide disulfide is retained and recorded under
`training_exception`. The training converter excludes external peptide bonds;
this evaluation exception preserves the native chemistry instead of dropping a
benchmark case or silently deleting its bond. The default `ccd` evaluation mode
remains available via `evaluate_peptide.sh` without the new wrapper.
