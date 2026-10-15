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

"""Paired checkpoint evaluation of the actual peptide training interfaces.

Native coordinates select a deterministic training crop and provide labels for
scoring. They are erased before featurization and never passed to the model.
This measures reconstruction of training crops, not blind full-assembly docking.
"""

import argparse
import collections
import copy
import csv
import fcntl
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.evaluation.peptide_pose_inputs import file_hash, write_json
from scripts.evaluation.peptide_pose_metrics import receptor_fit

REPO = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def verify_training_cache(cache_path):
    path = Path(cache_path).resolve()
    record_path = path.parent.parent / "records" / (path.name.removesuffix(".pkl.gz") + ".json")
    record = json.loads(record_path.read_text())
    checksum = file_hash(path)
    if record["status"] != "ok" or checksum != record["output_sha256"]:
        raise ValueError(f"Training cache differs from preprocessing record: {path}")
    return checksum


def make_dataset(config_path, *, load_msa=True):
    import yaml
    from ml_collections import ConfigDict
    from protenix.data.pipeline.dataset import BaseSingleDataset, get_msa_featurizer

    config = ConfigDict(yaml.safe_load(Path(config_path).read_text()))
    assert len(config.data.train_sets) == 1
    name = config.data.train_sets[0]
    data = config.data[name].to_dict()
    assert not data.get("constraint", {}).get("enable", False)
    if config.data.msa.enable_rna_msa:
        raise ValueError("Training reconstruction currently supports protein MSAs only")
    assert not config.data.template.enable_prot_template
    dataset = BaseSingleDataset(
        **data["base_info"],
        name=name,
        cropping_configs=data["cropping_configs"],
        msa_featurizer=get_msa_featurizer(config, name, "train") if load_msa else None,
        constraint=data.get("constraint", {}),
        ref_pos_augment=config.data.get("train_ref_pos_augment", True),
        lig_atom_rename=data.get("lig_atom_rename", False),
        template_use_probability=0.0,
    )
    return dataset, config


def feature_digest(features):
    """Content hash independent of pickle serialization and tensor device."""
    import torch

    digest = hashlib.sha256()
    for key, value in sorted(features.items()):
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def make_crop(dataset, index, crop_seed):
    from protenix.utils.seed import seed_everything

    row, full, cache_path = dataset._get_bioassembly_data(index)
    cache_hash = verify_training_cache(cache_path)
    # Preserve the target chemistry exactly as cached during finetuning.
    peptide_chain = str(row.chain_1_id if row.mol_1_type == "ligand" else row.chain_2_id)
    receptor_chain = str(row.chain_2_id if row.mol_1_type == "ligand" else row.chain_1_id)
    full_peptide_count = int((full["atom_array"].chain_id == peptide_chain).sum())
    for attempt in range(64):
        accepted_seed = (crop_seed + attempt) % (2**32)
        seed_everything(accepted_seed, deterministic=False)
        method, tokens, atoms, msa, templates, _ = dataset.crop(
            sample_indice=row, bioassembly_dict=full, template_use_requested=False,
            **dataset.cropping_configs,
        )
        peptide = atoms.chain_id == peptide_chain
        receptor = atoms.chain_id == receptor_chain
        anchors = receptor & atoms.is_resolved.astype(bool) & (
            ((atoms.atom_name == "CA") & atoms.is_protein.astype(bool))
            | ((atoms.atom_name == "C4'") & ~atoms.is_protein.astype(bool))
        )
        if int(peptide.sum()) == full_peptide_count and full_peptide_count and anchors.sum() >= 3:
            break
    else:
        raise ValueError("No complete indexed peptide and alignable receptor after 64 training-crop draws")
    clean_atoms = atoms.copy()
    clean_atoms.coord[:] = 0
    features, _, _ = dataset.get_feature_and_label(
        index, tokens, clean_atoms, msa, templates, clean_atoms,
        is_spatial_crop=True, max_entity_mol_id=int(clean_atoms.entity_mol_id.max()),
    )
    if "coordinate" in features or "coordinate_mask" in features:
        raise ValueError("Native coordinate labels appeared in inference features")
    if features["template_all_atom_mask"].any():
        raise ValueError("Unexpected templates in training reconstruction")
    metadata = {
        "index": index, "pdb_id": str(row.pdb_id), "receptor_chain": receptor_chain,
        "peptide_chain": peptide_chain, "crop_seed": accepted_seed, "initial_crop_seed": crop_seed,
        "crop_attempts": attempt + 1, "crop_method": method,
        "tokens": len(tokens), "atoms": len(atoms), "full_tokens": len(full["token_array"]),
        "peptide_atoms": full_peptide_count, "cache_path": cache_path, "cache_sha256": cache_hash,
        "feature_sha256": feature_digest(features),
        "source_atom_indices": atoms.peptide_source_atom_index.tolist(),
        "msa": {
            "rows": int(features["msa"].shape[0]),
            "protein_paired_rows": int(features["prot_pair_num_alignments"]),
            "protein_unpaired_rows": int(features["prot_unpair_num_alignments"]),
        },
    }
    return features, atoms, metadata


def equivalent_chain_indices(atoms, chain):
    """Only exchange copies with identical entity and retained atom identities."""
    target = np.flatnonzero(atoms.chain_id == chain)
    target_ids = atoms.mol_atom_index[target].tolist()
    entity = set(atoms.entity_mol_id[target].tolist())
    result = []
    for other in dict.fromkeys(atoms.chain_id):
        ids = np.flatnonzero(atoms.chain_id == other)
        if set(atoms.entity_mol_id[ids].tolist()) != entity:
            continue
        lookup = {int(atoms.mol_atom_index[i]): int(i) for i in ids}
        if len(lookup) == len(ids) and set(lookup) == set(target_ids):
            result.append((str(other), np.asarray([lookup[i] for i in target_ids])))
    return target, result


def score_interface(atoms, features, metadata, predicted):
    """Score the indexed peptide in its receptor frame, including atom symmetry."""
    native = np.asarray(atoms.coord, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if predicted.shape != native.shape or not np.isfinite(predicted).all():
        raise ValueError("Invalid prediction coordinates")
    receptor, receptor_copies = equivalent_chain_indices(atoms, metadata["receptor_chain"])
    peptide, peptide_copies = equivalent_chain_indices(atoms, metadata["peptide_chain"])
    protein = atoms.is_protein[receptor].astype(bool)
    anchors = ((atoms.atom_name[receptor] == "CA") & protein) | (
        (atoms.atom_name[receptor] == "C4'") & ~protein
    )
    anchors &= atoms.is_resolved[receptor].astype(bool)
    if anchors.sum() < 3:
        raise ValueError("Indexed receptor has fewer than three CA/C4' alignment atoms")
    permutations = np.asarray([features["atom_perm_list"][i] for i in peptide], dtype=int).T
    if not len(permutations) or any(set(p) != set(range(len(peptide))) for p in permutations):
        raise ValueError("Invalid complete-peptide atom permutation")
    native_peptide = native[peptide]
    best = None
    for receptor_name, receptor_ids in receptor_copies:
        rotation, translation, receptor_rmsd = receptor_fit(
            predicted[receptor_ids[anchors]], native[receptor[anchors]]
        )
        for peptide_name, peptide_ids in peptide_copies:
            aligned = predicted[peptide_ids] @ rotation + translation
            errors = np.mean(np.sum((aligned[permutations] - native_peptide) ** 2, axis=-1), axis=-1)
            rmsd = float(np.sqrt(errors.min()))
            shape_rmsd = min(receptor_fit(aligned[p], native_peptide)[2] for p in permutations)
            candidate = {
                "peptide_rmsd": rmsd, "peptide_shape_rmsd": shape_rmsd,
                "receptor_rmsd": receptor_rmsd, "receptor_anchor_atoms": int(anchors.sum()),
                "receptor_anchor_type": "CA" if protein.all() else "CA/C4'",
                "predicted_receptor_chain": receptor_name, "predicted_peptide_chain": peptide_name,
            }
            if best is None or receptor_rmsd < best["receptor_rmsd"] - 1e-6 or (
                abs(receptor_rmsd - best["receptor_rmsd"]) <= 1e-6 and rmsd < best["peptide_rmsd"]
            ):
                best = candidate
    if best is None:
        raise ValueError("No valid receptor/peptide correspondence")
    return best


def read_complete_jsonl(path):
    """Ignore only an unfinished final record while a training log is growing."""
    content = Path(path).read_bytes()
    end = content.rfind(b"\n") + 1
    return [json.loads(line) for line in content[:end].splitlines() if line.strip()]


def reconstruct_training_draws(dataset, config, observed, checkpoint_step):
    """Reconstruct consumed uniform draws through a zero-based saved step."""
    import torch

    if config.data[config.data.train_sets[0]].sampler_configs.sampler_type != "uniform":
        raise ValueError("Exposure reconstruction currently requires uniform sampling")
    worlds = {r["world_size"] for r in observed}
    if len(worlds) != 1 or config.iters_to_accumulate != 1:
        raise ValueError("Unsupported training world size or accumulation history")
    world_size = worlds.pop()
    if {r["rank"] for r in observed} != set(range(world_size)):
        raise ValueError("Training sample logs are required from every rank")
    if not 0 <= checkpoint_step < config.max_steps:
        raise ValueError("Checkpoint step is outside the configured training range")
    total = (checkpoint_step + 1) * world_size
    epoch_size = config.data.epoch_size
    if epoch_size % world_size:
        raise ValueError("Exposure reconstruction requires epoch_size divisible by world size")
    if config.data.train_sampler.get("soft_cluster_schedule_manifest", ""):
        raise ValueError("Precomputed sampling schedules are not supported")
    sampler_seed = config.data.train_sampler.get("seed", -1)
    sampler_seed = config.seed if sampler_seed is None or sampler_seed < 0 else sampler_seed
    sequences = []
    for epoch in range((total + epoch_size - 1) // epoch_size):
        generator = torch.Generator().manual_seed(sampler_seed + epoch)
        sequences.append(torch.multinomial(
            torch.ones(len(dataset), dtype=torch.double), epoch_size,
            config.data.train_sampler.get("replacement", True), generator=generator,
        ).tolist())
    for event in observed:
        sample = event["sample"]
        epoch = sample.get("sample_sampler_epoch", 0)
        position = sample["sample_global_position"]
        consumed_position = event["global_step"] * world_size + event["rank"]
        if divmod(consumed_position, epoch_size) != (epoch, position) or consumed_position >= total:
            raise ValueError("Training sample cursor disagrees with the checkpoint history")
        idx = sequences[epoch][position]
        if str(dataset.indices_list.iloc[idx].pdb_id) != sample["pdb_id"]:
            raise ValueError("Sampler reconstruction disagrees with a recorded training example")
    sequence = [index for epoch in sequences for index in epoch][:total]
    return sequence, sampler_seed


def prepare(args):
    dataset, config = make_dataset(args.training_config, load_msa=False)
    if (args.output_dir / "manifest.json").exists():
        raise ValueError("Selection already exists; launch it or use another output directory")
    checkpoint_step = args.checkpoint_step
    if checkpoint_step is None:
        if args.after is None:
            raise ValueError("prepare requires --after or --checkpoint-step to bound training exposure")
        journal = read_complete_jsonl(args.after.parent / "checkpoints.jsonl")
        matches = [r for r in journal if Path(r["path"]).resolve() == args.after.resolve()]
        if not matches:
            raise ValueError("Checkpoint has no completed entry in checkpoints.jsonl")
        checkpoint_step = int(matches[-1]["step"])
    logs = sorted(args.training_config.parent.joinpath("logs").glob("data_samples_rank*.jsonl"))
    observed = [r for path in logs for r in read_complete_jsonl(path) if r["global_step"] <= checkpoint_step]
    sequence, sampler_seed = reconstruct_training_draws(dataset, config, observed, checkpoint_step)
    total = len(sequence)
    counts = collections.Counter(sequence)
    structure_counts = collections.Counter(str(dataset.indices_list.iloc[i].pdb_id) for i in sequence)
    cases = []
    for index, row in dataset.indices_list.iterrows():
        cases.append({
            "name": f"{row.pdb_id}_{index:05d}", "index": int(index), "pdb_id": str(row.pdb_id),
            "training_exposures": counts[index], "structure_training_exposures": structure_counts[str(row.pdb_id)],
            "full_tokens": int(row.num_tokens), "interface_type": str(row.mol_type_group),
            "crop_seed": int.from_bytes(hashlib.sha256(f"training-crop-101:{index}".encode()).digest()[:4], "little"),
        })
    # Select distinct seen structures, then one actually sampled interface per structure.
    # Selection never uses native geometry, prediction metrics, or size thresholds.
    candidates = collections.defaultdict(list)
    for case in cases:
        if case["training_exposures"]:
            candidates[case["pdb_id"]].append(case)
    rng = np.random.default_rng(args.selection_seed)
    names = sorted(candidates)
    if not 0 < args.count <= len(names):
        raise ValueError("count must be positive and no larger than the seen structure population")
    chosen = rng.choice(names, size=args.count, replace=False).tolist()
    cases = [candidates[name][int(rng.integers(len(candidates[name])))] for name in chosen]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = args.output_dir / "training_samples_snapshot.json"
    write_json(snapshot, {"checkpoint_step": checkpoint_step, "source_logs": [str(p) for p in logs],
                          "records": observed})
    source_paths = [args.training_config, Path(dataset.indices_fpath), snapshot]
    msa_config = config.data.msa.to_dict()
    msa_config.update(config.data[config.data.train_sets[0]].get("msa", {}))
    manifest = {
        "training_config": str(args.training_config.resolve()),
        "sources_sha256": {str(p.resolve()): file_hash(p) for p in source_paths},
        "training_samples": total, "logged_samples_verified": len(observed),
        "checkpoint_step": checkpoint_step, "sampler_seed": sampler_seed,
        "after_checkpoint": str(args.after.resolve()) if args.after else None,
        "msa_config": msa_config,
        "seen_structures": len(structure_counts), "seen_interfaces": len(counts),
        "structures": int(dataset.indices_list.pdb_id.nunique()), "cases": cases,
        "selection": {"count": args.count, "seed": args.selection_seed,
                      "method": "uniform distinct seen structures; uniform actually sampled interface within each"},
        "crop_config": dict(dataset.cropping_configs),
        "protocol": (
            "Selected seen training structures; one deterministic training crop per selected interface; "
            "complete indexed peptide; saved training MSA settings; no templates; "
            "native coordinates erased before featurization; "
            "no labels passed to inference"
        ),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "cases"}, indent=2), flush=True)


def select_predictions(scores):
    """Select confidence-ranked and oracle poses over every seed and sample."""
    top = min(scores, key=lambda s: (-s["ranking_score"], s.get("seed", 0), s["sample"]))
    best = min(scores, key=lambda s: (s["peptide_rmsd"], s.get("seed", 0), s["sample"]))
    return {"samples": scores, "top1": top, "best": best}


def predict_seeds(runner, features, atoms, metadata, seeds, samples, description):
    """Predict one seed at a time, retaining coordinates in seed-major order."""
    import torch
    from protenix.utils.seed import seed_everything

    coordinates, scores = [], []
    for seed in seeds:
        print(f"Predicting {description}, seed {seed}, {samples} samples", flush=True)
        seed_everything(seed, deterministic=False)
        # Each seed and checkpoint receives its own copy of the same clean features.
        prediction = runner.predict({"input_feature_dict": copy.deepcopy(features)})
        coords = prediction["coordinate"].detach().float().cpu().numpy()
        if len(coords) != samples or len(prediction["summary_confidence"]) != samples:
            raise ValueError(f"Seed {seed}: expected {samples} predictions")
        for sample, coordinate in enumerate(coords):
            score = score_interface(atoms, features, metadata, coordinate)
            rank = float(prediction["summary_confidence"][sample]["ranking_score"])
            if not np.isfinite(rank):
                raise ValueError(f"Seed {seed}, sample {sample}: nonfinite ranking_score")
            scores.append({"seed": seed, "sample": sample, "ranking_score": rank, **score})
        coordinates.append(coords)
        del prediction
        torch.cuda.empty_cache()
    return select_predictions(scores), np.concatenate(coordinates, axis=0)


def worker(args):
    import torch
    from runner.inference import InferenceRunner
    from scripts.evaluation.evaluate_peptide_pose import configure_gpu_memory_limit, model_configs

    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    run_path = args.output_dir / "run_config.json"
    run_hash = file_hash(run_path) if run_path.exists() else None
    configure_gpu_memory_limit()
    dataset, config = make_dataset(Path(manifest["training_config"]))
    cases = manifest["cases"][:args.limit] if args.limit else manifest["cases"]
    cases = cases[args.worker_id::args.workers]
    runners = {}
    for label, checkpoint in (("before", args.before), ("after", args.after)):
        model_args = SimpleNamespace(
            output_dir=args.output_dir / label, checkpoint=checkpoint, cycles=args.cycles,
            steps=args.steps, samples=args.samples, chunk_size=args.chunk_size,
        )
        runners[label] = InferenceRunner(model_configs(
            model_args, args.output_dir / "unused.json", bool(config.data.msa.enable_prot_msa)
        ))
    for case in cases:
        destination = args.output_dir / "cases" / f"{case['name']}.json"
        if destination.exists():
            prior = json.loads(destination.read_text())
            if (
                prior.get("status") == "complete" and prior.get("run_sha256") == run_hash
                and destination.with_suffix(".npz").is_file()
                and file_hash(destination.with_suffix(".npz")) == prior.get("coordinates_sha256")
            ):
                continue
        start = time.time()
        result = {**case, "run_sha256": run_hash}
        try:
            features, atoms, metadata = make_crop(dataset, case["index"], case["crop_seed"])
            result["crop"] = metadata
            saved = {
                "native": atoms.coord, "source_atom_indices": atoms.peptide_source_atom_index,
                "prediction_seeds": np.repeat(np.asarray(args.seeds, dtype=np.int64), args.samples),
                "prediction_samples": np.tile(np.arange(args.samples), len(args.seeds)),
            }
            for label, runner in runners.items():
                result[label], saved[label] = predict_seeds(
                    runner, features, atoms, metadata, args.seeds, args.samples, f"{case['name']} ({label})",
                )
            if feature_digest(features) != metadata["feature_sha256"]:
                raise ValueError("Inference mutated the shared input features")
            coordinate_path = destination.with_suffix(".npz")
            coordinate_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(coordinate_path, **saved)
            result.update(status="complete", coordinates_sha256=file_hash(coordinate_path))
            print(
                f"{case['name']} n={metadata['tokens']} msa={metadata['msa']['rows']} "
                f"predictions={len(args.seeds) * args.samples} per checkpoint "
                f"seen={case['training_exposures']} "
                f"peptide {result['before']['top1']['peptide_rmsd']:.3f} -> "
                f"{result['after']['top1']['peptide_rmsd']:.3f} A; "
                f"receptor {result['before']['top1']['receptor_rmsd']:.3f} -> "
                f"{result['after']['top1']['receptor_rmsd']:.3f} A",
                flush=True,
            )
        except Exception as exc:
            result.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            print(f"FAILED {case['name']}: {result['error']}", flush=True)
            torch.cuda.empty_cache()
        result["elapsed_seconds"] = time.time() - start
        write_json(destination, result)


def summarize(output_dir):
    """Publish paired metrics without treating failed predictions as zero RMSD."""
    manifest = json.loads((output_dir / "manifest.json").read_text())
    requested = manifest["cases"]
    config_path = output_dir / "run_config.json"
    run = json.loads(config_path.read_text()) if config_path.exists() else {}
    seeds = run.get("seeds", [run.get("seed", 101)])
    samples = run.get("samples", 5)
    if run.get("limit"):
        requested = requested[:run["limit"]]
    results = []
    for case in requested:
        path = output_dir / "cases" / f"{case['name']}.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
    complete = [r for r in results if r["status"] == "complete"]
    groups = {
        "all_interfaces": lambda r: True,
        "seen_interfaces": lambda r: r["training_exposures"] > 0,
        "seen_structures": lambda r: r["structure_training_exposures"] > 0,
        "unseen_structures": lambda r: r["structure_training_exposures"] == 0,
        "protein_interfaces": lambda r: r["interface_type"] == "ligand_prot",
        "nucleic_acid_interfaces": lambda r: r["interface_type"] == "ligand_nuc",
    }
    summary = {
        "requested_interfaces": len(requested), "complete_interfaces": len(complete),
        "failed_interfaces": len(results) - len(complete), "pending_interfaces": len(requested) - len(results),
        "training_seen_structures": manifest["seen_structures"],
        "training_seen_interfaces": manifest["seen_interfaces"], "groups": {},
        "seeds": seeds, "samples_per_seed": samples, "samples_expected_per_model": len(seeds) * samples,
    }
    for name, predicate in groups.items():
        rows = [r for r in complete if predicate(r)]
        denominator = sum(predicate(r) for r in requested)
        group = {"requested": denominator, "paired_complete": len(rows)}
        for label in ("before", "after"):
            metrics = {}
            for selection in ("top1", "best"):
                for metric in ("peptide_rmsd", "peptide_shape_rmsd", "receptor_rmsd"):
                    values = [r[label][selection][metric] for r in rows]
                    metrics[f"{selection}_{metric}"] = {
                        "mean": statistics.mean(values) if values else None,
                        "median": statistics.median(values) if values else None,
                        "success_counts": {str(t): sum(x <= t for x in values) for t in (2, 5)},
                    }
            group[label] = metrics
        deltas = [r["after"]["top1"]["peptide_rmsd"] - r["before"]["top1"]["peptide_rmsd"] for r in rows]
        group["top1_peptide_improved"] = sum(d < 0 for d in deltas)
        group["top1_peptide_worsened"] = sum(d > 0 for d in deltas)
        summary["groups"][name] = group
    write_json(output_dir / "summary.json", summary)
    fields = ["name", "pdb_id", "training_exposures", "structure_training_exposures", "status"]
    fields += [f"{label}_{selection}_{metric}" for label in ("before", "after") for selection in ("top1", "best")
               for metric in ("seed", "sample", "peptide_rmsd", "peptide_shape_rmsd", "receptor_rmsd")]
    temporary = output_dir / "per_interface.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {key: result[key] for key in fields[:5]}
            if result["status"] == "complete":
                for label in ("before", "after"):
                    for selection in ("top1", "best"):
                        for metric in ("seed", "sample", "peptide_rmsd", "peptide_shape_rmsd", "receptor_rmsd"):
                            selected = result[label][selection]
                            row[f"{label}_{selection}_{metric}"] = (
                                selected.get("seed", seeds[0]) if metric == "seed" else selected[metric]
                            )
            writer.writerow(row)
    temporary.replace(output_dir / "per_interface.csv")
    lines = [
        "# Training-set reconstruction: official versus finetuned Protenix v2", "",
        f"Completed paired interfaces: {len(complete)}/{len(requested)}. "
        f"Failed: {summary['failed_interfaces']}; pending: {summary['pending_interfaces']}.", "",
        f"The same deterministic {manifest['crop_config']['crop_size']}-token training crop "
        "and prediction seeds are used for both checkpoints. "
        "Peptides retain their custom ligand atom representations. "
        f"Protein MSAs: {manifest.get('msa_config', {}).get('enable_prot_msa', False)}. "
        "No templates, native poses, or coordinate labels are passed to inference. "
        "Native coordinates determine the training crop; "
        "this is a training-distribution reconstruction test, not a blind full-assembly docking benchmark.", "",
        "Each metric evaluates the indexed receptor–peptide interface. Peptide RMSD uses receptor-only "
        "alignment (protein CA or nucleic-acid C4′), equivalent chain copies, and custom ligand graph "
        "symmetry. Shape RMSD additionally superposes the peptide itself. Top1 uses model ranking_score; "
        "best uses the native peptide RMSD. Means/medians below use paired completed cases only; failures "
        "and pending cases remain in each requested denominator. Each selected structure contributes "
        "one interface and receives equal weight.", "",
        f"Prediction seeds: {', '.join(map(str, seeds))}; {samples} samples per seed, "
        f"{len(seeds) * samples} predictions per interface per checkpoint. "
        f"Top1 and best-of-{len(seeds) * samples} pool all seeds. "
        "Seed and sample IDs are recorded for every prediction and selected pose.", "",
        f"Sampler reconstructed from seed {manifest.get('sampler_seed', 42)} and verified against "
        f"{manifest['logged_samples_verified']} logged "
        f"samples: {manifest['training_samples']} training draws, {manifest['seen_interfaces']} distinct interfaces, "
        f"{manifest['seen_structures']}/{manifest['structures']} structures seen. These crops are reproducible "
        "draws from the training crop policy, not a replay of the exact stochastic crops used by training.", "",
        "Crops missing the indexed peptide or fewer than three receptor alignment atoms are redrawn "
        "with successive fixed seeds (at most 64 attempts). Selection requires the complete indexed peptide. "
        "This conditioning uses no prediction outcomes and never replaces a selected structure.", "",
    ]
    for name, group in summary["groups"].items():
        lines += [f"## {name}", "", f"Paired complete: {group['paired_complete']}/{group['requested']}.", "",
                  "| Metric | Before | After |", "|---|---:|---:|"]
        for metric in ("top1_peptide_rmsd", "best_peptide_rmsd", "top1_peptide_shape_rmsd", "top1_receptor_rmsd"):
            for stat in ("mean", "median"):
                values = [group[label][metric][stat] for label in ("before", "after")]
                formatted = [f"{v:.3f} Å" if v is not None else "pending" for v in values]
                lines.append(f"| {stat} {metric} | {' | '.join(formatted)} |")
        lines += ["", f"Top-ranked peptide RMSD improves on {group['top1_peptide_improved']} interfaces "
                  f"and worsens on {group['top1_peptide_worsened']}.", ""]
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    return summary


def launch(args):
    """Run disjoint workers on free GPUs, preserving completed paired artifacts."""
    from scripts.evaluation.evaluate_peptide_pose import input_environment

    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("after_checkpoint") and str(args.after.resolve()) != manifest["after_checkpoint"]:
        raise ValueError("The selected training history belongs to a different checkpoint")
    for path, expected in manifest["sources_sha256"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Training source changed: {path}")
    model_files = list((REPO / "protenix/model").rglob("*.py")) + [REPO / "runner/inference.py"]
    contract = {
        "manifest_sha256": file_hash(manifest_path),
        "before": str(args.before.resolve()), "before_sha256": file_hash(args.before),
        "after": str(args.after.resolve()), "after_sha256": file_hash(args.after),
        "seeds": args.seeds, "samples": args.samples, "steps": args.steps, "cycles": args.cycles,
        "samples_expected_per_model": len(args.seeds) * args.samples,
        "chunk_size": args.chunk_size, "limit": args.limit,
        "input_environment": input_environment(), "evaluator_sha256": file_hash(__file__),
        "gpu_memory_fraction": os.environ.get("PROTENIX_EVAL_GPU_MEMORY_FRACTION"),
        "model_sources_sha256": hashlib.sha256(json.dumps(
            {str(p.relative_to(REPO)): file_hash(p) for p in model_files}, sort_keys=True
        ).encode()).hexdigest(),
    }
    path = args.output_dir / "run_config.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Run inputs, checkpoints, code or settings changed; use another output directory")
    write_json(path, contract)
    processes = []
    handles = []
    gpu_ids = args.gpus.split(",")
    try:
        for worker_id, gpu in enumerate(gpu_ids):
            log = args.output_dir / f"worker_{worker_id}.log"
            handle = log.open("a")
            handles.append(handle)
            command = [
                sys.executable, "-B", "-m", "scripts.evaluation.evaluate_peptide_training", "worker",
                "--training-config", str(args.training_config), "--output-dir", str(args.output_dir),
                "--before", str(args.before), "--after", str(args.after),
                "--worker-id", str(worker_id), "--workers", str(len(gpu_ids)),
                "--seeds", ",".join(map(str, args.seeds)),
            ]
            for option in ("samples", "steps", "cycles", "chunk_size", "limit"):
                command += ["--" + option.replace("_", "-"), str(getattr(args, option))]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
            process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
            processes.append(process)
        write_json(args.output_dir / "processes.json", {
            "coordinator_pid": os.getpid(), "workers": [
                {"pid": p.pid, "gpu": gpu} for p, gpu in zip(processes, gpu_ids)
            ],
        })
        while any(p.poll() is None for p in processes):
            status = summarize(args.output_dir)
            print(json.dumps({k: v for k, v in status.items() if k != "groups"}), flush=True)
            time.sleep(30)
        status = summarize(args.output_dir)
        write_json(args.output_dir / "exit_status.json", {
            "worker_exitcodes": [p.returncode for p in processes], "finished_at": time.time(),
            "complete": status["pending_interfaces"] == status["failed_interfaces"] == 0,
        })
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()


def interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "launch", "worker", "summarize"])
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--checkpoint-step", type=int, help="Zero-based saved step for exposure reconstruction")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--samples", type=int, default=5, help="Predictions per seed per interface (default: 5)")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=64)
    seed_options = parser.add_mutually_exclusive_group()
    seed_options.add_argument("--seed", type=int, help="Legacy single prediction seed (default: 101)")
    seed_options.add_argument("--seeds", help="Comma-separated prediction seeds (default: 101)")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--selection-seed", type=int, default=20260923)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    try:
        args.seeds = ([int(value) for value in args.seeds.split(",")] if args.seeds is not None
                      else [args.seed if args.seed is not None else 101])
    except ValueError:
        parser.error("seeds must be a nonempty comma-separated list of integers")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("seeds must be unique integers in [0, 2**32)")
    if any(getattr(args, name) <= 0 for name in ("samples", "steps", "cycles", "chunk_size", "workers", "count")):
        parser.error("sampling settings, worker count and structure count must be positive")
    if args.limit < 0 or not 0 <= args.worker_id < args.workers:
        parser.error("invalid limit or worker index")
    gpu_ids = args.gpus.split(",")
    if not all(gpu_ids) or len(gpu_ids) != len(set(gpu_ids)):
        parser.error("gpus must list distinct GPU IDs")
    if args.stage in {"launch", "worker"} and (args.before is None or args.after is None):
        parser.error("before and after checkpoints are required for prediction")
    os.chdir(REPO)
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "summarize":
        summarize(args.output_dir)
    elif args.stage == "launch":
        signal.signal(signal.SIGTERM, interrupt)
        with (args.output_dir / ".evaluation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            launch(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
