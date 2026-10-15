#!/usr/bin/env python3
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

"""Prepare, predict, and score full-assembly protein/peptide ligand test CIFs."""

import argparse
import copy
import csv
import fcntl
import gc
import hashlib
import json
import logging
import math
import os
import tempfile
import time
import traceback
from pathlib import Path

from scripts.evaluation.peptide_pose_inputs import (
    file_hash,
    prepare_case,
    validate_case,
    write_json,
)
from scripts.evaluation.peptide_pose_metrics import read_prediction, score_pose
from scripts.evaluation.peptide_pose_parallel import launch_workers, partition_cases


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = "/mnt/project_abd/shiyu/dataset/full_assembly_peptide_ligand_cif/cif"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def input_environment():
    """Track feature code and local CCD resources for safe reuse."""
    from configs.configs_data import data_configs

    sources = [
        p
        for folder in ("configs", "protenix/config", "protenix/data", "protenix/utils")
        for p in (REPO / folder).rglob("*.py")
    ]
    resources = {}
    for key in ("ccd_components_file", "ccd_components_rdkit_mol_file"):
        path = Path(data_configs[key]).resolve(strict=True)
        stat = path.stat()
        resources[key] = {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return {
        "sources_sha256": digest(
            {str(p.relative_to(REPO)): file_hash(p) for p in sorted(sources)}
        ),
        "ccd_resources": resources,
    }


def prepare(args):
    files = sorted(args.input_dir.resolve().glob("*.cif"))
    if args.ids:
        wanted = set(args.ids.split(","))
        files = [p for p in files if p.stem in wanted]
        if {p.stem for p in files} != wanted:
            raise ValueError("Some requested --ids are absent from the input directory")
    if not files:
        raise ValueError("No input CIFs found")
    msa_map = read_json(args.msa_map) if args.msa_map else None
    signature = {
        "sources": {str(p): file_hash(p) for p in files},
        "preparer_sha256": file_hash(
            Path(__file__).with_name("peptide_pose_inputs.py")
        ),
        "parser_sha256": file_hash(REPO / "protenix/data/core/parser.py"),
        "msa_map": msa_map,
        "input_environment": input_environment(),
    }
    if msa_map:
        signature["msa_hashes"] = {
            str(Path(p).resolve()): file_hash(p)
            for entry in msa_map.values()
            for p in entry.values()
        }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if (
            old["signature"] != signature
            and (args.output_dir / "run_config.json").exists()
        ):
            raise ValueError(
                "Inputs changed in an existing prediction run; use a new --output-dir"
            )
        if old["signature"] == signature and all(
            r["status"] == "ready"
            and all(
                (args.output_dir / p).is_file() and file_hash(args.output_dir / p) == h
                for p, h in r["artifacts"].items()
            )
            for r in old["cases"]
        ):
            print(f"Reusing {len(old['cases'])} verified evaluation inputs", flush=True)
            return old
    cases = []
    for index, path in enumerate(files, 1):
        record = {
            "name": path.stem,
            "source": str(path),
            "source_sha256": file_hash(path),
        }
        try:
            job, reference = prepare_case(path, msa_map)
            validate_case(job, reference)
            input_path = args.output_dir / "inputs" / f"{path.stem}.json"
            ref_path = args.output_dir / "references" / f"{path.stem}.json"
            write_json(input_path, [job])
            write_json(ref_path, reference)
            record.update(
                status="ready",
                input=str(input_path.relative_to(args.output_dir)),
                reference=str(ref_path.relative_to(args.output_dir)),
                peptide_chains=reference["peptide_chains"],
                validation=reference["validation"],
            )
            record["artifacts"] = {
                str(p.relative_to(args.output_dir)): file_hash(p)
                for p in (input_path, ref_path)
            }
        except Exception as exc:
            record.update(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
        cases.append(record)
        print(
            f"[{index}/{len(files)}] {path.stem}: {record['status']} {record.get('error', '')}",
            flush=True,
        )
    manifest = {"signature": signature, "cases": cases, "msa_enabled": bool(msa_map)}
    write_json(manifest_path, manifest)
    write_json(
        args.output_dir / "preparation_failures.json",
        [r for r in cases if r["status"] != "ready"],
    )
    return manifest


def load_manifest(args):
    manifest = read_json(args.output_dir / "manifest.json")
    if args.ids:
        raise ValueError(
            "--ids applies to prepare/all only; predict/score/watch use the saved manifest"
        )
    for path, expected in manifest["signature"].get("msa_hashes", {}).items():
        if file_hash(path) != expected:
            raise ValueError(
                f"Prepared MSA changed: {path}; use a new evaluation directory"
            )
    for record in manifest["cases"]:
        if file_hash(record["source"]) != record["source_sha256"]:
            raise ValueError(
                f"Source CIF changed: {record['name']}; prepare a new evaluation directory"
            )
        if record["status"] == "ready":
            for path, expected in record["artifacts"].items():
                if file_hash(args.output_dir / path) != expected:
                    raise ValueError(f"Prepared artifact changed: {path}")
    return manifest


def prediction_paths(root, name, seed, sample):
    folder = root / "predictions" / name / f"seed_{seed}" / "predictions"
    return (
        folder / f"{name}_sample_{sample}.cif",
        folder / f"{name}_summary_confidence_sample_{sample}.json",
    )


def make_run_config(args, manifest):
    return {
        "model_name": "protenix-v2",
        "checkpoint": str(args.checkpoint.resolve(strict=True)),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "seeds": args.seeds,
        "samples": args.samples,
        "steps": args.steps,
        "cycles": args.cycles,
        "chunk_size": args.chunk_size,
        "msa_enabled": manifest["msa_enabled"],
        "templates_enabled": False,
        "manifest_sha256": file_hash(args.output_dir / "manifest.json"),
        "evaluation_driver_sha256": file_hash(__file__),
        "parallel_driver_sha256": file_hash(
            Path(__file__).with_name("peptide_pose_parallel.py")
        ),
        "input_environment": input_environment(),
        "runner_sha256": digest(
            {
                name: file_hash(REPO / name)
                for name in ("runner/inference.py", "runner/dumper.py")
            }
        ),
        "model_sources_sha256": digest(
            {
                str(p.relative_to(REPO)): file_hash(p)
                for p in sorted((REPO / "protenix/model").rglob("*.py"))
            }
        ),
    }


def model_configs(args, input_json, use_msa):
    from configs.configs_base import configs as base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs as presets
    from protenix.config.config import parse_configs

    defaults = copy.deepcopy({**base, "data": data_configs, **inference_configs})

    def merge(destination, source):
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(destination.get(key), dict):
                merge(destination[key], value)
            else:
                destination[key] = copy.deepcopy(value)

    merge(defaults, presets["protenix-v2"])
    options = {
        "model_name": "protenix-v2",
        "input_json_path": input_json,
        "dump_dir": args.output_dir / "predictions",
        "load_checkpoint_dir": args.checkpoint.resolve(),
        "load_strict": "true",
        "dtype": "bf16",
        "use_msa": str(use_msa).lower(),
        "use_template": "false",
        "use_rna_msa": "false",
        "num_workers": 0,
        "triangle_attention": "torch",
        "triangle_multiplicative": "torch",
        "model.N_cycle": args.cycles,
        "sample_diffusion.N_step": args.steps,
        "sample_diffusion.N_sample": args.samples,
        "infer_setting.chunk_size": args.chunk_size,
        "infer_setting.dynamic_chunk_size": "false",
        "skip_amp.confidence_head": "false",
        "skip_amp.sample_diffusion": "true",
        "need_atom_confidence": "false",
        "sorted_by_ranking_score": "true",
    }
    if any(any(c.isspace() for c in str(value)) for value in options.values()):
        raise ValueError(
            "The repository config parser requires paths without whitespace"
        )
    return parse_configs(
        defaults,
        arg_str=" ".join(f"--{k} {v}" for k, v in options.items()),
        fill_required_with_null=True,
    )


def predict(args, manifest, on_complete=None):
    if manifest["signature"]["input_environment"] != input_environment():
        raise ValueError(
            "Feature code or CCD resources changed; prepare a new evaluation directory"
        )
    config = make_run_config(args, manifest)
    run_path = args.output_dir / "run_config.json"
    if run_path.exists() and read_json(run_path) != config:
        raise ValueError(
            "Checkpoint, inputs, code, or inference settings changed; use a new --output-dir"
        )
    write_json(run_path, config)
    progress_path = args.output_dir / "prediction_manifest.json"
    progress = read_json(progress_path) if progress_path.exists() else {}
    write_json(progress_path, progress)
    cases = [r for r in manifest["cases"] if r["status"] == "ready"]
    pending = []
    for case in cases:
        prior = progress.get(case["name"], {})
        if (
            prior.get("status") == "complete"
            and prior.get("run_sha256") == digest(config)
            and all(
                (args.output_dir / p).is_file() and file_hash(args.output_dir / p) == h
                for p, h in prior["files"].items()
            )
        ):
            if on_complete is not None:
                on_complete(case, prior)
            continue
        pending.append(case)
    if not pending:
        print("All predictions already complete and verified", flush=True)
        return all(r["status"] == "ready" for r in manifest["cases"])
    if args.gpus:
        predict_parallel(
            args, manifest, config, pending, progress, progress_path, on_complete
        )
    else:
        predict_cases(
            args,
            manifest,
            config,
            pending,
            progress_path,
            args.output_dir / "pending_inputs.json",
            on_complete=on_complete,
        )
    progress = read_json(progress_path)
    return all(
        r["status"] == "ready"
        and progress.get(r["name"], {}).get("status") == "complete"
        for r in manifest["cases"]
    )


def predict_parallel(
    args, manifest, config, pending, progress, progress_path, on_complete=None
):
    """Keep shared manifests under one writer while workers own disjoint cases."""
    workers_root = args.output_dir / "workers"
    workers_root.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=workers_root))
    jobs = [
        {
            "gpu": args.gpus[i],
            "cases": shard,
            "progress": run_dir / f"worker_{i}.json",
            "input": run_dir / f"worker_{i}_inputs.json",
            "log": run_dir / f"worker_{i}.log",
        }
        for i, shard in enumerate(partition_cases(pending, len(args.gpus)))
    ]
    write_json(
        run_dir / "assignment.json",
        {
            "run_sha256": digest(config),
            "workers": [
                {
                    "gpu": j["gpu"],
                    "cases": [c["name"] for c in j["cases"]],
                    "log": str(j["log"]),
                }
                for j in jobs
            ],
        },
    )

    def merge_progress():
        changed = False
        completed = []
        for job in jobs:
            if not job["progress"].exists():
                continue
            updates = read_json(job["progress"])
            for case in job["cases"]:
                name = case["name"]
                if name in updates and updates[name] != progress.get(name):
                    progress[name] = updates[name]
                    changed = True
                    if updates[name].get("status") in {"complete", "failed"}:
                        completed.append((case, updates[name]))
        if changed:
            write_json(progress_path, progress)
        if on_complete is not None:
            for case, record in completed:
                on_complete(case, record)

    exitcodes = launch_workers(
        predict_cases, (args, manifest, config), jobs, merge_progress
    )
    unreported = []
    for job, exitcode in zip(jobs, exitcodes):
        reported = read_json(job["progress"]) if job["progress"].exists() else {}
        for case in job["cases"]:
            if case["name"] not in reported:
                progress[case["name"]] = {
                    "status": "failed",
                    "error": (
                        f"GPU {job['gpu']} worker exited with status {exitcode} "
                        f"before recording this case; see {job['log']}"
                    ),
                }
                unreported.append(case)
    write_json(progress_path, progress)
    if on_complete is not None:
        for case in unreported:
            on_complete(case, progress[case["name"]])


def configure_gpu_memory_limit():
    """Optionally cap this evaluation worker's CUDA allocator on a shared GPU."""
    value = os.environ.get("PROTENIX_EVAL_GPU_MEMORY_FRACTION")
    if value is not None:
        import torch

        fraction = float(value)
        if not 0 < fraction <= 1:
            raise ValueError("PROTENIX_EVAL_GPU_MEMORY_FRACTION must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        print(f"Evaluation CUDA allocator limit: {fraction:.0%} of device memory", flush=True)


def predict_cases(
    args, manifest, config, pending, progress_path, input_json, on_complete=None
):
    """Load one model and predict this worker's assigned cases on its visible GPU."""
    import torch
    from protenix.data.inference.infer_dataloader import InferenceDataset
    from protenix.utils.seed import seed_everything
    from runner.inference import InferenceRunner

    progress = read_json(progress_path) if progress_path.exists() else {}
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Prediction requires a CUDA GPU; prepare/score modes use CPUs"
        )
    configure_gpu_memory_limit()
    write_json(
        input_json, [read_json(args.output_dir / r["input"])[0] for r in pending]
    )
    configs = model_configs(args, input_json, manifest["msa_enabled"])
    runner = InferenceRunner(configs)
    dataset = InferenceDataset(configs)
    for index, case in enumerate(pending):
        name = case["name"]
        data = atoms = prediction = None
        try:
            for seed in args.seeds:
                seed_everything(seed=seed, deterministic=False)
                data, atoms, error = dataset[index]
                if error:
                    raise ValueError(error)
                print(
                    f"Predicting {name}, seed {seed}, {int(data['N_token'])} tokens",
                    flush=True,
                )
                # The standard CLI has an advisory 2560-token cutoff. Full assemblies
                # here retain all chains and use configured chunking instead of cropping.
                prediction = runner.predict(data)
                runner.dumper.dump(
                    "",
                    name,
                    seed,
                    prediction,
                    atoms,
                    {
                        k: v
                        for k, v in data["entity_poly_type"].items()
                        if v != "non-polymer"
                    },
                )
                prediction = data = atoms = None
                gc.collect()
                torch.cuda.empty_cache()
            files = {}
            for seed in args.seeds:
                for sample in range(args.samples):
                    cif, confidence = prediction_paths(
                        args.output_dir, name, seed, sample
                    )
                    read_prediction(cif)
                    if not math.isfinite(float(read_json(confidence)["ranking_score"])):
                        raise ValueError("Nonfinite ranking_score")
                    files.update(
                        {
                            str(p.relative_to(args.output_dir)): file_hash(p)
                            for p in (cif, confidence)
                        }
                    )
            progress[name] = {
                "status": "complete",
                "run_sha256": digest(config),
                "files": files,
            }
        except Exception as exc:
            progress[name] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"Prediction failed for {name}: {exc}", flush=True)
        finally:
            prediction = data = atoms = None
            gc.collect()
            torch.cuda.empty_cache()
        write_json(progress_path, progress)
        if on_complete is not None:
            on_complete(case, progress[name])
    return all(progress.get(r["name"], {}).get("status") == "complete" for r in pending)


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True)
                    if isinstance(v, (dict, list))
                    else v
                    for k, v in row.items()
                }
            )


def summarize_cases(cases, rows, samples_expected, thresholds):
    summaries = []
    for case in cases:
        available = [row for row in rows if row["name"] == case["name"]]
        complete = case["status"] == "ready" and len(available) == samples_expected
        summary = {
            "name": case["name"],
            "complete": complete,
            "samples_scored": len(available),
            "samples_expected": samples_expected,
        }
        if available:
            top = sorted(
                available, key=lambda r: (-r["ranking_score"], r["seed"], r["sample"])
            )[0]
            best = min(
                available, key=lambda r: (r["max_peptide_rmsd"], r["seed"], r["sample"])
            )
            summary.update(
                top1_seed=top["seed"],
                top1_sample=top["sample"],
                top1_max_peptide_rmsd=top["max_peptide_rmsd"],
                best_seed=best["seed"],
                best_sample=best["sample"],
                best_max_peptide_rmsd=best["max_peptide_rmsd"],
            )
        for threshold in thresholds:
            for label in ("top1", "best"):
                summary[f"{label}_success_{threshold:g}A"] = bool(
                    complete and summary[f"{label}_max_peptide_rmsd"] <= threshold
                )
        summaries.append(summary)
    rates = {
        key: sum(row[key] for row in summaries) / len(cases)
        for key in summaries[0]
        if "_success_" in key
    }
    return summaries, rates


def score_case(args, case, run, record, cached=None):
    """Score a completed case, reusing live results only after verifying artifacts."""
    report = {
        "name": case["name"],
        "poses": [],
        "peptides": [],
        "errors": [],
        "context": None,
    }
    if case["status"] != "ready":
        report["errors"].append(
            {"name": case["name"], "stage": "prepare", "error": case["error"]}
        )
        return report
    if record.get("status") != "complete" or record.get("run_sha256") != digest(run):
        report["errors"].append(
            {
                "name": case["name"],
                "stage": "predict",
                "error": record.get(
                    "error", "No verified complete prediction for this case"
                ),
            }
        )
        return report
    try:
        ref_path = args.output_dir / case["reference"]
        reference_hash = file_hash(ref_path)
        expected_reference = case.get("artifacts", {}).get(case["reference"])
        if expected_reference is not None and expected_reference != reference_hash:
            raise ValueError(f"Prepared reference changed: {ref_path}")
        report["context"] = digest(
            {
                "run": digest(run),
                "files": record["files"],
                "reference": reference_hash,
                "contact_cutoff": args.contact_cutoff,
                "symmetry": not args.no_atom_symmetry,
                "min_receptor_coverage": args.min_receptor_coverage,
                "scorer": file_hash(
                    Path(__file__).with_name("peptide_pose_metrics.py")
                ),
                "input_reader": file_hash(
                    Path(__file__).with_name("peptide_pose_inputs.py")
                ),
            }
        )
        reference = read_json(ref_path)
    except Exception as exc:
        report["errors"].append(
            {
                "name": case["name"],
                "stage": "score",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return report
    cache = cached if cached and cached["context"] == report["context"] else {}
    cached_poses = {(r["seed"], r["sample"]): r for r in cache.get("poses", [])}
    for seed in run["seeds"]:
        for sample in range(run["samples"]):
            identity = {"name": case["name"], "seed": seed, "sample": sample}
            try:
                path, conf_path = prediction_paths(
                    args.output_dir, case["name"], seed, sample
                )
                for artifact in (path, conf_path):
                    if record["files"].get(
                        str(artifact.relative_to(args.output_dir))
                    ) != file_hash(artifact):
                        raise ValueError(f"Prediction artifact changed: {artifact}")
                if (seed, sample) in cached_poses:
                    report["poses"].append(cached_poses[seed, sample])
                    report["peptides"].extend(
                        r
                        for r in cache["peptides"]
                        if (r["seed"], r["sample"]) == (seed, sample)
                    )
                    continue
                confidence = float(read_json(conf_path)["ranking_score"])
                if not math.isfinite(confidence):
                    raise ValueError("Nonfinite ranking_score")
                result = score_pose(
                    reference,
                    read_prediction(path),
                    args.contact_cutoff,
                    not args.no_atom_symmetry,
                    args.min_receptor_coverage,
                )
                peptides = result.pop("peptides")
                report["poses"].append(
                    {**identity, "ranking_score": confidence, **result}
                )
                report["peptides"].extend({**identity, **row} for row in peptides)
            except Exception as exc:
                report["errors"].append(
                    {
                        **identity,
                        "stage": "score",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return report


def display_case_metrics(report, expected_samples):
    """Flush per-peptide metrics and confidence-selected/oracle RMSDs to the console."""
    name = report["name"]
    for row in report["peptides"]:
        backbone = (
            "n/a"
            if row["backbone_pose_rmsd"] is None
            else f"{row['backbone_pose_rmsd']:.3f} A"
        )
        fnat = "n/a" if row["fnat"] is None else f"{row['fnat']:.3f}"
        print(
            f"[metrics] {name} seed={row['seed']} sample={row['sample']} peptide={row['peptide_chain']}: "
            f"RMSD={row['pose_rmsd']:.3f} A, backbone RMSD={backbone}, fnat={fnat}",
            flush=True,
        )
    rows = report["poses"]
    message = f"[metrics] {name}: scored {len(rows)}/{expected_samples} samples"
    if len(rows) != expected_samples or report["errors"]:
        message += " (incomplete)"
    if rows:
        top = min(rows, key=lambda r: (-r["ranking_score"], r["seed"], r["sample"]))
        best = min(rows, key=lambda r: (r["max_peptide_rmsd"], r["seed"], r["sample"]))
        message += (
            f"; top1 max peptide RMSD={top['max_peptide_rmsd']:.3f} A "
            f"(seed={top['seed']} sample={top['sample']}); "
            f"best max peptide RMSD={best['max_peptide_rmsd']:.3f} A; "
            f"top1 receptor CA RMSD={top['receptor_rmsd']:.3f} A"
        )
    print(message, flush=True)
    for error in report["errors"]:
        sample = (
            f" seed={error['seed']} sample={error['sample']}" if "seed" in error else ""
        )
        print(
            f"[metrics] {name}{sample}: {error['stage']} error: {error['error']}",
            flush=True,
        )


def watch(args):
    """Read completed predictions from an active or older run without writing to it."""
    manifest = load_manifest(args)
    run = read_json(args.output_dir / "run_config.json")
    if run["manifest_sha256"] != file_hash(args.output_dir / "manifest.json"):
        raise ValueError("Prediction run does not match this input manifest")
    expected = len(run["seeds"]) * run["samples"]
    seen = {}
    progress_path = args.output_dir / "prediction_manifest.json"
    print(
        f"Monitoring {len(manifest['cases'])} complexes in {args.output_dir}; Ctrl+C stops this monitor.",
        flush=True,
    )
    while True:
        progress = read_json(progress_path) if progress_path.exists() else {}
        for case in manifest["cases"]:
            record = progress.get(case["name"], {})
            if case["status"] == "ready" and record.get("status") not in {
                "complete",
                "failed",
            }:
                continue
            revision = digest(record)
            if seen.get(case["name"]) == revision:
                continue
            display_case_metrics(score_case(args, case, run, record), expected)
            seen[case["name"]] = revision
        if args.once or len(seen) == len(manifest["cases"]):
            return
        time.sleep(args.poll_interval)


def score(args, manifest, cached_reports=None):
    run = read_json(args.output_dir / "run_config.json")
    if run["manifest_sha256"] != file_hash(args.output_dir / "manifest.json"):
        raise ValueError("Prediction run does not match this input manifest")
    progress = read_json(args.output_dir / "prediction_manifest.json")
    pose_rows, peptide_rows, errors = [], [], []
    expected = len(run["seeds"]) * run["samples"]
    for case in manifest["cases"]:
        record = progress.get(case["name"], {})
        cached = (cached_reports or {}).get(case["name"])
        report = score_case(args, case, run, record, cached)
        if report != cached:
            display_case_metrics(report, expected)
        pose_rows.extend(report["poses"])
        peptide_rows.extend(report["peptides"])
        errors.extend(report["errors"])
    summaries, rates = summarize_cases(
        manifest["cases"], pose_rows, expected, args.thresholds
    )
    output = args.output_dir / "scores"
    output.mkdir(exist_ok=True)
    write_csv(output / "poses.csv", pose_rows)
    write_csv(output / "peptides.csv", peptide_rows)
    write_csv(output / "complexes.csv", summaries)
    write_json(output / "errors.json", errors)
    summary = {
        "requested_complexes": len(manifest["cases"]),
        "prepared_complexes": sum(r["status"] == "ready" for r in manifest["cases"]),
        "complete_complexes": sum(r["complete"] for r in summaries),
        "samples_expected_per_complex": expected,
        "samples_scored": len(pose_rows),
        "success_rates_all_requested_complexes": rates,
        "success_definition": "All peptides in a complex have receptor-aligned heavy-atom RMSD <= threshold; incomplete cases fail",
        "ranking": "top1 by global ranking_score across all seeds; best is oracle minimum worst-peptide RMSD",
        "alignment": "receptor CA only; equivalent receptor permutations and one-to-one peptide assignment",
        "atom_symmetry": not args.no_atom_symmetry,
        "contact_cutoff_angstrom": args.contact_cutoff,
        "min_receptor_ca_coverage": args.min_receptor_coverage,
        "scorer_sha256": file_hash(Path(__file__).with_name("peptide_pose_metrics.py")),
        "run": run,
    }
    write_json(output / "summary.json", summary)
    print(
        json.dumps({k: v for k, v in summary.items() if k != "run"}, indent=2),
        flush=True,
    )
    return not errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        choices=("all", "prepare", "predict", "score", "watch"),
        default="all",
    )
    parser.add_argument("--input-dir", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument(
        "--output-dir", type=Path, default=REPO.parent / "peptide_pose_eval_v2"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            os.environ.get(
                "PROTENIX_ROOT_DIR",
                "/mnt/project_abd/public_databases/protenix.20260101",
            )
        )
        / "checkpoint/protenix-v2.pt",
    )
    parser.add_argument(
        "--ids", help="Comma-separated CIF stems for a separate small run"
    )
    parser.add_argument("--seeds", default="101", help="Comma-separated prediction seeds (default: 101)")
    parser.add_argument(
        "--gpus",
        help="Comma-separated physical GPU IDs/UUIDs, one independent worker per GPU",
    )
    parser.add_argument("--samples", type=int, default=5, help="Predictions per seed per complex (default: 5)")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--msa-map",
        type=Path,
        help="JSON: receptor sequence -> existing paired/unpaired MSA paths",
    )
    parser.add_argument("--contact-cutoff", type=float, default=5.0)
    parser.add_argument(
        "--thresholds", default="2,5", help="Pose success thresholds in angstroms"
    )
    parser.add_argument("--min-receptor-coverage", type=float, default=0.9)
    parser.add_argument("--no-atom-symmetry", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--once",
        action="store_true",
        help="With watch, print currently completed cases and exit",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=5.0, help="Seconds between watch updates"
    )
    args = parser.parse_args()
    if args.once and args.stage != "watch":
        parser.error("--once applies only to watch")
    if not math.isfinite(args.poll_interval) or args.poll_interval <= 0:
        parser.error("poll-interval must be finite and positive")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error(
            "Launch once with --gpus; torchrun would create competing output writers"
        )
    if args.gpus is not None:
        args.gpus = [value.strip() for value in args.gpus.split(",")]
        if any(
            not value or any(c.isspace() for c in value) for value in args.gpus
        ) or len(set(args.gpus)) != len(args.gpus):
            parser.error("gpus must be a nonempty list of unique GPU IDs/UUIDs")
    try:
        args.seeds = [int(value) for value in args.seeds.split(",")]
        args.thresholds = [float(value) for value in args.thresholds.split(",")]
    except ValueError:
        parser.error("seeds/thresholds must be nonempty comma-separated numbers")
    if (
        not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
        or any(seed < 0 or seed >= 2**32 for seed in args.seeds)
    ):
        parser.error("seeds must be unique integers in [0, 2**32)")
    settings = [
        args.samples,
        args.steps,
        args.cycles,
        args.chunk_size,
        args.contact_cutoff,
        *args.thresholds,
    ]
    if any(not math.isfinite(value) or value <= 0 for value in settings):
        parser.error("sampling settings and thresholds must be positive")
    if not 0 < args.min_receptor_coverage <= 1:
        parser.error("min-receptor-coverage must be in (0, 1]")
    args.output_dir = args.output_dir.resolve()
    if (
        args.output_dir == args.input_dir.resolve()
        or args.input_dir.resolve() in args.output_dir.parents
    ):
        parser.error("output-dir must be outside the native CIF directory")
    if args.dry_run:
        print(json.dumps(vars(args), default=str, indent=2))
        return
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.stage == "watch":
        try:
            watch(args)
        except KeyboardInterrupt:
            print("Stopped metric monitor.", flush=True)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # All stages write manifests or reports. Hold the lock until process exit.
    with (args.output_dir / ".evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another evaluation process is writing this output directory")
        run_stages(args)


def run_stages(args):
    manifest = (
        prepare(args) if args.stage in {"all", "prepare"} else load_manifest(args)
    )
    if args.stage == "prepare":
        if any(r["status"] != "ready" for r in manifest["cases"]):
            raise SystemExit("Some inputs failed; inspect preparation_failures.json")
        return
    reports = {}

    def on_complete(case, record):
        run = read_json(args.output_dir / "run_config.json")
        report = score_case(args, case, run, record)
        reports[case["name"]] = report
        display_case_metrics(report, len(run["seeds"]) * run["samples"])

    if args.stage in {"all", "predict"}:
        success = predict(args, manifest, on_complete)
        if args.stage == "predict" and not success:
            raise SystemExit(
                "Prediction incomplete; inspect prediction_manifest.json and preparation_failures.json"
            )
    if args.stage in {"all", "score"} and not score(args, manifest, reports):
        raise SystemExit(
            "Evaluation incomplete; failures remain in the denominator. See scores/errors.json"
        )


if __name__ == "__main__":
    main()
