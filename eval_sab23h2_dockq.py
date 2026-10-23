#!/usr/bin/env python3
"""Evaluate SAb23H2 Protenix predictions with DockQ.

The target names use the convention:
    pdb_id_immune_chain_immune_chain_or_NA_other_protein_chain

Predicted CIFs use generated chain IDs, and native CIF chain IDs can differ
from the target-name chain IDs. To identify immune-vs-other interfaces, this
script maps input JSON sequences to native and model CIF chains, then checks
DockQ's model-to-native mapping before filtering per-interface scores.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Polypeptide import protein_letters_3to1


BASE_DIR = Path("/mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2")
NATIVE_DIR = Path("/mnt/project_abd/klp/benchmark/SAb23H2/cif.files.native")
INPUT_JSON = (
    BASE_DIR
    / "px_output_finetune_7999_ema_0.999_template_5seeds"
    / "input_jsons/sab23h2_final_updated.json"
)
DEFAULT_OUT_DIR = BASE_DIR / "dockq_eval_immune_interfaces"
DEFAULT_DOCKQ_BIN = Path("/mnt/project_abd/klp/miniconda3/envs/protenix/bin/DockQ")

RUNS = [
    ("original", "px_output_original_protenix_v2_template_5seeds"),
    ("20260525_999_ema", "px_output_finetune_20260525_999_ema_0.999_template_5seeds"),
    ("1999_ema", "px_output_finetune_1999_ema_0.999_template_5seeds"),
    ("3999_ema", "px_output_finetune_3999_ema_0.999_template_5seeds"),
    ("5999_ema", "px_output_finetune_5999_ema_0.999_template_5seeds"),
    ("7999_ema", "px_output_finetune_7999_ema_0.999_template_5seeds"),
    ("20260531_999_ema", "px_output_finetune_20260531_999_ema_0.999_template_5seeds"),
    ("20260531_2999_ema", "px_output_finetune_20260531_2999_ema_0.999_template_5seeds"),
    ("20260531_4999_ema", "px_output_finetune_20260531_4999_ema_0.999_template_5seeds"),
    ("20260525_999_raw", "px_output_finetune_20260525_999_raw_template_5seeds"),
    ("20260531_999_raw", "px_output_finetune_20260531_999_raw_template_5seeds"),
]

SEED_RE = re.compile(r"seed_(\d+)$")
SAMPLE_RE = re.compile(r"_sample_(\d+)\.cif$")


@dataclass(frozen=True)
class ModelJob:
    run_label: str
    run_dir: str
    target: str
    seed: int
    seed_str: str
    sample: int
    model_cif_path: str
    summary_json_path: str
    native_path: str
    raw_json_path: str
    immune_native_chains: tuple[str, ...]
    other_native_chains: tuple[str, ...]
    input_queries_json: str
    native_chain_roles_json: str
    native_mapping_warnings_json: str


def _seq1(resname: str) -> str:
    return protein_letters_3to1.get(resname.strip().upper(), "X")


def extract_structure_sequences(structure_path: Path) -> dict[str, str]:
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", str(structure_path))
    chain_seqs: dict[str, str] = {}
    for model in structure:
        for chain in model:
            residues = [r for r in chain if r.id[0] == " "]
            seq = "".join(_seq1(r.get_resname()) for r in residues)
            if seq:
                chain_seqs[chain.id] = seq
        break
    return chain_seqs


def extract_native_sequences(native_path: Path) -> dict[str, str]:
    return extract_structure_sequences(native_path)


def load_input_entries(input_json: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(input_json.read_text())
    return {entry["name"]: entry for entry in data}


def target_role_queries(target: str, entry: dict[str, Any]) -> list[dict[str, Any]]:
    seqs = []
    for idx, seq_entry in enumerate(entry.get("sequences", [])):
        protein = seq_entry.get("proteinChain")
        if not protein:
            continue
        seq = protein.get("sequence", "")
        count = int(protein.get("count", 1) or 1)
        for _ in range(count):
            seqs.append((idx, seq))

    parts = target.split("_")
    if len(parts) != 4:
        raise ValueError(f"Unexpected target name format: {target}")

    if len(seqs) == 2 or parts[2] == "NA":
        target_chains = [parts[1], parts[3]]
        role_by_pos = ["immune", "other"]
    elif len(seqs) == 3:
        target_chains = [parts[1], parts[2], parts[3]]
        role_by_pos = ["immune", "immune", "other"]
    else:
        raise ValueError(f"Cannot infer chain roles for {target}: {len(seqs)} protein chains")

    if len(role_by_pos) != len(seqs):
        raise ValueError(
            f"Role count mismatch for {target}: roles={len(role_by_pos)} chains={len(seqs)}"
        )
    return [
        {
            "role": role_by_pos[pos],
            "input_index": seq_idx,
            "target_chain": target_chains[pos],
            "sequence": seq,
            "input_length": len(seq),
        }
        for pos, (seq_idx, seq) in enumerate(seqs)
    ]


def sequence_match_details(query: str, candidate: str) -> dict[str, Any]:
    is_exact = query == candidate
    is_subsequence = False
    if is_exact:
        return {
            "score": 1.0,
            "is_exact_match": True,
            "is_subsequence_match": True,
        }
    ratio = difflib.SequenceMatcher(None, query, candidate, autojunk=False).ratio()
    shorter = min(len(query), len(candidate))
    longer = max(len(query), len(candidate))
    if shorter >= 30 and (query in candidate or candidate in query):
        # Native benchmark CIFs can contain only resolved fragments. If one
        # sequence is a complete substring of the other, treat it as a strong
        # chain identity signal while still ranking near-full matches higher.
        is_subsequence = True
        ratio = max(ratio, 0.90 + 0.10 * shorter / longer)
    return {
        "score": ratio,
        "is_exact_match": False,
        "is_subsequence_match": is_subsequence,
    }


def sequence_match_score(query: str, native: str) -> float:
    return float(sequence_match_details(query, native)["score"])


def _queries_without_sequences(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {k: v for k, v in query.items() if k != "sequence"}
        for query in queries
    ]


def _assign_queries_to_chain_sequences(
    target: str,
    queries: list[dict[str, Any]],
    chain_seqs: dict[str, str],
    source: str,
) -> dict[str, Any]:
    candidates: list[tuple[float, int, str, dict[str, Any]]] = []
    for q_idx, query in enumerate(queries):
        for chain_id, chain_seq in chain_seqs.items():
            details = sequence_match_details(query["sequence"], chain_seq)
            candidates.append((float(details["score"]), q_idx, chain_id, details))
    candidates.sort(reverse=True)

    used_queries: set[int] = set()
    used_chains: set[str] = set()
    assignments: list[dict[str, Any]] = []
    for score, q_idx, chain_id, details in candidates:
        if q_idx in used_queries or chain_id in used_chains:
            continue
        query = queries[q_idx]
        assignments.append(
            {
                "source": source,
                "role": query["role"],
                "input_index": query["input_index"],
                "target_chain": query["target_chain"],
                "chain": chain_id,
                "score": round(score, 6),
                "input_length": len(query["sequence"]),
                "structure_length": len(chain_seqs[chain_id]),
                "is_exact_match": bool(details["is_exact_match"]),
                "is_subsequence_match": bool(details["is_subsequence_match"]),
            }
        )
        used_queries.add(q_idx)
        used_chains.add(chain_id)
        if len(used_queries) == len(queries):
            break

    if len(assignments) != len(queries):
        raise ValueError(
            f"Could not map all input chains to {source} chains for {target}: "
            f"{len(assignments)}/{len(queries)}"
        )

    low = [a for a in assignments if a["score"] < 0.65]
    if low:
        raise ValueError(f"Low-confidence {source} chain mapping for {target}: {low}")

    assignments = sorted(assignments, key=lambda x: x["input_index"])
    warnings = [
        {
            "input_index": a["input_index"],
            "target_chain": a["target_chain"],
            "chain": a["chain"],
            "score": a["score"],
            "reason": f"{source}_sequence_match_below_0.90",
        }
        for a in assignments
        if a["score"] < 0.90
    ]

    immune = sorted(a["chain"] for a in assignments if a["role"] == "immune")
    other = sorted(a["chain"] for a in assignments if a["role"] == "other")
    return {
        "target": target,
        "source": source,
        "chain_lengths": {k: len(v) for k, v in chain_seqs.items()},
        "input_queries": _queries_without_sequences(queries),
        "assignments": assignments,
        "immune_chains": immune,
        "other_chains": other,
        "warnings": warnings,
    }


def map_roles_to_structure(
    target: str, entry: dict[str, Any], structure_path: Path, source: str
) -> dict[str, Any]:
    chain_seqs = extract_structure_sequences(structure_path)
    mapping = _assign_queries_to_chain_sequences(
        target,
        target_role_queries(target, entry),
        chain_seqs,
        source,
    )
    mapping["structure_path"] = str(structure_path)
    return mapping


def map_roles_to_native(
    target: str, entry: dict[str, Any], native_path: Path
) -> dict[str, Any]:
    mapping = map_roles_to_structure(target, entry, native_path, source="native")
    mapping["native_path"] = str(native_path)
    mapping["native_chain_lengths"] = mapping["chain_lengths"]
    mapping["immune_native_chains"] = mapping["immune_chains"]
    mapping["other_native_chains"] = mapping["other_chains"]
    return mapping


def map_queries_to_model(job: ModelJob) -> dict[str, Any]:
    queries = json.loads(job.input_queries_json)
    chain_seqs = extract_structure_sequences(Path(job.model_cif_path))
    mapping = _assign_queries_to_chain_sequences(
        job.target,
        queries,
        chain_seqs,
        source="model",
    )
    mapping["structure_path"] = job.model_cif_path
    return mapping


def parse_summary_confidence(path: str) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text())
    return {
        "ptm": data.get("ptm"),
        "iptm": data.get("iptm"),
        "ranking_score": data.get("ranking_score"),
        "num_recycles": data.get("num_recycles"),
        "has_clash": data.get("has_clash"),
        "chain_ptm": json.dumps(data.get("chain_ptm", [])),
        "chain_iptm": json.dumps(data.get("chain_iptm", [])),
        "chain_pair_iptm": json.dumps(data.get("chain_pair_iptm", [])),
    }


def build_jobs(
    base_dir: Path,
    native_dir: Path,
    input_json: Path,
    out_dir: Path,
    run_filter: set[str] | None,
    limit: int | None,
) -> tuple[list[ModelJob], list[dict[str, Any]]]:
    entries = load_input_entries(input_json)
    chain_mappings: dict[str, dict[str, Any]] = {}
    jobs: list[ModelJob] = []

    for run_label, run_dir_name in RUNS:
        if run_filter and run_label not in run_filter:
            continue
        run_dir = base_dir / run_dir_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Prediction run directory not found: {run_dir}")
        for target_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
            target = target_dir.name
            if target == "run_logs":
                continue
            if target not in entries:
                continue
            native_path = native_dir / f"{target}.cif"
            if not native_path.exists():
                raise FileNotFoundError(f"Native CIF not found for {target}: {native_path}")
            if target not in chain_mappings:
                chain_mappings[target] = map_roles_to_native(target, entries[target], native_path)
            mapping = chain_mappings[target]
            input_queries = target_role_queries(target, entries[target])
            roles_json = json.dumps(mapping["assignments"], sort_keys=True)
            input_queries_json = json.dumps(input_queries, sort_keys=True)
            native_warnings_json = json.dumps(mapping.get("warnings", []), sort_keys=True)

            for seed_dir in sorted(p for p in target_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")):
                m = SEED_RE.match(seed_dir.name)
                if not m:
                    continue
                seed = int(m.group(1))
                pred_dir = seed_dir / "predictions"
                for model_cif in sorted(pred_dir.glob(f"{target}_sample_*.cif")):
                    sm = SAMPLE_RE.search(model_cif.name)
                    if not sm:
                        continue
                    sample = int(sm.group(1))
                    summary_json = pred_dir / f"{target}_summary_confidence_sample_{sample}.json"
                    raw_json = (
                        out_dir
                        / "raw_dockq_json"
                        / run_label
                        / target
                        / f"seed_{seed}_sample_{sample}.json"
                    )
                    jobs.append(
                        ModelJob(
                            run_label=run_label,
                            run_dir=run_dir_name,
                            target=target,
                            seed=seed,
                            seed_str=f"seed_{seed}",
                            sample=sample,
                            model_cif_path=str(model_cif),
                            summary_json_path=str(summary_json),
                            native_path=str(native_path),
                            raw_json_path=str(raw_json),
                            immune_native_chains=tuple(mapping["immune_native_chains"]),
                            other_native_chains=tuple(mapping["other_native_chains"]),
                            input_queries_json=input_queries_json,
                            native_chain_roles_json=roles_json,
                            native_mapping_warnings_json=native_warnings_json,
                        )
                    )
                    if limit and len(jobs) >= limit:
                        return jobs, list(chain_mappings.values())

    return jobs, list(chain_mappings.values())


def run_dockq(dockq_bin: str, job: ModelJob, force: bool, timeout: int) -> dict[str, Any]:
    raw_json = Path(job.raw_json_path)
    raw_json.parent.mkdir(parents=True, exist_ok=True)

    if force or not raw_json.exists() or raw_json.stat().st_size == 0:
        with tempfile.NamedTemporaryFile(prefix="dockq_", suffix=".json", delete=False) as tmp:
            tmp_json = Path(tmp.name)
        try:
            cmd = [dockq_bin, job.model_cif_path, job.native_path, "--json", str(tmp_json)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                return {
                    "dockq_status": "ERROR",
                    "dockq_error": f"DockQ exited {result.returncode}: {result.stderr[-1000:]}",
                }
            shutil.move(str(tmp_json), raw_json)
        except subprocess.TimeoutExpired:
            return {
                "dockq_status": "ERROR",
                "dockq_error": f"DockQ timed out after {timeout}s",
            }
        except Exception as exc:
            return {"dockq_status": "ERROR", "dockq_error": str(exc)}
        finally:
            if tmp_json.exists():
                tmp_json.unlink()

    try:
        data = json.loads(raw_json.read_text())
    except Exception as exc:
        return {"dockq_status": "ERROR", "dockq_error": f"Failed to parse DockQ JSON: {exc}"}

    return extract_dockq_metrics(data, job)


def parse_best_mapping(mapping_str: str) -> dict[str, str]:
    if not mapping_str or ":" not in mapping_str:
        return {}
    model_side, native_side = mapping_str.split(":", 1)
    if len(model_side) != len(native_side):
        return {}
    return dict(zip(model_side, native_side))


def weighted_average(interfaces: list[dict[str, Any]]) -> dict[str, Any]:
    if not interfaces:
        return {
            "dockq": None,
            "irmsd": None,
            "lrmsd": None,
            "fnat": None,
            "nat_total": 0,
            "model_total": 0,
        }

    total_nat = sum(float(x.get("nat_total") or 0) for x in interfaces)
    if total_nat <= 0:
        weights = [1.0 for _ in interfaces]
        total = float(len(interfaces))
    else:
        weights = [float(x.get("nat_total") or 0) for x in interfaces]
        total = total_nat

    def avg(key: str) -> float:
        return sum(float(x.get(key) or 0) * w for x, w in zip(interfaces, weights)) / total

    return {
        "dockq": round(avg("DockQ"), 6),
        "irmsd": round(avg("iRMSD"), 6),
        "lrmsd": round(avg("LRMSD"), 6),
        "fnat": round(avg("fnat"), 6),
        "nat_total": int(sum(int(x.get("nat_total") or 0) for x in interfaces)),
        "model_total": int(sum(int(x.get("model_total") or 0) for x in interfaces)),
    }


def extract_dockq_metrics(data: dict[str, Any], job: ModelJob) -> dict[str, Any]:
    best = data.get("best_result") or {}
    model_to_native = parse_best_mapping(data.get("best_mapping_str", ""))
    immune = set(job.immune_native_chains)
    other = set(job.other_native_chains)
    native_assignments = json.loads(job.native_chain_roles_json)
    native_by_input_index = {
        int(a["input_index"]): a["chain"]
        for a in native_assignments
    }

    mapping_checks: list[dict[str, Any]] = []
    mapping_warnings = json.loads(job.native_mapping_warnings_json)
    immune_model_chains: list[str] = []
    other_model_chains: list[str] = []
    model_chain_roles_json = ""
    try:
        model_mapping = map_queries_to_model(job)
        model_chain_roles_json = json.dumps(model_mapping["assignments"], sort_keys=True)
        immune_model_chains = model_mapping["immune_chains"]
        other_model_chains = model_mapping["other_chains"]
        mapping_warnings.extend(model_mapping.get("warnings", []))
        for assignment in model_mapping["assignments"]:
            model_chain = assignment["chain"]
            input_index = int(assignment["input_index"])
            expected_native = native_by_input_index.get(input_index)
            dockq_native = model_to_native.get(model_chain)
            ok = bool(expected_native and dockq_native == expected_native)
            check = {
                "input_index": input_index,
                "role": assignment["role"],
                "model_chain": model_chain,
                "expected_native_chain": expected_native,
                "dockq_native_chain": dockq_native,
                "ok": ok,
            }
            mapping_checks.append(check)
            if not ok:
                mapping_warnings.append(
                    {
                        "input_index": input_index,
                        "model_chain": model_chain,
                        "expected_native_chain": expected_native,
                        "dockq_native_chain": dockq_native,
                        "reason": "dockq_best_mapping_disagrees_with_sequence_mapping",
                    }
                )
    except Exception as exc:
        mapping_warnings.append(
            {
                "reason": "model_sequence_mapping_failed",
                "error": str(exc),
            }
        )

    all_interfaces: list[dict[str, Any]] = []
    immune_other_interfaces: list[dict[str, Any]] = []
    per_interface: dict[str, Any] = {}

    for iface_name, iface in sorted(best.items()):
        model_chain1 = iface.get("chain1", "")
        model_chain2 = iface.get("chain2", "")
        native_chain1 = model_to_native.get(model_chain1)
        native_chain2 = model_to_native.get(model_chain2)
        if (not native_chain1 or not native_chain2) and len(iface_name) == 2:
            native_chain1, native_chain2 = iface_name[0], iface_name[1]

        record = dict(iface)
        record["interface_name"] = iface_name
        record["native_chain1"] = native_chain1
        record["native_chain2"] = native_chain2
        all_interfaces.append(record)

        is_immune_other = (
            native_chain1 in immune
            and native_chain2 in other
            or native_chain2 in immune
            and native_chain1 in other
        )
        if is_immune_other:
            immune_other_interfaces.append(record)

        per_interface[iface_name] = {
            "native_chain1": native_chain1,
            "native_chain2": native_chain2,
            "model_chain1": model_chain1,
            "model_chain2": model_chain2,
            "DockQ": round(float(iface.get("DockQ") or 0), 6),
            "iRMSD": round(float(iface.get("iRMSD") or 0), 6),
            "LRMSD": round(float(iface.get("LRMSD") or 0), 6),
            "fnat": round(float(iface.get("fnat") or 0), 6),
            "nat_total": iface.get("nat_total", 0),
            "model_total": iface.get("model_total", 0),
            "is_immune_other": is_immune_other,
        }

    full = weighted_average(all_interfaces)
    immune_other = weighted_average(immune_other_interfaces)
    if not immune_other_interfaces:
        status = "NO_IMMUNE_OTHER_INTERFACE"
        error = "No DockQ interface matched immune-vs-other native chain roles"
    else:
        status = "OK"
        error = ""

    mapping_status = "OK" if not mapping_warnings else "WARN"

    return {
        "dockq_status": status,
        "dockq_error": error,
        "mapping_status": mapping_status,
        "mapping_warnings": json.dumps(mapping_warnings, sort_keys=True),
        "dockq_mapping_checks": json.dumps(mapping_checks, sort_keys=True),
        "model_chain_roles": model_chain_roles_json,
        "immune_model_chains": ",".join(immune_model_chains),
        "other_model_chains": ",".join(other_model_chains),
        "best_mapping_str": data.get("best_mapping_str", ""),
        "global_dockq": data.get("GlobalDockQ"),
        "full_dockq": full["dockq"],
        "full_irmsd": full["irmsd"],
        "full_lrmsd": full["lrmsd"],
        "full_fnat": full["fnat"],
        "full_nat_total": full["nat_total"],
        "full_model_total": full["model_total"],
        "immune_other_dockq": immune_other["dockq"],
        "immune_other_irmsd": immune_other["irmsd"],
        "immune_other_lrmsd": immune_other["lrmsd"],
        "immune_other_fnat": immune_other["fnat"],
        "immune_other_nat_total": immune_other["nat_total"],
        "immune_other_model_total": immune_other["model_total"],
        "n_interfaces": len(all_interfaces),
        "n_immune_other_interfaces": len(immune_other_interfaces),
        "immune_other_interfaces": ",".join(x["interface_name"] for x in immune_other_interfaces),
        "per_interface_scores": json.dumps(per_interface, sort_keys=True),
    }


def eval_job(args: tuple[str, ModelJob, bool, int]) -> dict[str, Any]:
    dockq_bin, job, force, timeout = args
    base = {
        "run_label": job.run_label,
        "run_dir": job.run_dir,
        "target": job.target,
        "seed": job.seed,
        "seed_str": job.seed_str,
        "sample": job.sample,
        "model_cif_path": job.model_cif_path,
        "summary_json_path": job.summary_json_path,
        "native_path": job.native_path,
        "raw_json_path": job.raw_json_path,
        "input_chain_roles": json.dumps(_queries_without_sequences(json.loads(job.input_queries_json)), sort_keys=True),
        "immune_native_chains": ",".join(job.immune_native_chains),
        "other_native_chains": ",".join(job.other_native_chains),
        "native_chain_roles": job.native_chain_roles_json,
        "native_mapping_warnings": job.native_mapping_warnings_json,
    }
    base.update(parse_summary_confidence(job.summary_json_path))
    metrics = run_dockq(dockq_bin, job, force=force, timeout=timeout)
    base.update(metrics)
    return base


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    vals = []
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            vals.append(float(value))
        except (TypeError, ValueError):
            continue
    return vals


def summarize_group(rows: list[dict[str, Any]], keys: dict[str, str]) -> dict[str, Any]:
    out = dict(keys)
    out["n_models"] = len(rows)
    out["n_ok"] = sum(1 for r in rows if r.get("dockq_status") == "OK")
    out["n_error"] = sum(1 for r in rows if r.get("dockq_status") == "ERROR")
    out["n_no_immune_other_interface"] = sum(
        1 for r in rows if r.get("dockq_status") == "NO_IMMUNE_OTHER_INTERFACE"
    )
    out["n_mapping_warn"] = sum(1 for r in rows if r.get("mapping_status") == "WARN")
    for metric in ["immune_other_dockq", "full_dockq", "ranking_score", "iptm"]:
        vals = numeric_values(rows, metric)
        prefix = metric
        if vals:
            out[f"{prefix}_mean"] = round(statistics.fmean(vals), 6)
            out[f"{prefix}_median"] = round(statistics.median(vals), 6)
            out[f"{prefix}_min"] = round(min(vals), 6)
            out[f"{prefix}_max"] = round(max(vals), 6)
        else:
            out[f"{prefix}_mean"] = ""
            out[f"{prefix}_median"] = ""
            out[f"{prefix}_min"] = ""
            out[f"{prefix}_max"] = ""

    scored = [r for r in rows if r.get("immune_other_dockq") not in (None, "")]
    if scored:
        best = max(scored, key=lambda r: float(r["immune_other_dockq"]))
        out["best_target"] = best["target"]
        out["best_seed"] = best["seed"]
        out["best_sample"] = best["sample"]
        out["best_immune_other_dockq"] = best["immune_other_dockq"]
        out["best_model_cif_path"] = best["model_cif_path"]
        vals = [float(r["immune_other_dockq"]) for r in scored]
        out["immune_other_high_ge_0.80"] = sum(v >= 0.80 for v in vals)
        out["immune_other_medium_0.49_0.80"] = sum(0.49 <= v < 0.80 for v in vals)
        out["immune_other_acceptable_0.23_0.49"] = sum(0.23 <= v < 0.49 for v in vals)
        out["immune_other_incorrect_lt_0.23"] = sum(v < 0.23 for v in vals)
    else:
        out["best_target"] = ""
        out["best_seed"] = ""
        out["best_sample"] = ""
        out["best_immune_other_dockq"] = ""
        out["best_model_cif_path"] = ""
        out["immune_other_high_ge_0.80"] = 0
        out["immune_other_medium_0.49_0.80"] = 0
        out["immune_other_acceptable_0.23_0.49"] = 0
        out["immune_other_incorrect_lt_0.23"] = 0
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summaries(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    by_run: dict[str, list[dict[str, Any]]] = {}
    by_run_target: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_run.setdefault(row["run_label"], []).append(row)
        by_run_target.setdefault((row["run_label"], row["target"]), []).append(row)

    checkpoint_rows = [
        summarize_group(group, {"run_label": run_label})
        for run_label, group in sorted(by_run.items())
    ]
    target_rows = [
        summarize_group(group, {"run_label": run_label, "target": target})
        for (run_label, target), group in sorted(by_run_target.items())
    ]
    write_csv(out_dir / "checkpoint_summary.csv", checkpoint_rows)
    write_csv(out_dir / "per_target_checkpoint_summary.csv", target_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--native-dir", type=Path, default=NATIVE_DIR)
    parser.add_argument("--input-json", type=Path, default=INPUT_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dockq-bin", type=Path, default=DEFAULT_DOCKQ_BIN)
    parser.add_argument("--runs", default=None, help="Comma-separated run labels, default all")
    parser.add_argument("--num-workers", "-n", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N jobs")
    parser.add_argument("--force", action="store_true", help="Re-run DockQ even if raw JSON exists")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dockq_bin.exists():
        raise FileNotFoundError(f"DockQ binary not found: {args.dockq_bin}")

    run_filter = set(args.runs.split(",")) if args.runs else None
    unknown = run_filter - {label for label, _ in RUNS} if run_filter else set()
    if unknown:
        raise ValueError(f"Unknown run labels: {sorted(unknown)}")

    jobs, mappings = build_jobs(
        args.base_dir,
        args.native_dir,
        args.input_json,
        args.out_dir,
        run_filter=run_filter,
        limit=args.limit,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "chain_mapping.csv", mappings)

    print(f"DockQ binary: {args.dockq_bin}")
    print(f"Jobs: {len(jobs)}")
    print(f"Targets with chain mappings: {len(mappings)}")
    print(f"Output: {args.out_dir}")
    if args.dry_run:
        return 0

    t0 = time.time()
    rows: list[dict[str, Any]] = []
    workers = max(1, args.num_workers)
    if workers == 1:
        for i, job in enumerate(jobs, 1):
            rows.append(eval_job((str(args.dockq_bin), job, args.force, args.timeout)))
            if i % 25 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] rows={len(rows)} elapsed={time.time() - t0:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(eval_job, (str(args.dockq_bin), job, args.force, args.timeout)): job
                for job in jobs
            }
            for i, future in enumerate(as_completed(future_map), 1):
                job = future_map[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    rows.append(
                        {
                            "run_label": job.run_label,
                            "target": job.target,
                            "seed": job.seed,
                            "sample": job.sample,
                            "model_cif_path": job.model_cif_path,
                            "native_path": job.native_path,
                            "dockq_status": "EXECUTOR_ERROR",
                            "dockq_error": str(exc),
                        }
                    )
                if i % 100 == 0 or i == len(jobs):
                    ok = sum(1 for r in rows if r.get("dockq_status") == "OK")
                    err = sum(1 for r in rows if r.get("dockq_status") in {"ERROR", "EXECUTOR_ERROR"})
                    rate = i / max(time.time() - t0, 1e-6)
                    print(
                        f"[{i}/{len(jobs)}] OK={ok} errors={err} rate={rate:.2f}/s "
                        f"elapsed={(time.time() - t0) / 60:.1f}m",
                        flush=True,
                    )

    rows.sort(key=lambda r: (r.get("run_label", ""), r.get("target", ""), int(r.get("seed", 0)), int(r.get("sample", 0))))
    per_model_csv = args.out_dir / "per_model_dockq.csv"
    write_csv(per_model_csv, rows)
    write_summaries(args.out_dir, rows)

    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_dir": str(args.base_dir),
        "native_dir": str(args.native_dir),
        "input_json": str(args.input_json),
        "dockq_bin": str(args.dockq_bin),
        "num_workers": workers,
        "n_jobs": len(jobs),
        "elapsed_seconds": round(time.time() - t0, 3),
        "per_model_csv": str(per_model_csv),
        "checkpoint_summary_csv": str(args.out_dir / "checkpoint_summary.csv"),
        "per_target_checkpoint_summary_csv": str(args.out_dir / "per_target_checkpoint_summary.csv"),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved: {per_model_csv}")
    print(f"Saved: {args.out_dir / 'checkpoint_summary.csv'}")
    print(f"Saved: {args.out_dir / 'per_target_checkpoint_summary.csv'}")
    print(f"Elapsed: {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
