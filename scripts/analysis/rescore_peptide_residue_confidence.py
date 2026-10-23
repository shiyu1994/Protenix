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

"""Recompute confidence on fixed peptide poses with optional residue tokenization.

Preserves the original CCD atom inventory and bonds. Residue mode promotes peptide
entities before rebuilding token/frame/MSA features; ligand mode retains the
original input representation. An optional checkpoint selects the scoring weights.
Only the diffusion sampling call is replaced, with saved predicted coordinates.
No native coordinates enter the model. Modified residues use native Protenix
tokenization (generally one token per atom); receptor MSA files are retained and
peptide MSAs are query-only unless --peptide-msa-map supplies alignments.
Run one independent worker per visible GPU.
"""

import argparse
import copy
import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protenix.data.inference import infer_dataloader
from protenix.data.constants import STD_RESIDUES
from protenix.data.inference.json_to_feature import SampleDictToFeatures
from protenix.data.msa.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.msa.msa_utils import MSA_GAP_IDX
from protenix.data.tokenizer import AtomArrayTokenizer
from protenix.utils.seed import seed_everything
from runner.inference import InferenceRunner
from scripts.evaluation.evaluate_peptide_pose import model_configs
from scripts.evaluation.peptide_pose_inputs import polymer_spec, write_json
from scripts.evaluation.peptide_pose_metrics import read_prediction


def set_single_atom_residue_centres(atoms, peptide_chains):
    """Use the sole observed atom for a partial residue without its C-alpha.

    No atom or coordinate is added. Its one-atom frame remains invalid under
    the standard featurizer. Multi-atom residues without a centre still fail.
    """
    fallback = np.zeros(len(atoms), dtype=bool)
    for token in AtomArrayTokenizer(atoms).tokenize():
        ids = token.atom_indices
        if atoms.chain_id[ids[0]] not in peptide_chains or atoms.centre_atom_mask[ids].any():
            continue
        if len(ids) != 1:
            raise ValueError("Multi-atom peptide residue lacks its standard centre atom")
        atoms.centre_atom_mask[ids[0]] = 1
        fallback[ids[0]] = True
    atoms.set_annotation("rescore_centre_fallback", fallback)
    return atoms


class ResidueFeatures(SampleDictToFeatures):
    """Reuse original CCD chemistry while rebuilding peptide polymer annotations."""

    def __init__(self, single_sample_dict, **kwargs):
        self.ligand_references = single_sample_dict.get("ligand_reference_overrides", {})
        residue_input = copy.deepcopy(single_sample_dict)
        residue_input.pop("ligand_reference_overrides", None)
        super().__init__(residue_input, **kwargs)
        peptide_chains = set(single_sample_dict["rescore_peptide_chains"])
        for index, item in enumerate(single_sample_dict["sequences"], 1):
            body = next(iter(item.values()))
            ids = set(body["id"])
            if ids & peptide_chains:
                if not ids <= peptide_chains:
                    raise ValueError("Peptide entity also contains non-peptide chains")
                self.entity_poly_type[str(index)] = "polypeptide(L)"

    def get_atom_array(self):
        if not self.ligand_references:
            return super().get_atom_array()
        atoms = self.add_bonds_between_entities(self.build_full_atom_array())
        atoms = self.mse_to_met(atoms)
        keep = np.ones(len(atoms), dtype=bool)
        for chain, reference in self.ligand_references.items():
            allowed = {tuple(k) for k in reference["source_keys"]}
            ids = np.flatnonzero(atoms.chain_id == chain)
            keep[ids] = [(int(atoms.res_id[i]), str(atoms.atom_name[i])) in allowed for i in ids]
            if int(keep[ids].sum()) != len(allowed):
                raise ValueError(f"Cannot restore residue atom inventory: {chain}")
        atoms = self.add_atom_array_attributes(atoms[keep], self.entity_poly_type)
        return set_single_atom_residue_centres(atoms, self.ligand_references)


def msa_bioassembly(sample, peptide_chains, peptide_msa_map=None):
    """Promote peptide queries and attach supplied MSAs, preserving receptor inputs."""
    result = copy.deepcopy(sample["sequences"])
    for i, item in enumerate(result):
        body = next(iter(item.values()))
        if set(body["id"]) & set(peptide_chains):
            residues = body["ligand"].removeprefix("CCD_").split("_")
            kind, spec = polymer_spec("polypeptide(L)", residues)
            if peptide_msa_map is not None:
                entry = peptide_msa_map[spec["sequence"]]
                paths = {key: entry[key] for key in ("unpairedMsaPath", "pairedMsaPath") if entry.get(key)}
                if not paths:
                    raise ValueError(f"No peptide MSA paths for {spec['sequence']}")
                for path in paths.values():
                    if not Path(path).is_file():
                        raise FileNotFoundError(path)
                spec.update(paths)
            result[i] = {kind: {**spec, "id": body["id"], "count": body["count"]}}
    return result


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def coordinate_batch(atoms, paths, reference=None):
    """Match each saved atom by chain, residue number and name, never by native geometry."""
    keys = list(zip(atoms.chain_id.tolist(), atoms.res_id.tolist(), atoms.atom_name.tolist()))
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate featurized atom identities")
    batch = []
    for path in paths:
        saved = read_prediction(path)
        if reference and reference.get("prediction_atom_mapping"):
            from scripts.evaluation.peptide_training_aligned import restore_prediction_keys

            saved = restore_prediction_keys(saved, reference)
        if set(keys) != saved.keys():
            raise ValueError(f"Atom inventory mismatch: {path}; missing={set(keys) - saved.keys()}; "
                             f"extra={saved.keys() - set(keys)}")
        batch.append(np.asarray([saved[k] for k in keys], dtype=np.float32))
    return np.stack(batch)


def audit_features(atoms, data, peptides, representation="residue"):
    """Record actual token counts and molecular types, including nonstandard residues."""
    features = data["input_feature_dict"]
    mapping = features["atom_to_token_idx"].numpy()
    result = {"representation": representation, "total_atoms": len(atoms), "total_tokens": int(data["N_token"]),
              "msa_rows": int(data["N_msa"]), "peptides": {}}
    if hasattr(atoms, "rescore_centre_fallback"):
        result["centre_atom_fallbacks"] = [
            {"chain": str(atoms.chain_id[i]), "position": int(atoms.res_id[i]),
             "residue": str(atoms.res_name[i]), "centre": str(atoms.atom_name[i]),
             "reason": "sole observed atom; missing standard centre; frame invalid"}
            for i in np.flatnonzero(atoms.rescore_centre_fallback)
        ]
    for chain in peptides:
        mask = atoms.chain_id == chain
        result["peptides"][chain] = {
            "atoms": int(mask.sum()), "tokens": len(np.unique(mapping[mask])),
            "protein_atoms": int(features["is_protein"].numpy()[mask].sum()),
            "residues": [],
        }
        if "msa" in features:
            columns = np.unique(mapping[mask])
            msa = features["msa"].numpy()[:, columns]
            result["peptides"][chain]["msa_non_gap_rows"] = int(np.any(msa != MSA_GAP_IDX, axis=1).sum())
            result["peptides"][chain]["msa_distinct_rows"] = len(np.unique(msa, axis=0))
            result["peptides"][chain]["msa_sha256"] = hashlib.sha256(msa.tobytes()).hexdigest()
        if representation == "ligand":
            if not np.all(features["is_ligand"].numpy()[mask] == 1):
                raise ValueError(f"Peptide not classified as ligand: {chain}")
            if len(np.unique(mapping[mask])) != int(mask.sum()):
                raise ValueError(f"Ligand peptide not tokenized per atom: {chain}")
        for pos in np.unique(atoms.res_id[mask]):
            selected = mask & (atoms.res_id == pos)
            residue = str(atoms.res_name[selected][0])
            n_tokens = len(np.unique(mapping[selected]))
            if representation == "residue" and residue in STD_RESIDUES and n_tokens != 1:
                raise ValueError(f"Standard peptide residue not tokenized as residue: {chain}/{pos}/{residue}")
            result["peptides"][chain]["residues"].append({
                "position": int(pos), "name": residue, "tokens": n_tokens,
                "mol_types": sorted(set(atoms.mol_type[selected].tolist())),
            })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Weights for confidence rescoring; defaults to the source run")
    parser.add_argument("--representation", choices=["residue", "ligand"], default="residue",
                        help="Use rebuilt residue features or the original ligand features")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--case")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--peptide-msa-map", type=Path,
                        help="JSON mapping exact peptide sequences to paired/unpaired MSA paths")
    parser.add_argument("--shared-trunk", action="store_true",
                        help="Use one feature/trunk seed (101) for every saved pose of a complex")
    args = parser.parse_args()
    if args.peptide_msa_map and args.representation != "residue":
        parser.error("--peptide-msa-map requires --representation residue")
    peptide_msa_map = None
    peptide_msa_sha256 = None
    if args.peptide_msa_map:
        raw_map = args.peptide_msa_map.read_bytes()
        peptide_msa_map = json.loads(raw_map)
        fingerprint = hashlib.sha256(raw_map)
        for sequence, entry in sorted(peptide_msa_map.items()):
            for key in ("unpairedMsaPath", "pairedMsaPath"):
                if entry.get(key):
                    fingerprint.update(sequence.encode())
                    fingerprint.update(key.encode())
                    fingerprint.update(Path(entry[key]).read_bytes())
        peptide_msa_sha256 = fingerprint.hexdigest()
    root, out = args.run.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    run = json.loads((root / "run_config.json").read_text())
    checkpoint = (args.checkpoint or Path(run["checkpoint"])).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    progress = json.loads((root / "prediction_manifest.json").read_text())
    # Large assemblies dominate trunk cost; distribute them first to avoid an idle
    # GPU waiting for a worker with several large cases at the end of its queue.
    queues = [[] for _ in range(args.workers)]
    loads = [0 for _ in range(args.workers)]
    for case in sorted(manifest["cases"], key=lambda c: (-c["validation"]["input_tokens"], c["name"])):
        worker = min(range(args.workers), key=lambda i: (loads[i], i))
        queues[worker].append(case)
        loads[worker] += case["validation"]["input_tokens"] ** 3
    cases = queues[args.worker]
    if args.case:
        cases = [c for c in manifest["cases"] if c["name"] == args.case]
        if not cases:
            raise ValueError(f"Unknown requested case: {args.case}")
    samples = []
    for case in cases:
        item = json.loads((root / case["input"]).read_text())[0]
        item["rescore_peptide_chains"] = case["peptide_chains"]
        samples.append(item)
    input_path = out / f"worker_{args.worker}_inputs.json"
    write_json(input_path, samples)
    config_args = SimpleNamespace(checkpoint=checkpoint, output_dir=out,
                                  cycles=run["cycles"], steps=run["steps"], samples=run["samples"],
                                  chunk_size=run["chunk_size"])
    configs = model_configs(config_args, input_path, True)
    write_json(out / f"worker_{args.worker}_config.json", configs.to_dict())
    dataset = infer_dataloader.InferenceDataset(configs)
    runner = None if args.prepare_only else InferenceRunner(configs)
    if runner:
        # Sampling is bypassed; avoid building caches used exclusively by diffusion.
        runner.model.enable_diffusion_shared_vars_cache = False
    make_msa = InferenceMSAFeaturizer.make_msa_feature
    errors = []
    for index, case in enumerate(cases):
        name = case["name"]
        data = atoms = audit = None
        trunk_cache = []
        trunk_versions = []

        def shared_trunk(**kwargs):
            if not trunk_cache:
                trunk_cache.extend(original_trunk(**kwargs))
                trunk_versions.extend(t._version for t in trunk_cache)
            return tuple(trunk_cache)

        original_trunk = runner.model.get_pairformer_output if runner else None
        try:
            for seed in ([args.seed] if args.seed else run["seeds"]):
                dest = out / "cases" / name / f"seed_{seed}.json"
                if dest.exists():
                    previous = json.loads(dest.read_text())
                    if (Path(previous["checkpoint"]).resolve() != checkpoint
                            or previous.get("representation", "residue") != args.representation
                            or previous.get("peptide_msa_sha256") != peptide_msa_sha256
                            or previous["shared_trunk"] != args.shared_trunk):
                        raise ValueError(f"Existing result uses different rescoring settings: {dest}")
                    print(f"Already complete {name} seed {seed}", flush=True)
                    continue
                started = time.time()
                feature_seed = 101 if args.shared_trunk else seed
                if data is None or not args.shared_trunk:
                    seed_everything(feature_seed, deterministic=False)
                    if args.representation == "ligand":
                        data, atoms, error = dataset[index]
                    else:
                        assembly = msa_bioassembly(samples[index], case["peptide_chains"], peptide_msa_map)

                        def residue_msa(**kwargs):
                            return make_msa(**{**kwargs, "bioassembly": assembly})

                        with patch.object(infer_dataloader, "SampleDictToFeatures", ResidueFeatures), \
                             patch.object(InferenceMSAFeaturizer, "make_msa_feature", side_effect=residue_msa):
                            data, atoms, error = dataset[index]
                    if error:
                        raise ValueError(error)
                    audit = audit_features(atoms, data, case["peptide_chains"], args.representation)
                    audit["original_total_tokens"] = case["validation"]["input_tokens"]
                    audit["peptide_msa_sha256"] = peptide_msa_sha256
                    if args.representation == "ligand" and audit["total_tokens"] != audit["original_total_tokens"]:
                        raise ValueError(f"Original token count changed: {name}")
                write_json(out / "audits" / name / f"seed_{seed}.json", audit)
                paths, hashes = [], {}
                for sample in range(run["samples"]):
                    rel = Path("predictions") / name / f"seed_{seed}" / "predictions" / f"{name}_sample_{sample}.cif"
                    original = root / rel
                    digest = hashlib.sha256(original.read_bytes()).hexdigest()
                    if digest != progress[name]["files"][str(rel)]:
                        raise ValueError(f"Original prediction changed: {original}")
                    converted = args.converted / rel
                    # Verify coordinates in the exported polymer file against its source.
                    a, b = read_prediction(original), read_prediction(converted)
                    if a.keys() != b.keys() or not all(np.array_equal(a[k], b[k]) for k in a):
                        raise ValueError(f"Converted coordinates differ: {converted}")
                    paths.append(converted)
                    hashes[str(converted)] = hashlib.sha256(converted.read_bytes()).hexdigest()
                reference = json.loads((root / case["reference"]).read_text()) if args.representation == "residue" else None
                coords = coordinate_batch(atoms, paths, reference)
                if args.prepare_only:
                    print(f"Prepared {name}: {audit['total_tokens']} tokens, exact atom mapping", flush=True)
                    break
                fixed = torch.from_numpy(coords).to(runner.device)
                trunk_function = shared_trunk if args.shared_trunk else original_trunk
                with patch.object(runner.model, "sample_diffusion", return_value=fixed) as bypass, \
                     patch.object(runner.model, "get_pairformer_output", side_effect=trunk_function):
                    prediction = runner.predict(data)
                if bypass.call_count != 1 or not torch.equal(prediction["coordinate"], fixed):
                    raise ValueError("Fixed-coordinate bypass failed")
                if args.shared_trunk and [t._version for t in trunk_cache] != trunk_versions:
                    raise ValueError("Confidence evaluation modified shared trunk tensors")
                summaries = jsonable(prediction["summary_confidence"])
                peptide_mask = torch.as_tensor(np.isin(atoms.chain_id, case["peptide_chains"]), device=fixed.device)
                for sample, summary in enumerate(summaries):
                    summary["sample"] = sample
                    summary["peptide_mean_plddt"] = float(
                        prediction["full_data"][sample]["atom_plddt"][peptide_mask].mean().item() * 100
                    )
                if not all(np.isfinite(s["ranking_score"]) for s in summaries):
                    raise ValueError("Nonfinite new ranking score")
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    dest.with_suffix(".npz"),
                    atom_plddt=np.stack([f["atom_plddt"].detach().float().cpu().numpy()
                                         for f in prediction["full_data"]]),
                    atom_resolved_probability=prediction["resolved"].float().softmax(-1)[..., 1].cpu().numpy(),
                    chain_id=atoms.chain_id, residue_id=atoms.res_id, atom_name=atoms.atom_name,
                )
                write_json(dest, {"name": name, "seed": seed, "checkpoint": str(checkpoint),
                                  "source_checkpoint": run["checkpoint"], "representation": args.representation,
                                  "feature_seed": feature_seed, "shared_trunk": args.shared_trunk,
                                  "peptide_msa_sha256": peptide_msa_sha256,
                                  "coordinates_unchanged": True, "feature_audit": audit,
                                  "input_hashes": hashes, "summary_confidence": summaries,
                                  "seconds": time.time() - started})
                print(f"Rescored {name} seed {seed}: {audit['total_tokens']} tokens, "
                      f"{time.time()-started:.1f}s", flush=True)
                del prediction, fixed
                if not args.shared_trunk:
                    data = atoms = None
                gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            error = {"name": name, "traceback": traceback.format_exc()}
            errors.append(error)
            write_json(out / "errors" / f"{name}.json", error)
            print(error["traceback"], flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        data = atoms = None
        trunk_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()
    write_json(out / f"worker_{args.worker}_status.json", {"cases": len(cases), "errors": errors})
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
