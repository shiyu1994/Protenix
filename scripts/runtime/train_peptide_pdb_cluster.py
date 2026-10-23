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

"""Train the 32-GPU peptide/PDB mixture and record runtime evidence."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import torch
import yaml
from ml_collections import ConfigDict

PEPTIDE = "weightedPDB_before2109_wopb_nometalc_0925"
PDB = "weightedPDB_before210930_v20260101"
CONFIDENCE_LOSSES = ("plddt_loss", "pde_loss", "resolved_loss", "pae_loss")


def validate_config(c):
    """Fail before allocating GPUs if any requested setting is lost."""
    assert c.model_name == "protenix-v2"
    assert c.load_params_only and c.load_strict
    assert Path(c.load_checkpoint_path).name == "protenix-v2.pt"
    assert c.distributed_strategy == "ddp" and c.optimizer_sharding == "none"
    assert c.iters_to_accumulate == c.diffusion_batch_size == c.diffusion_chunk_size == 1
    assert c.train_crop_size == 1024
    assert list(c.data.train_sets) == [PEPTIDE, PDB]
    assert list(c.data.train_sampler.train_sample_weights) == [0.5, 0.5]
    assert c.data.msa.enable_prot_msa and c.data.msa.max_size.train == 1024
    assert c.data.msa.sample_cutoff.train == 1024
    assert c.data.query_only_msa_probability == 0
    assert c.lr == c.adam.lr == 1e-4
    assert c.lr_scheduler == "cosine_annealing"
    assert c.max_steps == 100000 and c.warmup_steps == 3000 and c.min_lr_ratio == 0.1
    assert not c.train_confidence_only and c.model.confidence_head.stop_gradient
    assert c.loss.resolution.min == -1.0
    assert c.loss.weight.alpha_confidence == 1e-4
    assert c.loss.weight.alpha_pae == c.loss.weight.alpha_except_pae == 1.0
    assert c.use_wandb and c.wandb_id
    peptide_root = Path(c.data[PEPTIDE].base_info.bioassembly_dict_dir).parent
    recovery = json.loads((peptide_root / "msa_recovery_validation.json").read_text())
    assert recovery["all_previously_accepted_preserved"]
    assert recovery["all_output_hashes_verified"]
    assert recovery["mapping_version"] == 1
    index_path = Path(c.data[PEPTIDE].base_info.indices_fpath)
    assert hashlib.sha256(index_path.read_bytes()).hexdigest() == recovery["index_sha256"]
    for name in c.data.train_sets:
        d = c.data[name]
        assert d.cropping_configs.crop_size == 1024
        for key in ("indices_fpath", "bioassembly_dict_dir", "mmcif_dir"):
            assert Path(d.base_info[key]).exists(), d.base_info[key]
    for path in [c.load_checkpoint_path, *c.data.msa.prot_msadir_raw_paths,
                 *c.data.msa.prot_seq_or_filename_to_msadir_jsons]:
        assert Path(path).exists(), path


def scalar(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().item() if x.numel() == 1 else list(x.shape)
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return scalar(x[0])
    return x


def check_data(c, root):
    """Exercise the actual feature loader for each dataset, including nonquery MSAs."""
    from protenix.data.pipeline.dataset import get_datasets

    datasets, _ = get_datasets(c, str(root / "preflight_errors"))
    records, masses = [], {}
    offset = 0
    for name, dataset in zip(datasets.dataset_names, datasets.datasets):
        mass = float(datasets.merged_datapoint_weights[offset:offset + len(dataset)].sum())
        masses[name] = mass
        assert abs(mass - 0.5) < 1e-5, masses
        offset += len(dataset)
        found = False
        for i in range(min(24, len(dataset))):
            batch = dataset[i]
            features = batch["input_feature_dict"]
            msa = features["msa"]
            resolution = scalar(features["resolution"])
            record = {"dataset": name, "index": i, "basic": {
                k: scalar(v) for k, v in batch["basic"].items()
                if k in ("pdb_id", "N_token", "N_msa", "N_msa_prot_pair", "N_msa_prot_unpair")
            }, "msa_shape": list(msa.shape), "resolution": resolution}
            records.append(record)
            print(json.dumps(record), flush=True)
            assert msa.shape[-1] <= 1024
            if name == PEPTIDE:
                assert c.loss.resolution.min <= resolution <= c.loss.resolution.max
            if msa.shape[0] > 2:
                found = True
                break
        assert found, f"No nonquery MSA observed in preflight for {name}"
    report = {"dataset_probability_mass": masses, "feature_examples": records}
    (root / "data_preflight.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    c = ConfigDict(yaml.safe_load(args.config.read_text()))
    validate_config(c)
    root = args.config.parent
    if args.check_data:
        check_data(c, root)
    if args.check or args.check_data:
        from protenix.utils.lr_scheduler import get_lr_scheduler
        from protenix.model.modules.pairformer import MSAModule

        with torch.device("meta"):
            msa_module = MSAModule(**c.model.msa_module, msa_configs=c.data.msa)
        assert msa_module.msa_configs["train_cutoff"] == 1024

        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=c.lr)
        scheduler = get_lr_scheduler(c, optimizer)
        values = {str(s): scheduler._get_step_lr(s) for s in (0, 2999, 3000, 51500, 100000)}
        assert abs(values["3000"] - 1e-4) < 1e-12
        assert abs(values["100000"] - 1e-5) < 1e-12
        (root / "config_preflight.json").write_text(json.dumps({"valid": True, "lr": values}, indent=2))
        print(json.dumps({"valid": True, "lr": values}), flush=True)
        return
    assert int(os.environ["WORLD_SIZE"]) == 32
    assert int(os.environ["LOCAL_WORLD_SIZE"]) == 8
    assert os.environ.get("WANDB_API_KEY")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from runner.train import AF3Trainer

    class AuditedTrainer(AF3Trainer):
        def train_step(self, batch):
            audit = self.step < 20
            gradients, handles = {}, []
            if audit:
                confidence = self.raw_model.confidence_head
                assert confidence.stop_gradient
                for name in ("linear_no_bias_pae.weight", "linear_no_bias_pde.weight",
                             "plddt_weight", "resolved_weight"):
                    parameter = dict(confidence.named_parameters())[name]

                    def capture(grad, key=name):
                        gradients[key] = float(grad.detach().float().norm())

                    handles.append(parameter.register_hook(capture))
            result = super().train_step(batch)
            for handle in handles:
                handle.remove()
            if audit:
                features = batch["input_feature_dict"]
                resolution = scalar(features["resolution"])
                name = scalar(batch["basic"]["dataset_name"])
                losses = {k: scalar(v) for k, v in result["loss_dict"].items() if "loss" in k}
                if name == PEPTIDE:
                    assert c.loss.resolution.min <= resolution <= c.loss.resolution.max
                    assert all(k in losses and "weighted_" + k in losses for k in CONFIDENCE_LOSSES)
                trace = result["msa_depth_trace"]
                assert trace and all(1 <= t["sampled_depth"] <= 1024 for t in trace)
                assert features["msa"].shape[-1] <= 1024
                proof = {"rank": int(os.environ["RANK"]), "step": self.step, "dataset": name,
                         "pdb_id": scalar(batch["basic"].get("pdb_id")), "resolution": resolution,
                         "msa_shape": list(features["msa"].shape), "msa_module_calls": trace,
                         "losses": losses, "confidence_gradient_norms": gradients,
                         "did_update": result["did_update"], "optimizer_skipped": result["optimizer_skipped"],
                         "lr": self.lr_scheduler.get_last_lr()}
                with Path(self.run_dir, f"training_audit_rank{os.environ['RANK']}.jsonl").open("a") as f:
                    f.write(json.dumps(proof) + "\n")
            return result

    trainer = AuditedTrainer(c)
    assert trainer.step == trainer.start_step == trainer.global_step == 0
    assert len(trainer.optimizer.state) == 0
    assert trainer.raw_model.confidence_head.stop_gradient
    assert trainer.train_dl.batch_size == 1
    assert trainer.raw_model.diffusion_batch_size == 1
    assert trainer.raw_model.msa_module.msa_configs["train_cutoff"] == 1024
    dataset = trainer.train_dl.dataset
    masses, offset = {}, 0
    for name, child in zip(dataset.dataset_names, dataset.datasets):
        masses[name] = float(dataset.merged_datapoint_weights[offset:offset + len(child)].sum())
        offset += len(child)
        assert abs(masses[name] - 0.5) < 1e-5
    proof = {"rank": int(os.environ["RANK"]), "world_size": 32, "batch_per_gpu": 1,
             "accumulation": 1, "diffusion_batch": 1, "initial_step": trainer.step,
             "optimizer_state_entries": len(trainer.optimizer.state), "dataset_probability_mass": masses,
             "checkpoint": c.load_checkpoint_path, "confidence_stop_gradient": True}
    Path(trainer.run_dir, f"initialization_rank{os.environ['RANK']}.json").write_text(json.dumps(proof, indent=2))
    trainer.run()


if __name__ == "__main__":
    main()
