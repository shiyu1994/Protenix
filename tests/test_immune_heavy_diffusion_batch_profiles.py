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

import copy
import os
import shlex
import subprocess
import unittest
from collections.abc import Mapping
from pathlib import Path

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_model_type import model_configs
from protenix.config.config import parse_configs


class TestImmuneHeavyDiffusionBatchProfiles(unittest.TestCase):
    repo_dir = Path(__file__).resolve().parents[1]
    locked_launcher = (
        repo_dir
        / "run_msa2048_crop1024_nuc256_db32_chunk2048_fsdp2_finetune.sh"
    )
    db32_launcher = repo_dir / "run_immune_heavy_old_content_fsdp2_finetune.sh"
    db48_launcher = (
        repo_dir
        / "run_immune_heavy_old_content_crop1024_nuc256_db48_fsdp2_finetune.sh"
    )
    db40_nuc128_launcher = (
        repo_dir
        / "run_immune_heavy_old_content_crop1024_nuc128_db40_fsdp2_finetune.sh"
    )
    old_ema_checkpoint = (
        "/mnt/project_abd/klp/proteinx_output/"
        "protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/"
        "checkpoints/999_ema_0.999.pt"
    )
    dataset_names = (
        "weightedPDB_before210930_v20260101",
        "sabdab-train",
        "distill-sabdab",
    )

    @staticmethod
    def _deep_update(target, update):
        for key, value in update.items():
            if (
                isinstance(value, Mapping)
                and key in target
                and isinstance(target[key], Mapping)
            ):
                TestImmuneHeavyDiffusionBatchProfiles._deep_update(
                    target[key],
                    value,
                )
            else:
                target[key] = value

    def _resolved_config(self, train_args):
        base = copy.deepcopy({**configs_base, **{"data": data_configs}})
        self._deep_update(base, copy.deepcopy(model_configs["protenix-v2"]))
        return parse_configs(
            base,
            arg_str=" ".join(train_args),
            fill_required_with_null=True,
        )

    def _environment(self):
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "NPROC_PER_NODE": "1",
                "RUN_NAME": "immune_heavy_diffusion_batch_profile_test",
                "BASE_DIR": "/tmp/protenix_immune_heavy_diffusion_batch_test",
                # Both public wrappers must overwrite all of these hostile
                # values before entering the common locked launcher.
                "PROTENIX_TRAIN_CONTENT_PROFILE": "general_msa2048",
                "PROTENIX_DIFFUSION_BATCH_PROFILE": "64",
                "PROTENIX_MAX_NUC_TOKENS_PROFILE": "999",
                "PROTENIX_PAIR_MSA_PROFILE": "hostile_pair_mode",
                "DIFFUSION_BATCH_SIZE": "96",
                "diffusion_batch_size": "7",
                "train_crop_size": "768",
                "train_sets": "benchmark_long_protein_context_1024_800_msa2048",
                "train_sample_weights": "1.0",
                "msa_min_size_train": "2048",
                "msa_max_size_train": "4096",
                "msa_sample_cutoff_train": "4096",
                "msa_chunk_size": "128",
                "distributed_strategy": "ddp",
                "optimizer_sharding": "zero1",
                "ema_shadow_device": "model",
                "load_checkpoint_path": "/tmp/hostile_raw.pt",
                "load_ema_checkpoint_path": "/tmp/hostile_ema.pt",
            }
        )
        env.pop("MAX_STEPS", None)
        env.pop("max_steps", None)
        env.pop("PROTENIX_SMOKE_TEST", None)
        return env

    def _dry_run(self, launcher):
        result = subprocess.run(
            ["bash", str(launcher)],
            cwd=self.repo_dir,
            env=self._environment(),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        command_line = next(
            line for line in result.stdout.splitlines() if line.startswith("torchrun ")
        )
        command = shlex.split(command_line)
        train_entry_index = command.index("./runner/train.py")
        train_args = command[train_entry_index + 1 :]
        return result, self._resolved_config(train_args)

    def _scientific_signature(self, cfg):
        return {
            "checkpoint": (
                cfg.load_checkpoint_path,
                cfg.load_ema_checkpoint_path,
                cfg.load_params_only,
                cfg.load_strict,
            ),
            "data": (
                tuple(cfg.data.train_sets),
                tuple(cfg.data.train_sampler.train_sample_weights),
                cfg.data.train_sampler.seed,
                cfg.data.train_sampler.replacement,
            ),
            "crop": (
                cfg.train_crop_size,
                tuple(
                    cfg.data[name].cropping_configs.max_nuc_tokens
                    for name in self.dataset_names
                ),
            ),
            "msa": (
                cfg.data.msa.min_size.train,
                cfg.data.msa.sample_cutoff.train,
                cfg.data.msa.max_size.train,
                cfg.model.msa_module.msa_chunk_size,
                cfg.model.msa_module.msa_max_size,
                cfg.data.msa.pair_msa_as_unpaired_only,
                cfg.data.msa.shuffle_pair_msa_when_unpaired_only,
                cfg.data["sabdab-train"].msa.pair_msa_as_unpaired_only,
                cfg.data["distill-sabdab"].msa.pair_msa_as_unpaired_only,
            ),
            "fsdp": (
                cfg.distributed_strategy,
                cfg.optimizer_sharding,
                cfg.fsdp.mesh,
                cfg.fsdp.param_dtype,
                cfg.fsdp.reduce_dtype,
                cfg.fsdp.reshard_after_forward,
                cfg.ema_shadow_device,
                cfg.ema_decay,
            ),
            "model_and_runtime": (
                cfg.model_name,
                cfg.c_z,
                cfg.model.msa_module.c_m,
                cfg.diffusion_chunk_size,
                cfg.model.N_cycle,
                cfg.sample_diffusion.N_step,
                cfg.model.pairformer.blocks_per_ckpt,
                cfg.model.pairformer.checkpoint_cpu_offload_every_n,
                cfg.model.msa_module.checkpoint_cpu_offload_every_n,
            ),
            "schedule": (
                cfg.max_steps,
                cfg.lr,
                cfg.warmup_steps,
                cfg.checkpoint_interval,
                cfg.skip_final_checkpoint,
            ),
        }

    def test_db32_and_db48_wrappers_differ_only_in_diffusion_batch(self):
        db32_result, db32_cfg = self._dry_run(self.db32_launcher)
        db48_result, db48_cfg = self._dry_run(self.db48_launcher)

        self.assertEqual(db32_cfg.diffusion_batch_size, 32)
        self.assertEqual(db48_cfg.diffusion_batch_size, 48)
        self.assertIn("diffusion_batch/chunk=32/4", db32_result.stdout)
        self.assertIn("diffusion_batch/chunk=48/4", db48_result.stdout)
        self.assertEqual(
            self._scientific_signature(db32_cfg),
            self._scientific_signature(db48_cfg),
        )

        signature = self._scientific_signature(db48_cfg)
        self.assertEqual(
            signature["checkpoint"],
            (self.old_ema_checkpoint, self.old_ema_checkpoint, True, True),
        )
        self.assertEqual(
            signature["data"][:2],
            (self.dataset_names, (0.25, 0.5, 0.25)),
        )
        self.assertEqual(signature["crop"], (1024, (256, 256, 256)))
        self.assertEqual(
            signature["msa"][:5],
            (1, 2048, 16384, 2048, 2048),
        )
        self.assertEqual(
            signature["fsdp"][:6],
            ("fsdp2", "none", "auto", "fp32", "fp32", True),
        )

    def test_db40_nuc128_wrapper_changes_only_requested_settings(self):
        _, db32_cfg = self._dry_run(self.db32_launcher)
        db40_result, db40_cfg = self._dry_run(self.db40_nuc128_launcher)

        self.assertEqual(db40_cfg.diffusion_batch_size, 40)
        self.assertEqual(
            tuple(
                db40_cfg.data[name].cropping_configs.max_nuc_tokens
                for name in self.dataset_names
            ),
            (128, 128, 128),
        )
        self.assertIn(
            "crop=1024 max_nuc_tokens=128 diffusion_batch/chunk=40/4",
            db40_result.stdout,
        )
        self.assertIn(
            "pair_msa_profile=species_paired pair_as_unpaired=false "
            "shuffle_paired_hits=false",
            db40_result.stdout,
        )

        db32_signature = self._scientific_signature(db32_cfg)
        db40_signature = self._scientific_signature(db40_cfg)
        self.assertEqual(db40_signature["crop"], (1024, (128, 128, 128)))
        self.assertEqual(
            db32_signature["msa"][5:],
            (True, True, True, True),
        )
        self.assertEqual(
            db40_signature["msa"][5:],
            (False, False, False, False),
        )
        self.assertEqual(db32_signature["msa"][:5], db40_signature["msa"][:5])
        for section in db32_signature:
            if section not in {"crop", "msa"}:
                self.assertEqual(
                    db32_signature[section],
                    db40_signature[section],
                    msg=f"unexpected difference in {section}",
                )

    def test_common_launcher_rejects_an_unsupported_diffusion_batch_profile(self):
        env = self._environment()
        env["PROTENIX_TRAIN_CONTENT_PROFILE"] = "immune_heavy_old_content"
        env["PROTENIX_DIFFUSION_BATCH_PROFILE"] = "64"
        result = subprocess.run(
            ["bash", str(self.locked_launcher)],
            cwd=self.repo_dir,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "Unsupported PROTENIX_DIFFUSION_BATCH_PROFILE=64; expected 32, 40, or 48.",
            result.stderr,
        )
        self.assertNotIn("torchrun ", result.stdout)

    def test_common_launcher_rejects_an_unsupported_nuc_profile(self):
        env = self._environment()
        env["PROTENIX_TRAIN_CONTENT_PROFILE"] = "immune_heavy_old_content"
        env["PROTENIX_DIFFUSION_BATCH_PROFILE"] = "40"
        env["PROTENIX_MAX_NUC_TOKENS_PROFILE"] = "64"
        result = subprocess.run(
            ["bash", str(self.locked_launcher)],
            cwd=self.repo_dir,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "Unsupported PROTENIX_MAX_NUC_TOKENS_PROFILE=64; expected 128 or 256.",
            result.stderr,
        )
        self.assertNotIn("torchrun ", result.stdout)

    def test_common_launcher_rejects_an_unsupported_pair_msa_profile(self):
        env = self._environment()
        env["PROTENIX_TRAIN_CONTENT_PROFILE"] = "immune_heavy_old_content"
        env["PROTENIX_DIFFUSION_BATCH_PROFILE"] = "40"
        env["PROTENIX_MAX_NUC_TOKENS_PROFILE"] = "128"
        env["PROTENIX_PAIR_MSA_PROFILE"] = "invalid"
        result = subprocess.run(
            ["bash", str(self.locked_launcher)],
            cwd=self.repo_dir,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "Unsupported PROTENIX_PAIR_MSA_PROFILE=invalid; expected "
            "unpaired_shuffle or species_paired.",
            result.stderr,
        )
        self.assertNotIn("torchrun ", result.stdout)


if __name__ == "__main__":
    unittest.main()
