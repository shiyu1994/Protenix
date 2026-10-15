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


class TestLockedMSA2048Finetune(unittest.TestCase):
    repo_dir = Path(__file__).resolve().parents[1]
    launcher = (
        repo_dir
        / "run_msa2048_crop1024_nuc256_db32_chunk2048_fsdp2_finetune.sh"
    )

    def _dry_run(self, extra_env=None):
        env = os.environ.copy()
        env.pop("PROTENIX_TRAIN_CONTENT_PROFILE", None)
        env.pop("PROTENIX_SMOKE_TEST", None)
        env.update(
            {
                "DRY_RUN": "1",
                "SKIP_PATH_CHECK": "1",
                "NPROC_PER_NODE": "1",
                "RUN_NAME": "locked_msa2048_test",
                "BASE_DIR": "/tmp/protenix_locked_profile_test",
                "MAX_STEPS": "123",
                # Deliberately hostile inherited values.  A reused AIStudio
                # shell must not be able to mutate the locked profile.
                "BENCHMARK_EXACT_CROP": "true",
                "BENCHMARK_WARMUP_STEPS": "9",
                "BENCHMARK_MEASURE_STEPS": "11",
                "BENCHMARK_DATASET_NAME": "benchmark_long_protein_context_1024_800_msa2048",
                "distributed_strategy": "ddp",
                "ema_shadow_device": "model",
                "model_c_z": "128",
                "model_c_m": "64",
                "msa_chunk_size": "128",
                "train_crop_size": "768",
                "diffusion_batch_size": "24",
                "train_sets": "benchmark_long_protein_context_1024_800_msa2048",
                "train_sample_weights": "1.0",
                "skip_final_checkpoint": "true",
                "checkpoint_interval": "-1",
            }
        )
        env.update(extra_env or {})
        result = subprocess.run(
            ["bash", str(self.launcher)],
            cwd=self.repo_dir,
            env=env,
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
        return result, command, command[train_entry_index + 1 :]

    @staticmethod
    def _deep_update(target, update):
        for key, value in update.items():
            if (
                isinstance(value, Mapping)
                and key in target
                and isinstance(target[key], Mapping)
            ):
                TestLockedMSA2048Finetune._deep_update(target[key], value)
            else:
                target[key] = value

    def _resolved_config(self, train_args):
        base = copy.deepcopy({**configs_base, **{"data": data_configs}})
        self._deep_update(base, copy.deepcopy(model_configs["protenix-v2"]))
        return parse_configs(base, arg_str=" ".join(train_args), fill_required_with_null=True)

    def test_dry_run_resolves_the_complete_production_profile(self):
        result, command, train_args = self._dry_run()
        cfg = self._resolved_config(train_args)

        self.assertIn("Locked Protenix formal training profile:", result.stdout)
        self.assertNotIn("benchmark_long_protein_context", " ".join(command))
        self.assertEqual(command[command.index("--nproc_per_node") + 1], "1")

        self.assertEqual(cfg.model_name, "protenix-v2")
        self.assertEqual(cfg.c_z, 256)
        self.assertEqual(cfg.model.msa_module.c_m, 128)
        for module_name in (
            "relative_position_encoding",
            "template_embedder",
            "msa_module",
            "pairformer",
            "diffusion_module",
            "confidence_head",
            "distogram_head",
        ):
            self.assertEqual(cfg.model[module_name].c_z, 256)

        self.assertEqual(cfg.train_crop_size, 1024)
        self.assertEqual(cfg.diffusion_batch_size, 32)
        self.assertEqual(cfg.diffusion_chunk_size, 4)
        self.assertEqual(cfg.model.N_cycle, 4)
        self.assertEqual(cfg.sample_diffusion.N_step, 20)
        self.assertEqual(cfg.dtype, "bf16")
        self.assertEqual(cfg.max_atoms_per_token, 24)
        self.assertFalse(cfg.enable_tf32)

        self.assertEqual(cfg.data.msa.min_size.train, 2048)
        self.assertEqual(cfg.data.msa.sample_cutoff.train, 2048)
        self.assertEqual(cfg.data.msa.max_size.train, 16384)
        self.assertEqual(cfg.model.msa_module.msa_chunk_size, 2048)
        self.assertEqual(cfg.model.msa_module.msa_max_size, 2048)
        self.assertTrue(cfg.data.msa.pair_msa_as_unpaired_only)
        self.assertTrue(cfg.data.msa.shuffle_pair_msa_when_unpaired_only)
        self.assertTrue(cfg.data["sabdab-train"].msa.pair_msa_as_unpaired_only)
        self.assertTrue(cfg.data["distill-sabdab"].msa.pair_msa_as_unpaired_only)

        for dataset_name in (
            "weightedPDB_before210930_v20260101",
            "sabdab-train",
            "distill-sabdab",
        ):
            self.assertEqual(cfg.data[dataset_name].cropping_configs.max_nuc_tokens, 256)
            self.assertEqual(
                list(cfg.data[dataset_name].cropping_configs.method_weights),
                [0.2, 0.4, 0.4],
            )

        self.assertEqual(cfg.model.pairformer.blocks_per_ckpt, 2)
        self.assertEqual(cfg.model.msa_module.blocks_per_ckpt, 1)
        self.assertEqual(cfg.model.diffusion_module.blocks_per_ckpt, 1)
        self.assertTrue(cfg.model.diffusion_module.use_fine_grained_checkpoint)
        self.assertEqual(cfg.model.pairformer.checkpoint_cpu_offload_every_n, 2)
        self.assertEqual(cfg.model.msa_module.checkpoint_cpu_offload_every_n, 1)
        self.assertEqual(cfg.model.pairformer.train_pair_transition_chunk_size, 128)
        self.assertEqual(cfg.model.diffusion_module.train_pair_transition_chunk_size, 128)
        self.assertTrue(cfg.model.pairformer.train_pair_transition_checkpoint_chunks)
        self.assertTrue(cfg.model.diffusion_module.train_pair_transition_checkpoint_chunks)

        self.assertEqual(cfg.distributed_strategy, "fsdp2")
        self.assertEqual(cfg.optimizer_sharding, "none")
        self.assertEqual(cfg.fsdp.mesh, "auto")
        self.assertEqual(cfg.fsdp.param_dtype, "fp32")
        self.assertEqual(cfg.fsdp.reduce_dtype, "fp32")
        self.assertTrue(cfg.fsdp.reshard_after_forward)
        self.assertEqual(cfg.ema_shadow_device, "cpu")
        self.assertAlmostEqual(cfg.ema_decay, 0.999)

        self.assertEqual(
            list(cfg.data.train_sets),
            [
                "weightedPDB_before210930_v20260101",
                "sabdab-train",
                "distill-sabdab",
            ],
        )
        self.assertEqual(list(cfg.data.train_sampler.train_sample_weights), [0.5, 0.3, 0.2])
        self.assertEqual(cfg.data.train_sampler.seed, -1)
        self.assertTrue(cfg.data.train_sampler.replacement)
        self.assertEqual(cfg.data.epoch_size, 10000)
        self.assertEqual(cfg.data.num_dl_workers, 16)

        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.max_steps, 123)
        self.assertEqual(cfg.warmup_steps, 2000)
        self.assertAlmostEqual(cfg.lr, 1e-5)
        self.assertEqual(cfg.lr_scheduler, "af3")
        self.assertEqual(cfg.iters_to_accumulate, 1)
        self.assertEqual(cfg.grad_clip_norm, 10)
        self.assertEqual(cfg.decay_every_n_steps, 50000)
        self.assertAlmostEqual(cfg.adam.beta1, 0.9)
        self.assertAlmostEqual(cfg.adam.beta2, 0.95)
        self.assertAlmostEqual(cfg.adam.weight_decay, 1e-8)
        self.assertFalse(cfg.adam.use_adamw)

        official_checkpoint = (
            "/mnt/project_abd/public_databases/protenix.20260101/"
            "checkpoint/protenix-v2.pt"
        )
        self.assertEqual(cfg.load_checkpoint_path, official_checkpoint)
        self.assertEqual(cfg.load_ema_checkpoint_path, official_checkpoint)
        self.assertTrue(cfg.load_params_only)
        self.assertTrue(cfg.load_strict)
        self.assertEqual(cfg.checkpoint_interval, 1000)
        self.assertFalse(cfg.skip_final_checkpoint)
        self.assertEqual(cfg.benchmark_warmup_steps, -1)
        self.assertEqual(cfg.benchmark_measure_steps, -1)
        self.assertEqual(cfg.eval_interval, 0)
        self.assertFalse(cfg.eval_at_end)

    def test_run_name_is_required(self):
        env = os.environ.copy()
        env.update({"DRY_RUN": "1", "SKIP_PATH_CHECK": "1"})
        env.pop("RUN_NAME", None)
        env.pop("run_name", None)
        result = subprocess.run(
            ["bash", str(self.launcher)],
            cwd=self.repo_dir,
            env=env,
            check=False,
            text=True,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("RUN_NAME is required", result.stderr)

    def test_positional_overrides_are_rejected(self):
        env = os.environ.copy()
        env.update({"DRY_RUN": "1", "RUN_NAME": "locked_msa2048_test"})
        result = subprocess.run(
            ["bash", str(self.launcher), "--train_crop_size", "768"],
            cwd=self.repo_dir,
            env=env,
            check=False,
            text=True,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("accepts no positional CLI overrides", result.stderr)


if __name__ == "__main__":
    unittest.main()
