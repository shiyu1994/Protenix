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


class TestImmuneHeavyOldContentFinetune(unittest.TestCase):
    repo_dir = Path(__file__).resolve().parents[1]
    launcher = repo_dir / "run_immune_heavy_old_content_fsdp2_finetune.sh"
    old_ema_checkpoint = (
        "/mnt/project_abd/klp/proteinx_output/"
        "protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/"
        "checkpoints/999_ema_0.999.pt"
    )

    @staticmethod
    def _deep_update(target, update):
        for key, value in update.items():
            if (
                isinstance(value, Mapping)
                and key in target
                and isinstance(target[key], Mapping)
            ):
                TestImmuneHeavyOldContentFinetune._deep_update(target[key], value)
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

    def _environment(self, extra_env=None):
        env = os.environ.copy()
        # The public wrapper must select immune_heavy_old_content and overwrite
        # scientific/runtime settings inherited from a reused AIStudio shell.
        env.update(
            {
                "DRY_RUN": "1",
                "NPROC_PER_NODE": "1",
                "RUN_NAME": "immune_heavy_old_content_test",
                "BASE_DIR": "/tmp/protenix_immune_heavy_old_content_test",
                "PROTENIX_TRAIN_CONTENT_PROFILE": "general_msa2048",
                "distributed_strategy": "ddp",
                "optimizer_sharding": "zero1",
                "ema_shadow_device": "model",
                "load_checkpoint_path": "/tmp/hostile_raw.pt",
                "load_ema_checkpoint_path": "/tmp/hostile_ema.pt",
                "load_params_only": "false",
                "model_c_z": "128",
                "model_c_m": "64",
                "train_crop_size": "768",
                "diffusion_batch_size": "24",
                "diffusion_chunk_size": "1",
                "msa_min_size_train": "2048",
                "msa_max_size_train": "4096",
                "msa_sample_cutoff_train": "16384",
                "msa_chunk_size": "128",
                "pair_msa_as_unpaired_only": "false",
                "shuffle_pair_msa_when_unpaired_only": "false",
                "train_sets": "benchmark_long_protein_context_1024_800_msa2048",
                "train_sample_weights": "1.0",
                "lr": "0.1",
                "warmup_steps": "7",
                "checkpoint_interval": "999",
                "skip_final_checkpoint": "true",
                "train_confidence_only": "true",
                "pairformer_blocks_per_ckpt": "1",
                "pairformer_checkpoint_cpu_offload_every_n": "99",
                "msa_checkpoint_cpu_offload_every_n": "99",
                "pair_transition_chunk_size": "16",
                "pair_transition_checkpoint_chunks": "false",
                "BENCHMARK_EXACT_CROP": "true",
                "BENCHMARK_WARMUP_STEPS": "2",
                "BENCHMARK_MEASURE_STEPS": "3",
            }
        )
        # The formal old-content profile owns its exact 2,000-step schedule.
        # Avoid accidentally inheriting either spelling from the test runner.
        env.pop("MAX_STEPS", None)
        env.pop("max_steps", None)
        env.pop("PROTENIX_SMOKE_TEST", None)
        env.update(extra_env or {})
        return env

    def _run(self, extra_env=None, *, check=True):
        return subprocess.run(
            ["bash", str(self.launcher)],
            cwd=self.repo_dir,
            env=self._environment(extra_env),
            check=check,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _dry_run(self, extra_env=None):
        result = self._run(extra_env)
        command_line = next(
            line for line in result.stdout.splitlines() if line.startswith("torchrun ")
        )
        command = shlex.split(command_line)
        train_entry_index = command.index("./runner/train.py")
        train_args = command[train_entry_index + 1 :]
        return result, command, train_args, self._resolved_config(train_args)

    def test_dry_run_resolves_old_content_on_current_fsdp2_engine(self):
        result, command, train_args, cfg = self._dry_run()

        self.assertIn("content_profile=immune_heavy_old_content", result.stdout)
        self.assertNotIn("benchmark_long_protein_context", " ".join(command))
        self.assertEqual(command[command.index("--nproc_per_node") + 1], "1")

        # Old immune-heavy initialization and training content.
        self.assertEqual(cfg.load_checkpoint_path, self.old_ema_checkpoint)
        self.assertEqual(cfg.load_ema_checkpoint_path, self.old_ema_checkpoint)
        self.assertTrue(cfg.load_params_only)
        self.assertTrue(cfg.load_strict)
        self.assertEqual(
            list(cfg.data.train_sets),
            [
                "weightedPDB_before210930_v20260101",
                "sabdab-train",
                "distill-sabdab",
            ],
        )
        self.assertEqual(
            list(cfg.data.train_sampler.train_sample_weights),
            [0.25, 0.5, 0.25],
        )
        self.assertEqual(cfg.data.msa.min_size.train, 1)
        self.assertEqual(cfg.data.msa.max_size.train, 16384)
        self.assertEqual(cfg.data.msa.sample_cutoff.train, 2048)
        self.assertEqual(cfg.model.msa_module.msa_chunk_size, 2048)
        self.assertEqual(cfg.model.msa_module.msa_max_size, 2048)
        self.assertAlmostEqual(cfg.lr, 3e-6)
        self.assertEqual(cfg.warmup_steps, 2000)
        self.assertEqual(cfg.max_steps, 2000)
        self.assertEqual(cfg.checkpoint_interval, 250)
        self.assertFalse(cfg.skip_final_checkpoint)
        self.assertFalse(cfg.train_confidence_only)

        # Current, benchmarked workload and execution-engine settings remain.
        self.assertEqual(cfg.model_name, "protenix-v2")
        self.assertEqual(cfg.c_z, 256)
        self.assertEqual(cfg.model.msa_module.c_m, 128)
        self.assertEqual(cfg.train_crop_size, 1024)
        self.assertEqual(cfg.diffusion_batch_size, 32)
        self.assertEqual(cfg.diffusion_chunk_size, 4)
        for dataset_name in (
            "weightedPDB_before210930_v20260101",
            "sabdab-train",
            "distill-sabdab",
        ):
            self.assertEqual(
                cfg.data[dataset_name].cropping_configs.max_nuc_tokens,
                256,
            )

        self.assertTrue(cfg.data.msa.pair_msa_as_unpaired_only)
        self.assertTrue(cfg.data.msa.shuffle_pair_msa_when_unpaired_only)
        self.assertTrue(cfg.data["sabdab-train"].msa.pair_msa_as_unpaired_only)
        self.assertTrue(
            cfg.data["distill-sabdab"].msa.pair_msa_as_unpaired_only
        )

        self.assertEqual(cfg.distributed_strategy, "fsdp2")
        self.assertEqual(cfg.optimizer_sharding, "none")
        self.assertEqual(cfg.fsdp.mesh, "auto")
        self.assertEqual(cfg.fsdp.param_dtype, "fp32")
        self.assertEqual(cfg.fsdp.reduce_dtype, "fp32")
        self.assertTrue(cfg.fsdp.reshard_after_forward)
        self.assertEqual(cfg.ema_shadow_device, "cpu")
        self.assertAlmostEqual(cfg.ema_decay, 0.999)

        self.assertEqual(cfg.model.pairformer.blocks_per_ckpt, 2)
        self.assertEqual(cfg.model.msa_module.blocks_per_ckpt, 1)
        self.assertEqual(cfg.model.diffusion_module.blocks_per_ckpt, 1)
        self.assertTrue(cfg.model.diffusion_module.use_fine_grained_checkpoint)
        self.assertEqual(
            cfg.model.pairformer.checkpoint_cpu_offload_every_n,
            2,
        )
        self.assertEqual(cfg.model.msa_module.checkpoint_cpu_offload_every_n, 1)
        self.assertEqual(
            cfg.model.pairformer.train_pair_transition_chunk_size,
            128,
        )
        self.assertEqual(
            cfg.model.diffusion_module.train_pair_transition_chunk_size,
            128,
        )
        self.assertTrue(
            cfg.model.pairformer.train_pair_transition_checkpoint_chunks
        )
        self.assertTrue(
            cfg.model.diffusion_module.train_pair_transition_checkpoint_chunks
        )
        self.assertEqual(cfg.benchmark_warmup_steps, -1)
        self.assertEqual(cfg.benchmark_measure_steps, -1)

        # Keep the parsed argument list referenced so an accidental empty
        # torchrun payload produces a direct, useful failure above.
        self.assertTrue(train_args)

    def test_hostile_max_steps_is_rejected(self):
        result = self._run({"MAX_STEPS": "1999"}, check=False)

        self.assertEqual(result.returncode, 2)
        self.assertIn("locks MAX_STEPS=2000", result.stderr)
        self.assertNotIn("torchrun ", result.stdout)

    def test_smoke_mode_forces_one_step_and_disables_all_checkpoints(self):
        result, _, _, cfg = self._dry_run(
            {
                "PROTENIX_SMOKE_TEST": "1",
                "MAX_STEPS": "999999",
                "max_steps": "888888",
            }
        )

        self.assertEqual(cfg.max_steps, 1)
        self.assertEqual(cfg.checkpoint_interval, -1)
        self.assertTrue(cfg.skip_final_checkpoint)
        self.assertIn("max_steps=1", result.stdout)
        self.assertIn("checkpoint_interval=-1", result.stdout)
        self.assertIn("final_checkpoint=false", result.stdout)


if __name__ == "__main__":
    unittest.main()
