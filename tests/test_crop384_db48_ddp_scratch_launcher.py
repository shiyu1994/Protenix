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


class TestCrop384Db48DdpScratchLauncher(unittest.TestCase):
    repo_dir = Path(__file__).resolve().parents[1]
    launcher = repo_dir / "run_crop384_db48_ddp_scratch_50k.sh"
    checkpoint_environment_variables = (
        "CHECKPOINT",
        "checkpoint",
        "CHECKPOINT_PATH",
        "checkpoint_path",
        "LOAD_CHECKPOINT_PATH",
        "load_checkpoint_path",
        "LOAD_EMA_CHECKPOINT_PATH",
        "load_ema_checkpoint_path",
        "RAW_CHECKPOINT",
        "raw_checkpoint",
        "RAW_CHECKPOINT_PATH",
        "raw_checkpoint_path",
        "EMA_CHECKPOINT",
        "ema_checkpoint",
        "EMA_CHECKPOINT_PATH",
        "ema_checkpoint_path",
        "INITIAL_CHECKPOINT",
        "initial_checkpoint",
        "INITIAL_CHECKPOINT_PATH",
        "initial_checkpoint_path",
        "RESUME_CHECKPOINT",
        "resume_checkpoint",
        "RESUME_CHECKPOINT_PATH",
        "resume_checkpoint_path",
        "RESUME_RAW_CHECKPOINT",
        "RESUME_EMA_CHECKPOINT",
        "RESUME_RUN_DIR",
        "RESUME_STEP",
        "PROTENIX_RESUME_RAW_CHECKPOINT",
        "PROTENIX_RESUME_EMA_CHECKPOINT",
        "PROTENIX_RESUME_CHECKPOINT_STEP",
        "PROTENIX_RESUME_RUN_DIR",
        "PROTENIX_RESUME_STEP",
        "BENCHMARK_LOAD_CHECKPOINT_PATH",
        "BENCHMARK_LOAD_EMA_CHECKPOINT_PATH",
        "EXPECTED_CHECKPOINT_SHA256",
    )

    @staticmethod
    def _deep_update(target, update):
        for key, value in update.items():
            if (
                isinstance(value, Mapping)
                and key in target
                and isinstance(target[key], Mapping)
            ):
                TestCrop384Db48DdpScratchLauncher._deep_update(target[key], value)
            else:
                target[key] = value

    def _resolved_config(self, train_args):
        base = copy.deepcopy({**configs_base, **{"data": data_configs}})
        self._deep_update(base, copy.deepcopy(model_configs["protenix-v2"]))
        return parse_configs(
            base,
            arg_str=" ".join(shlex.quote(argument) for argument in train_args),
            fill_required_with_null=True,
        )

    def _environment(self, extra_env=None):
        env = os.environ.copy()
        for variable_name in self.checkpoint_environment_variables:
            env.pop(variable_name, None)
        for variable_name in (
            "PROTENIX_FULL_RESUME",
            "PROTENIX_STRICT_FULL_RESUME",
            "PROTENIX_CROP384_STRICT_RESUME_INTERNAL",
            "PROTENIX_TRAINING_CONTRACT_ID",
            "PROTENIX_TRAINING_DATA_PROFILE",
            "PROTENIX_EXTRA_TRAIN_DATA_REQUIRED_PATHS",
            "PROTENIX_RESUME_SMOKE_STEPS",
            "MAX_STEPS",
            "max_steps",
            "PROTENIX_SMOKE_TEST",
            "PROTENIX_SMOKE_UPDATES",
            "PROTENIX_CHECKPOINT_SOURCE_SMOKE",
            "PROTENIX_STRAGGLER_BENCHMARK_UPDATES",
            "PROTENIX_DATALOADER_PERSISTENT_WORKERS",
            "PROTENIX_DATALOADER_PREFETCH_FACTOR",
            "INDEPENDENT_PAIRING_NON_PAIRING_BUDGET",
            "PROTENIX_CUDA_HOST_CC",
            "PROTENIX_CUDA_HOST_CXX",
            "NNODES",
            "nnodes",
            "NODE_RANK",
            "node_rank",
            "WORLD_SIZE",
            "RANK",
            "MLP_WORKER_NUM",
            "MLP_ROLE_INDEX",
            "MLP_WORKER_0_HOST",
            "MLP_WORKER_0_PORT",
            "MASTER_ADDR",
            "MASTER_PORT",
            "master_addr",
            "master_port",
            "NPROC_PER_NODE",
            "nproc_per_node",
            "SLURM_NNODES",
            "SLURM_NODEID",
        ):
            env.pop(variable_name, None)
        env.update(
            {
                "DRY_RUN": "1",
                "SKIP_PATH_CHECK": "1",
                "MLP_WORKER_NUM": "16",
                "MLP_ROLE_INDEX": "0",
                "MLP_WORKER_0_HOST": "scratch-worker-0",
                "MLP_WORKER_0_PORT": "23456",
                "NPROC_PER_NODE": "8",
                "RUN_NAME": "crop384_db48_ddp_scratch_v9_straggler_test",
                "BASE_DIR": "/tmp/protenix_crop384_db48_ddp_scratch_test",
                # Verify that reused-shell scientific values cannot mutate the
                # locked scratch workload.
                "model_name": "protenix_base_default_v1.0.0",
                "train_crop_size": "1024",
                "diffusion_batch_size": "8",
                "diffusion_chunk_size": "4",
                "n_blocks": "48",
                "distributed_strategy": "fsdp2",
                "optimizer_sharding": "zero1",
                "lr": "3e-6",
                "warmup_steps": "7",
                "ema_decay": "0.5",
                "ema_shadow_device": "cpu",
                "train_sets": "sabdab-train,distill-sabdab",
                "train_sample_weights": "0.5,0.5",
                "msa_min_size_train": "2048",
                "msa_max_size_train": "4096",
                "msa_sample_cutoff_train": "16384",
                "msa_chunk_size": "128",
                "pair_msa_as_unpaired_only": "true",
                "shuffle_pair_msa_when_unpaired_only": "true",
                "pairformer_blocks_per_ckpt": "1",
                "pairformer_checkpoint_cpu_offload_every_n": "99",
                "msa_checkpoint_cpu_offload_every_n": "99",
                "pair_transition_chunk_size": "16",
                "pair_transition_checkpoint_chunks": "false",
            }
        )
        env.update(extra_env or {})
        return env

    def _run(self, extra_env=None, arguments=None, *, check=True):
        return subprocess.run(
            ["bash", str(self.launcher), *(arguments or [])],
            cwd=self.repo_dir,
            env=self._environment(extra_env),
            check=check,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    @staticmethod
    def _command(result):
        command_line = next(
            line for line in result.stdout.splitlines() if line.startswith("torchrun ")
        )
        return shlex.split(command_line)

    def _dry_run(self, extra_env=None):
        result = self._run(extra_env)
        command = self._command(result)
        train_entry_index = command.index("./runner/train.py")
        train_args = command[train_entry_index + 1 :]
        return result, command, self._resolved_config(train_args)

    def test_dry_run_resolves_locked_random_initial_training_profile(self):
        result, command, cfg = self._dry_run()

        self.assertIn(
            "initialization=random raw_checkpoint=<empty> ema_checkpoint=<empty>",
            result.stdout,
        )
        self.assertNotIn("--load_checkpoint_path", command)
        self.assertNotIn("--load_ema_checkpoint_path", command)
        self.assertEqual(cfg.load_checkpoint_path, "")
        self.assertEqual(cfg.load_ema_checkpoint_path, "")
        self.assertEqual(cfg.model_name, "protenix-v2")
        self.assertEqual(cfg.n_blocks, 64)
        self.assertEqual(cfg.c_s, 384)
        self.assertEqual(cfg.c_z, 256)
        self.assertEqual(cfg.model.pairformer.n_blocks, 64)
        self.assertEqual(cfg.model.diffusion_module.c_s, 384)
        self.assertEqual(cfg.model.diffusion_module.c_token, 768)
        self.assertEqual(cfg.model.msa_module.c_m, 128)
        self.assertEqual(cfg.train_crop_size, 384)
        self.assertEqual(cfg.diffusion_batch_size, 48)
        self.assertEqual(cfg.diffusion_chunk_size, 48)
        self.assertEqual(cfg.model.N_cycle, 4)
        self.assertEqual(cfg.sample_diffusion.N_step, 20)
        self.assertEqual(cfg.iters_to_accumulate, 4)

        self.assertEqual(cfg.distributed_strategy, "ddp")
        self.assertTrue(cfg.ddp.gradient_as_bucket_view)
        self.assertEqual(cfg.optimizer_sharding, "none")
        self.assertEqual(cfg.max_steps, 50000)
        self.assertAlmostEqual(cfg.lr, 1e-3)
        self.assertEqual(cfg.warmup_steps, 1000)
        self.assertEqual(cfg.lr_scheduler, "af3")
        self.assertEqual(cfg.decay_every_n_steps, 25000)
        self.assertAlmostEqual(cfg.af3_lr_scheduler.decay_factor, 0.95)
        self.assertEqual(cfg.log_interval, 25)
        self.assertEqual(cfg.checkpoint_interval, 200)
        self.assertEqual(cfg.eval_interval, 0)
        self.assertFalse(cfg.eval_at_end)
        self.assertFalse(cfg.eval_ema_only)
        self.assertEqual(cfg.adam.beta1, 0.9)
        self.assertEqual(cfg.adam.beta2, 0.95)
        self.assertAlmostEqual(cfg.adam.weight_decay, 1e-8)
        self.assertFalse(cfg.adam.use_adamw)
        self.assertTrue(cfg.adam.fused)
        self.assertEqual(cfg.grad_clip_norm, 10)
        self.assertAlmostEqual(cfg.ema_decay, 0.999)
        self.assertEqual(cfg.ema_shadow_device, "model")
        self.assertTrue(cfg.enable_diffusion_shared_vars_cache)
        self.assertFalse(cfg.train_confidence_only)
        self.assertEqual(cfg.data.epoch_size, 10240)
        self.assertEqual(cfg.data.num_dl_workers, 8)
        self.assertTrue(cfg.data.train_dl_persistent_workers)
        self.assertEqual(cfg.data.train_dl_prefetch_factor, 2)
        self.assertAlmostEqual(cfg.data.query_only_msa_probability, 0.2)
        self.assertAlmostEqual(
            cfg.data.pair_msa_as_unpaired_probability,
            0.3,
        )
        self.assertAlmostEqual(
            1.0
            - cfg.data.query_only_msa_probability
            - cfg.data.pair_msa_as_unpaired_probability,
            0.5,
        )
        self.assertAlmostEqual(cfg.data.template_use_probability, 0.2)
        self.assertEqual(
            cfg.data.augmentation_decision_version,
            "sample_occurrence_sha256_v1",
        )
        self.assertEqual(cfg.data.template.template_dropout_rate, 0.0)

        self.assertEqual(
            list(cfg.data.train_sets),
            ["weightedPDB_before210930_v20260101"],
        )
        self.assertEqual(
            list(cfg.data.train_sampler.train_sample_weights),
            [1.0],
        )
        self.assertEqual(cfg.data.msa.min_size.train, 1)
        self.assertEqual(cfg.data.msa.max_size.train, 16384)
        self.assertEqual(cfg.data.msa.sample_cutoff.train, 2048)
        self.assertEqual(
            cfg.data.msa.query_inclusion_max_sample_size.train,
            1024,
        )
        self.assertEqual(
            cfg.data.msa.query_inclusion_max_sample_size.test,
            1024,
        )
        self.assertEqual(cfg.data.msa.max_input_paired_msa_depth, 10000)
        self.assertEqual(cfg.data.msa.max_input_unpaired_msa_depth, 10000)
        self.assertFalse(cfg.data.msa.independent_pairing_non_pairing_budget)
        self.assertEqual(cfg.model.msa_module.msa_max_size, 2048)
        self.assertEqual(cfg.model.msa_module.msa_chunk_size, 2048)
        self.assertFalse(cfg.data.msa.pair_msa_as_unpaired_only)
        self.assertFalse(cfg.data.msa.shuffle_pair_msa_when_unpaired_only)
        self.assertEqual(
            cfg.data[
                "weightedPDB_before210930_v20260101"
            ].cropping_configs.max_nuc_tokens,
            128,
        )
        self.assertNotIn("--data.sabdab-train.base_info.indices_fpath", command)
        self.assertNotIn("--data.distill-sabdab.base_info.indices_fpath", command)

        self.assertEqual(cfg.model.pairformer.blocks_per_ckpt, 4)
        self.assertIsNone(
            cfg.model.pairformer.checkpoint_cpu_offload_every_n
        )
        self.assertEqual(cfg.model.msa_module.blocks_per_ckpt, 1)
        self.assertIsNone(
            cfg.model.msa_module.checkpoint_cpu_offload_every_n
        )
        self.assertEqual(cfg.model.diffusion_module.blocks_per_ckpt, 1)
        self.assertTrue(cfg.model.diffusion_module.use_fine_grained_checkpoint)
        self.assertEqual(
            cfg.model.pairformer.train_pair_transition_chunk_size, 128
        )
        self.assertEqual(
            cfg.model.diffusion_module.train_pair_transition_chunk_size, 128
        )
        self.assertTrue(
            cfg.model.pairformer.train_pair_transition_checkpoint_chunks
        )
        self.assertTrue(
            cfg.model.diffusion_module.train_pair_transition_checkpoint_chunks
        )

        self.assertAlmostEqual(cfg.loss.weight.alpha_confidence, 1e-4)
        self.assertEqual(cfg.loss.weight.alpha_pae, 0)
        self.assertEqual(cfg.loss.weight.alpha_diffusion, 4)
        self.assertAlmostEqual(cfg.loss.weight.alpha_distogram, 0.03)
        self.assertEqual(cfg.loss.weight.alpha_bond, 0)
        self.assertEqual(cfg.loss.weight.smooth_lddt, 1)
        self.assertEqual(command[command.index("--nnodes") + 1], "16")
        self.assertEqual(command[command.index("--nproc_per_node") + 1], "8")
        self.assertIn(
            "target_batch=128 targets/microstep x accumulation=4 = 512 targets/update",
            result.stdout,
        )
        self.assertIn(
            "budget=50000 optimizer_updates 200000 microsteps_per_rank "
            "25600000 global_target_draws",
            result.stdout,
        )
        self.assertIn(
            "ddp_accumulation=first_backward_sync_once then "
            "steady_state_no_syncx3+sync",
            result.stdout,
        )
        self.assertIn(
            "epoch_size=10240 local_microsteps/epoch=80 updates/epoch=20",
            result.stdout,
        )
        self.assertIn(
            "data_profile=weighted_pdb_only_v1 "
            "train_sets=weightedPDB_before210930_v20260101 weights=1.0",
            result.stdout,
        )
        self.assertIn(
            "contract=crop384_db48_ddp_scratch_50k_ws128_acc4_"
            "data_weightedpdb_only_v1_arch_pf64_cs384_dct768_"
            "pfckpt8_c48_msamix_q20_u30_p50_msaraw_p10000_u10000_"
            "final16384_msaquerypinle1024_templatecache_top60u20id90_"
            "chaindummy_ckpt200_dlpersist1_prefetch2_deferredhealth_v11",
            result.stdout,
        )
        self.assertIn(
            "query_row_policy=pin_index0_when_K_le_1024 "
            "train/test=1024/1024 final_depth_remains_K",
            result.stdout,
        )
        self.assertIn(
            "train_dataloader=num_workers=8 persistent_workers=true "
            "prefetch_factor=2 pin_memory=false",
            result.stdout,
        )
        self.assertIn(
            "loss_health_sync=local_each_microstep "
            "global_all_reduce=once_at_update_boundary",
            result.stdout,
        )
        self.assertIn(
            "train_log_interval=25 inline_eval=false "
            "external_ema_eval_every=5000_updates",
            result.stdout,
        )
        self.assertIn(
            "augmentation_gates=msa_categorical_per_target_occurrence "
            "template_independent",
            result.stdout,
        )
        self.assertIn(
            "msa_mode_probabilities=query_only/unpaired_shuffle/"
            "species_paired=0.2/0.3/0.5",
            result.stdout,
        )

    def test_unregistered_data_profile_is_rejected_before_torchrun(self):
        result = self._run(
            {"PROTENIX_TRAINING_DATA_PROFILE": "future_distillation_unregistered"},
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "locks PROTENIX_TRAINING_DATA_PROFILE=weighted_pdb_only_v1",
            result.stderr,
        )
        self.assertNotIn("torchrun ", result.stdout)

    def test_nonempty_checkpoint_environment_variables_are_rejected(self):
        for variable_name in self.checkpoint_environment_variables:
            with self.subTest(variable_name=variable_name):
                result = self._run(
                    {variable_name: "/tmp/forbidden_checkpoint.pt"},
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    f"checkpoint environment variable {variable_name}",
                    result.stderr,
                )
                self.assertNotIn("torchrun ", result.stdout)

    def test_checkpoint_command_line_parameters_are_rejected(self):
        for arguments in (
            ["--load_checkpoint_path", "/tmp/raw.pt"],
            ["--load_ema_checkpoint_path=/tmp/ema.pt"],
            ["--skip_load_optimizer", "false"],
        ):
            with self.subTest(arguments=arguments):
                result = self._run(arguments=arguments, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    "rejects checkpoint/resume command-line parameter",
                    result.stderr,
                )
                self.assertNotIn("torchrun ", result.stdout)

    def test_production_step_budget_is_locked(self):
        result = self._run({"MAX_STEPS": "49999"}, check=False)

        self.assertEqual(result.returncode, 2)
        self.assertIn("locks MAX_STEPS=50000", result.stderr)
        self.assertNotIn("torchrun ", result.stdout)

    def test_one_step_smoke_disables_checkpoint_writes(self):
        result, _, cfg = self._dry_run(
            {
                "PROTENIX_SMOKE_TEST": "1",
                "MAX_STEPS": "999999",
                "max_steps": "888888",
            }
        )

        self.assertEqual(cfg.max_steps, 1)
        self.assertEqual(cfg.checkpoint_interval, -1)
        self.assertTrue(cfg.skip_final_checkpoint)
        self.assertIn("launch_mode=smoke_no_checkpoint", result.stdout)
        self.assertIn("initialization=random", result.stdout)
        self.assertIn("max_steps=1", result.stdout)
        self.assertIn("checkpoint_interval=-1 final_checkpoint=false", result.stdout)

    def test_20_update_distributed_smoke_uses_80_microsteps_without_writes(self):
        result, _, cfg = self._dry_run(
            {
                "PROTENIX_SMOKE_TEST": "1",
                "PROTENIX_SMOKE_UPDATES": "20",
            }
        )

        self.assertEqual(cfg.max_steps, 20)
        self.assertEqual(cfg.iters_to_accumulate, 4)
        self.assertEqual(cfg.checkpoint_interval, -1)
        self.assertTrue(cfg.skip_final_checkpoint)
        self.assertIn("launch_mode=smoke_no_checkpoint", result.stdout)
        self.assertIn("max_steps=20", result.stdout)

    def test_straggler_benchmark_supports_only_matched_100_or_250_updates(self):
        for updates in (100, 250):
            with self.subTest(updates=updates):
                result, _, cfg = self._dry_run(
                    {
                        "PROTENIX_STRAGGLER_BENCHMARK_UPDATES": str(updates),
                    }
                )

                self.assertEqual(cfg.max_steps, updates)
                self.assertEqual(cfg.data.epoch_size, 10240)
                self.assertTrue(cfg.data.train_dl_persistent_workers)
                self.assertEqual(cfg.data.train_dl_prefetch_factor, 2)
                self.assertEqual(cfg.checkpoint_interval, -1)
                self.assertTrue(cfg.skip_final_checkpoint)
                self.assertIn(
                    "launch_mode=v9_straggler_benchmark_no_checkpoint",
                    result.stdout,
                )
                self.assertIn(f"max_steps={updates}", result.stdout)

    def test_straggler_benchmark_rejects_other_budgets_and_smoke_mix(self):
        for environment in (
            {"PROTENIX_STRAGGLER_BENCHMARK_UPDATES": "101"},
            {
                "PROTENIX_STRAGGLER_BENCHMARK_UPDATES": "100",
                "PROTENIX_SMOKE_TEST": "1",
            },
        ):
            with self.subTest(environment=environment):
                result = self._run(environment, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("torchrun ", result.stdout)

    def test_run_name_must_identify_v9_straggler(self):
        result = self._run({"RUN_NAME": "ambiguous_training_name"}, check=False)

        self.assertEqual(result.returncode, 2)
        self.assertIn("must contain v9_straggler", result.stderr)
        self.assertNotIn("torchrun ", result.stdout)

    def test_checkpoint_source_smoke_configures_step_zero_raw_and_ema_pair(self):
        result, command, cfg = self._dry_run(
            {
                "PROTENIX_CHECKPOINT_SOURCE_SMOKE": "1",
                "MAX_STEPS": "999999",
                "max_steps": "888888",
            }
        )

        self.assertEqual(cfg.max_steps, 1)
        self.assertEqual(cfg.checkpoint_interval, 1)
        self.assertFalse(cfg.skip_final_checkpoint)
        self.assertNotIn("--load_checkpoint_path", command)
        self.assertNotIn("--load_ema_checkpoint_path", command)
        self.assertIn("launch_mode=checkpoint_source_smoke", result.stdout)
        self.assertIn(
            "initialization=random raw_checkpoint=<empty> ema_checkpoint=<empty>",
            result.stdout,
        )
        self.assertIn("checkpoint_interval=1 final_checkpoint=true", result.stdout)

    def test_checkpoint_source_smoke_rejects_invalid_flag(self):
        result = self._run(
            {"PROTENIX_CHECKPOINT_SOURCE_SMOKE": "2"},
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "PROTENIX_CHECKPOINT_SOURCE_SMOKE must be 0 or 1.",
            result.stderr,
        )
        self.assertNotIn("torchrun ", result.stdout)

    def test_checkpoint_source_smoke_is_mutually_exclusive_with_regular_smoke(self):
        result = self._run(
            {
                "PROTENIX_SMOKE_TEST": "1",
                "PROTENIX_CHECKPOINT_SOURCE_SMOKE": "1",
            },
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("are mutually exclusive", result.stderr)
        self.assertNotIn("torchrun ", result.stdout)

    def test_aistudio_multinode_environment_is_preserved(self):
        _, command, cfg = self._dry_run(
            {
                "MLP_WORKER_NUM": "16",
                "MLP_ROLE_INDEX": "2",
                "MLP_WORKER_0_HOST": "scratch-worker-0",
                "MLP_WORKER_0_PORT": "23456",
            }
        )

        self.assertEqual(command[command.index("--nnodes") + 1], "16")
        self.assertEqual(command[command.index("--node_rank") + 1], "2")
        self.assertEqual(
            command[command.index("--master_addr") + 1],
            "scratch-worker-0",
        )
        self.assertEqual(command[command.index("--master_port") + 1], "23456")
        self.assertEqual(cfg.distributed_strategy, "ddp")

    def test_non_128_gpu_topology_is_rejected_before_torchrun(self):
        result = self._run(
            {
                "MLP_WORKER_NUM": "4",
                "NPROC_PER_NODE": "8",
            },
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("requires topology=16x8", result.stderr)
        self.assertNotIn("torchrun ", result.stdout)

    def test_same_world_size_with_wrong_physical_split_is_rejected(self):
        result = self._run(
            {
                "MLP_WORKER_NUM": "8",
                "NPROC_PER_NODE": "16",
            },
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("requires topology=16x8", result.stderr)
        self.assertNotIn("torchrun ", result.stdout)

    def test_aistudio_coordinates_override_stale_lowercase_shell_values(self):
        _, command, _ = self._dry_run(
            {
                "nnodes": "1",
                "node_rank": "9",
                "nproc_per_node": "1",
                "master_addr": "stale-master",
                "master_port": "1",
            }
        )

        self.assertEqual(command[command.index("--nnodes") + 1], "16")
        self.assertEqual(command[command.index("--node_rank") + 1], "0")
        self.assertEqual(command[command.index("--nproc_per_node") + 1], "8")
        self.assertEqual(
            command[command.index("--master_addr") + 1],
            "scratch-worker-0",
        )
        self.assertEqual(command[command.index("--master_port") + 1], "23456")

    def test_conflicting_explicit_and_aistudio_rank_is_rejected(self):
        result = self._run({"NODE_RANK": "3"}, check=False)

        self.assertEqual(result.returncode, 2)
        self.assertIn("MLP_ROLE_INDEX=0 disagrees", result.stderr)
        self.assertNotIn("torchrun ", result.stdout)


if __name__ == "__main__":
    unittest.main()
