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


class TestCrop384Db48DdpStrictResumeLauncher(unittest.TestCase):
    repo_dir = Path(__file__).resolve().parents[1]
    launcher = repo_dir / "resume_crop384_db48_ddp_scratch_50k.sh"

    @staticmethod
    def _deep_update(target, update):
        for key, value in update.items():
            if (
                isinstance(value, Mapping)
                and key in target
                and isinstance(target[key], Mapping)
            ):
                TestCrop384Db48DdpStrictResumeLauncher._deep_update(
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
            arg_str=" ".join(shlex.quote(argument) for argument in train_args),
            fill_required_with_null=True,
        )

    def _environment(self, extra_env=None):
        env = os.environ.copy()
        for variable_name in (
            "NNODES",
            "nnodes",
            "NODE_RANK",
            "node_rank",
            "NPROC_PER_NODE",
            "nproc_per_node",
            "WORLD_SIZE",
            "RANK",
            "MLP_WORKER_NUM",
            "MLP_ROLE_INDEX",
            "MLP_WORKER_0_HOST",
            "MLP_WORKER_0_PORT",
            "SLURM_NNODES",
            "SLURM_NODEID",
            "MASTER_ADDR",
            "master_addr",
            "MASTER_PORT",
            "master_port",
            "PROTENIX_FULL_RESUME",
            "PROTENIX_STRICT_FULL_RESUME",
            "PROTENIX_CROP384_STRICT_RESUME_INTERNAL",
            "PROTENIX_TRAINING_DATA_PROFILE",
            "PROTENIX_EXTRA_TRAIN_DATA_REQUIRED_PATHS",
            "PROTENIX_RESUME_RAW_CHECKPOINT",
            "PROTENIX_RESUME_EMA_CHECKPOINT",
            "PROTENIX_RESUME_CHECKPOINT_STEP",
            "PROTENIX_RESUME_TARGET_MAX_STEPS",
            "PROTENIX_RESUME_SMOKE_STEPS",
            "PROTENIX_STRAGGLER_BENCHMARK_UPDATES",
            "PROTENIX_DATALOADER_PERSISTENT_WORKERS",
            "PROTENIX_DATALOADER_PREFETCH_FACTOR",
            "PROTENIX_CUDA_HOST_CC",
            "PROTENIX_CUDA_HOST_CXX",
            "RESUME_NNODES",
            "RESUME_NPROC_PER_NODE",
            "RESUME_NODE_RANK",
            "RESUME_MASTER_ADDR",
            "RESUME_MASTER_PORT",
        ):
            env.pop(variable_name, None)
        env.update(
            {
                "DRY_RUN": "1",
                "SKIP_PATH_CHECK": "1",
                "RUN_NAME": "crop384_strict_resume_v9_straggler_test",
                "BASE_DIR": "/tmp/protenix_crop384_strict_resume_test",
                "RESUME_RUN_DIR": "/tmp/crop384_source_run",
                "RESUME_STEP": "4999",
                "RESUME_NNODES": "16",
                "RESUME_NPROC_PER_NODE": "8",
                "RESUME_NODE_RANK": "0",
                "RESUME_MASTER_ADDR": "resume-worker-0",
                "RESUME_MASTER_PORT": "29641",
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

    def test_formal_resume_is_full_state_fail_closed_and_keeps_profile(self):
        result, command, cfg = self._dry_run()

        self.assertIn("resume=full checkpoint_step=4999 next_step=5000", result.stdout)
        self.assertIn("fail_closed=true", result.stdout)
        self.assertIn("data_worker_rng=not_checkpointed", result.stdout)
        self.assertEqual(
            cfg.load_checkpoint_path,
            "/tmp/crop384_source_run/checkpoints/4999.pt",
        )
        self.assertEqual(
            cfg.load_ema_checkpoint_path,
            "/tmp/crop384_source_run/checkpoints/4999_ema_0.999.pt",
        )
        self.assertTrue(cfg.load_strict)
        self.assertFalse(cfg.load_params_only)
        self.assertFalse(cfg.skip_load_optimizer)
        self.assertFalse(cfg.skip_load_scheduler)
        self.assertFalse(cfg.skip_load_step)
        self.assertFalse(cfg.load_step_for_scheduler)
        self.assertEqual(cfg.max_steps, 50000)
        self.assertEqual(cfg.checkpoint_interval, 200)
        self.assertFalse(cfg.skip_final_checkpoint)
        self.assertEqual(cfg.iters_to_accumulate, 4)
        self.assertAlmostEqual(cfg.lr, 1e-3)
        self.assertEqual(cfg.warmup_steps, 1000)
        self.assertEqual(cfg.decay_every_n_steps, 25000)
        self.assertAlmostEqual(cfg.af3_lr_scheduler.decay_factor, 0.95)
        self.assertEqual(cfg.log_interval, 25)
        self.assertEqual(cfg.eval_interval, 0)
        self.assertFalse(cfg.eval_at_end)

        self.assertEqual(cfg.model_name, "protenix-v2")
        self.assertEqual(cfg.n_blocks, 64)
        self.assertEqual(cfg.c_s, 384)
        self.assertEqual(cfg.model.pairformer.n_blocks, 64)
        self.assertEqual(cfg.model.diffusion_module.c_s, 384)
        self.assertEqual(cfg.model.diffusion_module.c_token, 768)
        self.assertEqual(cfg.train_crop_size, 384)
        self.assertEqual(cfg.diffusion_batch_size, 48)
        self.assertEqual(cfg.diffusion_chunk_size, 48)
        self.assertEqual(cfg.distributed_strategy, "ddp")
        self.assertTrue(cfg.ddp.gradient_as_bucket_view)
        self.assertEqual(cfg.optimizer_sharding, "none")
        self.assertTrue(cfg.adam.fused)
        self.assertEqual(cfg.ema_shadow_device, "model")
        self.assertTrue(cfg.enable_diffusion_shared_vars_cache)
        self.assertEqual(cfg.model.pairformer.blocks_per_ckpt, 4)
        self.assertEqual(cfg.model.msa_module.blocks_per_ckpt, 1)
        self.assertEqual(cfg.model.diffusion_module.blocks_per_ckpt, 1)
        self.assertIsNone(
            cfg.model.pairformer.checkpoint_cpu_offload_every_n
        )
        self.assertIsNone(
            cfg.model.msa_module.checkpoint_cpu_offload_every_n
        )
        self.assertEqual(
            cfg.model.pairformer.train_pair_transition_chunk_size, 128
        )
        self.assertTrue(
            cfg.model.pairformer.train_pair_transition_checkpoint_chunks
        )
        self.assertEqual(cfg.data.num_dl_workers, 8)
        self.assertEqual(cfg.data.epoch_size, 10240)
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
        self.assertNotIn("--data.sabdab-train.base_info.indices_fpath", command)
        self.assertNotIn("--data.distill-sabdab.base_info.indices_fpath", command)
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
        self.assertEqual(cfg.model.msa_module.msa_max_size, 2048)
        self.assertEqual(cfg.model.msa_module.msa_chunk_size, 2048)
        self.assertEqual(command[command.index("--nnodes") + 1], "16")
        self.assertEqual(command[command.index("--nproc_per_node") + 1], "8")
        self.assertEqual(command[command.index("--master_port") + 1], "29641")
        self.assertIn("topology=16x8 world_size=128", result.stdout)
        self.assertIn(
            "contract=crop384_db48_ddp_scratch_50k_ws128_acc4_"
            "data_weightedpdb_only_v1_arch_pf64_cs384_dct768_"
            "pfckpt8_c48_msamix_q20_u30_p50_msaraw_p10000_u10000_"
            "final16384_msaquerypinle1024_templatecache_top60u20id90_"
            "chaindummy_ckpt200_dlpersist1_prefetch2_deferredhealth_v11",
            result.stdout,
        )
        self.assertIn(
            "persistent_workers=true prefetch_factor=2",
            result.stdout,
        )

    def test_resume_smoke_runs_only_requested_updates_without_writes(self):
        result, _, cfg = self._dry_run({"RESUME_SMOKE_STEPS": "2"})

        self.assertEqual(cfg.max_steps, 5002)
        self.assertEqual(cfg.checkpoint_interval, -1)
        self.assertTrue(cfg.skip_final_checkpoint)
        self.assertIn("target_max_steps=5002 smoke_updates=2", result.stdout)

    def test_custom_matching_checkpoint_paths_are_forwarded(self):
        _, _, cfg = self._dry_run(
            {
                "RESUME_RAW_CHECKPOINT": "/tmp/pair/raw-4999.pt",
                "RESUME_EMA_CHECKPOINT": "/tmp/pair/raw-4999_ema_0.999.pt",
            }
        )

        self.assertEqual(cfg.load_checkpoint_path, "/tmp/pair/raw-4999.pt")
        self.assertEqual(
            cfg.load_ema_checkpoint_path,
            "/tmp/pair/raw-4999_ema_0.999.pt",
        )

    def test_regular_200_update_checkpoint_restores_saved_sampler_cursor(self):
        result, _, cfg = self._dry_run({"RESUME_STEP": "199"})

        self.assertEqual(cfg.max_steps, 50000)
        self.assertEqual(
            cfg.load_checkpoint_path,
            "/tmp/crop384_source_run/checkpoints/199.pt",
        )
        self.assertIn("data_worker_rng=not_checkpointed", result.stdout)

    def test_aistudio_multinode_topology_is_preserved_for_runner_validation(self):
        result, command, _ = self._dry_run(
            {
                "MLP_WORKER_NUM": "16",
                "MLP_ROLE_INDEX": "2",
                "MLP_WORKER_0_HOST": "resume-worker-0",
                "MLP_WORKER_0_PORT": "23456",
            }
        )

        self.assertEqual(command[command.index("--nnodes") + 1], "16")
        self.assertEqual(command[command.index("--nproc_per_node") + 1], "8")
        self.assertEqual(command[command.index("--node_rank") + 1], "2")
        self.assertEqual(
            command[command.index("--master_addr") + 1],
            "resume-worker-0",
        )
        self.assertEqual(command[command.index("--master_port") + 1], "23456")
        self.assertIn("topology=16x8 world_size=128", result.stdout)

    def test_locked_topology_and_aistudio_hostname_are_safe_fallbacks(self):
        result, command, _ = self._dry_run(
            {
                "RESUME_NNODES": "",
                "MLP_WORKER_NUM": "",
                "SLURM_NNODES": "",
                "MLP_ROLE_INDEX": "",
                "SLURM_NODEID": "",
                "RESUME_NODE_RANK": "",
                "MLP_WORKER_0_HOST": "",
                "RESUME_MASTER_ADDR": "",
                "HOSTNAME": "jo-resume-contract-worker-13",
            }
        )

        self.assertEqual(command[command.index("--nnodes") + 1], "16")
        self.assertEqual(command[command.index("--node_rank") + 1], "13")
        self.assertEqual(
            command[command.index("--master_addr") + 1],
            "jo-resume-contract-worker-0",
        )
        self.assertIn(
            "node_rank=13 source=hostname_suffix",
            result.stdout,
        )
        self.assertIn(
            "master_source=hostname_worker_0",
            result.stdout,
        )

    def test_explicit_node_rank_precedes_hostname_fallback(self):
        result, command, _ = self._dry_run(
            {
                "MLP_ROLE_INDEX": "4",
                "HOSTNAME": "jo-resume-contract-worker-13",
            }
        )

        self.assertEqual(command[command.index("--node_rank") + 1], "4")
        self.assertIn("node_rank=4 source=MLP_ROLE_INDEX", result.stdout)

    def test_malformed_explicit_node_rank_is_not_hidden_by_hostname(self):
        result = self._run(
            {
                "MLP_ROLE_INDEX": "bad",
                "HOSTNAME": "jo-resume-contract-worker-13",
            },
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("requires a valid explicit node rank", result.stderr)
        self.assertNotIn("torchrun ", result.stdout)

    def test_invalid_hostname_rank_fallbacks_are_rejected(self):
        for hostname in (
            "jo-resume-contract-worker-16",
            "jo-resume-contract-worker-x",
            "jo-resume-contract-worker-13.example",
            "jo-resume-contract-worker-13-extra",
        ):
            with self.subTest(hostname=hostname):
                result = self._run(
                    {
                        "MLP_ROLE_INDEX": "",
                        "SLURM_NODEID": "",
                        "RESUME_NODE_RANK": "",
                        "MLP_WORKER_0_HOST": "",
                        "RESUME_MASTER_ADDR": "",
                        "HOSTNAME": hostname,
                    },
                    check=False,
                )

                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    "requires a valid explicit node rank",
                    result.stderr,
                )
                self.assertNotIn("torchrun ", result.stdout)

    def test_non_128_gpu_resume_smoke_is_rejected(self):
        result = self._run(
            {
                "RESUME_SMOKE_STEPS": "1",
                "RESUME_NNODES": "1",
            },
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("requires topology=16x8", result.stderr)
        self.assertNotIn("torchrun ", result.stdout)

    def test_multinode_requires_current_coordinates(self):
        result = self._run(
            {
                "RESUME_NODE_RANK": "",
                "RESUME_MASTER_ADDR": "",
            },
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("requires a valid explicit node rank", result.stderr)
        self.assertNotIn("torchrun ", result.stdout)

    def test_missing_source_cli_overrides_and_malformed_pair_are_rejected(self):
        cases = (
            ({"RESUME_RUN_DIR": ""}, None, "RESUME_RUN_DIR is required"),
            ({}, ["--load_params_only", "true"], "accepts no command-line"),
            (
                {"RESUME_RAW_CHECKPOINT": "/tmp/4999_ema_0.999.pt"},
                None,
                "must not point to an EMA",
            ),
            (
                {"RESUME_EMA_CHECKPOINT": "/tmp/4999.pt"},
                None,
                "must identify the matching EMA",
            ),
        )
        for environment, arguments, message in cases:
            with self.subTest(environment=environment, arguments=arguments):
                result = self._run(
                    environment,
                    arguments=arguments,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
                self.assertNotIn("torchrun ", result.stdout)


if __name__ == "__main__":
    unittest.main()
