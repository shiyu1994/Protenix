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


class TestCrop384Db48DdpSpeedTrial(unittest.TestCase):
    repo_dir = Path(__file__).resolve().parents[1]
    launcher = repo_dir / "run_crop384_db48_ddp_speed_trial.sh"
    dataset_name = "benchmark_long_protein_context_384_800_msa2048"
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
    profile_expectations = {
        "stable": {
            "bucket_view": False,
            "adam_fused": False,
            "ema_device": "cpu",
            "cache": False,
            "diffusion_chunk": 4,
            "pairformer_blocks": 2,
            "msa_blocks": 1,
            "diffusion_blocks": 1,
            "fine_checkpoint": True,
            "pairformer_offload": 2,
            "msa_offload": 1,
            "transition_chunk": 128,
            "transition_checkpoint": True,
        },
        "lowrisk": {
            "bucket_view": True,
            "adam_fused": True,
            "ema_device": "model",
            "cache": False,
            "diffusion_chunk": 4,
            "pairformer_blocks": 4,
            "msa_blocks": 4,
            "diffusion_blocks": 1,
            "fine_checkpoint": True,
            "pairformer_offload": None,
            "msa_offload": None,
            "transition_chunk": 128,
            "transition_checkpoint": True,
        },
        "cache": {
            "bucket_view": True,
            "adam_fused": True,
            "ema_device": "model",
            "cache": True,
            "diffusion_chunk": 4,
            "pairformer_blocks": 4,
            "msa_blocks": 4,
            "diffusion_blocks": 1,
            "fine_checkpoint": True,
            "pairformer_offload": None,
            "msa_offload": None,
            "transition_chunk": 128,
            "transition_checkpoint": True,
        },
        "chunk8": {"diffusion_chunk": 8},
        "chunk12": {"diffusion_chunk": 12},
        "chunk16": {"diffusion_chunk": 16},
        "chunk24": {"diffusion_chunk": 24},
        "chunk48": {"diffusion_chunk": 48},
        "grouped": {
            "diffusion_chunk": 24,
            "pairformer_blocks": 8,
            "msa_blocks": 4,
            "diffusion_blocks": 3,
        },
        "transition": {
            "diffusion_chunk": 24,
            "pairformer_blocks": 8,
            "msa_blocks": 4,
            "diffusion_blocks": 3,
            "transition_chunk": None,
            "transition_checkpoint": False,
        },
        "pf16": {
            "diffusion_chunk": 24,
            "pairformer_blocks": 16,
            "msa_blocks": 4,
            "diffusion_blocks": 3,
            "transition_chunk": None,
            "transition_checkpoint": False,
        },
        "pf18": {
            "diffusion_chunk": 24,
            "pairformer_blocks": 18,
            "msa_blocks": 4,
            "diffusion_blocks": 3,
            "transition_chunk": None,
            "transition_checkpoint": False,
        },
        "pf19": {
            "diffusion_chunk": 24,
            "pairformer_blocks": 19,
            "msa_blocks": 4,
            "diffusion_blocks": 3,
            "transition_chunk": None,
            "transition_checkpoint": False,
        },
        "pf20": {
            "diffusion_chunk": 24,
            "pairformer_blocks": 20,
            "msa_blocks": 4,
            "diffusion_blocks": 3,
            "transition_chunk": None,
            "transition_checkpoint": False,
        },
        "pf21": {
            "diffusion_chunk": 24,
            "pairformer_blocks": 21,
            "msa_blocks": 4,
            "diffusion_blocks": 3,
            "transition_chunk": None,
            "transition_checkpoint": False,
        },
        "pf22": {
            "diffusion_chunk": 24,
            "pairformer_blocks": 22,
            "msa_blocks": 4,
            "diffusion_blocks": 3,
            "transition_chunk": None,
            "transition_checkpoint": False,
        },
        "pf23": {
            "diffusion_chunk": 24,
            "pairformer_blocks": 23,
            "msa_blocks": 4,
            "diffusion_blocks": 3,
            "transition_chunk": None,
            "transition_checkpoint": False,
        },
        "pf24": {
            "diffusion_chunk": 24,
            "pairformer_blocks": 24,
            "msa_blocks": 4,
            "diffusion_blocks": 3,
            "transition_chunk": None,
            "transition_checkpoint": False,
        },
        "pf48": {
            "diffusion_chunk": 24,
            "pairformer_blocks": 48,
            "msa_blocks": 4,
            "diffusion_blocks": 3,
            "transition_chunk": None,
            "transition_checkpoint": False,
        },
        "trunk_nockpt": {
            "diffusion_chunk": 24,
            "pairformer_blocks": None,
            "msa_blocks": None,
            "diffusion_blocks": 3,
            "transition_chunk": None,
            "transition_checkpoint": False,
        },
        "all_nockpt": {
            "diffusion_chunk": 4,
            "pairformer_blocks": None,
            "msa_blocks": None,
            "diffusion_blocks": None,
            "fine_checkpoint": False,
            "transition_chunk": None,
            "transition_checkpoint": False,
        },
    }

    @staticmethod
    def _deep_update(target, update):
        for key, value in update.items():
            if (
                isinstance(value, Mapping)
                and key in target
                and isinstance(target[key], Mapping)
            ):
                TestCrop384Db48DdpSpeedTrial._deep_update(target[key], value)
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

    def _environment(self, profile="stable", extra_env=None):
        env = os.environ.copy()
        for variable_name in self.checkpoint_environment_variables:
            env.pop(variable_name, None)
        for variable_name in (
            "PROTENIX_FULL_RESUME",
            "PROTENIX_RESUME_SMOKE_STEPS",
            "TRIAL_DIFFUSION_CHUNK",
            "SPEED_SCREEN",
            "SPEED_NUC_STRESS",
            "NNODES",
            "nnodes",
            "NODE_RANK",
            "node_rank",
            "NPROC_PER_NODE",
            "nproc_per_node",
            "MLP_WORKER_NUM",
            "MLP_ROLE_INDEX",
            "SLURM_NNODES",
            "SLURM_NODEID",
            "MASTER_ADDR",
            "master_addr",
            "MASTER_PORT",
            "master_port",
            "BENCHMARK_DATASET_DIR",
            "PROTENIX_CUDA_HOST_CC",
            "PROTENIX_CUDA_HOST_CXX",
        ):
            env.pop(variable_name, None)
        env.update(
            {
                "DRY_RUN": "1",
                "SKIP_PATH_CHECK": "1",
                "SPEED_PROFILE": profile,
                "TRIAL_ID": f"test_{profile}",
                "MASTER_PORT": "29701",
                # Reused-shell values must not mutate a profile.
                "train_crop_size": "1024",
                "diffusion_batch_size": "8",
                "distributed_strategy": "fsdp2",
                "optimizer_sharding": "zero1",
                "ema_shadow_device": "cpu",
                "lr": "3e-6",
                "msa_min_size_train": "1",
                "msa_max_size_train": "4096",
                "msa_sample_cutoff_train": "32",
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

    def _run(
        self,
        profile="stable",
        extra_env=None,
        arguments=None,
        *,
        check=True,
    ):
        return subprocess.run(
            ["bash", str(self.launcher), *(arguments or [])],
            cwd=self.repo_dir,
            env=self._environment(profile, extra_env),
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

    def _dry_run(self, profile="stable", extra_env=None):
        result = self._run(profile, extra_env)
        command = self._command(result)
        train_entry_index = command.index("./runner/train.py")
        config = self._resolved_config(command[train_entry_index + 1 :])
        return result, command, config

    @staticmethod
    def _complete_expectation(profile, partial):
        if profile == "stable":
            return dict(partial)
        expectation = dict(
            TestCrop384Db48DdpSpeedTrial.profile_expectations["cache"]
        )
        if profile == "lowrisk":
            expectation.update(
                TestCrop384Db48DdpSpeedTrial.profile_expectations["lowrisk"]
            )
        expectation.update(partial)
        return expectation

    def test_every_profile_resolves_only_its_documented_deltas(self):
        for profile, partial in self.profile_expectations.items():
            with self.subTest(profile=profile):
                result, command, cfg = self._dry_run(profile)
                expected = self._complete_expectation(profile, partial)

                self.assertIn(f"profile={profile}", result.stdout)
                self.assertIn(
                    "initialization=random raw_checkpoint=<empty> "
                    "ema_checkpoint=<empty>",
                    result.stdout,
                )
                self.assertNotIn("--load_checkpoint_path", command)
                self.assertNotIn("--load_ema_checkpoint_path", command)
                self.assertEqual(cfg.load_checkpoint_path, "")
                self.assertEqual(cfg.load_ema_checkpoint_path, "")

                self.assertEqual(cfg.ddp.gradient_as_bucket_view, expected["bucket_view"])
                self.assertEqual(cfg.adam.fused, expected["adam_fused"])
                self.assertFalse(cfg.adam.use_adamw)
                self.assertEqual(cfg.ema_shadow_device, expected["ema_device"])
                self.assertEqual(
                    cfg.enable_diffusion_shared_vars_cache,
                    expected["cache"],
                )
                self.assertEqual(
                    cfg.diffusion_chunk_size,
                    expected["diffusion_chunk"],
                )
                self.assertEqual(
                    cfg.model.pairformer.blocks_per_ckpt,
                    expected["pairformer_blocks"],
                )
                self.assertEqual(
                    cfg.model.msa_module.blocks_per_ckpt,
                    expected["msa_blocks"],
                )
                self.assertEqual(
                    cfg.model.diffusion_module.blocks_per_ckpt,
                    expected["diffusion_blocks"],
                )
                self.assertEqual(
                    cfg.model.diffusion_module.use_fine_grained_checkpoint,
                    expected["fine_checkpoint"],
                )
                self.assertEqual(
                    cfg.model.pairformer.checkpoint_cpu_offload_every_n,
                    expected["pairformer_offload"],
                )
                self.assertEqual(
                    cfg.model.msa_module.checkpoint_cpu_offload_every_n,
                    expected["msa_offload"],
                )
                self.assertEqual(
                    cfg.model.pairformer.train_pair_transition_chunk_size,
                    expected["transition_chunk"],
                )
                self.assertEqual(
                    cfg.model.diffusion_module.train_pair_transition_chunk_size,
                    expected["transition_chunk"],
                )
                self.assertEqual(
                    cfg.model.pairformer.train_pair_transition_checkpoint_chunks,
                    expected["transition_checkpoint"],
                )
                self.assertEqual(
                    cfg.model.diffusion_module.train_pair_transition_checkpoint_chunks,
                    expected["transition_checkpoint"],
                )

    def test_matched_workload_and_builtin_benchmark_window_are_locked(self):
        result, command, cfg = self._dry_run("stable")
        dataset = cfg.data[self.dataset_name]
        default_source = (
            self.repo_dir.parent
            / "Protenix_train2_fsdp2_msa2048_offload"
            / "data"
            / "benchmark_long_protein_context_1024_800_msa2048"
        )

        self.assertEqual(command[command.index("--nnodes") + 1], "1")
        self.assertEqual(command[command.index("--nproc_per_node") + 1], "8")
        self.assertEqual(cfg.model_name, "protenix-v2")
        self.assertEqual(cfg.distributed_strategy, "ddp")
        self.assertEqual(cfg.optimizer_sharding, "none")
        self.assertEqual(cfg.train_crop_size, 384)
        self.assertEqual(cfg.diffusion_batch_size, 48)
        self.assertEqual(cfg.max_steps, 30)
        self.assertEqual(cfg.benchmark_warmup_steps, 10)
        self.assertEqual(cfg.benchmark_measure_steps, 20)
        self.assertEqual(cfg.checkpoint_interval, -1)
        self.assertTrue(cfg.skip_final_checkpoint)
        self.assertEqual(cfg.eval_interval, 0)
        self.assertFalse(cfg.eval_at_end)
        self.assertAlmostEqual(cfg.lr, 1e-3)

        self.assertEqual(list(cfg.data.train_sets), [self.dataset_name])
        self.assertEqual(list(cfg.data.train_sampler.train_sample_weights), [1.0])
        self.assertEqual(cfg.data.train_sampler.seed, 77)
        self.assertFalse(cfg.data.train_sampler.replacement)
        self.assertEqual(cfg.data.epoch_size, 800)
        self.assertEqual(cfg.data.msa.min_size.train, 2048)
        self.assertEqual(cfg.data.msa.max_size.train, 16384)
        self.assertEqual(cfg.data.msa.sample_cutoff.train, 2048)
        self.assertEqual(cfg.model.msa_module.msa_max_size, 2048)
        self.assertEqual(cfg.model.msa_module.msa_chunk_size, 2048)
        self.assertFalse(cfg.data.msa.pair_msa_as_unpaired_only)
        self.assertFalse(cfg.data.msa.shuffle_pair_msa_when_unpaired_only)
        self.assertEqual(dataset.cropping_configs.max_nuc_tokens, 128)
        self.assertEqual(list(dataset.cropping_configs.method_weights), [0, 1, 0])
        self.assertEqual(dataset.base_info.min_n_token, 1025)
        self.assertFalse(dataset.base_info.random_sample_if_failed)
        self.assertEqual(
            Path(dataset.base_info.indices_fpath),
            default_source / "indices.csv",
        )
        self.assertIn(
            "runner_artifact=benchmark/benchmark_result.json",
            result.stdout,
        )

    def test_grouped_chunk_override_and_transition_default(self):
        _, _, grouped = self._dry_run(
            "grouped",
            {"TRIAL_DIFFUSION_CHUNK": "12"},
        )
        _, _, transition = self._dry_run("transition")

        self.assertEqual(grouped.diffusion_chunk_size, 12)
        self.assertEqual(transition.diffusion_chunk_size, 24)
        self.assertIsNone(
            transition.model.pairformer.train_pair_transition_chunk_size
        )
        self.assertFalse(
            transition.model.pairformer.train_pair_transition_checkpoint_chunks
        )

    def test_pf_group_candidates_only_change_pairformer_grouping(self):
        c48_override = {"TRIAL_DIFFUSION_CHUNK": "48"}
        _, _, transition = self._dry_run("transition", c48_override)
        compared_values = (
            lambda cfg: (
                cfg.ddp.gradient_as_bucket_view,
                cfg.adam.fused,
                cfg.adam.use_adamw,
                cfg.ema_shadow_device,
                cfg.enable_diffusion_shared_vars_cache,
                cfg.diffusion_chunk_size,
                cfg.model.msa_module.blocks_per_ckpt,
                cfg.model.diffusion_module.blocks_per_ckpt,
                cfg.model.diffusion_module.use_fine_grained_checkpoint,
                cfg.model.pairformer.checkpoint_cpu_offload_every_n,
                cfg.model.msa_module.checkpoint_cpu_offload_every_n,
                cfg.model.pairformer.train_pair_transition_chunk_size,
                cfg.model.diffusion_module.train_pair_transition_chunk_size,
                cfg.model.pairformer.train_pair_transition_checkpoint_chunks,
                cfg.model.diffusion_module.train_pair_transition_checkpoint_chunks,
            )
        )
        transition_values = compared_values(transition)

        for profile, expected_group_size in (
            ("pf16", 16),
            ("pf18", 18),
            ("pf19", 19),
            ("pf20", 20),
            ("pf21", 21),
            ("pf22", 22),
            ("pf23", 23),
            ("pf24", 24),
        ):
            with self.subTest(profile=profile):
                _, _, candidate = self._dry_run(profile, c48_override)
                self.assertEqual(
                    candidate.model.pairformer.blocks_per_ckpt,
                    expected_group_size,
                )
                self.assertEqual(compared_values(candidate), transition_values)

    def test_invalid_profile_help_lists_boundary_candidates(self):
        result = self._run("unknown", check=False)

        self.assertEqual(result.returncode, 2)
        self.assertIn("Unsupported SPEED_PROFILE=unknown", result.stderr)
        self.assertIn("pf18, pf19, pf20, pf21, pf22", result.stderr)
        self.assertNotIn("torchrun ", result.stdout)

    def test_no_checkpoint_profiles_and_one_step_screen(self):
        result, _, all_nockpt = self._dry_run(
            "all_nockpt",
            {
                "TRIAL_DIFFUSION_CHUNK": "12",
                "SPEED_SCREEN": "1",
            },
        )

        self.assertIsNone(all_nockpt.model.pairformer.blocks_per_ckpt)
        self.assertIsNone(all_nockpt.model.msa_module.blocks_per_ckpt)
        self.assertIsNone(all_nockpt.model.diffusion_module.blocks_per_ckpt)
        self.assertFalse(
            all_nockpt.model.diffusion_module.use_fine_grained_checkpoint
        )
        self.assertEqual(all_nockpt.diffusion_chunk_size, 12)
        self.assertEqual(all_nockpt.max_steps, 1)
        self.assertEqual(all_nockpt.benchmark_warmup_steps, 0)
        self.assertEqual(all_nockpt.benchmark_measure_steps, 1)
        self.assertIn("screen=1", result.stdout)

    def test_nuc_stress_targets_human_step_31_without_changing_workload(self):
        result, _, cfg = self._dry_run(
            "transition",
            {"SPEED_NUC_STRESS": "1"},
        )
        dataset = cfg.data[self.dataset_name]

        self.assertEqual(cfg.max_steps, 31)
        self.assertEqual(cfg.benchmark_warmup_steps, 30)
        self.assertEqual(cfg.benchmark_measure_steps, 1)
        self.assertEqual(cfg.data.train_sampler.seed, 77)
        self.assertFalse(cfg.data.train_sampler.replacement)
        self.assertEqual(cfg.train_crop_size, 384)
        self.assertEqual(cfg.diffusion_batch_size, 48)
        self.assertEqual(cfg.data.msa.min_size.train, 2048)
        self.assertEqual(cfg.data.msa.sample_cutoff.train, 2048)
        self.assertEqual(cfg.model.msa_module.msa_max_size, 2048)
        self.assertEqual(cfg.model.msa_module.msa_chunk_size, 2048)
        self.assertEqual(dataset.cropping_configs.max_nuc_tokens, 128)
        self.assertFalse(cfg.data.msa.pair_msa_as_unpaired_only)
        self.assertFalse(cfg.data.msa.shuffle_pair_msa_when_unpaired_only)
        self.assertEqual(cfg.model.pairformer.blocks_per_ckpt, 8)
        self.assertEqual(cfg.model.msa_module.blocks_per_ckpt, 4)
        self.assertEqual(cfg.model.diffusion_module.blocks_per_ckpt, 3)
        self.assertIn("screen=0 nuc_stress=1", result.stdout)
        self.assertIn(
            "nuc_stress_target=7d1a rank=5 human_step=31 "
            "zero_based_step=30 sampler_seed=77",
            result.stdout,
        )

    def test_invalid_profiles_and_chunk_overrides_are_rejected(self):
        cases = (
            ("unknown", {}, "Unsupported SPEED_PROFILE"),
            (
                "grouped",
                {"TRIAL_DIFFUSION_CHUNK": "7"},
                "TRIAL_DIFFUSION_CHUNK must be one of",
            ),
            (
                "cache",
                {"TRIAL_DIFFUSION_CHUNK": "24"},
                "only valid for grouped, transition",
            ),
            ("stable", {"SPEED_SCREEN": "2"}, "SPEED_SCREEN must be 0 or 1"),
            (
                "stable",
                {"SPEED_NUC_STRESS": "2"},
                "SPEED_NUC_STRESS must be 0 or 1",
            ),
            (
                "stable",
                {"SPEED_SCREEN": "1", "SPEED_NUC_STRESS": "1"},
                "mutually exclusive",
            ),
        )
        for profile, extra_env, expected_error in cases:
            with self.subTest(profile=profile, extra_env=extra_env):
                result = self._run(profile, extra_env, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected_error, result.stderr)
                self.assertNotIn("torchrun ", result.stdout)

    def test_checkpoint_sources_and_cli_overrides_are_rejected(self):
        for variable_name in (
            "CHECKPOINT_PATH",
            "LOAD_CHECKPOINT_PATH",
            "LOAD_EMA_CHECKPOINT_PATH",
            "PROTENIX_RESUME_RAW_CHECKPOINT",
        ):
            with self.subTest(variable_name=variable_name):
                result = self._run(
                    extra_env={variable_name: "/tmp/forbidden.pt"},
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("checkpoint environment variable", result.stderr)

        result = self._run(
            arguments=["--load_checkpoint_path", "/tmp/forbidden.pt"],
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("rejects checkpoint/resume parameter", result.stderr)

        result = self._run(arguments=["--lr", "0.1"], check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("does not accept command-line overrides", result.stderr)

    def test_single_node_eight_rank_contract_is_enforced(self):
        for extra_env, expected_error in (
            ({"NPROC_PER_NODE": "4"}, "requires 8 local ranks"),
            ({"NNODES": "2"}, "requires one node"),
            ({"NODE_RANK": "1"}, "requires node rank 0"),
        ):
            with self.subTest(extra_env=extra_env):
                result = self._run(extra_env=extra_env, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected_error, result.stderr)
                self.assertNotIn("torchrun ", result.stdout)

    def test_trial_identity_master_port_and_dataset_override_are_auditable(self):
        custom_dataset_dir = "/tmp/custom_crop384_msa2048"
        result, command, cfg = self._dry_run(
            "grouped",
            {
                "TRIAL_ID": "repeat_b",
                "MASTER_PORT": "30123",
                "BENCHMARK_DATASET_DIR": custom_dataset_dir,
            },
        )

        self.assertEqual(command[command.index("--master_port") + 1], "30123")
        self.assertEqual(cfg.run_name, "crop384_db48_ddp_speed_grouped_repeat_b")
        self.assertEqual(
            Path(cfg.data[self.dataset_name].base_info.indices_fpath),
            Path(custom_dataset_dir) / "indices.csv",
        )
        self.assertIn("trial_id=repeat_b", result.stdout)
        self.assertIn(f"source_dir={custom_dataset_dir}", result.stdout)


if __name__ == "__main__":
    unittest.main()
