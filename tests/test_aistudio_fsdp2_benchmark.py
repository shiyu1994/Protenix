import os
import shlex
import subprocess
import unittest
from pathlib import Path


class TestAIStudioFSDP2Benchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_dir = Path(__file__).resolve().parents[1]

    def _dry_run(
        self, variant: str, inherited_overrides: dict[str, str] | None = None
    ) -> list[str]:
        env = os.environ.copy()
        for name in [
            "distributed_strategy",
            "optimizer_sharding",
            "ema_shadow_device",
            "fsdp_mesh",
            "load_checkpoint_path",
            "load_ema_checkpoint_path",
            "LOAD_CHECKPOINT_PATH",
            "LOAD_EMA_CHECKPOINT_PATH",
            "train_sample_weights",
            "msa_min_size_train",
            "msa_sample_cutoff_train",
            "model_c_z",
            "model_c_m",
            "pairformer_blocks_per_ckpt",
            "pair_transition_chunk_size",
            "pair_transition_checkpoint_chunks",
            "msa_chunk_size",
            "pairformer_checkpoint_cpu_offload_every_n",
            "msa_checkpoint_cpu_offload_every_n",
            "base_dir",
            "run_name",
            "nproc_per_node",
            "nnodes",
            "node_rank",
            "master_addr",
            "master_port",
            "BENCHMARK_CROP_SIZE",
            "BENCHMARK_TRAIN_CROP_SIZE",
            "BENCHMARK_DATASET_NAME",
            "BENCHMARK_DATASET_ROWS",
            "BENCHMARK_DATASET_DIR",
            "BENCHMARK_MAX_ROWS",
            "BENCHMARK_TRAIN_SAMPLER_REPLACEMENT",
            "BENCHMARK_TRAIN_SAMPLER_SEED",
            "BENCHMARK_RANDOM_SAMPLE_IF_FAILED",
            "BENCHMARK_WORKLOAD_PROFILE",
            "BENCHMARK_MSA_MIN_SIZE_TRAIN",
            "BENCHMARK_MSA_SAMPLE_CUTOFF_TRAIN",
            "BENCHMARK_MSA_MAX_SIZE_TRAIN",
            "BENCHMARK_MSA_CHUNK_SIZE",
            "BENCHMARK_PAIRFORMER_CHECKPOINT_CPU_OFFLOAD_EVERY_N",
            "BENCHMARK_MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N",
            "BENCHMARK_DIFFUSION_BATCH_SIZE",
            "BENCHMARK_MAX_NUC_TOKENS",
            "BENCHMARK_PAIR_MSA_AS_UNPAIRED_ONLY",
            "BENCHMARK_SHUFFLE_PAIR_MSA",
            "BENCHMARK_LR",
            "BENCHMARK_LOAD_CHECKPOINT_PATH",
            "BENCHMARK_LOAD_EMA_CHECKPOINT_PATH",
            "PROTENIX_BENCHMARK_PHASE_MEMORY",
            "PROTENIX_BENCHMARK_DIR",
        ]:
            env.pop(name, None)
        env.update(
            {
                "BENCHMARK_ID": f"test_{variant}",
                "BENCHMARK_VARIANT": variant,
                "BENCHMARK_WARMUP_STEPS": "2",
                "BENCHMARK_MEASURE_STEPS": "3",
                "NNODES": "1",
                "NODE_RANK": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29690",
                "NPROC_PER_NODE": "1",
                "DRY_RUN": "1",
                "SKIP_PATH_CHECK": "1",
            }
        )
        env.update(inherited_overrides or {})
        result = subprocess.run(
            ["bash", str(self.repo_dir / "benchmark.sh")],
            cwd=self.repo_dir,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        command_line = next(
            line for line in result.stdout.splitlines() if line.startswith("torchrun ")
        )
        return shlex.split(command_line)

    @staticmethod
    def _value(args: list[str], option: str) -> str:
        return args[args.index(option) + 1]

    def test_crop384_dataset_alias_reuses_audited_msa2048_artifact(self):
        from configs.configs_data import data_configs

        dataset = data_configs[
            "benchmark_long_protein_context_384_800_msa2048"
        ]
        index_path = Path(dataset["base_info"]["indices_fpath"])
        self.assertEqual(dataset["base_info"]["min_n_token"], 1025)
        self.assertFalse(dataset["base_info"]["random_sample_if_failed"])
        self.assertEqual(
            index_path.parent.name,
            "benchmark_long_protein_context_1024_800_msa2048",
        )

    def test_ddp_full_resolves_matched_exact_crop_settings(self):
        args = self._dry_run("ddp_full")
        expected = {
            "--distributed_strategy": "ddp",
            "--optimizer_sharding": "none",
            "--ema_shadow_device": "cpu",
            "--fsdp.mesh": "auto",
            "--fsdp.param_dtype": "fp32",
            "--fsdp.reduce_dtype": "fp32",
            "--fsdp.reshard_after_forward": "true",
            "--benchmark_warmup_steps": "2",
            "--benchmark_measure_steps": "3",
            "--max_steps": "5",
            "--skip_final_checkpoint": "true",
            "--eval_at_end": "false",
            "--load_params_only": "true",
            "--train_crop_size": "768",
            "--diffusion_batch_size": "24",
            "--diffusion_chunk_size": "4",
            "--c_z": "256",
            "--model.msa_module.c_m": "128",
            "--model.pairformer.blocks_per_ckpt": "2",
            "--model.pairformer.train_pair_transition_chunk_size": "128",
            "--model.pairformer.train_pair_transition_checkpoint_chunks": "true",
            "--model.diffusion_module.train_pair_transition_chunk_size": "128",
            "--model.diffusion_module.train_pair_transition_checkpoint_chunks": "true",
            "--data.train_sets": "benchmark_long_protein_context_768",
            "--data.train_sampler.train_sample_weights": "1.0",
            "--data.train_sampler.replacement": "true",
            "--data.test_sets": "benchmark_long_protein_context_768",
            "--data.num_dl_workers": "4",
            "--data.epoch_size": "10000",
            "--data.benchmark_long_protein_context_768.base_info.min_n_token": "769",
            "--data.benchmark_long_protein_context_768.base_info.random_sample_if_failed": "true",
            "--data.benchmark_long_protein_context_768.cropping_configs.method_weights": "0,1,0",
            "--data.benchmark_long_protein_context_768.cropping_configs.max_nuc_tokens": "-1",
            "--data.benchmark_long_protein_context_768.cropping_configs.contiguous_crop_complete_lig": "false",
            "--data.benchmark_long_protein_context_768.cropping_configs.spatial_crop_complete_lig": "false",
            "--data.benchmark_long_protein_context_768.cropping_configs.drop_last": "false",
            "--data.benchmark_long_protein_context_768.cropping_configs.remove_metal": "false",
            "--data.msa.min_size.train": "16",
            "--data.msa.sample_cutoff.train": "32",
            "--data.msa.pair_msa_as_unpaired_only": "true",
            "--data.msa.shuffle_pair_msa_when_unpaired_only": "true",
            "--triangle_attention": "cuequivariance",
            "--triangle_multiplicative": "cuequivariance",
        }
        for option, value in expected.items():
            self.assertEqual(self._value(args, option), value)
        self.assertNotIn("--train_post_pairformer_only", args)
        self.assertNotIn("--model.msa_module.msa_chunk_size", args)
        self.assertNotIn(
            "--model.pairformer.checkpoint_cpu_offload_every_n", args
        )
        self.assertNotIn(
            "--model.msa_module.checkpoint_cpu_offload_every_n", args
        )

    def test_fsdp2_full_changes_strategy_and_keeps_matched_settings(self):
        ddp_args = self._dry_run("ddp_full")
        fsdp_args = self._dry_run("fsdp2_full")
        self.assertEqual(self._value(fsdp_args, "--distributed_strategy"), "fsdp2")
        ddp_args[ddp_args.index("--distributed_strategy") + 1] = "fsdp2"
        for option in ["--run_name", "--base_dir"]:
            ddp_args[ddp_args.index(option) + 1] = "MATCHED_OUTPUT"
            fsdp_args[fsdp_args.index(option) + 1] = "MATCHED_OUTPUT"
        self.assertEqual(ddp_args, fsdp_args)

    def test_crop1024_resolves_exact_profile_for_both_strategies(self):
        resolved = []
        for variant in ("ddp_full", "fsdp2_full"):
            args = self._dry_run(variant, {"BENCHMARK_CROP_SIZE": "1024"})
            resolved.append(args)
            expected = {
                "--train_crop_size": "1024",
                "--data.train_sets": "benchmark_long_protein_context_1024",
                "--data.test_sets": "benchmark_long_protein_context_1024",
                "--data.benchmark_long_protein_context_1024.base_info.min_n_token": "1025",
                "--data.benchmark_long_protein_context_1024.cropping_configs.method_weights": "0,1,0",
            }
            for option, value in expected.items():
                self.assertEqual(self._value(args, option), value)
            index_option = (
                "--data.benchmark_long_protein_context_1024.base_info.indices_fpath"
            )
            self.assertTrue(
                self._value(args, index_option).endswith(
                    "data/benchmark_long_protein_context_1024/indices.csv"
                )
            )

        resolved[0][resolved[0].index("--distributed_strategy") + 1] = "fsdp2"
        for option in ["--run_name", "--base_dir"]:
            resolved[0][resolved[0].index(option) + 1] = "MATCHED_OUTPUT"
            resolved[1][resolved[1].index(option) + 1] = "MATCHED_OUTPUT"
        self.assertEqual(resolved[0], resolved[1])

    def test_invalid_or_conflicting_crop_profile_is_rejected(self):
        for overrides, message in [
            (
                {"BENCHMARK_CROP_SIZE": "999"},
                "must be 384, 768, 960, or 1024",
            ),
            (
                {
                    "BENCHMARK_CROP_SIZE": "1024",
                    "BENCHMARK_TRAIN_CROP_SIZE": "768",
                },
                "disagree",
            ),
        ]:
            env = os.environ.copy()
            env.update(
                {
                    "BENCHMARK_ID": "invalid_crop",
                    "BENCHMARK_VARIANT": "ddp_full",
                    "DRY_RUN": "1",
                    **overrides,
                }
            )
            result = subprocess.run(
                ["bash", str(self.repo_dir / "benchmark.sh")],
                cwd=self.repo_dir,
                env=env,
                check=False,
                text=True,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn(message, result.stderr)

    def test_crop1024_800_resolves_one_exact_global_epoch(self):
        args = self._dry_run(
            "fsdp2_full",
            {
                "BENCHMARK_CROP_SIZE": "1024",
                "BENCHMARK_DATASET_ROWS": "800",
                "BENCHMARK_WARMUP_STEPS": "10",
                "BENCHMARK_MEASURE_STEPS": "90",
                "NPROC_PER_NODE": "8",
            },
        )
        expected = {
            "--train_crop_size": "1024",
            "--max_steps": "100",
            "--benchmark_warmup_steps": "10",
            "--benchmark_measure_steps": "90",
            "--data.train_sets": "benchmark_long_protein_context_1024_800",
            "--data.test_sets": "benchmark_long_protein_context_1024_800",
            "--data.epoch_size": "800",
            "--data.train_sampler.replacement": "false",
            "--data.benchmark_long_protein_context_1024_800.base_info.min_n_token": "1025",
            "--data.benchmark_long_protein_context_1024_800.base_info.random_sample_if_failed": "false",
        }
        for option, value in expected.items():
            self.assertEqual(self._value(args, option), value)
        index_option = (
            "--data.benchmark_long_protein_context_1024_800.base_info.indices_fpath"
        )
        self.assertTrue(
            self._value(args, index_option).endswith(
                "data/benchmark_long_protein_context_1024_800/indices.csv"
            )
        )

    def test_msa2048_offload_profile_locks_full_100_step_workload(self):
        args = self._dry_run(
            "fsdp2_full",
            {
                "BENCHMARK_WORKLOAD_PROFILE": "crop1024_800_msa2048_offload",
                "BENCHMARK_WARMUP_STEPS": "10",
                "BENCHMARK_MEASURE_STEPS": "90",
                "NPROC_PER_NODE": "8",
            },
        )
        expected = {
            "--distributed_strategy": "fsdp2",
            "--train_crop_size": "1024",
            "--max_steps": "100",
            "--benchmark_warmup_steps": "10",
            "--benchmark_measure_steps": "90",
            "--data.train_sets": "benchmark_long_protein_context_1024_800_msa2048",
            "--data.test_sets": "benchmark_long_protein_context_1024_800_msa2048",
            "--data.epoch_size": "800",
            "--data.train_sampler.replacement": "false",
            "--data.benchmark_long_protein_context_1024_800_msa2048.base_info.min_n_token": "1025",
            "--data.benchmark_long_protein_context_1024_800_msa2048.base_info.random_sample_if_failed": "false",
            "--data.msa.min_size.train": "2048",
            "--data.msa.max_size.train": "16384",
            "--data.msa.sample_cutoff.train": "2048",
            "--model.msa_module.msa_chunk_size": "128",
            "--model.pairformer.checkpoint_cpu_offload_every_n": "2",
            "--model.msa_module.checkpoint_cpu_offload_every_n": "1",
        }
        for option, value in expected.items():
            self.assertEqual(self._value(args, option), value)
        self.assertTrue(
            self._value(
                args,
                "--data.benchmark_long_protein_context_1024_800_msa2048.base_info.indices_fpath",
            ).endswith(
                "data/benchmark_long_protein_context_1024_800_msa2048/indices.csv"
            )
        )

    def test_crop960_msa2048_profile_reuses_audited_artifacts_for_100_steps(self):
        args = self._dry_run(
            "fsdp2_full",
            {
                "BENCHMARK_WORKLOAD_PROFILE": "crop960_800_msa2048_offload",
                "BENCHMARK_WARMUP_STEPS": "10",
                "BENCHMARK_MEASURE_STEPS": "90",
                "NPROC_PER_NODE": "8",
            },
        )
        dataset = "benchmark_long_protein_context_960_800_msa2048"
        expected = {
            "--distributed_strategy": "fsdp2",
            "--train_crop_size": "960",
            "--max_steps": "100",
            "--benchmark_warmup_steps": "10",
            "--benchmark_measure_steps": "90",
            "--data.train_sets": dataset,
            "--data.test_sets": dataset,
            "--data.epoch_size": "800",
            "--data.train_sampler.replacement": "false",
            f"--data.{dataset}.base_info.min_n_token": "1025",
            f"--data.{dataset}.base_info.random_sample_if_failed": "false",
            "--data.msa.min_size.train": "2048",
            "--data.msa.max_size.train": "16384",
            "--data.msa.sample_cutoff.train": "2048",
            "--model.msa_module.msa_chunk_size": "128",
            "--model.pairformer.checkpoint_cpu_offload_every_n": "2",
            "--model.msa_module.checkpoint_cpu_offload_every_n": "1",
        }
        for option, value in expected.items():
            self.assertEqual(self._value(args, option), value)
        self.assertTrue(
            self._value(args, f"--data.{dataset}.base_info.indices_fpath").endswith(
                "data/benchmark_long_protein_context_1024_800_msa2048/indices.csv"
            )
        )

    def test_crop960_nuc128_db32_profile_resolves_100_step_experiment(self):
        args = self._dry_run(
            "fsdp2_full",
            {
                "BENCHMARK_WORKLOAD_PROFILE": "crop960_800_msa2048_offload_nuc128_db32",
                "BENCHMARK_WARMUP_STEPS": "10",
                "BENCHMARK_MEASURE_STEPS": "90",
                "NPROC_PER_NODE": "8",
            },
        )
        dataset = "benchmark_long_protein_context_960_800_msa2048"
        expected = {
            "--train_crop_size": "960",
            "--diffusion_batch_size": "32",
            "--max_steps": "100",
            "--benchmark_warmup_steps": "10",
            "--benchmark_measure_steps": "90",
            f"--data.{dataset}.cropping_configs.max_nuc_tokens": "128",
            "--data.msa.sample_cutoff.train": "2048",
        }
        for option, value in expected.items():
            self.assertEqual(self._value(args, option), value)

    def test_crop1024_nuc128_db32_profile_targets_7d1a_on_step_31(self):
        args = self._dry_run(
            "fsdp2_full",
            {
                "BENCHMARK_WORKLOAD_PROFILE": (
                    "crop1024_800_msa2048_offload_nuc128_db32_31step_7d1a"
                ),
                "BENCHMARK_WARMUP_STEPS": "10",
                "BENCHMARK_MEASURE_STEPS": "21",
                "NPROC_PER_NODE": "8",
            },
        )
        dataset = "benchmark_long_protein_context_1024_800_msa2048"
        expected = {
            "--seed": "42",
            "--train_crop_size": "1024",
            "--diffusion_batch_size": "32",
            "--max_steps": "31",
            "--benchmark_warmup_steps": "10",
            "--benchmark_measure_steps": "21",
            "--data.train_sampler.seed": "77",
            "--data.train_sampler.replacement": "false",
            f"--data.{dataset}.cropping_configs.max_nuc_tokens": "128",
            "--data.msa.sample_cutoff.train": "2048",
        }
        for option, value in expected.items():
            self.assertEqual(self._value(args, option), value)

    def test_crop1024_nuc256_changes_only_nucleic_cap(self):
        args = self._dry_run(
            "fsdp2_full",
            {
                "BENCHMARK_WORKLOAD_PROFILE": (
                    "crop1024_800_msa2048_offload_nuc256_db32_31step_7d1a"
                ),
                "BENCHMARK_WARMUP_STEPS": "10",
                "BENCHMARK_MEASURE_STEPS": "21",
                "NPROC_PER_NODE": "8",
            },
        )
        dataset = "benchmark_long_protein_context_1024_800_msa2048"
        expected = {
            "--seed": "42",
            "--train_crop_size": "1024",
            "--diffusion_batch_size": "32",
            "--max_steps": "31",
            "--benchmark_warmup_steps": "10",
            "--benchmark_measure_steps": "21",
            "--data.train_sampler.seed": "77",
            f"--data.{dataset}.cropping_configs.max_nuc_tokens": "256",
            "--data.msa.sample_cutoff.train": "2048",
            "--model.pairformer.checkpoint_cpu_offload_every_n": "2",
            "--model.msa_module.checkpoint_cpu_offload_every_n": "1",
        }
        for option, value in expected.items():
            self.assertEqual(self._value(args, option), value)

    def test_crop1024_nuc256_chunk256_profile_runs_exactly_ten_steps(self):
        profile = "crop1024_800_msa2048_offload_nuc256_db32_chunk256_10step"
        args = self._dry_run(
            "fsdp2_full",
            {
                "BENCHMARK_WORKLOAD_PROFILE": profile,
                "BENCHMARK_WARMUP_STEPS": "2",
                "BENCHMARK_MEASURE_STEPS": "8",
                "NPROC_PER_NODE": "8",
            },
        )
        dataset = "benchmark_long_protein_context_1024_800_msa2048"
        expected = {
            "--seed": "42",
            "--train_crop_size": "1024",
            "--diffusion_batch_size": "32",
            "--max_steps": "10",
            "--benchmark_warmup_steps": "2",
            "--benchmark_measure_steps": "8",
            "--data.train_sampler.seed": "77",
            f"--data.{dataset}.cropping_configs.max_nuc_tokens": "256",
            "--data.msa.min_size.train": "2048",
            "--data.msa.sample_cutoff.train": "2048",
            "--model.msa_module.msa_chunk_size": "256",
            "--model.pairformer.checkpoint_cpu_offload_every_n": "2",
            "--model.msa_module.checkpoint_cpu_offload_every_n": "1",
        }
        for option, value in expected.items():
            self.assertEqual(self._value(args, option), value)

        env = os.environ.copy()
        env.update(
            {
                "BENCHMARK_ID": "invalid_chunk256_profile",
                "BENCHMARK_VARIANT": "fsdp2_full",
                "BENCHMARK_WORKLOAD_PROFILE": profile,
                "BENCHMARK_MSA_CHUNK_SIZE": "128",
                "NPROC_PER_NODE": "8",
                "NNODES": "1",
                "DRY_RUN": "1",
            }
        )
        result = subprocess.run(
            ["bash", str(self.repo_dir / "benchmark.sh")],
            cwd=self.repo_dir,
            env=env,
            check=False,
            text=True,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("model/checkpoint/offload", result.stderr)

    def test_crop1024_nuc256_chunk2048_disables_msa_row_splitting(self):
        profile = "crop1024_800_msa2048_offload_nuc256_db32_chunk2048_10step"
        args = self._dry_run(
            "fsdp2_full",
            {
                "BENCHMARK_WORKLOAD_PROFILE": profile,
                "BENCHMARK_WARMUP_STEPS": "2",
                "BENCHMARK_MEASURE_STEPS": "8",
                "NPROC_PER_NODE": "8",
            },
        )
        dataset = "benchmark_long_protein_context_1024_800_msa2048"
        expected = {
            "--seed": "42",
            "--train_crop_size": "1024",
            "--diffusion_batch_size": "32",
            "--max_steps": "10",
            "--benchmark_warmup_steps": "2",
            "--benchmark_measure_steps": "8",
            "--data.train_sampler.seed": "77",
            f"--data.{dataset}.cropping_configs.max_nuc_tokens": "256",
            "--data.msa.min_size.train": "2048",
            "--data.msa.sample_cutoff.train": "2048",
            "--model.msa_module.msa_chunk_size": "2048",
            "--model.pairformer.checkpoint_cpu_offload_every_n": "2",
            "--model.msa_module.checkpoint_cpu_offload_every_n": "1",
        }
        for option, value in expected.items():
            self.assertEqual(self._value(args, option), value)

    def test_crop1024_no_pairformer_offload_is_matched_to_7d1a_profile(self):
        args = self._dry_run(
            "fsdp2_full",
            {
                "BENCHMARK_WORKLOAD_PROFILE": (
                    "crop1024_800_msa2048_msaoffload_no_pairformer_offload_"
                    "nuc128_db32_31step_7d1a"
                ),
                "BENCHMARK_WARMUP_STEPS": "10",
                "BENCHMARK_MEASURE_STEPS": "21",
                "NPROC_PER_NODE": "8",
            },
        )
        expected = {
            "--train_crop_size": "1024",
            "--diffusion_batch_size": "32",
            "--max_steps": "31",
            "--benchmark_warmup_steps": "10",
            "--benchmark_measure_steps": "21",
            "--data.train_sampler.seed": "77",
            "--data.msa.min_size.train": "2048",
            "--data.msa.sample_cutoff.train": "2048",
            "--model.pairformer.blocks_per_ckpt": "2",
            "--model.msa_module.checkpoint_cpu_offload_every_n": "1",
        }
        for option, value in expected.items():
            self.assertEqual(self._value(args, option), value)
        self.assertNotIn(
            "--model.pairformer.checkpoint_cpu_offload_every_n", args
        )

    def test_crop960_msa2048_profile_rejects_crop_drift(self):
        env = os.environ.copy()
        env.update(
            {
                "BENCHMARK_ID": "invalid_crop960_profile",
                "BENCHMARK_VARIANT": "fsdp2_full",
                "BENCHMARK_WORKLOAD_PROFILE": "crop960_800_msa2048_offload",
                "BENCHMARK_CROP_SIZE": "1024",
                "NPROC_PER_NODE": "8",
                "NNODES": "1",
                "DRY_RUN": "1",
            }
        )
        result = subprocess.run(
            ["bash", str(self.repo_dir / "benchmark.sh")],
            cwd=self.repo_dir,
            env=env,
            check=False,
            text=True,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires crop=960", result.stderr)

    def test_msa2048_offload_staged_profiles_lock_step_windows(self):
        for profile, warmup, measure in [
            ("crop1024_800_msa2048_offload_smoke", "0", "1"),
            ("crop1024_800_msa2048_offload_matched", "10", "20"),
        ]:
            args = self._dry_run(
                "fsdp2_full",
                {
                    "BENCHMARK_WORKLOAD_PROFILE": profile,
                    "BENCHMARK_WARMUP_STEPS": warmup,
                    "BENCHMARK_MEASURE_STEPS": measure,
                    "NPROC_PER_NODE": "8",
                },
            )
            self.assertEqual(self._value(args, "--benchmark_warmup_steps"), warmup)
            self.assertEqual(self._value(args, "--benchmark_measure_steps"), measure)
            self.assertEqual(
                self._value(args, "--max_steps"), str(int(warmup) + int(measure))
            )
            self.assertEqual(self._value(args, "--train_crop_size"), "1024")
            self.assertEqual(
                self._value(args, "--data.train_sets"),
                "benchmark_long_protein_context_1024_800_msa2048",
            )
            self.assertEqual(
                self._value(args, "--data.msa.sample_cutoff.train"), "2048"
            )
            self.assertEqual(
                self._value(args, "--model.msa_module.msa_chunk_size"), "128"
            )

    def test_db40_paired_chunk2048_profiles_lock_exact_workload(self):
        checkpoint = (
            "/mnt/project_abd/klp/proteinx_output/"
            "protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/"
            "checkpoints/999_ema_0.999.pt"
        )
        dataset = "benchmark_long_protein_context_1024_800_msa2048"
        profile_prefix = (
            "crop1024_800_msa2048_offload_nuc128_db40_paired_chunk2048"
        )
        for suffix, warmup, measure in (
            ("smoke", "0", "1"),
            ("matched", "10", "20"),
        ):
            args = self._dry_run(
                "fsdp2_full",
                {
                    "BENCHMARK_WORKLOAD_PROFILE": f"{profile_prefix}_{suffix}",
                    "BENCHMARK_WARMUP_STEPS": warmup,
                    "BENCHMARK_MEASURE_STEPS": measure,
                    "NPROC_PER_NODE": "8",
                },
            )
            expected = {
                "--distributed_strategy": "fsdp2",
                "--train_crop_size": "1024",
                "--diffusion_batch_size": "40",
                "--diffusion_chunk_size": "4",
                "--benchmark_warmup_steps": warmup,
                "--benchmark_measure_steps": measure,
                "--max_steps": str(int(warmup) + int(measure)),
                "--warmup_steps": "2000",
                "--lr": "0.000003",
                "--load_checkpoint_path": checkpoint,
                "--load_ema_checkpoint_path": checkpoint,
                "--data.train_sets": dataset,
                "--data.test_sets": dataset,
                "--data.train_sampler.seed": "77",
                "--data.train_sampler.replacement": "false",
                "--data.msa.min_size.train": "2048",
                "--data.msa.max_size.train": "16384",
                "--data.msa.sample_cutoff.train": "2048",
                "--data.msa.pair_msa_as_unpaired_only": "false",
                "--data.msa.shuffle_pair_msa_when_unpaired_only": "false",
                "--model.msa_module.msa_chunk_size": "2048",
                "--model.pairformer.checkpoint_cpu_offload_every_n": "2",
                "--model.msa_module.checkpoint_cpu_offload_every_n": "1",
                f"--data.{dataset}.cropping_configs.max_nuc_tokens": "128",
            }
            for option, value in expected.items():
                self.assertEqual(self._value(args, option), value)
            index_path = self._value(
                args, f"--data.{dataset}.base_info.indices_fpath"
            )
            self.assertEqual(
                index_path,
                "/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/"
                "Protenix_train2_fsdp2_msa2048_offload/data/"
                "benchmark_long_protein_context_1024_800_msa2048/indices.csv",
            )

    def test_db40_paired_chunk2048_profile_rejects_drift(self):
        profile = (
            "crop1024_800_msa2048_offload_nuc128_db40_"
            "paired_chunk2048_matched"
        )
        official_checkpoint = (
            "/mnt/project_abd/public_databases/protenix.20260101/"
            "checkpoint/protenix-v2.pt"
        )
        overrides = (
            {"BENCHMARK_DIFFUSION_BATCH_SIZE": "32"},
            {"BENCHMARK_MAX_NUC_TOKENS": "256"},
            {"BENCHMARK_MSA_CHUNK_SIZE": "128"},
            {"BENCHMARK_PAIR_MSA_AS_UNPAIRED_ONLY": "true"},
            {"BENCHMARK_SHUFFLE_PAIR_MSA": "true"},
            {"BENCHMARK_LR": "0.00001"},
            {"BENCHMARK_LOAD_CHECKPOINT_PATH": official_checkpoint},
            {"PROTENIX_BENCHMARK_PHASE_MEMORY": "true"},
        )
        for index, override in enumerate(overrides):
            env = os.environ.copy()
            env.update(
                {
                    "BENCHMARK_ID": f"invalid_db40_profile_{index}",
                    "BENCHMARK_VARIANT": "fsdp2_full",
                    "BENCHMARK_WORKLOAD_PROFILE": profile,
                    "BENCHMARK_WARMUP_STEPS": "10",
                    "BENCHMARK_MEASURE_STEPS": "20",
                    "NPROC_PER_NODE": "8",
                    "NNODES": "1",
                    "DRY_RUN": "1",
                    **override,
                }
            )
            result = subprocess.run(
                ["bash", str(self.repo_dir / "benchmark.sh")],
                cwd=self.repo_dir,
                env=env,
                check=False,
                text=True,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(
                result.returncode,
                2,
                msg=f"override unexpectedly accepted: {override}",
            )

    def test_crop384_db48_random_init_profiles_lock_exact_workload(self):
        dataset = "benchmark_long_protein_context_384_800_msa2048"
        profile_prefix = (
            "crop384_800_msa2048_randominit_nuc128_db48_"
            "paired_chunk2048"
        )
        for suffix, warmup, measure in (
            ("smoke", "0", "1"),
            ("matched", "10", "20"),
        ):
            resolved = []
            for variant, strategy in (
                ("ddp_full", "ddp"),
                ("fsdp2_full", "fsdp2"),
            ):
                args = self._dry_run(
                    variant,
                    {
                        "BENCHMARK_WORKLOAD_PROFILE": (
                            f"{profile_prefix}_{suffix}"
                        ),
                        "BENCHMARK_WARMUP_STEPS": warmup,
                        "BENCHMARK_MEASURE_STEPS": measure,
                        "NPROC_PER_NODE": "8",
                    },
                )
                expected = {
                    "--distributed_strategy": strategy,
                    "--train_crop_size": "384",
                    "--lr": "0.001",
                    "--diffusion_batch_size": "48",
                    "--diffusion_chunk_size": "4",
                    "--benchmark_warmup_steps": warmup,
                    "--benchmark_measure_steps": measure,
                    "--max_steps": str(int(warmup) + int(measure)),
                    "--data.train_sets": dataset,
                    "--data.test_sets": dataset,
                    "--data.train_sampler.seed": "77",
                    "--data.train_sampler.replacement": "false",
                    "--data.msa.min_size.train": "2048",
                    "--data.msa.max_size.train": "16384",
                    "--data.msa.sample_cutoff.train": "2048",
                    "--data.msa.pair_msa_as_unpaired_only": "false",
                    "--data.msa.shuffle_pair_msa_when_unpaired_only": "false",
                    "--model.msa_module.msa_chunk_size": "2048",
                    "--model.pairformer.blocks_per_ckpt": "2",
                    "--model.pairformer.checkpoint_cpu_offload_every_n": "2",
                    "--model.msa_module.checkpoint_cpu_offload_every_n": "1",
                    "--model.pairformer.train_pair_transition_chunk_size": "128",
                    "--model.pairformer.train_pair_transition_checkpoint_chunks": "true",
                    f"--data.{dataset}.cropping_configs.max_nuc_tokens": "128",
                }
                for option, value in expected.items():
                    self.assertEqual(self._value(args, option), value)
                self.assertNotIn("--load_checkpoint_path", args)
                self.assertNotIn("--load_ema_checkpoint_path", args)
                index_path = self._value(
                    args, f"--data.{dataset}.base_info.indices_fpath"
                )
                self.assertTrue(
                    index_path.endswith(
                        "data/benchmark_long_protein_context_1024_800_msa2048/"
                        "indices.csv"
                    )
                )
                resolved.append(args)

            resolved[0][
                resolved[0].index("--distributed_strategy") + 1
            ] = "fsdp2"
            for option in ("--run_name", "--base_dir"):
                resolved[0][resolved[0].index(option) + 1] = "MATCHED"
                resolved[1][resolved[1].index(option) + 1] = "MATCHED"
            self.assertEqual(resolved[0], resolved[1])

    def test_crop384_random_init_profile_rejects_checkpoint_or_shape_drift(self):
        profile = (
            "crop384_800_msa2048_randominit_nuc128_db48_"
            "paired_chunk2048_matched"
        )
        checkpoint = (
            "/mnt/project_abd/public_databases/protenix.20260101/"
            "checkpoint/protenix-v2.pt"
        )
        overrides = (
            {"BENCHMARK_LOAD_CHECKPOINT_PATH": checkpoint},
            {"BENCHMARK_LOAD_EMA_CHECKPOINT_PATH": checkpoint},
            {"LOAD_CHECKPOINT_PATH": checkpoint},
            {"load_ema_checkpoint_path": checkpoint},
            {"BENCHMARK_LR": "0.00001"},
            {"BENCHMARK_DIFFUSION_BATCH_SIZE": "40"},
            {"BENCHMARK_CROP_SIZE": "1024"},
            {"BENCHMARK_MAX_NUC_TOKENS": "256"},
            {"BENCHMARK_MSA_SAMPLE_CUTOFF_TRAIN": "1024"},
            {"BENCHMARK_MSA_CHUNK_SIZE": "128"},
            {"BENCHMARK_PAIR_MSA_AS_UNPAIRED_ONLY": "true"},
            {"BENCHMARK_SHUFFLE_PAIR_MSA": "true"},
        )
        for index, override in enumerate(overrides):
            env = os.environ.copy()
            env.update(
                {
                    "BENCHMARK_ID": f"invalid_crop384_profile_{index}",
                    "BENCHMARK_VARIANT": "ddp_full",
                    "BENCHMARK_WORKLOAD_PROFILE": profile,
                    "BENCHMARK_WARMUP_STEPS": "10",
                    "BENCHMARK_MEASURE_STEPS": "20",
                    "NPROC_PER_NODE": "8",
                    "NNODES": "1",
                    "DRY_RUN": "1",
                    **override,
                }
            )
            result = subprocess.run(
                ["bash", str(self.repo_dir / "benchmark.sh")],
                cwd=self.repo_dir,
                env=env,
                check=False,
                text=True,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(
                result.returncode,
                2,
                msg=f"override unexpectedly accepted: {override}",
            )

    def test_msa2048_offload_profile_rejects_workload_drift(self):
        for override, expected_message in [
            ({"BENCHMARK_CROP_SIZE": "768"}, "Supported exact-crop"),
            ({"BENCHMARK_MSA_SAMPLE_CUTOFF_TRAIN": "1024"}, "training/MSA"),
            ({"BENCHMARK_MSA_CHUNK_SIZE": "256"}, "model/checkpoint/offload"),
            (
                {"BENCHMARK_MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N": "2"},
                "model/checkpoint/offload",
            ),
        ]:
            env = os.environ.copy()
            env.update(
                {
                    "BENCHMARK_ID": "invalid_msa2048_profile",
                    "BENCHMARK_VARIANT": "fsdp2_full",
                    "BENCHMARK_WORKLOAD_PROFILE": "crop1024_800_msa2048_offload",
                    "NPROC_PER_NODE": "8",
                    "NNODES": "1",
                    "DRY_RUN": "1",
                    **override,
                }
            )
            result = subprocess.run(
                ["bash", str(self.repo_dir / "benchmark.sh")],
                cwd=self.repo_dir,
                env=env,
                check=False,
                text=True,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn(expected_message, result.stderr)

    def test_inherited_lower_case_variables_cannot_change_resolved_target(self):
        args = self._dry_run(
            "fsdp2_full",
            {
                "benchmark_exact_crop": "false",
                "model_c_z": "128",
                "model_c_m": "64",
                "pairformer_blocks_per_ckpt": "1",
                "pair_transition_chunk_size": "64",
                "pair_transition_checkpoint_chunks": "false",
            },
        )
        expected = {
            "--c_z": "256",
            "--model.msa_module.c_m": "128",
            "--model.pairformer.blocks_per_ckpt": "2",
            "--model.pairformer.train_pair_transition_chunk_size": "128",
            "--model.pairformer.train_pair_transition_checkpoint_chunks": "true",
        }
        for option, value in expected.items():
            self.assertEqual(self._value(args, option), value)
        self.assertIn(
            "--data.benchmark_long_protein_context_768.base_info.indices_fpath",
            args,
        )

    def test_target_setting_override_is_rejected_before_training(self):
        env = os.environ.copy()
        env.update(
            {
                "BENCHMARK_ID": "invalid_target",
                "BENCHMARK_VARIANT": "fsdp2_full",
                "BENCHMARK_DIFFUSION_BATCH_SIZE": "23",
                "DRY_RUN": "1",
            }
        )
        result = subprocess.run(
            ["bash", str(self.repo_dir / "benchmark.sh")],
            cwd=self.repo_dir,
            env=env,
            check=False,
            text=True,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Matched benchmark requires", result.stderr)

    def test_inherited_launcher_variables_and_check_only_are_overwritten(self):
        args = self._dry_run(
            "ddp_full",
            {
                "base_dir": "/wrong/base",
                "run_name": "wrong-run",
                "nproc_per_node": "8",
                "nnodes": "9",
                "node_rank": "8",
                "master_addr": "wrong-host",
                "master_port": "1",
                "CHECK_ONLY": "1",
            },
        )
        self.assertEqual(self._value(args, "--nproc_per_node"), "1")
        self.assertEqual(self._value(args, "--nnodes"), "1")
        self.assertEqual(self._value(args, "--node_rank"), "0")
        self.assertEqual(self._value(args, "--master_addr"), "127.0.0.1")
        self.assertEqual(self._value(args, "--master_port"), "29690")
        self.assertEqual(self._value(args, "--run_name"), "test_ddp_full")
        self.assertNotEqual(self._value(args, "--base_dir"), "/wrong/base")

    def test_unknown_variant_is_rejected(self):
        env = os.environ.copy()
        env.update(
            {
                "BENCHMARK_ID": "invalid_variant",
                "BENCHMARK_VARIANT": "ddp_postpair",
            }
        )
        result = subprocess.run(
            ["bash", str(self.repo_dir / "benchmark.sh")],
            cwd=self.repo_dir,
            env=env,
            check=False,
            text=True,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unsupported BENCHMARK_VARIANT", result.stderr)

    def test_cu129_label_is_rejected_for_the_locked_cu126_runtime(self):
        env = os.environ.copy()
        env.update(
            {
                "BENCHMARK_ID": "invalid_cuda_runtime",
                "BENCHMARK_VARIANT": "ddp_full",
                "BENCHMARK_EXPECTED_TORCH_CUDA": "12.9",
                "DRY_RUN": "1",
            }
        )
        result = subprocess.run(
            ["bash", str(self.repo_dir / "benchmark.sh")],
            cwd=self.repo_dir,
            env=env,
            check=False,
            text=True,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("PyTorch CUDA 12.6", result.stderr)


class TestBenchmarkWindowHelpers(unittest.TestCase):
    def test_validate_optional_and_enabled_windows(self):
        from runner.train import validate_benchmark_window

        self.assertFalse(validate_benchmark_window(-1, -1, 10))
        self.assertTrue(validate_benchmark_window(2, 3, 5))
        with self.assertRaises(ValueError):
            validate_benchmark_window(-1, 3, 5)
        with self.assertRaises(ValueError):
            validate_benchmark_window(2, 0, 5)
        with self.assertRaises(ValueError):
            validate_benchmark_window(2, 4, 5)

    def test_result_uses_slowest_rank_and_reports_health(self):
        from runner.train import build_benchmark_window_result

        result = build_benchmark_window_result(
            [
                {
                    "rank": 0,
                    "elapsed_seconds": 4.0,
                    "sample_count": 3,
                    "token_count": 30,
                    "atom_count": 300,
                    "max_allocated_bytes": 100,
                    "max_reserved_bytes": 120,
                    "loss_nonfinite_count": 0,
                    "gradient_nonfinite_count": 0,
                    "optimizer_skipped_count": 0,
                },
                {
                    "rank": 1,
                    "elapsed_seconds": 5.0,
                    "sample_count": 3,
                    "token_count": 40,
                    "atom_count": 400,
                    "max_allocated_bytes": 110,
                    "max_reserved_bytes": 130,
                    "loss_nonfinite_count": 1,
                    "gradient_nonfinite_count": 0,
                    "optimizer_skipped_count": 1,
                },
            ],
            warmup_steps=2,
            measure_steps=3,
        )
        self.assertEqual(result["slowest_rank"], 1)
        self.assertEqual(result["elapsed_seconds"], 5.0)
        self.assertAlmostEqual(result["seconds_per_step"], 5.0 / 3.0)
        self.assertEqual(result["total_tokens"], 70)
        self.assertEqual(result["max_allocated_bytes"], 110)
        self.assertEqual(result["max_loss_nonfinite_count_per_rank"], 1)
        self.assertEqual(result["max_optimizer_skipped_count_per_rank"], 1)


if __name__ == "__main__":
    unittest.main()
