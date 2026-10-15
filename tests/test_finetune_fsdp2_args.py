import os
import shlex
import subprocess
import unittest
from pathlib import Path


class TestFinetuneFSDP2Arguments(unittest.TestCase):
    _OPTION_ENV_NAMES = (
        "DISTRIBUTED_STRATEGY",
        "OPTIMIZER_SHARDING",
        "EMA_SHADOW_DEVICE",
        "FSDP_MESH",
        "FSDP_PARAM_DTYPE",
        "FSDP_REDUCE_DTYPE",
        "FSDP_RESHARD_AFTER_FORWARD",
        "SKIP_FINAL_CHECKPOINT",
        "LOAD_PARAMS_ONLY",
        "BENCHMARK_WARMUP_STEPS",
        "BENCHMARK_MEASURE_STEPS",
        "MSA_CHUNK_SIZE",
        "PAIRFORMER_CHECKPOINT_CPU_OFFLOAD_EVERY_N",
        "MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N",
        "distributed_strategy",
        "optimizer_sharding",
        "ema_shadow_device",
        "fsdp_mesh",
        "fsdp_param_dtype",
        "fsdp_reduce_dtype",
        "fsdp_reshard_after_forward",
        "skip_final_checkpoint",
        "load_params_only",
        "benchmark_warmup_steps",
        "benchmark_measure_steps",
        "msa_chunk_size",
        "pairformer_checkpoint_cpu_offload_every_n",
        "msa_checkpoint_cpu_offload_every_n",
    )

    def _dry_run(self, overrides=None):
        repo_dir = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        for name in self._OPTION_ENV_NAMES:
            env.pop(name, None)
        env.update(
            {
                "DRY_RUN": "1",
                "SKIP_PATH_CHECK": "1",
                "NPROC_PER_NODE": "1",
                "LOAD_CHECKPOINT_PATH": "",
                "LOAD_EMA_CHECKPOINT_PATH": "",
            }
        )
        env.update(overrides or {})
        result = subprocess.run(
            ["bash", str(repo_dir / "finetune_demo.sh")],
            cwd=repo_dir,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        return shlex.split(result.stdout)

    @staticmethod
    def _option_value(args, option):
        return args[args.index(option) + 1]

    def test_dry_run_uses_safe_ddp_defaults(self):
        args = self._dry_run()

        expected = {
            "--distributed_strategy": "ddp",
            "--optimizer_sharding": "none",
            "--ema_shadow_device": "model",
            "--fsdp.mesh": "auto",
            "--fsdp.param_dtype": "fp32",
            "--fsdp.reduce_dtype": "fp32",
            "--fsdp.reshard_after_forward": "true",
            "--skip_final_checkpoint": "false",
            "--load_params_only": "true",
            "--benchmark_warmup_steps": "-1",
            "--benchmark_measure_steps": "-1",
        }
        for option, value in expected.items():
            self.assertEqual(self._option_value(args, option), value)

    def test_dry_run_forwards_fsdp2_environment_overrides(self):
        args = self._dry_run(
            {
                "DISTRIBUTED_STRATEGY": "fsdp2",
                "OPTIMIZER_SHARDING": "none",
                "EMA_SHADOW_DEVICE": "cpu",
                "FSDP_MESH": "global",
                "FSDP_PARAM_DTYPE": "bf16",
                "FSDP_REDUCE_DTYPE": "fp32",
                "FSDP_RESHARD_AFTER_FORWARD": "false",
                "SKIP_FINAL_CHECKPOINT": "true",
                "LOAD_PARAMS_ONLY": "false",
                "BENCHMARK_WARMUP_STEPS": "5",
                "BENCHMARK_MEASURE_STEPS": "12",
            }
        )

        expected = {
            "--distributed_strategy": "fsdp2",
            "--optimizer_sharding": "none",
            "--ema_shadow_device": "cpu",
            "--fsdp.mesh": "global",
            "--fsdp.param_dtype": "bf16",
            "--fsdp.reduce_dtype": "fp32",
            "--fsdp.reshard_after_forward": "false",
            "--skip_final_checkpoint": "true",
            "--load_params_only": "false",
            "--benchmark_warmup_steps": "5",
            "--benchmark_measure_steps": "12",
        }
        for option, value in expected.items():
            self.assertEqual(self._option_value(args, option), value)

    def test_dry_run_forwards_msa_chunk_and_activation_offload_settings(self):
        args = self._dry_run(
            {
                "MSA_CHUNK_SIZE": "128",
                "PAIRFORMER_CHECKPOINT_CPU_OFFLOAD_EVERY_N": "2",
                "MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N": "1",
            }
        )

        expected = {
            "--model.msa_module.msa_chunk_size": "128",
            "--model.pairformer.checkpoint_cpu_offload_every_n": "2",
            "--model.msa_module.checkpoint_cpu_offload_every_n": "1",
        }
        for option, value in expected.items():
            self.assertEqual(self._option_value(args, option), value)

    def test_invalid_activation_offload_interval_fails_before_torchrun(self):
        repo_dir = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "SKIP_PATH_CHECK": "1",
                "NPROC_PER_NODE": "1",
                "LOAD_CHECKPOINT_PATH": "",
                "LOAD_EMA_CHECKPOINT_PATH": "",
                "MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N": "0",
            }
        )
        result = subprocess.run(
            ["bash", str(repo_dir / "finetune_demo.sh")],
            cwd=repo_dir,
            env=env,
            check=False,
            text=True,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("positive integer", result.stderr)


if __name__ == "__main__":
    unittest.main()
