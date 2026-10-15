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

import os
import shlex
import subprocess
import unittest
from pathlib import Path


class TestFullResumeLauncher(unittest.TestCase):
    repo_dir = Path(__file__).resolve().parents[1]
    launcher = repo_dir / "resume_immune_heavy_localattention_fsdp2.sh"
    source_run = Path(
        "/mnt/project_abd/klp/proteinx_output/"
        "immune_heavy_old_content_crop1024_nuc128_db40_pairedmsa_"
        "localattention_fsdp2_20260722_032221"
    )

    def _run(self, extra_env=None, *, check=True):
        env = os.environ.copy()
        env.pop("PROTENIX_CUDA_HOST_CC", None)
        env.pop("PROTENIX_CUDA_HOST_CXX", None)
        env.update(
            {
                "DRY_RUN": "1",
                "SKIP_PATH_CHECK": "1",
                "RESUME_NNODES": "1",
                "RUN_NAME": "resume_launcher_test",
                "BASE_DIR": "/tmp/protenix_resume_launcher_test",
            }
        )
        env.update(extra_env or {})
        return subprocess.run(
            ["bash", str(self.launcher)],
            cwd=self.repo_dir,
            env=env,
            check=check,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    @staticmethod
    def _command(result):
        line = next(
            item for item in result.stdout.splitlines() if item.startswith("torchrun ")
        )
        return shlex.split(line)

    @staticmethod
    def _arg(command, name):
        return command[command.index(name) + 1]

    def test_formal_resume_restores_full_state_and_total_step_target(self):
        result = self._run()
        command = self._command(result)

        self.assertIn("resume=full checkpoint_step=499 next_step=500", result.stdout)
        self.assertEqual(
            self._arg(command, "--load_checkpoint_path"),
            str(self.source_run / "checkpoints/499.pt"),
        )
        self.assertEqual(
            self._arg(command, "--load_ema_checkpoint_path"),
            str(self.source_run / "checkpoints/499_ema_0.999.pt"),
        )
        self.assertEqual(self._arg(command, "--load_params_only"), "false")
        self.assertEqual(self._arg(command, "--skip_load_optimizer"), "false")
        self.assertEqual(self._arg(command, "--skip_load_scheduler"), "false")
        self.assertEqual(self._arg(command, "--skip_load_step"), "false")
        self.assertEqual(self._arg(command, "--load_step_for_scheduler"), "true")
        self.assertEqual(self._arg(command, "--max_steps"), "2000")
        self.assertEqual(self._arg(command, "--data.num_dl_workers"), "8")
        self.assertEqual(self._arg(command, "--skip_final_checkpoint"), "false")
        self.assertEqual(self._arg(command, "--checkpoint_interval"), "250")
        self.assertEqual(self._arg(command, "--nnodes"), "1")
        self.assertEqual(self._arg(command, "--nproc_per_node"), "8")

    def test_one_update_smoke_ends_at_501_without_checkpoint_write(self):
        result = self._run({"RESUME_SMOKE_STEPS": "1"})
        command = self._command(result)

        self.assertEqual(self._arg(command, "--max_steps"), "501")
        self.assertEqual(self._arg(command, "--skip_final_checkpoint"), "true")
        self.assertEqual(self._arg(command, "--checkpoint_interval"), "-1")

    def test_custom_checkpoint_pair_is_supported(self):
        result = self._run(
            {
                "RESUME_STEP": "749",
                "RESUME_RUN_DIR": "/tmp/source_run",
                "RESUME_TOTAL_MAX_STEPS": "900",
            }
        )
        command = self._command(result)

        self.assertEqual(
            self._arg(command, "--load_checkpoint_path"),
            "/tmp/source_run/checkpoints/749.pt",
        )
        self.assertEqual(
            self._arg(command, "--load_ema_checkpoint_path"),
            "/tmp/source_run/checkpoints/749_ema_0.999.pt",
        )
        self.assertEqual(self._arg(command, "--max_steps"), "900")

    def test_raw_checkpoint_cannot_be_an_ema_file(self):
        result = self._run(
            {"RESUME_RAW_CHECKPOINT": "/tmp/499_ema_0.999.pt"},
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("must not point to an EMA checkpoint", result.stderr)

    def test_total_target_must_leave_at_least_one_update(self):
        result = self._run(
            {"RESUME_TOTAL_MAX_STEPS": "500"},
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("must be greater than checkpoint_step+1", result.stderr)

    def test_invalid_sha_verification_mode_is_rejected_in_dry_run(self):
        result = self._run(
            {"RESUME_VERIFY_SHA256": "sometimes"},
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("PROTENIX_VERIFY_RESUME_SHA256 must be 0 or 1", result.stderr)

    def test_stale_generic_topology_variables_cannot_override_resume_topology(self):
        result = self._run(
            {
                "NNODES": "9",
                "NPROC_PER_NODE": "3",
                "nnodes": "7",
                "nproc_per_node": "2",
                "node_rank": "6",
                "master_addr": "stale.invalid",
                "PROTENIX_MP_SHARING_STRATEGY": "file_system",
            }
        )
        command = self._command(result)

        self.assertEqual(self._arg(command, "--nnodes"), "1")
        self.assertEqual(self._arg(command, "--nproc_per_node"), "8")
        self.assertEqual(self._arg(command, "--node_rank"), "0")
        self.assertEqual(self._arg(command, "--master_addr"), "127.0.0.1")
        self.assertIn(
            "mp_sharing_strategy=file_descriptor",
            result.stdout,
        )

    def test_formal_four_node_topology_requires_current_aistudio_coordinates(self):
        result = self._run(
            {
                "RESUME_NNODES": "4",
                "MLP_WORKER_NUM": "4",
                "MLP_ROLE_INDEX": "2",
                "MLP_WORKER_0_HOST": "trainer-worker-0",
                "MLP_WORKER_0_PORT": "23456",
            }
        )
        command = self._command(result)

        self.assertEqual(self._arg(command, "--nnodes"), "4")
        self.assertEqual(self._arg(command, "--nproc_per_node"), "8")
        self.assertEqual(self._arg(command, "--node_rank"), "2")
        self.assertEqual(self._arg(command, "--master_addr"), "trainer-worker-0")
        self.assertEqual(self._arg(command, "--master_port"), "23456")

    def test_multinode_rejects_only_stale_generic_coordinates(self):
        result = self._run(
            {
                "RESUME_NNODES": "4",
                "NODE_RANK": "2",
                "RANK": "2",
                "MASTER_ADDR": "stale-host",
                "MASTER_PORT": "29999",
            },
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "requires a valid AIStudio MLP_ROLE_INDEX or RESUME_NODE_RANK",
            result.stderr,
        )

    def test_single_node_smoke_can_use_an_isolated_gpu_subset(self):
        result = self._run(
            {
                "RESUME_SMOKE_STEPS": "1",
                "RESUME_NPROC_PER_NODE": "5",
            }
        )
        command = self._command(result)

        self.assertEqual(self._arg(command, "--nnodes"), "1")
        self.assertEqual(self._arg(command, "--nproc_per_node"), "5")
        self.assertEqual(self._arg(command, "--max_steps"), "501")

    def test_cuda_host_compiler_pair_is_auto_detected(self):
        result = self._run()

        compiler_line = next(
            line
            for line in result.stdout.splitlines()
            if "cuda_host_compiler=" in line
        )
        compiler_pair = compiler_line.split("=", maxsplit=1)[1].strip().split(" / ")
        self.assertEqual(len(compiler_pair), 2)
        self.assertTrue(Path(compiler_pair[0]).is_file())
        self.assertTrue(os.access(compiler_pair[0], os.X_OK))
        self.assertTrue(Path(compiler_pair[1]).is_file())
        self.assertTrue(os.access(compiler_pair[1], os.X_OK))

    def test_missing_explicit_cuda_host_compiler_is_rejected(self):
        result = self._run(
            {
                "PROTENIX_CUDA_HOST_CC": "/definitely/missing/gcc",
                "PROTENIX_CUDA_HOST_CXX": "/definitely/missing/g++",
            },
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("No executable CUDA host C compiler found", result.stderr)

    def test_formal_resume_cannot_use_a_reduced_gpu_subset(self):
        result = self._run(
            {"RESUME_NPROC_PER_NODE": "5"},
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "Formal resume requires RESUME_NPROC_PER_NODE=8",
            result.stderr,
        )


if __name__ == "__main__":
    unittest.main()
