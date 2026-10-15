import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
FINETUNE_DEMO = REPO_DIR / "finetune_demo.sh"
SHARING_ENV = "PROTENIX_MP_SHARING_STRATEGY"


def _run_data_pipeline_import(strategy):
    env = os.environ.copy()
    if strategy is None:
        env.pop(SHARING_ENV, None)
        expected = "file_system"
    else:
        env[SHARING_ENV] = strategy
        expected = strategy
    env["PYTHONPATH"] = (
        str(REPO_DIR)
        + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "from protenix.data.pipeline.data_pipeline import "
                "MP_SHARING_STRATEGY; "
                "print('configured=' + MP_SHARING_STRATEGY); "
                "print('actual=' + torch.multiprocessing.get_sharing_strategy())"
            ),
        ],
        cwd=REPO_DIR,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result, expected


def _fake_conda_environment(tmp_path):
    conda_root = tmp_path / "conda"
    activate = conda_root / "bin" / "activate"
    env_bin = conda_root / "envs" / "protenix" / "bin"
    purelib = tmp_path / "purelib"
    torch_lib = purelib / "torch" / "lib"
    activate.parent.mkdir(parents=True)
    env_bin.mkdir(parents=True)
    torch_lib.mkdir(parents=True)
    activate.write_text("# test-only no-op activation\n")

    fake_python = env_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' '{purelib}'\n"
    )
    fake_python.chmod(0o755)
    return activate, purelib


def _run_dry_shell_preflight(tmp_path, strategy):
    activate, purelib = _fake_conda_environment(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "CONDA_ACTIVATE": str(activate),
            "CONDA_ENV": "protenix",
            "DRY_RUN": "1",
            "SKIP_PATH_CHECK": "1",
            "LOAD_CHECKPOINT_PATH": "",
            "LOAD_EMA_CHECKPOINT_PATH": "",
            SHARING_ENV: strategy,
        }
    )
    result = subprocess.run(
        ["bash", str(FINETUNE_DEMO)],
        cwd=REPO_DIR,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result, purelib


class TestMultiprocessingSharingStrategy(unittest.TestCase):
    def test_python_entry_configures_and_reports_requested_strategy(self):
        for strategy in (None, "file_descriptor", "file_system"):
            with self.subTest(strategy=strategy):
                result, expected = _run_data_pipeline_import(strategy)

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"configured={expected}", result.stdout)
                self.assertIn(f"actual={expected}", result.stdout)
                self.assertIn(
                    f"{SHARING_ENV}={expected}, actual={expected}",
                    result.stdout + result.stderr,
                )

    def test_python_entry_rejects_invalid_strategy_before_pipeline_import(self):
        result, _ = _run_data_pipeline_import("invalid")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(SHARING_ENV, result.stderr)
        self.assertIn("file_descriptor", result.stderr)
        self.assertIn("file_system", result.stderr)

    def test_file_descriptor_shell_path_does_not_require_torch_shm_manager(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result, purelib = _run_dry_shell_preflight(
                Path(temporary_directory), "file_descriptor"
            )

        self.assertFalse(
            (purelib / "torch" / "bin" / "torch_shm_manager").exists()
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.startswith("torchrun "))

    def test_file_system_shell_path_requires_torch_shm_manager(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result, purelib = _run_dry_shell_preflight(
                Path(temporary_directory), "file_system"
            )
            expected_manager = purelib / "torch" / "bin" / "torch_shm_manager"

            self.assertEqual(result.returncode, 1)
            self.assertIn(str(expected_manager), result.stderr)
            self.assertIn("required by file_system", result.stderr)

    def test_shell_rejects_invalid_strategy_before_conda_checks(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            env = os.environ.copy()
            env.update(
                {
                    "CONDA_ACTIVATE": str(
                        Path(temporary_directory) / "missing-activate"
                    ),
                    "DRY_RUN": "1",
                    SHARING_ENV: "invalid",
                }
            )
            result = subprocess.run(
                ["bash", str(FINETUNE_DEMO)],
                cwd=REPO_DIR,
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn(SHARING_ENV, result.stderr)
        self.assertNotIn("Missing conda activate script", result.stderr)

    def test_preflight_records_fd_limits_and_gates_manager_by_strategy(self):
        script = FINETUNE_DEMO.read_text()

        self.assertIn("resource.getrlimit(resource.RLIMIT_NOFILE)", script)
        self.assertIn('runtime["fd_limit_soft"]', script)
        self.assertIn('runtime["fd_limit_hard"]', script)
        self.assertIn('if sharing_strategy == "file_descriptor":', script)
        self.assertIn(
            'runtime["torch_shm_manager"] = "not_required"',
            script,
        )
        self.assertIn(
            'if [ "${mp_sharing_strategy}" = "file_system" ] '
            '&& [ ! -x "${torch_shm_manager}" ]; then',
            script,
        )


if __name__ == "__main__":
    unittest.main()
