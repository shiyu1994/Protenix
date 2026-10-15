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

import fcntl
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_DIR = Path(__file__).resolve().parents[1]
CUDA_HELPER = REPO_DIR / "scripts/runtime/cuda126_env.sh"
CUDA126_HOME = Path("/mnt/project_abd/klp/runtime/protenix_cuda126_toolkit")


def _load_provenance_module():
    path = REPO_DIR / "protenix/model/layer_norm/cache_provenance.py"
    spec = importlib.util.spec_from_file_location(
        "protenix_layer_norm_cache_provenance_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestCuda126RuntimeContract(unittest.TestCase):
    @unittest.skipUnless(
        (CUDA126_HOME / "bin/nvcc").is_file(),
        "the pinned shared CUDA 12.6 toolkit is unavailable",
    )
    def test_helper_overrides_polluted_system_cuda_and_exports_fingerprint(self):
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_HOME": "/usr/local/cuda",
                "CUDACXX": "/usr/local/cuda/bin/nvcc",
                "CC": "/usr/bin/gcc",
                "CXX": "/usr/bin/g++",
                "CONDA_ENV": "protenix",
                "CONDA_ACTIVATE": "/mnt/project_abd/klp/miniconda3/bin/activate",
            }
        )
        command = f"""
set -euo pipefail
source {CUDA_HELPER}
python - <<'PY'
import json
import os
import shutil
print(json.dumps({{
    "cuda_home": os.environ["CUDA_HOME"],
    "nvcc": shutil.which("nvcc"),
    "cc": os.environ["CC"],
    "cxx": os.environ["CXX"],
    "torch": os.environ["PROTENIX_EXPECTED_TORCH_VERSION"],
    "torch_cuda": os.environ["EXPECTED_TORCH_CUDA"],
    "arch": os.environ["PROTENIX_CUDA_ARCH_LIST"],
    "fingerprint": os.environ["PROTENIX_LAYER_NORM_CACHE_FINGERPRINT"],
}}))
PY
"""
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=REPO_DIR,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["cuda_home"], str(CUDA126_HOME))
        self.assertEqual(payload["nvcc"], str(CUDA126_HOME / "bin/nvcc"))
        self.assertEqual(
            payload["cc"],
            str(CUDA126_HOME / "bin/x86_64-conda-linux-gnu-gcc"),
        )
        self.assertEqual(
            payload["cxx"],
            str(CUDA126_HOME / "bin/x86_64-conda-linux-gnu-g++"),
        )
        self.assertEqual(payload["torch"], "2.7.1+cu126")
        self.assertEqual(payload["torch_cuda"], "12.6")
        self.assertEqual(payload["arch"], "9.0")
        self.assertEqual(len(payload["fingerprint"]), 64)

    def test_all_locked_entrypoints_source_the_cuda_contract(self):
        for relative_path in (
            "run_crop384_db48_ddp_scratch_200k.sh",
            "resume_crop384_db48_ddp_scratch_200k.sh",
            "run_crop384_db48_ddp_speed_trial.sh",
            "finetune_demo.sh",
            "scripts/benchmark/run_aistudio_benchmark.sh",
        ):
            contents = (REPO_DIR / relative_path).read_text(encoding="utf-8")
            self.assertIn("scripts/runtime/cuda126_env.sh", contents)

        production = (
            REPO_DIR / "run_crop384_db48_ddp_scratch_200k.sh"
        ).read_text(encoding="utf-8")
        speed = (
            REPO_DIR / "run_crop384_db48_ddp_speed_trial.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("/usr/bin/gcc-", production)
        self.assertNotIn("/usr/bin/gcc-", speed)
        self.assertIn("PROTENIX_LAYER_NORM_CACHE_FINGERPRINT", production)
        self.assertIn("PROTENIX_LAYER_NORM_CACHE_FINGERPRINT", speed)

    def test_extension_provenance_accepts_exact_cache_and_rejects_tampering(
        self,
    ):
        module = _load_provenance_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            cuda_home, build_directory, extension = _make_fake_extension_cache(
                tmp_path
            )
            environment = _provenance_environment(cuda_home)
            extension_module = SimpleNamespace(__file__=str(extension))

            with mock.patch.dict(
                os.environ, environment, clear=False
            ), mock.patch.object(module, "CUDA_HOME", str(cuda_home)):
                provenance_path = module.write_cache_provenance(
                    build_directory, extension_module
                )
                self.assertIsNotNone(provenance_path)
                self.assertIsNotNone(
                    module.validate_existing_cache(build_directory)
                )

                extension.write_bytes(b"tampered")
                with self.assertRaisesRegex(RuntimeError, "extension SHA256"):
                    module.validate_existing_cache(build_directory)

    def test_extension_cache_without_provenance_is_rejected(self):
        module = _load_provenance_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            cuda_home, build_directory, _ = _make_fake_extension_cache(
                tmp_path
            )
            with mock.patch.dict(
                os.environ,
                _provenance_environment(cuda_home),
                clear=False,
            ), mock.patch.object(module, "CUDA_HOME", str(cuda_home)):
                with self.assertRaisesRegex(RuntimeError, "unverified cached"):
                    module.validate_existing_cache(build_directory)

    def test_unrecorded_cache_is_removed_before_recompilation(self):
        module = _load_provenance_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            cuda_home, build_directory, extension = _make_fake_extension_cache(
                Path(temporary_directory)
            )
            build_ninja_text = (build_directory / "build.ninja").read_text()
            residuals = (
                "layer_norm_cuda.o", "layer_norm_cuda_kernel.cuda.o",
                "layer_norm_cuda_kernel.cuda.o.d", "lock", ".ninja_log",
                ".ninja_deps", f".{module.PROVENANCE_FILENAME}.123.tmp",
            )
            for name in residuals:
                (build_directory / name).write_bytes(b"interrupted-build")
            unrelated = build_directory / "unrelated.txt"
            unrelated.write_bytes(b"preserve")
            with mock.patch.dict(
                os.environ, _provenance_environment(cuda_home), clear=False
            ), mock.patch.object(module, "CUDA_HOME", str(cuda_home)):
                with module.extension_cache_lock(build_directory):
                    self.assertTrue(module.recover_incomplete_cache(build_directory))
                    self.assertIsNone(module.validate_existing_cache(build_directory))
                    self.assertFalse(extension.exists())
                    self.assertFalse((build_directory / "build.ninja").exists())
                    for name in residuals:
                        self.assertFalse((build_directory / name).exists())
                    self.assertEqual(unrelated.read_bytes(), b"preserve")
                    self.assertFalse(module.recover_incomplete_cache(build_directory))

                    # Only a newly compiled binary may receive a new record.
                    extension.write_bytes(b"rebuilt-extension")
                    (build_directory / "build.ninja").write_text(build_ninja_text)
                    module.write_cache_provenance(
                        build_directory, SimpleNamespace(__file__=str(extension))
                    )
                    self.assertFalse(module.recover_incomplete_cache(build_directory))
                    self.assertIsNotNone(module.validate_existing_cache(build_directory))

    def test_interrupted_build_without_binary_removes_stale_torch_lock(self):
        module = _load_provenance_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            cuda_home, build_directory, extension = _make_fake_extension_cache(
                Path(temporary_directory)
            )
            extension.unlink()
            torch_lock = build_directory / "lock"
            torch_lock.touch()
            with mock.patch.dict(
                os.environ, _provenance_environment(cuda_home), clear=False
            ), module.extension_cache_lock(build_directory):
                self.assertTrue(module.recover_incomplete_cache(build_directory))
                self.assertFalse(torch_lock.exists())
                self.assertIsNone(module.validate_existing_cache(build_directory))

    def test_recovery_does_not_bypass_recorded_cache_validation(self):
        module = _load_provenance_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            cuda_home, build_directory, extension = _make_fake_extension_cache(
                Path(temporary_directory)
            )
            with mock.patch.dict(
                os.environ, _provenance_environment(cuda_home), clear=False
            ), mock.patch.object(module, "CUDA_HOME", str(cuda_home)):
                module.write_cache_provenance(
                    build_directory, SimpleNamespace(__file__=str(extension))
                )
                extension.write_bytes(b"tampered")
                with module.extension_cache_lock(build_directory):
                    self.assertFalse(module.recover_incomplete_cache(build_directory))
                    with self.assertRaisesRegex(RuntimeError, "extension SHA256"):
                        module.validate_existing_cache(build_directory)
                    (build_directory / module.PROVENANCE_FILENAME).write_text("{")
                    self.assertFalse(module.recover_incomplete_cache(build_directory))
                    with self.assertRaisesRegex(RuntimeError, "invalid LayerNorm cache"):
                        module.validate_existing_cache(build_directory)

    def test_cache_lock_serializes_readers_and_releases_after_failure(self):
        module = _load_provenance_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            build_directory = Path(temporary_directory) / "cache"
            lock_path = build_directory / ".protenix_layer_norm.lock"
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with module.extension_cache_lock(build_directory):
                    inode = lock_path.stat().st_ino
                    with lock_path.open("a") as competitor:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(competitor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    raise RuntimeError("interrupted")
            with lock_path.open("a") as competitor:
                fcntl.flock(competitor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(lock_path.stat().st_ino, inode)
                fcntl.flock(competitor.fileno(), fcntl.LOCK_UN)

    def test_recovery_leaves_unmanaged_caches_unchanged(self):
        module = _load_provenance_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            _, build_directory, extension = _make_fake_extension_cache(
                Path(temporary_directory)
            )
            with mock.patch.dict(
                os.environ, {"PROTENIX_LAYER_NORM_CACHE_FINGERPRINT": ""}
            ), module.extension_cache_lock(build_directory):
                self.assertFalse(module.recover_incomplete_cache(build_directory))
                self.assertEqual(extension.read_bytes(), b"correct-cu126-extension")


def _provenance_environment(cuda_home: Path) -> dict[str, str]:
    return {
        "PROTENIX_LAYER_NORM_CACHE_FINGERPRINT": "a" * 64,
        "PROTENIX_CUDA_TOOLCHAIN_FINGERPRINT": "b" * 64,
        "PROTENIX_LAYER_NORM_SOURCE_SHA256": "c" * 64,
        "PROTENIX_LAYER_NORM_PROVENANCE_SCHEMA": "1",
        "PROTENIX_CUDA126_HOME": str(cuda_home),
        "PROTENIX_CUDA_HOST_CC": str(cuda_home / "bin/gcc"),
        "PROTENIX_CUDA_HOST_CXX": str(cuda_home / "bin/g++"),
        "PROTENIX_CUDA_NVCC_SHA256": "d" * 64,
        "PROTENIX_CUDA_HOST_CC_SHA256": "e" * 64,
        "PROTENIX_CUDA_HOST_CXX_SHA256": "f" * 64,
        "PROTENIX_CUDART_SHA256": "1" * 64,
        "PROTENIX_CUDA_ARCH_LIST": "9.0",
    }


def _make_fake_extension_cache(tmp_path: Path):
    cuda_home = tmp_path / "cuda"
    build_directory = tmp_path / "extensions" / "fast_layer_norm_cuda_v2"
    build_directory.mkdir(parents=True)
    extension = build_directory / "fast_layer_norm_cuda_v2.so"
    extension.write_bytes(b"correct-cu126-extension")
    cc = cuda_home / "bin/gcc"
    cxx = cuda_home / "bin/g++"
    nvcc = cuda_home / "bin/nvcc"
    (cuda_home / "bin").mkdir(parents=True)
    build_ninja = build_directory / "build.ninja"
    build_ninja.write_text(
        f"cxx = {cxx.resolve()}\n"
        f"nvcc = {nvcc.resolve()}\n"
        f"cuda_cflags = -ccbin {cc.resolve()} -gencode=arch=compute_90,code=sm_90\n",
        encoding="utf-8",
    )
    return cuda_home, build_directory, extension
