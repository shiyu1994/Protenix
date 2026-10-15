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

"""Fail-closed provenance for the fused LayerNorm CUDA extension cache."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.cpp_extension import CUDA_HOME


PROVENANCE_FILENAME = "protenix_cuda_extension_provenance.json"
_EXTENSION_GLOB = "fast_layer_norm_cuda_v2*.so"


@contextmanager
def extension_cache_lock(build_directory: str | os.PathLike[str]):
    """Serialize cache recovery, compilation and import across local ranks."""
    build_path = Path(build_directory)
    build_path.mkdir(parents=True, exist_ok=True)
    # Keep this file in place: unlinking it would let another rank lock a new
    # inode. flock itself is released by the OS if the building process exits.
    with (build_path / ".protenix_layer_norm.lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def recover_incomplete_cache(build_directory: str | os.PathLike[str]) -> bool:
    """Discard unfinished build artifacts under extension_cache_lock.

    A recorded cache still goes through the original strict validation. An
    unrecorded binary is never imported or relabeled as verified: rebuild it.
    """
    build_path = Path(build_directory)
    if not cache_contract_enabled() or (build_path / PROVENANCE_FILENAME).exists():
        return False
    removed = []
    for pattern in (
        _EXTENSION_GLOB,
        "layer_norm_cuda*.o",
        "layer_norm_cuda*.o.d",
        "build.ninja",
        ".ninja_deps",
        ".ninja_log",
        "lock",
        f".{PROVENANCE_FILENAME}.*.tmp",
    ):
        for path in build_path.glob(pattern):
            path.unlink()
            removed.append(path.name)
    if removed:
        logging.warning(
            "Rebuilding incomplete LayerNorm cache at %s: removed %s",
            build_path,
            ", ".join(sorted(removed)),
        )
    return bool(removed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is required by the pinned CUDA extension cache contract"
        )
    return value


def cache_contract_enabled() -> bool:
    return bool(os.environ.get("PROTENIX_LAYER_NORM_CACHE_FINGERPRINT", "").strip())


def _expected_record() -> dict[str, Any]:
    expected_cuda_home = Path(_required_env("PROTENIX_CUDA126_HOME")).resolve()
    detected_cuda_home = Path(CUDA_HOME or "").resolve()
    if detected_cuda_home != expected_cuda_home:
        raise RuntimeError(
            "torch.utils.cpp_extension.CUDA_HOME does not match the pinned "
            f"CUDA 12.6 toolkit: detected={detected_cuda_home}, "
            f"expected={expected_cuda_home}"
        )
    return {
        "schema_version": int(
            _required_env("PROTENIX_LAYER_NORM_PROVENANCE_SCHEMA")
        ),
        "cache_fingerprint": _required_env(
            "PROTENIX_LAYER_NORM_CACHE_FINGERPRINT"
        ),
        "toolchain_fingerprint": _required_env(
            "PROTENIX_CUDA_TOOLCHAIN_FINGERPRINT"
        ),
        "source_sha256": _required_env("PROTENIX_LAYER_NORM_SOURCE_SHA256"),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_home": str(expected_cuda_home),
        "nvcc": str((expected_cuda_home / "bin" / "nvcc").resolve()),
        "cc": str(Path(_required_env("PROTENIX_CUDA_HOST_CC")).resolve()),
        "cxx": str(Path(_required_env("PROTENIX_CUDA_HOST_CXX")).resolve()),
        "nvcc_sha256": _required_env("PROTENIX_CUDA_NVCC_SHA256"),
        "cc_sha256": _required_env("PROTENIX_CUDA_HOST_CC_SHA256"),
        "cxx_sha256": _required_env("PROTENIX_CUDA_HOST_CXX_SHA256"),
        "cudart_sha256": _required_env("PROTENIX_CUDART_SHA256"),
        "cuda_arch_list": _required_env("PROTENIX_CUDA_ARCH_LIST"),
    }


def _extension_candidates(build_directory: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in build_directory.glob(_EXTENSION_GLOB)
        if path.is_file()
    )


def _validate_build_ninja(
    build_directory: Path,
    record: dict[str, Any],
) -> None:
    build_ninja = build_directory / "build.ninja"
    if not build_ninja.is_file():
        raise RuntimeError(
            f"verified LayerNorm cache is missing {build_ninja}"
        )
    if record.get("build_ninja_sha256") != _sha256(build_ninja):
        raise RuntimeError("LayerNorm build.ninja SHA256 does not match provenance")
    build_text = build_ninja.read_text(encoding="utf-8")
    required_fragments = (
        f"nvcc = {record['nvcc']}",
        f"cxx = {record['cxx']}",
        f"-ccbin {record['cc']}",
    )
    missing = [fragment for fragment in required_fragments if fragment not in build_text]
    if missing:
        raise RuntimeError(
            "LayerNorm build.ninja was generated by a different CUDA toolchain: "
            + ", ".join(missing)
        )


def validate_existing_cache(
    build_directory: str | os.PathLike[str],
) -> Optional[dict[str, Any]]:
    """Validate an existing binary before layer_norm.py imports it directly."""
    if not cache_contract_enabled():
        return None
    build_path = Path(build_directory).resolve()
    provenance_path = build_path / PROVENANCE_FILENAME
    candidates = _extension_candidates(build_path) if build_path.is_dir() else []
    if not candidates and not provenance_path.exists():
        return None
    if len(candidates) != 1:
        raise RuntimeError(
            "LayerNorm cache must contain exactly one extension binary; "
            f"found {len(candidates)} under {build_path}"
        )
    if not provenance_path.is_file():
        raise RuntimeError(
            "refusing to import an unverified cached LayerNorm extension: "
            f"missing {provenance_path}"
        )
    try:
        record = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"invalid LayerNorm cache provenance: {provenance_path}"
        ) from exc

    expected = _expected_record()
    mismatches = {
        key: (record.get(key), expected_value)
        for key, expected_value in expected.items()
        if record.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(
            "cached LayerNorm extension provenance does not match the active "
            f"CUDA 12.6 contract: {mismatches}"
        )

    extension_path = candidates[0]
    if record.get("extension_filename") != extension_path.name:
        raise RuntimeError("LayerNorm extension filename does not match provenance")
    if record.get("extension_sha256") != _sha256(extension_path):
        raise RuntimeError("LayerNorm extension SHA256 does not match provenance")
    _validate_build_ninja(build_path, record)
    return record


def write_cache_provenance(
    build_directory: str | os.PathLike[str],
    extension_module: Any,
) -> Optional[Path]:
    """Atomically mark one extension produced by the active pinned toolchain."""
    if not cache_contract_enabled():
        return None
    build_path = Path(build_directory).resolve()
    extension_path = Path(extension_module.__file__).resolve()
    try:
        extension_path.relative_to(build_path)
    except ValueError as exc:
        raise RuntimeError(
            f"compiled LayerNorm extension escaped its build directory: "
            f"{extension_path} is not under {build_path}"
        ) from exc

    build_ninja = build_path / "build.ninja"
    if not build_ninja.is_file():
        raise RuntimeError(f"LayerNorm build did not produce {build_ninja}")
    record = {
        **_expected_record(),
        "extension_filename": extension_path.name,
        "extension_sha256": _sha256(extension_path),
        "build_ninja_sha256": _sha256(build_ninja),
    }
    _validate_build_ninja(build_path, record)

    provenance_path = build_path / PROVENANCE_FILENAME
    temporary_path = build_path / f".{PROVENANCE_FILENAME}.{os.getpid()}.tmp"
    temporary_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, provenance_path)
    return provenance_path


def validate_loaded_extension(
    build_directory: str | os.PathLike[str],
    extension_module: Any,
) -> None:
    if not cache_contract_enabled():
        return
    build_path = Path(build_directory).resolve()
    extension_path = Path(extension_module.__file__).resolve()
    try:
        extension_path.relative_to(build_path)
    except ValueError as exc:
        raise RuntimeError(
            "fast_layer_norm_cuda_v2 loaded outside the fingerprinted cache: "
            f"{extension_path} is not under {build_path}"
        ) from exc
    validate_existing_cache(build_path)
