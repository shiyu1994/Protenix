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
import re
import shutil
import subprocess
from typing import Any, Optional

import torch
from torch.utils.cpp_extension import load


def _resolve_cuda_arch_list() -> str:
    """Compile only for visible GPU architectures unless explicitly overridden."""
    from torch.utils.cpp_extension import CUDA_HOME

    nvcc = shutil.which("nvcc")
    if CUDA_HOME:
        candidate = os.path.join(CUDA_HOME, "bin", "nvcc")
        if os.path.isfile(candidate):
            nvcc = candidate
    supported: set[str] = set()
    if nvcc:
        try:
            output = subprocess.check_output(
                [nvcc, "--list-gpu-arch"],
                text=True,
                stderr=subprocess.STDOUT,
            )
            supported = set(re.findall(r"compute_(\d+)", output))
        except (OSError, subprocess.CalledProcessError):
            supported = set()

    requested = os.environ.get("PROTENIX_CUDA_ARCH_LIST", "").strip()
    if requested:
        candidates = [
            value.strip().removesuffix("+PTX")
            for value in re.split(r"[;, ]+", requested)
            if value.strip()
        ]
    elif torch.cuda.is_available():
        candidates = sorted(
            {
                f"{major}.{minor}"
                for major, minor in (
                    torch.cuda.get_device_capability(index)
                    for index in range(torch.cuda.device_count())
                )
            }
        )
    else:
        raise RuntimeError("CUDA must be available to compile FusedLayerNorm")

    resolved = []
    for candidate in candidates:
        match = re.fullmatch(r"(\d+)\.(\d+)", candidate)
        if match is None:
            raise ValueError(
                f"Invalid CUDA architecture {candidate!r}; expected forms like 9.0"
            )
        compact = "".join(match.groups())
        if supported and compact not in supported:
            raise RuntimeError(
                f"nvcc does not support requested compute_{compact}; "
                f"supported={sorted(supported)}"
            )
        normalized = f"{int(match.group(1))}.{int(match.group(2))}"
        if normalized not in resolved:
            resolved.append(normalized)
    if not resolved:
        raise RuntimeError("No CUDA architectures selected for FusedLayerNorm")
    return ";".join(resolved)


def compile(
    name: str,
    sources: list[str],
    extra_include_paths: list[str],
    build_directory: Optional[str] = None,
) -> Any:
    # cpp_extension translates this into one gencode pair per visible GPU
    # architecture. Avoid adding a second manual set of duplicate flags.
    os.environ["TORCH_CUDA_ARCH_LIST"] = _resolve_cuda_arch_list()

    extension_module = load(
        name=name,
        sources=sources,
        extra_include_paths=extra_include_paths,
        extra_cflags=[
            "-O3",
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
            "-std=c++17",
            "-maxrregcount=32",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ],
        verbose=True,
        build_directory=build_directory,
    )
    if build_directory is not None:
        from protenix.model.layer_norm.cache_provenance import (
            write_cache_provenance,
        )

        write_cache_provenance(build_directory, extension_module)
    return extension_module
