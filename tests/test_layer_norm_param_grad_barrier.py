# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_DIR = Path(__file__).resolve().parents[1]
CUDA_SOURCE = (
    REPO_DIR
    / "protenix/model/layer_norm/kernel/layer_norm_cuda_kernel.cu"
)
STEP2_KERNELS = (
    "LayerNormParamGradStep2",
    "LayerNormGammaGradStep2",
    "LayerNormBetaGradStep2",
)


def _kernel_body(source: str, kernel_name: str) -> str:
    marker = f"__global__ void {kernel_name}"
    start = source.index(marker)
    opening_brace = source.index("{", start)
    depth = 0
    for index in range(opening_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening_brace + 1 : index]
    raise AssertionError(f"unterminated CUDA kernel {kernel_name}")


def test_step2_column_tail_does_not_guard_block_barriers():
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    for kernel_name in STEP2_KERNELS:
        body = _kernel_body(source, kernel_name)
        assert "const bool column_is_valid = i2 < col;" in body
        assert "if (i2 < col) {" not in body
        assert body.count("__syncthreads();") == 2
        assert body.index("if (column_is_valid)") < body.index(
            "for (int offset = blockDim.y / 2; offset >= 1; offset /= 2)"
        )
        assert "if (column_is_valid && threadIdx.y >= offset" in body
        assert "if (column_is_valid && threadIdx.y < offset)" in body


def test_cuda_source_participates_in_extension_cache_fingerprint():
    helper = (REPO_DIR / "scripts/runtime/cuda126_env.sh").read_text(
        encoding="utf-8"
    )
    provenance = (
        REPO_DIR / "protenix/model/layer_norm/cache_provenance.py"
    ).read_text(encoding="utf-8")
    assert "-name '*.cu'" in helper
    assert "PROTENIX_LAYER_NORM_SOURCE_SHA256" in helper
    assert (
        '"source_sha256": _required_env("PROTENIX_LAYER_NORM_SOURCE_SHA256")'
        in provenance
    )


@pytest.mark.parametrize(
    ("create_scale", "create_offset"),
    ((True, True), (True, False), (False, True)),
)
@pytest.mark.parametrize("dtype_name", ("float32", "bfloat16"))
def test_fused_layer_norm_833_forward_backward_matches_torch(
    create_scale: bool,
    create_offset: bool,
    dtype_name: str,
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")

    from protenix.model.layer_norm.layer_norm import FusedLayerNorm

    dtype = getattr(torch, dtype_name)
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260818)
    input_fused = torch.randn(
        65, 833, device=device, dtype=dtype, generator=generator
    ).requires_grad_(True)
    input_reference = input_fused.detach().clone().requires_grad_(True)
    layer = FusedLayerNorm(
        833,
        create_scale=create_scale,
        create_offset=create_offset,
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        if layer.weight is not None:
            layer.weight.copy_(
                1.0
                + 0.1
                * torch.randn(
                    833, device=device, dtype=dtype, generator=generator
                )
            )
        if layer.bias is not None:
            layer.bias.copy_(
                0.1
                * torch.randn(
                    833, device=device, dtype=dtype, generator=generator
                )
            )

    reference_weight = (
        None
        if layer.weight is None
        else layer.weight.detach().clone().requires_grad_(True)
    )
    reference_bias = (
        None
        if layer.bias is None
        else layer.bias.detach().clone().requires_grad_(True)
    )
    output_fused = layer(input_fused)
    output_reference = torch.nn.functional.layer_norm(
        input_reference,
        (833,),
        reference_weight,
        reference_bias,
        layer.eps,
    )
    grad_output = torch.randn(
        output_fused.shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    fused_parameters = [input_fused]
    reference_parameters = [input_reference]
    if layer.weight is not None:
        fused_parameters.append(layer.weight)
        reference_parameters.append(reference_weight)
    if layer.bias is not None:
        fused_parameters.append(layer.bias)
        reference_parameters.append(reference_bias)
    fused_gradients = torch.autograd.grad(
        output_fused, fused_parameters, grad_output
    )
    reference_gradients = torch.autograd.grad(
        output_reference, reference_parameters, grad_output
    )
    torch.cuda.synchronize()

    if dtype == torch.float32:
        rtol, atol = 5e-4, 5e-4
    else:
        rtol, atol = 5e-2, 5e-2
    torch.testing.assert_close(
        output_fused, output_reference, rtol=rtol, atol=atol
    )
    for fused_gradient, reference_gradient in zip(
        fused_gradients, reference_gradients
    ):
        torch.testing.assert_close(
            fused_gradient,
            reference_gradient,
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.skipif(
    os.environ.get("PROTENIX_CUDA126_ENV_CONFIGURED") != "1",
    reason="cache rebuild smoke requires the production CUDA environment",
)
def test_interrupted_cache_rebuilds_once_for_concurrent_imports(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    build_directory = tmp_path / "fast_layer_norm_cuda_v2"
    build_directory.mkdir()
    for name in (
        "fast_layer_norm_cuda_v2.so", "layer_norm_cuda.o",
        "layer_norm_cuda_kernel.cuda.o", "build.ninja", "lock",
    ):
        (build_directory / name).write_bytes(b"interrupted-build")
    environment = os.environ.copy()
    environment.update(
        TORCH_EXTENSIONS_DIR=str(tmp_path),
        PROTENIX_LAYER_NORM_BUILD_DIR=str(build_directory),
        PYTHONPATH=str(REPO_DIR),
        MAX_JOBS="2",
    )
    code = f"""
import runpy
tests = runpy.run_path({str(Path(__file__).resolve())!r})
for dtype_name in ('float32', 'bfloat16'):
    for create_scale, create_offset in ((True, True), (True, False), (False, True)):
        tests['test_fused_layer_norm_833_forward_backward_matches_torch'](
            create_scale, create_offset, dtype_name
        )
print('833 forward/backward passed')
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code], cwd=REPO_DIR, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(2)
    ]
    try:
        outputs = [process.communicate(timeout=240) for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    for process, (stdout, stderr) in zip(processes, outputs):
        assert process.returncode == 0, stdout + stderr
        assert "833 forward/backward passed" in stdout
    assert sum(
        stderr.count("Rebuilding incomplete LayerNorm cache")
        for _, stderr in outputs
    ) == 1
    record = build_directory / "protenix_cuda_extension_provenance.json"
    binary = build_directory / "fast_layer_norm_cuda_v2.so"
    previous = [(path.read_bytes(), path.stat().st_mtime_ns) for path in (binary, record)]
    warm = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_DIR, env=environment,
        capture_output=True, text=True, timeout=60,
    )
    assert warm.returncode == 0, warm.stdout + warm.stderr
    assert "Rebuilding incomplete LayerNorm cache" not in warm.stderr
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in (binary, record)] == previous
