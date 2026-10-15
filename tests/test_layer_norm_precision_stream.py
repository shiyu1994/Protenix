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

"""Numerical and current-stream regressions for the production CUDA extension."""

from contextlib import nullcontext

import pytest


torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")

EPS = 1e-5
AFFINE_MODES = ("both", "weight", "bias", "none")
DTYPE_NAMES = ("float32", "bfloat16", "float16")
# These widths cover all 16/8/4/2-byte vector dispatches that each dtype supports.
STREAM_CASES = [
    (dtype_name, cols)
    for dtype_name in DTYPE_NAMES
    for cols in ((128, 130, 129) if dtype_name == "float32" else (128, 132, 130, 129))
]
STREAM_DELAY_CYCLES = 20_000_000


@pytest.fixture(scope="module")
def layer_norm_module():
    # Use the real loader/cache configuration; compilation/import errors must fail.
    from protenix.model.layer_norm import layer_norm

    return layer_norm


def _dtype(dtype_name):
    if dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")
    return getattr(torch, dtype_name)


def _samples(dtype, cols, kind="random", fp32_affine=False):
    generator = torch.Generator().manual_seed(718203 + cols)
    x = torch.randn(33, cols, generator=generator)
    if kind == "constant":
        x = x * 0.001
        weight = torch.full((cols,), 1.1)
        dy = torch.full_like(x, 1.1)
    else:
        weight = 1.0 + 0.1 * torch.randn(cols, generator=generator)
        dy = torch.randn(x.shape, generator=generator)
    bias = 0.1 * torch.randn(cols, generator=generator)
    parameter_dtype = torch.float32 if fp32_affine else dtype
    return x.to(dtype), weight.to(parameter_dtype), bias.to(parameter_dtype), dy.to(dtype)


def _make_layer(layer_norm_module, mode, weight, bias):
    layer = layer_norm_module.FusedLayerNorm(
        weight.numel(),
        create_scale=mode in ("both", "weight"),
        create_offset=mode in ("both", "bias"),
        eps=EPS,
    ).to(device="cuda", dtype=weight.dtype)
    with torch.no_grad():
        if layer.weight is not None:
            layer.weight.copy_(weight)
        if layer.bias is not None:
            layer.bias.copy_(bias)
    return layer


def _reference(x, weight, bias, dy, mode):
    # Convert AFTER quantization: the reference receives exactly the kernel inputs.
    x = x.double().requires_grad_(True)
    weight = weight.double().requires_grad_(True) if mode in ("both", "weight") else None
    bias = bias.double().requires_grad_(True) if mode in ("both", "bias") else None
    output = torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, EPS)
    targets = [x] + [parameter for parameter in (weight, bias) if parameter is not None]
    gradients = torch.autograd.grad(output, targets, dy.double())
    return output.detach(), *gradients


@pytest.mark.parametrize("dtype_name", DTYPE_NAMES)
@pytest.mark.parametrize("cols", (128, 256, 833))
@pytest.mark.parametrize("mode", AFFINE_MODES)
@pytest.mark.parametrize("kind", ("constant", "random"))
def test_input_gradient_matches_quantized_fp64_reference(layer_norm_module, dtype_name, cols, mode, kind):
    dtype = _dtype(dtype_name)
    samples = _samples(dtype, cols, kind)
    x_cpu, weight, bias, dy = samples
    expected = _reference(*samples, mode)
    layer = _make_layer(layer_norm_module, mode, weight, bias)
    x = x_cpu.cuda().requires_grad_(True)
    output = layer(x)
    gradients = torch.autograd.grad(output, (x, *layer.parameters()), dy.cuda())
    actual = tuple(value.detach().cpu().double() for value in (output, *gradients))

    tolerances = {"float32": (2e-5, 2e-5), "bfloat16": (8e-3, 2e-3), "float16": (1e-3, 2e-4)}
    rtol, atol = tolerances[dtype_name]
    for index, (result, reference) in enumerate(zip(actual, expected)):
        assert torch.isfinite(result).all()
        if kind == "constant" and index == 1:
            # Constant gamma and dy make the true dx zero. A rounded BF16/FP16
            # gamma*dy inside the reduction previously produced a large offset.
            torch.testing.assert_close(reference, torch.zeros_like(reference), rtol=0, atol=1e-10)
            torch.testing.assert_close(result, reference, rtol=0, atol=3e-4)
        else:
            torch.testing.assert_close(result, reference, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype_name", DTYPE_NAMES)
@pytest.mark.parametrize("cols", (128, 256))
def test_input_gradient_second_reduction_cancellation(layer_norm_module, dtype_name, cols):
    dtype = _dtype(dtype_name)
    x_cpu = torch.tensor((-1.0, 1.0), dtype=dtype).repeat(3, cols // 2)
    weight = torch.full((cols,), 1.1, dtype=dtype)
    bias = torch.zeros_like(weight)
    dy = x_cpu * weight
    expected = _reference(x_cpu, weight, bias, dy, "weight")[1]

    # For quantized a=gamma=abs(dy), mean(x)=mean(gamma*dy)=0, while
    # mean(x*gamma*dy)=a**2. The k1 reduction cancels exactly; k2 must preserve
    # the product before cancellation against the direct gamma*dy term.
    analytic = x_cpu.double() * weight.double().square() * EPS / (1.0 + EPS) ** 1.5
    torch.testing.assert_close(expected, analytic, rtol=0, atol=1e-12)
    layer = _make_layer(layer_norm_module, "weight", weight, bias)
    x = x_cpu.cuda().requires_grad_(True)
    (gradient,) = torch.autograd.grad(layer(x), (x,), dy.cuda())
    # True dx is about 1.2e-5. FP32 arithmetic contributes much less than
    # 1e-6; rounding just k2's product to FP16/BF16 leaves a much larger error.
    torch.testing.assert_close(gradient.cpu().double(), expected, rtol=0, atol=1e-6)


def _extension_forward(extension, mode, x, weight, bias):
    affine = ([weight] if mode in ("both", "weight") else []) + ([bias] if mode in ("both", "bias") else [])
    suffix = "none_affine" if mode == "none" else f"with_{mode}_affine"
    return getattr(extension, "forward_" + suffix)(x, [x.shape[-1]], *affine, EPS)


def _extension_backward(extension, mode, dy, stats, x, weight, bias):
    affine = ([weight] if mode in ("both", "weight") else []) + ([bias] if mode in ("both", "bias") else [])
    suffix = "none_affine" if mode == "none" else f"with_{mode}_affine"
    return getattr(extension, "backward_" + suffix)(dy, *stats, x, [x.shape[-1]], *affine, EPS)


def _consume(values):
    return tuple(None if value is None else value.clone() for value in values)


def _assert_equal(actual, expected):
    assert len(actual) == len(expected)
    for index, (result, reference) in enumerate(zip(actual, expected)):
        if result is None or reference is None:
            assert result is reference, f"tensor {index}: inconsistent optional output"
        else:
            assert torch.isfinite(result).all(), f"tensor {index}: nonfinite output"
            assert torch.equal(result, reference), f"tensor {index}: stream changed the result"


@pytest.mark.parametrize(("dtype_name", "cols"), STREAM_CASES)
@pytest.mark.parametrize("mode", AFFINE_MODES)
def test_nondefault_stream_producer_and_consumers_are_exactly_equal(layer_norm_module, dtype_name, cols, mode):
    dtype = _dtype(dtype_name)
    extension = layer_norm_module.fast_layer_norm_cuda_v2
    default = torch.cuda.default_stream()
    side = torch.cuda.Stream()
    with torch.cuda.stream(default):
        sources = tuple(value.cuda() for value in _samples(dtype, cols))
        x, weight, bias, dy = sources
        baseline_forward = _extension_forward(extension, mode, x, weight, bias)
        baseline_backward = _extension_backward(extension, mode, dy, baseline_forward[1:], x, weight, bias)
        staged = tuple(torch.zeros_like(value) for value in sources)
    default.synchronize()

    # The caller owns every source, destination, result and statistic through
    # completion. There is no cross-stream lifetime/reuse race in this test.
    side.wait_stream(default)
    with torch.cuda.stream(side):
        torch.cuda._sleep(STREAM_DELAY_CYCLES)
        for destination, source in zip(staged, sources):
            destination.copy_(source)
        sx, sw, sb, sdy = staged
        forward = _extension_forward(extension, mode, sx, sw, sb)
        backward = _extension_backward(extension, mode, sdy, forward[1:], sx, sw, sb)
        consumed = _consume((*forward, *backward))
    # Synchronizing after the consumer cannot repair an earlier out-of-order read.
    torch.cuda.synchronize()
    _assert_equal(consumed, (*baseline_forward, *baseline_backward))

    # Independently exercise backward with ready inputs and unrelated default-
    # stream work. A backward launch on default must not delay its side consumer.
    with torch.cuda.stream(default):
        torch.cuda._sleep(STREAM_DELAY_CYCLES)
    with torch.cuda.stream(side):
        independent_backward = _extension_backward(extension, mode, dy, baseline_forward[1:], x, weight, bias)
        backward_consumed = _consume(independent_backward)
    torch.cuda.synchronize()
    _assert_equal(backward_consumed, baseline_backward)


def _run_wrapper(layer_norm_module, sources, mode, execution, stream):
    from torch.utils.checkpoint import checkpoint

    x_source, weight, bias, dy = sources
    layer = _make_layer(layer_norm_module, mode, weight, bias)
    x = torch.zeros_like(x_source).requires_grad_(True)
    # Finish caller-side setup before transferring ownership to the chosen stream.
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(STREAM_DELAY_CYCLES)
        with torch.no_grad():
            x.copy_(x_source)
        context = torch.autograd.graph.save_on_cpu(pin_memory=True) if execution == "offload" else nullcontext()
        with context:
            output = layer(x) if execution == "plain" else checkpoint(layer, x, use_reentrant=False)
        gradients = torch.autograd.grad(output, (x, *layer.parameters()), dy)
        consumed = _consume((output, *gradients))
    # Keep layer, x, output and gradients alive until both streams have finished.
    torch.cuda.synchronize()
    return consumed


@pytest.mark.parametrize("dtype_name", DTYPE_NAMES)
@pytest.mark.parametrize("mode", AFFINE_MODES)
@pytest.mark.parametrize("execution", ("checkpoint", "offload"))
def test_wrapper_checkpoint_and_cpu_offload_preserve_stream_results(layer_norm_module, dtype_name, mode, execution):
    dtype = _dtype(dtype_name)
    # Production-style FP32 parameters exercise wrapper casts for BF16/FP16.
    sources = tuple(value.cuda() for value in _samples(dtype, 833, fp32_affine=True))
    baseline = _run_wrapper(layer_norm_module, sources, mode, "plain", torch.cuda.default_stream())
    actual = _run_wrapper(layer_norm_module, sources, mode, execution, torch.cuda.Stream())
    _assert_equal(actual, baseline)
