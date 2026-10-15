import copy
from contextlib import contextmanager

import pytest
import torch
import torch.utils.checkpoint  # Ensure torch.utils.checkpoint is registered.

from configs.configs_base import configs as configs_base
from protenix.config.config import parse_configs
from protenix.model.modules import pairformer as pairformer_module
from protenix.model.modules.pairformer import MSAModule, PairformerStack
from protenix.model.utils import checkpoint_blocks


def _add_one(x: torch.Tensor) -> torch.Tensor:
    return x + 1


def test_checkpoint_cpu_offload_selects_every_nth_region(monkeypatch):
    context_calls = []

    @contextmanager
    def fake_save_on_cpu(*, pin_memory):
        context_calls.append(pin_memory)
        yield

    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", fake_save_on_cpu)
    x = torch.tensor(1.0, requires_grad=True)
    (out,) = checkpoint_blocks(
        [_add_one] * 5,
        args=(x,),
        blocks_per_ckpt=1,
        checkpoint_cpu_offload_every_n=2,
    )
    out.backward()

    # Five regions select the one-based region indices 2 and 4.
    assert context_calls == [True, True]
    torch.testing.assert_close(x.grad, torch.tensor(1.0))


def test_checkpoint_cpu_offload_none_never_enters_context(monkeypatch):
    def unexpected_save_on_cpu(*, pin_memory):
        raise AssertionError(f"save_on_cpu unexpectedly called: {pin_memory=}")

    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", unexpected_save_on_cpu)
    x = torch.tensor(1.0, requires_grad=True)
    (out,) = checkpoint_blocks(
        [_add_one] * 3,
        args=(x,),
        blocks_per_ckpt=1,
        checkpoint_cpu_offload_every_n=None,
    )
    out.backward()

    torch.testing.assert_close(out, torch.tensor(4.0))
    torch.testing.assert_close(x.grad, torch.tensor(1.0))


def test_checkpoint_cpu_offload_is_inactive_when_grad_is_disabled(monkeypatch):
    def unexpected_checkpoint():
        raise AssertionError("checkpoint unexpectedly requested under no_grad")

    def unexpected_save_on_cpu(*, pin_memory):
        raise AssertionError(f"save_on_cpu unexpectedly called: {pin_memory=}")

    monkeypatch.setattr("protenix.model.utils.get_checkpoint_fn", unexpected_checkpoint)
    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", unexpected_save_on_cpu)
    with torch.no_grad():
        (out,) = checkpoint_blocks(
            [_add_one] * 3,
            args=(torch.tensor(1.0),),
            blocks_per_ckpt=1,
            checkpoint_cpu_offload_every_n=1,
        )

    torch.testing.assert_close(out, torch.tensor(4.0))


@pytest.mark.parametrize("invalid_value", [0, -1, False, 1.5, "2"])
def test_checkpoint_cpu_offload_rejects_non_positive_integer(invalid_value):
    with pytest.raises(
        ValueError,
        match="checkpoint_cpu_offload_every_n must be a positive integer or None",
    ):
        checkpoint_blocks(
            [_add_one],
            args=(torch.tensor(1.0),),
            blocks_per_ckpt=None,
            checkpoint_cpu_offload_every_n=invalid_value,
        )


class _AffineBlock(torch.nn.Module):
    def __init__(self, weight: float, bias: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight))
        self.bias = torch.nn.Parameter(torch.tensor(bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x * self.weight + self.bias)


def _run_affine_blocks(blocks, x, offload_every_n):
    (out,) = checkpoint_blocks(
        list(blocks),
        args=(x,),
        blocks_per_ckpt=2,
        checkpoint_cpu_offload_every_n=offload_every_n,
    )
    out.square().sum().backward()
    return (
        out.detach(),
        x.grad.detach().clone(),
        [
            parameter.grad.detach().clone()
            for block in blocks
            for parameter in block.parameters()
        ],
    )


def test_checkpoint_cpu_offload_preserves_outputs_and_gradients():
    baseline_blocks = torch.nn.ModuleList(
        [
            _AffineBlock(1.1, 0.1),
            _AffineBlock(0.9, -0.2),
            _AffineBlock(1.2, 0.3),
            _AffineBlock(0.8, -0.4),
        ]
    )
    offload_blocks = copy.deepcopy(baseline_blocks)
    baseline_x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    offload_x = baseline_x.detach().clone().requires_grad_(True)

    baseline = _run_affine_blocks(baseline_blocks, baseline_x, None)
    offloaded = _run_affine_blocks(offload_blocks, offload_x, 1)

    for baseline_value, offloaded_value in zip(baseline[:2], offloaded[:2]):
        torch.testing.assert_close(offloaded_value, baseline_value)
    for baseline_grad, offloaded_grad in zip(baseline[2], offloaded[2]):
        torch.testing.assert_close(offloaded_grad, baseline_grad)


def test_pairformer_stack_passes_its_offload_interval(monkeypatch):
    call = {}

    def fake_checkpoint_blocks(
        blocks, args, blocks_per_ckpt, checkpoint_cpu_offload_every_n=None
    ):
        call.update(
            blocks=len(blocks),
            blocks_per_ckpt=blocks_per_ckpt,
            offload_every_n=checkpoint_cpu_offload_every_n,
        )
        return args

    monkeypatch.setattr(pairformer_module, "checkpoint_blocks", fake_checkpoint_blocks)
    stack = PairformerStack(
        n_blocks=2,
        n_heads=2,
        c_z=8,
        c_s=4,
        blocks_per_ckpt=1,
        checkpoint_cpu_offload_every_n=2,
    )
    s = torch.zeros(3, 4, requires_grad=True)
    z = torch.zeros(3, 3, 8, requires_grad=True)
    stack(s, z, pair_mask=torch.ones(3, 3))

    assert call == {"blocks": 2, "blocks_per_ckpt": 1, "offload_every_n": 2}


def test_msa_module_passes_offload_interval_and_traces_depth(monkeypatch):
    call = {}

    def fake_sample(feat_dict, **_kwargs):
        return {key: value[:3] for key, value in feat_dict.items()}

    def fake_checkpoint_blocks(
        blocks, args, blocks_per_ckpt, checkpoint_cpu_offload_every_n=None
    ):
        call.update(
            blocks=len(blocks),
            blocks_per_ckpt=blocks_per_ckpt,
            offload_every_n=checkpoint_cpu_offload_every_n,
        )
        return args

    monkeypatch.setattr(
        pairformer_module,
        "sample_msa_feature_dict_random_without_replacement",
        fake_sample,
    )
    monkeypatch.setattr(pairformer_module, "checkpoint_blocks", fake_checkpoint_blocks)
    msa_module = MSAModule(
        n_blocks=1,
        c_m=4,
        c_z=8,
        c_s_inputs=6,
        blocks_per_ckpt=1,
        checkpoint_cpu_offload_every_n=3,
        msa_configs={
            "enable": True,
            "strategy": "random",
            "sample_cutoff": {"train": 3, "test": 3},
            "min_size": {"train": 1, "test": 1},
        },
    )
    features = {
        "msa": torch.zeros(5, 2, dtype=torch.long),
        "has_deletion": torch.zeros(5, 2),
        "deletion_value": torch.zeros(5, 2),
    }
    z = torch.zeros(2, 2, 8)
    s_inputs = torch.zeros(2, 6)
    pair_mask = torch.ones(2, 2)

    msa_module(features, z, s_inputs, pair_mask)
    msa_module({}, z, s_inputs, pair_mask)

    assert call == {"blocks": 1, "blocks_per_ckpt": 1, "offload_every_n": 3}
    assert msa_module.last_msa_input_depth is None
    assert msa_module.last_msa_sampled_depth is None
    assert msa_module.msa_depth_trace == (
        {"input_depth": 5, "sampled_depth": 3},
        {"input_depth": None, "sampled_depth": None},
    )
    assert not any("msa_depth" in key for key in msa_module.state_dict())

    msa_module.reset_msa_depth_trace()
    assert msa_module.msa_depth_trace == ()
    assert msa_module.last_msa_input_depth is None
    assert msa_module.last_msa_sampled_depth is None


def test_cpu_offload_config_defaults_off_and_parses_independently():
    defaults = parse_configs(
        configs=copy.deepcopy(configs_base),
        fill_required_with_null=True,
    )
    assert defaults.model.pairformer.checkpoint_cpu_offload_every_n is None
    assert defaults.model.msa_module.checkpoint_cpu_offload_every_n is None

    configured = parse_configs(
        configs=copy.deepcopy(configs_base),
        arg_str=(
            "--model.pairformer.checkpoint_cpu_offload_every_n 2 "
            "--model.msa_module.checkpoint_cpu_offload_every_n 1"
        ),
        fill_required_with_null=True,
    )
    assert configured.model.pairformer.checkpoint_cpu_offload_every_n == 2
    assert configured.model.msa_module.checkpoint_cpu_offload_every_n == 1
