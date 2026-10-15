import pytest
import torch
from torch.testing import assert_close

from protenix.model.modules.diffusion import DiffusionConditioning
from protenix.model.modules.pairformer import PairformerBlock
from protenix.model.modules.primitives import Transition


def _randomize_transition_parameters(module: Transition) -> None:
    generator = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.1)


def _forward_backward(
    module: Transition,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    module.train()
    x = x.detach().clone().requires_grad_(True)
    y = module(x)
    loss = y.square().mean()
    loss.backward()
    param_grads = [param.grad.detach().clone() for param in module.parameters()]
    return y.detach(), x.grad.detach().clone(), param_grads


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Transition LayerNorm uses a CUDA fused op"
)
def test_transition_training_chunk_matches_unchunked_forward_and_backward():
    device = torch.device("cuda")
    baseline = Transition(c_in=5, n=3).to(device)
    _randomize_transition_parameters(baseline)

    chunked = Transition(c_in=5, n=3, train_chunk_size=3).to(device)
    chunked.load_state_dict(baseline.state_dict())

    x = torch.randn(2, 7, 11, 5, device=device)

    baseline_y, baseline_x_grad, baseline_param_grads = _forward_backward(baseline, x)
    chunked_y, chunked_x_grad, chunked_param_grads = _forward_backward(chunked, x)

    assert_close(chunked_y, baseline_y, rtol=1e-6, atol=1e-6)
    assert_close(chunked_x_grad, baseline_x_grad, rtol=1e-6, atol=1e-6)
    for chunked_grad, baseline_grad in zip(chunked_param_grads, baseline_param_grads):
        assert_close(chunked_grad, baseline_grad, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Transition LayerNorm uses a CUDA fused op"
)
def test_transition_training_checkpointed_chunks_match_unchunked_forward_and_backward():
    device = torch.device("cuda")
    baseline = Transition(c_in=5, n=3).to(device)
    _randomize_transition_parameters(baseline)

    chunked = Transition(
        c_in=5,
        n=3,
        train_chunk_size=3,
        train_checkpoint_chunks=True,
    ).to(device)
    chunked.load_state_dict(baseline.state_dict())

    x = torch.randn(2, 7, 11, 5, device=device)

    baseline_y, baseline_x_grad, baseline_param_grads = _forward_backward(baseline, x)
    chunked_y, chunked_x_grad, chunked_param_grads = _forward_backward(chunked, x)

    assert_close(chunked_y, baseline_y, rtol=1e-6, atol=1e-6)
    assert_close(chunked_x_grad, baseline_x_grad, rtol=1e-6, atol=1e-6)
    for chunked_grad, baseline_grad in zip(chunked_param_grads, baseline_param_grads):
        assert_close(chunked_grad, baseline_grad, rtol=1e-6, atol=1e-6)


def test_pairformer_block_passes_training_chunk_options_to_pair_transition():
    block = PairformerBlock(
        c_z=16,
        c_s=0,
        c_hidden_mul=8,
        c_hidden_pair_att=4,
        no_heads_pair=4,
        train_pair_transition_chunk_size=4,
        train_pair_transition_checkpoint_chunks=True,
    )

    assert block.pair_transition.train_chunk_size == 4
    assert block.pair_transition.train_checkpoint_chunks is True


def test_diffusion_conditioning_passes_training_chunk_options_to_pair_transitions():
    conditioning = DiffusionConditioning(
        c_z=16,
        c_s=8,
        c_s_inputs=4,
        train_pair_transition_chunk_size=4,
        train_pair_transition_checkpoint_chunks=True,
    )

    assert conditioning.transition_z1.train_chunk_size == 4
    assert conditioning.transition_z1.train_checkpoint_chunks is True
    assert conditioning.transition_z2.train_chunk_size == 4
    assert conditioning.transition_z2.train_checkpoint_chunks is True


def _run_diffusion_conditioning_forward_backward(
    module: DiffusionConditioning,
    relp_feature: torch.Tensor,
    t_hat_noise_level: torch.Tensor,
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
):
    module.train()
    s_inputs = s_inputs.detach().clone().requires_grad_(True)
    s_trunk = s_trunk.detach().clone().requires_grad_(True)
    z_trunk = z_trunk.detach().clone().requires_grad_(True)
    single_s, pair_z = module(
        t_hat_noise_level=t_hat_noise_level,
        relp_feature=relp_feature,
        s_inputs=s_inputs,
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        pair_z=None,
    )
    (single_s.square().mean() + pair_z.square().mean()).backward()
    return single_s.detach(), pair_z.detach(), {
        "s_inputs": s_inputs.grad.detach().clone(),
        "s_trunk": s_trunk.grad.detach().clone(),
        "z_trunk": z_trunk.grad.detach().clone(),
        "params": [
            param.grad.detach().clone()
            for param in module.parameters()
            if param.grad is not None
        ],
    }


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Diffusion conditioning uses CUDA layer norm"
)
def test_diffusion_conditioning_chunked_transitions_match_unchunked_gradients():
    device = torch.device("cuda")
    torch.manual_seed(20260714)
    baseline = DiffusionConditioning(
        c_z=6,
        c_s=8,
        c_s_inputs=10,
        c_noise_embedding=12,
    ).to(device)
    chunked = DiffusionConditioning(
        c_z=6,
        c_s=8,
        c_s_inputs=10,
        c_noise_embedding=12,
        train_pair_transition_chunk_size=3,
        train_pair_transition_checkpoint_chunks=True,
    ).to(device)
    chunked.load_state_dict(baseline.state_dict())

    n_sample, n_token = 2, 5
    relp_dim = baseline.relpe.linear_no_bias.weight.shape[1]
    inputs = (
        torch.randn(n_token, n_token, relp_dim, device=device),
        torch.rand(n_sample, device=device) + 0.1,
        torch.randn(n_token, 10, device=device),
        torch.randn(n_token, 8, device=device),
        torch.randn(n_token, n_token, 6, device=device),
    )
    baseline_s, baseline_z, baseline_grads = (
        _run_diffusion_conditioning_forward_backward(baseline, *inputs)
    )
    chunked_s, chunked_z, chunked_grads = (
        _run_diffusion_conditioning_forward_backward(chunked, *inputs)
    )

    assert_close(chunked_s, baseline_s, rtol=1e-6, atol=1e-6)
    assert_close(chunked_z, baseline_z, rtol=1e-6, atol=1e-6)
    for key in ("s_inputs", "s_trunk", "z_trunk"):
        assert_close(
            chunked_grads[key], baseline_grads[key], rtol=1e-6, atol=1e-6
        )
    assert len(chunked_grads["params"]) == len(baseline_grads["params"])
    for chunked_grad, baseline_grad in zip(
        chunked_grads["params"], baseline_grads["params"]
    ):
        assert_close(chunked_grad, baseline_grad, rtol=1e-6, atol=1e-6)
