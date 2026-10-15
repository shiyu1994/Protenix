import copy

import pytest
import torch

from protenix.model.modules.pairformer import PairformerStack


def _make_stack(blocks_per_ckpt: int) -> PairformerStack:
    stack = PairformerStack(
        n_blocks=4,
        n_heads=4,
        c_z=32,
        c_s=16,
        dropout=0.25,
        blocks_per_ckpt=blocks_per_ckpt,
        train_pair_transition_chunk_size=4,
        train_pair_transition_checkpoint_chunks=True,
    )
    with torch.no_grad():
        for param in stack.parameters():
            if param.ndim > 0:
                param.normal_(mean=0.0, std=0.02)
            else:
                param.fill_(0.02)
    return stack


def _run_stack(stack: PairformerStack, s: torch.Tensor, z: torch.Tensor):
    stack.train()
    pair_mask = torch.ones(
        z.shape[-3], z.shape[-2], dtype=z.dtype, device=z.device
    )
    out_s, out_z = stack(
        s,
        z,
        pair_mask=pair_mask,
        triangle_multiplicative="torch",
        triangle_attention="torch",
        inplace_safe=False,
        chunk_size=None,
    )
    loss = out_s.square().mean() + out_z.square().mean()
    loss.backward()
    param_grads = {
        name: param.grad.detach().clone()
        for name, param in stack.named_parameters()
        if param.grad is not None
    }
    return (
        out_s.detach(),
        out_z.detach(),
        s.grad.detach().clone(),
        z.grad.detach().clone(),
        param_grads,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Pairformer fused layer norm requires CUDA"
)
def test_pairformer_checkpoint_group_two_matches_group_one_backward():
    device = torch.device("cuda:0")
    torch.manual_seed(20260711)
    group_one = _make_stack(blocks_per_ckpt=1).to(device)
    group_two = _make_stack(blocks_per_ckpt=2).to(device)
    group_two.load_state_dict(copy.deepcopy(group_one.state_dict()))

    s = torch.randn(8, 16, device=device)
    z = torch.randn(8, 8, 32, device=device)
    torch.manual_seed(20260712)
    group_one_result = _run_stack(
        group_one,
        s.clone().detach().requires_grad_(True),
        z.clone().detach().requires_grad_(True),
    )
    torch.manual_seed(20260712)
    group_two_result = _run_stack(
        group_two,
        s.clone().detach().requires_grad_(True),
        z.clone().detach().requires_grad_(True),
    )

    for group_one_tensor, group_two_tensor in zip(
        group_one_result[:-1], group_two_result[:-1]
    ):
        torch.testing.assert_close(
            group_two_tensor, group_one_tensor, rtol=1e-5, atol=1e-5
        )

    group_one_grads = group_one_result[-1]
    group_two_grads = group_two_result[-1]
    assert group_two_grads.keys() == group_one_grads.keys()
    for name, group_one_grad in group_one_grads.items():
        torch.testing.assert_close(
            group_two_grads[name], group_one_grad, rtol=1e-5, atol=1e-5
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Pairformer fused layer norm requires CUDA"
)
def test_pairformer_checkpoint_groups_match_for_ten_optimizer_steps():
    device = torch.device("cuda:0")
    torch.manual_seed(20260713)
    group_one = _make_stack(blocks_per_ckpt=1).to(device)
    group_two = _make_stack(blocks_per_ckpt=2).to(device)
    group_two.load_state_dict(copy.deepcopy(group_one.state_dict()))
    optimizer_one = torch.optim.Adam(group_one.parameters(), lr=1e-4)
    optimizer_two = torch.optim.Adam(group_two.parameters(), lr=1e-4)
    s = torch.randn(8, 16, device=device)
    z = torch.randn(8, 8, 32, device=device)
    pair_mask = torch.ones(8, 8, device=device)

    def train_step(stack, optimizer):
        optimizer.zero_grad(set_to_none=True)
        out_s, out_z = stack(
            s,
            z,
            pair_mask=pair_mask,
            triangle_multiplicative="torch",
            triangle_attention="torch",
            inplace_safe=False,
            chunk_size=None,
        )
        loss = out_s.square().mean() + out_z.square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach()

    for _ in range(10):
        common_cpu_rng_state = torch.get_rng_state()
        common_rng_state = torch.cuda.get_rng_state(device)
        loss_one = train_step(group_one, optimizer_one)
        group_one_cpu_rng_state = torch.get_rng_state()
        group_one_rng_state = torch.cuda.get_rng_state(device)

        torch.set_rng_state(common_cpu_rng_state)
        torch.cuda.set_rng_state(common_rng_state, device)
        loss_two = train_step(group_two, optimizer_two)
        group_two_cpu_rng_state = torch.get_rng_state()
        group_two_rng_state = torch.cuda.get_rng_state(device)

        torch.testing.assert_close(loss_two, loss_one, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(group_two_cpu_rng_state, group_one_cpu_rng_state)
        torch.testing.assert_close(group_two_rng_state, group_one_rng_state)
        for parameter_one, parameter_two in zip(
            group_one.parameters(), group_two.parameters(), strict=True
        ):
            torch.testing.assert_close(
                parameter_two, parameter_one, rtol=1e-5, atol=1e-6
            )

        torch.set_rng_state(group_one_cpu_rng_state)
        torch.cuda.set_rng_state(group_one_rng_state, device)
