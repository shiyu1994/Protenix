#!/usr/bin/env python
"""Two-rank CUDA lifecycle smoke test for the Protenix FSDP2 helpers.

Run independently from the repository root with::

    CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$(pwd)" \
      torchrun --standalone --nproc_per_node=2 \
      tests/distributed_fsdp2_smoke.py

Set ``FSDP2_SMOKE_RESHARD_AFTER_FORWARD=false`` to exercise the persistent
unsharded-after-forward mode. This toy smoke intentionally uses a global mesh;
multi-node hybrid meshes and the full Protenix runner are separate integration
tests.
"""

from __future__ import annotations

import copy
import os

import torch
import torch.distributed as dist

from protenix.utils.fsdp import (
    apply_fsdp2,
    build_fsdp_device_mesh,
    get_fsdp2_full_state_dict,
    set_fsdp2_full_model_state_dict,
    set_fsdp2_full_optimizer_state_dict,
    sync_module_states,
)
from runner.ema import EMAWrapper


def stage(name: str) -> None:
    print(f"rank={dist.get_rank()} stage={name}", flush=True)


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


class BlockContainer(torch.nn.Module):
    def __init__(self, count: int, width: int):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(width, width) for _ in range(count)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = torch.tanh(block(x))
        return x


class DiffusionModule(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.diffusion_transformer = BlockContainer(3, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.diffusion_transformer(x)


class ConfidenceHead(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.pairformer_stack = BlockContainer(2, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pairformer_stack(x)


class ToyProtenix(torch.nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        self.msa_module = BlockContainer(2, width)
        self.pairformer_stack = BlockContainer(4, width)
        self.diffusion_module = DiffusionModule(width)
        self.confidence_head = ConfidenceHead(width)
        self.output = torch.nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.msa_module(x)
        x = self.pairformer_stack(x)
        x = self.diffusion_module(x)
        x = self.confidence_head(x)
        return self.output(x)


def train_step(model, optimizer, x):
    model.train()
    loss = model(x).square().mean()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    from torch.distributed.tensor import DTensor

    if isinstance(grad_norm, DTensor):
        grad_norm = grad_norm.full_tensor()
    assert torch.isfinite(grad_norm)
    gathered_norms = [torch.zeros_like(grad_norm) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_norms, grad_norm)
    for gathered_norm in gathered_norms[1:]:
        torch.testing.assert_close(gathered_norm, gathered_norms[0])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach()


def clone_rank0_state(model_state, optimizer_state):
    if dist.get_rank() != 0:
        return {}, {}
    cloned_model = {key: value.clone() for key, value in model_state.items()}
    return cloned_model, copy.deepcopy(optimizer_state)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        assert dist.get_world_size() > 1, "run with at least two torchrun ranks"
        reshard_after_forward = env_flag(
            "FSDP2_SMOKE_RESHARD_AFTER_FORWARD",
            True,
        )
        stage(f"reshard-after-forward-{reshard_after_forward}")

        torch.manual_seed(1000 + dist.get_rank())
        model = ToyProtenix().cuda(local_rank)
        stage("model-created")
        sync_module_states(model)
        stage("model-synced")
        mesh = build_fsdp_device_mesh("global")
        model = apply_fsdp2(
            model,
            mesh=mesh,
            param_dtype="fp32",
            reduce_dtype="fp32",
            reshard_after_forward=reshard_after_forward,
        )
        stage("model-sharded")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ema = EMAWrapper(model, decay=0.9, shadow_device="cpu")
        ema.register()
        assert all(value.device.type == "cpu" for value in ema.shadow.values())
        stage("ema-registered")

        local_shadow_numel = torch.tensor(
            sum(value.numel() for value in ema.shadow.values()),
            device=local_rank,
            dtype=torch.int64,
        )
        dist.all_reduce(local_shadow_numel)
        global_model_numel = sum(param.numel() for param in model.parameters())
        assert local_shadow_numel.item() == global_model_numel
        stage("ema-size-checked")

        generator = torch.Generator(device=f"cuda:{local_rank}")
        generator.manual_seed(2000 + dist.get_rank())
        x = torch.randn(8, 16, generator=generator, device=local_rank)

        train_step(model, optimizer, x)
        stage("first-train-step")
        ema.update()
        stage("ema-updated")

        model.eval()
        with torch.no_grad():
            live_output = model(x).clone()
        stage("live-eval")

        ema.apply_shadow()
        stage("ema-applied")
        with torch.no_grad():
            ema_output = model(x).clone()
        assert torch.isfinite(ema_output).all()
        ema_model_state, _ = get_fsdp2_full_state_dict(model, optimizer)
        saved_ema_model_state, _ = clone_rank0_state(ema_model_state, {})
        stage("ema-checkpoint-gathered")

        ema.restore()
        assert not ema.backup
        stage("ema-restored")
        with torch.no_grad():
            restored_output = model(x)
        torch.testing.assert_close(restored_output, live_output, rtol=0, atol=0)

        model_state, optimizer_state = get_fsdp2_full_state_dict(model, optimizer)
        saved_model_state, saved_optimizer_state = clone_rank0_state(
            model_state,
            optimizer_state,
        )
        if dist.get_rank() == 0:
            assert saved_model_state
            assert saved_optimizer_state
            assert all(
                not key.startswith("module.") for key in saved_model_state
            )
        stage("live-checkpoint-gathered")

        train_step(model, optimizer, x)
        stage("second-train-step")

        # Reproduce the runner's two-file EMA/raw resume ordering: load the EMA
        # model params, register them as shadow, then load live model+optimizer.
        set_fsdp2_full_model_state_dict(
            model,
            saved_ema_model_state,
            strict=True,
        )
        ema.register()
        stage("ema-checkpoint-reloaded")
        set_fsdp2_full_model_state_dict(
            model,
            saved_model_state,
            strict=True,
        )
        set_fsdp2_full_optimizer_state_dict(
            model,
            optimizer,
            saved_optimizer_state,
        )
        stage("live-checkpoint-reloaded")

        model.eval()
        with torch.no_grad():
            reloaded_live_output = model(x)
        torch.testing.assert_close(reloaded_live_output, live_output, rtol=0, atol=0)

        ema.apply_shadow()
        with torch.no_grad():
            reloaded_ema_output = model(x)
        torch.testing.assert_close(reloaded_ema_output, ema_output, rtol=0, atol=0)
        ema.restore()
        with torch.no_grad():
            reloaded_restored_output = model(x)
        torch.testing.assert_close(
            reloaded_restored_output,
            live_output,
            rtol=0,
            atol=0,
        )
        stage("ema-live-round-trip-checked")

        resumed_loss = train_step(model, optimizer, x)
        assert torch.isfinite(resumed_loss)
        stage("resumed-train-step")
        dist.barrier()
        if dist.get_rank() == 0:
            print("FSDP2 two-rank lifecycle smoke passed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
