#!/usr/bin/env python
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
"""CPU/gloo smoke for deferred loss-health synchronization.

Run from the repository root with::

    PYTHONPATH="$(pwd)" torchrun --standalone --nproc_per_node=2 \
      tests/distributed_deferred_loss_health_sync_smoke.py

The smoke injects NaN on one designated rank at the first and fourth
microbatch of separate four-way accumulation windows.  It verifies that every
rank skips the same optimizer update, clears gradients, backs off GradScaler to
the same value, and then performs the same recovery update.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import runner.train as train_module
from runner.train import AF3Trainer, gradient_accumulation_sync_context


class _MetricSink:
    def add(self, *args, **kwargs) -> None:
        pass


class _CountingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure=closure)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.weight * value


def _build_trainer() -> AF3Trainer:
    trainer = AF3Trainer.__new__(AF3Trainer)
    trainer.model = DDP(_TinyModel(), static_graph=True)
    trainer.optimizer = _CountingSGD(trainer.model.parameters(), lr=0.1)
    trainer.lr_scheduler = _CountingScheduler()
    trainer.train_metric_wrapper = _MetricSink()
    trainer.global_step = 0
    trainer.step = 0
    trainer.start_step = 0
    trainer.iters_to_accumulate = 4
    trainer._skip_accumulated_update = False
    trainer.device = torch.device("cpu")
    trainer.grad_scaler = torch.GradScaler(
        device="cpu",
        init_scale=128.0,
        backoff_factor=0.5,
        growth_interval=100,
        enabled=True,
    )
    trainer.configs = SimpleNamespace(
        dtype="fp32",
        grad_clip_norm=10.0,
        empty_cache_interval=0,
        optimizer_sharding="none",
    )
    trainer.print = lambda message: None
    trainer.numeric_errors = []

    def log_numeric_error(self, reason, batch, loss_dict) -> None:
        self.numeric_errors.append(reason)

    def model_forward(self, batch, mode="train"):
        batch = dict(batch)
        batch["output"] = self.model(batch["x"])
        return batch, {"N_cycle": 1}

    def get_loss(self, batch, mode="train"):
        output = batch["output"]
        if batch["inject_nan"]:
            loss = output.square().mean() * output.new_tensor(float("nan"))
        else:
            loss = output.square().mean()
        return loss, {"loss": loss.detach()}, batch

    trainer.log_numeric_error = MethodType(log_numeric_error, trainer)
    trainer.model_forward = MethodType(model_forward, trainer)
    trainer.get_loss = MethodType(get_loss, trainer)
    return trainer


def _gather_int(value: int) -> list[int]:
    local = torch.tensor(value, dtype=torch.int64)
    gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return [int(item.item()) for item in gathered]


def _gather_float(value: float) -> list[float]:
    local = torch.tensor(value, dtype=torch.float64)
    gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return [float(item.item()) for item in gathered]


def _assert_rank_consistent_state(trainer: AF3Trainer) -> None:
    weights = _gather_float(float(trainer.model.module.weight.detach().item()))
    scales = _gather_float(float(trainer.grad_scaler.get_scale()))
    optimizer_steps = _gather_int(trainer.optimizer.step_calls)
    scheduler_steps = _gather_int(trainer.lr_scheduler.steps)
    assert len(set(weights)) == 1, weights
    assert len(set(scales)) == 1, scales
    assert len(set(optimizer_steps)) == 1, optimizer_steps
    assert len(set(scheduler_steps)) == 1, scheduler_steps


def _run_window(
    trainer: AF3Trainer,
    *,
    nan_rank: int | None,
    nan_microstep: int | None,
    force_first_backward_sync: bool = False,
) -> tuple[list[dict], int]:
    rank = dist.get_rank()
    results = []
    first_all_reduce_count = trainer._python_all_reduce_count
    for microstep in range(4):
        force_sync = force_first_backward_sync and microstep == 0
        with gradient_accumulation_sync_context(
            trainer.model,
            use_ddp=True,
            global_step=trainer.global_step,
            iters_to_accumulate=trainer.iters_to_accumulate,
            force_sync=force_sync,
        ):
            result = trainer.train_step(
                {
                    "x": torch.ones(1),
                    "inject_nan": (
                        nan_rank is not None
                        and rank == nan_rank
                        and microstep == nan_microstep
                    ),
                }
            )
        trainer.global_step += 1
        results.append(result)
    return results, trainer._python_all_reduce_count - first_all_reduce_count


def main() -> None:
    dist.init_process_group("gloo")
    real_all_reduce = train_module.dist.all_reduce
    try:
        assert dist.get_world_size() == 2, "run this smoke with exactly two ranks"
        trainer = _build_trainer()
        trainer._python_all_reduce_count = 0

        def tracked_all_reduce(*args, **kwargs):
            trainer._python_all_reduce_count += 1
            return real_all_reduce(*args, **kwargs)

        train_module.dist.all_reduce = tracked_all_reduce
        initial_weight = trainer.model.module.weight.detach().clone()

        # Rank 1 owns a bad first microbatch.  Only the boundary performs the
        # Python loss-health all_reduce; gradient health is bypassed on skip.
        bad_first, collective_count = _run_window(
            trainer,
            nan_rank=1,
            nan_microstep=0,
            force_first_backward_sync=True,
        )
        assert collective_count == 1
        assert [
            item["timings"]["loss_health_sync_seconds"] == 0.0
            for item in bad_first[:3]
        ] == [True, True, True]
        assert bad_first[-1]["loss_nonfinite_in_window"] is True
        assert bad_first[-1]["optimizer_skipped"] is True
        assert trainer.optimizer.step_calls == 0
        assert trainer.lr_scheduler.steps == 0
        assert trainer.grad_scaler.get_scale() == 64.0
        assert torch.equal(trainer.model.module.weight, initial_weight)
        assert all(parameter.grad is None for parameter in trainer.model.parameters())
        _assert_rank_consistent_state(trainer)

        # A finite window recovers normally.  Its two Python reductions are
        # loss health and gradient health, both only at the boundary.
        recovery_one, collective_count = _run_window(
            trainer,
            nan_rank=None,
            nan_microstep=None,
        )
        assert collective_count == 2
        assert recovery_one[-1]["did_update"] is True
        assert trainer.optimizer.step_calls == 1
        assert trainer.lr_scheduler.steps == 1
        assert trainer.grad_scaler.get_scale() == 64.0
        _assert_rank_consistent_state(trainer)

        recovered_weight = trainer.model.module.weight.detach().clone()

        # Rank 0 now owns a bad boundary (fourth) microbatch.
        bad_fourth, collective_count = _run_window(
            trainer,
            nan_rank=0,
            nan_microstep=3,
        )
        assert collective_count == 1
        assert bad_fourth[-1]["loss_nonfinite_in_window"] is True
        assert bad_fourth[-1]["optimizer_skipped"] is True
        assert trainer.optimizer.step_calls == 1
        assert trainer.lr_scheduler.steps == 1
        assert trainer.grad_scaler.get_scale() == 32.0
        assert torch.equal(trainer.model.module.weight, recovered_weight)
        assert all(parameter.grad is None for parameter in trainer.model.parameters())
        _assert_rank_consistent_state(trainer)

        recovery_two, collective_count = _run_window(
            trainer,
            nan_rank=None,
            nan_microstep=None,
        )
        assert collective_count == 2
        assert recovery_two[-1]["did_update"] is True
        assert trainer.optimizer.step_calls == 2
        assert trainer.lr_scheduler.steps == 2
        assert trainer.grad_scaler.get_scale() == 32.0
        _assert_rank_consistent_state(trainer)

        print(
            f"rank={dist.get_rank()} "
            "deferred_loss_health_sync_smoke=passed",
            flush=True,
        )
    finally:
        train_module.dist.all_reduce = real_all_reduce
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
