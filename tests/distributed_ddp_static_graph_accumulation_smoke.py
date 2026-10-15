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
"""Two-rank regression for static-graph DDP gradient accumulation.

Run from the repository root with::

    CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$(pwd)" \
      torchrun --standalone --nproc_per_node=2 \
      tests/distributed_ddp_static_graph_accumulation_smoke.py

PyTorch 2.7.1 fails when the first backward of a ``static_graph=True`` DDP
instance runs inside ``no_sync()``. This smoke verifies that Protenix forces
that lifecycle-local backward to synchronize, then restores normal
three-``no_sync``-microstep accumulation, including for a newly constructed
resumed process whose global step is already non-zero.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from runner.train import gradient_accumulation_sync_context


class ScalarModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.weight * value


def build_model(local_rank: int) -> DDP:
    model = ScalarModel().to(local_rank)
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        static_graph=True,
        gradient_as_bucket_view=True,
    )


def backward_window(
    model: DDP,
    *,
    local_rank: int,
    first_global_step: int,
    local_values: tuple[float, float, float, float],
    expected_gradient: float,
    force_first_backward_sync: bool,
) -> None:
    first_backward_sync_pending = force_first_backward_sync
    for offset, value in enumerate(local_values):
        force_sync = first_backward_sync_pending
        with gradient_accumulation_sync_context(
            model,
            use_ddp=True,
            global_step=first_global_step + offset,
            iters_to_accumulate=4,
            force_sync=force_sync,
        ):
            loss = model(torch.tensor(value, device=local_rank)) / 4
            loss.backward()
        first_backward_sync_pending = False

    expected = torch.tensor(expected_gradient, device=local_rank)
    torch.testing.assert_close(model.module.weight.grad, expected, rtol=0, atol=0)
    gathered = [torch.zeros_like(expected) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, model.module.weight.grad)
    for rank_gradient in gathered:
        torch.testing.assert_close(rank_gradient, expected, rtol=0, atol=0)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        assert dist.get_world_size() == 2, "run this smoke with exactly two ranks"
        rank = dist.get_rank()

        # Initial process: rank-local values are (1, 2, 3, 4) and
        # (5, 6, 7, 8). Their global mean gradient is 4.5.
        initial_model = build_model(local_rank)
        backward_window(
            initial_model,
            local_rank=local_rank,
            first_global_step=0,
            local_values=tuple(
                float(value + 4 * rank) for value in range(1, 5)
            ),
            expected_gradient=4.5,
            force_first_backward_sync=True,
        )
        initial_model.zero_grad(set_to_none=True)

        # Once initialized, three microbatches use no_sync and the fourth
        # synchronizes. Values (9..12) and (13..16) yield gradient 12.5.
        backward_window(
            initial_model,
            local_rank=local_rank,
            first_global_step=4,
            local_values=tuple(
                float(value + 4 * rank) for value in range(9, 13)
            ),
            expected_gradient=12.5,
            force_first_backward_sync=False,
        )
        del initial_model
        dist.barrier()

        # Resume process: a fresh DDP reducer still needs its first synchronized
        # backward even though the restored global step is non-zero.
        resumed_model = build_model(local_rank)
        backward_window(
            resumed_model,
            local_rank=local_rank,
            first_global_step=100,
            local_values=tuple(
                float(value + 4 * rank) for value in range(17, 21)
            ),
            expected_gradient=20.5,
            force_first_backward_sync=True,
        )
        print(f"rank={rank} static_graph_accumulation_smoke=passed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
