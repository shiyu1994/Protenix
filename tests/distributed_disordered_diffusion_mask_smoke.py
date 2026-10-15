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
"""Two-rank static-graph smoke for asymmetric diffusion supervision masks.

Run from the repository root with::

    PYTHONPATH="$(pwd)" torchrun --standalone --nproc_per_node=2 \
      tests/distributed_disordered_diffusion_mask_smoke.py

Each step gives one rank an empty smooth-LDDT mask and the other rank a normal
mask, then swaps their roles. This catches graph disconnection, NaN, unused
parameters, and rank divergence without requiring GPUs.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from protenix.model.loss import SmoothLDDTLoss


class CoordinateModel(torch.nn.Module):
    """Small coordinate parameter exercising the same loss graph boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.coordinate = torch.nn.Parameter(
            torch.tensor(
                [
                    [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.1, 0.0]],
                    [[0.1, 0.0, 0.0], [1.1, 0.1, 0.0], [0.0, 1.2, 0.1]],
                ]
            )
        )

    def forward(self) -> torch.Tensor:
        # Return a non-leaf output so DDP's reducer observes the normal model
        # autograd boundary rather than a parameter object directly.
        return self.coordinate * 1.0


def main() -> None:
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 2, "run this smoke with two ranks"
        rank = int(os.environ["RANK"])
        model = DDP(
            CoordinateModel(),
            static_graph=True,
            find_unused_parameters=False,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = SmoothLDDTLoss()
        true_coordinate = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        full_mask = 1.0 - torch.eye(3)

        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            predicted_coordinate = model()
            local_mask_is_empty = rank == step
            lddt_mask = (
                torch.zeros_like(full_mask)
                if local_mask_is_empty
                else full_mask
            )
            loss = loss_fn.sparse_forward(
                pred_coordinate=predicted_coordinate,
                true_coordinate=true_coordinate,
                lddt_mask=lddt_mask,
            )
            assert torch.isfinite(loss), (rank, step, loss)
            loss.backward()
            gradient = model.module.coordinate.grad
            assert gradient is not None
            assert torch.isfinite(gradient).all()
            optimizer.step()

            gathered = [
                torch.empty_like(model.module.coordinate)
                for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered, model.module.coordinate.detach())
            for rank_coordinate in gathered:
                torch.testing.assert_close(
                    rank_coordinate,
                    gathered[0],
                    rtol=0,
                    atol=0,
                )

        print(
            f"rank={rank} disordered_diffusion_mask_smoke=passed",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
