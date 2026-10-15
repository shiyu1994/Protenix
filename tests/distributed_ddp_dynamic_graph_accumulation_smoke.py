#!/usr/bin/env python
"""Two-rank regression for dynamic-graph DDP with gradient accumulation.

Run from the repository root with::

    CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$(pwd)" \
      torchrun --standalone --nproc_per_node=2 \
      tests/distributed_ddp_dynamic_graph_accumulation_smoke.py

The branch choice changes both across ranks and across the four microsteps in
one accumulation window.  ``branch_c`` is used only by an earlier no-sync
microstep and is unused by the final synchronized microstep.  The assertions
therefore cover the subtle case needed by the occurrence-level MSA/template
gates, not merely DDP lifecycle completion.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from runner.train import gradient_accumulation_sync_context


class DynamicBranchModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = torch.nn.Parameter(torch.zeros(()))
        self.branch_a = torch.nn.Parameter(torch.zeros(()))
        self.branch_b = torch.nn.Parameter(torch.zeros(()))
        self.branch_c = torch.nn.Parameter(torch.zeros(()))

    def forward(self, value: torch.Tensor, branch: str) -> torch.Tensor:
        output = self.shared * value
        if branch == "a":
            output = output + self.branch_a * value
        elif branch == "b":
            output = output + self.branch_b * value
        elif branch == "c":
            output = output + self.branch_c * value
        else:
            raise ValueError(f"unknown branch: {branch}")
        return output


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        assert dist.get_world_size() == 2, "run this smoke with exactly two ranks"
        rank = dist.get_rank()
        model = DDP(
            DynamicBranchModel().to(local_rank),
            device_ids=[local_rank],
            output_device=local_rank,
            static_graph=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

        values_by_rank = ((1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0))
        branches_by_rank = (("c", "a", "a", "b"), ("c", "b", "b", "a"))
        for global_step, (value, branch) in enumerate(
            zip(values_by_rank[rank], branches_by_rank[rank], strict=True)
        ):
            with gradient_accumulation_sync_context(
                model,
                use_ddp=True,
                global_step=global_step,
                iters_to_accumulate=4,
            ):
                loss = model(
                    torch.tensor(value, device=local_rank),
                    branch,
                )
                (loss / 4).backward()

        expected = {
            "shared": 4.5,
            "branch_a": 1.625,
            "branch_b": 2.125,
            "branch_c": 0.75,
        }
        for name, expected_value in expected.items():
            gradient = getattr(model.module, name).grad
            assert gradient is not None, f"{name} has no gradient"
            expected_tensor = torch.tensor(expected_value, device=local_rank)
            torch.testing.assert_close(
                gradient,
                expected_tensor,
                rtol=0,
                atol=0,
            )
            gathered = [
                torch.zeros_like(gradient) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered, gradient)
            for rank_gradient in gathered:
                torch.testing.assert_close(
                    rank_gradient,
                    expected_tensor,
                    rtol=0,
                    atol=0,
                )

        print(
            f"rank={rank} dynamic_graph_accumulation_smoke=passed",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
