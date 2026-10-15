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
"""Two-rank static-DDP regression for real/dummy template alternation.

Run from the repository root with::

    CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$(pwd)" \
      torchrun --standalone --nproc_per_node=2 \
      tests/distributed_dummy_template_static_graph_smoke.py

The four-microstep windows alternate fully masked template features and
real-like feature values in different rank-local orders.  TemplateEmbedder
must participate in every backward, so ``static_graph=True`` remains valid.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from protenix.data.template.template_featurizer import (
    make_fixed_dummy_template_features,
)
from protenix.model.modules.pairformer import TemplateEmbedder
from protenix.utils.torch_utils import dict_to_tensor
from runner.train import gradient_accumulation_sync_context


class TemplateGraphModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.template_embedder = TemplateEmbedder(
            n_blocks=1,
            c=8,
            c_z=16,
            dropout=0.0,
        )
        self.readout = torch.nn.Linear(16, 1, bias=False)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> torch.Tensor:
        template_update = self.template_embedder(
            input_feature_dict=features,
            z=z,
            pair_mask=z.new_ones(z.shape[:-1]),
        )
        return self.readout(z + template_update).sum()


def template_features(
    *,
    device: torch.device,
    real_like: bool,
) -> dict[str, torch.Tensor]:
    features = dict_to_tensor(make_fixed_dummy_template_features(num_res=5))
    features = {name: value.to(device) for name, value in features.items()}
    features["asym_id"] = torch.zeros(5, dtype=torch.long, device=device)
    if real_like:
        features["template_aatype"][0] = 0
        features["template_atom_mask"][0, :, :3] = True
        features["template_pseudo_beta_mask"][0].fill_(1)
        features["template_distogram"][0, :, :, 0].fill_(1)
        features["template_backbone_frame_mask"][0].fill_(1)
        features["template_unit_vector"][0, :, :, 0].fill_(1)
    return features


def backward_window(
    model: DDP,
    *,
    local_rank: int,
    first_global_step: int,
    rank: int,
    force_first_backward_sync: bool,
) -> None:
    device = torch.device("cuda", local_rank)
    real_like_by_rank = (
        (False, True, False, True),
        (True, False, True, False),
    )
    first_backward_sync_pending = force_first_backward_sync
    for offset, real_like in enumerate(real_like_by_rank[rank]):
        force_sync = first_backward_sync_pending
        with gradient_accumulation_sync_context(
            model,
            use_ddp=True,
            global_step=first_global_step + offset,
            iters_to_accumulate=4,
            force_sync=force_sync,
        ):
            features = template_features(
                device=device,
                real_like=real_like,
            )
            z = torch.randn(5, 5, 16, device=device)
            (model(features, z) / 4).backward()
        first_backward_sync_pending = False

    missing = [
        name
        for name, parameter in model.module.named_parameters()
        if parameter.grad is None
    ]
    assert not missing, f"parameters missing gradients: {missing}"


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        assert dist.get_world_size() == 2, "run this smoke with exactly two ranks"
        rank = dist.get_rank()
        model = DDP(
            TemplateGraphModel().to(local_rank),
            device_ids=[local_rank],
            output_device=local_rank,
            static_graph=True,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

        backward_window(
            model,
            local_rank=local_rank,
            first_global_step=0,
            rank=rank,
            force_first_backward_sync=True,
        )
        model.zero_grad(set_to_none=True)
        backward_window(
            model,
            local_rank=local_rank,
            first_global_step=4,
            rank=rank,
            force_first_backward_sync=False,
        )

        dist.barrier()
        print(
            f"rank={rank} dummy_template_static_graph_smoke=passed",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
