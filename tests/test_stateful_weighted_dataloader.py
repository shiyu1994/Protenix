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

from __future__ import annotations

import os

from ml_collections import ConfigDict
import pytest
import torch

from protenix.data.pipeline.dataloader import (
    DistributedDataLoader,
    DistributedWeightedSampler,
    IterDataLoader,
    WeightedSampler,
    _training_loader_worker_kwargs,
)


class _RangeDataset:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


class _WorkerPidDataset(_RangeDataset):
    def __getitem__(self, index: int) -> tuple[int, int]:
        return index, os.getpid()


def _single_rank_loader(
    *,
    seed: int = 17,
    num_samples: int = 10,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> IterDataLoader:
    dataset = _RangeDataset(20)
    sampler = WeightedSampler(
        torch.ones(len(dataset)),
        num_samples=num_samples,
        replacement=True,
        seed=seed,
    )
    return IterDataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=False,
    )


def _take(loader, count: int) -> list[int]:
    values: list[int] = []
    while len(values) < count:
        for batch in loader:
            values.append(int(batch.item()))
            if len(values) == count:
                return values
    return values


def test_split_resume_matches_uninterrupted_weighted_sample_stream():
    uninterrupted = _take(_single_rank_loader(), 17)

    first_loader = _single_rank_loader()
    first_part = _take(first_loader, 7)
    saved_state = first_loader.state_dict()
    assert saved_state["sampler_epoch"] == 0
    assert saved_state["next_local_offset"] == 7
    assert saved_state["consumed_global_batches"] == 7

    resumed_loader = _single_rank_loader()
    resumed_loader.load_state_dict(saved_state)
    second_part = _take(resumed_loader, 10)

    assert first_part + second_part == uninterrupted
    assert resumed_loader.state_dict()["consumed_global_batches"] == 17


def test_iterator_creation_and_worker_prefetch_do_not_advance_logical_cursor():
    loader = _single_rank_loader(
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
    )
    iterator = iter(loader)
    assert loader.state_dict()["consumed_global_batches"] == 0

    next(iterator)
    state = loader.state_dict()
    assert state["consumed_global_batches"] == 1
    assert state["next_local_offset"] == 1
    del iterator


def test_persistent_workers_are_reused_across_sampler_epoch_boundary():
    dataset = _WorkerPidDataset(8)
    # Exercise the production multi-rank loader constructor (with a one-rank
    # sampler here so the test does not require a process group).
    sampler = DistributedWeightedSampler(
        dataset,
        torch.ones(len(dataset)),
        num_samples=4,
        num_replicas=1,
        rank=0,
        replacement=True,
        seed=17,
    )
    loader = DistributedDataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=False,
    )
    try:
        first_epoch_pids = {int(batch[1].item()) for batch in loader}
        second_epoch_pids = {int(batch[1].item()) for batch in loader}

        assert first_epoch_pids
        assert first_epoch_pids == second_epoch_pids
        assert os.getpid() not in first_epoch_pids
        assert loader.state_dict()["sampler_epoch"] == 2
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_training_worker_profile_defaults_and_num_workers_zero_fallback():
    worker_profile = _training_loader_worker_kwargs(
        ConfigDict({"data": {"num_dl_workers": 8}})
    )
    assert worker_profile == {
        "num_workers": 8,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "pin_memory": False,
    }

    synchronous_profile = _training_loader_worker_kwargs(
        ConfigDict(
            {
                "data": {
                    "num_dl_workers": 0,
                    "train_dl_persistent_workers": True,
                    "train_dl_prefetch_factor": 9,
                }
            }
        )
    )
    assert synchronous_profile == {
        "num_workers": 0,
        "persistent_workers": False,
        "prefetch_factor": None,
        "pin_memory": False,
    }
    loader = _single_rank_loader(num_workers=0)
    assert list(loader)


def test_training_worker_profile_rejects_invalid_prefetch_factor():
    configs = ConfigDict(
        {
            "data": {
                "num_dl_workers": 1,
                "train_dl_persistent_workers": True,
                "train_dl_prefetch_factor": 0,
            }
        }
    )
    with pytest.raises(ValueError, match="train_dl_prefetch_factor must be > 0"):
        _training_loader_worker_kwargs(configs)


@pytest.mark.parametrize(
    ("rank", "local_epoch_size", "epoch", "offset"),
    [
        (0, 313, 1, 187),
        (15, 313, 1, 187),
        (16, 312, 1, 188),
        (31, 312, 1, 188),
    ],
)
def test_legacy_10000_over_32_cursor_uses_actual_unpadded_rank_length(
    rank: int,
    local_epoch_size: int,
    epoch: int,
    offset: int,
):
    dataset = _RangeDataset(10000)
    sampler = DistributedWeightedSampler(
        dataset,
        torch.ones(len(dataset)),
        num_samples=10000,
        num_replicas=32,
        rank=rank,
        replacement=True,
        seed=42,
    )
    loader = DistributedDataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=0,
    )

    assert len(sampler) == 313
    assert sampler.local_epoch_size() == local_epoch_size
    state = loader.reconstruct_state(500)
    assert state["sampler"]["layout"] == "legacy_strided_unpadded_v1"
    assert state["sampler_epoch"] == epoch
    assert state["next_local_offset"] == offset


def test_resume_rejects_sampler_configuration_change():
    state = _single_rank_loader(seed=1).state_dict()
    loader = _single_rank_loader(seed=2)

    with pytest.raises(ValueError, match="sampler configuration changed"):
        loader.load_state_dict(state)


def test_resume_state_records_and_validates_training_worker_profile():
    source = _single_rank_loader(
        num_workers=1,
        persistent_workers=True,
        prefetch_factor=2,
    )
    state = source.state_dict()
    assert state["persistent_workers"] is True
    assert state["prefetch_factor"] == 2
    assert state["pin_memory"] is False

    same_profile = _single_rank_loader(
        num_workers=1,
        persistent_workers=True,
        prefetch_factor=2,
    )
    same_profile.load_state_dict(state)

    changed_persistence = _single_rank_loader(
        num_workers=1,
        persistent_workers=False,
        prefetch_factor=2,
    )
    with pytest.raises(ValueError, match="persistent_workers changed"):
        changed_persistence.load_state_dict(state)

    changed_prefetch = _single_rank_loader(
        num_workers=1,
        persistent_workers=True,
        prefetch_factor=4,
    )
    with pytest.raises(ValueError, match="prefetch_factor changed"):
        changed_prefetch.load_state_dict(state)


def test_resume_accepts_legacy_state_without_worker_runtime_fields():
    state = _single_rank_loader().state_dict()
    state.pop("persistent_workers")
    state.pop("prefetch_factor")
    state.pop("pin_memory")

    loader = _single_rank_loader()
    loader.load_state_dict(state)
