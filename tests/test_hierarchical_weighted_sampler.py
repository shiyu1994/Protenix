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

import json

import numpy as np
import pytest
from ml_collections.config_dict import ConfigDict

from protenix.data.pipeline.dataloader import (
    DistributedDataLoader,
    DistributedHierarchicalWeightedSampler,
    IterDataLoader,
    WeightedSampler,
    get_dataloaders,
)
from protenix.data.pipeline.dataset import (
    HierarchicalWeightedMultiDataset,
    WeightedMultiDataset,
    get_datasets,
)
from protenix.data.pipeline.sample_context import (
    split_sample_index,
)


class _SizedDataset:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


class _EchoDataset(_SizedDataset):
    def __getitem__(self, index):
        local_index, occurrence = split_sample_index(index)
        return {
            "basic": {
                "local_index": local_index,
                "sampler_epoch": (
                    None if occurrence is None else occurrence.sampler_epoch
                ),
                "global_position": (
                    None if occurrence is None else occurrence.global_position
                ),
            }
        }


class _ClusterSource:
    def __init__(
        self,
        first_row_ids=(0, 2, 5),
        cluster_sizes=(2, 3, 1),
    ) -> None:
        self.first_row_ids = np.asarray(first_row_ids, dtype=np.uint64)
        self.cluster_sizes = np.asarray(cluster_sizes, dtype=np.uint32)
        self.cluster_count = len(self.first_row_ids)


class _MethodClusterSource:
    cluster_count = 2
    sampling_fingerprint = "method-source-v1"

    def __init__(self) -> None:
        self.calls = 0

    def sample_member_rows(self, cluster_ordinals, uniforms):
        assert isinstance(cluster_ordinals, np.ndarray)
        assert isinstance(uniforms, np.ndarray)
        self.calls += 1
        starts = np.asarray([0, 2], dtype=np.int64)[cluster_ordinals]
        sizes = np.asarray([2, 3], dtype=np.int64)[cluster_ordinals]
        return starts + np.floor(uniforms * sizes).astype(np.int64)


def _sampler(
    *,
    rank=0,
    num_replicas=1,
    num_samples=1024,
    seed=17,
    masses=(0.25, 0.75),
    cluster_source=None,
    emit=False,
):
    if cluster_source is None:
        cluster_source = _ClusterSource()
    return DistributedHierarchicalWeightedSampler(
        dataset=_SizedDataset(10),
        dataset_offsets=[0, 4],
        dataset_masses=masses,
        child_sampling_specs=[
            {"kind": "weighted", "weights": [0.0, 1.0, 0.0, 0.0]},
            {"kind": "clustered", "source": cluster_source},
        ],
        num_samples=num_samples,
        num_replicas=num_replicas,
        rank=rank,
        replacement=True,
        seed=seed,
        emit_sample_occurrence=emit,
    )


def _hierarchical_dataset(cluster_source=None):
    if cluster_source is None:
        cluster_source = _ClusterSource()
    return HierarchicalWeightedMultiDataset(
        datasets=[_EchoDataset(4), _EchoDataset(6)],
        dataset_names=["pdb", "distillation"],
        child_sampling_specs=[
            {"kind": "weighted", "weights": [0.0, 1.0, 0.0, 0.0]},
            {"kind": "clustered", "source": cluster_source},
        ],
        dataset_sample_weights=[0.25, 0.75],
    )


def _hierarchical_loader(*, rank, num_replicas, num_samples=17, seed=23):
    dataset = _hierarchical_dataset()
    sampler = DistributedHierarchicalWeightedSampler(
        dataset=dataset,
        dataset_offsets=dataset.hierarchical_dataset_offsets,
        dataset_masses=dataset.hierarchical_dataset_masses,
        child_sampling_specs=dataset.hierarchical_child_sampling_specs,
        num_samples=num_samples,
        num_replicas=num_replicas,
        rank=rank,
        replacement=True,
        seed=seed,
        emit_sample_occurrence=True,
    )
    loader_class = IterDataLoader if num_replicas == 1 else DistributedDataLoader
    return loader_class(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda samples: samples[0],
        sampler=sampler,
    )


def test_outer_distribution_and_child_boundaries():
    indices = list(_sampler(num_samples=30_000, seed=123))
    pdb_count = sum(index < 4 for index in indices)

    assert pdb_count / len(indices) == pytest.approx(0.25, abs=0.015)
    assert all(index == 1 for index in indices if index < 4)
    assert all(4 <= index < 10 for index in indices if index >= 4)


def test_weighted_rows_and_clusters_have_distinct_expected_probabilities():
    sampler = DistributedHierarchicalWeightedSampler(
        dataset=_SizedDataset(12),
        dataset_offsets=[0, 2],
        dataset_masses=[0.4, 0.6],
        child_sampling_specs=[
            {"kind": "weighted", "weights": [1.0, 3.0]},
            {
                "kind": "clustered",
                "source": _ClusterSource(
                    first_row_ids=(0, 1),
                    cluster_sizes=(1, 9),
                ),
            },
        ],
        num_samples=100_000,
        num_replicas=1,
        rank=0,
        replacement=True,
        seed=52,
    )
    counts = np.bincount(list(sampler), minlength=12) / 100_000

    # Weighted PDB rows receive 0.4 * [1/4, 3/4].
    assert counts[0] == pytest.approx(0.1, abs=0.006)
    assert counts[1] == pytest.approx(0.3, abs=0.008)
    # Distillation clusters, not rows, split the 0.6 outer mass equally.
    assert counts[2] == pytest.approx(0.3, abs=0.008)
    for probability in counts[3:]:
        assert probability == pytest.approx(0.3 / 9, abs=0.004)


@pytest.mark.parametrize("num_replicas", [2, 7])
def test_rank_striding_has_no_missing_or_duplicate_positions(num_replicas):
    expected = list(
        _sampler(
            num_samples=103,
            num_replicas=1,
            rank=0,
            seed=99,
            emit=True,
        )
    )
    received = []
    for rank in range(num_replicas):
        received.extend(
            _sampler(
                num_samples=103,
                num_replicas=num_replicas,
                rank=rank,
                seed=99,
                emit=True,
            )
        )

    assert len({item.global_position for item in received}) == 103
    assert sorted(item.global_position for item in received) == list(range(103))
    assert sorted(received, key=lambda item: item.global_position) == expected


def test_seed_epoch_determinism_and_epoch_change():
    left = _sampler(num_samples=256, seed=71)
    right = _sampler(num_samples=256, seed=71)

    assert list(left) == list(right)
    left.set_epoch(8)
    right.set_epoch(8)
    epoch_eight = list(left)
    assert epoch_eight == list(right)

    left.set_epoch(9)
    assert list(left) != epoch_eight


def test_start_offset_and_local_epoch_size():
    sampler = _sampler(
        rank=2,
        num_replicas=6,
        num_samples=29,
        seed=31,
        emit=True,
    )
    complete = list(sampler)

    assert sampler.local_epoch_size() == 5
    sampler.set_start_offset(3)
    assert list(sampler) == complete[3:]
    with pytest.raises(ValueError, match="start offset"):
        sampler.set_start_offset(6)


def test_resume_signature_tracks_masses_offsets_and_child_content():
    baseline = _sampler(num_samples=32).resume_signature()
    changed_masses = _sampler(
        num_samples=32, masses=(0.5, 0.5)
    ).resume_signature()
    changed_clusters = _sampler(
        num_samples=32,
        cluster_source=_ClusterSource(
            first_row_ids=(0, 2, 4),
            cluster_sizes=(2, 2, 2),
        ),
    ).resume_signature()

    assert baseline["dataset_offsets"] == [0, 4]
    assert baseline["dataset_masses"] == [0.25, 0.75]
    assert baseline != changed_masses
    assert (
        baseline["child_fingerprints"][1]["sha256"]
        != changed_clusters["child_fingerprints"][1]["sha256"]
    )
    assert len(json.dumps(baseline)) < 2_000


def test_cluster_sampling_reaches_members_but_never_leaves_child():
    indices = list(
        _sampler(num_samples=20_000, masses=(0.0, 1.0), seed=42)
    )

    assert set(indices) == set(range(4, 10))


def test_duck_typed_sample_member_rows_is_supported():
    source = _MethodClusterSource()
    indices = list(
        _sampler(
            num_samples=1000,
            masses=(0.0, 1.0),
            cluster_source=source,
        )
    )

    assert source.calls == 1
    assert all(4 <= index < 9 for index in indices)


@pytest.mark.parametrize(
    ("rank", "num_replicas", "consumed"),
    [(0, 1, 7), (2, 4, 2)],
)
def test_resumable_loader_cursor_replays_hierarchical_stream(
    rank, num_replicas, consumed
):
    uninterrupted = _hierarchical_loader(
        rank=rank,
        num_replicas=num_replicas,
    )
    iterator = iter(uninterrupted)
    for _ in range(consumed):
        next(iterator)
    state = uninterrupted.state_dict()
    expected_remainder = list(iterator)
    expected_next_epoch = list(uninterrupted)

    resumed = _hierarchical_loader(
        rank=rank,
        num_replicas=num_replicas,
    )
    resumed.load_state_dict(state)

    assert list(resumed) == expected_remainder
    assert list(resumed) == expected_next_epoch


def test_get_dataloaders_selects_hierarchical_only_for_compact_children(monkeypatch):
    configs = ConfigDict(
        {
            "data": {
                "num_dl_workers": 0,
                "epoch_size": 16,
                "train_sampler": {
                    "seed": -1,
                    "replacement": True,
                },
            }
        }
    )
    hierarchical = _hierarchical_dataset()
    monkeypatch.setattr(
        "protenix.data.pipeline.dataloader.get_datasets",
        lambda *_args, **_kwargs: (hierarchical, {}),
    )
    hierarchical_loader, _ = get_dataloaders(
        configs,
        world_size=1,
        seed=37,
    )
    assert isinstance(
        hierarchical_loader.sampler,
        DistributedHierarchicalWeightedSampler,
    )

    weighted = WeightedMultiDataset(
        datasets=[_EchoDataset(2)],
        dataset_names=["pdb"],
        datapoint_weights=[[1.0, 3.0]],
        dataset_sample_weights=[1.0],
    )
    monkeypatch.setattr(
        "protenix.data.pipeline.dataloader.get_datasets",
        lambda *_args, **_kwargs: (weighted, {}),
    )
    weighted_loader, _ = get_dataloaders(
        configs,
        world_size=1,
        seed=37,
    )
    assert isinstance(weighted_loader.sampler, WeightedSampler)
    assert weighted_loader.dataset is weighted


def test_get_datasets_builds_compact_mixture_and_preserves_pdb_only(monkeypatch):
    cluster_source = _ClusterSource()

    class _ConfiguredDataset(_EchoDataset):
        def __init__(self, name, runtime_index_dir=None, **_kwargs):
            super().__init__(6 if runtime_index_dir else 2)
            self.name = name
            self.runtime_distillation_index = (
                cluster_source if runtime_index_dir else None
            )
            self.indices_list = object()

    monkeypatch.setattr(
        "protenix.data.pipeline.dataset.BaseSingleDataset",
        _ConfiguredDataset,
    )
    monkeypatch.setattr(
        "protenix.data.pipeline.dataset.get_msa_featurizer",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "protenix.data.pipeline.dataset.get_template_featurizer",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "protenix.data.pipeline.dataset.get_sample_weights",
        lambda **_kwargs: [1.0, 3.0],
    )
    configs = ConfigDict(
        {
            "data": {
                "train_sets": ["pdb", "distillation"],
                "test_sets": [],
                "train_ref_pos_augment": True,
                "limits": -1,
                "train_sampler": {"train_sample_weights": [0.4, 0.6]},
                "pdb": {
                    "base_info": {},
                    "cropping_configs": {},
                    "sampler_configs": {"sampler_type": "weighted"},
                },
                "distillation": {
                    "base_info": {"runtime_index_dir": "runtime"},
                    "cropping_configs": {},
                    "sampler_configs": {"sampler_type": "clustered"},
                },
            }
        }
    )
    mixed, _ = get_datasets(configs, error_dir=None)

    assert isinstance(mixed, HierarchicalWeightedMultiDataset)
    assert mixed.hierarchical_dataset_offsets == (0, 2, 8)
    assert mixed.hierarchical_dataset_masses == [0.4, 0.6]
    assert mixed.hierarchical_child_sampling_specs[0]["kind"] == "weighted"
    assert mixed.hierarchical_child_sampling_specs[1] == {
        "kind": "clustered",
        "source": cluster_source,
    }

    configs.data.train_sets = ["pdb"]
    configs.data.train_sampler.train_sample_weights = [1.0]
    pdb_only, _ = get_datasets(configs, error_dir=None)
    assert isinstance(pdb_only, WeightedMultiDataset)
    assert not hasattr(pdb_only, "hierarchical_child_sampling_specs")


def test_without_replacement_is_rejected_explicitly():
    with pytest.raises(ValueError, match="replacement=True"):
        DistributedHierarchicalWeightedSampler(
            dataset=_SizedDataset(3),
            dataset_offsets=[0],
            dataset_masses=[1.0],
            child_sampling_specs=[
                {"kind": "weighted", "weights": [1.0, 1.0, 1.0]}
            ],
            num_samples=3,
            num_replicas=1,
            rank=0,
            replacement=False,
        )
