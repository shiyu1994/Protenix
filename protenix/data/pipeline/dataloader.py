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

import hashlib
import json
import math
import os
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
from ml_collections.config_dict import ConfigDict
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from protenix.data.pipeline.dataset import Dataset, get_datasets
from protenix.data.pipeline.sample_context import SampleOccurrence
from protenix.data.pipeline.soft_cluster_schedule import (
    DistributedPrecomputedScheduleSampler,
    SoftClusterScheduleError,
)

from protenix.utils.logger import get_logger
from protenix.utils.torch_utils import collate_fn_first

logger = get_logger(__name__)


_SOFT_CLUSTER_STRICT_RESUME_MIGRATIONS = frozenset(
    {
        "soft_cluster_schedule_manifest_full_state_v1",
        "soft_cluster_catalog_replacement_full_state_v2",
    }
)


def _weights_sha256(weights: torch.Tensor) -> str:
    """Return a stable fingerprint for validating resumed sampler streams."""
    weights = weights.detach().cpu().contiguous()
    return hashlib.sha256(weights.numpy().tobytes()).hexdigest()


def _compact_sequence_sha256(values: Any, count: int) -> str:
    """Hash an array-like object without materializing a large Python list."""
    hasher = hashlib.sha256()
    hasher.update(f"count={count}\0".encode())
    try:
        view = memoryview(values)
        hasher.update(f"format={view.format}\0shape={view.shape}\0".encode())
        byte_view = view.cast("B")
        chunk_size = 8 * 1024 * 1024
        for start in range(0, byte_view.nbytes, chunk_size):
            hasher.update(byte_view[start : start + chunk_size])
        return hasher.hexdigest()
    except (TypeError, ValueError):
        pass

    chunk_size = 1_000_000
    for start in range(0, count, chunk_size):
        chunk = torch.as_tensor(values[start : start + chunk_size]).cpu().contiguous()
        hasher.update(str(chunk.dtype).encode())
        hasher.update(memoryview(chunk.numpy()).cast("B"))
    return hasher.hexdigest()


def _get_spec_value(spec: Any, name: str, default: Any = None) -> Any:
    if isinstance(spec, Mapping):
        return spec.get(name, default)
    return getattr(spec, name, default)


def _take_array_values(values: Any, ordinals: torch.Tensor) -> torch.Tensor:
    """Vectorized indexing for tensors, NumPy arrays/memmaps, and sequences."""
    try:
        selected = values[ordinals.numpy()]
    except (IndexError, TypeError):
        try:
            selected = values[ordinals]
        except (IndexError, TypeError):
            selected = [values[index] for index in ordinals.tolist()]
    return torch.as_tensor(selected, dtype=torch.long)


class WeightedSampler(Sampler):
    """
    A weighted sampler for single node.

    Args:
        weights (list or numpy array): A list or numpy array of weights.
        num_samples (int): The number of samples to be drawn.
        replacement (bool): Whether sampling is done with replacement.
        seed (int): The seed for the random number generator.
    """

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool,
        seed: int = 0,
        emit_sample_occurrence: bool = False,
    ):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement
        self.seed = seed
        self.emit_sample_occurrence = bool(emit_sample_occurrence)
        self.epoch = 0
        self.num_samples = num_samples
        self.start_offset = 0
        self._weights_sha256 = _weights_sha256(self.weights)
        if not self.replacement and self.num_samples > len(self.weights):
            raise ValueError(
                "sampling without replacement requires num_samples <= dataset size"
            )
        if not self.replacement and int((self.weights > 0).sum()) < self.num_samples:
            raise ValueError(
                "sampling without replacement requires enough positive-weight samples"
            )

    def __iter__(self) -> Iterator[int]:
        """
        Generates an iterator over the sampled indices.

        This method uses a random number generator to sample indices based on the provided weights.
        The generator is seeded with the current seed and epoch to ensure reproducibility.

        Returns:
            iter: An iterator over the sampled indices.
        """
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.num_samples, self.replacement, generator=g
        ).tolist()
        indices = indices[self.start_offset :]
        if not self.emit_sample_occurrence:
            return iter(indices)
        positions = range(self.start_offset, self.num_samples)
        return iter(
            SampleOccurrence(
                index=index,
                sampler_seed=self.seed,
                sampler_epoch=self.epoch,
                global_position=position,
            )
            for index, position in zip(indices, positions)
        )

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start_offset(self, offset: int) -> None:
        local_size = self.local_epoch_size()
        if offset < 0 or offset > local_size:
            raise ValueError(
                f"sampler start offset must be in [0, {local_size}], got {offset}"
            )
        self.start_offset = offset

    def local_epoch_size(self) -> int:
        return self.num_samples

    def resume_signature(self) -> dict[str, Any]:
        return {
            "sampler_type": type(self).__name__,
            "layout": "single_rank_weighted_v1",
            "seed": self.seed,
            "num_samples": self.num_samples,
            "replacement": self.replacement,
            "emit_sample_occurrence": self.emit_sample_occurrence,
            "weights_sha256": self._weights_sha256,
        }


class DistributedWeightedSampler(DistributedSampler):
    """
    A distributed weighted sampler for multiple nodes.

    Args:
        dataset (Dataset): The dataset to be loaded.
        weights (list): The weights associated with the dataset.
        num_samples (int): The total number of samples to be drawn.
        num_replicas (int, optional): The number of replicas to use for distributed sampling. Defaults to None.
        rank (int, optional): The rank of the current process in a distributed environment. Defaults to None.
        replacement (bool, optional): Whether to sample with replacement. Defaults to True.
        seed (int, optional): The random seed for reproducibility. Defaults to 0.
    """

    def __init__(
        self,
        dataset: Dataset,
        weights: Sequence[float],
        num_samples: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        replacement: bool = True,
        seed: int = 0,
        emit_sample_occurrence: bool = False,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement
        self.seed = seed
        self.emit_sample_occurrence = bool(emit_sample_occurrence)
        self.epoch = 0
        self.num_samples = num_samples
        self.start_offset = 0
        self._weights_sha256 = _weights_sha256(self.weights)
        if not self.replacement and self.num_samples > len(self.weights):
            raise ValueError(
                "distributed sampling without replacement requires "
                "num_samples <= dataset size"
            )
        if not self.replacement and self.num_samples % self.num_replicas != 0:
            raise ValueError(
                "distributed sampling without replacement requires num_samples "
                "to be divisible by num_replicas"
            )
        if not self.replacement and int((self.weights > 0).sum()) < self.num_samples:
            raise ValueError(
                "distributed sampling without replacement requires enough "
                "positive-weight samples"
            )

        self.num_samples_per_replica = int(
            math.ceil(self.num_samples / self.num_replicas)
        )
        self.total_size = self.num_samples_per_replica * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        """
        Generates an iterator over the sampled indices for the current process in a distributed environment.

        This method uses a random number generator to sample indices based on the provided weights.
        The generator is seeded with the current seed and epoch to ensure reproducibility.
        The sampled indices are then distributed across the replicas according to the rank of the current process.

        Returns:
            iter: An iterator over the sampled indices for the current process.
        """
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.num_samples, self.replacement, generator=g
        ).tolist()
        indices = indices[self.rank : self.total_size : self.num_replicas]
        indices = indices[self.start_offset :]
        if not self.emit_sample_occurrence:
            return iter(indices)
        positions = range(
            self.rank + self.start_offset * self.num_replicas,
            self.num_samples,
            self.num_replicas,
        )
        return iter(
            SampleOccurrence(
                index=index,
                sampler_seed=self.seed,
                sampler_epoch=self.epoch,
                global_position=position,
            )
            for index, position in zip(indices, positions)
        )

    def __len__(self) -> int:
        return self.num_samples_per_replica

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start_offset(self, offset: int) -> None:
        local_size = self.local_epoch_size()
        if offset < 0 or offset > local_size:
            raise ValueError(
                f"sampler start offset must be in [0, {local_size}], got {offset}"
            )
        self.start_offset = offset

    def local_epoch_size(self) -> int:
        """Return this rank's actual unpadded sample count.

        Keep the historical strided/unpadded layout so existing checkpoints
        resume the exact stream they would have consumed before this state was
        checkpointed.  ``__len__`` remains the historical ceil value.
        """
        if self.rank >= self.num_samples:
            return 0
        return 1 + (self.num_samples - 1 - self.rank) // self.num_replicas

    def resume_signature(self) -> dict[str, Any]:
        return {
            "sampler_type": type(self).__name__,
            "layout": "legacy_strided_unpadded_v1",
            "seed": self.seed,
            "num_samples": self.num_samples,
            "replacement": self.replacement,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "emit_sample_occurrence": self.emit_sample_occurrence,
            "weights_sha256": self._weights_sha256,
        }


class DistributedHierarchicalWeightedSampler(DistributedSampler):
    """Sample a dataset mixture without expanding clustered member weights.

    The sampler first draws a child dataset according to ``dataset_masses``.
    A ``weighted`` child then samples its local rows with replacement using a
    conventional one-dimensional weight vector. A ``clustered`` child samples
    clusters uniformly and then samples one member uniformly within each
    selected cluster.

    ``child_sampling_specs`` accepts the following forms:

    * ``{"kind": "weighted", "weights": weights}``
    * ``{"kind": "clustered", "source": source}``
    * a duck-typed clustered source directly

    A clustered source exposes ``cluster_count`` and either
    ``sample_member_rows(cluster_ordinals, uniforms)`` or the two array-like
    attributes ``first_row_ids`` and ``cluster_sizes``. The custom method is
    called with CPU NumPy arrays and must return child-local row ids.
    """

    def __init__(
        self,
        dataset: Dataset,
        dataset_offsets: Sequence[int],
        dataset_masses: Sequence[float],
        child_sampling_specs: Sequence[Any],
        num_samples: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        replacement: bool = True,
        seed: int = 0,
        emit_sample_occurrence: bool = False,
    ):
        if not replacement:
            raise ValueError(
                "hierarchical sampling only supports replacement=True"
            )
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.replacement = True
        self.seed = int(seed)
        self.emit_sample_occurrence = bool(emit_sample_occurrence)
        self.epoch = 0
        self.num_samples = int(num_samples)
        self.start_offset = 0
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")

        self.dataset_masses = torch.as_tensor(dataset_masses, dtype=torch.double)
        child_count = len(child_sampling_specs)
        if self.dataset_masses.ndim != 1 or len(self.dataset_masses) != child_count:
            raise ValueError(
                "dataset_masses and child_sampling_specs must have the same length"
            )
        if not bool(torch.isfinite(self.dataset_masses).all()):
            raise ValueError("dataset_masses must be finite")
        if bool((self.dataset_masses < 0).any()) or not bool(
            (self.dataset_masses > 0).any()
        ):
            raise ValueError(
                "dataset_masses must be non-negative with at least one positive mass"
            )

        offsets = [int(offset) for offset in dataset_offsets]
        if len(offsets) == child_count:
            boundaries = offsets + [len(dataset)]
        elif len(offsets) == child_count + 1:
            boundaries = offsets
        else:
            raise ValueError(
                "dataset_offsets must contain one start per child or all child "
                "boundaries"
            )
        if not boundaries or boundaries[0] != 0 or boundaries[-1] != len(dataset):
            raise ValueError(
                "dataset offsets must span the flattened dataset from 0 to len(dataset)"
            )
        if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
            raise ValueError("every child dataset must have a positive contiguous size")
        self.dataset_offsets = tuple(boundaries[:-1])
        self._child_sizes = tuple(
            right - left for left, right in zip(boundaries, boundaries[1:])
        )
        self._children = tuple(
            self._normalize_child(spec, child_size)
            for spec, child_size in zip(child_sampling_specs, self._child_sizes)
        )

        self.num_samples_per_replica = int(
            math.ceil(self.num_samples / self.num_replicas)
        )
        self.total_size = self.num_samples_per_replica * self.num_replicas

    @staticmethod
    def _explicit_fingerprint(spec: Any, source: Any) -> Optional[str]:
        for owner in (spec, source):
            for name in (
                "child_fingerprint",
                "sampling_fingerprint",
                "fingerprint",
                "resume_signature",
            ):
                value = _get_spec_value(owner, name)
                if value is None:
                    continue
                if callable(value):
                    value = value()
                material = json.dumps(
                    value, sort_keys=True, separators=(",", ":"), default=str
                )
                return hashlib.sha256(material.encode()).hexdigest()
        return None

    def _normalize_child(self, spec: Any, child_size: int) -> dict[str, Any]:
        kind = _get_spec_value(spec, "kind")
        if kind is None:
            kind = _get_spec_value(spec, "type")
        if kind is None:
            kind = (
                "clustered"
                if _get_spec_value(spec, "cluster_count") is not None
                else "weighted"
            )
        kind = str(kind).lower()

        if kind == "weighted":
            weights = _get_spec_value(spec, "weights")
            if weights is None and not isinstance(spec, Mapping):
                weights = spec
            if weights is None:
                raise ValueError("weighted child spec requires weights")
            weights = torch.as_tensor(weights, dtype=torch.double)
            if weights.ndim != 1 or len(weights) != child_size:
                raise ValueError(
                    "weighted child weights must match its flattened dataset size"
                )
            if not bool(torch.isfinite(weights).all()):
                raise ValueError("weighted child weights must be finite")
            if bool((weights < 0).any()) or not bool((weights > 0).any()):
                raise ValueError(
                    "weighted child weights must be non-negative with at least "
                    "one positive weight"
                )
            return {
                "kind": kind,
                "weights": weights,
                "fingerprint": _weights_sha256(weights),
            }

        if kind != "clustered":
            raise ValueError(f"unsupported child sampling kind: {kind!r}")
        source = _get_spec_value(spec, "source")
        if source is None:
            source = _get_spec_value(spec, "catalog")
        if source is None:
            source = spec
        cluster_count = _get_spec_value(source, "cluster_count")
        if callable(cluster_count):
            cluster_count = cluster_count()
        if cluster_count is None:
            raise ValueError("clustered child source requires cluster_count")
        cluster_count = int(cluster_count)
        if cluster_count <= 0:
            raise ValueError("cluster_count must be positive")

        sample_member_rows = _get_spec_value(source, "sample_member_rows")
        first_row_ids = _get_spec_value(source, "first_row_ids")
        cluster_sizes = _get_spec_value(source, "cluster_sizes")
        if not callable(sample_member_rows) and (
            first_row_ids is None or cluster_sizes is None
        ):
            raise ValueError(
                "clustered child requires sample_member_rows() or "
                "first_row_ids/cluster_sizes"
            )
        if first_row_ids is not None and len(first_row_ids) < cluster_count:
            raise ValueError("first_row_ids is shorter than cluster_count")
        if cluster_sizes is not None and len(cluster_sizes) < cluster_count:
            raise ValueError("cluster_sizes is shorter than cluster_count")

        fingerprint = self._explicit_fingerprint(spec, source)
        if fingerprint is None and first_row_ids is not None:
            first_hash = _compact_sequence_sha256(first_row_ids, cluster_count)
            sizes_hash = _compact_sequence_sha256(cluster_sizes, cluster_count)
            fingerprint = hashlib.sha256(
                f"{cluster_count}\0{first_hash}\0{sizes_hash}".encode()
            ).hexdigest()
        if fingerprint is None:
            structural = (
                f"{type(source).__module__}.{type(source).__qualname__}:"
                f"{cluster_count}"
            )
            fingerprint = hashlib.sha256(structural.encode()).hexdigest()
        return {
            "kind": kind,
            "source": source,
            "cluster_count": cluster_count,
            "sample_member_rows": sample_member_rows,
            "first_row_ids": first_row_ids,
            "cluster_sizes": cluster_sizes,
            "fingerprint": fingerprint,
            "child_size": child_size,
        }

    @staticmethod
    def _sample_clustered_child(
        child: Mapping[str, Any],
        count: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        ordinals = torch.randint(
            int(child["cluster_count"]), (count,), generator=generator
        )
        uniforms = torch.rand(count, dtype=torch.double, generator=generator)
        sample_member_rows = child["sample_member_rows"]
        if callable(sample_member_rows):
            rows = torch.as_tensor(
                sample_member_rows(ordinals.numpy(), uniforms.numpy()),
                dtype=torch.long,
            )
        else:
            first_rows = _take_array_values(child["first_row_ids"], ordinals)
            cluster_sizes = _take_array_values(child["cluster_sizes"], ordinals)
            if bool((cluster_sizes <= 0).any()):
                raise ValueError("sampled cluster has no members")
            rows = first_rows + torch.floor(
                uniforms * cluster_sizes.to(torch.double)
            ).to(torch.long)
        if rows.ndim != 1 or len(rows) != count:
            raise ValueError(
                "sample_member_rows() must return one row id per cluster ordinal"
            )
        if bool((rows < 0).any()) or bool((rows >= int(child["child_size"])).any()):
            raise ValueError("clustered child returned a row outside its dataset")
        return rows

    def _global_indices(self) -> list[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        child_ordinals = torch.multinomial(
            self.dataset_masses,
            self.num_samples,
            replacement=True,
            generator=generator,
        )
        global_indices = torch.empty(self.num_samples, dtype=torch.long)
        for child_ordinal, child in enumerate(self._children):
            positions = torch.nonzero(
                child_ordinals == child_ordinal, as_tuple=False
            ).flatten()
            count = len(positions)
            if count == 0:
                continue
            if child["kind"] == "weighted":
                local_rows = torch.multinomial(
                    child["weights"],
                    count,
                    replacement=True,
                    generator=generator,
                )
            else:
                local_rows = self._sample_clustered_child(
                    child, count, generator
                )
            global_indices[positions] = int(
                self.dataset_offsets[child_ordinal]
            ) + local_rows
        return global_indices.tolist()

    def __iter__(self) -> Iterator[int]:
        indices = self._global_indices()
        indices = indices[self.rank : self.total_size : self.num_replicas]
        indices = indices[self.start_offset :]
        if not self.emit_sample_occurrence:
            return iter(indices)
        positions = range(
            self.rank + self.start_offset * self.num_replicas,
            self.num_samples,
            self.num_replicas,
        )
        return iter(
            SampleOccurrence(
                index=index,
                sampler_seed=self.seed,
                sampler_epoch=self.epoch,
                global_position=position,
            )
            for index, position in zip(indices, positions)
        )

    def __len__(self) -> int:
        return self.num_samples_per_replica

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_offset(self, offset: int) -> None:
        local_size = self.local_epoch_size()
        if offset < 0 or offset > local_size:
            raise ValueError(
                f"sampler start offset must be in [0, {local_size}], got {offset}"
            )
        self.start_offset = int(offset)

    def local_epoch_size(self) -> int:
        if self.rank >= self.num_samples:
            return 0
        return 1 + (self.num_samples - 1 - self.rank) // self.num_replicas

    def resume_signature(self) -> dict[str, Any]:
        return {
            "sampler_type": type(self).__name__,
            "layout": "hierarchical_strided_unpadded_v1",
            "seed": self.seed,
            "num_samples": self.num_samples,
            "replacement": self.replacement,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "emit_sample_occurrence": self.emit_sample_occurrence,
            "dataset_offsets": list(self.dataset_offsets),
            "dataset_masses": self.dataset_masses.tolist(),
            "child_fingerprints": [
                {
                    "kind": child["kind"],
                    "size": child_size,
                    "sha256": child["fingerprint"],
                }
                for child, child_size in zip(self._children, self._child_sizes)
            ],
        }


class KeySumBalancedSampler(Sampler):
    """
    This class implements the KeySumBalancedSampler.
    It calls the `get_balanced_assignments` method to distribute the dataset indices across workers based on the key sum.

    Args:
        dataset (Dataset): The dataset to sample from.
        key (str): The key by which data will be balanced (integer value).
        value_scale (float): The multiplier of key value when computing the worker assignment weight
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.
    """

    def __init__(
        self,
        dataset: Dataset,
        key: str,
        value_scale: float = 1.0,
        seed: Optional[int] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        self.dataset = dataset
        self.key = key
        self.value_scale = value_scale
        self.seed = seed
        self.num_replicas = num_replicas or dist.get_world_size()
        self.rank = rank or dist.get_rank()

        # Get indices for this process after balancing by key sum
        worker_assignments = self.get_balanced_assignments()
        self.indices = worker_assignments[self.rank]

    def get_balanced_assignments(self):
        """
        Distribute dataset indices across workers such that the sum of key values
        assigned to each worker is as balanced as possible.
        """
        if self.seed is not None:
            # deterministically shuffle based on seed
            g = torch.Generator()
            g.manual_seed(self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # pad for len(dataset) to self.num_replicas if len(dataset) < self.num_replicas
        while len(indices) < self.num_replicas:
            indices += indices[: (self.num_replicas - len(indices))]

        if isinstance(self.dataset.indices_list, list):
            # e.g. recentPDB test set
            dataset_values = [
                x[self.key].astype(int)[0] for x in self.dataset.indices_list
            ]
        else:
            # e.g. posebuster test set
            dataset_values = self.dataset.indices_list[self.key].astype(int).to_numpy()

        # Sort indices by key value
        key_value_pairs = [(idx, dataset_values[idx]) for idx in indices]
        key_value_pairs.sort(key=lambda x: x[1], reverse=True)

        # Calculate the target number of samples per worker
        num_samples_per_worker = len(self.dataset) // self.num_replicas

        # Initialize containers for worker assignments and their current key sum
        worker_assignments = [[] for _ in range(self.num_replicas)]
        worker_sums = [0] * self.num_replicas
        total_samples = num_samples_per_worker * self.num_replicas

        # Distribute samples using a greedy strategy to balance the key sum
        for idx, key_value in key_value_pairs[:total_samples]:
            # Find the worker with the smallest sum that hasn't exceeded its target sample count
            min_worker = min(
                range(self.num_replicas),
                key=lambda i: (
                    worker_sums[i]
                    if len(worker_assignments[i]) < num_samples_per_worker
                    else float("inf")
                ),
            )
            worker_assignments[min_worker].append(idx)
            worker_sums[min_worker] += key_value**2

        # Fix any discrepancies in the number of samples
        all_indices = [idx for idx, _ in key_value_pairs]

        # Assign remaining samples if the dataset isn't divisible perfectly
        if len(all_indices) > total_samples:
            for i in range(len(all_indices) - total_samples):
                worker_assignments[i % self.num_replicas].append(
                    all_indices[total_samples + i]
                )

        # Return the indices assigned to the current worker
        return worker_assignments

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class _ResumableDataLoaderMixin:
    """Track the next sampler position actually delivered to the trainer."""

    _STATE_VERSION = 1

    def _init_resume_state(self) -> None:
        self._sampler_epoch = 0
        self._next_local_offset = 0
        self._consumed_global_batches = 0
        self._preserve_torch_rng_on_next_iter = False
        # Retain the old public counter for compatibility with diagnostics.
        self.counter = 0

    def _local_epoch_size(self) -> int:
        local_epoch_size = getattr(self.sampler, "local_epoch_size", None)
        size = (
            int(local_epoch_size())
            if callable(local_epoch_size)
            else int(len(self.sampler))
        )
        if size <= 0:
            raise ValueError(
                "resumable training sampler must provide at least one local sample"
            )
        return size

    def _sampler_signature(self) -> dict[str, Any]:
        resume_signature = getattr(self.sampler, "resume_signature", None)
        if not callable(resume_signature):
            raise TypeError(
                "resumable training dataloader requires sampler.resume_signature()"
            )
        return dict(resume_signature())

    def _apply_sampler_cursor(self) -> None:
        self.sampler.set_epoch(self._sampler_epoch)
        set_start_offset = getattr(self.sampler, "set_start_offset", None)
        if callable(set_start_offset):
            set_start_offset(self._next_local_offset)
        elif self._next_local_offset:
            raise TypeError(
                "training sampler cannot resume a non-zero local offset"
            )

    def _set_cursor(
        self,
        *,
        epoch: int,
        next_local_offset: int,
        consumed_global_batches: int,
        preserve_torch_rng: bool,
    ) -> None:
        local_epoch_size = self._local_epoch_size()
        if epoch < 0:
            raise ValueError(f"sampler epoch must be >= 0, got {epoch}")
        if next_local_offset < 0 or next_local_offset >= local_epoch_size:
            raise ValueError(
                "next local sampler offset must be in "
                f"[0, {local_epoch_size}), got {next_local_offset}"
            )
        if consumed_global_batches < 0:
            raise ValueError(
                "consumed_global_batches must be >= 0, got "
                f"{consumed_global_batches}"
            )
        self._sampler_epoch = int(epoch)
        self._next_local_offset = int(next_local_offset)
        self._consumed_global_batches = int(consumed_global_batches)
        self._preserve_torch_rng_on_next_iter = bool(preserve_torch_rng)
        self.counter = self._sampler_epoch
        self._apply_sampler_cursor()

    def _advance_after_yield(self) -> None:
        batch_size = int(self.batch_size or 1)
        local_epoch_size = self._local_epoch_size()
        self._next_local_offset += batch_size
        self._consumed_global_batches += 1
        while self._next_local_offset >= local_epoch_size:
            self._next_local_offset -= local_epoch_size
            self._sampler_epoch += 1

    def _iter_from_cursor(self) -> Iterator[Any]:
        self._apply_sampler_cursor()
        self.counter = self._sampler_epoch + 1

        # Recreating a DataLoader iterator in the middle of an epoch consumes a
        # base-seed draw that an uninterrupted run would not consume. Preserve
        # the main-process torch RNG for this first resumed iterator. Worker RNG
        # streams themselves are not checkpointable by torch DataLoader.
        torch_rng_state = None
        if self._preserve_torch_rng_on_next_iter:
            torch_rng_state = torch.get_rng_state()
        iterator = super().__iter__()
        if torch_rng_state is not None:
            torch.set_rng_state(torch_rng_state)
        self._preserve_torch_rng_on_next_iter = False

        for batch in iterator:
            # Advance before yielding: save_checkpoint runs while this generator
            # is paused at yield, so its state denotes the next batch.
            self._advance_after_yield()
            yield batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self._STATE_VERSION,
            "sampler_epoch": self._sampler_epoch,
            "next_local_offset": self._next_local_offset,
            "consumed_global_batches": self._consumed_global_batches,
            "batch_size": int(self.batch_size or 1),
            "num_workers": int(self.num_workers),
            # These settings do not advance the logical sampler cursor, but
            # retaining them makes a full resume operationally auditable and
            # prevents an unnoticed loader-runtime profile change.
            "persistent_workers": bool(self.persistent_workers),
            "prefetch_factor": self.prefetch_factor,
            "pin_memory": bool(self.pin_memory),
            "sampler": self._sampler_signature(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if state_dict.get("schema_version") != self._STATE_VERSION:
            raise ValueError(
                "unsupported train dataloader state schema_version: "
                f"{state_dict.get('schema_version')!r}"
            )
        saved_batch_size = int(state_dict.get("batch_size", -1))
        current_batch_size = int(self.batch_size or 1)
        if saved_batch_size != current_batch_size:
            raise ValueError(
                "train dataloader batch size changed across resume: "
                f"checkpoint={saved_batch_size}, current={current_batch_size}"
            )
        saved_num_workers = state_dict.get("num_workers")
        if (
            saved_num_workers is not None
            and int(saved_num_workers) != int(self.num_workers)
        ):
            raise ValueError(
                "train dataloader worker count changed across resume: "
                f"checkpoint={saved_num_workers}, current={self.num_workers}"
            )
        saved_persistent_workers = state_dict.get("persistent_workers")
        if (
            saved_persistent_workers is not None
            and bool(saved_persistent_workers) != bool(self.persistent_workers)
        ):
            raise ValueError(
                "train dataloader persistent_workers changed across resume: "
                f"checkpoint={saved_persistent_workers}, "
                f"current={self.persistent_workers}"
            )
        if "prefetch_factor" in state_dict:
            saved_prefetch_factor = state_dict["prefetch_factor"]
            if saved_prefetch_factor != self.prefetch_factor:
                raise ValueError(
                    "train dataloader prefetch_factor changed across resume: "
                    f"checkpoint={saved_prefetch_factor}, "
                    f"current={self.prefetch_factor}"
                )
        saved_pin_memory = state_dict.get("pin_memory")
        if (
            saved_pin_memory is not None
            and bool(saved_pin_memory) != bool(self.pin_memory)
        ):
            raise ValueError(
                "train dataloader pin_memory changed across resume: "
                f"checkpoint={saved_pin_memory}, current={self.pin_memory}"
            )

        epoch = int(state_dict["sampler_epoch"])
        next_local_offset = int(state_dict["next_local_offset"])
        consumed_global_batches = int(state_dict["consumed_global_batches"])
        saved_signature = state_dict.get("sampler")
        current_signature = self._sampler_signature()
        if saved_signature != current_signature:
            validate_resume_signature = getattr(
                self.sampler,
                "validate_resume_signature",
                None,
            )
            migration_mode = os.environ.get(
                "PROTENIX_STRICT_RESUME_CONTRACT_MIGRATION",
                "",
            ).strip()
            allow_soft_cluster_migration = (
                migration_mode in _SOFT_CLUSTER_STRICT_RESUME_MIGRATIONS
            )
            if not callable(validate_resume_signature):
                raise ValueError(
                    "train sampler configuration changed across resume: "
                    f"checkpoint={saved_signature!r}, "
                    f"current={current_signature!r}"
                )
            try:
                validation_mode = validate_resume_signature(
                    saved_signature,
                    consumed_global_batches=consumed_global_batches,
                    allow_legacy_migration=allow_soft_cluster_migration,
                )
            except (SoftClusterScheduleError, TypeError, ValueError) as exc:
                raise ValueError(
                    "train sampler configuration changed across resume and "
                    "the checkpoint-bound migration was rejected: "
                    f"checkpoint={saved_signature!r}, "
                    f"current={current_signature!r}"
                ) from exc
            if validation_mode != "legacy_migration":
                raise ValueError(
                    "train sampler signature validator returned an unexpected "
                    f"mode: {validation_mode!r}"
                )
            logger.warning(
                "accepted the manifest-bound one-time legacy-to-soft-cluster "
                "sampler migration at consumed_global_batches="
                f"{consumed_global_batches}"
            )
        local_epoch_size = self._local_epoch_size()
        consumed_samples = consumed_global_batches * current_batch_size
        expected_epoch, expected_offset = divmod(
            consumed_samples, local_epoch_size
        )
        if (epoch, next_local_offset) != (expected_epoch, expected_offset):
            raise ValueError(
                "inconsistent train dataloader checkpoint cursor: "
                f"saved=(epoch={epoch}, offset={next_local_offset}), "
                f"derived=(epoch={expected_epoch}, offset={expected_offset})"
            )
        self._set_cursor(
            epoch=epoch,
            next_local_offset=next_local_offset,
            consumed_global_batches=consumed_global_batches,
            preserve_torch_rng=next_local_offset > 0,
        )

    def reconstruct_state(self, consumed_global_batches: int) -> dict[str, Any]:
        """Rebuild the historical cursor for a checkpoint without loader state."""
        consumed_global_batches = int(consumed_global_batches)
        if consumed_global_batches < 0:
            raise ValueError("consumed_global_batches must be >= 0")
        batch_size = int(self.batch_size or 1)
        local_epoch_size = self._local_epoch_size()
        epoch, next_local_offset = divmod(
            consumed_global_batches * batch_size,
            local_epoch_size,
        )
        self._set_cursor(
            epoch=epoch,
            next_local_offset=next_local_offset,
            consumed_global_batches=consumed_global_batches,
            preserve_torch_rng=next_local_offset > 0,
        )
        return self.state_dict()


class IterDataLoader(_ResumableDataLoaderMixin, DataLoader):
    """
    Iterative dataloader for single node.
    """

    def __init__(self, *args, **kwargs):
        super(IterDataLoader, self).__init__(*args, **kwargs)
        assert self.sampler is not None
        self._init_resume_state()

    def __iter__(self):
        return self._iter_from_cursor()


class DistributedDataLoader(_ResumableDataLoaderMixin, DataLoader):
    """
    Distributed dataloader for multiple nodes.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int = 0,
        collate_fn=None,
        seed: int = 42,
        drop_last: bool = True,
        shuffle: bool = True,
        sampler: Sampler = None,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        pin_memory: bool = False,
    ):
        if sampler is not None:
            self.sampler = sampler
        else:
            self.sampler = DistributedSampler(
                dataset, shuffle=shuffle, seed=seed, drop_last=drop_last
            )

        super(DistributedDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=self.sampler,
            shuffle=False,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
        )
        self._init_resume_state()

    def __iter__(self):
        return self._iter_from_cursor()


def _training_loader_worker_kwargs(configs: ConfigDict) -> dict[str, Any]:
    """Resolve a PyTorch-compatible worker profile for the training loader."""
    num_workers = int(configs.data.num_dl_workers)
    if num_workers < 0:
        raise ValueError(f"data.num_dl_workers must be >= 0, got {num_workers}")

    # PyTorch rejects persistent_workers=True and a non-None prefetch_factor
    # for a synchronous (num_workers=0) DataLoader.
    if num_workers == 0:
        return {
            "num_workers": 0,
            "persistent_workers": False,
            "prefetch_factor": None,
            "pin_memory": False,
        }

    persistent_workers = bool(
        configs.data.get("train_dl_persistent_workers", True)
    )
    prefetch_factor = int(configs.data.get("train_dl_prefetch_factor", 2))
    if prefetch_factor <= 0:
        raise ValueError(
            "data.train_dl_prefetch_factor must be > 0 when "
            f"data.num_dl_workers > 0, got {prefetch_factor}"
        )
    return {
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        # Data samples are nested structures with CPU-side preprocessing. Keep
        # the established explicit no-pinning policy for this training path.
        "pin_memory": False,
    }


def get_dataloaders(
    configs: ConfigDict, world_size: int, seed: int, error_dir: Optional[str] = None
):
    """
    Generate data loaders for training and testing based on the given configurations and seed.

    Args:
        configs (ConfigDict): An object containing the data configuration information.
        world_size (int): The number of processes in the distributed environment.
        seed (int): The random seed used for data sampling.
        error_dir (str, optional): The directory to store error information. Defaults to None.

    Returns:
        tuple: A tuple containing the training data loader and a dictionary of testing data loaders.

    """
    train_dataset, test_datasets = get_datasets(configs, error_dir)
    train_loader_worker_kwargs = _training_loader_worker_kwargs(configs)
    logger.info(
        "training DataLoader worker profile: "
        f"num_workers={train_loader_worker_kwargs['num_workers']}, "
        "persistent_workers="
        f"{train_loader_worker_kwargs['persistent_workers']}, "
        f"prefetch_factor={train_loader_worker_kwargs['prefetch_factor']}, "
        f"pin_memory={train_loader_worker_kwargs['pin_memory']}"
    )
    configured_sampler_seed = configs.data.train_sampler.get("seed", -1)
    sampler_seed = (
        seed
        if configured_sampler_seed is None or int(configured_sampler_seed) < 0
        else int(configured_sampler_seed)
    )
    soft_cluster_manifest = str(
        configs.data.train_sampler.get(
            "soft_cluster_schedule_manifest",
            "",
        )
        or ""
    ).strip()
    soft_cluster_gamma = float(
        configs.data.train_sampler.get("soft_cluster_gamma", 0.2)
    )
    sampler_rank = dist.get_rank() if world_size > 1 else 0
    hierarchical_specs = getattr(
        train_dataset, "hierarchical_child_sampling_specs", None
    )
    if hierarchical_specs is not None:
        legacy_train_sampler = DistributedHierarchicalWeightedSampler(
            train_dataset,
            dataset_offsets=train_dataset.hierarchical_dataset_offsets,
            dataset_masses=train_dataset.hierarchical_dataset_masses,
            child_sampling_specs=hierarchical_specs,
            num_samples=configs.data.epoch_size,
            num_replicas=world_size,
            rank=sampler_rank,
            replacement=configs.data.train_sampler.get("replacement", True),
            seed=sampler_seed,
            emit_sample_occurrence=True,
        )
        if soft_cluster_manifest:
            train_sampler = DistributedPrecomputedScheduleSampler(
                train_dataset,
                manifest_path=soft_cluster_manifest,
                num_replicas=world_size,
                rank=sampler_rank,
                epoch_size=int(configs.data.epoch_size),
                gamma=soft_cluster_gamma,
                seed=sampler_seed,
                emit_sample_occurrence=True,
                mmap=True,
            )
            # Validate the live target dataset immediately. At checkpoint
            # restore, validate_legacy_resume_signature separately checks the
            # saved source signature and exact migration cursor. These are the
            # same binding for schema v1, but intentionally differ for the
            # schema-v2 catalog replacement.
            train_sampler.validate_target_dataset_signature(
                legacy_train_sampler.resume_signature()
            )
            logger.info(
                "using checkpoint-bound soft-cluster schedule: "
                f"manifest={train_sampler.schedule.info.manifest_path}, "
                f"gamma={train_sampler.schedule.info.gamma}, "
                "origin_consumed_global_batches="
                f"{train_sampler.schedule.info.origin_consumed_global_batches}, "
                "target_consumed_global_batches="
                f"{train_sampler.schedule.info.target_consumed_global_batches}"
            )
        else:
            train_sampler = legacy_train_sampler
    elif world_size > 1:
        if soft_cluster_manifest:
            raise ValueError(
                "soft-cluster schedule requires a hierarchical training "
                "dataset with clustered child sampling specs"
            )
        train_sampler = DistributedWeightedSampler(
            train_dataset,
            train_dataset.merged_datapoint_weights,
            num_samples=configs.data.epoch_size,
            replacement=configs.data.train_sampler.get("replacement", True),
            seed=sampler_seed,
            emit_sample_occurrence=True,
        )
    else:
        if soft_cluster_manifest:
            raise ValueError(
                "soft-cluster schedule requires a hierarchical training "
                "dataset with clustered child sampling specs"
            )
        train_sampler = WeightedSampler(
            weights=train_dataset.merged_datapoint_weights,
            num_samples=configs.data.epoch_size,
            replacement=configs.data.train_sampler.get("replacement", True),
            seed=sampler_seed,
            emit_sample_occurrence=True,
        )

    if world_size > 1:
        train_dl = DistributedDataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn_first,
            sampler=train_sampler,
            **train_loader_worker_kwargs,
        )
    else:
        train_dl = IterDataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn_first,
            sampler=train_sampler,
            **train_loader_worker_kwargs,
        )

    test_dls = {}
    test_dataset_sizes = {}
    for test_name, test_dataset in test_datasets.items():
        test_dataset_sizes[test_name] = len(test_dataset)
        test_sampler = (
            KeySumBalancedSampler(test_dataset, key="num_tokens", seed=configs.seed)
            if world_size > 1
            else None
        )
        test_dls[test_name] = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=configs.data.num_dl_workers,
            sampler=test_sampler,
            collate_fn=collate_fn_first,
        )
    logger.info(
        f"train data size: {len(train_dataset)}, test size: {test_dataset_sizes}"
    )
    return train_dl, test_dls
