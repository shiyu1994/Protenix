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

"""CPU core for diversity-aware sampling from exposure-count buckets.

This module deliberately has no dependency on PyTorch or the training data
loader.  It implements the stateful part of the soft exposure policy:

    P(bucket=c) proportional to N_c * gamma**c

and samples uniformly within the chosen bucket.  A selected cluster is then
promoted from exposure bucket ``c`` to bucket ``c + 1``.

The implementation stores each bucket as a contiguous segment of one dense
cluster array.  Promotion is a swap with the end of the selected segment, so
both sampling and promotion are O(1) apart from the very small bucket lookup.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


_STATE_FORMAT = "protenix.soft_exposure_bucket_sampler.v1"
_MAX_EXPOSURE_COUNT = int(np.iinfo(np.uint32).max)


@dataclass(frozen=True)
class SoftExposureDraw:
    """One cluster selection and its exposure count before promotion."""

    cluster_ordinal: int
    previous_exposure_count: int

    @property
    def new_exposure_count(self) -> int:
        """Return the exposure count after this selection."""

        return self.previous_exposure_count + 1


def rotating_member_ordinal(
    *,
    cluster_ordinal: int,
    previous_exposure_count: int,
    member_count: int,
    seed: int = 0,
) -> int:
    """Choose members without repetition until one cluster cycle is complete.

    Each cluster gets a deterministic pseudo-random starting member.  Further
    visits rotate through all members before returning to that starting point.
    The helper is stateless because ``previous_exposure_count`` is already part
    of :class:`SoftExposureDraw` and of the sampler checkpoint.
    """

    if cluster_ordinal < 0:
        raise ValueError("cluster_ordinal must be non-negative")
    if previous_exposure_count < 0:
        raise ValueError("previous_exposure_count must be non-negative")
    if member_count <= 0:
        raise ValueError("member_count must be positive")

    # SplitMix64 gives a process-independent offset.  In particular, this does
    # not rely on Python's randomized hash seed.
    mask = (1 << 64) - 1
    value = (int(cluster_ordinal) ^ int(seed)) & mask
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    start = value % int(member_count)
    return int((start + int(previous_exposure_count)) % int(member_count))


class SoftExposureBucketSampler:
    """Stateful soft exposure sampler implemented entirely on CPU.

    Args:
        exposure_counts: One non-negative integer count per cluster.  Unsigned
            NumPy arrays are accepted directly; values are stored as uint32.
        gamma: Per-visit decay in ``(0, 1]``.  The training default is 0.2.
        seed: Seed for NumPy's PCG64 generator.

    The class is intentionally not thread-safe.  A distributed caller should
    own one authoritative global instance and partition its resulting stream.
    """

    def __init__(
        self,
        exposure_counts: Sequence[int] | np.ndarray,
        *,
        gamma: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.gamma = self._validate_gamma(gamma)
        self.seed = int(seed)
        self._counts = self._normalize_counts(exposure_counts)
        if self._counts.size == 0:
            raise ValueError("exposure_counts must contain at least one cluster")

        self._num_clusters = int(self._counts.size)
        self._index_dtype = (
            np.uint32
            if self._num_clusters <= int(np.iinfo(np.uint32).max)
            else np.uint64
        )
        # Stable sorting makes construction deterministic for equal-count
        # clusters and gives every bucket one contiguous range.
        self._cluster_at_position = np.argsort(
            self._counts, kind="stable"
        ).astype(self._index_dtype, copy=False)
        self._position_of_cluster = np.empty(
            self._num_clusters, dtype=self._index_dtype
        )
        self._position_of_cluster[self._cluster_at_position] = np.arange(
            self._num_clusters, dtype=self._index_dtype
        )
        self._rebuild_bucket_layout()
        self._rng = np.random.default_rng(self.seed)
        self._num_draws = 0

    @staticmethod
    def _validate_gamma(gamma: float) -> float:
        gamma = float(gamma)
        if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be finite and in (0, 1], got {gamma!r}")
        return gamma

    @staticmethod
    def _normalize_counts(
        exposure_counts: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(exposure_counts)
        if values.ndim != 1:
            raise ValueError(
                f"exposure_counts must be one-dimensional, got shape {values.shape}"
            )
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError(
                "exposure_counts must have an integer dtype, "
                f"got {values.dtype}"
            )
        if np.issubdtype(values.dtype, np.signedinteger) and np.any(values < 0):
            raise ValueError("exposure_counts must be non-negative")
        if values.size and int(values.max()) > _MAX_EXPOSURE_COUNT:
            raise OverflowError(
                f"exposure count exceeds uint32 maximum {_MAX_EXPOSURE_COUNT}"
            )
        # The sampler owns its mutable counts.  Always copy even when the input
        # is already contiguous uint32, otherwise a draw would mutate the
        # caller's checkpoint/audit array in place.
        return np.array(values, dtype=np.uint32, order="C", copy=True)

    def _rebuild_bucket_layout(self) -> None:
        max_count = int(self._counts.max())
        integer_counts = self._counts.astype(np.int64, copy=False)
        self._bucket_sizes = np.bincount(
            integer_counts, minlength=max_count + 1
        ).astype(np.int64, copy=False)
        self._bucket_starts = np.zeros(len(self._bucket_sizes), dtype=np.int64)
        if len(self._bucket_sizes) > 1:
            np.cumsum(
                self._bucket_sizes[:-1], out=self._bucket_starts[1:]
            )
        self._rebuild_bucket_weights()

    def _rebuild_bucket_weights(self) -> None:
        nonempty = np.flatnonzero(self._bucket_sizes)
        if nonempty.size == 0:
            raise RuntimeError("soft exposure sampler has no non-empty bucket")
        self._weight_base_count = int(nonempty[0])
        levels = np.arange(len(self._bucket_sizes), dtype=np.float64)
        per_cluster_weights = np.power(
            self.gamma, levels - float(self._weight_base_count)
        )
        self._bucket_weights = self._bucket_sizes * per_cluster_weights
        self._bucket_cumulative_weights = np.cumsum(self._bucket_weights)

    @property
    def num_clusters(self) -> int:
        """Number of clusters represented by this sampler."""

        return self._num_clusters

    @property
    def num_draws(self) -> int:
        """Selections performed since construction or the loaded checkpoint."""

        return self._num_draws

    @property
    def exposure_counts(self) -> np.ndarray:
        """Return a defensive copy of current uint32 exposure counts."""

        return self._counts.copy()

    @property
    def bucket_sizes(self) -> np.ndarray:
        """Return bucket populations indexed by exposure count."""

        return self._bucket_sizes.copy()

    def bucket_probabilities(self) -> np.ndarray:
        """Return current bucket probabilities indexed by exposure count."""

        total = float(self._bucket_cumulative_weights[-1])
        if not total > 0.0 or not math.isfinite(total):
            raise RuntimeError(f"invalid aggregate bucket weight: {total!r}")
        return self._bucket_weights / total

    def count_for(self, cluster_ordinal: int) -> int:
        """Return the current exposure count for one cluster."""

        cluster_ordinal = int(cluster_ordinal)
        if not 0 <= cluster_ordinal < self._num_clusters:
            raise IndexError(
                f"cluster ordinal {cluster_ordinal} is outside "
                f"[0, {self._num_clusters})"
            )
        return int(self._counts[cluster_ordinal])

    def draw(self) -> SoftExposureDraw:
        """Select one cluster and promote it to the next exposure bucket."""

        total_weight = float(self._bucket_cumulative_weights[-1])
        threshold = float(self._rng.random()) * total_weight
        exposure_count = int(
            np.searchsorted(
                self._bucket_cumulative_weights, threshold, side="right"
            )
        )
        bucket_size = int(self._bucket_sizes[exposure_count])
        if bucket_size <= 0:
            raise RuntimeError(
                f"weighted lookup selected empty exposure bucket {exposure_count}"
            )
        start = int(self._bucket_starts[exposure_count])
        position = int(self._rng.integers(start, start + bucket_size))
        cluster_ordinal = int(self._cluster_at_position[position])
        self._promote(
            cluster_ordinal=cluster_ordinal,
            position=position,
            exposure_count=exposure_count,
        )
        self._num_draws += 1
        return SoftExposureDraw(
            cluster_ordinal=cluster_ordinal,
            previous_exposure_count=exposure_count,
        )

    def draw_many(self, count: int) -> list[SoftExposureDraw]:
        """Draw ``count`` clusters in the same order as repeated ``draw`` calls."""

        count = int(count)
        if count < 0:
            raise ValueError("draw count must be non-negative")
        return [self.draw() for _ in range(count)]

    def _promote(
        self,
        *,
        cluster_ordinal: int,
        position: int,
        exposure_count: int,
    ) -> None:
        if exposure_count == _MAX_EXPOSURE_COUNT:
            raise OverflowError("uint32 exposure count cannot be incremented")

        boundary_position = (
            int(self._bucket_starts[exposure_count])
            + int(self._bucket_sizes[exposure_count])
            - 1
        )
        if position != boundary_position:
            boundary_cluster = int(
                self._cluster_at_position[boundary_position]
            )
            self._cluster_at_position[position] = boundary_cluster
            self._position_of_cluster[boundary_cluster] = position
            self._cluster_at_position[boundary_position] = cluster_ordinal
            self._position_of_cluster[cluster_ordinal] = boundary_position

        next_count = exposure_count + 1
        if next_count == len(self._bucket_sizes):
            self._bucket_sizes = np.append(self._bucket_sizes, np.int64(0))
            self._bucket_starts = np.append(
                self._bucket_starts, np.int64(self._num_clusters)
            )
            # New levels are rare.  Rebuilding here is simpler and avoids
            # special-casing extension of all derived weight arrays.
            self._bucket_sizes[exposure_count] -= 1
            self._bucket_sizes[next_count] += 1
            self._bucket_starts[next_count] -= 1
            self._counts[cluster_ordinal] = next_count
            self._rebuild_bucket_weights()
            return

        old_per_cluster_weight = self.gamma ** (
            exposure_count - self._weight_base_count
        )
        new_per_cluster_weight = self.gamma ** (
            next_count - self._weight_base_count
        )
        self._bucket_sizes[exposure_count] -= 1
        self._bucket_sizes[next_count] += 1
        self._bucket_starts[next_count] -= 1
        self._counts[cluster_ordinal] = next_count

        if self._bucket_sizes[self._weight_base_count] == 0:
            # Rescaling at a completed minimum-exposure cycle prevents weights
            # from underflowing after many full passes over the catalog.
            self._rebuild_bucket_weights()
            return

        self._bucket_weights[exposure_count] -= old_per_cluster_weight
        self._bucket_weights[next_count] += new_per_cluster_weight
        self._bucket_cumulative_weights[exposure_count:] -= (
            old_per_cluster_weight
        )
        self._bucket_cumulative_weights[next_count:] += new_per_cluster_weight

    def state_dict(self) -> dict[str, Any]:
        """Return all mutable state required for exact stream continuation."""

        return {
            "format": _STATE_FORMAT,
            "gamma": self.gamma,
            "seed": self.seed,
            "num_clusters": self._num_clusters,
            "num_draws": self._num_draws,
            "exposure_counts": self._counts.copy(),
            "cluster_at_position": self._cluster_at_position.copy(),
            "bucket_weights": self._bucket_weights.copy(),
            "bucket_cumulative_weights": self._bucket_cumulative_weights.copy(),
            "weight_base_count": self._weight_base_count,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(
        self, state_dict: Mapping[str, Any], *, strict: bool = True
    ) -> None:
        """Restore a checkpoint produced by :meth:`state_dict`.

        In strict mode the receiving sampler must have the same gamma and
        cluster count.  Exact within-bucket ordering and RNG state are restored,
        so subsequent draws match uninterrupted execution.
        """

        if state_dict.get("format") != _STATE_FORMAT:
            raise ValueError(
                "unsupported soft exposure sampler state format: "
                f"{state_dict.get('format')!r}"
            )
        loaded_gamma = self._validate_gamma(float(state_dict["gamma"]))
        loaded_num_clusters = int(state_dict["num_clusters"])
        if strict and loaded_gamma != self.gamma:
            raise ValueError(
                f"gamma mismatch: checkpoint={loaded_gamma}, runtime={self.gamma}"
            )
        if loaded_num_clusters != self._num_clusters:
            raise ValueError(
                "cluster count mismatch: "
                f"checkpoint={loaded_num_clusters}, runtime={self._num_clusters}"
            )

        counts = self._normalize_counts(state_dict["exposure_counts"])
        if counts.size != self._num_clusters:
            raise ValueError(
                "checkpoint exposure_counts length does not match num_clusters"
            )
        order = np.asarray(state_dict["cluster_at_position"])
        if order.shape != (self._num_clusters,) or not np.issubdtype(
            order.dtype, np.integer
        ):
            raise ValueError(
                "checkpoint cluster_at_position must be a one-dimensional "
                "integer permutation"
            )
        order = np.ascontiguousarray(order, dtype=self._index_dtype)
        if not np.array_equal(
            np.sort(order), np.arange(self._num_clusters, dtype=self._index_dtype)
        ):
            raise ValueError("checkpoint cluster_at_position is not a permutation")
        ordered_counts = counts[order]
        if ordered_counts.size > 1 and np.any(
            ordered_counts[1:] < ordered_counts[:-1]
        ):
            raise ValueError(
                "checkpoint cluster_at_position is inconsistent with exposure_counts"
            )

        self.gamma = loaded_gamma
        self.seed = int(state_dict["seed"])
        self._counts = counts
        self._cluster_at_position = order
        self._position_of_cluster = np.empty(
            self._num_clusters, dtype=self._index_dtype
        )
        self._position_of_cluster[order] = np.arange(
            self._num_clusters, dtype=self._index_dtype
        )
        self._rebuild_bucket_layout()

        loaded_weights = np.asarray(state_dict["bucket_weights"], dtype=np.float64)
        loaded_cumulative = np.asarray(
            state_dict["bucket_cumulative_weights"], dtype=np.float64
        )
        if loaded_weights.shape != self._bucket_weights.shape or loaded_cumulative.shape != (
            len(self._bucket_sizes),
        ):
            raise ValueError("checkpoint bucket weight arrays have invalid shapes")
        if int(state_dict["weight_base_count"]) != self._weight_base_count:
            raise ValueError("checkpoint weight base is inconsistent with counts")
        # Preserve accumulated floating-point state for bit-for-bit continuation.
        self._bucket_weights = loaded_weights.copy()
        self._bucket_cumulative_weights = loaded_cumulative.copy()
        if not np.all(np.isfinite(self._bucket_cumulative_weights)) or not float(
            self._bucket_cumulative_weights[-1]
        ) > 0.0:
            raise ValueError("checkpoint contains invalid bucket weights")

        num_draws = int(state_dict["num_draws"])
        if num_draws < 0:
            raise ValueError("checkpoint num_draws must be non-negative")
        self._num_draws = num_draws
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = copy.deepcopy(state_dict["rng_state"])
