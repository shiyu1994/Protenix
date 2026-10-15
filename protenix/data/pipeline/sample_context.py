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

"""Stable per-occurrence context for resumable training augmentations."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import NamedTuple, Optional, Union


AUGMENTATION_DECISION_VERSION = "sample_occurrence_sha256_v1"
MSA_MODE_POLICY_VERSION = (
    "categorical_query_only_unpaired_shuffle_species_paired_v1"
)
MSA_MODE_VIRTUAL_SEGMENT_POLICY_VERSION = (
    "categorical_query_only_all_protein_segment_shuffle_species_paired_v2"
)
MSA_MODE_CROP_AWARE_QUERY_ONLY_POLICY_VERSION = (
    "categorical_crop_aware_partial_query_only_"
    "all_protein_segment_shuffle_species_paired_v3"
)
MSA_MODE_MULTICHAIN_SHUFFLE_VARIANT_POLICY_VERSION = (
    "categorical_crop_aware_partial_query_only_all_protein_"
    "multichain_segment_or_whole_shuffle_species_paired_v4"
)
MSA_MODE_MULTICHAIN_SHUFFLE_VARIANT_GLOBAL_QUERY_POLICY_VERSION = (
    "categorical_global_query_only_all_protein_"
    "multichain_segment_or_whole_shuffle_species_paired_v4"
)
QUERY_ONLY_CHAIN_SELECTION_STREAM = "partial_query_only_chain_selection"
VIRTUAL_SEGMENT_MSA_SHUFFLE_STREAM = "virtual_segment_msa_shuffle"
MULTICHAIN_MSA_SHUFFLE_VARIANT_STREAM = (
    "multichain_msa_shuffle_variant"
)
MSA_MODE_CONFIGURED = "configured"
MSA_MODE_QUERY_ONLY = "query_only"
MSA_MODE_UNPAIRED_SHUFFLE = "unpaired_shuffle"
MSA_MODE_SPECIES_PAIRED = "species_paired"


class SampleOccurrence(NamedTuple):
    """Dataset index plus its stable position in the sampled training stream."""

    index: int
    sampler_seed: int
    sampler_epoch: int
    global_position: int

    def with_index(self, index: int) -> "SampleOccurrence":
        """Preserve occurrence identity while mapping into a child dataset."""
        return self._replace(index=int(index))


SampleIndex = Union[int, SampleOccurrence]


def split_sample_index(index: SampleIndex) -> tuple[int, Optional[SampleOccurrence]]:
    """Return the ordinary dataset index and optional occurrence metadata."""
    if isinstance(index, SampleOccurrence):
        return int(index.index), index
    return int(index), None


def _occurrence_hash_value(
    occurrence: Optional[SampleOccurrence],
    *,
    stream_name: str,
    algorithm_version: str,
) -> int:
    if occurrence is None:
        raise ValueError(
            "a SampleOccurrence is required for a non-trivial deterministic "
            f"{stream_name!r} decision"
        )
    if not str(algorithm_version).strip():
        raise ValueError("augmentation decision algorithm_version must be non-empty")
    if not str(stream_name).strip():
        raise ValueError("augmentation decision stream_name must be non-empty")

    material = "\0".join(
        (
            str(algorithm_version),
            str(stream_name),
            str(int(occurrence.sampler_seed)),
            str(int(occurrence.sampler_epoch)),
            str(int(occurrence.global_position)),
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def deterministic_occurrence_bernoulli(
    occurrence: Optional[SampleOccurrence],
    *,
    probability: float,
    stream_name: str,
    algorithm_version: str = AUGMENTATION_DECISION_VERSION,
) -> bool:
    """Draw a deterministic Bernoulli decision for one sampled occurrence.

    Different ``stream_name`` values make independent augmentation decisions
    without consuming NumPy, Python, or torch RNG state. Recreating a
    DataLoader after strict resume therefore reproduces the same decision.
    """
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    value = _occurrence_hash_value(
        occurrence,
        stream_name=stream_name,
        algorithm_version=algorithm_version,
    )
    return value < int(probability * (1 << 64))


def deterministic_occurrence_seed(
    occurrence: Optional[SampleOccurrence],
    *,
    stream_name: str,
    algorithm_version: str = AUGMENTATION_DECISION_VERSION,
) -> int:
    """Return a stable unsigned 64-bit seed for one sampled occurrence."""
    return _occurrence_hash_value(
        occurrence,
        stream_name=stream_name,
        algorithm_version=algorithm_version,
    )


def deterministic_occurrence_categorical(
    occurrence: Optional[SampleOccurrence],
    *,
    probabilities: Sequence[tuple[str, float]],
    stream_name: str,
    algorithm_version: str = AUGMENTATION_DECISION_VERSION,
) -> str:
    """Draw one mutually exclusive category for a sampled occurrence."""
    categories = [
        (str(name), float(probability))
        for name, probability in probabilities
    ]
    if not categories:
        raise ValueError("probabilities must contain at least one category")
    names = [name for name, _ in categories]
    if any(not name.strip() for name in names):
        raise ValueError("categorical decision names must be non-empty")
    if len(names) != len(set(names)):
        raise ValueError("categorical decision names must be unique")
    for name, probability in categories:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"probability for {name!r} must be in [0, 1], got {probability}"
            )
    total = sum(probability for _, probability in categories)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"categorical probabilities must sum to 1, got {total}")

    positive = [(name, probability) for name, probability in categories if probability]
    if len(positive) == 1:
        return positive[0][0]

    value = _occurrence_hash_value(
        occurrence,
        stream_name=stream_name,
        algorithm_version=algorithm_version,
    )
    cumulative = 0.0
    last_positive_name = positive[-1][0]
    for name, probability in categories:
        cumulative += probability
        if name == last_positive_name:
            return name
        if value < int(cumulative * (1 << 64)):
            return name
    return last_positive_name
