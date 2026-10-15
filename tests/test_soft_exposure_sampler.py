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

import numpy as np
import pytest

from protenix.data.pipeline.soft_exposure_sampler import (
    SoftExposureBucketSampler,
    rotating_member_ordinal,
)


def _draw_signature(sampler: SoftExposureBucketSampler, count: int):
    return [
        (draw.cluster_ordinal, draw.previous_exposure_count)
        for draw in sampler.draw_many(count)
    ]


def test_bucket_probability_is_population_times_gamma_power():
    sampler = SoftExposureBucketSampler(
        np.asarray([0, 0, 1, 1, 1, 2], dtype=np.uint16),
        gamma=0.2,
        seed=7,
    )

    # Raw bucket masses are [2 * 1, 3 * 0.2, 1 * 0.04].
    expected = np.asarray([2.0, 0.6, 0.04])
    expected /= expected.sum()
    np.testing.assert_allclose(sampler.bucket_probabilities(), expected)
    np.testing.assert_array_equal(sampler.bucket_sizes, [2, 3, 1])
    assert sampler.exposure_counts.dtype == np.uint32


def test_draw_promotes_exactly_one_cluster_and_preserves_partition():
    initial = np.asarray([0, 0, 1, 3], dtype=np.uint32)
    sampler = SoftExposureBucketSampler(initial, gamma=0.2, seed=9)

    draw = sampler.draw()

    expected = initial.copy()
    expected[draw.cluster_ordinal] += 1
    np.testing.assert_array_equal(sampler.exposure_counts, expected)
    assert draw.previous_exposure_count == int(initial[draw.cluster_ordinal])
    assert draw.new_exposure_count == sampler.count_for(draw.cluster_ordinal)
    assert int(sampler.bucket_sizes.sum()) == len(initial)
    assert sampler.num_draws == 1


def test_equal_seed_and_state_produce_identical_adaptive_streams():
    counts = np.asarray([0] * 20 + [1] * 7 + [2] * 3, dtype=np.uint8)
    left = SoftExposureBucketSampler(counts, gamma=0.2, seed=12345)
    right = SoftExposureBucketSampler(counts, gamma=0.2, seed=12345)

    assert _draw_signature(left, 500) == _draw_signature(right, 500)
    np.testing.assert_array_equal(left.exposure_counts, right.exposure_counts)


def test_state_dict_resumes_exact_order_and_rng_stream():
    counts = np.asarray([0] * 31 + [1] * 5, dtype=np.uint32)
    uninterrupted = SoftExposureBucketSampler(counts, gamma=0.2, seed=44)
    uninterrupted.draw_many(73)
    state = uninterrupted.state_dict()
    expected_tail = _draw_signature(uninterrupted, 300)

    resumed = SoftExposureBucketSampler(
        np.zeros_like(counts), gamma=0.2, seed=999
    )
    resumed.load_state_dict(state)

    assert resumed.num_draws == 73
    assert _draw_signature(resumed, 300) == expected_tail
    np.testing.assert_array_equal(
        resumed.exposure_counts, uninterrupted.exposure_counts
    )


def test_load_rejects_semantic_or_partition_mismatch():
    source = SoftExposureBucketSampler(
        np.zeros(5, dtype=np.uint8), gamma=0.2, seed=1
    )
    state = source.state_dict()

    wrong_gamma = SoftExposureBucketSampler(
        np.zeros(5, dtype=np.uint8), gamma=0.1, seed=1
    )
    with pytest.raises(ValueError, match="gamma mismatch"):
        wrong_gamma.load_state_dict(state)

    corrupted = source.state_dict()
    corrupted["cluster_at_position"][[0, 1]] = 0
    with pytest.raises(ValueError, match="not a permutation"):
        source.load_state_dict(corrupted)


def test_member_rotation_uses_every_member_before_repeating():
    selections = [
        rotating_member_ordinal(
            cluster_ordinal=81,
            previous_exposure_count=exposure,
            member_count=4,
            seed=2026,
        )
        for exposure in range(8)
    ]

    assert sorted(selections[:4]) == [0, 1, 2, 3]
    assert selections[4:] == selections[:4]
    assert rotating_member_ordinal(
        cluster_ordinal=81,
        previous_exposure_count=0,
        member_count=4,
        seed=2026,
    ) == selections[0]


@pytest.mark.parametrize("gamma", [0.0, -0.1, 1.01, float("nan")])
def test_invalid_gamma_is_rejected(gamma):
    with pytest.raises(ValueError, match="gamma"):
        SoftExposureBucketSampler(np.zeros(2, dtype=np.uint8), gamma=gamma)


def test_invalid_counts_and_member_arguments_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        SoftExposureBucketSampler(np.asarray([], dtype=np.uint8))
    with pytest.raises(ValueError, match="non-negative"):
        SoftExposureBucketSampler(np.asarray([0, -1], dtype=np.int32))
    with pytest.raises(TypeError, match="integer dtype"):
        SoftExposureBucketSampler(np.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match="member_count"):
        rotating_member_ordinal(
            cluster_ordinal=0,
            previous_exposure_count=0,
            member_count=0,
        )
