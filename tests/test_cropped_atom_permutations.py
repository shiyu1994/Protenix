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

from protenix.data.core.featurizer import (
    _canonicalize_cropped_residue_permutations,
)


def test_complete_residue_permutations_are_unchanged():
    atom_res_perm = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [2, 2, 3],
            [3, 3, 2],
        ],
        dtype=np.int64,
    )

    actual = _canonicalize_cropped_residue_permutations(atom_res_perm)

    np.testing.assert_array_equal(actual, atom_res_perm)


def test_partial_residue_keeps_closed_columns_and_reindexes_atom_ids():
    # Retained full-residue atom IDs are {1, 2, 4}.  Column 1 swaps 1/2
    # entirely within that set.  Column 2 maps atom 2 to cropped-out atom 3
    # and must be removed as a whole.
    atom_res_perm = np.array(
        [
            [1, 2, 1],
            [2, 1, 3],
            [4, 4, 4],
        ],
        dtype=np.int64,
    )

    actual = _canonicalize_cropped_residue_permutations(atom_res_perm)

    np.testing.assert_array_equal(
        actual,
        np.array(
            [
                [0, 1],
                [1, 0],
                [2, 2],
            ],
            dtype=np.int64,
        ),
    )


def test_partial_residue_suffix_identity_is_reindexed_not_dropped():
    retained_full_indices = np.arange(16, 27, dtype=np.int64)
    atom_res_perm = retained_full_indices[:, None]

    actual = _canonicalize_cropped_residue_permutations(atom_res_perm)

    np.testing.assert_array_equal(
        actual,
        np.arange(len(retained_full_indices), dtype=np.int64)[:, None],
    )


def test_partial_residue_prefix_drops_column_that_leaves_retained_subset():
    # The retained identity happens to be 0..N-1, but atom 1 maps to
    # cropped-out full-residue atom 4 in column 1.  This must not take the
    # complete-residue fast path.
    atom_res_perm = np.array(
        [
            [0, 3],
            [1, 4],
            [2, 2],
            [3, 0],
        ],
        dtype=np.int64,
    )

    actual = _canonicalize_cropped_residue_permutations(atom_res_perm)

    np.testing.assert_array_equal(
        actual,
        np.arange(4, dtype=np.int64)[:, None],
    )


def test_malformed_identity_column_is_rejected():
    atom_res_perm = np.array(
        [
            [4, 4],
            [4, 4],
        ],
        dtype=np.int64,
    )

    try:
        _canonicalize_cropped_residue_permutations(atom_res_perm)
    except ValueError as error:
        assert "distinct original index" in str(error)
    else:
        raise AssertionError("malformed identity permutation was accepted")
