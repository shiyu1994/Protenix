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

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.runtime.audit_train_sampler_coverage import (
    BIT_ORDER,
    DatasetCatalog,
    _effective_sampler_seed,
    _pack_seen_bitmap,
    merge_initial_coverages,
    replay_sampler_counts,
)


class _Dataset:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


class _ClusterSource:
    cluster_count = 2
    first_row_ids = np.asarray([0, 2], dtype=np.int64)
    cluster_sizes = np.asarray([2, 2], dtype=np.int64)

    def resume_signature(self) -> dict[str, object]:
        return {"fixture": "two_clusters_of_two_rows"}


def _catalog(
    name: str,
    size: int,
    sampling_spec,
) -> DatasetCatalog:
    return DatasetCatalog(
        name=name,
        dataset=_Dataset(size),  # type: ignore[arg-type]
        sampling_spec=sampling_spec,
        runtime_index=None,
        random_sample_if_failed=False,
        source_identity={"fixture": name},
    )


class AuditTrainSamplerCoverageTest(unittest.TestCase):
    def test_negative_sampler_seed_inherits_global_seed(self) -> None:
        config = {
            "seed": 42,
            "data": {"train_sampler": {"seed": -1}},
        }
        self.assertEqual(_effective_sampler_seed(config), 42)
        config["data"]["train_sampler"]["seed"] = 17
        self.assertEqual(_effective_sampler_seed(config), 17)

    def test_hierarchical_partial_epoch_accounting(self) -> None:
        source = _ClusterSource()
        catalogs = [
            _catalog("weighted", 3, torch.tensor([1.0, 2.0, 3.0])),
            _catalog(
                "clustered",
                4,
                {"kind": "clustered", "source": source},
            ),
        ]
        counts, replay = replay_sampler_counts(
            catalogs=catalogs,
            masses=[0.4, 0.6],
            updates=5,
            accumulation=1,
            world_size=2,
            epoch_size=6,
            seed=7,
            replacement=True,
        )
        self.assertEqual(sum(int(count.sum()) for count in counts), 10)
        self.assertEqual(replay["full_epochs"], 1)
        self.assertEqual(replay["partial_epoch_occurrences"], 4)
        self.assertEqual(replay["next_sampler_epoch"], 1)
        self.assertEqual(replay["next_local_epoch_offset"], 2)

    def test_seen_bitmap_round_trip(self) -> None:
        counts = np.asarray([0, 3, 0, 1, 9, 0, 0, 2, 0], dtype=np.uint32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seen.bin"
            seen_count = _pack_seen_bitmap(counts, path)
            unpacked = np.unpackbits(
                np.frombuffer(path.read_bytes(), dtype=np.uint8),
                bitorder=BIT_ORDER,
            )[: len(counts)]
        np.testing.assert_array_equal(unpacked.astype(bool), counts > 0)
        self.assertEqual(seen_count, 4)

    def test_merge_initial_coverage_requires_same_source_identity(self) -> None:
        catalog = _catalog("weighted", 3, torch.tensor([1.0, 2.0, 3.0]))
        counts = [np.asarray([0, 2, 0], dtype=np.uint32)]
        with tempfile.TemporaryDirectory() as directory:
            coverage_dir = Path(directory)
            np.savez_compressed(
                coverage_dir / "weighted.seen_rows.npz",
                row_id=np.asarray([0, 2], dtype=np.int64),
                sample_count=np.asarray([4, 1], dtype=np.uint32),
            )
            summary = {
                "event": "planned_train_sampler_coverage",
                "run_dir": "/fixture/prior",
                "updates": 5,
                "datasets": {
                    "weighted": {
                        "catalog_rows": 3,
                        "source_identity": {"fixture": "weighted"},
                        "artifacts": {
                            "seen_rows_npz": "weighted.seen_rows.npz"
                        },
                    }
                },
            }
            (coverage_dir / "summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )
            merged = merge_initial_coverages(
                catalogs=[catalog],
                counts=counts,
                coverage_dirs=[coverage_dir],
            )
        np.testing.assert_array_equal(
            counts[0],
            np.asarray([4, 2, 1], dtype=np.uint32),
        )
        self.assertEqual(merged[0]["sampled_occurrences_added"], 5)


if __name__ == "__main__":
    unittest.main()
