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

import csv
import gzip
import json
import threading
import unittest
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from scripts import build_sharded_distillation_bioassembly as builder


def _leaf(
    leaf_id: int,
    count: int,
    *,
    root_id: str = "0",
    relpath: str | None = None,
) -> builder.LeafJob:
    return builder.LeafJob(
        leaf_id=leaf_id,
        root_id=root_id,
        leaf_relpath=relpath or f"leaf-{leaf_id}",
        cif_count=count,
        cif_names_sha256=f"{leaf_id:064x}",
    )


def _build_leaf_plan(
    base: Path,
    *,
    leaf_count: int = 2,
    world_size: int = 2,
) -> tuple[Path, Path, dict]:
    root = base / "source"
    for leaf_index in range(leaf_count):
        leaf_dir = root / f"leaf-{leaf_index}"
        leaf_dir.mkdir(parents=True)
        (leaf_dir / f"protein-{leaf_index}.cif").write_text(
            "data",
            encoding="utf-8",
        )
    plan_dir = base / "plan"
    plan = builder.build_plan(
        [builder.SourceRoot("0", "afdb", str(root))],
        world_size=world_size,
        workers_per_rank=1,
        max_files_per_task=10,
        plan_dir=plan_dir,
    )
    return root, plan_dir, plan


class PlanAlgorithmTest(unittest.TestCase):
    def test_lpt_keeps_each_leaf_on_one_rank(self):
        leaves = [_leaf(index, weight) for index, weight in enumerate((10, 9, 8, 7, 6, 5))]

        assignments = builder._assign_leaf_jobs(leaves, world_size=3)

        self.assertEqual(
            [[leaf.cif_count for leaf in rank] for rank in assignments],
            [[10, 5], [9, 6], [8, 7]],
        )
        self.assertEqual(
            sorted(leaf.leaf_id for rank in assignments for leaf in rank),
            list(range(6)),
        )

    def test_large_leaf_is_split_into_balanced_chunks(self):
        chunks = builder._split_leaf_job(_leaf(7, 50_109), 10_000)

        self.assertEqual(len(chunks), 6)
        self.assertEqual(
            [chunk.size for chunk in chunks],
            [8_352, 8_352, 8_352, 8_351, 8_351, 8_351],
        )
        self.assertEqual(chunks[0].begin, 0)
        self.assertEqual(chunks[-1].end, 50_109)

    def test_rank_order_visits_first_chunks_before_second_chunks(self):
        leaves = [_leaf(0, 25), _leaf(1, 9), _leaf(2, 7)]
        tasks = [
            task
            for leaf in leaves
            for task in builder._split_leaf_job(leaf, 10)
        ]

        ordered = builder._order_rank_tasks(tasks)

        first_later_chunk = next(
            index
            for index, task in enumerate(ordered)
            if task.chunk_index > 0
        )
        self.assertTrue(
            all(task.chunk_index == 0 for task in ordered[:first_later_chunk])
        )
        self.assertEqual(
            {task.leaf_id for task in ordered[:first_later_chunk]},
            {0, 1, 2},
        )

    def test_dynamic_scheduler_uses_an_inactive_leaf_before_tail_lane(self):
        leaf_zero = builder._split_leaf_job(_leaf(0, 20), 10)
        leaf_one = builder._split_leaf_job(_leaf(1, 10), 10)
        queues = {
            0: deque([leaf_zero[1]]),
            1: deque([leaf_one[0]]),
        }

        selected = builder._choose_next_task(
            queues,
            {0: 1, 1: 0},
            tail_max_lanes_per_leaf=0,
        )

        self.assertEqual(selected.leaf_id, 1)
        self.assertEqual(selected.chunk_index, 0)

    def test_tail_lane_can_be_enabled_or_strictly_disabled(self):
        chunks = builder._split_leaf_job(_leaf(0, 20), 10)
        queues = {0: deque([chunks[1]])}

        tail_task = builder._choose_next_task(
            queues,
            {0: 1},
            tail_max_lanes_per_leaf=0,
        )
        strict_task = builder._choose_next_task(
            queues,
            {0: 1},
            tail_max_lanes_per_leaf=1,
        )

        self.assertEqual(tail_task, chunks[1])
        self.assertIsNone(strict_task)

    def test_parallel_inventory_is_deterministic_and_matches_walk_edges(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            root_a = base / "root-a"
            root_z = base / "root-z"
            (root_a / "only").mkdir(parents=True)
            (root_a / "only" / "one.cif").write_text(
                "data",
                encoding="utf-8",
            )
            (root_z / "leaf" / "bioassembly").mkdir(parents=True)
            (root_z / "dir.cif").mkdir()
            (root_z / "bioassembly").mkdir()
            for path in (
                root_z / "root.cif",
                root_z / "leaf" / "a.cif",
                root_z / "leaf" / "z.cif",
                root_z / "dir.cif" / "inside.cif",
                root_z / "bioassembly" / "ignored.cif",
                root_z / "leaf" / "bioassembly" / "ignored.cif",
            ):
                path.write_text("data", encoding="utf-8")
            target = root_z / "target.txt"
            target.write_text("data", encoding="utf-8")
            (root_z / "file-link.cif").symlink_to(target)
            (root_z / "dangling.cif").symlink_to(root_z / "missing")
            (root_z / "leaf-link").symlink_to(
                root_z / "leaf",
                target_is_directory=True,
            )
            (root_z / "dir-link.cif").symlink_to(
                root_z / "leaf",
                target_is_directory=True,
            )

            roots = [
                builder.SourceRoot(
                    "z",
                    "mgnify",
                    str(root_z),
                    expected_cif_count=6,
                ),
                builder.SourceRoot(
                    "a",
                    "afdb",
                    str(root_a),
                    expected_cif_count=1,
                ),
            ]
            serial = builder._inventory_leaves(roots, scan_workers=1)
            parallel = builder._inventory_leaves(roots, scan_workers=8)

            self.assertEqual(parallel, serial)
            self.assertEqual(
                [
                    (leaf.leaf_id, leaf.root_id, leaf.leaf_relpath)
                    for leaf in parallel
                ],
                [
                    (0, "a", "only"),
                    (1, "z", "."),
                    (2, "z", "dir.cif"),
                    (3, "z", "leaf"),
                ],
            )
            expected_names = {
                ("a", "only"): ("one.cif",),
                ("z", "."): (
                    "dangling.cif",
                    "file-link.cif",
                    "root.cif",
                ),
                ("z", "dir.cif"): ("inside.cif",),
                ("z", "leaf"): ("a.cif", "z.cif"),
            }
            self.assertEqual(
                {
                    (leaf.root_id, leaf.leaf_relpath):
                    leaf.cif_names_sha256
                    for leaf in parallel
                },
                {
                    key: builder._hash_cif_names(names)
                    for key, names in expected_names.items()
                },
            )
            self.assertEqual(
                builder._scan_leaf_cif_names(root_z),
                expected_names[("z", ".")],
            )

    def test_scan_workers_bounds_parallel_scandir(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index in range(8):
                leaf_dir = root / f"leaf-{index}"
                leaf_dir.mkdir()
                (leaf_dir / f"{index}.cif").write_text(
                    "data",
                    encoding="utf-8",
                )

            real_scandir = builder.os.scandir
            barrier = threading.Barrier(3)
            state_lock = threading.Lock()
            state = {"selected": 0, "active": 0, "maximum": 0}

            class TrackedScandir:
                def __init__(self, path):
                    self.path = Path(path)
                    self.inner = None
                    self.tracked = False

                def __enter__(self):
                    if self.path.parent == root:
                        with state_lock:
                            state["selected"] += 1
                            self.tracked = state["selected"] <= 3
                            if self.tracked:
                                state["active"] += 1
                                state["maximum"] = max(
                                    state["maximum"],
                                    state["active"],
                                )
                        if self.tracked:
                            barrier.wait(timeout=5)
                    self.inner = real_scandir(self.path)
                    return self.inner.__enter__()

                def __exit__(self, *args):
                    try:
                        return self.inner.__exit__(*args)
                    finally:
                        if self.tracked:
                            with state_lock:
                                state["active"] -= 1

            with mock.patch.object(
                builder.os,
                "scandir",
                side_effect=TrackedScandir,
            ):
                leaves = builder._inventory_leaves(
                    [builder.SourceRoot("0", "afdb", str(root))],
                    scan_workers=3,
                )

            self.assertEqual(len(leaves), 8)
            self.assertEqual(state["maximum"], 3)

    def test_plan_lock_allows_ranks_and_excludes_finalize(self):
        with TemporaryDirectory() as temporary_directory:
            plan_dir = Path(temporary_directory)
            with builder._plan_activity_lock(
                plan_dir,
                exclusive=False,
            ):
                with builder._plan_activity_lock(
                    plan_dir,
                    exclusive=False,
                ):
                    pass
                with self.assertRaisesRegex(
                    RuntimeError,
                    "incompatible active work",
                ):
                    with builder._plan_activity_lock(
                        plan_dir,
                        exclusive=True,
                    ):
                        pass
            with builder._plan_activity_lock(
                plan_dir,
                exclusive=True,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "incompatible active work",
                ):
                    with builder._plan_activity_lock(
                        plan_dir,
                        exclusive=False,
                    ):
                        pass

    def test_plan_keeps_absolute_roots_out_of_rank_jsonl(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            root = base / "source"
            (root / "alpha").mkdir(parents=True)
            (root / "beta").mkdir()
            for name in ("A.cif", "B.cif", "C.cif", "D.cif"):
                (root / "alpha" / name).write_text("data", encoding="utf-8")
            (root / "beta" / "E.cif").write_text("data", encoding="utf-8")
            plan_dir = base / "plan"

            plan = builder.build_plan(
                [builder.SourceRoot("0", "afdb", str(root))],
                world_size=2,
                workers_per_rank=128,
                max_files_per_task=3,
                plan_dir=plan_dir,
                scan_workers=2,
            )

            self.assertEqual(plan["version"], 3)
            self.assertEqual(plan["scan_workers"], 2)
            self.assertEqual(plan["leaf_count"], 2)
            self.assertEqual(plan["cif_count"], 5)
            self.assertEqual(plan["roots"][0]["path"], str(root.resolve()))
            rank_text = "".join(
                path.read_text(encoding="utf-8")
                for path in sorted(plan_dir.glob("rank-*.jsonl"))
            )
            self.assertNotIn(str(root.resolve()), rank_text)
            records = [
                json.loads(line)
                for path in sorted(plan_dir.glob("rank-*.jsonl"))
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertTrue(
                all("root_id" in record for record in records)
            )
            self.assertTrue(
                all("leaf_relpath" in record for record in records)
            )
            owners: dict[int, set[int]] = {}
            for rank, path in enumerate(
                sorted(plan_dir.glob("rank-*.jsonl"))
            ):
                for line in path.read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    owners.setdefault(record["leaf_id"], set()).add(rank)
            self.assertTrue(all(len(ranks) == 1 for ranks in owners.values()))


class PlanIntegrityTest(unittest.TestCase):
    def test_roots_json_expected_count_and_provenance_are_preserved(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            root = base / "source"
            leaf = root / "leaf"
            leaf.mkdir(parents=True)
            for name in ("one.cif", "two.cif"):
                (leaf / name).write_text("data", encoding="utf-8")
            roots_path = base / "roots.json"
            roots_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "roots": [
                            {
                                "root_id": "0",
                                "source_db": "afdb",
                                "source_prefix": "AFQ|",
                                "path": str(root),
                                "cif_files": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = base / "manifest.json"
            manifest_path.write_text(
                json.dumps({"counts": {"inventory_entries": 2}}),
                encoding="utf-8",
            )

            roots = builder._load_roots_json(roots_path)
            plan = builder.build_plan(
                roots,
                world_size=1,
                workers_per_rank=1,
                max_files_per_task=10,
                plan_dir=base / "plan",
                roots_json_path=roots_path,
            )

            self.assertEqual(roots[0].source_prefix, "AFQ|")
            self.assertEqual(roots[0].expected_cif_count, 2)
            self.assertEqual(plan["roots"][0]["source_prefix"], "AFQ|")
            self.assertEqual(plan["roots"][0]["expected_cif_count"], 2)
            self.assertEqual(plan["root_cif_counts"], {"0": 2})
            self.assertEqual(
                plan["input_provenance"]["roots_json"]["sha256"],
                builder._sha256_file(roots_path),
            )
            self.assertEqual(
                plan["input_provenance"]["catalog_manifest"]["sha256"],
                builder._sha256_file(manifest_path),
            )
            rank_path = base / "plan" / "rank-000.jsonl"
            self.assertEqual(
                plan["ranks"][0]["rank_plan_sha256"],
                builder._sha256_file(rank_path),
            )

    def test_authoritative_root_count_mismatch_fails_plan(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            root = base / "source"
            root.mkdir()
            (root / "only.cif").write_text("data", encoding="utf-8")
            source = builder.SourceRoot(
                "0",
                "afdb",
                str(root),
                source_prefix="AFQ|",
                expected_cif_count=2,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "does not match roots.json",
            ):
                builder.build_plan(
                    [source],
                    world_size=1,
                    workers_per_rank=1,
                    max_files_per_task=10,
                    plan_dir=base / "plan",
                )

    def test_catalog_manifest_count_mismatch_does_not_publish_rank_plans(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            root = base / "source"
            root.mkdir()
            (root / "only.cif").write_text("data", encoding="utf-8")
            manifest_path = base / "manifest.json"
            manifest_path.write_text(
                json.dumps({"counts": {"inventory_entries": 2}}),
                encoding="utf-8",
            )
            plan_dir = base / "plan"

            with self.assertRaisesRegex(
                RuntimeError,
                "catalog manifest inventory_entries",
            ):
                builder.build_plan(
                    [builder.SourceRoot("0", "afdb", str(root))],
                    world_size=1,
                    workers_per_rank=1,
                    max_files_per_task=10,
                    plan_dir=plan_dir,
                    catalog_manifest_path=manifest_path,
                )

            self.assertEqual(list(plan_dir.glob("rank-*.jsonl")), [])
            self.assertFalse((plan_dir / "plan.json").exists())

    def test_force_replan_is_exclusive_and_clears_finalize_success(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            root, plan_dir, original_plan = _build_leaf_plan(
                base,
                leaf_count=1,
                world_size=1,
            )
            finalize_success = (
                plan_dir / builder.FINALIZE_SUCCESS_NAME
            )
            finalize_success.write_text(
                '{"plan_id":"old"}\n',
                encoding="utf-8",
            )
            source_root = builder.SourceRoot("0", "afdb", str(root))
            with builder._plan_activity_lock(
                plan_dir,
                exclusive=False,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "incompatible active work",
                ):
                    builder.build_plan(
                        [source_root],
                        world_size=1,
                        workers_per_rank=1,
                        max_files_per_task=10,
                        plan_dir=plan_dir,
                        force=True,
                    )
            self.assertTrue(finalize_success.is_file())

            replacement_plan = builder.build_plan(
                [source_root],
                world_size=1,
                workers_per_rank=1,
                max_files_per_task=10,
                plan_dir=plan_dir,
                force=True,
            )
            self.assertNotEqual(
                replacement_plan["plan_id"],
                original_plan["plan_id"],
            )
            self.assertFalse(finalize_success.exists())

    def test_scandir_error_is_fail_closed(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = builder.SourceRoot("0", "afdb", str(root))

            with mock.patch.object(
                builder.os,
                "scandir",
                side_effect=OSError(5, "simulated I/O error", str(root)),
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "Failed to scan source root 0",
                ):
                    builder._inventory_leaves([source], scan_workers=4)

    def test_rank_json_sha_detects_removed_task(self):
        with TemporaryDirectory() as temporary_directory:
            _, plan_dir, plan = _build_leaf_plan(
                Path(temporary_directory),
                leaf_count=2,
                world_size=1,
            )
            rank_path = plan_dir / "rank-000.jsonl"
            rank_path.write_text(
                rank_path.read_text(encoding="utf-8").splitlines()[0] + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                builder._load_and_validate_all_rank_tasks(
                    plan_dir,
                    plan=plan,
                )

    def test_global_validation_rejects_resigned_duplicate_leaf(self):
        with TemporaryDirectory() as temporary_directory:
            _, plan_dir, plan = _build_leaf_plan(
                Path(temporary_directory),
                leaf_count=2,
                world_size=2,
            )
            rank_zero = plan_dir / "rank-000.jsonl"
            rank_one = plan_dir / "rank-001.jsonl"
            rank_one.write_text(
                rank_zero.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            plan["ranks"][1]["rank_plan_sha256"] = builder._sha256_file(
                rank_one
            )

            with self.assertRaisesRegex(
                ValueError,
                "Duplicate global chunk assignment",
            ):
                builder._load_and_validate_all_rank_tasks(
                    plan_dir,
                    plan=plan,
                )

    def test_global_validation_rejects_resigned_missing_leaf(self):
        with TemporaryDirectory() as temporary_directory:
            _, plan_dir, plan = _build_leaf_plan(
                Path(temporary_directory),
                leaf_count=2,
                world_size=1,
            )
            rank_path = plan_dir / "rank-000.jsonl"
            first_line = rank_path.read_text(
                encoding="utf-8"
            ).splitlines()[0]
            first_record = json.loads(first_line)
            rank_path.write_text(first_line + "\n", encoding="utf-8")
            plan["ranks"][0].update(
                {
                    "leaf_count": 1,
                    "task_count": 1,
                    "cif_count": first_record["cif_count"],
                    "rank_plan_sha256": builder._sha256_file(rank_path),
                }
            )

            with self.assertRaisesRegex(
                ValueError,
                "Global leaf_count mismatch",
            ):
                builder._load_and_validate_all_rank_tasks(
                    plan_dir,
                    plan=plan,
                )

    def test_predicted_cif_requires_exactly_one_index_row(self):
        builder._validate_single_index_row([{"pdb_id": "one"}])
        with self.assertRaisesRegex(
            ValueError,
            "exactly one training index row",
        ):
            builder._validate_single_index_row([])
        with self.assertRaisesRegex(
            ValueError,
            "found 2",
        ):
            builder._validate_single_index_row([{}, {}])

    def test_finalize_does_not_publish_incomplete_outputs(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            _, plan_dir, _ = _build_leaf_plan(
                base,
                leaf_count=1,
                world_size=1,
            )
            catalog_path = plan_dir / "leaf-catalog.csv"
            merged_path = plan_dir / "merged.csv.gz"

            with self.assertRaisesRegex(
                RuntimeError,
                '"incomplete_leaves": 1',
            ):
                builder.finalize_plan(
                    plan_dir,
                    catalog_path=catalog_path,
                    merged_indices_path=merged_path,
                    fast_path_mode="require",
                )

            self.assertFalse(catalog_path.exists())
            self.assertFalse(merged_path.exists())

    def test_finalize_require_fast_path_rejects_fallback_markers(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            root, plan_dir, plan = _build_leaf_plan(
                base,
                leaf_count=1,
                world_size=1,
            )
            task = builder._load_and_validate_all_rank_tasks(
                plan_dir,
                plan=plan,
            )[0][0]
            source_root = builder._roots_from_plan(plan)["0"]
            leaf_dir = root / task.leaf_relpath
            prepared = builder.PreparedLeaf(
                job=builder._leaf_job_from_task(task),
                source_root=source_root,
                leaf_dir=leaf_dir,
                cif_names=(),
            )
            paths = builder._chunk_paths(prepared, task)
            paths["state_dir"].mkdir(parents=True)
            paths["index"].touch()
            builder._atomic_json_dump(
                {
                    "plan_id": plan["plan_id"],
                    "cif_names_sha256": task.cif_names_sha256,
                    "chunk_index": task.chunk_index,
                    "chunk_count": task.chunk_count,
                    "begin": task.begin,
                    "end": task.end,
                    "success_count": task.size,
                    "index_rows": task.size,
                    "index_schema_sha256": builder.INDEX_SCHEMA_SHA256,
                    **builder._file_marker_fields(
                        paths["index"],
                        prefix="index",
                    ),
                    "fast_path_used": 0,
                    "fast_cif_reader_used": 0,
                },
                paths["success"],
            )
            output_dir = leaf_dir / builder.BIOASSEMBLY_DIRNAME
            leaf_index = output_dir / builder.LEAF_INDEX_NAME
            leaf_index.touch()
            expected_output_names = builder._expected_leaf_output_names(
                prepared
            )
            (output_dir / expected_output_names[0]).write_bytes(b"pickle")
            builder._atomic_json_dump(
                {
                    "plan_id": plan["plan_id"],
                    "root_id": "0",
                    "leaf_relpath": task.leaf_relpath,
                    "cif_names_sha256": task.cif_names_sha256,
                    "cif_count": task.cif_count,
                    "index_rows": task.cif_count,
                    "index_schema_sha256": builder.INDEX_SCHEMA_SHA256,
                    **builder._file_marker_fields(
                        leaf_index,
                        prefix="index",
                    ),
                    "pkl_count": task.cif_count,
                    "pkl_names_sha256": builder._hash_cif_names(
                        expected_output_names
                    ),
                    "fast_path_used": 0,
                    "fast_cif_reader_used": 0,
                },
                output_dir / builder.LEAF_SUCCESS_NAME,
            )
            strict_catalog = plan_dir / "strict.csv"

            with self.assertRaisesRegex(RuntimeError, "incomplete_leaves"):
                builder.finalize_plan(
                    plan_dir,
                    catalog_path=strict_catalog,
                    fast_path_mode="require",
                )
            self.assertFalse(strict_catalog.exists())

            auto_result = builder.finalize_plan(
                plan_dir,
                catalog_path=plan_dir / "auto.csv",
                fast_path_mode="auto",
            )
            self.assertEqual(auto_result["complete_cifs"], 1)
            finalize_success = (
                plan_dir / builder.FINALIZE_SUCCESS_NAME
            )
            self.assertTrue(finalize_success.is_file())
            with (plan_dir / "auto.csv").open(
                encoding="utf-8",
                newline="",
            ) as stream:
                audit_row = next(csv.DictReader(stream))
            self.assertEqual(
                audit_row["artifact_role"],
                builder.LEAF_AUDIT_ROLE,
            )

            with mock.patch.object(
                builder,
                "_write_merged_indices",
                side_effect=RuntimeError("merge failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "merge failed"):
                    builder.finalize_plan(
                        plan_dir,
                        catalog_path=plan_dir / "auto.csv",
                        merged_indices_path=plan_dir / "merged.csv.gz",
                        fast_path_mode="auto",
                    )
            self.assertFalse(finalize_success.exists())

    def test_finalize_outputs_are_guarded_conversion_audits(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            _, plan_dir, _ = _build_leaf_plan(
                base,
                leaf_count=1,
                world_size=1,
            )
            with self.assertRaisesRegex(
                ValueError,
                "inside the immutable plan directory",
            ):
                builder.finalize_plan(
                    plan_dir,
                    catalog_path=base / "outside.csv",
                )
            with self.assertRaisesRegex(
                ValueError,
                "authoritative training catalog name",
            ):
                builder.finalize_plan(
                    plan_dir,
                    catalog_path=plan_dir / "leaf-audit.csv",
                    merged_indices_path=(
                        plan_dir / "distillation_index.csv.gz"
                    ),
                )

    def test_finalize_cli_exposes_require_fast_path(self):
        arguments = builder._build_parser().parse_args(
            [
                "finalize",
                "--plan-dir",
                "/tmp/plan",
                "--catalog",
                "/tmp/catalog.csv",
                "--require-fast-path",
            ]
        )
        self.assertEqual(arguments.fast_path_mode, "require")

    def test_plan_cli_exposes_scan_workers_with_default(self):
        default_arguments = builder._build_parser().parse_args(
            ["plan", "--plan-dir", "/tmp/plan"]
        )
        custom_arguments = builder._build_parser().parse_args(
            [
                "plan",
                "--plan-dir",
                "/tmp/plan",
                "--scan-workers",
                "7",
            ]
        )

        self.assertEqual(default_arguments.scan_workers, 64)
        self.assertEqual(custom_arguments.scan_workers, 7)
        with self.assertRaisesRegex(ValueError, "scan_workers must be positive"):
            builder._inventory_leaves([], scan_workers=0)


class ChunkResumeTest(unittest.TestCase):
    def setUp(self):
        builder._PREPARED_LEAVES = {}

    def tearDown(self):
        builder._PREPARED_LEAVES = {}

    def test_resume_rejects_success_record_with_multiple_index_rows(self):
        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            (output_dir / "sample.pkl.gz").touch()
            record = {
                "status": "success",
                "output_name": "sample.pkl.gz",
                "index_rows": [{}, {}],
                "fast_path_used": True,
                "fast_cif_reader_used": True,
            }

            self.assertFalse(
                builder._record_is_complete(
                    record,
                    output_dir=output_dir,
                    fast_path_mode="require",
                )
            )

    @staticmethod
    def _prepare_fixture(base: Path, count: int = 5):
        root = base / "root"
        leaf_dir = root / "leaf"
        leaf_dir.mkdir(parents=True)
        cif_names = tuple(
            ["ProteinA.cif"]
            + [f"protein-{index}.cif" for index in range(1, count)]
        )
        for name in cif_names:
            (leaf_dir / name).write_text("cif", encoding="utf-8")
        job = builder.LeafJob(
            leaf_id=0,
            root_id="0",
            leaf_relpath="leaf",
            cif_count=len(cif_names),
            cif_names_sha256=builder._hash_cif_names(cif_names),
        )
        source_root = builder.SourceRoot("0", "afdb", str(root))
        prepared = builder.PreparedLeaf(
            job=job,
            source_root=source_root,
            leaf_dir=leaf_dir,
            cif_names=cif_names,
        )
        builder._PREPARED_LEAVES = {0: prepared}
        return prepared

    @staticmethod
    def _fake_convert(
        *,
        prepared,
        cif_name,
        compresslevel,
        fast_path_mode,
        durable_output,
    ):
        del compresslevel, fast_path_mode, durable_output
        sample_id = Path(cif_name).stem
        output_dir = prepared.leaf_dir / builder.BIOASSEMBLY_DIRNAME
        output_dir.mkdir(exist_ok=True)
        output_name = f"{sample_id}.pkl.gz"
        (output_dir / output_name).write_bytes(b"complete")
        return {
            "status": "success",
            "cif_name": cif_name,
            "sample_id": sample_id,
            "output_name": output_name,
            "fast_path_used": True,
            "fast_cif_reader_used": True,
            "index_rows": [
                {
                    "pdb_id": sample_id,
                    "source_db": prepared.source_root.source_db,
                    "source_root_id": prepared.source_root.root_id,
                    "cif_relpath": (
                        Path(prepared.job.leaf_relpath) / cif_name
                    ).as_posix(),
                    "bioassembly_relpath": (
                        Path(prepared.job.leaf_relpath)
                        / builder.BIOASSEMBLY_DIRNAME
                        / output_name
                    ).as_posix(),
                }
            ],
        }

    def test_chunk_writes_relative_journal_and_resumes_without_conversion(self):
        with TemporaryDirectory() as temporary_directory:
            prepared = self._prepare_fixture(Path(temporary_directory), count=3)
            task = builder._split_leaf_job(prepared.job, 10)[0]
            invocation = (task, "plan-id", 3, "require", False, None)

            with mock.patch.object(
                builder,
                "_convert_one",
                side_effect=self._fake_convert,
            ) as convert:
                first = builder._process_chunk(invocation)

            paths = builder._chunk_paths(prepared, task)
            self.assertEqual(first["status"], "complete")
            self.assertEqual(convert.call_count, 3)
            self.assertTrue(paths["success"].is_file())
            self.assertTrue(paths["index"].is_file())
            self.assertTrue(
                (
                    prepared.leaf_dir
                    / builder.BIOASSEMBLY_DIRNAME
                    / "ProteinA.pkl.gz"
                ).is_file()
            )
            journal_text = paths["journal"].read_text(encoding="utf-8")
            self.assertNotIn(str(prepared.leaf_dir), journal_text)
            self.assertIn('"cif_name":"ProteinA.cif"', journal_text)

            with mock.patch.object(
                builder,
                "_convert_one",
                side_effect=AssertionError("must not convert"),
            ):
                second = builder._process_chunk(invocation)

            self.assertEqual(second["status"], "complete")
            self.assertEqual(second["skipped"], 3)

    def test_partial_journal_tail_is_repaired_across_two_interruptions(self):
        with TemporaryDirectory() as temporary_directory:
            prepared = self._prepare_fixture(Path(temporary_directory), count=3)
            task = builder._split_leaf_job(prepared.job, 10)[0]
            invocation = (task, "plan-id", 3, "require", False, None)
            with mock.patch.object(
                builder,
                "_convert_one",
                side_effect=self._fake_convert,
            ):
                self.assertEqual(
                    builder._process_chunk(invocation)["status"],
                    "complete",
                )

            paths = builder._chunk_paths(prepared, task)
            for partial_tail in (b'{"status":', b'{"cif_name":"broken'):
                paths["success"].unlink()
                with paths["journal"].open("ab") as stream:
                    stream.write(partial_tail)
                with mock.patch.object(
                    builder,
                    "_convert_one",
                    side_effect=AssertionError("must not convert"),
                ):
                    result = builder._process_chunk(invocation)
                self.assertEqual(result["status"], "complete")
                journal_bytes = paths["journal"].read_bytes()
                self.assertTrue(journal_bytes.endswith(b"\n"))
                self.assertNotIn(partial_tail, journal_bytes)
                for line in journal_bytes.splitlines():
                    self.assertIsInstance(
                        json.loads(line.decode("utf-8")),
                        dict,
                    )

    def test_corrupt_chunk_index_is_rebuilt_from_journal(self):
        with TemporaryDirectory() as temporary_directory:
            prepared = self._prepare_fixture(Path(temporary_directory), count=3)
            task = builder._split_leaf_job(prepared.job, 10)[0]
            invocation = (task, "plan-id", 3, "require", False, None)
            with mock.patch.object(
                builder,
                "_convert_one",
                side_effect=self._fake_convert,
            ):
                builder._process_chunk(invocation)

            paths = builder._chunk_paths(prepared, task)
            paths["index"].write_bytes(b"not-a-gzip-index")
            self.assertFalse(
                builder._chunk_marker_matches(
                    task,
                    plan_id="plan-id",
                    fast_path_mode="require",
                    paths=paths,
                )
            )

            with mock.patch.object(
                builder,
                "_convert_one",
                side_effect=AssertionError("must not convert"),
            ):
                result = builder._process_chunk(invocation)

            self.assertEqual(result["status"], "complete")
            self.assertTrue(
                builder._chunk_marker_matches(
                    task,
                    plan_id="plan-id",
                    fast_path_mode="require",
                    paths=paths,
                )
            )
            with gzip.open(paths["index"], "rt", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)

    def test_missing_pickle_invalidates_own_chunk_and_is_regenerated(self):
        with TemporaryDirectory() as temporary_directory:
            prepared = self._prepare_fixture(Path(temporary_directory), count=5)
            tasks = builder._split_leaf_job(prepared.job, 3)
            with mock.patch.object(
                builder,
                "_convert_one",
                side_effect=self._fake_convert,
            ):
                for task in tasks:
                    builder._process_chunk(
                        (task, "plan-id", 3, "require", False, None)
                    )
            self.assertTrue(
                builder._finalize_leaf(
                    tasks,
                    prepared,
                    plan_id="plan-id",
                    fast_path_mode="require",
                    compresslevel=3,
                )
            )

            missing_name = (
                f"{Path(prepared.cif_names[-1]).stem}.pkl.gz"
            )
            (
                prepared.leaf_dir
                / builder.BIOASSEMBLY_DIRNAME
                / missing_name
            ).unlink()
            self.assertFalse(
                builder._leaf_marker_matches(
                    prepared,
                    plan_id="plan-id",
                    fast_path_mode="require",
                    verify_outputs=True,
                )
            )
            self.assertFalse(
                builder._finalize_leaf(
                    tasks,
                    prepared,
                    plan_id="plan-id",
                    fast_path_mode="require",
                    compresslevel=3,
                )
            )
            missing_task = tasks[-1]
            missing_paths = builder._chunk_paths(prepared, missing_task)
            self.assertFalse(missing_paths["success"].exists())

            with mock.patch.object(
                builder,
                "_convert_one",
                side_effect=self._fake_convert,
            ) as convert:
                result = builder._process_chunk(
                    (
                        missing_task,
                        "plan-id",
                        3,
                        "require",
                        False,
                        None,
                    )
                )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(convert.call_count, 1)
            self.assertTrue(
                builder._finalize_leaf(
                    tasks,
                    prepared,
                    plan_id="plan-id",
                    fast_path_mode="require",
                    compresslevel=3,
                )
            )

    def test_complete_rank_resume_does_not_remerge_leaf_index(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            root, plan_dir, plan = _build_leaf_plan(
                base,
                leaf_count=1,
                world_size=1,
            )
            task = builder._load_and_validate_all_rank_tasks(
                plan_dir,
                plan=plan,
            )[0][0]
            source_root = builder._roots_from_plan(plan)["0"]
            leaf_dir = root / task.leaf_relpath
            prepared = builder.PreparedLeaf(
                job=builder._leaf_job_from_task(task),
                source_root=source_root,
                leaf_dir=leaf_dir,
                cif_names=builder._scan_leaf_cif_names(leaf_dir),
            )
            builder._PREPARED_LEAVES = {task.leaf_id: prepared}
            with mock.patch.object(
                builder,
                "_convert_one",
                side_effect=self._fake_convert,
            ):
                builder._process_chunk(
                    (
                        task,
                        plan["plan_id"],
                        3,
                        "require",
                        False,
                        None,
                    )
                )
            self.assertTrue(
                builder._finalize_leaf(
                    [task],
                    prepared,
                    plan_id=plan["plan_id"],
                    fast_path_mode="require",
                    compresslevel=3,
                )
            )

            with mock.patch.object(
                builder,
                "_configure_protenix_root",
            ), mock.patch.object(
                builder,
                "_finalize_leaf",
                side_effect=AssertionError("must not remerge"),
            ) as finalize_leaf:
                result = builder.run_rank(
                    plan_dir,
                    rank=0,
                    workers=1,
                    fast_path_mode="require",
                )

            self.assertEqual(result["state"], "complete")
            self.assertEqual(result["skipped"], 1)
            finalize_leaf.assert_not_called()

    def test_rank_resume_repairs_one_missing_pickle_in_same_run(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            root, plan_dir, plan = _build_leaf_plan(
                base,
                leaf_count=1,
                world_size=1,
            )
            task = builder._load_and_validate_all_rank_tasks(
                plan_dir,
                plan=plan,
            )[0][0]
            source_root = builder._roots_from_plan(plan)["0"]
            leaf_dir = root / task.leaf_relpath
            prepared = builder.PreparedLeaf(
                job=builder._leaf_job_from_task(task),
                source_root=source_root,
                leaf_dir=leaf_dir,
                cif_names=builder._scan_leaf_cif_names(leaf_dir),
            )
            builder._PREPARED_LEAVES = {task.leaf_id: prepared}
            with mock.patch.object(
                builder,
                "_convert_one",
                side_effect=self._fake_convert,
            ):
                builder._process_chunk(
                    (
                        task,
                        plan["plan_id"],
                        3,
                        "require",
                        False,
                        None,
                    )
                )
            self.assertTrue(
                builder._finalize_leaf(
                    [task],
                    prepared,
                    plan_id=plan["plan_id"],
                    fast_path_mode="require",
                    compresslevel=3,
                )
            )
            output_name = f"{Path(prepared.cif_names[0]).stem}.pkl.gz"
            output_path = (
                leaf_dir / builder.BIOASSEMBLY_DIRNAME / output_name
            )
            output_path.unlink()

            def run_synchronously(
                invocations,
                *,
                workers,
                tail_max_lanes_per_leaf,
            ):
                del workers, tail_max_lanes_per_leaf
                for invocation in invocations:
                    yield builder._process_chunk(invocation)

            with mock.patch.object(
                builder,
                "_configure_protenix_root",
            ), mock.patch.object(
                builder,
                "_prewarm_parent",
                return_value={"elapsed_seconds": 0.0},
            ), mock.patch.object(
                builder,
                "_run_directory_affine_tasks",
                new=run_synchronously,
            ), mock.patch.object(
                builder,
                "_convert_one",
                side_effect=self._fake_convert,
            ) as convert:
                result = builder.run_rank(
                    plan_dir,
                    rank=0,
                    workers=1,
                    fast_path_mode="require",
                )

            self.assertEqual(result["state"], "complete")
            self.assertEqual(convert.call_count, 1)
            self.assertTrue(output_path.is_file())
            self.assertTrue(
                builder._chunk_paths(prepared, task)["success"].is_file()
            )
            self.assertTrue(
                (
                    leaf_dir
                    / builder.BIOASSEMBLY_DIRNAME
                    / builder.LEAF_SUCCESS_NAME
                ).is_file()
            )

    def test_broken_worker_pool_is_reported_as_crashed(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            _, plan_dir, plan = _build_leaf_plan(
                base,
                leaf_count=1,
                world_size=1,
            )
            with mock.patch.object(
                builder,
                "_configure_protenix_root",
            ), mock.patch.object(
                builder,
                "_prewarm_parent",
                return_value={"elapsed_seconds": 0.0},
            ), mock.patch.object(
                builder,
                "_run_directory_affine_tasks",
                side_effect=builder.BrokenProcessPool("simulated"),
            ):
                with self.assertRaises(builder.BrokenProcessPool):
                    builder.run_rank(
                        plan_dir,
                        rank=0,
                        workers=1,
                        fast_path_mode="require",
                    )

            status = json.loads(
                (plan_dir / "status" / "rank-000.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(status["plan_id"], plan["plan_id"])
            self.assertEqual(status["state"], "crashed")
            self.assertIn("BrokenProcessPool", status["message"])

    def test_prewarm_failure_is_reported_as_crashed(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            _, plan_dir, _ = _build_leaf_plan(
                base,
                leaf_count=1,
                world_size=1,
            )
            with mock.patch.object(
                builder,
                "_configure_protenix_root",
            ), mock.patch.object(
                builder,
                "_prewarm_parent",
                side_effect=RuntimeError("prewarm failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "prewarm failed"):
                    builder.run_rank(
                        plan_dir,
                        rank=0,
                        workers=1,
                        fast_path_mode="require",
                    )

            status = json.loads(
                (plan_dir / "status" / "rank-000.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(status["state"], "crashed")
            self.assertIn("prewarm failed", status["message"])

    def test_leaf_finalize_failure_is_reported_as_crashed(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            root, plan_dir, plan = _build_leaf_plan(
                base,
                leaf_count=1,
                world_size=1,
            )
            task = builder._load_and_validate_all_rank_tasks(
                plan_dir,
                plan=plan,
            )[0][0]
            source_root = builder._roots_from_plan(plan)["0"]
            leaf_dir = root / task.leaf_relpath
            prepared = builder.PreparedLeaf(
                job=builder._leaf_job_from_task(task),
                source_root=source_root,
                leaf_dir=leaf_dir,
                cif_names=builder._scan_leaf_cif_names(leaf_dir),
            )
            builder._PREPARED_LEAVES = {task.leaf_id: prepared}
            with mock.patch.object(
                builder,
                "_convert_one",
                side_effect=self._fake_convert,
            ):
                builder._process_chunk(
                    (
                        task,
                        plan["plan_id"],
                        3,
                        "require",
                        False,
                        None,
                    )
                )

            with mock.patch.object(
                builder,
                "_configure_protenix_root",
            ), mock.patch.object(
                builder,
                "_finalize_leaf",
                side_effect=RuntimeError("finalize failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "finalize failed"):
                    builder.run_rank(
                        plan_dir,
                        rank=0,
                        workers=1,
                        fast_path_mode="require",
                    )

            status = json.loads(
                (plan_dir / "status" / "rank-000.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(status["state"], "crashed")
            self.assertIn("finalize failed", status["message"])

    def test_two_chunks_merge_into_one_leaf_index_and_marker(self):
        with TemporaryDirectory() as temporary_directory:
            prepared = self._prepare_fixture(Path(temporary_directory), count=5)
            tasks = builder._split_leaf_job(prepared.job, 3)

            with mock.patch.object(
                builder,
                "_convert_one",
                side_effect=self._fake_convert,
            ):
                results = [
                    builder._process_chunk(
                        (task, "plan-id", 3, "require", False, None)
                    )
                    for task in tasks
                ]

            self.assertTrue(
                builder._finalize_leaf(
                    tasks,
                    prepared,
                    plan_id="plan-id",
                    fast_path_mode="require",
                    compresslevel=3,
                )
            )
            output_dir = prepared.leaf_dir / builder.BIOASSEMBLY_DIRNAME
            marker = json.loads(
                (output_dir / builder.LEAF_SUCCESS_NAME).read_text(
                    encoding="utf-8"
                )
            )
            with gzip.open(
                output_dir / builder.LEAF_INDEX_NAME,
                "rt",
                encoding="utf-8",
                newline="",
            ) as stream:
                rows = list(csv.DictReader(stream))

            self.assertTrue(
                all(result["status"] == "complete" for result in results)
            )
            self.assertEqual(marker["cif_count"], 5)
            self.assertEqual(marker["chunk_count"], 2)
            self.assertEqual(len(rows), 5)
            self.assertEqual(
                {row["pdb_id"] for row in rows},
                {Path(name).stem for name in prepared.cif_names},
            )

    def test_inventory_change_is_detected_before_fork(self):
        with TemporaryDirectory() as temporary_directory:
            prepared = self._prepare_fixture(Path(temporary_directory), count=2)
            task = builder._split_leaf_job(prepared.job, 10)[0]
            (prepared.leaf_dir / "new.cif").write_text(
                "new",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "no longer matches",
            ):
                builder._prepare_rank_leaves(
                    [task],
                    {"0": prepared.source_root},
                )


if __name__ == "__main__":
    unittest.main()
