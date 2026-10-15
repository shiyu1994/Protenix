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

from pathlib import Path

import torch

from protenix.utils.permutation import atom_permutation
from protenix.utils.permutation import utils as permutation_utils


def _all_tensors(data):
    if isinstance(data, torch.Tensor):
        yield data
    elif isinstance(data, dict):
        for value in data.values():
            yield from _all_tensors(value)
    elif isinstance(data, (list, tuple)):
        for value in data:
            yield from _all_tensors(value)


def test_snapshot_to_cpu_recursively_detaches_and_copies_tensors():
    tensor = torch.arange(6.0, requires_grad=True).reshape(2, 3)
    payload = {
        "tensor": tensor,
        "nested": [tensor[:, 0], ({"mask": torch.ones(2, dtype=torch.bool)},)],
    }

    snapshot = permutation_utils._snapshot_to_cpu(payload)

    snapshot_tensors = list(_all_tensors(snapshot))
    assert len(snapshot_tensors) == 3
    assert all(value.device.type == "cpu" for value in snapshot_tensors)
    assert all(not value.requires_grad for value in snapshot_tensors)
    assert snapshot["tensor"].data_ptr() != tensor.data_ptr()
    tensor.detach().zero_()
    assert torch.equal(snapshot["tensor"], torch.arange(6.0).reshape(2, 3))


def test_multirank_unique_atomic_files_obey_global_case_budget(
    tmp_path, monkeypatch
):
    rank_holder = [0]
    monkeypatch.setattr(
        permutation_utils,
        "_get_global_rank_and_world_size",
        lambda: (rank_holder[0], 8),
    )
    monkeypatch.setattr(permutation_utils.time, "time_ns", lambda: 123456789)

    for rank in range(8):
        rank_holder[0] = rank
        for case_index in range(20):
            permutation_utils.save_permutation_error(
                {"rank": rank, "case": case_index, "value": torch.tensor([rank])},
                error_dir=str(tmp_path),
                max_cases=50,
            )

    diagnostic_paths = sorted(tmp_path.glob("*.pt"))
    assert len(diagnostic_paths) == 50
    expected_rank_counts = [7, 7, 6, 6, 6, 6, 6, 6]
    for rank, expected_count in enumerate(expected_rank_counts):
        rank_paths = list(tmp_path.glob(f"T_rank{rank:05d}_*.pt"))
        assert len(rank_paths) == expected_count
        assert all(
            "_pid" in path.name and "_ns123456789_" in path.name
            for path in rank_paths
        )

    assert len({path.name for path in diagnostic_paths}) == len(diagnostic_paths)
    assert not list(tmp_path.glob(".*.tmp.*"))
    for path in diagnostic_paths:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert all(value.device.type == "cpu" for value in _all_tensors(saved))


def test_failed_save_removes_temporary_and_final_files(tmp_path, monkeypatch):
    monkeypatch.setattr(
        permutation_utils,
        "_get_global_rank_and_world_size",
        lambda: (0, 1),
    )

    def fail_after_partial_write(data, path):
        del data
        Path(path).write_bytes(b"partial")
        raise OSError("injected diagnostic write failure")

    monkeypatch.setattr(permutation_utils.torch, "save", fail_after_partial_write)

    permutation_utils.save_permutation_error(
        {"value": torch.tensor([1.0])},
        error_dir=str(tmp_path),
    )

    assert not list(tmp_path.glob("*.pt"))
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_rank_quota_is_checked_before_cpu_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(
        permutation_utils,
        "_get_global_rank_and_world_size",
        lambda: (0, 8),
    )
    permutation_utils.save_permutation_error(
        {"value": torch.tensor([1.0])},
        error_dir=str(tmp_path),
        max_cases=1,
    )

    snapshot_call_count = 0

    def count_snapshot_calls(data):
        nonlocal snapshot_call_count
        snapshot_call_count += 1
        return data

    monkeypatch.setattr(
        permutation_utils,
        "_snapshot_to_cpu",
        count_snapshot_calls,
    )
    permutation_utils.save_permutation_error(
        {"value": torch.tensor([2.0])},
        error_dir=str(tmp_path),
        max_cases=1,
    )

    assert snapshot_call_count == 0
    assert len(list(tmp_path.glob("*.pt"))) == 1


def test_atom_success_path_does_not_enter_diagnostics(monkeypatch):
    permuted_coord = torch.ones(3, 3)
    permuted_mask = torch.ones(3)
    info = {"permuted": 1}
    indices = torch.arange(3)

    def successful_permutation(**kwargs):
        del kwargs
        return permuted_coord, permuted_mask, info, indices

    def unexpected_diagnostic(*args, **kwargs):
        del args, kwargs
        raise AssertionError("success path must not save diagnostics")

    monkeypatch.setattr(
        atom_permutation,
        "correct_symmetric_atoms",
        successful_permutation,
    )
    monkeypatch.setattr(
        atom_permutation,
        "save_permutation_error",
        unexpected_diagnostic,
    )

    output, returned_info, returned_indices = atom_permutation.run(
        pred_coord=torch.zeros(3, 3),
        true_coord=torch.zeros(3, 3),
        true_coord_mask=torch.ones(3),
        ref_space_uid=torch.arange(3),
        atom_perm_list=[[0], [1], [2]],
    )

    assert output["coordinate"] is permuted_coord
    assert output["coordinate_mask"] is permuted_mask
    assert returned_info is info
    assert returned_indices is indices


def test_atom_error_records_traceback_and_preserves_fallback(monkeypatch):
    captured_diagnostics = []
    captured_warnings = []

    def raise_empty_assertion(**kwargs):
        del kwargs
        raise AssertionError()

    def capture_diagnostic(*, data, error_dir):
        captured_diagnostics.append((data, error_dir))

    monkeypatch.setattr(
        atom_permutation,
        "correct_symmetric_atoms",
        raise_empty_assertion,
    )
    monkeypatch.setattr(
        atom_permutation,
        "save_permutation_error",
        capture_diagnostic,
    )
    monkeypatch.setattr(
        atom_permutation,
        "_atom_permutation_error_log_count",
        0,
    )
    monkeypatch.setattr(
        atom_permutation.logger,
        "warning",
        captured_warnings.append,
    )

    for _ in range(10):
        output = atom_permutation.run(
            pred_coord=torch.zeros(3, 3),
            true_coord=torch.zeros(3, 3),
            true_coord_mask=torch.ones(3),
            ref_space_uid=torch.arange(3),
            atom_perm_list=[[0], [1], [2]],
            error_dir="/unused",
        )
        assert output == ({}, {}, None)

    assert len(captured_diagnostics) == 10
    assert all(error_dir == "/unused" for _, error_dir in captured_diagnostics)
    for data, _ in captured_diagnostics:
        error_message = data["error_message"]
        assert error_message.startswith("AssertionError:")
        assert "Traceback (most recent call last)" in error_message
        assert "raise_empty_assertion" in error_message

    assert len(captured_warnings) == 3
    assert all("Traceback (most recent call last)" not in msg for msg in captured_warnings)
    assert "Subsequent atom-permutation error warnings are suppressed" in (
        captured_warnings[-1]
    )
