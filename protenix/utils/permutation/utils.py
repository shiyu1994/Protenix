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

import itertools
import os
import time
from collections.abc import Mapping

import torch


_PERMUTATION_ERROR_SAVE_COUNTER = itertools.count()


class Checker:
    @staticmethod
    def is_permutation(x: torch.Tensor):
        """
        Checks if the input tensor `x` is a permutation of integers from 0 to N-1.

        Args:
            x (torch.Tensor): A 1D tensor of size [N].
        """
        assert x.dim() == 1
        N = x.size(0)
        assert torch.equal(torch.sort(x)[0], torch.arange(N, device=x.device))

    @staticmethod
    def are_permutations(x: torch.Tensor, dim=-1):
        """
        Checks if slices along the specified dimension in `x` are permutations of integers from 0 to N-1.

        Args:
            x (torch.Tensor): A tensor with any number of dimensions, containing slices of size N along `dim`.
            dim (int, optional): The dimension along which to check for permutations. Defaults to -1.
        """
        assert x.dim() > 0

        N = x.size(dim)
        # Create a view of x that moves the specified dimension to -1
        x = x.transpose(dim, -1).contiguous()
        x = x.reshape(-1, N)
        expected = torch.arange(N, device=x.device)
        for i in range(x.size(0)):
            Checker.is_permutation(x[i])

    @staticmethod
    def contains_identity(x: torch.Tensor, dim=-1):
        """
        Check if x contains the identity permutation

        Args:
            x (torch.Tensor): A tensor with any number of dimensions, containing slices of size N along `dim`.
            dim (int, optional): The dimension along which to check for permutations. Defaults to -1.
        """
        assert x.dim() > 0

        N = x.size(dim)
        # Create a view of x that moves the specified dimension to -1
        x = x.transpose(dim, -1).contiguous()
        x = x.reshape(-1, N)
        expected = torch.arange(N, device=x.device).unsqueeze(dim=0)
        assert (x == expected).all(dim=-1).any()

    @staticmethod
    def not_contain_identity(x: torch.Tensor, dim=-1):
        """
        Check if x does not contain the identity permutation

        Args:
            x (torch.Tensor): A tensor with any number of dimensions, containing slices of size N along `dim`.
            dim (int, optional): The dimension along which to check for permutations. Defaults to -1.
        """
        assert x.dim() > 0

        N = x.size(dim)
        # Create a view of x that moves the specified dimension to -1
        x = x.transpose(dim, -1).contiguous()
        x = x.reshape(-1, N)
        expected = torch.arange(N, device=x.device).unsqueeze(dim=0)
        assert not (x == expected).all(dim=-1).any()

    @staticmethod
    def batch_permute(perm: torch.Tensor, x: torch.Tensor, x_permuted: torch.Tensor):
        """
        Args:
            perm (torch.Tensor):
                [..., N]
            x (torch.Tensor):
                [N, batch_dims_x]
            x_permuted (torch.Tensor):
                [..., N, batch_dims_x]
        """
        batch_shape = perm.shape[:-1]
        N = perm.size(-1)
        assert x.size(0) == N
        perm = perm.view(-1, N)
        permuted_x = [x[perm[i]] for i in range(len(perm))]
        permuted_x = torch.stack(permuted_x, dim=0)  # [-1, N, batch_dims_x]
        target_shape = batch_shape + (N,) + x.shape[1:]
        assert torch.allclose(permuted_x.reshape(target_shape), x_permuted)


def _snapshot_to_cpu(data):
    """Create a detached CPU snapshot suitable for diagnostic serialization."""
    if isinstance(data, torch.Tensor):
        return data.detach().to(device="cpu", copy=True)
    if isinstance(data, Mapping):
        return {key: _snapshot_to_cpu(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_snapshot_to_cpu(value) for value in data]
    if isinstance(data, tuple):
        return tuple(_snapshot_to_cpu(value) for value in data)
    return data


def _get_global_rank_and_world_size() -> tuple[int, int]:
    """Return a valid global rank and world size without initializing distributed."""
    rank = None
    world_size = None
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = int(torch.distributed.get_rank())
            world_size = int(torch.distributed.get_world_size())
    except (RuntimeError, ValueError):
        rank = None
        world_size = None

    if rank is None or world_size is None:
        try:
            rank = int(os.environ.get("RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError:
            rank = 0
            world_size = 1

    if world_size < 1 or rank < 0 or rank >= world_size:
        return 0, 1
    return rank, world_size


def _rank_case_quota(max_cases: int, rank: int, world_size: int) -> int:
    """Deterministically partition a global diagnostic budget across ranks."""
    if max_cases <= 0:
        return 0
    cases_per_rank, remainder = divmod(max_cases, world_size)
    return cases_per_rank + int(rank < remainder)


def save_permutation_error(data, error_dir: str = None, max_cases: int = 50):
    """
    Saves the permutation error data to a specified directory.

    Args:
        data: The data to be saved.
        error_dir (str): The directory where the error data should be saved.
        max_cases (int): The maximum number of error cases to save.

    Raises:
        Exception: If an error occurs while saving the data, the exception is caught and printed.
    """
    if error_dir is None:
        return

    temporary_path = None
    try:
        rank, world_size = _get_global_rank_and_world_size()
        rank_quota = _rank_case_quota(max_cases, rank, world_size)
        if rank_quota == 0:
            return

        os.makedirs(error_dir, exist_ok=True)
        rank_prefix = f"T_rank{rank:05d}_"
        rank_case_count = sum(
            filename.startswith(rank_prefix) and filename.endswith(".pt")
            for filename in os.listdir(error_dir)
        )
        if rank_case_count >= rank_quota:
            return

        counter = next(_PERMUTATION_ERROR_SAVE_COUNTER)
        filename = (
            f"{rank_prefix}pid{os.getpid()}_ns{time.time_ns()}_"
            f"{counter:08d}.pt"
        )
        final_path = os.path.join(error_dir, filename)
        temporary_path = os.path.join(
            error_dir,
            f".{filename}.tmp.{os.getpid()}.{counter}",
        )

        cpu_snapshot = _snapshot_to_cpu(data)
        torch.save(cpu_snapshot, temporary_path)
        os.replace(temporary_path, final_path)
    except Exception as e:
        if temporary_path is not None:
            try:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
            except OSError:
                pass
        print(f"Exception occurs in save_permutation_error: {e}")
