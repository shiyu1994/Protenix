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

import os
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist


_FSDP_BLOCK_PATHS = (
    ("pairformer_stack", "blocks"),
    ("diffusion_module", "diffusion_transformer", "blocks"),
    ("confidence_head", "pairformer_stack", "blocks"),
)


def validate_distributed_strategy(configs: Any, world_size: int) -> str:
    strategy = getattr(configs, "distributed_strategy", "ddp")
    if strategy not in ("ddp", "fsdp2"):
        raise ValueError(
            "distributed_strategy must be 'ddp' or 'fsdp2', "
            f"got {strategy!r}"
        )

    optimizer_sharding = getattr(configs, "optimizer_sharding", "none")
    if strategy == "fsdp2" and optimizer_sharding != "none":
        raise ValueError(
            "distributed_strategy='fsdp2' and optimizer_sharding='zero1' "
            "must not be combined; FSDP2 already shards optimizer state"
        )
    if strategy == "fsdp2" and world_size <= 1:
        raise ValueError("distributed_strategy='fsdp2' requires world_size > 1")
    return strategy


def get_fsdp2_shard_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    # The MSA module has a data-dependent early return. Wrapping it as one unit
    # guarantees that every rank enters the same pre-forward collective.
    modules: list[torch.nn.Module] = [model.msa_module]
    for path in _FSDP_BLOCK_PATHS:
        value: Any = model
        for attr in path:
            value = getattr(value, attr)
        modules.extend(value)

    module_ids = [id(module) for module in modules]
    if len(module_ids) != len(set(module_ids)):
        raise ValueError("FSDP2 wrap policy selected a module more than once")
    return modules


def reshard_fsdp2_model(
    model: torch.nn.Module,
    *,
    fsdp_module_type: type[torch.nn.Module] | None = None,
) -> None:
    if fsdp_module_type is None:
        try:
            from torch.distributed.fsdp import FSDPModule
        except ImportError:
            return
        fsdp_module_type = FSDPModule

    for module in model.modules():
        if isinstance(module, fsdp_module_type):
            module.reshard()


def resolve_fsdp_mesh_shape(
    mode: str,
    world_size: int,
    local_world_size: int,
) -> tuple[int, ...]:
    if mode not in ("auto", "global", "hybrid"):
        raise ValueError("fsdp.mesh must be one of: auto, global, hybrid")
    if world_size <= 0 or local_world_size <= 0:
        raise ValueError("world_size and local_world_size must be positive")

    if mode == "auto":
        mode = "hybrid" if world_size > local_world_size else "global"
    if mode == "global":
        return (world_size,)
    if world_size % local_world_size != 0:
        raise ValueError(
            "world_size must be divisible by local_world_size for a hybrid FSDP mesh"
        )
    return (world_size // local_world_size, local_world_size)


def build_fsdp_device_mesh(mode: str):
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("FSDP2 requires an initialized distributed process group")
    if not torch.cuda.is_available():
        raise RuntimeError("This Protenix FSDP2 path requires CUDA")

    from torch.distributed.device_mesh import init_device_mesh

    world_size = dist.get_world_size()
    local_world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE", str(torch.cuda.device_count()))
    )
    shape = resolve_fsdp_mesh_shape(mode, world_size, local_world_size)
    mesh_dim_names = ("fsdp",) if len(shape) == 1 else ("replicate", "shard")
    return init_device_mesh("cuda", shape, mesh_dim_names=mesh_dim_names)


def _dtype_from_name(name: str, *, allow_none_for_fp32: bool) -> torch.dtype | None:
    if name == "fp32":
        return None if allow_none_for_fp32 else torch.float32
    if name == "bf16":
        return torch.bfloat16
    raise ValueError("FSDP dtype must be 'fp32' or 'bf16'")


def apply_fsdp2(
    model: torch.nn.Module,
    *,
    mesh,
    param_dtype: str,
    reduce_dtype: str,
    reshard_after_forward: bool,
    fully_shard_fn: Callable | None = None,
) -> torch.nn.Module:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    if fully_shard_fn is None:
        fully_shard_fn = fully_shard
    mp_policy = MixedPrecisionPolicy(
        # None preserves FP32 parameters and the existing operation-level autocast.
        param_dtype=_dtype_from_name(param_dtype, allow_none_for_fp32=True),
        reduce_dtype=_dtype_from_name(reduce_dtype, allow_none_for_fp32=False),
    )
    kwargs = {
        "mesh": mesh,
        "reshard_after_forward": reshard_after_forward,
        "mp_policy": mp_policy,
    }

    for module in get_fsdp2_shard_modules(model):
        fully_shard_fn(module, **kwargs)
    return fully_shard_fn(model, **kwargs)


def sync_module_states(
    model: torch.nn.Module,
    *,
    src: int = 0,
    broadcast_bucket_size: int = 250 * 1024 * 1024,
) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    if dist.get_world_size() <= 1:
        return

    # DDP performs this synchronization in its constructor; FSDP2 does not.
    from torch.distributed.utils import _sync_module_states

    _sync_module_states(
        model,
        process_group=dist.group.WORLD,
        broadcast_bucket_size=broadcast_bucket_size,
        src=src,
        params_and_buffers_to_ignore=set(),
        broadcast_buffers=True,
    )


def get_fsdp2_full_state_dict(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    return get_state_dict(
        model,
        optimizer,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )


def get_fsdp2_full_model_state_dict(
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Collect a rank-0 CPU full model state without optimizer state."""
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    return get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )


def set_fsdp2_full_model_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, Any],
    *,
    strict: bool,
) -> None:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    set_model_state_dict(
        model,
        state_dict,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            strict=strict,
        ),
    )


def set_fsdp2_full_optimizer_state_dict(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state_dict: dict[str, Any],
) -> None:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_optimizer_state_dict,
    )

    set_optimizer_state_dict(
        model,
        optimizer,
        state_dict,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
        ),
    )


def optimizer_state_uses_canonical_fqns(state_dict: dict[str, Any]) -> bool:
    param_groups = state_dict.get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        return False
    parameter_keys = [
        key
        for group in param_groups
        if isinstance(group, dict)
        for key in group.get("params", [])
    ]
    return bool(parameter_keys) and all(
        isinstance(key, str) and bool(key) for key in parameter_keys
    )
