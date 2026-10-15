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

from typing import List, Optional

import torch

from protenix.utils.fsdp import reshard_fsdp2_model


def _is_dtensor(tensor: torch.Tensor) -> bool:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    return isinstance(tensor, DTensor)


def _local_parameter_tensor(param: torch.nn.Parameter) -> torch.Tensor:
    value = param.detach()
    if _is_dtensor(value):
        return value.to_local()
    return value


class EMAWrapper(object):
    """
    A wrapper class for exponential moving average of model weights.

    Args:
        model (torch.nn.Module): The PyTorch model to apply EMA to.
        decay (float): The decay rate for exponential moving average.
        mutable_param_keywords (Optional[List[str]]): List of keywords to identify parameters
            that should be updated by EMA. If None, all parameters are updated.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,
        mutable_param_keywords: Optional[List[str]] = None,
        shadow_device: str = "model",
    ):
        self.model = model
        self.decay = decay
        if shadow_device not in ("model", "cpu"):
            raise ValueError(
                f"shadow_device must be 'model' or 'cpu', got {shadow_device!r}"
            )
        self.shadow_device = shadow_device
        if mutable_param_keywords is not None:
            self.mutable_param_keywords = [
                s.strip() for s in mutable_param_keywords if s.strip()
            ]
        else:
            self.mutable_param_keywords = None
        self.shadow = {}
        self.backup = {}

    def register(self) -> None:
        """
        Register the model's parameters and create initial shadow copies.
        """
        reshard_fsdp2_model(self.model)
        self.shadow = {}
        for name, param in self.model.named_parameters():
            value = _local_parameter_tensor(param)
            device = (
                torch.device("cpu") if self.shadow_device == "cpu" else value.device
            )
            self.shadow[name] = value.to(device=device, dtype=value.dtype, copy=True)

    def update(self) -> None:
        """
        Update the shadow copies of the parameters using the current model weights and decay rate.
        """
        reshard_fsdp2_model(self.model)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if self.mutable_param_keywords and not any(
                    [keyword in name for keyword in self.mutable_param_keywords]
                ):
                    continue
                assert name in self.shadow
                value = _local_parameter_tensor(param)
                shadow = self.shadow[name]
                param_on_shadow_device = value.to(
                    device=shadow.device,
                    dtype=shadow.dtype,
                    copy=self.shadow_device == "cpu",
                    non_blocking=False,
                )
                # Keep train2's allocation-free in-place EMA update.
                shadow.mul_(self.decay).add_(
                    param_on_shadow_device, alpha=1.0 - self.decay
                )

    def apply_shadow(self) -> None:
        """
        Apply the shadow copies (EMA weights) to the model, backing up current weights.
        """
        reshard_fsdp2_model(self.model)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                assert name in self.shadow
                if _is_dtensor(param):
                    value = _local_parameter_tensor(param)
                    backup_device = (
                        torch.device("cpu")
                        if self.shadow_device == "cpu"
                        else value.device
                    )
                    self.backup[name] = value.to(
                        device=backup_device,
                        dtype=value.dtype,
                        copy=True,
                    )
                    value.copy_(
                        self.shadow[name].to(
                            device=value.device,
                            dtype=value.dtype,
                            non_blocking=True,
                        )
                    )
                elif self.shadow_device == "model":
                    self.backup[name] = param.data
                    param.data = self.shadow[name]
                else:
                    self.backup[name] = param.detach().to(
                        device="cpu", dtype=param.dtype, copy=True
                    )
                    param.copy_(
                        self.shadow[name].to(
                            device=param.device,
                            dtype=param.dtype,
                            non_blocking=True,
                        )
                    )

    def restore(self) -> None:
        """
        Restore the original model weights from the backup.
        """
        reshard_fsdp2_model(self.model)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                assert name in self.backup
                if _is_dtensor(param):
                    value = _local_parameter_tensor(param)
                    value.copy_(
                        self.backup[name].to(
                            device=value.device,
                            dtype=value.dtype,
                            non_blocking=True,
                        )
                    )
                elif self.shadow_device == "model":
                    param.data = self.backup[name]
                else:
                    param.copy_(
                        self.backup[name].to(
                            device=param.device,
                            dtype=param.dtype,
                            non_blocking=True,
                        )
                    )
        self.backup = {}
