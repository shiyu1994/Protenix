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

import inspect

import torch
import torch.distributed as dist


def _get_optimizer_sharding(configs) -> str:
    optimizer_sharding = getattr(configs, "optimizer_sharding", "none")
    if optimizer_sharding not in ("none", "zero1"):
        raise ValueError(
            "optimizer_sharding must be 'none' or 'zero1', "
            f"got {optimizer_sharding!r}"
        )
    return optimizer_sharding


def _build_optimizer(
    params,
    optimizer_class: type[torch.optim.Optimizer],
    optimizer_sharding: str,
    **defaults,
) -> torch.optim.Optimizer:
    if optimizer_sharding == "none":
        return optimizer_class(params, **defaults)

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "optimizer_sharding='zero1' requires an initialized "
            "torch.distributed process group"
        )

    from torch.distributed.optim import ZeroRedundancyOptimizer

    return ZeroRedundancyOptimizer(
        params,
        optimizer_class=optimizer_class,
        **defaults,
    )


def _resolve_adam_fused(
    *,
    requested: bool,
    parameter_device_types: set[str],
) -> bool:
    """Validate the opt-in fused implementation for ordinary Adam."""
    if not isinstance(requested, bool):
        raise TypeError(f"adam.fused must be a bool, got {requested!r}")
    if not requested:
        return False
    if parameter_device_types != {"cuda"}:
        devices = ", ".join(sorted(parameter_device_types)) or "none"
        raise ValueError(
            "adam.fused=true requires all trainable parameters to be CUDA "
            f"tensors; found device types: {devices}"
        )
    if "fused" not in inspect.signature(torch.optim.Adam).parameters:
        raise RuntimeError(
            "adam.fused=true requires a PyTorch build whose torch.optim.Adam "
            "supports the fused argument"
        )
    return True


def get_adamw(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
    device_type: str,
    optimizer_sharding: str = "none",
) -> torch.optim.Optimizer:
    """
    Create an AdamW optimizer for the given model with specified parameters.

    Args:
        model (torch.nn.Module): The model for which the optimizer is created.
        weight_decay (float): The weight decay (L2 penalty) for the optimizer.
        learning_rate (float): The learning rate for the optimizer.
        betas (tuple): Coefficients used for computing running averages of gradient and its square.
        device_type (str): The device type ('cuda' or 'cpu') on which the optimizer will operate.

    Returns:
        torch.optim.AdamW: The AdamW optimizer configured with the specified parameters.
    """
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = _build_optimizer(
        optim_groups,
        optimizer_class=torch.optim.AdamW,
        optimizer_sharding=optimizer_sharding,
        lr=learning_rate,
        betas=betas,
        **extra_args,
    )
    print(f"using fused AdamW: {use_fused}")
    print(f"optimizer sharding: {optimizer_sharding}")

    return optimizer


def get_optimizer(
    configs, model: torch.nn.Module, param_names=None
) -> torch.optim.Optimizer:
    print("param_names: ", param_names)
    if param_names is None or len(param_names) == 0 or param_names[0] == "":
        param_names = None
    optimizer_sharding = _get_optimizer_sharding(configs)
    if configs.adam.use_adamw:
        optimizer = get_adamw(
            model=model,
            weight_decay=configs.adam.weight_decay,
            learning_rate=configs.adam.lr,
            betas=(configs.adam.beta1, configs.adam.beta2),
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            optimizer_sharding=optimizer_sharding,
        )
    else:
        trainable_parameter_device_types = {
            parameter.device.type
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        use_fused = _resolve_adam_fused(
            requested=getattr(configs.adam, "fused", False),
            parameter_device_types=trainable_parameter_device_types,
        )
        adam_extra_args = {"fused": True} if use_fused else {}
        if param_names is None or len(param_names) == 0:
            param_groups = [{"params": model.parameters(), "lr": configs.lr}]
        else:
            finetune_params = []
            finetune_params_names = []
            other_params = []
            for n, p in model.named_parameters():
                if any([name in n for name in param_names]):
                    finetune_params.append(p)
                    finetune_params_names.append(n)
                else:
                    other_params.append(p)
            param_groups = [{"params": finetune_params, "lr": configs.finetune.lr}]
            print("Sample Finetune Params Names: ", finetune_params_names[:5])
            if configs.lr > 0.0:
                param_groups.append({"params": other_params, "lr": configs.lr})

        optimizer = _build_optimizer(
            param_groups,
            optimizer_class=torch.optim.Adam,
            optimizer_sharding=optimizer_sharding,
            lr=configs.adam.lr,
            weight_decay=configs.adam.weight_decay,
            betas=(configs.adam.beta1, configs.adam.beta2),
            **adam_extra_args,
        )
        print(f"using fused Adam: {use_fused}")
        print(f"optimizer sharding: {optimizer_sharding}")
        for i, param_group in enumerate(optimizer.param_groups):
            print(
                f"Parameter Group {i}: Learning Rate = {param_group['lr']} Num = {len(param_group['params'])}"
            )
    return optimizer


def is_loss_nan_check(loss: torch.Tensor) -> bool:
    """check the validness of the current loss

    Args:
        loss: the loss from the model

    Returns:
        bool: if True, loss is not nan or inf
    """

    def is_nan(x):
        return torch.isnan(x).any() or torch.isinf(x).any()

    def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM):
        if dist.is_initialized():
            dist.all_reduce(tensor, op=op)
        return tensor

    nan_flag = torch.tensor(
        1.0 if is_nan(loss) else 0.0,
        device=loss.device if torch.cuda.is_available() else None,
    )  # support cpu
    # avoid "Watchdog caught collective operation timeout" error
    all_reduce_tensor(nan_flag)
    if nan_flag.item() > 0.0:
        return True
    return False
