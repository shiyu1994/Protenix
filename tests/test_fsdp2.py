import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import runner.ema as ema_module
from protenix.utils.fsdp import (
    apply_fsdp2,
    get_fsdp2_shard_modules,
    optimizer_state_uses_canonical_fqns,
    reshard_fsdp2_model,
    resolve_fsdp_mesh_shape,
    validate_distributed_strategy,
)
from runner.ema import EMAWrapper


class _BlockContainer(torch.nn.Module):
    def __init__(self, count: int):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2, bias=False) for _ in range(count)]
        )


class _DiffusionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_transformer = _BlockContainer(3)
        self.other = torch.nn.Linear(2, 2, bias=False)


class _ConfidenceHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pairformer_stack = _BlockContainer(2)


class _ToyProtenix(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.msa_module = _BlockContainer(2)
        self.pairformer_stack = _BlockContainer(4)
        self.diffusion_module = _DiffusionModule()
        self.confidence_head = _ConfidenceHead()
        self.root_only = torch.nn.Linear(2, 2, bias=False)


def _linear_with_weight(device: str = "cpu") -> torch.nn.Linear:
    model = torch.nn.Linear(2, 2, bias=False, device=device)
    with torch.no_grad():
        model.weight.copy_(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        )
    return model


def test_fsdp2_wrap_targets_follow_execution_stages_without_duplicates():
    model = _ToyProtenix()

    targets = get_fsdp2_shard_modules(model)

    assert targets == [
        model.msa_module,
        *model.pairformer_stack.blocks,
        *model.diffusion_module.diffusion_transformer.blocks,
        *model.confidence_head.pairformer_stack.blocks,
    ]
    assert len({id(module) for module in targets}) == len(targets)


def test_apply_fsdp2_shards_children_before_root():
    model = _ToyProtenix()
    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))
        return module

    result = apply_fsdp2(
        model,
        mesh="mesh",
        param_dtype="fp32",
        reduce_dtype="fp32",
        reshard_after_forward=True,
        fully_shard_fn=fake_fully_shard,
    )

    targets = get_fsdp2_shard_modules(model)
    assert result is model
    assert [module for module, _ in calls] == [*targets, model]
    assert all(call[1]["mesh"] == "mesh" for call in calls)
    assert all(call[1]["reshard_after_forward"] is True for call in calls)


@pytest.mark.parametrize(
    ("mode", "world_size", "local_world_size", "expected"),
    [
        ("global", 16, 8, (16,)),
        ("hybrid", 16, 8, (2, 8)),
        ("auto", 8, 8, (8,)),
        ("auto", 16, 8, (2, 8)),
    ],
)
def test_resolve_fsdp_mesh_shape(mode, world_size, local_world_size, expected):
    assert resolve_fsdp_mesh_shape(mode, world_size, local_world_size) == expected


@pytest.mark.parametrize("mode", ["invalid", "", "GLOBAL"])
def test_resolve_fsdp_mesh_shape_rejects_unknown_mode(mode):
    with pytest.raises(ValueError, match="auto, global, hybrid"):
        resolve_fsdp_mesh_shape(mode, world_size=8, local_world_size=8)


def test_hybrid_mesh_requires_complete_nodes():
    with pytest.raises(ValueError, match="divisible"):
        resolve_fsdp_mesh_shape("hybrid", world_size=10, local_world_size=8)


@pytest.mark.parametrize(
    ("world_size", "local_world_size"),
    [(0, 8), (8, 0), (-1, 8), (8, -1)],
)
def test_mesh_shape_requires_positive_process_counts(world_size, local_world_size):
    with pytest.raises(ValueError, match="positive"):
        resolve_fsdp_mesh_shape("global", world_size, local_world_size)


def test_fsdp2_and_zero1_are_mutually_exclusive():
    configs = SimpleNamespace(
        distributed_strategy="fsdp2",
        optimizer_sharding="zero1",
    )

    with pytest.raises(ValueError, match="must not be combined"):
        validate_distributed_strategy(configs, world_size=8)


def test_fsdp2_requires_multiple_processes():
    configs = SimpleNamespace(
        distributed_strategy="fsdp2",
        optimizer_sharding="none",
    )

    with pytest.raises(ValueError, match="world_size > 1"):
        validate_distributed_strategy(configs, world_size=1)


def test_unknown_distributed_strategy_is_rejected():
    configs = SimpleNamespace(
        distributed_strategy="fsdp1",
        optimizer_sharding="none",
    )

    with pytest.raises(ValueError, match="ddp.*fsdp2"):
        validate_distributed_strategy(configs, world_size=8)


def test_ddp_default_accepts_zero1():
    configs = SimpleNamespace(
        distributed_strategy="ddp",
        optimizer_sharding="zero1",
    )

    assert validate_distributed_strategy(configs, world_size=8) == "ddp"


def test_optimizer_checkpoint_requires_canonical_parameter_names():
    assert optimizer_state_uses_canonical_fqns(
        {"param_groups": [{"params": ["layer.weight", "layer.bias"]}]}
    )
    assert not optimizer_state_uses_canonical_fqns(
        {"param_groups": [{"params": [0, 1]}]}
    )
    assert not optimizer_state_uses_canonical_fqns({})
    assert not optimizer_state_uses_canonical_fqns({"param_groups": []})
    assert not optimizer_state_uses_canonical_fqns(
        {"param_groups": [{"params": []}]}
    )


def test_reshard_fsdp2_model_reshards_every_fsdp_module():
    class _FakeFSDPModule(torch.nn.Module):
        def __init__(self, child=None):
            super().__init__()
            if child is not None:
                self.child = child
            self.reshard_calls = 0

        def reshard(self):
            self.reshard_calls += 1

    child = _FakeFSDPModule()
    root = _FakeFSDPModule(child)

    reshard_fsdp2_model(root, fsdp_module_type=_FakeFSDPModule)

    assert root.reshard_calls == 1
    assert child.reshard_calls == 1


def test_ema_rejects_unknown_shadow_device():
    with pytest.raises(ValueError, match="shadow_device"):
        EMAWrapper(_linear_with_weight(), shadow_device="disk")


def test_cpu_ema_updates_in_place_and_preserves_parameter_identity():
    model = _linear_with_weight()
    parameter = model.weight
    ema = EMAWrapper(model, decay=0.5, shadow_device="cpu")
    ema.register()
    shadow_data_ptr = ema.shadow["weight"].data_ptr()

    with torch.no_grad():
        model.weight.add_(2.0)
    live_weight = model.weight.detach().clone()
    expected_ema = torch.tensor([[2.0, 3.0], [4.0, 5.0]])

    ema.update()
    assert ema.shadow["weight"].data_ptr() == shadow_data_ptr
    torch.testing.assert_close(ema.shadow["weight"], expected_ema)

    ema.apply_shadow()
    assert model.weight is parameter
    torch.testing.assert_close(model.weight, expected_ema)

    ema.restore()
    assert model.weight is parameter
    assert not ema.backup
    torch.testing.assert_close(model.weight, live_weight)


def test_ema_reshards_before_register_update_apply_and_restore(monkeypatch):
    model = _linear_with_weight()
    resharded_models = []
    monkeypatch.setattr(
        ema_module,
        "reshard_fsdp2_model",
        lambda value: resharded_models.append(value),
    )
    ema = EMAWrapper(model, decay=0.5, shadow_device="cpu")

    ema.register()
    ema.update()
    ema.apply_shadow()
    ema.restore()

    assert resharded_models == [model, model, model, model]


def test_cpu_shadow_tracks_a_plain_local_dtensor_shard():
    if dist.is_initialized():
        pytest.skip("distributed process group already initialized")

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    with tempfile.TemporaryDirectory() as init_dir:
        init_file = f"{init_dir}/process_group_init"
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=0,
            world_size=1,
        )
        try:
            mesh = init_device_mesh("cpu", (1,))
            weight = distribute_tensor(
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                mesh,
                placements=[Shard(0)],
            )

            class _DTensorModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = torch.nn.Parameter(weight)

            model = _DTensorModel()
            parameter = model.weight
            ema = EMAWrapper(model, decay=0.5, shadow_device="cpu")
            ema.register()

            assert isinstance(model.weight, DTensor)
            assert isinstance(ema.shadow["weight"], torch.Tensor)
            assert not isinstance(ema.shadow["weight"], DTensor)
            assert ema.shadow["weight"].device.type == "cpu"
            assert ema.shadow["weight"].shape == model.weight.to_local().shape

            with torch.no_grad():
                model.weight.to_local().add_(2.0)
            live_weight = model.weight.to_local().detach().clone()

            ema.update()
            expected_ema = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
            torch.testing.assert_close(ema.shadow["weight"], expected_ema)

            ema.apply_shadow()
            assert model.weight is parameter
            torch.testing.assert_close(model.weight.to_local(), expected_ema)
            ema.restore()
            assert model.weight is parameter
            assert not ema.backup
            torch.testing.assert_close(model.weight.to_local(), live_weight)
        finally:
            dist.destroy_process_group()
