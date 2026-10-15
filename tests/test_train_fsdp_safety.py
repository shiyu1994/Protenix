from __future__ import annotations

import os
import random
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from protenix.utils import training as training_utils
import runner.train as train_module
from runner.train import (
    AF3Trainer,
    configure_rank_local_triton_cache,
    synchronize_rank_exception,
    validate_checkpoint_header,
    validate_expected_resume_checkpoint_step,
)


class _MetricSink:
    def add(self, *args, **kwargs) -> None:
        pass


class _CountingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def state_dict(self) -> dict[str, int]:
        return {"steps": self.steps}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.steps = state_dict["steps"]


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure=closure)


class _TrackingScaler:
    """Small CPU scaler double that exposes lifetime and checkpoint state."""

    def __init__(self) -> None:
        self.scale_calls = 0
        self.step_calls = 0
        self.update_calls = 0

    def get_scale(self) -> float:
        return 1.0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        self.scale_calls += 1
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        pass

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.step_calls += 1
        optimizer.step()

    def update(self) -> None:
        self.update_calls += 1

    def is_enabled(self) -> bool:
        return False

    def state_dict(self) -> dict[str, int]:
        return {
            "scale_calls": self.scale_calls,
            "step_calls": self.step_calls,
            "update_calls": self.update_calls,
        }

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.scale_calls = state_dict["scale_calls"]
        self.step_calls = state_dict["step_calls"]
        self.update_calls = state_dict["update_calls"]


class _RuntimeStateLoader:
    def __init__(self, consumed_global_batches: int = 0) -> None:
        self.loaded_state = None
        self.reconstructed_from = None
        self.state = {
            "schema_version": 1,
            "sampler_epoch": consumed_global_batches // 10,
            "next_local_offset": consumed_global_batches % 10,
            "consumed_global_batches": consumed_global_batches,
            "batch_size": 1,
            "sampler": {"test": True},
        }

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state_dict):
        self.loaded_state = dict(state_dict)
        self.state = dict(state_dict)

    def reconstruct_state(self, consumed_global_batches: int):
        self.reconstructed_from = consumed_global_batches
        self.state.update(
            {
                "sampler_epoch": consumed_global_batches // 10,
                "next_local_offset": consumed_global_batches % 10,
                "consumed_global_batches": consumed_global_batches,
            }
        )
        return self.state_dict()


class _FiniteValueNonfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        ctx.shape = value.shape
        return value.new_tensor(1.0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        return (torch.full(ctx.shape, float("inf"), device=grad_output.device),)


def _make_trainer(
    *,
    iters_to_accumulate: int = 1,
    grad_clip_norm: float = 10.0,
) -> AF3Trainer:
    trainer = AF3Trainer.__new__(AF3Trainer)
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        trainer.model.weight.fill_(1.0)
    trainer.optimizer = _CountingSGD(trainer.model.parameters(), lr=0.1)
    trainer.lr_scheduler = _CountingScheduler()
    trainer.train_metric_wrapper = _MetricSink()
    trainer.global_step = 0
    trainer.step = 0
    trainer.start_step = 0
    trainer.iters_to_accumulate = iters_to_accumulate
    trainer._skip_accumulated_update = False
    trainer.device = torch.device("cpu")
    trainer.grad_scaler = torch.GradScaler(device="cpu", enabled=False)
    trainer.configs = SimpleNamespace(
        dtype="fp32",
        grad_clip_norm=grad_clip_norm,
        empty_cache_interval=0,
        optimizer_sharding="none",
    )
    trainer.messages = []
    trainer.numeric_errors = []
    trainer.backward_hook_calls = []
    trainer.print = trainer.messages.append

    def log_numeric_error(self, reason, batch, loss_dict) -> None:
        self.numeric_errors.append(reason)

    def model_forward(self, batch, mode="train"):
        batch = dict(batch)
        output = self.model(batch["x"])

        def record_backward(gradient: torch.Tensor) -> torch.Tensor:
            self.backward_hook_calls.append(gradient.detach().clone())
            return gradient

        output.register_hook(record_backward)
        batch["output"] = output
        return batch, {"N_cycle": 1}

    def get_loss(self, batch, mode="train"):
        output = batch["output"]
        loss_kind = batch.get("loss_kind", "finite")
        if loss_kind == "nan":
            loss = output.sum() * output.new_tensor(float("nan"))
        elif loss_kind == "nonfinite_grad":
            loss = _FiniteValueNonfiniteGradient.apply(output)
        else:
            loss = output.square().mean()
        return loss, {"loss": loss.detach()}, batch

    trainer.log_numeric_error = MethodType(log_numeric_error, trainer)
    trainer.model_forward = MethodType(model_forward, trainer)
    trainer.get_loss = MethodType(get_loss, trainer)
    return trainer


def _run_microbatch(trainer: AF3Trainer, loss_kind: str = "finite"):
    result = trainer.train_step(
        {
            "x": torch.ones(1, 1),
            "loss_kind": loss_kind,
        }
    )
    trainer.global_step += 1
    return result


def _optimizer_config(optimizer_sharding: str) -> SimpleNamespace:
    return SimpleNamespace(
        optimizer_sharding=optimizer_sharding,
        lr=1e-3,
        adam=SimpleNamespace(
            use_adamw=False,
            lr=1e-3,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.99,
        ),
    )


def test_nan_loss_uses_original_graph_and_skips_whole_accumulation_window():
    trainer = _make_trainer(iters_to_accumulate=2)
    initial_weight = trainer.model.weight.detach().clone()

    bad_result = _run_microbatch(trainer, "nan")

    assert bad_result["loss_nan"] is True
    assert bad_result["loss_nonfinite_in_window"] is True
    assert bad_result["timings"]["loss_health_sync_seconds"] == 0.0
    assert bad_result["did_update"] is False
    assert trainer._skip_accumulated_update is True
    assert len(trainer.backward_hook_calls) == 1
    assert trainer.numeric_errors == ["nan_or_inf_loss"]

    boundary_result = _run_microbatch(trainer)

    assert boundary_result["loss_nan"] is False
    assert boundary_result["loss_nonfinite_in_window"] is True
    assert boundary_result["timings"]["loss_health_sync_seconds"] == 0.0
    assert boundary_result["optimizer_skipped"] is True
    assert boundary_result["did_update"] is False
    assert trainer.optimizer.step_calls == 0
    assert trainer.lr_scheduler.steps == 0
    assert trainer._skip_accumulated_update is False
    assert torch.equal(trainer.model.weight, initial_weight)
    assert all(parameter.grad is None for parameter in trainer.model.parameters())

    first_recovery_result = _run_microbatch(trainer)
    recovery_boundary_result = _run_microbatch(trainer)

    assert first_recovery_result["did_update"] is False
    assert recovery_boundary_result["did_update"] is True
    assert recovery_boundary_result["optimizer_skipped"] is False
    assert trainer.optimizer.step_calls == 1
    assert trainer.lr_scheduler.steps == 1
    assert not torch.equal(trainer.model.weight, initial_weight)


def test_loss_health_collective_runs_once_at_accumulation_boundary(monkeypatch):
    trainer = _make_trainer(iters_to_accumulate=4)
    initial_weight = trainer.model.weight.detach().clone()
    all_reduce_calls = []

    monkeypatch.setattr(train_module.dist, "is_initialized", lambda: True)

    def record_all_reduce(flag, op):
        all_reduce_calls.append((int(flag.item()), op))

    monkeypatch.setattr(train_module.dist, "all_reduce", record_all_reduce)

    results = [
        _run_microbatch(trainer, "nan"),
        _run_microbatch(trainer),
        _run_microbatch(trainer),
        _run_microbatch(trainer),
    ]

    assert [result["loss_nan"] for result in results] == [
        True,
        False,
        False,
        False,
    ]
    assert [
        result["loss_nonfinite_in_window"] for result in results
    ] == [True, True, True, True]
    assert [
        result["timings"]["loss_health_sync_seconds"] == 0.0
        for result in results[:3]
    ] == [True, True, True]
    assert results[3]["timings"]["loss_health_sync_seconds"] >= 0.0
    assert len(all_reduce_calls) == 1
    assert all_reduce_calls[0][0] == 1
    assert results[3]["optimizer_skipped"] is True
    assert trainer.optimizer.step_calls == 0
    assert trainer.lr_scheduler.steps == 0
    assert trainer.numeric_errors == ["nan_or_inf_loss"]
    assert torch.equal(trainer.model.weight, initial_weight)
    assert all(parameter.grad is None for parameter in trainer.model.parameters())


def test_accumulation_one_preserves_global_nonfinite_loss_skip(monkeypatch):
    trainer = _make_trainer(iters_to_accumulate=1)
    initial_weight = trainer.model.weight.detach().clone()
    all_reduce_calls = 0

    monkeypatch.setattr(train_module.dist, "is_initialized", lambda: True)

    def inject_remote_nonfinite(flag, op):
        nonlocal all_reduce_calls
        all_reduce_calls += 1
        flag.fill_(1)

    monkeypatch.setattr(train_module.dist, "all_reduce", inject_remote_nonfinite)

    result = _run_microbatch(trainer)

    assert all_reduce_calls == 1
    assert result["loss_nan"] is True
    assert result["loss_nonfinite_in_window"] is True
    assert result["optimizer_skipped"] is True
    assert result["did_update"] is False
    assert trainer.optimizer.step_calls == 0
    assert trainer.lr_scheduler.steps == 0
    assert trainer.numeric_errors == ["nan_or_inf_loss"]
    assert torch.equal(trainer.model.weight, initial_weight)
    assert all(parameter.grad is None for parameter in trainer.model.parameters())


def test_nonfinite_gradient_skips_optimizer_scheduler_and_clears_gradients():
    trainer = _make_trainer(iters_to_accumulate=1)
    initial_weight = trainer.model.weight.detach().clone()

    result = _run_microbatch(trainer, "nonfinite_grad")

    assert result["loss_nan"] is False
    assert result["nonfinite_grad"] is True
    assert result["optimizer_skipped"] is True
    assert result["did_update"] is False
    assert trainer.optimizer.step_calls == 0
    assert trainer.lr_scheduler.steps == 0
    assert trainer.numeric_errors == ["nan_or_inf_gradient"]
    assert torch.equal(trainer.model.weight, initial_weight)
    assert all(parameter.grad is None for parameter in trainer.model.parameters())


def test_enabled_grad_scaler_recovers_after_nonfinite_gradient_window():
    trainer = _make_trainer(iters_to_accumulate=1)
    trainer.grad_scaler = torch.GradScaler(
        device="cpu",
        init_scale=128.0,
        backoff_factor=0.5,
        growth_interval=100,
        enabled=True,
    )
    initial_scale = trainer.grad_scaler.get_scale()
    initial_weight = trainer.model.weight.detach().clone()

    bad_result = _run_microbatch(trainer, "nonfinite_grad")

    backed_off_scale = initial_scale * trainer.grad_scaler.get_backoff_factor()
    assert bad_result["nonfinite_grad"] is True
    assert bad_result["optimizer_skipped"] is True
    assert bad_result["did_update"] is False
    assert bad_result["amp_scale"] == backed_off_scale
    assert trainer.grad_scaler.get_scale() == backed_off_scale
    assert trainer.optimizer.step_calls == 0
    assert trainer.lr_scheduler.steps == 0
    assert torch.equal(trainer.model.weight, initial_weight)

    recovery_result = _run_microbatch(trainer)

    assert recovery_result["nonfinite_grad"] is False
    assert recovery_result["optimizer_skipped"] is False
    assert recovery_result["did_update"] is True
    assert trainer.grad_scaler.get_scale() == backed_off_scale
    assert trainer.optimizer.step_calls == 1
    assert trainer.lr_scheduler.steps == 1
    assert not torch.equal(trainer.model.weight, initial_weight)


def test_train_step_reuses_one_grad_scaler_across_updates():
    trainer = _make_trainer(iters_to_accumulate=1)
    scaler = _TrackingScaler()
    trainer.grad_scaler = scaler

    first_result = _run_microbatch(trainer)
    second_result = _run_microbatch(trainer)

    assert trainer.grad_scaler is scaler
    assert first_result["did_update"] is True
    assert second_result["did_update"] is True
    assert scaler.scale_calls == 2
    assert scaler.step_calls == 2
    assert scaler.update_calls == 2


def test_checkpoint_round_trip_preserves_grad_scaler_state(tmp_path, monkeypatch):
    trainer = _make_trainer()
    trainer.grad_scaler = _TrackingScaler()
    _run_microbatch(trainer)
    trainer.use_fsdp = False
    trainer.use_ddp = False
    trainer.checkpoint_dir = str(tmp_path)
    trainer.checkpoint_index_path = str(tmp_path / "checkpoints.jsonl")
    trainer.latest_checkpoint_path = None
    trainer.latest_ema_checkpoint_path = None
    trainer.update_run_status = lambda **kwargs: None
    monkeypatch.setattr("runner.train.DIST_WRAPPER.rank", 0)

    trainer.save_checkpoint()
    checkpoint_path = tmp_path / f"{trainer.step}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["format_version"] == 2
    assert checkpoint["checkpoint_kind"] == "raw"
    assert checkpoint["grad_scaler"] == trainer.grad_scaler.state_dict()
    assert checkpoint["rank_runtime_states"]["schema_version"] == 1
    assert checkpoint["rank_runtime_states"]["world_size"] == 1
    assert set(checkpoint["rank_runtime_states"]["states"][0]["rng"]) == {
        "python",
        "numpy",
        "torch",
        "cuda",
    }
    assert list(tmp_path.glob("*.tmp.*")) == []

    restored = _make_trainer()
    restored.grad_scaler = _TrackingScaler()
    restored.use_fsdp = False
    restored.use_ddp = False
    restored.configs = SimpleNamespace(
        **vars(restored.configs),
        load_checkpoint_path=str(checkpoint_path),
        load_ema_checkpoint_path="",
        load_strict=True,
        load_params_only=False,
        skip_load_optimizer=False,
        skip_load_step=False,
        skip_load_scheduler=False,
        load_step_for_scheduler=False,
    )

    restored.try_load_checkpoint()

    assert restored.grad_scaler.state_dict() == trainer.grad_scaler.state_dict()
    assert restored.step == trainer.step + 1
    assert restored.global_step == restored.step * restored.iters_to_accumulate


def test_full_resume_restores_rng_and_train_dataloader_state(
    tmp_path, monkeypatch
):
    trainer = _make_trainer()
    trainer.use_fsdp = False
    trainer.use_ddp = False
    trainer.train_dl = _RuntimeStateLoader(consumed_global_batches=1)
    trainer.checkpoint_dir = str(tmp_path)
    trainer.checkpoint_index_path = str(tmp_path / "checkpoints.jsonl")
    trainer.latest_checkpoint_path = None
    trainer.latest_ema_checkpoint_path = None
    trainer.update_run_status = lambda **kwargs: None
    monkeypatch.setattr("runner.train.DIST_WRAPPER.rank", 0)
    monkeypatch.delenv("PROTENIX_RESUME_CHECKPOINT_STEP", raising=False)

    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    trainer.save_checkpoint()
    checkpoint_path = tmp_path / "0.pt"
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(4)

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    restored = _make_trainer()
    restored.use_fsdp = False
    restored.use_ddp = False
    restored.train_dl = _RuntimeStateLoader()
    restored.configs = SimpleNamespace(
        **vars(restored.configs),
        load_checkpoint_path=str(checkpoint_path),
        load_ema_checkpoint_path="",
        load_strict=True,
        load_params_only=False,
        skip_load_optimizer=False,
        skip_load_step=False,
        skip_load_scheduler=False,
        load_step_for_scheduler=False,
    )

    restored.try_load_checkpoint()

    assert restored.train_dl.loaded_state is not None
    assert restored.train_dl.loaded_state["consumed_global_batches"] == 1
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(4), expected_torch)


def test_legacy_full_resume_reconstructs_loader_from_consumed_batches(
    tmp_path, monkeypatch, caplog
):
    trainer = _make_trainer(iters_to_accumulate=2)
    trainer.step = 4
    trainer.global_step = 8
    trainer.use_fsdp = False
    trainer.use_ddp = False
    trainer.checkpoint_dir = str(tmp_path)
    trainer.checkpoint_index_path = str(tmp_path / "checkpoints.jsonl")
    trainer.latest_checkpoint_path = None
    trainer.latest_ema_checkpoint_path = None
    trainer.update_run_status = lambda **kwargs: None
    monkeypatch.setattr("runner.train.DIST_WRAPPER.rank", 0)
    monkeypatch.delenv("PROTENIX_RESUME_CHECKPOINT_STEP", raising=False)
    trainer.save_checkpoint()
    checkpoint_path = tmp_path / "4.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint.pop("rank_runtime_states")
    torch.save(checkpoint, checkpoint_path)

    restored = _make_trainer(iters_to_accumulate=2)
    restored.use_fsdp = False
    restored.use_ddp = False
    restored.train_dl = _RuntimeStateLoader()
    restored.configs = SimpleNamespace(
        **vars(restored.configs),
        load_checkpoint_path=str(checkpoint_path),
        load_ema_checkpoint_path="",
        load_strict=True,
        load_params_only=False,
        skip_load_optimizer=False,
        skip_load_step=False,
        skip_load_scheduler=False,
        load_step_for_scheduler=False,
    )

    with caplog.at_level("WARNING"):
        restored.try_load_checkpoint()

    assert restored.step == 5
    assert restored.global_step == 10
    assert restored.train_dl.reconstructed_from == 10
    assert restored.train_dl.state["sampler_epoch"] == 1
    assert restored.train_dl.state["next_local_offset"] == 0
    assert "Legacy checkpoint" in caplog.text
    assert "RNG streams cannot be recovered" in caplog.text


def test_expected_resume_step_checks_checkpoint_metadata_not_filename(
    monkeypatch,
):
    monkeypatch.setenv("PROTENIX_RESUME_CHECKPOINT_STEP", "499")
    validate_expected_resume_checkpoint_step(
        499,
        checkpoint_path="/checkpoints/misleading-name.pt",
    )
    with pytest.raises(ValueError, match="contains step=498"):
        validate_expected_resume_checkpoint_step(
            498,
            checkpoint_path="/checkpoints/499.pt",
        )


def test_ema_checkpoint_is_tagged_and_rejected_for_full_resume(
    tmp_path, monkeypatch
):
    trainer = _make_trainer()
    trainer.use_fsdp = False
    trainer.use_ddp = False
    trainer.checkpoint_dir = str(tmp_path)
    trainer.checkpoint_index_path = str(tmp_path / "checkpoints.jsonl")
    trainer.latest_checkpoint_path = None
    trainer.latest_ema_checkpoint_path = None
    trainer.update_run_status = lambda **kwargs: None
    monkeypatch.setattr("runner.train.DIST_WRAPPER.rank", 0)

    trainer.save_checkpoint("_ema")
    checkpoint_path = tmp_path / f"{trainer.step}_ema.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["format_version"] == 2
    assert checkpoint["checkpoint_kind"] == "ema"
    assert checkpoint["optimizer"] is None
    assert checkpoint["scheduler"] is None
    assert checkpoint["grad_scaler"] is None
    assert "rank_runtime_states" not in checkpoint
    assert list(tmp_path.glob("*.tmp.*")) == []

    restored = _make_trainer()
    restored.use_fsdp = False
    restored.use_ddp = False
    restored.configs = SimpleNamespace(
        **vars(restored.configs),
        load_checkpoint_path=str(checkpoint_path),
        load_ema_checkpoint_path="",
        load_strict=True,
        load_params_only=False,
        skip_load_optimizer=False,
        skip_load_step=False,
        skip_load_scheduler=False,
        load_step_for_scheduler=False,
    )

    with pytest.raises(ValueError, match="EMA checkpoints are params-only"):
        restored.try_load_checkpoint()


def test_full_resume_rejects_raw_checkpoint_as_ema_companion(
    tmp_path, monkeypatch
):
    trainer = _make_trainer()
    trainer.use_fsdp = False
    trainer.use_ddp = False
    trainer.checkpoint_dir = str(tmp_path)
    trainer.checkpoint_index_path = str(tmp_path / "checkpoints.jsonl")
    trainer.latest_checkpoint_path = None
    trainer.latest_ema_checkpoint_path = None
    trainer.update_run_status = lambda **kwargs: None
    monkeypatch.setattr("runner.train.DIST_WRAPPER.rank", 0)
    trainer.save_checkpoint()
    raw_checkpoint_path = tmp_path / "0.pt"

    restored = _make_trainer()
    restored.use_fsdp = False
    restored.use_ddp = False
    restored.ema_wrapper = SimpleNamespace(register=lambda: None)
    restored.configs = SimpleNamespace(
        **vars(restored.configs),
        load_checkpoint_path="",
        load_ema_checkpoint_path=str(raw_checkpoint_path),
        load_strict=True,
        load_params_only=True,
        skip_load_optimizer=False,
        skip_load_step=False,
        skip_load_scheduler=False,
        load_step_for_scheduler=False,
    )
    monkeypatch.setenv("PROTENIX_FULL_RESUME", "1")

    with pytest.raises(ValueError, match="EMA companion checkpoint"):
        restored.try_load_checkpoint()


def test_full_resume_rejects_ema_companion_with_wrong_internal_step(
    tmp_path, monkeypatch
):
    trainer = _make_trainer()
    trainer.use_fsdp = False
    trainer.use_ddp = False
    trainer.checkpoint_dir = str(tmp_path)
    trainer.checkpoint_index_path = str(tmp_path / "checkpoints.jsonl")
    trainer.latest_checkpoint_path = None
    trainer.latest_ema_checkpoint_path = None
    trainer.update_run_status = lambda **kwargs: None
    monkeypatch.setattr("runner.train.DIST_WRAPPER.rank", 0)
    trainer.save_checkpoint("_ema_0.999")
    ema_checkpoint_path = tmp_path / "0_ema_0.999.pt"

    restored = _make_trainer()
    restored.use_fsdp = False
    restored.use_ddp = False
    restored.ema_wrapper = SimpleNamespace(register=lambda: None)
    restored.configs = SimpleNamespace(
        **vars(restored.configs),
        load_checkpoint_path="",
        load_ema_checkpoint_path=str(ema_checkpoint_path),
        load_strict=True,
        load_params_only=True,
        skip_load_optimizer=False,
        skip_load_step=False,
        skip_load_scheduler=False,
        load_step_for_scheduler=False,
    )
    monkeypatch.setenv("PROTENIX_FULL_RESUME", "1")
    monkeypatch.setenv("PROTENIX_RESUME_CHECKPOINT_STEP", "1")

    with pytest.raises(ValueError, match="contains step=0"):
        restored.try_load_checkpoint()


def test_checkpoint_save_failure_removes_temporary_file(tmp_path, monkeypatch):
    trainer = _make_trainer()
    trainer.use_fsdp = False
    trainer.use_ddp = False
    trainer.checkpoint_dir = str(tmp_path)
    trainer.checkpoint_index_path = str(tmp_path / "checkpoints.jsonl")
    trainer.latest_checkpoint_path = None
    trainer.latest_ema_checkpoint_path = None
    trainer.update_run_status = lambda **kwargs: None
    monkeypatch.setattr("runner.train.DIST_WRAPPER.rank", 0)

    def fail_save(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr("runner.train.torch.save", fail_save)

    with pytest.raises(RuntimeError, match="Checkpoint save failed on rank 0"):
        trainer.save_checkpoint()

    assert not (tmp_path / f"{trainer.step}.pt").exists()
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_fsdp_ema_checkpoint_collects_model_state_only(tmp_path, monkeypatch):
    trainer = _make_trainer()
    trainer.use_fsdp = True
    trainer.use_ddp = False
    trainer.checkpoint_dir = str(tmp_path)
    trainer.checkpoint_index_path = str(tmp_path / "checkpoints.jsonl")
    trainer.latest_checkpoint_path = None
    trainer.latest_ema_checkpoint_path = None
    trainer.update_run_status = lambda **kwargs: None
    monkeypatch.setattr("runner.train.DIST_WRAPPER.rank", 0)

    calls = []

    def collect_model_only(model):
        calls.append("model")
        return {"weight": model.weight.detach().clone()}

    def reject_model_and_optimizer(*args, **kwargs):
        raise AssertionError("EMA save must not collect optimizer state")

    monkeypatch.setattr(
        "runner.train.get_fsdp2_full_model_state_dict", collect_model_only
    )
    monkeypatch.setattr(
        "runner.train.get_fsdp2_full_state_dict", reject_model_and_optimizer
    )

    trainer.save_checkpoint("_ema")

    checkpoint = torch.load(
        tmp_path / f"{trainer.step}_ema.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert calls == ["model"]
    assert checkpoint["checkpoint_kind"] == "ema"
    assert checkpoint["optimizer"] is None


def test_checkpoint_header_rejects_unsafe_full_resume_formats():
    assert (
        validate_checkpoint_header(
            {"format_version": 2, "checkpoint_kind": "raw"},
            checkpoint_path="100.pt",
            load_params_only=False,
        )
        == "raw"
    )
    assert (
        validate_checkpoint_header(
            {},
            checkpoint_path="legacy_raw.pt",
            load_params_only=False,
        )
        == "legacy"
    )
    assert (
        validate_checkpoint_header(
            {"format_version": 2, "checkpoint_kind": "ema"},
            checkpoint_path="100_ema.pt",
            load_params_only=True,
        )
        == "ema"
    )

    with pytest.raises(ValueError, match="format_version"):
        validate_checkpoint_header(
            {"format_version": 3, "checkpoint_kind": "raw"},
            checkpoint_path="100.pt",
            load_params_only=False,
        )
    with pytest.raises(ValueError, match="must define checkpoint_kind"):
        validate_checkpoint_header(
            {"format_version": 2},
            checkpoint_path="100.pt",
            load_params_only=False,
        )
    with pytest.raises(ValueError, match="legacy EMA-named"):
        validate_checkpoint_header(
            {},
            checkpoint_path="100_ema_0.999.pt",
            load_params_only=False,
        )


def test_synchronize_rank_exception_reports_remote_ddp_failure(monkeypatch):
    monkeypatch.setattr("runner.train.dist.is_available", lambda: True)
    monkeypatch.setattr("runner.train.dist.is_initialized", lambda: True)
    monkeypatch.setattr("runner.train.dist.get_rank", lambda: 0)
    monkeypatch.setattr("runner.train.dist.get_world_size", lambda: 2)

    def fake_all_gather_object(records, local_record):
        records[:] = [
            local_record,
            {"rank": 1, "error_type": "OSError", "error": "read failed"},
        ]

    monkeypatch.setattr(
        "runner.train.dist.all_gather_object", fake_all_gather_object
    )

    with pytest.raises(RuntimeError, match="rank 1: OSError: read failed"):
        synchronize_rank_exception("DDP checkpoint load/validation", None)


def test_configure_rank_local_triton_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTENIX_TRITON_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("RANK", "3")
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)

    cache_dir = configure_rank_local_triton_cache()

    assert cache_dir == str(tmp_path / "rank_3")
    assert (tmp_path / "rank_3").is_dir()
    assert os.environ["TRITON_CACHE_DIR"] == cache_dir


def test_invalid_optimizer_sharding_is_rejected():
    model = torch.nn.Linear(2, 1)

    with pytest.raises(ValueError, match="optimizer_sharding"):
        training_utils.get_optimizer(
            _optimizer_config("invalid"),
            model,
        )


def test_zero1_optimizer_sharding_requires_initialized_process_group(monkeypatch):
    model = torch.nn.Linear(2, 1)
    monkeypatch.setattr(training_utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(training_utils.dist, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="initialized.*process group"):
        training_utils.get_optimizer(
            _optimizer_config("zero1"),
            model,
        )


def test_none_optimizer_sharding_returns_normal_optimizer():
    model = torch.nn.Linear(2, 1)

    optimizer = training_utils.get_optimizer(
        _optimizer_config("none"),
        model,
    )

    assert isinstance(optimizer, torch.optim.Adam)
    assert not hasattr(optimizer, "consolidate_state_dict")
