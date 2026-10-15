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

"""CPU full-state resume with one-time step rebasing and a fresh data plan."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from protenix.data.pipeline import soft_cluster_schedule
import runner.train as train
from tests.test_strict_full_resume_contract import _configs, _EMAProbe, _RuntimeLoader


class _FreshLoader(_RuntimeLoader):
    def __init__(self, consumed=0):
        super().__init__(consumed=consumed)
        self.reconstructed = []

    def reconstruct_state(self, consumed):
        self.reconstructed.append(consumed)
        # Reconstruction may consume RNG; the trainer must restore RNG last.
        random.random()
        np.random.rand()
        torch.rand(())
        self.state = _RuntimeLoader(consumed=consumed).state_dict()
        self.state["sampler"] = {"fresh_target": True}
        return self.state_dict()


def _assert_state_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_state_equal(a, b)
    else:
        assert left == right


def _trainer(tmp_path, *, raw="", ema=""):
    trainer = train.AF3Trainer.__new__(train.AF3Trainer)
    trainer.model = torch.nn.Linear(2, 1)
    trainer.optimizer = torch.optim.Adam(
        trainer.model.parameters(), lr=0.0009025, betas=(0.9, 0.95), weight_decay=0.01
    )
    trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        trainer.optimizer, step_size=25000, gamma=0.95
    )
    trainer.grad_scaler = torch.GradScaler(device="cpu", enabled=False)
    trainer.configs = _configs(
        load_checkpoint_path=str(raw),
        load_ema_checkpoint_path=str(ema),
        max_steps=170000,
        data=SimpleNamespace(num_dl_workers=0),
    )
    trainer.device = torch.device("cpu")
    trainer.step = trainer.start_step = trainer.global_step = 0
    trainer.iters_to_accumulate = 4
    trainer.train_dl = _FreshLoader()
    trainer.use_fsdp = trainer.use_ddp = False
    trainer.run_name = "numbered_resume_test"
    trainer.run_manifest_path = ""
    trainer.checkpoint_dir = str(tmp_path)
    trainer.checkpoint_index_path = str(tmp_path / "checkpoints.jsonl")
    trainer.latest_checkpoint_path = trainer.latest_ema_checkpoint_path = None
    trainer.update_run_status = lambda **kwargs: None
    trainer.print = lambda message: None
    trainer.ema_wrapper = _EMAProbe(trainer.model)
    return trainer


@pytest.fixture
def source_pair(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTENIX_STRICT_FULL_RESUME", "1")
    monkeypatch.setenv("PROTENIX_FULL_RESUME", "1")
    monkeypatch.setenv("PROTENIX_TRAINING_CONTRACT_ID", "source-contract")
    monkeypatch.setenv("PROTENIX_TRAINING_DATA_PROFILE", "weighted_pdb_only_v1")
    monkeypatch.delenv("PROTENIX_RESUME_CHECKPOINT_STEP", raising=False)
    monkeypatch.delenv("PROTENIX_RESUME_SMOKE_STEPS", raising=False)
    monkeypatch.setenv("PROTENIX_RESET_DATA_SAMPLER", "0")
    monkeypatch.setenv("START_STEP", "68000")
    monkeypatch.setenv("RESUME_STEP", "67999")
    monkeypatch.setattr(train.DIST_WRAPPER, "rank", 0)
    monkeypatch.setattr(train.DIST_WRAPPER, "world_size", 1)
    monkeypatch.setattr(train, "_strict_resume_migration_expectations", lambda cfg: None)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = _trainer(source_dir)
    source.model(torch.ones(1, 2)).sum().backward()
    source.optimizer.step()
    source.optimizer.zero_grad(set_to_none=True)
    source.lr_scheduler.step()
    for state in source.optimizer.state.values():
        state["step"].fill_(48000)
    source.lr_scheduler.last_epoch = 48000
    source.lr_scheduler._step_count = 48001
    source.step = 47999
    source.global_step = 191999
    source.train_dl = _FreshLoader(consumed=192000)
    source.save_checkpoint()
    raw_model = copy.deepcopy(source.model.state_dict())
    with torch.no_grad():
        for parameter in source.model.parameters():
            parameter.add_(0.25)
    source.save_checkpoint("_ema_0.999")
    source.model.load_state_dict(raw_model)
    paths = []
    for suffix in ("", "_ema_0.999"):
        original = source_dir / f"47999{suffix}.pt"
        renamed = tmp_path / f"67999{suffix}.pt"
        shutil.copyfile(original, renamed)
        assert renamed.read_bytes() == original.read_bytes()
        paths.append(renamed)
    return source, paths[0], paths[1]


def _target_plan(monkeypatch, source):
    target = copy.deepcopy(train.build_training_data_contract(source.configs))
    target["profile_id"] = "six-source-fresh-test"
    target["train_sets"] = ["pdb", "legacy", "ge200", "disordered", "sab", "distill-sab"]
    target["train_sample_weights"] = [0.49, 0.005, 0.505, 0.02, 0.02, 0.01]
    target.pop("sha256", None)
    target["sha256"] = hashlib.sha256(
        json.dumps(target, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    monkeypatch.setattr(train, "build_training_data_contract", lambda cfg: copy.deepcopy(target))
    info = SimpleNamespace(
        manifest={"source_checkpoint": {"mode": "fresh"}},
        origin_consumed_global_batches=272000,
    )
    monkeypatch.setattr(soft_cluster_schedule, "load_soft_cluster_schedule_manifest", lambda path: info)
    monkeypatch.setenv("PROTENIX_TRAINING_CONTRACT_ID", "target-contract")
    monkeypatch.setenv("PROTENIX_RESET_DATA_SAMPLER", "1")
    return info


def test_rebased_source_preserves_optimizer_scheduler_ema_and_rng(source_pair, tmp_path, monkeypatch):
    source, raw, ema = source_pair
    original = torch.load(raw, weights_only=False)
    original_ema = torch.load(ema, weights_only=False)
    assert raw.name == "67999.pt" and original["step"] == 47999
    _target_plan(monkeypatch, source)
    restored = _trainer(tmp_path, raw=raw, ema=ema)
    messages = []
    restored.print = messages.append
    restored.try_load_checkpoint()
    assert "Finish loading model parameters" in messages
    assert "Finish loading checkpoint, current step: 68000" in messages
    assert (restored.step, restored.start_step, restored.global_step) == (68000, 68000, 272000)
    assert not hasattr(restored, "offset")
    _assert_state_equal(restored.model.state_dict(), original["model"])
    _assert_state_equal(restored.ema_wrapper.shadow, original_ema["model"])
    _assert_state_equal(restored.optimizer.state_dict(), original["optimizer"])
    _assert_state_equal(restored.lr_scheduler.state_dict(), original["scheduler"])
    assert restored.train_dl.reconstructed == [272000]
    assert restored.train_dl.loaded_state is None
    saved_rng = original["rank_runtime_states"]["states"][0]["rng"]
    assert random.getstate() == saved_rng["python"]
    np_state = np.random.get_state()
    assert np_state[0] == saved_rng["numpy"][0]
    assert np.array_equal(np_state[1], saved_rng["numpy"][1])
    assert np_state[2:] == saved_rng["numpy"][2:]
    assert torch.equal(torch.get_rng_state(), saved_rng["torch"])

    # Training numbering is rebased; Adam and scheduler advance from their
    # original history, and future filename/index/payload step numbers agree.
    for _ in range(25):
        restored.model(torch.ones(1, 2)).sum().backward()
        restored.optimizer.step()
        restored.optimizer.zero_grad(set_to_none=True)
        restored.lr_scheduler.step()
    restored.step = 68024
    restored.global_step = 272099
    restored.train_dl.state = _RuntimeLoader(consumed=272100).state_dict()
    restored.train_dl.state["sampler"] = {"fresh_target": True}
    restored.save_checkpoint()
    restored.save_checkpoint("_ema_0.999")
    saved = torch.load(tmp_path / "68024.pt", weights_only=False)
    assert saved["step"] == saved["resume_contract"]["step"] == 68024
    assert saved["resume_contract"]["global_step"] == 272099
    assert saved["resume_contract"]["next_step"] == 68025
    assert saved["resume_contract"]["next_global_step"] == 272100
    assert saved["scheduler"]["last_epoch"] == 48025
    assert saved["scheduler"]["_step_count"] == 48026
    assert all(state["step"].item() == 48025 for state in saved["optimizer"]["state"].values())
    assert saved["optimizer"]["param_groups"][0]["lr"] == original["optimizer"]["param_groups"][0]["lr"]
    assert "offset" not in saved and "offset" not in saved["resume_contract"]
    index = [json.loads(line) for line in (tmp_path / "checkpoints.jsonl").read_text().splitlines()]
    assert {record["step"] for record in index} == {68024}

    # RESET remains set, but this branch's matching data contract resumes exactly.
    monkeypatch.setenv("PROTENIX_RESUME_SMOKE_STEPS", "25")
    again = _trainer(tmp_path, raw=tmp_path / "68024.pt", ema=tmp_path / "68024_ema_0.999.pt")
    again.try_load_checkpoint()
    assert (again.step, again.start_step, again.global_step) == (68025, 68025, 272100)
    assert not hasattr(again, "offset")
    assert again.train_dl.reconstructed == []
    assert again.train_dl.loaded_state["consumed_global_batches"] == 272100
    assert again.train_dl.loaded_state["sampler"] == {"fresh_target": True}
    assert again.configs.max_steps == 68050  # Restored 68025 + 25, not START_STEP + 25.
    _assert_state_equal(again.optimizer.state_dict(), saved["optimizer"])
    _assert_state_equal(again.lr_scheduler.state_dict(), saved["scheduler"])


def test_data_change_without_explicit_reset_still_rejected(source_pair, monkeypatch):
    source, raw, _ = source_pair
    checkpoint = torch.load(raw, weights_only=False)
    _target_plan(monkeypatch, source)
    monkeypatch.setenv("PROTENIX_RESET_DATA_SAMPLER", "0")
    monkeypatch.setenv("PROTENIX_TRAINING_CONTRACT_ID", "source-contract")
    with pytest.raises(ValueError, match="data contract mismatch"):
        train.validate_strict_checkpoint_payload(
            checkpoint, checkpoint_path=str(raw), expected_kind="raw",
            configs=source.configs, grad_scaler_enabled=False, device=torch.device("cpu"),
        )


def test_rebasing_still_validates_the_original_source_cursor(source_pair, monkeypatch):
    source, raw, _ = source_pair
    checkpoint = torch.load(raw, weights_only=False)
    _target_plan(monkeypatch, source)
    checkpoint["rank_runtime_states"]["states"][0]["train_dataloader"]["consumed_global_batches"] = 272000
    with pytest.raises(ValueError, match="dataloader cursor does not match next_global_step"):
        train.validate_strict_checkpoint_payload(
            checkpoint, checkpoint_path=str(raw), expected_kind="raw",
            configs=source.configs, grad_scaler_enabled=False, device=torch.device("cpu"),
        )


def test_rebased_manifest_distinguishes_source_state_from_new_training_number(
    source_pair, tmp_path, monkeypatch
):
    source, raw, ema = source_pair
    _target_plan(monkeypatch, source)
    restored = _trainer(tmp_path, raw=raw, ema=ema)
    manifest_path = tmp_path / "run_manifest.json"
    train.write_json(manifest_path, {"resume": {}})
    restored.run_manifest_path = str(manifest_path)
    restored.try_load_checkpoint()
    resume = json.loads(manifest_path.read_text())["resume"]
    assert resume["loaded_checkpoint_step"] == 47999
    assert resume["next_training_step"] == 68000
    assert resume["data_profile_id"] == "six-source-fresh-test"
    assert resume["full_state_restored"] is False
    assert resume["fresh_data_sampling"] is True
    assert "optimizer" in resume["restored_state_components"]
    assert "scheduler" in resume["restored_state_components"]
    assert "step_and_global_step" not in resume["restored_state_components"]
    assert "train_sampler_cursor" not in resume["restored_state_components"]


def test_initializing_and_resumed_status_use_checkpoint_numbers(source_pair, tmp_path, monkeypatch):
    _, raw, ema = source_pair
    trainer = _trainer(tmp_path, raw=raw, ema=ema)
    trainer.run_dir = str(tmp_path)
    trainer.run_status_path = str(tmp_path / "status.json")
    trainer.run_started_at = "test"
    trainer.configs.max_steps = 170000
    records = []
    monkeypatch.setattr(train, "build_run_status_record", lambda **kwargs: kwargs)
    monkeypatch.setattr(train, "write_run_status", lambda path, record: records.append(record))
    train.AF3Trainer.update_run_status(trainer, "running", exit_reason="initializing")
    trainer.step = 68000
    train.AF3Trainer.update_run_status(trainer, "running", exit_reason="initialized")
    assert [record["step"] for record in records] == [67999, 68000]
    assert all(record["max_steps"] == 170000 for record in records)


@pytest.mark.parametrize("invalid", ["mode", "origin"])
def test_fresh_reset_requires_fresh_plan_at_requested_start(source_pair, monkeypatch, invalid):
    source, raw, _ = source_pair
    checkpoint = torch.load(raw, weights_only=False)
    info = _target_plan(monkeypatch, source)
    if invalid == "mode":
        info.manifest["source_checkpoint"]["mode"] = "resume"
    else:
        info.origin_consumed_global_batches = 192000
    with pytest.raises(ValueError, match="fresh sampling plan"):
        train.should_reset_training_data(checkpoint["resume_contract"], source.configs)


def test_rebasing_does_not_weaken_save_cursor_check(source_pair, tmp_path, monkeypatch):
    source, raw, ema = source_pair
    _target_plan(monkeypatch, source)
    restored = _trainer(tmp_path, raw=raw, ema=ema)
    restored.try_load_checkpoint()
    restored.global_step = 272003
    restored.train_dl.state = _RuntimeLoader(consumed=272003).state_dict()
    with pytest.raises(RuntimeError, match="progress mismatch"):
        restored.save_checkpoint()
