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

"""Commit complete training records and keep checkpoint steps separate from W&B history."""

from collections.abc import Mapping, Sequence
from typing import Any


def define_training_metrics(run: Any) -> None:
    """Use optimizer steps for training plots, including after checkpoint rollback."""
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")


def log_training_metrics(
    run: Any,
    metrics: Mapping[str, float],
    step: int,
    learning_rates: Sequence[float],
) -> None:
    """Publish loss and LR together without waiting for the next logging interval."""
    payload = dict(metrics)
    # Retain existing names while exposing ordinary train/loss and component names.
    payload.update(
        {key.removesuffix(".avg"): value for key, value in metrics.items() if key.endswith(".avg")}
    )
    payload["train/step"] = int(step)
    payload["train/lr"] = float(learning_rates[0])
    payload.update({f"train/group{i}_lr": float(lr) for i, lr in enumerate(learning_rates)})
    # An explicit W&B step defaults to commit=False. It also discards records if
    # training resumes behind the uploaded step. Let W&B advance its own cursor.
    run.log(payload, commit=True)
