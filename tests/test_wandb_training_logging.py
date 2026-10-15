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

"""Exercise committed history and checkpoint rollback using the real offline SDK."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from runner.wandb_logging import define_training_metrics, log_training_metrics


@unittest.skipUnless(importlib.util.find_spec("wandb"), "wandb SDK is unavailable")
class WandbTrainingLoggingTest(unittest.TestCase):
    def test_immediate_commit_and_checkpoint_rollback(self):
        import wandb
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal.datastore import DataStore

        with tempfile.TemporaryDirectory() as directory:
            run = wandb.init(
                project="training-logging-regression",
                mode="offline",
                dir=directory,
                settings=wandb.Settings(console="off", x_disable_stats=True),
            )
            try:
                define_training_metrics(run)
                metrics = {
                    "train/loss.avg": 1.5,
                    "train/mse_loss.avg": 0.08,
                    "train/smooth_lddt_loss.avg": 0.28,
                    "train/distogram_loss.avg": 1.1,
                    "train/bond_loss.avg": 0.0,
                    "train/weighted_mse_loss.avg": 0.32,
                    "train/weighted_smooth_lddt_loss.avg": 1.12,
                    "train/weighted_distogram_loss.avg": 0.033,
                    "train/weighted_bond_loss.avg": 0.0,
                }
                log_training_metrics(run, metrics, 1039, [1e-5])
                self.assertEqual(run.step, 1, "Record must commit before the next training update")
                log_training_metrics(run, {**metrics, "train/loss.avg": 1.4}, 1000, [1e-5])
                self.assertEqual(run.step, 2, "Checkpoint rollback must not discard the resumed loss")
            finally:
                run.finish()

            store = DataStore()
            store.open_for_scan(str(next(Path(directory).glob("wandb/offline-run-*/*.wandb"))))
            history = []
            while (data := store.scan_data()) is not None:
                record = wandb_internal_pb2.Record()
                record.ParseFromString(data)
                if record.HasField("history"):
                    history.append(
                        {
                            item.key or ".".join(item.nested_key): json.loads(item.value_json)
                            for item in record.history.item
                        }
                    )
            self.assertEqual(len(history), 2)
            self.assertEqual([row["train/step"] for row in history], [1039, 1000])
            self.assertEqual([row["train/loss"] for row in history], [1.5, 1.4])
            self.assertEqual([row["train/loss.avg"] for row in history], [1.5, 1.4])
            self.assertTrue(all(row["train/lr"] == 1e-5 for row in history))
            for key, value in metrics.items():
                self.assertEqual(history[0][key], value)
                self.assertEqual(history[0][key.removesuffix(".avg")], value)


if __name__ == "__main__":
    unittest.main()
