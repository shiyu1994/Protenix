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

"""Check checkpoint selection and dry-run isolation without GPUs or model loading."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]


class CheckpointLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "training"
        checkpoints = self.run / "checkpoints"
        checkpoints.mkdir(parents=True)
        (self.run / "config.yaml").write_text('data:\n  msa:\n    enable_prot_msa: true\n')
        self.old = checkpoints / "499.pt"
        self.new = checkpoints / "999.pt"
        self.old.write_bytes(b"old")
        self.new.write_bytes(b"new")
        entries = [
            {"path": str(self.old), "step": 499, "is_ema": False},
            {"path": str(self.new), "step": 999, "is_ema": False},
            {"path": str(checkpoints / "missing.pt"), "step": 1499, "is_ema": False},
        ]
        (checkpoints / "checkpoints.jsonl").write_text(
            ''.join(json.dumps(row) + '\n' for row in entries) + '{"step": 1999'
        )
        self.msa = self.root / "msa.json"
        self.msa.write_text('{}')
        resources = self.root / "resources"
        (resources / "checkpoint").mkdir(parents=True)
        (resources / "checkpoint/protenix-v2.pt").write_bytes(b"official")
        self.env = dict(os.environ, PYTHON_BIN=sys.executable, PROTENIX_ROOT_DIR=str(resources))
        self.output = self.root / "evaluation"

    def dry_run(self, *extra):
        return subprocess.run(
            ['bash', str(REPO / 'evaluate_peptide_checkpoint.sh'), '--dry-run',
             '--training-run', str(self.run), '--msa-map', str(self.msa),
             '--output-dir', str(self.output), *extra],
            env=self.env, text=True, capture_output=True, check=False,
        )

    def test_selects_completed_checkpoint_once_without_preparing_or_loading_models(self):
        result = self.dry_run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f'Checkpoint: {self.new}', result.stdout)
        self.assertEqual(result.stdout.count('Command:'), 3)
        self.assertIn(f'--after {self.new}', result.stdout)
        self.assertIn(f'--checkpoint {self.new}', result.stdout)
        self.assertIn('--chunk-size 32', result.stdout)
        self.assertEqual(result.stdout.count('--seeds 101 '), 2)
        self.assertIn('Predictions per complex per checkpoint: 5', result.stdout)
        self.assertFalse(self.output.exists())

    def test_multiple_seeds_reach_both_evaluators_without_launching(self):
        result = self.dry_run('--seeds', '101,102,103,104,105', '--samples', '5')
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line.removeprefix('Command: ')) for line in result.stdout.splitlines()
                    if line.startswith('Command: ') and '--seeds' in line]
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertEqual(command[command.index('--seeds') + 1], '101,102,103,104,105')
        self.assertIn('Predictions per complex per checkpoint: 25', result.stdout)
        self.assertFalse(self.output.exists())

    def test_invalid_seeds_are_rejected_before_preparing_outputs(self):
        for seeds in ('', '101,101', '101,', '-1', str(2**32), 'abc'):
            with self.subTest(seeds=seeds):
                result = self.dry_run('--seeds', seeds)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('--seeds', result.stderr)
                self.assertFalse(self.output.exists())

    def test_resume_rejects_changed_seeds_or_samples_in_either_stage(self):
        for folder, seed_fields, checkpoint_key in (
            ('train_subset', {'seed': 101}, 'after'),
            ('test51', {'seeds': [101]}, 'checkpoint'),
        ):
            with self.subTest(folder=folder):
                stage = self.output / folder
                stage.mkdir(parents=True)
                path = stage / 'run_config.json'
                path.write_text(json.dumps({**seed_fields, 'samples': 5, checkpoint_key: str(self.new)}))
                for extra in (('--seeds', '101,102'), ('--samples', '10')):
                    result = self.dry_run(*extra)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('seeds or samples differ', result.stderr)
                    self.assertNotIn('Command:', result.stdout)
                path.unlink()

    def test_resume_keeps_previous_checkpoint_and_rejects_changing_weights(self):
        train = self.output / 'train_subset'
        train.mkdir(parents=True)
        (train / 'manifest.json').write_text(json.dumps({
            'training_config': str(self.run / 'config.yaml'),
            'selection': {'count': 50}, 'after_checkpoint': str(self.old),
        }))
        result = self.dry_run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f'Checkpoint: {self.old}', result.stdout)
        self.assertEqual(result.stdout.count('Command:'), 2)
        result = self.dry_run('--checkpoint', str(self.new))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('different weights', result.stderr)


if __name__ == '__main__':
    unittest.main()
