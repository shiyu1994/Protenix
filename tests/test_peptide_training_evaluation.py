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

"""Training-set RMSD checks independent of model weights and CUDA."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HAS_DEPS = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "scipy", "biotite"))
if HAS_DEPS:
    import biotite.structure as struc
    import numpy as np
    import torch
    from scripts.evaluation.evaluate_peptide_training import (
        feature_digest, make_dataset, predict_seeds, read_complete_jsonl, reconstruct_training_draws,
        score_interface, select_predictions,
    )


@unittest.skipUnless(HAS_DEPS, "Requires scientific Python dependencies")
class TrainingPoseTests(unittest.TestCase):
    def setUp(self):
        self.atoms = struc.AtomArray(7)
        self.atoms.chain_id = np.array(["A"] * 4 + ["L"] * 3)
        self.atoms.atom_name = np.array(["CA"] * 4 + ["C000", "O001", "O002"])
        self.atoms.coord = np.array(
            [[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4], [1, 1, 1], [2, 1, 1], [1, 2, 1]],
            dtype=np.float32,
        )
        self.atoms.set_annotation("mol_atom_index", np.array([0, 1, 2, 3, 0, 1, 2]))
        self.atoms.set_annotation("entity_mol_id", np.array([0] * 4 + [1] * 3))
        self.atoms.set_annotation("is_protein", np.array([True] * 4 + [False] * 3))
        self.atoms.set_annotation("is_resolved", np.ones(7, dtype=bool))
        self.features = {"atom_perm_list": [[0]] * 4 + [[0, 0], [1, 2], [2, 1]]}
        self.metadata = {"receptor_chain": "A", "peptide_chain": "L"}

    def score(self, coords):
        return score_interface(self.atoms, self.features, self.metadata, coords)

    def test_common_rigid_transform(self):
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        result = self.score(self.atoms.coord @ rotation + [10, -3, 20])
        for key in ("peptide_rmsd", "peptide_shape_rmsd", "receptor_rmsd"):
            self.assertAlmostEqual(result[key], 0, places=6)

    def test_peptide_translation_is_not_aligned_away(self):
        coordinates = self.atoms.coord.copy()
        coordinates[4:] += [10, 0, 0]
        result = self.score(coordinates)
        self.assertAlmostEqual(result["peptide_rmsd"], 10)
        self.assertAlmostEqual(result["peptide_shape_rmsd"], 0)

    def test_custom_ligand_symmetry(self):
        coordinates = self.atoms.coord.copy()
        coordinates[[5, 6]] = coordinates[[6, 5]]
        self.assertAlmostEqual(self.score(coordinates)["peptide_rmsd"], 0)

    def test_nonfinite_and_missing_receptor_rejected(self):
        coordinates = self.atoms.coord.copy()
        coordinates[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.score(coordinates)
        self.atoms.is_resolved[:3] = False
        with self.assertRaises(ValueError):
            self.score(self.atoms.coord)

    def test_feature_fingerprint_detects_changed_input(self):
        f = {"scalar": torch.tensor(1.0), "tensor": torch.ones(2, 3), "perms": [[0, 1]]}
        original = feature_digest(f)
        self.assertEqual(original, feature_digest(f))
        f["tensor"][1, 1] = 2
        self.assertNotEqual(original, feature_digest(f))

    def test_multiple_seeds_pool_predictions_and_preserve_shared_features(self):
        calls = []
        current_seed = None
        features = {**self.features, "marker": torch.tensor([1.0])}

        def seed_everything(seed, deterministic):
            nonlocal current_seed
            current_seed = seed
            calls.append(seed)

        def predict(data):
            self.assertEqual(data['input_feature_dict']['marker'].item(), 1.0)
            data['input_feature_dict']['marker'].fill_(99)
            coords = np.repeat(self.atoms.coord[None], 2, axis=0)
            shifts = [7, 5] if current_seed == 102 else [3, 1]
            ranks = [0.9, 0.5] if current_seed == 102 else [0.8, 0.7]
            for xyz, shift in zip(coords, shifts):
                xyz[4:] += [shift, 0, 0]
            return {"coordinate": torch.tensor(coords),
                    "summary_confidence": [{"ranking_score": rank} for rank in ranks]}

        with patch('protenix.utils.seed.seed_everything', side_effect=seed_everything), patch(
            'torch.cuda.empty_cache'
        ), patch('builtins.print'):
            before, xyz_before = predict_seeds(
                SimpleNamespace(predict=predict), features, self.atoms, self.metadata, [102, 101], 2, 'before',
            )
            after, xyz_after = predict_seeds(
                SimpleNamespace(predict=predict), features, self.atoms, self.metadata, [102, 101], 2, 'after',
            )
        self.assertEqual(calls, [102, 101, 102, 101])
        self.assertEqual(before, after)
        np.testing.assert_array_equal(xyz_before, xyz_after)
        self.assertEqual(xyz_before.shape, (4, 7, 3))
        self.assertEqual([(s['seed'], s['sample']) for s in before['samples']],
                         [(102, 0), (102, 1), (101, 0), (101, 1)])
        self.assertEqual((before['top1']['seed'], before['top1']['sample']), (102, 0))
        self.assertEqual((before['best']['seed'], before['best']['sample']), (101, 1))
        self.assertAlmostEqual(before['best']['peptide_rmsd'], 1.0)
        self.assertEqual(features['marker'].item(), 1.0)

    def test_multiseed_selection_ties_use_seed_then_sample(self):
        scores = [{"seed": seed, "sample": sample, "ranking_score": 0.8, "peptide_rmsd": 2.0}
                  for seed, sample in [(102, 0), (101, 1), (101, 0)]]
        result = select_predictions(scores)
        self.assertEqual(result['top1'], scores[-1])
        self.assertEqual(result['best'], scores[-1])

    def test_partial_seed_predictions_are_rejected(self):
        prediction = {"coordinate": torch.tensor(self.atoms.coord[None]),
                      "summary_confidence": [{"ranking_score": 0.9}]}
        with patch('protenix.utils.seed.seed_everything'), patch('builtins.print'):
            with self.assertRaisesRegex(ValueError, 'expected 2 predictions'):
                predict_seeds(SimpleNamespace(predict=lambda data: prediction), self.features,
                              self.atoms, self.metadata, [101, 102], 2, 'fixture')

    def test_cli_seed_list_and_legacy_seed(self):
        from scripts.evaluation import evaluate_peptide_training as driver

        base = ['evaluate_peptide_training', 'prepare', '--training-config', '/tmp/config.yaml',
                '--output-dir', '/tmp/not_created']
        for options, expected in [([], [101]), (['--seed', '9'], [9]),
                                  (['--seeds', '102,101'], [102, 101])]:
            with self.subTest(options=options), patch('sys.argv', base + options), patch.object(
                driver, 'prepare'
            ) as prepare:
                driver.main()
                self.assertEqual(prepare.call_args.args[0].seeds, expected)
        for options in (['--seeds', '101,101'], ['--seeds', ''], ['--seeds', '-1'],
                        ['--seeds', str(2**32)], ['--seed', '1', '--seeds', '2,3']):
            with self.subTest(options=options), patch('sys.argv', base + options), patch.object(
                driver, 'prepare'
            ) as prepare, patch('sys.stderr'):
                with self.assertRaises(SystemExit) as error:
                    driver.main()
                self.assertEqual(error.exception.code, 2)
                prepare.assert_not_called()

    def test_worker_saves_verifiable_multiseed_coordinates_and_resumes(self):
        from scripts.evaluation import evaluate_peptide_training as driver
        from scripts.evaluation import verify_peptide_training_results as verifier
        from scripts.evaluation.peptide_pose_inputs import file_hash, write_json

        self.atoms.set_annotation('peptide_source_atom_index', np.arange(len(self.atoms)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'cache.pkl.gz'
            cache.write_bytes(b'cache mocked for CPU verification')
            case = {'name': 'fixture', 'index': 0, 'pdb_id': 'fixture', 'crop_seed': 42,
                    'training_exposures': 1, 'structure_training_exposures': 1, 'interface_type': 'ligand_prot'}
            manifest = {'cases': [case], 'selection': {'count': 1}, 'training_config': str(root / 'config.yaml'),
                        'seen_structures': 1, 'seen_interfaces': 1, 'structures': 1,
                        'training_samples': 1, 'logged_samples_verified': 1, 'crop_config': {'crop_size': 1024}}
            write_json(root / 'manifest.json', manifest)
            write_json(root / 'run_config.json', {'seeds': [102, 101], 'samples': 2,
                                                  'manifest_sha256': file_hash(root / 'manifest.json')})
            metadata = {**self.metadata, 'tokens': 7, 'msa': {'rows': 1},
                        'source_atom_indices': list(range(7)), 'feature_sha256': feature_digest(self.features),
                        'cache_path': str(cache), 'cache_sha256': file_hash(cache)}
            write_json(root / 'final_crop_validation.json',
                       [{'name': 'fixture', 'status': 'ok', 'metadata': metadata}])
            args = SimpleNamespace(output_dir=root, before=Path('/tmp/before.pt'), after=Path('/tmp/after.pt'),
                                   seeds=[102, 101], samples=2, cycles=10, steps=200, chunk_size=32,
                                   worker_id=0, workers=1, limit=0)
            coordinates = np.repeat(self.atoms.coord[None], 2, axis=0)
            coordinates[0, 4:] += [3, 0, 0]
            prediction = {'coordinate': torch.tensor(coordinates),
                          'summary_confidence': [{'ranking_score': 0.9}, {'ranking_score': 0.8}]}
            runner = SimpleNamespace(predict=lambda data: prediction)
            config = SimpleNamespace(data=SimpleNamespace(msa=SimpleNamespace(enable_prot_msa=True)))
            with patch.dict('sys.modules', {'runner.inference': SimpleNamespace(InferenceRunner=lambda c: runner)}), \
                 patch('scripts.evaluation.evaluate_peptide_pose.model_configs', return_value=None), \
                 patch('scripts.evaluation.evaluate_peptide_pose.configure_gpu_memory_limit'), \
                 patch.object(driver, 'make_dataset', return_value=(None, config)), \
                 patch.object(driver, 'make_crop', return_value=(self.features, self.atoms, metadata)) as crop, \
                 patch('protenix.utils.seed.seed_everything') as set_seed, \
                 patch('torch.cuda.empty_cache'), patch('builtins.print'):
                driver.worker(args)
                driver.worker(args)
                self.assertEqual(crop.call_count, 1)
                self.assertEqual([c.args[0] for c in set_seed.call_args_list], [102, 101, 102, 101])
            result = json.loads((root / 'cases/fixture.json').read_text())
            self.assertEqual(result['status'], 'complete', result.get('error'))
            self.assertEqual(len(result['before']['samples']), 4)
            with np.load(root / 'cases/fixture.npz') as saved:
                np.testing.assert_array_equal(saved['prediction_seeds'], [102, 102, 101, 101])
                np.testing.assert_array_equal(saved['prediction_samples'], [0, 1, 0, 1])
                self.assertEqual(saved['before'].shape, (4, 7, 3))
            with patch.object(verifier, 'load_gzip_pickle', return_value={'atom_array': self.atoms}), \
                 patch.object(verifier, 'Featurizer') as featurizer, patch('builtins.print'):
                featurizer.return_value.get_atom_permutation_list.return_value = self.features['atom_perm_list']
                report = verifier.verify(root)
            self.assertEqual(report['verified_predictions'], 8)
            summary = driver.summarize(root)
            self.assertEqual(summary['complete_interfaces'], 1)
            self.assertEqual(summary['samples_expected_per_model'], 4)
            self.assertEqual(summary['seeds'], [102, 101])
            self.assertIn('best-of-4', (root / 'comparison.md').read_text())
            self.assertIn('before_top1_seed', (root / 'per_interface.csv').read_text())

    def test_live_jsonl_ignores_only_incomplete_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live.jsonl"
            path.write_text('{"step": 9}\n{"step":')
            self.assertEqual(read_complete_jsonl(path), [{"step": 9}])
            path.write_text('{"step": 9}\ninvalid\n')
            with self.assertRaises(json.JSONDecodeError):
                read_complete_jsonl(path)

    def test_multi_epoch_exposure_matches_training_sampler_and_stops_at_checkpoint(self):
        import pandas as pd
        from ml_collections import ConfigDict
        from protenix.data.pipeline.dataloader import DistributedWeightedSampler

        class Dataset:
            indices_list = pd.DataFrame({"pdb_id": [f"p{i}" for i in range(7)]})

            def __len__(self):
                return len(self.indices_list)

        dataset = Dataset()
        config = ConfigDict({
            "seed": 42, "max_steps": 100, "iters_to_accumulate": 1,
            "data": {"train_sets": ["peptide"], "epoch_size": 8,
                     "peptide": {"sampler_configs": {"sampler_type": "uniform"}},
                     "train_sampler": {"seed": -1, "replacement": True}},
        })
        expected, observed = [], []
        for epoch in range(2):
            ranks = []
            for rank in range(2):
                sampler = DistributedWeightedSampler(
                    dataset, torch.ones(7), num_samples=8, num_replicas=2,
                    rank=rank, seed=42, emit_sample_occurrence=True,
                )
                sampler.set_epoch(epoch)
                ranks.append(list(sampler))
            for offset in range(4):
                step = epoch * 4 + offset
                if step > 5:
                    break
                for rank in range(2):
                    occurrence = ranks[rank][offset]
                    expected.append(occurrence.index)
                    observed.append({"global_step": step, "rank": rank, "world_size": 2,
                                     "sample": {"pdb_id": f"p{occurrence.index}",
                                                "sample_sampler_epoch": epoch,
                                                "sample_global_position": offset * 2 + rank}})
        actual, seed = reconstruct_training_draws(dataset, config, observed, 5)
        self.assertEqual(seed, 42)
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 12)
        observed[-1]["sample"]["pdb_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "recorded training example"):
            reconstruct_training_draws(dataset, config, observed, 5)

    def test_msa_enabled_config_uses_training_featurizer(self):
        import yaml

        config = {
            "data": {"train_sets": ["peptide"], "peptide": {"base_info": {}, "cropping_configs": {}},
                     "msa": {"enable_prot_msa": True, "enable_rna_msa": False,
                             "max_size": {"train": 512}},
                     "template": {"enable_prot_template": False}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            with patch("protenix.data.pipeline.dataset.BaseSingleDataset") as dataset_factory, patch(
                "protenix.data.pipeline.dataset.get_msa_featurizer"
            ) as msa_factory:
                _, loaded = make_dataset(path)
                self.assertTrue(loaded.data.msa.enable_prot_msa)
                self.assertEqual(msa_factory.call_args.args[1:], ("peptide", "train"))
                self.assertIs(dataset_factory.call_args.kwargs["msa_featurizer"], msa_factory.return_value)

    def test_evaluation_respects_explicit_chunk_size_with_msas(self):
        from types import SimpleNamespace
        from scripts.evaluation.evaluate_peptide_pose import model_configs

        args = SimpleNamespace(output_dir=Path('/tmp/eval'), checkpoint=Path('/tmp/checkpoint.pt'),
                               cycles=10, steps=200, samples=5, chunk_size=32)
        config = model_configs(args, Path('/tmp/input.json'), True)
        self.assertTrue(config.use_msa)
        self.assertEqual(config.infer_setting.chunk_size, 32)
        self.assertFalse(config.infer_setting.dynamic_chunk_size)


if __name__ == "__main__":
    unittest.main()
