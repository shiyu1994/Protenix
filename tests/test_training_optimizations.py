import unittest
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runner.ema import EMAWrapper


class TestConfigParserHyphenatedKeys(unittest.TestCase):
    def test_cli_override_preserves_hyphenated_config_keys(self):
        from protenix.config.config import parse_configs

        configs = {
            "data": {
                "sabdab-train": {
                    "base_info": {
                        "indices_fpath": "/default/sabdab-train.csv",
                    },
                },
            },
        }

        parsed = parse_configs(
            configs,
            arg_str=(
                "--data.sabdab-train.base_info.indices_fpath "
                "/tmp/sabdab-train.csv"
            ),
        )

        self.assertEqual(
            parsed.data["sabdab-train"].base_info.indices_fpath,
            "/tmp/sabdab-train.csv",
        )


class TestEMAUpdate(unittest.TestCase):
    def test_update_keeps_shadow_tensor_and_matches_ema_formula(self):
        model = torch.nn.Linear(2, 1)
        wrapper = EMAWrapper(model, decay=0.75)
        wrapper.register()

        old_shadow = {name: tensor.clone() for name, tensor in wrapper.shadow.items()}
        old_ptr = {name: tensor.data_ptr() for name, tensor in wrapper.shadow.items()}

        with torch.no_grad():
            for param in model.parameters():
                param.add_(2.0)

        wrapper.update()

        for name, param in model.named_parameters():
            expected = old_shadow[name] * 0.75 + param.data * 0.25
            self.assertEqual(wrapper.shadow[name].data_ptr(), old_ptr[name])
            self.assertTrue(torch.allclose(wrapper.shadow[name], expected))


class TestEmptyCacheInterval(unittest.TestCase):
    def test_should_empty_cache_only_when_interval_is_positive_and_step_matches(self):
        from runner.train import should_empty_cache

        self.assertFalse(should_empty_cache(step=0, interval=0))
        self.assertFalse(should_empty_cache(step=9, interval=10))
        self.assertTrue(should_empty_cache(step=10, interval=10))
        self.assertTrue(should_empty_cache(step=20, interval=10))


class TestEvaluationScheduling(unittest.TestCase):
    def test_final_eval_can_be_disabled_without_disabling_periodic_eval(self):
        from runner.train import should_run_evaluation

        self.assertTrue(
            should_run_evaluation(
                step_need_eval=True,
                is_last_step=False,
                eval_at_end=False,
            )
        )
        self.assertFalse(
            should_run_evaluation(
                step_need_eval=False,
                is_last_step=True,
                eval_at_end=False,
            )
        )
        self.assertTrue(
            should_run_evaluation(
                step_need_eval=False,
                is_last_step=True,
                eval_at_end=True,
            )
        )


class TestProgressBarSteps(unittest.TestCase):
    def test_eval_interval_zero_uses_single_step_progress_window(self):
        from runner.train import get_progress_bar_steps

        self.assertEqual(get_progress_bar_steps(eval_interval=0, iters_to_accumulate=1), 1)
        self.assertEqual(get_progress_bar_steps(eval_interval=400, iters_to_accumulate=2), 800)


class TestGradientAccumulationSync(unittest.TestCase):
    class _TrackingDdp:
        def __init__(self):
            self.no_sync_calls = 0
            self.active = False

        class _Context:
            def __init__(self, owner):
                self.owner = owner

            def __enter__(self):
                self.owner.active = True

            def __exit__(self, exc_type, exc_value, traceback):
                self.owner.active = False

        def no_sync(self):
            self.no_sync_calls += 1
            return self._Context(self)

    def test_ddp_suppresses_only_nonboundary_microbatch_reduction(self):
        from runner.train import gradient_accumulation_sync_context

        model = self._TrackingDdp()
        with gradient_accumulation_sync_context(
            model,
            use_ddp=True,
            global_step=0,
            iters_to_accumulate=2,
        ):
            self.assertTrue(model.active)
        self.assertEqual(model.no_sync_calls, 1)

        with gradient_accumulation_sync_context(
            model,
            use_ddp=True,
            global_step=1,
            iters_to_accumulate=2,
        ):
            self.assertFalse(model.active)
        self.assertEqual(model.no_sync_calls, 1)

    def test_forced_sync_bypasses_no_sync_at_initial_and_resumed_steps(self):
        from runner.train import gradient_accumulation_sync_context

        for global_step in (0, 100):
            with self.subTest(global_step=global_step):
                model = self._TrackingDdp()
                with gradient_accumulation_sync_context(
                    model,
                    use_ddp=True,
                    global_step=global_step,
                    iters_to_accumulate=2,
                    force_sync=True,
                ):
                    self.assertFalse(model.active)
                self.assertEqual(model.no_sync_calls, 0)

    def test_non_ddp_never_calls_no_sync(self):
        from runner.train import gradient_accumulation_sync_context

        model = self._TrackingDdp()
        with gradient_accumulation_sync_context(
            model,
            use_ddp=False,
            global_step=0,
            iters_to_accumulate=2,
        ):
            self.assertFalse(model.active)
        self.assertEqual(model.no_sync_calls, 0)


class TestExpectedDistributedTopology(unittest.TestCase):
    def test_matching_locked_topology_is_accepted(self):
        from runner.train import validate_expected_distributed_topology

        with mock.patch.dict(
            os.environ,
            {
                "PROTENIX_EXPECTED_WORLD_SIZE": "128",
                "PROTENIX_EXPECTED_LOCAL_WORLD_SIZE": "8",
                "PROTENIX_EXPECTED_NUM_NODES": "16",
            },
            clear=False,
        ):
            validate_expected_distributed_topology(
                world_size=128,
                local_world_size=8,
                num_nodes=16,
            )

    def test_mismatched_child_world_fails_before_process_group_init(self):
        from runner.train import validate_expected_distributed_topology

        with mock.patch.dict(
            os.environ,
            {"PROTENIX_EXPECTED_WORLD_SIZE": "128"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "expected=128, observed=16"):
                validate_expected_distributed_topology(
                    world_size=16,
                    local_world_size=8,
                    num_nodes=2,
                )


class TestFinetuneDemoEvalFlags(unittest.TestCase):
    def test_default_dry_run_includes_sabdab_training_sets(self):
        repo_dir = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "SKIP_PATH_CHECK": "1",
                "NPROC_PER_NODE": "1",
                "LOAD_CHECKPOINT_PATH": "",
                "LOAD_EMA_CHECKPOINT_PATH": "",
            }
        )

        result = subprocess.run(
            ["bash", str(repo_dir / "finetune_demo.sh")],
            cwd=repo_dir,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )

        args = shlex.split(result.stdout)
        train_sets_idx = args.index("--data.train_sets")
        train_weights_idx = args.index("--data.train_sampler.train_sample_weights")

        self.assertEqual(
            args[train_sets_idx + 1],
            "weightedPDB_before210930_v20260101,sabdab-train,distill-sabdab",
        )
        self.assertEqual(args[train_weights_idx + 1], "0.5,0.3,0.2")
        self.assertIn("--data.sabdab-train.base_info.indices_fpath", args)
        self.assertIn("--data.distill-sabdab.base_info.indices_fpath", args)

    def test_dry_run_exposes_independent_eval_flags_and_checkpoint_paths(self):
        repo_dir = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "SKIP_PATH_CHECK": "1",
                "NPROC_PER_NODE": "1",
                "EVAL_AT_END": "false",
                "EVAL_ONLY": "true",
                "EVAL_EMA_ONLY": "true",
                "LOAD_CHECKPOINT_PATH": "/tmp/base.pt",
                "LOAD_EMA_CHECKPOINT_PATH": "/tmp/ema.pt",
            }
        )

        result = subprocess.run(
            ["bash", str(repo_dir / "finetune_demo.sh")],
            cwd=repo_dir,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )

        output = result.stdout
        self.assertIn("--eval_at_end false", output)
        self.assertIn("--eval_only true", output)
        self.assertIn("--eval_ema_only true", output)
        self.assertIn("--load_checkpoint_path /tmp/base.pt", output)
        self.assertIn("--load_ema_checkpoint_path /tmp/ema.pt", output)

    def test_empty_checkpoint_paths_are_omitted_from_dry_run(self):
        repo_dir = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "SKIP_PATH_CHECK": "1",
                "NPROC_PER_NODE": "1",
                "LOAD_CHECKPOINT_PATH": "",
                "LOAD_EMA_CHECKPOINT_PATH": "",
            }
        )

        result = subprocess.run(
            ["bash", str(repo_dir / "finetune_demo.sh")],
            cwd=repo_dir,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )

        output = result.stdout
        self.assertNotIn("--load_checkpoint_path", output)
        self.assertNotIn("--load_ema_checkpoint_path", output)


class TestTrainingJsonlLogging(unittest.TestCase):
    def test_weighted_multi_dataset_annotates_selected_dataset_name(self):
        from protenix.data.pipeline.dataset import WeightedMultiDataset

        class FakeDataset:
            def __init__(self, pdb_id):
                self.pdb_id = pdb_id

            def __getitem__(self, index):
                return {"basic": {"pdb_id": f"{self.pdb_id}{index}"}}

        dataset = WeightedMultiDataset(
            datasets=[FakeDataset("a"), FakeDataset("b")],
            dataset_names=["weightedPDB_before210930_v20260101", "sabdab-train"],
            datapoint_weights=[[1.0], [1.0]],
            dataset_sample_weights=[0.5, 0.5],
        )

        first = dataset[0]
        second = dataset[1]

        self.assertEqual(
            first["basic"]["dataset_name"],
            "weightedPDB_before210930_v20260101",
        )
        self.assertEqual(second["basic"]["dataset_name"], "sabdab-train")

    def test_build_train_dataset_window_stats_groups_losses_and_tokens(self):
        from runner.train import build_train_dataset_window_stats

        records = [
            {
                "dataset_name": "weightedPDB_before210930_v20260101",
                "n_token": 10,
                "n_atom": 100,
                "msa_mode": "query_only",
                "query_only_msa": True,
                "template_use_requested": False,
                "N_template_valid": 0,
                "losses": {"loss": 3.0, "diffusion_loss": 2.0},
            },
            {
                "dataset_name": "sabdab-train",
                "n_token": 20,
                "n_atom": 200,
                "msa_mode": "unpaired_shuffle",
                "query_only_msa": False,
                "pair_msa_as_unpaired_only": True,
                "template_use_requested": True,
                "N_template_valid": 1,
                "losses": {"loss": 5.0, "diffusion_loss": 4.0},
            },
            {
                "dataset_name": "weightedPDB_before210930_v20260101",
                "n_token": 30,
                "n_atom": 300,
                "msa_mode": "species_paired",
                "query_only_msa": False,
                "pair_msa_as_unpaired_only": False,
                "template_use_requested": True,
                "N_template_valid": 0,
                "losses": {"loss": 1.0, "diffusion_loss": 0.5},
            },
        ]

        stats = build_train_dataset_window_stats(records)

        self.assertEqual(stats["total_samples"], 3)
        self.assertEqual(stats["total_tokens"], 60)
        self.assertEqual(
            stats["sample_count_by_dataset"],
            {
                "weightedPDB_before210930_v20260101": 2,
                "sabdab-train": 1,
            },
        )
        self.assertAlmostEqual(
            stats["sample_ratio_by_dataset"]["weightedPDB_before210930_v20260101"],
            2 / 3,
        )
        self.assertAlmostEqual(
            stats["token_ratio_by_dataset"]["weightedPDB_before210930_v20260101"],
            40 / 60,
        )
        self.assertEqual(
            stats["avg_n_token_by_dataset"],
            {
                "weightedPDB_before210930_v20260101": 20.0,
                "sabdab-train": 20.0,
            },
        )
        self.assertEqual(
            stats["loss_avg_by_dataset"]["weightedPDB_before210930_v20260101"],
            {"diffusion_loss": 1.25, "loss": 2.0},
        )
        augmentation = stats["augmentation"]
        self.assertEqual(augmentation["observation_count"], 3)
        self.assertAlmostEqual(augmentation["query_only_msa_ratio"], 1 / 3)
        self.assertEqual(
            augmentation["msa_mode_counts"],
            {
                "query_only": 1,
                "unpaired_shuffle": 1,
                "species_paired": 1,
            },
        )
        self.assertAlmostEqual(augmentation["template_requested_ratio"], 2 / 3)
        self.assertAlmostEqual(augmentation["template_realized_ratio"], 1 / 3)
        self.assertEqual(
            augmentation["template_requested_but_unavailable_count"],
            1,
        )
        self.assertEqual(
            augmentation["joint_mode_counts"],
            {
                "query_only__template_requested": 0,
                "query_only__no_template": 1,
                "full_msa__template_requested": 2,
                "full_msa__no_template": 0,
            },
        )
        self.assertEqual(
            augmentation["msa_template_joint_counts"],
            {
                "query_only__template_requested": 0,
                "query_only__no_template": 1,
                "unpaired_shuffle__template_requested": 1,
                "unpaired_shuffle__no_template": 0,
                "species_paired__template_requested": 1,
                "species_paired__no_template": 0,
            },
        )

    def test_train_dataset_window_stats_accepts_tensor_losses_and_resets(self):
        from runner.train import TrainDatasetWindowStats

        tracker = TrainDatasetWindowStats(need_gather=False)
        tracker.add(
            {"dataset_name": ["distill-sabdab"], "n_token": 8, "n_atom": 64},
            {"loss": torch.tensor(2.5), "ranking_score": torch.tensor(9.0)},
        )

        stats = tracker.calc()
        empty_stats = tracker.calc()

        self.assertEqual(stats["sample_count_by_dataset"], {"distill-sabdab": 1})
        self.assertEqual(stats["loss_avg_by_dataset"]["distill-sabdab"], {"loss": 2.5})
        self.assertEqual(empty_stats["total_samples"], 0)

    def test_build_train_runtime_window_stats_tracks_throughput_memory_and_grad_health(self):
        from runner.train import build_train_runtime_window_stats

        records = [
            {
                "rank": 0,
                "global_step": 0,
                "n_token": 10,
                "n_atom": 100,
                "step_wall_seconds": 2.0,
                "to_device_seconds": 0.1,
                "ema_seconds": 0.01,
                "timings": {
                    "forward_seconds": 0.8,
                    "loss_seconds": 0.2,
                    "backward_seconds": 0.7,
                    "optimizer_seconds": 0.2,
                    "train_step_seconds": 1.9,
                },
                "grad_norm": 5.0,
                "grad_clipped": True,
                "loss_nan": False,
                "nonfinite_grad": False,
                "optimizer_skipped": False,
                "amp_scale": 1024.0,
                "N_sequence_template_cache_chain_dummy_fallback_unique_sequence": 0,
                "N_sequence_template_cache_chain_dummy_fallback_chain": 0,
                "cuda_memory": {
                    "max_allocated_bytes": 100,
                    "max_reserved_bytes": 200,
                    "free_bytes": 900,
                },
            },
            {
                "rank": 1,
                "global_step": 0,
                "n_token": 20,
                "n_atom": 200,
                "step_wall_seconds": 3.0,
                "to_device_seconds": 0.2,
                "ema_seconds": 0.02,
                "timings": {
                    "forward_seconds": 1.0,
                    "loss_seconds": 0.3,
                    "backward_seconds": 1.2,
                    "optimizer_seconds": 0.3,
                    "train_step_seconds": 2.8,
                },
                "grad_norm": float("inf"),
                "grad_clipped": False,
                "loss_nan": True,
                "nonfinite_grad": True,
                "optimizer_skipped": True,
                "amp_scale": 512.0,
                "N_sequence_template_cache_chain_dummy_fallback_unique_sequence": 1,
                "N_sequence_template_cache_chain_dummy_fallback_chain": 2,
                "N_sequence_template_cache_chain_dummy_fallback_reason_invalid_payload": 1,
                "cuda_memory": {
                    "max_allocated_bytes": 150,
                    "max_reserved_bytes": 250,
                    "free_bytes": 800,
                },
            },
        ]

        stats = build_train_runtime_window_stats(records)

        self.assertEqual(stats["total_samples"], 2)
        self.assertEqual(stats["total_tokens"], 30)
        self.assertEqual(stats["total_atoms"], 300)
        self.assertEqual(stats["elapsed_seconds"], 3.0)
        self.assertEqual(stats["slowest_rank"], 1)
        self.assertAlmostEqual(stats["samples_per_sec"], 2 / 3)
        self.assertEqual(stats["timings"]["step_wall_seconds"]["max"], 3.0)
        self.assertEqual(stats["timings"]["step_wall_seconds"]["max_rank"], 1)
        self.assertEqual(stats["memory"]["max_allocated_bytes"]["max"], 150)
        self.assertEqual(stats["memory"]["max_allocated_bytes"]["rank"], 1)
        self.assertEqual(stats["memory"]["free_bytes"]["min"], 800)
        self.assertEqual(stats["memory"]["free_bytes"]["rank"], 1)
        self.assertEqual(stats["grad_health"]["grad_clip_count"], 1)
        self.assertEqual(stats["grad_health"]["loss_nan_count"], 1)
        self.assertEqual(stats["grad_health"]["nonfinite_grad_count"], 1)
        self.assertEqual(stats["grad_health"]["optimizer_skipped_count"], 1)
        self.assertEqual(stats["grad_health"]["grad_norm_max"], 5.0)
        self.assertEqual(stats["grad_health"]["amp_scale_min"], 512.0)
        self.assertEqual(
            stats["template_cache"][
                "chain_dummy_fallback_unique_sequence_count"
            ],
            1,
        )
        self.assertEqual(
            stats["template_cache"][
                "chain_dummy_fallback_affected_chain_count"
            ],
            2,
        )
        self.assertEqual(
            stats["template_cache"]["chain_dummy_fallback_target_rate"],
            0.5,
        )
        self.assertEqual(
            stats["template_cache"]["chain_dummy_fallback_reason_counts"][
                "invalid_payload"
            ],
            1,
        )

    def test_build_train_runtime_window_stats_tracks_data_phases_and_slow_samples(self):
        from runner.train import build_train_runtime_window_stats

        data_timing_keys = [
            "bioassembly_load_seconds",
            "crop_seconds",
            "feature_label_seconds",
            "msa_featurizer_seconds",
            "template_featurizer_seconds",
            "template_load_seconds",
            "template_align_seconds",
            "template_feature_seconds",
            "data_process_seconds",
        ]
        records = [
            {
                "rank": 7,
                "global_step": 11,
                "pdb_id": "fast",
                "dataset_name": "weightedPDB",
                "data_id": "fast-row",
                "n_token": 384,
                "n_atom": 3000,
                "sampler_epoch_boundary": False,
                "data_wait_seconds": 0.25,
                "step_wall_seconds": 4.0,
                **{key: 0.1 for key in data_timing_keys},
            },
            {
                "rank": 123,
                "global_step": 11,
                "pdb_id": "6w2t",
                "dataset_name": "weightedPDB",
                "data_id": "slow-row",
                "n_token": 361,
                "n_atom": 4641,
                "sampler_epoch_boundary": True,
                "data_wait_seconds": 88.0,
                "step_wall_seconds": 8.0,
                **{key: float(index + 1) for index, key in enumerate(data_timing_keys)},
            },
        ]

        stats = build_train_runtime_window_stats(records)

        self.assertEqual(stats["timings"]["data_wait_seconds"]["max"], 88.0)
        self.assertEqual(stats["timings"]["data_wait_seconds"]["max_rank"], 123)
        self.assertEqual(stats["timings"]["data_wait_seconds"]["p50"], 0.25)
        self.assertEqual(stats["timings"]["data_wait_seconds"]["p99"], 88.0)
        self.assertEqual(stats["compute_elapsed_seconds"], 8.0)
        self.assertEqual(stats["end_to_end_elapsed_seconds"], 96.0)
        self.assertEqual(stats["slowest_end_to_end_rank"], 123)
        self.assertAlmostEqual(stats["end_to_end_samples_per_sec"], 2 / 96)
        epoch_comparison = stats["sampler_epoch_boundary_comparison"]
        self.assertEqual(epoch_comparison["boundary_record_count"], 1)
        self.assertEqual(epoch_comparison["non_boundary_record_count"], 1)
        self.assertEqual(
            epoch_comparison["boundary_timings"]["data_wait_seconds"]["max"],
            88.0,
        )
        self.assertEqual(
            epoch_comparison["non_boundary_timings"]["data_wait_seconds"]["max"],
            0.25,
        )
        for index, key in enumerate(data_timing_keys):
            self.assertEqual(stats["timings"][key]["max"], float(index + 1))
            self.assertEqual(stats["timings"][key]["max_rank"], 123)

        slow_wait = stats["top_outliers"]["data_wait_seconds"][0]
        self.assertEqual(slow_wait["data_wait_seconds"], 88.0)
        self.assertEqual(slow_wait["rank"], 123)
        self.assertEqual(slow_wait["global_step"], 11)
        self.assertEqual(slow_wait["pdb_id"], "6w2t")
        self.assertEqual(slow_wait["dataset_name"], "weightedPDB")
        self.assertEqual(slow_wait["data_id"], "slow-row")

        slow_process = stats["top_outliers"]["data_process_seconds"][0]
        self.assertEqual(slow_process["data_process_seconds"], 9.0)
        self.assertEqual(slow_process["rank"], 123)
        self.assertEqual(slow_process["pdb_id"], "6w2t")

    def test_build_train_runtime_window_stats_groups_runtime_and_loss_by_token_bins(self):
        from runner.train import build_train_runtime_window_stats

        records = [
            {
                "rank": 0,
                "global_step": 0,
                "n_token": 64,
                "n_atom": 640,
                "step_wall_seconds": 1.0,
                "timings": {"forward_seconds": 0.4, "backward_seconds": 0.5},
                "losses": {"loss": 2.0, "diffusion_loss": 1.0},
            },
            {
                "rank": 1,
                "global_step": 0,
                "n_token": 300,
                "n_atom": 3000,
                "step_wall_seconds": 3.0,
                "timings": {"forward_seconds": 1.2, "backward_seconds": 1.4},
                "losses": {"loss": 4.0, "diffusion_loss": 2.0},
            },
            {
                "rank": 0,
                "global_step": 1,
                "n_token": 900,
                "n_atom": 9000,
                "step_wall_seconds": 5.0,
                "timings": {"forward_seconds": 2.0, "backward_seconds": 2.5},
                "losses": {"loss": 6.0},
            },
        ]

        stats = build_train_runtime_window_stats(records)

        token_bins = stats["length_bins"]["token_bins"]
        self.assertEqual(
            list(token_bins),
            ["0000_0128", "0129_0256", "0257_0512", "0513_0768", "0769_plus"],
        )
        self.assertEqual(token_bins["0000_0128"]["sample_count"], 1)
        self.assertEqual(token_bins["0000_0128"]["token_count"], 64)
        self.assertEqual(token_bins["0000_0128"]["atom_count"], 640)
        self.assertEqual(token_bins["0000_0128"]["avg_n_token"], 64.0)
        self.assertEqual(
            token_bins["0000_0128"]["loss_avg"],
            {"diffusion_loss": 1.0, "loss": 2.0},
        )
        self.assertEqual(token_bins["0000_0128"]["step_wall_seconds_avg"], 1.0)
        self.assertEqual(token_bins["0000_0128"]["forward_seconds_avg"], 0.4)
        self.assertEqual(token_bins["0000_0128"]["backward_seconds_avg"], 0.5)
        self.assertEqual(token_bins["0129_0256"]["sample_count"], 0)
        self.assertEqual(token_bins["0129_0256"]["loss_avg"], {})
        self.assertIsNone(token_bins["0129_0256"]["step_wall_seconds_avg"])
        self.assertEqual(token_bins["0257_0512"]["loss_avg"]["loss"], 4.0)
        self.assertEqual(token_bins["0769_plus"]["step_wall_seconds_max"], 5.0)

    def test_train_runtime_window_stats_accepts_tensor_values_and_resets(self):
        from runner.train import TrainRuntimeWindowStats

        tracker = TrainRuntimeWindowStats(need_gather=False)
        tracker.add(
            rank=0,
            step=3,
            global_step=5,
            batch_summary={
                "pdb_id": "1abc",
                "dataset_name": "weightedPDB",
                "data_id": "row-5",
                "sample_global_position": torch.tensor(0),
                "n_token": torch.tensor(12),
                "n_atom": torch.tensor(120),
                "data_process_seconds": torch.tensor(0.75),
                "N_msa_shuffle_group": torch.tensor(3),
                "msa_shuffle_requested_variant_code": torch.tensor(2),
                "msa_shuffle_variant_eligible": torch.tensor(True),
                "msa_shuffle_variant_applied": torch.tensor(True),
                "msa_shuffle_whole_physical_chain_applied": torch.tensor(
                    True
                ),
                "N_msa_crop_protein_chain": torch.tensor(3),
                "N_msa_query_only_protein_chain": torch.tensor(2),
                "N_msa_retained_protein_chain": torch.tensor(1),
                "msa_partial_query_only_applied": torch.tensor(True),
                "msa_global_query_only_applied": torch.tensor(False),
                "N_openfold3_representative_query_mismatch": torch.tensor(2),
                "N_openfold3_representative_remap": torch.tensor(2),
                "N_openfold3_representative_query_only_fallback": torch.tensor(
                    0
                ),
                "N_openfold3_explicit_query_only": torch.tensor(1),
                "N_sequence_template_cache_chain_dummy_fallback_unique_sequence": torch.tensor(
                    1
                ),
                "N_sequence_template_cache_chain_dummy_fallback_chain": torch.tensor(
                    2
                ),
                "N_sequence_template_cache_chain_dummy_fallback_reason_missing": torch.tensor(
                    1
                ),
            },
            train_info={
                "grad_norm": torch.tensor(2.0),
                "grad_clipped": False,
                "loss_nan": False,
                "nonfinite_grad": False,
                "optimizer_skipped": False,
                "amp_scale": torch.tensor(1.0),
                "timings": {"forward_seconds": torch.tensor(0.5)},
                "loss_dict": {
                    "loss": torch.tensor(2.5),
                    "ranking_score": torch.tensor(0.1),
                },
            },
            data_wait_seconds=torch.tensor(0.25),
            to_device_seconds=torch.tensor(0.1),
            ema_seconds=torch.tensor(0.0),
            step_wall_seconds=torch.tensor(1.0),
            cuda_memory={"max_allocated_bytes": torch.tensor(64)},
        )

        self.assertEqual(tracker.records[0]["pdb_id"], "1abc")
        self.assertEqual(tracker.records[0]["dataset_name"], "weightedPDB")
        self.assertEqual(tracker.records[0]["data_id"], "row-5")
        self.assertEqual(tracker.records[0]["data_wait_seconds"], 0.25)
        self.assertEqual(tracker.records[0]["data_process_seconds"], 0.75)
        self.assertEqual(
            tracker.records[0]["N_msa_shuffle_group"],
            3,
        )
        self.assertEqual(
            tracker.records[0]["msa_shuffle_requested_variant_code"],
            2,
        )
        self.assertIs(
            tracker.records[0]["msa_shuffle_variant_eligible"],
            True,
        )
        self.assertIs(
            tracker.records[0]["msa_shuffle_variant_applied"],
            True,
        )
        self.assertIs(
            tracker.records[0][
                "msa_shuffle_whole_physical_chain_applied"
            ],
            True,
        )
        self.assertEqual(
            tracker.records[0]["N_msa_crop_protein_chain"],
            3,
        )
        self.assertEqual(
            tracker.records[0]["N_msa_query_only_protein_chain"],
            2,
        )
        self.assertEqual(
            tracker.records[0]["N_msa_retained_protein_chain"],
            1,
        )
        self.assertIs(
            tracker.records[0]["msa_partial_query_only_applied"],
            True,
        )
        self.assertIs(
            tracker.records[0]["msa_global_query_only_applied"],
            False,
        )
        self.assertEqual(
            tracker.records[0][
                "N_openfold3_representative_query_mismatch"
            ],
            2,
        )
        self.assertEqual(
            tracker.records[0]["N_openfold3_representative_remap"],
            2,
        )
        self.assertEqual(
            tracker.records[0][
                "N_openfold3_representative_query_only_fallback"
            ],
            0,
        )
        self.assertEqual(
            tracker.records[0]["N_openfold3_explicit_query_only"],
            1,
        )
        self.assertEqual(
            tracker.records[0][
                "N_sequence_template_cache_chain_dummy_fallback_unique_sequence"
            ],
            1,
        )
        self.assertEqual(
            tracker.records[0][
                "N_sequence_template_cache_chain_dummy_fallback_chain"
            ],
            2,
        )
        self.assertEqual(
            tracker.records[0][
                "N_sequence_template_cache_chain_dummy_fallback_reason_missing"
            ],
            1,
        )
        self.assertTrue(tracker.records[0]["sampler_epoch_boundary"])

        stats = tracker.calc()
        empty_stats = tracker.calc()

        self.assertEqual(stats["total_samples"], 1)
        self.assertEqual(stats["total_tokens"], 12)
        self.assertEqual(stats["grad_health"]["grad_norm_avg"], 2.0)
        self.assertEqual(
            stats["template_cache"][
                "chain_dummy_fallback_unique_sequence_count"
            ],
            1,
        )
        self.assertEqual(
            stats["length_bins"]["token_bins"]["0000_0128"]["loss_avg"],
            {"loss": 2.5},
        )
        self.assertEqual(stats["timings"]["data_wait_seconds"]["max"], 0.25)
        self.assertEqual(
            stats["top_outliers"]["data_wait_seconds"][0]["pdb_id"],
            "1abc",
        )
        outlier = stats["top_outliers"]["data_wait_seconds"][0]
        self.assertEqual(outlier["N_msa_shuffle_group"], 3)
        self.assertEqual(
            outlier["msa_shuffle_requested_variant_code"],
            2,
        )
        self.assertIs(outlier["msa_shuffle_variant_eligible"], True)
        self.assertIs(outlier["msa_shuffle_variant_applied"], True)
        self.assertIs(
            outlier["msa_shuffle_whole_physical_chain_applied"],
            True,
        )
        self.assertEqual(outlier["N_msa_crop_protein_chain"], 3)
        self.assertEqual(outlier["N_msa_query_only_protein_chain"], 2)
        self.assertEqual(outlier["N_msa_retained_protein_chain"], 1)
        self.assertIs(outlier["msa_partial_query_only_applied"], True)
        self.assertIs(outlier["msa_global_query_only_applied"], False)
        self.assertEqual(
            outlier["N_openfold3_representative_query_mismatch"],
            2,
        )
        self.assertEqual(
            outlier["N_openfold3_representative_remap"],
            2,
        )
        self.assertEqual(
            outlier["N_openfold3_representative_query_only_fallback"],
            0,
        )
        self.assertEqual(
            outlier["N_openfold3_explicit_query_only"],
            1,
        )
        self.assertEqual(empty_stats["total_samples"], 0)

    def test_iter_dataloader_with_timing_charges_setup_to_first_batch(self):
        from runner.train import iter_dataloader_with_timing

        with mock.patch(
            "runner.train.time.perf_counter",
            side_effect=[0.0, 1.0, 2.0, 5.0, 6.0, 8.0, 9.0],
        ):
            batches = list(iter_dataloader_with_timing(["first", "second"]))

        self.assertEqual(
            batches,
            [("first", 4.0), ("second", 2.0)],
        )

    def test_append_jsonl_converts_tensors_and_creates_parent_directory(self):
        from runner.train import append_jsonl

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "logs" / "train_metrics.jsonl"
            append_jsonl(path, {"step": torch.tensor(3), "loss": torch.tensor(1.25)})

            records = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(records, [{"step": 3, "loss": 1.25}])

    def test_summarize_batch_records_sample_identity_and_sizes(self):
        from runner.train import summarize_batch

        batch = {
            "basic": {
                "pdb_id": "1abc",
                "dataset_name": "weightedPDB_before210930_v20260101",
                "N_msa": torch.tensor([16384]),
                "N_msa_prot_pair": torch.tensor([1]),
                "N_msa_prot_unpair": torch.tensor([16383]),
                "N_msa_rna_pair": torch.tensor([1]),
                "N_msa_rna_unpair": torch.tensor([0]),
                "N_msa_virtual_segment": torch.tensor([3]),
                "msa_virtual_segment_shuffle_applied": torch.tensor([True]),
                "N_msa_shuffle_group": torch.tensor([4]),
                "msa_shuffle_requested_variant_code": torch.tensor([2]),
                "msa_shuffle_variant_eligible": torch.tensor([True]),
                "msa_shuffle_variant_applied": torch.tensor([True]),
                "msa_shuffle_whole_physical_chain_applied": torch.tensor(
                    [True]
                ),
                "N_msa_crop_protein_chain": torch.tensor([3]),
                "N_msa_query_only_protein_chain": torch.tensor([2]),
                "N_msa_retained_protein_chain": torch.tensor([1]),
                "msa_partial_query_only_applied": torch.tensor([True]),
                "msa_global_query_only_applied": torch.tensor([False]),
                "N_openfold3_representative_query_mismatch": torch.tensor([2]),
                "N_openfold3_representative_remap": torch.tensor([2]),
                "N_openfold3_representative_query_only_fallback": torch.tensor(
                    [0]
                ),
                "N_openfold3_explicit_query_only": torch.tensor([1]),
                "N_sequence_template_cache_chain_dummy_fallback_unique_sequence": torch.tensor(
                    [1]
                ),
                "N_sequence_template_cache_chain_dummy_fallback_chain": torch.tensor(
                    [2]
                ),
                "N_sequence_template_cache_chain_dummy_fallback_reason_unreadable": torch.tensor(
                    [1]
                ),
                "msa_mode": "query_only",
                "query_only_msa": torch.tensor([True]),
                "template_use_requested": torch.tensor([False]),
                "template_dropout_applied": torch.tensor([True]),
                "N_template_valid": torch.tensor([0]),
                "sample_sampler_epoch": torch.tensor([7]),
                "sample_global_position": torch.tensor([91]),
                "N_prot_token": torch.tensor([8]),
                "N_lig_token": torch.tensor([1]),
                "N_rna_token": torch.tensor([2]),
                "N_dna_token": torch.tensor([1]),
                "bioassembly_load_seconds": 0.11,
                "crop_seconds": 0.12,
                "feature_label_seconds": 0.13,
                "msa_featurizer_seconds": 0.14,
                "template_featurizer_seconds": 0.15,
                "template_load_seconds": 0.16,
                "template_align_seconds": 0.17,
                "template_feature_seconds": 0.18,
                "data_process_seconds": 0.19,
            },
            "input_feature_dict": {
                "residue_index": torch.zeros(1, 12),
                "ref_pos": torch.zeros(1, 42, 3),
            },
        }

        summary = summarize_batch(batch)

        self.assertEqual(summary["pdb_id"], "1abc")
        self.assertEqual(summary["dataset_name"], "weightedPDB_before210930_v20260101")
        self.assertEqual(summary["n_token"], 12)
        self.assertEqual(summary["n_atom"], 42)
        self.assertEqual(summary["N_msa"], 16384)
        self.assertEqual(summary["N_msa_prot_pair"], 1)
        self.assertEqual(summary["N_msa_prot_unpair"], 16383)
        self.assertEqual(summary["N_msa_rna_pair"], 1)
        self.assertEqual(summary["N_msa_rna_unpair"], 0)
        self.assertEqual(summary["N_msa_virtual_segment"], 3)
        self.assertEqual(
            summary["msa_virtual_segment_shuffle_applied"],
            True,
        )
        self.assertEqual(summary["N_msa_shuffle_group"], 4)
        self.assertEqual(
            summary["msa_shuffle_requested_variant_code"],
            2,
        )
        self.assertEqual(summary["msa_shuffle_variant_eligible"], True)
        self.assertEqual(summary["msa_shuffle_variant_applied"], True)
        self.assertEqual(
            summary["msa_shuffle_whole_physical_chain_applied"],
            True,
        )
        self.assertEqual(summary["N_msa_crop_protein_chain"], 3)
        self.assertEqual(summary["N_msa_query_only_protein_chain"], 2)
        self.assertEqual(summary["N_msa_retained_protein_chain"], 1)
        self.assertEqual(summary["msa_partial_query_only_applied"], True)
        self.assertEqual(summary["msa_global_query_only_applied"], False)
        self.assertEqual(
            summary["N_openfold3_representative_query_mismatch"],
            2,
        )
        self.assertEqual(summary["N_openfold3_representative_remap"], 2)
        self.assertEqual(
            summary["N_openfold3_representative_query_only_fallback"],
            0,
        )
        self.assertEqual(summary["N_openfold3_explicit_query_only"], 1)
        self.assertEqual(
            summary[
                "N_sequence_template_cache_chain_dummy_fallback_unique_sequence"
            ],
            1,
        )
        self.assertEqual(
            summary[
                "N_sequence_template_cache_chain_dummy_fallback_chain"
            ],
            2,
        )
        self.assertEqual(
            summary[
                "N_sequence_template_cache_chain_dummy_fallback_reason_unreadable"
            ],
            1,
        )
        self.assertEqual(summary["msa_mode"], "query_only")
        self.assertEqual(summary["query_only_msa"], True)
        self.assertEqual(summary["template_use_requested"], False)
        self.assertEqual(summary["template_dropout_applied"], True)
        self.assertEqual(summary["N_template_valid"], 0)
        self.assertEqual(summary["sample_sampler_epoch"], 7)
        self.assertEqual(summary["sample_global_position"], 91)
        self.assertEqual(summary["N_prot_token"], 8)
        self.assertEqual(summary["N_lig_token"], 1)
        self.assertEqual(summary["N_rna_token"], 2)
        self.assertEqual(summary["N_dna_token"], 1)
        self.assertEqual(summary["bioassembly_load_seconds"], 0.11)
        self.assertEqual(summary["crop_seconds"], 0.12)
        self.assertEqual(summary["feature_label_seconds"], 0.13)
        self.assertEqual(summary["msa_featurizer_seconds"], 0.14)
        self.assertEqual(summary["template_featurizer_seconds"], 0.15)
        self.assertEqual(summary["template_load_seconds"], 0.16)
        self.assertEqual(summary["template_align_seconds"], 0.17)
        self.assertEqual(summary["template_feature_seconds"], 0.18)
        self.assertEqual(summary["data_process_seconds"], 0.19)

    def test_build_run_manifest_contains_environment_and_data_summary(self):
        from runner.train import build_run_manifest

        configs = SimpleNamespace(
            run_name="run",
            base_dir="/tmp/out",
            model_name="protenix-v2",
            seed=42,
            c_s=384,
            c_z=256,
            c_s_inputs=449,
            c_atom=128,
            c_atompair=16,
            c_token=384,
            max_atoms_per_token=24,
            no_bins=64,
            model=SimpleNamespace(
                pairformer=SimpleNamespace(
                    n_blocks=64,
                    n_heads=16,
                    c_z=256,
                    c_s=384,
                    hidden_scale_up=True,
                ),
                diffusion_module=SimpleNamespace(
                    c_token=768,
                    c_atom=128,
                    c_atompair=16,
                    c_z=256,
                    c_s=384,
                    c_s_inputs=449,
                    atom_encoder=SimpleNamespace(n_blocks=3, n_heads=4),
                    transformer=SimpleNamespace(n_blocks=24, n_heads=16),
                    atom_decoder=SimpleNamespace(n_blocks=3, n_heads=4),
                ),
            ),
            data=SimpleNamespace(
                train_sets="weightedPDB_before210930_v20260101",
                test_sets="posebusters_0925",
                epoch_size=10240,
                train_sampler=SimpleNamespace(
                    train_sample_weights=[1.0],
                    seed=-1,
                    replacement=True,
                ),
            ),
        )

        with mock.patch.dict(
            os.environ,
            {"PROTENIX_TRAINING_DATA_PROFILE": "weighted_pdb_only_v1"},
            clear=False,
        ):
            manifest = build_run_manifest(
                configs,
                run_name="run_20260101",
                run_dir="/tmp/out/run_20260101",
            )

        self.assertEqual(manifest["run_name"], "run_20260101")
        self.assertEqual(manifest["model_name"], "protenix-v2")
        self.assertEqual(manifest["data"]["train_sets"], "weightedPDB_before210930_v20260101")
        self.assertEqual(manifest["data"]["profile_id"], "weighted_pdb_only_v1")
        self.assertEqual(
            manifest["data"]["contract"]["train_sample_weights"],
            [1.0],
        )
        self.assertEqual(len(manifest["data"]["contract"]["sha256"]), 64)
        self.assertEqual(
            manifest["model"]["architecture_contract"]["pairformer"][
                "n_blocks"
            ],
            64,
        )
        self.assertEqual(
            manifest["model"]["architecture_contract"]["diffusion_module"][
                "c_token"
            ],
            768,
        )
        self.assertEqual(
            len(manifest["model"]["architecture_contract"]["sha256"]),
            64,
        )
        critical_code_paths = {
            "run_crop384_db48_ddp_disordered_v4_from_v8_step999.sh",
            "resume_crop384_db48_ddp_disordered_v4.sh",
            "configs/crop384_db48_data_profiles.sh",
            "protenix/data/msa/msa_featurizer.py",
            "protenix/data/msa/msa_utils.py",
            "protenix/data/template/template_featurizer.py",
            "protenix/data/template/template_utils.py",
            "scripts/runtime/sequence_template_cache_contract.sh",
        }
        self.assertTrue(
            critical_code_paths.issubset(manifest["code_sha256"])
        )
        self.assertTrue(
            all(
                len(manifest["code_sha256"][path]) == 64
                for path in critical_code_paths
            )
        )
        self.assertIn("torch_version", manifest["environment"])
        self.assertIn("command", manifest)

    def test_data_contract_records_effective_per_dataset_augmentation(self):
        from runner.train import build_training_data_contract

        dataset_name = "weightedPDB_before210930_v20260101"
        data = SimpleNamespace(
            train_sets=[dataset_name],
            epoch_size=10240,
            query_only_msa_probability=0.2,
            pair_msa_as_unpaired_probability=0.3,
            template_use_probability=0.2,
            augmentation_decision_version="sample_occurrence_sha256_v1",
            template=SimpleNamespace(template_dropout_rate=0.0),
            train_sampler=SimpleNamespace(
                train_sample_weights=[1.0],
                seed=42,
                replacement=True,
            ),
            **{
                dataset_name: SimpleNamespace(
                    augmentation=SimpleNamespace(template_use_probability=0.6),
                    template=SimpleNamespace(template_dropout_rate=0.1),
                )
            },
        )

        with mock.patch.dict(
            os.environ,
            {"PROTENIX_TRAINING_DATA_PROFILE": "test_profile"},
            clear=False,
        ):
            contract = build_training_data_contract(SimpleNamespace(data=data))

        self.assertEqual(contract["schema_version"], 3)
        self.assertNotIn("loss_and_crop_policy", contract)
        self.assertEqual(
            contract["augmentation"]["msa_mode_policy"],
            "categorical_query_only_unpaired_shuffle_species_paired_v1",
        )
        self.assertEqual(
            contract["augmentation"]["msa_mode_order"],
            ["query_only", "unpaired_shuffle", "species_paired"],
        )
        self.assertAlmostEqual(
            contract["augmentation"]["species_paired_msa_probability"],
            0.5,
        )
        self.assertEqual(
            contract["augmentation"]["unpaired_shuffle_flags"],
            {
                "pair_msa_as_unpaired_only": True,
                "shuffle_pair_msa_when_unpaired_only": True,
            },
        )
        self.assertEqual(
            contract["augmentation"]["effective_by_dataset"][dataset_name],
            {
                "query_only_msa_probability": 0.2,
                "pair_msa_as_unpaired_probability": 0.3,
                "species_paired_msa_probability": 0.5,
                "template_use_probability": 0.6,
                "legacy_template_dropout_rate": 0.1,
            },
        )

    def test_manifest_recognizes_crop384_msa2048_dataset_alias(self):
        from runner.train import build_run_manifest

        dataset_name = "benchmark_long_protein_context_384_800_msa2048"
        dataset = {
            "base_info": {"indices_fpath": "/tmp/benchmark/indices.csv"},
            "cropping_configs": {
                "crop_size": 384,
                "max_nuc_tokens": 128,
            },
        }
        configs = SimpleNamespace(
            model_name="protenix-v2",
            seed=42,
            train_crop_size=384,
            data=SimpleNamespace(
                train_sets=[dataset_name],
                test_sets=[dataset_name],
                **{dataset_name: dataset},
            ),
        )
        with mock.patch.dict(
            os.environ,
            {"BENCHMARK_DATASET_NAME": dataset_name},
            clear=False,
        ):
            manifest = build_run_manifest(
                configs,
                run_name="crop384",
                run_dir="/tmp/out/crop384",
            )

        self.assertEqual(manifest["benchmark"]["dataset_name"], dataset_name)
        self.assertEqual(
            manifest["data"]["benchmark_dataset"]["name"], dataset_name
        )
        self.assertEqual(
            manifest["data"]["benchmark_dataset"]["cropping_configs"][
                "max_nuc_tokens"
            ],
            128,
        )

    def test_to_jsonable_converts_configdict_like_values_to_mappings(self):
        from runner.train import to_jsonable

        class ConfigDictLike:
            def to_dict(self):
                return {"mesh": "global", "reshard_after_forward": True}

            def __str__(self):
                return "mesh: wrong"

        self.assertEqual(
            to_jsonable(ConfigDictLike()),
            {"mesh": "global", "reshard_after_forward": True},
        )

    def test_build_checkpoint_record_marks_ema_checkpoints(self):
        from runner.train import build_checkpoint_record

        record = build_checkpoint_record("/tmp/checkpoints/3_ema_0.999.pt", step=3, ema_suffix="_ema_0.999")

        self.assertEqual(record["path"], "/tmp/checkpoints/3_ema_0.999.pt")
        self.assertEqual(record["step"], 3)
        self.assertTrue(record["is_ema"])
        self.assertEqual(record["ema_suffix"], "_ema_0.999")

    def test_build_run_status_record_captures_exception_details(self):
        from runner.train import build_run_status_record

        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            record = build_run_status_record(
                status="failed",
                run_name="run_20260101",
                run_dir="/tmp/out/run_20260101",
                step=7,
                global_step=9,
                max_steps=20,
                latest_checkpoint="/tmp/out/run_20260101/checkpoints/7.pt",
                latest_ema_checkpoint="/tmp/out/run_20260101/checkpoints/7_ema_0.999.pt",
                exit_reason="exception",
                exception=exc,
            )

        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["last_step"], 7)
        self.assertEqual(record["last_global_step"], 9)
        self.assertEqual(record["latest_checkpoint"], "/tmp/out/run_20260101/checkpoints/7.pt")
        self.assertEqual(record["latest_ema_checkpoint"], "/tmp/out/run_20260101/checkpoints/7_ema_0.999.pt")
        self.assertEqual(record["exception"]["type"], "RuntimeError")
        self.assertEqual(record["exception"]["message"], "boom")
        self.assertIn("RuntimeError: boom", record["exception"]["traceback"])

    def test_write_run_status_and_heartbeat_persist_json_records(self):
        from runner.train import append_run_heartbeat, build_run_status_record, write_run_status

        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = Path(tmpdir) / "manifest" / "run_status.json"
            heartbeat_path = Path(tmpdir) / "logs" / "heartbeat_rank0.jsonl"
            record = build_run_status_record(
                status="completed",
                run_name="run_20260101",
                run_dir="/tmp/out/run_20260101",
                step=20,
                global_step=20,
                max_steps=20,
                latest_checkpoint="/tmp/out/run_20260101/checkpoints/20.pt",
                latest_ema_checkpoint=None,
                exit_reason="max_steps_reached",
            )

            write_run_status(status_path, record)
            append_run_heartbeat(heartbeat_path, record)

            status_record = json.loads(status_path.read_text())
            heartbeat_records = [
                json.loads(line) for line in heartbeat_path.read_text().splitlines()
            ]

        self.assertEqual(status_record["status"], "completed")
        self.assertEqual(status_record["exit_reason"], "max_steps_reached")
        self.assertEqual(heartbeat_records, [status_record])


class TestEvalDiagnosticsLogging(unittest.TestCase):
    def test_eval_aggregator_names_include_distribution_stats(self):
        from runner.train import get_eval_aggregator_names

        self.assertEqual(
            get_eval_aggregator_names(),
            ["avg", "median", "pct90", "min", "max"],
        )

    def test_build_eval_sample_record_keeps_key_metrics(self):
        from runner.train import build_eval_sample_record

        record = build_eval_sample_record(
            test_name="posebusters_0925",
            ema_suffix="ema0.999_",
            step=3,
            global_step=3,
            rank=0,
            world_size=1,
            batch_index=5,
            total_batches=10,
            sample={"pdb_id": "1abc", "n_token": 128},
            metrics={
                "loss": torch.tensor(4.5),
                "lddt/complex/mean": torch.tensor(0.25),
                "lddt/complex/ranking_score.rank1": torch.tensor(0.75),
                "lddt/complex/plddt.rank1": torch.tensor(0.65),
                "lddt/complex/gpde.rank1": torch.tensor(0.35),
            },
        )

        self.assertEqual(record["event"], "eval_sample")
        self.assertTrue(record["is_ema"])
        self.assertEqual(record["sample"]["pdb_id"], "1abc")
        self.assertEqual(record["key_metrics"]["loss"], 4.5)
        self.assertEqual(record["key_metrics"]["lddt_complex_mean"], 0.25)
        self.assertEqual(record["key_metrics"]["ranking_score"], 0.75)
        self.assertAlmostEqual(record["key_metrics"]["plddt"], 0.65)
        self.assertAlmostEqual(record["key_metrics"]["gpde"], 0.35)

    def test_select_worst_eval_samples_tracks_high_loss_and_low_lddt(self):
        from runner.train import select_worst_eval_samples

        records = [
            {
                "sample": {"pdb_id": "good"},
                "key_metrics": {"loss": 1.0, "lddt_complex_mean": 0.9},
            },
            {
                "sample": {"pdb_id": "bad_loss"},
                "key_metrics": {"loss": 9.0, "lddt_complex_mean": 0.5},
            },
            {
                "sample": {"pdb_id": "bad_lddt"},
                "key_metrics": {"loss": 2.0, "lddt_complex_mean": 0.1},
            },
        ]

        summary = select_worst_eval_samples(records, n=2)

        self.assertEqual(
            [item["sample"]["pdb_id"] for item in summary["highest_loss"]],
            ["bad_loss", "bad_lddt"],
        )
        self.assertEqual(
            [item["sample"]["pdb_id"] for item in summary["lowest_lddt_complex_mean"]],
            ["bad_lddt", "bad_loss"],
        )


if __name__ == "__main__":
    unittest.main()
