import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark.summarize_aistudio_benchmark import (
    IMMUNE_HEAVY_CHECKPOINT,
    IMMUNE_HEAVY_CHECKPOINT_SHA256,
    REQUIRED_CODE_HASH_PATHS,
    _summarize_compute_apps,
    benchmark_crop_profile,
    benchmark_initialization_expectation,
    benchmark_workload_profile,
    build_summary,
    render_markdown,
    validate_compute_process_topology,
    validate_msa_depth_coverage,
    validate_token_composition_coverage,
)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class TestComputeProcessClassification(unittest.TestCase):
    @staticmethod
    def _write_compute_rows(
        path: Path, rows: list[list[object]], *, include_role: bool = True
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            header = [
                "epoch_seconds",
                "node_rank",
                "hostname",
                "pid",
                "gpu_uuid",
                "used_memory_mib",
                "process_name",
                "is_benchmark_process",
            ]
            if include_role:
                header.append("process_role")
            writer.writerow(header)
            writer.writerows(rows)

    def test_proven_auxiliary_context_does_not_change_rank_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "compute_apps.csv"
            rows = []
            for gpu_index in range(2):
                rows.extend(
                    [
                        [
                            99.0,
                            0,
                            "host0",
                            100 + gpu_index,
                            f"gpu{gpu_index}",
                            1000,
                            "[Not Found]",
                            1,
                            "namespace_unresolved",
                        ],
                        [
                            101.0,
                            0,
                            "host0",
                            100 + gpu_index,
                            f"gpu{gpu_index}",
                            2000,
                            "[Not Found]",
                            1,
                            "namespace_unresolved",
                        ],
                        [
                            101.0,
                            0,
                            "host0",
                            200 + gpu_index,
                            f"gpu{gpu_index}",
                            10,
                            "python",
                            1,
                            "benchmark_descendant",
                        ],
                    ]
                )
            self._write_compute_rows(path, rows)

            summary = _summarize_compute_apps(
                [path],
                measurement_start=100.0,
                measurement_end=102.0,
            )
            expected_gpus = {(0, "gpu0"), (0, "gpu1")}
            self.assertEqual(len(summary["training_rank_processes"]), 2)
            self.assertEqual(len(summary["auxiliary_processes"]), 2)
            self.assertEqual(summary["unclassified_processes"], [])
            self.assertEqual(
                validate_compute_process_topology(
                    summary,
                    world_size=2,
                    expected_compute_gpus=expected_gpus,
                ),
                [],
            )

    def test_later_namespace_unresolved_context_is_not_silently_allowed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "compute_apps.csv"
            rows = []
            for gpu_index in range(2):
                rows.extend(
                    [
                        [
                            99.0,
                            0,
                            "host0",
                            100 + gpu_index,
                            f"gpu{gpu_index}",
                            1000,
                            "[Not Found]",
                            1,
                            "namespace_unresolved",
                        ],
                        [
                            101.0,
                            0,
                            "host0",
                            100 + gpu_index,
                            f"gpu{gpu_index}",
                            2000,
                            "[Not Found]",
                            1,
                            "namespace_unresolved",
                        ],
                        [
                            101.0,
                            0,
                            "host0",
                            300 + gpu_index,
                            f"gpu{gpu_index}",
                            3000,
                            "[Not Found]",
                            1,
                            "namespace_unresolved",
                        ],
                    ]
                )
            self._write_compute_rows(path, rows)

            summary = _summarize_compute_apps(
                [path],
                measurement_start=100.0,
                measurement_end=102.0,
            )
            errors = validate_compute_process_topology(
                summary,
                world_size=2,
                expected_compute_gpus={(0, "gpu0"), (0, "gpu1")},
            )
            self.assertEqual(len(summary["training_rank_processes"]), 2)
            self.assertEqual(len(summary["unclassified_processes"]), 2)
            self.assertIn(
                "unclassified additional GPU compute contexts overlap the "
                "measurement window",
                errors,
            )

    def test_wrong_world_size_or_duplicate_rank_gpu_still_fails(self):
        compute_apps = {
            "training_rank_processes": [
                {"node_rank": 0, "gpu_uuid": "gpu0", "pid": 100},
                {"node_rank": 0, "gpu_uuid": "gpu0", "pid": 101},
            ],
            "auxiliary_processes": [],
            "unclassified_processes": [],
        }
        errors = validate_compute_process_topology(
            compute_apps,
            world_size=2,
            expected_compute_gpus={(0, "gpu0"), (0, "gpu1")},
        )
        self.assertIn(
            "multiple training ranks were classified on the same GPU", errors
        )
        self.assertTrue(
            any(
                error.startswith(
                    "training-rank GPU coverage differs from measured GPUs"
                )
                for error in errors
            )
        )

        errors = validate_compute_process_topology(
            {
                "training_rank_processes": [
                    {"node_rank": 0, "gpu_uuid": "gpu0", "pid": 100}
                ],
                "auxiliary_processes": [],
                "unclassified_processes": [],
            },
            world_size=2,
            expected_compute_gpus={(0, "gpu0"), (0, "gpu1")},
        )
        self.assertIn(
            "measurement window must contain exactly one training rank "
            "per world-size slot: 1 != 2",
            errors,
        )


class TestAIStudioBenchmarkSummary(unittest.TestCase):
    def test_combines_runner_node_and_measurement_nvml_results(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            official_root = Path(
                "/mnt/project_abd/public_databases/protenix.20260101"
            )
            official_checkpoint = str(official_root / "checkpoint/protenix-v2.pt")
            official_source = str(
                official_root
                / "indices/indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz"
            )
            _write_json(
                root / "train/run/benchmark/benchmark_result.json",
                {
                    "schema_version": 1,
                    "world_size": 2,
                    "warmup_steps": 1,
                    "measure_steps": 2,
                    "started_at_epoch": 100.0,
                    "completed_at_epoch": 102.0,
                    "elapsed_seconds": 2.0,
                    "samples_per_sec": 2.0,
                    "tokens_per_sec": 200.0,
                    "atoms_per_sec": 400.0,
                    "total_samples": 4,
                    "total_tokens": 3072,
                    "total_atoms": 4000,
                    "rank_elapsed_imbalance": 1.01,
                    "max_allocated_bytes": 8 * 1024**3,
                    "max_reserved_bytes": 9 * 1024**3,
                    "benchmark_variant": "ddp_full",
                    "distributed_strategy": "ddp",
                    "optimizer_sharding": "none",
                    "fsdp": {
                        "mesh": "auto",
                        "param_dtype": "fp32",
                        "reduce_dtype": "fp32",
                        "reshard_after_forward": True,
                    },
                    "resolved_fsdp_mesh": None,
                    "ema": {
                        "enabled": True,
                        "decay": 0.999,
                        "shadow_device": "cpu",
                    },
                    "load_params_only": True,
                    "skip_final_checkpoint": True,
                    "max_loss_nonfinite_count_per_rank": 0,
                    "max_gradient_nonfinite_count_per_rank": 0,
                    "max_optimizer_skipped_count_per_rank": 0,
                    "initial_checkpoint": {
                        "model": official_checkpoint,
                        "ema": official_checkpoint,
                    },
                    "topology": {
                        "world_size": 2,
                        "local_world_size": 1,
                        "num_nodes": 2,
                    },
                    "rank_records": [
                        {
                            "rank": 0,
                            "local_rank": 0,
                            "node_rank": 0,
                            "elapsed_seconds": 1.9,
                            "sample_count": 2,
                            "token_count": 1536,
                            "atom_count": 2000,
                            "min_token_count": 768,
                            "max_token_count": 768,
                            "workload_sha256": "0" * 64,
                        },
                        {
                            "rank": 1,
                            "local_rank": 0,
                            "node_rank": 1,
                            "elapsed_seconds": 2.0,
                            "sample_count": 2,
                            "token_count": 1536,
                            "atom_count": 2000,
                            "min_token_count": 768,
                            "max_token_count": 768,
                            "workload_sha256": "1" * 64,
                        },
                    ],
                    "model_parameters": {
                        "total": 464442431,
                        "trainable": 464441841,
                        "frozen": 590,
                        "fixed_parameters": {
                            "diffusion_module.diffusion_conditioning.fourier_embedding.w": 256,
                            "diffusion_module.diffusion_conditioning.fourier_embedding.b": 256,
                            "confidence_head.lower_bins": 39,
                            "confidence_head.upper_bins": 39,
                        },
                    },
                },
            )
            _write_json(
                root / "train/run/manifest/run_status.json",
                {"status": "completed"},
            )
            _write_json(
                root / "train/run/manifest/run_manifest.json",
                {
                    "cwd": str(root),
                    "code_sha256": {
                        path: "a" * 64 for path in REQUIRED_CODE_HASH_PATHS
                    },
                    "model_name": "protenix-v2",
                    "seed": 42,
                    "training": {
                        "load_params_only": True,
                        "dtype": "bf16",
                        "ema_shadow_device": "cpu",
                        "ema_decay": 0.999,
                        "train_crop_size": 768,
                        "diffusion_batch_size": 24,
                        "diffusion_chunk_size": 4,
                        "iters_to_accumulate": 1,
                        "lr": 0.00001,
                        "lr_warmup_steps": 2000,
                        "triangle_attention": "cuequivariance",
                        "triangle_multiplicative": "cuequivariance",
                        "n_cycle": 4,
                        "diffusion_n_step": 20,
                    },
                    "benchmark": {
                        "variant": "ddp_full",
                        "warmup_steps": 1,
                        "measure_steps": 2,
                        "exact_crop": "true",
                        "expected_torch_cuda": "12.6",
                        "official_checkpoint_sha256": (
                            "8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599"
                        ),
                        "skip_final_checkpoint": True,
                    },
                    "model": {
                        "c_z": 256,
                        "msa_c_m": 128,
                        "pairformer_n_blocks": 48,
                        "pairformer_blocks_per_ckpt": 2,
                        "pairformer_transition_chunk_size": 128,
                        "pairformer_transition_checkpoint_chunks": True,
                        "diffusion_transition_chunk_size": 128,
                        "diffusion_transition_checkpoint_chunks": True,
                    },
                    "data": {
                        "train_sets": ["benchmark_long_protein_context_768"],
                        "test_sets": ["benchmark_long_protein_context_768"],
                        "num_dl_workers": 4,
                        "epoch_size": 10000,
                        "train_sample_weights": [1.0],
                        "msa": {
                            "min_size": {"train": 16},
                            "max_size": {"train": 16384},
                            "sample_cutoff": {"train": 32},
                            "pair_msa_as_unpaired_only": True,
                            "shuffle_pair_msa_when_unpaired_only": True,
                        },
                        "benchmark_dataset": {
                            "name": "benchmark_long_protein_context_768",
                            "base_info": {
                                "min_n_token": 769,
                                "mmcif_dir": str(official_root / "mmcif"),
                                "bioassembly_dict_dir": str(
                                    official_root / "mmcif_bioassembly"
                                ),
                            },
                            "cropping_configs": {
                                "crop_size": 768,
                                "method_weights": [0.0, 1.0, 0.0],
                                "max_nuc_tokens": -1,
                                "contiguous_crop_complete_lig": False,
                                "spatial_crop_complete_lig": False,
                                "drop_last": False,
                                "remove_metal": False,
                            },
                            "artifacts": {
                                "index_sha256": (
                                    "6bdf93c0e8f5a6a527c50112516a2f3c1cd27d4c0bfc825b5a7465fdadf5fd6a"
                                ),
                                "selection_summary": {
                                    "schema_version": 1,
                                    "source_index": official_source,
                                    "source_index_sha256": (
                                        "e9fc812d2250c8d19a995dbd38991ac05204d2ace5e1e4878d4760e708ad2bc1"
                                    ),
                                    "index_sha256": (
                                        "6bdf93c0e8f5a6a527c50112516a2f3c1cd27d4c0bfc825b5a7465fdadf5fd6a"
                                    ),
                                    "max_rows": 128,
                                    "filter_min_num_tokens": 769,
                                    "filter_max_num_tokens": 1536,
                                    "selection": "top_tokens",
                                    "seed": 42,
                                    "rows": 128,
                                },
                            },
                        },
                    },
                    "environment": {
                        "cuda_version": "12.6",
                        "cc": "/mnt/project_abd/klp/runtime/protenix_cuda126_toolkit/bin/x86_64-conda-linux-gnu-gcc",
                        "cxx": "/mnt/project_abd/klp/runtime/protenix_cuda126_toolkit/bin/x86_64-conda-linux-gnu-g++",
                        "torch_lib_dir": str(root / "torch_lib"),
                        "ld_library_path": (
                            f"{root / 'torch_lib'}:/usr/local/cuda/lib64"
                        ),
                        "tmpdir": "/tmp/ptx_test",
                        "torch_multiprocessing_sharing_strategy": "file_system",
                        "torch_extensions_dir": str(root / "torch_extensions"),
                        "fused_layer_norm_extension": str(
                            root / "torch_extensions/fast_layer_norm_cuda_v2.so"
                        ),
                        "module_locations": {
                            "configs": str(root / "configs/__init__.py"),
                            "protenix": str(root / "protenix/__init__.py"),
                        },
                    },
                    "initial_checkpoint": {
                        "model": official_checkpoint,
                        "ema": official_checkpoint,
                    },
                    "distributed": {
                        "world_size": 2,
                        "strategy": "ddp",
                        "optimizer_sharding": "none",
                        "fsdp": {
                            "mesh": "auto",
                            "param_dtype": "fp32",
                            "reduce_dtype": "fp32",
                            "reshard_after_forward": True,
                        },
                    },
                },
            )
            config_path = root / "train/run/config.yaml"
            config_path.write_text("model_name: protenix-v2\n", encoding="utf-8")
            (root / "configs").mkdir()
            (root / "configs/__init__.py").touch()
            (root / "protenix").mkdir()
            (root / "protenix/__init__.py").touch()
            (root / "torch_extensions").mkdir()
            (root / "torch_extensions/fast_layer_norm_cuda_v2.so").touch()
            (root / "torch_lib").mkdir()
            metrics_path = root / "train/run/logs/train_metrics.jsonl"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(
                    {
                        "runtime_window": {
                            "grad_health": {
                                "loss_nan_count": 0,
                                "nonfinite_grad_count": 0,
                                "optimizer_skipped_count": 0,
                            }
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            for node_rank in range(2):
                node = root / f"nodes/node_{node_rank}_host{node_rank}"
                _write_json(
                    node / "node_status.json",
                    {
                        "benchmark_id": "bench",
                        "benchmark_variant": "ddp_full",
                        "node_rank": node_rank,
                        "hostname": f"host{node_rank}",
                        "expected_nodes": 2,
                        "poll_seconds": 0.5,
                        "exit_code": 0,
                        "wall_seconds": 10 + node_rank,
                    },
                )
                with (node / "nvml.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        [
                            "epoch_seconds",
                            "node_rank",
                            "hostname",
                            "gpu_index",
                            "gpu_uuid",
                            "gpu_name",
                            "driver_version",
                            "memory_used_mib",
                            "memory_total_mib",
                            "utilization_gpu_pct",
                        ]
                    )
                    writer.writerow(
                        [99.0, node_rank, f"host{node_rank}", 0, f"gpu{node_rank}", "A100", "570.1", 5, 80, 1]
                    )
                    writer.writerow(
                        [101.0, node_rank, f"host{node_rank}", 0, f"gpu{node_rank}", "A100", "570.1", 20 + node_rank, 80, 90]
                    )
                with (node / "system.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        [
                            "epoch_seconds",
                            "node_rank",
                            "hostname",
                            "mem_available_kib",
                            "load_1m",
                        ]
                    )
                    writer.writerow([101.0, node_rank, f"host{node_rank}", 1000, 2.0])
                with (node / "compute_apps.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        [
                            "epoch_seconds",
                            "node_rank",
                            "hostname",
                            "pid",
                            "gpu_uuid",
                            "used_memory_mib",
                            "process_name",
                            "is_benchmark_process",
                        ]
                    )
                    writer.writerow(
                        [
                            101.0,
                            node_rank,
                            f"host{node_rank}",
                            1000 + node_rank,
                            f"gpu{node_rank}",
                            20,
                            "python",
                            1,
                        ]
                    )
                with (node / "preflight_compute_apps.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        [
                            "node_rank",
                            "hostname",
                            "pid",
                            "gpu_uuid",
                            "used_memory_mib",
                            "process_name",
                        ]
                    )
                (node / "code_tree.sha256").write_text(
                    "".join(
                        f"{'a' * 64}  {path}\n"
                        for path in sorted(REQUIRED_CODE_HASH_PATHS)
                    ),
                    encoding="utf-8",
                )
                (node / "artifact_hashes.sha256").write_text(
                    "".join(
                        [
                            f"8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599  {official_checkpoint}\n",
                            f"e9fc812d2250c8d19a995dbd38991ac05204d2ace5e1e4878d4760e708ad2bc1  {official_source}\n",
                            f"6bdf93c0e8f5a6a527c50112516a2f3c1cd27d4c0bfc825b5a7465fdadf5fd6a  {root}/benchmark/indices.csv\n",
                            f"{'b' * 64}  {root}/benchmark/summary.json\n",
                        ]
                    ),
                    encoding="utf-8",
                )

            summary = build_summary(root)

            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["nodes"]["observed"], 2)
            self.assertEqual(
                summary["nvml_device"]["measurement_window"][
                    "hottest_gpu_peak_mib"
                ],
                21,
            )
            self.assertIn("Tokens/s: 200.0", render_markdown(summary))
            self.assertIn("Distributed strategy: ddp", render_markdown(summary))

            runner_path = root / "train/run/benchmark/benchmark_result.json"
            invalid_runner = json.loads(runner_path.read_text(encoding="utf-8"))
            invalid_runner["ema"]["enabled"] = False
            invalid_runner["rank_records"][0].pop("workload_sha256")
            _write_json(runner_path, invalid_runner)
            invalid_summary = build_summary(root)
            self.assertEqual(invalid_summary["status"], "failed")
            self.assertIn(
                "matched DDP/FSDP2 benchmark requires EMA to be enabled",
                invalid_summary["validation_errors"],
            )
            self.assertTrue(
                any(
                    "workload SHA256 is missing or invalid" in error
                    for error in invalid_summary["validation_errors"]
                )
            )

    def test_missing_runner_result_fails_validation_without_crashing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            summary = build_summary(Path(temporary_directory))
            self.assertEqual(summary["status"], "failed")
            self.assertIn(
                "runner benchmark_result.json not found",
                summary["validation_errors"],
            )

    def test_crop_profiles_are_strict_and_keep_the_same_source_index(self):
        crop384, profile384 = benchmark_crop_profile(
            384,
            "benchmark_long_protein_context_384_800_msa2048",
        )
        crop768, profile768 = benchmark_crop_profile(768)
        crop1024, profile1024 = benchmark_crop_profile("1024")

        self.assertEqual(crop384, 384)
        self.assertEqual(profile384["min_num_tokens"], 1025)
        self.assertEqual(profile384["rows"], 800)
        self.assertIs(profile384["replacement"], False)
        self.assertEqual(crop768, 768)
        self.assertEqual(crop1024, 1024)
        self.assertEqual(profile768["min_num_tokens"], 769)
        self.assertEqual(profile1024["min_num_tokens"], 1025)
        self.assertEqual(profile1024["dataset_name"], "benchmark_long_protein_context_1024")
        self.assertEqual(profile768["index_sha256"], profile1024["index_sha256"])
        crop800, profile800 = benchmark_crop_profile(
            1024,
            "benchmark_long_protein_context_1024_800",
        )
        self.assertEqual(crop800, 1024)
        self.assertEqual(profile800["rows"], 800)
        self.assertEqual(profile800["epoch_size"], 800)
        self.assertIs(profile800["replacement"], False)
        self.assertIs(profile800["random_sample_if_failed"], False)
        self.assertNotEqual(profile800["index_sha256"], profile1024["index_sha256"])
        _, msa2048_profile = benchmark_crop_profile(
            1024,
            "benchmark_long_protein_context_1024_800_msa2048",
        )
        self.assertEqual(msa2048_profile["rows"], 800)
        self.assertNotEqual(
            msa2048_profile["index_sha256"], profile800["index_sha256"]
        )
        crop960, profile960 = benchmark_crop_profile(
            960,
            "benchmark_long_protein_context_960_800_msa2048",
        )
        self.assertEqual(crop960, 960)
        self.assertEqual(profile960["min_num_tokens"], 1025)
        self.assertEqual(profile960["rows"], 800)
        self.assertEqual(
            profile960["index_sha256"], msa2048_profile["index_sha256"]
        )
        self.assertEqual(
            profile384["index_sha256"], msa2048_profile["index_sha256"]
        )
        with self.assertRaises(ValueError):
            benchmark_crop_profile(999)
        with self.assertRaises(ValueError):
            benchmark_crop_profile(
                768,
                "benchmark_long_protein_context_1024_800",
            )

    def test_msa2048_workload_profiles_lock_staged_windows_and_memory_settings(self):
        legacy_name, legacy = benchmark_workload_profile(None)
        self.assertEqual(legacy_name, "legacy_msa32")
        self.assertEqual(legacy["msa_sample_cutoff"], 32)
        self.assertIsNone(legacy["pairformer_cpu_offload_every_n"])

        expected_profiles = {
            "crop1024_800_msa2048_offload_smoke": (
                1024,
                "benchmark_long_protein_context_1024_800_msa2048",
                0,
                1,
                8,
            ),
            "crop1024_800_msa2048_offload_matched": (
                1024,
                "benchmark_long_protein_context_1024_800_msa2048",
                10,
                20,
                240,
            ),
            "crop1024_800_msa2048_offload": (
                1024,
                "benchmark_long_protein_context_1024_800_msa2048",
                10,
                90,
                800,
            ),
            "crop1024_800_msa2048_offload_nuc128_db32_31step_7d1a": (
                1024,
                "benchmark_long_protein_context_1024_800_msa2048",
                10,
                21,
                248,
            ),
            "crop960_800_msa2048_offload": (
                960,
                "benchmark_long_protein_context_960_800_msa2048",
                10,
                90,
                800,
            ),
            "crop960_800_msa2048_offload_nuc128_db32": (
                960,
                "benchmark_long_protein_context_960_800_msa2048",
                10,
                90,
                800,
            ),
        }
        for profile_name, (
            crop_size,
            dataset_name,
            warmup,
            measure,
            global_samples,
        ) in expected_profiles.items():
            resolved_name, profile = benchmark_workload_profile(profile_name)
            self.assertEqual(resolved_name, profile_name)
            self.assertEqual(profile["crop_size"], crop_size)
            self.assertEqual(profile["dataset_rows"], 800)
            self.assertEqual(profile["dataset_name"], dataset_name)
            self.assertEqual(profile["msa_min_size"], 2048)
            self.assertEqual(profile["msa_sample_cutoff"], 2048)
            self.assertEqual(profile["msa_max_size"], 16384)
            self.assertEqual(profile["msa_chunk_size"], 128)
            self.assertEqual(profile["pairformer_cpu_offload_every_n"], 2)
            self.assertEqual(profile["msa_cpu_offload_every_n"], 1)
            self.assertEqual(
                (profile["warmup_steps"], profile["measure_steps"]),
                (warmup, measure),
            )
            self.assertEqual(
                profile["world_size"] * (warmup + measure), global_samples
            )

        _, experimental = benchmark_workload_profile(
            "crop960_800_msa2048_offload_nuc128_db32"
        )
        self.assertEqual(experimental["diffusion_batch_size"], 32)
        self.assertEqual(experimental["max_nuc_tokens"], 128)
        self.assertIs(experimental["requires_exact_token_count"], False)

        _, targeted = benchmark_workload_profile(
            "crop1024_800_msa2048_offload_nuc128_db32_31step_7d1a"
        )
        self.assertEqual(targeted["diffusion_batch_size"], 32)
        self.assertEqual(targeted["max_nuc_tokens"], 128)
        self.assertEqual(targeted["train_sampler_seed"], 77)
        self.assertEqual(
            targeted["target_sample"],
            {"pdb_id": "7d1a", "rank": 5, "step": 30},
        )

        _, nuc256 = benchmark_workload_profile(
            "crop1024_800_msa2048_offload_nuc256_db32_31step_7d1a"
        )
        self.assertEqual(nuc256["diffusion_batch_size"], 32)
        self.assertEqual(nuc256["max_nuc_tokens"], 256)
        self.assertEqual(nuc256["pairformer_cpu_offload_every_n"], 2)
        self.assertEqual(nuc256["msa_cpu_offload_every_n"], 1)
        self.assertEqual(nuc256["train_sampler_seed"], 77)
        self.assertEqual(
            nuc256["target_sample"],
            {"pdb_id": "7d1a", "rank": 5, "step": 30},
        )

        chunk256_name = (
            "crop1024_800_msa2048_offload_nuc256_db32_chunk256_10step"
        )
        resolved_name, chunk256 = benchmark_workload_profile(chunk256_name)
        self.assertEqual(resolved_name, chunk256_name)
        self.assertEqual(chunk256["warmup_steps"], 2)
        self.assertEqual(chunk256["measure_steps"], 8)
        self.assertEqual(chunk256["world_size"] * 10, 80)
        self.assertEqual(chunk256["msa_chunk_size"], 256)
        self.assertEqual(chunk256["diffusion_batch_size"], 32)
        self.assertEqual(chunk256["max_nuc_tokens"], 256)
        self.assertEqual(chunk256["pairformer_cpu_offload_every_n"], 2)
        self.assertEqual(chunk256["msa_cpu_offload_every_n"], 1)
        self.assertEqual(chunk256["train_sampler_seed"], 77)
        self.assertNotIn("target_sample", chunk256)

        chunk2048_name = (
            "crop1024_800_msa2048_offload_nuc256_db32_chunk2048_10step"
        )
        resolved_name, chunk2048 = benchmark_workload_profile(chunk2048_name)
        self.assertEqual(resolved_name, chunk2048_name)
        self.assertEqual(chunk2048["warmup_steps"], 2)
        self.assertEqual(chunk2048["measure_steps"], 8)
        self.assertEqual(chunk2048["world_size"] * 10, 80)
        self.assertEqual(chunk2048["msa_chunk_size"], 2048)
        self.assertEqual(chunk2048["diffusion_batch_size"], 32)
        self.assertEqual(chunk2048["max_nuc_tokens"], 256)
        self.assertEqual(chunk2048["pairformer_cpu_offload_every_n"], 2)
        self.assertEqual(chunk2048["msa_cpu_offload_every_n"], 1)
        self.assertEqual(chunk2048["train_sampler_seed"], 77)
        self.assertNotIn("target_sample", chunk2048)

        no_pairformer_offload_name = (
            "crop1024_800_msa2048_msaoffload_no_pairformer_offload_"
            "nuc128_db32_31step_7d1a"
        )
        resolved_name, no_pairformer_offload = benchmark_workload_profile(
            no_pairformer_offload_name
        )
        self.assertEqual(resolved_name, no_pairformer_offload_name)
        self.assertIsNone(
            no_pairformer_offload["pairformer_cpu_offload_every_n"]
        )
        self.assertEqual(
            no_pairformer_offload["msa_cpu_offload_every_n"], 1
        )
        self.assertEqual(no_pairformer_offload["train_sampler_seed"], 77)
        self.assertEqual(
            no_pairformer_offload["target_sample"],
            {"pdb_id": "7d1a", "rank": 5, "step": 30},
        )

        with self.assertRaises(ValueError):
            benchmark_workload_profile("msa2048_unlocked")

    def test_db40_paired_chunk2048_profiles_lock_production_settings(self):
        prefix = (
            "crop1024_800_msa2048_offload_nuc128_db40_paired_chunk2048"
        )
        expected_common = {
            "crop_size": 1024,
            "dataset_name": "benchmark_long_protein_context_1024_800_msa2048",
            "dataset_rows": 800,
            "world_size": 8,
            "msa_min_size": 2048,
            "msa_sample_cutoff": 2048,
            "msa_max_size": 16384,
            "msa_chunk_size": 2048,
            "pairformer_cpu_offload_every_n": 2,
            "msa_cpu_offload_every_n": 1,
            "diffusion_batch_size": 40,
            "max_nuc_tokens": 128,
            "requires_exact_token_count": False,
            "train_sampler_seed": 77,
            "pair_msa_as_unpaired_only": False,
            "shuffle_pair_msa_when_unpaired_only": False,
            "checkpoint_path": str(IMMUNE_HEAVY_CHECKPOINT),
            "checkpoint_sha256": IMMUNE_HEAVY_CHECKPOINT_SHA256,
            "lr": 0.000003,
        }
        for suffix, warmup, measure, phase_cuda_memory in (
            ("smoke", 0, 1, True),
            ("matched", 10, 20, False),
        ):
            profile_name = f"{prefix}_{suffix}"
            resolved_name, profile = benchmark_workload_profile(profile_name)
            self.assertEqual(resolved_name, profile_name)
            for key, value in expected_common.items():
                self.assertEqual(profile[key], value, msg=key)
            self.assertEqual(profile["warmup_steps"], warmup)
            self.assertEqual(profile["measure_steps"], measure)
            self.assertEqual(profile["phase_cuda_memory"], phase_cuda_memory)
            expected_global_samples = 8 if suffix == "smoke" else 240
            self.assertEqual(
                profile["world_size"] * (warmup + measure),
                expected_global_samples,
            )

    def test_crop384_db48_profiles_lock_random_initialization(self):
        prefix = (
            "crop384_800_msa2048_randominit_nuc128_db48_"
            "paired_chunk2048"
        )
        expected_common = {
            "crop_size": 384,
            "dataset_name": "benchmark_long_protein_context_384_800_msa2048",
            "dataset_rows": 800,
            "world_size": 8,
            "msa_min_size": 2048,
            "msa_sample_cutoff": 2048,
            "msa_max_size": 16384,
            "msa_chunk_size": 2048,
            "pairformer_cpu_offload_every_n": 2,
            "msa_cpu_offload_every_n": 1,
            "diffusion_batch_size": 48,
            "max_nuc_tokens": 128,
            "requires_exact_token_count": False,
            "train_sampler_seed": 77,
            "pair_msa_as_unpaired_only": False,
            "shuffle_pair_msa_when_unpaired_only": False,
            "initialization": "random",
            "lr": 0.001,
        }
        for suffix, warmup, measure, phase_cuda_memory in (
            ("smoke", 0, 1, True),
            ("matched", 10, 20, False),
        ):
            profile_name = f"{prefix}_{suffix}"
            resolved_name, profile = benchmark_workload_profile(profile_name)
            self.assertEqual(resolved_name, profile_name)
            for key, value in expected_common.items():
                self.assertEqual(profile[key], value, msg=key)
            self.assertNotIn("checkpoint_path", profile)
            self.assertNotIn("checkpoint_sha256", profile)
            self.assertEqual(profile["warmup_steps"], warmup)
            self.assertEqual(profile["measure_steps"], measure)
            self.assertEqual(profile["phase_cuda_memory"], phase_cuda_memory)
            initial_checkpoint, checkpoint_sha256 = (
                benchmark_initialization_expectation(profile)
            )
            self.assertEqual(initial_checkpoint, {"model": "", "ema": ""})
            self.assertEqual(checkpoint_sha256, "")
            invalid_profile = dict(profile, checkpoint_path="/tmp/stale.pt")
            with self.assertRaises(ValueError):
                benchmark_initialization_expectation(invalid_profile)

    def test_msa2048_depth_coverage_requires_real_2048_rows_on_every_call(self):
        coverage = {
            "target_depth": 2048,
            "step_record_count": 8,
            "call_count": 20,
            "missing_trace_step_count": 0,
            "missing_input_depth_call_count": 0,
            "missing_sampled_depth_call_count": 0,
            "input_depth_min": 2048,
            "input_depth_max": 8192,
            "sampled_depth_min": 2048,
            "sampled_depth_max": 2048,
            "sampled_target_call_count": 20,
            "below_target_call_count": 0,
            "per_rank_step_count": [1] * 8,
            "per_rank_call_count": [3, 3, 3, 3, 2, 2, 2, 2],
        }
        self.assertEqual(
            validate_msa_depth_coverage(
                coverage,
                world_size=8,
                warmup_steps=0,
                measure_steps=1,
            ),
            [],
        )

        shallow = dict(coverage)
        shallow.update(
            {
                "input_depth_min": 1024,
                "sampled_depth_min": 1024,
                "sampled_target_call_count": 19,
                "below_target_call_count": 1,
            }
        )
        errors = validate_msa_depth_coverage(
            shallow,
            world_size=8,
            warmup_steps=0,
            measure_steps=1,
        )
        self.assertTrue(any("sampled_depth_min" in error for error in errors))
        self.assertTrue(any("every MSA-module call" in error for error in errors))
        self.assertTrue(any("input_depth_min" in error for error in errors))

    def test_nuc128_token_composition_coverage_validates_every_sample(self):
        records = []
        for rank in range(2):
            for step in range(2):
                records.append(
                    {
                        "rank": rank,
                        "step": step,
                        "n_token": 900,
                        "N_prot_token": 770,
                        "N_lig_token": 2,
                        "N_rna_token": 100,
                        "N_dna_token": 28,
                        "N_nuc_token": 128,
                    }
                )
        coverage = {
            "target_max_nuc_tokens": 128,
            "sample_count": 4,
            "complete_composition_count": 4,
            "missing_composition_count": 0,
            "per_rank_sample_count": [2, 2],
            "over_nuc_limit_count": 0,
            "token_type_sum_mismatch_count": 0,
            "nuc_token_count": {"count": 4, "sum": 512, "min": 128, "max": 128},
            "sample_records": records,
        }

        self.assertEqual(
            validate_token_composition_coverage(
                coverage,
                world_size=2,
                steps_per_rank=2,
                crop_size=960,
                max_nuc_tokens=128,
            ),
            [],
        )

        records[0]["N_rna_token"] = 101
        records[0]["N_nuc_token"] = 129
        errors = validate_token_composition_coverage(
            coverage,
            world_size=2,
            steps_per_rank=2,
            crop_size=960,
            max_nuc_tokens=128,
        )
        self.assertTrue(any("above max_nuc_tokens=128" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
