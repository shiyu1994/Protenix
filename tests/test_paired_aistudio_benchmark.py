import copy

from scripts.benchmark.compare_aistudio_benchmarks import compare_summaries


def _summary(*, fsdp2: bool):
    variant = "fsdp2_full" if fsdp2 else "ddp_full"
    strategy = "fsdp2" if fsdp2 else "ddp"
    runner = {
        "schema_version": 1,
        "benchmark_variant": variant,
        "benchmark_dataset_name": "benchmark_long_protein_context_768",
        "train_crop_size": 768,
        "distributed_strategy": strategy,
        "resolved_fsdp_mesh": "DeviceMesh('cuda', [0, 1])" if fsdp2 else None,
        "world_size": 2,
        "warmup_steps": 10,
        "measure_steps": 20,
        "total_samples": 40,
        "total_tokens": 30720,
        "total_atoms": 100000,
        "topology": {"world_size": 2, "local_world_size": 2, "num_nodes": 1},
        "model_parameters": {"total": 464442431, "trainable": 464441841},
        "ema": {"enabled": True, "decay": 0.999, "shadow_device": "cpu"},
        "skip_final_checkpoint": True,
        "load_params_only": True,
        "initial_checkpoint": {"model": "/official.pt", "ema": "/official.pt"},
        "optimizer_sharding": "none",
        "fsdp": {
            "mesh": "global",
            "param_dtype": "fp32",
            "reduce_dtype": "fp32",
            "reshard_after_forward": True,
        },
        "iters_to_accumulate": 1,
        "max_loss_nonfinite_count_per_rank": 0,
        "max_gradient_nonfinite_count_per_rank": 0,
        "max_optimizer_skipped_count_per_rank": 0,
        "rank_records": [
            {
                "rank": rank,
                "local_rank": rank,
                "node_rank": 0,
                "sample_count": 20,
                "token_count": 15360,
                "atom_count": 50000,
                "min_token_count": 768,
                "max_token_count": 768,
                "workload_sha256": str(rank) * 64,
                "gpu_name": "A100",
                "loss_nonfinite_count": 0,
                "gradient_nonfinite_count": 0,
                "optimizer_skipped_count": 0,
            }
            for rank in range(2)
        ],
        "seconds_per_step": 1.1 if fsdp2 else 1.0,
        "samples_per_sec": 36.0 if fsdp2 else 40.0,
        "tokens_per_sec": 27927.0 if fsdp2 else 30720.0,
        "atoms_per_sec": 90909.0 if fsdp2 else 100000.0,
        "max_allocated_bytes": 80 * 1024**3 if fsdp2 else 100 * 1024**3,
        "max_reserved_bytes": 90 * 1024**3 if fsdp2 else 110 * 1024**3,
        "rank_elapsed_imbalance": 1.01,
    }
    manifest = {
        "project": "protenix",
        "model_name": "protenix-v2",
        "seed": 42,
        "cwd": "/repo",
        "training": {"train_crop_size": 768, "diffusion_batch_size": 24},
        "model": {"c_z": 256, "msa_c_m": 128},
        "data": {
            "index_sha256": "same",
            "num_dl_workers": 4,
            "epoch_size": 10000,
        },
        "initial_checkpoint": runner["initial_checkpoint"],
        "benchmark": {"variant": variant, "warmup_steps": 10, "measure_steps": 20},
        "distributed": {
            "rank": 0,
            "node_rank": 0,
            "strategy": strategy,
            "world_size": 2,
            "local_world_size": 2,
            "num_nodes": 1,
            "optimizer_sharding": "none",
            "fsdp": runner["fsdp"],
        },
        "environment": {
            "python_version": "3.11",
            "python_executable": "/env/bin/python",
            "torch_version": "2.7",
            "cuda_version": "12.6",
            "cc": "/mnt/project_abd/klp/runtime/protenix_cuda126_toolkit/bin/x86_64-conda-linux-gnu-gcc",
            "cxx": "/mnt/project_abd/klp/runtime/protenix_cuda126_toolkit/bin/x86_64-conda-linux-gnu-g++",
            "protenix_root_dir": "/data",
            "module_locations": {"protenix": "/repo/protenix/__init__.py"},
            "torch_lib_dir": "/env/lib/python3.11/site-packages/torch/lib",
            "torch_multiprocessing_sharing_strategy": "file_system",
        },
        "git": {"commit": "abc", "dirty": True},
        "code_sha256": {"runner/train.py": "a" * 64},
        "command": [
            "runner/train.py",
            "--run_name",
            variant,
            "--base_dir",
            f"/out/{variant}",
            "--distributed_strategy",
            strategy,
            "--train_crop_size",
            "768",
        ],
    }
    return {
        "schema_version": 1,
        "benchmark_id": variant,
        "status": "passed",
        "validation_errors": [],
        "runner": runner,
        "run_manifest": manifest,
        "nodes": {
            "expected": 1,
            "observed": 1,
            "max_launcher_wall_seconds": 50 if fsdp2 else 45,
        },
        "nvml_device": {
            "measurement_window": {
                "gpu_count": 2,
                "hottest_gpu_peak_mib": 80000 if fsdp2 else 100000,
                "max_time_bucket_sum_mib": 160000 if fsdp2 else 200000,
                "per_gpu": [
                    {
                        "node_rank": 0,
                        "hostname": "host0",
                        "gpu_index": 0,
                        "gpu_uuid": "GPU-0",
                        "gpu_name": "A100",
                        "driver_version": "570.1",
                        "memory_total_mib": 81920,
                    },
                    {
                        "node_rank": 0,
                        "hostname": "host0",
                        "gpu_index": 1,
                        "gpu_uuid": "GPU-1",
                        "gpu_name": "A100",
                        "driver_version": "570.1",
                        "memory_total_mib": 81920,
                    },
                ],
            }
        },
        "node_evidence": {
            "code_tree": {
                "mapping_sha256": "c" * 64,
                "entry_count": 100,
                "mapping": {"runner/train.py": "a" * 64},
            },
            "artifacts": {
                "mapping_sha256": "d" * 64,
                "mapping": {"/official.pt": "b" * 64},
            },
        },
    }


def test_paired_comparison_computes_metrics_only_for_matched_runs():
    result = compare_summaries(_summary(fsdp2=False), _summary(fsdp2=True))

    assert result["paired_status"] == "passed"
    assert result["mismatches"] == []
    assert round(result["relative_metrics"]["step_time_overhead_percent"], 6) == 10.0
    assert result["relative_metrics"]["allocated_memory_saving_percent"] == 20.0


def test_paired_comparison_rejects_model_config_drift():
    ddp = _summary(fsdp2=False)
    fsdp2 = copy.deepcopy(_summary(fsdp2=True))
    fsdp2["run_manifest"]["model"]["c_z"] = 384

    result = compare_summaries(ddp, fsdp2)

    assert result["paired_status"] == "failed"
    assert result["relative_metrics"] is None
    assert any(item["path"] == "manifest.model.c_z" for item in result["mismatches"])


def test_paired_comparison_rejects_sparse_self_declared_summaries():
    ddp = {
        "schema_version": 1,
        "benchmark_id": "ddp",
        "status": "passed",
        "validation_errors": [],
        "runner": {
            "benchmark_variant": "ddp_full",
            "distributed_strategy": "ddp",
        },
        "run_manifest": {},
    }
    fsdp2 = copy.deepcopy(ddp)
    fsdp2["benchmark_id"] = "fsdp2"
    fsdp2["runner"]["benchmark_variant"] = "fsdp2_full"
    fsdp2["runner"]["distributed_strategy"] = "fsdp2"

    result = compare_summaries(ddp, fsdp2)

    assert result["paired_status"] == "failed"
    assert result["relative_metrics"] is None
    assert any(item.get("category") == "single_run_shape" for item in result["mismatches"])


def test_paired_comparison_rejects_workload_order_drift():
    ddp = _summary(fsdp2=False)
    fsdp2 = _summary(fsdp2=True)
    fsdp2["runner"]["rank_records"][0]["workload_sha256"] = "f" * 64

    result = compare_summaries(ddp, fsdp2)

    assert result["paired_status"] == "failed"
    assert result["relative_metrics"] is None
    assert any(
        item["path"] == "runner.rank_workloads[0].workload_sha256"
        for item in result["mismatches"]
    )


def test_paired_comparison_rejects_cross_crop_runs():
    ddp = _summary(fsdp2=False)
    fsdp2 = _summary(fsdp2=True)
    fsdp2["runner"]["train_crop_size"] = 1024
    fsdp2["runner"]["benchmark_dataset_name"] = (
        "benchmark_long_protein_context_1024"
    )

    result = compare_summaries(ddp, fsdp2)

    assert result["paired_status"] == "failed"
    assert result["relative_metrics"] is None
    mismatch_paths = {item["path"] for item in result["mismatches"]}
    assert "runner.train_crop_size" in mismatch_paths
    assert "runner.benchmark_dataset_name" in mismatch_paths
