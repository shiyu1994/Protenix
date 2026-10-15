import copy
import hashlib
import json
import unittest

from scripts.benchmark.compare_implementation_benchmarks import (
    IMPLEMENTATION_SPECS,
    compare_implementation_summaries,
)


SPEC = IMPLEMENTATION_SPECS["local_rearrange_v1"]
PRIMITIVES_PATH = SPEC["changed_path"]
COMMON_CODE_PATH = "runner/train.py"
EXTRA_CODE_PATH = "protenix/model/modules/diffusion.py"


def _mapping_sha256(mapping):
    payload = json.dumps(
        mapping, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _evidence(mapping):
    return {
        "entry_count": len(mapping),
        "mapping_sha256": _mapping_sha256(mapping),
        "mapping": mapping,
    }


def _refresh_evidence(summary, evidence_name):
    evidence = summary["node_evidence"][evidence_name]
    evidence["entry_count"] = len(evidence["mapping"])
    evidence["mapping_sha256"] = _mapping_sha256(evidence["mapping"])


def _summary(*, candidate):
    label = "candidate" if candidate else "baseline"
    primitives_hash = (
        SPEC["candidate_sha256"] if candidate else SPEC["baseline_sha256"]
    )
    code_mapping = {
        COMMON_CODE_PATH: "1" * 64,
        EXTRA_CODE_PATH: "2" * 64,
        PRIMITIVES_PATH: primitives_hash,
    }
    artifact_mapping = {
        "/shared/checkpoint.pt": "3" * 64,
        "/shared/benchmark/indices.csv": "4" * 64,
    }
    seconds_per_step = 1.0 if candidate else 2.0
    tokens_per_sec = 200.0 if candidate else 100.0
    allocated = (80 if candidate else 100) * 1024**3
    reserved = (90 if candidate else 110) * 1024**3
    nvml_peak = 80_000 if candidate else 100_000

    rank_records = []
    for rank in range(2):
        rank_records.append(
            {
                "rank": rank,
                "local_rank": rank,
                "node_rank": 0,
                "hostname": "host0",
                "elapsed_seconds": seconds_per_step * 20 + rank * 0.01,
                "sample_count": 20,
                "token_count": 20_480,
                "atom_count": 100_000,
                "min_token_count": 1024,
                "max_token_count": 1024,
                "workload_sha256": str(rank) * 64,
                "loss_nonfinite_count": 0,
                "gradient_nonfinite_count": 0,
                "optimizer_skipped_count": 0,
                "max_allocated_bytes": allocated - rank,
                "max_reserved_bytes": reserved - rank,
                "gpu_name": "A100",
            }
        )

    runner = {
        "schema_version": 1,
        "run_name": f"db40_{label}",
        "run_dir": f"/profiles/{label}/train/db40_{label}",
        "started_at": f"2026-07-21T00:00:0{int(candidate)}+08:00",
        "started_at_epoch": 1000.0 + int(candidate),
        "completed_at": f"2026-07-21T00:10:0{int(candidate)}+08:00",
        "completed_at_epoch": 1600.0 + int(candidate),
        "scope": "steady-state training window",
        "benchmark_variant": "fsdp2_full",
        "benchmark_dataset_name": (
            "benchmark_long_protein_context_1024_800_msa2048"
        ),
        "benchmark_workload_profile": (
            "crop1024_800_msa2048_offload_paired_nuc128_db40_matched"
        ),
        "train_crop_size": 1024,
        "distributed_strategy": "fsdp2",
        "optimizer_sharding": "none",
        "fsdp": {
            "mesh": "global",
            "param_dtype": "fp32",
            "reduce_dtype": "fp32",
            "reshard_after_forward": True,
        },
        "resolved_fsdp_mesh": "DeviceMesh('cuda', [0, 1])",
        "ema": {"enabled": True, "decay": 0.999, "shadow_device": "cpu"},
        "model_parameters": {"total": 464_442_431, "trainable": 464_441_841},
        "skip_final_checkpoint": True,
        "load_params_only": True,
        "initial_checkpoint": {
            "model": "/shared/checkpoint.pt",
            "ema": "/shared/checkpoint.pt",
        },
        "topology": {"world_size": 2, "local_world_size": 2, "num_nodes": 1},
        "iters_to_accumulate": 1,
        "train_sampler_replacement": False,
        "train_sampler_seed": 42,
        "run_sample_coverage": {
            "sample_count": 60,
            "unique_sample_count": 60,
            "duplicate_sample_count": 0,
            "ordered_sha256": "5" * 64,
        },
        "token_composition_coverage": {
            "sample_count": 60,
            "max_nuc_tokens": 128,
            "ordered_sha256": "6" * 64,
        },
        "msa_depth_coverage": {
            "sample_count": 60,
            "target_depth": 2048,
            "ordered_sha256": "7" * 64,
        },
        "max_loss_nonfinite_count_per_rank": 0,
        "max_gradient_nonfinite_count_per_rank": 0,
        "max_optimizer_skipped_count_per_rank": 0,
        "warmup_steps": 10,
        "measure_steps": 20,
        "world_size": 2,
        "elapsed_seconds": seconds_per_step * 20,
        "seconds_per_step": seconds_per_step,
        "total_samples": 40,
        "total_tokens": 40_960,
        "total_atoms": 200_000,
        "samples_per_sec": 2.0 / seconds_per_step,
        "tokens_per_sec": tokens_per_sec,
        "atoms_per_sec": 10_000.0 / seconds_per_step,
        "slowest_rank": 1,
        "rank_elapsed_imbalance": 1.001 + int(candidate) * 0.001,
        "max_allocated_bytes": allocated,
        "max_reserved_bytes": reserved,
        "rank_records": rank_records,
    }

    checkout = f"/workspace/db40_{label}"
    manifest = {
        "project": "protenix",
        "model_name": "protenix-v2",
        "seed": 42,
        "cwd": checkout,
        "training": {
            "train_crop_size": 1024,
            "diffusion_batch_size": 40,
            "diffusion_chunk_size": 4,
            "ema_decay": 0.999,
        },
        "model": {
            "c_z": 256,
            "msa_c_m": 128,
            "msa_chunk_size": 2048,
            "pairformer_blocks_per_ckpt": 2,
        },
        "data": {
            "train_sets": [
                "benchmark_long_protein_context_1024_800_msa2048"
            ],
            "benchmark_dataset": {
                "index_path": "/shared/benchmark/indices.csv",
            },
            "msa": {
                "pair_msa_as_unpaired_only": False,
                "shuffle_pair_msa_when_unpaired_only": False,
                "sample_cutoff": {"train": 2048},
            },
        },
        "initial_checkpoint": runner["initial_checkpoint"],
        "benchmark": {
            "variant": "fsdp2_full",
            "workload_profile": runner["benchmark_workload_profile"],
            "warmup_steps": 10,
            "measure_steps": 20,
        },
        "distributed": {
            "world_size": 2,
            "rank": 0,
            "local_rank": 0,
            "node_rank": 0,
            "strategy": "fsdp2",
            "optimizer_sharding": "none",
            "fsdp": runner["fsdp"],
        },
        "environment": {
            "hostname": "host0",
            "python_version": "3.11.10",
            "python_executable": "/env/bin/python",
            "torch_version": "2.7.1+cu126",
            "cuda_available": True,
            "cuda_version": "12.6",
            "cuda_visible_devices": "0,1",
            "protenix_root_dir": "/shared/protenix.20260101",
            "module_locations": {
                "configs": f"{checkout}/configs/__init__.py",
                "protenix": f"{checkout}/protenix/__init__.py",
            },
            "cc": "/mnt/project_abd/klp/runtime/protenix_cuda126_toolkit/bin/x86_64-conda-linux-gnu-gcc",
            "cxx": "/mnt/project_abd/klp/runtime/protenix_cuda126_toolkit/bin/x86_64-conda-linux-gnu-g++",
            "torch_lib_dir": "/env/lib/python3.11/site-packages/torch/lib",
            "ld_library_path": "/env/lib/python3.11/site-packages/torch/lib",
            "torch_multiprocessing_sharing_strategy": "file_system",
        },
        "git": {
            "commit": "abc123",
            "branch": "main",
            "dirty": True,
            "status_short": [f" M {PRIMITIVES_PATH}"] if candidate else [],
        },
        "code_sha256": dict(code_mapping),
        "command": [
            "runner/train.py",
            "--run_name",
            f"db40_{label}",
            "--base_dir",
            f"/profiles/{label}/train",
            "--distributed_strategy",
            "fsdp2",
            "--train_crop_size",
            "1024",
        ],
    }

    return {
        "schema_version": 1,
        "benchmark_id": f"db40_{label}",
        "status": "passed",
        "validation_errors": [],
        "runner": runner,
        "run_manifest": manifest,
        "nodes": {
            "expected": 1,
            "observed": 1,
            "max_launcher_wall_seconds": 100 + int(candidate),
        },
        "nvml_device": {
            "measurement_window": {
                "gpu_count": 2,
                "hottest_gpu_peak_mib": nvml_peak,
                "max_time_bucket_sum_mib": nvml_peak * 2,
                "per_gpu": [
                    {
                        "node_rank": 0,
                        "hostname": "host0",
                        "gpu_index": rank,
                        "gpu_uuid": f"GPU-{rank}",
                        "gpu_name": "A100",
                        "driver_version": "570.1",
                        "memory_total_mib": 81_920,
                    }
                    for rank in range(2)
                ],
            }
        },
        "node_evidence": {
            "code_tree": _evidence(dict(code_mapping)),
            "artifacts": _evidence(dict(artifact_mapping)),
        },
    }


class TestImplementationAIStudioBenchmark(unittest.TestCase):
    def test_matched_implementation_pair_passes_and_reports_deltas(self):
        result = compare_implementation_summaries(
            _summary(candidate=False),
            _summary(candidate=True),
            spec=SPEC,
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["mismatches"], [])
        self.assertEqual(
            [item["path"] for item in result["observed_code_delta"]["code_tree"]],
            [PRIMITIVES_PATH],
        )
        self.assertAlmostEqual(
            result["relative_metrics"]["step_time_change_percent"], -50.0
        )
        self.assertAlmostEqual(result["relative_metrics"]["speedup_ratio"], 2.0)
        self.assertAlmostEqual(
            result["relative_metrics"]["token_throughput_ratio"], 2.0
        )
        self.assertAlmostEqual(
            result["relative_metrics"]["allocated_memory_saving_percent"], 20.0
        )
        self.assertAlmostEqual(
            result["relative_metrics"]["nvml_memory_saving_percent"], 20.0
        )

    def test_phase_memory_outcomes_are_reported_without_timing_comparison(self):
        baseline = _summary(candidate=False)
        candidate = _summary(candidate=True)
        for summary, peak in ((baseline, 100), (candidate, 80)):
            summary["runner"]["phase_cuda_memory"] = {
                "throughput_comparable": False,
                "overall_max_allocated_bytes": peak * 1024**3,
                "phases": {"diffusion_train": {"max_peak_allocated_bytes": peak}},
            }
            for rank, record in enumerate(summary["runner"]["rank_records"]):
                record["phase_cuda_memory"] = {
                    "throughput_comparable": False,
                    "overall_max_allocated_bytes": (peak - rank) * 1024**3,
                    "steps": [{"step": 0, "phases": [{"name": "diffusion_train"}]}],
                }

        result = compare_implementation_summaries(
            baseline, candidate, spec=SPEC
        )

        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["relative_metrics"]["timing_comparable"])
        self.assertNotIn("speedup_ratio", result["relative_metrics"])
        self.assertEqual(
            result["absolute_metrics"]["candidate"]["phase_cuda_memory"][
                "overall_max_allocated_bytes"
            ],
            80 * 1024**3,
        )

    def test_extra_code_difference_is_rejected(self):
        baseline = _summary(candidate=False)
        candidate = _summary(candidate=True)
        candidate["node_evidence"]["code_tree"]["mapping"][EXTRA_CODE_PATH] = (
            "8" * 64
        )
        candidate["run_manifest"]["code_sha256"][EXTRA_CODE_PATH] = "8" * 64
        _refresh_evidence(candidate, "code_tree")

        result = compare_implementation_summaries(
            baseline, candidate, spec=SPEC
        )

        self.assertEqual(result["status"], "failed")
        paths = {item["path"] for item in result["mismatches"]}
        self.assertIn("node_evidence.code_tree.changed_paths", paths)
        self.assertIn("manifest.code_sha256.changed_paths", paths)
        self.assertIsNone(result["relative_metrics"])

    def test_swapped_expected_hashes_are_rejected(self):
        baseline = _summary(candidate=False)
        candidate = _summary(candidate=True)
        baseline["node_evidence"]["code_tree"]["mapping"][PRIMITIVES_PATH] = (
            SPEC["candidate_sha256"]
        )
        baseline["run_manifest"]["code_sha256"][PRIMITIVES_PATH] = SPEC[
            "candidate_sha256"
        ]
        candidate["node_evidence"]["code_tree"]["mapping"][PRIMITIVES_PATH] = (
            SPEC["baseline_sha256"]
        )
        candidate["run_manifest"]["code_sha256"][PRIMITIVES_PATH] = SPEC[
            "baseline_sha256"
        ]
        _refresh_evidence(baseline, "code_tree")
        _refresh_evidence(candidate, "code_tree")

        result = compare_implementation_summaries(
            baseline, candidate, spec=SPEC
        )

        self.assertEqual(result["status"], "failed")
        paths = {item["path"] for item in result["mismatches"]}
        self.assertTrue(any(path.endswith(".baseline_sha256") for path in paths))
        self.assertTrue(any(path.endswith(".candidate_sha256") for path in paths))

    def test_workload_hash_drift_is_rejected(self):
        baseline = _summary(candidate=False)
        candidate = _summary(candidate=True)
        candidate["runner"]["rank_records"][0]["workload_sha256"] = "f" * 64

        result = compare_implementation_summaries(
            baseline, candidate, spec=SPEC
        )

        self.assertEqual(result["status"], "failed")
        self.assertTrue(
            any(
                item["path"]
                == "runner.rank_records[0].workload_sha256"
                for item in result["mismatches"]
            )
        )

    def test_hardware_drift_is_rejected(self):
        baseline = _summary(candidate=False)
        candidate = _summary(candidate=True)
        candidate["nvml_device"]["measurement_window"]["per_gpu"][0][
            "gpu_uuid"
        ] = "GPU-different"

        result = compare_implementation_summaries(
            baseline, candidate, spec=SPEC
        )

        self.assertEqual(result["status"], "failed")
        self.assertTrue(
            any(
                item.get("category") == "hardware"
                and item["path"].endswith(".gpu_uuid")
                for item in result["mismatches"]
            )
        )

    def test_artifact_drift_is_rejected(self):
        baseline = _summary(candidate=False)
        candidate = _summary(candidate=True)
        candidate["node_evidence"]["artifacts"]["mapping"][
            "/shared/checkpoint.pt"
        ] = "9" * 64
        _refresh_evidence(candidate, "artifacts")

        result = compare_implementation_summaries(
            baseline, candidate, spec=SPEC
        )

        self.assertEqual(result["status"], "failed")
        self.assertTrue(
            any(
                item.get("category") == "artifacts"
                for item in result["mismatches"]
            )
        )


if __name__ == "__main__":
    unittest.main()
