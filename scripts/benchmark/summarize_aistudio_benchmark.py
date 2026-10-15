#!/usr/bin/env python3
"""Summarize one multi-node AIStudio benchmark profile directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


MIB_PER_GIB = 1024.0
OFFICIAL_PROTENIX_ROOT = Path(
    "/mnt/project_abd/public_databases/protenix.20260101"
)
OFFICIAL_CHECKPOINT = OFFICIAL_PROTENIX_ROOT / "checkpoint/protenix-v2.pt"
OFFICIAL_CHECKPOINT_SHA256 = (
    "8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599"
)
IMMUNE_HEAVY_CHECKPOINT = Path(
    "/mnt/project_abd/klp/proteinx_output/"
    "protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/"
    "checkpoints/999_ema_0.999.pt"
)
IMMUNE_HEAVY_CHECKPOINT_SHA256 = (
    "9e0df397d9deb18ea46921c678a62afc4a587241aacd01acd27887729e3a0c3e"
)
OFFICIAL_SOURCE_INDEX = (
    OFFICIAL_PROTENIX_ROOT
    / "indices/indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz"
)
OFFICIAL_SOURCE_INDEX_SHA256 = (
    "e9fc812d2250c8d19a995dbd38991ac05204d2ace5e1e4878d4760e708ad2bc1"
)
OFFICIAL_BENCHMARK_128_INDEX_SHA256 = (
    "6bdf93c0e8f5a6a527c50112516a2f3c1cd27d4c0bfc825b5a7465fdadf5fd6a"
)
OFFICIAL_BENCHMARK_800_INDEX_SHA256 = (
    "f49cf926cb21dd271151700042b23ce787a8b4ae8783f9a74a4141a457491991"
)
OFFICIAL_BENCHMARK_MSA2048_INDEX_SHA256 = (
    "dce9db1758566a67f626959ba3e1cb5b56f8f693d91f2c37a44cec99ca2c54f1"
)
OFFICIAL_MSA_MAPPING = OFFICIAL_PROTENIX_ROOT / "common/seq_to_pdb_index.json"
OFFICIAL_MSA_MAPPING_SHA256 = (
    "cb5fb9aba9372dab1bbb0ed4322c3420fa3e606b597b4857c6bb577f94687ea9"
)
BENCHMARK_DATASET_PROFILES = {
    "benchmark_long_protein_context_384_800_msa2048": {
        "crop_size": 384,
        # This logical view reuses the exact MSA-aware artifact selected with
        # the stricter crop1024 source threshold.
        "min_num_tokens": 1025,
        "max_num_tokens": 1536,
        "index_sha256": OFFICIAL_BENCHMARK_MSA2048_INDEX_SHA256,
        "rows": 800,
        "epoch_size": 800,
        "replacement": False,
        "random_sample_if_failed": False,
    },
    "benchmark_long_protein_context_768": {
        "crop_size": 768,
        "min_num_tokens": 769,
        "max_num_tokens": 1536,
        "index_sha256": OFFICIAL_BENCHMARK_128_INDEX_SHA256,
        "rows": 128,
        "epoch_size": 10000,
        "replacement": True,
        "random_sample_if_failed": True,
    },
    "benchmark_long_protein_context_1024": {
        "crop_size": 1024,
        "min_num_tokens": 1025,
        "max_num_tokens": 1536,
        # The 1024 experiment intentionally keeps the same 128 source rows as
        # 768, isolating crop length rather than changing the PDB population.
        "index_sha256": OFFICIAL_BENCHMARK_128_INDEX_SHA256,
        "rows": 128,
        "epoch_size": 10000,
        "replacement": True,
        "random_sample_if_failed": True,
    },
    "benchmark_long_protein_context_1024_800": {
        "crop_size": 1024,
        "min_num_tokens": 1025,
        "max_num_tokens": 1536,
        "index_sha256": OFFICIAL_BENCHMARK_800_INDEX_SHA256,
        "rows": 800,
        "epoch_size": 800,
        "replacement": False,
        "random_sample_if_failed": False,
    },
    "benchmark_long_protein_context_1024_800_msa2048": {
        "crop_size": 1024,
        "min_num_tokens": 1025,
        "max_num_tokens": 1536,
        "index_sha256": OFFICIAL_BENCHMARK_MSA2048_INDEX_SHA256,
        "rows": 800,
        "epoch_size": 800,
        "replacement": False,
        "random_sample_if_failed": False,
    },
    "benchmark_long_protein_context_960_800_msa2048": {
        "crop_size": 960,
        # The crop960 view intentionally reuses the exact artifact audited
        # with this stricter source threshold and identical SHA256.
        "min_num_tokens": 1025,
        "max_num_tokens": 1536,
        "index_sha256": OFFICIAL_BENCHMARK_MSA2048_INDEX_SHA256,
        "rows": 800,
        "epoch_size": 800,
        "replacement": False,
        "random_sample_if_failed": False,
    },
}
BENCHMARK_CROP_PROFILES = {
    384: {
        "dataset_name": "benchmark_long_protein_context_384_800_msa2048",
        **BENCHMARK_DATASET_PROFILES[
            "benchmark_long_protein_context_384_800_msa2048"
        ],
    },
    768: {
        "dataset_name": "benchmark_long_protein_context_768",
        **BENCHMARK_DATASET_PROFILES["benchmark_long_protein_context_768"],
    },
    1024: {
        "dataset_name": "benchmark_long_protein_context_1024",
        **BENCHMARK_DATASET_PROFILES["benchmark_long_protein_context_1024"],
    },
    960: {
        "dataset_name": "benchmark_long_protein_context_960_800_msa2048",
        **BENCHMARK_DATASET_PROFILES[
            "benchmark_long_protein_context_960_800_msa2048"
        ],
    },
}
BENCHMARK_WORKLOAD_PROFILES = {
    # Kept as the implicit profile for summaries produced before workload
    # profiles were recorded in node_status/run_manifest.
    "legacy_msa32": {
        "crop_size": None,
        "dataset_name": None,
        "dataset_rows": None,
        "warmup_steps": None,
        "measure_steps": None,
        "world_size": None,
        "msa_min_size": 16,
        "msa_sample_cutoff": 32,
        "msa_max_size": 16384,
        "msa_chunk_size": 2048,
        "pairformer_cpu_offload_every_n": None,
        "msa_cpu_offload_every_n": None,
        "diffusion_batch_size": 24,
        "max_nuc_tokens": -1,
        "requires_exact_token_count": True,
    },
    "crop384_800_msa2048_randominit_nuc128_db48_paired_chunk2048_smoke": {
        "crop_size": 384,
        "dataset_name": "benchmark_long_protein_context_384_800_msa2048",
        "dataset_rows": 800,
        "warmup_steps": 0,
        "measure_steps": 1,
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
        "phase_cuda_memory": True,
    },
    "crop384_800_msa2048_randominit_nuc128_db48_paired_chunk2048_matched": {
        "crop_size": 384,
        "dataset_name": "benchmark_long_protein_context_384_800_msa2048",
        "dataset_rows": 800,
        "warmup_steps": 10,
        "measure_steps": 20,
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
        "phase_cuda_memory": False,
    },
    "crop1024_800_msa2048_offload": {
        "crop_size": 1024,
        "dataset_name": "benchmark_long_protein_context_1024_800_msa2048",
        "dataset_rows": 800,
        "warmup_steps": 10,
        "measure_steps": 90,
        "world_size": 8,
        "msa_min_size": 2048,
        "msa_sample_cutoff": 2048,
        "msa_max_size": 16384,
        "msa_chunk_size": 128,
        "pairformer_cpu_offload_every_n": 2,
        "msa_cpu_offload_every_n": 1,
        "diffusion_batch_size": 24,
        "max_nuc_tokens": -1,
        "requires_exact_token_count": True,
    },
    "crop1024_800_msa2048_offload_smoke": {
        "crop_size": 1024,
        "dataset_name": "benchmark_long_protein_context_1024_800_msa2048",
        "dataset_rows": 800,
        "warmup_steps": 0,
        "measure_steps": 1,
        "world_size": 8,
        "msa_min_size": 2048,
        "msa_sample_cutoff": 2048,
        "msa_max_size": 16384,
        "msa_chunk_size": 128,
        "pairformer_cpu_offload_every_n": 2,
        "msa_cpu_offload_every_n": 1,
        "diffusion_batch_size": 24,
        "max_nuc_tokens": -1,
        "requires_exact_token_count": True,
    },
    "crop1024_800_msa2048_offload_matched": {
        "crop_size": 1024,
        "dataset_name": "benchmark_long_protein_context_1024_800_msa2048",
        "dataset_rows": 800,
        "warmup_steps": 10,
        "measure_steps": 20,
        "world_size": 8,
        "msa_min_size": 2048,
        "msa_sample_cutoff": 2048,
        "msa_max_size": 16384,
        "msa_chunk_size": 128,
        "pairformer_cpu_offload_every_n": 2,
        "msa_cpu_offload_every_n": 1,
        "diffusion_batch_size": 24,
        "max_nuc_tokens": -1,
        "requires_exact_token_count": True,
    },
    "crop1024_800_msa2048_offload_nuc128_db40_paired_chunk2048_smoke": {
        "crop_size": 1024,
        "dataset_name": "benchmark_long_protein_context_1024_800_msa2048",
        "dataset_rows": 800,
        "warmup_steps": 0,
        "measure_steps": 1,
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
        "phase_cuda_memory": True,
    },
    "crop1024_800_msa2048_offload_nuc128_db40_paired_chunk2048_matched": {
        "crop_size": 1024,
        "dataset_name": "benchmark_long_protein_context_1024_800_msa2048",
        "dataset_rows": 800,
        "warmup_steps": 10,
        "measure_steps": 20,
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
        "phase_cuda_memory": False,
    },
    "crop1024_800_msa2048_offload_nuc128_db32_31step_7d1a": {
        "crop_size": 1024,
        "dataset_name": "benchmark_long_protein_context_1024_800_msa2048",
        "dataset_rows": 800,
        "warmup_steps": 10,
        "measure_steps": 21,
        "world_size": 8,
        "msa_min_size": 2048,
        "msa_sample_cutoff": 2048,
        "msa_max_size": 16384,
        "msa_chunk_size": 128,
        "pairformer_cpu_offload_every_n": 2,
        "msa_cpu_offload_every_n": 1,
        "diffusion_batch_size": 32,
        "max_nuc_tokens": 128,
        "requires_exact_token_count": False,
        "train_sampler_seed": 77,
        "target_sample": {"pdb_id": "7d1a", "rank": 5, "step": 30},
    },
    "crop1024_800_msa2048_offload_nuc256_db32_31step_7d1a": {
        "crop_size": 1024,
        "dataset_name": "benchmark_long_protein_context_1024_800_msa2048",
        "dataset_rows": 800,
        "warmup_steps": 10,
        "measure_steps": 21,
        "world_size": 8,
        "msa_min_size": 2048,
        "msa_sample_cutoff": 2048,
        "msa_max_size": 16384,
        "msa_chunk_size": 128,
        "pairformer_cpu_offload_every_n": 2,
        "msa_cpu_offload_every_n": 1,
        "diffusion_batch_size": 32,
        "max_nuc_tokens": 256,
        "requires_exact_token_count": False,
        "train_sampler_seed": 77,
        "target_sample": {"pdb_id": "7d1a", "rank": 5, "step": 30},
    },
    "crop1024_800_msa2048_offload_nuc256_db32_chunk256_10step": {
        "crop_size": 1024,
        "dataset_name": "benchmark_long_protein_context_1024_800_msa2048",
        "dataset_rows": 800,
        "warmup_steps": 2,
        "measure_steps": 8,
        "world_size": 8,
        "msa_min_size": 2048,
        "msa_sample_cutoff": 2048,
        "msa_max_size": 16384,
        "msa_chunk_size": 256,
        "pairformer_cpu_offload_every_n": 2,
        "msa_cpu_offload_every_n": 1,
        "diffusion_batch_size": 32,
        "max_nuc_tokens": 256,
        "requires_exact_token_count": False,
        "train_sampler_seed": 77,
    },
    "crop1024_800_msa2048_offload_nuc256_db32_chunk2048_10step": {
        "crop_size": 1024,
        "dataset_name": "benchmark_long_protein_context_1024_800_msa2048",
        "dataset_rows": 800,
        "warmup_steps": 2,
        "measure_steps": 8,
        "world_size": 8,
        "msa_min_size": 2048,
        "msa_sample_cutoff": 2048,
        "msa_max_size": 16384,
        "msa_chunk_size": 2048,
        "pairformer_cpu_offload_every_n": 2,
        "msa_cpu_offload_every_n": 1,
        "diffusion_batch_size": 32,
        "max_nuc_tokens": 256,
        "requires_exact_token_count": False,
        "train_sampler_seed": 77,
    },
    "crop1024_800_msa2048_msaoffload_no_pairformer_offload_nuc128_db32_31step_7d1a": {
        "crop_size": 1024,
        "dataset_name": "benchmark_long_protein_context_1024_800_msa2048",
        "dataset_rows": 800,
        "warmup_steps": 10,
        "measure_steps": 21,
        "world_size": 8,
        "msa_min_size": 2048,
        "msa_sample_cutoff": 2048,
        "msa_max_size": 16384,
        "msa_chunk_size": 128,
        "pairformer_cpu_offload_every_n": None,
        "msa_cpu_offload_every_n": 1,
        "diffusion_batch_size": 32,
        "max_nuc_tokens": 128,
        "requires_exact_token_count": False,
        "train_sampler_seed": 77,
        "target_sample": {"pdb_id": "7d1a", "rank": 5, "step": 30},
    },
    "crop960_800_msa2048_offload": {
        "crop_size": 960,
        "dataset_name": "benchmark_long_protein_context_960_800_msa2048",
        "dataset_rows": 800,
        "warmup_steps": 10,
        "measure_steps": 90,
        "world_size": 8,
        "msa_min_size": 2048,
        "msa_sample_cutoff": 2048,
        "msa_max_size": 16384,
        "msa_chunk_size": 128,
        "pairformer_cpu_offload_every_n": 2,
        "msa_cpu_offload_every_n": 1,
        "diffusion_batch_size": 24,
        "max_nuc_tokens": -1,
        "requires_exact_token_count": True,
    },
    "crop960_800_msa2048_offload_nuc128_db32": {
        "crop_size": 960,
        "dataset_name": "benchmark_long_protein_context_960_800_msa2048",
        "dataset_rows": 800,
        "warmup_steps": 10,
        "measure_steps": 90,
        "world_size": 8,
        "msa_min_size": 2048,
        "msa_sample_cutoff": 2048,
        "msa_max_size": 16384,
        "msa_chunk_size": 128,
        "pairformer_cpu_offload_every_n": 2,
        "msa_cpu_offload_every_n": 1,
        "diffusion_batch_size": 32,
        "max_nuc_tokens": 128,
        "requires_exact_token_count": False,
    },
}
REQUIRED_CODE_HASH_PATHS = {
    "benchmark.sh",
    "finetune_submit.sh",
    "finetune_demo.sh",
    "run_crop384_db48_ddp_scratch_50k.sh",
    "run_crop384_db48_ddp_scratch_200k.sh",
    "run_crop384_db48_ddp_disordered_v4_from_v8_step999.sh",
    "resume_crop384_db48_ddp_scratch_50k.sh",
    "resume_crop384_db48_ddp_scratch_200k.sh",
    "resume_crop384_db48_ddp_disordered_v4.sh",
    "configs/crop384_db48_data_profiles.sh",
    "configs/configs_base.py",
    "configs/configs_data.py",
    "runner/train.py",
    "runner/ema.py",
    "protenix/data/pipeline/sample_context.py",
    "protenix/data/pipeline/dataset.py",
    "protenix/data/pipeline/data_pipeline.py",
    "protenix/data/msa/msa_featurizer.py",
    "protenix/data/msa/msa_utils.py",
    "protenix/data/template/template_featurizer.py",
    "protenix/data/template/template_utils.py",
    "protenix/data/template/sequence_template_cache.py",
    "protenix/utils/fsdp.py",
    "protenix/model/protenix.py",
    "protenix/model/modules/primitives.py",
    "protenix/model/modules/pairformer.py",
    "protenix/model/modules/diffusion.py",
    "protenix/model/layer_norm/cache_provenance.py",
    "protenix/model/layer_norm/layer_norm.py",
    "protenix/model/layer_norm/torch_ext_compile.py",
    "protenix/model/layer_norm/kernel/layer_norm_cuda.cpp",
    "protenix/model/layer_norm/kernel/layer_norm_cuda_kernel.cu",
    "scripts/benchmark/build_long_protein_benchmark.py",
    "scripts/benchmark/build_msa_aware_long_protein_benchmark.py",
    "scripts/benchmark/audit_msa_depth.py",
    "scripts/benchmark/compare_aistudio_benchmarks.py",
    "scripts/benchmark/compare_implementation_benchmarks.py",
    "scripts/benchmark/run_aistudio_benchmark.sh",
    "scripts/benchmark/summarize_aistudio_benchmark.py",
    "scripts/runtime/cuda126_env.sh",
    "scripts/runtime/sequence_template_cache_contract.sh",
    "scripts/runtime/validate_v8_step999_warmstart_pair.py",
}


def benchmark_crop_profile(
    value: Any,
    dataset_name: Any = None,
) -> tuple[int, dict[str, Any]]:
    """Return the strict exact-crop profile, rejecting unapproved shapes."""
    if isinstance(value, bool):
        raise ValueError("benchmark crop size must be 384, 768, 960, or 1024")
    try:
        crop_size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "benchmark crop size must be 384, 768, 960, or 1024"
        ) from exc
    if crop_size not in BENCHMARK_CROP_PROFILES:
        raise ValueError("benchmark crop size must be 384, 768, 960, or 1024")
    if dataset_name not in {None, ""}:
        dataset_name = str(dataset_name)
        dataset_profile = BENCHMARK_DATASET_PROFILES.get(dataset_name)
        if dataset_profile is None or dataset_profile["crop_size"] != crop_size:
            raise ValueError(
                "benchmark dataset profile does not match the approved crop/row profile"
            )
        return crop_size, {"dataset_name": dataset_name, **dataset_profile}
    return crop_size, BENCHMARK_CROP_PROFILES[crop_size]


def benchmark_workload_profile(value: Any) -> tuple[str, dict[str, Any]]:
    """Resolve a strict workload profile while accepting legacy evidence."""
    profile_name = "legacy_msa32" if value is None or value == "" else str(value)
    profile = BENCHMARK_WORKLOAD_PROFILES.get(profile_name)
    if profile is None:
        raise ValueError(f"unsupported benchmark workload profile: {profile_name!r}")
    return profile_name, dict(profile)


def benchmark_initialization_expectation(
    workload_profile: dict[str, Any],
) -> tuple[dict[str, str], str]:
    """Resolve checkpoint evidence required by one workload profile."""
    initialization = workload_profile.get("initialization", "checkpoint")
    if initialization == "random":
        if workload_profile.get("checkpoint_path") not in {None, ""} or (
            workload_profile.get("checkpoint_sha256") not in {None, ""}
        ):
            raise ValueError(
                "random-initialization profile must not declare a checkpoint"
            )
        return {"model": "", "ema": ""}, ""
    if initialization != "checkpoint":
        raise ValueError(
            f"unsupported benchmark initialization mode: {initialization!r}"
        )
    checkpoint = str(
        workload_profile.get("checkpoint_path", OFFICIAL_CHECKPOINT)
    )
    checkpoint_sha256 = str(
        workload_profile.get("checkpoint_sha256", OFFICIAL_CHECKPOINT_SHA256)
    )
    return {"model": checkpoint, "ema": checkpoint}, checkpoint_sha256


def validate_msa_depth_coverage(
    coverage: Any,
    *,
    world_size: int,
    warmup_steps: int,
    measure_steps: int,
    target_depth: int = 2048,
) -> list[str]:
    """Validate real per-step/per-recycle MSA depth evidence."""
    errors: list[str] = []
    if not isinstance(coverage, dict):
        return ["MSA-2048 depth coverage is missing"]
    expected_steps_per_rank = warmup_steps + measure_steps
    expected_total_steps = world_size * expected_steps_per_rank
    for key, expected_value in {
        "target_depth": target_depth,
        "step_record_count": expected_total_steps,
        "missing_trace_step_count": 0,
        "missing_input_depth_call_count": 0,
        "missing_sampled_depth_call_count": 0,
        "sampled_depth_min": target_depth,
        "sampled_depth_max": target_depth,
        "below_target_call_count": 0,
        "per_rank_step_count": [expected_steps_per_rank] * world_size,
    }.items():
        if coverage.get(key) != expected_value:
            errors.append(
                f"MSA-2048 coverage field {key} must be {expected_value!r}"
            )
    call_count = coverage.get("call_count")
    if (
        isinstance(call_count, bool)
        or not isinstance(call_count, int)
        or call_count < expected_total_steps
    ):
        errors.append("MSA-2048 coverage call_count is missing or too small")
        call_count = None
    if call_count is not None and coverage.get("sampled_target_call_count") != call_count:
        errors.append(
            "MSA-2048 coverage requires every MSA-module call to sample 2048 rows"
        )
    per_rank_call_count = coverage.get("per_rank_call_count")
    if (
        not isinstance(per_rank_call_count, list)
        or len(per_rank_call_count) != world_size
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < expected_steps_per_rank
            for value in per_rank_call_count
        )
        or (call_count is not None and sum(per_rank_call_count) != call_count)
    ):
        errors.append("MSA-2048 coverage per_rank_call_count is invalid")
    input_depth_min = coverage.get("input_depth_min")
    input_depth_max = coverage.get("input_depth_max")
    valid_input_min = (
        isinstance(input_depth_min, int)
        and not isinstance(input_depth_min, bool)
        and input_depth_min >= target_depth
    )
    if not valid_input_min:
        errors.append("MSA-2048 coverage input_depth_min must be at least 2048")
    if (
        not isinstance(input_depth_max, int)
        or isinstance(input_depth_max, bool)
        or (valid_input_min and input_depth_max < input_depth_min)
    ):
        errors.append("MSA-2048 coverage input_depth_max is missing or invalid")
    return errors


def validate_token_composition_coverage(
    coverage: Any,
    *,
    world_size: int,
    steps_per_rank: int,
    crop_size: int,
    max_nuc_tokens: int,
) -> list[str]:
    """Validate all crop-level protein/ligand/RNA/DNA token records."""
    if not isinstance(coverage, dict):
        return ["benchmark token-composition coverage is missing"]

    errors: list[str] = []
    expected_samples = world_size * steps_per_rank
    records = coverage.get("sample_records")
    if not isinstance(records, list):
        return ["benchmark token-composition sample records are missing"]

    for key, expected_value in {
        "target_max_nuc_tokens": max_nuc_tokens,
        "sample_count": expected_samples,
        "complete_composition_count": expected_samples,
        "missing_composition_count": 0,
        "per_rank_sample_count": [steps_per_rank] * world_size,
    }.items():
        if coverage.get(key) != expected_value:
            errors.append(
                f"token-composition coverage field {key} must be {expected_value!r}"
            )

    if len(records) != expected_samples:
        errors.append(
            "token-composition sample record count must equal the full benchmark window"
        )

    required_numeric = (
        "rank",
        "step",
        "n_token",
        "N_prot_token",
        "N_lig_token",
        "N_rna_token",
        "N_dna_token",
        "N_nuc_token",
    )
    observed_rank_steps: set[tuple[int, int]] = set()
    observed_nuc_counts: list[int] = []
    observed_over_limit = 0
    observed_partition_mismatch = 0
    for record in records:
        if not isinstance(record, dict) or any(
            isinstance(record.get(key), bool)
            or not isinstance(record.get(key), int)
            for key in required_numeric
        ):
            errors.append("token-composition record contains missing/non-integer fields")
            continue
        if any(record[key] < 0 for key in required_numeric[2:]):
            errors.append("token-composition record contains a negative token count")
            continue
        rank = record["rank"]
        step = record["step"]
        observed_rank_steps.add((rank, step))
        if not (0 <= rank < world_size and 0 <= step < steps_per_rank):
            errors.append("token-composition record rank/step is out of range")
        n_token = record["n_token"]
        if not (0 < n_token <= crop_size):
            errors.append(
                f"token-composition n_token must satisfy 0 < n_token <= {crop_size}"
            )
        partition_sum = sum(
            record[key]
            for key in (
                "N_prot_token",
                "N_lig_token",
                "N_rna_token",
                "N_dna_token",
            )
        )
        if partition_sum != n_token:
            observed_partition_mismatch += 1
        nuc_tokens = record["N_rna_token"] + record["N_dna_token"]
        if record["N_nuc_token"] != nuc_tokens:
            errors.append("token-composition N_nuc_token is inconsistent")
        observed_nuc_counts.append(nuc_tokens)
        observed_over_limit += int(
            max_nuc_tokens > 0 and nuc_tokens > max_nuc_tokens
        )

    if len(observed_rank_steps) != expected_samples:
        errors.append("token-composition rank/step coverage is incomplete or duplicated")
    if coverage.get("over_nuc_limit_count") != observed_over_limit:
        errors.append("token-composition over-limit aggregate is inconsistent")
    if observed_over_limit:
        errors.append(
            f"token-composition contains {observed_over_limit} samples above "
            f"max_nuc_tokens={max_nuc_tokens}"
        )
    if coverage.get("token_type_sum_mismatch_count") != (
        observed_partition_mismatch
    ):
        errors.append("token-composition partition aggregate is inconsistent")
    if observed_partition_mismatch:
        errors.append(
            f"token-composition contains {observed_partition_mismatch} partition mismatches"
        )
    reported_nuc_distribution = coverage.get("nuc_token_count") or {}
    for key, expected_value in {
        "count": len(observed_nuc_counts),
        "sum": sum(observed_nuc_counts),
        "min": min(observed_nuc_counts) if observed_nuc_counts else None,
        "max": max(observed_nuc_counts) if observed_nuc_counts else None,
    }.items():
        if reported_nuc_distribution.get(key) != expected_value:
            errors.append(
                f"token-composition nuc distribution field {key} is inconsistent"
            )
    return errors


def _nested(value: Any, *keys: str, default: Any = None) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _config_mapping(value: Any) -> dict[str, Any]:
    """Normalize legacy ConfigDict strings while remaining fail-closed."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _summarize_sha256_files(paths: list[Path]) -> dict[str, Any]:
    per_file: list[dict[str, Any]] = []
    for path in paths:
        values: dict[str, str] = {}
        invalid_lines: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                parts = line.split(maxsplit=1)
                if len(parts) != 2 or not _is_sha256(parts[0]):
                    if line:
                        invalid_lines.append(line)
                    continue
                filename = parts[1].lstrip(" *")
                if not filename or filename in values:
                    invalid_lines.append(line)
                    continue
                values[filename] = parts[0]
        canonical = json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        per_file.append(
            {
                "path": str(path),
                "entry_count": len(values),
                "invalid_lines": invalid_lines,
                "mapping": values,
                "mapping_sha256": hashlib.sha256(canonical).hexdigest(),
            }
        )
    unique_mappings = {
        record["mapping_sha256"] for record in per_file
    }
    return {
        "file_count": len(paths),
        "consistent_across_nodes": len(unique_mappings) <= 1,
        "mapping_sha256": (
            per_file[0]["mapping_sha256"] if len(unique_mappings) == 1 else None
        ),
        "entry_count": per_file[0]["entry_count"] if per_file else 0,
        "mapping": per_file[0]["mapping"] if per_file else {},
        "files": per_file,
    }


def _single_match(paths: list[Path], description: str) -> Path | None:
    if not paths:
        return None
    if len(paths) > 1:
        joined = ", ".join(str(path) for path in paths)
        raise ValueError(f"Multiple {description} files found: {joined}")
    return paths[0]


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key].strip())


def _int(row: dict[str, str], key: str) -> int:
    return int(float(row[key].strip()))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _integer_or(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _summarize_nvml(
    csv_paths: list[Path],
    *,
    measurement_start: float | None,
    measurement_end: float | None,
    expected_nodes: int | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in csv_paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                try:
                    rows.append(
                        {
                            "epoch_seconds": _float(raw, "epoch_seconds"),
                            "node_rank": _int(raw, "node_rank"),
                            "hostname": raw["hostname"].strip(),
                            "gpu_index": _int(raw, "gpu_index"),
                            "gpu_uuid": raw["gpu_uuid"].strip(),
                            "gpu_name": raw["gpu_name"].strip(),
                            "driver_version": raw["driver_version"].strip(),
                            "memory_used_mib": _int(raw, "memory_used_mib"),
                            "memory_total_mib": _int(raw, "memory_total_mib"),
                            "utilization_gpu_pct": _int(
                                raw, "utilization_gpu_pct"
                            ),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue

    measured_rows = [
        row
        for row in rows
        if measurement_start is not None
        and measurement_end is not None
        and measurement_start <= row["epoch_seconds"] <= measurement_end
    ]

    def aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
        per_gpu: dict[tuple[int, str, str], dict[str, Any]] = {}
        sample_node_sums: dict[tuple[float, int], int] = defaultdict(int)
        for row in selected:
            key = (row["node_rank"], row["hostname"], row["gpu_uuid"])
            current = per_gpu.setdefault(
                key,
                {
                    "node_rank": row["node_rank"],
                    "hostname": row["hostname"],
                    "gpu_index": row["gpu_index"],
                    "gpu_uuid": row["gpu_uuid"],
                    "gpu_name": row["gpu_name"],
                    "driver_version": row["driver_version"],
                    "memory_total_mib": row["memory_total_mib"],
                    "peak_memory_used_mib": 0,
                    "peak_utilization_gpu_pct": 0,
                    "first_epoch_seconds": row["epoch_seconds"],
                    "last_epoch_seconds": row["epoch_seconds"],
                    "sample_count": 0,
                },
            )
            current["peak_memory_used_mib"] = max(
                current["peak_memory_used_mib"], row["memory_used_mib"]
            )
            current["peak_utilization_gpu_pct"] = max(
                current["peak_utilization_gpu_pct"], row["utilization_gpu_pct"]
            )
            current["first_epoch_seconds"] = min(
                current["first_epoch_seconds"], row["epoch_seconds"]
            )
            current["last_epoch_seconds"] = max(
                current["last_epoch_seconds"], row["epoch_seconds"]
            )
            current["sample_count"] += 1

            sample_node_sums[
                (row["epoch_seconds"], row["node_rank"])
            ] += row["memory_used_mib"]

        gpu_records = sorted(
            per_gpu.values(),
            key=lambda record: (record["node_rank"], record["gpu_index"]),
        )
        # Node clocks are expected to be synchronized. Half-second buckets
        # merge samples from different nodes without claiming exact NVML timing.
        bucket_node_sums: dict[float, dict[int, int]] = defaultdict(dict)
        for (epoch_seconds, node_rank), memory_sum in sample_node_sums.items():
            bucket = round(epoch_seconds * 2.0) / 2.0
            previous = bucket_node_sums[bucket].get(node_rank, 0)
            bucket_node_sums[bucket][node_rank] = max(previous, memory_sum)
        complete_sums = []
        for node_sums in bucket_node_sums.values():
            if expected_nodes is None or len(node_sums) == expected_nodes:
                complete_sums.append(sum(node_sums.values()))
        return {
            "sample_count": len(selected),
            "gpu_count": len(gpu_records),
            "hottest_gpu_peak_mib": max(
                (record["peak_memory_used_mib"] for record in gpu_records),
                default=None,
            ),
            "sum_of_per_gpu_peaks_mib": (
                sum(record["peak_memory_used_mib"] for record in gpu_records)
                if gpu_records
                else None
            ),
            "max_time_bucket_sum_mib": max(complete_sums, default=None),
            "per_gpu": gpu_records,
        }

    return {
        "scope_note": (
            "NVML memory.used is device-level sampled memory and may include "
            "CUDA/NCCL contexts or unrelated processes"
        ),
        "all_samples": aggregate(rows),
        "measurement_window": aggregate(measured_rows),
    }


def _summarize_system(csv_paths: list[Path]) -> dict[str, Any]:
    per_node: dict[tuple[int, str], dict[str, Any]] = {}
    for path in csv_paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                try:
                    key = (_int(raw, "node_rank"), raw["hostname"].strip())
                    available = _int(raw, "mem_available_kib")
                    load = _float(raw, "load_1m")
                except (KeyError, TypeError, ValueError):
                    continue
                current = per_node.setdefault(
                    key,
                    {
                        "node_rank": key[0],
                        "hostname": key[1],
                        "min_mem_available_kib": available,
                        "max_load_1m": load,
                        "sample_count": 0,
                    },
                )
                current["min_mem_available_kib"] = min(
                    current["min_mem_available_kib"], available
                )
                current["max_load_1m"] = max(current["max_load_1m"], load)
                current["sample_count"] += 1
    return {
        "per_node": sorted(per_node.values(), key=lambda item: item["node_rank"])
    }


def _summarize_compute_apps(
    csv_paths: list[Path],
    *,
    measurement_start: float | None,
    measurement_end: float | None,
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    measured_rows: list[dict[str, Any]] = []
    invalid_row_count = 0
    for path in csv_paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                try:
                    row = {
                        "epoch_seconds": _float(raw, "epoch_seconds"),
                        "node_rank": _int(raw, "node_rank"),
                        "hostname": raw["hostname"].strip(),
                        "pid": _int(raw, "pid"),
                        "gpu_uuid": raw["gpu_uuid"].strip(),
                        "used_memory_mib": _int(raw, "used_memory_mib"),
                        "process_name": raw["process_name"].strip(),
                        "is_benchmark_process": _int(
                            raw, "is_benchmark_process"
                        ),
                        # Older profiles do not contain process_role.  They are
                        # intentionally treated as unverified ancestry evidence
                        # below instead of assuming that every namespace-
                        # invisible PID is a training rank.
                        "process_role": (raw.get("process_role") or "").strip(),
                    }
                except (KeyError, TypeError, ValueError):
                    invalid_row_count += 1
                    continue
                all_rows.append(row)
                if (
                    measurement_start is not None
                    and measurement_end is not None
                    and measurement_start <= row["epoch_seconds"] <= measurement_end
                ):
                    measured_rows.append(row)

    process_stats: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    for row in all_rows:
        key = (
            row["node_rank"],
            row["hostname"],
            row["pid"],
            row["gpu_uuid"],
        )
        record = process_stats.setdefault(
            key,
            {
                "node_rank": row["node_rank"],
                "hostname": row["hostname"],
                "pid": row["pid"],
                "gpu_uuid": row["gpu_uuid"],
                "process_name": row["process_name"],
                "first_epoch_seconds": row["epoch_seconds"],
                "last_epoch_seconds": row["epoch_seconds"],
                "sample_count": 0,
                "max_used_memory_mib": row["used_memory_mib"],
                "is_benchmark_values": set(),
                "evidence_roles": set(),
            },
        )
        record["process_name"] = row["process_name"] or record["process_name"]
        record["first_epoch_seconds"] = min(
            record["first_epoch_seconds"], row["epoch_seconds"]
        )
        record["last_epoch_seconds"] = max(
            record["last_epoch_seconds"], row["epoch_seconds"]
        )
        record["sample_count"] += 1
        record["max_used_memory_mib"] = max(
            record["max_used_memory_mib"], row["used_memory_mib"]
        )
        record["is_benchmark_values"].add(row["is_benchmark_process"])
        if row["process_role"]:
            record["evidence_roles"].add(row["process_role"])

    measured_keys = {
        (
            row["node_rank"],
            row["hostname"],
            row["pid"],
            row["gpu_uuid"],
        )
        for row in measured_rows
    }
    measured_processes = [process_stats[key] for key in measured_keys]

    classification: dict[tuple[int, str, int, str], str] = {}
    rank_candidates_by_gpu: dict[
        tuple[int, str], list[dict[str, Any]]
    ] = {}
    explicit_training_by_gpu: dict[
        tuple[int, str], list[dict[str, Any]]
    ] = {}
    for record in measured_processes:
        key = (
            record["node_rank"],
            record["hostname"],
            record["pid"],
            record["gpu_uuid"],
        )
        roles = record["evidence_roles"]
        benchmark_values = record["is_benchmark_values"]
        if roles & {"external", "unknown"} or benchmark_values == {0}:
            classification[key] = "external"
        elif "training_rank" in roles:
            classification[key] = "training_rank"
            explicit_training_by_gpu.setdefault(
                (record["node_rank"], record["gpu_uuid"]), []
            ).append(record)
        elif "benchmark_auxiliary" in roles:
            classification[key] = "auxiliary"
        elif benchmark_values == {1}:
            rank_candidates_by_gpu.setdefault(
                (record["node_rank"], record["gpu_uuid"]), []
            ).append(record)
        else:
            classification[key] = "unclassified"

    # AIStudio's nvidia-smi commonly reports host-namespace PIDs as
    # ``[Not Found]``.  For those backward-compatible/unresolved records, the
    # training rank is the earliest context on each measured GPU that is still
    # present in the synchronized measurement window.  A later unresolved PID
    # is *not* silently accepted as an auxiliary process: it may be an
    # unrelated job that started after the empty-GPU preflight (as observed
    # with SFM).  Only ancestry-proven benchmark_descendant records may be
    # classified as auxiliaries.
    for gpu_key, candidates in rank_candidates_by_gpu.items():
        primary = (
            None
            if explicit_training_by_gpu.get(gpu_key)
            else min(
                candidates,
                key=lambda item: (item["first_epoch_seconds"], item["pid"]),
            )
        )
        for record in candidates:
            key = (
                record["node_rank"],
                record["hostname"],
                record["pid"],
                record["gpu_uuid"],
            )
            if primary is not None and record is primary:
                classification[key] = "training_rank"
            elif "benchmark_descendant" in record["evidence_roles"]:
                classification[key] = "auxiliary"
            else:
                classification[key] = "unclassified"

    def public_process_record(
        record: dict[str, Any], process_class: str
    ) -> dict[str, Any]:
        return {
            "node_rank": record["node_rank"],
            "hostname": record["hostname"],
            "pid": record["pid"],
            "gpu_uuid": record["gpu_uuid"],
            "process_name": record["process_name"],
            "process_class": process_class,
            "evidence_roles": sorted(record["evidence_roles"]),
            "first_epoch_seconds": record["first_epoch_seconds"],
            "last_epoch_seconds": record["last_epoch_seconds"],
            "sample_count": record["sample_count"],
            "max_used_memory_mib": record["max_used_memory_mib"],
        }

    classified_processes: dict[str, list[dict[str, Any]]] = {
        "training_rank": [],
        "auxiliary": [],
        "external": [],
        "unclassified": [],
    }
    for record in measured_processes:
        key = (
            record["node_rank"],
            record["hostname"],
            record["pid"],
            record["gpu_uuid"],
        )
        process_class = classification.get(key, "unclassified")
        classified_processes[process_class].append(
            public_process_record(record, process_class)
        )
    for records in classified_processes.values():
        records.sort(
            key=lambda item: (
                item["node_rank"],
                item["gpu_uuid"],
                item["first_epoch_seconds"],
                item["pid"],
            )
        )

    training_rank_processes = classified_processes["training_rank"]
    auxiliary_processes = classified_processes["auxiliary"]
    external_processes = classified_processes["external"]
    unclassified_processes = classified_processes["unclassified"]
    benchmark_gpu_uuids = sorted(
        {
            (record["node_rank"], record["gpu_uuid"])
            for record in training_rank_processes
        }
    )
    benchmark_processes = sorted(
        training_rank_processes + auxiliary_processes,
        key=lambda item: (
            item["node_rank"],
            item["gpu_uuid"],
            item["first_epoch_seconds"],
            item["pid"],
        ),
    )
    return {
        "profile_file_count": len(csv_paths),
        "measurement_row_count": len(measured_rows),
        "benchmark_gpu_uuids": [
            {"node_rank": node_rank, "gpu_uuid": gpu_uuid}
            for node_rank, gpu_uuid in benchmark_gpu_uuids
        ],
        "invalid_row_count": invalid_row_count,
        "external_processes": external_processes,
        "unclassified_processes": unclassified_processes,
        "training_rank_processes": training_rank_processes,
        "auxiliary_processes": auxiliary_processes,
        "benchmark_processes": benchmark_processes,
    }


def validate_compute_process_topology(
    compute_apps: dict[str, Any],
    *,
    world_size: int,
    expected_compute_gpus: set[tuple[int, str]],
) -> list[str]:
    """Validate one training rank per GPU while allowing proven auxiliaries."""

    errors: list[str] = []
    training_ranks = compute_apps.get("training_rank_processes") or []
    rank_gpus = [
        (int(record["node_rank"]), str(record["gpu_uuid"]))
        for record in training_ranks
    ]
    if len(training_ranks) != world_size:
        errors.append(
            "measurement window must contain exactly one training rank "
            f"per world-size slot: {len(training_ranks)} != {world_size}"
        )
    if len(set(rank_gpus)) != len(rank_gpus):
        errors.append("multiple training ranks were classified on the same GPU")
    if set(rank_gpus) != expected_compute_gpus:
        errors.append(
            "training-rank GPU coverage differs from measured GPUs: "
            f"observed={sorted(set(rank_gpus))}, "
            f"expected={sorted(expected_compute_gpus)}"
        )

    auxiliary_gpus = {
        (int(record["node_rank"]), str(record["gpu_uuid"]))
        for record in compute_apps.get("auxiliary_processes") or []
    }
    if not auxiliary_gpus.issubset(set(rank_gpus)):
        errors.append("benchmark auxiliary context exists on a GPU without a rank")
    if compute_apps.get("unclassified_processes"):
        errors.append(
            "unclassified additional GPU compute contexts overlap the "
            "measurement window"
        )
    return errors


def _summarize_preflight_compute_apps(csv_paths: list[Path]) -> dict[str, Any]:
    processes: list[dict[str, Any]] = []
    invalid_row_count = 0
    for path in csv_paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                try:
                    processes.append(
                        {
                            "node_rank": _int(raw, "node_rank"),
                            "hostname": raw["hostname"].strip(),
                            "pid": _int(raw, "pid"),
                            "gpu_uuid": raw["gpu_uuid"].strip(),
                            "used_memory_mib": _int(raw, "used_memory_mib"),
                            "process_name": raw["process_name"].strip(),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    invalid_row_count += 1
    return {
        "profile_file_count": len(csv_paths),
        "invalid_row_count": invalid_row_count,
        "processes": processes,
    }


def build_summary(profile_root: Path) -> dict[str, Any]:
    profile_root = profile_root.resolve()
    status_paths = sorted(profile_root.glob("nodes/node_*/node_status.json"))
    statuses = [_read_json(path) for path in status_paths]
    expected_nodes = (
        int(statuses[0]["expected_nodes"])
        if statuses and statuses[0].get("expected_nodes") is not None
        else None
    )

    runner_path = _single_match(
        sorted(profile_root.glob("train/*/benchmark/benchmark_result.json")),
        "runner benchmark result",
    )
    runner = _read_json(runner_path) if runner_path else None
    run_status_path = _single_match(
        sorted(profile_root.glob("train/*/manifest/run_status.json")),
        "training run status",
    )
    run_status = _read_json(run_status_path) if run_status_path else None
    manifest_path = _single_match(
        sorted(profile_root.glob("train/*/manifest/run_manifest.json")),
        "training run manifest",
    )
    manifest = _read_json(manifest_path) if manifest_path else None
    # Runs produced before ConfigDict gained explicit to_dict serialization
    # contain a few YAML strings.  Canonicalize those fields so the same strict
    # schema checks apply; malformed strings become empty mappings and fail the
    # required-value checks below.
    if runner is not None:
        runner["fsdp"] = _config_mapping(runner.get("fsdp"))
    if manifest is not None:
        manifest_distributed = manifest.get("distributed")
        if isinstance(manifest_distributed, dict):
            manifest_distributed["fsdp"] = _config_mapping(
                manifest_distributed.get("fsdp")
            )
        manifest_data = manifest.get("data")
        if isinstance(manifest_data, dict):
            manifest_data["msa"] = _config_mapping(manifest_data.get("msa"))
            benchmark_dataset = manifest_data.get("benchmark_dataset")
            if isinstance(benchmark_dataset, dict):
                benchmark_dataset["base_info"] = _config_mapping(
                    benchmark_dataset.get("base_info")
                )
                benchmark_dataset["cropping_configs"] = _config_mapping(
                    benchmark_dataset.get("cropping_configs")
                )
    config_path = _single_match(
        sorted(profile_root.glob("train/*/config.yaml")),
        "resolved training config",
    )
    train_metrics_path = _single_match(
        sorted(profile_root.glob("train/*/logs/train_metrics.jsonl")),
        "training metrics log",
    )
    train_metric_records = (
        _read_jsonl(train_metrics_path) if train_metrics_path else []
    )
    final_train_metrics = train_metric_records[-1] if train_metric_records else None

    measurement_start = (
        float(runner["started_at_epoch"])
        if runner and runner.get("started_at_epoch") is not None
        else None
    )
    measurement_end = (
        float(runner["completed_at_epoch"])
        if runner and runner.get("completed_at_epoch") is not None
        else None
    )
    nvml = _summarize_nvml(
        sorted(profile_root.glob("nodes/node_*/nvml.csv")),
        measurement_start=measurement_start,
        measurement_end=measurement_end,
        expected_nodes=expected_nodes,
    )
    system = _summarize_system(
        sorted(profile_root.glob("nodes/node_*/system.csv"))
    )
    compute_apps = _summarize_compute_apps(
        sorted(profile_root.glob("nodes/node_*/compute_apps.csv")),
        measurement_start=measurement_start,
        measurement_end=measurement_end,
    )
    preflight_compute_apps = _summarize_preflight_compute_apps(
        sorted(profile_root.glob("nodes/node_*/preflight_compute_apps.csv"))
    )
    node_code_tree = _summarize_sha256_files(
        sorted(profile_root.glob("nodes/node_*/code_tree.sha256"))
    )
    node_artifact_hashes = _summarize_sha256_files(
        sorted(profile_root.glob("nodes/node_*/artifact_hashes.sha256"))
    )

    errors: list[str] = []
    declared_workload_profiles = {
        str(status["workload_profile"])
        for status in statuses
        if status.get("workload_profile") not in {None, ""}
    }
    if len(declared_workload_profiles) > 1:
        errors.append(
            "node status files contain different workload profiles: "
            f"{sorted(declared_workload_profiles)}"
        )
    manifest_workload_profile = _nested(
        manifest or {}, "benchmark", "workload_profile"
    )
    reported_workload_profile = (
        next(iter(declared_workload_profiles))
        if len(declared_workload_profiles) == 1
        else manifest_workload_profile
    )
    try:
        workload_profile_name, workload_profile = benchmark_workload_profile(
            reported_workload_profile
        )
    except ValueError as exc:
        errors.append(str(exc))
        workload_profile_name, workload_profile = benchmark_workload_profile(None)
    expected_random_initialization = (
        workload_profile.get("initialization", "checkpoint") == "random"
    )
    try:
        (
            expected_initial_checkpoint,
            expected_checkpoint_sha256,
        ) = benchmark_initialization_expectation(workload_profile)
    except ValueError as exc:
        errors.append(str(exc))
        expected_initial_checkpoint = {"model": "", "ema": ""}
        expected_checkpoint_sha256 = ""
    expected_checkpoint = (
        Path(expected_initial_checkpoint["model"])
        if expected_initial_checkpoint["model"]
        else None
    )
    expected_lr = workload_profile.get("lr", 0.00001)
    expected_pair_msa_as_unpaired_only = workload_profile.get(
        "pair_msa_as_unpaired_only", True
    )
    expected_shuffle_pair_msa = workload_profile.get(
        "shuffle_pair_msa_when_unpaired_only", True
    )
    expected_phase_cuda_memory = workload_profile.get(
        "phase_cuda_memory", False
    )
    if workload_profile_name != "legacy_msa32":
        if len(declared_workload_profiles) != 1 or any(
            status.get("workload_profile") != workload_profile_name
            for status in statuses
        ):
            errors.append(
                "every node status must record the selected non-legacy workload profile"
            )
        if manifest_workload_profile != workload_profile_name:
            errors.append(
                "run manifest workload profile is missing or differs from node status"
            )
    elif manifest_workload_profile not in {None, "", workload_profile_name}:
        errors.append("run manifest workload profile differs from node status")
    if not statuses:
        errors.append("no node status files found")
    if expected_nodes is not None and len(statuses) != expected_nodes:
        errors.append(
            f"expected {expected_nodes} node status files but found {len(statuses)}"
        )
    declared_node_counts = {
        int(status["expected_nodes"])
        for status in statuses
        if status.get("expected_nodes") is not None
    }
    if len(declared_node_counts) > 1:
        errors.append(
            "node status files disagree on expected_nodes: "
            f"{sorted(declared_node_counts)}"
        )
    observed_node_ranks = [
        int(status["node_rank"])
        for status in statuses
        if status.get("node_rank") is not None
    ]
    if len(observed_node_ranks) != len(set(observed_node_ranks)):
        errors.append(f"duplicate node ranks found: {observed_node_ranks}")
    if expected_nodes is not None and set(observed_node_ranks) != set(
        range(expected_nodes)
    ):
        errors.append(
            "observed node ranks do not match the expected range: "
            f"{sorted(observed_node_ranks)} != {list(range(expected_nodes))}"
        )
    benchmark_ids = {
        str(status["benchmark_id"])
        for status in statuses
        if status.get("benchmark_id") is not None
    }
    if len(benchmark_ids) > 1:
        errors.append(
            "node status files contain different benchmark IDs: "
            f"{sorted(benchmark_ids)}"
        )
    hostnames = [
        str(status["hostname"])
        for status in statuses
        if status.get("hostname") is not None
    ]
    if len(hostnames) != len(set(hostnames)):
        errors.append(f"duplicate node hostnames found: {hostnames}")
    benchmark_variants = {
        str(status["benchmark_variant"])
        for status in statuses
        if status.get("benchmark_variant")
    }
    if statuses and not benchmark_variants:
        errors.append("benchmark variant is missing from node status files")
    elif len(benchmark_variants) > 1:
        errors.append(
            "node status files contain different variants: "
            f"{sorted(benchmark_variants)}"
        )
    failed_nodes = [
        status.get("node_rank")
        for status in statuses
        if int(status.get("exit_code", 1)) != 0
    ]
    if failed_nodes:
        errors.append(f"non-zero launcher exit on nodes {failed_nodes}")
    if runner is None:
        errors.append("runner benchmark_result.json not found")
    if run_status is None:
        errors.append("training run_status.json not found")
    elif run_status.get("status") != "completed":
        errors.append(f"training status is {run_status.get('status')!r}")
    if manifest is None:
        errors.append("training run_manifest.json not found")
    if config_path is None:
        errors.append("resolved training config.yaml not found")
    if final_train_metrics is None:
        errors.append("training train_metrics.jsonl not found or empty")

    health_keys = [
        "loss_nan_count",
        "nonfinite_grad_count",
        "optimizer_skipped_count",
    ]
    training_health = {key: 0 for key in health_keys}
    for health_key in health_keys:
        missing_health = False
        for record in train_metric_records:
            grad_health = record.get("runtime_window", {}).get("grad_health", {})
            if health_key not in grad_health:
                missing_health = True
                continue
            training_health[health_key] += int(grad_health[health_key] or 0)
        if train_metric_records and missing_health:
            errors.append(f"training health check {health_key} is missing")
        if training_health[health_key] != 0:
            errors.append(f"training health check {health_key} is non-zero")

    if statuses and nvml["all_samples"]["gpu_count"] == 0:
        errors.append("no valid NVML GPU samples found")
    if runner and nvml["measurement_window"]["sample_count"] == 0:
        errors.append("no NVML samples overlap the runner measurement window")
    if expected_nodes is not None and compute_apps["profile_file_count"] != expected_nodes:
        errors.append("compute-process profile coverage does not match node count")
    if runner and compute_apps["measurement_row_count"] == 0:
        errors.append("no GPU compute-process samples overlap the measurement window")
    if compute_apps["invalid_row_count"]:
        errors.append("compute-process profile contains rows without ancestry labels")
    if compute_apps["external_processes"]:
        errors.append("unrelated GPU compute processes overlap the measurement window")
    if expected_nodes is not None and (
        preflight_compute_apps["profile_file_count"] != expected_nodes
    ):
        errors.append("preflight compute-process evidence does not match node count")
    if preflight_compute_apps["invalid_row_count"]:
        errors.append("preflight compute-process evidence contains invalid rows")
    if preflight_compute_apps["processes"]:
        errors.append("GPU compute processes existed before benchmark launch")
    for evidence_name, evidence in (
        ("code-tree", node_code_tree),
        ("artifact", node_artifact_hashes),
    ):
        if expected_nodes is not None and evidence["file_count"] != expected_nodes:
            errors.append(
                f"{evidence_name} hash file coverage does not match node count"
            )
        if not evidence["consistent_across_nodes"]:
            errors.append(f"{evidence_name} hashes differ across nodes")
        if any(record["invalid_lines"] for record in evidence["files"]):
            errors.append(f"{evidence_name} hash files contain invalid lines")
    code_tree_mapping = node_code_tree["mapping"]
    if not REQUIRED_CODE_HASH_PATHS.issubset(code_tree_mapping):
        errors.append("node code-tree hash mapping misses benchmark-critical files")
    expected_artifact_hashes = {
        str(OFFICIAL_SOURCE_INDEX): OFFICIAL_SOURCE_INDEX_SHA256,
    }
    if expected_checkpoint is not None:
        expected_artifact_hashes[str(expected_checkpoint)] = (
            expected_checkpoint_sha256
        )
    elif any(
        Path(filename).suffix in {".pt", ".pth", ".ckpt"}
        for filename in node_artifact_hashes["mapping"]
    ):
        errors.append(
            "random-initialization benchmark must not hash a checkpoint artifact"
        )
    for artifact_path, expected_sha256 in expected_artifact_hashes.items():
        if node_artifact_hashes["mapping"].get(artifact_path) != expected_sha256:
            errors.append(
                f"node artifact hash is missing or invalid for {artifact_path}"
            )
    generated_index_hashes = {
        digest
        for filename, digest in node_artifact_hashes["mapping"].items()
        if Path(filename).name == "indices.csv"
    }
    approved_index_hashes = {
        profile["index_sha256"] for profile in BENCHMARK_DATASET_PROFILES.values()
    }
    if len(generated_index_hashes) != 1 or not generated_index_hashes.issubset(
        approved_index_hashes
    ):
        errors.append("node artifact hash for generated benchmark index is invalid")

    crop_size: int | None = None
    crop_profile: dict[str, Any] | None = None
    reported_crop_size = (
        runner.get("train_crop_size")
        if runner and runner.get("train_crop_size") is not None
        else _nested(manifest or {}, "training", "train_crop_size")
    )
    reported_dataset_name = (
        runner.get("benchmark_dataset_name")
        if runner and runner.get("benchmark_dataset_name")
        else _nested(manifest or {}, "benchmark", "dataset_name")
    )
    try:
        crop_size, crop_profile = benchmark_crop_profile(
            reported_crop_size,
            reported_dataset_name,
        )
    except ValueError as exc:
        errors.append(str(exc))
    if crop_profile is not None and generated_index_hashes != {
        crop_profile["index_sha256"]
    }:
        errors.append("node artifact hash differs from the selected dataset profile")
    if workload_profile_name != "legacy_msa32" and crop_profile is not None:
        if crop_size != workload_profile["crop_size"]:
            errors.append("workload profile crop size differs from the locked target")
        if crop_profile["dataset_name"] != workload_profile["dataset_name"]:
            errors.append("workload profile dataset differs from the locked target")
        if crop_profile["rows"] != workload_profile["dataset_rows"]:
            errors.append("workload profile dataset row count differs from the locked target")
        if (
            node_artifact_hashes["mapping"].get(str(OFFICIAL_MSA_MAPPING))
            != OFFICIAL_MSA_MAPPING_SHA256
        ):
            errors.append("MSA-2048 workload is missing the locked sequence-to-MSA mapping hash")

    if runner:
        world_size = int(runner.get("world_size", 0))
        if workload_profile_name != "legacy_msa32":
            if world_size != workload_profile["world_size"]:
                errors.append(
                    f"{workload_profile_name} requires world_size={workload_profile['world_size']}"
                )
            if int(runner.get("warmup_steps", -1)) != workload_profile[
                "warmup_steps"
            ] or int(runner.get("measure_steps", -1)) != workload_profile[
                "measure_steps"
            ]:
                errors.append(
                    f"{workload_profile_name} requires exactly "
                    f"{workload_profile['warmup_steps']}+{workload_profile['measure_steps']} steps"
                )
            runner_workload_profile = runner.get("benchmark_workload_profile")
            if runner_workload_profile not in {None, workload_profile_name}:
                errors.append("runner workload profile differs from node status")
        all_gpu_count = int(nvml["all_samples"]["gpu_count"])
        measured_gpu_count = int(nvml["measurement_window"]["gpu_count"])
        if all_gpu_count and world_size != all_gpu_count:
            errors.append(
                "runner world_size does not match the number of profiled GPUs: "
                f"{world_size} != {all_gpu_count}"
            )
        if measured_gpu_count != world_size:
            errors.append(
                "measurement-window GPU coverage does not match runner world_size: "
                f"{measured_gpu_count} != {world_size}"
            )

        measured_gpu_records = nvml["measurement_window"]["per_gpu"]
        expected_compute_gpus = {
            (int(record["node_rank"]), str(record["gpu_uuid"]))
            for record in measured_gpu_records
        }
        observed_compute_gpus = {
            (int(record["node_rank"]), str(record["gpu_uuid"]))
            for record in compute_apps["benchmark_gpu_uuids"]
        }
        if observed_compute_gpus != expected_compute_gpus:
            errors.append(
                "benchmark compute-process GPU coverage differs from measured GPUs: "
                f"observed={sorted(observed_compute_gpus)}, "
                f"expected={sorted(expected_compute_gpus)}"
            )
        errors.extend(
            validate_compute_process_topology(
                compute_apps,
                world_size=world_size,
                expected_compute_gpus=expected_compute_gpus,
            )
        )
        measured_node_ranks = {
            int(record["node_rank"]) for record in measured_gpu_records
        }
        if expected_nodes is not None and measured_node_ranks != set(
            range(expected_nodes)
        ):
            errors.append(
                "measurement-window NVML node coverage is incomplete: "
                f"{sorted(measured_node_ranks)}"
            )

        if measurement_start is not None and measurement_end is not None:
            max_poll_seconds = max(
                (float(status.get("poll_seconds", 0.5)) for status in statuses),
                default=0.5,
            )
            # One monitor iteration performs both an NVML device query and a
            # compute-process query before sleeping.  On AIStudio that makes
            # the observed cadence about one second even when poll_seconds is
            # 0.5.  Allow two real iterations at each synchronized boundary;
            # this remains negligible compared with the 20-step formal window.
            coverage_tolerance = max(2.0, 4.0 * max_poll_seconds)
            incomplete_gpu_coverage = [
                record["gpu_uuid"]
                for record in measured_gpu_records
                if record["first_epoch_seconds"]
                > measurement_start + coverage_tolerance
                or record["last_epoch_seconds"]
                < measurement_end - coverage_tolerance
            ]
            if incomplete_gpu_coverage:
                errors.append(
                    "NVML sampling does not span the measurement window on GPUs: "
                    + ", ".join(incomplete_gpu_coverage)
                )

        runner_variant = str(runner.get("benchmark_variant", ""))
        if benchmark_variants and runner_variant not in benchmark_variants:
            errors.append(
                "runner benchmark variant does not match node status files: "
                f"{runner_variant!r}"
            )
        expected_strategy = {
            "ddp_full": "ddp",
            "fsdp2_full": "fsdp2",
        }.get(runner_variant)
        if expected_strategy is None:
            errors.append(f"unsupported or missing runner variant: {runner_variant!r}")
        elif runner.get("distributed_strategy") != expected_strategy:
            errors.append(
                "runner variant does not match distributed_strategy"
            )
        if runner.get("optimizer_sharding") != "none":
            errors.append("matched DDP/FSDP2 benchmark requires optimizer_sharding=none")
        expected_fsdp = {
            "param_dtype": "fp32",
            "reduce_dtype": "fp32",
            "reshard_after_forward": True,
        }
        runner_fsdp = runner.get("fsdp") or {}
        for key, expected_value in expected_fsdp.items():
            if runner_fsdp.get(key) != expected_value:
                errors.append(
                    f"matched benchmark FSDP setting {key} must be {expected_value!r}"
                )
        if runner_fsdp.get("mesh") not in {"auto", "global", "hybrid"}:
            errors.append("matched benchmark FSDP mesh setting is invalid")
        runner_ema = runner.get("ema") or {}
        if runner_ema.get("enabled") is not True:
            errors.append("matched DDP/FSDP2 benchmark requires EMA to be enabled")
        if runner_ema.get("shadow_device") != "cpu":
            errors.append("matched DDP/FSDP2 benchmark requires CPU EMA")
        if runner_ema.get("decay") != 0.999:
            errors.append("matched DDP/FSDP2 benchmark requires EMA decay 0.999")
        if runner.get("skip_final_checkpoint") is not True:
            errors.append("matched benchmark requires skip_final_checkpoint=true")
        if runner.get("load_params_only") is not True:
            errors.append("matched benchmark requires load_params_only=true")
        if expected_strategy == "fsdp2" and not runner.get("resolved_fsdp_mesh"):
            errors.append("FSDP2 benchmark is missing the resolved DeviceMesh")
        if runner.get("initial_checkpoint") != expected_initial_checkpoint:
            errors.append("matched benchmark raw/EMA initialization differs from its workload profile")
        phase_cuda_memory = runner.get("phase_cuda_memory")
        if expected_phase_cuda_memory:
            if not isinstance(phase_cuda_memory, dict):
                errors.append("phase-memory workload is missing phase_cuda_memory results")
            elif phase_cuda_memory.get("throughput_comparable") is not False:
                errors.append("phase-memory workload must mark throughput as non-comparable")
        elif phase_cuda_memory is not None:
            errors.append("performance workload unexpectedly enabled phase-memory diagnostics")
        parameter_counts = runner.get("model_parameters") or {}
        total_parameters = int(parameter_counts.get("total", -1))
        trainable_parameters = int(parameter_counts.get("trainable", -1))
        fixed_parameter_count = int(parameter_counts.get("frozen", -1))
        if total_parameters != 464_442_431:
            errors.append(
                "protenix-v2 benchmark total parameter count must be 464442431"
            )
        if trainable_parameters != 464_441_841:
            errors.append(
                "protenix-v2 benchmark trainable parameter count must be 464441841"
            )
        expected_fixed_names = {
            "diffusion_module.diffusion_conditioning.fourier_embedding.w",
            "diffusion_module.diffusion_conditioning.fourier_embedding.b",
            "confidence_head.lower_bins",
            "confidence_head.upper_bins",
        }
        fixed_parameters = parameter_counts.get("fixed_parameters")
        if not isinstance(fixed_parameters, dict):
            errors.append("model fixed_parameters must be a mapping")
        else:
            observed_fixed_names = set(fixed_parameters)
            if observed_fixed_names != expected_fixed_names:
                errors.append(
                    "full-model benchmark fixed parameter names differ from "
                    "the four intentional Protenix-v2 constants"
                )
            if sum(int(value) for value in fixed_parameters.values()) != (
                fixed_parameter_count
            ):
                errors.append("model fixed parameter counts are inconsistent")
        if trainable_parameters + fixed_parameter_count != total_parameters:
            errors.append("model trainable/fixed/total parameter counts differ")
        if crop_size is not None:
            if runner.get("train_crop_size") is not None and int(
                runner.get("train_crop_size")
            ) != crop_size:
                errors.append("runner crop size differs from the manifest")
            if crop_size != 768 and runner.get("train_crop_size") != crop_size:
                errors.append(
                    "runner crop size is missing from the explicit-crop result"
                )
            expected_dataset_name = crop_profile["dataset_name"]
            runner_dataset_name = runner.get("benchmark_dataset_name")
            if runner_dataset_name not in {None, "", expected_dataset_name}:
                errors.append("runner benchmark dataset differs from the crop profile")
            if crop_size != 768 and runner_dataset_name != expected_dataset_name:
                errors.append(
                    "runner benchmark dataset is missing from the explicit-crop result"
                )
        expected_measured_samples = world_size * int(runner.get("measure_steps", 0))
        if int(runner.get("total_samples", -1)) != expected_measured_samples:
            errors.append("runner measured sample count is inconsistent")
        if crop_size is not None:
            expected_measured_tokens = expected_measured_samples * crop_size
            measured_tokens = int(runner.get("total_tokens", -1))
            if workload_profile["requires_exact_token_count"]:
                if measured_tokens != expected_measured_tokens:
                    errors.append(
                        "exact-crop benchmark did not measure exactly "
                        f"{crop_size} tokens per sample"
                    )
            elif not (0 < measured_tokens <= expected_measured_tokens):
                errors.append(
                    "capped-crop benchmark measured token total must be positive "
                    f"and no larger than {expected_measured_tokens}"
                )
        if crop_profile is not None and crop_profile["rows"] == 800:
            if runner.get("train_sampler_replacement") is not False:
                errors.append("800-row coverage benchmark requires replacement=false")
            expected_run_samples = world_size * (
                int(runner.get("warmup_steps", 0))
                + int(runner.get("measure_steps", 0))
            )
            coverage = runner.get("run_sample_coverage") or {}
            if workload_profile_name == "legacy_msa32" and expected_run_samples != 800:
                errors.append("legacy 800-row coverage benchmark requires 800 global samples")
            for key, expected_value in {
                "sample_count": expected_run_samples,
                "unique_sample_count": expected_run_samples,
                "duplicate_sample_count": 0,
                "missing_identity_count": 0,
            }.items():
                if coverage.get(key) != expected_value:
                    errors.append(
                        f"800-row no-replacement coverage field {key} must be {expected_value}"
                    )
            expected_steps_per_rank = int(runner.get("warmup_steps", 0)) + int(
                runner.get("measure_steps", 0)
            )
            if coverage.get("per_rank_sample_count") != [
                expected_steps_per_rank
            ] * world_size:
                errors.append(
                    "800-row no-replacement coverage has an invalid per-rank sample count"
                )
            if not _is_sha256(coverage.get("ordered_sha256")):
                errors.append("800-row coverage ordered SHA256 is missing or invalid")
            if workload_profile["max_nuc_tokens"] > 0:
                errors.extend(
                    validate_token_composition_coverage(
                        runner.get("token_composition_coverage"),
                        world_size=world_size,
                        steps_per_rank=expected_steps_per_rank,
                        crop_size=crop_size,
                        max_nuc_tokens=workload_profile["max_nuc_tokens"],
                    )
                )
                target_sample = workload_profile.get("target_sample")
                if target_sample is not None:
                    composition = runner.get("token_composition_coverage") or {}
                    records = composition.get("sample_records") or []
                    matches = [
                        record
                        for record in records
                        if record.get("pdb_id") == target_sample["pdb_id"]
                        and record.get("rank") == target_sample["rank"]
                        and record.get("step") == target_sample["step"]
                    ]
                    if len(matches) != 1:
                        errors.append(
                            "targeted workload must contain exactly one "
                            f"{target_sample['pdb_id']} record at rank "
                            f"{target_sample['rank']} step {target_sample['step']}"
                        )
                    elif int(matches[0].get("N_nuc_token", -1)) > int(
                        workload_profile["max_nuc_tokens"]
                    ):
                        errors.append("targeted 7d1a sample exceeds the nucleic cap")
        if workload_profile_name != "legacy_msa32":
            errors.extend(
                validate_msa_depth_coverage(
                    runner.get("msa_depth_coverage"),
                    world_size=workload_profile["world_size"],
                    warmup_steps=workload_profile["warmup_steps"],
                    measure_steps=workload_profile["measure_steps"],
                )
            )
        for health_key in [
            "max_loss_nonfinite_count_per_rank",
            "max_gradient_nonfinite_count_per_rank",
            "max_optimizer_skipped_count_per_rank",
        ]:
            if int(runner.get(health_key, 0)) != 0:
                errors.append(f"runner benchmark health check {health_key} is non-zero")

        topology = runner.get("topology") or {}
        if int(topology.get("world_size", 0)) != world_size:
            errors.append("runner topology world_size is missing or inconsistent")
        if expected_nodes is not None and int(topology.get("num_nodes", 0)) != int(
            expected_nodes
        ):
            errors.append("runner topology num_nodes is missing or inconsistent")
        local_world_size = int(topology.get("local_world_size", 0))
        if (
            expected_nodes is not None
            and local_world_size * expected_nodes != world_size
        ):
            errors.append("runner topology local_world_size is inconsistent")

        rank_records = runner.get("rank_records") or []
        runner_ranks = [int(record.get("rank", -1)) for record in rank_records]
        if set(runner_ranks) != set(range(world_size)):
            errors.append("runner rank records are incomplete or duplicated")
        measured_rank_samples = 0
        measured_rank_tokens = 0
        measured_rank_atoms = 0
        for record in rank_records:
            sample_count = _integer_or(record.get("sample_count"))
            token_count = _integer_or(record.get("token_count"))
            atom_count = _integer_or(record.get("atom_count"))
            measured_rank_samples += sample_count
            measured_rank_tokens += token_count
            measured_rank_atoms += atom_count
            if sample_count != _integer_or(runner.get("measure_steps"), 0):
                errors.append(
                    f"runner rank {record.get('rank')} sample count differs from measure_steps"
                )
            if crop_size is not None:
                if workload_profile["requires_exact_token_count"]:
                    if token_count != sample_count * crop_size:
                        errors.append(
                            f"runner rank {record.get('rank')} did not measure "
                            f"{crop_size} tokens per sample"
                        )
                    if record.get("min_token_count") != crop_size or record.get(
                        "max_token_count"
                    ) != crop_size:
                        errors.append(
                            f"runner rank {record.get('rank')} contains a "
                            f"non-{crop_size}-token batch"
                        )
                elif (
                    token_count <= 0
                    or token_count > sample_count * crop_size
                    or not isinstance(record.get("min_token_count"), int)
                    or not isinstance(record.get("max_token_count"), int)
                    or not 0 < record["min_token_count"] <= record["max_token_count"] <= crop_size
                ):
                    errors.append(
                        f"runner rank {record.get('rank')} has invalid capped-crop token counts"
                    )
            if atom_count <= 0:
                errors.append(
                    f"runner rank {record.get('rank')} has no measured atoms"
                )
            if not _is_sha256(record.get("workload_sha256")):
                errors.append(
                    f"runner rank {record.get('rank')} workload SHA256 is missing or invalid"
                )
        if measured_rank_samples != int(runner.get("total_samples", -1)):
            errors.append("runner rank sample totals differ from aggregate")
        if measured_rank_tokens != int(runner.get("total_tokens", -1)):
            errors.append("runner rank token totals differ from aggregate")
        if measured_rank_atoms != int(runner.get("total_atoms", -1)):
            errors.append("runner rank atom totals differ from aggregate")
        if expected_nodes is not None and local_world_size > 0:
            for expected_node_rank in range(expected_nodes):
                node_records = [
                    record
                    for record in rank_records
                    if int(record.get("node_rank", -1)) == expected_node_rank
                ]
                local_ranks = {
                    int(record.get("local_rank", -1)) for record in node_records
                }
                if local_ranks != set(range(local_world_size)):
                    errors.append(
                        "runner local ranks are incomplete on node "
                        f"{expected_node_rank}: {sorted(local_ranks)}"
                    )

    if manifest is not None and runner is not None:
        manifest_distributed = manifest.get("distributed", {})
        manifest_training = manifest.get("training", {})
        manifest_benchmark = manifest.get("benchmark", {})
        manifest_model = manifest.get("model", {})
        manifest_data = manifest.get("data", {})
        manifest_environment = manifest.get("environment", {})
        if int(manifest_distributed.get("world_size", 0)) != int(
            runner.get("world_size", 0)
        ):
            errors.append("run manifest world_size does not match runner result")
        if manifest_benchmark.get("variant") != runner.get("benchmark_variant"):
            errors.append("run manifest variant does not match runner result")
        if int(manifest_benchmark.get("warmup_steps", -1)) != int(
            runner.get("warmup_steps", -1)
        ) or int(manifest_benchmark.get("measure_steps", -1)) != int(
            runner.get("measure_steps", -1)
        ):
            errors.append("run manifest benchmark window does not match runner result")
        if manifest_distributed.get("strategy") != runner.get(
            "distributed_strategy"
        ):
            errors.append("run manifest strategy does not match runner result")
        if manifest_distributed.get("optimizer_sharding") != runner.get(
            "optimizer_sharding"
        ):
            errors.append("run manifest optimizer sharding does not match runner result")
        if manifest_training.get("ema_shadow_device") != (
            runner.get("ema") or {}
        ).get("shadow_device"):
            errors.append("run manifest EMA device does not match runner result")
        if not bool(manifest_training.get("load_params_only")):
            errors.append("benchmark must initialize with load_params_only=true")
        if manifest.get("model_name") != "protenix-v2":
            errors.append("benchmark model_name must be protenix-v2")
        expected_training = {
            "dtype": "bf16",
            "train_crop_size": crop_size,
            "diffusion_batch_size": workload_profile["diffusion_batch_size"],
            "diffusion_chunk_size": 4,
            "iters_to_accumulate": 1,
            "ema_decay": 0.999,
            "ema_shadow_device": "cpu",
            "lr": expected_lr,
            "lr_warmup_steps": 2000,
            "triangle_attention": "cuequivariance",
            "triangle_multiplicative": "cuequivariance",
            "n_cycle": 4,
            "diffusion_n_step": 20,
        }
        for key, expected_value in expected_training.items():
            if manifest_training.get(key) != expected_value:
                errors.append(
                    f"matched benchmark training setting {key} must be {expected_value!r}"
                )
        expected_model = {
            "c_z": 256,
            "msa_c_m": 128,
            "pairformer_n_blocks": 48,
            "pairformer_blocks_per_ckpt": 2,
            "pairformer_transition_chunk_size": 128,
            "pairformer_transition_checkpoint_chunks": True,
            "diffusion_transition_chunk_size": 128,
            "diffusion_transition_checkpoint_chunks": True,
        }
        for key, expected_value in expected_model.items():
            if manifest_model.get(key) != expected_value:
                errors.append(
                    f"matched benchmark model setting {key} must be {expected_value!r}"
                )
        expected_workload_model = {
            "msa_blocks_per_ckpt": 1,
            "msa_chunk_size": workload_profile["msa_chunk_size"],
            "pairformer_checkpoint_cpu_offload_every_n": workload_profile[
                "pairformer_cpu_offload_every_n"
            ],
            "msa_checkpoint_cpu_offload_every_n": workload_profile[
                "msa_cpu_offload_every_n"
            ],
        }
        for key, expected_value in expected_workload_model.items():
            observed_value = manifest_model.get(key)
            if workload_profile_name == "legacy_msa32" and key not in manifest_model:
                continue
            if observed_value != expected_value:
                errors.append(
                    f"workload profile model setting {key} must be {expected_value!r}"
                )
        if manifest_benchmark.get("exact_crop") != "true":
            errors.append("matched benchmark requires exact_crop=true")
        if crop_size is not None and crop_profile is not None:
            recorded_benchmark_crop = manifest_benchmark.get("crop_size")
            if recorded_benchmark_crop not in {None, crop_size}:
                errors.append("run manifest benchmark crop differs from training crop")
            recorded_benchmark_dataset = manifest_benchmark.get("dataset_name")
            if recorded_benchmark_dataset not in {
                None,
                "",
                crop_profile["dataset_name"],
            }:
                errors.append("run manifest benchmark dataset differs from crop profile")
            if crop_size != 768 and recorded_benchmark_crop != crop_size:
                errors.append(
                    "explicit-crop manifest is missing benchmark crop metadata"
                )
            if crop_size != 768 and recorded_benchmark_dataset != crop_profile[
                "dataset_name"
            ]:
                errors.append(
                    "explicit-crop manifest is missing benchmark dataset metadata"
                )
            recorded_dataset_rows = manifest_benchmark.get("dataset_rows")
            if recorded_dataset_rows not in {None, crop_profile["rows"]}:
                errors.append("run manifest benchmark row count differs from profile")
            if crop_profile["rows"] == 800 and recorded_dataset_rows != 800:
                errors.append("800-row run manifest is missing benchmark row metadata")
        if manifest_benchmark.get("skip_final_checkpoint") is not True:
            errors.append("run manifest requires skip_final_checkpoint=true")
        if manifest_benchmark.get("expected_torch_cuda") != "12.6":
            errors.append("matched benchmark requires PyTorch CUDA 12.6")
        if (
            manifest_benchmark.get(
                "initial_checkpoint_sha256",
                manifest_benchmark.get("official_checkpoint_sha256"),
            )
            != expected_checkpoint_sha256
        ):
            if expected_random_initialization:
                errors.append(
                    "random-initialization manifest must leave checkpoint SHA256 empty"
                )
            else:
                errors.append(
                    "run manifest is missing the workload checkpoint SHA256"
                )
        if manifest_benchmark.get("phase_cuda_memory", "false") != (
            "true" if expected_phase_cuda_memory else "false"
        ):
            errors.append("run manifest phase-memory mode differs from workload profile")
        if manifest_environment.get("cuda_version") != "12.6":
            errors.append("run manifest reports a PyTorch CUDA runtime other than 12.6")
        for compiler_key, expected_compiler in {
            "cc": "/mnt/project_abd/klp/runtime/protenix_cuda126_toolkit/bin/x86_64-conda-linux-gnu-gcc",
            "cxx": "/mnt/project_abd/klp/runtime/protenix_cuda126_toolkit/bin/x86_64-conda-linux-gnu-g++",
        }.items():
            if manifest_environment.get(compiler_key) != expected_compiler:
                errors.append(
                    f"run manifest {compiler_key} must be {expected_compiler}"
                )
        if manifest.get("seed") != 42:
            errors.append("matched benchmark seed must be 42")
        expected_sampler_seed = workload_profile.get("train_sampler_seed", -1)
        if manifest_data.get("train_sampler_seed", -1) != expected_sampler_seed:
            errors.append(
                "matched benchmark train sampler seed must be "
                f"{expected_sampler_seed}"
            )
        manifest_fsdp = manifest_distributed.get("fsdp") or {}
        for key, expected_value in {
            "param_dtype": "fp32",
            "reduce_dtype": "fp32",
            "reshard_after_forward": True,
        }.items():
            if manifest_fsdp.get(key) != expected_value:
                errors.append(
                    f"run manifest FSDP setting {key} must be {expected_value!r}"
                )
        if manifest_fsdp.get("mesh") not in {"auto", "global", "hybrid"}:
            errors.append("run manifest FSDP mesh setting is invalid")
        expected_dataset_name = (
            crop_profile["dataset_name"] if crop_profile is not None else None
        )
        if manifest_data.get("train_sets") != [expected_dataset_name]:
            errors.append(
                "matched benchmark must use the crop-specific long-protein set"
            )
        if manifest_data.get("train_sample_weights") != [1.0]:
            errors.append("matched benchmark exact-crop sample weight must be 1.0")
        if manifest_data.get("test_sets") != [expected_dataset_name]:
            errors.append("matched benchmark test set must match the exact-crop train set")
        if manifest_data.get("num_dl_workers") != 4:
            errors.append("matched benchmark requires four DataLoader workers per rank")
        expected_epoch_size = (
            crop_profile["epoch_size"] if crop_profile is not None else None
        )
        if manifest_data.get("epoch_size") != expected_epoch_size:
            errors.append(
                f"matched benchmark requires data epoch_size={expected_epoch_size}"
            )
        expected_replacement = (
            crop_profile["replacement"] if crop_profile is not None else None
        )
        recorded_replacement = manifest_data.get("train_sampler_replacement")
        if recorded_replacement not in {None, expected_replacement}:
            errors.append(
                "matched benchmark sampler replacement differs from dataset profile"
            )
        if crop_profile is not None and crop_profile["rows"] == 800 and (
            recorded_replacement is not False
        ):
            errors.append("800-row run manifest requires replacement=false")
        manifest_msa = manifest_data.get("msa") or {}
        expected_msa = {
            ("min_size", "train"): workload_profile["msa_min_size"],
            ("max_size", "train"): workload_profile["msa_max_size"],
            ("sample_cutoff", "train"): workload_profile[
                "msa_sample_cutoff"
            ],
            ("pair_msa_as_unpaired_only",): expected_pair_msa_as_unpaired_only,
            ("shuffle_pair_msa_when_unpaired_only",): expected_shuffle_pair_msa,
        }
        for keys, expected_value in expected_msa.items():
            if _nested(manifest_msa, *keys) != expected_value:
                errors.append(
                    "matched benchmark MSA setting "
                    + ".".join(keys)
                    + f" must be {expected_value!r}"
                )
        benchmark_dataset = manifest_data.get("benchmark_dataset", {})
        if benchmark_dataset.get("name") != expected_dataset_name:
            errors.append("manifest benchmark dataset name differs from crop profile")
        crop_config = benchmark_dataset.get("cropping_configs", {})
        expected_crop_config = {
            "crop_size": crop_size,
            "method_weights": [0.0, 1.0, 0.0],
            "max_nuc_tokens": workload_profile["max_nuc_tokens"],
            "contiguous_crop_complete_lig": False,
            "spatial_crop_complete_lig": False,
            "drop_last": False,
            "remove_metal": False,
        }
        for key, expected_value in expected_crop_config.items():
            if crop_config.get(key) != expected_value:
                errors.append(
                    f"exact-crop dataset setting {key} must be {expected_value!r}"
                )
        benchmark_base_info = benchmark_dataset.get("base_info", {})
        expected_min_num_tokens = (
            crop_profile["min_num_tokens"] if crop_profile is not None else -1
        )
        if int(benchmark_base_info.get("min_n_token", -1)) != expected_min_num_tokens:
            errors.append(
                "exact-crop benchmark source must require "
                f"min_n_token={expected_min_num_tokens}"
            )
        expected_random_fallback = (
            crop_profile["random_sample_if_failed"]
            if crop_profile is not None
            else None
        )
        recorded_random_fallback = benchmark_base_info.get(
            "random_sample_if_failed"
        )
        if recorded_random_fallback not in {None, expected_random_fallback}:
            errors.append(
                "benchmark random_sample_if_failed differs from dataset profile"
            )
        if crop_profile is not None and crop_profile["rows"] == 800 and (
            recorded_random_fallback is not False
        ):
            errors.append("800-row benchmark requires random_sample_if_failed=false")
        if benchmark_base_info.get("mmcif_dir") != str(
            OFFICIAL_PROTENIX_ROOT / "mmcif"
        ):
            errors.append("exact-crop benchmark must use the official mmCIF directory")
        if benchmark_base_info.get("bioassembly_dict_dir") != str(
            OFFICIAL_PROTENIX_ROOT / "mmcif_bioassembly"
        ):
            errors.append("exact-crop benchmark must use the official bioassembly directory")
        benchmark_artifacts = benchmark_dataset.get("artifacts") or {}
        selection_summary = benchmark_artifacts.get("selection_summary") or {}
        if benchmark_artifacts.get("metadata_error"):
            errors.append("benchmark dataset artifact metadata could not be read")
        if benchmark_artifacts.get("index_sha256") != selection_summary.get(
            "index_sha256"
        ):
            errors.append("benchmark index SHA256 differs from its selection summary")
        expected_selection = {
            "schema_version": 1,
            "source_index": str(OFFICIAL_SOURCE_INDEX),
            "source_index_sha256": OFFICIAL_SOURCE_INDEX_SHA256,
            "index_sha256": (
                crop_profile["index_sha256"] if crop_profile is not None else None
            ),
            "max_rows": crop_profile["rows"] if crop_profile is not None else None,
            "filter_min_num_tokens": expected_min_num_tokens,
            "filter_max_num_tokens": (
                crop_profile["max_num_tokens"] if crop_profile is not None else None
            ),
            "selection": "top_tokens",
            "seed": 42,
            "rows": crop_profile["rows"] if crop_profile is not None else None,
        }
        for key, expected_value in expected_selection.items():
            if selection_summary.get(key) != expected_value:
                errors.append(
                    f"benchmark dataset selection field {key} must be {expected_value!r}"
                )
        if workload_profile_name != "legacy_msa32":
            expected_msa_artifact_settings = {
                "mapping": str(OFFICIAL_MSA_MAPPING),
                "mapping_sha256": OFFICIAL_MSA_MAPPING_SHA256,
                "bioassembly_dir": str(OFFICIAL_PROTENIX_ROOT / "mmcif_bioassembly"),
                "msa_root": str(OFFICIAL_PROTENIX_ROOT / "mmcif_msa_template"),
                "audit_semantics": "pairing_and_non_pairing_as_unpaired_deduplicated",
                "target_depth": 2048,
                "max_msa_size": 16384,
                "accepted_protein_entities": 1094,
                "accepted_min_effective_depth": 2222,
            }
            for key, expected_value in expected_msa_artifact_settings.items():
                if selection_summary.get(key) != expected_value:
                    errors.append(
                        f"MSA-2048 dataset evidence field {key} must be {expected_value!r}"
                    )
            for path_field, hash_field in {
                "msa_depth_json": "msa_depth_json_sha256",
                "msa_depth_csv": "msa_depth_csv_sha256",
                "msa_depth_candidates_csv": "msa_depth_candidates_csv_sha256",
            }.items():
                artifact_path = selection_summary.get(path_field)
                expected_hash = selection_summary.get(hash_field)
                if (
                    not isinstance(artifact_path, str)
                    or not _is_sha256(expected_hash)
                    or node_artifact_hashes["mapping"].get(artifact_path)
                    != expected_hash
                ):
                    errors.append(
                        f"MSA-2048 dataset artifact {path_field} is missing or has a stale hash"
                    )
        module_locations = manifest_environment.get("module_locations") or {}
        manifest_cwd = Path(str(manifest.get("cwd", "."))).resolve()
        for module_name in ("configs", "protenix"):
            location = module_locations.get(module_name)
            if not isinstance(location, str) or not location:
                errors.append(f"{module_name} module location is missing")
                continue
            try:
                location_path = Path(location).resolve()
                location_path.relative_to(manifest_cwd)
            except (TypeError, ValueError):
                errors.append(
                    f"{module_name} was imported outside the recorded checkout"
                )
                continue
            if not location_path.exists():
                errors.append(f"{module_name} module location does not exist")
        torch_lib_dir = manifest_environment.get("torch_lib_dir")
        ld_library_path = manifest_environment.get("ld_library_path")
        if not isinstance(torch_lib_dir, str) or not torch_lib_dir:
            errors.append("selected Conda PyTorch library directory is missing")
        elif not Path(torch_lib_dir).is_dir():
            errors.append("selected Conda PyTorch library directory does not exist")
        if not isinstance(ld_library_path, str) or not ld_library_path:
            errors.append("LD_LIBRARY_PATH is missing from the run manifest")
        elif isinstance(torch_lib_dir, str) and torch_lib_dir:
            first_library_dir = ld_library_path.split(":", 1)[0]
            if Path(first_library_dir).resolve() != Path(torch_lib_dir).resolve():
                errors.append(
                    "selected Conda torch/lib must be first in LD_LIBRARY_PATH"
                )
        tmpdir = manifest_environment.get("tmpdir")
        if not isinstance(tmpdir, str) or not tmpdir:
            errors.append("TMPDIR is missing from the run manifest")
        elif len(str(Path(tmpdir).resolve())) > 60:
            errors.append("TMPDIR is too long for torch_shm_manager Unix sockets")
        if (
            manifest_environment.get("torch_multiprocessing_sharing_strategy")
            != "file_system"
        ):
            errors.append("PyTorch multiprocessing sharing strategy must be file_system")
        torch_extensions_dir = manifest_environment.get("torch_extensions_dir")
        layer_norm_extension = manifest_environment.get(
            "fused_layer_norm_extension"
        )
        if not isinstance(torch_extensions_dir, str) or not torch_extensions_dir:
            errors.append("TORCH_EXTENSIONS_DIR is missing from the run manifest")
        if not isinstance(layer_norm_extension, str) or not layer_norm_extension:
            errors.append("fused LayerNorm extension path is missing")
        if isinstance(torch_extensions_dir, str) and torch_extensions_dir and isinstance(
            layer_norm_extension, str
        ) and layer_norm_extension:
            try:
                extension_path = Path(layer_norm_extension).resolve()
                extension_path.relative_to(Path(torch_extensions_dir).resolve())
            except ValueError:
                errors.append(
                    "fused LayerNorm extension was not loaded from TORCH_EXTENSIONS_DIR"
                )
            else:
                if not extension_path.is_file():
                    errors.append("fused LayerNorm extension file does not exist")
        code_sha256 = manifest.get("code_sha256") or {}
        if set(code_sha256) != REQUIRED_CODE_HASH_PATHS:
            errors.append("run manifest code SHA256 file set is incomplete")
        elif any(
            not _is_sha256(value)
            for value in code_sha256.values()
        ):
            errors.append("run manifest contains an invalid code SHA256")
        elif any(
            code_tree_mapping.get(path) != digest
            for path, digest in code_sha256.items()
        ):
            errors.append(
                "rank0 run-manifest code hashes differ from node code-tree hashes"
            )
        if manifest.get("initial_checkpoint") != runner.get("initial_checkpoint"):
            errors.append(
                "run manifest initial checkpoint does not match runner result"
            )
        if manifest.get("initial_checkpoint") != expected_initial_checkpoint:
            errors.append("run manifest initial checkpoint differs from workload profile")

    benchmark_id = (
        statuses[0].get("benchmark_id") if statuses else profile_root.name
    )
    return {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "workload_profile": workload_profile_name,
        "profile_root": str(profile_root),
        "status": "passed" if not errors else "failed",
        "validation_errors": errors,
        "nodes": {
            "expected": expected_nodes,
            "observed": len(statuses),
            "max_launcher_wall_seconds": max(
                (int(status.get("wall_seconds", 0)) for status in statuses),
                default=None,
            ),
            "statuses": statuses,
        },
        "runner": runner,
        "run_manifest": manifest,
        "training_run_status": run_status,
        "final_train_metrics": final_train_metrics,
        "training_health": training_health,
        "pytorch_allocator_note": (
            "runner max_allocated/max_reserved are reset at the measurement "
            "window boundary and cover that synchronized window"
        ),
        "gpu_isolation_note": (
            "compute-app PIDs are classified by process ancestry relative to the "
            "node benchmark launcher; any unrelated process in the measured "
            "window fails validation"
        ),
        "compute_processes": compute_apps,
        "preflight_compute_processes": preflight_compute_apps,
        "node_evidence": {
            "code_tree": node_code_tree,
            "artifacts": node_artifact_hashes,
        },
        "nvml_device": nvml,
        "host_system": system,
        "artifacts": {
            "runner_result": str(runner_path) if runner_path else None,
            "run_status": str(run_status_path) if run_status_path else None,
            "run_manifest": str(manifest_path) if manifest_path else None,
            "resolved_config": str(config_path) if config_path else None,
            "train_metrics": (
                str(train_metrics_path) if train_metrics_path else None
            ),
            "node_statuses": [str(path) for path in status_paths],
            "compute_apps": [
                str(path)
                for path in sorted(
                    profile_root.glob("nodes/node_*/compute_apps.csv")
                )
            ],
            "code_tree_hashes": [
                str(path)
                for path in sorted(
                    profile_root.glob("nodes/node_*/code_tree.sha256")
                )
            ],
            "artifact_hashes": [
                str(path)
                for path in sorted(
                    profile_root.glob("nodes/node_*/artifact_hashes.sha256")
                )
            ],
        },
    }


def _gib(value_mib: int | float | None) -> str:
    if value_mib is None:
        return "n/a"
    return f"{float(value_mib) / MIB_PER_GIB:.2f} GiB"


def render_markdown(summary: dict[str, Any]) -> str:
    runner = summary.get("runner") or {}
    manifest = summary.get("run_manifest") or {}
    manifest_training = manifest.get("training") or {}
    manifest_model = manifest.get("model") or {}
    manifest_environment = manifest.get("environment") or {}
    params = runner.get("model_parameters") or {}
    measured_nvml = summary["nvml_device"]["measurement_window"]
    all_nvml = summary["nvml_device"]["all_samples"]
    max_allocated = runner.get("max_allocated_bytes")
    max_reserved = runner.get("max_reserved_bytes")
    allocated_mib = (
        max_allocated / (1024 * 1024) if max_allocated is not None else None
    )
    reserved_mib = (
        max_reserved / (1024 * 1024) if max_reserved is not None else None
    )
    warmup = runner.get("warmup_steps", "n/a")
    measured = runner.get("measure_steps", "n/a")
    trainable = params.get("trainable", "n/a")
    frozen = params.get("frozen", "n/a")
    measured_hottest = measured_nvml.get("hottest_gpu_peak_mib")
    launcher_wall = summary["nodes"].get("max_launcher_wall_seconds", "n/a")
    elapsed = runner.get("elapsed_seconds")
    initial_checkpoint = runner.get("initial_checkpoint") or {}
    training_health = summary.get("training_health") or {}
    runtime_window = (
        (summary.get("final_train_metrics") or {}).get("runtime_window") or {}
    )
    diagnostic_timings = runtime_window.get("timings") or {}
    token_composition = runner.get("token_composition_coverage") or {}
    nuc_token_distribution = token_composition.get("nuc_token_count") or {}
    compute_processes = summary.get("compute_processes") or {}
    seconds_per_step = (
        elapsed / measured
        if isinstance(elapsed, (int, float))
        and isinstance(measured, (int, float))
        and measured > 0
        else "n/a"
    )

    lines = [
        f"# Benchmark summary: {summary['benchmark_id']}",
        "",
        f"- Status: **{summary['status']}**",
        f"- Nodes: {summary['nodes']['observed']} / {summary['nodes']['expected']}",
        f"- World size: {runner.get('world_size', 'n/a')}",
        f"- Variant: {runner.get('benchmark_variant', 'n/a')}",
        f"- Workload profile: {summary.get('workload_profile', 'n/a')}",
        f"- Warm-up / measured steps: {warmup} / {measured}",
        f"- Distributed strategy: {runner.get('distributed_strategy', 'n/a')}",
        f"- Resolved FSDP mesh: {runner.get('resolved_fsdp_mesh', 'n/a')}",
        "- PyTorch / CUDA runtime: "
        f"{manifest_environment.get('torch_version', 'n/a')} / "
        f"{manifest_environment.get('cuda_version', 'n/a')}",
        "- c_z / MSA c_m: "
        f"{manifest_model.get('c_z', 'n/a')} / "
        f"{manifest_model.get('msa_c_m', 'n/a')}",
        "- Crop size / diffusion batch: "
        f"{manifest_training.get('train_crop_size', 'n/a')} / "
        f"{manifest_training.get('diffusion_batch_size', 'n/a')}",
        "- Pairformer blocks/checkpoint: "
        f"{manifest_model.get('pairformer_blocks_per_ckpt', 'n/a')}",
        "- MSA blocks/checkpoint / chunk / Pairformer offload / MSA offload: "
        f"{manifest_model.get('msa_blocks_per_ckpt', 'n/a')} / "
        f"{manifest_model.get('msa_chunk_size', 'n/a')} / "
        f"{manifest_model.get('pairformer_checkpoint_cpu_offload_every_n', 'n/a')} / "
        f"{manifest_model.get('msa_checkpoint_cpu_offload_every_n', 'n/a')}",
        "- Pair Transition chunk/checkpoint: "
        f"{manifest_model.get('pairformer_transition_chunk_size', 'n/a')} / "
        f"{manifest_model.get('pairformer_transition_checkpoint_chunks', 'n/a')}",
        f"- EMA: {runner.get('ema', 'n/a')}",
        f"- MSA depth coverage: {runner.get('msa_depth_coverage', 'n/a')}",
        "- Nucleic token cap / observed max / violations / complete: "
        f"{token_composition.get('target_max_nuc_tokens', 'n/a')} / "
        f"{nuc_token_distribution.get('max', 'n/a')} / "
        f"{token_composition.get('over_nuc_limit_count', 'n/a')} / "
        f"{token_composition.get('complete_composition_count', 'n/a')}",
        f"- Trainable / frozen parameters: {trainable} / {frozen}",
        f"- Measurement seconds: {runner.get('elapsed_seconds', 'n/a')}",
        f"- Measurement seconds/step: {seconds_per_step}",
        f"- Samples/s: {runner.get('samples_per_sec', 'n/a')}",
        f"- Tokens/s: {runner.get('tokens_per_sec', 'n/a')}",
        f"- Atoms/s: {runner.get('atoms_per_sec', 'n/a')}",
        f"- Rank elapsed imbalance: {runner.get('rank_elapsed_imbalance', 'n/a')}",
        f"- Slowest rank: {runner.get('slowest_rank', 'n/a')}",
        f"- Initial model checkpoint: {initial_checkpoint.get('model', 'n/a')}",
        f"- Initial EMA checkpoint: {initial_checkpoint.get('ema', 'n/a')}",
        f"- Training health: {training_health}",
        f"- PyTorch max allocated: {_gib(allocated_mib)}",
        f"- PyTorch max reserved: {_gib(reserved_mib)}",
        f"- NVML hottest GPU (measurement): {_gib(measured_hottest)}",
        f"- NVML hottest GPU (whole job): {_gib(all_nvml.get('hottest_gpu_peak_mib'))}",
        "- GPU compute contexts (rank / auxiliary / unclassified / external): "
        f"{len(compute_processes.get('training_rank_processes') or [])} / "
        f"{len(compute_processes.get('auxiliary_processes') or [])} / "
        f"{len(compute_processes.get('unclassified_processes') or [])} / "
        f"{len(compute_processes.get('external_processes') or [])}",
        f"- Launcher wall seconds: {launcher_wall}",
    ]
    if summary["validation_errors"]:
        lines.extend(["", "## Validation errors", ""])
        lines.extend(f"- {error}" for error in summary["validation_errors"])
    if diagnostic_timings:
        lines.extend(
            [
                "",
                "## Diagnostic phase timings",
                "",
                "These unsynchronized phase timings cover the final logging "
                "window (normally warm-up plus measurement), not only the "
                "synchronized measurement window.",
                "",
                "| Phase | Average seconds | P95 seconds |",
                "| --- | ---: | ---: |",
            ]
        )
        for phase in [
            "to_device_seconds",
            "forward_seconds",
            "loss_seconds",
            "backward_seconds",
            "optimizer_seconds",
            "ema_seconds",
            "train_step_seconds",
            "step_wall_seconds",
        ]:
            stats = diagnostic_timings.get(phase) or {}
            lines.append(
                f"| {phase} | {stats.get('avg', 'n/a')} | "
                f"{stats.get('p95', 'n/a')} |"
            )
    lines.extend(
        [
            "",
            "PyTorch allocator peaks are measurement-window values. NVML is sampled ",
            "device-level memory and may include CUDA/NCCL or unrelated processes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    summary = build_summary(args.profile_root)
    output_json = args.output_json or args.profile_root / "summary.json"
    output_md = args.output_md or args.profile_root / "summary.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
