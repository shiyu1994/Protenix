#!/usr/bin/env bash
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
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"
# Put this checkout first. AIStudio shells are often reused across experiments,
# so an inherited PYTHONPATH may still point at train3 or fast-training.
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/mnt/project_abd/klp/miniconda3/bin/activate}"
CONDA_ENV="${CONDA_ENV:-protenix}"
mp_sharing_strategy="${PROTENIX_MP_SHARING_STRATEGY-file_system}"
case "${mp_sharing_strategy}" in
    file_descriptor|file_system)
        ;;
    *)
        echo "PROTENIX_MP_SHARING_STRATEGY must be file_descriptor or file_system; got '${mp_sharing_strategy}'" >&2
        exit 2
        ;;
esac
export PROTENIX_MP_SHARING_STRATEGY="${mp_sharing_strategy}"
if [ ! -f "${CONDA_ACTIVATE}" ]; then
    echo "Missing conda activate script: ${CONDA_ACTIVATE}" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${CONDA_ACTIVATE}" "${CONDA_ENV}"
CONDA_ROOT="${CONDA_ACTIVATE%/bin/activate}"
CONDA_ENV_BIN="${CONDA_ROOT}/envs/${CONDA_ENV}/bin"
if [ ! -d "${CONDA_ENV_BIN}" ]; then
    echo "Missing conda env bin directory: ${CONDA_ENV_BIN}" >&2
    exit 1
fi
export PATH="${CONDA_ENV_BIN}:${PATH}"
hash -r

# Re-assert the locked CUDA environment after Conda activation has modified
# PATH. This deliberately overrides an inherited /usr/local/cuda (cu124 on
# some machines, and an nvcc-less compatibility directory on h1003).
# shellcheck disable=SC1091
source "${repo_dir}/scripts/runtime/cuda126_env.sh"
requested_cc="${PROTENIX_CUDA_HOST_CC}"
requested_cxx="${PROTENIX_CUDA_HOST_CXX}"
if [ ! -x "${requested_cc}" ] || [ ! -x "${requested_cxx}" ]; then
    echo "Pinned CUDA host compilers are not executable." >&2
    exit 2
fi
export CC="${requested_cc}"
export CXX="${requested_cxx}"
hash -r

# Even callers that provide their own cache root are confined to a
# toolchain+source fingerprint. This prevents direct import of a .so produced
# by CUDA 12.4 or by older LayerNorm sources.
node_rank_for_extension_cache="${NODE_RANK:-${RANK:-${MLP_ROLE_INDEX:-0}}}"
if [ -z "${TORCH_EXTENSIONS_DIR:-}" ]; then
    TORCH_EXTENSIONS_DIR="/tmp/protenix_torch_extensions/finetune/node_${node_rank_for_extension_cache}"
fi
case "/${TORCH_EXTENSIONS_DIR#/}/" in
    *"/${PROTENIX_LAYER_NORM_CACHE_FINGERPRINT}/"*) ;;
    *)
        TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR%/}/${PROTENIX_LAYER_NORM_CACHE_FINGERPRINT}"
        ;;
esac
export TORCH_EXTENSIONS_DIR
if [ -n "${PROTENIX_LAYER_NORM_BUILD_DIR:-}" ]; then
    echo "PROTENIX_LAYER_NORM_BUILD_DIR is not allowed by the pinned cache contract." >&2
    exit 2
fi
mkdir -p "${TORCH_EXTENSIONS_DIR}"

# AIStudio prepends the system Python's PyTorch libraries to LD_LIBRARY_PATH.
# That is unsafe after activating another PyTorch build: torch_shm_manager is
# an executable and would otherwise load the system libshm/libc10 instead of
# the libraries shipped with the selected Conda environment.  The mismatch is
# especially visible when DataLoader workers use the file_system sharing
# strategy.  Put this environment's torch/lib first and verify it below before
# torchrun starts.
python_purelib="$(python - <<'PY'
import sysconfig

print(sysconfig.get_path("purelib"))
PY
)"
torch_lib_dir="${python_purelib}/torch/lib"
torch_shm_manager="${python_purelib}/torch/bin/torch_shm_manager"
if [ ! -d "${torch_lib_dir}" ]; then
    echo "Missing PyTorch runtime libraries in selected environment: ${python_purelib}/torch" >&2
    exit 1
fi
if [ "${mp_sharing_strategy}" = "file_system" ] && [ ! -x "${torch_shm_manager}" ]; then
    echo "Missing torch_shm_manager required by file_system sharing: ${torch_shm_manager}" >&2
    exit 1
fi
export PROTENIX_TORCH_LIB_DIR="${torch_lib_dir}"
export LD_LIBRARY_PATH="${torch_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# A system CUDA toolkit upgrade does not change the CUDA runtime embedded in
# the active PyTorch wheel. Benchmarks may set EXPECTED_TORCH_CUDA (for example
# 12.6) so a mislabeled run fails before allocating model/GPU memory.
if [ "${DRY_RUN:-0}" != "1" ]; then
    python - "${EXPECTED_TORCH_CUDA:-}" "${repo_dir}" "${mp_sharing_strategy}" <<'PY'
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys

import torch
from torch.utils.cpp_extension import CUDA_HOME as detected_cuda_home

expected = sys.argv[1]
expected_repo = Path(sys.argv[2]).resolve()
sharing_strategy = sys.argv[3]
supported_sharing_strategies = {"file_descriptor", "file_system"}
if sharing_strategy not in supported_sharing_strategies:
    supported = ", ".join(sorted(supported_sharing_strategies))
    raise SystemExit(
        "PROTENIX_MP_SHARING_STRATEGY must be one of "
        f"{{{supported}}}, got {sharing_strategy!r}"
    )
module_locations = {}
for module_name in ("configs", "protenix"):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise SystemExit(f"Cannot resolve required module {module_name!r}")
    location = spec.origin
    if location is None and spec.submodule_search_locations:
        location = next(iter(spec.submodule_search_locations), None)
    resolved = Path(location).resolve() if location else None
    module_locations[module_name] = str(resolved) if resolved else None
    if resolved is None or expected_repo not in (resolved, *resolved.parents):
        raise SystemExit(
            f"{module_name} resolves outside the selected checkout: {resolved}; "
            f"expected under {expected_repo}"
        )
runtime = {
    "python": sys.executable,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cudnn": torch.backends.cudnn.version(),
    "module_locations": module_locations,
    "torch_lib_dir": str((Path(torch.__file__).resolve().parent / "lib").resolve()),
    "cuda_home": str(Path(detected_cuda_home or "").resolve()),
    "nvcc": shutil.which("nvcc"),
    "cc": os.environ.get("CC"),
    "cxx": os.environ.get("CXX"),
    "cuda_arch_list": os.environ.get("PROTENIX_CUDA_ARCH_LIST"),
    "cuda_toolchain_fingerprint": os.environ.get(
        "PROTENIX_CUDA_TOOLCHAIN_FINGERPRINT"
    ),
    "layer_norm_cache_fingerprint": os.environ.get(
        "PROTENIX_LAYER_NORM_CACHE_FINGERPRINT"
    ),
}
soft_fd_limit, hard_fd_limit = resource.getrlimit(resource.RLIMIT_NOFILE)


def json_fd_limit(value):
    return "infinity" if value == resource.RLIM_INFINITY else int(value)


runtime["fd_limit_soft"] = json_fd_limit(soft_fd_limit)
runtime["fd_limit_hard"] = json_fd_limit(hard_fd_limit)
runtime["fd_limit_checked"] = True
print("Protenix runtime: " + json.dumps(runtime, sort_keys=True))
if not runtime["cuda_available"]:
    raise SystemExit("CUDA is not available in the selected Python environment")
if expected and runtime["torch_cuda"] != expected:
    raise SystemExit(
        f"Expected PyTorch CUDA {expected}, got {runtime['torch_cuda']} "
        f"from {runtime['python']}"
    )
expected_torch_version = os.environ.get("PROTENIX_EXPECTED_TORCH_VERSION", "")
if not expected_torch_version or runtime["torch"] != expected_torch_version:
    raise SystemExit(
        "Expected the pinned PyTorch build "
        f"{expected_torch_version!r}, got {runtime['torch']!r}"
    )
expected_cuda_home = Path(
    os.environ.get("PROTENIX_CUDA126_HOME", "")
).resolve()
if not str(expected_cuda_home) or Path(runtime["cuda_home"]) != expected_cuda_home:
    raise SystemExit(
        "torch.utils.cpp_extension selected the wrong CUDA toolkit: "
        f"detected={runtime['cuda_home']}, expected={expected_cuda_home}"
    )
expected_tools = {
    "nvcc": expected_cuda_home / "bin" / "nvcc",
    "cc": Path(os.environ.get("PROTENIX_CUDA_HOST_CC", "")),
    "cxx": Path(os.environ.get("PROTENIX_CUDA_HOST_CXX", "")),
}
for tool_name, expected_path in expected_tools.items():
    observed_path = runtime[tool_name]
    if not observed_path or Path(observed_path).resolve() != expected_path.resolve():
        raise SystemExit(
            f"Pinned CUDA tool mismatch for {tool_name}: "
            f"observed={observed_path}, expected={expected_path}"
        )
nvcc_version = subprocess.run(
    [runtime["nvcc"], "--version"],
    check=True,
    capture_output=True,
    text=True,
).stdout
if "release 12.6, V12.6.85" not in nvcc_version:
    raise SystemExit("Pinned nvcc version check failed")
if runtime["cuda_arch_list"] != "9.0":
    raise SystemExit(
        "PROTENIX_CUDA_ARCH_LIST must be exactly 9.0 for this H100 profile"
    )
device_capabilities = [
    list(torch.cuda.get_device_capability(index))
    for index in range(torch.cuda.device_count())
]
expected_local_world_size = int(
    os.environ.get("PROTENIX_EXPECTED_LOCAL_WORLD_SIZE", "0")
)
if (
    expected_local_world_size > 0
    and torch.cuda.device_count() != expected_local_world_size
):
    raise SystemExit(
        "Visible GPU count does not match the locked local topology: "
        f"visible={torch.cuda.device_count()}, "
        f"expected={expected_local_world_size}"
    )
if not device_capabilities or any(
    capability != [9, 0] for capability in device_capabilities
):
    raise SystemExit(
        "This locked profile requires only H100/sm90 visible GPUs; "
        f"got capabilities={device_capabilities}"
    )
runtime["device_capabilities"] = device_capabilities
if not runtime["cuda_toolchain_fingerprint"] or not runtime[
    "layer_norm_cache_fingerprint"
]:
    raise SystemExit("CUDA toolchain/cache fingerprints are missing")

if sharing_strategy == "file_descriptor":
    minimum_fd_limit = 1024
    runtime["fd_limit_minimum"] = minimum_fd_limit
    if soft_fd_limit != resource.RLIM_INFINITY and soft_fd_limit < minimum_fd_limit:
        raise SystemExit(
            "file_descriptor sharing requires RLIMIT_NOFILE soft limit >= "
            f"{minimum_fd_limit}, got {soft_fd_limit}; raise `ulimit -n` or use "
            "PROTENIX_MP_SHARING_STRATEGY=file_system"
        )
    if soft_fd_limit != resource.RLIM_INFINITY and soft_fd_limit < 4096:
        runtime["fd_limit_warning"] = (
            "finite soft limit is below the recommended 4096 descriptors"
        )
    runtime["torch_shm_manager"] = "not_required"
else:
    # torch_shm_manager runs outside the Python process. Confirm the dynamic
    # linker selected this exact wheel's ABI-matched libshm/libc10 rather than
    # an AIStudio system PyTorch installation inherited through
    # LD_LIBRARY_PATH.
    torch_root = Path(torch.__file__).resolve().parent
    torch_lib_dir = (torch_root / "lib").resolve()
    shm_manager = (torch_root / "bin" / "torch_shm_manager").resolve()
    linker_output = subprocess.run(
        ["ldd", str(shm_manager)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    resolved_shm_libraries = {}
    for library_name in ("libshm.so", "libc10.so", "libtorch_cpu.so"):
        match = re.search(
            rf"^\s*{re.escape(library_name)}\s+=>\s+(\S+)",
            linker_output,
            flags=re.MULTILINE,
        )
        if match is None:
            raise SystemExit(f"Cannot resolve {library_name} for {shm_manager}")
        resolved_path = Path(match.group(1)).resolve()
        resolved_shm_libraries[library_name] = str(resolved_path)
        try:
            resolved_path.relative_to(torch_lib_dir)
        except ValueError as exc:
            raise SystemExit(
                f"PyTorch ABI mismatch: {shm_manager} resolves {library_name} to "
                f"{resolved_path}, expected a library under {torch_lib_dir}"
            ) from exc
    runtime["torch_shm_manager"] = str(shm_manager)
    runtime["torch_shm_manager_libraries"] = resolved_shm_libraries

torch.multiprocessing.set_sharing_strategy(sharing_strategy)
actual_sharing_strategy = torch.multiprocessing.get_sharing_strategy()
if actual_sharing_strategy != sharing_strategy:
    raise SystemExit(
        "PyTorch multiprocessing sharing strategy mismatch: "
        f"requested={sharing_strategy!r}, actual={actual_sharing_strategy!r}"
    )
runtime["torch_multiprocessing_sharing_strategy"] = actual_sharing_strategy
shared_probe = torch.arange(8, dtype=torch.int64)
shared_probe.share_memory_()
if not shared_probe.is_shared():
    raise SystemExit(
        f"PyTorch {sharing_strategy} shared-memory preflight failed"
    )
probe_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.arange(4, dtype=torch.int64)),
    batch_size=2,
    num_workers=1,
    timeout=30,
)
probe_batch = next(iter(probe_loader))[0]
if probe_batch.tolist() != [0, 1]:
    raise SystemExit("PyTorch DataLoader shared-memory roundtrip produced bad data")
runtime[f"{sharing_strategy}_sharing_smoke"] = "passed"
runtime["multiprocessing_sharing_smoke"] = "passed"
runtime["dataloader_queue_smoke"] = "passed"

# Compile/import the fused CUDA LayerNorm once before torchrun starts. The
# extension cache is node-local in the benchmark launcher, so local ranks only
# load the verified artifact instead of racing to rebuild it.
import protenix.model.layer_norm.layer_norm as layer_norm_module

layer = layer_norm_module.FusedLayerNorm(8).cuda()
sample = torch.randn(2, 8, device="cuda", dtype=torch.float32, requires_grad=True)
output = layer(sample)
output.square().mean().backward()
torch.cuda.synchronize()
if not torch.isfinite(output).all() or not torch.isfinite(sample.grad).all():
    raise SystemExit("FusedLayerNorm CUDA preflight produced non-finite values")
runtime["fused_layer_norm_extension"] = str(
    Path(layer_norm_module.fast_layer_norm_cuda_v2.__file__).resolve()
)
extension_path = Path(runtime["fused_layer_norm_extension"])
extensions_root = Path(os.environ.get("TORCH_EXTENSIONS_DIR", "")).resolve()
try:
    extension_path.relative_to(extensions_root)
except ValueError as exc:
    raise SystemExit(
        "FusedLayerNorm loaded outside TORCH_EXTENSIONS_DIR: "
        f"{extension_path} is not under {extensions_root}"
    ) from exc
provenance_path = (
    extension_path.parent / "protenix_cuda_extension_provenance.json"
)
if not provenance_path.is_file():
    raise SystemExit(
        f"FusedLayerNorm cache provenance is missing: {provenance_path}"
    )
runtime["fused_layer_norm_provenance"] = str(provenance_path)
runtime["fused_layer_norm_smoke"] = "passed"
print("Protenix CUDA extension preflight: " + json.dumps(runtime, sort_keys=True))
PY
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_TIMEOUT_SECOND="${NCCL_TIMEOUT_SECOND:-1800}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
default_2026_root="/mnt/project_abd/public_databases/protenix.20260101"
local_2026_root="${repo_dir}/../../../../../public_databases/protenix.20260101"
if [ -z "${PROTENIX_ROOT_DIR:-}" ]; then
    if [ -d "${default_2026_root}" ]; then
        export PROTENIX_ROOT_DIR="${default_2026_root}"
    elif [ -d "${local_2026_root}" ]; then
        export PROTENIX_ROOT_DIR="${local_2026_root}"
    else
        export PROTENIX_ROOT_DIR="${default_2026_root}"
    fi
fi

checkpoint_path="${checkpoint_path:-${CHECKPOINT_PATH:-${PROTENIX_ROOT_DIR}/checkpoint/protenix-v2.pt}}"
load_checkpoint_path="${load_checkpoint_path-${LOAD_CHECKPOINT_PATH-${checkpoint_path}}}"
load_ema_checkpoint_path="${load_ema_checkpoint_path-${LOAD_EMA_CHECKPOINT_PATH-${checkpoint_path}}}"
base_dir="${base_dir:-${BASE_DIR:-/mnt/project_abd/klp/proteinx_output}}"
run_name="${run_name:-${RUN_NAME:-protenix_finetune_20260101_weightedpdb_lr1e-5}}"
model_name="${model_name:-${MODEL_NAME:-protenix-v2}}"

nproc_per_node="${nproc_per_node:-${NPROC_PER_NODE:-gpu}}"
nnodes="${nnodes:-${NNODES:-${MLP_WORKER_NUM:-${SLURM_NNODES:-1}}}}"
node_rank="${node_rank:-${NODE_RANK:-${MLP_ROLE_INDEX:-${SLURM_NODEID:-0}}}}"
master_addr="${master_addr:-${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}}"
master_port="${master_port:-${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}}"
if [ "${nnodes}" != "1" ] && [ "${master_addr}" = "127.0.0.1" ]; then
    echo "Multi-node training requires MASTER_ADDR or MLP_WORKER_0_HOST." >&2
    exit 1
fi

weighted_train_set="${weighted_train_set:-${WEIGHTED_TRAIN_SET:-weightedPDB_before210930_v20260101}}"
weighted_indices_default="${PROTENIX_ROOT_DIR}/indices/indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz"
if [ "${weighted_train_set}" = "weightedPDB_before250701_v20260101" ]; then
    weighted_indices_default="${PROTENIX_ROOT_DIR}/indices/indices_20260107-20chains_before_2025-07-01_res4.5.csv.gz"
fi
weighted_mmcif_dir="${WEIGHTED_MMCIF_DIR:-${PROTENIX_ROOT_DIR}/mmcif}"
weighted_bioassembly_dir="${WEIGHTED_BIOASSEMBLY_DIR:-${PROTENIX_ROOT_DIR}/mmcif_bioassembly}"
weighted_indices_fpath="${WEIGHTED_INDICES_FPATH:-${weighted_indices_default}}"
kalign_binary_path="${KALIGN_BINARY_PATH:-$(command -v kalign || true)}"
if [ -z "${kalign_binary_path}" ]; then
    kalign_binary_path="/usr/bin/kalign"
fi

sabdab_root_dir="${sabdab_root_dir:-${SABDAB_ROOT_DIR:-/mnt/project_abd/zhe/protenix_copy/vepfs-mlp2/project-abd/zhujianwei/public_databases/protenix}}"
export SABDAB_ROOT_DIR="${sabdab_root_dir}"
sabdab_train_mmcif_dir="${SABDAB_TRAIN_MMCIF_DIR:-${sabdab_root_dir}/sabdab-train_mmcif}"
sabdab_train_bioassembly_dir="${SABDAB_TRAIN_BIOASSEMBLY_DIR:-${sabdab_root_dir}/sabdab-train_bioassembly}"
sabdab_train_indices_fpath="${SABDAB_TRAIN_INDICES_FPATH:-${sabdab_root_dir}/indices/sabdab-train.csv}"
sabdab_train_seq_index_json="${SABDAB_TRAIN_SEQ_INDEX_JSON:-${sabdab_root_dir}/sabdab-train.json}"
distill_sabdab_mmcif_dir="${DISTILL_SABDAB_MMCIF_DIR:-${sabdab_root_dir}/distill-sabdab_cif}"
distill_sabdab_bioassembly_dir="${DISTILL_SABDAB_BIOASSEMBLY_DIR:-${sabdab_root_dir}/distill-sabdab_bioassembly}"
distill_sabdab_indices_fpath="${DISTILL_SABDAB_INDICES_FPATH:-${sabdab_root_dir}/indices/distill-sabdab.csv}"
distill_sabdab_seq_index_json="${DISTILL_SABDAB_SEQ_INDEX_JSON:-${sabdab_root_dir}/sab-20260416.json}"
sabdab_msa_template_dir="${SABDAB_MSA_TEMPLATE_DIR:-${sabdab_root_dir}/mmcif_msa_template}"

posebusters_mmcif_dir="${POSEBUSTERS_MMCIF_DIR:-${PROTENIX_ROOT_DIR}/posebusters_mmcif}"
posebusters_bioassembly_dir="${POSEBUSTERS_BIOASSEMBLY_DIR:-${PROTENIX_ROOT_DIR}/posebusters_bioassembly}"
posebusters_indices_fpath="${POSEBUSTERS_INDICES_FPATH:-${PROTENIX_ROOT_DIR}/indices/posebusters_indices_mainchain_interface.csv}"
default_train_sets="${weighted_train_set},sabdab-train,distill-sabdab"
default_train_sample_weights="0.5,0.3,0.2"
resolved_train_sets="${train_sets:-${default_train_sets}}"
resolved_train_sample_weights="${train_sample_weights:-${default_train_sample_weights}}"
resolved_test_sets="${test_sets:-posebusters_0925}"
IFS=',' read -r -a resolved_train_set_names <<< "${resolved_train_sets}"
IFS=',' read -r -a resolved_test_set_names <<< "${resolved_test_sets}"

train_set_is_selected() {
    local candidate="$1"
    local selected
    for selected in "${resolved_train_set_names[@]}"; do
        if [ "${selected}" = "${candidate}" ]; then
            return 0
        fi
    done
    return 1
}

test_set_is_selected() {
    local candidate="$1"
    local selected
    for selected in "${resolved_test_set_names[@]}"; do
        if [ "${selected}" = "${candidate}" ]; then
            return 0
        fi
    done
    return 1
}

benchmark_exact_crop="${benchmark_exact_crop:-${BENCHMARK_EXACT_CROP:-false}}"
if [ "${benchmark_exact_crop}" != "true" ] && [ "${benchmark_exact_crop}" != "false" ]; then
    echo "BENCHMARK_EXACT_CROP must be true or false" >&2
    exit 2
fi
benchmark_crop_size="${BENCHMARK_CROP_SIZE:-${train_crop_size:-${BENCHMARK_TRAIN_CROP_SIZE:-768}}}"
if [ "${benchmark_exact_crop}" = "true" ] && [ -n "${train_crop_size:-}" ] && \
   [ "${train_crop_size}" != "${benchmark_crop_size}" ]; then
    echo "train_crop_size and BENCHMARK_CROP_SIZE disagree" >&2
    exit 2
fi
benchmark_dataset_name="${BENCHMARK_DATASET_NAME:-benchmark_long_protein_context_${benchmark_crop_size}}"
case "${benchmark_crop_size}:${benchmark_dataset_name}" in
    384:benchmark_long_protein_context_384_800_msa2048)
        # This logical view reuses the crop1024 MSA-aware index, whose durable
        # audit intentionally keeps the stricter min_n_token=1025 threshold.
        benchmark_min_num_tokens=1025
        ;;
    768:benchmark_long_protein_context_768)
        benchmark_min_num_tokens=769
        ;;
    960:benchmark_long_protein_context_960_800_msa2048)
        # This view reuses the crop1024 MSA-aware index, whose durable audit
        # was intentionally selected and hashed with min_n_token=1025.
        benchmark_min_num_tokens=1025
        ;;
    1024:benchmark_long_protein_context_1024|1024:benchmark_long_protein_context_1024_800|1024:benchmark_long_protein_context_1024_800_msa2048)
        benchmark_min_num_tokens=1025
        ;;
    *)
        if [ "${benchmark_exact_crop}" = "true" ]; then
            echo "Exact-crop benchmark requires a matched 384, 768, 960, or 1024 dataset profile" >&2
            exit 2
        fi
        benchmark_min_num_tokens=769
        ;;
esac
benchmark_dataset_dir="${BENCHMARK_DATASET_DIR:-${PROTENIX_BENCHMARK_DIR:-${repo_dir}/data/${benchmark_dataset_name}}}"
benchmark_indices_fpath="${benchmark_dataset_dir}/indices.csv"
if [ "${benchmark_exact_crop}" = "true" ]; then
    required_paths=(
        "${weighted_mmcif_dir}"
        "${weighted_bioassembly_dir}"
        "${benchmark_indices_fpath}"
        "${PROTENIX_ROOT_DIR}/common/seq_to_pdb_index.json"
        "${PROTENIX_ROOT_DIR}/mmcif_msa_template"
        "${PROTENIX_ROOT_DIR}/rna_msa/rna_sequence_to_pdb_chains.json"
        "${PROTENIX_ROOT_DIR}/rna_msa/msas"
        "${PROTENIX_ROOT_DIR}/common/release_date_cache.json"
        "${PROTENIX_ROOT_DIR}/common/obsolete_to_successor.json"
        "${kalign_binary_path}"
    )
else
    required_paths=()
    if train_set_is_selected "${weighted_train_set}"; then
        required_paths+=(
            "${weighted_mmcif_dir}"
            "${weighted_bioassembly_dir}"
            "${weighted_indices_fpath}"
            "${PROTENIX_ROOT_DIR}/common/seq_to_pdb_index.json"
            "${PROTENIX_ROOT_DIR}/mmcif_msa_template"
            "${PROTENIX_ROOT_DIR}/rna_msa/rna_sequence_to_pdb_chains.json"
            "${PROTENIX_ROOT_DIR}/rna_msa/msas"
            "${PROTENIX_ROOT_DIR}/common/release_date_cache.json"
            "${PROTENIX_ROOT_DIR}/common/obsolete_to_successor.json"
            "${kalign_binary_path}"
        )
    fi
    if train_set_is_selected "sabdab-train"; then
        required_paths+=(
            "${sabdab_train_mmcif_dir}"
            "${sabdab_train_bioassembly_dir}"
            "${sabdab_train_indices_fpath}"
            "${sabdab_train_seq_index_json}"
            "${sabdab_msa_template_dir}"
        )
    fi
    if train_set_is_selected "distill-sabdab"; then
        required_paths+=(
            "${distill_sabdab_mmcif_dir}"
            "${distill_sabdab_bioassembly_dir}"
            "${distill_sabdab_indices_fpath}"
            "${distill_sabdab_seq_index_json}"
            "${sabdab_msa_template_dir}"
        )
    fi
    if test_set_is_selected "posebusters_0925"; then
        required_paths+=(
            "${posebusters_mmcif_dir}"
            "${posebusters_bioassembly_dir}"
            "${posebusters_indices_fpath}"
        )
    fi
fi
required_paths+=(
    "${PROTENIX_ROOT_DIR}/common/components.cif"
    "${PROTENIX_ROOT_DIR}/common/components.cif.rdkit_mol.pkl"
    "${PROTENIX_ROOT_DIR}/common/obsolete_release_date.csv"
    "${PROTENIX_ROOT_DIR}/common/clusters-by-entity-40.txt"
)
extra_train_data_required_paths="${PROTENIX_EXTRA_TRAIN_DATA_REQUIRED_PATHS:-}"
if [ -n "${extra_train_data_required_paths}" ]; then
    IFS=':' read -r -a extra_train_data_required_path_names \
        <<< "${extra_train_data_required_paths}"
    required_paths+=("${extra_train_data_required_path_names[@]}")
fi
if [ -n "${load_checkpoint_path}" ]; then
    required_paths+=("${load_checkpoint_path}")
fi
if [ -n "${load_ema_checkpoint_path}" ]; then
    required_paths+=("${load_ema_checkpoint_path}")
fi

if [ "${DRY_RUN:-0}" != "1" ] && [ "${SKIP_PATH_CHECK:-0}" != "1" ]; then
    missing=0
    for path in "${required_paths[@]}"; do
        if [ ! -e "${path}" ]; then
            echo "Missing required path: ${path}" >&2
            missing=1
        fi
    done
    if [ "${missing}" != "0" ]; then
        exit 1
    fi
fi

if [ "${CHECK_ONLY:-0}" = "1" ]; then
    echo "finetune_demo.sh path check passed for PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR}"
    exit 0
fi

torchrun_args=(
    --nproc_per_node "${nproc_per_node}"
    --nnodes "${nnodes}"
    --node_rank "${node_rank}"
    --master_addr "${master_addr}"
    --master_port "${master_port}"
)
train_args=(
    --model_name "${model_name}"
    --run_name "${run_name}"
    --seed "${seed:-42}"
    --base_dir "${base_dir}"
    --dtype "${dtype:-bf16}"
    --project "${project:-protenix}"
    --use_wandb "${use_wandb:-false}"
    --diffusion_batch_size "${diffusion_batch_size:-48}"
    --diffusion_chunk_size "${diffusion_chunk_size:-${DIFFUSION_CHUNK_SIZE:-4}}"
    --eval_interval "${eval_interval:-400}"
    --eval_at_end "${eval_at_end:-${EVAL_AT_END:-true}}"
    --eval_only "${eval_only:-${EVAL_ONLY:-false}}"
    --eval_ema_only "${eval_ema_only:-${EVAL_EMA_ONLY:-false}}"
    --eval_worst_sample_count "${eval_worst_sample_count:-${EVAL_WORST_SAMPLE_COUNT:-20}}"
    --log_interval "${log_interval:-50}"
    --checkpoint_interval "${checkpoint_interval:-1000}"
    --empty_cache_interval "${empty_cache_interval:-${EMPTY_CACHE_INTERVAL:-0}}"
    --ema_decay "${ema_decay:-0.999}"
    --ema_shadow_device "${ema_shadow_device:-${EMA_SHADOW_DEVICE:-model}}"
    --distributed_strategy "${distributed_strategy:-${DISTRIBUTED_STRATEGY:-ddp}}"
    --optimizer_sharding "${optimizer_sharding:-${OPTIMIZER_SHARDING:-none}}"
    --fsdp.mesh "${fsdp_mesh:-${FSDP_MESH:-auto}}"
    --fsdp.param_dtype "${fsdp_param_dtype:-${FSDP_PARAM_DTYPE:-fp32}}"
    --fsdp.reduce_dtype "${fsdp_reduce_dtype:-${FSDP_REDUCE_DTYPE:-fp32}}"
    --fsdp.reshard_after_forward "${fsdp_reshard_after_forward:-${FSDP_RESHARD_AFTER_FORWARD:-true}}"
    --skip_final_checkpoint "${skip_final_checkpoint:-${SKIP_FINAL_CHECKPOINT:-false}}"
    --load_params_only "${load_params_only:-${LOAD_PARAMS_ONLY:-true}}"
    --benchmark_warmup_steps "${benchmark_warmup_steps:-${BENCHMARK_WARMUP_STEPS:--1}}"
    --benchmark_measure_steps "${benchmark_measure_steps:-${BENCHMARK_MEASURE_STEPS:--1}}"
    --train_crop_size "${train_crop_size:-${benchmark_crop_size}}"
    --max_steps "${max_steps:-20000}"
    --warmup_steps "${warmup_steps:-2000}"
    --lr "${lr:-0.00001}"
    --model.N_cycle "${N_cycle:-4}"
    --sample_diffusion.N_step "${N_step:-20}"
    --triangle_attention "${triangle_attention:-cuequivariance}"
    --triangle_multiplicative "${triangle_multiplicative:-cuequivariance}"
    --data.train_sets "${resolved_train_sets}"
    --data.train_sampler.train_sample_weights "${resolved_train_sample_weights}"
    --data.train_sampler.seed "${train_sampler_seed:-${TRAIN_SAMPLER_SEED:--1}}"
    --data.train_sampler.replacement "${train_sampler_replacement:-${TRAIN_SAMPLER_REPLACEMENT:-true}}"
    --data.test_sets "${resolved_test_sets}"
    --data.num_dl_workers "${num_dl_workers:-${NUM_DL_WORKERS:-16}}"
    --data.epoch_size "${epoch_size:-${EPOCH_SIZE:-10000}}"
    --data.posebusters_0925.base_info.max_n_token "${posebusters_max_n_token:-512}"
    --data.template.kalign_binary_path "${kalign_binary_path}"
    --data.msa.min_size.train "${msa_min_size_train:-${MSA_MIN_SIZE_TRAIN:-1}}"
    --data.msa.max_size.train "${msa_max_size_train:-16384}"
    --data.msa.sample_cutoff.train "${msa_sample_cutoff_train:-${MSA_SAMPLE_CUTOFF_TRAIN:-2048}}"
    --data.msa.pair_msa_as_unpaired_only "${pair_msa_as_unpaired_only:-${PAIR_MSA_AS_UNPAIRED_ONLY:-false}}"
    --data.msa.shuffle_pair_msa_when_unpaired_only "${shuffle_pair_msa_when_unpaired_only:-${SHUFFLE_PAIR_MSA_WHEN_UNPAIRED_ONLY:-false}}"
    --iters_to_accumulate "${iters_to_accumulate:-1}"
)
if [ -n "${pairformer_blocks_per_ckpt:-${PAIRFORMER_BLOCKS_PER_CKPT:-}}" ]; then
    train_args+=(
        --model.pairformer.blocks_per_ckpt
        "${pairformer_blocks_per_ckpt:-${PAIRFORMER_BLOCKS_PER_CKPT}}"
    )
fi
resolved_msa_chunk_size="${msa_chunk_size:-${MSA_CHUNK_SIZE:-}}"
if [ -n "${resolved_msa_chunk_size}" ]; then
    if [[ ! "${resolved_msa_chunk_size}" =~ ^[1-9][0-9]*$ ]]; then
        echo "MSA_CHUNK_SIZE/msa_chunk_size must be a positive integer when set" >&2
        exit 2
    fi
    train_args+=(
        --model.msa_module.msa_chunk_size
        "${resolved_msa_chunk_size}"
    )
fi
resolved_pairformer_cpu_offload_every_n="${pairformer_checkpoint_cpu_offload_every_n:-${PAIRFORMER_CHECKPOINT_CPU_OFFLOAD_EVERY_N:-}}"
if [ -n "${resolved_pairformer_cpu_offload_every_n}" ]; then
    if [[ ! "${resolved_pairformer_cpu_offload_every_n}" =~ ^[1-9][0-9]*$ ]]; then
        echo "PAIRFORMER_CHECKPOINT_CPU_OFFLOAD_EVERY_N must be a positive integer when set" >&2
        exit 2
    fi
    train_args+=(
        --model.pairformer.checkpoint_cpu_offload_every_n
        "${resolved_pairformer_cpu_offload_every_n}"
    )
fi
resolved_msa_cpu_offload_every_n="${msa_checkpoint_cpu_offload_every_n:-${MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N:-}}"
if [ -n "${resolved_msa_cpu_offload_every_n}" ]; then
    if [[ ! "${resolved_msa_cpu_offload_every_n}" =~ ^[1-9][0-9]*$ ]]; then
        echo "MSA_CHECKPOINT_CPU_OFFLOAD_EVERY_N must be a positive integer when set" >&2
        exit 2
    fi
    train_args+=(
        --model.msa_module.checkpoint_cpu_offload_every_n
        "${resolved_msa_cpu_offload_every_n}"
    )
fi
if [ -n "${model_c_z:-${MODEL_C_Z:-}}" ]; then
    resolved_model_c_z="${model_c_z:-${MODEL_C_Z}}"
    train_args+=(
        --c_z "${resolved_model_c_z}"
        --model.relative_position_encoding.c_z "${resolved_model_c_z}"
        --model.template_embedder.c_z "${resolved_model_c_z}"
        --model.msa_module.c_z "${resolved_model_c_z}"
        --model.constraint_embedder.c_constraint_z "${resolved_model_c_z}"
        --model.pairformer.c_z "${resolved_model_c_z}"
        --model.diffusion_module.c_z "${resolved_model_c_z}"
        --model.confidence_head.c_z "${resolved_model_c_z}"
        --model.distogram_head.c_z "${resolved_model_c_z}"
    )
fi
if [ -n "${model_c_m:-${MODEL_C_M:-}}" ]; then
    train_args+=(
        --model.msa_module.c_m "${model_c_m:-${MODEL_C_M}}"
    )
fi
if [ -n "${pair_transition_chunk_size:-${PAIR_TRANSITION_CHUNK_SIZE:-}}" ]; then
    resolved_transition_chunk_size="${pair_transition_chunk_size:-${PAIR_TRANSITION_CHUNK_SIZE}}"
    train_args+=(
        --model.pairformer.train_pair_transition_chunk_size
        "${resolved_transition_chunk_size}"
        --model.diffusion_module.train_pair_transition_chunk_size
        "${resolved_transition_chunk_size}"
    )
fi
if [ -n "${pair_transition_checkpoint_chunks:-${PAIR_TRANSITION_CHECKPOINT_CHUNKS:-}}" ]; then
    resolved_transition_checkpoint="${pair_transition_checkpoint_chunks:-${PAIR_TRANSITION_CHECKPOINT_CHUNKS}}"
    train_args+=(
        --model.pairformer.train_pair_transition_checkpoint_chunks
        "${resolved_transition_checkpoint}"
        --model.diffusion_module.train_pair_transition_checkpoint_chunks
        "${resolved_transition_checkpoint}"
    )
fi
if [ "${benchmark_exact_crop}" = "true" ]; then
    crop_prefix="data.${benchmark_dataset_name}.cropping_configs"
    resolved_benchmark_max_nuc_tokens="${benchmark_max_nuc_tokens:-${BENCHMARK_MAX_NUC_TOKENS:--1}}"
    train_args+=(
        --data.${benchmark_dataset_name}.base_info.indices_fpath
        "${benchmark_indices_fpath}"
        --data.${benchmark_dataset_name}.base_info.min_n_token
        "${BENCHMARK_MIN_NUM_TOKENS:-${benchmark_min_num_tokens}}"
        --data.${benchmark_dataset_name}.base_info.random_sample_if_failed
        "${benchmark_random_sample_if_failed:-${BENCHMARK_RANDOM_SAMPLE_IF_FAILED:-true}}"
        --${crop_prefix}.method_weights
        "${BENCHMARK_CROP_METHOD_WEIGHTS:-0,1,0}"
        --${crop_prefix}.max_nuc_tokens
        "${resolved_benchmark_max_nuc_tokens}"
        --${crop_prefix}.contiguous_crop_complete_lig false
        --${crop_prefix}.spatial_crop_complete_lig false
        --${crop_prefix}.drop_last false
        --${crop_prefix}.remove_metal false
    )
fi
if [ -n "${load_checkpoint_path}" ]; then
    train_args+=(--load_checkpoint_path "${load_checkpoint_path}")
fi
if [ -n "${load_ema_checkpoint_path}" ]; then
    train_args+=(--load_ema_checkpoint_path "${load_ema_checkpoint_path}")
fi
if [ "${benchmark_exact_crop}" = "true" ]; then
    train_args+=(
        --data.${benchmark_dataset_name}.base_info.mmcif_dir
        "${weighted_mmcif_dir}"
        --data.${benchmark_dataset_name}.base_info.bioassembly_dict_dir
        "${weighted_bioassembly_dir}"
    )
else
    if train_set_is_selected "${weighted_train_set}"; then
        train_args+=(
            --data.${weighted_train_set}.base_info.mmcif_dir "${weighted_mmcif_dir}"
            --data.${weighted_train_set}.base_info.bioassembly_dict_dir "${weighted_bioassembly_dir}"
            --data.${weighted_train_set}.base_info.indices_fpath "${weighted_indices_fpath}"
        )
    fi
    if train_set_is_selected "sabdab-train"; then
        train_args+=(
            --data.sabdab-train.base_info.mmcif_dir "${sabdab_train_mmcif_dir}"
            --data.sabdab-train.base_info.bioassembly_dict_dir "${sabdab_train_bioassembly_dir}"
            --data.sabdab-train.base_info.indices_fpath "${sabdab_train_indices_fpath}"
            --data.sabdab-train.msa.prot_seq_or_filename_to_msadir_jsons "${sabdab_train_seq_index_json}"
            --data.sabdab-train.msa.prot_msadir_raw_paths "${sabdab_msa_template_dir}"
            --data.sabdab-train.template.prot_seq_or_filename_to_templatedir_jsons "${sabdab_train_seq_index_json}"
            --data.sabdab-train.template.prot_template_raw_paths "${sabdab_msa_template_dir}"
        )
    fi
    if train_set_is_selected "distill-sabdab"; then
        train_args+=(
            --data.distill-sabdab.base_info.mmcif_dir "${distill_sabdab_mmcif_dir}"
            --data.distill-sabdab.base_info.bioassembly_dict_dir "${distill_sabdab_bioassembly_dir}"
            --data.distill-sabdab.base_info.indices_fpath "${distill_sabdab_indices_fpath}"
            --data.distill-sabdab.msa.prot_seq_or_filename_to_msadir_jsons "${distill_sabdab_seq_index_json}"
            --data.distill-sabdab.msa.prot_msadir_raw_paths "${sabdab_msa_template_dir}"
            --data.distill-sabdab.template.prot_seq_or_filename_to_templatedir_jsons "${distill_sabdab_seq_index_json}"
            --data.distill-sabdab.template.prot_template_raw_paths "${sabdab_msa_template_dir}"
        )
    fi
    if test_set_is_selected "posebusters_0925"; then
        train_args+=(
            --data.posebusters_0925.base_info.mmcif_dir "${posebusters_mmcif_dir}"
            --data.posebusters_0925.base_info.bioassembly_dict_dir "${posebusters_bioassembly_dir}"
            --data.posebusters_0925.base_info.indices_fpath "${posebusters_indices_fpath}"
        )
    fi
fi
cmd=(torchrun "${torchrun_args[@]}" ./runner/train.py "${train_args[@]}" "$@")

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf "%q " "${cmd[@]}"
    echo
    exit 0
fi

"${cmd[@]}"
