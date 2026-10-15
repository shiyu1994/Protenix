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

# Fail-closed CUDA environment contract for the crop384/db48 H100 training
# launchers. This file is intended to be sourced, not executed.

_protenix_cuda126_fail() {
    echo "CUDA 12.6 runtime contract failed: $*" >&2
    return 2
}

_protenix_cuda126_root="/mnt/project_abd/klp/runtime/protenix_cuda126_toolkit"
_protenix_conda_activate="/mnt/project_abd/klp/miniconda3/bin/activate"
_protenix_conda_env="protenix"
_protenix_python_root="/mnt/project_abd/klp/miniconda3/envs/protenix/lib/python3.11/site-packages"
_protenix_nvcc="${_protenix_cuda126_root}/bin/nvcc"
_protenix_cc="${_protenix_cuda126_root}/bin/x86_64-conda-linux-gnu-gcc"
_protenix_cxx="${_protenix_cuda126_root}/bin/x86_64-conda-linux-gnu-g++"
_protenix_cudart="${_protenix_cuda126_root}/lib/libcudart.so.12"

if [ -n "${CONDA_ACTIVATE:-}" ] \
    && [ "$(readlink -m "${CONDA_ACTIVATE}")" != "${_protenix_conda_activate}" ]; then
    _protenix_cuda126_fail \
        "CONDA_ACTIVATE must be ${_protenix_conda_activate}; got ${CONDA_ACTIVATE}"
fi
if [ -n "${CONDA_ENV:-}" ] && [ "${CONDA_ENV}" != "${_protenix_conda_env}" ]; then
    _protenix_cuda126_fail \
        "CONDA_ENV must be ${_protenix_conda_env}; got ${CONDA_ENV}"
fi

for _protenix_required_path in \
    "${_protenix_conda_activate}" \
    "${_protenix_python_root}/torch" \
    "${_protenix_nvcc}" \
    "${_protenix_cc}" \
    "${_protenix_cxx}" \
    "${_protenix_cudart}" \
    "${_protenix_cuda126_root}/include" \
    "${_protenix_cuda126_root}/targets/x86_64-linux/include"; do
    if [ ! -e "${_protenix_required_path}" ]; then
        _protenix_cuda126_fail "missing required path ${_protenix_required_path}"
    fi
done
for _protenix_required_executable in \
    "${_protenix_nvcc}" "${_protenix_cc}" "${_protenix_cxx}"; do
    if [ ! -x "${_protenix_required_executable}" ]; then
        _protenix_cuda126_fail \
            "required executable is not executable: ${_protenix_required_executable}"
    fi
done

_protenix_nvcc_sha256="$(
    sha256sum "${_protenix_nvcc}" | awk '{print $1}'
)"
_protenix_cc_sha256="$(
    sha256sum "${_protenix_cc}" | awk '{print $1}'
)"
_protenix_cxx_sha256="$(
    sha256sum "${_protenix_cxx}" | awk '{print $1}'
)"
_protenix_cudart_sha256="$(
    sha256sum "$(readlink -f "${_protenix_cudart}")" | awk '{print $1}'
)"

if [ "${_protenix_nvcc_sha256}" \
    != "29466bd84e6b7086557eda96d023a81cac0dfb23010be8b4011456e73730d661" ]; then
    _protenix_cuda126_fail "unexpected nvcc SHA256 ${_protenix_nvcc_sha256}"
fi
if [ "${_protenix_cc_sha256}" \
    != "92edc7d479c63429559dc49bc080aa919bc91a484fba5fc815fa793b07c8098d" ]; then
    _protenix_cuda126_fail "unexpected GCC SHA256 ${_protenix_cc_sha256}"
fi
if [ "${_protenix_cxx_sha256}" \
    != "ce445b39aa6d688102169aac8c8cc60bb5997c8cdd049a5ba8beabb01d023251" ]; then
    _protenix_cuda126_fail "unexpected G++ SHA256 ${_protenix_cxx_sha256}"
fi
if [ "${_protenix_cudart_sha256}" \
    != "8a8579b35d2c4a15d0eabd9a8501cc03134b5ba2be300f7e00aa69795d9b1981" ]; then
    _protenix_cuda126_fail "unexpected libcudart SHA256 ${_protenix_cudart_sha256}"
fi

_protenix_nvcc_version="$("${_protenix_nvcc}" --version 2>&1)"
if ! grep -Fq "release 12.6, V12.6.85" <<< "${_protenix_nvcc_version}"; then
    _protenix_cuda126_fail "nvcc is not the pinned CUDA 12.6.85 compiler"
fi
if ! "${_protenix_cc}" --version 2>&1 | head -n 1 | grep -Fq "12.4.0"; then
    _protenix_cuda126_fail "host GCC is not the pinned 12.4.0 compiler"
fi
if ! "${_protenix_cxx}" --version 2>&1 | head -n 1 | grep -Fq "12.4.0"; then
    _protenix_cuda126_fail "host G++ is not the pinned 12.4.0 compiler"
fi

_protenix_nvidia_includes="$(
    find "${_protenix_python_root}/nvidia" \
        -mindepth 2 -maxdepth 2 -type d -name include \
        -print 2>/dev/null | LC_ALL=C sort | paste -sd: -
)"
if [ -z "${_protenix_nvidia_includes}" ]; then
    _protenix_cuda126_fail \
        "no NVIDIA component include directories found under ${_protenix_python_root}"
fi

# Preserve the caller's original CPATH once, then construct the pinned prefix
# deterministically if this helper is sourced more than once.
if [ -z "${PROTENIX_CUDA126_ORIGINAL_CPATH+x}" ]; then
    export PROTENIX_CUDA126_ORIGINAL_CPATH="${CPATH:-}"
fi
export CUDA_HOME="${_protenix_cuda126_root}"
export CUDACXX="${_protenix_nvcc}"
export PATH="${_protenix_cuda126_root}/bin:${PATH}"
export PROTENIX_CUDA_HOST_CC="${_protenix_cc}"
export PROTENIX_CUDA_HOST_CXX="${_protenix_cxx}"
export CC="${_protenix_cc}"
export CXX="${_protenix_cxx}"
export CPATH="${_protenix_cuda126_root}/include:${_protenix_cuda126_root}/targets/x86_64-linux/include:${_protenix_nvidia_includes}${PROTENIX_CUDA126_ORIGINAL_CPATH:+:${PROTENIX_CUDA126_ORIGINAL_CPATH}}"
export CONDA_ACTIVATE="${_protenix_conda_activate}"
export CONDA_ENV="${_protenix_conda_env}"
export EXPECTED_TORCH_CUDA="12.6"
export PROTENIX_EXPECTED_TORCH_VERSION="2.7.1+cu126"
export PROTENIX_CUDA_ARCH_LIST="9.0"
export PROTENIX_CUDA126_HOME="${_protenix_cuda126_root}"
export PROTENIX_CUDA_NVCC_SHA256="${_protenix_nvcc_sha256}"
export PROTENIX_CUDA_HOST_CC_SHA256="${_protenix_cc_sha256}"
export PROTENIX_CUDA_HOST_CXX_SHA256="${_protenix_cxx_sha256}"
export PROTENIX_CUDART_SHA256="${_protenix_cudart_sha256}"

if [ "$(readlink -f "${CUDA_HOME}")" != "${_protenix_cuda126_root}" ]; then
    _protenix_cuda126_fail "CUDA_HOME did not resolve to the pinned toolkit"
fi
if [ "$(readlink -f "$(command -v nvcc)")" != "${_protenix_nvcc}" ]; then
    _protenix_cuda126_fail \
        "PATH resolves nvcc outside the pinned toolkit: $(command -v nvcc)"
fi

_protenix_runtime_helper_dir="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
)"
_protenix_repo_dir="$(
    cd "${_protenix_runtime_helper_dir}/../.." && pwd
)"
_protenix_layer_norm_source_sha256="$(
    {
        printf '%s\0' "${_protenix_runtime_helper_dir}/cuda126_env.sh"
        find "${_protenix_repo_dir}/protenix/model/layer_norm" -type f \
            \( -name '*.py' -o -name '*.cpp' -o -name '*.cc' \
               -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' \) \
            ! -path '*/__pycache__/*' -print0
    } | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'
)"
if [ -z "${_protenix_layer_norm_source_sha256}" ]; then
    _protenix_cuda126_fail "could not fingerprint the LayerNorm sources"
fi

_protenix_toolchain_fingerprint="$(
    {
        printf '%s\n' \
            "schema=cuda126-h100-v1" \
            "cuda_home=${_protenix_cuda126_root}" \
            "nvcc_sha256=${_protenix_nvcc_sha256}" \
            "cc_sha256=${_protenix_cc_sha256}" \
            "cxx_sha256=${_protenix_cxx_sha256}" \
            "cudart_sha256=${_protenix_cudart_sha256}" \
            "torch=2.7.1+cu126" \
            "torch_cuda=12.6" \
            "python=cp311" \
            "cuda_arch=9.0"
    } | sha256sum | awk '{print $1}'
)"
_protenix_layer_norm_cache_fingerprint="$(
    printf '%s\n%s\n' \
        "${_protenix_toolchain_fingerprint}" \
        "${_protenix_layer_norm_source_sha256}" \
        | sha256sum | awk '{print $1}'
)"
export PROTENIX_CUDA_TOOLCHAIN_FINGERPRINT="${_protenix_toolchain_fingerprint}"
export PROTENIX_LAYER_NORM_SOURCE_SHA256="${_protenix_layer_norm_source_sha256}"
export PROTENIX_LAYER_NORM_CACHE_FINGERPRINT="${_protenix_layer_norm_cache_fingerprint}"
export PROTENIX_LAYER_NORM_PROVENANCE_SCHEMA="1"
export PROTENIX_CUDA126_ENV_CONFIGURED="1"

unset _protenix_required_path _protenix_required_executable
unset _protenix_nvcc_version _protenix_nvidia_includes
unset _protenix_runtime_helper_dir _protenix_repo_dir
unset _protenix_layer_norm_source_sha256
unset _protenix_toolchain_fingerprint _protenix_layer_norm_cache_fingerprint
unset _protenix_nvcc_sha256 _protenix_cc_sha256 _protenix_cxx_sha256
unset _protenix_cudart_sha256
unset _protenix_cuda126_root _protenix_conda_activate _protenix_conda_env
unset _protenix_python_root _protenix_nvcc _protenix_cc _protenix_cxx
unset _protenix_cudart
unset -f _protenix_cuda126_fail
