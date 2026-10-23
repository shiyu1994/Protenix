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

usage() {
    cat <<'HELP'
Usage: bash preprocess_peptide.sh [--check] [--dry-run]

Prepare converted peptide CIFs for finetune_peptide.sh using CPUs only.
Successful, unchanged cache entries are reused; no training is launched.

  --check      Verify the existing cache and index without converting CIFs.
  --dry-run    Print the converter command without running it.
  -h, --help   Show this help.

Environment variables (shared with finetune_peptide.sh):
  DATA_DIR             Source CIFs (default: the supplied peptide directory)
  PREPARED_DIR         Output directory (default: ../peptide_finetune_data)
  PYTHON_BIN           Interpreter (default: the protenix Conda environment)
  PROTENIX_ROOT_DIR    Shared CCD resource root
  ORIGINAL_MMCIF_DIR   Original PDB CIFs (default: PROTENIX_ROOT_DIR/mmcif)
  PREP_REUSE_DIR       Optional existing cache to repair into PREPARED_DIR
  PREP_WORKERS         Parallel preprocessing workers (default: 4)
  SEED                 Reference-conformer seed (default: 42)
  CIF_IDS              Optional comma-separated CIF stems; unset processes all CIFs
  SKIP_PREP_ERRORS     Set 1 to explicitly exclude failed CIFs (default: 0)

Outputs under PREPARED_DIR:
  bioassembly/*.pkl.gz  Structures consumed by the training loader
  indices.csv          Peptide-interface training index
  records/*.json       Per-CIF cache signatures and chemistry audits
  preparation.json     Counts and readiness status
  failures.json        Rejected CIFs and reasons

Then run: bash finetune_peptide.sh train
Use the same DATA_DIR, PREPARED_DIR, SEED, CIF_IDS, PROTENIX_ROOT_DIR, and
SKIP_PREP_ERRORS for preprocessing and training. See docs/peptide_finetuning.md.
HELP
}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dry_run="${DRY_RUN:-0}"
check_only=0
for arg in "$@"; do
    case "${arg}" in
        --check) check_only=1 ;;
        --dry-run) dry_run=1 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "${arg}" >&2; usage >&2; exit 2 ;;
    esac
done

data_dir="${DATA_DIR:-/root/shiyu/pdb_peptide_to_lig_20190930_strict_filtered_nan0.7_mmcif_ligand_peptide}"
prepared_dir="${PREPARED_DIR:-${repo_dir}/../peptide_finetune_data}"
python_bin="${PYTHON_BIN:-/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/mnt/project_abd/public_databases/protenix.20260101}"
export PYTHONPATH="${repo_dir}"
export CUDA_VISIBLE_DEVICES=""
export LAYERNORM_TYPE=torch
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

prep=("${python_bin}" "${repo_dir}/scripts/prepare_peptide_training_data.py"
    --input-dir "${data_dir}" --output-dir "${prepared_dir}"
    --original-mmcif-dir "${ORIGINAL_MMCIF_DIR:-${PROTENIX_ROOT_DIR}/mmcif}"
    --workers "${PREP_WORKERS:-4}" --seed "${SEED:-42}")
if [[ -n ${PREP_REUSE_DIR:-} ]]; then prep+=(--reuse-prepared-dir "${PREP_REUSE_DIR}"); fi
if [[ -n ${CIF_IDS:-} ]]; then prep+=(--ids "${CIF_IDS}"); fi
if [[ ${SKIP_PREP_ERRORS:-0} == 1 ]]; then prep+=(--skip-errors); fi
if [[ ${check_only} == 1 ]]; then prep+=(--check); fi

printf 'Preprocessing command: '; printf '%q ' "${prep[@]}"; printf '\n'
if [[ ${dry_run} == 1 ]]; then exit 0; fi
if [[ ! -x ${python_bin} ]]; then echo "Missing executable PYTHON_BIN: ${python_bin}" >&2; exit 1; fi
python_env_bin="$(dirname "${python_bin}")"
export PATH="${python_env_bin}:${PATH}"
# Biotite/parser imports depend on the installed PyTorch shared libraries.
torch_lib="$("${python_bin}" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="${torch_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${repo_dir}"
"${prep[@]}"
printf 'Prepared cache is ready: %s\n' "${prepared_dir}"
printf 'Start finetuning with the same data settings: bash %q train\n' "${repo_dir}/finetune_peptide.sh"
