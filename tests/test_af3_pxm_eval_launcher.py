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

"""Static contracts for the locked PXM evaluation launcher."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY_ROOT / "eval_af3_pxm_legacy_ema_checkpoint.sh"


def test_launcher_requires_and_fully_validates_frozen_inputs() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    required_fragments = (
        "set -euo pipefail",
        '"${pxm_input_success}"',
        '"${pxm_template_cache_success}"',
        '"${pxm_msa_lock_manifest}"',
        '"${pxm_msa_lock_success}"',
        "validate_af3_pxm_template_cache.py",
        "validate_af3_pxm_msa_lock.py",
        '--pdb-list "${pxm_full_pdb_list}"',
        '--requested-pdb-list "${pdb_list}"',
        "PROTENIX_EVAL_SEQUENCE_TEMPLATE_CACHE_MODE=required",
        "--data.template.evaluation_sequence_template_cache_mode required",
        (
            '--data.msa.prot_seq_or_filename_to_msadir_jsons '
            '"${official_root}/common/seq_to_pdb_index.json,'
            '${pxm_msa_overlay_mapping}"'
        ),
    )
    for fragment in required_fragments:
        assert fragment in text

    template_validator_position = text.index(
        "validate_af3_pxm_template_cache.py"
    )
    msa_validator_position = text.index("validate_af3_pxm_msa_lock.py")
    evaluator_position = text.index(
        'exec bash "${repo_dir}/finetune_demo.sh"'
    )
    assert template_validator_position < evaluator_position
    assert msa_validator_position < evaluator_position


def test_launcher_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
