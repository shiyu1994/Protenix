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

# Locked, fully built cache used by the formal disordered-v4 stage. Keep this
# contract in one sourced file so params-only stage initialization and strict
# full-state resume cannot drift.
readonly PROTENIX_LOCKED_SEQUENCE_TEMPLATE_CACHE_ROOT="/mnt/project_abd/klp/databases/protenix_sequence_template_cache_top60_uniform20_id90_v1"
readonly PROTENIX_LOCKED_SEQUENCE_TEMPLATE_CACHE_MODE="required"
readonly PROTENIX_LOCKED_SEQUENCE_TEMPLATE_CACHE_ERROR_POLICY="chain_dummy"
readonly PROTENIX_LOCKED_SEQUENCE_TEMPLATE_CACHE_FILENAME="template_features.pkl.gz"

lock_and_validate_sequence_template_cache() {
    local variable_name
    for variable_name in \
        PROTENIX_SEQUENCE_TEMPLATE_CACHE_ROOT \
        PROTENIX_SEQUENCE_TEMPLATE_CACHE_MODE \
        PROTENIX_SEQUENCE_TEMPLATE_CACHE_ERROR_POLICY \
        PROTENIX_SEQUENCE_TEMPLATE_CACHE_FILENAME; do
        if [ -n "${!variable_name-}" ]; then
            echo "Locked template-cache contract rejects inherited ${variable_name}." >&2
            return 2
        fi
    done

    local cache_root="${PROTENIX_LOCKED_SEQUENCE_TEMPLATE_CACHE_ROOT}"
    export PROTENIX_SEQUENCE_TEMPLATE_CACHE_ROOT="${cache_root}"
    export PROTENIX_SEQUENCE_TEMPLATE_CACHE_MODE="${PROTENIX_LOCKED_SEQUENCE_TEMPLATE_CACHE_MODE}"
    export PROTENIX_SEQUENCE_TEMPLATE_CACHE_ERROR_POLICY="${PROTENIX_LOCKED_SEQUENCE_TEMPLATE_CACHE_ERROR_POLICY}"
    export PROTENIX_SEQUENCE_TEMPLATE_CACHE_FILENAME="${PROTENIX_LOCKED_SEQUENCE_TEMPLATE_CACHE_FILENAME}"

    # Dry-run launcher tests and path-check-only audits are portable to hosts
    # that do not mount the production database. A real launch has neither flag
    # and therefore always performs the strict completion audit below.
    if [ "${DRY_RUN:-0}" = "1" ] || [ "${SKIP_PATH_CHECK:-0}" = "1" ]; then
        return 0
    fi

    if [ ! -d "${cache_root}" ]; then
        echo "Locked sequence-template cache root does not exist: ${cache_root}" >&2
        return 2
    fi

    local python_bin="/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python"
    if [ ! -x "${python_bin}" ]; then
        echo "Template-cache preflight Python is not executable: ${python_bin}" >&2
        return 2
    fi
    if ! "${python_bin}" - "${cache_root}" <<'PY'
import json
import sys
from pathlib import Path

cache_root = Path(sys.argv[1])
expected = (
    (0, 89_101),
    (1, 88_760),
)
policy = "top60_uniform20_identity_le_0.90_v1"
errors = []
for shard_id, expected_total in expected:
    stats_path = (
        cache_root
        / "_manifests"
        / f"stats.shard-{shard_id:05d}-of-00002.json"
    )
    try:
        stats = json.loads(stats_path.read_text())
    except Exception as exc:
        errors.append(f"{stats_path}: unreadable stats ({exc})")
        continue
    required_values = {
        "schema_version": 1,
        "num_shards": 2,
        "shard_id": shard_id,
        "policy_version": policy,
        "total_tasks": expected_total,
        "completed_tasks": expected_total,
        "remaining_tasks": 0,
    }
    for key, expected_value in required_values.items():
        if stats.get(key) != expected_value:
            errors.append(
                f"{stats_path}: {key}={stats.get(key)!r}, "
                f"expected {expected_value!r}"
            )
    status_counts = stats.get("status_counts")
    if not isinstance(status_counts, dict):
        errors.append(f"{stats_path}: status_counts is not an object")
        continue
    failed = int(status_counts.get("failed", 0))
    completed = int(status_counts.get("complete", 0)) + int(
        status_counts.get("complete_empty", 0)
    )
    if failed != 0:
        errors.append(f"{stats_path}: status_counts.failed={failed}, expected 0")
    if completed != expected_total:
        errors.append(
            f"{stats_path}: complete+complete_empty={completed}, "
            f"expected {expected_total}"
        )

if errors:
    raise SystemExit("\n".join(errors))
PY
    then
        echo "Locked sequence-template cache completion preflight failed." >&2
        return 2
    fi
}
