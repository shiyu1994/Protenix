#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

run_dir="${tmp_dir}/sampling_run"
mkdir -p "${run_dir}/checkpoints"

for step in 249 499 749 999 1249 1499 1749 1999; do
    : > "${run_dir}/checkpoints/${step}_ema_0.999.pt"
done

if ! output="$(
    env \
        DRY_RUN=1 \
        TRAIN_RUN_DIRS="toy=${run_dir}" \
        "${repo_dir}/run_sab23h2_sampling_checkpoint_eval_queue.sh" 2>&1
)"; then
    echo "Expected dry-run evaluation planning to succeed." >&2
    echo "Actual output:" >&2
    echo "${output}" >&2
    exit 1
fi

assert_contains() {
    local needle="$1"
    if [[ "${output}" != *"${needle}"* ]]; then
        echo "Expected output to contain: ${needle}" >&2
        echo "Actual output:" >&2
        echo "${output}" >&2
        exit 1
    fi
}

assert_not_contains() {
    local needle="$1"
    if [[ "${output}" == *"${needle}"* ]]; then
        echo "Expected output not to contain: ${needle}" >&2
        echo "Actual output:" >&2
        echo "${output}" >&2
        exit 1
    fi
}

assert_contains "planned_inference label=toy_249_ema"
assert_contains "planned_inference label=toy_1999_ema"
assert_contains "run_labels=toy_249_ema,toy_499_ema,toy_749_ema,toy_999_ema,toy_1249_ema,toy_1499_ema,toy_1749_ema,toy_1999_ema"
assert_not_contains "skip_missing_checkpoint"
assert_not_contains "step=250"

echo "run_sab23h2_sampling_checkpoint_eval_queue.sh defaults to saved zero-based checkpoint steps"
