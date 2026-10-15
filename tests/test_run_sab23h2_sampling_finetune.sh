#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

output="$(
    env \
        DRY_RUN=1 \
        SKIP_PATH_CHECK=1 \
        WORLD_SIZE=3 \
        RANK=2 \
        MASTER_ADDR=10.1.2.3 \
        MASTER_PORT=45678 \
        NPROC_PER_NODE=4 \
        EXPERIMENT_NAME=sampling_conservative \
        TRAIN_SAMPLE_WEIGHTS=0.4,0.4,0.2 \
        RUN_TIMESTAMP=DRYRUN \
        "${repo_dir}/run_sab23h2_sampling_finetune.sh"
)"

assert_contains() {
    local needle="$1"
    if [[ "${output}" != *"${needle}"* ]]; then
        echo "Expected output to contain: ${needle}" >&2
        echo "Actual output:" >&2
        echo "${output}" >&2
        exit 1
    fi
}

assert_contains "experiment_name=sampling_conservative"
assert_contains "train_sample_weights=0.4,0.4,0.2"
assert_contains "999_ema_0.999.pt"
assert_contains "--nproc_per_node 4 --nnodes 3 --node_rank 2 --master_addr 10.1.2.3 --master_port 45678"
assert_contains "--checkpoint_interval 250"
assert_contains "--max_steps 2000"
assert_contains "--lr 0.000003"
assert_contains "--eval_interval 0 --eval_at_end false"

echo "run_sab23h2_sampling_finetune.sh maps AIStudio variables and sampling parameters correctly"
