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
        "${repo_dir}/finetune_submit.sh"
)"

case "${output}" in
    *"--nproc_per_node 4 --nnodes 3 --node_rank 2 --master_addr 10.1.2.3 --master_port 45678"*)
        ;;
    *)
        echo "Expected AIStudio distributed variables to be mapped into torchrun args." >&2
        echo "Actual output:" >&2
        echo "${output}" >&2
        exit 1
        ;;
esac

case "${output}" in
    *"--eval_interval 0 --eval_at_end false"*)
        ;;
    *)
        echo "Expected finetune_submit.sh to disable periodic and final eval by default." >&2
        echo "Actual output:" >&2
        echo "${output}" >&2
        exit 1
        ;;
esac

case "${output}" in
    *"--data.test_sets posebusters_0925"*)
        ;;
    *)
        echo "Expected finetune_submit.sh to keep default test_sets configured for initialization." >&2
        echo "Actual output:" >&2
        echo "${output}" >&2
        exit 1
        ;;
esac

case "${output}" in
    *"--data.msa.pair_msa_as_unpaired_only true"*)
        ;;
    *)
        echo "Expected finetune_submit.sh to enable unpaired-only MSA pairing mode." >&2
        echo "Actual output:" >&2
        echo "${output}" >&2
        exit 1
        ;;
esac

case "${output}" in
    *"--data.msa.shuffle_pair_msa_when_unpaired_only false"*)
        ;;
    *)
        echo "Expected finetune_submit.sh to leave pair-MSA shuffle disabled by default." >&2
        echo "Actual output:" >&2
        echo "${output}" >&2
        exit 1
        ;;
esac

case "${output}" in
    *"--data.msa.max_size.train 16384"*)
        ;;
    *)
        echo "Expected finetune_submit.sh to keep train MSA feature max size at 16384." >&2
        echo "Actual output:" >&2
        echo "${output}" >&2
        exit 1
        ;;
esac

case "${output}" in
    *"--data.msa.sample_cutoff.train 2048"*)
        ;;
    *)
        echo "Expected finetune_submit.sh to cap train MSA sampling at 2048." >&2
        echo "Actual output:" >&2
        echo "${output}" >&2
        exit 1
        ;;
esac

echo "finetune_submit.sh maps distributed variables, disables training-time eval, configures MSA pairing mode, and caps train MSA sampling correctly"
