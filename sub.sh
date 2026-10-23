  #!/bin/bash
  set -euo pipefail

  cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix

  export NNODES="${WORLD_SIZE:-${NNODES:-1}}"
  export NODE_RANK="${RANK:-${NODE_RANK:-0}}"
  export MASTER_PORT="${MASTER_PORT:-29500}"
  export NPROC_PER_NODE="${NPROC_PER_NODE:-gpu}"

  bash ./finetune_demo.sh
