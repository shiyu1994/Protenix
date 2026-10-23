  PROTENIX_MSA_DEBUG_LOG_LIMIT=50 \
  PROTENIX_MSA_SAMPLE_DEBUG_LOG_LIMIT=220 \
  NPROC_PER_NODE=1 \
  max_steps=50 \
  log_interval=1 \
  eval_interval=0 \
  eval_at_end=false \
  EVAL_AT_END=false \
  run_name=msa_debug_default_50steps_$(date +%Y%m%d_%H%M%S) \
  bash finetune_submit.sh
