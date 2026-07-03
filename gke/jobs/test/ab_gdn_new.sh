#!/bin/bash
# Qwen3-Next 8B memory experiment: XLA scheduler flags, TP=1 / FSDP-only.
# Tries latency-hiding-scheduler off to reduce compile-time HLO temporaries.
set -euo pipefail

bash gke/setup/preflight.sh

# xla_tpu_* flags are libtpu flags: append to the LIBTPU_INIT_ARGS exported by submit.py

python -m megatext.trainers.pretrain \
  model=qwen3-next \
  base_emb_dim=4096 \
  base_mlp_dim=5120 \
  base_moe_mlp_dim=5120 \
  base_num_query_heads=32 \
  base_num_kv_heads=8 \
  base_num_decoder_layers=36 \
  head_dim=128 \
  dataset_type=synthetic \
  steps=500 \
  max_target_length=4096 \
  per_device_batch_size=1 \
  learning_rate=3e-4 \
  scan_layers=True \
  remat_policy=full \
  ici_fsdp_parallelism=-1 \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  enable_checkpointing=False \
  log_period=1 \
  opt_type=adamw \
  adam_weight_decay=0.1 \
  gdn_chunk_size=128 \
  run_name=ab-gdn-new \
  base_output_directory=gs://lmt-tpu-datasets/experiments/ryan
