#!/bin/bash
set -euo pipefail
bash gke/setup/preflight.sh

python -m megatext.trainers.pretrain \
  model=qwen3-swa \
  dataset_type=synthetic \
  steps=5 \
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
  run_name=probe-swa-bs1 \
  base_output_directory=gs://lmt-tpu-datasets/experiments/ryan
