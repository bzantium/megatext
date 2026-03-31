#!/usr/bin/env bash
# Example: Pretrain Qwen3-8B on v5e-256
set -euo pipefail

bash gke/setup/preflight.sh
MOUNT_PATH=/mnt/bucket
bash gke/setup/setup_gcsfuse.sh BUCKET=lmt-tpu-datasets MOUNT_PATH=${MOUNT_PATH}

python -m megatext.trainers.pretrain \
  model=qwen3-8b \
  dataset_type=synthetic \
  steps=10 \
  max_target_length=4096 \
  per_device_batch_size=4 \
  learning_rate=3e-4 \
  scan_layers=true \
  remat_policy=full \
  ici_fsdp_parallelism=-1 \
  enable_checkpointing=false \
  log_period=1 \
  run_name=pretrain-qwen3-8b \
  base_output_directory=${MOUNT_PATH}/experiments/ryan
