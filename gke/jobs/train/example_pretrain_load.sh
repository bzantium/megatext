#!/usr/bin/env bash
# Example: Pretrain Qwen3-1.7B on v5e-256
set -euo pipefail

bash gke/setup/preflight.sh
MOUNT_PATH=/mnt/bucket
bash gke/setup/setup_gcsfuse.sh BUCKET=lmt-tpu-datasets MOUNT_PATH=${MOUNT_PATH}

python -m megatext.trainers.pretrain \
  model=qwen3 \
  dataset_type=mmap \
  steps=10 \
  max_target_length=4096 \
  per_device_batch_size=4 \
  learning_rate=3e-4 \
  scan_layers=true \
  remat_policy=full \
  ici_fsdp_parallelism=-1 \
  enable_checkpointing=false \
  log_period=1 \
  run_name=pretrain-qwen3-1.7b-load-ckpt \
  base_num_decoder_layers=28 \
  base_emb_dim=2048 \
  base_mlp_dim=6144 \
  base_num_query_heads=16 \
  base_num_kv_heads=8 \
  head_dim=128 \
  logits_via_embedding=true \
  load_parameters_path=${MOUNT_PATH}/users/ryan/models/maxtext/Qwen3-1.7B-Base/0/items \
  base_output_directory=${MOUNT_PATH}/experiments/ryan
