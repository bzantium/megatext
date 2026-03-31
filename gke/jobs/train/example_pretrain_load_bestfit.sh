#!/usr/bin/env bash
# Example: Pretrain Qwen3-1.7B on v5e-256
set -euo pipefail

bash gke/setup/preflight.sh
BUCKET_PREFIX=gs://lmt-tpu-datasets
MOUNT_PATH=/mnt/bucket
bash gke/setup/setup_gcsfuse.sh BUCKET=lmt-tpu-datasets MOUNT_PATH=${MOUNT_PATH}

DATA_ROOT=${MOUNT_PATH}/datasets/kanana-2-corpus/mmap
DATA_PATH="\
10 ${DATA_ROOT}/nemotron-cc-v2-high-synthetic-sampled \
10 ${DATA_ROOT}/opc-annealing-synthetic"

python -m megatext.trainers.pretrain \
  model=qwen3 \
  dataset_type=mmap \
  "dataset_path=${DATA_PATH}" \
  data_cache_dir=${BUCKET_PREFIX}/cache/indices \
  steps=10 \
  max_target_length=4096 \
  per_device_batch_size=4 \
  learning_rate=3e-4 \
  scan_layers=true \
  remat_policy=full \
  ici_fsdp_parallelism=-1 \
  enable_checkpointing=true \
  log_period=1 \
  run_name=pretrain-qwen3-1.7b-bestfit \
  base_num_decoder_layers=28 \
  base_emb_dim=2048 \
  base_mlp_dim=6144 \
  base_num_query_heads=16 \
  base_num_kv_heads=8 \
  head_dim=128 \
  logits_via_embedding=true \
  grain_packing_type=best_fit \
  load_parameters_path=${BUCKET_PREFIX}/users/ryan/models/maxtext/Qwen3-1.7B-Base/0/items \
  base_output_directory=${BUCKET_PREFIX}/experiments/ryan