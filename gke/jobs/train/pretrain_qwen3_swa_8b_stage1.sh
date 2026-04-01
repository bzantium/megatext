#!/usr/bin/env bash
# Pretrain Qwen3-SWA-8B
set -euo pipefail

bash gke/setup/preflight.sh
BUCKET_PREFIX=gs://lmt-tpu-datasets
MOUNT_PATH=/mnt/bucket
bash gke/setup/setup_gcsfuse.sh BUCKET=lmt-tpu-datasets MOUNT_PATH=${MOUNT_PATH}

DATA_DIR=${MOUNT_PATH}/datasets/kanana-2-corpus/4096
DATA_PATH="\
1657.4 $DATA_DIR/nemotron-cc-v2-high \
355.9 $DATA_DIR/finepdfs-edu \
503.3 $DATA_DIR/dq-csgk-ko \
88.6 $DATA_DIR/dq-csgk-ja \
102.7 $DATA_DIR/dq-csgk-zh \
35.2 $DATA_DIR/dq-csgk-vi \
25.2 $DATA_DIR/dq-csgk-th \
32.4 $DATA_DIR/finewiki-en \
1.8 $DATA_DIR/finewiki-ko \
9.4 $DATA_DIR/finewiki-ja \
4.2 $DATA_DIR/finewiki-zh \
1.8 $DATA_DIR/finewiki-vi \
0.9 $DATA_DIR/finewiki-th \
251.7 $DATA_DIR/dq-academic \
503.3 $DATA_DIR/dq-mathnl-enko-v2 \
377.5 $DATA_DIR/dq-codenl-enko \
1082.1 $DATA_DIR/dq-codescr-entire-v2"

python -m megatext.trainers.pretrain \
  model=qwen3-swa \
  dataset_type=fixed_arecord \
  "dataset_path=$DATA_PATH" \
  base_output_directory=gs://lmt-tpu-datasets/experiments/ryan \
  run_name=pretrain-qwen3-swa-8b-stage1 \
  steps=240000 \
  per_device_batch_size=2 \
  global_batch_size=7680 \
  learning_rate=1.23e-4 \
  warmup_steps=2400 \
  final_learning_rate=1.23e-5 \
  remat_policy=full \
  scan_layers=true \
  sa_use_fused_bwd_kernel=true \
  sa_block_q=1024 \
  sa_block_kv=1024 \
  sa_block_kv_compute=1024 \
  sa_block_q_dkv=1024 \
  sa_block_kv_dkv=1024 \
  sa_block_kv_dkv_compute=1024 \
  sa_block_q_dq=1024 \
  sa_block_kv_dq=1024 \
  opt_type=muon \
  muon_weight_decay=0.1 \
  muon_consistent_rms=0.2 \
  adam_b1=0.9 \
  adam_b2=0.95 \
  gradient_clipping_threshold=1.0 \
  z_loss_multiplier=5e-6 \
  max_target_length=4096 \
  attention=flash \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  checkpoint_period=500 \
  log_period=1
