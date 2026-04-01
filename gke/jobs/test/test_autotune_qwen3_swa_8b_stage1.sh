#!/usr/bin/env bash
# Autotune: all + sa_block for Qwen3-SWA-8B (muon, custom unroll=False)
set -euo pipefail

bash gke/setup/preflight.sh

python -m megatext.autotune.search \
    gke/jobs/train/pretrain_qwen3_swa_8b_stage1.sh \
    --scope all \
    --include-sa-block \
    --max-batch-size 8 \
    --num-profile-steps 5
