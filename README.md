<p align="center">
  <h1 align="center">Megatext</h1>
  <p align="center">Simple, performant and scalable JAX LLM pretraining</p>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#data-pipeline">Data Pipeline</a> &bull;
  <a href="#supported-models">Models</a> &bull;
  <a href="#configuration">Configuration</a> &bull;
  <a href="#gke-deployment">GKE Deployment</a>
</p>

---

**Megatext** is a streamlined pretraining framework for large language models on TPUs.
Forked from [Google's MaxText](https://github.com/AI-Hypercomputer/maxtext), it strips away
the complexity and keeps what matters: a fast data pipeline, clean configuration, and
effortless scaling from a single host to thousands of TPU chips.

### Why Megatext?

| | MaxText | Megatext |
|---|---------|----------|
| **Config** | 65 hardcoded model names (`Literal["qwen3-8b", ...]`) | Free-form `model=qwen3` + CLI overrides |
| **Model configs** | 68 size-specific YAMLs | 9 architecture templates |
| **Data indexing** | Python loops (minutes for 19M docs) | C++ segment tree (7 seconds) |
| **Packing** | greedy only | greedy + best-fit bin packing |
| **Cache I/O** | Local filesystem only | GCS-native via `gcsfs` |
| **Codebase** | ~1M lines, inference server, benchmarks, post-training | ~20K lines, training only |

---

## Quickstart

### Install

```bash
pip install -e .          # CPU (includes C++ packing extension)
pip install -e ".[tpu]"   # TPU
```

### Train (synthetic data)

```bash
python -m megatext.trainers.pretrain \
  model=qwen3 \
  dataset_type=synthetic \
  steps=100 \
  per_device_batch_size=4 \
  scan_layers=true \
  remat_policy=full \
  ici_fsdp_parallelism=-1 \
  enable_checkpointing=false
```

### Train (real data on GKE)

```bash
python gke/submit.py pretrain \
  --infra gke/infra/v5e.yaml \
  --build \
  gke/jobs/train/pretrain_qwen3_swa_8b_stage1.yaml
```

---

## Architecture

```
src/megatext/
  trainers/pretrain.py       # Training loop entry point
  configs/
    base.yaml                # Default config (all fields documented)
    models/                  # Architecture templates (qwen3, llama3, ...)
    types.py                 # Pydantic config model with validation
    pyconfig.py              # YAML + CLI + env merge logic
  data/
    _helpers.cpp             # C++ packing (pybind11, segment tree)
    packing.py               # greedy + best-fit sample index builders
    data_sources.py          # Document-level data source with cached indexing
    data_processing.py       # Grain pipeline (shard -> repeat -> transform -> batch)
  models/                    # Decoder implementations (qwen3, llama, deepseek, ...)
  layers/                    # Attention, MLP, MoE, normalization, quantization
  optimizers/                # AdamW, Adam-PAX, SGD, Muon
  schedulers/                # Constant, Cosine, WSD learning rate schedules
  conversion/                # Bidirectional HF <-> Megatext checkpoint conversion
  autotune/                  # Automated batch size / remat / SA block search
```

---

## Data Pipeline

Megatext supports four dataset types:

| Type | Description |
|------|-------------|
| `mmap` | Memory-mapped tokenized files (Megatron-style) |
| `arecord` | Variable-length arecord format |
| `fixed_arecord` | Fixed-length pre-packed arecord (fastest) |
| `synthetic` | Generated data for debugging/profiling |

### Multi-source blending

Blend multiple datasets with proportional sampling:

```yaml
dataset_path: >-
  10 /data/nemotron-cc
  5 /data/opc-annealing
  3 /data/code-corpus
```

Weights are normalized to percentages and logged at startup:

```
Data blend: weight=10 (55.56%) path=/data/nemotron-cc
Data blend: weight=5 (27.78%) path=/data/opc-annealing
Data blend: weight=3 (16.67%) path=/data/code-corpus
```

### C++ Packing Helpers

Index building uses C++ with pybind11:

| Algorithm | 19M docs | Description |
|-----------|----------|-------------|
| `greedy` | 1.6s | Cross-document concatenation (Megatron-style) |
| `best_fit` | 7.4s | Segment-tree bin packing, O(N log seq_len) |

### GCS-Native Caching

Data indices are cached and reused across runs, supporting both local and GCS paths:

```bash
python -m megatext.trainers.pretrain data_cache_dir=gs://my-bucket/cache ...
```

### Pipeline Flow

```
mmap/arecord files
  -> DocumentDataSource (cached index: document -> sample -> shuffle)
  -> Grain MapDataset (shard per process -> infinite repeat -> transform -> batch)
  -> MultiHostDataLoadIterator (form global JAX arrays across hosts)
  -> DataLoader (device_put with mesh sharding)
```

---

## Supported Models

Select an architecture template and override any parameter via CLI:

```bash
python -m megatext.trainers.pretrain \
  model=qwen3 \
  base_emb_dim=2048 \
  base_num_decoder_layers=28 \
  ...
```

| Template | Decoder Block | Notes |
|----------|--------------|-------|
| `qwen3` | Qwen3 dense | QK-norm, SwiGLU |
| `qwen3-moe` | Qwen3 MoE | Routed experts |
| `qwen3-swa` | Qwen3 SWA | Sliding window + global attention cycles |
| `qwen3-next-dense` | Qwen3-Next | Gated delta net + full attention hybrid |
| `qwen3-next-moe` | Qwen3-Next MoE | Same with routed experts |
| `llama3` | LLaMA 2/3 | GQA, RoPE |
| `deepseek` | DeepSeek V2/V3 | MLA attention, MoE |
| `gemma3` | Gemma 3 | Sliding window + global, post-norms |
| `gpt_oss` | GPT-OSS | Attention sinks, MoE |

---

## Configuration

Megatext uses a three-layer config system:

```
base.yaml  ->  models/{model}.yaml  ->  CLI overrides
```

Every field is a Pydantic model with type validation:

```bash
python -m megatext.trainers.pretrain \
  model=qwen3-swa \
  learning_rate=3e-4 \
  lr_schedule_type=constant \
  warmup_steps=1000 \
  steps=100000 \
  per_device_batch_size=8 \
  max_target_length=4096 \
  global_batch_size=7680 \
  opt_type=muon \
  scan_layers=true \
  remat_policy=full \
  ici_fsdp_parallelism=-1 \
  base_output_directory=gs://my-bucket/experiments/run-1
```

### Optimizers

| Type | Description |
|------|-------------|
| `adamw` | AdamW (default, Llama2-style) |
| `adam_pax` | Adam following PAX/Praxis implementation |
| `sgd` | SGD |
| `muon` | Muon with Newton-Schulz orthogonalization |

### LR Schedules

| Type | Description |
|------|-------------|
| `constant` | Constant LR with optional warmup |
| `cosine` | Cosine decay with warmup (default) |
| `wsd` | Warmup-Stable-Decay (linear or cosine decay phase) |

### Training Logs

Each step logs performance, loss, learning rate, and gradient norm:

```
completed step: 5, seconds: 20.585, TFLOP/s/device: 97.566, Tokens/s/device: 1989.820, total_weights: 31457280, loss: 12.409, lr: 3.000e-04, grad_norm: 1.234e+01
```

### Checkpoint Conversion

Convert HuggingFace checkpoints to Megatext format and back:

```bash
# HF -> Megatext
python -m megatext.conversion.convert \
  --model-type qwen3 \
  --hf-path /path/to/hf/model \
  --mt-path gs://bucket/megatext/model

# Megatext -> HF
python -m megatext.conversion.convert \
  --model-type qwen3 \
  --mt-path gs://bucket/megatext/model \
  --hf-path /path/to/hf/output \
  --to-hf
```

Supported model types: `qwen3`, `qwen3-swa`, `qwen3-moe`, `qwen3-next`, `llama3`, `deepseek`, `gemma3`, `gpt_oss`

---

## GKE Deployment

### Setup

```bash
bash gke/setup/setup_xpk.sh   # Install xpk CLI
```

### Job YAML Format

Jobs are defined as YAML files with config overrides:

```yaml
workload_name: pretrain-qwen3-8b

bucket: my-tpu-bucket
mount_path: /mnt/bucket

vars:
  DATA_ROOT: /mnt/bucket/datasets

config:
  model: qwen3-swa
  dataset_type: mmap
  dataset_path: >-
    10 ${DATA_ROOT}/corpus-a
    5 ${DATA_ROOT}/corpus-b
  steps: 240000
  per_device_batch_size: 2
  global_batch_size: 7680
  opt_type: muon
  ...
```

### Submit

```bash
# Build image and submit
python gke/submit.py pretrain --infra gke/infra/v5e.yaml --build gke/jobs/train/job.yaml

# Resubmit (reuse image, replace existing workload)
python gke/submit.py pretrain --infra gke/infra/v5e.yaml --force gke/jobs/train/job.yaml

# Autotune batch size and remat policy
python gke/submit.py autotune --infra gke/infra/v5e.yaml gke/jobs/train/job.yaml
```

### Monitor

```bash
xpk workload list --cluster tpu --project my-project --zone us-west4-a
kubectl logs <pod-name> --tail=20
```

---

## License

Apache 2.0 — see individual file headers for details.

Megatext builds upon [MaxText](https://github.com/AI-Hypercomputer/maxtext) by Google,
with significant restructuring, new features, and performance improvements.
