<p align="center">
  <h1 align="center">Megatext</h1>
  <p align="center">Simple, performant and scalable JAX LLM pretraining</p>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#data-pipeline">Data Pipeline</a> &bull;
  <a href="#supported-models">Models</a> &bull;
  <a href="#configuration">Configuration</a>
</p>

---

**Megatext** is a streamlined pretraining framework for large language models on TPUs.
Inspired by [Google's MaxText](https://github.com/AI-Hypercomputer/maxtext), it strips away
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
python gke/submit.py \
  --infra gke/infra/v5e.yaml \
  --build \
  gke/jobs/train/example_pretrain_load.sh
```

---

## Architecture

```
src/megatext/
  trainers/pretrain.py       # Training loop entry point
  configs/
    base.yaml                # Default config (all fields documented)
    models/                  # Architecture templates (qwen3, llama3, ...)
    types.py                 # Pydantic config model
    pyconfig.py              # YAML + CLI + env merge logic
  data/
    _helpers.cpp             # C++ packing (pybind11, segment tree)
    packing.py               # greedy + best-fit sample index builders
    data_sources.py          # Document-level data source with cached indexing
    data_processing.py       # Grain pipeline (shard -> repeat -> transform -> batch)
  models/                    # Decoder implementations (qwen3, llama, deepseek, ...)
  layers/                    # Attention, MLP, normalization, quantization
  optimizers/                # AdamW, Muon
  schedulers/                # Cosine, WSD learning rate schedules
  conversion/                # Bidirectional HF <-> Megatext checkpoint conversion
  autotune/                  # Automated hyperparameter search
```

---

## Data Pipeline

Megatext's data pipeline is built for speed at scale.

### C++ Packing Helpers

Index building uses C++ with pybind11 — no Python loops over millions of documents.

| Algorithm | 19M docs | Description |
|-----------|----------|-------------|
| `greedy` | 1.6s | Cross-document concatenation (Megatron-style) |
| `best_fit` | 7.4s | Segment-tree bin packing, O(N log seq_len) |

```bash
# Use greedy (default)
python -m megatext.trainers.pretrain grain_packing_type=greedy ...

# Use best-fit for higher token efficiency
python -m megatext.trainers.pretrain grain_packing_type=best_fit ...
```

### GCS-Native Caching

Data indices are cached and reused across runs. Cache supports both local and GCS paths:

```bash
# Cache on GCS (fast, persistent, shared across pods)
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

Every field is a Pydantic model with type validation and documentation:

```bash
# All config fields are CLI-settable
python -m megatext.trainers.pretrain \
  model=qwen3 \
  learning_rate=3e-4 \
  warmup_steps=1000 \
  steps=100000 \
  per_device_batch_size=8 \
  max_target_length=4096 \
  scan_layers=true \
  remat_policy=full \
  ici_fsdp_parallelism=-1 \
  base_output_directory=gs://my-bucket/experiments/run-1
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

---

## GKE Deployment

### Setup

```bash
bash gke/setup/setup_xpk.sh   # Install xpk CLI
```

### Submit a Job

```bash
python gke/submit.py \
  --infra gke/infra/v5e.yaml \
  --build \
  gke/jobs/train/example_pretrain.sh
```

`--build` builds and pushes the Docker image. `--force` replaces an existing workload. Omit `--build` to reuse the last image.

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
