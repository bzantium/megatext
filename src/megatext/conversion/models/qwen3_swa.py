"""Qwen3-SWA model_type: mapping + transforms + shapes."""
from __future__ import annotations

from typing import Any

from megatext.conversion.models.qwen3 import build_qwen3, build_qwen3_transforms
from megatext.conversion.utils import (
    ArchSpec,
    Mapping,
    _cfg,
    _global_keys,
    pad_vocab_size,
)


def _qwen3_swa_cycle(cfg: Any) -> int:
  interval = getattr(cfg, "full_attention_interval", getattr(cfg, "inhomogeneous_layer_cycle_interval", None))
  if interval:
    return int(interval)

  layer_types = list(getattr(cfg, "layer_types", []) or [])
  if layer_types:
    for cycle in range(1, len(layer_types) + 1):
      if len(layer_types) % cycle != 0:
        continue
      pattern = layer_types[:cycle]
      if pattern * (len(layer_types) // cycle) == layer_types:
        return cycle

  return 4


def build_qwen3_swa(arch: ArchSpec, hf_config: Any, scan_layers: bool) -> Mapping:
  """Build Qwen3-SWA checkpoint mapping.

  Unscanned checkpoints share dense Qwen3 layout. Scanned checkpoints use one
  stacked parameter array per position inside the SWA cycle, matching the
  explicit Linen scanned training path.
  """
  if not scan_layers:
    return build_qwen3(arch, hf_config, scan_layers=False)

  cfg = _cfg(hf_config)
  n_layers = cfg.num_hidden_layers
  cycle = _qwen3_swa_cycle(cfg)
  if n_layers % cycle != 0:
    raise ValueError(f"qwen3_swa scan requires num_hidden_layers divisible by cycle={cycle}, got {n_layers}.")

  tie = getattr(cfg, "tie_word_embeddings", False)
  prefix = arch.hf_prefix

  mapping: Mapping = {}
  mapping.update(_global_keys(prefix, tie))

  components: list[tuple[str, str]] = [
      ("self_attention-query-kernel", "self_attn.q_proj.weight"),
      ("self_attention-key-kernel", "self_attn.k_proj.weight"),
      ("self_attention-value-kernel", "self_attn.v_proj.weight"),
      ("self_attention-out-kernel", "self_attn.o_proj.weight"),
      ("mlp-wi_0-kernel", "mlp.gate_proj.weight"),
      ("mlp-wi_1-kernel", "mlp.up_proj.weight"),
      ("mlp-wo-kernel", "mlp.down_proj.weight"),
      ("pre_self_attention_layer_norm-scale", "input_layernorm.weight"),
      ("post_self_attention_layer_norm-scale", "post_attention_layernorm.weight"),
      ("self_attention-query_norm-scale", "self_attn.q_norm.weight"),
      ("self_attention-key_norm-scale", "self_attn.k_norm.weight"),
  ]

  for block_idx in range(cycle):
    hf_indices = list(range(block_idx, n_layers, cycle))
    block_prefix = f"params-decoder-layers-layers_{block_idx}"
    for mt_suffix, hf_suffix in components:
      mapping[f"{block_prefix}-{mt_suffix}"] = [
          f"{prefix}.layers.{i}.{hf_suffix}" for i in hf_indices
      ]

  return mapping


build_qwen3_swa_transforms = build_qwen3_transforms


def compute_qwen3_swa_shapes(
    hf_config: Any,
    arch: ArchSpec,
    scan_layers: bool,
    *,
    tie_word_embeddings: bool = False,
) -> dict[str, tuple]:
  """Compute Qwen3-SWA Megatext parameter shapes."""
  cfg = _cfg(hf_config)
  if not scan_layers:
    # Dense layout matches qwen3.
    emb = cfg.hidden_size
    mlp = cfg.intermediate_size
    nq = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", emb // nq)
    vocab = cfg.vocab_size
    n_layers = cfg.num_hidden_layers
    padded_vocab = pad_vocab_size(vocab)

    shapes: dict[str, tuple] = {
        "params-token_embedder-embedding": (padded_vocab, emb),
        "params-decoder-decoder_norm-scale": (emb,),
    }
    if not tie_word_embeddings:
      shapes["params-decoder-logits_dense-kernel"] = (emb, padded_vocab)

    per_layer_shapes: dict[str, tuple] = {
        "self_attention-query-kernel": (emb, nq, hd),
        "self_attention-key-kernel": (emb, nkv, hd),
        "self_attention-value-kernel": (emb, nkv, hd),
        "self_attention-out-kernel": (nq, hd, emb),
        "mlp-wi_0-kernel": (emb, mlp),
        "mlp-wi_1-kernel": (emb, mlp),
        "mlp-wo-kernel": (mlp, emb),
        "pre_self_attention_layer_norm-scale": (emb,),
        "post_self_attention_layer_norm-scale": (emb,),
        "self_attention-query_norm-scale": (hd,),
        "self_attention-key_norm-scale": (hd,),
    }
    for layer_idx in range(n_layers):
      prefix = f"params-decoder-layers_{layer_idx}"
      for suffix, shape in per_layer_shapes.items():
        shapes[f"{prefix}-{suffix}"] = shape
    return shapes

  emb = cfg.hidden_size
  mlp = cfg.intermediate_size
  nq = cfg.num_attention_heads
  nkv = cfg.num_key_value_heads
  hd = getattr(cfg, "head_dim", emb // nq)
  vocab = cfg.vocab_size
  n_layers = cfg.num_hidden_layers
  cycle = _qwen3_swa_cycle(cfg)
  if n_layers % cycle != 0:
    raise ValueError(f"qwen3_swa scan requires num_hidden_layers divisible by cycle={cycle}, got {n_layers}.")
  padded_vocab = pad_vocab_size(vocab)

  shapes: dict[str, tuple] = {
      "params-token_embedder-embedding": (padded_vocab, emb),
      "params-decoder-decoder_norm-scale": (emb,),
  }
  if not tie_word_embeddings:
    shapes["params-decoder-logits_dense-kernel"] = (emb, padded_vocab)

  for block_idx in range(cycle):
    n_stacked = len(range(block_idx, n_layers, cycle))
    lp = (n_stacked,)
    prefix = f"params-decoder-layers-layers_{block_idx}"
    shapes[f"{prefix}-self_attention-query-kernel"] = (*lp, emb, nq, hd)
    shapes[f"{prefix}-self_attention-key-kernel"] = (*lp, emb, nkv, hd)
    shapes[f"{prefix}-self_attention-value-kernel"] = (*lp, emb, nkv, hd)
    shapes[f"{prefix}-self_attention-out-kernel"] = (*lp, nq, hd, emb)
    shapes[f"{prefix}-mlp-wi_0-kernel"] = (*lp, emb, mlp)
    shapes[f"{prefix}-mlp-wi_1-kernel"] = (*lp, emb, mlp)
    shapes[f"{prefix}-mlp-wo-kernel"] = (*lp, mlp, emb)
    shapes[f"{prefix}-pre_self_attention_layer_norm-scale"] = (*lp, emb)
    shapes[f"{prefix}-post_self_attention_layer_norm-scale"] = (*lp, emb)
    shapes[f"{prefix}-self_attention-query_norm-scale"] = (*lp, hd)
    shapes[f"{prefix}-self_attention-key_norm-scale"] = (*lp, hd)

  return shapes
