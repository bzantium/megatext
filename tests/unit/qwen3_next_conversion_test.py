"""Qwen3-Next conversion mapping/shape tests (dense num_experts=0 + MoE)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict
from jax.sharding import Mesh

from megatext.configs import pyconfig
from megatext.conversion.models import ARCH_SPECS
from megatext.conversion.models.qwen3_next import (
    build_qwen3_next,
    compute_qwen3_next_shapes,
)
from megatext.models import models
from megatext.utils.sharding import create_device_mesh
from tests.utils.test_helpers import get_decoupled_parallelism_overrides, get_test_config_path


def _ns(obj):
  if isinstance(obj, dict):
    return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
  if isinstance(obj, list):
    return [_ns(v) for v in obj]
  return obj


def _qwen3_next_hf_config(*, num_experts: int, num_layers: int = 8):
  """Small qwen3-next HF-style config. num_experts=0 → dense, >0 → MoE."""
  cfg = {
      "model_type": "qwen3_next",
      "tie_word_embeddings": False,
      "hidden_size": 128,
      "intermediate_size": 256,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      "head_dim": 32,
      "vocab_size": 1024,
      "num_hidden_layers": num_layers,
      "full_attention_interval": 4,
      "num_experts": num_experts,
      "linear_num_key_heads": 4,
      "linear_num_value_heads": 4,
      "linear_key_head_dim": 32,
      "linear_value_head_dim": 32,
      "linear_conv_kernel_dim": 4,
  }
  if num_experts > 0:
    cfg["moe_intermediate_size"] = 64
    cfg["shared_expert_intermediate_size"] = 64
  return _ns(cfg)


def _unwrap_shape(value):
  return getattr(value, "value", value).shape


# ── Dense (num_experts=0) ───────────────────────────────────────────────────


def test_build_qwen3_next_dense_maps_plain_swiglu():
  hf = _qwen3_next_hf_config(num_experts=0, num_layers=8)
  mapping = build_qwen3_next(ARCH_SPECS["qwen3_next"], hf, scan_layers=True)

  assert mapping["params-decoder-layers-layers_0-mlp-wi_0-kernel"] == [
      "model.layers.0.mlp.gate_proj.weight",
      "model.layers.4.mlp.gate_proj.weight",
  ]
  assert mapping["params-decoder-layers-layers_0-mlp-wi_1-kernel"] == [
      "model.layers.0.mlp.up_proj.weight",
      "model.layers.4.mlp.up_proj.weight",
  ]
  assert mapping["params-decoder-layers-layers_0-mlp-wo-kernel"] == [
      "model.layers.0.mlp.down_proj.weight",
      "model.layers.4.mlp.down_proj.weight",
  ]
  # No MoE keys in the dense mapping.
  assert not any("routed_experts" in str(k) or "shared_expert" in str(k) for k in mapping)


def test_compute_qwen3_next_dense_shapes():
  hf = _qwen3_next_hf_config(num_experts=0, num_layers=8)
  shapes = compute_qwen3_next_shapes(hf, ARCH_SPECS["qwen3_next"], scan_layers=True)

  # (n_stacked=2, emb=128, intermediate=256) and (n_stacked, intermediate, emb).
  assert shapes["params-decoder-layers-layers_0-mlp-wi_0-kernel"] == (2, 128, 256)
  assert shapes["params-decoder-layers-layers_0-mlp-wi_1-kernel"] == (2, 128, 256)
  assert shapes["params-decoder-layers-layers_0-mlp-wo-kernel"] == (2, 256, 128)
  # Attention/GDN shapes preserved: flat-2D out kernel, conv1d (conv_k, 1, conv_dim).
  assert shapes["params-decoder-layers-layers_3-attention-attention-out-kernel"] == (2, 4 * 32, 128)
  # conv1d kernel is (n_stacked, conv_k=4, 1, conv_dim); conv_k leads the matrix dims.
  conv = shapes["params-decoder-layers-layers_0-attention-conv1d-kernel"]
  assert conv[0] == 2 and conv[1] == 4 and conv[2] == 1


# ── MoE (num_experts>0) regression: unchanged ──────────────────────────────


def test_build_qwen3_next_moe_still_maps_routed_and_shared():
  hf = _qwen3_next_hf_config(num_experts=512, num_layers=8)
  mapping = build_qwen3_next(ARCH_SPECS["qwen3_next"], hf, scan_layers=True)

  assert any("routed_experts" in str(k) for k in mapping)
  assert any("shared_expert" in str(k) for k in mapping)
  # Dense keys must NOT appear when MoE.
  assert "params-decoder-layers-layers_0-mlp-wi_0-kernel" not in mapping


# ── Linen param layout matches the dense conversion keys ────────────────────


def _make_dense_runtime_config():
  config_kwargs = {
      **get_decoupled_parallelism_overrides(),
      "run_name": "qwen3_next_conversion_test",
      "enable_checkpointing": False,
      "log_config": False,
      "per_device_batch_size": 1.0,
      "decoder_block": "qwen3_next",
      "num_experts": 0,
      "base_num_decoder_layers": 8,
      "attention": "dot_product",
      "max_target_length": 8,
      "base_emb_dim": 128,
      "base_mlp_dim": 256,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 2,
      "head_dim": 32,
      "scan_layers": True,
      "inhomogeneous_layer_cycle_interval": 4,
  }
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **config_kwargs)


def test_dense_qwen3_next_linen_params_match_conversion_layout():
  cfg = _make_dense_runtime_config()
  mesh = Mesh(create_device_mesh(cfg), cfg.mesh_axes)
  model = models.transformer_as_linen(config=cfg, mesh=mesh, quant=None, model_mode="train")
  batch = cfg.global_batch_size_to_train_on
  seq_len = cfg.max_target_length
  ids = jnp.arange(batch * seq_len, dtype=jnp.int32).reshape(batch, seq_len) % cfg.vocab_size
  seg = jnp.ones((batch, seq_len), dtype=jnp.int32)
  pos = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None], (batch, seq_len))
  rng = jax.random.PRNGKey(0)
  variables = model.init(
      {"params": rng, "aqt": rng, "dropout": rng}, ids, pos, seg, enable_dropout=False
  )
  flat = flatten_dict(variables["params"])

  # Dense MLP present, MoE absent.
  assert ("decoder", "layers", "layers_0", "mlp", "wi_0", "kernel") in flat
  assert ("decoder", "layers", "layers_0", "mlp", "wi_1", "kernel") in flat
  assert ("decoder", "layers", "layers_0", "mlp", "wo", "kernel") in flat
  assert not any("routed_experts" in "/".join(map(str, k)) for k in flat)
  assert not any("shared_expert" in "/".join(map(str, k)) for k in flat)
