"""Qwen3-SWA conversion mapping/shape tests."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict
from jax.sharding import Mesh

from megatext.configs import pyconfig
from megatext.conversion.convert import compute_megatext_shapes
from megatext.conversion.models import ARCH_SPECS
from megatext.conversion.models.qwen3_swa import build_qwen3_swa
from megatext.models import models
from megatext.utils.sharding import create_device_mesh
from tests.utils.test_helpers import get_decoupled_parallelism_overrides, get_test_config_path


def _ns(obj):
  if isinstance(obj, dict):
    return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
  if isinstance(obj, list):
    return [_ns(v) for v in obj]
  return obj


def _qwen3_swa_hf_config(*, num_layers: int = 8):
  return _ns(
      {
          "model_type": "qwen3_swa",
          "tie_word_embeddings": False,
          "hidden_size": 128,
          "intermediate_size": 256,
          "num_attention_heads": 4,
          "num_key_value_heads": 4,
          "head_dim": 32,
          "vocab_size": 1024,
          "num_hidden_layers": num_layers,
          "full_attention_interval": 4,
      }
  )


def _make_runtime_config(**overrides):
  config_kwargs = {
      **get_decoupled_parallelism_overrides(),
      "run_name": "qwen3_swa_conversion_test",
      "enable_checkpointing": False,
      "log_config": False,
      "per_device_batch_size": 1.0,
      "base_num_decoder_layers": 8,
      "attention": "dot_product",
      "max_target_length": 8,
      "base_emb_dim": 128,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 4,
      "base_mlp_dim": 256,
      "max_prefill_predict_length": 4,
      "scan_layers": True,
      "decoder_block": "qwen3_swa",
      "inhomogeneous_layer_cycle_interval": 4,
      "sliding_window_size": 4,
      **overrides,
  }
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **config_kwargs)


def _make_mesh(cfg):
  devices_array = create_device_mesh(cfg)
  return Mesh(devices_array, cfg.mesh_axes)


def _unwrap_shape(value):
  return getattr(value, "value", value).shape


def test_build_qwen3_swa_scanned_mapping_uses_cycle_prefixes():
  hf_config = _qwen3_swa_hf_config(num_layers=8)
  mapping = build_qwen3_swa(ARCH_SPECS["qwen3_swa"], hf_config, scan_layers=True)

  assert mapping["params-decoder-layers-layers_0-self_attention-query-kernel"] == [
      "model.layers.0.self_attn.q_proj.weight",
      "model.layers.4.self_attn.q_proj.weight",
  ]
  assert mapping["params-decoder-layers-layers_3-self_attention-key-kernel"] == [
      "model.layers.3.self_attn.k_proj.weight",
      "model.layers.7.self_attn.k_proj.weight",
  ]


def test_compute_qwen3_swa_scanned_shapes_match_nested_cycle_layout():
  hf_config = _qwen3_swa_hf_config(num_layers=8)
  shapes = compute_megatext_shapes(hf_config, ARCH_SPECS["qwen3_swa"], scan_layers=True, tie_word_embeddings=False)

  assert shapes["params-decoder-layers-layers_0-self_attention-query-kernel"] == (2, 128, 4, 32)
  assert shapes["params-decoder-layers-layers_3-self_attention-value-kernel"] == (2, 128, 4, 32)
  assert shapes["params-decoder-layers-layers_1-mlp-wi_0-kernel"] == (2, 128, 256)
  assert shapes["params-decoder-logits_dense-kernel"] == (128, 1024)


def test_scanned_qwen3_swa_linen_params_match_conversion_layout():
  cfg = _make_runtime_config(base_num_decoder_layers=8)
  mesh = _make_mesh(cfg)
  model = models.transformer_as_linen(config=cfg, mesh=mesh, quant=None, model_mode="train")
  batch = cfg.global_batch_size_to_train_on
  seq_len = cfg.max_target_length
  ids = jnp.arange(batch * seq_len, dtype=jnp.int32).reshape(batch, seq_len) % cfg.vocab_size
  decoder_segment_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)
  decoder_positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None], (batch, seq_len))
  rng = jax.random.PRNGKey(0)

  variables = model.init(
      {"params": rng, "aqt": rng, "dropout": rng},
      ids,
      decoder_positions,
      decoder_segment_ids,
      enable_dropout=False,
  )
  flat = flatten_dict(variables["params"])

  query0 = flat[("decoder", "layers", "layers_0", "self_attention", "query", "kernel")]
  query3 = flat[("decoder", "layers", "layers_3", "self_attention", "query", "kernel")]

  shape0 = _unwrap_shape(query0)
  shape3 = _unwrap_shape(query3)

  assert shape0[0] == cfg.emb_dim
  assert shape3[0] == cfg.emb_dim
  assert shape0[1] == 2
  assert shape3[1] == 2
  assert ("decoder", "layers", "layers_0", "self_attention", "query", "kernel") in flat
  assert ("decoder", "layers", "layers_3", "self_attention", "query", "kernel") in flat
