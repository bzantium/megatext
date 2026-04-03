# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for scanned multi-layer decoder blocks."""

import sys
from types import SimpleNamespace

from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh

from megatext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_TRAIN, DecoderBlockType
from megatext.common.decoder_registry import get_layer_call_kwargs, get_scannable_block_layer_count
from megatext.configs import pyconfig
from megatext.layers import attentions
from megatext.layers.decoders import Decoder
from megatext.layers.embeddings import Embed
from megatext.layers.nnx_decoders import NNXDecoder
from megatext.models import gemma2, models
from megatext.utils.sharding import create_device_mesh
from tests.utils.test_helpers import get_decoupled_parallelism_overrides, get_test_config_path

GEMMA4_BLOCK = getattr(DecoderBlockType, "GEMMA4", None)


_BASE_CONFIG = {
    "per_device_batch_size": 1.0,
    "run_name": "scannable_decoder_block_test",
    "enable_checkpointing": False,
    "base_num_decoder_layers": 4,
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
}


def _make_config(**overrides):
  extra_args = get_decoupled_parallelism_overrides()
  config_kwargs = {
      **_BASE_CONFIG,
      **extra_args,
      **overrides,
  }
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      **config_kwargs,
  )


def _make_mesh(cfg):
  devices_array = create_device_mesh(cfg)
  return Mesh(devices_array, cfg.mesh_axes)


def _make_inputs(cfg):
  batch = cfg.global_batch_size_to_train_on
  seq_len = cfg.max_target_length
  ids = jnp.arange(batch * seq_len, dtype=jnp.int32).reshape(batch, seq_len) % cfg.vocab_size
  decoder_segment_ids = jnp.full((batch, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
  decoder_positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None], (batch, seq_len))
  return ids, decoder_segment_ids, decoder_positions


def test_multi_layer_scannable_models_expose_expected_scan_body_counts():
  configs_by_block = {
      DecoderBlockType.QWEN3_SWA: SimpleNamespace(scan_layers=True, inhomogeneous_layer_cycle_interval=4),
      DecoderBlockType.QWEN3_NEXT: SimpleNamespace(scan_layers=True, inhomogeneous_layer_cycle_interval=4),
      DecoderBlockType.GEMMA2: SimpleNamespace(scan_layers=True),
      DecoderBlockType.GPT_OSS: SimpleNamespace(scan_layers=True),
      DecoderBlockType.OLMO3: SimpleNamespace(scan_layers=True),
  }
  expected_counts = {
      DecoderBlockType.QWEN3_SWA: 4,
      DecoderBlockType.QWEN3_NEXT: 4,
      DecoderBlockType.GEMMA2: 2,
      DecoderBlockType.GPT_OSS: 2,
      DecoderBlockType.OLMO3: 4,
  }
  if GEMMA4_BLOCK is not None:
    configs_by_block[GEMMA4_BLOCK] = SimpleNamespace(scan_layers=True)
    expected_counts[GEMMA4_BLOCK] = 6

  for block_type, cfg in configs_by_block.items():
    assert get_scannable_block_layer_count(block_type, cfg) == expected_counts[block_type]


@pytest.mark.parametrize(
    ("decoder_block", "cycle_interval", "sliding_window_size"),
    (
        ("qwen3_swa", 4, 4),
        ("qwen3_next", 4, 0),
        ("gemma2", 1, 4),
    ),
)
def test_scanned_multi_layer_blocks_use_outer_block_remat(decoder_block, cycle_interval, sliding_window_size):
  cfg = _make_config(
      decoder_block=decoder_block,
      inhomogeneous_layer_cycle_interval=cycle_interval,
      sliding_window_size=sliding_window_size,
  )
  mesh = _make_mesh(cfg)
  decoder = Decoder(config=cfg, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)

  block_layer = decoder.get_decoder_layers()[0]
  policy = decoder.get_remat_policy()
  rematted = decoder.set_remat_policy([block_layer], policy)[0]

  assert rematted is not block_layer


def test_gemma2_attention_types_alternate_in_unscanned_decoder():
  cfg = _make_config(decoder_block="gemma2", scan_layers=False, inhomogeneous_layer_cycle_interval=1, sliding_window_size=4)
  mesh = _make_mesh(cfg)
  rngs = nnx.Rngs(params=0, dropout=1)
  decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=rngs)

  assert decoder.layers[0].attention_type == attentions.AttentionType.LOCAL_SLIDING
  assert decoder.layers[1].attention_type == attentions.AttentionType.GLOBAL
  assert decoder.layers[2].attention_type == attentions.AttentionType.LOCAL_SLIDING
  assert gemma2.get_attention_type(3) == attentions.AttentionType.GLOBAL


def test_qwen3_swa_attention_types_alternate_in_unscanned_decoder():
  cfg = _make_config(decoder_block="qwen3_swa", scan_layers=False, inhomogeneous_layer_cycle_interval=4, sliding_window_size=4)
  mesh = _make_mesh(cfg)
  rngs = nnx.Rngs(params=0, dropout=1)
  decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=rngs)

  assert decoder.layers[0].attention_type == attentions.AttentionType.LOCAL_SLIDING
  assert decoder.layers[1].attention_type == attentions.AttentionType.LOCAL_SLIDING
  assert decoder.layers[2].attention_type == attentions.AttentionType.LOCAL_SLIDING
  assert decoder.layers[3].attention_type == attentions.AttentionType.GLOBAL


def test_gemma4_layer_call_kwargs_come_from_registry():
  if GEMMA4_BLOCK is None:
    pytest.skip("DecoderBlockType.GEMMA4 not available in this branch.")

  kwargs = get_layer_call_kwargs(
      GEMMA4_BLOCK,
      SimpleNamespace(scan_layers=False),
      bidirectional_mask="mask",
  )

  assert kwargs == {"bidirectional_mask": "mask"}


def test_qwen3_swa_scanned_model_forward_runs():
  cfg = _make_config()
  mesh = _make_mesh(cfg)
  model = models.transformer_as_linen(config=cfg, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)
  ids, decoder_segment_ids, decoder_positions = _make_inputs(cfg)
  rng = jax.random.PRNGKey(0)

  variables = model.init(
      {"params": rng, "aqt": rng, "dropout": rng},
      ids,
      decoder_positions,
      decoder_segment_ids,
      enable_dropout=False,
  )

  logits = model.apply(
      variables,
      ids,
      decoder_positions,
      decoder_segment_ids,
      enable_dropout=False,
      model_mode=MODEL_MODE_TRAIN,
      rngs={"aqt": rng},
  )

  assert logits.shape == (cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.vocab_size)


def test_qwen3_swa_scanned_nnx_decoder_forward_runs():
  cfg = _make_config()
  mesh = _make_mesh(cfg)
  rngs = nnx.Rngs(params=0, dropout=1)
  decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=rngs)
  shared_embedding = Embed(
      num_embeddings=cfg.vocab_size,
      num_features=cfg.emb_dim,
      dtype=cfg.dtype,
      embedding_init=nn.initializers.normal(stddev=1.0),
      config=cfg,
      mesh=mesh,
      rngs=rngs,
  )
  ids, decoder_segment_ids, decoder_positions = _make_inputs(cfg)

  logits, hidden_state, kv_caches = decoder(
      shared_embedding,
      ids,
      decoder_positions,
      decoder_segment_ids=decoder_segment_ids,
      deterministic=True,
      model_mode=MODEL_MODE_TRAIN,
  )

  assert logits.shape == (cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.vocab_size)
  assert hidden_state.shape == (cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.emb_dim)
  assert kv_caches is None
