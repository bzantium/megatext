# Copyright 2023–2026 Google LLC
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

"""Specialised layers for Gemma2."""

from typing import Optional

from flax import linen as nn
from flax import nnx
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from megatext.common.common_types import MODEL_MODE_PREFILL, Config
from megatext.layers import attentions
from megatext.layers import initializers
from megatext.layers import nnx_wrappers
from megatext.layers import quantizations
from megatext.layers.attentions import Attention
from megatext.layers.linears import Dropout, MlpBlock
from megatext.layers.normalizations import RMSNorm
from megatext.layers.quantizations import AqtQuantization as Quant
from megatext.layers.scannable_block import ScannableBlock
from megatext.utils.training import get_batch_seq_len_for_mode


GEMMA2_ATTENTION_PATTERN = (
    attentions.AttentionType.LOCAL_SLIDING,
    attentions.AttentionType.GLOBAL,
)


def get_attention_type(layer_id):
  """Get attention type based on layer ID."""
  layer_id %= len(GEMMA2_ATTENTION_PATTERN)
  return GEMMA2_ATTENTION_PATTERN[layer_id]


class Gemma2DecoderLayer(nnx.Module):
  """Single-attention Gemma2 decoder layer."""

  config: Config
  mesh: Mesh
  model_mode: str
  quant: None | Quant = None

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      attention_type: attentions.AttentionType = attentions.AttentionType.LOCAL_SLIDING,
      quant: Optional[Quant] = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.attention_type = attention_type
    self.quant = quant
    self.rngs = rngs

    batch_size, seq_len = get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)
    is_global = attention_type == attentions.AttentionType.GLOBAL

    self.pre_self_attention_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    self.self_attention = Attention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        mesh=self.mesh,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        dropout_rate=config.dropout_rate,
        float32_qk_product=True if is_global else config.float32_qk_product,
        float32_logits=True if is_global else config.float32_logits,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(config),
        attention_type=self.attention_type,
        sliding_window_size=config.sliding_window_size,
        attn_logits_soft_cap=config.attn_logits_soft_cap,
        model_mode=self.model_mode,
        rngs=self.rngs,
    )

    if config.use_post_attn_norm:
      self.post_self_attention_norm = RMSNorm(
          num_features=config.emb_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
    else:
      self.post_self_attention_norm = None

    self.pre_ffw_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    self.mlp = MlpBlock(
        config=config,
        mesh=self.mesh,
        in_features=config.emb_dim,
        intermediate_dim=config.mlp_dim,
        activations=config.mlp_activations,
        intermediate_dropout_rate=config.dropout_rate,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        quant=self.quant,
        model_mode=self.model_mode,
        rngs=self.rngs,
    )

    if config.use_post_ffw_norm:
      self.post_ffw_norm = RMSNorm(
          num_features=config.emb_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
    else:
      self.post_ffw_norm = None

    self.dropout = Dropout(rate=config.dropout_rate, broadcast_dims=(-2,), rngs=self.rngs)

    if model_mode == MODEL_MODE_PREFILL:
      self.activation_axis_names = ("activation_batch", "prefill_activation_norm_length", "activation_embed")
    else:
      self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state=None,
      slot=None,
      kv_cache=None,
      attention_metadata=None,
  ):
    del previous_chunk, page_state, slot

    if isinstance(inputs, tuple):
      inputs = inputs[0]
    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    lnx = self.pre_self_attention_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, self.activation_axis_names)

    attention_lnx, kv_cache = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    if self.post_self_attention_norm is not None:
      attention_lnx = self.post_self_attention_norm(attention_lnx)

    attention_lnx = nn.with_logical_constraint(attention_lnx, self.activation_axis_names)
    residual = attention_lnx + inputs

    attn_output = self.pre_ffw_norm(residual)
    mlp_lnx = self.mlp(attn_output, deterministic=deterministic)

    if self.post_ffw_norm is not None:
      mlp_lnx = self.post_ffw_norm(mlp_lnx)

    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)
    layer_output = self.dropout(mlp_lnx + residual, deterministic=deterministic)
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    if self.config.record_internal_nn_metrics:
      self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
      self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
      self.sow(
          "intermediates",
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if self.config.scan_layers:
      return layer_output, None
    return layer_output, kv_cache


Gemma2DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Gemma2DecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


def _gemma2_layer_kwargs(i, config):
  del config
  return {"attention_type": get_attention_type(i)}


class Gemma2ScannableBlock(ScannableBlock):
  """A repeatable block of Gemma2 decoder layers."""

  @classmethod
  def scan_body_layer_count(cls, config: Config) -> int:
    del config
    return len(GEMMA2_ATTENTION_PATTERN)

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: Optional[Quant] = None,
      *,
      rngs: nnx.Rngs,
  ):
    super().__init__(
        config=config,
        mesh=mesh,
        model_mode=model_mode,
        quant=quant,
        layer_cls=Gemma2DecoderLayer,
        layer_kwargs_fn=_gemma2_layer_kwargs,
        rngs=rngs,
    )

