"""Qwen3 decoder block with sliding window + global attention pattern.

Alternates between LOCAL_SLIDING and GLOBAL attention layers using
inhomogeneous_layer_cycle_interval.
The last layer in each cycle uses global attention; the rest use sliding window.

Usage:
    decoder_block: qwen3_swa
    inhomogeneous_layer_cycle_interval: 4   # 3 sliding + 1 global
    sliding_window_size: 4096
    scan_layers: true
"""

from __future__ import annotations

from typing import Any

from flax import linen as nn
from flax import nnx
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh

from megatext.common.common_types import AttentionType
from megatext.inference import page_manager
from megatext.layers import quantizations
from megatext.layers.scannable_block import ScannableBlock
from megatext.layers.attentions import Attention
from megatext.layers.normalizations import RMSNorm
from megatext.models.qwen3 import MlpBlock
from megatext.utils.training import get_batch_seq_len_for_mode

Config = Any
Quant = Any


class Qwen3SWADecoderLayer(nnx.Module):
  """Qwen3 decoder layer with configurable attention type (global or sliding window)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant = None,
      attention_type: AttentionType = AttentionType.GLOBAL,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.attention_type = attention_type

    batch_size, seq_len = get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)
    self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

    sliding_window_size = config.sliding_window_size if attention_type == AttentionType.LOCAL_SLIDING else None
    query_pre_attn_scalar = config.head_dim**-0.5

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
        mesh=mesh,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        dropout_rate=config.dropout_rate,
        float32_qk_product=config.float32_qk_product,
        float32_logits=config.float32_logits,
        quant=quant,
        kv_quant=quantizations.configure_kv_quant(config),
        use_ragged_attention=config.use_ragged_attention,
        ragged_block_size=config.ragged_block_size,
        use_qk_norm=config.use_qk_norm,
        query_pre_attn_scalar=query_pre_attn_scalar,
        model_mode=model_mode,
        attention_type=attention_type,
        sliding_window_size=sliding_window_size,
        rngs=rngs,
    )

    self.post_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

    self.mlp = MlpBlock(
        in_features=config.emb_dim,
        intermediate_dim=config.mlp_dim,
        activations=config.mlp_activations,
        intermediate_dropout_rate=config.dropout_rate,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        config=config,
        mesh=mesh,
        quant=quant,
        model_mode=model_mode,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
      kv_cache: None | jnp.ndarray = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    # Pre-norm + attention
    lnx = self.pre_self_attention_layer_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, self.activation_axis_names)
    attention_lnx, kv_cache = self.self_attention(
        lnx, lnx, decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    attention_lnx = nn.with_logical_constraint(attention_lnx, self.activation_axis_names)
    intermediate = inputs + attention_lnx

    # Post-norm + MLP
    hidden = self.post_self_attention_layer_norm(intermediate)
    hidden = nn.with_logical_constraint(hidden, self.activation_axis_names)
    mlp_out = self.mlp(hidden, deterministic=deterministic)
    mlp_out = nn.with_logical_constraint(mlp_out, self.activation_axis_names)
    output = intermediate + mlp_out
    output = nn.with_logical_constraint(output, self.activation_axis_names)

    if self.config.scan_layers:
      return output, None
    return output, kv_cache


def _qwen3_swa_layer_kwargs(i, config):
  """Return per-layer kwargs for Qwen3 SWA: last in cycle is global, rest are sliding."""
  cycle = config.inhomogeneous_layer_cycle_interval
  is_global = (i + 1) % cycle == 0
  return {"attention_type": AttentionType.GLOBAL if is_global else AttentionType.LOCAL_SLIDING}


class Qwen3SWAScannableBlock(ScannableBlock):
  """Scannable block of Qwen3 layers with mixed sliding/global attention.

  Within each cycle of inhomogeneous_layer_cycle_interval layers:
  - Last layer: global attention
  - All others: sliding window attention
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant = None,
      *,
      rngs: nnx.Rngs,
  ):
    super().__init__(
        config, mesh, model_mode, quant,
        layer_cls=Qwen3SWADecoderLayer,
        layer_kwargs_fn=_qwen3_swa_layer_kwargs,
        rngs=rngs,
    )
