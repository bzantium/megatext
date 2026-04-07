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

"""Specialised layers for Gemma4."""

from __future__ import annotations

from typing import Optional

from flax import linen as nn
from flax import nnx
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from megatext.common.common_types import AttentionType, Config, MODEL_MODE_PREFILL
from megatext.layers import initializers
from megatext.layers import moe
from megatext.layers import nnx_wrappers
from megatext.layers import quantizations
from megatext.layers.attentions import Attention
from megatext.layers.linears import MlpBlock
from megatext.layers.normalizations import RMSNorm
from megatext.layers.quantizations import AqtQuantization as Quant
from megatext.layers.scannable_block import ScannableBlock
from megatext.utils.training import get_batch_seq_len_for_mode


GEMMA4_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)


def get_attention_type(layer_id: int) -> AttentionType:
  """Returns the Gemma4 attention type for a logical decoder layer."""
  layer_id %= len(GEMMA4_ATTENTION_PATTERN)
  return GEMMA4_ATTENTION_PATTERN[layer_id]


def _gemma4_layer_kwargs(i: int, config: Config) -> dict[str, AttentionType]:
  del config
  return {"attention_type": get_attention_type(i)}


def _gemma4_layer_call_kwargs(*, config: Config, bidirectional_mask=None, **unused_context):
  del config, unused_context
  return {"bidirectional_mask": bidirectional_mask}


class Gemma4MoE(nnx.Module):
  """Gemma4-specific MoE block containing layer norms and routed/shared experts."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: None | Quant = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs

    self.moe_block = moe.RoutedAndSharedMoE(
        config=config,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        quant=self.quant,
        rngs=self.rngs,
    )
    self.pre_forward_scale_2 = nnx.Param(
        jnp.ones((self.config.emb_dim,), dtype=self.config.weight_dtype),
        sharding=("embed",),
    )
    self.pre_feedforward_layernorm_2 = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    gate_dtype = jnp.float32 if getattr(self.config, "float32_gate_logits", False) else self.config.dtype
    self.gate_norm = RMSNorm(
        num_features=self.config.emb_dim,
        epsilon=self.config.normalization_layer_epsilon,
        dtype=gate_dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        with_scale=False,
        rngs=self.rngs,
    )
    self.post_feedforward_layernorm_1 = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.post_feedforward_layernorm_2 = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      original_inputs: jnp.ndarray | None = None,
  ) -> tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]:
    shared_experts = self.moe_block.shared_experts(inputs)
    shared_experts = self.post_feedforward_layernorm_1(shared_experts)

    base_inputs = original_inputs if original_inputs is not None else inputs
    routed_inputs = self.pre_feedforward_layernorm_2(base_inputs)
    gate_dtype = jnp.float32 if getattr(self.config, "float32_gate_logits", False) else self.config.dtype
    unscaled_norm = self.gate_norm(base_inputs)
    root_size = self.config.emb_dim**-0.5
    router_scale = jnp.asarray(self.pre_forward_scale_2.value, gate_dtype)
    gate_inputs = unscaled_norm * root_size * router_scale
    routed_experts, load_balance_loss, moe_bias_updates = self.moe_block.routed_moe(
        routed_inputs,
        gate_inputs=gate_inputs,
    )
    routed_experts = self.post_feedforward_layernorm_2(routed_experts)

    return routed_experts + shared_experts, load_balance_loss, moe_bias_updates


class Gemma4DecoderLayer(nnx.Module):
  """Transformer decoder layer for Gemma4."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant = None,
      attention_type: AttentionType = AttentionType.LOCAL_SLIDING,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs
    self.attention_type = attention_type

    batch_size, seq_len = get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

    self.pre_self_attention_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim
    share_kv_projections = False
    if attention_type == AttentionType.GLOBAL:
      if config.global_num_kv_heads > 0:
        num_kv_heads = config.global_num_kv_heads
      if config.global_head_dim > 0:
        head_dim = config.global_head_dim
      share_kv_projections = config.share_kv_projections

    if attention_type == AttentionType.GLOBAL:
      partial_rotary_factor = config.global_rope_proportion
      rope_theta = (
          config.global_rope_theta if config.global_rope_theta > 0 else config.rope_theta
      )
    else:
      partial_rotary_factor = config.local_rope_proportion
      rope_theta = (
          config.local_rope_theta if config.local_rope_theta > 0 else config.rope_theta
      )

    self.self_attention = Attention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
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
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(config),
        attention_type=self.attention_type,
        sliding_window_size=config.sliding_window_size,
        attn_logits_soft_cap=config.attn_logits_soft_cap,
        use_qk_norm=True,
        use_v_norm=True,
        query_pre_attn_scalar=1.0,
        share_kv_projections=share_kv_projections,
        rope_theta=rope_theta,
        partial_rotary_factor=partial_rotary_factor,
        model_mode=model_mode,
        rngs=self.rngs,
    )

    if config.use_post_attn_norm:
      self.post_self_attention_norm = RMSNorm(
          num_features=config.emb_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          epsilon=config.normalization_layer_epsilon,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
    else:
      self.post_self_attention_norm = None

    self.pre_ffw_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    if config.num_experts > 1:
      self.mlp = Gemma4MoE(
          config=config,
          mesh=mesh,
          quant=self.quant,
          rngs=self.rngs,
      )
    else:
      self.mlp = MlpBlock(
          config=config,
          mesh=mesh,
          in_features=config.emb_dim,
          intermediate_dim=config.mlp_dim,
          activations=config.mlp_activations,
          intermediate_dropout_rate=config.dropout_rate,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          quant=self.quant,
          model_mode=model_mode,
          rngs=self.rngs,
      )

    if config.use_post_ffw_norm:
      self.post_ffw_norm = RMSNorm(
          num_features=config.emb_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          epsilon=config.normalization_layer_epsilon,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
    else:
      self.post_ffw_norm = None

    self.layer_scalar = nnx.Param(jnp.ones((1,), dtype=config.dtype), sharding=(None,))
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
      bidirectional_mask=None,
      kv_cache=None,
      attention_metadata=None,
  ):
    del previous_chunk, page_state, slot
    cfg = self.config
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    lnx = self.pre_self_attention_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, self.activation_axis_names)

    if self.attention_type != AttentionType.LOCAL_SLIDING:
      bidirectional_mask = None

    attention_lnx, kv_cache = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        bidirectional_mask=bidirectional_mask,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    if self.post_self_attention_norm is not None:
      attention_lnx = self.post_self_attention_norm(attention_lnx)
    attention_lnx = nn.with_logical_constraint(attention_lnx, self.activation_axis_names)

    residual = attention_lnx + inputs
    attn_output = self.pre_ffw_norm(residual)

    if cfg.num_experts > 1:
      mlp_lnx, load_balance_loss, _ = self.mlp(attn_output, original_inputs=residual)
      if cfg.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
        self.sow("intermediates", "moe_lb_loss", load_balance_loss)
    else:
      mlp_lnx = self.mlp(attn_output, deterministic=deterministic)

    if self.post_ffw_norm is not None:
      mlp_lnx = self.post_ffw_norm(mlp_lnx)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)

    layer_output = (mlp_lnx + residual) * self.layer_scalar.value
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    if cfg.record_internal_nn_metrics:
      self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
      self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
      self.sow(
          "intermediates",
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if cfg.scan_layers:
      return layer_output, None
    return layer_output, kv_cache


Gemma4DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Gemma4DecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class Gemma4ScannableBlock(ScannableBlock):
  """Scannable block for one Gemma4 local/local/local/local/local/global cycle."""

  @classmethod
  def scan_body_layer_count(cls, config: Config) -> int:
    del config
    return len(GEMMA4_ATTENTION_PATTERN)

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
        config=config,
        mesh=mesh,
        model_mode=model_mode,
        quant=quant,
        layer_cls=Gemma4DecoderLayer,
        layer_kwargs_fn=_gemma4_layer_kwargs,
        rngs=rngs,
    )
