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

"""Vision transformer implementation for Gemma4."""

from typing import cast

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx
from jax.sharding import Mesh

from megatext.common.common_types import AttentionType, Config, DecoderBlockType
from megatext.layers import attentions
from megatext.layers import initializers
from megatext.layers import linears
from megatext.layers import nnx_wrappers
from megatext.layers import normalizations

DEFAULT_GEMMA4_IMAGE_HEIGHT = 672
DEFAULT_GEMMA4_IMAGE_WIDTH = 960
DEFAULT_GEMMA4_PATCH_SIZE = 16
DEFAULT_GEMMA4_OUTPUT_LENGTH = 280


def _decoder_block_value(config: Config) -> str:
  return getattr(config.decoder_block, "value", config.decoder_block)


def _resolve_image_hw(config: Config) -> tuple[int, int]:
  """Returns image height/width for Gemma4 vision tower."""
  image_size = getattr(config, "image_size_for_vit", None)
  if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
    return int(image_size[0]), int(image_size[1])
  if isinstance(image_size, int) and image_size > 0:
    # Megatext base config defaults to a square ViT image size. If this is
    # unchanged for Gemma4, fall back to Gemma4's rectangular dimensions.
    if _decoder_block_value(config) == DecoderBlockType.GEMMA4.value and image_size == 896:
      return DEFAULT_GEMMA4_IMAGE_HEIGHT, DEFAULT_GEMMA4_IMAGE_WIDTH
    return image_size, image_size
  return DEFAULT_GEMMA4_IMAGE_HEIGHT, DEFAULT_GEMMA4_IMAGE_WIDTH


def _resolve_patch_size(config: Config) -> int:
  """Returns Gemma4 patch size with sensible fallback from Megatext defaults."""
  patch_size = getattr(config, "patch_size_for_vit", DEFAULT_GEMMA4_PATCH_SIZE)
  if isinstance(patch_size, int) and patch_size > 0:
    if _decoder_block_value(config) == DecoderBlockType.GEMMA4.value and patch_size == 14:
      return DEFAULT_GEMMA4_PATCH_SIZE
    return patch_size
  return DEFAULT_GEMMA4_PATCH_SIZE


def _resolve_vision_output_length(config: Config, image_hw: tuple[int, int], patch_size: int) -> int:
  """Returns requested number of soft visual tokens after pooling."""
  output_length = getattr(config, "vision_output_length", -1)
  if isinstance(output_length, int) and output_length > 0:
    return output_length

  image_height, image_width = image_hw
  num_patches = (image_height // patch_size) * (image_width // patch_size)
  if num_patches % 9 == 0:
    return num_patches // 9
  return DEFAULT_GEMMA4_OUTPUT_LENGTH


def factorized_posemb(posemb: jax.Array, positions_xy: jax.Array, precision) -> jax.Array:
  """Computes factorized position embedding from (x, y) coordinates."""
  one_hot = jax.nn.one_hot(positions_xy, posemb.shape[0], dtype=posemb.dtype)
  nan = jnp.logical_not(one_hot.any(axis=-1, keepdims=True))
  nan = jnp.logical_and(nan, positions_xy[..., None] != -1)
  pos_oh = jnp.where(nan, jnp.nan, one_hot)
  pe_seq = jnp.einsum("...is,sid->i...d", pos_oh, posemb, precision=precision).astype(posemb.dtype)
  return jnp.sum(pe_seq, axis=0)


def patchify(images: jax.Array, patch_size: int) -> tuple[jax.Array, jax.Array]:
  """Patchifies images and returns patches and (x, y) coordinates."""
  *batch_dims, height, width, channels = images.shape

  reshaped = jax.lax.reshape(
      images,
      tuple(batch_dims) + (height // patch_size, patch_size, width // patch_size, patch_size, channels),
  )
  transposed = jnp.transpose(
      reshaped,
      axes=tuple(range(len(batch_dims))) + (len(batch_dims), len(batch_dims) + 2, len(batch_dims) + 1, len(batch_dims) + 3, len(batch_dims) + 4),
  )
  patches = jax.lax.reshape(
      transposed,
      tuple(batch_dims) + ((height // patch_size) * (width // patch_size), patch_size * patch_size * channels),
  )

  xy = jnp.meshgrid(jnp.arange(width // patch_size), jnp.arange(height // patch_size))
  positions_xy = jnp.stack(xy, axis=-1)
  positions_xy = jnp.reshape(positions_xy, (-1, 2))
  return patches, jnp.broadcast_to(positions_xy, tuple(batch_dims) + positions_xy.shape)


class VisionEntry(nnx.Module):
  """The Gemma4 vision entry layer."""

  def __init__(
      self,
      d_model: int,
      patch_size: int,
      pos_emb_shape_yx: tuple[int, int],
      normalize_input_range: bool = False,
      *,
      rngs: nnx.Rngs,
      dtype,
      weight_dtype,
      matmul_precision,
  ):
    self.d_model = d_model
    self.patch_size = patch_size
    self.pos_emb_shape_yx = pos_emb_shape_yx
    self.normalize_input_range = normalize_input_range
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.matmul_precision = matmul_precision

    self.input_projection = linears.DenseGeneral(
        in_features_shape=self.patch_size * self.patch_size * 3,
        out_features_shape=self.d_model,
        use_bias=False,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        matmul_precision=self.matmul_precision,
        axis=-1,
        rngs=rngs,
    )

    if self.pos_emb_shape_yx[-1] != 2:
      raise ValueError(f"{self.pos_emb_shape_yx=} must have final dimension 2.")
    pos_emb_init = nnx.initializers.normal(stddev=0.02)
    self.pos_emb_param = nnx.Param(
        pos_emb_init(
            rngs.params(),
            (self.pos_emb_shape_yx[0], self.pos_emb_shape_yx[1], self.d_model),
            jnp.float32,
        )
    )

  def __call__(
      self,
      images_or_patches: jax.Array,
      positions_xy: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    """Projects image patches and adds factorized positional embeddings."""
    if positions_xy is None:
      patches, positions_xy = patchify(images_or_patches, self.patch_size)
    else:
      patches = images_or_patches
      if patches.ndim != 3:
        raise ValueError(f"Expected patch input of rank 3, got rank {patches.ndim}.")
      if positions_xy.shape[0] == patches.shape[0]:
        pass
      elif positions_xy.ndim == 2:
        positions_xy = jnp.broadcast_to(positions_xy, (patches.shape[0],) + positions_xy.shape)
      else:
        raise ValueError(f"Unexpected positions shape: {positions_xy.shape}.")

    if self.normalize_input_range:
      patches = 2 * (patches - 0.5)

    x = self.input_projection(patches)
    pos_embed = factorized_posemb(cast(jax.Array, self.pos_emb_param.value), positions_xy, self.matmul_precision).astype(
        x.dtype
    )
    return x + pos_embed, positions_xy


def apply_multidimensional_rope(
    inputs: jax.Array,
    positions: jax.Array,
    *,
    base_frequency: int,
    rotary_fraction: float | None = None,
    scale_factor: float = 1.0,
) -> jax.Array:
  """Applies multidimensional RoPE. Based on Gemma4 implementation."""

  def _apply_rope(x_in: jax.Array, pos: jax.Array, base_freq: int, scale: float) -> jax.Array:
    del scale
    dim = x_in.shape[-1]
    half_dim = dim // 2
    fraction = 2 * jnp.arange(0, half_dim) / dim
    timescale = base_freq**fraction

    reshaped_pos = pos[..., jnp.newaxis, jnp.newaxis]
    sinusoid_inp = reshaped_pos / timescale

    sin_half = jnp.sin(sinusoid_inp).astype(x_in.dtype)
    cos_half = jnp.cos(sinusoid_inp).astype(x_in.dtype)

    sin = jnp.concatenate([sin_half, sin_half], axis=-1)
    cos = jnp.concatenate([cos_half, cos_half], axis=-1)

    x1, x2 = jnp.split(x_in, 2, axis=-1)
    rotated_x = jnp.concatenate((-x2, x1), axis=-1)
    return (x_in * cos) + (rotated_x * sin)

  if positions.ndim + 2 == inputs.ndim:
    if rotary_fraction is None or rotary_fraction == 1.0:
      return _apply_rope(inputs, positions, base_frequency, scale_factor)
    dim_to_rope = int(rotary_fraction * inputs.shape[-1])
    if dim_to_rope == inputs.shape[-1]:
      return _apply_rope(inputs, positions, base_frequency, scale_factor)
    if dim_to_rope == 0:
      return inputs
    x1 = inputs[..., :dim_to_rope]
    x2 = inputs[..., dim_to_rope:]
    x1 = _apply_rope(x1, positions, base_frequency, scale_factor)
    return jnp.concatenate([x1, x2], axis=-1)

  ndim = positions.shape[-1]
  num_input_channels = inputs.shape[-1]
  num_rotated_channels = num_input_channels
  if rotary_fraction is not None:
    num_rotated_channels = int(round(num_rotated_channels * rotary_fraction))
  num_rotated_channels_per_dim = 2 * (num_rotated_channels // (2 * ndim))

  if num_rotated_channels_per_dim <= 0:
    raise ValueError(f"Requirement not satisfied: 2 * {ndim=} <= {num_input_channels=}.")

  split_points = [(k + 1) * num_rotated_channels_per_dim for k in range(ndim)]
  if rotary_fraction is None:
    split_points = split_points[:-1]
  x_parts = jnp.split(inputs, split_points, axis=-1)
  y_parts = [
      _apply_rope(
          x_parts[k],
          positions[..., k],
          base_frequency,
          scale_factor,
      )
      for k in range(ndim)
  ]
  if rotary_fraction is not None:
    y_parts.append(x_parts[-1])
  return jnp.concatenate(y_parts, axis=-1)


def avg_pool_by_positions(
    x: jax.Array,
    *,
    positions_xy: jax.Array,
    length: int,
    precision,
) -> tuple[jax.Array, jax.Array]:
  """Performs 2D spatial pooling according to patch positions."""
  k = max(1, int((x.shape[1] // length) ** 0.5))
  if k * k * length != x.shape[1]:
    raise ValueError(f"Cannot pool {x.shape=} to {length=}.")

  max_x = positions_xy[..., 0].max(axis=-1, keepdims=True) + 1
  kernel_idxs = jnp.floor_divide(positions_xy, k)
  flat_kernel_idx = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
  weights = jax.nn.one_hot(flat_kernel_idx, length) / k**2
  output = jnp.einsum("bLl,bLd->bld", weights, x, precision=precision)
  mask = jnp.logical_not((weights == 0).all(axis=1))
  return output, mask


class VisionExit(nnx.Module):
  """Vision exit layer with scaling and optional spatial pooling."""

  def __init__(self, d_model: int, output_length: int | tuple[int, ...], *, rngs: nnx.Rngs, precision):
    del rngs
    self.d_model = d_model
    self.output_length = output_length
    self.precision = precision

  def _maybe_downsample(
      self,
      x: jax.Array,
      *,
      positions_xy: jax.Array | None = None,
      length: int,
  ) -> tuple[jax.Array, jax.Array | None]:
    """Downsamples the vision features if required by the output length."""
    cur_length = x.shape[1]
    positions_pad_value = -1

    if cur_length == length:
      if positions_xy is None:
        mask = jnp.ones(x.shape[:-1], dtype=jnp.bool_)
      else:
        mask = jnp.logical_not((positions_xy == positions_pad_value).all(axis=-1))
      return x, mask

    if positions_xy is not None:
      return avg_pool_by_positions(x, positions_xy=positions_xy, length=length, precision=self.precision)

    cur_width = int(cur_length**0.5)
    if cur_width**2 != cur_length:
      raise ValueError(f"x.shape[1]={cur_length} must be a perfect square.")
    output_width = int(length**0.5)
    if output_width**2 != length:
      raise ValueError(f"{length=} must be a perfect square.")
    if cur_width % output_width != 0:
      raise ValueError(f"{cur_width=} must be divisible by {output_width=}.")

    x_2d = x.reshape(x.shape[0], cur_width, cur_width, x.shape[-1])
    window = cur_width // output_width
    window_shape = (window, window)
    x_2d = nnx.avg_pool(x_2d, window_shape=window_shape, strides=window_shape)
    x_pooled = x_2d.reshape(x.shape[0], length, x.shape[-1])
    mask = jnp.ones(x_pooled.shape[:-1], dtype=jnp.bool_)
    return x_pooled, mask

  def _single_call(
      self,
      x: jax.Array,
      *,
      positions_xy: jax.Array | None = None,
      length: int,
  ) -> tuple[jax.Array, jax.Array | None]:
    x, mask = self._maybe_downsample(x, positions_xy=positions_xy, length=length)
    x = x * jnp.sqrt(self.d_model)
    return x, mask

  def __call__(
      self,
      x: jax.Array,
      *,
      positions_xy: jax.Array | None = None,
      output_length_overrides: tuple[int, ...] | None = None,
  ) -> tuple[tuple[jax.Array, jax.Array | None], ...]:
    """Applies exit processing and returns one tuple per requested output length."""
    lens = (self.output_length,) if isinstance(self.output_length, int) else self.output_length
    if output_length_overrides is not None:
      lens = output_length_overrides
    return tuple(self._single_call(x, positions_xy=positions_xy, length=length) for length in lens)


class Gemma4VisionRotaryEmbedding(nnx.Module):
  """Rotary position embedding for Gemma4 vision."""

  def __init__(
      self,
      base_frequency: int,
      rotary_fraction: float | None = None,
      scale_factor: float = 1.0,
  ):
    self.base_frequency = base_frequency
    self.rotary_fraction = rotary_fraction
    self.scale_factor = scale_factor

  def __call__(self, inputs: jax.Array, positions: jax.Array) -> jax.Array:
    return apply_multidimensional_rope(
        inputs,
        positions,
        base_frequency=self.base_frequency,
        rotary_fraction=self.rotary_fraction,
        scale_factor=self.scale_factor,
    )


class Gemma4Attention(attentions.Attention):
  """Gemma4-specific attention module with multidimensional vision RoPE."""

  def init_rotary_embedding(self) -> Gemma4VisionRotaryEmbedding:
    return Gemma4VisionRotaryEmbedding(
        base_frequency=self.config.rope_theta_for_vit if hasattr(self.config, "rope_theta_for_vit") else 100,
        rotary_fraction=None,
    )


class Gemma4EncoderBlock(nnx.Module):
  """Single transformer encoder block (MHSA + MLP)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs

    self.pre_attention_norm = normalizations.RMSNorm(
        num_features=config.hidden_size_for_vit,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.post_attention_norm = normalizations.RMSNorm(
        num_features=config.hidden_size_for_vit,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    batch_size = config.per_device_batch_size
    image_height, image_width = _resolve_image_hw(config)
    patch_size = _resolve_patch_size(config)
    seq_len = (image_height // patch_size) * (image_width // patch_size)
    dummy_shape = (batch_size, seq_len, config.hidden_size_for_vit)

    self.attention = Gemma4Attention(
        config=config,
        num_query_heads=config.num_attention_heads_for_vit,
        num_kv_heads=config.num_attention_heads_for_vit,
        head_dim=config.hidden_size_for_vit // config.num_attention_heads_for_vit,
        max_target_length=seq_len,
        mesh=mesh,
        attention_kernel="dot_product",
        inputs_q_shape=dummy_shape,
        inputs_kv_shape=dummy_shape,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        float32_qk_product=config.float32_qk_product,
        float32_logits=config.float32_logits,
        dropout_rate=config.dropout_rate,
        attention_type=AttentionType.FULL,
        use_qk_norm=True,
        use_v_norm=True,
        query_pre_attn_scalar=1.0,
        is_vision=True,
        rngs=self.rngs,
    )

    self.pre_ffw_norm = normalizations.RMSNorm(
        num_features=config.hidden_size_for_vit,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.post_ffw_norm = normalizations.RMSNorm(
        num_features=config.hidden_size_for_vit,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    self.mlp = linears.MlpBlock(
        config=config,
        mesh=mesh,
        in_features=config.hidden_size_for_vit,
        intermediate_dim=config.intermediate_size_for_vit,
        activations=("gelu", "linear"),
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        intermediate_dropout_rate=config.dropout_rate,
        rngs=self.rngs,
    )

  def __call__(self, x: jax.Array, positions: jax.Array | None = None, deterministic: bool = False) -> jax.Array:
    x_normed = self.pre_attention_norm(x)
    x_attn, _ = self.attention(x_normed, x_normed, inputs_positions=positions, deterministic=deterministic)
    x_attn = self.post_attention_norm(x_attn)
    x_after_attn = x_attn + x

    x_ffw_normed = self.pre_ffw_norm(x_after_attn)
    x_ffw = self.mlp(x_ffw_normed, deterministic=deterministic)
    x_ffw = self.post_ffw_norm(x_ffw)
    return x_ffw + x_after_attn


class Gemma4VisionEncoderLayer(nnx.Module):
  """Gemma4 vision encoder layer."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs

    self.vision_entry = VisionEntry(
        d_model=config.hidden_size_for_vit,
        patch_size=_resolve_patch_size(config),
        pos_emb_shape_yx=(config.num_position_embeddings_for_vit, 2),
        normalize_input_range=True,
        rngs=self.rngs,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
    )

    for i in range(config.num_hidden_layers_for_vit):
      setattr(self, f"layer_{i}", Gemma4EncoderBlock(config, mesh, rngs=self.rngs))

    image_hw = _resolve_image_hw(config)
    vision_output_length = _resolve_vision_output_length(config, image_hw, _resolve_patch_size(config))
    self.vision_exit = VisionExit(
        d_model=config.hidden_size_for_vit,
        output_length=vision_output_length,
        rngs=self.rngs,
        precision=config.matmul_precision,
    )
    self.std_bias = nnx.Param(
        nnx.initializers.zeros(self.rngs.params(), (config.hidden_size_for_vit,), config.weight_dtype),
        sharding=(None,),
    )
    self.std_scale = nnx.Param(
        nnx.initializers.ones(self.rngs.params(), (config.hidden_size_for_vit,), config.weight_dtype),
        sharding=(None,),
    )

  def __call__(self, inputs: jax.Array, deterministic: bool = False) -> jax.Array:
    """Applies the vision encoder layer."""
    if inputs.ndim == 4:
      inputs = jnp.expand_dims(inputs, 1)
    batch_size, num_images, height, width, channels = inputs.shape
    inputs_flat = jnp.reshape(inputs, (batch_size * num_images, height, width, channels))

    x, positions_xy = self.vision_entry(inputs_flat)
    for i in range(self.config.num_hidden_layers_for_vit):
      x = getattr(self, f"layer_{i}")(x, positions=positions_xy, deterministic=deterministic)

    vision_exit_results = self.vision_exit(x, positions_xy=positions_xy)
    embeddings, _ = vision_exit_results[0]
    embeddings = (embeddings - self.std_bias.value.astype(embeddings.dtype)) * self.std_scale.value.astype(
        embeddings.dtype
    )
    return jnp.reshape(embeddings, (batch_size, num_images, embeddings.shape[1], embeddings.shape[2]))


class Gemma4VisionProjector(nnx.Module):
  """Projects image embeddings to the text embedding space."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs

    self.norm = normalizations.RMSNorm(
        num_features=config.hidden_size_for_vit,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        with_scale=False,
        rngs=self.rngs,
    )
    self.projection = linears.DenseGeneral(
        in_features_shape=config.hidden_size_for_vit,
        out_features_shape=config.emb_dim,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_axes=("embed", "mlp"),
        rngs=self.rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    x_normed = self.norm(x)
    return self.projection(x_normed)


def gemma4_vision_encoder_as_linen(config: Config, mesh: Mesh) -> nn.Module:
  """Wraps the Gemma4 vision encoder as a Linen module."""
  return nnx_wrappers.to_linen(
      Gemma4VisionEncoderLayer,
      config=config,
      mesh=mesh,
      name="Gemma4VisionEncoderLayer_0",
      abstract_init=False,
      metadata_fn=initializers.variable_to_logically_partitioned,
  )
