# Copyright 2023-2025 Google LLC
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

"""TFLOP calculation utilities for training and prefill."""

import jax

from megatext.common.common_types import DecoderBlockType
from megatext.utils import logging as max_logging


def calculate_tokens_training_per_device(config):
  """Calculate training Tokens per device"""
  return config.max_target_length * config.per_device_batch_size * config.gradient_accumulation_steps


def calculate_gemma2_tflops_training_per_device(config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops):
  """
  Calculate training TFLOP for Gemma2 as in Gemma2 we combine [local_attention, global_attention] into one decoder
  layer and we use sliding window attention in local_attention
  """
  noncausal_attention_flops = (
      # global attention
      4 * config.per_device_batch_size * config.max_target_length**2 * config.num_query_heads * config.head_dim
      +
      # local attention
      4
      * config.per_device_batch_size
      * config.max_target_length
      * min(config.sliding_window_size, config.max_target_length)
      * config.num_query_heads
      * config.head_dim
  )
  causal_attention_flops = noncausal_attention_flops / 2
  attention_tflops = causal_attention_flops * config.num_decoder_layers * 3 / 10**12

  # multiply num_decoder_layers by 2 because we combine [local_attention, global_attention] into one decoder layer
  learnable_weight_tflops = (
      ((total_ffn_flops + qkv_flops + projection_flops) * config.num_decoder_layers * 2 + embedding_flops) * 3 / 10**12
  )

  return attention_tflops, learnable_weight_tflops


def calculate_mixed_attention_model_tflops_training_per_device(
    config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops, attention_pattern_length
):
  """
  Calculate training TFLOPs for models with a mixed attention pattern of local
  and global attention layers, like Gemma3 and GPT-OSS.
  """
  num_layers = config.num_decoder_layers

  num_global_layers = num_layers // attention_pattern_length
  num_local_layers = num_layers - num_global_layers

  # FLOPs for a single global attention layer (full attention)
  # Formula: 4 * batch_size * seq_len^2 * num_heads * head_dim
  global_attention_flops_per_layer = (
      4 * config.per_device_batch_size * config.max_target_length**2 * config.num_query_heads * config.head_dim
  )

  # FLOPs for a single local attention layer (sliding window)
  # Formula: 4 * batch_size * seq_len * window_size * num_heads * head_dim
  local_attention_flops_per_layer = (
      4
      * config.per_device_batch_size
      * config.max_target_length
      * min(config.sliding_window_size, config.max_target_length)
      * config.num_query_heads
      * config.head_dim
  )

  # Total attention FLOPs = (num_global_layers * FLOPs_per_global) + (num_local_layers * FLOPs_per_local)
  noncausal_attention_flops = (
      num_global_layers * global_attention_flops_per_layer + num_local_layers * local_attention_flops_per_layer
  )
  causal_attention_flops = noncausal_attention_flops / 2

  # Convert to TFLOPs and multiply by 3 for fwd/bwd pass
  attention_tflops = causal_attention_flops * 3 / 10**12

  # Learnable weights (FFN, QKV, Projections) are present in every layer.
  learnable_weight_tflops = ((total_ffn_flops + qkv_flops + projection_flops) * num_layers + embedding_flops) * 3 / 10**12

  return attention_tflops, learnable_weight_tflops


def _calculate_chunked_attention_flops_per_layer(config, seq_len, chunk_size):
  """Calculates the non-causal FLOPs for a single layer of chunked attention."""
  num_chunks = seq_len // chunk_size
  rem_chunk_size = seq_len % chunk_size
  # The complexity of chunked attention is the sum of squares of chunk lengths.
  chunked_complexity = (num_chunks * chunk_size**2) + (rem_chunk_size**2)
  # The formula for non-causal attention FLOPs is 4 * B * complexity * H * D,
  # where B=batch_size, H=num_heads, D=head_dim.
  return 4 * config.per_device_batch_size * chunked_complexity * config.num_query_heads * config.head_dim


def calculate_llama4_attention_tflops(config):
  """
  Calculates attention-only training TFLOPs for Llama4's specific architecture,
  which has an alternating pattern of global and chunked attention layers.
  """
  num_layers = config.num_decoder_layers
  seq_len = config.max_target_length
  chunk_size = config.chunk_attn_window_size

  # Determine number of global vs. chunked layers based on the NoPE interval.
  # A "NoPE" layer uses global attention.
  num_global_layers = num_layers // config.nope_layer_interval
  num_chunked_layers = num_layers - num_global_layers

  # FLOPs for a single global attention layer (full attention, non-causal)
  global_attention_flops_per_layer = (
      4 * config.per_device_batch_size * seq_len**2 * config.num_query_heads * config.head_dim
  )

  # FLOPs for a single chunked attention layer (non-causal)
  chunked_attention_flops_per_layer = _calculate_chunked_attention_flops_per_layer(config, seq_len, chunk_size)

  # Total non-causal attention FLOPs is the sum of all global and all chunked layers
  noncausal_attention_flops = (num_global_layers * global_attention_flops_per_layer) + (
      num_chunked_layers * chunked_attention_flops_per_layer
  )

  # Apply causal mask and convert to TFLOPs (multiply by 3 for fwd/bwd pass)
  causal_attention_flops = noncausal_attention_flops / 2
  attention_tflops = causal_attention_flops * 3 / 10**12

  return attention_tflops


def calculate_indexer_mask_ratio(indexer_topk, max_target_length):
  """
  Calculates the sparse-to-dense ratio for Indexer TFLOPs.

  The indexer evaluates all previous tokens in a causal manner until it hits
  the Top-K limit.

  Visual Representation (T=8, K=4):
  Key (S) ->
  Q1 [X . . . . . . .]  <- 1 token scored
  Q2 [X X . . . . . .]  <- 2 tokens scored
  Q3 [X X X . . . . .]  <- 3 tokens scored
  Q4 [X X X X . . . .]  <- 4 tokens scored (K limit reached)
  Q5 [X X X . X . . .]  <- 4 tokens scored
  Q6 [X X . X . X . .]  <- 4 tokens scored
  Q7 [X . X X . . X .]  <- 4 tokens scored
  Q8 [X X . X . . . X]  <- 4 tokens scored

  For MFU calculation:

  Visual Representation (T=8, K=4):
  Key (S) ->
  Q1 [X . . . . . . .]  <- 1 token scored
  Q2 [X X . . . . . .]  <- 2 tokens scored
  Q3 [X X X . . . . .]  <- 3 tokens scored
  Q4 [X X X X . . . .]  <- 4 tokens scored (K limit reached)
  Q5 [X X X X . . . .]  <- 4 tokens scored
  Q6 [X X X X . . . .]  <- 4 tokens scored
  Q7 [X X X X . . . .]  <- 4 tokens scored
  Q8 [X X X X . . . .]  <- 4 tokens scored

  Mathematical Calculation:
  - Triangle (Phase 1: 1 to K): K^2 / 2
  - Rectangle (Phase 2: K+1 to T): (T - K) * K
  - Total Active Area = TK - K^2 / 2
  - Dense Area = T^2

  Ratio = (TK - 0.5*K^2) / T^2  =>  (K/T) - 0.5*(K/T)^2
  """

  T = float(max_target_length)
  K = float(indexer_topk)

  ratio = K / T
  mask_multiplier = ratio - (0.5 * ratio**2)
  return mask_multiplier


def calculate_indexer_tflops_per_device(config):
  """Calculates TFLOPs for the DeepSeek Lightning Indexer (handles causal reduction)."""
  batch_len = config.per_device_batch_size * config.max_target_length

  # 1. Calculate projections flops
  # Query: [batch, seq, q_lora_rank] @ [q_lora_rank, indexer_n_heads, indexer_head_dim]
  q_flops = 2 * batch_len * config.q_lora_rank * config.indexer_n_heads * config.indexer_head_dim
  # Key: [batch, seq, emb_dim] @ [emb_dim, indexer_head_dim]
  k_flops = 2 * batch_len * config.emb_dim * config.indexer_head_dim
  # Head weight: [batch, seq, emb_dim] @ [emb_dim, indexer_n_heads]
  head_weight_flops = 2 * batch_len * config.emb_dim * config.indexer_n_heads
  proj_flops = q_flops + k_flops + head_weight_flops

  # 2. Calculate index score flops
  # QK product [batch, seq, indexer_n_heads, indexer_head_dim] @ [batch, seq, indexer_head_dim]
  # --> [batch, seq, seq, indexer_n_heads]
  qk_product_flops = 2 * batch_len * config.max_target_length * config.indexer_n_heads * config.indexer_head_dim
  # Aggregate heads [batch, seq, seq, indexer_n_heads] @ [batch, seq, indexer_n_heads]
  head_reduction_flops = 2 * batch_len * config.max_target_length * config.indexer_n_heads
  # Apply causal mask: Divide by 2 to account for triangular interactions
  # The mask restricts the indexer's search space prior to Top-K filtering
  scoring_flops = (qk_product_flops + head_reduction_flops) / 2

  return proj_flops, scoring_flops


def calculate_mla_tflops_per_device(config):
  """Calculate Multi-Head Latent Attention TFLOP (handles causal reduction)"""
  batch_len = config.per_device_batch_size * config.max_target_length
  qk_head_dim_sum = config.qk_nope_head_dim + config.qk_rope_head_dim

  # 1. calculate mla query projection
  if config.q_lora_rank == 0:
    q_flops = 2 * batch_len * config.emb_dim * config.num_query_heads * qk_head_dim_sum
  else:
    # calculate query down and up flops
    q_flops = (
        2
        * batch_len
        * (config.emb_dim * config.q_lora_rank + config.q_lora_rank * config.num_query_heads * qk_head_dim_sum)
    )

  # 2. calculate mla kv projection
  kv_flops = (
      2
      * batch_len
      * (
          config.emb_dim * (config.kv_lora_rank + config.qk_rope_head_dim)
          + config.kv_lora_rank * config.num_query_heads * (config.qk_nope_head_dim + config.v_head_dim)
      )
  )
  qkv_flops = q_flops + kv_flops

  # 3. calculate attention
  if config.use_indexer and config.max_target_length > config.indexer_topk:
    # get indexer flops
    indexer_proj_flops, indexer_scoring_flops = calculate_indexer_tflops_per_device(config)
    qkv_flops += indexer_proj_flops

    # calculate the proportion of the T x T causal matrix that the Indexer actually explores
    # this follows the area: (TK - 0.5*K^2) / T^2 (T: max_target_length, K: indexer_topk)
    multiplier = calculate_indexer_mask_ratio(config.indexer_topk, config.max_target_length)
    attention_flops = (
        2
        * batch_len
        * config.max_target_length
        * config.num_query_heads
        * (qk_head_dim_sum + config.v_head_dim)
        * multiplier
    )
    attention_flops += indexer_scoring_flops
  else:
    # standard MLA & max_target_length <= indexer_topk in sparse indexer
    # in both cases, the indexer is bypassed as the causal mask remains the efficient representation
    attention_flops = (
        2 * batch_len * config.max_target_length * config.num_query_heads * (qk_head_dim_sum + config.v_head_dim)
    )
    attention_flops = attention_flops / 2
  projection_flops = 2 * batch_len * config.emb_dim * config.num_query_heads * config.v_head_dim
  return qkv_flops, attention_flops, projection_flops


def calculate_ffn_mamtul_tflops_per_device(config, mlp_dim):
  """Helper function to calculate matmul TFLOP in ffn based on MLP dimension.

  Applies to:
    - Dense FFN layers (mlp_dim = config.mlp_dim).
    - MoE FFN layers (mlp_dim = config.moe_mlp_dim),
      need to scale by shared_experts or num_experts_per_tok.
  """
  ffn1_flops = (
      2 * config.per_device_batch_size * config.max_target_length * mlp_dim * config.emb_dim * len(config.mlp_activations)
  )
  ffn2_flops = 2 * config.per_device_batch_size * config.max_target_length * mlp_dim * config.emb_dim
  return ffn1_flops + ffn2_flops


def calculate_routed_and_shared_ffn_tflops_per_device(config):
  """Helper function to calculate DeepSeek-style ffn TFLOP"""
  gate_flops = 2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.num_experts
  # Due to the mixed decoder layers, the flops is multiplied by num of layers for both dense and moe
  num_dense_layers, num_moe_layers = get_dense_moe_layers(config)
  dense_ffn_flops = calculate_ffn_mamtul_tflops_per_device(config, config.mlp_dim) * num_dense_layers
  shared_experts_flops = calculate_ffn_mamtul_tflops_per_device(config, config.moe_mlp_dim) * config.shared_experts
  routed_experts_flops = calculate_ffn_mamtul_tflops_per_device(config, config.moe_mlp_dim) * config.num_experts_per_tok
  moe_ffn_flops = (gate_flops + shared_experts_flops + routed_experts_flops) * num_moe_layers
  total_ffn_flops = dense_ffn_flops + moe_ffn_flops
  return total_ffn_flops


def get_dense_moe_layers(config):
  """Helper function to calculate number of dense and moe layers"""
  if config.decoder_block == DecoderBlockType.DEEPSEEK:
    num_dense_layers = config.first_num_dense_layers
    num_moe_layers = config.num_decoder_layers - config.first_num_dense_layers
    return num_dense_layers, num_moe_layers
  elif config.decoder_block == DecoderBlockType.LLAMA4:
    num_moe_layers = config.num_decoder_layers // config.interleave_moe_layer_step
    num_dense_layers = config.num_decoder_layers - num_moe_layers
  elif config.decoder_block == DecoderBlockType.QWEN3_NEXT:
    num_moe_layers = config.num_decoder_layers
    num_dense_layers = 0
  else:
    raise ValueError("Currently we only support DeepSeek, Llama4, and Qwen3-Next calculation.")

  return num_dense_layers, num_moe_layers


def calculate_gated_delta_net_flops_per_device(config):
  """
  - Calculates the FLOPs for a single Gated Delta Net (Linear Attention) layer.
  - Ref: Megatron calculation for the gated delta net:
    - https://github.com/NVIDIA/Megatron-LM/blob/8f1c2f8ae53b4e3f32c0ae7f397d8b38a675eaa2/megatron/training/training.py#L513
  - Core complexity is based on the recurrent state update view (4 ops * 2 FLOPs = 8).
  """
  B = config.per_device_batch_size
  S = config.max_target_length
  E = config.emb_dim

  H_k = config.gdn_num_key_heads
  H_v = config.gdn_num_value_heads
  D_k = config.gdn_key_head_dim
  D_v = config.gdn_value_head_dim
  K_conv = config.gdn_conv_kernel_dim

  K_dim = H_k * D_k
  V_dim = H_v * D_v

  # 1. Projections (Learnable Weights)
  # Represents: in_proj_qkvz (2*K + 2*V) + in_proj_ba (2*H_v)
  # We multiply by 2 for FMA (Multiply + Add)
  flops_qkvz_ba = 2 * B * S * E * (2 * K_dim + 2 * V_dim + 2 * H_v)

  # Represents: out_proj
  flops_out = 2 * B * S * E * V_dim

  flops_projections = flops_qkvz_ba + flops_out

  # 2. Convolution (Learnable Weights)
  # We multiply by 2 for FMA
  flops_conv = 2 * B * S * K_conv * (2 * K_dim + V_dim)

  # 3. Core Gated Delta Net
  # This counts 4 distinct O(D^2) operations in the recurrent update:
  #   KK^T, VK^T, S(a(I-bKK^T)), and SQ.
  # We multiply by 2 for FMA.
  # Total Core FLOPs = 2 (FMA) * 4 (Ops) * H * D^2 = 8 * H * D^2 per token.
  # We use D_k * D_v to generalize D^2 for potentially differing head dimensions.
  flops_core_per_token = H_v * (D_k * D_v) * 8
  flops_core = B * S * flops_core_per_token

  # Weights part: Projections + Conv
  gdn_weight_flops = flops_projections + flops_conv
  # Attention part: Core
  gdn_attn_flops = flops_core

  return gdn_weight_flops, gdn_attn_flops


def calculate_gemma3_vision_layers_tflops_per_device(config):
  """
  Estimate TFLOPs for Gemma3 vision encoder (ViT-style).
  Returns:
      total_tflops: Total TFLOPs (counts for fwd + bwd + optimizer)
      learnable_weight_tflops: TFLOPs from learnable weights (patch embedding, qkv, MLP, projections)
      attention_tflops: TFLOPs from attention multiplications
  """
  # Config values
  B = config.per_device_batch_size
  C = config.num_channels_for_vit
  H = W = config.image_size_for_vit  # Gemma3 default 896
  embed_dim = config.emb_dim  # text embedding dim after projection
  # Values below are hardcoded in Gemma3VisionEncoderLayer
  patch_size = 14
  hidden_dim = 1152
  intermediate_dim = 4304
  num_layers = 27
  vision_exit_pooling_window = 4

  # 1. Patch embedding (Conv2D)
  num_patches_h = H // patch_size
  num_patches_w = W // patch_size
  seq_len = num_patches_h * num_patches_w  # 64*64=4096
  patch_embed_flops = 2 * B * seq_len * (C * patch_size * patch_size) * hidden_dim

  # 2. gemma3.Encoder: num_layers * gemma3.Encoder1DBlock
  qkv_flops_per_layer = 3 * (2 * B * seq_len * hidden_dim * hidden_dim)
  attn_flops_per_layer = 4 * B * seq_len * seq_len * hidden_dim
  projection_flops_per_layer = 2 * B * seq_len * hidden_dim * hidden_dim  # projection after attention multiplication
  mlp_flops_per_layer = 2 * (2 * B * seq_len * hidden_dim * intermediate_dim)  # two fc layers
  total_attn_flops = attn_flops_per_layer * num_layers
  encoder_flops = (+qkv_flops_per_layer + projection_flops_per_layer + mlp_flops_per_layer) * num_layers

  # 4. VisionEmbedder
  seq_len_after_pooling = (num_patches_h // vision_exit_pooling_window) * (num_patches_w // vision_exit_pooling_window)
  vision_embedder_flops = 2 * B * seq_len_after_pooling * hidden_dim * embed_dim  # One linear projection

  # Learnable weights summation
  learnable_weight_flops = patch_embed_flops + encoder_flops + vision_embedder_flops

  if config.freeze_vision_encoder_params:
    learnable_weight_flops += 2 * vision_embedder_flops  # only projector is learnable, add fwd+optimizer
  else:
    learnable_weight_flops *= 3  # multiply by 3 for fwd + bwd + optimizer

  # Convert to TFLOPs
  learnable_weight_tflops = learnable_weight_flops / 1e12
  total_attn_tflops = total_attn_flops / 1e12
  total_tflops = learnable_weight_tflops + total_attn_tflops

  return total_tflops, learnable_weight_tflops, total_attn_tflops


def calculate_llama4_vision_layers_tflops_per_device(config):
  """
  Estimate TFLOPs for Llama4 vision encoder (ViT-style).
  Returns:
      total_tflops: Total TFLOPs (counts for fwd + bwd + optimizer)
      learnable_weight_tflops: TFLOPs from learnable weights (patch embedding, qkv, MLP, projections)
      attention_tflops: TFLOPs from attention multiplications
  """
  # Config values
  B = config.per_device_batch_size
  C = config.num_channels_for_vit
  H = W = config.tile_size_for_vit
  patch_size = config.patch_size_for_vit
  hidden_dim = config.hidden_size_for_vit
  intermediate_dim = config.intermediate_size_for_vit
  num_layers = config.num_hidden_layers_for_vit
  pixel_shuffle_fc1_out_dim = config.projector_input_dim_for_vit  # 4096
  pixel_shuffle_fc2_out_dim = config.projector_output_dim_for_vit  # 4096
  base_emb_dim = config.base_emb_dim
  pixel_shuffle_ratio = config.pixel_shuffle_ratio_for_vit  # 0.5
  num_patches = (H // patch_size) * (W // patch_size)  # 24*24 = 576
  pixel_shuffle_tokens = num_patches * pixel_shuffle_ratio**2  # 144

  # 1. Llama4UnfoldConvolution (flops by linear projection)
  # lax.conv_general_dilated_patches extracts patches through reshaping/indexing without flops
  # Each patch: C * patch_size * patch_size -> hidden_dim
  patch_embed_flops = 2 * B * num_patches * (C * patch_size * patch_size) * hidden_dim

  # 2. Llama4VisionEncoder: num_layers * (qkv + att_projection + mlp)
  seq_len = num_patches + 1  # +1 for class token, so 577
  qkv_flops_per_layer = 3 * (2 * B * seq_len * hidden_dim * hidden_dim)  # Q, K, V projections
  attn_flops_per_layer = 4 * B * seq_len * seq_len * hidden_dim  # Attention scores and weighted sum
  projection_flops_per_layer = 2 * B * seq_len * hidden_dim * hidden_dim  # projection after attention multiplication
  mlp_flops_per_layer = 2 * (2 * B * seq_len * hidden_dim * intermediate_dim)  # two fc layers
  total_attn_flops = attn_flops_per_layer * num_layers
  vision_encoder_flops = (+qkv_flops_per_layer + projection_flops_per_layer + mlp_flops_per_layer) * num_layers

  # 3. Llama4VisionPixelShuffleMLP
  # (B, 144, 5632) -> (B, 144, 4096) -> (B, 144, 4096)
  pixel_shuffle_fc1_flops = 2 * B * pixel_shuffle_tokens * intermediate_dim * pixel_shuffle_fc1_out_dim
  pixel_shuffle_fc2_flops = 2 * B * pixel_shuffle_tokens * pixel_shuffle_fc1_out_dim * pixel_shuffle_fc2_out_dim
  pixel_shuffle_total_flops = pixel_shuffle_fc1_flops + pixel_shuffle_fc2_flops

  # 4. Llama4MultiModalProjector: (B, 144, 5120) x (5120, base_emb_dim)
  projector_flops = 2 * B * pixel_shuffle_tokens * pixel_shuffle_fc1_out_dim * base_emb_dim

  # Learnable weights: all matmuls above
  learnable_weight_flops = patch_embed_flops + vision_encoder_flops + pixel_shuffle_total_flops + projector_flops

  if config.freeze_vision_encoder_params:
    learnable_weight_flops += 2 * projector_flops  # only projector is learnable, add fwd+optimizer
  else:
    learnable_weight_flops *= 3  # multiply by 3 for fwd + bwd + optimizer

  # Convert to TFLOPs
  learnable_weight_tflops = learnable_weight_flops / 1e12
  total_attn_tflops = total_attn_flops / 1e12
  total_tflops = learnable_weight_tflops + total_attn_tflops

  return total_tflops, learnable_weight_tflops, total_attn_tflops


def calculate_engram_tflops(config):
  """Calculate engram TFLOPs per device."""
  B = config.per_device_batch_size
  S = config.max_target_length
  G = config.mhc_expansion_rate  # Multi-manifold branches
  D = config.emb_dim  # Hidden dimension
  k = config.engram_kernel_size  # Conv window

  num_ngram_orders = config.engram_max_ngram_size - 1
  engram_dim = config.engram_num_heads * config.engram_head_dim * num_ngram_orders

  # 1. Key Projection
  key_flops = 2 * (B * S) * engram_dim * (G * D)
  # 2. Value Projection
  value_flops = 2 * (B * S) * engram_dim * D
  # 3. QK Attention
  attention_flops = 2 * (B * S) * G * D
  # 4. Short Convolution
  # Standard flops as 2 * kernel_size * input_channels * output_channels / feature_group_count
  # In Engram, the feature_group_count = input_channels = output_channels
  # Unlike global attention, convolution work is constant per token (not O(S^2)),
  # so we do not apply the 0.5 causal scaling factor.
  total_channels = G * D
  conv_flops = 2 * (B * S) * k * total_channels

  num_layers = len(config.engram_layers)
  # account for both the forward (1x) and backward (2x) passes
  learnable_tflops = num_layers * (key_flops + value_flops + conv_flops) * 3 / 1e12
  attention_tflops = num_layers * attention_flops * 3 / 1e12
  return learnable_tflops, attention_tflops


def calculate_vision_encoder_tflops(config):
  """Calculate vision encoder TFLOPs per prefill step per device."""
  if config.decoder_block == "gemma3":
    mm_total_tflops, mm_learnable_weight_tflops, mm_attention_tflops = calculate_gemma3_vision_layers_tflops_per_device(
        config
    )
  elif config.decoder_block == "llama4":
    mm_total_tflops, mm_learnable_weight_tflops, mm_attention_tflops = calculate_llama4_vision_layers_tflops_per_device(
        config
    )
  else:
    max_logging.log(
        f"Vision encoder TFLOPs calculation not implemented for model {config.model}, counting as 0 for now."
    )
    mm_total_tflops = mm_learnable_weight_tflops = mm_attention_tflops = 0

  return mm_total_tflops, mm_learnable_weight_tflops, mm_attention_tflops


def calculate_tflops_training_per_device(config, log=True):
  """Calculate training TFLOP"""
  # MLP flops
  if config.num_experts > 1:
    # calculation based on dropless implementation
    if config.decoder_block in (DecoderBlockType.DEEPSEEK, DecoderBlockType.LLAMA4, DecoderBlockType.QWEN3_NEXT):
      total_ffn_flops = calculate_routed_and_shared_ffn_tflops_per_device(config)
    else:
      gate_flops = 2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.num_experts
      total_ffn_flops = (
          gate_flops + calculate_ffn_mamtul_tflops_per_device(config, config.mlp_dim) * config.num_experts_per_tok
      )
  else:
    total_ffn_flops = calculate_ffn_mamtul_tflops_per_device(config, config.mlp_dim)

  # Attention flops
  if config.attention_type == "mla":
    qkv_flops, causal_attention_flops, projection_flops = calculate_mla_tflops_per_device(config)
  else:
    qkv_flops = (
        2
        * config.per_device_batch_size
        * config.max_target_length
        * config.emb_dim
        * (config.num_query_heads + 2 * config.num_kv_heads)
        * config.head_dim
    )
    noncausal_attention_flops = (
        4 * config.per_device_batch_size * config.max_target_length**2 * config.num_query_heads * config.head_dim
    )
    projection_flops = (
        2
        * config.per_device_batch_size
        * config.max_target_length
        * config.emb_dim
        * config.num_query_heads
        * config.head_dim
    )

    # Divide attention flops by 2 due to causal mask
    # References:
    # NVIDIA/Megatron-LM (2025 March): https://github.com/NVIDIA/Megatron-LM/blob/250b79415dcc4b660521273c87f15334c804eeae/megatron/training/training.py#L361-L362
    # NVIDIA/NeMo (2025 April): https://github.com/NVIDIA/NeMo/blob/ba4d6d116463de512ff0cfc14641aa6cf4577a42/nemo/utils/flops_formulas.py#L259-L272
    causal_attention_flops = noncausal_attention_flops / 2

  # Embedding flops
  embedding_flops = 2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.vocab_size

  # Combine flops with number of decoder layers
  if config.decoder_block == DecoderBlockType.GEMMA2:
    attention_tflops, learnable_weight_tflops = calculate_gemma2_tflops_training_per_device(
        config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops
    )
  elif config.decoder_block == DecoderBlockType.GEMMA3:
    attention_tflops, learnable_weight_tflops = calculate_mixed_attention_model_tflops_training_per_device(
        config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops, attention_pattern_length=6
    )
  elif config.decoder_block == DecoderBlockType.GPT_OSS:
    attention_tflops, learnable_weight_tflops = calculate_mixed_attention_model_tflops_training_per_device(
        config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops, attention_pattern_length=2
    )
  elif config.decoder_block == DecoderBlockType.QWEN3_SWA:
    attention_tflops, learnable_weight_tflops = calculate_mixed_attention_model_tflops_training_per_device(
        config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops,
        attention_pattern_length=config.inhomogeneous_layer_cycle_interval,
    )
  elif config.decoder_block == DecoderBlockType.LLAMA4:
    # Use the new helper to calculate attention TFLOPs correctly.
    attention_tflops = calculate_llama4_attention_tflops(config)
    # The learnable weight calculation remains the same as it correctly handles Llama4's MoE structure.
    learnable_weight_tflops = (
        (total_ffn_flops + (qkv_flops + projection_flops) * config.num_decoder_layers + embedding_flops) * 3 / 10**12
    )
  elif config.decoder_block == DecoderBlockType.DEEPSEEK:
    learnable_weight_tflops = (
        (total_ffn_flops + (qkv_flops + projection_flops) * config.num_decoder_layers + embedding_flops) * 3 / 10**12
    )
    attention_tflops = causal_attention_flops * config.num_decoder_layers * 3 / 10**12
  elif config.decoder_block == DecoderBlockType.QWEN3_NEXT:
    gdn_weight_flops_per_layer, gdn_attn_flops_per_layer = calculate_gated_delta_net_flops_per_device(config)
    cycle_interval = config.inhomogeneous_layer_cycle_interval
    num_full_attn_layers = config.num_decoder_layers // cycle_interval
    num_linear_attn_layers = config.num_decoder_layers - num_full_attn_layers

    # Weights TFLOPs:
    total_weights = (
        total_ffn_flops
        + embedding_flops
        + (qkv_flops + projection_flops) * num_full_attn_layers
        + gdn_weight_flops_per_layer * num_linear_attn_layers
    )
    learnable_weight_tflops = total_weights * 3 / 10**12

    # Attention TFLOPs:
    total_attn = (causal_attention_flops * num_full_attn_layers) + (gdn_attn_flops_per_layer * num_linear_attn_layers)
    attention_tflops = total_attn * 3 / 10**12
  else:
    # multiply by 3 for both feed forward and back propagation flops
    learnable_weight_tflops = (
        ((total_ffn_flops + qkv_flops + projection_flops) * config.num_decoder_layers + embedding_flops) * 3 / 10**12
    )
    attention_tflops = causal_attention_flops * config.num_decoder_layers * 3 / 10**12

  # Engram flops
  if config.engram_layers:
    engram_learnable_tflops, engram_attention_tflops = calculate_engram_tflops(config)
    learnable_weight_tflops += engram_learnable_tflops
    attention_tflops += engram_attention_tflops

  if config.use_multimodal:
    # Add vision layers TFLOPs for multimodal models
    mm_total_tflops, mm_learnable_weight_tflops, mm_attention_tflops = calculate_vision_encoder_tflops(config)
    if log:
      print(
          f"{config.model} vision layers per train step:\n",
          f"Total TFLOPs: {mm_total_tflops:.2f} \n",
          f"split as {100 * mm_learnable_weight_tflops/mm_total_tflops:.2f}% learnable weight flops",
          f"and {100 * mm_attention_tflops/mm_total_tflops:.2f}% attention flops;\n",
          f"learnable weight {mm_learnable_weight_tflops:.2f} TFLOPs, attention {mm_attention_tflops:.2f} TFLOPs",
      )
    learnable_weight_tflops += mm_learnable_weight_tflops
    attention_tflops += mm_attention_tflops

  learnable_weight_tflops = learnable_weight_tflops * config.gradient_accumulation_steps
  attention_tflops = attention_tflops * config.gradient_accumulation_steps

  total_tflops = learnable_weight_tflops + attention_tflops

  if log:
    print(
        "Per train step:\n",
        f"Total TFLOPs: {total_tflops:.2f} \n",
        f"split as {100 * learnable_weight_tflops/total_tflops:.2f}% learnable weight flops",
        f"and {100 * attention_tflops/total_tflops:.2f}% attention flops",
    )
  return total_tflops, learnable_weight_tflops, attention_tflops


# https://arxiv.org/pdf/2204.02311.pdf Appendix B
def calculate_prefill_tflops_per_device(num_model_parameters, prefill_length, config, log=True):
  """Calculate training TFLOP"""
  learnable_weight_tflops = 2 * num_model_parameters * prefill_length / jax.device_count() / 1e12
  noncausal_attention_flops = (
      4
      * config.num_query_heads
      * config.num_decoder_layers
      * config.head_dim
      * prefill_length**2
      / jax.device_count()
      / 1e12
  )
  causal_attention_tflops = noncausal_attention_flops / 2  # due to causality in attention
  total_tflops = learnable_weight_tflops + causal_attention_tflops

  if log:
    print(
        "Per prefill step per device: \n",
        f"\tTotal TFLOPs: {total_tflops:.2f} \n",
        f"\t\tLearnable weight TFLOPs: {learnable_weight_tflops:.2f} ",
        f"({100 * learnable_weight_tflops/total_tflops:.2f})% of Total\n",
        f"\t\tCausal attention TFLOPs: {causal_attention_tflops:.2f} ",
        f"({100 * causal_attention_tflops/total_tflops:.2f})% of Total",
    )
  return total_tflops, learnable_weight_tflops, causal_attention_tflops
