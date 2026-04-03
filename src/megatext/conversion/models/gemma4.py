"""Gemma4 model_type: mapping + transforms + shapes."""
from __future__ import annotations

from typing import Any

import numpy as np

from megatext.conversion.utils import (
    ArchSpec,
    Mapping,
    TransformFn,
    _build_embedding_transforms,
    _global_keys,
    reshape_kernel,
    reshape_kernel_inv,
    simple_transpose,
    pad_vocab_size,
)


def _get(cfg: Any, name: str, default=None):
  if isinstance(cfg, dict):
    return cfg.get(name, default)
  return getattr(cfg, name, default)


def _text_cfg(hf_config: Any) -> Any:
  return _get(hf_config, "text_config", hf_config)


def _vision_cfg(hf_config: Any) -> Any | None:
  return _get(hf_config, "vision_config", None)


def _repeat_cycle_length(layer_types: list[str]) -> int:
  if not layer_types:
    return 6
  n = len(layer_types)
  for cycle in range(1, n + 1):
    if n % cycle == 0 and all(layer_types[i] == layer_types[i % cycle] for i in range(n)):
      return cycle
  return n


def _layer_types(cfg: Any) -> list[str]:
  layer_types = list(_get(cfg, "layer_types", []) or [])
  if layer_types:
    return layer_types
  n_layers = int(_get(cfg, "num_hidden_layers", 0))
  cycle = ["sliding_attention"] * 5 + ["full_attention"]
  if n_layers <= 0:
    return cycle
  repeats = (n_layers + len(cycle) - 1) // len(cycle)
  return (cycle * repeats)[:n_layers]


def _cycle_types(cfg: Any) -> list[str]:
  layer_types = _layer_types(cfg)
  cycle = _repeat_cycle_length(layer_types)
  return layer_types[:cycle]


def _is_global_layer(cfg: Any, layer_idx: int) -> bool:
  layer_types = _layer_types(cfg)
  if not layer_types:
    return layer_idx % 6 == 5
  return layer_types[layer_idx % len(layer_types)] in ("full_attention", "global")


def _share_kv_projections(cfg: Any) -> bool:
  return bool(_get(cfg, "attention_k_eq_v", False) or _get(cfg, "share_kv_projections", False))


def _uses_moe(cfg: Any) -> bool:
  if _get(cfg, "enable_moe_block", False):
    return True
  num_experts = _get(cfg, "num_experts", None)
  return num_experts is not None and int(num_experts) > 1


def _num_experts(cfg: Any) -> int:
  num_experts = _get(cfg, "num_experts", None)
  if num_experts is None:
    return 1
  return int(num_experts)


def _expert_intermediate_size(cfg: Any) -> int:
  return int(
      _get(
          cfg,
          "expert_intermediate_size",
          _get(cfg, "moe_intermediate_size", _get(cfg, "intermediate_size", 0)),
      )
  )


def _shared_intermediate_size(cfg: Any) -> int:
  return int(_get(cfg, "intermediate_size", 0))


def _global_num_kv_heads(cfg: Any) -> int:
  return int(_get(cfg, "num_global_key_value_heads", _get(cfg, "global_num_kv_heads", _get(cfg, "num_key_value_heads", 0))))


def _global_head_dim(cfg: Any) -> int:
  return int(_get(cfg, "global_head_dim", _get(cfg, "head_dim", 0)))


def _v_norm_with_scale(cfg: Any) -> bool:
  return bool(_get(cfg, "v_norm_with_scale", False))


def _dense_kernel_transform(to_hf: bool) -> TransformFn:
  return reshape_kernel_inv if to_hf else reshape_kernel


def _swap_pos_emb_axes(x: np.ndarray, shape: tuple) -> np.ndarray:
  del shape
  return np.transpose(x, (1, 0, 2))


def _build_vision_mapping(mapping: Mapping, vcfg: Any) -> None:
  nvis = int(_get(vcfg, "num_hidden_layers", 0))
  mapping.update(
      {
          "params-vision_encoder-Gemma4VisionEncoderLayer_0-vision_entry-input_projection-kernel": (
              "model.vision_tower.patch_embedder.input_proj.weight"
          ),
          "params-vision_encoder-Gemma4VisionEncoderLayer_0-vision_entry-pos_emb_param": (
              "model.vision_tower.patch_embedder.position_embedding_table"
          ),
          "params-vision_encoder-Gemma4VisionEncoderLayer_0-std_scale": "model.vision_tower.std_scale",
          "params-vision_encoder-Gemma4VisionEncoderLayer_0-std_bias": "model.vision_tower.std_bias",
          "params-vision_encoder-Gemma4VisionProjector_0-projection-kernel": (
              "model.embed_vision.embedding_projection.weight"
          ),
      }
  )
  for i in range(nvis):
    prefix = f"params-vision_encoder-Gemma4VisionEncoderLayer_0-layer_{i}"
    hf_prefix = f"model.vision_tower.encoder.layers.{i}"
    mapping.update(
        {
            f"{prefix}-attention-query-kernel": f"{hf_prefix}.self_attn.q_proj.linear.weight",
            f"{prefix}-attention-key-kernel": f"{hf_prefix}.self_attn.k_proj.linear.weight",
            f"{prefix}-attention-value-kernel": f"{hf_prefix}.self_attn.v_proj.linear.weight",
            f"{prefix}-attention-out-kernel": f"{hf_prefix}.self_attn.o_proj.linear.weight",
            f"{prefix}-attention-query_norm-scale": f"{hf_prefix}.self_attn.q_norm.weight",
            f"{prefix}-attention-key_norm-scale": f"{hf_prefix}.self_attn.k_norm.weight",
            f"{prefix}-pre_attention_norm-scale": f"{hf_prefix}.input_layernorm.weight",
            f"{prefix}-post_attention_norm-scale": f"{hf_prefix}.post_attention_layernorm.weight",
            f"{prefix}-pre_ffw_norm-scale": f"{hf_prefix}.pre_feedforward_layernorm.weight",
            f"{prefix}-post_ffw_norm-scale": f"{hf_prefix}.post_feedforward_layernorm.weight",
            f"{prefix}-mlp-wi_0-kernel": f"{hf_prefix}.mlp.gate_proj.linear.weight",
            f"{prefix}-mlp-wi_1-kernel": f"{hf_prefix}.mlp.up_proj.linear.weight",
            f"{prefix}-mlp-wo-kernel": f"{hf_prefix}.mlp.down_proj.linear.weight",
        }
    )


def _build_text_layer_mapping(mapping: Mapping, cfg: Any, *, prefix: str, scan_layers: bool) -> None:
  n_layers = int(_get(cfg, "num_hidden_layers", 0))
  cycle_types = _cycle_types(cfg)
  cycle = len(cycle_types)
  share_kv = _share_kv_projections(cfg)
  use_moe = _uses_moe(cfg)
  use_v_norm = _v_norm_with_scale(cfg)

  if scan_layers:
    if n_layers % cycle != 0:
      raise ValueError(f"Gemma4 scan requires num_hidden_layers divisible by cycle={cycle}, got {n_layers}.")
    layer_indices_groups = [list(range(block_idx, n_layers, cycle)) for block_idx in range(cycle)]
    prefixes = [f"params-decoder-layers-layers_{block_idx}" for block_idx in range(cycle)]
  else:
    layer_indices_groups = [[i] for i in range(n_layers)]
    prefixes = [f"params-decoder-layers_{i}" for i in range(n_layers)]

  for layer_indices, mt_prefix in zip(layer_indices_groups, prefixes, strict=True):
    layer_idx = layer_indices[0]
    is_global = _is_global_layer(cfg, layer_idx)
    hf_prefixes = [f"{prefix}.layers.{i}" for i in layer_indices]

    def _stacked_path(suffix: str):
      if scan_layers:
        return [f"{hf_prefix}.{suffix}" for hf_prefix in hf_prefixes]
      return f"{hf_prefixes[0]}.{suffix}"

    mapping.update(
        {
            f"{mt_prefix}-self_attention-query-kernel": _stacked_path("self_attn.q_proj.weight"),
            f"{mt_prefix}-self_attention-key-kernel": _stacked_path("self_attn.k_proj.weight"),
            f"{mt_prefix}-self_attention-out-kernel": _stacked_path("self_attn.o_proj.weight"),
            f"{mt_prefix}-self_attention-query_norm-scale": _stacked_path("self_attn.q_norm.weight"),
            f"{mt_prefix}-self_attention-key_norm-scale": _stacked_path("self_attn.k_norm.weight"),
            f"{mt_prefix}-pre_self_attention_norm-scale": _stacked_path("input_layernorm.weight"),
            f"{mt_prefix}-post_self_attention_norm-scale": _stacked_path("post_attention_layernorm.weight"),
            f"{mt_prefix}-pre_ffw_norm-scale": _stacked_path("pre_feedforward_layernorm.weight"),
            f"{mt_prefix}-post_ffw_norm-scale": _stacked_path("post_feedforward_layernorm.weight"),
            f"{mt_prefix}-layer_scalar": _stacked_path("layer_scalar"),
        }
    )
    if not (share_kv and is_global):
      mapping[f"{mt_prefix}-self_attention-value-kernel"] = _stacked_path("self_attn.v_proj.weight")
    if use_v_norm:
      mapping[f"{mt_prefix}-self_attention-value_norm-scale"] = _stacked_path("self_attn.v_norm.weight")

    if use_moe:
      mapping.update(
          {
              f"{mt_prefix}-mlp-pre_feedforward_layernorm_2-scale": _stacked_path("pre_feedforward_layernorm_2.weight"),
              f"{mt_prefix}-mlp-post_feedforward_layernorm_1-scale": _stacked_path("post_feedforward_layernorm_1.weight"),
              f"{mt_prefix}-mlp-post_feedforward_layernorm_2-scale": _stacked_path("post_feedforward_layernorm_2.weight"),
              f"{mt_prefix}-mlp-pre_forward_scale_2": _stacked_path("router.scale"),
              f"{mt_prefix}-mlp-moe_block-MoeBlock_0-gate-kernel": _stacked_path("router.proj.weight"),
              f"{mt_prefix}-mlp-moe_block-MoeBlock_0-per_expert_scale": _stacked_path("router.per_expert_scale"),
              f"{mt_prefix}-mlp-moe_block-shared_experts-wi_0-kernel": _stacked_path("mlp.gate_proj.weight"),
              f"{mt_prefix}-mlp-moe_block-shared_experts-wi_1-kernel": _stacked_path("mlp.up_proj.weight"),
              f"{mt_prefix}-mlp-moe_block-shared_experts-wo-kernel": _stacked_path("mlp.down_proj.weight"),
          }
      )
      mapping[
          (
              f"{mt_prefix}-mlp-moe_block-MoeBlock_0-wi_0",
              f"{mt_prefix}-mlp-moe_block-MoeBlock_0-wi_1",
          )
      ] = _stacked_path("experts.gate_up_proj")
      mapping[f"{mt_prefix}-mlp-moe_block-MoeBlock_0-wo"] = _stacked_path("experts.down_proj")
    else:
      mapping.update(
          {
              f"{mt_prefix}-mlp-wi_0-kernel": _stacked_path("mlp.gate_proj.weight"),
              f"{mt_prefix}-mlp-wi_1-kernel": _stacked_path("mlp.up_proj.weight"),
              f"{mt_prefix}-mlp-wo-kernel": _stacked_path("mlp.down_proj.weight"),
          }
      )


def build_gemma4(arch: ArchSpec, hf_config: Any, scan_layers: bool) -> Mapping:
  """Builds Hugging Face → Megatext key mapping for Gemma4 text and vision."""
  cfg = _text_cfg(hf_config)
  vcfg = _vision_cfg(hf_config)
  tie = bool(_get(cfg, "tie_word_embeddings", _get(hf_config, "tie_word_embeddings", False)))

  mapping: Mapping = {}
  mapping.update(_global_keys(arch.hf_prefix, tie))
  _build_text_layer_mapping(mapping, cfg, prefix=arch.hf_prefix, scan_layers=scan_layers)
  if arch.model_type == "gemma4" and vcfg is not None:
    _build_vision_mapping(mapping, vcfg)
  return mapping


def build_gemma4_transforms(
    arch: ArchSpec,
    hf_config_dict: dict[str, Any],
    *,
    to_hf: bool,
) -> dict[str, TransformFn]:
  """Builds tensor transforms for Gemma4 conversion."""
  text_cfg = hf_config_dict.get("text_config", hf_config_dict)
  hidden_size = text_cfg.get("hidden_size", 0)
  vocab_size = text_cfg.get("vocab_size", 0)

  transforms: dict[str, TransformFn] = {}
  dense_kernel = _dense_kernel_transform(to_hf)

  for suffix in [
      "self_attention-query-kernel",
      "self_attention-key-kernel",
      "self_attention-value-kernel",
      "self_attention-out-kernel",
      "mlp-wi_0-kernel",
      "mlp-wi_1-kernel",
      "mlp-wo-kernel",
      "mlp-moe_block-shared_experts-wi_0-kernel",
      "mlp-moe_block-shared_experts-wi_1-kernel",
      "mlp-moe_block-shared_experts-wo-kernel",
      "attention-query-kernel",
      "attention-key-kernel",
      "attention-value-kernel",
      "attention-out-kernel",
      "vision_entry-input_projection-kernel",
      "Gemma4VisionProjector_0-projection-kernel",
  ]:
    transforms[suffix] = dense_kernel

  for suffix in [
      "mlp-moe_block-MoeBlock_0-gate-kernel",
      "mlp-moe_block-MoeBlock_0-wi_0",
      "mlp-moe_block-MoeBlock_0-wi_1",
      "mlp-moe_block-MoeBlock_0-wo",
  ]:
    transforms[suffix] = simple_transpose

  transforms["vision_entry-pos_emb_param"] = _swap_pos_emb_axes
  transforms.update(_build_embedding_transforms(arch, hidden_size, vocab_size, to_hf))
  return transforms


def compute_gemma4_shapes(
    hf_config: Any,
    arch: ArchSpec,
    scan_layers: bool,
    *,
    tie_word_embeddings: bool = False,
) -> dict[str, tuple]:
  """Computes Megatext parameter shapes for Gemma4 checkpoints."""
  cfg = _text_cfg(hf_config)
  vcfg = _vision_cfg(hf_config)

  emb = int(_get(cfg, "hidden_size", 0))
  mlp = int(_get(cfg, "intermediate_size", 0))
  nq = int(_get(cfg, "num_attention_heads", 0))
  nkv = int(_get(cfg, "num_key_value_heads", 0))
  hd = int(_get(cfg, "head_dim", emb // nq if nq else 0))
  global_nkv = _global_num_kv_heads(cfg)
  global_hd = _global_head_dim(cfg)
  vocab = int(_get(cfg, "vocab_size", 0))
  n_layers = int(_get(cfg, "num_hidden_layers", 0))
  padded_vocab = pad_vocab_size(vocab)
  use_moe = _uses_moe(cfg)
  n_experts = _num_experts(cfg)
  expert_mlp = _expert_intermediate_size(cfg)
  shared_mlp = _shared_intermediate_size(cfg)
  use_v_norm = _v_norm_with_scale(cfg)
  share_kv = _share_kv_projections(cfg)

  shapes: dict[str, tuple] = {
      "params-token_embedder-embedding": (padded_vocab, emb),
      "params-decoder-decoder_norm-scale": (emb,),
  }
  if not tie_word_embeddings:
    shapes["params-decoder-logits_dense-kernel"] = (emb, padded_vocab)

  cycle_types = _cycle_types(cfg)
  cycle = len(cycle_types)
  if scan_layers:
    if n_layers % cycle != 0:
      raise ValueError(f"Gemma4 scan requires num_hidden_layers divisible by cycle={cycle}, got {n_layers}.")
    layer_indices_groups = [list(range(block_idx, n_layers, cycle)) for block_idx in range(cycle)]
    prefixes = [f"params-decoder-layers-layers_{block_idx}" for block_idx in range(cycle)]
  else:
    layer_indices_groups = [[i] for i in range(n_layers)]
    prefixes = [f"params-decoder-layers_{i}" for i in range(n_layers)]

  for layer_indices, prefix in zip(layer_indices_groups, prefixes, strict=True):
    layer_idx = layer_indices[0]
    stacked = len(layer_indices)
    is_global = _is_global_layer(cfg, layer_idx)
    q_hd = global_hd if is_global else hd
    kv_hd = global_hd if is_global else hd
    nkv_this = global_nkv if is_global else nkv
    lp = (stacked,) if scan_layers else ()

    shapes[f"{prefix}-self_attention-query-kernel"] = (*lp, emb, nq, q_hd)
    shapes[f"{prefix}-self_attention-key-kernel"] = (*lp, emb, nkv_this, kv_hd)
    if not (share_kv and is_global):
      shapes[f"{prefix}-self_attention-value-kernel"] = (*lp, emb, nkv_this, kv_hd)
    shapes[f"{prefix}-self_attention-out-kernel"] = (*lp, nq, q_hd, emb)
    shapes[f"{prefix}-self_attention-query_norm-scale"] = (*lp, q_hd)
    shapes[f"{prefix}-self_attention-key_norm-scale"] = (*lp, kv_hd)
    if use_v_norm:
      shapes[f"{prefix}-self_attention-value_norm-scale"] = (*lp, kv_hd)
    shapes[f"{prefix}-pre_self_attention_norm-scale"] = (*lp, emb)
    shapes[f"{prefix}-post_self_attention_norm-scale"] = (*lp, emb)
    shapes[f"{prefix}-pre_ffw_norm-scale"] = (*lp, emb)
    shapes[f"{prefix}-post_ffw_norm-scale"] = (*lp, emb)
    shapes[f"{prefix}-layer_scalar"] = (*lp, 1)

    if use_moe:
      shapes[f"{prefix}-mlp-pre_feedforward_layernorm_2-scale"] = (*lp, emb)
      shapes[f"{prefix}-mlp-post_feedforward_layernorm_1-scale"] = (*lp, emb)
      shapes[f"{prefix}-mlp-post_feedforward_layernorm_2-scale"] = (*lp, emb)
      shapes[f"{prefix}-mlp-pre_forward_scale_2"] = (*lp, emb)
      shapes[f"{prefix}-mlp-moe_block-MoeBlock_0-gate-kernel"] = (*lp, emb, n_experts)
      shapes[f"{prefix}-mlp-moe_block-MoeBlock_0-per_expert_scale"] = (*lp, n_experts)
      shapes[f"{prefix}-mlp-moe_block-MoeBlock_0-wi_0"] = (*lp, n_experts, emb, expert_mlp)
      shapes[f"{prefix}-mlp-moe_block-MoeBlock_0-wi_1"] = (*lp, n_experts, emb, expert_mlp)
      shapes[f"{prefix}-mlp-moe_block-MoeBlock_0-wo"] = (*lp, n_experts, expert_mlp, emb)
      shapes[f"{prefix}-mlp-moe_block-shared_experts-wi_0-kernel"] = (*lp, emb, shared_mlp)
      shapes[f"{prefix}-mlp-moe_block-shared_experts-wi_1-kernel"] = (*lp, emb, shared_mlp)
      shapes[f"{prefix}-mlp-moe_block-shared_experts-wo-kernel"] = (*lp, shared_mlp, emb)
    else:
      shapes[f"{prefix}-mlp-wi_0-kernel"] = (*lp, emb, mlp)
      shapes[f"{prefix}-mlp-wi_1-kernel"] = (*lp, emb, mlp)
      shapes[f"{prefix}-mlp-wo-kernel"] = (*lp, mlp, emb)

  if arch.model_type == "gemma4" and vcfg is not None:
    hidden_vit = int(_get(vcfg, "hidden_size", 0))
    mlp_vit = int(_get(vcfg, "intermediate_size", 0))
    num_heads_vit = int(_get(vcfg, "num_attention_heads", 0))
    head_dim_vit = int(_get(vcfg, "head_dim", hidden_vit // num_heads_vit if num_heads_vit else 0))
    num_pos_vit = int(_get(vcfg, "position_embedding_size", _get(vcfg, "num_position_embeddings_for_vit", 0)))
    patch_size = int(_get(vcfg, "patch_size", 0))
    num_channels = int(_get(hf_config, "num_channels_for_vit", 3))
    nvis = int(_get(vcfg, "num_hidden_layers", 0))

    shapes["params-vision_encoder-Gemma4VisionEncoderLayer_0-vision_entry-input_projection-kernel"] = (
        patch_size * patch_size * num_channels,
        hidden_vit,
    )
    shapes["params-vision_encoder-Gemma4VisionEncoderLayer_0-vision_entry-pos_emb_param"] = (
        num_pos_vit,
        2,
        hidden_vit,
    )
    shapes["params-vision_encoder-Gemma4VisionEncoderLayer_0-std_scale"] = (hidden_vit,)
    shapes["params-vision_encoder-Gemma4VisionEncoderLayer_0-std_bias"] = (hidden_vit,)
    shapes["params-vision_encoder-Gemma4VisionProjector_0-projection-kernel"] = (
        hidden_vit,
        emb,
    )

    for i in range(nvis):
      prefix = f"params-vision_encoder-Gemma4VisionEncoderLayer_0-layer_{i}"
      shapes[f"{prefix}-attention-query-kernel"] = (hidden_vit, num_heads_vit, head_dim_vit)
      shapes[f"{prefix}-attention-key-kernel"] = (hidden_vit, num_heads_vit, head_dim_vit)
      shapes[f"{prefix}-attention-value-kernel"] = (hidden_vit, num_heads_vit, head_dim_vit)
      shapes[f"{prefix}-attention-out-kernel"] = (num_heads_vit, head_dim_vit, hidden_vit)
      shapes[f"{prefix}-attention-query_norm-scale"] = (head_dim_vit,)
      shapes[f"{prefix}-attention-key_norm-scale"] = (head_dim_vit,)
      shapes[f"{prefix}-pre_attention_norm-scale"] = (hidden_vit,)
      shapes[f"{prefix}-post_attention_norm-scale"] = (hidden_vit,)
      shapes[f"{prefix}-pre_ffw_norm-scale"] = (hidden_vit,)
      shapes[f"{prefix}-post_ffw_norm-scale"] = (hidden_vit,)
      shapes[f"{prefix}-mlp-wi_0-kernel"] = (hidden_vit, mlp_vit)
      shapes[f"{prefix}-mlp-wi_1-kernel"] = (hidden_vit, mlp_vit)
      shapes[f"{prefix}-mlp-wo-kernel"] = (mlp_vit, hidden_vit)

  return shapes
