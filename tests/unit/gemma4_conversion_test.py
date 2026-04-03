"""Gemma4 conversion mapping/shape tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from megatext.conversion.convert import _infer_hf_shape_from_key, compute_megatext_shapes
from megatext.conversion.models import ARCH_SPECS
from megatext.conversion.models.gemma4 import build_gemma4, build_gemma4_transforms


def _ns(obj):
  if isinstance(obj, dict):
    return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
  if isinstance(obj, list):
    return [_ns(v) for v in obj]
  return obj


def _gemma4_config(*, moe: bool = False, vision: bool = True, num_layers: int = 12):
  cycle = ["sliding_attention"] * 5 + ["full_attention"]
  layer_types = (cycle * ((num_layers + len(cycle) - 1) // len(cycle)))[:num_layers]
  text_cfg = {
      "model_type": "gemma4_text",
      "tie_word_embeddings": True,
      "hidden_size": 128,
      "intermediate_size": 256,
      "expert_intermediate_size": 64,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      "num_global_key_value_heads": 1,
      "head_dim": 32,
      "global_head_dim": 64,
      "vocab_size": 1024,
      "num_hidden_layers": num_layers,
      "attention_k_eq_v": True,
      "layer_types": layer_types,
      "enable_moe_block": moe,
      "num_experts": 8 if moe else None,
      "v_norm_with_scale": False,
  }
  if not vision:
    return _ns(text_cfg)

  return _ns(
      {
          "model_type": "gemma4",
          "tie_word_embeddings": True,
          "text_config": text_cfg,
          "vision_config": {
              "model_type": "gemma4_vision",
              "hidden_size": 96,
              "intermediate_size": 192,
              "num_attention_heads": 4,
              "head_dim": 24,
              "num_hidden_layers": 2,
              "patch_size": 16,
              "position_embedding_size": 128,
          },
      }
  )


def test_build_gemma4_scanned_mapping_uses_cycle_prefixes_and_skips_global_v_proj():
  hf_config = _gemma4_config(moe=False, vision=True, num_layers=12)
  mapping = build_gemma4(ARCH_SPECS["gemma4"], hf_config, scan_layers=True)

  assert mapping["params-decoder-layers-layers_0-self_attention-query-kernel"] == [
      "model.language_model.layers.0.self_attn.q_proj.weight",
      "model.language_model.layers.6.self_attn.q_proj.weight",
  ]
  assert "params-decoder-layers-layers_5-self_attention-value-kernel" not in mapping
  assert mapping["params-decoder-layers-layers_5-self_attention-key-kernel"] == [
      "model.language_model.layers.5.self_attn.k_proj.weight",
      "model.language_model.layers.11.self_attn.k_proj.weight",
  ]
  assert (
      mapping["params-vision_encoder-Gemma4VisionProjector_0-projection-kernel"]
      == "model.embed_vision.embedding_projection.weight"
  )


def test_compute_gemma4_scanned_moe_and_vision_shapes():
  hf_config = _gemma4_config(moe=True, vision=True, num_layers=6)
  shapes = compute_megatext_shapes(hf_config, ARCH_SPECS["gemma4"], scan_layers=True, tie_word_embeddings=True)

  assert shapes["params-decoder-layers-layers_0-mlp-pre_forward_scale_2"] == (1, 128)
  assert shapes["params-decoder-layers-layers_0-mlp-moe_block-MoeBlock_0-per_expert_scale"] == (1, 8)
  assert shapes["params-decoder-layers-layers_0-mlp-moe_block-MoeBlock_0-wi_0"] == (1, 8, 128, 64)
  assert shapes["params-decoder-layers-layers_5-self_attention-query-kernel"] == (1, 128, 4, 64)
  assert "params-decoder-layers-layers_5-self_attention-value-kernel" not in shapes
  assert shapes["params-vision_encoder-Gemma4VisionEncoderLayer_0-vision_entry-input_projection-kernel"] == (
      16 * 16 * 3,
      96,
  )
  assert shapes["params-vision_encoder-Gemma4VisionProjector_0-projection-kernel"] == (96, 128)


def test_infer_hf_shapes_for_gemma4_text_and_vision():
  hf_config = _gemma4_config(moe=True, vision=True, num_layers=6)

  assert _infer_hf_shape_from_key("model.language_model.layers.0.self_attn.q_proj.weight", hf_config) == (128, 128)
  assert _infer_hf_shape_from_key("model.language_model.layers.5.self_attn.q_proj.weight", hf_config) == (256, 128)
  assert _infer_hf_shape_from_key("model.language_model.layers.5.self_attn.k_proj.weight", hf_config) == (64, 128)
  assert _infer_hf_shape_from_key("model.language_model.layers.0.router.scale", hf_config) == (128,)
  assert _infer_hf_shape_from_key("model.language_model.layers.0.router.per_expert_scale", hf_config) == (8,)
  assert _infer_hf_shape_from_key("model.vision_tower.patch_embedder.position_embedding_table", hf_config) == (
      2,
      128,
      96,
  )
  assert _infer_hf_shape_from_key("model.embed_vision.embedding_projection.weight", hf_config) == (128, 96)


def test_build_gemma4_transforms_include_concat_moe_and_pos_emb_axes():
  hf_config_dict = {
      "text_config": {
          "hidden_size": 128,
          "vocab_size": 1024,
      }
  }
  transforms = build_gemma4_transforms(ARCH_SPECS["gemma4"], hf_config_dict, to_hf=False)

  gate = np.arange(8 * 128, dtype=np.float32).reshape(8, 128)
  assert transforms["mlp-moe_block-MoeBlock_0-gate-kernel"](gate, ()).shape == (128, 8)

  pos_emb = np.arange(2 * 128 * 96, dtype=np.float32).reshape(2, 128, 96)
  swapped = transforms["vision_entry-pos_emb_param"](pos_emb, ())
  assert swapped.shape == (128, 2, 96)
