"""Qwen3 dense model_type: mapping + transforms."""
from __future__ import annotations

from typing import Any

from megatext.conversion.utils import (
    ArchSpec, Mapping, TransformFn,
    _build_dense_kernel_transforms, _build_embedding_transforms,
    _cfg, _global_keys, _layer_mapping, _make_logits_transform,
)


# ── Mapping ───────────────────────────────────────────────────────────────────


def build_qwen3(arch: ArchSpec, hf_config: Any, scan_layers: bool) -> Mapping:
    cfg = _cfg(hf_config)
    n_layers = cfg.num_hidden_layers
    tie = getattr(cfg, "tie_word_embeddings", False)
    prefix = arch.hf_prefix

    mapping: Mapping = {}
    mapping.update(_global_keys(prefix, tie))
    mapping.update(_layer_mapping([
        ("self_attention-query-kernel", "self_attn.q_proj.weight"),
        ("self_attention-key-kernel", "self_attn.k_proj.weight"),
        ("self_attention-value-kernel", "self_attn.v_proj.weight"),
        ("self_attention-out-kernel", "self_attn.o_proj.weight"),
        ("mlp-wi_0-kernel", "mlp.gate_proj.weight"),
        ("mlp-wi_1-kernel", "mlp.up_proj.weight"),
        ("mlp-wo-kernel", "mlp.down_proj.weight"),
        ("pre_self_attention_layer_norm-scale", "input_layernorm.weight"),
        ("post_self_attention_layer_norm-scale", "post_attention_layernorm.weight"),
        ("self_attention-query_norm-scale", "self_attn.q_norm.weight"),
        ("self_attention-key_norm-scale", "self_attn.k_norm.weight"),
    ], n_layers, scan_layers, prefix))
    return mapping


# ── Transforms ────────────────────────────────────────────────────────────────


def build_qwen3_transforms(
    arch: ArchSpec, hf_config_dict: dict[str, Any], *, to_hf: bool,
) -> dict[str, TransformFn]:
    text_cfg = hf_config_dict.get("text_config", hf_config_dict)
    hidden_size = text_cfg.get("hidden_size", 0)
    vocab_size = text_cfg.get("vocab_size", 0)

    transforms: dict[str, TransformFn] = {}
    transforms.update(_build_dense_kernel_transforms([
        "self_attention-query-kernel",
        "self_attention-key-kernel",
        "self_attention-value-kernel",
        "self_attention-out-kernel",
        "mlp-wi_0-kernel",
        "mlp-wi_1-kernel",
        "mlp-wo-kernel",
    ], to_hf))
    transforms.update(_build_embedding_transforms(arch, hidden_size, vocab_size, to_hf))
    transforms["logits_dense-kernel"] = _make_logits_transform(vocab_size, to_hf=to_hf)
    return transforms
