"""Llama model_type: mapping + transforms."""
from __future__ import annotations

from typing import Any

from megatext.conversion.utils import (
    ArchSpec, Mapping, TransformFn,
    _build_dense_kernel_transforms, _build_embedding_transforms,
    _cfg, _chain, _global_keys, _layer_mapping, _make_logits_transform,
    reorder_rope_fwd, reorder_rope_inv,
    reshape_kernel, reshape_kernel_inv,
    scale_query_fwd, scale_query_inv,
)


# ── Mapping ───────────────────────────────────────────────────────────────────


def build_llama(arch: ArchSpec, hf_config: Any, scan_layers: bool) -> Mapping:
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
    ], n_layers, scan_layers, prefix))
    return mapping


# ── Transforms ────────────────────────────────────────────────────────────────


def build_llama_transforms(
    arch: ArchSpec, hf_config_dict: dict[str, Any], *, to_hf: bool,
) -> dict[str, TransformFn]:
    text_cfg = hf_config_dict.get("text_config", hf_config_dict)
    hidden_size = text_cfg.get("hidden_size", 0)
    n_heads = text_cfg.get("num_attention_heads", 1)
    head_dim = text_cfg.get("head_dim", hidden_size // n_heads if n_heads else 0)
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

    # Query: reshape + rope + scale
    q_chain: list[TransformFn] = []
    k_chain: list[TransformFn] = []
    if to_hf:
        if arch.scale_query:
            q_chain.append(lambda x, s, hd=head_dim: scale_query_inv(x, s, hd))
        if arch.reorder_rope:
            q_chain.append(lambda x, s, hd=head_dim: reorder_rope_inv(x, s, hd))
            k_chain.append(lambda x, s, hd=head_dim: reorder_rope_inv(x, s, hd))
        q_chain.append(reshape_kernel_inv)
        k_chain.append(reshape_kernel_inv)
    else:
        q_chain.append(reshape_kernel)
        k_chain.append(reshape_kernel)
        if arch.reorder_rope:
            q_chain.append(lambda x, s, hd=head_dim: reorder_rope_fwd(x, s, hd))
            k_chain.append(lambda x, s, hd=head_dim: reorder_rope_fwd(x, s, hd))
        if arch.scale_query:
            q_chain.append(lambda x, s, hd=head_dim: scale_query_fwd(x, s, hd))

    transforms["self_attention-query-kernel"] = _chain(q_chain)
    if arch.reorder_rope:
        transforms["self_attention-key-kernel"] = _chain(k_chain)

    transforms.update(_build_embedding_transforms(arch, hidden_size, vocab_size, to_hf))
    transforms["logits_dense-kernel"] = _make_logits_transform(vocab_size, to_hf=to_hf)
    return transforms
