"""Qwen3 MoE model_type: mapping + transforms + shapes."""
from __future__ import annotations

from typing import Any

from megatext.conversion.utils import (
    ArchSpec, Mapping, TransformFn,
    _build_dense_kernel_transforms, _build_embedding_transforms,
    _cfg, _global_keys, _layer_mapping, _make_logits_transform,
    simple_transpose,
)
from megatext.conversion.utils import pad_vocab_size


# ── Mapping ───────────────────────────────────────────────────────────────────


def build_qwen3_moe(arch: ArchSpec, hf_config: Any, scan_layers: bool) -> Mapping:
    cfg = _cfg(hf_config)
    n_layers = cfg.num_hidden_layers
    tie = getattr(cfg, "tie_word_embeddings", False)
    n_experts = getattr(cfg, "num_experts", getattr(cfg, "num_local_experts", 0))
    prefix = arch.hf_prefix

    mapping: Mapping = {}
    mapping.update(_global_keys(prefix, tie))

    # Attention + norms + gate (regular layer params)
    mapping.update(_layer_mapping([
        ("self_attention-query-kernel", "self_attn.q_proj.weight"),
        ("self_attention-key-kernel", "self_attn.k_proj.weight"),
        ("self_attention-value-kernel", "self_attn.v_proj.weight"),
        ("self_attention-out-kernel", "self_attn.o_proj.weight"),
        ("pre_self_attention_layer_norm-scale", "input_layernorm.weight"),
        ("post_self_attention_layer_norm-scale", "post_attention_layernorm.weight"),
        ("self_attention-query_norm-scale", "self_attn.q_norm.weight"),
        ("self_attention-key_norm-scale", "self_attn.k_norm.weight"),
        ("moe_block-gate-kernel", "mlp.gate.weight"),
    ], n_layers, scan_layers, prefix))

    # Expert weights — HF uses fused 3D tensors (via transformers library):
    #   experts.gate_up_proj: (n_experts, 2*moe_intermediate, hidden)
    #   experts.down_proj: (n_experts, hidden, moe_intermediate)
    if scan_layers:
        # Composite: fused gate_up_proj → (wi_0, wi_1)
        mt_key = (
            "params-decoder-layers-moe_block-wi_0",
            "params-decoder-layers-moe_block-wi_1",
        )
        mapping[mt_key] = [
            f"{prefix}.layers.{i}.mlp.experts.gate_up_proj"
            for i in range(n_layers)
        ]
        # Direct: down_proj → wo
        mapping["params-decoder-layers-moe_block-wo"] = [
            f"{prefix}.layers.{i}.mlp.experts.down_proj"
            for i in range(n_layers)
        ]
    else:
        for i in range(n_layers):
            mt_key = (
                f"params-decoder-layers_{i}-moe_block-wi_0",
                f"params-decoder-layers_{i}-moe_block-wi_1",
            )
            mapping[mt_key] = f"{prefix}.layers.{i}.mlp.experts.gate_up_proj"
            mapping[f"params-decoder-layers_{i}-moe_block-wo"] = (
                f"{prefix}.layers.{i}.mlp.experts.down_proj"
            )

    return mapping


# ── Transforms ────────────────────────────────────────────────────────────────


def build_qwen3_moe_transforms(
    arch: ArchSpec, hf_config_dict: dict[str, Any], *, to_hf: bool,
) -> dict[str, TransformFn]:
    text_cfg = hf_config_dict.get("text_config", hf_config_dict)
    hidden_size = text_cfg.get("hidden_size", 0)
    vocab_size = text_cfg.get("vocab_size", 0)

    transforms: dict[str, TransformFn] = {}
    # Attention kernels: reshape
    transforms.update(_build_dense_kernel_transforms([
        "self_attention-query-kernel",
        "self_attention-key-kernel",
        "self_attention-value-kernel",
        "self_attention-out-kernel",
    ], to_hf))
    # Expert weights + gate: simple transpose (2D per-tensor)
    tp = simple_transpose
    for suffix in ["moe_block-wi_0", "moe_block-wi_1", "moe_block-wo",
                    "moe_block-gate-kernel"]:
        transforms[suffix] = tp
    transforms.update(_build_embedding_transforms(arch, hidden_size, vocab_size, to_hf))
    transforms["logits_dense-kernel"] = _make_logits_transform(vocab_size, to_hf=to_hf)
    return transforms


# ── Shapes ────────────────────────────────────────────────────────────────────


def compute_qwen3_moe_shapes(
    cfg: Any, arch: ArchSpec, scan_layers: bool,
    *, tie_word_embeddings: bool = False,
) -> dict[str, tuple]:
    emb = cfg.hidden_size
    nq = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", emb // nq)
    vocab = cfg.vocab_size
    n_layers = cfg.num_hidden_layers
    n_experts = getattr(cfg, "num_experts", getattr(cfg, "num_local_experts", 0))
    moe_mlp = getattr(cfg, "moe_intermediate_size", cfg.intermediate_size)
    padded_vocab = pad_vocab_size(vocab)
    lp = (n_layers,) if scan_layers else ()

    shapes: dict[str, tuple] = {}
    shapes["params-token_embedder-embedding"] = (padded_vocab, emb)
    shapes["params-decoder-decoder_norm-scale"] = (emb,)
    if not tie_word_embeddings:
        shapes["params-decoder-logits_dense-kernel"] = (emb, padded_vocab)

    kernel_shapes: dict[str, tuple] = {
        "self_attention-query-kernel": (*lp, emb, nq, hd),
        "self_attention-key-kernel": (*lp, emb, nkv, hd),
        "self_attention-value-kernel": (*lp, emb, nkv, hd),
        "self_attention-out-kernel": (*lp, nq, hd, emb),
        "pre_self_attention_layer_norm-scale": (*lp, emb),
        "post_self_attention_layer_norm-scale": (*lp, emb),
        "self_attention-query_norm-scale": (*lp, hd),
        "self_attention-key_norm-scale": (*lp, hd),
        "moe_block-gate-kernel": (*lp, emb, n_experts),
    }

    for suffix, shape in kernel_shapes.items():
        if scan_layers:
            shapes[f"params-decoder-layers-{suffix}"] = shape
        else:
            for i in range(n_layers):
                shapes[f"params-decoder-layers_{i}-{suffix}"] = shape

    # Expert weights: (n_layers, n_experts, ...) for scanned (layer dim first for scan)
    expert_lp = (*lp, n_experts)
    expert_shapes = {
        "moe_block-wi_0": (*expert_lp, emb, moe_mlp),
        "moe_block-wi_1": (*expert_lp, emb, moe_mlp),
        "moe_block-wo": (*expert_lp, moe_mlp, emb),
    }
    for suffix, shape in expert_shapes.items():
        if scan_layers:
            shapes[f"params-decoder-layers-{suffix}"] = shape
        else:
            for i in range(n_layers):
                # Unscanned: per-layer, shape is (n_experts, ...)
                shapes[f"params-decoder-layers_{i}-{suffix}"] = (n_experts, emb, moe_mlp) \
                    if "wo" not in suffix else (n_experts, moe_mlp, emb)

    return shapes
