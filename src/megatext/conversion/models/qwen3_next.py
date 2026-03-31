"""Qwen3-Next model_type: mapping + transforms + shapes."""
from __future__ import annotations

from typing import Any

from megatext.conversion.utils import (
    ArchSpec, Mapping, TransformFn,
    _build_dense_kernel_transforms, _build_embedding_transforms,
    _cfg, _global_keys, _make_logits_transform,
    permute_conv, simple_transpose,
)
from megatext.conversion.utils import pad_vocab_size


# ── Mapping ───────────────────────────────────────────────────────────────────


def build_qwen3_next(arch: ArchSpec, hf_config: Any, scan_layers: bool) -> Mapping:
    if not scan_layers:
        raise ValueError("qwen3_next only supports scanned layers")

    cfg = _cfg(hf_config)
    n_layers = cfg.num_hidden_layers
    tie = getattr(cfg, "tie_word_embeddings", False)
    cycle = getattr(cfg, "full_attention_interval",
                    getattr(cfg, "inhomogeneous_layer_cycle_interval", 4))
    n_experts = getattr(cfg, "num_experts", getattr(cfg, "num_local_experts", 0))
    prefix = arch.hf_prefix

    mapping: Mapping = {}
    mapping.update(_global_keys(prefix, tie))

    for b in range(cycle):
        is_full_attn = (b + 1) % cycle == 0
        hf_indices = list(range(b, n_layers, cycle))
        block_prefix = f"params-decoder-layers-layer_{b}"

        comps: list[tuple[str, str]] = [
            ("input_layernorm-scale", "input_layernorm.weight"),
            ("post_attention_layernorm-scale", "post_attention_layernorm.weight"),
        ]

        if is_full_attn:
            comps.extend([
                ("attention-attention-query-kernel", "self_attn.q_proj.weight"),
                ("attention-attention-key-kernel", "self_attn.k_proj.weight"),
                ("attention-attention-value-kernel", "self_attn.v_proj.weight"),
                ("attention-attention-out-kernel", "self_attn.o_proj.weight"),
                ("attention-attention-query_norm-scale", "self_attn.q_norm.weight"),
                ("attention-attention-key_norm-scale", "self_attn.k_norm.weight"),
            ])
        else:
            comps.extend([
                ("attention-in_proj_qkvz-kernel", "linear_attn.in_proj_qkvz.weight"),
                ("attention-in_proj_ba-kernel", "linear_attn.in_proj_ba.weight"),
                ("attention-conv1d-kernel", "linear_attn.conv1d.weight"),
                ("attention-A_log", "linear_attn.A_log"),
                ("attention-dt_bias", "linear_attn.dt_bias"),
                ("attention-norm-rms_norm-scale", "linear_attn.norm.weight"),
                ("attention-out_proj-kernel", "linear_attn.out_proj.weight"),
            ])

        # MoE regular components (all blocks)
        comps.extend([
            ("mlp-routed_experts-gate-kernel", "mlp.gate.weight"),
            ("mlp-shared_expert-wi_0-kernel", "mlp.shared_expert.gate_proj.weight"),
            ("mlp-shared_expert-wi_1-kernel", "mlp.shared_expert.up_proj.weight"),
            ("mlp-shared_expert-wo-kernel", "mlp.shared_expert.down_proj.weight"),
            ("mlp-shared_expert_gate-kernel", "mlp.shared_expert_gate.weight"),
        ])

        for mt_suffix, hf_suffix in comps:
            mt_key = f"{block_prefix}-{mt_suffix}"
            hf_keys = [f"{prefix}.layers.{i}.{hf_suffix}" for i in hf_indices]
            mapping[mt_key] = hf_keys

        # Routed experts: fused gate_up_proj → composite (wi_0, wi_1)
        mt_key = (
            f"{block_prefix}-mlp-routed_experts-wi_0",
            f"{block_prefix}-mlp-routed_experts-wi_1",
        )
        mapping[mt_key] = [
            f"{prefix}.layers.{i}.mlp.experts.gate_up_proj" for i in hf_indices
        ]
        # Routed experts: down_proj (direct, 3D per layer)
        mapping[f"{block_prefix}-mlp-routed_experts-wo"] = [
            f"{prefix}.layers.{i}.mlp.experts.down_proj" for i in hf_indices
        ]

    return mapping


# ── Transforms ────────────────────────────────────────────────────────────────


def build_qwen3_next_transforms(
    arch: ArchSpec, hf_config_dict: dict[str, Any], *, to_hf: bool,
) -> dict[str, TransformFn]:
    text_cfg = hf_config_dict.get("text_config", hf_config_dict)
    hidden_size = text_cfg.get("hidden_size", 0)
    vocab_size = text_cfg.get("vocab_size", 0)

    transforms: dict[str, TransformFn] = {}

    # Full attention kernels: reshape_kernel
    transforms.update(_build_dense_kernel_transforms([
        "attention-attention-query-kernel",
        "attention-attention-key-kernel",
        "attention-attention-value-kernel",
        "attention-attention-out-kernel",
    ], to_hf))

    # GDN kernels: simple transpose (2D)
    tp = simple_transpose
    for suffix in [
        "attention-in_proj_qkvz-kernel",
        "attention-in_proj_ba-kernel",
        "attention-out_proj-kernel",
    ]:
        transforms[suffix] = tp

    # Conv1d: axis permutation (K,1,C) ↔ (C,1,K)
    transforms["attention-conv1d-kernel"] = permute_conv

    # MoE: router gate transpose, expert weights transpose
    transforms["mlp-routed_experts-gate-kernel"] = tp
    for suffix in [
        "mlp-routed_experts-wi_0", "mlp-routed_experts-wi_1", "mlp-routed_experts-wo",
        "mlp-shared_expert-wi_0-kernel", "mlp-shared_expert-wi_1-kernel",
        "mlp-shared_expert-wo-kernel",
        "mlp-shared_expert_gate-kernel",
    ]:
        transforms[suffix] = tp

    transforms.update(_build_embedding_transforms(arch, hidden_size, vocab_size, to_hf))
    transforms["logits_dense-kernel"] = _make_logits_transform(vocab_size, to_hf=to_hf)
    return transforms


# ── Shapes ────────────────────────────────────────────────────────────────────


def compute_qwen3_next_shapes(
    cfg: Any, arch: ArchSpec, scan_layers: bool,
    *, tie_word_embeddings: bool = False,
) -> dict[str, tuple]:
    if not scan_layers:
        raise ValueError("qwen3_next only supports scanned layers")

    emb = cfg.hidden_size
    nq = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", emb // nq)
    vocab = cfg.vocab_size
    n_layers = cfg.num_hidden_layers
    cycle = getattr(cfg, "full_attention_interval",
                    getattr(cfg, "inhomogeneous_layer_cycle_interval", 4))
    n_experts = getattr(cfg, "num_experts", getattr(cfg, "num_local_experts", 0))
    moe_mlp = getattr(cfg, "moe_intermediate_size", getattr(cfg, "intermediate_size", 0))
    shared_mlp = getattr(cfg, "shared_expert_intermediate_size", cfg.intermediate_size)
    padded_vocab = pad_vocab_size(vocab)

    # GDN dimensions (HF uses linear_* field names)
    gdn_kh = getattr(cfg, "linear_num_key_heads", nkv)
    gdn_vh = getattr(cfg, "linear_num_value_heads", nkv)
    gdn_khd = getattr(cfg, "linear_key_head_dim", hd)
    gdn_vhd = getattr(cfg, "linear_value_head_dim", hd)
    gdn_conv_k = getattr(cfg, "linear_conv_kernel_dim", 4)

    shapes: dict[str, tuple] = {}
    shapes["params-token_embedder-embedding"] = (padded_vocab, emb)
    shapes["params-decoder-decoder_norm-scale"] = (emb,)
    if not tie_word_embeddings:
        shapes["params-decoder-logits_dense-kernel"] = (emb, padded_vocab)

    for b in range(cycle):
        is_full = (b + 1) % cycle == 0
        n_stacked = len(range(b, n_layers, cycle))
        lp = (n_stacked,)
        bp = f"params-decoder-layers-layer_{b}"

        # Norms
        shapes[f"{bp}-input_layernorm-scale"] = (*lp, emb)
        shapes[f"{bp}-post_attention_layernorm-scale"] = (*lp, emb)

        if is_full:
            # Qwen3Next query has 2*head_dim per head (nope + rope parts)
            shapes[f"{bp}-attention-attention-query-kernel"] = (*lp, emb, nq, 2 * hd)
            shapes[f"{bp}-attention-attention-key-kernel"] = (*lp, emb, nkv, hd)
            shapes[f"{bp}-attention-attention-value-kernel"] = (*lp, emb, nkv, hd)
            shapes[f"{bp}-attention-attention-out-kernel"] = (*lp, nq, hd, emb)
            shapes[f"{bp}-attention-attention-query_norm-scale"] = (*lp, hd)
            shapes[f"{bp}-attention-attention-key_norm-scale"] = (*lp, hd)
        else:
            # GDN components
            key_dim = gdn_kh * gdn_khd
            value_dim = gdn_vh * gdn_vhd
            qkvz_dim = 2 * key_dim + 2 * value_dim
            conv_dim = 2 * key_dim + value_dim
            ba_dim = 2 * gdn_vh
            shapes[f"{bp}-attention-in_proj_qkvz-kernel"] = (*lp, emb, qkvz_dim)
            shapes[f"{bp}-attention-in_proj_ba-kernel"] = (*lp, emb, ba_dim)
            shapes[f"{bp}-attention-conv1d-kernel"] = (*lp, conv_dim, 1, gdn_conv_k)
            shapes[f"{bp}-attention-A_log"] = (*lp, gdn_vh)
            shapes[f"{bp}-attention-dt_bias"] = (*lp, gdn_vh)
            shapes[f"{bp}-attention-norm-rms_norm-scale"] = (*lp, gdn_vhd)
            shapes[f"{bp}-attention-out_proj-kernel"] = (*lp, value_dim, emb)

        # MoE
        shapes[f"{bp}-mlp-routed_experts-gate-kernel"] = (*lp, emb, n_experts)
        shapes[f"{bp}-mlp-routed_experts-wi_0"] = (*lp, n_experts, emb, moe_mlp)
        shapes[f"{bp}-mlp-routed_experts-wi_1"] = (*lp, n_experts, emb, moe_mlp)
        shapes[f"{bp}-mlp-routed_experts-wo"] = (*lp, n_experts, moe_mlp, emb)
        shapes[f"{bp}-mlp-shared_expert-wi_0-kernel"] = (*lp, emb, shared_mlp)
        shapes[f"{bp}-mlp-shared_expert-wi_1-kernel"] = (*lp, emb, shared_mlp)
        shapes[f"{bp}-mlp-shared_expert-wo-kernel"] = (*lp, shared_mlp, emb)
        shapes[f"{bp}-mlp-shared_expert_gate-kernel"] = (*lp, emb, 1)

    return shapes
