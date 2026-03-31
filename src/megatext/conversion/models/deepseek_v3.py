"""DeepSeek V3 model_type: mapping + transforms + shapes."""
from __future__ import annotations

from typing import Any

from megatext.conversion.utils import (
    ArchSpec, Mapping, TransformFn,
    _build_embedding_transforms, _cfg, _global_keys, _make_logits_transform,
    reshape_kernel, reshape_kernel_inv, simple_transpose,
)
from megatext.conversion.utils import pad_vocab_size


# ── Mapping ───────────────────────────────────────────────────────────────────


def build_deepseek(arch: ArchSpec, hf_config: Any, scan_layers: bool) -> Mapping:
    if not scan_layers:
        raise ValueError("deepseek_v3 only supports scanned layers")

    cfg = _cfg(hf_config)
    n_layers = cfg.num_hidden_layers
    tie = getattr(cfg, "tie_word_embeddings", False)
    first_k = getattr(cfg, "first_k_dense_replace", 1)
    n_experts = getattr(cfg, "n_routed_experts", getattr(cfg, "num_local_experts", 0))
    q_lora_rank = getattr(cfg, "q_lora_rank", 0)
    prefix = arch.hf_prefix

    dense_indices = list(range(first_k))
    moe_indices = list(range(first_k, n_layers))

    mapping: Mapping = {}
    mapping.update(_global_keys(prefix, tie))

    # ── MLA attention (shared structure for dense + MoE layers) ──
    mla_comps: list[tuple[str, str]] = []
    if q_lora_rank > 0:
        mla_comps.extend([
            ("self_attention-wq_a-kernel", "self_attn.q_a_proj.weight"),
            ("self_attention-q_norm-scale", "self_attn.q_a_layernorm.weight"),
            ("self_attention-wq_b-kernel", "self_attn.q_b_proj.weight"),
        ])
    else:
        mla_comps.append(
            ("self_attention-query-kernel", "self_attn.q_proj.weight"),
        )
    mla_comps.extend([
        ("self_attention-wkv_a-kernel", "self_attn.kv_a_proj_with_mqa.weight"),
        ("self_attention-kv_norm-scale", "self_attn.kv_a_layernorm.weight"),
        ("self_attention-wkv_b-kernel", "self_attn.kv_b_proj.weight"),
        ("self_attention-out-kernel", "self_attn.o_proj.weight"),
        ("pre_self_attention_layer_norm-scale", "input_layernorm.weight"),
        ("post_self_attention_layer_norm-scale", "post_attention_layernorm.weight"),
    ])

    # Dense layers (scanned over dense_indices)
    for mt_suffix, hf_suffix in mla_comps:
        mt_key = f"params-decoder-dense_layers-{mt_suffix}"
        mapping[mt_key] = [
            f"{prefix}.layers.{i}.{hf_suffix}" for i in dense_indices
        ]
    for mt_suffix, hf_suffix in [
        ("mlp-wi_0-kernel", "mlp.gate_proj.weight"),
        ("mlp-wi_1-kernel", "mlp.up_proj.weight"),
        ("mlp-wo-kernel", "mlp.down_proj.weight"),
    ]:
        mt_key = f"params-decoder-dense_layers-{mt_suffix}"
        mapping[mt_key] = [
            f"{prefix}.layers.{i}.{hf_suffix}" for i in dense_indices
        ]

    # MoE layers — attention (scanned over moe_indices)
    for mt_suffix, hf_suffix in mla_comps:
        mt_key = f"params-decoder-moe_layers-{mt_suffix}"
        mapping[mt_key] = [
            f"{prefix}.layers.{i}.{hf_suffix}" for i in moe_indices
        ]

    # MoE gate
    mp = "params-decoder-moe_layers"
    mapping[f"{mp}-DeepSeekMoeBlock_0-MoeBlock_0-gate-kernel"] = [
        f"{prefix}.layers.{i}.mlp.gate.weight" for i in moe_indices
    ]
    if q_lora_rank > 0:
        mapping[f"{mp}-DeepSeekMoeBlock_0-MoeBlock_0-gate-bias"] = [
            f"{prefix}.layers.{i}.mlp.gate.e_score_correction_bias"
            for i in moe_indices
        ]

    # Shared experts
    for mt_suf, hf_suf in [
        ("DeepSeekMoeBlock_0-shared_experts-wi_0-kernel",
         "mlp.shared_experts.gate_proj.weight"),
        ("DeepSeekMoeBlock_0-shared_experts-wi_1-kernel",
         "mlp.shared_experts.up_proj.weight"),
        ("DeepSeekMoeBlock_0-shared_experts-wo-kernel",
         "mlp.shared_experts.down_proj.weight"),
    ]:
        mapping[f"{mp}-{mt_suf}"] = [
            f"{prefix}.layers.{i}.{hf_suf}" for i in moe_indices
        ]

    # Routed experts: fused gate_up_proj → composite (wi_0, wi_1)
    mt_key = (
        f"{mp}-DeepSeekMoeBlock_0-MoeBlock_0-wi_0",
        f"{mp}-DeepSeekMoeBlock_0-MoeBlock_0-wi_1",
    )
    mapping[mt_key] = [
        f"{prefix}.layers.{i}.mlp.experts.gate_up_proj" for i in moe_indices
    ]
    # Routed experts: down_proj (direct, 3D per layer)
    mapping[f"{mp}-DeepSeekMoeBlock_0-MoeBlock_0-wo"] = [
        f"{prefix}.layers.{i}.mlp.experts.down_proj" for i in moe_indices
    ]

    return mapping


# ── Transforms ────────────────────────────────────────────────────────────────


def build_deepseek_transforms(
    arch: ArchSpec, hf_config_dict: dict[str, Any], *, to_hf: bool,
) -> dict[str, TransformFn]:
    text_cfg = hf_config_dict.get("text_config", hf_config_dict)
    hidden_size = text_cfg.get("hidden_size", 0)
    vocab_size = text_cfg.get("vocab_size", 0)

    transforms: dict[str, TransformFn] = {}

    # MLA kernels that need reshape (multi-head split)
    rk = reshape_kernel_inv if to_hf else reshape_kernel
    for suffix in [
        "self_attention-wq_b-kernel",
        "self_attention-wkv_b-kernel",
        "self_attention-out-kernel",
        "self_attention-query-kernel",  # fallback when no LoRA
    ]:
        transforms[suffix] = rk

    # MLA kernels that are just 2D transpose (no head split)
    tp = simple_transpose
    for suffix in [
        "self_attention-wq_a-kernel",
        "self_attention-wkv_a-kernel",
    ]:
        transforms[suffix] = tp

    # Dense MLP kernels
    for suffix in [
        "mlp-wi_0-kernel", "mlp-wi_1-kernel", "mlp-wo-kernel",
    ]:
        transforms[suffix] = tp

    # MoE gate + shared experts + routed experts: transpose
    for suffix in [
        "DeepSeekMoeBlock_0-MoeBlock_0-gate-kernel",
        "DeepSeekMoeBlock_0-shared_experts-wi_0-kernel",
        "DeepSeekMoeBlock_0-shared_experts-wi_1-kernel",
        "DeepSeekMoeBlock_0-shared_experts-wo-kernel",
        "DeepSeekMoeBlock_0-MoeBlock_0-wi_0",
        "DeepSeekMoeBlock_0-MoeBlock_0-wi_1",
        "DeepSeekMoeBlock_0-MoeBlock_0-wo",
    ]:
        transforms[suffix] = tp

    transforms.update(_build_embedding_transforms(arch, hidden_size, vocab_size, to_hf))
    transforms["logits_dense-kernel"] = _make_logits_transform(vocab_size, to_hf=to_hf)
    return transforms


# ── Shapes ────────────────────────────────────────────────────────────────────


def compute_deepseek_shapes(
    cfg: Any, arch: ArchSpec, scan_layers: bool,
    *, tie_word_embeddings: bool = False,
) -> dict[str, tuple]:
    if not scan_layers:
        raise ValueError("deepseek_v3 only supports scanned layers")

    emb = cfg.hidden_size
    mlp = cfg.intermediate_size
    nq = cfg.num_attention_heads
    nkv = getattr(cfg, "num_key_value_heads", nq)
    vocab = cfg.vocab_size
    n_layers = cfg.num_hidden_layers
    first_k = getattr(cfg, "first_k_dense_replace", 1)
    n_experts = getattr(cfg, "n_routed_experts", getattr(cfg, "num_local_experts", 0))
    q_lora_rank = getattr(cfg, "q_lora_rank", 0)
    kv_lora_rank = getattr(cfg, "kv_lora_rank", 512)
    qk_rope_hd = getattr(cfg, "qk_rope_head_dim", 64)
    qk_nope_hd = getattr(cfg, "qk_nope_head_dim", 128)
    v_hd = getattr(cfg, "v_head_dim", 128)
    n_shared = getattr(cfg, "n_shared_experts", 1)
    moe_mlp = getattr(cfg, "moe_intermediate_size", mlp)
    shared_mlp = getattr(
        cfg, "shared_expert_intermediate_size", moe_mlp * n_shared
    )
    padded_vocab = pad_vocab_size(vocab)

    n_dense = first_k
    n_moe = n_layers - first_k

    shapes: dict[str, tuple] = {}
    shapes["params-token_embedder-embedding"] = (padded_vocab, emb)
    shapes["params-decoder-decoder_norm-scale"] = (emb,)
    if not tie_word_embeddings:
        shapes["params-decoder-logits_dense-kernel"] = (emb, padded_vocab)

    # MLA attention shapes (shared by dense and MoE)
    def _add_mla_shapes(prefix: str, n: int) -> None:
        lp = (n,)
        if q_lora_rank > 0:
            shapes[f"{prefix}-self_attention-wq_a-kernel"] = (*lp, emb, q_lora_rank)
            shapes[f"{prefix}-self_attention-q_norm-scale"] = (*lp, q_lora_rank)
            shapes[f"{prefix}-self_attention-wq_b-kernel"] = (
                *lp, q_lora_rank, nq, qk_nope_hd + qk_rope_hd
            )
        else:
            shapes[f"{prefix}-self_attention-query-kernel"] = (
                *lp, emb, nq, qk_nope_hd + qk_rope_hd
            )
        shapes[f"{prefix}-self_attention-wkv_a-kernel"] = (
            *lp, emb, kv_lora_rank + qk_rope_hd
        )
        shapes[f"{prefix}-self_attention-kv_norm-scale"] = (*lp, kv_lora_rank)
        shapes[f"{prefix}-self_attention-wkv_b-kernel"] = (
            *lp, kv_lora_rank, nkv, qk_nope_hd + v_hd
        )
        shapes[f"{prefix}-self_attention-out-kernel"] = (*lp, nq, v_hd, emb)
        shapes[f"{prefix}-pre_self_attention_layer_norm-scale"] = (*lp, emb)
        shapes[f"{prefix}-post_self_attention_layer_norm-scale"] = (*lp, emb)

    _add_mla_shapes("params-decoder-dense_layers", n_dense)
    _add_mla_shapes("params-decoder-moe_layers", n_moe)

    # Dense MLP
    dp = "params-decoder-dense_layers"
    shapes[f"{dp}-mlp-wi_0-kernel"] = (n_dense, emb, mlp)
    shapes[f"{dp}-mlp-wi_1-kernel"] = (n_dense, emb, mlp)
    shapes[f"{dp}-mlp-wo-kernel"] = (n_dense, mlp, emb)

    # MoE components
    mp = "params-decoder-moe_layers"
    shapes[f"{mp}-DeepSeekMoeBlock_0-MoeBlock_0-gate-kernel"] = (n_moe, emb, n_experts)
    if q_lora_rank > 0:
        shapes[f"{mp}-DeepSeekMoeBlock_0-MoeBlock_0-gate-bias"] = (n_moe, n_experts)

    # Shared experts
    shapes[f"{mp}-DeepSeekMoeBlock_0-shared_experts-wi_0-kernel"] = (n_moe, emb, shared_mlp)
    shapes[f"{mp}-DeepSeekMoeBlock_0-shared_experts-wi_1-kernel"] = (n_moe, emb, shared_mlp)
    shapes[f"{mp}-DeepSeekMoeBlock_0-shared_experts-wo-kernel"] = (n_moe, shared_mlp, emb)

    # Routed experts: fused format, layer dim first for scan
    shapes[f"{mp}-DeepSeekMoeBlock_0-MoeBlock_0-wi_0"] = (n_moe, n_experts, emb, moe_mlp)
    shapes[f"{mp}-DeepSeekMoeBlock_0-MoeBlock_0-wi_1"] = (n_moe, n_experts, emb, moe_mlp)
    shapes[f"{mp}-DeepSeekMoeBlock_0-MoeBlock_0-wo"] = (n_moe, n_experts, moe_mlp, emb)

    return shapes
