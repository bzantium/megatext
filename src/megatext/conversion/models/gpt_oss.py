"""GPT-OSS model_type: mapping + transforms + shapes."""
from __future__ import annotations

from typing import Any

from megatext.conversion.utils import (
    ArchSpec, Mapping, TransformFn,
    _build_dense_kernel_transforms, _build_embedding_transforms,
    _cfg, _global_keys, _make_logits_transform,
    reshape_bias, simple_transpose,
)
from megatext.conversion.utils import pad_vocab_size


# ── Helper ────────────────────────────────────────────────────────────────────


def _gpt_oss_cycle(cfg: Any) -> int:
    """Compute cycle interval from layer_types list."""
    layer_types = getattr(cfg, "layer_types", None)
    if not layer_types:
        return 1
    n = len(layer_types)
    for c in range(1, n + 1):
        if n % c == 0 and all(
            layer_types[i] == layer_types[i % c] for i in range(n)
        ):
            return c
    return n


# ── Mapping ───────────────────────────────────────────────────────────────────


def build_gpt_oss(arch: ArchSpec, hf_config: Any, scan_layers: bool) -> Mapping:
    if not scan_layers:
        raise ValueError("gpt_oss only supports scanned layers")

    cfg = _cfg(hf_config)
    n_layers = cfg.num_hidden_layers
    tie = getattr(cfg, "tie_word_embeddings", False)
    cycle = _gpt_oss_cycle(cfg)
    prefix = arch.hf_prefix
    n_experts = getattr(cfg, "num_experts", getattr(cfg, "num_local_experts", 0))

    mapping: Mapping = {}
    mapping.update(_global_keys(prefix, tie))

    for b in range(cycle):
        hf_indices = list(range(b, n_layers, cycle))
        block_prefix = f"params-decoder-layers-layers_{b}"

        comps: list[tuple[str, str]] = [
            ("GptOssAttention-query-kernel", "self_attn.q_proj.weight"),
            ("GptOssAttention-key-kernel", "self_attn.k_proj.weight"),
            ("GptOssAttention-value-kernel", "self_attn.v_proj.weight"),
            ("GptOssAttention-out-kernel", "self_attn.o_proj.weight"),
            ("GptOssAttention-query-bias", "self_attn.q_proj.bias"),
            ("GptOssAttention-key-bias", "self_attn.k_proj.bias"),
            ("GptOssAttention-value-bias", "self_attn.v_proj.bias"),
            ("GptOssAttention-out-bias", "self_attn.o_proj.bias"),
            ("GptOssAttention-sinks", "self_attn.sinks"),
            ("pre_self_attention_layer_norm-scale", "input_layernorm.weight"),
            ("post_self_attention_layer_norm-scale", "post_attention_layernorm.weight"),
            # MoE gate (router)
            ("GptOssMlp-gate-kernel", "mlp.router.weight"),
            ("GptOssMlp-gate-bias", "mlp.router.bias"),
            # Expert down (direct, 3D per layer)
            ("GptOssMlp-wo", "mlp.experts.down_proj"),
            ("GptOssMlp-wo_bias", "mlp.experts.down_proj_bias"),
        ]

        for mt_suffix, hf_suffix in comps:
            mt_key = f"{block_prefix}-{mt_suffix}"
            hf_keys = [
                f"{prefix}.layers.{i}.{hf_suffix}" for i in hf_indices
            ]
            mapping[mt_key] = hf_keys

        # Composite: fused gate_up_proj → (wi_0, wi_1)
        for mt_pair, hf_suffix in [
            (("GptOssMlp-wi_0", "GptOssMlp-wi_1"),
             "mlp.experts.gate_up_proj"),
            (("GptOssMlp-wi_0_bias", "GptOssMlp-wi_1_bias"),
             "mlp.experts.gate_up_proj_bias"),
        ]:
            mt_key = tuple(f"{block_prefix}-{s}" for s in mt_pair)
            hf_keys = [
                f"{prefix}.layers.{i}.{hf_suffix}" for i in hf_indices
            ]
            mapping[mt_key] = hf_keys

    return mapping


# ── Transforms ────────────────────────────────────────────────────────────────


def build_gpt_oss_transforms(
    arch: ArchSpec, hf_config_dict: dict[str, Any], *, to_hf: bool,
) -> dict[str, TransformFn]:
    text_cfg = hf_config_dict.get("text_config", hf_config_dict)
    hidden_size = text_cfg.get("hidden_size", 0)
    vocab_size = text_cfg.get("vocab_size", 0)

    transforms: dict[str, TransformFn] = {}

    # Attention kernels: reshape_kernel
    transforms.update(_build_dense_kernel_transforms([
        "GptOssAttention-query-kernel",
        "GptOssAttention-key-kernel",
        "GptOssAttention-value-kernel",
        "GptOssAttention-out-kernel",
    ], to_hf))

    # Attention biases: reshape (flatten/split heads)
    for suffix in [
        "GptOssAttention-query-bias",
        "GptOssAttention-key-bias",
        "GptOssAttention-value-bias",
        "GptOssAttention-out-bias",
    ]:
        transforms[suffix] = reshape_bias

    # MoE gate: transpose
    tp = simple_transpose
    transforms["GptOssMlp-gate-kernel"] = tp

    # Expert weights: transpose last 2 dims (3D: num_experts, dim1, dim2)
    for suffix in ["GptOssMlp-wo", "GptOssMlp-wi_0", "GptOssMlp-wi_1"]:
        transforms[suffix] = tp

    transforms.update(_build_embedding_transforms(arch, hidden_size, vocab_size, to_hf))
    transforms["logits_dense-kernel"] = _make_logits_transform(vocab_size, to_hf=to_hf)
    return transforms


# ── Shapes ────────────────────────────────────────────────────────────────────


def compute_gpt_oss_shapes(
    cfg: Any, arch: ArchSpec, scan_layers: bool,
    *, tie_word_embeddings: bool = False,
) -> dict[str, tuple]:
    if not scan_layers:
        raise ValueError("gpt_oss only supports scanned layers")

    emb = cfg.hidden_size
    mlp = cfg.intermediate_size
    nq = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", emb // nq)
    vocab = cfg.vocab_size
    n_layers = cfg.num_hidden_layers
    cycle = _gpt_oss_cycle(cfg)
    n_experts = getattr(cfg, "num_experts", getattr(cfg, "num_local_experts", 0))
    padded_vocab = pad_vocab_size(vocab)

    shapes: dict[str, tuple] = {}
    shapes["params-token_embedder-embedding"] = (padded_vocab, emb)
    shapes["params-decoder-decoder_norm-scale"] = (emb,)
    if not tie_word_embeddings:
        shapes["params-decoder-logits_dense-kernel"] = (emb, padded_vocab)

    for b in range(cycle):
        n_stacked = len(range(b, n_layers, cycle))
        lp = (n_stacked,)
        bp = f"params-decoder-layers-layers_{b}"

        # Attention kernels
        shapes[f"{bp}-GptOssAttention-query-kernel"] = (*lp, emb, nq, hd)
        shapes[f"{bp}-GptOssAttention-key-kernel"] = (*lp, emb, nkv, hd)
        shapes[f"{bp}-GptOssAttention-value-kernel"] = (*lp, emb, nkv, hd)
        shapes[f"{bp}-GptOssAttention-out-kernel"] = (*lp, nq, hd, emb)

        # Attention biases
        shapes[f"{bp}-GptOssAttention-query-bias"] = (*lp, nq, hd)
        shapes[f"{bp}-GptOssAttention-key-bias"] = (*lp, nkv, hd)
        shapes[f"{bp}-GptOssAttention-value-bias"] = (*lp, nkv, hd)
        shapes[f"{bp}-GptOssAttention-out-bias"] = (*lp, emb)

        # Sinks: 1D per head
        shapes[f"{bp}-GptOssAttention-sinks"] = (*lp, nq)

        # Norms
        shapes[f"{bp}-pre_self_attention_layer_norm-scale"] = (*lp, emb)
        shapes[f"{bp}-post_self_attention_layer_norm-scale"] = (*lp, emb)

        # MoE gate (router)
        shapes[f"{bp}-GptOssMlp-gate-kernel"] = (*lp, emb, n_experts)
        shapes[f"{bp}-GptOssMlp-gate-bias"] = (*lp, n_experts)

        # Expert weights
        shapes[f"{bp}-GptOssMlp-wi_0"] = (*lp, n_experts, mlp, emb)
        shapes[f"{bp}-GptOssMlp-wi_1"] = (*lp, n_experts, mlp, emb)
        shapes[f"{bp}-GptOssMlp-wo"] = (*lp, n_experts, emb, mlp)
        shapes[f"{bp}-GptOssMlp-wi_0_bias"] = (*lp, n_experts, mlp)
        shapes[f"{bp}-GptOssMlp-wi_1_bias"] = (*lp, n_experts, mlp)
        shapes[f"{bp}-GptOssMlp-wo_bias"] = (*lp, n_experts, emb)

    return shapes
