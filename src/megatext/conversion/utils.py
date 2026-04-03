"""Conversion utilities: ArchSpec, mapping, transforms, and config translation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from megatext.utils import logging as max_logging

# ══════════════════════════════════════════════════════════════════════════════
# ArchSpec — model type descriptors
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ArchSpec:
    """Describes how to convert an HF model_type family."""

    model_type: str
    decoder_block: str
    template_name: str

    # HF key prefix (most: "model", multimodal Gemma4 text tower: "model.language_model")
    hf_prefix: str = "model"

    # Per-layer component flags
    has_qk_norm: bool = False
    has_post_attn_norm: bool = False
    has_pre_ffw_norm: bool = False
    has_post_ffw_norm: bool = False

    # Transform flags
    scale_query: bool = False
    reorder_rope: bool = False
    scale_embedding: bool = False
    scale_rmsnorm: bool = False

    # How fused gate_up expert weights are split/merged
    # "deinterleave": even/odd indexing (default for dense models)
    # "concat": first-half/second-half split
    # "interleave_last": even/odd on last axis (GPT-OSS)
    composite_split: str = "deinterleave"


# ══════════════════════════════════════════════════════════════════════════════
# Mapping — HF↔Megatext key name mapping
# ══════════════════════════════════════════════════════════════════════════════

# Extended mapping value types:
#   str                → single param (non-layer or unscanned)
#   list[str]          → scanned layer param (stack axis=0), OR unscanned expert
#                        param (stack axis=0 → (n_experts, ...))
#   list[list[str]]    → scanned expert param (outer=experts axis=0,
#                        inner=layers axis=1 → (n_experts, n_layers, ...))
#
# Extended mapping key types:
#   str                → single Megatext param key
#   tuple[str, ...]    → composite: one HF source splits into multiple MT keys
#                        (used for GPT-OSS fused gate_up_proj)

def pad_vocab_size(vocab: int, multiple: int = 128) -> int:
    """Round vocab size up to the nearest multiple (default 128)."""
    return ((vocab + multiple - 1) // multiple) * multiple


MappingKey = str | tuple[str, ...]
MappingValue = str | list[str] | list[list[str]]
Mapping = dict[MappingKey, MappingValue]


def build_mapping(
    arch: ArchSpec,
    hf_config: Any,
    scan_layers: bool,
    builder: Callable[..., Mapping],
) -> Mapping:
    """Build Megatext key → HF key mapping using the given builder."""
    return builder(arch, hf_config, scan_layers)


def _cfg(hf_config: Any) -> Any:
    """Extract text config from potentially multimodal config."""
    return getattr(hf_config, "text_config", hf_config)


def _global_keys(
    prefix: str,
    tie_word_embeddings: bool,
    *,
    embed_key: str = "embed_tokens.weight",
    norm_key: str = "norm.weight",
    lm_head_key: str = "lm_head.weight",
) -> Mapping:
    """Non-layer params common to most models."""
    m: Mapping = {
        "params-token_embedder-embedding": f"{prefix}.{embed_key}" if prefix else embed_key,
        "params-decoder-decoder_norm-scale": f"{prefix}.{norm_key}" if prefix else norm_key,
    }
    if not tie_word_embeddings:
        m["params-decoder-logits_dense-kernel"] = lm_head_key
    return m


def _layer_mapping(
    components: list[tuple[str, str]],
    n_layers: int,
    scan_layers: bool,
    prefix: str,
    *,
    layer_fmt: str = "layers_{i}",
    scan_prefix: str = "layers",
) -> Mapping:
    """Build layer-param mapping from (mt_suffix, hf_suffix) pairs."""
    mapping: Mapping = {}
    for mt_suffix, hf_suffix in components:
        if scan_layers:
            mt_key = f"params-decoder-{scan_prefix}-{mt_suffix}"
            hf_keys = [
                f"{prefix}.layers.{i}.{hf_suffix}" for i in range(n_layers)
            ]
            mapping[mt_key] = hf_keys
        else:
            for i in range(n_layers):
                mt_key = f"params-decoder-{layer_fmt.format(i=i)}-{mt_suffix}"
                hf_key = f"{prefix}.layers.{i}.{hf_suffix}"
                mapping[mt_key] = hf_key
    return mapping


# ══════════════════════════════════════════════════════════════════════════════
# Transforms — tensor transforms for checkpoint conversion
# ══════════════════════════════════════════════════════════════════════════════

TransformFn = Callable[[np.ndarray, tuple], np.ndarray]


# ── Primitives ───────────────────────────────────────────────────────────────


def reshape_kernel(x: np.ndarray, shape: tuple) -> np.ndarray:
    """HF (out, in) → Megatext multi-head shape (transpose + reshape)."""
    return x.T.reshape(shape)


def reshape_kernel_inv(x: np.ndarray, shape: tuple) -> np.ndarray:
    """Megatext multi-head → HF (out, in) (reshape + transpose)."""
    out, inp = shape
    return x.reshape(inp, out).T


def simple_transpose(x: np.ndarray, shape: tuple) -> np.ndarray:
    """Swap last two dimensions. Works for 2D (.T) and higher-dim tensors."""
    if x.ndim <= 1:
        return x
    return np.swapaxes(x, -1, -2)


def permute_conv(x: np.ndarray, shape: tuple) -> np.ndarray:
    """Conv1d weight permutation: (K,1,C) ↔ (C,1,K)."""
    return x.transpose(2, 1, 0)


def pad_embedding(x: np.ndarray, shape: tuple) -> np.ndarray:
    """Pad vocab dim to target shape with zeros."""
    if x.shape == shape:
        return x
    pad_width = [(0, s - x_s) for s, x_s in zip(shape, x.shape)]
    return np.pad(x, pad_width, mode="constant")


def unpad_embedding(x: np.ndarray, shape: tuple) -> np.ndarray:
    """Truncate vocab dim back to HF size."""
    slices = tuple(slice(0, s) for s in shape)
    return x[slices]


def scale_embedding_fwd(x: np.ndarray, shape: tuple, hidden_size: int) -> np.ndarray:
    """HF→MT: x * sqrt(hidden_size)."""
    return x * math.sqrt(hidden_size)


def scale_embedding_inv(x: np.ndarray, shape: tuple, hidden_size: int) -> np.ndarray:
    """MT→HF: x / sqrt(hidden_size)."""
    return x / math.sqrt(hidden_size)


def scale_rmsnorm_fwd(x: np.ndarray, shape: tuple) -> np.ndarray:
    """HF→MT: x + 1."""
    return x + 1


def scale_rmsnorm_inv(x: np.ndarray, shape: tuple) -> np.ndarray:
    """MT→HF: x - 1."""
    return x - 1


def scale_query_fwd(x: np.ndarray, shape: tuple, head_dim: int) -> np.ndarray:
    """HF→MT: x / sqrt(head_dim)."""
    return x / math.sqrt(head_dim)


def scale_query_inv(x: np.ndarray, shape: tuple, head_dim: int) -> np.ndarray:
    """MT→HF: x * sqrt(head_dim)."""
    return x * math.sqrt(head_dim)


def reorder_rope_fwd(x: np.ndarray, shape: tuple, head_dim: int) -> np.ndarray:
    """HF→MT: concatenated RoPE → interleaved RoPE."""
    *prefix, n_heads, hd = x.shape
    half = hd // 2
    x = x.reshape(*prefix, n_heads, 2, half)
    x = x.transpose(*range(len(prefix)), len(prefix), len(prefix) + 2, len(prefix) + 1)
    return x.reshape(*prefix, n_heads, hd)


def reorder_rope_inv(x: np.ndarray, shape: tuple, head_dim: int) -> np.ndarray:
    """MT→HF: interleaved RoPE → concatenated RoPE."""
    *prefix, n_heads, hd = x.shape
    half = hd // 2
    x = x.reshape(*prefix, n_heads, half, 2)
    x = x.transpose(*range(len(prefix)), len(prefix), len(prefix) + 2, len(prefix) + 1)
    return x.reshape(*prefix, n_heads, hd)


def deinterleave(x: np.ndarray, shape: tuple) -> list[np.ndarray]:
    """Split fused gate_up tensor into [gate, up] via even/odd indexing."""
    if x.ndim >= 3:
        axis = 1
    elif x.ndim == 2:
        axis = 0
    else:
        axis = 0
    return [
        np.take(x, range(0, x.shape[axis], 2), axis=axis),
        np.take(x, range(1, x.shape[axis], 2), axis=axis),
    ]


def interleave(parts: list[np.ndarray], shape: tuple) -> np.ndarray:
    """Merge [gate, up] back into fused gate_up tensor via interleaving."""
    a, b = parts
    if a.ndim >= 3:
        axis = 1
    elif a.ndim == 2:
        axis = 0
    else:
        axis = 0

    fused_size = a.shape[axis] + b.shape[axis]
    out_shape = list(a.shape)
    out_shape[axis] = fused_size
    out = np.empty(out_shape, dtype=a.dtype)

    slices_a = [slice(None)] * a.ndim
    slices_b = [slice(None)] * a.ndim
    slices_a[axis] = slice(0, None, 2)
    slices_b[axis] = slice(1, None, 2)
    out[tuple(slices_a)] = a
    out[tuple(slices_b)] = b
    return out


def split_concat(x: np.ndarray, shape: tuple) -> list[np.ndarray]:
    """Split concatenated [gate; up] into [gate, up] along fused dim."""
    axis = 1 if x.ndim >= 3 else 0
    mid = x.shape[axis] // 2
    return [
        np.take(x, range(0, mid), axis=axis),
        np.take(x, range(mid, x.shape[axis]), axis=axis),
    ]


def merge_concat(parts: list[np.ndarray], shape: tuple) -> np.ndarray:
    """Merge [gate, up] back into concatenated [gate; up]."""
    axis = 1 if parts[0].ndim >= 3 else 0
    return np.concatenate(parts, axis=axis)


def deinterleave_last(x: np.ndarray, shape: tuple) -> list[np.ndarray]:
    """Split fused gate_up via even/odd on the last axis (GPT-OSS)."""
    return [x[..., ::2], x[..., 1::2]]


def interleave_last(parts: list[np.ndarray], shape: tuple) -> np.ndarray:
    """Merge [gate, up] back via interleaving on the last axis."""
    a, b = parts
    out = np.empty((*a.shape[:-1], a.shape[-1] + b.shape[-1]), dtype=a.dtype)
    out[..., ::2] = a
    out[..., 1::2] = b
    return out


def reshape_bias(x: np.ndarray, shape: tuple) -> np.ndarray:
    """Reshape bias: flatten or split into (heads, head_dim) ↔ (hidden,)."""
    return x.reshape(shape)


def _make_logits_transform(vocab_size: int, *, to_hf: bool) -> TransformFn:
    """Build logits_dense transform handling vocab padding.

    HF→MT: transpose (vocab,emb) → (emb,vocab), then pad vocab to 128-multiple.
    MT→HF: unpad vocab, then transpose (emb,vocab) → (vocab,emb).
    """
    padded = ((vocab_size + 127) // 128) * 128
    if to_hf:
        def fn(x: np.ndarray, shape: tuple) -> np.ndarray:
            unpadded = x[:, :vocab_size] if x.ndim == 2 else x[..., :vocab_size]
            return np.swapaxes(unpadded, -1, -2)
        return fn
    else:
        def fn(x: np.ndarray, shape: tuple) -> np.ndarray:
            t = np.swapaxes(x, -1, -2)  # (emb, vocab)
            if t.shape[-1] < padded:
                pad_width = [(0, 0)] * (t.ndim - 1) + [(0, padded - t.shape[-1])]
                return np.pad(t, pad_width, mode="constant")
            return t
        return fn


# ── Chain helper ─────────────────────────────────────────────────────────────


def _chain(fns: list[TransformFn]) -> TransformFn:
    """Chain multiple transform functions."""
    def chained(x: np.ndarray, shape: tuple) -> np.ndarray:
        for fn in fns:
            x = fn(x, shape)
        return x
    return chained


# ── Per-model transform builder helpers ──────────────────────────────────────


def _build_dense_kernel_transforms(
    kernel_suffixes: list[str],
    to_hf: bool,
) -> dict[str, TransformFn]:
    """Standard reshape_kernel transforms for a list of kernel suffixes."""
    fn = reshape_kernel_inv if to_hf else reshape_kernel
    return {k: fn for k in kernel_suffixes}


def _build_embedding_transforms(
    arch: ArchSpec,
    hidden_size: int,
    vocab_size: int,
    to_hf: bool,
) -> dict[str, TransformFn]:
    """Embedding pad/unpad + optional scaling."""
    transforms: dict[str, TransformFn] = {}
    if to_hf:
        chain: list[TransformFn] = []
        if arch.scale_embedding:
            chain.append(lambda x, s, hs=hidden_size: scale_embedding_inv(x, s, hs))
        chain.append(lambda x, s, v=vocab_size: unpad_embedding(x, (v, x.shape[1])))
        transforms["token_embedder-embedding"] = _chain(chain)
    else:
        chain = []
        if arch.scale_embedding:
            chain.append(lambda x, s, hs=hidden_size: scale_embedding_fwd(x, s, hs))
        chain.append(pad_embedding)
        transforms["token_embedder-embedding"] = _chain(chain)
    return transforms


def _build_rmsnorm_transforms(
    arch: ArchSpec,
    to_hf: bool,
    norm_suffixes: list[str],
) -> dict[str, TransformFn]:
    """RMSNorm ±1 transforms for Gemma-style models with folded offsets."""
    if not arch.scale_rmsnorm:
        return {}
    fn = scale_rmsnorm_inv if to_hf else scale_rmsnorm_fwd
    suffixes = list(norm_suffixes) + ["decoder-decoder_norm-scale"]
    return {s: fn for s in suffixes}


def build_transforms(
    arch: ArchSpec,
    hf_config_dict: dict[str, Any],
    *,
    to_hf: bool,
    builder: Callable[..., dict[str, TransformFn]],
) -> dict[str, TransformFn]:
    """Build Megatext param suffix → transform_fn mapping using the given builder."""
    return builder(arch, hf_config_dict, to_hf=to_hf)


def _match_transform(
    transforms: dict[str, TransformFn],
    mt_key: str,
) -> TransformFn | None:
    """Find the transform for a Megatext param key by suffix matching."""
    for suffix, fn in transforms.items():
        if mt_key.endswith(suffix):
            return fn
    return None
