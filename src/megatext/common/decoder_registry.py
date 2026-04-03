"""Decoder block registry — single source of truth for layer classes and norm types.

New decoder blocks only need to be registered here with a uniform interface:

    register(DecoderBlockType.MY_MODEL, DecoderBlockSpec(
        nnx_layer=my_model.MyDecoderLayer,
        nnx_scannable=my_model.MyScannableBlock,  # or None
        norm_type="rmsnorm",
    ))

The registry automatically:
- Resolves NNX layers (explicit scan body if provided, else scans nnx_layer directly)
- Generates Linen wrappers via to_linen_class() for models with only NNX classes
- Handles multi-layer models (e.g. DeepSeek dense+MoE) via nnx_layers/linen_layers lists

All registered decoder blocks are scan-compatible. `nnx_scannable` is only
needed when one scan iteration should represent multiple logical decoder
layers or otherwise requires a custom scan body.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from megatext.common.common_types import DecoderBlockType


@dataclass
class DecoderBlockSpec:
    """Specification for a decoder block type.

    For most models, provide nnx_layer.
    Provide nnx_scannable only when the scanned path needs an explicit scan
    body instead of directly scanning the single-layer class.
    For multi-stack models (e.g. DeepSeek), use nnx_layers_fn instead.
    """

    # Single logical decoder layer class.
    nnx_layer: type | None = None
    # Optional explicit scan body class. If None, nnx_layer is scanned directly.
    nnx_scannable: type | None = None
    # For multi-layer models: callable(config) -> list[type] (overrides nnx_layer/nnx_scannable)
    nnx_layers_fn: Callable[..., list[type]] | None = None
    linen_layers_fn: Callable[..., list[type]] | None = None
    # Optional per-layer init kwargs used by unscanned stacks.
    layer_kwargs_fn: Callable[[int, Any], dict[str, Any]] | None = None
    # Optional per-layer/per-block call kwargs from runtime context.
    layer_call_kwargs_fn: Callable[..., dict[str, Any]] | None = None
    # Norm type: "rmsnorm" (default), "gpt3_layernorm", "qwen3_next_rmsnorm"
    norm_type: str = "rmsnorm"


_REGISTRY: dict[DecoderBlockType, DecoderBlockSpec] = {}
_POPULATED = False


def register(block_type: DecoderBlockType, spec: DecoderBlockSpec):
    """Register a decoder block specification."""
    _REGISTRY[block_type] = spec


def get_spec(block_type: DecoderBlockType) -> DecoderBlockSpec:
    """Look up the spec for a decoder block type (lazy-populates on first call)."""
    global _POPULATED
    if not _POPULATED:
        _populate_registry()
        _POPULATED = True
    if block_type not in _REGISTRY:
        raise ValueError(
            f"Unknown decoder_block: {block_type.value!r}. "
            f"Registered: {sorted(k.value for k in _REGISTRY)}"
        )
    return _REGISTRY[block_type]


def get_nnx_layers(block_type: DecoderBlockType, config) -> list[type]:
    """Get NNX layer classes for a decoder block."""
    spec = get_spec(block_type)
    if spec.nnx_layers_fn is not None:
        return spec.nnx_layers_fn(config)
    if config.scan_layers and spec.nnx_scannable is not None:
        return [spec.nnx_scannable]
    return [spec.nnx_layer]


def get_linen_layers(block_type: DecoderBlockType, config) -> list[type]:
    """Get Linen layer classes for a decoder block."""
    spec = get_spec(block_type)
    if spec.linen_layers_fn is not None:
        return spec.linen_layers_fn(config)
    cls = spec.nnx_scannable if (config.scan_layers and spec.nnx_scannable) else spec.nnx_layer
    return [_to_linen(cls)]


def get_scannable_block_cls(block_type: DecoderBlockType, config) -> type | None:
    """Get the underlying NNX scannable block class when scan_layers=True."""
    spec = get_spec(block_type)
    if config.scan_layers and spec.nnx_scannable is not None:
        return spec.nnx_scannable
    return None


def get_scannable_block_layer_count(block_type: DecoderBlockType, config) -> int | None:
    """Get the number of logical layers represented by one scanned block iteration."""
    scannable_cls = get_scannable_block_cls(block_type, config)
    if scannable_cls is None:
        return None
    helper = getattr(scannable_cls, "scan_body_layer_count", None)
    if callable(helper):
        return int(helper(config))
    return None


def get_scan_iteration_count(block_type: DecoderBlockType, config, total_layers: int) -> int:
    """Convert logical layer count to scan iteration count for a decoder block."""
    block_layer_count = get_scannable_block_layer_count(block_type, config)
    if block_layer_count is None:
        return total_layers
    if total_layers % block_layer_count != 0:
        raise ValueError(
            f"{block_type} scan body covers {block_layer_count} layers, "
            f"but total_layers={total_layers} is not divisible by it."
        )
    return total_layers // block_layer_count


def get_layer_init_kwargs(block_type: DecoderBlockType, config, layer_index: int) -> dict[str, Any]:
    """Get per-layer constructor kwargs for a logical decoder layer."""
    spec = get_spec(block_type)
    if spec.layer_kwargs_fn is None:
        return {}
    return dict(spec.layer_kwargs_fn(layer_index, config))


def get_layer_call_kwargs(block_type: DecoderBlockType, config, **context) -> dict[str, Any]:
    """Get per-layer or per-block call kwargs from runtime context."""
    spec = get_spec(block_type)
    if spec.layer_call_kwargs_fn is None:
        return {}
    kwargs = spec.layer_call_kwargs_fn(config=config, **context)
    if kwargs is None:
        return {}
    return dict(kwargs)


# ── Linen wrapper cache ─────────────────────────────────────────────────────

_LINEN_CACHE: dict[type, type] = {}


def _to_linen(nnx_cls: type) -> type:
    """Convert an NNX class to a Linen class, caching the result."""
    if nnx_cls not in _LINEN_CACHE:
        from megatext.layers import nnx_wrappers
        from megatext.layers.initializers import variable_to_logically_partitioned
        _LINEN_CACHE[nnx_cls] = nnx_wrappers.to_linen_class(
            nnx_cls, base_metadata_fn=variable_to_logically_partitioned,
        )
    return _LINEN_CACHE[nnx_cls]


# ── Registry population ─────────────────────────────────────────────────────

def _populate_registry():
    """Lazy import and registration of all decoder blocks."""
    # pylint: disable=import-outside-toplevel
    from megatext.layers.nnx_decoders import NNXDecoderLayer
    from megatext.models import (
        deepseek, deepseek_batchsplit,
        gemma, gemma2, gemma4,
        gpt3, gpt_oss,
        llama2, llama4,
        mistral, mixtral,
        olmo3, qwen2, qwen3, qwen3_swa,
        simple_layer,
    )

    # ── Single-layer scan-body models (scan nnx_layer directly) ──────────

    register(DecoderBlockType.DEFAULT, DecoderBlockSpec(
        nnx_layer=NNXDecoderLayer,
    ))
    register(DecoderBlockType.LLAMA2, DecoderBlockSpec(
        nnx_layer=llama2.LlamaDecoderLayer,
    ))
    register(DecoderBlockType.MISTRAL, DecoderBlockSpec(
        nnx_layer=mistral.MistralDecoderLayer,
    ))
    register(DecoderBlockType.MIXTRAL, DecoderBlockSpec(
        nnx_layer=mixtral.MixtralDecoderLayer,
    ))
    register(DecoderBlockType.GEMMA, DecoderBlockSpec(
        nnx_layer=gemma.GemmaDecoderLayer,
    ))
    register(DecoderBlockType.GPT3, DecoderBlockSpec(
        nnx_layer=gpt3.Gpt3DecoderLayer,
        norm_type="gpt3_layernorm",
    ))
    register(DecoderBlockType.QWEN2, DecoderBlockSpec(
        nnx_layer=qwen2.Qwen2DecoderLayer if hasattr(qwen2, 'Qwen2DecoderLayer') else NNXDecoderLayer,
    ))
    register(DecoderBlockType.QWEN3, DecoderBlockSpec(
        nnx_layer=qwen3.Qwen3DecoderLayer,
    ))
    register(DecoderBlockType.QWEN3_MOE, DecoderBlockSpec(
        nnx_layer=qwen3.Qwen3MoeDecoderLayer,
    ))
    register(DecoderBlockType.SIMPLE, DecoderBlockSpec(
        nnx_layer=simple_layer.SimpleDecoderLayer,
    ))
    register(DecoderBlockType.SIMPLE_MLP, DecoderBlockSpec(
        nnx_layer=simple_layer.SimpleMlpDecoderLayer,
    ))

    # ── Explicit multi-layer scan-body models ────────────────────────────

    register(DecoderBlockType.GEMMA2, DecoderBlockSpec(
        nnx_layer=gemma2.Gemma2DecoderLayer,
        nnx_scannable=gemma2.Gemma2ScannableBlock,
        layer_kwargs_fn=gemma2._gemma2_layer_kwargs,
    ))
    register(DecoderBlockType.GEMMA4, DecoderBlockSpec(
        nnx_layer=gemma4.Gemma4DecoderLayer,
        nnx_scannable=gemma4.Gemma4ScannableBlock,
        layer_kwargs_fn=gemma4._gemma4_layer_kwargs,
        layer_call_kwargs_fn=gemma4._gemma4_layer_call_kwargs,
    ))

    register(DecoderBlockType.QWEN3_SWA, DecoderBlockSpec(
        nnx_layer=qwen3_swa.Qwen3SWADecoderLayer,
        nnx_scannable=qwen3_swa.Qwen3SWAScannableBlock,
        layer_kwargs_fn=qwen3_swa._qwen3_swa_layer_kwargs,
    ))
    register(DecoderBlockType.QWEN3_NEXT, DecoderBlockSpec(
        nnx_layer=qwen3.Qwen3NextDecoderLayer,
        nnx_scannable=qwen3.Qwen3NextScannableBlock,
        layer_kwargs_fn=qwen3._qwen3_next_layer_kwargs,
        norm_type="qwen3_next_rmsnorm",
    ))
    register(DecoderBlockType.GPT_OSS, DecoderBlockSpec(
        nnx_layer=gpt_oss.GptOssDecoderLayer,
        nnx_scannable=gpt_oss.GptOssScannableBlock,
        layer_kwargs_fn=gpt_oss._gpt_oss_layer_kwargs,
    ))
    register(DecoderBlockType.LLAMA4, DecoderBlockSpec(
        nnx_layer=llama4.Llama4DecoderLayer,
        nnx_scannable=llama4.Llama4ScannableBlock,
        layer_kwargs_fn=llama4._llama4_layer_kwargs,
    ))
    register(DecoderBlockType.OLMO3, DecoderBlockSpec(
        nnx_layer=olmo3.Olmo3DecoderLayer,
        nnx_scannable=olmo3.Olmo3ScannableBlock,
        layer_kwargs_fn=olmo3._olmo3_layer_kwargs,
    ))

    # ── Multi-layer models (custom resolution) ───────────────────────────

    def _deepseek_nnx(cfg):
        if cfg.use_batch_split_schedule:
            return [deepseek_batchsplit.DeepSeekDenseLayer, deepseek_batchsplit.DeepSeekMoELayer]
        return [deepseek.DeepSeekDenseLayer, deepseek.DeepSeekMoELayer]

    def _deepseek_linen(cfg):
        if cfg.use_batch_split_schedule:
            return [_to_linen(deepseek_batchsplit.DeepSeekDenseLayer), _to_linen(deepseek_batchsplit.DeepSeekMoELayer)]
        return [_to_linen(deepseek.DeepSeekDenseLayer), _to_linen(deepseek.DeepSeekMoELayer)]

    register(DecoderBlockType.DEEPSEEK, DecoderBlockSpec(
        nnx_layers_fn=_deepseek_nnx,
        linen_layers_fn=_deepseek_linen,
    ))
