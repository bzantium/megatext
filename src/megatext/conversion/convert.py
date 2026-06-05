"""Bidirectional HuggingFace <-> Megatext checkpoint conversion.

Uses Transformers for HF config/model I/O, applies Megatext-specific
mapping + transforms, restores Megatext checkpoints from Orbax, and writes
HF checkpoints via direct sharded safetensors save with meta-device
validation.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from typing import Any

import numpy as np
from tqdm import tqdm

from megatext.utils import logging as max_logging
from megatext.utils import storage as gcs_utils
from megatext.conversion.io_megatext import (
    load_megatext_checkpoint,
    save_megatext_checkpoint,
)
from megatext.conversion.utils import (
    ArchSpec,
    Mapping,
    _match_transform,
    build_mapping,
    build_transforms,
    deinterleave,
    interleave,
    merge_concat,
    split_concat,
    deinterleave_last,
    interleave_last,
    pad_vocab_size,
)
from megatext.conversion.models import (
    ARCH_SPECS,
    MAPPING_BUILDERS,
    TRANSFORM_BUILDERS,
    SHAPE_BUILDERS,
    resolve,
)


def _resolve_model_type(hf_config: Any) -> str:
    """Extract runtime HF model_type, checking root config first then text_config."""
    mt = getattr(hf_config, "model_type", None)
    if mt and mt in ARCH_SPECS:
        return mt
    cfg = getattr(hf_config, "text_config", hf_config)
    mt = getattr(cfg, "model_type", None)
    if mt and mt in ARCH_SPECS:
        return mt
    if mt:
        return mt
    raise ValueError("HF config has no model_type")


def _resolve_conversion_model_type(hf_config: Any) -> str:
    """Resolve conversion family from HF config.

    This may differ from the runtime HF model_type. For example, Qwen3 SWA
    checkpoints use the standard HF Qwen3 runtime config/model class, but need
    Qwen3-SWA-specific Megatext checkpoint mapping and shape rules.
    """
    runtime_model_type = _resolve_model_type(hf_config)
    cfg = getattr(hf_config, "text_config", hf_config)

    if runtime_model_type == "qwen3":
        layer_types = list(getattr(cfg, "layer_types", []) or [])
        if layer_types and "sliding_attention" in layer_types and "full_attention" in layer_types:
            return "qwen3_swa"

    return runtime_model_type


def _to_numpy(tensor: "torch.Tensor") -> np.ndarray:
    """Convert PyTorch tensor to numpy, handling bfloat16."""
    import torch

    if tensor.dtype == torch.bfloat16:
        import ml_dtypes

        return tensor.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    return tensor.numpy()


def _to_torch(x: np.ndarray) -> "torch.Tensor":
    """Convert numpy array to PyTorch tensor, handling bfloat16."""
    import torch

    if hasattr(x.dtype, "name") and x.dtype.name == "bfloat16":
        arr = x.view(np.uint16)
        if not arr.flags.writeable:
            arr = arr.copy()
        return torch.from_numpy(np.ascontiguousarray(arr)).view(torch.bfloat16)
    x = np.ascontiguousarray(x)
    if not x.flags.writeable:
        x = x.copy()
    return torch.from_numpy(x)


def _load_hf_config(
    hf_config_path: str, hf_token: str | None = None
) -> Any:
    """Load the pre-prepared HF runtime config/template for conversion."""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(
        hf_config_path, token=hf_token, trust_remote_code=True
    )


def _normalize_path(path: str) -> str:
    if path.startswith("gs://"):
        return path
    return os.path.abspath(path)


def _resolve_megatext_checkpoint_input(
    megatext_model_path: str,
    checkpoint_step: int | None = None,
) -> tuple[str, int | None]:
    """Resolve a Megatext checkpoint root + optional step from user input.

    Accepted forms:
    - `<checkpoint_root>`: convert latest step unless `checkpoint_step` is set
    - `<checkpoint_root>/<step>`: convert that step
    - `<checkpoint_root>/<step>/items`: convert that step
    """
    path = _normalize_path(megatext_model_path).rstrip("/")
    items_match = re.match(r"^(?P<root>.+)/(?P<step>\d+)/items$", path)
    if items_match:
        resolved_step = int(items_match.group("step"))
        if checkpoint_step is not None and checkpoint_step != resolved_step:
            raise ValueError(
                "checkpoint_step does not match step encoded in megatext_model_path."
            )
        return items_match.group("root"), resolved_step

    step_match = re.match(r"^(?P<root>.+)/(?P<step>\d+)$", path)
    if step_match:
        resolved_step = int(step_match.group("step"))
        if checkpoint_step is not None and checkpoint_step != resolved_step:
            raise ValueError(
                "checkpoint_step does not match step encoded in megatext_model_path."
            )
        return step_match.group("root"), resolved_step

    return path, checkpoint_step


def _is_multimodal_hf_model(model_type: str, hf_config: Any) -> bool:
    return model_type == "gemma4" and getattr(hf_config, "vision_config", None) is not None


def _load_hf_model_for_conversion(
    hf_model_path: str,
    hf_config: Any,
    *,
    hf_token: str | None = None,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    common_kwargs = dict(
        token=hf_token,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model_type = _resolve_model_type(hf_config)
    if _is_multimodal_hf_model(model_type, hf_config):
        return AutoModelForImageTextToText.from_pretrained(hf_model_path, **common_kwargs)
    return AutoModelForCausalLM.from_pretrained(hf_model_path, **common_kwargs)


def _init_hf_model_for_save(hf_config: Any):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    model_type = _resolve_model_type(hf_config)
    if _is_multimodal_hf_model(model_type, hf_config):
        return AutoModelForImageTextToText.from_config(hf_config, torch_dtype=torch.bfloat16)
    return AutoModelForCausalLM.from_config(hf_config, torch_dtype=torch.bfloat16)


def _init_hf_model_for_validation(hf_config: Any):
    import torch

    with torch.device("meta"):
        return _init_hf_model_for_save(hf_config)


# ── Shape computation ────────────────────────────────────────────────────────


def compute_megatext_shapes(
    hf_config: Any, arch: ArchSpec, scan_layers: bool,
    *, tie_word_embeddings: bool = False,
) -> dict[str, tuple]:
    """Compute Megatext param shapes from HF config dims."""
    builder = SHAPE_BUILDERS.get(arch.model_type)
    if builder is not None:
        return builder(hf_config, arch, scan_layers, tie_word_embeddings=tie_word_embeddings)
    return _compute_dense_shapes(hf_config, arch, scan_layers, tie_word_embeddings=tie_word_embeddings)


def _compute_dense_shapes(
    hf_config: Any, arch: ArchSpec, scan_layers: bool,
    *, tie_word_embeddings: bool = False,
) -> dict[str, tuple]:
    """Shape computation for dense models (qwen3, llama, gemma4)."""
    cfg = getattr(hf_config, "text_config", hf_config)
    emb = cfg.hidden_size
    mlp = cfg.intermediate_size
    nq = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", emb // nq)
    vocab = cfg.vocab_size
    n_layers = cfg.num_hidden_layers
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
        "mlp-wi_0-kernel": (*lp, emb, mlp),
        "mlp-wi_1-kernel": (*lp, emb, mlp),
        "mlp-wo-kernel": (*lp, mlp, emb),
    }

    norm_shape = (*lp, emb)
    if arch.has_post_attn_norm:
        kernel_shapes["pre_self_attention_norm-scale"] = norm_shape
        kernel_shapes["post_self_attention_norm-scale"] = norm_shape
    else:
        kernel_shapes["pre_self_attention_layer_norm-scale"] = norm_shape
        kernel_shapes["post_self_attention_layer_norm-scale"] = norm_shape

    if arch.has_qk_norm:
        kernel_shapes["self_attention-query_norm-scale"] = (*lp, hd)
        kernel_shapes["self_attention-key_norm-scale"] = (*lp, hd)

    if arch.has_pre_ffw_norm:
        kernel_shapes["pre_ffw_norm-scale"] = norm_shape
    if arch.has_post_ffw_norm:
        kernel_shapes["post_ffw_norm-scale"] = norm_shape

    for suffix, shape in kernel_shapes.items():
        if scan_layers:
            shapes[f"params-decoder-layers-{suffix}"] = shape
        else:
            for i in range(n_layers):
                shapes[f"params-decoder-layers_{i}-{suffix}"] = shape

    return shapes


# ── Composite split/merge dispatch ────────────────────────────────────────────

_COMPOSITE_SPLIT_FNS = {
    "deinterleave": deinterleave,
    "concat": split_concat,
    "interleave_last": deinterleave_last,
}

_COMPOSITE_MERGE_FNS = {
    "deinterleave": interleave,
    "concat": merge_concat,
    "interleave_last": interleave_last,
}

def _get_composite_split_fn(arch: ArchSpec):
    return _COMPOSITE_SPLIT_FNS[arch.composite_split]


def _get_composite_merge_fn(arch: ArchSpec):
    return _COMPOSITE_MERGE_FNS[arch.composite_split]


# ── Conversion loops ─────────────────────────────────────────────────────────


def hf_to_megatext(
    hf_model_path: str,
    output_dir: str,
    scan_layers: bool = True,
    hf_token: str | None = None,
) -> str:
    """Convert a HuggingFace checkpoint to Megatext format.

    Args:
        hf_model_path: Pre-prepared HF model/config template path or HF hub ID.
            This supplies the runtime HF config and tokenizer artifacts used
            as the conversion source.
        output_dir: Directory to save the converted Megatext checkpoint.
        scan_layers: Whether to use scanned layers (default True for training).
        hf_token: HuggingFace access token for gated models.

    Returns:
        Path to the converted checkpoint directory.
    """
    hf_config = _load_hf_config(hf_model_path, hf_token)
    model_type = _resolve_conversion_model_type(hf_config)
    cfg = getattr(hf_config, "text_config", hf_config)

    arch = resolve(model_type)
    tie = getattr(cfg, "tie_word_embeddings", False)
    mapping_builder = MAPPING_BUILDERS[model_type]
    transform_builder = TRANSFORM_BUILDERS[model_type]
    mapping = build_mapping(arch, hf_config, scan_layers, mapping_builder)
    transforms = build_transforms(arch, hf_config.to_dict(), to_hf=False, builder=transform_builder)
    shapes = compute_megatext_shapes(hf_config, arch, scan_layers, tie_word_embeddings=tie)

    t0 = time.time()
    max_logging.log(f"Loading HF model {hf_model_path} (model_type={model_type})...")
    model = _load_hf_model_for_conversion(hf_model_path, hf_config, hf_token=hf_token)
    state_dict = model.state_dict()
    del model
    max_logging.log(f"HF model loaded in {time.time() - t0:.1f}s ({len(state_dict)} tensors)")

    mt_weights: dict[str, np.ndarray] = {}
    split_fn = _get_composite_split_fn(arch)
    t1 = time.time()
    _convert_hf_to_mt(mapping, transforms, shapes, state_dict, mt_weights,
                      composite_split_fn=split_fn)
    max_logging.log(f"Tensor conversion done in {time.time() - t1:.1f}s ({len(mt_weights)} params)")

    del state_dict

    output_dir = _normalize_path(output_dir)
    t2 = time.time()
    max_logging.log(f"Saving Megatext checkpoint to {output_dir}...")
    checkpoint_dir = save_megatext_checkpoint(mt_weights, output_dir)
    max_logging.log(f"Checkpoint saved in {time.time() - t2:.1f}s")
    max_logging.log(f"Conversion complete in {time.time() - t0:.1f}s: {checkpoint_dir}")
    return checkpoint_dir


def _convert_hf_to_mt(
    mapping: Mapping,
    transforms: dict,
    shapes: dict[str, tuple],
    state_dict: dict,
    mt_weights: dict[str, np.ndarray],
    *,
    composite_split_fn=deinterleave,
) -> None:
    """Core HF→MT conversion loop handling all mapping value types."""
    for mt_key, hf_keys in tqdm(mapping.items(), desc="HF → Megatext", unit="param"):
        if isinstance(mt_key, tuple):
            _convert_composite_hf_to_mt(
                mt_key, hf_keys, transforms, shapes, state_dict, mt_weights,
                split_fn=composite_split_fn,
            )
            continue

        target_shape = shapes.get(mt_key)
        xform = _match_transform(transforms, mt_key)

        if isinstance(hf_keys, str):
            tensor = _to_numpy(state_dict.pop(hf_keys))
            mt_weights[mt_key] = xform(tensor, target_shape) if xform else tensor

        elif isinstance(hf_keys, list) and len(hf_keys) > 0 and isinstance(hf_keys[0], list):
            expert_stacks = []
            for expert_layer_keys in hf_keys:
                layer_tensors = []
                for k in expert_layer_keys:
                    t = _to_numpy(state_dict.pop(k))
                    layer_tensors.append(xform(t, _per_tensor_shape(target_shape, 2)) if xform else t)
                expert_stacks.append(np.stack(layer_tensors, axis=0))
            mt_weights[mt_key] = np.stack(expert_stacks, axis=0)

        elif isinstance(hf_keys, list):
            slices = []
            slice_shape = _per_tensor_shape(target_shape, 1) if target_shape else None
            for k in hf_keys:
                t = _to_numpy(state_dict.pop(k))
                slices.append(xform(t, slice_shape) if xform else t)
            mt_weights[mt_key] = np.stack(slices, axis=0)


def _convert_composite_hf_to_mt(
    mt_keys: tuple[str, ...],
    hf_keys: str | list[str],
    transforms: dict,
    shapes: dict[str, tuple],
    state_dict: dict,
    mt_weights: dict[str, np.ndarray],
    *,
    split_fn=deinterleave,
) -> None:
    """Handle composite keys (e.g., fused gate_up_proj → wi_0, wi_1)."""
    xforms = [_match_transform(transforms, k) for k in mt_keys]

    if isinstance(hf_keys, list):
        per_key_slices: dict[str, list[np.ndarray]] = {k: [] for k in mt_keys}
        for hf_key in hf_keys:
            tensor = _to_numpy(state_dict.pop(hf_key))
            parts = split_fn(tensor, ())
            for mt_k, part, xf in zip(mt_keys, parts, xforms):
                target = shapes.get(mt_k)
                slice_shape = _per_tensor_shape(target, 1) if target else None
                per_key_slices[mt_k].append(xf(part, slice_shape) if xf else part)
        for mt_k in mt_keys:
            mt_weights[mt_k] = np.stack(per_key_slices[mt_k], axis=0)
    else:
        tensor = _to_numpy(state_dict.pop(hf_keys))
        parts = split_fn(tensor, ())
        for mt_k, part, xf in zip(mt_keys, parts, xforms):
            target = shapes.get(mt_k)
            mt_weights[mt_k] = xf(part, target) if xf else part


def _per_tensor_shape(full_shape: tuple | None, n_stacked_dims: int) -> tuple | None:
    """Strip the first n stacked dims from shape (for per-tensor transforms)."""
    if full_shape is None:
        return None
    return full_shape[n_stacked_dims:]


def megatext_to_hf(
    megatext_model_path: str,
    output_dir: str,
    hf_config_path: str,
    scan_layers: bool = True,
    hf_token: str | None = None,
    checkpoint_step: int | None = None,
    tokenizer_path: str | None = None,
) -> str:
    """Convert a Megatext checkpoint to HuggingFace format.

    Args:
        megatext_model_path: Megatext checkpoint root, step directory, or
            `items/` directory. If a step is encoded in the path, a separate
            checkpoint_step is not required.
        output_dir: Directory to save the HF checkpoint.
        hf_config_path: Pre-prepared HF model/config template path or HF hub ID.
            This supplies the runtime HF config/model class and tokenizer
            artifacts for the exported checkpoint. The conversion mapping
            family may differ from the runtime HF model_type.
        scan_layers: Whether the checkpoint uses scanned layers.
        hf_token: HuggingFace access token for gated models.
        checkpoint_step: Optional Megatext checkpoint step override. This is
            only needed when megatext_model_path points at a checkpoint root.
        tokenizer_path: Optional path or HF hub ID to load the tokenizer from.
            If omitted, the tokenizer is copied from hf_config_path.

    Returns:
        Path to the converted HuggingFace checkpoint directory.
    """
    max_logging.log(f"Loading HF config from {hf_config_path}...")
    hf_config = _load_hf_config(hf_config_path, hf_token)
    model_type = _resolve_conversion_model_type(hf_config)
    cfg = getattr(hf_config, "text_config", hf_config)
    max_logging.log(f"HF config loaded (model_type={model_type})")

    arch = resolve(model_type)
    tie = getattr(cfg, "tie_word_embeddings", False)

    max_logging.log(f"Building mapping and transforms for {model_type}...")
    mapping_builder = MAPPING_BUILDERS[model_type]
    transform_builder = TRANSFORM_BUILDERS[model_type]
    mapping = build_mapping(arch, hf_config, scan_layers, mapping_builder)
    transforms = build_transforms(arch, hf_config.to_dict(), to_hf=True, builder=transform_builder)
    shapes = compute_megatext_shapes(hf_config, arch, scan_layers, tie_word_embeddings=tie)
    max_logging.log(f"Mapping ready ({len(mapping)} entries, scan_layers={scan_layers})")

    max_logging.log(f"Resolving Megatext checkpoint path...")
    checkpoint_path, checkpoint_step = _resolve_megatext_checkpoint_input(
        megatext_model_path,
        checkpoint_step,
    )
    output_dir = _normalize_path(output_dir)

    hf_state_dict: dict[str, Any] = {}
    merge_fn = _get_composite_merge_fn(arch)
    t0 = time.time()
    max_logging.log(
        f"Loading Megatext checkpoint from {checkpoint_path} "
        f"(step={checkpoint_step if checkpoint_step is not None else 'latest'})..."
    )
    mt_weights = load_megatext_checkpoint(checkpoint_path, step=checkpoint_step)
    max_logging.log(f"Checkpoint loaded in {time.time() - t0:.1f}s ({len(mt_weights)} params)")

    t1 = time.time()
    _convert_mt_to_hf(
        mapping,
        transforms,
        mt_weights,
        hf_state_dict,
        hf_config,
        composite_merge_fn=merge_fn,
    )
    max_logging.log(f"Tensor conversion done in {time.time() - t1:.1f}s ({len(hf_state_dict)} tensors)")
    del mt_weights

    t2 = time.time()
    max_logging.log(f"Validating HF state dict...")
    hf_state_dict = _validate_hf_state_dict_for_save(hf_state_dict, hf_config)
    max_logging.log(f"Validation done in {time.time() - t2:.1f}s")

    t3 = time.time()
    max_logging.log(f"Saving HF model to {output_dir}...")
    tokenizer_src = tokenizer_path or hf_config_path
    if output_dir.startswith("gs://"):
        with tempfile.TemporaryDirectory(prefix="megatext-hf-convert-") as tmp_output_dir:
            _save_hf_checkpoint_direct(tmp_output_dir, hf_state_dict, hf_config)
            _copy_tokenizer(tokenizer_src, tmp_output_dir, hf_token)
            gcs_utils.upload_dump(tmp_output_dir, output_dir, delete_local_after=False)
    else:
        _save_hf_checkpoint_direct(output_dir, hf_state_dict, hf_config)
        _copy_tokenizer(tokenizer_src, output_dir, hf_token)
    max_logging.log(f"Save done in {time.time() - t3:.1f}s")

    max_logging.log(f"Conversion complete in {time.time() - t0:.1f}s: {output_dir}")
    return output_dir


def _convert_mt_to_hf(
    mapping: Mapping,
    transforms: dict,
    mt_weights: dict[str, np.ndarray],
    hf_state_dict: dict,
    hf_config: Any,
    *,
    composite_merge_fn=interleave,
) -> None:
    """Core MT→HF conversion loop."""
    for mt_key, hf_keys in tqdm(mapping.items(), desc="Megatext → HF", unit="param"):
        if isinstance(mt_key, tuple):
            _convert_composite_mt_to_hf(
                mt_key, hf_keys, transforms, mt_weights, hf_state_dict,
                merge_fn=composite_merge_fn,
            )
            continue

        xform = _match_transform(transforms, mt_key)

        if isinstance(hf_keys, str):
            tensor = mt_weights.pop(mt_key)
            hf_shape = _infer_hf_shape_from_key(hf_keys, hf_config)
            result = xform(tensor, hf_shape) if xform else tensor
            hf_state_dict[hf_keys] = _to_torch(result)

        elif isinstance(hf_keys, list) and len(hf_keys) > 0 and isinstance(hf_keys[0], list):
            stacked = mt_weights.pop(mt_key)
            for e_idx, expert_layer_keys in enumerate(hf_keys):
                for l_idx, hf_key in enumerate(expert_layer_keys):
                    tensor = stacked[e_idx, l_idx]
                    hf_shape = _infer_hf_shape_from_key(hf_key, hf_config)
                    result = xform(tensor, hf_shape) if xform else tensor
                    hf_state_dict[hf_key] = _to_torch(result)

        elif isinstance(hf_keys, list):
            stacked = mt_weights.pop(mt_key)
            for i, hf_key in enumerate(hf_keys):
                tensor = stacked[i]
                hf_shape = _infer_hf_shape_from_key(hf_key, hf_config)
                result = xform(tensor, hf_shape) if xform else tensor
                hf_state_dict[hf_key] = _to_torch(result)


def _convert_composite_mt_to_hf(
    mt_keys: tuple[str, ...],
    hf_keys: str | list[str],
    transforms: dict,
    mt_weights: dict[str, np.ndarray],
    hf_state_dict: dict,
    *,
    merge_fn=interleave,
) -> None:
    """Handle composite MT→HF: merge multiple MT keys into one HF key."""
    if isinstance(hf_keys, list):
        parts_per_layer: list[list[np.ndarray]] = []
        mt_arrays = [mt_weights.pop(k) for k in mt_keys]
        xforms = [_match_transform(transforms, k) for k in mt_keys]
        n_layers = mt_arrays[0].shape[0]
        for layer in range(n_layers):
            layer_parts = []
            for arr, xf in zip(mt_arrays, xforms):
                t = arr[layer]
                layer_parts.append(xf(t, ()) if xf else t)
            parts_per_layer.append(layer_parts)
        for layer, hf_key in enumerate(hf_keys):
            hf_state_dict[hf_key] = _to_torch(
                merge_fn(parts_per_layer[layer], ())
            )
    else:
        parts = []
        for k in mt_keys:
            t = mt_weights.pop(k)
            xf = _match_transform(transforms, k)
            parts.append(xf(t, ()) if xf else t)
        hf_state_dict[hf_keys] = _to_torch(merge_fn(parts, ()))


def _maybe_materialize_tied_hf_weights(
    hf_state_dict: dict[str, Any],
    hf_config: Any,
) -> dict[str, Any]:
    """Fill common tied HF weights without duplicating storage unnecessarily."""
    cfg = getattr(hf_config, "text_config", hf_config)
    if not getattr(cfg, "tie_word_embeddings", False):
        return hf_state_dict
    if "lm_head.weight" not in hf_state_dict and "model.embed_tokens.weight" in hf_state_dict:
        hf_state_dict = dict(hf_state_dict)
        hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]
    return hf_state_dict


def _validate_hf_state_dict_for_save(
    hf_state_dict: dict[str, Any],
    hf_config: Any,
) -> dict[str, Any]:
    """Validate converted HF weights against a lightweight runtime model skeleton."""
    hf_state_dict = _maybe_materialize_tied_hf_weights(hf_state_dict, hf_config)
    model = _init_hf_model_for_validation(hf_config)
    expected_state_dict = model.state_dict()
    expected_keys = list(expected_state_dict.keys())
    expected_key_set = set(expected_keys)
    actual_key_set = set(hf_state_dict)

    missing = sorted(expected_key_set - actual_key_set)
    unexpected = sorted(actual_key_set - expected_key_set)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing[:10]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:10]}")
        raise ValueError("Converted HF state_dict keys do not match runtime model: " + ", ".join(details))

    ordered_state_dict = {key: hf_state_dict[key] for key in expected_keys}
    try:
        model.load_state_dict(ordered_state_dict, strict=False, assign=True)
    except RuntimeError as exc:
        raise ValueError("Converted HF state_dict failed runtime-model validation") from exc
    return ordered_state_dict


def _save_hf_checkpoint_direct(
    output_dir: str,
    hf_state_dict: dict[str, Any],
    hf_config: Any,
) -> None:
    """Write HF config + sharded safetensors directly without materializing a full HF model."""
    from huggingface_hub import save_torch_state_dict
    from transformers import GenerationConfig

    os.makedirs(output_dir, exist_ok=True)
    hf_config.save_pretrained(output_dir)
    try:
        GenerationConfig.from_model_config(hf_config).save_pretrained(output_dir)
    except Exception:
        max_logging.debug("Could not derive generation config from HF config (non-fatal)")

    save_torch_state_dict(
        hf_state_dict,
        output_dir,
        safe_serialization=True,
        metadata={"format": "pt"},
    )


def _infer_hf_shape_from_key(hf_key: str, hf_config: Any) -> tuple:
    """Infer HF tensor shape from key name and config."""
    cfg = getattr(hf_config, "text_config", hf_config)
    vcfg = getattr(hf_config, "vision_config", None)
    emb = cfg.hidden_size
    mlp = getattr(cfg, "intermediate_size", 0)
    nq = cfg.num_attention_heads
    nkv = getattr(cfg, "num_key_value_heads", nq)
    hd = getattr(cfg, "head_dim", emb // nq if nq else 0)
    vocab = cfg.vocab_size

    q_hd_mult = 2 if getattr(cfg, "full_attention_interval", 0) > 0 else 1

    model_type = getattr(hf_config, "model_type", getattr(cfg, "model_type", ""))
    if model_type == "gemma4" and vcfg is not None:
        hidden_vit = getattr(vcfg, "hidden_size", 0)
        mlp_vit = getattr(vcfg, "intermediate_size", 0)
        num_heads_vit = getattr(vcfg, "num_attention_heads", 0)
        hd_vit = getattr(vcfg, "head_dim", hidden_vit // num_heads_vit if num_heads_vit else 0)
        patch_size = getattr(vcfg, "patch_size", 0)
        pos_emb = getattr(vcfg, "position_embedding_size", 0)
        if hf_key == "model.vision_tower.patch_embedder.input_proj.weight":
            return (hidden_vit, patch_size * patch_size * 3)
        if hf_key == "model.vision_tower.patch_embedder.position_embedding_table":
            return (2, pos_emb, hidden_vit)
        if hf_key == "model.embed_vision.embedding_projection.weight":
            return (emb, hidden_vit)
        if hf_key in ("model.vision_tower.std_scale", "model.vision_tower.std_bias"):
            return (hidden_vit,)
        if "vision_tower.encoder.layers." in hf_key:
            if "self_attn.q_proj.linear.weight" in hf_key:
                return (num_heads_vit * hd_vit, hidden_vit)
            if "self_attn.k_proj.linear.weight" in hf_key:
                return (num_heads_vit * hd_vit, hidden_vit)
            if "self_attn.v_proj.linear.weight" in hf_key:
                return (num_heads_vit * hd_vit, hidden_vit)
            if "self_attn.o_proj.linear.weight" in hf_key:
                return (hidden_vit, num_heads_vit * hd_vit)
            if "self_attn.q_norm.weight" in hf_key or "self_attn.k_norm.weight" in hf_key:
                return (hd_vit,)
            if "input_layernorm.weight" in hf_key or "post_attention_layernorm.weight" in hf_key:
                return (hidden_vit,)
            if "pre_feedforward_layernorm.weight" in hf_key or "post_feedforward_layernorm.weight" in hf_key:
                return (hidden_vit,)
            if "mlp.gate_proj.linear.weight" in hf_key or "mlp.up_proj.linear.weight" in hf_key:
                return (mlp_vit, hidden_vit)
            if "mlp.down_proj.linear.weight" in hf_key:
                return (hidden_vit, mlp_vit)

    layer_match = re.search(r"\.layers\.(\d+)\.", hf_key)
    if model_type in ("gemma4", "gemma4_text") and layer_match:
        layer_idx = int(layer_match.group(1))
        layer_types = list(getattr(cfg, "layer_types", []) or [])
        is_global = (
            layer_types[layer_idx % len(layer_types)] in ("full_attention", "global")
            if layer_types
            else layer_idx % 6 == 5
        )
        q_hd = getattr(cfg, "global_head_dim", hd) if is_global else hd
        kv_hd = getattr(cfg, "global_head_dim", hd) if is_global else hd
        nkv_this = getattr(cfg, "num_global_key_value_heads", nkv) if is_global else nkv
        expert_mlp = getattr(cfg, "expert_intermediate_size", getattr(cfg, "moe_intermediate_size", mlp))
        if "self_attn.q_proj.weight" in hf_key:
            return (nq * q_hd, emb)
        if "self_attn.k_proj.weight" in hf_key:
            return (nkv_this * kv_hd, emb)
        if "self_attn.v_proj.weight" in hf_key:
            return (nkv_this * kv_hd, emb)
        if "self_attn.o_proj.weight" in hf_key:
            return (emb, nq * q_hd)
        if "self_attn.q_norm.weight" in hf_key:
            return (q_hd,)
        if "self_attn.k_norm.weight" in hf_key or "self_attn.v_norm.weight" in hf_key:
            return (kv_hd,)
        if "input_layernorm.weight" in hf_key or "post_attention_layernorm.weight" in hf_key:
            return (emb,)
        if "pre_feedforward_layernorm.weight" in hf_key or "post_feedforward_layernorm.weight" in hf_key:
            return (emb,)
        if "pre_feedforward_layernorm_2.weight" in hf_key:
            return (emb,)
        if "post_feedforward_layernorm_1.weight" in hf_key:
            return (emb,)
        if "post_feedforward_layernorm_2.weight" in hf_key:
            return (emb,)
        if hf_key.endswith("router.scale"):
            return (emb,)
        if hf_key.endswith("router.per_expert_scale"):
            return (getattr(cfg, "num_experts", 0),)
        if hf_key.endswith("layer_scalar"):
            return (1,)
        if hf_key.endswith("router.proj.weight"):
            return (getattr(cfg, "num_experts", 0), emb)
        if hf_key.endswith("experts.gate_up_proj"):
            return (getattr(cfg, "num_experts", 0), 2 * expert_mlp, emb)
        if hf_key.endswith("experts.down_proj"):
            return (getattr(cfg, "num_experts", 0), emb, expert_mlp)

    if "q_proj.weight" in hf_key or "wq.weight" in hf_key:
        return (nq * q_hd_mult * hd, emb)
    if "k_proj.weight" in hf_key or "wk.weight" in hf_key:
        return (nkv * hd, emb)
    if "v_proj.weight" in hf_key or "wv.weight" in hf_key:
        return (nkv * hd, emb)
    if "o_proj.weight" in hf_key or "wo.weight" in hf_key:
        return (emb, nq * hd)

    if "q_proj.bias" in hf_key:
        return (nq * q_hd_mult * hd,)
    if "k_proj.bias" in hf_key:
        return (nkv * hd,)
    if "v_proj.bias" in hf_key:
        return (nkv * hd,)
    if "o_proj.bias" in hf_key:
        return (emb,)

    if "self_attn.sinks" in hf_key:
        return (nq,)

    # MLA attention (DeepSeek)
    q_lora_rank = getattr(cfg, "q_lora_rank", 0)
    kv_lora_rank = getattr(cfg, "kv_lora_rank", 0)
    qk_rope_hd = getattr(cfg, "qk_rope_head_dim", 0)
    qk_nope_hd = getattr(cfg, "qk_nope_head_dim", 0)
    v_head_dim = getattr(cfg, "v_head_dim", hd)
    if "q_a_proj.weight" in hf_key:
        return (q_lora_rank, emb)
    if "q_b_proj.weight" in hf_key:
        return (nq * (qk_nope_hd + qk_rope_hd), q_lora_rank)
    if "kv_a_proj_with_mqa.weight" in hf_key:
        return (kv_lora_rank + qk_rope_hd, emb)
    if "kv_b_proj.weight" in hf_key:
        return (nkv * (qk_nope_hd + v_head_dim), kv_lora_rank)

    # MLP
    if "gate_proj.weight" in hf_key or "up_proj.weight" in hf_key:
        moe_mlp = getattr(cfg, "moe_intermediate_size", mlp)
        shared_mlp = getattr(cfg, "shared_expert_intermediate_size", mlp)
        if "shared_expert" in hf_key:
            return (shared_mlp, emb)
        if "experts" in hf_key:
            return (moe_mlp, emb)
        return (mlp, emb)
    if "down_proj.weight" in hf_key:
        moe_mlp = getattr(cfg, "moe_intermediate_size", mlp)
        shared_mlp = getattr(cfg, "shared_expert_intermediate_size", mlp)
        if "shared_expert" in hf_key:
            return (emb, shared_mlp)
        if "experts" in hf_key:
            return (emb, moe_mlp)
        return (emb, mlp)

    # Embeddings
    if "embed_tokens.weight" in hf_key or "tok_embeddings.weight" in hf_key:
        return (vocab, emb)
    if "lm_head.weight" in hf_key or hf_key == "output.weight":
        return (vocab, emb)

    # Norms
    if "norm.weight" in hf_key or "layernorm.weight" in hf_key:
        return (emb,)
    if "q_norm.weight" in hf_key or "k_norm.weight" in hf_key:
        return (hd,)
    if "q_a_layernorm.weight" in hf_key:
        return (q_lora_rank,)
    if "kv_a_layernorm.weight" in hf_key:
        return (kv_lora_rank,)

    # Linear attention (Qwen3Next GDN)
    lin_nk = getattr(cfg, "linear_num_key_heads", 0)
    lin_nv = getattr(cfg, "linear_num_value_heads", 0)
    lin_kd = getattr(cfg, "linear_key_head_dim", 0)
    lin_vd = getattr(cfg, "linear_value_head_dim", 0)
    lin_conv_k = getattr(cfg, "linear_conv_kernel_dim", 0)
    key_dim = lin_nk * lin_kd
    value_dim = lin_nv * lin_vd
    if "in_proj_qkvz.weight" in hf_key:
        return (2 * key_dim + 2 * value_dim, emb)
    if "in_proj_ba.weight" in hf_key:
        return (2 * lin_nv, emb)
    if "linear_attn.out_proj.weight" in hf_key:
        return (emb, value_dim)
    if "conv1d.weight" in hf_key:
        return (2 * key_dim + value_dim, 1, lin_conv_k)
    if "A_log" in hf_key:
        return (lin_nv,)
    if "dt_bias" in hf_key:
        return (lin_nv,)
    if "linear_attn.norm.weight" in hf_key:
        return (lin_vd,)

    # MoE gate / router
    n_experts = getattr(cfg, "num_experts", getattr(cfg, "num_local_experts",
                        getattr(cfg, "n_routed_experts", 0)))
    if "gate.weight" in hf_key or "router.weight" in hf_key:
        return (n_experts, emb)
    if "gate.bias" in hf_key or "router.bias" in hf_key:
        return (n_experts,)
    if "e_score_correction_bias" in hf_key:
        return (n_experts,)

    # Fused expert weights (3D)
    moe_mlp = getattr(cfg, "moe_intermediate_size", mlp)
    last_axis_fused = getattr(cfg, "model_type", "") == "gpt_oss"
    if hf_key.endswith("experts.gate_up_proj"):
        return (n_experts, emb, 2 * moe_mlp) if last_axis_fused else (n_experts, 2 * moe_mlp, emb)
    if hf_key.endswith("experts.down_proj"):
        return (n_experts, moe_mlp, emb) if last_axis_fused else (n_experts, emb, moe_mlp)

    # Expert biases (GPT-OSS)
    if "gate_up_proj_bias" in hf_key:
        return (n_experts, 2 * moe_mlp)
    if "down_proj_bias" in hf_key:
        return (n_experts, emb)

    return ()


def _copy_tokenizer(
    hf_model_path: str, output_dir: str, hf_token: str | None
) -> None:
    """Copy tokenizer from HF model to output directory."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_path, token=hf_token, trust_remote_code=True
        )
        tokenizer.save_pretrained(output_dir)
    except Exception:
        max_logging.warning(f"Could not copy tokenizer from {hf_model_path}. "
                           "Use --tokenizer-path to specify a tokenizer source.")


# ── CLI entry points ─────────────────────────────────────────────────────────


def _build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        prog="convert",
        description="Bidirectional HuggingFace <-> Megatext checkpoint conversion.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("hf-to-megatext", help="Convert HuggingFace checkpoint to Megatext format.")
    p1.add_argument(
        "--hf-model-path",
        required=True,
        help=(
            "HF model path or hub ID readable by transformers.AutoModelForCausalLM"
            ".from_pretrained(...) or, for multimodal models, "
            "transformers.AutoModelForImageTextToText.from_pretrained(...)."
        ),
    )
    p1.add_argument("--output-dir", required=True, help="Directory for the converted Megatext checkpoint.")
    p1.add_argument("--scan-layers", action=argparse.BooleanOptionalAction, default=True,
                    help="Use scanned layers (default: True). Use --no-scan-layers to disable.")
    p1.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                    help="HuggingFace access token (default: $HF_TOKEN env var).")

    p2 = sub.add_parser("megatext-to-hf", help="Convert Megatext checkpoint to HuggingFace format.")
    p2.add_argument(
        "--megatext-model-path",
        required=True,
        help=(
            "Megatext checkpoint root, step directory, or items directory. "
            "Examples: /tmp/checkpoints, /tmp/checkpoints/8000, "
            "/tmp/checkpoints/8000/items."
        ),
    )
    p2.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Optional checkpoint step override when --megatext-model-path points at a checkpoint root.",
    )
    p2.add_argument("--output-dir", required=True, help="Directory for the converted HF checkpoint.")
    p2.add_argument(
        "--hf-config-path",
        required=True,
        help=(
            "Pre-prepared HF model/config template path or HF hub ID readable by "
            "transformers.AutoConfig.from_pretrained(...). "
            "Used for the runtime HF config/model class and tokenizer artifacts."
        ),
    )
    p2.add_argument("--scan-layers", action=argparse.BooleanOptionalAction, default=True,
                    help="Whether the checkpoint uses scanned layers (default: True).")
    p2.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                    help="HuggingFace access token (default: $HF_TOKEN env var).")
    p2.add_argument("--tokenizer-path", default=None,
                    help="Optional path or HF hub ID to load the tokenizer from (default: use --hf-config-path).")

    return parser


def main() -> None:
    """CLI entry point with hf-to-megatext / megatext-to-hf subcommands."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.INFO)

    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "hf-to-megatext":
        result = hf_to_megatext(
            hf_model_path=args.hf_model_path,
            output_dir=args.output_dir,
            scan_layers=args.scan_layers,
            hf_token=args.hf_token,
        )
        print(f"Done: HF → Megatext: {result}")
    elif args.command == "megatext-to-hf":
        result = megatext_to_hf(
            megatext_model_path=args.megatext_model_path,
            output_dir=args.output_dir,
            hf_config_path=args.hf_config_path,
            scan_layers=args.scan_layers,
            hf_token=args.hf_token,
            checkpoint_step=args.checkpoint_step,
            tokenizer_path=args.tokenizer_path,
        )
        print(f"Done: Megatext → HF: {result}")


if __name__ == "__main__":
    main()
