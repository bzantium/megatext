"""Qwen3-SWA checkpoint conversion — reuses Qwen3 dense converter.

Sliding window attention is a runtime configuration; the checkpoint
structure is identical to the base Qwen3 dense model.
"""

from megatext.conversion.models.qwen3 import build_qwen3, build_qwen3_transforms

build_qwen3_swa = build_qwen3
build_qwen3_swa_transforms = build_qwen3_transforms
