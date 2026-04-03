"""Model-specific conversion builders — explicit wiring, no side effects."""
from megatext.conversion.utils import ArchSpec

# ── Model functions ──────────────────────────────────────────────────────────
from megatext.conversion.models.llama import (
    build_llama, build_llama_transforms,
)
from megatext.conversion.models.gemma4 import (
    build_gemma4, build_gemma4_transforms, compute_gemma4_shapes,
)
from megatext.conversion.models.qwen3 import (
    build_qwen3, build_qwen3_transforms,
)
from megatext.conversion.models.qwen3_swa import (
    build_qwen3_swa, build_qwen3_swa_transforms, compute_qwen3_swa_shapes,
)
from megatext.conversion.models.qwen3_moe import (
    build_qwen3_moe, build_qwen3_moe_transforms, compute_qwen3_moe_shapes,
)
from megatext.conversion.models.qwen3_next import (
    build_qwen3_next, build_qwen3_next_transforms,
    compute_qwen3_next_shapes,
)
from megatext.conversion.models.deepseek_v3 import (
    build_deepseek, build_deepseek_transforms,
    compute_deepseek_shapes,
)
from megatext.conversion.models.gpt_oss import (
    build_gpt_oss, build_gpt_oss_transforms,
    compute_gpt_oss_shapes,
)

# ── Architecture specs ───────────────────────────────────────────────────────
ARCH_SPECS: dict[str, ArchSpec] = {
    "llama": ArchSpec(
        model_type="llama", decoder_block="llama2", template_name="llama3",
        scale_query=True, reorder_rope=True,
    ),
    "gemma4": ArchSpec(
        model_type="gemma4", decoder_block="gemma4", template_name="gemma4",
        hf_prefix="model.language_model",
        has_qk_norm=True, has_post_attn_norm=True,
        has_pre_ffw_norm=True, has_post_ffw_norm=True,
        scale_embedding=True, composite_split="concat",
    ),
    "gemma4_text": ArchSpec(
        model_type="gemma4_text", decoder_block="gemma4", template_name="gemma4",
        hf_prefix="model",
        has_qk_norm=True, has_post_attn_norm=True,
        has_pre_ffw_norm=True, has_post_ffw_norm=True,
        scale_embedding=True, composite_split="concat",
    ),
    "qwen3": ArchSpec(
        model_type="qwen3", decoder_block="qwen3", template_name="qwen3",
        has_qk_norm=True,
    ),
    "qwen3_swa": ArchSpec(
        model_type="qwen3_swa", decoder_block="qwen3_swa", template_name="qwen3-swa",
        has_qk_norm=True,
    ),
    "qwen3_moe": ArchSpec(
        model_type="qwen3_moe", decoder_block="qwen3_moe", template_name="qwen3-moe",
        has_qk_norm=True, composite_split="concat",
    ),
    "qwen3_next": ArchSpec(
        model_type="qwen3_next", decoder_block="qwen3_next", template_name="qwen3-next",
        composite_split="concat",
    ),
    "deepseek_v3": ArchSpec(
        model_type="deepseek_v3", decoder_block="deepseek", template_name="deepseek",
        composite_split="concat",
    ),
    "gpt_oss": ArchSpec(
        model_type="gpt_oss", decoder_block="gpt_oss", template_name="gpt-oss",
        composite_split="interleave_last",
    ),
}

MAPPING_BUILDERS = {
    "llama": build_llama,
    "gemma4": build_gemma4,
    "gemma4_text": build_gemma4,
    "qwen3": build_qwen3,
    "qwen3_swa": build_qwen3_swa,
    "qwen3_moe": build_qwen3_moe,
    "qwen3_next": build_qwen3_next,
    "deepseek_v3": build_deepseek,
    "gpt_oss": build_gpt_oss,
}

TRANSFORM_BUILDERS = {
    "llama": build_llama_transforms,
    "gemma4": build_gemma4_transforms,
    "gemma4_text": build_gemma4_transforms,
    "qwen3": build_qwen3_transforms,
    "qwen3_swa": build_qwen3_swa_transforms,
    "qwen3_moe": build_qwen3_moe_transforms,
    "qwen3_next": build_qwen3_next_transforms,
    "deepseek_v3": build_deepseek_transforms,
    "gpt_oss": build_gpt_oss_transforms,
}

SHAPE_BUILDERS = {
    "gemma4": compute_gemma4_shapes,
    "gemma4_text": compute_gemma4_shapes,
    "qwen3_swa": compute_qwen3_swa_shapes,
    "qwen3_moe": compute_qwen3_moe_shapes,
    "qwen3_next": compute_qwen3_next_shapes,
    "deepseek_v3": compute_deepseek_shapes,
    "gpt_oss": compute_gpt_oss_shapes,
}


def resolve(model_type: str) -> ArchSpec:
    if model_type not in ARCH_SPECS:
        raise ValueError(
            f"Unsupported HF model_type: {model_type!r}. "
            f"Supported: {sorted(ARCH_SPECS.keys())}"
        )
    return ARCH_SPECS[model_type]
