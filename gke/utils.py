"""GKE utilities: TPU-specific LIBTPU_INIT_ARGS and helpers."""

from __future__ import annotations


# Default LIBTPU_INIT_ARGS per TPU generation.
# These are the common XLA compiler flags that optimize performance for each TPU type.
# Job scripts can append additional flags if needed.
LIBTPU_INIT_ARGS: dict[str, str] = {
    "v4": (
        "--xla_enable_async_all_gather=true "
        "TPU_MEGACORE=MEGACORE_DENSE"
    ),
    "v5e": (
        "--xla_tpu_enable_data_parallel_all_reduce_opt=true "
        "--xla_tpu_data_parallel_opt_different_sized_ops=true "
        "--xla_tpu_enable_async_collective_fusion=true "
        "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
        "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
        "--xla_tpu_overlap_compute_collective_tc=true "
        "--xla_enable_async_all_gather=true"
    ),
    "v5p": (
        "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
        "--xla_tpu_megacore_fusion_allow_ags=false "
        "--xla_enable_async_collective_permute=true "
        "--xla_tpu_enable_ag_backward_pipelining=true "
        "--xla_tpu_enable_data_parallel_all_reduce_opt=true "
        "--xla_tpu_data_parallel_opt_different_sized_ops=true "
        "--xla_tpu_enable_async_collective_fusion=true "
        "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
        "--xla_tpu_overlap_compute_collective_tc=true "
        "--xla_enable_async_all_gather=true"
    ),
    "v6e": (
        "--xla_tpu_scoped_vmem_limit_kib=98304 "
        "--xla_tpu_use_minor_sharding_for_major_trivial_input=true "
        "--xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 "
        "--xla_tpu_assign_all_reduce_scatter_layout "
        "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
        "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
        "--xla_tpu_overlap_compute_collective_tc=true "
        "--xla_enable_async_all_gather=true"
    ),
}


def get_tpu_generation(tpu_type: str) -> str:
    """Extract TPU generation from tpu_type string.

    Examples:
        'v5litepod-256' -> 'v5e'
        'v5p-128' -> 'v5p'
        'v6e-256' -> 'v6e'
        'v4-32' -> 'v4'
    """
    base = tpu_type.split("-")[0]
    if base.startswith("v5litepod") or base.startswith("v5e"):
        return "v5e"
    return base


def get_libtpu_init_args(tpu_type: str) -> str | None:
    """Get default LIBTPU_INIT_ARGS for a given TPU type.

    Returns None if no defaults are configured for this TPU generation.
    """
    gen = get_tpu_generation(tpu_type)
    return LIBTPU_INIT_ARGS.get(gen)
