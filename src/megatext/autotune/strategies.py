"""Candidate generation for hyperparameter auto-tuning.

Generates valid combinations of parallelism, remat, and batch size
configurations based on the detected topology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Ordered by HBM usage ascending (least memory first, slowest first).
# From megatext base.yaml: "trade-off between speed (fastest to slowest) and HBM usage (highest to lowest)"
# Ordered by HBM usage ascending (least memory first).
# NOTE: offloaded policies (minimal_offloaded, qkv_proj_offloaded) are excluded
# because they offload activations to host RAM, causing extreme host memory usage
# (~170GB+) that triggers OOMKilled on GKE TPU nodes.
REMAT_POLICIES = [
    "full",
    "save_out_proj",
    "save_qkv_proj",
    "save_dot_except_mlp",
    "save_dot_except_mlpwi",
    "save_dot_with_context_except_mlp",
    "minimal",
    "minimal_with_context",
]

BATCH_SIZES = list(range(1, 17))

# Splash attention block sizes to search (must be multiples of 128, <= seq_len).
# Larger = faster but more memory. Ordered ascending.
SA_BLOCK_SIZES = [512, 1024, 2048, 4096]

# Forward-path splash attention params.
SA_BLOCK_FORWARD_KEYS = [
    "sa_block_q",
    "sa_block_kv",
    "sa_block_kv_compute",
]

# Backward-path splash attention params.
SA_BLOCK_BACKWARD_KEYS = [
    "sa_block_q_dkv",
    "sa_block_kv_dkv",
    "sa_block_kv_dkv_compute",
    "sa_block_q_dq",
    "sa_block_kv_dq",
]


@dataclass
class Candidate:
    """A single hyperparameter configuration candidate."""

    ici_fsdp_parallelism: int = -1
    ici_tensor_parallelism: int = 1
    dcn_data_parallelism: int = 1
    dcn_fsdp_parallelism: int = 1
    remat_policy: str = "full"
    per_device_batch_size: float = 1.0
    gradient_accumulation_steps: int = 1
    scan_layers: bool = True
    sa_block_size: int = 512
    sa_block_backward_size: int | None = None

    @classmethod
    def from_config_dict(cls, config_overrides: dict) -> Candidate:
        """Build a base candidate from a dict of config overrides."""
        return cls(
            ici_fsdp_parallelism=config_overrides.get("ici_fsdp_parallelism", -1),
            ici_tensor_parallelism=config_overrides.get("ici_tensor_parallelism", 1),
            dcn_data_parallelism=config_overrides.get("dcn_data_parallelism", 1),
            dcn_fsdp_parallelism=config_overrides.get("dcn_fsdp_parallelism", 1),
            remat_policy=config_overrides.get("remat_policy", "full"),
            per_device_batch_size=config_overrides.get("per_device_batch_size", 1.0),
            scan_layers=config_overrides.get("scan_layers", True),
            sa_block_size=config_overrides.get("sa_block_q", 512),
            sa_block_backward_size=config_overrides.get("sa_block_q_dkv", config_overrides.get("sa_block_q", 512)),
        )

    def to_overrides(self) -> dict[str, Any]:
        """Convert to config override dict."""
        sa_block_backward_size = self.sa_block_backward_size or self.sa_block_size
        return {
            "ici_data_parallelism": 1,
            "ici_fsdp_parallelism": self.ici_fsdp_parallelism,
            "ici_tensor_parallelism": self.ici_tensor_parallelism,
            "dcn_data_parallelism": self.dcn_data_parallelism,
            "dcn_fsdp_parallelism": self.dcn_fsdp_parallelism,
            "remat_policy": self.remat_policy,
            "per_device_batch_size": self.per_device_batch_size,
            "scan_layers": self.scan_layers,
            "sa_use_fused_bwd_kernel": True,
            **{k: self.sa_block_size for k in SA_BLOCK_FORWARD_KEYS},
            **{k: sa_block_backward_size for k in SA_BLOCK_BACKWARD_KEYS},
        }

    def __repr__(self) -> str:
        sa_block_repr = (
            f"sa_block=(fwd={self.sa_block_size},bwd={self.sa_block_backward_size})"
            if self.sa_block_backward_size is not None and self.sa_block_backward_size != self.sa_block_size
            else f"sa_block={self.sa_block_size}"
        )
        return (
            f"Candidate(ici=(fsdp={self.ici_fsdp_parallelism},"
            f"tp={self.ici_tensor_parallelism}), "
            f"dcn=({self.dcn_data_parallelism},{self.dcn_fsdp_parallelism}), "
            f"remat={self.remat_policy}, batch={self.per_device_batch_size}, "
            f"{sa_block_repr})"
        )


@dataclass
class ModelConstraints:
    """Model and hardware constraints for staged search."""

    num_kv_heads: int = 8
    num_decoder_layers: int = 32
    num_devices: int = 8
    hbm_per_device_gb: float = 32.0

    @classmethod
    def from_config_dict(cls, config_overrides: dict, topology):
        """Build constraints from a dict of config overrides and topology."""
        return cls(
            num_kv_heads=config_overrides.get("base_num_kv_heads", 8),
            num_decoder_layers=config_overrides.get("base_num_decoder_layers", 32),
            num_devices=topology.device_count,
            hbm_per_device_gb=topology.hbm_per_device_gb,
        )

    def valid_tp_values(self) -> list[int]:
        """Return valid TP values that divide both num_devices and num_kv_heads.

        allow_split_physical_axes=True is set in profiler, so physical mesh
        compatibility is not a constraint.
        """
        return sorted(
            t for t in _divisors(self.num_devices)
            if self.num_kv_heads % t == 0
        )


def _divisors(n: int) -> list[int]:
    """Return sorted list of divisors of n."""
    divs = set()
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs)
