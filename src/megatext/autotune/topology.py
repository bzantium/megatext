"""TPU topology detection and device mesh information."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TPUTopology:
    """Detected TPU topology information."""

    device_count: int
    local_device_count: int
    process_count: int
    process_index: int
    platform: str  # "tpu", "gpu", "cpu"
    chip_name: str  # e.g. "v4", "v5e", "v5p", "v6e"
    devices_per_chip: int
    num_slices: int
    chips_per_host: int
    hbm_per_device_gb: float

    @property
    def total_hbm_gb(self) -> float:
        return self.hbm_per_device_gb * self.device_count

    @property
    def is_multi_host(self) -> bool:
        return self.process_count > 1

    @property
    def is_multi_slice(self) -> bool:
        return self.num_slices > 1


def _detect_local() -> TPUTopology:
    """Detect the current TPU/accelerator topology.

    Imports JAX — only call from subprocess workers, not from the parent.
    """
    import jax
    from megatext.utils import logging as max_logging

    devices = jax.devices()
    local_devices = jax.local_devices()
    device_count = len(devices)
    local_device_count = len(local_devices)
    process_count = jax.process_count()
    process_index = jax.process_index()

    # Detect platform
    platform = devices[0].platform if devices else "cpu"

    # Detect chip type and HBM
    chip_name = "unknown"
    hbm_per_device_gb = 16.0  # default
    devices_per_chip = 1

    if platform == "tpu":
        device_kind = getattr(devices[0], "device_kind", "")
        max_logging.log(f"TPU device_kind: {device_kind!r}")
        chip_name, default_hbm, devices_per_chip = _parse_tpu_device(device_kind)
        hbm_per_device_gb = _query_hbm_total_gb() or default_hbm
    elif platform == "gpu":
        chip_name = "gpu"
        hbm_per_device_gb = 80.0  # assume A100/H100

    # Estimate slices: multi-slice uses DCN between slices
    # Heuristic: devices per host * num hosts per slice
    chips_per_host = local_device_count // devices_per_chip
    num_slices = _estimate_num_slices(device_count, local_device_count, process_count)

    topology = TPUTopology(
        device_count=device_count,
        local_device_count=local_device_count,
        process_count=process_count,
        process_index=process_index,
        platform=platform,
        chip_name=chip_name,
        devices_per_chip=devices_per_chip,
        num_slices=num_slices,
        chips_per_host=chips_per_host,
        hbm_per_device_gb=hbm_per_device_gb,
    )

    max_logging.log(
        f"Topology: {platform} {chip_name}, {device_count} devices ({local_device_count} local), "
        f"{process_count} processes, {num_slices} slices, "
        f"{hbm_per_device_gb:.1f} GB HBM/device ({topology.total_hbm_gb:.1f} GB total)"
    )

    return topology


_TOPO_MARKER = "__AUTOTUNE_TOPOLOGY__"


def detect_topology() -> TPUTopology:
    """Detect topology by running a subprocess that initializes JAX.

    The parent process never imports JAX, keeping the TPU free for workers.
    """
    import json
    import subprocess
    import sys

    cmd = [
        sys.executable, "-c",
        "import json; "
        "from megatext.autotune.topology import _detect_local, _TOPO_MARKER; "
        "from dataclasses import asdict; "
        "t = _detect_local(); "
        f"print(_TOPO_MARKER + json.dumps(asdict(t)))",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"Topology detection failed: {proc.stderr.strip()[-500:]}")

    for line in proc.stdout.splitlines():
        if line.startswith(_TOPO_MARKER):
            data = json.loads(line[len(_TOPO_MARKER):])
            return TPUTopology(**data)

    raise RuntimeError(f"No topology output. stderr: {proc.stderr.strip()[-500:]}")


def _query_hbm_total_gb() -> float | None:
    """Query actual HBM total per device via tpustat at runtime."""
    try:
        from tpustat.core import TPUStatCollection
        stats = TPUStatCollection.new_query()
        if stats.devices:
            return stats.devices[0].hbm_total_mib / 1024
    except Exception:
        pass
    return None


def _parse_tpu_device(device_kind: str) -> tuple[str, float, int]:
    """Parse TPU device kind string to chip name, HBM, devices per chip."""
    kind = device_kind.lower()

    # TPU v4: 32GB HBM per chip, 1 device per chip
    if "v4" in kind:
        return "v4", 32.0, 1
    # TPU v5e / v5 lite: 16GB HBM per chip, 1 device per chip
    elif "v5e" in kind or "v5litepod" in kind or "v5 lite" in kind:
        return "v5e", 16.0, 1
    # TPU v5p (must check after v5e): 95GB HBM per chip, 1 device per chip
    elif "v5p" in kind:
        return "v5p", 95.0, 1
    # TPU v6e: 32GB HBM per chip, 1 device per chip
    elif "v6e" in kind or "v6" in kind:
        return "v6e", 32.0, 1
    else:
        return "unknown", 16.0, 1


def _estimate_num_slices(
    device_count: int,
    local_device_count: int,
    process_count: int,
) -> int:
    """Estimate number of TPU slices.

    Multi-slice setups have multiple independent slices connected via DCN.
    Each slice is a self-contained TPU pod or pod slice.
    """
    if process_count <= 1:
        return 1

    # Check environment variable (xpk sets this)
    import os
    num_slices_env = os.environ.get("NUM_SLICES", "")
    if num_slices_env.isdigit():
        return int(num_slices_env)

    # Heuristic: if all processes see the same number of local devices,
    # and total devices / local devices == process count, it's single slice multi-host
    # Otherwise estimate based on typical slice sizes
    devices_per_process = device_count // process_count
    if devices_per_process == local_device_count:
        return 1

    return max(1, device_count // (local_device_count * process_count))
