"""Short profiling runs for auto-tuning candidates.

Each candidate is profiled in an isolated subprocess to avoid:
1. JAX mesh change crashes (parallelism search changes TP/FSDP)
2. JIT compilation cache accumulation (host RAM growth)
3. XLA state corruption after OOM

The parent process never imports JAX, keeping the TPU free for workers.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass

from megatext.autotune.strategies import Candidate

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Result of profiling a single candidate."""

    candidate: Candidate
    mean_step_time_seconds: float
    max_step_time_seconds: float
    min_step_time_seconds: float
    peak_memory_gb: float
    tflops_per_device: float
    oom: bool
    error: str | None = None

    def __repr__(self) -> str:
        if self.oom:
            return f"ProfileResult(OOM, candidate={self.candidate})"
        if self.error:
            return f"ProfileResult(ERROR: {self.error}, candidate={self.candidate})"
        return (
            f"ProfileResult(mean={self.mean_step_time_seconds:.3f}s, "
            f"peak_mem={self.peak_memory_gb:.1f}GB, "
            f"tflops={self.tflops_per_device:.1f}, "
            f"candidate={self.candidate})"
        )


def profile_candidate(
    config_overrides: dict,
    candidate: Candidate,
    num_steps: int = 10,
    warmup_steps: int = 2,
    timeout: int = 600,
) -> ProfileResult:
    """Profile a candidate by running a subprocess worker.

    Each candidate runs in an isolated subprocess so that:
    - Mesh changes (TP/FSDP) don't crash the process
    - JIT caches are freed on subprocess exit (no host RAM leak)
    - OOM doesn't corrupt JAX state for subsequent candidates

    Args:
        config_overrides: Dict of key=value overrides for megatext pyconfig.
        candidate: Candidate configuration to profile.
        num_steps: Number of profiling steps.
        warmup_steps: Number of warmup steps before profiling.
        timeout: Maximum time in seconds for the subprocess.
    """
    config_dict = dict(config_overrides)
    candidate_dict = asdict(candidate)

    result_file = f"/tmp/autotune_result_{uuid.uuid4().hex}.json"

    cmd = [
        sys.executable, "-m", "megatext.autotune.profiler_worker",
        "--config-json", json.dumps(config_dict),
        "--candidate-json", json.dumps(candidate_dict),
        "--num-steps", str(num_steps),
        "--warmup-steps", str(warmup_steps),
        "--result-file", result_file,
    ]

    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate()
        logger.error(f"Timeout ({timeout}s) for {candidate}")
        if os.path.exists(result_file):
            os.unlink(result_file)
        return ProfileResult(
            candidate=candidate,
            mean_step_time_seconds=float("inf"),
            max_step_time_seconds=float("inf"),
            min_step_time_seconds=float("inf"),
            peak_memory_gb=0.0,
            tflops_per_device=0.0,
            oom=False,
            error=f"Timeout after {timeout}s",
        )

    if proc.returncode != 0:
        error_tail = stdout.decode(errors="replace")[-2000:] if stdout else ""
        logger.error(f"Worker failed (rc={proc.returncode}) for {candidate}:\n{error_tail}")

    # Read result from file
    result_data = None
    if os.path.exists(result_file):
        try:
            with open(result_file) as f:
                result_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
        finally:
            os.unlink(result_file)

    if result_data is None:
        return ProfileResult(
            candidate=candidate,
            mean_step_time_seconds=float("inf"),
            max_step_time_seconds=float("inf"),
            min_step_time_seconds=float("inf"),
            peak_memory_gb=0.0,
            tflops_per_device=0.0,
            oom=True,
            error=None,
        )

    return ProfileResult(
        candidate=candidate,
        mean_step_time_seconds=result_data["mean_step_time_seconds"],
        max_step_time_seconds=result_data["max_step_time_seconds"],
        min_step_time_seconds=result_data["min_step_time_seconds"],
        peak_memory_gb=result_data["peak_memory_gb"],
        tflops_per_device=result_data["tflops_per_device"],
        oom=result_data.get("oom", False),
        error=result_data.get("error"),
    )
