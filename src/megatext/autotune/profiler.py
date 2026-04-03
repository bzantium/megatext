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
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass

from megatext.autotune.strategies import Candidate

logger = logging.getLogger(__name__)

EXIT_CLASS_SUCCESS = "success"
EXIT_CLASS_CONFIRMED_OOM = "confirmed_oom"
EXIT_CLASS_TIMEOUT = "timeout"
EXIT_CLASS_SIGNAL = "signal"
EXIT_CLASS_WORKER_ERROR = "worker_error"
EXIT_CLASS_INFRA_ERROR = EXIT_CLASS_WORKER_ERROR
INFRA_EXIT_CLASSES = {EXIT_CLASS_TIMEOUT, EXIT_CLASS_SIGNAL, EXIT_CLASS_WORKER_ERROR}
_OOM_PATTERNS = (
    re.compile(r"out of memory"),
    re.compile(r"resource[_ ]exhausted"),
    re.compile(r"\bhbmoom\b"),
    re.compile(r"\bvmemoom\b"),
    re.compile(r"scoped vmem"),
    re.compile(r"\boom\b"),
)


def _looks_like_oom(message: str | None) -> bool:
    """Classify worker stderr/stdout tails without relying on exit codes alone."""
    if not message:
        return False
    err_str = message.lower()
    return any(pattern.search(err_str) for pattern in _OOM_PATTERNS)


def _first_error_line(message: str | None) -> str | None:
    """Return a concise single-line diagnostic."""
    if not message:
        return None
    for line in message.splitlines():
        line = line.strip()
        if line:
            return line[:300]
    return None


def _describe_returncode(returncode: int) -> str:
    """Render subprocess exit status in a form useful for TPU worker debugging."""
    if returncode < 0:
        signum = -returncode
        try:
            return f"terminated by signal {signal.Signals(signum).name}"
        except ValueError:
            return f"terminated by signal {signum}"
    return f"exit code {returncode}"


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
    compile_peak_memory_gb: float = 0.0
    runtime_peak_hbm_gb: float = 0.0
    error: str | None = None
    exit_class: str = EXIT_CLASS_SUCCESS
    returncode: int | None = None
    returncode_detail: str | None = None
    stdout_tail: str | None = None
    failure_stage: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.exit_class == EXIT_CLASS_SUCCESS and not self.oom and self.error is None

    @property
    def confirmed_oom(self) -> bool:
        return self.exit_class == EXIT_CLASS_CONFIRMED_OOM or self.oom

    @property
    def infra_error(self) -> bool:
        return self.exit_class in INFRA_EXIT_CLASSES

    def __repr__(self) -> str:
        if self.confirmed_oom:
            return f"ProfileResult(OOM, candidate={self.candidate})"
        if self.infra_error or self.error:
            return f"ProfileResult(ERROR: {self.error}, candidate={self.candidate})"
        return (
            f"ProfileResult(mean={self.mean_step_time_seconds:.3f}s, "
            f"peak_mem={self.peak_memory_gb:.1f}GB, "
            f"tflops={self.tflops_per_device:.1f}, "
            f"candidate={self.candidate})"
        )


def _log_worker_diagnostics(result: ProfileResult) -> None:
    """Emit full subprocess diagnostics to the parent log stream."""
    banner = "*" * 100
    header = (
        f"\n{banner}\n"
        f"* AUTOTUNE WORKER FAILURE: {result.candidate} "
        f"(exit_class={result.exit_class}, stage={result.failure_stage or 'unknown'})\n"
        f"{banner}"
    )
    footer = banner
    if result.error:
        logger.error(
            "%s\n%s\n%s",
            header,
            result.error,
            footer,
        )
    elif result.stdout_tail and result.exit_class != EXIT_CLASS_SUCCESS:
        logger.error(
            "%s\n%s\n%s",
            header,
            result.stdout_tail,
            footer,
        )


def profile_candidate(
    config_overrides: dict,
    candidate: Candidate,
    num_steps: int = 5,
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
    # Do not let the worker re-enable a shared compilation cache from pyconfig.
    config_dict["jax_cache_dir"] = ""
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
    # Isolate XLA cache per subprocess to prevent accumulation across runs.
    cache_dir = f"/tmp/autotune_xla_cache_{uuid.uuid4().hex}"
    env["JAX_COMPILATION_CACHE_DIR"] = cache_dir

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result: ProfileResult
    try:
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
                compile_peak_memory_gb=0.0,
                runtime_peak_hbm_gb=0.0,
                tflops_per_device=0.0,
                oom=False,
                error=f"Timeout after {timeout}s",
                exit_class=EXIT_CLASS_TIMEOUT,
                returncode=None,
                returncode_detail=f"timeout after {timeout}s",
                stdout_tail=stdout.decode(errors='replace')[-2000:] if stdout else "",
                failure_stage="timeout",
            )

        error_tail = stdout.decode(errors="replace")[-2000:] if stdout else ""
        returncode_detail = _describe_returncode(proc.returncode)
        if proc.returncode != 0:
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
            if _looks_like_oom(error_tail):
                exit_class = EXIT_CLASS_CONFIRMED_OOM
                oom = True
                error = error_tail.strip() or f"Worker exited without result ({returncode_detail})."
            else:
                exit_class = EXIT_CLASS_SIGNAL if proc.returncode < 0 else EXIT_CLASS_WORKER_ERROR
                oom = False
                error = (
                    f"Worker exited without result ({returncode_detail}). "
                    f"Output tail:\n{error_tail.strip()}"
                ).strip()
            result = ProfileResult(
                candidate=candidate,
                mean_step_time_seconds=float("inf"),
                max_step_time_seconds=float("inf"),
                min_step_time_seconds=float("inf"),
                peak_memory_gb=0.0,
                compile_peak_memory_gb=0.0,
                runtime_peak_hbm_gb=0.0,
                tflops_per_device=0.0,
                oom=oom,
                error=error,
                exit_class=exit_class,
                returncode=proc.returncode,
                returncode_detail=returncode_detail,
                stdout_tail=error_tail or None,
                failure_stage=None,
            )
        else:
            compile_peak = result_data.get("compile_peak_memory_gb", result_data.get("peak_memory_gb", 0.0))
            runtime_peak = result_data.get("runtime_peak_hbm_gb", 0.0)
            result = ProfileResult(
                candidate=candidate,
                mean_step_time_seconds=result_data["mean_step_time_seconds"],
                max_step_time_seconds=result_data["max_step_time_seconds"],
                min_step_time_seconds=result_data["min_step_time_seconds"],
                peak_memory_gb=result_data.get("peak_memory_gb", max(compile_peak, runtime_peak)),
                compile_peak_memory_gb=compile_peak,
                runtime_peak_hbm_gb=runtime_peak,
                tflops_per_device=result_data["tflops_per_device"],
                oom=result_data.get("oom", False),
                error=result_data.get("error"),
                exit_class=result_data.get(
                    "exit_class",
                    EXIT_CLASS_CONFIRMED_OOM if result_data.get("oom", False) else EXIT_CLASS_SUCCESS,
                ),
                returncode=result_data.get("returncode", proc.returncode),
                returncode_detail=result_data.get("returncode_detail", returncode_detail),
                stdout_tail=result_data.get("stdout_tail", error_tail or None),
                failure_stage=result_data.get("failure_stage"),
            )
    finally:
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)

    # Give TPU driver time to fully reclaim HBM before next subprocess.
    # Worker calls jax.clear_backends() + sleep(5) before exit, but the OS-level
    # resource reclamation after process death also needs time.
    if result.oom or result.error:
        time.sleep(10)
    else:
        time.sleep(3)

    if result.exit_class != EXIT_CLASS_SUCCESS:
        _log_worker_diagnostics(result)

    return result
