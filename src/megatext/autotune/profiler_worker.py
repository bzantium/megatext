"""Subprocess worker for profiling a single autotune candidate.

Invoked by profile_candidate() as a subprocess to isolate JAX state.
Each invocation initializes JAX from scratch, profiles, and writes
the result as JSON to a file. The parent process reads this file.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import signal
import threading
import time
import traceback
import gc

import jax

from megatext.autotune.profiler import (
    EXIT_CLASS_CONFIRMED_OOM,
    EXIT_CLASS_INFRA_ERROR,
    EXIT_CLASS_SUCCESS,
    _looks_like_oom,
)
from megatext.utils import logging as max_logging


def _sample_runtime_hbm_peak(stop_event: threading.Event, out: dict, interval_seconds: float = 0.2) -> None:
    """Best-effort HBM sampler using tpustat. Records the max observed used HBM."""
    try:
        from tpustat.core import TPUStatCollection  # pylint: disable=import-outside-toplevel
    except Exception:
        out["available"] = False
        return

    out["available"] = True
    peak_gb = 0.0
    sample_count = 0

    while not stop_event.is_set():
        try:
            stats = TPUStatCollection.new_query()
            devices = getattr(stats, "devices", []) or []
            used_gb = 0.0
            for device in devices:
                mib = None
                for attr in (
                    "hbm_used_mib",
                    "hbm_usage_mib",
                    "hbm_used",
                    "memory_used_mib",
                ):
                    value = getattr(device, attr, None)
                    if value is not None:
                        mib = value
                        break
                if mib is None:
                    continue
                try:
                    used_gb = max(used_gb, float(mib) / 1024.0)
                except (TypeError, ValueError):
                    continue
            peak_gb = max(peak_gb, used_gb)
            sample_count += 1
        except Exception:
            pass
        stop_event.wait(interval_seconds)

    out["runtime_peak_hbm_gb"] = peak_gb
    out["sample_count"] = sample_count


def _run(config_overrides: dict, candidate_dict: dict, num_steps: int, warmup_steps: int) -> dict:
    """Run profiling and return result as a dict."""
    from megatext.autotune.strategies import Candidate
    from megatext.utils.constants import MEGATEXT_PKG_DIR

    stage = "decode_candidate"
    hbm_state: dict[str, float | int | bool] = {"runtime_peak_hbm_gb": 0.0, "sample_count": 0, "available": False}
    sampler_stop = threading.Event()
    sampler = threading.Thread(
        target=_sample_runtime_hbm_peak,
        args=(sampler_stop, hbm_state),
        name="autotune-hbm-sampler",
        daemon=True,
    )
    sampler.start()

    candidate = Candidate(**candidate_dict)
    try:
        overrides = candidate.to_overrides()
        overrides["steps"] = warmup_steps + num_steps
        overrides["enable_checkpointing"] = False
        overrides["dataset_type"] = "synthetic"
        overrides["allow_split_physical_axes"] = True
        overrides["global_batch_size"] = "None"  # ensures ga_steps=1 in profiling
        overrides["log_config"] = False
        # Never reuse shared compilation cache during autotune profiling. Candidate
        # subprocesses must not observe each other's cache/HBM side effects.
        overrides["jax_cache_dir"] = ""

        # Merge candidate overrides into a new dict (don't mutate caller's dict)
        config_overrides = {**config_overrides, **overrides}

        max_logging.log(f"Profiling: {candidate} (steps={warmup_steps}w+{num_steps})")

        from megatext.configs import pyconfig

        # Build argv for pyconfig.initialize:
        #   ["script_name", "base.yaml", "key1=val1", "key2=val2", ...]
        stage = "pyconfig_initialize"
        config_path = os.path.join(MEGATEXT_PKG_DIR, "configs", "base.yaml")
        argv = ["megatext.autotune.profiler_worker", config_path]
        for k, v in config_overrides.items():
            argv.append(f"{k}={v}")

        megatext_config = pyconfig.initialize(argv)

        from megatext.utils import train_utils
        from megatext.trainers import pretrain as mt_pretrain
        from megatext.common.goodput import create_goodput_recorder
        from flax.linen import partitioning as nn_partitioning

        recorder = create_goodput_recorder(megatext_config)

        stage = "setup_train_loop"
        (
            init_rng, _, state_mesh_shardings, model, mesh,
            learning_rate_schedule, data_iterator, data_loader,
            rampup_manager, _, state,
        ) = train_utils.setup_train_loop(megatext_config, recorder)

        from megatext.utils import sharding
        params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(
            megatext_config, state_mesh_shardings
        )

        stage = "jit_train_step"
        p_train_step, _ = train_utils.jit_train_and_eval_step(
            megatext_config, model, mesh, state, state_mesh_shardings,
            mt_pretrain.train_step, mt_pretrain.eval_step, None, params_shardings,
        )

        from megatext.utils.flops import calculate_tflops_training_per_device

        stage = "calculate_tflops"
        total_tflops, _, _ = calculate_tflops_training_per_device(megatext_config, log=False)

        # Profile
        step_times = []
        _fold_in = jax.jit(jax.random.fold_in)
        for step in range(warmup_steps + num_steps):
            stage = f"profile_step_{step}"
            example_batch = data_loader.load_next_batch(rampup_manager=rampup_manager)
            nextrng = _fold_in(init_rng, step)

            start = time.monotonic()
            with jax.set_mesh(mesh), nn_partitioning.axis_rules(megatext_config.logical_axis_rules):
                state, metrics = p_train_step(state, example_batch, nextrng)
            jax.block_until_ready(state)
            jax.block_until_ready(metrics)
            elapsed = time.monotonic() - start

            if step >= warmup_steps:
                step_times.append(elapsed)

        # Memory measurement: XLA compile-time estimate + JAX runtime peak
        stage = "memory_analysis"
        compile_peak_memory_gb = 0.0
        try:
            from megatext.trainers.pretrain import get_shaped_batch
            shaped_batch = get_shaped_batch(megatext_config)
            init_rng_compile = _fold_in(init_rng, 0)
            with jax.set_mesh(mesh), nn_partitioning.axis_rules(megatext_config.logical_axis_rules):
                compiled = p_train_step.lower(state, shaped_batch, init_rng_compile).compile()
            compiled_stats = compiled.memory_analysis()
            if compiled_stats is not None:
                peak_bytes = (
                    compiled_stats.temp_size_in_bytes
                    + compiled_stats.argument_size_in_bytes
                    + compiled_stats.output_size_in_bytes
                    - compiled_stats.alias_size_in_bytes
                )
                compile_peak_memory_gb = peak_bytes / (1024 ** 3)
        except Exception:
            pass
        runtime_peak_hbm_gb = float(hbm_state.get("runtime_peak_hbm_gb", 0.0) or 0.0)
        peak_memory_gb = max(compile_peak_memory_gb, runtime_peak_hbm_gb)
        mean_step_time = sum(step_times) / len(step_times)
        tflops_per_device = total_tflops / mean_step_time if mean_step_time > 0 else 0.0

        return {
            "mean_step_time_seconds": mean_step_time,
            "max_step_time_seconds": max(step_times),
            "min_step_time_seconds": min(step_times),
            "peak_memory_gb": peak_memory_gb,
            "compile_peak_memory_gb": compile_peak_memory_gb,
            "runtime_peak_hbm_gb": runtime_peak_hbm_gb,
            "tflops_per_device": tflops_per_device,
            "oom": False,
            "error": None,
            "exit_class": EXIT_CLASS_SUCCESS,
            "sampler_sample_count": int(hbm_state.get("sample_count", 0) or 0),
            "failure_stage": None,
        }
    except Exception as exc:
        setattr(exc, "_autotune_failure_stage", stage)
        setattr(exc, "_autotune_runtime_peak_hbm_gb", float(hbm_state.get("runtime_peak_hbm_gb", 0.0) or 0.0))
        setattr(exc, "_autotune_sampler_sample_count", int(hbm_state.get("sample_count", 0) or 0))
        raise
    finally:
        sampler_stop.set()
        sampler.join(timeout=2.0)


def main():
    import absl.logging
    absl.logging.set_verbosity(absl.logging.INFO)
    faulthandler.enable()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()

    config_overrides = json.loads(args.config_json)
    candidate_dict = json.loads(args.candidate_json)

    def _write_result(payload: dict) -> None:
        with open(args.result_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())

    try:
        result = _run(config_overrides, candidate_dict, args.num_steps, args.warmup_steps)
    except Exception as e:
        failure_stage = getattr(e, "_autotune_failure_stage", "unknown")
        runtime_peak_hbm_gb = float(getattr(e, "_autotune_runtime_peak_hbm_gb", 0.0) or 0.0)
        sampler_sample_count = int(getattr(e, "_autotune_sampler_sample_count", 0) or 0)
        tb = traceback.format_exc()
        error_text = f"[stage={failure_stage}] {type(e).__name__}: {e}\n{tb}".strip()
        is_oom = _looks_like_oom(error_text)
        result = {
            "mean_step_time_seconds": float("inf"),
            "max_step_time_seconds": float("inf"),
            "min_step_time_seconds": float("inf"),
            "peak_memory_gb": runtime_peak_hbm_gb,
            "compile_peak_memory_gb": 0.0,
            "runtime_peak_hbm_gb": runtime_peak_hbm_gb,
            "tflops_per_device": 0.0,
            "oom": is_oom,
            "error": error_text,
            "exit_class": EXIT_CLASS_CONFIRMED_OOM if is_oom else EXIT_CLASS_INFRA_ERROR,
            "sampler_sample_count": sampler_sample_count,
            "failure_stage": failure_stage,
        }

    _write_result(result)

    # Release TPU HBM via jax.clear_backends(), then force-exit.
    # clear_backends() notifies libtpu to reclaim device memory.
    # os._exit(0) afterwards avoids atexit/finalizer deadlocks
    # (e.g. finalizers touching the already-cleared backend).
    # SIGALRM guards against clear_backends() itself hanging after OOM.
    def _force_exit(signum, frame):
        del signum, frame
        os._exit(0)

    signal.signal(signal.SIGALRM, _force_exit)
    signal.alarm(30)

    try:
        if hasattr(jax, "clear_caches"):
            jax.clear_caches()
        if hasattr(jax, "clear_backends"):
            jax.clear_backends()
    except Exception:
        pass
    gc.collect()
    # Give libtpu time to finish async HBM deallocation before process dies.
    time.sleep(5)
    os._exit(0)


if __name__ == "__main__":
    main()
