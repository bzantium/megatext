"""Subprocess worker for profiling a single autotune candidate.

Invoked by profile_candidate() as a subprocess to isolate JAX state.
Each invocation initializes JAX from scratch, profiles, and writes
the result as JSON to a file. The parent process reads this file.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax

from megatext.utils import logging as max_logging


def _run(config_overrides: dict, candidate_dict: dict, num_steps: int, warmup_steps: int) -> dict:
    """Run profiling and return result as a dict."""
    from megatext.autotune.strategies import Candidate
    from megatext.utils.constants import MEGATEXT_PKG_DIR

    candidate = Candidate(**candidate_dict)

    overrides = candidate.to_overrides()
    overrides["steps"] = warmup_steps + num_steps
    overrides["enable_checkpointing"] = False
    overrides["dataset_type"] = "synthetic"
    overrides["allow_split_physical_axes"] = True
    overrides["gradient_accumulation_steps"] = 1
    overrides["log_config"] = False

    # Merge candidate overrides into a new dict (don't mutate caller's dict)
    config_overrides = {**config_overrides, **overrides}

    max_logging.log(f"Profiling: {candidate} (steps={warmup_steps}w+{num_steps})")

    from megatext.configs import pyconfig

    # Build argv for pyconfig.initialize:
    #   ["script_name", "base.yaml", "key1=val1", "key2=val2", ...]
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

    (
        init_rng, _, state_mesh_shardings, model, mesh,
        learning_rate_schedule, data_iterator, data_loader,
        rampup_manager, _, state,
    ) = train_utils.setup_train_loop(megatext_config, recorder)

    from megatext.utils import sharding
    params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(
        megatext_config, state_mesh_shardings
    )

    p_train_step, _ = train_utils.jit_train_and_eval_step(
        megatext_config, model, mesh, state, state_mesh_shardings,
        mt_pretrain.train_step, mt_pretrain.eval_step, None, params_shardings,
    )

    from megatext.utils.flops import calculate_tflops_training_per_device
    total_tflops, _, _ = calculate_tflops_training_per_device(megatext_config, log=False)

    # Profile
    step_times = []
    _fold_in = jax.jit(jax.random.fold_in)
    for step in range(warmup_steps + num_steps):
        example_batch = data_loader.load_next_batch(rampup_manager=rampup_manager)
        nextrng = _fold_in(init_rng, step)

        start = time.monotonic()
        with jax.set_mesh(mesh), nn_partitioning.axis_rules(megatext_config.logical_axis_rules):
            state, metrics = p_train_step(state, example_batch, nextrng)
        jax.block_until_ready(state)
        elapsed = time.monotonic() - start

        if step >= warmup_steps:
            step_times.append(elapsed)

    # Query peak HBM via tpustat
    peak_memory_gb = 0.0
    try:
        from tpustat.core import TPUStatCollection
        stats = TPUStatCollection.new_query()
        if stats.devices:
            peak_memory_gb = max(d.hbm_used_mib for d in stats.devices) / 1024
    except Exception:
        pass

    mean_step_time = sum(step_times) / len(step_times)
    tflops_per_device = total_tflops / mean_step_time if mean_step_time > 0 else 0.0

    return {
        "mean_step_time_seconds": mean_step_time,
        "max_step_time_seconds": max(step_times),
        "min_step_time_seconds": min(step_times),
        "peak_memory_gb": peak_memory_gb,
        "tflops_per_device": tflops_per_device,
        "oom": False,
        "error": None,
    }


def main():
    import absl.logging
    absl.logging.set_verbosity(absl.logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()

    config_overrides = json.loads(args.config_json)
    candidate_dict = json.loads(args.candidate_json)

    try:
        result = _run(config_overrides, candidate_dict, args.num_steps, args.warmup_steps)
    except Exception as e:
        err_str = str(e).lower()
        is_oom = any(
            p in err_str
            for p in ("out of memory", "resource_exhausted", "resource exhausted",
                      "hbmoom", "oom", "vmemoom", "scoped vmem")
        )
        result = {
            "mean_step_time_seconds": float("inf"),
            "max_step_time_seconds": float("inf"),
            "min_step_time_seconds": float("inf"),
            "peak_memory_gb": 0.0,
            "tflops_per_device": 0.0,
            "oom": is_oom,
            "error": str(e) if not is_oom else None,
        }

    # Write result to file
    with open(args.result_file, "w") as f:
        json.dump(result, f)

    # Force immediate process termination to release ALL memory.
    os._exit(0)


if __name__ == "__main__":
    main()
