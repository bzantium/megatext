# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Detailed profiling entrypoint for Megatext training jobs.

This trainer reuses the regular training setup and train_step path, but emits a
more detailed profiling log than `megatext.trainers.pretrain`:
  - compile duration
  - StableHLO summary
  - compiled memory analysis
  - per-step timing / TFLOPs / tokens
  - best-effort runtime HBM peak sampling
"""

from __future__ import annotations

import contextlib
import dataclasses
import re
import threading
import time
from typing import Any, Sequence

from absl import app

import jax
from flax.linen import partitioning as nn_partitioning

from megatext.common import profiler
from megatext.common.common_types import MODEL_MODE_TRAIN
from megatext.common.gcloud_stub import is_decoupled
from megatext.common.goodput import RECORD_JOB_END_TIME, RECORD_JOB_START_TIME, record_goodput
from megatext.configs import pyconfig
from megatext.trainers import pretrain as pretrain_trainer
from megatext.utils import logging as max_logging
from megatext.utils import sharding, train_utils
from megatext.utils.debug import print_compiled_memory_stats, print_mem_stats
from megatext.utils.flops import calculate_tflops_training_per_device, calculate_tokens_training_per_device
from megatext.utils.training import maybe_get_transformer_engine_context


@dataclasses.dataclass(frozen=True)
class HloSummary:
  text_length_chars: int
  while_count: int
  call_count: int
  custom_call_count: int
  dot_general_count: int
  conditional_count: int


@dataclasses.dataclass(frozen=True)
class RuntimeMemorySnapshot:
  used_gb: float | None
  limit_gb: float | None
  peak_gb: float | None
  raw_keys: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ProfileStepResult:
  step: int
  step_time_seconds: float
  tflops_per_device_per_sec: float
  tokens_per_device_per_sec: float
  loss: float | None
  runtime_memory: RuntimeMemorySnapshot
  sampled_peak_hbm_gb: float | None = None


def _format_step_ranges(steps: Sequence[int]) -> str:
  if not steps:
    return ""
  ordered = sorted(dict.fromkeys(int(step) for step in steps))
  ranges: list[str] = []
  start = ordered[0]
  end = ordered[0]
  for step in ordered[1:]:
    if step == end + 1:
      end = step
      continue
    ranges.append(f"{start}-{end}" if start != end else str(start))
    start = end = step
  ranges.append(f"{start}-{end}" if start != end else str(start))
  return ", ".join(ranges)


def _select_measured_steps(
    profile_results: Sequence[ProfileStepResult],
    *,
    skip_first_steps: int,
    profiler_steps: int,
    excluded_steps: frozenset[int] = frozenset(),
) -> list[ProfileStepResult]:
  candidates = [result for result in profile_results if result.step > skip_first_steps and result.step not in excluded_steps]
  measured_steps = candidates[:profiler_steps]
  if measured_steps:
    return measured_steps
  fallback = [result for result in profile_results if result.step not in excluded_steps]
  return fallback or list(profile_results)


def _bytes_to_gb(num_bytes: int | float | None) -> float | None:
  if num_bytes is None:
    return None
  return float(num_bytes) / (1024 ** 3)


def summarize_stablehlo_text(text: str) -> HloSummary:
  return HloSummary(
      text_length_chars=len(text),
      while_count=text.count("stablehlo.while"),
      call_count=len(re.findall(r"=\s+call\s+@", text)),
      custom_call_count=text.count("custom_call"),
      dot_general_count=text.count("stablehlo.dot_general"),
      conditional_count=text.count("stablehlo.case") + text.count("stablehlo.if"),
  )


def _extract_first_float(mapping: dict[str, Any], keys: tuple[str, ...]) -> float | None:
  for key in keys:
    value = mapping.get(key)
    if value is None:
      continue
    try:
      return float(value)
    except (TypeError, ValueError):
      continue
  return None


def capture_runtime_memory_snapshot() -> RuntimeMemorySnapshot:
  used_bytes = None
  limit_bytes = None
  peak_bytes = None
  raw_keys: set[str] = set()
  try:
    for device in jax.local_devices():
      stats = device.memory_stats()
      if not stats:
        continue
      raw_keys.update(stats.keys())
      used_candidate = _extract_first_float(stats, ("bytes_in_use", "bytes_reserved"))
      limit_candidate = _extract_first_float(stats, ("bytes_limit",))
      peak_candidate = _extract_first_float(
          stats,
          (
              "peak_bytes_in_use",
              "bytes_peak",
              "peak_bytes_reserved",
              "peak_bytes",
          ),
      )
      if used_candidate is not None:
        used_bytes = max(used_bytes or 0.0, used_candidate)
      if limit_candidate is not None:
        limit_bytes = max(limit_bytes or 0.0, limit_candidate)
      if peak_candidate is not None:
        peak_bytes = max(peak_bytes or 0.0, peak_candidate)
  except Exception:
    return RuntimeMemorySnapshot(None, None, None, ())

  return RuntimeMemorySnapshot(
      used_gb=_bytes_to_gb(used_bytes),
      limit_gb=_bytes_to_gb(limit_bytes),
      peak_gb=_bytes_to_gb(peak_bytes),
      raw_keys=tuple(sorted(raw_keys)),
  )


def _sample_runtime_hbm_peak(
    stop_event: threading.Event,
    out: dict[str, float | int | bool],
    lock: threading.Lock,
    interval_seconds: float = 0.2,
) -> None:
  """Best-effort HBM sampler using tpustat."""
  try:
    from tpustat.core import TPUStatCollection  # pylint: disable=import-outside-toplevel
  except Exception:
    with lock:
      out["available"] = False
    return

  with lock:
    out["available"] = True
    out.setdefault("runtime_peak_hbm_gb", 0.0)
    out.setdefault("sample_count", 0)
    out.setdefault("window_peak_hbm_gb", 0.0)
    out.setdefault("window_sample_count", 0)
  peak_gb = 0.0
  sample_count = 0
  while not stop_event.is_set():
    try:
      stats = TPUStatCollection.new_query()
      devices = getattr(stats, "devices", []) or []
      used_gb = 0.0
      for device in devices:
        mib = None
        for attr in ("hbm_used_mib", "hbm_usage_mib", "hbm_used", "memory_used_mib"):
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
      with lock:
        out["runtime_peak_hbm_gb"] = max(float(out.get("runtime_peak_hbm_gb", 0.0) or 0.0), used_gb)
        out["window_peak_hbm_gb"] = max(float(out.get("window_peak_hbm_gb", 0.0) or 0.0), used_gb)
        out["sample_count"] = int(out.get("sample_count", 0) or 0) + 1
        out["window_sample_count"] = int(out.get("window_sample_count", 0) or 0) + 1
    except Exception:
      pass
    stop_event.wait(interval_seconds)

  with lock:
    out["runtime_peak_hbm_gb"] = max(float(out.get("runtime_peak_hbm_gb", 0.0) or 0.0), peak_gb)
    out["sample_count"] = max(int(out.get("sample_count", 0) or 0), sample_count)


def _reset_runtime_hbm_window(state: dict[str, float | int | bool], lock: threading.Lock) -> None:
  with lock:
    state["window_peak_hbm_gb"] = 0.0
    state["window_sample_count"] = 0


def _read_runtime_hbm_window(state: dict[str, float | int | bool], lock: threading.Lock) -> tuple[float | None, int]:
  with lock:
    peak = state.get("window_peak_hbm_gb")
    sample_count = int(state.get("window_sample_count", 0) or 0)
  if peak is None:
    return None, sample_count
  return float(peak), sample_count


def _log_profile_header(config, total_tflops_per_device: float, tokens_per_device: float) -> None:
  max_logging.log("=== Megatext Profile Run ===")
  max_logging.log(
      "Config: "
      f"model={config.model}, decoder_block={config.decoder_block}, "
      f"scan_layers={config.scan_layers}, remat_policy={config.remat_policy}, "
      f"attention={config.attention}, sliding_window_size={config.sliding_window_size}"
  )
  max_logging.log(
      "Shape: "
      f"layers={config.num_decoder_layers}, batch/device={config.per_device_batch_size}, "
      f"seq={config.max_target_length}, vocab={config.vocab_size}"
  )
  max_logging.log(
      "Theoretical throughput base: "
      f"{total_tflops_per_device:.2f} TFLOPs/device/step, "
      f"{tokens_per_device:.1f} tokens/device/step"
  )


def _log_hlo_summary(summary: HloSummary) -> None:
  max_logging.log(
      "StableHLO summary: "
      f"text={summary.text_length_chars} chars, "
      f"while={summary.while_count}, "
      f"call={summary.call_count}, "
      f"custom_call={summary.custom_call_count}, "
      f"dot_general={summary.dot_general_count}, "
      f"conditionals={summary.conditional_count}"
  )


def _log_runtime_memory(prefix: str, snapshot: RuntimeMemorySnapshot) -> None:
  if snapshot.used_gb is None and snapshot.limit_gb is None and snapshot.peak_gb is None:
    max_logging.log(f"{prefix}: runtime memory stats unavailable")
    return
  details = []
  if snapshot.used_gb is not None:
    details.append(f"used={snapshot.used_gb:.2f}GB")
  if snapshot.limit_gb is not None:
    details.append(f"limit={snapshot.limit_gb:.2f}GB")
  if snapshot.peak_gb is not None:
    details.append(f"peak={snapshot.peak_gb:.2f}GB")
  if snapshot.raw_keys:
    details.append(f"keys={','.join(snapshot.raw_keys[:6])}")
  max_logging.log(f"{prefix}: " + ", ".join(details))


def _extract_loss(metrics: dict[str, Any]) -> float | None:
  scalar = metrics.get("scalar")
  if not isinstance(scalar, dict):
    return None
  value = scalar.get("learning/loss")
  if value is None:
    return None
  try:
    return float(value)
  except (TypeError, ValueError):
    try:
      return float(jax.device_get(value))
    except Exception:
      return None


def _compile_profiled_train_step(config, state, shaped_batch, init_rng, p_train_step) -> tuple[Any, HloSummary]:
  lower_started = time.monotonic()
  lowered = p_train_step.lower(state, shaped_batch, init_rng)
  lower_elapsed = time.monotonic() - lower_started
  max_logging.log(f"Lowering finished in {lower_elapsed:.2f}s")

  stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
  hlo_summary = summarize_stablehlo_text(stablehlo_text)
  _log_hlo_summary(hlo_summary)

  compile_started = time.monotonic()
  compiled = lowered.compile()
  compile_elapsed = time.monotonic() - compile_started
  max_logging.log(f"Compilation finished in {compile_elapsed:.2f}s")
  return compiled, hlo_summary


def profile_loop(config, recorder, state=None):  # pylint: disable=unused-argument
  (
      init_rng,
      _checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      _data_iterator,
      data_loader,
      rampup_manager,
      _eval_data_iterator,
      state,
  ) = train_utils.setup_train_loop(config, recorder)

  params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)
  total_tflops_per_device, _, _ = calculate_tflops_training_per_device(config, log=False)
  tokens_per_device = calculate_tokens_training_per_device(config)
  _log_profile_header(config, total_tflops_per_device, tokens_per_device)

  with jax.set_mesh(mesh), mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    p_train_step, _ = train_utils.jit_train_and_eval_step(
        config,
        model,
        mesh,
        state,
        state_mesh_shardings,
        pretrain_trainer.train_step,
        pretrain_trainer.eval_step,
        None,
        params_shardings,
    )
    shaped_batch = train_utils.get_shaped_batch(config)
    if config.shard_optimizer_over_data:
      state = sharding.maybe_shard_with_name(state, state_mesh_shardings, config.shard_mode)
    compiled, _ = _compile_profiled_train_step(config, state, shaped_batch, init_rng, p_train_step)
    compiled_stats = compiled.memory_analysis()
    print_compiled_memory_stats(compiled_stats)

  print_mem_stats("After train_step compilation")
  initial_runtime_memory = capture_runtime_memory_snapshot()
  _log_runtime_memory("Initial local runtime memory snapshot", initial_runtime_memory)

  prof = profiler.Profiler(config, offset_step=0)
  profile_results: list[ProfileStepResult] = []
  runtime_hbm_state: dict[str, float | int | bool] = {
      "runtime_peak_hbm_gb": 0.0,
      "sample_count": 0,
      "available": False,
      "window_peak_hbm_gb": 0.0,
      "window_sample_count": 0,
  }
  runtime_hbm_lock = threading.Lock()
  sampler_stop = threading.Event()
  sampler = threading.Thread(
      target=_sample_runtime_hbm_peak,
      args=(sampler_stop, runtime_hbm_state, runtime_hbm_lock),
      name="profile-hbm-sampler",
      daemon=True,
  )
  sampler.start()
  profile_terminated_step = None
  if prof.mode != "" and prof.finished_initial_profile_step + 1 <= config.steps:
    profile_terminated_step = prof.finished_initial_profile_step + 1

  try:
    for step in range(1, config.steps + 1):
      prof.maybe_activate_profiler(step, state)
      example_batch = data_loader.load_next_batch(rampup_manager=rampup_manager)
      nextrng = jax.jit(jax.random.fold_in)(init_rng, step)

      _reset_runtime_hbm_window(runtime_hbm_state, runtime_hbm_lock)
      started = time.monotonic()
      with jax.profiler.StepTraceAnnotation("profile", step_num=step):
        with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
          if config.shard_optimizer_over_data:
            state = sharding.maybe_shard_with_name(state, state_mesh_shardings, config.shard_mode)
          state, metrics = p_train_step(state, example_batch, nextrng)
      jax.block_until_ready(state)
      jax.block_until_ready(metrics)
      step_time = time.monotonic() - started
      runtime_memory = capture_runtime_memory_snapshot()
      sampled_peak_hbm_gb, sampled_peak_samples = _read_runtime_hbm_window(runtime_hbm_state, runtime_hbm_lock)
      loss = _extract_loss(metrics)
      result = ProfileStepResult(
          step=step,
          step_time_seconds=step_time,
          tflops_per_device_per_sec=(total_tflops_per_device / step_time) if step_time > 0 else 0.0,
          tokens_per_device_per_sec=(tokens_per_device / step_time) if step_time > 0 else 0.0,
          loss=loss,
          runtime_memory=runtime_memory,
          sampled_peak_hbm_gb=sampled_peak_hbm_gb,
      )
      profile_results.append(result)
      prof.maybe_deactivate_profiler(step, state)

      step_notes = []
      if profile_terminated_step is not None and step == profile_terminated_step:
        step_notes.append("profile terminated")
      memory_parts = []
      if sampled_peak_hbm_gb is not None and sampled_peak_samples > 0:
        memory_parts.append(f"sampled_peak={sampled_peak_hbm_gb:.2f}GB")
      loss_part = f", loss={loss:.4f}" if loss is not None else ""
      max_logging.log(
          f"[profile step {step:03d}] "
          f"{step_time:.3f}s, "
          f"{result.tflops_per_device_per_sec:.1f} TFLOPs/s/device, "
          f"{result.tokens_per_device_per_sec:.1f} tokens/s/device"
          f"{loss_part}"
          + (f", {', '.join(memory_parts)}" if memory_parts else "")
          + (f" ({'; '.join(step_notes)})" if step_notes else "")
      )
  finally:
    sampler_stop.set()
    sampler.join(timeout=2.0)

  excluded_steps = frozenset({profile_terminated_step}) if profile_terminated_step is not None else frozenset()
  measured_steps = _select_measured_steps(
      profile_results,
      skip_first_steps=config.skip_first_n_steps_for_profiler,
      profiler_steps=config.profiler_steps,
      excluded_steps=excluded_steps,
  )

  mean_step_time = sum(r.step_time_seconds for r in measured_steps) / len(measured_steps)
  mean_tflops = sum(r.tflops_per_device_per_sec for r in measured_steps) / len(measured_steps)
  mean_tokens = sum(r.tokens_per_device_per_sec for r in measured_steps) / len(measured_steps)
  runtime_peak_from_sampler = float(runtime_hbm_state.get("runtime_peak_hbm_gb", 0.0) or 0.0)
  runtime_peak_from_snapshots = max(
      (r.runtime_memory.peak_gb or r.runtime_memory.used_gb or 0.0) for r in profile_results
  )
  runtime_peak = max(runtime_peak_from_sampler, runtime_peak_from_snapshots)
  compile_peak = 0.0
  if compiled_stats is not None:
    compile_peak = (
        compiled_stats.temp_size_in_bytes
        + compiled_stats.argument_size_in_bytes
        + compiled_stats.output_size_in_bytes
        - compiled_stats.alias_size_in_bytes
    ) / (1024 ** 3)

  max_logging.log("=== Profile Summary ===")
  measured_step_label = _format_step_ranges([result.step for result in measured_steps])
  exclusion_suffix = (
      f", excluded post-profile-sync step {profile_terminated_step}"
      if profile_terminated_step is not None and any(result.step == profile_terminated_step for result in profile_results)
      else ""
  )
  max_logging.log(
      f"Measured clean steps: {measured_step_label} "
      f"(skip_first_n_steps_for_profiler={config.skip_first_n_steps_for_profiler}, "
      f"profiler_steps={config.profiler_steps}{exclusion_suffix})"
  )
  max_logging.log(
      f"Mean step time: {mean_step_time:.3f}s, "
      f"Mean TFLOPs/s/device: {mean_tflops:.1f}, "
      f"Mean tokens/s/device: {mean_tokens:.1f}"
  )
  max_logging.log(
      f"Peak memory: compile={compile_peak:.2f}GB, runtime={runtime_peak:.2f}GB, "
      f"sampler_samples={int(runtime_hbm_state.get('sample_count', 0) or 0)}"
  )
  if profile_results:
    fastest = min(profile_results, key=lambda r: r.step_time_seconds)
    slowest = max(profile_results, key=lambda r: r.step_time_seconds)
    max_logging.log(
        f"Fastest step: {fastest.step} ({fastest.step_time_seconds:.3f}s), "
        f"Slowest step: {slowest.step} ({slowest.step_time_seconds:.3f}s)"
    )

  return state


def initialize(argv: Sequence[str]) -> tuple[pyconfig.HyperParameters, Any, Any]:
  return pretrain_trainer.initialize(argv)


def run(config, recorder, diagnostic_config):
  diagnostics_context = (
      contextlib.nullcontext()
      if is_decoupled() or getattr(pretrain_trainer.diagnostic, "__class__", None).__name__ == "_StubDiag"
      else pretrain_trainer.diagnostic.diagnose(diagnostic_config)
  )

  if is_decoupled() or getattr(pretrain_trainer.diagnostic, "__class__", None).__name__ == "_StubDiag":
    max_logging.log("[DECOUPLED NO-OP] skipping cloud diagnostics wrapper.")

  with (
      diagnostics_context,
      maybe_get_transformer_engine_context(config),
  ):
    profile_loop(config, recorder)


def main(argv: Sequence[str]) -> None:
  config, recorder, diagnostic_config = initialize(argv)
  record_goodput(recorder, RECORD_JOB_START_TIME)
  with pretrain_trainer.maybe_monitor_goodput(config):
    try:
      run(config, recorder, diagnostic_config)
    finally:
      record_goodput(recorder, RECORD_JOB_END_TIME)


if __name__ == "__main__":
  app.run(main)
