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

from __future__ import annotations

from types import SimpleNamespace

from megatext.trainers.profile import (
    ProfileStepResult,
    RuntimeMemorySnapshot,
    _format_step_ranges,
    _select_measured_steps,
    capture_runtime_memory_snapshot,
    summarize_stablehlo_text,
)


def test_summarize_stablehlo_text_counts_patterns():
  text = """
module {
  %0 = stablehlo.while(...)
  %1 = call @foo
  %2 = call @bar
  %3 = stablehlo.dot_general %a, %b
  %4 = stablehlo.dot_general %c, %d
  %5 = stablehlo.case %pred
  %6 = custom_call @baz
}
"""
  summary = summarize_stablehlo_text(text)

  assert summary.text_length_chars == len(text)
  assert summary.while_count == 1
  assert summary.call_count == 2
  assert summary.custom_call_count == 1
  assert summary.dot_general_count == 2
  assert summary.conditional_count == 1


def test_capture_runtime_memory_snapshot_handles_available_stats(monkeypatch):
  fake_device = SimpleNamespace(
      memory_stats=lambda: {
          "bytes_in_use": 3 * 1024**3,
          "bytes_limit": 8 * 1024**3,
          "peak_bytes_in_use": 5 * 1024**3,
      }
  )
  monkeypatch.setattr("megatext.trainers.profile.jax.local_devices", lambda: [fake_device])

  snapshot = capture_runtime_memory_snapshot()

  assert snapshot.used_gb == 3.0
  assert snapshot.limit_gb == 8.0
  assert snapshot.peak_gb == 5.0
  assert "bytes_in_use" in snapshot.raw_keys


def test_capture_runtime_memory_snapshot_handles_missing_stats(monkeypatch):
  fake_device = SimpleNamespace(memory_stats=lambda: None)
  monkeypatch.setattr("megatext.trainers.profile.jax.local_devices", lambda: [fake_device])

  snapshot = capture_runtime_memory_snapshot()

  assert snapshot.used_gb is None
  assert snapshot.limit_gb is None
  assert snapshot.peak_gb is None
  assert snapshot.raw_keys == ()


def test_format_step_ranges_compacts_consecutive_runs():
  assert _format_step_ranges([3, 4, 6, 8, 9, 10]) == "3-4, 6, 8-10"


def test_select_measured_steps_skips_post_profile_sync_step():
  snapshot = RuntimeMemorySnapshot(None, None, None, ())
  results = [
      ProfileStepResult(
          step=step,
          step_time_seconds=1.0,
          tflops_per_device_per_sec=1.0,
          tokens_per_device_per_sec=1.0,
          loss=None,
          runtime_memory=snapshot,
      )
      for step in range(1, 9)
  ]

  measured = _select_measured_steps(
      results,
      skip_first_steps=2,
      profiler_steps=3,
      excluded_steps=frozenset({5}),
  )

  assert [result.step for result in measured] == [3, 4, 6]
