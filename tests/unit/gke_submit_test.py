# Copyright 2025 Google LLC
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

"""Unit tests for the GKE submit helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_submit_module():
  repo_root = Path(__file__).resolve().parents[2]
  submit_path = repo_root / "gke" / "submit.py"
  spec = importlib.util.spec_from_file_location("gke_submit", submit_path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


submit = _load_submit_module()


def test_build_profile_job_overrides_synthetic_inputs_without_mutating_source():
  base_job = {
      "workload_name": "pretrain-qwen3",
      "bucket": "lmt-tpu-datasets",
      "mount_path": "/mnt/bucket",
      "config": {
          "run_name": "pretrain-qwen3",
          "dataset_type": "fixed_arecord",
          "dataset_path": "/mnt/bucket/data",
          "data_cache_dir": "gs://cache/indices",
          "steps": 50_000,
          "profiler": "",
          "enable_checkpointing": True,
          "save_checkpoint_on_completion": True,
          "eval_interval": 100,
          "log_period": 100,
      },
  }

  profile_job = submit._build_profile_job(
      base_job,
      steps=10,
      profiler_steps=3,
      skip_first_steps=2,
      dataset_type="synthetic",
  )

  assert profile_job["workload_name"] == "profile-pretrain-qwen3"
  assert "bucket" not in profile_job
  assert "mount_path" not in profile_job
  assert profile_job["config"]["run_name"] == "profile-pretrain-qwen3"
  assert profile_job["config"]["dataset_type"] == "synthetic"
  assert "dataset_path" not in profile_job["config"]
  assert "data_cache_dir" not in profile_job["config"]
  assert profile_job["config"]["steps"] == 10
  assert profile_job["config"]["profiler"] == "xplane"
  assert profile_job["config"]["skip_first_n_steps_for_profiler"] == 2
  assert profile_job["config"]["profiler_steps"] == 3
  assert profile_job["config"]["profile_cleanly"] is True
  assert profile_job["config"]["enable_checkpointing"] is False
  assert profile_job["config"]["save_checkpoint_on_completion"] is False
  assert profile_job["config"]["eval_interval"] == -1
  assert profile_job["config"]["log_period"] == 1
  assert profile_job["config"]["jax_cache_dir"] == ""

  assert base_job["workload_name"] == "pretrain-qwen3"
  assert base_job["bucket"] == "lmt-tpu-datasets"
  assert base_job["mount_path"] == "/mnt/bucket"
  assert base_job["config"]["run_name"] == "pretrain-qwen3"
  assert base_job["config"]["dataset_type"] == "fixed_arecord"
  assert base_job["config"]["dataset_path"] == "/mnt/bucket/data"
  assert base_job["config"]["data_cache_dir"] == "gs://cache/indices"


def test_build_profile_job_keeps_real_dataset_mounts_when_not_synthetic():
  base_job = {
      "workload_name": "pretrain-qwen3",
      "bucket": "lmt-tpu-datasets",
      "mount_path": "/mnt/bucket",
      "config": {
          "run_name": "pretrain-qwen3",
          "dataset_type": "fixed_arecord",
          "dataset_path": "/mnt/bucket/data",
          "data_cache_dir": "gs://cache/indices",
      },
  }

  profile_job = submit._build_profile_job(
      base_job,
      steps=10,
      profiler_steps=3,
      skip_first_steps=2,
      dataset_type="fixed_arecord",
  )

  assert profile_job["bucket"] == "lmt-tpu-datasets"
  assert profile_job["mount_path"] == "/mnt/bucket"
  assert profile_job["config"]["dataset_type"] == "fixed_arecord"
  assert profile_job["config"]["dataset_path"] == "/mnt/bucket/data"
  assert profile_job["config"]["data_cache_dir"] == "gs://cache/indices"


def test_build_smoke_pretrain_job_overrides_steps_without_mutating_source():
  base_job = {
      "workload_name": "pretrain-qwen3",
      "config": {
          "run_name": "pretrain-qwen3",
          "steps": 50_000,
          "warmup_steps": 2_400,
          "per_device_batch_size": 5,
          "enable_checkpointing": True,
          "save_checkpoint_on_completion": True,
          "eval_interval": 100,
          "enable_tensorboard": True,
          "use_vertex_tensorboard": True,
          "gcs_metrics": True,
          "metrics_file": "/tmp/metrics.jsonl",
          "save_config_to_gcs": True,
      },
  }

  smoke_job = submit._build_smoke_pretrain_job(
      base_job,
      steps=5,
      warmup_steps=0,
  )

  assert smoke_job["workload_name"] == "smoke-pretrain-qwen3"
  assert smoke_job["config"]["run_name"] == "smoke-pretrain-qwen3"
  assert smoke_job["config"]["steps"] == 5
  assert smoke_job["config"]["warmup_steps"] == 0
  assert smoke_job["config"]["per_device_batch_size"] == 5
  assert smoke_job["config"]["enable_checkpointing"] is False
  assert smoke_job["config"]["save_checkpoint_on_completion"] is False
  assert smoke_job["config"]["eval_interval"] == -1
  assert smoke_job["config"]["enable_tensorboard"] is False
  assert smoke_job["config"]["use_vertex_tensorboard"] is False
  assert smoke_job["config"]["gcs_metrics"] is False
  assert smoke_job["config"]["metrics_file"] == ""
  assert smoke_job["config"]["save_config_to_gcs"] is False

  assert base_job["workload_name"] == "pretrain-qwen3"
  assert base_job["config"]["steps"] == 50_000
  assert base_job["config"]["warmup_steps"] == 2_400
  assert base_job["config"]["run_name"] == "pretrain-qwen3"
  assert base_job["config"]["enable_checkpointing"] is True
  assert base_job["config"]["save_checkpoint_on_completion"] is True
  assert base_job["config"]["eval_interval"] == 100
  assert base_job["config"]["enable_tensorboard"] is True
  assert base_job["config"]["use_vertex_tensorboard"] is True
  assert base_job["config"]["gcs_metrics"] is True
  assert base_job["config"]["metrics_file"] == "/tmp/metrics.jsonl"
  assert base_job["config"]["save_config_to_gcs"] is True


def test_build_autotune_job_prefixes_run_and_workload_names_without_mutating_source():
  base_job = {
      "workload_name": "pretrain-qwen3",
      "config": {
          "run_name": "pretrain-qwen3",
          "steps": 50_000,
      },
  }

  autotune_job = submit._build_autotune_job(base_job)

  assert autotune_job["workload_name"] == "autotune-pretrain-qwen3"
  assert autotune_job["config"]["run_name"] == "autotune-pretrain-qwen3"
  assert autotune_job["config"]["steps"] == 50_000

  assert base_job["workload_name"] == "pretrain-qwen3"
  assert base_job["config"]["run_name"] == "pretrain-qwen3"
  assert base_job["config"]["steps"] == 50_000


def test_apply_config_overrides_parses_scalars_without_mutating_source():
  base_job = {
      "workload_name": "pretrain-qwen3",
      "config": {
          "run_name": "pretrain-qwen3",
          "enable_checkpointing": True,
          "steps": 50_000,
      },
  }

  overridden_job = submit._apply_config_overrides(
      base_job,
      ["enable_checkpointing=false", "steps=5", "model=qwen3-swa"],
  )

  assert overridden_job["config"]["enable_checkpointing"] is False
  assert overridden_job["config"]["steps"] == 5
  assert overridden_job["config"]["model"] == "qwen3-swa"

  assert base_job["config"]["enable_checkpointing"] is True
  assert base_job["config"]["steps"] == 50_000
  assert "model" not in base_job["config"]
