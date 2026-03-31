# Copyright 2023–2025 Google LLC
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

"""Logging utilities."""
import logging as std_logging
import os
import subprocess
from pathlib import Path

from absl import logging

import jax

from megatext.common.gcloud_stub import is_decoupled
from megatext.common.gcloud_stub import writer, _TENSORBOARDX_AVAILABLE


def log(user_str):
  """Logs a message at the INFO level."""
  # Note, stacklevel=2 makes the log show the caller of this function.
  logging.info(user_str, stacklevel=2)


def debug(user_str):
  """Logs a message at the DEBUG level."""
  logging.debug(user_str, stacklevel=2)


def info(user_str, stacklevel=2):
  """Logs a message at the INFO level."""
  logging.info(user_str, stacklevel=stacklevel)


def warning(user_str):
  """Logs a message at the WARNING level."""
  logging.warning(user_str, stacklevel=2)


def error(user_str):
  """Logs a message at the ERROR level."""
  logging.error(user_str, stacklevel=2)


# Define filter at module level to avoid pickling issues and ensure visibility
class NoisyLogFilter(std_logging.Filter):
  """
  Class for defining log patterns to filter out
  """

  def filter(self, record):
    # Get the message; check both the raw msg and formatted message
    msg = record.getMessage()
    # Suppress "Type mismatch" warnings from tunix/generate/utils.py
    if "Type mismatch on" in msg:
      return False
    if "No mapping for flat state" in msg:
      return False
    return True


# ---------------------------------------------------------------------------
# TensorBoard helpers (moved from max_utils.py)
# ---------------------------------------------------------------------------


def initialize_summary_writer(tensorboard_dir, run_name):
  """Return a tensorboardX SummaryWriter or a no-op stub.

  In decoupled mode (no Google Cloud), this prefers a repo-local
  ``local_tensorboard`` directory when tensorboardX is available.
  """
  if jax.process_index() != 0:
    return None

  if not _TENSORBOARDX_AVAILABLE:
    log("tensorboardX not available; using no-op SummaryWriter.")
    return writer.SummaryWriter()

  if is_decoupled():
    # decoupled and tensorboardX is available -> write to repo-local 'local_tensorboard'
    try:
      repo_tb = Path(__file__).resolve().parents[2] / "local_tensorboard"
      repo_tb.mkdir(parents=True, exist_ok=True)
      summary_writer_path = str(repo_tb / run_name) if run_name else str(repo_tb)
      log(f"Decoupled: using local tensorboard dir {summary_writer_path}")
      return writer.SummaryWriter(summary_writer_path)
    except Exception as e:  # pylint: disable=broad-exception-caught
      log(f"Decoupled: failed to use local tensorboard dir: {e}; using no-op SummaryWriter.")
      return writer.SummaryWriter()

  # Check if dir or run_name exists!
  if not tensorboard_dir or not run_name:
    log("tensorboard_dir or run_name missing; using no-op SummaryWriter to avoid crash.")
    return writer.SummaryWriter()

  summary_writer_path = os.path.join(tensorboard_dir, run_name)
  return writer.SummaryWriter(summary_writer_path)


def close_summary_writer(summary_writer):
  if jax.process_index() == 0:
    summary_writer.close()


def add_text_to_summary_writer(key, value, summary_writer):
  """Writes given key-value pair to tensorboard as text/summary."""
  if jax.process_index() == 0:
    summary_writer.add_text(key, value)


def get_project():
  """Get project"""
  if is_decoupled():
    return os.environ.get("LOCAL_GCLOUD_PROJECT", "local-maxtext-project")
  try:
    completed_command = subprocess.run(["gcloud", "config", "get", "project"], check=True, capture_output=True)
    project_outputs = completed_command.stdout.decode().strip().split("\n")
    if len(project_outputs) < 1 or project_outputs[-1] == "":
      log("You must specify config.vertex_tensorboard_project or set 'gcloud config set project <project>'")
      return None
    return project_outputs[-1]
  except (FileNotFoundError, subprocess.CalledProcessError) as ex:
    log(f"Unable to retrieve gcloud project (decoupled={is_decoupled()}): {ex}")
    return None
