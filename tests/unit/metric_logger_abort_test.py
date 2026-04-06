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

"""Tests for monitoring metrics"""
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from megatext.common.metric_logger import MetadataKey, MetricLogger


class MetricLoggerAbortTest(unittest.TestCase):
  def _make_logger(self, abort_on_nan_loss, abort_on_inf_loss):
    logger = MetricLogger.__new__(MetricLogger)  # skip __init__
    logger.config = SimpleNamespace(
        abort_on_nan_loss=abort_on_nan_loss,
        abort_on_inf_loss=abort_on_inf_loss,
        enable_tensorboard=True,
        metrics_file="/tmp/fake_metrics.jsonl",
        gcs_metrics=True,
        managed_mldiagnostics=True,
    )
    return logger

  def _metrics(self, loss):
    return {"scalar": {"learning/loss": loss}}

  @mock.patch("jax.process_index", return_value=0)
  def test_abort_on_nan_exits_after_writes(self, _):
    logger = self._make_logger(True, False)

    with (
        mock.patch.object(logger, "log_metrics") as log_metrics,
        mock.patch.object(logger, "write_metrics_to_tensorboard") as tb,
        mock.patch.object(logger, "write_metrics_locally") as local,
        mock.patch.object(logger, "write_metrics_for_gcs") as gcs,
        mock.patch.object(logger, "write_metrics_to_managed_mldiagnostics") as mldiag,
    ):
      with self.assertRaises(SystemExit) as cm:
        logger.write_metrics(self._metrics(np.nan), step=1, is_training=True)

    self.assertEqual(cm.exception.code, 1)
    log_metrics.assert_called_once()
    tb.assert_called_once()
    local.assert_called_once()
    gcs.assert_called_once()
    mldiag.assert_called_once()

  @mock.patch("jax.process_index", return_value=0)
  def test_abort_on_inf_exits_after_writes(self, _):
    logger = self._make_logger(False, True)
    with mock.patch.object(logger, "log_metrics"), \
         mock.patch.object(logger, "write_metrics_to_tensorboard"), \
         mock.patch.object(logger, "write_metrics_locally"), \
         mock.patch.object(logger, "write_metrics_for_gcs"), \
         mock.patch.object(logger, "write_metrics_to_managed_mldiagnostics"):
      with self.assertRaises(SystemExit):
        logger.write_metrics(self._metrics(np.inf), step=1, is_training=True)

  def test_finite_loss_does_not_exit(self):
    logger = self._make_logger(True, True)
    with mock.patch.object(logger, "log_metrics"), \
         mock.patch.object(logger, "write_metrics_to_tensorboard"), \
         mock.patch.object(logger, "write_metrics_locally"), \
         mock.patch.object(logger, "write_metrics_to_managed_mldiagnostics"), \
         mock.patch("jax.process_index", return_value=1):  # skip gcs branch
      logger.write_metrics(self._metrics(1.23), step=1, is_training=True)

  def test_abort_flags_disabled_does_not_exit(self):
    logger = self._make_logger(False, False)
    with mock.patch.object(logger, "log_metrics"), \
         mock.patch.object(logger, "write_metrics_to_tensorboard"), \
         mock.patch.object(logger, "write_metrics_locally"), \
         mock.patch.object(logger, "write_metrics_to_managed_mldiagnostics"), \
         mock.patch("jax.process_index", return_value=1):
      logger.write_metrics(self._metrics(np.nan), step=1, is_training=True)

  def test_init_uses_noop_writer_when_tensorboard_disabled(self):
    config = SimpleNamespace(
        tensorboard_dir="/tmp/tb",
        run_name="smoke-run",
        enable_tensorboard=False,
        gcs_metrics=False,
        managed_mldiagnostics=False,
        report_heartbeat_metric_for_gcp_monitoring=False,
        report_performance_metric_for_gcp_monitoring=False,
    )
    lr_schedule = lambda _: 0.0

    with mock.patch("megatext.common.metric_logger.initialize_summary_writer") as init_writer:
      MetricLogger(config, lr_schedule)

    init_writer.assert_called_once_with(config.tensorboard_dir, enabled=False)

  def test_write_setup_info_skips_tensorboard_writes_when_disabled(self):
    logger = MetricLogger.__new__(MetricLogger)
    logger.config = SimpleNamespace(enable_tensorboard=False)
    logger.metadata = {}
    logger.writer = object()

    with (
        mock.patch("megatext.common.metric_logger.calculate_tflops_training_per_device", return_value=(1.5, None, None)),
        mock.patch("megatext.common.metric_logger.calculate_tokens_training_per_device", return_value=2048),
        mock.patch("megatext.common.metric_logger.add_text_to_summary_writer") as add_text,
        mock.patch("megatext.common.metric_logger.add_config_to_summary_writer") as add_config,
    ):
      logger.write_setup_info_to_tensorboard({"params": np.zeros((2, 2))})

    self.assertEqual(logger.metadata[MetadataKey.PER_DEVICE_TFLOPS], 1.5)
    self.assertEqual(logger.metadata[MetadataKey.PER_DEVICE_TOKENS], 2048)
    add_text.assert_not_called()
    add_config.assert_not_called()

  def test_log_training_metrics_includes_update_norm_and_omits_total_weights(self):
    logger = MetricLogger.__new__(MetricLogger)
    logger.config = SimpleNamespace(
        rampup_end_step=0,
        hide_profiler_step_metric=False,
        mtp_num_layers=0,
    )
    metrics = {
        "scalar": {
            "learning/loss": 1.23,
            "perf/step_time_seconds": 4.56,
            "perf/per_device_tflops_per_sec": 7.89,
            "perf/per_device_tokens_per_sec": 10.11,
            "learning/current_learning_rate": 1.0e-4,
            "learning/total_weights": 4096,
            "learning/grad_norm": 0.5,
            "learning/update_norm": 0.25,
        }
    }

    with mock.patch("megatext.common.metric_logger.max_logging.log") as log:
      logger._log_training_metrics(metrics, step=3)

    logged = log.call_args.args[0]
    self.assertIn("update_norm: 2.500e-01", logged)
    self.assertNotIn("total_weights", logged)

  @mock.patch("jax.process_index", return_value=0)
  def test_tensorboard_only_scalars_are_written_only_to_tensorboard(self, _):
    logger = MetricLogger.__new__(MetricLogger)
    logger.config = SimpleNamespace(log_period=1, tensorboard_dir="/tmp/tb")
    logger.writer = mock.Mock()
    metrics = {
        "scalar": {"learning/loss": 1.0},
        "scalars": {},
    }

    logger.write_metrics_to_tensorboard(metrics, step=1, is_training=True)

    logger.writer.add_scalar.assert_any_call("learning/loss", np.array(1.0), 1)
