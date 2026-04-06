# Copyright 2023–2026 Google LLC
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

"""Tests for megatext.utils.train_utils."""

from types import SimpleNamespace
from unittest import mock

import jax.numpy as jnp

from megatext.utils import train_utils


def _make_config():
  return SimpleNamespace(
      init_weights_seed=0,
      enable_multi_tier_checkpointing=False,
      enable_emergency_checkpoint=False,
      checkpoint_storage_use_ocdbt=False,
      checkpoint_storage_use_zarr3=False,
      enable_single_controller=False,
      colocated_python_checkpointing=False,
      enable_checkpointing=False,
      checkpoint_dir="",
      async_checkpointing=False,
      checkpoint_period=100,
      dataset_type="synthetic",
      enable_continuous_checkpointing=False,
      max_num_checkpoints_to_keep=1,
      checkpoint_storage_concurrent_gb=32,
      enable_single_replica_ckpt_restoring=False,
  )


def test_create_training_tools_shifts_optimizer_schedule_by_one_step():
  config = _make_config()
  model = object()
  mesh = object()
  base_schedule = mock.Mock(side_effect=lambda step: jnp.asarray(step * 10.0, dtype=jnp.float32))

  with (
      mock.patch.object(train_utils, "create_learning_rate_schedule", return_value=base_schedule),
      mock.patch("megatext.common.checkpointing.setup_checkpoint_logger", return_value=mock.sentinel.logger),
      mock.patch("megatext.common.checkpointing.create_orbax_checkpoint_manager", return_value=mock.sentinel.ckpt_mgr),
      mock.patch("megatext.optimizers.optimizers.get_optimizer", return_value=mock.sentinel.tx) as get_optimizer,
  ):
    _, checkpoint_manager, learning_rate_schedule, tx = train_utils.create_training_tools(config, model, mesh)

  assert checkpoint_manager is mock.sentinel.ckpt_mgr
  assert learning_rate_schedule is base_schedule
  assert tx is mock.sentinel.tx

  optimizer_schedule = get_optimizer.call_args.args[1]
  assert float(optimizer_schedule(0)) == 10.0
  assert float(optimizer_schedule(1)) == 20.0
  assert [call.args[0] for call in base_schedule.call_args_list[-2:]] == [1, 2]
