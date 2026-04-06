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

"""Tests for Megatext checkpoint conversion helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from transformers import Qwen3Config

from megatext.conversion import convert
from megatext.conversion import io_megatext


def test_load_megatext_checkpoint_uses_requested_step():
  fake_manager = mock.Mock()

  with (
      mock.patch.object(io_megatext.ocp, "CheckpointManager", return_value=fake_manager),
      mock.patch.object(io_megatext, "load_megatext_checkpoint_params", return_value={}) as mock_load_params,
  ):
    io_megatext.load_megatext_checkpoint("/tmp/checkpoints", step=123)

  fake_manager.latest_step.assert_not_called()
  mock_load_params.assert_called_once_with("/tmp/checkpoints/123/items")


def test_megatext_to_hf_supports_specific_step_and_gcs_output():
  hf_config = SimpleNamespace(
      model_type="qwen3",
      text_config=SimpleNamespace(tie_word_embeddings=False),
      to_dict=lambda: {},
  )
  fake_arch = SimpleNamespace(model_type="qwen3", composite_split="deinterleave")

  with (
      mock.patch.object(convert, "_load_hf_config", return_value=hf_config),
      mock.patch.object(convert, "resolve", return_value=fake_arch),
      mock.patch.object(convert, "build_mapping", return_value={}),
      mock.patch.object(convert, "build_transforms", return_value={}),
      mock.patch.object(convert, "compute_megatext_shapes", return_value={}),
      mock.patch.object(convert, "load_megatext_checkpoint", return_value={}) as mock_load_checkpoint,
      mock.patch.object(convert, "_convert_mt_to_hf"),
      mock.patch.object(convert, "_validate_hf_state_dict_for_save", return_value={}) as mock_validate,
      mock.patch.object(convert, "_save_hf_checkpoint_direct") as mock_save,
      mock.patch.object(convert, "_copy_tokenizer") as mock_copy_tokenizer,
      mock.patch.object(convert.gcs_utils, "upload_dump") as mock_upload_dump,
  ):
    result = convert.megatext_to_hf(
        megatext_model_path="gs://bucket/run/checkpoints",
        checkpoint_step=77,
        output_dir="gs://bucket/run/checkpoints/77/hf",
        hf_config_path="Qwen/Qwen3-8B",
        scan_layers=True,
        hf_token="secret",
  )

  assert result == "gs://bucket/run/checkpoints/77/hf"
  mock_load_checkpoint.assert_called_once_with("gs://bucket/run/checkpoints", step=77)
  mock_validate.assert_called_once_with({}, hf_config)
  mock_save.assert_called_once()
  temp_output_dir = mock_save.call_args.args[0]
  mock_copy_tokenizer.assert_called_once_with("Qwen/Qwen3-8B", temp_output_dir, "secret")
  mock_upload_dump.assert_called_once_with(temp_output_dir, "gs://bucket/run/checkpoints/77/hf", delete_local_after=False)


def test_megatext_to_hf_uses_qwen3_swa_conversion_family_for_qwen3_sliding_config():
  hf_config = SimpleNamespace(
      model_type="qwen3",
      tie_word_embeddings=False,
      hidden_size=128,
      intermediate_size=256,
      num_attention_heads=4,
      num_key_value_heads=4,
      head_dim=32,
      vocab_size=1024,
      num_hidden_layers=8,
      use_sliding_window=True,
      sliding_window=4,
      layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"] * 2,
      to_dict=lambda: {},
  )
  fake_arch = SimpleNamespace(model_type="qwen3_swa", composite_split="deinterleave")

  with (
      mock.patch.object(convert, "_load_hf_config", return_value=hf_config),
      mock.patch.object(convert, "resolve", return_value=fake_arch) as mock_resolve,
      mock.patch.object(convert, "build_mapping", return_value={}) as mock_build_mapping,
      mock.patch.object(convert, "build_transforms", return_value={}),
      mock.patch.object(convert, "compute_megatext_shapes", return_value={}),
      mock.patch.object(convert, "load_megatext_checkpoint", return_value={}),
      mock.patch.object(convert, "_convert_mt_to_hf"),
      mock.patch.object(convert, "_validate_hf_state_dict_for_save", return_value={}),
      mock.patch.object(convert, "_save_hf_checkpoint_direct"),
      mock.patch.object(convert, "_copy_tokenizer"),
  ):
    convert.megatext_to_hf(
        megatext_model_path="/tmp/checkpoints",
        checkpoint_step=77,
        output_dir="/tmp/hf_out",
        hf_config_path="/tmp/qwen3_swa_like_hf",
        scan_layers=True,
    )

  mock_resolve.assert_called_once_with("qwen3_swa")
  assert mock_build_mapping.call_args.args[0] is fake_arch


def test_megatext_to_hf_loads_megatext_checkpoint_once():
  hf_config = SimpleNamespace(
      model_type="qwen3",
      text_config=SimpleNamespace(tie_word_embeddings=False),
      to_dict=lambda: {},
  )
  fake_arch = SimpleNamespace(model_type="qwen3", composite_split="deinterleave")
  mapping = {"params-token_embedder-embedding": "model.embed_tokens.weight"}

  with (
      mock.patch.object(convert, "_load_hf_config", return_value=hf_config),
      mock.patch.object(convert, "resolve", return_value=fake_arch),
      mock.patch.object(convert, "build_mapping", return_value=mapping),
      mock.patch.object(convert, "build_transforms", return_value={}),
      mock.patch.object(convert, "compute_megatext_shapes", return_value={}),
      mock.patch.object(
          convert,
          "load_megatext_checkpoint",
          return_value={"params-token_embedder-embedding": np.ones((2, 3), dtype=np.float32)},
      ) as mock_load_checkpoint,
      mock.patch.object(convert, "_convert_mt_to_hf"),
      mock.patch.object(convert, "_validate_hf_state_dict_for_save", return_value={}),
      mock.patch.object(convert, "_save_hf_checkpoint_direct"),
      mock.patch.object(convert, "_copy_tokenizer"),
  ):
    convert.megatext_to_hf(
        megatext_model_path="/tmp/checkpoints",
        checkpoint_step=77,
        output_dir="/tmp/hf_out",
        hf_config_path="/tmp/qwen3_hf_template",
        scan_layers=True,
    )

  mock_load_checkpoint.assert_called_once_with("/tmp/checkpoints", step=77)


def test_resolve_megatext_checkpoint_input_accepts_items_path():
  root, step = convert._resolve_megatext_checkpoint_input("/tmp/checkpoints/123/items")
  assert root == "/tmp/checkpoints"
  assert step == 123


def test_resolve_megatext_checkpoint_input_accepts_step_dir():
  root, step = convert._resolve_megatext_checkpoint_input("/tmp/checkpoints/456")
  assert root == "/tmp/checkpoints"
  assert step == 456


def test_load_megatext_checkpoint_params_restores_params_prefix():
  array = np.arange(6, dtype=np.float32).reshape(2, 3)
  fake_leaf = SimpleNamespace(shape=array.shape, dtype=array.dtype)
  fake_metadata = SimpleNamespace(item_metadata=SimpleNamespace(tree={"params": {"params": {"foo": fake_leaf}}}))
  fake_ckptr = mock.Mock()
  fake_ckptr.metadata.return_value = fake_metadata
  fake_ckptr.restore.return_value = {"params": {"params": {"foo": array}}}

  with (
      mock.patch.object(io_megatext.ocp, "Checkpointer", return_value=fake_ckptr),
      mock.patch.object(io_megatext.ocp.checkpoint_utils, "construct_restore_args", return_value={"params": "restore_args"}),
  ):
    restored = io_megatext.load_megatext_checkpoint_params("/tmp/checkpoints/123/items")

  assert restored == {"params-foo": array}


def test_cpu_abstract_from_metadata_prefers_cpu_device():
  fake_leaf = SimpleNamespace(shape=(2, 3), dtype=np.float32)
  fake_cpu_device = object()
  fake_default_device = object()

  def fake_devices(kind=None):
    if kind == "cpu":
      return [fake_cpu_device]
    return [fake_default_device]

  with (
      mock.patch.object(io_megatext.jax, "devices", side_effect=fake_devices),
      mock.patch.object(
          io_megatext.jax.sharding,
          "SingleDeviceSharding",
          side_effect=lambda device: ("sharding", device),
      ),
      mock.patch.object(
          io_megatext.jax,
          "ShapeDtypeStruct",
          side_effect=lambda *, shape, dtype, sharding: SimpleNamespace(
              shape=shape,
              dtype=dtype,
              sharding=sharding,
          ),
      ),
  ):
    abstract = io_megatext._cpu_abstract_from_metadata(fake_leaf)

  assert abstract.shape == (2, 3)
  assert abstract.dtype == np.float32
  assert abstract.sharding == ("sharding", fake_cpu_device)


def test_validate_hf_state_dict_for_save_accepts_matching_meta_model():
  hf_config = Qwen3Config(
      hidden_size=16,
      intermediate_size=32,
      num_hidden_layers=2,
      num_attention_heads=2,
      num_key_value_heads=2,
      head_dim=8,
      vocab_size=32,
  )
  model = convert._init_hf_model_for_validation(hf_config)
  state_dict = {
      key: torch.empty_like(value, device="cpu")
      for key, value in model.state_dict().items()
  }

  ordered = convert._validate_hf_state_dict_for_save(state_dict, hf_config)

  assert list(ordered) == list(model.state_dict())


def test_validate_hf_state_dict_for_save_rejects_shape_mismatch():
  hf_config = Qwen3Config(
      hidden_size=16,
      intermediate_size=32,
      num_hidden_layers=2,
      num_attention_heads=2,
      num_key_value_heads=2,
      head_dim=8,
      vocab_size=32,
  )
  model = convert._init_hf_model_for_validation(hf_config)
  state_dict = {
      key: torch.empty_like(value, device="cpu")
      for key, value in model.state_dict().items()
  }
  first_key = next(iter(state_dict))
  state_dict[first_key] = torch.empty((1,), dtype=state_dict[first_key].dtype)

  with pytest.raises(ValueError, match="runtime-model validation"):
    convert._validate_hf_state_dict_for_save(state_dict, hf_config)


def test_save_hf_checkpoint_direct_writes_config_and_weights(tmp_path):
  hf_config = Qwen3Config(
      hidden_size=16,
      intermediate_size=32,
      num_hidden_layers=2,
      num_attention_heads=2,
      num_key_value_heads=2,
      head_dim=8,
      vocab_size=32,
  )
  model = convert._init_hf_model_for_validation(hf_config)
  state_dict = {
      key: torch.empty_like(value, device="cpu")
      for key, value in model.state_dict().items()
  }
  ordered = convert._validate_hf_state_dict_for_save(state_dict, hf_config)

  convert._save_hf_checkpoint_direct(str(tmp_path), ordered, hf_config)

  assert (tmp_path / "config.json").exists()
  assert (tmp_path / "generation_config.json").exists()
  assert any(path.name.endswith(".safetensors") for path in tmp_path.iterdir())
