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
"""
Test suite for validating Megatext YAML configurations against Pydantic models.

This test suite uses explicit, hardcoded lists of configuration files grouped
by architecture family to test them directly against the Pydantic
`MegatextConfig` model. It avoids programmatic file discovery and the complex
`pyconfig.initialize` function to provide fast, targeted feedback on validation
errors like "Extra inputs are not permitted."
"""

import os
import functools
from copy import deepcopy

import pytest

import yaml

from pydantic import ValidationError
from yaml import YAMLError

from megatext.configs import types as pydantic_types
from megatext.utils.constants import MEGATEXT_CONFIGS_DIR

CONFIGS_DIR = MEGATEXT_CONFIGS_DIR


@functools.lru_cache(maxsize=None)
def load_and_merge_yamls(yaml_path: str) -> dict:
  """
  Recursively loads a YAML file and merges it with its base configurations.

  A cache is used to avoid re-reading and re-parsing the same base files
  multiple times (e.g., base.yaml).

  Args:
      yaml_path: The absolute path to the YAML file to load.

  Returns:
      A single merged dictionary representing the fully resolved configuration.
  """
  with open(yaml_path, "rt", encoding="utf-8") as f:
    data = yaml.safe_load(f)

  if data and "base_config" in data:
    base_path_str = data["base_config"]
    # base_config paths are relative to the current YAML file's directory.
    base_path = os.path.abspath(os.path.join(os.path.dirname(yaml_path), base_path_str))
    if not os.path.exists(base_path):
      # Fallback to the main configs directory
      base_path = os.path.join(CONFIGS_DIR, base_path_str)

    base_data = deepcopy(load_and_merge_yamls(base_path))
    # The child's values overwrite the base's values.
    base_data.update(data)
    return base_data

  return data if data is not None else {}


def run_config_validation(config_file_path: str):
  """
  Core validation logic: loads, merges, and validates a single config file.
  """
  print(f"\nTesting configuration file: {config_file_path}")
  try:
    # Step 1: Load the YAML file and all its parents.
    config_dict = load_and_merge_yamls(config_file_path)

    # Pre-process dictionary to align with Pydantic model before validation.
    if "base_config" in config_dict:
      del config_dict["base_config"]
    if "num_epochs" in config_dict:
      config_dict["num_epoch"] = config_dict.pop("num_epochs")
    for key, value in config_dict.items():
      if isinstance(value, str) and value.lower() == "none":
        config_dict[key] = None

    # Step 2: Attempt to instantiate the Pydantic model.
    # This is where validation happens. If there are extra fields,
    # missing fields, or type mismatches, a ValidationError is raised.
    pydantic_instance = pydantic_types.MegatextConfig(**config_dict)

    # Step 3: Test the "emit" part by dumping the model back to a dict.
    dumped_config = pydantic_instance.model_dump()
    assert isinstance(dumped_config, dict), "model_dump() did not return a dictionary."

  except ValidationError as e:
    pytest.fail(f"Pydantic validation FAILED for {config_file_path}:\n{e}", pytrace=False)
  except (TypeError, IOError, YAMLError) as e:
    pytest.fail(f"An unexpected error occurred for {config_file_path}:\n{type(e).__name__}: {e}", pytrace=True)


# ==============================================================================
# Begin Test Functions
# ==============================================================================

# --- Test Group 1: Base Config ---

BASE_CONFIGS = [
    os.path.join(CONFIGS_DIR, "base.yaml"),
]


@pytest.mark.parametrize("config_file", BASE_CONFIGS)
def test_base_configs(config_file):
  run_config_validation(config_file)


# --- Test Group 2: Architecture Template Configs (models/) ---

MODEL_CONFIGS = [
    os.path.join(CONFIGS_DIR, "models", "deepseek.yaml"),
    os.path.join(CONFIGS_DIR, "models", "gpt-oss.yaml"),
    os.path.join(CONFIGS_DIR, "models", "llama3.yaml"),
    os.path.join(CONFIGS_DIR, "models", "qwen3.yaml"),
    os.path.join(CONFIGS_DIR, "models", "qwen3-moe.yaml"),
    os.path.join(CONFIGS_DIR, "models", "qwen3-swa.yaml"),
    os.path.join(CONFIGS_DIR, "models", "qwen3-next.yaml"),
    os.path.join(CONFIGS_DIR, "models", "qwen3-next-moe.yaml"),
]
MODEL_CONFIGS.extend([
    path for path in [
        os.path.join(CONFIGS_DIR, "models", "gemma4-moe.yaml"),
        os.path.join(CONFIGS_DIR, "models", "gemma4.yaml"),
    ] if os.path.exists(path)
])


@pytest.mark.parametrize("config_file", MODEL_CONFIGS)
def test_model_configs(config_file):
  run_config_validation(config_file)
