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

"""Global variable constants used throughout the codebase"""

import os.path

# This is the megatext package root (src/megatext)
# Since this file is at src/megatext/utils/constants.py, we need to go up 2 levels
MEGATEXT_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# This is the megatext repo root: with ".git" folder; "README.md"; "pyproject.toml"; &etc.
MEGATEXT_REPO_ROOT = os.environ.get(
    "MEGATEXT_REPO_ROOT",
    r
    if os.path.isdir(
        os.path.join(r := os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), ".git")
    )
    else MEGATEXT_PKG_DIR,
)

# This is the configs root: with "base.yaml"; "models/"; &etc.
MEGATEXT_CONFIGS_DIR = os.environ.get("MEGATEXT_CONFIGS_DIR", os.path.join(MEGATEXT_PKG_DIR, "configs"))

# This is the assets root: with "tokenizers/"; &etc.
MEGATEXT_ASSETS_ROOT = os.environ.get("MEGATEXT_ASSETS_ROOT", os.path.join(MEGATEXT_REPO_ROOT, "src", "megatext", "assets"))

# This is the test assets root: with "golden_logits"; &etc.
MEGATEXT_TEST_ASSETS_ROOT = os.environ.get("MEGATEXT_TEST_ASSETS_ROOT", os.path.join(MEGATEXT_REPO_ROOT, "tests", "assets"))

EPS = 1e-8  # Epsilon to calculate loss
DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = 2 * 1024**3  # Default checkpoint file size

__all__ = [
    "DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE",
    "EPS",
    "MEGATEXT_ASSETS_ROOT",
    "MEGATEXT_CONFIGS_DIR",
    "MEGATEXT_PKG_DIR",
    "MEGATEXT_REPO_ROOT",
    "MEGATEXT_TEST_ASSETS_ROOT",
]
