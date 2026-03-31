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

"""Data pipeline entry point."""

from megatext.configs import pyconfig
from megatext.data.synthetic_data_processing import SyntheticDataIterator
from megatext.utils import max_logging


def create_data_iterator(config: pyconfig.HyperParameters, mesh):
  """Create train and eval data iterators given configs and mesh."""

  if config.dataset_type == "synthetic":
    return SyntheticDataIterator(config, mesh), None

  if config.dataset_type in ("mmap", "arecord", "fixed_arecord"):
    from megatext.data import data_processing
    return data_processing.create_data_iterator(config, mesh)

  max_logging.log(
      f"WARNING: '{config.dataset_type}' is not a supported dataset type. "
      "Using synthetic data. Supported types: 'synthetic', 'mmap', 'arecord', 'fixed_arecord'."
  )
  return SyntheticDataIterator(config, mesh), None
