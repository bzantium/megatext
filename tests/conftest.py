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

"""Pytest configuration helpers for decoupled test selection.

Automatically apply the `decoupled` marker (when DECOUPLE_GCLOUD=TRUE) to
tests that remain collected. Tests that are explicitly skipped because they
require external integrations or specific hardware (for example `tpu_only`)
are not marked.
"""

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest
import jax

from megatext.common.gcloud_stub import is_decoupled

# Configure JAX to use unsafe_rbg PRNG implementation to match main scripts.
if is_decoupled():
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")

try:
  _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover  pylint: disable=broad-exception-caught
  _HAS_TPU = False

try:
  _HAS_GPU = any(d.platform == "gpu" for d in jax.devices())
except Exception:  # pragma: no cover  pylint: disable=broad-exception-caught
  _HAS_GPU = False


GCP_MARKERS = {"external_serving", "external_training"}


def _has_tpu_backend_support() -> bool:
  """Whether JAX has TPU backend support installed (PJRT TPU plugin).

  This is intentionally *not* the same as having TPU hardware available.
  """
  try:
    if importlib.util.find_spec("jaxlib") is None:
      return False
  except Exception:  # pragma: no cover  pylint: disable=broad-exception-caught
    return False

  # Heuristic: TPU backend support is provided via the `libtpu` package.
  try:
    return importlib.util.find_spec("libtpu") is not None
  except Exception:  # pragma: no cover  pylint: disable=broad-exception-caught
    return False


def pytest_collection_modifyitems(config, items):
  """Customize pytest collection behavior.

  - Skip hardware-specific tests when hardware is missing.
  - Deselect tests marked as external_serving/training in decoupled mode.
  - Mark remaining tests with the `decoupled` marker when running decoupled.
  """
  decoupled = is_decoupled()
  remaining = []
  deselected = []

  skip_no_tpu = None
  skip_no_gpu = None
  skip_no_tpu_backend = None
  if not _HAS_TPU:
    skip_no_tpu = pytest.mark.skip(reason="Skipped: requires TPU hardware, none detected")

  if not _HAS_GPU:
    skip_no_gpu = pytest.mark.skip(reason="Skipped: requires GPU hardware, none detected")

  if not _has_tpu_backend_support():
    skip_no_tpu_backend = pytest.mark.skip(
        reason=(
            "Skipped: requires a TPU-enabled JAX install (TPU PJRT plugin). "
            "Install a TPU-enabled jax/jaxlib build to run this test."
        )
    )

  for item in items:
    # Iterate thru the markers of every test.
    cur_test_markers = {m.name for m in item.iter_markers()}

    # Hardware skip retains skip semantics.
    if skip_no_tpu and "tpu_only" in cur_test_markers:
      item.add_marker(skip_no_tpu)
      remaining.append(item)
      continue

    if skip_no_gpu and "gpu_only" in cur_test_markers:
      item.add_marker(skip_no_gpu)
      remaining.append(item)
      continue

    if skip_no_tpu_backend and "tpu_backend" in cur_test_markers:
      item.add_marker(skip_no_tpu_backend)
      remaining.append(item)
      continue

    if decoupled and (cur_test_markers & GCP_MARKERS):
      # Deselect tests marked as external_serving/training entirely.
      deselected.append(item)
      continue

    remaining.append(item)

  # Update items in-place to only keep remaining tests.
  items[:] = remaining
  if deselected:
    config.hook.pytest_deselected(items=deselected)

  # Add decoupled marker to all remaining tests when running decoupled.
  if decoupled:
    for item in remaining:
      item.add_marker(pytest.mark.decoupled)


def pytest_configure(config):
  for m in [
      "gpu_only: tests that require GPU hardware",
      "tpu_only: tests that require TPU hardware",
      "tpu_backend: tests that require a TPU-enabled JAX install (TPU PJRT plugin), but not TPU hardware",
      "external_serving: JetStream / serving / decode server components",
      "external_training: goodput integrations",
      "decoupled: marked on tests that are not skipped due to GCP deps, when DECOUPLE_GCLOUD=TRUE",
  ]:
    config.addinivalue_line("markers", m)


# ---------------------------------------------------------------------------
# Fixtures for mmap / arecord dataset tests
# ---------------------------------------------------------------------------

def _create_mmap_dataset(prefix: str, docs: list, dtype=np.uint16):
    """Helper to create a single mmap dataset (prefix.idx + prefix.bin)."""
    from megatext.data.indexed_dataset import _NUMPY_TO_DTYPE_CODE

    dtype_code = _NUMPY_TO_DTYPE_CODE[dtype]

    bin_path = Path(f"{prefix}.bin")
    idx_path = Path(f"{prefix}.idx")

    sequence_lengths = []
    sequence_pointers = []
    pointer = 0

    with open(bin_path, "wb") as f:
        for tokens in docs:
            data = tokens.astype(dtype).tobytes()
            sequence_pointers.append(pointer)
            sequence_lengths.append(len(tokens))
            f.write(data)
            pointer += len(data)

    # Write .idx file
    IDX_MAGIC = b"MMIDIDX\x00\x00"
    seq_count = len(docs)
    with open(idx_path, "wb") as f:
        f.write(IDX_MAGIC)
        f.write(struct.pack("<Q", 1))  # version
        f.write(struct.pack("<B", dtype_code))  # dtype
        f.write(struct.pack("<Q", seq_count))  # seq_count
        f.write(struct.pack("<Q", seq_count + 1))  # doc_count entries

        np.array(sequence_lengths, dtype=np.int32).tofile(f)
        np.array(sequence_pointers, dtype=np.int64).tofile(f)
        np.arange(seq_count + 1, dtype=np.int64).tofile(f)


@pytest.fixture
def sample_mmap_dataset(tmp_path):
    """Create a small mmap dataset for testing.

    Creates 10 documents with varying lengths (10-100 tokens each).
    Returns the path prefix (without .bin/.idx extension).
    """
    prefix = str(tmp_path / "test_data")
    num_docs = 10
    rng = np.random.RandomState(42)

    # Generate documents with random token IDs
    docs = []
    for _ in range(num_docs):
        doc_len = rng.randint(10, 101)
        tokens = rng.randint(0, 1000, size=doc_len).astype(np.uint16)
        docs.append(tokens)

    _create_mmap_dataset(prefix, docs)
    return prefix, docs


@pytest.fixture
def sample_mmap_directory(tmp_path):
    """Create a directory with 3 mmap datasets for testing MultiFileIndexedDataset.

    Returns (dir_path, concatenated_docs).
    """
    data_dir = tmp_path / "mmap_multi"
    data_dir.mkdir()

    rng = np.random.RandomState(123)
    all_docs = []

    for i in range(3):
        prefix = str(data_dir / f"split_{i:02d}")
        docs = []
        for _ in range(5):
            doc_len = rng.randint(10, 51)
            tokens = rng.randint(0, 1000, size=doc_len).astype(np.uint16)
            docs.append(tokens)
        _create_mmap_dataset(prefix, docs)
        all_docs.extend(docs)

    return str(data_dir), all_docs


@pytest.fixture
def sample_arecord_dataset(tmp_path):
    """Create a small arecord dataset for testing (prefix-based).

    Returns (prefix_path, list_of_doc_token_arrays).
    """
    pytest.importorskip("array_record")
    from array_record.python.array_record_module import ArrayRecordWriter
    from megatext.data.indexed_dataset import write_index

    prefix = str(tmp_path / "arecord_data")

    rng = np.random.RandomState(42)
    num_docs = 10
    docs = []
    doc_lengths = []

    writer = ArrayRecordWriter(f"{prefix}.arecord", "group_size:1")
    for _ in range(num_docs):
        doc_len = rng.randint(10, 101)
        tokens = rng.randint(0, 1000, size=doc_len).astype(np.uint16)
        docs.append(tokens)
        doc_lengths.append(doc_len)
        writer.write(tokens.tobytes())
    writer.close()

    write_index(f"{prefix}.idx", np.array(doc_lengths, dtype=np.int32))

    return prefix, docs


@pytest.fixture
def sample_arecord_directory(tmp_path):
    """Create a directory with 3 arecord prefix datasets for testing MultiFileIndexedDataset.

    Returns (dir_path, concatenated_docs).
    """
    pytest.importorskip("array_record")
    from array_record.python.array_record_module import ArrayRecordWriter
    from megatext.data.indexed_dataset import write_index

    data_dir = tmp_path / "arecord_multi"
    data_dir.mkdir()

    rng = np.random.RandomState(456)
    all_docs = []

    for i in range(3):
        prefix = str(data_dir / f"split_{i:02d}")
        docs = []
        doc_lengths = []

        writer = ArrayRecordWriter(f"{prefix}.arecord", "group_size:1")
        for _ in range(5):
            doc_len = rng.randint(10, 51)
            tokens = rng.randint(0, 1000, size=doc_len).astype(np.uint16)
            docs.append(tokens)
            doc_lengths.append(doc_len)
            writer.write(tokens.tobytes())
        writer.close()

        write_index(f"{prefix}.idx", np.array(doc_lengths, dtype=np.int32))
        all_docs.extend(docs)

    return str(data_dir), all_docs
