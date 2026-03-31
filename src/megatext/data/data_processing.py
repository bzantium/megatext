# Copyright 2023-2025 Google LLC
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

"""Grain pipeline with multi-host support via maxtext's MultiHostDataLoadIterator.

Combines pre-computed packing with maxtext's multi-host data loading.
Process 0 builds/caches indices, then all hosts load from cache.
"""

from __future__ import annotations

import math

import grain.python as grain
import jax
import numpy as np

from megatext.utils import logging as max_logging
from megatext.data.data_sources import (
    BlendedDataSource,
    DocumentDataSource,
    FixedArecordDataSource,
    SyntheticDataSource,
)


def create_data_iterator(config, mesh):
    """Create train and eval data iterators with multi-host support.

    Args:
        config: pyconfig.HyperParameters with all settings.
        mesh: JAX device mesh.

    Returns:
        (train_iterator, eval_iterator) - eval_iterator may be None
    """
    train_iter = _create_multihost_iterator(config, mesh, split_index=0)

    eval_iter = None
    if config.eval_interval > 0:
        eval_iter = _create_multihost_iterator(config, mesh, split_index=1)

    return train_iter, eval_iter


def _create_multihost_iterator(config, mesh, split_index: int):
    """Build a Grain MapDataset pipeline wrapped in MultiHostDataLoadIterator."""
    from megatext.data.multihost_dataloading import MultiHostDataLoadIterator

    if config.dataset_type == "fixed_arecord":
        if not config.add_extra_token:
            raise ValueError("fixed_arecord requires add_extra_token=True (records are seq_len+1 tokens)")
        return _create_fixed_arecord_iterator(config, mesh)

    if config.dataset_type == "synthetic":
        data_source = _build_synthetic_source(config)
    else:
        # Process 0 builds indices, others wait
        if jax.process_count() > 1:
            if jax.process_index() == 0:
                max_logging.log("Process 0: building/loading data indices...")
                _build_data_source(config, split_index)
            # Barrier: all processes wait for process 0 to finish
            jax.experimental.multihost_utils.sync_global_devices("data_index_built")

        # All processes now create their data source (loading from cache)
        data_source = _build_data_source(config, split_index)

    # Compute per-process batch size
    global_batch_size = int(config.per_device_batch_size * jax.device_count())
    num_processes = jax.process_count()
    if global_batch_size % num_processes != 0:
        raise ValueError(
            f"Global batch size ({global_batch_size}) must be divisible by "
            f"process count ({num_processes})"
        )
    per_process_batch = global_batch_size // num_processes

    ga_steps = int(getattr(config, "gradient_accumulation_steps", 1))
    effective_batch = global_batch_size * ga_steps
    total_samples = int(config.steps) * effective_batch
    per_device_bs = int(config.per_device_batch_size)
    max_logging.log(
        f"Data ({['train', 'eval'][split_index]}): {total_samples} samples "
        f"global_batch={effective_batch}, per_device_batch={per_device_bs}"
        + (f", ga_steps={ga_steps}" if ga_steps > 1 else "")
    )

    # Build grain MapDataset pipeline with mp_prefetch for async data loading
    dataset = grain.MapDataset.source(data_source)

    # Per-process sharding via index slicing
    dataset = dataset[jax.process_index()::num_processes]

    # Infinite epochs
    dataset = dataset.repeat(None)

    # Apply transforms
    dataset = dataset.map(FormatForMaxText(
        seq_len=config.max_target_length,
        add_extra_token=config.add_extra_token,
    ))
    dataset = dataset.batch(per_process_batch, drop_remainder=True)

    # Convert to IterDataset, then apply async prefetch in background workers
    # (safe for TPU — workers only run data pipeline, don't touch JAX/TPU)
    dataset = dataset.to_iter_dataset()
    worker_count = config.grain_worker_count
    if worker_count > 0:
        dataset = dataset.mp_prefetch(grain.MultiprocessingOptions(
            num_workers=worker_count,
            per_worker_buffer_size=config.grain_per_worker_buffer_size,
        ))

    return MultiHostDataLoadIterator(dataset, mesh)


def _create_fixed_arecord_iterator(config, mesh):
    """Build a Grain IterDataset pipeline for fixed-length arecord data.

    Uses grain.IterDataset.mix for multi-source blending — each source is
    independently shuffled/sharded/repeated, then mixed proportionally.
    No pre-built blend indices needed.
    """
    from megatext.data.multihost_dataloading import MultiHostDataLoadIterator

    seq_len = config.max_target_length
    entries = _parse_data_path(config.dataset_path)
    process_index = jax.process_index()
    num_processes = jax.process_count()

    global_batch_size = int(config.per_device_batch_size * jax.device_count())
    if global_batch_size % num_processes != 0:
        raise ValueError(
            f"Global batch size ({global_batch_size}) must be divisible by "
            f"process count ({num_processes})"
        )
    per_process_batch = global_batch_size // num_processes

    ga_steps = int(getattr(config, "gradient_accumulation_steps", 1))
    effective_batch = global_batch_size * ga_steps
    total_samples = int(config.steps) * effective_batch
    max_logging.log(
        f"Data (train): {total_samples} samples "
        f"global_batch={effective_batch}, per_device_batch={int(config.per_device_batch_size)}"
        + (f", ga_steps={ga_steps}" if ga_steps > 1 else "")
    )

    fmt = FormatForMaxText(seq_len=seq_len, add_extra_token=True)

    def _make_iter_dataset(path: str) -> grain.IterDataset:
        source = FixedArecordDataSource(data_path=path, seq_len=seq_len)
        dataset = grain.MapDataset.source(source)
        dataset = dataset.shuffle(seed=config.data_shuffle_seed)
        dataset = dataset.repeat(None)
        dataset = dataset[process_index::num_processes]
        dataset = dataset.map(fmt)
        dataset = dataset.batch(per_process_batch, drop_remainder=True)
        return dataset.to_iter_dataset()

    if len(entries) == 1:
        _, path = entries[0]
        dataset = _make_iter_dataset(path)
    else:
        datasets = []
        weights = []
        for weight, path in entries:
            datasets.append(_make_iter_dataset(path))
            weights.append(weight)
        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]
        dataset = grain.IterDataset.mix(datasets, norm_weights)

    worker_count = config.grain_worker_count
    if worker_count > 0:
        dataset = dataset.mp_prefetch(grain.MultiprocessingOptions(
            num_workers=worker_count,
            per_worker_buffer_size=config.grain_per_worker_buffer_size,
        ))

    return MultiHostDataLoadIterator(dataset, mesh)


def _build_synthetic_source(config) -> SyntheticDataSource:
    """Build a synthetic data source for debugging."""
    seq_len = config.max_target_length
    num_samples = config.steps * int(
        config.per_device_batch_size * jax.device_count()
    )
    # Use vocab_size from maxtext config
    vocab_size = int(config.vocab_size)
    max_logging.log(f"Synthetic data: {num_samples} samples, seq_len={seq_len}, vocab_size={vocab_size}")
    return SyntheticDataSource(seq_len=seq_len, num_samples=num_samples, vocab_size=vocab_size, seed=config.data_shuffle_seed)


def _build_data_source(config, split_index: int):
    """Build the (possibly blended) Grain data source."""
    split = _parse_split(config.data_split)
    entries = _parse_data_path(config.dataset_path)

    total_num_samples = config.steps * int(config.per_device_batch_size * jax.device_count())
    total_weight = sum(w for w, _ in entries)

    data_type = config.dataset_type
    sources = []
    weights = []
    for weight, path in entries:
        # Per-source num_samples (Megatron: weight * total * 1.005 margin)
        norm_weight = weight / total_weight
        num_samples = int(math.ceil(total_num_samples * norm_weight * 1.005))

        src = DocumentDataSource(
            data_path=path,
            data_type=data_type,
            seq_len=config.max_target_length,
            seed=config.data_shuffle_seed,
            num_samples=num_samples,
            split=split,
            split_index=split_index,
            cache_dir=config.data_cache_dir or None,
            add_extra_token=config.add_extra_token,
            packing_type=config.grain_packing_type,
            max_chunks_per_sample=config.max_chunks_per_sample,
        )
        sources.append(src)
        weights.append(weight)

    if len(sources) == 1:
        return sources[0]

    total_samples = sum(len(s) for s in sources)
    return BlendedDataSource(sources, weights, size=total_samples)


def _parse_data_path(data_path: str) -> list[tuple[float, str]]:
    """Parse 'weight1 path1 weight2 path2 ...' or plain 'path' format."""
    parts = data_path.strip().split()

    if len(parts) == 1:
        return [(1.0, parts[0])]

    entries = []
    i = 0
    while i < len(parts):
        try:
            weight = float(parts[i])
            if i + 1 >= len(parts):
                raise ValueError(f"Weight {weight} at position {i} has no path")
            path = parts[i + 1]
            entries.append((weight, path))
            i += 2
        except ValueError:
            if not entries:
                return [(1.0, data_path.strip())]
            raise

    return entries


def _parse_split(split_str: str) -> tuple[float, float, float]:
    """Parse 'train,val,test' proportions string."""
    parts = [float(x) for x in split_str.split(",")]
    if len(parts) != 3:
        raise ValueError(f"data_split must have 3 values, got: {split_str}")
    return (parts[0], parts[1], parts[2])


class FormatForMaxText(grain.MapTransform):
    """Convert tokens dict to maxtext's expected format.

    Input: {"tokens": [seq_len+1], optional "segment_ids", "loss_mask"}
    Output: {"inputs": [seq_len], "targets": [seq_len],
             "inputs_position": [seq_len], "targets_position": [seq_len],
             "inputs_segmentation": [seq_len], "targets_segmentation": [seq_len]}
    """

    def __init__(self, seq_len: int, add_extra_token: bool = True):
        self._seq_len = seq_len
        self._add_extra_token = add_extra_token

    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        tokens = element["tokens"]
        seq_len = self._seq_len

        if self._add_extra_token:
            inputs = tokens[:seq_len]
            targets = tokens[1 : seq_len + 1]
        else:
            inputs = tokens[:seq_len]
            targets = np.roll(tokens[:seq_len], -1)

        # Build segmentation and position arrays
        if "segment_ids" in element:
            segment_ids = element["segment_ids"]
            if self._add_extra_token:
                segment_ids = segment_ids[:seq_len]
            inputs_segmentation = segment_ids
            targets_segmentation = segment_ids

            # Positions reset per segment (vectorized)
            if segment_ids.max() > 0:
                # At segment boundaries (where segment_ids changes), reset position to 0
                boundary = np.empty(seq_len, dtype=np.bool_)
                boundary[0] = True
                boundary[1:] = segment_ids[1:] != segment_ids[:-1]
                # cumsum of ones gives position; subtract cummax of boundary positions
                cumpos = np.arange(seq_len, dtype=np.int32)
                # For each position, subtract the index of its segment start
                seg_starts = np.maximum.accumulate(np.where(boundary, cumpos, 0))
                positions = (cumpos - seg_starts).astype(np.int32)
            else:
                positions = np.arange(seq_len, dtype=np.int32)
        else:
            # No packing: simple sequential positions, all-ones segmentation
            positions = np.arange(seq_len, dtype=np.int32)
            inputs_segmentation = np.ones(seq_len, dtype=np.int32)
            targets_segmentation = np.ones(seq_len, dtype=np.int32)

        # Apply loss_mask to targets_segmentation if present
        if "loss_mask" in element:
            loss_mask = element["loss_mask"][:seq_len]
            targets_segmentation = (targets_segmentation * loss_mask).astype(np.int32)

        return {
            "inputs": inputs.astype(np.int32),
            "targets": targets.astype(np.int32),
            "inputs_position": positions.astype(np.int32),
            "targets_position": positions.astype(np.int32),
            "inputs_segmentation": inputs_segmentation.astype(np.int32),
            "targets_segmentation": targets_segmentation.astype(np.int32),
        }
