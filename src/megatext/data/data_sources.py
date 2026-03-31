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

"""Grain-compatible data sources with cached sample indexing.

Works with either MMapIndexedDataset or ArrayRecordDocDataset backend.
Implements document/sample indexing with cached sample boundaries in pure NumPy.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import grain.python as grain
import numpy as np

from megatext.utils import logging as max_logging
from megatext.data.indexed_dataset import make_arecord_dataset, make_mmap_dataset
from megatext.data.bin_packing import build_packed_sample_index


def _get_num_epochs(
    num_samples: int, tokens_per_epoch: int, seq_len: int, add_extra_token: bool
) -> int:
    """Megatron pattern: keep adding epochs until tokens >= num_samples * seq_len."""
    num_epochs = 1
    num_tokens = tokens_per_epoch
    num_tokens_requested = num_samples * seq_len + int(add_extra_token)
    while num_tokens < num_tokens_requested:
        num_epochs += 1
        num_tokens += tokens_per_epoch
    return num_epochs


def _build_shuffle_index(
    num_samples: int, total_size: int, rng: np.random.RandomState
) -> np.ndarray:
    """Megatron pattern: shuffle [0, num_samples) and [num_samples, total_size) independently."""
    dtype = np.uint32 if total_size < np.iinfo(np.uint32).max - 1 else np.int64
    first = np.arange(num_samples, dtype=dtype)
    rng.shuffle(first)
    if num_samples == total_size:
        return first
    last = np.arange(num_samples, total_size, dtype=dtype)
    rng.shuffle(last)
    return np.concatenate([first, last])


class DocumentDataSource(grain.RandomAccessDataSource):
    """Grain data source backed by document-level data (mmap or arecord).

    Builds document_index and sample_index per sequence length,
    cached as .npy files for instant reload on subsequent runs.
    """

    def __init__(
        self,
        data_path: str,
        data_type: str,
        seq_len: int,
        seed: int,
        num_samples: int,
        split: tuple[float, float, float],
        split_index: int,
        cache_dir: str | None = None,
        add_extra_token: bool = True,
        packing_type: str = "greedy",
        max_chunks_per_sample: int = 0,
    ) -> None:
        super().__init__()
        self._data_path = data_path
        self._data_type = data_type
        self._seq_len = seq_len
        self._seed = seed
        self._num_samples = num_samples
        self._split = split
        self._split_index = split_index
        self._add_extra_token = add_extra_token
        self._packing_type = packing_type

        # Open backend
        if data_type == "mmap":
            self._backend = make_mmap_dataset(data_path)
        elif data_type == "arecord":
            self._backend = make_arecord_dataset(data_path)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        all_doc_lengths = self._backend.doc_lengths
        total_docs = len(all_doc_lengths)

        # Compute split boundaries
        split_sum = sum(split)
        fracs = [s / split_sum for s in split]
        boundaries = [0]
        for frac in fracs:
            boundaries.append(boundaries[-1] + int(round(frac * total_docs)))
        boundaries[-1] = total_docs  # ensure last split covers remainder

        doc_start = boundaries[split_index]
        doc_end = boundaries[split_index + 1]
        self._doc_ids = np.arange(doc_start, doc_end, dtype=np.int32)
        self._split_doc_lengths = all_doc_lengths[doc_start:doc_end]

        if len(self._doc_ids) == 0:
            self._document_index = None
            self._sample_index = np.zeros((1, 2), dtype=np.int64)
            self._shuffle_index = None
            self._chunk_index = None
            self._packed_sample_index = None
            return

        # Auto-compute num_epochs from num_samples
        tokens_per_epoch = int(np.sum(self._split_doc_lengths))
        num_epochs = _get_num_epochs(num_samples, tokens_per_epoch, seq_len, add_extra_token)

        # Build or load cached indices
        if cache_dir:
            cache_path = Path(cache_dir)
        else:
            cache_path = Path(data_path)

        if packing_type == "greedy":
            self._document_index, self._sample_index, self._shuffle_index = (
                self._load_or_build_indices(
                    data_path=data_path,
                    seq_len=seq_len,
                    seed=seed,
                    num_epochs=num_epochs,
                    num_samples=num_samples,
                    split=(doc_start, doc_end),
                    cache_dir=cache_path,
                    add_extra_token=add_extra_token,
                )
            )
            self._chunk_index = None
            self._packed_sample_index = None
        else:
            self._chunk_index, self._packed_sample_index, self._shuffle_index = (
                self._load_or_build_packed_indices(
                    data_path=data_path,
                    seq_len=seq_len,
                    seed=seed,
                    num_epochs=num_epochs,
                    num_samples=num_samples,
                    split=(doc_start, doc_end),
                    cache_dir=cache_path,
                    add_extra_token=add_extra_token,
                    packing_type=packing_type,
                    max_chunks_per_sample=max_chunks_per_sample,
                )
            )
            self._document_index = None
            self._sample_index = None

    def __repr__(self) -> str:
        return (
            f"DocumentDataSource(data_path={self._data_path!r}, "
            f"data_type={self._data_type!r}, seq_len={self._seq_len}, "
            f"seed={self._seed}, num_samples={self._num_samples}, "
            f"split={self._split}, split_index={self._split_index})"
        )

    def __len__(self) -> int:
        if self._packing_type == "greedy":
            return max(0, len(self._sample_index) - 1)
        return max(0, len(self._packed_sample_index) - 1)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        if self._shuffle_index is not None:
            idx = int(self._shuffle_index[idx])
        if self._packing_type == "greedy":
            return self._getitem_greedy(idx)
        return self._getitem_packed(idx)

    def _getitem_greedy(self, idx: int) -> dict[str, np.ndarray]:
        """Return {"tokens": np.ndarray} of length seq_len + add_extra_token."""
        target_len = self._seq_len + self._add_extra_token

        doc_pos_start, offset_start = self._sample_index[idx]
        doc_pos_end, offset_end = self._sample_index[idx + 1]

        tokens_parts = []
        remaining = target_len

        pos = int(doc_pos_start)
        offset = int(offset_start)

        while remaining > 0 and pos < len(self._document_index):
            doc_id = int(self._document_index[pos])
            real_doc_id = int(self._doc_ids[doc_id]) if doc_id < len(self._doc_ids) else doc_id
            doc_len = int(self._split_doc_lengths[doc_id]) if doc_id < len(self._split_doc_lengths) else 0
            available = doc_len - offset

            if available <= 0:
                pos += 1
                offset = 0
                continue

            take = min(available, remaining)
            chunk = self._backend.get(real_doc_id, offset=offset, length=take)
            tokens_parts.append(chunk)
            remaining -= take

            if take >= available:
                pos += 1
                offset = 0
            else:
                offset += take

        if tokens_parts:
            tokens = np.concatenate(tokens_parts)
        else:
            tokens = np.zeros(0, dtype=np.int32)

        # Pad if needed (shouldn't happen if index is built correctly)
        if len(tokens) < target_len:
            tokens = np.pad(tokens, (0, target_len - len(tokens)), constant_values=0)

        return {"tokens": tokens[:target_len]}

    def _getitem_packed(self, idx: int) -> dict[str, np.ndarray]:
        """Return packed sample with tokens, segment_ids, and loss_mask."""
        target_len = self._seq_len + self._add_extra_token
        tokens = np.zeros(target_len, dtype=np.int32)
        segment_ids = np.zeros(target_len, dtype=np.int32)

        chunk_start = int(self._packed_sample_index[idx])
        chunk_end = int(self._packed_sample_index[idx + 1])

        pos = 0
        for ci in range(chunk_start, chunk_end):
            doc_id_local = int(self._chunk_index[ci, 0])
            offset = int(self._chunk_index[ci, 1])
            length = int(self._chunk_index[ci, 2])

            real_doc_id = int(self._doc_ids[doc_id_local])
            chunk_tokens = self._backend.get(real_doc_id, offset=offset, length=length)

            end_pos = min(pos + length, target_len)
            actual_len = end_pos - pos
            tokens[pos:end_pos] = chunk_tokens[:actual_len]
            segment_ids[pos:end_pos] = ci - chunk_start + 1  # 1-indexed segment IDs
            pos = end_pos

        # loss_mask: 1.0 where position and next position are in the same segment
        # and segment_ids > 0 (not padding). Size is seq_len (not target_len).
        loss_mask = np.zeros(self._seq_len, dtype=np.float32)
        for i in range(self._seq_len):
            if segment_ids[i] > 0 and segment_ids[i] == segment_ids[i + 1]:
                loss_mask[i] = 1.0

        return {"tokens": tokens, "segment_ids": segment_ids, "loss_mask": loss_mask}

    def _load_or_build_indices(
        self,
        data_path: str,
        seq_len: int,
        seed: int,
        num_epochs: int,
        num_samples: int,
        split: tuple[int, int],
        cache_dir: Path,
        add_extra_token: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load cached indices or build and cache them."""
        config_str = json.dumps(
            {
                "add_extra_token": add_extra_token,
                "dataset_path": data_path,
                "sequence_length": seq_len,
                "random_seed": seed,
                "split": f"{split[0]}:{split[1]}",
                "num_epochs": num_epochs,
                "num_samples": num_samples,
            },
            indent=4,
            sort_keys=True,
        )
        config_hash = hashlib.md5(config_str.encode()).hexdigest()

        cache_dir.mkdir(parents=True, exist_ok=True)
        doc_idx_path = cache_dir / f"{config_hash}-document_index.npy"
        sample_idx_path = cache_dir / f"{config_hash}-sample_index.npy"
        shuffle_idx_path = cache_dir / f"{config_hash}-shuffle_index.npy"

        if doc_idx_path.exists() and sample_idx_path.exists() and shuffle_idx_path.exists():
            max_logging.log(f"Loading cached indices from {cache_dir}")
            document_index = np.load(doc_idx_path)
            sample_index = np.load(sample_idx_path)
            shuffle_index = np.load(shuffle_idx_path)
            return document_index, sample_index, shuffle_index

        max_logging.log(f"Building indices (seq_len={seq_len}, epochs={num_epochs}, docs={len(self._doc_ids)})...")

        rng = np.random.RandomState(seed)
        num_split_docs = len(self._doc_ids)

        document_index = _build_document_index(num_split_docs, num_epochs, rng)
        sample_index = _build_sample_index(
            document_index, self._split_doc_lengths, seq_len,
            add_extra_token=add_extra_token,
        )

        total_samples = len(sample_index) - 1
        shuffle_index = _build_shuffle_index_with_separate_epoch(
            sample_index, document_index, num_split_docs, num_epochs,
            num_samples, total_samples, rng,
        )

        np.save(doc_idx_path, document_index)
        np.save(sample_idx_path, sample_index)
        np.save(shuffle_idx_path, shuffle_index)
        max_logging.log(f"Built {len(sample_index) - 1} samples, cached to {cache_dir}")

        return document_index, sample_index, shuffle_index

    def _load_or_build_packed_indices(
        self,
        data_path: str,
        seq_len: int,
        seed: int,
        num_epochs: int,
        num_samples: int,
        split: tuple[int, int],
        cache_dir: Path,
        add_extra_token: bool = True,
        packing_type: str = "first_fit",
        max_chunks_per_sample: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load cached packed indices or build and cache them."""
        config_str = json.dumps(
            {
                "add_extra_token": add_extra_token,
                "dataset_path": data_path,
                "sequence_length": seq_len,
                "random_seed": seed,
                "split": f"{split[0]}:{split[1]}",
                "num_epochs": num_epochs,
                "num_samples": num_samples,
                "packing_type": packing_type,
                "max_chunks_per_sample": max_chunks_per_sample,
            },
            indent=4,
            sort_keys=True,
        )
        config_hash = hashlib.md5(config_str.encode()).hexdigest()

        cache_dir.mkdir(parents=True, exist_ok=True)
        chunk_idx_path = cache_dir / f"{config_hash}-chunk_index.npy"
        sample_idx_path = cache_dir / f"{config_hash}-sample_index.npy"
        shuffle_idx_path = cache_dir / f"{config_hash}-shuffle_index.npy"

        if chunk_idx_path.exists() and sample_idx_path.exists() and shuffle_idx_path.exists():
            max_logging.log(f"Loading cached packed indices from {cache_dir}")
            return np.load(chunk_idx_path), np.load(sample_idx_path), np.load(shuffle_idx_path)

        max_logging.log(
            f"Building packed indices ({packing_type}, seq_len={seq_len}, epochs={num_epochs}, docs={len(self._doc_ids)})..."
        )

        rng = np.random.RandomState(seed)
        num_split_docs = len(self._doc_ids)
        document_index = _build_document_index(num_split_docs, num_epochs, rng)
        chunk_index, sample_index = build_packed_sample_index(
            document_index, self._split_doc_lengths, seq_len,
            packing_type=packing_type,
            add_extra_token=add_extra_token,
            max_chunks_per_sample=max_chunks_per_sample,
        )

        total_samples = len(sample_index) - 1
        # For packed indices, we don't have sample_index with doc positions,
        # so we use a simple full shuffle (no separate_final_epoch logic)
        shuffle_index = _build_shuffle_index(total_samples, total_samples, rng)

        np.save(chunk_idx_path, chunk_index)
        np.save(sample_idx_path, sample_index)
        np.save(shuffle_idx_path, shuffle_index)
        max_logging.log(f"Built {len(sample_index) - 1} packed samples, cached to {cache_dir}")

        return chunk_index, sample_index, shuffle_index


def _build_document_index(
    num_documents: int, num_epochs: int, rng: np.random.RandomState
) -> np.ndarray:
    """Shuffled document ordering for N epochs. Returns 1-D int32 array."""
    epochs = []
    for _ in range(num_epochs):
        order = np.arange(num_documents, dtype=np.int32)
        rng.shuffle(order)
        epochs.append(order)
    return np.concatenate(epochs)


def _build_sample_index(
    document_index: np.ndarray,
    doc_lengths: np.ndarray,
    seq_len: int,
    add_extra_token: bool = True,
) -> np.ndarray:
    """Map samples to document spans.

    Returns 2-D int64 array of shape (N+1, 2).
    Each row = (doc_index_position, offset_within_doc).
    Pure NumPy implementation of sample index building.
    """
    target = seq_len + add_extra_token
    samples = [(0, 0)]
    doc_pos = 0
    offset = 0
    remaining = target

    while doc_pos < len(document_index):
        doc_id = int(document_index[doc_pos])
        doc_len = int(doc_lengths[doc_id])
        available = doc_len - offset

        if available <= 0:
            doc_pos += 1
            offset = 0
            continue

        if available >= remaining:
            offset += remaining
            samples.append((doc_pos, offset))
            remaining = target
            if offset >= doc_len:
                doc_pos += 1
                offset = 0
        else:
            remaining -= available
            doc_pos += 1
            offset = 0

    return np.array(samples, dtype=np.int64)


def _build_shuffle_index_with_separate_epoch(
    sample_index: np.ndarray,
    document_index: np.ndarray,
    num_documents: int,
    num_epochs: int,
    num_samples: int,
    total_samples: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Build shuffle_index with automatic separate_final_epoch handling.

    Megatron pattern: if the final epoch is incomplete and contributes <80%
    of a full epoch's samples, keep final-epoch samples at the end
    (shuffled among themselves) to avoid mixing partial-epoch data with
    earlier complete epochs.
    """
    if num_epochs <= 1:
        return _build_shuffle_index(total_samples, total_samples, rng)

    # Find the boundary where samples start coming from the final epoch.
    # A sample belongs to the final epoch if its starting document position
    # in the document_index falls in the last epoch's range.
    final_epoch_doc_start = num_documents * (num_epochs - 1)

    # Find the first sample whose doc position is >= final_epoch_doc_start
    num_samples_sans_final = total_samples
    for i in range(total_samples):
        doc_pos = int(sample_index[i, 0])
        if doc_pos >= final_epoch_doc_start:
            num_samples_sans_final = i
            break

    num_samples_from_final_epoch = total_samples - num_samples_sans_final

    if num_samples_from_final_epoch == 0:
        # No final epoch samples, just shuffle everything
        return _build_shuffle_index(total_samples, total_samples, rng)

    # Compute samples per full epoch
    num_samples_per_epoch = num_samples_sans_final / max(num_epochs - 1, 1)

    # separate_final_epoch: True if final epoch contributes <80% of a full epoch
    separate = num_samples_from_final_epoch < int(0.80 * num_samples_per_epoch)

    if separate:
        return _build_shuffle_index(num_samples_sans_final, total_samples, rng)
    else:
        return _build_shuffle_index(total_samples, total_samples, rng)


class FixedArecordDataSource(grain.RandomAccessDataSource):
    """Fixed-length pre-tokenized arecord data source.

    Each record is a complete training sample of seq_len+1 int32 tokens.
    No packing or indexing needed — Grain's ArrayRecordDataSource handles
    sharded file loading directly.
    """

    def __init__(self, data_path: str, seq_len: int) -> None:
        super().__init__()
        import glob

        self._data_path = data_path
        self._seq_len = seq_len

        files = sorted(glob.glob(os.path.join(data_path, "*.arecord")))
        if not files:
            raise FileNotFoundError(f"No .arecord files found in {data_path}")
        self._source = grain.ArrayRecordDataSource(files)

        # Validate record size: each record must be exactly (seq_len+1) int32 tokens
        num_tokens = seq_len + 1
        first_record = self._source[0]
        actual_tokens = len(first_record) // 4
        if len(first_record) != num_tokens * 4:
            raise ValueError(
                f"fixed_arecord record size mismatch: got {actual_tokens} int32 tokens, "
                f"expected {num_tokens} (max_target_length + 1)"
            )

        max_logging.log(f"FixedArecordDataSource: {len(files)} shards, {len(self._source)} records, {num_tokens} tokens/record")

    def __repr__(self) -> str:
        return f"FixedArecordDataSource(data_path={self._data_path!r}, seq_len={self._seq_len})"

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        record = self._source[idx]
        tokens = np.frombuffer(record, dtype=np.int32).copy()
        return {"tokens": tokens}


class SyntheticDataSource(grain.RandomAccessDataSource):
    """Generate random token sequences for debugging and development."""

    def __init__(self, seq_len: int, num_samples: int, vocab_size: int = 151936, seed: int = 42) -> None:
        super().__init__()
        self._seq_len = seq_len
        self._num_samples = num_samples
        self._vocab_size = vocab_size
        self._seed = seed

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        rng = np.random.RandomState(self._seed + idx)
        tokens = rng.randint(0, self._vocab_size, size=self._seq_len + 1, dtype=np.int32)
        return {"tokens": tokens}


class BlendedDataSource(grain.RandomAccessDataSource):
    """Blends multiple data sources with Megatron-style greedy-by-error interleaving.

    Pre-builds dataset_index and dataset_sample_index arrays using vectorized
    numpy for O(size) construction and O(1) access.
    """

    def __init__(
        self,
        sources: list,
        weights: list[float],
        size: int,
    ) -> None:
        super().__init__()
        if len(sources) != len(weights):
            raise ValueError("sources and weights must have same length")
        if not sources:
            raise ValueError("Must provide at least one source")

        self._sources = sources
        self._size = size

        source_lens = np.array([len(s) for s in sources], dtype=np.int64)
        self._dataset_index, self._dataset_sample_index = _build_blend_indices(
            weights, source_lens, size,
        )

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        ds_id = int(self._dataset_index[idx])
        sample_id = int(self._dataset_sample_index[idx])
        return self._sources[ds_id][sample_id]


def _build_blend_indices(
    weights: list[float],
    source_lens: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Megatron-style greedy-by-error blend index construction (vectorized).

    For each position i, assigns the source whose cumulative target count
    most exceeds its actual count. This produces even interleaving proportional
    to the given weights.

    Args:
        weights: Per-source sampling weights.
        source_lens: Number of samples in each source.
        size: Total number of blend indices to generate.

    Returns:
        (dataset_index, dataset_sample_index) — both shape (size,).
    """
    num_sources = len(weights)
    total_w = sum(weights)
    norm_weights = np.array([w / total_w for w in weights], dtype=np.float64)

    # Per-source target counts at each position: target[ds] = weight[ds] * (i+1)
    # Instead of iterating, compute the assignment for each source in bulk:
    # source ds gets assigned at positions where floor(target) increments.
    #
    # For source ds with weight w, its k-th sample appears at position
    # ceil((k+1) / w) - 1 in the greedy-by-error schedule. We compute
    # all positions for all sources, then sort to get the interleaved order.

    dataset_index = np.empty(size, dtype=np.int32)
    dataset_sample_index = np.empty(size, dtype=np.int64)

    # Compute per-source counts (how many samples each source contributes)
    per_source = np.round(norm_weights * size).astype(np.int64)
    diff = size - per_source.sum()
    per_source[np.argmax(norm_weights)] += diff  # fix rounding

    # For each source, compute the ideal positions using Megatron's
    # greedy-by-error formula: source ds's k-th sample goes at position
    # floor((k + 0.5) / weight) approximately. We use the exact formula:
    # position_k = smallest i where weight * (i+1) > k, i.e. i = ceil((k+1)/weight) - 1
    all_positions = []
    all_ds_ids = []
    all_sample_ids = []
    for ds in range(num_sources):
        count = int(per_source[ds])
        if count == 0:
            continue
        w = norm_weights[ds]
        k = np.arange(count, dtype=np.float64)
        # Position where the k-th sample of this source would be placed
        # in greedy-by-error: ceil((k+1)/w) - 1
        positions = np.ceil((k + 1) / w).astype(np.int64) - 1
        positions = np.clip(positions, 0, size - 1)
        all_positions.append(positions)
        all_ds_ids.append(np.full(count, ds, dtype=np.int32))
        all_sample_ids.append(k.astype(np.int64) % source_lens[ds])

    all_positions = np.concatenate(all_positions)
    all_ds_ids = np.concatenate(all_ds_ids)
    all_sample_ids = np.concatenate(all_sample_ids)

    # Sort by position to get the interleaved order.
    # Ties broken by source weight (heavier sources first) for consistency.
    sort_keys = all_positions * num_sources + all_ds_ids
    order = np.argsort(sort_keys, kind="stable")

    # Take the first `size` entries (should be exactly size after rounding fix)
    dataset_index[:] = all_ds_ids[order[:size]]
    dataset_sample_index[:] = all_sample_ids[order[:size]]

    return dataset_index, dataset_sample_index
