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

"""Sequence packing algorithms for document-isolated training.

Implements first-fit and best-fit bin packing to pack document remainders
into fixed-length samples while tracking document boundaries.

Both algorithms use First-Fit Decreasing (FFD) / Best-Fit Decreasing (BFD)
ordering — remainders are sorted by size descending before packing.

**FFD ≡ BFD equivalence (offline mode, no chunk limit):**  When
``max_chunks_per_sample == 0`` (unlimited), remainders are sorted in
descending order, so the first bin with enough capacity is *always* the
tightest-fitting bin — every candidate gap is at least as large as the
current item, so the first feasible gap is the smallest feasible gap.  This
means ``first_fit`` and ``best_fit`` produce **identical** bin counts and
total padding.  ``first_fit`` is slightly faster because it skips the
``argmin`` over feasible bins.

When ``max_chunks_per_sample > 0``, the chunk-count constraint can exclude
different bins from the feasible set, breaking the monotonicity that
guarantees equivalence.  In practice the difference is negligible, but the
results may not be bit-identical.

The two algorithms only diverge significantly in *online* (unsorted)
packing scenarios, where arrival order breaks the monotonicity assumption.

Inner loops are vectorized with numpy: each element checks all existing bins
simultaneously via ``np.flatnonzero``, avoiding Python-level iteration over
bins.
"""

from __future__ import annotations

import numpy as np

from megatext.utils import logging as max_logging


def build_packed_sample_index(
    document_index: np.ndarray,
    doc_lengths: np.ndarray,
    seq_len: int,
    packing_type: str,
    add_extra_token: bool = True,
    max_chunks_per_sample: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build packed sample index using bin packing.

    Full *seq_len* chunks from documents become single-chunk samples.
    Remainders (``doc_len % seq_len``) are packed into bins using the
    specified packing algorithm.

    .. note::

       In offline (sorted) mode with ``max_chunks_per_sample=0``,
       ``first_fit`` and ``best_fit`` produce identical results — same bin
       count and same total padding.  ``first_fit`` is marginally faster
       due to no ``argmin`` overhead.  With a chunk limit the results may
       differ slightly.  See module docstring for details.

    Args:
        document_index: 1-D array of document IDs (shuffled, possibly
            multi-epoch).
        doc_lengths: Per-document token counts (indexed by doc ID).
        seq_len: Target sequence length per sample.
        packing_type: ``"first_fit"`` or ``"best_fit"``.
        add_extra_token: Whether samples need an extra token for
            input/target shift.
        max_chunks_per_sample: Max documents per packed sample
            (0 = unlimited).

    Returns:
        chunk_index: ``[N, 3]`` int64 array.  Columns are
            ``(doc_id, offset, length)``.
        sample_index: ``[M+1]`` int64 array.  Boundaries into
            *chunk_index*.
    """
    # Phase 1: Separate full chunks and remainders.
    full_chunks: list[tuple[int, int, int]] = []
    remainder_docs: list[tuple[int, int, int]] = []

    for doc_idx_pos in range(len(document_index)):
        doc_id = int(document_index[doc_idx_pos])
        doc_len = int(doc_lengths[doc_id])
        num_full = doc_len // seq_len
        remainder = doc_len % seq_len

        for i in range(num_full):
            full_chunks.append((doc_id, i * seq_len, seq_len))

        if remainder > 0:
            remainder_docs.append((doc_id, remainder, num_full * seq_len))

    # Phase 2: Bin-pack remainders.
    if packing_type == "first_fit":
        bins = _first_fit_pack(remainder_docs, seq_len, max_chunks_per_sample)
    elif packing_type == "best_fit":
        bins = _best_fit_pack(remainder_docs, seq_len, max_chunks_per_sample)
    else:
        raise ValueError(f"Unknown packing_type: {packing_type}")

    # Phase 3: Assemble chunk_index and sample_index.
    total_chunks = len(full_chunks) + sum(len(b) for b in bins)
    chunk_index = np.zeros((total_chunks, 3), dtype=np.int64)
    sample_boundaries: list[int] = []

    ci = 0

    # Full chunks: one sample per chunk.
    for doc_id, offset, length in full_chunks:
        sample_boundaries.append(ci)
        chunk_index[ci] = (doc_id, offset, length)
        ci += 1

    # Packed bins: one sample per bin.
    for bin_contents in bins:
        sample_boundaries.append(ci)
        for doc_id, offset, length in bin_contents:
            chunk_index[ci] = (doc_id, offset, length)
            ci += 1

    # Final boundary.
    sample_boundaries.append(ci)
    sample_index = np.array(sample_boundaries, dtype=np.int64)

    num_samples = len(sample_index) - 1
    packed_tokens = sum(sum(length for _, _, length in b) for b in bins)
    waste = len(bins) * seq_len - packed_tokens if bins else 0
    efficiency = 100.0 * packed_tokens / (packed_tokens + waste) if packed_tokens > 0 else 100.0
    max_logging.log(
        f"Packing ({packing_type}): {len(full_chunks)} full + {len(bins)} packed = {num_samples} samples, "
        f"{len(remainder_docs)} remainder docs, {efficiency:.1f}% packing efficiency"
    )

    return chunk_index, sample_index


def _first_fit_pack(
    remainder_docs: list[tuple[int, int, int]],
    seq_len: int,
    max_chunks_per_sample: int,
) -> list[list[tuple[int, int, int]]]:
    """First-fit decreasing bin packing (numpy-vectorized)."""
    return _bin_pack(remainder_docs, seq_len, max_chunks_per_sample, best_fit=False)


def _best_fit_pack(
    remainder_docs: list[tuple[int, int, int]],
    seq_len: int,
    max_chunks_per_sample: int,
) -> list[list[tuple[int, int, int]]]:
    """Best-fit decreasing bin packing (numpy-vectorized)."""
    return _bin_pack(remainder_docs, seq_len, max_chunks_per_sample, best_fit=True)


def _bin_pack(
    remainder_docs: list[tuple[int, int, int]],
    seq_len: int,
    max_chunks_per_sample: int,
    *,
    best_fit: bool,
) -> list[list[tuple[int, int, int]]]:
    """Decreasing bin packing (numpy-vectorized).

    For each element, all existing bins are checked simultaneously via
    ``np.flatnonzero``.  When *best_fit* is True the tightest-fitting bin
    is selected; otherwise the first feasible bin is used.
    """
    sorted_docs = sorted(remainder_docs, key=lambda x: x[1], reverse=True)

    max_bins = len(sorted_docs)
    bins: list[list[tuple[int, int, int]]] = []
    bin_remaining = np.zeros(max_bins, dtype=np.int64)
    bin_counts = np.zeros(max_bins, dtype=np.int64)
    num_bins = 0

    for doc_id, remainder, offset in sorted_docs:
        if remainder <= 0:
            continue

        fits = bin_remaining[:num_bins] >= remainder
        if max_chunks_per_sample > 0:
            fits &= bin_counts[:num_bins] < max_chunks_per_sample

        valid = np.flatnonzero(fits)
        if len(valid) > 0:
            i = valid[np.argmin(bin_remaining[:num_bins][valid])] if best_fit else valid[0]
            bins[i].append((doc_id, offset, remainder))
            bin_remaining[i] -= remainder
            bin_counts[i] += 1
        else:
            bins.append([(doc_id, offset, remainder)])
            bin_remaining[num_bins] = seq_len - remainder
            bin_counts[num_bins] = 1
            num_bins += 1

    return bins
