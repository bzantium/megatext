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

"""Sample index builders: greedy (cross-document) and best-fit bin packing.

Both use C++ implementations from megatext.data._helpers for speed.
"""

from __future__ import annotations

import numpy as np

from megatext.data._helpers import build_best_fit_sample_idx, build_greedy_sample_idx
from megatext.utils import logging as max_logging


def build_greedy_sample_index(
    document_index: np.ndarray,
    doc_lengths: np.ndarray,
    seq_len: int,
    add_extra_token: bool = True,
    num_epochs: int = 1,
    tokens_per_epoch: int = 0,
) -> np.ndarray:
    """Build sample index using greedy cross-document concatenation (C++).

    Returns 2-D int64 array of shape (N+1, 2).
    Each row = (doc_index_position, offset_within_doc).
    """
    if tokens_per_epoch <= 0:
        tokens_per_epoch = int(np.sum(doc_lengths[document_index]))

    return build_greedy_sample_idx(
        doc_lengths.astype(np.int32, copy=False),
        document_index.astype(np.int64, copy=False),
        seq_len,
        num_epochs,
        tokens_per_epoch,
        True,  # drop_last_partial_sequence
        1 if add_extra_token else 0,
    )


def build_packed_sample_index(
    document_index: np.ndarray,
    doc_lengths: np.ndarray,
    seq_len: int,
    add_extra_token: bool = True,
    max_chunks_per_sample: int = 0,
    num_epochs: int = 1,
    tokens_per_epoch: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build packed sample index using best-fit bin packing (C++).

    Args:
        document_index: 1-D int64 array of document IDs (shuffled).
        doc_lengths: Per-document token counts (int32, indexed by doc ID).
        seq_len: Target sequence length per sample.
        add_extra_token: Whether samples need an extra token for input/target shift.
        max_chunks_per_sample: Max documents per packed sample (0 = unlimited).
        num_epochs: Number of epochs.
        tokens_per_epoch: Total tokens per epoch.

    Returns:
        chunk_index: ``[N, 3]`` int64 — (doc_id, offset, length).
        sample_index: ``[M+1]`` int64 — boundaries into chunk_index.
    """
    if tokens_per_epoch <= 0:
        tokens_per_epoch = int(np.sum(doc_lengths[document_index]))

    max_chunks_cpp = max_chunks_per_sample if max_chunks_per_sample > 0 else -1
    chunk_index, sample_index = build_best_fit_sample_idx(
        doc_lengths.astype(np.int32, copy=False),
        document_index.astype(np.int64, copy=False),
        seq_len,
        num_epochs,
        tokens_per_epoch,
        True,  # drop_last_partial_sequence
        1 if add_extra_token else 0,
        max_chunks_cpp,
    )
    num_samples = len(sample_index) - 1
    num_full = int(np.sum(doc_lengths[document_index] // seq_len))
    num_packed = num_samples - num_full
    max_logging.log(
        f"Packing (best_fit/C++): {num_full} full + {num_packed} packed = {num_samples} samples"
    )
    return np.asarray(chunk_index, dtype=np.int64), np.asarray(sample_index, dtype=np.int64)
