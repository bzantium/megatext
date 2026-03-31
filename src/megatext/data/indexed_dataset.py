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

"""Low-level readers for .bin+.idx and document-level .arecord formats.

Both expose the same document-level interface so upstream indexing code is
backend-agnostic: __len__, __getitem__, get(idx, offset, length), doc_lengths.

Naming convention (strict 1:1):
  - Single prefix: {prefix}.idx + {prefix}.bin  OR  {prefix}.idx + {prefix}.arecord
  - Directory:     glob *.idx → match each with same-prefix .bin/.arecord
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


# --- .idx format constants ---

_IDX_MAGIC = b"MMIDIDX\x00\x00"
_IDX_HEADER_SIZE = 9 + 8 + 1 + 8 + 8  # magic(9) + version(8) + dtype(1) + seq_count(8) + doc_count(8) = 34

_DTYPE_CODE_TO_NUMPY = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}

_NUMPY_TO_DTYPE_CODE = {v: k for k, v in _DTYPE_CODE_TO_NUMPY.items()}


def write_index(path: str | Path, doc_lengths: np.ndarray) -> None:
    """Write document lengths to an index.idx file (raw int32 binary)."""
    arr = np.asarray(doc_lengths, dtype=np.int32)
    with open(path, "wb") as f:
        f.write(arr.tobytes())


def read_index(path: str | Path) -> np.ndarray:
    """Read document lengths from an index.idx file (raw int32 binary)."""
    data = Path(path).read_bytes()
    return np.frombuffer(data, dtype=np.int32).copy()


class MMapIndexedDataset:
    """Read-only access to a single .bin+.idx mmap file pair.

    Documents are contiguous runs of sequences delimited by document_indices.
    """

    def __init__(self, path_prefix: str) -> None:
        idx_path = Path(f"{path_prefix}.idx")
        bin_path = Path(f"{path_prefix}.bin")

        if not idx_path.exists():
            raise FileNotFoundError(f"Index file not found: {idx_path}")
        if not bin_path.exists():
            raise FileNotFoundError(f"Data file not found: {bin_path}")

        with open(idx_path, "rb") as f:
            magic = f.read(9)
            if magic != _IDX_MAGIC:
                raise ValueError(f"Invalid .idx magic: {magic!r}")

            version = struct.unpack("<Q", f.read(8))[0]
            if version != 1:
                raise ValueError(f"Unsupported .idx version: {version}")

            dtype_code = struct.unpack("<B", f.read(1))[0]
            if dtype_code not in _DTYPE_CODE_TO_NUMPY:
                raise ValueError(f"Unknown dtype code: {dtype_code}")
            self._dtype = _DTYPE_CODE_TO_NUMPY[dtype_code]
            self._dtype_size = np.dtype(self._dtype).itemsize

            seq_count = struct.unpack("<Q", f.read(8))[0]
            doc_count = struct.unpack("<Q", f.read(8))[0]

        idx_mmap = np.memmap(idx_path, mode="r", order="C")
        offset = _IDX_HEADER_SIZE

        sl_bytes = seq_count * 4
        self._sequence_lengths = np.frombuffer(
            idx_mmap, dtype=np.int32, count=seq_count, offset=offset
        )
        offset += sl_bytes

        sp_bytes = seq_count * 8
        self._sequence_pointers = np.frombuffer(
            idx_mmap, dtype=np.int64, count=seq_count, offset=offset
        )
        offset += sp_bytes

        self._document_indices = np.frombuffer(
            idx_mmap, dtype=np.int64, count=doc_count, offset=offset
        )

        self._idx_mmap = idx_mmap
        self._bin_mmap = np.memmap(bin_path, mode="r", dtype=np.uint8)
        self._num_docs = len(self._document_indices) - 1

    def __len__(self) -> int:
        return self._num_docs

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0:
            idx += self._num_docs
        if idx < 0 or idx >= self._num_docs:
            raise IndexError(f"Document index {idx} out of range [0, {self._num_docs})")

        seq_start = int(self._document_indices[idx])
        seq_end = int(self._document_indices[idx + 1])

        if seq_end - seq_start == 1:
            return self._read_sequence(seq_start)

        parts = []
        for seq_idx in range(seq_start, seq_end):
            parts.append(self._read_sequence(seq_idx))
        return np.concatenate(parts)

    def get(self, idx: int, offset: int = 0, length: int | None = None) -> np.ndarray:
        tokens = self[idx]
        if length is None:
            return tokens[offset:]
        return tokens[offset : offset + length]

    @property
    def doc_lengths(self) -> np.ndarray:
        cumsum = np.zeros(len(self._sequence_lengths) + 1, dtype=np.int64)
        np.cumsum(self._sequence_lengths, out=cumsum[1:])
        starts = self._document_indices[:-1].astype(np.intp)
        ends = self._document_indices[1:].astype(np.intp)
        return (cumsum[ends] - cumsum[starts]).astype(np.int32)

    def _read_sequence(self, seq_idx: int) -> np.ndarray:
        pointer = int(self._sequence_pointers[seq_idx])
        length = int(self._sequence_lengths[seq_idx])
        return np.frombuffer(
            self._bin_mmap, dtype=self._dtype, count=length, offset=pointer,
        ).copy()


class ArrayRecordDocDataset:
    """Read-only access to a single .arecord+.idx file pair."""

    def __init__(self, path_prefix: str) -> None:
        from array_record.python.array_record_module import ArrayRecordReader

        idx_path = Path(f"{path_prefix}.idx")
        arecord_path = Path(f"{path_prefix}.arecord")

        if not idx_path.exists():
            raise FileNotFoundError(f"Index file not found: {idx_path}")
        if not arecord_path.exists():
            raise FileNotFoundError(f"Data file not found: {arecord_path}")

        self._doc_lengths_arr = read_index(idx_path)
        self._reader = ArrayRecordReader(str(arecord_path))
        self._total_docs = self._reader.num_records()

        if len(self._doc_lengths_arr) != self._total_docs:
            raise ValueError(
                f"doc_lengths has {len(self._doc_lengths_arr)} entries "
                f"but arecord file has {self._total_docs} records"
            )

        first_record = self._reader.read([0])[0]
        n_bytes = len(first_record)
        if n_bytes == self._doc_lengths_arr[0] * 2:
            self._token_dtype = np.uint16
        elif n_bytes == self._doc_lengths_arr[0] * 4:
            self._token_dtype = np.int32
        else:
            self._token_dtype = np.int32

    def __len__(self) -> int:
        return self._total_docs

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0:
            idx += self._total_docs
        if idx < 0 or idx >= self._total_docs:
            raise IndexError(f"Document index {idx} out of range [0, {self._total_docs})")

        record = self._reader.read([idx])[0]
        return np.frombuffer(record, dtype=self._token_dtype).copy()

    def get(self, idx: int, offset: int = 0, length: int | None = None) -> np.ndarray:
        tokens = self[idx]
        if length is None:
            return tokens[offset:]
        return tokens[offset : offset + length]

    @property
    def doc_lengths(self) -> np.ndarray:
        return self._doc_lengths_arr


class MultiFileIndexedDataset:
    """Concatenates multiple sub-datasets discovered from *.idx files in a directory.

    Globs *.idx at the top level (no subdirectories), strips .idx to get
    the prefix, validates that a matching data file (prefix + data_ext) exists,
    then creates one sub-dataset per pair.
    """

    def __init__(self, path: str, dataset_cls: type, data_ext: str) -> None:
        p = Path(path)
        if not p.is_dir():
            raise FileNotFoundError(f"Not a directory: {p}")

        idx_files = sorted(p.glob("*.idx"))
        if not idx_files:
            raise FileNotFoundError(f"No .idx files found in {p}")

        # Validate 1:1 matching: each .idx must have a matching data file
        self._datasets = []
        self._offsets: list[int] = []
        total = 0
        for idx_file in idx_files:
            prefix = str(idx_file)[: -len(".idx")]
            data_file = Path(f"{prefix}{data_ext}")
            if not data_file.exists():
                raise FileNotFoundError(
                    f"No matching data file for {idx_file.name}: "
                    f"expected {data_file.name} in {p}"
                )
            ds = dataset_cls(prefix)
            self._datasets.append(ds)
            self._offsets.append(total)
            total += len(ds)

        self._total_len = total

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0:
            idx += self._total_len
        if idx < 0 or idx >= self._total_len:
            raise IndexError(f"Index {idx} out of range [0, {self._total_len})")

        ds_id = self._find_dataset(idx)
        local_idx = idx - self._offsets[ds_id]
        return self._datasets[ds_id][local_idx]

    def get(self, idx: int, offset: int = 0, length: int | None = None) -> np.ndarray:
        if idx < 0:
            idx += self._total_len
        if idx < 0 or idx >= self._total_len:
            raise IndexError(f"Index {idx} out of range [0, {self._total_len})")

        ds_id = self._find_dataset(idx)
        local_idx = idx - self._offsets[ds_id]
        return self._datasets[ds_id].get(local_idx, offset=offset, length=length)

    @property
    def doc_lengths(self) -> np.ndarray:
        return np.concatenate([ds.doc_lengths for ds in self._datasets])

    def _find_dataset(self, idx: int) -> int:
        lo, hi = 0, len(self._offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        return lo


# --- Factory functions ---


def make_mmap_dataset(path: str) -> MMapIndexedDataset | MultiFileIndexedDataset:
    """Create an mmap dataset from a prefix (single) or directory (multi-file)."""
    if Path(f"{path}.idx").is_file() and Path(f"{path}.bin").is_file():
        return MMapIndexedDataset(path)
    if Path(path).is_dir():
        return MultiFileIndexedDataset(path, MMapIndexedDataset, ".bin")
    raise FileNotFoundError(
        f"mmap dataset not found at '{path}': expected {path}.idx + {path}.bin, "
        f"or a directory containing *.idx + *.bin pairs."
    )


def make_arecord_dataset(path: str) -> ArrayRecordDocDataset | MultiFileIndexedDataset:
    """Create an arecord dataset from a prefix (single) or directory (multi-file)."""
    if Path(f"{path}.idx").is_file() and Path(f"{path}.arecord").is_file():
        return ArrayRecordDocDataset(path)
    if Path(path).is_dir():
        return MultiFileIndexedDataset(path, ArrayRecordDocDataset, ".arecord")
    raise FileNotFoundError(
        f"arecord dataset not found at '{path}': expected {path}.idx + {path}.arecord, "
        f"or a directory containing *.idx + *.arecord pairs."
    )
