"""Convert raw text (jsonl/parquet/gz/zstd) to .bin+.idx mmap format.

Tokenizes text, writes each document as a separate sequence in the mmap
format compatible with MMapIndexedDataset.
Output is a directory of shard-{shard_idx}-{part_idx}.{bin,idx} pairs,
loadable via make_mmap_dataset(output_dir).

Supports Slurm-style parallel preprocessing via --num-shards / --shard-idx.
"""

from __future__ import annotations

import argparse
import array
import logging
import multiprocessing
import struct
from pathlib import Path

import numpy as np
from megatext.data.indexed_dataset import _IDX_MAGIC, _NUMPY_TO_DTYPE_CODE

from _common import (
    Tokenizer,
    add_input_args,
    add_output_args,
    add_sharding_args,
    init_tokenize_worker,
    parse_file_size,
    read_documents,
    resolve_input_files,
    resolve_output,
    tokenize_doc,
)


class ShardedMMapWriter:
    """Write tokens across size-limited .bin+.idx shards into an output directory.

    Each part produces a 1:1 pair of .bin + .idx files, so the output
    directory can be loaded directly via MultiFileIndexedDataset.

    Output: output_dir/shard-{shard_idx:03d}-{part_idx:05d}.{bin,idx}
    """

    def __init__(self, output_dir: str | Path, max_bytes: int, dtype: np.dtype, shard_idx: int = 0, prefix: str = "shard") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_bytes
        self._shard_idx = shard_idx
        self._prefix = prefix
        self._dtype = dtype
        self._dtype_code = _NUMPY_TO_DTYPE_CODE[dtype]
        self._part_idx = 0
        self._current_bytes = 0

        # Per-part accumulators
        self._sequence_lengths: array.array = array.array("i")
        self._sequence_pointers: array.array = array.array("q")
        self._current_pointer = 0

        self._bin_f = self._open_bin()

    def _shard_name(self, part_idx: int) -> str:
        return f"{self._prefix}-{self._shard_idx:03d}-{part_idx:05d}"

    def _open_bin(self):
        path = self._output_dir / f"{self._shard_name(self._part_idx)}.bin"
        return open(path, "wb")

    def write(self, token_ids: np.ndarray) -> None:
        data = token_ids.tobytes()
        record_size = len(data)

        # Split if adding this doc exceeds max_bytes (but not on first doc of part)
        if self._current_bytes > 0 and self._current_bytes + record_size > self._max_bytes:
            self._finalize_part()
            self._part_idx += 1
            self._current_bytes = 0
            self._current_pointer = 0
            self._sequence_lengths = array.array("i")
            self._sequence_pointers = array.array("q")
            self._bin_f = self._open_bin()

        self._sequence_pointers.append(self._current_pointer)
        self._sequence_lengths.append(len(token_ids))
        self._bin_f.write(data)
        self._current_pointer += record_size
        self._current_bytes += record_size

    def _finalize_part(self) -> None:
        """Close current .bin and write matching .idx."""
        self._bin_f.close()

        seq_count = len(self._sequence_lengths)
        idx_path = self._output_dir / f"{self._shard_name(self._part_idx)}.idx"
        with open(idx_path, "wb") as idx_f:
            idx_f.write(_IDX_MAGIC)
            idx_f.write(struct.pack("<Q", 1))  # version
            idx_f.write(struct.pack("<B", self._dtype_code))
            idx_f.write(struct.pack("<Q", seq_count))
            idx_f.write(struct.pack("<Q", seq_count + 1))  # doc_count entries

            np.array(self._sequence_lengths, dtype=np.int32).tofile(idx_f)
            np.array(self._sequence_pointers, dtype=np.int64).tofile(idx_f)
            # Each doc = 1 sequence
            np.arange(seq_count + 1, dtype=np.int64).tofile(idx_f)

    def close(self) -> int:
        """Finalize last part. Returns total part count."""
        self._finalize_part()
        return self._part_idx + 1


def text_to_mmap(
    input_files: list[str],
    output_dir: str,
    tokenizer_path: str,
    tokenizer_type: str = "huggingface",
    max_file_size: str = "50G",
    prefix: str = "shard",
    workers: int = 1,
    text_key: str = "text",
    append_eos: bool = False,
    num_shards: int = 1,
    shard_idx: int = 0,
) -> None:
    """Convert raw text files to .bin+.idx mmap format.

    Args:
        input_files: Resolved input file paths (jsonl/parquet/gz/zstd).
        output_dir: Output directory for .bin + .idx shard pairs.
        tokenizer_path: Path or name of the tokenizer.
        tokenizer_type: "huggingface" or "sentencepiece".
        max_file_size: Maximum size per output .bin file (e.g. "50G", "500M").
        prefix: Filename prefix for shards.
        workers: Number of tokenization workers (within this process).
        text_key: JSON key for the text field.
        append_eos: Whether to append EOS token to each document.
        num_shards: Total number of parallel Slurm jobs.
        shard_idx: This job's index (0-based).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    max_bytes = parse_file_size(max_file_size)

    # Stripe input files across shards
    if num_shards > 1:
        input_files = [input_files[i] for i in range(shard_idx, len(input_files), num_shards)]
        if not input_files:
            logging.warning("Shard %d/%d: no input files assigned, exiting.", shard_idx, num_shards)
            return

    logging.info("Shard %d/%d: processing %d input files", shard_idx, num_shards, len(input_files))

    # Determine vocab size to pick dtype
    tok = Tokenizer(tokenizer_path, type=tokenizer_type)
    out_dtype = np.uint16 if tok.vocab_size < 65536 else np.int32
    logging.info("Vocab size: %d, token dtype: %s", tok.vocab_size, out_dtype)

    writer = ShardedMMapWriter(output_dir, max_bytes, dtype=out_dtype, shard_idx=shard_idx, prefix=prefix)
    doc_count = 0

    documents = read_documents(input_files, text_key=text_key)

    if workers > 1:
        pool = multiprocessing.Pool(
            workers,
            initializer=init_tokenize_worker,
            initargs=(tokenizer_path, tokenizer_type, append_eos),
        )
    else:
        pool = None
        init_tokenize_worker(tokenizer_path, tokenizer_type, append_eos)

    try:
        token_iter = pool.imap(tokenize_doc, documents, chunksize=64) if pool else (tokenize_doc(doc) for doc in documents)

        for token_ids in token_iter:
            if not token_ids:
                continue

            arr = np.array(token_ids, dtype=out_dtype)
            writer.write(arr)
            doc_count += 1

            if doc_count % 10000 == 0:
                logging.info("Processed %d documents", doc_count)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    total_parts = writer.close()
    logging.info("Shard %d/%d: wrote %d documents to %d parts in %s",
                 shard_idx, num_shards, doc_count, total_parts, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_args(parser)
    add_output_args(parser)
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer")
    parser.add_argument("--tokenizer-type", default="huggingface", choices=["huggingface", "sentencepiece"])
    parser.add_argument("--workers", type=int, default=1, help="Number of tokenization workers")
    parser.add_argument("--append-eos", action="store_true", help="Append EOS token to each document")
    add_sharding_args(parser)
    args = parser.parse_args()

    input_files = resolve_input_files(args)
    output_dir, prefix = resolve_output(args)

    text_to_mmap(
        input_files=input_files,
        output_dir=output_dir,
        tokenizer_path=args.tokenizer_path,
        tokenizer_type=args.tokenizer_type,
        max_file_size=args.max_file_size,
        prefix=prefix,
        workers=args.workers,
        text_key=args.text_key,
        append_eos=args.append_eos,
        num_shards=args.num_shards,
        shard_idx=args.shard_idx,
    )
