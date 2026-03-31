"""Convert .bin+.idx files to document-level .arecord format.

Each ArrayRecord record = one document's tokens (variable length).
Output is a directory of shard-{shard_idx}-{part_idx}.{arecord,idx} pairs,
loadable via make_arecord_dataset(output_dir).
"""

from __future__ import annotations

import argparse
import logging

from megatext.data.indexed_dataset import MMapIndexedDataset

from _common import ShardedArrayRecordWriter, parse_file_size


def mmap_to_arecord(
    input: str,
    output: str,
    max_file_size: str = "50G",
    prefix: str = "shard",
    shard_idx: int = 0,
) -> None:
    """Convert .bin+.idx mmap files to document-level .arecord format.

    Args:
        input: Path prefix for .bin/.idx files (without extension).
        output: Output directory for .arecord + .idx shard pairs.
        max_file_size: Maximum size per output shard (e.g. "50G", "500M").
        prefix: Filename prefix for shards.
        shard_idx: Shard index for output naming (for Slurm-style parallel runs).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    max_bytes = parse_file_size(max_file_size)

    # Open mmap dataset
    dataset = MMapIndexedDataset(input)
    num_docs = len(dataset)
    logging.info("Opened %s: %d documents", input, num_docs)

    # Use the dtype from the mmap dataset header
    out_dtype = dataset._dtype
    logging.info("Using %s for token storage (from dataset header)", out_dtype)

    doc_lengths = dataset.doc_lengths
    writer = ShardedArrayRecordWriter(output, max_bytes, shard_idx=shard_idx, prefix=prefix)

    for doc_idx in range(num_docs):
        tokens = dataset[doc_idx]
        writer.write(tokens.astype(out_dtype).tobytes(), doc_length=int(doc_lengths[doc_idx]))

        if (doc_idx + 1) % 10000 == 0:
            logging.info("Processed %d / %d documents", doc_idx + 1, num_docs)

    total_parts = writer.close()
    logging.info("Wrote %d documents to %d parts in %s", num_docs, total_parts, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmap-path", required=True, help="Path prefix for .bin/.idx files (without extension)")
    parser.add_argument("--output-path", required=True, help="Output directory for .arecord + .idx shard pairs")
    parser.add_argument("--max-file-size", default="50G", help="Maximum size per output shard (e.g. 5G, 500M)")
    parser.add_argument("--prefix", default="shard", help="Filename prefix for shards (default: shard)")
    parser.add_argument("--shard-idx", type=int, default=0, help="Shard index for output naming")
    args = parser.parse_args()

    mmap_to_arecord(
        input=args.mmap_path,
        output=args.output_path,
        max_file_size=args.max_file_size,
        prefix=args.prefix,
        shard_idx=args.shard_idx,
    )
