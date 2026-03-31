"""Split a single .bin+.idx mmap dataset into a directory of smaller shard pairs.

Input:  prefix.bin + prefix.idx (single mmap dataset)
Output: output_dir/shard-{shard_idx:03d}-{part_idx:05d}.{bin,idx} pairs,
        loadable via make_mmap_dataset(output_dir).
"""

from __future__ import annotations

import argparse
import logging

from megatext.data.indexed_dataset import MMapIndexedDataset

from _common import parse_file_size
from text_to_mmap import ShardedMMapWriter


def split_mmap(
    input: str,
    output: str,
    max_file_size: str = "50G",
    prefix: str = "shard",
) -> None:
    """Split a single .bin+.idx mmap dataset into smaller shard pairs.

    Args:
        input: Path prefix for .bin/.idx files (without extension).
        output: Output directory for split .bin + .idx shard pairs.
        max_file_size: Maximum size per output .bin file (e.g. "50G", "500M").
        prefix: Filename prefix for shards.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    max_bytes = parse_file_size(max_file_size)

    dataset = MMapIndexedDataset(input)
    num_docs = len(dataset)
    logging.info("Opened %s: %d documents", input, num_docs)

    writer = ShardedMMapWriter(output, max_bytes, dtype=dataset._dtype, prefix=prefix)

    for doc_idx in range(num_docs):
        tokens = dataset[doc_idx]
        writer.write(tokens)

        if (doc_idx + 1) % 10000 == 0:
            logging.info("Processed %d / %d documents", doc_idx + 1, num_docs)

    total_parts = writer.close()
    logging.info("Split %d documents into %d parts in %s", num_docs, total_parts, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path prefix for .bin/.idx files (without extension)")
    parser.add_argument("--output", required=True, help="Output directory for split .bin + .idx shard pairs")
    parser.add_argument("--max-file-size", default="50G", help="Maximum size per output .bin file (e.g. 5G, 500M)")
    parser.add_argument("--prefix", default="shard", help="Filename prefix for shards (default: shard)")
    args = parser.parse_args()

    split_mmap(
        input=args.input,
        output=args.output,
        max_file_size=args.max_file_size,
        prefix=args.prefix,
    )
