"""Convert raw text (jsonl/parquet/gz/zstd) to document-level .arecord format.

Tokenizes text, writes each document as a separate ArrayRecord record.
Output is a directory of shard-{shard_idx}-{part_idx}.{arecord,idx} pairs,
loadable via make_arecord_dataset(output_dir).

Supports Slurm-style parallel preprocessing via --num-shards / --shard-idx.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing

import numpy as np

from _common import (
    ShardedArrayRecordWriter,
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


def text_to_arecord(
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
    """Convert raw text files to document-level .arecord format.

    Args:
        input_files: Resolved input file paths (jsonl/parquet/gz/zstd).
        output_dir: Output directory for .arecord + .idx shard pairs.
        tokenizer_path: Path or name of the tokenizer.
        tokenizer_type: "huggingface" or "sentencepiece".
        max_file_size: Maximum size per output shard (e.g. "50G", "500M").
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

    writer = ShardedArrayRecordWriter(output_dir, max_bytes, shard_idx=shard_idx, prefix=prefix)
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
            writer.write(arr.tobytes(), doc_length=len(token_ids))
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

    text_to_arecord(
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
