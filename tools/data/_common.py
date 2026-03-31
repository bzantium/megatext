"""Shared utilities for data preparation tools: tokenizer wrapper, file I/O, size parsing."""

from __future__ import annotations

import gzip
import io
import json
import re
from pathlib import Path

import numpy as np


# --- File size parsing ---

_SIZE_MULTIPLIERS = {
    "K": 1024,
    "KB": 1024,
    "M": 1024**2,
    "MB": 1024**2,
    "G": 1024**3,
    "GB": 1024**3,
    "T": 1024**4,
    "TB": 1024**4,
}

_SIZE_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*(K|KB|M|MB|G|GB|T|TB)$", re.IGNORECASE)


def parse_file_size(size_str: str) -> int:
    """Parse human-readable file size to bytes. Supports K/M/G/T suffixes."""
    s = size_str.strip()
    try:
        return int(s)
    except ValueError:
        pass

    m = _SIZE_PATTERN.match(s)
    if not m:
        raise ValueError(
            f"Invalid file size: {size_str!r}. "
            f"Expected a number with optional suffix (K, M, G, T)."
        )
    value = float(m.group(1))
    suffix = m.group(2).upper()
    return int(value * _SIZE_MULTIPLIERS[suffix])


# --- Tokenizer ---

class Tokenizer:
    """Wraps HuggingFace or SentencePiece tokenizer."""

    def __init__(self, path: str, type: str = "huggingface") -> None:
        if type == "huggingface":
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            self._vocab_size = self._tok.vocab_size
            self._eos_id = self._tok.eos_token_id or 0
            self._pad_id = self._tok.pad_token_id if self._tok.pad_token_id is not None else 0
        elif type == "sentencepiece":
            import sentencepiece as spm

            self._tok = spm.SentencePieceProcessor(model_file=path)
            self._vocab_size = self._tok.get_piece_size()
            self._eos_id = self._tok.eos_id()
            self._pad_id = self._tok.pad_id() if self._tok.pad_id() >= 0 else 0
        else:
            raise ValueError(f"Unknown tokenizer type: {type}")

        self._type = type

    def encode(self, text: str) -> list[int]:
        if self._type == "huggingface":
            return self._tok.encode(text, add_special_tokens=False)
        else:
            return self._tok.encode(text)

    def decode(self, ids: list[int]) -> str:
        if self._type == "huggingface":
            return self._tok.decode(ids, skip_special_tokens=False)
        else:
            return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def pad_id(self) -> int:
        return self._pad_id


# --- Multiprocessing tokenization helpers ---

_global_tokenizer: Tokenizer | None = None
_global_append_eos: bool = False


def init_tokenize_worker(tokenizer_path: str, tokenizer_type: str, append_eos: bool):
    """Initializer for multiprocessing pool workers."""
    global _global_tokenizer, _global_append_eos
    _global_tokenizer = Tokenizer(tokenizer_path, type=tokenizer_type)
    _global_append_eos = append_eos


def tokenize_doc(text: str) -> list[int]:
    """Tokenize a single document. Must be called after init_tokenize_worker."""
    assert _global_tokenizer is not None
    ids = _global_tokenizer.encode(text)
    if _global_append_eos:
        ids.append(_global_tokenizer.eos_id)
    return ids


# --- Sharded ArrayRecord writer ---

class ShardedArrayRecordWriter:
    """Write records across size-limited .arecord shards into an output directory.

    Each part produces a 1:1 pair of .arecord + .idx files, so the output
    directory can be loaded directly via MultiFileIndexedDataset.

    Output: output_dir/shard-{shard_idx:03d}-{part_idx:05d}.{arecord,idx}

    Args:
        output_dir: Output directory (created if needed).
        max_bytes: Maximum bytes per .arecord file before splitting.
        shard_idx: Worker/node index for Slurm-style parallel preprocessing.
    """

    def __init__(self, output_dir: str | Path, max_bytes: int, shard_idx: int = 0, prefix: str = "shard") -> None:
        from array_record.python.array_record_module import ArrayRecordWriter
        self._ArrayRecordWriter = ArrayRecordWriter
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_bytes
        self._shard_idx = shard_idx
        self._prefix = prefix
        self._part_idx = 0
        self._current_bytes = 0
        self._part_doc_lengths: list[list[int]] = [[]]
        self._paths: list[Path] = []
        self._writer = self._open_writer()

    def _shard_name(self, part_idx: int) -> str:
        return f"{self._prefix}-{self._shard_idx:03d}-{part_idx:05d}"

    def _open_writer(self):
        path = self._output_dir / f"{self._shard_name(self._part_idx)}.arecord"
        self._paths.append(path)
        return self._ArrayRecordWriter(str(path), "group_size:1")

    def write(self, record: bytes, doc_length: int) -> None:
        record_size = len(record)
        if self._current_bytes > 0 and self._current_bytes + record_size > self._max_bytes:
            self._writer.close()
            self._part_idx += 1
            self._writer = self._open_writer()
            self._part_doc_lengths.append([])
            self._current_bytes = 0
        self._writer.write(record)
        self._part_doc_lengths[self._part_idx].append(doc_length)
        self._current_bytes += record_size

    def close(self) -> int:
        """Close writer and write per-part .idx files. Returns total part count."""
        from megatext.data.indexed_dataset import write_index

        self._writer.close()
        total_parts = self._part_idx + 1
        for i in range(total_parts):
            idx_path = self._output_dir / f"{self._shard_name(i)}.idx"
            write_index(idx_path, np.array(self._part_doc_lengths[i], dtype=np.int32))
        return total_parts


# --- Document reading ---

def read_documents(file_paths: list[str], text_key: str = "text"):
    """Yield text documents from files (jsonl/parquet/gz/zstd)."""
    for fpath in file_paths:
        p = Path(fpath)
        suffix = "".join(p.suffixes).lower()

        if suffix.endswith(".parquet"):
            try:
                from pyarrow import parquet as pq
            except ImportError:
                raise ImportError("pyarrow required for parquet files: pip install pyarrow")
            table = pq.read_table(fpath, columns=[text_key])
            for text in table[text_key].to_pylist():
                if text:
                    yield text

        elif suffix.endswith(".gz"):
            with gzip.open(fpath, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        text = data.get(text_key, "")
                        if text:
                            yield text

        elif suffix.endswith(".zstd") or suffix.endswith(".zst"):
            import zstandard as zstd
            with open(fpath, "rb") as raw:
                dctx = zstd.ZstdDecompressor()
                reader = io.BufferedReader(dctx.stream_reader(raw))
                for line in reader:
                    line = line.decode("utf-8").strip()
                    if line:
                        data = json.loads(line)
                        text = data.get(text_key, "")
                        if text:
                            yield text

        else:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            text = data.get(text_key, "")
                        except json.JSONDecodeError:
                            text = line
                        if text:
                            yield text


def expand_globs(patterns: list[str]) -> list[str]:
    """Expand glob patterns into sorted file paths."""
    result = []
    for pattern in patterns:
        p = Path(pattern)
        if "*" in str(p) or "?" in str(p):
            result.extend(str(f) for f in sorted(p.parent.glob(p.name)))
        else:
            result.append(str(p))
    return result


# --- Shared CLI argument patterns ---

_SUPPORTED_EXTENSIONS = {"jsonl", "jsonl.gz", "json.gz", "jsonl.zstd", "jsonl.zst", "parquet"}


def add_input_args(parser):
    """Add input arguments (--input, --input-with-pattern, --extension)."""
    group = parser.add_argument_group("input data")
    group.add_argument("--input", type=str, required=True,
                       help="Input path. With --input-with-pattern: base directory for recursive search. "
                            "Without: comma-separated file paths or glob patterns.")
    group.add_argument("--input-with-pattern", action="store_true",
                       help="Recursively discover files under --input matching --extension")
    group.add_argument("--extension", type=str, default="jsonl",
                       help="Comma-separated extensions for --input-with-pattern (default: jsonl)")
    group.add_argument("--text-key", default="text", help="JSON key for text field")


def add_output_args(parser):
    """Add output arguments (--output-dir / --output-prefix, --prefix, --max-file-size)."""
    group = parser.add_argument_group("output data")
    exclusive = group.add_mutually_exclusive_group(required=True)
    exclusive.add_argument("--output-dir", type=str, default=None,
                           help="Output directory for shard files")
    exclusive.add_argument("--output-prefix", type=str, default=None,
                           help="Combined output path prefix, e.g. /path/to/dir/myprefix "
                                "(dirname becomes output-dir, basename becomes prefix)")
    group.add_argument("--prefix", type=str, default="shard",
                       help="Filename prefix for shards (default: shard). "
                            "Produces {prefix}-{shard_idx}-{part_idx}.{ext}")
    group.add_argument("--max-file-size", type=str, default="50G",
                       help="Maximum size per output shard (e.g. 50G, 500M)")


def add_sharding_args(parser):
    """Add Slurm-style sharding arguments."""
    group = parser.add_argument_group("sharding")
    group.add_argument("--num-shards", type=int, default=1,
                       help="Total number of parallel jobs")
    group.add_argument("--shard-idx", type=int, default=0,
                       help="This job's shard index (0-based)")


def resolve_input_files(args) -> list[str]:
    """Resolve input files from parsed args. Returns sorted file list."""
    if args.input_with_pattern:
        import glob as glob_mod
        files = []
        for ext in args.extension.split(","):
            ext = ext.strip()
            if ext not in _SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported extension: {ext!r}. Supported: {_SUPPORTED_EXTENSIONS}")
            files += sorted(glob_mod.glob(f"{args.input}/**/*.{ext}", recursive=True))
        if not files:
            raise ValueError(f"No files found matching extensions in {args.input}")
        return files
    else:
        # Comma-separated paths or glob patterns
        patterns = [p.strip() for p in args.input.split(",")]
        files = sorted(expand_globs(patterns))
        if not files:
            raise ValueError("No input files found")
        return files


def resolve_output(args) -> tuple[str, str]:
    """Resolve output-dir and prefix from parsed args. Returns (output_dir, prefix)."""
    if args.output_prefix is not None:
        output_dir = str(Path(args.output_prefix).parent) or "."
        basename = Path(args.output_prefix).name
        prefix = basename if basename else args.prefix
    else:
        output_dir = args.output_dir
        prefix = args.prefix
    return output_dir, prefix
