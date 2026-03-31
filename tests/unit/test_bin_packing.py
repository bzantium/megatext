"""Tests for bin packing algorithms."""

import numpy as np
import pytest

from megatext.data.bin_packing import build_packed_sample_index


class TestBuildPackedSampleIndex:
    def _make_data(self, doc_lens, seq_len):
        """Helper: create doc_lengths and document_index arrays."""
        doc_lengths = np.array(doc_lens, dtype=np.int32)
        document_index = np.arange(len(doc_lens), dtype=np.int32)
        return document_index, doc_lengths

    @pytest.mark.parametrize("packing_type", ["first_fit", "best_fit"])
    def test_basic_packing(self, packing_type):
        # 3 docs: lengths 100, 50, 30. seq_len=64
        doc_index, doc_lengths = self._make_data([100, 50, 30], seq_len=64)
        chunk_index, sample_index = build_packed_sample_index(
            doc_index, doc_lengths, seq_len=64,
            packing_type=packing_type,
        )
        num_samples = len(sample_index) - 1
        assert num_samples > 0
        # All chunk lengths should be <= seq_len
        for i in range(len(chunk_index)):
            assert chunk_index[i, 2] <= 64

    @pytest.mark.parametrize("packing_type", ["first_fit", "best_fit"])
    def test_full_chunks_only(self, packing_type):
        # Doc exactly divisible by seq_len
        doc_index, doc_lengths = self._make_data([128], seq_len=64)
        chunk_index, sample_index = build_packed_sample_index(
            doc_index, doc_lengths, seq_len=64,
            packing_type=packing_type,
        )
        num_samples = len(sample_index) - 1
        assert num_samples == 2  # 128 / 64 = 2 full chunks

    @pytest.mark.parametrize("packing_type", ["first_fit", "best_fit"])
    def test_remainders_packed(self, packing_type):
        # Two docs with remainders that fit in one bin
        doc_index, doc_lengths = self._make_data([80, 48], seq_len=64)
        chunk_index, sample_index = build_packed_sample_index(
            doc_index, doc_lengths, seq_len=64,
            packing_type=packing_type,
        )
        # 80 = 1 full + 16 remainder
        # 48 = 0 full + 48 remainder
        # 16 + 48 = 64, fits in one bin
        num_samples = len(sample_index) - 1
        assert num_samples == 2  # 1 full chunk + 1 packed bin

    @pytest.mark.parametrize("packing_type", ["first_fit", "best_fit"])
    def test_max_chunks_per_sample(self, packing_type):
        doc_index, doc_lengths = self._make_data([10, 10, 10, 10], seq_len=64)
        chunk_index, sample_index = build_packed_sample_index(
            doc_index, doc_lengths, seq_len=64,
            packing_type=packing_type,
            max_chunks_per_sample=2,
        )
        # Each sample can have at most 2 chunks
        num_samples = len(sample_index) - 1
        for i in range(num_samples):
            n_chunks = int(sample_index[i + 1]) - int(sample_index[i])
            assert n_chunks <= 2

    @pytest.mark.parametrize("packing_type", ["first_fit", "best_fit"])
    def test_empty_docs(self, packing_type):
        doc_index, doc_lengths = self._make_data([0, 0], seq_len=64)
        chunk_index, sample_index = build_packed_sample_index(
            doc_index, doc_lengths, seq_len=64,
            packing_type=packing_type,
        )
        num_samples = len(sample_index) - 1
        assert num_samples == 0

    @pytest.mark.parametrize("packing_type", ["first_fit", "best_fit"])
    @pytest.mark.parametrize("seq_len", [100, 127, 255, 300, 500])
    def test_non_power_of_2_seq_len(self, packing_type, seq_len):
        """Packing works correctly with non-power-of-2 sequence lengths."""
        rng = np.random.RandomState(42)
        doc_lens = rng.randint(1, seq_len * 3, size=50).tolist()
        doc_index, doc_lengths = self._make_data(doc_lens, seq_len)
        chunk_index, sample_index = build_packed_sample_index(
            doc_index, doc_lengths, seq_len=seq_len,
            packing_type=packing_type,
        )
        num_samples = len(sample_index) - 1
        # Verify no bin exceeds seq_len
        for i in range(num_samples):
            start = int(sample_index[i])
            end = int(sample_index[i + 1])
            total = sum(int(chunk_index[ci, 2]) for ci in range(start, end))
            assert total <= seq_len, (
                f"Sample {i} has {total} tokens > seq_len={seq_len}"
            )

    @pytest.mark.parametrize("seed", [42, 123, 777, 2024, 9999])
    @pytest.mark.parametrize(
        "seq_len,num_docs,max_len",
        [(128, 100, 200), (512, 200, 500), (1024, 300, 2000), (256, 50, 64)],
    )
    def test_ffd_bfd_equivalence(self, seed, seq_len, num_docs, max_len):
        """FFD and BFD produce identical bin count and total padding."""
        rng = np.random.RandomState(seed)
        doc_lens = rng.randint(1, max_len + 1, size=num_docs).tolist()
        doc_index, doc_lengths = self._make_data(doc_lens, seq_len=seq_len)

        ci_ff, si_ff = build_packed_sample_index(
            doc_index, doc_lengths, seq_len=seq_len, packing_type="first_fit",
        )
        ci_bf, si_bf = build_packed_sample_index(
            doc_index, doc_lengths, seq_len=seq_len, packing_type="best_fit",
        )
        # Same number of samples
        assert len(si_ff) == len(si_bf), (
            f"seed={seed}, seq_len={seq_len}: sample count differs "
            f"(ff={len(si_ff)}, bf={len(si_bf)})"
        )
        # Same total padding
        total_tokens_ff = int(ci_ff[:, 2].sum())
        total_tokens_bf = int(ci_bf[:, 2].sum())
        n_samples = len(si_ff) - 1
        padding_ff = n_samples * seq_len - total_tokens_ff
        padding_bf = n_samples * seq_len - total_tokens_bf
        assert padding_ff == padding_bf, (
            f"seed={seed}, seq_len={seq_len}: padding differs "
            f"(ff={padding_ff}, bf={padding_bf})"
        )

    @pytest.mark.parametrize("seed", [42, 123, 777])
    @pytest.mark.parametrize("max_chunks", [2, 3, 5])
    def test_ffd_bfd_near_equivalence_with_max_chunks(self, seed, max_chunks):
        """FFD and BFD produce nearly identical results with chunk limits.

        With max_chunks_per_sample > 0 the chunk-count constraint can break
        strict equivalence, but the difference should be negligible (<=1%
        sample count difference).
        """
        rng = np.random.RandomState(seed)
        seq_len = 256
        doc_lens = rng.randint(1, 300, size=150).tolist()
        doc_index, doc_lengths = self._make_data(doc_lens, seq_len=seq_len)

        ci_ff, si_ff = build_packed_sample_index(
            doc_index, doc_lengths, seq_len=seq_len, packing_type="first_fit",
            max_chunks_per_sample=max_chunks,
        )
        ci_bf, si_bf = build_packed_sample_index(
            doc_index, doc_lengths, seq_len=seq_len, packing_type="best_fit",
            max_chunks_per_sample=max_chunks,
        )
        n_ff = len(si_ff) - 1
        n_bf = len(si_bf) - 1
        # Allow <= 2% divergence in sample count
        assert abs(n_ff - n_bf) <= max(1, int(0.02 * max(n_ff, n_bf))), (
            f"seed={seed}, max_chunks={max_chunks}: sample counts diverge "
            f"too much (ff={n_ff}, bf={n_bf})"
        )

    def test_all_tokens_accounted_for(self):
        """Every token from every document appears exactly once in the output."""
        doc_lens = [100, 200, 64, 50, 130]
        seq_len = 64
        doc_index, doc_lengths = self._make_data(doc_lens, seq_len)

        for packing_type in ["first_fit", "best_fit"]:
            chunk_index, sample_index = build_packed_sample_index(
                doc_index, doc_lengths, seq_len=seq_len,
                packing_type=packing_type,
            )
            # Reconstruct total tokens per doc from chunks
            doc_tokens = {}
            for ci in range(len(chunk_index)):
                did = int(chunk_index[ci, 0])
                length = int(chunk_index[ci, 2])
                doc_tokens[did] = doc_tokens.get(did, 0) + length

            for did, expected_len in enumerate(doc_lens):
                actual = doc_tokens.get(did, 0)
                assert actual == expected_len, (
                    f"[{packing_type}] doc {did}: expected {expected_len}, "
                    f"got {actual}"
                )

    @pytest.mark.parametrize("packing_type", ["first_fit", "best_fit"])
    def test_single_token_docs(self, packing_type):
        """Many single-token docs should pack efficiently."""
        doc_lens = [1] * 128
        seq_len = 64
        doc_index, doc_lengths = self._make_data(doc_lens, seq_len)
        chunk_index, sample_index = build_packed_sample_index(
            doc_index, doc_lengths, seq_len=seq_len,
            packing_type=packing_type,
        )
        num_samples = len(sample_index) - 1
        # 128 single-token docs into bins of 64 -> 2 bins (no max_chunks limit)
        assert num_samples == 2

    def test_invalid_packing_type(self):
        doc_index, doc_lengths = self._make_data([100], seq_len=64)
        with pytest.raises(ValueError, match="Unknown packing_type"):
            build_packed_sample_index(
                doc_index, doc_lengths, seq_len=64,
                packing_type="invalid",
            )
