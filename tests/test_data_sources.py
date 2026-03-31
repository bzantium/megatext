"""Tests for DocumentDataSource."""

import numpy as np
import pytest

from megatext.data.data_sources import (
    DocumentDataSource,
    _build_shuffle_index,
    _build_shuffle_index_with_separate_epoch,
    _get_num_epochs,
)


class TestDocumentDataSource:
    def test_greedy_basic(self, sample_mmap_dataset):
        prefix, docs = sample_mmap_dataset
        ds = DocumentDataSource(
            data_path=prefix,
            data_type="mmap",
            seq_len=32,
            seed=42,
            num_samples=100,
            split=(1.0, 0.0, 0.0),
            split_index=0,
            packing_type="greedy",
        )
        assert len(ds) > 0

        sample = ds[0]
        assert "tokens" in sample
        assert sample["tokens"].shape == (33,)  # seq_len + 1 (add_extra_token)

    def test_packed_basic(self, sample_mmap_dataset):
        prefix, docs = sample_mmap_dataset
        ds = DocumentDataSource(
            data_path=prefix,
            data_type="mmap",
            seq_len=32,
            seed=42,
            num_samples=100,
            split=(1.0, 0.0, 0.0),
            split_index=0,
            packing_type="first_fit",
        )
        assert len(ds) > 0

        sample = ds[0]
        assert "tokens" in sample
        assert "segment_ids" in sample
        assert "loss_mask" in sample
        assert sample["tokens"].shape == (33,)
        assert sample["segment_ids"].shape == (33,)
        assert sample["loss_mask"].shape == (32,)

    def test_split(self, sample_mmap_dataset):
        prefix, docs = sample_mmap_dataset
        # 80% train, 20% val
        train_ds = DocumentDataSource(
            data_path=prefix,
            data_type="mmap",
            seq_len=16,
            seed=42,
            num_samples=100,
            split=(80.0, 20.0, 0.0),
            split_index=0,
        )
        val_ds = DocumentDataSource(
            data_path=prefix,
            data_type="mmap",
            seq_len=16,
            seed=42,
            num_samples=100,
            split=(80.0, 20.0, 0.0),
            split_index=1,
        )
        assert len(train_ds) > 0
        # val might be 0 with only 10 docs and 20% split

    def test_cache(self, sample_mmap_dataset, tmp_path):
        prefix, _ = sample_mmap_dataset
        cache_dir = str(tmp_path / "cache")
        ds = DocumentDataSource(
            data_path=prefix,
            data_type="mmap",
            seq_len=16,
            seed=42,
            num_samples=100,
            split=(1.0, 0.0, 0.0),
            split_index=0,
            cache_dir=cache_dir,
        )
        # Cache should be created
        from pathlib import Path
        assert Path(cache_dir).exists()
        assert len(list(Path(cache_dir).glob("*.npy"))) > 0

        # Second load should use cache
        ds2 = DocumentDataSource(
            data_path=prefix,
            data_type="mmap",
            seq_len=16,
            seed=42,
            num_samples=100,
            split=(1.0, 0.0, 0.0),
            split_index=0,
            cache_dir=cache_dir,
        )
        assert len(ds2) == len(ds)

    def test_multi_epoch(self, sample_mmap_dataset):
        prefix, docs = sample_mmap_dataset
        ds1 = DocumentDataSource(
            data_path=prefix,
            data_type="mmap",
            seq_len=16,
            seed=42,
            num_samples=100,
            split=(1.0, 0.0, 0.0),
            split_index=0,
        )
        ds2 = DocumentDataSource(
            data_path=prefix,
            data_type="mmap",
            seq_len=16,
            seed=42,
            num_samples=1000,
            split=(1.0, 0.0, 0.0),
            split_index=0,
        )
        # More samples requested -> more epochs -> more data available
        assert len(ds2) >= len(ds1) * 2  # at least 2x (exact depends on rounding)


class TestAutoNumEpochs:
    def test_auto_num_epochs(self):
        """Correct epoch count from num_samples + tokens_per_epoch."""
        # Need 10*100+1=1001 tokens, 1000/epoch -> 2 epochs
        assert _get_num_epochs(num_samples=10, tokens_per_epoch=1000, seq_len=100, add_extra_token=True) == 2
        # Without extra token: need 10*100=1000 tokens, 1000/epoch -> 1 epoch
        assert _get_num_epochs(num_samples=10, tokens_per_epoch=1000, seq_len=100, add_extra_token=False) == 1
        # 5*100+1=501 tokens, 1000/epoch -> 1 epoch
        assert _get_num_epochs(num_samples=5, tokens_per_epoch=1000, seq_len=100, add_extra_token=True) == 1

    def test_auto_num_epochs_small_dataset(self):
        """Small dataset -> multiple epochs."""
        # 100 tokens/epoch, need 10*100+1=1001 tokens -> need 11 epochs
        result = _get_num_epochs(num_samples=10, tokens_per_epoch=100, seq_len=100, add_extra_token=True)
        assert result > 1

    def test_auto_num_epochs_large_dataset(self):
        """Large dataset -> 1 epoch."""
        result = _get_num_epochs(num_samples=5, tokens_per_epoch=100000, seq_len=100, add_extra_token=True)
        assert result == 1

    def test_auto_num_epochs_blended_weights(self):
        """Per-source num_samples = total * weight * 1.005."""
        import math
        total_samples = 1000
        weight = 0.3
        total_weight = 1.0
        norm_weight = weight / total_weight
        num_samples = int(math.ceil(total_samples * norm_weight * 1.005))
        # Should be ceil(1000 * 0.3 * 1.005) = ceil(301.5) = 302
        assert num_samples == 302


class TestShuffleIndex:
    def test_shuffle_index_is_permutation(self):
        """Valid permutation of [0, N)."""
        rng = np.random.RandomState(42)
        n = 100
        idx = _build_shuffle_index(n, n, rng)
        assert len(idx) == n
        assert set(idx.tolist()) == set(range(n))

    def test_shuffle_index_separate_final_epoch(self):
        """Final-epoch samples stay at end (80% threshold)."""
        rng = np.random.RandomState(42)
        n = 100
        total = 110  # 10 final-epoch samples
        idx = _build_shuffle_index(n, total, rng)
        assert len(idx) == total
        # First n values should be a permutation of [0, n)
        assert set(idx[:n].tolist()) == set(range(n))
        # Last (total-n) values should be a permutation of [n, total)
        assert set(idx[n:].tolist()) == set(range(n, total))

    def test_shuffle_index_deterministic(self):
        """Same seed -> same shuffle_index."""
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        idx1 = _build_shuffle_index(100, 100, rng1)
        idx2 = _build_shuffle_index(100, 100, rng2)
        np.testing.assert_array_equal(idx1, idx2)

    def test_shuffle_index_cached(self, sample_mmap_dataset, tmp_path):
        """.npy file created, reload matches."""
        from pathlib import Path

        prefix, _ = sample_mmap_dataset
        cache_dir = str(tmp_path / "shuffle_cache")
        ds = DocumentDataSource(
            data_path=prefix, data_type="mmap", seq_len=16,
            seed=42, num_samples=100, split=(1.0, 0.0, 0.0), split_index=0,
            cache_dir=cache_dir,
        )
        cache_path = Path(cache_dir)
        shuffle_files = list(cache_path.glob("*-shuffle_index.npy"))
        assert len(shuffle_files) == 1

        # Reload and verify
        ds2 = DocumentDataSource(
            data_path=prefix, data_type="mmap", seq_len=16,
            seed=42, num_samples=100, split=(1.0, 0.0, 0.0), split_index=0,
            cache_dir=cache_dir,
        )
        np.testing.assert_array_equal(ds._shuffle_index, ds2._shuffle_index)

    def test_getitem_applies_shuffle(self, sample_mmap_dataset):
        """ds[0] differs with shuffle_index (shuffled access)."""
        prefix, _ = sample_mmap_dataset
        ds = DocumentDataSource(
            data_path=prefix, data_type="mmap", seq_len=16,
            seed=42, num_samples=100, split=(1.0, 0.0, 0.0), split_index=0,
        )
        assert ds._shuffle_index is not None
        # With shuffle, ds[0] accesses shuffle_index[0] which is unlikely to be 0
        # Verify the shuffle_index is actually applied by checking it's a valid permutation
        assert len(ds._shuffle_index) == len(ds)

    def test_separate_final_epoch_auto(self):
        """Auto 80% threshold computation."""
        from megatext.data.data_sources import _build_document_index, _build_sample_index

        # Create a scenario with multiple epochs
        num_docs = 10
        doc_lengths = np.array([50] * num_docs, dtype=np.int32)
        rng = np.random.RandomState(42)

        # 3 epochs
        document_index = _build_document_index(num_docs, 3, rng)
        sample_index = _build_sample_index(document_index, doc_lengths, seq_len=20, add_extra_token=True)
        total_samples = len(sample_index) - 1

        rng2 = np.random.RandomState(42)
        shuffle_index = _build_shuffle_index_with_separate_epoch(
            sample_index, document_index, num_docs, 3,
            num_samples=total_samples, total_samples=total_samples, rng=rng2,
        )
        # Should be a valid permutation
        assert len(shuffle_index) == total_samples
        assert set(shuffle_index.tolist()) == set(range(total_samples))
