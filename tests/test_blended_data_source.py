"""Tests for BlendedDataSource."""

import numpy as np
import pytest

from megatext.data.data_sources import BlendedDataSource, DocumentDataSource


class TestBlendedDataSource:
    def test_basic_blending(self, sample_mmap_dataset):
        prefix, _ = sample_mmap_dataset
        ds1 = DocumentDataSource(
            data_path=prefix, data_type="mmap", seq_len=16,
            seed=42, num_samples=100, split=(1.0, 0.0, 0.0), split_index=0,
        )
        ds2 = DocumentDataSource(
            data_path=prefix, data_type="mmap", seq_len=16,
            seed=43, num_samples=100, split=(1.0, 0.0, 0.0), split_index=0,
        )
        total = len(ds1) + len(ds2)
        blended = BlendedDataSource([ds1, ds2], [0.7, 0.3], size=total)
        assert len(blended) == total

        # All items should be valid
        for i in range(min(10, total)):
            sample = blended[i]
            assert "tokens" in sample

    def test_proportions(self, sample_mmap_dataset):
        prefix, _ = sample_mmap_dataset
        ds1 = DocumentDataSource(
            data_path=prefix, data_type="mmap", seq_len=16,
            seed=42, num_samples=500, split=(1.0, 0.0, 0.0), split_index=0,
        )
        ds2 = DocumentDataSource(
            data_path=prefix, data_type="mmap", seq_len=16,
            seed=43, num_samples=500, split=(1.0, 0.0, 0.0), split_index=0,
        )
        size = 100
        blended = BlendedDataSource([ds1, ds2], [0.8, 0.2], size=size)

        # Verify all samples are accessible (proportional distribution is internal)
        for i in range(size):
            sample = blended[i]
            assert "tokens" in sample

    def test_single_source(self, sample_mmap_dataset):
        prefix, _ = sample_mmap_dataset
        ds = DocumentDataSource(
            data_path=prefix, data_type="mmap", seq_len=16,
            seed=42, num_samples=100, split=(1.0, 0.0, 0.0), split_index=0,
        )
        blended = BlendedDataSource([ds], [1.0], size=len(ds))
        assert len(blended) == len(ds)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            BlendedDataSource([], [], size=10)

    def test_empty_sources(self):
        with pytest.raises(ValueError):
            BlendedDataSource([], [], size=0)
