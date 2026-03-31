"""Tests for MMapIndexedDataset, ArrayRecordDocDataset, and MultiFileIndexedDataset."""

import numpy as np
import pytest

from megatext.data.indexed_dataset import (
    MMapIndexedDataset,
    MultiFileIndexedDataset,
    make_arecord_dataset,
    make_mmap_dataset,
    read_index,
    write_index,
)


class TestMMapIndexedDataset:
    def test_len(self, sample_mmap_dataset):
        prefix, docs = sample_mmap_dataset
        dataset = MMapIndexedDataset(prefix)
        assert len(dataset) == len(docs)

    def test_getitem(self, sample_mmap_dataset):
        prefix, docs = sample_mmap_dataset
        dataset = MMapIndexedDataset(prefix)
        for i, expected in enumerate(docs):
            actual = dataset[i]
            np.testing.assert_array_equal(actual, expected)

    def test_negative_index(self, sample_mmap_dataset):
        prefix, docs = sample_mmap_dataset
        dataset = MMapIndexedDataset(prefix)
        np.testing.assert_array_equal(dataset[-1], docs[-1])

    def test_out_of_range(self, sample_mmap_dataset):
        prefix, _ = sample_mmap_dataset
        dataset = MMapIndexedDataset(prefix)
        with pytest.raises(IndexError):
            dataset[len(dataset)]

    def test_get_slice(self, sample_mmap_dataset):
        prefix, docs = sample_mmap_dataset
        dataset = MMapIndexedDataset(prefix)
        # Get first 5 tokens of doc 0
        result = dataset.get(0, offset=0, length=5)
        np.testing.assert_array_equal(result, docs[0][:5])

    def test_get_with_offset(self, sample_mmap_dataset):
        prefix, docs = sample_mmap_dataset
        dataset = MMapIndexedDataset(prefix)
        result = dataset.get(0, offset=3, length=5)
        np.testing.assert_array_equal(result, docs[0][3:8])

    def test_doc_lengths(self, sample_mmap_dataset):
        prefix, docs = sample_mmap_dataset
        dataset = MMapIndexedDataset(prefix)
        expected = np.array([len(d) for d in docs], dtype=np.int32)
        np.testing.assert_array_equal(dataset.doc_lengths, expected)

    def test_missing_idx(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MMapIndexedDataset(str(tmp_path / "nonexistent"))


class TestArrayRecordDocDataset:
    def test_len(self, sample_arecord_dataset):
        prefix, docs = sample_arecord_dataset
        from megatext.data.indexed_dataset import ArrayRecordDocDataset
        dataset = ArrayRecordDocDataset(prefix)
        assert len(dataset) == len(docs)

    def test_getitem(self, sample_arecord_dataset):
        prefix, docs = sample_arecord_dataset
        from megatext.data.indexed_dataset import ArrayRecordDocDataset
        dataset = ArrayRecordDocDataset(prefix)
        for i, expected in enumerate(docs):
            actual = dataset[i]
            np.testing.assert_array_equal(actual, expected)

    def test_doc_lengths(self, sample_arecord_dataset):
        prefix, docs = sample_arecord_dataset
        from megatext.data.indexed_dataset import ArrayRecordDocDataset
        dataset = ArrayRecordDocDataset(prefix)
        expected = np.array([len(d) for d in docs], dtype=np.int32)
        np.testing.assert_array_equal(dataset.doc_lengths, expected)

    def test_missing_idx(self, tmp_path):
        from megatext.data.indexed_dataset import ArrayRecordDocDataset
        with pytest.raises(FileNotFoundError):
            ArrayRecordDocDataset(str(tmp_path / "nonexistent"))


class TestMultiFileIndexedDataset:
    def test_len_mmap(self, sample_mmap_directory):
        dir_path, all_docs = sample_mmap_directory
        dataset = MultiFileIndexedDataset(dir_path, MMapIndexedDataset, ".bin")
        assert len(dataset) == len(all_docs)

    def test_getitem_mmap(self, sample_mmap_directory):
        dir_path, all_docs = sample_mmap_directory
        dataset = MultiFileIndexedDataset(dir_path, MMapIndexedDataset, ".bin")
        for i, expected in enumerate(all_docs):
            actual = dataset[i]
            np.testing.assert_array_equal(actual, expected)

    def test_doc_lengths_mmap(self, sample_mmap_directory):
        dir_path, all_docs = sample_mmap_directory
        dataset = MultiFileIndexedDataset(dir_path, MMapIndexedDataset, ".bin")
        expected = np.array([len(d) for d in all_docs], dtype=np.int32)
        np.testing.assert_array_equal(dataset.doc_lengths, expected)

    def test_len_arecord(self, sample_arecord_directory):
        dir_path, all_docs = sample_arecord_directory
        from megatext.data.indexed_dataset import ArrayRecordDocDataset
        dataset = MultiFileIndexedDataset(dir_path, ArrayRecordDocDataset, ".arecord")
        assert len(dataset) == len(all_docs)

    def test_getitem_arecord(self, sample_arecord_directory):
        dir_path, all_docs = sample_arecord_directory
        from megatext.data.indexed_dataset import ArrayRecordDocDataset
        dataset = MultiFileIndexedDataset(dir_path, ArrayRecordDocDataset, ".arecord")
        for i, expected in enumerate(all_docs):
            actual = dataset[i]
            np.testing.assert_array_equal(actual, expected)

    def test_doc_lengths_arecord(self, sample_arecord_directory):
        dir_path, all_docs = sample_arecord_directory
        from megatext.data.indexed_dataset import ArrayRecordDocDataset
        dataset = MultiFileIndexedDataset(dir_path, ArrayRecordDocDataset, ".arecord")
        expected = np.array([len(d) for d in all_docs], dtype=np.int32)
        np.testing.assert_array_equal(dataset.doc_lengths, expected)

    def test_empty_directory(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No .idx files"):
            MultiFileIndexedDataset(str(empty_dir), MMapIndexedDataset, ".bin")

    def test_not_a_directory(self, tmp_path):
        fake_file = tmp_path / "not_a_dir"
        fake_file.touch()
        with pytest.raises(FileNotFoundError, match="Not a directory"):
            MultiFileIndexedDataset(str(fake_file), MMapIndexedDataset, ".bin")

    def test_negative_index_mmap(self, sample_mmap_directory):
        dir_path, all_docs = sample_mmap_directory
        dataset = MultiFileIndexedDataset(dir_path, MMapIndexedDataset, ".bin")
        np.testing.assert_array_equal(dataset[-1], all_docs[-1])

    def test_get_mmap(self, sample_mmap_directory):
        dir_path, all_docs = sample_mmap_directory
        dataset = MultiFileIndexedDataset(dir_path, MMapIndexedDataset, ".bin")
        result = dataset.get(0, offset=0, length=5)
        np.testing.assert_array_equal(result, all_docs[0][:5])

    def test_missing_data_file(self, tmp_path):
        """idx without matching data file raises FileNotFoundError."""
        from megatext.data.indexed_dataset import write_index
        data_dir = tmp_path / "orphan"
        data_dir.mkdir()
        write_index(data_dir / "orphan.idx", np.array([10, 20], dtype=np.int32))
        with pytest.raises(FileNotFoundError, match="No matching data file"):
            MultiFileIndexedDataset(str(data_dir), MMapIndexedDataset, ".bin")


class TestFactoryFunctions:
    def test_make_mmap_single(self, sample_mmap_dataset):
        prefix, docs = sample_mmap_dataset
        dataset = make_mmap_dataset(prefix)
        assert isinstance(dataset, MMapIndexedDataset)
        assert len(dataset) == len(docs)

    def test_make_mmap_directory(self, sample_mmap_directory):
        dir_path, all_docs = sample_mmap_directory
        dataset = make_mmap_dataset(dir_path)
        assert isinstance(dataset, MultiFileIndexedDataset)
        assert len(dataset) == len(all_docs)

    def test_make_mmap_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            make_mmap_dataset(str(tmp_path / "nonexistent"))

    def test_make_arecord_single(self, sample_arecord_dataset):
        prefix, docs = sample_arecord_dataset
        from megatext.data.indexed_dataset import ArrayRecordDocDataset
        dataset = make_arecord_dataset(prefix)
        assert isinstance(dataset, ArrayRecordDocDataset)
        assert len(dataset) == len(docs)

    def test_make_arecord_directory(self, sample_arecord_directory):
        dir_path, all_docs = sample_arecord_directory
        dataset = make_arecord_dataset(dir_path)
        assert isinstance(dataset, MultiFileIndexedDataset)
        assert len(dataset) == len(all_docs)

    def test_make_arecord_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            make_arecord_dataset(str(tmp_path / "nonexistent"))


class TestWriteReadIndex:
    def test_roundtrip(self, tmp_path):
        lengths = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        path = tmp_path / "test.idx"
        write_index(path, lengths)
        loaded = read_index(path)
        np.testing.assert_array_equal(loaded, lengths)
