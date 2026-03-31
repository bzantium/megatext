"""Tests for FixedArecordDataSource and its pipeline integration."""

from __future__ import annotations

import numpy as np
import pytest

import grain.python as grain


# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def fixed_arecord_dir(tmp_path):
    """Create a directory with 2 sharded arecord files, each 20 records of 17 int32 tokens.

    seq_len=16, so each record is seq_len+1 = 17 tokens.
    Returns (dir_path, seq_len, all_tokens_list).
    """
    pytest.importorskip("array_record")
    from array_record.python.array_record_module import ArrayRecordWriter

    seq_len = 16
    num_tokens = seq_len + 1
    rng = np.random.RandomState(42)
    all_tokens = []

    for shard_idx in range(2):
        path = str(tmp_path / f"shard-{shard_idx:03d}.arecord")
        writer = ArrayRecordWriter(path, "group_size:1")
        for _ in range(20):
            tokens = rng.randint(0, 10000, size=num_tokens, dtype=np.int32)
            all_tokens.append(tokens)
            writer.write(tokens.tobytes())
        writer.close()

    return str(tmp_path), seq_len, all_tokens


@pytest.fixture
def fixed_arecord_single(tmp_path):
    """Single shard with 10 records."""
    pytest.importorskip("array_record")
    from array_record.python.array_record_module import ArrayRecordWriter

    seq_len = 8
    num_tokens = seq_len + 1
    rng = np.random.RandomState(99)
    all_tokens = []

    path = str(tmp_path / "data.arecord")
    writer = ArrayRecordWriter(path, "group_size:1")
    for _ in range(10):
        tokens = rng.randint(0, 5000, size=num_tokens, dtype=np.int32)
        all_tokens.append(tokens)
        writer.write(tokens.tobytes())
    writer.close()

    return str(tmp_path), seq_len, all_tokens


# -- FixedArecordDataSource unit tests ---------------------------------------


class TestFixedArecordDataSource:
    """Unit tests for FixedArecordDataSource."""

    def test_len(self, fixed_arecord_dir):
        from megatext.data.data_sources import FixedArecordDataSource

        data_path, seq_len, all_tokens = fixed_arecord_dir
        ds = FixedArecordDataSource(data_path=data_path, seq_len=seq_len)
        assert len(ds) == len(all_tokens)

    def test_getitem_returns_tokens(self, fixed_arecord_dir):
        from megatext.data.data_sources import FixedArecordDataSource

        data_path, seq_len, all_tokens = fixed_arecord_dir
        ds = FixedArecordDataSource(data_path=data_path, seq_len=seq_len)

        for i in range(len(ds)):
            sample = ds[i]
            assert "tokens" in sample
            assert sample["tokens"].dtype == np.int32
            assert len(sample["tokens"]) == seq_len + 1
            np.testing.assert_array_equal(sample["tokens"], all_tokens[i])

    def test_random_access(self, fixed_arecord_dir):
        """Access in non-sequential order should work."""
        from megatext.data.data_sources import FixedArecordDataSource

        data_path, seq_len, all_tokens = fixed_arecord_dir
        ds = FixedArecordDataSource(data_path=data_path, seq_len=seq_len)

        indices = [5, 0, 39, 20, 10]
        for i in indices:
            np.testing.assert_array_equal(ds[i]["tokens"], all_tokens[i])

    def test_is_random_access_data_source(self, fixed_arecord_dir):
        from megatext.data.data_sources import FixedArecordDataSource

        data_path, seq_len, _ = fixed_arecord_dir
        ds = FixedArecordDataSource(data_path=data_path, seq_len=seq_len)
        assert isinstance(ds, grain.RandomAccessDataSource)

    def test_no_arecord_files_raises(self, tmp_path):
        from megatext.data.data_sources import FixedArecordDataSource

        with pytest.raises(FileNotFoundError, match="No .arecord files"):
            FixedArecordDataSource(data_path=str(tmp_path), seq_len=16)

    def test_wrong_record_size_raises(self, fixed_arecord_dir):
        """Passing wrong seq_len should raise ValueError."""
        from megatext.data.data_sources import FixedArecordDataSource

        data_path, seq_len, _ = fixed_arecord_dir
        with pytest.raises(ValueError, match="record size mismatch"):
            FixedArecordDataSource(data_path=data_path, seq_len=seq_len + 10)

    def test_single_shard(self, fixed_arecord_single):
        from megatext.data.data_sources import FixedArecordDataSource

        data_path, seq_len, all_tokens = fixed_arecord_single
        ds = FixedArecordDataSource(data_path=data_path, seq_len=seq_len)
        assert len(ds) == 10
        np.testing.assert_array_equal(ds[0]["tokens"], all_tokens[0])


# -- Grain pipeline integration ----------------------------------------------


class TestFixedArecordGrainPipeline:
    """Test FixedArecordDataSource works with Grain's MapDataset pipeline."""

    def test_map_dataset_pipeline(self, fixed_arecord_dir):
        """Full pipeline: source -> shard -> repeat -> transform -> batch."""
        from megatext.data.data_sources import FixedArecordDataSource
        from megatext.data.data_processing import FormatForMegatext

        data_path, seq_len, _ = fixed_arecord_dir
        ds = FixedArecordDataSource(data_path=data_path, seq_len=seq_len)

        dataset = grain.MapDataset.source(ds)
        dataset = dataset.map(FormatForMegatext(seq_len=seq_len, add_extra_token=True))
        dataset = dataset.batch(4, drop_remainder=True)

        batch = dataset[0]
        assert batch["inputs"].shape == (4, seq_len)
        assert batch["targets"].shape == (4, seq_len)
        assert batch["inputs_position"].shape == (4, seq_len)
        assert batch["targets_position"].shape == (4, seq_len)
        assert batch["inputs_segmentation"].shape == (4, seq_len)
        assert batch["targets_segmentation"].shape == (4, seq_len)

    def test_format_input_target_shift(self, fixed_arecord_dir):
        """Verify inputs = tokens[:seq_len], targets = tokens[1:seq_len+1]."""
        from megatext.data.data_sources import FixedArecordDataSource
        from megatext.data.data_processing import FormatForMegatext

        data_path, seq_len, all_tokens = fixed_arecord_dir
        ds = FixedArecordDataSource(data_path=data_path, seq_len=seq_len)

        fmt = FormatForMegatext(seq_len=seq_len, add_extra_token=True)
        sample = ds[0]
        formatted = fmt.map(sample)

        np.testing.assert_array_equal(formatted["inputs"], all_tokens[0][:seq_len])
        np.testing.assert_array_equal(formatted["targets"], all_tokens[0][1:seq_len + 1])

    def test_segmentation_all_ones(self, fixed_arecord_dir):
        """Without packing, segmentation should be all ones."""
        from megatext.data.data_sources import FixedArecordDataSource
        from megatext.data.data_processing import FormatForMegatext

        data_path, seq_len, _ = fixed_arecord_dir
        ds = FixedArecordDataSource(data_path=data_path, seq_len=seq_len)

        fmt = FormatForMegatext(seq_len=seq_len, add_extra_token=True)
        formatted = fmt.map(ds[0])

        np.testing.assert_array_equal(
            formatted["inputs_segmentation"], np.ones(seq_len, dtype=np.int32)
        )
        np.testing.assert_array_equal(
            formatted["targets_segmentation"], np.ones(seq_len, dtype=np.int32)
        )

    def test_positions_sequential(self, fixed_arecord_dir):
        """Without packing, positions should be 0, 1, 2, ..., seq_len-1."""
        from megatext.data.data_sources import FixedArecordDataSource
        from megatext.data.data_processing import FormatForMegatext

        data_path, seq_len, _ = fixed_arecord_dir
        ds = FixedArecordDataSource(data_path=data_path, seq_len=seq_len)

        fmt = FormatForMegatext(seq_len=seq_len, add_extra_token=True)
        formatted = fmt.map(ds[0])

        expected_pos = np.arange(seq_len, dtype=np.int32)
        np.testing.assert_array_equal(formatted["inputs_position"], expected_pos)
        np.testing.assert_array_equal(formatted["targets_position"], expected_pos)


# -- Multi-host sharding simulation -----------------------------------------


class TestFixedArecordMultiHostSharding:
    """Simulate multi-host sharding via Grain's index slicing."""

    def test_two_process_sharding_covers_all_data(self, fixed_arecord_dir):
        """Two processes should collectively cover all records without overlap."""
        from megatext.data.data_sources import FixedArecordDataSource

        data_path, seq_len, all_tokens = fixed_arecord_dir
        ds = FixedArecordDataSource(data_path=data_path, seq_len=seq_len)
        total = len(ds)

        dataset = grain.MapDataset.source(ds)
        shard_0 = dataset[0::2]  # process 0
        shard_1 = dataset[1::2]  # process 1

        # Each shard gets half
        assert len(shard_0) == total // 2
        assert len(shard_1) == total // 2
        assert len(shard_0) + len(shard_1) == total

        # No overlap: collect all tokens from both shards
        tokens_0 = {tuple(shard_0[i]["tokens"].tolist()) for i in range(len(shard_0))}
        tokens_1 = {tuple(shard_1[i]["tokens"].tolist()) for i in range(len(shard_1))}
        assert len(tokens_0 & tokens_1) == 0, "Shards overlap"
        assert len(tokens_0 | tokens_1) == total

    def test_four_process_sharding(self, fixed_arecord_dir):
        """Four processes should each get 1/4 of data."""
        from megatext.data.data_sources import FixedArecordDataSource

        data_path, seq_len, _ = fixed_arecord_dir
        ds = FixedArecordDataSource(data_path=data_path, seq_len=seq_len)
        total = len(ds)

        dataset = grain.MapDataset.source(ds)
        shards = [dataset[i::4] for i in range(4)]

        total_from_shards = sum(len(s) for s in shards)
        assert total_from_shards == total

        # No overlap
        all_tokens = []
        for shard in shards:
            for i in range(len(shard)):
                all_tokens.append(tuple(shard[i]["tokens"].tolist()))
        assert len(set(all_tokens)) == total

    def test_sharded_batching(self, fixed_arecord_dir):
        """Each process shard should produce correct batch shapes."""
        from megatext.data.data_sources import FixedArecordDataSource
        from megatext.data.data_processing import FormatForMegatext

        data_path, seq_len, _ = fixed_arecord_dir
        ds = FixedArecordDataSource(data_path=data_path, seq_len=seq_len)

        num_processes = 2
        batch_size = 4  # per-process batch

        for proc_idx in range(num_processes):
            dataset = grain.MapDataset.source(ds)
            dataset = dataset[proc_idx::num_processes]
            dataset = dataset.map(FormatForMegatext(seq_len=seq_len, add_extra_token=True))
            dataset = dataset.batch(batch_size, drop_remainder=True)

            batch = dataset[0]
            assert batch["inputs"].shape == (batch_size, seq_len)
            assert batch["targets"].shape == (batch_size, seq_len)

    def test_repeat_produces_infinite_data(self, fixed_arecord_single):
        """Repeat(None) should allow reading beyond dataset size."""
        from megatext.data.data_sources import FixedArecordDataSource

        data_path, seq_len, all_tokens = fixed_arecord_single
        ds = FixedArecordDataSource(data_path=data_path, seq_len=seq_len)
        total = len(ds)  # 10

        dataset = grain.MapDataset.source(ds)
        dataset = dataset.repeat(None)

        # Should be able to access well beyond the dataset size
        for i in range(total * 3):
            sample = dataset[i]
            expected = all_tokens[i % total]
            np.testing.assert_array_equal(sample["tokens"], expected)


# -- Blended source integration ----------------------------------------------


class TestFixedArecordBlended:
    """Test FixedArecordDataSource works with BlendedDataSource."""

    def test_blended_two_sources(self, tmp_path):
        """BlendedDataSource should work with multiple FixedArecordDataSources."""
        pytest.importorskip("array_record")
        from array_record.python.array_record_module import ArrayRecordWriter
        from megatext.data.data_sources import FixedArecordDataSource, BlendedDataSource

        seq_len = 8
        num_tokens = seq_len + 1

        # Create two separate directories with different data
        dirs = []
        for d in range(2):
            data_dir = tmp_path / f"source_{d}"
            data_dir.mkdir()
            writer = ArrayRecordWriter(str(data_dir / "data.arecord"), "group_size:1")
            rng = np.random.RandomState(d)
            for _ in range(10):
                tokens = rng.randint(0, 5000, size=num_tokens, dtype=np.int32)
                writer.write(tokens.tobytes())
            writer.close()
            dirs.append(str(data_dir))

        src_a = FixedArecordDataSource(data_path=dirs[0], seq_len=seq_len)
        src_b = FixedArecordDataSource(data_path=dirs[1], seq_len=seq_len)
        total = len(src_a) + len(src_b)

        blended = BlendedDataSource([src_a, src_b], [1.0, 1.0], size=total)
        assert isinstance(blended, grain.RandomAccessDataSource)
        assert len(blended) == total

        # All samples should be valid
        for i in range(len(blended)):
            sample = blended[i]
            assert "tokens" in sample
            assert len(sample["tokens"]) == num_tokens


# -- Checkpointing compatibility ---------------------------------------------


class TestFixedArecordCheckpointing:
    """Test Grain checkpointing works with FixedArecordDataSource."""

    def test_iterator_state_roundtrip(self, fixed_arecord_dir):
        """Save/restore iterator state should produce identical data."""
        from megatext.data.data_sources import FixedArecordDataSource

        data_path, seq_len, _ = fixed_arecord_dir

        def make_loader():
            ds = FixedArecordDataSource(data_path=data_path, seq_len=seq_len)
            sampler = grain.IndexSampler(
                num_records=len(ds), shard_options=grain.NoSharding(),
                shuffle=False, num_epochs=None, seed=42,
            )
            return grain.DataLoader(
                data_source=ds, sampler=sampler,
                operations=[grain.Batch(batch_size=2, drop_remainder=True)],
                worker_count=0,
            )

        # Collect 5 batches
        loader1 = make_loader()
        it1 = iter(loader1)
        batches = [next(it1) for _ in range(5)]

        # Save state after 3 batches
        loader2 = make_loader()
        it2 = iter(loader2)
        for _ in range(3):
            next(it2)
        state = it2.get_state()

        # Restore
        loader3 = make_loader()
        it3 = iter(loader3)
        it3.set_state(state)

        restored = next(it3)
        np.testing.assert_array_equal(restored["tokens"], batches[3]["tokens"])
