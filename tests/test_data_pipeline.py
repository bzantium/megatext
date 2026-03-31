"""Tests for the data pipeline (FormatForMaxText transform)."""

import numpy as np
import pytest


class TestFormatForMaxText:
    def test_basic_format(self):
        from megatext.data.data_processing import FormatForMaxText

        transform = FormatForMaxText(seq_len=8, add_extra_token=True)
        element = {"tokens": np.arange(9, dtype=np.int32)}  # 8 + 1

        result = transform.map(element)

        assert "inputs" in result
        assert "targets" in result
        assert "inputs_position" in result
        assert "inputs_segmentation" in result
        assert "targets_segmentation" in result

        assert result["inputs"].shape == (8,)
        assert result["targets"].shape == (8,)
        np.testing.assert_array_equal(result["inputs"], np.arange(8))
        np.testing.assert_array_equal(result["targets"], np.arange(1, 9))

    def test_positions_sequential(self):
        from megatext.data.data_processing import FormatForMaxText

        transform = FormatForMaxText(seq_len=8)
        element = {"tokens": np.arange(9, dtype=np.int32)}

        result = transform.map(element)
        np.testing.assert_array_equal(
            result["inputs_position"], np.arange(8, dtype=np.int32)
        )

    def test_segmentation_no_packing(self):
        from megatext.data.data_processing import FormatForMaxText

        transform = FormatForMaxText(seq_len=8)
        element = {"tokens": np.arange(9, dtype=np.int32)}

        result = transform.map(element)
        np.testing.assert_array_equal(
            result["inputs_segmentation"], np.ones(8, dtype=np.int32)
        )

    def test_with_packing_segment_ids(self):
        from megatext.data.data_processing import FormatForMaxText

        transform = FormatForMaxText(seq_len=8)
        segment_ids = np.array([1, 1, 1, 2, 2, 2, 2, 0, 0], dtype=np.int32)
        element = {
            "tokens": np.arange(9, dtype=np.int32),
            "segment_ids": segment_ids,
        }

        result = transform.map(element)
        # Segment IDs truncated to seq_len
        np.testing.assert_array_equal(
            result["inputs_segmentation"], segment_ids[:8]
        )

    def test_with_loss_mask(self):
        from megatext.data.data_processing import FormatForMaxText

        transform = FormatForMaxText(seq_len=4)
        loss_mask = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)
        element = {
            "tokens": np.arange(5, dtype=np.int32),
            "segment_ids": np.ones(5, dtype=np.int32),
            "loss_mask": loss_mask,
        }

        result = transform.map(element)
        # targets_segmentation should have 0 where loss_mask is 0
        expected_target_seg = np.array([1, 0, 1, 1], dtype=np.int32)
        np.testing.assert_array_equal(result["targets_segmentation"], expected_target_seg)


class TestParseDataPath:
    def test_single_path(self):
        from megatext.data.data_processing import _parse_data_path

        result = _parse_data_path("/data/train")
        assert result == [(1.0, "/data/train")]

    def test_weighted_paths(self):
        from megatext.data.data_processing import _parse_data_path

        result = _parse_data_path("0.7 /data/a 0.3 /data/b")
        assert result == [(0.7, "/data/a"), (0.3, "/data/b")]

    def test_path_with_spaces(self):
        from megatext.data.data_processing import _parse_data_path

        result = _parse_data_path("/data/no_spaces")
        assert result == [(1.0, "/data/no_spaces")]


class TestParseSplit:
    def test_basic(self):
        from megatext.data.data_processing import _parse_split

        result = _parse_split("99,1,0")
        assert result == (99.0, 1.0, 0.0)

    def test_invalid(self):
        from megatext.data.data_processing import _parse_split

        with pytest.raises(ValueError):
            _parse_split("50,50")
