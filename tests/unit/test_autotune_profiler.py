"""Unit tests for autotune profiler diagnostics."""

from megatext.autotune.profiler import _first_error_line, _looks_like_oom


def test_looks_like_oom_matches_strong_signatures():
    assert _looks_like_oom("RESOURCE_EXHAUSTED: Out of memory on HBM")
    assert _looks_like_oom("HBMOOM while compiling")
    assert _looks_like_oom("vmemoom on TPU")


def test_looks_like_oom_does_not_match_arbitrary_substrings():
    assert not _looks_like_oom("BloomModel failed to initialize")
    assert not _looks_like_oom("room temperature test")
    assert not _looks_like_oom("zoom level mismatch")


def test_first_error_line_returns_first_non_empty_line():
    message = "\n\n  RuntimeError: bad config\nTraceback ..."
    assert _first_error_line(message) == "RuntimeError: bad config"
