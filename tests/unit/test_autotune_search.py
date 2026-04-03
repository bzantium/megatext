"""Unit tests for autotune staircase boundary walk search algorithm.

Mocks profile_candidate to simulate OOM/success based on a memory model,
verifying the search finds Pareto-optimal points efficiently.
"""

from unittest.mock import patch

import pytest

from megatext.autotune.profiler import EXIT_CLASS_INFRA_ERROR, ProfileResult
from megatext.autotune.search import AutoTuneConfig, _search_batch_remat, pareto_frontier, run_search
from megatext.autotune.strategies import Candidate, REMAT_POLICIES
from megatext.autotune.topology import TPUTopology


# ---------------------------------------------------------------------------
# Mock profiler
# ---------------------------------------------------------------------------

# Map remat policy to memory factor (ascending = more memory)
_REMAT_MEMORY = {policy: i + 1 for i, policy in enumerate(REMAT_POLICIES)}
# Map remat policy to speed factor (less aggressive remat = faster)
_REMAT_SPEED = {policy: len(REMAT_POLICIES) - i for i, policy in enumerate(REMAT_POLICIES)}


def _make_mock_profiler(memory_threshold: float):
    """Create a mock profile_candidate that uses a simple memory model.

    memory = batch * remat_factor * (sa_block / 512)
    tflops = batch * speed_factor * (sa_block / 512)
    OOM if memory > threshold.
    """
    call_count = 0

    def mock_profile(config_overrides, candidate, num_steps=3, warmup_steps=3):
        nonlocal call_count
        call_count += 1

        batch = candidate.per_device_batch_size
        remat = candidate.remat_policy
        sa_block = candidate.sa_block_size

        remat_mem = _REMAT_MEMORY[remat]
        remat_speed = _REMAT_SPEED[remat]
        sa_factor = sa_block / 512

        memory = batch * remat_mem * sa_factor
        tflops = batch * remat_speed * sa_factor

        oom = memory > memory_threshold

        return ProfileResult(
            candidate=candidate,
            mean_step_time_seconds=1.0 / tflops if not oom else 0.0,
            max_step_time_seconds=1.0 / tflops if not oom else 0.0,
            min_step_time_seconds=1.0 / tflops if not oom else 0.0,
            peak_memory_gb=memory,
            tflops_per_device=tflops if not oom else 0.0,
            oom=oom,
        )

    mock_profile.get_call_count = lambda: call_count
    return mock_profile


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("megatext.autotune.search.profile_candidate")
def test_finds_optimal_on_boundary(mock_profile):
    """Best result should be the Pareto-optimal point (highest TFLOPs on boundary)."""
    profiler = _make_mock_profiler(memory_threshold=16)
    mock_profile.side_effect = profiler

    all_results = []
    best = _search_batch_remat(
        config_overrides={},
        all_results=all_results,
        num_steps=3,
        warmup_steps=3,
        valid_sa_blocks=[512],
        batch_sizes=[1, 2, 3, 4, 5, 6, 7, 8],
    )

    assert best is not None
    assert not best.oom
    # With threshold=16, sa=512, remat=full (mem_factor=1): max batch = 16 → all fit
    # remat=full, batch=8: memory=8*1*1=8 ≤ 16, tflops=8*8*1=64
    # This should be the best since full remat has highest speed factor
    assert best.candidate.remat_policy == "full"
    assert best.candidate.per_device_batch_size == 8


@patch("megatext.autotune.search.profile_candidate")
def test_all_oom(mock_profile):
    """When nothing fits, returns None."""
    profiler = _make_mock_profiler(memory_threshold=0.5)
    mock_profile.side_effect = profiler

    all_results = []
    best = _search_batch_remat(
        config_overrides={},
        all_results=all_results,
        num_steps=3,
        warmup_steps=3,
        valid_sa_blocks=[512],
        batch_sizes=[1, 2, 3, 4],
    )

    assert best is None


@patch("megatext.autotune.search.profile_candidate")
def test_single_feasible(mock_profile):
    """Only batch=1, remat=full fits."""
    # threshold=1: only batch=1, remat=full (mem=1*1*1=1) fits
    profiler = _make_mock_profiler(memory_threshold=1)
    mock_profile.side_effect = profiler

    all_results = []
    best = _search_batch_remat(
        config_overrides={},
        all_results=all_results,
        num_steps=3,
        warmup_steps=3,
        valid_sa_blocks=[512],
        batch_sizes=[1, 2, 3, 4],
    )

    assert best is not None
    assert best.candidate.per_device_batch_size == 1
    assert best.candidate.remat_policy == "full"


@patch("megatext.autotune.search.profile_candidate")
def test_staircase_shape(mock_profile):
    """Multiple remat policies feasible at different batch sizes.

    With threshold=24:
      remat=full (factor=1): max batch=24 → capped at 8
      remat=save_out_proj (factor=2): max batch=12 → capped at 8
      remat=save_qkv_proj (factor=3): max batch=8
      remat=save_dot_except_mlp (factor=4): max batch=6
      remat=save_dot_except_mlpwi (factor=5): max batch=4
      remat=save_dot_with_context_except_mlp (factor=6): max batch=4
      remat=minimal (factor=7): max batch=3
      remat=minimal_with_context (factor=8): max batch=3

    Best should be remat=full, batch=8 (tflops=8*8=64)
    """
    profiler = _make_mock_profiler(memory_threshold=24)
    mock_profile.side_effect = profiler

    all_results = []
    best = _search_batch_remat(
        config_overrides={},
        all_results=all_results,
        num_steps=3,
        warmup_steps=3,
        valid_sa_blocks=[512],
        batch_sizes=[1, 2, 3, 4, 5, 6, 7, 8],
    )

    assert best is not None
    assert best.candidate.remat_policy == "full"
    assert best.candidate.per_device_batch_size == 8

    # Verify multiple Pareto points were found (successful results from different remats)
    successful = [r for r in all_results if not r.oom and r.error is None]
    successful_remats = {r.candidate.remat_policy for r in successful}
    # At least full and some others should have been profiled successfully
    assert "full" in successful_remats
    assert len(successful_remats) >= 3


@patch("megatext.autotune.search.profile_candidate")
def test_probe_count_with_ceiling_shrink(mock_profile):
    """Linear scan with ceiling shrink stays well below the full staged grid."""
    profiler = _make_mock_profiler(memory_threshold=24)
    mock_profile.side_effect = profiler

    batch_sizes = [1, 2, 3, 4, 5, 6, 7, 8]
    B = len(batch_sizes)
    R = len(REMAT_POLICIES)

    all_results = []
    _search_batch_remat(
        config_overrides={},
        all_results=all_results,
        num_steps=3,
        warmup_steps=3,
        valid_sa_blocks=[512],
        batch_sizes=batch_sizes,
    )

    total_probes = profiler.get_call_count()
    # For sa_block=512 the backward refinement stays at 512, so this falls back
    # to the original 1D search budget.
    assert total_probes < R * B, (
        f"Too many probes: {total_probes} >= {R * B}"
    )


@patch("megatext.autotune.search.profile_candidate")
def test_sa_block_binary_search(mock_profile):
    """SA block binary search finds max feasible block."""
    # threshold=4: sa=512 (factor=1) fits at batch=1, sa=1024 (factor=2) fits,
    # sa=2048 (factor=4) OOMs, sa=4096 (factor=8) OOMs
    profiler = _make_mock_profiler(memory_threshold=3)
    mock_profile.side_effect = profiler

    all_results = []
    best = _search_batch_remat(
        config_overrides={},
        all_results=all_results,
        num_steps=3,
        warmup_steps=3,
        valid_sa_blocks=[512, 1024, 2048, 4096],
        batch_sizes=[1, 2, 3, 4],
    )

    assert best is not None
    # sa=512 (factor=1): batch=1, remat=full, mem=1 ≤ 3 → tflops=8
    # sa=1024 (factor=2): batch=1, remat=full, mem=2 ≤ 3 → tflops=16
    # sa=1024 is better, and should be the max feasible sa_block
    assert best.candidate.sa_block_size in [512, 1024]


@patch("megatext.autotune.search.profile_candidate")
def test_failure_gets_pruned_like_other_failed_candidates(mock_profile):
    """Any failed candidate should shrink the search boundary and let search continue."""

    def side_effect(config_overrides, candidate, num_steps=3, warmup_steps=3):
        if candidate.remat_policy == "save_out_proj":
            return ProfileResult(
                candidate=candidate,
                mean_step_time_seconds=float("inf"),
                max_step_time_seconds=float("inf"),
                min_step_time_seconds=float("inf"),
                peak_memory_gb=0.0,
                tflops_per_device=0.0,
                oom=False,
                error="worker exited without result",
                exit_class=EXIT_CLASS_INFRA_ERROR,
            )
        return ProfileResult(
            candidate=candidate,
            mean_step_time_seconds=1.0,
            max_step_time_seconds=1.0,
            min_step_time_seconds=1.0,
            peak_memory_gb=1.0,
            tflops_per_device=10.0,
            oom=False,
        )

    mock_profile.side_effect = side_effect

    with patch("megatext.autotune.search.REMAT_POLICIES", ["full", "save_out_proj", "minimal"]):
        with patch("megatext.autotune.search.BATCH_SIZES", [1, 2, 3]):
            best = _search_batch_remat(
                config_overrides={},
                all_results=[],
                num_steps=3,
                warmup_steps=3,
                valid_sa_blocks=[512],
                batch_sizes=[1, 2, 3],
            )

    assert best is not None
    assert best.candidate.remat_policy == "full"


@patch("megatext.autotune.search.profile_candidate")
def test_batch_scan_stops_after_throughput_decline(mock_profile):
    """Once throughput declines at larger batch, the scan should stop early."""

    calls = []
    tflops_by_batch = {1: 10.0, 2: 20.0, 3: 19.0, 4: 18.0}

    def side_effect(config_overrides, candidate, num_steps=3, warmup_steps=3):
        del config_overrides, num_steps, warmup_steps
        calls.append(candidate.per_device_batch_size)
        return ProfileResult(
            candidate=candidate,
            mean_step_time_seconds=1.0 / tflops_by_batch[candidate.per_device_batch_size],
            max_step_time_seconds=1.0 / tflops_by_batch[candidate.per_device_batch_size],
            min_step_time_seconds=1.0 / tflops_by_batch[candidate.per_device_batch_size],
            peak_memory_gb=float(candidate.per_device_batch_size),
            tflops_per_device=tflops_by_batch[candidate.per_device_batch_size],
            oom=False,
        )

    mock_profile.side_effect = side_effect

    with patch("megatext.autotune.search.REMAT_POLICIES", ["full"]):
        best = _search_batch_remat(
            config_overrides={},
            all_results=[],
            num_steps=3,
            warmup_steps=3,
            valid_sa_blocks=[512],
            batch_sizes=[1, 2, 3, 4],
        )

    assert best is not None
    assert best.candidate.per_device_batch_size == 2
    assert calls == [1, 2, 3]


def test_pareto_frontier_filters_dominated_points():
    """Higher-memory lower-throughput points should not appear on the frontier."""
    results = [
        ProfileResult(candidate=Candidate(sa_block_size=512), mean_step_time_seconds=0.0, max_step_time_seconds=0.0, min_step_time_seconds=0.0, peak_memory_gb=4.0, tflops_per_device=10.0, oom=False),
        ProfileResult(candidate=Candidate(sa_block_size=1024), mean_step_time_seconds=0.0, max_step_time_seconds=0.0, min_step_time_seconds=0.0, peak_memory_gb=5.0, tflops_per_device=9.0, oom=False),
        ProfileResult(candidate=Candidate(sa_block_size=2048), mean_step_time_seconds=0.0, max_step_time_seconds=0.0, min_step_time_seconds=0.0, peak_memory_gb=6.0, tflops_per_device=12.0, oom=False),
    ]

    frontier = pareto_frontier(results)
    assert [r.peak_memory_gb for r in frontier] == [4.0, 6.0]


def test_run_search_limits_sa_blocks_by_sliding_window():
    seen_sa_blocks = []

    def evaluator(config_overrides, candidate, num_steps=3, warmup_steps=3):
        del config_overrides, num_steps, warmup_steps
        seen_sa_blocks.append(candidate.sa_block_size)
        return ProfileResult(
            candidate=candidate,
            mean_step_time_seconds=1.0,
            max_step_time_seconds=1.0,
            min_step_time_seconds=1.0,
            peak_memory_gb=float(candidate.sa_block_size) / 512.0,
            tflops_per_device=float(candidate.sa_block_size) / 512.0,
            oom=False,
        )

    topology = TPUTopology(
        device_count=8,
        local_device_count=8,
        process_count=1,
        process_index=0,
        platform="tpu",
        chip_name="v5e",
        devices_per_chip=1,
        num_slices=1,
        chips_per_host=8,
        hbm_per_device_gb=16.0,
    )

    run_search(
        config_overrides={
            "max_target_length": 4096,
            "sliding_window_size": 512,
        },
        autotune_config=AutoTuneConfig(scope="batch_remat", include_sa_block=True),
        topology=topology,
        max_batch_size=1,
        evaluator=evaluator,
    )

    assert seen_sa_blocks
    assert set(seen_sa_blocks) == {512}


def test_run_search_refines_backward_sa_block_with_half_step():
    seen_candidates = []

    def evaluator(config_overrides, candidate, num_steps=3, warmup_steps=3):
        del config_overrides, num_steps, warmup_steps
        seen_candidates.append((candidate.sa_block_size, candidate.sa_block_backward_size))
        backward = candidate.sa_block_backward_size or candidate.sa_block_size
        if candidate.sa_block_size == 1024 and backward == 1024:
            return ProfileResult(
                candidate=candidate,
                mean_step_time_seconds=float("inf"),
                max_step_time_seconds=float("inf"),
                min_step_time_seconds=float("inf"),
                peak_memory_gb=0.0,
                tflops_per_device=0.0,
                oom=True,
            )
        tflops = 20.0 if candidate.sa_block_size == 1024 and backward == 512 else 10.0
        return ProfileResult(
            candidate=candidate,
            mean_step_time_seconds=1.0 / tflops,
            max_step_time_seconds=1.0 / tflops,
            min_step_time_seconds=1.0 / tflops,
            peak_memory_gb=float(candidate.sa_block_size + backward) / 512.0,
            tflops_per_device=tflops,
            oom=False,
        )

    topology = TPUTopology(
        device_count=8,
        local_device_count=8,
        process_count=1,
        process_index=0,
        platform="tpu",
        chip_name="v5e",
        devices_per_chip=1,
        num_slices=1,
        chips_per_host=8,
        hbm_per_device_gb=16.0,
    )

    result = run_search(
        config_overrides={
            "max_target_length": 4096,
            "sliding_window_size": 512,
        },
        autotune_config=AutoTuneConfig(
            scope="batch_remat",
            include_sa_block=True,
            refine_sa_backward=True,
        ),
        topology=topology,
        max_batch_size=1,
        evaluator=evaluator,
    )

    assert result.best_candidate is not None
    assert result.best_candidate.sa_block_size == 1024
    assert result.best_candidate.sa_block_backward_size == 512
    assert (1024, 1024) in seen_candidates
    assert (1024, 512) in seen_candidates
