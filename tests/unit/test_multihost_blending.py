"""Test multi-host data loading and blending logic.

Simulates multi-process sharding on a single CPU host by manually varying
process_index / num_processes against Grain's MapDataset slice + IterDataset.mix.
"""

import grain.python as grain
import numpy as np
import numpy.testing as npt


# ---------------------------------------------------------------------------
# Helpers: minimal data source that returns (source_id, sample_index) tuples
# ---------------------------------------------------------------------------


class TaggedSource(grain.RandomAccessDataSource):
    """Returns {'source': source_id, 'index': i} for each sample."""

    def __init__(self, source_id: int, size: int):
        super().__init__()
        self._source_id = source_id
        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return {"source": self._source_id, "index": idx}


def _build_pipeline(
    sources: list[TaggedSource],
    weights: list[float],
    process_index: int,
    num_processes: int,
    batch_size_per_process: int,
    num_batches: int,
    seed: int = 42,
):
    """Build the same pipeline as _create_fixed_arecord_iterator but with TaggedSources."""

    def _make_iter_dataset(source: TaggedSource) -> grain.IterDataset:
        dataset = grain.MapDataset.source(source)
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset.repeat(None)
        dataset = dataset[process_index::num_processes]
        return dataset.to_iter_dataset()

    if len(sources) == 1:
        dataset = _make_iter_dataset(sources[0])
    else:
        datasets = [_make_iter_dataset(s) for s in sources]
        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]
        dataset = grain.IterDataset.mix(datasets, norm_weights)

    dataset = dataset.batch(batch_size_per_process, drop_remainder=True)

    batches = []
    it = iter(dataset)
    for _ in range(num_batches):
        batches.append(next(it))
    return batches


# ---------------------------------------------------------------------------
# Test 1: No sample overlap between processes
# ---------------------------------------------------------------------------


def test_no_overlap_between_processes():
    """Each process should get disjoint sample indices (before repeat wraps)."""
    num_processes = 4
    source_size = 100
    batch_size = 5
    num_batches = 4  # 4 batches * 5 = 20 samples per process

    source = TaggedSource(source_id=0, size=source_size)

    per_process_indices = {}
    for pid in range(num_processes):
        batches = _build_pipeline(
            sources=[source],
            weights=[1.0],
            process_index=pid,
            num_processes=num_processes,
            batch_size_per_process=batch_size,
            num_batches=num_batches,
        )
        indices = set()
        for b in batches:
            for idx in b["index"]:
                indices.add(int(idx))
        per_process_indices[pid] = indices

    # Verify pairwise disjoint
    for i in range(num_processes):
        for j in range(i + 1, num_processes):
            overlap = per_process_indices[i] & per_process_indices[j]
            assert len(overlap) == 0, (
                f"Process {i} and {j} share indices: {overlap}"
            )

    # Verify union covers a contiguous range
    all_indices = set()
    for s in per_process_indices.values():
        all_indices |= s
    # With 100 samples, 4 processes, each gets 25. 20 drawn per process.
    # All drawn indices should be valid
    assert all(0 <= idx < source_size for idx in all_indices)


# ---------------------------------------------------------------------------
# Test 2: Blend ratios match configured weights
# ---------------------------------------------------------------------------


def test_blend_ratios():
    """Sample-level blend should produce source proportions close to weights."""
    sources = [
        TaggedSource(source_id=0, size=200),
        TaggedSource(source_id=1, size=200),
        TaggedSource(source_id=2, size=200),
    ]
    weights = [0.5, 0.3, 0.2]
    batch_size = 10
    num_batches = 200  # 2000 total samples

    batches = _build_pipeline(
        sources=sources,
        weights=weights,
        process_index=0,
        num_processes=1,
        batch_size_per_process=batch_size,
        num_batches=num_batches,
    )

    counts = {0: 0, 1: 0, 2: 0}
    for b in batches:
        for sid in b["source"]:
            counts[int(sid)] += 1

    total = sum(counts.values())
    actual_ratios = {k: v / total for k, v in counts.items()}

    # Allow 5% tolerance for stochastic sampling
    npt.assert_allclose(actual_ratios[0], 0.5, atol=0.05)
    npt.assert_allclose(actual_ratios[1], 0.3, atol=0.05)
    npt.assert_allclose(actual_ratios[2], 0.2, atol=0.05)


# ---------------------------------------------------------------------------
# Test 3: Each batch contains samples from multiple sources
# ---------------------------------------------------------------------------


def test_batches_are_mixed():
    """With sample-level blending, most batches should contain multiple sources."""
    sources = [
        TaggedSource(source_id=0, size=200),
        TaggedSource(source_id=1, size=200),
    ]
    weights = [0.5, 0.5]
    batch_size = 20
    num_batches = 50

    batches = _build_pipeline(
        sources=sources,
        weights=weights,
        process_index=0,
        num_processes=1,
        batch_size_per_process=batch_size,
        num_batches=num_batches,
    )

    mixed_count = 0
    for b in batches:
        unique_sources = set(int(s) for s in b["source"])
        if len(unique_sources) > 1:
            mixed_count += 1

    # With 50/50 split and batch_size=20, nearly all batches should be mixed
    mix_ratio = mixed_count / num_batches
    assert mix_ratio > 0.9, (
        f"Only {mix_ratio:.0%} of batches are mixed — expected >90%"
    )
