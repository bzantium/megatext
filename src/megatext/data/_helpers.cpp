/* Fast index-building helpers for megatext data pipelines.
 *
 * Provides two pybind11-exposed functions:
 *   - build_greedy_sample_idx: Megatron-style cross-document concatenation
 *   - build_best_fit_sample_idx: Segment-tree best-fit bin packing
 *
 * Reference: Megatron-Patch helpers.cpp
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <deque>
#include <iostream>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// 1. Greedy (cross-document sequential concatenation)
//    Equivalent to Megatron build_sample_idx.
//    Returns int64 array of shape [num_samples+1, 2]:
//      [..., 0] = index into doc_idx
//      [..., 1] = offset within that document
// ---------------------------------------------------------------------------
py::array_t<int64_t> build_greedy_sample_idx(
    const py::array_t<int32_t> &sizes_,
    const py::array_t<int64_t> &doc_idx_,
    const int32_t seq_length,
    const int32_t num_epochs,
    const int64_t tokens_per_epoch,
    const bool drop_last_partial_sequence = true,
    const int add_extra_token = 1) {

  assert(seq_length > 1);
  assert(num_epochs > 0);
  assert(tokens_per_epoch > 1);

  auto sizes = sizes_.unchecked<1>();
  auto doc_idx = doc_idx_.unchecked<1>();

  int64_t num_samples = 0;
  if (drop_last_partial_sequence) {
    num_samples =
        (num_epochs * tokens_per_epoch - add_extra_token) / seq_length;
  } else {
    num_samples = static_cast<int64_t>(std::ceil(
        static_cast<double>(num_epochs * tokens_per_epoch - add_extra_token) /
        seq_length));
  }

  int64_t *sample_idx = new int64_t[2 * (num_samples + 1)];

  int64_t sample_idx_index = 0;
  int64_t document_idx_index = 0;
  int64_t doc_offset = 0;

  sample_idx[0] = document_idx_index;
  sample_idx[1] = doc_offset;
  ++sample_idx_index;

  while (sample_idx_index <= num_samples) {
    int32_t remaining = seq_length + add_extra_token;
    while (remaining != 0) {
      auto document_index = doc_idx[document_idx_index];
      auto document_length = sizes[document_index] - doc_offset;
      remaining -= document_length;
      if (remaining <= 0) {
        doc_offset += (remaining + document_length - add_extra_token);
        remaining = 0;
      } else {
        if (document_idx_index == (doc_idx_.shape(0) - 1)) {
          assert(sample_idx_index == num_samples);
          doc_offset = sizes[doc_idx[document_idx_index]] - add_extra_token;
          break;
        }
        ++document_idx_index;
        doc_offset = 0;
      }
    }
    sample_idx[2 * sample_idx_index] = document_idx_index;
    sample_idx[2 * sample_idx_index + 1] = doc_offset;
    ++sample_idx_index;
  }

  py::capsule free_when_done(sample_idx, [](void *p) {
    delete[] reinterpret_cast<int64_t *>(p);
  });

  return py::array_t<int64_t>(
      {num_samples + 1, static_cast<int64_t>(2)},
      {2 * static_cast<int64_t>(sizeof(int64_t)),
       static_cast<int64_t>(sizeof(int64_t))},
      sample_idx, free_when_done);
}

// ---------------------------------------------------------------------------
// 2. Best-fit bin packing with segment tree
//    Returns (chunk_index [N, 3], sample_index [M+1])
// ---------------------------------------------------------------------------
py::tuple build_best_fit_sample_idx(
    const py::array_t<int32_t> &sizes_,
    const py::array_t<int64_t> &doc_idx_,
    const int32_t seq_length,
    const int32_t num_epochs,
    const int64_t tokens_per_epoch,
    const bool drop_last_partial_sequence = true,
    const int add_extra_token = 1,
    const int32_t max_num_chunks_per_sample = -1) {

  assert(seq_length > 1);
  assert(num_epochs > 0);
  assert(tokens_per_epoch > 1);

  auto sizes = sizes_.unchecked<1>();
  auto doc_idx = doc_idx_.unchecked<1>();

  // Upper bound on samples (used for pre-allocation).
  int64_t num_samples = 0;
  if (drop_last_partial_sequence) {
    num_samples =
        (num_epochs * tokens_per_epoch - add_extra_token) / seq_length;
  } else {
    num_samples = static_cast<int64_t>(std::ceil(
        static_cast<double>(num_epochs * tokens_per_epoch - add_extra_token) /
        seq_length));
  }

  // Phase 1: Bucket-sort remainders, count full chunks.
  auto *len_to_docs = new std::deque<int64_t>[seq_length];
  auto *remain_to_bins =
      new std::deque<std::vector<int64_t>>[seq_length + 1];

  // Pre-allocate empty bins.
  for (int64_t i = 0; i < 2 * num_samples; ++i) {
    remain_to_bins[seq_length].emplace_back();
  }

  int64_t num_full_chunks = 0;
  for (int64_t i = 0; i < doc_idx_.shape(0); ++i) {
    auto doc_id = doc_idx[i];
    auto doc_length = sizes[doc_id];
    auto remain = doc_length % seq_length;
    len_to_docs[remain].push_back(doc_id);
    num_full_chunks += doc_length / seq_length;
  }

  // Phase 2: Segment-tree best-fit bin packing.
  // Segment tree over [0, seq_length) — each leaf tracks the max available
  // capacity among bins in remain_to_bins[leaf_value].
  std::vector<int32_t> segtree(2 * seq_length, 0);
  segtree[2 * seq_length - 1] = seq_length; // leaf for capacity=seq_length
  {
    auto parent = seq_length - 1;
    while (parent > 0) {
      segtree[parent] =
          std::max(segtree[parent << 1], segtree[parent << 1 | 1]);
      parent >>= 1;
    }
  }

  auto update_segtree = [&](std::size_t leaf_idx, int32_t new_val) {
    auto cur = static_cast<std::size_t>(seq_length) + leaf_idx - 1;
    segtree[cur] = new_val;
    cur >>= 1;
    while (cur > 0) {
      segtree[cur] = std::max(segtree[cur << 1], segtree[cur << 1 | 1]);
      cur >>= 1;
    }
  };

  for (int32_t doc_length = seq_length - 1; doc_length > 0; --doc_length) {
    while (!len_to_docs[doc_length].empty()) {
      auto doc_id = len_to_docs[doc_length].front();
      len_to_docs[doc_length].pop_front();

      // Best-fit search via segment tree.
      std::size_t cur = 1;
      while (cur < static_cast<std::size_t>(seq_length)) {
        std::size_t left = cur << 1;
        std::size_t right = cur << 1 | 1;
        if (doc_length <= segtree[left]) {
          cur = left;
        } else {
          cur = right;
        }
      }
      auto best_fit = segtree[cur];

      // Move bin from remain_to_bins[best_fit] → remain_to_bins[new_remain].
      auto bin = std::move(remain_to_bins[best_fit].front());
      remain_to_bins[best_fit].pop_front();
      bin.push_back(doc_id);
      auto new_remain = best_fit - doc_length;
      remain_to_bins[new_remain].push_back(std::move(bin));

      // Update segment tree for new_remain (now has bins).
      if (new_remain > 0) {
        update_segtree(new_remain, new_remain);
      }
      // If best_fit bucket is now empty, zero its leaf.
      if (remain_to_bins[best_fit].empty()) {
        update_segtree(best_fit, 0);
      }
    }
  }

  // Collect valid bins (optionally filtering by max_num_chunks_per_sample).
  std::vector<std::vector<int64_t>> valid_bins;
  int64_t total_partial_chunks = 0;

  for (int32_t cap = 0; cap < seq_length; ++cap) {
    for (auto &bin : remain_to_bins[cap]) {
      if (max_num_chunks_per_sample == -1 ||
          static_cast<int32_t>(bin.size()) <= max_num_chunks_per_sample) {
        total_partial_chunks += bin.size();
        valid_bins.push_back(std::move(bin));
      }
    }
  }

  // Optionally drop last partial sequence.
  if (drop_last_partial_sequence && !valid_bins.empty()) {
    int32_t last_total = 0;
    for (auto doc_id : valid_bins.back()) {
      last_total += sizes[doc_id] % seq_length;
    }
    if (last_total < seq_length) {
      total_partial_chunks -= valid_bins.back().size();
      valid_bins.pop_back();
    }
  }

  // Phase 3: Assemble chunk_index and sample_index.
  int64_t exact_num_samples = num_full_chunks + static_cast<int64_t>(valid_bins.size());
  int64_t total_chunks = num_full_chunks + total_partial_chunks;

  std::vector<int64_t> documents(3 * total_chunks);
  std::vector<int64_t> sample_idx(exact_num_samples + 1);

  int64_t doc_ci = 0;
  int64_t sample_ci = 0;

  // Full chunks first.
  for (int64_t i = 0; i < doc_idx_.shape(0); ++i) {
    auto doc_id = doc_idx[i];
    auto doc_length = sizes[doc_id];
    auto chunks = doc_length / seq_length;
    for (int32_t c = 0; c < chunks; ++c) {
      sample_idx[sample_ci++] = doc_ci;
      documents[3 * doc_ci] = doc_id;
      documents[3 * doc_ci + 1] = static_cast<int64_t>(seq_length) * c;
      documents[3 * doc_ci + 2] = seq_length;
      ++doc_ci;
    }
  }

  // Packed bins.
  for (auto &bin : valid_bins) {
    sample_idx[sample_ci++] = doc_ci;
    for (auto doc_id : bin) {
      documents[3 * doc_ci] = doc_id;
      documents[3 * doc_ci + 1] =
          static_cast<int64_t>(seq_length) * (sizes[doc_id] / seq_length);
      documents[3 * doc_ci + 2] = sizes[doc_id] % seq_length;
      ++doc_ci;
    }
  }
  sample_idx[sample_ci] = doc_ci;

  delete[] len_to_docs;
  delete[] remain_to_bins;

  // Return numpy arrays.
  const auto bs = static_cast<int64_t>(sizeof(int64_t));
  py::array chunk_arr(
      {total_chunks, static_cast<int64_t>(3)}, {3 * bs, bs}, documents.data());
  py::array sample_arr(
      {exact_num_samples + 1}, {bs}, sample_idx.data());

  return py::make_tuple(chunk_arr, sample_arr);
}

// ---------------------------------------------------------------------------
PYBIND11_MODULE(_helpers, m) {
  m.doc() = "Fast index-building helpers for megatext data pipelines";
  m.def("build_greedy_sample_idx", &build_greedy_sample_idx,
        py::arg("sizes"), py::arg("doc_idx"), py::arg("seq_length"),
        py::arg("num_epochs"), py::arg("tokens_per_epoch"),
        py::arg("drop_last_partial_sequence") = true,
        py::arg("add_extra_token") = 1);
  m.def("build_best_fit_sample_idx", &build_best_fit_sample_idx,
        py::arg("sizes"), py::arg("doc_idx"), py::arg("seq_length"),
        py::arg("num_epochs"), py::arg("tokens_per_epoch"),
        py::arg("drop_last_partial_sequence") = true,
        py::arg("add_extra_token") = 1,
        py::arg("max_num_chunks_per_sample") = -1);
}
