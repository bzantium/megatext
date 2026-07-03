# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pallas TPU kernel for the Gated Delta Net inter-chunk scan.

The chunked gated delta rule splits into a chunk-parallel precomputation
(the WY factors, handled well by XLA) and a strictly sequential inter-chunk
recurrence. The recurrence is the throughput bottleneck when expressed as
`lax.scan`: every step is a separate XLA loop iteration whose recurrent
state round-trips through HBM between many small fusions.

This module fuses the recurrence into a single Pallas kernel. The grid is
(batch, heads, num_chunks) with the chunk dimension marked "arbitrary" so
each (batch, head) program walks its chunks sequentially while the
recurrent state lives in a VMEM scratch buffer for the whole walk.

Forward additionally materializes the per-chunk input states (FLA-style
chunkwise checkpointing) so the backward kernel can walk the chunks in
reverse with the same fusion structure, carrying the state cotangent in
VMEM.

Shapes (per chunk c, head-batched by the grid):
  w, u, q, k: [C, D]   g (cumulative log-decay within chunk): [C]
  recurrent state h: [D_k, D_v]
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _bdot(a, b, contract, compute_dtype):
  """Batched MXU dot over the leading (heads) axis: contract a-dim vs b-dim."""
  return jax.lax.dot_general(
      a.astype(compute_dtype),
      b.astype(compute_dtype),
      dimension_numbers=(((contract[0],), (contract[1],)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )


def _tril_mask(chunk_size: int, include_diag: bool = True) -> jax.Array:
  rows = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 0)
  cols = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 1)
  return rows >= cols if include_diag else rows > cols


# Chunked tensors reach the kernels in one of two HBM layouts:
#   - head-first [B, N, H, C, D]: blocks (1, 1, TH, C, D) arrive as [TH, C, D].
#   - seq-major  [B, N, C, H, D]: the natural (transpose-free) reshape of
#     [B, S, H, D]. Blocks (1, 1, C, TH, D) arrive as [C, TH, D] and are
#     swapped to the common [TH, C, D] kernel convention in VMEM (register
#     shuffles, not HBM copies). TPU block tiling requires the last two block
#     dims to be multiples of (8, 128) or equal to the array dims, so
#     seq-major blocks are legal exactly when TH is a multiple of 8; per-chunk
#     vectors ([B, N, C, H] g/beta) ride along with a trailing singleton
#     ((TH, 1) trailing block dims: TH multiple of 8, 1 equals the array dim).
_SEQ_MAJOR_HEAD_TILE = 8


def _load_chunk(ref, head_first: bool):
  x = ref[0, 0]
  return x if head_first else jnp.swapaxes(x, 0, 1)


def _load_vec(ref, head_first: bool):
  x = ref[0, 0, :, :, 0]
  return x if head_first else x.T


def _store_chunk(ref, val, head_first: bool):
  ref[0, 0] = (val if head_first else jnp.swapaxes(val, 0, 1)).astype(ref.dtype)


def _store_vec(ref, val, head_first: bool):
  ref[0, 0, :, :, 0] = (val if head_first else val.T).astype(ref.dtype)


def _gdn_scan_fwd_kernel(
    w_ref, u_ref, q_ref, k_ref, g_ref, h0_ref,
    o_ref, h_saved_ref, h_final_ref,
    h_scratch,
    *, chunk_size: int, num_chunks: int, compute_dtype: jnp.dtype, head_first: bool,
):
  """Forward inter-chunk recurrence for one (batch, head) program."""
  n = pl.program_id(2)

  @pl.when(n == 0)
  def _init():
    h_scratch[...] = h0_ref[0].astype(jnp.float32)

  # [TH, d_k, d_v] — the walk is sequential per head, but the TH heads in
  # this cell are independent, so every dot below is batched over the
  # leading axis and Mosaic can pipeline the MXU across heads.
  h = h_scratch[...]
  h_saved_ref[0, 0] = h

  def mxu_dot(a, b, contract=(2, 1)):
    return _bdot(a, b, contract, compute_dtype)

  w = _load_chunk(w_ref, head_first)
  u = _load_chunk(u_ref, head_first).astype(jnp.float32)
  q = _load_chunk(q_ref, head_first)
  k = _load_chunk(k_ref, head_first)
  g = _load_vec(g_ref, head_first).astype(jnp.float32)  # [TH, C]

  exp_g = jnp.exp(g)
  q_g = q.astype(jnp.float32) * exp_g[..., None]
  # Merged matmul: [q_g ; w] @ h doubles the LHS rows for the MXU.
  both = mxu_dot(jnp.concatenate([q_g, w.astype(jnp.float32)], axis=1), h)
  attn_inter, v_prime = both[:, :chunk_size], both[:, chunk_size:]
  v_new = u - v_prime

  p = mxu_dot(q, k, contract=(2, 2))
  g_diff = g[:, :, None] - g[:, None, :]
  mask = _tril_mask(chunk_size)
  decay = jnp.where(mask, jnp.exp(jnp.where(mask, g_diff, 0.0)), 0.0)
  s = p * decay

  _store_chunk(o_ref, attn_inter + mxu_dot(s, v_new), head_first)

  # State update.
  g_last = g[:, chunk_size - 1]
  gamma = jnp.exp(g_last)[:, None, None]
  dvec = jnp.exp(g_last[:, None] - g)
  kd = k.astype(jnp.float32) * dvec[..., None]
  h_new = h * gamma + mxu_dot(kd, v_new, contract=(1, 1))
  h_scratch[...] = h_new

  @pl.when(n == num_chunks - 1)
  def _final():
    h_final_ref[0] = h_new


def _gdn_scan_bwd_kernel(
    w_ref, u_ref, q_ref, k_ref, g_ref, h_saved_ref, do_ref, dh_final_ref,
    dw_ref, du_ref, dq_ref, dk_ref, dg_ref, dh0_ref,
    dh_scratch,
    *, chunk_size: int, num_chunks: int, compute_dtype: jnp.dtype, head_first: bool,
):
  """Reverse inter-chunk recurrence: walks chunks from last to first."""
  n = pl.program_id(2)

  @pl.when(n == 0)
  def _init():
    # Seed the state cotangent with the final-state cotangent.
    dh_scratch[...] = dh_final_ref[0].astype(jnp.float32)

  dh_next = dh_scratch[...]  # [TH, d_k, d_v]; batched over heads like the fwd

  def mxu_dot(a, b, contract=(2, 1)):
    return _bdot(a, b, contract, compute_dtype)

  w = _load_chunk(w_ref, head_first).astype(jnp.float32)
  u = _load_chunk(u_ref, head_first).astype(jnp.float32)
  q = _load_chunk(q_ref, head_first).astype(jnp.float32)
  k = _load_chunk(k_ref, head_first).astype(jnp.float32)
  g = _load_vec(g_ref, head_first).astype(jnp.float32)  # [TH, C]
  h = h_saved_ref[0, 0].astype(jnp.float32)
  do = _load_chunk(do_ref, head_first).astype(jnp.float32)

  # --- Recompute forward intermediates for this chunk ---
  exp_g = jnp.exp(g)
  q_g = q * exp_g[..., None]
  v_prime = mxu_dot(w, h)
  v_new = u - v_prime
  p = mxu_dot(q, k, contract=(2, 2))
  g_diff = g[:, :, None] - g[:, None, :]
  mask = _tril_mask(chunk_size)
  decay = jnp.where(mask, jnp.exp(jnp.where(mask, g_diff, 0.0)), 0.0)
  s = p * decay
  g_last = g[:, chunk_size - 1]
  gamma = jnp.exp(g_last)[:, None, None]
  dvec = jnp.exp(g_last[:, None] - g)
  kd = k * dvec[..., None]

  # --- Backward through the state update: h' = gamma*h + kd^T @ v_new ---
  dgamma = jnp.sum(h * dh_next, axis=(1, 2))
  dkd = mxu_dot(v_new, dh_next, contract=(2, 2))  # [TH, C, D_k]
  dv_new = mxu_dot(kd, dh_next)  # [TH, C, D_v]
  dh = dh_next * gamma

  dk = dkd * dvec[..., None]
  ddvec = jnp.sum(dkd * k, axis=2)

  # --- Backward through the output: o = q_g @ h + s @ v_new ---
  ds = mxu_dot(do, v_new, contract=(2, 2))
  ds = jnp.where(mask, ds, 0.0)
  dv_new = dv_new + mxu_dot(s, do, contract=(1, 1))
  dp = ds * decay
  ddecay = ds * p
  dq = mxu_dot(dp, k)
  dk = dk + mxu_dot(dp, q, contract=(1, 1))
  # decay = exp(g_i - g_j) on the tril: dg_i += sum_j(ddecay*decay); dg_j -= sum_i(...)
  dgd = ddecay * decay
  dg = jnp.sum(dgd, axis=2) - jnp.sum(dgd, axis=1)

  dq_g = mxu_dot(do, h, contract=(2, 2))
  dh = dh + mxu_dot(q_g, do, contract=(1, 1))

  # --- Backward through v_new = u - w @ h ---
  du = dv_new
  dw = -mxu_dot(dv_new, h, contract=(2, 2))
  dh = dh - mxu_dot(w, dv_new, contract=(1, 1))

  # --- Backward through q_g = q * exp(g) ---
  dq = dq + dq_g * exp_g[..., None]
  dg = dg + jnp.sum(dq_g * q, axis=2) * exp_g

  # --- Decay-vector and gamma contributions to g ---
  # dvec = exp(g_last - g); gamma = exp(g_last)
  dg = dg - ddvec * dvec
  dg_last = jnp.sum(ddvec * dvec, axis=1) + dgamma * jnp.exp(g_last)
  one_hot_last = (jax.lax.broadcasted_iota(jnp.int32, (chunk_size,), 0) == chunk_size - 1).astype(jnp.float32)
  dg = dg + dg_last[:, None] * one_hot_last[None, :]

  _store_chunk(dw_ref, dw, head_first)
  _store_chunk(du_ref, du, head_first)
  _store_chunk(dq_ref, dq, head_first)
  _store_chunk(dk_ref, dk, head_first)
  _store_vec(dg_ref, dg, head_first)

  dh_scratch[...] = dh

  @pl.when(n == num_chunks - 1)
  def _final():
    # After the reverse walk over chunk 0, dh is dL/dh_0.
    dh0_ref[0] = dh


def _head_tile(num_heads: int, max_tile: int = 8) -> int:
  """Heads per grid cell: independent heads batched to pipeline the MXU.

  Head-first layout only. The backward cap is VMEM-driven: with f32
  gradient outputs the kernel keeps 14 head-tiled blocks resident
  (8 inputs + 6 outputs), which exceeded the 16MB scoped VMEM limit at
  8 heads, so the head-first backward runs with 4.
  """
  for cand in (max_tile, max_tile // 2, 2):
    if 1 < cand <= num_heads and num_heads % cand == 0:
      return cand
  return 1


def _scan_layout(q_shape, head_first: bool, th: int):
  """Grid/BlockSpec factory for both chunk layouts of the scan kernels.

  Returns (dims, grid, chunk_spec, vec_spec) where dims unpacks q's shape as
  (batch, num_chunks, num_heads, chunk_size, d_k) regardless of layout, and
  chunk_spec/vec_spec take an optional reverse flag for the backward walk.
  """
  if head_first:
    batch, num_chunks, num_heads, chunk_size, d_k = q_shape
    block = lambda last: (1, 1, th, chunk_size, last)
    index = lambda rev: (lambda b, h, n: (b, num_chunks - 1 - n if rev else n, h, 0, 0))
  else:
    batch, num_chunks, chunk_size, num_heads, d_k = q_shape
    assert num_heads % th == 0, "seq-major GDN kernels require num_heads % 8 == 0"
    block = lambda last: (1, 1, chunk_size, th, last)
    index = lambda rev: (lambda b, h, n: (b, num_chunks - 1 - n if rev else n, 0, h, 0))
  grid = (batch, num_heads // th, num_chunks)
  chunk_spec = lambda d, rev=False: pl.BlockSpec(block(d), index(rev))
  vec_spec = lambda rev=False: pl.BlockSpec(block(1), index(rev))
  return (batch, num_chunks, num_heads, chunk_size, d_k), grid, chunk_spec, vec_spec


def _chunk_shape(batch, num_chunks, num_heads, chunk_size, d, head_first: bool):
  if head_first:
    return (batch, num_chunks, num_heads, chunk_size, d)
  return (batch, num_chunks, chunk_size, num_heads, d)


def _fwd_pallas(w, u, q, k, g, h0, *, compute_dtype=jnp.bfloat16, head_first=True, interpret=False):
  """Runs the forward kernel.

  Inputs are [B, N, H, C, D] / g [B, N, H, C] when head_first, or the
  transpose-free seq-major [B, N, C, H, D] / g [B, N, C, H] otherwise;
  h0 is [B, H, D_k, D_v] in both layouts.
  """
  th = _head_tile(q.shape[2]) if head_first else _SEQ_MAJOR_HEAD_TILE
  dims, grid, chunk_spec, vec_spec = _scan_layout(q.shape, head_first, th)
  batch, num_chunks, num_heads, chunk_size, d_k = dims
  d_v = u.shape[-1]
  # The saved per-chunk states are kernel-private (only the backward kernel
  # reads them), so they keep the head-first layout in both modes.
  state_spec = pl.BlockSpec((1, 1, th, d_k, d_v), lambda b, h, n: (b, n, h, 0, 0))
  bh_state_spec = pl.BlockSpec((1, th, d_k, d_v), lambda b, h, n: (b, h, 0, 0))

  g = g[..., None]
  kernel = functools.partial(
      _gdn_scan_fwd_kernel, chunk_size=chunk_size, num_chunks=num_chunks,
      compute_dtype=compute_dtype, head_first=head_first,
  )
  o, h_saved, h_final = pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[chunk_spec(d_k), chunk_spec(d_v), chunk_spec(d_k), chunk_spec(d_k), vec_spec(), bh_state_spec],
      out_specs=[chunk_spec(d_v), state_spec, bh_state_spec],
      out_shape=[
          jax.ShapeDtypeStruct(_chunk_shape(batch, num_chunks, num_heads, chunk_size, d_v, head_first), compute_dtype),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, d_k, d_v), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_heads, d_k, d_v), jnp.float32),
      ],
      scratch_shapes=[pltpu.VMEM((th, d_k, d_v), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
      ),
      interpret=interpret,
      name="gdn_scan_fwd",
  )(w, u, q, k, g, h0)
  return o, h_saved, h_final


def _bwd_pallas(w, u, q, k, g, h_saved, do, dh_final, *, compute_dtype=jnp.bfloat16, head_first=True, interpret=False):
  """Runs the backward kernel (reverse chunk walk via index remapping).

  Seq-major blocks are only legal at 8-head tiles (4 is not a multiple of
  8), so the seq-major backward must fit TH=8 in scoped VMEM. It does so by
  emitting dw/du/dq/dk directly in the primal dtypes (the VJP wrapper cast
  the former f32 outputs to exactly these dtypes anyway, so the values are
  bit-identical). Resident blocks at TH=8, C=64, D_k=D_v=128 (the shipped
  GDN configs), bf16 primals:
    inputs  w,u,q,k,do (bf16, 128KB each), g (f32 vec, 2KB),
            h_saved (f32, 512KB), dh_final (f32, 512KB)
    outputs dw,du,dq,dk (bf16, 128KB each), dg (f32 vec, 2KB),
            dh0 (f32, 512KB)
    scratch dh (f32, 512KB)
  = 14 blocks + scratch, ~2.7MB single-buffered, ~4.8MB with the chunk-
  stepped blocks double-buffered — comfortably inside scoped VMEM. The
  vmem_limit_bytes bump below gives Mosaic headroom for its in-kernel f32
  temporaries (the [TH, C, C] decay/ds tiles and [TH, C, D] upcasts),
  which the block count above does not capture.
  """
  d_v = u.shape[-1]
  th = _head_tile(q.shape[2], max_tile=4) if head_first else _SEQ_MAJOR_HEAD_TILE
  dims, grid, chunk_spec, vec_spec = _scan_layout(q.shape, head_first, th)
  batch, num_chunks, num_heads, chunk_size, d_k = dims
  # Reverse the chunk dimension: grid step n touches chunk (N-1-n).
  state_spec = pl.BlockSpec((1, 1, th, d_k, d_v), lambda b, h, n: (b, num_chunks - 1 - n, h, 0, 0))
  bh_state_spec = pl.BlockSpec((1, th, d_k, d_v), lambda b, h, n: (b, h, 0, 0))

  g = g[..., None]
  kernel = functools.partial(
      _gdn_scan_bwd_kernel, chunk_size=chunk_size, num_chunks=num_chunks,
      compute_dtype=compute_dtype, head_first=head_first,
  )
  rev = True
  dw, du, dq, dk, dg, dh0 = pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[chunk_spec(d_k, rev), chunk_spec(d_v, rev), chunk_spec(d_k, rev), chunk_spec(d_k, rev), vec_spec(rev), state_spec, chunk_spec(d_v, rev), bh_state_spec],
      out_specs=[chunk_spec(d_k, rev), chunk_spec(d_v, rev), chunk_spec(d_k, rev), chunk_spec(d_k, rev), vec_spec(rev), bh_state_spec],
      out_shape=[
          jax.ShapeDtypeStruct(w.shape, w.dtype),
          jax.ShapeDtypeStruct(u.shape, u.dtype),
          jax.ShapeDtypeStruct(q.shape, q.dtype),
          jax.ShapeDtypeStruct(k.shape, k.dtype),
          jax.ShapeDtypeStruct(g.shape, jnp.float32),
          jax.ShapeDtypeStruct((batch, num_heads, d_k, d_v), jnp.float32),
      ],
      scratch_shapes=[pltpu.VMEM((th, d_k, d_v), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          vmem_limit_bytes=64 * 1024 * 1024,
      ),
      interpret=interpret,
      name="gdn_scan_bwd",
  )(w, u, q, k, g, h_saved, do, dh_final)
  return dw, du, dq, dk, dg[..., 0], dh0


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8))
def gdn_inter_chunk_scan(w, u, q, k, g, h0, interpret=False, compute_dtype=jnp.bfloat16, head_first=True):
  """Fused inter-chunk gated-delta-rule scan.

  Args:
    w, u: WY factors, [B, N, H, C, D_k] / [B, N, H, C, D_v].
    q, k: chunked queries/keys, [B, N, H, C, D_k].
    g: per-chunk cumulative log-decay, [B, N, H, C] (float32).
    h0: initial recurrent state, [B, H, D_k, D_v] (float32).
    interpret: run the Pallas kernels in interpret mode (CPU testing).
    compute_dtype: operand dtype for the MXU matmuls (accumulation is f32).
    head_first: chunked operands use [B, N, H, C, D] / [B, N, H, C]. With
      head_first=False they instead use the transpose-free seq-major layout
      [B, N, C, H, D] / [B, N, C, H] (requires num_heads % 8 == 0).

  Returns:
    (o, h_final): chunk outputs [B, N, H, C, D_v] ([B, N, C, H, D_v] when
    seq-major) and the final recurrent state [B, H, D_k, D_v] (float32).
  """
  o, _, h_final = _fwd_pallas(w, u, q, k, g, h0, compute_dtype=compute_dtype, head_first=head_first, interpret=interpret)
  return o, h_final


def _gdn_scan_vjp_fwd(w, u, q, k, g, h0, interpret, compute_dtype, head_first):
  o, h_saved, h_final = _fwd_pallas(w, u, q, k, g, h0, compute_dtype=compute_dtype, head_first=head_first, interpret=interpret)
  return (o, h_final), (w, u, q, k, g, h_saved)


def _gdn_scan_vjp_bwd(interpret, compute_dtype, head_first, residuals, cotangents):
  w, u, q, k, g, h_saved = residuals
  do, dh_final = cotangents
  # The backward kernel seeds the reverse walk with dh_final and emits dh0,
  # so the state chain is differentiated end-to-end inside the kernel.
  # Gradients are accumulated in float32 inside the kernel and stored in the
  # primal dtypes, so they come back ready to use as cotangents.
  dw, du, dq, dk, dg, dh0 = _bwd_pallas(
      w, u, q, k, g, h_saved, do, dh_final, compute_dtype=compute_dtype, head_first=head_first, interpret=interpret
  )
  return dw, du, dq, dk, dg.astype(g.dtype), dh0


gdn_inter_chunk_scan.defvjp(_gdn_scan_vjp_fwd, _gdn_scan_vjp_bwd)


# =============================================================================
# Unit-lower-triangular inversion kernel
# =============================================================================
# Replaces jax.scipy.linalg.solve_triangular for the UT-transform inverse
# A = (I + S)^{-1}. The TPU triangular solve substitutes row by row and
# barely uses the MXU; blockwise inversion (exact 16x16 base case plus
# hierarchical [[A,0],[C,B]] combines) is pure matmuls, and running it as a
# Pallas kernel keeps every [C, C] tile in VMEM instead of round-tripping
# the intermediate block products through HBM (which made the same
# algorithm speed-neutral at the XLA level).


def _mm_bf16x3(a, b):
  """bf16x3 batched matmul: f32-grade products from bf16 MXU passes.

  Mosaic's default f32 dot truncates operands to bf16 (Precision.HIGH is
  unsupported), so we split each operand into a bf16 hi part plus a bf16
  residual and sum three MXU passes. Any leading axes are batched dots: the
  per-tile products are independent, so Mosaic can pipeline them through
  the MXU and hide the fill/drain of sequentially dependent chains.
  Contracts a's last dim against b's second-to-last dim.
  """
  a_hi = a.astype(jnp.bfloat16)
  b_hi = b.astype(jnp.bfloat16)
  a_lo = (a - a_hi.astype(jnp.float32)).astype(jnp.bfloat16)
  b_lo = (b - b_hi.astype(jnp.float32)).astype(jnp.bfloat16)
  dims = (((a.ndim - 1,), (a.ndim - 2,)), (tuple(range(a.ndim - 2)),) * 2)
  dot = functools.partial(jax.lax.dot_general, dimension_numbers=dims, preferred_element_type=jnp.float32)
  return dot(a_hi, b_hi) + dot(a_hi, b_lo) + dot(a_lo, b_hi)


def _invert_unit_lower_mxu(s: jax.Array) -> jax.Array:
  """(I + s)^{-1} for strictly lower-triangular s: stable blockwise ladder.

  Level doubling: given X = (I + S_b)^{-1} for the block-diagonal part S_b
  at block size b, the size-2b inverse is exactly X - X @ S_l @ X, where
  S_l holds the entries between sibling b-blocks ([[A,0],[C,B]]^{-1} =
  [[A^{-1},0],[-B^{-1}CA^{-1},B^{-1}]]). Starting from X = I (b=1) and
  doubling to full size costs 2*log2(C) full-width MXU matmuls (bf16x3 for
  f32-grade accuracy), and never forms powers of s — only realized
  inverses times raw s — so intermediate magnitudes stay at the scale of
  the true inverse (stable for unbounded s, unlike Neumann doubling which
  overflows once |s| > 1).
  """
  size = s.shape[-1]
  rows = jax.lax.broadcasted_iota(jnp.int32, (size, size), 0)
  cols = jax.lax.broadcasted_iota(jnp.int32, (size, size), 1)

  x = jnp.broadcast_to(jnp.eye(size, dtype=jnp.float32), s.shape).astype(jnp.float32)
  block = 1
  while block < size:
    sibling = (rows // block != cols // block) & (rows // (2 * block) == cols // (2 * block))
    s_level = jnp.where(sibling, s, 0.0)
    x = x - _mm_bf16x3(x, _mm_bf16x3(s_level, x))
    block *= 2
  return x


def _invert_unit_lower_kernel(s_ref, a_ref, *, chunk_size: int):
  del chunk_size
  a_ref[0, :, 0] = _invert_unit_lower_mxu(s_ref[0, :, 0].astype(jnp.float32))


def _invert_pallas(s, *, interpret=False):
  batch, num_chunks, num_heads, c, _ = s.shape
  # Group chunks per grid cell: the ladder's matmuls are sequentially
  # dependent within a tile, so batching independent tiles keeps the MXU
  # pipeline full. VMEM per cell stays ~1MB at tile_n=8, C=128.
  tile_n = 1
  for cand in (8, 4, 2):
    if num_chunks % cand == 0:
      tile_n = cand
      break
  grid = (batch, num_heads, num_chunks // tile_n)
  spec = pl.BlockSpec((1, tile_n, 1, c, c), lambda b, h, n: (b, n, h, 0, 0))
  return pl.pallas_call(
      functools.partial(_invert_unit_lower_kernel, chunk_size=c),
      grid=grid,
      in_specs=[spec],
      out_specs=spec,
      out_shape=jax.ShapeDtypeStruct(s.shape, jnp.float32),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel"),
      ),
      interpret=interpret,
      name="gdn_invert_unit_lower",
  )(s)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def invert_unit_lower(s, interpret=False):
  """A = (I + s)^{-1} for strictly lower-triangular s, [B, N, H, C, C] float32."""
  return _invert_pallas(s, interpret=interpret)


def _invert_vjp_fwd(s, interpret):
  a = _invert_pallas(s, interpret=interpret)
  return a, a


def _invert_vjp_bwd(interpret, a, da):
  del interpret
  # d(L^{-1}) = -L^{-1} dL L^{-1}  =>  dL = -A^T da A^T; L = I + s with s
  # strictly lower and a unit diagonal, so only the strict lower part flows.
  a_t = a.swapaxes(-1, -2)
  ds = -jnp.matmul(a_t, jnp.matmul(da, a_t))
  c = a.shape[-1]
  rows = jax.lax.broadcasted_iota(jnp.int32, (c, c), 0)
  cols = jax.lax.broadcasted_iota(jnp.int32, (c, c), 1)
  return (jnp.where(rows > cols, ds, 0.0),)


invert_unit_lower.defvjp(_invert_vjp_fwd, _invert_vjp_bwd)


# =============================================================================
# UT-transform kernel: WY factors (w, u) from (k, v, beta, g)
# =============================================================================
# Fuses the whole stage-2 precomputation — k_beta, the decay-masked S, the
# blockwise unit-lower inversion A = (I+S)^{-1}, and the two triangular
# applies — into one all-parallel Pallas kernel. The [C, C] float32 tensors
# S and A live only in VMEM: nothing chunk-squared ever touches HBM, and
# the custom VJP keeps only the small primal inputs (k, v, beta, g) as
# residuals, recomputing S and A in the backward kernel.


def _ut_head_tile(num_heads: int, max_tile: int = 4) -> int:
  """Heads per grid cell for the UT-transform kernels (head-first layout).

  VMEM arithmetic at the cap (TH=4, worst case C=128, D_k=D_v=256):
  - forward: 4 head-tiled [C, D] blocks (k, v in compute dtype; w, u out)
    = 4 * (4*128*256*2B) = 1.0MB, plus ~8 [TH, C, C] f32 intermediates
    (S, A, ladder temporaries) = 8 * (4*128*128*4B) = 2.0MB  ->  ~3MB.
  - backward: 8 wide blocks (k, v, dw, du in; dk, dv out at f32)
    ~= 3.0MB, plus ~10 [TH, C, C] f32 intermediates (S, A, dA, dS, masked
    ladder temps) = 2.5MB  ->  ~5.5MB.
  Both sit comfortably under the 16MB scoped VMEM limit; 4 is kept
  conservative because the ladder's temporaries are not visible to this
  block-count estimate and Mosaic double-buffers the grid blocks.

  The seq-major layout instead pins TH=8 (TPU tiling; see _SEQ_MAJOR_HEAD_
  TILE). At the shipped GDN configs (C=64, D_k=D_v=128) the same counts at
  TH=8 give ~1.5MB of blocks + ~2MB of [TH, C, C] intermediates in the
  forward and ~3MB + ~2.5MB in the backward (dk/dv/dbeta emitted in the
  primal dtypes) — still far under the limit.
  """
  for cand in (max_tile, max_tile // 2, 2):
    if 1 < cand <= num_heads and num_heads % cand == 0:
      return cand
  return 1


def _ut_stage2_core(k, v, beta, g, chunk_size, compute_dtype):
  """Shared head-batched stage-2 math on [TH, C, ...] operands.

  Builds the decay-masked S and its unit-lower inverse A (f32, VMEM only)
  plus the products both the forward and the backward pass consume.
  """
  kf = k.astype(jnp.float32)
  exp_g = jnp.exp(g)
  k_beta = kf * beta[..., None]
  mask_strict = _tril_mask(chunk_size, include_diag=False)
  g_diff = g[:, :, None] - g[:, None, :]
  decay = jnp.where(mask_strict, jnp.exp(jnp.where(mask_strict, g_diff, 0.0)), 0.0)
  s = _bdot(k_beta, kf, (2, 2), compute_dtype) * decay
  a = _invert_unit_lower_mxu(s)
  v_beta = v.astype(jnp.float32) * beta[..., None]
  k_beta_g = k_beta * exp_g[..., None]
  return a, s, decay, mask_strict, exp_g, k_beta, v_beta, k_beta_g


def _gdn_ut_fwd_kernel(
    k_ref, v_ref, beta_ref, g_ref,
    w_ref, u_ref,
    *, chunk_size: int, compute_dtype: jnp.dtype, head_first: bool,
):
  k = _load_chunk(k_ref, head_first)  # [TH, C, D_k]
  v = _load_chunk(v_ref, head_first)  # [TH, C, D_v]
  beta = _load_vec(beta_ref, head_first).astype(jnp.float32)  # [TH, C]
  g = _load_vec(g_ref, head_first).astype(jnp.float32)  # [TH, C]

  a, _, _, _, _, _, v_beta, k_beta_g = _ut_stage2_core(k, v, beta, g, chunk_size, compute_dtype)
  # Merged matmul: A @ [k_beta_g | v_beta] doubles the RHS width for the MXU.
  wu = _bdot(a, jnp.concatenate([k_beta_g, v_beta], axis=-1), (2, 1), compute_dtype)
  d_k = k.shape[-1]
  _store_chunk(w_ref, wu[..., :d_k], head_first)
  _store_chunk(u_ref, wu[..., d_k:], head_first)


def _gdn_ut_bwd_kernel(
    k_ref, v_ref, beta_ref, g_ref, dw_ref, du_ref,
    dk_ref, dv_ref, dbeta_ref, dg_ref,
    *, chunk_size: int, compute_dtype: jnp.dtype, head_first: bool,
):
  """Recomputes S and A from the primal inputs, then runs the chain rule."""
  k = _load_chunk(k_ref, head_first).astype(jnp.float32)
  v = _load_chunk(v_ref, head_first).astype(jnp.float32)
  beta = _load_vec(beta_ref, head_first).astype(jnp.float32)
  g = _load_vec(g_ref, head_first).astype(jnp.float32)
  dw = _load_chunk(dw_ref, head_first).astype(jnp.float32)
  du = _load_chunk(du_ref, head_first).astype(jnp.float32)

  (a, s, decay, mask_strict, exp_g, k_beta, v_beta, k_beta_g) = _ut_stage2_core(
      k, v, beta, g, chunk_size, compute_dtype
  )
  a_t = a.swapaxes(-1, -2)

  # w = A @ k_beta_g ; u = A @ v_beta  (merged pairs for wider matmuls).
  # A math stays f32-grade via the bf16x3 mm, matching the forward ladder.
  dwu = jnp.concatenate([dw, du], axis=-1)
  rhs = jnp.concatenate([k_beta_g, v_beta], axis=-1)
  da = _mm_bf16x3(dwu, rhs.swapaxes(-1, -2))  # dA = dw @ kbg^T + du @ vb^T
  back = _mm_bf16x3(a_t, dwu)  # [A^T dw | A^T du]
  d_k_dim = k.shape[-1]
  dk_beta_g, dv_beta = back[..., :d_k_dim], back[..., d_k_dim:]

  # A = (I + S)^{-1}  =>  dS = -A^T dA A^T, projected to the strict lower part.
  ds = -_mm_bf16x3(a_t, _mm_bf16x3(da, a_t))
  ds = jnp.where(mask_strict, ds, 0.0)

  # S = (k_beta @ k^T) * decay with decay_ij = exp(g_i - g_j) on the strict
  # lower triangle: dP = dS*decay flows into both matmul operands, and the
  # decay factor contributes +row-sums / -col-sums of dS*S to g.
  dp = ds * decay
  dk_beta = _bdot(dp, k, (2, 1), compute_dtype)  # dP @ k
  dk = _bdot(dp, k_beta, (1, 1), compute_dtype)  # dP^T @ k_beta
  dgd = ds * s
  dg = jnp.sum(dgd, axis=2) - jnp.sum(dgd, axis=1)

  # k_beta_g = k_beta * exp(g)
  dk_beta = dk_beta + dk_beta_g * exp_g[..., None]
  dg = dg + jnp.sum(dk_beta_g * k_beta_g, axis=2)

  # k_beta = k * beta ; v_beta = v * beta
  dk = dk + dk_beta * beta[..., None]
  dbeta = jnp.sum(dk_beta * k, axis=2) + jnp.sum(dv_beta * v, axis=2)
  dv = dv_beta * beta[..., None]

  _store_chunk(dk_ref, dk, head_first)
  _store_chunk(dv_ref, dv, head_first)
  _store_vec(dbeta_ref, dbeta, head_first)
  _store_vec(dg_ref, dg, head_first)


def _ut_specs(shape_k, shape_v, head_first):
  if head_first:
    batch, num_chunks, num_heads, chunk_size, d_k = shape_k
    th = _ut_head_tile(num_heads)
    block = lambda last: (1, 1, th, chunk_size, last)
    index = lambda b, h, n: (b, n, h, 0, 0)
  else:
    batch, num_chunks, chunk_size, num_heads, d_k = shape_k
    th = _SEQ_MAJOR_HEAD_TILE
    assert num_heads % th == 0, "seq-major GDN kernels require num_heads % 8 == 0"
    block = lambda last: (1, 1, chunk_size, th, last)
    index = lambda b, h, n: (b, n, 0, h, 0)
  d_v = shape_v[-1]
  grid = (batch, num_heads // th, num_chunks)
  chunk_spec = lambda d: pl.BlockSpec(block(d), index)
  # TPU block tiling requires the last two dims to be (8k, 128k) or equal to
  # the array dims; per-chunk vectors ride along with a trailing singleton.
  vec_spec = pl.BlockSpec(block(1), index)
  return grid, chunk_spec, vec_spec, d_k, d_v


def _ut_fwd_pallas(k, v, beta, g, *, compute_dtype=jnp.bfloat16, head_first=True, interpret=False):
  chunk_size = k.shape[3] if head_first else k.shape[2]
  grid, chunk_spec, vec_spec, d_k, d_v = _ut_specs(k.shape, v.shape, head_first)
  kernel = functools.partial(_gdn_ut_fwd_kernel, chunk_size=chunk_size, compute_dtype=compute_dtype, head_first=head_first)
  return pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[chunk_spec(d_k), chunk_spec(d_v), vec_spec, vec_spec],
      out_specs=[chunk_spec(d_k), chunk_spec(d_v)],
      out_shape=[
          jax.ShapeDtypeStruct(k.shape, compute_dtype),
          jax.ShapeDtypeStruct(v.shape, compute_dtype),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel"),
      ),
      interpret=interpret,
      name="gdn_ut_fwd",
  )(k, v, beta[..., None], g[..., None])


def _ut_bwd_pallas(k, v, beta, g, dw, du, *, compute_dtype=jnp.bfloat16, head_first=True, interpret=False):
  chunk_size = k.shape[3] if head_first else k.shape[2]
  grid, chunk_spec, vec_spec, d_k, d_v = _ut_specs(k.shape, v.shape, head_first)
  kernel = functools.partial(_gdn_ut_bwd_kernel, chunk_size=chunk_size, compute_dtype=compute_dtype, head_first=head_first)
  # Gradients are accumulated in f32 in-kernel and stored in the primal
  # dtypes (identical to the former post-kernel casts, at half the VMEM
  # and HBM for the bf16 primals); dg stays f32 like g.
  dk, dv, dbeta, dg = pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[chunk_spec(d_k), chunk_spec(d_v), vec_spec, vec_spec, chunk_spec(d_k), chunk_spec(d_v)],
      out_specs=[chunk_spec(d_k), chunk_spec(d_v), vec_spec, vec_spec],
      out_shape=[
          jax.ShapeDtypeStruct(k.shape, k.dtype),
          jax.ShapeDtypeStruct(v.shape, v.dtype),
          jax.ShapeDtypeStruct(beta[..., None].shape, beta.dtype),
          jax.ShapeDtypeStruct(g[..., None].shape, jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel"),
      ),
      interpret=interpret,
      name="gdn_ut_bwd",
  )(k, v, beta[..., None], g[..., None], dw, du)
  return dk, dv, dbeta[..., 0], dg[..., 0]


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def gdn_ut_transform(k, v, beta, g, interpret=False, compute_dtype=jnp.bfloat16, head_first=True):
  """Fused UT transform: WY factors (w, u) from the chunked GDN inputs.

  Computes, per (batch, chunk, head):
    k_beta = k * beta[:, None]
    S = (k_beta @ k^T) * exp(g_i - g_j) on the strict lower triangle
    A = (I + S)^{-1}
    w = A @ (k_beta * exp(g)[:, None]) ;  u = A @ (v * beta[:, None])
  without materializing the [C, C] float32 S and A in HBM.

  Args:
    k: [B, N, H, C, D_k] chunked keys (compute dtype).
    v: [B, N, H, C, D_v] chunked values (compute dtype).
    beta: [B, N, H, C] update strengths (compute dtype).
    g: [B, N, H, C] per-chunk cumulative log-decay (float32).
    interpret: run the Pallas kernels in interpret mode (CPU testing).
    compute_dtype: operand dtype for the MXU matmuls (accumulation is f32;
      the inversion ladder always runs f32-grade via bf16x3).
    head_first: chunked operands use [B, N, H, C, D] / [B, N, H, C]. With
      head_first=False they instead use the transpose-free seq-major layout
      [B, N, C, H, D] / [B, N, C, H] (requires num_heads % 8 == 0).

  Returns:
    (w, u): shaped like k / v, in compute_dtype.
  """
  return _ut_fwd_pallas(k, v, beta, g, compute_dtype=compute_dtype, head_first=head_first, interpret=interpret)


def _gdn_ut_vjp_fwd(k, v, beta, g, interpret, compute_dtype, head_first):
  w, u = _ut_fwd_pallas(k, v, beta, g, compute_dtype=compute_dtype, head_first=head_first, interpret=interpret)
  # Residuals are just the (small) primal inputs: the backward kernel
  # recomputes S and A in VMEM instead of saving [C, C] tensors.
  return (w, u), (k, v, beta, g)


def _gdn_ut_vjp_bwd(interpret, compute_dtype, head_first, residuals, cotangents):
  k, v, beta, g = residuals
  dw, du = cotangents
  # Gradients are accumulated in float32 inside the kernel and stored in
  # the primal dtypes, so they come back ready to use as cotangents.
  dk, dv, dbeta, dg = _ut_bwd_pallas(
      k, v, beta, g, dw, du, compute_dtype=compute_dtype, head_first=head_first, interpret=interpret
  )
  return dk, dv, dbeta, dg.astype(g.dtype)


gdn_ut_transform.defvjp(_gdn_ut_vjp_fwd, _gdn_ut_vjp_bwd)


# =============================================================================
# Fully fused chunked gated delta rule
# =============================================================================
# Fuses the WY-factor precomputation (k_beta, decay-masked S, the blockwise
# unit-lower-triangular inversion, u and w) together with the inter-chunk
# recurrence in a single kernel per (batch, head) program. This removes the
# HBM round-trips between the many small XLA ops that the two-stage
# implementation pays per chunk.


def _tril_mask_strict(chunk_size: int) -> jax.Array:
  rows = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 0)
  cols = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 1)
  return rows > cols


def _invert_unit_lower(s: jax.Array, base_block: int = 16) -> jax.Array:
  """(I + s)^{-1} for strictly lower-triangular s. In-kernel friendly."""
  size = s.shape[-1]
  if size <= base_block:
    identity = jnp.eye(size, dtype=jnp.float32)
    x = identity
    for _ in range(size - 1):
      x = identity - jnp.matmul(s, x, preferred_element_type=jnp.float32)
    return x
  half = size // 2
  a_inv = _invert_unit_lower(s[..., :half, :half], base_block)
  b_inv = _invert_unit_lower(s[..., half:, half:], base_block)
  c = s[..., half:, :half]
  off = -jnp.matmul(b_inv, jnp.matmul(c, a_inv, preferred_element_type=jnp.float32), preferred_element_type=jnp.float32)
  top = jnp.concatenate([a_inv, jnp.zeros((size - half, half), jnp.float32).T * 0 + jnp.zeros((half, size - half), jnp.float32)], axis=-1)
  bottom = jnp.concatenate([off, b_inv], axis=-1)
  return jnp.concatenate([top, bottom], axis=-2)


def _wy_factors(q, k, v, g, beta, chunk_size, compute_dtype):
  """Shared stage-2 math: returns (w, u, exp_g, decay_incl, mask_strict, k_beta)."""
  del q  # unused here; kept for signature clarity
  exp_g = jnp.exp(g)
  k_beta = k * beta[:, None]
  mask_strict = _tril_mask_strict(chunk_size)
  g_diff = g[:, None] - g[None, :]
  decay_strict = jnp.where(mask_strict, jnp.exp(jnp.where(mask_strict, g_diff, 0.0)), 0.0)
  s = jnp.matmul(
      k_beta.astype(compute_dtype), k.astype(compute_dtype).T, preferred_element_type=jnp.float32
  ) * decay_strict
  a_inv = _invert_unit_lower(s)
  v_beta = v * beta[:, None]
  k_beta_g = k_beta * exp_g[:, None]
  # Merged matmul: A @ [k_beta_g | v_beta] doubles the RHS width for the MXU.
  d_k = k.shape[-1]
  wu = jnp.matmul(
      a_inv.astype(compute_dtype),
      jnp.concatenate([k_beta_g, v_beta], axis=-1).astype(compute_dtype),
      preferred_element_type=jnp.float32,
  )
  w, u = wu[:, :d_k], wu[:, d_k:]
  return w, u, exp_g, decay_strict, mask_strict, k_beta, a_inv, v_beta, k_beta_g


def _gdn_fused_fwd_kernel(
    q_ref, k_ref, v_ref, g_ref, beta_ref,
    o_ref, h_saved_ref, h_final_ref,
    h_scratch,
    *, chunk_size: int, num_chunks: int, compute_dtype: jnp.dtype,
):
  n = pl.program_id(2)

  @pl.when(n == 0)
  def _init():
    h_scratch[...] = jnp.zeros_like(h_scratch)

  h = h_scratch[0, 0, 0]
  h_saved_ref[0, 0, 0] = h

  q = q_ref[0, 0, 0].astype(jnp.float32)
  k = k_ref[0, 0, 0].astype(jnp.float32)
  v = v_ref[0, 0, 0].astype(jnp.float32)
  g = g_ref[0, 0, 0, :, 0].astype(jnp.float32)
  beta = beta_ref[0, 0, 0, :, 0].astype(jnp.float32)

  def mxu(a, b):
    return jnp.matmul(a.astype(compute_dtype), b.astype(compute_dtype), preferred_element_type=jnp.float32)

  # ---- Stage 2: WY factors (fused, VMEM resident) ----
  w, u, exp_g, _, _, _, _, _, _ = _wy_factors(q, k, v, g, beta, chunk_size, compute_dtype)

  # ---- Stage 3: recurrence ----
  q_g = q * exp_g[:, None]
  # Merged matmul: [q_g ; w] @ h doubles the LHS rows for the MXU.
  both = mxu(jnp.concatenate([q_g, w], axis=0), h)
  attn_inter, v_prime = both[:chunk_size], both[chunk_size:]
  v_new = u - v_prime

  rows = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 0)
  cols = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 1)
  mask_incl = rows >= cols
  g_diff = g[:, None] - g[None, :]
  decay_incl = jnp.where(mask_incl, jnp.exp(jnp.where(mask_incl, g_diff, 0.0)), 0.0)
  attn = mxu(q, k.T) * decay_incl

  o_ref[0, 0, 0] = (attn_inter + mxu(attn, v_new)).astype(o_ref.dtype)

  g_last = g[chunk_size - 1]
  gamma = jnp.exp(g_last)
  dvec = jnp.exp(g_last - g)
  kd = k * dvec[:, None]
  h_new = h * gamma + mxu(kd.T, v_new)
  h_scratch[0, 0, 0] = h_new

  @pl.when(n == num_chunks - 1)
  def _final():
    h_final_ref[0, 0] = h_new


def _gdn_fused_bwd_kernel(
    q_ref, k_ref, v_ref, g_ref, beta_ref, h_saved_ref, do_ref,
    dq_ref, dk_ref, dv_ref, dg_ref, dbeta_ref,
    dh_scratch,
    *, chunk_size: int, num_chunks: int, compute_dtype: jnp.dtype,
):
  n = pl.program_id(2)

  @pl.when(n == 0)
  def _init():
    dh_scratch[...] = jnp.zeros_like(dh_scratch)

  dh_next = dh_scratch[0, 0, 0]

  q = q_ref[0, 0, 0].astype(jnp.float32)
  k = k_ref[0, 0, 0].astype(jnp.float32)
  v = v_ref[0, 0, 0].astype(jnp.float32)
  g = g_ref[0, 0, 0, :, 0].astype(jnp.float32)
  beta = beta_ref[0, 0, 0, :, 0].astype(jnp.float32)
  h = h_saved_ref[0, 0, 0].astype(jnp.float32)
  do = do_ref[0, 0, 0].astype(jnp.float32)

  def mxu(a, b):
    return jnp.matmul(a.astype(compute_dtype), b.astype(compute_dtype), preferred_element_type=jnp.float32)

  # ---- Recompute stage-2 forward (A kept for its VJP) ----
  (w, u, exp_g, decay_strict, mask_strict, k_beta, a_inv, v_beta, k_beta_g) = _wy_factors(
      q, k, v, g, beta, chunk_size, compute_dtype
  )

  # ---- Recompute stage-3 forward intermediates ----
  q_g = q * exp_g[:, None]
  v_prime = mxu(w, h)
  v_new = u - v_prime
  rows = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 0)
  cols = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 1)
  mask_incl = rows >= cols
  g_diff = g[:, None] - g[None, :]
  decay_incl = jnp.where(mask_incl, jnp.exp(jnp.where(mask_incl, g_diff, 0.0)), 0.0)
  p_incl = mxu(q, k.T)
  s_incl = p_incl * decay_incl
  g_last = g[chunk_size - 1]
  gamma = jnp.exp(g_last)
  dvec = jnp.exp(g_last - g)
  kd = k * dvec[:, None]

  # ---- Stage-3 backward (mirrors the validated inter-chunk bwd) ----
  dgamma = jnp.sum(h * dh_next)
  dkd = mxu(v_new, dh_next.T)
  dv_new = mxu(kd, dh_next)
  dh = dh_next * gamma

  dk = dkd * dvec[:, None]
  ddvec = jnp.sum(dkd * k, axis=1)

  ds_incl = mxu(do, v_new.T)
  ds_incl = jnp.where(mask_incl, ds_incl, 0.0)
  dv_new = dv_new + mxu(s_incl.T, do)
  dp_incl = ds_incl * decay_incl
  ddecay_incl = ds_incl * p_incl
  dq = mxu(dp_incl, k)
  dk = dk + mxu(dp_incl.T, q)
  dgd_incl = ddecay_incl * decay_incl
  dg = jnp.sum(dgd_incl, axis=1) - jnp.sum(dgd_incl, axis=0)

  # Merged: dq_g and dw share the RHS h^T; the two dh contributions share
  # a stacked contraction over 2C rows.
  dq_g_dw = mxu(jnp.concatenate([do, dv_new], axis=0), h.T)
  dq_g, dw = dq_g_dw[:chunk_size], -dq_g_dw[chunk_size:]
  dh = dh + mxu(
      jnp.concatenate([q_g, -w], axis=0).T, jnp.concatenate([do, dv_new], axis=0)
  )

  du = dv_new

  dq = dq + dq_g * exp_g[:, None]
  dg = dg + jnp.sum(dq_g * q, axis=1) * exp_g

  dg = dg - ddvec * dvec
  dg_last_extra = jnp.sum(ddvec * dvec) + dgamma * gamma

  # ---- Stage-2 backward ----
  # w = A @ k_beta_g ; u = A @ v_beta   (merged pairs for wider matmuls)
  d_k_dim = k.shape[-1]
  da = mxu(jnp.concatenate([dw, du], axis=-1), jnp.concatenate([k_beta_g, v_beta], axis=-1).T)
  dwu = mxu(a_inv.T, jnp.concatenate([dw, du], axis=-1))
  dk_beta_g, dv_beta = dwu[:, :d_k_dim], dwu[:, d_k_dim:]

  # A = (I + S)^{-1}  =>  dS = -A^T dA A^T, projected to the strict lower part
  ds = -mxu(a_inv.T, mxu(da, a_inv.T))
  ds = jnp.where(mask_strict, ds, 0.0)

  # S = (k_beta @ k^T) * decay_strict
  p_strict = mxu(k_beta, k.T)
  dp_strict = ds * decay_strict
  ddecay_strict = ds * p_strict
  dk_beta = mxu(dp_strict, k)
  dk = dk + mxu(dp_strict.T, k_beta)
  dgd_strict = ddecay_strict * decay_strict
  dg = dg + jnp.sum(dgd_strict, axis=1) - jnp.sum(dgd_strict, axis=0)

  # k_beta_g = k_beta * exp(g)
  dk_beta = dk_beta + dk_beta_g * exp_g[:, None]
  dg = dg + jnp.sum(dk_beta_g * k_beta, axis=1) * exp_g

  # v_beta = v * beta ; k_beta = k * beta
  dv = dv_beta * beta[:, None]
  dbeta = jnp.sum(dv_beta * v, axis=1)
  dk = dk + dk_beta * beta[:, None]
  dbeta = dbeta + jnp.sum(dk_beta * k, axis=1)

  one_hot_last = (jax.lax.broadcasted_iota(jnp.int32, (chunk_size,), 0) == chunk_size - 1).astype(jnp.float32)
  dg = dg + dg_last_extra * one_hot_last

  dq_ref[0, 0, 0] = dq.astype(dq_ref.dtype)
  dk_ref[0, 0, 0] = dk.astype(dk_ref.dtype)
  dv_ref[0, 0, 0] = dv.astype(dv_ref.dtype)
  dg_ref[0, 0, 0, :, 0] = dg.astype(dg_ref.dtype)
  dbeta_ref[0, 0, 0, :, 0] = dbeta.astype(dbeta_ref.dtype)

  dh_scratch[0, 0, 0] = dh


def _fused_fwd_pallas(q, k, v, g, beta, *, compute_dtype=jnp.bfloat16, interpret=False):
  batch, num_chunks, num_heads, chunk_size, d_k = q.shape
  d_v = v.shape[-1]
  grid = (batch, num_heads, num_chunks)
  chunk_spec = lambda d: pl.BlockSpec((1, 1, 1, chunk_size, d), lambda b, h, n: (b, n, h, 0, 0))
  vec_spec = pl.BlockSpec((1, 1, 1, chunk_size, 1), lambda b, h, n: (b, n, h, 0, 0))
  state_spec = pl.BlockSpec((1, 1, 1, d_k, d_v), lambda b, h, n: (b, n, h, 0, 0))
  final_spec = pl.BlockSpec((1, 1, d_k, d_v), lambda b, h, n: (b, h, 0, 0))

  kernel = functools.partial(
      _gdn_fused_fwd_kernel, chunk_size=chunk_size, num_chunks=num_chunks, compute_dtype=compute_dtype
  )
  o, h_saved, h_final = pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[chunk_spec(d_k), chunk_spec(d_k), chunk_spec(d_v), vec_spec, vec_spec],
      out_specs=[chunk_spec(d_v), state_spec, final_spec],
      out_shape=[
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, d_v), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, d_k, d_v), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_heads, d_k, d_v), jnp.float32),
      ],
      scratch_shapes=[pltpu.VMEM((1, 1, 1, d_k, d_v), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
      ),
      interpret=interpret,
      name="gdn_fused_fwd",
  )(q, k, v, g[..., None], beta[..., None])
  return o, h_saved, h_final


def _fused_bwd_pallas(q, k, v, g, beta, h_saved, do, *, compute_dtype=jnp.bfloat16, interpret=False):
  batch, num_chunks, num_heads, chunk_size, d_k = q.shape
  d_v = v.shape[-1]
  grid = (batch, num_heads, num_chunks)
  rev = lambda b, h, n: (b, num_chunks - 1 - n, h, 0, 0)
  chunk_spec = lambda d: pl.BlockSpec((1, 1, 1, chunk_size, d), rev)
  vec_spec = pl.BlockSpec((1, 1, 1, chunk_size, 1), rev)
  state_spec = pl.BlockSpec((1, 1, 1, d_k, d_v), rev)

  kernel = functools.partial(
      _gdn_fused_bwd_kernel, chunk_size=chunk_size, num_chunks=num_chunks, compute_dtype=compute_dtype
  )
  dq, dk, dv, dg, dbeta = pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[chunk_spec(d_k), chunk_spec(d_k), chunk_spec(d_v), vec_spec, vec_spec, state_spec, chunk_spec(d_v)],
      out_specs=[chunk_spec(d_k), chunk_spec(d_k), chunk_spec(d_v), vec_spec, vec_spec],
      out_shape=[
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, d_k), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, d_k), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, d_v), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, 1), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, 1), jnp.float32),
      ],
      scratch_shapes=[pltpu.VMEM((1, 1, 1, d_k, d_v), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
      ),
      interpret=interpret,
      name="gdn_fused_bwd",
  )(q, k, v, g[..., None], beta[..., None], h_saved, do)
  return dq, dk, dv, dg[..., 0], dbeta[..., 0]


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def gdn_chunked_delta(q, k, v, g, beta, interpret=False, compute_dtype=jnp.bfloat16):
  """Fully fused chunked gated delta rule.

  Args:
    q, k: [B, N, H, C, D_k] chunked queries/keys (q pre-scaled).
    v: [B, N, H, C, D_v] chunked values.
    g: [B, N, H, C] per-chunk cumulative log-decay (float32).
    beta: [B, N, H, C] update strengths.

  Returns:
    (o, h_final): [B, N, H, C, D_v] float32 outputs and final state.
  """
  o, _, h_final = _fused_fwd_pallas(q, k, v, g, beta, compute_dtype=compute_dtype, interpret=interpret)
  return o, h_final


def _gdn_fused_vjp_fwd(q, k, v, g, beta, interpret, compute_dtype):
  o, h_saved, h_final = _fused_fwd_pallas(q, k, v, g, beta, compute_dtype=compute_dtype, interpret=interpret)
  return (o, h_final), (q, k, v, g, beta, h_saved)


def _gdn_fused_vjp_bwd(interpret, compute_dtype, residuals, cotangents):
  q, k, v, g, beta, h_saved = residuals
  do, dh_final = cotangents
  dq, dk, dv, dg, dbeta = _fused_bwd_pallas(
      q, k, v, g, beta, h_saved, do, compute_dtype=compute_dtype, interpret=interpret
  )

  def state_chain(q_, k_, v_, g_, beta_):
    # Pure-JAX state-only replay (autodiff-differentiable); mirrors the
    # kernel's stage-2 + state-update math without the output path.
    del q_
    chunk_size = k_.shape[-2]
    mask_strict = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool)).T & ~jnp.eye(chunk_size, dtype=bool)
    exp_g = jnp.exp(g_)
    k_beta = k_ * beta_[..., None]
    g_diff = g_[..., :, None] - g_[..., None, :]
    dec_s = jnp.where(mask_strict, jnp.exp(jnp.where(mask_strict, g_diff, 0.0)), 0.0)
    s_mat = jnp.einsum("bnhcd,bnhed->bnhce", k_beta, k_) * dec_s
    identity = jnp.eye(chunk_size)
    a_inv = jnp.linalg.inv(identity + s_mat)
    u_ = jnp.einsum("bnhce,bnhed->bnhcd", a_inv, v_ * beta_[..., None])
    w_ = jnp.einsum("bnhce,bnhed->bnhcd", a_inv, k_beta * exp_g[..., None])

    def step(h, xs):
      w_c, u_c, k_c, g_c = xs
      v_new = u_c - jnp.einsum("bhcd,bhde->bhce", w_c, h)
      g_lastc = g_c[..., -1]
      kd = k_c * jnp.exp(g_lastc[..., None] - g_c)[..., None]
      h_new = h * jnp.exp(g_lastc)[..., None, None] + jnp.einsum("bhcd,bhce->bhde", kd, v_new)
      return h_new, None

    xs = tuple(jnp.moveaxis(a, 1, 0) for a in (w_, u_, k_, g_))
    h0 = jnp.zeros((k_.shape[0], k_.shape[2], k_.shape[-1], v_.shape[-1]), jnp.float32)
    hf, _ = jax.lax.scan(step, h0, xs)
    return hf

  needs_final = jnp.any(dh_final != 0)

  def add_final(args):
    dq_, dk_, dv_, dg_, dbeta_ = args
    _, vjp_fn = jax.vjp(lambda *a: state_chain(*a), q, k, v, g, beta)
    extras = vjp_fn(dh_final)
    return tuple(x + e for x, e in zip(args, extras))

  dq, dk, dv, dg, dbeta = jax.lax.cond(needs_final, add_final, lambda a: a, (dq, dk, dv, dg, dbeta))
  return dq, dk, dv, dg, dbeta


gdn_chunked_delta.defvjp(_gdn_fused_vjp_fwd, _gdn_fused_vjp_bwd)

