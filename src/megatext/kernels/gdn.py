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


def _tril_mask(chunk_size: int, include_diag: bool = True) -> jax.Array:
  rows = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 0)
  cols = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 1)
  return rows >= cols if include_diag else rows > cols


def _gdn_scan_fwd_kernel(
    w_ref, u_ref, q_ref, k_ref, g_ref, h0_ref,
    o_ref, h_saved_ref, h_final_ref,
    h_scratch,
    *, chunk_size: int, num_chunks: int, compute_dtype: jnp.dtype,
):
  """Forward inter-chunk recurrence for one (batch, head) program."""
  n = pl.program_id(2)

  @pl.when(n == 0)
  def _init():
    h_scratch[0, 0, 0] = h0_ref[0, 0].astype(jnp.float32)

  h = h_scratch[0, 0, 0]
  # Save the chunk-input state for the backward pass.
  h_saved_ref[0, 0, 0] = h

  # Tokamax-style MXU dots: operands in the compute dtype (bf16), float32
  # accumulation via preferred_element_type. State and decay math stay f32.
  def mxu_dot(a, b):
    return jax.lax.dot(
        a.astype(compute_dtype), b.astype(compute_dtype), preferred_element_type=jnp.float32
    )

  w = w_ref[0, 0, 0]
  u = u_ref[0, 0, 0].astype(jnp.float32)
  q = q_ref[0, 0, 0]
  k = k_ref[0, 0, 0]
  g = g_ref[0, 0, 0, :, 0].astype(jnp.float32)

  exp_g = jnp.exp(g)
  q_g = q.astype(jnp.float32) * exp_g[:, None]
  # Merged matmul: [q_g ; w] @ h doubles the LHS rows for the MXU.
  both = mxu_dot(jnp.concatenate([q_g, w.astype(jnp.float32)], axis=0), h)
  attn_inter, v_prime = both[:chunk_size], both[chunk_size:]
  v_new = u - v_prime

  p = mxu_dot(q, k.T)
  g_diff = g[:, None] - g[None, :]
  mask = _tril_mask(chunk_size)
  decay = jnp.where(mask, jnp.exp(jnp.where(mask, g_diff, 0.0)), 0.0)
  s = p * decay

  o_ref[0, 0, 0] = (attn_inter + mxu_dot(s, v_new)).astype(o_ref.dtype)

  # State update.
  g_last = g[chunk_size - 1]
  gamma = jnp.exp(g_last)
  dvec = jnp.exp(g_last - g)
  kd = k.astype(jnp.float32) * dvec[:, None]
  h_new = h * gamma + mxu_dot(kd.T, v_new)
  h_scratch[0, 0, 0] = h_new

  @pl.when(n == num_chunks - 1)
  def _final():
    h_final_ref[0, 0] = h_new


def _gdn_scan_bwd_kernel(
    w_ref, u_ref, q_ref, k_ref, g_ref, h_saved_ref, do_ref, dh_final_ref,
    dw_ref, du_ref, dq_ref, dk_ref, dg_ref, dh0_ref,
    dh_scratch,
    *, chunk_size: int, num_chunks: int, compute_dtype: jnp.dtype,
):
  """Reverse inter-chunk recurrence: walks chunks from last to first."""
  n = pl.program_id(2)

  @pl.when(n == 0)
  def _init():
    # Seed the state cotangent with the final-state cotangent.
    dh_scratch[0, 0, 0] = dh_final_ref[0, 0].astype(jnp.float32)

  dh_next = dh_scratch[0, 0, 0]  # dL/dh_{c+1} for the chunk being processed

  def mxu_dot(a, b):
    return jax.lax.dot(
        a.astype(compute_dtype), b.astype(compute_dtype), preferred_element_type=jnp.float32
    )

  w = w_ref[0, 0, 0].astype(jnp.float32)
  u = u_ref[0, 0, 0].astype(jnp.float32)
  q = q_ref[0, 0, 0].astype(jnp.float32)
  k = k_ref[0, 0, 0].astype(jnp.float32)
  g = g_ref[0, 0, 0, :, 0].astype(jnp.float32)
  h = h_saved_ref[0, 0, 0].astype(jnp.float32)
  do = do_ref[0, 0, 0].astype(jnp.float32)

  # --- Recompute forward intermediates for this chunk ---
  exp_g = jnp.exp(g)
  q_g = q * exp_g[:, None]
  v_prime = mxu_dot(w, h)
  v_new = u - v_prime
  p = mxu_dot(q, k.T)
  g_diff = g[:, None] - g[None, :]
  mask = _tril_mask(chunk_size)
  decay = jnp.where(mask, jnp.exp(jnp.where(mask, g_diff, 0.0)), 0.0)
  s = p * decay
  g_last = g[chunk_size - 1]
  gamma = jnp.exp(g_last)
  dvec = jnp.exp(g_last - g)
  kd = k * dvec[:, None]

  # --- Backward through the state update: h' = gamma*h + kd^T @ v_new ---
  dgamma = jnp.sum(h * dh_next)
  dkd = mxu_dot(v_new, dh_next.T)  # [C, D_k]
  dv_new = mxu_dot(kd, dh_next)   # [C, D_v]
  dh = dh_next * gamma

  dk = dkd * dvec[:, None]
  ddvec = jnp.sum(dkd * k, axis=1)

  # --- Backward through the output: o = q_g @ h + s @ v_new ---
  ds = mxu_dot(do, v_new.T)
  ds = jnp.where(mask, ds, 0.0)
  dv_new = dv_new + mxu_dot(s.T, do)
  dp = ds * decay
  ddecay = ds * p
  dq = mxu_dot(dp, k)
  dk = dk + mxu_dot(dp.T, q)
  # decay = exp(g_i - g_j) on the tril: dg_i += sum_j(ddecay*decay); dg_j -= sum_i(...)
  dgd = ddecay * decay
  dg = jnp.sum(dgd, axis=1) - jnp.sum(dgd, axis=0)

  dq_g = mxu_dot(do, h.T)
  dh = dh + mxu_dot(q_g.T, do)

  # --- Backward through v_new = u - w @ h ---
  du = dv_new
  dw = -mxu_dot(dv_new, h.T)
  dh = dh - mxu_dot(w.T, dv_new)

  # --- Backward through q_g = q * exp(g) ---
  dq = dq + dq_g * exp_g[:, None]
  dg = dg + jnp.sum(dq_g * q, axis=1) * exp_g

  # --- Decay-vector and gamma contributions to g ---
  # dvec = exp(g_last - g); gamma = exp(g_last)
  dg = dg - ddvec * dvec
  dg_last = jnp.sum(ddvec * dvec) + dgamma * gamma
  one_hot_last = (jax.lax.broadcasted_iota(jnp.int32, (chunk_size,), 0) == chunk_size - 1).astype(jnp.float32)
  dg = dg + dg_last * one_hot_last

  dw_ref[0, 0, 0] = dw.astype(dw_ref.dtype)
  du_ref[0, 0, 0] = du.astype(du_ref.dtype)
  dq_ref[0, 0, 0] = dq.astype(dq_ref.dtype)
  dk_ref[0, 0, 0] = dk.astype(dk_ref.dtype)
  dg_ref[0, 0, 0, :, 0] = dg.astype(dg_ref.dtype)

  dh_scratch[0, 0, 0] = dh

  @pl.when(n == num_chunks - 1)
  def _final():
    # After the reverse walk over chunk 0, dh is dL/dh_0.
    dh0_ref[0, 0] = dh


def _fwd_pallas(w, u, q, k, g, h0, *, compute_dtype=jnp.bfloat16, interpret=False):
  """Runs the forward kernel. Inputs are [B, N, H, C, D] / g [B, N, H, C] / h0 [B, H, D_k, D_v]."""
  batch, num_chunks, num_heads, chunk_size, d_k = q.shape
  d_v = u.shape[-1]

  grid = (batch, num_heads, num_chunks)
  chunk_spec = lambda d: pl.BlockSpec((1, 1, 1, chunk_size, d), lambda b, h, n: (b, n, h, 0, 0))
  # TPU block tiling requires the last two dims to be (8k, 128k) or equal to
  # the array dims; per-chunk vectors ride along with a trailing singleton.
  g_spec = pl.BlockSpec((1, 1, 1, chunk_size, 1), lambda b, h, n: (b, n, h, 0, 0))
  state_spec = pl.BlockSpec((1, 1, 1, d_k, d_v), lambda b, h, n: (b, n, h, 0, 0))
  bh_state_spec = pl.BlockSpec((1, 1, d_k, d_v), lambda b, h, n: (b, h, 0, 0))

  g = g[..., None]
  kernel = functools.partial(_gdn_scan_fwd_kernel, chunk_size=chunk_size, num_chunks=num_chunks, compute_dtype=compute_dtype)
  o, h_saved, h_final = pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[chunk_spec(d_k), chunk_spec(d_v), chunk_spec(d_k), chunk_spec(d_k), g_spec, bh_state_spec],
      out_specs=[chunk_spec(d_v), state_spec, bh_state_spec],
      out_shape=[
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, d_v), compute_dtype),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, d_k, d_v), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_heads, d_k, d_v), jnp.float32),
      ],
      scratch_shapes=[pltpu.VMEM((1, 1, 1, d_k, d_v), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
      ),
      interpret=interpret,
      name="gdn_scan_fwd",
  )(w, u, q, k, g, h0)
  return o, h_saved, h_final


def _bwd_pallas(w, u, q, k, g, h_saved, do, dh_final, *, compute_dtype=jnp.bfloat16, interpret=False):
  """Runs the backward kernel (reverse chunk walk via index remapping)."""
  batch, num_chunks, num_heads, chunk_size, d_k = q.shape
  d_v = u.shape[-1]

  grid = (batch, num_heads, num_chunks)
  # Reverse the chunk dimension: grid step n touches chunk (N-1-n).
  rev = lambda b, h, n: (b, num_chunks - 1 - n, h, 0, 0)
  chunk_spec = lambda d: pl.BlockSpec((1, 1, 1, chunk_size, d), rev)
  g_spec = pl.BlockSpec((1, 1, 1, chunk_size, 1), rev)
  state_spec = pl.BlockSpec((1, 1, 1, d_k, d_v), rev)
  bh_state_spec = pl.BlockSpec((1, 1, d_k, d_v), lambda b, h, n: (b, h, 0, 0))

  g = g[..., None]
  kernel = functools.partial(_gdn_scan_bwd_kernel, chunk_size=chunk_size, num_chunks=num_chunks, compute_dtype=compute_dtype)
  dw, du, dq, dk, dg, dh0 = pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[chunk_spec(d_k), chunk_spec(d_v), chunk_spec(d_k), chunk_spec(d_k), g_spec, state_spec, chunk_spec(d_v), bh_state_spec],
      out_specs=[chunk_spec(d_k), chunk_spec(d_v), chunk_spec(d_k), chunk_spec(d_k), g_spec, bh_state_spec],
      out_shape=[
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, d_k), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, d_v), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, d_k), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, d_k), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, 1), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_heads, d_k, d_v), jnp.float32),
      ],
      scratch_shapes=[pltpu.VMEM((1, 1, 1, d_k, d_v), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
      ),
      interpret=interpret,
      name="gdn_scan_bwd",
  )(w, u, q, k, g, h_saved, do, dh_final)
  return dw, du, dq, dk, dg[..., 0], dh0


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def gdn_inter_chunk_scan(w, u, q, k, g, h0, interpret=False, compute_dtype=jnp.bfloat16):
  """Fused inter-chunk gated-delta-rule scan.

  Args:
    w, u: WY factors, [B, N, H, C, D_k] / [B, N, H, C, D_v].
    q, k: chunked queries/keys, [B, N, H, C, D_k].
    g: per-chunk cumulative log-decay, [B, N, H, C] (float32).
    h0: initial recurrent state, [B, H, D_k, D_v] (float32).
    interpret: run the Pallas kernels in interpret mode (CPU testing).
    compute_dtype: operand dtype for the MXU matmuls (accumulation is f32).

  Returns:
    (o, h_final): chunk outputs [B, N, H, C, D_v] (float32) and the final
    recurrent state [B, H, D_k, D_v] (float32).
  """
  o, _, h_final = _fwd_pallas(w, u, q, k, g, h0, compute_dtype=compute_dtype, interpret=interpret)
  return o, h_final


def _gdn_scan_vjp_fwd(w, u, q, k, g, h0, interpret, compute_dtype):
  o, h_saved, h_final = _fwd_pallas(w, u, q, k, g, h0, compute_dtype=compute_dtype, interpret=interpret)
  return (o, h_final), (w, u, q, k, g, h_saved)


def _gdn_scan_vjp_bwd(interpret, compute_dtype, residuals, cotangents):
  w, u, q, k, g, h_saved = residuals
  do, dh_final = cotangents
  # The backward kernel seeds the reverse walk with dh_final and emits dh0,
  # so the state chain is differentiated end-to-end inside the kernel.
  dw, du, dq, dk, dg, dh0 = _bwd_pallas(
      w, u, q, k, g, h_saved, do, dh_final, compute_dtype=compute_dtype, interpret=interpret
  )
  # Gradients are accumulated in float32 inside the kernel; cotangents must
  # match the primal dtypes (inputs may arrive in bf16).
  return dw.astype(w.dtype), du.astype(u.dtype), dq.astype(q.dtype), dk.astype(k.dtype), dg.astype(g.dtype), dh0


gdn_inter_chunk_scan.defvjp(_gdn_scan_vjp_fwd, _gdn_scan_vjp_bwd)


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

