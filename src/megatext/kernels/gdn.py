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
    w_ref, u_ref, q_ref, k_ref, g_ref,
    o_ref, h_saved_ref, h_final_ref,
    h_scratch,
    *, chunk_size: int, num_chunks: int, compute_dtype: jnp.dtype,
):
  """Forward inter-chunk recurrence for one (batch, head) program."""
  n = pl.program_id(2)

  @pl.when(n == 0)
  def _init():
    h_scratch[...] = jnp.zeros_like(h_scratch)

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
  attn_inter = mxu_dot(q_g, h)

  v_prime = mxu_dot(w, h)
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
    w_ref, u_ref, q_ref, k_ref, g_ref, h_saved_ref, do_ref,
    dw_ref, du_ref, dq_ref, dk_ref, dg_ref,
    dh_scratch,
    *, chunk_size: int, num_chunks: int, compute_dtype: jnp.dtype,
):
  """Reverse inter-chunk recurrence: walks chunks from last to first."""
  n = pl.program_id(2)

  @pl.when(n == 0)
  def _init():
    dh_scratch[...] = jnp.zeros_like(dh_scratch)

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


def _fwd_pallas(w, u, q, k, g, *, compute_dtype=jnp.bfloat16, interpret=False):
  """Runs the forward kernel. Inputs are [B, N, H, C, D] / g [B, N, H, C]."""
  batch, num_chunks, num_heads, chunk_size, d_k = q.shape
  d_v = u.shape[-1]

  grid = (batch, num_heads, num_chunks)
  chunk_spec = lambda d: pl.BlockSpec((1, 1, 1, chunk_size, d), lambda b, h, n: (b, n, h, 0, 0))
  # TPU block tiling requires the last two dims to be (8k, 128k) or equal to
  # the array dims; per-chunk vectors ride along with a trailing singleton.
  g_spec = pl.BlockSpec((1, 1, 1, chunk_size, 1), lambda b, h, n: (b, n, h, 0, 0))
  state_spec = pl.BlockSpec((1, 1, 1, d_k, d_v), lambda b, h, n: (b, n, h, 0, 0))
  final_spec = pl.BlockSpec((1, 1, d_k, d_v), lambda b, h, n: (b, h, 0, 0))

  g = g[..., None]
  kernel = functools.partial(_gdn_scan_fwd_kernel, chunk_size=chunk_size, num_chunks=num_chunks, compute_dtype=compute_dtype)
  o, h_saved, h_final = pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[chunk_spec(d_k), chunk_spec(d_v), chunk_spec(d_k), chunk_spec(d_k), g_spec],
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
      name="gdn_scan_fwd",
  )(w, u, q, k, g)
  return o, h_saved, h_final


def _bwd_pallas(w, u, q, k, g, h_saved, do, *, compute_dtype=jnp.bfloat16, interpret=False):
  """Runs the backward kernel (reverse chunk walk via index remapping)."""
  batch, num_chunks, num_heads, chunk_size, d_k = q.shape
  d_v = u.shape[-1]

  grid = (batch, num_heads, num_chunks)
  # Reverse the chunk dimension: grid step n touches chunk (N-1-n).
  rev = lambda b, h, n: (b, num_chunks - 1 - n, h, 0, 0)
  chunk_spec = lambda d: pl.BlockSpec((1, 1, 1, chunk_size, d), rev)
  g_spec = pl.BlockSpec((1, 1, 1, chunk_size, 1), rev)
  state_spec = pl.BlockSpec((1, 1, 1, d_k, d_v), rev)

  g = g[..., None]
  kernel = functools.partial(_gdn_scan_bwd_kernel, chunk_size=chunk_size, num_chunks=num_chunks, compute_dtype=compute_dtype)
  dw, du, dq, dk, dg = pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[chunk_spec(d_k), chunk_spec(d_v), chunk_spec(d_k), chunk_spec(d_k), g_spec, state_spec, chunk_spec(d_v)],
      out_specs=[chunk_spec(d_k), chunk_spec(d_v), chunk_spec(d_k), chunk_spec(d_k), g_spec],
      out_shape=[
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, d_k), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, d_v), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, d_k), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, d_k), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, 1), jnp.float32),
      ],
      scratch_shapes=[pltpu.VMEM((1, 1, 1, d_k, d_v), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
      ),
      interpret=interpret,
      name="gdn_scan_bwd",
  )(w, u, q, k, g, h_saved, do)
  return dw, du, dq, dk, dg[..., 0]


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def gdn_inter_chunk_scan(w, u, q, k, g, interpret=False, compute_dtype=jnp.bfloat16):
  """Fused inter-chunk gated-delta-rule scan.

  Args:
    w, u: WY factors, [B, N, H, C, D_k] / [B, N, H, C, D_v].
    q, k: chunked queries/keys, [B, N, H, C, D_k].
    g: per-chunk cumulative log-decay, [B, N, H, C] (float32).
    interpret: run the Pallas kernels in interpret mode (CPU testing).

  Returns:
    (o, h_final): chunk outputs [B, N, H, C, D_v] (float32) and the final
    recurrent state [B, H, D_k, D_v] (float32).
  """
  o, _, h_final = _fwd_pallas(w, u, q, k, g, compute_dtype=compute_dtype, interpret=interpret)
  return o, h_final


def _gdn_scan_vjp_fwd(w, u, q, k, g, interpret, compute_dtype):
  o, h_saved, h_final = _fwd_pallas(w, u, q, k, g, compute_dtype=compute_dtype, interpret=interpret)
  return (o, h_final), (w, u, q, k, g, h_saved)


def _gdn_scan_vjp_bwd(interpret, compute_dtype, residuals, cotangents):
  w, u, q, k, g, h_saved = residuals
  do, dh_final = cotangents
  # dh_final feeds the last chunk's state cotangent. The backward kernel
  # initializes dh to zero at the reverse walk start; fold dh_final in by
  # seeding through an extra virtual contribution on the last chunk.
  # We implement it by adding dh_final via the kernel's zero-init replacement:
  # simplest correct route — absorb dh_final analytically before the kernel.
  # h_final = gamma_last*h_last + kd_last^T @ vnew_last, which the kernel
  # differentiates when dh carries dh_final at reverse step 0. We achieve
  # this by augmenting `do` of the last chunk is NOT equivalent; instead we
  # run the kernel with dh seeded via a zero-padded extra chunk trick.
  # For training use dh_final is zero (the final state is unused), so we
  # assert-free fall back: when dh_final is symbolically zero this is exact.
  dw, du, dq, dk, dg = _bwd_pallas(w, u, q, k, g, h_saved, do, compute_dtype=compute_dtype, interpret=interpret)

  # Correction for a non-zero dh_final: propagate it analytically through
  # the pure-JAX reference of the state chain (cheap: state-only recurrence).
  def state_chain(w_, u_, k_, g_):
    b, n_chunks, h_heads, c, dk_ = k_.shape
    def step(h, xs):
      w_c, u_c, k_c, g_c = xs
      v_new = u_c - jnp.einsum("bhcd,bhde->bhce", w_c, h)
      g_last = g_c[..., -1]
      dvec = jnp.exp(g_last[..., None] - g_c)
      kd = k_c * dvec[..., None]
      h_new = h * jnp.exp(g_last)[..., None, None] + jnp.einsum("bhcd,bhce->bhde", kd, v_new)
      return h_new, None
    xs = (
        jnp.moveaxis(w_, 1, 0), jnp.moveaxis(u_, 1, 0),
        jnp.moveaxis(k_, 1, 0), jnp.moveaxis(g_, 1, 0),
    )
    h0 = jnp.zeros((b, h_heads, dk_, u_.shape[-1]), jnp.float32)
    h_fin, _ = jax.lax.scan(step, h0, xs)
    return h_fin

  needs_final = jnp.any(dh_final != 0)
  def add_final_grads(args):
    dw_, du_, dq_, dk_, dg_ = args
    _, vjp_fn = jax.vjp(state_chain, w, u, k, g)
    dw2, du2, dk2, dg2 = vjp_fn(dh_final)
    return dw_ + dw2, du_ + du2, dq_, dk_ + dk2, dg_ + dg2
  dw, du, dq, dk, dg = jax.lax.cond(
      needs_final, add_final_grads, lambda a: a, (dw, du, dq, dk, dg)
  )
  return dw, du, dq, dk, dg


gdn_inter_chunk_scan.defvjp(_gdn_scan_vjp_fwd, _gdn_scan_vjp_bwd)
