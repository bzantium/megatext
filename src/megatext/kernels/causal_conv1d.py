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

"""Pallas TPU kernel for the fused causal depthwise Conv1D + SiLU.

The Qwen3-Next gated delta net runs its qkv activations through a short
(window 4) depthwise causal convolution followed by SiLU. Expressed as
`nnx.Conv` + `jax.nn.silu`, XLA lowers this to a convolution plus separate
elementwise fusions with an HBM round-trip of the [B, S, 2*K+V]
pre-activation between them — pure VPU work that is entirely
memory-bound.

This module fuses the taps and the SiLU into one Pallas pass:

  out[t, f] = silu(sum_{i<W} x[t - (W-1) + i, f] * w[i, f])

The grid is (batch, feature_blocks, seq_blocks) with the sequence
dimension innermost and marked "arbitrary", so each (batch, feature)
program walks its sequence blocks in order. The W-1 rows of left context
a block needs from its predecessor (the halo) are carried in a small VMEM
scratch buffer: each step consumes the carry, computes its block, and
stores its own last W-1 rows for the next step — no overlapping block
reads and no padded copy of x in HBM.

The backward pass is pure JAX (shifted adds, no conv primitive): the
fused forward is the win, while the backward is a handful of cheap
elementwise ops that XLA fuses well on its own. pre is recomputed from
the (x, w) residuals, then

  dpre[t]  = do[t] * silu'(pre[t])
  dx[t]    = sum_i dpre[t + (W-1) - i] * w[i]
  dw[i, f] = sum_{b, t} dpre[t, f] * x[t - (W-1) + i, f]

I/O is in the compute dtype (e.g. bf16); all accumulation is float32.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _conv_silu_fwd_kernel(x_ref, w_ref, out_ref, carry, *, window: int):
  """One (batch, feature-block) step over one sequence block."""
  s = pl.program_id(2)

  @pl.when(s == 0)
  def _init():
    # Causal left padding: the first block sees zeros before t=0.
    carry[...] = jnp.zeros_like(carry)

  x = x_ref[0].astype(jnp.float32)  # [block_s, block_f]
  w = w_ref[...].astype(jnp.float32)  # [window, block_f]
  block_s = x.shape[0]

  # [block_s + window - 1, block_f]: halo rows from the previous block,
  # then this block. Row t+window-1 of xc is x[t].
  xc = jnp.concatenate([carry[...], x], axis=0)

  # Elementwise multiply-accumulate over the taps (VPU, f32 accumulation).
  pre = xc[0:block_s] * w[0]
  for i in range(1, window):
    pre = pre + xc[i : i + block_s] * w[i]

  out_ref[0] = (pre * jax.nn.sigmoid(pre)).astype(out_ref.dtype)

  # Halo for the next sequence block: our last window-1 input rows.
  carry[...] = xc[block_s:]


def _block_sizes(seq_len: int, features: int) -> tuple[int, int]:
  """Sequence/feature block sizes honoring the (8, 128) TPU tile rule."""
  block_s = seq_len
  for cand in (512, 256, 128, 64, 32, 16, 8):
    if seq_len % cand == 0:
      block_s = cand
      break
  block_f = features
  if features % 128 == 0:
    for cand in (1024, 512, 256, 128):
      if features % cand == 0:
        block_f = cand
        break
  return block_s, block_f


def _fwd_pallas(x, w, *, interpret=False):
  """x [B, S, F] (compute dtype), w [window, F] -> silu(conv(x)) [B, S, F]."""
  batch, seq_len, features = x.shape
  window = w.shape[0]

  # Pad the sequence up to a block multiple; the conv is causal, so extra
  # trailing rows never influence the rows we keep.
  block_s, block_f = _block_sizes(seq_len, features)
  pad_len = (block_s - seq_len % block_s) % block_s
  if pad_len > 0:
    x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0)))

  grid = (batch, features // block_f, x.shape[1] // block_s)
  out = pl.pallas_call(
      functools.partial(_conv_silu_fwd_kernel, window=window),
      grid=grid,
      in_specs=[
          pl.BlockSpec((1, block_s, block_f), lambda b, f, s: (b, s, f)),
          pl.BlockSpec((window, block_f), lambda b, f, s: (0, f)),
      ],
      out_specs=pl.BlockSpec((1, block_s, block_f), lambda b, f, s: (b, s, f)),
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      scratch_shapes=[pltpu.VMEM((window - 1, block_f), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
      ),
      interpret=interpret,
      name="causal_conv1d_silu_fwd",
  )(x, w)
  if pad_len > 0:
    out = out[:, :seq_len, :]
  return out


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def causal_conv1d_silu(x, w, interpret=False):
  """Fused causal depthwise Conv1D + SiLU.

  Args:
    x: [B, S, F] input activations (compute dtype, e.g. bf16).
    w: [window, F] depthwise taps (tap 0 is the oldest input row).
    interpret: run the Pallas kernel in interpret mode (CPU testing).

  Returns:
    silu(causal_depthwise_conv(x, w)): [B, S, F] in x.dtype.
  """
  return _fwd_pallas(x, w, interpret=interpret)


def _conv_silu_vjp_fwd(x, w, interpret):
  return _fwd_pallas(x, w, interpret=interpret), (x, w)


def _conv_silu_vjp_bwd(interpret, residuals, do):
  del interpret
  x, w = residuals
  window = w.shape[0]
  seq_len = x.shape[1]

  xf = x.astype(jnp.float32)
  wf = w.astype(jnp.float32)
  dof = do.astype(jnp.float32)

  # Recompute pre = conv(x, w) with shifted adds; x_pad row t+i is x[t-(W-1)+i].
  x_pad = jnp.pad(xf, ((0, 0), (window - 1, 0), (0, 0)))
  pre = sum(x_pad[:, i : i + seq_len, :] * wf[i] for i in range(window))

  # silu'(p) = sigmoid(p) * (1 + p * (1 - sigmoid(p)))
  sig = jax.nn.sigmoid(pre)
  dpre = dof * sig * (1.0 + pre * (1.0 - sig))

  # dx[t] = sum_i dpre[t + (W-1) - i] * w[i]; pad dpre on the right so the
  # shifts stay in range (dpre is zero past the sequence end).
  dpre_pad = jnp.pad(dpre, ((0, 0), (0, window - 1), (0, 0)))
  dx = sum(dpre_pad[:, window - 1 - i : window - 1 - i + seq_len, :] * wf[i] for i in range(window))

  # dw[i, f] = sum_{b, t} dpre[t, f] * x[t - (W-1) + i, f]
  dw = jnp.stack([jnp.sum(dpre * x_pad[:, i : i + seq_len, :], axis=(0, 1)) for i in range(window)])

  return dx.astype(x.dtype), dw.astype(w.dtype)


causal_conv1d_silu.defvjp(_conv_silu_vjp_fwd, _conv_silu_vjp_bwd)
