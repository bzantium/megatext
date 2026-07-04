"""Muon speed/memory optimizations for megatext.

Two pieces:

1. ``orthogonalize_via_newton_schulz``: drop-in replacement for optax's
   per-parameter Newton-Schulz orthogonalization. Kept as the
   ``muon_batched_ns=False`` fallback path (monkey-patched into
   ``optax.contrib._muon`` by ``optimizers.get_optimizer``).

2. ``scale_by_muon_batched``: drop-in replacement for
   ``optax.contrib._muon.scale_by_muon`` that buckets all Muon parameters
   whose reshaped matrix shape (rows, cols) is identical, concatenates them
   along the vmapped batch axis, and runs ONE Newton-Schulz chain per bucket
   instead of one per parameter tensor. For qwen3-next this collapses ~60
   sequential per-leaf NS chains into ~15 fused batched ones, letting XLA
   overlap the small matmuls and issue far fewer collectives.

   Optionally (when a ``mesh`` is provided and ``batch_reshard=True``), each
   stacked bucket is resharded ONCE onto the leading batch axis across the
   mesh so that every NS iteration's gram matmuls run fully locally on each
   device (no per-iteration collectives under FSDP); the result is resharded
   back by the consumer. This changes layout only, not math.

   ``batch_reshard`` defaults to OFF: the layout change right after the
   backward pass gives XLA's SPMD partitioner a second sharding "opinion" on
   gradient-derived tensors and inserts resharding collectives between the
   gradient all-reduce and its consumers, which can defeat the TPU
   data-parallel all-reduce optimization (LIBTPU
   xla_tpu_enable_data_parallel_all_reduce_opt) that overlaps the cross-slice
   (DCN) gradient reduce with backward compute on multi-slice runs. With it
   off, the NS matmuls simply run on whatever sharding the momentum tensors
   already have (per-iteration ICI collectives, like the pre-bucketed code).

   In BOTH modes, the nesterov momentum tensors feeding NS are pinned to the
   momentum state's own sharding (``jax.experimental.shard_alike``) before any
   reshape/concat, so the bucketing machinery can never back-propagate a
   different layout into the gradient all-reduce -> elementwise-update chain
   that XLA pattern-matches for DP overlap.

The per-matrix math is byte-for-byte the same computation as optax's
implementation (same frobenius preconditioning, same NS-5 iteration, same
dtypes), so updates are equivalent to the per-leaf version at f32 rounding
level and runs stay comparable to baselines trained with the current code.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax.experimental.shard_alike import shard_alike

import optax.tree
from optax._src import base
from optax._src import numerics
from optax._src import utils

from optax.contrib._muon import (
    MuonDimensionNumbers,
    MuonState,
    _DEFAULT_NS_COEFFS,
    _compute_muon_reshape,
    _is_weight_dim_nums,
    _base_ns_iterator,
    _aol_ns_iterator,
    _schatten_ns_iterator,
)

# Original optax implementation, captured before any monkey-patching so the
# `muon_batched_ns=False` fallback can restore it.
from optax.contrib._muon import scale_by_muon as OPTAX_SCALE_BY_MUON  # noqa: F401


_NS_ITERATORS = {
    "frobenius": _base_ns_iterator,
    "spectral": _base_ns_iterator,
    "aol": _aol_ns_iterator,
    "schatten": _schatten_ns_iterator,
}


def _validate_ns_coeffs(ns_coeffs: jax.Array) -> None:
    if ns_coeffs.ndim > 2 or ns_coeffs.shape[-1] != 3:
        raise ValueError(
            f"Newton-Schulz coefficients must have shape (3,) or (n, 3), "
            f"got {ns_coeffs.shape}"
        )


def _orthogonalize_2d(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: jax.typing.ArrayLike,
    preconditioning: str,
    eps: jax.typing.ArrayLike,
    unroll: bool = False,
) -> jax.Array:
    """Exact per-matrix math of optax's orthogonalize_via_newton_schulz.

    `unroll` is a pure scheduling knob (same op sequence either way):
    unroll=False lets XLA reuse buffers across NS iterations (lower
    compile-time HBM); unroll=True emits straight-line code XLA can overlap.
    """
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True

    if preconditioning not in _NS_ITERATORS:
        raise ValueError(f"Unknown preconditioning {preconditioning}")
    _ns_iterator = _NS_ITERATORS[preconditioning]

    if preconditioning == "frobenius":
        x /= jnp.linalg.norm(x, ord="fro") + eps
    elif preconditioning == "spectral":
        x /= jnp.linalg.norm(x, ord=2) + eps

    ns_coeffs_ = ns_coeffs.astype(x.dtype)

    if ns_coeffs_.ndim == 1:
        x = jax.lax.fori_loop(
            0, ns_steps,
            lambda i, x: _ns_iterator(i, x, ns_coeffs_),
            x, unroll=unroll,
        )
    else:
        def _scan_body(carry, coeffs_step):
            i, x = carry
            return (i + 1, _ns_iterator(i, x, coeffs_step)), None
        (_, x), _ = jax.lax.scan(
            _scan_body, (jnp.asarray(0, jnp.int32), x), ns_coeffs_,
        )

    if transposed:
        x = x.T
    return x


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: jax.typing.ArrayLike = 5,
    preconditioning: Literal[
        "frobenius", "spectral", "aol", "schatten"
    ] = "frobenius",
    eps: jax.typing.ArrayLike = 1e-8,
    dimension_numbers: MuonDimensionNumbers | None = None,
) -> jax.Array:
    """Orthogonalize via Newton-Schulz iteration (per-leaf, non-bucketed).

    Drop-in replacement for optax's version; used by the
    ``muon_batched_ns=False`` fallback path.
    """
    if x.ndim != 2 and not isinstance(dimension_numbers, MuonDimensionNumbers):
        raise ValueError(
            f"Input must have shape (m, n) or weight dimension numbers must be "
            f"provided. Got shape={x.shape} and {dimension_numbers=}."
        )
    if x.ndim == 2:
        dimension_numbers = MuonDimensionNumbers(reduction_axis=0, output_axis=1)
    _validate_ns_coeffs(ns_coeffs)

    def _orthogonalize(x: jax.Array) -> jax.Array:
        return _orthogonalize_2d(x, ns_coeffs, ns_steps, preconditioning, eps)

    reshape_fn, inverse_fn = _compute_muon_reshape(x, dimension_numbers)
    return inverse_fn(jax.vmap(_orthogonalize)(reshape_fn(x)))


def _ns_batch_sharding(mesh) -> jax.sharding.NamedSharding | None:
    """Sharding that spreads a stacked NS bucket over its leading batch axis.

    Shards only over non-trivial mesh axes, excluding pure-replica axes
    ('data': grads/momentum are already replicated across it, so leaving the
    NS computation replicated there avoids cross-replica gathers) and 'stage'
    (pipeline stages own disjoint params).
    """
    if mesh is None:
        return None
    axes = tuple(
        name for name in mesh.axis_names
        if mesh.shape[name] > 1 and name not in ("data", "stage")
    )
    if not axes:
        return None
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(axes))


def _pin_tree_sharding(x_tree, anchor_tree):
    """Pin every array leaf of `x_tree` to the sharding of `anchor_tree`.

    Identity on values (shard_alike only adds sharding annotations). Used to
    anchor the nesterov momentum tensors feeding Newton-Schulz to the momentum
    state's fixed (params-like) sharding, so that whatever layout the NS
    bucketing downstream prefers cannot back-propagate through the elementwise
    momentum/nesterov ops into the gradient all-reduce's consumer chain.
    Returns the pinned (x_tree, anchor_tree) pair; use BOTH returned trees.
    """
    xs, tdef = jax.tree.flatten(x_tree)
    anchors = tdef.flatten_up_to(anchor_tree)
    out_x, out_a = [], []
    for x, a in zip(xs, anchors):
        if (
            hasattr(x, "shape") and hasattr(a, "shape")
            and getattr(x, "shape", None) == getattr(a, "shape", None)
        ):
            x, a = shard_alike(x, a)
        out_x.append(x)
        out_a.append(a)
    return jax.tree.unflatten(tdef, out_x), jax.tree.unflatten(tdef, out_a)


def _orthogonalize_tree_batched(
    mu_hat,
    dim_nums_tree,
    ns_coeffs: jax.Array,
    ns_steps: jax.typing.ArrayLike,
    preconditioning: str,
    eps: jax.typing.ArrayLike,
    mesh=None,
    ns_compute_dtype=None,
    batch_reshard: bool = False,
):
    """Shape-bucketed batched Newton-Schulz over a whole parameter tree.

    Equivalent to
    ``jax.tree.map(orthogonalize_via_newton_schulz, mu_hat, dim_nums_tree)``
    but runs one vmapped NS chain per distinct (rows, cols) matrix shape
    instead of one per leaf. Per-matrix math is identical: the leaf is
    reshaped with `_compute_muon_reshape` to (batch, rows, cols), transposed
    to rows <= cols (shape-static, so hoisting it out of the vmap is exact),
    and the same vmapped `_orthogonalize_2d` runs on the concatenated batch.
    """
    _validate_ns_coeffs(ns_coeffs)

    entries = []

    def _collect(x, dim_num):
        entries.append((x, dim_num))
        return len(entries) - 1

    index_tree = jax.tree.map(
        _collect, mu_hat, dim_nums_tree, is_leaf=_is_weight_dim_nums
    )

    sharding = _ns_batch_sharding(mesh) if batch_reshard else None

    # Bucket leaves by their post-transpose 2D matrix shape (and dtype).
    buckets: dict = {}  # key -> list of (B_i, rows, cols) arrays
    specs = []  # per entry: (key, offset, batch, transposed, inverse_fn)
    for x, dim_num in entries:
        if x.ndim != 2 and not isinstance(dim_num, MuonDimensionNumbers):
            raise ValueError(
                f"Input must have shape (m, n) or weight dimension numbers "
                f"must be provided. Got shape={x.shape} and {dim_num=}."
            )
        if x.ndim == 2:
            dim_num = MuonDimensionNumbers(reduction_axis=0, output_axis=1)
        reshape_fn, inverse_fn = _compute_muon_reshape(x, dim_num)
        x3 = reshape_fn(x)  # (batch, reduction, output)
        transposed = x3.shape[1] > x3.shape[2]
        if transposed:
            x3 = x3.swapaxes(1, 2)
        key = (x3.shape[1], x3.shape[2], x3.dtype)
        parts = buckets.setdefault(key, [])
        offset = sum(p.shape[0] for p in parts)
        parts.append(x3)
        specs.append((key, offset, x3.shape[0], transposed, inverse_fn))

    # One batched NS chain per bucket.
    ortho_buckets = {}
    for key, parts in buckets.items():
        stacked = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=0)
        out_dtype = stacked.dtype
        if ns_compute_dtype is not None:
            stacked = stacked.astype(ns_compute_dtype)
        if sharding is not None:
            # Reshard ONCE onto the batch axis: every device owns whole
            # matrices, so all NS-iteration matmuls are collective-free.
            stacked = jax.lax.with_sharding_constraint(stacked, sharding)
        # When batch-sharded, each device holds only ceil(B/devices) matrices,
        # so unrolling is HBM-cheap and lets XLA overlap across buckets.
        # Unsharded (e.g. single device), keep the memory-lean rolled loop.
        unroll = sharding is not None
        ortho = jax.vmap(
            lambda m: _orthogonalize_2d(
                m, ns_coeffs, ns_steps, preconditioning, eps, unroll=unroll)
        )(stacked)
        ortho_buckets[key] = ortho.astype(out_dtype)

    # Unstack and restore original layouts.
    outputs = []
    for key, offset, batch, transposed, inverse_fn in specs:
        sub = ortho_buckets[key][offset:offset + batch]
        if transposed:
            sub = sub.swapaxes(1, 2)
        outputs.append(inverse_fn(sub))
    return jax.tree.map(lambda i: outputs[i], index_tree)


def scale_by_muon_batched(
    ns_coeffs=_DEFAULT_NS_COEFFS,
    ns_steps: jax.typing.ArrayLike = 5,
    beta: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    mu_dtype=None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    preconditioning: Literal[
        "frobenius", "spectral", "aol", "schatten"
    ] = "frobenius",
    weight_dimension_numbers=None,
    mesh=None,
    ns_compute_dtype=None,
    batch_reshard: bool = False,
) -> base.GradientTransformation:
    """`optax.contrib._muon.scale_by_muon` with shape-bucketed batched NS.

    Identical to the optax implementation except that the per-leaf
    `jax.tree.map(orthogonalize_via_newton_schulz, ...)` is replaced by
    `_orthogonalize_tree_batched` (same math, fused execution).

    Extra args vs optax:
      mesh: optional `jax.sharding.Mesh`; if given, the nesterov momentum
        tensors feeding NS are pinned to the momentum state's sharding
        (identity on values) so NS layout choices cannot back-propagate into
        the gradient all-reduce's consumer chain.
      ns_compute_dtype: optional dtype (e.g. jnp.bfloat16) to run NS in.
        None keeps the update dtype (matches baseline numerics).
      batch_reshard: if True (and mesh given), reshard each stacked NS bucket
        onto its batch axis so NS matmuls are collective-free. Off by
        default; see module docstring (can defeat the multi-slice DCN
        gradient all-reduce/backward overlap).
    """
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = optax.tree.zeros_like(params, dtype=mu_dtype)  # First moment
        ns_coeffs_ = jnp.asarray(ns_coeffs)
        _validate_ns_coeffs(ns_coeffs_)
        if ns_coeffs_.ndim == 2:
            if not ns_coeffs_.shape[0] <= ns_steps:
                raise ValueError(f"Not enough coeffs to perform {ns_steps} steps")
            ns_coeffs_ = ns_coeffs_[-ns_steps:]
        return MuonState(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            ns_coeffs=ns_coeffs_,
        )

    def update_fn(updates, state, params=None):
        del params
        if callable(weight_dimension_numbers):
            resolved_weight_dim_nums = weight_dimension_numbers(updates)
        else:
            resolved_weight_dim_nums = weight_dimension_numbers

        mu = optax.tree.update_moment(updates, state.mu, beta, 1)
        count_inc = numerics.safe_increment(state.count)
        if nesterov:
            mu_hat = jax.tree.map(
                lambda m, g: beta * m + (1 - beta) * g,
                optax.tree.bias_correction(
                    mu, beta, numerics.safe_increment(count_inc)
                ),
                optax.tree.bias_correction(updates, beta, count_inc),
            )
        else:
            mu_hat = optax.tree.bias_correction(mu, beta, count_inc)

        if mesh is not None:
            # Anchor the NS inputs to the momentum state's (params-like)
            # sharding so the bucketed reshape/concat below cannot
            # back-propagate a different layout onto gradient-derived
            # tensors (which would break XLA's data-parallel all-reduce
            # overlap pattern on multi-slice runs). Value identity.
            mu_hat, mu = _pin_tree_sharding(mu_hat, mu)

        # Apply Newton-Schulz orthogonalization (bucketed + batched).
        new_updates = _orthogonalize_tree_batched(
            mu_hat,
            resolved_weight_dim_nums,
            state.ns_coeffs,
            ns_steps,
            preconditioning,
            eps,
            mesh=mesh,
            ns_compute_dtype=ns_compute_dtype,
            batch_reshard=batch_reshard,
        )
        if adaptive:
            # Scale the orthogonalized updates by the dual norm of the
            # original updates. See https://arxiv.org/abs/2409.20325.
            new_updates = jax.tree.map(
                lambda x, y: jnp.sum(x.conj() * y) * y, mu_hat, new_updates
            )

        mu = optax.tree.cast(mu, mu_dtype)
        return new_updates, MuonState(
            count=count_inc,
            mu=mu,
            ns_coeffs=state.ns_coeffs,
        )

    return base.GradientTransformation(init_fn, update_fn)
