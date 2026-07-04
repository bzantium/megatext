"""Newton-Schulz orthogonalization tuned for TPU/FSDP, math-identical to optax.

Two deviations from optax's ``orthogonalize_via_newton_schulz``, both pure
execution changes (same iterates, same coefficients, same dtypes):

1. ``fori_loop(unroll=False)``: optax unrolls, forcing XLA to keep every NS
   iteration's intermediates alive at once (~5.8GB for a 1.7B model);
   unroll=False lets XLA reuse buffers across iterations (-> 3.3GB, ~AdamW).

2. Gather-once (Moonlight-style, https://arxiv.org/abs/2502.16982). The Muon
   momentum inherits the parameter sharding ``embed -> fsdp``, which for these
   weights is exactly the NS *reduction* axis. NS converges to the polar factor
   ``(XX^T)^{-1/2} X``, whose every output row depends on every input row, so
   the sharded reduction axis MUST be mixed by a collective -- it cannot be
   removed algebraically. Left alone, optax pays that collective on the two
   tail matmuls (``A@A`` and ``B@X``) EVERY iteration (~2 all-gathers x 5 steps
   per leaf). Instead we all-gather the reshaped matrix ONCE (via a replicated
   ``with_sharding_constraint``) before the 5-step loop, run all matmuls on the
   now-local matrix, and let the consumer scatter the result back -- one
   gather + one scatter per leaf instead of ten. When no mesh is given (single
   device / CPU tests) the constraint is skipped and this is a no-op.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

from optax.contrib._muon import (
    MuonDimensionNumbers,
    _compute_muon_reshape,
    _base_ns_iterator,
    _aol_ns_iterator,
    _schatten_ns_iterator,
)


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: jax.typing.ArrayLike = 5,
    preconditioning: Literal[
        "frobenius", "spectral", "aol", "schatten"
    ] = "frobenius",
    eps: jax.typing.ArrayLike = 1e-8,
    dimension_numbers: MuonDimensionNumbers | None = None,
    mesh: jax.sharding.Mesh | None = None,
) -> jax.Array:
    """Orthogonalize via Newton-Schulz iteration (unroll=False, gather-once).

    Drop-in replacement for optax's version. Differences are execution-only
    (same math): ``fori_loop(unroll=False)`` and, when ``mesh`` is provided, a
    single all-gather of the NS input's sharded reduction axis before the loop
    so the iterations run collective-free (see module docstring).
    """
    if x.ndim != 2 and not isinstance(dimension_numbers, MuonDimensionNumbers):
        raise ValueError(
            f"Input must have shape (m, n) or weight dimension numbers must be "
            f"provided. Got shape={x.shape} and {dimension_numbers=}."
        )
    if x.ndim == 2:
        dimension_numbers = MuonDimensionNumbers(reduction_axis=0, output_axis=1)
    if ns_coeffs.ndim > 2 or ns_coeffs.shape[-1] != 3:
        raise ValueError(
            f"Newton-Schulz coefficients must have shape (3,) or (n, 3), "
            f"got {ns_coeffs.shape}"
        )

    ns_iterators = {
        "frobenius": _base_ns_iterator,
        "spectral": _base_ns_iterator,
        "aol": _aol_ns_iterator,
        "schatten": _schatten_ns_iterator,
    }

    def _orthogonalize(x: jax.Array) -> jax.Array:
        transposed = False
        if x.shape[0] > x.shape[1]:
            x = x.T
            transposed = True

        _ns_iterator = ns_iterators[preconditioning]

        if preconditioning == "frobenius":
            x /= jnp.linalg.norm(x, ord="fro") + eps
        elif preconditioning == "spectral":
            x /= jnp.linalg.norm(x, ord=2) + eps

        ns_coeffs_ = ns_coeffs.astype(x.dtype)

        if ns_coeffs_.ndim == 1:
            x = jax.lax.fori_loop(
                0, ns_steps,
                lambda i, x: _ns_iterator(i, x, ns_coeffs_),
                x, unroll=False,
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

    reshape_fn, inverse_fn = _compute_muon_reshape(x, dimension_numbers)
    x3 = reshape_fn(x)  # (batch, reduction, output)

    if mesh is not None:
        # Gather-once: replicate the matrix (reduction, output) axes so every NS
        # matmul runs locally. Keep the leading batch/leaf axis free so XLA is
        # free to keep the per-leaf parallelism it already has. This forces a
        # single all-gather here; the consumer of `inverse_fn(...)` (the
        # elementwise param update, params-sharded) scatters the result back.
        replicated = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(None, None, None)
        )
        x3 = jax.lax.with_sharding_constraint(x3, replicated)

    return inverse_fn(jax.vmap(_orthogonalize)(x3))
