"""Newton-Schulz orthogonalization with unroll=False for reduced compile-time HBM.

optax's Muon uses unroll=True in fori_loop, forcing XLA to keep all NS iteration
intermediates alive simultaneously (~5.8GB for 1.7B model). Using unroll=False
lets XLA reuse buffers across iterations (-> 3.3GB, matching AdamW).
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
) -> jax.Array:
    """Orthogonalize via Newton-Schulz iteration (unroll=False).

    Drop-in replacement for optax's version. Only difference:
    fori_loop(unroll=False) instead of unroll=True.
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
    return inverse_fn(jax.vmap(_orthogonalize)(reshape_fn(x)))
