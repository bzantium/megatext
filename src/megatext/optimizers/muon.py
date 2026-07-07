"""Newton-Schulz orthogonalization with unroll=False for reduced compile-time HBM.

optax's Muon uses unroll=True in fori_loop, forcing XLA to keep all NS iteration
intermediates alive simultaneously (~5.8GB for 1.7B model). Using unroll=False
lets XLA reuse buffers across iterations (-> 3.3GB, matching AdamW).
"""

from __future__ import annotations

from math import inf, sqrt
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from optax.contrib._muon import (
    MuonDimensionNumbers,
    _compute_muon_reshape,
    _base_ns_iterator,
    _aol_ns_iterator,
    _schatten_ns_iterator,
)


# ── Polar Express coefficient schedule (Amsel et al. 2025, arXiv:2505.16932) ──
# Per-iteration minimax-optimal odd-quintic coefficients for approximating the
# constant 1 on [l, u] (== approximating sign(x)); the composition converges to
# the polar factor faster than the fixed Newton-Schulz coeffs. Computed offline
# (once, at optimizer build) via the paper's closed-form Remez for degree 5,
# then fed to the same NS iteration as a (T, 3) coefficient array.


def _pe_optimal_quintic(l: float, u: float) -> tuple[float, float, float]:
    """Minimax odd quintic p(x)=a x + b x^3 + c x^5 approximating 1 on [l, u]."""
    assert 0 <= l <= u
    if 1 - 5e-6 <= l / u:
        return (15 / 8) / u, (-10 / 8) / (u**3), (3 / 8) / (u**5)
    q = (3 * l + u) / 4
    r = (l + 3 * u) / 4
    E, old_E = inf, None
    while old_E is None or abs(old_E - E) > 1e-15:
        old_E = E
        lhs = np.array([
            [l, l**3, l**5, 1],
            [q, q**3, q**5, -1],
            [r, r**3, r**5, 1],
            [u, u**3, u**5, -1],
        ])
        a, b, c, E = np.linalg.solve(lhs, np.ones(4))
        q, r = np.sqrt((-3 * b + np.array([-1, 1]) * sqrt(9 * b**2 - 20 * a * c)) / (10 * c))
    return float(a), float(b), float(c)


def polar_express_coeffs(
    num_iters: int, lower_bound: float = 1e-3, safety_eps: float = 1e-2, cushion: float = 0.02
) -> np.ndarray:
    """(num_iters, 3) minimax-optimal quintic coeffs, greedily composed per Thm 3.1."""
    l, u = float(lower_bound), 1.0
    coeffs = []
    for it in range(num_iters):
        a, b, c = _pe_optimal_quintic(max(l, cushion * u), u)
        if cushion * u > l:
            pl = a * l + b * l**3 + c * l**5
            pu = a * u + b * u**3 + c * u**5
            rescale = 2 / (pl + pu)
            a, b, c = a * rescale, b * rescale, c * rescale
        if it < num_iters - 1:  # safety factor keeps intermediates < basin edge
            sf = 1 + safety_eps
            a, b, c = a / sf, b / sf**3, c / sf**5
        coeffs.append((a, b, c))
        l = a * l + b * l**3 + c * l**5
        u = 2 - l
    return np.asarray(coeffs, dtype=np.float32)


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
