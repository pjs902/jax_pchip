"""JAX-compatible PCHIP monotone cubic interpolation.

Implements the Fritsch-Carlson (1980) algorithm for piecewise cubic Hermite
interpolation.  Knot *positions* are treated as static (fixed at construction
time); knot *heights* are dynamic JAX arrays, fully compatible with jit, grad,
vmap, and scan.

This is analogous to what jax_cosmo provides for B-splines, but for PCHIP.

Reference
---------
Fritsch, F. N. & Carlson, R. E. (1980). Monotone piecewise cubic interpolation.
SIAM Journal on Numerical Analysis, 17(2), 238–246.
"""

import numpy as np
import jax.numpy as jnp

__all__ = ["build_interpolator", "PchipInterpolator"]


def _endpoint_slope(h0: float, h1: float, m0: jnp.ndarray, m1: jnp.ndarray) -> jnp.ndarray:
    """Scipy-compatible one-sided 3-point endpoint slope estimate.

    Fits a quadratic through the first (or last) three knots and evaluates its
    derivative at the endpoint, then clamps to maintain monotonicity:
    - If the quadratic slope disagrees in sign with the adjacent divided
      difference, set it to zero.
    - If adjacent divided differences have opposite sign and the quadratic
      slope exceeds 3× the adjacent difference, clamp to 3× that difference.

    This matches ``scipy.interpolate.PchipInterpolator`` exactly.
    """
    d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    d = jnp.where(jnp.sign(d) != jnp.sign(m0), 0.0, d)
    d = jnp.where(
        (jnp.sign(m0) != jnp.sign(m1)) & (jnp.abs(d) > 3.0 * jnp.abs(m0)),
        3.0 * m0,
        d,
    )
    return d


def _slopes(h: np.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    """Fritsch-Carlson monotone slopes at each knot.

    Parameters
    ----------
    h :
        Interval widths (n-1,).  Concrete numpy — never JAX-traced.
    delta :
        Divided differences dy/h, shape (n-1,).  JAX array — fully traced.

    Returns
    -------
    d :
        Derivative estimates at each of the n knots, shape (n,).
    """
    h_km1 = h[:-1]
    h_k   = h[1:]
    d_km1 = delta[:-1]
    d_k   = delta[1:]
    w1    = 2.0 * h_k   + h_km1
    w2    = h_k + 2.0 * h_km1

    # NaN-safe weighted harmonic mean.
    # JAX evaluates both branches of jnp.where during gradient tracing, so
    # substituting 1.0 in the zero-denominator branch prevents NaN gradients
    # even though those values are ultimately discarded.
    safe_d_km1 = jnp.where(d_km1 == 0.0, 1.0, d_km1)
    safe_d_k   = jnp.where(d_k   == 0.0, 1.0, d_k)
    raw_denom  = w1 / safe_d_km1 + w2 / safe_d_k
    safe_denom = jnp.where(raw_denom == 0.0, 1.0, raw_denom)
    whmean     = (w1 + w2) / safe_denom

    same_sign = jnp.sign(d_km1) * jnp.sign(d_k)
    d_int     = jnp.where(same_sign > 0, whmean, 0.0)

    # Endpoint slopes: scipy-compatible 3-point quadratic formula.
    # For n=2 (single interval) there is no second interval to fit a quadratic,
    # so both endpoint slopes equal the single divided difference.
    if len(h) == 1:
        return jnp.concatenate([delta, delta])

    d_left  = _endpoint_slope(h[0],  h[1],  delta[0],  delta[1])
    d_right = _endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])

    return jnp.concatenate([d_left[jnp.newaxis], d_int, d_right[jnp.newaxis]])


def build_interpolator(x, *, extrapolate: bool = False):
    """Return a JAX-jittable PCHIP interpolation function for fixed knot positions.

    Knot positions ``x`` are baked in as static constants; only the knot
    heights ``y`` and query points ``x_eval`` are JAX-traced.

    Parameters
    ----------
    x :
        Strictly increasing knot positions, shape (n,).
    extrapolate :
        If ``False`` (default), queries outside ``[x[0], x[-1]]`` return the
        nearest endpoint value (flat extrapolation).  If ``True``, the boundary
        cubic is extended beyond the knot range.

    Returns
    -------
    interpolate : callable
        Signature ``(y, x_eval) -> values``.
        ``y`` shape (n,), ``x_eval`` shape (...). Returns same shape as
        ``x_eval``.  Differentiable w.r.t. ``y`` via ``jax.grad``; compatible
        with ``jax.jit`` and ``jax.vmap``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("x must be a 1-D array with at least 2 elements")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing")

    # h is a concrete numpy array — its values become compile-time constants
    # inside the JIT-traced closure, so re-tracing is not triggered when only y
    # or x_eval change.
    h_np   = np.diff(x)
    x_jnp  = jnp.array(x)
    h_jnp  = jnp.array(h_np)
    n      = len(x)

    def interpolate(y, x_eval):
        """Evaluate PCHIP at ``x_eval`` given knot heights ``y``.

        Parameters
        ----------
        y :
            Knot heights, shape (n,).  Must be a JAX-compatible array.
        x_eval :
            Query points, shape (...).

        Returns
        -------
        values :
            Interpolated values, same shape as ``x_eval``.
        """
        y      = jnp.asarray(y, float)
        x_eval = jnp.asarray(x_eval, float)
        shape  = x_eval.shape
        x_flat = x_eval.ravel()

        delta = jnp.diff(y) / h_jnp
        d     = _slopes(h_np, delta)

        # Interval index clipped to [0, n-2] so we never index out of bounds
        idx = jnp.clip(
            jnp.searchsorted(x_jnp, x_flat, side="right") - 1,
            0, n - 2,
        )

        xk0 = jnp.take(x_jnp, idx)
        hk  = jnp.take(h_jnp, idx)
        y0  = jnp.take(y, idx)
        y1  = jnp.take(y, idx + 1)
        d0  = jnp.take(d, idx)
        d1  = jnp.take(d, idx + 1)

        # Local coordinate t ∈ [0, 1] along each interval
        t = jnp.clip((x_flat - xk0) / hk, 0.0, 1.0)

        # Cubic Hermite basis polynomials
        h00 = (1.0 + 2.0 * t) * (1.0 - t) ** 2
        h10 = t * (1.0 - t) ** 2
        h01 = t ** 2 * (3.0 - 2.0 * t)
        h11 = t ** 2 * (t - 1.0)

        vals = h00 * y0 + h10 * hk * d0 + h01 * y1 + h11 * hk * d1

        if not extrapolate:
            vals = jnp.where(x_flat < x_jnp[0],  y[0],    vals)
            vals = jnp.where(x_flat > x_jnp[-1], y[-1],   vals)

        return vals.reshape(shape)

    return interpolate


class PchipInterpolator:
    """JAX-compatible PCHIP monotone cubic interpolator.

    A drop-in replacement for ``scipy.interpolate.PchipInterpolator`` that
    supports JAX ``jit``, ``grad``, and ``vmap`` with respect to the knot
    heights ``y``.

    The key distinction from scipy's version is the separation between
    *construction* (where knot positions ``x`` are fixed) and *evaluation*
    (where knot heights ``y`` and query points ``x_eval`` are JAX-traced):

    .. code-block:: python

        interp = PchipInterpolator(x_knots)          # x is static
        values = interp(y_knots, x_query)            # y is dynamic

        # jit over y:
        jit_interp = jax.jit(interp)

        # grad w.r.t. y:
        grad_fn = jax.grad(lambda y: interp(y, x_query).sum())

        # vmap over a batch of y heights (e.g. posterior samples):
        batch_fn = jax.vmap(lambda y: interp(y, x_query))

    Parameters
    ----------
    x :
        Strictly increasing knot positions, shape (n,).
    extrapolate :
        If ``False`` (default), flat-extrapolate outside ``[x[0], x[-1]]``.
    """

    def __init__(self, x, *, extrapolate: bool = False):
        self.x          = np.asarray(x, float)
        self._fn        = build_interpolator(x, extrapolate=extrapolate)
        self.extrapolate = extrapolate

    def __call__(self, y, x_eval):
        return self._fn(y, x_eval)

    def __repr__(self):
        return (f"PchipInterpolator(n_knots={len(self.x)}, "
                f"x=[{self.x[0]:.3g}, …, {self.x[-1]:.3g}], "
                f"extrapolate={self.extrapolate})")
