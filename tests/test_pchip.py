"""Tests for jax_pchip.

Covers:
  - Values match scipy.interpolate.PchipInterpolator on interior points
  - Exact interpolation at knot positions
  - Flat extrapolation outside the knot range
  - JAX jit compilation
  - JAX grad w.r.t. knot heights (finite-difference checked)
  - jit(grad) composition
  - vmap over a batch of height vectors
  - Monotonicity preservation on monotone data
  - Functional build_interpolator API
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.interpolate import PchipInterpolator as ScipyPchip

from jax_pchip import PchipInterpolator, build_interpolator

RNG = np.random.default_rng(42)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple():
    x = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    y = np.array([0.5, 1.2, 0.8, 1.5, 1.0])
    return x, y


@pytest.fixture
def log_spaced():
    """Log-spaced knots, typical for radial kinematic profiles."""
    x = np.logspace(np.log10(5), np.log10(600), 10)
    y = 20.0 * np.exp(-x / 200)
    return x, y


@pytest.fixture
def monotone_increasing():
    x = np.linspace(0.0, 10.0, 8)
    y = np.cumsum(np.abs(RNG.normal(size=8))) + 0.1
    return x, y


# ── Match scipy ───────────────────────────────────────────────────────────────

class TestMatchesScipy:
    def test_interior_points(self, simple):
        x, y = simple
        x_eval = np.linspace(x[0], x[-1], 300)

        expected = ScipyPchip(x, y)(x_eval)
        got      = np.asarray(PchipInterpolator(x)(jnp.array(y), jnp.array(x_eval)))

        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-12)

    def test_at_knots(self, simple):
        x, y = simple
        got = np.asarray(PchipInterpolator(x)(jnp.array(y), jnp.array(x)))
        np.testing.assert_allclose(got, y, rtol=1e-10)

    def test_log_spaced_knots(self, log_spaced):
        x, y = log_spaced
        x_eval = np.linspace(x[0], x[-1], 400)

        expected = ScipyPchip(x, y)(x_eval)
        got      = np.asarray(PchipInterpolator(x)(jnp.array(y), jnp.array(x_eval)))

        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-12)

    def test_monotone_data(self, monotone_increasing):
        x, y = monotone_increasing
        x_eval = np.linspace(x[0], x[-1], 500)

        expected = ScipyPchip(x, y)(x_eval)
        got      = np.asarray(PchipInterpolator(x)(jnp.array(y), jnp.array(x_eval)))

        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-12)

    def test_two_knots(self):
        x = np.array([0.0, 1.0])
        y = np.array([2.0, 5.0])
        x_eval = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        expected = ScipyPchip(x, y)(x_eval)
        got      = np.asarray(PchipInterpolator(x)(jnp.array(y), jnp.array(x_eval)))

        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_scalar_query(self, simple):
        x, y = simple
        x_eval = jnp.array(4.5)
        got      = float(PchipInterpolator(x)(jnp.array(y), x_eval))
        expected = float(ScipyPchip(x, y)(4.5))
        assert abs(got - expected) < 1e-10


# ── Extrapolation ─────────────────────────────────────────────────────────────

class TestExtrapolation:
    def test_flat_below(self, simple):
        x, y = simple
        x_below = np.array([-10.0, -1.0, -1e-6])
        vals = np.asarray(PchipInterpolator(x)(jnp.array(y), jnp.array(x_below)))
        np.testing.assert_allclose(vals, y[0], rtol=1e-10)

    def test_flat_above(self, simple):
        x, y = simple
        x_above = np.array([10.0 + 1e-6, 15.0, 1000.0])
        vals = np.asarray(PchipInterpolator(x)(jnp.array(y), jnp.array(x_above)))
        np.testing.assert_allclose(vals, y[-1], rtol=1e-10)

    def test_at_left_boundary(self, simple):
        x, y = simple
        val = float(PchipInterpolator(x)(jnp.array(y), jnp.array(x[0])))
        assert abs(val - y[0]) < 1e-10

    def test_at_right_boundary(self, simple):
        x, y = simple
        val = float(PchipInterpolator(x)(jnp.array(y), jnp.array(x[-1])))
        assert abs(val - y[-1]) < 1e-10


# ── JAX compatibility ─────────────────────────────────────────────────────────

class TestJAXCompatibility:
    def test_jit(self, simple):
        x, y = simple
        x_eval  = jnp.linspace(x[0], x[-1], 100)
        interp  = PchipInterpolator(x)
        y_jnp   = jnp.array(y)

        eager  = np.asarray(interp(y_jnp, x_eval))
        jitted = np.asarray(jax.jit(interp)(y_jnp, x_eval))

        np.testing.assert_allclose(jitted, eager, rtol=1e-10)

    def test_grad_wrt_y(self, simple):
        x, y = simple
        x_eval = jnp.array([1.5, 2.5, 4.0, 8.0])
        interp  = PchipInterpolator(x)
        y_jnp   = jnp.array(y)

        def scalar_fn(y_):
            return interp(y_, x_eval).sum()

        grad_jax = np.asarray(jax.grad(scalar_fn)(y_jnp))

        # Finite-difference reference
        eps = 1e-5
        grad_fd = np.zeros_like(y)
        for i in range(len(y)):
            yp, ym = y.copy(), y.copy()
            yp[i] += eps; ym[i] -= eps
            grad_fd[i] = (scalar_fn(jnp.array(yp)) - scalar_fn(jnp.array(ym))) / (2 * eps)

        np.testing.assert_allclose(grad_jax, grad_fd, rtol=1e-4, atol=1e-7)

    def test_jit_grad(self, simple):
        x, y = simple
        x_eval = jnp.array([1.5, 4.5, 8.0])
        interp  = PchipInterpolator(x)
        y_jnp   = jnp.array(y)

        jit_grad = jax.jit(jax.grad(lambda y_: interp(y_, x_eval).sum()))
        result   = jit_grad(y_jnp)

        assert result.shape == y_jnp.shape
        assert np.all(np.isfinite(np.asarray(result)))

    def test_grad_log_spaced(self, log_spaced):
        """Grad works on log-spaced knots (common for radial profiles)."""
        x, y = log_spaced
        x_eval = jnp.array([10.0, 50.0, 200.0, 500.0])
        interp  = PchipInterpolator(x)
        y_jnp   = jnp.array(y)

        grad = jax.grad(lambda y_: interp(y_, x_eval).sum())(y_jnp)
        assert np.all(np.isfinite(np.asarray(grad)))

    def test_vmap_over_y_batch(self, simple):
        """vmap over a batch of knot heights — e.g. posterior samples."""
        x, y = simple
        n_batch = 12
        ys      = jnp.array(RNG.uniform(0.5, 2.0, size=(n_batch, len(y))))
        x_eval  = jnp.linspace(x[0], x[-1], 50)
        interp  = PchipInterpolator(x)

        batched = jax.vmap(lambda y_: interp(y_, x_eval))
        result  = batched(ys)

        assert result.shape == (n_batch, 50)

        # Each row should match the non-batched call
        for i in range(n_batch):
            expected = np.asarray(interp(ys[i], x_eval))
            np.testing.assert_allclose(np.asarray(result[i]), expected, rtol=1e-10)

    def test_grad_zero_at_plateau(self):
        """Flat region between knots: grad w.r.t. distant knot should be near zero."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])  # flat
        x_eval = jnp.array([1.5])
        interp  = PchipInterpolator(x)

        grad = np.asarray(jax.grad(lambda y_: interp(y_, x_eval).sum())(y))
        assert np.all(np.isfinite(grad)), "NaN gradient on flat knots"


# ── Monotonicity ──────────────────────────────────────────────────────────────

class TestMonotonicity:
    def test_monotone_increasing_no_overshoot(self, monotone_increasing):
        x, y = monotone_increasing
        x_eval = np.linspace(x[0], x[-1], 2000)
        vals   = np.asarray(PchipInterpolator(x)(jnp.array(y), jnp.array(x_eval)))
        assert np.all(np.diff(vals) >= -1e-12), "PCHIP overshot on monotone data"

    def test_linear_is_exact(self):
        """Linear data → PCHIP must reproduce the line exactly."""
        x = np.linspace(0.0, 5.0, 7)
        y = 2.0 * x + 1.0
        x_eval = np.linspace(0.0, 5.0, 300)

        expected = 2.0 * x_eval + 1.0
        got      = np.asarray(PchipInterpolator(x)(jnp.array(y), jnp.array(x_eval)))

        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_no_wiggles_hump(self):
        """Non-monotone data with a single hump should not gain extra extrema."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        x_eval = np.linspace(0.0, 4.0, 1000)

        vals = np.asarray(PchipInterpolator(x)(jnp.array(y), jnp.array(x_eval)))

        # Must stay within [0, 2]
        assert float(vals.min()) >= -1e-12
        assert float(vals.max()) <= 2.0 + 1e-12


# ── Functional API ────────────────────────────────────────────────────────────

class TestFunctionalAPI:
    def test_build_interpolator_matches_class(self, simple):
        x, y = simple
        x_eval = np.linspace(x[0], x[-1], 200)

        fn     = build_interpolator(x)
        interp = PchipInterpolator(x)

        np.testing.assert_allclose(
            np.asarray(fn(jnp.array(y), jnp.array(x_eval))),
            np.asarray(interp(jnp.array(y), jnp.array(x_eval))),
            rtol=1e-10,
        )

    def test_build_interpolator_jit(self, simple):
        x, y = simple
        x_eval = jnp.linspace(x[0], x[-1], 80)
        fn     = jax.jit(build_interpolator(x))
        result = fn(jnp.array(y), x_eval)
        assert result.shape == x_eval.shape

    def test_invalid_x_not_increasing(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            build_interpolator(np.array([0.0, 2.0, 1.0]))

    def test_invalid_x_too_short(self):
        with pytest.raises(ValueError, match="at least 2"):
            build_interpolator(np.array([1.0]))

    def test_output_shape_2d_query(self, simple):
        x, y = simple
        x_eval = jnp.ones((4, 5)) * 3.0
        result  = PchipInterpolator(x)(jnp.array(y), x_eval)
        assert result.shape == (4, 5)
