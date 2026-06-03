# jax_pchip

[![Tests](https://github.com/pjs902/jax_pchip/actions/workflows/tests.yml/badge.svg)](https://github.com/pjs902/jax_pchip/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/jax-pchip/badge/?version=latest)](https://jax-pchip.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**`scipy.interpolate.PchipInterpolator`, translated to JAX.**

This is just a translation of the scipy implementation to JAX in the same style as the [`jax_cosmo`](https://github.com/DifferentiableUniverseInitiative/jax_cosmo) implementation of B-Splines.  It is the same Fritsch-Carlson (1980) PCHIP used by scipy, re-expressed so that knot *heights* are JAX arrays that can be differentiated and JIT-compiled.

Results match `scipy.interpolate.PchipInterpolator` to floating-point precision (rtol ≈ 1e-6), including the Fritsch-Carlson interior slopes and the 3-point quadratic endpoint formula.

---

## Motivation

The standard scipy `PchipInterpolator` cannot be used inside a JAX-traced function because it relies on numpy control flow.  If you want to sample PCHIP knot heights with NUTS (or any other JAX-based gradient method), you need a version where the heights are dynamic JAX arrays.

The key concept is that knot *positions* `x` are usually fixed throughout a computation (e.g. a set of radial bins that never change), while knot *heights* `y` are the parameters being optimised or sampled.  By freezing `x` at construction time and tracing only `y`, the interpolator compiles once and then runs at XLA speed for any `y`.

```
         static (compile-time)       dynamic (JAX-traced)
         ─────────────────────       ────────────────────
scipy:   x, y both passed at call   —
jax_pchip: x frozen at construction  y, x_eval passed at call
```

---

## Installation

```bash
pip install -e .           # editable, from repo root
pip install ".[test]"      # include pytest + scipy for tests
```

Requires JAX ≥ 0.4 and numpy.

---

## Quick start

```python
import jax
import jax.numpy as jnp
from jax_pchip import PchipInterpolator

jax.config.update("jax_enable_x64", True)   # recommended for numerical accuracy

x_knots = jnp.array([0.0, 1.0, 3.0, 6.0, 10.0])   # fixed knot positions
interp  = PchipInterpolator(x_knots)                 # bakes x in at construction

y = jnp.array([0.5, 1.2, 0.8, 1.5, 1.0])            # knot heights — dynamic
x_query = jnp.linspace(0, 10, 200)

# Plain call
values = interp(y, x_query)

# JIT-compiled — x is a compile-time constant, y and x_query are traced
values = jax.jit(interp)(y, x_query)

# Gradient w.r.t. knot heights
grad_fn = jax.grad(lambda y: interp(y, x_query).sum())
g = grad_fn(y)

# Batch over many height vectors (e.g. posterior samples)
y_batch = jnp.ones((100, 5)) * y
batch_fn = jax.vmap(lambda y: interp(y, x_query))
values_batch = batch_fn(y_batch)   # shape (100, 200)
```

---

## API

### `PchipInterpolator(x, *, extrapolate=False)`

Class-based interface.  Construct once with fixed knot positions, then call with dynamic heights.

```python
interp = PchipInterpolator(x)
values = interp(y, x_eval)
```

**Parameters**

| name | type | description |
|---|---|---|
| `x` | array-like (n,) | Strictly increasing knot positions. Fixed at construction. |
| `extrapolate` | bool | If `False` (default), queries outside `[x[0], x[-1]]` return the nearest endpoint value. |

**Call signature:** `interp(y, x_eval) → array`

| name | type | description |
|---|---|---|
| `y` | JAX array (n,) | Knot heights. Fully JAX-traced. |
| `x_eval` | JAX array (...) | Query points. Any shape; output matches. |

---

### `build_interpolator(x, *, extrapolate=False)`

Functional factory.  Returns a `(y, x_eval) → values` callable with `x` captured in a closure.

```python
fn = build_interpolator(x)
values = jax.jit(fn)(y, x_eval)
```

Equivalent to `PchipInterpolator(x).__call__` but useful when you want a plain function rather than a class instance (e.g. for `functools.partial`).

---

## Algorithm

Implements Fritsch & Carlson (1980) with scipy-compatible endpoint slopes.

**Interior slopes** — weighted harmonic mean of adjacent divided differences:

```
d_i = (w1 + w2) / (w1/Δ_{i-1} + w2/Δ_i)
```

where `w1 = 2h_i + h_{i-1}`, `w2 = h_i + 2h_{i-1}`.  Set to zero where adjacent divided differences have opposite sign (local extremum).

**Endpoint slopes** — one-sided 3-point quadratic estimate:

```
d_0 = ((2h_0 + h_1)·Δ_0 - h_0·Δ_1) / (h_0 + h_1)
```

Clamped: zeroed if it disagrees in sign with Δ₀; clamped to 3Δ₀ if Δ₀ and Δ₁ have opposite signs and the quadratic slope would overshoot.

**Evaluation** — cubic Hermite basis on each interval:

```
p(t) = h₀₀(t)·y_k + h₁₀(t)·hₖ·dₖ + h₀₁(t)·y_{k+1} + h₁₁(t)·hₖ·d_{k+1}
```

where `t = (x - xₖ)/hₖ ∈ [0, 1]`.

**NaN-safe gradients** — `jnp.where` evaluates both branches during gradient tracing.  Denominators that could be zero in the discarded branch are replaced with `1.0` before the division, preventing NaN gradients in zero-slope regions.

---

## Comparison with scipy

```python
from scipy.interpolate import PchipInterpolator as ScipyPchip
import numpy as np

expected = ScipyPchip(x, y)(x_eval)
got      = np.asarray(PchipInterpolator(x)(jnp.array(y), jnp.array(x_eval)))

np.testing.assert_allclose(got, expected, rtol=1e-6)   # passes
```

The only difference from scipy is the static/dynamic split: scipy takes `(x, y)` at construction and `x_eval` at call time; `jax_pchip` takes `x` at construction and `(y, x_eval)` at call time.

---

## Reference

Fritsch, F. N. & Carlson, R. E. (1980). Monotone piecewise cubic interpolation. *SIAM Journal on Numerical Analysis*, 17(2), 238–246.
