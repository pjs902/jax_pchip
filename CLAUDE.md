# CLAUDE.md

Minimal Python package: JAX-compatible PCHIP interpolation.

## Environment

Uses the `main` conda environment (same as the omegaCen project):

```bash
conda run -n main python -m pytest tests/
```

## Commands

```bash
# Run tests
conda run -n main python -m pytest tests/ -v

# Install editable (already done — do not re-run unless setting up fresh)
conda run -n main pip install -e ".[test]"
```

## Architecture

Single public module `jax_pchip/_core.py`.  Two public symbols:

- `PchipInterpolator(x)` — class, freezes knot positions at construction
- `build_interpolator(x)` — functional factory, returns `(y, x_eval) → values`

Key invariant: `x` (knot positions) is always a concrete numpy array captured
at construction time and never JAX-traced.  Only `y` (heights) and `x_eval`
are dynamic.  This is what makes `jit` and `grad` work without recompilation.

## Tests

`tests/test_pchip.py` — 24 tests across five classes:

- `TestMatchesScipy` — numerical agreement with scipy (rtol 1e-6)
- `TestExtrapolation` — flat-extrapolation outside knot range
- `TestJAXCompatibility` — jit, grad (FD-verified), jit+grad, vmap
- `TestMonotonicity` — no overshoot on monotone data, linear exactness
- `TestFunctionalAPI` — `build_interpolator`, input validation, shape handling

`tests/conftest.py` sets `jax_enable_x64 = True` globally.

## Relationship to omegaCen project

Originally developed inline in
`~/research/omegaCen/subpopulations/paper_plots/kinematic_explorer.py`
(functions `_pchip_slopes_jax` and `_pchip_eval_jax`).  Extracted here for
reuse.  The omegaCen project imports via `from jax_pchip import PchipInterpolator`.

## Known differences from the original inline implementation

The original used one-sided first divided differences for endpoint slopes
(`delta[:1]` and `delta[-1:]`).  This library uses scipy's 3-point quadratic
endpoint formula, which matches `scipy.interpolate.PchipInterpolator` exactly
and gives better accuracy near the knot boundaries.
