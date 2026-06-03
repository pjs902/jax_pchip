jax_pchip
=========

**scipy's** ``PchipInterpolator`` **, translated to JAX.**

This package is not a new algorithm — it is the same Fritsch-Carlson (1980) PCHIP
used by ``scipy.interpolate.PchipInterpolator``, re-expressed so that knot *heights*
``y`` are JAX arrays that can be differentiated and JIT-compiled.
This is the same approach that `jax_cosmo <https://github.com/DifferentiableUniverseInitiative/jax_cosmo>`_
takes for B-splines.

The key design: knot *positions* ``x`` are fixed at construction time (static);
knot *heights* ``y`` are JAX-traced dynamic arrays.  This lets the interpolator
compile once and run at XLA speed for any ``y``.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from jax_pchip import PchipInterpolator

   jax.config.update("jax_enable_x64", True)

   x_knots = jnp.array([0.0, 1.0, 3.0, 6.0, 10.0])
   interp  = PchipInterpolator(x_knots)

   y = jnp.array([0.5, 1.2, 0.8, 1.5, 1.0])
   x_query = jnp.linspace(0, 10, 200)

   values = jax.jit(interp)(y, x_query)           # JIT-compiled
   g      = jax.grad(lambda y: interp(y, x_query).sum())(y)  # gradient w.r.t. y

Installation
------------

.. code-block:: bash

   pip install jax-pchip

or from source::

   git clone https://github.com/pjs902/jax_pchip
   pip install -e ".[test]"

.. toctree::
   :maxdepth: 1

   api
