"""Generate static figures for the jax_pchip documentation."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import PchipInterpolator as ScipyPchip

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax_pchip import PchipInterpolator

STATIC = "_static"

# ── colour palette ────────────────────────────────────────────────────────────
C0 = "#2B7BB9"  # blue  – jax_pchip curve
C1 = "#E05C2A"  # orange – scipy curve / knot markers
C2 = "#444444"  # dark grey – residual line


# ── Figure 1: interpolation demo ─────────────────────────────────────────────
# Use knots that highlight PCHIP's monotone-preserving behaviour:
# a steep rise, a flat plateau, another steep rise.  A plain cubic spline
# would overshoot the flat region; PCHIP does not.
x_knots = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0])
y_knots = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

interp = PchipInterpolator(x_knots)
x_dense = np.linspace(x_knots[0], x_knots[-1], 500)
y_dense = np.asarray(interp(jnp.array(y_knots), jnp.array(x_dense)))

fig, ax = plt.subplots(figsize=(6, 3.2), layout="constrained")
ax.plot(x_dense, y_dense, color=C0, lw=2, label="PCHIP curve")
ax.scatter(x_knots, y_knots, color=C1, zorder=5, s=50, label="knots $(x_i, y_i)$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("PCHIP interpolation: monotone preservation", pad=8)
ax.legend()
ax.set_xlim(x_knots[0] - 0.2, x_knots[-1] + 0.2)
ax.set_ylim(-0.3, 2.5)
fig.savefig(f"{STATIC}/interpolation.png", dpi=200)
plt.close(fig)
print("wrote interpolation.png")


# ── Figure 2: match with scipy ────────────────────────────────────────────────
x_k = np.array([0.0, 1.0, 3.0, 4.5, 6.0, 8.0, 10.0])
y_k = np.array([0.3, 1.4, 0.6, 1.8, 1.1, 0.9, 1.6])

x_eval = np.linspace(x_k[0], x_k[-1], 600)
scipy_y = ScipyPchip(x_k, y_k)(x_eval)
jax_y = np.asarray(PchipInterpolator(x_k)(jnp.array(y_k), jnp.array(x_eval)))
residual = jax_y - scipy_y

fig = plt.figure(figsize=(6, 5), layout="constrained")
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], figure=fig)

ax0 = fig.add_subplot(gs[0])
ax0.plot(x_eval, scipy_y, color=C1, lw=2.5, label="scipy")
ax0.plot(x_eval, jax_y, color=C0, lw=1.5, ls="--", label="jax_pchip")
ax0.scatter(x_k, y_k, color=C2, zorder=5, s=40, label="knots")
ax0.set_ylabel("$y$")
ax0.set_title("jax_pchip vs scipy: numerical agreement", pad=8)
ax0.legend()
ax0.set_xticklabels([])
ax0.set_xlim(x_k[0], x_k[-1])

ax1 = fig.add_subplot(gs[1])
ax1.plot(x_eval, residual, color=C2, lw=1)
ax1.axhline(0, color=C2, lw=0.5, ls=":")
ax1.set_ylabel("residual")
ax1.set_xlabel("$x$")
ax1.set_xlim(x_k[0], x_k[-1])
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

fig.savefig(f"{STATIC}/scipy_comparison.png", dpi=200)
plt.close(fig)
print("wrote scipy_comparison.png")
