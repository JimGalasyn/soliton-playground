"""The clock gate 1 counts in must be measured, not assumed.

gpe_lab.characteristic_period is tau = L / c, so the whole gate rests on c being
what the preset claims. The module asserts c = 1 in its docstring; this measures
it by launching a small density pulse and tracking the front, so a change to the
interaction term or the stepper that shifted the sound speed would fail here
rather than silently rescale every lifetime in the bestiary.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jax_solitons.grid import BoxGrid
from soliton_playground.gpe_lab import (C_SOUND, characteristic_period, evolve)


def test_sound_speed_matches_the_preset_convention():
    """A small-amplitude density pulse travels at the Bogoliubov speed. Tolerance
    is 12%: the front position is tracked on a dx = 0.5 lattice, so the speed
    inherits a few percent of quantization error over the fit window."""
    N, L = 128, 64.0
    grid = BoxGrid(N=N, L=L, dtype=jnp.float64)
    ax = np.asarray(grid.axis())
    z = ax[None, None, :]
    psi = jnp.asarray(
        np.sqrt(1.0 + 0.01 * np.exp(-(z / 2.0) ** 2)) * np.ones((N, N, N)),
        dtype=jnp.complex128)

    pos = []

    def obs(t, p):
        d = np.asarray(jnp.abs(p[N // 2, N // 2, :]) ** 2) - 1.0
        m = ax > 0
        pos.append((t, float(ax[m][d[m].argmax()])))
        return dict(E=0.0)

    evolve(grid, psi, T=16.0, dt=0.01, sample_dt=2.0, observer=obs)
    t = np.array([p[0] for p in pos])
    x = np.array([p[1] for p in pos])
    win = (t >= 4.0) & (t <= 14.0)
    c = float(np.polyfit(t[win], x[win], 1)[0])
    assert abs(c - C_SOUND) / C_SOUND < 0.12, \
        f"measured sound speed {c:.4f} vs preset {C_SOUND}"


def test_characteristic_period_is_length_over_speed():
    assert characteristic_period(155.49, c=1.0) == 155.49
    assert characteristic_period(100.0, c=2.0) == 50.0


def test_trefoil_decays_inside_one_period():
    """The finding that makes gate 1 inapplicable as a threshold to this entrant,
    pinned so it cannot be quietly forgotten: the trefoil's curve is ~155 xi, so
    one traversal takes ~155 time units, while it unties by t ~ 40 and its entire
    production run is T = 80. Less than one period, against a default N = 50."""
    tau = characteristic_period(155.49)
    assert 40.0 / tau < 1.0, "untying should fall inside one characteristic period"
    assert 80.0 / tau < 1.0, "the whole run is under one characteristic period"
