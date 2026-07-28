"""The 1D-axis depletion metrics must agree with the 3D-coords formulation.

gpe_lab.depletion_metrics / dip_centroid_z were rewritten to use grid.axis()
(1D, N floats) instead of grid.coords()[2] (a materialized N^3 grid, built along
with two more it discarded) because the per-sample churn swap-thrashed the first
N=256 trefoil run. The rewrite is only safe if it is numerically identical AND
the 1D array broadcasts along the correct axis, so both are pinned here: a
transposed broadcast would silently mask in x instead of z and still produce
plausible-looking numbers, which is exactly the failure a census would not
notice.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import ndimage

from jax_solitons.grid import BoxGrid
from soliton_playground.gpe_lab import depletion_metrics, dip_centroid_z

N, L = 24, 12.0


def _grid():
    return BoxGrid(N=N, L=L, dtype=jnp.float64)


def _field(grid, seed=20260727):
    """A lopsided-in-z depleted field: two off-center voids of unequal size, so
    the z-centroid is nonzero and a transposed broadcast cannot coincide."""
    X, Y, Z = (np.asarray(c) for c in grid.coords())
    rng = np.random.default_rng(seed)
    d1 = np.sqrt(X**2 + Y**2 + (Z + 3.5) ** 2)
    d2 = np.sqrt((X - 1.0) ** 2 + Y**2 + (Z - 2.0) ** 2)
    n = np.tanh(d1 / 1.0) ** 2 * np.tanh(d2 / 0.6) ** 2
    n = np.clip(n + 0.01 * rng.standard_normal(n.shape), 0.0, None)
    return jnp.asarray(np.sqrt(n), dtype=jnp.complex128)


# --- reference implementations: the original 3D-coords code, verbatim --------
def _ref_depletion(psi, grid, thresh=0.5, zmax=None):
    dens = np.asarray(jnp.abs(psi) ** 2)
    z = np.asarray(grid.coords()[2])
    mask = dens < thresh
    if zmax is not None:
        mask &= z < zmax
    vol = float(mask.sum()) * grid.dx**3
    n_blobs = int(ndimage.label(mask)[1]) if mask.any() else 0
    zc = float(z[mask].mean()) if mask.any() else float("nan")
    return dict(V_dep=vol, n_blobs=n_blobs, z_dep=zc, n_min=float(dens.min()))


def _ref_dip(psi, grid, floor=0.05, zmax=None):
    dens = np.asarray(jnp.abs(psi) ** 2)
    w = np.clip(1.0 - dens - floor, 0.0, None)
    z = np.asarray(grid.coords()[2])
    if zmax is not None:
        w = np.where(z < zmax, w, 0.0)
    tot = w.sum()
    return float((w * z).sum() / tot) if tot > 0 else float("nan")


@pytest.mark.parametrize("zmax", [None, 0.0, -2.0])
def test_depletion_metrics_matches_3d_reference(zmax):
    grid = _grid()
    psi = _field(grid)
    got, ref = (depletion_metrics(psi, grid, zmax=zmax),
                _ref_depletion(psi, grid, zmax=zmax))
    assert got.keys() == ref.keys()
    for k in ref:
        assert got[k] == pytest.approx(ref[k], rel=1e-12, abs=1e-12), k


@pytest.mark.parametrize("zmax", [None, 0.0, -2.0])
def test_dip_centroid_matches_3d_reference(zmax):
    grid = _grid()
    psi = _field(grid)
    assert dip_centroid_z(psi, grid, zmax=zmax) == pytest.approx(
        _ref_dip(psi, grid, zmax=zmax), rel=1e-12, abs=1e-12)


def test_centroid_is_actually_in_z_not_a_transposed_axis():
    """Guards the broadcast direction: the field's depletion sits mostly at
    z < 0, and is deliberately NOT symmetric under swapping z with x or y, so a
    1D array broadcast along the wrong axis changes the answer."""
    grid = _grid()
    psi = _field(grid)
    z_dep = depletion_metrics(psi, grid)["z_dep"]
    assert z_dep < -0.5, f"expected a z<0 centroid, got {z_dep}"

    arr = np.asarray(psi)
    for swap in ((0, 2), (1, 2)):
        rolled = jnp.asarray(np.swapaxes(arr, *swap))
        other = depletion_metrics(rolled, grid)["z_dep"]
        assert not np.isclose(other, z_dep, atol=1e-6), (
            f"swapping axes {swap} left z_dep unchanged ({other}) — the metric "
            "is not reading the z axis")


def test_zmax_masks_in_z():
    """zmax must cut on z. With the cut at z=0 every retained voxel is z<0, so
    the centroid must be strictly negative and the volume must shrink."""
    grid = _grid()
    psi = _field(grid)
    full, cut = depletion_metrics(psi, grid), depletion_metrics(psi, grid, zmax=0.0)
    assert cut["V_dep"] < full["V_dep"]
    assert cut["z_dep"] < 0.0
