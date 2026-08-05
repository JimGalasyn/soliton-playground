"""The optional render modes: cell clouds, unsmoothed facets, and swept tubes.

These were ported from null-worldtube-private, where they existed as five separate
scripts. Each test here pins either a property the port must preserve or a defect
found while porting it.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("skimage")

import matplotlib.pyplot as plt  # noqa: E402

from soliton_playground import viz  # noqa: E402

N, L = 24, 6.0
DX = L / N


def _blob():
    """A single hot cell plus a warm shell, so a level cuts a known cell count."""
    a = np.zeros((N, N, N))
    a[12, 12, 12] = 10.0
    a[11:14, 11:14, 11:14] = np.maximum(a[11:14, 11:14, 11:14], 5.0)
    a[12, 12, 12] = 10.0
    return a


def _ring_curve(n=160, R=2.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([R * np.cos(t), R * np.sin(t), np.zeros_like(t)], axis=1)


def _trefoil_curve(n=240, R=2.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([(R + np.cos(3 * t)) * np.cos(2 * t),
                     (R + np.cos(3 * t)) * np.sin(2 * t),
                     np.sin(3 * t)], axis=1)


# ------------------------------------------------------------------ cell cloud
def test_cell_parts_makes_six_quads_per_cell():
    a = _blob()
    part = viz.cell_parts(a, 4.0, DX)
    assert part is not None
    faces, cols = part
    n_cells = int((a > 4.0).sum())
    assert len(faces) == 6 * n_cells, "a cube is six quads"
    assert faces.shape[1:] == (4, 3), "cell faces must be quads"
    assert len(cols) == len(faces), "one colour per face, not per cell"


def test_cell_parts_cubes_are_cell_sized_and_placed_on_cells():
    """Cube extent must track dx, so the picture does not silently rescale with
    dpi the way screen-space scatter markers do."""
    a = np.zeros((N, N, N))
    a[5, 6, 7] = 1.0
    faces, _ = viz.cell_parts(a, 0.5, DX, shrink=1.0)
    span = faces.reshape(-1, 3).max(axis=0) - faces.reshape(-1, 3).min(axis=0)
    assert np.allclose(span, DX), f"cube span {span} should equal dx={DX}"
    centre = faces.reshape(-1, 3).mean(axis=0)
    expect = (np.array([5, 6, 7]) + 0.5) * DX - 0.5 * np.array([N, N, N]) * DX
    assert np.allclose(centre, expect, atol=1e-9)


def test_cell_parts_truncation_keeps_the_hottest(capsys):
    a = np.arange(N ** 3, dtype=float).reshape(N, N, N)
    part = viz.cell_parts(a, -1.0, DX, max_cells=50)
    assert part is not None
    faces, _ = part
    assert len(faces) == 6 * 50
    assert "keeping the hottest" in capsys.readouterr().out, \
        "truncation must be reported, not silent"


def test_cell_parts_empty_returns_none():
    assert viz.cell_parts(np.zeros((8, 8, 8)), 1.0, DX) is None


# ------------------------------------- mixing polygon kinds in one collection
def test_add_parts_mixes_triangles_and_quads():
    """The regression that broke every gauged view: parts legitimately mix vertex
    counts (marching cubes gives triangles, cubes and tubes give quads), so
    add_parts must accumulate a LIST and not np.concatenate.
    """
    tris = (np.zeros((3, 3, 3)), np.ones((3, 4)))
    quads = (np.zeros((2, 4, 3)), np.ones((2, 4)))
    fig = plt.figure(figsize=(2, 2), dpi=50)
    ax = fig.add_subplot(111, projection="3d")
    n = viz.add_parts(ax, [tris, quads])
    assert n == 5, "three triangles plus two quads is five polygons"
    assert len(ax.collections) == 1, "still exactly one collection"
    plt.close(fig)


def test_add_parts_rejects_colour_count_mismatch():
    bad = (np.zeros((3, 4, 3)), np.ones((2, 4)))
    fig = plt.figure(figsize=(2, 2), dpi=50)
    ax = fig.add_subplot(111, projection="3d")
    with pytest.raises(ValueError, match="one colour per face"):
        viz.add_parts(ax, [bad])
    plt.close(fig)


# ------------------------------------------------------------- facets stance
def test_facets_mode_forces_no_smoothing():
    """--mode facets is the upstream "no cheat" stance: the grid facets stay in.
    A stale --sigma must not quietly smooth them away.
    """
    _, sig, tag = viz._make_scene("facets", sigma=3.0, step_size=1, cmap=None,
                                 max_cells=10)
    assert sig == 0.0, "facets mode must override sigma, not defer to it"
    assert "no cheat" in tag


def test_surface_mode_keeps_requested_smoothing():
    _, sig, tag = viz._make_scene("surface", sigma=1.5, step_size=1, cmap=None,
                                 max_cells=10)
    assert sig == 1.5 and "facets" not in tag


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        viz._make_scene("voxels", sigma=1.0, step_size=1, cmap=None, max_cells=1)


# ---------------------------------------------------------------- swept tubes
def test_rmf_frame_is_orthonormal_and_closes():
    """The holonomy unwind is what makes a swept tube seamless. Without it M[0]
    and M[-1] disagree and the tube shows a poloidal seam, and any colour spiral
    on it is partly frame artefact rather than real framing twist.
    """
    C = _trefoil_curve()
    T, M, B = viz.rmf_frame(C)
    for name, V in (("T", T), ("M", M), ("B", B)):
        assert np.allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-6), name
    assert np.allclose(np.einsum("ij,ij->i", T, M), 0.0, atol=1e-6)
    assert np.allclose(np.einsum("ij,ij->i", T, B), 0.0, atol=1e-6)
    # the frame closes: first and last normals agree once the holonomy is removed
    step = np.linalg.norm(np.diff(C, axis=0), axis=1).mean()
    assert np.linalg.norm(M[0] - M[-1]) < 6 * step, \
        "RMF did not close; the tube will show a seam"


def test_tube_parts_are_quads_on_the_curve():
    C = _ring_curve()
    faces, cols = viz.tube_parts(C, 0.25, npol=16)
    assert faces.shape[1:] == (4, 3) and len(cols) == len(faces)
    r = np.linalg.norm(faces.reshape(-1, 3)[:, :2], axis=1)
    assert r.min() > 1.5 and r.max() < 2.5, "tube is not wrapped around the ring"


def test_tube_parts_too_short_a_curve_returns_none():
    assert viz.tube_parts(np.zeros((2, 3)), 0.2) is None


def test_kelvin_deform_adds_transverse_lobes():
    """A Kelvin wave is transverse, so the deformation must move points off the
    curve without changing its length scale wildly."""
    C = _ring_curve(n=200, R=2.0)
    D = viz.kelvin_deform(C, mode=5, amp=0.3)
    off = np.linalg.norm(D - C, axis=1)
    assert np.allclose(off, 0.3, atol=1e-6), "displacement should be the amplitude"
    assert viz.kelvin_deform(C, 5, 0.0).shape == C.shape
    assert np.allclose(viz.kelvin_deform(C, 5, 0.0), C, atol=1e-9)


def test_standing_coloring_darkens_nodes():
    """coloring="standing" is a different physical claim from "phase": the nodes
    of a standing wave must actually go dark, which a cyclic hue never does.
    """
    theta = np.stack([np.linspace(0, 2 * np.pi, 33)] * 4, axis=0)
    standing = viz.theta_facecolors(theta, "standing")
    phase = viz.theta_facecolors(theta, "phase")
    lum = standing[:, :3].sum(axis=1)
    assert lum.min() < 0.25 * lum.max(), "standing wave has no dark nodes"
    assert phase[:, :3].sum(axis=1).min() > 0.25, "cyclic hue should stay lit"


def test_unknown_coloring_is_rejected():
    with pytest.raises(ValueError, match="unknown coloring"):
        viz.theta_facecolors(np.zeros((4, 4)), "rainbow-sparkle")


def test_quad_phase_colors_survive_the_branch_cut():
    """Averaging wrapped phase arithmetically flips a quad straddling 2*pi to the
    opposite colour -- a speckle on a still, a flickering seam in an animation.
    Circular averaging must give the two sides of the cut nearly equal colours.
    """
    lo = np.full((2, 2), 0.999)
    hi = np.full((2, 2), 0.001)
    seam = np.stack([lo.ravel(), hi.ravel()], axis=0)
    cols = viz.quad_phase_colors(seam, "twilight")
    mid = viz.quad_phase_colors(np.full((2, 4), 0.5), "twilight")
    assert cols.shape[1] == 4 and mid.shape[1] == 4
    # the seam colour must be near the phase-0 colour, not the opposite one
    zero = viz.quad_phase_colors(np.zeros((2, 4)), "twilight")[0, :3]
    half = mid[0, :3]
    d_zero = np.linalg.norm(cols[0, :3] - zero)
    d_half = np.linalg.norm(cols[0, :3] - half)
    assert d_zero < d_half, "branch-cut quad averaged to the opposite colour"
