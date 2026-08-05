"""Pins the choices that only matter once a still becomes a sequence.

Rendering one field is forgiving. Rendering a run is not: a rule that looks fine
on frame 0 can make the object appear to breathe, evaporate, or shatter over the
following frames, and each of those reads as physics that is not there. Every test
here corresponds to one such failure that actually occurred while building viz.py
against a real-time trefoil.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("skimage")
pytest.importorskip("scipy")

from soliton_playground import viz  # noqa: E402

N, L, R, CORE = 32, 8.0, 2.2, 0.7
DX = L / N


def _ring_n_field(shift=0.0, spread=1.0):
    """A unit n-field with its energy concentrated on a ring core.

    n3 runs from -1 on the core to +1 far away through a tanh of the distance, and
    (n1, n2) carry the azimuth, so |n| = 1 everywhere by construction. spread > 1
    fattens the core, which is what real-time evolution does to a marginally
    resolved one.
    """
    g = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    X = X - shift
    rho = np.sqrt(X ** 2 + Y ** 2)
    d = np.sqrt((rho - R) ** 2 + Z ** 2)
    theta = np.pi * np.clip(d / (CORE * spread), 0.0, 1.0)
    phi = np.arctan2(Z, rho - R)
    n3 = -np.cos(theta)
    s = np.sqrt(np.clip(1.0 - n3 ** 2, 0.0, 1.0))
    return np.stack([s * np.cos(phi), s * np.sin(phi), n3])


def _write_seq(tmp_path, n_frames=4):
    """A short sequence whose core both fattens and drifts, so a fixed level and a
    frame-0 bounding box are both stressed.
    """
    paths = []
    for k in range(n_frames):
        n = _ring_n_field(shift=0.25 * k, spread=1.0 + 0.5 * k)
        p = tmp_path / f"n_{k:08d}.npz"
        np.savez_compressed(p, n=n.astype(np.float32), t=0.1 * k, L=L, c4=4.0,
                            Q_H=1.0)
        paths.append(p)
    return paths


# ----------------------------------------------------------- smoothing boundary
def test_smooth_wraps_instead_of_reflecting():
    """The box is periodic, so smoothing must wrap. scipy's default 'reflect'
    invents a mirror outside each face, which perturbs the boundary shell enough
    to move a quantile level and fragment the rendered core.
    """
    a = np.zeros((16, 16, 16))
    a[0, 8, 8] = 1.0
    got = viz.smooth_periodic(a, 1.0)
    assert got[-1, 8, 8] > 1e-3, (
        "signal did not cross the periodic boundary; smoothing is not wrapping")
    assert got[-1, 8, 8] == pytest.approx(got[1, 8, 8], rel=1e-6), (
        "the two neighbours of a face-adjacent cell should smooth identically "
        "under a periodic kernel")


# ----------------------------------------------------------------- level choice
def test_volume_level_encloses_the_requested_fraction():
    rng = np.random.default_rng(0)
    e = rng.gamma(2.0, size=(24, 24, 24))
    for vf in (0.002, 0.01, 0.05):
        frac = float((e > viz.volume_level(e, vf)).mean())
        assert frac == pytest.approx(vf, rel=0.08), f"vf={vf} got {frac}"


def test_fixed_level_from_frame_zero_can_evaporate(tmp_path):
    """Why volume_frac is the timelapse default. A level pinned to frame 0's
    maximum sat above every later frame's maximum on the real run, and marching
    cubes then correctly returned nothing at all.
    """
    paths = _write_seq(tmp_path, n_frames=4)
    e_of = lambda p: viz.faddeev_energy_density(
        *(viz.load_n_field(p, L=L, c4=4.0)[k] for k in ("n1", "n2", "n3")),
        DX, 4.0)
    maxes = [float(e_of(p).max()) for p in paths]
    assert maxes[-1] < maxes[0], (
        "the synthetic core did not decay, so this fixture proves nothing")
    level = viz.iso_level(e_of(paths[0]), 0.80)
    assert level > maxes[-1], (
        "frame-0 level should overshoot the decayed frames; if not, widen the "
        "spread in _ring_n_field")
    assert viz.iso_parts(e_of(paths[-1]), level, DX, (1, 1, 1, 1)) is None


def test_volume_level_keeps_a_surface_in_every_frame(tmp_path):
    """The same sequence under the default rule: a mesh in EVERY frame.

    This is the whole reason volume_frac is the timelapse default, and it is the
    strongest claim this fixture can carry. The separate finding that periodic
    smoothing keeps a spreading core in one connected piece was measured on real
    N=64 trefoil data (see viz.smooth_periodic); this N=32 synthetic is too
    coarse to reproduce it, so asserting it here would pin a number the fixture
    does not actually support.
    """
    paths = _write_seq(tmp_path, n_frames=4)
    for p in paths:
        F = viz.load_n_field(p, L=L, c4=4.0)
        e = viz.smooth_periodic(viz.faddeev_energy_density(
            F["n1"], F["n2"], F["n3"], F["dx"], F["c4"]), 1.0)
        lv = viz.volume_level(e, 0.008)
        assert viz.iso_parts(e, lv, F["dx"], (1, 1, 1, 1)) is not None, \
            f"no surface in {p.name}"


def test_scene_parts_reports_fragmentation(tmp_path):
    """The fragmentation diagnostic must actually fire, because a shattered mesh
    and a reconnection look identical in a GIF and the log is what tells them
    apart. A too-tight level catches only the hottest specks of the core; a
    sensible one gives a single piece.
    """
    p = _write_seq(tmp_path, n_frames=1)[0]
    F = viz.load_n_field(p, L=L, c4=4.0)
    kw = dict(sigma=1.0, step_size=1, cmap="twilight")
    _, tight = viz._scene_parts(F, lambda e: viz.volume_level(e, 0.0005), **kw)
    _, ok = viz._scene_parts(F, lambda e: viz.volume_level(e, 0.02), **kw)
    assert tight["ncomp"] > 1, "a too-tight level should report fragments"
    assert ok["ncomp"] == 1, f"a sensible level should be one tube, got {ok}"
    assert tight["level"] > ok["level"], "tighter volume fraction => higher level"


# --------------------------------------------------------------------- framing
def test_bbox_spans_every_part_not_just_the_first():
    """Framing is fitted to the union of the ends. A box fitted to frame 0 alone
    clips a core that grows, and a clipped surface shows a flat cut face that
    reads as structure.
    """
    small = (np.array([[[-1., 0, 0], [1, 0, 0], [0, 1, 0]]]), np.ones((1, 4)))
    big = (np.array([[[-5., 0, 0], [5, 0, 0], [0, 5, 0]]]), np.ones((1, 4)))
    _, half_small = viz.bbox_of([small])
    _, half_union = viz.bbox_of([small, big])
    assert half_union > half_small
    assert half_union >= 5.0, "union framing does not contain the larger part"


def test_bbox_of_empty_is_safe():
    center, half = viz.bbox_of([None, (np.zeros((0, 3, 3)), np.zeros((0, 4)))])
    assert half > 0 and center.shape == (3,)


# ------------------------------------------------------------------ end to end
def test_timelapse_writes_a_gif_and_frames(tmp_path):
    from PIL import Image

    paths = _write_seq(tmp_path, n_frames=4)
    out = tmp_path / "anim.gif"
    got = viz.timelapse(paths, out, volume_frac=0.008, sigma=1.0, fps=8,
                        figsize=(2.4, 2.4), dpi=60, L=L, c4=4.0)
    assert got == out and out.exists()
    with Image.open(out) as im:
        assert im.n_frames == 4
    frames = sorted((tmp_path / "anim_frames").glob("frame_*.png"))
    assert len(frames) == 4, "per-frame PNGs should survive for re-cutting"


def test_timelapse_drop_frames_removes_pngs(tmp_path):
    paths = _write_seq(tmp_path, n_frames=3)
    out = tmp_path / "anim.gif"
    viz.timelapse(paths, out, volume_frac=0.008, sigma=1.0, figsize=(2.0, 2.0),
                  dpi=50, L=L, c4=4.0, keep_frames=False)
    assert out.exists()
    assert not sorted((tmp_path / "anim_frames").glob("frame_*.png"))


def test_turntable_moves_the_camera_only(tmp_path):
    """Geometry is built once; frames must differ (camera moved) while the object
    is provably identical because it was never rebuilt.
    """
    from PIL import Image

    p = _write_seq(tmp_path, n_frames=1)[0]
    out = tmp_path / "tt.gif"
    viz.turntable(p, out, volume_frac=0.008, sigma=1.0, frames=4,
                  figsize=(2.4, 2.4), dpi=60, L=L, c4=4.0)
    fr = sorted((tmp_path / "tt_frames").glob("frame_*.png"))
    assert len(fr) == 4
    a, b = (np.asarray(Image.open(f).convert("L")) for f in (fr[0], fr[1]))
    assert not np.array_equal(a, b), "the camera did not move between frames"


def test_load_n_field_accepts_both_layouts(tmp_path):
    """The real-time leg saves a (3,N,N,N) stack; the catalog saves n1/n2/n3."""
    n = _ring_n_field()
    stacked = tmp_path / "stacked.npz"
    split = tmp_path / "split.npz"
    np.savez_compressed(stacked, n=n.astype(np.float32), L=L, c4=4.0)
    np.savez_compressed(split, n1=n[0], n2=n[1], n3=n[2], L=L, c4=4.0)
    A, B = viz.load_n_field(stacked), viz.load_n_field(split)
    assert A["N"] == B["N"] == N
    assert A["L"] == B["L"] == L
    assert np.allclose(A["n1"], B["n1"], atol=1e-6)
