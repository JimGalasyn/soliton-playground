"""The gauged two-scalar portraits: loader, vortex tracer, EM, field lines.

Ported from three separate files in null-worldtube-private (render_portrait,
nwt_em_fields, gpe_vortex_topology). The properties pinned here are the ones the
port has to preserve to be worth anything: that the tracer finds a core at all,
that the spectral curl is genuinely divergence-free, and that field lines stop
where the field stops meaning something.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("skimage")
pytest.importorskip("scipy")

from soliton_playground import viz_em  # noqa: E402

N, L, R, XI = 32, 12.0, 3.2, 0.9
DX = L / N


def _ring(N=N, L=L, R=R, xi=XI, axis="z"):
    """A vortex ring: |phi| = tanh(d/xi) vanishing on the core circle, with phase
    winding 2*pi around it -- so the phase-winding tracer has something to find."""
    g = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    if axis == "z":
        a1, a2, a3 = X, Y, Z
    else:
        a1, a2, a3 = Y, Z, X
    rho = np.sqrt(a1 ** 2 + a2 ** 2)
    d = np.sqrt((rho - R) ** 2 + a3 ** 2)
    return np.tanh(d / xi) * np.exp(1j * np.arctan2(a3, rho - R))


def _write_field(tmp_path, n_components=7):
    p1 = _ring()
    p2 = _ring(R=R * 0.7, axis="x")
    g = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, _ = np.meshgrid(g, g, g, indexing="ij")
    r2 = X ** 2 + Y ** 2 + 1.0
    u = np.zeros((n_components, N, N, N))
    for i, comp in enumerate((p1.real, p1.imag, p2.real, p2.imag,
                              -Y / r2, X / r2)):
        if i < n_components:                          # short-u fixtures are valid
            u[i] = comp
    s = np.exp(-(X ** 2 + Y ** 2) / 8.0)              # smooth scalar potential
    np.savez_compressed(tmp_path / "field.npz", u=u, s=s)
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"params": {"N": N, "L": L, "nlink": 1}, "cross_lk": 1.0}))
    return tmp_path


# --------------------------------------------------------------------- loader
def test_load_gauged_field_layout(tmp_path):
    F = viz_em.load_gauged_field(_write_field(tmp_path))
    assert F["N"] == N and F["L"] == L and F["dx"] == pytest.approx(DX)
    assert np.iscomplexobj(F["p1"]) and np.iscomplexobj(F["p2"])
    assert len(F["A"]) == 3 and F["s"] is not None
    assert F["n_components"] == 7


def test_load_gauged_field_rejects_short_u(tmp_path):
    with pytest.raises(SystemExit, match="need >= 7"):
        viz_em.load_gauged_field(_write_field(tmp_path, n_components=4))


def test_load_gauged_field_tolerates_extra_components(tmp_path):
    """Real battery fields carry ten components; the extras are undocumented and
    must be ignored rather than guessed at."""
    F = viz_em.load_gauged_field(_write_field(tmp_path, n_components=10))
    assert F["n_components"] == 10 and len(F["A"]) == 3


# --------------------------------------------------------------- vortex tracer
def test_vortex_skeleton_finds_the_ring():
    P, T, C = viz_em.vortex_skeleton(_ring())
    assert len(P) > 20, "no core segments found on a vortex ring"
    assert P.shape[1] == 3 and T.shape == P.shape
    assert np.allclose(np.linalg.norm(T, axis=1), 1.0), "tangents must be unit"


def test_vortex_skeleton_finds_nothing_in_a_uniform_phase():
    """Plaquette winding is sound-immune: a field with no vortex, even a bumpy
    one, must yield no core at all."""
    g = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    psi = (1.0 + 0.3 * np.sin(X) * np.cos(Y) * np.sin(Z)) * np.exp(1j * 0.4)
    P, _, _ = viz_em.vortex_skeleton(psi)
    assert len(P) == 0, f"phase winding invented {len(P)} segments from a ripple"


def test_core_curve_recovers_the_ring_geometry():
    core = viz_em.core_curve(_ring(), N, DX)
    assert core is not None and len(core) > 50
    r = np.linalg.norm(core[:, :2], axis=1)
    assert abs(r.mean() - R) < 1.0, f"ring radius {r.mean():.2f}, expected ~{R}"
    assert abs(core[:, 2]).max() < 1.5, "ring should lie near the z=0 plane"
    # closed: the ends meet
    step = np.linalg.norm(np.diff(core, axis=0), axis=1).mean()
    assert np.linalg.norm(core[0] - core[-1]) < 5 * step, "curve is not closed"


def test_core_curve_returns_none_without_a_vortex():
    g = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, _, _ = np.meshgrid(g, g, g, indexing="ij")
    assert viz_em.core_curve(np.ones_like(X) + 0j, N, DX) is None


# -------------------------------------------------------------------- EM parts
def test_curl_matches_an_analytic_field():
    """curl of (-y, x, 0)/1 is (0, 0, 2). Use a periodic-friendly sinusoid instead
    of a linear ramp, since a ramp is discontinuous across the box wrap."""
    g = np.linspace(-np.pi, np.pi, 32, endpoint=False)
    X, Y, _ = np.meshgrid(g, g, g, indexing="ij")
    dx = float(g[1] - g[0])
    F = [np.zeros_like(X), np.sin(X), np.zeros_like(X)]      # curl = (0,0,cos x)
    cx, cy, cz = viz_em.curl(F, dx)
    assert np.allclose(cz, np.cos(X), atol=1e-8)
    assert np.allclose(cx, 0.0, atol=1e-8) and np.allclose(cy, 0.0, atol=1e-8)


def test_curl_is_divergence_free():
    """The reason the curl is spectral rather than a stencil: B = curl A must come
    out solenoidal to machine precision, or field-line tracing sees spurious
    sources and sinks."""
    rng = np.random.default_rng(3)
    A = [viz_em.viz.smooth_periodic(rng.normal(size=(24, 24, 24)), 2.0)
         for _ in range(3)]
    dx = 0.3
    B = viz_em.curl(A, dx)
    KX, KY, KZ, _ = viz_em._k_grids(24, dx)
    div = np.real(np.fft.ifftn(1j * (KX * np.fft.fftn(B[0])
                                     + KY * np.fft.fftn(B[1])
                                     + KZ * np.fft.fftn(B[2]))))
    scale = max(np.abs(B[0]).max(), 1e-30)
    assert np.abs(div).max() / scale < 1e-10, "spectral curl is not solenoidal"


def test_electric_field_of_a_smooth_charge_points_outward():
    """A SMOOTH blob, not a single-cell delta. A delta has power out to Nyquist,
    so its spectral gradient rings: measured on one, E_x alternated sign cell to
    cell (-0.053, 0, +0.053, -0.005, +0.012) and only the immediate neighbours
    carried the physical sign. That is a property of the probe, not of the solver,
    and a Gaussian source avoids it entirely.
    """
    n, dx = 32, 0.4
    g = (np.arange(n) - n // 2) * dx
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    rho = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (4 * dx ** 2))
    E = viz_em.electric_field(rho, dx)
    for off in (2, 3, 4, 5):
        assert E[0][n // 2 + off, n // 2, n // 2] > 0, f"+x at +{off}"
        assert E[0][n // 2 - off, n // 2, n // 2] < 0, f"-x at -{off}"
    # and it falls off away from the source
    near = abs(E[0][n // 2 + 2, n // 2, n // 2])
    far = abs(E[0][n // 2 + 8, n // 2, n // 2])
    assert near > far, "field did not decay with distance"


# ---------------------------------------------------------------- field lines
def test_trace_field_lines_follows_a_uniform_field():
    g = np.linspace(-1.0, 1.0, 16)
    ones = np.ones((16, 16, 16))
    fld = [ones, np.zeros_like(ones), np.zeros_like(ones)]
    lines = viz_em.trace_field_lines(fld, g, [np.array([-0.9, 0.0, 0.0])],
                                     n_steps=40, ds=0.05, both_ways=False)
    assert len(lines) == 1
    ln = lines[0]
    assert np.allclose(ln[:, 1], 0.0, atol=1e-9), "line drifted off a straight field"
    assert ln[-1, 0] > ln[0, 0], "line ran backwards"


def test_min_mag_stops_lines_in_a_weak_field():
    """The fix for the boxy cage: where the field is weak its direction is noise,
    so a normalised tracer wanders and draws structure that is not there.
    """
    g = np.linspace(-1.0, 1.0, 16)
    X = np.meshgrid(g, g, g, indexing="ij")[0]
    strong = np.where(X < 0.0, 1.0, 1e-4)          # field dies for x > 0
    fld = [strong, np.zeros_like(strong), np.zeros_like(strong)]
    seed = [np.array([-0.9, 0.0, 0.0])]
    free = viz_em.trace_field_lines(fld, g, seed, n_steps=200, ds=0.02,
                                    both_ways=False)
    gated = viz_em.trace_field_lines(fld, g, seed, n_steps=200, ds=0.02,
                                     both_ways=False, min_mag=0.5)
    assert free and gated
    assert gated[0][-1, 0] < 0.15, "gated line ran on into the dead region"
    assert len(gated[0]) < len(free[0]), "min_mag did not shorten the line"


def test_mag_floor_uses_a_percentile_not_the_max():
    fld = [np.ones((8, 8, 8)), np.zeros((8, 8, 8)), np.zeros((8, 8, 8))]
    fld[0][0, 0, 0] = 1000.0                        # one hot cell
    assert viz_em._mag_floor(fld, 0.5) == pytest.approx(0.5, rel=0.05), \
        "one outlier cell moved the floor; use a percentile"


def test_lines_to_ribbons_are_quads_facing_the_camera():
    ln = [np.stack([np.linspace(0, 1, 10), np.zeros(10), np.zeros(10)], axis=1)]
    part = viz_em.lines_to_ribbons(ln, viz_em.view_dir(0.0, 0.0), 0.05, (1, 0, 0, 1))
    assert part is not None
    faces, cols = part
    assert faces.shape[1:] == (4, 3) and len(cols) == len(faces)
    assert viz_em.lines_to_ribbons([], viz_em.view_dir(0, 0), 0.1, (1, 1, 1, 1)) is None


def test_view_dir_is_a_unit_vector():
    for elev, azim in ((0, 0), (22, -56), (89, 130)):
        assert np.linalg.norm(viz_em.view_dir(elev, azim)) == pytest.approx(1.0)


# ------------------------------------------------------------------ the views
def test_view_raw_leaves_the_facets_in(tmp_path):
    F = viz_em.load_gauged_field(_write_field(tmp_path))
    parts, cap = viz_em.view_raw(F)
    assert len(parts) == 2, "both scalars should produce a surface"
    assert "no cheat" in cap
    assert parts[0][0].shape[1:] == (3, 3), "isosurfaces are triangles"


def test_view_cells_gives_cubes(tmp_path):
    F = viz_em.load_gauged_field(_write_field(tmp_path))
    parts, cap = viz_em.view_cells(F, volume_frac=0.02)
    assert parts and parts[0][0].shape[1:] == (4, 3), "cells are quads"
    assert "one cube per cell" in cap


def test_portrait_writes_each_view(tmp_path):
    fdir = _write_field(tmp_path)
    for view in ("raw", "cells", "twist", "bfield"):
        out = tmp_path / f"{view}.png"
        viz_em.portrait(fdir, view, out, dpi=50, figsize=(2.5, 2.5))
        assert out.exists() and out.stat().st_size > 0, view


def test_build_view_rejects_an_unknown_view(tmp_path):
    F = viz_em.load_gauged_field(_write_field(tmp_path))
    with pytest.raises(ValueError, match="unknown view"):
        viz_em.build_view(F, "hologram", 22.0, -56.0)


# ------------------------------------------------- modulated sources / cycle
def _reference_deposit(curve, XYZ, m, width):
    """Full-grid deposit, the way the upstream did it. The shipped version
    restricts each Gaussian to a local window; this is what it must reproduce."""
    X, Y, Z = XYZ
    C = np.asarray(curve, float)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    out = {k: (np.zeros(X.shape) if k.startswith("R")
               else [np.zeros(X.shape) for _ in range(3)])
           for k in ("Jdc", "Jc", "Js", "Rdc", "Rc", "Rs")}
    M = len(C)
    for i in range(M):
        p, t = C[i], T[i]
        g = np.exp(-((X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2)
                   / (2 * width ** 2))
        cph, sph = np.cos(2 * np.pi * m * i / M), np.sin(2 * np.pi * m * i / M)
        out["Rdc"] += g
        out["Rc"] += g * cph
        out["Rs"] += g * sph
        for d in range(3):
            out["Jdc"][d] += g * t[d]
            out["Jc"][d] += g * cph * t[d]
            out["Js"][d] += g * sph * t[d]
    return out


def _small_grid(n=20, box=8.0):
    g = np.linspace(-box / 2, box / 2, n, endpoint=False)
    return g, float(box / n), np.meshgrid(g, g, g, indexing="ij")


def _circle(n=64, r=2.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t), np.zeros_like(t)], axis=1)


def test_windowed_deposit_matches_the_full_grid():
    """The local-window optimisation must be numerically equivalent. The full-grid
    version costs len(curve) * N^3 exponentials -- 283 million on a 320-point core
    at N=96 -- which is the only reason to window it at all.
    """
    g, dx, XYZ = _small_grid()
    C = _circle()
    width = 2.0 * dx
    got = viz_em.deposit_modulated_sources(C, XYZ, 3, width=width, dx=dx)
    ref = _reference_deposit(C, XYZ, 3, width)
    for k in ("Rdc", "Rc", "Rs"):
        scale = max(np.abs(ref[k]).max(), 1e-30)
        assert np.abs(got[k] - ref[k]).max() / scale < 2e-3, k
    for k in ("Jdc", "Jc", "Js"):
        for d in range(3):
            scale = max(np.abs(ref[k][d]).max(), 1e-30)
            assert np.abs(got[k][d] - ref[k][d]).max() / scale < 2e-3, f"{k}[{d}]"


def test_deposit_width_defaults_to_cells_not_a_fixed_length():
    """width must scale with dx. The upstream's hard-coded 0.6 was ~2 cells on its
    grid (dx=0.3); reused at dx=0.8 it is a sub-cell spike that aliases.
    """
    g, dx, XYZ = _small_grid()
    C = _circle()
    auto = viz_em.deposit_modulated_sources(C, XYZ, 3, dx=dx)
    explicit = viz_em.deposit_modulated_sources(C, XYZ, 3, width=2.0 * dx, dx=dx)
    assert np.allclose(auto["Rdc"], explicit["Rdc"])
    narrow = viz_em.deposit_modulated_sources(C, XYZ, 3, width=0.25 * dx, dx=dx)
    assert narrow["Rdc"].max() < auto["Rdc"].max(), \
        "a sub-cell width should deposit less total weight per cell"


def test_deposit_dc_is_positive_and_ac_oscillates():
    g, dx, XYZ = _small_grid()
    C = _circle()
    S = viz_em.deposit_modulated_sources(C, XYZ, 3, dx=dx)
    assert S["Rdc"].min() >= 0.0 and S["Rdc"].max() > 0.0
    # an m>=1 harmonic integrates to ~zero, which is why the periodic solver is
    # legitimate for the AC parts even though it cannot carry a net charge
    dc = S["Rdc"].sum()
    for k in ("Rc", "Rs"):
        assert abs(S[k].sum()) < 0.05 * dc, f"{k} carries net charge"


def test_deposit_current_follows_the_tangent():
    """A circle in the z=0 plane has no z-current."""
    g, dx, XYZ = _small_grid()
    S = viz_em.deposit_modulated_sources(_circle(), XYZ, 2, dx=dx)
    jz = np.abs(S["Jdc"][2]).max()
    jxy = max(np.abs(S["Jdc"][0]).max(), np.abs(S["Jdc"][1]).max())
    assert jz < 1e-9 * max(jxy, 1e-30) + 1e-12, "current left the curve's plane"


def test_magnetic_field_is_divergence_free():
    g, dx, XYZ = _small_grid(n=24)
    S = viz_em.deposit_modulated_sources(_circle(), XYZ, 1, dx=dx)
    B = viz_em.magnetic_field(S["Jdc"], dx)
    KX, KY, KZ, _ = viz_em._k_grids(24, dx)
    div = np.real(np.fft.ifftn(1j * (KX * np.fft.fftn(B[0])
                                     + KY * np.fft.fftn(B[1])
                                     + KZ * np.fft.fftn(B[2]))))
    assert np.abs(div).max() / max(np.abs(B[0]).max(), 1e-30) < 1e-10


def test_cycle_animates_the_phase_without_moving_the_object(tmp_path):
    """The whole point: colour travels around a tube that does not move. Geometry
    and the base phase are sampled once, so the silhouette must be frame-invariant
    while the colours on it change.
    """
    from PIL import Image

    fdir = _write_field(tmp_path)
    out = tmp_path / "cyc.gif"
    viz_em.cycle(fdir, out, n_frames=6, n_cycles=1, fields="none",
                 dpi=60, figsize=(2.6, 2.6))
    assert out.exists()
    fr = sorted((tmp_path / "cyc_frames").glob("frame_*.png"))
    assert len(fr) == 6
    a = np.asarray(Image.open(fr[0]).convert("RGB")).astype(int)
    b = np.asarray(Image.open(fr[3]).convert("RGB")).astype(int)
    lit_a, lit_b = a.sum(-1) > 40, b.sum(-1) > 40
    iou = (lit_a & lit_b).sum() / max(1, (lit_a | lit_b).sum())
    assert iou > 0.95, f"the object moved between frames (IoU {iou:.3f})"
    shared = lit_a & lit_b
    assert np.abs(a - b)[shared].mean() > 5.0, "the phase did not travel"


def test_cycle_rejects_an_unknown_fields_mode(tmp_path):
    with pytest.raises(ValueError, match="fields must be one of"):
        viz_em.cycle(_write_field(tmp_path), tmp_path / "x.gif", fields="magic")


def test_cycle_travelling_fields_actually_change(tmp_path):
    """fields="travelling" recombines the cos/sin field components per frame, so
    the field lines must differ between frames -- otherwise the linearity trick is
    wired up wrong and the fields are merely static.
    """
    from PIL import Image

    fdir = _write_field(tmp_path)
    out = tmp_path / "trav.gif"
    viz_em.cycle(fdir, out, n_frames=4, n_cycles=1, fields="travelling", m=2,
                 dpi=60, figsize=(2.6, 2.6))
    fr = sorted((tmp_path / "trav_frames").glob("frame_*.png"))
    assert len(fr) == 4
    def blue(p):
        a = np.asarray(Image.open(p).convert("RGB")).astype(int)
        return a[..., 2] - a[..., 0] > 40
    m0, m2 = blue(fr[0]), blue(fr[2])
    if m0.any() or m2.any():
        iou = (m0 & m2).sum() / max(1, (m0 | m2).sum())
        assert iou < 0.98, "B lines identical across half a cycle; not travelling"
