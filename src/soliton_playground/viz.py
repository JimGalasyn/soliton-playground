"""Frame-by-frame 3D rendering and animated-GIF assembly.

Factored out of null-worldtube-private, where the same isosurface code had been
rewritten five times (turntable_hopfion, render_hopfion, render_portrait,
nwt_surface_current_portraits, nwt_deuteron_portrait) across three incompatible
GIF idioms. What is kept here is the part that was hard to get right.

THE DEPTH PROBLEM, AND WHY add_parts EXISTS
-------------------------------------------
mplot3d has no z-buffer. It is a painter's-algorithm renderer: it asks each
artist for ONE depth via do_3d_projection and draws whole artists back to front.
A Poly3DCollection is therefore ATOMIC -- every face of it lands either wholly in
front of or wholly behind every face of the next collection. Draw two linked
tubes as two collections and one passes in front at BOTH crossings, so the link
reads as two adjacent rings and the knot loses its weave.

Insertion order is not a workaround. Measured on a Hopf link of two isosurface
tori (tests/test_viz_depth.py, which holds this result as a regression):

    two collections : the near tube won 0.6% of contested pixels  -- uniform
    one collection  : the two tubes won 45% / 55%                 -- weaving
    swapping insertion order changed nothing (byte-identical renders)

So the ONLY fix is to merge every triangle into a single Poly3DCollection and let
matplotlib depth-sort face by face. That is what add_parts does, and it is the
reason this module represents geometry as (faces, colors) "parts" that accumulate
in a list rather than drawing anything as it goes.

Two consequences follow, and neither is optional:

  * A merged collection carries explicit per-face colors, which means shade=False,
    which means matplotlib's own lighting is gone. Shading has to be baked into
    the face colors instead -- shade_faces. Merging and baked shading are a
    package deal; you cannot have the first without the second.
  * Face-centroid sorting ('average' zsort, the default) is only as accurate as
    the faces are small. A coarse marching-cubes step_size makes big triangles
    whose centroid ordering disagrees with their pixels, and the weave degrades.
    Hence step_size=1 by default here, where the upstream turntable used 2.

WHAT ANIMATION ADDS
-------------------
Three things a single still never had to face. Each was got wrong first and fixed
against measurements on a real-time trefoil; the fix is not the obvious one in any
of the three cases.

  * THE CAMERA must be resolved once, not per frame. Auto-framing each frame makes
    the view drift, which is indistinguishable from the object moving. bbox_of
    fixes centre and half-width, spanning the FIRST and LAST meshes so a growing
    core is not clipped at the end of the run.

  * THE LEVEL must not be a fraction of a maximum. "level_frac * max" is the
    upstream convention and it fails twice over: a few leftover hot cells set the
    maximum, so 0.80 of it landed in the far tail and rendered 168 faces of
    scatter from a clean trefoil; and pinning that level from frame 0 put it above
    every later frame's maximum, so the surface did not breathe, it vanished. The
    default is volume_level -- a quantile, "the surface around the hottest 0.4% of
    the box" -- which is re-derived per frame ON PURPOSE. It trades away visible
    amplitude decay (read that off the energy series) to keep topology legible,
    which is the question a real-time run is asking.

  * THE SMOOTHING must be periodic. The box is periodic and scipy's default is
    'reflect', which invents a mirror neighbour outside each face; that was enough
    to move a quantile level and shatter the rendered core into pieces in 4 of 17
    frames, and no sigma or volume fraction fixed it. With wrap, every frame came
    out as one clean tube across every setting tried.

This module is deliberately CPU-only -- numpy, matplotlib, scikit-image, scipy,
Pillow, and no jax -- so that rendering a run's frames never contends for VRAM
with a relaxation that is still going.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from PIL import Image  # noqa: E402

# A "part" is (faces, colors): faces (nface, nvert, 3) in physical coords,
# colors (nface, 4) RGBA. Parts are merged by add_parts into ONE collection.
Part = tuple[np.ndarray, np.ndarray]

LIGHT = (0.35, 0.5, 0.9)


# ------------------------------------------------------------------ geometry
def shade_faces(faces, rgba, light=LIGHT) -> np.ndarray:
    """Bake two-sided Lambert shading into per-face colours.

    Required by add_parts: a merged collection must be drawn with shade=False, so
    this is the only remaining source of 3D form. Two-sided (|n.l|) because
    marching-cubes winding is not reliably outward and one-sided shading leaves
    black patches where the normals flip.
    """
    faces = np.asarray(faces, float)
    n = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    lt = np.asarray(light, float)
    lt = lt / np.linalg.norm(lt)
    s = 0.32 + 0.68 * np.abs(n @ lt)
    cols = np.ones((len(faces), 4))
    cols[:, :3] = np.asarray(rgba[:3], float)[None, :] * s[:, None]
    cols[:, 3] = rgba[3] if len(rgba) > 3 else 1.0
    return cols


def smooth_periodic(scalar, sigma: float) -> np.ndarray:
    """Gaussian smoothing with PERIODIC wrap, because the box is periodic.

    scipy's default mode is 'reflect', which invents a mirror of the field just
    outside every face. On a periodic box that is the wrong neighbour, and it
    perturbs the field in the boundary shell enough to change where a quantile
    level falls. Measured: with 'reflect' the rendered core of a real-time trefoil
    broke into 2-3 disconnected components in 4 of 17 frames and no choice of
    sigma or volume fraction fixed it; with 'wrap' every frame came out as one
    clean tube for every sigma in 0.8..2.0 and every volume fraction in
    0.004..0.008. The fix was the boundary condition, not a tuned parameter.
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(np.asarray(scalar, float), sigma, mode="wrap")


def iso_level(scalar, level_frac: float) -> float:
    """Resolve a fractional isosurface level to an absolute one, ONCE.

    Call this on a reference frame and reuse the result for every frame of an
    animation. Re-deriving level_frac * max per frame ties the surface to the
    field's instantaneous amplitude, so it inflates and deflates with the energy
    and reads as motion the field is not doing.

    For a sequence, prefer scan_level: a level taken from frame 0 alone can sit
    above every later frame's maximum, and then the surface does not breathe --
    it disappears.
    """
    return float(level_frac) * float(np.asarray(scalar).max())


def volume_level(scalar, volume_frac: float) -> float:
    """Level enclosing the hottest volume_frac of the box.

    This is the level rule that keeps a knot readable across a run, and it is the
    timelapse default. It is a quantile, so it is a geometric statement -- "the
    surface around the hottest 0.2% of cells" -- and unlike a fraction of the
    maximum it does not hang off one hot cell.

    It IS re-derived per frame, which is a deliberate trade and the opposite of
    what iso_level warns about. Measured on the same real-time trefoil: at a fixed
    absolute level the mesh went from 8212 faces to a handful of specks as the
    core spread, so the topology became unreadable exactly when it mattered; at a
    fixed volume fraction the mesh held 3564 -> 4190 faces and the weave stayed
    legible start to finish.

    What that hides is amplitude: a thinning core renders the same size. So the
    decay is NOT visible in the animation and must be read from the energy series
    in summary.json, which is where it is quantitative anyway. Use level= for a
    fixed absolute level when amplitude is the thing being shown.
    """
    return float(np.quantile(np.asarray(scalar), 1.0 - float(volume_frac)))


def scan_level(field_paths: Sequence, level_frac: float, *, samples: int = 5,
               L=None, c4: float = 6.0, sigma: float = 0.0) -> tuple[float, dict]:
    """Pick ONE isosurface level that yields a surface in EVERY frame.

    Taking level_frac * max from the first frame is not enough. Measured on a
    real-time trefoil (N=64, core/dx=2.50): peak energy density fell 54.1 -> 21.1
    over the run, a factor of 2.6, while the total only fell 17% -- the core was
    spreading, not radiating. A level of 0.80*54.1 = 43.3 therefore exceeded the
    maximum of all but the first three frames, and marching cubes correctly
    returned nothing for the rest.

    So the level is level_frac * min(max) over a sample of frames, which keeps it
    fixed (no breathing) while guaranteeing it stays inside the data everywhere.
    A wide spread in the report is worth reading rather than smoothing over: it
    means the core's peak is decaying, which is physics, and the animation will
    honestly show the surface thinning.
    """
    paths = list(field_paths)
    k = max(1, min(int(samples), len(paths)))
    picks = np.unique(np.linspace(0, len(paths) - 1, k).astype(int))
    maxes = []
    for i in picks:
        F = load_n_field(paths[i], L=L, c4=c4)
        e = faddeev_energy_density(F["n1"], F["n2"], F["n3"], F["dx"], F["c4"])
        if sigma > 0:
            e = smooth_periodic(e, sigma)
        maxes.append(float(e.max()))
    lo, hi = min(maxes), max(maxes)
    return float(level_frac) * lo, dict(
        sampled=[int(i) for i in picks], maxes=maxes, min_max=lo, max_max=hi,
        decay=hi / lo if lo > 0 else float("inf"))


def iso_parts(scalar, level: float, dx: float, rgba, *, sigma: float = 0.0,
              step_size: int = 1, center: bool = True) -> Part | None:
    """Isosurface of a scalar field as one shaded part, or None if empty.

    step_size stays at 1 unless you have measured that you can afford otherwise:
    it is the mesh coarseness, and coarse meshes break the face-centroid depth
    sort that makes a knot weave (see module docstring).
    """
    from skimage import measure

    sc = np.asarray(scalar, float)
    if sigma > 0:
        sc = smooth_periodic(sc, sigma)
    if not (sc.min() < level < sc.max()):
        return None                      # level outside the data: no surface
    verts, faces, _, _ = measure.marching_cubes(
        sc, level=level, spacing=(dx, dx, dx), step_size=step_size)
    if center:
        verts = verts - 0.5 * np.asarray(sc.shape, float) * dx
    tri = verts[faces]
    return tri, shade_faces(tri, rgba)


def phase_facecolors(faces, n1, n2, dx, *, cmap="twilight", alpha=1.0,
                     center: bool = True, light=LIGHT) -> np.ndarray:
    """Colour faces by the azimuthal phase atan2(n2, n1), with shading baked in.

    The per-face phase is a CIRCULAR mean over the face's vertices. Averaging the
    wrapped value linearly (as the upstream turntable did) is wrong at the 2*pi
    branch cut: a face straddling the cut averages to the opposite colour, which
    on a still is a stray speckle and across an animation is a band of pixels
    that flickers every frame.
    """
    faces = np.asarray(faces, float)
    n1, n2 = np.asarray(n1), np.asarray(n2)
    shape = np.asarray(n1.shape, float)
    pts = faces.reshape(-1, 3)
    if center:
        pts = pts + 0.5 * shape * dx
    idx = np.clip(np.round(pts / dx).astype(int), 0, n1.shape[0] - 1)
    phi = np.arctan2(n2[idx[:, 0], idx[:, 1], idx[:, 2]],
                     n1[idx[:, 0], idx[:, 1], idx[:, 2]])
    z = np.exp(1j * phi).reshape(len(faces), -1).mean(axis=1)
    frac = (np.angle(z) / (2 * np.pi)) % 1.0
    cols = matplotlib.colormaps[cmap](frac)
    n = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    lt = np.asarray(light, float)
    lt = lt / np.linalg.norm(lt)
    cols[:, :3] *= (0.55 + 0.45 * np.abs(n @ lt))[:, None]
    cols[:, 3] = alpha
    return cols


def quad_phase_colors(phase01, cmap="twilight_shifted", alpha=1.0) -> np.ndarray:
    """Per-quad colours from a per-vertex cyclic phase in [0,1).

    The four corner phases are averaged ON THE CIRCLE, not arithmetically, so a
    quad straddling the 2*pi branch cut does not average to the opposite colour.
    Arithmetic averaging leaves a seam of wrong-coloured quads that is a speckle
    on a still and a flickering line across an animation.
    """
    z = np.exp(2j * np.pi * np.asarray(phase01, float))
    zq = 0.25 * (z + np.roll(z, -1, 0)
                 + np.roll(np.roll(z, -1, 0), -1, 1) + np.roll(z, -1, 1))
    cols = matplotlib.colormaps[cmap]((np.angle(zq).reshape(-1)
                                       / (2 * np.pi)) % 1.0)
    cols[:, 3] = alpha
    return cols


def bbox_of_points(points, pad: float = 1.12, fallback: float = 1.0):
    """(center, half) enclosing a point cloud or polyline."""
    p = np.asarray(points, float).reshape(-1, 3)
    if not len(p):
        return np.zeros(3), fallback
    lo, hi = p.min(axis=0), p.max(axis=0)
    return 0.5 * (lo + hi), float((hi - lo).max()) * 0.5 * pad or fallback


def bbox_of(parts: Sequence[Part], pad: float = 1.12, fallback: float = 1.0):
    """(center, half) enclosing every face, resolved ONCE for a whole animation.

    Re-fitting per frame makes the camera breathe and drift, which is
    indistinguishable from the object moving.
    """
    pts = [p[0].reshape(-1, 3) for p in parts if p is not None and len(p[0])]
    if not pts:
        return np.zeros(3), fallback
    v = np.concatenate(pts)
    lo, hi = v.min(axis=0), v.max(axis=0)
    center = 0.5 * (lo + hi)
    return center, float((hi - lo).max()) * 0.5 * pad


# ---------------------------------------------------------------- cell clouds
# A unit cube's six faces as quads, half-size 1, centred on the origin.
_CUBE = np.array([
    [[+1, -1, -1], [+1, +1, -1], [+1, +1, +1], [+1, -1, +1]],   # +x
    [[-1, -1, -1], [-1, -1, +1], [-1, +1, +1], [-1, +1, -1]],   # -x
    [[-1, +1, -1], [-1, +1, +1], [+1, +1, +1], [+1, +1, -1]],   # +y
    [[-1, -1, -1], [+1, -1, -1], [+1, -1, +1], [-1, -1, +1]],   # -y
    [[-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1]],   # +z
    [[-1, -1, -1], [-1, +1, -1], [+1, +1, -1], [+1, -1, -1]],   # -z
], float)


def cell_parts(scalar, level: float, dx: float, *, cmap="inferno", alpha=1.0,
               max_cells: int = 20000, shrink: float = 0.86,
               center: bool = True, light=LIGHT) -> Part | None:
    """The individual grid cells above `level`, drawn as actual cubes.

    This is the "show me the lattice" view: one cube per cell, coloured by the
    field value there, so the render admits its own resolution instead of
    implying a smooth object. The upstream version (nwt_alpha_portrait's
    make_alpha_shape) did this with ax.scatter of cell centres, which is cheaper
    but has two problems this fixes:

      * a scatter is its own artist, so it cannot z-interleave with an
        isosurface or a tube -- the cloud lands wholly in front or behind. Cubes
        come back as a normal part and merge through add_parts, so a cell cloud
        can be overlaid on a surface and still be read correctly.
      * markers are screen-space, so their apparent size does not track the cell
        size and the picture silently rescales with dpi and figure size.

    Cost is 6 quads per cell, so max_cells caps it; above the cap the HOTTEST
    max_cells cells are kept and the rest dropped, which is reported rather than
    silent because a truncated cloud is a misleading picture of an extended one.
    """
    sc = np.asarray(scalar, float)
    sel = sc > level
    n = int(sel.sum())
    if n == 0:
        return None
    vals = sc[sel]
    idx = np.argwhere(sel)
    if n > max_cells:
        keep = np.argpartition(vals, n - max_cells)[n - max_cells:]
        idx, vals = idx[keep], vals[keep]
        print(f"  cell cloud: {n} cells over the level, keeping the hottest "
              f"{max_cells} (raise max_cells or the level to see them all)")
    centres = (idx.astype(float) + 0.5) * dx
    if center:
        centres = centres - 0.5 * np.asarray(sc.shape, float) * dx
    faces = (centres[:, None, None, :]
             + _CUBE[None, :, :, :] * (0.5 * dx * shrink)).reshape(-1, 4, 3)

    lo, hi = float(vals.min()), float(vals.max())
    t = (vals - lo) / (hi - lo) if hi > lo else np.full(len(vals), 0.75)
    cols = matplotlib.colormaps[cmap](t)                     # per CELL
    cols = np.repeat(cols, len(_CUBE), axis=0)               # per FACE
    nrm = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    lt = np.asarray(light, float)
    lt = lt / np.linalg.norm(lt)
    cols[:, :3] *= (0.40 + 0.60 * np.abs(nrm @ lt))[:, None]
    cols[:, 3] = alpha
    return faces, cols


# --------------------------------------------------------------- swept tubes
def rmf_frame(curve):
    """Rotation-minimising frame (T, M, B) along a closed curve.

    The frame is double-reflection propagated and then its holonomy is removed by
    unwinding the residual angle linearly along the curve, so M closes on itself.
    Without that correction a swept tube has a visible poloidal seam where the
    frame fails to meet, and any colour spiral painted on it is partly frame
    artefact rather than real framing twist.
    """
    C = np.asarray(curve, float)
    S = len(C)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    ref = np.array([0.0, 0.0, 1.0])
    if abs(T[0] @ ref) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    M = [np.cross(T[0], ref) / (np.linalg.norm(np.cross(T[0], ref)) + 1e-12)]
    for i in range(1, S):
        v1 = C[i] - C[i - 1]
        c1 = v1 @ v1 + 1e-12
        rL = M[-1] - (2 / c1) * (v1 @ M[-1]) * v1
        tL = T[i - 1] - (2 / c1) * (v1 @ T[i - 1]) * v1
        v2 = T[i] - tL
        c2 = v2 @ v2 + 1e-12
        Mi = rL - (2 / c2) * (v2 @ rL) * v2
        M.append(Mi / (np.linalg.norm(Mi) + 1e-12))
    M = np.asarray(M)
    B = np.cross(T, M)
    axis = T[0]
    delta = np.arctan2(np.dot(np.cross(M[0], M[-1]), axis), np.dot(M[0], M[-1]))
    tw = -delta * np.arange(S) / (S - 1)
    cs, sn = np.cos(tw)[:, None], np.sin(tw)[:, None]
    return T, M * cs + B * sn, -M * sn + B * cs


def tube_surface(curve, radius: float, npol: int = 30):
    """Sweep a closed tube of the given radius. Returns (X, Y, Z, s, pol) with
    s in [0,1] along the curve and pol in [0,2pi) around it."""
    C = np.asarray(curve, float)
    _, M, B = rmf_frame(C)
    phi = np.linspace(0.0, 2 * np.pi, npol)
    cphi, sphi = np.cos(phi), np.sin(phi)
    X = C[:, 0, None] + radius * (M[:, 0, None] * cphi + B[:, 0, None] * sphi)
    Y = C[:, 1, None] + radius * (M[:, 1, None] * cphi + B[:, 1, None] * sphi)
    Z = C[:, 2, None] + radius * (M[:, 2, None] * cphi + B[:, 2, None] * sphi)
    s = np.linspace(0.0, 1.0, len(C))[:, None] + 0 * phi
    return X, Y, Z, s, phi[None, :] + 0 * s


def kelvin_deform(curve, mode: int, amp: float):
    """Helical Kelvin-wave deformation: a transverse standing wave with `mode`
    lobes, applied in the curve's own RMF normal plane so the excitation is
    genuinely transverse rather than a wobble in fixed lab axes."""
    C = np.asarray(curve, float)
    _, M, B = rmf_frame(C)
    s = 2 * np.pi * np.arange(len(C)) / len(C)
    return C + amp * (np.cos(mode * s)[:, None] * M
                      + np.sin(mode * s)[:, None] * B)


def surface_current_theta(s, pol, m: int, q: int = 1, phase_offset: float = 0.0):
    """The surface-current phase on a swept tube: m turns along the loop and q
    around it. Advancing phase_offset through 2*pi is exactly one period, so a
    sweep of it loops seamlessly."""
    return 2 * np.pi * m * s + q * pol + phase_offset


def theta_facecolors(theta, coloring="phase", cmap=None, alpha=1.0):
    """Per-quad colours from a per-vertex angle field.

    coloring="phase"    cyclic hue -- reads as a travelling current
    coloring="standing" intensity 0.5*(1+cos), so NODES GO DARK -- reads as a
                        standing wave, which is a different physical claim and
                        the reason both exist
    """
    th = np.asarray(theta, float)[:, :-1]
    if coloring == "phase":
        return matplotlib.colormaps[cmap or "hsv"](
            (th % (2 * np.pi)) / (2 * np.pi)).reshape(-1, 4)
    if coloring == "standing":
        cols = matplotlib.colormaps[cmap or "inferno"](
            0.5 * (1.0 + np.cos(th))).reshape(-1, 4)
        cols[:, 3] = alpha
        return cols
    raise ValueError(f"unknown coloring {coloring!r}")


def tube_parts(curve, radius: float, *, npol: int = 30, m: int = 0, q: int = 1,
               phase_offset: float = 0.0, coloring: str = "phase", cmap=None,
               alpha: float = 1.0, kelvin=None) -> Part | None:
    """A closed curve swept into a coloured tube, as one merged part.

    kelvin=(mode, amplitude) excites the filament first. m/q set the surface
    current the colouring paints; m=0, q=1 is a plain poloidal banding.
    """
    C = np.asarray(curve, float)
    if len(C) < 4:
        return None
    if kelvin is not None:
        C = kelvin_deform(C, kelvin[0], kelvin[1])
    X, Y, Z, s, pol = tube_surface(C, radius, npol)
    V = np.stack([X, Y, Z], -1)
    ip = (np.arange(V.shape[0]) + 1) % V.shape[0]
    faces = np.stack([V[:, :-1], V[ip][:, :-1], V[ip][:, 1:], V[:, 1:]],
                     axis=2).reshape(-1, 4, 3)
    theta = surface_current_theta(s, pol, m, q, phase_offset)
    return faces, theta_facecolors(theta, coloring, cmap, alpha)


# ------------------------------------------------------------------- drawing
def add_parts(ax, parts: Iterable[Part | None]) -> int:
    """Merge every part into ONE Poly3DCollection so faces depth-sort against
    each other. This is the whole reason the module is shaped this way -- see the
    module docstring. Returns the face count drawn.

    Do not "optimise" this into one collection per part. That is the bug.

    Polygons are accumulated as a LIST, not concatenated into one array, because
    parts legitimately mix vertex counts: marching cubes gives triangles while
    tubes, cell cubes and field-line ribbons give quads. Concatenating raises on
    the mismatch and would make exactly the mixed scenes -- a tube threaded by
    ribbons, a cell cloud over a surface -- impossible, which is the whole point
    of merging.
    """
    parts = [p for p in parts if p is not None and len(p[0])]
    if not parts:
        return 0
    polys: list[np.ndarray] = []
    for faces, _ in parts:
        polys.extend(np.asarray(faces, float))
    cols = np.concatenate([np.asarray(p[1], float) for p in parts])
    if len(cols) != len(polys):
        raise ValueError(f"{len(polys)} polygons but {len(cols)} colours; every "
                         "part must carry one colour per face")
    pc = Poly3DCollection(polys, facecolors=cols, linewidths=0, shade=False)
    pc.set_zsort("average")
    ax.add_collection3d(pc)
    return len(polys)


def dark_3d(ax) -> None:
    """Black 3D axes with no furniture. set_axis_off alone is not enough: the
    axes patch still paints its default light face behind the geometry.
    """
    ax.set_facecolor("black")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
        axis.line.set_color((0, 0, 0, 0))
    ax.grid(False)
    ax.set_axis_off()


def fit_axes(ax, center, half, zoom: float = 1.0) -> None:
    """Frame a cube of side 2*half on `center`, with equal aspect.

    zoom > 1 enlarges the object inside the axes. mplot3d reserves a lot of
    padding for axis furniture that set_axis_off then hides, so a correctly
    fitted object still lands in roughly a quarter of a panel's area; zoom
    reclaims it. Left at 1.0 by default so existing framing is unchanged, and
    raised deliberately where a panel is visibly under-filled.
    """
    for setlim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setlim(c - half, c + half)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)


def draw_scene(ax, parts, center, half, *, elev=22.0, azim=-58.0, title=None,
               zoom=1.0):
    """One merged, shaded, framed, dark 3D scene."""
    nf = add_parts(ax, parts)
    fit_axes(ax, center, half, zoom)
    ax.view_init(elev=elev, azim=azim)
    dark_3d(ax)
    if title:
        ax.set_title(title, color="#DDDDDD", fontsize=9, family="monospace")
    return nf


# ---------------------------------------------------------------- gif output
def _as_rgb(frame) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if isinstance(frame, (str, Path)):
        return Image.open(frame).convert("RGB")
    return Image.fromarray(np.asarray(frame, np.uint8)[..., :3]).convert("RGB")


def write_gif(frames: Sequence, out, *, fps: int = 18, loop: int = 0,
              palette_samples: int = 8) -> Path:
    """Assemble frames (paths, arrays, or PIL images) into an animated GIF.

    Frames are quantised against ONE palette shared by the whole animation,
    sampled from palette_samples frames spread across the sequence. Converting
    each frame with its own ADAPTIVE palette -- what the upstream turntable did --
    gives every frame a different 256 colours, so flat regions shift hue from
    frame to frame and the animation crawls even where the field is static.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("no frames to write")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    k = min(palette_samples, len(frames))
    picks = [_as_rgb(frames[i]) for i in
             np.unique(np.linspace(0, len(frames) - 1, k).astype(int))]
    w, h = picks[0].size
    montage = Image.new("RGB", (w, h * len(picks)))
    for i, im in enumerate(picks):
        montage.paste(im.resize((w, h)), (0, i * h))
    master = montage.quantize(colors=256, method=Image.MEDIANCUT)

    seq = [_as_rgb(f).quantize(palette=master, dither=Image.Dither.NONE)
           for f in frames]
    seq[0].save(out, save_all=True, append_images=seq[1:],
                duration=max(20, int(round(1000 / max(1, fps)))),
                loop=loop, disposal=2, optimize=True)
    return out


def render_frames(draw: Callable[[plt.Figure, int], None], n_frames: int,
                  frame_dir, *, figsize=(6.0, 6.0), dpi: int = 110,
                  prefix: str = "frame", progress: bool = True) -> list[Path]:
    """Render n_frames PNGs by calling draw(fig, k) for each, one figure per frame.

    Frames land on disk rather than accumulating in memory -- a few hundred frames
    of RGB at this size is gigabytes held live, and the PNGs are wanted anyway for
    re-cutting a GIF at another frame rate without re-rendering.
    """
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(n_frames):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor("black")
        draw(fig, k)
        p = frame_dir / f"{prefix}_{k:05d}.png"
        fig.savefig(p, facecolor="black", dpi=dpi)
        plt.close(fig)
        paths.append(p)
        if progress and (k + 1) % 10 == 0:
            print(f"  frame {k + 1}/{n_frames}", flush=True)
    return paths


def animate(draw, n_frames: int, out, *, frame_dir=None, fps: int = 18,
            figsize=(6.0, 6.0), dpi: int = 110, keep_frames: bool = True,
            progress: bool = True) -> Path:
    """render_frames + write_gif. frame_dir defaults to <out>_frames/."""
    out = Path(out)
    frame_dir = Path(frame_dir) if frame_dir else out.with_suffix("").parent / (
        out.with_suffix("").name + "_frames")
    paths = render_frames(draw, n_frames, frame_dir, figsize=figsize, dpi=dpi,
                          progress=progress)
    gif = write_gif(paths, out, fps=fps)
    if not keep_frames:
        for p in paths:
            p.unlink()
        try:                       # take the directory too, if we emptied it
            frame_dir.rmdir()
        except OSError:
            pass                   # something else is in there; leave it alone
    print(f"wrote {gif}  ({n_frames} frames @ {fps} fps)")
    return gif


# ------------------------------------------------- Faddeev-Skyrme convenience
def faddeev_energy_density(n1, n2, n3, dx: float, c4: float) -> np.ndarray:
    """Faddeev-Skyrme energy density on the lattice, in numpy.

    e2 from forward differences; e4 from the lattice field strength as a solid
    angle per plaquette, which stays finite where a naive curl of the
    stereographic projection does not. A numpy copy lives here on purpose: the
    renderer must not import jax (see module docstring).
    """
    def fwd(f, axis):
        return (np.roll(f, -1, axis) - f) / dx

    def solid_angle(a, b, c):
        cx = b[1] * c[2] - b[2] * c[1]
        cy = b[2] * c[0] - b[0] * c[2]
        cz = b[0] * c[1] - b[1] * c[0]
        num = a[0] * cx + a[1] * cy + a[2] * cz
        den = (1.0 + a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
               + b[0] * c[0] + b[1] * c[1] + b[2] * c[2]
               + c[0] * a[0] + c[1] * a[1] + c[2] * a[2])
        return 2.0 * np.arctan2(num, den)

    n = [np.asarray(x, float) for x in (n1, n2, n3)]
    e2 = np.zeros_like(n[0])
    for c in n:
        for ax in (0, 1, 2):
            e2 += fwd(c, ax) ** 2
    e4 = np.zeros_like(n[0])
    for i, j in ((0, 1), (1, 2), (0, 2)):
        B = [np.roll(c, -1, i) for c in n]
        D = [np.roll(c, -1, j) for c in n]
        C = [np.roll(c, -1, j) for c in B]
        e4 += ((solid_angle(n, B, C) + solid_angle(n, C, D)) / dx ** 2) ** 2
    return e2 + c4 * e4


def load_n_field(path, *, L: float | None = None, c4: float = 6.0) -> dict:
    """Read an n-field npz. Accepts both layouts in use: separate n1/n2/n3 arrays,
    or the (3, N, N, N) stack that the real-time leg saves per checkpoint.
    """
    d = np.load(path)
    if "n" in d:
        n = np.asarray(d["n"], float)
        n1, n2, n3 = n[0], n[1], n[2]
    else:
        n1, n2, n3 = (np.asarray(d[k], float) for k in ("n1", "n2", "n3"))
    N = n1.shape[0]
    box = float(d["L"]) if "L" in d else (float(L) if L is not None else float(N))
    return dict(n1=n1, n2=n2, n3=n3, N=N, L=box, dx=box / N,
                c4=float(d["c4"]) if "c4" in d else c4,
                t=float(d["t"]) if "t" in d else float("nan"),
                Q_H=float(d["Q_H"]) if "Q_H" in d else float("nan"),
                det=int(d["det"]) if "det" in d else None)


KNOT_BY_DET = {1: "unknot", 3: "trefoil", 5: "cinquefoil", 7: "septafoil"}


def _cells_scene(F, level_of, *, sigma, cmap, max_cells):
    """The n-field's energy density as individual grid cells (see cell_parts)."""
    from scipy.ndimage import label

    e = faddeev_energy_density(F["n1"], F["n2"], F["n3"], F["dx"], F["c4"])
    if sigma > 0:
        e = smooth_periodic(e, sigma)
    level = float(level_of(e))
    part = cell_parts(e, level, F["dx"], cmap=cmap, max_cells=max_cells)
    return part, dict(level=level, ncomp=int(label(e > level)[1]))


def _scene_parts(F, level_of, *, sigma, step_size, cmap):
    """Energy-density isosurface of an n-field, phase-coloured.

    Returns (part, info) where info carries the resolved level and the number of
    connected components of the thresholded region.

    level_of is a callable on the scalar field, not a number, so a quantile rule
    sees exactly the array marching cubes will see. Smoothing therefore happens
    HERE and iso_parts is called with sigma=0 -- smoothing afterwards would move
    the surface off the level the quantile was computed for.
    """
    from scipy.ndimage import label

    e = faddeev_energy_density(F["n1"], F["n2"], F["n3"], F["dx"], F["c4"])
    if sigma > 0:
        e = smooth_periodic(e, sigma)
    level = float(level_of(e))
    # Component count of the thresholded region: a clean single tube gives 1.
    # This is a MESH diagnostic, not a topological measurement -- a jump can mean
    # the render is shattering on lattice noise or that the object really did
    # reconnect, and only the knot ID in the run's summary can tell you which.
    ncomp = int(label(e > level)[1])
    part = iso_parts(e, level, F["dx"], (1.0, 1.0, 1.0, 1.0),
                     sigma=0.0, step_size=step_size)
    info = dict(level=level, ncomp=ncomp)
    if part is None:
        return None, info
    faces, _ = part
    return (faces, phase_facecolors(faces, F["n1"], F["n2"], F["dx"],
                                    cmap=cmap)), info


MODES = ("surface", "facets", "cells")
_MODE_CMAP = {"surface": "twilight", "facets": "twilight", "cells": "inferno"}


def _make_scene(mode: str, *, sigma, step_size, cmap, max_cells):
    """(scene_fn, sigma, tag) for a render mode. scene_fn(F, level_of) -> (part, info).

    "facets" forces sigma=0 rather than merely defaulting it. That is the upstream
    "no cheat" stance: the marching-cubes triangles are left at lattice scale so
    the picture shows the resolution it actually has, instead of implying a smooth
    object. Smoothing it would defeat the point of asking for the mode, so the
    mode overrides sigma instead of letting a stale flag quietly win.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; pick from {MODES}")
    cmap = cmap or _MODE_CMAP[mode]
    if mode == "cells":
        return (lambda F, lv: _cells_scene(F, lv, sigma=sigma, cmap=cmap,
                                           max_cells=max_cells)), sigma, "cells"
    sig = 0.0 if mode == "facets" else sigma
    tag = "iso facets (no cheat)" if mode == "facets" else "iso"
    return (lambda F, lv: _scene_parts(F, lv, sigma=sig, step_size=step_size,
                                       cmap=cmap)), sig, tag


def _label(F, extra=""):
    det = F.get("det")
    name = KNOT_BY_DET.get(det, f"det-{det}" if det is not None else "core")
    q = F.get("Q_H", float("nan"))
    head = f"{name}   Q_H={q:+.3f}" if q == q else name
    return head + extra


def turntable(field_path, out, *, volume_frac=0.004, level_frac=None, level=None,
              frames=36, fps=18, elev=22.0, sigma=1.0, step_size=1,
              cmap=None, mode="surface", max_cells=20000, zoom=1.4,
              figsize=(6.0, 6.0), dpi=110, L=None, c4=6.0,
              keep_frames=True) -> Path:
    """Sweep the camera 360 degrees around ONE static field.

    Geometry is built once and only the camera moves, so this is cheap and the
    object provably does not change between frames.

    volume_frac is the default rather than level_frac even though there is only
    one field here, because a fraction of the MAXIMUM is only as good as the
    maximum. The upstream turntable used level_frac=0.80, and on a real relaxed
    trefoil that yielded a 168-face scatter of specks: a handful of very hot cells
    left over from the seed set the maximum, so 0.80 of it lands far out in the
    tail. The same field at volume_frac=0.004 gave 5268 faces and a clean tube.
    """
    F = load_n_field(field_path, L=L, c4=c4)
    if level is not None:
        level_of, how = (lambda e: float(level)), f"level={level:.4g}"
    elif level_frac is not None:
        level_of = lambda e: iso_level(e, level_frac)
        how = f"{level_frac:.2f}*max"
    else:
        level_of = lambda e: volume_level(e, volume_frac)
        how = f"hottest {volume_frac:.3%}"
    scene, sig, tag = _make_scene(mode, sigma=sigma, step_size=step_size,
                                  cmap=cmap, max_cells=max_cells)
    part, info = scene(F, level_of)
    if part is None:
        raise SystemExit(f"nothing to draw at {how} in mode {mode}")
    center, half = bbox_of([part])
    azims = np.linspace(0.0, 360.0, frames, endpoint=False)
    title = _label(F, f"\n{tag} {how} — turntable")
    print(f"{Path(field_path).name}: N={F['N']} L={F['L']:g} mode={mode} "
          f"sigma={sig:g}, {len(part[0])} faces, {info['ncomp']} component(s), "
          f"{frames} frames")

    def draw(fig, k):
        ax = fig.add_subplot(111, projection="3d")
        draw_scene(ax, [part], center, half, elev=elev, azim=azims[k],
                   title=title, zoom=zoom)
        fig.tight_layout(pad=0.2)

    return animate(draw, frames, out, fps=fps, figsize=figsize, dpi=dpi,
                   keep_frames=keep_frames)


def timelapse(field_paths: Sequence, out, *, volume_frac=0.004, level=None,
              level_frac=None, level_samples=5, fps=12, elev=22.0, azim=-58.0,
              spin=0.0, sigma=1.0, step_size=1, cmap=None, mode="surface",
              max_cells=20000, zoom=1.4, figsize=(6.0, 6.0), dpi=110, L=None,
              c4=6.0, keep_frames=True) -> Path:
    """Animate a sequence of saved fields in time -- one frame per field.

    Camera framing is resolved once and held, so the object moves and the camera
    does not. The isosurface level has three modes, because no single rule is
    right for every question:

      volume_frac (default)  per-frame quantile: the surface around the hottest
                             volume_frac of the box. Tube thickness stays roughly
                             constant, so TOPOLOGY stays readable for the whole
                             run. Hides amplitude decay -- read that off the
                             energy series instead.
      level=X                one fixed absolute level. Honest about amplitude, but
                             the surface evaporates if the peak decays much.
      level_frac=f           fixed absolute, chosen by scan_level as f * the
                             LOWEST sampled max, so it stays inside the data in
                             every frame. A compromise; still thins a lot.

    See volume_level and scan_level for the measurements behind those claims.
    spin adds a slow azimuth drift, which helps read 3D structure out of a 2D
    loop without changing the object.

    sigma defaults to 1.0 CELL rather than 0. A single level through a spreading
    core catches only its hottest segments, so unsmoothed the tube renders as a
    string of disconnected blobs -- counting connected components of the
    thresholded region across a real-time trefoil, sigma=0 went from 7 to 116
    components over the run, while sigma=1 holds at 1. The smoothing is periodic
    (see smooth_periodic); with scipy's default reflecting boundary no amount of tuning
    kept every frame in one piece.

    The component count is reported per frame. Treat a jump as "look at this
    frame", not as evidence of reconnection: only the knot ID in the run's
    summary distinguishes a real reconnection from a mesh that fell apart.
    """
    paths = [Path(p) for p in field_paths]
    if not paths:
        raise SystemExit("no fields to animate")

    if level is not None:
        level_of = lambda e: float(level)
        level_mode = f"fixed level={level:.4g}"
        lvl_tag = f"@{level:.3g}"
    elif level_frac is not None:
        lv, rep = scan_level(paths, level_frac, samples=level_samples,
                             L=L, c4=c4, sigma=sigma)
        print(f"level scan over frames {rep['sampled']}: max(e) "
              f"{rep['min_max']:.4g}..{rep['max_max']:.4g} "
              f"(x{rep['decay']:.2f} decay)")
        if rep["decay"] > 1.5:
            print(f"  peak decayed x{rep['decay']:.2f}: a fixed level thins hard "
                  f"across this run -- consider the volume_frac default instead")
        level_of = lambda e: lv
        level_mode = f"fixed level={lv:.4g} ({level_frac:.2f} * lowest sampled max)"
        lvl_tag = f"@{lv:.3g} (fixed)"
    else:
        level_of = lambda e: volume_level(e, volume_frac)
        level_mode = f"per-frame level enclosing the hottest {volume_frac:.3%} of cells"
        lvl_tag = f"hottest {volume_frac:.2%}"
    print(f"level mode: {level_mode}")

    scene, sig, mode_tag = _make_scene(mode, sigma=sigma, step_size=step_size,
                                       cmap=cmap, max_cells=max_cells)
    F0 = load_n_field(paths[0], L=L, c4=c4)
    p0, i0 = scene(F0, level_of)
    if p0 is None:
        raise SystemExit(f"nothing to draw in frame 0 ({level_mode}, mode={mode})")
    # Framing spans the FIRST and LAST meshes, not just the first. A spreading core
    # grows outward, so a box fitted to frame 0 alone can clip the end of the run,
    # and a clipped surface shows a flat cut face that is easy to misread as
    # structure.
    ends = [p0]
    if len(paths) > 1:
        pN, _ = scene(load_n_field(paths[-1], L=L, c4=c4), level_of)
        if pN is not None:
            ends.append(pN)
    center, half = bbox_of(ends)
    print(f"{len(paths)} fields, N={F0['N']} L={F0['L']:g}, mode={mode} "
          f"sigma={sig:g}; framing spans frames 0 and {len(paths) - 1}, held fixed")

    # One frame's field and mesh at a time: an N=320 n-field is ~100 MB and the
    # sequence can be hundreds of frames, so nothing is retained across frames.
    cache: dict[int, tuple] = {0: (p0, i0, F0)}
    seen = {}

    def draw(fig, k):
        if k not in cache:
            Fk = load_n_field(paths[k], L=L, c4=c4)
            pk, ik = scene(Fk, level_of)
            cache.clear()
            cache[k] = (pk, ik, Fk)
        part, info, Fk = cache[k]
        seen[k] = info["ncomp"]
        if part is None:
            print(f"  frame {k}: no isosurface — blank")
        elif info["ncomp"] > 1:
            print(f"  frame {k}: mesh in {info['ncomp']} components "
                  f"(level={info['level']:.4g}) — inspect before reading it as "
                  f"reconnection")
        ax = fig.add_subplot(111, projection="3d")
        t = Fk.get("t", float("nan"))
        title = _label(Fk, f"\nt={t:.3f}" if t == t else "") + \
            f"   {mode_tag} {lvl_tag}   frame {k + 1}/{len(paths)}"
        draw_scene(ax, [part], center, half, elev=elev,
                   azim=azim + spin * k, title=title, zoom=zoom)
        fig.tight_layout(pad=0.2)

    gif = animate(draw, len(paths), out, fps=fps, figsize=figsize, dpi=dpi,
                  keep_frames=keep_frames)
    comps = [seen[k] for k in sorted(seen)]
    if comps and max(comps) > 1:
        print(f"mesh components per frame: {comps}\n  a clean single tube is all "
              f"1s; anything else is worth looking at in the frames")
    return gif


# ----------------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", required=True)
        p.add_argument("--fps", type=int, default=None)
        p.add_argument("--elev", type=float, default=22.0)
        p.add_argument("--sigma", type=float, default=1.0,
                       help="pre-smooth the scalar (in CELLS) before marching "
                            "cubes; 0 shatters a spreading core into blobs")
        p.add_argument("--step", type=int, default=1,
                       help="marching-cubes step_size; >1 coarsens the mesh AND "
                            "degrades the face-centroid depth sort")
        p.add_argument("--mode", default="surface", choices=MODES,
                       help="surface = smoothed isosurface; facets = unsmoothed, "
                            "grid facets left in (no cheat); cells = one cube per "
                            "grid cell above the level")
        p.add_argument("--max-cells", type=int, default=20000,
                       help="cells mode: cap on cubes drawn (6 quads each)")
        p.add_argument("--cmap", default=None,
                       help="default depends on --mode: twilight for surfaces "
                            "(cyclic phase), inferno for cells (magnitude)")
        p.add_argument("--zoom", type=float, default=1.4,
                       help="enlarge the object inside the axes; mplot3d "
                            "reserves padding that set_axis_off then hides")
        p.add_argument("--dpi", type=int, default=110)
        p.add_argument("--size", type=float, default=6.0)
        p.add_argument("--L", type=float, default=None)
        p.add_argument("--c4", type=float, default=6.0)
        p.add_argument("--drop-frames", action="store_true",
                       help="delete the PNG frames after writing the GIF")

    t = sub.add_parser("turntable", help="camera sweep around one static field")
    t.add_argument("--field", required=True)
    t.add_argument("--frames", type=int, default=36)
    tlv = t.add_mutually_exclusive_group()
    tlv.add_argument("--volume-frac", type=float, default=0.004,
                     help="level enclosing the hottest fraction of the box")
    tlv.add_argument("--level-frac", type=float, default=None,
                     help="level as a fraction of this field's max; fragile when "
                          "a few hot cells set the max (see turntable's docstring)")
    tlv.add_argument("--level", type=float, default=None,
                     help="absolute isosurface level")
    common(t)

    m = sub.add_parser("timelapse", help="one frame per saved field, in time")
    m.add_argument("--fields", required=True,
                   help="directory of n_*.npz (or a glob)")
    m.add_argument("--glob", default="n_*.npz")
    m.add_argument("--stride", type=int, default=1)
    m.add_argument("--azim", type=float, default=-58.0)
    m.add_argument("--spin", type=float, default=0.0,
                   help="degrees of azimuth drift per frame")
    # Three level modes, mutually exclusive; default is --volume-frac (see
    # timelapse's docstring for why that is the default for a sequence).
    lv = m.add_mutually_exclusive_group()
    lv.add_argument("--volume-frac", type=float, default=0.004,
                    help="per-frame level enclosing the hottest fraction of the "
                         "box; keeps topology readable as the core spreads")
    lv.add_argument("--level", type=float, default=None,
                    help="one fixed absolute level (honest about amplitude, but "
                         "the surface can evaporate)")
    lv.add_argument("--level-frac-fixed", type=float, default=None,
                    dest="level_frac_fixed",
                    help="fixed level at this fraction of the LOWEST sampled max")
    m.add_argument("--level-samples", type=int, default=5,
                   help="frames sampled by --level-frac-fixed")
    common(m)

    a = ap.parse_args(argv)
    kw = dict(elev=a.elev, sigma=a.sigma, step_size=a.step, cmap=a.cmap,
              mode=a.mode, max_cells=a.max_cells, zoom=a.zoom, dpi=a.dpi,
              figsize=(a.size, a.size), L=a.L, c4=a.c4,
              keep_frames=not a.drop_frames)
    if a.cmd == "turntable":
        return turntable(a.field, a.out, frames=a.frames, fps=a.fps or 18,
                         level_frac=a.level_frac, volume_frac=a.volume_frac,
                         level=a.level, **kw)
    src = Path(a.fields)
    paths = sorted(src.glob(a.glob)) if src.is_dir() else sorted(
        Path().glob(a.fields))
    if not paths:
        raise SystemExit(f"no fields matched {a.fields}/{a.glob}")
    return timelapse(paths[::a.stride], a.out, azim=a.azim, spin=a.spin,
                     volume_frac=a.volume_frac, level=a.level,
                     level_frac=a.level_frac_fixed,
                     level_samples=a.level_samples, fps=a.fps or 12, **kw)


if __name__ == "__main__":
    main()
