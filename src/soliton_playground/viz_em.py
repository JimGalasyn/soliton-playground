"""Portraits of the GAUGED two-scalar field: raw isosurfaces, the RMF twist tube,
and E/B field lines.

Factored out of null-worldtube-private's simulations/engine_dogfood/render_portrait.py
together with the EM helpers from simulations/nwt_em_fields.py and the phase-winding
vortex tracer from simulations/gpe_vortex_topology.py.

This is a SEPARATE module from viz because it applies to a different object. viz
renders a bare Faddeev n-field -- one unit 3-vector field per site, what the
real-time leg evolves. Everything here needs the `ehn-two-scalar` layout: two
complex scalars, a gauge potential, and a scalar potential. The real-time runs do
NOT have those fields, so none of this applies to them; what it does apply to is
the ten catalog entries in ehn_lab (`model: ehn-two-scalar`), whose loader is the
`render_portrait.load_field` that field_store.py's comments have been pointing at
across a repo boundary.

WHAT THE UPSTREAM GOT RIGHT AND THIS KEEPS
------------------------------------------
  * Ribbons, not lines. Field lines are widened into camera-facing quads and
    merged into the same collection as the surfaces, because mplot3d draws whole
    artists back to front: a Line3DCollection of field lines cannot z-interleave
    with an isosurface, so the lines would land wholly in front of or behind the
    object they are meant to thread. See viz.add_parts.
  * No smoothing on the raw view. `view_raw` leaves the marching-cubes facets at
    lattice scale -- the upstream caption called it "grid facets left in (no
    cheat)", and `sigma` is never passed above 0 anywhere in the original file.
    That stance is kept as the default here and is what `facets=True` means.

ONE CONSEQUENCE WORTH KNOWING
-----------------------------
Ribbon geometry depends on the camera, since the quads are widened perpendicular
to the view direction. So unlike a pure isosurface scene, a field-line scene
CANNOT be built once and spun -- move the camera and the ribbons must be rebuilt.
The animation helpers here rebuild per frame for that reason.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import viz  # noqa: E402

C_PHI1 = (0.82, 0.13, 0.55, 1.0)      # magenta: the knotted, gauged scalar
C_PHI2 = (0.10, 0.74, 0.80, 1.0)      # cyan: the ring
C_E = (1.00, 0.42, 0.24, 1.0)
C_B = (0.22, 0.72, 1.00, 1.0)


# ------------------------------------------------------------------- loading
def load_gauged_field(fdir) -> dict:
    """Read an ehn-two-scalar field directory (field.npz + manifest.json).

    Layout, matching the upstream loader: u[0..1] are Re/Im of phi1, u[2..3] of
    phi2, u[4..6] are the gauge potential A. Files carry MORE than seven
    components (the observed battery fields have ten); the extras are not
    described by the manifest and are deliberately not guessed at here.
    """
    fdir = Path(fdir)
    d = np.load(fdir / "field.npz")
    u = d["u"]
    if u.shape[0] < 7:
        raise SystemExit(f"{fdir}: u has {u.shape[0]} components, need >= 7")
    meta = json.loads((fdir / "manifest.json").read_text())
    N = u.shape[1]
    L = float(meta["params"]["L"])
    return dict(p1=u[0] + 1j * u[1], p2=u[2] + 1j * u[3],
                A=[np.asarray(u[4]), np.asarray(u[5]), np.asarray(u[6])],
                s=np.asarray(d["s"]) if "s" in d.files else None,
                N=N, L=L, dx=L / N, n_components=int(u.shape[0]), meta=meta)


def grid1d(N: int, L: float) -> np.ndarray:
    return np.linspace(-L / 2, L / 2, N, endpoint=False)


def _phys(P, N: int, dx: float) -> np.ndarray:
    """Cell-index coords -> physical coords, box centred on zero. Matches the
    convention viz.iso_parts uses, so meshes from both compose."""
    return (np.asarray(P, float) - N / 2.0) * dx


# ------------------------------------------------------- core curve (tracing)
def _wrap(d):
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def _winding(theta, a, b):
    """Signed phase winding around each (a, b) plaquette, in units of 2*pi.

    Summing four wrapped phase differences around a face is the sound-immune way
    to find a vortex core: it is exactly zero unless the face is pierced, so
    density ripples and phonons cannot fake one.
    """
    d1 = _wrap(np.roll(theta, -1, a) - theta)
    d2 = _wrap(np.roll(np.roll(theta, -1, a), -1, b) - np.roll(theta, -1, a))
    d3 = _wrap(np.roll(theta, -1, b) - np.roll(np.roll(theta, -1, a), -1, b))
    d4 = _wrap(theta - np.roll(theta, -1, b))
    return np.rint((d1 + d2 + d3 + d4) / (2.0 * np.pi)).astype(np.int8)


# plaquette plane (a, b) -> (normal axis, cell-centre offset, unit tangent)
_PLAQ = [((1, 2), 0), ((2, 0), 1), ((0, 1), 2)]


def vortex_skeleton(psi):
    """Directed segments on the |psi|=0 core lines, from plaquette phase winding.

    Returns (P, T, C): sub-cell positions in index units, signed unit tangents,
    and integer cell indices.
    """
    theta = np.angle(np.asarray(psi))
    P, T, C = [], [], []
    for (a, b), ax in _PLAQ:
        w = _winding(theta, a, b)
        hit = np.argwhere(w != 0)
        if not len(hit):
            continue
        sgn = w[hit[:, 0], hit[:, 1], hit[:, 2]].astype(float)
        off = np.zeros((len(hit), 3))
        off[:, a] += 0.5
        off[:, b] += 0.5
        tan = np.zeros((len(hit), 3))
        tan[:, ax] = sgn
        P.append(hit.astype(float) + off)
        T.append(tan)
        C.append(hit)
    if not P:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3), int)
    return np.concatenate(P), np.concatenate(T), np.concatenate(C)


def _label_lines(cells, shape, min_seg):
    """Connected-component label over the occupied cells; returns (labels, ids of
    components at least min_seg long, sizes)."""
    import scipy.ndimage as ndi

    vol = np.zeros(shape, np.uint8)
    vol[cells[:, 0], cells[:, 1], cells[:, 2]] = 1
    lab, _ = ndi.label(vol, structure=np.ones((3, 3, 3)))
    at = lab[cells[:, 0], cells[:, 1], cells[:, 2]]
    ids, counts = np.unique(at, return_counts=True)
    order = np.argsort(-counts)
    big = [int(i) for i, c in zip(ids[order], counts[order]) if c >= min_seg]
    return at, big, dict(zip(ids.tolist(), counts.tolist()))


def _order_line(P, T, max_gap=2.2):
    """Greedy nearest-neighbour walk along a segment cloud, seeded at one end and
    following the tangent, so the result is a polyline rather than a point set."""
    n = len(P)
    if n < 3:
        return np.arange(n)
    used = np.zeros(n, bool)
    order = [0]
    used[0] = True
    for _ in range(n - 1):
        cur = order[-1]
        d = np.linalg.norm(P - P[cur], axis=1)
        d[used] = np.inf
        ahead = (P - P[cur]) @ T[cur]
        d = np.where(ahead >= 0, d, d + 0.75)      # prefer forward along tangent
        nxt = int(np.argmin(d))
        if not np.isfinite(d[nxt]) or d[nxt] > max_gap:
            break
        order.append(nxt)
        used[nxt] = True
    return np.asarray(order)


def _smooth_closed(curve, npts=320, smoothing=0.5):
    """Periodic cubic-spline resample, so a swept tube and its RMF are not
    grid-stair-stepped. Falls back to the raw polyline if the fit fails."""
    from scipy.interpolate import splev, splprep

    c = np.asarray(curve, float)
    keep = np.r_[True, np.abs(np.diff(c, axis=0)).sum(1) > 1e-6]
    c = c[keep]
    if len(c) < 8:
        return np.asarray(curve, float)
    try:
        tck, _ = splprep([c[:, 0], c[:, 1], c[:, 2]],
                         s=len(c) * smoothing, per=1, k=3)
        uu = np.linspace(0, 1, npts, endpoint=False)
        return np.stack(splev(uu, tck), axis=1)
    except Exception:
        return c


def core_curve(psi, N: int, dx: float, min_seg=20, smooth=True):
    """The largest |psi|=0 vortex core as an ordered polyline in physical coords."""
    P, T, C = vortex_skeleton(psi)
    if not len(P):
        return None
    at, big, _ = _label_lines(C, np.asarray(psi).shape, min_seg)
    if not big:
        return None
    m = at == big[0]
    curve = _phys(P[m][_order_line(P[m], T[m])], N, dx)
    return _smooth_closed(curve) if smooth else curve


# ------------------------------------------------------------ EM from the field
def _k_grids(N: int, dx: float):
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    K2[0, 0, 0] = 1.0                      # the k=0 mode is zeroed, not divided
    return KX, KY, KZ, K2


def curl(F, dx: float):
    """Spectral curl. Spectral rather than finite-difference because the box is
    periodic and B = curl A must come out divergence-free to machine precision;
    a stencil leaves a residual that seeds spurious field-line sources."""
    KX, KY, KZ, _ = _k_grids(F[0].shape[0], dx)
    Fx, Fy, Fz = (np.fft.fftn(c) for c in F)
    return [np.real(np.fft.ifftn(1j * (KY * Fz - KZ * Fy))),
            np.real(np.fft.ifftn(1j * (KZ * Fx - KX * Fz))),
            np.real(np.fft.ifftn(1j * (KX * Fy - KY * Fx)))]


def electric_field(rho, dx: float, eps0: float = 1.0):
    """E from a charge density by solving the periodic Poisson equation."""
    KX, KY, KZ, K2 = _k_grids(np.asarray(rho).shape[0], dx)
    rk = np.fft.fftn(rho)
    rk[0, 0, 0] = 0.0
    phi = rk / (eps0 * K2)
    return [np.real(np.fft.ifftn(-1j * K * phi)) for K in (KX, KY, KZ)]


def magnetic_field(j, dx: float, mu0: float = 1.0):
    """B from a current density via the periodic vector Poisson equation, B = curl A.

    Periodic, like electric_field: the k=0 mode is dropped rather than solved. For
    the AC (cos/sin) components of a modulated source that costs nothing, since
    those integrate to zero by construction; it is only a net DC monopole that a
    periodic box cannot represent. The upstream used an open-BC solver from
    nwt_substrate for exactly that reason and it is not available here.
    """
    KX, KY, KZ, K2 = _k_grids(np.asarray(j[0]).shape[0], dx)
    Ak = []
    for jc in j:
        jk = np.fft.fftn(jc)
        jk[0, 0, 0] = 0.0
        Ak.append(jk * mu0 / K2)
    Ax, Ay, Az = Ak
    return [np.real(np.fft.ifftn(1j * (KY * Az - KZ * Ay))),
            np.real(np.fft.ifftn(1j * (KZ * Ax - KX * Az))),
            np.real(np.fft.ifftn(1j * (KX * Ay - KY * Ax)))]


def deposit_modulated_sources(curve, XYZ, m: int, width: float | None = None,
                              dx: float | None = None, trunc: float = 4.0):
    """Deposit DC, cos- and sin-weighted current and charge along a carrier curve.

    This is the trick that makes a travelling-field animation affordable. A
    travelling surface-current wave cos(2*pi*m*s + phi) is a linear combination of
    a cos-weighted and a sin-weighted source, and Maxwell is linear, so the three
    field problems can be solved ONCE and recombined per frame:

        F(phi) = F_dc + cos(phi) * F_cos - sin(phi) * F_sin

    Re-solving Poisson every frame instead would cost n_frames times as many FFT
    solves for exactly the same pictures.

    width defaults to 2*dx, in CELLS rather than a fixed physical length. The
    upstream's hard-coded 0.6 was two cells on ITS grid (dx=0.3); reused on a
    coarser grid it becomes a sub-cell spike that the lattice cannot represent, so
    the deposit aliases instead of smoothing.

    Each Gaussian is deposited into a local window of radius trunc*width rather
    than over the whole box. Beyond 4 sigma the contribution is below 3e-4 of the
    peak, and the full-grid version costs len(curve) * N^3 exponentials -- 283
    million for a 320-point core on a 96^3 grid.

    Returns dict with Jdc/Jc/Js (vector) and Rdc/Rc/Rs (scalar).
    """
    X, Y, Z = XYZ
    shape = X.shape
    if dx is None:
        dx = float(abs(X[1, 0, 0] - X[0, 0, 0])) if shape[0] > 1 else 1.0
    if width is None:
        width = 2.0 * dx
    C = np.asarray(curve, float)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    lo = np.array([X[0, 0, 0], Y[0, 0, 0], Z[0, 0, 0]], float)
    rad = max(1, int(np.ceil(trunc * width / dx)))

    out = {k: ([np.zeros(shape), np.zeros(shape), np.zeros(shape)]
               if k.startswith("J") else np.zeros(shape))
           for k in ("Jdc", "Jc", "Js", "Rdc", "Rc", "Rs")}
    M = len(C)
    two_w2 = 2.0 * width ** 2
    for i in range(M):
        p, t = C[i], T[i]
        c0 = np.rint((p - lo) / dx).astype(int)
        sl, sub = [], []
        for d in range(3):
            a = max(0, c0[d] - rad)
            b = min(shape[d], c0[d] + rad + 1)
            if a >= b:
                break
            sl.append(slice(a, b))
            sub.append(lo[d] + np.arange(a, b) * dx)
        if len(sl) < 3:
            continue
        sx, sy, sz = np.meshgrid(*sub, indexing="ij")
        g = np.exp(-((sx - p[0]) ** 2 + (sy - p[1]) ** 2
                     + (sz - p[2]) ** 2) / two_w2)
        cph = np.cos(2 * np.pi * m * i / M)
        sph = np.sin(2 * np.pi * m * i / M)
        w = tuple(sl)
        out["Rdc"][w] += g
        out["Rc"][w] += g * cph
        out["Rs"][w] += g * sph
        for d in range(3):
            out["Jdc"][d][w] += g * t[d]
            out["Jc"][d][w] += g * cph * t[d]
            out["Js"][d][w] += g * sph * t[d]
    return out


def trace_field_lines(field, g, seeds, *, n_steps=400, ds=0.15, both_ways=True,
                      min_mag=0.0):
    """RK2 field-line integration, all seeds advanced together.

    Vectorised over seeds rather than looping one at a time as the upstream did:
    the interpolator call is the entire cost, and one call on M points is far
    cheaper than M calls on one point. Lines still terminate individually, on a
    vanishing field or on leaving the box.

    min_mag is a magnitude FLOOR, and it matters more than it sounds. The tracer
    follows the normalised direction, so once a line drifts into a region where
    the field is weak its direction is set by whatever noise is left and it
    wanders -- on a real gauged trefoil, |grad s| averaged 25x smaller than its
    peak, and the E lines wandered into a boxy cage that looked like structure
    and was not. Stopping a line when the field falls below min_mag keeps each
    line inside the region where its direction actually means something.
    """
    from scipy.interpolate import RegularGridInterpolator

    interp = [RegularGridInterpolator((g, g, g), np.asarray(F),
                                      bounds_error=False, fill_value=0.0)
              for F in field]
    lo, hi = float(g[0]), float(g[-1])

    def sample(P):
        v = np.stack([f(P) for f in interp], axis=-1)
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v, n

    lines = []
    seeds = np.asarray(seeds, float)
    if not len(seeds):
        return lines
    for direction in ((1.0, -1.0) if both_ways else (1.0,)):
        P = seeds.copy()
        alive = np.ones(len(P), bool)
        traj = [P.copy()]
        length = np.zeros(len(P), int)
        floor = max(float(min_mag), 1e-9)
        for _ in range(n_steps):
            v, n = sample(P)
            ok = n[..., 0] > floor
            vh = np.where(ok[:, None], direction * v / np.maximum(n, 1e-30), 0.0)
            vm, nm = sample(P + 0.5 * ds * vh)
            ok &= nm[..., 0] > floor
            step = np.where(ok[:, None],
                            ds * direction * vm / np.maximum(nm, 1e-30), 0.0)
            P = P + step
            inside = np.all((P > lo) & (P < hi), axis=1)
            alive &= ok & inside
            length += alive
            traj.append(P.copy())
            if not alive.any():
                break
        T = np.stack(traj, axis=0)                       # (nstep+1, nseed, 3)
        for i, ln in enumerate(length):
            if ln > 3:
                lines.append(T[:ln + 1, i, :])
    return lines


def view_dir(elev: float, azim: float) -> np.ndarray:
    el, az = np.radians(elev), np.radians(azim)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                     np.sin(el)])


def lines_to_ribbons(lines, view, width: float, color) -> viz.Part | None:
    """Polylines as camera-facing quads, so they merge into the scene's single
    collection and depth-sort against the surfaces (see the module docstring)."""
    quads = []
    for ln in lines:
        if len(ln) < 2:
            continue
        p0, p1 = ln[:-1], ln[1:]
        wd = np.cross(p1 - p0, view)
        n = np.linalg.norm(wd, axis=1, keepdims=True)
        wd = np.where(n > 1e-9, wd / np.maximum(n, 1e-30), 0.0) * width
        quads.append(np.stack([p0 - wd, p0 + wd, p1 + wd, p1 - wd], axis=1))
    if not quads:
        return None
    F = np.concatenate(quads, axis=0)
    return F, np.tile(np.asarray(color, float), (len(F), 1))


def poloidal_seeds(core, dx: float, roff=3.2, n_along=12, n_pol=1):
    """Seeds just outside the flux tube, in the local poloidal plane.

    B loops poloidally around the tube, so lines started here close into the ring
    loops of a flux tube instead of running off to the boundary.
    """
    if core is None or not len(core):
        return []
    _, M, B = viz.rmf_frame(core)
    idx = np.linspace(0, len(core) - 1, n_along, endpoint=False).astype(int)
    return [core[i] + roff * dx * (np.cos(2 * np.pi * k / n_pol) * M[i]
                                   + np.sin(2 * np.pi * k / n_pol) * B[i])
            for i in idx for k in range(n_pol)]


# ------------------------------------------------------------- the phase tube
def sample_phase(psi, N: int, L: float, pts):
    """arg(psi) at physical points, interpolating Re and Im separately so the
    2*pi branch cut is never averaged across."""
    from scipy.interpolate import RegularGridInterpolator

    g = grid1d(N, L)
    re = RegularGridInterpolator((g, g, g), np.real(psi), bounds_error=False,
                                 fill_value=0.0)
    im = RegularGridInterpolator((g, g, g), np.imag(psi), bounds_error=False,
                                 fill_value=0.0)
    return np.arctan2(im(pts), re(pts))


def phase_tube_parts(core, F, rad_cells=2.2, npol=30, alpha=1.0,
                     cmap="twilight_shifted") -> viz.Part | None:
    """The core swept as a closed RMF tube, painted by arg(phi1).

    Because the frame is rotation-minimising and its holonomy is closed out, any
    colour spiral on the tube is REAL framing twist rather than an artefact of the
    frame turning with the curve.
    """
    if core is None or len(core) < 4:
        return None
    X, Y, Z, _, _ = viz.tube_surface(core, rad_cells * F["dx"], npol)
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)
    p01 = (sample_phase(F["p1"], F["N"], F["L"], pts).reshape(X.shape)
           + np.pi) / (2 * np.pi)
    V = np.stack([X, Y, Z], -1)
    Vi1 = np.roll(V, -1, 0)
    Vj1 = np.roll(V, -1, 1)
    faces = np.stack([V, Vi1, np.roll(Vi1, -1, 1), Vj1],
                     axis=2).reshape(-1, 4, 3)
    return faces, viz.quad_phase_colors(p01, cmap, alpha)


# ------------------------------------------------------------------- the views
def view_raw(F, *, sigma=0.0, level=0.5):
    """Both scalars' isosurfaces, merged so a link weaves.

    sigma defaults to 0: this is the "no cheat" view and the grid facets are left
    in on purpose, so the picture shows the resolution it actually has.
    """
    parts = [viz.iso_parts(np.abs(F["p1"]), level, F["dx"], C_PHI1, sigma=sigma),
             viz.iso_parts(np.abs(F["p2"]), level, F["dx"], C_PHI2, sigma=sigma)]
    facet = " grid facets left in (no cheat)" if sigma == 0 else \
            f" smoothed sigma={sigma:g}"
    return [p for p in parts if p is not None], \
        f"raw |phi1| (magenta) + |phi2| (cyan), merged mesh.{facet}"


def view_cells(F, *, volume_frac=0.004, max_cells=20000, cmap="inferno"):
    """The individual grid cells of the knotted scalar's core, as cubes."""
    m = 1.0 - np.abs(F["p1"]) ** 2                 # core measure: 1 where phi1 -> 0
    level = viz.volume_level(m, volume_frac)
    part = viz.cell_parts(m, level, F["dx"], cmap=cmap, max_cells=max_cells)
    return ([part] if part is not None else []), \
        f"grid cells of the phi1 core (hottest {volume_frac:.2%}), one cube per cell"


def view_twist(F, *, npol=30, rad_cells=2.2):
    core = core_curve(F["p1"], F["N"], F["dx"])
    if core is None:
        return [], "twist: no phi1 core found"
    parts = [viz.iso_parts(np.abs(F["p2"]), 0.5, F["dx"],
                           C_PHI2[:3] + (0.32,), sigma=0.0),
             phase_tube_parts(core, F, rad_cells, npol)]
    return [p for p in parts if p is not None], \
        "twist: phi1 core tube painted by arg(phi1) on a closed RMF frame " \
        "-- colour spiral = real framing twist"


def _mag_floor(field, frac=0.15) -> float:
    """A magnitude floor for tracing, as a fraction of the field's 99th
    percentile. Percentile not maximum, so one hot cell cannot set the scale."""
    mag = np.sqrt(sum(np.asarray(c, float) ** 2 for c in field))
    return frac * float(np.percentile(mag, 99))


def field_line_parts(F, kind: str, core, center, half, elev, azim, *,
                     mag_frac=0.15):
    """Just the E or B ribbons, no carrier. Returns (part_or_None, n_lines, tag).

    E is -grad of the scalar potential; B is curl A. Both are lightly smoothed
    before tracing, because the tracer follows the field DIRECTION and unsmoothed
    lattice noise makes lines wander off the structure they belong to.
    """
    N, L, dx = F["N"], F["L"], F["dx"]
    g = grid1d(N, L)
    if kind == "E":
        if F.get("s") is None:
            return None, 0, "E: no scalar potential stored in this field"
        gs = np.gradient(viz.smooth_periodic(F["s"], 1.0), dx)
        fld = [-gs[0], -gs[1], -gs[2]]
        col, width = C_E, 0.008 * half
        th = np.linspace(0, 2 * np.pi, 18, endpoint=False)
        ph = np.linspace(0.28, np.pi - 0.28, 6)
        rs = half * 1.05
        seeds = [center + rs * np.array([np.sin(p) * np.cos(t),
                                         np.sin(p) * np.sin(t), np.cos(p)])
                 for t in th for p in ph]
        lines = trace_field_lines(fld, g, seeds,
                                  n_steps=max(20, int(2.6 * half / dx)),
                                  ds=0.7 * dx, both_ways=False,
                                  min_mag=_mag_floor(fld, mag_frac))
        tag = "E = -grad A0 (radial spokes)"
    else:
        fld = [viz.smooth_periodic(b, 1.0) for b in curl(F["A"], dx)]
        col, width = C_B, 0.010 * half
        seeds = poloidal_seeds(core, dx, roff=3.2, n_along=12)
        lines = trace_field_lines(fld, g, seeds, n_steps=500, ds=0.45 * dx,
                                  both_ways=True,
                                  min_mag=_mag_floor(fld, mag_frac))
        tag = "B = curl A (poloidal loops)"
    return (lines_to_ribbons(lines, view_dir(elev, azim), width, col),
            len(lines), tag)


def carrier_parts(F, core):
    """The core tube that the field lines thread, in cyclic hue."""
    p = phase_tube_parts(core, F, 2.2, 26, cmap="hsv")
    return [p] if p is not None else []


def view_field(F, kind: str, elev: float, azim: float, *, core=None):
    """Carrier tube plus its E or B field lines."""
    if core is None:
        core = core_curve(F["p1"], F["N"], F["dx"])
    if core is None:
        return [], f"{kind}field: no phi1 core found"
    center, half = viz.bbox_of_points(core)
    ribbons, n, tag = field_line_parts(F, kind, core, center, half, elev, azim)
    parts = carrier_parts(F, core) + ([ribbons] if ribbons is not None else [])
    return parts, f"{tag} -- {n} lines"


VIEWS = ("raw", "cells", "twist", "efield", "bfield")


def build_view(F, view: str, elev: float, azim: float, **kw):
    if view == "raw":
        return view_raw(F, sigma=kw.get("sigma", 0.0))
    if view == "cells":
        return view_cells(F, volume_frac=kw.get("volume_frac", 0.004),
                          max_cells=kw.get("max_cells", 20000))
    if view == "twist":
        return view_twist(F)
    if view in ("efield", "bfield"):
        return view_field(F, "E" if view == "efield" else "B", elev, azim)
    raise ValueError(f"unknown view {view!r}; pick from {VIEWS}")


# ----------------------------------------------------------------- the drivers
def _header(fdir, F):
    lk = F["meta"].get("cross_lk")
    nl = F["meta"].get("params", {}).get("nlink")
    return (f"{Path(fdir).name}   N={F['N']} L={F['L']:g}   "
            f"nlink={nl}  Lk(phi1,phi2)={lk}")


def _scene(ax, parts, center, half, elev, azim, title, zoom=1.3):
    viz.draw_scene(ax, parts, center, half, elev=elev, azim=azim, title=title,
                   zoom=zoom)


def _frame_box(F, parts):
    center, half = viz.bbox_of(parts)
    if half <= 0:
        center, half = np.zeros(3), F["L"] / 4
    return center, half


def portrait(fdir, view: str, out, *, elev=22.0, azim=-56.0, dpi=140,
             figsize=(7.5, 7.5), zoom=1.3, **kw) -> Path:
    """One still of one view."""
    F = load_gauged_field(fdir)
    parts, caption = build_view(F, view, elev, azim, **kw)
    if not parts:
        raise SystemExit(f"{view}: nothing to draw ({caption})")
    center, half = _frame_box(F, parts)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    _scene(ax, parts, center, half, elev, azim,
           f"{_header(fdir, F)}  --  {view}\n{caption}", zoom)
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, facecolor="black", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({view}: {caption})")
    return out


def triptych(fdir, out, *, elev=22.0, azim=-56.0, dpi=140, zoom=1.3) -> Path:
    """E | B | E+B on the same field, the composition the alpha portrait used."""
    F = load_gauged_field(fdir)
    core = core_curve(F["p1"], F["N"], F["dx"])
    if core is None:
        raise SystemExit("triptych: no phi1 core found to thread")
    center, half = viz.bbox_of_points(core)
    base = carrier_parts(F, core)
    e_rib, n_e, e_tag = field_line_parts(F, "E", core, center, half, elev, azim)
    b_rib, n_b, b_tag = field_line_parts(F, "B", core, center, half, elev, azim)
    e_only = [p for p in (e_rib,) if p is not None]
    b_only = [p for p in (b_rib,) if p is not None]
    panels = [(f"{e_tag} x{n_e}", base + e_only),
              (f"{b_tag} x{n_b}", base + b_only),
              ("composed  E + B", base + e_only + b_only)]
    center, half = _frame_box(F, panels[2][1])

    fig = plt.figure(figsize=(16.5, 6.2), dpi=dpi)
    fig.patch.set_facecolor("black")
    for i, (cap, parts) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        _scene(ax, parts, center, half, elev, azim, cap, zoom)
    fig.suptitle(_header(fdir, F) + "   --   E | B | composed",
                 color="#DDDDDD", fontsize=12, y=0.97)
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, facecolor="black", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def spin(fdir, view: str, out, *, frames=36, fps=18, elev=22.0, dpi=110,
         figsize=(6.0, 6.0), zoom=1.3, keep_frames=True, **kw) -> Path:
    """Turntable of any view.

    Note this rebuilds the scene EVERY frame, unlike viz.turntable which builds
    geometry once. It has to: ribbon quads are widened perpendicular to the view
    direction, so they are only correct for the camera they were built for.
    """
    F = load_gauged_field(fdir)
    azims = np.linspace(0.0, 360.0, frames, endpoint=False)
    parts0, caption = build_view(F, view, elev, float(azims[0]), **kw)
    center, half = _frame_box(F, parts0)
    needs_rebuild = view in ("efield", "bfield")
    how = ("rebuilt per frame (camera-dependent ribbons)" if needs_rebuild
           else "geometry built once")
    print(f"{Path(fdir).name}: {view}, {frames} frames, {how}")

    def draw(fig, k):
        az = float(azims[k])
        parts = build_view(F, view, elev, az, **kw)[0] if needs_rebuild else parts0
        ax = fig.add_subplot(111, projection="3d")
        _scene(ax, parts, center, half, elev, az,
               f"{_header(fdir, F)}  --  {view}\n{caption}", zoom)
        fig.tight_layout(pad=0.2)

    return viz.animate(draw, frames, out, fps=fps, figsize=figsize, dpi=dpi,
                       keep_frames=keep_frames)


CYCLE_FIELDS = ("none", "static", "travelling")


def cycle(fdir, out, *, n_frames=90, n_cycles=3, fields="none", m=3,
          rad_cells=2.2, npol=30, elev=22.0, azim=-56.0, fps=20, dpi=110,
          figsize=(6.0, 6.0), zoom=1.3, I_dc=0.5, I_ac=1.0, rho_amp=0.7,
          rho_dc=0.0, mag_frac=0.15, keep_frames=True) -> Path:
    """Animate the charge phase travelling around the core tube.

    The leading time dependence of a static soliton is its internal rotation
    Phi1 -> e^{-i*theta} Phi1. Because arg(phi1) winds SPATIALLY along the loop, a
    uniform sweep of theta slides the colour bands AROUND the tube -- the
    circulation you would see running it forward, imposed cleanly rather than
    obtained by evolving the ungauged scalar and watching it disperse.

    Tube geometry and the base phase are sampled from the field ONCE; each frame
    only recolours. That is why this is cheap even at 90 frames.

    fields:
      "none"        just the phase travelling on the tube.
      "static"      the field's OWN E and B from its stored s and A, drawn once
                    and held. Honest: those fields have no phase parameter, so
                    they cannot travel. The tube animates, the lines do not.
      "travelling"  E and B from a MODELLED m-harmonic source deposited on the
                    traced core, recombined per frame by linearity so the fields
                    travel with the phase. Labelled as a model in the frame title
                    because it is NOT the field's stored gauge sector -- it is
                    what a travelling current on this core WOULD produce.

    rho_dc defaults to 0, i.e. the DC charge term is EXCLUDED. The solvers here
    are periodic and drop the k=0 mode, so a net charge cannot be represented: the
    DC term came back as a near-uniform field whose lines ran straight across the
    whole box instead of radiating from the core. The AC (cos/sin) components carry
    zero net charge by construction, so the periodic solve is exactly right for
    them -- and they are the travelling part, which is the point of the animation.
    Restoring the monopole needs the open-BC solver the upstream had and this does
    not.
    """
    if fields not in CYCLE_FIELDS:
        raise ValueError(f"fields must be one of {CYCLE_FIELDS}")
    F = load_gauged_field(fdir)
    N, L, dx = F["N"], F["L"], F["dx"]
    g = grid1d(N, L)
    core = core_curve(F["p1"], N, dx)
    if core is None:
        raise SystemExit("cycle: no phi1 core found")
    center, half = viz.bbox_of_points(core)
    view = view_dir(elev, azim)

    # sample the N^3 phase field ONCE; frames only recolour these quads
    X, Y, Z, _, _ = viz.tube_surface(core, rad_cells * dx, npol)
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)
    base = (sample_phase(F["p1"], N, L, pts).reshape(X.shape) + np.pi) / (2 * np.pi)
    V = np.stack([X, Y, Z], -1)
    Vi1 = np.roll(V, -1, 0)
    tube_faces = np.stack([V, Vi1, np.roll(Vi1, -1, 1), np.roll(V, -1, 1)],
                          axis=2).reshape(-1, 4, 3)
    ring = viz.iso_parts(np.abs(F["p2"]), 0.5, dx, C_PHI2[:3] + (0.30,), sigma=0.0)

    static_parts, decomp, note = [], None, ""
    if fields == "static":
        for kind in ("E", "B"):
            rib, n, _ = field_line_parts(F, kind, core, center, half, elev, azim,
                                         mag_frac=mag_frac)
            if rib is not None:
                static_parts.append(rib)
        note = "\nstored E,B, held static -- they carry no phase to advance"
    elif fields == "travelling":
        print(f"depositing modulated sources on the core (m={m}) and solving "
              f"Maxwell once per component...")
        Xg, Yg, Zg = np.meshgrid(g, g, g, indexing="ij")
        S = deposit_modulated_sources(core, (Xg, Yg, Zg), m, dx=dx)
        jmax = np.sqrt(sum(c ** 2 for c in S["Jdc"])).max() + 1e-12
        B3 = tuple(magnetic_field([c / jmax for c in S[k]], dx)
                   for k in ("Jdc", "Jc", "Js"))
        E3 = tuple(electric_field(S[k], dx) for k in ("Rdc", "Rc", "Rs"))
        decomp = (E3, B3)
        note = (f"\nMODELLED travelling E,B: m={m} source on the core, "
                f"AC only (not the stored A)")

    hdr = _header(fdir, F)
    print(f"phase cycle: {n_frames} frames, {n_cycles} turn(s) around the loop, "
          f"fields={fields}")

    def draw(fig, k):
        phi = 2 * np.pi * k / n_frames
        p01 = (base - n_cycles * k / n_frames) % 1.0
        parts = ([ring] if ring is not None else []) + \
            [(tube_faces, viz.quad_phase_colors(p01, "twilight_shifted"))] + \
            list(static_parts)
        if decomp is not None:
            (Edc, Ec, Es), (Bdc, Bc, Bs) = decomp
            cph, sph = np.cos(phi), np.sin(phi)
            B = [I_dc * Bdc[d] + I_ac * (cph * Bc[d] - sph * Bs[d])
                 for d in range(3)]
            E = [rho_dc * Edc[d] + rho_amp * (cph * Ec[d] - sph * Es[d])
                 for d in range(3)]
            th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            b_seeds = poloidal_seeds(core, dx, roff=3.2, n_along=12)
            rs = half * 1.05
            ph = np.linspace(0.3, np.pi - 0.3, 5)
            e_seeds = [center + rs * np.array([np.sin(p) * np.cos(t),
                                               np.sin(p) * np.sin(t), np.cos(p)])
                       for t in th for p in ph]
            for fld, seeds, col, wid, ds in (
                    (B, b_seeds, C_B, 0.010 * half, 0.45 * dx),
                    (E, e_seeds, C_E, 0.008 * half, 0.7 * dx)):
                ln = trace_field_lines(fld, g, seeds, n_steps=320, ds=ds,
                                       min_mag=_mag_floor(fld, mag_frac))
                rib = lines_to_ribbons(ln, view, wid, col)
                if rib is not None:
                    parts.append(rib)
        ax = fig.add_subplot(111, projection="3d")
        _scene(ax, parts, center, half, elev, azim,
               f"{hdr}\nphase cycle, {n_cycles}x around the loop{note}", zoom)
        fig.tight_layout(pad=0.2)

    return viz.animate(draw, n_frames, out, fps=fps, figsize=figsize, dpi=dpi,
                       keep_frames=keep_frames)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--field", required=True,
                    help="directory holding field.npz + manifest.json")
    ap.add_argument("--view", default="raw",
                    choices=(*VIEWS, "triptych", "cycle", "all"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--spin", type=int, default=0,
                    help="render a turntable GIF with this many frames")
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--azim", type=float, default=-56.0)
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--zoom", type=float, default=1.3,
                    help="enlarge the object inside the axes")
    ap.add_argument("--sigma", type=float, default=0.0,
                    help="raw view smoothing; 0 keeps the grid facets (no cheat)")
    ap.add_argument("--volume-frac", type=float, default=0.004,
                    help="cells view: fraction of the box drawn as cubes")
    ap.add_argument("--max-cells", type=int, default=20000)
    ap.add_argument("--cycle-fields", default="none", choices=CYCLE_FIELDS,
                    help="cycle view: none | static (stored E,B held) | "
                         "travelling (MODELLED source, fields travel)")
    ap.add_argument("--frames", type=int, default=90,
                    help="cycle view: number of frames")
    ap.add_argument("--cycles", type=int, default=3,
                    help="cycle view: phase turns around the loop")
    ap.add_argument("--m", type=int, default=3,
                    help="cycle view: harmonic of the modelled source")
    ap.add_argument("--drop-frames", action="store_true",
                    help="delete the PNG frames after writing a GIF")
    a = ap.parse_args(argv)

    fdir = Path(a.field)
    kw = dict(sigma=a.sigma, volume_frac=a.volume_frac,
              max_cells=a.max_cells)
    views = list(VIEWS) if a.view == "all" else [a.view]
    for v in views:
        if v == "cycle":
            cycle(fdir, a.out or fdir / "portrait_cycle.gif",
                  n_frames=a.frames, n_cycles=a.cycles, fields=a.cycle_fields,
                  m=a.m, elev=a.elev, azim=a.azim, fps=a.fps, dpi=a.dpi,
                  zoom=a.zoom, keep_frames=not a.drop_frames)
        elif v == "triptych":
            triptych(fdir, a.out or fdir / "portrait_triptych.png",
                     elev=a.elev, azim=a.azim, dpi=a.dpi, zoom=a.zoom)
        elif a.spin:
            spin(fdir, v, a.out or fdir / f"portrait_{v}.gif", frames=a.spin,
                 fps=a.fps, elev=a.elev, dpi=a.dpi, zoom=a.zoom,
                 keep_frames=not a.drop_frames, **kw)
        else:
            portrait(fdir, v, a.out or fdir / f"portrait_{v}.png",
                     elev=a.elev, azim=a.azim, dpi=a.dpi, zoom=a.zoom, **kw)


if __name__ == "__main__":
    main()
