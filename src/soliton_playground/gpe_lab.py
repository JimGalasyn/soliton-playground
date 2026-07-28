"""gpe_lab — shared GPE playground instruments (dimensionless: xi = c = 1, g = 1).

Seeds are wrap-clean by construction (mirror partners / kink-antikink pairs) and
every experiment should call seed_gate() before evolving — the census's
seed-artifact gate, added after the opener's run-1 boundary-sheet contamination
(jax-solitons#67).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from scipy import ndimage

from jax_solitons.grid import BoxGrid
from jax_solitons.models.gpe import GPEKineticTerm, GPEPotentialTerm
from jax_solitons.models.nlkg import _ring_factor
from jax_solitons.steppers.splitstep import make_splitstep

G = 1.0

# ------------------------------------------------------------------ provenance
# Every summary names the medium it ran in and the charge that forbids the
# entrant's decay (CENSUS_PROTOCOL.md "Output per entrant"), so two entrants
# with the same geometry in different media never collide in the bestiary. The
# case that forced this: the GPE trefoil below unties (knot type is not a GPE
# charge) while the Faddeev T(2,3) hopfion is pinned by Hopf charge Q_H = 2
# (jax_solitons.seeds.torus_knot_hopfion) — same name, same curve, different
# sector. Bare "trefoil T(2,3) -> METASTABLE" would read as a contradiction.
PRESET = "gpe-dimensionless"          # xi = c = 1, g = 1, no damping/pumping
MODEL = "GPE (split-step)"

# Protecting charges, as recorded per entrant. Short greppable tokens; "none"
# means nothing in this preset forbids the decay, so the bin is earned by
# lifetime alone and can never be `protected`.
CHARGE_NONE = "none"
CHARGE_WINDING = "winding W (phase circulation quantum)"
# Knot type is NOT protected in GPE: reconnection leaves the +-1 winding around
# every strand intact while freely changing the knot, so it cannot bin as
# protected no matter how long it lives.
CHARGE_KNOT_UNPROTECTED = "none (knot type); winding W conserved per strand"


def provenance(protecting_charge: str) -> dict:
    """The census provenance block every summary.json carries."""
    return dict(preset=PRESET, model=MODEL,
                protecting_charge=protecting_charge)


def zoo_provenance(protecting_charge: str) -> dict:
    """The same provenance as event-graph particle attrs, so the lineage record
    is self-describing when read back without its summary."""
    return {"zoo.preset": PRESET, "zoo.model": MODEL,
            "zoo.protecting_charge": protecting_charge}


# ----------------------------------------------------------------- energetics
def make_energy(grid: BoxGrid):
    kin, pot = GPEKineticTerm(), GPEPotentialTerm(g=G)

    @jax.jit
    def energy(psi):
        return kin(psi, grid), pot(psi, grid)

    return energy


def smooth(grid: BoxGrid, psi, steps=30, dt=0.01):
    """Brief imaginary-time healing of an analytic seed."""
    step = make_splitstep(grid, dt, g=G, imaginary_time=True)
    for _ in range(steps):
        psi = step(psi)
    return psi


# ----------------------------------------------------------------- seeds
def ring_pair_seed(grid: BoxGrid, R: float, z0: float, xi: float = 1.0):
    """Vortex ring at z0 + mirror anti-ring at -z0 (wrap-clean pair; the z<0
    ring is the measured object)."""
    psi = (_ring_factor(grid, R=R, xi=xi, center=(0.0, 0.0, z0), axis="z", sign=1)
           * _ring_factor(grid, R=R, xi=xi, center=(0.0, 0.0, -z0), axis="z",
                          sign=-1))
    return jnp.asarray(psi, dtype=jnp.complex128)


def planar_soliton_pair_seed(grid: BoxGrid, z1: float, z2: float,
                             noise_amp: float = 0.05, noise_k: float = 0.5,
                             seed: int = 20260714):
    """Two black (stationary) planar dark solitons at z1 < z2 as a periodic
    kink-antikink pair: psi = tanh(z - z1 - d1(x,y)) * tanh(z2 + d2(x,y) - z).
    d_i are small smooth random displacement fields (low-pass-filtered noise)
    that seed the snake instability reproducibly."""
    X, Y, Z = (np.asarray(c) for c in grid.coords())
    rng = np.random.default_rng(seed)

    def displacement(salt):
        f = rng.standard_normal((grid.N, grid.N))
        fk = np.fft.fft2(f)
        k = 2 * np.pi * np.fft.fftfreq(grid.N, d=grid.dx)
        KX, KY = np.meshgrid(k, k, indexing="ij")
        fk *= np.exp(-(KX**2 + KY**2) / (2 * noise_k**2))
        d = np.real(np.fft.ifft2(fk))
        return noise_amp * d / np.abs(d).max()

    d1 = displacement(1)[:, :, None]
    d2 = displacement(2)[:, :, None]
    psi = np.tanh(Z - z1 - d1) * np.tanh(z2 + d2 - Z)
    return jnp.asarray(psi, dtype=jnp.complex128)


# ----------------------------------------------------------------- kick (gate 4)
def smooth_noise(grid: BoxGrid, k_cut=0.5, seed=0):
    """Low-pass-filtered 3D Gaussian noise, unit-normalized by peak |amplitude|.

    k-space envelope built by broadcasting the 1D wavenumber axis rather than
    np.meshgrid, which would materialize three N^3 grids to use one sum (the
    same waste that swap-thrashed the first N=256 trefoil run)."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((grid.N,) * 3)
    k = 2 * np.pi * np.fft.fftfreq(grid.N, d=grid.dx)
    k2 = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
    f = np.real(np.fft.ifftn(np.fft.fftn(f) * np.exp(-k2 / (2 * k_cut**2))))
    return f / np.abs(f).max()


def knot_envelope(grid: BoxGrid, scale: float, r0=2.2, width=0.3):
    """The Milnor seed's own radial blend, w = 1 inside r0*scale -> 0 outside.
    Reused as the kick window so a perturbation cannot reach the boundary."""
    ax = np.asarray(grid.axis()) / scale
    r = np.sqrt(ax[:, None, None] ** 2 + ax[None, :, None] ** 2
                + ax[None, None, :] ** 2)
    return 0.5 * (1.0 - np.tanh((r - r0) / width))


def kick_field(grid: BoxGrid, psi, eps=0.10, k_cut=0.5, envelope=None, seed=0):
    """Census gate-4 kick: psi -> psi * (1 + eps * chi * w), where chi is complex
    smooth noise with |chi| <= 1 and w is a window.

    COMPLEX on purpose: a real (amplitude-only) kick injects no current, so it
    cannot excite the velocity field a knot actually lives in. This perturbs
    density and phase together, i.e. an acoustic kick, with the field amplitude
    moved by at most eps where the window is open.

    WINDOWED because gate 0 outranks gate 4: an unwindowed eps=0.10 kick leaves
    the boundary shell at 1 - n ~ 0.19, twenty times seed_gate's 0.02 tolerance,
    which would invalidate the run before any physics happened. The kick is
    supposed to perturb the object, not the box, and the window is what makes it
    the object's kick. Pass envelope=knot_envelope(...) for a localized entrant.
    """
    chi = smooth_noise(grid, k_cut, 2 * seed + 1) \
        + 1j * smooth_noise(grid, k_cut, 2 * seed + 2)
    chi /= np.abs(chi).max()
    if envelope is not None:
        chi = chi * envelope
    arr = np.asarray(psi) * (1.0 + eps * chi)
    return jnp.asarray(arr, dtype=jnp.complex128)


# ----------------------------------------------------------------- gates
def seed_gate(grid: BoxGrid, psi, shell=4.0, tol_density=0.02, tol_jump=0.05,
              axes=(0, 1, 2)):
    """Seed-artifact gate: the boundary shell must be vacuum and the wrap phase
    mismatch negligible. Returns (ok, report).

    `axes` names the directions in which the entrant is LOCALIZED — the shell
    density check applies only there (a planar soliton legitimately crosses the
    transverse boundaries; pass axes=(2,) for a z-localized plane pair). The
    wrap-jump check runs on every axis but only where the boundary planes carry
    bulk density (phase is branch-cut noise inside depleted cores)."""
    arr = np.asarray(psi)
    dens = np.abs(arr) ** 2
    ax = np.asarray(grid.axis())
    m = np.abs(ax) > (grid.L / 2 - shell)
    slicers = [(np.s_[m, :, :]), (np.s_[:, m, :]), (np.s_[:, :, m])]
    shell_min = min(float(dens[slicers[i]].min()) for i in axes)
    jumps = []
    for axis_i in range(3):
        a = np.moveaxis(np.angle(arr), axis_i, -1)
        d = np.moveaxis(dens, axis_i, -1)
        bulk = np.minimum(d[..., 0], d[..., -1]) > 0.5
        if not bulk.any():
            continue
        j = (a[..., 0] - a[..., -1] + np.pi) % (2 * np.pi) - np.pi
        jumps.append(float(np.abs(j[bulk]).max()))
    jump_max = max(jumps) if jumps else 0.0
    ok = (1.0 - shell_min) < tol_density and jump_max < tol_jump
    return ok, dict(shell_min_density=float(shell_min), wrap_jump_max=jump_max)


# ----------------------------------------------------------------- trackers
# Both metrics below run once per sample (81x per campaign run) and used to call
# grid.coords()[2], which materializes all THREE N^3 float64 coordinate grids to
# hand back one: 402 MB per call at N=256. That churn (plus the N^3 z-indexed
# temporaries it fed) swap-thrashed the first N=256 trefoil attempt to 10 GB RSS
# and 18% CPU on a 15 GB host — the run had to be killed, having reached neither
# a verdict nor an OOM. Since z varies along one axis only, the 1D axis suffices:
# it broadcasts along axis 2 for masking, and both centroids collapse to a
# per-plane count/weight dotted with that axis. Numerically identical (pinned by
# tests/test_metrics_equivalence.py), ~400x less memory traffic.
def depletion_metrics(psi, grid: BoxGrid, thresh=0.5, zmax=None):
    """Volume, blob count, and z-centroid of depletion (optionally z < zmax)."""
    dens = np.asarray(jnp.abs(psi) ** 2)
    z = np.asarray(grid.axis())                  # 1D; broadcasts along axis 2
    mask = dens < thresh
    if zmax is not None:
        mask &= z < zmax
    vol = float(mask.sum()) * grid.dx**3
    n_blobs = int(ndimage.label(mask)[1]) if mask.any() else 0
    # sum_{ijk} z_k m_ijk / sum m == (per-plane counts) . z / total
    counts = mask.sum(axis=(0, 1))
    n_tot = counts.sum()
    zc = float((counts * z).sum() / n_tot) if n_tot else float("nan")
    return dict(V_dep=vol, n_blobs=n_blobs, z_dep=zc, n_min=float(dens.min()))


def dip_centroid_z(psi, grid: BoxGrid, floor=0.05, zmax=None):
    """z-centroid weighted by density deficit (1 - n - floor)_+ — tracks both
    vortex cores AND rarefaction pulses (whose minimum density is > 0)."""
    dens = np.asarray(jnp.abs(psi) ** 2)
    w = np.clip(1.0 - dens - floor, 0.0, None)
    z = np.asarray(grid.axis())                  # 1D; broadcasts along axis 2
    if zmax is not None:
        w = np.where(z < zmax, w, 0.0)
    tot = w.sum()
    return float((w.sum(axis=(0, 1)) * z).sum() / tot) if tot > 0 \
        else float("nan")


def winding_xz(psi, grid: BoxGrid, x_c, z_c, half=3.0):
    """Phase winding (units of 2*pi) around a square loop in the y=0 plane."""
    ph = np.angle(np.asarray(psi)[:, grid.N // 2, :])
    ax = np.asarray(grid.axis())
    i0, i1 = np.searchsorted(ax, [x_c - half, x_c + half])
    k0, k1 = np.searchsorted(ax, [z_c - half, z_c + half])
    i1, k1 = min(i1, grid.N - 1), min(k1, grid.N - 1)
    path = ([(i, k0) for i in range(i0, i1)] +
            [(i1, k) for k in range(k0, k1)] +
            [(i, k1) for i in range(i1, i0, -1)] +
            [(i0, k) for k in range(k1, k0, -1)])
    tot = 0.0
    for (a, b), (c, d) in zip(path, path[1:] + path[:1]):
        dphi = ph[c, d] - ph[a, b]
        tot += (dphi + np.pi) % (2 * np.pi) - np.pi
    return tot / (2 * np.pi)


# ----------------------------------------------------------------- evolution
def evolve(grid: BoxGrid, psi, *, T, dt, sample_dt, observer, keep_slices_at=()):
    """Real-time split-step evolution with periodic sampling; observer(t, psi)
    returns a dict row. Returns (psi, rows, slices{t: |psi|^2 y=0 plane})."""
    step = make_splitstep(grid, dt, g=G, imaginary_time=False)
    every = max(1, int(round(sample_dt / dt)))
    steps = int(round(T / dt))
    rows, slices = [], {}
    for i in range(steps + 1):
        t = i * dt
        if i % every == 0:
            rows.append(dict(t=t, **observer(t, psi)))
        for ts in keep_slices_at:
            if abs(t - ts) < 0.5 * dt:
                slices[float(ts)] = np.asarray(jnp.abs(psi[:, grid.N // 2, :]) ** 2)
        if i < steps:
            psi = step(psi)
    return psi, rows, slices


# ----------------------------------------------------------------- figure style
DARK_STYLE = {
    "figure.facecolor": "black", "axes.facecolor": "black",
    "savefig.facecolor": "black", "text.color": "#DDDDDD",
    "axes.edgecolor": "#555555", "axes.labelcolor": "#BBBBBB",
    "xtick.color": "#999999", "ytick.color": "#999999",
    "font.family": "monospace", "axes.grid": False,
}
C_BLUE, C_ORANGE, C_GREEN = "#56B4E9", "#E69F00", "#009E73"  # Okabe–Ito
