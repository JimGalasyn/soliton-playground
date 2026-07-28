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


# ------------------------------------------------- characteristic period (gate 1)
# Bogoliubov sound speed in this preset. The module convention asserts c = 1;
# measured empirically at 1.0446 (N=256, L=128) and 1.0714 (N=128, L=64) by
# tracking a small density pulse, the ~5% excess being peak-quantization at
# dx = 0.5. Pinned by tests/test_characteristic_period.py so the clock gate 1
# depends on is a measured quantity and not an assumed one.
C_SOUND = 1.0
XI = 1.0                       # healing length in this preset (dimensionless)

# Gate-1 clocks. WHICH CLOCK WAS USED MUST BE NAMED BESIDE THE BIN, because the
# same lifetime reads as two different bins depending on the choice: the trefoil
# unties at 0.13-0.26 traversal periods but 20-40 local-reconnection periods.
CLOCK_TRAVERSAL = "traversal (tau = L_structure / c)"
CLOCK_LOCAL_RECONNECTION = "local reconnection (tau = xi / c)"


def local_reconnection_period(xi: float = XI, c: float = C_SOUND) -> float:
    """Gate-1 clock for an entrant whose decay is a LOCAL event.

    A knot does not untie by anything traversing it; it unties where two strands
    approach within a core radius, so the process is set by the core-crossing time
    xi/c rather than the whole structure's transit time. The right clock for
    counting survival is the one the decay mechanism actually runs on.

    Note this does NOT rescue the trefoil from a 50-period threshold: it unties at
    20-40 local periods, still short of the default N = 50. The clock changes the
    margin (194x short becomes 1.25-2.5x short), not the verdict, and metastable
    would require a declared N <= 20.
    """
    return xi / c


def characteristic_period(structure_length: float, c: float = C_SOUND) -> float:
    """Time for a wave to traverse the entire structure once: tau = L / c.

    The clock gate 1 counts in. For an extended entrant the length is its own
    traced extent — for a closed vortex curve, the full arc length, since
    traversing a loop once means going all the way around it.

    Note what this implies for a DECAYING entrant. The trefoil's curve is
    155.5 xi, so tau ~ 155.5, while it unties by t ~ 20-40: a quarter of one
    traversal. Its whole 80-unit run is half a period. A survival THRESHOLD is
    therefore not something such an entrant can be measured against — see the
    gate 1 note in CENSUS_PROTOCOL.md.
    """
    return structure_length / c


# ----------------------------------------------------------------- calorimeter
def helmholtz_energies(grid: BoxGrid, u):
    """Split a vector field u = (ux, uy, uz) into solenoidal and irrotational
    energies. Returns (E_incompressible, E_compressible, E_total).

    Exact by Parseval: the two projections are orthogonal in k-space, so the
    energies add with no cross term.

    ASSUMES u IS PERIODIC. That is not a formality — it is the trap this module
    exists to avoid. A single straight vortex in a box is NOT periodic (its phase
    winds by 2 pi), and an FFT of it puts an O(1/dx) discontinuity at the box
    face which lands in BOTH sectors and grows under refinement, looking exactly
    like a convergence failure of the physics. Feed this only fields that pass
    seed_gate; the wrap-clean mirror-pair seeds in this module are pairs
    precisely so that they do.
    """
    N, dx = grid.N, grid.dx
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    kx, ky, kz = k[:, None, None], k[None, :, None], k[None, None, :]
    k2 = kx**2 + ky**2 + kz**2
    uk = [np.fft.fftn(c) for c in u]
    kdotu = kx * uk[0] + ky * uk[1] + kz * uk[2]
    # k=0 is the uniform-flow mode: solenoidal and irrotational at once, so the
    # split is ambiguous there. Assigned to incompressible by leaving it out of
    # the compressible projection; in a box with no net flow it is ~0 anyway.
    safe = np.where(k2 > 0, k2, 1.0)
    fac = np.where(k2 > 0, kdotu / safe, 0.0)
    pref = 0.5 * dx**3 / N**3
    dens_c = sum(np.abs(kk * fac) ** 2 for kk in (kx, ky, kz))
    E_c = pref * float(np.sum(dens_c))
    E_flow = pref * sum(float(np.sum(np.abs(c) ** 2)) for c in uk)
    # Fraction of the SOUND energy living above half-Nyquist. This is the test
    # that separates "radiated into the medium" from "on its way into the grid":
    # physical phonons from reconnection sit at k ~ 1/xi, well resolved, while
    # energy piling up near k_max is about to be truncated away and would show
    # up as drift, not as sound. A rising high-k fraction is the warning sign.
    k_nyq = np.pi / dx
    hi = k2 > (0.5 * k_nyq) ** 2
    E_c_hi = pref * float(np.sum(dens_c[hi]))
    return E_flow - E_c, E_c, E_flow, (E_c_hi / E_c if E_c > 0 else 0.0)


def energy_partition(grid: BoxGrid, psi, g=G, delta=1e-10):
    """Nore-Abid-Brachet calorimeter: split the GPE energy into four sectors.

    Writing psi = sqrt(n) e^{i phi}, the identity |grad psi|^2 = |grad sqrt(n)|^2
    + n |grad phi|^2 splits the kinetic energy into a quantum-pressure part and a
    flow part. The flow part carries the density-weighted velocity
    u = sqrt(n) grad phi, and Helmholtz-splitting u into a solenoidal and an
    irrotational piece separates the two things gate 2 needs to tell apart:

      E_i    incompressible flow, div u = 0    -> BOUND in the vortex lines
      E_c    compressible flow, curl u = 0     -> SOUND, the radiated sector
      E_q    quantum pressure, 0.5|grad sqrt(n)|^2
      E_int  interaction, 0.5 g (n-1)^2

    E_c is the phonon budget. This is what makes "every loss accounted by the
    calorimeter (radiated sector), not the grid" a measurement instead of an
    assertion: energy leaving E_i must show up in E_c.

    u is computed as Im(conj(psi) grad psi)/sqrt(n + delta) rather than from an
    unwrapped phase, which would be singular at the cores. u itself is finite
    there (n ~ r^2 and grad phi ~ 1/r, so u ~ r * 1/r), but both numerator and
    denominator vanish, hence delta. The sum rule below is what validates that
    choice: if delta were distorting the cores, E_q + E_flow would stop matching
    the spectral kinetic energy.

    SPECTRAL, whereas the ledger's GPEKineticTerm uses forward differences. The
    two therefore disagree by a discretization term, reported as
    `kinetic_fd_minus_spectral` rather than papered over. Compare drifts and
    transfers within one convention, never across the two.

    Returns a dict of the four sectors plus:
      E_flow      = E_i + E_c
      E_tot       = E_q + E_i + E_c + E_int   (spectral total)
      sum_rule_residual  E_kin_spectral - (E_q + E_flow); the calorimeter's OWN
                  closure error, distinct from the integrator's drift
      kinetic_fd_minus_spectral   ledger convention minus spectral
      E_c_highk_frac  fraction of E_c above half-Nyquist: the "sound vs grid"
                  discriminator, since energy piling up near k_max is about to be
                  truncated rather than radiated
    """
    arr = np.asarray(psi)
    n = np.abs(arr) ** 2
    N, dx = grid.N, grid.dx
    vol = dx**3
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    kx, ky, kz = k[:, None, None], k[None, :, None], k[None, None, :]
    k2 = kx**2 + ky**2 + kz**2

    pk = np.fft.fftn(arr)
    dpsi = [np.fft.ifftn(1j * kk * pk) for kk in (kx, ky, kz)]
    E_kin_spec = 0.5 * sum(float(np.sum(np.abs(d) ** 2)) for d in dpsi) * vol

    root = np.sqrt(n + delta)
    u = [np.imag(np.conj(arr) * d) / root for d in dpsi]

    sk = np.fft.fftn(np.sqrt(n))
    E_q = 0.5 * sum(float(np.sum(np.abs(np.fft.ifftn(1j * kk * sk)) ** 2))
                    for kk in (kx, ky, kz)) * vol

    E_i, E_c, E_flow, E_c_hi_frac = helmholtz_energies(grid, u)
    E_int = 0.5 * g * float(np.sum((n - 1.0) ** 2)) * vol

    kin_fd, _ = make_energy(grid)(jnp.asarray(arr))
    return dict(E_i=E_i, E_c=E_c, E_q=E_q, E_int=E_int, E_flow=E_flow,
                E_c_highk_frac=E_c_hi_frac,
                E_tot=E_q + E_i + E_c + E_int,
                E_kin_spectral=E_kin_spec,
                sum_rule_residual=E_kin_spec - (E_q + E_flow),
                kinetic_fd_minus_spectral=float(kin_fd) - E_kin_spec)


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


def kick_energy_referenced(grid: BoxGrid, psi, eps=0.10, k_cut=0.5,
                           envelope=None, seed=0, E_ref=None, tol=2e-3,
                           max_iter=60):
    """Kick whose INJECTED ENERGY is eps * E_ref. Returns (psi_kicked, report).

    Ported from the retired program's kick fleet (null-worldtube-private,
    `simulations/engine_dogfood/eps_kick_batch.py`), which scales its noise by
    sqrt(eps * Epot / KE(w)) so that eps *is* the injected energy fraction, and
    uses a common absolute reference across models so comparisons are
    apples-to-apples. Adopt that convention: an amplitude-referenced kick is not
    comparable between objects, models, or even seeds. Our own gate-4 run used
    10% AMPLITUDE, which turned out to be ~1% ENERGY — a tenth of that fleet's
    smallest step — so its PASS was much weaker than it sounded.

    Two adaptations were forced by the model. GPE is FIRST-ORDER in time, so
    there is no independent velocity field to kick: their trick of leaving the
    configuration untouched (`bn = n0`) and putting all the noise into `bv` has
    no GPE analogue, and the perturbation must move psi itself. And because the
    GPE energy is not quadratic in the perturbation amplitude, their closed-form
    sqrt rescaling does not hit the target; the amplitude is bisected instead.

    A consequence worth stating: this kick necessarily perturbs the
    configuration, so unlike the Faddeev velocity kick it cannot be guaranteed
    topology-preserving at large eps. Check the seed gate and the initial
    topology after kicking, not just before.
    """
    energy = make_energy(grid)
    arr = np.asarray(psi)
    if E_ref is None:
        k, p = energy(jnp.asarray(arr))
        E_ref = float(k + p)
    target = abs(eps * E_ref)

    chi = smooth_noise(grid, k_cut, 2 * seed + 1) \
        + 1j * smooth_noise(grid, k_cut, 2 * seed + 2)
    chi /= np.abs(chi).max()
    if envelope is not None:
        chi = chi * envelope

    def dE(a):
        k, p = energy(jnp.asarray(arr * (1.0 + a * chi), dtype=jnp.complex128))
        return abs(float(k + p) - E_ref)

    lo, hi = 0.0, 0.05
    for _ in range(40):                     # bracket
        if dE(hi) >= target:
            break
        hi *= 1.8
    else:
        raise RuntimeError(f"could not reach eps={eps} (max dE {dE(hi):.4g})")

    for _ in range(max_iter):               # bisect
        mid = 0.5 * (lo + hi)
        d = dE(mid)
        if abs(d - target) <= tol * target:
            break
        lo, hi = (mid, hi) if d < target else (lo, mid)
    a = 0.5 * (lo + hi)
    out = jnp.asarray(arr * (1.0 + a * chi), dtype=jnp.complex128)
    k, p = energy(out)
    return out, dict(eps_requested=eps, amplitude=a,
                     dE_over_E=(float(k + p) - E_ref) / E_ref, E_ref=E_ref)


def survival_bucket(crossings, reference):
    """Score one kicked realization against the unkicked knot type, in the three
    buckets the prior program's `eps_kick_id.py` used.

    The third bucket is the one that matters and that a naive pass/fail loses: a
    tracer that cannot identify the curve is NOT evidence of decay. Our own
    unperturbed t=80 trace was degenerate and would have been miscounted as a
    decay by a two-way score.
    """
    if crossings is None:
        return "unidentifiable"
    return "survived" if crossings == reference else "decayed"


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
