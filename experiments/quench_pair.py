#!/usr/bin/env python3
"""Release a composed pair FORWARD IN TIME — the quench half of compose_pair.

`compose_pair.py` stages two catalog particles and then descends: its motion
meter is `ER.relax_iter`, and gradient flow IS force-following motion, so what
it measures is the SIGN of a quasi-static force. It cannot show a reconnection,
it cannot radiate, and by DESIGN.md's P7 note descent cannot create topology at
all. This driver takes the same composed state and hands it to
`jax_solitons.ehn.quench`, where the gauge sector carries its own conjugate
momentum E and the thing is allowed to ring.

THE MAPPING IS SEVEN-EIGHTHS FREE. `compose_pair()` returns a 10-tuple
(φ₁re, φ₁im, φ₂re, φ₂im, Ax, Ay, Az, Bx, By, Bz) and quench wants
(φ₁, φ₂, Ax, Ay, Az, Ex, Ey, Ez) plus `s`:

  φ₁,φ₂   u[0]+iu[1], u[2]+iu[3]        straight across
  A       u[4:7]                        straight across (composition is linear in A)
  B       u[7:10]                       DROPPED — quench has no auxiliary B, it
                                        recomputes B ≡ ∇×A (see quench's note on
                                        the one difference between the engines)
  w       n/a                           DROPPED — no multiplier in the quench
  s       solve_A0(...)                 the A₀ the R-C-LC-1 gate requires
  E       *** the only field that is not there ***

E IS ZERO HERE, AND THAT IS THE EXPERIMENT. A relaxed state has no velocities —
gradient flow has no momenta to store — so the initial E is a CHOICE, not a
recovery. E = 0 is "placed at separation and released from rest", which is the
dynamical version of exactly the question compose_pair asks quasi-statically.
A boosted collision needs a boost prescription for the GAUGED two-scalar state
(`steppers/verlet.boost_velocity` is the ungauged n-field and does not reach
here) and is deliberately not attempted.

ON THE GAUSS RESIDUAL, WHICH IS NOT AN ADMISSION GATE. It is tempting to screen
the initial data with `gauss_residual`, and it does not work: in temporal gauge
with the transverse projection ∇·E ≡ 0 by construction, so at E = 0 the residual
is ‖0 + src‖/‖src‖ = 1.0 EXACTLY, for any state whatsoever. It carries no
information at n=0. What it can say is whether the transverse treatment is being
asked to carry charge dynamics it cannot represent, and that shows up as GROWTH,
so it is sampled along the trajectory and reported as a trend.

THE CORRECTNESS GATE IS dH/H, AND IT ONLY EXISTS AT C_l3 = 0. The ℒ₃ lock is a
descent step on φ₂ (`−α_l3·∇E_L3`), dissipative by construction, so energy
conservation is a statement about the C_l3 = 0 sector only — quench's own tests
assert it exactly there. Hence two phases, and `--calibrate` runs the first:

  1. CALIBRATE (C_l3 = 0): the same second-order-in-dt drift check the engine's
     contract tests use, but at THIS box and THESE parameters rather than the
     tests' N=24, L=6, λ=50. It matters: the catalog relaxes at λ=1000, and the
     nonlinear kick's phase advance goes like dt·2λ·|c|, so a dt that is
     conservative in the test suite can be nonsense here. Halving dt must
     quarter the drift; if it does not, the dt is not in the asymptotic regime
     and no number from phase 2 means anything.
  2. RELEASE (C_l3 = C from the catalog entry, default): the physics run. The
     lock is ON because the transverse-only form FAILED the R-C-LC-1 gate —
     without the A₀-coupled energy the link is not protected — so this phase
     trades the conservation gate for topology protection, knowingly.

THE LOCK'S DESCENT STEP DOES NOT SURVIVE THE JUMP TO A CATALOG STATE, and this
is measured, not suspected. `quench.evolve`'s default alpha_l3 = 4e-4 is set
against the contract tests, which run C_l3 ~ 1. A catalog entry carries C = 400,
and the ℒ₃ force is `−alpha_l3·∇E_L3` with E_L3 linear in C_l3, so the effective
step is 400x what was validated. Measured on trefoil_t23_compact, N=192, dt=1e-4,
released from rest (single particle, no composition):

    alpha_l3   C_l3    step 1      step 2      step 3      step 4
    4e-4       400     +4.5e+01    +5.7e+05    +6.9e+73    NaN
    4e-6       400     +1.8e-02    +1.8e-02    +1.8e-02    +1.8e-02
    4e-8       400     +1.8e-02    +1.8e-02    +1.8e-02    +1.8e-02
    (any)      0       +1.8e-02    +1.8e-02    +1.8e-02    +1.8e-02   [dH/H]

Two things to read off. The base integrator is FINE at dt=1e-4: the C_l3 = 0 row
is flat after a +1.8% first step, and that first step is the re-settle transient
below, not an error. And the instability tracks alpha_l3*C_l3, not C_l3 alone —
keep that product at or under the validated ~4e-4. The default is left at the
engine's value rather than silently re-tuned here, because the number that
belongs in the engine is a question for whoever owns the R-C-LC-1 gate; this
driver refuses to run past the product instead of guessing.

THE CATALOG'S dx = 0.8 IS UNDER-RESOLVED, AND THAT — NOT dt — IS WHAT LIMITS A
RELEASE RUN. Measured with `--regrid`, which spectrally resamples one catalog
state onto finer grids at fixed L, so the physical field and its distance from
stationarity are held constant and only dx moves (T = 0.04, C_l3 = 0, f32):

    N     dx      transient   amplitude   dt-ratio   [dt = 1e-4]
    192   0.800   1.840e-02   4.645e-04   1.06
    224   0.686   2.269e-03   2.395e-04   5.64
    256   0.600   2.215e-03   2.557e-04   8.10

The transient falls 8x between dx = 0.8 and 0.686 and then PLATEAUS, and the
dt-ratio goes from 1.06 (dt-blind: the error is spatial) to dt-sensitive over the
same step. Both say the same thing: at dx = 0.8 the quartic at lam = 1000 puts
content past Nyquist with no dealiasing, and by dx ~ 0.69 it no longer does.

Refining dx alone is not enough, because k_max ~ 1/dx tightens the CFL as the
grid improves — at the coarse dt = 2e-4 the amplitude gets WORSE on finer grids
(4.9e-4 -> 1.35e-3 -> 2.07e-3 for the three rows above). dt must come down with
dx. At N = 256:

    dt        amplitude   ratio vs previous
    1.0e-04   2.557e-04   --
    5.0e-05   7.796e-05   3.28   <- second-order scaling recovered; gate passes
    2.5e-05   6.605e-05   1.18   <- floor reached, no further gain

So dt = 5e-5 at dx = 0.6 is the working point: it is where the ~4x scaling
appears and where refining dt stops buying anything. The residual 6.6e-5 floor is
dt-independent again, i.e. it is the spatial error AT dx = 0.6 — an order of
magnitude better than dx = 0.8's 4.6e-4, and the lever on it is a finer grid.

WHAT THE GATE STILL CANNOT SAY. `secular/amplitude` never drops under 0.1 (0.59
at the best point) and that is a limitation of the measurement, not a verdict on
the engine: the oscillation period is ~0.03 in time, so T = 0.04 covers barely
one cycle, and a least-squares line through less than a full period fits the
oscillation itself. Distinguishing a genuine drain from a wobble needs T over
several periods. That run has not been made; do not read the DRAINING flag as
evidence of dissipation until it has.

A sharper trap sits underneath: the blow-up gets WORSE as A₀ converges
(--a0-iters 20 survives, 500 diverges), because a better-converged s makes the
ℒ₃ force bigger. A short A₀ solve therefore looks stable for the wrong reason,
which is exactly the kind of accidental pass worth naming.

Re-settle caveat (particle_catalog, engine-static): a cached state is stationary
under THE INTEGRATOR THAT RELAXED IT. Quench is a different integrator, so some
initial transient is expected and is not by itself a failure — the +1.8% first
step above is it, and it settles rather than compounding.

  python quench_pair.py --name trefoil_t23_compact --calibrate
  python quench_pair.py --name trefoil_t23_compact --sep 76.8 --steps 20000
  python quench_pair.py --name trefoil_t23 --partner unknot_bare --orient mirror
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from jax_solitons.ehn import quench as Q
from jax_solitons.ehn import relax as ER

# Sibling script, not a package: experiments/ is sys.path[0] when this is run
# directly. Imported rather than re-derived so the composition here is the SAME
# composition compose_pair scores — an independent transcription of the product
# ansatz is exactly how the two drivers would silently diverge.
from compose_pair import (
    ER_DX,
    cat_load,
    compose_pair,
    knot_determinants,
    pair_separation,
    solve_A0,
)


def regrid(f, M):
    """Spectral resample of one real field from N³ to M³ at the SAME box length.

    Zero-padding in k-space is exact for a band-limited field: the continuous
    function is unchanged, only its sampling is. That is what makes this the
    right instrument for the resolution question — a state relaxed afresh at
    another N would differ BOTH in dx and in how stationary it is, and the two
    effects would be inseparable. Here only dx moves.

    Upsampling (M > N) is the clean direction and the one the sweep uses.
    Downsampling truncates real content and is genuinely lossy; it is allowed
    because a strong response in that direction is itself evidence, but the
    field it produces is a different physical field and is labelled as such.

    THE INVARIANCE IS NOT TOTAL, AND THE EXCEPTION IS LOUD. Spectral functionals
    are preserved exactly (measured on a band-limited test field: gradient energy
    agrees to 13 digits at N=48 vs 64). POINTWISE ones are not, because
    band-limited interpolation preserves Fourier coefficients, not the value of a
    nonlinear function of the field between the original sample points. On
    trefoil_t23_compact the effect is enormous — the components at dx=0.8 vs 0.6:

        term    N=192      N=256      note
        grad1     851.3      855.2    spectral, invariant
        grad2     903.6      896.4    spectral, invariant
        mag      1106.8     1111.1    spectral, invariant
        pot      -723.5   +15676.1    POINTWISE lam*c^2 -- rings, then squares
        total    2138.3    18538.7

    Read that as evidence, not noise: an interpolant that rings this hard between
    its own sample points is what an under-resolved core looks like. But a regrid
    state is NOT a physical soliton on the fine grid, and anything sensitive to
    the potential term must re-settle it (see --resettle) before drawing physics
    from it. Conservation tests survive the ringing — with gamma = eta = C_l3 = 0
    the flow is Hamiltonian and H is conserved from ANY initial state, so ringing
    redistributes energy among the terms without moving the total.
    """
    N = f.shape[0]
    if M == N:
        return f
    F = jnp.fft.fftshift(jnp.fft.fftn(f))
    if M > N:
        p = (M - N) // 2
        F = jnp.pad(F, ((p, M - N - p),) * 3)
    else:
        c = (N - M) // 2
        F = F[c:c + M, c:c + M, c:c + M]
    return jnp.real(jnp.fft.ifftn(jnp.fft.ifftshift(F))) * (M / N) ** 3


def energy_stats(samples):
    """Amplitude, secular slope and an oscillation-period estimate over a run.

    Skips n=0: that sample carries the initial-data transient (E starts at zero
    while curl B - gJ does not) and including it swamps everything after it.

    The period estimate is what makes the secular number readable. A least-squares
    line through LESS than one full oscillation fits the oscillation itself, so a
    pure wobble reports as a drain — which is exactly why the short-horizon gate
    could not settle this. `periods` says how much of a cycle the window actually
    covered, so the caller can tell a measurement from an artifact.
    """
    if len(samples) < 8:
        return {}

    def stats(sl):
        H = np.array([r["E_total"] for r in samples[sl]], dtype=float)
        n = np.array([r["n"] for r in samples[sl]], dtype=float)
        Hbar = float(H.mean())
        fit = np.polyfit(n, H, 1)
        d = H - np.polyval(fit, n)
        turns = int(np.sum(np.sign(np.diff(d))[:-1] != np.sign(np.diff(d))[1:]))
        return {"amplitude": float(H.max() - H.min()) / Hbar,
                "secular": abs(float(fit[0]) * (n[-1] - n[0]) / Hbar),
                "slope_per_step": float(fit[0]), "periods": turns / 2.0,
                "Hbar": Hbar}

    # THE TAIL IS THE MEASUREMENT; the full window is context. A run started from
    # a state that is not stationary for THIS engine spends its first stretch
    # shedding that mismatch, and a single line fitted across the whole window is
    # dominated by that transient -- it reported "SECULAR COMPONENT PRESENT" on a
    # run whose last quarter was drifting UP at 8e-6 per period. Measured on the
    # 12000-step N=256 run: the decay e-folds in ~10 oscillation periods and then
    # changes sign, so a drain is only real if it survives into the tail.
    full = stats(slice(1, None))
    tail = stats(slice(max(1, len(samples) // 2), None))
    out = {"full_" + k: v for k, v in full.items()}
    out.update({"tail_" + k: v for k, v in tail.items()})
    out["secular_over_amplitude"] = (tail["secular"] / tail["amplitude"]
                                     if tail["amplitude"] > 0 else float("inf"))
    out["periods"] = tail["periods"]
    return out


def preflight_knot_id():
    """Prove THIS process can identify a knot, before anything expensive runs.

    The lesson of the EHN-box rental (e55f4fe): a run that relaxes for hours and
    only then discovers it cannot name what it produced has bought nothing. The
    same asymmetry holds here at smaller scale — the A₀ solve and the evolution
    both precede the first determinant — so the check goes first and costs under
    a second. Identifies a known T(2,3) and requires det == 3, which exercises
    the whole path (pyknotid import, the numpy-alias shim, the Alexander
    routine) rather than merely that a module resolves.
    """
    try:
        from jax_solitons.knots import identify_knot, torus_knot
        det = identify_knot(torus_knot(2, 3))["determinant"]
    except Exception as exc:                                # ImportError or worse
        return f"{type(exc).__name__}: {exc}"
    return None if det == 3 else f"T(2,3) identified as det {det}, expected 3"


def to_quench(u, dx, C, U, eps_a, beta, a0_iters):
    """The 10-tuple relax state -> quench's field set, with E = 0.

    Returns (phi1, phi2, Ax, Ay, Az, Ex, Ey, Ez, s). B and w are dropped on
    purpose (see module docstring); `s` is converged with the fields frozen,
    because a half-solved A₀ would inject a transient that is indistinguishable
    from the physics this run exists to see.
    """
    phi1 = u[0] + 1j * u[1]
    phi2 = u[2] + 1j * u[3]
    Ax, Ay, Az = u[4], u[5], u[6]
    z = jnp.zeros_like(u[0])
    s = solve_A0(u, dx, C, U, eps_a, beta, iters=a0_iters)
    return phi1, phi2, Ax, Ay, Az, z, z, z, s


def drift(fields, kv, dx, *, dt, T, lam, kappa, agrad, n_samples=40):
    """Energy behaviour over horizon T at step dt, in the conservative sector.

    Returns (transient, amplitude, secular, steps).

    MEASURED, NOT ASSUMED: on a catalog state this engine's H does not drift, it
    OSCILLATES — 2177.23 -> 2176.78 -> 2177.07 -> 2176.88 over 400 steps, a
    bounded ~1.4e-4 wobble with no secular part. Differencing the endpoints (the
    obvious gate, and the first one written here) therefore measures the phase of
    that oscillation and nothing else: it came out dt-INDEPENDENT, which reads as
    "the integrator is not converging" when the truth is "the integrator is
    symplectic and I pointed the wrong instrument at it". Hence amplitude and
    least-squares slope over the whole sampled window instead of two endpoints.

    The first step carries a one-time offset that the rest of the trajectory does
    not: E is initialised to zero while ∇×B − gJ₁ is not, so the field energy the
    dynamics actually owns appears during step 1 rather than being present in the
    state we hand it. That offset is INDEPENDENT OF dt, so including it in the
    gate makes the ratio tend to 1 no matter how small dt gets — the engine looks
    dissipative, consistently, which is exactly the failure quench's own test
    docstring warns about. `from_1` is therefore the number that says something
    about the INTEGRATOR; `from_0` is the size of the initial-data transient, and
    is worth reporting rather than hiding.
    """
    phi1, phi2, Ax, Ay, Az, Ex, Ey, Ez, s = fields
    steps = max(2 * n_samples, int(round(T / dt)))
    # Split the first step out so H(1) is exact rather than interpolated: chaining
    # evolve is bit-identical to one call, since it is a pure state -> state map.
    st, s0 = Q.evolve(phi1, phi2, Ax, Ay, Az, Ex, Ey, Ez, kv, steps=1, dt=dt,
                      dx=dx, lam=lam, kappa=kappa, C_l3=0.0, agrad=agrad, s=s,
                      sample_every=1)
    _, s1 = Q.evolve(*st[:8], kv, steps=steps - 1, dt=dt, dx=dx, lam=lam,
                     kappa=kappa, C_l3=0.0, agrad=agrad, s=st[8],
                     sample_every=max(1, (steps - 1) // n_samples))
    h0, h1 = s0[0]["E_total"], s0[-1]["E_total"]
    H = np.array([r["E_total"] for r in s1], dtype=float)
    n = np.array([r["n"] for r in s1], dtype=float)
    Hbar = H.mean()
    amp = (H.max() - H.min()) / Hbar
    # Secular part: least-squares slope over the sampled window, expressed as the
    # fractional change it would produce across the whole horizon. A symplectic
    # step gives amp > 0 with slope ~ 0; a dissipative one gives slope < 0.
    slope = np.polyfit(n, H, 1)[0] * (n[-1] - n[0]) / Hbar if len(n) > 2 else 0.0
    return abs((h1 - h0) / h0), amp, abs(slope), steps


def calibrate(fields, kv, dx, *, dt, T, lam, kappa, agrad):
    """Second-order-in-dt drift check at THIS box. Returns True if dt is usable.

    Asserting merely "drift is small" would pass for a dissipative engine at
    small dt; the ORDER is what separates truncation error from physics. Same
    logic as the engine's own contract test, re-run where the numbers differ.
    """
    print("CALIBRATE (C_l3 = 0, the only sector where energy behaviour is a gate)")
    ct, ca, cs, ns_c = drift(fields, kv, dx, dt=2 * dt, T=T, lam=lam,
                             kappa=kappa, agrad=agrad)
    print(f"  dt={2*dt:.2e}  {ns_c:6d} steps  transient={ct:.3e}  "
          f"amplitude={ca:.3e}  secular={cs:.3e}", flush=True)
    ft, fa, fs, ns_f = drift(fields, kv, dx, dt=dt, T=T, lam=lam, kappa=kappa,
                             agrad=agrad)
    print(f"  dt={dt:.2e}  {ns_f:6d} steps  transient={ft:.3e}  "
          f"amplitude={fa:.3e}  secular={fs:.3e}", flush=True)
    if fa <= 0.0:
        print("  amplitude is exactly zero — suspect a sampler reading a "
              "constant, not a perfect integrator")
        return False
    ratio = ca / fa
    ok_order = 3.0 < ratio < 5.0
    ok_secular = fs < 0.1 * fa            # no drain hiding under the wobble
    ok_size = fa < 1e-3
    print(f"  the n=0 -> n=1 transient ({ft:.3e}) is initial data, not "
          f"integration: E starts at zero while curl B - gJ does not")
    print(f"  amplitude ratio = {ratio:.2f} (want ~4 for 2nd order)  ->"
          f" {'OK' if ok_order else 'NOT IN THE ASYMPTOTIC REGIME'}")
    print(f"  amplitude at dt = {fa:.3e} (want < 1e-3)          ->"
          f" {'OK' if ok_size else 'TOO LARGE'}")
    print(f"  secular / amplitude = {fs / fa:.2f} (want < 0.1)      ->"
          f" {'OK — oscillating, not draining' if ok_secular else 'DRAINING'}")
    if not (ok_order and ok_size and ok_secular):
        print("  -> lower --dt until all three pass. Until they do, a release "
              "run measures the integrator, not the physics.")
    return ok_order and ok_size and ok_secular


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="trefoil_t23_compact")
    ap.add_argument("--partner", default=None,
                    help="second species (default: a copy of --name)")
    ap.add_argument("--orient", default="id",
                    choices=("id", "x180", "x90", "mirror"))
    ap.add_argument("--sep", type=float, default=76.8,
                    help="initial pair separation (physical units)")
    ap.add_argument("--single", action="store_true",
                    help="one particle, no composition — the catalog->quench "
                         "seam alone, without the composition variable on top")
    ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--det-every", type=int, default=0,
                    help="re-identify knot type every N samples (0 = ends only)")
    ap.add_argument("--calibrate", action="store_true",
                    help="run the dt gate and stop")
    ap.add_argument("--calibrate-T", type=float, default=0.02)
    ap.add_argument("--c-l3", type=float, default=None,
                    help="ℒ₃ coupling (default: C from the catalog entry)")
    ap.add_argument("--regrid", type=int, default=0,
                    help="spectrally resample the loaded state to N=REGRID at "
                         "the same L before evolving — varies dx with the "
                         "physical field held fixed (the resolution sweep)")
    ap.add_argument("--resettle", type=int, default=0,
                    help="after --regrid, run N gradient-flow steps on the NEW "
                         "grid before quenching. Required for anything sensitive "
                         "to the pointwise potential term, which interpolation "
                         "inflates (see regrid's docstring)")
    ap.add_argument("--a0-iters", type=int, default=20000)
    ap.add_argument("--alpha-l3", type=float, default=4e-4,
                    help="ℒ₃ lock descent step. The engine default (4e-4) is set "
                         "against the contract tests' scale and DIVERGES at a "
                         "catalog state (C=400): measured NaN by step 4. Keep "
                         "alpha_l3*C_l3 <~ 4e-4 (see module docstring)")
    ap.add_argument("--x64", action="store_true", default=True)
    ap.add_argument("--no-x64", dest="x64", action="store_false",
                    help="float32: faster, but dH/H stops being trustworthy at "
                         "the 1e-3 level the gate is written against")
    ap.add_argument("--no-det", dest="det", action="store_false", default=True,
                    help="skip knot identification entirely (dynamics only) — "
                         "the ONLY way to run without pyknotid present")
    ap.add_argument("--out", default="out_quench_pair")
    a = ap.parse_args()

    if a.x64:
        jax.config.update("jax_enable_x64", True)

    # Before the A₀ solve, before the evolution, before anything that costs.
    if a.det and not a.calibrate:
        why = preflight_knot_id()
        if why is not None:
            print(f"PREFLIGHT FAILED — this process cannot identify knots, so the "
                  f"run would evolve and record no determinant. Nothing started.\n"
                  f"  {why}\n"
                  f"  fix: pip install 'jax-solitons[knots]'   "
                  f"(or pass --no-det to run the dynamics without topology)")
            return 90
        print("  preflight OK: T(2,3) identifies as det 3 in this process")

    u1, s1, w1, entry = cat_load(a.name)
    p = entry["relaxation"]["params"]
    N, L = p["N"], p["L"]
    dx = L / N
    ER_DX[0] = dx
    # Same constants compose_pair scores with, so the two drivers are comparable.
    # NOT quench.evolve's defaults (lam=50, kappa=0.5) — those are the contract
    # tests' numbers and are wrong by a factor of 20 for a catalog state.
    lam, kappa, C, U = 1000.0, 0.0008, p["C"], p["U"]
    eps_a, beta = 0.05, p["beta"]
    agrad = p["agrad"]
    ER.AGRAD = agrad                                    # before first jit trace
    kv = ER.EK.kvecs(N, L)
    C_l3 = C if a.c_l3 is None else a.c_l3

    # Refuse rather than guess. The product alpha_l3*C_l3 is what diverged
    # (4e-4 * 400 -> NaN in 4 steps); 4e-4 is the largest value validated.
    # Not on the --calibrate path: that measurement forces C_l3 = 0, so the lock
    # never fires and its step size cannot matter. Refusing there would block the
    # one run whose whole job is to establish a safe dt.
    LOCK_PRODUCT_MAX = 4e-4
    if not a.calibrate and a.alpha_l3 * C_l3 > LOCK_PRODUCT_MAX:
        print(f"REFUSING: alpha_l3*C_l3 = {a.alpha_l3 * C_l3:.2e} > "
              f"{LOCK_PRODUCT_MAX:.0e}, the largest validated lock step.\n"
              f"  At this product the ℒ₃ descent diverges (measured: NaN by step "
              f"4 at alpha_l3=4e-4, C_l3=400).\n"
              f"  fix: --alpha-l3 {LOCK_PRODUCT_MAX / C_l3:.1e}   "
              f"(or --c-l3 0 for the conservative sector / the dH/H gate)")
        return 2

    print(f"{'single' if a.single else 'pair'}: '{a.name}'"
          + (f" x '{a.partner}'" if a.partner else "")
          + f"  N={N} L={L} dx={dx:.3f}  lam={lam} kappa={kappa} C={C} "
            f"agrad={agrad}  x64={a.x64}")

    if a.single:
        u = u1
        sep0 = float("nan")
    else:
        u2 = None
        if a.partner:
            u2, _, _, entry2 = cat_load(a.partner)
            if entry2["relaxation"]["params"]["N"] != N:
                raise SystemExit("partner must be cached on the same grid")
        cells = int(round(a.sep / dx))
        u = compose_pair(u1, cells, orient=a.orient, u2=u2)
        sep0 = cells * dx

    if a.regrid:
        u = tuple(regrid(f, a.regrid) for f in u)
        N = a.regrid
        dx = L / N
        ER_DX[0] = dx
        kv = ER.EK.kvecs(N, L)
        print(f"  REGRID -> N={N} dx={dx:.4f} (same L={L}, same Fourier "
              f"coefficients; {'up' if a.regrid > p['N'] else 'DOWN'}sampled)")

    if a.resettle:
        # particle_catalog's engine-static caveat, finally enforced: a cached
        # state is stationary under the integrator that relaxed it AND on the grid
        # it was relaxed on. After a regrid neither holds, and the pointwise
        # potential is the term that shows it. Descend on the new grid until the
        # interpolation ringing is gone, THEN hand it to the real-time engine.
        zz = jnp.zeros((N, N, N))
        s_, w_ = zz, (zz, zz, zz)

        def report(uu):
            # quench.total_energy, NOT relax.energy_report: the latter wants
            # 6.5 GiB at N=256/x64 and OOMs on a 16 GB card, while this one is
            # what the release path already runs at that size for hours. Same
            # question, an instrument that fits.
            return Q.total_energy(uu[0] + 1j * uu[1], uu[2] + 1j * uu[3],
                                  uu[4], uu[5], uu[6], zz, zz, zz, kv, dx,
                                  lam, kappa, g=1.0, components=True)

        e0 = report(u)
        for _ in range(a.resettle):
            u, s_, w_ = ER.relax_iter(u, s_, w_, dx, lam, kappa, C, U, eps_a,
                                      p["alpha"], beta, 1.0, 0.0)
        e1 = report(u)
        print(f"  RESETTLE {a.resettle} relax steps on the new grid: "
              f"E {e0['total']:.1f} -> {e1['total']:.1f}  "
              f"(pot {e0['pot']:.1f} -> {e1['pot']:.1f}, "
              f"grad1 {e0['grad1']:.1f} -> {e1['grad1']:.1f})")

    t0 = time.time()
    fields = to_quench(u, dx, C, U, eps_a, beta, a.a0_iters)
    print(f"  A0 converged ({a.a0_iters} iters), E initialised to zero "
          f"[released from rest]   {time.time() - t0:.1f}s")

    if a.calibrate:
        ok = calibrate(fields, kv, dx, dt=a.dt, T=a.calibrate_T,
                       lam=lam, kappa=kappa, agrad=agrad)
        return 0 if ok else 1

    phi1, phi2, Ax, Ay, Az, Ex, Ey, Ez, s = fields
    p1c = np.asarray(phi1)
    if not a.single:
        m0, ncomp = pair_separation(phi1, dx)
        print(f"  composed at sep={sep0:.1f} ({ncomp} φ₁ components, "
              f"measured {m0:.1f})")
    if a.det:
        print("  dets at n=0:", knot_determinants(p1c, dx, L), flush=True)

    outp = Path(a.out)
    outp.mkdir(parents=True, exist_ok=True)
    every = max(1, a.steps // max(1, a.samples))
    traj, t0 = [], time.time()

    def observer(f1, f2, ax_, ay_, az_, ex_, ey_, ez_, s_):
        """Per-sample extras. Returned dict is merged into quench's own record
        (E_total / Q / helicity / gauss_res), so the two never disagree about
        which step they describe."""
        rec = {}
        if not a.single:
            sep, nc = pair_separation(f1, dx)
            rec["sep"], rec["ncomp"] = sep, nc
        rec["Efield"] = float(jnp.linalg.norm(ex_) ** 2 + jnp.linalg.norm(ey_) ** 2
                              + jnp.linalg.norm(ez_) ** 2)
        return rec

    # Driven in CHUNKS rather than one evolve() call. `evolve` accumulates its
    # samples and hands them back only at the end, so a single call to it is a
    # black box for the whole run -- on a multi-hour horizon that means no
    # progress, no partial manifest, and no way to tell slow from hung. Chaining
    # is exact: evolve is a pure state -> state map, so N chunks are bit-for-bit
    # one run of the same length (the same property drift() relies on).
    state = (phi1, phi2, Ax, Ay, Az, Ex, Ey, Ez, s)
    samples, done, h0 = [], 0, None
    while done < a.steps:
        k = min(every, a.steps - done)
        state, sm = Q.evolve(*state[:8], kv, steps=k, dt=a.dt, dx=dx, lam=lam,
                             kappa=kappa, C_l3=C_l3, eps_a=eps_a, agrad=agrad,
                             beta=beta, alpha_l3=a.alpha_l3, s=state[8],
                             sample_every=k, observer=observer)
        # sm holds the chunk's endpoints; keep n=0 only from the first chunk, and
        # renumber to global step count.
        for r in (sm if not samples else sm[1:]):
            r["n"] += done
            samples.append(r)
        done += k
        h0 = h0 if h0 is not None else samples[0]["E_total"]
        r = samples[-1]
        line = (f"  n{r['n']:7d} t={r['n'] * a.dt:8.3f}  E={r['E_total']:10.2f} "
                f"dH/H={(r['E_total'] - h0) / h0:+.2e}  Q={r['Q']:+.3f} "
                f"hel={r['helicity']:+.3e} gauss={r['gauss_res']:.3f}")
        if "sep" in r:
            line += f"  sep={r['sep']:7.2f} nc={r['ncomp']}"
        print(line, flush=True)
        # Partial manifest every chunk: a run killed at hour 2 keeps its
        # trajectory instead of losing it with the process.
        outp.mkdir(parents=True, exist_ok=True)
        (outp / "manifest.json").write_text(json.dumps(
            {"partial": True, "done": done, "steps": a.steps,
             "traj": samples, "wall_s": time.time() - t0}, indent=1, default=float))
    wall = time.time() - t0

    phi1_end = state[0]
    dets_end = knot_determinants(np.asarray(phi1_end), dx, L) if a.det else []
    if a.det:
        print("  dets at end:", dets_end)

    manifest = {
        "name": a.name, "partner": a.partner, "orient": a.orient,
        "single": a.single, "sep0": sep0, "dt": a.dt, "steps": a.steps,
        "C_l3": C_l3, "lam": lam, "kappa": kappa, "x64": a.x64,
        "E_init": "zero (released from rest)",
        "params": {"N": N, "L": L, "dx": dx, "agrad": agrad, "C": C, "U": U},
        # knot_determinants returns [(component_length, determinant), ...] — one
        # pair per skeleton component, NOT a flat list of ints.
        "dets_end": [[int(n_), int(det)] for n_, det in dets_end],
        "traj": samples, "wall_s": wall,
    }
    (outp / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float))

    # The verdict is deliberately thin. dH/H is NOT a pass/fail here: with the
    # lock on (C_l3 != 0) the φ₂ descent step makes the engine dissipative by
    # construction, so a falling H is the lock working, not the integrator
    # failing. Quote it, and let --calibrate be the place that judges.
    hN = samples[-1]["E_total"]
    print(f"\n  dH/H over the run: {(hN - h0) / h0:+.3e}"
          + ("  (C_l3 = 0 — this IS the conservation gate)" if C_l3 == 0 else
             f"  (C_l3 = {C_l3} — lock is dissipative by construction; "
             f"run --calibrate for the gate)"))
    st_ = energy_stats(samples)
    if st_:
        manifest["energy_stats"] = st_
        (outp / "manifest.json").write_text(
            json.dumps(manifest, indent=1, default=float))
        print(f"  energy, FULL window: amplitude={st_['full_amplitude']:.3e}  "
              f"secular={st_['full_secular']:.3e}  "
              f"(includes any start-up transient — context, not the verdict)")
        print(f"  energy, TAIL (last half, {st_['periods']:.1f} period(s)): "
              f"amplitude={st_['tail_amplitude']:.3e}  "
              f"secular={st_['tail_secular']:.3e}  "
              f"ratio={st_['secular_over_amplitude']:.2f}")
        if st_["periods"] < 3:
            print("    <-- TAIL TOO SHORT to separate drain from wobble: a line "
                  "through under ~3 cycles fits the oscillation. Raise --steps "
                  "before reading the secular number.")
        elif st_["secular_over_amplitude"] < 0.1:
            print("    -> OSCILLATING, NOT DRAINING: no secular component "
                  "resolvable above the wobble once the transient has decayed")
        else:
            print(f"    -> SECULAR COMPONENT SURVIVES INTO THE TAIL: H drifts at "
                  f"{st_['tail_slope_per_step']:+.3e} per step on top of the "
                  f"wobble. Compare full_secular: if that is much larger, this "
                  f"is a decaying transient, not a constant drain.")

    g0, gN = samples[0]["gauss_res"], samples[-1]["gauss_res"]
    print(f"  gauss residual {g0:.3f} -> {gN:.3f}"
          + ("  <-- GROWING: the transverse form is being asked to carry charge "
             "dynamics it cannot represent" if gN > g0 * 1.5 else ""))
    if not a.single:
        seps = [r["sep"] for r in samples if np.isfinite(r.get("sep", np.nan))]
        if len(seps) >= 2:
            verdict = ("SEPARATING" if seps[-1] > seps[0] + dx else
                       "APPROACHING" if seps[-1] < seps[0] - dx else
                       "FLAT (< dx)")
            print(f"  sep {seps[0]:.2f} -> {seps[-1]:.2f}  ({verdict})")
    print(f"  wall {wall:.1f}s -> {outp / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
