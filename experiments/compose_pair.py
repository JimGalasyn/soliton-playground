#!/usr/bin/env python3
"""Stage TWO catalog particles in the periodic box and ask: do they repel?

The first `compose()` of PARTICLES.md, specialized to the EHN two-scalar model
and a pair. Composition is exact-on-the-grid by construction:

  PLACE    — np.roll of the cached state (relaxed ON this grid) by integer
             cells: periodic translation, no resampling, no seams.
  COMPOSE  — product ansatz per U(1) component: φᵢ = √2 · φᵢᴬ · φᵢᴮ.
             In the vacuum |φᵢ| = 1/√2 so the product lands back ON the vacuum
             manifold (√2·(1/√2)(1/√2) = 1/√2); phases (hence windings, hence
             charges) ADD — the standard Abrikosov composition. Gauge field
             A = Aᴬ + Aᴮ, B = curl A recomputed; w = 0; A₀ rebuilt with a short
             C-ramp so the electric sector is not shocked.
  MEASURE  — two independent meters (outline §8):
             (a) overdamped dynamics: gradient flow IS force-following motion,
                 so d(sep)/dn > 0 = repulsion, < 0 = attraction. sep = distance
                 between the two largest φ₁-skeleton components' centroids.
             (b) rigid score at n=0: E_int(d) = E(pair at d) − 2·E(single),
                 same box, A₀ solved to convergence, fields frozen. Rigid =
                 overlap-repulsion upper bound (outline §11) — sign only.

Physics note: A₀ exchange is Yukawa-screened (M² = 2g²|φ₁|²), so any
long-range force here is axion/Goldstone-mediated through the global φ₂
sector. Sign is the experiment.

  python compose_pair.py --name trefoil_t23_compact --sep 76.8 --steps 8000
  python compose_pair.py --name trefoil_t23_compact --score-seps 60 70 80
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp

# Migrated 2026-08-01 out of null-worldtube-private, which is being retired. The
# three imports below used to reach into sibling directories through two
# sys.path.insert hacks, because the dependency graph spanned three directories
# -- the mistake that, per that repo's EXTRACTION_DECISIONS.md, cost two rented
# runs. All three are packages now, so the hacks are gone and each import says
# where the code actually lives.
from jax_solitons.ehn import relax as ER
from jax_solitons.vortex_topology import (
    _label_lines,
    knot_determinants,
    vortex_skeleton,
)
from soliton_playground.ehn_lab.particle_catalog import load as cat_load
from soliton_playground.provenance import code_provenance

PI = np.pi


def _roll(u, cells):
    """Periodic translation of the full state tuple along x by integer cells."""
    return tuple(jnp.roll(f, cells, axis=0) for f in u)


def _flip0(f, ax):
    """Exact grid reflection about coordinate 0 (index N/2): j = N - i mod N.
    np.flip alone reflects about -dx/2 (j = N-1-i); the extra roll fixes it."""
    return jnp.roll(jnp.flip(f, axis=ax), 1, axis=ax)


def orient_state(u, op):
    """Exact (no-resampling) orientation ops on a CENTERED state tuple.

    id     — aligned (identity)
    x180   — proper rotation 180° about x: (x,y,z)->(x,-y,-z). Antiparallel
             channel: B's ring circulation reverses relative to A. Q, Lk kept.
    x90    — proper rotation 90° about x: (x,y,z)->(x,-z,y); new(r)=old(R⁻¹r)
             with R⁻¹:(x,y,z)->(x,z,-y). Perpendicular channel (ring planes
             orthogonal). Q, Lk kept.
    mirror — improper z-reflection: new(x,y,z)=old(x,y,-z). The ANTIKNOT
             (EHN's antiknot: Q, Lk, electric charge all flip) — the p-p̄
             analog and the positive control for the method.
    Vector components (A, B_aux, w) transform with the rotation; scalars don't."""
    if op == "id":
        return u
    p1r, p1i, p2r, p2i, Ax, Ay, Az, Bx, By, Bz = u
    if op == "x180":
        S = lambda f: _flip0(_flip0(f, 1), 2)
        return (S(p1r), S(p1i), S(p2r), S(p2i),
                S(Ax), -S(Ay), -S(Az), S(Bx), -S(By), -S(Bz))
    if op == "x90":
        # scalar map new(r) = old(x, -z, y) = old(R·r) — i.e. rotation by R⁻¹,
        # so vectors transform with R⁻¹ too: (Vx,Vy,Vz) -> (Vx, Vz, -Vy).
        # (First cut used R for the vectors while the scalars used R⁻¹ — the
        # mismatched A blew g1 to ~190k; scalar-only tracer checks can't catch
        # a vector-sense bug, the ENERGY is the vector validator.)
        S = lambda f: _flip0(jnp.swapaxes(f, 1, 2), 2)
        return (S(p1r), S(p1i), S(p2r), S(p2i),
                S(Ax), S(Az), -S(Ay), S(Bx), S(Bz), -S(By))
    if op == "mirror":
        S = lambda f: _flip0(f, 2)
        return (S(p1r), S(p1i), S(p2r), S(p2i),
                S(Ax), S(Ay), -S(Az), S(Bx), S(By), -S(Bz))
    raise SystemExit(f"unknown orientation {op}")


def compose_pair(u1, sep_cells, orient="id", u2=None):
    """States A (u1) and B (u2, default = u1) at ±sep_cells/2 along x, product-
    ansatz composed; B carries the orientation op (applied centered, pre-roll).
    Mixed species (e.g. `trefoil_t23` × `unknot_bare`) require both cached on
    the SAME grid."""
    ua = _roll(u1, +sep_cells // 2)
    ub = _roll(orient_state(u2 if u2 is not None else u1, orient),
               -(sep_cells - sep_cells // 2))
    out = []
    for k in (0, 2):                                        # φ₁ then φ₂
        pa = ua[k] + 1j * ua[k + 1]
        pb = ub[k] + 1j * ub[k + 1]
        p = np.sqrt(2.0) * pa * pb
        out.extend([jnp.real(p), jnp.imag(p)])
    for k in (4, 5, 6):                                     # A linear
        out.append(ua[k] + ub[k])
    Bx, By, Bz = ER.curlA(out[4], out[5], out[6], ER_DX[0])
    out.extend([Bx, By, Bz])
    return tuple(out)


ER_DX = [0.8]                                               # set in main


def pair_separation(p1, dx):
    """Distance between the two largest φ₁-skeleton components (min-image x)."""
    P, T, C = vortex_skeleton(np.asarray(p1))
    if len(P) < 4:
        return float("nan"), 0
    lab, big, sizes = _label_lines(C, p1.shape, 20)
    if len(big) < 2:
        return float("nan"), len(big)
    N = p1.shape[0]
    cents = []
    for b in big[:2]:
        pts = P[lab == b]
        ref = pts[0]                                        # min-image about a ref point
        rel = (pts - ref + N / 2) % N - N / 2
        cents.append((ref + rel.mean(0)) % N)
    d = np.abs(cents[0] - cents[1])
    d = np.minimum(d, N - d)                                # min-image distance
    return float(np.linalg.norm(d) * dx), len(big)


def solve_A0(u, dx, C, U, eps_a, beta, iters=20000):
    """Converge the A₀ Gauss law with fields frozen (for the rigid score)."""
    a1 = u[0] ** 2 + u[1] ** 2
    a2 = u[2] ** 2 + u[3] ** 2
    rho = ER._rho(u, dx, eps_a) if hasattr(ER, "_rho") else None
    if rho is None:                                          # ehn_relax names it differently
        p2 = u[2] + 1j * u[3]
        ga = ER.axion_grad(p2, dx, eps_a)
        rho = sum(u[7 + i] * ga[i] for i in range(3))
    s = jnp.zeros_like(rho)
    for _ in range(iters):
        s = s + beta * (ER.lap(s, dx) - 2.0 * a1 * s + C * rho)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="trefoil_t23_compact")
    ap.add_argument("--sep", type=float, default=76.8, help="pair separation (physical)")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--cramp", type=int, default=1500)
    ap.add_argument("--score-seps", type=float, nargs="*", default=None,
                    help="rigid E_int(d) at these separations instead of dynamics")
    ap.add_argument("--orient", default="id", choices=["id", "x180", "x90", "mirror"],
                    help="orientation of copy B (exact grid op; mirror = antiknot channel)")
    ap.add_argument("--partner", default=None,
                    help="catalog name for particle B (default: same as --name); "
                         "e.g. trefoil_t23_twist1 for the n+p deuteron channel")
    ap.add_argument("--out", default="out_compose_pair")
    a = ap.parse_args()

    u1, s1, w1, entry = cat_load(a.name)
    p = entry["relaxation"]["params"]
    N, L = p["N"], p["L"]
    dx = L / N
    ER_DX[0] = dx
    lam, kappa, C, U = 1000.0, 0.0008, p["C"], p["U"]
    eps_a, alpha, beta, q1, q2 = 0.05, p["alpha"], p["beta"], 1.0, 0.0
    ER.AGRAD = p["agrad"]                                   # before first jit trace
    kv = ER.EK.kvecs(N, L)
    z = jnp.zeros((N, N, N))
    E1 = ER.energy_report(u1, s1, w1, dx, lam, kappa, C, U, eps_a, q1, q2)
    print(f"single '{a.name}': E={E1['total']:.1f} (mag={E1['mag']:.0f} el={E1['elec']:.0f})")
    u2 = None
    E2 = E1
    if a.partner:
        u2, s2, w2, entry2 = cat_load(a.partner)
        if entry2["relaxation"]["params"]["N"] != N:
            raise SystemExit("partner must be cached on the same grid")
        E2 = ER.energy_report(u2, s2, w2, dx, lam, kappa, C, U, eps_a, q1, q2)
        print(f"single '{a.partner}': E={E2['total']:.1f} (mag={E2['mag']:.0f} el={E2['elec']:.0f})")

    if a.score_seps:                                        # (b) rigid E_int(d)
        print(f"{'d':>7} {'E_pair':>9} {'E_int':>9}   (rigid = overlap-repulsion upper bound)")
        for d in a.score_seps:
            cells = int(round(d / dx))
            u = compose_pair(u1, cells, orient=a.orient, u2=u2)
            s = solve_A0(u, dx, C, U, eps_a, beta, iters=20000)
            E = ER.energy_report(u, s, (z, z, z), dx, lam, kappa, C, U, eps_a, q1, q2)
            print(f"{cells * dx:7.1f} {E['total']:9.1f} "
                  f"{E['total'] - E1['total'] - E2['total']:+9.1f}", flush=True)
        return

    # (a) overdamped dynamics
    cells = int(round(a.sep / dx))
    u = compose_pair(u1, cells, orient=a.orient, u2=u2)
    s, w = z, (z, z, z)
    outp = Path(a.out); outp.mkdir(parents=True, exist_ok=True)
    p1c = u[0] + 1j * u[1]
    sep0, ncomp = pair_separation(p1c, dx)
    print(f"composed pair at sep={cells * dx:.1f} ({ncomp} φ₁ components, measured sep={sep0:.1f})")
    print("dets at n=0:", knot_determinants(np.asarray(p1c), dx, L))
    t0 = time.time(); every = max(1, a.steps // a.samples); traj = []
    for n in range(a.steps + 1):
        Cn = C * min(1.0, n / a.cramp) if a.cramp > 0 else C
        if n % every == 0:
            p1c = u[0] + 1j * u[1]
            sep, ncomp = pair_separation(p1c, dx)
            E = ER.energy_report(u, s, w, dx, lam, kappa, Cn, U, eps_a, q1, q2)
            Q = float(ER.EK.skyrmion_number(p1c, u[2] + 1j * u[3], kv, dx))
            print(f"  n{n:6d}: sep={sep:7.2f} ncomp={ncomp} Q={Q:+.2f} E={E['total']:8.1f} "
                  f"(mag={E['mag']:.0f} el={E['elec']:.0f})", flush=True)
            traj.append({"n": n, "sep": sep, "ncomp": ncomp, "Q": Q, **E})
            (outp / "manifest.json").write_text(json.dumps(
                {"name": a.name, "sep0": cells * dx, "traj": traj,
                 "wall_s": time.time() - t0,
                 "code": code_provenance()}, indent=1))
        if n < a.steps:
            u, s, w = ER.relax_iter(u, s, w, dx, lam, kappa, Cn, U, eps_a,
                                    alpha, beta, q1, q2)
    p1c = u[0] + 1j * u[1]
    print("dets at end:", knot_determinants(np.asarray(p1c), dx, L))
    ER._save_field(outp / "field.npz", u, s, w, a.steps)
    # Endpoints of the FINITE SUBSET, not of the run -- see the note in
    # quench_pair.py. Named here so the printed range cannot be read as the run's.
    fin = [t for t in traj if np.isfinite(t["sep"])]
    dropped = len(traj) - len(fin)
    seps = [t["sep"] for t in fin]
    if dropped:
        print(f"NOTE: {dropped} of {len(traj)} samples were non-finite and are not "
              f"in the verdict below; it spans n={fin[0]['n']}..{fin[-1]['n']} of "
              f"{traj[-1]['n']}.")
    print(f"VERDICT: sep {seps[0]:.2f} -> {seps[-1]:.2f}  "
          f"({'REPEL' if seps[-1] > seps[0] + dx else 'ATTRACT' if seps[-1] < seps[0] - dx else 'FLAT (< dx drift)'})")


if __name__ == "__main__":
    main()
