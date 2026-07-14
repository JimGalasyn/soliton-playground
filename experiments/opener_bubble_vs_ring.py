#!/usr/bin/env python3
"""Census opener: bubble vs ring at matched energy — UNSCORED DEMO (protocol DRAFT).

The negative control and the protection demo in one run: a vortex ring (winding
W=1, topologically protected) and a plain spherical density bubble (W=0, no
protection) are prepared at the SAME GPE energy and evolved in real time. The
expected verdicts, stated before running:

    ring   -> propagates along +z, depletion volume ~constant, winding retained
    bubble -> fills in on ~R/c timescale, energy radiated as sound (the
              calorimeter's potential->kinetic conversion), no structure left

Everything is internal to the medium (dimensionless GPE, xi = c = 1). No number
here is compared to any measured physical constant.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jax_solitons.grid import BoxGrid  # noqa: E402
from jax_solitons.steppers.splitstep import make_splitstep  # noqa: E402
from jax_solitons.models.gpe import GPEKineticTerm, GPEPotentialTerm  # noqa: E402
from jax_solitons.models.nlkg import _ring_factor  # noqa: E402
from jax_solitons.event_graph import EventGraph, PDG_PRIVATE  # noqa: E402

G = 1.0
RING_R = 6.0
SEED_Z = -12.0          # both objects start here (comparable slice panels)
DEP_THRESH = 0.5        # |psi|^2 below this counts as depletion (core/void)


# ---------------------------------------------------------------- energetics
def make_energy(grid):
    kin, pot = GPEKineticTerm(), GPEPotentialTerm(g=G)

    @jax.jit
    def energy(psi):
        return kin(psi, grid), pot(psi, grid)

    return energy


def smooth(grid, psi, steps=30, dt=0.01):
    """Brief imaginary-time healing of an analytic seed (too short to shrink it)."""
    step = make_splitstep(grid, dt, g=G, imaginary_time=True)
    for _ in range(steps):
        psi = step(psi)
    return psi


# ---------------------------------------------------------------- seeds
def ring_seed(grid):
    """Vortex ring at (rho=RING_R, z=SEED_Z) with the two-term phase
        phase = atan2(z', rho - R) + atan2(z', rho + R)
    (the rho<0 half-plane image term). The single-atan2 form in
    nlkg._ring_factor leaves a pi phase sheet across the periodic z-boundary
    for rho > R (the dark bands seen in run 1); with the image term the
    boundary jump is 2*pi == 0 and the seed is wrap-clean. Amplitude: tanh
    core on the real ring only."""
    psi = (_ring_factor(grid, R=RING_R, xi=1.0,
                        center=(0.0, 0.0, SEED_Z), axis="z", sign=1)
           * _ring_factor(grid, R=RING_R, xi=1.0,
                          center=(0.0, 0.0, -SEED_Z), axis="z", sign=-1))
    return jnp.asarray(psi, dtype=jnp.complex128)


def bubble_seed(grid, Rb):
    X, Y, Z = grid.coords()
    r = jnp.sqrt(X**2 + Y**2 + (Z - SEED_Z) ** 2)
    f = 0.5 * (1.0 + jnp.tanh((r - Rb) / 1.0))
    return f.astype(jnp.complex128)


def match_bubble_radius(grid, energy, target, lo=2.0, hi=14.0, iters=18):
    """Bisect the bubble radius so its smoothed GPE energy matches the ring's."""
    def e_of(Rb):
        k, p = energy(smooth(grid, bubble_seed(grid, Rb)))
        return float(k + p)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if e_of(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- observables
def depletion(psi, grid):
    """Depletion metrics in the z<0 half-box: the measured object lives there
    (the ring's mirror anti-ring at z>0 is the periodicity partner, not the
    census entrant; the bubble is seeded in z<0 too)."""
    dens = np.asarray(jnp.abs(psi) ** 2)
    z = np.asarray(grid.coords()[2])
    mask = (dens < DEP_THRESH) & (z < 0)
    vol = float(mask.sum()) * grid.dx**3
    zc = float(z[mask].mean()) if mask.any() else float("nan")
    return vol, zc, float(dens.min())


def winding_xz(psi, grid, x_c, z_c, half=3.0):
    """Phase winding (units of 2*pi) around a square loop in the y=0 plane,
    centred at (x_c, z_c) — wrapped phase increments, exact on the lattice."""
    ph = np.angle(np.asarray(psi)[:, grid.N // 2, :])   # (x, z) plane at y ~ 0
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


# ---------------------------------------------------------------- run
def evolve(grid, psi, energy, T, dt, sample_dt, keep_slices_at):
    step = make_splitstep(grid, dt, g=G, imaginary_time=False)
    every = max(1, int(round(sample_dt / dt)))
    steps = int(round(T / dt))
    rows, slices = [], {}
    t0 = time.time()
    for i in range(steps + 1):
        t = i * dt
        if i % every == 0:
            k, p = energy(psi)
            vol, zc, dmin = depletion(psi, grid)
            rows.append(dict(t=t, E_kin=float(k), E_pot=float(p),
                             E_tot=float(k + p), V_dep=vol, z_dep=zc, n_min=dmin))
        for ts in keep_slices_at:
            if abs(t - ts) < 0.5 * dt:
                slices[ts] = np.asarray(jnp.abs(psi[:, grid.N // 2, :]) ** 2)
        if i < steps:
            psi = step(psi)
    rows[-1]["wall_s"] = time.time() - t0
    return psi, rows, slices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--L", type=float, default=64.0)
    ap.add_argument("--T", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--out", type=Path, default=Path("outputs/opener"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    grid = BoxGrid(N=args.N, L=args.L, dtype=jnp.float64)
    energy = make_energy(grid)
    tmid, tend = round(args.T / 2), args.T
    keep = (0.0, float(tmid), float(tend))

    # --- seeds at matched energy (bubble matched to the SINGLE measured ring,
    # i.e. half the ring+anti-ring pair energy)
    ring0 = smooth(grid, ring_seed(grid))
    kr, pr = energy(ring0)
    E_pair = float(kr + pr)
    E_ring = 0.5 * E_pair
    Rb = match_bubble_radius(grid, energy, E_ring)
    bub0 = smooth(grid, bubble_seed(grid, Rb))
    kb, pb = energy(bub0)
    E_bub = float(kb + pb)
    print(f"ring E={E_ring:.4f}   bubble R_b={Rb:.3f} E={E_bub:.4f} "
          f"(match {100 * abs(E_bub - E_ring) / E_ring:.2f}%)")

    w0 = winding_xz(ring0, grid, RING_R, SEED_Z)

    # --- evolve both
    ring1, ring_rows, ring_slices = evolve(grid, ring0, energy, args.T, args.dt,
                                           1.0, keep)
    bub1, bub_rows, bub_slices = evolve(grid, bub0, energy, args.T, args.dt,
                                        1.0, keep)

    # --- gates (informal; protocol is DRAFT)
    zc_end = ring_rows[-1]["z_dep"]
    w1 = winding_xz(ring1, grid, RING_R, zc_end) if np.isfinite(zc_end) else 0.0
    ring_drift = abs(ring_rows[-1]["E_tot"] / ring_rows[0]["E_tot"] - 1.0)
    bub_drift = abs(bub_rows[-1]["E_tot"] / bub_rows[0]["E_tot"] - 1.0)
    ring_vol_ratio = ring_rows[-1]["V_dep"] / ring_rows[0]["V_dep"]
    bub_vol_ratio = bub_rows[-1]["V_dep"] / bub_rows[0]["V_dep"]
    travel = zc_end - ring_rows[0]["z_dep"] if np.isfinite(zc_end) else 0.0

    # --- event graphs: the calorimeter closes both stories
    zoo = dict(ns="zoo", charge_keys=("E", "W"), receipt_pdg={"PHONON": 22})
    g_ring = EventGraph("ring", **zoo)
    pr_id = g_ring.add_particle(PDG_PRIVATE, 4, {"E": E_ring, "W": 1},
                                {"zoo.object": "vortex_ring", "zoo.R": RING_R})
    g_bub = EventGraph("bubble", **zoo)
    pb_id = g_bub.add_particle(PDG_PRIVATE, 4, {"E": E_bub, "W": 0},
                               {"zoo.object": "bubble", "zoo.R": round(Rb, 3)})
    rc_id = g_bub.add_particle(22, 1, {"E": E_bub}, {"zoo.receipt": "PHONON"})
    g_bub.add_vertex("DECAY", [pb_id], [rc_id])
    closure = g_bub.check_conservation()

    summary = dict(
        status="UNSCORED DEMO (census protocol DRAFT)",
        grid=dict(N=args.N, L=args.L, dt=args.dt, T=args.T),
        ring=dict(E=E_ring, E_pair=E_pair, winding_t0=w0, winding_T=w1,
                  vol_ratio=ring_vol_ratio, travel_z=travel,
                  ledger_drift=ring_drift,
                  verdict="SURVIVES" if ring_vol_ratio > 0.5 and abs(w1) > 0.5
                          else "DIED"),
        bubble=dict(E=E_bub, R_b=Rb, vol_ratio=bub_vol_ratio,
                    ledger_drift=bub_drift,
                    verdict="DECAYED" if bub_vol_ratio < 0.1 else "SURVIVES"),
        calorimeter_closure={str(k): v for k, v in closure.items()},
        series=dict(ring=ring_rows, bubble=bub_rows),
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(args.out / "slices.npz",
             **{f"ring_{t}": s for t, s in ring_slices.items()},
             **{f"bubble_{t}": s for t, s in bub_slices.items()},
             axis=np.asarray(grid.axis()))
    (args.out / "ring.hepmc3").write_text(g_ring.to_hepmc3())
    (args.out / "bubble.hepmc3").write_text(g_bub.to_hepmc3())
    print(json.dumps({k: summary[k] for k in ("ring", "bubble")}, indent=2))
    print(f"calorimeter closure (bubble DECAY vertex): {closure}")


if __name__ == "__main__":
    main()
