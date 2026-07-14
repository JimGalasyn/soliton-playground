#!/usr/bin/env python3
"""Census: trefoil vortex-knot decay cascade — UNSCORED DEMO (protocol DRAFT).

A trefoil vortex knot is seeded from the Milnor map (zero line of u^2 - v^3 on
the Milnor fibration coordinates — wrap-clean by construction: the far field is
vacuum with zero phase in every direction) and evolved in real time. Expected,
stated before running: the knot is METASTABLE — it unties via a small number of
reconnection events into unknotted, unlinked vortex rings, radiating sound at
each reconnection; the rings then persist (protected bin). The lineage is
measured, not narrated: vortex lines are traced as the closed components of
{Re psi = 0, Im psi = 0}, with loop count, lengths, and the pairwise Gauss
linking matrix at each checkpoint. No external numbers are compared.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jax_solitons.grid import BoxGrid  # noqa: E402
from jax_solitons.measure import curve_length, gauss_lk, resample_curve, \
    trace_curves  # noqa: E402
from jax_solitons.event_graph import EventGraph, PDG_PRIVATE  # noqa: E402
from soliton_playground.gpe_lab import (  # noqa: E402
    C_BLUE, C_GREEN, C_ORANGE, DARK_STYLE, depletion_metrics, evolve,
    make_energy, seed_gate, smooth)

LOOP_COLORS = (C_BLUE, C_ORANGE, C_GREEN, "#CC79A7", "#F0E442", "#0072B2")


def milnor_trefoil_seed(grid: BoxGrid, scale: float):
    """psi = 1 + w(r) * (u^2 - v^3 - 1) on Milnor coordinates of x/scale — the
    zero line is a trefoil of size O(scale). The raw Milnor field approaches
    vacuum only algebraically (a dipole-like phase tail that is NOT wrap-clean),
    so a smooth radial envelope w blends to exact vacuum outside r = 2.6*scale;
    the blend cannot create spurious zeros because |f - 1| < 1 there."""
    X, Y, Z = (np.asarray(c) / scale for c in grid.coords())
    r2 = X**2 + Y**2 + Z**2
    u = (r2 - 1.0 + 2.0j * Z) / (r2 + 1.0)
    v = 2.0 * (X + 1.0j * Y) / (r2 + 1.0)
    f = u**2 - v**3
    w = 0.5 * (1.0 - np.tanh((np.sqrt(r2) - 2.2) / 0.3))
    return jnp.asarray(1.0 + w * (f - 1.0), dtype=jnp.complex128)


def min_projected_crossings(pts, n_proj=60, n_pts=360, seed=7):
    """Knot-type proxy: minimum self-crossing count of the closed polyline over
    random planar projections. An unknot generically finds projections with
    <= 2 crossings; a trefoil is pinned at >= 3 in EVERY projection. This is an
    upper-bound sampler of the crossing number, not an exact invariant."""
    P = resample_curve(pts, n_pts)
    rng = np.random.default_rng(seed)
    seg_a = np.arange(n_pts)
    seg_b = (seg_a + 1) % n_pts
    ii, jj = np.triu_indices(n_pts, k=2)
    adj = (ii == 0) & (jj == n_pts - 1)
    ii, jj = ii[~adj], jj[~adj]
    best = None
    for _ in range(n_proj):
        u = rng.standard_normal(3)
        u /= np.linalg.norm(u)
        e1 = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(u, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        Q = P @ np.stack([e1, e2], 1)
        A, B = Q[seg_a], Q[seg_b]
        d = B - A
        cross2 = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        # segment pair (i, j): solve A_i + s d_i = A_j + t d_j
        cr = cross2(d[ii], d[jj])
        dA = Q[seg_a][jj] - Q[seg_a][ii]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = cross2(dA, d[jj]) / cr
            t = cross2(dA, d[ii]) / cr
        hits = ((np.abs(cr) > 1e-12) & (s > 0) & (s < 1) & (t > 0) & (t < 1))
        n_cross = int(hits.sum())
        best = n_cross if best is None else min(best, n_cross)
        if best <= 2:
            break
    return best


def topology_report(psi, grid: BoxGrid):
    """Trace vortex loops; return loop polylines + count/lengths/linking +
    a min-projected-crossings knot proxy for every loop longer than 30 xi."""
    arr = np.asarray(psi)
    loops = trace_curves(arr.real, arr.imag, grid, seed_tol=0.30, max_loops=8)
    lengths = [round(curve_length(p), 2) for p in loops]
    n = len(loops)
    lk = [[round(gauss_lk(resample_curve(loops[i], 400),
                          resample_curve(loops[j], 400)), 2)
           for j in range(n)] for i in range(n)]
    for i in range(n):
        lk[i][i] = 0.0
    def degenerate(p, k=40, frac_thresh=0.2):
        """A trace that (partially) doubled back on itself: a substantial
        fraction of points have a NON-adjacent partner at sub-core distance.
        Physical strands reconnect below ~xi, so sub-core parallel running
        marks a tracer artifact (measure.py's documented tangled-field gap),
        not a crumpled loop. Fraction-based: a median-based check misses
        partial doubling because most points' k-nearest are chain-adjacent."""
        from scipy.spatial import cKDTree
        m = len(p)
        d, idx = cKDTree(p).query(p, k=min(k, m))
        near = 0
        for i in range(m):
            for dd, j in zip(d[i], idx[i]):
                if 15 < abs(i - j) < m - 15 and dd < 0.5:
                    near += 1
                    break
        return near / m > frac_thresh

    crossings = []
    for i, p in enumerate(loops):
        if lengths[i] <= 30.0:
            crossings.append(0)
        elif degenerate(p):
            crossings.append(None)      # trace artifact: proxy inapplicable
        else:
            crossings.append(min_projected_crossings(p))
    return loops, dict(n_loops=n, lengths=lengths, linking=lk,
                       min_crossings=crossings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--L", type=float, default=64.0)
    ap.add_argument("--scale", type=float, default=8.0)
    ap.add_argument("--T", type=float, default=80.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--out", type=Path, default=Path("outputs/trefoil"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    grid = BoxGrid(N=args.N, L=args.L, dtype=jnp.float64)
    energy = make_energy(grid)

    psi0 = smooth(grid, milnor_trefoil_seed(grid, args.scale), steps=60)
    ok, gate = seed_gate(grid, psi0)
    print(f"seed gate: {'PASS' if ok else 'FAIL'} {gate}")
    if not ok:
        raise SystemExit("seed gate failed")
    k0, p0 = energy(psi0)
    E0 = float(k0 + p0)

    checkpoints = tuple(round(f * args.T, 6) for f in (0, 0.25, 0.5, 0.75, 1.0))
    snapshots = {}

    def observer(t, psi):
        k, p = energy(psi)
        m = depletion_metrics(psi, grid)
        if any(abs(t - c) < 0.5 * args.dt for c in checkpoints):
            snapshots[round(t, 3)] = np.asarray(psi)
        return dict(E_kin=float(k), E_pot=float(p), E_tot=float(k + p), **m)

    psi1, rows, slices = evolve(grid, psi0, T=args.T, dt=args.dt, sample_dt=1.0,
                                observer=observer, keep_slices_at=checkpoints)

    # topology at each checkpoint
    topo, loops_at = {}, {}
    for t in sorted(snapshots):
        loops, rep = topology_report(snapshots[t], grid)
        topo[t] = rep
        loops_at[t] = loops
        print(f"t={t:6.1f}: {rep['n_loops']} loop(s), lengths {rep['lengths']}, "
              f"min_crossings {rep['min_crossings']}, linking {rep['linking']}")
    np.savez(args.out / "loops.npz",
             **{f"t{t}_loop{i}": loop for t in loops_at
                for i, loop in enumerate(loops_at[t])})

    t_first, t_last = min(topo), max(topo)
    n0, n1 = topo[t_first]["n_loops"], topo[t_last]["n_loops"]
    drift = abs(rows[-1]["E_tot"] / rows[0]["E_tot"] - 1.0)
    # lifetime: first sample where blob count leaves the initial value
    n_blob0 = rows[0]["n_blobs"]
    t_life = next((r["t"] for r in rows if r["n_blobs"] != n_blob0), args.T)
    unlinked = all(abs(x) < 0.5 for row in topo[t_last]["linking"] for x in row)
    verdict = ("METASTABLE->RINGS" if n1 > n0 and unlinked
               else "SURVIVES" if n1 == n0 else "DECAYED")

    # event graph: trefoil -> RECONNECT cascade -> rings + phonon receipt
    zoo = dict(ns="zoo", charge_keys=("E",), receipt_pdg={"PHONON": 22})
    g = EventGraph("trefoil_cascade", **zoo)
    knot = g.add_particle(PDG_PRIVATE, 4, {"E": E0},
                          {"zoo.object": "trefoil T(2,3)",
                           "zoo.scale": args.scale})
    rings = [g.add_particle(PDG_PRIVATE, 2, {},
                            {"zoo.object": f"ring #{i}",
                             "zoo.length": topo[t_last]["lengths"][i]})
             for i in range(n1)]
    receipt = g.add_particle(22, 1, {"E": E0}, {"zoo.receipt": "PHONON"})
    g.add_vertex("RECONNECT", [knot], rings + [receipt],
                 attrs={"zoo.t_first_reconnection": t_life})
    closure = g.check_conservation()

    summary = dict(status="UNSCORED DEMO (census protocol DRAFT)",
                   grid=dict(N=args.N, L=args.L, dt=args.dt, T=args.T,
                             scale=args.scale),
                   seed_gate=gate, E0=E0, verdict=verdict,
                   lifetime_first_reconnection=t_life,
                   topology={str(t): topo[t] for t in topo},
                   ledger_drift=drift,
                   calorimeter_closure={str(k): v for k, v in closure.items()},
                   series=rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "cascade.hepmc3").write_text(g.to_hepmc3())

    # ---- figure: slices row, 3D loops row, metrics row
    plt.rcParams.update(DARK_STYLE)
    fig = plt.figure(figsize=(19, 12))
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.8], hspace=0.32,
                          wspace=0.22)
    ax_r = np.asarray(grid.axis())
    ext = [ax_r[0], ax_r[-1], ax_r[0], ax_r[-1]]

    for col, t in enumerate(checkpoints):
        a = fig.add_subplot(gs[0, col])
        a.imshow(slices[float(t)].T, origin="lower", extent=ext, cmap="magma",
                 vmin=0, vmax=1.3, aspect="equal")
        a.set_title(f"t = {t:g}", fontsize=11, color=C_BLUE)
        if col == 0:
            a.set_ylabel("z / ξ   (|ψ|², y=0)")
        a.tick_params(labelsize=7)

    for col, t in enumerate(sorted(loops_at)):
        a = fig.add_subplot(gs[1, col], projection="3d")
        a.set_facecolor("black")
        for i, loop in enumerate(loops_at[t]):
            a.plot(loop[:, 0], loop[:, 1], loop[:, 2],
                   color=LOOP_COLORS[i % len(LOOP_COLORS)], lw=1.6)
        rep = topo[t]
        a.set_title(f"{rep['n_loops']} loop(s), L={rep['lengths']}",
                    fontsize=9, color=C_ORANGE)
        lim = args.scale * 2.2
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_zlim(-lim, lim)
        a.set_axis_off()

    t_s = [r["t"] for r in rows]
    a = fig.add_subplot(gs[2, 0])
    a.plot(t_s, [r["V_dep"] for r in rows], color=C_BLUE, lw=2)
    a.set_title("depletion volume (ξ³)", fontsize=10); a.set_xlabel("t")

    a = fig.add_subplot(gs[2, 1])
    a.plot(t_s, [r["n_blobs"] for r in rows], color=C_ORANGE, lw=2)
    a.set_title("depletion blobs", fontsize=10); a.set_xlabel("t")

    a = fig.add_subplot(gs[2, 2])
    a.plot(t_s, [r["E_pot"] for r in rows], color=C_BLUE, lw=2, label="E_pot")
    a.plot(t_s, [r["E_tot"] for r in rows], color=C_BLUE, lw=1, ls="--",
           alpha=0.6, label="E_tot")
    a.set_title("ledger", fontsize=10); a.set_xlabel("t")
    a.legend(frameon=False, fontsize=9)

    a = fig.add_subplot(gs[2, 3:]); a.axis("off")
    lk_last = topo[t_last]["linking"]
    card = (f"UNSCORED DEMO — protocol DRAFT\n\n"
            f"seed gate: PASS  (shell {gate['shell_min_density']:.4f}, "
            f"wrap {gate['wrap_jump_max']:.1e})\n\n"
            f"TREFOIL T(2,3), scale {args.scale:g}ξ, E={E0:.1f}: {verdict}\n"
            f"  loops {n0} → {n1}   lengths {topo[t_last]['lengths']}\n"
            f"  min projected crossings "
            f"{' → '.join('n/a(degen.trace)' if c is None else str(c) for c in (topo[t]['min_crossings'][0] for t in sorted(topo)))}\n"
            f"    (3=knotted, ≤2=unknot; untying bracketed by the 0)\n"
            f"  final linking matrix {lk_last}\n"
            f"  first reconnection t ≈ {t_life:g}\n"
            f"  ledger drift {drift:.1e}\n\n"
            f"channel: reconnection cascade → rings + sound\n"
            f"no external numbers were compared.")
    a.text(0, 0.95, card, va="top", fontsize=10.5, color="#DDDDDD",
           linespacing=1.6)

    fig.suptitle("CENSUS — TREFOIL KNOT DECAY CASCADE: traced vortex-line "
                 "lineage", fontsize=14, color="white", y=0.99)
    out = args.out / "trefoil_cascade.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(json.dumps({k: summary[k] for k in
                      ("verdict", "lifetime_first_reconnection",
                       "ledger_drift")}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
