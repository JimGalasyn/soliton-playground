#!/usr/bin/env python3
"""Census gate 2 (ledger) for the trefoil — UNSCORED DEMO (protocol DRAFT).

Gate 2: "energy drift within the integrator's measured floor; every loss
accounted by the calorimeter (radiated sector), not the grid." Both clauses were
previously unmet — nobody had measured the floor, and check_conservation returned
{} because the charges fed to it were fabricated (rings carried NO energy and the
phonon receipt carried all of E0, so the balance was a tautology).

PHASE A — measure the floor, twice. A1 sweeps dt over T=4, BEFORE the first
reconnection (t~5), so the dynamics are smooth and the drift is a clean
integrator diagnostic rather than a mix of integrator error and violent events;
this is what shows whether the drift responds to dt at all. A2 sweeps dt over the
full production T, reconnections included, because that is the only floor the
production drift can fairly be judged against. The floor is where refining dt
stops helping: what remains is the spatial error at that N.

PHASE B — spend the budget. Full cascade, tracking the Nore-Abid-Brachet
partition (gpe_lab.energy_partition) every sample:

    E_i    incompressible flow  -> BOUND in the vortex lines
    E_c    compressible flow    -> SOUND, the radiated sector
    E_q    quantum pressure
    E_int  interaction

The event graph is then built from MEASURED sector energies, with NO fudge term:
one vertex, knot(E_tot at t=0) in, bound(E_i+E_q+E_int at T) + sound(E_c at T)
out, so check_conservation's residual is the energy no sector accounts for.

BE HONEST ABOUT WHAT THAT RESIDUAL IS. Since E_i + E_q + E_int + E_c is E_tot by
construction, the residual equals the ledger drift — it is a re-expression of it,
not an independent check, and presenting it as one would be dressing up a
tautology. The calorimeter earns its keep on two other things the drift alone
cannot tell you:

  1. WHERE the energy went — the sector transfers, dE_i vs dE_c.
  2. Whether the sound is RESOLVED. Energy piling up near k_max is on its way
     into the truncation, not into the medium; physical reconnection phonons sit
     at k ~ 1/xi. `E_c_highk_frac` measures this, and it is the clause that
     actually distinguishes "radiated sector" from "the grid".

DECLARED BEFORE RUNNING — gated:
  (a) production drift over the full T within 1.5x a floor measured over that
      SAME T (phase A2, reconnections included), in the SAME convention. Both
      matchings are load-bearing: an earlier version of this script compared a
      spectral drift (6.7e-4) against a forward-difference floor (2.0e-3) over a
      different T, and "passed" on the convention gap rather than the physics.
      The ledger's kinetic term uses forward differences and drifts ~10x more
      than the spectral one, so mixing them decides the gate by bookkeeping.
  (b) drift is dt-INDEPENDENT on smooth dynamics (phase A1): if refining dt does
      not reduce the drift, the drift is spatial truncation and the integrator is
      already at its floor. This is the clause that gives "the integrator's
      measured floor" content.
  (c) sound stays resolved: max E_c_highk_frac < 0.25.
  (d) calorimeter self-consistent: sum-rule residual < 1e-2 of E_tot.
REPORTED, not gated: sector transfers, spectral drift, kinetic_fd_minus_spectral.

Per-ring energy is NOT attributed: the Helmholtz split is global, so there is no
way to say which ring holds which joule. The bound sector is one aggregate
particle, and the rings carry their lengths as attributes only. Claiming a
per-ring energy would be inventing a number.
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
from jax_solitons.event_graph import EventGraph, PDG_PRIVATE  # noqa: E402
from soliton_playground.gpe_lab import (  # noqa: E402
    CHARGE_KNOT_UNPROTECTED, C_BLUE, C_GREEN, C_ORANGE, DARK_STYLE,
    depletion_metrics, energy_partition, evolve, make_energy, provenance,
    seed_gate, smooth, zoo_provenance)

from trefoil_cascade import milnor_trefoil_seed, topology_report  # noqa: E402


def phase_a(grid, energy, args, T_floor, dts, label):
    """Drift vs dt -> the floor. Records the drift in BOTH conventions, because
    the ledger's forward-difference kinetic energy and the calorimeter's spectral
    one differ by O(dx) and drift by ~10x different amounts: gating a spectral
    drift against an FD floor would pass or fail on the convention gap rather
    than on the physics."""
    psi0 = smooth(grid, milnor_trefoil_seed(grid, args.scale), steps=60)
    rows = []
    for dt in dts:
        def obs(t, psi):
            k, p = energy(psi)
            return dict(E_tot=float(k + p))
        psi_end, r, _ = evolve(grid, psi0, T=T_floor, dt=dt,
                               sample_dt=T_floor, observer=obs)
        drift = abs(r[-1]["E_tot"] / r[0]["E_tot"] - 1.0)
        q0 = energy_partition(grid, psi0)["E_tot"]
        q1 = energy_partition(grid, psi_end)["E_tot"]
        drift_sp = abs(q1 / q0 - 1.0)
        rows.append(dict(dt=dt, steps=int(round(T_floor / dt)),
                         drift=drift, drift_spectral=drift_sp))
        print(f"  [{label}] dt={dt:<8g} steps={rows[-1]['steps']:<7d} "
              f"drift_fd={drift:.3e}  drift_spectral={drift_sp:.3e}", flush=True)
    # local slopes in log-log: drift ~ dt^p
    slopes = []
    for a, b in zip(rows, rows[1:]):
        if a["drift"] > 0 and b["drift"] > 0:
            slopes.append(round(float(np.log(a["drift"] / b["drift"])
                                      / np.log(a["dt"] / b["dt"])), 2))
    return dict(T=T_floor, label=label, rows=rows, local_slopes=slopes,
                floor=min(r["drift"] for r in rows),
                floor_spectral=min(r["drift_spectral"] for r in rows))


def phase_b(grid, energy, args):
    """Full cascade with the calorimeter running; returns series + partitions."""
    psi0 = smooth(grid, milnor_trefoil_seed(grid, args.scale), steps=60)
    ok, gate = seed_gate(grid, psi0)
    print(f"  gate 0: {'PASS' if ok else 'FAIL'} {gate}", flush=True)
    if not ok:
        raise SystemExit("seed gate failed")

    checkpoints = tuple(round(f * args.T, 6) for f in (0, 0.25, 0.5, 0.75, 1.0))
    snapshots, parts = {}, []

    def observer(t, psi):
        k, p = energy(psi)
        m = depletion_metrics(psi, grid)
        q = energy_partition(grid, psi)
        parts.append(dict(t=t, **{kk: q[kk] for kk in
                                  ("E_i", "E_c", "E_q", "E_int", "E_tot",
                                   "E_c_highk_frac", "sum_rule_residual",
                                   "kinetic_fd_minus_spectral")}))
        if any(abs(t - c) < 0.5 * args.dt for c in checkpoints):
            snapshots[round(t, 3)] = np.asarray(psi)
        return dict(E_kin=float(k), E_pot=float(p), E_tot=float(k + p), **m)

    _, rows, _ = evolve(grid, psi0, T=args.T, dt=args.dt,
                        sample_dt=args.sample_dt, observer=observer,
                        keep_slices_at=())

    topo = {}
    for t in sorted(snapshots):
        _, rep = topology_report(snapshots[t], grid)
        topo[t] = rep
        print(f"  t={t:6.1f}: {rep['n_loops']} loop(s) {rep['lengths']}",
              flush=True)
    snapshots.clear()
    return gate, rows, parts, topo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--L", type=float, default=64.0)
    ap.add_argument("--scale", type=float, default=8.0)
    ap.add_argument("--T", type=float, default=80.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--sample-dt", type=float, default=1.0)
    ap.add_argument("--T-floor", type=float, default=4.0)
    ap.add_argument("--dts", type=float, nargs="*",
                    default=[0.02, 0.01, 0.005, 0.0025, 0.00125])
    ap.add_argument("--dts-full", type=float, nargs="*",
                    default=[0.02, 0.01, 0.005],
                    help="dt sweep over the FULL production T, so the gated "
                         "comparison is at matched T as well as matched "
                         "convention (this sweep includes the reconnections)")
    ap.add_argument("--skip-floor", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("outputs/gate2"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    grid = BoxGrid(N=args.N, L=args.L, dtype=jnp.float64)
    energy = make_energy(grid)

    print("PHASE A: integrator floor", flush=True)
    floor_rep = full_rep = None
    if not args.skip_floor:
        # A1: smooth, pre-reconnection -> is the drift dt-sensitive at all?
        floor_rep = phase_a(grid, energy, args, args.T_floor, args.dts,
                            "smooth")
        # A2: the FULL production T, reconnections included -> the floor the
        # production drift must actually be judged against
        full_rep = phase_a(grid, energy, args, args.T, args.dts_full, "full-T")

    print("\nPHASE B: cascade with calorimeter", flush=True)
    gate, rows, parts, topo = phase_b(grid, energy, args)

    p0, p1 = parts[0], parts[-1]
    E_tot0, E_tot1 = p0["E_tot"], p1["E_tot"]
    bound0 = p0["E_i"] + p0["E_q"] + p0["E_int"]
    bound1 = p1["E_i"] + p1["E_q"] + p1["E_int"]

    # ---- event graph from MEASURED sectors, with no absorbing term
    zoo = dict(ns="zoo", charge_keys=("E",), receipt_pdg={"PHONON": 22})
    g = EventGraph("gate2_ledger", **zoo)
    knot = g.add_particle(PDG_PRIVATE, 4, {"E": E_tot0},
                          {"zoo.object": "trefoil T(2,3)",
                           "zoo.scale": args.scale,
                           "zoo.E_i": p0["E_i"], "zoo.E_c": p0["E_c"],
                           **zoo_provenance(CHARGE_KNOT_UNPROTECTED)})
    t_last = max(topo)
    bound = g.add_particle(PDG_PRIVATE, 2, {"E": bound1},
                           {"zoo.object": "bound sector (rings)",
                            "zoo.n_loops": topo[t_last]["n_loops"],
                            "zoo.lengths": str(topo[t_last]["lengths"]),
                            "zoo.note": "aggregate: the Helmholtz split is "
                                        "global, so per-ring energy is not "
                                        "attributable"})
    sound = g.add_particle(22, 1, {"E": p1["E_c"]},
                           {"zoo.receipt": "PHONON",
                            "zoo.E_c_initial": p0["E_c"],
                            "zoo.E_c_radiated": p1["E_c"] - p0["E_c"]})
    vid = g.add_vertex("RECONNECT_CASCADE", [knot], [bound, sound])
    closure = g.check_conservation()

    resid = closure.get(vid, {}).get("E", 0.0)
    unaccounted = abs(resid) / abs(E_tot0)
    floor = floor_rep["floor"] if floor_rep else None

    # Gate in the LEDGER's own convention (forward differences), over the FULL
    # production T, against a floor measured over that same T. Both matchings
    # matter and an earlier version of this script got both wrong: it compared a
    # SPECTRAL drift (6.7e-4) against an FD floor (2.0e-3) measured over T=4 vs a
    # production T=80, and "passed" on the convention gap rather than the physics.
    fd0, fd1 = rows[0]["E_tot"], rows[-1]["E_tot"]
    drift_fd_full = abs(fd1 / fd0 - 1.0)
    floor_full = full_rep["floor"] if full_rep else None
    floor = floor_rep["floor"] if floor_rep else None      # smooth, reported
    at_floor = min(parts, key=lambda q: abs(q["t"] - args.T_floor))
    drift_at_floor = abs(at_floor["E_tot"] / E_tot0 - 1.0)

    # Independent of the drift: is the sound RESOLVED, or piling up at k_max?
    hi0 = parts[0]["E_c_highk_frac"]
    hi_max = max(q["E_c_highk_frac"] for q in parts)
    sound_resolved = hi_max < 0.25

    checks = dict(
        drift_within_floor=(floor_full is not None
                            and drift_fd_full <= 1.5 * floor_full),
        drift_is_dt_independent=(
            floor_rep is not None
            and max(r["drift"] for r in floor_rep["rows"])
            <= 1.05 * min(r["drift"] for r in floor_rep["rows"])),
        sound_stays_resolved=sound_resolved,
        calorimeter_self_consistent=(
            max(abs(q["sum_rule_residual"]) for q in parts) / abs(E_tot0)
            < 1e-2),
    )
    gate2 = ("UNDETERMINED (no floor)" if floor_full is None
             else "PASS" if all(checks.values()) else "FAIL")

    summary = dict(
        status="UNSCORED DEMO (census protocol DRAFT)",
        **provenance(CHARGE_KNOT_UNPROTECTED),
        object="trefoil T(2,3)", gate="2 (ledger)",
        gate2_verdict=gate2, checks=checks,
        declared=dict(
            gated=["production drift over full T within 1.5x the full-T floor, "
                   "both forward-difference (matched convention AND matched T)",
                   "drift is dt-independent on smooth dynamics (phase A1)",
                   "sound stays resolved: max high-k fraction of E_c < 0.25",
                   "calorimeter self-consistent: sum-rule residual < 1e-2 of "
                   "E_tot"],
            reported_not_gated=["sector transfers dE_i, dE_c, dE_q, dE_int",
                                "final drift over the full T",
                                "kinetic_fd_minus_spectral"],
            honest_caveat="bound(T) + sound(T) = E_i+E_q+E_int+E_c is E_tot(T) "
                          "BY CONSTRUCTION, so the closure residual is a "
                          "re-expression of the ledger drift, not an "
                          "independent check. The calorimeter's non-trivial "
                          "content is (a) WHERE the energy went, via the "
                          "sector transfers, and (b) whether the sound is at "
                          "resolved k, which the high-k fraction tests and the "
                          "drift alone cannot.",
            no_fudge="the vertex has no absorbing term, so the residual IS the "
                     "unaccounted (grid) energy"),
        grid=dict(N=args.N, L=args.L, dt=args.dt, T=args.T, scale=args.scale),
        seed_gate=gate,
        integrator_floor=floor_rep, integrator_floor_full_T=full_rep,
        closure=dict(E_tot_initial=E_tot0, E_tot_final=E_tot1,
                     bound_initial=bound0, bound_final=bound1,
                     sound_initial=p0["E_c"], sound_final=p1["E_c"],
                     residual=resid, unaccounted_fraction=unaccounted,
                     drift_at_T_floor_spectral=drift_at_floor,
                     drift_fd_full_T=drift_fd_full,
                     floor_full_T_fd=floor_full,
                     floor_smooth_fd=floor,
                     convention_note="gated quantity and floor are both "
                                     "forward-difference (the ledger's own "
                                     "convention) over the same T; the "
                                     "spectral drift is ~10x smaller and must "
                                     "not be mixed in",
                     E_c_highk_frac_initial=hi0,
                     E_c_highk_frac_max=hi_max, check_conservation={str(k): v for k, v
                                                      in closure.items()}),
        transfers=dict(dE_i=p1["E_i"] - p0["E_i"], dE_c=p1["E_c"] - p0["E_c"],
                       dE_q=p1["E_q"] - p0["E_q"],
                       dE_int=p1["E_int"] - p0["E_int"],
                       dE_tot=E_tot1 - E_tot0),
        calorimeter=dict(
            sum_rule_residual_max=max(abs(p["sum_rule_residual"])
                                      for p in parts),
            kinetic_fd_minus_spectral_t0=p0["kinetic_fd_minus_spectral"]),
        topology={str(t): topo[t] for t in topo},
        partition_series=parts, series=rows,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "gate2.hepmc3").write_text(g.to_hepmc3())

    # ---- figure
    plt.rcParams.update(DARK_STYLE)
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.24)
    ts = [p["t"] for p in parts]

    a = fig.add_subplot(gs[0, 0])
    a.plot(ts, [p["E_i"] for p in parts], color=C_BLUE, lw=2, label="E_i bound")
    a.plot(ts, [p["E_c"] for p in parts], color=C_ORANGE, lw=2,
           label="E_c sound")
    a.plot(ts, [p["E_q"] for p in parts], color=C_GREEN, lw=1.4, label="E_q")
    a.set_title("calorimeter sectors", fontsize=11); a.set_xlabel("t")
    a.legend(frameon=False, fontsize=9)

    a = fig.add_subplot(gs[0, 1])
    a.plot(ts, [p["E_i"] - parts[0]["E_i"] for p in parts], color=C_BLUE, lw=2,
           label="ΔE_i")
    a.plot(ts, [p["E_c"] - parts[0]["E_c"] for p in parts], color=C_ORANGE,
           lw=2, label="ΔE_c")
    a.axhline(0, color="#666666", lw=0.8)
    a.set_title("transfer: does bound loss become sound?", fontsize=11)
    a.set_xlabel("t"); a.legend(frameon=False, fontsize=9)

    a = fig.add_subplot(gs[0, 2])
    a.plot(ts, [abs(p["E_tot"] / parts[0]["E_tot"] - 1.0) for p in parts],
           color=C_BLUE, lw=2, label="|drift| (spectral total)")
    a.plot([r["t"] for r in rows],
           [abs(r["E_tot"] / rows[0]["E_tot"] - 1.0) for r in rows],
           color=C_ORANGE, lw=2, label="|drift| (FD ledger, GATED)")
    if floor_full is not None:
        a.axhline(floor_full, color=C_GREEN, ls="--", lw=1.4,
                  label=f"full-T floor (FD) {floor_full:.1e}")
    a.set_yscale("log")
    a.set_title("ledger drift vs floor", fontsize=11); a.set_xlabel("t")
    a.legend(frameon=False, fontsize=8)

    a2 = fig.add_subplot(gs[1, 0])
    a2.plot(ts, [p["E_c_highk_frac"] for p in parts], color=C_ORANGE, lw=2)
    a2.axhline(0.25, color=C_GREEN, ls="--", lw=1.2, label="gate 0.25")
    a2.set_title("sound above half-Nyquist\n(rising => heading for the grid)",
                 fontsize=10)
    a2.set_xlabel("t"); a2.legend(frameon=False, fontsize=8)

    a = fig.add_subplot(gs[1, 1]); a.axis("off")
    if floor_rep:
        ax2 = fig.add_axes([0.40, 0.10, 0.16, 0.30])
        ax2.set_facecolor("black")
        d = [r["dt"] for r in floor_rep["rows"]]
        v = [r["drift"] for r in floor_rep["rows"]]
        ax2.loglog(d, v, marker="o", color=C_ORANGE, lw=2)
        if full_rep:
            ax2.loglog([r["dt"] for r in full_rep["rows"]],
                       [r["drift"] for r in full_rep["rows"]],
                       marker="s", color=C_BLUE, lw=2)
        ax2.set_title(f"floor: drift vs dt\nT={floor_rep['T']:g} (o) / "
                      f"T={args.T:g} (s)", fontsize=8)
        ax2.set_xlabel("dt", fontsize=8); ax2.set_ylabel("|drift|", fontsize=8)
        ax2.tick_params(labelsize=7)

    a = fig.add_subplot(gs[1, 2]); a.axis("off")
    tr = summary["transfers"]
    card = (f"UNSCORED DEMO — protocol DRAFT\n\n"
            f"GATE 2 (ledger): {gate2}\n\n"
            f"  floor, smooth  T={args.T_floor:<4g}   : "
            f"{'%.3e' % floor if floor is not None else 'n/a'}   (FD)\n"
            f"  floor, full-T  T={args.T:<4g}   : "
            f"{'%.3e' % floor_full if floor_full is not None else 'n/a'}   (FD)\n"
            f"  local log-log slopes      : "
            f"{floor_rep['local_slopes'] if floor_rep else 'n/a'}\n\n"
            f"  E_tot(0)                  : {E_tot0:.4f}\n"
            f"  bound(T) + sound(T)       : {bound1 + p1['E_c']:.4f}\n"
            f"  closure residual          : {resid:+.4e}\n"
            f"  unaccounted, full T       : {unaccounted:.3e}\n"
            f"  GATED: FD drift over T    : {drift_fd_full:.3e}"
            f"   (matched convention + T)\n"
            f"  spectral drift over T      : {unaccounted:.3e}"
            f"   (~10x smaller; NOT mixed in)\n"
            f"  sound above half-Nyquist  : {hi0:.4f} -> max {hi_max:.4f}\n\n"
            + "".join(f"  {'PASS' if v else 'FAIL'}  {k}\n"
                      for k, v in checks.items()) + "\n"
            f"  transfers (reported, not gated)\n"
            f"    ΔE_i bound  {tr['dE_i']:+10.3f}\n"
            f"    ΔE_c sound  {tr['dE_c']:+10.3f}\n"
            f"    ΔE_q        {tr['dE_q']:+10.3f}\n"
            f"    ΔE_int      {tr['dE_int']:+10.3f}\n"
            f"    ΔE_tot      {tr['dE_tot']:+10.3f}\n\n"
            f"  calorimeter own sum-rule residual (max) "
            f"{summary['calorimeter']['sum_rule_residual_max']:.3e}\n"
            f"  ledger FD minus spectral kinetic, t=0  "
            f"{p0['kinetic_fd_minus_spectral']:+.3f}\n\n"
            f"no external numbers were compared.")
    a.text(0, 0.98, card, va="top", fontsize=10.5, color="#DDDDDD",
           family="monospace", linespacing=1.5)

    fig.suptitle("CENSUS GATE 2 — LEDGER: is the trefoil's energy loss "
                 "accounted to sound, or to the grid?", fontsize=13,
                 color="white", y=0.98)
    out = args.out / "gate2_ledger.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")

    print(f"\nGATE 2: {gate2}")
    print(f"  floor(full-T,FD) {floor_full}  gated FD drift {drift_fd_full:.3e}"
          f"  spectral drift {unaccounted:.3e}")
    print(f"  transfers {tr}")
    print(out)


if __name__ == "__main__":
    main()
