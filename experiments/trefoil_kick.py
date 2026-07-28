#!/usr/bin/env python3
"""Census gate 4 (kick test) for the trefoil — UNSCORED DEMO (protocol DRAFT).

Gate 4 reads "10% perturbation -> returns to the same basin". Taken literally it
does not apply here: the trefoil unties at t ~ 5 whatever you do, so there is no
basin to return to. For a METASTABLE entrant the basin IS the decay channel, so
the gate is declared as: a 10% kick must leave the channel and its topological
signature unchanged. What passes and what merely gets reported is fixed BEFORE
running, below.

Why this kick. The Milnor seed is highly symmetric, and symmetric initial data
can produce a non-generic reconnection sequence — the sharpest thing a kick can
tell us is whether 3 -> 3 -> 0 -> 2 -> 1 is physics or an artifact of that
symmetry. So the kick is asymmetric smooth complex noise (gpe_lab.kick_field):
density and phase together, since an amplitude-only kick injects no current and
would leave the velocity field a knot lives in untouched. It is windowed to the
knot by the Milnor seed's own radial blend, because gate 0 outranks gate 4 — an
unwindowed eps=0.10 kick leaves the boundary shell at 1 - n ~ 0.19 against
seed_gate's 0.02 tolerance and would invalidate the run before any physics.

A pure translation is deliberately NOT tested: the box is homogeneous and
periodic, so translating the knot changes nothing by symmetry.

DECLARED BEFORE RUNNING
  PASS requires, for every realization:
    (a) gate 0 re-passes after the kick   [else the kick is invalid, not failed]
    (b) verdict == METASTABLE->RINGS      [the channel is unchanged]
    (c) the knot unties: main-loop min_crossings >= 3 at t=0, and <= 2 at some
        later checkpoint
    (d) final loops mutually unlinked: |Lk_ij| < 0.5 for all i != j
  REPORTED, NOT GATED (measured here for the first time, no threshold declared):
    first-reconnection time and its spread across seeds, final loop count, loop
    lengths, ledger drift, and the energy the kick actually added.
  A shift in WHICH checkpoint first shows the 0 is reported as a channel-timing
  shift, not a failure. A change of verdict, or rings emerging linked, FAILS.

Control is the committed unperturbed N=128 run (outputs/trefoil/summary.json),
which already passed the resolution doubling; it is not re-run.
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
    depletion_metrics, evolve, kick_field, knot_envelope, make_energy,
    provenance, seed_gate, smooth, zoo_provenance)

from trefoil_cascade import (  # noqa: E402
    LOOP_COLORS, milnor_trefoil_seed, topology_report)

BASELINE = Path("outputs/trefoil/summary.json")


def run_one(grid, energy, args, kick_seed):
    """One realization; kick_seed=None runs unperturbed. Returns a result dict
    plus the traced loops at each checkpoint."""
    psi0 = smooth(grid, milnor_trefoil_seed(grid, args.scale), steps=60)
    k0, p0 = energy(psi0)
    E_clean = float(k0 + p0)

    if kick_seed is not None:
        # kick the HEALED state, not the analytic seed: imaginary-time smoothing
        # would partly relax the perturbation away and weaken the test
        psi0 = kick_field(grid, psi0, eps=args.eps,
                          envelope=knot_envelope(grid, args.scale),
                          seed=kick_seed)
    ok, gate = seed_gate(grid, psi0)
    k0, p0 = energy(psi0)
    E0 = float(k0 + p0)
    print(f"  gate 0: {'PASS' if ok else 'FAIL'}  shell "
          f"{gate['shell_min_density']:.5f}  wrap {gate['wrap_jump_max']:.2e}  "
          f"dE/E {(E0 - E_clean) / E_clean:+.4f}", flush=True)
    if not ok:
        return dict(kick_seed=kick_seed, gate0_pass=False, seed_gate=gate), {}

    checkpoints = tuple(round(f * args.T, 6) for f in (0, 0.25, 0.5, 0.75, 1.0))
    snapshots = {}

    def observer(t, psi):
        k, p = energy(psi)
        m = depletion_metrics(psi, grid)
        if any(abs(t - c) < 0.5 * args.dt for c in checkpoints):
            snapshots[round(t, 3)] = np.asarray(psi)
        return dict(E_kin=float(k), E_pot=float(p), E_tot=float(k + p), **m)

    _, rows, _ = evolve(grid, psi0, T=args.T, dt=args.dt, sample_dt=1.0,
                        observer=observer, keep_slices_at=())

    topo, loops_at = {}, {}
    for t in sorted(snapshots):
        loops, rep = topology_report(snapshots[t], grid)
        topo[t], loops_at[t] = rep, loops
        print(f"  t={t:6.1f}: {rep['n_loops']} loop(s), lengths "
              f"{rep['lengths']}, min_crossings {rep['min_crossings']}",
              flush=True)
    snapshots.clear()

    t_first, t_last = min(topo), max(topo)
    n0, n1 = topo[t_first]["n_loops"], topo[t_last]["n_loops"]
    unlinked = all(abs(x) < 0.5 for row in topo[t_last]["linking"] for x in row)
    verdict = ("METASTABLE->RINGS" if n1 > n0 and unlinked
               else "SURVIVES" if n1 == n0 else "DECAYED")
    n_blob0 = rows[0]["n_blobs"]
    t_life = next((r["t"] for r in rows if r["n_blobs"] != n_blob0), args.T)

    # main-loop (longest) crossing sequence; None = tracer artifact, not a value
    main_cross = [topo[t]["min_crossings"][0] for t in sorted(topo)]
    knotted_at_start = main_cross[0] is not None and main_cross[0] >= 3
    untied = any(c is not None and c <= 2 for c in main_cross[1:])

    return dict(
        kick_seed=kick_seed, gate0_pass=True, seed_gate=gate,
        E_clean=E_clean, E0=E0, dE_over_E=(E0 - E_clean) / E_clean,
        verdict=verdict, lifetime_first_reconnection=t_life,
        n_loops_seq=[topo[t]["n_loops"] for t in sorted(topo)],
        main_crossing_seq=main_cross, knotted_at_start=knotted_at_start,
        untied=untied, unlinked_final=unlinked,
        lengths_final=topo[t_last]["lengths"],
        ledger_drift=abs(rows[-1]["E_tot"] / rows[0]["E_tot"] - 1.0),
        topology={str(t): topo[t] for t in topo}, series=rows,
    ), loops_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128,
                    help="ensemble resolution; note run_one holds all five "
                         "checkpoint fields in RAM, fine at 128 (168 MB), use "
                         "trefoil_cascade's disk path for 256")
    ap.add_argument("--L", type=float, default=64.0)
    ap.add_argument("--scale", type=float, default=8.0)
    ap.add_argument("--T", type=float, default=80.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--out", type=Path, default=Path("outputs/trefoil_kick"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not BASELINE.exists():
        raise SystemExit(f"control missing: {BASELINE} — run trefoil_cascade.py "
                         "at N=128 first (the kick test is a comparison)")
    base = json.loads(BASELINE.read_text())
    base_cross = [base["topology"][k]["min_crossings"][0]
                  for k in sorted(base["topology"], key=float)]
    base_loops = [base["topology"][k]["n_loops"]
                  for k in sorted(base["topology"], key=float)]
    print(f"control (committed, N={base['grid']['N']}): {base['verdict']}, "
          f"loops {base_loops}, main crossings {base_cross}, "
          f"t_reconnect {base['lifetime_first_reconnection']}", flush=True)

    grid = BoxGrid(N=args.N, L=args.L, dtype=jnp.float64)
    energy = make_energy(grid)

    results, loops_by_seed = [], {}
    for s in args.seeds:
        print(f"\n=== kick seed {s} (eps={args.eps}) ===", flush=True)
        r, loops = run_one(grid, energy, args, s)
        results.append(r)
        loops_by_seed[s] = loops

    # ---- the declared gate
    valid = [r for r in results if r["gate0_pass"]]
    checks = dict(
        all_gate0=all(r["gate0_pass"] for r in results),
        all_channel=bool(valid) and all(r["verdict"] == "METASTABLE->RINGS"
                                        for r in valid),
        all_knotted_at_start=bool(valid) and all(r["knotted_at_start"]
                                                 for r in valid),
        all_untied=bool(valid) and all(r["untied"] for r in valid),
        all_unlinked=bool(valid) and all(r["unlinked_final"] for r in valid),
    )
    gate4 = "PASS" if all(checks.values()) else "FAIL"
    t_rec = [r["lifetime_first_reconnection"] for r in valid]
    timing_shift = any(r["main_crossing_seq"] != base_cross for r in valid)

    summary = dict(
        status="UNSCORED DEMO (census protocol DRAFT)",
        **provenance(CHARGE_KNOT_UNPROTECTED),
        object="trefoil T(2,3)", gate="4 (kick test)",
        gate4_verdict=gate4, checks=checks,
        declared=dict(
            basin="decay channel, not the knot: a METASTABLE entrant has no "
                  "basin to return to",
            pass_requires=["gate 0 re-passes", "verdict METASTABLE->RINGS",
                           "main crossings >=3 at t=0 and <=2 later",
                           "final loops unlinked"],
            reported_not_gated=["first-reconnection time + spread",
                                "final loop count", "loop lengths",
                                "ledger drift", "energy added by the kick"]),
        kick=dict(eps=args.eps, kind="complex smooth noise (density+phase), "
                                     "windowed by the Milnor radial blend",
                  seeds=args.seeds),
        grid=dict(N=args.N, L=args.L, dt=args.dt, T=args.T, scale=args.scale),
        control=dict(source=str(BASELINE), N=base["grid"]["N"],
                     verdict=base["verdict"], n_loops_seq=base_loops,
                     main_crossing_seq=base_cross,
                     lifetime_first_reconnection=base[
                         "lifetime_first_reconnection"],
                     ledger_drift=base["ledger_drift"]),
        reported=dict(
            t_reconnect=t_rec,
            t_reconnect_mean=float(np.mean(t_rec)) if t_rec else None,
            t_reconnect_spread=float(np.ptp(t_rec)) if t_rec else None,
            channel_timing_shift=timing_shift,
            dE_over_E=[r.get("dE_over_E") for r in valid],
            ledger_drift=[r["ledger_drift"] for r in valid]),
        realizations=results,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    # persist the traced curves: the figure's 3D panels are the only consumer,
    # and without these a cosmetic replot costs a full re-run of the ensemble
    np.savez(args.out / "loops.npz",
             **{f"s{s}_t{t}_loop{i}": loop
                for s, at in loops_by_seed.items() for t in at
                for i, loop in enumerate(at[t])})

    # ---- event graph: one KICK vertex per realization, same channel out
    zoo = dict(ns="zoo", charge_keys=("E",), receipt_pdg={"PHONON": 22})
    g = EventGraph("trefoil_kick", **zoo)
    for r in valid:
        knot = g.add_particle(PDG_PRIVATE, 4, {"E": r["E0"]},
                              {"zoo.object": "trefoil T(2,3) kicked",
                               "zoo.kick_eps": args.eps,
                               "zoo.kick_seed": r["kick_seed"],
                               **zoo_provenance(CHARGE_KNOT_UNPROTECTED)})
        rings = [g.add_particle(PDG_PRIVATE, 2, {},
                                {"zoo.object": f"ring #{i}",
                                 "zoo.length": L})
                 for i, L in enumerate(r["lengths_final"])]
        receipt = g.add_particle(22, 1, {"E": r["E0"]},
                                 {"zoo.receipt": "PHONON"})
        g.add_vertex("RECONNECT", [knot], rings + [receipt],
                     attrs={"zoo.t_first_reconnection":
                            r["lifetime_first_reconnection"]})
    (args.out / "kick.hepmc3").write_text(g.to_hepmc3())

    # ---- figure
    plt.rcParams.update(DARK_STYLE)
    n_col = max(len(valid), 1)
    fig = plt.figure(figsize=(4.6 * n_col + 5.0, 9.5))
    gs = fig.add_gridspec(2, n_col + 1, height_ratios=[1.15, 1],
                          hspace=0.28, wspace=0.22)

    for col, r in enumerate(valid):
        loops = loops_by_seed[r["kick_seed"]]
        t_last = max(loops)
        a = fig.add_subplot(gs[0, col], projection="3d")
        a.set_facecolor("black")
        for i, loop in enumerate(loops[t_last]):
            a.plot(loop[:, 0], loop[:, 1], loop[:, 2],
                   color=LOOP_COLORS[i % len(LOOP_COLORS)], lw=1.6)
        a.set_title(f"kick seed {r['kick_seed']}  t={t_last:g}\n"
                    f"{r['n_loops_seq'][0]} → {r['n_loops_seq'][-1]} loops",
                    fontsize=10, color=C_ORANGE)
        lim = args.scale * 2.2
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_zlim(-lim, lim)
        a.set_axis_off()

    # Realizations that take the same pathway land on IDENTICAL integer
    # sequences, so traces coincide exactly. That degeneracy IS the result (the
    # cascade has a small number of discrete routes), but a single opaque line
    # would hide every seed but the last drawn — so markers, widths and dashes
    # all differ, letting coincident curves stay individually readable.
    MK = ("o", "s", "^", "D", "v", "P")
    ts_b = [float(k) for k in sorted(base["topology"], key=float)]

    def nums(seq):
        return [np.nan if c is None else c for c in seq]

    def overlay(ax, key, base_seq):
        for i, r in enumerate(valid):
            ts = [float(t) for t in sorted(r["topology"], key=float)]
            ax.plot(ts, nums(r[key]), marker=MK[i % len(MK)], ms=9 - 1.6 * i,
                    lw=4.0 - 0.9 * i, alpha=0.85, mfc="none", mew=1.8,
                    label=f"seed {r['kick_seed']}")
        ax.plot(ts_b, nums(base_seq), marker="x", ms=7, lw=1.4, color="white",
                ls=":", label="control")

    a = fig.add_subplot(gs[1, 0])
    overlay(a, "n_loops_seq", base_loops)
    a.set_title("loop count", fontsize=10); a.set_xlabel("t")
    a.legend(frameon=False, fontsize=8)

    a = fig.add_subplot(gs[1, 1])
    overlay(a, "main_crossing_seq", base_cross)
    a.axhline(2.5, color=C_GREEN, lw=1, ls=":")
    a.set_title("main-loop min crossings\n(≥3 knotted, ≤2 unknot)", fontsize=10)
    a.set_xlabel("t")

    a = fig.add_subplot(gs[1, 2:] if n_col >= 2 else gs[1, 1:]); a.axis("off")
    lines = [f"UNSCORED DEMO — protocol DRAFT",
             f"", f"GATE 4 (kick test): {gate4}",
             f"  kick: eps={args.eps:g} complex smooth noise, windowed to knot",
             f"  basin declared as the DECAY CHANNEL (metastable: no basin",
             f"  to return to)", f""]
    for k, v in checks.items():
        lines.append(f"  {'✓' if v else '✗'} {k}")
    lines += ["",
              f"control  : {base['verdict']}, loops {base_loops},",
              f"           crossings {base_cross}, t_rec "
              f"{base['lifetime_first_reconnection']:g}"]
    for r in valid:
        lines.append(f"seed {r['kick_seed']}   : {r['verdict']}, loops "
                     f"{r['n_loops_seq']},")
        lines.append(f"           crossings {r['main_crossing_seq']}, t_rec "
                     f"{r['lifetime_first_reconnection']:g}, "
                     f"dE/E {r['dE_over_E']:+.3f}")
    if t_rec:
        lines += ["", f"t_reconnect spread {np.ptp(t_rec):g} "
                      f"(mean {np.mean(t_rec):.2f}) — reported, not gated"]
    lines += ["", "no external numbers were compared."]
    a.text(0, 0.98, "\n".join(lines), va="top", fontsize=10.5, color="#DDDDDD",
           family="monospace", linespacing=1.55)

    fig.suptitle("CENSUS GATE 4 — TREFOIL KICK TEST: is the untying cascade "
                 "robust to a 10% asymmetric kick?", fontsize=13,
                 color="white", y=0.99)
    out = args.out / "trefoil_kick.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")

    print(f"\nGATE 4: {gate4}   checks {checks}")
    print(f"t_reconnect {t_rec} (control "
          f"{base['lifetime_first_reconnection']}), timing_shift {timing_shift}")
    print(out)


if __name__ == "__main__":
    main()
