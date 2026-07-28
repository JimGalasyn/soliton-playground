#!/usr/bin/env python3
"""Does a Faddeev-Skyrme knot hold its knot type under REAL-TIME evolution?

UNSCORED DEMO (census protocol DRAFT). This is the gap nothing in the program has
covered. Everything establishing Faddeev knot stability came from RELAXATION:
`jax_solitons` has `runfns.faddeev_relax_then_id` as its one physics seam, the
retired fleet (`null-worldtube-private`, `simulations/engine_dogfood/`) was
relax -> kick -> bath -> POST-RELAX -> ID throughout, and jax-solitons'
`tests/test_acceptance_gates.py::test_gate_trefoil_q7_determinant_held` is an
EMPTY placeholder. Gradient flow descends, so by construction it cannot exhibit a
dynamical instability: "stable" so far means "is a minimizer", not "survives
real-time evolution". Those come apart, because a configuration can sit in a local
energy minimum and still be dynamically unstable once modes can exchange energy at
fixed total.

Three phases, cheap -> expensive, after the prior fleet's own ordering:

  1. SEED       Q_H and core-curve determinant on the analytic seed.
  2. RELAX-SURVIVAL (their cheap pre-filter). Deep arrested flow. Q_H lost in
     descent => no soliton to evolve, stop. Note this is NOT automatic even in
     Faddeev: stability_compare.py records that "bare L3 can UNWIND a low-Q knot
     while the L2 flux tube holds it open".
  3. REAL-TIME  constrained Verlet from the relaxed state with v = 0. The state is
     only near-converged, so residual forces excite the internal modes -- the same
     settling that dominated the GPE trefoil's early transfers -- and the question
     is whether that settling unties the knot. Tracks Q_H, H = E + KE, and the
     determinant at checkpoints.

THE COMPARISON THIS IS FOR. The GPE trefoil unties at 0.26 traversal periods of
its own core (155 xi long). If a Faddeev knot holds its determinant for even one
traversal period, it beats GPE by a factor of a few on the SAME clock, and the
model explanation for the two "trefoils" behaving oppositely is established. Note
what is NOT claimed: 50 periods is out of reach here (tau ~ 30-40 and dt ~ 5e-4
puts N = 50 at millions of steps), so gate 1's threshold is REPORTED, not met.

CLOCK CAVEAT, stated because the protocol requires naming it: c = 1 is the
model's natural unit here, NOT a measured wave speed. The GPE side measured its
sound speed (1.0446/1.0714). The Faddeev equivalent has not been measured, so
periods below carry that assumption. Measuring it is a separate experiment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
from jax_solitons.knots import (core_curves_from_n, curve_energy_scores,  # noqa: E402
                                identify_core_knot)
from jax_solitons.models import faddeev_model  # noqa: E402
from jax_solitons.models.faddeev import faddeev_energy_density  # noqa: E402
from jax_solitons.seeds import torus_knot_hopfion  # noqa: E402
from jax_solitons.steppers import (arrested_flow, kinetic_energy,  # noqa: E402
                                   verlet_evolve)
from jax_solitons.topology import hopf_charge  # noqa: E402
from soliton_playground.gpe_lab import (C_BLUE, C_GREEN, C_ORANGE,  # noqa: E402
                                        DARK_STYLE, characteristic_period)

C_FADDEEV = 1.0          # model natural unit, NOT measured -- see docstring


# jax_solitons.knots labels used to read "trefoil/baryon T(2,3)". Fixed upstream
# (branch knot-labels-drop-particle-sectors), but enforced here too: this repo's
# charter has exactly ONE rule -- no structure is ever identified with a Standard
# Model particle -- and it must hold whichever upstream version is installed.
_SECTOR_WORDS = ("baryon", "nucleon", "lepton", "meson", "hyperon")


def knot_label_only(carrier):
    """Strip any particle-sector name from an upstream knot label."""
    if not carrier:
        return carrier
    kept = [w for w in str(carrier).replace("/", " ").split()
            if w.lower() not in _SECTOR_WORDS]
    return " ".join(kept)


def id_knot(n, grid, c4, budget=240.0):
    """Trace the core and identify it. Returns (report, curves).

    pole="auto" is load-bearing: torus_knot_hopfion + arrested_flow leave the
    vacuum at +z, needing the -z sheet. knots.py records that the old hard
    pole=+1 default traced the entire +z-vacuum bulk instead -- millions of seed
    points and hour-long tracer hangs.
    """
    arr = np.asarray(n)
    # THREE 1D coordinate arrays, not one: trace_implicit_curve does
    # `for a in axes` and RGI(axes, ...). Passing a single array silently
    # iterates its scalars and dies with "array is 0-dimensional".
    ax1 = np.asarray(grid.axis(), float)
    axes = (ax1, ax1, ax1)
    try:
        curves = core_curves_from_n(arr[0], arr[1], arr[2], axes, pole="auto")
    except Exception as e:                      # tracer is the fragile step
        return dict(ok=False, error=f"trace: {type(e).__name__}: {e}"), []
    if not curves:
        return dict(ok=False, error="no core curve found"), []
    e_dens = np.asarray(faddeev_energy_density(n, grid, c4=c4))
    try:
        scores = curve_energy_scores(curves, e_dens, axes)
        info = identify_core_knot(curves, scores=scores, max_points=200)
    except Exception as e:
        return dict(ok=False, n_curves=len(curves),
                    lengths=[round(float(np.sum(np.linalg.norm(
                        np.diff(np.vstack([c, c[:1]]), axis=0), axis=1))), 2)
                        for c in curves],
                    error=f"id: {type(e).__name__}: {e}"), curves
    lengths = [round(float(np.sum(np.linalg.norm(
        np.diff(np.vstack([c, c[:1]]), axis=0), axis=1))), 2) for c in curves]
    return dict(ok=True, n_curves=len(curves), lengths=lengths,
                determinant=info.get("determinant"),
                knot=knot_label_only(info.get("carrier"))), curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--L", type=float, default=18.0)
    ap.add_argument("--c4", type=float, default=4.0)
    ap.add_argument("--relax-steps", type=int, default=4000)
    ap.add_argument("--relax-dt", type=float, default=2e-4)
    ap.add_argument("--dt", type=float, default=5e-4,
                    help="real-time step. The prior fleet's warning: the GAUGED "
                         "integrator needs <=5e-5 and a bare-tuned 7e-4 blew it "
                         "to NaN, which circulated as 'the full model is more "
                         "fragile'. This is the BARE model, so 5e-4 is in range, "
                         "but the energy trace is the check, not this comment.")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--checkpoints", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("outputs/faddeev_rt"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    grid = BoxGrid(N=args.N, L=args.L, dtype=jnp.float64)
    model = faddeev_model(c4=args.c4)
    Q_target = args.p * args.m

    # ---- phase 1: seed
    n = torus_knot_hopfion(grid, args.p, args.q, args.m)
    q_seed = float(hopf_charge(n, grid))
    e_seed = float(model.energy(n, grid))
    id_seed, _ = id_knot(n, grid, args.c4)
    print(f"SEED   T({args.p},{args.q}) m={args.m}: Q_H={q_seed:+.4f} "
          f"(target {Q_target})  E={e_seed:.2f}  {id_seed}", flush=True)

    # ---- phase 2: relax-survival
    t0 = time.time()
    n, hist = arrested_flow(model, n, grid, dt=args.relax_dt,
                            steps=args.relax_steps, log_every=max(1, args.relax_steps // 6))
    q_relax = float(hopf_charge(n, grid))
    e_relax = float(model.energy(n, grid))
    id_relax, curves = id_knot(n, grid, args.c4)
    print(f"RELAX  ({time.time()-t0:.0f}s, {args.relax_steps} steps): "
          f"Q_H={q_relax:+.4f}  E={e_seed:.2f}->{e_relax:.2f}  {id_relax}",
          flush=True)
    relax_survived = abs(abs(q_relax) - Q_target) < 0.15
    if not relax_survived:
        print("RELAX-SURVIVAL FAILED: Q_H lost in descent, no soliton to evolve")

    # clock from the relaxed core
    L_core = max(id_relax.get("lengths", [np.nan])) if id_relax.get("lengths") \
        else float("nan")
    tau = characteristic_period(L_core, c=C_FADDEEV) if L_core == L_core else float("nan")

    # ---- phase 3: real time
    v = jnp.zeros_like(n)
    every = max(1, args.steps // 200)
    ck_at = {int(round(f * args.steps)) for f in
             np.linspace(0, 1, args.checkpoints)}
    rows, ck = [], {}

    def observer(i, nn, vv):
        e = float(model.energy(nn, grid))
        k = float(kinetic_energy(vv, grid))
        q = float(hopf_charge(nn, grid))
        rows.append(dict(step=i, t=i * args.dt, E=e, KE=k, H=e + k, Q_H=q))
        return None

    t0 = time.time()
    n_rt, v_rt = n, v
    from jax_solitons.steppers.verlet import make_verlet_step
    step_fn = make_verlet_step(model, grid, dt=args.dt)
    for i in range(args.steps + 1):
        if i % every == 0:
            observer(i, n_rt, v_rt)
        if i in ck_at:
            rep, _ = id_knot(n_rt, grid, args.c4)
            ck[i] = dict(t=i * args.dt, **rep)
            print(f"  t={i*args.dt:8.3f} ({i:7d}): Q_H={rows[-1]['Q_H']:+.4f}  "
                  f"dH/H={(rows[-1]['H']-rows[0]['H'])/rows[0]['H']:+.2e}  {rep}",
                  flush=True)
        if i < args.steps:
            n_rt, v_rt = step_fn(n_rt, v_rt)
    rt_secs = time.time() - t0

    T = args.steps * args.dt
    H0, H1 = rows[0]["H"], rows[-1]["H"]
    dets = [ck[i].get("determinant") for i in sorted(ck)]
    det0 = dets[0]
    det_held = all(d == det0 for d in dets if d is not None) and det0 is not None
    q_all = [r["Q_H"] for r in rows]
    q_held = all(abs(abs(x) - Q_target) < 0.15 for x in q_all)

    verdict = ("KNOT HELD" if det_held and q_held
               else "Q_H HELD, KNOT CHANGED" if q_held
               else "Q_H LOST")

    summary = dict(
        status="UNSCORED DEMO (census protocol DRAFT)",
        preset="faddeev-skyrme", model=f"Faddeev-Skyrme (constrained Verlet), c4={args.c4}",
        object=f"T({args.p},{args.q}) hopfion m={args.m}",
        protecting_charge=f"Hopf charge Q_H = p*m = {Q_target} (pi_3(S^2) = Z)",
        gate="real-time persistence (fills jax-solitons' empty "
             "test_gate_trefoil_q7_determinant_held)",
        verdict=verdict,
        clock=dict(kind="traversal (tau = L_core / c)", L_core=L_core, tau=tau,
                   c=C_FADDEEV,
                   caveat="c is the model's natural unit, NOT measured; the GPE "
                          "side measured its sound speed and this side has not"),
        grid=dict(N=args.N, L=args.L, dx=grid.dx, dt=args.dt, steps=args.steps, T=T),
        seed=dict(Q_H=q_seed, E=e_seed, **id_seed),
        relax=dict(Q_H=q_relax, E=e_relax, survived=relax_survived,
                   steps=args.relax_steps, **id_relax),
        realtime=dict(T=T, periods=(T / tau if tau == tau else None),
                      H_initial=H0, H_final=H1, dH_over_H=(H1 - H0) / H0,
                      Q_H_min=min(q_all), Q_H_max=max(q_all),
                      Q_H_held=q_held, determinant_sequence=dets,
                      determinant_held=det_held, wall_seconds=rt_secs),
        checkpoints={str(k): ck[k] for k in sorted(ck)},
        series=rows,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figure
    plt.rcParams.update(DARK_STYLE)
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.26)
    ts = [r["t"] for r in rows]

    a = fig.add_subplot(gs[0, 0])
    a.plot(ts, [r["Q_H"] for r in rows], color=C_BLUE, lw=2)
    a.axhline(Q_target, color=C_GREEN, ls="--", lw=1.2, label=f"p·m = {Q_target}")
    a.axhline(-Q_target, color=C_GREEN, ls="--", lw=1.2)
    a.set_title("Hopf charge (the protecting charge)", fontsize=11)
    a.set_xlabel("t"); a.legend(frameon=False, fontsize=9)

    a = fig.add_subplot(gs[0, 1])
    a.plot(ts, [abs(r["H"] / H0 - 1.0) for r in rows], color=C_ORANGE, lw=2)
    a.set_yscale("log"); a.set_title("|dH/H| (energy + kinetic)", fontsize=11)
    a.set_xlabel("t")

    a = fig.add_subplot(gs[0, 2])
    a.plot(ts, [r["E"] for r in rows], color=C_BLUE, lw=2, label="E (potential)")
    a.plot(ts, [r["KE"] for r in rows], color=C_ORANGE, lw=2, label="KE")
    a.set_title("energy exchange (settling)", fontsize=11)
    a.set_xlabel("t"); a.legend(frameon=False, fontsize=9)

    a = fig.add_subplot(gs[1, 0], projection="3d")
    a.set_facecolor("black")
    for c in curves[:6]:
        a.plot(c[:, 0], c[:, 1], c[:, 2], color=C_BLUE, lw=1.6)
    a.set_title(f"relaxed core: det {id_relax.get('determinant')}", fontsize=10,
                color=C_ORANGE)
    a.set_axis_off()

    a = fig.add_subplot(gs[1, 1:]); a.axis("off")
    card = (f"UNSCORED DEMO — protocol DRAFT\n\n"
            f"T({args.p},{args.q}) hopfion m={args.m}   N={args.N} L={args.L:g} "
            f"dx={grid.dx:.3f}   dt={args.dt:g}\n\n"
            f"REAL-TIME VERDICT: {verdict}\n\n"
            f"  Q_H  seed {q_seed:+.4f} -> relax {q_relax:+.4f} -> "
            f"real-time [{min(q_all):+.4f}, {max(q_all):+.4f}]  (target "
            f"{Q_target})\n"
            f"  determinant  seed {id_seed.get('determinant')} -> relax "
            f"{id_relax.get('determinant')} -> {dets}\n"
            f"  dH/H over T={T:g}:  {(H1-H0)/H0:+.3e}\n\n"
            f"  clock: tau = L_core/c = {L_core:.1f}/{C_FADDEEV:g} = {tau:.1f}\n"
            f"  ran {T/tau if tau==tau else float('nan'):.2f} traversal periods\n"
            f"    (GPE trefoil untied at 0.26 of its own; gate 1's N=50 is out\n"
            f"     of reach here and is REPORTED, not met)\n"
            f"  c = {C_FADDEEV:g} is the model's natural unit, NOT measured\n\n"
            f"no external numbers were compared.")
    a.text(0, 0.98, card, va="top", fontsize=10.5, color="#DDDDDD",
           family="monospace", linespacing=1.55)

    fig.suptitle("FADDEEV-SKYRME KNOT, REAL TIME: does Q_H protect it "
                 "dynamically, or only under descent?", fontsize=13,
                 color="white", y=0.98)
    out = args.out / "faddeev_realtime.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")

    print(f"\nVERDICT: {verdict}")
    print(f"  Q_H held {q_held}  det {dets}  dH/H {(H1-H0)/H0:+.3e}  "
          f"periods {T/tau if tau==tau else float('nan'):.2f}")
    print(out)


if __name__ == "__main__":
    main()
