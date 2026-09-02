#!/usr/bin/env python3
"""Does a Faddeev-Skyrme knot hold its knot type under REAL-TIME evolution?

UNSCORED DEMO (census protocol DRAFT). The gap this targets: every result
establishing Faddeev knot stability came from RELAXATION. `runfns.
faddeev_relax_then_id` is jax-solitons' one physics seam, the retired fleet
(`null-worldtube-private`, `simulations/engine_dogfood/`) was relax -> kick ->
bath -> post-relax -> ID throughout, and jax-solitons'
`tests/test_acceptance_gates.py::test_gate_trefoil_q7_determinant_held` is an
EMPTY placeholder. Gradient flow descends, so it cannot exhibit a dynamical
instability by construction: "stable" has meant "is a minimizer", not "survives
real-time evolution".

ON NOT RE-SEARCHING PARAMETER SPACE. The standard box (SB-1) and the particle
compendium exist precisely so configurations are looked up, not rediscovered.
They are used here where they reach, and they do NOT reach this run: every one of
the ten compendium entries
(`engine_dogfood/particles/*/entry.json`) is `model: ehn-two-scalar`, the GAUGED
theory. The June hopfion-era bare-Faddeev configurations were never locked in --
searched for in null-worldtube-private, nwt-analysis, null-worldtube,
nwt-substrate, nwt-audit and ClaudeSessionShare, and only their EHN
REPRODUCTIONS survive ("June hopfion-era cinquefoil REPRODUCED in EHN"). So the
geometry below is INFERRED from published ratios rather than looked up, and is
labelled as such everywhere it appears. If this run holds, its configuration
should be locked into the compendium as the first bare-Faddeev entry, which is
the only way the gap closes.

FOUR THINGS THIS VERSION FIXES, all defects in the version before it, three
confirmed by measurement rather than argued:

1. CP^1 SPINOR FRAME for relaxation. Relaxing the S^2 n-field stalled hard --
   |proj grad|/dof = 3.72e-01 against an energy density of ~0.027/dof, with
   arrested_flow breaking out early ("converged/stalled: no descending step
   exists above dt_min") and the energy pinned at 20904.71 from 6k to 100k
   steps. Evolving from a NON-critical state tests residual-force settling, not
   a minimizer. jax-solitons documents the cure in its own README -- "CP^1
   spinor frame for deep relaxation" -- and torus_knot_hopfion_cp1 is "ready to
   relax in the CONVERGENT spinor frame". Relax in CP^1, hand n(Z) to real time.

2. POST-RELAX BEFORE EVERY ID -- never identify topology mid-bath. The
   discipline the retired program wrote down after its own false result:
   `nwt-audit` PREREG, "relax-then-ID; never identify topology mid-bath
   (correction-cinquefoil-decays-were-artifacts)". The previous version traced
   straight out of the bath and produced a spurious det 3 -> 1 transition plus
   three tracer failures in six checkpoints. During dynamics the core preimage
   is smeared; the tracer splits it or loses it.

   What this measures, plainly: post-relax-then-ID answers "is the state still
   in the KNOTTED BASIN", not "is it instantaneously knotted". That is the right
   question for a census -- same basin logic as gate 4 -- but it is a different
   question, so the raw mid-bath ID is recorded alongside.

3. GEOMETRY at the validated ratios. torus_knot_hopfion's defaults give
   R/core = 3.57 REGARDLESS of N or L (R=0.2L, b=0.4R, w=0.7b); the hold regime
   is R/core = 11, core/dx ~ 2.56, R/L ~ 0.29 (I1_PHASE2_SCALE_pilot_P.md 3B),
   and at R~3 that note records "the wrapped lock cannot hold even the
   geometry". Those ratios pin N ~ 96 -- the N they used. Corroborated by EHN
   finding 7 ("seed radius must be moderate, R ~ 0.25 L").
   TRANSFER RISK: that regime is the GAUGED model, and stability_compare.py says
   (now at experiments/reference/stability_compare.py, migrated 2026-08-01)
   "bare L3 can UNWIND a low-Q knot while the L2 flux tube holds it open". Bare
   Faddeev may not hold at ANY geometry, so a failure here is NOT evidence
   against Q_H protection -- it may only mean the gauge sector is what protects.

4. CHECKPOINT FIELDS SAVED. Re-analysis must never cost another run. The
   previous version discarded them, so fixing the tracer discipline required
   repeating 2.25 GPU-hours.

Q_H is measured on the raw field throughout: a field integral, not a tracer
reading, so it is immune to the mid-bath artifact and needs no post-relax.

CLOCK CAVEAT (the protocol requires naming it): c = 1 is the model's natural
unit, NOT a measured wave speed. The GPE side measured its sound speed
(1.0446/1.0714); this side has not.
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
from jax_solitons.models.faddeev import (faddeev_cp1_model,  # noqa: E402
                                         faddeev_energy_density, n_from_state)
from jax_solitons.seeds import torus_knot_hopfion_cp1  # noqa: E402
from jax_solitons.steppers import arrested_flow, kinetic_energy  # noqa: E402
from jax_solitons.steppers.verlet import make_verlet_step  # noqa: E402
from jax_solitons.topology import hopf_charge  # noqa: E402
from soliton_playground import viz  # noqa: E402
from soliton_playground.provenance import code_provenance  # noqa: E402
from soliton_playground.gpe_lab import (C_BLUE, C_GREEN, C_ORANGE,  # noqa: E402
                                        DARK_STYLE, characteristic_period)

C_FADDEEV = 1.0          # model natural unit, NOT measured -- see docstring
_SECTOR_WORDS = ("baryon", "nucleon", "lepton", "meson", "hyperon")


def knot_label_only(carrier):
    """Strip particle-sector names: the charter forbids identifying any
    structure with a Standard Model particle, whichever upstream is installed."""
    if not carrier:
        return carrier
    return " ".join(w for w in str(carrier).replace("/", " ").split()
                    if w.lower() not in _SECTOR_WORDS)


def id_knot(n, grid, c4):
    """Trace the core of an n-field and identify it.

    pole="auto" is load-bearing: torus_knot_hopfion + arrested_flow leave the
    vacuum at +z and need the -z sheet; knots.py records that the old hard
    pole=+1 default traced the whole +z-vacuum bulk instead (hour-long hangs).
    """
    arr = np.asarray(n)
    ax1 = np.asarray(grid.axis(), float)          # THREE 1D arrays, not one
    axes = (ax1, ax1, ax1)
    try:
        curves = core_curves_from_n(arr[0], arr[1], arr[2], axes, pole="auto")
    except Exception as e:
        return dict(ok=False, error=f"trace: {type(e).__name__}: {e}"), []
    if not curves:
        return dict(ok=False, error="no core curve found"), []
    lengths = [round(float(np.sum(np.linalg.norm(
        np.diff(np.vstack([c, c[:1]]), axis=0), axis=1))), 2) for c in curves]
    try:
        e_d = np.asarray(faddeev_energy_density(jnp.asarray(n), grid, c4=c4))
        info = identify_core_knot(
            curves, scores=curve_energy_scores(curves, e_d, axes), max_points=200)
    except Exception as e:
        return dict(ok=False, n_curves=len(curves), lengths=lengths,
                    error=f"id: {type(e).__name__}: {e}"), curves
    return dict(ok=True, n_curves=len(curves), lengths=lengths,
                determinant=info.get("determinant"),
                knot=knot_label_only(info.get("carrier"))), curves


def draw_core_3d(ax, n_np, dx, c4, curves, *, volume_frac=0.004, sigma=1.0,
                 elev=22.0, azim=-58.0, zoom=1.75):
    """The core in 3D, as an energy-density isosurface, with the traced curves as
    the fallback.

    The isosurface is preferred because the curves CANNOT show the weave: each
    ax.plot is its own Line3D artist and mplot3d depth-sorts whole artists, so a
    knot drawn as lines never occludes itself and reads as a flat tangle from
    every camera. viz.add_parts merges surface faces into one collection, which
    is the only construction in matplotlib that crosses over and under correctly
    (measured in tests/test_viz_depth.py).

    Deliberately the SAME level rule as the animation, driven by the same flag, so
    the still panel and the GIF show the same surface. They briefly did not: the
    panel used 0.80 * max and rendered 168 faces of scatter while the frames
    rendered a clean tube from the identical field.
    """
    part = None
    try:
        e = viz.faddeev_energy_density(n_np[0], n_np[1], n_np[2], dx, c4)
        if sigma > 0:
            e = viz.smooth_periodic(e, sigma)
        part = viz.iso_parts(e, viz.volume_level(e, volume_frac), dx,
                             (1.0, 1.0, 1.0, 1.0))
        if part is not None:
            faces, _ = part
            part = (faces, viz.phase_facecolors(faces, n_np[0], n_np[1], dx))
    except Exception as exc:                      # never lose the figure to viz
        print(f"  (isosurface panel unavailable: {type(exc).__name__}: {exc})")
    if part is not None:
        center, half = viz.bbox_of([part])
        # zoom>1 because this panel is one cell of a 2x3 gridspec: correctly
        # fitted, mplot3d still left the knot in about a quarter of the cell.
        viz.draw_scene(ax, [part], center, half, elev=elev, azim=azim, zoom=zoom)
        return
    ax.set_facecolor("black")
    for c in curves[:6]:
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=C_BLUE, lw=1.6)
    ax.set_axis_off()


def n_to_cp1(nn):
    """Lift a unit n-field back to a CP^1 spinor (gauge phase fixed to zero).
    n = Z^dag sigma Z inverts up to the U(1) phase, which the energy ignores."""
    rho = jnp.clip(jnp.sqrt(nn[0] ** 2 + nn[1] ** 2), 1e-12, None)
    up = jnp.sqrt(jnp.clip((1.0 + nn[2]) / 2.0, 0.0, 1.0))
    dn = jnp.sqrt(jnp.clip((1.0 - nn[2]) / 2.0, 0.0, 1.0))
    return jnp.stack([up, jnp.zeros_like(up), dn * nn[0] / rho, dn * nn[1] / rho])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--L", type=float, default=76.8)
    ap.add_argument("--R", type=float, default=22.0, help="major radius; R/L~0.29")
    ap.add_argument("--w", type=float, default=2.0, help="core radius; R/w~11")
    ap.add_argument("--c4", type=float, default=4.0)
    ap.add_argument("--relax-steps", type=int, default=40000)
    ap.add_argument("--relax-dt", type=float, default=2e-4)
    ap.add_argument("--post-relax", type=int, default=600,
                    help="descent steps before each ID (the anti-mid-bath fix)")
    ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=160000)
    ap.add_argument("--checkpoints", type=int, default=6)
    # Field snapshots are decoupled from checkpoints because the two have
    # different costs: a checkpoint runs --post-relax descent steps plus a knot
    # trace (seconds to minutes), while a snapshot is one savez. Animation wants
    # many snapshots and no more IDs than before. Default 0 keeps the old
    # behaviour -- snapshots only at checkpoints -- because a snapshot is ~3*N^3
    # float32 on disk and defaulting this to 60 would silently cost gigabytes.
    ap.add_argument("--save-fields", type=int, default=0,
                    help="number of field snapshots over the run (0 = only at "
                         "checkpoints). Set this to animate the result.")
    ap.add_argument("--animate", action="store_true",
                    help="render a timelapse GIF from the saved fields when the "
                         "run finishes; implies --save-fields 48 if unset")
    ap.add_argument("--anim-fps", type=int, default=12)
    # Keep this in step with viz.timelapse's own default; they were briefly out
    # of sync and the experiment silently rendered a fragmented core.
    ap.add_argument("--anim-volume-frac", type=float, default=0.004,
                    help="isosurface encloses the hottest fraction of the box; "
                         "held per-frame so the knot stays readable while the "
                         "core spreads (see viz.volume_level)")
    ap.add_argument("--out", type=Path, default=Path("outputs/faddeev_rt2"))
    args = ap.parse_args()
    if args.animate and args.save_fields == 0:
        args.save_fields = 48
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fields").mkdir(exist_ok=True)

    # Resolved flags beside the ledger, same convention as run_ehn_box_vast (see
    # 48fa3d5), and written BEFORE the run: a run that dies is exactly when you
    # need to know what it was asked to do. It is also what makes outputs/ safe to
    # keep gitignored -- the artifacts can be regenerated from this file alone.
    (args.out / "launch.json").write_text(json.dumps(
        {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         # Was a bare `git rev-parse HEAD` on THIS repo: it named the driver and
         # said nothing about the engine, which is the half the physics lives in
         # and the half that is a live sibling checkout. code_provenance() carries
         # both, plus the dirty flags and the solver version.
         "code": code_provenance(),
         "argv": sys.argv[1:],
         "flags": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()}}, indent=1))
    print(f"  launch record -> {args.out / 'launch.json'}", flush=True)

    grid = BoxGrid(N=args.N, L=args.L, dtype=jnp.float64)
    cp1 = faddeev_cp1_model(c4=args.c4)      # convergent frame, for descent
    nmod = faddeev_model(c4=args.c4)         # n-field frame, for real time
    Q_target = args.p * args.m
    b = args.w / 0.7                         # seeds use w = 0.7 b
    print(f"geometry (INFERRED, not compendium-locked): R={args.R:g} "
          f"core={args.w:g} R/core={args.R/args.w:.2f} "
          f"core/dx={args.w/grid.dx:.2f} R/L={args.R/args.L:.3f}", flush=True)

    # ---- phase 1: seed + DEEP relax in the CP^1 spinor frame
    z = torus_knot_hopfion_cp1(grid, args.p, args.q, args.m, R=args.R, b=b, w=args.w)
    n = n_from_state(z)
    q_seed, e_seed = float(hopf_charge(n, grid)), float(nmod.energy(n, grid))
    id_seed, _ = id_knot(n, grid, args.c4)
    print(f"SEED  Q_H={q_seed:+.4f} (target {Q_target}) E={e_seed:.1f} {id_seed}",
          flush=True)

    t0 = time.time()
    z, _ = arrested_flow(cp1, z, grid, dt=args.relax_dt, steps=args.relax_steps,
                         log_every=0)
    gr = cp1.constraint.project_tangent(
        z, jax.grad(lambda s: cp1.energy(s, grid))(z))
    gnorm = float(jnp.sqrt(jnp.sum(gr ** 2)) / np.sqrt(gr.size))
    n = n_from_state(z)
    q_relax, e_relax = float(hopf_charge(n, grid)), float(nmod.energy(n, grid))
    id_relax, curves = id_knot(n, grid, args.c4)
    # Kept because real time overwrites n, and the figure's 3D panel is the
    # RELAXED core. float32 on the host: ~3*N^3*4 bytes, and it never goes back
    # to the device.
    n_relax = np.asarray(n, np.float32)
    print(f"RELAX (CP1, {time.time()-t0:.0f}s) Q_H={q_relax:+.4f} "
          f"E={e_seed:.1f}->{e_relax:.1f} |projgrad|/dof={gnorm:.2e} "
          f"{'STALLED' if gnorm > 1e-3 else 'near-critical'} | {id_relax}",
          flush=True)
    relax_survived = abs(abs(q_relax) - Q_target) < 0.15
    L_core = (max(id_relax["lengths"]) if id_relax.get("lengths")
              else float("nan"))
    tau = (characteristic_period(L_core, c=C_FADDEEV) if L_core == L_core
           else float("nan"))

    # ---- phase 2: real time, post-relax-then-ID at checkpoints
    def post_relax_id(nn):
        raw, _ = id_knot(nn, grid, args.c4)
        zz, _ = arrested_flow(cp1, n_to_cp1(nn), grid, dt=args.relax_dt,
                              steps=args.post_relax, log_every=0)
        settled, _ = id_knot(n_from_state(zz), grid, args.c4)
        return dict(raw_mid_bath=raw, post_relaxed=settled)

    v = jnp.zeros_like(n)
    step_fn = make_verlet_step(nmod, grid, dt=args.dt)
    every = max(1, args.steps // 200)
    ck_at = {int(round(f * args.steps)) for f in np.linspace(0, 1, args.checkpoints)}
    # Snapshots are a superset of checkpoints, so every ID still has the field it
    # was computed from sitting beside it on disk.
    save_at = set(ck_at)
    if args.save_fields > 0:
        save_at |= {int(round(f * args.steps))
                    for f in np.linspace(0, 1, args.save_fields)}
    print(f"real time: {args.steps} steps, {len(ck_at)} checkpoints (ID), "
          f"{len(save_at)} field snapshots", flush=True)
    rows, ck = [], {}
    t0 = time.time()
    for i in range(args.steps + 1):
        if i % every == 0:
            e = float(nmod.energy(n, grid)); k = float(kinetic_energy(v, grid))
            rows.append(dict(step=i, t=i * args.dt, E=e, KE=k, H=e + k,
                             Q_H=float(hopf_charge(n, grid))))
        if i in save_at:
            # Q_H measured HERE, not copied from the last sample row: snapshot
            # steps need not land on sample steps, and a label on a frame should
            # describe that frame.
            np.savez_compressed(args.out / "fields" / f"n_{i:08d}.npz",
                                n=np.asarray(n, np.float32), t=i * args.dt,
                                L=args.L, c4=args.c4,
                                Q_H=float(hopf_charge(n, grid)))
        if i in ck_at:
            rep = post_relax_id(n)
            ck[i] = dict(t=i * args.dt, **rep)
            print(f"  t={i*args.dt:8.3f} ({i:7d}) Q_H={rows[-1]['Q_H']:+.4f} "
                  f"dH/H={(rows[-1]['H']-rows[0]['H'])/rows[0]['H']:+.2e}\n"
                  f"      mid-bath : {rep['raw_mid_bath']}\n"
                  f"      settled  : {rep['post_relaxed']}", flush=True)
        if i < args.steps:
            n, v = step_fn(n, v)
    rt_secs = time.time() - t0

    T = args.steps * args.dt
    H0, H1 = rows[0]["H"], rows[-1]["H"]
    dets_s = [ck[i]["post_relaxed"].get("determinant") for i in sorted(ck)]
    dets_r = [ck[i]["raw_mid_bath"].get("determinant") for i in sorted(ck)]
    d0 = dets_s[0]
    held = d0 is not None and all(d == d0 for d in dets_s if d is not None)
    n_unid = sum(1 for d in dets_s if d is None)
    q_all = [r["Q_H"] for r in rows]
    q_held = all(abs(abs(x) - Q_target) < 0.15 for x in q_all)
    verdict = ("KNOT HELD" if held and q_held and n_unid == 0
               else "KNOT HELD (some unidentifiable)" if held and q_held
               else "Q_H HELD, KNOT CHANGED" if q_held else "Q_H LOST")

    summary = dict(
        status="UNSCORED DEMO (census protocol DRAFT)",
        code=code_provenance(),
        preset="faddeev-skyrme",
        model=f"Faddeev-Skyrme c4={args.c4}; CP1 frame for descent, n-field "
              "constrained Verlet for real time",
        object=f"T({args.p},{args.q}) hopfion m={args.m}",
        protecting_charge=f"Hopf charge Q_H = p*m = {Q_target} (pi_3(S^2) = Z)",
        gate="real-time persistence (fills jax-solitons' empty "
             "test_gate_trefoil_q7_determinant_held)",
        verdict=verdict, determinant_held=held, unidentifiable_checkpoints=n_unid,
        method=dict(
            id_discipline="post-relax-then-ID; never mid-bath (nwt-audit PREREG, "
                          "correction-cinquefoil-decays-were-artifacts)",
            post_relax_steps=args.post_relax,
            measures="basin membership, NOT instantaneous knottedness; raw "
                     "mid-bath ID recorded for comparison",
            relax_frame="CP1 spinor (convergent; the S^2 n-field frame stalls "
                        "at |projgrad|/dof ~ 3.7e-01)"),
        geometry=dict(R=args.R, core_w=args.w, R_over_core=args.R / args.w,
                      core_over_dx=args.w / grid.dx, R_over_L=args.R / args.L,
                      provenance="INFERRED from I1_PHASE2_SCALE_pilot_P.md (3B) "
                                 "ratios (R/core~11, core/dx>=2.5, R/L<=0.35), "
                                 "NOT compendium-locked: all 10 particle "
                                 "entries are model ehn-two-scalar, and no "
                                 "bare-Faddeev config was ever locked in"),
        clock=dict(kind="traversal (tau = L_core/c)", L_core=L_core, tau=tau,
                   c=C_FADDEEV, caveat="c is the model's natural unit, NOT measured"),
        grid=dict(N=args.N, L=args.L, dx=grid.dx, dt=args.dt, steps=args.steps, T=T),
        seed=dict(Q_H=q_seed, E=e_seed, **id_seed),
        relax=dict(Q_H=q_relax, E=e_relax, proj_grad_per_dof=gnorm,
                   survived=relax_survived, steps=args.relax_steps, **id_relax),
        realtime=dict(T=T, periods=(T / tau if tau == tau else None),
                      dH_over_H=(H1 - H0) / H0, Q_H_min=min(q_all),
                      Q_H_max=max(q_all), Q_H_held=q_held,
                      determinant_settled=dets_s, determinant_raw=dets_r,
                      wall_seconds=rt_secs),
        checkpoints={str(k): ck[k] for k in sorted(ck)}, series=rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figure
    plt.rcParams.update(DARK_STYLE)
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.26)
    ts = [r["t"] for r in rows]

    a = fig.add_subplot(gs[0, 0])
    a.plot(ts, [r["Q_H"] for r in rows], color=C_BLUE, lw=2)
    for s in (1, -1):
        a.axhline(s * Q_target, color=C_GREEN, ls="--", lw=1.2)
    a.set_title("Hopf charge (immune to the tracer artifact)", fontsize=10)
    a.set_xlabel("t")

    a = fig.add_subplot(gs[0, 1])
    a.plot(ts, [abs(r["H"] / H0 - 1.0) for r in rows], color=C_ORANGE, lw=2)
    a.set_yscale("log"); a.set_title("|dH/H|", fontsize=10); a.set_xlabel("t")

    a = fig.add_subplot(gs[0, 2])
    tc = [ck[i]["t"] for i in sorted(ck)]
    a.plot(tc, [np.nan if d is None else d for d in dets_s], marker="o", lw=2.4,
           color=C_BLUE, label="post-relaxed (the verdict)")
    a.plot(tc, [np.nan if d is None else d for d in dets_r], marker="x", lw=1.2,
           ls=":", color=C_ORANGE, label="raw mid-bath (artifact-prone)")
    a.set_title("core determinant", fontsize=10); a.set_xlabel("t")
    a.legend(frameon=False, fontsize=8)

    a = fig.add_subplot(gs[1, 0], projection="3d")
    draw_core_3d(a, n_relax, grid.dx, args.c4, curves,
                 volume_frac=args.anim_volume_frac)
    a.set_title(f"relaxed core: det {id_relax.get('determinant')}", fontsize=10,
                color=C_ORANGE)

    a = fig.add_subplot(gs[1, 1:]); a.axis("off")
    card = (f"UNSCORED DEMO — protocol DRAFT\n\nT({args.p},{args.q}) m={args.m}   "
            f"N={args.N} L={args.L:g} dx={grid.dx:.3f} dt={args.dt:g}\n"
            f"R/core={args.R/args.w:.1f}  core/dx={args.w/grid.dx:.2f}  "
            f"R/L={args.R/args.L:.3f}   (INFERRED, not compendium-locked)\n\n"
            f"VERDICT: {verdict}\n\n"
            f"  Q_H  {q_seed:+.4f} -> relax {q_relax:+.4f} -> "
            f"[{min(q_all):+.4f}, {max(q_all):+.4f}]  target {Q_target}\n"
            f"  relax |projgrad|/dof {gnorm:.2e} (CP1 frame)\n"
            f"  det settled   {dets_s}\n"
            f"  det mid-bath  {dets_r}   <- artifact-prone, NOT the verdict\n"
            f"  dH/H over T={T:g}: {(H1-H0)/H0:+.3e}\n"
            f"  tau={tau:.1f}, ran {T/tau if tau==tau else float('nan'):.3f} periods\n\n"
            f"  post-relax-then-ID measures BASIN membership, not instantaneous\n"
            f"  knottedness. Fields saved: re-analysis costs no GPU.\n\n"
            f"no external numbers were compared.")
    a.text(0, 0.98, card, va="top", fontsize=10, color="#DDDDDD",
           family="monospace", linespacing=1.5)

    fig.suptitle("FADDEEV-SKYRME KNOT, REAL TIME (v2: CP1 relax + "
                 "post-relax-then-ID + validated-ratio geometry)", fontsize=13,
                 color="white", y=0.98)
    fig.savefig(args.out / "faddeev_realtime.png", dpi=110, bbox_inches="tight")
    print(f"\nVERDICT: {verdict}\n  settled {dets_s}  raw {dets_r}  "
          f"dH/H {(H1-H0)/H0:+.3e}")

    # ---- animation, from the snapshots already on disk. Deliberately last and
    # in its own try: the run's verdict and figure are the product, and a
    # rendering failure must not cost hours of GPU time.
    if args.animate:
        fields = sorted((args.out / "fields").glob("n_*.npz"))
        print(f"\nanimating {len(fields)} snapshots (CPU; the GPU is free now)")
        try:
            viz.timelapse(fields, args.out / "faddeev_realtime.gif",
                          volume_frac=args.anim_volume_frac, fps=args.anim_fps,
                          L=args.L, c4=args.c4)
        except Exception as exc:
            print(f"  animation failed ({type(exc).__name__}: {exc}); the "
                  f"snapshots are in {args.out/'fields'} and "
                  f"`python -m soliton_playground.viz timelapse` can retry")


if __name__ == "__main__":
    main()
