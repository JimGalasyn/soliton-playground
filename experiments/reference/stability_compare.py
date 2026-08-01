"""Bare L3 vs full L2+L3: does the gauge sector make a knot MORE stable?

Controlled, matched-IC comparison that answers "is the full NWT Lagrangian
producing more stable particles?". For a chosen torus knot T(p,q) at Hopf charge
Q_H = p*m, seed the SAME topology in two models and compare stability:

  bare  = faddeev_model            seed torus_knot_hopfion(p,q,m)      -> n (3,N^3)
  full  = gauged_faddeev_model     seed flux_threaded_knot_seed(p,q,m) -> psi (7,N^3)
          (the Skyrme field for ID/Q_H is n_from_doublet(state))

Two stability probes, cheap -> expensive:
  (1) RELAX-SURVIVAL: deep arrested flow from the seed. Does Q_H survive? This is
      the jax-solitons regime -- bare L3 can UNWIND a low-Q knot while the L2 flux
      tube holds it open. Q_H lost in relax => no stable soliton in that model
      (eps* = 0); Q_H held => a stable soliton to kick.
  (2) eps* BARRIER (only if BOTH survive (1)): white-noise kick sweep with the
      relax-then-ID verdict (the decay_hunt protocol). Higher eps* = more stable.

Verdict for both probes is RELAX-THEN-ID of the unit field n (never mid-bath).

Usage (from /home/jim/.venv):
  python stability_compare.py --p 2 --q 3 --m 1                 # relax-survival only
  python stability_compare.py --p 2 --q 3 --m 3 --eps 1 2 4     # + eps* sweep
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gauged_relaxer"))
sys.path.insert(0, os.path.join(HERE, ".."))                    # nwt_engine shim
OUT = Path(HERE) / "output" / "stability_compare"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--L", type=float, default=18.0)
    ap.add_argument("--c4", type=float, default=4.0, help="Skyrme quartic (both models)")
    ap.add_argument("--c2", type=float, default=1.0, help="Skyrme stiffness (both)")
    ap.add_argument("--e", type=float, default=1.0, help="gauge coupling (full only)")
    ap.add_argument("--v", type=float, default=1.0, help="Higgs vev (full only)")
    ap.add_argument("--relax-steps", type=int, default=4000)
    ap.add_argument("--relax-dt", type=float, default=2e-4)
    ap.add_argument("--eps", type=float, nargs="*", default=None,
                    help="if given, kick-sweep both survivors at these eps")
    ap.add_argument("--kick-steps", type=int, default=3500)
    ap.add_argument("--kick-dt", type=float, default=7e-4)
    ap.add_argument("--post-relax", type=int, default=600)
    ap.add_argument("--id-budget", type=float, default=600)
    ap.add_argument("--common-eref", action="store_true",
                    help="kick all models with the same absolute energy "
                         "(eps x bare relaxed E), not eps x each model's own E0")
    ap.add_argument("--id-sharpen", type=int, default=300,
                    help="bare-faddeev relax steps before tracing (sharpens the "
                         "full model's diffuse gauge core; topology-safe)")
    ap.add_argument("--corr-len", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260613)
    args = ap.parse_args()

    import jax.numpy as jnp
    from nwt_engine import (
        BoxGrid, faddeev_model, faddeev_energy_density, gauged_faddeev_model,
        torus_knot_hopfion, flux_threaded_knot_seed, n_from_doublet,
        hopf_charge, arrested_flow, make_verlet_step, kinetic_energy)
    from core_knot_id import (core_curves_from_n, curve_energy_scores,
                              identify_core_knot, with_time_limit)

    dtype = jnp.float64 if os.environ.get("NWT_X64") == "1" else jnp.float32
    grid = BoxGrid(N=args.N, L=args.L, dtype=dtype)
    axes = (np.linspace(-args.L / 2, args.L / 2, args.N, endpoint=False),) * 3
    Q_target = args.p * args.m
    OUT.mkdir(parents=True, exist_ok=True)

    sharpen_model = faddeev_model(c4=args.c4)

    def ident(n_arr, budget):
        def _go():
            # Sharpen in BARE faddeev before tracing: the full model's gauge halo
            # leaves a diffuse core whose {n1=0,n2=0} preimage makes the tracer go
            # combinatorial (observed: 22-min hangs at N=160). A short topology-safe
            # descent tightens it to a clean curve (knot type preserved; area-form
            # barrier). Harmless for the already-sharp bare field.
            st = jnp.stack([jnp.asarray(np.asarray(n_arr[i], np.float64))
                            for i in range(3)])
            st, _ = arrested_flow(sharpen_model, st, grid, dt=2e-4,
                                  steps=args.id_sharpen)
            a = [np.asarray(st[i], dtype=np.float64) for i in range(3)]
            # pole defaults to "auto" (anti-vacuum = -sign(mean n3)) in the library.
            curves = core_curves_from_n(*a, axes, seed_tol=0.30)
            scores = None
            if curves:
                e = np.asarray(faddeev_energy_density(
                    jnp.stack([jnp.asarray(x) for x in a]), grid, c4=args.c4))
                scores = curve_energy_scores(curves, e, axes)
            # cap points: pure-Python pyknotid goes combinatorial >~150 jittery pts.
            info = identify_core_knot(curves, scores=scores, max_points=150)
            return info["determinant"], info["n_components"]
        return with_time_limit(budget, _go, (-2, -1))

    # n-field accessor per model (full: slaved to the doublet)
    def n_of(model_name, state):
        return state if model_name == "bare" else n_from_doublet(state)

    models = {
        # bare E2Term is c2=1 (fixed), matching gauged_faddeev's default c2=1.0
        "bare": (faddeev_model(c4=args.c4),
                 lambda: torus_knot_hopfion(grid, args.p, args.q, args.m).astype(dtype)),
        "full": (gauged_faddeev_model(e=args.e, v=args.v, c2=args.c2, c4=args.c4),
                 lambda: flux_threaded_knot_seed(grid, args.p, args.q, args.m,
                                                 e=args.e, v=args.v).astype(dtype)),
    }

    print("=" * 78)
    print(f"STABILITY COMPARE  T({args.p},{args.q}) m={args.m}  Q_H target={Q_target}  "
          f"N={args.N} L={args.L} c4={args.c4} dtype={dtype.__name__}")
    print("=" * 78, flush=True)

    report = {"knot": dict(p=args.p, q=args.q, m=args.m, Q_target=Q_target),
              "N": args.N, "L": args.L, "c4": args.c4, "models": {}}

    for name, (model, seed_fn) in models.items():
        t0 = time.time()
        state = seed_fn()
        E_seed = float(model.energy(state, grid))
        Q_seed = float(hopf_charge(n_of(name, state), grid))
        # (1) relax-survival
        state_r, _ = arrested_flow(model, state, grid, dt=args.relax_dt,
                                   steps=args.relax_steps)
        E_r = float(model.energy(state_r, grid))
        Q_r = float(hopf_charge(n_of(name, state_r), grid))
        held = abs(abs(Q_r) - Q_target) < 0.5
        det_r, nc_r = ident(n_of(name, state_r), args.id_budget) if held else (-3, -1)
        print(f"[{name}] seed E={E_seed:.0f} Q_H={Q_seed:+.2f} -> relax "
              f"{args.relax_steps}: E={E_r:.0f} Q_H={Q_r:+.2f}  "
              f"{'HELD' if held else 'UNWOUND'}  det={det_r}  [{time.time()-t0:.0f}s]",
              flush=True)
        np.savez_compressed(OUT / f"{name}_relaxed_p{args.p}q{args.q}m{args.m}.npz",
                            state=np.asarray(state_r), Q_H=Q_r, E=E_r)
        report["models"][name] = dict(E_seed=E_seed, Q_seed=Q_seed, E_relax=E_r,
                                      Q_relax=Q_r, relax_held=held,
                                      det_relax=det_r, eps_sweep={})

    # (2) eps* sweep — only for models whose knot survived relaxation
    if args.eps:
        survivors = [n for n in models if report["models"][n]["relax_held"]]
        # --common-eref: kick BOTH models with the SAME ABSOLUTE energy (eps x the
        # bare relaxed energy) instead of eps x each model's own E0. The full model
        # carries ~50% more energy in its gauge/Higgs sector, so per-model eps
        # over-kicks it — an apples-to-apples robustness test fixes the reference.
        eref_common = None
        if args.common_eref:
            ref = "bare" if "bare" in survivors else (survivors[0] if survivors else None)
            eref_common = report["models"][ref]["E_relax"] if ref else None
        print(f"\neps* sweep on relax-survivors {survivors}: eps={args.eps}"
              f"{f' (common E_ref={eref_common:.0f} from {ref})' if eref_common else ''}",
              flush=True)
        for name in survivors:
            model, _ = models[name]
            d = np.load(OUT / f"{name}_relaxed_p{args.p}q{args.q}m{args.m}.npz")
            base = jnp.asarray(d["state"])
            step_fn = make_verlet_step(model, grid, dt=args.kick_dt)
            _, _, _, K2 = grid.k_vectors()
            filt = jnp.exp(-K2 * args.corr_len**2)
            E0 = eref_common if eref_common is not None else float(model.energy(base, grid))
            for eps in args.eps:
                rng = np.random.default_rng(args.seed)
                w = jnp.stack([jnp.real(jnp.fft.ifftn(jnp.fft.fftn(
                    jnp.asarray(rng.standard_normal(base.shape[1:]))) * filt))
                    for _ in range(base.shape[0])]).astype(dtype)
                T = float(kinetic_energy(w, grid))
                v = w * np.sqrt(eps * E0 / T)
                s = base
                for _ in range(args.kick_steps):
                    s, v = step_fn(s, v)
                s_r, _ = arrested_flow(model, s, grid, dt=args.relax_dt,
                                       steps=args.post_relax)
                det_f, nc_f = ident(n_of(name, s_r), args.id_budget)
                verdict = ("SURVIVED" if det_f == report["models"][name]["det_relax"]
                           else "INDETERMINATE" if det_f == -2 else "DECAYED")
                report["models"][name]["eps_sweep"][f"{eps}"] = dict(
                    det_final=det_f, verdict=verdict)
                print(f"  [{name}] eps={eps:.2f}: {verdict} (det={det_f})", flush=True)

    (OUT / f"compare_p{args.p}q{args.q}m{args.m}.json").write_text(
        json.dumps(report, indent=1) + "\n")
    print("-" * 78)
    for name in models:
        r = report["models"][name]
        print(f"  {name}: relax {'HELD' if r['relax_held'] else 'UNWOUND'} "
              f"(Q_H={r['Q_relax']:+.2f}, det={r['det_relax']})"
              + (f"  eps*: {r['eps_sweep']}" if r['eps_sweep'] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
