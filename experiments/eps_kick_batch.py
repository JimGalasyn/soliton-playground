"""On-box compute leg for the multi-seed eps* grid (bare vs full vs fullcp1). Pure jax_solitons.

RUNBOOK split: the GPU-heavy part (relax -> vmap-kick over K seeds -> per-seed
post-relax) runs on the rented box; knot ID (core_knot_id / pyknotid) stays LOCAL
on the returned n-fields (eps_kick_id.py). The box has jax_solitons from GitHub
(onstart) but NOT the private nwt_engine shim or core_knot_id.

One leg = one knot (p,q,m) at the given eps list, BOTH models, K seeds:
  - relax bare + full; eref = bare relaxed energy (common reference -> same
    ABSOLUTE kick for both models, apples-to-apples; the full model carries ~50%
    more energy in its gauge sector so per-model eps would over-kick it).
  - per (model, eps): build K white-noise kicks scaled to KE = eps*eref, evolve
    all K in ONE vmapped Verlet pass (the kick IS vmappable), then post-relax each
    seed (arrested_flow is NOT vmappable) and save n_of(final).

CRITICAL: kick-dt MUST be <= 5e-5. The gauged (7-component) integrator's CFL
limit is ~14x tighter than bare; at the bare-tuned 7e-4 the full run blows up to
NaN (that blow-up was the spurious "full more fragile" result). Same dt both
models.

MEMORY: the kick vmaps over the K seeds, so the K-wide batch -- NOT the field --
is the memory hog. A single config fits a 24GB GPU even at N=250; the 8-wide vmap
OOMs at N=160. Use --seed-batch to cap seeds-per-pass (e.g. 4 or 2): fits larger N
on the same GPU, bit-identical results (vmap is independent per seed). The problem
is WIDE (many independent seeds), not TALL (one giant field) -> scale by fanning
seeds across the fleet, not by sharding one field (domain-decomp would need an
all-to-all 3D-FFT and is unnecessary until one config no longer fits a GPU).

Usage (on box, or locally):
  python eps_kick_batch.py --p 2 --q 3 --m 1 --eps 0.1 0.25 0.5 1.0 \
      --nseeds 8 --kick-dt 5e-5 --kick-steps 20000 --out out_kick
  # larger N on a 24GB GPU: add --seed-batch 4   (or 2 for N>=224)
"""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from jax_solitons.grid import BoxGrid
from jax_solitons.model import Model
from jax_solitons.models import faddeev_model
from jax_solitons.models.gauged_faddeev import (
    DoubletCovariantKineticTerm, DoubletMagneticTerm, SkyrmeE2Term, SkyrmeE4Term,
    gauged_faddeev_model, n_from_doublet, magnetic_flux, hopf_charge_doublet)
from jax_solitons.seeds import torus_knot_hopfion, flux_threaded_knot_seed
from jax_solitons.topology import hopf_charge
from jax_solitons.steppers import arrested_flow, kinetic_energy
from jax_solitons.steppers.verlet import make_verlet_step

# Migrated 2026-08-01 out of null-worldtube-private. This used to be a loose
# sibling-directory import that only worked because the driver was shipped
# flat beside gauged_iso.py; it is a package module here.
from soliton_playground.gauged_iso import gauged_iso_model, n_of_iso
from soliton_playground.provenance import code_provenance


class _FrozenModulusConstraint:
    """Hard |psi| = v on the doublet block (state[0:4]); the gauge field
    (state[4:7]) is unconstrained. This is the gauged-CP^1 limit: psi lives on
    the radius-v 3-sphere and NEVER vanishes, so n = psi^dag.sigma.psi/(psi^dag
    psi) has no 0/0 vortex core and the (d_i n)^2 Skyrme term cannot log(dx)-
    sharpen -- the proposed cure for the soft-potential full model's resolution
    divergence. `project_tangent` removes the radial-psi component of a gradient
    / velocity (so a kick excites only the gauge-invariant CP^1 + gauge
    directions, never the frozen modulus); `retract` rescales psi onto the
    v-sphere. Same project+retract protocol as faddeev's S2Constraint, applied to
    the doublet sub-block of the (7,N,N,N) state.
    """

    def __init__(self, v: float = 1.0):
        self.v = v

    def project_tangent(self, state, grad):
        psi = state[:4]
        mod2 = jnp.sum(psi * psi, axis=0, keepdims=True)          # |psi|^2 > 0 on surface
        radial = jnp.sum(grad[:4] * psi, axis=0, keepdims=True) / mod2
        gp = grad[:4] - radial * psi                              # tangent to S^3_v
        return jnp.concatenate([gp, grad[4:]], axis=0)

    def retract(self, state):
        psi = state[:4]
        mod = jnp.sqrt(jnp.sum(psi * psi, axis=0, keepdims=True))
        return jnp.concatenate([self.v * psi / mod, state[4:]], axis=0)


def frozen_modulus_gauged_model(c4: float, e: float = 1.0, v: float = 1.0,
                                c2: float = 1.0) -> Model:
    """Frozen-modulus (|psi|=v) gauged Faddeev-Skyrme model = the same gauged
    terms as `gauged_faddeev_model` MINUS the Higgs potential (identically zero
    on the constraint surface, and its gradient is purely radial -> projected out
    anyway), with a hard `_FrozenModulusConstraint`. Self-contained (public
    jax_solitons terms only) so the single shipped driver runs it on the box.
    """
    terms = (DoubletCovariantKineticTerm(e=e), DoubletMagneticTerm(e=e),
             SkyrmeE2Term(c2=c2), SkyrmeE4Term(c4=c4))
    return Model(name="frozen_modulus_gauged_faddeev", terms=terms,
                 constraint=_FrozenModulusConstraint(v=v),
                 charges=(lambda s, g: magnetic_flux(s[2:7], g, e=e),
                          hopf_charge_doublet))


def _state_diag(state, n, model, grid):
    """Higgs-vortex / sharpening-gradient diagnostics for the gauged model.
    `state` is the bare unit field (3,N,N,N) or the full doublet (7,N,N,N); `n`
    is the derived unit field. The full model slaves n = psi^dag.sigma.psi /
    (psi^dag psi), a *regularized* 0/0 at psi-zeros (Higgs vortex cores): the
    denominator carries an eps floor, so n stays bounded (n -> 0 at a true zero,
    |n| < 1 on the eps-shell) rather than literally diverging. But the (d_i n)^2
    Skyrme gradients on that eps-shell sharpen as dx -> 0 -- finer grids resolve
    more of the near-core structure, so e2_n grows with N (the idealized-limit
    log(dx) signature, smoothed by the regulator) -- the suspected cause of the
    full model's resolution divergence. We record psi_min / low-|psi| fraction
    (vortex presence) and the (d_i n)^2 Dirichlet energy of n (the sharpening
    quantity) + the true total energy, none of which survive the n-only npz save.
    """
    dx = grid.dx
    acc = 0.0
    for a in range(3):
        for i in range(3):
            d = (jnp.roll(n[a], -1, axis=i) - n[a]) / dx
            acc = acc + d * d
    diag = dict(e2_n=float(jnp.sum(acc) * dx**3),
                E_total=float(model.energy(state, grid)))
    if state.shape[0] >= 7:                       # full doublet: |psi| from reals 0..3
        mod = jnp.sqrt(state[0]**2 + state[1]**2 + state[2]**2 + state[3]**2)
        diag.update(psi_min=float(jnp.min(mod)), lowfrac=float(jnp.mean(mod < 0.1)))
    else:                                          # bare: n is fundamental, no psi-zero
        diag.update(psi_min=float("nan"), lowfrac=float("nan"))
    return diag


def _progress(phase, **fields):
    """Emit one structured progress marker on stdout (parsed by the campaign
    FleetExecutor's `parse_progress_line` when the leg runs with stream_progress).
    Lives BETWEEN the monolithic jitted blocks, so it costs nothing -- but on a
    multi-hour N>=160 leg it's the difference between a live phase readout and
    flying blind (stdout otherwise only surfaces at leg completion)."""
    kv = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[[progress phase={phase} {kv}]]".rstrip(), flush=True)


def _save_npz_atomic(path: Path, **arrays):
    """Write a .npz via a temp file + os.replace, so a box killed mid-write never
    leaves a half-written archive that skip-if-exists (resume) would later trust.
    The tmp lands in the SAME dir (atomic rename needs one filesystem)."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)   # tmp ends in .npz -> numpy won't re-append
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _write_text_atomic(path: Path, text: str):
    """Atomic text write (manifest.json) -- temp file + os.replace."""
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--q", type=int, required=True)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--L", type=float, default=18.0)
    ap.add_argument("--c4", type=float, default=4.0)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--nseeds", type=int, default=8)
    ap.add_argument("--relax-steps", type=int, default=2500)
    ap.add_argument("--relax-dt", type=float, default=2e-4)
    ap.add_argument("--kick-dt", type=float, default=5e-5)
    ap.add_argument("--kick-steps", type=int, default=20000)
    ap.add_argument("--post-relax", type=int, default=1500)
    ap.add_argument("--corr-len", type=float, default=1.0)
    ap.add_argument("--seed-batch", type=int, default=0,
                    help="vmap at most this many seeds per kick pass (0=all nseeds, "
                         "current behavior). The kick vmaps over seeds, so the 8-wide "
                         "batch -- not the field -- is the memory hog: a single config "
                         "fits a 24GB GPU even at N=250, but x8 OOMs at N=160. Chunking "
                         "(e.g. 4 or 2) fits larger N on the same GPU; per-seed results "
                         "are bit-identical (vmap is independent per element).")
    ap.add_argument("--seed0", type=int, default=20260615)
    ap.add_argument("--out", default="out_kick")
    ap.add_argument("--models", nargs="+", default=["bare", "full"],
                    choices=["bare", "full", "fullcp1", "fulliso"],
                    help="which models to run; 'bare' is required (sets eref kick "
                         "reference). 'fullcp1' = frozen-modulus (|psi|=v) gauged-CP^1 "
                         "limit; 'fulliso' = gauged-isorotation U(1) on a fundamental "
                         "S^2-constrained n (Q_H protected by construction).")
    ap.add_argument("--kick-sector", default="all", choices=["all", "cp1"],
                    help="'all' (default) kicks every component incl. the gauge "
                         "field A; 'cp1' zeros the gauge-field velocity so the kick "
                         "energy lands only in the doublet (CP^1) sector -- the "
                         "control that isolates whether the gauge-sector kick is what "
                         "scatters Q_H in the gauged models. No-op for bare.")
    args = ap.parse_args()

    assert args.kick_dt <= 5e-5, "gauged CFL: kick-dt must be <= 5e-5 or full NaNs"
    dtype = jnp.float32
    grid = BoxGrid(N=args.N, L=args.L, dtype=dtype)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    _, _, _, K2 = grid.k_vectors()
    filt = jnp.exp(-K2 * args.corr_len**2)
    seeds = [args.seed0 + i for i in range(args.nseeds)]

    specs = {
        "bare": (faddeev_model(c4=args.c4),
                 lambda: torus_knot_hopfion(grid, args.p, args.q, args.m).astype(dtype),
                 lambda s: s),
        "full": (gauged_faddeev_model(c4=args.c4),
                 lambda: flux_threaded_knot_seed(grid, args.p, args.q, args.m).astype(dtype),
                 n_from_doublet),
        "fullcp1": (frozen_modulus_gauged_model(c4=args.c4),
                    lambda: flux_threaded_knot_seed(grid, args.p, args.q, args.m).astype(dtype),
                    n_from_doublet),
        # gauged-isorotation (option 4): n fundamental + S^2-constrained, U(1)
        # gauged via the e3-isorotation link. Seed = bare hopfion + A=0; n stays
        # fundamental so Q_H is a protected invariant. State is (6,N,N,N).
        "fulliso": (gauged_iso_model(c4=args.c4),
                    lambda: jnp.concatenate(
                        [torus_knot_hopfion(grid, args.p, args.q, args.m).astype(dtype),
                         jnp.zeros((3, args.N, args.N, args.N), dtype)], axis=0),
                    n_of_iso),
    }
    specs = {k: v for k, v in specs.items() if k in args.models}
    assert "bare" in specs, "eref is the bare relaxed energy; include bare in --models"

    print("=" * 78)
    print(f"EPS KICK BATCH  T({args.p},{args.q}) m={args.m}  N={args.N} K={args.nseeds} "
          f"kick-dt={args.kick_dt} steps={args.kick_steps}  eps={args.eps}")
    print("=" * 78, flush=True)

    # relax both models; eref = bare relaxed energy (common absolute kick reference)
    relaxed, eref, relaxed_diag = {}, None, {}
    nmodels = len(specs)
    for mi, (name, (model, seed_fn, n_of)) in enumerate(specs.items()):
        # Resume: relaxation is deterministic and shared across all seeds/eps, so
        # re-burning it on every relaunch is pure waste (the N=160 ladder's biggest
        # avoidable cost). If a prior {name}_relaxed.npz with the full STATE is
        # present (restored off-box), reload st and skip the relax. Old npz files
        # saved only the derived n (no `state`) -> fall back to recompute.
        rpath = out / f"{name}_relaxed.npz"
        st = E = None
        if rpath.exists():
            with np.load(rpath) as d:
                if "state" in d.files:
                    st = jnp.asarray(d["state"])
                    E = float(d["E"])
        resumed = st is not None
        _progress("relax", model=name, done=f"{mi}/{nmodels}",
                  **({"resumed": 1} if resumed else {}))
        if not resumed:
            st, _ = arrested_flow(model, seed_fn(), grid, dt=args.relax_dt,
                                  steps=args.relax_steps)
            st.block_until_ready()
            E = float(model.energy(st, grid))
        relaxed[name] = st
        if name == "bare":
            eref = E
        n_st = n_of(st)
        relaxed_diag[name] = _state_diag(st, n_st, model, grid)
        if not resumed:
            _save_npz_atomic(rpath, state=np.asarray(st),
                             n=np.asarray(n_st), Q_H=float(hopf_charge(n_st, grid)),
                             E=E, p=args.p, q=args.q, m=args.m, N=args.N, L=args.L,
                             c4=args.c4)
        print(f"[{name}] relaxed{' (resumed)' if resumed else ''} E={E:.0f} "
              f"Q_H={float(hopf_charge(n_st, grid)):+.2f} "
              f"psi_min={relaxed_diag[name]['psi_min']:.3f} "
              f"e2_n={relaxed_diag[name]['e2_n']:.0f}", flush=True)

    manifest = dict(knot=[args.p, args.q, args.m], N=args.N, L=args.L, c4=args.c4,
                    eref=eref, kick_dt=args.kick_dt, kick_steps=args.kick_steps,
                    nseeds=args.nseeds, seeds=seeds, eps=args.eps,
                    kick_sector=args.kick_sector,
                    relaxed_diag=relaxed_diag, legs=[],
                    code=code_provenance())

    for name, (model, seed_fn, n_of) in specs.items():
        step1 = make_verlet_step(model, grid, dt=args.kick_dt)
        bstep = jax.jit(jax.vmap(step1))
        base = relaxed[name]
        for eps in args.eps:
            tag = f"{name}_eps{eps}"
            tpath = out / f"{tag}.npz"
            # Resume: a completed per-eps leg is independent -> if its npz is
            # already present (restored off-box), fold it back into the manifest
            # and skip the recompute. diag_json carries the per-seed diagnostics
            # (the npz's n-fields alone can't reconstruct psi_min/E, which need
            # the post-relax STATE), so the manifest is faithful on resume.
            if tpath.exists():
                with np.load(tpath) as d:
                    qh = [float(x) for x in d["Q_H"]]
                    seed_diags = (json.loads(str(d["diag_json"]))
                                  if "diag_json" in d.files else [])
                manifest["legs"].append(dict(model=name, eps=eps, seeds=seeds,
                                             Q_H=qh, diag=seed_diags, wall_s=0.0,
                                             resumed=True))
                print(f"  {tag}: resumed from {tpath.name} (skipped recompute)",
                      flush=True)
                _progress("leg_done", model=name, eps=eps, resumed=1,
                          qh_min=f"{min(qh):+.2f}", qh_max=f"{max(qh):+.2f}")
                continue
            t0 = time.time()
            sb = args.seed_batch or args.nseeds        # 0 -> all at once (old path)
            nb = (args.nseeds + sb - 1) // sb           # number of seed-batches
            ns = []
            seed_diags = []
            # Chunk the seed-vmap so the per-pass working set fits memory. Per-seed
            # trajectories are independent under vmap, so chunking is bit-identical
            # to one wide pass; build each chunk's kicks lazily so unused seeds'
            # fields never sit resident, and free GPU arrays between chunks.
            for c0 in range(0, args.nseeds, sb):
                chunk = seeds[c0:c0 + sb]
                vs = []
                for s in chunk:
                    rng = np.random.default_rng(s)
                    w = jnp.stack([jnp.real(jnp.fft.ifftn(jnp.fft.fftn(
                        jnp.asarray(rng.standard_normal(base.shape[1:]))) * filt))
                        for _ in range(base.shape[0])]).astype(dtype)
                    # CP1-only control: zero the gauge-field velocity (comps 4:7)
                    # BEFORE the KE normalization, so all eps*eref lands in the
                    # doublet (CP1) sector and the gauge field A gets no initial
                    # kick -- isolates whether the gauge-sector kick is what
                    # scatters Q_H. No-op for bare (3-comp, no gauge sector).
                    if args.kick_sector == "cp1" and w.shape[0] >= 7:
                        w = w.at[4:7].set(0.0)
                    T = float(kinetic_energy(w, grid))
                    vs.append(w * np.sqrt(eps * eref / T))
                bn = jnp.stack([base] * len(chunk))
                bv = jnp.stack(vs)
                del vs
                _progress("kick", model=name, eps=eps,
                          batch=f"{c0 // sb + 1}/{nb}", steps=args.kick_steps)
                for _ in range(args.kick_steps):
                    bn, bv = bstep(bn, bv)
                bn.block_until_ready()
                _progress("postrelax", model=name, eps=eps,
                          batch=f"{c0 // sb + 1}/{nb}")
                for i in range(len(chunk)):
                    sr, _ = arrested_flow(model, bn[i], grid, dt=args.relax_dt,
                                          steps=args.post_relax)
                    n_jax = n_of(sr)
                    seed_diags.append(_state_diag(sr, n_jax, model, grid))
                    ns.append(np.asarray(n_jax))
                del bn, bv
            arr = np.stack(ns)                                   # (K,3,N,N,N)
            qh = [float(hopf_charge(jnp.asarray(x), grid)) for x in ns]
            _save_npz_atomic(tpath, n=arr, seeds=seeds, eps=eps,
                             model=name, Q_H=qh, diag_json=json.dumps(seed_diags),
                             p=args.p, q=args.q, m=args.m,
                             N=args.N, L=args.L, c4=args.c4)
            manifest["legs"].append(dict(model=name, eps=eps, seeds=seeds, Q_H=qh,
                                         diag=seed_diags, wall_s=time.time() - t0))
            print(f"  {tag}: {args.nseeds} seeds kicked+relaxed  "
                  f"Q_H[{min(qh):+.2f},{max(qh):+.2f}]  [{time.time()-t0:.0f}s]", flush=True)
            _progress("leg_done", model=name, eps=eps,
                      qh_min=f"{min(qh):+.2f}", qh_max=f"{max(qh):+.2f}")

    _write_text_atomic(out / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    print(f"wrote -> {out}/manifest.json", flush=True)
    _progress("done", legs=len(manifest["legs"]))


if __name__ == "__main__":
    main()
