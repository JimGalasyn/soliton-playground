#!/usr/bin/env python3
"""Census: dark-soliton snake decay chain — UNSCORED DEMO (protocol DRAFT).

Two stationary planar dark solitons (a periodic kink-antikink pair at z = ∓16)
are given a small smooth transverse displacement. Expected chain, stated before
running: plane → snake instability → pinch-off into vortex rings + sound.
The plane is the census's first *unstable-with-a-channel* entrant; the rings it
sheds are the protected bin's population growing at the expense of the unstable
one. Everything is internal to the medium; no external numbers are compared.
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
    C_BLUE, C_ORANGE, DARK_STYLE, depletion_metrics, evolve, make_energy,
    planar_soliton_pair_seed, seed_gate, smooth)

Z_PLANES = (-16.0, 16.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--L", type=float, default=64.0)
    ap.add_argument("--T", type=float, default=120.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("outputs/snake"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    grid = BoxGrid(N=args.N, L=args.L, dtype=jnp.float64)
    energy = make_energy(grid)

    psi0 = smooth(grid, planar_soliton_pair_seed(grid, *Z_PLANES,
                                                 noise_amp=args.noise))
    ok, gate = seed_gate(grid, psi0, axes=(2,))   # planes are z-localized only
    print(f"seed gate: {'PASS' if ok else 'FAIL'} {gate}")
    if not ok:
        raise SystemExit("seed gate failed")
    k0, p0 = energy(psi0)
    E0 = float(k0 + p0)

    def observer(t, psi):
        k, p = energy(psi)
        m = depletion_metrics(psi, grid)
        return dict(E_kin=float(k), E_pot=float(p), E_tot=float(k + p), **m)

    keep = tuple(round(f * args.T, 6) for f in (0.0, 1 / 3, 2 / 3, 1.0))
    psi1, rows, slices = evolve(grid, psi0, T=args.T, dt=args.dt,
                                sample_dt=1.0, observer=observer,
                                keep_slices_at=keep)

    # face-on view of the lower plane's debris: min-density projection over the
    # slab z in [-24, -8]
    dens = np.asarray(jnp.abs(psi1) ** 2)
    ax = np.asarray(grid.axis())
    slab = dens[:, :, (ax > -24) & (ax < -8)].min(axis=2)

    end = rows[-1]
    drift = abs(end["E_tot"] / rows[0]["E_tot"] - 1.0)
    verdict = ("DECAYED->RINGS" if end["n_blobs"] >= 4 and end["V_dep"] > 0
               else "SURVIVES" if end["n_blobs"] <= 2 else "DECAYED")

    # event graph: two planes, each DECAY -> ring debris + phonon receipt
    zoo = dict(ns="zoo", charge_keys=("E",), receipt_pdg={"PHONON": 22})
    g = EventGraph("snake_chain", **zoo)
    z_end = np.asarray(grid.coords()[2])
    dens_mask = dens < 0.5
    n_lo = int(__import__("scipy.ndimage", fromlist=["label"])
               .label(dens_mask & (z_end < 0))[1])
    n_hi = int(__import__("scipy.ndimage", fromlist=["label"])
               .label(dens_mask & (z_end >= 0))[1])
    for half, n in (("z<0", n_lo), ("z>0", n_hi)):
        plane = g.add_particle(PDG_PRIVATE, 4, {"E": E0 / 2},
                               {"zoo.object": f"dark_plane {half}"})
        rings = [g.add_particle(PDG_PRIVATE, 2, {},
                                {"zoo.object": f"ring {half} #{i}"})
                 for i in range(n)]
        receipt = g.add_particle(22, 1, {"E": E0 / 2}, {"zoo.receipt": "PHONON"})
        g.add_vertex("DECAY", [plane], rings + [receipt])
    closure = g.check_conservation()

    summary = dict(status="UNSCORED DEMO (census protocol DRAFT)",
                   grid=dict(N=args.N, L=args.L, dt=args.dt, T=args.T,
                             noise=args.noise),
                   seed_gate=gate, E0=E0, verdict=verdict,
                   blobs_final=end["n_blobs"], blobs_lo=n_lo, blobs_hi=n_hi,
                   ledger_drift=drift,
                   calorimeter_closure={str(k): v for k, v in closure.items()},
                   series=rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "chain.hepmc3").write_text(g.to_hepmc3())

    # ---- figure
    plt.rcParams.update(DARK_STYLE)
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.3, wspace=0.2)
    ext = [ax[0], ax[-1], ax[0], ax[-1]]
    for col, t in enumerate(keep):
        a = fig.add_subplot(gs[0, col])
        a.imshow(slices[float(t)].T, origin="lower", extent=ext, cmap="magma",
                 vmin=0, vmax=1.3, aspect="equal")
        a.set_title(f"t = {t:g}", fontsize=11, color=C_BLUE)
        a.set_xlabel("x / ξ")
        if col == 0:
            a.set_ylabel("z / ξ")
        a.tick_params(labelsize=8)

    a = fig.add_subplot(gs[1, 0])
    a.imshow(slab.T, origin="lower", extent=ext, cmap="magma", vmin=0, vmax=1.1,
             aspect="equal")
    a.set_title("lower-plane debris (min over z∈[−24,−8])", fontsize=10,
                color=C_ORANGE)
    a.set_xlabel("x / ξ"); a.set_ylabel("y / ξ"); a.tick_params(labelsize=8)

    t_s = [r["t"] for r in rows]
    a = fig.add_subplot(gs[1, 1])
    a.plot(t_s, [r["V_dep"] for r in rows], color=C_BLUE, lw=2)
    a.set_title("depletion volume (ξ³)", fontsize=10)
    a.set_xlabel("t")

    a = fig.add_subplot(gs[1, 2])
    a.plot(t_s, [r["n_blobs"] for r in rows], color=C_ORANGE, lw=2)
    a.set_title("depletion blobs (2 planes → ring debris)", fontsize=10)
    a.set_xlabel("t")

    a = fig.add_subplot(gs[1, 3]); a.axis("off")
    card = (f"UNSCORED DEMO — protocol DRAFT\n\n"
            f"seed gate: PASS\n"
            f"  shell density {gate['shell_min_density']:.4f}\n"
            f"  wrap jump {gate['wrap_jump_max']:.3f} rad\n\n"
            f"DARK PLANES (W=0, stationary):\n  {verdict}\n"
            f"  blobs 2 → {end['n_blobs']}"
            f"  (z<0: {n_lo}, z>0: {n_hi})\n"
            f"  ledger drift {drift:.1e}\n\n"
            f"channel: snake → vortex rings + sound\n"
            f"no external numbers were compared.")
    a.text(0, 0.95, card, va="top", fontsize=10.5, color="#DDDDDD",
           linespacing=1.6)

    fig.suptitle("CENSUS — DARK-SOLITON SNAKE DECAY CHAIN  (|ψ|², y=0 plane)",
                 fontsize=14, color="white", y=0.99)
    out = args.out / "snake_chain.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(json.dumps({k: summary[k] for k in
                      ("verdict", "blobs_final", "blobs_lo", "blobs_hi",
                       "ledger_drift")}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
