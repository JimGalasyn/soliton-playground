#!/usr/bin/env python3
"""Census: Jones–Roberts branch hunt — UNSCORED DEMO (protocol DRAFT).

The GPE's solitary-wave family connects vortex rings (slow, cored, winding ±1)
to rarefaction pulses (fast, coreless, winding 0) along one branch. The hunt:
seed rings of decreasing radius and measure each object's speed U, final
minimum density, and winding. Expected, stated before running: below a critical
seed radius the core fills in and a coreless density dip keeps propagating —
the branch crossover, at U somewhere above half the sound speed.

The crossover speed is a mathematical property of the equation (GPE lore puts
it near U ≈ 0.63c); finding it in OUR box is the anchor unit test permitted by
the census's anchors policy. No measured physical constant is involved.
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
from soliton_playground.gpe_lab import (  # noqa: E402
    C_BLUE, C_ORANGE, DARK_STYLE, dip_centroid_z, evolve, make_energy,
    ring_pair_seed, seed_gate, smooth, winding_xz)

Z0 = -10.0
RADII = (6.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.2)


def run_radius(grid, energy, R, T, dt):
    psi0 = smooth(grid, ring_pair_seed(grid, R=R, z0=Z0))
    ok, gate = seed_gate(grid, psi0)
    if not ok:
        raise SystemExit(f"seed gate failed for R={R}: {gate}")

    def observer(t, psi):
        return dict(z_dip=dip_centroid_z(psi, grid, zmax=0.0))

    psi1, rows, _ = evolve(grid, psi0, T=T, dt=dt, sample_dt=0.5,
                           observer=observer)

    # speed: linear fit over the post-transient window
    ts = np.array([r["t"] for r in rows])
    zs = np.array([r["z_dip"] for r in rows])
    win = (ts >= min(5.0, T / 3)) & np.isfinite(zs)
    U = float(np.polyfit(ts[win], zs[win], 1)[0])

    # final core status in the z<0 half, y=0 plane
    dens = np.asarray(jnp.abs(psi1[:, grid.N // 2, :]) ** 2)
    ax = np.asarray(grid.axis())
    half = dens[:, ax < 0]
    n_min = float(half.min())
    ix, kz = np.unravel_index(half.argmin(), half.shape)
    x_core, z_core = float(ax[ix]), float(ax[ax < 0][kz])
    W = winding_xz(psi1, grid, abs(x_core), z_core, half=2.0)
    kind = "vortex ring" if abs(W) > 0.5 and n_min < 0.05 else "rarefaction pulse"
    slice_end = dens
    return dict(R=R, U=abs(U), n_min=n_min, winding=float(W), kind=kind,
                x_core=x_core, z_core=z_core), slice_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--L", type=float, default=64.0)
    ap.add_argument("--T", type=float, default=25.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--radii", type=float, nargs="*", default=list(RADII))
    ap.add_argument("--out", type=Path, default=Path("outputs/jr"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    grid = BoxGrid(N=args.N, L=args.L, dtype=jnp.float64)
    energy = make_energy(grid)

    results, end_slices = [], {}
    for R in args.radii:
        row, sl = run_radius(grid, energy, R, args.T, args.dt)
        results.append(row)
        end_slices[R] = sl
        print(f"R={R:4.1f}  U={row['U']:.3f}c  n_min={row['n_min']:.3f}  "
              f"W={row['winding']:+.1f}  -> {row['kind']}")

    vort = [r for r in results if r["kind"] == "vortex ring"]
    rare = [r for r in results if r["kind"] == "rarefaction pulse"]
    crossover = (max(r["U"] for r in vort) if vort else None,
                 min(r["U"] for r in rare) if rare else None)

    summary = dict(status="UNSCORED DEMO (census protocol DRAFT)",
                   grid=dict(N=args.N, L=args.L, dt=args.dt, T=args.T),
                   results=results,
                   crossover_bracket_U=crossover,
                   note="anchor unit test: GPE lore puts the JR crossover near "
                        "U ~ 0.63c; internal to the equation, not a measured "
                        "physical constant")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figure
    plt.rcParams.update(DARK_STYLE)
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.25)
    ax_r = np.asarray(grid.axis())
    ext = [ax_r[0], ax_r[-1], ax_r[0], ax_r[-1]]

    show = (args.radii[0], args.radii[-1])
    for col, R in enumerate(show):
        a = fig.add_subplot(gs[0, col])
        a.imshow(end_slices[R].T, origin="lower", extent=ext, cmap="magma",
                 vmin=0, vmax=1.3, aspect="equal")
        r = next(x for x in results if x["R"] == R)
        a.set_title(f"seed R={R:g}ξ → {r['kind']}   (t = {args.T:g})",
                    fontsize=10, color=C_BLUE if r["kind"] == "vortex ring"
                    else C_ORANGE)
        a.set_xlabel("x / ξ"); a.set_ylabel("z / ξ" if col == 0 else "")
        a.tick_params(labelsize=8)

    a = fig.add_subplot(gs[0, 2]); a.axis("off")
    lines = [f"UNSCORED DEMO — protocol DRAFT\n",
             f"{'R':>5} {'U/c':>7} {'n_min':>7} {'W':>4}  kind"]
    for r in results:
        lines.append(f"{r['R']:5.1f} {r['U']:7.3f} {r['n_min']:7.3f} "
                     f"{r['winding']:+4.0f}  {r['kind']}")
    lo, hi = crossover
    if lo and hi:
        lines.append(f"\ncrossover bracket: U ∈ ({lo:.2f}, {hi:.2f})c")
        lines.append("anchor (internal): JR lore ~0.63c")
    a.text(0, 0.95, "\n".join(lines), va="top", fontsize=10,
           color="#DDDDDD", linespacing=1.5)

    a = fig.add_subplot(gs[1, 0])
    a.plot([r["R"] for r in results], [r["U"] for r in results], "o-",
           color=C_BLUE, lw=2, ms=7)
    a.set_xlabel("seed radius R / ξ"); a.set_ylabel("measured speed U / c")
    a.set_title("smaller ring → faster object", fontsize=10)

    a = fig.add_subplot(gs[1, 1])
    for r in results:
        c = C_BLUE if r["kind"] == "vortex ring" else C_ORANGE
        a.plot(r["U"], r["n_min"], "o", color=c, ms=9)
    a.axhline(0.05, color="#666666", lw=1, ls=":")
    a.set_xlabel("U / c"); a.set_ylabel("final min density")
    a.set_title("core lift-off along the branch", fontsize=10)

    a = fig.add_subplot(gs[1, 2])
    for r in results:
        c = C_BLUE if r["kind"] == "vortex ring" else C_ORANGE
        a.plot(r["U"], abs(r["winding"]), "o", color=c, ms=9)
    a.set_xlabel("U / c"); a.set_ylabel("|winding|")
    a.set_title("topology: carried, then shed", fontsize=10)

    fig.suptitle("CENSUS — JONES–ROBERTS BRANCH HUNT: vortex ring → "
                 "rarefaction pulse", fontsize=14, color="white", y=0.99)
    out = args.out / "jr_branch.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
