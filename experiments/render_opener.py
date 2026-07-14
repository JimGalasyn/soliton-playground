#!/usr/bin/env python3
"""Render the bubble-vs-ring opener figure from outputs/opener/{summary.json,slices.npz}."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/opener")
S = json.loads((OUT / "summary.json").read_text())
Z = np.load(OUT / "slices.npz")
axis = Z["axis"]
times = sorted({float(k.split("_")[1]) for k in Z.files if k != "axis"})

# CVD-safe categorical pair (Okabe–Ito), fixed assignment: ring=blue, bubble=orange
C_RING, C_BUBBLE = "#56B4E9", "#E69F00"
plt.rcParams.update({
    "figure.facecolor": "black", "axes.facecolor": "black",
    "savefig.facecolor": "black", "text.color": "#DDDDDD",
    "axes.edgecolor": "#555555", "axes.labelcolor": "#BBBBBB",
    "xtick.color": "#999999", "ytick.color": "#999999",
    "font.family": "monospace", "axes.grid": False,
})

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.85], hspace=0.32, wspace=0.18)
ext = [axis[0], axis[-1], axis[0], axis[-1]]

for row, (obj, cmap) in enumerate([("ring", "magma"), ("bubble", "magma")]):
    for col, t in enumerate(times):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(Z[f"{obj}_{t}"].T, origin="lower", extent=ext,
                  cmap=cmap, vmin=0.0, vmax=1.3, aspect="equal")
        ax.set_title(f"{obj}   t = {t:g}", fontsize=11,
                     color=C_RING if obj == "ring" else C_BUBBLE)
        if col == 0:
            ax.set_ylabel("z / ξ")
        ax.set_xlabel("x / ξ" if row == 1 else "")
        ax.tick_params(labelsize=8)

# --- depletion volume
axv = fig.add_subplot(gs[2, 0])
for obj, c in [("ring", C_RING), ("bubble", C_BUBBLE)]:
    rows = S["series"][obj]
    axv.plot([r["t"] for r in rows], [r["V_dep"] for r in rows],
             color=c, lw=2, label=obj)
    axv.annotate(obj, (rows[-1]["t"], rows[-1]["V_dep"]), color=c,
                 fontsize=10, xytext=(-40, 8), textcoords="offset points")
axv.set_xlabel("t (ξ/c units)"); axv.set_ylabel("depletion volume (ξ³)")
axv.set_title("structure volume: protected vs unprotected", fontsize=11)
axv.legend(frameon=False, fontsize=9)

# --- calorimeter: potential (structure) energy converts to kinetic (sound)
axe = fig.add_subplot(gs[2, 1])
for obj, c in [("ring", C_RING), ("bubble", C_BUBBLE)]:
    rows = S["series"][obj]
    axe.plot([r["t"] for r in rows], [r["E_pot"] for r in rows],
             color=c, lw=2, label=f"{obj} E_pot")
    axe.plot([r["t"] for r in rows], [r["E_tot"] for r in rows],
             color=c, lw=1, ls="--", alpha=0.6)
axe.set_xlabel("t (ξ/c units)"); axe.set_ylabel("energy (GPE units)")
axe.set_title("ledger: E_pot solid, E_tot dashed", fontsize=11)
axe.legend(frameon=False, fontsize=9)

# --- verdict card
axt = fig.add_subplot(gs[2, 2]); axt.axis("off")
r, b = S["ring"], S["bubble"]
card = (
    f"UNSCORED DEMO — census protocol DRAFT\n\n"
    f"matched energy: {r['E']:.1f} vs {b['E']:.1f}\n\n"
    f"(ring panels show the mirror anti-ring at z>0:\n"
    f" the periodicity partner, not the entrant;\n"
    f" metrics are for the z<0 ring only)\n\n"
    f"RING   (W=1): {r['verdict']}\n"
    f"  winding {r['winding_t0']:+.1f} → {r['winding_T']:+.1f}\n"
    f"  volume ×{r['vol_ratio']:.2f}   travel Δz={r['travel_z']:+.1f}ξ\n"
    f"  ledger drift {r['ledger_drift']:.1e}\n\n"
    f"BUBBLE (W=0): {b['verdict']}\n"
    f"  R_b={b['R_b']:.2f}ξ   volume ×{b['vol_ratio']:.2f}\n"
    f"  ledger drift {b['ledger_drift']:.1e}\n\n"
    f"same energy. only the winding differs.\n"
    f"no external numbers were compared."
)
axt.text(0.0, 0.95, card, va="top", ha="left", fontsize=10.5,
         color="#DDDDDD", linespacing=1.5)

fig.suptitle("CENSUS OPENER — BUBBLE vs RING AT MATCHED ENERGY  "
             "(|ψ|² in the y=0 plane)", fontsize=14, color="white", y=0.98)
out = OUT / "opener_bubble_vs_ring.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(out)
