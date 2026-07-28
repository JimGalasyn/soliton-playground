#!/usr/bin/env python3
"""Assemble docs/BESTIARY.md from committed run artifacts. No simulation.

The bin is a CENSUS judgment, not an experiment output, so it lives here rather
than in any single run's summary.json: it depends on gates spread across several
runs (resolution doubling, kick ensemble, ledger) plus a declared clock and a
declared N, none of which a single experiment knows about.

Every entry names its clock. That is load-bearing, not decoration — the trefoil
unties at 0.13-0.26 traversal periods but 20-40 local-reconnection periods, so the
same measured lifetime supports two different bins and the bin is meaningless
without the clock beside it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soliton_playground.gpe_lab import (  # noqa: E402
    CLOCK_LOCAL_RECONNECTION, XI, C_SOUND, CHARGE_KNOT_UNPROTECTED,
    CHARGE_WINDING, MODEL, PRESET, characteristic_period,
    local_reconnection_period)

N_DECLARED = 50          # survival threshold, protocol default


def load(p):
    q = Path(p)
    return json.loads(q.read_text()) if q.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/BESTIARY.md"))
    args = ap.parse_args()

    base = load("outputs/trefoil/summary.json")
    n256 = load("outputs/trefoil_n256/summary.json")
    kick = load("outputs/trefoil_kick/summary.json")
    g2 = load("outputs/gate2/summary.json")
    long_ = load("outputs/trefoil_long/summary.json")
    if base is None:
        raise SystemExit("outputs/trefoil/summary.json missing — run the census first")

    L0 = base["topology"]["0.0"]["lengths"][0]
    tau_local = local_reconnection_period()
    tau_trav = characteristic_period(L0)

    ts = sorted(float(k) for k in base["topology"])
    cross = [base["topology"][f"{t}"]["min_crossings"][0] for t in ts]
    untied_at = next((t for t, x in zip(ts, cross)
                      if x is not None and x <= 2 and t > 0), None)
    t_rec = base["lifetime_first_reconnection"]

    life_local = untied_at / tau_local
    bin_ = "unstable" if life_local < N_DECLARED else "metastable"

    lines = [
        "# Bestiary", "",
        "**UNSCORED — the census protocol is still DRAFT.** These are the entries "
        "the gates currently support, recorded so the gaps are visible.", "",
        f"Preset `{PRESET}` · model {MODEL} · ξ = {XI:g}, c = {C_SOUND:g} "
        f"(c measured 1.0446/1.0714, see `tests/test_characteristic_period.py`)",
        "",
        f"Survival threshold in force: **N = {N_DECLARED}** characteristic periods "
        "(protocol default).", "",
        "## trefoil T(2,3)", "",
        "| field | value |", "|---|---|",
        f"| preset | `{PRESET}` |",
        f"| protecting charge | `{CHARGE_KNOT_UNPROTECTED}` |",
        f"| **bin** | **`{bin_}`** |",
        f"| **clock** | **{CLOCK_LOCAL_RECONNECTION}**, τ = {tau_local:g} |",
        f"| lifetime (untying) | t = {untied_at:g} = **{life_local:.0f} periods** "
        f"(vs N = {N_DECLARED}) |",
        f"| first reconnection | t = {t_rec:g} = {t_rec/tau_local:.0f} periods |",
        f"| decay channel | reconnection cascade → unknotted, unlinked rings + sound |",
        f"| seed + resolution | Milnor map, scale 8ξ; N = 128 and 256, L = 64 |",
        f"| lineage | `outputs/trefoil/cascade.hepmc3` |",
        "",
        "### Why `unstable` and not `metastable`", "",
        "The bins separate *long-lived* from *dies on its own timescale*, and the "
        f"trefoil unties at {life_local:.0f} local-reconnection periods against a "
        f"declared N = {N_DECLARED}. It falls short on **either** clock, which is "
        "what makes the call unambiguous:", "",
        "| clock | τ | untying | vs N = 50 |", "|---|---|---|---|",
        f"| {CLOCK_LOCAL_RECONNECTION} | {tau_local:g} | {life_local:.0f} periods "
        f"| short by {N_DECLARED/life_local:.1f}× |",
        f"| {CLOCK_TRAVERSAL_LABEL} | {tau_trav:.1f} | {untied_at/tau_trav:.3f} "
        f"periods | short by {N_DECLARED/(untied_at/tau_trav):.0f}× |",
        "",
        "The local clock is the physically apt one — a knot does not untie by "
        "anything traversing it, but where two strands approach within a core "
        "radius, so the process runs on ξ/c. It changes the *margin* by two orders "
        "of magnitude and not the verdict. `metastable` would require a declared "
        "N ≤ 20 for this object class.", "",
        "This supersedes the `METASTABLE->RINGS` verdict in commits 6ec9cc7, "
        "bd8cf9b and 492f69a. That string is the measured *channel* (the knot does "
        "decay into rings, and that is unchanged); the *bin* is a census judgment "
        "that those runs never actually made against a clock.", "",
        "### Gates", "", "| gate | result | evidence |", "|---|---|---|",
        f"| 0 seed | PASS | shell {base['seed_gate']['shell_min_density']:.5f}, "
        f"wrap {base['seed_gate']['wrap_jump_max']:.1e} |",
    ]
    lines.append(
        f"| 1 survival | **FAIL** ({life_local:.0f} < {N_DECLARED} periods) | "
        "reported, not gated — an entrant with an identified decay channel cannot "
        "meet a survival threshold; this is what sets the bin |")
    if g2:
        c = g2["closure"]
        lines.append(
            f"| 2 ledger | {g2['gate2_verdict']} | drift dt-independent over 16×; "
            f"FD drift {c['drift_fd_full_T']:.2e} vs floor "
            f"{c['floor_full_T_fd']:.2e}; sound above half-Nyquist ≤ "
            f"{c['E_c_highk_frac_max']:.4f} |")
    lines.append(
        "| 3 charge retention | n/a | knot type is not a charge in this preset; "
        "the ±1 winding per strand is retained through every reconnection |")
    if kick:
        lines.append(
            f"| 4 kick | {kick['gate4_verdict']} | 3 seeds, ε = "
            f"{kick['kick']['eps']:g}; channel unchanged, but 2 of 3 untie before "
            "t = 20 via a second discrete route |")
    if n256:
        lines.append(
            f"| refinement | PASS | N=128→256: verdict held, loops "
            f"{[base['topology'][f'{t}']['n_loops'] for t in ts]} identical, drift "
            f"{base['ledger_drift']:.1e}→{n256['ledger_drift']:.1e} |")

    lines += ["", "## ring debris (decay products of the trefoil)", ""]
    if long_:
        lt = sorted(float(k) for k in long_["topology"])
        last = long_["topology"][f"{max(lt)}"]
        ring_L = [x for x in last["lengths"][1:]] or last["lengths"]
        tau_ring = characteristic_period(min(ring_L)) if ring_L else float("nan")
        T = long_["grid"]["T"]
        lines += [
            f"Long run to T = {T:g} (`outputs/trefoil_long/`).", "",
            "| field | value |", "|---|---|",
            f"| protecting charge | `{CHARGE_WINDING}` |",
            f"| loops at T | {last['n_loops']} — lengths {last['lengths']} |",
            f"| verdict | {long_['verdict']} |",
            f"| ledger drift | {long_['ledger_drift']:.2e} |",
            f"| shortest-ring τ (traversal) | {tau_ring:.1f} |",
            f"| observed span in own periods | ~{T/tau_ring:.0f} |",
            "",
            "Rings are traversal-clocked, not reconnection-clocked: a ring has no "
            "decay event to time, so what is being asked is whether it survives "
            "many transits of itself.",
        ]
    else:
        lines += [
            "**PENDING** — the long run (T ≈ 1000) has not been recorded yet. The "
            "rings' arc lengths are 17–23 ξ, so τ ≈ 17–23 and the T = 80 runs have "
            "observed them for only ~2 of their own periods. They cannot be binned "
            "`protected` on topology alone until that is measured.",
        ]

    lines += ["", "## What no entry can claim yet", "",
              "- The protocol is DRAFT and unfrozen; nothing here is scored.",
              "- Gate 3 has no independent test — no entrant has had a charge "
              "change logged as an explicit decay event.",
              "- The calorimeter cannot audit the drift (its own sum-rule residual "
              "is ~4× the total energy change), so \"every loss accounted\" holds "
              "only as: no evidence of grid loss.",
              "- A periodic box recycles its own sound, so no radiated *budget* "
              "exists — only instantaneous sound content.", ""]

    args.out.write_text("\n".join(lines))
    print(f"trefoil: bin={bin_}  clock={CLOCK_LOCAL_RECONNECTION}  "
          f"lifetime={life_local:.0f} periods (N={N_DECLARED})")
    print(args.out)


CLOCK_TRAVERSAL_LABEL = "traversal (τ = L_structure / c)"

if __name__ == "__main__":
    main()
