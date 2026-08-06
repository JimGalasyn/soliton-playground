#!/usr/bin/env python3
"""chamber — envelope preflight + stage validation + cut-flow (main/P-side).

The chamber POLICY that gates a run BEFORE it rents a host and accounts every
leg's fate. Ported to main from the C-side worktree as part of the farm
ownership transfer (all sims run through the chamber now); the NWT envelope
constants live ONCE in standard_box (P's authoritative copy) and this module
imports them — the port unifies the two copies rather than duplicating them.

  preflight(cfg)            -> [] if inside the spec-§3 walls, else typed drops
  validate_stage_plan(...)  -> [] if the staging geometry is admissible

These are the NWT POLICY plug-ins injected into jax_solitons.campaign.FarmCampaign
(the general cut-flow + governed-campaign machinery lives there now).
"""
from __future__ import annotations

import math

from .standard_box import (el_mag, alpha_max, seam_energy, min_separation,  # noqa: E402
                          THRESH, G2_R_MAX_FRAC, CORE_MIN_DX, SEAM_FLOOR)


def preflight(cfg: dict) -> list[str]:
    """cfg: {L, dx, lam, alpha, C, agrad, R_min, xi_c}. Returns the violated
    walls (empty = inside the envelope), each with the arithmetic that fired."""
    v = []
    em = el_mag(cfg["R_min"], cfg["C"])
    # THRESH is MEASURED per arm (el_probe_R.py, ±1%), so an arm nobody has probed
    # has no threshold and must not borrow one. Reporting the gap as a violation is
    # the honest reading: "we cannot certify this arm is inside its wall" is a
    # reason to stop, and it keeps --force-envelope as the single deliberate
    # override rather than letting an unmeasured arm sail through a check that
    # silently did not run. Inventing a number here would be worse than either.
    th = THRESH.get(cfg["agrad"])
    if th is None:
        v.append(f"expulsion: NO MEASURED THRESHOLD for agrad={cfg['agrad']!r} "
                 f"(have {sorted(THRESH)}); el/mag(R={cfg['R_min']},"
                 f"C={cfg['C']})={em:.0f} cannot be judged. Probe it with "
                 f"el_probe_R.py, or proceed deliberately.")
    elif em >= th:
        v.append(f"expulsion: el/mag(R={cfg['R_min']},C={cfg['C']})={em:.0f} "
                 f">= {th:.0f} ({cfg['agrad']} threshold)")
    am = alpha_max(cfg["dx"], cfg["lam"])
    if cfg["alpha"] > am:
        v.append(f"stability: alpha={cfg['alpha']:.1e} > "
                 f"alpha_max(dx={cfg['dx']},lam={cfg['lam']})={am:.1e}")
    if cfg["R_min"] > G2_R_MAX_FRAC * cfg["L"]:
        v.append(f"g2 ceiling: R={cfg['R_min']} > "
                 f"{G2_R_MAX_FRAC}*L={G2_R_MAX_FRAC * cfg['L']:.1f}")
    if cfg["xi_c"] < CORE_MIN_DX * cfg["dx"]:
        v.append(f"core resolution: xi_c={cfg['xi_c']} < "
                 f"2*dx={CORE_MIN_DX * cfg['dx']}")
    return v


def validate_stage_plan(plan: list[dict], L: float, certified_oids: set,
                        seam_tol: float = SEAM_FLOOR) -> list[str]:
    """plan: [{oid, pos:(x,y,z), orient, R}]. Certificate lineage, pairwise
    seam separations, per-object + collective g2 extent."""
    v = []
    for p in plan:
        if p["oid"] not in certified_oids:
            v.append(f"lineage: {p['oid']} has no certificate — staging "
                     "uncertified states is forbidden")
    dmin = min_separation(seam_tol)
    for i in range(len(plan)):
        for j in range(i + 1, len(plan)):
            a, b = plan[i], plan[j]
            d = math.dist(a["pos"], b["pos"])
            if d < a["R"] + b["R"]:
                v.append(f"overlap: {a['oid']}@{i}/{b['oid']}@{j} d={d:.1f} < "
                         f"R_a+R_b={a['R'] + b['R']:.1f} (product ansatz invalid)")
            elif d < dmin:
                v.append(f"seam: {a['oid']}@{i}/{b['oid']}@{j} d={d:.1f} < "
                         f"min_sep(tol={seam_tol:.0f})={dmin:.1f} "
                         f"(E_int={seam_energy(d):.0f})")
    if plan:
        for p in plan:
            if p["R"] > G2_R_MAX_FRAC * L:
                v.append(f"g2 ceiling: {p['oid']} R={p['R']} > {G2_R_MAX_FRAC * L:.1f}")
        cx = [sum(p["pos"][i] for p in plan) / len(plan) for i in range(3)]
        ext = max(math.dist(p["pos"], cx) + p["R"] for p in plan)
        if ext > 0.45 * L:
            v.append(f"box fit: extent {ext:.1f} from centroid > "
                     f"0.45*L={0.45 * L:.1f} (boundary margin)")
    return v
