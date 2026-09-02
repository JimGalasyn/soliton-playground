#!/usr/bin/env python3
"""STANDARD BOX (SB-1) battery — conformance gate for the `ehn-two-scalar` preset.

Moved here from `null-worldtube-private/simulations/engine_dogfood/` on
2026-08-01 (that repo is deprecated; see its EXTRACTION_DECISIONS.md). The engine
it drives moved too, to `jax_solitons.ehn`, so this file now spans two repos: the
battery is soliton-playground's, the physics is jax-solitons'. `engine_sha()`
records both.

Its normative spec, `analysis/STANDARD_BOX_SPEC.md`, was C-side on
more-cosmogenesis and is NOT on this machine or in any repo here -- the section
references below (§2-§6) point at a document that did not come across. The
constants are therefore the authoritative copy by default rather than by design.

What this file is:
  SB1            — the frozen §2 reference configuration (the de facto
                   standard every held-knot artifact already lives in).
  ENVELOPE       — the §3 walls as arithmetic (P's authoritative copy;
                   rider 1 = these constants confirmed against this engine).
  BATTERY        — B1-B7 as runnable legs. Each leg launches the relaxer as
                   a SUBPROCESS via `-m jax_solitons.ehn.relax` (exactly how
                   every archived artifact was
                   made — and the wrapped/bilinear AGRAD module global is
                   jit-trace-frozen per process, so mixed-agrad legs in one
                   process would silently reuse the wrong kernels).
  CERTIFICATE    — SHA-keyed conformance record; any engine edit invalidates
                   (engine_sha = hash over the engine blobs, listed).

Battery legs (spec §4; criteria = constants below, calibrated against the
archived reference runs cited next to each):
  B1 Hopf dual-meter  T(1,1) R=22.5: wrapped arm holds Lk=-1 + charge;
                      bilinear control arm keeps the GEOMETRY but the
                      CHARGES die. Both behaviors must reproduce — the
                      discrimination IS the instrument.
  B2 trefoil hold     T(2,3) R=0.22L, 12k battery version of the 36k
                      certified hold: Lk=-3.000, det=3, no shrink trend.
  B3 paired parity    the +Lk arm is seeded --tq -3 (the EXACT mirror
                      curve: cos even / sin odd flips the meridian sense);
                      both signs held, |Lk+ + Lk-| small (RT-5 made standard).
  B4 charge retention Q, link fraction, el within stated bands at leg end
                      (evaluated on the B2/B3 artifacts; bands from the
                      trefoil_t23 reference trajectory).
  B5 tracer conformance  CPU: every catalog entry re-measured from its
                      field.npz must reproduce entry["measured"]; plus the
                      sub-grid-offset guard (an exactly-grid-aligned IC has
                      a degenerate winding detector — read +25% high before
                      the off-nudge; the guard rebuilds a small IC and
                      demands the tracer reads the seeded Lk).
  B6 stability margin B2 trajectory: no NaN, post-ramp E monotone; then
                      resumes at 2x, 2.5x and 4x alpha, PASSING at >=2x. The 4x
                      (EHN's own 4e-4) was a recorded hypothesis, not a
                      requirement -- and is now measured FALSE: see below. 2.5x
                      is probed so measured_headroom can express the 2x-2.5x
                      resolution instead of quantising to {0, 2, 4}.
  B7 map fidelity     (conditional, --with-m1) M1 magnification phi(x) ->
                      phi(x/s): per-realization Lk + component count
                      preserved; lost-loop fraction receipted.

Reference calibration (archived runs, this engine, N=192):
  wrapped  trefoil 12k: Q=-2.31 link=-25% el=254 Lk=-3.000 det=3
  bilinear trefoil 12k: Q=-0.010 link=-7% el=27  Lk=-3.0 (geometry held,
                        charges dead)                    [out_trefoil_t23_n192*]
  36k hold: 978 skeleton segments bit-stable 12k->36k (zero shrink).

Alpha headroom, MEASURED 2026-08-02 -- B6's "spec_hypothesis_4x" resolves FALSE.
Offline sweep, 200 STEPS from a settled trefoil (N=192 dx=0.8 lam=1000); B6's own
probes run 500, which matters for the marginal row and only that row:
  2.00e-4 (2x)    E=3333.3 Lk=-3.0   survives
  2.05e-4         E=3334.3 Lk=-3.0   survives
  2.50e-4 (2.5x)  E=3332.1 Lk=-3.0   survives, MARGINAL
  3.00e-4 (3x)    E=7.2e8            diverges
  4.00e-4 (4x)    NaN                diverges
So the measured headroom is 2x-2.5x, and B6's >=2x pass criterion is the right
one; the >=4x in the original spec was never achievable. The cliff sits where
the leg's own comment predicted, 2/(8*lam) = 2.5e-4 -- 2.5e-4 surviving is the
marginal case (|1-alpha*H| = 1), not a refutation. This is the rider-1
correction the leg was written to trigger.

The 2.5x row is the ONLY one whose verdict can depend on the horizon: at
alpha = 2/H exactly, |1-alpha*H| = 1 is neutral growth, so it neither decays nor
explodes and 200 steps does not settle it. The 2x and 4x rows are decaying and
exploding respectively and are horizon-insensitive. B6 therefore probes 2.5x
directly at its own 500 steps rather than inheriting this number -- if a
certificate reports measured_headroom 2.0 where this table says 2.5, the
certificate is right and the difference IS the horizon effect.

NOTE for any refinement below dx ~ 0.14: 8*lam is the potential-dominated
plateau only. The Hessian grows as 1/dx^2 once the gradient and gauge terms take
over -- H ~ max(8*lam, 153/dx^2), the branches crossing at dx = 0.14 rather than
the 0.11 an analytic 2U/dx^2 = 8*lam predicts, since the measured coefficient is
153 and not 2U = 100. So the cliff MOVES: 1.30e-4 at dx=0.1, 3.27e-5 at dx=0.05,
and SB-1's own alpha=1e-4 would be unsafe below dx = 0.088. SB-1 is fixed at
dx=0.8 and nothing here refines, but a successor box might; see
jax_solitons.ehn.relax's module docstring for the table.

  python -m soliton_playground.ehn_lab.standard_box --battery --quick   # N=96 smoke
  python -m soliton_playground.ehn_lab.standard_box --battery           # SB-1 cert
  python -m soliton_playground.ehn_lab.standard_box --battery --legs B5 # CPU-only
  python -m soliton_playground.ehn_lab.standard_box --envelope R=22.5 C=400
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
# The two sys.path inserts that stood here are gone: the engine is an installed
# package (jax_solitons.ehn) and the catalog a sibling module, so nothing needs
# the importing file's directory on the path any more.

BATTERY_VERSION = "v1"

# ================= SB-1 (spec §2, frozen) =================
# The configuration every held-knot artifact lives in. Battery legs derive
# their command lines from THIS dict — edit here = new certificate.
SB1 = {
    "box": "SB-1", "N": 192, "L": 153.6, "dx": 0.8,
    "lam": 1000.0, "kappa": 8e-4, "C": 400.0, "U": 50.0, "eps_a": 0.05,
    "alpha": 1e-4, "beta": 2e-3, "q1": 1.0, "q2": 0.0,
    "agrad": "wrapped", "ic": "screened", "cramp": 8000, "core": 2.0,
}

# Engine blobs whose edit invalidates a certificate (SHA-keyed).
#
# The both-layouts `_engine_file()` resolver that used to live here is GONE, and
# so is the failure it patched around: the engine is an installed package now, so
# `import jax_solitons` resolves identically in a checkout, in a venv, and on a
# rented box. That resolver existed because gpe_vortex_topology.py sat in a
# sibling directory locally and flat in /workspace remotely, and its absence from
# this file is what made the first rented B2 attempt exit 1 before running a leg.
# A packaging problem was being solved with path arithmetic. That history now lives
# with the code, in `soliton_playground.provenance`.
#
# PROVENANCE MOVED THERE so BOTH wings can stamp a result. The census wing
# (gpe_lab) had no code provenance at all until 2026-09-02: every entrant,
# cascade and quench manifest in `output/` was produced by whichever
# jax-solitons working tree happened to be checked out, unrecorded. The discipline
# was here, in this file, and stayed here because it lived in one battery instead
# of in an instrument. These names are kept as aliases: the battery is what every
# archived certificate was made by, and renaming its internals buys nothing.
from soliton_playground.provenance import (             # noqa: E402
    engine_files as _engine_files, engine_sha as _engine_sha)


# ================= ENVELOPE (spec §3, P's authoritative copy) ==========
# Same constants as C's chamber.py — they unify at merge; rider 1 is P
# confirming THESE numbers against the as-committed engine.
EL_MAG_REF, R_REF, C_REF = 2349.0, 14.0, 400.0     # el_probe_R.py, ±1%
THRESH = {"wrapped": 147.0, "bilinear": 37.0}
ALPHA_REF, DX_REF, LAM_REF = 1e-4, 0.8, 1000.0
SEAM_E0, SEAM_D0, SEAM_LAMBDA = 7408.0, 60.0, 2.62  # two-proton wall fit
SEAM_FLOOR = 75.0
G2_R_MAX_FRAC = 0.35
CORE_MIN_DX = 2.0


def el_mag(R, C):
    return EL_MAG_REF * (R_REF / R) ** 2 * (C / C_REF) ** 2


def alpha_max(dx, lam):
    return ALPHA_REF * (dx / DX_REF) ** 2 * (LAM_REF / lam)


def seam_energy(d):
    return SEAM_E0 * math.exp(-(d - SEAM_D0) / SEAM_LAMBDA)


def min_separation(tol=SEAM_FLOOR):
    return SEAM_D0 + SEAM_LAMBDA * math.log(SEAM_E0 / tol)


# ================= acceptance bands (calibrated, see header) ============
BANDS = {
    "lk_tol": 0.05,          # |Lk - integer target| at leg end
    "lk_pair_tol": 0.10,     # |Lk(+arm) + Lk(-arm)| — chirality artifact
    "q_retained_frac": 0.50, # wrapped: |Q_end| >= frac * |Lk target|
    "q_dead_abs": 0.30,      # bilinear control: |Q_end| <= this = charges died
    #   (differential form also allowed: |Q_b| <= 0.3|Q_w|; ref kill timeline
    #    bilinear trefoil Q -2.7 -> -0.25 by 8k steps, -0.01 by 12k)
    "link_retained_pct": 15.0,   # B4 wrapped trefoil: |link%| at end (ref 25%)
    "el_band": (20.0, 800.0),    # B4: electric energy at leg end (ref 254)
    "shrink_frac": 0.85,     # skeleton segments end >= frac * post-ramp
    "e_uptick_rel": 1e-3,    # post-ramp monotonicity slack per sample
    "headroom_min": 2.0,     # B6: must survive >=2x alpha (4x is measured)
    "m1_lost_loop_max": 0.0, # B7: components lost by the map (Lk-carrying)
}


# ================= engine SHA / certificate =================

def engine_sha(explicit_commit=None):
    """Engine provenance as a git commit — see `soliton_playground.provenance`.

    MOVED, not changed in shape: the certificate body, C's `chamber.py` format and
    the cid formula are untouched, and this stays the name the rest of the battery
    calls. What moved with it is a defect fix the census wing forced into the open —
    a wheel install used to be git-stamped with the ENCLOSING repo's SHA, because
    `.venv` sits inside this working tree. It never fired here (`jax-solitons` is
    installed editable) and it could not have: the branch has no test case in this
    repo's install mix. See that module's header, and `tests/test_provenance.py`,
    which manufactures the condition from numpy rather than waiting for a port.
    """
    return _engine_sha(explicit_commit)

def issue_certificate(battery_version, engine_sha_, box, results):
    """C's chamber.py issue_certificate, verbatim (format compatibility)."""
    verdict = "PASS" if all(results.values()) else "FAIL"
    body = {"battery": battery_version, "engine_sha": engine_sha_,
            "box": box, "results": results, "verdict": verdict}
    cid = "cert_" + hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
    return {"cid": cid, **body}


# ================= leg plumbing =================

def _relax_cmd(out, *, geom_args, steps, samples=24, agrad=None, alpha=None,
               resume=None, box=None, save_every=None, topo_every=1):
    b = box or SB1
    # `-m jax_solitons.ehn.relax`, not a file path: the engine is an installed
    # package, so this resolves the same in a checkout, a venv and on a rented box
    # -- and the leg no longer depends on the engine sitting beside this file.
    cmd = [sys.executable, "-m", "jax_solitons.ehn.relax",
           "--N", str(b["N"]), "--L", str(b["L"]), "--core", str(b["core"]),
           "--lam", str(b["lam"]), "--kappa", str(b["kappa"]),
           "--C", str(b["C"]), "--U", str(b["U"]), "--eps-a", str(b["eps_a"]),
           "--alpha", str(alpha if alpha is not None else b["alpha"]),
           "--beta", str(b["beta"]), "--ic", b["ic"],
           "--agrad", agrad or b["agrad"], "--cramp", str(b["cramp"]),
           "--steps", str(steps), "--samples", str(samples),
           "--topo-every", str(topo_every),
           # Checkpoint FOUR times per leg, not once at the end. The old default
           # (save_every=steps) wrote field.npz only on the final step, so a host
           # dying at step 11000 of 12000 left resumable=True with a log and a
           # manifest and NO FIELD -- nothing scoreable, the whole arm lost. A
           # mid-run fetch can only rescue what the remote job has written.
           # ehn_relax.py already supports --resume from these.
           "--save-every", str(save_every if save_every else max(1, steps // 4)),
           "--out", str(out)]
    if resume:
        cmd += ["--resume", str(resume)]
    else:
        cmd += geom_args
    return cmd


# Legs that failed to EXECUTE, as distinct from legs whose physics missed the
# criteria. Reset per battery run; read back in run_battery to decide citability.
# Without this split a leg that OOMed was indistinguishable in the certificate
# from a leg that ran clean and failed the physics: cert_71ce0966fc63_full.json
# recorded verdict=FAIL with citable=TRUE after both B2 arms died at wall_s=4.8
# on a 3.16 GiB allocation, and it now sits beside a genuine PASS with nothing to
# tell them apart.
_LEG_ERRORS = []

_CRASH_SIGNATURES = (
    ("oom", ("RESOURCE_EXHAUSTED", "Out of memory", "ran out of memory",
             "OutOfMemoryError", "Killed")),
    ("nan", ("NaN", "not finite")),
    ("import", ("ModuleNotFoundError", "ImportError")),
    ("traceback", ("Traceback (most recent call last)",)),
)


def _classify_crash(log_path):
    """Best-effort cause tag from a failed leg's log, so the certificate says WHY
    rather than just FAIL. Ordered: the first match wins, so OOM outranks the
    generic traceback it inevitably also produced."""
    try:
        text = log_path.read_text(errors="replace")[-20000:]
    except OSError:
        return "no_log", ""
    for tag, needles in _CRASH_SIGNATURES:
        for n in needles:
            if n in text:
                return tag, next((ln.strip() for ln in reversed(text.splitlines())
                                  if n in ln), n)[:300]
    return "unknown", text.strip().splitlines()[-1][:300] if text.strip() else ""


def _run_leg(name, cmd, outdir, reuse=True):
    """Run a leg, or REUSE an existing manifest.

    Reuse makes the battery resumable and, more importantly, enables the split the
    Vast RUNBOOK prescribes: "all knot ID / census stays local". The heavy relax
    (GPU, 24 GB at N=192) runs on a rented box; its out dir is fetched; and this
    scoring pass finds the manifest, skips the subprocess, and does the
    determinant work locally where pyknotid lives. Without reuse the scoring pass
    would re-run the relaxation it was handed.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    if reuse and (outdir / "manifest.json").exists():
        print(f"  [{name}] REUSE existing manifest (no re-run)", flush=True)
        return True
    log = outdir.parent / f"{name}.log"
    print(f"  [{name}] {' '.join(cmd[1:3])} ... -> {outdir}", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                           cwd=str(_HERE))
    dt = time.time() - t0
    has_manifest = (outdir / "manifest.json").exists()
    ok = r.returncode == 0 and has_manifest
    if not ok:
        cause, evidence = _classify_crash(log)
        _LEG_ERRORS.append({"leg": name, "returncode": r.returncode,
                            "manifest_written": has_manifest, "cause": cause,
                            "evidence": evidence, "wall_s": round(dt, 1),
                            "log": str(log)})
        print(f"  [{name}] FAILED ({dt:.0f}s) cause={cause}: {evidence[:120]}",
              flush=True)
    else:
        print(f"  [{name}] done ({dt:.0f}s)", flush=True)
    return ok


def _measure_end(outdir, box):
    """Post-hoc topology of the leg's final field: cross-Lk, per-field
    dets/components, skeleton size, plus the manifest trajectory."""
    from jax_solitons.ehn.cross_linking import cross_linking
    from jax_solitons.vortex_topology import knot_determinants, vortex_skeleton
    man = json.loads((outdir / "manifest.json").read_text())
    d = np.load(outdir / "field.npz")
    u = d["u"]
    p1 = u[0] + 1j * u[1]
    p2 = u[2] + 1j * u[3]
    dx = box["L"] / box["N"]
    lk, ns1, ns2 = cross_linking(p1, p2, dx)
    dets = knot_determinants(p1, dx, box["L"])
    P1, _, _ = vortex_skeleton(p1)
    return {"manifest": man, "lk": float(lk), "nseg1": int(ns1),
            "nseg2": int(ns2), "dets": [[int(s), det if isinstance(det, str)
                                         else int(det)] for s, det in dets],
            "skeleton_pts": int(len(P1)),
            "field_sha": hashlib.sha256(
                (outdir / "field.npz").read_bytes()).hexdigest()[:16]}


def _traj_end(man):
    return man["traj"][-1] if man.get("traj") else {}


def _no_reconnect(man, lk_target):
    """Every in-run topo sample within 0.5 of target once past the C-ramp
    (pre-ramp the tracer sees the still-annealing IC) — a jump = reconnect."""
    cramp = man["params"].get("cramp", 0)
    for t in man.get("traj", []):
        if t["n"] >= cramp and t.get("xlk") is not None:
            if abs(t["xlk"] - lk_target) > 0.5:
                return False, t
    return True, None


def _monotone_post_ramp(man, slack_rel):
    cramp = man["params"].get("cramp", 0)
    es = [(t["n"], t["total"]) for t in man.get("traj", [])
          if t["n"] >= cramp]
    if any(not np.isfinite(e) for _, e in es):
        return False, "NaN/inf in trajectory"
    for (n0, e0), (n1, e1) in zip(es, es[1:]):
        if e1 > e0 * (1 + slack_rel):
            return False, f"E rose {e0:.1f}->{e1:.1f} at n={n1}"
    return True, None


# ================= the legs =================

def leg_B1(outroot, box, steps):
    """Hopf dual-meter: wrapped holds charge, bilinear control loses it while
    the geometry survives. Returns (results-dict, detail)."""
    R = 22.5 * box["L"] / SB1["L"]      # scale the reference R with the box
    geom = ["--geom", "torus", "--tp", "1", "--tq", "1", "--R", f"{R}"]
    det_ = {}
    res = {}
    for arm, agrad in (("wrapped", "wrapped"), ("bilinear", "bilinear")):
        out = outroot / f"B1_{arm}"
        ok = _run_leg(f"B1_{arm}", _relax_cmd(
            out, geom_args=geom, steps=steps, agrad=agrad, box=box), out)
        if not ok:
            res[f"B1_{arm}"] = False
            continue
        m = _measure_end(out, box)
        end = _traj_end(m["manifest"])
        floor = (2 * math.pi) ** 2 * 1
        linkpct = abs(end.get("link", 0.0)) / floor * 100
        m["end"] = {"Q": end.get("Q"), "link_pct": linkpct,
                    "el": end.get("elec")}
        det_[arm] = m
        if arm == "wrapped":
            noreco, bad = _no_reconnect(m["manifest"], -1.0)
            res["B1_wrapped"] = (
                abs(m["lk"] - (-1.0)) <= BANDS["lk_tol"]
                and abs(end.get("Q", 0)) >= BANDS["q_retained_frac"] * 1.0
                and noreco)
            m["no_reconnect"] = noreco
            if bad:
                m["reconnect_sample"] = bad
    # control is judged DIFFERENTIALLY vs the wrapped arm (link%/Q scales
    # are target-dependent; the discrimination is wrapped-vs-bilinear)
    if "wrapped" in det_ and "bilinear" in det_:
        qw = abs(det_["wrapped"]["end"].get("Q") or 0)
        qb = abs(det_["bilinear"]["end"].get("Q") or 1)
        res["B1_bilinear_control"] = (
            abs(det_["bilinear"]["lk"]) >= 0.5          # geometry survives
            and qb <= max(BANDS["q_dead_abs"], 0.3 * qw))   # charges die
    return res, det_


def leg_B2_B3_B4_B6(outroot, box, steps, headroom_steps):
    """Trefoil hold (B2), mirror parity arm (B3), charge bands (B4),
    stability + alpha headroom (B6) — one pair of relax legs, four verdicts."""
    R = 0.22 * box["L"]
    res, det_ = {}, {}
    arms = {}
    for arm, tq, lk_t in (("neg", "3", -3.0), ("pos", "-3", +3.0)):
        geom = ["--geom", "torus", "--tp", "2", "--tq", tq, "--R", f"{R}"]
        out = outroot / f"B2_trefoil_{arm}"
        ok = _run_leg(f"B2_{arm}", _relax_cmd(
            out, geom_args=geom, steps=steps, box=box), out)
        if not ok:
            res["B2_trefoil_hold" if arm == "neg" else "B3_paired_parity"] = False
            continue
        m = _measure_end(out, box)
        m["end"] = _traj_end(m["manifest"])
        arms[arm] = m
        det_[f"trefoil_{arm}"] = m

    if "neg" in arms:
        m = arms["neg"]
        man = m["manifest"]
        # no-shrink: skeleton-linked topo samples post-ramp
        segs = [(t["n"], t["nseg1"]) for t in man["traj"]
                if t.get("nseg1") and t["n"] >= man["params"]["cramp"]]
        shrink_ok = (len(segs) < 2 or
                     segs[-1][1] >= BANDS["shrink_frac"] * segs[0][1])
        one_comp_det3 = (len(m["dets"]) == 1 and m["dets"][0][1] == 3)
        res["B2_trefoil_hold"] = (
            abs(m["lk"] - (-3.0)) <= BANDS["lk_tol"]
            and one_comp_det3 and shrink_ok)
        m["no_shrink"] = {"ok": shrink_ok, "segs": segs[:1] + segs[-1:]}

        # B4 on the held arm (bands are SB-1/N=192-calibrated: in a smoke
        # box B4 is recorded but not verdict-bearing)
        end = m["end"]
        floor = (2 * math.pi) ** 2 * 3
        linkpct = abs(end.get("link", 0.0)) / floor * 100
        el = end.get("elec", 0.0)
        b4 = (abs(end.get("Q", 0)) >= BANDS["q_retained_frac"] * 3.0 * 0.5
              # Q reads low on the grid mid-relax (ref -2.31 of -3 at 12k,
              # self-heals to -2.87 by 36k): band = 50% of the wrapped floor
              and linkpct >= BANDS["link_retained_pct"]
              and BANDS["el_band"][0] <= el <= BANDS["el_band"][1])
        smoke = "SMOKE" in box["box"]
        det_["B4"] = {"Q": end.get("Q"), "link_pct": linkpct, "el": el,
                      "bands_met": bool(b4),
                      "note": "informational in smoke box" if smoke else ""}
        if not smoke:
            res["B4_charge_retention"] = b4

        # B6: monotone post-ramp + MEASURED alpha headroom. Probe 2x, 2.5x and
        # 4x from the relaxed state; criterion = survives >=2x (not at the
        # cliff); the measured factor goes in the certificate. Gradient-
        # descent stability predicts the cliff near 2/(8*lam) = 2.5e-4 at
        # lam=1000.
        #
        # RESOLVED FALSE 2026-08-02 -- the spec's ">=4x" was the hypothesis this
        # leg was built to measure, and 4x diverges at SB-1 scale (E=7.2e8 at 3x,
        # NaN at 4x). The rider-1 correction is filed in this module's header
        # with the full table; read it before re-opening the question.
        #
        # 2.5x IS PROBED, not just bracketed, because the header claims a
        # resolution of 2x-2.5x and a (2.0, 4.0) tuple can only ever emit
        # measured_headroom in {0.0, 2.0, 4.0} -- so every certificate would
        # read 2.0 and fail to corroborate the header it is supposed to
        # evidence. 2.5x is also the one probe whose outcome is horizon-
        # dependent by construction: at alpha = 2/H exactly, |1-alpha*H| = 1 is
        # NEUTRAL growth, so it may survive the 200-step offline sweep and fail
        # this leg's 500. That is a real measurement, not an inconsistency.
        mono, why = _monotone_post_ramp(man, BANDS["e_uptick_rel"])
        fld = outroot / "B2_trefoil_neg" / "field.npz"
        total = steps + headroom_steps
        # samples: ehn_relax strides every=steps//samples over the FULL
        # range — pick a stride landing >=4 samples in the resumed window
        nsamp = max(4, total // max(1, headroom_steps // 4))
        headroom = 0.0
        probes = {}
        for mult in (2.0, 2.5, 4.0):
            out = outroot / f"B6_headroom_{mult:g}x"
            ok = _run_leg(f"B6_{mult:g}x", _relax_cmd(
                out, geom_args=[], steps=total, samples=nsamp,
                alpha=mult * box["alpha"], resume=fld, box=box,
                topo_every=0), out)
            surv = False
            if ok:
                hman = json.loads((out / "manifest.json").read_text())
                hs = [t["total"] for t in hman["traj"]]
                surv = (len(hs) >= 2 and all(np.isfinite(e) for e in hs)
                        and hs[-1] <= hs[0] * (1 + BANDS["e_uptick_rel"]))
                probes[f"{mult:g}x"] = {"survived": surv,
                                        "E": hs[:1] + hs[-1:]}
            if surv:
                headroom = mult
            else:
                break
        det_["B6"] = {"monotone": mono, "why": why,
                      "measured_headroom": headroom, "probes": probes,
                      "spec_hypothesis_4x": headroom >= 4.0}
        res["B6_stability_margin"] = mono and headroom >= 2.0

    if "pos" in arms and "neg" in arms:
        mp, mn = arms["pos"], arms["neg"]
        res["B3_paired_parity"] = (
            abs(mp["lk"] - 3.0) <= BANDS["lk_tol"]
            and abs(mp["lk"] + mn["lk"]) <= BANDS["lk_pair_tol"]
            and len(mp["dets"]) == 1 and mp["dets"][0][1] == 3)
        det_["B3"] = {"lk_pos": mp["lk"], "lk_neg": mn["lk"],
                      "sum": mp["lk"] + mn["lk"]}
    return res, det_


def leg_B5(outroot, box):
    """Tracer conformance (CPU + one small IC build): every catalog entry
    re-measured must reproduce its registration; the sub-grid-offset guard
    rebuilds a small T(2,3) IC and demands the tracer read the seeded Lk."""
    from .particle_catalog import CATALOG, _measure
    det_ = {"catalog": {}, "ic_guard": None}
    ok_all = True
    unconformed = []
    entries = sorted(CATALOG.glob("*/entry.json"))
    for p in entries:
        e = json.loads(p.read_text())
        if not (p.parent / "field.npz").exists():
            # Pull it from the content-addressed store if held there, so a machine
            # that HAS the states conforms against them without anyone remembering
            # to stage files by hand.
            try:
                from . import field_store
                field_store.materialize(e["name"])
            except Exception:                             # noqa: BLE001
                pass
        if not (p.parent / "field.npz").exists():
            # NOT a skip. This leg's claim is "every catalog entry re-measured
            # reproduces its registration"; an entry that cannot be re-measured
            # leaves that claim unsupported. `continue` used to leave ok_all True,
            # so with all ten fields absent this printed
            # B5_tracer_conformance: True while conforming against NOTHING --
            # the same shape as the done_when trap and as citable=TRUE on a FAILED
            # certificate. Absence of evidence is now recorded as absence.
            det_["catalog"][e["name"]] = "UNCONFORMED (no field.npz, none in store)"
            unconformed.append(e["name"])
            continue
        m = _measure(p.parent / "field.npz")
        ref = e["measured"]
        good = (abs(m["Lk_phi1_phi2"] - ref["Lk_phi1_phi2"]) <= BANDS["lk_tol"]
                and m["phi1_knot"] == ref["phi1_knot"])
        det_["catalog"][e["name"]] = {
            "ok": good, "lk": m["Lk_phi1_phi2"], "lk_ref": ref["Lk_phi1_phi2"],
            "det": m["phi1_knot"], "det_ref": ref["phi1_knot"]}
        ok_all &= good
    if not entries:
        ok_all = False
        det_["catalog"] = "EMPTY CATALOG — nothing to conform against"
    if unconformed:
        # The empty-catalog guard above only ever caught ZERO entries; ten entries
        # with zero fields sailed straight past it. Both are the same failure --
        # no evidence -- and both must read as one.
        ok_all = False
        det_["unconformed"] = {
            "count": len(unconformed), "of": len(entries), "names": unconformed,
            "fix": "python field_store.py status  (then `put` the states, or "
                   "re-run the relaxation that produces them)"}

    # sub-grid-offset guard (the +25% degeneracy bias, kept caught)
    from jax_solitons.ehn import relax as ER
    from jax_solitons.ehn.cross_linking import cross_linking
    N, L = 96, 76.8
    phi1, phi2 = ER.build_ic_torus(N, L, 2, 3, 0.22 * L, 0.45 * 0.22 * L,
                                   2.0, n=800)
    lk, _, _ = cross_linking(np.asarray(phi1), np.asarray(phi2), L / N)
    guard_ok = abs(lk - (-3.0)) <= 0.15
    det_["ic_guard"] = {"ok": guard_ok, "lk": float(lk), "seeded": -3.0}
    return {"B5_tracer_conformance": ok_all and guard_ok}, det_


def leg_B7_m1(outroot, box, s=2.0):
    """M1 magnification fidelity: phi(x) -> phi(x/s) by spectral-safe spline
    interpolation of Re/Im; per-realization Lk + Lk-carrying component count
    preserved; lost-loop fraction receipted. Runs on the catalog trefoil."""
    from scipy.ndimage import zoom
    from .particle_catalog import load as cat_load
    from jax_solitons.ehn.cross_linking import cross_linking
    u, _, _, entry = cat_load("trefoil_t23")
    p = entry["relaxation"]["params"]
    N, L = p["N"], p["L"]
    dx = L / N
    p1 = np.asarray(u[0]) + 1j * np.asarray(u[1])
    p2 = np.asarray(u[2]) + 1j * np.asarray(u[3])
    lk0, n10, n20 = cross_linking(p1, p2, dx)

    def mag(f):
        # magnify structure by s at fixed N: crop the central N/s cube and
        # zoom it back to N (phi(x) -> phi(x/s) on the same grid)
        c = int(round(N / (2 * s)))
        lo, hi = N // 2 - c, N // 2 + c
        crop = f[lo:hi, lo:hi, lo:hi]
        zf = N / crop.shape[0]
        return zoom(crop.real, zf, order=3) + 1j * zoom(crop.imag, zf, order=3)

    m1, m2 = mag(p1), mag(p2)
    lk1, n11, n21 = cross_linking(m1, m2, dx)  # dx label arbitrary post-map
    lost = 0 if np.isfinite(lk1) else 1
    ok = (np.isfinite(lk1) and abs(lk1 - lk0) <= 2 * BANDS["lk_tol"]
          and lost <= BANDS["m1_lost_loop_max"])
    det_ = {"s": s, "lk_before": float(lk0), "lk_after": float(lk1),
            "lost_loop_fraction": lost}
    return {"B7_m1_map_fidelity": bool(ok)}, det_


# ================= battery driver =================

QUICK_BOX = {**SB1, "box": "SB-1-quick(N=96 SMOKE)", "N": 96, "L": 76.8,
             "cramp": 2000}
QUICK = {"b1_steps": 9000, "b2_steps": 4000, "headroom": 200}
FULL = {"b1_steps": 12000, "b2_steps": 12000, "headroom": 500}
# b1 12k = the archived dual-meter pair's exact length (bilinear kill
# completes: Q -2.7 -> -0.25 by 8k, -0.01 by 12k, cramp 8000)


def run_battery(quick=False, legs=None, with_m1=False, outroot=None):
    box = QUICK_BOX if quick else SB1
    par = QUICK if quick else FULL
    tag = "quick" if quick else "full"
    outroot = Path(outroot or (_HERE / f"out_sbx_battery_{tag}"))
    outroot.mkdir(parents=True, exist_ok=True)
    sha, per_file = engine_sha()
    want = set(x.upper() for x in legs) if legs else None

    def wanted(*names):
        return want is None or any(n in want for n in names)

    print(f"SB-1 battery {BATTERY_VERSION}{'-quick(SMOKE)' if quick else ''}"
          f"  engine_sha={sha}  box={box['box']}")
    _LEG_ERRORS.clear()
    results, detail = {}, {}
    t0 = time.time()
    if wanted("B5"):
        r, d = leg_B5(outroot, box)
        results.update(r); detail["B5"] = d
        print(f"  B5 tracer conformance: {r}")
    if wanted("B1"):
        r, d = leg_B1(outroot, box, par["b1_steps"])
        results.update(r); detail["B1"] = {k: {kk: v[kk] for kk in
                       ("lk", "dets", "end") if kk in v}
                       for k, v in d.items()}
        print(f"  B1 dual-meter: {r}")
    if wanted("B2", "B3", "B4", "B6"):
        r, d = leg_B2_B3_B4_B6(outroot, box, par["b2_steps"], par["headroom"])
        results.update(r)
        detail["B2346"] = {k: (v if not isinstance(v, dict) or "manifest"
                               not in v else {kk: v[kk] for kk in
                               ("lk", "dets", "end", "no_shrink",
                                "skeleton_pts") if kk in v})
                           for k, v in d.items()}
        print(f"  B2/B3/B4/B6: {r}")
    if with_m1 and wanted("B7"):
        r, d = leg_B7_m1(outroot, box)
        results.update(r); detail["B7"] = d
        print(f"  B7 M1 fidelity: {r}")

    ver = BATTERY_VERSION + ("-quick(SMOKE)" if quick else "")
    cert = issue_certificate(ver, sha, {k: box[k] for k in
                             ("box", "N", "L", "dx", "lam", "kappa", "C", "U",
                              "eps_a", "alpha", "beta", "agrad", "ic",
                              "cramp", "core")}, results)
    # A certificate is citable only if it is quick-free AND every leg actually
    # RAN. `verdict` stays exactly as issue_certificate hashed it (the cid is a
    # hash over that body, and the format is C's), so the execution status is
    # carried in fields ALONGSIDE it: `outcome` is the one to read.
    errors = list(_LEG_ERRORS)
    dirty = bool(per_file.get("dirty"))
    cert.update({"date": time.strftime("%Y-%m-%d %H:%M"),
                 "citable": (not quick) and not errors and not dirty,
                 "outcome": "ERROR" if errors else cert["verdict"],
                 "execution_errors": errors,
                 "not_citable_because": (
                     ["quick(SMOKE) box"] if quick else []) + (
                     [f"{e['leg']} failed to execute ({e['cause']})"
                      for e in errors]) + (
                     [f"engine tree dirty: {per_file.get('dirty_files')}"]
                     if dirty else []),
                 "engine_provenance": per_file,
                 "engine_files": per_file, "detail": detail,
                 "wall_s": round(time.time() - t0, 1),
                 "bands": BANDS})
    cdir = _HERE / "certificates"
    cdir.mkdir(exist_ok=True)
    cpath = cdir / f"{cert['cid']}_{tag}.json"
    cpath.write_text(json.dumps(cert, indent=1, default=str))
    print(f"\ncertificate {cert['cid']}  outcome={cert['outcome']}"
          f"  verdict={cert['verdict']}  citable={cert['citable']}  -> {cpath}")
    for e in errors:
        print(f"  ERROR  {e['leg']} did not execute — {e['cause']}: "
              f"{e['evidence'][:120]}")
    if errors:
        print("  results below are NOT physics verdicts for the failed legs; "
              "a leg that never ran cannot fail a criterion.")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    return cert


# ================= CLI =================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--battery", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="N=96 smoke battery — certificate NOT citable")
    ap.add_argument("--legs", nargs="*", default=None,
                    help="subset, e.g. --legs B5 B1")
    ap.add_argument("--with-m1", action="store_true",
                    help="include the conditional B7 M1 map-fidelity leg")
    ap.add_argument("--out", default=None)
    ap.add_argument("--envelope", nargs="*", default=None, metavar="K=V",
                    help="preflight arithmetic, e.g. --envelope R=22.5 C=400")
    a = ap.parse_args()
    if a.envelope is not None:
        kv = dict(x.split("=") for x in a.envelope)
        R, C = float(kv.get("R", 22.5)), float(kv.get("C", SB1["C"]))
        em = el_mag(R, C)
        print(f"el/mag(R={R}, C={C}) = {em:.1f}  "
              f"(wrapped<{THRESH['wrapped']:.0f}: "
              f"{'HOLDS' if em < THRESH['wrapped'] else 'EXPELS'})")
        print(f"alpha_max(dx={SB1['dx']}, lam={SB1['lam']}) = "
              f"{alpha_max(SB1['dx'], SB1['lam']):.1e}")
        print(f"min_separation(tol={SEAM_FLOOR:.0f}) = "
              f"{min_separation():.1f}")
        return
    if a.battery:
        cert = run_battery(quick=a.quick, legs=a.legs, with_m1=a.with_m1,
                           outroot=a.out)
        sys.exit(0 if cert["verdict"] == "PASS" else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
