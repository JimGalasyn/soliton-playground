#!/usr/bin/env python3
"""The N_link ladder: does the ∂a discretisation alone reproduce EHN's floor?

Design, criteria and predicted outcome are PRE-REGISTERED in docs/NLINK_LADDER.md.
Read that first; this file executes it and must not quietly diverge from it.

    arms   agrad ∈ {wrapped, bilinear}
    rungs  nlink ∈ {1, 2, 3, 4, 5}
    → 10 legs, everything else frozen at SB-1 (N=192, L=153.6, R=38.4, C=400)

WHY THIS IS CHEAP. Unlike the quench campaign, nothing is shipped: every leg builds
its own initial condition on-box from `knot_batch.build_ic`, so there is no 792 MB
field.npz upload. The legs are independent, so `--max-parallel` buys wall-clock
directly. 0.118 s/step measured at N=192 (form-1e4, 10k steps in 19:41) ⇒ ~1.18 h
per leg at 36k steps, ~11.8 gpu-h for the ladder.

THE WRAPPED ARM IS A REPRODUCTION TEST, NOT JUST A CONTROL. steps=36000 is the
catalog's own pre-registered horizon, so wrapped/nlink∈{1,3,5} must return what
`unknot_bare`, `trefoil_t23` and `cinquefoil_t25` already record. `--check-repro`
scores that automatically. If those three do not reproduce, the campaign is broken
and no bilinear result from it counts — which is why it is checked here rather than
left to a reader comparing tables by eye.

THE CONFOUND, AND WHY THE LADDER SURVIVES IT. The expulsion wall is arm-specific
(THRESH = {wrapped: 147, bilinear: 37}) and at L=153.6/C=400 NO radius satisfies
both it and the g2 ceiling — wrapped wants R ≥ 56.0, bilinear R ≥ 111.5, ceiling
53.8. Every leg here runs outside its own wall and the bilinear arm runs further
outside, so chamber.preflight will report violations for all ten. That is expected
and is NOT waved away with --force-envelope silently: the count is printed per arm.

The reason the result still means something is that `el_mag(R, C)` has NO nlink
dependence, so the wall cannot produce an nlink-DEPENDENT floor. Hence rungs 4 and 5
are mandatory: they are where EHN say knots do bind, and a bilinear arm that passes
there while failing at 1–3 cannot be explained by a wall that does not know what
nlink is. A bilinear arm that fails at EVERY rung does not support the hypothesis —
see the corollary in NLINK_LADDER.md.

REMOTE ENV IS PINNED, and this campaign is exactly the caller that needs it. The
legs are resumable/reattachable, so a leg may be resumed on a replacement host; XLA
GPU autotuning selects kernels per process, so without
`XLA_FLAGS=--xla_gpu_autotune_level=0` a resumed leg is not bit-identical to an
uninterrupted one (run-farm measured 1 divergence in 3 attempts, 73% of entries).
An arm-to-arm comparison whose legs are not individually reproducible is not a
comparison.

run-farm's `RemoteEnvPinned` guards that property but reads
`ProviderExecutor.remote_env`, and this campaign runs on `FleetExecutor`, which has
no such parameter — passing it raises TypeError instead of pinning anything. So the
export goes into the leg command itself (build_command's `env`) and `LegEnvPinned`
below checks the built commands. That is the stronger placement anyway: a
non-interactive `ssh host cmd` sources no profile, so onstart's exports would not
reach the payload by either route, and a variable in the command appears in
whatever records the command.

PROVIDER. RunPod by default rather than Vast, because these are ~1.2 h legs and
Vast's spot pool has already cost this project real money for nothing: a
`host_failed` billed 2753 s for a box that never came up, and an ssh channel died
5500 steps into a 12000-step run. `RunPodProvider` is behind the same Provider
seam, so the only differences that leak here are that its offers are GPU *types*
rather than named hosts and that the CUDA floor is applied at pod-create rather
than in offers(). `--provider vast` switches back; `--cloud-type SECURE` is the
stability upgrade within RunPod and needs a higher --max-dph.

  python run_nlink_ladder.py --dry-run
  python run_nlink_ladder.py --max-parallel 5
  python run_nlink_ladder.py --provider vast
  python run_nlink_ladder.py --score-only        # re-score a finished campaign
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from run_farm.budget import BudgetExceeded, CappedProvider, estimate
from run_farm.fleet import FleetExecutor, FleetLeg, SentinelReady
from run_farm.gauntlet import (CheckResult, GauntletError, require_gauntlet,
                               standard_gauntlet)
from run_farm.ledger import RentalLedger
from run_farm.protocols import HostSpec, LaunchSpec
from run_farm.runpod import RunPodProvider
from run_farm.vast import VastProvider

from soliton_playground.ehn_lab.chamber import preflight as envelope_preflight

# Sibling script, not a package: experiments/ is sys.path[0] when run directly.
# build_command is IMPORTED rather than copied — every branch of that leg script
# was paid for by a specific failure (a dead ssh channel, an OOM 0.92 s in, a
# truncated field.npz), and a second transcription would not inherit the next fix.
from run_ehn_box_vast import (IMG, ONSTART, PIP_ENGINE, PIP_LAB, build_command,
                              remote_exit, unpushed_blockers)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT_SUB = "out_nlink"

ARMS = ("wrapped", "bilinear")
RUNGS = (1, 2, 3, 4, 5)

#: rung -> the catalog entry the wrapped arm must reproduce at that rung.
REPRO = {1: "unknot_bare", 3: "trefoil_t23", 5: "cinquefoil_t25"}

#: The pinned worker environment. See the module docstring.
REMOTE_ENV = {"XLA_FLAGS": "--xla_gpu_autotune_level=0"}

#: RunPod's own resolver looks only at $RUNPOD_API_KEY and ~/.runpod_api_key;
#: this machine keeps its keys in ~/tokens/, so the path is passed explicitly.
RUNPOD_KEY = "~/tokens/.runpod_api_key"


class LegEnvPinned:
    """Every leg's command actually exports the variables this campaign needs.

    run-farm's `RemoteEnvPinned` guards the same property but inspects
    `ProviderExecutor.remote_env`, and this campaign runs on `FleetExecutor`, which
    has no such attribute — passing it raises TypeError rather than pinning
    anything. So the exports live in the leg command itself (build_command's `env`),
    which is strictly more robust: a non-interactive `ssh host cmd` sources no
    profile, so onstart's exports never reach the payload either way, and putting it
    in the command means it appears in whatever records the command.

    This checks the built commands rather than the intent that produced them, so it
    fails if build_command stops honouring `env` — which is the failure that would
    otherwise be silent, every leg running and producing plausible numbers with only
    the reproducibility claim void.
    """

    name = "leg-env-pinned"

    def __init__(self, legs, required, *, fatal=True):
        self.legs, self.required, self.fatal = list(legs), dict(required), fatal

    def __call__(self):
        proves = ("every leg's command string exports these variables. Does NOT "
                  "prove the box honours them or that the values are right for this "
                  "engine -- only that what you meant to send is in what is sent.")
        missing = []
        for leg in self.legs:
            for k, v in self.required.items():
                if f"export {k}=" not in leg.command:
                    missing.append(f"{leg.label}:{k}")
                elif shlex.quote(str(v)) not in leg.command:
                    missing.append(f"{leg.label}:{k}(wrong value)")
        if missing:
            return CheckResult(self.name, False,
                               f"{len(missing)} leg/var pair(s) unpinned: "
                               + ", ".join(missing[:4])
                               + ("..." if len(missing) > 4 else ""),
                               proves, fatal=self.fatal)
        keys = ", ".join(sorted(self.required))
        return CheckResult(self.name, True,
                           f"all {len(self.legs)} leg(s) export {keys}", proves)


def label_of(agrad, nlink):
    return f"{agrad}_nlink{nlink}"


def relax_args(a, agrad, nlink, out):
    """The engine invocation for one rung of one arm.

    `--geom rings` (not torus): build_ic seeds one big φ₂ ring with `nlink` small
    φ₁ rings pierced by it, so N_link is the seed parameter rather than a torus
    winding. That is the axis EHN's floor is stated on.
    """
    return (
        f"-m jax_solitons.ehn.relax "
        f"--geom rings --nlink {nlink} --R {a.R} --core {a.core} "
        f"--N {a.N} --L {a.L} --C {a.C} --U 50 --lam {a.lam} --kappa {a.kappa} "
        f"--alpha {a.alpha} --beta 2e-3 --cramp 8000 --agrad {agrad} "
        f"--ic screened --steps {a.steps} --samples {a.samples} "
        f"--topo-every {a.topo_every} "
        f"--det-every {a.det_every} --det-timeout {a.det_timeout} "
        f"--out {out}")


# ------------------------------------------------------------------ scoring --
def _final(manifest):
    """The last trajectory sample, or None."""
    traj = (manifest or {}).get("traj") or []
    return traj[-1] if traj else None


def load_manifest(outdir, label):
    p = Path(outdir) / label / OUT_SUB / "manifest.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def score_charge(manifest, nlink):
    """PRE-REGISTERED charge meter: |Q| >= 0.5*nlink.

    Calibrated on the one control that already exists: at nlink=3 the wrapped arm
    gives Q = -2.868 (≈ -nlink) and bilinear gives -0.01. The threshold sits an
    order of magnitude clear of both, so it is not fitted to either.

    `elec` is returned for the record but NOT gated: its scaling with N_link is
    unmeasured, and inventing a threshold from a single trefoil datum would fit the
    gate to the thing it is supposed to test.
    """
    f = _final(manifest)
    if f is None:
        return None, {}
    q, el = f.get("Q"), f.get("elec")
    if q is None:
        return None, {"Q": None, "elec": el}
    return abs(q) >= 0.5 * nlink, {"Q": round(float(q), 4),
                                   "elec": None if el is None else round(float(el), 1)}


def score_geometry(manifest, ref_det=None):
    """PRE-REGISTERED geometric meter: det matches the arm-paired reference, one
    dominant component, and skeleton size stable to ±1% over the last third.

    `ref_det` is the wrapped arm's determinant at the SAME rung, so this never
    assumes which knot type a rung produces — the ladder pairs itself.
    """
    if not manifest:
        return None, {}
    det1 = manifest.get("det1")
    ev = {"det1": det1, "cross_lk": manifest.get("cross_lk"),
          "e_finite": manifest.get("e_finite")}
    if not det1:
        return None, ev
    ncomp = len(det1)
    det = int(det1[0][1]) if det1[0][1] is not None else None
    ev.update({"ncomp": ncomp, "det": det})

    # segment stability over the last third, from the topo samples
    segs = [(s["n"], s["nseg1"]) for s in (manifest.get("traj") or [])
            if s.get("nseg1") is not None]
    stable = None
    if len(segs) >= 2:
        nmax = segs[-1][0]
        tail = [v for n, v in segs if n >= (2 / 3) * nmax]
        if len(tail) >= 2 and tail[0] > 0:
            drift = (max(tail) - min(tail)) / tail[0]
            ev["seg_drift"] = round(drift, 4)
            ev["nseg1"] = tail[-1]
            stable = drift <= 0.01
    if manifest.get("e_finite") is False:
        return False, ev
    ok = (ncomp == 1) and (stable is not False)
    if ref_det is not None and det is not None:
        ok = ok and (det == ref_det)
    return ok, ev


def check_repro(outdir, results_by_label):
    """Score the wrapped arm against the catalog entries it must reproduce.

    A campaign that cannot reproduce what the repo already holds has not earned the
    right to report anything about the arm it is testing.
    """
    lines, all_ok = [], True
    cat = REPO / "src" / "soliton_playground" / "ehn_lab" / "particles"
    for nlink, name in sorted(REPRO.items()):
        lab = label_of("wrapped", nlink)
        man = load_manifest(outdir, lab)
        try:
            entry = json.loads((cat / name / "entry.json").read_text())
        except OSError:
            lines.append(f"  {lab}: catalog entry {name} not readable — cannot check")
            all_ok = False
            continue
        want = entry.get("measured", {})
        got_det = (man or {}).get("det1")
        want_det = want.get("phi1_knot")
        f = _final(man)
        got_q = None if f is None else round(float(f.get("Q", float("nan"))), 3)
        want_q = round(float(entry.get("energy_final", {}).get("Q", float("nan"))), 3)
        ok = (got_det is not None and want_det is not None
              and int(got_det[0][1]) == int(want_det[0][1]))
        all_ok = all_ok and ok
        lines.append(f"  {lab} vs {name}: det {got_det} vs {want_det} "
                     f"| Q {got_q} vs {want_q}  -> {'REPRODUCES' if ok else 'DOES NOT'}")
    return all_ok, lines


def report(outdir, legs_meta, results=None):
    """The campaign verdict: two meters per leg, arms side by side."""
    print("\n" + "=" * 78)
    print("N_LINK LADDER — pre-registered scoring (docs/NLINK_LADDER.md)")
    print("=" * 78)

    # wrapped first, so its determinant is available to pair the bilinear arm
    ref_det = {}
    table = {}
    for agrad in ARMS:
        for nlink in RUNGS:
            lab = label_of(agrad, nlink)
            man = load_manifest(outdir, lab)
            geo, gev = score_geometry(man, ref_det.get(nlink))
            chg, cev = score_charge(man, nlink)
            if agrad == "wrapped" and gev.get("det") is not None:
                ref_det[nlink] = gev["det"]
            rc = None
            meta = legs_meta.get(lab)
            if meta is not None:
                rc = remote_exit(outdir, meta)
            bound = (geo is True) and (chg is True)
            table[lab] = dict(geo=geo, chg=chg, bound=bound, rc=rc, **gev, **cev)

    hdr = f"{'leg':22s} {'rc':>4s} {'det':>5s} {'ncomp':>5s} {'nseg':>6s} {'Q':>9s} {'elec':>8s}  geo  chg  BOUND"
    print(hdr)
    print("-" * len(hdr))
    for agrad in ARMS:
        for nlink in RUNGS:
            lab = label_of(agrad, nlink)
            r = table[lab]
            f = lambda v: "-" if v is None else ("PASS" if v is True else "fail")
            print(f"{lab:22s} {str(r['rc']):>4s} {str(r.get('det')):>5s} "
                  f"{str(r.get('ncomp')):>5s} {str(r.get('nseg1')):>6s} "
                  f"{str(r.get('Q')):>9s} {str(r.get('elec')):>8s}  "
                  f"{f(r['geo']):4s} {f(r['chg']):4s} "
                  f"{'YES' if r['bound'] else 'no'}")
        print()

    ok, lines = check_repro(outdir, table)
    print("REPRODUCTION TEST (wrapped arm vs the catalog):")
    for ln in lines:
        print(ln)
    if not ok:
        print("  -> CAMPAIGN IS BROKEN. The wrapped arm does not reproduce states this "
              "repo already holds; no bilinear result from this run counts.")
        return 1

    wfloor = [n for n in RUNGS if table[label_of('wrapped', n)]['bound']]
    bfloor = [n for n in RUNGS if table[label_of('bilinear', n)]['bound']]
    print(f"\nBOUND rungs:  wrapped {wfloor or 'none'}   bilinear {bfloor or 'none'}")
    if not bfloor:
        print("  -> bilinear fails at EVERY rung including 4 and 5. Per "
              "NLINK_LADDER.md this does NOT support the hypothesis: it is what an "
              "inadmissible geometry looks like, not what a discretisation-driven "
              "floor looks like. Go to stage 2 / a box where bilinear is inside its "
              "wall before concluding anything.")
    elif set(wfloor) == set(RUNGS) and min(bfloor) > min(RUNGS):
        print(f"  -> bilinear has a floor at nlink={min(bfloor)} and wrapped does not. "
              f"el_mag has no nlink dependence, so the expulsion wall cannot produce "
              f"this shape. The ∂a discretisation can.")
    else:
        print("  -> outcome does not match any pre-registered pattern; read the "
              "falsifier list in NLINK_LADDER.md before interpreting.")
    return 0


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=192, help="SB-1 grid")
    ap.add_argument("--L", type=float, default=153.6, help="SB-1 box, dx=0.8")
    ap.add_argument("--R", type=float, default=38.4, help="seed radius, R/L=0.25")
    ap.add_argument("--core", type=float, default=2.0)
    ap.add_argument("--C", type=float, default=400.0)
    ap.add_argument("--lam", type=float, default=1000.0)
    ap.add_argument("--kappa", type=float, default=8e-4)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=36000,
                    help="the catalog's pre-registered horizon; changing it breaks "
                         "the reproduction test")
    ap.add_argument("--samples", type=int, default=36)
    ap.add_argument("--topo-every", type=int, default=1)
    ap.add_argument("--det-every", type=int, default=6)
    ap.add_argument("--det-timeout", type=float, default=180.0)
    ap.add_argument("--rungs", type=int, nargs="+", default=list(RUNGS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--provider", choices=("runpod", "vast"), default="runpod",
                    help="RunPod by default: these legs are ~1.2 h each and Vast's "
                         "spot pool has already cost this project a host_failed "
                         "(2753 billed seconds for a box that never came up) plus a "
                         "mid-run ssh death at 5500/12000 steps.")
    ap.add_argument("--runpod-key", default=RUNPOD_KEY)
    ap.add_argument("--cloud-type", default="COMMUNITY",
                    choices=("COMMUNITY", "SECURE"),
                    help="RunPod tier. SECURE is the stability upgrade (datacentre "
                         "rather than community hosts) and costs more; raise "
                         "--max-dph with it or offers() will return nothing.")
    ap.add_argument("--gpu", default="RTX_4090",
                    help="N=192 fits comfortably; this is NOT the N=320 leg that "
                         "needed A100-class")
    ap.add_argument("--max-dph", type=float, default=0.40)
    ap.add_argument("--cap-usd", type=float, default=12.00,
                    help="ENFORCED total for this ledger. Expected spend is ~11.8 "
                         "gpu-h ~= $4.15; the worst case the estimator prints "
                         "assumes EVERY leg runs to --run-timeout, and the cap has "
                         "to clear that or the campaign aborts mid-flight having "
                         "already paid. Headroom above it is acquisition tax.")
    ap.add_argument("--max-parallel", type=int, default=5,
                    help="legs are independent, so this buys wall-clock without "
                         "changing gpu-hours")
    ap.add_argument("--run-timeout", type=int, default=7200,
                    help="2 h against a measured 1.18 h leg: enough margin for a "
                         "slow host, tight enough that a hung leg is not billed for "
                         "2.5 h before anyone notices")
    ap.add_argument("--ready-timeout", type=int, default=1800)
    ap.add_argument("--out", default="output/nlink_ladder")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--score-only", action="store_true",
                    help="re-score an existing output dir, rent nothing")
    ap.add_argument("--force-envelope", action="store_true")
    a = ap.parse_args()

    outdir = Path(a.out).expanduser()
    rungs, arms = tuple(a.rungs), tuple(a.arms)

    commit = subprocess.run(["git", "rev-parse", "HEAD"],
                            cwd=str(REPO.parent / "jax-solitons"),
                            capture_output=True, text=True).stdout.strip()

    legs, legs_meta = [], {}
    for agrad in arms:
        for nlink in rungs:
            lab = label_of(agrad, nlink)
            leg = FleetLeg(
                label=lab,
                command=build_command(relax_args(a, agrad, nlink, OUT_SUB),
                                      commit, out=OUT_SUB, env=REMOTE_ENV),
                ship=(), fetch=OUT_SUB, done_when=f"{OUT_SUB}/DONE",
                resumable=True, reattachable=True)
            legs.append(leg)
            legs_meta[lab] = leg

    if a.score_only:
        return report(outdir, legs_meta)

    print(f"N_LINK LADDER: {len(arms)} arm(s) x {len(rungs)} rung(s) = {len(legs)} legs")
    print(f"  SB-1: N={a.N} L={a.L} dx={a.L/a.N:.2f} R={a.R} (R/L={a.R/a.L:.3f}) "
          f"C={a.C}  steps={a.steps}")
    print(f"  engine commit {commit[:8]}")
    print(f"  provider {a.provider}"
          + (f" ({a.cloud_type})" if a.provider == "runpod" else "")
          + f"  gpu {a.gpu}  max ${a.max_dph}/hr  parallel {a.max_parallel}")

    # ---- guard layer: DOMAIN ENVELOPE, per arm ------------------------------
    # Expected to fail here, and reported rather than suppressed. See the module
    # docstring: at this geometry NO radius satisfies both walls, the whole existing
    # catalog was built outside them, and the ladder's logic does not depend on
    # being inside — it depends on el_mag having no nlink dependence.
    print("\n  domain envelope (chamber.preflight), per arm:")
    outside = {}
    for agrad in arms:
        drops = envelope_preflight(dict(L=a.L, dx=a.L / a.N, lam=a.lam,
                                        alpha=a.alpha, C=a.C, agrad=agrad,
                                        R_min=a.R, xi_c=1.6))
        outside[agrad] = drops
        print(f"    {agrad:9s}: {'inside SB-1' if not drops else drops[0]}")
    if any(outside.values()) and not (a.force_envelope or a.dry_run):
        print("\n  -> outside SB-1. This is EXPECTED for this campaign and is why the "
              "verdict rests on the nlink-DEPENDENCE of the result, not on the "
              "absolute pass/fail of either arm. Pass --force-envelope to proceed, "
              "deliberately.")
        return 5

    acq = a.ready_timeout / 3600.0 * a.max_dph
    est = estimate(len(legs), a.run_timeout, a.max_dph, acq_tax_usd=acq,
                   failure_tax=0.0)
    print(f"\n  COST: ${est['usd']:.2f} worst case = {est['gpu_h']:.2f} gpu-h at the "
          f"{a.run_timeout/3600:.2f} hr timeout x ${a.max_dph}/hr + ${acq:.2f} "
          f"acquisition tax per failed host")
    print(f"  cap: ${a.cap_usd:.2f} ENFORCED before every rent()")

    if a.dry_run:
        print(f"\n{len(legs)} legs -> {outdir}")
        for lg in legs:
            print(f"  {lg.label:22s} fetch={lg.fetch} done_when={lg.done_when}")
        print(f"\n  example command ({legs[0].label}):\n    {legs[0].command[:400]} ...")
        return 0

    blockers = unpushed_blockers()
    if blockers:
        print("REFUSING TO RENT — the box would install different code:")
        for b in blockers:
            print(f"  {b}")
        return 4

    ledger = RentalLedger(outdir / "rental_ledger.jsonl")
    if a.provider == "runpod":
        # RunPod resolves the CUDA floor at pod CREATE, not in offers() (its
        # catalog is GPU *types*, not hosts), so min_cuda is constructor config
        # here rather than a HostSpec filter as it is for Vast.
        key = Path(a.runpod_key).expanduser().read_text().strip()
        provider = RunPodProvider(api_key=key, ledger=ledger,
                                  cloud_type=a.cloud_type, min_cuda=12.2)
    else:
        provider = VastProvider(ledger=ledger)
    capped = CappedProvider(provider, cap_usd=a.cap_usd, ledger=ledger)
    spec = HostSpec(gpu_name=a.gpu, num_gpus=1, max_dph=a.max_dph,
                    min_reliability=0.97, min_cuda=12.2, min_gpu_frac=1.0)
    launch = LaunchSpec(image=IMG, onstart=ONSTART, disk_gb=32, label="nlink-ladder")

    # The env pin is checked on the LEG COMMANDS (LegEnvPinned), not on the
    # executor: run-farm's RemoteEnvPinned reads ProviderExecutor.remote_env and
    # FleetExecutor has no such parameter — passing it raises rather than pinning.
    ex = FleetExecutor(capped, launch, local_out_dir=str(outdir), host_spec=spec,
                       ready=SentinelReady(), ready_timeout=a.ready_timeout,
                       run_timeout=a.run_timeout, ledger=ledger,
                       max_parallel=a.max_parallel)

    try:
        # The RAW provider, not `capped`. CappedProvider exposes only offers/rent
        # (its rent() is a context manager that destroys on exit, so nothing calls
        # destroy through it), and ProviderCapable asked the wrapper whether it
        # could destroy — it cannot, and said so. Checking the wrapper tests the
        # budget shim's surface, not the cloud's.
        checks = standard_gauntlet(provider=provider, host_spec=spec,
                                   out_dir=outdir, legs=legs, payload=None)
        checks.append(LegEnvPinned(legs, REMOTE_ENV))
        # skip= belongs to require_gauntlet, not standard_gauntlet: the former
        # REPORTS the bypass as SKIPPED, the latter deletes it from the report where
        # it reads exactly like a check that passed.
        require_gauntlet(checks, skip=("ssh-key-registered",))
    except GauntletError:
        print("\nGAUNTLET FAILED — nothing rented, nothing spent.")
        return 6

    try:
        results = ex.run(legs)
    except BudgetExceeded as e:
        print(f"\nBUDGET CAP HIT: {e}")
        print(f"  verify nothing is still billing: "
              f"python -m run_farm.reap --ledger {ledger.path}")
        return 7

    for r in results:
        print(f"  {r.label}: {r.status}" + (f"  ({r.detail})" if r.detail else ""))
    live = provider.list_instances()
    print(f"\nTEARDOWN CHECK: {len(live)} instance(s) still live"
          + ("" if not live else "  <-- DESTROY THESE"))
    for i in live:
        print(f"  id={i.id} status={i.status} dph=${i.dph:.4f}/hr")

    return report(outdir, legs_meta, results)


if __name__ == "__main__":
    raise SystemExit(main())
