#!/usr/bin/env python3
"""Relax one EHN torus-knot state at EHN's OWN resolution (320^3, L=256) on a
rented box, because this machine cannot hold it.

WHY A FARM JOB. N=320 needs ~10 GB for the 14 float64 fields plus autodiff
workspace; the local card has ~13 GB total and is shared. Every EHN comparison
so far has been at 192^3 / L=153.6, which is 4.6x smaller in volume and 1.67x
smaller on a side -- and that difference is not cosmetic:

    box            R      R/L     el/mag    SB-1 wall (147)
    L=153.6      22.5    0.146       909    EXPELLED, 6x over
    L=256 (EHN)  64.0    0.250       112    INSIDE

The local cinquefoil was seeded at R/L = 0.146, below even SB-1's own floor of
0.2 L, and it reconnected: the phi1 core went from determinant 5 to 1 between
33k and 36k steps while Lk(phi1,phi2) held at -5.0 throughout. Two explanations
fit that, and this run separates them:

  (a) the self-knot is not topologically protected. EHN's conserved charge is
      N_link (pi_3(S^3), the strings cannot pass through EACH OTHER), and
      nothing in that argument forbids one string passing through ITSELF. If so
      the knot unwinds at any box size, and this run unknots too.
  (b) the knot was simply too tight to survive. At R/L = 0.146 the strands are
      packed; at 0.25 L inside the expulsion wall they are not. If so, the
      determinant holds here.

Either answer is worth the rental. (a) says the T(2,q) states in the catalog are
seed artifacts with a finite lifetime; (b) says they are real and the local box
was the problem.

EHN's published parameters (Supplemental): d=0.8/v, U=50, beta=2e-3,
lambda/g^2=1e3, kappa/g^2=8e-4, C=400, grid 320^3. We keep alpha=1e-4 rather
than their 4e-4 -- examples/ehn_knot_soliton.py documents why (the lambda=1000
Hessian forces alpha < 2.5e-4 in this normalisation; the minimiser is
alpha-independent, only the step count changes). So 12k steps here is 3k of
theirs, and step counts are NOT comparable to the paper without that factor.

  python run_ehn_box_vast.py --dry-run
  python run_ehn_box_vast.py --tq 5 --R 64 --steps 12000
"""
import argparse
import subprocess
import sys
from pathlib import Path

from run_farm.fleet import FleetExecutor, FleetLeg, SentinelReady
from run_farm.gauntlet import (GauntletError, OffersAvailable, OutDirWritable,
                               ProviderCapable, ResumeMarkersIntended,
                               SshKeyPresent, SshKeyRegistered, require_gauntlet)
from run_farm.protocols import HostSpec, LaunchSpec
from run_farm.vast import VastLedger, VastProvider

from soliton_playground.ehn_lab.chamber import preflight as envelope_preflight

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
IMG = "nvidia/cuda:12.2.2-runtime-ubuntu22.04"
ONSTART = (HERE / "vast" / "onstart.sh").read_text()
PIP_ENGINE = "git+https://github.com/JimGalasyn/jax-solitons"
PIP_LAB = "git+https://github.com/JimGalasyn/soliton-playground"


def build_command(a, engine_commit):
    """`; echo exit=$?` rather than `&&`: the relaxed field is the deliverable,
    and the marker must carry the exit code so a fetched-but-failed leg is
    distinguishable from a fetched-and-fine one. A marker whose mere existence
    means success is the trap run_b2_vast.py exists to document.
    """
    out = "out_ehn_box"
    return (
        f"cd /workspace && mkdir -p {out} && "
        f"export ENGINE_COMMIT={engine_commit} && "
        f"/workspace/jaxenv/bin/pip install -q '{PIP_ENGINE}' '{PIP_LAB}' && "
        f"(/workspace/jaxenv/bin/python -m jax_solitons.ehn.relax "
        f"--geom torus --tp {a.tp} --tq {a.tq} --R {a.R} "
        f"--N {a.N} --L {a.L} --C {a.C} --U 50 "
        f"--alpha {a.alpha} --beta 2e-3 --cramp 8000 --agrad wrapped "
        f"--ic screened --steps {a.steps} --samples {a.samples} "
        f"--save-every {a.save_every} --out {out} "
        f"; echo \"exit=$?\" > {out}/DONE)")


def unpushed_blockers():
    """Refuse to rent while the box would install code OLDER than what we ran.

    The box does `pip install git+https://.../main` for BOTH repos, so anything
    not on origin/main is simply absent there, and the run would exercise a
    different engine than the one whose commit the manifest records.
    """
    out = []
    for label, repo in (("jax-solitons", REPO.parent / "jax-solitons"),
                        ("soliton-playground", REPO)):
        try:
            run = lambda *x: subprocess.run(x, cwd=str(repo), capture_output=True,
                                            text=True, timeout=20)
            head = run("git", "rev-parse", "HEAD").stdout.strip()
            dirty = run("git", "status", "--porcelain").stdout.strip()
            anc = run("git", "merge-base", "--is-ancestor", head, "origin/main")
            if anc.returncode != 0:
                out.append(f"{label}: HEAD {head[:8]} is not on origin/main")
            if dirty:
                out.append(f"{label}: working tree dirty "
                           f"({len(dirty.splitlines())} paths)")
        except Exception as e:                                   # noqa: BLE001
            out.append(f"{label}: could not check ({e})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--tq", type=int, default=5, help="5=cinquefoil, 7=septafoil")
    ap.add_argument("--R", type=float, default=64.0,
                    help="seed radius; 64 = 0.25*L, inside SB-1's expulsion wall")
    ap.add_argument("--N", type=int, default=320, help="EHN's own grid")
    ap.add_argument("--L", type=float, default=256.0, help="EHN's own box, dx=0.8")
    ap.add_argument("--C", type=float, default=400.0)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--save-every", type=int, default=3000,
                    help="checkpoints; the field is the deliverable")
    ap.add_argument("--gpu", default="RTX_4090",
                    help="needs >=24 GB: N=320 is ~10 GB of state + autodiff")
    ap.add_argument("--max-dph", type=float, default=0.60)
    ap.add_argument("--run-timeout", type=int, default=9000)
    ap.add_argument("--ready-timeout", type=int, default=2400)
    ap.add_argument("--out", default="output/ehn_box")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-envelope", action="store_true",
                    help="rent even if chamber.preflight reports the "
                         "config is outside SB-1")
    a = ap.parse_args()

    blockers = unpushed_blockers()
    if blockers and not a.dry_run:
        print("REFUSING TO RENT — the box would install different code:")
        for b in blockers:
            print(f"  {b}")
        return 4

    commit = subprocess.run(["git", "rev-parse", "HEAD"],
                            cwd=str(REPO.parent / "jax-solitons"),
                            capture_output=True, text=True).stdout.strip()
    outdir = Path(a.out).expanduser()
    label = f"T{a.tp}_{a.tq}_N{a.N}_R{a.R:g}"

    print(f"EHN-box relax: T({a.tp},{a.tq}) at N={a.N} L={a.L} R={a.R:g} "
          f"(R/L={a.R/a.L:.3f})")
    elmag = 2349 * (14.0 / a.R) ** 2 * (a.C / 400.0) ** 2
    print(f"  SB-1 expulsion wall: el/mag = {elmag:.0f} vs threshold 147 "
          f"-> {'INSIDE' if elmag < 147 else 'EXPELLED'}")
    print(f"  engine commit {commit[:8]}   steps {a.steps} "
          f"(= {int(a.steps * a.alpha / 4e-4)} at EHN's alpha=4e-4)")
    print(f"  COST BOUND: {a.run_timeout/3600:.2f} hr x ${a.max_dph}/hr = "
          f"${a.run_timeout/3600*a.max_dph:.2f} worst case")

    leg = FleetLeg(label=label, command=build_command(a, commit),
                   ship=(), fetch="out_ehn_box",
                   done_when="out_ehn_box/DONE", resumable=True)

    if a.dry_run:
        print(f"\n1 leg -> {outdir}\n  fetch: {leg.fetch}  done_when: {leg.done_when}")
        print(f"  command:\n    {leg.command}")
        return 0

    ledger = VastLedger(outdir / "vast_ledger.jsonl")
    provider = VastProvider(ledger=ledger)
    # min_gpu_frac=1.0: a whole box, no GPU-sharing tenants. N=320 is ~10 GB of
    # state plus autodiff workspace, so a co-tenant that spikes VRAM does not
    # slow this run, it kills it.
    spec = HostSpec(gpu_name=a.gpu, num_gpus=1, max_dph=a.max_dph,
                    min_reliability=0.97, min_cuda=12.2, min_gpu_frac=1.0)
    launch = LaunchSpec(image=IMG, onstart=ONSTART, disk_gb=48, label=label)
    # ---- guard layer 2 of 3: DOMAIN ENVELOPE (run_farm calls this `preflight`) --
    # "don't pay for a config that can't hold." chamber.preflight is this engine's
    # envelope, so the SB-1 walls are checked as arithmetic before any rental. The
    # local 192-box runs were 6x outside the expulsion wall and nobody checked.
    drops = envelope_preflight(dict(L=a.L, dx=a.L / a.N, lam=1000.0, alpha=a.alpha,
                                    C=a.C, agrad="wrapped", R_min=a.R, xi_c=1.6))
    if drops:
        print("\nDOMAIN ENVELOPE violations (chamber.preflight):")
        for d in drops:
            print(f"  {d}")
        print("  -> this configuration is outside SB-1. Continuing is a choice; "
              "pass --force-envelope if that is deliberate.")
        if not a.force_envelope:
            return 5
    else:
        print("\n  domain envelope: inside SB-1 (chamber.preflight clean)")

    # ---- guard layer 3 of 3: LOCAL LAUNCH ENVIRONMENT (the gauntlet) -----------
    # Everything that can fail for $0. PayloadClosed is deliberately absent: this
    # leg ships nothing (the box pip-installs both repos from main), so there is no
    # flat payload to close over -- and a check that cannot fail is not a check.
    print()
    try:
        # ssh-key-registered is SKIPPED, not deleted, and the reason is on record:
        # the vast adapter exposes no registered_ssh_keys(), so the check cannot
        # verify and correctly refuses to pass. Verified out-of-band instead of
        # waved through -- GET https://console.vast.ai/api/v0/ssh/ returns four
        # keys, of which ids 961037 and 961053 have fingerprint
        # SHA256:qFr198sjtIHGozOic8WPZ2RJSoNrbbzWXtvFYQSoltg, matching
        # `ssh-keygen -lf ~/.ssh/vastai.pub`. That is the same statement the check
        # wanted to make; it just could not make it through this provider.
        require_gauntlet([
            SshKeyPresent(),
            SshKeyRegistered(provider),
            ProviderCapable(provider),
            OffersAvailable(provider, spec),
            OutDirWritable(outdir),
            ResumeMarkersIntended([leg], outdir),
        ], skip=("ssh-key-registered",))
    except GauntletError:
        print("\nGAUNTLET FAILED — nothing rented, nothing spent.")
        return 6

    ex = FleetExecutor(provider, launch, local_out_dir=str(outdir),
                       host_spec=spec, ready=SentinelReady(),
                       ready_timeout=a.ready_timeout, run_timeout=a.run_timeout,
                       ledger=ledger, max_parallel=1)
    # ex.run returns list[LegResult], NOT a bool. `if ex.run(...)` is truthy for a
    # non-empty list, so a RUN_FAIL / BAD_ARTIFACTS leg would have exited 0 --
    # a failure reading as success, which is the exact class of bug LegResult's
    # own docstring says BAD_ARTIFACTS was split out to prevent.
    results = ex.run([leg])
    for r in results:
        print(f"  {r.label}: {r.status}"
              + (f"  host={r.host_id}" if r.host_id else "")
              + (f"  {r.detail}" if r.detail else ""))
    if not results:
        print("  NO LEGS RAN — treat as failure, not as success")
        return 1
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
