#!/usr/bin/env python3
"""One rented 24 GB GPU, one B2 leg: the citable SB-1 trefoil-hold certificate.

WHY THIS EXISTS. B2 at N=192 needs a 3.16 GiB allocation that an 8 GB card cannot
serve, so `standard_box.py --battery --legs B2` OOMs locally and only the N=96
--quick (explicitly NOT citable) battery runs here. Everything else about the
result is already known: the quick battery reproduces the 2026-07-11 archive
BIT-IDENTICALLY (lk = -3.0001967524330566, det 3, 486 segments held), and
engine_sha is unchanged at 6f41e1267b668bfa. So this leg is a known-answer job --
if it comes back different, the pipeline is broken, not the physics.

SPLIT, per vast/RUNBOOK.md ("all knot ID / census stays local"): the rented box
does the GPU-heavy relaxation and writes manifest.json + field.npz; scoring
(cross_linking, knot_determinants -> the det discriminator, which needs pyknotid)
happens locally afterwards. standard_box._run_leg now REUSES an existing manifest,
so the local pass scores the fetched artifacts instead of re-running them. The box
also tries pyknotid opportunistically -- if PyPI is not throttled it scores there
too and we get a cross-check -- but nothing depends on that succeeding.

TEARDOWN. FleetExecutor owns rent / failover / signal-safe teardown; this file is
just the (command + ship + fetch) config, exactly as run_stability_fleet.py is.
`--run-timeout` is a HARD wall-clock cap so a hung leg cannot quietly bill: at the
3090 spot price seen on 2026-07-29 ($0.113/hr) the default 90 min caps exposure at
about $0.17. ALWAYS confirm with `vastai show instances` (or provider
list_instances) after a run; the runbook shouts RELEASE THE METER for a reason.

  python run_b2_vast.py --dry-run     # legs + cost estimate, no key, no spend
  python run_b2_vast.py               # rent one box and run it
  # then, locally, to score the fetched artifacts into a citable certificate:
  python standard_box.py --battery --legs B2
"""
import argparse
import sys
from pathlib import Path

try:
    from jax_solitons.campaign import (FleetExecutor, FleetLeg, HostSpec,
                                       LaunchSpec, SentinelReady, VastLedger,
                                       VastProvider)
# ⚠ ImportError, not ModuleNotFoundError. `jax_solitons/campaign/` did not vanish in
# the 2026-07 extraction -- it survives upstream as a directory containing nothing but
# `__pycache__`, which Python happily treats as a NAMESPACE PACKAGE. So the import
# resolves, exports nothing, and raises plain ImportError; the narrower except never
# fired and this entire script has been unimportable since. A stale directory made a
# gate unreachable, and nothing said so because nobody ran the script that owns it.
except ImportError:              # campaign layer extracted to run-farm, 2026-07
    from run_farm import (FleetExecutor, FleetLeg, HostSpec, LaunchSpec,
                          SentinelReady, VastLedger, VastProvider)
from run_farm import RunPodProvider

# Sibling script, not a package: experiments/ is sys.path[0] when run directly.
# JAX_PIN and unpushed_blockers are IMPORTED rather than copied, on the same rule
# build_command already states in run_nlink_ladder -- every branch of these was paid
# for by a specific failure and a second transcription does not inherit the next fix.
# The copy that stood here had already drifted: its dirty check was UNSCOPED, so a
# previous run's uncommitted manifest under output/ would refuse a rental. That is
# the check firing on the wrong thing; the shared one is scoped to what pip installs.
# This file keeps its own stricter `local_engine_state()` gate on the ehn_lab payload.
from run_ehn_box_vast import JAX_PIN, unpushed_blockers

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                       # soliton-playground checkout root
# 12.2 (not 12.4): the image's NVIDIA_REQUIRE_CUDA floor must be <= host driver.
IMG = "nvidia/cuda:12.2.2-runtime-ubuntu22.04"
ONSTART = (HERE / "vast" / "onstart.sh").read_text()

# NOTHING IS SHIPPED ANY MORE. Before the 2026-08-01 extraction this listed seven
# loose .py files spanning three directories, and getting that list wrong cost two
# rentals -- gpe_vortex_topology.py was in a sibling dir, core_knot_id.py in a
# third. Both the engine (jax_solitons.ehn) and this battery
# (soliton_playground.ehn_lab) are installable packages now, so the box pip-installs
# them and the dependency graph is pip's problem rather than a hand-maintained
# tuple. PAYLOAD survives only as the set whose dirtiness makes a run
# unreproducible.
PAYLOAD = (str(REPO / "src" / "soliton_playground" / "ehn_lab"),)

# What the box installs. onstart.sh already pulls jax-solitons from GitHub main;
# the battery has to come from somewhere too, and it is a separate repo.
PIP_ENGINE = "git+https://github.com/JimGalasyn/jax-solitons"
PIP_BATTERY = "git+https://github.com/JimGalasyn/soliton-playground"


def build_command(engine_commit):
    """`; echo exit=$?` not `&&`: the relaxations are the deliverable, and a
    scoring failure for want of pyknotid must NOT cost us the artifacts or the
    sentinel.

    ENGINE_COMMIT is exported because the box has no git repo of ours, so
    standard_box.engine_sha() cannot derive provenance there. Passing the local
    commit makes the box's certificate record the tree it actually ran, instead of
    falling back to a content hash.
    """
    return (
        f"cd /workspace && mkdir -p out_sbx_battery_full && "
        f"export ENGINE_COMMIT={engine_commit} && "
        f"/workspace/jaxenv/bin/pip install -q '{PIP_BATTERY}' && "
        # Solver identity, in the `&&` prefix rather than the `;` tail: a certificate
        # issued under a solver that is not the pin is not a weaker certificate, it
        # is a certificate for a different integrator. So the box records what it
        # resolved and refuses to run the battery at all if it is not {JAX_PIN}.
        f"/workspace/jaxenv/bin/python -c \"import json, sys, jax, jaxlib; "
        f"e = {{'jax': jax.__version__, 'jaxlib': jaxlib.__version__, "
        f"'pin': '{JAX_PIN}', 'python': sys.version.split()[0]}}; "
        f"json.dump(e, open('out_sbx_battery_full/env.json', 'w'), indent=1); "
        f"sys.exit(0 if jax.__version__ == '{JAX_PIN}' else 91)\" && "
        "(/workspace/jaxenv/bin/pip install -q pyknotid 2>/dev/null "
        "|| echo 'pyknotid unavailable on box; scoring stays local') && "
        "(/workspace/jaxenv/bin/python -m soliton_playground.ehn_lab.standard_box "
        "--battery --legs B2 "
        "; echo \"exit=$?\" > out_sbx_battery_full/DONE)")




def local_engine_state():
    """(commit, dirty_payload_files) for the tree being shipped."""
    import subprocess
    run = lambda *a: subprocess.run(a, cwd=str(HERE), capture_output=True,
                                    text=True, timeout=20).stdout.strip()
    commit = run("git", "rev-parse", "HEAD")
    porcelain = run("git", "status", "--porcelain", "--", *PAYLOAD)
    return commit, [ln[2:].strip() for ln in porcelain.splitlines() if ln.strip()]


def _remote_exit(outdir, leg):
    """The exit status the REMOTE command recorded in its done_when marker.

    Returns the int code, or None when there is nothing to read (no marker, an
    unparseable body, a leg whose marker is not a DONE file). None means "no
    evidence", NOT "success" -- the caller must not treat it as a pass, which is
    the whole failure mode this function exists to close.

    build_command() ends with `; echo "exit=$?" > .../DONE`, so the marker whose
    mere EXISTENCE satisfies the fleet layer also carries the code. Reading it is
    the difference between "the artifacts arrived" and "the job worked".
    """
    marker = Path(outdir) / leg.label / leg.done_when
    try:
        body = marker.read_text().strip()
    except OSError:
        return None
    if not body.startswith("exit="):
        return None
    try:
        return int(body[len("exit="):])
    except ValueError:
        return None


def preflight(key_path="~/.ssh/vastai"):
    """Fail BEFORE renting if the SSH prerequisite is missing.

    This is the check whose absence cost 7 rentals and 72 minutes: FleetExecutor
    defaults to key_path=~/.ssh/vastai, that file did not exist, so every host
    rejected the connection ("Failed publickey for root ... [preauth]") and each
    one looked like a spot-pool failure (TimeoutError / HostProbeFailed) instead
    of the single local cause it was. A Vast API key proves NOTHING about this:
    list_instances() is an API call and never touches SSH. vast/RUNBOOK.md lists
    both steps under One-time setup; only the API half had been done.
    """
    import json as _json, urllib.request as _u
    problems = []
    kp = Path(key_path).expanduser()
    if not kp.exists():
        problems.append(f"missing private key {kp} -- "
                        f"ssh-keygen -t ed25519 -f {kp} -N ''")
        return problems
    pub = kp.with_suffix(kp.suffix + ".pub") if kp.suffix else Path(str(kp) + ".pub")
    if not pub.exists():
        problems.append(f"missing public key {pub}")
        return problems
    mine = pub.read_text().split()[1] if len(pub.read_text().split()) > 1 else ""
    try:
        keyfile = next((Path(q).expanduser() for q in
                        ("~/.config/vastai/vast_api_key", "~/.vast_api_key")
                        if Path(q).expanduser().exists()), None)
        api = keyfile.read_text().strip()
        r = _u.urlopen(_u.Request("https://console.vast.ai/api/v0/ssh/",
                                 headers={"Authorization": f"Bearer {api}"}))
        got = _json.loads(r.read())
        reg = got.get("results", got) if isinstance(got, dict) else got
        blobs = " ".join(k.get("public_key", "") for k in reg
                         if isinstance(k, dict))
        if mine and mine not in blobs:
            problems.append(f"{pub.name} is NOT registered with Vast -- "
                            "POST it to /api/v0/ssh/ (RUNBOOK one-time setup)")
    except Exception as e:
        problems.append(f"could not verify key registration: {type(e).__name__}: {e}")
    return problems


# A path-exercising leg that costs almost nothing: real image, real onstart (so
# the conda/jax install and the run-farm PyPI dependency are genuinely tested),
# real ship/ssh/fetch/destroy -- only the 40-minute physics is replaced.
SMOKE_COMMAND = (
    "cd /workspace && mkdir -p smoke && "
    "{ hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; "
    "/workspace/jaxenv/bin/python -c "
    "\"import jax, jax_solitons; d=jax.devices(); "
    "print('jax', jax.__version__, d); "
    "print('jax_solitons', jax_solitons.__version__ "
    "if hasattr(jax_solitons,'__version__') else 'ok')\"; "
    "} > smoke/report.txt 2>&1; cat smoke/report.txt; "
    "echo done > smoke/DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud-type", default="SECURE",
                    choices=["SECURE", "COMMUNITY"],
                    help="runpod tier. SECURE is the DEFAULT because it is what "
                         "this program already established: run_ehn_relax_fleet.py "
                         "defaults --runpod-tier SECURE, vu_confirm.py hardcodes "
                         "cloud_type='SECURE', and numerical_methods_outline.md "
                         "says to 'fall back to RunPod SECURE (dedicated by "
                         "design) when Vast is dry'. I defaulted to COMMUNITY for "
                         "the lower price and hit exactly the capacity wall that "
                         "guidance exists to avoid (A5000 and 4090 both empty, "
                         "2026-07-29). ~1.7-2x the price, dedicated pods.")
    ap.add_argument("--provider", choices=["vast", "runpod"], default="vast",
                    help="runpod is ON-DEMAND (interruptible=False) rather than "
                         "spot, which is where its reliability comes from; vast "
                         "is cheaper per hour but pays it back in failovers")
    ap.add_argument("--gpu", default=None,
                    help="default RTX_3090 (vast) / 'RTX A5000' (runpod, 24 GB "
                         "at the best COMMUNITY price seen 2026-07-29)")
    ap.add_argument("--max-dph", type=float, default=0.25,
                    help="hard price ceiling; 3090 spot was 0.113 on 2026-07-29")
    ap.add_argument("--ready-timeout", type=int, default=2400,
                    help="seconds for a host to reach /tmp/worker-ready. "
                         "FleetExecutor defaults to 1200; a RunPod SECURE A5000 "
                         "was placed and RUNNING, then HostProbeFailed ~19 min in "
                         "-- suspiciously close to that 1200. onstart pulls "
                         "miniconda + conda-forge jax, and the Vast host that "
                         "SUCCEEDED had 830 Mbps down. If a slower datacenter link "
                         "just needs longer, 20 min is the whole bug. Raised to 40.")
    ap.add_argument("--run-timeout", type=int, default=5400,
                    help="HARD wall-clock cap per leg (s). Bounds the bill.")
    ap.add_argument("--out", default="output/b2",
                    help="provider is appended: output/b2_vast, output/b2_runpod. "
                         "The default composes to output/b2_vast, which is where "
                         "arm 1's fetched field.npz already lives -- do not "
                         "rename it or the resume loses a completed arm.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the leg and the cost bound; no key, no API, no spend")
    ap.add_argument("--smoke", action="store_true",
                    help="rent one box, verify ssh/ship/jax/fetch/teardown with a "
                         "trivial command, destroy. Minutes and cents, not the "
                         "40-minute leg -- run this after ANY credential change.")
    args = ap.parse_args()

    commit, dirty = local_engine_state()
    if dirty and not args.dry_run:
        # The commit we stamp onto the certificate must describe the BYTES we
        # ship. With payload files uncommitted it would not, and the certificate
        # would name a tree that never ran.
        print("PREFLIGHT FAILED -- payload files are uncommitted, so the "
              "ENGINE_COMMIT stamp would not describe what ships:")
        for d in dirty:
            print(f"  - {d}")
        return 2
    # Checked even on --dry-run, because this is the one preflight whose answer you
    # want BEFORE planning a run, not after paying for one.
    stale = unpushed_blockers()
    if stale:
        print("PUSH-STATE PREFLIGHT: the rented box installs both repos from "
              "origin/main, and would NOT get:")
        for s in stale:
            print(f"  - {s}")
        print("  Push is blocked on this machine -- see the transfer-handoff "
              "bundle protocol. Renting now would test the PREVIOUS code and "
              "report OK.")
        if not args.dry_run:
            return 2
    else:
        print("push-state: both repos' HEAD are on origin/main")

    if not args.dry_run:
        bad = preflight()
        if bad:
            print("PREFLIGHT FAILED -- not renting anything:")
            for b in bad:
                print(f"  - {b}")
            return 2
        print("preflight: ssh key present and registered with Vast")

    # Provider goes in the path: otherwise a RunPod run silently SKIPs because a
    # previous VAST run left the done_when marker, and you learn nothing while
    # believing you tested something.
    outdir = HERE / (args.out + f"_{args.provider}"
                     + ("_smoke" if args.smoke else ""))
    if args.smoke:
        leg = FleetLeg(label="SMOKE", command=SMOKE_COMMAND,
                       ship=(str(HERE / "standard_box.py"),),   # prove ship works
                       fetch="smoke", done_when="smoke/DONE")
    else:
        # resumable: FleetExecutor then pulls the fetch dir every ~120 s mid-run,
        # so a spot host dying at minute 35 of 40 does not cost the whole leg.
        leg = FleetLeg(label="B2_N192", command=build_command(commit),
                       ship=PAYLOAD, fetch="out_sbx_battery_full",
                       done_when="out_sbx_battery_full/DONE", resumable=True)

    if args.dry_run:
        print(f"1 leg -> {outdir}")
        print(f"  ship ({len(PAYLOAD)}):")
        for f in PAYLOAD:
            print(f"    {'OK     ' if Path(f).exists() else 'MISSING'} "
                  f"{Path(f).name}  <- {Path(f).parent.name}/")
        print(f"  fetch: {leg.fetch}   done_when: {leg.done_when}")
        print(f"  gpu={args.gpu} max_dph=${args.max_dph}/hr "
              f"timeout={args.run_timeout}s")
        print(f"  COST BOUND: {args.run_timeout/3600:.2f} hr x ${args.max_dph}/hr "
              f"= ${args.run_timeout/3600*args.max_dph:.2f} worst case")
        print(f"  expected: ~0.7 hr at $0.113/hr = ~$0.08 "
              f"(archive wall times were 20-37 min/arm)")
        print(f"  engine_commit: {commit[:16]}"
              + ("  (DIRTY: " + ", ".join(dirty) + ")" if dirty else "  (clean)"))
        print(f"\n  command:\n    {build_command(commit)}")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    ledger = VastLedger(outdir / "vast_ledger.jsonl")
    if args.provider == "runpod":
        # Key read from ~/tokens IN PLACE -- not symlinked into ~, and never put
        # on a command line or into a proc --env, where it would land in the unit
        # and the transcript.
        key = Path("~/tokens/.runpod_api_key").expanduser().read_text().strip()
        provider = RunPodProvider(api_key=key, ledger=ledger,
                                  cloud_type=args.cloud_type,
                                  interruptible=False, min_cuda=12.2)
        # A5000 (24 GB) suffices at N=192: run_ehn_relax_fleet.py records the
        # backward pass holding ~70 N^3 arrays => ~24 GB at 320^3, so ~5 GB here.
        # That file uses RTX_A6000 (48 GB) for the 320^3 ladder -- size up there.
        gpu = args.gpu or "RTX A5000"
    else:
        provider = VastProvider(ledger=ledger)
        gpu = args.gpu or "RTX_3090"
    launch = LaunchSpec(image=IMG, onstart=ONSTART, disk_gb=32,
                        label="b2-cert")      # reap can attribute by label
    spec = HostSpec(gpu_name=gpu, num_gpus=1, max_dph=args.max_dph,
                    min_reliability=0.97 if args.provider == "vast" else 0.0,
                    min_inet_mbps=300, min_cuda=12.2)
    ex = FleetExecutor(provider, launch, local_out_dir=str(outdir),
                       host_spec=spec, ready=SentinelReady(), max_parallel=1,
                       ready_timeout=args.ready_timeout,
                       run_timeout=args.run_timeout, ledger=ledger)
    results = ex.run([leg])
    print("=" * 60)
    for r in results:
        print(f"  {r.label}: {r.status}" + (f"  ({r.detail})" if r.detail else ""))
    print(f"ledger: {ledger.path}")

    # The REMOTE command's exit status, which the fleet layer cannot see: it only
    # knows the done_when marker appeared. On 2026-08-01 the marker said complete
    # while DONE recorded exit=1 and field.npz was truncated, and this script
    # printed "B2_N192: OK" over both -- the done_when trap wearing the marker's
    # own clothes. `; echo exit=$?` (see leg_cmd) exists precisely so the code is
    # recoverable; nothing read it until now.
    remote_rc = _remote_exit(outdir, leg)
    if remote_rc:
        print(f"  REMOTE COMMAND FAILED: {leg.done_when} records exit={remote_rc}"
              "  <-- the leg's artifacts are suspect even if status says OK")

    # Instance is a dataclass (id/status/dph/raw), NOT a dict -- an earlier
    # version called .get() here and would have raised precisely when the
    # teardown check mattered most, turning the safety net into a crash.
    live = provider.list_instances()
    print(f"\nTEARDOWN CHECK: {len(live)} instance(s) still live"
          + ("" if not live else "  <-- DESTROY THESE"))
    for i in live:
        print(f"  id={i.id} status={i.status} dph=${i.dph:.4f}/hr")
    if live:
        print("  destroy with: python -c \"from run_farm import VastProvider as V;"
              " p=V(); [p.destroy(i.id) for i in p.list_instances()]\"")
    print("\nnext: python standard_box.py --battery --legs B2   "
          "(scores the fetched artifacts locally -> citable certificate)")
    # LegResult.status is "OK" (upper); a lowercase compare made the smoke test
    # exit 1 on a fully successful run, which would have misreported the real leg.
    # remote_rc is ANDed in: a leg can be transported perfectly and still have run
    # a command that failed, and those are different facts about the same run.
    fleet_ok = all(str(r.status).lower() == "ok" for r in results)
    return 0 if (fleet_ok and not remote_rc) else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
