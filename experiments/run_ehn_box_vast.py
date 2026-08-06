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
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from run_farm.budget import BudgetExceeded, CappedProvider, estimate
from run_farm.fleet import FleetExecutor, FleetLeg, SentinelReady
from run_farm.gauntlet import GauntletError, require_gauntlet, standard_gauntlet
from run_farm.protocols import HostSpec, LaunchSpec
from run_farm.vast import VastLedger, VastProvider

from soliton_playground.ehn_lab.chamber import preflight as envelope_preflight

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
IMG = "nvidia/cuda:12.2.2-runtime-ubuntu22.04"
ONSTART = (HERE / "vast" / "onstart.sh").read_text()
PIP_ENGINE = "git+https://github.com/JimGalasyn/jax-solitons"
PIP_LAB = "git+https://github.com/JimGalasyn/soliton-playground"


def relax_args(a, out):
    """The engine invocation, without any of the launch scaffolding."""
    return (
        f"-m jax_solitons.ehn.relax "
        f"--geom torus --tp {a.tp} --tq {a.tq} --R {a.R} "
        f"--N {a.N} --L {a.L} --C {a.C} --U 50 "
        f"--alpha {a.alpha} --beta 2e-3 --cramp 8000 --agrad wrapped "
        f"--ic screened --steps {a.steps} --samples {a.samples} "
        f"--topo-every {a.topo_every} "
        f"--det-every {a.det_every} --det-timeout {a.det_timeout} "
        f"--save-every {a.save_every} --out {out}")


def build_command(engine_argv, engine_commit, out="out_ehn_box", env=None):
    """A RE-ENTERABLE leg: launch the relaxation detached, then wait on the marker.

    `engine_argv` is the engine invocation (see relax_args) and `out` its output
    directory, both parameters rather than derived from this script's argparse, so
    a multi-leg campaign can reuse this contract instead of copying it. The copy is
    what would drift: every branch below was paid for by a specific failure, and a
    second transcription would not inherit the next fix.

    WHY NOT JUST RUN IT. The old shape ran the engine as the ssh command's own
    child, so the relaxation died with the channel. On 2026-08-03 `ssh5.vast.ai`
    stopped responding 5500 steps into a 12000-step N=320 run; ssh returned 255,
    the A100 was perfectly alive, and the leg was filed terminal RUN_FAIL -- so the
    box was destroyed with its checkpoint still on it. Detaching alone would not
    have saved it, because the fleet tears the host down when the leg returns; what
    saves it is `reattachable=True` (run-farm) plus a command that can be re-run
    against the same box WITHOUT starting a second relaxation.

    That is the contract this script implements, and every branch of it exists
    because the naive version gets something wrong:

      DONE exists          the job finished during the blip -> exit 0, touch nothing
      PID live on THIS box a reattach -> resume WAITING, do not relaunch
      otherwise            launch detached, record boot_id + pid

    The pidfile records the BOOT ID as well as the pid, and both must match. That
    is not paranoia: `resumable=True` restores the local partial onto a REPLACEMENT
    host, PID file included, where that number means a different process or none at
    all -- so a bare pid check would 'reattach' to an unrelated process and wait out
    the whole timeout.

    `; echo exit=$?` into DONE is kept: the marker must carry the exit code so a
    fetched-but-failed leg is distinguishable from a fetched-and-fine one. Note the
    exit code the FLEET sees is now the wait loop's, which is what makes run-farm's
    255-means-transport inference sound -- the payload can no longer produce a 255.
    """
    py = "/workspace/jaxenv/bin/python"
    # A heredoc rather than a nested-quote one-liner: this needs `if`, a loop and a
    # subshell, and the version of it that fits on one line is unreadable and was
    # where the quoting bugs lived.
    return (
        f"cd /workspace && mkdir -p {out} && "
        f"export ENGINE_COMMIT={engine_commit} && "
        # Worker environment exported HERE rather than via the executor, because
        # FleetExecutor has no remote_env (that is ProviderExecutor's) and a
        # non-interactive `ssh host cmd` sources no profile, so onstart's exports
        # never reach the payload. Putting it in the command means it also appears
        # in whatever records the command, instead of living in launcher config.
        + "".join(f"export {k}={shlex.quote(str(v))} && "
                  for k, v in sorted((env or {}).items())) +
        # pyknotid is listed EXPLICITLY rather than via jax-solitons' `knots` extra.
        # It is optional upstream for a good reason -- tracing curves needs only
        # numpy/scipy, only the Alexander determinant needs pyknotid -- and this leg
        # is precisely the caller that needs it. Naming it here means the dependency
        # of THIS run is legible in the command the manifest records, instead of
        # hiding behind extras resolution.
        #
        # Its absence cost the 2026-08-03 rental its entire measurement: 73 samples
        # of `det1 = [[2352, 'e:ImportError']]`, $1.50, and a leg that reported OK.
        f"/workspace/jaxenv/bin/pip install -q '{PIP_ENGINE}' '{PIP_LAB}' "
        f"'pyknotid>=0.5' && "
        f"cat > leg.sh <<'EOSCRIPT'\n"
        f"set -u\n"
        f"OUT={out}\n"
        f"PY={py}\n"
        f"BOOT=$(cat /proc/sys/kernel/random/boot_id)\n"
        f'if [ -f "$OUT/DONE" ]; then echo "already complete: $(cat $OUT/DONE)"; exit 0; fi\n'
        f'RUNNING=no\n'
        f'if [ -s "$OUT/PID" ]; then\n'
        f'  read -r PBOOT PPID_ < "$OUT/PID" || true\n'
        f'  if [ "$PBOOT" = "$BOOT" ] && kill -0 "$PPID_" 2>/dev/null; then RUNNING=yes; fi\n'
        f'fi\n'
        f'if [ "$RUNNING" = yes ]; then\n'
        f'  echo "REATTACHED to live pid $PPID_ on this box; not relaunching"\n'
        f'else\n'
        # ---- on-box preflight, on the LAUNCH path only ---------------------------
        # The gauntlet checks the local launch environment exhaustively; nothing
        # checked whether the REMOTE box can perform the thing we are renting it for.
        # Identifies a known T(2,3) and requires det == 3, so it exercises the whole
        # path -- pyknotid import, the numpy-alias shim, the Alexander routine -- and
        # not merely that a module resolves. Under a second, and it runs BEFORE the
        # relaxation: it turns 2.7 h and $1.50 of non-measurement into an early exit.
        #
        # Here rather than at the top of the script, because a reattach has already
        # been validated by the launch that preceded it, and an already-complete leg
        # has nothing left to validate. `exit=90` goes into DONE so the marker
        # distinguishes this from a relaxation that failed: the fleet reads it and
        # reports FETCHED BUT FAILED rather than a bare nonzero.
        f'  if ! "$PY" -c "from jax_solitons.knots import identify_knot, torus_knot; '
        f'assert identify_knot(torus_knot(2, 3))[\'determinant\'] == 3" '
        f'> "$OUT/preflight.log" 2>&1; then\n'
        f'    echo "PREFLIGHT FAILED — this box cannot identify knots, so the run '
        f'would relax for hours and record no determinant. Nothing launched."\n'
        f'    cat "$OUT/preflight.log"\n'
        f'    echo "exit=90" > "$OUT/DONE"\n'
        f'    exit 90\n'
        f'  fi\n'
        f'  echo "preflight OK: T(2,3) identifies as det 3 on this box"\n'
        f'  RESUME=""\n'
        f'  if [ -f "$OUT/field.npz" ]; then\n'
        # Structural check only (reads the zip central directory), which is exactly
        # what a truncated transfer breaks and is O(1) rather than CRC-ing 3.7 GB.
        # It does NOT prove the members are intact; np.load would fail later if not.
        f'    if "$PY" -c "import zipfile; zipfile.ZipFile(\'$OUT/field.npz\').namelist()" 2>/dev/null; then\n'
        f'      RESUME="--resume $OUT/field.npz"; echo "RESUMING from $OUT/field.npz"\n'
        f'    else\n'
        # Deleted, not renamed: it is provably unopenable, and anything left inside
        # $OUT gets fetched -- so keeping it would drag GB of known garbage back
        # over the same link that tore it.
        f'      echo "DISCARDING $OUT/field.npz: not a readable zip (torn restore)"\n'
        f'      rm -f "$OUT/field.npz"\n'
        f'    fi\n'
        f'  fi\n'
        f'  setsid nohup bash -c "$PY {engine_argv} $RESUME ; '
        f'echo \\"exit=\\$?\\" > $OUT/DONE" > "$OUT/run.log" 2>&1 < /dev/null &\n'
        f'  echo "$BOOT $!" > "$OUT/PID"\n'
        f'  echo "LAUNCHED detached: $(cat $OUT/PID)"\n'
        f'fi\n'
        # The payload can die without writing DONE (an OOM kill takes the shell
        # too). Noticing that costs one poll; not noticing it burns the entire
        # --run-timeout on a box that is doing nothing, which is money.
        f'while [ ! -f "$OUT/DONE" ]; do\n'
        f'  read -r PBOOT PPID_ < "$OUT/PID" || true\n'
        f'  if ! kill -0 "$PPID_" 2>/dev/null; then\n'
        f'    sleep 5\n'
        f'    if [ ! -f "$OUT/DONE" ]; then\n'
        f'      echo "payload pid $PPID_ vanished with no DONE marker"; exit 1\n'
        f'    fi\n'
        f'  fi\n'
        f'  sleep 15\n'
        f'done\n'
        f'echo "marker: $(cat $OUT/DONE)"\n'
        f"EOSCRIPT\n"
        f"bash leg.sh")



def remote_exit(outdir, leg):
    """The exit status the REMOTE command recorded in its done_when marker.

    `LegResult.ok` means the marker ARRIVED, not that the job worked. This run
    proved the difference the expensive way: a 51-minute rental reported
    "LEG T2_5_N320_R64: OK" while the marker read exit=1 and no field.npz existed
    -- the box OOMed 0.92 s in. Reading the marker is what separates "the
    artifacts arrived" from "the job did something".

    None means NO EVIDENCE, never success: no marker, an unparseable body, or a
    done_when that is not an exit= file. A caller that treats None as a pass has
    reintroduced the bug.
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


#: The paths whose state can change what the BOX runs: the installed package, its
#: metadata, and the driver that composes the command. Deliberately NOT `output/`
#: — a run in flight rewrites its own manifest continuously, and blocking a rental
#: on a results file that pip never sees is the check firing on the wrong thing.
PAYLOAD = ("src", "pyproject.toml", "experiments")


def unpushed_blockers():
    """Refuse to rent while the box would install code OLDER than what we ran.

    The box does `pip install git+https://.../main` for BOTH repos, so anything
    not on origin/main is simply absent there, and the run would exercise a
    different engine than the one whose commit the manifest records.

    The dirty check is scoped to PAYLOAD for that reason: the question is whether
    what pip installs matches what we believe, not whether every file in the tree
    is committed. The HEAD-on-origin/main check stays unscoped, because a commit
    that has not been pushed is absent from the box no matter which paths it
    touches.
    """
    out = []
    for label, repo in (("jax-solitons", REPO.parent / "jax-solitons"),
                        ("soliton-playground", REPO)):
        try:
            run = lambda *x: subprocess.run(x, cwd=str(repo), capture_output=True,
                                            text=True, timeout=20)
            head = run("git", "rev-parse", "HEAD").stdout.strip()
            dirty = run("git", "status", "--porcelain", "--",
                        *PAYLOAD).stdout.strip()
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
                    help="checkpoints; the field is the deliverable. All writes go "
                         "to ONE field.npz (~3.7 GB at N=320), so this bounds how "
                         "much progress a timeout costs, not disk.")
    ap.add_argument("--topo-every", type=int, default=1,
                    help="record Lk(phi1,phi2) + per-species segment counts every "
                         "Kth sample. The ENGINE defaults this to 0 = off, which "
                         "would have made this rental produce energy and Q only -- "
                         "and the trajectory IS the deliverable: the question is "
                         "whether the phi1 self-knot survives, and a single "
                         "end-of-run diagnostic cannot say WHEN it stopped. K=1 "
                         "gives 500-step resolution at --steps 12000 --samples 24, "
                         "finer than the 1200-step local run whose reconnection "
                         "showed no energy signature. Costs a host-side skeleton "
                         "extraction per sample; failures are caught per-sample and "
                         "the checkpoint protects the run regardless.")
    ap.add_argument("--det-every", type=int, default=1,
                    help="phi1 SELF-KNOT determinant every Kth sample, computed ON "
                         "THE BOX into the manifest (jax-solitons 9a449fc). This is "
                         "the measurement that decides the run, and it is here "
                         "rather than computed locally from field.npz because on "
                         "2026-08-03 that field truncated 3.7 GB -> 2.1 GB crossing "
                         "a vast SSH proxy and took the whole deliverable with it. "
                         "Costs ~22 s/sample at N=320, ~9 min over 24 samples.")
    ap.add_argument("--det-timeout", type=float, default=180.0,
                    help="per-determinant wall-clock budget on the box. identify_knot "
                         "can grind for hours on a noisy evolved curve; on timeout "
                         "the sample records null and the descent continues.")
    ap.add_argument("--gpu", default="A100_SXM4",
                    help="N=320 needs an A100-class card. A 24 GB RTX_4090 was "
                         "tried and OOMed 0.92 s in, before one relaxation step: "
                         "reverse-mode AD over 10 fields at 320^3 float64 holds "
                         "far more than the 3.4 GB the state alone suggests. "
                         "examples/ehn_knot_soliton.py said A100-class; it was right.")
    ap.add_argument("--max-dph", type=float, default=0.60)
    ap.add_argument("--cap-usd", type=float, default=3.00,
                    help="ENFORCED ceiling on booked + in-flight spend for this "
                         "ledger, re-checked before every rent(). Distinct from "
                         "--max-dph, which caps the hourly RATE and therefore "
                         "cannot bound a total: a rate cap says nothing about how "
                         "many hosts get acquired.")
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
    # Cost as a quantity rather than a print. `hours x rate` omitted the tax that
    # has actually been paid on this leg: output/ehn_box_t25/vast_ledger.jsonl
    # records instance 46643731 destroyed as host_failed, "worker not ready within
    # 2400s", 2753 billed seconds, $0.2238 -- for a box that never came up.
    #
    # Vast bills WALL-CLOCK from `rented` to `destroyed` x dph, and the ledger says
    # so to four decimals on both of its records:
    #
    #   2753.6 s x $0.29263/hr = $0.2238   (46643731, host_failed)
    #    311.9 s x $0.33472/hr = $0.0290   (46646833, ok)
    #
    # So a failed acquisition costs the whole --ready-timeout, and the tax is set by
    # that timeout rather than by the tier being cheap. But that also makes it a
    # RATE, not a constant -- and $0.22 was measured on an RTX 4090 at $0.293/hr,
    # while this leg rents A100_SXM4 at up to $0.60/hr. Hard-coding 0.22 here
    # under-stated the tax by ~2x on the very tier it was about to be spent on, and
    # A100_SXM4 being scarcer argues the failure RATE is higher too, not lower.
    # Computed, so it tracks whatever --ready-timeout and --max-dph actually select.
    #
    # failure_tax stays 0.0 deliberately: it is a fraction of USEFUL gpu-hours, and
    # this leg has never produced one, so there is no denominator to derive it from.
    # Guessing a fraction here would dress an unknown up as a measurement.
    acq_tax = a.ready_timeout / 3600.0 * a.max_dph
    est = estimate(1, a.run_timeout, a.max_dph, acq_tax_usd=acq_tax, failure_tax=0.0)
    print(f"  COST: ${est['usd']:.2f} worst case = {est['gpu_h']:.2f} gpu-h at the "
          f"{a.run_timeout/3600:.2f} hr timeout x ${a.max_dph}/hr "
          f"+ ${acq_tax:.2f} acquisition tax "
          f"({a.ready_timeout/3600:.2f} hr ready-timeout x ${a.max_dph}/hr, "
          f"one failed host)")

    # reattachable: build_command's script is re-enterable (boot-id + pid guard,
    # then wait on the marker), so run-farm may retry the ssh channel against the
    # SAME live box instead of filing a transport failure as a failed job. Without
    # the re-enterable script this flag would start a second relaxation racing the
    # first; the two go together and neither is useful alone.
    leg = FleetLeg(label=label,
                   command=build_command(relax_args(a, "out_ehn_box"), commit),
                   ship=(), fetch="out_ehn_box",
                   done_when="out_ehn_box/DONE", resumable=True,
                   reattachable=True)

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
    #
    # Recorded while wiring this up: of the two repos the box installs, only
    # jax-solitons is REACHED. `grep -rn soliton_playground` over the engine's src/
    # is empty, and importing jax_solitons.ehn.relax pulls in no soliton_playground
    # module, so PIP_LAB is installed and never imported. Left in place rather than
    # removed -- dropping an install that both prior attempts carried is a change to
    # the remote environment, and this run is not the place to test it -- but it
    # bounds what unpushed_blockers() is actually protecting: the ENGINE commit is
    # what reaches the computation, and this driver never does.
    print()
    try:
        # ssh-key-registered is SKIPPED, not deleted, and the reason is on record:
        # the vast adapter exposes no registered_ssh_keys(), so the check cannot
        # verify and correctly refuses to pass. Verified out-of-band instead of
        # waved through -- GET https://console.vast.ai/api/v0/ssh/ returns four
        # keys, of which id 1154757 has fingerprint
        # SHA256:pFSw++WvyMjzJ+IJmv1lXulKpun4YJMT7mdQpJooSV4, matching
        # `ssh-keygen -lf ~/.ssh/vastai.pub` (comment vastai-fleet-IBM-85CB6G4,
        # ED25519). That is the same statement the check wanted to make; it just
        # could not make it through this provider.
        #
        # RE-VERIFIED 2026-08-03, and the previous note was stale: it named ids
        # 961037/961053 and fingerprint SHA256:qFr198sj..., which are a DIFFERENT
        # key that is still on the account. The local key had been rotated, so the
        # recorded evidence no longer described the key being skipped over. It
        # happened to remain safe, which is the problem with a stale verification --
        # it reads as checked. Re-run the GET above rather than trusting this line
        # if ~/.ssh/vastai changes again.
        #
        # The list is now `standard_gauntlet`'s rather than hand-rolled, for three
        # reasons the hand-rolled version got wrong:
        #
        #   1. ProviderCapable(provider) passed NO method names, so `required`
        #      defaulted to () and the check reported "has all 0 required method(s)"
        #      -- it could not fail. That is the same defect this file's comment
        #      calls out one line above about PayloadClosed, committed one line
        #      below it. standard_gauntlet supplies offers/rent/destroy as required
        #      and dead_reason/logs/list_instances as optional.
        #   2. Order. The assembly is cheapest-first by design: local and free
        #      before anything touching the network. The hand-rolled list ran
        #      SshKeyRegistered and OffersAvailable -- two API round trips -- ahead
        #      of OutDirWritable, which fails in milliseconds.
        #   3. payload=None omits PayloadClosed for us, so the reason it is absent
        #      lives in the assembly's contract instead of depending on whoever
        #      edits this list next remembering why.
        #
        # skip= goes to require_gauntlet, NOT to standard_gauntlet, and the
        # difference matters: standard_gauntlet's skip filters the check out of the
        # list, so it vanishes from the report and reads exactly like one that
        # passed. require_gauntlet's skip (run-farm d4682bd) emits it as SKIPPED
        # with "a skip proves only that someone chose to accept this risk
        # out-of-band". A bypass should appear in the report as a bypass.
        require_gauntlet(
            standard_gauntlet(provider=provider, host_spec=spec, out_dir=outdir,
                              legs=[leg], payload=None),
            skip=("ssh-key-registered",))
    except GauntletError:
        print("\nGAUNTLET FAILED — nothing rented, nothing spent.")
        return 6

    # ---- what was actually launched, recorded before anything is rented ---------
    # Review catch, 2026-08-03: the committed run record for the failed N=320 attempt
    # shows a ledger dph of 0.6152 against a driver default --max-dph of 0.60, and
    # vast.py filters offers on `dph_total <= max_dph`. So that run cannot have used
    # the committed defaults -- it was launched with --max-dph 0.65 -- and anyone
    # re-running from the defaults draws a SMALLER, possibly empty, A100_SXM4 pool
    # than the one that produced the record. (The README did say so in prose; the
    # reviewer read the record and not the prose, which is the point: a launch
    # parameter recorded only in a sentence someone remembered to write is recorded
    # by luck.)
    #
    # So the resolved flags go next to the ledger, automatically, every run -- and
    # BEFORE the rental, because a run that dies is exactly when you need to know
    # what it was asked to do.
    launch_record = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "engine_commit": commit,
                     "lab_commit": subprocess.run(
                         ["git", "rev-parse", "HEAD"], cwd=str(REPO),
                         capture_output=True, text=True).stdout.strip(),
                     "argv": sys.argv[1:],
                     "flags": vars(a),
                     "remote_command": leg.command}
    (outdir / "launch.json").write_text(json.dumps(launch_record, indent=1))
    print(f"  launch record -> {outdir/'launch.json'}")

    # ---- the cap, at the one seam money starts: rent() -------------------------
    # --max-dph caps the RATE; it cannot bound a TOTAL, because nothing in a rate
    # says how many hosts get acquired -- and the failover path acquires another
    # host on RentUnavailable or HostProbeFailed. CappedProvider counts booked
    # spend (teardown costs from the ledger) PLUS in-flight burn (open rentals at
    # dph x elapsed, which ledger.summary() alone does not see) and raises
    # BudgetExceeded BEFORE calling the inner rent, so an over-cap attempt creates
    # no host. It is itself a Provider, so this composes with no executor change:
    # FleetExecutor touches only provider.offers and provider.rent, both of which
    # the wrapper implements. The gauntlet above deliberately ran against the RAW
    # adapter, since destroy() -- the teardown surface -- lives there.
    #
    # The guarantee is a pre-rent gate, not a mid-rental tripwire: one box can
    # still overshoot by its own runtime, which --run-timeout bounds.
    capped = CappedProvider(provider, cap_usd=a.cap_usd, ledger=ledger)
    print(f"  cap ${a.cap_usd:.2f}, already spent on this ledger "
          f"${capped.spent_usd():.4f} -> ${a.cap_usd - capped.spent_usd():.2f} "
          f"of headroom")

    ex = FleetExecutor(capped, launch, local_out_dir=str(outdir),
                       host_spec=spec, ready=SentinelReady(),
                       ready_timeout=a.ready_timeout, run_timeout=a.run_timeout,
                       ledger=ledger, max_parallel=1)
    # ex.run returns list[LegResult], NOT a bool. `if ex.run(...)` is truthy for a
    # non-empty list, so a RUN_FAIL / BAD_ARTIFACTS leg would have exited 0 --
    # a failure reading as success, which is the exact class of bug LegResult's
    # own docstring says BAD_ARTIFACTS was split out to prevent.
    reap = f"python -m run_farm.reap --ledger {outdir/'vast_ledger.jsonl'} --yes"
    try:
        results = ex.run([leg])
    except BudgetExceeded as e:
        # A deliberate stop, not a bad host: it must halt rather than fail over to
        # the next offer, so it is caught here instead of inside the failover path.
        #
        # This clause was UNREACHABLE as first written, and the exit code with it.
        # `BudgetExceeded` lived in run_farm.budget, unknown to run_farm.fleet, so
        # FleetExecutor's catch-all swallowed it into LegResult(status="ERROR") --
        # the cap still worked, but it reported as an ordinary host failure and the
        # campaign carried on to attempt a rent per remaining leg. Fixed in run-farm
        # c08c712: the exception moved to run_farm.protocols beside the other
        # provider-raised signals, fleet re-raises it, and unstarted legs are
        # cancelled. REQUIRES run-farm >= c08c712; against an older run-farm this
        # path goes quiet again rather than misfiring.
        print(f"\nBUDGET CAP — refused before creating a host: {e}")
        print(f"  a host rented EARLIER in this run may still be live: {reap}")
        return 7

    if not results:
        print("  NO LEGS RAN — treat as failure, not as success")
        print(f"  check for a live box first: {reap}")
        return 1

    bad = False
    for r in results:
        code = remote_exit(outdir, leg)
        # LegResult.ok says the marker ARRIVED. The marker's CONTENT says whether
        # the job worked. Both must be good, and "no evidence" is not good.
        verdict = ("OK" if (r.ok and code == 0) else
                   "FETCHED BUT FAILED" if (r.ok and code not in (0, None)) else
                   "FETCHED, NO EXIT EVIDENCE" if r.ok else r.status)
        bad = bad or verdict != "OK"
        print(f"  {r.label}: fleet={r.status}  remote_exit={code}  -> {verdict}"
              + (f"  host={r.host_id}" if r.host_id else ""))
        if verdict != "OK":
            print(f"    artifacts are in {outdir/leg.label}; the relaxation did NOT "
                  f"complete, so any manifest there is a partial record")
    if bad:
        # rent()'s teardown-verify only covers the exit paths it can intercept. A
        # SIGKILL, a crash, or a destroy REST call that itself fails on a flaky
        # resolver all leave a GPU billing by the second, and this driver is meant
        # to be run detached. --ledger scope touches only boxes THIS ledger rented
        # and never recorded destroyed, so it is safe while other sessions farm.
        print(f"\n  verify nothing is still billing: {reap}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
