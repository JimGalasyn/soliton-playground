"""Rent ONE Vast box, run the real onstart, and watch /workspace/onstart.log to
find where the worker-ready build stalls/fails (the ladder's repeated
'worker not ready within 1200s'). rent() is a context manager with guaranteed
verify-teardown on exit, so the box is destroyed no matter how this ends.
"""
import os
import sys
import time
from pathlib import Path

from jax_solitons.campaign import VastProvider, HostSpec, LaunchSpec, VastLedger
from jax_solitons.campaign.provider_exec import _ssh, DEFAULT_KEY

ONSTART = Path(os.path.expanduser(
    "~/repos/null-worldtube-private/simulations/engine_dogfood/vast/onstart.sh")).read_text()
IMG = "nvidia/cuda:12.2.2-runtime-ubuntu22.04"
DEADLINE = 18 * 60
POLL = 45

ledger = VastLedger(Path(os.path.expanduser("~/campaigns/smoke/vast_ledger.jsonl")))
provider = VastProvider(ledger=ledger)
spec = HostSpec(gpu_name="RTX_4090", max_dph=0.6, min_reliability=0.97,
                min_inet_mbps=300, min_cuda=12.2)
launch = LaunchSpec(image=IMG, onstart=ONSTART, disk_gb=24, label="onstart-smoke")

offer = provider.cheapest_offer(spec)
if offer is None:
    print("NO OFFER matched spec", flush=True); sys.exit(1)
print(f"offer {offer.id} {offer.gpu_name} {offer.dph:.3f}$/hr "
      f"inet_down={getattr(offer, 'inet_down', '?')} cuda={getattr(offer, 'cuda_max', getattr(offer, 'cuda', '?'))} "
      f"{offer.geolocation.strip(', ')}", flush=True)

PROBE = (
    "echo ===LOG===; cat /workspace/onstart.log 2>/dev/null; "
    "echo ===MARKERS===; ls -la /tmp/worker-* 2>/dev/null; "
    "echo ===PROC===; ps -eo etimes,comm,args 2>/dev/null | "
    "grep -iE 'conda|mamba|onstart|pip|jaxlib' | grep -v grep | head; "
    "echo ===GPU===; nvidia-smi -L 2>&1 | head -2; "
    "echo ===ENVDIR===; du -sh /workspace/jaxenv 2>/dev/null; ls /workspace 2>/dev/null"
)

with provider.rent(offer, launch, timeout_s=900) as host:
    print(f"RUNNING ssh={host.ssh_host}:{host.ssh_port}", flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < DEADLINE:
        rc, out = _ssh(DEFAULT_KEY, host.ssh_host, host.ssh_port, PROBE, timeout=60)
        el = int(time.monotonic() - t0)
        print(f"\n========== t+{el}s  ssh_rc={rc} ==========", flush=True)
        print(out if out.strip() else "(no output / ssh not up yet)", flush=True)
        if "worker-ready" in out:
            print("\n*** WORKER READY -- onstart succeeded ***", flush=True); break
        if "worker-bad-network" in out:
            print("\n*** WORKER BAD-NETWORK -- onstart hit a fail branch ***", flush=True); break
        time.sleep(POLL)
    else:
        print(f"\n*** reached {DEADLINE}s with neither sentinel -- onstart STALLED ***", flush=True)

print("smoke done; box torn down by rent() finally", flush=True)
