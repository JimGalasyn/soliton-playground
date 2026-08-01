"""Confirm the key fix worked: for each leg's current running instance, SSH in
(now that ~/.ssh/vastai is present) and check /tmp/worker-ready + whether
eps_kick_batch is actually running. Exits when both legs are confirmed computing,
or a box dies, or the deadline. Positive confirmation, not just 'didn't die'.
"""
import json
import os
import re
import time
from pathlib import Path

from jax_solitons.campaign import VastProvider
from jax_solitons.campaign.provider_exec import _ssh, DEFAULT_KEY

LEGS = ["eps-N128", "eps-N160"]
CAMP = Path(os.path.expanduser("~/campaigns"))
DEADLINE = 16 * 60
provider = VastProvider()


def latest_running(ledger):
    inst = None
    for line in ledger.read_text().splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("event") == "running":
            inst = d["instance_id"]
        if d.get("event") == "destroyed" and d.get("instance_id") == inst:
            inst = None
    return inst


PROBE = ("ls /tmp/worker-ready /tmp/worker-bad-network 2>/dev/null; "
         "echo P=$(pgrep -fc eps_kick_batch); "
         "echo ---; tail -2 /workspace/onstart.log 2>/dev/null")

t0 = time.monotonic()
confirmed = set()
while time.monotonic() - t0 < DEADLINE and len(confirmed) < len(LEGS):
    el = int(time.monotonic() - t0)
    print(f"\n===== t+{el}s =====", flush=True)
    for leg in LEGS:
        if leg in confirmed:
            continue
        iid = latest_running(CAMP / leg / "vast_ledger.jsonl")
        if not iid:
            print(f"  {leg}: no running instance in ledger", flush=True)
            continue
        try:
            inst = provider.status(iid)
            host = str(inst.raw.get("ssh_host") or "")
            port = int(inst.raw.get("ssh_port") or 0)
        except Exception as e:
            print(f"  {leg}: status({iid}) err {e}", flush=True)
            continue
        if not host or not port:
            print(f"  {leg}: inst {iid} running, ssh coords not yet populated", flush=True)
            continue
        rc, out = _ssh(DEFAULT_KEY, host, port, PROBE, timeout=40)
        ready = "worker-ready" in out
        m = re.search(r"P=(\d+)", out)
        nproc = int(m.group(1)) if m else 0
        flag = "READY+COMPUTING" if (ready and nproc) else (
            "ready(no-kick-yet)" if ready else "building/onstart")
        print(f"  {leg}: inst {iid} {host}:{port}  {flag}  (kick procs={nproc})", flush=True)
        print("   " + out.replace("\n", "\n   "), flush=True)
        if ready and nproc:
            confirmed.add(leg)
            print(f"  *** {leg} CONFIRMED: worker-ready + eps_kick_batch running ***", flush=True)
    if len(confirmed) < len(LEGS):
        time.sleep(75)

print(f"\nCONFIRMED {sorted(confirmed)}  ({len(confirmed)}/{len(LEGS)})", flush=True)
