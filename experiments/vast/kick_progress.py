"""kick_progress.py <instance_id> — peek at on-box eps_kick_batch progress:
the eps_kick_batch proc + runtime, its out_kick dir milestone files (with mtimes),
and GPU. Milestones in order: bare_relaxed -> full_relaxed -> bare_eps*.npz
(bare kick done) -> full_eps*.npz (full kick done) -> manifest.json."""
import sys
from jax_solitons.campaign import VastProvider
from jax_solitons.campaign.provider_exec import _ssh, DEFAULT_KEY

iid = int(sys.argv[1])
inst = VastProvider().status(iid)
host = str(inst.raw.get("ssh_host") or ""); port = int(inst.raw.get("ssh_port") or 0)
print(f"inst {iid}  ssh={host}:{port}")
probe = (
    "P=$(pgrep -f eps_kick_batch.py | head -1); "
    "echo \"pid=$P  elapsed=$(ps -o etime= -p $P 2>/dev/null)\"; "
    "D=$(readlink /proc/$P/cwd 2>/dev/null)/out_kick; echo \"out_kick=$D\"; "
    "ls -la --time-style=+%H:%M $D 2>/dev/null; "
    "echo ---now=$(date +%H:%M)---; "
    "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null"
)
rc, out = _ssh(DEFAULT_KEY, host, port, probe, timeout=40)
print(out)
