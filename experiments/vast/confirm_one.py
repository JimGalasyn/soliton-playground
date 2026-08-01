"""One-shot box probe: confirm_one.py <instance_id>. Gets ssh coords via the
provider and checks worker-ready + eps_kick_batch running + GPU memory (does the
seed-batch vmap fit, or is it about to OOM?)."""
import sys
from jax_solitons.campaign import VastProvider
from jax_solitons.campaign.provider_exec import _ssh, DEFAULT_KEY

iid = int(sys.argv[1])
inst = VastProvider().status(iid)
host = str(inst.raw.get("ssh_host") or ""); port = int(inst.raw.get("ssh_port") or 0)
print(f"inst {iid}  ssh={host}:{port}")
probe = ("ls /tmp/worker-ready 2>/dev/null; echo kick=$(pgrep -fc eps_kick_batch); "
         "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu "
         "--format=csv,noheader 2>/dev/null; "
         "echo ---; tail -3 /workspace/onstart.log 2>/dev/null")
rc, out = _ssh(DEFAULT_KEY, host, port, probe, timeout=40)
print(f"rc={rc}\n{out}")
