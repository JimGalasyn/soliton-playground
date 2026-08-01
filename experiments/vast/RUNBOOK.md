# Vast.ai decay-hunt runbook

One rented consumer GPU = one sweep leg. Public-side payload only
(engine + IC field + `decay_hunt_batch.py`, which is pure jax-solitons);
all knot ID / census stays local (relax-then-ID on the returned finals).

## One-time setup
```bash
pip install vastai
vastai set api-key <key-from-account-page>
ssh-keygen -f ~/.ssh/vastai -N ""        # dedicated key for rented hosts
vastai create ssh-key "$(cat ~/.ssh/vastai.pub)"
```

## Rent (cheapest reliable 3090/4090)
```bash
vastai search offers 'gpu_name in [RTX_3090,RTX_4090] num_gpus=1 \
    reliability>0.95 inet_down>200' -o 'dph+'
vastai create instance <OFFER_ID> \
    --image nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
    --disk 24 --onstart vast/onstart.sh
vastai show instances   # note INSTANCE_ID, ssh host/port
```

## Run a leg
```bash
# up: driver + IC (93 MB)
vastai copy ../decay_hunt_batch.py <ID>:/workspace/
vastai copy ../../gauged_relaxer/output/knot_q25_Q11_ic_N160.npz <ID>:/workspace/

ssh -i ~/.ssh/vastai -p <PORT> root@<HOST> \
  "cd /workspace && python3 decay_hunt_batch.py \
      --ic knot_q25_Q11_ic_N160.npz --eps 1.25 --nseeds 8 --bench"

# down: manifest-registered finals; then RELEASE THE METER
vastai copy <ID>:/workspace/output/decay_hunt_batch ./output/from_vast/
vastai destroy instance <ID>
```

## Then locally
```bash
# verdicts: relax-then-ID each returned final (fragment_census.py for decays)
python id_snapshot.py output/from_vast/.../snap.npz --relax 400
```

Notes
- **PyPI (Fastly) and GitHub throttle Vast datacenter IP ranges** —
  measured 2026-06-12 on three hosts (CA/Quebec/Nevada): Fastly 0.9KB/s,
  GitHub 35KB/s, all mirrors (nvidia/aliyun/tsinghua) similar, while
  Cloudflare ran 29MB/s and CloudFront 55MB/s on the SAME hosts. It is
  not the host's pipe; don't bother re-renting over it.
- **Delivery paths that work:** (1) JAX+CUDA via **conda-forge**
  (anaconda.org rides Cloudflare, 43MB/s):
  `conda create -n jaxenv -c conda-forge "jaxlib=*=*cuda*" jax scipy`;
  (2) the engine + driver + IC by **scp through the vast proxy**
  (inbound is fast) — engine is pure Python, run with
  `PYTHONPATH=/workspace/src`, no pip at all.
- Still probe outbound in onstart (bails to /tmp/worker-bad-network) —
  it cost three rentals (~$0.40) to learn the pattern; the probe makes
  any NEW pattern visible immediately.
- Spot/interruptible instances are fine: finals are full-state engine
  checkpoints (bit-identical restart, R4) — re-rent and resume.
- Cost compass (live 2026-06-12): 3090 $0.122/hr, 4090 $0.281/hr
  (reliability-filtered floor). A full 1,400-run campaign ~ $8 serial,
  ~$1-2 vmap-batched. AWS spot best (us-east-2): g6e L40S $0.41/hr.
- Don't put private-repo code, tokens, or the main SSH key on rented
  hosts. The IC field + engine are public-side.
