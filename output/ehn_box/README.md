# EHN-box N=320 attempt, 2026-08-03 — FAILED IN TRANSIT, kept for what it settles

A T(2,5) relaxation at EHN's own resolution (320³, L=256, R=64 = 0.25 L), run to
separate two readings of the local cinquefoil's reconnection. **It did not answer
that question.** The record is kept because it settles a different one, and because
the trajectory is the only surviving artifact of a paid run.

Same precedent as `output/ehn_box_t25/` and `output/alpha_formation/`: a run record
kept without its field.

## What happened

| | |
|---|---|
| host | A100 SXM4, Alberta CA, reliability 0.9948, **$0.6152/hr** |
| instance | 46720095, provisioned in **127.3 s** |
| billed | 2468.6 s = **$0.4218**, destroyed, `verify: gone` |
| reached | **n=5500 of 12000** steps, 12 samples, wall 1357 s |
| failed | `rc=255: Timeout, server ssh5.vast.ai not responding` |

The SSH proxy stopped responding mid-run. The box itself was healthy and was torn
down cleanly; `reap --ledger` confirmed nothing left billing. This is neither an OOM
nor a configuration error — it is the network path between here and a live box
failing while a 3.7 GB file was coming across it.

## What this run DOES establish

1. **N=320 runs on an A100 without OOM.** The previous attempt (`../ehn_box_t25/`)
   died 0.92 s in on a 24 GB RTX 4090, before a single relaxation step. This one
   completed 5500 steps. The `--gpu A100_SXM4` correction is validated, and so is
   the rest of the configuration at this box size.
2. **Throughput: ~4 steps/s**, so 12k steps is ≈50 min of relaxation — comfortably
   inside the 2.5 hr `--run-timeout`. The timeout was never the binding constraint.
3. **`--topo-every 1` works and is load-bearing.** The `xlk`/`nseg1`/`nseg2` columns
   below exist only because it was passed; the engine defaults it to 0 = off.
4. **The C-ramp is exactly on schedule.** C = 275 at n = 5500 against `cramp=8000`
   (400 × 5500/8000 = 275).

## What it does NOT show

**Nothing about whether the φ₁ self-knot is topologically protected.** The local
cinquefoil that motivated this run reconnected between **33k and 36k** steps. This
run reached **5500** — about a sixth of that descent distance, at the same alpha. Lk
holding at −5.0 and the segment counts not moving is therefore unremarkable: the
local box held to 33k too. Read this as a null of no consequence, not as support for
either hypothesis.

**No determinant.** The manifest carries Lk(φ₁,φ₂) and per-species segment counts,
but the det-5-to-1 signal needs `identify_knot` on the field, and the field did not
survive (below). `nseg1` is a reconnection *proxy* only.

**The field is gone.** `field.npz` arrived at 2.1 GB against the ~3.7 GB the engine
documents, and is **not a readable zip** — `zipfile.ZipFile` refuses to open it, so
it is a torn download rather than a recoverable one. Deleted rather than kept: a
2.1 GB file named `field.npz` that fails only on open is precisely the trap the
field store's CRC check exists to prevent. Nothing to resume from, here or on the
box, which was destroyed.

## Trajectory

`Q` is the skyrmion number, `xlk` is Lk(φ₁,φ₂), `nseg` the per-species skeleton
segment counts. E rises while C ramps in, then descends.

```
  n      Q        E        C    xlk    nseg1  nseg2
     0  -4.942   25998    0    -5.0    2352    640
   500  -4.923   29593   25    -5.0    2352    640
  1000  -4.899   32922   50    -5.0    2352    640
  1500  -4.829   36780   75    -5.0    2352    640
  2000  -4.558   32102  100    -5.0    2352    640
  2500  -4.451   28886  125    -5.0    2352    640
  3000  -4.397   26227  150    -5.0    2352    640
  3500  -4.298   24511  175    -5.0    2352    640
  4000  -4.212   23181  200    -5.0    2352    640
  4500  -4.125   22059  225    -5.0    2352    640
  5000  -4.041   21094  250    -5.0    2352    640
  5500  -3.963   20246  275    -5.0    2352    640
```

`nlink=5` in the manifest params was derived from `tq`, not passed — the torus
branch sets `nlink = tq` because the q meridian winds encircle φ₂.

## What this changes about the next attempt

Two things, both consequences of the above rather than opinions about it:

1. **12k steps cannot answer the question even on a run that succeeds.** Reaching
   comparable depth needs ~36k, which is ~2.5 hr of compute and a `--run-timeout`
   above the current 9000 s.
2. **Moving the field is the fragile step, and it is avoidable.** The determinant can
   be computed *on the box* — `vortex_topology.knot_determinants` already returns the
   `(size, det)` pairs the catalog records — and written into the 9 KB manifest. Then
   a dead SSH proxy costs the tail of a trajectory instead of the whole deliverable.
   That is an engine change (jax-solitons), and it is the one being made before the
   next rental.

## Guards

All three layers passed and cost nothing: envelope clean (el/mag 112 vs wall 147),
gauntlet 6/6 with `ssh-key-registered` reported as SKIPPED, cap `$3.00` against a
`$2.06` worst case and `$0.4218` actually spent. `offers-available` had refused an
earlier attempt outright — the only A100_SXM4 on the market was $0.615 against a
`--max-dph` of 0.60 — so that attempt rented nothing. `--max-dph 0.65` was passed on
the command line for this run; the default is unchanged.
