# EHN-box N=320, 36k steps — the cinquefoil HOLDS inside the expulsion wall

The run the previous two attempts were trying to be. A T(2,5) at EHN's own resolution
(320³, L=256, dx=0.8) seeded at R = 64 = 0.25 L, relaxed 36000 steps — past the
33k–36k window where the *local* cinquefoil at R/L = 0.146 reconnected.

**Result: `det(φ₁) = [(2350, 5)]` at n = 36000. The self-knot is intact.**

## The question this was built to separate

`experiments/run_ehn_box_vast.py`'s docstring poses two readings of that local
reconnection (φ₁ determinant 5 → 1 between 33k and 36k while Lk held at −5.0):

- **(a)** the self-knot is not topologically protected. EHN's conserved charge is
  N_link — the strings cannot pass through *each other* — and nothing in that argument
  stops one string passing through *itself*. Then it unknots at any box size.
- **(b)** the knot was simply too tight at R/L = 0.146. Inside the expulsion wall the
  strands are not packed, and the determinant holds.

At matched descent depth, same `alpha`, same `dx`, same λ/κ/C/U, the tighter seed
reconnected and this looser one did not. **That supports (b).**

## What it does NOT establish

- **Not a converged minimum.** E is still descending at n = 36000 — 7732.4 → 7690.4 →
  7650.5 over the last two samples, ~40 per 500 steps. "Holds at 36k" is the claim;
  reconnection at 50k or 100k is not excluded.
- **Not a determinant trajectory.** The on-box determinant failed on all 73 samples
  (below), so nothing here can *date* a reconnection — only report the endpoint. Had one
  occurred and reversed, this record could not tell.
- **E sits ~9% above EHN's reference.** 7650.5 against the ≈7.0e3 the engine prints for
  nlink = 5, and still falling. Consistent with an unconverged descent; not reconciled.
- **One box, one seed.** No R-scan, no repeat.

## The rental

| | |
|---|---|
| launched with | `--steps 36000 --samples 72 --run-timeout 13000 --cap-usd 4.50 --max-dph 0.65` (all resolved flags in `launch.json`) |
| host | A100 SXM4, Alberta CA, reliability 0.9986, **$0.5485/hr** |
| instance | 46743588, provisioned in 116.9 s |
| billed | 9870.4 s = **$1.5039** (worst case was $2.78; cap $4.50), destroyed, `verify: gone` |
| on-box relax | 9069 s wall for 36000 steps + 73 diagnostic samples |
| `field.npz` | 3,670,017,098 B, **every zip member CRC-verified** — not inferred from the size |

72 samples were used rather than the default 24 deliberately: at 36000 steps, 24 samples
is one every 1500 steps, *coarser* than the 1200-step local run whose reconnection showed
no energy signature. 72 keeps the 500-step resolution the measurement is for.

Everything the previous attempts broke on worked: `remote_exit=0`, the DONE marker
carried it, teardown verified, `launch.json` written before renting, `run.log` fetched,
and the 3.67 GB field arrived whole. The reattach path was never exercised — no transport
failure occurred.

## THE DEFECT: the on-box determinant computed nothing

Every sample recorded `det1 = [[2352, 'e:ImportError']]` (later `[[2350, ...]]`). 73 of
73. The rental's headline measurement did not run once, and the leg still reported **OK**
with `remote_exit=0`.

Root cause: **`pyknotid` is an optional dependency** of jax-solitons — the `knots` extra,
deliberately optional because tracing curves needs only numpy/scipy while the Alexander
determinant needs pyknotid. The driver's remote command installs neither extra, so
`identify_knot` could not import on the box. `vortex_topology.knot_determinants` catches
per-line exceptions and records `f"e:{ExceptionName}"` in place of that line's integer —
so an *environment* failure, identical for every line, was written into the manifest in
the shape of a per-curve result.

It passed local validation because this machine's venv has pyknotid (via the `test`
extra). Nine leg-script scenarios and two new tests all exercised the measurement in an
environment that differed from the target in exactly the dependency that mattered.

## How the answer was recovered anyway

The field came back intact, and pyknotid is present locally, so the determinant was
computed here from the fetched `field.npz`:

```
knot_determinants(φ₁, dx=0.8, L=256) -> [(2350, 5)]     14 s
```

Which is the answer — but it is worth being clear that **the on-box determinant did not
remove the dependency on that 3.67 GB transfer this time, it only appeared to.** Had the
field torn as it did on 08-03, this rental would have bought nothing.

## Trajectory (every 2500 steps; full 73 samples in the manifest)

`xlk` = Lk(φ₁,φ₂), constant at **−5.0** for all 73 samples. `nseg1` moved 2352 → 2350
once, at n = 22500 — a two-segment change in a 2350-segment skeleton, not a reconnection
signature.

```
     n        E       C     xlk   nseg1   det1 (on box)
     0   25998.4     0    -5.0    2352   e:ImportError
  1500   36779.8    75    -5.0    2352   e:ImportError   <- peak, C ramping in
  5000   21093.8   250    -5.0    2352   e:ImportError
 10000   14454.8   400    -5.0    2352   e:ImportError
 15000   11336.5   400    -5.0    2352   e:ImportError
 20000    9823.3   400    -5.0    2352   e:ImportError
 25000    8879.7   400    -5.0    2350   e:ImportError
 30000    8221.6   400    -5.0    2350   e:ImportError
 33000    7912.4   400    -5.0    2350   e:ImportError   <- local box reconnected in
 36000    7650.5   400    -5.0    2350   e:ImportError      the 33k-36k window
```

Final link% = −67.5 against a floor of 197.4. `e_finite: true`, so the NaN gate added in
jax-solitons `6ffb93a` correctly did not fire — this is a real field, not a diverged one.

## Files

`DONE` is committed, and it contains `exit=0`. That makes the fleet's resume gate
pre-skip a leg re-run with this same `--out` and label — which is *reported* by the
`ResumeMarkersIntended` gauntlet check ("no leg pre-skips" / "N leg(s) will be SKIPPED"),
not silent, so it is a legible consequence rather than a trap. Use a fresh `--out` for
the next attempt regardless. `PID` is committed too: it carries the boot_id of a box that
no longer exists, which is exactly the case the re-enterable leg script's boot-id guard
is built to ignore.

`field.npz` (3.67 GB) is gitignored and stays out.
