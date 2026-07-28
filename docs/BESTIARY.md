# Bestiary

**UNSCORED — the census protocol is still DRAFT.** These are the entries the gates currently support, recorded so the gaps are visible.

Preset `gpe-dimensionless` · model GPE (split-step) · ξ = 1, c = 1 (c measured 1.0446/1.0714, see `tests/test_characteristic_period.py`)

Survival threshold in force: **N = 50** characteristic periods (protocol default).

## trefoil T(2,3)

| field | value |
|---|---|
| preset | `gpe-dimensionless` |
| protecting charge | `none (knot type); winding W conserved per strand` |
| **bin** | **`unstable`** |
| **clock** | **local reconnection (tau = xi / c)**, τ = 1 |
| lifetime (untying) | t = 40 = **40 periods** (vs N = 50) |
| first reconnection | t = 5 = 5 periods |
| decay channel | reconnection cascade → unknotted, unlinked rings + sound |
| seed + resolution | Milnor map, scale 8ξ; N = 128 and 256, L = 64 |
| lineage | `outputs/trefoil/cascade.hepmc3` |

### Why `unstable` and not `metastable`

The bins separate *long-lived* from *dies on its own timescale*, and the trefoil unties at 40 local-reconnection periods against a declared N = 50. It falls short on **either** clock, which is what makes the call unambiguous:

| clock | τ | untying | vs N = 50 |
|---|---|---|---|
| local reconnection (tau = xi / c) | 1 | 40 periods | short by 1.2× |
| traversal (τ = L_structure / c) | 155.5 | 0.257 periods | short by 194× |

The local clock is the physically apt one — a knot does not untie by anything traversing it, but where two strands approach within a core radius, so the process runs on ξ/c. It changes the *margin* by two orders of magnitude and not the verdict. `metastable` would require a declared N ≤ 20 for this object class.

This supersedes the `METASTABLE->RINGS` verdict in commits 6ec9cc7, bd8cf9b and 492f69a. That string is the measured *channel* (the knot does decay into rings, and that is unchanged); the *bin* is a census judgment that those runs never actually made against a clock.

### Gates

| gate | result | evidence |
|---|---|---|
| 0 seed | PASS | shell 0.99995, wrap 6.8e-06 |
| 1 survival | **FAIL** (40 < 50 periods) | reported, not gated — an entrant with an identified decay channel cannot meet a survival threshold; this is what sets the bin |
| 2 ledger | PASS | drift dt-independent over 16×; FD drift 6.83e-03 vs floor 6.83e-03; sound above half-Nyquist ≤ 0.0106 |
| 3 charge retention | n/a | knot type is not a charge in this preset; the ±1 winding per strand is retained through every reconnection |
| 4 kick | PASS | 3 seeds, ε = 0.1; channel unchanged, but 2 of 3 untie before t = 20 via a second discrete route |
| refinement | PASS | N=128→256: verdict held, loops [1, 2, 3, 2, 4] identical, drift 6.8e-03→1.7e-03 |

## ring debris (decay products of the trefoil)

Recorded as a **population**, not as individual rings. The loop count wanders (6→5→6→5 across the long run's checkpoints) and the depletion-blob count swings 1–13, so reconnection among the products never stops and no ring keeps its identity. "This ring survived N periods" is therefore not a claim the data supports; what survives is the population.

Long run to T = 1000 (`outputs/trefoil_long/`).

| field | value |
|---|---|
| protecting charge | `winding W (phase circulation quantum)` |
| loops at T | 5 — lengths [69.38, 18.25, 18.25, 15.89, 11.56] |
| source-run channel verdict | METASTABLE->RINGS (the trefoil's cascade, not a ring bin) |
| ledger drift | 5.23e-02 |
| shortest-ring τ (traversal) | 11.6 |
| observed span in own periods | ~83 (from ring formation ~t=40) |
| **bin** | **`unstable` population** — line length decays 175→133 over the run, and reconnection never stops |

Rings are traversal-clocked, not reconnection-clocked: a ring has no decay event to time, so what is being asked is whether it survives many transits of itself. Per-ring, that span is size-dependent — the small rings clear N=50 (52–83 periods) and the largest loop does not (13.8) — which is another reason the entry is the population.

**Provisional.** The ledger drift over this run is 5.2e-02, about 7.7x the T=80 value. It is genuinely at the integrator's floor (dt-independent over a 4x refinement) but it is 5% of the total energy, so the gate-2 floor comparison passes while the physics deserves less trust. See the gate-2 note on capping drift absolutely.

`protected` is NOT awarded and cannot be: in this preset the only conserved topological quantity is the per-strand winding, which survives reconnection and so protects circulation rather than any object. See the `protected` bin note in CENSUS_PROTOCOL.md.

## What no entry can claim yet

- The protocol is DRAFT and unfrozen; nothing here is scored.
- Gate 3 has no independent test — no entrant has had a charge change logged as an explicit decay event.
- The calorimeter cannot audit the drift (its own sum-rule residual is ~4× the total energy change), so "every loss accounted" holds only as: no evidence of grid loss.
- A periodic box recycles its own sound, so no radiated *budget* exists — only instantaneous sound content.
