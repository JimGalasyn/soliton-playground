# Bestiary

**UNSCORED — the census protocol is still DRAFT.** These are the entries the gates currently support, recorded so the gaps are visible.

**Two presets are catalogued here** and they must never be quoted against each other without reading the `preset`, `protecting charge` and `clock` fields first — the same knot appears in both with opposite verdicts. See [Reading across presets](#reading-across-presets).

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

## Preset `ehn-two-scalar` — relaxation holds

Preset `ehn-two-scalar` · model EHN gauged two-scalar (Eto–Hamada–Nitta, [arXiv:2407.11731](https://arxiv.org/abs/2407.11731), their Eqs. 5/11/12/13) · relaxer `ehn_relax.relax_iter`, interleaved gradient descent

Box **SB-1**, frozen and identical for every entry below:

    N = 192   L = 153.6   dx = 0.8   core = 2.0
    lam = 1000   kappa = 8e-4   C = 400   U = 50   eps_a = 0.05
    alpha = 1e-4   beta = 2e-3   ic = screened   agrad = wrapped   cramp = 8000

**`held` is not a census bin.** These entries were obtained by *relaxation* — gradient descent at fixed topology — so the clock is **relaxation steps, not time**, and no entry here has faced a real-time evolution. Descent cannot exhibit a dynamical instability by construction. Read "held 36k" as *stationary under the recorded integrator at 36 000 descent steps*, and nothing more.

Protecting charge, all entries: **Lk(φ₁,φ₂) lock** — the linking number of the φ₁ skeleton with the φ₂ ring, integer-exact at every entry but one.

| entry | seed | Lk | det | components (segs) | held | Q | E_total |
|---|---|---|---|---|---|---|---|
| `trefoil_t23` | T(2,3), R = 0.22 L ‡ | −3.000 | 3 | 1 (978) | 36k | −2.87 | 3341.5 |
| `cinquefoil_t25` | T(2,5), R = 22.5 | −5.000 | 5 | 1 (822) | 24k | −4.66 | 2766.7 |
| `septafoil_t27` | T(2,7), R = 25.0 | −7.000 | 7 | 1 (1148) | 24k | −6.46 | 4232.8 |
| `torus_t34_819` | T(3,4) = 8₁₉, R = 22.5 | −4.000 | 3 | 1 (952) | 24k | −3.67 | 2729.1 |
| `torus_t35_10124` | T(3,5) = 10₁₂₄, R = 22.5 | −5.000 | 1 | 1 (1024) | 24k | −4.66 | 3108.7 |
| `trefoil_t23_compact` | T(2,3), R = 22.5 | −3.000 | 3 | 1 (660) | 24k | −2.76 | 2007.1 |
| `trefoil_t23_twist1` | T(2,3) + twist 1 † | −3.000 | 3, 1 | 2 (662 + 224) | 24k | −2.71 | 2747.1 |
| `trefoil_t23_twistm1` | T(2,3) + twist −1 † | −2.993 | 3, 1 | 2 (658 + 224) | 24k | −2.81 | 2700.6 |
| `unknot_bare` | T(1,1) † | −1.000 | 1 | 1 (288) | 24k | −0.92 | 931.3 |
| `unknot_framed_twist1` | T(1,1) + twist 1 † | −1.000 | 1, 1 | 2 (290 + 224) | 24k | −0.89 | 1693.2 |

† Seed radius not recorded anywhere in the tree — these came from one-off runs whose command line did not survive. The other six are recoverable verbatim from `run_periodic_table_fleet.py` or the entry's own description.

‡ `trefoil_t23`'s R is **inferred**, not recorded: `standard_box.leg_B2` seeds R = 0.22 L = 33.792 and its 2026-07-29 N=192 arm reproduced `nseg1 = 978` — the exact segment count in this entry. Strong, but an inference.

### What these entries establish

- **T(2,3) holds below EHN's floor.** EHN report no bound knot at N_link < 4 and suspected box size. `trefoil_t23` holds at N_link = 3 in a *smaller* box (L = 153.6 against their 256), which points at the wrapped-∂a discretisation rather than box size as the enabler. Pre-registered: the three criteria were frozen before the verdict run and all three passed at 36k — E-slope −170 → −29 per 1000 and decelerating, 978 segments bit-stable from 12k, Lk/det exact.
- **The T(2,q) ladder extends past EHN.** T(2,5) and T(2,7) hold with det = 5 and det = 7, single component, Lk locked to −q exactly. `unknot_bare` holds at N_link = 1, three rungs below the floor.
- **det and Lk are independent.** `torus_t35_10124` is the discriminator: Lk = −5 at det = 1. A linking number does not fix a knot type, and an entry reporting only Lk would have mislabelled it.
- **The framing ring looks like a quantum.** `trefoil_t23_twist1` and `unknot_framed_twist1` both carry a 224-segment satellite — the *same* count on different hosts — neutral in net linking and confined. `trefoil_t23_twistm1` mirrors every topological sign, with a 1.7 % endpoint gap measured as a lattice-asymmetry systematic.
- **Two meters are needed.** The bilinear control at identical geometry keeps the link fully intact (Lk = −3.000, det = 3, the same 978 segments) while the charge sector dies (Q → −0.01, el → 27). Geometric topology can outlive the charges, so a live-soliton claim requires both the invariants and the charge ledger.

### Caveats carried from the compendium

- **Engine-static.** Every entry is stationary *only under the recorded integrator*; any other stepper needs a re-settle. This is why the engine port's acceptance gate is bit-reproducibility rather than agreement to tolerance.
- **Relaxation, not real time.** Per the bin note above. Whether the Lk lock survives real-time evolution is unmeasured — that is the question the Faddeev real-time run exists to answer.
- **The fields are not held — yet, and the gap is now counted.** Each entry records a `field.npz` sha256, but `particles/*/field.npz` is gitignored and no state survives. As of 2026-08-01 there is a content-addressed store for them (`src/soliton_playground/ehn_lab/field_store.py`), keyed by the sha256 the catalog already pins; `field_store.py status` currently reports **0/10 held, 10 MISSING** and exits 1. Its `index.json` records what is held and is created by the first `put`, so with nothing held there is no index in git yet — the missing side is legible from the tracked catalog, not from the index. Registration now banks a field at the moment it is created, so this cannot silently recur. Refilling the ten needs a 24 GB GPU: N = 192 requires a 3.16 GiB allocation an 8 GB card cannot serve (confirmed locally 2026-08-01, and already on record in `standard_box.py`).
- **A re-run restores the physics, not the bytes.** `np.savez` output is not stable across engine versions and the writer itself changed on 2026-07-30, so a regenerated state will not match its declared sha256. The store labels that case `rederived` rather than accepting it as the original, because "same physics" and "same bytes" are different claims and the catalog's sha256 asserts the second.
- **Provenance, recovered.** The recorded `git_commit` values (ee8abb3, 809740c, a8f6031, b999911, 12dd7ee, 8ca40b9) predate the extraction. They were described here as "unreachable from any repo now on this machine"; that was wrong — all six resolve in `null-worldtube-private`, checked 2026-08-01 with `git cat-file -e <sha>^{commit}`. The provenance is intact and each entry can still be traced to the commit that produced it.

### Charter exemption: `source_out_dir`

Two fields retain Standard-Model words, and both are recorded here rather than left to be discovered:

- `source_out_dir` — `out_electron_n192` (`unknot_framed_twist1`) and `out_lepton_bare_n192` (`unknot_bare`).
- `renamed_from` — `electron` and `lepton_bare` on those same two entries.

Both are **historical facts, not identification claims**: the first is a filesystem path from July 2026, kept so the artifacts stay traceable; the second records what the entry used to be called, which is precisely the information a rename would otherwise destroy. Suppressing either would make the charter's own history unauditable. This is an explicit recorded exemption to the naming rule, not something that slipped past a regex — no field asserting what the state *is* carries such a label.

### Reading across presets

The bestiary now holds the same knot twice, with opposite verdicts:

| entry | preset | protecting charge | clock | verdict |
|---|---|---|---|---|
| trefoil T(2,3) | `gpe-dimensionless` | none (knot type) | local reconnection, τ = ξ/c | **`unstable`** — unties at 40 periods |
| `trefoil_t23` | `ehn-two-scalar` | Lk = −3 lock | relaxation steps | **held** at 36k |

Not a contradiction, and the fields are what make it legible: different medium, different protecting charge, different clock — and decisively, one is a real-time decay measurement and the other is a descent. Never quote one against the other without all four.

## What no entry can claim yet

The bullets below are the `gpe-dimensionless` gaps; the `ehn-two-scalar` caveats are in that preset's own section above.

- The protocol is DRAFT and unfrozen; nothing here is scored.
- Gate 3 has no independent test — no entrant has had a charge change logged as an explicit decay event.
- The calorimeter cannot audit the drift (its own sum-rule residual is ~4× the total energy change), so "every loss accounted" holds only as: no evidence of grid loss.
- A periodic box recycles its own sound, so no radiated *budget* exists — only instantaneous sound content.
