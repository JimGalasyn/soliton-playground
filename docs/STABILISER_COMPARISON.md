# c4 vs C — which term holds the knot up?

**Verdict: INCONCLUSIVE, by its own control.** The comparison as configured
cannot distinguish the stabilisers, because the arm with *no stabiliser at all*
preserves the topology just as well as either of them. Recorded here so the
design fault is not rediscovered.

## The question

EHN stabilise against Derrick collapse with a Chern-Simons charge term,
`rho = C grad(a).B`. The Skyrme quartic `c4` is the other classical answer. If
both arrest collapse, they should be interchangeable at the level of "does the
knot survive"; if only one does, that is a statement about the mechanism.

## The design, and what it actually measured

Five arms, N=96, L=76.8, rings/nlink=3, R=19.6, wrapped agrad, 36000 steps.
Each ran alone under an exclusive GPU lease (`proc run --gpu`), because a
contended card produced contradictory divergence verdicts earlier in this
campaign; two arms were killed mid-flight and rerun for exactly that reason.

    arm      C     c4   Q_end   |Q|/|Q0|  lines   xlk    nseg1   E_tot
    none     0      0  -2.9644     0.988      3  -3.0      342    1143
    EHN    400      0  -2.9170     0.972      3  -3.0      360    1768
    Sky      0    100  -2.9997     1.000      3  -3.0      332    3750
    both   400    100  -3.0087     1.003      3  -3.0      314    4277
    Sky      0   1000 -49.0932    16.366     12  -0.558   1518    3934

**Four of five arms are indistinguishable on every topology meter**, including
the control. Reading a ranking off `|Q|/|Q0|` -- c4=100 "scores best" at 1.000 --
would be meaningless: the control scores 0.988 with nothing switched on.

## Why the test could not fail

There is no collapse here to arrest. `nseg1` (core length) moves only 366 -> 342
in the control over 36000 steps, and the total energy is still falling steadily
at the end (3845 -> 1899 -> 1416 -> 1143 at 9k/18k/36k, roughly 20% per
doubling), driven almost entirely by the magnetic sector shedding 2153 -> 152.

So the run is **unconverged, not stable**. It stops long before the timescale on
which a stabiliser would matter. A check that cannot fail is not a check.

## What the scan does establish

The two terms are not interchangeable in their *side effects*, which is a real
result even though the headline comparison is void:

- **C=400 substantially perturbs the configuration.** The linking energy
  collapses -118 -> -7.9 and `grad1` jumps 229 -> 851, with Q dipping to -2.29
  mid-run before recovering to -2.92. Switching the charge term on does work on
  the state, then partially relaxes back.
- **c4=100 is nearly inert.** Q pinned at -3.000, `link` steady at -118,
  everything else tracking the control. It adds energy (`sky` 7331 -> 2538)
  without changing the outcome.
- **c4=1000 destroys the knot**, shattering the tangle into 12 components with
  xlk -0.558 and nseg1 1518. There is an upper bound on the quartic well below
  the strength at which it would dominate.
- **`both` shows no interaction.** It tracks the c4=100 arm, so the two terms are
  not fighting each other at these strengths.

## What would make it a real test — REGISTERED 2026-09-02, not yet run

Registered before the rerun rather than described after it, because the fault above
was a *design* fault and a design fault is only fixed in advance. Nothing below has
been executed; when it is, the results append rather than replace.

### What is IMPORTED, and from where

Stated as a table so no reader has to reconstruct it, and so each import carries the
caveat that comes with it. Every criterion here was measured on a *different object*
than the stabiliser arms, and saying so is the point.

| quantity | value | source | status |
|---|---|---|---|
| skeleton stability | segment count stable to ±1% across the last third | `NLINK_LADDER.md:110` | ⚠ measured on the **rings/nlink** family, not on a torus knot under a stabiliser scan |
| convergence validity | E-slope decelerating over the last third | `NLINK_LADDER.md:121` | ⚠ same object caveat; used here as a VOID rule, never as a pass |
| charge gate form | `|Q| >= 0.5 × (target)` | `NLINK_LADDER.md:112-115` | calibrated against wrapped/bilinear at nlink=3; the FORM is imported, the target is this campaign's own `|Q0|` |
| refinement rule | one resolution doubling, gates re-applied | `CENSUS_PROTOCOL.md:274` | repo-wide convention |
| NaN handling | a diverged leg is VOID, not failed | commit `7c93f49` | repo-wide convention |

⚠ **These are convergence-validity and VOID criteria, not the verdict.** That is the
only reason the cross-object import is admissible: none of them can make an arm
*pass*, so a criterion that turns out to be mis-tuned for this object costs a leg its
interpretation and cannot manufacture a result. Any criterion that could produce a
PASS is registered below from this campaign's own measurements instead.

### R0 — the precondition is a MEASUREMENT, and it runs first

Run the `C=0, c4=0` control until it reaches one of three terminal states:

* **COLLAPSE** — topology lost: skeleton component count leaves 3, **or** `|xlk|`
  falls below 2.5, **or** `|Q|/|Q0| < 0.5`.
* **CONVERGED** — the last third satisfies both imported criteria above.
* **BUDGET** — `--steps` exhausted in neither state. The leg is **VOID**, and so is
  every arm beside it.

⚠ **The 36k scan above terminates in BUDGET.** `nseg1` moves 366 → 342 over the whole
run and `E_tot` is still shedding ~20% per doubling at the end, so the control is
neither collapsed nor converged — which is exactly the finding recorded above, now
expressed as a rule that fires automatically instead of as a fault someone noticed.

### R1 — the campaign length is a RULE, not a number

    steps = ceil(3 × S_collapse)

where `S_collapse` is the step count at which the **control leg of this campaign**
first meets COLLAPSE. The factor 3 is registered here; `S_collapse` is not chosen at
all, it is read off a leg in the same campaign, at the same configuration, on the same
integrator. A length picked any other way is a length picked to be long enough to look
convincing.

⚠ **If the control CONVERGES rather than collapsing, the campaign does not proceed —
it is re-registered.** A stabiliser cannot be shown to arrest a collapse that does not
happen, and reinterpreting a converged control as "stable, so the arms mean something"
is the same vacuous-gate move in a new costume.

### R2 — the convergence check is a LEG of this campaign, not an import

One arm re-run at doubled resolution (`N → 2N` at fixed `L`), inside this campaign, at
the same engine commit and the same pinned solver, fetched alongside the science legs.
Not the catalog's doubling, not an earlier local one: a convergence property measured
on another object, at another resolution, or on another integrator is not evidence
about this one.

**Equivalence margin, declared before the leg runs:**

* discrete meters (`lines`, `det`, integer `xlk` target) — **exact equality required**
* `|Q|/|Q0|` — **EQ = 0.05** absolute

⚠ **Where EQ comes from, and why that is uncomfortable.** The four indistinguishable
arms in the void scan span `|Q|/|Q0|` 0.972–1.003, a spread of 0.031, and 0.05 sits
just outside it. That is the only same-object spread in existence, and it is being
read off the very run this registration exists to replace — four arms, one resolution,
no replicate. **EQ is therefore a pre-declared margin, not a measured noise floor**,
and the doubling leg is the first thing that will actually test it. It is registered
in advance precisely so it cannot be widened afterwards to accommodate a result.

### R3 — the provenance conditions, which are also verdict conditions

Every leg carries its engine commit, lab commit, dirty flags and resolved solver
(`launch.json` + each leg's `env.json`; `soliton_playground/provenance.py`). **A leg
whose recorded solver is not the campaign pin is VOID**, not a data point — a
different integrator at the same engine commit is a different experiment, and the
stabiliser differences this campaign is looking for are smaller than a solver change.

### R4 — ABORT conditions, pre-committed

1. Control terminates in BUDGET → the whole campaign is VOID. No arm is interpreted.
2. The doubling leg disagrees beyond EQ → **the finding is the resolution**, and the
   stabiliser comparison is not interpreted at all. "Every entrant must survive a
   resolution doubling or it's cataloguing the lattice" (README) applies to scans too.
3. Any leg produces a NaN → that leg is VOID, per the repo convention.
4. An arm passes while the control also passes → the design fault above has recurred;
   report it as such rather than ranking the arms.

### What this registration deliberately does NOT buy

A collapsing control licenses one claim: **does this term arrest THIS collapse, at
THIS strength, in THIS box.** It says nothing about interchangeability in general, and
the `c4=1000` row already shows why — the quartic has an upper bound well below the
strength at which it would dominate, so any `c4` claim needs a strength scan and not a
single value. Nothing here is a claim about EHN's stabiliser; that remains a lab
proposal (commit `0c96ccf`).
