# The N_link ladder — pre-registered campaign design

**Question.** EHN report no bound knot soliton below N_link = 4 and suspect box size.
This repo holds bound knots at N_link = 3 (`trefoil_t23`) and N_link = 1
(`unknot_bare`) in a box *smaller* than theirs, so box size is the wrong suspect.
What, exactly, is the difference?

> **CORRECTION, 2026-08-07 (Amendment 2, end of file).** That first sentence
> overstates them. EHN write that they "did not find any other knot solitons with
> our numerical code", which "would imply" instability below N_link = 4, and that
> "more detailed studies are necessary to conclude." That is a hedged
> non-detection, not a reported floor -- and a non-detection does not contradict a
> detection. The campaign below was designed against the stronger reading. Read it
> knowing the premise is softer than it states.

**Hypothesis.** The ∂a discretisation, and nothing else.

    bilinear  d_i a = Im(phi2* d_i phi2)/(|phi2|^2 + eps_a)      modulus-SUPPRESSED
    wrapped   d_i a = angle(phi2(x+i) conj(phi2(x-i)))/2dx       modulus-BLIND

Under `bilinear` the field has an escape hatch: it can drain the L3 charge density
by rearranging |phi2| — thinning the modulus where rho lives — without ever
unwinding the phase. Under `wrapped`, curl grad a is an exact integer string delta,
so integral rho is quasi-algebraically locked to N_link and the hatch is shut. EHN's
knot is bound *by* the L3/Chern-Simons structure (the `C eps^ijk d_i a d_j A_k`
term), so a configuration that has shed its charge is not their soliton even when
the geometric knot is still sitting there.

**Why we think EHN used bilinear.** Inference, not quotation — worth stating
plainly. Their supplemental says "naive spatial discretization ... second-order
central-difference scheme" and lists the lattice variables as
`{Re phi_1,2, Im phi_1,2, A_i, B_i}`. Their descent needs `dE_disc/du` in those
variables, which forces a smooth expression in Re/Im — i.e. the bilinear form. A
wrapped `angle()` difference is not that. **They never write d_i a's discretisation
explicitly**, so this campaign tests a mechanism, not a quoted method.

A third candidate exists and is unimplemented here: per-site `a = arctan2(Im, Re)`
then a *naive unwrapped* central difference. That is the most literal reading of
their sentence and is worse than bilinear — it breaks at every branch cut. If the
bilinear arm does NOT reproduce their floor, this is the next thing to try.

## The single measurement that already exists

`trefoil_t23`, N_link = 3, identical geometry, 36k steps:

| arm | Lk | det | segments | Q | elec |
| --- | --- | --- | --- | --- | --- |
| wrapped  | -3.000 | 3 | 978 | **-2.868** | **156.2** |
| bilinear | -3.000 | 3 | 978 | **-0.01**  | **27** |

The geometry is untouched in both. Only the charge dies. This campaign asks whether
that death is N_link-dependent in the way EHN's floor requires.

## Design

Two arms x five rungs = 10 legs, everything else frozen.

    agrad   in {wrapped, bilinear}
    nlink   in {1, 2, 3, 4, 5}
    geom    rings (build_ic: one big phi2 ring, nlink small phi1 rings pierced by it)

Frozen for every leg (SB-1, and the parameters the catalog was built with):

    N = 192   L = 153.6   dx = 0.8   core = 2.0   R = 38.4  (R/L = 0.25)
    lam = 1000   kappa = 8e-4   C = 400   U = 50   eps_a = 0.05
    alpha = 1e-4   beta = 2e-3   q1 = 1   q2 = 0
    ic = screened   cramp = 8000   steps = 36000

`steps = 36000` is not a guess: it is the horizon the catalog's pre-registration
used, so the wrapped arm doubles as a **reproduction test**. wrapped/nlink=3 must
return Lk = -3.000, det 3, 978 segments, Q ~ -2.87, elec ~ 156. wrapped/nlink=5 must
reproduce `cinquefoil_t25` (det 5) and wrapped/nlink=1 `unknot_bare`. If those three
do not reproduce, the campaign is broken and no result from it counts.

## The confound, and why the ladder survives it

**The expulsion wall is arm-specific: THRESH = {wrapped: 147, bilinear: 37}.** At
L = 153.6 and C = 400 there is NO radius satisfying both that wall and the g2
ceiling — wrapped needs R >= 56.0, bilinear needs R >= 111.5, ceiling is 53.8. Every
leg here runs outside its own wall, and the bilinear arm runs *further* outside.
This is not new: the entire existing catalog was built ~6x outside the wall and
holds regardless, which is itself evidence the wall over-predicts expulsion.

A skeptic's objection is therefore available and must be answered in advance: *the
bilinear arm fails because it is further outside its envelope, not because of the
discretisation.*

**The answer is the shape of the result, not the magnitude.** `el_mag(R, C)` depends
on R and C only — **it has no N_link dependence at all**. So the wall cannot produce
an N_link-dependent floor. If bilinear binds at nlink = 4, 5 and fails at 1, 2, 3 at
one fixed geometry, no wall argument explains that pattern; only something that
scales with the winding can. This is why the ladder must include the rungs where EHN
say knots DO bind — **nlink = 4 and 5 are the positive control and are not optional.**

Corollary: a bilinear arm that fails at EVERY rung including 4 and 5 does NOT support
the hypothesis. It would mean the geometry is simply inadmissible for that arm, and
the campaign would have to move to a box where bilinear is inside its wall
(R >= 111.5 needs L >= 319, i.e. N = 400 at dx = 0.8 — expensive, and stage 2 below
is the cheaper probe).

## Pre-registered criteria

Frozen before any leg runs. Two meters, because geometric topology can outlive the
charges (BESTIARY, "Two meters are needed").

**Geometric.** PASS requires all three:
  - `det(phi1)` equals the value the wrapped arm returns at that rung (self-paired,
    so this does not assume which knot type each rung produces)
  - single dominant phi1 component
  - skeleton segment count stable to +-1% across the last third (24k -> 36k)

**Charge.** PASS requires:
  - `|Q| >= 0.5 * nlink`  — primary. Calibrated on the existing control: wrapped
    gives Q = -2.87 at nlink = 3 (i.e. Q ~ -nlink), bilinear gives -0.01. The
    threshold sits an order of magnitude clear of both.
  - `elec` is REPORTED BUT NOT GATED. Its scaling with N_link is unmeasured, and
    inventing a threshold from the one trefoil point would be fitting the gate to
    the datum it is meant to test.

**Convergence validity** (per leg, else the leg is void rather than failing):
  - E-slope decelerating over the last third, as in the catalog's pre-registration.

**A leg is a "bound knot soliton" only if BOTH meters pass.** That is the definition
under which EHN's floor is a floor.

## Predicted outcome, stated before the run

    nlink        1      2      3      4      5
    wrapped     PASS   PASS   PASS   PASS   PASS      (1, 3, 5 are reproductions)
    bilinear    fail   fail   fail   PASS   PASS      <- EHN's floor, reproduced

Falsifiers, each of which kills or redirects the hypothesis:
  - wrapped fails at 1 or 3 -> campaign broken (contradicts the catalog)
  - bilinear passes everywhere -> discretisation is not the mechanism
  - bilinear fails everywhere -> geometry inadmissible for that arm, see corollary
  - the floor lands somewhere other than 4 -> mechanism real, calibration different

## Cost

Measured basis: 0.118 s/step at N = 192 on a 4090 (`form-1e4`, 10k steps in 19:41).

**Stage 1 — the ladder, N = 192.** 36000 steps = 1.18 h per leg, 10 legs =
**11.8 gpu-h**. At $0.35/hr that is **~$4.15** plus acquisition tax (budget one
failed host at the ready-timeout, as `run_ehn_box_vast` already estimates). Ships
nothing: every leg builds its own IC on-box from `build_ic`, so there is no 792 MB
field to upload. Embarrassingly parallel.

**Stage 2 — confirmation at EHN's own box, only if stage 1 lands.** N = 320,
L = 256, R = 64 (R/L = 0.25, el/mag = 112, inside the wrapped wall). Volume scaling
gives ~0.55 s/step, so 36000 steps = 5.5 h per leg. Run only the two rungs that
bracket the floor, both arms: nlink in {3, 4} x 2 arms = 4 legs = **22 gpu-h**,
**~$7.70**. This is the leg that makes the claim directly comparable to their paper,
because it is their grid, their box, their spacing.

Total if both stages run: **~$12** and ~34 gpu-h.

## What a positive result would say

Not "we found a knot and they did not". Rather: *their floor is reproducible from
their own stated method, and it moves when one discretisation choice changes* — the
compact-angle derivative that makes integral rho an integer invariant instead of a
modulus-weighted quantity the field can drain. That is a statement about lattice
field theory, testable by anyone, and it predicts that any future implementation
using a smooth phase-velocity derivative will rediscover the same floor.

---

# Amendment 1 — 2026-08-06: the reproduction gate was untestable as written

**This section is appended, not merged into the text above.** The original
pre-registration stays verbatim so the change is auditable. Everything not named
here remains frozen exactly as pre-registered.

## What was found

The first wrapped arm ran 2026-08-05 (five legs, 36k steps, all complete) and
scored **CAMPAIGN IS BROKEN**. The cause is not physics. It is that the
reproduction gate compares two different initial conditions.

`build_ic(geom="rings", nlink=k)` seeds **k separate small φ₁ rings** threaded on
the big φ₂ ring — a k-component link of unknots. That is EHN's IC and is the right
seed for this campaign. The catalog states the gate checks against are
**single-component torus knots** built by `build_ic_torus`:

| | `unknot_bare` | `trefoil_t23` | `cinquefoil_t25` | this ladder |
| --- | --- | --- | --- | --- |
| seed        | —        | torus T(2,3) | torus T(2,5) | rings, nlink loops |
| `phi1_knot` | [[288,1]] | [[978,3]]   | [[822,5]]    | nlink x [[~220,1]] |
| components  | 1        | 1            | 1            | nlink |
| R           | —        | 33.792       | 22.5         | 38.4 (frozen) |
| steps       | 24000    | 36000        | 24000        | 36000 |

Generating commands, recovered independently of the manifests:

    trefoil_t23     output/alpha_formation/README.md:24
                    --N 192 --L 153.6 --R 33.792 --geom torus --tp 2 --tq 3
                    (that README's control reproduces the archived entry to 0.06%
                    on E and exactly on topology: 978 segs, Lk -3.0)
    cinquefoil_t25  experiments/reference/run_periodic_table_fleet.py:77
                    --geom torus --tp 2 --tq 5 --R 22.5 --steps 24000

**Why the error was available to make.** The catalog landed 2026-08-01, one day
before seed geometry was recorded in manifests (jax-solitons cdcd263, 2026-08-02).
So `trefoil_t23/manifest.json` carries `nlink: 3` and no `geom` — and the torus
branch sets `nlink = tq`, so that `3` is a torus winding, not a rings count.
`relax.py:_seed_params` documents this exact trap ("the held T(2,3) trefoil at
N_link=3 exists only on the torus branch"), having hit it on 2026-08-02. This doc
was written 2026-08-05 and read the `3` as a rings count.

Consequences, both structural rather than empirical:

  - `check_repro` compares `det[0][1]`. Rings at nlink=3 gives det 1 (an unknot);
    the trefoil is det 3. **It cannot pass at nlink 3 or 5 on any physics.**
  - `score_geometry` requires `ncomp == 1`. Rings gives `ncomp == nlink`, so
    **every rung >= 2 fails by construction, in both arms.** Note this is not
    fixed by switching to torus: T(2,q) has gcd(2,q) components, so q = 2 and 4
    would fail the same test. No seed family satisfies `ncomp == 1` across
    rungs 1-5.

Also incorrect above: "steps = 36000 ... is the horizon the catalog's
pre-registration used". True for `trefoil_t23` only; the other two targets were
measured at 24000.

## What changes

**1. The geometric meter's component criterion becomes per-rung.** PASS requires
`ncomp == nlink` for a rings seed (the seed's own construction), replacing
`ncomp == 1`. The determinant and segment-stability criteria are unchanged, and
the determinant stays self-paired against the wrapped arm at the same rung.

**2. The reproduction gate moves off the science legs and onto its own leg.**
What the gate is for is unchanged and still load-bearing: certify that THIS box,
THIS engine commit and THESE parameters produce known-good physics before any
bilinear number is believed. That job needs the catalog's own seed, so it gets a
dedicated leg run at the catalog's own parameters:

    repro_t23   --geom torus --tp 2 --tq 3 --R 33.792 --steps 36000 --agrad wrapped
    PASS iff    phi1_knot == [[978, 3]] and Lk == -3.0 and Q within 10% of -2.868

One leg, ~1.18 h, ~$0.42. It is a pipeline check, not a rung of the ladder, and
its result gates the campaign exactly as the old wrapped/{1,3,5} check was meant
to. If it fails, no result from this campaign counts.

**3. The 2026-08-05 wrapped arm becomes the rings baseline.** It is the first
rings-seeded measurement at this geometry, so nothing held could have adjudicated
it. It is recorded as reference, and future rings campaigns reproduce IT:

    nlink        1        2        3        4        5
    Lk          -1.000   -2.000   -3.000   -4.000   -5.000   exact at every rung
    ncomp        1        2        3        4        5
    det          1        1        1        1        1        (unknot components)
    Q           -0.927   -1.879   -2.734   -3.785   -4.473
    elec         107.6    227.1    365.9    517.0    704.7
    nseg1        202      390      676      780      1122

**Deliberately NOT changed.** The charge meter (`|Q| >= 0.5*nlink`), the arms, the
rungs, R = 38.4, steps = 36000, the confound argument, and the predicted outcome
all stand as pre-registered. The charge meter passed at every rung on 2026-08-05
(Q ~ -0.9*nlink against a 0.5*nlink threshold) and is not touched here — amending
a meter that already fired would be fitting the gate to the datum.

## Status of the 2026-08-05 wrapped arm

Not void. Its legs converged, `Lk` was exactly integral at every rung, and the
charge meter passed at every rung. It is void only as a *reproduction* of the
torus catalog, which it was never capable of being. Under the criteria above it
is scored as the rings baseline, and the bilinear arm remains unrun and unjudged.

---

# Result — 2026-08-06: stage 1 complete, the hypothesis is NOT supported

Campaign closed. All eleven legs ran at SB-1 (N=192, L=153.6, R=38.4, C=400,
36k steps), scored under the criteria above as modified by Amendment 1.

## Pipeline check

    repro_t23 vs trefoil_t23 (torus T(2,3), R = 33.792)
      det  [[978, 3]]  vs  [[978, 3]]     exact
      Lk   -3.0        vs  -3.0           exact
      Q    -2.867      vs  -2.868         0.03%, against a +-10% gate
    -> CERTIFIED

The box, engine commit f50853ff and these parameters reproduce a held state, so
the arms below rest on a certified pipeline.

## Outcome vs prediction

    nlink            1      2      3      4      5
    wrapped  pred   PASS   PASS   PASS   PASS   PASS
             obs    PASS   PASS   PASS   PASS   PASS     <- as predicted
    bilinear pred   fail   fail   fail   PASS   PASS
             obs    fail   fail   fail   fail   fail     <- NOT as predicted

The wrapped arm landed exactly as pre-registered. The bilinear arm did not: it
fails at EVERY rung, including 4 and 5, which are the positive control.

This fires a pre-registered falsifier, verbatim from the list above:

> bilinear fails everywhere -> geometry inadmissible for that arm, see corollary

and the corollary's instruction stands: this does NOT support the hypothesis, and
nothing here should be read as a discretisation-driven floor.

## The measurements

    rung   Q wrapped   Q bilinear   elec w   elec b   |Q|/n w   |Q|/n b   b/w elec
      1      -0.927      -0.430      107.6      7.0     0.927     0.430      0.065
      2      -1.879      -0.833      227.1     14.5     0.939     0.416      0.064
      3      -2.734      -1.006      365.9     22.6     0.911     0.335      0.062
      4      -3.785      -1.593      517.0     31.6     0.946     0.398      0.061
      5      -4.473      -1.642      704.7     41.8     0.895     0.328      0.059

## What this does and does not establish

**The discretisation is a large effect on charge.** Bilinear holds electric charge
at a near-constant ~6% of wrapped, and |Q|/nlink drops from ~0.92 to ~0.38. The
mechanism the hypothesis proposed — that bilinear lets the field drain L3 charge
by rearranging |phi2| — is visible and substantial. That much reproduces.

**It is not an N_link-dependent effect, and that is what the hypothesis needed.**
The b/w electric ratio drifts only 0.065 -> 0.059 across a FIVEFOLD change in
N_link, and |Q|/nlink is flat in both arms. An N_link-independent suppression
cannot produce a floor at N_link = 4 at any magnitude. The mechanism is real; the
floor it was invoked to explain is not in this data.

**The confound is NOT resolved, and this data cannot resolve it.** The bilinear arm
ran ~8x outside its own expulsion wall (el/mag = 312 against a threshold of 37) and
required --force-envelope. The argument above says the wall cannot produce an
N_link-DEPENDENT floor, which is true and is why the design was sound — but the
observed result is uniform failure, and a wall predicts uniform failure just as
well as a genuinely inadmissible discretisation does. Both hypotheses predict a
flat ratio. Stage 1 therefore cannot separate them, and no claim that distinguishes
them should be made from it.

**Convergence caveat.** seg_drift on the bilinear legs is 0.6%-5.7% against the 1%
stability criterion, so three of five fail the geometric meter on skeleton
instability rather than on topology. ncomp tracked nlink exactly at all five rungs
and det was 1 throughout: the geometry held, the skeletons had not settled. Read
the bilinear geometric column as "not converged", not as "topology lost".

## Where this goes next

Both were named before the run, and the order is unchanged by it:

1. **The third discretisation.** Per-site `a = arctan2(Im, Re)` then a naive
   unwrapped central difference — the most literal reading of EHN's sentence, and
   the pre-registered next candidate if bilinear did not reproduce the floor. Not
   yet implemented in the engine.
2. **Stage 2, N = 320 / L = 256**, which puts bilinear INSIDE its wall and is the
   only way to retire the confound above. ~22 gpu-h, ~$7.70.

Stage 2 was pre-registered as "only if stage 1 lands". Stage 1 did not land, so
running it now is a change of purpose: it would no longer be confirmation of a
positive result, but the disambiguation this result requires. Worth doing for that
reason, and worth saying plainly that the reason is different.

## Cost, honestly

Pre-registered cap: **$12**. Actually spent on this ledger: **$18.58**.

The overrun is not experiment cost. The ladder's own legs came to roughly $2.30
(wrapped ~$1.20, bilinear $1.08, pipeline check $0.23). **$16.29 of the $18.58 was
a single incident**: on 2026-08-05 the driver was SIGHUP'd at session end, five
finished boxes were left running with nobody to tear them down, and they idled ~10
h before the next morning caught them. Fixed in run-farm (SIGHUP added to the
signal-safe teardown; reap now closes the ledger rows a dead driver never wrote)
and in this campaign's driver (it reaps its own ledger in a `finally`). Recorded
here because a budget line that says $18.58 with no explanation invites the
conclusion that the physics was expensive. It was not.

---

# Result — 2026-08-06: the third discretisation is VOID, not failed

The pre-registered next candidate after bilinear ("per-site `a = arctan2(Im, Re)`
then a naive unwrapped central difference") was implemented (jax-solitons #95) and
run on the same rungs, same geometry, same certified pipeline. Five legs, $0.88.

    rung    Q         elec    ncomp  seg_drift   geo    chg
      1    -0.9988    84.6      1      0.033     fail   PASS
      2    -1.9967   154.6      2      0.033     fail   PASS
      3    -2.9934   264.8      3      0.036     fail   PASS
      4    -3.9982   299.6      4      0.055     fail   PASS
      5    -4.9820   420.8      5      0.044     fail   PASS

At first reading this is the best arm in the campaign: |Q| = N_link to 0.1-0.4%,
10-30x tighter than `wrapped`, the mode built to BE the lock. It is not. It is an
arm in which the term under test does nothing.

## Why: the mode carries no winding

Max circulation of grad(a) around closed periodic loops, in units of 2*pi,
measured on this campaign's own IC:

    wrapped    1.000000     exactly one winding
    bilinear   0.010731     leaky, but nonzero -- it does carry some
    naive      0.000000     none, at all

Exactly zero, and not by accident. `a = arctan2(Im, Re)` is SINGLE-VALUED, so
summing its central difference around a closed periodic loop telescopes to 0
identically. The branch-cut sheet is not noise on top of an otherwise good
gradient -- it is precisely the term that cancels the smooth winding. On the real
IC the sheet is 194 of 110592 cells (0.18%); delete it and naive's sum goes
0.0000 -> 18.0956 against wrapped's 20.1062.

So rho = B.grad(a) has ZERO NET under naive -- identically, for uniform B and for
any divergence-free B -- while carrying the LARGEST |rho| of the three modes.

**What that establishes, narrowly: rho cannot represent a winding-derived charge
under naive.** EHN's floor is a statement about integral-rho being LOCKED to
N_link, so an arm in which integral-rho is identically 0 whatever the field does
cannot exhibit the mechanism this campaign exists to test. That is what voids it.

CORRECTION (2026-08-07, from review of jax-solitons #96). An earlier version of
this section said "the L3 coupling contributes nothing, nothing acts on the
topology" and called the integer Q "an absence of force". That does not hold. The
engine forms `eelec = 0.5*C*sum(rho*s)` with a spatially varying A0, and
sum(rho) = 0 does NOT imply sum(rho*s) = 0. Measured on this IC with a smooth A0
and B = curl A, naive's eelec is -152 against wrapped's -45965 -- small, but not
zero; with a RANDOM A0 it is the LARGEST of the three. The magnitude depends
entirely on how rho's cancelling spikes correlate with A0.

So why the naive arm's Q sits at its seed value to 0.1-0.4% is NOT established
here. `Q` is `skyrmion_number(phi1, phi2)` and never touches agrad; the honest
position is that integral-rho vanishing is proven and the persistence of Q is
unexplained.

The n = 0 trajectories are CONSISTENT with this, though not independent evidence
of it -- `mag` below is the MAGNETIC energy, and B is sourced from rho by the
screened solve, so a zero rho gives a zero B by construction rather than by a
separate mechanism:

    arm        link       mag
    wrapped   -118.44   4185.3
    bilinear  -102.87   1288.0
    naive        0.00      0.0

and the L3 linking energy stays ~0 for the whole run (final: +0.09, -1.33, -2.58
at rungs 1, 3, 5 -- positive at rung 1).

Pinned as a regression test in jax-solitons (`test_ehn_axion_grad.py`), so no
future arm can quietly ship a discretisation that carries no winding.

## Status: VOID

Not "fails at every rung". Its meters carry no information about EHN's floor,
because the mechanism the campaign exists to test was absent from the run. The
third candidate is retired on the ground that it is not a valid discretisation of
a COMPACT angle at all -- which also answers the question NLINK_LADDER.md posed
when it named this candidate: the most literal reading of EHN's sentence cannot be
what they implemented, because under it integral-rho is identically 0 and so cannot
be LOCKED to N_link, which is what their floor is a claim about.

(That sentence previously read "it would have given them no L3 coupling either".
Wrong -- see the correction above: sum(rho)=0 does not bound sum(rho*s), which is
the energy the engine actually forms.)

**No envelope probe was run, deliberately.** THRESH is the el/mag ratio at which
L3 charge is expelled; under naive there is no L3 charge, so there is nothing for
a wall to expel and a measured threshold would describe an artefact. (`el_probe_R.py`
is also cited in this repo only as provenance -- it is not present to run.) This
is the same refusal as declining to let the arm borrow wrapped's threshold, one
step further out.

## Where stage 1 leaves the campaign

    wrapped    BOUND at all five rungs
    bilinear   BOUND at none -- charge dies; wall confound UNRESOLVED
    naive      VOID -- carries no winding

No arm shows an N_link-dependent floor. Stage 2 (N = 320, L = 256, ~$7.70) is now
the only route that can settle anything, because it is the only one that removes
the bilinear confound -- and per the note above, running it is disambiguation, not
the confirmation it was pre-registered as.

Ledger after stage 1: $19.46.

---

# Result — 2026-08-07: stage 2 is VOID, and why

Run at N=320, L=256, R=64 on A100 80GB. Pipeline re-CERTIFIED on that hardware
(det [[978,3]], Lk -3.0, Q -2.867 vs the archived -2.868) -- a real check, since
this was a new provider, a new GPU class and a different engine commit than the
one stage 1 certified.

    leg              agrad     e_finite  last_n   Q
    bilinear_nlink3  bilinear  True       36000   -1.1614
    bilinear_nlink4  bilinear  True       36000   -1.5697
    wrapped_nlink3   wrapped   False       1000   nan
    wrapped_nlink4   wrapped   False       1000   nan

**The wrapped arm is VOID, not failed.** It hit a degenerate initial condition, not
an instability: at N=320 a lattice site lies EXACTLY on the phi2 seed ring, so
`tanh(d/core)` is exactly 0 there and |phi2| = 0 at a grid point, leaving
`a = arg phi2` undefined precisely where the wrapped axion gradient reads it. Fixed
and guarded in jax-solitons #97; that geometry is now refused at IC build.

Whether it happens is decided by whether R/dx is an integer in BINARY floating
point -- L=38.4 and L=38.400000000000006 are the same box to 15 digits and have
opposite fates. Stage 1's N=192 has no such coincidence, which is why it ran clean;
that was luck, not design.

**Without a wrapped arm there is no reference**, since the geometric meter is
self-paired against wrapped's determinant. So stage 2 answers nothing about the
floor and must not be read as a null result.

## What the bilinear legs DO establish

They ran clean and are real, because the same coincidence is harmless to bilinear
-- it computes Im(conj(phi2) d phi2)/(|phi2|^2 + eps_a), where the zero is
regularised. Identical geometry, opposite outcomes by arm, which is also the
cleanest evidence for the mechanism above.

    rung   |Q|/nlink at N=192    |Q|/nlink at N=320
      3          0.335                 0.387
      4          0.398                 0.392

**bilinear is grid-independent across a 4.6x volume change.** That argues against
its charge suppression being a resolution artefact, and it is the one physics
result stage 2 delivered.

## Still true, and still not settled

Stage 2 does NOT put bilinear inside its expulsion wall: el/mag(R=64, C=400) = 112
against a bilinear threshold of 37, so it runs ~3x outside (down from 8.4x at stage
1, not inside). Only R >= 111.5 -- L >= 319, N=400 -- would clear it. The confound
stage 1 could not resolve is still unresolved.

## To redo

Re-run the wrapped arm with R nudged off the lattice (R += dx/2). ~2 A100 legs,
~$3.50. The guard means the same failure can no longer happen silently.

---

# Amendment 2 — 2026-08-07: what the EHN paper actually says

Read from the source (PRL 135, 091603 and its supplemental) rather than inferred.
Three things here change how this campaign should be described, and one of them
means the framing at the top of this document overstates their claim.

## 1. Their claim is a hedged NON-DETECTION, not a floor

This document opens with "EHN report no bound knot soliton below N_link = 4".
What they write is:

> "We did **not find any** other knot solitons with our numerical code, which
> **would imply** that the knot soliton with the linking number smaller than 4
> cannot be stable due to the smaller electric charges and that it necessitates a
> larger simulation box size than that we used to show the knot solitons with the
> linking number larger than 5. **However, more detailed studies are necessary to
> conclude.**"

That is: their code did not find one, that *would imply* instability, and more
work is needed. It is not a demonstrated floor.

**A non-detection and a detection are compatible**, especially when the
non-detecting party says so explicitly. So "what, exactly, is the difference?" may
have no answer because there may be no contradiction. Our catalog holding bound
knots at N_link = 1 and 3 does not contradict a search that did not find them --
it is the more detailed study they call for.

The rest of this campaign is still worth what it cost, but it was aimed at
reconciling a disagreement that is weaker than we stated.

## 2. They use BOTH IC types, and their headline solutions are RINGS

Long-open question, answered from their figures:

> Fig. 1: "the single phi2 string loop is linked with **five phi1 string loops**"
>         -> N_link = 5, RINGS
> Fig. 2 leftmost: "a single loop of the phi2 string linking with **four phi1
>         string loops**" -> N_link = 4, RINGS
> Fig. 2 others: "single phi1 and phi2 loops make a higher linking number by
>         **linking multiple times**" -> TORUS

So both constructions appear, and the two solutions they report as lowest-energy
at their linking numbers -- Fig. 1 at N_link=5 and Fig. 2 leftmost at N_link=4 --
are the RINGS type. That is exactly what `build_ic(geom="rings")` seeds.

**The ladder was comparing like with like.** The torus/rings mismatch was real
INTERNALLY (Amendment 1, our own reproduction gate) but it is not the difference
with EHN.

## 3. Their alpha only works under the d3-weighted reading, and now that is forced

They state their parameters: **lambda/g^2 = 10^3, kappa/g^2 = 0.0008, C = 400** --
identical to SB-1. That closes the one free variable in this file's step-size note.

Eq. (12) of the supplemental descends `dE_disc/du`, and `E ~ d^3 sum E_disc`. Read
literally (descend the unweighted sum), their alpha = 4e-4 is 1.6x this
functional's stability bound at lambda = 1000. Measured, not argued:

    lambda   alpha/alpha_max   outcome (N=48, R=10.0 off-lattice, 1000 steps)
      400          0.64        ok,  E = 2174.2
      500          0.80        ok,  E = 2177.7
      625          1.00        ok,  E = 2173.4          <- exactly marginal
      800          1.28        "ok" but E = 9,855,835   <- blowing up
     1000          1.60        DIVERGED at n = 50

alpha = 4e-4 sits exactly on the bound at lambda = 625; their own lambda is 1000.
Under the d3-weighted reading the effective step is 4e-4 * 0.8^3 = 2.05e-4, which
is 0.82x the bound and stable.

Their published solutions exist, so **the d3-weighted reading is the correct one**
-- not by preference but because the alternative is excluded by divergence. The
step-size note in `relax.py` recorded this as "an open discrepancy, not a resolved
one" because one dx cannot separate 0.512 from 0.5. It is now resolved from the
other side: with lambda pinned at their stated value, the unweighted reading does
not run.

Nothing to change in our engine, which descends the unweighted density at
alpha = 1e-4 -- 0.4x the bound. The correction is to how their alpha is read.

## 4. Energy benchmarks, in v/g

From their captions, and directly comparable to a run at their box:

    N_link   type    energy
      4      rings   6.0e3      (Fig. 2, leftmost)
      4      torus   6.3e3      (Fig. 2)
      5      rings   7.0e3      (Fig. 1)
      5      torus   7.3e3      (Fig. 2)
      5      torus   7.5e3      (Fig. 2)

Within a linking number the rings solution is the lower-energy one, and they note
energy grows with N_link because the electric charge on the phi1 string grows.

---

# Result — 2026-08-07: stage 2 rerun off-lattice. Both rungs BOUND.

The wrapped arm re-run at N=320, L=256 with **R = 64.4** instead of 64 -- one half
cell, moving the phi2 ring off the lattice. Everything else identical: same box,
spacing, lambda/kappa/C, steps, agrad, engine commit.

    leg              rc  det ncomp  nseg       Q      elec   geo   chg   BOUND
    wrapped_nlink3    0    1     3  1164  -2.4996   882.8  PASS  PASS   YES
    wrapped_nlink4    0    1     4  1330  -3.6082  1225.9  PASS  PASS   YES

    Lk(phi1,phi2) = -3.000 and -4.000 exactly.  Pipeline CERTIFIED (4th time).
    E = 4532.1 and 5739.5.  Wall 7400 s and 7114 s.  Cost $4.59.

**The lattice-coincidence diagnosis is confirmed in production.** At R = 64 both
legs NaNed by step 1000; at R = 64.4 both run 36000 steps to a bound state. The
only change is whether a lattice site lands exactly on the seed ring.

## The comparison against EHN, at their own parameters

    wrapped_nlink4:  E = 5739.5   vs EHN's 6.0e3   ->  ratio 0.957

**Within 4.3%**, at their grid, their box, their spacing, their lambda/kappa/C, and
their rings construction (Amendment 2: their Fig. 1 and Fig. 2-leftmost solutions
are rings, which is what build_ic(geom="rings") seeds). Stage 1 at our smaller box
came in ~25% LOW against the same benchmarks, so moving to their box closed most of
the gap -- the direction that says the earlier discrepancy was the box, which is
what EHN themselves suspected.

**And N_link = 3 is BOUND at their parameters**, E = 4532.1, both meters passing,
Lk exactly -3.000 -- where they report no solution. Together with the separate
finding that our catalog states satisfy EHN's own convergence criterion Eq. (14) by
2-3 orders of magnitude, the "we used a laxer standard" objection is closed from
both directions: their bar, their box.

Read this as the "more detailed study" their own text calls for, NOT as a
contradiction. Their claim is a hedged non-detection (Amendment 2 section 1); a
detection does not contradict it.

## Deviations to record

- **R/L = 0.2516, not the frozen 0.25.** The half-cell nudge is a deliberate
  departure from a pre-registered parameter, forced by the IC bug. It is 0.6% in R
  and ~1.2% in el/mag against a 3x margin, and each leg is scored on its own
  meters rather than against the other arm, so it does not affect the verdicts.
  It does mean this wrapped arm is not at byte-identical geometry to the stage-2
  bilinear legs (R = 64), which is why the two are not presented as a paired
  comparison.
- **wrapped is INSIDE its expulsion wall here** -- el_mag(64.4, 400) = 111 against
  a threshold of 147. First time in this campaign a science arm needed no
  `--force-envelope`. bilinear is still ~3x outside at 112 vs 37; only N = 400
  would clear it.
- |Q|/nlink is 0.83 and 0.90 here against 0.91 and 0.95 at N = 192, so the wrapped
  arm sheds slightly more charge at the larger box. Both remain well clear of the
  0.5*nlink gate.
