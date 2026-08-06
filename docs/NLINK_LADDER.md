# The N_link ladder — pre-registered campaign design

**Question.** EHN report no bound knot soliton below N_link = 4 and suspect box size.
This repo holds bound knots at N_link = 3 (`trefoil_t23`) and N_link = 1
(`unknot_bare`) in a box *smaller* than theirs, so box size is the wrong suspect.
What, exactly, is the difference?

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
