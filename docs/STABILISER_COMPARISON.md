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

## What would make it a real test

Establish that the control collapses *before* comparing anything against it. Run
the C=0, c4=0 arm until it either loses the topology or converges, and use that
timescale to set the campaign length. Until the control fails, no arm passing
means anything.
