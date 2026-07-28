# Stability census protocol (DRAFT — freeze before the first scored run)

## Gates (declared before running, applied to every candidate)

0. **Seed gate** (before evolving): boundary-shell density ≈ vacuum
   (1 − n < 0.02) on every axis the entrant is *localized* in, and wrap phase
   mismatch < 0.05 rad, checked only where the boundary planes carry bulk
   density (`gpe_lab.seed_gate(axes=...)`). Added after the opener's run 1,
   where a non-periodic ring phase laid a boundary sheet that contaminated
   the box and inflated the ledger drift 60× (jax-solitons#67); the axes
   parameter added after the gate misfired on planar solitons, which
   legitimately cross the transverse boundaries and carry branch-cut phase
   noise in their cores. A dirty seed invalidates the run before physics
   starts; an extended entrant declares its localization axes up front.
1. **Survival**: lifetime ≥ N characteristic periods of the object itself
   (N declared per campaign; default 50).
2. **Ledger**: energy drift within the integrator's measured floor; every loss
   accounted by the calorimeter (radiated sector), not the grid. Four rules
   learned from the trefoil's gate-2 run:
   - **"Measured floor" means a dt sweep, and its content is dt-INDEPENDENCE.**
     If refining dt does not reduce the drift, the drift is spatial truncation
     and the integrator is already at its floor. Merely comparing a drift to a
     floor measured from the same run at the same dt is circular; the sweep is
     what makes it a measurement. The trefoil's drift moved 0.1% across a 16x dt
     refinement, so it is entirely spatial.
   - **Never compare drifts across energy conventions.** `GPEKineticTerm` uses
     forward differences; the calorimeter is spectral. They differ by O(dx) and
     drift ~10x differently (6.8e-3 vs 6.7e-4 for the same run). A first version
     of the trefoil's gate compared a spectral drift against a forward-difference
     floor and "passed" on the convention gap. Match the convention *and* the
     time interval.
   - **A sector closure that sums to the total by construction is a tautology.**
     Since E_i + E_c + E_q + E_int is E_tot identically, the closure residual is
     just the drift re-expressed. The clause that actually distinguishes
     "radiated sector" from "the grid" is whether the sound sits at RESOLVED
     wavenumbers (`energy_partition`'s `E_c_highk_frac`) — energy accumulating
     near k_max is headed into truncation, not into the medium.
   - **Know the calorimeter's resolution before claiming a loss is accounted.**
     Its own sum-rule residual was 7.98, about 4x the 2.03 of total energy change
     it would have had to attribute. It resolves sector transfers (O(100-1000))
     and cannot audit the drift (O(1)). "Every loss accounted" is then only the
     weaker claim: no evidence of grid loss, and the loss is below what the
     instrument can attribute. Say which one you mean.

   **A periodic box recycles its own sound.** E_c is instantaneous sound content,
   NOT cumulative radiation: phonons cannot leave, so E_i and E_c oscillate in
   antiphase as sound is reabsorbed. A radiated *budget* needs an absorbing far
   field, which this engine does not yet have.
3. **Charge retention**: topological/winding numbers (event-graph charges)
   unchanged, or their change logged as an explicit decay event.
4. **Kick test**: 10% perturbation → returns to the same basin (position/shape
   tolerance declared per object class). **For a `metastable` entrant, "the same
   basin" means the same decay channel** — such an entrant has no basin to return
   to, since it decays on its own regardless of the kick, so the gate asks
   whether the channel and its topological signature survive perturbation.
   Pass/fail criteria and the observables that are only *reported* must both be
   declared before running; do not invent a threshold for a quantity nobody has
   measured yet. Two rules learned from the trefoil's gate-4 run:
   - **Gate 0 outranks gate 4.** The kick must be windowed to the entrant, not
     applied to the box: an unwindowed 10% kick left the boundary shell at
     1 − n = 0.180 against gate 0's 0.02, invalidating the run before physics.
     `gpe_lab.kick_field(envelope=...)`.
   - **One realization proves nothing about a basin.** Kick with an ensemble
     (≥3 seeds). The trefoil's ensemble found the cascade has *two* discrete
     intermediate routes, which a single kick would have reported as "matches"
     or "differs" with equal confidence and no way to tell which.

   Note that "10%" is amplitude, not energy: a 10% windowed field kick moved the
   trefoil's energy by only ~1% (and *lowered* it for one seed, legitimate since
   an imaginary-time-smoothed seed is not an exact minimizer). Declare which
   measure is meant.

## Bins

- **protected** — decay forbidden by topology or a conservation law of the preset.
- **metastable** — long-lived with an identifiable decay channel (knots, oscillons).
- **unstable** — dies on its own timescale with an identifiable channel.
- **grid-stabilized** — survives at resolution N, dies at 2N. Not physics.

**Refinement rule**: no bestiary entry without passing one resolution doubling
with gates re-applied.

## Output per entrant

object, preset, **protecting charge**, bin, lifetime (own-period units), decay
channel(s), radiated budget (calorimeter partition), lineage graph reference,
seed + resolution.

**Protecting charge is mandatory, and `none` is a real answer.** It names the
conserved quantity that forbids this entrant's decay *in this preset*, and it
is what makes the `protected` bin earnable: an entrant recording `none` can
never bin as protected no matter how long it lives — only `metastable`.

The rule exists because geometry alone does not identify an entrant. The GPE
trefoil unties (`METASTABLE`) while the Faddeev T(2,3) hopfion is pinned by
Hopf charge Q_H = 2 (`protected`) — same knot, same name, different sector,
because knot type is not a charge of the GPE but Q_H is an integer homotopy
invariant. Without `preset` + `protecting_charge` in the record, those two
bestiary rows read as a contradiction. Emitted by `gpe_lab.provenance()` into
every summary and by `gpe_lab.zoo_provenance()` into the event-graph attrs, so
a lineage file read back without its summary still says which medium it came
from.

## Opening sequence (cheapest decisive first)

1. Bubble vs. ring at matched energy (negative control + protection demo).
2. Dark-soliton snake decay chain (full instrument exercise on a known answer).
3. Jones–Roberts branch hunt (calibration that feels like discovery).
4. Trefoil decay cascade (crown jewel of the metastable bin).

## Anchors policy

Physical presets: one literature anchor as a unit test, named in the preset file.
Unphysical presets: exempt, labeled `unphysical: true`. Anchors are the toy's unit
tests, never results.
