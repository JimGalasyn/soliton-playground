# Stability census protocol (DRAFT — freeze before the first scored run)

## Gate −1: preflight envelope (MISSING, and it has already cost us)

Before gate 0 there should be **arithmetic walls on the parameters**, checked
without running anything, saying whether the requested configuration is inside a
regime where the entrant is known to behave. This protocol has no such gate, and
the Faddeev real-time run is what the absence costs: it went out at N=64 in an
L=18 box — half the resolution `stability_compare.py` uses for bare Faddeev in
that same box — and produced a core reorganization that cannot be told apart from
an under-resolution artifact. Two hours of GPU to learn something arithmetic would
have flagged for free.

**Status changed 2026-08-01: the chamber is HERE now, for one preset.** The
apparatus was migrated out of the deprecated `null-worldtube-private` into
`src/soliton_playground/ehn_lab/` — `chamber.preflight(cfg)` returns typed drops in
milliseconds, and `standard_box.py --envelope R=… C=…` prints the walls. So gate −1
**exists and is runnable for `ehn-two-scalar`**, and remains **MISSING for
`gpe-dimensionless` and for bare Faddeev**, which is where the census actually runs.
The numbers below do not transfer to either (see the closing note).

Two corrections to this section's original claims, both found by going and looking:

- `analysis/STANDARD_BOX_SPEC.md` was cited as living in `null-worldtube-private`,
  then corrected on 2026-08-01 to say it "does not, and never has" — **and that
  correction was itself wrong.** The spec is in that repo: commit `d266443`,
  2026-07-10, "STANDARD BOX spec v1 (Jim + C, for P riders)", on branch
  `worktree-more-cosmogenesis`, present locally and on `origin`. It is absent only
  from `main`, which is why looking at the working tree missed it. Retrieve with
  `git show d266443:analysis/STANDARD_BOX_SPEC.md` — or just read
  [`STANDARD_BOX_SPEC.md`](STANDARD_BOX_SPEC.md), restored into this repo on
  2026-08-01 so the normative document sits beside the battery that cites it
  rather than on an unmerged branch of a deprecated repo.

  So the constants are authoritative **by design, not by default**: the spec's
  measured envelope and this battery agree exactly — `2349·(14/R)²·(C/400)²` with a
  wrapped threshold of 147 (`THRESH["wrapped"]`), an R ceiling of 0.35·L
  (`G2_R_MAX_FRAC`), `α ≲ O(dx²/λ)` and `ξ_c ≳ 2·dx`. The battery transcribes its
  normative source faithfully.

  **What that settles.** The open question was whether `R_min` is the R the wall was
  calibrated on, or whether the wall expels its own calibration point. It is
  neither: at C=400 the two walls have no common feasible region. Expulsion needs
  R ≥ 55.96 (0.364·L); the g2 ceiling caps R ≤ 53.76 (0.35·L). Even at the most
  favourable R the spec allows, el/mag = 159 against a threshold of 147. The
  reported 909 corresponds to R ≈ 22.5, below the spec's seed range of
  [0.2, 0.35]·L = [30.7, 53.8]. The largest C admitting a non-empty envelope is
  **≈ 384**, so SB-1's stated `C-ramp→400` sits about 4% above feasibility. Either
  the threshold is not 147 for this configuration, or SB-1 has been running just
  outside the envelope its own spec draws.
- `chamber.py` was described as being "on more-cosmogenesis". It had already been
  ported into `null-worldtube-private` on 2026-07-29 as part of a farm-ownership
  transfer, and now lives here. Its docstring still claims it plugs into
  `jax_solitons.campaign.FarmCampaign`; that module no longer exists — the campaign
  layer became `run_farm.farm.FarmCampaign`.

Its shape is the thing to copy — walls as closed-form arithmetic with a
`--envelope` preflight mode, calibrated against archived reference runs, plus
acceptance BANDS:

    el_mag(R, C)      = 2349 * (14/R)^2 * (C/400)^2     # < 147 HOLDS, else EXPELS
    alpha_max(dx,lam) = 1e-4 * (dx/0.8)^2 * (1000/lam)  # step wall, ~dx^2
    min_separation()  = 60 + 2.62 * ln(7408/75)         # two-body seam wall
    CORE_MIN_DX = 2.0    G2_R_MAX_FRAC = 0.35

**Do not copy those NUMBERS.** They are the EHN *gauged* model: C is a
Chern-Simons coefficient, lam/kappa are the EHN potential, el_mag is an
electric/magnetic ratio. None of it transfers to bare Faddeev or to GPE, and no
envelope exists for either. What transfers is the discipline: for each preset,
derive the walls that matter (resolution per core radius, step size vs dx, object
size vs box), calibrate them against runs that are known to have held, and check
them before spending compute rather than after.

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

   **The characteristic period is the time for a wave to traverse the entire
   structure once**, τ = L/c on the entrant's own traced extent — for a closed
   vortex curve, its full arc length, since traversing a loop once means going all
   the way around (`gpe_lab.characteristic_period`; c is measured, not assumed,
   and pinned by `tests/test_characteristic_period.py`).

   **This gate is REPORTED, not gated, for any entrant with an identified decay
   channel.** A survival threshold presupposes an object whose persistence is in
   question; applying it to something that decays by construction is a category
   error, and the trefoil is the case that showed it. Its curve is 155.5 ξ, so
   τ ≈ 155.5, and it unties by t ≈ 20–40 — a quarter of one traversal, with its
   whole 80-unit production run amounting to 0.51 periods. Against the default
   N = 50 it falls short by ~194×, and no achievable run length changes that,
   because the object is gone inside one period. For such entrants, record the
   lifetime *in own-period units* and let it place the entrant between the bins.

   **Where the threshold does belong: the metastable/unstable boundary.** Those
   two bins differ precisely on "long-lived" versus "dies on its own timescale",
   so N is the number that separates them and must be declared per object class.
   Note the consequence for the trefoil, which is currently binned `metastable`:
   at 0.03–0.26 periods it dies well inside a single traversal of itself, which on
   this clock reads as `unstable`. The bin is therefore clock-dependent — a local
   reconnection time (ξ/c ≈ 1, giving ~5–40 periods) would keep it metastable —
   so **the clock must be named in the bestiary entry alongside the bin**, or the
   bin means nothing.

   Also unmet by everything so far: the ring products. Their arc lengths are
   17–23 ξ, so τ ≈ 17–23, and they have been observed from t ≈ 40 to 80, i.e.
   ~2 of their own periods. Reaching 50 would need T ≈ 900–1200, about 12–15× the
   current run.
2. **Ledger**: energy drift within the integrator's measured floor; every loss
   accounted by the calorimeter (radiated sector), not the grid.

   **This clause as written is too weak for long runs, and needs an absolute cap
   alongside the floor.** The floor grows with T — the trefoil's is 1.97e-3 at
   T=4, 6.83e-3 at T=80 and 5.24e-2 at T=1000, accumulating linearly at ~5.4e-5
   per unit time — and it is dt-independent at every one of those (drift@1000 is
   5.238e-2 at dt=0.02 against 5.232e-2 at dt=0.005, a 4x refinement). So the
   drift really is at the integrator's floor even when it reaches **5.2% of total
   energy**, and "within the floor" is satisfiable by any run of any length while
   losing an arbitrary amount of energy. Declare a cap on the drift itself (a few
   times 1e-3 is what the T≤80 runs actually achieve) as well as the floor
   comparison, and treat a long run's physics as provisional above it.

   Four further rules learned from the trefoil's gate-2 run:
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

   **Supersede this gate with an eps\* barrier, per the prior program's protocol.**
   The retired program's kick fleet (`null-worldtube-private`,
   `simulations/engine_dogfood/eps_kick_batch.py`, `stability_compare.py`) did
   this better in four ways, and its choices should be adopted:
   - **Kick the VELOCITY, not the field.** Their `bn = n0` is left untouched and
     the noise becomes the initial velocity, projected onto the constraint's
     tangent space (`w -= (w·n0) n0`). This injects kinetic energy without
     displacing the configuration or perturbing its topology, which a field kick
     inevitably does.
   - **Reference eps to ENERGY, absolutely.** They scale by
     `sqrt(eps * Epot / KE(w))`, so eps *is* the injected energy fraction, and
     `--common-eref` kicks every model with the same absolute energy so
     cross-model comparison is apples-to-apples. Our gate-4 run was ~1% in energy
     against their sweep of 0.1 / 0.25 / 0.5 / 1.0 — so its PASS is a much weaker
     statement than "a 10% kick" sounds, roughly a tenth of their smallest step.
   - **Report eps\*, the threshold where survival fails**, not a binary pass at one
     amplitude. A barrier height is a measurement; a single-eps pass is an anecdote.
   - **K ≥ 8 seeds per (object, eps)**, scored as a survival *fraction* with a
     `survived / decayed / unidentifiable` bucket — the third bucket matters,
     since a tracer failure is not a decay.

   And their recorded false result is worth the warning: the gauged integrator's
   CFL limit is ~14× tighter than the bare one, and at a bare-tuned dt the full
   model blew up to NaN — which was published internally as "the full model is
   more fragile" before being traced to the timestep. Tune dt per model and
   re-derive any comparison that crosses integrators.

## Bins

- **protected** — decay forbidden by topology or a conservation law of the preset.
  Requires `protecting_charge` ≠ `none`; see "Output per entrant".

  **This bin is EMPTY in the `gpe-dimensionless` preset for every compact
  entrant, and that is structural rather than a gap in the campaign.** GPE's only
  conserved topological quantity is the ±1 phase winding around a strand, and it
  survives every reconnection untouched — so it protects the existence of
  *circulation*, never the identity of any *object*. Reconnection is therefore
  unforbidden in any configuration whose pieces can reach each other, and the
  T=1000 run shows it never stops: 5–6 loops for a thousand time units with the
  count wandering 6→5→6→5 and the depletion-blob count swinging 1–13, total line
  length decaying 175→133, nothing ever settling. Even an isolated ring is not
  protected, since it shrinks by radiating sound without meeting anything. What is
  left for this bin in GPE is essentially the translationally invariant straight
  line, which cannot self-approach — not a zoo.

  There is also **no knotted minimizer for relaxation to find**. GPE
  imaginary-time descent does not untie a knot, it annihilates the vortices: on
  the Milnor trefoil, 8000 steps take E_tot 3020 → 265, depleted cells
  36818 → 732 and the incompressible (vortex) energy 651 → 101. The ground state
  at fixed norm is the uniform condensate, so vortices are excited states with
  nowhere to sit — GPE has nothing knotted to be stable *in*, which is the
  mechanism behind "GPE by itself always unknots".

  Populating `protected` therefore requires a preset whose charge is a genuine
  homotopy invariant, i.e. the Faddeev–Skyrme wing, where Q_H ∈ π₃(S²) = ℤ cannot
  change under continuous evolution — and where knots *are* minimizers, so
  relaxation has something to converge to. Prior program results (outside this
  repo) established Faddeev–Skyrme trefoil and cinquefoil stability under
  relaxation, and found that Hopf-linked pairs cannot be reached by descent at
  all — they must be "born linked", which is itself a direct signature of the
  invariant: continuous descent cannot cross linking classes. **Open and untested:** every Faddeev result
  the program inherited came from *relaxation* (gradient flow downhill, which by
  construction cannot exhibit a dynamical instability). Whether Q_H protects a
  knot under real-time evolution has not been measured, and until it is, no
  entrant anywhere in this census has earned `protected`.
- **metastable** — long-lived with an identifiable decay channel: lifetime ≥ N
  own-periods *and* a channel. (Oscillons; **not** the GPE vortex knots — the
  trefoil was assumed to belong here and does not. See `BESTIARY.md`.)
- **unstable** — dies on its own timescale with an identifiable channel, i.e.
  lifetime < N own-periods. The GPE trefoil lands here on either clock.
- **grid-stabilized** — survives at resolution N, dies at 2N. Not physics.

**Every entry names its clock next to its bin.** The metastable/unstable boundary
is a lifetime threshold, so the bin is only as meaningful as the clock counting it,
and the same measured lifetime can support either bin: the trefoil unties at 0.26
traversal periods but 40 local-reconnection periods. Choose the clock the *decay
mechanism* runs on — a knot unties where two strands approach within a core radius,
not by anything traversing it, so ξ/c and not L/c. A bin without a clock beside it
carries no information.

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
