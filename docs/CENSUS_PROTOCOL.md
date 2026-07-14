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
   accounted by the calorimeter (radiated sector), not the grid.
3. **Charge retention**: topological/winding numbers (event-graph charges)
   unchanged, or their change logged as an explicit decay event.
4. **Kick test**: 10% perturbation → returns to the same basin (position/shape
   tolerance declared per object class).

## Bins

- **protected** — decay forbidden by topology or a conservation law of the preset.
- **metastable** — long-lived with an identifiable decay channel (knots, oscillons).
- **unstable** — dies on its own timescale with an identifiable channel.
- **grid-stabilized** — survives at resolution N, dies at 2N. Not physics.

**Refinement rule**: no bestiary entry without passing one resolution doubling
with gates re-applied.

## Output per entrant

object, preset, bin, lifetime (own-period units), decay channel(s), radiated
budget (calorimeter partition), lineage graph reference, seed + resolution.

## Opening sequence (cheapest decisive first)

1. Bubble vs. ring at matched energy (negative control + protection demo).
2. Dark-soliton snake decay chain (full instrument exercise on a known answer).
3. Jones–Roberts branch hunt (calibration that feels like discovery).
4. Trefoil decay cascade (crown jewel of the metastable bin).

## Anchors policy

Physical presets: one literature anchor as a unit test, named in the preset file.
Unphysical presets: exempt, labeled `unphysical: true`. Anchors are the toy's unit
tests, never results.
