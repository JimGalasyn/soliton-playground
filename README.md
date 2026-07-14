# soliton-playground

**A sandbox for topological structures in simulated media. Play, not science.**

This repo drives [jax-solitons](https://github.com/JimGalasyn/jax-solitons) as a
toybox: what's stable, how it scatters, how it decays, and how all of that changes
across environments — dilute-BEC-like, superfluid-helium-cartoon, and
unphysical-but-cool. The structures here (vortex knots, rings, solitons, oscillons,
Q-balls, droplets, bubbles) are **quasiparticles of media we define**, full stop.

## The one rule

> No number produced here is ever compared to a measured physical constant, and no
> structure is ever identified with a Standard Model particle.

Everything upstream of that line — cross-sections, lifetimes, decay channels,
helicity budgets, bestiary tables — is unconditionally in-bounds, because it's all
internal to a medium we specify. If a result ever seems to want an external claim,
the comparison target is the BEC / quantum-turbulence literature, and the claim
process starts from zero, outside this repo.

*Provenance note: this sandbox plays with the surviving instruments of a retired
physics program ([retrospective](https://doi.org/10.5281/zenodo.21339662)). The
theory died; the engine, the calorimeter, and the event graph were worth keeping.*

## Instruments

- **Event graph / calorimeter / ECS** — `jax_solitons.event_graph`: one causal
  graph; per-vertex charge closure is the calorimeter, the committed trace is the
  lineage record, HepMC3 output for event-display tooling.
- **Models** — `jax_solitons.models`: GPE (split-step; laboratory-media presets)
  and NLKG (verlet; relativistic/cosmological presets), plus Faddeev sectors for
  the hopfion wing.
- **Invariants & tracking** — `jax_solitons.invariants` / `topology`: curve
  tracing, linking numbers, Hopf charge.

## Program

1. **Stability census** (`docs/CENSUS_PROTOCOL.md`): run the zoo through declared
   gates → bestiary bins (protected / metastable / unstable / grid-stabilized).
   Every entrant must survive a resolution doubling or it's cataloguing the
   lattice.
2. **Scattering & decay**: knot–knot collisions over impact parameter, phase, and
   velocity; unknotting cascades with helicity accounting; outcome taxonomies with
   the calorimeter watching the phonon budget.
3. **Environments**: a preset = {kinetic operator, interaction kernel,
   damping/pumping, external potential, components} **plus its own declared energy
   functional** (the calorimeter's per-universe ledger). Physical presets carry one
   literature anchor as a unit test (helium-cartoon must show a roton minimum);
   unphysical presets are exempt but labeled.

## Setup

```bash
git clone https://github.com/JimGalasyn/jax-solitons ../jax-solitons
pip install -e ../jax-solitons
pip install -e .
```
