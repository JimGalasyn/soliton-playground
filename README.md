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
- **Renderer** — `soliton_playground.viz`: frame-by-frame 3D isosurfaces and
  animated GIFs, for the real-time legs where the answer is a movie rather than a
  number. CPU-only by design (numpy / matplotlib / scikit-image / scipy / Pillow,
  **no jax**), so rendering a finished run never contends for VRAM with one that
  is still going.

  ```bash
  # animate a run's saved snapshots (frames land beside the GIF, for re-cutting)
  python -m soliton_playground.viz timelapse --fields outputs/<run>/fields \
      --out outputs/<run>/anim.gif --L 51.2 --c4 4.0 --spin 4
  # or sweep the camera around one static field
  python -m soliton_playground.viz turntable --field <field>.npz --out tt.gif
  ```

  Three `--mode`s: `surface` (smoothed isosurface), `facets` (unsmoothed — the
  grid facets are left in, the upstream's "no cheat" stance), and `cells` (one
  cube per grid cell above the level, so the render admits its own resolution).
- **Gauged portraits** — `soliton_playground.viz_em`: the same machinery for the
  `ehn-two-scalar` catalog states, which carry two complex scalars plus a gauge
  and scalar potential. Views: `raw`, `cells`, `twist` (core swept as a
  rotation-minimising tube painted by arg φ₁, so a colour spiral is real framing
  twist), `efield`, `bfield`, `triptych`, and `cycle`. This is also the home of the
  `load_field` that `ehn_lab/field_store.py`'s comments used to point at across a
  repo boundary. None of it applies to the bare-Faddeev real-time runs — they
  have no gauge sector.

  ```bash
  python -m soliton_playground.viz_em --field <dir-with-field.npz> --view triptych
  # the charge phase travelling around the tube, with the fields travelling too
  python -m soliton_playground.viz_em --field <dir> --view cycle \
      --cycle-fields travelling --frames 36 --m 3
  ```

  `cycle` animates Φ₁→e^{−iθ}Φ₁: arg(φ₁) winds spatially along the loop, so
  sweeping θ slides the colour around the tube. `--cycle-fields static` adds the
  field's own E and B (held, since they carry no phase); `travelling` adds a
  modelled m-harmonic source whose fields travel with the phase, affordable
  because Maxwell linearity lets the cos/sin components be solved once and
  recombined per frame.

  Read `viz.add_parts` before touching the drawing code. matplotlib's 3D backend
  is a painter's-algorithm renderer with no z-buffer, so it draws whole artists
  back to front: **one Poly3DCollection per tube means a link never weaves**, and
  no amount of reordering helps. Merging every triangle into a single collection
  is the only construction that crosses over and under correctly, which is why
  the module passes geometry around as `(faces, colors)` parts. That is pinned in
  `tests/test_viz_depth.py` on a Hopf link — separate collections give the near
  tube 0.6% of the contested pixels, merged gives 45%/55%.

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
git clone https://github.com/JimGalasyn/run-farm     ../run-farm
pip install -e ../jax-solitons
pip install -e ../run-farm          # NOT optional -- see below
pip install -e .
```

**`run-farm` must be installed from the sibling checkout, not left to resolve.**
Omitting that line does not fail: `jax-solitons` declares `run-farm>=0.1.1` as a plain
specifier (PyPI forbids direct references in published metadata), so pip quietly
installs the **published** run-farm from PyPI and everything appears fine — until
`experiments/run_ehn_box_vast.py` dies at leg construction with

```
TypeError: FleetLeg.__init__() got an unexpected keyword argument 'reattachable'
```

**The version number cannot tell you why.** PyPI's run-farm and this workspace's
checkout both report `0.2.0`; the checkout is ten commits past the `v0.2.0` tag with the
version string unchanged, and `reattachable` is in those ten. So `pip list` shows the
expected version while the driver is broken. Check the install PATH, not the version:

```bash
pip list | grep run-farm     # want a /home/.../run-farm path, not a bare version
```

The same applies to `jax-solitons`: the rented box `pip install`s both repos from
GitHub `main`, so the engine that runs is main's, while `ENGINE_COMMIT` in the manifest
is resolved from the *local* checkout. A local clone that is merely behind records a
commit that is not the code that ran.

Found on 2026-08-03 by an intake that had no venv here and borrowed a sibling
interpreter plus `PYTHONPATH=src` to run the driver at all — the setup above is what
makes a clean checkout reproducible. (`pyproject.toml`'s `pythonpath = ["src"]` already
covers `pytest`, which is why the suite passed while the driver did not.)
