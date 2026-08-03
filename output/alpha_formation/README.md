# alpha formation test — does step size decide whether a trefoil forms?

Run records only. The two `field.npz` (793 MB each) were **deleted**; these
manifests are the whole surviving artifact, which is why they are tracked.
Same precedent as `output/ehn_box_t25/` — a run record kept without its field.

## The question

`jax_solitons.ehn` reaches a held T(2,3) trefoil at **N_link = 3**, below the
floor Eto–Hamada–Nitta report (PRL 135, 091603). Why do we get it and they
don't? Four candidate explanations; this run tests the cheapest.

**Step size.** EHN publish α = 4e-4. Under the d³-weighted reading that is an
effective 2.05e-4, or ~82% of the functional's stability bound (2/H = 2.5e-4 at
λ=1000) — against our 1e-4 at 40%. A large step can skip a shallow basin, and
EHN's own remark that low-N_link states have *"smaller electric charges"* implies
that basin **is** shallow. Plausible, and cheap to falsify.

## The runs

Seeded fresh from n = 0, identical in every parameter but α. SB-1 B2 geometry:

    python -m jax_solitons.ehn.relax \
      --N 192 --L 153.6 --R 33.792 --geom torus --tp 2 --tq 3 \
      --ic screened --cramp 8000 --agrad wrapped \
      --C 400 --beta 2e-3 --U 50 --alpha {1e-4 | 2.05e-4} \
      --steps 12000 --samples 24 --topo-every 1 --save-every 12000

| α | Lk(φ₁,φ₂) | phi1_knot | segs | E @ 12k | wall |
|---|---|---|---|---|---|
| 1e-4 (control) | −3.0 | `[[978, 3]]` | 978 | 5076.5 | 1157 s |
| 2.05e-4 | −3.0 | `[[978, 3]]` | 978 | 3794.0 | 1233 s |

**Answer: no.** Same knot, same determinant 3, same 978-segment skeleton. The
trefoil forms at 82% of the bound as readily as at 40%, so step size is not what
separates us from EHN. Determinants were measured off the saved fields with
`particle_catalog._measure`, not inferred from the segment counts.

## Reading these numbers correctly

- **Formation, not convergence.** Neither arm is settled. 2.05× the step buys
  ~24.6k steps' worth of descent in 12k, which is the entire reason E differs —
  3794 sits between the control's 12k value and the settled 3333. The claim
  tested is that the topology *forms and holds*, not that it has converged.
- **The control is load-bearing.** It reproduces the archived `trefoil_t23` to
  0.06% on E and exactly on topology (978 segs, Lk −3.0, link −25%, el 253 vs
  the reference 254). Without that, a null result would be uninterpretable —
  two runs can agree because α doesn't matter, or because neither run works.
- **Earlier α work tested something else.** Every prior α measurement *resumed*
  a settled trefoil, so it measured survival. This is the only run that starts
  from a seed, which is the only way to ask about formation.

## What survives as the explanation

**The seed.** EHN's IC threads N_link separate φ₁ loops onto the φ₂ ring; a
trefoil is a single φ₁ curve winding p times round and q times through. Not
reachable from their initial condition at any step size. Untested but strongly
supported — it is now the surviving hypothesis rather than merely the leading one.

**The wrapped phase discretisation**, which retains linking flux where the
bilinear form drains it. Still untested. B1's own calibration predicts what a
bilinear arm should look like (`Q=-0.010 link=-7% el=27`, geometry held and
charges dead), so it is falsifiable the same way for the same cost.

**Not the box size.** EHN conjecture that N_link < 4 *"necessitates a larger
simulation box size than that we used"*. This box is 1.67× smaller on a side
than their 320³.

Recorded 2026-08-02. Analysis in jax-solitons PR #88; the α bound itself in #87.
