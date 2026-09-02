# Testing EHN's stabiliser in a magnetic thin film

A proposal, written 2026-08-07, off the back of the N_link ladder campaign.

**Status: RETRACTED IN PART, 2026-09-02.** Measurement 1 and the `H^2` scaling
argument are WRONG — a Néel hopfion carries **zero net axion charge at any Hopf
index**, by the same theorem this repo already used to void the ladder's naive arm.
See "The retraction" immediately below before reading anything else. The rest of
the document is left standing as written, because the point of keeping it is to
show what the argument looked like when it was wrong.

**Original status: speculative.** The numbers below are computed, the
correspondence is exact, and the materials are real. What is NOT here is any
experimental expertise — this is a theory-side argument for why a particular
measurement is worth someone's time, not a protocol. The references are cited from
memory and **should be checked before anyone acts on this**; this session has
already demonstrated twice what an unverified claim costs.

---

## The retraction

Raised by the null-worldtube-private session after a review with Jim; verified here
independently before being accepted, with this repo's own numerics and a positive
control. Script: `experiments/reference/axion_charge_identity.py`.

### The identity

With `div B = 0`,

    grad(theta) . B = div(theta B)

so the net bound charge is a **pure surface term**:

    Q = (alpha/4pi^2) * closed-surface-integral of theta B.n

If `theta` is single-valued and returns to a constant on the boundary, **Q = 0
exactly, for any divergence-free B** — uniform, structured, or linking. Net charge
requires a `theta` that is *not* single-valued: a compact angle winding around a
string, so `theta` jumps `2pi` across a cut surface and `Q = 2pi * (flux through
the cut)`. Q then counts the **linking of B flux with the string**.

⚠ **This repo already proved this, one document over.** `NLINK_LADDER.md:407`:
*"rho = B.grad(a) has ZERO NET under naive — identically, for uniform B and for any
divergence-free B — while carrying the LARGEST |rho| of the three modes."* This
proposal was written the same day and did not apply the theorem to itself.

### Verified here, not taken on trust

    single-valued theta = pi + 0.3 n_z, on a genuine Hopf texture
      H = 1, 2, 2, 4    B = uniform z     Q = 0.0e+00   local |rho| L1 up to 34.5
      H = 1, 2, 2, 4    B = curl A        Q < 3e-16     local |rho| L1 ~ 8-9

    POSITIVE CONTROL — theta winding once around the z axis, B linking it
      wrapped grad      Q = 27.7294  vs analytic 2*pi*flux 27.7474  (-0.06%)
      naive grad        Q = 2.6e-03                    <- the voided arm
      B ALONG the string rather than linking it        Q = 0.0e+00

The control is the load-bearing part: `Q = 0` from a broken integrator is
indistinguishable from `Q = 0` from physics, so the winding arm has to return the
analytic answer before the zeros mean anything. It does, to 0.06%.

### Why no `theta` slaved to a Néel texture can escape this

The peer's argument is the linearised dynamical-axion response: `theta` is the
k-space Chern-Simons integral, `delta_theta = delta_m5 / g` with `m5` proportional
to the Néel component, so reversing the Néel vector takes `theta` from `pi + d` to
`pi - d` — not around a circle. A `2pi` winding would need the band mass to pass
through zero (the TI/NI transition), which a spin rotation does not do. That
argument is sound and is **not** independently checked here; it rests on
[Li, Wang, Qi, Zhang arXiv:0908.1537] and [arXiv:1906.07891] as cited.

**The topology says something stronger, and it does not depend on the
linearisation.** Homotopy classes of continuous maps into a circle are
`[X, S^1] = H^1(X; Z)`, and `H^1(S^2; Z) = 0`. So **every** continuous
`theta(n)` defined on the target sphere is null-homotopic and therefore lifts to a
single-valued real function — no matter how nonlinear the slaving is. Any `theta`
that is a function of the *local Néel direction alone* is single-valued on `R^3`
whenever the texture is, and Q = 0 follows. The failure is not that this particular
response happens not to wind; it is that no pointwise slaving to a unit-vector
texture *can* wind.

The escape is therefore not a better material or a stronger response. It is a
different order parameter: one with a genuinely circle-valued component, i.e. a
**phase-vortex line threaded by flux**, which is EHN's geometry — two distinct
strings, one carrying a compact phase. A magnetic hopfion is a single-field `pi_3`
texture with no compact angle and no string.

### What falls, and what does not

| claim | status |
|---|---|
| Measurement 1, "charge proportional to Hopf index" | **RETRACTED.** Net charge is zero at any H. |
| "charge ~ H so Coulomb ~ H^2" | **RETRACTED**, it rests on measurement 1. |
| "the same equation with a different prefactor" | **True of the local density, false of the integral.** EHN's floor depends on a LOCKED INTEGRAL charge, and that is exactly the part that does not carry over. |
| Measurement 2, index-dependent resonance splitting | **NOT killed.** A neutral multipole still has Coulomb self-energy — the same distinction as the 2026-08-07 correction in `NLINK_LADDER.md`: `sum(rho) = 0` does not make `sum(rho*A0) = 0`. But it has lost its `H^2` scaling and therefore its motivation, and at 1e-3..1e-5 it now needs a separate argument for why it is resolvable against other H-dependent energies. |
| Measurement 3, the sign | Survives only as far as measurement 2 does. |
| The energy-scale arithmetic (`alpha/4pi^2` ~ 2e6 weaker than C = 400) | Unaffected. |
| The ladder and stabiliser results | Untouched. Nothing else in the repo depends on this document. |

⚠ **The proposal's own falsifier was already decided before it was written.** The
closing section below says the premise fails "if the bound charge on a hopfion turns
out NOT to scale with the Hopf index" and calls that "a single measurement and it is
the first one to do." The identity settles it with no measurement at all. A
falsifier that an existing theorem already answers is not a test, and this is the
second time in this campaign that a check turned out to be decidable in advance —
the first was the `c4`-vs-`C` scan whose control could not fail
(`STABILISER_COMPARISON.md`).

---

## Why bother

The campaign (NLINK_LADDER.md) asked whether the `∂a` discretisation reproduces
EHN's N_link floor. It does not. What survived is narrower and more interesting:
their knot is held open by **charge binding** — the phi1 loops acquire electric
charge from the Chern-Simons coupling, and the field's own repulsion resists
collapse.

That stabiliser is the load-bearing part of the whole construction, and it has
been demonstrated in exactly one place: numerically, in their code and in ours.
Everything downstream — the knot-dominated era, the baryogenesis, the GW signal —
rests on it. So it is worth asking whether the ingredient can be checked anywhere
a person can stand next to.

## The correspondence is exact, not analogical

    EHN:                    rho = C * grad(a) . B          C = 400
    axion electrodynamics:  rho = (alpha/4pi^2) * grad(theta) . B

Magnetic topological insulators host a **dynamical axion**: theta is slaved to the
antiferromagnetic Neel vector, so a Neel texture IS a theta texture [Li, Wang, Qi,
Zhang, Nat. Phys. 2010 — verify]. In a magnetic field that texture binds charge by
literally EHN's formula. Their `C a F F~` is the term condensed matter calls the
magnetoelectric response.

This is not "similar physics". It is the same equation with a different prefactor.

⚠ **True of the local density and false of the integral** — see "The retraction".
EHN's stabiliser is a LOCKED INTEGRAL charge counting the linking of flux with a
phase string. A Néel texture supplies the local `grad(theta).B` and supplies no
string, so its integral is zero. Sharing a formula for `rho` is not sharing the
quantity the floor is made of.

## Where the field already is

Hopfions in magnetic multilayers were created and observed in 2021 [Kent et al.,
Nat. Commun. — verify], stabilised by **DMI + perpendicular anisotropy + film
confinement**. Thin films are the right form factor: confinement is part of what
holds them open.

So the topology is solved. pi_3 knots exist on a bench today. What is missing from
the lab version is the charge.

## The charge is real and measurable

Bound charge on a hopfion of size L, from a theta texture of order pi:

    system                              100 nm hopfion
    magnetic TI, B = 1 T                     1.2 e
    magnetic TI, B = 10 T                   12.1 e
    type-II multiferroic, P ~ 1e-4 C/m^2     6.2 e

One to twelve electrons is comfortably within single-charge sensing. The
multiferroic route is stronger because the polarisation comes from spin-orbit (the
Katsura-Nagaosa-Balatsky spin-current mechanism, P ~ e_ij x (S_i x S_j)) rather
than from an alpha-suppressed response.

## It will not stabilise anything, and that is fine

    hopfion 100 nm:  E_exchange     ~ 6.2 eV
                     E_coulomb(1e)  ~ 14 meV (vacuum), 0.29 meV (eps_r = 50)
                     ratio            2.3e-03 to 4.6e-05

EHN's `C = 400` is what makes their electric term competitive with the string
tension. `alpha/4pi^2 ~ 1.8e-4` is **~2e6 times weaker**. The lab hopfion stays
DMI-stabilised and the charge is a perturbation at the 1e-3 to 1e-5 level.

**This does not sink the experiment, because EHN's floor never required charge to
be the dominant energy.** It required charge to be an N_link-DEPENDENT energy. If
bound charge scales with the Hopf index H, the Coulomb self-energy scales as H^2 —
an index-dependent term, which is exactly the shape the floor needs and exactly
what the campaign could not find in the discretisation.

A 1e-3 perturbation that SPLITS otherwise-degenerate Hopf sectors is measurable,
because magnetic textures have sharp resonance spectra (FMR, Brillouin light
scattering) where meV splittings are routine.

## The proposal

**An antiferromagnetic topological insulator hosting a hopfion in the Neel vector
itself.** MnBi2Te4 is the candidate material.

This is the elegant version because theta is slaved to the Neel texture: knot the
Neel vector and the theta texture is knotted automatically, so the charge binding
is intrinsic. No heterostructure, no interface, no second material to align. It is
a van der Waals material, exfoliable, and a ~100 nm film is ~70 septuple layers —
thick enough for a 3D hopfion while still being a film.

Measure, in order of decreasing confidence:

1. ~~**Charge proportional to Hopf index.** The direct signature, 1-12 e, by charge
   sensing / Kelvin probe / STM.~~ **RETRACTED — see "The retraction". The net
   charge is zero at every Hopf index.**
2. **An H^2-dependent splitting** in the resonance spectrum between sectors that
   are degenerate without the magnetoelectric term. ⚠ The `H^2` is retracted with
   item 1; a splitting from a neutral multipole is not excluded, but nothing
   predicts its scaling.
3. **Its sign** — whether charge binding stabilises or destabilises higher index.
   This is the one that speaks to EHN's floor, because their whole mechanism is
   that charge binding STABILISES, and more of it stabilises more.

## What it would and would not show

**Would.** That topological charge binding is real, index-dependent, and of
measurable size — the ingredient EHN's floor is built from, in a system you can
put on a stage and perturb.

**Would not.** That charge binding can hold a soliton open against Derrick
collapse. That needs C ~ 400 and no condensed-matter system supplies it. Nor
anything about N_link = 4 specifically: the lab hopfion's stability hierarchy is
DMI's, not the charge's.

So it tests the INGREDIENT, not the RESULT. Given that this campaign found the
ingredient does not produce the result at the parameters tested, checking the
ingredient directly is arguably the more informative move.

## Known problems, stated up front

- **Getting a hopfion into the Neel texture of an AFM.** Antiferromagnetic
  textures are hard to create, image and manipulate; the 2021 hopfions were in
  ferromagnetic multilayers. This is the main experimental risk and it is not
  small.
- **Thickness versus surface physics.** MnBi2Te4's axion response is a bulk
  property, but thickness changes the surface gap, and 70 layers is far from the
  few-layer regime most of its physics has been done in.
- **The multiferroic route trades cleanliness for signal.** Stronger charge, but
  the texture and the polarisation are coupled through a messier mechanism and
  the material class is less well characterised for hosting hopfions at all.
- **eps_r matters a lot.** TIs have large dielectric constants, which is exactly
  the wrong direction: it screens the effect being measured by ~50x.

## What would falsify the premise

⚠ **ANSWERED, 2026-09-02, with no measurement — see "The retraction" at the top.**
The paragraph below stands as written so the record shows what was proposed.

If the bound charge on a hopfion turns out NOT to scale with the Hopf index — if
it tracks, say, only the total winding of the boundary or the sample
magnetisation — then the "topological charge binding" framing is wrong and the
correspondence to EHN's rho is superficial. That is a single measurement and it
is the first one to do.

The bound charge does not scale with the Hopf index. It is zero at every Hopf
index. **The correspondence to EHN's rho is superficial** in exactly the way this
paragraph named — the local densities share a form, and the integral charge EHN's
floor is built on does not carry over at all.
