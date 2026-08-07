# Testing EHN's stabiliser in a magnetic thin film

A proposal, written 2026-08-07, off the back of the N_link ladder campaign.

**Status: speculative.** The numbers below are computed, the correspondence is
exact, and the materials are real. What is NOT here is any experimental
expertise — this is a theory-side argument for why a particular measurement is
worth someone's time, not a protocol. The references are cited from memory and
**should be checked before anyone acts on this**; this session has already
demonstrated twice what an unverified claim costs.

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

1. **Charge proportional to Hopf index.** The direct signature, 1-12 e, by charge
   sensing / Kelvin probe / STM.
2. **An H^2-dependent splitting** in the resonance spectrum between sectors that
   are degenerate without the magnetoelectric term.
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

If the bound charge on a hopfion turns out NOT to scale with the Hopf index — if
it tracks, say, only the total winding of the boundary or the sample
magnetisation — then the "topological charge binding" framing is wrong and the
correspondence to EHN's rho is superficial. That is a single measurement and it
is the first one to do.
