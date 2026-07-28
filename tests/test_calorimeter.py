"""Validation of the Nore-Abid-Brachet calorimeter (census gate 2).

Gate 2 asks that every energy loss be accounted to the radiated sector rather
than the grid, which is only meaningful if the split between bound (vortex) and
radiated (sound) energy is actually correct. These tests pin that on fields whose
answer is known analytically, because a decomposition that merely sums to the
right total can still put the energy in the wrong sector:

  - a pure longitudinal phase wave is irrotational   -> ~100% COMPRESSIBLE
  - a pure density ripple has no flow at all         -> ~100% QUANTUM PRESSURE
  - a vortex ring's flow circulates around its core  -> mostly INCOMPRESSIBLE

The middle case is the one that catches a sign or projection error: a field with
zero velocity must not report flow energy in either flow sector.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from jax_solitons.grid import BoxGrid
from soliton_playground.gpe_lab import (energy_partition, helmholtz_energies,
                                        ring_pair_seed, seed_gate, smooth)

N, L = 48, 24.0


def _grid():
    return BoxGrid(N=N, L=L, dtype=jnp.float64)


def _zaxis(grid):
    """z as a broadcastable 1D axis (varies along axis 2)."""
    return np.asarray(grid.axis())[None, None, :]


def test_uniform_bulk_has_no_energy():
    grid = _grid()
    psi = jnp.ones((N, N, N), dtype=jnp.complex128)
    p = energy_partition(grid, psi)
    for key in ("E_i", "E_c", "E_q", "E_int", "E_tot"):
        assert abs(p[key]) < 1e-18, f"{key} = {p[key]}"


def test_longitudinal_phase_wave_is_compressible():
    """psi = exp(i b sin k z) at uniform density: u = b k cos(kz) zhat, which is
    curl-free, so the flow energy must land essentially entirely in E_c."""
    grid = _grid()
    m = 3                                    # integer mode -> exactly on the grid
    kz = 2 * np.pi * m / L
    z = _zaxis(grid)
    psi = jnp.asarray(np.exp(1j * 0.05 * np.sin(kz * z))
                      * np.ones((N, N, N)), dtype=jnp.complex128)
    p = energy_partition(grid, psi)
    assert p["E_flow"] > 1e-6, "test field carries no flow; test is vacuous"
    assert p["E_c"] / p["E_flow"] > 0.999, f"E_c/E_flow = {p['E_c']/p['E_flow']}"
    assert abs(p["E_i"]) / p["E_flow"] < 1e-3


def test_density_ripple_is_quantum_pressure_not_flow():
    """A real psi has zero phase, hence zero velocity: both flow sectors must
    vanish and the kinetic energy must sit entirely in quantum pressure."""
    grid = _grid()
    kz = 2 * np.pi * 3 / L
    z = _zaxis(grid)
    n = 1.0 + 0.05 * np.cos(kz * z) * np.ones((N, N, N))
    psi = jnp.asarray(np.sqrt(n), dtype=jnp.complex128)   # real -> phi = 0
    p = energy_partition(grid, psi)
    assert p["E_q"] > 1e-9, "test field has no density structure; test is vacuous"
    assert abs(p["E_flow"]) / p["E_q"] < 1e-12, \
        f"real field reported flow energy: E_flow = {p['E_flow']}"


def test_helmholtz_projection_is_exact_on_synthetic_fields():
    """The strongest available check, because both answers are exactly 1.
    u = curl(0,0,sin kx) = (0, -k cos kx, 0) is solenoidal;
    u = grad(sin kx)     = (k cos kx, 0, 0) is irrotational.
    Both are periodic, so there is no boundary artifact to confound them."""
    grid = _grid()
    kk = 2 * np.pi * 3 / L
    x = np.asarray(grid.axis())[:, None, None] * np.ones((N, N, N))
    zero = np.zeros((N, N, N))

    Ei, Ec, Ef, _ = helmholtz_energies(grid, [zero, -kk * np.cos(kk * x), zero])
    assert Ef > 1e-6
    assert Ei / Ef == pytest.approx(1.0, abs=1e-9), f"solenoidal leaked {Ec/Ef}"

    Ei, Ec, Ef, _ = helmholtz_energies(grid, [kk * np.cos(kk * x), zero, zero])
    assert Ec / Ef == pytest.approx(1.0, abs=1e-9), f"irrotational leaked {Ei/Ef}"


def test_vortex_ring_incompressible_sector_dominates():
    """A ring's flow circulates around its core, so the solenoidal sector must be
    the larger one. NOT ~100%: unlike a straight vortex (where div u = 0 exactly,
    since grad sqrt(n) is radial while grad phi is azimuthal and lap phi = 0), a
    ring has neither property, so it carries genuine compressible energy of order
    xi/R. At R = 5 xi that is tens of percent — see the scaling test below, which
    is what distinguishes real finite-core physics from a projection bug."""
    grid = BoxGrid(N=64, L=32.0, dtype=jnp.float64)
    psi = smooth(grid, ring_pair_seed(grid, R=5.0, z0=-8.0), steps=60)
    assert seed_gate(grid, psi)[0], "test field must be wrap-clean"
    p = energy_partition(grid, psi)
    frac = p["E_i"] / p["E_flow"]
    assert frac > 0.5, f"ring only {frac:.3f} incompressible"
    assert p["E_i"] > p["E_c"], "solenoidal sector should dominate"


def test_ring_compressible_fraction_falls_as_ring_grows():
    """The physics check on the split: compressible content is a finite-core
    effect of order xi/R, so a FATTER ring must be MORE incompressible. A
    projection bug or a boundary artifact would not track ring radius."""
    grid = BoxGrid(N=128, L=48.0, dtype=jnp.float64)
    fracs = []
    for R in (5.0, 8.0, 11.0):
        psi = smooth(grid, ring_pair_seed(grid, R=R, z0=-12.0), steps=60)
        assert seed_gate(grid, psi)[0], f"R={R} seed not wrap-clean"
        p = energy_partition(grid, psi)
        fracs.append(p["E_i"] / p["E_flow"])
    assert fracs[0] < fracs[1] < fracs[2], \
        f"incompressible fraction did not grow with R: {fracs}"
    assert fracs[2] > 0.75, f"widest ring only {fracs[2]:.3f} incompressible"


@pytest.mark.parametrize("case", ["ring", "phase_wave"])
def test_sum_rule_closes(case):
    """E_q + E_flow must reproduce the spectral kinetic energy. This is the
    calorimeter's own closure, and it doubles as the check that the delta
    regularization in u is not distorting the vortex cores."""
    if case == "ring":
        grid = BoxGrid(N=64, L=32.0, dtype=jnp.float64)
        psi = smooth(grid, ring_pair_seed(grid, R=5.0, z0=-8.0), steps=60)
        # loose for the ring: sqrt(n) has a near-kink at each core, so its
        # spectral derivative carries Gibbs error at O(dx). Measured to fall
        # 4.2e-3 -> 1.0e-3 -> 2.1e-4 for dx 0.50 -> 0.33 -> 0.25, i.e. it
        # converges; this bound is the dx=0.5 value with headroom.
        tol = 8e-3
    else:
        grid = _grid()
        z = _zaxis(grid)
        psi = jnp.asarray(np.exp(1j * 0.05 * np.sin(2 * np.pi * 3 / L * z))
                          * np.ones((N, N, N)), dtype=jnp.complex128)
        tol = 1e-6                     # smooth field: no core, no Gibbs
    p = energy_partition(grid, psi)
    rel = abs(p["sum_rule_residual"]) / abs(p["E_tot"])
    assert rel < tol, (f"sum rule off by {rel:.2e} of E_tot "
                       f"(residual {p['sum_rule_residual']:.3e})")


def test_sum_rule_converges_under_refinement():
    """The residual is a discretization error, not a modelling error, so it must
    shrink as dx does. This is what licenses trusting the split at all."""
    res = []
    for N_ in (48, 72, 96):
        grid = BoxGrid(N=N_, L=32.0, dtype=jnp.float64)
        psi = smooth(grid, ring_pair_seed(grid, R=5.0, z0=-8.0), steps=60)
        p = energy_partition(grid, psi)
        res.append(abs(p["sum_rule_residual"]) / abs(p["E_tot"]))
    assert res[0] > res[1] > res[2], f"sum-rule residual not converging: {res}"


# --- energy-referenced kick (ported from the prior program's eps-kick fleet) ---
def test_energy_referenced_kick_hits_its_target():
    """eps must mean the injected energy fraction, to within the bisection
    tolerance, across a range spanning the prior fleet's sweep. This is the whole
    point of the port: an amplitude-referenced kick is not comparable between
    objects or models, and our gate-4 run's '10%' was ~1% in energy."""
    from soliton_playground.gpe_lab import kick_energy_referenced, knot_envelope
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
    from trefoil_cascade import milnor_trefoil_seed

    grid = BoxGrid(N=64, L=64.0, dtype=jnp.float64)
    psi = smooth(grid, milnor_trefoil_seed(grid, 8.0), steps=40)
    env = knot_envelope(grid, 8.0)
    for eps in (0.01, 0.1, 0.25):
        _, rep = kick_energy_referenced(grid, psi, eps=eps, envelope=env, seed=1)
        assert abs(abs(rep["dE_over_E"]) - eps) / eps < 0.05, \
            f"eps={eps} requested, {rep['dE_over_E']:+.4f} delivered"


def test_amplitude_kick_is_far_weaker_than_its_label():
    """Pins the units correction: a 10% AMPLITUDE kick is ~1% in ENERGY, so the
    committed gate-4 result must not be read as a 10% energy perturbation."""
    from soliton_playground.gpe_lab import kick_field, knot_envelope, make_energy
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
    from trefoil_cascade import milnor_trefoil_seed

    grid = BoxGrid(N=64, L=64.0, dtype=jnp.float64)
    psi = smooth(grid, milnor_trefoil_seed(grid, 8.0), steps=40)
    energy = make_energy(grid)
    k, p = energy(psi); E0 = float(k + p)
    kicked = kick_field(grid, psi, eps=0.10,
                        envelope=knot_envelope(grid, 8.0), seed=1)
    k, p = energy(kicked)
    assert abs(float(k + p) - E0) / E0 < 0.05, "a 10% amplitude kick is not 10% energy"


def test_survival_buckets_distinguish_unidentifiable_from_decayed():
    from soliton_playground.gpe_lab import survival_bucket
    assert survival_bucket(3, 3) == "survived"
    assert survival_bucket(0, 3) == "decayed"
    assert survival_bucket(None, 3) == "unidentifiable"
