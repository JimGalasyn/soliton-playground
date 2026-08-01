"""Gauged-isorotation Faddeev-Skyrme model (option 4).

Gauge the U(1) that rotates the FUNDAMENTAL, S^2-constrained unit field n about
its vacuum axis e3 -- rather than slaving n to a charged Higgs doublet. Because n
stays fundamental and hard-constrained to S^2, its Hopf charge remains a protected
topological invariant: a smooth S^2-valued map cannot change Hopf class without a
singularity, and the Maxwell flux of A is a SEPARATE Z that the charge cannot leak
into. This is the structural fix for the N=96 falsification -- gauged_faddeev makes
n = psi^dag.sigma.psi/|psi|^2 DERIVED from a doublet carrying its own pi_3, which
hands the gauge sector a handle to rewind Q_H (the fullcp1 [-6.94,+6.92] scatter at
psi_min=1.000). Here the leak channel is closed by construction.

State layout: (6,N,N,N) = n[0:3] (unit field) + A[3:6] (real abelian gauge
potential, temporal gauge A_0=0). Gauge group U(1) ~ SO(2) acts as rotation of
(n1,n2) about e3; A is its connection. The covariant derivative is realized as a
rotation LINK (D_i n = (R(e A_i dx) n(x+i) - n(x))/dx), so it is exactly gauge-
covariant on the lattice: under n -> R(chi) n, A_i -> A_i - d_i chi / e, one has
D_i n -> R(chi) D_i n, and |D n|^2, |D_i n x D_j n|^2, F_ij are all invariant.

At A=0 the energy is the bare Faddeev-Skyrme functional (E2 + the area-form E4 in
its covariant-cross-product form, which equals the n.(d_i n x d_j n) area form on
the |n|=1 surface since D_i n perp n). So this reduces to `faddeev_model` in the
continuum and adds a genuine, gauge-invariant Maxwell sector -- nothing slaved.
"""
import dataclasses

import jax.numpy as jnp

from jax_solitons.grid import BoxGrid
from jax_solitons.model import Model
from jax_solitons.models.faddeev import hopf_charge
from jax_solitons.topology import solid_angle

_PLAQ = ((0, 1), (0, 2), (1, 2))


def _rot(c, s, v):
    """Rotate a (3,N,N,N) vector field about e3 by angle with (cos, sin)=(c, s)."""
    return jnp.stack([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def unpack_iso(state):
    """(6,N,N,N) -> (n[0:3], A[3:6])."""
    return state[0:3], state[3:6]


def n_of_iso(state):
    """The (gauge-invariant) unit field carried by the state -- n is fundamental,
    so this is just the first three components (no 0/0, no derivation)."""
    return state[0:3]


def _cov_diff(n, A, e, dx):
    """Covariant forward differences [D_0 n, D_1 n, D_2 n], rotation-link form.

    D_i n = (R(theta_i) n(x + e_i) - n(x)) / dx,  theta_i = e * A_i * dx,
    R(theta) = rotation of (n1, n2) about e3 (the gauged isorotation generator
    J n = e3 x n). As dx->0, R ~ I + theta J, so D_i n -> d_i n + e A_i (e3 x n)
    -- minimal coupling with gauge coupling e. Exactly gauge-covariant: the link
    R parallel-transports n(x+e_i) back to x.
    """
    out = []
    for i in range(3):
        th = e * A[i] * dx                       # (N,N,N) link rotation angle
        c, s = jnp.cos(th), jnp.sin(th)
        nf = jnp.roll(n, -1, axis=1 + i)         # n(x + e_i), shape (3,N,N,N)
        Rn = jnp.stack([c * nf[0] - s * nf[1],
                        s * nf[0] + c * nf[1],
                        nf[2]])
        out.append((Rn - n) / dx)
    return out


@dataclasses.dataclass(frozen=True)
class IsoCovariantE2Term:
    """c2 * sum_i |D_i n|^2 -- the gauge-covariant Dirichlet term."""
    e: float = 1.0
    c2: float = 1.0
    name: str = "iso_e2"

    def __call__(self, state, grid: BoxGrid):
        n, A = unpack_iso(state)
        D = _cov_diff(n, A, self.e, grid.dx)
        acc = sum(jnp.sum(Di * Di, axis=0) for Di in D)
        return self.c2 * jnp.sum(acc) * grid.dx**3


def _gauged_area_form(n, A, e, dx, i, j):
    """Gauge-covariant area form on the (i,j) plaquette: parallel-transport the
    three neighbour corners back to the base point x via the rotation links, then
    take the solid angle of the n-quadrilateral. Because all corners are rotated
    to a common frame, the solid angle (rotation-invariant) is gauge-invariant;
    the link plaquette U_i(x)U_j(x+e_i) carries the magnetic flux F_ij into the
    transported corner C, so the area form sees the gauge field. At A=0 every link
    is the identity and this is exactly the bare `area_form_plaquette`.
    """
    thi = e * A[i] * dx
    thj = e * A[j] * dx
    thj_i = e * jnp.roll(A[j], -1, axis=i) * dx          # A_j(x + e_i)
    ci, si = jnp.cos(thi), jnp.sin(thi)
    cj, sj = jnp.cos(thj), jnp.sin(thj)
    cji, sji = jnp.cos(thj_i), jnp.sin(thj_i)
    base = n
    B = _rot(ci, si, jnp.roll(n, -1, axis=1 + i))                       # U_i n(x+e_i)
    Dd = _rot(cj, sj, jnp.roll(n, -1, axis=1 + j))                      # U_j n(x+e_j)
    nC = jnp.roll(jnp.roll(n, -1, axis=1 + i), -1, axis=1 + j)          # n(x+e_i+e_j)
    C = _rot(ci, si, _rot(cji, sji, nC))                               # U_i(x) U_j(x+e_i) n
    return solid_angle(base, B, C) + solid_angle(base, C, Dd)


@dataclasses.dataclass(frozen=True)
class IsoCovariantE4Term:
    """c4 * sum_{i<j} F_ij^2 via the gauge-covariant area form (parallel-transported
    solid angle). Reduces EXACTLY to bare E4AreaFormTerm at A=0 -- the geometric
    discretization that actually stabilizes hopfions (the naive cross-product form
    does not)."""
    e: float = 1.0
    c4: float = 4.0
    name: str = "iso_e4"

    def __call__(self, state, grid: BoxGrid):
        n, A = unpack_iso(state)
        dx = grid.dx
        e4 = 0.0
        for (i, j) in _PLAQ:
            Om = _gauged_area_form(n, A, self.e, dx, i, j)   # = F_ij * dx^2
            e4 = e4 + (Om / dx**2) ** 2
        return self.c4 * jnp.sum(e4) * dx**3


@dataclasses.dataclass(frozen=True)
class IsoMagneticTerm:
    """gmag * (1/2) sum_{i<j} F_ij^2 -- abelian Maxwell, F_ij = d_i A_j - d_j A_i
    (forward differences, gauge-invariant). This is the photon kinetic term; its
    flux is a SEPARATE topological sector from Q_H."""
    gmag: float = 1.0
    name: str = "iso_magnetic"

    def __call__(self, state, grid: BoxGrid):
        _, A = unpack_iso(state)
        dx = grid.dx
        s = 0.0
        for (i, j) in _PLAQ:
            dAj = (jnp.roll(A[j], -1, axis=i) - A[j]) / dx
            dAi = (jnp.roll(A[i], -1, axis=j) - A[i]) / dx
            F = dAj - dAi
            s = s + F * F
        return self.gmag * 0.5 * jnp.sum(s) * dx**3


@dataclasses.dataclass(frozen=True)
class IsoS2Constraint:
    """|n|=1 on the n block (state[0:3]); gauge field A (state[3:6]) free. n stays
    a fundamental S^2 field -> its Hopf charge is a hard topological invariant."""

    def project_tangent(self, state, grad):
        n = state[0:3]
        gn = grad[0:3]
        gnt = gn - jnp.sum(n * gn, axis=0, keepdims=True) * n
        return jnp.concatenate([gnt, grad[3:6]], axis=0)

    def retract(self, state):
        n = state[0:3]
        n = n / jnp.sqrt(jnp.sum(n * n, axis=0, keepdims=True))
        return jnp.concatenate([n, state[3:6]], axis=0)


def gauged_iso_model(c4: float = 4.0, e: float = 1.0, c2: float = 1.0,
                     gmag: float = 1.0) -> Model:
    """Gauged-isorotation Faddeev-Skyrme model (option 4). n fundamental + S^2-
    constrained, U(1) gauged via the e3-isorotation link, abelian Maxwell sector."""
    return Model(
        name="gauged_iso_faddeev",
        terms=(IsoCovariantE2Term(e=e, c2=c2),
               IsoCovariantE4Term(e=e, c4=c4),
               IsoMagneticTerm(gmag=gmag)),
        constraint=IsoS2Constraint(),
        charges=(lambda s, g: hopf_charge(s[0:3], g),),
    )
