#!/usr/bin/env python3
"""Does a Neel hopfion carry net axion charge? No -- and not by a near-miss.

Written 2026-09-02 to check a claim from the null-worldtube-private session against
this repo's own numerics rather than take it on trust. It retracts item 1 of
docs/LAB_CHARGE_BINDING.md.

THE IDENTITY. With div B = 0,

    grad(theta) . B = div(theta B)

so the net bound charge is a pure surface term:

    Q = (alpha/4pi^2) * closed-surface-integral of theta B.n

If theta is SINGLE-VALUED and tends to a constant on the boundary, Q = 0 exactly,
for ANY divergence-free B -- uniform, structured, or linking. Net charge requires a
theta that is NOT single-valued: a compact angle winding around a string, so theta
jumps 2pi across a cut and Q = 2pi * (flux through the cut). That is EHN's geometry
(a = arg phi2 winds around the phi2 string; phi1 flux links it) and it is what the
wrapped agrad preserves.

This is the SAME theorem NLINK_LADDER.md:407 already states to void the naive arm.
The lab proposal was written the same day and did not apply it to itself.

⚠ THE POSITIVE CONTROL IS NOT OPTIONAL. Every arm below reports a number, and
"Q = 0" from a broken integrator is indistinguishable from Q = 0 from physics. The
winding arm must return the analytic 2*pi*flux or nothing here is evidence. It does,
to 0.06% of the box-truncated value.

Run: python experiments/reference/axion_charge_identity.py
"""
import numpy as np
from scipy import integrate

N, L = 96, 8.0
dx = L / N
ax = (np.arange(N) - N // 2) * dx
X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
R2 = X**2 + Y**2 + Z**2


def hopf_texture(m, n):
    """Unit-vector field of Hopf index m*n: inverse stereographic R^3 -> S^3,
    then the generalised Hopf map on the spinor (u^m, v^n).

    Smooth, single-valued, and tending to the CONSTANT (0,0,-1) at large r --
    which is the only property the identity below actually needs.
    """
    d = R2 + 1.0
    a1, a2, a3, a4 = 2 * X / d, 2 * Y / d, 2 * Z / d, (R2 - 1.0) / d
    u, v = (a1 + 1j * a2) ** m, (a3 + 1j * a4) ** n
    nrm = np.sqrt(np.abs(u) ** 2 + np.abs(v) ** 2) + 1e-300
    u, v = u / nrm, v / nrm
    return (2 * np.real(np.conj(u) * v), 2 * np.imag(np.conj(u) * v),
            np.abs(u) ** 2 - np.abs(v) ** 2)


def grad(f, wrap=False):
    """Central differences, periodic. wrap=True is the compact-angle ('wrapped')
    construction: the difference is folded into (-pi, pi], which is what keeps a
    winding's branch cut from cancelling the winding it encodes."""
    out = []
    for a in range(3):
        d = np.roll(f, -1, a) - np.roll(f, 1, a)
        if wrap:
            d = (d + np.pi) % (2 * np.pi) - np.pi
        out.append(d / (2 * dx))
    return out


def d_(f, a):
    return (np.roll(f, -1, a) - np.roll(f, 1, a)) / (2 * dx)


def divergence(B):
    return sum(d_(B[a], a) for a in range(3))


def charge(theta, B, wrap=False):
    g = grad(theta, wrap=wrap)
    rho = sum(g[a] * B[a] for a in range(3))
    return rho.sum() * dx**3, np.abs(rho).sum() * dx**3


def main():
    # B fields. The first two are divergence-free in the SAME discrete sense as
    # the gradient above -- which is what the identity requires. Building the
    # second as a discrete curl, rather than writing a nice formula and hoping,
    # is the difference between testing the theorem and testing the stencil.
    B_uniform = (np.zeros_like(X), np.zeros_like(X), np.ones_like(X))
    A = (np.exp(-R2 / 4) * Y, np.exp(-R2 / 3) * Z, np.exp(-R2 / 5) * X)
    B_curl = (d_(A[2], 1) - d_(A[1], 2), d_(A[0], 2) - d_(A[2], 0),
              d_(A[1], 0) - d_(A[0], 1))

    # An azimuthal field circling the z axis: the one whose flux LINKS a string
    # on that axis. Its discrete divergence is not machine-zero near the axis
    # (the 1/rho), so it is used for the winding control, where the effect is
    # O(30), and never to carry the Q = 0 claim, which rests on the two above.
    rho_c = np.sqrt(X**2 + Y**2) + 1e-12
    b_amp = np.exp(-((rho_c - 1.5) ** 2) / 0.5) * np.exp(-Z**2 / 4)
    B_link = (-Y / rho_c * b_amp, X / rho_c * b_amp, np.zeros_like(X))

    print(f"grid N={N} L={L} dx={dx:.4f}")
    for nm, B in (("uniform z", B_uniform), ("curl A", B_curl),
                  ("azimuthal (links z)", B_link)):
        print(f"  max|div B| = {np.abs(divergence(B)).max():.2e}  [{nm}]")

    print("\nSINGLE-VALUED theta -- the dynamical-axion case, theta = pi + 0.3 n_z")
    for (m, n) in ((1, 1), (1, 2), (2, 1), (2, 2)):
        theta = np.pi + 0.3 * hopf_texture(m, n)[2]
        for lab, B in (("uniform", B_uniform), ("curlA", B_curl)):
            q, l1 = charge(theta, B)
            print(f"  H={m * n}  B={lab:8s}  Q = {q:+.3e}   local |rho| L1 = {l1:6.3f}")
    print("  -> zero at EVERY Hopf index, while the LOCAL density is large.")

    print("\nPOSITIVE CONTROL -- theta winds once around the z axis")
    theta_w = np.arctan2(Y, X)
    q_w, l1_w = charge(theta_w, B_link, wrap=True)
    q_naive, _ = charge(theta_w, B_link, wrap=False)
    f_r = integrate.quad(lambda r: np.exp(-(r - 1.5) ** 2 / 0.5), 0, L / 2)[0]
    f_z = integrate.quad(lambda z: np.exp(-z**2 / 4), -L / 2, L / 2)[0]
    q_box = 2 * np.pi * f_r * f_z
    print(f"  wrapped grad  Q = {q_w:8.4f}   analytic 2*pi*flux = {q_box:8.4f}"
          f"   ({100 * (q_w / q_box - 1):+.2f}%)   local |rho| L1 = {l1_w:.3f}")
    print(f"  naive grad    Q = {q_naive:+.3e}   <- the ladder's voided arm")
    ok = abs(q_w / q_box - 1) < 0.02
    print(f"  -> CONTROL {'PASSES' if ok else 'FAILS'}: a nonzero charge IS detected "
          f"when one exists.")

    q_along, _ = charge(theta_w, B_uniform, wrap=True)
    print(f"\n  B ALONG the string instead of linking it: Q = {q_along:+.3e}")
    print("  -> the charge counts LINKING, not field strength. grad(theta) is "
          "azimuthal,\n     so a field parallel to the string contributes nothing.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
