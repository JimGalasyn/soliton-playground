import math
e, h, eps0, hbar = 1.602e-19, 6.626e-34, 8.854e-12, 1.055e-34
alpha = 1/137.036
PHI0 = h/e                      # flux quantum, Wb

print("EHN:                  rho = C * grad(a) . B          C = 400")
print("axion electrodynamics: rho = (alpha/4pi^2) grad(theta) . B")
print("  -> SAME FORM. a magnetic TI has a dynamical theta tied to the Neel vector.\n")

# bound charge on a hopfion of size Lh in field B, from a theta texture of ~pi
for Lh_nm, B in ((100, 1.0), (100, 10.0), (50, 1.0)):
    Lh = Lh_nm*1e-9
    flux = B * Lh**2
    n_e = 0.5 * flux/PHI0          # half-quantised surface Hall response
    print(f"hopfion {Lh_nm} nm in B = {B:4.1f} T:  flux/flux0 = {flux/PHI0:6.2f}"
          f"  -> bound charge ~ {n_e:.2f} e")

# is that charge enough to stabilise, i.e. compete with exchange?
print()
A_ex = 1e-11                       # J/m, typical exchange stiffness
for Lh_nm, eps_r in ((100, 1.0), (100, 50.0)):
    Lh = Lh_nm*1e-9
    E_hopf = A_ex * Lh                       # ~ A * L for a texture of size L
    E_coul = e**2/(4*math.pi*eps0*eps_r*Lh)
    print(f"hopfion {Lh_nm} nm, eps_r={eps_r:4.0f}:  E_exchange ~ {E_hopf/e:8.2f} eV"
          f"   E_coulomb(1e) ~ {E_coul/e*1e3:7.3f} meV   ratio {E_coul/E_hopf:.1e}")

# multiferroic route: polarisation from the spin-current (KNB) mechanism
print("\nmultiferroic (type-II) route, NOT alpha-suppressed:")
P_typ = 1e-4                       # C/m^2, spiral multiferroic e.g. TbMnO3 ~ 8e-5
for Lh_nm in (100, 50):
    Lh = Lh_nm*1e-9
    Q = P_typ * Lh**2               # bound charge ~ P * area for a texture
    print(f"  hopfion {Lh_nm} nm: P ~ {P_typ:.0e} C/m^2 -> bound charge ~ {Q/e:.1f} e")
