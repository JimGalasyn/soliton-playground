<!--
RESTORED 2026-08-01 into soliton-playground, verbatim below this comment.

Origin: null-worldtube-private, commit d266443 (2026-07-10), path
analysis/STANDARD_BOX_SPEC.md, on branch worktree-more-cosmogenesis. It was
never merged to that repo's main, which is why two successive versions of
docs/CENSUS_PROTOCOL.md recorded it as missing and then as never having existed.
That repo is deprecated; this copy puts the normative document beside the battery
that cites it (src/soliton_playground/ehn_lab/standard_box.py).

Nothing below is edited. It is DRAFT v1 and it is the source of the acceptance
constants the battery implements: el/mag = 2349*(14/R)^2*(C/400)^2 with a wrapped
threshold of 147, the 0.35*L radius ceiling, alpha <~ O(dx^2/lambda), and
xi_c >= 2*dx. Verified against the code on restore -- they agree exactly.

Known tension, recorded here so it is not rediscovered: at C=400 the expulsion
wall and the g2 ceiling have no common feasible region. Expulsion needs
R >= 0.364*L, the ceiling caps R <= 0.35*L, and el/mag = 159 vs 147 even at the
most favourable R this spec allows. The largest C admitting a non-empty envelope
is ~384. See docs/CENSUS_PROTOCOL.md.
-->

# STANDARD BOX (SBX) — a certified stability domain + conformance battery for knot/link experiments

**DRAFT v1, 2026-07-10, Jim + Program C. For P riders (the engine-side
numbers are P's to confirm). Motivation: the R-C-LC-1 gate has killed
THREE designs pre-data (2a, 2b, native-scale 2c) for one root cause —
experiments landing outside the wrapped lock's validated envelope,
rediscovered bespoke each time. This spec converts that tribal
knowledge into an instrument: a frozen reference configuration
CERTIFIED to support stable knots/links, a conformance battery any
engine/pipeline must pass to claim it, and a certificate registry
contracts can cite instead of re-deriving gates. Charter §5 shared
instrument: additive, announced, versioned.**

## 1. The concept

A **standard box** is NOT just a parameter set — it is three things:

1. **A certified point**: box + couplings + engine mode with named
   artifacts proving stability (SHA-pinned relaxed states, the
   ParticleCatalog pattern).
2. **A measured envelope**: the walls around that point, with the
   quantitative laws that locate them — so "will my experiment hold?"
   is arithmetic, not a fleet run.
3. **A conformance battery**: a versioned test suite that (re)issues a
   CERTIFICATE for any (engine SHA, box, mode) triple. Contracts cite
   the certificate; the per-contract gate reduces to "certificate
   valid + experiment inside the envelope."

## 2. SB-1 "particle box" (the de facto standard, now formalized)

The configuration every held-knot result already lives in:

- **Grid**: N=192, L=153.6, dx=0.8 (cores ξ_c ≈ 1–2 units, resolved).
- **Couplings**: λ=1000, U=50, κ=8e-4, C-ramp→400 (cramp 8000).
- **Engine mode**: `ehn_relax`, **wrapped ∂a** (the lock), α=1e-4.
- **Seed scale**: structure radius R ∈ [0.2, 0.35]·L (≈ 31–54 units).
- **Certified capabilities (artifacts exist)**: Hopf nlink=1 hold;
  T(2,3) trefoil Lk=−3.000 / det=3 / zero shrink, 36k steps
  (`particles/trefoil_t23`, SHA-pinned); two-body composition
  (`compose_pair.py`); N_link=4 hold at the N=320 scale-up.

## 3. The measured envelope (the walls, with their laws)

| wall | law | consequence |
|---|---|---|
| electric expulsion | el/mag = 2349·(14/R)²·(C/400)²; wrapped threshold ≈147, bilinear ≈37 | minimum R at given C; OR maximum C at given R — **C is a free lever when binding energetics aren't the claim** |
| global-string tension | g2 ∝ R·ln L | R has a CEILING (~0.35L); can't buy retention with unbounded R |
| gradient-flow stability | α ≲ O(dx²/λ) | fine grids need α rescaled (dx 0.8→0.15 ⟹ α 1e-4→~3e-6) |
| core resolution | ξ_c ≳ 2·dx | can't shrink cores to fake large R/ξ_c |
| tracer validity | skeleton/net_linking needs min_seg=10 resolvable | sets the smallest countable loop |

**The structural theorem this forces (the census lesson):** a quench's
freeze-out loops have R/ξ_c ~ few — set by KZ physics, NOT by box size.
**No standard box exists at quench-native scale**; making L bigger
makes MORE small loops, not bigger ones. Therefore census-class
experiments always enter SB-1 through a **certified map** (§5), or
through the low-C corner of the envelope. This is why the standard-box
architecture is "one certified box + certified maps," not "a box per
experiment."

## 4. The conformance battery (versioned; certificate = JSON, SHA-keyed)

Run on any claimed (engine SHA, box, mode); ~single-GPU, hours:

- **B1 Hopf dual-meter**: wrapped — Lk→integer, zero RECONNECT, Q
  retained; bilinear control — geometry holds but charges die. BOTH
  behaviors must reproduce (the discrimination IS the instrument).
- **B2 trefoil hold**: T(2,3), 12k-step battery version; Lk=−3, det=3,
  no shrink trend.
- **B3 paired parity**: hand-seeded +Lk and −Lk; both signs held; no
  chirality artifact (the RT-5 leg, made standard).
- **B4 charge retention**: Q, ∫ρ within stated bands at battery end.
- **B5 tracer conformance**: net_linking exact on known configs
  (catalog states); the +25% grid-degeneracy bias check.
- **B6 stability margin**: energy monotone (relax), no NaN, ≥4×
  α-headroom demonstrated.
- **B7 (conditional) map fidelity**: if the pipeline claims a certified
  map (§5), per-realization Lk + component count preserved across it;
  lost-loop fraction receipted.

**Certificate**: {battery version, engine blob SHA(s), box params,
mode flags, per-test results + bands, artifact hashes, date}. Any
engine edit invalidates (SHA-keyed). Registry file alongside the
ParticleCatalog. The Auditor can re-run the battery from a certificate
alone — it doubles as the reproduction harness.

## 5. Certified maps (how experiments ENTER the box)

- **M1 magnification** (census → SB-1): φ(x) → φ(x/s) interpolation;
  Lk scale-invariant; B7 gates it; parity-even loss receipted.
- **M2 handoff** (real-time → relax): the 2c state translation
  (φ,A carried; B=∇×A; E dropped with disposal receipt) — already
  spec'd in the frozen contract; battery-ized here.
- Future maps register the same way: state the invariant they
  preserve, add a B7-class fidelity test, receipt what they discard.

## 6. What this buys, program-wide

- Locked census: gate becomes "SB-1 certificate + M1+M2 fidelity."
- Neutron lifecycle Stage B / Genesis run: same certificate, no
  bespoke re-validation.
- EXP29b cell volumes + carrier-cell promotion receipts (A5): measured
  IN a certified box, so "cell geometry" numbers are comparable across
  experiments.
- I2 pilot: c_B-flatness across carrier content needs runs whose
  stability is not a confound — cite the certificate.
- Auditor: every claim's "runnable evidence" becomes battery-replayable.

## 7. Open riders for P (engine side)

1. Confirm/correct the SB-1 numbers (§2) and the wall constants (§3)
   against the as-committed engine.
2. Does SB-1 want a second certified point at N=96 (cheap battery for
   CI-style regression) with its known charge-sector caveat stated?
3. Battery runtime budget: which tests run per-commit vs per-release.
4. Ownership: propose joint (the twist-flux-meter pattern) — P hosts
   `standard_box.py` battery in engine_dogfood; C owns the spec + the
   certificate registry format; either can run.

— Jim + NWT/more-cosmogenesis (Program C), for P riders
