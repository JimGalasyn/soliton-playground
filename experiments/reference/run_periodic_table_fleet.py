#!/usr/bin/env python3
"""PRESERVED FOR REFERENCE -- DOES NOT RUN AS-IS.

Migrated 2026-08-01 out of null-worldtube-private, which is being retired.
docs/BESTIARY.md states that six catalog entries' seed radii are "recoverable
verbatim from run_periodic_table_fleet.py"; that promise pointed into a repo
about to disappear. The leg table below is that record, unedited -- the `--R`
values (22.5, and 25.0 for the septafoil) are what BESTIARY relies on.

WHY IT CANNOT RUN HERE: RELAX_SHIP / CENSUS_SHIP list loose .py files that were
shipped flat to a rented box. That mechanism is gone -- both the engine
(jax_solitons.ehn) and the lab (soliton_playground.ehn_lab) are installable
packages now, so run_b2_vast.py pip-installs them instead. Reviving this driver
means replacing the ship lists with that approach, which cannot be validated
without renting a box. It is kept as the provenance record it is cited as, not
as a working entry point.

The `jax_solitons.campaign` imports were repointed to run-farm (that subpackage
was extracted in 0.0.8) and a hardcoded /home/jim/.venv path dropped, so the
file at least parses and its data is readable.

CHARTER: leg identifiers below include `conv48_lepton`, `conv48_electron` and
`e_twistm1_positron`. These are historical command-line labels from July 2026,
not identification claims, and BESTIARY's promise is that they are recoverable
VERBATIM -- rewriting them would falsify the provenance this file exists to
carry. Recorded as an explicit exemption in docs/BESTIARY.md.

--- original docstring follows ---

FIRST PERIODIC TABLE campaign — overnight fleet: the knot-spectrum zoo,
convergence runs, and the EXP29 census legs, on one Vast fleet.

Three leg families (all N=192-class, dedicated 4090s):
  ZOO  — new torus-knot / framed states via ehn_relax --geom torus
         (T(2,5), T(3,4)=8_19, T(2,7), T(3,5)=10_124 [det=1 at Lk=5 — the
         det-vs-Lk discriminator], Hopf rung, twist=2 ladder, framed
         cinquefoil, positron-parity electron), 24k steps + saved field for
         morning battery/registration/portraits.
  CONV — 48k-step reruns of the four core catalog particles (converged E for
         the ΔE ledger controls: framing quantum, mass ratios).
  EXP29 — the frozen census: 24 scoring legs (3 arms × 8 paired seeds,
         N=192 fast) + 16 locality legs (+mu0, seeds 0-3, {N=96,N=192} ×
         {fast,slow}) + one dt/2 integrator control (descriptive, outside
         prereg). Manifests carry the FULL timeline (both R5 readings
         computable post-hoc) — SCORING stays gated on the prereg session's
         A1/R5 ack; running the legs is reading-agnostic.

  python run_periodic_table_fleet.py --dry-run
  python run_periodic_table_fleet.py --max-parallel 8 --out output/periodic_table
"""
import argparse
import sys
from pathlib import Path

from run_farm.fleet import FleetExecutor, FleetLeg, SentinelReady
from run_farm.protocols import HostSpec, LaunchSpec
from run_farm.vast import VastLedger, VastProvider

HERE = Path(__file__).resolve().parent
IMG = "nvidia/cuda:12.2.2-runtime-ubuntu22.04"
ONSTART = (HERE / "vast" / "onstart.sh").read_text()
PYBOX = "/workspace/jaxenv/bin/python"

RELAX_SHIP = (str(HERE / "ehn_relax.py"), str(HERE / "ehn_knot_batch.py"),
              str(HERE.parent / "gpe_vortex_topology.py"), str(HERE / "ehn_cross_linking.py"))
CENSUS_SHIP = (str(HERE / "exp29_census.py"), str(HERE.parent / "ehn_cme_quench.py"),
               str(HERE.parent / "gpe_vortex_topology.py"), str(HERE / "ehn_cross_linking.py"))

RELAX_BASE = (f"{PYBOX} ehn_relax.py --N 192 --L 153.6 --core 2.0 --ic screened "
              f"--agrad wrapped --C 400 --cramp 8000 --alpha 1e-4 --samples 24 "
              f"--out out_ehn_relax ")


def zoo_legs():
    """(label, extra ehn_relax args) — the new-particle spectroscopy."""
    Z = [
        ("t25_cinquefoil", "--geom torus --tp 2 --tq 5 --R 22.5 --steps 24000"),
        ("t34_819_det3lk4", "--geom torus --tp 3 --tq 4 --R 22.5 --steps 24000"),
        ("t27_septafoil", "--geom torus --tp 2 --tq 7 --R 25.0 --steps 24000"),
        ("t35_10124_det1lk5", "--geom torus --tp 3 --tq 5 --R 22.5 --steps 24000"),
        ("hopf_rung2", "--geom torus --tp 1 --tq 2 --R 22.5 --steps 24000"),
        ("t23_twist2_ladder", "--geom torus --tp 2 --tq 3 --R 22.5 --twist 2 --steps 24000"),
        ("t25_twist1_framed", "--geom torus --tp 2 --tq 5 --R 22.5 --twist 1 --steps 24000"),
        ("e_twistm1_positron", "--geom torus --tp 1 --tq 1 --R 22.5 --twist -1 --steps 24000"),
    ]
    C = [
        ("conv48_t23", "--geom torus --tp 2 --tq 3 --R 22.5 --steps 48000"),
        ("conv48_t23_twist1", "--geom torus --tp 2 --tq 3 --R 22.5 --twist 1 --steps 48000"),
        ("conv48_lepton", "--geom torus --tp 1 --tq 1 --R 22.5 --steps 48000"),
        ("conv48_electron", "--geom torus --tp 1 --tq 1 --R 22.5 --twist 1 --steps 48000"),
    ]
    legs = []
    for label, extra in Z + C:
        steps = 48000 if label.startswith("conv") else 24000
        legs.append(FleetLeg(
            label=label,
            command=RELAX_BASE + extra + f" --save-every {steps}",
            ship=RELAX_SHIP, fetch="out_ehn_relax",
            done_when="out_ehn_relax/manifest.json"))
    return legs


def census_legs():
    legs = []
    for arm in ("+mu0", "-mu0", "0"):
        for seed in range(8):
            d = f"out_exp29/{arm}_N192_fast_s{seed}"
            legs.append(FleetLeg(
                label=f"x29_{arm.replace('+','p').replace('-','m')}_s{seed}",
                command=f"{PYBOX} exp29_census.py run --arm={arm} --seed {seed} --N 192 --rate fast",
                ship=CENSUS_SHIP, fetch="out_exp29",
                done_when=f"{d}/manifest.json"))
    for seed in range(4):                       # locality: +mu0 only, box × rate
        for N in (96, 192):
            for rate in ("fast", "slow"):
                if N == 192 and rate == "fast":
                    continue                    # covered by the scoring arm (same seeds)
                d = f"out_exp29/+mu0_N{N}_{rate}_s{seed}"
                legs.append(FleetLeg(
                    label=f"x29_loc_N{N}_{rate}_s{seed}",
                    command=f"{PYBOX} exp29_census.py run --arm +mu0 --seed {seed} --N {N} --rate {rate}",
                    ship=CENSUS_SHIP, fetch="out_exp29",
                    done_when=f"{d}/manifest.json"))
    # dt/2 integrator control (descriptive, OUTSIDE the prereg — §1 discipline)
    legs.append(FleetLeg(
        label="x29_dt2_control_s0",
        command=(f"{PYBOX} -c \"import exp29_census as XC; XC.CFG.update(dt=0.003); "
                 f"XC.mode_run('+mu0', 0, 192)\""),
        ship=CENSUS_SHIP, fetch="out_exp29",
        done_when="out_exp29/+mu0_N192_fast_s0/manifest.json"))
    return legs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-parallel", type=int, default=8)
    ap.add_argument("--run-timeout", type=int, default=14400)
    ap.add_argument("--max-dph", type=float, default=0.45)
    ap.add_argument("--gpu", default="RTX_4090")
    ap.add_argument("--skip-census", action="store_true")
    ap.add_argument("--out", default="output/periodic_table")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    legs = zoo_legs() + ([] if args.skip_census else census_legs())
    outdir = HERE / args.out
    if args.dry_run:
        print(f"{len(legs)} legs, {args.max_parallel} parallel -> {outdir}; "
              f"gpu={args.gpu} max_dph={args.max_dph}")
        for leg in legs:
            done = (outdir / leg.label / leg.marker()).exists()
            print(f"  {leg.label}: {'COMPLETE (skip)' if done else 'pending'}")
        return

    ledger = VastLedger(outdir / "vast_ledger.jsonl")
    provider = VastProvider(ledger=ledger)
    launch = LaunchSpec(image=IMG, onstart=ONSTART, disk_gb=32, label="periodic-table")
    spec = HostSpec(gpu_name=args.gpu, max_dph=args.max_dph, min_reliability=0.97,
                    min_inet_mbps=300, min_cuda=12.2, min_gpu_frac=1.0)
    ex = FleetExecutor(provider, launch, local_out_dir=str(outdir), host_spec=spec,
                       ready=SentinelReady(), max_parallel=args.max_parallel,
                       run_timeout=args.run_timeout, ledger=ledger)
    results = ex.run(legs)
    print("=" * 60)
    for r in sorted(results, key=lambda r: r.label):
        print(f"  {r.label}: {r.status}" + (f"  ({r.detail})" if r.detail else ""))
    print(f"ledger: {ledger.path}")


if __name__ == "__main__":
    main()
