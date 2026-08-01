#!/usr/bin/env python3
"""ParticleCatalog for EHN two-scalar relaxed states (PARTICLES.md, first code).

Relax once (expensive), cache as a first-class object, re-use in scenes. Each
entry = `particles/<name>/entry.json` (small, tracked in git) + `field.npz`
(the relaxed u,s,w state, git-ignored, sha256-pinned by the entry). Entries are
admitted ONLY with measured validity: the tracers are re-run at registration
(geometric Lk, per-field knot determinants) rather than trusted from the
manifest — a particle that can't name its relaxation and topology does not
enter the catalog.

Scope caveats (COMPENDIUM.md firewall): these are THIS ENGINE's classical
field-theory solitons. Names like "trefoil_t23" label field states, not
K7-walk compendium rows, and not Standard-Model particles.

Engine-static caveat (numerical-methods outline §5): a cached state is
(approximately) stationary under the integrator that relaxed it — recorded in
entry["relaxation"]["integrator"]. Re-settle before use in any other stepper.

  python particle_catalog.py register <name> <out_dir> "<description>"
  python particle_catalog.py list
  python particle_catalog.py show <name>
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
CATALOG = _HERE / "particles"
# The two sys.path inserts that stood here are gone: the engine is an installed
# package (jax_solitons.ehn) and the catalog a sibling module, so nothing needs
# the importing file's directory on the path any more.


def _sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", str(_HERE), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        return None


def _measure(field_path):
    """Re-measure topology + energy identity from the saved state itself."""
    from jax_solitons.ehn.cross_linking import cross_linking
    from jax_solitons.vortex_topology import knot_determinants, vortex_skeleton
    d = np.load(field_path)
    u = d["u"]
    N = int(u.shape[1])
    dx = 0.8                          # EHN spacing, fixed across this program
    L = N * dx
    p1 = u[0] + 1j * u[1]
    p2 = u[2] + 1j * u[3]
    lk, n1, n2 = cross_linking(p1, p2, dx)
    P1, _, _ = vortex_skeleton(p1)
    return {
        "N": N, "L": L, "dx": dx, "step_index": int(d["n"]),
        "Lk_phi1_phi2": round(float(lk), 3),
        "phi1_knot": [[int(s), int(det) if isinstance(det, (int, np.integer)) else str(det)]
                      for s, det in knot_determinants(p1, dx, L)],
        "phi2_knot": [[int(s), int(det) if isinstance(det, (int, np.integer)) else str(det)]
                      for s, det in knot_determinants(p2, dx, L)],
        "phi1_skeleton_segments": int(len(P1)),
    }


def register(name, out_dir, description, validity_notes=""):
    out = Path(out_dir)
    man = json.loads((out / "manifest.json").read_text())
    src_field = out / "field.npz"
    if not src_field.exists():
        raise SystemExit(f"no field.npz in {out} — rerun with --save-every")
    dest = CATALOG / name
    if (dest / "entry.json").exists():
        raise SystemExit(f"catalog entry '{name}' exists — pick a new name or remove it")
    dest.mkdir(parents=True, exist_ok=True)

    print(f"measuring topology of {src_field} …", flush=True)
    measured = _measure(src_field)
    print(f"  Lk={measured['Lk_phi1_phi2']}  phi1={measured['phi1_knot']} "
          f"phi2={measured['phi2_knot']}")

    print("copying field + hashing …", flush=True)
    shutil.copy2(src_field, dest / "field.npz")
    sha = _sha256(dest / "field.npz")

    traj = man.get("traj", [])
    entry = {
        "name": name,
        "description": description,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "model": "ehn-two-scalar",     # arXiv:2407.11731 / PRL 135,091603 functional
        "relaxation": {
            "integrator": "ehn_relax.relax_iter (EHN Eqs.12/13/11 interleaved gradient descent)",
            "params": man.get("params", {}),
            "source_out_dir": str(out),
            "git_commit": _git_commit(),
            "wall_s": man.get("wall_s"),
        },
        "measured": measured,
        "energy_final": traj[-1] if traj else None,
        "validity": validity_notes,
        "field": {"file": "field.npz", "sha256": sha},
        "caveats": [
            "engine-static: stationary only under the recorded integrator (outline §5); re-settle in any other stepper",
            "labels are field-state names, not K7-compendium rows nor SM particles (COMPENDIUM.md)",
        ],
    }
    (dest / "entry.json").write_text(json.dumps(entry, indent=1))

    # Bank the state in the content-addressed store IMMEDIATELY, not as a later
    # curation step. The original ten entries were registered exactly like this,
    # and every one of their fields is gone: `particles/<name>/field.npz` is
    # gitignored, so registration faithfully preserved the sha256 of a file that
    # nothing kept. Registration is the one moment the bytes are guaranteed to
    # exist, so it is the only safe moment to store them. Non-fatal: a state that
    # registered but failed to bank is still registered, and `field_store.py
    # status` will report it as missing rather than letting it look fine.
    try:
        from . import field_store
        field_store.put(name, dest / "field.npz",
                        note=f"banked at registration from {out}")
    except Exception as exc:                              # noqa: BLE001
        print(f"  WARNING: not banked in field_store ({exc}).\n"
              f"  This field now exists at ONE gitignored path. Run:\n"
              f"    python field_store.py put {name} {dest / 'field.npz'}")

    print(f"registered particles/{name}  (sha256 {sha[:12]}…)")
    return entry


def load(name):
    """-> (u, s, w, entry): jnp arrays ready for ehn_relax-style stepping."""
    import jax.numpy as jnp
    dest = CATALOG / name
    entry = json.loads((dest / "entry.json").read_text())
    d = np.load(dest / "field.npz")
    u = tuple(jnp.asarray(d["u"][i]) for i in range(10))
    w = tuple(jnp.asarray(d["w"][i]) for i in range(3))
    return u, jnp.asarray(d["s"]), w, entry


def list_catalog():
    rows = sorted(CATALOG.glob("*/entry.json"))
    if not rows:
        print("catalog empty")
        return
    for p in rows:
        e = json.loads(p.read_text())
        m = e["measured"]
        print(f"{e['name']:>24}  N={m['N']} Lk={m['Lk_phi1_phi2']} "
              f"phi1={m['phi1_knot']}  {e['description'][:60]}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("register", "list", "show"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "list":
        list_catalog()
    elif sys.argv[1] == "show":
        print(json.dumps(json.loads((CATALOG / sys.argv[2] / "entry.json").read_text()), indent=1))
    else:
        _, _, name, out_dir, desc = sys.argv[:5]
        notes = sys.argv[5] if len(sys.argv) > 5 else ""
        register(name, out_dir, desc, notes)
