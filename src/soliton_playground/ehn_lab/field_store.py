#!/usr/bin/env python3
"""Content-addressed store for catalog `field.npz` states — the relaxed states
stop being re-runs.

WHY THIS EXISTS. `particle_catalog.py` pins every entry's field by sha256 and
then keeps it at exactly one gitignored path, `particles/<name>/field.npz`. Ten
entries record a sha256; zero of those files survive. What was preserved is a
*description* of each state — Lk, det, segment counts, an energy budget — and a
description cannot be restarted from. So every verification became a fresh
relaxation, every relaxation needed a 24 GB GPU the working machine does not
have, and each attempt re-entered the rental lottery. Three attempts at the
N=192 trefoil's determinant have now failed in transport, not in physics.

This module keeps the bytes. Objects are addressed by their own sha256, which is
already the catalog's primary key, so no entry schema changes.

    field_store/
      objects/<sha[:2]>/<sha>.npz     the states (gitignored, see PROMOTION)
      index.json                      what is held + where it came from (tracked;
                                      created by the first `put`, which happened
                                      2026-08-02 — see RECOVERY below)

DESIGN RULES, each one a bug this program has already paid for:

  * Never conclude from a name or a size. A torn `field.npz` keeps a valid PK
    header and a plausible 86 MB, and failed only on open. Every put and every
    verify opens the zip and CRCs each member (`run_farm.arrival` does this, and
    exists because of this exact file).
  * MISSING is never a pass. `status()` reports absence as absence; callers that
    want a gate must treat a skipped entry as a failure. `leg_B5` did not, and
    reported `B5_tracer_conformance: True` while conforming against nothing.
  * Writes are atomic (tmp + fsync + os.replace), the same fix `ehn_relax`
    needed: a crash mid-write must not destroy the copy already held.
  * A re-derived state is labelled, not silently accepted. A fresh relaxation
    will NOT reproduce July's bytes — `np.savez` output is not stable across
    engine versions, and the writer itself changed. When a stored object's
    sha256 differs from the entry's declared one, that is recorded as
    `rederived`, because "same physics" and "same bytes" are different claims
    and the catalog's sha256 asserts the second.

RECOVERY, 2026-08-02. This store was built to hold ten states that three separate
documents recorded as lost: `status()` said 0/10 held, `docs/BESTIARY.md` said
"All ten states are gone", and refilling them was costed at a 24 GB GPU. **All
ten were found**, gitignored, in `null-worldtube-private`
(`simulations/engine_dogfood/out_*_n192*` and `output/periodic_table/*/out_ehn_relax`),
and every sha256 matches what the catalog declared in July — so none is
`rederived`; they are the originals. `put` CRC-verified each on the way in, and
`leg_B5` now conforms 10 of 10 against their registrations, reproducing July's
linking numbers and knot determinants exactly. The gate that reported
`B5_tracer_conformance: False` for the right reason now reports True for the
right reason.

They were never regenerated because they were never actually missing — only
unfindable. That is the argument for this store existing.

PROMOTION. `objects/` is gitignored because 7.4 GB does not belong in git. The
store is local-first and self-describing: `index.json` is tracked, so the repo
states what is held, and `status()` derives what is missing from the tracked
catalog. To promote,
either enable git-lfs on `objects/**` where a push works, or `tar` the store and
attach it to a release — the layout is designed to survive both without a
rewrite, since object names are content hashes.

  python field_store.py status                     what is held / missing
  python field_store.py put <name> <field.npz>     verify, hash, store
  python field_store.py materialize [<name>]       link into particles/<name>/
  python field_store.py verify                     re-CRC everything held
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _repo_root(start=_HERE):
    """Nearest ancestor holding pyproject.toml, else the package dir.

    The objects are hundreds of megabytes and do not belong inside an installable
    package directory; the small catalog metadata does. So the two are split:
    entries travel with the code, states sit at the repo root.
    """
    for d in [start, *start.parents]:
        if (d / "pyproject.toml").exists():
            return d
    return start


CATALOG = _HERE / "particles"
STORE = Path(os.environ.get("SOLITON_FIELD_STORE") or (_repo_root() / "field_store"))
OBJECTS = STORE / "objects"
INDEX = STORE / "index.json"

FIELD_NAME = "field.npz"


# ---------------------------------------------------------------- integrity --
def verify_npz(path):
    """None if `path` is a complete, CRC-clean npz; else a one-line reason.

    Prefers `run_farm.arrival`, which was written for precisely this failure and
    is the program's authoritative answer to "did this artifact arrive intact".
    Falls back to a local zip test so the store still works where run-farm is
    old or absent (a rented box, an offline checkout) rather than silently
    skipping the check — an integrity check that can be skipped is not one.
    """
    path = Path(path)
    if not path.exists():
        return "missing"
    if path.stat().st_size == 0:
        return "empty"
    try:
        from run_farm.arrival import verify_report
        problems = verify_report(path.parent, patterns=(path.name,)).problems
        fatal = [p for p in problems if p.fatal and Path(p.path).name == path.name]
        return f"{fatal[0].kind}: {fatal[0].detail}" if fatal else None
    except ImportError:
        pass
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
        return None if bad is None else f"corrupt member {bad}"
    except zipfile.BadZipFile as e:
        return f"truncated: not a valid zip ({e})"
    except OSError as e:                                   # noqa: BLE001
        return f"unreadable: {e}"


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _atomic_write_text(path, text):
    """tmp + fsync + os.replace. The index is small, but a torn index is worse
    than a torn field: it is the only record of what the store holds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# -------------------------------------------------------------------- index --
def load_index():
    try:
        return json.loads(INDEX.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_index(ix):
    _atomic_write_text(INDEX, json.dumps(ix, indent=1, sort_keys=True) + "\n")


def object_path(sha):
    return OBJECTS / sha[:2] / f"{sha}.npz"


def catalog_entries():
    """{name: entry-dict} for every registered particle."""
    out = {}
    for p in sorted(CATALOG.glob("*/entry.json")):
        try:
            e = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out[e.get("name", p.parent.name)] = e
    return out


def declared_sha(entry):
    return ((entry or {}).get("field") or {}).get("sha256")


# ---------------------------------------------------------------------- put --
def put(name, src, *, note="", force=False):
    """Verify `src`, hash it, store it by content, and index it under `name`.

    Refuses a corrupt file outright: the store's whole purpose is that what it
    returns is loadable, and admitting a torn object would move the failure to
    whoever reads it next -- which is exactly how three GPU runs were spent.
    """
    src = Path(src)
    why = verify_npz(src)
    if why:
        raise SystemExit(f"REFUSED {name}: {src} is not intact -- {why}")

    entries = catalog_entries()
    if name not in entries and not force:
        raise SystemExit(f"REFUSED {name}: no catalog entry. Register it first "
                         f"(particle_catalog.py register), or pass --force to "
                         f"store an unregistered state.")
    sha = sha256_file(src)
    want = declared_sha(entries.get(name))
    rederived = bool(want) and sha != want

    dest = object_path(sha)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".npz.tmp")
        with open(src, "rb") as fin, open(tmp, "wb") as fout:
            while blk := fin.read(1 << 22):
                fout.write(blk)
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(tmp, dest)
        # Re-verify AFTER landing: the copy is what future readers get, and a
        # verified source says nothing about a full disk on this side.
        why = verify_npz(dest)
        if why:
            dest.unlink(missing_ok=True)
            raise SystemExit(f"REFUSED {name}: stored copy failed re-check -- {why}")

    ix = load_index()
    ix[name] = {
        "sha256": sha,
        "declared_sha256": want,
        "rederived": rederived,
        "bytes": dest.stat().st_size,
        "added": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(src),
        "note": note,
    }
    save_index(ix)
    tag = "  [REDERIVED -- bytes differ from the entry's declared sha256]" if rederived else ""
    print(f"stored {name}  sha256 {sha[:12]}…  {dest.stat().st_size:,} B{tag}")
    return ix[name]


# ---------------------------------------------------------------------- get --
def get(name, *, verify=True):
    """Path to the held object for `name`, or None. Verifies before handing it
    over unless explicitly told not to."""
    rec = load_index().get(name)
    if not rec:
        return None
    p = object_path(rec["sha256"])
    if not p.exists():
        return None
    if verify and verify_npz(p):
        return None
    return p


def materialize(name, *, link=True):
    """Place the held state at `particles/<name>/field.npz`, where every existing
    consumer already looks (`particle_catalog.load`, `_measure`, `leg_B5`).

    Hardlink by default: same bytes, no second copy, and the store stays the
    single owner. Falls back to a copy across filesystems.
    """
    src = get(name)
    if src is None:
        return None
    dest = CATALOG / name / FIELD_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.stat().st_ino == src.stat().st_ino:
            return dest
        dest.unlink()
    try:
        if link:
            os.link(src, dest)
        else:
            raise OSError("copy requested")
    except OSError:
        tmp = dest.with_suffix(".npz.tmp")
        with open(src, "rb") as fin, open(tmp, "wb") as fout:
            while blk := fin.read(1 << 22):
                fout.write(blk)
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(tmp, dest)
    return dest


# ------------------------------------------------------------------- status --
def status():
    """One row per catalog entry. `state` is one of:

      held        object present, CRC clean, bytes match the declared sha256
      rederived   present and clean, but NOT July's bytes (re-run, not restore)
      corrupt     indexed but the object fails its integrity check
      missing     nothing held -- and this is a FAILURE for any gate, not a skip
    """
    ix = load_index()
    rows = []
    for name, entry in sorted(catalog_entries().items()):
        rec = ix.get(name)
        want = declared_sha(entry)
        if not rec:
            rows.append({"name": name, "state": "missing", "sha256": None,
                         "declared_sha256": want, "detail": "no object held"})
            continue
        p = object_path(rec["sha256"])
        why = verify_npz(p)
        if why:
            state, detail = "corrupt", why
        elif rec.get("rederived"):
            state, detail = "rederived", "bytes differ from declared sha256"
        else:
            state, detail = "held", ""
        rows.append({"name": name, "state": state, "sha256": rec["sha256"],
                     "declared_sha256": want, "detail": detail,
                     "materialized": (CATALOG / name / FIELD_NAME).exists()})
    return rows


def verify_all():
    """Re-CRC every held object. Returns the list of problems (empty = clean)."""
    problems = []
    for name, rec in sorted(load_index().items()):
        p = object_path(rec["sha256"])
        why = verify_npz(p)
        if why:
            problems.append(f"{name}: {why}")
            continue
        actual = sha256_file(p)
        if actual != rec["sha256"]:
            problems.append(f"{name}: content hash drifted "
                            f"{rec['sha256'][:12]}… -> {actual[:12]}…")
    return problems


# ---------------------------------------------------------------------- CLI --
def _print_status():
    rows = status()
    if not rows:
        print("no catalog entries found under", CATALOG)
        return 1
    width = max(len(r["name"]) for r in rows)
    counts = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
        mark = {"held": "OK      ", "rederived": "REDERIVED",
                "corrupt": "CORRUPT ", "missing": "MISSING "}[r["state"]]
        sha = (r["sha256"] or "-")[:12]
        mat = " -> particles/" if r.get("materialized") else ""
        print(f"  {mark} {r['name']:<{width}}  {sha}{mat}"
              + (f"  ({r['detail']})" if r["detail"] else ""))
    total = len(rows)
    held = counts.get("held", 0) + counts.get("rederived", 0)
    print(f"\n{held}/{total} states held"
          + (f"  --  {counts['missing']} MISSING" if counts.get("missing") else "")
          + (f", {counts['corrupt']} CORRUPT" if counts.get("corrupt") else ""))
    return 0 if held == total else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="what is held / missing (exit 1 if any missing)")
    p_put = sub.add_parser("put", help="verify, hash and store a field.npz")
    p_put.add_argument("name")
    p_put.add_argument("path")
    p_put.add_argument("--note", default="")
    p_put.add_argument("--force", action="store_true",
                       help="store even without a catalog entry")
    p_mat = sub.add_parser("materialize", help="link held states into particles/")
    p_mat.add_argument("name", nargs="?", default=None)
    sub.add_parser("verify", help="re-CRC everything held (exit 1 on any problem)")
    a = ap.parse_args()

    if a.cmd == "status":
        return _print_status()
    if a.cmd == "put":
        put(a.name, a.path, note=a.note, force=a.force)
        return 0
    if a.cmd == "materialize":
        names = [a.name] if a.name else list(load_index())
        if not names:
            print("nothing held to materialize")
            return 1
        rc = 0
        for n in names:
            dest = materialize(n)
            print(f"  {'linked ' if dest else 'FAILED '} {n}"
                  + (f" -> {dest.relative_to(_HERE)}" if dest else " (not held or corrupt)"))
            rc |= 0 if dest else 1
        return rc
    if a.cmd == "verify":
        problems = verify_all()
        for p in problems:
            print("  PROBLEM", p)
        held = len(load_index())
        print(f"{held - len(problems)}/{held} held objects verified clean")
        return 1 if problems else 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
