"""What code produced this number — the engine's identity, not the physics'.

`gpe_lab.provenance()` names the MEDIUM (preset, model, protecting charge). That is
scientific provenance and it is not this module's job. This module answers the other
question, which until now only the EHN wing asked: **which tree computed it.**

⚠ **An editable install turns another repo's working tree into an unrecorded
independent variable.** `jax-solitons` is installed here from a sibling checkout
(`pip install -e ../jax-solitons`, README "Setup"), so it resolves to
`~/repos/jax-solitons/src/jax_solitons` — a live tree, not a copy. Every census
entrant, cascade, quench and opener manifest in `output/` was therefore produced by
whatever was checked out at that instant, including branch state and mid-edit state,
and none of them records which. The EHN wing has recorded it since 2026-08-01
(`ehn_lab/standard_box.py`, whose `engine_sha()` this module now owns); the census
wing did not, and the two wings drifted because the discipline lived in one campaign
script's docstring instead of in a shared instrument.

⚠⚠ **A NON-EDITABLE install must not be git-stamped at all, and the old code stamped
it anyway.** `.venv` sits INSIDE this working tree, so for a wheel install
`git -C <site-packages>/<pkg> rev-parse HEAD` walks *up* and describes
SOLITON-PLAYGROUND. Verified here against numpy, which is a real dist install in this
venv: git run from `numpy/` returns this repo's HEAD, and `status --porcelain` scoped
to that path returns EMPTY — so the stamp would have read as a clean, definite commit
naming the wrong repository. **A wrong identity is worse than a missing one, because
it is never read as a gap and so is never chased.**

⚠ **It was latent, not observed.** Both trees this repo depends on (`jax-solitons`,
`run-farm`) happen to be installed editable, so the branch never executed. That is
exactly how the same defect survived nine days in `abiogenesis`, where both members of
its `PACKAGES` were editable too; it was found only when the module was ported to
`Morphospace`, whose install mix supplied the test case and whose first stamp reported
`exoclimate` wearing Morphospace's SHA. **Transfer is the test.**
`tests/test_provenance.py` manufactures the condition locally from numpy rather than
waiting for the next port.

⚠ **The guard is OWNERSHIP, not a path heuristic.** "Is it under site-packages" would
miss a vendored copy and would have to be kept in step with every install layout. The
question git can answer directly is whether the repo it found actually TRACKS the file
being stamped: `git ls-files --error-unmatch`. soliton-playground does not track
`jax_solitons/ehn/relax.py`, so the enclosing-repo answer is rejected on positive
evidence rather than on a guess about directory names.

⚠ **`jax_solitons.__version__` does not solve this** — the dist version was `0.0.8`
before and after the commits that moved the stage-2 numbers. The record has to be the
git SHA plus a dirty flag, and for a wheel the dist version IS the whole identity,
recorded as such and marked so it never renders like a commit.

Never raises. A campaign must not die because provenance could not be read, and an
*unavailable* stamp is still a record — it says the tree was not a checkout, which is
strictly better than the field being absent, which is indistinguishable from "nobody
thought to look".
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent

#: The paths inside THIS repo whose state can change a result: the package and the
#: drivers. Deliberately not the whole tree -- a run in flight rewrites its own
#: manifest under `output/` continuously, and a dirty flag that fires on results
#: files is a check firing on the wrong thing. Same scoping rule, and same reason,
#: as `run_ehn_box_vast.PAYLOAD`.
LAB_PAYLOAD = ("src", "experiments", "pyproject.toml")


def _run(cwd, *args, timeout=20):
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True,
                          timeout=timeout)


def _git_state(repo_dir, scope=(), owns=()):
    """(commit, branch, origin, dirty_files) for the repo that OWNS `owns`, or None.

    `scope` limits the dirty check to paths that matter; empty means the whole tree.
    `owns` is the ownership evidence: at least one of these paths must be TRACKED by
    the repo git finds from `repo_dir`, or this returns None. Without that test a
    wheel install inside an in-tree `.venv` is described by the enclosing repository
    with full confidence -- see this module's header.

    Untracked files are NOT special-cased away: `status --porcelain` reports them as
    `??` and they land in `dirty_files`, because a brand-new importable module that
    no SHA describes is exactly as disqualifying as an edited one.
    """
    try:
        repo_dir = Path(repo_dir)
        if not repo_dir.exists():
            return None
        commit = _run(repo_dir, "git", "rev-parse", "HEAD").stdout.strip()
        if not commit:
            return None

        # OWNERSHIP FIRST. Everything below this point would otherwise describe
        # whichever repo happens to enclose `repo_dir`.
        probes = [Path(p) for p in owns]
        probes = [p for p in probes if p.exists()]
        if probes and not any(
                _run(repo_dir, "git", "ls-files", "--error-unmatch", "--",
                     str(p)).returncode == 0 for p in probes):
            return None

        args = [str(p) for p in scope if (repo_dir / p).exists() or Path(p).exists()]
        st = (_run(repo_dir, "git", "status", "--porcelain", "--", *args) if args
              else _run(repo_dir, "git", "status", "--porcelain"))
        porcelain = st.stdout.strip()
        # ln[2:].strip(), not ln[3:]: porcelain is XY<space>PATH but the XY field is
        # space-padded, and a fixed 3-char slice ate the first character of the path
        # ("imulations/..."), which is exactly the kind of wrong-looking filename that
        # sends someone hunting a nonexistent file.
        branch = _run(repo_dir, "git", "rev-parse", "--abbrev-ref", "HEAD")
        origin = _run(repo_dir, "git", "config", "--get", "remote.origin.url")
        return {
            "commit": commit,
            "branch": branch.stdout.strip(),
            "origin": origin.stdout.strip(),
            "dirty_files": [ln[2:].strip()
                            for ln in porcelain.splitlines() if ln.strip()],
        }
    except Exception:                                          # noqa: BLE001
        return None


def install_kind(dist_name):
    """"editable" / "dist" / "unknown" -- RECORDED, never the decision.

    pip writes `direct_url.json` with `dir_info.editable` for an editable install and
    omits the file entirely for a plain wheel from an index. It is good evidence and
    it is not sufficient: a `--target` install or a vendored copy has no direct_url
    either way, so `_git_state`'s ownership test is what actually gates the stamp and
    this string only explains the answer to a reader.
    """
    try:
        import importlib.metadata as md
        txt = md.distribution(dist_name).read_text("direct_url.json")
        if not txt:
            return "dist"
        editable = json.loads(txt).get("dir_info", {}).get("editable")
        return "editable" if editable else "dist"
    except Exception:                                          # noqa: BLE001
        return "unknown"


def dist_version(dist_name):
    try:
        import importlib.metadata as md
        return md.version(dist_name)
    except Exception:                                          # noqa: BLE001
        return None


def engine_files():
    """The engine's own module files, wherever `jax_solitons` is installed from.

    Includes `knots.py`: the old hand-listed ENGINE_FILES omitted core_knot_id, which
    `knot_determinants()` genuinely imports, so the "engine hash" excluded a module the
    scoring depends on. Deriving the list from the package makes that class of omission
    impossible to write.
    """
    import jax_solitons
    root = Path(jax_solitons.__file__).resolve().parent
    return ([root / "ehn" / f"{m}.py"
             for m in ("relax", "energy", "knot_batch", "cross_linking")]
            + [root / "vortex_topology.py", root / "knots.py"])


def engine_sha(explicit_commit=None, files=None):
    """Engine provenance as a GIT COMMIT, not a content hash of source files.

    The content hash this replaced was wrong in three ways. It was INCOMPLETE (the
    hand-listed file set omitted a module the scoring imports). It was FRAGILE (one
    `ruff --fix` orphaned every certificate with no way to see the change was
    cosmetic). And it was BLIND TO UNCOMMITTED EDITS in the sense that mattered: it
    blessed a dirty tree with a definite-looking hash. A commit is auditable -- you can
    `git diff` two of them -- and dirty trees are now marked and make a certificate
    NON-CITABLE, on the rule that a certificate must never look authoritative about a
    state nobody can reconstruct.

    The FIELD NAME stays `engine_sha`, so the certificate body and C's `chamber.py`
    format are unchanged in shape. Only the derivation moved.

    TWO REPOS. The engine is `jax_solitons.ehn`, the battery is
    `ehn_lab/standard_box.py`, and they live in repositories that move independently.
    `engine_sha` names the ENGINE; the lab's own commit rides alongside as `battery`,
    because a hand-edit to the file that computes the verdict makes a certificate just
    as unreconstructible as an edited engine. Either one dirty marks the whole thing.

    Resolution order: an explicit commit (how a payload shipped FLAT to a rented box,
    where there is no git repo, carries the provenance of the tree it came from) -> git
    in the engine's OWN repo -> a marked identity for a wheel install.
    """
    files = list(engine_files() if files is None else files)
    per = {q.name: hashlib.sha256(q.read_bytes()).hexdigest()
           for q in files if q.exists()}

    lab = lab_state()
    lab_rec = None if lab is None else {
        "commit": lab["commit"], "branch": lab["branch"],
        "dirty": bool(lab["dirty_files"]), "dirty_files": lab["dirty_files"]}

    if explicit_commit is None:
        explicit_commit = os.environ.get("ENGINE_COMMIT") or None
    if explicit_commit:
        return explicit_commit[:16], {"source": "explicit/shipped",
                                      "commit": explicit_commit, "dirty": None,
                                      "install": install_kind("jax-solitons"),
                                      "content_hashes": per, "battery": lab_rec}

    root = files[0].parent.parent if files else None
    eng = _git_state(root, owns=files, scope=files) if root else None
    if eng:
        dirty = bool(eng["dirty_files"]) or bool(lab and lab["dirty_files"])
        sha = eng["commit"][:16] + ("-dirty" if dirty else "")
        return sha, {"source": "git", "commit": eng["commit"], "branch": eng["branch"],
                     "origin": eng["origin"], "dirty": dirty,
                     "install": install_kind("jax-solitons"),
                     "dirty_files": eng["dirty_files"],
                     "content_hashes": per, "battery": lab_rec}

    # No repo OWNS these files: a wheel install, a vendored copy, or a tree that is
    # not a checkout. `dist_version` is then the complete identity and is rendered so
    # it can never be misread as a commit -- `dist0.0.8:ab12cd34ef` has a prefix a
    # 40-hex SHA never has. The content hashes still ride along for lineage.
    dist = dist_version("jax-solitons")
    h = hashlib.sha256()
    for k in sorted(per):
        h.update(per[k].encode())
    tag = f"dist{dist}:" if dist else "nogit:"
    return tag + h.hexdigest()[:10], {
        "source": f"dist identity (jax-solitons {dist or 'unknown'}, "
                  f"no owning git tree)",
        "commit": None, "dirty": None, "install": install_kind("jax-solitons"),
        "dist_version": dist, "content_hashes": per, "battery": lab_rec}


def lab_state():
    """This repo's own state, scoped to LAB_PAYLOAD, with the same ownership test.

    Run from the repo ROOT, not from this file's directory: LAB_PAYLOAD is written
    repo-relative (`src`, `experiments`, `pyproject.toml`) and a scope path that does
    not resolve is silently dropped, which would quietly widen the dirty check to the
    whole tree and let `output/` churn from a run in flight mark the lab dirty.

    The ownership probe is a LIST and the test is `any`, deliberately: a brand-new
    module is untracked by definition, and disqualifying its own repo on that basis
    would mean this file could never stamp anything on the commit that introduces it.
    One tracked sibling is sufficient evidence of which repo we are standing in; the
    new file still shows up as `??` in `dirty_files`, which is the correct report.
    """
    root = _HERE.parent.parent
    return _git_state(root, scope=LAB_PAYLOAD,
                      owns=[Path(__file__).resolve(), _HERE / "__init__.py",
                            _HERE / "gpe_lab.py", root / "pyproject.toml"])


@functools.lru_cache(maxsize=1)
def code_provenance():
    """The block every census summary carries beside `gpe_lab.provenance()`.

    Flat and short on purpose: it goes into `summary.json` next to the measured
    quantities so the tree is checkable after the fact, rather than reconstructible
    only while the session that ran it is still alive.

    CACHED for the process: the tree cannot change under a run that has already
    started, and a cache is what makes every stamp a run emits — the summary's and
    each event-graph particle's — provably the same answer rather than several
    reads that merely ought to agree. It also keeps ~5 `git` subprocesses off the
    per-particle path.

    `jax` is in here because the census runs the engine's steppers and nothing else
    records the solver. The farm legs pin it (`run_ehn_box_vast.PIP_JAX`); local runs
    take whatever the venv resolves, so the version has to be recorded even though it
    is not controlled.
    """
    sha, detail = engine_sha()
    lab = detail.get("battery")
    return {
        "engine_sha": sha,
        "engine_source": detail["source"],
        "engine_install": detail.get("install"),
        "engine_dirty": bool(detail.get("dirty")),
        "engine_dirty_files": detail.get("dirty_files") or [],
        "lab_commit": (lab or {}).get("commit"),
        "lab_dirty": bool((lab or {}).get("dirty")),
        "lab_dirty_files": (lab or {}).get("dirty_files") or [],
        "jax": dist_version("jax"),
        "jaxlib": dist_version("jaxlib"),
    }
