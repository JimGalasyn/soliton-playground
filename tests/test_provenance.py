"""The enclosing-repo trap, manufactured locally instead of waited for.

⚠ **The defect this file exists for could not be observed in this repo's own
configuration.** Both trees soliton-playground depends on (`jax-solitons`, `run-farm`)
are installed editable, so the wheel-install branch of `engine_sha()` never executes
here, and no amount of re-reading or re-running finds it. The same defect survived nine
days in `abiogenesis` for exactly the same reason, and was found only when the module
was ported to a repo whose install mix supplied the test case — where the first stamp
reported a dependency wearing the WRONG repository's SHA. **Transfer is the test.**

⚠⚠ **A control that skips itself in the configuration that hides the bug is not a
control.** The obvious shape here — "if nothing is installed as a wheel, skip" — passes
its way through every run in this repo and delivers ZERO executing coverage of the
branch it was written for. A skip and a pass are both "not failing" in the summary
line. So these tests manufacture the missing condition from **numpy**, a real dist
install that is already in this venv, and FIRST assert the trap is LIVE — that git run
from numpy's directory really does return this repo's HEAD — so they cannot pass
quietly on a machine where `.venv` sits outside the working tree.

Proven able to fail: deleting the ownership block in `_git_state` makes
`test_ownership_guard_refuses_the_enclosing_repo` report a site-packages install
stamped with this repo's own commit.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy
import pytest

from soliton_playground import provenance
from soliton_playground.provenance import (_git_state, _owns, code_provenance,
                                           engine_sha, engine_files,
                                           install_kind)

REPO = Path(__file__).resolve().parents[1]
NUMPY_DIR = Path(numpy.__file__).resolve().parent


def _git(cwd, *args):
    return subprocess.run(("git",) + args, cwd=str(cwd), capture_output=True, text=True)


def _numpy_is_inside_this_worktree():
    """Is the positive control's condition actually present on this machine?

    The trap needs a non-editable install sitting UNDER a git worktree. That is the
    normal layout here (`.venv` at the repo root) and it is the only reason these
    tests can run at home rather than waiting for a port.
    """
    return REPO in NUMPY_DIR.parents and install_kind("numpy") == "dist"


requires_trap = pytest.mark.skipif(
    not _numpy_is_inside_this_worktree(),
    reason="no dist install under this worktree — .venv is outside the repo, so the "
           "enclosing-repo trap cannot be manufactured here. THIS IS THE SELF-SKIP "
           "the module header warns about: it means this machine is not testing the "
           "branch, not that the branch is fine.")


@requires_trap
def test_the_enclosing_repo_trap_is_live():
    """Assert the hazard exists before asserting the guard handles it.

    Without this, the guard test could pass because git said nothing useful rather
    than because the guard did its job.
    """
    from_numpy = _git(NUMPY_DIR, "rev-parse", "HEAD").stdout.strip()
    ours = _git(REPO, "rev-parse", "HEAD").stdout.strip()
    assert from_numpy == ours != "", (
        "expected raw git from a site-packages directory to walk UP and describe this "
        f"repo; got {from_numpy!r} vs {ours!r}")

    # The other half of why a wrong identity is worse than a missing one: scoped to a
    # path git does not track, `status --porcelain` is EMPTY, so the wrong SHA would
    # also have been reported CLEAN.
    scoped = _git(NUMPY_DIR, "status", "--porcelain", "--",
                  str(NUMPY_DIR / "__init__.py")).stdout.strip()
    assert scoped == "", (
        f"expected an empty (falsely clean) dirty report, got {scoped!r}")


@requires_trap
def test_ownership_guard_refuses_the_enclosing_repo():
    """`_git_state` must return None rather than a confident wrong identity."""
    state = _git_state(NUMPY_DIR, owns=[NUMPY_DIR / "__init__.py"])
    assert state is None, (
        f"a site-packages install was stamped {state and state['commit'][:8]!r}, which "
        f"is THIS repo's SHA — the ownership test in _git_state is not running")


@requires_trap
def test_engine_sha_on_an_unowned_tree_never_renders_like_a_commit(monkeypatch):
    """The dist arm's identity must be unmistakable, not merely different.

    `dist0.0.8:ab12cd34ef` carries a prefix no 40-hex SHA has, so a reader who sees it
    in a manifest cannot take it for a commit and go looking for a `git show`.
    """
    monkeypatch.delenv("ENGINE_COMMIT", raising=False)
    sha, detail = engine_sha(files=[NUMPY_DIR / "__init__.py"])
    ours = _git(REPO, "rev-parse", "HEAD").stdout.strip()

    assert detail["commit"] is None, detail
    assert detail["source"].startswith("dist identity"), detail["source"]
    assert not sha.startswith(ours[:12]), f"stamp {sha!r} is wearing this repo's SHA"
    assert sha.startswith(("dist", "nogit:")), sha


def test_the_editable_engine_is_still_git_stamped():
    """The guard must not have bought its safety by refusing the normal case."""
    if install_kind("jax-solitons") != "editable":
        pytest.skip("jax-solitons is not installed editable in this venv")
    sha, detail = engine_sha()
    assert detail["source"] == "git", detail
    assert detail["commit"] and len(detail["commit"]) == 40, detail
    assert sha.startswith(detail["commit"][:16])
    # The engine files really do live in the tree that got stamped.
    assert any(f.exists() for f in engine_files())


def test_census_summaries_carry_code_provenance():
    """The gap this whole change closes: a census summary that names only its medium.

    `gpe_lab.provenance()` said preset/model/protecting_charge and nothing about which
    tree computed the number, while jax-solitons was a live sibling checkout.
    """
    from soliton_playground.gpe_lab import CHARGE_NONE, provenance, zoo_provenance

    block = provenance(CHARGE_NONE)
    assert block["preset"] and block["model"]          # the medium, unchanged
    code = block["code"]
    assert code["engine_sha"], code
    assert code["engine_source"], code
    assert code["jax"], "the solver version is not recorded"
    assert set(code) >= {"engine_sha", "engine_dirty", "lab_commit", "lab_dirty", "jax"}

    flat = zoo_provenance(CHARGE_NONE)
    assert flat["zoo.engine_sha"] == code["engine_sha"]
    assert all(not isinstance(v, (dict, list)) for v in flat.values()), (
        "event-graph particle attrs must stay scalar")


def test_code_provenance_is_one_answer_per_process():
    """Cached, so a run's summary and its event-graph particles cannot disagree."""
    assert code_provenance() is code_provenance()


# ---------------------------------------------------------------------------
# THE TESTS WE DID NOT HAVE, AND WHICH ANOTHER REPO'S SUITE FOUND FOR US.
#
# The first cut of the ownership guard returned a bool and caught every exception
# as "not owned". abiogenesis-15 adopted it and it immediately broke an EXISTING
# test of theirs — one that monkeypatches git into raising and asserts the stamp
# records its unavailability. We had no such test, which is exactly why we shipped
# the bug: a legitimate editable tree on a box with no git was silently downgraded
# to a dist install and had its real SHA dropped. Same confident-wrong-identity
# failure the module exists to prevent, reintroduced by its own fix.
#
# Symmetry worth keeping in view: our transfer found their defect, their suite
# found ours. The receiving repo's EXISTING tests are the instrument, not the new
# ones that ship with a fix.
# ---------------------------------------------------------------------------


def test_git_exit_codes_are_actually_tri_state():
    """The guard's premise, asserted rather than assumed.

    If git ever collapsed 1 and 128 into one code, `_owns` could not distinguish
    "not tracked" from "cannot ask" and the tri-state would be decoration.
    """
    tracked = _git(REPO, "ls-files", "--error-unmatch", "--", "README.md")
    assert tracked.returncode == 0

    untracked_inside = _git(REPO, "ls-files", "--error-unmatch", "--",
                            ".venv/nonexistent-probe.py")
    assert untracked_inside.returncode == 1, untracked_inside.stderr

    with tempfile.TemporaryDirectory() as td:          # not a repository at all
        cannot_ask = _git(td, "ls-files", "--error-unmatch", "--", "x.py")
        assert cannot_ask.returncode == 128, cannot_ask.stderr


def test_owns_reports_could_not_ask_as_None_not_False():
    """None and False are different answers and must not be merged."""
    with tempfile.TemporaryDirectory() as td:
        probe = Path(td) / "thing.py"
        probe.write_text("x = 1\n")
        assert _owns(td, [probe]) is None, "rc=128 must not read as 'not owned'"

    assert _owns(REPO, [REPO / "README.md"]) is True
    if _numpy_is_inside_this_worktree():
        assert _owns(REPO, [NUMPY_DIR / "__init__.py"]) is False


def test_a_tree_that_cannot_be_read_records_why_and_claims_nothing(monkeypatch):
    """Git missing entirely: the stamp must say UNAVAILABLE, not 'no owning tree'.

    This is the one that would have caught the bool guard. Note what it asserts is
    NOT merely 'something was recorded' — it is that the stamp does not make the
    POSITIVE claim that no git tree owns the code, which on a box without git is
    simply false.
    """
    def boom(*a, **k):
        raise FileNotFoundError("git: command not found")

    monkeypatch.setattr(provenance.subprocess, "run", boom)
    monkeypatch.delenv("ENGINE_COMMIT", raising=False)
    sha, detail = provenance.engine_sha()

    assert detail["commit"] is None
    assert "unavailable" in detail, detail
    assert "git" in detail["unavailable"].lower() or \
           "filenotfound" in detail["unavailable"].lower(), detail["unavailable"]
    assert detail["source"].startswith("UNAVAILABLE"), detail["source"]
    assert "no owning git tree" not in detail["source"], (
        "the stamp asserts there is no owning tree, when the truth is that it "
        "could not ask — this is the bool-guard defect")
    assert not sha.startswith("dist"), sha


def test_code_provenance_survives_an_unreadable_tree(monkeypatch):
    """A campaign must not die because provenance could not be read."""
    def boom(*a, **k):
        raise FileNotFoundError("git: command not found")

    provenance.code_provenance.cache_clear()
    monkeypatch.setattr(provenance.subprocess, "run", boom)
    monkeypatch.delenv("ENGINE_COMMIT", raising=False)
    try:
        block = provenance.code_provenance()
        assert block["lab_commit"] is None
        assert block["engine_source"].startswith("UNAVAILABLE")
        assert block["jax"], "the solver version does not come from git"
    finally:
        provenance.code_provenance.cache_clear()
