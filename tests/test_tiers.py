"""Guards on the test tiering itself.

The tiering lives in `tests/conftest.py` and is keyed by module name. That is convenient, but it
fails silently in the worst possible way: a new test file that nobody adds to `TIERS` would be
absent from every `-m fast` and `-m "fast or medium"` run, so CI would go green without ever
executing it. These tests make that impossible.
"""
from __future__ import annotations

import glob
import os
import pathlib

import pytest

from tests.conftest import KNOWN_MARKERS, TIERS, ntop_available

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
VALID_TIERS = {"fast", "medium", "slow"}


def _modules_on_disk() -> set[str]:
    return {
        os.path.basename(p)[:-3]
        for p in glob.glob(os.path.join(TESTS_DIR, "test_*.py"))
    }


def test_tiering_covers_every_module():
    """Every test module must be tiered, or CI can skip it without anyone noticing."""
    missing = sorted(_modules_on_disk() - set(TIERS))
    assert not missing, (
        "these test modules are not in tests/conftest.py TIERS, so they would be excluded "
        f"from -m fast and -m 'fast or medium': {missing}. Add them."
    )


def test_tiers_do_not_name_modules_that_no_longer_exist():
    """A stale entry is harmless at runtime but misleads anyone reading the tiering."""
    stale = sorted(set(TIERS) - _modules_on_disk())
    assert not stale, f"TIERS names modules that do not exist: {stale}"


def test_every_tier_is_a_known_name():
    bad = {m: t for m, t in TIERS.items() if t not in VALID_TIERS}
    assert not bad, f"unknown tier names: {bad}. Valid tiers are {sorted(VALID_TIERS)}."


def test_every_tier_is_populated():
    """If a tier empties out, the workflow job for it would pass by doing nothing."""
    for tier in VALID_TIERS:
        assert any(t == tier for t in TIERS.values()), f"tier {tier!r} has no modules"


def test_this_test_is_marked_fast(request):
    """Self-check that the conftest hook actually applied a marker."""
    assert request.node.get_closest_marker("fast") is not None


def test_ntop_availability_reports_a_reason_when_unavailable():
    ok, reason = ntop_available()
    if ok:
        assert reason == ""
    else:
        assert reason, "an unavailable nTop must say which piece is missing"


@pytest.mark.parametrize("module", sorted(m for m, t in TIERS.items() if t == "slow"))
def test_slow_modules_are_the_ntop_ones(module):
    """The slow tier exists because of `ntopcl`, not because of arithmetic.

    If a module is slow for some other reason it belongs in medium with a note, so that the
    'slow cannot run on hosted CI' statement stays true for the whole tier.
    """
    path = os.path.join(TESTS_DIR, f"{module}.py")
    if not os.path.isfile(path):
        pytest.skip(f"{module} not on disk")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert (
        "ntopgen" in text or "NtopRunner" in text or "measure_" in text
    ), f"{module} is tiered slow but does not appear to drive nTop"


# --------------------------------------------------------------------------------------
#   The guards that stop a tier from passing without running anything
# --------------------------------------------------------------------------------------


def test_every_tier_name_is_a_known_marker():
    """`-m <tier>` has to resolve, or the tier cannot be selected at all."""
    assert set(TIERS.values()) <= set(KNOWN_MARKERS)


def test_bad_mark_expression_is_rejected(pytester):
    """A typo in -m must fail, not deselect everything and exit 0.

    This is the failure mode the guard exists for: `pytest -m fats` collects every test, deselects
    all of them, and exits 0, which in CI is indistinguishable from a green run.
    """
    pytester.makeconftest(
        (pathlib.Path(__file__).parent / "conftest.py").read_text(encoding="utf-8")
    )
    probe = "def test_ok():\n    assert True\n"
    pytester.makepyfile(test_probe=probe)
    result = pytester.runpytest("-m", "not_a_real_tier")
    assert result.ret != 0, "a bad -m expression must not exit 0"
    result.stderr.fnmatch_lines(["*unknown marker*"])


def test_ntop_skip_env_forces_unavailable(monkeypatch):
    """CI relies on this to exercise the skip path on a machine that does have nTop."""
    monkeypatch.setenv("NTOP_SKIP", "1")
    ok, reason = ntop_available()
    assert not ok
    assert "NTOP_SKIP" in reason


@pytest.mark.parametrize("value", ["", "0", "false", "False"])
def test_ntop_skip_falsey_values_do_not_force_unavailable(monkeypatch, value):
    """An empty or false-y NTOP_SKIP must not accidentally disable the whole geometry tier."""
    monkeypatch.setenv("NTOP_SKIP", value)
    _ok, reason = ntop_available()
    assert "NTOP_SKIP" not in reason
