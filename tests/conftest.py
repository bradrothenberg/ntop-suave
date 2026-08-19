"""Test tiering, and the guard that keeps the nTop tests from failing a machine without nTop.

Three tiers, assigned by module so no test file has to carry a marker of its own. Measured on a
developer workstation, so treat the times as relative rather than absolute:

| tier | tests | wall clock | needs |
|---|---|---|---|
| `fast` | 297 | about 11 s | SUAVE only |
| `medium` | 267 | about 85 s | SUAVE only |
| `slow` | 126 | about 11 min | SUAVE, a licensed `ntopcl`, and the nTop block universe |

    pytest -m fast                 the inner loop: run this constantly
    pytest -m "fast or medium"     everything that runs without nTop. 563 tests, about 86 s
    pytest -m slow                 the nTop subprocess tests
    pytest                         all of it

The split is not arbitrary. `fast` is closed-form and analytic: mass build-ups, aerodynamic
coefficient evaluation, atmosphere lookups. `medium` integrates trajectories and runs trade
studies, so it is dominated by many cheap steps rather than by any one slow test; the slowest
single test in either tier is 5.65 s. `slow` is every test that spawns a real `ntopcl` process,
and it is slow because conversion evaluates the notebook and measurement queries implicit fields.

**The `slow` tier cannot run on a hosted CI runner.** `ntopcl` is licensed, and the block universe
it needs is nTop's property and is not redistributable, so `vendor/functions.json` is gitignored.
Those tests are therefore skipped automatically when either is absent, with a reason that says
which one was missing. That is a skip and not a pass: a green hosted CI run has NOT exercised the
geometry path, and the workflow says so in its summary.
"""
from __future__ import annotations

import os
import re
import shutil

import pytest

# Module stem -> tier. Every test module must appear here; `test_tiering_covers_every_module`
# in tests/test_tiers.py asserts that, so a new file cannot land untiered and silently
# escape the fast and medium runs.
TIERS: dict[str, str] = {
    # fast: closed form, analytic, no trajectory integration
    "test_masses": "fast",
    "test_aero": "fast",
    "test_aero_iv1": "fast",
    "test_atmosphere_high": "fast",
    "test_oml_spline": "fast",
    # medium: trajectory integration, motor closure, trade studies
    "test_propulsion": "medium",
    "test_propulsion_iv1": "medium",
    "test_trajectory": "medium",
    "test_trajectory_iv1": "medium",
    "test_doe": "medium",
    "test_wavedrag": "medium",
    # slow: spawns a real ntopcl process
    "test_ntopgen": "slow",
    "test_rocket_notebook": "slow",
    "test_stack_notebook": "slow",
    "test_spline_notebook": "slow",
    # this file's own guard tests
    "test_tiers": "fast",
}


# `pytester` lets tests/test_tiers.py run pytest in a sandbox to prove that a bad -m expression
# actually fails. Without enabling the plugin the fixture does not exist.
pytest_plugins = ("pytester",)

KNOWN_MARKERS = frozenset({"fast", "medium", "slow", "ntop"})


def _ntopcl_path() -> str | None:
    """Resolve ntopcl the same way rocketgen.config does, without importing it."""
    for env in ("NTOPCL", "NTOPCL_FALLBACK"):
        p = os.environ.get(env)
        if p and os.path.isfile(p):
            return p
    default = r"C:/Program Files/nTopology/nTopology/ntopcl.exe"
    if os.path.isfile(default):
        return default
    return shutil.which("ntopcl")


def _universe_present() -> bool:
    here = os.path.dirname(os.path.abspath(__file__))
    vendor = os.path.join(os.path.dirname(here), "vendor")
    return all(
        os.path.isfile(os.path.join(vendor, n))
        for n in ("functions.json", "types.json", "type_defaults.json")
    )


def ntop_available() -> tuple[bool, str]:
    """Whether the `slow` tier can run here, and why not when it cannot.

    `NTOP_SKIP=1` forces unavailable. That exists because the resolution order falls back to the
    standard install path, so on a developer machine you cannot simulate absence just by pointing
    NTOPCL at nothing. CI uses it to assert the skip path still works.
    """
    if os.environ.get("NTOP_SKIP", "").strip() not in ("", "0", "false", "False"):
        return False, "NTOP_SKIP is set"
    if _ntopcl_path() is None:
        return False, "ntopcl not found; set NTOPCL or install nTop Automate"
    if not _universe_present():
        return False, (
            "the nTop block universe is missing from vendor/; run scripts/bootstrap.py "
            "or set NTOP_UNIVERSE_DIR"
        )
    return True, ""


def pytest_configure(config: pytest.Config) -> None:
    for tier, why in (
        ("fast", "closed-form and analytic; the inner loop"),
        ("medium", "integrates trajectories and runs trade studies"),
        ("slow", "spawns a real ntopcl process; cannot run on a hosted CI runner"),
        ("ntop", "requires a licensed ntopcl and the nTop block universe"),
    ):
        config.addinivalue_line("markers", f"{tier}: {why}")


def _check_mark_expression(config: pytest.Config) -> None:
    """Fail on a typo in `-m` instead of silently running nothing.

    `--strict-markers` does not do this: it only rejects markers used in code that are not
    declared. A bad `-m` expression deselects every test and exits 0, so `pytest -m fats` would
    report success having executed nothing. In CI that is indistinguishable from a green run,
    which is the worst possible failure mode for a test gate.
    """
    expr = (getattr(config.option, "markexpr", "") or "").strip()
    if not expr:
        return
    names = set(re.findall(r"[A-Za-z_]\w*", expr)) - {"and", "or", "not"}
    unknown = sorted(names - KNOWN_MARKERS)
    if unknown:
        raise pytest.UsageError(
            f"unknown marker(s) in -m {expr!r}: {unknown}. "
            f"Known markers are {sorted(KNOWN_MARKERS)}."
        )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    _check_mark_expression(config)

    ok, reason = ntop_available()
    skip_ntop = pytest.mark.skip(reason=f"nTop unavailable: {reason}")

    for item in items:
        stem = item.path.stem if hasattr(item, "path") else item.fspath.purebasename
        tier = TIERS.get(stem)
        if tier is None:
            # Untiered module. Do not silently guess: mark it slow so it cannot sneak into a
            # fast run, and let tests/test_tiers.py report it as a real failure.
            item.add_marker(pytest.mark.slow)
            continue
        item.add_marker(getattr(pytest.mark, tier))
        if tier == "slow":
            item.add_marker(pytest.mark.ntop)
            if not ok:
                item.add_marker(skip_ntop)


def pytest_collection_finish(session: pytest.Session) -> None:
    """A tier that selects nothing is a bug, not a pass.

    Guards against a tier being emptied by a rename or a bad filter. Without this, CI would go
    green having run zero tests.
    """
    expr = (getattr(session.config.option, "markexpr", "") or "").strip()
    if expr and not session.items:
        raise pytest.UsageError(
            f"-m {expr!r} selected 0 tests. A tier that selects nothing cannot pass; "
            f"check tests/conftest.py TIERS."
        )


def pytest_report_header(config: pytest.Config) -> list[str]:
    ok, reason = ntop_available()
    state = "available" if ok else f"UNAVAILABLE ({reason})"
    return [
        f"tiers: fast / medium / slow  (see tests/conftest.py)",
        f"nTop:  {state}",
    ]
