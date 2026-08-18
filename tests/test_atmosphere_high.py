"""Validation of the atmosphere above 30 km, needed for the lofted IV-1 intercept arc.

Why this file exists: the cached table used to stop at 30 km, and above the ceiling `atmo()` clamps.
A lofted two-stage intercept apogees between 45 and 54 km, so every point above the old ceiling was
being flown at 30 km density, which overstates drag. The ceiling is now 86 km, the upper limit of
the US Standard 1976 lower atmosphere.

The reference values below are the layer base conditions printed in the standard. They are
tabulated against **geopotential** altitude. SUAVE takes **geometric** altitude and converts
internally. Comparing a geometric query against a geopotential table row is an error of about
4 percent in pressure at 47 km, and it is the comparison that is wrong, not SUAVE. Every case here
converts properly:

    H = r0 * z / (r0 + z)        geometric z to geopotential H
    z = r0 * H / (r0 - H)        geopotential H to geometric z

with r0 = 6356766 m, the effective earth radius the standard specifies.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from rocketgen.sizing.atmosphere import (
    H_MAX,
    H_MAX_LEGACY,
    H_MIN,
    AtmoState,
    atmo,
    prime,
    table,
)

# Effective earth radius used by US Standard 1976 for the geopotential conversion, m.
R0 = 6_356_766.0

#: Layer base conditions from NASA-TM-X-74335, U.S. Standard Atmosphere 1976, Table I.
#: (geopotential altitude H [m], temperature [K], pressure [Pa])
US1976_LAYER_BASES = [
    (0.0, 288.150, 101_325.0),
    (11_000.0, 216.650, 22_632.1),
    (20_000.0, 216.650, 5_474.89),
    (32_000.0, 228.650, 868.019),
    (47_000.0, 270.650, 110.906),
    (51_000.0, 270.650, 66.9389),
    (71_000.0, 214.650, 3.95642),
]


def geometric_from_geopotential(H: float) -> float:
    """Geometric altitude z for a geopotential altitude H, both in metres."""
    return R0 * H / (R0 - H)


def geopotential_from_geometric(z: float) -> float:
    return R0 * z / (R0 + z)


# --------------------------------------------------------------------------------------
#   The extension itself
# --------------------------------------------------------------------------------------


def test_the_table_now_reaches_86_km():
    """The old 30 km ceiling silently overstated drag on any lofted arc."""
    assert H_MAX == pytest.approx(86_000.0)
    assert H_MAX > H_MAX_LEGACY
    tb = table()
    assert tb["altitude"][0] == pytest.approx(H_MIN)
    assert tb["altitude"][-1] == pytest.approx(H_MAX)


def test_extending_the_ceiling_did_not_disturb_the_region_below_it():
    """Every quantity below the old ceiling must be unchanged by the extension.

    The SV-1 results were produced with the 30 km table, so a change here would invalidate them.
    """
    for h in (0.0, 500.0, 5_000.0, 10_000.0, 12_000.0, 20_000.0, 29_999.0):
        s = atmo(h)
        direct = _suave_direct(h)
        assert s.pressure == pytest.approx(direct["pressure"], rel=2e-6), h
        assert s.temperature == pytest.approx(direct["temperature"], rel=2e-6), h
        assert s.density == pytest.approx(direct["density"], rel=2e-6), h


def _suave_direct(h: float) -> dict[str, float]:
    """Evaluate SUAVE directly, bypassing the table, for a single altitude."""
    from rocketgen.config import add_suave_to_path

    add_suave_to_path()
    from SUAVE.Analyses.Atmospheric import US_Standard_1976

    v = US_Standard_1976().compute_values(np.array([float(h)]))
    return {k: float(np.ravel(v[k])[0]) for k in ("pressure", "temperature", "density")}


# --------------------------------------------------------------------------------------
#   Agreement with the published standard
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("H,T_ref,p_ref", US1976_LAYER_BASES)
def test_layer_base_conditions_match_the_published_standard(H, T_ref, p_ref):
    """At every layer break, query the GEOMETRIC altitude that corresponds to the printed
    GEOPOTENTIAL row, and require agreement to 0.5 percent.

    0.5 percent is the tolerance, not a discovered value: the standard prints six significant
    figures and the interpolation table has a 10 m step, so anything looser would be hiding a real
    disagreement and anything tighter would be measuring the table step.
    """
    z = geometric_from_geopotential(H)
    if z > H_MAX:
        pytest.skip(f"geopotential {H/1e3:.1f} km is above the table ceiling")
    s = atmo(z)
    assert s.temperature == pytest.approx(T_ref, rel=5e-3), f"T at H={H/1e3:.1f} km"
    assert s.pressure == pytest.approx(p_ref, rel=5e-3), f"p at H={H/1e3:.1f} km"


def test_the_geopotential_correction_is_what_reconciles_the_47_km_row():
    """Guard the reasoning in this file's docstring, so nobody 'fixes' a non-error later.

    Querying 47 km GEOMETRIC and comparing against the printed 47 km GEOPOTENTIAL row looks like a
    4 percent pressure error. Doing the conversion removes it. If a future change made the naive
    comparison pass, something would be wrong.
    """
    p_row = 110.906
    naive = atmo(47_000.0).pressure
    corrected = atmo(geometric_from_geopotential(47_000.0)).pressure

    assert corrected == pytest.approx(p_row, rel=5e-3)
    # the naive comparison should be off by several percent, in the direction of higher pressure
    assert naive > p_row
    assert 0.02 < (naive - p_row) / p_row < 0.10


def test_density_falls_monotonically_all_the_way_up():
    h = np.arange(0.0, H_MAX + 1.0, 250.0)
    rho = atmo(h).density
    assert np.all(np.diff(rho) < 0.0)


def test_temperature_reproduces_the_stratopause_maximum():
    """US Standard 1976 is isothermal at 270.65 K between 47 and 51 km geopotential.

    A model that missed the stratopause would put the speed of sound badly wrong exactly where a
    lofted intercept spends its midcourse.
    """
    z_lo = geometric_from_geopotential(47_500.0)
    z_hi = geometric_from_geopotential(50_500.0)
    assert atmo(z_lo).temperature == pytest.approx(270.65, rel=5e-3)
    assert atmo(z_hi).temperature == pytest.approx(270.65, rel=5e-3)
    # and it is a local maximum: cooler on both sides
    assert atmo(geometric_from_geopotential(40_000.0)).temperature < 270.0
    assert atmo(geometric_from_geopotential(60_000.0)).temperature < 270.0


def test_speed_of_sound_follows_the_temperature():
    """a = sqrt(gamma R T), so the ratio must be constant everywhere."""
    h = np.arange(0.0, H_MAX + 1.0, 1000.0)
    s = atmo(h)
    ratio = s.speed_of_sound / np.sqrt(s.temperature)
    assert np.ptp(ratio) / np.mean(ratio) < 1e-6


# --------------------------------------------------------------------------------------
#   Behaviour that the trajectory relies on
# --------------------------------------------------------------------------------------


def test_the_lofted_intercept_band_is_inside_the_table():
    """Measured IV-1 apogees are 45 to 54 km. All of it must be interpolated, not clamped."""
    for h in (45_000.0, 50_000.0, 54_000.0, 60_000.0):
        assert h < H_MAX
        s = atmo(h)
        assert s.density > 0.0
        assert s.pressure > 0.0


def test_drag_above_the_old_ceiling_is_now_far_smaller():
    """Quantify what the old clamp was doing, so the fix is justified by a number.

    At 50 km the true density is a small fraction of the 30 km value the clamp was holding, so the
    old table inflated drag there by more than an order of magnitude.
    """
    rho_50 = atmo(50_000.0).density
    rho_clamped = atmo(H_MAX_LEGACY).density
    assert rho_50 < rho_clamped
    assert rho_clamped / rho_50 > 10.0


def test_clamping_still_happens_above_the_new_ceiling_and_is_not_extrapolated():
    """Above 86 km the standard itself stops. Clamping is correct; extrapolating would not be."""
    top = atmo(H_MAX)
    above = atmo(H_MAX + 20_000.0)
    assert above.density == pytest.approx(top.density, rel=1e-12)
    assert above.pressure == pytest.approx(top.pressure, rel=1e-12)


def test_array_and_scalar_paths_agree_across_the_whole_range():
    h = np.array([0.0, 1_000.0, 15_000.0, 35_000.0, 50_000.0, 70_000.0, 85_000.0])
    arr = atmo(h)
    for i, hi in enumerate(h):
        s = atmo(float(hi))
        assert arr.density[i] == pytest.approx(s.density, rel=1e-12)
        assert arr.pressure[i] == pytest.approx(s.pressure, rel=1e-12)


def test_interpolation_still_matches_direct_suave_high_up():
    """The table is an interpolation. Prove it is faithful where IV-1 actually flies."""
    worst = 0.0
    for h in (31_000.0, 40_000.0, 47_000.0, 55_000.0, 65_000.0, 80_000.0):
        s = atmo(h)
        d = _suave_direct(h)
        for field in ("pressure", "temperature", "density"):
            rel = abs(getattr(s, field) - d[field]) / d[field]
            worst = max(worst, rel)
    assert worst < 1e-4, f"worst interpolation error {worst:.2e}"


def test_state_is_returned_for_scalars_as_floats():
    s = atmo(40_000.0)
    assert isinstance(s, AtmoState)
    assert isinstance(float(s.density), float)
    assert math.isfinite(s.kinematic_viscosity)


def test_index_arithmetic_is_bit_identical_to_np_interp():
    """`atmo` computes the bracketing index by arithmetic instead of searching.

    That is only legitimate because the grid is uniform. This test is the guard: if anyone makes
    the grid non-uniform, or gets the clamping wrong, the fast path stops agreeing with the
    reference interpolation and this fails.
    """
    from rocketgen.sizing.atmosphere import _FIELDS, H_STEP

    tb = table()
    grid = tb["altitude"]
    assert np.allclose(np.diff(grid), H_STEP, rtol=0, atol=1e-9), "grid is no longer uniform"

    probes = np.concatenate(
        [
            np.linspace(H_MIN, H_MAX, 2001),
            np.array([-500.0, H_MAX + 500.0, 12_345.6, 47_000.0, H_STEP * 0.5]),
        ]
    )
    worst = 0.0
    for h in probes:
        s = atmo(float(h))
        h_clamped = min(max(float(h), H_MIN), H_MAX)
        for name in _FIELDS:
            ref = float(np.interp(h_clamped, grid, tb[name]))
            got = getattr(s, name)
            worst = max(worst, abs(got - ref) / max(abs(ref), 1e-300))
    assert worst < 1e-12, f"fast path deviates from np.interp by {worst:.2e}"


def test_the_lookup_is_fast_enough_for_the_trajectory_integrator():
    """The integrator calls this tens of thousands of times per trajectory.

    Raising the ceiling from 30 km to 86 km tripled the grid. With a binary search per field per
    call that cost showed up directly in the trajectory time budget, which is why the lookup uses
    index arithmetic. 20 us per call is a generous ceiling that still catches a regression to
    searching.
    """
    import time

    prime()
    n = 20_000
    start = time.perf_counter()
    for i in range(n):
        atmo(1_000.0 + (i % 60_000))
    per_call = (time.perf_counter() - start) / n
    assert per_call < 20e-6, f"{per_call*1e6:.1f} us per call"
