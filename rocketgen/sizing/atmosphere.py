"""Fast, cached US Standard Atmosphere 1976 for the sizing loop.

Why this module exists
----------------------
SUAVE's ``Analyses.Atmospheric.US_Standard_1976`` is the authority for the atmosphere
(SPEC.md section 5), but a single ``compute_values`` call costs of order 10 ms because it
rebuilds a ``Conditions`` data structure every time. The 3-DOF trajectory integrator of WP3
evaluates the atmosphere tens of thousands of times, so a direct SUAVE call per step is far
too slow.

This module therefore calls SUAVE **once**, on a fine altitude grid from 0 to 30 km, and then
serves every later request by linear interpolation of that table with ``numpy.interp``. The
grid step is 10 m, and the US 1976 layer breakpoints inside the range (11 000 m and 20 000 m)
fall exactly on grid nodes, so no interpolation ever straddles a temperature-gradient kink.
``tests/test_aero.py`` proves the interpolated values match direct SUAVE evaluation to better
than 0.1 percent.

Everything is SI: metre, kelvin, pascal, kg/m^3, m/s, Pa.s.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import add_suave_to_path, register_sources

# --------------------------------------------------------------------------------------
#   Sources
# --------------------------------------------------------------------------------------

SOURCES: dict[str, str] = {
    "atmo_model": (
        "U.S. Standard Atmosphere 1976 (NOAA/NASA/USAF, NASA-TM-X-74335), evaluated through "
        "SUAVE 2.5.2 Analyses.Atmospheric.US_Standard_1976"
    ),
    "atmo_table": (
        "No empirical constant. Table built by direct evaluation of the SUAVE US 1976 model on "
        "a uniform 10 m altitude grid from 0 to 30 km; served by piecewise-linear interpolation "
        "(numpy.interp). The 11 km and 20 km layer breakpoints lie on grid nodes."
    ),
    "atmo_sutherland_S": (
        "Sutherland constant S = 110.4 K for air, used only to scale viscosity to a reference "
        "temperature. White, Viscous Fluid Flow 3rd ed., Table 1-2"
    ),
}
register_sources(SOURCES)


# --------------------------------------------------------------------------------------
#   Constants
# --------------------------------------------------------------------------------------

#: Sutherland reference constant for air, K. See SOURCES["atmo_sutherland_S"].
SUTHERLAND_S: float = 110.4

#: Sutherland reference temperature and viscosity for air (SI). White, Viscous Fluid Flow.
SUTHERLAND_T0: float = 273.15
SUTHERLAND_MU0: float = 1.716e-5

H_MIN: float = 0.0

# Table ceiling.
#
# This was 30 km, which was ample for the SV-1 cruise-and-dive mission. It is NOT ample for a
# lofted two-stage intercept: measured IV-1 arcs apogee between 45 and 54 km. Above the ceiling
# `atmo()` clamps, which holds density at its ceiling value and therefore OVERSTATES drag on
# everything above it. The ceiling is now 86 km, which is the upper limit of the US Standard 1976
# lower atmosphere and of SUAVE's implementation of it.
#
# SUAVE takes GEOMETRIC altitude and converts internally to geopotential, which is the convention
# the standard's tables are printed in. That distinction matters above about 30 km: at 47 km
# geometric the geopotential altitude is 46.655 km, and evaluating the standard there gives
# 115.8 Pa, which is what SUAVE returns. Comparing SUAVE against the printed 47 km row instead
# looks like a 4 percent error and is not one. `tests/test_atmosphere_high.py` does the conversion
# properly and pins the agreement.
H_MAX: float = 86_000.0
H_STEP: float = 10.0

#: The old ceiling, kept so a test can prove the extension did not disturb the region below it.
H_MAX_LEGACY: float = 30_000.0

_FIELDS = ("pressure", "temperature", "density", "speed_of_sound", "dynamic_viscosity")


# --------------------------------------------------------------------------------------
#   State container
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AtmoState:
    """Atmospheric state. Each field is a float for scalar input, an ndarray for array input."""

    pressure: Any            # Pa
    temperature: Any         # K
    density: Any             # kg/m^3
    speed_of_sound: Any      # m/s
    dynamic_viscosity: Any   # Pa.s

    @property
    def kinematic_viscosity(self) -> Any:
        """m^2/s."""
        return self.dynamic_viscosity / self.density


# --------------------------------------------------------------------------------------
#   The table
# --------------------------------------------------------------------------------------

_TABLE: dict[str, np.ndarray] | None = None


def _build_table() -> dict[str, np.ndarray]:
    """Evaluate SUAVE once on the altitude grid. Called at most once per process."""
    add_suave_to_path()
    from SUAVE.Analyses.Atmospheric import US_Standard_1976  # noqa: PLC0415

    n = int(round((H_MAX - H_MIN) / H_STEP)) + 1
    h = np.linspace(H_MIN, H_MAX, n)
    values = US_Standard_1976().compute_values(h)

    table: dict[str, np.ndarray] = {"altitude": h}
    for name in _FIELDS:
        table[name] = np.ravel(np.asarray(values[name], dtype=float)).copy()
        if table[name].size != n:
            raise RuntimeError(f"SUAVE returned {table[name].size} values for {name}, expected {n}")
    return table


def table() -> dict[str, np.ndarray]:
    """The cached altitude table. Builds it on first use."""
    global _TABLE
    if _TABLE is None:
        _TABLE = _build_table()
    return _TABLE


def prime() -> None:
    """Force the table to be built now. Call before timing anything."""
    table()


# --------------------------------------------------------------------------------------
#   Public API
# --------------------------------------------------------------------------------------


def atmo(h: float | np.ndarray) -> AtmoState:
    """Atmospheric state at geometric altitude `h` in metres.

    Vectorised: `h` may be a scalar or any array-like. Altitudes outside `H_MIN` to `H_MAX` are
    clamped to the table ends, which is deliberate. The sizing loop must never silently
    extrapolate the atmosphere, and the trajectory integrator can overshoot by a metre at a
    segment boundary. Clamping keeps that harmless; use `is_in_range` to test explicitly.

    The grid is uniform with a constant `H_STEP`, so the bracketing index is computed by
    arithmetic rather than by search. `np.interp` would binary-search the whole array on every
    field of every call, which cost real time once the ceiling went from 30 km to 86 km and the
    grid grew to 8601 points. Index arithmetic is O(1) in the table size and gives bit-identical
    results on a uniform grid, so this is a pure speed-up, not an approximation. There is a test
    that pins the agreement against `np.interp`.
    """
    t = table()
    scalar = np.isscalar(h) or (isinstance(h, np.ndarray) and h.ndim == 0)

    if scalar:
        x = (float(h) - H_MIN) / H_STEP
        n = t["altitude"].size
        if x <= 0.0:
            i, f = 0, 0.0
        elif x >= n - 1:
            i, f = n - 2, 1.0
        else:
            i = int(x)
            f = x - i
        out: list[Any] = []
        for name in _FIELDS:
            col = t[name]
            a = col[i]
            out.append(float(a + f * (col[i + 1] - a)))
        return AtmoState(*out)

    hq = np.asarray(h, dtype=float)
    x = (hq - H_MIN) / H_STEP
    n = t["altitude"].size
    idx = np.clip(np.floor(x).astype(np.intp), 0, n - 2)
    frac = np.clip(x - idx, 0.0, 1.0)
    arr_out: list[Any] = []
    for name in _FIELDS:
        col = t[name]
        a = col[idx]
        arr_out.append(a + frac * (col[idx + 1] - a))
    return AtmoState(*arr_out)


def is_in_range(h: float | np.ndarray) -> bool | np.ndarray:
    """True where `h` lies inside the tabulated band, so callers can assert rather than guess."""
    hq = np.asarray(h, dtype=float)
    inside = (hq >= H_MIN) & (hq <= H_MAX)
    return bool(inside) if inside.ndim == 0 else inside


def speed_of_sound(h: float | np.ndarray) -> float | np.ndarray:
    """m/s. Cheaper than building a full AtmoState when only `a` is wanted."""
    t = table()
    v = np.interp(np.asarray(h, dtype=float), t["altitude"], t["speed_of_sound"])
    return float(v) if v.ndim == 0 else v


def velocity(mach: float | np.ndarray, h: float | np.ndarray) -> float | np.ndarray:
    """True airspeed, m/s, from Mach number and altitude."""
    return np.asarray(mach, dtype=float) * speed_of_sound(h)


def q_dynamic(h: float | np.ndarray, V: float | np.ndarray) -> float | np.ndarray:
    """Free-stream dynamic pressure 0.5 rho V^2, Pa."""
    t = table()
    rho = np.interp(np.asarray(h, dtype=float), t["altitude"], t["density"])
    out = 0.5 * rho * np.asarray(V, dtype=float) ** 2
    return float(out) if np.ndim(out) == 0 else out


def reynolds_per_metre(h: float | np.ndarray, V: float | np.ndarray) -> float | np.ndarray:
    """Unit Reynolds number rho V / mu, 1/m. Multiply by a reference length to get Re."""
    t = table()
    hq = np.asarray(h, dtype=float)
    rho = np.interp(hq, t["altitude"], t["density"])
    mu = np.interp(hq, t["altitude"], t["dynamic_viscosity"])
    out = rho * np.asarray(V, dtype=float) / mu
    return float(out) if np.ndim(out) == 0 else out


def sutherland_viscosity(T: float | np.ndarray) -> float | np.ndarray:
    """Dynamic viscosity of air at temperature `T` in K, Pa.s.

    Needed by the reference-temperature skin-friction method in `aero.py`, which must
    evaluate viscosity at a temperature that is not the free-stream temperature and so
    cannot use the tabulated value. Sutherland's law, constants in SOURCES.
    """
    Tq = np.asarray(T, dtype=float)
    out = SUTHERLAND_MU0 * (Tq / SUTHERLAND_T0) ** 1.5 * (SUTHERLAND_T0 + SUTHERLAND_S) / (
        Tq + SUTHERLAND_S
    )
    return float(out) if np.ndim(out) == 0 else out
