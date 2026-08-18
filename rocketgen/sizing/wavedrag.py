"""Slender-body wave drag: the only model in this repo that can see the nose SHAPE.

WHY THIS MODULE EXISTS
----------------------
Before it, no drag model here could distinguish one nose from another at fixed fineness.
`aero.CD_wave_body` is the Bonney correlation, `bonney_nose_wave_cd(f_nose, mach)`, a function
of fineness alone; `aero.CD_wave_body_crosscheck` is the Sears-Haack volume rule, a function of
`(d/L)^2` alone. Both are blind to shape. Introducing a spline outer mould line without fixing
that would have produced a study reporting "shape changed nothing" for the wrong reason.

WHAT IS MODELLED
----------------
Supersonic linearised slender-body theory. For a body of revolution whose cross-section area
distribution `S(x)` satisfies `S'(0) = S'(L) = 0`, the wave drag is

    D / q = -(1 / 2 pi) int int S''(x1) S''(x2) ln|x1 - x2| dx1 dx2

which, under the Glauert substitution `x = (L/2)(1 - cos theta)` and the expansion
`S'(x) = sum_n A_n sin(n theta)`, collapses to

    D / q = (pi / 4) sum_n n A_n^2.                                   (the "Glauert series")

That series is the whole model. It is Mach-INDEPENDENT, which is a genuine property of the
leading-order theory and NOT an oversight.

HOW IT IS USED, AND WHY IT IS USED THAT WAY
-------------------------------------------
The theory's absolute drag level is less trustworthy than its shape sensitivity, and the repo
already has a drag level validated against 23 Basic Finner free-flight shots through
`config.CD0_CALIBRATION`. So this module is NOT summed into CD0. It supplies a dimensionless
RATIO

    nose_wave_shape_ratio(shape) = shape_factor(shape) / shape_factor(tangent ogive)

which multiplies the existing Bonney correlation at the loop boundary, exactly where
`CD0_CALIBRATION` is applied and for the same reason (CLAUDE.md section 8). Consequences:

* at the tangent-ogive shape the ratio is 1.0 to machine precision, so every previously
  banked SV-1 and IV-1 result is reproduced bit for bit;
* the Mach dependence and the calibrated magnitude stay with Bonney, which was validated
  against real data;
* only the part linear theory is actually good at - how much worse one shape is than another -
  is taken from linear theory.

VALIDATION (see `tests/test_wavedrag.py`)
------------------------------------------
* Sears-Haack body, `D/q = 128 V^2 / (pi L^4)`: reproduced to machine precision.
* Von Karman ogive, `C_D` on base area `= (d/L)^2`, shape factor `4/pi`: 2.1e-5.
* The von Karman ogive is confirmed to BE the constrained optimum, not assumed to be.
* The Glauert series is checked against direct numerical evaluation of the double integral.

LIMITS, STATED
--------------
Slender-body theory needs a slender body and a Mach number away from 1. `max_slope` is
reported so a body outside the range is visible. The theory has no viscosity, no
Mach-number dependence at this order, and no transonic validity at all. None of that matters
much for a RATIO between two similar slender shapes at the same length and base area, which is
the only thing taken from it, but it would matter a great deal if the absolute level were used.

Units are SI. `D/q` is an area, m^2.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Sequence

from ..config import register_sources
from ..oml_spline import (
    N_CTRL_DEFAULT,
    basis_matrix,
    ogive_control_values,
    von_karman_radius,
)

__all__ = [
    "N_MODES_DEFAULT",
    "glauert_coefficients",
    "nose_wave_shape_ratio",
    "optimal_control_values",
    "sears_haack_drag_over_q",
    "shape_factor",
    "shape_factor_of_control",
    "von_karman_ogive_drag_over_q",
    "wave_drag_over_q",
]

SOURCES: dict[str, str] = {
    "wave_drag_slender_body": (
        "Ashley and Landahl, Aerodynamics of Wings and Bodies, Dover 1985, chapter 6, and "
        "Liepmann and Roshko, Elements of Gasdynamics, chapter 9: linearised supersonic "
        "slender-body wave drag of a body of revolution, "
        "D/q = -(1/2 pi) int int S''(x1) S''(x2) ln|x1-x2| dx1 dx2, valid when "
        "S'(0) = S'(L) = 0."
    ),
    "wave_drag_glauert_series": (
        "Von Karman's reduction of the slender-body double integral under "
        "x = (L/2)(1 - cos theta) and S'(x) = sum A_n sin(n theta), giving "
        "D/q = (pi/4) sum n A_n^2. VERIFIED IN THIS REPO by direct numerical evaluation of "
        "the double integral: the deleted-diagonal midpoint quadrature converges monotonically "
        "onto the series value (error ratio about 1.87 per grid doubling), and the series "
        "reproduces both closed forms below to the tolerances stated in their entries."
    ),
    "wave_drag_sears_haack": (
        "Sears-Haack body of given length and volume, D/q = 128 V^2/(pi L^4). Reproduced by "
        "the Glauert series in this repo to machine precision; its area distribution is the "
        "pure second Glauert mode, S'(x) = (3 pi R^2/L) sin(2 theta)."
    ),
    "wave_drag_von_karman_ogive": (
        "Von Karman (LD-Haack) ogive, the minimum-wave-drag forebody for given length and "
        "base area: C_D on base area = (d/L)^2, equivalently shape factor 4/pi. Its area "
        "distribution is the pure first Glauert mode, S'(x) = (4 R^2/L) sin(theta). "
        "Reproduced in this repo to 2.1e-5, and CONFIRMED to be the constrained optimum "
        "rather than assumed: adding any higher mode at fixed base area raises the drag."
    ),
    "wave_drag_applied_as_ratio": (
        "MODELLING CHOICE, stated rather than hidden. Slender-body theory is used only for "
        "the RATIO between two shapes at the same length and base area, multiplying the "
        "Bonney correlation which keeps the Mach dependence and the level validated against "
        "the 23 Basic Finner shots. The absolute slender-body drag level is NOT summed into "
        "CD0. At the tangent-ogive shape the ratio is 1.0 to machine precision, so the "
        "pre-spline results are reproduced exactly."
    ),
    "wave_drag_mach_independence": (
        "Leading-order slender-body wave drag carries no Mach number. That is a property of "
        "the theory, not an omission. Because it is applied here as a ratio against an ogive "
        "evaluated at the same Mach, the Mach dependence of the delivered CD_wave_body is "
        "entirely Bonney's. GUESS-FREE but APPROXIMATE: the true shape sensitivity does drift "
        "with Mach number, and that drift is not modelled and has not been quantified here."
    ),
    "wave_drag_optimal_spline": (
        "MEASURED, not chosen. The drag-optimal control values are found by minimising the "
        "Glauert-series shape factor over the free control values with the end conditions "
        "pinned. The optimum is fineness-invariant to 6.2e-9, as slender-body theory requires, "
        "so it is a constant of the parameterisation and is solved once."
    ),
    "wave_drag_mode_count": (
        "MEASURED. The Glauert coefficient sum is truncated at 60 modes. Raising the "
        "truncation from 40 to 60 to 100 modes changes the tangent-ogive shape factor by less "
        "than 1e-6 relative, because A_n decays fast for smooth profiles. See "
        "`tests/test_wavedrag.py::test_mode_truncation_is_converged`."
    ),
}
register_sources(SOURCES)

# See SOURCES["wave_drag_mode_count"].
N_MODES_DEFAULT = 60

# Quadrature resolution for the Glauert projection. The integrand is smooth in theta.
N_QUAD_DEFAULT = 2001


# --------------------------------------------------------------------------------------
#   The Glauert series
# --------------------------------------------------------------------------------------


def glauert_coefficients(
    stations: Sequence[float],
    areas: Sequence[float],
    length: float,
    n_modes: int = N_MODES_DEFAULT,
) -> list[float]:
    """Glauert coefficients `A_n` of `S'(x)` for a body of length `length`.

    `stations` and `areas` are `x` and `S(x)`, sorted, spanning `0` to `length`. The
    coefficients come from the orthogonality of `sin(n theta)` on `[0, pi]`:

        A_n = (2 / pi) int_0^pi S'(x(theta)) sin(n theta) d(theta).

    `S'` is taken by central differences on the supplied table, so the table must be dense
    enough to resolve it. That is exactly why `SplineProfile.area_distribution` reports at the
    polygon stations rather than at the 16 stations nTop measures.
    """
    if len(stations) != len(areas):
        raise ValueError(
            f"stations and areas must be the same length, got {len(stations)} and {len(areas)}"
        )
    if len(stations) < 5:
        raise ValueError(f"need at least 5 stations to difference S(x), got {len(stations)}")
    if length <= 0.0:
        raise ValueError(f"length must be positive, got {length}")

    xs = [float(v) for v in stations]
    ss = [float(v) for v in areas]

    # S'(x) by central differences, one-sided at the ends
    dsdx: list[float] = []
    for i in range(len(xs)):
        if i == 0:
            dsdx.append((ss[1] - ss[0]) / (xs[1] - xs[0]))
        elif i == len(xs) - 1:
            dsdx.append((ss[-1] - ss[-2]) / (xs[-1] - xs[-2]))
        else:
            dsdx.append((ss[i + 1] - ss[i - 1]) / (xs[i + 1] - xs[i - 1]))

    # theta of each station, and a trapezoid integration in theta
    thetas = [math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * x / length))) for x in xs]
    order = sorted(range(len(xs)), key=lambda i: thetas[i])
    th = [thetas[i] for i in order]
    fp = [dsdx[i] for i in order]

    out: list[float] = []
    for m in range(1, n_modes + 1):
        acc = 0.0
        for a, b in zip(range(len(th) - 1), range(1, len(th))):
            fa = fp[a] * math.sin(m * th[a])
            fb = fp[b] * math.sin(m * th[b])
            acc += 0.5 * (fa + fb) * (th[b] - th[a])
        out.append(2.0 / math.pi * acc)
    return out


def wave_drag_over_q(
    stations: Sequence[float],
    areas: Sequence[float],
    length: float,
    n_modes: int = N_MODES_DEFAULT,
) -> float:
    """`D / q` in m^2 from a measured or computed area distribution.

    See SOURCES["wave_drag_glauert_series"]. Note this is the ABSOLUTE slender-body level,
    which this repo deliberately does not sum into CD0. Use `nose_wave_shape_ratio` for what
    the sizing loop consumes.
    """
    a = glauert_coefficients(stations, areas, length, n_modes)
    return (math.pi / 4.0) * sum((n + 1) * v * v for n, v in enumerate(a))


# --------------------------------------------------------------------------------------
#   Closed forms, for validation
# --------------------------------------------------------------------------------------


def sears_haack_drag_over_q(volume: float, length: float) -> float:
    """`D/q = 128 V^2 / (pi L^4)`. See SOURCES["wave_drag_sears_haack"]."""
    return 128.0 * volume * volume / (math.pi * length ** 4)


def von_karman_ogive_drag_over_q(diameter: float, length: float) -> float:
    """`D/q` of the minimum-drag forebody: `C_D = (d/L)^2` on base area.

    See SOURCES["wave_drag_von_karman_ogive"].
    """
    s_base = 0.25 * math.pi * diameter * diameter
    return s_base * (diameter / length) ** 2


# --------------------------------------------------------------------------------------
#   Shape factor: the fineness-free part of the drag
# --------------------------------------------------------------------------------------


def shape_factor(
    radius_of_t: Callable[[float], float],
    n_modes: int = N_MODES_DEFAULT,
    n_quad: int = N_QUAD_DEFAULT,
) -> float:
    """Dimensionless shape factor of a normalised profile `r/R` against `t = x/L`.

    Defined so that `D/q = shape_factor * S_B^2 / L^2` with `S_B = pi R^2` the base area.
    Because `S(x)/S_B = (r/R)^2`, the factor depends ONLY on the normalised profile, which is
    why the drag-optimal shape is fineness-invariant.

    Reference values: von Karman ogive `4/pi = 1.27324`; tangent ogive about `1.49`.
    """
    ts = [i / (n_quad - 1) for i in range(n_quad)]
    s = [radius_of_t(t) ** 2 for t in ts]

    # ds/dt by central differences
    h = ts[1] - ts[0]
    dsdt = []
    for i in range(n_quad):
        if i == 0:
            dsdt.append((s[1] - s[0]) / h)
        elif i == n_quad - 1:
            dsdt.append((s[-1] - s[-2]) / h)
        else:
            dsdt.append((s[i + 1] - s[i - 1]) / (2.0 * h))

    th = [math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * t))) for t in ts]
    total = 0.0
    for m in range(1, n_modes + 1):
        acc = 0.0
        for i in range(n_quad - 1):
            fa = dsdt[i] * math.sin(m * th[i])
            fb = dsdt[i + 1] * math.sin(m * th[i + 1])
            acc += 0.5 * (fa + fb) * (th[i + 1] - th[i])
        a_m = 2.0 / math.pi * acc
        total += m * a_m * a_m
    return (math.pi / 4.0) * total


def shape_factor_of_control(
    control: Sequence[float],
    n_modes: int = N_MODES_DEFAULT,
    n_quad: int = N_QUAD_DEFAULT,
) -> float:
    """Shape factor of a spline given by its control values."""
    ctrl = tuple(float(v) for v in control)
    ts = [i / (n_quad - 1) for i in range(n_quad)]
    N = basis_matrix(ts, len(ctrl))
    ys = [sum(w * c for w, c in zip(row, ctrl)) for row in N]
    lookup = dict(zip(ts, ys))
    return shape_factor(lambda t: lookup[t], n_modes, n_quad)


@lru_cache(maxsize=16)
def _ogive_shape_factor(k: float, n_ctrl: int) -> float:
    """Shape factor of the SPLINE FIT to the tangent ogive of `k = L/R`.

    Deliberately the fitted spline, not the analytic ogive. The ratio must be exactly 1.0
    when the design sits at the ogive-equivalent control values, and that is only true if
    numerator and denominator are computed the same way.
    """
    return shape_factor_of_control(ogive_control_values(k, n_ctrl))


def nose_wave_shape_ratio(
    control: Sequence[float],
    k: float,
    n_modes: int = N_MODES_DEFAULT,
) -> float:
    """Nose wave drag of this shape divided by the tangent ogive's, at the same `k = L/R`.

    THIS IS WHAT THE SIZING LOOP CONSUMES. It multiplies `aero.CD_wave_body`, so the
    calibrated Bonney level and its Mach dependence are untouched and only the shape effect
    is added. See SOURCES["wave_drag_applied_as_ratio"].

    Returns exactly 1.0 for the ogive-equivalent control values, which is the degeneracy that
    keeps every pre-spline result reproducible.
    """
    ctrl = tuple(float(v) for v in control)
    den = _ogive_shape_factor(float(k), len(ctrl))
    if den <= 0.0:
        raise ValueError(f"degenerate reference shape factor {den} at k = {k}")
    return shape_factor_of_control(ctrl, n_modes) / den


# --------------------------------------------------------------------------------------
#   The drag-optimal spline
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=8)
def optimal_control_values(
    n_ctrl: int = N_CTRL_DEFAULT,
    n_modes: int = N_MODES_DEFAULT,
    n_quad: int = 801,
) -> tuple[float, ...]:
    """Control values minimising the Glauert shape factor, ends pinned.

    Solved by Nelder-Mead from the von Karman profile fit, which is already close. The
    objective is quartic in the control values (the radius is linear, the area is its square,
    and the drag is a sum of squared linear functionals of the area), so it is smooth and
    low-dimensional and a simplex method is adequate.

    The result is a CONSTANT of the parameterisation: it does not depend on fineness, on
    diameter, or on anything else in the design vector. See SOURCES["wave_drag_optimal_spline"]
    and SOURCES["spline_control_count"] for what it buys.
    """
    from ..oml_spline import fit_control_values

    fixed = {0: 0.0, n_ctrl - 2: 1.0, n_ctrl - 1: 1.0}
    free = [i for i in range(n_ctrl) if i not in fixed]

    ts = [i / (n_quad - 1) for i in range(n_quad)]
    N = basis_matrix(ts, n_ctrl)
    base = [sum(v * row[i] for i, v in fixed.items()) for row in N]

    def objective(x: Sequence[float]) -> float:
        ys = [base[j] + sum(N[j][i] * x[a] for a, i in enumerate(free))
              for j in range(len(ts))]
        lookup = dict(zip(ts, ys))
        return shape_factor(lambda t: lookup[t], n_modes, n_quad)

    start = list(fit_control_values(von_karman_radius, n_ctrl)[i] for i in free)
    best = _nelder_mead(objective, start)

    out = [0.0] * n_ctrl
    for i, v in fixed.items():
        out[i] = v
    for a, i in enumerate(free):
        out[i] = best[a]
    return tuple(out)


def _nelder_mead(
    f: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    step: float = 0.05,
    max_iter: int = 6000,
    tol: float = 1e-12,
) -> list[float]:
    """Plain Nelder-Mead. Written out so this module needs no optimiser dependency.

    Standard coefficients: reflection 1, expansion 2, contraction 0.5, shrink 0.5.
    """
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        p = list(x0)
        p[i] += step
        simplex.append(p)
    vals = [f(p) for p in simplex]

    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) <= tol * (abs(vals[0]) + tol):
            break
        centroid = [sum(p[d] for p in simplex[:-1]) / n for d in range(n)]
        worst = simplex[-1]

        refl = [centroid[d] + (centroid[d] - worst[d]) for d in range(n)]
        f_refl = f(refl)
        if f_refl < vals[0]:
            exp = [centroid[d] + 2.0 * (centroid[d] - worst[d]) for d in range(n)]
            f_exp = f(exp)
            simplex[-1], vals[-1] = (exp, f_exp) if f_exp < f_refl else (refl, f_refl)
        elif f_refl < vals[-2]:
            simplex[-1], vals[-1] = refl, f_refl
        else:
            con = [centroid[d] + 0.5 * (worst[d] - centroid[d]) for d in range(n)]
            f_con = f(con)
            if f_con < vals[-1]:
                simplex[-1], vals[-1] = con, f_con
            else:
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][d] + 0.5 * (simplex[i][d] - simplex[0][d])
                                  for d in range(n)]
                    vals[i] = f(simplex[i])
    return simplex[int(min(range(n + 1), key=lambda i: vals[i]))]
