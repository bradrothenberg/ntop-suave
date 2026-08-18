"""Spline outer mould lines: the shape family, and the arithmetic nTop has to do.

WHY A SPLINE, AND WHAT IS ACTUALLY SPLINED
------------------------------------------
The SV-1 and IV-1 outer mould lines were tangent ogives: a one-parameter family, fixed by
fineness alone. A tangent ogive carries about 17 percent more supersonic nose wave drag than
the linear-theory optimum for the same length and base area (`sizing/wavedrag.py` establishes
that number). A spline is a shape family wide enough to go and collect most of that.

HOW IT IS BUILT IN nTOP
-----------------------
nTop revolves the spline itself. There is no chord polygon and no discretisation error. The
block chain, all four of which need `Recipe.raw_block` because NONE of them are in the vendored
`functions.json`:

    spline_by_control_points<list<point>,integer>[5.20.0]      (points, degree) -> spline
    core.list<curve_interface>                                 -> list<curve_interface>
    profile_from_curves<list<curve_interface>,vector>[5.20.0]  (curves, NORMAL) -> new_profile
    revolve<new_profile,axis,real>[5.20.0]                     (profile, axis, angle) -> implicit

Four traps, each of which alone makes `convert` fail with a bare "Error loading recipe":
the curve type is `curve_interface` and not `curve`; `profile_from_curves` returns
`new_profile` and not `profile`, with no props bridge to `implicit_2d`; the Normal vector is
DIMENSIONLESS; and the degree is a plain integer literal. `docs/NTOP_NOTES.md` section 25 has
the full account.

A NOTE ON HOW THIS MODULE USED TO WORK
---------------------------------------
It previously sampled the spline into a 24-segment chord polygon and revolved that, because the
vendored block universe lists no route from a curve to a revolvable profile and that was taken
as proof no such route existed. It was not proof. The universe is missing whole BLOCKS and whole
TYPES that the installed build exposes, so "absent from functions.json" says nothing about what
nTop can do. Every closed form below is now EXACT for the solid nTop builds, where before each
one carried a discretisation error that had to be budgeted for.

WHY THE SPLINE STAYS LIVE IN THE NOTEBOOK
-----------------------------------------
The notebook emits the CONTROL POINTS, and nTop does the rest. Control point `i` is

    P_i = ( L * STATION_FRACTIONS[i],  r0 + (R - r0) * c_i )

so each one is two multiplies of live notebook inputs. `L`, `R` and every `c_i` stay real
inputs, one converted `.ntop` serves every design point, and `docs/NTOP_NOTES.md` section 3
("convert once, run many") holds.

The axial fractions are FIXED and only the `c_i` vary. That is the one modelling choice in the
parameterisation and it earns its keep: with `x(u)` fixed, `y(u)` is LINEAR in the control
values, so the enclosed volume is quadratic in them and a blend between two valid shapes is
itself a valid shape rather than an approximation of one. Letting the axial positions move too
would buy a little more shape freedom and cost that property.

END CONDITIONS
--------------
A clamped B-spline interpolates its first and last control values, so with control values
`c` of length `n`:

    c[0]   = 0     sharp tip, r(0) = 0
    c[n-1] = 1     radius R at the shoulder
    c[n-2] = 1     zero slope at the shoulder, so the nose meets the cylinder tangentially

The last one matters more than it looks. Slender-body wave drag is only expressible as the
Glauert series `sizing/wavedrag.py` uses when S'(0) = S'(L) = 0. A nose with a slope
discontinuity at the shoulder breaks that, and the drag model would then be answering a
different question from the one asked. So tangency is enforced by construction, not checked
afterwards.

The remaining `n - 3` control values are the free shape degrees of freedom.

Units are SI: metres. Control values are dimensionless (fractions of R).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from .config import register_sources

__all__ = [
    "DEGREE",
    "derivative_control",
    "station_fractions",
    "boattail_control_values",
    "N_CTRL_DEFAULT",
    "N_POLY_DEFAULT",
    "SplineProfile",
    "basis_matrix",
    "clamped_knots",
    "cone_control_values",
    "control_values_for",
    "fit_control_values",
    "ogive_control_values",
    "tangent_ogive_radius",
    "von_karman_control_values",
    "von_karman_radius",
]

SOURCES: dict[str, str] = {
    "spline_basis": (
        "Piegl and Tiller, The NURBS Book, 2nd ed., Springer 1997, sections 2.2 to 2.5: "
        "Cox-de Boor recursion for the B-spline basis, and the clamped (open uniform) knot "
        "vector whose end multiplicity equals degree+1, which makes the curve interpolate "
        "its first and last control points."
    ),
    "spline_degree": (
        "Cubic (degree 3) chosen because it is the lowest degree with continuous curvature, "
        "which a body of revolution needs if its area distribution S(x) is to have a "
        "continuous second derivative. `sizing/wavedrag.py` differentiates S(x) twice, so a "
        "quadratic spline would make the wave-drag integrand discontinuous."
    ),
    "spline_control_count": (
        "MEASURED, not chosen by taste. Optimising the free control values against the "
        "slender-body wave-drag functional at f_nose = 3.0 recovers this fraction of the gap "
        "between the tangent ogive and the von Karman optimum: 7 control points 79.2 percent "
        "(-11.53 percent nose wave drag), 9 points 86.1 percent (-12.54 percent), 11 points "
        "89.6 percent (-13.05 percent), 13 points 91.7 percent, 15 points 93.1 percent. "
        "9 is the default: it takes most of the available gain for 6 free shape variables. "
        "The residual gap is the cubic spline's finite tip slope against the von Karman "
        "profile's infinite one (r ~ x^(1/4) at the tip), which no polynomial spline with a "
        "fixed tip control point can reproduce."
    ),
    "spline_no_discretisation": (
        "NOT APPLICABLE BY CONSTRUCTION, and recorded so the absence is deliberate rather "
        "than forgotten. nTop revolves the SPLINE (docs/NTOP_NOTES.md section 25), so the "
        "solid has no chord-polygon discretisation and there is no sample count to justify. "
        "`SplineProfile`'s volume and planform integrals are exact for their polynomial "
        "integrands under an 8-point Gauss rule per knot span, and the wetted-area integral, "
        "which carries a square root, is refined 16-fold per span. An earlier version of this "
        "module DID sample a 24-segment chord polygon and carried a -0.079 percent volume "
        "error for it; that error is now zero."
    ),
    "spline_greville_stations": (
        "MEASURED. The control points' axial fractions are the Greville abscissae of the knot "
        "vector, which is the unique choice making a B-spline reproduce its own parameter, so "
        "x(u) = length * u exactly (verified to 2.2e-16). With evenly spaced fractions instead "
        "the axial coordinate becomes a nonlinear reparameterisation, the least-squares fit to "
        "a tangent ogive degrades from 1.0e-6 to 2.7e-3 of R, and the enclosed volume moves by "
        "0.096 percent instead of 1.7e-7."
    ),
    "spline_boattail_family": (
        "CHOICE OF FAMILY, not a derived optimum, and stated as such. The splined boattail "
        "interpolates between the straight cone (blend 0, reproduced exactly, which is the "
        "degeneracy anchor against the existing Prandtl-Meyer boattail model) and a cubic "
        "ease-in t^2(3-2t) (blend 1), which turns gently at the shoulder and sharply near the "
        "base. No claim is made here that the curved boattail has lower drag; "
        "`sizing/aero.py` integrates the actual contour and reports the result."
    ),
    "spline_ogive_degeneracy": (
        "MEASURED. A 9-control-point cubic B-spline least-squares fitted to the tangent ogive "
        "reproduces it to 1.046e-6 of R in maximum radius error, 1.7e-7 in relative volume and "
        "6e-8 in relative wetted area, at f_nose = 3.0. At 7 control points it is 4.882e-6 and "
        "1.35e-6. All are orders of magnitude inside the 1 percent volume gate the ogive "
        "polygon was itself accepted under, so selecting the spline path does not silently "
        "move the validated ogive baseline."
    ),
}
register_sources(SOURCES)

DEGREE = 3

# Default free-shape resolution. See SOURCES["spline_control_count"].
N_CTRL_DEFAULT = 9

# Sample count for REPORTING tables only: area distributions, plots, exported profiles. It no
# longer affects any measured quantity, because nTop revolves the spline and the closed forms
# integrate it exactly. `tests/test_oml_spline.py::test_the_closed_forms_need_no_refinement`
# asserts that changing it leaves volume, area and planform bit-identical.
N_POLY_DEFAULT = 201


# --------------------------------------------------------------------------------------
#   B-spline basis
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=64)
def clamped_knots(n_ctrl: int, degree: int = DEGREE) -> tuple[float, ...]:
    """Uniform clamped knot vector for `n_ctrl` control values.

    End multiplicity is `degree + 1`, so the curve interpolates the first and last control
    values. Interior knots are uniformly spaced on (0, 1). See SOURCES["spline_basis"].
    """
    if n_ctrl < degree + 1:
        raise ValueError(
            f"need at least degree+1 = {degree + 1} control values, got {n_ctrl}"
        )
    n_interior = n_ctrl - degree - 1
    interior = [(i + 1) / (n_interior + 1) for i in range(n_interior)]
    return tuple([0.0] * (degree + 1) + interior + [1.0] * (degree + 1))


def _basis_at(t: float, n_ctrl: int, degree: int = DEGREE) -> list[float]:
    """The `n_ctrl` basis-function values at parameter `t`, by Cox-de Boor.

    Evaluated one basis at a time via de Boor's algorithm on the unit control vectors. That
    is O(n_ctrl * degree) rather than the O(degree) an optimised span-local evaluation would
    cost, but this runs once per notebook build, not per design point, so clarity wins.
    """
    kv = clamped_knots(n_ctrl, degree)
    out: list[float] = []
    for i in range(n_ctrl):
        coeffs = [1.0 if j == i else 0.0 for j in range(n_ctrl)]
        out.append(_de_boor(t, kv, coeffs, degree))
    return out


def _de_boor(t: float, kv: Sequence[float], c: Sequence[float], degree: int) -> float:
    """de Boor evaluation of a spline with knot vector `kv` and coefficients `c` at `t`."""
    n = len(c)
    if t >= kv[n]:
        k = n - 1
    else:
        # span index: largest k with kv[k] <= t, clamped into the valid range
        k = degree
        for j in range(degree, n):
            if kv[j] <= t:
                k = j
            else:
                break
    d = [c[k - degree + r] for r in range(degree + 1)]
    for r in range(1, degree + 1):
        for s in range(degree, r - 1, -1):
            left = kv[k + s - degree]
            right = kv[k + 1 + s - r]
            a = 0.0 if right == left else (t - left) / (right - left)
            d[s] = (1.0 - a) * d[s - 1] + a * d[s]
    return d[degree]


def basis_matrix(
    ts: Sequence[float],
    n_ctrl: int = N_CTRL_DEFAULT,
    degree: int = DEGREE,
) -> list[list[float]]:
    """`N[j][i]` = the i-th basis function evaluated at `ts[j]`.

    THESE ARE THE CONSTANTS THE NOTEBOOK BAKES IN. Each row is the weight vector for one
    polygon vertex, so vertex j has radius `R * sum_i N[j][i] * c_i` with `R` and every `c_i`
    a live notebook input. See the module docstring.

    Every row sums to 1 (partition of unity), which is what makes a control value of all-ones
    give radius exactly R.
    """
    return [_basis_at(float(t), n_ctrl, degree) for t in ts]


@lru_cache(maxsize=64)
def station_fractions(n_ctrl: int = N_CTRL_DEFAULT, degree: int = DEGREE) -> tuple[float, ...]:
    """Axial position of each control point, as a fraction of the run length.

    These are the GREVILLE ABSCISSAE of the knot vector, `g_i = mean(kv[i+1 .. i+degree])`,
    and that choice is not cosmetic. The Greville abscissae are the unique control values for
    which a B-spline reproduces its own parameter:

        sum_i N_i(u) g_i = u   exactly, for all u.

    So with these axial fractions `x(u) = length * u` identically, and the curve's radius is a
    spline in the AXIAL STATION rather than in an arbitrary parameter. That matters a lot in
    practice: with evenly spaced fractions instead, `x(u)` is a nonlinear reparameterisation,
    the fit to a tangent ogive degrades from 1.0e-6 to 2.7e-3 of R, and the enclosed volume
    moves by 0.096 percent instead of 1.7e-7. Measured, both ways.

    They are FIXED. Only the radius control values are design variables.
    """
    if n_ctrl < degree + 1:
        raise ValueError(f"need at least {degree + 1} control points, got {n_ctrl}")
    kv = clamped_knots(n_ctrl, degree)
    return tuple(sum(kv[i + 1: i + 1 + degree]) / degree for i in range(n_ctrl))


def sample_stations(n_poly: int = N_POLY_DEFAULT) -> tuple[float, ...]:
    """Uniform parameter stations for REPORTING only. No measured quantity depends on these."""
    if n_poly < 2:
        raise ValueError(f"need at least 2 polygon samples, got {n_poly}")
    return tuple(i / (n_poly - 1) for i in range(n_poly))


# --------------------------------------------------------------------------------------
#   Reference profiles, as control values
# --------------------------------------------------------------------------------------


def tangent_ogive_radius(t: float, k: float) -> float:
    """Tangent-ogive `r / R` at `t = x / L`, for `k = L / R`.

    Dimensionless form of `y = sqrt(rho^2 - (L-x)^2) - (rho - R)` with `rho = (R^2+L^2)/(2R)`:
        y / R = sqrt(c^2 - k^2 (1-t)^2) - (c - 1),   c = (1 + k^2) / 2.
    This is the same algebra `ntopgen/rocket_notebook.py::_ogive_points` emits into nTop, so
    the two cannot drift apart without a test noticing.
    """
    c = (1.0 + k * k) / 2.0
    inner = c * c - k * k * (1.0 - t) ** 2
    return math.sqrt(max(inner, 0.0)) - (c - 1.0)


def von_karman_radius(t: float) -> float:
    """Von Karman (LD-Haack) `r / R` at `t = x / L`. Independent of fineness.

    `theta = arccos(1 - 2t)`, `r/R = sqrt((theta - sin(2 theta)/2) / pi)`.

    This is the minimum-wave-drag forebody for a given length and base area under slender-body
    theory. `sizing/wavedrag.py` reproduces its closed-form drag to 2.1e-5, and confirms it is
    the constrained optimum. It is included here as the target the spline chases, and as the
    reference the tests check against, NOT as a shape the notebook builds: its tip slope is
    infinite (`r ~ x^(1/4)`), which no clamped polynomial spline reproduces.
    """
    th = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * float(t))))
    return math.sqrt(max((th - 0.5 * math.sin(2.0 * th)) / math.pi, 0.0))


def _fixed_ends(n_ctrl: int) -> dict[int, float]:
    """The three control values pinned by the end conditions. See the module docstring."""
    return {0: 0.0, n_ctrl - 2: 1.0, n_ctrl - 1: 1.0}


def fit_control_values(
    radius_of_t,
    n_ctrl: int = N_CTRL_DEFAULT,
    n_fit: int = 2001,
) -> tuple[float, ...]:
    """Least-squares control values reproducing `radius_of_t(t) -> r/R`, ends pinned.

    Solves the normal equations directly rather than importing a least-squares routine, so
    this module has no numpy dependency and can be imported by the notebook builder without
    dragging the science stack in.
    """
    fixed = _fixed_ends(n_ctrl)
    free = [i for i in range(n_ctrl) if i not in fixed]
    us = [i / (n_fit - 1) for i in range(n_fit)]
    N = basis_matrix(us, n_ctrl)

    # x is ALSO a spline of the parameter now, not the parameter itself, so the target radius
    # must be evaluated at x(u) rather than at u. Getting this wrong shifts the whole fitted
    # profile forward and shows up as a volume error of order a percent.
    xf = station_fractions(n_ctrl)
    x_of_u = [sum(N[j][i] * xf[i] for i in range(n_ctrl)) for j in range(n_fit)]

    # residual target after removing the pinned contributions
    y = [radius_of_t(x_of_u[j]) - sum(v * N[j][i] for i, v in fixed.items())
         for j in range(n_fit)]

    m = len(free)
    ata = [[sum(N[j][free[a]] * N[j][free[b]] for j in range(n_fit)) for b in range(m)]
           for a in range(m)]
    atb = [sum(N[j][free[a]] * y[j] for j in range(n_fit)) for a in range(m)]
    sol = _solve(ata, atb)

    c = [0.0] * n_ctrl
    for i, v in fixed.items():
        c[i] = v
    for a, i in enumerate(free):
        c[i] = sol[a]
    return tuple(c)


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting. Small dense systems only."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        p = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[p][col]) < 1e-14:
            raise ValueError("singular normal equations: duplicate or degenerate stations")
        m[col], m[p] = m[p], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for cc in range(col, n + 1):
                m[r][cc] -= f * m[col][cc]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / m[r][r]
    return x


@lru_cache(maxsize=64)
def ogive_control_values(k: float, n_ctrl: int = N_CTRL_DEFAULT) -> tuple[float, ...]:
    """Control values reproducing the tangent ogive of `k = L / R`.

    This is the DEGENERACY ANCHOR. Building the spline path with these values must reproduce
    the ogive result, which is what makes the spline an extension of the validated baseline
    rather than a replacement for it. See SOURCES["spline_ogive_degeneracy"].

    Unlike the von Karman shape, this one DOES depend on fineness, because a tangent ogive of
    a different fineness is a genuinely different normalised profile.
    """
    return fit_control_values(lambda t: tangent_ogive_radius(t, k), n_ctrl)


@lru_cache(maxsize=64)
def cone_control_values(n_ctrl: int = N_CTRL_DEFAULT) -> tuple[float, ...]:
    """Control values for a cone. Kept because `nose_shape='cone'` is a validation case.

    A cone has a slope discontinuity at the shoulder, so the pinned tangency condition CANNOT
    represent it and this fit is deliberately poor near `t = 1`. It exists so the cone
    validation case still runs through the spline path; it is not a shape to design with, and
    `SplineProfile.wave_drag_shape_factor` is not meaningful for it.
    """
    return fit_control_values(lambda t: t, n_ctrl)


@lru_cache(maxsize=64)
def von_karman_control_values(n_ctrl: int = N_CTRL_DEFAULT) -> tuple[float, ...]:
    """Least-squares control values chasing the von Karman profile.

    NOTE this is NOT the minimum-drag spline. Fitting the PROFILE and minimising the DRAG are
    different problems, and the fit recovers only about 75 percent of the drag gap where a
    direct drag optimisation recovers 86 percent at the same control count. Use
    `sizing.wavedrag.optimal_control_values` for the drag-optimal shape. This function is here
    for comparison figures and tests.
    """
    return fit_control_values(von_karman_radius, n_ctrl)


def control_values_for(
    shape: str,
    k: float,
    n_ctrl: int = N_CTRL_DEFAULT,
) -> tuple[float, ...]:
    """Control values for a named reference shape. `k = L / R`."""
    if shape == "tangent_ogive":
        return ogive_control_values(k, n_ctrl)
    if shape == "cone":
        return cone_control_values(n_ctrl)
    if shape == "von_karman":
        return von_karman_control_values(n_ctrl)
    raise ValueError(
        f"unknown reference shape {shape!r}; expected 'tangent_ogive', 'cone' or 'von_karman'"
    )


# --------------------------------------------------------------------------------------
#   B-spline derivatives, needed for the exact closed forms
# --------------------------------------------------------------------------------------


def derivative_control(
    coeffs: Sequence[float],
    degree: int = DEGREE,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """`(derivative coefficients, derivative knot vector)` of a clamped B-spline.

    The derivative of a degree-`p` B-spline is a degree-`p-1` B-spline on the same knot vector
    with its first and last knot dropped, with coefficients

        Q_i = p (P_{i+1} - P_i) / (kv[i+p+1] - kv[i+1]).

    Piegl and Tiller section 3.3. This is EXACT, which is what lets `SplineProfile` integrate
    the revolved solid analytically instead of summing frusta.
    """
    kv = clamped_knots(len(coeffs), degree)
    q = []
    for i in range(len(coeffs) - 1):
        den = kv[i + degree + 1] - kv[i + 1]
        q.append(0.0 if den == 0.0 else degree * (coeffs[i + 1] - coeffs[i]) / den)
    return tuple(q), tuple(kv[1:-1])


def _eval_on(u: float, kv: Sequence[float], c: Sequence[float], degree: int) -> float:
    """de Boor on an explicit knot vector, for derivative splines whose knots are not clamped
    to the `clamped_knots` pattern."""
    n = len(c)
    if u >= kv[n]:
        k = n - 1
    else:
        k = degree
        for j in range(degree, n):
            if kv[j] <= u:
                k = j
            else:
                break
    d = [c[k - degree + r] for r in range(degree + 1)]
    for r in range(1, degree + 1):
        for s in range(degree, r - 1, -1):
            left = kv[k + s - degree]
            right = kv[k + 1 + s - r]
            a = 0.0 if right == left else (u - left) / (right - left)
            d[s] = (1.0 - a) * d[s - 1] + a * d[s]
    return d[degree]


# 8-point Gauss-Legendre on [-1, 1]. Exact for polynomials up to degree 15, which covers the
# volume integrand (y^2 x', degree 3p-1 = 8 for cubics) and the planform integrand exactly.
_GAUSS_X = (
    -0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
    0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363,
)
_GAUSS_W = (
    0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
    0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763,
)


def _integrate(f, spans: Sequence[tuple[float, float]], refine: int = 1) -> float:
    """Composite Gauss-Legendre over the given parameter spans.

    Integrating span by span matters: the spline is only piecewise polynomial, so a single
    Gauss rule across a knot would not be exact. `refine` subdivides each span, which the
    wetted-area integrand needs because its square root is not a polynomial.
    """
    total = 0.0
    for lo, hi in spans:
        if hi <= lo:
            continue
        for s in range(refine):
            a = lo + (hi - lo) * s / refine
            b = lo + (hi - lo) * (s + 1) / refine
            mid, half = 0.5 * (a + b), 0.5 * (b - a)
            total += half * sum(w * f(mid + half * x) for x, w in zip(_GAUSS_X, _GAUSS_W))
    return total


# --------------------------------------------------------------------------------------
#   A splined profile, and the exact closed forms of the solid nTop revolves
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SplineProfile:
    """One splined run of an outer mould line, in metres.

    The profile is the clamped cubic B-spline through control points

        P_i = ( length * STATION_FRACTIONS[i],  r0 + (radius - r0) * control[i] )

    where `STATION_FRACTIONS` are FIXED and only `control` varies. Fixing the axial fractions
    is what keeps the shape a one-parameter-per-control-value family: `y(u)` is then linear in
    `control`, so the enclosed volume is quadratic in it and the blend between two shapes is
    itself a valid shape rather than an approximation of one.

    `r0_over_r` lets the run START at a nonzero radius, which the boattail and the interstage
    flare both need.

    EVERY CLOSED FORM HERE IS EXACT for the solid nTop actually builds, because nTop revolves
    this same spline (see the module docstring for the block chain). Volume and planform are
    integrated with a Gauss rule that is exact for their polynomial integrands; wetted area
    carries a square root and is refined instead. That is a real improvement on the chord
    polygon this class used to describe, where every measurement carried a discretisation
    error that had to be budgeted for.
    """

    length: float
    radius: float
    control: tuple[float, ...]
    n_poly: int = N_POLY_DEFAULT          # retained only for reporting sample tables
    r0_over_r: float = 0.0

    def __post_init__(self) -> None:
        if self.length <= 0.0:
            raise ValueError(f"length must be positive, got {self.length}")
        if self.radius <= 0.0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        if len(self.control) < DEGREE + 1:
            raise ValueError(
                f"need at least {DEGREE + 1} control values, got {len(self.control)}"
            )

    @property
    def n_ctrl(self) -> int:
        return len(self.control)

    # ---- the control polygon, which is exactly what the notebook emits ------------------

    def control_points(self, x0: float = 0.0) -> tuple[tuple[float, float], ...]:
        """`(x, y)` control points in metres. THE notebook input, not a sampled curve.

        `ntopgen` emits precisely these as `point<real,real,real>` blocks, so this function and
        the notebook cannot describe different geometry.
        """
        r0 = self.r0_over_r * self.radius
        dr = self.radius - r0
        return tuple(
            (x0 + f * self.length, r0 + dr * c)
            for f, c in zip(station_fractions(self.n_ctrl), self.control)
        )

    # ---- evaluation --------------------------------------------------------------------

    def _xy_coeffs(self, x0: float = 0.0) -> tuple[tuple[float, ...], tuple[float, ...]]:
        pts = self.control_points(x0)
        return tuple(p[0] for p in pts), tuple(p[1] for p in pts)

    def point_at(self, u: float, x0: float = 0.0) -> tuple[float, float]:
        """`(x, y)` on the curve at parameter `u` in [0, 1]."""
        cx, cy = self._xy_coeffs(x0)
        kv = clamped_knots(self.n_ctrl)
        return _de_boor(u, kv, cx, DEGREE), _de_boor(u, kv, cy, DEGREE)

    def _spans(self) -> tuple[tuple[float, float], ...]:
        """Distinct knot intervals, i.e. the polynomial pieces."""
        kv = clamped_knots(self.n_ctrl)
        breaks = sorted(set(kv))
        return tuple((a, b) for a, b in zip(breaks, breaks[1:]))

    def _derivs(self, x0: float = 0.0):
        cx, cy = self._xy_coeffs(x0)
        qx, kvx = derivative_control(cx)
        qy, kvy = derivative_control(cy)
        return (
            lambda u: _eval_on(u, kvx, qx, DEGREE - 1),
            lambda u: _eval_on(u, kvy, qy, DEGREE - 1),
        )

    def sample(self, n: int | None = None, x0: float = 0.0):
        """`(x, y)` samples along the curve. For plotting and for area tables only."""
        n = int(n or self.n_poly)
        cx, cy = self._xy_coeffs(x0)
        kv = clamped_knots(self.n_ctrl)
        return tuple(
            (_de_boor(i / (n - 1), kv, cx, DEGREE), _de_boor(i / (n - 1), kv, cy, DEGREE))
            for i in range(n)
        )

    # ---- exact closed forms ------------------------------------------------------------

    def volume(self) -> float:
        """Enclosed volume of the revolved spline, m^3. EXACT.

        `V = int pi y(u)^2 x'(u) du`. The integrand is a polynomial of degree `3p - 1 = 8` per
        span for cubics, and the 8-point Gauss rule is exact to degree 15, so this carries no
        quadrature error beyond floating point.
        """
        cy = self._xy_coeffs()[1]
        kv = clamped_knots(self.n_ctrl)
        dx, _ = self._derivs()
        f = lambda u: math.pi * _de_boor(u, kv, cy, DEGREE) ** 2 * dx(u)
        return abs(_integrate(f, self._spans()))

    def lateral_area(self) -> float:
        """Lateral (wetted) area of the revolved spline, m^2.

        `A = int 2 pi y(u) sqrt(x'^2 + y'^2) du`. The square root is not polynomial, so the
        Gauss rule is refined per span; 16 subdivisions puts the residual below 1e-12 relative
        on the shapes used here. EXCLUDES both end discs.
        """
        cy = self._xy_coeffs()[1]
        kv = clamped_knots(self.n_ctrl)
        dx, dy = self._derivs()
        f = lambda u: 2.0 * math.pi * _de_boor(u, kv, cy, DEGREE) * math.hypot(dx(u), dy(u))
        return abs(_integrate(f, self._spans(), refine=16))

    def planform_area_and_centroid(self) -> tuple[float, float]:
        """`(planform area, centroid station from the run start)`, m^2 and m. EXACT.

        Planform of a body of revolution is `int 2 y dx`, and its first moment `int 2 y x dx`.
        Both integrands are polynomials, so both are exact under the Gauss rule.
        """
        cx, cy = self._xy_coeffs()
        kv = clamped_knots(self.n_ctrl)
        dx, _ = self._derivs()
        area = abs(_integrate(
            lambda u: 2.0 * _de_boor(u, kv, cy, DEGREE) * dx(u), self._spans()))
        if area <= 0.0:
            return 0.0, 0.5 * self.length
        moment = abs(_integrate(
            lambda u: 2.0 * _de_boor(u, kv, cy, DEGREE) * _de_boor(u, kv, cx, DEGREE) * dx(u),
            self._spans()))
        return area, moment / area

    def area_distribution(self, x0: float = 0.0, n: int | None = None):
        """`(x, S(x))` cross-section areas along the run, m and m^2.

        Feeds `sizing/wavedrag.py`, which differentiates S twice, so it is reported densely
        from the analytic curve rather than at a handful of measured stations.
        """
        return tuple((x, math.pi * y * y) for x, y in self.sample(n or 201, x0))

    def max_slope(self) -> float:
        """Largest `|dy/dx|` along the curve, dimensionless.

        Slender-body theory needs a slender body; `sizing/wavedrag.py` reports this so a body
        outside the theory's range is visible rather than silent.
        """
        dx, dy = self._derivs()
        worst = 0.0
        for i in range(401):
            u = i / 400.0
            ddx = dx(u)
            if abs(ddx) > 1.0e-12:
                worst = max(worst, abs(dy(u) / ddx))
        return worst

    def is_monotone(self, tol: float = 1.0e-9) -> bool:
        """True when the radius never decreases along the run.

        A nose that bulges past its shoulder radius is not a nose, and an optimiser will find
        one if nothing stops it.
        """
        prev = None
        for i in range(401):
            _, y = self.point_at(i / 400.0)
            if prev is not None and y - prev < -tol * max(1.0, abs(self.radius)):
                return False
            prev = y
        return True
# --------------------------------------------------------------------------------------
#   Boattail
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=64)
def boattail_control_values(
    blend: float,
    n_ctrl: int = N_CTRL_DEFAULT,
) -> tuple[float, ...]:
    """Control values of a splined boattail contraction, `blend` in [0, 1].

    The run is expressed on the CONTRACTION fraction: the profile radius is
    `R - (R - r_base) * f(t)`, so `f(0) = 0` at the start of the boattail and `f(1) = 1` at
    the base rim. `SplineProfile` builds it through `r0_over_r`.

    `blend = 0` gives `f(t) = t`, the straight cone, EXACTLY. That is the degeneracy anchor:
    a splined boattail at zero blend must reproduce the conical boattail the existing
    Prandtl-Meyer boattail drag model was written for, or the comparison is not like for like.

    `blend = 1` gives a contraction that starts tangent to the cylinder and turns most sharply
    near the base. That ordering is deliberate: a boattail that turns gently at the shoulder
    and sharply at the base keeps the expansion weak where the flow is still attached, which
    is the direction real boattails are shaped. The magnitude of any drag benefit is NOT
    claimed here; `sizing/aero.py` integrates the actual contour and reports what it gets.

    The shaping curve is a cubic ease-in, `t^2 (3 - 2t)` blended against the straight line.
    That is a CHOICE OF FAMILY, not a derived optimum, and it is recorded as such in
    SOURCES["spline_boattail_family"].
    """
    b = float(blend)
    ts = [i / 200 for i in range(201)]

    def f(t: float) -> float:
        smooth = t * t * (3.0 - 2.0 * t)
        return (1.0 - b) * t + b * smooth

    fixed = {0: 0.0, n_ctrl - 1: 1.0}
    free = [i for i in range(n_ctrl) if i not in fixed]
    N = basis_matrix(ts, n_ctrl)
    y = [f(t) - sum(v * N[j][i] for i, v in fixed.items()) for j, t in enumerate(ts)]
    m = len(free)
    ata = [[sum(N[j][free[a]] * N[j][free[c]] for j in range(len(ts))) for c in range(m)]
           for a in range(m)]
    atb = [sum(N[j][free[a]] * y[j] for j in range(len(ts))) for a in range(m)]
    sol = _solve(ata, atb)
    out = [0.0] * n_ctrl
    for i, v in fixed.items():
        out[i] = v
    for a, i in enumerate(free):
        out[i] = sol[a]
    return tuple(out)
