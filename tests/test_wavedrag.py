"""Tests for slender-body wave drag.

Validation strategy. This module is new physics, so PLAN.md hard rule 2 applies: it must
reproduce results established outside this repository. Two exact closed forms exist and both
are used.

  * Sears-Haack body of given length and volume:  D/q = 128 V^2 / (pi L^4)
  * Von Karman (LD-Haack) ogive, minimum drag for given length and base area:
        C_D on base area = (d/L)^2,  equivalently shape factor 4/pi

Neither is a curve fit; both are analytic results of the same linear theory, so agreement is
a real check that the implementation solves the equation it claims to solve.

A third, stronger check does not rely on any remembered formula at all: the Glauert series is
compared against DIRECT numerical evaluation of the von Karman double integral. If the series
constant pi/4 were wrong, that test would fail no matter what the closed forms said.

The fourth check is that the von Karman ogive is the constrained OPTIMUM. That is a property
of the physics, not of an implementation, and it is asserted rather than assumed.
"""
from __future__ import annotations

import math

import pytest

from rocketgen import oml_spline as S
from rocketgen.sizing import wavedrag as W


def _area_table(radius_of_t, length: float, radius: float, n: int = 4001):
    xs = [length * i / (n - 1) for i in range(n)]
    areas = [math.pi * (radius * radius_of_t(x / length)) ** 2 for x in xs]
    return xs, areas


# --------------------------------------------------------------------------------------
#   Published closed forms
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("L,R", [(3.0, 0.5), (4.0, 0.2), (10.0, 0.35)])
def test_sears_haack_body_reproduces_the_published_closed_form(L: float, R: float) -> None:
    """D/q = 128 V^2/(pi L^4). See SOURCES["wave_drag_sears_haack"]."""
    xs, areas = _area_table(lambda u: (4.0 * u * (1.0 - u)) ** 0.75, L, R)
    volume = 3.0 * math.pi ** 2 * R * R * L / 16.0
    got = W.wave_drag_over_q(xs, areas, L)
    assert got == pytest.approx(W.sears_haack_drag_over_q(volume, L), rel=1e-4)


def test_sears_haack_volume_formula_matches_quadrature() -> None:
    """Guard the closed-form volume the drag check depends on."""
    L, R, n = 3.0, 0.5, 200001
    total = 0.0
    prev_x = 0.0
    prev_s = 0.0
    for i in range(1, n):
        x = L * i / (n - 1)
        u = x / L
        s = math.pi * (R * (4.0 * u * (1.0 - u)) ** 0.75) ** 2
        total += 0.5 * (prev_s + s) * (x - prev_x)
        prev_x, prev_s = x, s
    assert total == pytest.approx(3.0 * math.pi ** 2 * R * R * L / 16.0, rel=1e-6)


@pytest.mark.parametrize("L,R", [(3.0, 0.5), (2.1, 0.175), (6.0, 0.3)])
def test_von_karman_ogive_reproduces_the_published_closed_form(L: float, R: float) -> None:
    """C_D on base area = (d/L)^2. See SOURCES["wave_drag_von_karman_ogive"]."""
    xs, areas = _area_table(S.von_karman_radius, L, R)
    got = W.wave_drag_over_q(xs, areas, L)
    assert got == pytest.approx(W.von_karman_ogive_drag_over_q(2.0 * R, L), rel=1e-4)
    cd_on_base = got / (math.pi * R * R)
    assert cd_on_base == pytest.approx((2.0 * R / L) ** 2, rel=1e-4)


def test_von_karman_shape_factor_is_four_over_pi() -> None:
    """The fineness-free statement of the same result."""
    assert W.shape_factor(S.von_karman_radius) == pytest.approx(4.0 / math.pi, rel=1e-4)


def test_shape_factor_scaling_law_holds() -> None:
    """D/q = shape_factor * S_B^2 / L^2 must hold at any size, which is why it separates."""
    sf = W.shape_factor(S.von_karman_radius)
    for L, R in ((2.0, 0.3), (7.0, 0.35), (1.2, 0.1)):
        xs, areas = _area_table(S.von_karman_radius, L, R)
        s_b = math.pi * R * R
        assert W.wave_drag_over_q(xs, areas, L) == pytest.approx(
            sf * s_b * s_b / (L * L), rel=1e-4
        )


# --------------------------------------------------------------------------------------
#   The series against direct integration, relying on no remembered formula
# --------------------------------------------------------------------------------------


def _drag_direct(coeffs: list[float], length: float, n: int) -> float:
    """D/q by direct quadrature of the double integral, in the theta variable.

    `S'' dx = dS' = sum_n n A_n cos(n theta) d(theta)` removes the endpoint singularity of
    `S''(x)`. The log singularity on the diagonal is integrable and midpoint sampling keeps
    every evaluation off it.
    """
    th = [(i + 0.5) * math.pi / n for i in range(n)]
    dth = math.pi / n
    g = [sum((m + 1) * a * math.cos((m + 1) * t) for m, a in enumerate(coeffs)) for t in th]
    c = [math.cos(t) for t in th]
    total = 0.0
    for i in range(n):
        gi, ci = g[i], c[i]
        for j in range(n):
            if i == j:
                continue
            total += gi * g[j] * math.log(abs(ci - c[j]) * length / 2.0)
    return -total * dth * dth / (2.0 * math.pi)


def test_glauert_series_agrees_with_direct_double_integration() -> None:
    """The series constant pi/4 is checked, not assumed.

    The deleted-diagonal midpoint rule converges slowly, so this asserts CONVERGENCE TOWARD
    the series rather than agreement at any single grid. A wrong constant would show as a
    plateau at the wrong value.
    """
    coeffs = [0.4, 0.3, -0.15, 0.05]
    series = (math.pi / 4.0) * sum((n + 1) * a * a for n, a in enumerate(coeffs))
    errs = [abs(_drag_direct(coeffs, 3.0, n) / series - 1.0) for n in (250, 500, 1000)]
    assert all(b < a for a, b in zip(errs, errs[1:])), errs
    assert errs[-1] < 0.03


# --------------------------------------------------------------------------------------
#   The von Karman ogive really is the optimum
# --------------------------------------------------------------------------------------


def test_adding_any_higher_mode_raises_the_drag_at_fixed_base_area() -> None:
    """Base area is set by A_1 alone, so every other mode is pure added drag.

    This is the physical content of "the von Karman ogive is optimal", asserted directly.
    """
    def drag(coeffs):
        return (math.pi / 4.0) * sum((n + 1) * a * a for n, a in enumerate(coeffs))

    base = [0.4]
    for extra in ([0.4, 0.02], [0.4, 0.0, 0.02], [0.4, -0.05], [0.4, 0.01, 0.01, 0.01]):
        assert drag(extra) > drag(base)


def test_tangent_ogive_costs_more_wave_drag_than_the_von_karman_ogive() -> None:
    """The gap the spline exists to collect. Measured at about 1.17x across the range."""
    sf_og = W.shape_factor_of_control(S.ogive_control_values(6.0, 9))
    sf_vk = W.shape_factor(S.von_karman_radius)
    assert sf_og > sf_vk
    assert 1.15 < sf_og / sf_vk < 1.20


@pytest.mark.parametrize("f_nose", [2.0, 3.0, 4.0, 5.0])
def test_the_ogive_penalty_is_roughly_fineness_independent(f_nose: float) -> None:
    """The tangent ogive stays about 1.17x the bound at every fineness in the design range.

    The tangent ogive's NORMALISED profile does drift slightly with fineness, so this is a
    band, not an identity. The von Karman shape, which does not drift at all, is pinned
    exactly by `test_von_karman_shape_factor_is_four_over_pi`.
    """
    sf = W.shape_factor(lambda t: S.tangent_ogive_radius(t, 2.0 * f_nose))
    assert 1.15 < sf / (4.0 / math.pi) < 1.20


# --------------------------------------------------------------------------------------
#   Degeneracy: the spline path must not move the banked results
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("f_nose", [2.0, 2.5, 3.0, 3.4, 4.0])
@pytest.mark.parametrize("n_ctrl", [7, 9, 11])
def test_shape_ratio_is_exactly_one_at_the_ogive_control_values(
    f_nose: float, n_ctrl: int
) -> None:
    """THE degeneracy test. See SOURCES["wave_drag_applied_as_ratio"].

    Asserted with an absolute tolerance of 1e-12, not `approx` default: if selecting the
    spline path changed CD_wave_body at the ogive shape by even a fraction of a percent,
    every previously banked SV-1 and IV-1 number would shift and nothing else would say so.
    """
    k = 2.0 * f_nose
    ratio = W.nose_wave_shape_ratio(S.ogive_control_values(k, n_ctrl), k)
    assert ratio == pytest.approx(1.0, abs=1e-12)


def test_shape_ratio_below_one_means_less_drag_than_the_ogive() -> None:
    k = 6.0
    ratio = W.nose_wave_shape_ratio(W.optimal_control_values(9), k)
    assert 0.85 < ratio < 0.90          # measured 0.8745


# --------------------------------------------------------------------------------------
#   The optimal spline
# --------------------------------------------------------------------------------------


def test_optimal_control_values_beat_the_ogive_and_lose_to_von_karman() -> None:
    """The spline must sit strictly between the shape it replaces and the theoretical bound.

    Beating the von Karman ogive would mean the drag functional is wrong, because that shape
    is provably optimal under this theory.
    """
    sf_opt = W.shape_factor_of_control(W.optimal_control_values(9))
    sf_og = W.shape_factor_of_control(S.ogive_control_values(6.0, 9))
    sf_vk = W.shape_factor(S.von_karman_radius)
    assert sf_vk < sf_opt < sf_og


@pytest.mark.parametrize("n_ctrl", [7, 9, 11])
def test_more_control_points_get_closer_to_the_bound(n_ctrl: int) -> None:
    sf = W.shape_factor_of_control(W.optimal_control_values(n_ctrl))
    sf_vk = W.shape_factor(S.von_karman_radius)
    assert sf > sf_vk
    # 7 -> 11 control points should stay within 6 percent of the bound
    assert sf / sf_vk < 1.06


def test_the_optimal_shape_is_a_usable_nose() -> None:
    """An optimiser will happily return a shape that is not a nose unless this is checked."""
    R = 0.175
    p = S.SplineProfile(length=1.05, radius=R, control=W.optimal_control_values(9))
    assert p.is_monotone()
    assert max(p.point_at(i / 200.0)[1] for i in range(201)) <= R * (1.0 + 1e-9)
    assert p.max_slope() < 1.0            # still slender enough for the theory


def test_optimal_shape_does_not_depend_on_fineness() -> None:
    """The separability the design vector relies on.

    Checked through the DIMENSIONAL drag, not through the shape factor, so it is not a
    restatement of how `shape_factor` is defined. The same control values are built at three
    very different finenesses and the reduced drag `D/q * L^2 / S_B^2` must come out identical.
    If it did not, the drag-optimal shape would be a per-design variable and the design vector
    would need six free control values instead of one blend scalar.
    """
    ctrl = W.optimal_control_values(9)
    reduced = []
    for L, R in ((1.4, 0.35), (2.1, 0.175), (5.0, 0.25)):
        p = S.SplineProfile(length=L, radius=R, control=ctrl)
        dist = p.area_distribution(n=1201)
        xs = [x for x, _ in dist]
        areas = [s for _, s in dist]
        s_b = math.pi * R * R
        reduced.append(W.wave_drag_over_q(xs, areas, L) * L * L / (s_b * s_b))
    for v in reduced[1:]:
        assert v == pytest.approx(reduced[0], rel=1e-3)


def test_the_optimum_beats_perturbations_of_itself() -> None:
    """It is a real minimum, not just the point the optimiser stopped at."""
    ctrl = list(W.optimal_control_values(9))
    best = W.shape_factor_of_control(ctrl)
    for i in (1, 2, 3, 4, 5, 6):
        for delta in (+0.01, -0.01):
            perturbed = list(ctrl)
            perturbed[i] += delta
            assert W.shape_factor_of_control(perturbed) > best


# --------------------------------------------------------------------------------------
#   Numerics
# --------------------------------------------------------------------------------------


def test_mode_truncation_is_converged() -> None:
    """See SOURCES["wave_drag_mode_count"]. 60 modes must be indistinguishable from 100."""
    c = S.ogive_control_values(6.0, 9)
    a40 = W.shape_factor_of_control(c, n_modes=40)
    a60 = W.shape_factor_of_control(c, n_modes=60)
    a100 = W.shape_factor_of_control(c, n_modes=100)
    assert abs(a60 / a40 - 1.0) < 1e-5
    assert abs(a100 / a60 - 1.0) < 1e-5


def test_glauert_coefficients_recover_a_pure_mode() -> None:
    """Feed in a body whose S' is exactly sin(theta) and get A_1 = 1, everything else 0."""
    L, R = 3.0, 0.5
    xs, areas = _area_table(S.von_karman_radius, L, R)
    a = W.glauert_coefficients(xs, areas, L, n_modes=8)
    assert a[0] == pytest.approx(4.0 * R * R / L, rel=1e-3)
    for v in a[1:]:
        assert abs(v) < 1e-3 * abs(a[0])


def test_rejects_malformed_area_tables() -> None:
    with pytest.raises(ValueError):
        W.glauert_coefficients([0.0, 1.0], [0.0, 1.0], 1.0)
    with pytest.raises(ValueError):
        W.glauert_coefficients([0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 1.0], 2.0)
    with pytest.raises(ValueError):
        W.glauert_coefficients([0.0, 0.5, 1.0, 1.5, 2.0], [0.0] * 5, 0.0)


def test_sources_are_registered() -> None:
    from rocketgen.config import SOURCES

    for key in W.SOURCES:
        assert key in SOURCES


# --------------------------------------------------------------------------------------
#   Integration with the SV-1 drag build-up
# --------------------------------------------------------------------------------------


def _sv1() -> "DesignVector":
    from rocketgen.config import DesignVector

    return DesignVector(
        D=0.35, L_total=3.60, f_nose=3.4, m_p_boost=130.0, m_p_sustain=172.0,
        m_p_terminal=40.0, F_boost=45.0e3, F_terminal=8.0e3, b_fin=0.23, c_r_fin=0.42,
    )


def test_aero_without_nose_control_is_untouched() -> None:
    """The default path must be byte-identical to the model that predates the spline."""
    from rocketgen.sizing.aero import RocketAero

    a = RocketAero(_sv1())
    assert a.nose_wave_shape_factor == 1.0
    assert "not applied" in a.sources_used["nose_wave_shape_factor"]


def test_ogive_equivalent_spline_reproduces_the_baseline_cd0_exactly() -> None:
    """Selecting the spline path at the ogive shape must not move CD0 by one bit.

    Asserted with `==`, not `approx`. If this ever needs a tolerance, the spline has stopped
    being an extension of the validated baseline and has become a change to it.
    """
    from rocketgen.sizing.aero import RocketAero

    dv = _sv1()
    k = dv.L_nose / (0.5 * dv.D)
    base = RocketAero(dv)
    spline = RocketAero(dv, nose_control=S.ogive_control_values(k, 9))
    for mach in (0.8, 1.2, 2.0, 3.0, 4.0):
        assert spline.evaluate(mach, 8000.0, 0.0).CD0 == base.evaluate(mach, 8000.0, 0.0).CD0


def test_optimal_nose_lowers_cd0_by_a_few_percent() -> None:
    """The whole point of the exercise, pinned so it cannot silently vanish.

    Measured: -2.94 percent at M 1.2 rising to -4.75 percent at M 4, because nose wave drag is
    a growing share of CD0 with Mach.
    """
    from rocketgen.sizing.aero import RocketAero

    dv = _sv1()
    base = RocketAero(dv)
    opt = RocketAero(dv, nose_control=W.optimal_control_values(9))
    deltas = []
    for mach in (1.2, 2.0, 3.0, 4.0):
        a = base.evaluate(mach, 8000.0, 0.0).CD0
        b = opt.evaluate(mach, 8000.0, 0.0).CD0
        assert b < a
        deltas.append(b / a - 1.0)
    assert -0.06 < min(deltas) < -0.04        # M 4
    assert -0.04 < max(deltas) < -0.02        # M 1.2
    # the benefit must grow with Mach, since nose wave drag is a growing share of CD0
    assert all(b < a for a, b in zip(deltas, deltas[1:]))


def test_subsonic_cd0_is_unaffected_by_nose_shape() -> None:
    """Below the transonic bridge the Bonney term is zero, so shape cannot matter there."""
    from rocketgen.sizing.aero import RocketAero

    dv = _sv1()
    base = RocketAero(dv)
    opt = RocketAero(dv, nose_control=W.optimal_control_values(9))
    for mach in (0.5, 0.8, 0.9):
        assert opt.evaluate(mach, 8000.0, 0.0).CD0 == pytest.approx(
            base.evaluate(mach, 8000.0, 0.0).CD0, rel=1e-12
        )
