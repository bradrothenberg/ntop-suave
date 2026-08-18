"""Tests for the spline outer mould line.

Validation strategy. The spline is a REPRESENTATION, not new physics, so the things worth
checking are: that the B-spline machinery is actually a B-spline (partition of unity, clamped
end interpolation, local support), that the closed forms describing the revolved chord polygon
are exact rather than quadratures, and above all that the spline DEGENERATES onto the tangent
ogive that the rest of the repo is validated against.

That last one is the important one. If selecting the spline path silently moved the geometry,
every banked SV-1 and IV-1 result would be invalidated and nothing would say so.
"""
from __future__ import annotations

import math

import pytest

from rocketgen import oml_spline as S
from rocketgen.sizing import aero as A
from rocketgen.sizing import masses as M


# --------------------------------------------------------------------------------------
#   Is it actually a B-spline?
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_ctrl", [7, 9, 11, 13])
@pytest.mark.parametrize("t", [0.0, 0.017, 0.25, 0.5, 0.731, 0.99, 1.0])
def test_basis_is_a_partition_of_unity(n_ctrl: int, t: float) -> None:
    """Basis functions must sum to 1 everywhere, or a control vector of ones is not radius R."""
    row = S.basis_matrix([t], n_ctrl)[0]
    assert sum(row) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("n_ctrl", [7, 9, 11])
def test_basis_is_non_negative(n_ctrl: int) -> None:
    """Negative basis values would let a convex control polygon produce a non-convex curve."""
    for i in range(201):
        for w in S.basis_matrix([i / 200.0], n_ctrl)[0]:
            assert w >= -1e-12


@pytest.mark.parametrize("n_ctrl", [7, 9, 11])
def test_clamped_ends_interpolate_the_first_and_last_control_values(n_ctrl: int) -> None:
    """This is what makes the end conditions in the module docstring hold by construction."""
    first, last = S.basis_matrix([0.0, 1.0], n_ctrl)
    assert first[0] == pytest.approx(1.0, abs=1e-12)
    assert sum(first[1:]) == pytest.approx(0.0, abs=1e-12)
    assert last[n_ctrl - 1] == pytest.approx(1.0, abs=1e-12)
    assert sum(last[:-1]) == pytest.approx(0.0, abs=1e-12)


def test_knot_vector_has_the_clamped_multiplicity() -> None:
    kv = S.clamped_knots(9, 3)
    assert kv[:4] == (0.0, 0.0, 0.0, 0.0)
    assert kv[-4:] == (1.0, 1.0, 1.0, 1.0)
    assert len(kv) == 9 + 3 + 1


def test_too_few_control_values_is_rejected() -> None:
    with pytest.raises(ValueError):
        S.clamped_knots(3, 3)


def test_radius_is_linear_in_the_control_values() -> None:
    """The property the whole nTop approach rests on.

    If this fails, the basis weights cannot be baked into the recipe as constants and every
    design point would need its own `ntopcl convert`.
    """
    n_ctrl, R, L = 9, 0.2, 1.0
    a = (0.0, 0.10, 0.30, 0.55, 0.75, 0.88, 0.97, 1.0, 1.0)
    b = (0.0, 0.15, 0.35, 0.60, 0.80, 0.92, 0.99, 1.0, 1.0)
    lam = 0.37
    mix = tuple(lam * x + (1.0 - lam) * y for x, y in zip(a, b))

    pa = S.SplineProfile(length=L, radius=R, control=a)
    pb = S.SplineProfile(length=L, radius=R, control=b)
    pm = S.SplineProfile(length=L, radius=R, control=mix)
    for i in range(51):
        u = i / 50.0
        ya, yb, ym = pa.point_at(u)[1], pb.point_at(u)[1], pm.point_at(u)[1]
        assert ym == pytest.approx(lam * ya + (1.0 - lam) * yb, abs=1e-14)


# --------------------------------------------------------------------------------------
#   Degeneracy onto the validated tangent ogive
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("f_nose", [2.0, 2.5, 3.0, 3.4, 4.0])
def test_spline_reproduces_the_tangent_ogive_profile(f_nose: float) -> None:
    """The anchor. See SOURCES["spline_ogive_degeneracy"].

    Tolerance is 5e-6 of R at 9 control points, which is three orders of magnitude inside the
    1 percent volume gate the ogive chord polygon was itself accepted under.
    """
    k = 2.0 * f_nose                       # L / R
    c = S.ogive_control_values(k, 9)
    worst = 0.0
    for i in range(1001):
        t = i / 1000.0
        row = S.basis_matrix([t], 9)[0]
        got = sum(w * v for w, v in zip(row, c))
        worst = max(worst, abs(got - S.tangent_ogive_radius(t, k)))
    assert worst < 5e-6


def test_more_control_points_fit_the_ogive_better() -> None:
    """Monotone convergence. A fit that stopped improving would mean a bug, not a limit."""
    k = 6.0
    prev = None
    for n_ctrl in (5, 7, 9, 11, 13):
        c = S.ogive_control_values(k, n_ctrl)
        worst = max(
            abs(sum(w * v for w, v in zip(S.basis_matrix([i / 500.0], n_ctrl)[0], c))
                - S.tangent_ogive_radius(i / 500.0, k))
            for i in range(501)
        )
        if prev is not None:
            assert worst < prev
        prev = worst


def test_spline_volume_matches_the_validated_ogive_quadrature() -> None:
    """Against `masses._tangent_ogive_volume`, which is itself checked against a hemisphere.

    The residual is the chord polygon being inscribed, exactly as it was for the ogive path,
    so the gate is the same 1 percent and the achieved number is far inside it.
    """
    L, R = 1.05, 0.175
    c = S.ogive_control_values(L / R, 9)
    p = S.SplineProfile(length=L, radius=R, control=c)
    ref = M._tangent_ogive_volume(L, R, n=20000)
    # No sampling allowance any more: nTop revolves this spline and the closed form
    # integrates it exactly, so the only residual is how well 9 control points fit an
    # ogive. Measured 1.7e-7 relative, against 7.9e-4 for the chord polygon this replaced.
    assert abs(p.volume() / ref - 1.0) < 1e-6


def test_spline_wetted_area_matches_the_validated_ogive_quadrature() -> None:
    L, R = 1.05, 0.175
    c = S.ogive_control_values(L / R, 9)
    p = S.SplineProfile(length=L, radius=R, control=c)
    ref = M._tangent_ogive_surface_area(L, R, n=20000)
    assert abs(p.lateral_area() / ref - 1.0) < 1e-6


def test_spline_planform_matches_the_validated_aero_quadrature() -> None:
    """`aero._nose_wetted_and_planform` drives the Barrowman build-up, so it must agree."""
    L, D = 1.05, 0.35
    c = S.ogive_control_values(L / (0.5 * D), 9)
    p = S.SplineProfile(length=L, radius=0.5 * D, control=c)
    _, plan_ref, xbar_ref = A._nose_wetted_and_planform("tangent_ogive", L, D)
    plan, xbar = p.planform_area_and_centroid()
    assert plan == pytest.approx(plan_ref, rel=2e-4)
    assert xbar == pytest.approx(xbar_ref, rel=2e-4)


def test_the_closed_forms_need_no_refinement() -> None:
    """The volume is an exact integral, so it must not depend on any sampling parameter.

    This is the property the chord-polygon version could not have. `n_poly` now only affects
    reporting tables, so changing it must leave every measured quantity bit-identical.
    """
    L, R = 1.05, 0.175
    c = S.ogive_control_values(L / R, 9)
    ref = S.SplineProfile(length=L, radius=R, control=c, n_poly=12)
    for n_poly in (24, 40, 200):
        other = S.SplineProfile(length=L, radius=R, control=c, n_poly=n_poly)
        assert other.volume() == ref.volume()
        assert other.lateral_area() == ref.lateral_area()
        assert other.planform_area_and_centroid() == ref.planform_area_and_centroid()


def test_x_of_u_is_exactly_the_axial_fraction() -> None:
    """The Greville-abscissae choice, checked directly. See `station_fractions`.

    If this drifts, the radius stops being a spline in the axial station and the fit to the
    tangent ogive degrades by three orders of magnitude (measured: 1.0e-6 -> 2.7e-3).
    """
    L = 1.05
    p = S.SplineProfile(length=L, radius=0.175, control=S.ogive_control_values(6.0, 9))
    for i in range(201):
        u = i / 200.0
        assert p.point_at(u)[0] == pytest.approx(L * u, abs=1e-12)


# --------------------------------------------------------------------------------------
#   The closed forms are exact for the revolved polygon
# --------------------------------------------------------------------------------------


def test_a_straight_taper_is_the_exact_cone() -> None:
    """Control values equal to the Greville abscissae give an exactly straight profile.

    That is the same identity `station_fractions` relies on, applied to the radius instead
    of the station, so the curve is the line y = R x / L and the solid is an exact cone.
    Every closed form must hit the textbook cone value to machine precision.
    """
    L, R = 2.0, 0.4
    c = S.station_fractions(9)                 # the straight-line control values
    p = S.SplineProfile(length=L, radius=R, control=c)
    assert p.volume() == pytest.approx(math.pi * L * R * R / 3.0, rel=1e-12)
    assert p.lateral_area() == pytest.approx(math.pi * R * math.hypot(L, R), rel=1e-12)
    plan, xbar = p.planform_area_and_centroid()
    assert plan == pytest.approx(L * R, rel=1e-12)      # triangle, base 2R, height L
    assert xbar == pytest.approx(2.0 * L / 3.0, rel=1e-12)


def test_cylinder_case_is_exact() -> None:
    """`r0_over_r = 1` makes every station radius R, so the run is a cylinder."""
    L, R = 1.5, 0.3
    p = S.SplineProfile(length=L, radius=R, control=(0.0,) * 6 + (1.0, 1.0),
                        n_poly=17, r0_over_r=1.0)
    assert p.volume() == pytest.approx(math.pi * R * R * L, rel=1e-12)
    assert p.lateral_area() == pytest.approx(2.0 * math.pi * R * L, rel=1e-12)
    plan, xbar = p.planform_area_and_centroid()
    assert plan == pytest.approx(2.0 * R * L, rel=1e-12)
    assert xbar == pytest.approx(0.5 * L, rel=1e-12)


def test_area_distribution_is_pi_r_squared_at_every_station() -> None:
    L, R = 1.05, 0.175
    p = S.SplineProfile(length=L, radius=R, control=S.ogive_control_values(6.0, 9))
    for (x, area), (xp, r) in zip(p.area_distribution(n=41), p.sample(41)):
        assert x == pytest.approx(xp, abs=1e-15)
        assert area == pytest.approx(math.pi * r * r, rel=1e-15)


def test_area_distribution_offsets_by_the_run_start() -> None:
    p = S.SplineProfile(length=1.0, radius=0.2, control=S.ogive_control_values(5.0, 9),
                        n_poly=9)
    base = p.area_distribution(0.0)
    shifted = p.area_distribution(2.5)
    for (x0, s0), (x1, s1) in zip(base, shifted):
        assert x1 == pytest.approx(x0 + 2.5, abs=1e-14)
        assert s1 == pytest.approx(s0, rel=1e-15)


# --------------------------------------------------------------------------------------
#   Shape sanity
# --------------------------------------------------------------------------------------


def test_the_ogive_equivalent_spline_is_monotone_and_does_not_bulge() -> None:
    R = 0.175
    p = S.SplineProfile(length=1.05, radius=R, control=S.ogive_control_values(6.0, 9))
    assert p.is_monotone()
    assert max(p.point_at(i / 200.0)[1] for i in range(201)) <= R * (1.0 + 1e-12)


def test_von_karman_reference_profile_matches_its_definition() -> None:
    """Endpoints and the known quarter-length value of the LD-Haack profile."""
    assert S.von_karman_radius(0.0) == pytest.approx(0.0, abs=1e-12)
    assert S.von_karman_radius(1.0) == pytest.approx(1.0, abs=1e-12)
    # theta = arccos(1 - 2t); at t = 0.5, theta = pi/2 and r/R = sqrt(1/2)
    assert S.von_karman_radius(0.5) == pytest.approx(math.sqrt(0.5), rel=1e-12)


def test_von_karman_is_blunter_than_the_tangent_ogive_at_the_tip() -> None:
    """The physical reason the spline cannot close the last 14 percent of the drag gap.

    The von Karman profile has infinite tip slope (r ~ x^(1/4)); a clamped cubic spline has a
    finite one. So the spline must fall short near t = 0, and the test pins that it is the
    von Karman shape that is fatter there, not the other way round.
    """
    for t in (0.001, 0.01, 0.05):
        assert S.von_karman_radius(t) > S.tangent_ogive_radius(t, 6.0)


def test_rejects_degenerate_geometry() -> None:
    c = S.ogive_control_values(6.0, 9)
    with pytest.raises(ValueError):
        S.SplineProfile(length=0.0, radius=0.1, control=c)
    with pytest.raises(ValueError):
        S.SplineProfile(length=1.0, radius=-0.1, control=c)
    with pytest.raises(ValueError):
        S.SplineProfile(length=1.0, radius=0.1, control=(0.0, 1.0))


def test_control_values_for_rejects_an_unknown_shape() -> None:
    with pytest.raises(ValueError):
        S.control_values_for("parabolic", 6.0)


def test_sources_are_registered_and_flag_no_hidden_guesses() -> None:
    """Every constant here is measured or cited. If one becomes a guess it must say so."""
    from rocketgen.config import SOURCES

    for key in S.SOURCES:
        assert key in SOURCES
    # the module claims measurements; make sure none of them silently became a guess
    assert not any("GUESS" in v for v in S.SOURCES.values())


# --------------------------------------------------------------------------------------
#   The study drivers must actually select the spline
# --------------------------------------------------------------------------------------


def test_design_vectors_report_a_spline_nose_only_when_asked() -> None:
    """`nose_control` is the single source of truth every consumer reads.

    This exists because of a real defect. `scripts/iv1_converge.py --oml spline` once changed
    only the OUTPUT DIRECTORY: the code that switched the shape had failed to apply, so a full
    nTop run produced a tangent-ogive result inside a directory named `IV-1_spline` and every
    number in it was silently the baseline. Nothing caught it except comparing two results and
    finding them identical to 13 digits.
    """
    from rocketgen.config import DesignVector

    assert DesignVector().nose_control is None
    assert DesignVector(nose_shape="spline").nose_control is not None
    assert DesignVector().boattail_control is None
    assert DesignVector(boattail_shape="spline").boattail_control is not None


def test_iv1_design_vector_reports_a_spline_nose_only_when_asked() -> None:
    import dataclasses as dc

    from rocketgen.config_iv1 import default_iv1

    base = default_iv1()
    assert base.nose_control is None
    assert base.interstage_control is None
    assert dc.replace(base, nose_shape="spline").nose_control is not None
    assert dc.replace(base, interstage_shape="spline").interstage_control is not None


def test_the_blend_actually_changes_the_shape() -> None:
    """A blend that moved nothing would make every study result identical and look like a
    null finding rather than a broken switch."""
    from rocketgen.config import DesignVector

    a = DesignVector(nose_shape="spline", nose_blend=0.0).nose_control
    b = DesignVector(nose_shape="spline", nose_blend=1.0).nose_control
    assert a != b
    assert max(abs(x - y) for x, y in zip(a, b)) > 1e-3
