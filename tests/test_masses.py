"""Tests for the mass build-up.

Validation strategy. There is no published mass statement for the invented SV-1, so these tests
check the three things that can be checked: closed-form geometry against exact analytic results,
conservation and bookkeeping identities, and agreement with published correlation bands.
"""
from __future__ import annotations

import math

import pytest

from rocketgen.config import MATERIALS, DesignVector, Requirements
from rocketgen.sizing import masses as M


# --------------------------------------------------------------------------------------
#   Geometry quadrature against exact analytic results
# --------------------------------------------------------------------------------------


def test_ogive_quadrature_degenerates_to_a_hemisphere():
    """A tangent ogive with L == R is exactly a hemisphere.

    Generating radius rho = (R^2 + R^2)/(2R) = R, so the profile is the circle of radius R.
    Surface area must be 2*pi*R^2 and volume 2/3*pi*R^3.
    """
    R = 0.175
    area = M._tangent_ogive_surface_area(R, R, n=4000)
    vol = M._tangent_ogive_volume(R, R, n=4000)
    assert area == pytest.approx(2.0 * math.pi * R**2, rel=1e-3)
    assert vol == pytest.approx(2.0 / 3.0 * math.pi * R**3, rel=1e-3)


def test_ogive_quadrature_converges():
    """Refining the quadrature must change the answer by less than 0.05 percent."""
    coarse = M._tangent_ogive_volume(1.05, 0.175, n=200)
    fine = M._tangent_ogive_volume(1.05, 0.175, n=8000)
    assert abs(fine - coarse) / fine < 5e-4


def test_ogive_volume_is_between_cone_and_cylinder():
    """A tangent ogive is fatter than the cone on the same base and thinner than the cylinder."""
    L, R = 1.05, 0.175
    v = M._tangent_ogive_volume(L, R, n=2000)
    v_cone = math.pi * R * R * L / 3.0
    v_cyl = math.pi * R * R * L
    assert v_cone < v < v_cyl


def test_analytic_geometry_cylinder_terms_are_exact():
    dv = DesignVector()
    g = M.analytic_geometry(dv)
    R = 0.5 * dv.D
    assert g["area_wetted_cyl"] == pytest.approx(2.0 * math.pi * R * dv.L_body_cyl, rel=1e-12)
    assert g["volume_cyl"] == pytest.approx(math.pi * R * R * dv.L_body_cyl, rel=1e-12)
    assert g["area_base"] == pytest.approx(0.25 * math.pi * dv.d_base**2, rel=1e-12)


def test_wetted_area_scales_with_the_square_of_size():
    """Doubling every length must quadruple wetted area and octuple volume."""
    dv = DesignVector()
    big = dv.replace(
        D=2 * dv.D,
        L_total=2 * dv.L_total,
        L_boattail=2 * dv.L_boattail,
        d_base=2 * dv.d_base,
        b_fin=2 * dv.b_fin,
        c_r_fin=2 * dv.c_r_fin,
    )
    g, gb = M.analytic_geometry(dv), M.analytic_geometry(big)
    assert gb["area_wetted_body"] / g["area_wetted_body"] == pytest.approx(4.0, rel=1e-6)
    assert gb["volume_total"] / g["volume_total"] == pytest.approx(8.0, rel=1e-6)
    assert gb["area_wetted_fins"] / g["area_wetted_fins"] == pytest.approx(4.0, rel=1e-6)


# --------------------------------------------------------------------------------------
#   Motor case
# --------------------------------------------------------------------------------------


def test_case_thickness_follows_hoop_stress_when_strength_driven():
    """At a chamber pressure high enough to beat the minimum gauge, t = SF*p*R/sigma exactly."""
    dv = DesignVector(p_c=60.0e6)
    _, detail = M.motor_case_mass(dv, L_motor=2.7, material_key="motorcase_cfrp")
    R_in = 0.5 * dv.D - dv.t_wall - M.T_INSULATION
    expected = M.CASE_SF_COMPOSITE * dv.p_c * R_in / MATERIALS["motorcase_cfrp"].sigma_yield
    assert not detail["gauge_limited"]
    assert detail["t_case"] == pytest.approx(expected, rel=1e-12)


def test_case_thickness_is_floored_at_minimum_gauge():
    """At the SV-1 chamber pressure the case is gauge-driven, not strength-driven."""
    dv = DesignVector()
    _, detail = M.motor_case_mass(dv, L_motor=2.7, material_key="motorcase_cfrp")
    assert detail["gauge_limited"]
    assert detail["t_case"] == pytest.approx(M.T_CASE_MIN_COMPOSITE)
    assert detail["t_stress"] < M.T_CASE_MIN_COMPOSITE


def test_steel_case_is_heavier_than_composite():
    dv = DesignVector()
    m_cfrp, _ = M.motor_case_mass(dv, 2.7, "motorcase_cfrp")
    m_steel, _ = M.motor_case_mass(dv, 2.7, "motorcase_4130")
    assert m_steel > m_cfrp


def test_motor_case_rejects_an_impossible_wall():
    """A body too thin to hold a case must raise, not return a negative mass."""
    dv = DesignVector(D=0.010)
    with pytest.raises(ValueError):
        M.motor_case_mass(dv, 2.7)


def test_motor_mass_fraction_lands_in_the_published_band():
    """SOURCES['mass_frac_tactical_motor']: propellant mass fraction 0.80 to 0.92.

    The build-up applies the correlation floor, so the delivered motor must sit inside the band.
    """
    dv, rq = DesignVector(), Requirements()
    mb = M.build_masses(dv, rq)
    motor_inert = mb.subset(
        "Motor case",
        "Motor insulation",
        "Nozzle assembly",
        "Igniter",
        "Motor hardware not modelled",
    )
    m_prop = dv.m_p_boost + dv.m_p_sustain
    frac = m_prop / (m_prop + motor_inert)
    assert 0.80 <= frac <= 0.92, f"motor propellant mass fraction {frac:.3f} outside 0.80 to 0.92"


def test_the_correlation_shortfall_is_declared_not_hidden():
    """When the bottom-up sum is raised to the correlation floor, it must be visible."""
    dv, rq = DesignVector(), Requirements()
    mb = M.build_masses(dv, rq)
    assert any("correlation floor" in w for w in mb.warnings)
    assert any(e.name == "Motor hardware not modelled" for e in mb.entries)


# --------------------------------------------------------------------------------------
#   Bookkeeping identities
# --------------------------------------------------------------------------------------


def test_total_and_cg_are_consistent():
    dv, rq = DesignVector(), Requirements()
    mb = M.build_masses(dv, rq)
    assert mb.total == pytest.approx(sum(e.mass for e in mb.entries), rel=1e-12)
    moment = sum(e.mass * e.x_cg for e in mb.entries)
    assert mb.x_cg == pytest.approx(moment / mb.total, rel=1e-12)
    assert 0.0 < mb.x_cg < dv.L_total


def test_burnout_removes_exactly_the_propellant():
    dv, rq = DesignVector(), Requirements()
    mb = M.build_masses(dv, rq)
    m_bo, x_bo = mb.excluding("Propellant, boost", "Propellant, sustain")
    assert mb.total - m_bo == pytest.approx(dv.m_p_boost + dv.m_p_sustain, rel=1e-12)
    assert 0.0 < x_bo < dv.L_total


def test_cg_moves_forward_at_burnout():
    """Propellant sits aft of the payload, so burning it must move the CG forward.

    This matters: the forward CG shift is what erodes static margin through the burn, and
    SPEC R10 constrains static margin over the whole flight.
    """
    dv, rq = DesignVector(), Requirements()
    mb = M.build_masses(dv, rq)
    _, x_bo = mb.excluding("Propellant, boost", "Propellant, sustain")
    assert x_bo < mb.x_cg


def test_payload_masses_are_passed_through_untouched():
    """Warhead and guidance are requirements. The build-up must not adjust them."""
    dv, rq = DesignVector(), Requirements()
    mb = M.build_masses(dv, rq)
    by_name = {e.name: e for e in mb.entries}
    assert by_name["Warhead"].mass == pytest.approx(rq.m_warhead)
    assert by_name["Guidance, seeker, actuation"].mass == pytest.approx(rq.m_guidance)
    assert by_name["Warhead"].provenance == "requirement"


def test_contingency_excludes_payload_and_propellant():
    """SOURCES['contingency_margin'] applies to dry mass only."""
    dv, rq = DesignVector(), Requirements()
    mb = M.build_masses(dv, rq)
    cont = next(e for e in mb.entries if e.name == "Dry-mass contingency")
    excluded = mb.subset("Warhead", "Propellant, boost", "Propellant, sustain")
    dry = mb.total - excluded - cont.mass
    assert cont.mass == pytest.approx(M.DRY_MARGIN * dry, rel=1e-9)


def test_no_entry_has_negative_mass_or_an_off_body_station():
    dv, rq = DesignVector(), Requirements()
    mb = M.build_masses(dv, rq)
    for e in mb.entries:
        assert e.mass >= 0.0, e.name
        assert 0.0 <= e.x_cg <= dv.L_total, f"{e.name} at x = {e.x_cg}"


def test_every_entry_declares_a_known_provenance():
    dv, rq = DesignVector(), Requirements()
    mb = M.build_masses(dv, rq)
    allowed = {"ntop_measured", "analytic", "requirement", "correlation"}
    for e in mb.entries:
        assert e.provenance in allowed, f"{e.name} has provenance {e.provenance!r}"


def test_add_rejects_negative_mass():
    mb = M.MassBuildup()
    with pytest.raises(ValueError):
        mb.add("bad", -1.0, 1.0, "analytic")


# --------------------------------------------------------------------------------------
#   The nTop coupling path
# --------------------------------------------------------------------------------------


def test_ntop_measurement_overrides_the_analytic_airframe():
    """Supplying a measured structure mass must replace the analytic shell and be marked."""
    from rocketgen.config import NtopMeasurements

    dv, rq = DesignVector(), Requirements()
    analytic = M.build_masses(dv, rq)

    meas = NtopMeasurements(
        volume_total=0.335,
        volume_structure=0.0142,
        volume_cavity=0.290,
        area_wetted_body=4.01,
        mass_structure=52.0,
        cg_structure=(2.31, 0.0, 0.0),
    )
    measured = M.build_masses(dv, rq, meas=meas)

    names = {e.name for e in measured.entries}
    assert "Airframe structure and fins" in names
    assert "Airframe shell" not in names
    assert measured.measured_fraction > 0.0
    assert analytic.measured_fraction == 0.0
    assert not any("airframe mass is analytic" in w for w in measured.warnings)
    # a heavier measured airframe must raise the launch mass
    assert measured.total > analytic.total


def test_volume_closure_fails_loudly_when_the_cavity_is_too_small():
    from rocketgen.config import NtopMeasurements

    dv, rq = DesignVector(), Requirements()
    meas = NtopMeasurements(
        volume_total=0.335,
        volume_structure=0.0142,
        volume_cavity=0.050,       # far too small for 360 kg of propellant
        area_wetted_body=4.01,
        mass_structure=52.0,
        cg_structure=(2.31, 0.0, 0.0),
    )
    mb = M.build_masses(dv, rq, meas=meas)
    assert any("volume closure FAILS" in w for w in mb.warnings)


# --------------------------------------------------------------------------------------
#   Guard rails on impossible designs
# --------------------------------------------------------------------------------------


def test_a_design_with_no_motor_room_raises():
    dv = DesignVector(L_total=3.0, L_warhead=2.0, L_seeker=0.5, L_guidance=0.4)
    with pytest.raises(ValueError):
        M.build_masses(dv, Requirements())


def test_static_margin_sign_convention():
    """Positive static margin means the centre of pressure is aft of the centre of gravity."""
    assert M.static_margin(x_cp=2.5, x_cg=2.1, D=0.35) == pytest.approx(0.4 / 0.35)
    assert M.static_margin(x_cp=2.0, x_cg=2.1, D=0.35) < 0.0


def test_every_guess_is_labelled_as_a_guess():
    """PLAN.md hard rule 2: a value that is a guess must say so in its source string."""
    for key in ("radome_thickness", "bulkhead_allowance", "case_minimum_gauge", "igniter_mass"):
        assert "GUESS" in M.SOURCES[key].upper(), key
