"""WP3a validation - solid motor model.

Reference cases used here, and where they come from:

1. Published isentropic flow tables, Purdue University School of Aeronautics and
   Astronautics, "1-D Isentropic Flow", gamma = 1.2 and gamma = 1.4
   (engineering.purdue.edu/~propulsi/propulsion/flow/, retrieved 2026-08-17):
       gamma 1.2:  M 2.00 -> A/A* 1.884, pt/p 7.530, Tt/T 1.400
                   M 2.50 -> A/A* 3.421, pt/p 18.41
                   M 3.00 -> A/A* 6.735, pt/p 47.05
                   M 3.50 -> A/A* 13.76, pt/p 121.3
                   M 4.00 -> A/A* 28.36, pt/p 308.9
       gamma 1.4:  M 2.00 -> A/A* 1.687, pt/p 7.824
                   M 3.00 -> A/A* 4.235, pt/p 36.73
                   M 4.00 -> A/A* 10.72, pt/p 151.8
   The tables are given to four significant figures, so they are checked to 0.1 %.

2. Sutton and Biblarz, "Rocket Propulsion Elements", Chapter 12, Table 12-1, row
   HTPB/AP/Al: Is 260 to 265 s at 1000 psia expanding to 14.7 psia (ideal), flame
   temperature 5700 F = 3440 K, specific gravity 1.86, burning rate 0.25 to 3.0 in/s at
   1000 psia, pressure exponent n = 0.40.

3. Sutton and Biblarz, Chapter 12, Figure 12-3: mean molecular mass of the combustion
   gases of HTPB-based composite propellant at 68 atm is in the 25 to 30 kg/kmol band for
   68 to 72 % AP. This is a figure read, so it is used as a band check only.
"""
from __future__ import annotations

import math

import pytest

from rocketgen.config import DesignVector
from rocketgen.sizing import propulsion as P

# --------------------------------------------------------------------------------------
#   Published isentropic table values
# --------------------------------------------------------------------------------------

PURDUE_GAMMA_12 = [
    # (mach, A/A*, pt/p)
    (2.00, 1.884, 7.530),
    (2.50, 3.421, 18.41),
    (3.00, 6.735, 47.05),
    (3.50, 13.76, 121.3),
    (4.00, 28.36, 308.9),
]
PURDUE_GAMMA_14 = [
    (2.00, 1.687, 7.824),
    (3.00, 4.235, 36.73),
    (4.00, 10.72, 151.8),
]


@pytest.mark.parametrize("mach,area_ratio,pressure_ratio", PURDUE_GAMMA_12)
def test_isentropic_relations_match_published_table_gamma_1p2(
    mach: float, area_ratio: float, pressure_ratio: float
) -> None:
    """Area-Mach and pressure ratio against the published gamma = 1.2 table."""
    assert P.area_ratio_from_mach(mach, 1.2) == pytest.approx(area_ratio, rel=1e-3)
    assert P.total_over_static_pressure(mach, 1.2) == pytest.approx(
        pressure_ratio, rel=1e-3
    )


@pytest.mark.parametrize("mach,area_ratio,pressure_ratio", PURDUE_GAMMA_14)
def test_isentropic_relations_match_published_table_gamma_1p4(
    mach: float, area_ratio: float, pressure_ratio: float
) -> None:
    """Area-Mach and pressure ratio against the published gamma = 1.4 table."""
    assert P.area_ratio_from_mach(mach, 1.4) == pytest.approx(area_ratio, rel=1e-3)
    assert P.total_over_static_pressure(mach, 1.4) == pytest.approx(
        pressure_ratio, rel=1e-3
    )


@pytest.mark.parametrize("mach,area_ratio,_pr", PURDUE_GAMMA_12)
def test_area_ratio_inversion_round_trips(mach: float, area_ratio: float, _pr: float) -> None:
    """The supersonic branch of the area-Mach inversion recovers the table Mach number."""
    assert P.mach_from_area_ratio(area_ratio, 1.2) == pytest.approx(mach, rel=1e-3)


def test_area_ratio_inversion_subsonic_branch() -> None:
    """The subsonic branch is a real second root, not a mislabelled supersonic one."""
    subsonic = P.mach_from_area_ratio(6.735, 1.2, supersonic=False)
    assert 0.0 < subsonic < 1.0
    assert P.area_ratio_from_mach(subsonic, 1.2) == pytest.approx(6.735, rel=1e-6)


# --------------------------------------------------------------------------------------
#   Thrust coefficient
# --------------------------------------------------------------------------------------


def test_thrust_coefficient_matches_published_table_pair() -> None:
    """C_F at a (gamma, eps) pair taken from the published table, by two routes.

    Reference pair: gamma = 1.2, eps = 6.735, for which the published table gives
    M_e = 3.00 and p_c / p_e = 47.05. Route 1 is the closed-form C_F relation. Route 2
    builds the thrust from the momentum integral using the TABLE values of exit Mach and
    exit pressure ratio, and the choked mass flow from the Vandenkerckhove function:

        F = mdot * V_e + (p_e - p_a) * A_e
        mdot   = p_c * A_t * Gamma / sqrt(R * T_c)
        V_e    = M_e * sqrt(gamma * R * T_e)

    The two routes are independent expressions of the same physics and must agree.
    """
    gamma = 1.2
    area_ratio = 6.735
    mach_exit_table = 3.00
    pressure_ratio_table = 47.05
    total_over_static_temp_table = 1.400 * 0 + (1.0 + 0.5 * (gamma - 1.0) * mach_exit_table ** 2)

    closed_form = P.thrust_coefficient(gamma, area_ratio, 0.0)

    # Route 2: momentum integral, chamber conditions chosen arbitrarily because C_F is
    # dimensionless and independent of them.
    r_specific = 300.0          # J/(kg.K), arbitrary
    t_chamber = 3000.0          # K, arbitrary
    p_chamber = 5.0e6           # Pa, arbitrary
    throat_area = 1.0e-3        # m^2, arbitrary
    exit_area = area_ratio * throat_area
    mdot = p_chamber * throat_area * P.vandenkerckhove(gamma) / math.sqrt(
        r_specific * t_chamber
    )
    t_exit = t_chamber / total_over_static_temp_table
    v_exit = mach_exit_table * math.sqrt(gamma * r_specific * t_exit)
    p_exit = p_chamber / pressure_ratio_table
    thrust_vacuum = mdot * v_exit + p_exit * exit_area
    momentum_route = thrust_vacuum / (p_chamber * throat_area)

    assert closed_form == pytest.approx(momentum_route, rel=2e-3)


def test_thrust_coefficient_pressure_term_is_exact() -> None:
    """Ambient pressure enters C_F exactly as -eps * p_a / p_c."""
    gamma, eps = 1.2, 8.0
    vacuum = P.thrust_coefficient(gamma, eps, 0.0)
    ratio = 0.0145
    assert P.thrust_coefficient(gamma, eps, ratio) == pytest.approx(
        vacuum - eps * ratio, rel=0.0, abs=1e-12
    )


def test_thrust_coefficient_is_monotone_in_area_ratio_in_vacuum() -> None:
    """In vacuum a longer nozzle always gains thrust; a basic sanity limit."""
    values = [P.thrust_coefficient(1.2, eps, 0.0) for eps in (2.0, 4.0, 8.0, 16.0, 40.0)]
    assert all(b > a for a, b in zip(values, values[1:]))


# --------------------------------------------------------------------------------------
#   Propellant performance derivation and cross-check
# --------------------------------------------------------------------------------------


def test_reference_isp_is_the_cited_sutton_value() -> None:
    """The model's reference Isp is the mid-range of Sutton Table 12-1, row HTPB/AP/Al."""
    assert 260.0 <= P.ISP_REFERENCE_S <= 265.0


def test_c_star_reproduces_the_cited_reference_isp() -> None:
    """Isp at the Table 12-1 reference conditions, rebuilt from c*, matches the citation.

    Round trip: c* was derived as Isp_ref * g0 / C_F_ref, so C_F_ref * c* / g0 must give
    Isp_ref back exactly. This checks the derivation is self-consistent and that the
    reference area ratio really is the optimum-expansion one.
    """
    c_star = P.characteristic_velocity()
    c_f_ref = P.reference_thrust_coefficient()
    assert c_f_ref * c_star / P.G0 == pytest.approx(P.ISP_REFERENCE_S, rel=1e-12)
    # Sanity band for aluminized AP/HTPB: c* of order 1.5 to 1.7 km/s.
    assert 1500.0 < c_star < 1700.0


def test_implied_exhaust_molar_mass_agrees_with_sutton_figure_12_3() -> None:
    """The assumed gamma is cross-checked through the cited flame temperature.

    With gamma = 1.20 and the cited flame temperature of 3440 K, the derived c* implies a
    mean exhaust molar mass. Sutton Figure 12-3 puts that quantity in the 25 to
    30 kg/kmol band for HTPB-based composite propellant at 68 atm. Falling inside the
    band is the only independent evidence for the assumed gamma, so this test is the
    justification for SOURCES['prop.gamma'].
    """
    check = P.propellant_cross_check()
    assert 25.0 <= check["implied_molar_mass"] <= 30.0


def test_reference_area_ratio_is_optimum_expansion() -> None:
    """The reference area ratio really expands 1000 psia to 14.7 psia."""
    gamma = P.GAMMA_EXHAUST
    mach = P.mach_from_pressure_ratio(P.P_C_REFERENCE / P.P_E_REFERENCE, gamma)
    eps = P.area_ratio_from_mach(mach, gamma)
    assert P.exit_pressure_ratio(eps, gamma) == pytest.approx(
        P.P_E_REFERENCE / P.P_C_REFERENCE, rel=1e-9
    )


def test_burn_rate_law_matches_the_cited_reference_point() -> None:
    """r = a * p^n reproduces 10.0 mm/s at 1000 psia with the cited exponent n = 0.40."""
    assert P.BURN_RATE_EXPONENT_N == pytest.approx(0.40)
    assert P.burn_rate(P.P_C_REFERENCE) == pytest.approx(0.0100, rel=1e-12)
    # Inside the Sutton Table 12-1 range 0.25 to 3.0 in/s at 1000 psia.
    rate_ips = P.burn_rate(P.P_C_REFERENCE) / P.INCH_TO_M
    assert 0.25 <= rate_ips <= 3.0


def test_kn_closure_is_invertible() -> None:
    """chamber_pressure_from_kn and burning_area_from_chamber_pressure are inverses."""
    rho, c_star = 1800.0, P.characteristic_velocity()
    throat = 4.0e-3
    for p_c in (1.0e6, 3.5e6, 7.0e6, 1.4e7):
        area = P.burning_area_from_chamber_pressure(p_c, throat, rho, c_star)
        assert P.chamber_pressure_from_kn(area, throat, rho, c_star) == pytest.approx(
            p_c, rel=1e-9
        )


# --------------------------------------------------------------------------------------
#   Motor level checks
# --------------------------------------------------------------------------------------


@pytest.fixture()
def motor() -> P.SolidMotor:
    return P.SolidMotor(DesignVector())


def test_boost_thrust_matches_the_design_vector_at_sea_level(motor: P.SolidMotor) -> None:
    """The throat is sized so the boost plateau gives F_boost at sea level."""
    dv = motor.dv
    plateau_time = 0.5 * (P.T_RISE_S + motor.t_boost_end)
    assert motor.thrust(plateau_time, 0.0) == pytest.approx(dv.F_boost, rel=1e-9)


def test_sea_level_and_vacuum_thrust_differ_by_exactly_pa_times_ae(
    motor: P.SolidMotor,
) -> None:
    """Requirement: F_vac - F_sl = p_a * A_e, exactly, in every flowing phase."""
    for t in (
        0.5 * (P.T_RISE_S + motor.t_boost_end),
        0.5 * (motor.t_boost_end + motor.t_sustain_end),
        motor.t_boost_end,                       # inside the blend
        motor.t_boost_end + 0.5 * P.T_TRANSITION_S,
    ):
        exit_area = motor.exit_area_at(t)
        f_sea = motor.thrust(t, 0.0)
        # Vacuum thrust is reconstructed rather than sampled at a huge altitude, because
        # the atmosphere model clamps at the top of its table instead of going to zero.
        f_vac = f_sea + P.P_SEA_LEVEL * exit_area
        assert f_vac - f_sea == pytest.approx(P.P_SEA_LEVEL * exit_area, rel=1e-12)
        # And thrust falls with ambient pressure at exactly the slope A_e, using whichever
        # pressure function the module resolved (WP2's atmosphere or the fallback).
        f_at_11km = motor.thrust(t, 11_000.0)
        expected = f_vac - P._ambient_pressure(11_000.0) * exit_area
        assert f_at_11km == pytest.approx(expected, rel=1e-12)


def test_propellant_mass_is_conserved_by_the_ramps(motor: P.SolidMotor) -> None:
    """The integral of mdot over the burn equals the total propellant mass.

    This is the check that the ignition rise, the boost-to-sustain blend and the tail-off
    do not silently lose or gain propellant. Simpson quadrature on a fine grid; the ramps
    are piecewise linear so the residual is quadrature error only.
    """
    n = 400_001
    t_end = motor.t_burnout
    step = t_end / (n - 1)
    total = 0.0
    for i in range(n):
        weight = 1.0 if i in (0, n - 1) else (4.0 if i % 2 else 2.0)
        total += weight * motor.mdot(i * step)
    total *= step / 3.0
    assert total == pytest.approx(motor.propellant_mass, rel=2e-4)


def test_total_impulse_equals_isp_times_g0_times_propellant_mass(
    motor: P.SolidMotor,
) -> None:
    """Vacuum total impulse equals sum over ALL phases of Isp_vac * g0 * m_p.

    The ramps are mass-conserving and thrust is linear in the phase mass flows, so the
    identity is exact for this model and the "ramp-loss allowance" only has to cover the
    quadrature error of the numerical integral. Tolerance 0.05 %.
    """
    _assert_total_impulse_identity(motor)


def _assert_total_impulse_identity(motor: P.SolidMotor, samples: int = 400_001) -> None:
    """Integrate the vacuum thrust trace and compare against sum(Isp_vac * g0 * m_p)."""
    t_end = motor.t_burnout
    step = t_end / (samples - 1)
    impulse = 0.0
    for i in range(samples):
        weight = 1.0 if i in (0, samples - 1) else (4.0 if i % 2 else 2.0)
        t = i * step
        exit_area = motor.exit_area_at(t)
        # Vacuum thrust: add back the sea-level pressure term.
        impulse += weight * (motor.thrust(t, 0.0) + P.P_SEA_LEVEL * exit_area)
    impulse *= step / 3.0

    phases = motor.operating_point()
    analytic = sum(
        entry["isp_vacuum"] * P.G0 * entry["propellant_mass"] for entry in phases.values()
    )
    assert impulse == pytest.approx(analytic, rel=5e-4)
    assert motor.total_impulse_vacuum == pytest.approx(analytic, rel=1e-12)


def test_thrust_is_continuous(motor: P.SolidMotor) -> None:
    """No step in thrust anywhere, so the RK4 integrator sees a smooth right-hand side."""
    # Sample ten times finer than the shortest ramp, so a genuinely continuous trace
    # cannot change by more than about a tenth of the peak between samples.
    step = min(P.T_RISE_S, P.T_TRANSITION_S, P.T_TAILOFF_S) / 10.0
    t_end = motor.t_burnout + 0.5
    samples = int(t_end / step) + 1
    previous = motor.thrust(0.0, 10_000.0)
    peak = max(motor.thrust(0.5 * (P.T_RISE_S + motor.t_boost_end), 10_000.0), 1.0)
    for i in range(1, samples + 1):
        t = min(t_end, i * step)
        current = motor.thrust(t, 10_000.0)
        assert abs(current - previous) < 0.15 * peak, f"thrust jump at t = {t:.4f} s"
        previous = current


def test_phase_labels_cover_the_whole_timeline(motor: P.SolidMotor) -> None:
    assert motor.phase(0.0) == "boost"
    assert motor.phase(0.5 * motor.t_boost_end) == "boost"
    assert motor.phase(0.5 * (motor.t_boost_end + motor.t_sustain_end)) == "sustain"
    assert motor.phase(motor.t_burnout + 1.0) == "burnout"
    assert motor.thrust(motor.t_burnout + 1.0, 0.0) == 0.0
    assert motor.mdot(motor.t_burnout + 1.0) == 0.0


def test_burn_times_are_consistent_with_mass_and_web(motor: P.SolidMotor) -> None:
    """t_b = m_p / mdot = web / r for each segment, which is the grain closure."""
    geom = motor.grain_geometry()
    phases = motor.operating_point()
    assert phases["boost"]["burn_time"] == pytest.approx(
        geom.web_boost / phases["boost"]["burn_rate"], rel=1e-9
    )
    assert phases["sustain"]["burn_time"] == pytest.approx(
        geom.web_sustain / phases["sustain"]["burn_rate"], rel=1e-9
    )
    assert phases["boost"]["burn_time"] == pytest.approx(
        motor.dv.m_p_boost / phases["boost"]["mdot"], rel=1e-12
    )


def test_grain_volume_matches_the_propellant_mass(motor: P.SolidMotor) -> None:
    """The modelled tubular and end-burner volumes hold exactly the propellant mass."""
    geom = motor.grain_geometry()
    rho = motor.propellant.density
    tube_volume = (
        0.25 * math.pi * (geom.d_outer ** 2 - geom.d_inner_boost ** 2) * geom.length_boost
    )
    face_volume = 0.25 * math.pi * geom.d_face_sustain ** 2 * geom.length_sustain
    assert tube_volume == pytest.approx(motor.dv.m_p_boost / rho, rel=1e-9)
    assert face_volume == pytest.approx(motor.dv.m_p_sustain / rho, rel=1e-9)


def test_grain_geometry_reports_l_over_d_and_bay_fit(motor: P.SolidMotor) -> None:
    geom = motor.grain_geometry()
    assert geom.L_over_D > 0.0
    assert geom.bay_length_available == pytest.approx(
        motor.dv.L_total
        - motor.dv.L_seeker
        - motor.dv.L_guidance
        - motor.dv.L_warhead
        - motor.dv.L_boattail
    )
    assert 0.0 < geom.volumetric_loading
    # An infeasible grain must SAY so, not silently pass.
    assert geom.feasible == (not geom.warnings) or bool(geom.warnings)


def test_size_sustain_for_thrust_hits_the_request() -> None:
    """The sizing hook delivers the requested sustain thrust at the cruise altitude."""
    motor = P.SolidMotor(DesignVector())
    for request in (1500.0, 2611.0, 5000.0):
        achieved = motor.size_sustain_for_thrust(request)
        assert achieved == pytest.approx(request, rel=1e-6)
        phases = motor.operating_point()
        # Burn time must follow from the propellant mass, not be an independent knob.
        assert phases["sustain"]["burn_time"] == pytest.approx(
            motor.dv.m_p_sustain / phases["sustain"]["mdot"], rel=1e-12
        )
        assert motor.t_sustain == phases["sustain"]["burn_time"]


def test_higher_sustain_thrust_shortens_the_sustain_burn() -> None:
    motor = P.SolidMotor(DesignVector())
    motor.size_sustain_for_thrust(2000.0)
    long_burn = motor.t_sustain
    motor.size_sustain_for_thrust(4000.0)
    assert motor.t_sustain < long_burn


def test_single_throat_mode_is_reported_as_infeasible() -> None:
    """One fixed throat can reach the sustain thrust only at an unusable operating point.

    This is the numerical justification for the two-position-throat modelling choice in
    SOURCES['prop.two_position_throat']. With the throat sized for a 45 kN boost, holding
    a 2.6 kN sustain forces the chamber pressure below 1 MPa and demands a burning area
    larger than the motor bay can hold. The model must say so, in warnings and in
    `grain_geometry().feasible`, not quietly return a number.
    """
    motor = P.SolidMotor(DesignVector(), two_position_throat=False)
    motor.size_sustain_for_thrust(2611.0)
    warnings = motor.warnings
    assert any("single fixed throat" in w for w in warnings)
    assert any("below" in w and "1 MPa" in w for w in warnings)
    assert motor.operating_point()["sustain"]["p_c"] < 1.0e6
    geom = motor.grain_geometry()
    assert not geom.feasible
    assert any("end-burning face diameter" in w for w in geom.warnings)


def test_separation_check_reports_an_altitude(motor: P.SolidMotor) -> None:
    """The Summerfield check returns the altitude below which the flow separates."""
    checks = motor.separation_check()
    for name in ("boost", "sustain"):
        assert checks[name]["p_a_separation"] == pytest.approx(
            checks[name]["p_e"] / P.SEPARATION_PE_OVER_PA, rel=1e-12
        )
        assert checks[name]["separation_altitude"] >= 0.0
    # The default sustain nozzle is heavily overexpanded at sea level, so the model must
    # raise a separation warning for it.
    assert any("separation" in w for w in motor.warnings)


def test_inert_mass_breakdown_is_positive_and_reports_both_routes(
    motor: P.SolidMotor,
) -> None:
    inert = motor.inert_mass_breakdown()
    for key in ("case", "nozzle", "insulation", "igniter"):
        assert inert[key] > 0.0
    assert inert["total_physics"] == pytest.approx(
        inert["case"] + inert["nozzle"] + inert["insulation"] + inert["igniter"], rel=1e-12
    )
    assert inert["correlation_min"] < inert["correlation_max"]
    assert inert["recommended"] >= inert["total_physics"]
    # Hoop stress: sigma = p * r / t at the design pressure with the safety factor.
    case_radius = 0.5 * (motor.dv.D - 2.0 * motor.dv.t_wall)
    p_design = P.CASE_SAFETY_FACTOR * 7.0e6
    expected = max(
        P.CASE_MIN_GAUGE_M, p_design * case_radius / motor.case_material.sigma_yield
    )
    assert inert["case_thickness"] == pytest.approx(expected, rel=1e-12)


def test_case_mass_scales_with_chamber_pressure() -> None:
    """Doubling p_c must roughly double the case thickness while above minimum gauge."""
    low = P.SolidMotor(DesignVector(p_c=6.0e6)).inert_mass_breakdown()
    high = P.SolidMotor(DesignVector(p_c=12.0e6)).inert_mass_breakdown()
    assert high["case_thickness"] == pytest.approx(2.0 * low["case_thickness"], rel=1e-9)


def test_summary_has_no_missing_keys(motor: P.SolidMotor) -> None:
    summary = motor.summary()
    for key in (
        "c_star",
        "isp_vacuum",
        "total_impulse_vacuum",
        "t_boost",
        "t_sustain",
        "grain_L_over_D",
        "phases",
        "inert",
        "cross_check",
        "warnings",
    ):
        assert key in summary


# --------------------------------------------------------------------------------------
#   Terminal boost pulse
# --------------------------------------------------------------------------------------
#
# Reference values captured from the validated two-phase model before the terminal phase
# was added. They are hard-coded so the terminal work cannot silently regress the motor
# that was already validated against Sutton Table 12-1 and the Purdue isentropic tables.
TWO_PHASE_REFERENCE = {
    "c_star": 1612.3420577584536,
    "C_F_vacuum": 1.7132833450430267,
    "isp_vacuum": 281.6862837023816,
    "total_impulse_vacuum": 994463.5658651857,
    "t_boost": 5.723754630625626,
    "t_sustain": 278.5928297513194,
    "t_boost_end": 5.6987546306256265,
    "t_burnout": 284.441584381945,
    "grain_L_over_D": 8.002803933183996,
    "volumetric_loading": 0.8255491785192465,
    "grain_length_total": 2.7049477294161903,
}


def _terminal_motor(
    m_p_terminal: float = 32.0,
    F_terminal: float = 8000.0,
    m_p_sustain: float = 228.0,
    **kwargs: object,
) -> P.SolidMotor:
    """A motor with a terminal pulse, with the mass traded out of the sustain charge."""
    dv = DesignVector().replace(
        m_p_terminal=m_p_terminal, F_terminal=F_terminal, m_p_sustain=m_p_sustain
    )
    motor = P.SolidMotor(dv, **kwargs)   # type: ignore[arg-type]
    motor.size_sustain_for_thrust(2600.0)
    return motor


def test_zero_terminal_propellant_reproduces_the_two_phase_reference(
    motor: P.SolidMotor,
) -> None:
    """REGRESSION GUARD. With m_p_terminal = 0 every headline number is unchanged.

    The reference values were captured from the validated two-phase model. Equality is
    exact (rel=0), not approximate, because adding a phase whose propellant mass is zero
    must not perturb a single floating-point operation in the other two.
    """
    summary = motor.summary()
    assert motor.dv.m_p_terminal == 0.0
    assert not motor.has_terminal
    for key, expected in TWO_PHASE_REFERENCE.items():
        assert summary[key] == expected, f"{key}: {summary[key]!r} != {expected!r}"
    assert motor.t_burnout_sustain == TWO_PHASE_REFERENCE["t_burnout"]
    assert motor.t_terminal == 0.0
    assert motor.propellant_mass == motor.dv.m_p_boost + motor.dv.m_p_sustain


def test_zero_terminal_propellant_leaves_the_grain_untouched(motor: P.SolidMotor) -> None:
    """Every terminal grain field is exactly zero and the total length is unchanged."""
    geom = motor.grain_geometry()
    assert geom.d_inner_terminal == 0.0
    assert geom.web_terminal == 0.0
    assert geom.length_terminal == 0.0
    assert geom.burning_area_terminal == 0.0
    assert geom.volume_terminal == 0.0
    assert geom.length_total == geom.length_boost + geom.length_sustain
    assert geom.length_total == TWO_PHASE_REFERENCE["grain_length_total"]


def test_zero_terminal_propellant_thrust_trace_is_unchanged(motor: P.SolidMotor) -> None:
    """Arming and even commanding ignition changes nothing when there is no propellant."""
    samples = [0.0, 1.0, 3.0, 5.7, 5.75, 100.0, 284.0, 284.5, 300.0]
    before = [motor.thrust(t, 5_000.0) for t in samples]
    mdot_before = [motor.mdot(t) for t in samples]
    phase_before = [motor.phase(t) for t in samples]
    assert motor.arm_terminal() is False
    assert motor.ignite_terminal(200.0) is False
    assert motor.terminal_ignition_time is None
    assert [motor.thrust(t, 5_000.0) for t in samples] == before
    assert [motor.mdot(t) for t in samples] == mdot_before
    assert [motor.phase(t) for t in samples] == phase_before
    assert motor.t_burnout == TWO_PHASE_REFERENCE["t_burnout"]
    assert "terminal" not in phase_before


def test_terminal_pulse_needs_arming_before_ignition() -> None:
    motor = _terminal_motor()
    with pytest.raises(ValueError, match="arm_terminal"):
        motor.ignite_terminal(250.0)
    assert motor.arm_terminal() is True
    assert motor.ignite_terminal(250.0) is True
    # A second command is a no-op, not a re-light.
    assert motor.ignite_terminal(260.0) is False
    assert motor.terminal_ignition_time == pytest.approx(250.0)


def test_terminal_pulse_extends_burnout_and_reports_its_phase() -> None:
    motor = _terminal_motor()
    assert motor.has_terminal
    assert motor.t_burnout == motor.t_burnout_sustain   # not lit yet
    assert motor.phase(motor.t_burnout_sustain + 1.0) == "burnout"

    ignition = motor.t_burnout_sustain + 20.0
    motor.arm_terminal()
    motor.ignite_terminal(ignition)
    assert motor.t_burnout > motor.t_burnout_sustain
    assert motor.t_burnout == pytest.approx(
        ignition + motor.t_terminal + 0.5 * P.T_RISE_S + 0.5 * P.T_TAILOFF_S
    )
    assert motor.phase(ignition - 1.0) == "burnout"
    assert motor.phase(ignition + 0.5 * motor.t_terminal) == "terminal"
    assert motor.phase(motor.t_burnout + 1.0) == "burnout"
    assert motor.thrust(ignition + 0.5 * motor.t_terminal, 1000.0) > 0.0
    assert motor.thrust(motor.t_burnout + 1.0, 1000.0) == 0.0


def test_terminal_pulse_conserves_propellant_mass() -> None:
    """The integral of mdot over the whole timeline equals all three charges."""
    motor = _terminal_motor()
    motor.arm_terminal()
    motor.ignite_terminal(motor.t_burnout_sustain + 20.0)
    samples = 800_001
    step = motor.t_burnout / (samples - 1)
    total = 0.0
    for i in range(samples):
        weight = 1.0 if i in (0, samples - 1) else (4.0 if i % 2 else 2.0)
        total += weight * motor.mdot(i * step)
    total *= step / 3.0
    assert total == pytest.approx(motor.propellant_mass, rel=5e-4)
    assert motor.propellant_mass == pytest.approx(
        motor.dv.m_p_boost + motor.dv.m_p_sustain + motor.dv.m_p_terminal
    )


def test_total_impulse_covers_all_three_phases() -> None:
    """Total impulse identity still holds with the terminal pulse burning."""
    motor = _terminal_motor()
    motor.arm_terminal()
    motor.ignite_terminal(motor.t_burnout_sustain + 20.0)
    _assert_total_impulse_identity(motor, samples=800_001)
    assert motor.operating_point()["terminal"]["propellant_mass"] == 32.0


def test_terminal_thrust_trace_is_continuous() -> None:
    motor = _terminal_motor()
    motor.arm_terminal()
    motor.ignite_terminal(motor.t_burnout_sustain + 20.0)
    step = min(P.T_RISE_S, P.T_TRANSITION_S, P.T_TAILOFF_S) / 10.0
    t_end = motor.t_burnout + 0.5
    peak = motor.thrust(0.5 * (P.T_RISE_S + motor.t_boost_end), 10_000.0)
    previous = motor.thrust(0.0, 10_000.0)
    for i in range(1, int(t_end / step) + 2):
        t = min(t_end, i * step)
        current = motor.thrust(t, 10_000.0)
        assert abs(current - previous) < 0.15 * peak, f"thrust jump at t = {t:.4f} s"
        previous = current


def test_size_terminal_for_thrust_hits_the_request() -> None:
    motor = _terminal_motor(F_terminal=0.0)
    for request in (3000.0, 8000.0, 12000.0):
        achieved = motor.size_terminal_for_thrust(request)
        assert achieved == pytest.approx(request, rel=1e-6)
        entry = motor.operating_point()["terminal"]
        assert entry["burn_time"] == pytest.approx(
            motor.m_p_terminal / entry["mdot"], rel=1e-12
        )
        assert motor.t_terminal == entry["burn_time"]


def test_design_vector_f_terminal_is_honoured_at_construction() -> None:
    """DesignVector.F_terminal sizes the pulse without any extra call from the loop."""
    motor = _terminal_motor(F_terminal=9500.0)
    entry = motor.operating_point()["terminal"]
    sea_level = entry["thrust_vacuum"] - P.P_SEA_LEVEL * entry["exit_area"]
    assert sea_level == pytest.approx(9500.0, rel=1e-6)


def test_higher_terminal_thrust_shortens_the_terminal_burn() -> None:
    low = _terminal_motor(F_terminal=4000.0).t_terminal
    high = _terminal_motor(F_terminal=12000.0).t_terminal
    assert high < low


def test_terminal_grain_counts_towards_length_and_l_over_d() -> None:
    """SPEC.md section 4 constrains grain L/D, so the terminal segment must be in it."""
    motor = _terminal_motor()
    geom = motor.grain_geometry()
    assert geom.length_terminal > 0.0
    assert geom.length_total == pytest.approx(
        geom.length_boost + geom.length_sustain + geom.length_terminal
    )
    assert geom.L_over_D == pytest.approx(geom.length_total / geom.d_outer)
    # Volume closure: the tubular terminal grain holds exactly its propellant mass.
    tube_volume = (
        0.25
        * math.pi
        * (geom.d_outer ** 2 - geom.d_inner_terminal ** 2)
        * geom.length_terminal
    )
    assert tube_volume == pytest.approx(
        motor.m_p_terminal / motor.propellant.density, rel=1e-9
    )
    assert geom.volume_total == pytest.approx(
        motor.propellant_mass / motor.propellant.density, rel=1e-12
    )


def test_terminal_grain_reports_an_impossible_web() -> None:
    """Too much terminal propellant for the burning area must be flagged, not hidden."""
    motor = _terminal_motor(m_p_terminal=200.0, F_terminal=3000.0, m_p_sustain=100.0)
    geom = motor.grain_geometry()
    assert not geom.feasible
    assert any("terminal web" in w for w in geom.warnings)


def test_terminal_shares_a_throat_and_adds_no_new_transition() -> None:
    """The default terminal throat source introduces no extra hardware transition."""
    motor = _terminal_motor()
    report = motor.throat_transition_report()
    assert [(t["from"], t["to"]) for t in report] == [
        ("boost", "sustain"),
        ("sustain", "terminal"),
    ]
    sustain_to_terminal = report[1]
    assert sustain_to_terminal["direction"] == "unchanged"
    assert sustain_to_terminal["credible"] is True


def test_boost_throat_source_gives_a_credible_increase_but_a_longer_grain() -> None:
    """The sustain-to-boost throat change is the one an ejectable insert can do."""
    shared = _terminal_motor(terminal_throat_source="sustain")
    from_boost = _terminal_motor(terminal_throat_source="boost")
    report = from_boost.throat_transition_report()
    assert report[1]["direction"] == "increase"
    assert report[1]["credible"] is True
    # It costs grain length, which is why "sustain" is the default.
    assert (
        from_boost.grain_geometry().length_terminal
        > shared.grain_geometry().length_terminal
    )


def test_boost_to_sustain_throat_change_is_reported_as_not_credible(
    motor: P.SolidMotor,
) -> None:
    """The honest hardware statement must be machine-readable, not just prose.

    The boost throat is larger than the sustain throat, so the transition needs the throat
    to shrink. No ejectable insert does that. See
    SOURCES['prop.throat_transition_credibility'].
    """
    report = motor.throat_transition_report()
    assert len(report) == 1
    assert report[0]["from"] == "boost" and report[0]["to"] == "sustain"
    assert report[0]["area_from"] > report[0]["area_to"]
    assert report[0]["direction"] == "decrease"
    assert report[0]["credible"] is False
    assert any("throat area decreases" in w for w in motor.warnings)


def test_case_pays_for_the_terminal_chamber_pressure() -> None:
    """A high-pressure terminal pulse must thicken the case, not ride for free."""
    low = _terminal_motor(F_terminal=3000.0)
    high = _terminal_motor(F_terminal=14000.0)
    assert (
        high.operating_point()["terminal"]["p_c"]
        > high.operating_point()["boost"]["p_c"]
    )
    assert (
        high.inert_mass_breakdown()["case_thickness"]
        > low.inert_mass_breakdown()["case_thickness"]
    )


def test_terminal_pulse_appears_in_the_separation_check() -> None:
    motor = _terminal_motor()
    checks = motor.separation_check()
    assert set(checks) == {"boost", "sustain", "terminal"}
    # Sized at sea level, so the terminal nozzle is close to optimum there.
    assert checks["terminal"]["separation_altitude"] < 1000.0


def test_every_source_string_is_populated() -> None:
    """PLAN.md hard rule 2: every constant is declared with a source."""
    for key, value in P.SOURCES.items():
        assert key.startswith("prop.")
        assert len(value) > 40, key
    # Values the module admits are guesses must say so.
    for key in (
        "prop.insulation_thickness",
        "prop.nozzle_mass_model",
        "prop.igniter_mass",
        "prop.motor_mass_fraction",
        "prop.gamma",
        "prop.terminal_pulse",
    ):
        assert "guess" in P.SOURCES[key].lower(), key
    for key in ("prop.throat_transition_credibility", "prop.terminal_grain"):
        assert key in P.SOURCES
