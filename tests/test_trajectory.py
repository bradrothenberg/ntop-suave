"""WP3b validation - 3-DOF trajectory integrator.

The reference cases here are ANALYTIC, not published: with drag, thrust or gravity
switched off the 3-DOF equations have closed-form solutions, and those are the strongest
possible check on an integrator. Tolerances are stated per test.

  (a) vacuum ballistic       range and apogee against the closed-form parabola, 0.1 %
  (b) drag-free vertical burn burnout speed against Tsiolkovsky minus g * t_b, 0.5 %
  (c) terminal velocity      steady fall speed against sqrt(2*m*g/(rho*S*CD))
  (d) energy bookkeeping     V^2/2 + g*h conserved to 0.01 % over 100 s
  (e) RK4 order              halving dt on case (a) drops the error by about 16x

The aerodynamic model used by the mission-level tests is the stub in
`tests/simple_aero.py`. WP2 owns the real one; this file must not import
`rocketgen.sizing.aero`.
"""
from __future__ import annotations

import math
import os
import time

import pytest

from rocketgen.config import DesignVector, MassStatement, Requirements
from rocketgen.sizing import trajectory as T
from rocketgen.sizing.propulsion import SolidMotor
from tests.simple_aero import ConstantDragAero, SimpleAero

G = T.G0


# --------------------------------------------------------------------------------------
#   Force models for the analytic cases
# --------------------------------------------------------------------------------------


def _no_forces(_state: T.FlightState) -> T.Forces:
    """Vacuum, unpowered: gravity only."""
    return T.Forces(phase="ballistic")


def _constant_thrust(thrust: float, mdot: float):
    """Drag-free constant thrust along the body axis, alpha = 0."""

    def model(_state: T.FlightState) -> T.Forces:
        return T.Forces(thrust=thrust, mdot=mdot, phase="boost")

    return model


def _constant_density_drag(rho: float, area: float, cd: float):
    """Constant-density, constant-CD drag with no thrust and no normal force."""

    def model(state: T.FlightState) -> T.Forces:
        q = 0.5 * rho * state.V * state.V
        return T.Forces(drag=q * area * cd, phase="fall", extras={"q": q})

    return model


# --------------------------------------------------------------------------------------
#   (a) vacuum ballistic
# --------------------------------------------------------------------------------------

BALLISTIC_V0 = 500.0
BALLISTIC_GAMMA = math.radians(45.0)


def _fly_ballistic(dt: float) -> T.TrajectoryResult:
    integrator = T.PointMass3DOF(_no_forces)
    state0 = T.FlightState(
        t=0.0, V=BALLISTIC_V0, gamma=BALLISTIC_GAMMA, x=0.0, h=0.0, m=500.0
    )
    return integrator.integrate(state0, dt=dt, t_max=400.0, velocity_floor=0.0)


def test_vacuum_ballistic_range_matches_the_closed_form_parabola() -> None:
    """Range = V0^2 * sin(2*gamma0) / g for a drag-free launch from and to h = 0."""
    result = _fly_ballistic(0.02)
    analytic = BALLISTIC_V0 ** 2 * math.sin(2.0 * BALLISTIC_GAMMA) / G
    assert result.converged
    assert result.h[-1] == pytest.approx(0.0, abs=1e-6)
    assert result.range_final == pytest.approx(analytic, rel=1e-3)


def test_vacuum_ballistic_apogee_matches_the_closed_form_parabola() -> None:
    """Apogee = (V0 * sin(gamma0))^2 / (2*g)."""
    result = _fly_ballistic(0.02)
    analytic = (BALLISTIC_V0 * math.sin(BALLISTIC_GAMMA)) ** 2 / (2.0 * G)
    assert T.apogee(result) == pytest.approx(analytic, rel=1e-3)


def test_vacuum_ballistic_flight_time_matches_the_closed_form() -> None:
    """Time of flight = 2 * V0 * sin(gamma0) / g."""
    result = _fly_ballistic(0.02)
    analytic = 2.0 * BALLISTIC_V0 * math.sin(BALLISTIC_GAMMA) / G
    assert result.time[-1] == pytest.approx(analytic, rel=1e-3)


def test_vacuum_ballistic_speed_returns_to_launch_speed() -> None:
    """With no drag the impact speed equals the launch speed."""
    result = _fly_ballistic(0.02)
    assert result.V[-1] == pytest.approx(BALLISTIC_V0, rel=1e-6)


@pytest.mark.parametrize("gamma_deg", [15.0, 30.0, 60.0, 75.0])
def test_vacuum_ballistic_range_over_a_range_of_launch_angles(gamma_deg: float) -> None:
    gamma0 = math.radians(gamma_deg)
    integrator = T.PointMass3DOF(_no_forces)
    state0 = T.FlightState(t=0.0, V=400.0, gamma=gamma0, x=0.0, h=0.0, m=100.0)
    result = integrator.integrate(state0, dt=0.02, t_max=400.0, velocity_floor=0.0)
    analytic = 400.0 ** 2 * math.sin(2.0 * gamma0) / G
    assert result.range_final == pytest.approx(analytic, rel=1e-3)


# --------------------------------------------------------------------------------------
#   (e) RK4 order
# --------------------------------------------------------------------------------------


def test_rk4_is_fourth_order_on_the_ballistic_case() -> None:
    """Halving dt must drop the range error by about 16x.

    Large step sizes are used on purpose: at dt = 0.02 the truncation error on this case
    is already near double-precision noise, so the ratio would not be measurable. The
    accepted band is 10 to 22 because the endpoint bisection also carries an O(dt^4)
    error of its own.
    """
    analytic = BALLISTIC_V0 ** 2 * math.sin(2.0 * BALLISTIC_GAMMA) / G
    errors = []
    for dt in (0.8, 0.4, 0.2):
        result = _fly_ballistic(dt)
        errors.append(abs(result.range_final - analytic))
    assert errors[0] > 0.0
    for coarse, fine in zip(errors, errors[1:]):
        ratio = coarse / fine
        assert 10.0 < ratio < 22.0, f"order ratio {ratio:.2f} is not near 16"


# --------------------------------------------------------------------------------------
#   (b) drag-free constant-thrust vertical burn
# --------------------------------------------------------------------------------------


def test_drag_free_vertical_burn_matches_tsiolkovsky_minus_gravity_loss() -> None:
    """V(t_b) = V0 + v_e * ln(m0/m_f) - g * t_b for a vertical drag-free burn.

    Vertical flight (gamma = 90 deg) is chosen because it makes gamma_dot vanish
    identically, so the normal-force equation cannot contaminate the result and the
    speed equation is exactly the rocket equation with a constant gravity loss.
    Tolerance 0.5 %.
    """
    m0, mdot, thrust = 600.0, 12.0, 60_000.0
    burn_time = 20.0
    v_exhaust = thrust / mdot
    v0 = 100.0

    integrator = T.PointMass3DOF(_constant_thrust(thrust, mdot))
    state0 = T.FlightState(t=0.0, V=v0, gamma=0.5 * math.pi, x=0.0, h=0.0, m=m0)
    result = integrator.integrate(
        state0,
        dt=0.005,
        t_max=burn_time,
        velocity_floor=0.0,
        stop_on_ground=False,
    )
    # The integration stops on t_max, which is the burnout time here.
    assert result.time[-1] == pytest.approx(burn_time, abs=1e-9)
    mass_final = m0 - mdot * burn_time
    analytic = v0 + v_exhaust * math.log(m0 / mass_final) - G * burn_time
    assert result.V[-1] == pytest.approx(analytic, rel=5e-3)
    # Tighter than the stated tolerance in practice; guard against a silent regression.
    assert result.V[-1] == pytest.approx(analytic, rel=1e-8)
    assert result.mass[-1] == pytest.approx(mass_final, rel=1e-12)


def test_drag_free_vertical_burn_altitude_matches_the_closed_form() -> None:
    """The climb during the burn integrates the same rocket equation once more.

    h(t_b) = v0*t_b + v_e*t_b - v_e*(m0/mdot - t_b)*ln(m0/m_f)... written directly as
    the closed-form integral of the Tsiolkovsky speed profile:
        h = v0*t + v_e*t + v_e*(m0/mdot - t)*ln(m_f/m0) - 0.5*g*t^2
    """
    m0, mdot, thrust = 600.0, 12.0, 60_000.0
    burn_time = 20.0
    v_exhaust = thrust / mdot
    v0 = 100.0
    integrator = T.PointMass3DOF(_constant_thrust(thrust, mdot))
    state0 = T.FlightState(t=0.0, V=v0, gamma=0.5 * math.pi, x=0.0, h=0.0, m=m0)
    result = integrator.integrate(
        state0, dt=0.005, t_max=burn_time, velocity_floor=0.0, stop_on_ground=False
    )
    tau = m0 / mdot
    mass_ratio = (m0 - mdot * burn_time) / m0
    analytic = (
        v0 * burn_time
        + v_exhaust * burn_time
        + v_exhaust * (tau - burn_time) * math.log(mass_ratio)
        - 0.5 * G * burn_time ** 2
    )
    assert result.h[-1] == pytest.approx(analytic, rel=5e-3)


# --------------------------------------------------------------------------------------
#   (c) terminal velocity
# --------------------------------------------------------------------------------------


def test_terminal_velocity_matches_the_closed_form() -> None:
    """A vertical fall at constant density and CD settles at sqrt(2*m*g/(rho*S*CD))."""
    mass, rho, area, cd = 400.0, 1.225, 0.0962, 0.50
    analytic = math.sqrt(2.0 * mass * G / (rho * area * cd))

    integrator = T.PointMass3DOF(_constant_density_drag(rho, area, cd))
    state0 = T.FlightState(
        t=0.0, V=10.0, gamma=-0.5 * math.pi, x=0.0, h=50_000.0, m=mass
    )
    result = integrator.integrate(
        state0, dt=0.01, t_max=400.0, velocity_floor=0.0, stop_on_ground=False
    )
    assert result.V[-1] == pytest.approx(analytic, rel=1e-4)
    # And it approaches from below, monotonically.
    assert all(b >= a - 1e-9 for a, b in zip(result.V, result.V[1:]))


def test_terminal_velocity_is_approached_from_above_too() -> None:
    """Starting faster than terminal velocity, the fall decelerates to the same value."""
    mass, rho, area, cd = 400.0, 1.225, 0.0962, 0.50
    analytic = math.sqrt(2.0 * mass * G / (rho * area * cd))
    integrator = T.PointMass3DOF(_constant_density_drag(rho, area, cd))
    state0 = T.FlightState(
        t=0.0, V=3.0 * analytic, gamma=-0.5 * math.pi, x=0.0, h=50_000.0, m=mass
    )
    result = integrator.integrate(
        state0, dt=0.01, t_max=200.0, velocity_floor=0.0, stop_on_ground=False
    )
    assert result.V[-1] == pytest.approx(analytic, rel=1e-4)


# --------------------------------------------------------------------------------------
#   (d) energy bookkeeping
# --------------------------------------------------------------------------------------


def test_specific_energy_is_conserved_without_thrust_or_drag() -> None:
    """With no thrust and no drag, V^2/2 + g*h is constant to better than 0.01 %."""
    integrator = T.PointMass3DOF(_no_forces)
    state0 = T.FlightState(
        t=0.0, V=600.0, gamma=math.radians(30.0), x=0.0, h=12_000.0, m=500.0
    )
    result = integrator.integrate(
        state0, dt=0.02, t_max=100.0, velocity_floor=0.0, stop_on_ground=False
    )
    assert result.time[-1] == pytest.approx(100.0, abs=1e-9)
    energy = T.specific_energy(result)
    drift = max(abs(e - energy[0]) for e in energy) / abs(energy[0])
    assert drift < 1.0e-4, f"specific energy drifted by {drift * 100.0:.6f} %"


# --------------------------------------------------------------------------------------
#   Integrator mechanics
# --------------------------------------------------------------------------------------


def test_adaptive_mode_agrees_with_the_fixed_step_answer() -> None:
    """The adaptive integrator must reach the same ballistic range as the fixed step."""
    analytic = BALLISTIC_V0 ** 2 * math.sin(2.0 * BALLISTIC_GAMMA) / G
    integrator = T.PointMass3DOF(_no_forces)
    state0 = T.FlightState(
        t=0.0, V=BALLISTIC_V0, gamma=BALLISTIC_GAMMA, x=0.0, h=0.0, m=500.0
    )
    result = integrator.integrate(
        state0, dt=0.05, t_max=400.0, adaptive=True, tolerance=1e-9, velocity_floor=0.0
    )
    assert result.converged
    assert result.range_final == pytest.approx(analytic, rel=1e-3)
    # Adaptive mode should need far fewer steps than the fixed 0.02 s grid.
    assert len(result.time) < len(_fly_ballistic(0.02).time)


def test_t_max_termination_is_reported_as_not_converged() -> None:
    """Running out of time is not convergence, and the message must say so."""
    integrator = T.PointMass3DOF(_no_forces)
    state0 = T.FlightState(t=0.0, V=600.0, gamma=math.radians(80.0), x=0.0, h=0.0, m=500.0)
    result = integrator.integrate(state0, dt=0.02, t_max=5.0, velocity_floor=0.0)
    assert not result.converged
    assert "t_max" in result.message


def test_velocity_floor_termination() -> None:
    """Dropping below the velocity floor stops the run and counts as converged."""
    integrator = T.PointMass3DOF(_constant_density_drag(1.225, 5.0, 1.0))
    state0 = T.FlightState(t=0.0, V=100.0, gamma=0.0, x=0.0, h=20_000.0, m=10.0)
    result = integrator.integrate(
        state0, dt=0.01, t_max=200.0, velocity_floor=60.0, stop_on_ground=False
    )
    assert result.converged
    assert "velocity" in result.message
    assert result.V[-1] < 60.0


def test_every_result_list_has_the_same_length() -> None:
    """TrajectoryResult must be fully and consistently populated."""
    result = _fly_ballistic(0.05)
    lengths = {
        len(result.time),
        len(result.x),
        len(result.h),
        len(result.V),
        len(result.mach),
        len(result.mass),
        len(result.gamma),
        len(result.thrust),
        len(result.drag),
        len(result.q),
        len(result.alpha),
        len(result.phase),
        len(result.CN_required),
        len(result.alpha_limited),
    }
    assert len(lengths) == 1
    assert result.phase[0] == "ballistic"


# --------------------------------------------------------------------------------------
#   Atmosphere adapter
# --------------------------------------------------------------------------------------


def test_inline_us1976_fallback_matches_published_values() -> None:
    """The fallback atmosphere reproduces the standard US-1976 table entries.

    Reference values, U.S. Standard Atmosphere 1976 (NASA-TM-X-74335), quoted at
    GEOPOTENTIAL altitude:
      H = 0 m      T 288.15 K,  p 101325 Pa,  rho 1.2250 kg/m^3,   a 340.29 m/s
      H = 11 000 m T 216.65 K,  p 22632 Pa,   rho 0.36392 kg/m^3,  a 295.07 m/s
      H = 20 000 m T 216.65 K,  p 5474.9 Pa,  rho 0.088035 kg/m^3

    The model takes GEOMETRIC altitude, so the geopotential altitudes are converted with
    z = R * H / (R - H). This is also the sanity check WP2's atmosphere.py must pass; if
    it disagrees at 11 km by 0.12 K, it is using geometric altitude for the layer
    boundaries.
    """

    def geometric(geopotential: float) -> float:
        return T._EARTH_RADIUS * geopotential / (T._EARTH_RADIUS - geopotential)

    rho, p, temperature, sound = T._us1976(0.0)
    assert temperature == pytest.approx(288.15, rel=1e-6)
    assert p == pytest.approx(101325.0, rel=1e-6)
    assert rho == pytest.approx(1.2250, rel=1e-3)
    assert sound == pytest.approx(340.29, rel=1e-3)

    rho, p, temperature, sound = T._us1976(geometric(11_000.0))
    assert temperature == pytest.approx(216.65, rel=1e-4)
    assert p == pytest.approx(22_632.0, rel=1e-3)
    assert rho == pytest.approx(0.36392, rel=2e-3)
    assert sound == pytest.approx(295.07, rel=1e-3)

    rho, p, _T, _a = T._us1976(geometric(20_000.0))
    assert p == pytest.approx(5474.9, rel=2e-3)
    assert rho == pytest.approx(0.088035, rel=3e-3)


def test_atmosphere_source_is_recorded() -> None:
    """Whichever atmosphere is in use must be named, so integration can check it."""
    assert isinstance(T.ATMOSPHERE_SOURCE, str) and T.ATMOSPHERE_SOURCE


def test_stubs_satisfy_the_aero_protocol() -> None:
    """Both test stubs match the AeroCallable protocol the trajectory declares."""
    for stub in (SimpleAero(), ConstantDragAero()):
        assert isinstance(stub, T.AeroCallable)
        coefficients = stub.evaluate(2.0, 12_000.0, 0.05, power_on=True)
        assert coefficients.CD > 0.0
        assert stub.trim_alpha(2.0, 12_000.0, 0.0) == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
#   Mission level
# --------------------------------------------------------------------------------------


def _mass_statement(dv: DesignVector, reqs: Requirements) -> MassStatement:
    """A plausible group-weight statement for the mission tests. Not a WP5 build-up."""
    mass = MassStatement()
    mass.add("seeker", 8.0, 0.5 * dv.L_seeker)
    mass.add("guidance", reqs.m_guidance, dv.L_seeker + 0.5 * dv.L_guidance)
    mass.add("warhead", reqs.m_warhead, dv.L_seeker + dv.L_guidance + 0.5 * dv.L_warhead)
    mass.add("airframe", 90.0, 0.5 * dv.L_total)
    mass.add("motor_inert", 36.0, 0.75 * dv.L_total)
    mass.add(
        "propellant",
        dv.m_p_boost + dv.m_p_sustain + dv.m_p_terminal,
        0.70 * dv.L_total,
    )
    return mass


def _mission(
    dive_rule: str = "max_range",
    m_p_terminal: float = 0.0,
    F_terminal: float = 0.0,
    m_p_sustain: float = 260.0,
    **mission_kwargs: object,
) -> tuple[T.Mission, SolidMotor]:
    dv = DesignVector().replace(
        m_p_terminal=m_p_terminal, F_terminal=F_terminal, m_p_sustain=m_p_sustain
    )
    reqs = Requirements()
    motor = SolidMotor(dv)
    motor.size_sustain_for_thrust(2600.0)
    aero = SimpleAero()
    mass = _mass_statement(dv, reqs)
    mission = T.Mission(
        dv, reqs, motor, aero, mass, dive_rule=dive_rule, **mission_kwargs  # type: ignore[arg-type]
    )
    return mission, motor


def test_mission_flies_the_whole_profile_to_the_ground() -> None:
    mission, _motor = _mission()
    result = mission.fly(dt=0.02, t_max=900.0)
    assert result.converged, result.message
    assert result.h[-1] == pytest.approx(0.0, abs=1e-3)
    assert result.range_final > 0.0
    assert result.mass[-1] < result.mass[0]


def test_mission_visits_every_phase_in_order() -> None:
    mission, _motor = _mission()
    result = mission.fly(dt=0.02)
    seen: list[str] = []
    for phase in result.phase:
        if not seen or seen[-1] != phase:
            seen.append(phase)
    assert seen[0] == "separation"
    for expected in ("boost", "sustain", "coast", "terminal"):
        assert expected in seen, f"{expected} missing from {seen}"
    assert seen.index("boost") < seen.index("sustain") < seen.index("terminal")


def test_mission_burns_exactly_the_propellant_mass() -> None:
    """Mass at the end equals launch mass minus total propellant, to 0.2 %."""
    mission, motor = _mission()
    result = mission.fly(dt=0.02)
    burnt = result.mass[0] - result.mass[-1]
    assert burnt == pytest.approx(motor.propellant_mass, rel=2e-3)


def test_mission_records_alpha_and_required_cn() -> None:
    """The sizing loop needs both histories to check fin authority."""
    mission, _motor = _mission()
    result = mission.fly(dt=0.02)
    assert len(result.alpha) == len(result.time)
    assert len(result.CN_required) == len(result.time)
    assert max(abs(a) for a in result.alpha) > 0.0
    assert max(abs(a) for a in result.alpha) <= mission.alpha_max + 1e-12
    assert "alpha_limit_hits" in mission.diagnostics


def test_mission_holds_the_cruise_altitude_during_sustain() -> None:
    """The sustain phase must stay near h_cruise, which is what the profile asks for."""
    mission, _motor = _mission()
    result = mission.fly(dt=0.02)
    sustain_h = [
        h for h, phase in zip(result.h, result.phase) if phase == "sustain"
    ]
    assert sustain_h
    # Ignore the first few seconds while the altitude loop captures.
    settled = sustain_h[len(sustain_h) // 4:]
    assert max(abs(h - mission.reqs.h_cruise) for h in settled) < 400.0


def test_mission_reports_a_sustain_thrust_shortfall_instead_of_faking_it() -> None:
    """Ask the motor for far too little sustain thrust and check the model complains."""
    dv = DesignVector()
    reqs = Requirements()
    motor = SolidMotor(dv)
    motor.size_sustain_for_thrust(400.0)   # far below cruise drag
    mission = T.Mission(dv, reqs, motor, SimpleAero(), _mass_statement(dv, reqs))
    result = mission.fly(dt=0.02)
    assert mission.diagnostics["sustain_thrust_deficit_max"] > 100.0
    assert "could not hold constant Mach" in result.message
    # And the trajectory must show the deceleration, not a faked constant Mach.
    sustain_mach = [m for m, p in zip(result.mach, result.phase) if p == "sustain"]
    assert sustain_mach[-1] < sustain_mach[len(sustain_mach) // 4]


def test_mission_reports_the_message_and_diagnostics_honestly() -> None:
    mission, _motor = _mission()
    result = mission.fly(dt=0.02)
    assert result.message
    for key in ("t_burnout", "t_dive_entry", "alpha_limit_hits", "atmosphere_source"):
        assert key in mission.diagnostics


def test_dive_rule_range_lands_near_the_required_range() -> None:
    """The 'range' dive rule aims the dive so impact happens at Requirements.range_min."""
    mission, _motor = _mission(dive_rule="range")
    result = mission.fly(dt=0.02)
    if not math.isnan(mission.diagnostics["t_dive_entry"]):  # type: ignore[arg-type]
        assert result.range_final == pytest.approx(mission.reqs.range_min, rel=0.05)


def test_max_range_rule_flies_further_than_the_range_rule() -> None:
    range_rule, _ = _mission(dive_rule="range")
    max_rule, _ = _mission(dive_rule="max_range")
    a = range_rule.fly(dt=0.05).range_final
    b = max_rule.fly(dt=0.05).range_final
    assert b >= a - 1.0


def test_dynamic_pressure_and_mach_are_populated() -> None:
    mission, _motor = _mission()
    result = mission.fly(dt=0.02)
    assert result.q_max > 0.0
    assert max(result.mach) > 1.0
    assert all(q >= 0.0 for q in result.q)


def test_full_trajectory_runs_inside_the_time_budget() -> None:
    """Performance target: a 300 s trajectory at dt = 0.02 in under 2 s wall clock."""
    mission, _motor = _mission()
    start = time.perf_counter()
    result = mission.fly(dt=0.02, t_max=300.0)
    elapsed = time.perf_counter() - start
    assert len(result.time) > 10_000
    assert elapsed < 2.0, f"took {elapsed:.2f} s for {len(result.time)} steps"


# --------------------------------------------------------------------------------------
#   Terminal boost, and the SPEC R6 infeasibility it exists to fix
# --------------------------------------------------------------------------------------


def test_unpowered_dive_cannot_meet_spec_r6_at_any_dive_angle() -> None:
    """LOCKED FINDING. An unpowered terminal dive cannot reach Mach 1.50 at impact.

    The dive is terminal-velocity limited: at the burnout mass, sea-level density and any
    fixed drag coefficient the steady fall speed is sqrt(2*m*g/(rho*S*CD)), which for this
    airframe is subsonic. Steepening the dive only walks the impact Mach towards that
    asymptote, it never crosses it. No propellant loading changes this, because the dive
    is unpowered by definition.

    This test asserts impact Mach stays below 1.1 for dive angles from -25 to -89 deg,
    INCLUDING a near-vertical dive, so the finding cannot silently disappear from the
    suite. The fix is the terminal boost, tested below.
    """
    dv = DesignVector()
    results: list[tuple[float, float]] = []
    for gamma_deg in (-25.0, -35.0, -50.0, -70.0, -89.0):
        reqs = Requirements()
        reqs.gamma_terminal = math.radians(gamma_deg)
        motor = SolidMotor(dv)
        motor.size_sustain_for_thrust(2600.0)
        mission = T.Mission(
            dv, reqs, motor, SimpleAero(), _mass_statement(dv, reqs), dive_rule="range"
        )
        result = mission.fly(dt=0.02, adaptive=True, tolerance=1e-7)
        assert result.converged, result.message
        results.append((gamma_deg, result.mach_final))
        assert result.mach_final < 1.1, (
            f"dive at {gamma_deg:.0f} deg reached Mach {result.mach_final:.3f}; the "
            "unpowered-dive infeasibility finding no longer holds and SPEC R6 needs "
            "re-auditing"
        )
    # Steeper dives approach the terminal-velocity asymptote from below.
    machs = [m for _g, m in results]
    assert machs[-1] > machs[0]
    assert max(machs) < 1.1


def test_analytic_terminal_velocity_confirms_the_r6_infeasibility() -> None:
    """The same finding straight from the closed form, with no integration involved.

    A vertical unpowered dive settles at sqrt(2*m*g/(rho*S*CD)). At sea level
    (rho 1.225 kg/m^3, a 340.29 m/s) with the SV-1 reference area and burnout mass, that
    speed is subsonic for the calibrated drag coefficient (0.338, the value the loop uses
    after the Basic Finner calibration) and for anything draggier.

    The test also reports the drag coefficient the airframe would need for the asymptote
    to reach Mach 1.50. It comes out near 0.15, less than half the calibrated supersonic
    value, so no plausible ogive-cylinder-fin body gets there. Two burnout masses are
    checked to show the conclusion does not hinge on the mass build-up.
    """
    dv = DesignVector()
    reqs = Requirements()
    rho, _p, _T, sound = T.atmosphere_properties(0.0)
    stub_burnout = _mass_statement(dv, reqs).total_mass - dv.m_p_boost - dv.m_p_sustain
    for burnout_mass in (201.6, stub_burnout):
        for cd in (0.338, 0.45, 0.60):
            v_terminal = math.sqrt(2.0 * burnout_mass * T.G0 / (rho * dv.S_ref * cd))
            assert v_terminal / sound < 1.1, (
                f"m {burnout_mass:.1f} kg, CD {cd}: Mach {v_terminal / sound:.3f}"
            )
        # Drag coefficient that would be needed to fall at Mach 1.50.
        v_required = reqs.M_terminal_min * sound
        cd_required = (
            2.0 * burnout_mass * T.G0 / (rho * dv.S_ref * v_required ** 2)
        )
        assert cd_required < 0.20, (
            f"m {burnout_mass:.1f} kg would only need CD {cd_required:.3f} to fall at "
            "Mach 1.50; re-audit the finding"
        )


def test_zero_terminal_propellant_flies_the_identical_trajectory() -> None:
    """REGRESSION GUARD. The default vector must fly exactly as it did before.

    Compares the full trajectory of the plain default vector against one that names
    m_p_terminal = 0.0 and F_terminal = 0.0 explicitly, element by element with no
    tolerance. Adding a phase of zero propellant mass must not move a single float.
    """
    plain, _ = _mission()
    explicit, _ = _mission(m_p_terminal=0.0, F_terminal=0.0)
    a = plain.fly(dt=0.02)
    b = explicit.fly(dt=0.02)
    assert a.time == b.time
    assert a.x == b.x
    assert a.h == b.h
    assert a.V == b.V
    assert a.mach == b.mach
    assert a.mass == b.mass
    assert a.gamma == b.gamma
    assert a.thrust == b.thrust
    assert a.drag == b.drag
    assert a.q == b.q
    assert a.alpha == b.alpha
    assert a.phase == b.phase
    assert a.CN_required == b.CN_required
    assert a.converged == b.converged
    assert "terminal_boost" not in a.phase
    assert a.diagnostics["terminal_boost_lit"] is False
    assert a.diagnostics["has_terminal_boost"] is False


def test_repeated_flights_give_the_same_answer_with_a_terminal_pulse() -> None:
    """The motor's ignition time is state; fly() must reset it. The loop calls fly often."""
    mission, _motor = _mission(
        dive_rule="terminal_boost", m_p_terminal=32.0, F_terminal=8000.0, m_p_sustain=228.0
    )
    first = mission.fly(dt=0.02)
    second = mission.fly(dt=0.02)
    assert first.time == second.time
    assert first.V == second.V
    assert first.mach_final == second.mach_final
    assert first.diagnostics["t_terminal_ignition"] == pytest.approx(
        second.diagnostics["t_terminal_ignition"]
    )


def test_terminal_boost_raises_the_impact_mach_materially() -> None:
    """The fix. Terminal boost must lift impact Mach well clear of the unpowered case."""
    baseline, _ = _mission(dive_rule="terminal_boost")
    boosted, motor = _mission(
        dive_rule="terminal_boost", m_p_terminal=40.0, F_terminal=8000.0, m_p_sustain=220.0
    )
    unpowered = baseline.fly(dt=0.02)
    powered = boosted.fly(dt=0.02)
    assert unpowered.mach_final < 1.1
    assert powered.mach_final > unpowered.mach_final + 0.4
    assert powered.diagnostics["terminal_boost_lit"] is True
    assert powered.diagnostics["terminal_burn_time"] > 0.0
    assert powered.diagnostics["impact_mach"] == pytest.approx(powered.mach_final)


def test_more_terminal_propellant_gives_a_higher_impact_mach() -> None:
    machs = []
    for m_p_terminal in (0.0, 20.0, 40.0):
        mission, _motor = _mission(
            dive_rule="terminal_boost",
            m_p_terminal=m_p_terminal,
            F_terminal=8000.0 if m_p_terminal else 0.0,
            m_p_sustain=260.0 - m_p_terminal,
        )
        machs.append(mission.fly(dt=0.02, adaptive=True, tolerance=1e-7).mach_final)
    assert machs[0] < machs[1] < machs[2]


def test_enough_terminal_propellant_meets_spec_r6() -> None:
    """The headline result: with terminal boost, impact Mach 1.50 is reachable.

    32 kg of terminal propellant at 8 kN traded out of the sustain charge takes the impact
    Mach past the SPEC R6 threshold on the calibrated aero used by the loop. This test
    runs on the cheap stub instead, so it asserts the crossing happens rather than a
    specific value.
    """
    reqs = Requirements()
    mission, _motor = _mission(
        dive_rule="terminal_boost", m_p_terminal=60.0, F_terminal=8000.0, m_p_sustain=200.0
    )
    result = mission.fly(dt=0.02, adaptive=True, tolerance=1e-7)
    assert result.converged, result.message
    assert result.mach_final >= reqs.M_terminal_min, (
        f"impact Mach {result.mach_final:.3f} still short of R6"
    )


def test_terminal_boost_phase_is_recorded_in_the_phase_history() -> None:
    mission, _motor = _mission(
        dive_rule="terminal_boost", m_p_terminal=32.0, F_terminal=8000.0, m_p_sustain=228.0
    )
    result = mission.fly(dt=0.02)
    seen: list[str] = []
    for phase in result.phase:
        if not seen or seen[-1] != phase:
            seen.append(phase)
    for expected in ("separation", "boost", "sustain", "terminal", "terminal_boost"):
        assert expected in seen, f"{expected} missing from {seen}"
    assert seen.index("terminal") < seen.index("terminal_boost")
    assert seen[-1] == "terminal_boost", "the pulse should still be burning at impact"
    # Thrust must actually be non-zero during the boosted part of the dive.
    boosted_thrust = [
        f for f, p in zip(result.thrust, result.phase) if p == "terminal_boost"
    ]
    assert min(boosted_thrust) >= 0.0
    assert max(boosted_thrust) > 1000.0


def test_terminal_boost_diagnostics_are_complete() -> None:
    mission, _motor = _mission(
        dive_rule="terminal_boost", m_p_terminal=32.0, F_terminal=8000.0, m_p_sustain=228.0
    )
    result = mission.fly(dt=0.02)
    for key in (
        "has_terminal_boost",
        "terminal_boost_lit",
        "t_terminal_ignition",
        "terminal_burn_time",
        "impact_mach",
        "t_sustain_burnout",
        "t_burnout",
    ):
        assert key in mission.diagnostics
        assert key in result.diagnostics
    assert "terminal boost lit" in result.message


def test_carry_to_impact_lights_the_pulse_so_it_is_still_burning_at_impact() -> None:
    """The whole point of the carry_to_impact rule: do not burn out above the ground."""
    mission, motor = _mission(
        dive_rule="terminal_boost", m_p_terminal=32.0, F_terminal=8000.0, m_p_sustain=228.0
    )
    result = mission.fly(dt=0.02)
    ignition = float(mission.diagnostics["t_terminal_ignition"])   # type: ignore[arg-type]
    burn = float(mission.diagnostics["terminal_burn_time"])        # type: ignore[arg-type]
    impact = result.time[-1]
    assert ignition < impact
    # Still burning at impact, i.e. the burn would have ended after the ground.
    assert ignition + burn >= impact


def test_terminal_ignition_altitude_scales_with_speed_and_burn_time() -> None:
    """The dive-entry to ignition coupling is exposed, not hard-coded."""
    mission, motor = _mission(
        dive_rule="terminal_boost", m_p_terminal=32.0, F_terminal=8000.0, m_p_sustain=228.0
    )
    expected = (
        mission.terminal_ignition_margin
        * 500.0
        * abs(math.sin(mission.reqs.gamma_terminal))
        * motor.t_terminal
    )
    assert mission.terminal_ignition_altitude(500.0) == pytest.approx(expected)
    assert mission.terminal_ignition_altitude(1000.0) == pytest.approx(2.0 * expected)
    # No terminal pulse means no ignition altitude at all.
    plain, _ = _mission()
    assert plain.terminal_ignition_altitude(500.0) == 0.0


def test_larger_ignition_margin_lights_the_pulse_higher_and_earlier() -> None:
    ignition_times = []
    for margin in (0.6, 1.0, 1.4):
        mission, _motor = _mission(
            dive_rule="terminal_boost",
            m_p_terminal=32.0,
            F_terminal=8000.0,
            m_p_sustain=228.0,
            terminal_ignition_margin=margin,
        )
        mission.fly(dt=0.02, adaptive=True, tolerance=1e-7)
        ignition_times.append(float(mission.diagnostics["t_terminal_ignition"]))  # type: ignore[arg-type]
    assert ignition_times[0] > ignition_times[1] > ignition_times[2]


def test_dive_entry_ignition_rule_lights_at_dive_entry() -> None:
    mission, _motor = _mission(
        dive_rule="terminal_boost",
        m_p_terminal=32.0,
        F_terminal=8000.0,
        m_p_sustain=228.0,
        terminal_ignition_rule="dive_entry",
    )
    mission.fly(dt=0.02)
    assert mission.diagnostics["t_terminal_ignition"] == pytest.approx(
        mission.diagnostics["t_dive_entry"]
    )


def test_never_ignition_rule_leaves_the_pulse_unlit() -> None:
    """A terminal pulse that is never lit must be reported, not quietly ignored."""
    mission, _motor = _mission(
        dive_rule="terminal_boost",
        m_p_terminal=32.0,
        F_terminal=8000.0,
        m_p_sustain=228.0,
        terminal_ignition_rule="never",
    )
    result = mission.fly(dt=0.02)
    assert mission.diagnostics["terminal_boost_lit"] is False
    assert mission.diagnostics["has_terminal_boost"] is True
    assert "never lit" in result.message
    assert result.mach_final < 1.1


def test_terminal_boost_dive_rule_removes_the_dead_coast() -> None:
    """dive_rule='terminal_boost' dives at sustain burnout, with no level coast."""
    mission, _motor = _mission(dive_rule="terminal_boost")
    result = mission.fly(dt=0.02)
    dive = float(mission.diagnostics["t_dive_entry"])          # type: ignore[arg-type]
    burnout = float(mission.diagnostics["t_sustain_burnout"])  # type: ignore[arg-type]
    assert dive == pytest.approx(burnout, abs=0.05)
    assert "coast" not in result.phase or result.phase.count("coast") <= 2
    # The default max_range rule instead coasts for a long time before diving.
    lazy, _ = _mission(dive_rule="max_range")
    lazy.fly(dt=0.02)
    assert float(lazy.diagnostics["t_dive_entry"]) > dive + 10.0   # type: ignore[arg-type]


def test_invalid_rules_are_rejected() -> None:
    dv = DesignVector()
    reqs = Requirements()
    motor = SolidMotor(dv)
    mass = _mass_statement(dv, reqs)
    with pytest.raises(ValueError, match="dive_rule"):
        T.Mission(dv, reqs, motor, SimpleAero(), mass, dive_rule="nope")
    with pytest.raises(ValueError, match="terminal_ignition_rule"):
        T.Mission(dv, reqs, motor, SimpleAero(), mass, terminal_ignition_rule="nope")


def test_loop_call_signature_still_works() -> None:
    """WP5's loop.py calls exactly this sequence. It must keep working unchanged."""
    dv = DesignVector()
    reqs = Requirements()
    motor = SolidMotor(dv)
    motor.size_sustain_for_thrust(2600.0)
    mission = T.Mission(dv, reqs, motor, SimpleAero(), _mass_statement(dv, reqs))
    result = mission.fly(dt=0.02, adaptive=True, tolerance=1e-7)
    assert result.converged
    assert result.range_final > 0.0


def test_trajectory_figure_renders(tmp_path) -> None:   # type: ignore[no-untyped-def]
    """The WP3 figure module must run off a TrajectoryResult and write a PNG."""
    from rocketgen.report.fig_trajectory import plot_trajectory

    mission, _motor = _mission()
    result = mission.fly(dt=0.05)
    path = str(tmp_path / "trajectory.png")
    written = plot_trajectory(result, path=path, title="test")
    assert written == path
    assert os.path.getsize(path) > 10_000


def test_trajectory_figure_rejects_an_empty_result() -> None:
    from rocketgen.config import TrajectoryResult
    from rocketgen.report.fig_trajectory import plot_trajectory

    with pytest.raises(ValueError):
        plot_trajectory(TrajectoryResult())


def test_every_source_string_is_populated() -> None:
    """PLAN.md hard rule 2, applied to the trajectory module."""
    for key, value in T.SOURCES.items():
        assert key.startswith("traj.")
        assert len(value) > 40, key
    for key in ("traj.guidance_gains", "traj.alpha_limit"):
        assert "guess" in T.SOURCES[key].lower() or "no source" in T.SOURCES[key].lower()
