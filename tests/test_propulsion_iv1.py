"""IV-1 validation - multi-stage solid propulsion.

WHAT IS AND IS NOT VALIDATED HERE, stated plainly per CLAUDE.md section 3.2
---------------------------------------------------------------------------
No published performance table for a two-stage tactical booster stack was found in this
session, so there is NO new external data set to reproduce here. What this file does instead:

1. **It inherits the external validation.** `propulsion_iv1.MultiStageMotor` does not contain
   any thermochemistry, nozzle relation, burn-rate law or grain closure of its own: it calls the
   free functions in `rocketgen.sizing.propulsion`, which `tests/test_propulsion.py` validates
   against two published references, the Purdue University 1-D isentropic flow tables
   (gamma = 1.2 and 1.4) and Sutton and Biblarz, Rocket Propulsion Elements, Chapter 12,
   Table 12-1, row HTPB/AP/Al. `test_single_stage_stack_reproduces_the_solid_motor_*` proves the
   inheritance is real by showing a one-stage stack reproduces the validated `SolidMotor`
   BIT FOR BIT, not approximately.

2. **It checks conservation identities that are analytically known outside this repo:**
   - Total impulse against its own definition, I_t = Isp * g0 * m_p (Sutton and Biblarz,
     Chapter 2, definition of specific impulse).
   - The multi-stage rocket equation. For a stack whose exhaust velocity is constant within
     each stage, the ideal vacuum velocity gain is
         dV = sum_stage  c_stage * ln(m_initial / m_final)
     with the spent stage removed between the terms (Sutton and Biblarz, Chapter 4, multistage
     vehicles; Tsiolkovsky applied stage by stage). `test_two_stage_rocket_equation` integrates
     dV/dt = F_vac / m and dm/dt = -mdot with RK4 through both burns, the coast and the
     jettison, and compares. This is the check that the staging bookkeeping, the jettisoned
     mass and the thrust trace are mutually consistent, and it is the one that would catch a
     mass that leaves the vehicle twice or not at all.
   - Propellant closure: the integral of mdot over the whole timeline against
     `StackDesignVector.m_propellant_total`.

3. **It locks in the findings about the default stack**, which is NOT feasible: both grains are
   too long for their bays with a plain tubular geometry, stage 2 needs 151 percent volumetric
   loading, and the stage-1 nozzle exit is wider than the stage-1 body. Those are results, not
   defects in the model, and CLAUDE.md section 3.3 says they travel upward rather than being
   smoothed away. `default_iv1()` says it is unsized.

Every tolerance in this file was measured, then stated. None was widened to make a test pass.
"""
from __future__ import annotations

import math

import pytest

from rocketgen.config import DesignVector
from rocketgen.config_iv1 import (
    InterceptRequirements,
    StackDesignVector,
    StageSpec,
    default_iv1,
)
from rocketgen.sizing import propulsion as P
from rocketgen.sizing import propulsion_iv1 as PI

# --------------------------------------------------------------------------------------
#   Reference values of the DEFAULT stack, stated here so a change is visible in the diff
# --------------------------------------------------------------------------------------
#
# Captured from the model on 2026-08-17. They are not measurements of anything real: the IV-1
# requirements and the default design vector are invented (config_iv1.SOURCES["iv1_requirements"]).
# They are here as a regression lock and so that the per-stage operating point is written down in
# the test suite rather than only in a report.
DEFAULT_GRAIN_L_OVER_D = {1: 8.611019803860515, 2: 5.773945749248076}
DEFAULT_EVENTS = {
    "t_ignition_1": 0.0,
    "t_burnout_1": 5.947901897856558,
    "t_separation": 6.5479018978565575,
    "t_ignition_2": 6.5479018978565575,
    "t_all_burnout": 16.398642188854424,
}
DEFAULT_TOTAL_IMPULSE_VACUUM = 1_505_152.6512566472      # N.s
DEFAULT_JETTISONED_MASS = 47.76211488169512              # kg


@pytest.fixture()
def reqs() -> InterceptRequirements:
    return InterceptRequirements()


@pytest.fixture()
def motor(reqs: InterceptRequirements) -> PI.MultiStageMotor:
    return PI.MultiStageMotor(default_iv1(), reqs)


def _vacuum_thrust(motor: PI.MultiStageMotor, t: float) -> float:
    """Thrust in vacuum, N, reconstructed rather than sampled at a huge altitude.

    The atmosphere model clamps at the top of its table instead of going to zero, so asking for
    the thrust at 100 km would not give a vacuum value. Adding the sea-level pressure term back
    to the sea-level thrust is exact, because the ambient term is linear in p_a.
    """
    return motor.thrust(t, 0.0) + P.P_SEA_LEVEL * motor.exit_area_at(t)


def _simpson(f, t_end: float, samples: int = 400_001) -> float:
    """Composite Simpson quadrature of f on [0, t_end]. `samples` must be odd."""
    step = t_end / (samples - 1)
    total = 0.0
    for i in range(samples):
        weight = 1.0 if i in (0, samples - 1) else (4.0 if i % 2 else 2.0)
        total += weight * f(i * step)
    return total * step / 3.0


# --------------------------------------------------------------------------------------
#   1. Total impulse
# --------------------------------------------------------------------------------------


def test_total_impulse_equals_sum_of_isp_g0_mp(motor: PI.MultiStageMotor) -> None:
    """I_total = sum over stages of Isp_vac * g0 * m_p, to within the quadrature error.

    The stages have different area ratios (10 and 18 by default), so C_F_vacuum and therefore
    Isp differ per stage and the sum cannot be collapsed onto a single Isp. The identity is
    exact for this model, because the ramps conserve mass and thrust is linear in the stage mass
    flows, so the "ramp allowance" only has to cover the numerical integration.

    Measured residual of the Simpson integral against the analytic sum: 3.5e-10 relative with
    400001 samples. Asserted at 1e-8, which is nearly 30 times the measured value. The residual
    is small because mdot is piecewise linear, so Simpson is exact except at the ramp corners.
    """
    analytic = sum(
        entry["isp_vacuum"] * P.G0 * entry["propellant_mass"]
        for entry in motor.operating_point().values()
    )
    assert motor.total_impulse_vacuum() == pytest.approx(analytic, rel=1e-12)

    integrated = _simpson(lambda t: _vacuum_thrust(motor, t), motor.t_all_burnout)
    assert integrated == pytest.approx(analytic, rel=1e-8)
    assert motor.total_impulse_vacuum() == pytest.approx(
        DEFAULT_TOTAL_IMPULSE_VACUUM, rel=1e-12
    )


def test_per_stage_impulse_uses_that_stage_own_area_ratio(motor: PI.MultiStageMotor) -> None:
    """A larger area ratio must buy a larger vacuum Isp. Basic limit check on the stack."""
    stages = motor.operating_point()
    assert stages[2]["eps_nozzle"] > stages[1]["eps_nozzle"]
    assert stages[2]["isp_vacuum"] > stages[1]["isp_vacuum"]
    total = sum(entry["total_impulse_vacuum"] for entry in stages.values())
    assert total == pytest.approx(motor.total_impulse_vacuum(), rel=1e-12)


# --------------------------------------------------------------------------------------
#   2. Staging conservation
# --------------------------------------------------------------------------------------


def test_propellant_burned_over_the_whole_timeline_equals_the_design_vector(
    motor: PI.MultiStageMotor,
) -> None:
    """Integral of mdot over ignition, both burns, the coast and both tail-offs = m_propellant_total.

    This is the check that the ignition rises, the tail-offs and the separation coast neither
    lose nor gain propellant. mdot is piecewise linear, so the residual is quadrature error at
    the ramp kinks only.

    Measured residual: 3.6e-10 relative with 400001 Simpson samples. Asserted at 1e-8.
    """
    burned = _simpson(motor.mdot, motor.t_all_burnout)
    expected = motor.dv.m_propellant_total
    assert expected == pytest.approx(530.0)      # 380 + 150 by default
    assert burned == pytest.approx(expected, rel=1e-8)
    assert motor.propellant_mass == pytest.approx(expected, rel=1e-12)


def test_each_stage_burns_exactly_its_own_charge(motor: PI.MultiStageMotor) -> None:
    """Per-stage closure, so a mass error cannot cancel between stages."""
    for index in motor.stage_indices:
        t_lo = motor.t_ignition(index)
        t_hi = motor.t_burnout(index)
        span = t_hi - t_lo
        burned = _simpson(lambda u: motor.mdot(t_lo + u), span, samples=200_001)
        # Measured residuals: 1.8e-10 for stage 1 and 3.6e-10 for stage 2, at 200001 samples.
        assert burned == pytest.approx(
            motor.dv.stage_at(index).m_propellant, rel=1e-8
        ), f"stage {index}"


def test_no_propellant_flows_during_the_coast_or_after_burnout(
    motor: PI.MultiStageMotor,
) -> None:
    """The separation coast is genuinely unpowered, at every sample inside it."""
    t_lo, t_hi = motor.t_burnout(1), motor.t_separation
    assert t_hi > t_lo
    steps = 200
    for i in range(steps + 1):
        t = t_lo + (t_hi - t_lo) * i / steps
        assert motor.mdot(t) == 0.0, f"mass flow during the coast at t = {t:.4f}"
        assert motor.thrust(t, 8_000.0) == 0.0
    for t in (motor.t_all_burnout, motor.t_all_burnout + 1.0, motor.t_all_burnout + 60.0):
        assert motor.mdot(t) == 0.0
        assert motor.thrust(t, 8_000.0) == 0.0


def test_burn_times_and_webs_are_mutually_consistent(motor: PI.MultiStageMotor) -> None:
    """t_b = m_p / mdot = web / r for every stage, which is the mean-web grain closure."""
    rho = motor.propellant.density
    for index in motor.stage_indices:
        entry = motor.operating_point(index)
        geom = motor.grain_geometry(index)
        spec = motor.dv.stage_at(index)
        assert entry["burn_time"] == pytest.approx(spec.m_propellant / entry["mdot"], rel=1e-12)
        assert entry["burn_time"] == pytest.approx(
            geom.web_boost / entry["burn_rate"], rel=1e-9
        )
        # The tube holds exactly the propellant mass it is charged with.
        tube_volume = (
            0.25
            * math.pi
            * (geom.d_outer ** 2 - geom.d_inner_boost ** 2)
            * geom.length_boost
        )
        assert tube_volume == pytest.approx(spec.m_propellant / rho, rel=1e-9)
        assert geom.volume_total == pytest.approx(spec.m_propellant / rho, rel=1e-12)


# --------------------------------------------------------------------------------------
#   The multi-stage rocket equation - the one analytic reference for the STAGING layer
# --------------------------------------------------------------------------------------


def _sv1_like_stack(dv: DesignVector) -> StackDesignVector:
    """A one-stage stack whose propulsion parameters match an SV-1 `DesignVector`.

    Only the boost charge is carried across: the SV-1 sustain and terminal segments are extra
    burning geometries inside one case, which a stage does not have. Diameter, wall thickness,
    chamber pressure, area ratio, propellant mass and thrust all come straight over, and the
    sea-level sizing convention is the same for an SV-1 boost phase and for a first stage.
    """
    return StackDesignVector(
        stages=[
            StageSpec(
                index=1,
                D=dv.D,
                L=dv.L_total,
                m_propellant=dv.m_p_boost,
                F_thrust=dv.F_boost,
                t_wall=dv.t_wall,
                p_c=dv.p_c,
                eps_nozzle=dv.eps_nozzle,
                jettisoned=False,
            )
        ]
    )


def _rk4_vacuum_delta_v(
    motor: PI.MultiStageMotor,
    mass_initial: float,
    t_start: float,
    t_end: float,
    steps: int,
) -> tuple[float, float]:
    """RK4 integrate dV/dt = F_vac/m and dm/dt = -mdot on [t_start, t_end].

    Returns (delta_V, mass_final). This is a deliberately independent integration: it uses only
    `thrust`, `exit_area_at` and `mdot` through their public interfaces, so it cannot inherit an
    algebraic error from the model's own impulse bookkeeping.
    """
    h = (t_end - t_start) / steps
    mass = mass_initial
    delta_v = 0.0

    def rates(t: float, m: float) -> tuple[float, float]:
        return _vacuum_thrust(motor, t) / m, -motor.mdot(t)

    for i in range(steps):
        t = t_start + i * h
        a1, b1 = rates(t, mass)
        a2, b2 = rates(t + 0.5 * h, mass + 0.5 * h * b1)
        a3, b3 = rates(t + 0.5 * h, mass + 0.5 * h * b2)
        a4, b4 = rates(t + h, mass + h * b3)
        delta_v += h * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0
        mass += h * (b1 + 2.0 * b2 + 2.0 * b3 + b4) / 6.0
    return delta_v, mass


def test_two_stage_rocket_equation(motor: PI.MultiStageMotor, reqs: InterceptRequirements) -> None:
    """Numerically integrated vacuum delta-V matches the staged Tsiolkovsky closed form.

    Reference: Sutton and Biblarz, Rocket Propulsion Elements, Chapter 4, multistage vehicles.
    For a stage whose effective exhaust velocity c = F / mdot is constant,
        dV = c * ln(m_initial / m_final)
    and for a stack the terms add with the spent stage removed in between. c is constant within
    a stage here for a reason worth stating: thrust and mass flow are both the same normalised
    ramp shape times a constant, so their ratio is shape-independent and the identity holds
    THROUGH the ignition rise and the tail-off, not only on the plateau.

    The coast contributes nothing, and the jettison contributes nothing to delta-V but changes
    the mass the second stage has to push, which is the whole point of staging.

    Measured with 8000 RK4 steps per segment: the ideal vacuum delta-V of the default stack is
    5139.1 m/s, and the residuals against the closed form are 2.4e-8 relative on delta-V,
    1.4e-8 on the mass at separation and 4.0e-8 on the final mass. They are not smaller because
    RK4 drops to second order locally at the ramp corners, where dm/dt has a kink. Asserted at
    1e-7 throughout, roughly a factor of three of headroom.
    """
    inert = {i: motor.inert_mass_breakdown(i) for i in motor.stage_indices}
    m_interstage = motor.interstage_mass()
    m_launch = (
        reqs.m_payload
        + motor.stage_wet_mass(1)
        + motor.stage_wet_mass(2)
        + m_interstage
    )

    # --- closed form ---
    m_p1 = motor.dv.stage_at(1).m_propellant
    m_p2 = motor.dv.stage_at(2).m_propellant
    c1 = motor.operating_point(1)["isp_vacuum"] * P.G0
    c2 = motor.operating_point(2)["isp_vacuum"] * P.G0

    m_after_stage_1 = m_launch - m_p1
    m_after_jettison = m_after_stage_1 - motor.jettisoned_mass()
    m_final = m_after_jettison - m_p2
    closed_form = c1 * math.log(m_launch / m_after_stage_1) + c2 * math.log(
        m_after_jettison / m_final
    )
    # Locked datum for the default stack, so the docstring figure is testable. This is an IDEAL
    # vacuum delta-V: no gravity, no drag, no back pressure, and the ideal-nozzle Isp of
    # propulsion.SOURCES["prop.ideal_nozzle"], which is 3 to 7 percent optimistic.
    assert closed_form == pytest.approx(5139.1, rel=1e-4)

    # The jettison must remove exactly the booster and the interstage, leaving the payload,
    # the second stage and its propellant. If this line fails the mass bookkeeping is wrong,
    # not the integrator.
    assert m_after_jettison == pytest.approx(
        reqs.m_payload + m_p2 + inert[2]["total_recommended"], rel=1e-12
    )

    # --- numeric ---
    dv1, mass_at_separation = _rk4_vacuum_delta_v(
        motor, m_launch, 0.0, motor.t_separation, steps=8000
    )
    assert mass_at_separation == pytest.approx(m_after_stage_1, rel=1e-7)
    dv2, mass_end = _rk4_vacuum_delta_v(
        motor,
        mass_at_separation - motor.jettisoned_mass(),
        motor.t_separation,
        motor.t_all_burnout,
        steps=8000,
    )
    assert mass_end == pytest.approx(m_final, rel=1e-7)
    assert dv1 + dv2 == pytest.approx(closed_form, rel=1e-7)

    # Staging must actually pay: dropping the booster before the second burn is worth
    # real velocity compared with carrying it.
    carried = c2 * math.log(m_after_stage_1 / (m_after_stage_1 - m_p2))
    assert c2 * math.log(m_after_jettison / m_final) > carried


def test_effective_exhaust_velocity_is_constant_within_each_stage(
    motor: PI.MultiStageMotor,
) -> None:
    """F_vac / mdot is Isp_vac * g0 at every instant a stage flows, ramps included.

    This is the property the rocket-equation check leans on, so it is asserted directly rather
    than left implicit.
    """
    for index in motor.stage_indices:
        expected = motor.operating_point(index)["isp_vacuum"] * P.G0
        t_lo, t_hi = motor.t_ignition(index), motor.t_burnout(index)
        for frac in (0.001, 0.005, 0.02, 0.3, 0.7, 0.97, 0.995, 0.999):
            t = t_lo + (t_hi - t_lo) * frac
            flow = motor.mdot(t)
            assert flow > 0.0
            assert _vacuum_thrust(motor, t) / flow == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------------------
#   3. Single-stage degeneracy against the validated SV-1 motor
# --------------------------------------------------------------------------------------


def test_single_stage_stack_reproduces_the_solid_motor_operating_point() -> None:
    """Every boost-phase number of the SV-1 `SolidMotor` comes back BIT-IDENTICAL.

    Equality is exact (`==`, not `approx`), which is only possible because the stage is sized by
    the same `propulsion.throat_area_for_thrust` and closed by the same
    `propulsion.design_phase` as the SV-1 boost phase. If someone reimplements the nozzle or the
    Kn closure in `propulsion_iv1.py`, even in an algebraically equivalent form, the last bits
    move and this test fails. That is the intent.
    """
    dv = DesignVector()
    reference = P.SolidMotor(dv)
    stack = PI.MultiStageMotor(_sv1_like_stack(dv), InterceptRequirements())

    boost = reference.operating_point()["boost"]
    stage = stack.operating_point(1)
    for key in (
        "propellant_mass",
        "p_c",
        "p_e",
        "throat_area",
        "throat_diameter",
        "exit_area",
        "exit_diameter",
        "burning_area",
        "Kn",
        "burn_rate",
        "mdot",
        "burn_time",
        "thrust_vacuum",
        "thrust_sea_level",
        "C_F_vacuum",
        "isp_vacuum",
    ):
        assert stage[key] == boost[key], f"{key}: {stage[key]!r} != {boost[key]!r}"

    # Total impulse of the one-stage stack equals the boost-phase contribution of the SV-1
    # motor exactly. It is compared against the boost term alone because the SV-1 motor also
    # carries a sustain charge, which a single stage does not.
    boost_impulse = reference.c_star * boost["C_F_vacuum"] * boost["propellant_mass"]
    assert stack.total_impulse_vacuum() == boost_impulse
    assert stack.c_star == reference.c_star


def test_single_stage_stack_reproduces_the_solid_motor_thrust_curve() -> None:
    """The thrust traces are identical to the last bit while both are in the boost shape.

    WHERE AN EXACT MATCH ENDS, AND WHY. The two models place the end of the boost plateau in
    different places, and they have to:

      * `SolidMotor` blends boost into a sustain phase over T_TRANSITION = 0.10 s, so its
        mass-conserving plateau end is  t_b + 0.5*T_RISE - 0.5*T_TRANSITION.
      * A stage has nothing after it, so it tails off over T_TAILOFF = 0.20 s and its
        mass-conserving plateau end is  t_b + 0.5*T_RISE - 0.5*T_TAILOFF.

    Those differ by 0.5*(T_TAILOFF - T_TRANSITION) = 0.05 s. Before the earlier of the two
    plateau ends the sustain shape of `SolidMotor` is identically zero and both traces are the
    same rise ramp times the same plateau thrust, so the difference is EXACTLY ZERO, asserted
    with `==` at 20001 samples. After that point the SV-1 motor is transitioning to a phase the
    stage does not have, so no tolerance would be meaningful and none is claimed. The tail
    difference is a difference in what is being modelled, not a numerical discrepancy.
    """
    dv = DesignVector()
    reference = P.SolidMotor(dv)
    stack = PI.MultiStageMotor(_sv1_like_stack(dv), InterceptRequirements())

    t_common = min(reference.t_boost_end, stack.t_plateau_end(1))
    assert t_common == stack.t_plateau_end(1)
    assert reference.t_boost_end - t_common == pytest.approx(
        0.5 * (P.T_TAILOFF_S - P.T_TRANSITION_S), rel=1e-12
    )

    samples = 20_001
    for i in range(samples):
        t = t_common * i / (samples - 1)
        for altitude in (0.0, 3_000.0, 12_000.0):
            assert stack.thrust(t, altitude) == reference.thrust(t, altitude), (
                f"t = {t:.6f} s, h = {altitude:.0f} m"
            )
        assert stack.mdot(t) == reference.mdot(t), f"mdot at t = {t:.6f} s"

    # And the plateau really does deliver the requested sea-level thrust, which is the SV-1
    # sizing convention and also the IV-1 stage-1 convention.
    plateau = 0.5 * (P.T_RISE_S + t_common)
    assert stack.thrust(plateau, 0.0) == pytest.approx(dv.F_boost, rel=1e-9)


def test_single_stage_stack_reproduces_the_solid_motor_grain() -> None:
    """The stage grain closure is the SV-1 boost-segment closure, to the last bit.

    Only the burning-area-driven dimensions can match: the bay the grain sits in is defined by
    a different internal layout (`L_seeker` plus `L_payload_bay` for a payload stage against
    `L_seeker` plus `L_guidance` plus `L_warhead` plus `L_boattail` for SV-1), so bay length,
    volumetric loading and L/D are properties of the vehicle, not of the closure.
    """
    dv = DesignVector()
    reference = P.SolidMotor(dv).grain_geometry()
    stage = PI.MultiStageMotor(_sv1_like_stack(dv), InterceptRequirements()).grain_geometry(1)
    assert stage.d_outer == reference.d_outer
    assert stage.burning_area_boost == reference.burning_area_boost
    assert stage.web_boost == reference.web_boost
    assert stage.d_inner_boost == reference.d_inner_boost
    assert stage.length_boost == reference.length_boost
    assert stage.volume_boost == reference.volume_boost
    # A stage has one charge, so the sustain and terminal fields are all zero and the inherited
    # aliases still mean what they say.
    assert stage.length_sustain == 0.0
    assert stage.length_terminal == 0.0
    assert stage.length == stage.length_boost
    assert stage.web == stage.web_boost
    assert stage.d_inner == stage.d_inner_boost


# --------------------------------------------------------------------------------------
#   4. Thrust continuity
# --------------------------------------------------------------------------------------


def test_thrust_is_continuous_over_the_whole_timeline(motor: PI.MultiStageMotor) -> None:
    """No step anywhere, so the RK4 trajectory integrator never sees a discontinuous force.

    The sample step is a tenth of the shortest ramp, so a genuinely continuous trace cannot
    change by more than about a tenth of the peak between samples. The bound is stated against
    the stack peak thrust: 15 percent, the same bound `tests/test_propulsion.py` uses for SV-1.
    """
    step = min(P.T_RISE_S, P.T_TAILOFF_S) / 10.0
    t_end = motor.t_all_burnout + 1.0
    peak = max(_vacuum_thrust(motor, motor.t_plateau_end(i) - 0.01) for i in motor.stage_indices)
    assert peak > 0.0
    previous = motor.thrust(0.0, 0.0)
    for i in range(1, int(t_end / step) + 2):
        t = min(t_end, i * step)
        current = motor.thrust(t, 5_000.0)
        assert abs(current - previous) < 0.15 * peak, f"thrust jump at t = {t:.4f} s"
        previous = current


def test_thrust_is_continuous_across_every_event_at_fine_resolution(
    motor: PI.MultiStageMotor,
) -> None:
    """Dense local sweep across every ignition, plateau end, burnout and separation instant.

    The step is 1.0e-4 s, which is 1/500 of the ignition rise and 1/2000 of the tail-off, so a
    continuous trace can move at most 0.2 percent of that stage's plateau thrust per sample. The
    bound is asserted against the LOCAL stage plateau, not the stack peak, because stage 2 is a
    quarter of stage 1 and a bound scaled to stage 1 would hide a genuine step in stage 2.
    """
    events: list[tuple[str, float, float]] = []
    for index in motor.stage_indices:
        scale = motor.operating_point(index)["thrust_vacuum"]
        events.append((f"stage {index} ignition", motor.t_ignition(index), scale))
        events.append((f"stage {index} plateau end", motor.t_plateau_end(index), scale))
        events.append((f"stage {index} burnout", motor.t_burnout(index), scale))
    # The separation instant carries both scales, so use the smaller of the two as the bound.
    events.append(
        (
            "separation",
            motor.t_separation,
            min(motor.operating_point(i)["thrust_vacuum"] for i in motor.stage_indices),
        )
    )

    step = 1.0e-4
    window = 0.5
    for name, t_event, scale in events:
        n = int(2.0 * window / step)
        previous = motor.thrust(max(0.0, t_event - window), 6_000.0)
        for i in range(1, n + 1):
            t = max(0.0, t_event - window) + i * step
            current = motor.thrust(t, 6_000.0)
            assert abs(current - previous) < 0.01 * scale, (
                f"thrust jump of {abs(current - previous):.1f} N near {name} "
                f"at t = {t:.5f} s"
            )
            previous = current


def test_thrust_and_mdot_are_zero_at_the_stage_boundaries(motor: PI.MultiStageMotor) -> None:
    """The endpoints themselves, not just their neighbourhoods."""
    for index in motor.stage_indices:
        assert motor.thrust(motor.t_ignition(index), 0.0) == 0.0
        assert motor.mdot(motor.t_ignition(index)) == 0.0
        assert motor.thrust(motor.t_burnout(index), 0.0) == 0.0
        assert motor.mdot(motor.t_burnout(index)) == 0.0
    assert motor.thrust(motor.t_separation, 0.0) == 0.0
    assert motor.mdot(motor.t_separation) == 0.0


# --------------------------------------------------------------------------------------
#   5. active_stage, phase and the event times must all tell the same story
# --------------------------------------------------------------------------------------


def test_event_times_follow_the_specified_sequence(motor: PI.MultiStageMotor) -> None:
    """t = 0 ignition, burnout, coast, separation = stage-2 ignition, burnout."""
    assert motor.t_ignition(1) == 0.0
    assert motor.t_separation == pytest.approx(
        motor.t_burnout(1) + motor.reqs.t_coast_separation, rel=1e-12
    )
    assert motor.t_ignition(2) == pytest.approx(motor.t_separation, rel=1e-12)
    assert motor.t_burnout(1) < motor.t_separation < motor.t_burnout(2)
    assert motor.t_all_burnout == motor.t_burnout(2)
    assert motor.separation_times() == [motor.t_separation]
    for key, expected in DEFAULT_EVENTS.items():
        actual = {
            "t_ignition_1": motor.t_ignition(1),
            "t_burnout_1": motor.t_burnout(1),
            "t_separation": motor.t_separation,
            "t_ignition_2": motor.t_ignition(2),
            "t_all_burnout": motor.t_all_burnout,
        }[key]
        assert actual == pytest.approx(expected, rel=1e-12), key


def test_burnout_is_the_ideal_burn_time_plus_the_ramp_allowance(
    motor: PI.MultiStageMotor,
) -> None:
    """The plateau is placed so the shape integral is the ideal burn time exactly."""
    for index in motor.stage_indices:
        t_b = motor.t_burn_ideal(index)
        assert motor.t_plateau_end(index) == pytest.approx(
            motor.t_ignition(index) + t_b + 0.5 * P.T_RISE_S - 0.5 * P.T_TAILOFF_S, rel=1e-12
        )
        assert motor.t_burnout(index) == pytest.approx(
            motor.t_plateau_end(index) + P.T_TAILOFF_S, rel=1e-12
        )
        # Integral of the normalised shape is the ideal burn time.
        shape_integral = _simpson(
            lambda u: motor.shapes(motor.t_ignition(index) + u)[index],
            motor.t_burnout(index) - motor.t_ignition(index),
            samples=200_001,
        )
        assert shape_integral == pytest.approx(t_b, rel=1e-8)   # measured 3.6e-10


def test_active_stage_and_phase_agree_everywhere(motor: PI.MultiStageMotor) -> None:
    """At every sample: the label, the index and the event times say the same thing.

    Also checks the converse, that a stage reported as active is really flowing and a coast or
    burnout label really is unpowered. A label that disagreed with the mass flow would let the
    trajectory jettison at the wrong time.
    """
    t_end = motor.t_all_burnout + 2.0
    samples = 40_001
    seen: set[str] = set()
    for i in range(samples):
        t = t_end * i / (samples - 1)
        active = motor.active_stage(t)
        label = motor.phase(t)
        seen.add(label)

        if active:
            assert label == f"stage_{active}_boost", f"t = {t:.5f}"
            assert motor.t_ignition(active) <= t < motor.t_burnout(active)
            # Every other stage must be silent.
            for other in motor.stage_indices:
                if other != active:
                    assert motor.shapes(t)[other] == 0.0
        else:
            assert label in ("separation_coast", "burnout"), f"t = {t:.5f} gave {label}"
            assert motor.mdot(t) == 0.0
            assert motor.thrust(t, 4_000.0) == 0.0
            if label == "burnout":
                assert t >= motor.t_all_burnout
            else:
                assert motor.t_burnout(1) <= t < motor.t_separation

    assert seen == {"stage_1_boost", "separation_coast", "stage_2_boost", "burnout"}


def test_phase_labels_at_the_named_instants(motor: PI.MultiStageMotor) -> None:
    """Spot checks at the instants the mission cares about."""
    assert motor.phase(0.0) == "stage_1_boost"
    assert motor.active_stage(0.0) == 1
    assert motor.phase(0.5 * motor.t_burnout(1)) == "stage_1_boost"
    assert motor.phase(motor.t_burnout(1)) == "separation_coast"
    assert motor.phase(0.5 * (motor.t_burnout(1) + motor.t_separation)) == "separation_coast"
    assert motor.phase(motor.t_separation) == "stage_2_boost"
    assert motor.active_stage(motor.t_separation) == 2
    assert motor.phase(motor.t_all_burnout) == "burnout"
    assert motor.phase(motor.t_all_burnout + 100.0) == "burnout"
    assert motor.active_stage(motor.t_all_burnout) == 0


def test_zero_coast_leaves_no_gap_and_still_conserves_mass(
    reqs: InterceptRequirements,
) -> None:
    """A back-to-back stack: stage 2 lights the instant stage 1 finishes tailing off.

    The limiting case matters because it is the one where the two stage shapes could overlap
    and double-count propellant. They must not: the shapes are zero outside their own windows.
    """
    motor = PI.MultiStageMotor(default_iv1(), InterceptRequirements(t_coast_separation=0.0))
    assert motor.t_separation == motor.t_burnout(1) == motor.t_ignition(2)
    assert motor.phase(motor.t_separation) == "stage_2_boost"
    burned = _simpson(motor.mdot, motor.t_all_burnout)
    assert burned == pytest.approx(motor.dv.m_propellant_total, rel=1e-8)
    # No sample may have two stages flowing at once.
    for i in range(4001):
        t = motor.t_all_burnout * i / 4000
        flowing = [j for j, s in motor.shapes(t).items() if s > 0.0]
        assert len(flowing) <= 1, f"stages {flowing} overlap at t = {t:.5f}"


# --------------------------------------------------------------------------------------
#   6. The ambient pressure term
# --------------------------------------------------------------------------------------


def test_sea_level_and_vacuum_thrust_differ_by_exactly_pa_times_ae(
    motor: PI.MultiStageMotor,
) -> None:
    """F_vac - F_sl = p_a * A_e, exactly, for stage 1 and everywhere else it flows.

    Checked on the plateau, inside the ignition rise and inside the tail-off, because the exit
    area is shape-weighted and a mistake there would only show up in a ramp.
    """
    for index in motor.stage_indices:
        t_ig = motor.t_ignition(index)
        t_pe = motor.t_plateau_end(index)
        for t in (
            t_ig + 0.5 * P.T_RISE_S,          # inside the rise
            0.5 * (t_ig + t_pe),              # plateau
            t_pe + 0.5 * P.T_TAILOFF_S,       # inside the tail-off
        ):
            exit_area = motor.exit_area_at(t)
            assert exit_area > 0.0
            f_sea = motor.thrust(t, 0.0)
            f_vac = f_sea + P.P_SEA_LEVEL * exit_area
            assert f_vac - f_sea == pytest.approx(P.P_SEA_LEVEL * exit_area, rel=1e-12)
            # And the slope against ambient pressure is exactly the exit area.
            f_high = motor.thrust(t, 11_000.0)
            assert f_high == pytest.approx(
                f_vac - P._ambient_pressure(11_000.0) * exit_area, rel=1e-12
            )


def test_stage_1_is_sized_at_sea_level_and_stage_2_in_vacuum(
    motor: PI.MultiStageMotor,
) -> None:
    """The sizing convention of `StageSpec.F_thrust`, asserted rather than assumed.

    Stage 1 delivers its requested thrust at SEA LEVEL, because it is lit in the canister.
    Stage 2 delivers its requested thrust in VACUUM, because it is lit above 10 km. Both come
    out of the same relation with a different sizing ambient pressure.
    """
    stage_1 = motor.operating_point(1)
    assert stage_1["p_a_sizing"] == P.P_SEA_LEVEL
    assert stage_1["thrust_sea_level"] == pytest.approx(
        motor.dv.stage_at(1).F_thrust, rel=1e-12
    )
    assert stage_1["thrust_vacuum"] > stage_1["thrust_sea_level"]

    stage_2 = motor.operating_point(2)
    assert stage_2["p_a_sizing"] == 0.0
    assert stage_2["thrust_vacuum"] == pytest.approx(
        motor.dv.stage_at(2).F_thrust, rel=1e-12
    )
    assert stage_2["thrust_sea_level"] < stage_2["thrust_vacuum"]
    assert stage_2["thrust_vacuum"] - stage_2["thrust_sea_level"] == pytest.approx(
        P.P_SEA_LEVEL * stage_2["exit_area"], rel=1e-12
    )

    # An upper stage may also be sized at a real altitude instead of vacuum. It then delivers
    # its requested thrust there, and more above it. Holding the same thrust against a non-zero
    # ambient pressure needs a LARGER throat, because the (p_e - p_a) * A_e term is a debit.
    at_altitude = PI.MultiStageMotor(
        default_iv1(),
        motor.reqs,
        upper_stage_sizing_pressure=P._ambient_pressure(15_000.0),
    )
    entry = at_altitude.operating_point(2)
    assert entry["thrust_sizing"] == pytest.approx(motor.dv.stage_at(2).F_thrust, rel=1e-12)
    assert entry["throat_area"] > stage_2["throat_area"]
    assert entry["thrust_vacuum"] > motor.dv.stage_at(2).F_thrust
    # Stage 1 is untouched by an upper-stage sizing choice.
    assert at_altitude.operating_point(1)["throat_area"] == stage_1["throat_area"]


def test_flow_separation_is_reported_per_stage(motor: PI.MultiStageMotor) -> None:
    """The Summerfield criterion, per nozzle. Stage 1 at sea level is the case that matters.

    Result for the default stack, which is why no separation warning appears: stage 1 has an
    exit pressure of 100.3 kPa, so separation would need an ambient of 250.7 kPa, which is 2.5
    times sea level and never occurs in flight. Stage 2 needs 116.2 kPa, still above sea level.
    Both nozzles are therefore separation-free over the whole trajectory, at the default
    p_c = 8 MPa and area ratios of 10 and 18.
    """
    checks = motor.separation_check()
    assert set(checks) == {1, 2}
    for index, check in checks.items():
        assert check["p_a_separation"] == pytest.approx(
            check["p_e"] / P.SEPARATION_PE_OVER_PA, rel=1e-12
        )
        assert check["separation_altitude"] == 0.0, f"stage {index}"
        assert check["p_a_separation"] > P.P_SEA_LEVEL
    assert not any("separation" in w for w in motor.warnings)

    # A deliberately overexpanded stage 1 MUST raise the warning, so the check is live rather
    # than vacuously passing on a nozzle that happens to be well matched.
    over = default_iv1()
    over.stages[0].eps_nozzle = 40.0
    over_motor = PI.MultiStageMotor(over, motor.reqs)
    assert over_motor.separation_check()[1]["separation_altitude"] > 0.0
    assert any("Summerfield" in w for w in over_motor.warnings)


# --------------------------------------------------------------------------------------
#   7. Jettisoned mass
# --------------------------------------------------------------------------------------


def test_jettisoned_mass_is_positive_and_less_than_the_stage_1_wet_mass(
    motor: PI.MultiStageMotor,
) -> None:
    """It has to be a real, bounded number: the trajectory subtracts it at separation."""
    jettisoned = motor.jettisoned_mass()
    assert jettisoned > 0.0
    assert jettisoned < motor.stage_wet_mass(1)
    assert jettisoned == pytest.approx(DEFAULT_JETTISONED_MASS, rel=1e-12)
    # And it is small compared with the propellant it burned, as a booster should be.
    assert jettisoned < 0.25 * motor.dv.stage_at(1).m_propellant


def test_jettisoned_mass_contains_every_promised_item(motor: PI.MultiStageMotor) -> None:
    """Case, insulation, nozzle, igniter, fins and the interstage, and nothing else."""
    parts = motor.jettisoned_mass_breakdown()
    inert = motor.inert_mass_breakdown(1)
    for key in (
        "stage_1_case",
        "stage_1_insulation",
        "stage_1_nozzle",
        "stage_1_igniter",
        "stage_1_fins",
        "interstage",
    ):
        assert parts[key] > 0.0, key
    assert parts["stage_1_inert_physics"] == pytest.approx(
        inert["case"] + inert["insulation"] + inert["nozzle"] + inert["igniter"], rel=1e-12
    )
    assert parts["total"] == pytest.approx(
        inert["total_recommended"] + parts["interstage"], rel=1e-12
    )
    # The recommended route is at least the bottom-up route, never less.
    assert parts["total"] >= parts["total_bottom_up"] - 1e-12
    # Stage 2 is not jettisoned: it is the payload stage.
    assert motor.is_payload_stage(2)
    assert not motor.is_payload_stage(1)
    assert parts["total"] < motor.stage_wet_mass(1) + parts["interstage"]


@pytest.mark.parametrize("length", [0.10, 0.28, 0.56, 1.00])
def test_interstage_contribution_is_linear_in_its_length(
    reqs: InterceptRequirements, length: float
) -> None:
    """Doubling `L_interstage` doubles the interstage mass, exactly.

    The charged interstage is a cylindrical shell at the booster diameter, so its mass is
    pi * D1 * L * t * rho and the length enters linearly. The conical frustum alternative is
    NOT linear, because its lateral area uses the slant length sqrt(L^2 + (r1-r2)^2); it is
    computed and reported alongside for comparison but not charged. See
    propulsion_iv1.SOURCES["prop_iv1.interstage_shell"].
    """
    base = PI.MultiStageMotor(default_iv1().replace(L_interstage=0.28), reqs)
    scaled = PI.MultiStageMotor(default_iv1().replace(L_interstage=length), reqs)
    ratio = length / 0.28

    assert scaled.interstage_mass() == pytest.approx(
        base.interstage_mass() * ratio, rel=1e-12
    )
    # It reaches the jettisoned mass one for one: nothing else in the stack moves with it.
    assert scaled.jettisoned_mass() - base.jettisoned_mass() == pytest.approx(
        scaled.interstage_mass() - base.interstage_mass(), rel=1e-9, abs=1e-12
    )
    # An independent closed form for the charged shell.
    dv = scaled.dv
    expected = (
        math.pi
        * dv.booster.D
        * dv.L_interstage
        * dv.t_interstage
        * PI.INTERSTAGE_MATERIAL.density
    )
    assert scaled.interstage_mass() == pytest.approx(expected, rel=1e-12)
    # The taper is lighter than the cylinder for the same axial length.
    parts = scaled.interstage_breakdown()
    assert parts["conical"] < parts["cylindrical"]
    assert parts["slant_over_length"] > 1.0


def test_interstage_thickness_also_scales_linearly(reqs: InterceptRequirements) -> None:
    """A shell mass must be linear in its wall thickness too."""
    thin = PI.MultiStageMotor(default_iv1().replace(t_interstage=0.0025), reqs)
    thick = PI.MultiStageMotor(default_iv1().replace(t_interstage=0.0050), reqs)
    assert thick.interstage_mass() == pytest.approx(2.0 * thin.interstage_mass(), rel=1e-12)


def test_a_single_stage_stack_jettisons_nothing() -> None:
    """A one-stage stack has no separation, so nothing may leave the vehicle.

    The interstage is not charged either, whatever `L_interstage` happens to say, because there
    is no stage above the booster to adapt to. Getting this wrong would let a degenerate stack
    lose mass it never carried.
    """
    stack = _sv1_like_stack(DesignVector())
    assert stack.L_interstage > 0.0            # the field still has its default value
    motor = PI.MultiStageMotor(stack, InterceptRequirements())
    assert motor.has_separation is False
    assert motor.jettisoned_mass() == 0.0
    assert motor.interstage_mass() == 0.0
    assert all(value == 0.0 for value in motor.jettisoned_mass_breakdown().values())
    assert motor.is_payload_stage(1)
    # The single stage still has real inert mass; it just is not thrown away.
    assert motor.inert_mass_breakdown(1)["total_recommended"] > 0.0


def test_a_jettisoned_flag_that_contradicts_the_burn_order_is_reported(
    reqs: InterceptRequirements,
) -> None:
    """Burn order governs, and the disagreement is surfaced rather than silently resolved."""
    dv = default_iv1()
    dv.stages[0].jettisoned = False            # but stage 1 is not the payload stage
    dv.stages[1].jettisoned = True             # and stage 2 is
    motor = PI.MultiStageMotor(dv, reqs)
    assert motor.is_payload_stage(2)
    assert not motor.is_payload_stage(1)
    assert motor.jettisoned_mass() > 0.0       # the flag did not change the physics
    assert sum("StageSpec.jettisoned" in w for w in motor.warnings) == 2


def test_the_interstage_adapts_to_the_next_stage_to_burn(
    reqs: InterceptRequirements,
) -> None:
    """For three or more stages the interstage taper ends at stage 2 of the burn order.

    Using the payload stage diameter here would be wrong for any stack deeper than two, and
    the error would be invisible in the two-stage case where they are the same stage.
    """
    dv = default_iv1()
    dv.stages.insert(
        1,
        StageSpec(
            index=3, D=0.34, L=1.20, m_propellant=90.0, F_thrust=70.0e3, jettisoned=True
        ),
    )
    motor = PI.MultiStageMotor(dv, reqs)
    parts = motor.interstage_breakdown()
    r_aft, r_next = 0.5 * dv.stages[0].D, 0.5 * 0.34
    expected_conical_area = math.pi * (r_aft + r_next) * math.hypot(
        dv.L_interstage, r_aft - r_next
    )
    assert parts["area_conical"] == pytest.approx(expected_conical_area, rel=1e-12)


def test_inert_mass_reports_both_routes_for_every_stage(motor: PI.MultiStageMotor) -> None:
    """Physics sum, correlation band and a recommendation, per stage. CLAUDE.md section 3.3."""
    for index in motor.stage_indices:
        inert = motor.inert_mass_breakdown(index)
        for key in ("case", "insulation", "nozzle", "igniter", "fins"):
            assert inert[key] > 0.0, f"stage {index} {key}"
        assert inert["total_physics"] == pytest.approx(
            inert["case"] + inert["insulation"] + inert["nozzle"] + inert["igniter"],
            rel=1e-12,
        )
        assert inert["correlation_min"] < inert["correlation_max"]
        assert inert["recommended"] >= inert["total_physics"]
        assert inert["recommended"] >= inert["correlation_min"] - 1e-12
        assert inert["total_recommended"] == pytest.approx(
            inert["recommended"] + inert["fins"], rel=1e-12
        )
        assert inert["shortfall"] == pytest.approx(
            max(0.0, inert["correlation_min"] - inert["total_physics"]), rel=1e-12, abs=1e-12
        )
        # Hoop stress at the stage chamber pressure with the shared safety factor.
        spec = motor.dv.stage_at(index)
        radius = 0.5 * (spec.D - 2.0 * spec.t_wall)
        expected = max(
            P.CASE_MIN_GAUGE_M,
            P.CASE_SAFETY_FACTOR * spec.p_c * radius / motor.case_material.sigma_yield,
        )
        assert inert["case_thickness"] == pytest.approx(expected, rel=1e-12)


def test_case_mass_scales_with_the_stage_chamber_pressure(reqs: InterceptRequirements) -> None:
    """Doubling p_c doubles the case thickness while above the minimum gauge."""
    low = default_iv1()
    low.stages[0].p_c = 6.0e6
    high = default_iv1()
    high.stages[0].p_c = 12.0e6
    t_low = PI.MultiStageMotor(low, reqs).inert_mass_breakdown(1)["case_thickness"]
    t_high = PI.MultiStageMotor(high, reqs).inert_mass_breakdown(1)["case_thickness"]
    assert t_high == pytest.approx(2.0 * t_low, rel=1e-9)


def test_fin_mass_matches_the_masses_module_model(motor: PI.MultiStageMotor) -> None:
    """The fin model is carried over from `masses.py`, so it must give the same number."""
    from rocketgen.config import MATERIALS

    for index in motor.stage_indices:
        spec = motor.dv.stage_at(index)
        expected = (
            spec.n_fin
            * spec.S_fin_exposed
            * spec.t_fin
            * 0.65
            * MATERIALS["fin_ti64"].density
        )
        assert motor.fin_mass(index) == pytest.approx(expected, rel=1e-12)
        assert motor.fin_mass(index) > 0.0


# --------------------------------------------------------------------------------------
#   8. Per-stage grain L/D, and the bays the grains have to fit in
# --------------------------------------------------------------------------------------


def test_grain_l_over_d_is_reported_per_stage(motor: PI.MultiStageMotor) -> None:
    """SPEC_IV1.md section 6 constrains grain L/D to 1.0 to 8.0 for EACH stage.

    The default stack's values, printed below and locked in `DEFAULT_GRAIN_L_OVER_D`:

        stage 1 (booster, D = 0.400 m, 380 kg, 170 kN):  L/D = 8.61   OUTSIDE 1.0 to 8.0
        stage 2 (payload, D = 0.280 m, 150 kg,  45 kN):  L/D = 5.77   inside

    Stage 1 fails the limit with a plain tubular grain. That is a real result about the default
    design vector, not a modelling artefact: a 170 kN booster needs 3.42 m^2 of burning area at
    8 MPa, and a tube of 0.388 m bore can only reach that by being 3.34 m long, which is longer
    than the 2.00 m bay AND past L/D 8. The fixes are a shaped (slotted, star or finocyl) grain,
    a larger bore, or less thrust; see propulsion_iv1.SOURCES["prop_iv1.tubular_stage_grain"].
    """
    ratios = motor.grain_l_over_d()
    print(f"default IV-1 per-stage grain L/D: {ratios}")
    assert set(ratios) == {1, 2}
    for index, value in ratios.items():
        assert value == pytest.approx(DEFAULT_GRAIN_L_OVER_D[index], rel=1e-12), index
        assert motor.grain_geometry(index).L_over_D == value

    lo, hi = PI.GRAIN_L_OVER_D_LIMITS
    assert (lo, hi) == (1.0, 8.0)
    assert not (lo <= ratios[1] <= hi)          # stage 1 fails, and says so
    assert lo <= ratios[2] <= hi                # stage 2 passes
    assert any("L/D" in w and "stage 1" in w for w in motor.warnings)


def test_bay_lengths_follow_the_specified_layout(motor: PI.MultiStageMotor) -> None:
    """Payload stage: L minus nose and forward bays. Booster: L minus the aft closure."""
    dv = motor.dv
    assert motor.bay_length(2) == pytest.approx(
        dv.payload_stage.L - dv.L_nose - dv.L_seeker - dv.L_payload_bay, rel=1e-12
    )
    assert motor.bay_length(1) == pytest.approx(
        dv.booster.L - PI.BOOSTER_AFT_CLOSURE_ALLOWANCE_M, rel=1e-12
    )
    assert PI.BOOSTER_AFT_CLOSURE_ALLOWANCE_M == P.CASE_LENGTH_ALLOWANCE_M
    for index in motor.stage_indices:
        spec = motor.dv.stage_at(index)
        assert motor.bay_diameter(index) == pytest.approx(
            spec.D - 2.0 * spec.t_wall - 2.0 * motor.insulation_thickness, rel=1e-12
        )
        assert motor.grain_geometry(index).bay_length_available == motor.bay_length(index)
        assert motor.grain_geometry(index).bay_diameter == motor.bay_diameter(index)


def test_the_default_stack_is_reported_infeasible_rather_than_smoothed_over(
    motor: PI.MultiStageMotor,
) -> None:
    """LOCKED FINDING. `default_iv1()` does not close, and the model must say so.

    Three separate statements about the default design vector:
      1. Both grains are longer than their bays with a tubular geometry.
      2. Stage 2 needs 151 percent volumetric loading, which is impossible.
      3. The stage-1 nozzle exit is 409 mm across, wider than the 400 mm stage-1 body.

    If a future change makes these pass silently, the change has either fixed the design vector
    or hidden a constraint. Either way it must edit this test deliberately.
    """
    for index in motor.stage_indices:
        geom = motor.grain_geometry(index)
        assert not geom.feasible, f"stage {index} unexpectedly closes"
        assert any("exceeds the available motor bay" in w for w in geom.warnings)
    assert motor.grain_geometry(2).volumetric_loading > 1.0
    assert any("volumetric loading" in w for w in motor.warnings)
    assert any("exit diameter" in w and "exceeds" in w for w in motor.warnings)
    assert motor.operating_point(1)["exit_diameter"] > motor.dv.booster.D


def test_a_grain_that_cannot_hold_its_propellant_says_so(reqs: InterceptRequirements) -> None:
    """Too much propellant for the burning area is a web thicker than the bay radius."""
    dv = default_iv1()
    dv.stages[1].m_propellant = 290.0        # inside the SPEC_IV1 bound of 60 to 300 kg
    dv.stages[1].F_thrust = 20.0e3           # low thrust means a small burning area
    motor = PI.MultiStageMotor(dv, reqs)
    geom = motor.grain_geometry(2)
    assert not geom.feasible
    assert any("web" in w and "bay radius" in w for w in geom.warnings)
    assert geom.d_inner_boost > 0.0          # clamped, so the length stays finite
    assert math.isfinite(geom.length_total)


# --------------------------------------------------------------------------------------
#   Throat credibility: clean for a stack, and verified rather than asserted
# --------------------------------------------------------------------------------------


def test_no_stage_needs_a_throat_transition(motor: PI.MultiStageMotor) -> None:
    """Each stage runs one throat area for its whole burn, so no mechanism is needed.

    This is the contrast with the SV-1 dual-thrust motor, whose boost-to-sustain transition
    needs a throat that SHRINKS and which `propulsion.SolidMotor.throat_transition_report()`
    reports as not credible. Both halves are asserted here so the contrast is on the record.
    """
    report = motor.throat_credibility_report()
    assert [entry["stage"] for entry in report] == motor.stage_indices
    for entry in report:
        assert entry["n_throat_areas"] == 1
        assert entry["credible"] is True
        assert "fixed throat" in str(entry["mechanism"])

    # The SV-1 motor, for contrast: one throat serving two thrust levels is NOT credible.
    sv1 = P.SolidMotor(DesignVector())
    transitions = sv1.throat_transition_report()
    assert transitions and transitions[0]["credible"] is False

    # The two stages do have different throats, which is fine: they are different nozzles on
    # different hardware, and one of them is thrown away.
    assert motor.operating_point(1)["throat_area"] != motor.operating_point(2)["throat_area"]


# --------------------------------------------------------------------------------------
#   Contract, reporting and provenance
# --------------------------------------------------------------------------------------


def test_summary_is_a_string_that_names_every_stage(motor: PI.MultiStageMotor) -> None:
    text = motor.summary()
    assert isinstance(text, str)
    for fragment in (
        "stage 1",
        "stage 2",
        "jettisoned mass",
        "total vacuum impulse",
        "separation at t",
        "L/D",
        "warnings",
    ):
        assert fragment in text, fragment
    print(text)


def test_summary_data_has_no_missing_keys(motor: PI.MultiStageMotor) -> None:
    data = motor.summary_data()
    for key in (
        "c_star",
        "total_impulse_vacuum",
        "t_separation",
        "t_all_burnout",
        "jettisoned_mass",
        "grain_L_over_D",
        "stages",
        "inert",
        "separation_check",
        "throat_credibility",
        "warnings",
    ):
        assert key in data, key
    assert data["t_separation"] == motor.t_separation
    assert data["jettisoned_mass"] == motor.jettisoned_mass()


def test_unknown_stage_index_is_an_error_not_a_default(motor: PI.MultiStageMotor) -> None:
    """A silent default here would report the wrong stage's numbers."""
    for call in (
        lambda: motor.t_ignition(3),
        lambda: motor.t_burnout(0),
        lambda: motor.grain_geometry(7),
        lambda: motor.inert_mass_breakdown(-1),
    ):
        with pytest.raises(KeyError):
            call()


def test_duplicate_stage_indices_are_rejected(reqs: InterceptRequirements) -> None:
    dv = default_iv1()
    dv.stages[1].index = 1
    with pytest.raises(ValueError, match="duplicate stage"):
        PI.MultiStageMotor(dv, reqs)


def test_negative_coast_is_rejected(reqs: InterceptRequirements) -> None:
    with pytest.raises(ValueError, match="t_coast_separation"):
        PI.MultiStageMotor(default_iv1(), InterceptRequirements(t_coast_separation=-0.5))


def test_a_three_stage_stack_works_without_special_casing(
    reqs: InterceptRequirements,
) -> None:
    """Nothing in the model assumes exactly two stages.

    The IV-1 vehicle has two, but the timeline, the mass flow sum and the phase labels are
    written for N. A third stage must slot in with a second coast and no new code path.
    """
    dv = default_iv1()
    dv.stages.insert(
        1,
        StageSpec(
            index=3, D=0.34, L=1.20, m_propellant=90.0, F_thrust=70.0e3, jettisoned=True
        ),
    )
    # Burn order is the list order: 1, then 3, then 2. The indices are labels, not the order.
    motor = PI.MultiStageMotor(dv, reqs)
    assert motor.stage_indices == [1, 3, 2]
    assert motor.is_payload_stage(2)
    assert not motor.is_payload_stage(3)
    assert motor.t_ignition(3) == pytest.approx(
        motor.t_burnout(1) + reqs.t_coast_separation, rel=1e-12
    )
    assert motor.t_ignition(2) == pytest.approx(
        motor.t_burnout(3) + reqs.t_coast_separation, rel=1e-12
    )
    # t_separation is stage 1 leaving, which is the FIRST separation of several.
    assert motor.t_separation == pytest.approx(motor.t_ignition(3), rel=1e-12)
    assert len(motor.separation_times()) == 2
    assert motor.phase(0.5 * (motor.t_burnout(3) + motor.t_ignition(2))) == "separation_coast"
    assert motor.phase(motor.t_ignition(3) + 0.1) == "stage_3_boost"
    burned = _simpson(motor.mdot, motor.t_all_burnout)
    assert burned == pytest.approx(dv.m_propellant_total, rel=1e-8)


def test_the_motor_satisfies_the_trajectory_motor_protocol(
    motor: PI.MultiStageMotor,
) -> None:
    """Integration guard: `trajectory_iv1.MultiStageMotorLike` is the contract between the two.

    `trajectory_iv1.py` deliberately does not import this module, so nothing but a test can
    catch a signature drift between them. Skipped rather than failed when that module is not
    importable, because it is a separate work package and its state is not this module's
    business.
    """
    trajectory_iv1 = pytest.importorskip("rocketgen.sizing.trajectory_iv1")
    protocol = getattr(trajectory_iv1, "MultiStageMotorLike", None)
    if protocol is None:
        pytest.skip("trajectory_iv1 does not declare MultiStageMotorLike")
    assert isinstance(motor, protocol)
    # The protocol is runtime-checkable, which only checks that the names exist. Check the
    # shapes of the two that are properties rather than methods, since getting those wrong is
    # the drift that a name check would miss.
    assert isinstance(motor.t_separation, float)
    assert isinstance(motor.t_all_burnout, float)
    assert isinstance(motor.jettisoned_mass(), float)
    assert isinstance(motor.total_impulse_vacuum(), float)
    assert isinstance(motor.phase(0.0), str)
    assert isinstance(motor.active_stage(0.0), int)


def test_every_source_string_is_populated_and_registered() -> None:
    """CLAUDE.md section 3.1: every constant declares where it came from, guesses say GUESS."""
    from rocketgen.config import SOURCES as CONFIG_SOURCES

    assert PI.SOURCES
    for key, value in PI.SOURCES.items():
        assert key.startswith("prop_iv1."), key
        assert len(value) > 40, key
        assert CONFIG_SOURCES.get(key) == value, f"{key} not registered in config.SOURCES"

    # The entries this module admits are guesses must say so, in the word the report greps for.
    for key in (
        "prop_iv1.separation_hardware",
        "prop_iv1.interstage_shell",
        "prop_iv1.fin_mass_model",
    ):
        assert "GUESS" in PI.SOURCES[key].upper(), key

    # And the shared-physics entry must point at the module that carries the citations rather
    # than restating them, so there is one place to correct.
    assert "propulsion" in PI.SOURCES["prop_iv1.shared_physics"]
    assert "Sutton" in PI.SOURCES["prop_iv1.shared_physics"]
