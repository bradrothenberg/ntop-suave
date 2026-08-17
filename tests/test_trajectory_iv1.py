"""Validation of the IV-1 staged ascent integrator, `rocketgen/sizing/trajectory_iv1.py`.

THERE IS NO PUBLISHED REFERENCE TRAJECTORY FOR IV-1: the vehicle and its requirements are
invented for the demonstration (SPEC_IV1.md section 1). So, exactly as
`tests/test_trajectory.py` does for the single-stage mission, the reference cases here are
ANALYTIC. With drag switched off the 3-DOF equations have closed-form solutions, and a
staged vacuum ascent has a closed-form final speed. Those are the strongest available
checks on an integrator, and the tolerance is stated on every case.

  (1) vertical vacuum climb    apogee and time of apogee against h0 + V0^2/(2g) and V0/g,
                               to better than 1e-10 relative
  (2) two-stage vacuum rocket  burnout speed against the sum of the per-stage Tsiolkovsky
      equation                 increments minus g*t, with the jettisoned mass taken out
                               between stages, to better than 0.5 percent. Then the same
                               total propellant flown as ONE stage, which must end up
                               SLOWER. That is the test that proves staging works.
  (3) mass bookkeeping         m(t) = m0 - burned(t) - jettisoned(t) at every sample, to
                               machine precision, and a jettison of exactly
                               motor.jettisoned_mass() at separation
  (4) slant-range termination  a case that reaches the range while DESCENDING must end on
                               'slant_range', not on ground impact
  (5) RK4 order                halving dt drops the error by about 16x
  (6) energy conservation      V^2/2 + g*h over 100 s, to better than 1e-10 relative
  (7) determinism              two fly() calls on the same objects give identical arrays
  (8) event ordering           pitchover < stage-1 burnout < separation < stage-2 ignition
                               < stage-2 burnout

The motor and aerodynamic stubs live in this file on purpose. `propulsion_iv1` and
`aero_iv1` are written by other work packages; `trajectory_iv1` declares Protocols for
both and imports neither, so these tests validate the integrator against models this file
controls completely. EVERY CONSTANT IN THE STUBS IS MADE UP. That is the point: the stubs
are here to make the analytic cases exact, not to model air.
"""
from __future__ import annotations

import math
import time

import pytest

from rocketgen.config import AeroCoefficients, MassStatement
from rocketgen.config_iv1 import (
    InterceptRequirements,
    StackDesignVector,
    StageSpec,
    StrakeSpec,
    default_iv1,
    lateral_g,
)
from rocketgen.sizing import trajectory_iv1 as A

G = A.G0


# ======================================================================================
#   Stubs. Not physics. See the module docstring.
# ======================================================================================


class StubMultiStageMotor:
    """A deterministic stack of constant-thrust, constant-flow stages.

    Constant thrust and constant mass flow are what make the staged Tsiolkovsky case
    exact: the vacuum rocket equation assumes both. Thrust does not vary with altitude
    here, which a real `MultiStageMotor` will do through the nozzle pressure term.

    Timeline, for a two-stage stack:

        t_ignition(1) ... t_burnout(1) ... +t_coast ... t_separation = t_ignition(2)
        ... t_burnout(2)

    `t_separation` can be pushed past the end of the run with `t_separation_override`,
    which is how the single-stage comparison case flies with no staging at all.
    """

    def __init__(
        self,
        thrusts: tuple[float, ...] = (152_000.0, 24_000.0),
        propellants: tuple[float, ...] = (380.0, 150.0),
        burn_times: tuple[float, ...] = (6.0, 15.0),
        t_coast: float = 0.6,
        m_jettison: float = 120.0,
        t_ignition_1: float = 0.0,
        t_separation_override: float | None = None,
    ) -> None:
        assert len(thrusts) == len(propellants) == len(burn_times)
        self.n_stages = len(thrusts)
        self.t_coast = t_coast
        self.m_jettison = m_jettison
        self._F = {i + 1: thrusts[i] for i in range(self.n_stages)}
        self._m_p = {i + 1: propellants[i] for i in range(self.n_stages)}
        self._t_b = {i + 1: burn_times[i] for i in range(self.n_stages)}
        self._mdot = {
            s: (self._m_p[s] / self._t_b[s] if self._t_b[s] > 0.0 else 0.0)
            for s in self._F
        }
        self._t_ign = {1: t_ignition_1}
        self._t_bo = {1: t_ignition_1 + burn_times[0]}
        for s in range(2, self.n_stages + 1):
            self._t_ign[s] = self._t_bo[s - 1] + t_coast
            self._t_bo[s] = self._t_ign[s] + self._t_b[s]
        self._t_sep = (
            self._t_bo[1] + t_coast if t_separation_override is None
            else t_separation_override
        )
        self.calls = 0

    # --- the MultiStageMotorLike protocol ---

    def thrust(self, t: float, altitude: float) -> float:
        self.calls += 1
        stage = self.active_stage(t)
        return self._F[stage] if stage else 0.0

    def mdot(self, t: float) -> float:
        stage = self.active_stage(t)
        return self._mdot[stage] if stage else 0.0

    def active_stage(self, t: float) -> int:
        for s in sorted(self._F):
            if self._t_ign[s] <= t < self._t_bo[s]:
                return s
        return 0

    def phase(self, t: float) -> str:
        stage = self.active_stage(t)
        if stage:
            return f"stage_{stage}_boost"
        if t >= self.t_all_burnout:
            return "burnout"
        if t < self._t_ign[1]:
            return "pre_ignition"
        return "separation_coast"

    def t_ignition(self, stage: int) -> float:
        return self._t_ign[stage]

    def t_burnout(self, stage: int) -> float:
        return self._t_bo[stage]

    @property
    def t_separation(self) -> float:
        return self._t_sep

    @property
    def t_all_burnout(self) -> float:
        return max(self._t_bo.values())

    def jettisoned_mass(self) -> float:
        return self.m_jettison

    def total_impulse_vacuum(self) -> float:
        return sum(self._F[s] * self._t_b[s] for s in self._F)

    def reset(self) -> None:
        """State the mission must clear between flights. Here it is only a call count."""
        self.calls = 0

    # --- helpers used by the tests, not part of the protocol ---

    def exhaust_velocity(self, stage: int) -> float:
        return self._F[stage] / self._mdot[stage]

    def propellant_burned(self, t: float) -> float:
        """Closed-form propellant burned by time `t`, kg."""
        total = 0.0
        for s in sorted(self._F):
            burning = min(max(t - self._t_ign[s], 0.0), self._t_b[s])
            total += self._mdot[s] * burning
        return total


class StubCoastMotor:
    """A motor that never fires. Used by every vacuum-ballistic case."""

    def __init__(self) -> None:
        self.calls = 0

    def thrust(self, t: float, altitude: float) -> float:
        return 0.0

    def mdot(self, t: float) -> float:
        return 0.0

    def active_stage(self, t: float) -> int:
        return 0

    def phase(self, t: float) -> str:
        return "burnout"

    def t_ignition(self, stage: int) -> float:
        return 1.0e9

    def t_burnout(self, stage: int) -> float:
        return 1.0e9

    @property
    def t_separation(self) -> float:
        return 1.0e9

    @property
    def t_all_burnout(self) -> float:
        return 1.0e9

    def jettisoned_mass(self) -> float:
        return 0.0

    def total_impulse_vacuum(self) -> float:
        return 0.0

    def reset(self) -> None:
        self.calls = 0

    def propellant_burned(self, t: float) -> float:
        return 0.0


class StubStackAero:
    """Per-stage constant-CD0, linear-CN stub matching `StagedAeroLike`.

    The only thing that has to be right is that the answer CHANGES with the stage index,
    because that is what the mission has to switch at separation.
    """

    def __init__(
        self,
        S_ref: dict[int, float] | None = None,
        CD0: float = 0.42,
        CN_alpha: float = 12.0,
        induced_factor: float = 0.35,
        base_drag_relief: float = 0.06,
    ) -> None:
        self._S = S_ref or {1: 0.1257, 2: 0.0616}
        self.CD0 = CD0
        self.CN_alpha = CN_alpha
        self.induced_factor = induced_factor
        self.base_drag_relief = base_drag_relief

    def S_ref(self, stage: int) -> float:
        return self._S[stage]

    def _cn_alpha(self, stage: int) -> float:
        # Stage 2 keeps the strakes and loses the booster fins, so give the two stages
        # different slopes. Both numbers are arbitrary.
        return self.CN_alpha if stage == 1 else 0.8 * self.CN_alpha

    def evaluate(
        self,
        mach: float,
        altitude: float,
        alpha: float,
        stage: int,
        power_on: bool = False,
    ) -> AeroCoefficients:
        cn = self._cn_alpha(stage) * alpha
        cd0 = self.CD0 - (self.base_drag_relief if power_on else 0.0)
        cd = cd0 + self.induced_factor * cn * cn
        return AeroCoefficients(
            mach=mach,
            altitude=altitude,
            alpha=alpha,
            CD0=cd0,
            CD=cd,
            CN=cn,
            CN_alpha=self._cn_alpha(stage),
            CM=0.0,
            x_cp=1.0,
            L_over_D=0.0,
            breakdown={"CD0": cd0},
        )

    def trim_alpha(
        self,
        mach: float,
        altitude: float,
        required_CN: float,
        stage: int,
        power_on: bool = False,
    ) -> float:
        return required_CN / self._cn_alpha(stage)

    def CN_max(self, mach: float, altitude: float, stage: int, alpha_max: float) -> float:
        return self._cn_alpha(stage) * alpha_max


class VacuumAero(StubStackAero):
    """Zero drag and zero normal force, so the analytic cases are exact."""

    def evaluate(
        self,
        mach: float,
        altitude: float,
        alpha: float,
        stage: int,
        power_on: bool = False,
    ) -> AeroCoefficients:
        return AeroCoefficients(
            mach=mach,
            altitude=altitude,
            alpha=0.0,
            CD0=0.0,
            CD=0.0,
            CN=0.0,
            CN_alpha=0.0,
            CM=0.0,
            x_cp=0.0,
            L_over_D=0.0,
        )

    def trim_alpha(
        self,
        mach: float,
        altitude: float,
        required_CN: float,
        stage: int,
        power_on: bool = False,
    ) -> float:
        return 0.0


# ======================================================================================
#   Fixtures
# ======================================================================================


def _stack_dv(n_stages: int = 2, vertical: bool = False) -> StackDesignVector:
    """A design vector for the tests. `vertical=True` disables the pitch programme."""
    stages = [
        StageSpec(index=1, D=0.40, L=2.10, m_propellant=380.0, F_thrust=152.0e3),
        StageSpec(
            index=2, D=0.28, L=2.70, m_propellant=150.0, F_thrust=24.0e3, jettisoned=False
        ),
    ][:n_stages]
    dv = StackDesignVector(stages=stages, strakes=StrakeSpec(x_le=0.95))
    if vertical:
        # gamma_pitch at the launch angle means there is nothing to turn to, so the
        # mission flies a pure gravity turn from vertical, which holds vertical exactly.
        dv.gamma_pitch = math.radians(90.0)
    return dv


def _s_ref(dv: StackDesignVector) -> dict[int, float]:
    return {s.index: s.S_ref for s in dv.stages}


def _vertical_vacuum_mission(
    motor: object,
    m0: float,
    v_start: float = 1.0,
    n_stages: int = 2,
) -> A.AscentMission:
    """Straight up, no drag, no pitch programme, no stall floor."""
    dv = _stack_dv(n_stages=n_stages, vertical=True)
    reqs = InterceptRequirements(
        slant_range_min=1.0e9,          # never reached, so the case ends on t_max
        gamma_launch=math.radians(90.0),
    )
    return A.AscentMission(
        dv,
        reqs,
        motor,                            # type: ignore[arg-type]
        VacuumAero(S_ref=_s_ref(dv)),
        m0,
        v_start=v_start,
        velocity_floor=-1.0,              # the analytic cases must not stop on a stall
    )


def _ballistic_mission(
    gamma_launch_deg: float,
    v_start: float,
    slant_range_min: float = 1.0e9,
    h_launch: float = 0.0,
) -> A.AscentMission:
    """A vacuum parabola: no thrust, no drag, launched at an angle."""
    dv = _stack_dv(n_stages=1, vertical=True)
    dv.gamma_pitch = math.radians(gamma_launch_deg)
    reqs = InterceptRequirements(
        slant_range_min=slant_range_min,
        gamma_launch=math.radians(gamma_launch_deg),
        h_launch=h_launch,
    )
    return A.AscentMission(
        dv,
        reqs,
        StubCoastMotor(),                 # type: ignore[arg-type]
        VacuumAero(S_ref=_s_ref(dv)),
        500.0,
        v_start=v_start,
        velocity_floor=-1.0,
    )


def _iv1_mission(
    dt_aero: StubStackAero | None = None,
    slant_range_min: float | None = None,
    m0: float = 1300.0,
) -> tuple[A.AscentMission, StubMultiStageMotor]:
    """The full SPEC_IV1.md section 5 profile on the cheap stubs."""
    dv = default_iv1()
    reqs = InterceptRequirements()
    if slant_range_min is not None:
        reqs.slant_range_min = slant_range_min
    motor = StubMultiStageMotor(
        thrusts=(170.0e3, 45.0e3),
        propellants=(380.0, 150.0),
        burn_times=(5.4, 8.0),
        t_coast=reqs.t_coast_separation,
        m_jettison=90.0,
    )
    aero = dt_aero or StubStackAero(S_ref=_s_ref(dv))
    mission = A.AscentMission(dv, reqs, motor, aero, m0)   # type: ignore[arg-type]
    return mission, motor


# ======================================================================================
#   (1) vertical vacuum climb
# ======================================================================================


def test_vertical_vacuum_climb_apogee_matches_the_closed_form() -> None:
    """A drag-free unpowered vertical climb reaches h0 + V0^2/(2g) at t = V0/g.

    Vertical flight is chosen because gamma_dot vanishes identically at 90 deg, so the
    normal-force equation cannot contaminate the answer. The run is stopped exactly at
    the analytic apogee time by t_max, which the integrator clamps the final step to, so
    the last sample IS the apogee and no peak-finding interpolation is involved.
    Tolerance 1e-10 relative on both apogee and time.
    """
    v0, h0 = 500.0, 100.0
    dv = _stack_dv(n_stages=1, vertical=True)
    reqs = InterceptRequirements(
        slant_range_min=1.0e9, gamma_launch=math.radians(90.0), h_launch=h0
    )
    mission = A.AscentMission(
        dv,
        reqs,
        StubCoastMotor(),                 # type: ignore[arg-type]
        VacuumAero(S_ref=_s_ref(dv)),
        500.0,
        v_start=v0,
        velocity_floor=-1.0,
    )
    t_apogee = v0 / G
    result = mission.fly(dt=0.02, t_max=t_apogee, adaptive=False)

    assert result.time[-1] == pytest.approx(t_apogee, rel=1e-10)
    assert result.h[-1] == pytest.approx(h0 + v0 * v0 / (2.0 * G), rel=1e-10)
    assert abs(result.V[-1]) < 1.0e-10 * v0
    assert result.diagnostics["termination"] == "t_max"
    assert not result.converged      # running out of clock is not convergence


def test_vertical_vacuum_climb_holds_the_launch_angle() -> None:
    """A gravity turn from exactly 90 deg does not turn: -g*cos(90 deg)/V is zero."""
    mission = _vertical_vacuum_mission(StubCoastMotor(), 500.0, v_start=400.0)
    result = mission.fly(dt=0.05, t_max=30.0, adaptive=False)
    assert max(abs(g - math.radians(90.0)) for g in result.gamma) < 1e-12
    assert max(abs(x) for x in result.x) < 1e-9


# ======================================================================================
#   (2) two-stage vacuum rocket equation, and the point of staging
# ======================================================================================

# Both cases below fly 530 kg of propellant from 1300 kg at the same effective exhaust
# velocity over the same 21.6 s, so the gravity loss g*t is identical and the ONLY
# difference is that the staged vehicle throws 120 kg of booster inert away first.
_M0 = 1300.0
_MP1, _MP2 = 380.0, 150.0
_TB1, _TB2, _TCOAST = 6.0, 15.0, 0.6
_VE = 2400.0
_M_JETTISON = 120.0
_T_TOTAL = _TB1 + _TCOAST + _TB2


def _staged_motor() -> StubMultiStageMotor:
    return StubMultiStageMotor(
        thrusts=(_VE * _MP1 / _TB1, _VE * _MP2 / _TB2),
        propellants=(_MP1, _MP2),
        burn_times=(_TB1, _TB2),
        t_coast=_TCOAST,
        m_jettison=_M_JETTISON,
    )


def _single_stage_motor() -> StubMultiStageMotor:
    return StubMultiStageMotor(
        thrusts=(_VE * (_MP1 + _MP2) / _T_TOTAL,),
        propellants=(_MP1 + _MP2,),
        burn_times=(_T_TOTAL,),
        t_coast=0.0,
        m_jettison=0.0,
        t_separation_override=1.0e9,      # nothing is ever jettisoned
    )


def test_two_stage_vacuum_burnout_speed_matches_the_rocket_equation() -> None:
    """V(t_all_burnout) = V0 + sum(ve*ln(mass ratio)) - g*t, jettison included.

    Vertical, drag free, constant thrust and constant flow per stage. The jettisoned mass
    leaves between the two Tsiolkovsky increments, which is exactly what staging is. The
    coast between the stages contributes gravity loss and nothing else, so the total
    gravity loss is g times the whole elapsed time.

    Stated tolerance 0.5 percent; the tighter assertion below is the regression guard.
    """
    motor = _staged_motor()
    mission = _vertical_vacuum_mission(motor, _M0)
    result = mission.fly(dt=0.01, t_max=motor.t_all_burnout, adaptive=False)

    m1 = _M0 - _MP1
    m2 = m1 - _M_JETTISON
    m3 = m2 - _MP2
    analytic = (
        mission.v_start
        + _VE * math.log(_M0 / m1)
        + _VE * math.log(m2 / m3)
        - G * _T_TOTAL
    )
    assert result.time[-1] == pytest.approx(_T_TOTAL, abs=1e-9)
    assert result.mass[-1] == pytest.approx(m3, rel=1e-12)
    assert result.V[-1] == pytest.approx(analytic, rel=5e-3)
    assert result.V[-1] == pytest.approx(analytic, rel=1e-9)


def test_staging_beats_the_equivalent_single_stage() -> None:
    """THE POINT OF STAGING. Same propellant, same ve, same total burn time, same m0.

    The staged vehicle throws 120 kg of booster inert away between the two increments, so
    its second mass ratio is better and it must end up faster. If this ever fails, the
    jettison is not reaching the equations of motion.
    """
    staged_motor = _staged_motor()
    staged = _vertical_vacuum_mission(staged_motor, _M0)
    staged_result = staged.fly(
        dt=0.01, t_max=staged_motor.t_all_burnout, adaptive=False
    )

    single_motor = _single_stage_motor()
    single = _vertical_vacuum_mission(single_motor, _M0, n_stages=1)
    single_result = single.fly(
        dt=0.01, t_max=single_motor.t_all_burnout, adaptive=False
    )

    # Same propellant burned, same clock, same gravity loss.
    assert single_motor.t_all_burnout == pytest.approx(staged_motor.t_all_burnout)
    burned_staged = staged_result.mass[0] - staged_result.mass[-1] - _M_JETTISON
    burned_single = single_result.mass[0] - single_result.mass[-1]
    assert burned_staged == pytest.approx(burned_single, rel=1e-12)
    assert burned_staged == pytest.approx(_MP1 + _MP2, rel=1e-12)

    gain = staged_result.V[-1] - single_result.V[-1]
    assert gain > 0.0, (
        f"staging lost {-gain:.1f} m/s against the equivalent single stage; the "
        "jettisoned mass is not reaching the equations of motion"
    )
    # Closed form for the gain, so the test states the size it expects and not just a sign.
    m1 = _M0 - _MP1
    analytic_gain = _VE * (
        math.log((m1 - _M_JETTISON) / (m1 - _M_JETTISON - _MP2))
        - math.log(m1 / (m1 - _MP2))
    )
    assert gain == pytest.approx(analytic_gain, rel=1e-6)
    assert staged_result.h[-1] > single_result.h[-1]


# ======================================================================================
#   (3) mass bookkeeping
# ======================================================================================


def test_mass_at_every_step_is_m0_minus_burned_minus_jettisoned() -> None:
    """m(t) = m0 - propellant_burned(t) - jettisoned(t), to machine precision.

    Exactness is only possible because no RK4 step crosses an ignition, a burnout or the
    separation: with a piecewise-constant flow rate, RK4 integrates the mass equation
    exactly inside a segment. `diagnostics['separation_index']` marks the first sample
    that has lost the booster, which is needed because two samples are recorded at the
    separation instant, one on each side of the jettison.
    """
    mission, motor = _iv1_mission()
    result = mission.fly(dt=0.05, t_max=400.0, adaptive=False)
    index = int(result.diagnostics["separation_index"])
    assert index > 0

    worst = 0.0
    for i, (t, m) in enumerate(zip(result.time, result.mass)):
        jettisoned = motor.jettisoned_mass() if i >= index else 0.0
        expected = mission.m0 - motor.propellant_burned(t) - jettisoned
        worst = max(worst, abs(m - expected) / expected)
    assert worst < 1e-13, f"mass bookkeeping drifted by {worst:.3e} relative"


def test_mass_drops_by_exactly_the_jettisoned_mass_at_separation() -> None:
    """One sample either side of the jettison, and the step is exact."""
    mission, motor = _iv1_mission()
    result = mission.fly(dt=0.05, t_max=400.0, adaptive=False)
    index = int(result.diagnostics["separation_index"])

    assert result.time[index] == result.time[index - 1] == mission.diagnostics[
        "t_separation"
    ]
    drop = result.mass[index - 1] - result.mass[index]
    assert drop == pytest.approx(motor.jettisoned_mass(), rel=1e-15)

    event = next(e for e in mission.events if e.name == "separation")
    assert event.mass_jettisoned == pytest.approx(motor.jettisoned_mass(), rel=1e-15)
    assert event.mass_before == pytest.approx(result.mass[index - 1], rel=1e-15)
    assert event.mass_after == pytest.approx(result.mass[index], rel=1e-15)
    assert event.time == pytest.approx(motor.t_separation)


def test_the_reference_area_changes_at_separation() -> None:
    """SPEC_IV1.md section 5.4: the aerodynamic reference area becomes the stage-2 area.

    Flown with a drag model whose CD does not depend on alpha or on the motor, so that the
    drag ratio across the jettison IS the reference-area ratio and nothing else. The two
    samples at the separation instant share a speed and an altitude, so they share q.
    """
    dv = default_iv1()
    aero = StubStackAero(
        S_ref=_s_ref(dv), induced_factor=0.0, base_drag_relief=0.0
    )
    mission, _motor = _iv1_mission(dt_aero=aero)
    result = mission.fly(dt=0.05, t_max=400.0, adaptive=False)
    index = int(result.diagnostics["separation_index"])
    assert mission._S_ref[1] == pytest.approx(dv.stages[0].S_ref)
    assert mission._S_ref[2] == pytest.approx(dv.stages[1].S_ref)
    assert result.q[index] == pytest.approx(result.q[index - 1], rel=1e-15)
    ratio = result.drag[index] / result.drag[index - 1]
    assert ratio == pytest.approx(dv.stages[1].S_ref / dv.stages[0].S_ref, rel=1e-12)


def test_the_total_propellant_burned_equals_the_motor_loading() -> None:
    mission, motor = _iv1_mission()
    result = mission.fly(dt=0.05, t_max=400.0, adaptive=False)
    burned = result.mass[0] - result.mass[-1] - motor.jettisoned_mass()
    assert burned == pytest.approx(motor.propellant_burned(1.0e6), rel=1e-12)


# ======================================================================================
#   (4) slant-range termination outranks ground impact
# ======================================================================================


def _analytic_ballistic_range(v0: float, gamma_deg: float) -> float:
    return v0 * v0 * math.sin(2.0 * math.radians(gamma_deg)) / G


def test_slant_range_reached_while_descending_is_a_success() -> None:
    """SPEC_IV1.md section 5: reaching A2 on the way down is an intercept, not a failure.

    A 45 deg vacuum parabola has a slant range of 0.559*R at apogee and R at impact, so a
    threshold at 0.90*R can only be crossed on the way down. The run must end on
    'slant_range' with the vehicle still above the ground and still descending.
    """
    v0, gamma_deg = 500.0, 45.0
    r_analytic = _analytic_ballistic_range(v0, gamma_deg)
    threshold = 0.90 * r_analytic
    mission = _ballistic_mission(gamma_deg, v0, slant_range_min=threshold)
    result = mission.fly(dt=0.02, t_max=400.0, adaptive=False)

    intercept = mission.intercept
    assert intercept.termination == "slant_range"
    assert intercept.reached_slant_range is True
    assert result.converged
    assert result.h[-1] > 0.0, "the run must not have reached the ground"
    assert math.sin(result.gamma[-1]) < 0.0, "and it must be descending"
    # The bisection must land on the threshold, not one step past it.
    assert intercept.slant_range == pytest.approx(threshold, abs=1e-6)
    assert math.hypot(result.x[-1], result.h[-1]) == pytest.approx(threshold, abs=1e-6)


def test_ground_impact_is_still_reported_when_the_range_is_never_reached() -> None:
    """The same parabola with an out-of-reach threshold falls back to ground impact."""
    v0, gamma_deg = 500.0, 45.0
    r_analytic = _analytic_ballistic_range(v0, gamma_deg)
    mission = _ballistic_mission(gamma_deg, v0, slant_range_min=10.0 * r_analytic)
    result = mission.fly(dt=0.02, t_max=400.0, adaptive=False)

    assert mission.intercept.termination == "ground_impact"
    assert mission.intercept.reached_slant_range is False
    assert result.converged
    assert result.h[-1] == pytest.approx(0.0, abs=1e-6)
    # And the flight is the closed-form parabola, which is the same check
    # tests/test_trajectory.py makes on the single-stage integrator.
    assert result.range_final == pytest.approx(r_analytic, rel=1e-3)
    assert result.time[-1] == pytest.approx(
        2.0 * v0 * math.sin(math.radians(gamma_deg)) / G, rel=1e-3
    )
    assert result.V[-1] == pytest.approx(v0, rel=1e-6)


def test_a_staged_ascent_can_intercept_on_the_way_down() -> None:
    """The same priority rule on the real profile, not just on a bare parabola.

    A full two-stage ascent, staged and coasting on a lofted arc, crossing the range
    threshold after apogee. This is the case SPEC_IV1.md section 5 says must not be thrown
    away as a ground impact.
    """
    dv = default_iv1()
    aero = StubStackAero(S_ref=_s_ref(dv), CD0=0.30, induced_factor=0.05)
    mission, motor = _iv1_mission(dt_aero=aero, slant_range_min=90_000.0)
    result = mission.fly(dt=0.02, t_max=600.0, adaptive=False)
    intercept = mission.intercept

    assert intercept.termination == "slant_range"
    assert intercept.reached_slant_range is True
    assert intercept.time > motor.t_all_burnout, "it should be coasting by then"
    assert intercept.altitude > 0.0
    assert math.sin(result.gamma[-1]) < 0.0, "and it must be past apogee"
    assert intercept.altitude < max(result.h), "which means below the apogee it flew"
    assert intercept.slant_range == pytest.approx(90_000.0, abs=1e-6)
    assert intercept.mass == pytest.approx(
        mission.m0 - motor.propellant_burned(1e6) - motor.jettisoned_mass(), rel=1e-12
    )
    assert result.phase[-1] == "midcourse_coast"
    # A3, A4 and A11 are intercept conditions, so a shortfall must reach the message once
    # the range condition is met. The sizing loop owns the residuals; this is the warning.
    for value, minimum, text in (
        (intercept.altitude, mission.reqs.h_intercept_min, "below the A3 minimum"),
        (intercept.mach, mission.reqs.mach_intercept_min, "below the A4 minimum"),
        (
            intercept.lateral_g_available,
            mission.reqs.lateral_g_min,
            "below the A11 minimum",
        ),
    ):
        assert (value < minimum) == (text in result.message)


def test_the_stall_floor_is_armed_only_after_the_vehicle_has_flown() -> None:
    """A 1 m/s launch must not trip the stall test, but a real stall must."""
    dv = _stack_dv(n_stages=1, vertical=True)
    reqs = InterceptRequirements(
        slant_range_min=1.0e9, gamma_launch=math.radians(90.0)
    )
    mission = A.AscentMission(
        dv,
        reqs,
        StubCoastMotor(),                 # type: ignore[arg-type]
        VacuumAero(S_ref=_s_ref(dv)),
        500.0,
        v_start=1.0,
        velocity_floor=20.0,
    )
    result = mission.fly(dt=0.01, t_max=100.0, adaptive=False)
    # It never gets above 20 m/s, so the floor is never armed and the run ends on the
    # ground rather than immediately at t = 0.
    assert result.time[-1] > 0.1
    assert mission.intercept.termination in ("ground_impact", "stalled")

    faster = A.AscentMission(
        dv,
        reqs,
        StubCoastMotor(),                 # type: ignore[arg-type]
        VacuumAero(S_ref=_s_ref(dv)),
        500.0,
        v_start=200.0,
        velocity_floor=20.0,
    )
    stalled = faster.fly(dt=0.01, t_max=100.0, adaptive=False)
    assert faster.intercept.termination == "stalled"
    assert stalled.V[-1] < 20.0
    assert stalled.converged


# ======================================================================================
#   (5) RK4 order
# ======================================================================================


def test_rk4_is_fourth_order_on_the_vacuum_parabola() -> None:
    """Halving dt must drop the range error by about 16x.

    Case (1) cannot be used for this: a vertical drag-free climb has a constant
    acceleration, so its solution is a cubic in t and RK4 integrates it EXACTLY. The error
    there is double-precision noise and its ratio is meaningless. The 45 deg parabola is
    the same case with the nonlinear gamma_dot term switched on, which is what the
    single-stage suite uses (tests/test_trajectory.py) for the same reason. Large steps
    are used on purpose, because at dt = 0.02 the truncation error is already near noise.
    The accepted band is 10 to 22, because the ground-impact bisection carries an
    O(dt^4) error of its own.
    """
    v0, gamma_deg = 500.0, 45.0
    analytic = _analytic_ballistic_range(v0, gamma_deg)
    errors: list[float] = []
    for dt in (0.8, 0.4, 0.2):
        mission = _ballistic_mission(gamma_deg, v0)
        result = mission.fly(dt=dt, t_max=400.0, adaptive=False)
        errors.append(abs(result.range_final - analytic))
    assert errors[0] > 0.0
    for coarse, fine in zip(errors, errors[1:]):
        ratio = coarse / fine
        assert 10.0 < ratio < 22.0, f"order ratio {ratio:.2f} is not near 16"


def test_the_vertical_vacuum_climb_is_step_size_independent() -> None:
    """Case (1) is exact for RK4, so halving dt must not move the answer at all."""
    v0 = 500.0
    apogees: list[float] = []
    for dt in (0.5, 0.05):
        mission = _vertical_vacuum_mission(StubCoastMotor(), 500.0, v_start=v0)
        apogees.append(mission.fly(dt=dt, t_max=v0 / G, adaptive=False).h[-1])
    assert apogees[0] == pytest.approx(apogees[1], rel=1e-12)


# ======================================================================================
#   (6) energy conservation
# ======================================================================================


def test_specific_energy_is_conserved_without_thrust_or_drag() -> None:
    """V^2/2 + g*h is constant to better than 1e-10 relative over 100 s."""
    mission = _ballistic_mission(30.0, 600.0, h_launch=20_000.0)
    result = mission.fly(dt=0.02, t_max=100.0, adaptive=False)
    assert result.time[-1] == pytest.approx(100.0, abs=1e-9)
    energy = [0.5 * v * v + G * h for v, h in zip(result.V, result.h)]
    drift = max(abs(e - energy[0]) for e in energy) / abs(energy[0])
    assert drift < 1.0e-10, f"specific energy drifted by {drift:.3e} relative"


# ======================================================================================
#   (7) determinism
# ======================================================================================


def test_two_identical_flights_give_identical_arrays() -> None:
    """The sizing loop calls fly() many times on one motor and one aero object.

    Everything fly() touches has to be reset, including the motor. The single-stage
    mission had the same requirement for its terminal-pulse ignition time.
    """
    mission, motor = _iv1_mission()
    first = mission.fly(dt=0.05, t_max=400.0, adaptive=False)
    calls_after_first = motor.calls
    second = mission.fly(dt=0.05, t_max=400.0, adaptive=False)

    assert first.time == second.time
    assert first.x == second.x
    assert first.h == second.h
    assert first.V == second.V
    assert first.mach == second.mach
    assert first.mass == second.mass
    assert first.gamma == second.gamma
    assert first.thrust == second.thrust
    assert first.drag == second.drag
    assert first.q == second.q
    assert first.alpha == second.alpha
    assert first.phase == second.phase
    assert first.CN_required == second.CN_required
    assert first.alpha_limited == second.alpha_limited
    assert first.message == second.message
    assert first.converged == second.converged
    # fly() must have called motor.reset(), or the call count would keep climbing.
    assert motor.calls == calls_after_first
    assert [e.time for e in mission.events] == pytest.approx(
        [d["time"] for d in first.diagnostics["events"]]
    )
    assert first.diagnostics["intercept"] == second.diagnostics["intercept"]


def test_the_adaptive_mode_agrees_with_the_fixed_step_answer() -> None:
    """Step doubling must reach the same intercept, in far fewer steps."""
    v0, gamma_deg = 500.0, 45.0
    threshold = 0.90 * _analytic_ballistic_range(v0, gamma_deg)
    fixed = _ballistic_mission(gamma_deg, v0, slant_range_min=threshold).fly(
        dt=0.02, t_max=400.0, adaptive=False
    )
    adaptive = _ballistic_mission(gamma_deg, v0, slant_range_min=threshold).fly(
        dt=0.05, t_max=400.0, adaptive=True, tolerance=1e-9
    )
    assert adaptive.time[-1] == pytest.approx(fixed.time[-1], rel=1e-6)
    assert adaptive.h[-1] == pytest.approx(fixed.h[-1], rel=1e-5)
    assert len(adaptive.time) < len(fixed.time)


# ======================================================================================
#   (8) event ordering
# ======================================================================================


def test_the_events_happen_in_the_right_order() -> None:
    mission, motor = _iv1_mission()
    mission.fly(dt=0.05, t_max=400.0, adaptive=False)
    times = {e.name: e.time for e in mission.events}
    for name in (
        "pitchover",
        "pitchover_complete",
        "stage_1_burnout",
        "separation",
        "stage_2_ignition",
        "stage_2_burnout",
    ):
        assert name in times, f"{name} missing from {sorted(times)}"
    assert (
        times["pitchover"]
        < times["stage_1_burnout"]
        <= times["separation"]
        <= times["stage_2_ignition"]
        < times["stage_2_burnout"]
    )
    assert times["pitchover"] == pytest.approx(mission.dv.t_pitch)
    assert times["stage_1_burnout"] == pytest.approx(motor.t_burnout(1))
    assert times["separation"] == pytest.approx(motor.t_separation)
    assert times["stage_2_ignition"] == pytest.approx(motor.t_ignition(2))
    assert times["stage_2_burnout"] == pytest.approx(motor.t_burnout(2))
    # Events are in time order in the list, not just in the dictionary.
    assert [e.time for e in mission.events] == sorted(e.time for e in mission.events)
    # Only the separation moves mass.
    for event in mission.events:
        if event.name != "separation":
            assert event.mass_jettisoned == 0.0


def test_the_separation_coast_matches_the_requirement() -> None:
    """SPEC_IV1.md section 5.4: t_coast_separation of unpowered coast before separation."""
    mission, motor = _iv1_mission()
    result = mission.fly(dt=0.05, t_max=400.0, adaptive=False)
    coast = motor.t_separation - motor.t_burnout(1)
    assert coast == pytest.approx(mission.reqs.t_coast_separation)
    coasting = [
        thrust
        for thrust, t in zip(result.thrust, result.time)
        if motor.t_burnout(1) <= t < motor.t_ignition(2) - 1e-9
    ]
    assert coasting
    assert max(coasting) == 0.0


def test_a_motor_that_separates_at_the_wrong_time_is_reported() -> None:
    """Rule 3.3: a disagreement between the motor and the requirements is bad news."""
    dv = default_iv1()
    reqs = InterceptRequirements()
    motor = StubMultiStageMotor(
        burn_times=(5.4, 8.0), t_coast=reqs.t_coast_separation + 3.0
    )
    mission = A.AscentMission(
        dv, reqs, motor, StubStackAero(S_ref=_s_ref(dv)), 1300.0   # type: ignore[arg-type]
    )
    result = mission.fly(dt=0.1, t_max=200.0, adaptive=False)
    assert "disagree about the coast" in result.message


# ======================================================================================
#   The guidance programme
# ======================================================================================


def test_the_vertical_rise_is_held_until_t_pitch() -> None:
    """Vertical, at zero alpha, to machine precision, until the pitch command arrives.

    Nothing holds the vehicle vertical except the physics: at gamma = 90 deg the gravity
    turn term -g*cos(gamma)/V is zero. So this also checks that no step leaks the pitch
    command backwards across the t_pitch boundary.
    """
    mission, _motor = _iv1_mission()
    result = mission.fly(dt=0.02, t_max=400.0, adaptive=False)
    t_pitch = float(mission.dv.t_pitch)
    index = result.time.index(t_pitch)
    assert index > 0
    for gamma, alpha in zip(result.gamma[:index], result.alpha[:index]):
        assert gamma == pytest.approx(mission.reqs.gamma_launch, abs=1e-12)
        assert alpha == 0.0
    # The sample AT t_pitch is the first one flying the command.
    assert result.gamma[index] == pytest.approx(mission.reqs.gamma_launch, abs=1e-12)
    assert result.alpha[index] < 0.0
    assert result.CN_required[index] < 0.0
    assert result.diagnostics["t_pitch_start"] == pytest.approx(t_pitch)
    # cos(90 deg) is 6e-17 in double precision, not zero, so the rise drifts downrange by
    # tens of femtometres. That is the floating-point floor, not a modelled crossrange.
    assert abs(result.x[index]) < 1.0e-9


def test_the_pitchover_ends_exactly_on_the_commanded_angle() -> None:
    """The bisection lands on gamma_pitch, so the turn cannot overshoot by a step."""
    mission, _motor = _iv1_mission()
    result = mission.fly(dt=0.05, t_max=400.0, adaptive=False)
    t_complete = float(result.diagnostics["t_pitch_complete"])
    assert not math.isnan(t_complete)
    index = result.time.index(t_complete)
    assert result.gamma[index] == pytest.approx(mission.dv.gamma_pitch, abs=1e-11)
    # After the turn the gravity turn takes over, so gamma only ever decreases.
    tail = result.gamma[index:]
    assert all(b <= a + 1e-12 for a, b in zip(tail, tail[1:]))


def test_the_pitchover_is_alpha_limited_and_says_so() -> None:
    """No thrust-vector control means the commanded pitch rate is not achievable.

    SPEC_IV1.md section 8 requires this to be visible rather than assumed away. The turn
    is flown on aerodynamic normal force at the alpha limit, so it takes longer than
    (gamma_launch - gamma_pitch) / pitch_rate_max and the shortfall is reported.
    """
    mission, _motor = _iv1_mission()
    result = mission.fly(dt=0.05, t_max=400.0, adaptive=False)
    commanded = (
        mission.reqs.gamma_launch - mission.dv.gamma_pitch
    ) / mission.dv.pitch_rate_max
    flown = float(result.diagnostics["t_pitch_complete"]) - mission.dv.t_pitch
    assert flown > commanded, "the commanded rate was somehow achieved without TVC"
    assert any(result.alpha_limited)
    assert result.diagnostics["alpha_limit_fraction"] > 0.0
    assert "no thrust-vector control" in result.message
    assert max(abs(a) for a in result.alpha) <= mission.alpha_max + 1e-12


def test_a_smaller_gamma_pitch_flies_a_flatter_trajectory() -> None:
    """Turning further over lowers the apogee, monotonically.

    Range is NOT monotone in `gamma_pitch` and this test asserts that it is not. With drag
    on, a ballistic ascent has an interior best-range pitch angle: too steep wastes the
    boost on altitude, too shallow spends it in dense air. On these stubs the 55 deg case
    out-ranges both 70 and 40 deg. That is the classical result, and it is why the sizer
    has to search `gamma_pitch` rather than drive it to a bound.
    """
    apogees: list[float] = []
    ranges: list[float] = []
    for gamma_deg in (70.0, 55.0, 40.0):
        mission, _motor = _iv1_mission(slant_range_min=1.0e9)
        mission.dv.gamma_pitch = math.radians(gamma_deg)
        result = mission.fly(dt=0.05, t_max=600.0, adaptive=False)
        apogees.append(max(result.h))
        ranges.append(result.range_final)
    assert apogees[0] > apogees[1] > apogees[2]
    assert ranges[1] == max(ranges), (
        f"best-range pitch angle is no longer interior: {ranges}"
    )


def test_no_pitch_programme_means_no_pitch_events() -> None:
    mission = _vertical_vacuum_mission(_staged_motor(), _M0)
    mission.fly(dt=0.05, t_max=20.0, adaptive=False)
    assert not [e for e in mission.events if e.name.startswith("pitchover")]
    names = [s["name"] for s in mission.diagnostics["guidance_segments"]]
    assert names == ["gravity_turn"]


# ======================================================================================
#   Result and diagnostics completeness
# ======================================================================================


def test_every_result_list_has_the_same_length() -> None:
    mission, _motor = _iv1_mission()
    result = mission.fly(dt=0.1, t_max=400.0, adaptive=False)
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
    assert all(q >= 0.0 for q in result.q)
    assert all(m > 0.0 for m in result.mass)


def test_the_phase_history_visits_every_segment_in_order() -> None:
    mission, _motor = _iv1_mission()
    result = mission.fly(dt=0.02, t_max=400.0, adaptive=False)
    seen: list[str] = []
    for phase in result.phase:
        if not seen or seen[-1] != phase:
            seen.append(phase)
    for expected in (
        "stage_1_boost",
        "separation_coast",
        "stage_2_boost",
        "midcourse_coast",
    ):
        assert expected in seen, f"{expected} missing from {seen}"
    assert (
        seen.index("stage_1_boost")
        < seen.index("separation_coast")
        < seen.index("stage_2_boost")
        < seen.index("midcourse_coast")
    )


def test_dynamic_pressure_is_recorded_and_checked_against_a10() -> None:
    """A10 caps q at 250 kPa, and a vertical sea-level launch is the case that drives it.

    The ASCENT peak is the one A10 is about, and it must sit low down and inside the boost,
    which is the whole reason SPEC_IV1.md raised the limit from SV-1's 200 kPa. The overall
    peak can be larger still, because an unpowered fall back through sea-level air
    reaccelerates; both are recorded and both are checked against the limit in the message.
    """
    mission, motor = _iv1_mission()
    result = mission.fly(dt=0.02, t_max=400.0, adaptive=False)
    assert result.q_max > 0.0
    assert result.diagnostics["q_max"] == pytest.approx(result.q_max)
    assert result.diagnostics["q_max_limit"] == mission.reqs.q_max
    assert all(q >= 0.0 for q in result.q)

    ascent = [
        (q, t, h)
        for q, t, h in zip(result.q, result.time, result.h)
        if t <= motor.t_all_burnout
    ]
    q_ascent, t_ascent, h_ascent = max(ascent)
    assert h_ascent < 5_000.0, "the ascent q peak is not near the ground any more"
    assert t_ascent <= motor.t_burnout(1) + 1e-9, "nor inside the stage-1 boost"
    assert q_ascent > 0.5 * mission.reqs.q_max, "and it should be the sizing case for A10"
    if result.q_max > mission.reqs.q_max:
        assert "exceeds the A10 limit" in result.message


def test_the_intercept_result_is_filled_completely() -> None:
    mission, _motor = _iv1_mission(slant_range_min=40_000.0)
    result = mission.fly(dt=0.02, t_max=400.0, adaptive=False)
    intercept = mission.intercept
    assert intercept.termination == "slant_range"
    assert intercept.reached_slant_range is True
    assert intercept.time == pytest.approx(result.time[-1])
    assert intercept.altitude == pytest.approx(result.h[-1])
    assert intercept.ground_range == pytest.approx(result.x[-1])
    assert intercept.velocity == pytest.approx(result.V[-1])
    assert intercept.mach == pytest.approx(result.mach[-1])
    assert intercept.mass == pytest.approx(result.mass[-1])
    assert intercept.q == pytest.approx(result.q[-1])
    assert intercept.slant_range == pytest.approx(40_000.0, abs=1e-6)
    assert intercept.slant_range_miles == pytest.approx(40_000.0 / 1609.344)
    assert intercept.lateral_g_available > 0.0


def test_the_lateral_g_capability_uses_a11_quantities() -> None:
    """A11: q * S_ref * CN_max(alpha_max) / (m*g0), on the STAGE-2 area and the mass at
    termination. The sizing loop reads this number, so it must be reproducible by hand."""
    mission, _motor = _iv1_mission(slant_range_min=40_000.0)
    result = mission.fly(dt=0.02, t_max=400.0, adaptive=False)
    intercept = mission.intercept
    aero = mission.aero
    cn_max = aero.CN_max(
        intercept.mach, intercept.altitude, 2, mission.reqs.alpha_max
    )
    expected = lateral_g(
        intercept.q, aero.S_ref(2), cn_max, intercept.mass
    )
    assert intercept.lateral_g_available == pytest.approx(expected, rel=1e-15)
    assert result.diagnostics["lateral_g_available"] == pytest.approx(expected)
    # It must use the stage-2 area, not the booster area, and the mass after jettison.
    wrong = lateral_g(intercept.q, aero.S_ref(1), cn_max, intercept.mass)
    assert intercept.lateral_g_available != pytest.approx(wrong)


def test_the_diagnostics_carry_the_events_and_the_intercept() -> None:
    mission, _motor = _iv1_mission()
    result = mission.fly(dt=0.1, t_max=400.0, adaptive=False)
    for key in (
        "termination",
        "events",
        "intercept",
        "guidance_segments",
        "reached_slant_range",
        "slant_range",
        "lateral_g_available",
        "q_max",
        "h_apogee",
        "h_stage_1_burnout",
        "h_above_atmosphere_table",
        "atmosphere_table_ceiling",
        "atmosphere_source",
        "t_pitch_start",
        "t_pitch_complete",
        "t_separation",
        "separation_index",
        "mass_jettisoned",
        "m0",
        "alpha_limit_hits",
        "alpha_limit_fraction",
        "steps",
    ):
        assert key in result.diagnostics, key
        assert key in mission.diagnostics, key
    assert len(result.diagnostics["events"]) == len(mission.events)
    assert result.diagnostics["intercept"]["termination"] == result.diagnostics[
        "termination"
    ]
    assert result.message


def test_flying_above_the_atmosphere_table_is_reported_not_hidden() -> None:
    """The mission reports how far above the atmosphere table it flew, and with the extended
    table that distance is now zero even on the loftiest arc the bounds allow.

    This test originally asserted a 30 km ceiling AND that the loftiest arc cleared it. Both were
    symptoms of a real bug: a lofted intercept apogees at 45 to 54 km, and above the ceiling the
    lookup clamps, holding density at its ceiling value and OVERSTATING drag by more than an order
    of magnitude. The table was extended to 86 km, the upper limit of US Standard 1976, so the arc
    no longer leaves it.

    What is worth asserting now is that the diagnostic still exists and reads zero, and that no
    warning fires. The reporting path itself is exercised by driving the check against a
    deliberately low ceiling below.
    """
    from rocketgen.sizing.atmosphere import H_MAX

    mission, _motor = _iv1_mission(slant_range_min=1.0e9)
    mission.dv.gamma_pitch = math.radians(75.0)      # the loftiest arc the bounds allow
    result = mission.fly(dt=0.05, t_max=600.0, adaptive=False)

    assert "atmosphere_table_ceiling" in result.diagnostics
    assert "h_above_atmosphere_table" in result.diagnostics
    ceiling = float(result.diagnostics["atmosphere_table_ceiling"])
    over = float(result.diagnostics["h_above_atmosphere_table"])

    assert ceiling == pytest.approx(H_MAX)
    apogee = max(result.h)
    assert apogee < H_MAX, f"apogee {apogee/1e3:.1f} km is above the {H_MAX/1e3:.0f} km table"
    assert over == 0.0
    assert "atmosphere" not in (result.message or "").lower()


def test_the_atmosphere_overshoot_report_still_works_when_it_should_fire() -> None:
    """Guard the reporting path itself, independent of where the ceiling happens to sit.

    Extending the table to 86 km stopped the nominal arc from leaving it, which means the
    overshoot branch is no longer exercised by any real trajectory. Without this test the branch
    could rot silently and a future ceiling change would go unreported.
    """
    from rocketgen.sizing import trajectory_iv1 as T

    mission, _motor = _iv1_mission(slant_range_min=1.0e9)
    mission.dv.gamma_pitch = math.radians(75.0)

    original = T.atmosphere_ceiling if hasattr(T, "atmosphere_ceiling") else None
    if original is None:
        pytest.skip("trajectory_iv1 reads the ceiling directly; no seam to drive it low")

    T.atmosphere_ceiling = lambda: 5_000.0        # type: ignore[assignment]
    try:
        result = mission.fly(dt=0.05, t_max=600.0, adaptive=False)
    finally:
        T.atmosphere_ceiling = original           # type: ignore[assignment]

    assert float(result.diagnostics["h_above_atmosphere_table"]) > 0.0
    assert "atmosphere" in (result.message or "").lower()

def test_the_mission_reports_the_alpha_limit_and_the_q_floor_honestly() -> None:
    """Rule 3.3: bad news travels upward. Both flags must reach the message."""
    mission, _motor = _iv1_mission(slant_range_min=1.0e9)
    result = mission.fly(dt=0.05, t_max=600.0, adaptive=False)
    # `alpha_limit_hits` counts FORCE EVALUATIONS, four per RK4 step, so it is larger than
    # the number of flagged samples. Both are non-zero on this case.
    hits = float(result.diagnostics["alpha_limit_hits"])
    assert hits > 0.0
    assert sum(1 for flag in result.alpha_limited if flag) > 0
    assert hits >= sum(1 for flag in result.alpha_limited if flag)
    assert "control authority is short" in result.message
    assert result.diagnostics["alpha_limit_fraction"] == pytest.approx(
        hits / float(result.diagnostics["force_calls"])
    )
    if result.diagnostics["q_floor_hit"]:
        assert "Pa floor" in result.message


# ======================================================================================
#   Contract, stubs and performance
# ======================================================================================


def test_the_stubs_satisfy_the_declared_protocols() -> None:
    """If a stub drifts from the Protocol, the real modules would too."""
    for motor in (StubMultiStageMotor(), StubCoastMotor()):
        assert isinstance(motor, A.MultiStageMotorLike)
    for aero in (StubStackAero(), VacuumAero()):
        assert isinstance(aero, A.StagedAeroLike)


def test_launch_mass_is_resolved_from_every_documented_shape() -> None:
    assert A.resolve_launch_mass(1300.0) == 1300.0
    assert A.resolve_launch_mass(1300) == 1300.0

    statement = MassStatement()
    statement.add("airframe", 800.0, 1.0)
    statement.add("propellant", 500.0, 2.0)
    assert A.resolve_launch_mass(statement) == pytest.approx(1300.0)

    class WithTotal:
        total = 1300.0

    class WithMassAt:
        def mass_at(self, t: float, burned: float, jettisoned: bool) -> float:
            return 1300.0 - burned - (90.0 if jettisoned else 0.0)

    assert A.resolve_launch_mass(WithTotal()) == 1300.0
    assert A.resolve_launch_mass(WithMassAt()) == 1300.0
    assert A.resolve_launch_mass(lambda: 1300.0) == 1300.0

    with pytest.raises(TypeError):
        A.resolve_launch_mass("1300")
    with pytest.raises(TypeError):
        A.resolve_launch_mass(True)
    with pytest.raises(ValueError):
        A.resolve_launch_mass(-1.0)


def test_the_alpha_limit_defaults_to_the_requirement() -> None:
    mission, _motor = _iv1_mission()
    assert mission.alpha_max == pytest.approx(mission.reqs.alpha_max)
    dv = default_iv1()
    reqs = InterceptRequirements()
    explicit = A.AscentMission(
        dv,
        reqs,
        StubMultiStageMotor(),            # type: ignore[arg-type]
        StubStackAero(S_ref=_s_ref(dv)),
        1300.0,
        alpha_max=math.radians(8.0),
    )
    assert explicit.alpha_max == pytest.approx(math.radians(8.0))


def test_an_impossible_jettison_raises_rather_than_flying_negative_mass() -> None:
    dv = default_iv1()
    reqs = InterceptRequirements()
    motor = StubMultiStageMotor(burn_times=(5.4, 8.0), m_jettison=5000.0)
    mission = A.AscentMission(
        dv, reqs, motor, StubStackAero(S_ref=_s_ref(dv)), 1300.0   # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="mass statement and the motor disagree"):
        mission.fly(dt=0.1, t_max=100.0, adaptive=False)


def test_the_helpers_work_off_a_result() -> None:
    mission, motor = _iv1_mission(slant_range_min=40_000.0)
    result = mission.fly(dt=0.05, t_max=400.0, adaptive=False)
    history = A.slant_range_history(result)
    assert len(history) == len(result.time)
    assert history[-1] == pytest.approx(40_000.0, abs=1e-6)
    assert A.event_time(mission.events, "separation") == pytest.approx(
        motor.t_separation
    )
    assert math.isnan(A.event_time(mission.events, "no_such_event"))


def test_an_unflown_mission_reports_nothing_rather_than_guessing() -> None:
    mission, _motor = _iv1_mission()
    assert mission.events == []
    assert mission.intercept.termination == ""
    assert mission.intercept.slant_range == 0.0


def test_the_full_ascent_runs_inside_the_time_budget() -> None:
    """Performance target: a full ascent at dt = 0.02 in a couple of seconds."""
    mission, _motor = _iv1_mission(slant_range_min=1.0e9)
    start = time.perf_counter()
    result = mission.fly(dt=0.02, t_max=300.0, adaptive=False)
    elapsed = time.perf_counter() - start
    assert len(result.time) > 5_000
    assert elapsed < 4.0, f"took {elapsed:.2f} s for {len(result.time)} steps"


def test_every_source_string_is_populated_and_guesses_are_flagged() -> None:
    """CLAUDE.md hard rule 3.1, applied to this module."""
    for key, value in A.SOURCES.items():
        assert key.startswith("traj_iv1.")
        assert len(value) > 40, key
    for key in ("traj_iv1.launch_speed", "traj_iv1.velocity_floor"):
        assert "GUESS" in A.SOURCES[key], key
    for key in ("traj_iv1.pitch_programme", "traj_iv1.q_floor", "traj_iv1.alpha_limit"):
        assert (
            "MODELLING CHOICE" in A.SOURCES[key]
            or "modelling choice" in A.SOURCES[key]
            or "invented" in A.SOURCES[key]
        ), key
    # SPEC_IV1.md section 8 requires the pitchover and the separation model to declare
    # themselves. Both must name the section they answer to.
    for key in ("traj_iv1.pitch_programme", "traj_iv1.separation"):
        assert "SPEC_IV1.md section 8" in A.SOURCES[key], key
