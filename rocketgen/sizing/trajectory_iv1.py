"""IV-1 staged ascent mission: the SPEC_IV1.md section 5 profile, 3-DOF.

What this module is, and what it is not
--------------------------------------
This is `sizing/trajectory.py` generalised from one stage to a stack. The equations of
motion, the classical RK4 step and the Richardson step-doubling error control are
INHERITED from `trajectory.PointMass3DOF` and are not re-derived here, so SV-1 and IV-1 fly
the same physics and the same integrator. What is new is everything that a two-stage
ascent needs and a single-stage cruise-and-dive does not:

  1. a vertical rise from a canister at `reqs.gamma_launch`, held until `dv.t_pitch`;
  2. a bounded-rate pitchover to `dv.gamma_pitch`, followed by a pure gravity turn;
  3. steps that never cross an ignition, burnout or separation time, so the right-hand
     side is continuous inside every RK4 stage;
  4. an instantaneous mass jettison at separation, and an aerodynamic reference area that
     changes with it;
  5. termination on SLANT range with a priority above ground impact, so that reaching
     A2 while descending is a success and not a failure.

Point 3 is not cosmetic. A single RK4 stage that straddles a burnout evaluates thrust on
both sides of the discontinuity, which destroys both the fourth-order accuracy and the
mass bookkeeping: the last boost step would book only five sixths of the propellant burned
in it. Every event time is therefore a hard step boundary.

Guidance: there is no autopilot gain in this module
--------------------------------------------------
SPEC_IV1.md section 5 prefers a gravity turn over an invented autopilot gain, and section 8
requires the pitchover model to declare itself as a modelling choice. The programme here
has no gain of any kind:

  * Vertical rise. At gamma = 90 deg the gravity-turn term -g*cos(gamma)/V is identically
    zero, so commanding zero normal force holds the vehicle vertical exactly. Nothing is
    held by feedback.
  * Pitchover. The commanded pitch rate is exactly `dv.pitch_rate_max`, a design variable,
    applied open loop. It is not a gain on an angle error. The turn ends when the flown
    gamma reaches `dv.gamma_pitch`, found by bisecting the step that crosses it, so there
    is no overshoot and no dt dependence.
  * Everything after pitchover. Zero commanded normal force, that is a pure gravity turn.
    Boost, the separation coast, stage-2 boost and the midcourse arc all fly at alpha = 0.

The one physical consequence worth stating: with no thrust-vector control the commanded
pitch rate is usually NOT achievable. The normal force needed to turn at 8 deg/s just
after a sea-level launch is far beyond the alpha limit, so the turn is alpha-limited and
takes longer than `(gamma_launch - gamma_pitch) / pitch_rate_max`. That is recorded, not
hidden: see `TrajectoryResult.alpha_limited`, `diagnostics['alpha_limit_fraction']` and the
warning text in `TrajectoryResult.message`.

Mass programme
--------------
`mass_stack` supplies ONE number, the launch mass. The integrator then owns the mass
programme completely:

    m(t) = m0 - (propellant burned by t) - (jettisoned mass, once, at separation)

Propellant burned is the RK4 integral of `motor.mdot`, which is exact for a piecewise
constant flow rate because no step crosses a boundary. The jettison is applied to the
state in one step at `motor.t_separation`. This is the simplest design that is correct,
and it means the sizing loop can hand in a `config.MassStatement`, a
`masses.MassBuildup`, or a bare float, with no adapter. See `resolve_launch_mass`.

The atmosphere and its ceiling
------------------------------
The atmosphere is the cached table in `sizing/atmosphere.py`, reached through the adapter
`trajectory.atmosphere_properties`. That table stops at 30 km and CLAMPS above it, which
is deliberate there but material here: a lofted intercept goes well above 30 km, and a
clamped density overstates drag aloft. The mission measures the overshoot, puts it in
`diagnostics['h_above_atmosphere_table']` and says so in `TrajectoryResult.message`. It
never silently extrapolates.

Approximations and modelling choices are collected in `SOURCES`.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Callable, Protocol, runtime_checkable

from ..config import AeroCoefficients, TrajectoryResult, register_sources
from ..config_iv1 import (
    InterceptRequirements,
    InterceptResult,
    StackDesignVector,
    StageEvent,
    lateral_g,
    slant_range,
)
from .trajectory import (
    ATMOSPHERE_SOURCE,
    G0,
    FlightState,
    Forces,
    PointMass3DOF,
    atmosphere_properties,
)

# --------------------------------------------------------------------------------------
#   Settings. These are modelling and numerical choices, not physics.
# --------------------------------------------------------------------------------------

#: Initial speed on the launch rail, m/s. See SOURCES["traj_iv1.launch_speed"].
V_START = 1.0

#: Speed below which the run is declared stalled, m/s. See SOURCES["traj_iv1.velocity_floor"].
VELOCITY_FLOOR = 20.0

#: Dynamic pressure below which the trim solution is frozen, Pa. As in trajectory.py.
Q_FLOOR = 200.0

#: Fixed-point passes on the thrust-times-sin(alpha) term of the normal-force balance.
TRIM_ITERATIONS = 2

#: Bisection budget for event refinement. 80 halvings take a 1 s bracket below 1e-24 s.
BISECTION_ITERATIONS = 80

#: Time tolerance for "this instant is an event boundary", s.
EVENT_EPS = 1.0e-9

#: The motor is queried strictly inside the active burn interval, s.
#: See SOURCES["traj_iv1.motor_interior_query"].
MOTOR_EPS = 1.0e-9

#: Flight-path-angle tolerance for "the pitchover has finished", rad.
GAMMA_EPS = 1.0e-9

#: Altitude tolerance for the ground-impact and slant-range refinements, m.
REFINE_ATOL = 1.0e-9

SOURCES: dict[str, str] = {
    "traj_iv1.equations": (
        "The equations of motion, the classical RK4 step and the Richardson step-doubling "
        "error control are inherited unchanged from rocketgen.sizing.trajectory "
        "(PointMass3DOF), whose own sources are registered under the 'traj.' keys. This "
        "module adds staging, the ascent guidance programme and the termination logic, and "
        "changes no term of the right-hand side."
    ),
    "traj_iv1.launch_speed": (
        "GUESS: the vertical rise starts at 1.0 m/s rather than at rest. A 3-DOF point "
        "mass at exactly zero speed has an undefined flight-path angle and a degenerate "
        "gamma_dot equation, so the speed must be non-zero. No canister exit-velocity "
        "figure was sourced. The choice is close to harmless: at the IV-1 launch "
        "thrust-to-weight the vehicle passes 1 m/s within about 10 ms of ignition, so the "
        "programme is shifted by that much and the burnout speed changes by under 1 m/s. "
        "Exposed as the AscentMission(v_start=...) argument so it can be traded."
    ),
    "traj_iv1.velocity_floor": (
        "GUESS: the run is declared stalled below 20 m/s. Nothing in SPEC_IV1.md sets a "
        "floor. It exists because the gamma_dot equation divides by speed, so an "
        "arbitrarily slow point mass is numerically meaningless, and because the aero "
        "model is not valid there either. The check is ARMED only after the vehicle has "
        "once exceeded the floor, so the 1 m/s launch condition cannot trip it. Exposed "
        "as AscentMission(velocity_floor=...)."
    ),
    "traj_iv1.q_floor": (
        "MODELLING CHOICE, carried over from rocketgen.sizing.trajectory: below 200 Pa of "
        "dynamic pressure the required-normal-force coefficient is frozen at zero instead "
        "of being divided by a vanishing q*S_ref. Reaching this on a lofted arc is "
        "recorded in TrajectoryResult.message and in diagnostics['q_floor_hit']."
    ),
    "traj_iv1.pitch_programme": (
        "MODELLING CHOICE, and SPEC_IV1.md section 8 requires it to be declared. The "
        "vehicle rises vertically until dv.t_pitch, then turns at the constant commanded "
        "rate dv.pitch_rate_max until the flown flight-path angle reaches dv.gamma_pitch, "
        "then flies a pure gravity turn (zero commanded normal force) for the rest of the "
        "flight. There is NO autopilot gain: the commanded rate is a design variable "
        "applied open loop, and the end of the turn is found by bisecting the step that "
        "crosses gamma_pitch. There is no thrust-vector control and no side thruster, so "
        "the turn is flown on aerodynamic normal force alone and is frequently limited by "
        "the alpha limit; the limited steps are flagged rather than flown as commanded."
    ),
    "traj_iv1.alpha_limit": (
        "The trim angle of attack is limited to InterceptRequirements.alpha_max (20 deg "
        "for IV-1, itself an invented requirement). Demands beyond the limit are recorded "
        "as an authority shortfall and the flown alpha is clipped, exactly as in the "
        "single-stage mission. Override with AscentMission(alpha_max=...)."
    ),
    "traj_iv1.event_boundaries": (
        "NUMERICAL REQUIREMENT, not a choice: every ignition time, every burnout time, the "
        "separation time and dv.t_pitch are hard step boundaries, so no RK4 stage ever "
        "straddles a thrust or mass discontinuity. Without this the step containing a "
        "burnout books only five sixths of the propellant burned in it (the k4 stage lands "
        "on the far side of the discontinuity), which breaks both the fourth-order "
        "convergence and the mass bookkeeping."
    ),
    "traj_iv1.motor_interior_query": (
        "NUMERICAL GUARD: the motor is queried at a time clamped 1e-9 s inside the active "
        "burn interval. Steps end exactly on ignition and burnout times, so an RK4 stage "
        "would otherwise evaluate thrust exactly ON a boundary, where the answer depends "
        "on whether the motor implements its intervals half-open or closed. Clamping "
        "removes that dependence. For a constant-thrust stage the clamp changes nothing at "
        "all; for a time-varying grain it shifts the query by 1e-9 s."
    ),
    "traj_iv1.separation": (
        "Separation is instantaneous, as SPEC_IV1.md section 8 requires: motor."
        "jettisoned_mass() leaves the vehicle in one step at motor.t_separation, the "
        "aerodynamic reference stage becomes 2 at the same instant, and no impulse, tip-off "
        "moment or drag transient is modelled. Two samples are recorded at the separation "
        "time, one on each side of the jettison, so the discontinuity is visible in the "
        "trajectory arrays rather than smeared across a step."
    ),
    "traj_iv1.termination": (
        "SPEC_IV1.md section 5.7 and the note under it. Termination is checked in the "
        "priority order slant_range, ground_impact, t_max, stalled, and the slant-range "
        "condition OUTRANKS ground impact: reaching A2 while descending is a legitimate "
        "intercept for this vehicle class. The terminating step is refined by bisection to "
        "1e-9 m on the terminating residual, so the reported intercept conditions are not "
        "a step-size artefact. The only case where a satisfied slant range does not end "
        "the run is a step in which the ground is crossed first; that is then reported as "
        "ground_impact."
    ),
    "traj_iv1.lateral_g": (
        "Available lateral acceleration at termination is config_iv1.lateral_g evaluated "
        "with the dynamic pressure, the stage-2 reference area, aero.CN_max at "
        "reqs.alpha_max and the mass AT TERMINATION, which is requirement A11. It is a "
        "static capability figure and says nothing about autopilot response; see "
        "config_iv1.SOURCES['iv1_lateral_accel']."
    ),
    "traj_iv1.atmosphere_ceiling": (
        "rocketgen.sizing.atmosphere tabulates 0 to 30 km and CLAMPS outside that band. A "
        "lofted IV-1 intercept goes above 30 km, where the clamp holds density at its 30 km "
        "value and therefore OVERSTATES drag. The mission measures the overshoot into "
        "diagnostics['h_above_atmosphere_table'] and warns in TrajectoryResult.message. It "
        "does not extrapolate the table and it does not silently accept the clamp."
    ),
    "traj_iv1.mass_programme": (
        "The mass programme is owned by the integrator, not by the mass statement: "
        "m(t) = m0 - propellant burned - jettisoned mass. `mass_stack` supplies only m0, "
        "read from a float, from `total_mass` (config.MassStatement), from `total` "
        "(masses.MassBuildup), from `m0`, from `launch_mass`, from a zero-argument "
        "callable, or from mass_at(0, 0, False). Nothing is guessed and no default launch "
        "mass exists: an unrecognised object raises."
    ),
}

register_sources(SOURCES)

# The atmosphere table ceiling, needed so the mission can report when it flies above it.
try:                                        # pragma: no cover - trivial import guard
    from .atmosphere import H_MAX as ATMOSPHERE_H_MAX
except Exception:                           # pragma: no cover
    ATMOSPHERE_H_MAX = 30_000.0


# --------------------------------------------------------------------------------------
#   Protocols. This module imports neither propulsion_iv1 nor aero_iv1.
# --------------------------------------------------------------------------------------


@runtime_checkable
class MultiStageMotorLike(Protocol):
    """What the staged ascent needs from a motor stack.

    `rocketgen.sizing.propulsion_iv1.MultiStageMotor` is the real implementation. This
    module never imports it, so the two work packages stay independent, exactly as
    `trajectory.AeroCallable` keeps the single-stage mission independent of `aero.py`.
    `tests/test_trajectory_iv1.py` carries a stub that satisfies this protocol.

    Times are mission times: t = 0 is the launch instant, and stage 1 ignites at
    `t_ignition(1)`, normally 0.
    """

    def thrust(self, t: float, altitude: float) -> float:
        """Thrust, N, at mission time `t` and altitude `altitude`."""
        ...

    def mdot(self, t: float) -> float:
        """Propellant mass flow, kg/s, positive while burning."""
        ...

    def active_stage(self, t: float) -> int:
        """1-based index of the burning stage, 0 while coasting."""
        ...

    def phase(self, t: float) -> str:
        """Phase name, for example 'stage_1_boost' or 'separation_coast'."""
        ...

    def t_ignition(self, stage: int) -> float:
        """Ignition time of `stage`, s."""
        ...

    def t_burnout(self, stage: int) -> float:
        """Burnout time of `stage`, s."""
        ...

    @property
    def t_separation(self) -> float:
        """Time at which stage 1 and the interstage leave the vehicle, s."""
        ...

    @property
    def t_all_burnout(self) -> float:
        """Time at which the last stage stops thrusting, s."""
        ...

    def jettisoned_mass(self) -> float:
        """Stage-1 inert mass plus the interstage, kg."""
        ...

    def total_impulse_vacuum(self) -> float:
        """Vacuum total impulse of the whole stack, N.s."""
        ...


@runtime_checkable
class StagedAeroLike(Protocol):
    """What the staged ascent needs from an aerodynamic model.

    Every call carries the stage index, because the reference area, the fins and the
    strake contribution all change at separation. `rocketgen.sizing.aero_iv1` is the real
    implementation and is not imported here.
    """

    def evaluate(
        self,
        mach: float,
        altitude: float,
        alpha: float,
        stage: int,
        power_on: bool = False,
    ) -> AeroCoefficients:
        """Coefficients at a flight point. CD and CN are on `S_ref(stage)`."""
        ...

    def trim_alpha(
        self,
        mach: float,
        altitude: float,
        required_CN: float,
        stage: int,
        power_on: bool = False,
    ) -> float:
        """Angle of attack, rad, that produces `required_CN` on `S_ref(stage)`."""
        ...

    def S_ref(self, stage: int) -> float:
        """Aerodynamic reference area of `stage`, m^2."""
        ...

    def CN_max(self, mach: float, altitude: float, stage: int, alpha_max: float) -> float:
        """Normal-force coefficient available at the alpha limit, on `S_ref(stage)`."""
        ...


# --------------------------------------------------------------------------------------
#   Launch mass resolution
# --------------------------------------------------------------------------------------

_M0_ATTRIBUTES = ("total_mass", "total", "m0", "launch_mass")


def resolve_launch_mass(mass_stack: Any) -> float:
    """Launch mass in kg from whatever the caller passed as `mass_stack`.

    See SOURCES["traj_iv1.mass_programme"] for why only one number is needed. Accepted
    shapes, in order:

      * a number;
      * an object carrying `total_mass` (`config.MassStatement`), `total`
        (`masses.MassBuildup`), `m0` or `launch_mass`, as an attribute or a property;
      * a zero-argument callable;
      * an object with `mass_at(t, propellant_burned, stage_jettisoned)`, queried at
        `mass_at(0.0, 0.0, False)`.

    Anything else raises. There is deliberately no default: a mission that quietly
    invented its own launch mass would report a sized vehicle that was never sized.
    """
    if isinstance(mass_stack, bool):
        raise TypeError("mass_stack must be a mass, not a bool")
    if isinstance(mass_stack, (int, float)):
        m0 = float(mass_stack)
    else:
        m0 = float("nan")
        for name in _M0_ATTRIBUTES:
            value = getattr(mass_stack, name, None)
            if callable(value):
                value = value()
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                m0 = float(value)
                break
        if math.isnan(m0):
            mass_at = getattr(mass_stack, "mass_at", None)
            if callable(mass_at):
                m0 = float(mass_at(0.0, 0.0, False))
            elif callable(mass_stack):
                m0 = float(mass_stack())
        if math.isnan(m0):
            raise TypeError(
                "mass_stack must be a number, a callable, or carry one of "
                f"{_M0_ATTRIBUTES} or mass_at(t, burned, jettisoned); got "
                f"{type(mass_stack).__name__}"
            )
    if not math.isfinite(m0) or m0 <= 0.0:
        raise ValueError(f"launch mass must be finite and positive, got {m0}")
    return m0


# --------------------------------------------------------------------------------------
#   Integrator primitives
# --------------------------------------------------------------------------------------


class StagedStepper(PointMass3DOF):
    """`PointMass3DOF` plus the two primitives a staged ascent needs.

    Nothing about the dynamics is touched. `adaptive_step` only exposes the base class's
    step-doubling controller under a public name, and `bisect` generalises the base
    class's `_refine_to_ground`, which can only bisect on altitude, to any scalar
    residual. That is what lets one piece of code refine the pitchover completion, the
    ground impact and the slant-range intercept.
    """

    def adaptive_step(
        self,
        state: FlightState,
        dt: float,
        tolerance: float,
        dt_min: float,
        dt_max: float,
        k1: tuple[float, float, float, float, float] | None = None,
    ) -> tuple[FlightState, float]:
        """One Richardson-controlled step. Returns the state and the next step size."""
        return self._adaptive_step(state, dt, tolerance, dt_min, dt_max, k1)

    def bisect(
        self,
        state: FlightState,
        dt: float,
        residual: Callable[[FlightState], float],
        k1: tuple[float, float, float, float, float] | None = None,
        atol: float = REFINE_ATOL,
        iterations: int = BISECTION_ITERATIONS,
    ) -> FlightState:
        """Sub-step of `state` on which `residual` is zero, by bisecting the step length.

        `residual` must be negative at `state` and non-negative at `state` stepped by
        `dt`. Every trial is a fresh RK4 step from `state`, never a chain of short steps,
        so the refined point carries the same fourth-order accuracy as the step it
        replaces.
        """
        lo, hi = 0.0, dt
        best = self.step(state, dt, k1)
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            trial = self.step(state, mid, k1)
            value = residual(trial)
            if abs(value) <= atol:
                return trial
            if value < 0.0:
                lo = mid
            else:
                hi = mid
                best = trial
        return best


# --------------------------------------------------------------------------------------
#   The IV-1 ascent mission
# --------------------------------------------------------------------------------------


class AscentMission:
    """Fly the SPEC_IV1.md section 5 profile with a given motor stack, aero and mass.

    Phases, in order, as recorded in `TrajectoryResult.phase`:

      pre_ignition       before `motor.t_ignition(1)`. Normally empty: IV-1 lights at t=0.
      stage_1_boost      whatever `motor.phase` calls it while stage 1 burns. The vertical
                         rise and the pitchover happen inside this phase; their times are
                         in `events` and in `diagnostics['guidance_segments']`.
      separation_coast   unpowered, between stage-1 burnout and stage-2 ignition.
      stage_2_boost      whatever `motor.phase` calls it while stage 2 burns.
      midcourse_coast    unpowered after the last burnout, on the lofted arc. The motor
                         calls this 'burnout'; SPEC_IV1.md section 5.6 calls it the
                         midcourse coast, and that is the name recorded.

    Usage:

        mission = AscentMission(dv, reqs, motor, aero, mass_stack)
        result = mission.fly(dt=0.02)
        result.diagnostics["events"]      # also mission.events, as StageEvent objects
        result.diagnostics["intercept"]   # also mission.intercept, as InterceptResult

    `fly` is repeatable: it resets every piece of mission state, and it calls
    `motor.reset()` when the motor offers one, because the sizing loop flies one motor
    object many times.
    """

    def __init__(
        self,
        dv: StackDesignVector,
        reqs: InterceptRequirements,
        motor: MultiStageMotorLike,
        aero: StagedAeroLike,
        mass_stack: Any,
        alpha_max: float | None = None,
        v_start: float = V_START,
        velocity_floor: float = VELOCITY_FLOOR,
    ) -> None:
        self.dv = dv
        self.reqs = reqs
        self.motor = motor
        self.aero = aero
        self.mass_stack = mass_stack
        self.m0 = resolve_launch_mass(mass_stack)
        # The alpha limit defaults to the requirement rather than to a number invented
        # here. See SOURCES["traj_iv1.alpha_limit"].
        self.alpha_max = float(reqs.alpha_max if alpha_max is None else alpha_max)
        self.v_start = float(v_start)
        self.velocity_floor = float(velocity_floor)

        self.diagnostics: dict[str, Any] = {}
        self._events: list[StageEvent] = []
        self._intercept = InterceptResult()

        # Reference areas are read once per stage: they are geometry, not flight state.
        self._S_ref: dict[int, float] = {
            stage: float(aero.S_ref(stage)) for stage in range(1, dv.n_stages + 1)
        }

        self._read_motor_timeline()
        self._reset_flight_state()

    # ------------------------------------------------------------------ timeline ---

    def _read_motor_timeline(self) -> None:
        """Cache the motor's event times. They are geometry of the burn, not state."""
        n = self.dv.n_stages
        self._t_ignition = {s: float(self.motor.t_ignition(s)) for s in range(1, n + 1)}
        self._t_burnout = {s: float(self.motor.t_burnout(s)) for s in range(1, n + 1)}
        self._t_separation = float(self.motor.t_separation)
        self._t_all_burnout = float(self.motor.t_all_burnout)
        self._t_first_ignition = min(self._t_ignition.values())
        self._m_jettison = float(self.motor.jettisoned_mass())
        # IV-1 is a two-stage stack, so the aerodynamic stage is 1 before separation and 2
        # after it. This clamp is the ONE place a three-stage stack would generalise: the
        # rule would become "one more than the number of stages jettisoned so far".
        self._stage_after_separation = min(2, n)

        # The pitch programme is a no-op when the commanded angle is at or above the
        # launch angle, which is how the analytic vacuum cases fly straight up.
        self._pitch_enabled = (
            self.dv.gamma_pitch < self.reqs.gamma_launch - GAMMA_EPS
            and self.dv.pitch_rate_max > 0.0
        )

    def _hard_boundaries(self, t_max: float) -> list[float]:
        """Times at which a step must end. See SOURCES["traj_iv1.event_boundaries"]."""
        candidates: list[float] = [self._t_separation]
        candidates.extend(self._t_ignition.values())
        candidates.extend(self._t_burnout.values())
        if self._pitch_enabled:
            candidates.append(float(self.dv.t_pitch))
        out: list[float] = []
        for t in sorted(candidates):
            if t <= EVENT_EPS or t >= t_max - EVENT_EPS:
                continue
            if out and t - out[-1] <= EVENT_EPS:
                continue
            out.append(t)
        return out

    def _pending_events(self) -> list[tuple[float, int, str]]:
        """(time, tie-break rank, name) for the time-triggered events, in order.

        The rank orders events that share an instant: separation must be recorded before
        the stage-2 ignition it enables, and a burnout before both.
        """
        pending: list[tuple[float, int, str]] = []
        if self._pitch_enabled:
            pending.append((float(self.dv.t_pitch), 0, "pitchover"))
        for stage in sorted(self._t_burnout):
            pending.append((self._t_burnout[stage], 1, f"stage_{stage}_burnout"))
        pending.append((self._t_separation, 2, "separation"))
        for stage in sorted(self._t_ignition):
            if stage == 1 and self._t_ignition[stage] <= EVENT_EPS:
                continue     # launch is not an event; it is the initial condition
            pending.append((self._t_ignition[stage], 3, f"stage_{stage}_ignition"))
        pending.sort()
        return pending

    # --------------------------------------------------------------------- state ---

    def _reset_flight_state(self) -> None:
        """Clear everything `fly` mutates, so two flights cannot differ."""
        self._separated = False
        self._active_stage = 0
        self._pitch_armed = False
        self._stall_armed = False
        self._alpha_limit_hits = 0
        self._force_calls = 0
        self._q_floor_hit = False
        self._t_pitch_complete = float("nan")
        self._separation_index = -1
        self._events = []
        self._intercept = InterceptResult()

    @property
    def events(self) -> list[StageEvent]:
        """The discrete events of the last flight, in time order. Empty before `fly`."""
        return list(self._events)

    @property
    def intercept(self) -> InterceptResult:
        """Conditions at the end of the last flight. Default-valued before `fly`."""
        return self._intercept

    # ------------------------------------------------------------------ guidance ---

    def _segment_stage(self, t: float) -> int:
        """Burning stage over the segment that STARTS at `t`, 0 if it is a coast.

        Read from the cached ignition and burnout times rather than from
        `motor.active_stage`, so the answer cannot depend on whether the motor treats its
        burn intervals as half-open or closed. Steps never cross a boundary, so one call
        per step is valid for every RK4 stage inside it.
        """
        for stage in sorted(self._t_ignition):
            if self._t_ignition[stage] - EVENT_EPS <= t < self._t_burnout[stage] - EVENT_EPS:
                return stage
        return 0

    def _motor_query_time(self, t: float) -> float:
        """`t` clamped strictly inside the active burn.

        See SOURCES["traj_iv1.motor_interior_query"] for why this guard exists.
        """
        stage = self._active_stage
        if stage == 0:
            return t
        t_lo = self._t_ignition[stage] + MOTOR_EPS
        t_hi = self._t_burnout[stage] - MOTOR_EPS
        if t_hi < t_lo:                       # a burn shorter than 2 ns: use its midpoint
            return 0.5 * (self._t_ignition[stage] + self._t_burnout[stage])
        return min(max(t, t_lo), t_hi)

    def _is_pitching(self, state: FlightState) -> bool:
        """True while the commanded pitchover is still running.

        `_pitch_armed` is latched once per step from the time at the START of the step,
        not read from the substage time, for the same reason the active stage is: `t_pitch`
        is a hard step boundary, so a substage landing exactly on it would otherwise see
        the command switch on inside the step that is supposed to end before it, and the
        vertical rise would turn by a fraction of a step before it was told to.

        The remaining condition is a pure function of the state. It cannot re-arm after the
        turn finishes, because the gravity turn drives gamma monotonically down from
        `gamma_pitch`.
        """
        if not (self._pitch_enabled and self._pitch_armed):
            return False
        return state.gamma > self.dv.gamma_pitch + GAMMA_EPS

    def _normal_accel_command(self, state: FlightState) -> float:
        """Commanded normal specific force, m/s^2, as V*gamma_dot + g*cos(gamma).

        Zero everywhere except during the pitchover, which is what makes every other
        segment a pure gravity turn with no gain. See SOURCES["traj_iv1.pitch_programme"].
        """
        if not self._is_pitching(state):
            return 0.0
        return G0 * math.cos(state.gamma) - state.V * self.dv.pitch_rate_max

    def _phase_label(self, t: float) -> str:
        """Phase name for the recorded sample. See the class docstring."""
        if t < self._t_first_ignition - EVENT_EPS:
            return "pre_ignition"
        if self._active_stage > 0:
            return str(self.motor.phase(self._motor_query_time(t)))
        if t >= self._t_all_burnout - EVENT_EPS:
            return "midcourse_coast"
        return "separation_coast"

    # --------------------------------------------------------------- force model ---

    def force_model(self, state: FlightState) -> Forces:
        """Trim the airframe and return the forces at this state."""
        self._force_calls += 1
        rho, _p, _T, sound = atmosphere_properties(state.h)
        mach = state.V / sound if sound > 0.0 else 0.0
        q = 0.5 * rho * state.V * state.V

        stage = self._stage_after_separation if self._separated else 1
        qs = q * self._S_ref[stage]

        if self._active_stage > 0:
            t_q = self._motor_query_time(state.t)
            thrust = float(self.motor.thrust(t_q, state.h))
            mdot = float(self.motor.mdot(t_q))
        else:
            thrust, mdot = 0.0, 0.0
        powered = thrust > 0.0

        # Fixed-point solve for alpha, as in the single-stage mission: the required normal
        # force depends on T*sin(alpha), which depends on alpha. Started from alpha = 0
        # every call so the force model stays a pure function of the state.
        accel_cmd = self._normal_accel_command(state)
        alpha = 0.0
        cn_required = 0.0
        alpha_limited = False
        if accel_cmd != 0.0:
            passes = TRIM_ITERATIONS if powered else 1
            for _ in range(passes):
                normal_required = state.m * accel_cmd - thrust * math.sin(alpha)
                if q > Q_FLOOR:
                    cn_required = normal_required / qs
                else:
                    cn_required = 0.0
                    self._q_floor_hit = True
                alpha_trim = float(
                    self.aero.trim_alpha(mach, state.h, cn_required, stage, powered)
                )
                if alpha_trim > self.alpha_max:
                    alpha_trim, alpha_limited = self.alpha_max, True
                elif alpha_trim < -self.alpha_max:
                    alpha_trim, alpha_limited = -self.alpha_max, True
                else:
                    alpha_limited = False
                alpha = alpha_trim

        coeffs = self.aero.evaluate(mach, state.h, alpha, stage, powered)
        if alpha_limited:
            self._alpha_limit_hits += 1

        return Forces(
            thrust=thrust,
            drag=qs * coeffs.CD,
            normal=qs * coeffs.CN,
            alpha=alpha,
            mdot=mdot,
            phase=self._phase_label(state.t),
            extras={
                "mach": mach,
                "q": q,
                "CN_required": cn_required,
                "alpha_limited": 1.0 if alpha_limited else 0.0,
                "CD": coeffs.CD,
                "CN": coeffs.CN,
                "stage": float(stage),
                "S_ref": self._S_ref[stage],
            },
        )

    # -------------------------------------------------------------------- flying ---

    def fly(
        self,
        dt: float = 0.02,
        t_max: float = 600.0,
        adaptive: bool = True,
        tolerance: float = 1.0e-7,
    ) -> TrajectoryResult:
        """Fly the ascent and return the trajectory.

        Termination, in priority order (SOURCES["traj_iv1.termination"]):

          slant_range    sqrt(x^2 + h^2) >= reqs.slant_range_min. Checked FIRST, so an
                         intercept reached while descending is a success.
          ground_impact  h <= 0 while descending.
          t_max          the clock ran out. Reported as not converged.
          stalled        the speed fell below `velocity_floor`, after having once
                         exceeded it.

        `converged` is True for every physical termination and False on `t_max`.
        """
        self._reset_flight_state()
        # Motor state, if the motor carries any. The sizing loop flies one motor many
        # times, and the single-stage mission had to do the same for its pulse ignition.
        reset = getattr(self.motor, "reset", None)
        if callable(reset):
            reset()
        self._read_motor_timeline()

        result = TrajectoryResult()
        cn_required: list[float] = []
        alpha_limited: list[bool] = []
        stepper = StagedStepper(self.force_model)

        boundaries = self._hard_boundaries(t_max)
        pending = self._pending_events()
        next_boundary = 0

        state = FlightState(
            t=0.0,
            V=self.v_start,
            gamma=float(self.reqs.gamma_launch),
            x=0.0,
            h=float(self.reqs.h_launch),
            m=self.m0,
        )
        r_min = float(self.reqs.slant_range_min)

        def slant_residual(s: FlightState) -> float:
            return slant_range(s.x, s.h) - r_min

        reason = ""
        step_dt = dt

        while True:
            # Latched once per step, from the time at the START of the step. See
            # `_segment_stage` and `_is_pitching`.
            self._active_stage = self._segment_stage(state.t)
            self._pitch_armed = state.t >= float(self.dv.t_pitch) - EVENT_EPS
            forces = self.force_model(state)
            k1 = stepper.rhs(forces, state)
            self._record(result, state, forces, cn_required, alpha_limited)
            if state.V >= self.velocity_floor:
                self._stall_armed = True

            # Termination on the sample just recorded. The first sample is exempt: at
            # launch the vehicle sits on the ground at 1 m/s, which would trip both the
            # ground test and the stall test.
            if len(result.time) > 1:
                if slant_residual(state) >= 0.0:
                    reason, result.converged = "slant_range", True
                    break
                if state.h <= 0.0 and math.sin(state.gamma) < 0.0:
                    reason, result.converged = "ground_impact", True
                    break
                if self._stall_armed and state.V < self.velocity_floor:
                    reason, result.converged = "stalled", True
                    break
            if state.t >= t_max - EVENT_EPS:
                reason = "t_max"
                break

            # Clamp the step to the next hard boundary and to t_max, so the run ends
            # exactly on t_max and no RK4 stage straddles a discontinuity.
            while (
                next_boundary < len(boundaries)
                and boundaries[next_boundary] <= state.t + EVENT_EPS
            ):
                next_boundary += 1
            horizon = t_max
            if next_boundary < len(boundaries):
                horizon = min(horizon, boundaries[next_boundary])
            # The last step of a segment absorbs any rounding fragment rather than
            # leaving a step of 1e-14 s behind, which would put the event a rounding
            # before its own boundary.
            remaining = horizon - state.t
            at_horizon = remaining <= step_dt * (1.0 + 1.0e-9)
            this_dt = remaining if at_horizon else step_dt

            was_pitching = self._is_pitching(state)
            if adaptive:
                state_next, step_dt = stepper.adaptive_step(
                    state, this_dt, tolerance, 1.0e-4, 0.5, k1
                )
            else:
                state_next = stepper.step(state, this_dt, k1)
            # Snap the clock onto the boundary the step aimed at. `t + (horizon - t)` is
            # only exact to a rounding, and over many boundaries those roundings would
            # accumulate and put the recorded event times off the motor's own times.
            if at_horizon and abs(state_next.t - horizon) < 1.0e-9:
                state_next.t = horizon
            taken_dt = state_next.t - state.t

            # --- priority 1: the slant range was reached inside this step ---
            if slant_residual(state_next) >= 0.0:
                refined = stepper.bisect(state, taken_dt, slant_residual, k1)
                if refined.h >= -REFINE_ATOL:
                    self._record(
                        result,
                        refined,
                        self.force_model(refined),
                        cn_required,
                        alpha_limited,
                    )
                    state = refined
                    reason, result.converged = "slant_range", True
                    break
                # The ground came first inside this step. Fall through and report that.

            # --- priority 2: the ground was reached inside this step, descending ---
            if state_next.h <= 0.0 and state_next.h < state.h:
                refined = stepper.bisect(state, taken_dt, lambda s: -s.h, k1)
                self._record(
                    result, refined, self.force_model(refined), cn_required, alpha_limited
                )
                state = refined
                reason, result.converged = "ground_impact", True
                break

            # --- the commanded pitchover finished inside this step ---
            if was_pitching and state_next.gamma < self.dv.gamma_pitch:
                state_next = stepper.bisect(
                    state,
                    taken_dt,
                    lambda s: self.dv.gamma_pitch - s.gamma,
                    k1,
                    atol=1.0e-12,
                )
                self._t_pitch_complete = state_next.t
                self._add_event(
                    "pitchover_complete",
                    state_next,
                    state_next.m,
                    note=(
                        f"flight path angle reached {math.degrees(self.dv.gamma_pitch):.2f} "
                        f"deg {state_next.t - self.dv.t_pitch:.2f} s after the command"
                    ),
                )

            state = state_next

            # --- time-triggered events, including the separation mass jettison ---
            while pending and state.t >= pending[0][0] - EVENT_EPS:
                _t_event, _rank, name = pending.pop(0)
                if name == "separation":
                    state = self._separate(
                        result, state, cn_required, alpha_limited
                    )
                else:
                    self._add_event(name, state, state.m)

        result.CN_required = cn_required
        result.alpha_limited = alpha_limited
        self._finish(result, reason, dt, adaptive)
        return result

    # ------------------------------------------------------------------- staging ---

    def _separate(
        self,
        result: TrajectoryResult,
        state: FlightState,
        cn_required: list[float],
        alpha_limited: list[bool],
    ) -> FlightState:
        """Jettison stage 1 in one step and switch the aero to stage 2.

        Two samples are written at the separation time, one on each side of the jettison,
        so the mass step and the reference-area step are visible in the trajectory arrays
        instead of being smeared over a step. See SOURCES["traj_iv1.separation"].
        """
        self._record(result, state, self.force_model(state), cn_required, alpha_limited)
        mass_before = state.m
        mass_after = mass_before - self._m_jettison
        if mass_after <= 0.0:
            raise ValueError(
                f"jettisoning {self._m_jettison:.1f} kg at separation would leave "
                f"{mass_after:.1f} kg; the mass statement and the motor disagree"
            )
        after = FlightState(
            t=state.t, V=state.V, gamma=state.gamma, x=state.x, h=state.h, m=mass_after
        )
        self._separated = True
        self._separation_index = len(result.time)
        self._add_event(
            "separation",
            after,
            mass_before,
            mass_after,
            note=(
                f"stage-1 inert plus interstage, {self._m_jettison:.1f} kg, left the "
                f"vehicle; reference area {self._S_ref[1]:.4f} -> "
                f"{self._S_ref[self._stage_after_separation]:.4f} m^2"
            ),
        )
        return after

    def _add_event(
        self,
        name: str,
        state: FlightState,
        mass_before: float,
        mass_after: float | None = None,
        note: str = "",
    ) -> None:
        _rho, _p, _T, sound = atmosphere_properties(state.h)
        self._events.append(
            StageEvent(
                name=name,
                time=state.t,
                altitude=state.h,
                mach=state.V / sound if sound > 0.0 else 0.0,
                mass_before=mass_before,
                mass_after=mass_before if mass_after is None else mass_after,
                note=note,
            )
        )

    # -------------------------------------------------------------- bookkeeping ---

    @staticmethod
    def _record(
        result: TrajectoryResult,
        state: FlightState,
        f: Forces,
        cn_required: list[float],
        alpha_limited: list[bool],
    ) -> None:
        """Append one sample. Same field set as the single-stage integrator records."""
        result.time.append(state.t)
        result.x.append(state.x)
        result.h.append(state.h)
        result.V.append(state.V)
        result.mach.append(f.extras.get("mach", 0.0))
        result.mass.append(state.m)
        result.gamma.append(state.gamma)
        result.thrust.append(f.thrust)
        result.drag.append(f.drag)
        result.q.append(f.extras.get("q", 0.0))
        result.alpha.append(f.alpha)
        result.phase.append(f.phase)
        cn_required.append(f.extras.get("CN_required", 0.0))
        alpha_limited.append(bool(f.extras.get("alpha_limited", 0.0)))

    def _build_intercept(self, result: TrajectoryResult, termination: str) -> InterceptResult:
        """Fill `InterceptResult` from the final sample, including A11 lateral g."""
        if not result.time:
            return InterceptResult(termination=termination)
        x, h = result.x[-1], result.h[-1]
        mach, mass, q = result.mach[-1], result.mass[-1], result.q[-1]
        stage = self._stage_after_separation if self._separated else 1
        s_ref = self._S_ref[stage]
        cn_max = float(self.aero.CN_max(mach, h, stage, self.reqs.alpha_max))
        # Read off the terminal state, not off the termination label. They agree except in
        # one case: a step in which the ground was crossed before the slant range was
        # satisfied ends on 'ground_impact' with the range requirement nonetheless met.
        reached = slant_range(x, h) >= self.reqs.slant_range_min - REFINE_ATOL
        return InterceptResult(
            reached_slant_range=reached,
            slant_range=slant_range(x, h),
            ground_range=x,
            altitude=h,
            mach=mach,
            velocity=result.V[-1],
            time=result.time[-1],
            mass=mass,
            q=q,
            lateral_g_available=lateral_g(q, s_ref, cn_max, mass),
            termination=termination,
        )

    def _guidance_segments(self, t_end: float) -> list[dict[str, Any]]:
        """Run-length record of the guidance programme, for the report and the tests."""
        segments: list[dict[str, Any]] = []
        if not self._pitch_enabled:
            segments.append({"name": "gravity_turn", "t_start": 0.0, "t_end": t_end})
            return segments
        t_pitch = min(float(self.dv.t_pitch), t_end)
        segments.append({"name": "vertical_rise", "t_start": 0.0, "t_end": t_pitch})
        t_complete = (
            t_end if math.isnan(self._t_pitch_complete) else self._t_pitch_complete
        )
        segments.append({"name": "pitchover", "t_start": t_pitch, "t_end": t_complete})
        if t_complete < t_end:
            segments.append(
                {"name": "gravity_turn", "t_start": t_complete, "t_end": t_end}
            )
        return segments

    def _finish(
        self,
        result: TrajectoryResult,
        reason: str,
        dt: float,
        adaptive: bool,
    ) -> None:
        """Assemble the message, the diagnostics and the intercept summary."""
        self._intercept = self._build_intercept(result, reason)
        intercept = self._intercept
        samples = len(result.time)
        h_peak = max(result.h) if result.h else 0.0
        limit_fraction = (
            self._alpha_limit_hits / float(self._force_calls) if self._force_calls else 0.0
        )
        h_over = max(0.0, h_peak - ATMOSPHERE_H_MAX)

        notes: list[str] = [
            {
                "slant_range": (
                    f"reached the {self.reqs.slant_range_min / 1000.0:.1f} km slant range "
                    f"at t = {intercept.time:.2f} s, h = {intercept.altitude / 1000.0:.2f} "
                    f"km, Mach {intercept.mach:.2f}"
                ),
                "ground_impact": (
                    f"fell back to the ground at t = {intercept.time:.2f} s with only "
                    f"{intercept.slant_range / 1000.0:.1f} km of slant range"
                ),
                "t_max": f"t_max {result.time[-1] if result.time else 0.0:.1f} s reached",
                "stalled": (
                    f"speed fell below the {self.velocity_floor:.0f} m/s floor at "
                    f"t = {intercept.time:.2f} s"
                ),
            }.get(reason, reason)
        ]
        if math.isnan(self._t_pitch_complete) and self._pitch_enabled:
            notes.append(
                f"the commanded pitchover to {math.degrees(self.dv.gamma_pitch):.1f} deg "
                "never finished"
            )
        elif self._pitch_enabled:
            commanded = (
                self.reqs.gamma_launch - self.dv.gamma_pitch
            ) / self.dv.pitch_rate_max
            flown = self._t_pitch_complete - self.dv.t_pitch
            if flown > 1.05 * commanded:
                notes.append(
                    f"the pitchover took {flown:.1f} s against {commanded:.1f} s "
                    "commanded, because there is no thrust-vector control and the "
                    "aerodynamic normal force at the alpha limit could not turn faster"
                )
        if self._alpha_limit_hits:
            notes.append(
                f"trim angle of attack hit the {math.degrees(self.alpha_max):.0f} deg "
                f"limit on {limit_fraction * 100.0:.0f} percent of force evaluations; "
                "control authority is short there"
            )
        if self._q_floor_hit:
            notes.append(
                f"dynamic pressure fell below the {Q_FLOOR:.0f} Pa floor; the required "
                "normal force was frozen at zero in that region"
            )
        # A3, A4 and A11 are conditions AT the intercept, so they are only meaningful once
        # the range condition is met. The sizing loop owns the constraint residuals; these
        # notes exist so a shortfall is visible in a single flight without running a loop.
        if intercept.reached_slant_range:
            if intercept.altitude < self.reqs.h_intercept_min:
                notes.append(
                    f"intercept altitude {intercept.altitude / 1000.0:.2f} km is below the "
                    f"A3 minimum of {self.reqs.h_intercept_min / 1000.0:.1f} km"
                )
            if intercept.mach < self.reqs.mach_intercept_min:
                notes.append(
                    f"intercept Mach {intercept.mach:.2f} is below the A4 minimum of "
                    f"{self.reqs.mach_intercept_min:.2f}"
                )
            if intercept.lateral_g_available < self.reqs.lateral_g_min:
                notes.append(
                    f"available lateral acceleration {intercept.lateral_g_available:.1f} g "
                    f"is below the A11 minimum of {self.reqs.lateral_g_min:.1f} g"
                )
        if self._separation_index < 0:
            notes.append("separation never happened inside the flown time")
        if h_over > 0.0:
            notes.append(
                f"flew {h_over / 1000.0:.1f} km above the {ATMOSPHERE_H_MAX / 1000.0:.0f} "
                "km ceiling of rocketgen.sizing.atmosphere, which CLAMPS there; density "
                "and therefore drag are overstated above the ceiling"
            )
        expected_separation = self._t_burnout[1] + self.reqs.t_coast_separation
        if abs(self._t_separation - expected_separation) > 1.0e-6:
            notes.append(
                f"the motor separates at t = {self._t_separation:.3f} s but "
                f"reqs.t_coast_separation implies {expected_separation:.3f} s; the motor "
                "and the requirements disagree about the coast"
            )
        if result.q_max > self.reqs.q_max:
            notes.append(
                f"peak dynamic pressure {result.q_max / 1000.0:.0f} kPa exceeds the A10 "
                f"limit of {self.reqs.q_max / 1000.0:.0f} kPa"
            )
        if ATMOSPHERE_SOURCE.startswith("trajectory.py"):
            notes.append(
                "atmosphere came from the inline US-1976 fallback, not from "
                "rocketgen.sizing.atmosphere"
            )
        result.message = "; ".join(n for n in notes if n)

        h_burnout_1 = next(
            (e.altitude for e in self._events if e.name == "stage_1_burnout"), float("nan")
        )
        self.diagnostics = {
            "termination": reason,
            "events": [asdict(e) for e in self._events],
            "intercept": asdict(intercept),
            "guidance_segments": self._guidance_segments(
                result.time[-1] if result.time else 0.0
            ),
            "reached_slant_range": intercept.reached_slant_range,
            "slant_range": intercept.slant_range,
            "lateral_g_available": intercept.lateral_g_available,
            "q_max": result.q_max,
            "q_max_limit": self.reqs.q_max,
            "h_apogee": h_peak,
            "h_stage_1_burnout": h_burnout_1,
            "h_above_atmosphere_table": h_over,
            "atmosphere_table_ceiling": ATMOSPHERE_H_MAX,
            "atmosphere_source": ATMOSPHERE_SOURCE,
            "t_pitch_start": float(self.dv.t_pitch) if self._pitch_enabled else float("nan"),
            "t_pitch_complete": self._t_pitch_complete,
            "t_separation": self._t_separation,
            "separation_index": self._separation_index,
            "mass_jettisoned": self._m_jettison if self._separation_index >= 0 else 0.0,
            "m0": self.m0,
            "alpha_limit_hits": float(self._alpha_limit_hits),
            "alpha_limit_fraction": limit_fraction,
            "q_floor_hit": self._q_floor_hit,
            "steps": float(samples),
            "force_calls": float(self._force_calls),
            "dt_requested": dt,
            "adaptive": adaptive,
        }
        result.diagnostics = dict(self.diagnostics)


# --------------------------------------------------------------------------------------
#   Small helpers for the report and the sizing loop
# --------------------------------------------------------------------------------------


def slant_range_history(result: TrajectoryResult) -> list[float]:
    """Slant range from the launch point at each sample, m."""
    return [slant_range(x, h) for x, h in zip(result.x, result.h)]


def event_time(events: list[StageEvent], name: str) -> float:
    """Time of the named event, or NaN when it never happened."""
    for event in events:
        if event.name == name:
            return event.time
    return float("nan")
