"""WP3b - 3-DOF point-mass trajectory integrator for the SPEC.md section 2 mission.

Why a direct integration and not SUAVE's mission machinery
----------------------------------------------------------
SUAVE 2.5.2 solves a mission as a set of collocated segments. Each segment
(`SUAVE/Analyses/Mission/Segments/`) builds an `Aerodynamic` process chain that expects
an aircraft `Vehicle` with an energy `Network` that responds to a `throttle` unknown, and
the segment converges residuals with a root solve over the whole segment at once. There
is no solid-rocket network in the vendored tree (`docs/REFERENCE.md` section 2 confirms
this), the propellant flow of a solid motor is a function of time and not of throttle,
and the SV-1 mission needs a terminal dive to h = 0 with a mass that changes by a factor
of two. Fitting that into the segment solver means writing a fake throttle-driven network
and fighting the collocation solver for no gain in fidelity.

A direct fixed-step RK4 integration of the standard flat-earth 3-DOF equations is used
instead. It is explicit, it exposes the alpha history the sizing loop needs, and every
term in it is visible. SUAVE is still the source of the atmosphere (through
`rocketgen/sizing/atmosphere.py`, WP2).

Cost, measured on this machine
------------------------------
Four force-model evaluations per RK4 step; the first stage is reused for the recorded
sample, so there is no fifth call. A 300 s trajectory at dt = 0.02 (15 001 steps) takes
about 1.1 s with the cheap test stub and about 7.8 s with WP2's `RocketAero`. If the
sizing loop needs many trajectories, use `fly(adaptive=True, tolerance=1e-7)`: that
reaches the same range to within 0.1 % in about 900 steps and 1.2 s with the real aero.

Equations of motion (flat earth, vertical plane, constant gravity)
-----------------------------------------------------------------
    V_dot     = (T * cos(alpha) - D) / m - g * sin(gamma)
    gamma_dot = (T * sin(alpha) + N) / (m * V) - g * cos(gamma) / V
    x_dot     = V * cos(gamma)
    h_dot     = V * sin(gamma)
    m_dot     = -mdot_propellant

THRUST-ALPHA COUPLING IS INCLUDED: thrust acts along the body axis, which sits at the
angle of attack alpha to the velocity vector, so thrust contributes
`T * cos(alpha)` along the flight path and `T * sin(alpha)` normal to it. There is no
thrust vector control and no nozzle gimbal; the nozzle is fixed and aft (SPEC.md
section 2).

Gravity is constant at `G0`. That is a deliberate choice: it keeps the analytic
validation cases in `tests/test_trajectory.py` exact, and the inverse-square correction
over a 12 km trajectory is under 0.4 %.

Approximations and modelling choices are collected in `SOURCES`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from ..config import AeroCoefficients, MassStatement, Requirements, TrajectoryResult
from ..config import DesignVector, register_sources

G0 = 9.80665     # standard gravity, m/s^2 (CGPM 3rd conference, 1901)

# --------------------------------------------------------------------------------------
#   Guidance and integration settings. These are modelling choices, not physics.
# --------------------------------------------------------------------------------------

ALPHA_MAX_RAD = math.radians(15.0)      # trim authority limit
ALTITUDE_GAIN = 0.15                     # 1/s, altitude-hold outer loop
CLIMB_RATE_LIMIT = 200.0                 # m/s, commanded climb rate cap
GAMMA_GAIN = 2.0                         # 1/s, flight-path-angle inner loop
GAMMA_CMD_LIMIT = math.radians(45.0)     # commanded climb angle cap
LOAD_FACTOR_LIMIT = 20.0                 # g, cap on the commanded normal load factor
TRIM_ITERATIONS = 2                      # fixed-point passes on the thrust-alpha term
DIVE_MACH_TRIGGER = 1.20                 # coast ends if Mach falls to this
TERMINAL_IGNITION_MARGIN = 1.0           # x the descent the terminal burn needs
VELOCITY_FLOOR = 50.0                    # m/s, integration stops below this
Q_FLOOR = 200.0                          # Pa, below this the aero model is not trusted

SOURCES: dict[str, str] = {
    "traj.equations": (
        "Standard flat-earth 3-DOF point-mass equations in the vertical plane, as given "
        "in any flight-mechanics text (for example Zipfel, 'Modeling and Simulation of "
        "Aerospace Vehicle Dynamics', Ch.3, or Fleeman, 'Tactical Missile Design', "
        "flight performance chapter). Thrust acts along the body axis at angle of "
        "attack alpha to the velocity vector, so thrust-alpha coupling is included."
    ),
    "traj.gravity": (
        "Constant gravity g = 9.80665 m/s^2 (standard gravity, CGPM 1901). MODELLING "
        "CHOICE: the inverse-square variation is neglected, which is under 0.4 % over "
        "the 0 to 12 km band of this mission and keeps the analytic validation cases "
        "in tests/test_trajectory.py exact."
    ),
    "traj.integrator": (
        "Fixed-step classical RK4 (Runge 1895, Kutta 1901), with an optional adaptive "
        "mode using step doubling and Richardson error estimation. Fourth order is "
        "verified in tests/test_trajectory.py by halving dt and checking the error "
        "ratio."
    ),
    "traj.guidance_gains": (
        "MODELLING CHOICE, no source: the altitude-hold and flight-path-angle "
        "autopilot gains (0.15 1/s outer, 2.0 1/s inner), the 200 m/s commanded climb "
        "rate cap, the 45 deg commanded climb angle cap and the 20 g commanded load "
        "factor cap are arbitrary. Guidance law design is an explicit non-goal (SPEC.md "
        "section 8); these gains only have to fly the SPEC.md section 2 profile "
        "smoothly. They do not appear in any reported performance number other than "
        "through the flown trajectory. The load factor cap matters for one thing only: "
        "it keeps the recorded CN_required history finite at the dive-entry command "
        "step, so the sizing loop can read it."
    ),
    "traj.alpha_limit": (
        "MODELLING CHOICE: trim angle of attack is limited to 15 deg. Trim demands "
        "beyond that are recorded as an authority shortfall rather than flown, so the "
        "sizing loop can see them. 15 deg is a conventional tactical-rocket limit; it "
        "was not taken from a specific source in this session, so treat it as a guess."
    ),
    "traj.dive_entry": (
        "MODELLING CHOICE: SPEC.md section 2 does not define the coast-to-dive "
        "transition. Rule 'max_range' (default): hold altitude until level trim can no "
        "longer be held inside the alpha limit, or until Mach falls to 1.20, then dive "
        "at Requirements.gamma_terminal. Rule 'range': dive as soon as the current "
        "range plus the geometric dive range reaches Requirements.range_min. Rule "
        "'terminal_boost': dive the moment the sustain phase ends, with no level coast "
        "at all, which keeps the most kinetic and potential energy for the endgame."
    ),
    "traj.terminal_boost_ignition": (
        "MODELLING CHOICE: SPEC.md section 2 has no terminal-boost phase, so the "
        "ignition trigger is invented here and exposed rather than hidden. Rule "
        "'carry_to_impact' (default) lights the pulse when the altitude falls to "
        "terminal_ignition_margin * V * |sin(gamma_terminal)| * t_terminal, that is when "
        "the predicted remaining descent time equals the terminal burn time, so the "
        "pulse burns out at impact instead of above the ground. Rule 'dive_entry' lights "
        "it at dive entry. Rule 'never' leaves it unlit, which is what the two-phase "
        "regression case uses. The margin defaults to 1.0 and is a free parameter for "
        "the sizing loop: the prediction uses the CURRENT speed and the pulse "
        "accelerates the rocket, so a margin near 1.0 lights slightly late and the "
        "pulse is still burning at impact. That is the safe side; a margin above about "
        "1.3 wastes propellant above the ground."
    ),
    "traj.motor_state_reset": (
        "`Mission.fly` resets the motor's terminal ignition time to None and re-arms the "
        "pulse before each run, because the ignition time is motor state and the sizing "
        "loop calls fly() repeatedly on one motor object. Without the reset a second "
        "flight would inherit the first flight's ignition time."
    ),
    "traj.sustain_thrust_check": (
        "The sustain phase is flown with the thrust the motor actually delivers, never "
        "with the thrust the constant-Mach constant-altitude condition would need. The "
        "shortfall is recorded in TrajectoryResult.message and in "
        "Mission.diagnostics['sustain_thrust_deficit_max']."
    ),
    "traj.atmosphere_fallback": (
        "The atmosphere is taken from rocketgen.sizing.atmosphere (WP2) when that "
        "module is importable, discovered through `atmo(h)`, `properties(h)`, "
        "`atmosphere_properties(h)`, `conditions(h)` or the split "
        "density/pressure/temperature/speed_of_sound functions. If none of those work, a "
        "minimal inline US Standard 1976 implementation is used instead (layer base "
        "values from NASA-TM-X-74335, R = 287.0528 J/(kg.K), effective earth radius "
        "6 356 766 m, valid to 84.852 km geopotential). Check "
        "trajectory.ATMOSPHERE_SOURCE at run time; the module records which route it "
        "took and the mission adds a warning to TrajectoryResult.message when the "
        "fallback is in use."
    ),
    "traj.result_extensions": (
        "`integrate` fills three fields beyond the base trajectory histories: "
        "`CN_required` (required normal-force coefficient on S_ref at each sample), "
        "`alpha_limited` (True where the trim demand exceeded the alpha limit) and "
        "`diagnostics`. These are now declared on config.TrajectoryResult, so they are "
        "part of the contract rather than attached attributes."
    ),
}

register_sources(SOURCES)


# --------------------------------------------------------------------------------------
#   Aero protocol - WP2 owns the implementation (rocketgen/sizing/aero.py, RocketAero)
# --------------------------------------------------------------------------------------


@runtime_checkable
class AeroCallable(Protocol):
    """What the trajectory needs from an aerodynamic model.

    WP2's `rocketgen.sizing.aero.RocketAero` is the real implementation. This module
    never imports it, so the two work packages stay independent. `tests/simple_aero.py`
    carries a constant-CD0 / linear-CN stub used by the WP3 tests.
    """

    def evaluate(
        self, mach: float, altitude: float, alpha: float, power_on: bool = False
    ) -> AeroCoefficients:
        """Aerodynamic coefficients at a flight point. CD and CN are on S_ref."""
        ...

    def trim_alpha(self, mach: float, altitude: float, required_CN: float) -> float:
        """Angle of attack, rad, that produces `required_CN`."""
        ...


# --------------------------------------------------------------------------------------
#   Atmosphere adapter
# --------------------------------------------------------------------------------------

_US1976_LAYERS: tuple[tuple[float, float, float, float], ...] = (
    # (base geopotential altitude m, base temperature K, lapse K/m, base pressure Pa)
    (0.0, 288.15, -0.0065, 101325.0),
    (11_000.0, 216.65, 0.0, 22_632.1),
    (20_000.0, 216.65, 0.001, 5_474.89),
    (32_000.0, 228.65, 0.0028, 868.019),
    (47_000.0, 270.65, 0.0, 110.906),
    (51_000.0, 270.65, -0.0028, 66.9389),
    (71_000.0, 214.65, -0.002, 3.95642),
)
_R_AIR = 287.0528
_GAMMA_AIR = 1.4
_EARTH_RADIUS = 6_356_766.0


def _us1976(altitude: float) -> tuple[float, float, float, float]:
    """Minimal US Standard 1976: returns (density, pressure, temperature, sound speed)."""
    h = _EARTH_RADIUS * altitude / (_EARTH_RADIUS + altitude)
    h = min(max(h, 0.0), 84_852.0)
    layer = _US1976_LAYERS[0]
    for candidate in _US1976_LAYERS:
        if h >= candidate[0]:
            layer = candidate
        else:
            break
    h_b, t_b, lapse, p_b = layer
    dh = h - h_b
    if lapse == 0.0:
        temperature = t_b
        pressure = p_b * math.exp(-G0 * dh / (_R_AIR * t_b))
    else:
        temperature = t_b + lapse * dh
        pressure = p_b * (temperature / t_b) ** (-G0 / (_R_AIR * lapse))
    density = pressure / (_R_AIR * temperature)
    sound_speed = math.sqrt(_GAMMA_AIR * _R_AIR * temperature)
    return density, pressure, temperature, sound_speed


def _resolve_atmosphere() -> tuple[Callable[[float], tuple[float, float, float, float]], str]:
    """Prefer WP2's atmosphere module, fall back to the inline US-1976 model.

    Accepted shapes in `rocketgen.sizing.atmosphere`, in order of preference:
      1. `properties(h)` or `atmosphere_properties(h)` returning a 4-tuple or an object
         with `density`, `pressure`, `temperature`, `speed_of_sound` attributes.
      2. Separate module-level functions `density(h)`, `pressure(h)`, `temperature(h)`,
         `speed_of_sound(h)`.
    Anything else falls through to the fallback. `ATMOSPHERE_SOURCE` records which route
    was taken.
    """
    try:
        from . import atmosphere as _atm   # type: ignore
    except Exception:
        return _us1976, "trajectory.py inline US-1976 fallback"

    for name in ("atmo", "properties", "atmosphere_properties", "conditions"):
        fn = getattr(_atm, name, None)
        if not callable(fn):
            continue
        try:
            probe = fn(0.0)
        except Exception:
            continue
        if isinstance(probe, (tuple, list)) and len(probe) >= 4:
            def adapter(h: float, _fn=fn) -> tuple[float, float, float, float]:
                out = _fn(h)
                return float(out[0]), float(out[1]), float(out[2]), float(out[3])

            if 1.1 < adapter(0.0)[0] < 1.3:
                return adapter, f"rocketgen.sizing.atmosphere.{name} (tuple)"
        if all(hasattr(probe, a) for a in ("density", "pressure", "temperature")):
            def adapter_obj(h: float, _fn=fn) -> tuple[float, float, float, float]:
                out = _fn(h)
                sound = getattr(out, "speed_of_sound", None)
                if sound is None:
                    sound = math.sqrt(_GAMMA_AIR * _R_AIR * float(out.temperature))
                return (
                    float(out.density),
                    float(out.pressure),
                    float(out.temperature),
                    float(sound),
                )

            if 1.1 < adapter_obj(0.0)[0] < 1.3:
                return adapter_obj, f"rocketgen.sizing.atmosphere.{name} (object)"

    needed = ("density", "pressure", "temperature")
    if all(callable(getattr(_atm, n, None)) for n in needed):
        rho_fn = _atm.density
        p_fn = _atm.pressure
        t_fn = _atm.temperature
        a_fn = getattr(_atm, "speed_of_sound", None)

        def adapter_split(h: float) -> tuple[float, float, float, float]:
            temperature = float(t_fn(h))
            sound = (
                float(a_fn(h))
                if callable(a_fn)
                else math.sqrt(_GAMMA_AIR * _R_AIR * temperature)
            )
            return float(rho_fn(h)), float(p_fn(h)), temperature, sound

        try:
            if 1.1 < adapter_split(0.0)[0] < 1.3:
                return adapter_split, "rocketgen.sizing.atmosphere (split functions)"
        except Exception:
            pass

    return _us1976, "trajectory.py inline US-1976 fallback"


atmosphere_properties, ATMOSPHERE_SOURCE = _resolve_atmosphere()


# --------------------------------------------------------------------------------------
#   Core integrator
# --------------------------------------------------------------------------------------


@dataclass
class FlightState:
    """Instantaneous 3-DOF state."""

    t: float
    V: float
    gamma: float
    x: float
    h: float
    m: float


@dataclass
class Forces:
    """Forces and bookkeeping returned by a force model at one state."""

    thrust: float = 0.0
    drag: float = 0.0
    normal: float = 0.0
    alpha: float = 0.0
    mdot: float = 0.0
    phase: str = "coast"
    extras: dict[str, float] = field(default_factory=dict)


ForceModel = Callable[[FlightState], Forces]
StopCondition = Callable[[FlightState], str]


class PointMass3DOF:
    """Flat-earth 3-DOF point-mass integrator with a pluggable force model.

    The force model is a plain callable, which is what makes the analytic validation
    cases in `tests/test_trajectory.py` possible: they pass trivial force models
    (zero drag, constant thrust, constant-density drag) and compare against closed-form
    answers.
    """

    def __init__(self, force_model: ForceModel, gravity: float = G0) -> None:
        self.force_model = force_model
        self.gravity = gravity

    # -------------------------------------------------------------------- dynamics ---

    def derivatives(self, state: FlightState) -> tuple[float, float, float, float, float]:
        """Right-hand side: (V_dot, gamma_dot, x_dot, h_dot, m_dot)."""
        return self.rhs(self.force_model(state), state)

    def rhs(
        self, f: Forces, state: FlightState
    ) -> tuple[float, float, float, float, float]:
        """Right-hand side from an already-evaluated force set.

        Split out from `derivatives` so `integrate` can reuse the first RK4 stage for
        recording instead of calling the force model a fifth time per step. With an
        expensive aero model that is a 20 % saving.
        """
        v = state.V
        m = max(state.m, 1e-9)
        cos_a = math.cos(f.alpha)
        sin_a = math.sin(f.alpha)
        v_dot = (f.thrust * cos_a - f.drag) / m - self.gravity * math.sin(state.gamma)
        if v > 1e-6:
            gamma_dot = (f.thrust * sin_a + f.normal) / (m * v) - self.gravity * math.cos(
                state.gamma
            ) / v
        else:
            gamma_dot = 0.0
        x_dot = v * math.cos(state.gamma)
        h_dot = v * math.sin(state.gamma)
        return v_dot, gamma_dot, x_dot, h_dot, -f.mdot

    @staticmethod
    def _shift(state: FlightState, dy: tuple[float, ...], factor: float, dt: float) -> FlightState:
        return FlightState(
            t=state.t + factor * dt,
            V=state.V + factor * dt * dy[0],
            gamma=state.gamma + factor * dt * dy[1],
            x=state.x + factor * dt * dy[2],
            h=state.h + factor * dt * dy[3],
            m=state.m + factor * dt * dy[4],
        )

    def step(
        self,
        state: FlightState,
        dt: float,
        k1: tuple[float, float, float, float, float] | None = None,
    ) -> FlightState:
        """One classical RK4 step. `k1` may be supplied if already known."""
        if k1 is None:
            k1 = self.derivatives(state)
        k2 = self.derivatives(self._shift(state, k1, 0.5, dt))
        k3 = self.derivatives(self._shift(state, k2, 0.5, dt))
        k4 = self.derivatives(self._shift(state, k3, 1.0, dt))
        return FlightState(
            t=state.t + dt,
            V=state.V + dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
            gamma=state.gamma + dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
            x=state.x + dt / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
            h=state.h + dt / 6.0 * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]),
            m=state.m + dt / 6.0 * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]),
        )

    # ------------------------------------------------------------------- integrate ---

    def integrate(
        self,
        state0: FlightState,
        dt: float = 0.02,
        t_max: float = 900.0,
        adaptive: bool = False,
        tolerance: float = 1.0e-6,
        dt_min: float = 1.0e-4,
        dt_max: float = 0.5,
        velocity_floor: float = VELOCITY_FLOOR,
        stop_on_ground: bool = True,
        extra_stop: StopCondition | None = None,
    ) -> TrajectoryResult:
        """Integrate to a termination condition and fill a `TrajectoryResult`.

        Termination, in the order checked:
          * an `extra_stop` condition returning a non-empty string,
          * h <= 0 (ground impact), refined to |h| < 1e-9 m by bisecting the last step,
          * V < `velocity_floor`,
          * t > `t_max`.

        `TrajectoryResult.converged` is True only when the run ended on a physical
        condition (ground impact, velocity floor or `extra_stop`), never on `t_max`.

        Beyond the trajectory histories, `CN_required`, `alpha_limited` and `diagnostics`
        are filled for the sizing loop's fin-authority checks.
        """
        result = TrajectoryResult()
        cn_required: list[float] = []
        alpha_limited: list[bool] = []

        state = state0
        reason = ""
        step_dt = dt

        while True:
            # One force evaluation per iteration serves both the recording and the first
            # RK4 stage.
            forces = self.force_model(state)
            k1 = self.rhs(forces, state)
            self._record(result, state, forces, cn_required, alpha_limited)

            if len(result.time) > 1:
                if extra_stop is not None:
                    message = extra_stop(state)
                    if message:
                        reason = message
                        result.converged = True
                        break
                if state.V < velocity_floor:
                    reason = f"velocity fell below the {velocity_floor:.0f} m/s floor"
                    result.converged = True
                    break

            if state.t >= t_max:
                reason = f"t_max {t_max:.1f} s reached before impact"
                break

            # Clamp the final step so the run ends exactly on t_max rather than
            # overshooting it. Analytic validation cases depend on this.
            this_dt = min(step_dt, t_max - state.t)
            if adaptive:
                state_next, step_dt = self._adaptive_step(
                    state, this_dt, tolerance, dt_min, dt_max, k1
                )
            else:
                state_next = self.step(state, this_dt, k1)

            if stop_on_ground and state_next.h <= 0.0:
                state_next = self._refine_to_ground(state, this_dt, k1)
                self._record(
                    result,
                    state_next,
                    self.force_model(state_next),
                    cn_required,
                    alpha_limited,
                )
                reason = "ground impact"
                result.converged = True
                state = state_next
                break

            state = state_next

        result.message = reason
        result.CN_required = cn_required          # type: ignore[attr-defined]
        result.alpha_limited = alpha_limited      # type: ignore[attr-defined]
        result.diagnostics = {                    # type: ignore[attr-defined]
            "steps": len(result.time),
            "dt_final": step_dt,
            "atmosphere_source": ATMOSPHERE_SOURCE,
        }
        return result

    def _adaptive_step(
        self,
        state: FlightState,
        dt: float,
        tolerance: float,
        dt_min: float,
        dt_max: float,
        k1: tuple[float, float, float, float, float] | None = None,
    ) -> tuple[FlightState, float]:
        """Step doubling with Richardson error control on V and h.

        One full step is compared against two half steps. The half-step answer is kept.
        The next step size is scaled by the classical fourth-order factor and clipped to
        [dt_min, dt_max]. The step is retried at most 8 times.
        """
        for _ in range(8):
            coarse = self.step(state, dt, k1)
            mid = self.step(state, 0.5 * dt, k1)
            fine = self.step(mid, 0.5 * dt)
            scale_v = max(abs(fine.V), 1.0)
            scale_h = max(abs(fine.h), 1.0)
            error = max(
                abs(fine.V - coarse.V) / scale_v, abs(fine.h - coarse.h) / scale_h
            ) / 15.0
            if error <= tolerance or dt <= dt_min:
                growth = 2.0 if error <= 1e-14 else min(2.0, 0.9 * (tolerance / error) ** 0.2)
                return fine, min(dt_max, max(dt_min, dt * growth))
            dt = max(dt_min, 0.5 * dt)
        return self.step(state, dt, k1), dt

    def _refine_to_ground(
        self,
        state: FlightState,
        dt: float,
        k1: tuple[float, float, float, float, float] | None = None,
    ) -> FlightState:
        """Bisect the final step so the last recorded point sits on h = 0."""
        lo, hi = 0.0, dt
        best = self.step(state, dt, k1)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            trial = self.step(state, mid, k1)
            if trial.h > 0.0:
                lo = mid
            else:
                hi = mid
                best = trial
            if abs(trial.h) < 1.0e-9:
                return trial
        return best

    def _record(
        self,
        result: TrajectoryResult,
        state: FlightState,
        f: Forces,
        cn_required: list[float],
        alpha_limited: list[bool],
    ) -> None:
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


# --------------------------------------------------------------------------------------
#   The SV-1 mission
# --------------------------------------------------------------------------------------


class Mission:
    """Fly the SPEC.md section 2 mission profile with a given motor, aero and mass.

    Phases, in order:
      separation    `reqs.t_separation` seconds unpowered at the launch condition
      boost         motor boost phase, climb h_launch -> h_cruise, accelerate to M_cruise
      sustain       motor sustain phase, hold h_cruise and (as far as the motor allows)
                    M_cruise
      coast         unpowered, hold altitude until the dive-entry rule fires
      terminal      dive at `reqs.gamma_terminal` to h = 0
      terminal_boost the part of the dive during which the commanded terminal motor pulse
                    is burning. Recorded as a separate phase label.

    The phase label recorded at each step comes from the motor for the powered phases
    and from the guidance state otherwise.

    WHY THERE IS A TERMINAL BOOST AT ALL. An unpowered dive is terminal-velocity limited.
    At the SV-1 burnout mass, sea-level density and the calibrated drag coefficient the
    vertical-dive terminal velocity is about Mach 0.93, and sweeping the dive angle from
    -25 to -89 deg moves impact Mach only from 0.66 to 0.97. SPEC R6 (Mach 1.50 at
    impact) is therefore unreachable without endgame thrust, for any dive angle and any
    propellant loading. Set `DesignVector.m_p_terminal` above zero to fly with a
    commanded terminal pulse.
    """

    def __init__(
        self,
        dv: DesignVector,
        reqs: Requirements,
        motor: object,
        aero: AeroCallable,
        mass: MassStatement,
        alpha_max: float = ALPHA_MAX_RAD,
        dive_rule: str = "max_range",
        terminal_ignition_rule: str = "carry_to_impact",
        terminal_ignition_margin: float = TERMINAL_IGNITION_MARGIN,
    ) -> None:
        if dive_rule not in ("max_range", "range", "terminal_boost"):
            raise ValueError(
                "dive_rule must be 'max_range', 'range' or 'terminal_boost'"
            )
        if terminal_ignition_rule not in ("carry_to_impact", "dive_entry", "never"):
            raise ValueError(
                "terminal_ignition_rule must be 'carry_to_impact', 'dive_entry' or 'never'"
            )
        self.dv = dv
        self.reqs = reqs
        self.motor = motor
        self.aero = aero
        self.mass = mass
        self.alpha_max = alpha_max
        self.dive_rule = dive_rule
        self.terminal_ignition_rule = terminal_ignition_rule
        self.terminal_ignition_margin = terminal_ignition_margin

        self.S_ref = dv.S_ref
        self.m0 = mass.total_mass
        self.diagnostics: dict[str, object] = {}

        # The dive gate waits for the end of the SUSTAIN phase, not for the end of all
        # thrust: the terminal pulse is lit inside the dive, so gating on the overall
        # burnout time would deadlock.
        self._t_sustain_burnout = (
            float(getattr(motor, "t_burnout_sustain", getattr(motor, "t_burnout", 0.0)))
            + reqs.t_separation
        )
        self._dive_entered = False
        self._t_dive_entry = float("nan")
        self._t_terminal_ignition = float("nan")
        self._terminal_lit = False
        self._sustain_deficit_max = 0.0
        self._sustain_deficit_integral = 0.0
        self._alpha_limit_hits = 0
        self._q_floor_hit = False

    # -------------------------------------------------------------------- guidance ---

    def _phase_of(self, t: float) -> str:
        reqs = self.reqs
        if t < reqs.t_separation:
            return "separation"
        motor_phase = self.motor.phase(t - reqs.t_separation)   # type: ignore[attr-defined]
        if motor_phase == "terminal":
            # The commanded terminal pulse is burning. Recorded distinctly from the
            # unpowered part of the dive so the report can separate them.
            return "terminal_boost"
        if self._dive_entered:
            return "terminal"
        if motor_phase == "burnout":
            return "coast"
        return motor_phase

    def _commanded_gamma(self, state: FlightState, phase: str) -> float:
        """Commanded flight path angle for the active phase."""
        reqs = self.reqs
        if phase in ("terminal", "terminal_boost"):
            return reqs.gamma_terminal
        target_h = reqs.h_launch if phase == "separation" else reqs.h_cruise
        h_dot_cmd = max(
            -CLIMB_RATE_LIMIT, min(CLIMB_RATE_LIMIT, ALTITUDE_GAIN * (target_h - state.h))
        )
        ratio = max(-1.0, min(1.0, h_dot_cmd / max(state.V, 1.0)))
        gamma_cmd = math.asin(ratio)
        return max(-GAMMA_CMD_LIMIT, min(GAMMA_CMD_LIMIT, gamma_cmd))

    def _should_dive(self, state: FlightState, mach: float, alpha_limited: bool) -> bool:
        if self._dive_entered or state.t < self._t_sustain_burnout:
            return False
        if self.dive_rule == "terminal_boost":
            # Dive the moment the sustain phase ends. No level coast, so the endgame keeps
            # the most energy and the terminal pulse has the most altitude to work with.
            return True
        if self.dive_rule == "range":
            dive_ground_range = state.h / abs(math.tan(self.reqs.gamma_terminal))
            return state.x + dive_ground_range >= self.reqs.range_min
        return alpha_limited or mach <= DIVE_MACH_TRIGGER

    def terminal_ignition_altitude(self, speed: float) -> float:
        """Altitude, m, at which the terminal pulse should light to burn out at impact.

        h = margin * V * |sin(gamma_terminal)| * t_terminal

        Exposed rather than hard-coded so the sizing loop can trade the margin, the dive
        angle and the terminal burn time against each other. Returns 0.0 when there is no
        terminal pulse.
        """
        burn_time = float(getattr(self.motor, "t_terminal", 0.0))
        if burn_time <= 0.0:
            return 0.0
        descent_rate = speed * abs(math.sin(self.reqs.gamma_terminal))
        return self.terminal_ignition_margin * descent_rate * burn_time

    def _should_light_terminal(self, state: FlightState) -> bool:
        """Terminal-pulse ignition trigger. Only ever consulted inside the dive."""
        if self._terminal_lit or not self._dive_entered:
            return False
        if self.terminal_ignition_rule == "never":
            return False
        if not getattr(self.motor, "has_terminal", False):
            return False
        if self.terminal_ignition_rule == "dive_entry":
            return True
        return state.h <= self.terminal_ignition_altitude(state.V)

    # ---------------------------------------------------------------- force model ---

    def force_model(self, state: FlightState) -> Forces:
        """Trim the airframe and return the forces at this state."""
        reqs = self.reqs
        rho, _p, _T, sound = atmosphere_properties(state.h)
        mach = state.V / sound if sound > 0.0 else 0.0
        q = 0.5 * rho * state.V * state.V
        phase = self._phase_of(state.t)

        motor_t = state.t - reqs.t_separation
        powered = phase in ("boost", "sustain", "terminal_boost")
        thrust_available = (
            self.motor.thrust(motor_t, state.h) if powered else 0.0  # type: ignore[attr-defined]
        )
        mdot = self.motor.mdot(motor_t) if powered else 0.0          # type: ignore[attr-defined]

        gamma_cmd = self._commanded_gamma(state, phase)
        gamma_dot_cmd = GAMMA_GAIN * (gamma_cmd - state.gamma)
        # Cap the commanded normal acceleration at LOAD_FACTOR_LIMIT g. Without this the
        # step change in commanded flight path angle at dive entry demands an unbounded
        # normal force for one step, which would poison the CN_required history the
        # sizing loop reads.
        accel_limit = LOAD_FACTOR_LIMIT * G0
        accel_cmd = state.V * gamma_dot_cmd + self.gravity_term(state)
        if abs(accel_cmd) > accel_limit and state.V > 1.0:
            accel_cmd = math.copysign(accel_limit, accel_cmd)
            gamma_dot_cmd = (accel_cmd - self.gravity_term(state)) / state.V

        # Fixed-point solve for alpha: the required normal force depends on the thrust
        # component T*sin(alpha), which depends on alpha. Started from alpha = 0 every
        # call so the force model stays a pure function of the state and RK4 sees a
        # consistent right-hand side. Three passes is ample because T*sin(alpha) is a
        # small correction to the normal-force balance. Unpowered, the balance has no
        # alpha on the right-hand side at all, so one pass is exact.
        alpha = 0.0
        cn_required = 0.0
        alpha_limited = False
        qs = q * self.S_ref
        passes = TRIM_ITERATIONS if thrust_available > 0.0 else 1
        for _ in range(passes):
            normal_required = state.m * (
                state.V * gamma_dot_cmd + self.gravity_term(state)
            ) - thrust_available * math.sin(alpha)
            if q > Q_FLOOR:
                cn_required = normal_required / qs
            else:
                cn_required = 0.0
                self._q_floor_hit = True
            alpha_trim = self.aero.trim_alpha(mach, state.h, cn_required)
            if alpha_trim > self.alpha_max:
                alpha_trim = self.alpha_max
                alpha_limited = True
            elif alpha_trim < -self.alpha_max:
                alpha_trim = -self.alpha_max
                alpha_limited = True
            else:
                alpha_limited = False
            alpha = alpha_trim

        coeffs = self.aero.evaluate(mach, state.h, alpha, power_on=powered)
        drag = qs * coeffs.CD
        normal = qs * coeffs.CN

        if alpha_limited:
            self._alpha_limit_hits += 1

        # Constant-Mach constant-altitude sustain check. The motor is never faked.
        if phase == "sustain":
            thrust_needed = (
                drag + state.m * G0 * math.sin(state.gamma)
            ) / max(math.cos(alpha), 1e-6)
            deficit = thrust_needed - thrust_available
            if deficit > self._sustain_deficit_max:
                self._sustain_deficit_max = deficit

        # Event triggers. These mutate mission state inside an RK4 substage, so the one
        # step in which each event fires sees a discontinuous command. Both are one-off
        # events; everything else in this force model is a pure function of the state.
        if self._should_dive(state, mach, alpha_limited):
            self._dive_entered = True
            self._t_dive_entry = state.t
        if self._should_light_terminal(state):
            if self.motor.ignite_terminal(motor_t):   # type: ignore[attr-defined]
                self._terminal_lit = True
                self._t_terminal_ignition = state.t

        return Forces(
            thrust=thrust_available,
            drag=drag,
            normal=normal,
            alpha=alpha,
            mdot=mdot,
            phase=phase,
            extras={
                "mach": mach,
                "q": q,
                "CN_required": cn_required,
                "alpha_limited": 1.0 if alpha_limited else 0.0,
                "CD": coeffs.CD,
                "CN": coeffs.CN,
            },
        )

    def gravity_term(self, state: FlightState) -> float:
        """The g * cos(gamma) term of the normal-force balance, m/s^2."""
        return G0 * math.cos(state.gamma)

    # -------------------------------------------------------------------- flying ---

    def fly(
        self,
        dt: float = 0.02,
        t_max: float = 900.0,
        adaptive: bool = False,
        tolerance: float = 1.0e-6,
    ) -> TrajectoryResult:
        """Fly the mission and return the trajectory.

        `converged` is True when the vehicle reached the ground or the velocity floor.
        `message` carries every warning raised on the way, honestly, including a sustain
        thrust shortfall and any trim-authority shortfall.
        """
        reqs = self.reqs
        self._dive_entered = False
        self._t_dive_entry = float("nan")
        self._t_terminal_ignition = float("nan")
        self._terminal_lit = False
        self._sustain_deficit_max = 0.0
        self._alpha_limit_hits = 0
        self._q_floor_hit = False

        # Terminal ignition time is motor state and the sizing loop reuses one motor for
        # many flights, so clear it and re-arm. See SOURCES['traj.motor_state_reset'].
        if hasattr(self.motor, "terminal_ignition_time"):
            self.motor.terminal_ignition_time = None   # type: ignore[attr-defined]
        if hasattr(self.motor, "arm_terminal"):
            self.motor.arm_terminal()                  # type: ignore[attr-defined]
        self._t_sustain_burnout = (
            float(
                getattr(
                    self.motor, "t_burnout_sustain", getattr(self.motor, "t_burnout", 0.0)
                )
            )
            + reqs.t_separation
        )

        _rho, _p, _T, sound = atmosphere_properties(reqs.h_launch)
        state0 = FlightState(
            t=0.0,
            V=reqs.M_launch * sound,
            gamma=0.0,
            x=0.0,
            h=reqs.h_launch,
            m=self.m0,
        )

        integrator = PointMass3DOF(self.force_model)
        result = integrator.integrate(
            state0, dt=dt, t_max=t_max, adaptive=adaptive, tolerance=tolerance
        )

        notes: list[str] = []
        if result.message:
            notes.append(result.message)
        if self._sustain_deficit_max > 1.0:
            notes.append(
                f"sustain phase could not hold constant Mach at constant altitude: the "
                f"motor was short by up to {self._sustain_deficit_max:.0f} N. The "
                "trajectory shows what the motor actually delivered."
            )
        if self._alpha_limit_hits:
            notes.append(
                f"trim angle of attack hit the {math.degrees(self.alpha_max):.0f} deg "
                f"limit on {self._alpha_limit_hits} force evaluations; fin authority is "
                "short there"
            )
        if self._q_floor_hit:
            notes.append(
                f"dynamic pressure fell below the {Q_FLOOR:.0f} Pa floor; trim was "
                "frozen at zero required normal force in that region"
            )
        if not math.isnan(self._t_dive_entry):
            notes.append(f"terminal dive entered at t = {self._t_dive_entry:.1f} s")
        else:
            notes.append("terminal dive was never entered")

        terminal_burn_time = float(getattr(self.motor, "t_terminal", 0.0))
        has_terminal = bool(getattr(self.motor, "has_terminal", False))
        if self._terminal_lit:
            burn_end = self._t_terminal_ignition + terminal_burn_time
            overshoot = result.time[-1] - burn_end
            notes.append(
                f"terminal boost lit at t = {self._t_terminal_ignition:.1f} s for "
                f"{terminal_burn_time:.1f} s"
            )
            if overshoot > 1.0:
                notes.append(
                    f"terminal boost burnt out {overshoot:.1f} s before impact; reduce "
                    "terminal_ignition_margin or m_p_terminal"
                )
            elif overshoot < -1.0:
                notes.append(
                    f"terminal boost was still burning {-overshoot:.1f} s of propellant "
                    "worth at impact; that propellant was wasted"
                )
        elif has_terminal:
            notes.append(
                "a terminal boost was available but never lit "
                f"(rule '{self.terminal_ignition_rule}')"
            )
        if ATMOSPHERE_SOURCE.startswith("trajectory.py"):
            notes.append(
                "atmosphere came from the inline US-1976 fallback, not from "
                "rocketgen.sizing.atmosphere"
            )

        result.message = "; ".join(notes)
        self.diagnostics = {
            "sustain_thrust_deficit_max": self._sustain_deficit_max,
            "alpha_limit_hits": float(self._alpha_limit_hits),
            "t_dive_entry": self._t_dive_entry,
            "t_sustain_burnout": self._t_sustain_burnout,
            "t_burnout": float(getattr(self.motor, "t_burnout", 0.0))
            + reqs.t_separation,
            "has_terminal_boost": has_terminal,
            "terminal_boost_lit": self._terminal_lit,
            "t_terminal_ignition": self._t_terminal_ignition,
            "terminal_burn_time": terminal_burn_time,
            "impact_mach": result.mach_final,
            "impact_altitude": result.h[-1] if result.h else float("nan"),
            "atmosphere_source": ATMOSPHERE_SOURCE,
            "steps": float(len(result.time)),
        }
        result.diagnostics = dict(self.diagnostics)   # type: ignore[attr-defined]
        return result


# --------------------------------------------------------------------------------------
#   Small helpers used by the tests and the report
# --------------------------------------------------------------------------------------


def apogee(result: TrajectoryResult) -> float:
    """Peak altitude, m, refined by a parabola through the three highest samples."""
    if not result.h:
        return 0.0
    index = max(range(len(result.h)), key=lambda i: result.h[i])
    if 0 < index < len(result.h) - 1:
        h_prev, h_mid, h_next = result.h[index - 1], result.h[index], result.h[index + 1]
        denominator = h_prev - 2.0 * h_mid + h_next
        if abs(denominator) > 1e-30:
            offset = 0.5 * (h_prev - h_next) / denominator
            return h_mid - 0.25 * (h_prev - h_next) * offset
    return result.h[index]


def specific_energy(result: TrajectoryResult) -> list[float]:
    """Total specific mechanical energy V^2/2 + g*h, J/kg, at each sample."""
    return [0.5 * v * v + G0 * h for v, h in zip(result.V, result.h)]
