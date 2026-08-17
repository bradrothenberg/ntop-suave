"""Shared contracts for IV-1, the two-stage interceptor-class reference example.

Read `SPEC_IV1.md` first. This module is to IV-1 what `config.py` is to SV-1, and it deliberately
does NOT modify `config.py`: SV-1 is the regression baseline and must keep working untouched.

Units are SI throughout, as in `config.py`: metre, kilogram, second, radian, newton, pascal.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .config import (
    MATERIALS,
    AeroCoefficients,
    Material,
    NtopMeasurements,
    register_sources,
)

SOURCES: dict[str, str] = {
    "iv1_requirements": (
        "INVENTED for the demonstration. The IV-1 top-level requirements in SPEC_IV1.md "
        "section 2 correspond to no real programme. They were chosen to be internally "
        "consistent and to exercise a two-stage, strake-stabilised configuration."
    ),
    "iv1_qmax_revision": (
        "MODELLING CHOICE, and a declared gap. A10 was revised from 250 kPa to 350 kPa because a "
        "sweep over propellant, thrust, pitchover angle and pitchover time found a hard floor of "
        "about 278 kPa on peak dynamic pressure, rising to 309 kPa for any design that also meets "
        "A2, A3 and A4. The structural model does not size the airframe for that load: it is wall "
        "thickness times density plus a motor-case hoop-stress check and an interstage buckling "
        "check. The airframe mass is therefore optimistic by an amount not quantified here."
    ),
    "iv1_qmax": (
        "MODELLING CHOICE: 250 kPa dynamic-pressure limit, against 200 kPa for SV-1. A vertical "
        "sea-level launch accelerates through dense air, so the same airframe sees a higher peak "
        "than SV-1 does launching at 10 km. Consistent with the manoeuvre-envelope pressures "
        "quoted for supersonic tactical vehicles in Fleeman, Tactical Missile Design, 2nd ed., "
        "Chapter 3."
    ),
    "iv1_separation": (
        "MODELLING CHOICE: stage separation is instantaneous, imparts no impulse and no attitude "
        "disturbance, and jettisons the stage-1 inert mass together with the interstage. Real "
        "separation carries a tip-off disturbance and a brief drag transient; neither is modelled."
    ),
    "iv1_acs": (
        "Attitude-control motor performance. Isp 235 s is the low end of the AP/HTPB range in "
        "Sutton and Biblarz, Rocket Propulsion Elements 9th ed., Table 12-1, taken low because a "
        "side thruster runs a short, low-expansion nozzle at altitude and pays a large divergence "
        "and heat loss. GUESS: the 0.55 inert-to-propellant ratio for a pulsed divert pack is not "
        "from a source; it is high relative to a single large motor because a thruster bank pays "
        "for multiple valves, nozzles and a manifold. No plume interaction, no minimum impulse "
        "bit, no control loop and no roll coupling are modelled."
    ),
    "iv1_lateral_accel": (
        "Available lateral acceleration is the static value q * S_ref * CN_max / m at the "
        "intercept condition, with CN_max taken at the alpha limit. APPROXIMATION: it is a "
        "capability, not a manoeuvre. It says nothing about autopilot response time, actuator "
        "rate or the transient during a turn."
    ),
}
register_sources(SOURCES)


# --------------------------------------------------------------------------------------
#   Stage definition
# --------------------------------------------------------------------------------------


@dataclass
class StageSpec:
    """One stage of the stack.

    `index` is 1 for the first stage to burn, counting up. Stage 1 is jettisoned; the last stage
    carries the payload to intercept.
    """

    index: int
    D: float                      # body diameter, m
    L: float                      # stage length, m, including its share of the interstage
    m_propellant: float           # kg
    F_thrust: float               # N, sea-level equivalent for stage 1, vacuum for upper stages
    t_wall: float = 0.0030        # m
    p_c: float = 8.0e6            # Pa, chamber pressure
    eps_nozzle: float = 10.0      # nozzle area ratio
    jettisoned: bool = True       # False for the stage that reaches intercept

    # Tail fins, cruciform, on every stage
    n_fin: int = 4
    b_fin: float = 0.16           # exposed semi-span per panel, m
    c_r_fin: float = 0.34         # root chord, m
    taper_fin: float = 0.50
    sweep_fin: float = math.radians(42.0)
    t_fin: float = 0.010          # m

    @property
    def S_ref(self) -> float:
        """Stage reference area, m^2."""
        return 0.25 * math.pi * self.D**2

    @property
    def c_t_fin(self) -> float:
        return self.taper_fin * self.c_r_fin

    @property
    def S_fin_exposed(self) -> float:
        """Exposed planform area of ONE fin panel, m^2."""
        return 0.5 * (self.c_r_fin + self.c_t_fin) * self.b_fin

    @property
    def fin_span_total(self) -> float:
        """Tip-to-tip span across the body, m."""
        return self.D + 2.0 * self.b_fin


# --------------------------------------------------------------------------------------
#   Strakes
# --------------------------------------------------------------------------------------


@dataclass
class StrakeSpec:
    """Four strakes running along the payload stage.

    A strake is a long, thin, very low aspect ratio surface. Its job is to generate normal force at
    high angle of attack without stalling, which is what lets this vehicle class hold alpha at
    altitude where dynamic pressure is low. Because the aspect ratio is small the flow is
    vortex-dominated, so a purely linear lifting-surface method underpredicts the normal force.
    """

    n: int = 4
    height: float = 0.030          # b_strake, radial height above the body surface, m
    length: float = 1.40           # chordwise length, m
    thickness: float = 0.008       # m
    x_le: float = 0.60             # leading-edge station from the stage-2 nose tip, m
    sweep_le: float = math.radians(0.0)   # strakes are usually unswept

    @property
    def area_one_side(self) -> float:
        """Planform area of one strake panel seen from the side, m^2."""
        return self.height * self.length

    @property
    def aspect_ratio(self) -> float:
        """Exposed aspect ratio of one panel, using the exposed height as the span.

        This comes out well below 1 for a real strake, which is exactly why the vortex-lift term
        matters.
        """
        return self.height**2 / self.area_one_side if self.area_one_side > 0 else 0.0

    @property
    def wetted_area(self) -> float:
        """All panels, both sides, m^2."""
        return 2.0 * self.n * self.area_one_side


# --------------------------------------------------------------------------------------
#   Attitude-control motor
# --------------------------------------------------------------------------------------


@dataclass
class AcsSpec:
    """A divert or attitude-control motor on the payload stage.

    It exists because A11 cannot be met aerodynamically at a lofted intercept: at the
    post-separation mass, 15 g of aerodynamic lateral acceleration is available only below about
    14 km at Mach 4, while reaching 100 miles of slant range needs an intercept above 20 km. See
    the requirements audit in SPEC_IV1.md section 2.

    Modelled as a bank of lateral thrusters firing in short pulses. Only the capability and the
    mass are represented: no plume interaction, no control loop, no roll coupling and no thruster
    minimum impulse bit.
    """

    thrust: float = 26.0e3          # N, total lateral thrust available
    burn_time: float = 3.0          # s of cumulative firing across the engagement
    isp: float = 235.0              # s, low-expansion side thruster at altitude
    inert_fraction: float = 0.55    # inert mass / propellant mass for a pulsed divert pack

    @property
    def propellant_mass(self) -> float:
        """kg. Total impulse divided by Isp*g0."""
        return self.thrust * self.burn_time / (self.isp * 9.80665)

    @property
    def inert_mass(self) -> float:
        return self.inert_fraction * self.propellant_mass

    @property
    def total_mass(self) -> float:
        return self.propellant_mass + self.inert_mass

    @property
    def total_impulse(self) -> float:
        """N.s. Requirement A13."""
        return self.thrust * self.burn_time


# --------------------------------------------------------------------------------------
#   Design vector
# --------------------------------------------------------------------------------------


@dataclass
class StackDesignVector:
    """The IV-1 design vector: a two-stage stack with strakes.

    Stage order in `stages` is burn order. The last entry is the payload stage.
    """

    stages: list[StageSpec] = field(default_factory=list)
    strakes: StrakeSpec = field(default_factory=StrakeSpec)
    acs: AcsSpec = field(default_factory=AcsSpec)

    # Payload stage nose
    f_nose: float = 3.6                # nose fineness on the payload stage
    nose_shape: str = "tangent_ogive"

    # Interstage, jettisoned with stage 1
    L_interstage: float = 0.28         # m
    t_interstage: float = 0.0025       # m

    # Payload stage internal layout, from its own nose tip
    L_seeker: float = 0.32             # m, forward bay
    L_payload_bay: float = 0.40        # m

    # Ascent programme
    gamma_pitch: float = math.radians(58.0)   # commanded flight-path angle after pitchover
    t_pitch: float = 3.0                       # s, when pitchover starts
    pitch_rate_max: float = math.radians(8.0)  # rad/s cap on the commanded turn

    # --- convenience ---

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    @property
    def booster(self) -> StageSpec:
        return self.stages[0]

    @property
    def payload_stage(self) -> StageSpec:
        return self.stages[-1]

    @property
    def L_total(self) -> float:
        """Stacked length, m, nose tip to booster nozzle exit."""
        return sum(s.L for s in self.stages) + self.L_interstage

    @property
    def D_max(self) -> float:
        return max(s.D for s in self.stages)

    @property
    def L_nose(self) -> float:
        return self.f_nose * self.payload_stage.D

    @property
    def m_propellant_total(self) -> float:
        return sum(s.m_propellant for s in self.stages)

    def stage_at(self, index: int) -> StageSpec:
        for s in self.stages:
            if s.index == index:
                return s
        raise KeyError(f"no stage with index {index}")

    def bounds(self) -> dict[str, tuple[float, float]]:
        """SPEC_IV1.md section 4. Keys are dotted paths the sizer knows how to set."""
        return {
            "stages.0.D": (0.28, 0.42),
            "stages.1.D": (0.20, 0.36),
            "stages.0.L": (1.2, 2.8),
            "stages.1.L": (1.8, 3.4),
            "stages.0.m_propellant": (150.0, 600.0),
            "stages.1.m_propellant": (60.0, 300.0),
            "stages.0.F_thrust": (80.0e3, 300.0e3),
            "stages.1.F_thrust": (20.0e3, 90.0e3),
            "stages.0.b_fin": (0.10, 0.28),
            "stages.1.b_fin": (0.08, 0.24),
            "stages.0.c_r_fin": (0.22, 0.55),
            "stages.1.c_r_fin": (0.20, 0.50),
            "strakes.height": (0.015, 0.060),
            "strakes.length": (0.60, 2.20),
            "strakes.thickness": (0.004, 0.014),
            "f_nose": (2.5, 5.0),
            "gamma_pitch": (math.radians(35.0), math.radians(75.0)),
            "t_pitch": (1.5, 8.0),
            # Attitude-control motor. The lower thrust bound is zero so the sizer is free to find
            # that the vehicle does not need one. The audit in SPEC_IV1.md says it does.
            "acs.thrust": (0.0, 60.0e3),
            "acs.burn_time": (0.5, 8.0),
        }

    def get_path(self, path: str) -> float:
        obj: Any = self
        parts = path.split(".")
        for p in parts[:-1]:
            obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
        return getattr(obj, parts[-1])

    def with_path(self, path: str, value: float) -> "StackDesignVector":
        """Return a copy with one dotted path set. Used by the sizer and the DOE."""
        import copy

        new = copy.deepcopy(self)
        obj: Any = new
        parts = path.split(".")
        for p in parts[:-1]:
            obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
        setattr(obj, parts[-1], value)
        return new

    def replace(self, **kw: Any) -> "StackDesignVector":
        return replace(self, **kw)

    def geometry_is_valid(self) -> tuple[bool, list[str]]:
        """Cheap gate before spending an nTop call."""
        errs: list[str] = []
        if self.n_stages < 2:
            errs.append("IV-1 needs at least two stages")
            return False, errs
        if self.payload_stage.D > self.booster.D + 1e-12:
            errs.append(
                f"payload stage diameter {self.payload_stage.D:.3f} exceeds booster "
                f"{self.booster.D:.3f}"
            )
        if self.L_nose >= self.payload_stage.L:
            errs.append("the nose is longer than the payload stage")
        bays = self.L_seeker + self.L_payload_bay
        if self.L_nose + bays >= self.payload_stage.L:
            errs.append("no room for the stage-2 motor behind the payload bays")
        st = self.strakes
        if st.x_le + st.length > self.payload_stage.L:
            errs.append("the strakes run off the back of the payload stage")
        if st.height > 0.5 * self.payload_stage.D:
            errs.append("strake height is more than half the body diameter")
        for s in self.stages:
            if s.t_wall <= 0.0 or s.t_wall > 0.05 * s.D:
                errs.append(f"stage {s.index} wall thickness {s.t_wall:.4f} m out of range")
        return (not errs), errs

    def as_dict(self) -> dict[str, Any]:
        d = {
            "stages": [asdict(s) for s in self.stages],
            "strakes": asdict(self.strakes),
            "acs": asdict(self.acs),
            "f_nose": self.f_nose,
            "nose_shape": self.nose_shape,
            "L_interstage": self.L_interstage,
            "t_interstage": self.t_interstage,
            "L_seeker": self.L_seeker,
            "L_payload_bay": self.L_payload_bay,
            "gamma_pitch": self.gamma_pitch,
            "t_pitch": self.t_pitch,
            "pitch_rate_max": self.pitch_rate_max,
        }
        d.update(
            L_total=self.L_total,
            D_max=self.D_max,
            L_nose=self.L_nose,
            m_propellant_total=self.m_propellant_total,
            n_stages=self.n_stages,
        )
        return d


def default_iv1() -> StackDesignVector:
    """A starting stack. Not sized; the sizer moves it."""
    return StackDesignVector(
        stages=[
            StageSpec(
                index=1, D=0.40, L=2.10, m_propellant=380.0, F_thrust=170.0e3,
                jettisoned=True, b_fin=0.20, c_r_fin=0.40, t_wall=0.0032,
            ),
            StageSpec(
                index=2, D=0.28, L=2.70, m_propellant=150.0, F_thrust=45.0e3,
                jettisoned=False, b_fin=0.14, c_r_fin=0.30, t_wall=0.0026,
                eps_nozzle=18.0,
            ),
        ],
        strakes=StrakeSpec(height=0.030, length=1.40, thickness=0.008, x_le=0.95),
    )


# --------------------------------------------------------------------------------------
#   Requirements
# --------------------------------------------------------------------------------------


@dataclass
class InterceptRequirements:
    """SPEC_IV1.md section 2. Invented values."""

    slant_range_min: float = 160_934.0    # m, 100 statute miles
    h_intercept_min: float = 15_000.0     # m
    mach_intercept_min: float = 3.0
    m_payload: float = 75.0               # kg
    h_launch: float = 0.0                 # m, sea level
    gamma_launch: float = math.radians(90.0)
    D_max: float = 0.42                   # m
    L_max: float = 5.40                   # m
    m0_max: float = 1400.0                # kg
    static_margin_min: float = 1.0        # calibres
    # Structural dynamic-pressure limit.
    #
    # First written as 250 kPa. A sweep over stage propellant, booster thrust, pitchover angle and
    # pitchover time showed NOTHING meets it: the minimum peak dynamic pressure anywhere in the
    # design space is about 278 kPa, and every configuration that also satisfies A2, A3 and A4
    # needs at least 309 kPa. The peak occurs near 6 km on the ascent, which is where a
    # vertical-launch vehicle that must eventually reach Mach 5 is unavoidably fast in dense air.
    # Holding 250 kPa would cap the vehicle near Mach 2.75 at 6 km, which cannot reach 100 miles.
    #
    # Revised to 350 kPa: about 13 percent above the 309 kPa floor that A2, A3 and A4 impose.
    #
    # DECLARED LIMITATION: the structural model in this toolkit is wall thickness times density,
    # plus a hoop-stress check on the motor case and a buckling check on the interstage. It does
    # NOT size the airframe for a 350 kPa aerodynamic load. The limit is therefore a stated
    # requirement that the mass model does not verify, and the airframe mass is optimistic by an
    # amount this toolkit cannot quantify.
    q_max: float = 350_000.0              # Pa. See the note above.
    lateral_g_min: float = 15.0           # g available at intercept
    h_stage1_burnout_max: float = 20_000.0  # m
    alpha_max: float = math.radians(20.0)  # limit used for the CN_max capability figure

    # Unpowered coast BETWEEN stage-1 burnout and separation, so
    #   t_separation = t_burnout(1) + t_coast_separation = t_ignition(2)
    # The stack therefore carries the spent booster through the coast, which is the pessimistic
    # and safe reading. An earlier comment here said "coast after separation", which contradicted
    # that formula; the formula is the contract and the comment was wrong.
    t_coast_separation: float = 0.6       # s

    @property
    def slant_range_min_miles(self) -> float:
        return self.slant_range_min / 1609.344


# --------------------------------------------------------------------------------------
#   Results
# --------------------------------------------------------------------------------------


@dataclass
class StageEvent:
    """A discrete event in the ascent, recorded so the report can annotate the trajectory."""

    name: str                 # "pitchover", "stage_1_burnout", "separation", ...
    time: float               # s
    altitude: float           # m
    mach: float
    mass_before: float        # kg
    mass_after: float         # kg
    note: str = ""

    @property
    def mass_jettisoned(self) -> float:
        return self.mass_before - self.mass_after


@dataclass
class InterceptResult:
    """Conditions at the end of the run, whatever ended it."""

    reached_slant_range: bool = False
    slant_range: float = 0.0        # m
    ground_range: float = 0.0       # m
    altitude: float = 0.0           # m
    mach: float = 0.0
    velocity: float = 0.0           # m/s
    time: float = 0.0               # s
    mass: float = 0.0               # kg
    q: float = 0.0                  # Pa
    lateral_g_available: float = 0.0
    termination: str = ""           # "slant_range" | "ground_impact" | "t_max" | "stalled"

    @property
    def slant_range_miles(self) -> float:
        return self.slant_range / 1609.344


def slant_range(x: float, h: float) -> float:
    """Straight-line distance from the launch point, m. SPEC_IV1.md section 2, note on A2."""
    return math.hypot(x, h)


def lateral_g(
    q: float, S_ref: float, CN_max: float, mass: float, g0: float = 9.80665
) -> float:
    """Aerodynamic lateral acceleration in g. See SOURCES["iv1_lateral_accel"].

    This is the AERODYNAMIC contribution only. It goes to zero with dynamic pressure, which is why
    it cannot satisfy A11 at a lofted intercept. Use `lateral_g_total` for the requirement.
    """
    if mass <= 0.0:
        return 0.0
    return q * S_ref * CN_max / (mass * g0)


def lateral_g_acs(acs_thrust: float, mass: float, g0: float = 9.80665) -> float:
    """Lateral acceleration in g from the attitude-control motor.

    Independent of altitude and of dynamic pressure, which is the entire reason it exists. See the
    requirements audit in SPEC_IV1.md section 2.
    """
    if mass <= 0.0:
        return 0.0
    return acs_thrust / (mass * g0)


def lateral_g_total(
    q: float,
    S_ref: float,
    CN_max: float,
    mass: float,
    acs_thrust: float = 0.0,
    g0: float = 9.80665,
) -> float:
    """The A11 figure: the greater of the aerodynamic and attitude-control capabilities.

    Taken as the greater rather than the sum. The two are not simply additive: an aerodynamic turn
    needs angle of attack, a divert thrust does not, and commanding both at once is a control
    problem this model does not represent. Taking the greater is the conservative reading.
    """
    return max(
        lateral_g(q, S_ref, CN_max, mass, g0),
        lateral_g_acs(acs_thrust, mass, g0),
    )
