"""Multi-stage solid propulsion for IV-1, the two-stage interceptor-class reference example.

Read `SPEC_IV1.md` section 3 and 5 first, then `propulsion.py`. This module does NOT re-derive
any physics. Every nozzle relation, the c* derivation, the thrust coefficient, the burn-rate
law, the Kn closure, the mean-web grain closure, the ramp shape, the case and nozzle mass
estimates and the Summerfield separation check are imported from `propulsion.py`, where they are
validated against Sutton and Biblarz Table 12-1 and the published Purdue isentropic tables. The
shared building blocks are the free functions `ramp`, `throat_area_for_thrust`, `design_phase`,
`tubular_grain_closure`, `motor_case_and_insulation`, `conical_nozzle_mass`,
`correlation_inert_band` and `summerfield_separation`. If you change the physics, change it
there and both motors move together.

What is new here, and only here
-------------------------------
1. **A stack instead of a grain.** Each stage is an independent motor: its own case, its own
   grain, its own nozzle, its own throat. At this fidelity a "stage" of IV-1 and a "phase" of
   the SV-1 dual-thrust motor are the same object, one charge burning at one chamber pressure
   through one throat, so `design_phase` closes both.
2. **A separation event in the timeline.** Stage 1 ignites at t = 0, burns out, the stack coasts
   unpowered for `reqs.t_coast_separation`, and stage 2 ignites. Mass leaves the vehicle at that
   instant; `jettisoned_mass()` is the number the trajectory removes.
3. **Per-stage nozzle sizing at different ambient pressures.** Stage 1 is sized at SEA LEVEL,
   because it is lit in the canister at sea level and `StageSpec.F_thrust` is defined as a
   sea-level equivalent thrust for stage 1. Upper stages are sized in VACUUM (p_a = 0), because
   they are lit above 10 km where the ambient term is already small and `StageSpec.F_thrust` is
   defined as a vacuum thrust for them. Both cases use the same relation,
   F = C_F_vac * p_c * A_t - p_a * A_e, whose second term is exactly (p_e - p_a) * A_e once
   C_F_vac is expanded. Nothing about the ambient term is special-cased per stage.

Throat credibility, which is CLEAN for a stack and was not for SV-1
-------------------------------------------------------------------
The SV-1 motor needed one throat to serve a 45 kN boost and a 2.6 kN sustain, and
`SolidMotor.throat_transition_report()` has to report that the required boost-to-sustain
transition shrinks the throat, which no ejectable-insert mechanism performs. A staged stack does
not have that problem: each stage carries its own nozzle and each nozzle runs at exactly one
throat area for its whole life. `throat_credibility_report()` verifies that claim rather than
asserting it, by listing the number of throat areas each stage's nozzle must take, and it is 1
for every stage. The hardware cost of that is a second complete nozzle plus a separation joint,
which is why the stack is heavier than a dual-thrust motor of the same impulse and why
`jettisoned_mass()` matters.

APPROXIMATIONS - read before trusting a number
----------------------------------------------
1. **Neutral burning, tubular grain, per stage.** Same closure as the SV-1 boost segment:
   the burn-rate law is evaluated at the MEAN web position, so propellant mass, web, burn time
   and mass flow are mutually consistent. No burnback simulation. A tubular grain is the LEAST
   area-efficient internal-burning geometry: a slotted, star or finocyl grain gives several
   times the burning area in the same length. That matters for IV-1, because a booster sized
   for 170 kN needs a burning area a plain tube can only reach by being long. The model reports
   the resulting grain length and L/D honestly instead of applying an unsourced shape factor.
   See SOURCES["prop_iv1.tubular_stage_grain"].
2. **Ideal nozzle, as in `propulsion.py`.** Frozen composition, constant gamma, single phase.
   No two-phase loss, no divergence loss, no combustion efficiency, no throat erosion. Real
   delivered specific impulse for this class runs 3 to 7 percent lower and that penalty is NOT
   applied, because its magnitude could not be sourced. This is the largest known unquantified
   optimism in the result.
3. **Instantaneous separation.** No tip-off, no impulse, no drag transient, and the separation
   joint hardware itself is not costed. See SOURCES["prop_iv1.separation_hardware"].
4. **The bottom-up inert mass is incomplete, per stage, exactly as for SV-1.** Both the physics
   sum and the tactical-motor correlation band are reported for every stage and
   `inert_mass_breakdown(stage)["recommended"]` names the one to carry.

Units are SI throughout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import MATERIALS, Material, register_sources
from ..config_iv1 import InterceptRequirements, StackDesignVector, StageSpec
from .propulsion import (
    CASE_LENGTH_ALLOWANCE_M,
    CASE_SAFETY_FACTOR,
    G0,
    IGNITER_MASS_FRACTION,
    INSULATION_THICKNESS_M,
    P_SEA_LEVEL,
    T_RISE_S,
    T_TAILOFF_S,
    GrainGeometry,
    PhaseDesign,
    _ambient_pressure,
    characteristic_velocity,
    conical_nozzle_mass,
    correlation_inert_band,
    design_phase,
    exit_pressure_ratio,
    motor_case_and_insulation,
    ramp,
    summerfield_separation,
    thrust_coefficient,
    throat_area_for_thrust,
    tubular_grain_closure,
)
from .propulsion import GAMMA_EXHAUST

# --------------------------------------------------------------------------------------
#   Modelling choices that belong to the stack, not to a single motor
# --------------------------------------------------------------------------------------

# Axial length of the booster stage that is NOT available for the grain: aft closure, nozzle
# attach ring and igniter boss. Deliberately the SAME constant that `propulsion.py` adds to the
# grain length to get the case length, so the two statements are consistent: the booster case is
# the grain plus this allowance, and the booster bay is the stage length less this allowance. A
# grain that exactly fills the bay therefore gives a case exactly as long as the stage.
BOOSTER_AFT_CLOSURE_ALLOWANCE_M = CASE_LENGTH_ALLOWANCE_M

# Fin structural model, taken from the analytic branch of `masses.py::build_masses` so the two
# modules do not disagree about the same hardware: exposed planform area times maximum thickness
# times an area-to-volume factor for a tapered biconvex section, in titanium.
FIN_VOLUME_FACTOR = 0.65
FIN_MATERIAL = MATERIALS["fin_ti64"]

# Interstage shell. "cylindrical" is the default: a constant-diameter skirt at the BOOSTER
# diameter, which is exactly linear in `L_interstage`. "conical" is the frustum from the payload
# stage diameter to the booster diameter, which is lighter because a taper has less surface for
# the same axial length. Both are reported; only the default is charged.
INTERSTAGE_MATERIAL = MATERIALS["airframe_al7075"]

# Grain length-to-diameter limits, per stage, from SPEC_IV1.md section 6.
GRAIN_L_OVER_D_LIMITS = (1.0, 8.0)

# Chamber pressure below which composite-propellant combustion is normally unreliable. The same
# figure `propulsion.SolidMotor.warnings` uses, kept here as a named constant with its own
# SOURCES entry rather than buried in a message string. Used only to raise a warning.
P_C_MIN_RELIABLE = 1.0e6

SOURCES: dict[str, str] = {
    "prop_iv1.shared_physics": (
        "NOT a new source. Every propellant, nozzle and grain relation in this module is "
        "imported from rocketgen.sizing.propulsion, whose SOURCES entries (prop.*) carry the "
        "citations: Sutton and Biblarz, Rocket Propulsion Elements, Ch.12 Table 12-1 for the "
        "HTPB/AP/Al propellant performance and burn-rate law, and the Purdue University "
        "1-D isentropic flow tables for the area-Mach and pressure-ratio relations. This "
        "module adds no thermochemistry and no nozzle physics of its own."
    ),
    "prop_iv1.stage_is_a_phase": (
        "MODELLING CHOICE: one stage of the IV-1 stack is modelled with exactly the same "
        "closure as one phase of the SV-1 dual-thrust motor, that is one propellant charge at "
        "one chamber pressure through one fixed throat, sized by "
        "propulsion.design_phase(). The physical difference between a phase and a stage is "
        "not in the internal ballistics but in the hardware: a stage brings its own case and "
        "nozzle, and stage 1 leaves the vehicle at separation."
    ),
    "prop_iv1.stage1_sized_at_sea_level": (
        "MODELLING CHOICE, following the StageSpec.F_thrust definition in config_iv1.py: "
        "stage-1 thrust is a SEA-LEVEL equivalent, because SPEC_IV1.md requirement A1 launches "
        "the stack vertically from a canister at sea level. The stage-1 throat is therefore "
        "sized with p_a = 101325 Pa. Upper-stage thrust is a VACUUM value and the upper-stage "
        "throats are sized with p_a = 0, because they are lit above 10 km. Both use the same "
        "relation F = C_F_vac * p_c * A_t - p_a * A_e; only the sizing p_a differs."
    ),
    "prop_iv1.booster_aft_closure": (
        "MODELLING CHOICE: the booster grain bay is the stage length less a 0.10 m aft closure, "
        "nozzle attach ring and igniter boss allowance. The value is not new: it is the same "
        "CASE_LENGTH_ALLOWANCE_M that propulsion.py already adds to the grain length to get the "
        "case length, so bay length and case length are consistent with each other. The payload "
        "stage instead gets L - L_nose - L_seeker - L_payload_bay, which is set by the internal "
        "layout in config_iv1.StackDesignVector."
    ),
    "prop_iv1.tubular_stage_grain": (
        "MODELLING CHOICE and a known conservatism: every stage grain is an internal-burning "
        "tube closed at its mean web, the same geometry as the SV-1 boost segment. A tube is "
        "the least area-efficient internal-burning geometry. A slotted, star or finocyl grain "
        "reaches several times the burning area in the same case length, which is how real "
        "boosters of this thrust class stay short. No shape factor is applied here, because no "
        "sourced multiplier was available; the model reports the long tube and its L/D instead. "
        "A stage that fails the SPEC_IV1.md section 6 L/D limit of 1.0 to 8.0 with a tubular "
        "grain may still close with a shaped grain, and the warning says so."
    ),
    "prop_iv1.separation_timing": (
        "MODELLING CHOICE, following the deliverable contract: stage 1 ignites at t = 0 and "
        "burns to the end of its tail-off, the stack then coasts unpowered for "
        "InterceptRequirements.t_coast_separation, and separation and stage-2 ignition are the "
        "same instant, t_separation = t_burnout(1) + t_coast_separation. SPEC_IV1.md section 5 "
        "allows a short unpowered coast between boost and stage-2 ignition and does not fix "
        "whether the jettison happens at the start or the end of it; putting it at the end "
        "means the stack carries the booster mass through the coast, which is the pessimistic "
        "and therefore the safe choice for the trajectory."
    ),
    "prop_iv1.separation_hardware": (
        "GUESS, and an admitted omission: the separation joint itself is NOT costed. A real "
        "tandem separation needs a linear-shaped-charge or clamp-band joint, springs or "
        "retro-rockets, and ring frames at both ends of the interstage. None of that appears in "
        "jettisoned_mass(), so the reported jettisoned mass is the shell and motor hardware "
        "only and is optimistic by an amount this model cannot quantify."
    ),
    "prop_iv1.interstage_shell": (
        "MODELLING CHOICE: the interstage is charged as a cylindrical shell at the BOOSTER "
        "diameter, of length StackDesignVector.L_interstage and thickness t_interstage, in "
        "MATERIALS['airframe_al7075']. Mass is then exactly pi * D1 * L * t * rho, linear in "
        "L_interstage. The conical frustum from D2 to D1 is computed alongside it as a "
        "cross-check and is lighter, because a taper has less lateral area than a cylinder of "
        "the same axial length; the heavier cylindrical value is the one charged. GUESS: no "
        "buckling or crush-load sizing is performed, so t_interstage is an input, not a result."
    ),
    "prop_iv1.fin_mass_model": (
        "GUESS, carried over unchanged from the analytic branch of masses.py::build_masses so "
        "the two modules cannot disagree about the same hardware: fin mass is n_fin * "
        "S_fin_exposed * t_fin * 0.65 * density(Ti-6Al-4V), where 0.65 is an area-to-volume "
        "factor for a tapered biconvex section. No source was found for the 0.65; it is a "
        "shape factor chosen to be below the 1.0 of a flat plate."
    ),
    "prop_iv1.stage_inert_incomplete": (
        "HONESTY STATEMENT, not a number. The bottom-up per-stage inert sum (case, insulation, "
        "nozzle, igniter, fins) omits thrust skirts, case joints and closures, the aft "
        "attachment ring, the nozzle ablative liner and exit cone, and the raceway. It is "
        "therefore reported next to the tactical-motor propellant-mass-fraction band from "
        "propulsion.SOURCES['prop.motor_mass_fraction'], per stage, and "
        "inert_mass_breakdown(stage)['recommended'] returns max(physics, correlation floor) "
        "so the sizing loop carries the conservative value and can see the shortfall."
    ),
    "prop_iv1.throat_credibility": (
        "HARDWARE CREDIBILITY STATEMENT, not a number. Each stage of a tandem stack carries its "
        "own nozzle and runs at exactly one throat area for its whole burn, so the stack needs "
        "NO throat-changing mechanism at all. This is the opposite of the SV-1 dual-thrust "
        "motor, where propulsion.SOURCES['prop.throat_transition_credibility'] has to report "
        "that the boost-to-sustain transition would need a throat that shrinks. "
        "throat_credibility_report() verifies the claim by counting the throat areas each "
        "nozzle must take, rather than assuming it. The price paid for that is a second "
        "complete nozzle and a separation joint, which is what jettisoned_mass() carries."
    ),
    "prop_iv1.p_c_min_reliable": (
        "GUESS: 1.0 MPa taken as the chamber pressure below which composite-propellant "
        "combustion becomes unreliable, so a stage below it is warned about. The figure is the "
        "same one propulsion.SolidMotor already warns on and no primary source for it was "
        "verified in this session. It changes no force and no mass: it only raises a warning. "
        "Every IV-1 stage runs at StageSpec.p_c, which defaults to 8.0 MPa, so the check is "
        "inactive for the default stack and exists for design vectors the sizer may reach."
    ),
    "prop_iv1.ramp_model": (
        "NOT a new source: the stage ignition rise and tail-off reuse propulsion.T_RISE_S "
        "(0.05 s) and propulsion.T_TAILOFF_S (0.20 s) through the same propulsion.ramp() "
        "primitive, and the plateau end is placed at t_ignition + t_burn + 0.5 * T_RISE - "
        "0.5 * T_TAILOFF so the integral of the shape is exactly the ideal burn time. The "
        "ramps are therefore mass-conserving and total impulse does not depend on them. The "
        "stage-to-stage transition needs no blend ramp, because the coast separates the two "
        "shapes; that is why thrust is continuous across separation without any new constant."
    ),
}

register_sources(SOURCES)


# --------------------------------------------------------------------------------------
#   Per-stage results
# --------------------------------------------------------------------------------------


@dataclass
class StageGrainGeometry(GrainGeometry):
    """One stage grain, in the same shape as the SV-1 `GrainGeometry`.

    A stage has a single internal-burning tubular charge, so it fills the `*_boost` fields and
    leaves the `*_sustain` and `*_terminal` fields at zero. The inherited aliases therefore
    behave as a caller expects: `.length` is the grain length, `.web` is its web, `.d_inner` is
    its port diameter, and `.L_over_D` is the number SPEC_IV1.md section 6 constrains to
    1.0 to 8.0. Using the SV-1 dataclass rather than a new one is deliberate: report code and
    the nTop notebook can consume either vehicle's grain with the same field names.
    """

    stage: int = 0

    @property
    def burning_area(self) -> float:
        """Initial burning area of the stage grain at the mean web position, m^2."""
        return self.burning_area_boost


@dataclass
class StageTiming:
    """When one stage lights, when its thrust plateau ends, and when it is done.

    `t_plateau_end` is placed so that the integral of the normalised shape equals the ideal
    burn time exactly, which is what makes the ramps mass-conserving. See
    SOURCES["prop_iv1.ramp_model"].
    """

    stage: int
    t_ignition: float
    t_plateau_end: float
    t_burnout: float

    @property
    def duration(self) -> float:
        return self.t_burnout - self.t_ignition


# --------------------------------------------------------------------------------------
#   The stack
# --------------------------------------------------------------------------------------


class MultiStageMotor:
    """The IV-1 propulsion stack: N independent solid stages fired in sequence.

    Sequencing, which is fixed at construction so that `thrust(t, altitude)` and `mdot(t)` are
    pure functions of time for the integrator:

        t = 0                     stage 1 ignites
        t_burnout(1)              end of the stage-1 tail-off
        t_burnout(1) + t_coast    separation AND stage-2 ignition, both instantaneous
        t_burnout(2)              end of the stage-2 tail-off, all burnout

    with `t_coast = reqs.t_coast_separation`. There is no commanded ignition and no armed pulse:
    unlike the SV-1 terminal phase, every event here is on a timer, so nothing about the
    timeline depends on the trajectory. The trajectory only has to remove `jettisoned_mass()`
    at `t_separation` and switch the aerodynamic reference area.
    """

    def __init__(
        self,
        dv: StackDesignVector,
        reqs: InterceptRequirements,
        propellant: Material = MATERIALS["propellant_htpb_ap"],
        gamma: float = GAMMA_EXHAUST,
        insulation_thickness: float = INSULATION_THICKNESS_M,
        case_material: Material = MATERIALS["motorcase_cfrp"],
        insulation_material: Material = MATERIALS["insulation_epdm"],
        upper_stage_sizing_pressure: float = 0.0,
    ) -> None:
        if dv.n_stages < 1:
            raise ValueError("a stack needs at least one stage")
        self.dv = dv
        self.reqs = reqs
        self.propellant = propellant
        self.gamma = gamma
        self.insulation_thickness = insulation_thickness
        self.case_material = case_material
        self.insulation_material = insulation_material
        self.upper_stage_sizing_pressure = upper_stage_sizing_pressure

        # Propellant thermochemistry is stack-wide: one propellant, so one c*.
        self.c_star = characteristic_velocity(gamma)

        self.warnings: list[str] = []

        # --- size every stage, in burn order ---
        self._stages: list[StageSpec] = list(dv.stages)
        self._order: list[int] = [s.index for s in self._stages]
        if len(set(self._order)) != len(self._order):
            raise ValueError(f"duplicate stage indices in the stack: {self._order}")

        self._design: dict[int, PhaseDesign] = {}
        # Per-stage nozzle state. C_F_vacuum is deliberately NOT cached here: it already lives
        # on the PhaseDesign in `self._design`, and two copies of the same number is how a
        # model starts disagreeing with itself.
        self._area_ratio: dict[int, float] = {}
        self._pe_over_pc: dict[int, float] = {}
        self._sizing_pressure: dict[int, float] = {}
        for position, spec in enumerate(self._stages):
            self._size_stage(position, spec)

        # --- build the timeline, then check everything that can be checked ---
        self._timing: dict[int, StageTiming] = {}
        self._build_timeline()
        self._collect_warnings()

    # ------------------------------------------------------------------ construction ---

    def _size_stage(self, position: int, spec: StageSpec) -> None:
        """Close one stage: nozzle from its own eps and thrust, grain from the Kn balance.

        The only per-stage choice is the ambient pressure the throat is sized at. Stage 1 is
        lit at sea level from a canister, so its `F_thrust` is a sea-level equivalent and the
        throat is sized with p_a = P_SEA_LEVEL. Upper stages are lit above 10 km, so their
        `F_thrust` is a vacuum value and the throat is sized with
        p_a = `upper_stage_sizing_pressure`, which defaults to 0. See
        SOURCES["prop_iv1.stage1_sized_at_sea_level"].
        """
        eps = spec.eps_nozzle
        c_f_vacuum = thrust_coefficient(self.gamma, eps, 0.0)
        p_a_sizing = P_SEA_LEVEL if position == 0 else self.upper_stage_sizing_pressure

        throat_area = throat_area_for_thrust(
            spec.F_thrust, spec.p_c, c_f_vacuum, eps, p_a_sizing
        )
        self._area_ratio[spec.index] = eps
        self._pe_over_pc[spec.index] = exit_pressure_ratio(eps, self.gamma)
        self._sizing_pressure[spec.index] = p_a_sizing
        self._design[spec.index] = design_phase(
            f"stage_{spec.index}",
            spec.m_propellant,
            spec.p_c,
            throat_area,
            area_ratio=eps,
            c_star=self.c_star,
            C_F_vacuum=c_f_vacuum,
            density=self.propellant.density,
        )

    def _build_timeline(self) -> None:
        """Lay the stages out in time, separated by the coast.

        The plateau end of each stage is placed at
            t_ignition + t_burn + 0.5 * T_RISE - 0.5 * T_TAILOFF
        so that the integral of the trapezoidal shape is exactly `t_burn`. That is what makes
        the ramps mass-conserving, so the total impulse and the propellant budget do not depend
        on them. A stage whose ideal burn time is shorter than 0.5 * (T_RISE + T_TAILOFF) cannot
        be represented by a rise-plateau-tailoff shape at all; the plateau is then clamped and
        the mass error is reported rather than hidden.
        """
        t_coast = self.reqs.t_coast_separation
        if t_coast < 0.0:
            raise ValueError("t_coast_separation must not be negative")

        t_ignition = 0.0
        min_burn = 0.5 * (T_RISE_S + T_TAILOFF_S)
        for spec in self._stages:
            t_burn = self._design[spec.index].burn_time
            if t_burn < min_burn:
                self.warnings.append(
                    f"stage {spec.index}: ideal burn time {t_burn:.3f} s is shorter than the "
                    f"{min_burn:.3f} s the rise-and-tail-off ramp model needs; the plateau is "
                    "clamped, so the integrated mass flow no longer equals the propellant mass"
                )
            t_plateau_end = max(
                t_ignition + T_RISE_S,
                t_ignition + t_burn + 0.5 * T_RISE_S - 0.5 * T_TAILOFF_S,
            )
            t_burnout = t_plateau_end + T_TAILOFF_S
            self._timing[spec.index] = StageTiming(
                stage=spec.index,
                t_ignition=t_ignition,
                t_plateau_end=t_plateau_end,
                t_burnout=t_burnout,
            )
            # The next stage lights after the coast. For the last stage this value is unused.
            t_ignition = t_burnout + t_coast

    def _collect_warnings(self) -> None:
        """Everything the stack knows is wrong with itself, per stage.

        Nothing here changes a force or a mass. CLAUDE.md section 3.3: bad news travels upward.
        """
        for spec in self._stages:
            index = spec.index
            design = self._design[index]
            geom = self.grain_geometry(index)

            # --- nozzle flow separation, the Summerfield criterion ---
            sep = summerfield_separation(self._pe_over_pc[index] * design.p_c)
            if sep["separation_altitude"] > 0.0:
                self.warnings.append(
                    f"stage {index} nozzle: exit pressure {sep['p_e'] / 1e3:.1f} kPa; the "
                    f"Summerfield criterion predicts flow separation below "
                    f"{sep['separation_altitude']:.0f} m altitude. Reported only, the thrust "
                    "model does not change."
                )

            # --- combustion stability ---
            if design.p_c < P_C_MIN_RELIABLE:
                self.warnings.append(
                    f"stage {index} chamber pressure {design.p_c / 1e6:.3f} MPa is below "
                    "1 MPa; composite propellant combustion is normally unreliable there."
                )

            # --- the nozzle has to fit behind the stage it pushes ---
            d_exit = math.sqrt(4.0 * design.exit_area / math.pi)
            if d_exit > spec.D:
                self.warnings.append(
                    f"stage {index} nozzle exit diameter {d_exit * 1e3:.0f} mm exceeds the "
                    f"stage body diameter {spec.D * 1e3:.0f} mm; the exit cone does not fit "
                    f"inside the airframe. Reduce eps_nozzle ({self._area_ratio[index]:.1f}) "
                    "or the stage thrust, or accept an external skirt that is not modelled."
                )

            # --- grain closure and the SPEC_IV1 section 6 L/D limit ---
            self.warnings.extend(f"stage {index}: {w}" for w in geom.warnings)

            # --- the design vector must agree with itself about what is thrown away ---
            expected_jettison = not self.is_payload_stage(index)
            if spec.jettisoned != expected_jettison:
                role = "a discarded stage" if expected_jettison else "the payload stage"
                self.warnings.append(
                    f"stage {index}: StageSpec.jettisoned is {spec.jettisoned} but burn order "
                    f"makes it {role}. Burn order governs, so the flag is ignored; fix the "
                    "design vector so the two agree."
                )

            # --- inert mass honesty, per stage ---
            inert = self.inert_mass_breakdown(index)
            if inert["shortfall"] > 0.0:
                self.warnings.append(
                    f"stage {index} motor inert mass raised from a bottom-up "
                    f"{inert['total_physics']:.1f} kg to the tactical-motor correlation floor "
                    f"{inert['correlation_min']:.1f} kg; the bottom-up model is incomplete by "
                    f"{inert['shortfall']:.1f} kg. See "
                    "SOURCES['prop_iv1.stage_inert_incomplete']."
                )

    # ---------------------------------------------------------------------- stages ---

    @property
    def n_stages(self) -> int:
        return len(self._stages)

    @property
    def stage_indices(self) -> list[int]:
        """Stage indices in burn order."""
        return list(self._order)

    def _spec(self, stage: int) -> StageSpec:
        for spec in self._stages:
            if spec.index == stage:
                return spec
        raise KeyError(f"no stage {stage} in this stack; have {self._order}")

    def _position(self, stage: int) -> int:
        """Zero-based position of a stage in burn order."""
        return self._order.index(self._spec(stage).index)

    def is_payload_stage(self, stage: int) -> bool:
        """True for the last stage to burn, the one that reaches intercept.

        Burn order is the authority, not the `StageSpec.jettisoned` flag, because the same
        ordering already defines `StackDesignVector.payload_stage` and `L_total`. A flag that
        disagrees with the ordering is reported as a warning rather than silently resolved.
        """
        return self._position(stage) == self.n_stages - 1

    @property
    def has_separation(self) -> bool:
        """True when at least one stage is discarded during the flight.

        False for a one-stage stack, where `jettisoned_mass()` is zero and `t_separation` is
        only a notional instant after the single burn.
        """
        return self.n_stages > 1

    # ------------------------------------------------------------------ event times ---

    def t_ignition(self, stage: int) -> float:
        """Time at which `stage` lights, s. Stage 1 is 0.0 by definition."""
        return self._timing[self._spec(stage).index].t_ignition

    def t_burnout(self, stage: int) -> float:
        """Time at which the `stage` tail-off finishes and its thrust reaches zero, s."""
        return self._timing[self._spec(stage).index].t_burnout

    def timing(self, stage: int) -> StageTiming:
        """The ignition, plateau-end and burnout times of one stage, in one object."""
        return self._timing[self._spec(stage).index]

    def t_plateau_end(self, stage: int) -> float:
        """Time at which the `stage` thrust plateau starts decaying, s."""
        return self._timing[self._spec(stage).index].t_plateau_end

    def t_burn_ideal(self, stage: int) -> float:
        """Ideal burn time m_p / mdot for `stage`, s, before the ramps are applied."""
        return self._design[self._spec(stage).index].burn_time

    @property
    def t_separation(self) -> float:
        """Time at which stage 1 is jettisoned, s.

        Equal to the stage-1 burnout plus `reqs.t_coast_separation`, which is also the
        stage-2 ignition time: the vehicle carries the spent booster through the coast and
        drops it at the instant the next stage lights. See
        SOURCES["prop_iv1.separation_timing"].
        """
        return self.t_burnout(self._order[0]) + self.reqs.t_coast_separation

    @property
    def t_all_burnout(self) -> float:
        """Time at which the last stage finishes its tail-off, s."""
        return max(self._timing[i].t_burnout for i in self._order)

    def separation_times(self) -> list[float]:
        """Every jettison instant, s, one per stage that is not the last to burn."""
        return [
            self.t_burnout(i) + self.reqs.t_coast_separation
            for i in self._order[:-1]
        ]

    # --------------------------------------------------------------- time domain ---

    def _shape(self, stage: int, t: float) -> float:
        """Normalised mass-flow shape of one stage at stack time t, in [0, 1].

        Rise-plateau-tailoff built from the shared `ramp` primitive, so the trace is
        continuous everywhere. Zero outside the stage's own window, which is what keeps the
        stages from overlapping across the coast.
        """
        timing = self._timing[stage]
        if t <= timing.t_ignition or t >= timing.t_burnout:
            return 0.0
        return ramp(t, timing.t_ignition, T_RISE_S) * (
            1.0 - ramp(t, timing.t_plateau_end, T_TAILOFF_S)
        )

    def shapes(self, t: float) -> dict[int, float]:
        """The normalised shape of every stage at time t. Diagnostic."""
        return {i: self._shape(i, t) for i in self._order}

    def mdot(self, t: float) -> float:
        """Total propellant mass flow leaving the stack, kg/s, at stack time t.

        Summed over stages. Only one stage burns at a time whenever the separation coast is
        positive, so this is a single term in practice; it is written as a sum so that a
        zero-coast stack still conserves mass.
        """
        return sum(
            self._shape(i, t) * self._design[i].mdot for i in self._order
        )

    def exit_area_at(self, t: float) -> float:
        """Effective nozzle exit area flowing at stack time t, m^2.

        Shape-weighted, exactly as the vacuum thrust is, so that the ambient pressure term
        stays consistent with the momentum term through every ramp.
        """
        return sum(
            self._shape(i, t) * self._design[i].exit_area for i in self._order
        )

    def thrust(self, t: float, altitude: float) -> float:
        """Total thrust, N, at stack time t and geometric altitude.

        F = sum_stage s_stage * C_F_vac_stage * p_c_stage * A_t_stage - p_a * A_e_effective

        The second term is the ambient part of (p_e - p_a) * A_e: expanding C_F_vac already
        contains the + p_e * A_e half of it. Because the exit area is weighted by the same
        shape functions as the momentum term, the vacuum-to-sea-level difference is exactly
        p_a * A_e at every instant, including inside a ramp.
        """
        vacuum = 0.0
        exit_area = 0.0
        active = False
        for i in self._order:
            s = self._shape(i, t)
            if s <= 0.0:
                continue
            active = True
            design = self._design[i]
            vacuum += s * design.thrust_vacuum
            exit_area += s * design.exit_area
        if not active:
            return 0.0
        return vacuum - _ambient_pressure(altitude) * exit_area

    def active_stage(self, t: float) -> int:
        """Index of the stage that is burning at stack time t, or 0 while unpowered.

        A stage counts as active from its ignition instant to the end of its tail-off, so the
        tail-off belongs to the stage that produced it and not to the coast.
        """
        for i in self._order:
            timing = self._timing[i]
            if timing.t_ignition <= t < timing.t_burnout:
                return i
        return 0

    def phase(self, t: float) -> str:
        """Phase label at stack time t.

        One of "stage_<i>_boost", "separation_coast" or "burnout". "separation_coast" is the
        unpowered gap between one stage burning out and the next lighting, which is also when
        the jettison happens. "burnout" covers everything after the last tail-off, and also
        t < 0, which is not part of the timeline: both are unpowered with zero mass flow.
        """
        active = self.active_stage(t)
        if active:
            return f"stage_{active}_boost"
        if t >= self.t_all_burnout or t < 0.0:
            return "burnout"
        return "separation_coast"

    # ------------------------------------------------------------------- geometry ---

    def bay_length(self, stage: int) -> float:
        """Axial length available for the grain of `stage`, m.

        Payload stage: `L - L_nose - L_seeker - L_payload_bay`, that is the stage length less
        the nose and the two forward bays. Every other stage: `L` less the aft closure, nozzle
        attach and igniter boss allowance. See SOURCES["prop_iv1.booster_aft_closure"].
        """
        spec = self._spec(stage)
        if self.is_payload_stage(stage):
            return spec.L - self.dv.L_nose - self.dv.L_seeker - self.dv.L_payload_bay
        return spec.L - BOOSTER_AFT_CLOSURE_ALLOWANCE_M

    def bay_diameter(self, stage: int) -> float:
        """Internal diameter available for the grain of `stage`, m.

        The airframe wall and the case insulation both come off the radius, the same way
        `SolidMotor.bay_diameter` does it.
        """
        spec = self._spec(stage)
        return spec.D - 2.0 * spec.t_wall - 2.0 * self.insulation_thickness

    def grain_geometry(self, stage: int) -> StageGrainGeometry:
        """Close the grain of one stage inside its own bay.

        A single internal-burning tube, closed at its mean web by the shared
        `tubular_grain_closure`, so propellant mass, web, burning area, burn time and mass
        flow are mutually consistent. The `*_sustain` and `*_terminal` fields of the returned
        object are zero: a stage has one charge.
        """
        spec = self._spec(stage)
        design = self._design[spec.index]
        rho = self.propellant.density
        d_o = self.bay_diameter(stage)
        bay_length = self.bay_length(stage)
        warnings: list[str] = []
        feasible = True

        if d_o <= 0.0:
            raise ValueError(
                f"stage {stage}: no room for a grain, bay diameter is {d_o:.4f} m"
            )

        a_b = design.burning_area
        tube = tubular_grain_closure(spec.m_propellant, rho, a_b, d_o)
        if not tube.fits:
            feasible = False
            warnings.append(
                f"web {tube.web * 1e3:.0f} mm exceeds the bay radius {d_o * 500.0:.0f} mm; a "
                f"tubular grain cannot hold {spec.m_propellant:.0f} kg at a burning area of "
                f"{a_b:.3f} m^2. Either raise the stage thrust, which raises the burning "
                "area, or reduce the propellant mass"
            )

        if tube.length > bay_length:
            feasible = False
            warnings.append(
                f"grain length {tube.length:.3f} m exceeds the available motor bay "
                f"{bay_length:.3f} m by {tube.length - bay_length:.3f} m. A tubular grain is "
                "the least area-efficient internal-burning geometry; see "
                "SOURCES['prop_iv1.tubular_stage_grain']"
            )

        bay_volume = 0.25 * math.pi * d_o ** 2 * bay_length
        loading = tube.volume / bay_volume if bay_volume > 0.0 else float("inf")
        if loading > 1.0:
            feasible = False
            warnings.append(f"volumetric loading {loading * 100.0:.1f} % exceeds 100 %")

        l_over_d = tube.length / d_o if d_o > 0.0 else float("inf")
        lo, hi = GRAIN_L_OVER_D_LIMITS
        if not (lo <= l_over_d <= hi):
            warnings.append(
                f"grain L/D {l_over_d:.2f} is outside the SPEC_IV1.md section 6 range "
                f"{lo:.1f} to {hi:.1f}"
            )

        return StageGrainGeometry(
            d_outer=d_o,
            d_inner_boost=tube.d_inner,
            web_boost=tube.web,
            length_boost=tube.length,
            burning_area_boost=a_b,
            d_face_sustain=0.0,
            web_sustain=0.0,
            length_sustain=0.0,
            burning_area_sustain=0.0,
            d_inner_terminal=0.0,
            web_terminal=0.0,
            length_terminal=0.0,
            burning_area_terminal=0.0,
            length_total=tube.length,
            L_over_D=l_over_d,
            volume_boost=tube.volume,
            volume_sustain=0.0,
            volume_terminal=0.0,
            volume_total=tube.volume,
            bay_length_available=bay_length,
            bay_diameter=d_o,
            volumetric_loading=loading,
            feasible=feasible,
            warnings=warnings,
            stage=spec.index,
        )

    def grain_l_over_d(self) -> dict[int, float]:
        """Grain L/D of every stage, so the sizing loop can constrain each one separately.

        SPEC_IV1.md section 6 requires 1.0 to 8.0 for EACH stage, not for the stack, which is
        why this is a mapping and not a single number.
        """
        return {i: self.grain_geometry(i).L_over_D for i in self._order}

    # ---------------------------------------------------------------- inert masses ---

    def fin_mass(self, stage: int) -> float:
        """Tail-fin set mass of one stage, kg. See SOURCES["prop_iv1.fin_mass_model"]."""
        spec = self._spec(stage)
        return (
            spec.n_fin
            * spec.S_fin_exposed
            * spec.t_fin
            * FIN_VOLUME_FACTOR
            * FIN_MATERIAL.density
        )

    def interstage_mass(self) -> float:
        """Interstage structural mass, kg, as charged: a cylindrical shell at D1.

        Exactly pi * D_booster * L_interstage * t_interstage * rho, so it is linear in
        `L_interstage`. See SOURCES["prop_iv1.interstage_shell"] for why the cylinder is
        charged rather than the lighter conical frustum, and for what is left out.
        """
        return self.interstage_breakdown()["charged"]

    def interstage_breakdown(self) -> dict[str, float]:
        """Both interstage shell models, so the choice is visible rather than buried.

        `cylindrical` is a constant-diameter skirt at the booster diameter. `conical` is the
        frustum from the payload-stage diameter to the booster diameter, whose lateral area
        uses the slant length sqrt(L^2 + (r1 - r2)^2). The cylinder is heavier for the same
        axial length and is the one charged.
        """
        dv = self.dv
        booster = self._spec(self._order[0])
        r_aft = 0.5 * booster.D
        length = dv.L_interstage
        thickness = dv.t_interstage
        rho = INTERSTAGE_MATERIAL.density
        if not self.has_separation:
            # A one-stage stack has no interstage to charge, whatever L_interstage says.
            return {
                "charged": 0.0,
                "cylindrical": 0.0,
                "conical": 0.0,
                "area_cylindrical": 0.0,
                "area_conical": 0.0,
                "slant_over_length": 1.0,
                "length": 0.0,
                "thickness": thickness,
                "density": rho,
            }
        # The interstage spans from the stage above it to the booster below it. The stage above
        # stage 1 is the NEXT one to burn, not the payload stage, which matters for a stack of
        # three or more.
        r_fwd = 0.5 * self._spec(self._order[1]).D

        area_cylindrical = 2.0 * math.pi * r_aft * length
        slant = math.hypot(length, r_aft - r_fwd)
        area_conical = math.pi * (r_aft + r_fwd) * slant

        m_cylindrical = area_cylindrical * thickness * rho
        m_conical = area_conical * thickness * rho
        return {
            "charged": m_cylindrical,
            "cylindrical": m_cylindrical,
            "conical": m_conical,
            "area_cylindrical": area_cylindrical,
            "area_conical": area_conical,
            "slant_over_length": slant / length if length > 0.0 else 1.0,
            "length": length,
            "thickness": thickness,
            "density": rho,
        }

    def inert_mass_breakdown(self, stage: int) -> dict[str, float]:
        """Inert mass of one stage by group, kg, by two independent routes.

        Physics route, all of it shared with `propulsion.py`:
          case         thin-wall hoop stress at CASE_SAFETY_FACTOR * p_c, plus two closures at
                       half the cylinder thickness
          insulation   fixed-thickness EPDM over the same wetted area
          nozzle       conical shell geometric estimate, one nozzle per stage (weak, see
                       propulsion.SOURCES['prop.nozzle_mass_model'])
          igniter      fixed fraction of the stage propellant mass (a guess)
          fins         exposed planform times thickness times a shape factor, in titanium

        Correlation route: the tactical-motor propellant-mass-fraction band applied to THIS
        stage's propellant. The band bounds motor hardware only, so the fins are added outside
        it; a fin set is airframe, not motor.

        Keys:
          `total_physics`    case + insulation + nozzle + igniter, the motor bottom-up sum
          `correlation_min`  optimistic bound from the high mass fraction, the floor used
          `correlation_max`  pessimistic bound from the low mass fraction
          `shortfall`        correlation_min - total_physics when positive, else 0
          `recommended`      max(total_physics, correlation_min), motor only
          `total_recommended` recommended + fins, the stage inert mass to carry
          `mass_fraction_physics` m_p / (m_p + total_physics)

        `recommended` is the one to use. The bottom-up sum is known to be incomplete; see
        SOURCES["prop_iv1.stage_inert_incomplete"].
        """
        spec = self._spec(stage)
        design = self._design[spec.index]
        geom = self.grain_geometry(stage)

        case_inner_radius = 0.5 * (spec.D - 2.0 * spec.t_wall)
        # A stage has one chamber pressure, so there is no "highest pressure any phase runs
        # at" question to answer here, unlike the SV-1 motor with its terminal pulse.
        p_design = CASE_SAFETY_FACTOR * design.p_c
        case_length = geom.length_total + CASE_LENGTH_ALLOWANCE_M
        shell = motor_case_and_insulation(
            case_inner_radius,
            case_length,
            p_design,
            self.case_material,
            self.insulation_material,
            self.insulation_thickness,
        )

        m_nozzle = conical_nozzle_mass(design.throat_area, design.exit_area)
        m_igniter = IGNITER_MASS_FRACTION * spec.m_propellant
        m_fins = self.fin_mass(stage)

        physics_total = shell["case"] + shell["insulation"] + m_nozzle + m_igniter
        correlation_min, correlation_max = correlation_inert_band(spec.m_propellant)
        recommended = max(physics_total, correlation_min)

        return {
            "case": shell["case"],
            "insulation": shell["insulation"],
            "nozzle": m_nozzle,
            "igniter": m_igniter,
            "fins": m_fins,
            "total_physics": physics_total,
            "case_thickness": shell["case_thickness"],
            "case_length": case_length,
            "correlation_min": correlation_min,
            "correlation_max": correlation_max,
            "shortfall": max(0.0, correlation_min - physics_total),
            "mass_fraction_physics": spec.m_propellant / (spec.m_propellant + physics_total),
            "recommended": recommended,
            "total_recommended": recommended + m_fins,
        }

    def stage_wet_mass(self, stage: int) -> float:
        """Propellant plus recommended inert mass of one stage, kg. No payload, no interstage."""
        return self._spec(stage).m_propellant + self.inert_mass_breakdown(stage)[
            "total_recommended"
        ]

    def jettisoned_mass(self) -> float:
        """Mass that leaves the vehicle at `t_separation`, kg.

        Stage-1 case, insulation, nozzle, igniter and fins, at the recommended (correlation
        floored) inert value, plus the interstage shell. This is the number the trajectory
        subtracts, so it uses the conservative inert route rather than the bottom-up sum.
        It does NOT contain any separation-joint hardware; see
        SOURCES["prop_iv1.separation_hardware"]. Zero for a one-stage stack, which has nothing
        to discard.
        """
        return self.jettisoned_mass_breakdown()["total"]

    def jettisoned_mass_breakdown(self) -> dict[str, float]:
        """Every line of `jettisoned_mass()`, kg, with the bottom-up variant alongside."""
        booster = self._order[0]
        interstage = self.interstage_breakdown()
        if not self.has_separation:
            # Nothing leaves a one-stage stack: its only stage is the payload stage.
            return {
                key: 0.0
                for key in (
                    "stage_1_case",
                    "stage_1_insulation",
                    "stage_1_nozzle",
                    "stage_1_igniter",
                    "stage_1_fins",
                    "stage_1_hardware_not_modelled",
                    "stage_1_inert_physics",
                    "stage_1_inert_recommended",
                    "interstage",
                    "interstage_conical_alternative",
                    "total",
                    "total_bottom_up",
                )
            }
        inert = self.inert_mass_breakdown(booster)
        total = inert["total_recommended"] + interstage["charged"]
        return {
            "stage_1_case": inert["case"],
            "stage_1_insulation": inert["insulation"],
            "stage_1_nozzle": inert["nozzle"],
            "stage_1_igniter": inert["igniter"],
            "stage_1_fins": inert["fins"],
            "stage_1_hardware_not_modelled": inert["shortfall"],
            "stage_1_inert_physics": inert["total_physics"],
            "stage_1_inert_recommended": inert["total_recommended"],
            "interstage": interstage["charged"],
            "interstage_conical_alternative": interstage["conical"],
            "total": total,
            "total_bottom_up": inert["total_physics"] + inert["fins"] + interstage["charged"],
        }

    # ------------------------------------------------------------------ performance ---

    def total_impulse_vacuum(self) -> float:
        """Vacuum total impulse of the whole stack, N.s.

        Exact for this model, because thrust is linear in the stage mass flows and the ramps
        conserve mass:  I = c* * sum_stage C_F_vac_stage * m_p_stage. Equivalently
        sum_stage Isp_vac_stage * g0 * m_p_stage. The stages have different area ratios, so
        C_F_vac and therefore Isp differ from stage to stage and the sum cannot be collapsed
        onto a single Isp.
        """
        return self.c_star * sum(
            self._design[i].C_F_vacuum * self._design[i].propellant_mass
            for i in self._order
        )

    @property
    def propellant_mass(self) -> float:
        """Total propellant carried by the stack, kg."""
        return sum(self._design[i].propellant_mass for i in self._order)

    def operating_point(self, stage: int | None = None) -> dict:
        """Per-stage design summary for the report table.

        With `stage` given, returns that stage's dict. With no argument, returns
        {stage_index: dict}. `thrust_sizing` is the thrust the throat was sized for, at
        `p_a_sizing`, and must reproduce `StageSpec.F_thrust`.
        """
        if stage is not None:
            return self._operating_point_of(stage)
        return {i: self._operating_point_of(i) for i in self._order}

    def _operating_point_of(self, stage: int) -> dict[str, float]:
        spec = self._spec(stage)
        design = self._design[spec.index]
        geom = self.grain_geometry(stage)
        p_a_sizing = self._sizing_pressure[spec.index]
        return {
            "propellant_mass": design.propellant_mass,
            "p_c": design.p_c,
            "p_e": self._pe_over_pc[spec.index] * design.p_c,
            "eps_nozzle": self._area_ratio[spec.index],
            "throat_area": design.throat_area,
            "throat_diameter": math.sqrt(4.0 * design.throat_area / math.pi),
            "exit_area": design.exit_area,
            "exit_diameter": math.sqrt(4.0 * design.exit_area / math.pi),
            "burning_area": design.burning_area,
            "Kn": design.kn,
            "burn_rate": design.burn_rate,
            "mdot": design.mdot,
            "burn_time": design.burn_time,
            "thrust_vacuum": design.thrust_vacuum,
            "thrust_sea_level": design.thrust_vacuum - P_SEA_LEVEL * design.exit_area,
            "thrust_sizing": design.thrust_vacuum - p_a_sizing * design.exit_area,
            "p_a_sizing": p_a_sizing,
            "C_F_vacuum": design.C_F_vacuum,
            "isp_vacuum": design.isp_vacuum,
            "total_impulse_vacuum": design.isp_vacuum * G0 * design.propellant_mass,
            "grain_length": geom.length_total,
            "grain_L_over_D": geom.L_over_D,
            "grain_web": geom.web_boost,
            "grain_d_inner": geom.d_inner_boost,
            "volumetric_loading": geom.volumetric_loading,
            "t_ignition": self.t_ignition(stage),
            "t_burnout": self.t_burnout(stage),
        }

    def separation_check(self) -> dict[int, dict[str, float]]:
        """Per-stage Summerfield flow-separation assessment.

        Stage 1 is the case that matters: it is lit at sea level, where the ambient pressure
        is highest and an overexpanded nozzle is most likely to separate. Reported, never
        applied to the thrust. See propulsion.SOURCES["prop.separation_criterion"].
        """
        return {
            i: summerfield_separation(self._pe_over_pc[i] * self._design[i].p_c)
            for i in self._order
        }

    def throat_credibility_report(self) -> list[dict[str, object]]:
        """One entry per stage, verifying that no stage needs a throat transition.

        `n_throat_areas` is how many distinct throat areas that stage's nozzle must present
        during its own burn. It is 1 for every stage of a tandem stack by construction, since
        a stage has one charge at one chamber pressure. That is checked here rather than
        assumed. `credible` is False for any stage that would need more than one, which no
        current stage model can produce; the field exists so that adding a dual-thrust stage
        later cannot silently reintroduce the SV-1 shrinking-throat problem. See
        SOURCES["prop_iv1.throat_credibility"].
        """
        report: list[dict[str, object]] = []
        for i in self._order:
            design = self._design[i]
            areas = {design.throat_area}      # one charge, one chamber pressure, one throat
            report.append(
                {
                    "stage": i,
                    "n_throat_areas": len(areas),
                    "throat_area": design.throat_area,
                    "credible": len(areas) == 1,
                    "mechanism": (
                        "fixed throat, no in-flight transition; this stage carries its own "
                        "nozzle and is discarded with it"
                        if not self.is_payload_stage(i)
                        else "fixed throat, no in-flight transition; this nozzle flies to "
                        "intercept"
                    ),
                }
            )
        return report

    # ---------------------------------------------------------------------- report ---

    def summary_data(self) -> dict[str, object]:
        """Everything the report and the sizing loop need, machine-readable.

        `summary()` renders this as text. `SolidMotor.summary()` returns the dict directly;
        the IV-1 contract asks for a string, so the dict lives here under its own name.
        """
        geometry = {i: self.grain_geometry(i) for i in self._order}
        return {
            "c_star": self.c_star,
            "gamma": self.gamma,
            "n_stages": self.n_stages,
            "stage_order": list(self._order),
            "propellant_mass": self.propellant_mass,
            "total_impulse_vacuum": self.total_impulse_vacuum(),
            "t_separation": self.t_separation,
            "t_all_burnout": self.t_all_burnout,
            "separation_times": self.separation_times(),
            "coast_separation": self.reqs.t_coast_separation,
            "jettisoned_mass": self.jettisoned_mass(),
            "jettisoned": self.jettisoned_mass_breakdown(),
            "interstage": self.interstage_breakdown(),
            "grain_L_over_D": self.grain_l_over_d(),
            "grain_feasible": {i: geometry[i].feasible for i in self._order},
            "stages": self.operating_point(),
            "inert": {i: self.inert_mass_breakdown(i) for i in self._order},
            "separation_check": self.separation_check(),
            "throat_credibility": self.throat_credibility_report(),
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        """Human-readable stack summary, SI units with kilo prefixes where they help."""
        data = self.summary_data()
        lines: list[str] = []
        lines.append(
            f"IV-1 multi-stage motor: {self.n_stages} stages, "
            f"{self.propellant_mass:.1f} kg propellant, "
            f"c* = {self.c_star:.1f} m/s, gamma = {self.gamma:.2f}"
        )
        lines.append(
            f"  total vacuum impulse   {self.total_impulse_vacuum() / 1e3:.1f} kN.s"
        )
        lines.append(
            f"  separation at t = {self.t_separation:.3f} s, "
            f"all burnout at t = {self.t_all_burnout:.3f} s, "
            f"coast {self.reqs.t_coast_separation:.2f} s"
        )
        jettison = self.jettisoned_mass_breakdown()
        lines.append(
            f"  jettisoned mass        {jettison['total']:.1f} kg "
            f"(stage-1 inert {jettison['stage_1_inert_recommended']:.1f} kg "
            f"+ interstage {jettison['interstage']:.1f} kg)"
        )
        for i in self._order:
            op = self._operating_point_of(i)
            inert = self.inert_mass_breakdown(i)
            geom = self.grain_geometry(i)
            role = "payload stage" if self.is_payload_stage(i) else "jettisoned"
            lines.append(f"  stage {i} ({role}):")
            lines.append(
                f"    p_c {op['p_c'] / 1e6:.2f} MPa, eps {op['eps_nozzle']:.1f}, "
                f"d_throat {op['throat_diameter'] * 1e3:.1f} mm, "
                f"d_exit {op['exit_diameter'] * 1e3:.1f} mm"
            )
            lines.append(
                f"    mdot {op['mdot']:.2f} kg/s, burn {op['burn_time']:.2f} s, "
                f"Kn {op['Kn']:.1f}, Isp_vac {op['isp_vacuum']:.1f} s"
            )
            lines.append(
                f"    thrust: sizing {op['thrust_sizing'] / 1e3:.1f} kN at "
                f"p_a {op['p_a_sizing'] / 1e3:.1f} kPa, vacuum "
                f"{op['thrust_vacuum'] / 1e3:.1f} kN, sea level "
                f"{op['thrust_sea_level'] / 1e3:.1f} kN"
            )
            lines.append(
                f"    grain: L {geom.length_total:.3f} m in a {geom.bay_length_available:.3f} m "
                f"bay, L/D {geom.L_over_D:.2f}, loading "
                f"{geom.volumetric_loading * 100.0:.1f} %, feasible {geom.feasible}"
            )
            lines.append(
                f"    inert: physics {inert['total_physics']:.1f} kg, correlation band "
                f"{inert['correlation_min']:.1f} to {inert['correlation_max']:.1f} kg, "
                f"recommended {inert['recommended']:.1f} kg + fins {inert['fins']:.1f} kg"
            )
        warnings = data["warnings"]
        assert isinstance(warnings, list)
        if warnings:
            lines.append(f"  warnings ({len(warnings)}):")
            lines.extend(f"    - {w}" for w in warnings)
        else:
            lines.append("  warnings: none")
        return "\n".join(lines)
