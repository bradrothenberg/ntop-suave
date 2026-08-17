"""WP3a - boost-sustain solid rocket motor model.

Scope
-----
Conceptual (Class-I) solid motor sizing for the SV-1 demo rocket. The model computes
nozzle performance from isentropic relations, closes the grain geometry with a cited
burn-rate law, and reports motor inert mass by two independent routes.

Architecture modelled
---------------------
One case, one propellant charge, three burning geometries (a genuine multi-thrust grain,
not three motors):

  * boost segment  - internal-burning tubular (case-bonded) grain. Large burning area,
    high chamber pressure, short burn.
  * sustain segment - end-burning ("cigarette") free-standing grain. Small burning area,
    low mass flow, long burn. End burning is the standard sustainer geometry for tactical
    dual-thrust motors (see SOURCES["prop.end_burning_sustainer"]).
  * terminal segment - internal-burning tubular grain, ignited on COMMAND, not on a
    timer. It exists only when `DesignVector.m_p_terminal` is above zero. Its purpose is
    to carry thrust into the endgame so the impact Mach requirement (SPEC R6) is
    reachable; an unpowered dive is terminal-velocity limited and cannot reach Mach 1.5
    at sea level for any dive angle. See `arm_terminal` and `ignite_terminal`.

Throat area between phases - what is and is not credible hardware
-----------------------------------------------------------------
A single fixed throat cannot support both a 45 kN boost and a ~2.6 kN sustain phase. The
Kn (burning-area / throat-area) closure makes this explicit: with the boost throat sized
for 45 kN, a full-bore end-burning sustainer gives Kn ~= 22 and a chamber pressure near
0.13 MPa, far below any stable operating pressure. This model therefore runs the boost
and sustain phases at different throat areas with the same nozzle area ratio
`eps_nozzle`, and calls that a "two-position throat".

BE CLEAR ABOUT WHAT THAT MEANS. The boost throat is LARGER than the sustain throat, so
the transition needs the throat to SHRINK. No ejectable-insert mechanism does that: an
ejected insert can only enlarge a throat, and throat erosion only enlarges it too. The
credible hardware that gives a smaller effective throat after boost is a separate,
smaller sustainer nozzle in the same aft closure, or a tandem booster that is jettisoned.
Neither is a single-throat motor. This model does not invent a shrinking-throat
mechanism: it reports the throat area of each phase and the direction of every
transition, and `throat_transition_report()` names the ones a detailed design must
resolve. `two_position_throat=False` forces one throat and reports the resulting
infeasibility instead.

The TERMINAL phase deliberately does NOT need a third throat position. It shares the
sustain throat by default (`terminal_throat_source="sustain"`), so it adds no new
hardware transition at all; its chamber pressure follows from its own burning area
through the Kn closure. `terminal_throat_source="boost"` is offered because a
sustain-to-boost throat change is the one transition that IS credible with an ejectable
insert, at the cost of a longer terminal grain.

APPROXIMATIONS - read this before trusting a number
---------------------------------------------------
1. Neutral burning within each phase. The burn-rate law is closed at the MEAN web
   position of each segment, which makes propellant mass, web thickness, burn time and
   mass flow mutually consistent (w = V_p / A_b_mean, t_b = w / r). Thrust is then held
   constant across the phase. A real dual-thrust grain is shaped and inhibited to give
   this neutrality; this model does NOT run a burnback simulation, so the reported
   tubular / end-burner dimensions are volume-and-area-equivalent representations of the
   real grain, not a manufacturing drawing.
2. Frozen-composition, constant-gamma, single-species nozzle flow. No two-phase
   (Al2O3) loss, no divergence loss, no combustion-efficiency factor, no throat erosion.
   Reported specific impulse is therefore an IDEAL value.
3. Ignition rise, phase transitions and tail-off are prescribed ramp times, not modelled
   transients. They are mass-conserving by construction, so total impulse is unaffected
   by them.
4. The nozzle-mass and insulation-thickness estimates are weak. See SOURCES.
5. The inter-pulse bulkhead, igniter and insulation a real commanded-ignition terminal
   pulse needs are NOT costed separately; the terminal grain is charged only the same
   igniter fraction as the rest of the motor. See SOURCES["prop.terminal_pulse"].

Units are SI throughout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from ..config import MATERIALS, DesignVector, Material, register_sources

# --------------------------------------------------------------------------------------
#   Physical constants
# --------------------------------------------------------------------------------------

G0 = 9.80665                # standard gravity, m/s^2 (CGPM 3rd conference, 1901)
R_UNIVERSAL = 8314.462618   # universal gas constant, J/(kmol.K) (SI 2019 exact values)
P_SEA_LEVEL = 101325.0      # US Standard Atmosphere 1976 sea-level pressure, Pa

PSI_TO_PA = 6894.757293168361      # exact by definition of lbf and inch
INCH_TO_M = 0.0254                 # exact
LBIN3_TO_KGM3 = 27679.904710203122  # exact: lb/in^3 -> kg/m^3

# --------------------------------------------------------------------------------------
#   Propellant performance - every number here has a source
# --------------------------------------------------------------------------------------

# Sutton and Biblarz, "Rocket Propulsion Elements", Chapter 12, Table 12-1
# "Characteristics of Some Operational Solid Propellants", row HTPB/AP/Al.
# Read directly from the chapter PDF on 2026-08-17. Table footnotes: (b) Is is "at
# 1000 psia expanding to 14.7 psia, ideal or theoretical value at reference conditions";
# (c) burning rate is "at 1000 psia".
_T12_1_IS_MIN_S = 260.0             # s
_T12_1_IS_MAX_S = 265.0             # s
_T12_1_FLAME_TEMP_K = 3440.0        # K (table also gives 5700 F)
_T12_1_DENSITY_LBIN3 = 0.067        # lb/in^3 (table also gives specific gravity 1.86)
_T12_1_BURN_RATE_MIN_IPS = 0.25     # in/s at 1000 psia
_T12_1_BURN_RATE_MAX_IPS = 3.0      # in/s at 1000 psia
_T12_1_PRESSURE_EXPONENT = 0.40     # dimensionless

# The Table 12-1 values converted to SI, for the report table. The model itself uses the
# density of MATERIALS['propellant_htpb_ap'] so config.py stays the single source of truth.
TABLE_12_1_DENSITY_KGM3 = _T12_1_DENSITY_LBIN3 * LBIN3_TO_KGM3
TABLE_12_1_BURN_RATE_BAND_MS = (
    _T12_1_BURN_RATE_MIN_IPS * INCH_TO_M,
    _T12_1_BURN_RATE_MAX_IPS * INCH_TO_M,
)

# Reference conditions of the Table 12-1 specific impulse.
P_C_REFERENCE = 1000.0 * PSI_TO_PA      # 6.8948 MPa
P_E_REFERENCE = 14.7 * PSI_TO_PA        # 0.10135 MPa, optimum expansion

ISP_REFERENCE_S = 0.5 * (_T12_1_IS_MIN_S + _T12_1_IS_MAX_S)   # 262.5 s, mid-range
FLAME_TEMPERATURE_K = _T12_1_FLAME_TEMP_K

# Ratio of specific heats of the combustion products. NOT tabulated in Table 12-1.
# Taken as 1.20, the value normally used for aluminized AP/HTPB exhaust. This is the
# single assumed thermochemical input of the model. It is cross-checked in
# `propellant_cross_check()`: with gamma = 1.20 the cited Isp and flame temperature
# imply a mean exhaust molar mass near 26 kg/kmol, which agrees with Sutton Figure 12-3
# (mean molecular mass of the combustion gases of HTPB-based composite propellant at
# 68 atm, 25 to 30 kg/kmol for 68 to 72 % AP). Sensitivity: +/- 0.02 on gamma moves the
# derived c* by about 1.5 %.
GAMMA_EXHAUST = 1.20

# Burn-rate law r = a * p_c^n, r in m/s, p_c in Pa.
# n = 0.40 is cited (Table 12-1, HTPB/AP/Al).
# The reference rate is a modelling choice INSIDE a cited range: Table 12-1 gives
# 0.25 to 3.0 in/s (6.35 to 76.2 mm/s) at 1000 psia, and the Chapter 12 text (p. 476)
# says most composite propellants have "burning rates between 7 and 20 mm/sec".
# 10.0 mm/s at 1000 psia sits inside both bands and is representative of a cast
# tactical-motor formulation.
BURN_RATE_EXPONENT_N = _T12_1_PRESSURE_EXPONENT
BURN_RATE_REFERENCE_MS = 0.0100          # m/s at P_C_REFERENCE
BURN_RATE_COEFF_A = BURN_RATE_REFERENCE_MS / P_C_REFERENCE ** BURN_RATE_EXPONENT_N

# Nozzle flow separation. Summerfield criterion: the flow separates from the wall when
# the exit static pressure falls below roughly 0.4 of ambient.
SEPARATION_PE_OVER_PA = 0.40

# Prescribed thrust-trace ramp times. These are modelling choices that make the thrust
# function continuous for the trajectory integrator. They are mass-conserving, so they
# do not change total impulse.
T_RISE_S = 0.05          # ignition rise, specified by the WP3 task statement
T_TRANSITION_S = 0.10    # boost to sustain blend
T_TAILOFF_S = 0.20       # burnout decay

# Terminal pulse default operating point: the same chamber pressure the case is already
# sized for. Nothing is invented; the fraction is 1.0 of DesignVector.p_c. Overridden by
# DesignVector.F_terminal or size_terminal_for_thrust().
TERMINAL_PC_FRACTION = 1.0

# Structural and layout choices for the inert-mass estimate.
CASE_SAFETY_FACTOR = 1.5         # on material yield at chamber pressure
CASE_MIN_GAUGE_M = 0.0010        # practical minimum wall, 1 mm
INSULATION_THICKNESS_M = 0.0030  # EPDM internal insulation
CASE_LENGTH_ALLOWANCE_M = 0.10   # closures, igniter boss, joints
NOZZLE_HALF_ANGLE_RAD = math.radians(15.0)
NOZZLE_WALL_THICKNESS_M = 0.0080
NOZZLE_MATERIAL_DENSITY = 1800.0    # bulk graphite / carbon-carbon, kg/m^3
NOZZLE_STRUCTURE_FACTOR = 1.5       # convergent section, throat insert, attach ring
IGNITER_MASS_FRACTION = 0.005        # of total propellant mass
MOTOR_MASS_FRACTION_BAND = (0.80, 0.91)   # propellant / (propellant + inert)

SOURCES: dict[str, str] = {
    "prop.isp_reference": (
        "Sutton and Biblarz, Rocket Propulsion Elements, Ch.12 Table 12-1 "
        "'Characteristics of Some Operational Solid Propellants', row HTPB/AP/Al: "
        "Is range 260 to 265 s, footnote (b) 'at 1000 psia expanding to 14.7 psia, "
        "ideal or theoretical value at reference conditions'. Mid-range 262.5 s used. "
        "Read from the chapter PDF on 2026-08-17."
    ),
    "prop.flame_temperature": (
        "Sutton and Biblarz, Rocket Propulsion Elements, Ch.12 Table 12-1, row "
        "HTPB/AP/Al: flame temperature 5700 F = 3440 K."
    ),
    "prop.density_table": (
        "Sutton and Biblarz, Rocket Propulsion Elements, Ch.12 Table 12-1, row "
        "HTPB/AP/Al: 0.067 lb/in^3, specific gravity 1.86. The model uses the density "
        "of MATERIALS['propellant_htpb_ap'] (1800 kg/m^3) so config.py stays the single "
        "source of truth; 1860 kg/m^3 is the Table 12-1 value for this exact formulation."
    ),
    "prop.gamma": (
        "ASSUMPTION (not tabulated, treat as a guess inside a physically bounded band): "
        "ratio of specific heats of the combustion products taken as 1.20, the value "
        "normally used for aluminized AP/HTPB exhaust. Cross-checked: with gamma = 1.20 "
        "the cited Isp (262.5 s) and flame temperature (3440 K) imply a mean exhaust "
        "molar mass of about 26 kg/kmol, inside the 25 to 30 kg/kmol band of Sutton "
        "Figure 12-3 for HTPB-based composite propellant at 68 atm. A +/- 0.02 change "
        "in gamma moves the derived c* by about 1.5 %."
    ),
    "prop.c_star": (
        "DERIVED, not tabulated: c* = Isp_ref * g0 / C_F_ref, where Isp_ref = 262.5 s "
        "(Sutton Table 12-1) and C_F_ref is the ideal thrust coefficient at the Table "
        "12-1 reference conditions (p_c = 1000 psia, optimum expansion to 14.7 psia) "
        "computed from the isentropic relations with gamma = 1.20."
    ),
    "prop.burn_rate_exponent": (
        "Sutton and Biblarz, Rocket Propulsion Elements, Ch.12 Table 12-1, row "
        "HTPB/AP/Al: pressure exponent n = 0.40."
    ),
    "prop.burn_rate_coefficient": (
        "MODELLING CHOICE inside a cited range: reference burn rate 10.0 mm/s at "
        "1000 psia. Sutton Table 12-1 (HTPB/AP/Al) gives 0.25 to 3.0 in/s = 6.35 to "
        "76.2 mm/s at 1000 psia, and the Ch.12 text on p.476 states most composite "
        "propellants burn at 7 to 20 mm/s. The coefficient a follows from "
        "a = r_ref / p_ref^n."
    ),
    "prop.end_burning_sustainer": (
        "J. M. Seitzman, Georgia Tech AE6450 Rocket Propulsion, 'Solid Propellants' "
        "lecture notes, slide 17: free-standing grains are 'better suited for end "
        "burning motors (used in sustain portion of some rockets)'. Retrieved "
        "2026-08-17 from seitzman.gatech.edu/classes/ae6450/solid_propellants.pdf."
    ),
    "prop.isentropic_tables": (
        "Purdue University School of Aeronautics and Astronautics, '1-D Isentropic "
        "Flow' tables, gamma = 1.2 and gamma = 1.4, "
        "engineering.purdue.edu/~propulsi/propulsion/flow/. Used as the published "
        "validation reference for the area-Mach and pressure-ratio relations "
        "(gamma 1.2: M 2.00 -> A/A* 1.884, pt/p 7.530; M 3.00 -> A/A* 6.735, "
        "pt/p 47.05; gamma 1.4: M 4.00 -> A/A* 10.72, pt/p 151.8)."
    ),
    "prop.separation_criterion": (
        "Summerfield flow-separation criterion, separation when p_e / p_a is below "
        "about 0.4 (Summerfield, Foster and Swan, 1954; the 0.4 figure is the commonly "
        "quoted approximation and was NOT verified against the primary paper in this "
        "session). Used only to raise a warning, never to change a force."
    ),
    "prop.ramp_times": (
        "MODELLING CHOICE, not physics: ignition rise 0.05 s (specified by the WP3 "
        "task statement), boost-to-sustain blend 0.10 s, tail-off 0.20 s. The ramps are "
        "mass-conserving by construction so total impulse is unchanged."
    ),
    "prop.case_safety_factor": (
        "MODELLING CHOICE: safety factor 1.5 applied to material yield strength in the "
        "thin-wall hoop-stress case sizing (sigma = p * r / t), plus a 1.0 mm minimum "
        "practical gauge. 1.5 is the conventional aerospace pressure-vessel factor; it "
        "was not taken from a specific standard in this session."
    ),
    "prop.insulation_thickness": (
        "GUESS: 3.0 mm of EPDM internal insulation on the case wall and domes. Real "
        "insulation is sized from local gas-side heat flux and exposure time, which is "
        "out of scope (SPEC.md section 8). Density from "
        "MATERIALS['insulation_epdm']."
    ),
    "prop.nozzle_mass_model": (
        "GUESS-LEVEL geometric estimate: conical nozzle, 15 deg half angle, 8.0 mm wall, "
        "bulk graphite / carbon-carbon density 1800 kg/m^3 (handbook range 1.7 to "
        "1.9 g/cm^3, not verified against a datasheet in this session), multiplied by "
        "1.5 to allow for the convergent section, throat insert and attach ring. This "
        "estimate is known to be optimistic; compare against "
        "SOURCES['prop.motor_mass_fraction']."
    ),
    "prop.igniter_mass": (
        "GUESS: igniter and safe-arm mass taken as 0.5 % of total propellant mass."
    ),
    "prop.motor_mass_fraction": (
        "CROSS-CHECK CORRELATION, quoted band only, treat the endpoints as a guess: "
        "tactical solid rocket motors have propellant mass fractions "
        "m_p / (m_p + m_inert) of roughly 0.80 to 0.91. This band was not verified "
        "against a primary source in this session; it is used only to bound the "
        "physics-route inert mass, never to replace it silently."
    ),
    "prop.terminal_pulse": (
        "MODELLING CHOICE with one guess: the terminal pulse is a third grain segment in "
        "the same case, ignited on command rather than on a timer, which is how real "
        "dual-pulse tactical motors work (an inter-pulse bulkhead separates the charges "
        "and a second igniter fires the aft one). Its default chamber pressure is 1.0 x "
        "DesignVector.p_c, that is the pressure the case is already sized for, so no new "
        "number is introduced. GUESS: the inter-pulse bulkhead, its insulation and the "
        "second igniter are NOT costed as separate inert mass; the terminal grain is "
        "charged only the same 0.5 % igniter fraction as the rest of the motor, so the "
        "reported inert mass is optimistic for a pulsed motor. Physical motivation: an "
        "unpowered terminal dive is terminal-velocity limited and cannot reach Mach 1.5 "
        "at sea level for any dive angle, so thrust in the endgame is the only way to "
        "meet SPEC R6."
    ),
    "prop.throat_transition_credibility": (
        "HARDWARE CREDIBILITY STATEMENT, not a number. The boost throat is larger than "
        "the sustain throat, so the boost-to-sustain transition needs the throat to "
        "SHRINK. No ejectable-insert mechanism does that; an ejected insert and throat "
        "erosion both only enlarge a throat. The credible hardware for a smaller "
        "effective throat after boost is a separate smaller sustainer nozzle in the same "
        "aft closure, or a tandem booster that is jettisoned, neither of which is a "
        "single-throat motor. This model does not invent a shrinking-throat mechanism: "
        "`throat_transition_report()` names every transition and its direction so a "
        "detailed design must resolve it. The terminal pulse shares the sustain throat "
        "by default precisely so that it adds no further transition."
    ),
    "prop.terminal_grain": (
        "MODELLING CHOICE: the terminal segment is an internal-burning tubular grain, "
        "the same geometry and the same mean-web closure as the boost segment, because "
        "an end burner at this bore cannot supply the burning area a multi-kilonewton "
        "terminal thrust needs at a credible chamber pressure. Its length counts towards "
        "the SPEC.md section 4 grain L/D limit of 1.0 to 8.0."
    ),
    "prop.two_position_throat": (
        "MODELLING CHOICE: the motor is given a two-position throat (ejectable boost "
        "throat insert) so the boost and sustain phases can run at different throat "
        "areas with the same nozzle area ratio. Without it, a single throat sized for "
        "the boost thrust drives the sustain chamber pressure to about 0.13 MPa and "
        "Kn to about 22, which no composite propellant will sustain. Set "
        "two_position_throat=False to model one fixed throat instead."
    ),
    "prop.neutral_burning": (
        "APPROXIMATION: neutral (constant) burning is assumed within each phase. The "
        "burn-rate law is closed at the MEAN web position, which makes propellant mass, "
        "web thickness, mass flow and burn time mutually consistent "
        "(w = V_p / A_b_mean, t_b = w / r, mdot = rho * A_b_mean * r). No burnback "
        "simulation is performed, so the reported grain dimensions are volume- and "
        "area-equivalent, not a manufacturing drawing."
    ),
    "prop.ideal_nozzle": (
        "APPROXIMATION: frozen-composition, constant-gamma, single-phase isentropic "
        "nozzle flow. No two-phase Al2O3 loss, no divergence loss, no combustion "
        "efficiency and no throat erosion. Reported Isp is therefore an ideal value; "
        "real delivered Isp for a tactical motor of this class is typically 3 to 7 % "
        "lower (magnitude of that correction NOT sourced here, so it is not applied)."
    ),
}

register_sources(SOURCES)


# --------------------------------------------------------------------------------------
#   Isentropic nozzle relations
# --------------------------------------------------------------------------------------


def vandenkerckhove(gamma: float) -> float:
    """The Vandenkerckhove function Gamma(gamma).

    Gamma = sqrt(gamma) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))

    It is the dimensionless choked-mass-flow coefficient:
    mdot = p_c * A_t * Gamma / sqrt(R_specific * T_c).
    """
    return math.sqrt(gamma) * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


def area_ratio_from_mach(mach: float, gamma: float) -> float:
    """Isentropic area ratio A / A_throat for a given Mach number."""
    if mach <= 0.0:
        raise ValueError("mach must be positive")
    term = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * mach * mach)
    return (1.0 / mach) * term ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


def total_over_static_pressure(mach: float, gamma: float) -> float:
    """Isentropic stagnation-to-static pressure ratio p_0 / p."""
    return (1.0 + 0.5 * (gamma - 1.0) * mach * mach) ** (gamma / (gamma - 1.0))


def mach_from_pressure_ratio(p0_over_p: float, gamma: float) -> float:
    """Invert the isentropic pressure ratio for Mach number."""
    if p0_over_p < 1.0:
        raise ValueError("p0/p must be at least 1")
    return math.sqrt(2.0 / (gamma - 1.0) * (p0_over_p ** ((gamma - 1.0) / gamma) - 1.0))


def mach_from_area_ratio(area_ratio: float, gamma: float, supersonic: bool = True) -> float:
    """Invert the isentropic area relation by bisection.

    `supersonic=True` returns the branch with M > 1, which is the nozzle exit branch.
    Bisection is used rather than Newton because the relation is flat near M = 1 and
    bisection cannot leave the bracket. 200 iterations is machine precision.
    """
    if area_ratio < 1.0:
        raise ValueError("area ratio must be at least 1")
    if abs(area_ratio - 1.0) < 1e-14:
        return 1.0
    if supersonic:
        lo, hi = 1.0 + 1e-12, 2.0
        while area_ratio_from_mach(hi, gamma) < area_ratio:
            hi *= 2.0
            if hi > 1.0e6:
                raise ValueError("area ratio too large to bracket")
    else:
        lo, hi = 1.0e-9, 1.0 - 1e-12
    # On the supersonic branch A/A* increases with M, on the subsonic branch it
    # decreases, so the bracket update differs.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        below = area_ratio_from_mach(mid, gamma) < area_ratio
        if below == supersonic:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def exit_pressure_ratio(area_ratio: float, gamma: float) -> float:
    """Exit static over chamber total pressure, p_e / p_c, for a given area ratio."""
    mach_e = mach_from_area_ratio(area_ratio, gamma, supersonic=True)
    return 1.0 / total_over_static_pressure(mach_e, gamma)


def thrust_coefficient(
    gamma: float,
    area_ratio: float,
    pa_over_pc: float = 0.0,
) -> float:
    """Ideal thrust coefficient C_F = F / (p_c * A_t).

    C_F = Gamma * sqrt(2 * gamma / (gamma - 1) * (1 - (p_e/p_c) ** ((gamma-1)/gamma)))
          + (p_e/p_c - p_a/p_c) * eps

    The first term is the momentum thrust, the second the pressure thrust. Setting
    `pa_over_pc = 0` gives the vacuum thrust coefficient. The relation is validated in
    tests/test_propulsion.py against the published isentropic tables
    (SOURCES["prop.isentropic_tables"]) and against an independent momentum-integral
    evaluation.
    """
    pe_over_pc = exit_pressure_ratio(area_ratio, gamma)
    momentum = vandenkerckhove(gamma) * math.sqrt(
        2.0 * gamma / (gamma - 1.0) * (1.0 - pe_over_pc ** ((gamma - 1.0) / gamma))
    )
    return momentum + (pe_over_pc - pa_over_pc) * area_ratio


def reference_thrust_coefficient(gamma: float = GAMMA_EXHAUST) -> float:
    """C_F at the Sutton Table 12-1 reference conditions (1000 psia, optimum to 14.7 psia)."""
    area_ratio = area_ratio_from_mach(
        mach_from_pressure_ratio(P_C_REFERENCE / P_E_REFERENCE, gamma), gamma
    )
    return thrust_coefficient(gamma, area_ratio, P_E_REFERENCE / P_C_REFERENCE)


def characteristic_velocity(gamma: float = GAMMA_EXHAUST) -> float:
    """Characteristic velocity c*, m/s, derived from the cited reference Isp.

    c* = Isp_ref * g0 / C_F_ref. See SOURCES["prop.c_star"].
    """
    return ISP_REFERENCE_S * G0 / reference_thrust_coefficient(gamma)


def propellant_cross_check(gamma: float = GAMMA_EXHAUST) -> dict[str, float]:
    """Cross-check the derived c* against the cited flame temperature.

    From c* = sqrt(R_specific * T_c) / Gamma(gamma), the implied mean molar mass of the
    exhaust is M = R_universal * T_c / (c* * Gamma) ** 2. Sutton Figure 12-3 puts the
    mean molecular mass of HTPB-based composite exhaust at 68 atm in the 25 to
    30 kg/kmol band, so a value in that band confirms the assumed gamma.
    """
    c_star = characteristic_velocity(gamma)
    molar_mass = R_UNIVERSAL * FLAME_TEMPERATURE_K / (c_star * vandenkerckhove(gamma)) ** 2
    return {
        "c_star": c_star,
        "C_F_reference": reference_thrust_coefficient(gamma),
        "implied_molar_mass": molar_mass,
        "flame_temperature": FLAME_TEMPERATURE_K,
        "gamma": gamma,
    }


def burn_rate(p_c: float) -> float:
    """Propellant linear regression rate, m/s, from r = a * p_c ** n."""
    return BURN_RATE_COEFF_A * p_c ** BURN_RATE_EXPONENT_N


def chamber_pressure_from_kn(
    burning_area: float,
    throat_area: float,
    density: float,
    c_star: float,
) -> float:
    """Equilibrium chamber pressure from the Kn closure.

    Mass balance: rho_p * A_b * a * p_c ** n = p_c * A_t / c*, so
    p_c = (rho_p * a * c* * A_b / A_t) ** (1 / (1 - n)).
    """
    kn = burning_area / throat_area
    return (density * BURN_RATE_COEFF_A * c_star * kn) ** (1.0 / (1.0 - BURN_RATE_EXPONENT_N))


def burning_area_from_chamber_pressure(
    p_c: float,
    throat_area: float,
    density: float,
    c_star: float,
) -> float:
    """Inverse of `chamber_pressure_from_kn`: burning area needed to hold p_c."""
    return p_c ** (1.0 - BURN_RATE_EXPONENT_N) * throat_area / (
        density * BURN_RATE_COEFF_A * c_star
    )


# --------------------------------------------------------------------------------------
#   Grain geometry
# --------------------------------------------------------------------------------------


@dataclass
class GrainGeometry:
    """Volume- and area-equivalent multi-thrust grain, sized inside the motor bay.

    All lengths in metres, areas in m^2, volumes in m^3.

    The boost segment is an internal-burning tube: outer diameter `d_outer`, initial
    port diameter `d_inner_boost`, web `web_boost`, length `length_boost`.
    The sustain segment is an end burner: face diameter `d_face_sustain`, web equal to
    its length `length_sustain`.
    The terminal segment is an internal-burning tube like the boost segment. All of its
    fields are zero when `DesignVector.m_p_terminal` is zero, so `length_total` and
    `L_over_D` are then exactly what the two-phase model reported.
    """

    d_outer: float
    d_inner_boost: float
    web_boost: float
    length_boost: float
    burning_area_boost: float

    d_face_sustain: float
    web_sustain: float
    length_sustain: float
    burning_area_sustain: float

    d_inner_terminal: float
    web_terminal: float
    length_terminal: float
    burning_area_terminal: float

    length_total: float
    L_over_D: float

    volume_boost: float
    volume_sustain: float
    volume_terminal: float
    volume_total: float

    bay_length_available: float
    bay_diameter: float
    volumetric_loading: float

    feasible: bool
    warnings: list[str] = field(default_factory=list)

    # Aliases requested by the WP3 contract, for the overall grain assembly.
    @property
    def d_inner(self) -> float:
        """Representative inner (port) diameter: the boost port."""
        return self.d_inner_boost

    @property
    def length(self) -> float:
        return self.length_total

    @property
    def web(self) -> float:
        """Governing web thickness: the largest segment web."""
        return max(self.web_boost, self.web_sustain, self.web_terminal)


# --------------------------------------------------------------------------------------
#   The motor
# --------------------------------------------------------------------------------------


@dataclass
class _PhaseDesign:
    """Internal per-phase operating point."""

    name: str
    propellant_mass: float
    p_c: float
    throat_area: float
    exit_area: float
    burning_area: float
    burn_rate: float
    mdot: float
    burn_time: float
    thrust_vacuum: float
    C_F_vacuum: float
    isp_vacuum: float
    kn: float


#: Public alias. A "phase" of the SV-1 dual-thrust motor and a "stage" of the IV-1 stack are
#: the same object at this fidelity: one propellant charge burning at one chamber pressure
#: through one throat. `propulsion_iv1.py` builds these for its stages.
PhaseDesign = _PhaseDesign


# --------------------------------------------------------------------------------------
#   Shared building blocks
# --------------------------------------------------------------------------------------
#
# Everything below is used by `SolidMotor` AND by `propulsion_iv1.MultiStageMotor`. It lives
# here, at module level, so the two motor models cannot drift apart: there is exactly one
# implementation of the nozzle sizing, the mean-web grain closure, the ramp shape, the case
# and nozzle mass estimates and the Summerfield check. If you change one of these you change
# both motors, which is the intent. The expressions are lifted verbatim from the methods that
# used to hold them, so the numbers are bit-identical to the validated SV-1 result.


def ramp(t: float, t0: float, width: float) -> float:
    """Linear 0 to 1 ramp starting at t0 over `width` seconds, clamped.

    This is the only shape primitive in either motor model. Ignition rise, phase blend,
    stage transition and tail-off are all built from it, which is what keeps the thrust
    trace continuous for the RK4 integrator. See SOURCES["prop.ramp_times"].
    """
    if width <= 0.0:
        return 1.0 if t >= t0 else 0.0
    return min(1.0, max(0.0, (t - t0) / width))


def throat_area_for_thrust(
    thrust: float,
    p_c: float,
    C_F_vacuum: float,
    area_ratio: float,
    ambient_pressure: float,
) -> float:
    """Throat area that delivers `thrust` at `ambient_pressure` and `p_c`, m^2.

    F = (C_F_vac - eps * p_a / p_c) * p_c * A_t, which is the momentum thrust plus the
    ambient term (p_e - p_a) * A_e written in coefficient form. Setting
    `ambient_pressure = 0` sizes the nozzle in vacuum; passing P_SEA_LEVEL sizes it at sea
    level. Raises when the nozzle is so overexpanded at the sizing pressure that no throat
    area can deliver a positive thrust.
    """
    c_f_sizing = C_F_vacuum - ambient_pressure * area_ratio / p_c
    if c_f_sizing <= 0.0:
        raise ValueError(
            "nozzle is so overexpanded that the sizing thrust coefficient is not "
            "positive; reduce eps_nozzle or raise p_c"
        )
    return thrust / (c_f_sizing * p_c)


def design_phase(
    name: str,
    propellant_mass: float,
    p_c: float,
    throat_area: float,
    area_ratio: float,
    c_star: float,
    C_F_vacuum: float,
    density: float,
) -> _PhaseDesign:
    """Close one burning phase (or one stage) at a fixed chamber pressure and throat.

    The closure is: choked mass flow from c*, burning area from the burn-rate law through
    the Kn balance, burn time from the propellant mass, vacuum thrust from C_F. Every
    quantity follows from the four inputs; nothing here is free.
    """
    exit_area = area_ratio * throat_area
    r = burn_rate(p_c)
    mdot = p_c * throat_area / c_star
    burning_area = mdot / (density * r)
    burn_time = propellant_mass / mdot if mdot > 0.0 else 0.0
    thrust_vac = C_F_vacuum * p_c * throat_area
    return _PhaseDesign(
        name=name,
        propellant_mass=propellant_mass,
        p_c=p_c,
        throat_area=throat_area,
        exit_area=exit_area,
        burning_area=burning_area,
        burn_rate=r,
        mdot=mdot,
        burn_time=burn_time,
        thrust_vacuum=thrust_vac,
        C_F_vacuum=C_F_vacuum,
        isp_vacuum=C_F_vacuum * c_star / G0,
        kn=burning_area / throat_area,
    )


@dataclass
class TubularGrain:
    """Mean-web closure of one internal-burning tubular segment.

    For a tube burning at its MEAN web position the closure is exact:
        A_b = pi * L * (d_o + d_i) / 2,   V_p = A_b * w,   w = (d_o - d_i) / 2
    so w = V_p / A_b and L = 2 * A_b / (pi * (d_o + d_i)). See
    SOURCES["prop.neutral_burning"].

    `fits` is False when the web is thicker than the bay radius, that is when the segment
    needs more propellant than a tube of this burning area can hold. `d_inner` is then
    clamped to a positive sliver so the length stays finite and the caller can report the
    overrun instead of dividing by zero.
    """

    volume: float
    web: float
    d_inner: float
    length: float
    fits: bool


def tubular_grain_closure(
    propellant_mass: float, density: float, burning_area: float, d_outer: float
) -> TubularGrain:
    """Close an internal-burning tubular segment inside a bay of diameter `d_outer`."""
    volume = propellant_mass / density
    web = volume / burning_area
    d_inner = d_outer - 2.0 * web
    fits = d_inner > 0.0
    if not fits:
        d_inner = 1.0e-6
    length = 2.0 * burning_area / (math.pi * (d_outer + d_inner))
    return TubularGrain(
        volume=volume, web=web, d_inner=d_inner, length=length, fits=fits
    )


def motor_case_and_insulation(
    case_inner_radius: float,
    case_length: float,
    p_design: float,
    case_material: Material,
    insulation_material: Material,
    insulation_thickness: float,
) -> dict[str, float]:
    """Case and insulation mass from thin-wall hoop stress, kg.

    t_cyl = p_design * r / sigma_yield, floored at the minimum practical gauge. The two
    closures are membrane spheres, so they carry p*r/(2t) and are half as thick. Insulation
    is a constant-thickness liner over the same wetted area. `p_design` must already carry
    the safety factor: pass CASE_SAFETY_FACTOR * max(p_c over all phases or stages), so the
    case pays for the highest pressure it ever contains.
    """
    t_cyl = max(CASE_MIN_GAUGE_M, p_design * case_inner_radius / case_material.sigma_yield)
    t_dome = 0.5 * t_cyl   # membrane sphere stress is p*r/(2t)

    area_cyl = 2.0 * math.pi * case_inner_radius * case_length
    area_domes = 2.0 * (2.0 * math.pi * case_inner_radius ** 2)
    m_case = case_material.density * (area_cyl * t_cyl + area_domes * t_dome)
    m_insulation = insulation_material.density * (
        (area_cyl + area_domes) * insulation_thickness
    )
    return {
        "case": m_case,
        "insulation": m_insulation,
        "case_thickness": t_cyl,
        "dome_thickness": t_dome,
        "area_cyl": area_cyl,
        "area_domes": area_domes,
    }


def conical_nozzle_mass(throat_area: float, exit_area: float) -> float:
    """Geometric conical-nozzle mass estimate, kg. See SOURCES['prop.nozzle_mass_model'].

    This is the weakest number in either inert-mass model. It is a lateral shell area times
    a wall thickness times a structure factor, and it is known to be optimistic.
    """
    r_t = math.sqrt(throat_area / math.pi)
    r_e = math.sqrt(exit_area / math.pi)
    axial = (r_e - r_t) / math.tan(NOZZLE_HALF_ANGLE_RAD)
    slant = math.hypot(axial, r_e - r_t)
    lateral = math.pi * (r_t + r_e) * slant
    return (
        NOZZLE_MATERIAL_DENSITY * lateral * NOZZLE_WALL_THICKNESS_M * NOZZLE_STRUCTURE_FACTOR
    )


def correlation_inert_band(propellant_mass: float) -> tuple[float, float]:
    """Inert mass implied by the tactical-motor propellant-mass-fraction band, kg.

    zeta = m_p / (m_p + inert) so inert = m_p * (1 - zeta) / zeta. Returns
    (optimistic, pessimistic) = (from the high zeta, from the low zeta). See
    SOURCES["prop.motor_mass_fraction"]: the band endpoints are a quoted correlation, not a
    measurement, and the band exists to bound the bottom-up sum, never to replace it
    silently.
    """
    zeta_lo, zeta_hi = MOTOR_MASS_FRACTION_BAND
    inert_from_zeta_hi = propellant_mass * (1.0 - zeta_hi) / zeta_hi   # optimistic bound
    inert_from_zeta_lo = propellant_mass * (1.0 - zeta_lo) / zeta_lo   # pessimistic bound
    return inert_from_zeta_hi, inert_from_zeta_lo


def summerfield_separation(p_e: float) -> dict[str, float]:
    """Summerfield flow-separation assessment for one nozzle.

    Returns the exit static pressure, the ambient pressure at which separation starts
    (p_e / 0.40) and the altitude below which the flow would separate. Separation is
    reported, never applied to the thrust. See SOURCES["prop.separation_criterion"].
    """
    p_a_sep = p_e / SEPARATION_PE_OVER_PA
    return {
        "p_e": p_e,
        "p_a_separation": p_a_sep,
        "separation_altitude": _altitude_for_pressure(p_a_sep),
    }


class SolidMotor:
    """Boost-sustain solid rocket motor for the SV-1 demo rocket.

    The boost phase is sized from `dv.F_boost` at `sizing_altitude_pressure` (sea level
    by default, matching the SPEC.md section 3 definition of F_boost as a sea-level
    equivalent thrust) at the chamber pressure `dv.p_c`. The sustain phase defaults to a
    chamber pressure of `sustain_pc_fraction * dv.p_c` and should be re-sized by the
    sizing loop with `size_sustain_for_thrust(cruise_drag)`.

    Read the module docstring for the approximations. In particular, thrust is neutral
    within each phase and the nozzle flow is ideal.
    """

    def __init__(
        self,
        dv: DesignVector,
        propellant: Material = MATERIALS["propellant_htpb_ap"],
        gamma: float = GAMMA_EXHAUST,
        sizing_ambient_pressure: float = P_SEA_LEVEL,
        sustain_pc_fraction: float = 0.25,
        two_position_throat: bool = True,
        insulation_thickness: float = INSULATION_THICKNESS_M,
        case_material: Material = MATERIALS["motorcase_cfrp"],
        insulation_material: Material = MATERIALS["insulation_epdm"],
        terminal_throat_source: str = "sustain",
        terminal_propellant_mass: float | None = None,
        terminal_thrust: float | None = None,
        terminal_sizing_ambient_pressure: float = P_SEA_LEVEL,
    ) -> None:
        if terminal_throat_source not in ("sustain", "boost"):
            raise ValueError("terminal_throat_source must be 'sustain' or 'boost'")
        self.dv = dv
        self.propellant = propellant
        self.gamma = gamma
        self.sizing_ambient_pressure = sizing_ambient_pressure
        self.two_position_throat = two_position_throat
        self.insulation_thickness = insulation_thickness
        self.case_material = case_material
        self.insulation_material = insulation_material
        self.terminal_throat_source = terminal_throat_source
        self.terminal_sizing_ambient_pressure = terminal_sizing_ambient_pressure
        self.m_p_terminal = (
            dv.m_p_terminal if terminal_propellant_mass is None else terminal_propellant_mass
        )

        self._sizing_warnings: list[str] = []

        self.c_star = characteristic_velocity(gamma)
        self.area_ratio = dv.eps_nozzle
        self.pe_over_pc = exit_pressure_ratio(self.area_ratio, gamma)
        self.C_F_vacuum = thrust_coefficient(gamma, self.area_ratio, 0.0)

        # --- boost: chamber pressure fixed by the design vector, throat from thrust ---
        p_c_boost = dv.p_c
        throat_area_boost = throat_area_for_thrust(
            dv.F_boost,
            p_c_boost,
            self.C_F_vacuum,
            self.area_ratio,
            sizing_ambient_pressure,
        )
        self._boost = self._make_phase(
            "boost", dv.m_p_boost, p_c_boost, throat_area_boost
        )

        # --- sustain: default operating point, re-sizable ---
        self._sustain_pc_fraction = sustain_pc_fraction
        self._sustain = self._size_sustain_at_pressure(sustain_pc_fraction * dv.p_c)

        # --- terminal: commanded ignition, no timer. Inert when the mass is zero. ---
        self._terminal_armed = False
        self._terminal_ignition_time: float | None = None
        self._terminal = self._size_terminal_at_pressure(TERMINAL_PC_FRACTION * dv.p_c)
        requested_terminal_thrust = (
            dv.F_terminal if terminal_thrust is None else terminal_thrust
        )
        if self.has_terminal and requested_terminal_thrust > 0.0:
            self.size_terminal_for_thrust(requested_terminal_thrust)

        self._update_timeline()

    # ---------------------------------------------------------------- construction ---

    def _make_phase(
        self, name: str, propellant_mass: float, p_c: float, throat_area: float
    ) -> _PhaseDesign:
        """One burning phase, closed by the shared `design_phase` free function."""
        return design_phase(
            name,
            propellant_mass,
            p_c,
            throat_area,
            area_ratio=self.area_ratio,
            c_star=self.c_star,
            C_F_vacuum=self.C_F_vacuum,
            density=self.propellant.density,
        )

    def _size_sustain_at_pressure(self, p_c: float) -> _PhaseDesign:
        """Sustain operating point at a given chamber pressure.

        With a two-position throat the sustain burning area is the full-bore end-burning
        face (fixed by the motor bay diameter) and the throat follows from the Kn
        closure. With a single throat the throat is fixed and the burning area follows.
        """
        rho = self.propellant.density
        if self.two_position_throat:
            burning_area = 0.25 * math.pi * self.bay_diameter ** 2
            throat_area = burning_area_to_throat_area(
                burning_area, p_c, rho, self.c_star
            )
        else:
            throat_area = self._boost.throat_area
        return self._make_phase("sustain", self.dv.m_p_sustain, p_c, throat_area)

    def _terminal_throat_area(self) -> float:
        """Throat area the terminal pulse runs through. No new hardware by default."""
        if self.terminal_throat_source == "boost":
            return self._boost.throat_area
        return self._sustain.throat_area

    def _size_terminal_at_pressure(self, p_c: float) -> _PhaseDesign:
        """Terminal operating point at a given chamber pressure through a fixed throat.

        Unlike the sustain phase, the throat is fixed (it is shared) and the burning area
        follows from the Kn closure inside `_make_phase`.
        """
        return self._make_phase(
            "terminal", self.m_p_terminal, p_c, self._terminal_throat_area()
        )

    @property
    def has_terminal(self) -> bool:
        """True when a terminal pulse exists at all."""
        return self.m_p_terminal > 0.0

    def _update_timeline(self) -> None:
        """Recompute the mass-conserving ramp timeline.

        `t_burnout_sustain` is the end of the sustain tail-off, which is what the mission
        waits for before it may enter the dive. `t_burnout` is when ALL thrust has
        finished, including a commanded terminal pulse. With no terminal propellant the
        two are identical and equal to the two-phase value.
        """
        t_b = self._boost.burn_time
        t_s = self._sustain.burn_time
        # Plateau end times chosen so that the integral of each normalised shape equals
        # the ideal burn time exactly. See the module docstring, item 3.
        self.t_boost_end = max(0.0, t_b + 0.5 * T_RISE_S - 0.5 * T_TRANSITION_S)
        self.t_sustain_end = max(
            self.t_boost_end,
            self.t_boost_end + t_s + 0.5 * T_TRANSITION_S - 0.5 * T_TAILOFF_S,
        )
        self.t_burnout_sustain = self.t_sustain_end + T_TAILOFF_S

        if self._terminal_ignition_time is None or not self.has_terminal:
            self.t_terminal_end = float("nan")
            self.t_burnout = self.t_burnout_sustain
            return
        t_ignition = self._terminal_ignition_time
        self.t_terminal_end = (
            t_ignition + self._terminal.burn_time + 0.5 * T_RISE_S - 0.5 * T_TAILOFF_S
        )
        self.t_burnout = max(
            self.t_burnout_sustain, self.t_terminal_end + T_TAILOFF_S
        )

    def separation_check(self) -> dict[str, dict[str, float]]:
        """Per-phase Summerfield flow-separation assessment.

        Returns, for each phase, the exit static pressure, the ambient pressure at which
        separation starts (p_e / 0.40) and the altitude below which the flow would
        separate. Separation is reported, never applied to the thrust.
        """
        out: dict[str, dict[str, float]] = {}
        phases = [self._boost, self._sustain]
        if self.has_terminal:
            phases.append(self._terminal)
        for phase in phases:
            out[phase.name] = summerfield_separation(self.pe_over_pc * phase.p_c)
        return out

    @property
    def warnings(self) -> list[str]:
        """All motor warnings, recomputed from the current operating point."""
        out = list(self._sizing_warnings)
        for name, sep in self.separation_check().items():
            if sep["separation_altitude"] > 0.0:
                out.append(
                    f"{name} nozzle: exit pressure {sep['p_e'] / 1e3:.1f} kPa; the "
                    f"Summerfield criterion predicts flow separation below "
                    f"{sep['separation_altitude']:.0f} m altitude. Reported only, the "
                    "thrust model does not change."
                )
        if not self.two_position_throat:
            out.append(
                "single fixed throat: the sustain thrust can only be reached by dropping "
                "the chamber pressure, which drives the required burning area past what "
                "the motor bay can hold. Check grain_geometry().feasible."
            )
        if self._sustain.p_c < 1.0e6:
            out.append(
                f"sustain chamber pressure {self._sustain.p_c / 1e6:.3f} MPa is below "
                "1 MPa; composite propellant combustion is normally unreliable there."
            )
        if self.has_terminal and self._terminal.p_c < 1.0e6:
            out.append(
                f"terminal chamber pressure {self._terminal.p_c / 1e6:.3f} MPa is below "
                "1 MPa; composite propellant combustion is normally unreliable there."
            )
        for transition in self.throat_transition_report():
            if not transition["credible"]:
                out.append(
                    f"throat area {transition['direction']}s from "
                    f"{transition['from']} ({float(transition['area_from']) * 1e6:.0f} mm^2) "
                    f"to {transition['to']} "
                    f"({float(transition['area_to']) * 1e6:.0f} mm^2): "
                    f"{transition['mechanism']}"
                )
        return out

    # ------------------------------------------------------------------- geometry ---

    @property
    def bay_length(self) -> float:
        """Axial length available for the grain, m."""
        dv = self.dv
        return dv.L_total - dv.L_seeker - dv.L_guidance - dv.L_warhead - dv.L_boattail

    @property
    def bay_diameter(self) -> float:
        """Internal diameter available for the grain, m."""
        dv = self.dv
        return dv.D - 2.0 * dv.t_wall - 2.0 * self.insulation_thickness

    def grain_geometry(self) -> GrainGeometry:
        """Close the dual-thrust grain geometry inside the motor bay.

        For a tubular segment burning at its MEAN web position the closure is exact:
            A_b = pi * L * (d_o + d_i) / 2,   V_p = A_b * w,   w = (d_o - d_i) / 2
        so w = V_p / A_b and L = 2 * A_b / (pi * (d_o + d_i)). See
        SOURCES["prop.neutral_burning"].

        For the end-burning sustain segment the face area is A_b and the web is the
        segment length, so L = V_p / A_b.
        """
        rho = self.propellant.density
        d_o = self.bay_diameter
        warnings: list[str] = []
        feasible = True

        # --- boost: internal-burning tube ---
        a_boost = self._boost.burning_area
        tube_boost = tubular_grain_closure(
            self._boost.propellant_mass, rho, a_boost, d_o
        )
        v_boost = tube_boost.volume
        web_boost = tube_boost.web
        d_i_boost = tube_boost.d_inner
        length_boost = tube_boost.length
        if not tube_boost.fits:
            feasible = False
            warnings.append(
                f"boost web {web_boost * 1e3:.0f} mm exceeds the bay radius "
                f"{d_o * 500.0:.0f} mm; a tubular boost grain cannot hold "
                f"{self._boost.propellant_mass:.0f} kg at a burning area of "
                f"{a_boost:.3f} m^2"
            )

        # --- sustain: end burner ---
        v_sustain = self._sustain.propellant_mass / rho
        a_sustain = self._sustain.burning_area
        d_face = math.sqrt(4.0 * a_sustain / math.pi)
        if d_face > d_o + 1e-12:
            feasible = False
            warnings.append(
                f"sustain end-burning face diameter {d_face * 1e3:.0f} mm exceeds the "
                f"bay diameter {d_o * 1e3:.0f} mm; the requested sustain thrust needs "
                f"more burning area than an end burner of this bore can provide"
            )
        length_sustain = v_sustain / a_sustain
        web_sustain = length_sustain

        # --- terminal: internal-burning tube, same closure as the boost segment ---
        if self.has_terminal:
            a_terminal = self._terminal.burning_area
            tube_terminal = tubular_grain_closure(
                self._terminal.propellant_mass, rho, a_terminal, d_o
            )
            v_terminal = tube_terminal.volume
            web_terminal = tube_terminal.web
            d_i_terminal = tube_terminal.d_inner
            length_terminal = tube_terminal.length
            if not tube_terminal.fits:
                feasible = False
                warnings.append(
                    f"terminal web {web_terminal * 1e3:.0f} mm exceeds the bay radius "
                    f"{d_o * 500.0:.0f} mm; a tubular terminal grain cannot hold "
                    f"{self._terminal.propellant_mass:.0f} kg at a burning area of "
                    f"{a_terminal:.3f} m^2. Either raise the terminal thrust (which "
                    "raises the burning area) or reduce m_p_terminal"
                )
        else:
            v_terminal = 0.0
            a_terminal = 0.0
            web_terminal = 0.0
            d_i_terminal = 0.0
            length_terminal = 0.0

        length_total = length_boost + length_sustain + length_terminal
        if length_total > self.bay_length:
            feasible = False
            warnings.append(
                f"grain length {length_total:.3f} m exceeds the available motor bay "
                f"{self.bay_length:.3f} m by {length_total - self.bay_length:.3f} m"
            )

        volume_total = v_boost + v_sustain + v_terminal
        bay_volume = 0.25 * math.pi * d_o ** 2 * self.bay_length
        loading = volume_total / bay_volume if bay_volume > 0.0 else float("inf")
        if loading > 1.0:
            feasible = False
            warnings.append(
                f"volumetric loading {loading * 100.0:.1f} % exceeds 100 %"
            )

        l_over_d = length_total / d_o if d_o > 0.0 else float("inf")
        if not (1.0 <= l_over_d <= 8.0):
            warnings.append(
                f"grain L/D {l_over_d:.2f} is outside the SPEC.md section 4 range "
                "1.0 to 8.0"
            )

        return GrainGeometry(
            d_outer=d_o,
            d_inner_boost=d_i_boost,
            web_boost=web_boost,
            length_boost=length_boost,
            burning_area_boost=a_boost,
            d_face_sustain=d_face,
            web_sustain=web_sustain,
            length_sustain=length_sustain,
            burning_area_sustain=a_sustain,
            d_inner_terminal=d_i_terminal,
            web_terminal=web_terminal,
            length_terminal=length_terminal,
            burning_area_terminal=a_terminal,
            length_total=length_total,
            L_over_D=l_over_d,
            volume_boost=v_boost,
            volume_sustain=v_sustain,
            volume_terminal=v_terminal,
            volume_total=volume_total,
            bay_length_available=self.bay_length,
            bay_diameter=d_o,
            volumetric_loading=loading,
            feasible=feasible,
            warnings=warnings,
        )

    # ----------------------------------------------------------------- sizing api ---

    def size_sustain_for_thrust(
        self, thrust_required: float, ambient_pressure: float | None = None
    ) -> float:
        """Set the sustain operating point to deliver `thrust_required`.

        `ambient_pressure` defaults to the US-1976 pressure at the cruise altitude of
        SPEC.md R2 (12 000 m), because the sustain phase is flown there. Returns the
        thrust actually achieved, which differs from the request only when the motor
        cannot supply it (single-throat mode, or a request outside the bracket). In that
        case a warning is appended to `self.warnings`.
        """
        if ambient_pressure is None:
            ambient_pressure = _us1976_pressure(12_000.0)
        if thrust_required <= 0.0:
            raise ValueError("thrust_required must be positive")

        def achieved(p_c: float) -> float:
            phase = self._size_sustain_at_pressure(p_c)
            return phase.C_F_vacuum * p_c * phase.throat_area - ambient_pressure * phase.exit_area

        lo, hi = 1.0e4, 5.0e7
        f_lo, f_hi = achieved(lo), achieved(hi)
        if not (f_lo <= thrust_required <= f_hi):
            best = lo if abs(f_lo - thrust_required) < abs(f_hi - thrust_required) else hi
            self._sustain = self._size_sustain_at_pressure(best)
            self._update_timeline()
            self._sizing_warnings.append(
                f"sustain thrust request {thrust_required:.0f} N is outside what the "
                f"motor can deliver ({f_lo:.0f} to {f_hi:.0f} N at "
                f"p_a = {ambient_pressure / 1e3:.1f} kPa); clamped to "
                f"{achieved(best):.0f} N"
            )
            return achieved(best)

        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if achieved(mid) < thrust_required:
                lo = mid
            else:
                hi = mid
        p_c = 0.5 * (lo + hi)
        self._sustain = self._size_sustain_at_pressure(p_c)
        self._update_timeline()
        return achieved(p_c)

    # ------------------------------------------------------------------- thrust ---

    # Linear 0 to 1 ramp: bound DIRECTLY to the shared `ramp` free function, not wrapped in a
    # forwarding method. `_shapes` calls it six times per thrust evaluation and a mission calls
    # `thrust` tens of thousands of times, so an extra Python frame here is not free: a
    # forwarding wrapper measured about 7 percent slower on `SolidMotor.thrust`, against the
    # 2 s wall-clock budget in tests/test_trajectory.py. Binding the alias keeps exactly the
    # frame count of the inlined staticmethod this replaced, so the refactor is
    # performance-neutral for SV-1 by construction rather than by luck.
    _ramp = staticmethod(ramp)

    def _shapes(self, t: float) -> tuple[float, float, float]:
        """Normalised boost, sustain and terminal mass-flow shapes at motor time t.

        The terminal shape is identically zero unless a terminal pulse exists and has
        been given an ignition time, so the two-phase behaviour is untouched.
        """
        if t <= 0.0 or t >= self.t_burnout:
            return 0.0, 0.0, 0.0
        s_boost = self._ramp(t, 0.0, T_RISE_S) * (
            1.0 - self._ramp(t, self.t_boost_end, T_TRANSITION_S)
        )
        s_sustain = self._ramp(t, self.t_boost_end, T_TRANSITION_S) * (
            1.0 - self._ramp(t, self.t_sustain_end, T_TAILOFF_S)
        )
        s_terminal = 0.0
        if self.has_terminal and self._terminal_ignition_time is not None:
            s_terminal = self._ramp(t, self._terminal_ignition_time, T_RISE_S) * (
                1.0 - self._ramp(t, self.t_terminal_end, T_TAILOFF_S)
            )
        return s_boost, s_sustain, s_terminal

    def mdot(self, t: float) -> float:
        """Propellant mass flow, kg/s, at motor time t (t = 0 at boost ignition)."""
        s_b, s_s, s_t = self._shapes(t)
        return (
            s_b * self._boost.mdot
            + s_s * self._sustain.mdot
            + s_t * self._terminal.mdot
        )

    def thrust(self, t: float, altitude: float) -> float:
        """Thrust, N, at motor time t and geometric altitude.

        F = sum_phase s_phase * C_F_vac * p_c_phase * A_t_phase - p_a * A_e_eff

        The ambient term uses the exit area of whichever phase is flowing, weighted by
        the same shape functions, so thrust is continuous through the transitions and the
        sea-level to vacuum difference is exactly p_a * A_e.
        """
        s_b, s_s, s_t = self._shapes(t)
        if s_b <= 0.0 and s_s <= 0.0 and s_t <= 0.0:
            return 0.0
        p_a = _ambient_pressure(altitude)
        vacuum = (
            s_b * self._boost.thrust_vacuum
            + s_s * self._sustain.thrust_vacuum
            + s_t * self._terminal.thrust_vacuum
        )
        exit_area = (
            s_b * self._boost.exit_area
            + s_s * self._sustain.exit_area
            + s_t * self._terminal.exit_area
        )
        return vacuum - p_a * exit_area

    def exit_area_at(self, t: float) -> float:
        """Effective nozzle exit area flowing at motor time t, m^2."""
        s_b, s_s, s_t = self._shapes(t)
        return (
            s_b * self._boost.exit_area
            + s_s * self._sustain.exit_area
            + s_t * self._terminal.exit_area
        )

    def phase(self, t: float) -> str:
        """Motor phase label at motor time t: 'boost', 'sustain', 'terminal' or 'burnout'."""
        if self.has_terminal and self._terminal_ignition_time is not None:
            if self._terminal_ignition_time <= t < self.t_burnout:
                return "terminal"
        if t >= self.t_burnout_sustain:
            return "burnout"
        if t < self.t_boost_end:
            return "boost"
        return "sustain"

    @property
    def t_boost(self) -> float:
        """Ideal boost burn time, s, equal to m_p_boost / mdot_boost."""
        return self._boost.burn_time

    @property
    def t_sustain(self) -> float:
        """Ideal sustain burn time, s, equal to m_p_sustain / mdot_sustain."""
        return self._sustain.burn_time

    @property
    def t_terminal(self) -> float:
        """Ideal terminal-pulse burn time, s. Zero when there is no terminal pulse."""
        return self._terminal.burn_time if self.has_terminal else 0.0

    @property
    def propellant_mass(self) -> float:
        return self.dv.m_p_boost + self.dv.m_p_sustain + self.m_p_terminal

    # ------------------------------------------------- terminal pulse control ---

    @property
    def terminal_armed(self) -> bool:
        return self._terminal_armed

    @property
    def terminal_ignition_time(self) -> float | None:
        """Motor time at which the terminal pulse was commanded, or None."""
        return self._terminal_ignition_time

    @terminal_ignition_time.setter
    def terminal_ignition_time(self, value: float | None) -> None:
        self._terminal_ignition_time = value
        self._update_timeline()

    def arm_terminal(self) -> bool:
        """Arm the terminal pulse so the mission may ignite it. Idempotent.

        Returns True when there is a terminal pulse to arm. Arming is a separate step
        from ignition so that `ignite_terminal` cannot fire by accident, and so the
        mission can arm once at the start of a run.
        """
        self._terminal_armed = True
        return self.has_terminal

    def ignite_terminal(self, t: float) -> bool:
        """Command terminal-pulse ignition at motor time `t`.

        Returns True when the pulse was actually lit. Returns False, changing nothing,
        when there is no terminal propellant or the pulse is already burning; that is the
        path taken by every two-phase design vector.

        Raises ValueError if the pulse was never armed, because that is API misuse rather
        than a design outcome. After this call `thrust(t, altitude)` is again a pure
        function of time.
        """
        if not self._terminal_armed:
            raise ValueError("call arm_terminal() before ignite_terminal()")
        if not self.has_terminal or self._terminal_ignition_time is not None:
            return False
        if t < self.t_boost_end:
            self._sizing_warnings.append(
                f"terminal pulse commanded at t = {t:.2f} s, which is inside the boost "
                "phase; the model will superpose the two mass flows"
            )
        self.terminal_ignition_time = t
        return True

    def size_terminal_for_thrust(
        self, thrust_required: float, ambient_pressure: float | None = None
    ) -> float:
        """Set the terminal operating point to deliver `thrust_required`.

        The throat is fixed (it is shared with the sustain or boost phase), so the only
        unknown is the chamber pressure; the burning area then follows from the Kn
        closure. `ambient_pressure` defaults to sea level, because the terminal pulse
        exists to work at impact. Returns the thrust actually achieved and appends a
        warning if the request had to be clamped.
        """
        if not self.has_terminal:
            return 0.0
        if ambient_pressure is None:
            ambient_pressure = self.terminal_sizing_ambient_pressure
        if thrust_required <= 0.0:
            raise ValueError("thrust_required must be positive")

        throat_area = self._terminal_throat_area()
        exit_area = self.area_ratio * throat_area

        def achieved(p_c: float) -> float:
            return self.C_F_vacuum * p_c * throat_area - ambient_pressure * exit_area

        lo, hi = 1.0e4, 5.0e7
        f_lo, f_hi = achieved(lo), achieved(hi)
        if not (f_lo <= thrust_required <= f_hi):
            best = lo if abs(f_lo - thrust_required) < abs(f_hi - thrust_required) else hi
            self._terminal = self._size_terminal_at_pressure(best)
            self._update_timeline()
            self._sizing_warnings.append(
                f"terminal thrust request {thrust_required:.0f} N is outside what the "
                f"shared {self.terminal_throat_source} throat can deliver "
                f"({f_lo:.0f} to {f_hi:.0f} N at p_a = {ambient_pressure / 1e3:.1f} kPa); "
                f"clamped to {achieved(best):.0f} N"
            )
            return achieved(best)

        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if achieved(mid) < thrust_required:
                lo = mid
            else:
                hi = mid
        p_c = 0.5 * (lo + hi)
        self._terminal = self._size_terminal_at_pressure(p_c)
        self._update_timeline()
        return achieved(p_c)

    def throat_transition_report(self) -> list[dict[str, object]]:
        """Every throat-area transition, with its direction and hardware credibility.

        See SOURCES['prop.throat_transition_credibility']. A transition where the throat
        area DECREASES cannot be produced by an ejectable insert and is flagged
        `credible=False`; the caller must resolve it in detailed design (separate
        sustainer nozzle, or a jettisoned tandem booster).
        """
        order: list[tuple[str, float]] = [
            ("boost", self._boost.throat_area),
            ("sustain", self._sustain.throat_area),
        ]
        if self.has_terminal:
            order.append(("terminal", self._terminal_throat_area()))

        report: list[dict[str, object]] = []
        for (name_a, area_a), (name_b, area_b) in zip(order, order[1:]):
            if abs(area_b - area_a) <= 1e-12 * max(area_a, 1e-12):
                direction, credible, mechanism = "unchanged", True, "no change needed"
            elif area_b > area_a:
                direction, credible, mechanism = (
                    "increase",
                    True,
                    "ejectable throat insert, standard hardware",
                )
            else:
                direction, credible, mechanism = (
                    "decrease",
                    False,
                    "no insert mechanism shrinks a throat; needs a separate smaller "
                    "sustainer nozzle or a jettisoned tandem booster",
                )
            report.append(
                {
                    "from": name_a,
                    "to": name_b,
                    "area_from": area_a,
                    "area_to": area_b,
                    "direction": direction,
                    "credible": credible,
                    "mechanism": mechanism,
                }
            )
        return report

    @property
    def total_impulse_vacuum(self) -> float:
        """Vacuum total impulse, N.s.

        Exact for this model because thrust is linear in the phase mass flows and the
        ramps conserve mass:
            I = c* * sum_phase C_F_vac_phase * m_p_phase
        The terminal term is zero when there is no terminal propellant, so this is
        identical to the two-phase value for a two-phase design vector.
        """
        return self.c_star * (
            self._boost.C_F_vacuum * self._boost.propellant_mass
            + self._sustain.C_F_vacuum * self._sustain.propellant_mass
            + self._terminal.C_F_vacuum * self._terminal.propellant_mass
        )

    def operating_point(self) -> dict[str, dict[str, float]]:
        """Per-phase design summary, for the report table.

        The 'terminal' entry is always present. When there is no terminal pulse its
        propellant mass, burn time and mass flow contribution are zero, so summing over
        all three phases is always correct.
        """
        out: dict[str, dict[str, float]] = {}
        for phase in (self._boost, self._sustain, self._terminal):
            out[phase.name] = {
                "propellant_mass": phase.propellant_mass,
                "p_c": phase.p_c,
                "p_e": self.pe_over_pc * phase.p_c,
                "throat_area": phase.throat_area,
                "throat_diameter": math.sqrt(4.0 * phase.throat_area / math.pi),
                "exit_area": phase.exit_area,
                "exit_diameter": math.sqrt(4.0 * phase.exit_area / math.pi),
                "burning_area": phase.burning_area,
                "Kn": phase.kn,
                "burn_rate": phase.burn_rate,
                "mdot": phase.mdot,
                "burn_time": phase.burn_time,
                "thrust_vacuum": phase.thrust_vacuum,
                "thrust_sea_level": phase.thrust_vacuum - P_SEA_LEVEL * phase.exit_area,
                "C_F_vacuum": phase.C_F_vacuum,
                "isp_vacuum": phase.isp_vacuum,
            }
        return out

    # -------------------------------------------------------------- inert masses ---

    def inert_mass_breakdown(self) -> dict[str, float]:
        """Motor inert mass by group, kg.

        Physics route:
          case        thin-wall hoop stress at p_c with a safety factor, plus
                      hemispherical closures at half the cylinder thickness
          insulation  fixed-thickness EPDM on the case wall and closures
          nozzle      conical shell geometric estimate (weak, see SOURCES)
          igniter     fixed fraction of propellant mass (guess)

        The returned dict also carries the cross-check correlation results:
          `correlation_min` / `correlation_max`  inert mass implied by the tactical-motor
                                                propellant mass fraction band
          `mass_fraction_physics`               m_p / (m_p + inert) from the physics route
          `recommended`                         max(physics, correlation_min), the
                                                conservative value the sizing loop
                                                should carry

        WARNING on double counting: the case modelled here is the pressure vessel over
        the motor bay. If WP5 also charges an airframe wall mass over the same stations
        from the nTop-measured structure volume, one of the two must be removed. The
        required case thickness is reported as `case_thickness` so WP5 can compare it
        against `DesignVector.t_wall`.
        """
        dv = self.dv
        geom = self.grain_geometry()
        case_inner_radius = 0.5 * (dv.D - 2.0 * dv.t_wall)
        # The case must hold the highest pressure ANY phase runs at, including a
        # high-pressure terminal pulse. Ignoring the terminal phase here would let the
        # sizing loop buy terminal thrust without paying for the case that contains it.
        peak_pressures = [self._boost.p_c, self._sustain.p_c]
        if self.has_terminal:
            peak_pressures.append(self._terminal.p_c)
        p_design = CASE_SAFETY_FACTOR * max(peak_pressures)
        case_length = geom.length_total + CASE_LENGTH_ALLOWANCE_M
        shell = motor_case_and_insulation(
            case_inner_radius,
            case_length,
            p_design,
            self.case_material,
            self.insulation_material,
            self.insulation_thickness,
        )
        t_cyl = shell["case_thickness"]
        m_case = shell["case"]
        m_insulation = shell["insulation"]

        m_nozzle = self._nozzle_mass()
        m_igniter = IGNITER_MASS_FRACTION * self.propellant_mass

        physics_total = m_case + m_insulation + m_nozzle + m_igniter
        m_p = self.propellant_mass
        inert_from_zeta_hi, inert_from_zeta_lo = correlation_inert_band(m_p)

        return {
            "case": m_case,
            "nozzle": m_nozzle,
            "insulation": m_insulation,
            "igniter": m_igniter,
            "total_physics": physics_total,
            "case_thickness": t_cyl,
            "case_length": case_length,
            "correlation_min": inert_from_zeta_hi,
            "correlation_max": inert_from_zeta_lo,
            "mass_fraction_physics": m_p / (m_p + physics_total),
            "recommended": max(physics_total, inert_from_zeta_hi),
        }

    def _nozzle_mass(self) -> float:
        """Geometric conical-nozzle mass estimate. See SOURCES['prop.nozzle_mass_model'].

        The terminal pulse shares an existing throat, so it adds no nozzle mass here.
        """
        total = 0.0
        for phase in (self._boost, self._sustain):
            total += conical_nozzle_mass(phase.throat_area, phase.exit_area)
            if not self.two_position_throat:
                break   # one throat, one nozzle
        return total

    # --------------------------------------------------------------------- report ---

    def summary(self) -> dict[str, object]:
        """Everything the report and the sizing loop need, in one dict."""
        geom = self.grain_geometry()
        return {
            "c_star": self.c_star,
            "gamma": self.gamma,
            "area_ratio": self.area_ratio,
            "pe_over_pc": self.pe_over_pc,
            "C_F_vacuum": self.C_F_vacuum,
            "isp_vacuum": self.C_F_vacuum * self.c_star / G0,
            "total_impulse_vacuum": self.total_impulse_vacuum,
            "t_boost": self.t_boost,
            "t_sustain": self.t_sustain,
            "t_terminal": self.t_terminal,
            "t_boost_end": self.t_boost_end,
            "t_burnout_sustain": self.t_burnout_sustain,
            "t_burnout": self.t_burnout,
            "has_terminal": self.has_terminal,
            "terminal_ignition_time": self._terminal_ignition_time,
            "terminal_throat_source": self.terminal_throat_source,
            "grain_L_over_D": geom.L_over_D,
            "grain_length_total": geom.length_total,
            "grain_length_terminal": geom.length_terminal,
            "grain_feasible": geom.feasible,
            "volumetric_loading": geom.volumetric_loading,
            "phases": self.operating_point(),
            "inert": self.inert_mass_breakdown(),
            "cross_check": propellant_cross_check(self.gamma),
            "throat_transitions": self.throat_transition_report(),
            "warnings": list(self.warnings) + list(geom.warnings),
        }


def burning_area_to_throat_area(
    burning_area: float, p_c: float, density: float, c_star: float
) -> float:
    """Throat area that holds `p_c` for a given burning area, from the Kn closure."""
    return (
        density * BURN_RATE_COEFF_A * c_star * burning_area
        / p_c ** (1.0 - BURN_RATE_EXPONENT_N)
    )


# --------------------------------------------------------------------------------------
#   Ambient pressure
# --------------------------------------------------------------------------------------
#
# The motor only needs ambient pressure. WP2 owns rocketgen/sizing/atmosphere.py and is
# writing it in parallel, so this module prefers that module when it is importable and
# otherwise falls back to a minimal inline US Standard 1976 pressure model. Remove the
# fallback at integration; see the WP3 report.

_US1976_LAYERS: tuple[tuple[float, float, float, float], ...] = (
    # (base geopotential altitude m, base temperature K, lapse rate K/m, base pressure Pa)
    (0.0, 288.15, -0.0065, 101325.0),
    (11_000.0, 216.65, 0.0, 22_632.1),
    (20_000.0, 216.65, 0.001, 5_474.89),
    (32_000.0, 228.65, 0.0028, 868.019),
    (47_000.0, 270.65, 0.0, 110.906),
    (51_000.0, 270.65, -0.0028, 66.9389),
    (71_000.0, 214.65, -0.002, 3.95642),
)
_R_AIR = 287.0528       # J/(kg.K), US Standard Atmosphere 1976
_EARTH_RADIUS = 6_356_766.0   # m, US Standard Atmosphere 1976 effective earth radius


def _us1976_pressure(altitude: float) -> float:
    """Minimal US Standard 1976 static pressure, Pa. Fallback only."""
    h = _EARTH_RADIUS * altitude / (_EARTH_RADIUS + altitude)   # geopotential altitude
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
        return p_b * math.exp(-G0 * dh / (_R_AIR * t_b))
    t = t_b + lapse * dh
    return p_b * (t / t_b) ** (-G0 / (_R_AIR * lapse))


def _altitude_for_pressure(pressure: float) -> float:
    """Geometric altitude, m, at which US-1976 static pressure equals `pressure`.

    Returns 0.0 when `pressure` is at or above sea-level pressure, i.e. when the
    condition is never met in flight. The search is capped at 30 km, which is the top of
    WP2's tabulated atmosphere and well above anything this mission flies; a returned
    30 000.0 means "at every altitude this model covers".
    """
    if pressure >= P_SEA_LEVEL:
        return 0.0
    lo, hi = 0.0, 30_000.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if _ambient_pressure(mid) > pressure:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _resolve_pressure_function() -> tuple[Callable[[float], float], str]:
    """Prefer WP2's atmosphere module, fall back to the inline US-1976 model."""
    try:
        from . import atmosphere as _atm   # type: ignore
    except Exception:
        return _us1976_pressure, "propulsion.py inline US-1976 fallback"
    # WP2's module exposes atmo(h) -> AtmoState with a .pressure attribute.
    for name in ("atmo", "properties", "atmosphere_properties", "conditions"):
        fn = getattr(_atm, name, None)
        if not callable(fn):
            continue

        def wrapped(h: float, _fn=fn) -> float:
            return float(_fn(h).pressure)

        try:
            if 9.0e4 < wrapped(0.0) < 1.1e5:
                return wrapped, f"rocketgen.sizing.atmosphere.{name}().pressure"
        except Exception:
            pass

    for name in ("pressure", "static_pressure", "get_pressure"):
        fn = getattr(_atm, name, None)
        if callable(fn):
            try:
                value = float(fn(0.0))
            except Exception:
                continue
            if 9.0e4 < value < 1.1e5:
                return fn, f"rocketgen.sizing.atmosphere.{name}"
    return _us1976_pressure, "propulsion.py inline US-1976 fallback"


_ambient_pressure, PRESSURE_SOURCE = _resolve_pressure_function()
