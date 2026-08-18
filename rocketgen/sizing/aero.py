"""Component drag and normal-force build-up for an ogive-cylinder-boattail body with fins.

Scope and fidelity
------------------
Class-I (conceptual sizing) build-up, valid M 0.3 to 5.0 and alpha 0 to 15 deg, per SPEC.md
section 5. Every empirical coefficient carries a source in `SOURCES`. Two quantities are
guesses and say so in their source strings and in their breakdown key names:

  * the lumped protuberance allowance, and
  * the nozzle-exit-area fraction used only when `area_nozzle_exit` is not supplied and
    `power_on=True` is requested.

Design intent
-------------
1. Fast. One `evaluate` call is pure scalar Python and `math`, no numpy scalar boxing, no
   allocation beyond the returned dataclass. Measured at well under 0.1 ms per call.
2. Smooth in Mach. Every branch switch is bridged with the same cubic Hermite blend SUAVE
   uses (`Methods.Aerodynamics.Supersonic_Zero.Drag.Cubic_Spline_Blender`), so the trajectory
   integrator and any optimiser see a C1 function of Mach. Where a blend stands in for physics
   we do not have, the docstring and `SOURCES` say it is a blend.
3. Honest about provenance. `RocketAero.sources_used` records, per geometric quantity, whether
   the value came from an nTop measurement or from closed-form `DesignVector` geometry.

What is reused from SUAVE, and what is not
-----------------------------------------
Reused: the atmosphere (through `atmosphere.py`) and the cubic Hermite blending polynomial.
Not reused: SUAVE's `Fidelity_Zero` and `Supersonic_Zero` drag methods. Those take a
`vehicle` object with `wings`, `fuselages` and a `state.conditions` block, return results into
`conditions.aerodynamics`, and are written around a transport aircraft (a fuselage that does
not produce lift, wings with a main wing, Sears-Haack volume wave drag on an aircraft area
ruling). Bending a rocket into that shape costs more than it saves and would hide the
component breakdown the sizing loop needs. SUAVE's own compressible flat-plate routine is
used in `tests/test_aero.py` as an independent cross-check of the skin-friction model here.

Sign and reference conventions
------------------------------
* All coefficients are referenced to `S_ref` = body maximum cross-section, and moments to
  `S_ref * D`.
* `CD0` is the axial-force coefficient at zero incidence. `CD` is the wind-axis drag,
  `CD0 cos(alpha) + CN sin(alpha)`.
* `CM` is about the nose tip, positive nose-up, so `CM = -CN * x_cp / D`.
* `x_cp` is measured aft from the nose tip, in metres.
* `CN_alpha` is the secant slope `CN / alpha` at the requested alpha, not the derivative at
  alpha = 0. Free-flight and wind-tunnel reductions are secant slopes over a finite alpha
  band, so this is the like-for-like quantity to compare. At alpha = 0 the analytic
  derivative is returned instead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from ..config import AeroCoefficients, DesignVector, NtopMeasurements, register_sources
from . import atmosphere as atm

# --------------------------------------------------------------------------------------
#   Sources. PLAN.md hard rule 2: no invented numbers.
# --------------------------------------------------------------------------------------

SOURCES: dict[str, str] = {
    "aero_skin_friction": (
        "Sommer and Short T-prime reference-temperature method, NACA TN 3391 (1955): "
        "T'/T_inf = 1 + 0.035 M^2 + 0.45 (T_w/T_inf - 1), adiabatic wall with recovery factor "
        "r = 0.89. Incompressible mean turbulent flat-plate law from Prandtl-Schlichting, "
        "Cf = 0.455/(log10 Re)^2.58 (Schlichting, Boundary-Layer Theory 7th ed.); the same "
        "expression SUAVE uses in "
        "Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions."
        "compressible_turbulent_flat_plate. Viscosity at T' from Sutherland's law."
    ),
    "aero_body_form_factor": (
        "Body-of-revolution form factor FF = 1 + 60/f^3 + f/400 with f = L/D. "
        "Raymer, Aircraft Design: A Conceptual Approach 5th ed., eq. 12.31. Raymer intends this "
        "for subsonic flow; it is applied at all Mach here, which is conservative and worth "
        "4 to 9 percent of the body friction term at the fineness ratios of interest."
    ),
    "aero_fin_form_factor": (
        "Fin thickness form factor FF = 1 + 0.6 (t/c)/(x/c)_m + 100 (t/c)^4, the geometric "
        "bracket of Raymer, Aircraft Design 5th ed., eq. 12.30. Raymer's additional "
        "1.34 M^0.18 (cos Lambda)^0.28 bracket is deliberately NOT applied: it is a subsonic "
        "interference allowance and would inject a spurious Mach dependence supersonically."
    ),
    "aero_body_wave_drag": (
        "Forebody zero-lift wave drag (CD0)body,wave = (1.59 + 1.83/M^2) "
        "[atan(0.5/(l_N/d))]^1.69 for M > 1, arctangent in radians. Fleeman, Tactical Missile "
        "Design / Maximizing Missile Flight Performance course notes, attributed there to the "
        "Bonney correlation. The correlation is fitted to pointed forebodies and does not "
        "distinguish a cone from a tangent ogive of the same fineness."
    ),
    "aero_wave_drag_crosscheck": (
        "Second correlation for cross-check only (SPEC.md section 5): Sears-Haack type III "
        "equivalent-body volume wave drag CD = 1.5 pi^2 (d/L)^2 A_max/S_ref, from "
        "Sieron et al., Procedures and Design Data for the Formulation of Aircraft "
        "Configurations, WL-TR-93-3901 p. B-3, as implemented in SUAVE "
        "Methods.Aerodynamics.Supersonic_Zero.Drag.wave_drag_volume_sears_haack. Never added "
        "into CD0."
    ),
    "aero_nose_shape_factor": (
        "The Bonney correlation depends on nose FINENESS alone and cannot distinguish one "
        "nose profile from another at fixed fineness, so on its own it would report a splined "
        "nose and a tangent ogive as identical. The shape sensitivity is therefore supplied as "
        "a dimensionless ratio from linearised slender-body theory, `sizing/wavedrag.py`, "
        "which multiplies the Bonney value. The ratio is exactly 1.0 for the tangent ogive, "
        "so this leaves every pre-spline result unchanged. See "
        "wavedrag.SOURCES['wave_drag_applied_as_ratio'] for why only the ratio is taken."
    ),
    "aero_transonic_bridge": (
        "NOT PHYSICS. A cubic Hermite blend, 2 e^3 - 3 e^2 + 1, bridging the subsonic value to "
        "the M = 1.2 supersonic correlation over 0.95 <= M <= 1.2. Identical polynomial to "
        "SUAVE Methods.Aerodynamics.Supersonic_Zero.Drag.Cubic_Spline_Blender. Chosen because "
        "it is C1 at both ends, so the trajectory integrator and the optimiser see no kink. "
        "It does not reproduce the real transonic drag-rise peak, which lies above the M = 1.2 "
        "supersonic value."
    ),
    "aero_base_drag": (
        "Base drag on the base area. Coast: CD_base = 0.25/M for M > 1 and "
        "CD_base = 0.12 + 0.13 M^2 for M < 1, referenced to the base area. Power-on: multiplied "
        "by (1 - A_e/A_base). Fleeman, Tactical Missile Design / Maximizing Missile Flight "
        "Performance course notes. The two branches are continuous at M = 1 (both give 0.25) "
        "but have a slope kink there, so they are blended over 0.95 <= M <= 1.05."
    ),
    "aero_nozzle_exit_fraction_GUESS": (
        "GUESS. Used only when `area_nozzle_exit` is None and `power_on=True`: the nozzle exit "
        "is assumed to occupy A_e/A_base = 0.50. This is a placeholder for the real value, "
        "which WP3's motor model must supply. It changes only the powered-base drag relief."
    ),
    "aero_boattail_drag": (
        "Boattail pressure drag from the exact Prandtl-Meyer expansion through the boattail "
        "half-angle, applied as -Cp on the projected annulus (S_ref - S_base): "
        "Cp = (2/(gamma M^2))(p2/p1 - 1). Standard oblique-expansion gas dynamics, Liepmann "
        "and Roshko, Elements of Gasdynamics (1957), the same reference McCoy's MC DRAG "
        "(ARBRL-MR-02293) cites for boattail pressure drag. Pressure recovery along the "
        "boattail is NOT modelled, so this is an upper bound on the boattail contribution."
    ),
    "aero_oblique_shock": (
        "Exact oblique-shock relations for a compressive turn. The theta-beta-Mach relation "
        "tan(theta) = 2 cot(beta)(M^2 sin^2 beta - 1)/(M^2(gamma + cos 2beta) + 2) is solved in "
        "CLOSED FORM for the weak root using the explicit cubic solution given by "
        "J. D. Anderson, Modern Compressible Flow, and by T. W. Thompson (1950) / "
        "L. Rudd and M. J. Lewis, 'Comparison of Shock Calculation Methods', J. Aircraft 35(4), "
        "1998: tan(beta) = (M^2 - 1 + 2 lambda cos((4 pi d + arccos chi)/3)) / "
        "(3 (1 + (gamma-1)/2 M^2) tan theta), with d = 1 for the weak solution, "
        "lambda^2 = (M^2-1)^2 - 3(1+(gamma-1)/2 M^2)(1+(gamma+1)/2 M^2) tan^2 theta and "
        "chi = ((M^2-1)^3 - 9(1+(gamma-1)/2 M^2)(1+(gamma-1)/2 M^2+(gamma+1)/4 M^4) tan^2 "
        "theta)/lambda^3. Static pressure from Rankine-Hugoniot, "
        "p2/p1 = 1 + 2 gamma/(gamma+1)(M^2 sin^2 beta - 1), giving "
        "Cp = 4/(gamma+1) (M^2 sin^2 beta - 1)/M^2. Verified against the textbook case "
        "M = 2, theta = 10 deg -> beta = 39.31 deg, p2/p1 = 1.707. chi = -1 is exactly the "
        "detachment condition, which gives the maximum attached deflection with no table lookup."
    ),
    "aero_isentropic_compression_turn": (
        "Pressure rise through a compressive turn taken as an exact isentropic Prandtl-Meyer "
        "compression, nu2 = nu1 - delta, with Cp = (2/(gamma M1^2))(p2/p1 - 1). Same gas "
        "dynamics and the same reference as the boattail expansion, Liepmann and Roshko, "
        "Elements of Gasdynamics (1957), run in the opposite direction. APPROXIMATION: the "
        "entropy rise across the real oblique shock is not modelled, and when delta exceeds "
        "the available Prandtl-Meyer angle the shock detaches and the value is clamped at the "
        "sonic result, which UNDERSTATES the detached-shock pressure. Used by "
        "`aero_iv1` for the interstage shoulder; the single-body SV-1 build-up has no "
        "compressive turn and so does not exercise it."
    ),
    "aero_fin_wave_drag": (
        "Fin thickness (wave) drag from Ackeret linearised supersonic thin-airfoil theory for a "
        "symmetric double wedge: cd = (t/c)^2 / [x_t (1 - x_t) sqrt(M^2 - 1)], which reduces to "
        "the textbook 4 (t/c)^2 / sqrt(M^2 - 1) at x_t = 0.5. Liepmann and Roshko, Elements of "
        "Gasdynamics ch. 4; Anderson, Fundamentals of Aerodynamics, supersonic thin-airfoil "
        "theory. Leading-edge sweep relief is NOT credited, which is conservative."
    ),
    "aero_fin_wave_drag_crosscheck": (
        "Cross-check only, never added into CD0: Newtonian-impact leading-edge drag "
        "CD = n_pairs Cp_max sin^2(delta_LE) cos(Lambda_LE) t_mac b / S_ref with Cp_max from the "
        "Rayleigh pitot relation. Fleeman, Maximizing Missile Flight Performance course notes."
    ),
    "aero_body_normal_force": (
        "CN_body = (A_base/S_ref) sin(2 alpha) cos(alpha/2) + eta Cd_cross (A_p/S_ref) "
        "sin^2(alpha). First term is slender-body potential lift (Munk; Pitts, Nielsen and "
        "Kaattari, NACA TR 1307), whose alpha -> 0 slope is 2 A_base/S_ref per radian. Second "
        "term is the Allen and Perkins viscous cross-flow term (H. J. Allen and E. W. Perkins, "
        "NACA Report 1048 / NACA RM A50L07, 1951), generalised by Jorgensen (NASA TN D-6996). "
        "eta Cd_cross = pi/2 = 1.5708 is fixed so the expression reduces exactly to the "
        "published closed form in Fleeman, |CN|_body = sin(2a) cos(a/2) + 2 (l/d) sin^2(a), for "
        "a cylinder of planform area l*d. With Cd_cross = 1.2 for a circular cylinder that "
        "implies a cross-flow drag proportionality factor eta = 1.31."
    ),
    "aero_fin_normal_force": (
        "Fin panel-pair normal force, two branches with the switch at "
        "M_c = sqrt(1 + (8/(pi A))^2). Supersonic (linearised, Ackeret/Busemann first order with "
        "the Prandtl-Glauert factor): |CN| = [4 |sin a' cos a'|/sqrt(M^2-1) + 2 sin^2 a'] "
        "S_pair/S_ref. Subsonic (slender-wing lifting-surface estimate): "
        "|CN| = [(pi A/2) |sin a' cos a'| + 2 sin^2 a'] S_pair/S_ref. The 2 sin^2 a' term is "
        "Newtonian impact. Fleeman, Maximizing Missile Flight Performance course notes. The two "
        "branches are equal at M_c by construction; they are blended over 0.9 M_c to 1.1 M_c to "
        "remove the slope kink."
    ),
    "aero_fin_body_upwash": (
        "Fin-body upwash interference factor K_upwash = 1 + a/s, with a the body radius and s "
        "the fin tip radius from the body axis. This is Barrowman's fin-body interference "
        "factor (J. S. Barrowman, The Practical Calculation of the Aerodynamic Characteristics "
        "of Slender Finned Vehicles, 1967) and is identically the span average of the two-"
        "dimensional cross-flow upwash 1 + a^2/y^2 about a circular cylinder from y = a to "
        "y = s, the same potential solution Allen and Perkins use."
    ),
    "aero_fin_afterbody_carryover": (
        "Fin-to-body afterbody carryover factor K_carryover, default 1.0. Pitts, Nielsen and "
        "Kaattari (NACA TR 1307) tabulate K_B(W); at supersonic speed the carryover region is "
        "bounded by the Mach cone, and for a tail fin whose trailing edge is at or near the body "
        "base there is no afterbody left to carry the load, which is the case for this "
        "configuration. Exposed as a constructor argument so a DATCOM or measured value can be "
        "supplied when the fins move forward."
    ),
    "aero_nose_cp": (
        "Potential-lift centre of pressure of the forebody: 2/3 of the nose length for a cone "
        "and 0.466 of the nose length for a tangent ogive. Barrowman, The Practical Calculation "
        "of the Aerodynamic Characteristics of Slender Finned Vehicles (1967). Equivalent to the "
        "centroid of dS/dx under slender-body theory, which is what is computed directly when "
        "nTop supplies the cross-section area distribution. For a SPLINED nose there is no "
        "published constant, so the same slender-body identity is used in closed form, "
        "x_cp = L_n - V_nose/S_base. Evaluated on the tangent ogive it gives 0.4634 against "
        "Barrowman's published 0.466, a 0.6 percent agreement that confirms it is the same "
        "quantity. For the drag-optimal splined nose it gives 0.4974, i.e. the centre of "
        "pressure moves AFT because the optimal nose carries less forebody volume."
    ),
    "aero_crossflow_cp": (
        "Viscous cross-flow lift acts at the centroid of the body planform area. Allen and "
        "Perkins, NACA Report 1048."
    ),
    "aero_fin_cp": (
        "Fin centre of pressure aft of the fin root leading edge. Subsonic: Barrowman (1967), "
        "X_f = (X_R/3)(c_r + 2 c_t)/(c_r + c_t) + (1/6)[c_r + c_t - c_r c_t/(c_r + c_t)] with "
        "X_R = b tan(Lambda_LE). Supersonic: linearised theory gives a chordwise-uniform load "
        "inside the Mach cones, so the load centroid sits at mid-chord of the mean aerodynamic "
        "chord. Blended over 0.95 <= M <= 1.2 with the cubic Hermite blend."
    ),
    "aero_protuberance_GUESS": (
        "GUESS. A single lumped allowance for antennas, umbilicals, fin-root fairings, lugs, "
        "joints and surface imperfections, taken as 5 percent of the sum of the other zero-lift "
        "components. There is no correlation behind this number. It appears in the breakdown as "
        "'CD_protuberance_GUESS' so it can never be mistaken for a computed quantity."
    ),
    "aero_validation": (
        "Validated against free-flight data for the Army-Navy Basic Finner reference "
        "projectile: A. D. Dupuis and W. Hathaway, 'Aeroballistic Range Tests of the Basic "
        "Finner Reference Projectile at Supersonic Velocities', Defence Research Establishment "
        "Valcartier, DREV-TM-9703 (1997), DTIC AD-A636861, Table VII (linear-theory reduction, "
        "26 shots, M 1.06 to 4.47). See tests/test_aero.py for the digitised table and the "
        "tolerances achieved."
    ),
}
register_sources(SOURCES)


# --------------------------------------------------------------------------------------
#   Constants
# --------------------------------------------------------------------------------------

GAMMA: float = 1.4              # ratio of specific heats for air
RECOVERY_FACTOR: float = 0.89   # turbulent adiabatic-wall recovery factor, Sommer and Short
ETA_CD_CROSS: float = math.pi / 2.0   # see SOURCES["aero_body_normal_force"]

#: Transonic bridge window. See SOURCES["aero_transonic_bridge"].
M_BRIDGE_LO: float = 0.95
M_BRIDGE_HI: float = 1.20

#: Base-drag branch blend window. See SOURCES["aero_base_drag"].
M_BASE_LO: float = 0.95
M_BASE_HI: float = 1.05

#: GUESS. See SOURCES["aero_protuberance_GUESS"].
PROTUBERANCE_FRACTION_GUESS: float = 0.05

#: GUESS. See SOURCES["aero_nozzle_exit_fraction_GUESS"].
NOZZLE_EXIT_AREA_FRACTION_GUESS: float = 0.50

#: Barrowman potential-lift centre of pressure, as a fraction of nose length.
X_CP_NOSE_FRACTION: dict[str, float] = {"cone": 2.0 / 3.0, "tangent_ogive": 0.466}

def x_cp_nose_fraction(shape: str, L_n: float, D: float,
                       control: "Sequence[float] | None" = None) -> float:
    """Barrowman nose centre-of-pressure station as a fraction of nose length.

    For the two tabulated shapes this returns the published constants unchanged. For a splined
    nose there is no published constant, so it is derived from the SAME slender-body result
    Barrowman's own numbers come from: the potential-flow normal force of a slender body is
    distributed as dS/dx, so the centre of pressure is the dS/dx-weighted mean station,

        x_cp = L_n - V_nose / S_base,

    which is exact for any body of revolution under that theory. Evaluated on the tangent
    ogive this reproduces 0.466 to three decimals, which is the check that it is the same
    quantity and not a different one. See SOURCES["aero_nose_cp"].
    """
    if shape != "spline":
        return X_CP_NOSE_FRACTION[shape]
    if control is None:
        raise ValueError("nose_shape 'spline' needs control values")
    from ..oml_spline import SplineProfile

    R = 0.5 * D
    p = SplineProfile(length=L_n, radius=R, control=tuple(control), n_poly=400)
    s_base = math.pi * R * R
    return (L_n - p.volume() / s_base) / L_n


MACH_MIN_VALID: float = 0.3
MACH_MAX_VALID: float = 5.0
ALPHA_MAX_VALID: float = math.radians(15.0)


# --------------------------------------------------------------------------------------
#   Small helpers
# --------------------------------------------------------------------------------------


def cubic_blend(x: float, x0: float, x1: float) -> float:
    """Cubic Hermite blend: 1 at x <= x0, 0 at x >= x1, C1 at both ends.

    Same polynomial as SUAVE's `Cubic_Spline_Blender.compute`. See
    SOURCES["aero_transonic_bridge"].
    """
    if x <= x0:
        return 1.0
    if x >= x1:
        return 0.0
    e = (x - x0) / (x1 - x0)
    return 2.0 * e * e * e - 3.0 * e * e + 1.0


def _mu_sutherland(t: float) -> float:
    """Sutherland viscosity, scalar fast path. Same law and constants as `atmosphere`."""
    return (
        atm.SUTHERLAND_MU0
        * (t / atm.SUTHERLAND_T0) ** 1.5
        * (atm.SUTHERLAND_T0 + atm.SUTHERLAND_S)
        / (t + atm.SUTHERLAND_S)
    )


def cf_turbulent(re: float, mach: float, t_inf: float) -> float:
    """Compressible turbulent mean flat-plate skin friction, on free-stream dynamic pressure.

    Sommer and Short T-prime reference-temperature method. See SOURCES["aero_skin_friction"].

    Args:
        re: Reynolds number on the reference length, free-stream properties.
        mach: free-stream Mach number.
        t_inf: free-stream static temperature, K.
    """
    if re < 1.0e4:
        re = 1.0e4   # below this the turbulent law is meaningless; clamp rather than blow up

    # Adiabatic wall temperature ratio, recovery factor r = 0.89.
    tw_ratio = 1.0 + RECOVERY_FACTOR * 0.5 * (GAMMA - 1.0) * mach * mach
    # Sommer and Short reference temperature.
    tp_ratio = 1.0 + 0.035 * mach * mach + 0.45 * (tw_ratio - 1.0)
    t_ref = t_inf * tp_ratio

    # Constant static pressure across the layer, so rho'/rho_inf = T_inf/T'.
    rho_ratio = 1.0 / tp_ratio
    mu_ratio = _mu_sutherland(t_ref) / _mu_sutherland(t_inf)
    re_ref = re * rho_ratio / mu_ratio
    if re_ref < 1.0e4:
        re_ref = 1.0e4

    # Prandtl-Schlichting incompressible mean turbulent flat plate, evaluated at Re'.
    cf_inc = 0.455 / (math.log10(re_ref) ** 2.58)

    # Refer the wall shear back to free-stream dynamic pressure.
    return cf_inc * rho_ratio


def _prandtl_meyer_nu(mach: float) -> float:
    """Prandtl-Meyer angle, rad."""
    if mach <= 1.0:
        return 0.0
    b = math.sqrt((GAMMA + 1.0) / (GAMMA - 1.0))
    m2 = mach * mach - 1.0
    return b * math.atan(math.sqrt(m2) / b) - math.atan(math.sqrt(m2))


def _prandtl_meyer_mach(nu: float) -> float:
    """Invert the Prandtl-Meyer function.

    Newton on nu(M) using the analytic derivative dnu/dM = sqrt(M^2-1)/(M(1+(g-1)/2 M^2)),
    with a bracket guard so a bad step falls back to bisection. Converges in a handful of
    iterations, which matters because the boattail term is on the trajectory integrator's
    hot path.
    """
    if nu <= 0.0:
        return 1.0
    lo, hi = 1.0000001, 60.0
    m = 2.0
    for _ in range(40):
        f = _prandtl_meyer_nu(m) - nu
        if abs(f) < 1.0e-11:
            return m
        if f > 0.0:
            hi = m
        else:
            lo = m
        m2 = m * m - 1.0
        dnu = math.sqrt(m2) / (m * (1.0 + 0.5 * (GAMMA - 1.0) * m * m)) if m2 > 0.0 else 0.0
        m_new = m - f / dnu if dnu > 1.0e-12 else 0.5 * (lo + hi)
        m = m_new if lo < m_new < hi else 0.5 * (lo + hi)
    return m


def _cp_max_stagnation(mach: float) -> float:
    """Stagnation pressure coefficient. Rayleigh pitot above M = 1, isentropic below.

    Used only by the Newtonian-impact fin leading-edge cross-check.
    """
    if mach <= 1.0e-6:
        return 2.0
    g = GAMMA
    if mach < 1.0:
        pt_p = (1.0 + 0.5 * (g - 1.0) * mach * mach) ** (g / (g - 1.0))
    else:
        pt_p = (((g + 1.0) * mach * mach) / 2.0) ** (g / (g - 1.0)) * (
            (g + 1.0) / (2.0 * g * mach * mach - (g - 1.0))
        ) ** (1.0 / (g - 1.0))
    return 2.0 / (g * mach * mach) * (pt_p - 1.0)


# --------------------------------------------------------------------------------------
#   Shared physics kernels
#
#   These are the validated pieces of the build-up, exposed as free functions so that a
#   multi-body configuration can reuse them without a second implementation. `RocketAero`
#   below is a thin assembly over exactly these functions, and so is
#   `aero_iv1.StackAero`. Every one of them keeps the arithmetic it had when it lived
#   inside a `RocketAero` method, so the Basic Finner validation still covers them.
# --------------------------------------------------------------------------------------


def body_form_factor(fineness: float) -> float:
    """Body-of-revolution friction form factor. See SOURCES["aero_body_form_factor"]."""
    return 1.0 + 60.0 / fineness ** 3 + fineness / 400.0


def surface_form_factor(t_over_c: float, x_t: float) -> float:
    """Lifting-surface thickness form factor. See SOURCES["aero_fin_form_factor"]."""
    return 1.0 + 0.6 * t_over_c / x_t + 100.0 * t_over_c ** 4


def bonney_nose_wave_cd(f_nose: float, mach: float) -> float:
    """Forebody wave drag on the NOSE BASE AREA. See SOURCES["aero_body_wave_drag"].

    The caller refers it to whatever reference area it is using. Blended to zero below
    M = 0.95 with the cubic Hermite bridge, which is not physics.
    """
    shape = math.atan(0.5 / f_nose) ** 1.69
    m = max(mach, M_BRIDGE_HI)
    cd_super = (1.59 + 1.83 / (m * m)) * shape
    return (1.0 - cubic_blend(mach, M_BRIDGE_LO, M_BRIDGE_HI)) * cd_super


def base_cd_on_base_area(mach: float) -> float:
    """Base drag referred to the BASE AREA, coast. See SOURCES["aero_base_drag"]."""
    m_sup = max(mach, M_BASE_LO)
    cd_sup = 0.25 / m_sup
    cd_sub = 0.12 + 0.13 * mach * mach
    w = cubic_blend(mach, M_BASE_LO, M_BASE_HI)
    return w * cd_sub + (1.0 - w) * cd_sup


def _isentropic_cp(m1: float, m2: float) -> float:
    """Pressure coefficient of an isentropic turn from M1 to M2, on the M1 dynamic pressure."""
    h = 0.5 * (GAMMA - 1.0)
    p_ratio = ((1.0 + h * m1 * m1) / (1.0 + h * m2 * m2)) ** (GAMMA / (GAMMA - 1.0))
    return 2.0 / (GAMMA * m1 * m1) * (p_ratio - 1.0)


def expansion_turn_cp(mach: float, delta: float) -> float:
    """Cp after an isentropic expansion through `delta` rad. See SOURCES["aero_boattail_drag"].

    Negative, because an expansion drops the static pressure. Exact Prandtl-Meyer, not a
    linearisation.
    """
    if delta <= 0.0 or mach <= 1.0:
        return 0.0
    nu2 = _prandtl_meyer_nu(mach) + delta
    return _isentropic_cp(mach, _prandtl_meyer_mach(nu2))


def compression_turn_cp(mach: float, delta: float) -> float:
    """Cp after an isentropic compression through `delta` rad, on the upstream q.

    Positive. The exact Prandtl-Meyer function run backwards, so it is the mirror image of
    `expansion_turn_cp` and shares its gas dynamics. See
    SOURCES["aero_isentropic_compression_turn"].

    When `delta` exceeds the available Prandtl-Meyer angle the turn cannot be made
    isentropically and the real flow throws a detached shock. The value is then clamped at
    the sonic result, which UNDERSTATES the pressure behind a detached shock.
    """
    if delta <= 0.0 or mach <= 1.0:
        return 0.0
    nu1 = _prandtl_meyer_nu(mach)
    nu2 = nu1 - delta
    m2 = _prandtl_meyer_mach(nu2) if nu2 > 0.0 else 1.0
    return _isentropic_cp(mach, m2)


def _oblique_shock_chi(mach: float, theta: float) -> tuple[float, float]:
    """(lambda, chi) of the explicit oblique-shock solution, or (0, -inf) past detachment.

    Helper for `oblique_shock_cp`. `chi` reaches -1 exactly at the maximum attached deflection,
    and `lambda**2` goes negative beyond it, so the pair is also the detachment test.
    """
    m2 = mach * mach
    a = 1.0 + 0.5 * (GAMMA - 1.0) * m2
    b = 1.0 + 0.5 * (GAMMA + 1.0) * m2
    c = 1.0 + 0.5 * (GAMMA - 1.0) * m2 + 0.25 * (GAMMA + 1.0) * m2 * m2
    tt = math.tan(theta) ** 2
    lam2 = (m2 - 1.0) ** 2 - 3.0 * a * b * tt
    if lam2 <= 0.0:
        return 0.0, -float("inf")
    lam = math.sqrt(lam2)
    chi = ((m2 - 1.0) ** 3 - 9.0 * a * c * tt) / (lam * lam * lam)
    return lam, chi


def oblique_shock_theta_max(mach: float) -> float:
    """Maximum deflection angle, rad, for which an oblique shock stays attached.

    Found by bisection on the condition chi = -1, where the weak and strong solutions merge.
    See SOURCES["aero_oblique_shock"]. Returns 0 at or below M = 1.
    """
    if mach <= 1.0:
        return 0.0
    m2 = mach * mach
    a = 1.0 + 0.5 * (GAMMA - 1.0) * m2
    b = 1.0 + 0.5 * (GAMMA + 1.0) * m2
    # chi is monotone decreasing in theta, and lambda^2 vanishes above the merge point, so this
    # upper bracket is always past detachment.
    hi = math.atan(math.sqrt((m2 - 1.0) ** 2 / (3.0 * a * b)))
    lo = 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        _, chi = _oblique_shock_chi(mach, mid)
        if chi < -1.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1.0e-10:
            break
    return lo


def oblique_shock_cp(mach: float, delta: float) -> tuple[float, bool]:
    """(pressure coefficient, attached) behind a weak oblique shock turning the flow `delta`.

    Exact closed-form weak solution of the theta-beta-Mach relation, then the Rankine-Hugoniot
    static-pressure ratio. This is the tangent-wedge estimate for a flare on a cylinder.
    See SOURCES["aero_oblique_shock"].

    When `delta` exceeds the maximum attached deflection the shock detaches and there is no
    weak solution. The value is then CLAMPED at the maximum attached deflection, which is
    continuous in Mach and is a LOWER bound on the real detached-shock surface pressure.
    """
    if mach <= 1.0 or delta <= 0.0:
        return 0.0, False
    m2 = mach * mach
    a = 1.0 + 0.5 * (GAMMA - 1.0) * m2

    lam, chi = _oblique_shock_chi(mach, delta)
    attached = chi >= -1.0
    if not attached:
        delta = oblique_shock_theta_max(mach)
        if delta <= 0.0:
            return 0.0, False
        lam, chi = _oblique_shock_chi(mach, delta)
        chi = max(chi, -1.0)
    chi = min(max(chi, -1.0), 1.0)

    t = math.tan(delta)
    # delta = 1 selects the weak (physically realised) root of the cubic.
    tan_beta = (m2 - 1.0 + 2.0 * lam * math.cos((4.0 * math.pi + math.acos(chi)) / 3.0)) / (
        3.0 * a * t
    )
    beta = math.atan(tan_beta)
    msb2 = m2 * math.sin(beta) ** 2
    if msb2 < 1.0:
        msb2 = 1.0
    cp = 4.0 / (GAMMA + 1.0) * (msb2 - 1.0) / m2
    return cp, attached


def fin_wave_cd_2d(t_over_c: float, x_t: float, mach: float) -> float:
    """Ackeret double-wedge thickness drag, 2D, on the PLANFORM area of one side.

    See SOURCES["aero_fin_wave_drag"]. Blended to zero below M = 0.95.
    """
    m = max(mach, M_BRIDGE_HI)
    beta = math.sqrt(m * m - 1.0)
    cd_2d = t_over_c * t_over_c / (x_t * (1.0 - x_t) * beta)
    return (1.0 - cubic_blend(mach, M_BRIDGE_LO, M_BRIDGE_HI)) * cd_2d


def newtonian_le_cd(
    n_pairs: float,
    sweep_le: float,
    t_max: float,
    x_t: float,
    c_mac: float,
    span: float,
    S_ref: float,
    mach: float,
) -> float:
    """Newtonian-impact leading-edge drag of a surface set, on S_ref.

    See SOURCES["aero_fin_wave_drag_crosscheck"]. Reported as a cross-check, never summed.
    """
    if S_ref <= 0.0 or c_mac <= 0.0:
        return 0.0
    m_le = mach * math.cos(sweep_le)
    cp_max = _cp_max_stagnation(m_le)
    delta_le = 2.0 * math.atan(t_max / (2.0 * x_t * c_mac))
    return (
        n_pairs
        * cp_max
        * math.sin(delta_le) ** 2
        * math.cos(sweep_le)
        * t_max
        * span
        / S_ref
    )


def body_normal_force_terms(
    S_base: float, S_ref: float, area_planform: float, alpha: float
) -> tuple[float, float]:
    """(potential, cross-flow) body normal force on S_ref.

    See SOURCES["aero_body_normal_force"]. No Mach dependence, by construction.
    """
    a = abs(alpha)
    potential = (S_base / S_ref) * math.sin(2.0 * a) * math.cos(0.5 * a)
    crossflow = ETA_CD_CROSS * (area_planform / S_ref) * math.sin(a) ** 2
    s = 1.0 if alpha >= 0.0 else -1.0
    return s * potential, s * crossflow


def lifting_surface_cn_alone(aspect_ratio_pair: float, mach: float, alpha: float) -> float:
    """Normal force of one opposing panel pair, referred to the PAIR EXPOSED AREA.

    Slender-wing below the branch Mach number, linearised supersonic above, blended across.
    Interference factors are the caller's job. See SOURCES["aero_fin_normal_force"].
    """
    if aspect_ratio_pair <= 0.0:
        return 0.0
    a = abs(alpha)
    sc = abs(math.sin(a) * math.cos(a))
    s2 = math.sin(a) ** 2
    ar = aspect_ratio_pair

    m_c = math.sqrt(1.0 + (8.0 / (math.pi * ar)) ** 2)
    slender = (math.pi * ar / 2.0) * sc + 2.0 * s2
    m_lin = max(mach, m_c)
    linear = 4.0 * sc / math.sqrt(m_lin * m_lin - 1.0) + 2.0 * s2

    w = cubic_blend(mach, 0.9 * m_c, 1.1 * m_c)
    cn = w * slender + (1.0 - w) * linear
    return cn if alpha >= 0.0 else -cn


def barrowman_upwash(radius_body: float, radius_tip: float) -> float:
    """Body-upwash interference factor 1 + a/s. See SOURCES["aero_fin_body_upwash"]."""
    if radius_tip <= 0.0:
        return 1.0
    return 1.0 + radius_body / radius_tip


def solve_alpha_for_cn(
    cn_of, target_cn: float, alpha_max: float
) -> float:
    """Angle of attack, rad, at which `cn_of(alpha)` reaches `target_cn`.

    Bracketed bisection with a secant accelerator. `cn_of` must be non-decreasing on
    [0, alpha_max], which every normal-force model in this package is. Saturates at
    `alpha_max` rather than raising, so a sizing loop sees a clipped control instead of an
    exception. Signs are carried through, so a negative target returns a negative alpha.
    """
    target = float(target_cn)
    sign = 1.0 if target >= 0.0 else -1.0
    target = abs(target)
    if target <= 0.0:
        return 0.0

    cn_max = cn_of(alpha_max)
    if target >= cn_max:
        return sign * alpha_max

    lo, hi = 0.0, alpha_max
    f_lo, f_hi = -target, cn_max - target
    a = target / (cn_max / alpha_max)   # linear first guess
    for _ in range(60):
        f = cn_of(a) - target
        if abs(f) < 1.0e-12:
            break
        if f > 0.0:
            hi, f_hi = a, f
        else:
            lo, f_lo = a, f
        # secant inside the bracket, bisect if it steps outside
        if f_hi != f_lo:
            a_sec = lo - f_lo * (hi - lo) / (f_hi - f_lo)
        else:
            a_sec = 0.5 * (lo + hi)
        a = a_sec if lo < a_sec < hi else 0.5 * (lo + hi)
        if hi - lo < 1.0e-14:
            break
    return sign * a


# --------------------------------------------------------------------------------------
#   Resolved geometry, with provenance
# --------------------------------------------------------------------------------------


@dataclass
class _Geometry:
    """Geometry actually used by the model, after nTop measurements override analytics."""

    # body
    D: float
    S_ref: float
    L_total: float
    L_nose: float
    L_cyl: float
    L_boattail: float
    S_base: float
    area_wetted_body: float
    area_planform_body: float
    x_cp_potential: float          # from nose tip, m
    x_cp_crossflow: float          # from nose tip, m
    beta_boattail: float           # boattail half-angle, rad (0 if none)
    nose_shape: str

    # fins
    n_fin: int
    b_fin: float
    c_r: float
    c_t: float
    sweep_le: float
    t_fin: float
    x_fin_le: float
    x_t_fin: float                 # chordwise station of maximum thickness, fraction of chord
    S_fin_panel: float             # one exposed panel, m^2
    area_wetted_fins: float        # all panels, both sides, m^2
    c_mac: float
    x_mac_le: float                # from the fin root leading edge, m
    aspect_ratio_pair: float
    S_fin_pair: float              # exposed planform of one opposing pair, m^2
    k_upwash: float
    x_cp_fin_sub: float            # from nose tip, m
    x_cp_fin_sup: float            # from nose tip, m

    # aft end
    area_nozzle_exit: float
    nozzle_exit_is_guess: bool

    provenance: dict[str, str] = field(default_factory=dict)


def _nose_wetted_and_planform(
    shape: str,
    L_n: float,
    D: float,
    control: Sequence[float] | None = None,
) -> tuple[float, float, float]:
    """(wetted area, planform area, planform centroid from tip) for a nose of revolution.

    `control` supplies the spline control values when `shape == "spline"`. The closed forms
    used then are the EXACT frustum sums of the revolved chord polygon, which is the solid nTop
    actually builds, rather than a quadrature of the smooth spline. At the ogive-equivalent
    control values they reproduce this function's tangent-ogive branch to 5e-5 relative.
    """
    R = 0.5 * D
    if shape == "spline":
        if control is None:
            raise ValueError("nose_shape 'spline' needs control values")
        from ..oml_spline import SplineProfile

        p = SplineProfile(length=L_n, radius=R, control=tuple(control), n_poly=160)
        planform, x_bar = p.planform_area_and_centroid()
        return p.lateral_area(), planform, x_bar
    if shape == "cone":
        slant = math.hypot(L_n, R)
        wetted = math.pi * R * slant
        planform = 0.5 * L_n * D                    # triangle
        x_bar = (2.0 / 3.0) * L_n
        return wetted, planform, x_bar
    if shape == "tangent_ogive":
        # Tangent ogive of length L_n and base radius R. rho is the ogive radius.
        rho = (R * R + L_n * L_n) / (2.0 * R)
        # r(x) = sqrt(rho^2 - (L_n - x)^2) - (rho - R)
        n = 64
        wetted = 0.0
        planform = 0.0
        moment = 0.0
        dx = L_n / n
        r_prev = 0.0
        for i in range(1, n + 1):
            x = i * dx
            r = math.sqrt(max(rho * rho - (L_n - x) ** 2, 0.0)) - (rho - R)
            r = max(r, 0.0)
            r_mid = 0.5 * (r_prev + r)
            dl = math.hypot(dx, r - r_prev)
            wetted += 2.0 * math.pi * r_mid * dl
            planform += 2.0 * r_mid * dx
            moment += 2.0 * r_mid * dx * (x - 0.5 * dx)
            r_prev = r
        x_bar = moment / planform if planform > 0.0 else 0.5 * L_n
        return wetted, planform, x_bar
    raise ValueError(f"unknown nose_shape {shape!r}, expected 'cone' or 'tangent_ogive'")


def _frustum_planform_centroid(R: float, r_b: float, L: float) -> tuple[float, float]:
    """(planform area, centroid from the frustum's forward station) for a conical frustum."""
    if L <= 0.0:
        return 0.0, 0.0
    area = (R + r_b) * L                       # trapezoid, full width
    k = (R - r_b) / L                          # -dr/dx
    # centroid of 2 r(x) with r = R - k x
    num = R * L * L / 2.0 - k * L ** 3 / 3.0
    den = L * (R - 0.5 * k * L)
    return area, (num / den if den > 0.0 else 0.5 * L)


def _dsdx_centroid_from_distribution(dist: list[tuple[float, float]]) -> tuple[float, float]:
    """(net area change, dS/dx-weighted centroid) from a measured cross-section distribution.

    Slender-body theory puts the potential normal force at the centroid of dS/dx, and the net
    slope-integral equals S(x_end) - S(x_start). Using nTop's measured S(x) removes the need for
    the analytic nose-shape centre-of-pressure fractions.
    """
    if len(dist) < 3:
        raise ValueError("area_distribution needs at least 3 stations")
    pts = sorted(dist, key=lambda p: p[0])
    num = 0.0
    den = 0.0
    for (x0, s0), (x1, s1) in zip(pts[:-1], pts[1:]):
        if x1 <= x0:
            continue
        ds = s1 - s0
        num += ds * 0.5 * (x0 + x1)
        den += ds
    return den, (num / den if abs(den) > 1.0e-12 else pts[0][0])


def _planform_from_distribution(dist: list[tuple[float, float]]) -> tuple[float, float]:
    """(planform area, planform centroid) from a measured cross-section area distribution."""
    pts = sorted(dist, key=lambda p: p[0])
    area = 0.0
    moment = 0.0
    for (x0, s0), (x1, s1) in zip(pts[:-1], pts[1:]):
        if x1 <= x0:
            continue
        r0 = math.sqrt(max(s0, 0.0) / math.pi)
        r1 = math.sqrt(max(s1, 0.0) / math.pi)
        da = (r0 + r1) * (x1 - x0)              # trapezoid of full width 2r
        area += da
        moment += da * 0.5 * (x0 + x1)
    return area, (moment / area if area > 0.0 else 0.0)


# --------------------------------------------------------------------------------------
#   The model
# --------------------------------------------------------------------------------------


class RocketAero:
    """Component build-up for an ogive-cylinder-boattail body with `n_fin` cruciform fins.

    Args:
        dv: the design vector. Closed-form geometry from it is the fallback for everything.
        meas: nTop measurements. When present, `area_wetted_body`, `area_wetted_fins`,
            `area_base` and `area_distribution` replace the analytic estimates, and
            `sources_used` records that they did.
        nose_shape: 'tangent_ogive' (the SV-1 shape) or 'cone'. `DesignVector` has no nose-shape
            field, so this is a constructor argument. See the module report notes.
        fin_max_thickness_station: chordwise station of maximum fin thickness as a fraction of
            chord, for the double-wedge wave-drag model. 0.5 is a symmetric double wedge.
        k_afterbody_carryover: fin-to-body carryover factor.
            See SOURCES["aero_fin_afterbody_carryover"].
        area_nozzle_exit: nozzle exit area, m^2, used only for powered-base drag relief. When
            None a GUESSED fraction of the base area is used and flagged.
        nose_control: spline control values of the nose profile, when the OML is splined.
            `None` means the nose is the tangent ogive this model was written for, and the
            nose-shape wave-drag factor is exactly 1.0, so every pre-spline result is
            reproduced bit for bit. See `CD_wave_body` and SOURCES["aero_nose_shape_factor"].
    """

    def __init__(
        self,
        dv: DesignVector,
        meas: NtopMeasurements | None = None,
        *,
        nose_shape: str = "tangent_ogive",
        fin_max_thickness_station: float = 0.5,
        k_afterbody_carryover: float = 1.0,
        area_nozzle_exit: float | None = None,
        nose_control: Sequence[float] | None = None,
    ) -> None:
        self.dv = dv
        self.meas = meas
        self.k_afterbody_carryover = float(k_afterbody_carryover)
        self.sources_used: dict[str, str] = {}
        if nose_shape == "tangent_ogive" and getattr(dv, "nose_shape", None) == "spline":
            # The design vector, not the constructor default, is the authority on shape.
            nose_shape = "spline"
        self.geom = self._resolve_geometry(
            dv, meas, nose_shape, fin_max_thickness_station, area_nozzle_exit
        )
        if nose_control is None:
            nose_control = getattr(dv, "nose_control", None)
        self.nose_control = tuple(nose_control) if nose_control is not None else None
        self.nose_wave_shape_factor = self._resolve_nose_shape_factor(nose_shape)

    def _resolve_nose_shape_factor(self, nose_shape: str) -> float:
        """Nose wave drag of this profile divided by the tangent ogive's, at the same L/R.

        The ONLY route by which nose shape reaches the drag build-up. Returns exactly 1.0 when
        no spline control values were supplied, which is what keeps this addition invisible to
        the 296 tests that predate it.
        """
        if self.nose_control is None:
            self.sources_used["nose_wave_shape_factor"] = (
                "not applied: nose is the tangent ogive the Bonney correlation assumes"
            )
            return 1.0
        # imported here, not at module scope: `wavedrag` imports `oml_spline`, which imports
        # `config`, and a module-level import would make the aero/config import order matter.
        from .wavedrag import nose_wave_shape_ratio

        k = self.geom.L_nose / (0.5 * self.geom.D)
        ratio = nose_wave_shape_ratio(self.nose_control, k)
        self.sources_used["nose_wave_shape_factor"] = (
            f"slender-body (Glauert series) shape ratio {ratio:.6f} against the tangent "
            f"ogive at L/R = {k:.4f}; see wavedrag.SOURCES['wave_drag_applied_as_ratio']"
        )
        return ratio

    # ---------------------------------------------------------------- geometry

    def _resolve_geometry(
        self,
        dv: DesignVector,
        meas: NtopMeasurements | None,
        nose_shape: str,
        x_t_fin: float,
        area_nozzle_exit: float | None,
    ) -> _Geometry:
        prov = self.sources_used
        ANALYTIC = "analytic (DesignVector closed form)"
        NTOP = "nTop measured"

        D = dv.D
        S_ref = dv.S_ref
        L_total = dv.L_total
        L_nose = dv.L_nose
        L_bt = dv.L_boattail
        L_cyl = dv.L_body_cyl
        R = 0.5 * D

        # --- base area ---
        if meas is not None and meas.area_base is not None:
            S_base = float(meas.area_base)
            prov["area_base"] = NTOP
        else:
            S_base = dv.S_base
            prov["area_base"] = ANALYTIC
        r_base = math.sqrt(S_base / math.pi)
        beta_bt = math.atan2(R - r_base, L_bt) if L_bt > 0.0 else 0.0

        # --- analytic nose / cylinder / boattail breakdown, always computed as the fallback ---
        nose_control = getattr(dv, "nose_control", None)
        wet_nose, plan_nose, xbar_nose = _nose_wetted_and_planform(
            nose_shape, L_nose, D, nose_control
        )
        wet_cyl = math.pi * D * L_cyl
        plan_cyl = D * L_cyl
        xbar_cyl = L_nose + 0.5 * L_cyl
        plan_bt, xbar_bt_local = _frustum_planform_centroid(R, r_base, L_bt)
        slant_bt = math.hypot(L_bt, R - r_base)
        wet_bt = math.pi * (R + r_base) * slant_bt
        xbar_bt = L_nose + L_cyl + xbar_bt_local

        wet_analytic = wet_nose + wet_cyl + wet_bt
        plan_analytic = plan_nose + plan_cyl + plan_bt
        xcp_cross_analytic = (
            plan_nose * xbar_nose + plan_cyl * xbar_cyl + plan_bt * xbar_bt
        ) / plan_analytic

        # --- wetted body area ---
        if meas is not None and meas.area_wetted_body is not None:
            area_wetted_body = float(meas.area_wetted_body)
            prov["area_wetted_body"] = NTOP
        else:
            area_wetted_body = wet_analytic
            prov["area_wetted_body"] = ANALYTIC

        # --- planform area and the two body centres of pressure ---
        dist = list(meas.area_distribution) if meas is not None else []
        if len(dist) >= 3:
            plan_meas, xcp_cross = _planform_from_distribution(dist)
            _, xcp_pot = _dsdx_centroid_from_distribution(dist)
            area_planform_body = plan_meas
            x_cp_potential = xcp_pot
            x_cp_crossflow = xcp_cross
            prov["area_distribution"] = NTOP
            prov["area_planform_body"] = NTOP
            prov["x_cp_potential"] = NTOP + " (centroid of dS/dx)"
            prov["x_cp_crossflow"] = NTOP + " (planform centroid)"
        else:
            area_planform_body = plan_analytic
            # Net potential lift is the nose gain minus the boattail loss; place it at the
            # dS/dx-weighted centroid of those two contributions.
            f_nose = x_cp_nose_fraction(nose_shape, L_nose, D, nose_control)
            ds_nose = S_ref
            ds_bt = -(S_ref - S_base)
            den = ds_nose + ds_bt
            x_cp_potential = (
                (ds_nose * f_nose * L_nose + ds_bt * xbar_bt) / den
                if abs(den) > 1.0e-12
                else f_nose * L_nose
            )
            x_cp_crossflow = xcp_cross_analytic
            prov["area_distribution"] = "absent"
            prov["area_planform_body"] = ANALYTIC
            prov["x_cp_potential"] = ANALYTIC + f" (Barrowman {nose_shape} fraction)"
            prov["x_cp_crossflow"] = ANALYTIC + " (planform centroid)"

        # --- fins ---
        c_r = dv.c_r_fin
        c_t = dv.c_t_fin
        b_fin = dv.b_fin
        n_fin = int(dv.n_fin)
        lam = c_t / c_r if c_r > 0.0 else 0.0
        c_mac = (2.0 / 3.0) * c_r * (1.0 + lam + lam * lam) / (1.0 + lam)
        y_mac = (b_fin / 3.0) * (1.0 + 2.0 * lam) / (1.0 + lam)
        x_mac_le = y_mac * math.tan(dv.sweep_fin)

        if meas is not None and meas.area_wetted_fins is not None:
            area_wetted_fins = float(meas.area_wetted_fins)
            # Both sides of every panel, so the total exposed planform is half of it.
            S_fin_panel = area_wetted_fins / (2.0 * n_fin)
            prov["area_wetted_fins"] = NTOP
            prov["S_fin_panel"] = NTOP + " (area_wetted_fins / (2 n_fin))"
        else:
            S_fin_panel = dv.S_fin_exposed
            area_wetted_fins = 2.0 * n_fin * S_fin_panel
            prov["area_wetted_fins"] = ANALYTIC
            prov["S_fin_panel"] = ANALYTIC

        S_fin_pair = 2.0 * S_fin_panel
        ar_pair = (2.0 * b_fin) ** 2 / S_fin_pair if S_fin_pair > 0.0 else 0.0
        k_upwash = barrowman_upwash(R, R + b_fin)

        x_fin_le = dv.x_fin_le
        X_R = b_fin * math.tan(dv.sweep_fin)
        x_f_barrowman = (X_R / 3.0) * (c_r + 2.0 * c_t) / (c_r + c_t) + (1.0 / 6.0) * (
            c_r + c_t - c_r * c_t / (c_r + c_t)
        )
        x_cp_fin_sub = x_fin_le + x_f_barrowman
        x_cp_fin_sup = x_fin_le + x_mac_le + 0.5 * c_mac

        # --- nozzle exit area, only used for powered base drag ---
        if area_nozzle_exit is not None:
            a_e = float(area_nozzle_exit)
            guessed = False
            prov["area_nozzle_exit"] = "supplied by caller"
        else:
            a_e = NOZZLE_EXIT_AREA_FRACTION_GUESS * S_base
            guessed = True
            prov["area_nozzle_exit"] = "GUESS (fraction of base area)"

        prov["nose_shape"] = f"constructor argument: {nose_shape}"
        prov["k_afterbody_carryover"] = f"constructor argument: {self.k_afterbody_carryover}"

        return _Geometry(
            D=D, S_ref=S_ref, L_total=L_total, L_nose=L_nose, L_cyl=L_cyl, L_boattail=L_bt,
            S_base=S_base, area_wetted_body=area_wetted_body,
            area_planform_body=area_planform_body,
            x_cp_potential=x_cp_potential, x_cp_crossflow=x_cp_crossflow,
            beta_boattail=beta_bt, nose_shape=nose_shape,
            n_fin=n_fin, b_fin=b_fin, c_r=c_r, c_t=c_t, sweep_le=dv.sweep_fin, t_fin=dv.t_fin,
            x_fin_le=x_fin_le, x_t_fin=float(x_t_fin), S_fin_panel=S_fin_panel,
            area_wetted_fins=area_wetted_fins, c_mac=c_mac, x_mac_le=x_mac_le,
            aspect_ratio_pair=ar_pair, S_fin_pair=S_fin_pair, k_upwash=k_upwash,
            x_cp_fin_sub=x_cp_fin_sub, x_cp_fin_sup=x_cp_fin_sup,
            area_nozzle_exit=a_e, nozzle_exit_is_guess=guessed,
            provenance=dict(prov),
        )

    # ---------------------------------------------------------------- drag components

    @staticmethod
    def _flow(altitude: float, mach: float) -> tuple[float, float]:
        """(unit Reynolds number per metre, static temperature K) at one flight point.

        One atmosphere lookup shared by both friction terms. The atmosphere is table-driven
        and cached, but at 5 microseconds a call it is still worth doing once per evaluate.
        """
        st = atm.atmo(altitude)
        v = mach * st.speed_of_sound
        return st.density * v / st.dynamic_viscosity, st.temperature

    def CD_friction_body(
        self, mach: float, altitude: float, flow: tuple[float, float] | None = None
    ) -> float:
        """Compressible turbulent skin friction on the body wetted area, on S_ref.

        Sommer and Short T-prime method plus a body-of-revolution form factor.
        See SOURCES["aero_skin_friction"] and SOURCES["aero_body_form_factor"].

        `flow` is the optional pre-computed `(unit Reynolds number, temperature)` pair from
        `_flow`, so `evaluate` pays for one atmosphere lookup instead of two.
        """
        g = self.geom
        re_per_m, t_inf = flow if flow is not None else self._flow(altitude, mach)
        cf = cf_turbulent(re_per_m * g.L_total, mach, t_inf)
        ff = body_form_factor(g.L_total / g.D)
        return ff * cf * g.area_wetted_body / g.S_ref

    def CD_wave_body(self, mach: float) -> float:
        """Forebody wave drag, on S_ref.

        Supersonic: the Bonney correlation as published by Fleeman. Transonic: a cubic Hermite
        BLEND, not physics, from zero at M = 0.95 to the correlation value at M = 1.2.
        See SOURCES["aero_body_wave_drag"] and SOURCES["aero_transonic_bridge"].
        """
        g = self.geom
        # The nose base diameter is the body diameter, so the nose base area is S_ref and no
        # area referral is needed. `aero_iv1` has a nose narrower than its reference area and
        # therefore does need one.
        #
        # `nose_wave_shape_factor` is 1.0 unless a splined nose was supplied. Bonney depends on
        # fineness alone and cannot tell one nose from another at fixed fineness, so the shape
        # sensitivity comes from slender-body theory as a RATIO and the calibrated Bonney level
        # and its Mach dependence are left alone. See SOURCES["aero_nose_shape_factor"].
        return bonney_nose_wave_cd(g.L_nose / g.D, mach) * self.nose_wave_shape_factor

    def CD_wave_body_crosscheck(self, mach: float) -> float:
        """Second wave-drag correlation, reported but never summed into CD0.

        Sears-Haack type III equivalent body. See SOURCES["aero_wave_drag_crosscheck"].
        """
        g = self.geom
        a_max = g.S_ref            # maximum cross-section area is the reference area by definition
        cd = 1.5 * math.pi * math.pi * (g.D / g.L_total) ** 2 * (a_max / g.S_ref)
        return (1.0 - cubic_blend(mach, M_BRIDGE_LO, M_BRIDGE_HI)) * cd

    def CD_base(self, mach: float, power_on: bool = False) -> float:
        """Base drag on the base area, on S_ref. See SOURCES["aero_base_drag"]."""
        g = self.geom
        cd = base_cd_on_base_area(mach) * g.S_base / g.S_ref
        if power_on:
            relief = 1.0 - g.area_nozzle_exit / g.S_base
            cd *= max(relief, 0.0)
        return cd

    def CD_boattail(self, mach: float) -> float:
        """Boattail pressure drag on the projected annulus, on S_ref.

        Exact Prandtl-Meyer expansion through the boattail half-angle, blended to zero
        subsonically. Pressure recovery is not modelled, so this is an upper bound.
        See SOURCES["aero_boattail_drag"].
        """
        g = self.geom
        if g.L_boattail <= 0.0 or g.beta_boattail <= 0.0:
            return 0.0
        annulus = (g.S_ref - g.S_base) / g.S_ref
        if annulus <= 0.0:
            return 0.0
        m1 = max(mach, M_BRIDGE_HI)
        cd_super = -expansion_turn_cp(m1, g.beta_boattail) * annulus
        return (1.0 - cubic_blend(mach, M_BRIDGE_LO, M_BRIDGE_HI)) * max(cd_super, 0.0)

    def CD_fin_friction(
        self, mach: float, altitude: float, flow: tuple[float, float] | None = None
    ) -> float:
        """Skin friction on the fin wetted area, on S_ref.

        Reynolds number on the mean aerodynamic chord. Thickness form factor from Raymer.
        See SOURCES["aero_skin_friction"] and SOURCES["aero_fin_form_factor"].
        """
        g = self.geom
        if g.area_wetted_fins <= 0.0:
            return 0.0
        re_per_m, t_inf = flow if flow is not None else self._flow(altitude, mach)
        cf = cf_turbulent(re_per_m * g.c_mac, mach, t_inf)
        ff = surface_form_factor(g.t_fin / g.c_mac, g.x_t_fin)
        return ff * cf * g.area_wetted_fins / g.S_ref

    def CD_fin_wave(self, mach: float) -> float:
        """Fin thickness (wave) drag, on S_ref.

        Ackeret linearised double-wedge thickness drag above M = 1.2, blended to zero at
        M = 0.95. Leading-edge sweep relief is not credited.
        See SOURCES["aero_fin_wave_drag"] and SOURCES["aero_transonic_bridge"].
        """
        g = self.geom
        if g.n_fin < 1 or g.S_fin_panel <= 0.0:
            return 0.0
        cd_2d = fin_wave_cd_2d(g.t_fin / g.c_mac, g.x_t_fin, mach)
        return cd_2d * g.n_fin * g.S_fin_panel / g.S_ref

    def CD_fin_wave_crosscheck(self, mach: float) -> float:
        """Newtonian-impact fin leading-edge drag. Reported, never summed into CD0.

        See SOURCES["aero_fin_wave_drag_crosscheck"].
        """
        g = self.geom
        if g.n_fin < 1 or g.S_fin_panel <= 0.0:
            return 0.0
        return newtonian_le_cd(
            n_pairs=g.n_fin / 2.0,
            sweep_le=g.sweep_le,
            t_max=g.t_fin,
            x_t=g.x_t_fin,
            c_mac=g.c_mac,
            span=2.0 * (0.5 * g.D + g.b_fin),
            S_ref=g.S_ref,
            mach=mach,
        )

    def CD_protuberance(self, cd_clean: float) -> float:
        """Lumped protuberance allowance. THIS IS A GUESS.

        See SOURCES["aero_protuberance_GUESS"].
        """
        return PROTUBERANCE_FRACTION_GUESS * cd_clean

    # ---------------------------------------------------------------- normal force

    def CN_body(self, mach: float, alpha: float) -> tuple[float, float]:
        """(potential term, cross-flow term) of the body normal force, on S_ref.

        Slender-body potential lift plus the Allen and Perkins viscous cross-flow term.
        No Mach dependence: slender-body theory has none, and the published closed form this
        follows has none either. See SOURCES["aero_body_normal_force"].
        """
        g = self.geom
        return body_normal_force_terms(g.S_base, g.S_ref, g.area_planform_body, alpha)

    def CN_fins(self, mach: float, alpha: float) -> float:
        """Fin normal force including interference, on S_ref.

        Linearised supersonic theory above the branch Mach number, slender-wing theory below,
        blended across it. Multiplied by the body-upwash factor and the afterbody carryover
        factor. See SOURCES["aero_fin_normal_force"], SOURCES["aero_fin_body_upwash"] and
        SOURCES["aero_fin_afterbody_carryover"].
        """
        g = self.geom
        if g.n_fin < 2 or g.S_fin_pair <= 0.0 or g.aspect_ratio_pair <= 0.0:
            return 0.0
        cn_alone = (
            lifting_surface_cn_alone(g.aspect_ratio_pair, mach, alpha)
            * g.S_fin_pair
            / g.S_ref
        )
        return cn_alone * g.k_upwash * self.k_afterbody_carryover

    def x_cp_fins(self, mach: float) -> float:
        """Fin centre of pressure from the nose tip, m. See SOURCES["aero_fin_cp"]."""
        g = self.geom
        w = cubic_blend(mach, M_BRIDGE_LO, M_BRIDGE_HI)
        return w * g.x_cp_fin_sub + (1.0 - w) * g.x_cp_fin_sup

    # ---------------------------------------------------------------- assembly

    def evaluate(
        self,
        mach: float,
        altitude: float,
        alpha: float,
        power_on: bool = False,
    ) -> AeroCoefficients:
        """Full aerodynamic state at one flight point.

        Args:
            mach: free-stream Mach number. Valid 0.3 to 5.0; outside that the model still
                returns a value but `breakdown['out_of_validity_range']` is set to 1.
            altitude: geometric altitude, m.
            alpha: angle of attack, rad. Valid 0 to 15 deg in magnitude.
            power_on: True while the motor is thrusting, which reduces base drag.
        """
        g = self.geom
        mach = float(mach)
        alpha = float(alpha)

        # --- zero-lift components ---
        flow = self._flow(altitude, mach)
        cd_fric_body = self.CD_friction_body(mach, altitude, flow)
        cd_wave_body = self.CD_wave_body(mach)
        cd_base = self.CD_base(mach, power_on)
        cd_boattail = self.CD_boattail(mach)
        cd_fric_fin = self.CD_fin_friction(mach, altitude, flow)
        cd_wave_fin = self.CD_fin_wave(mach)
        cd_clean = (
            cd_fric_body + cd_wave_body + cd_base + cd_boattail + cd_fric_fin + cd_wave_fin
        )
        cd_prot = self.CD_protuberance(cd_clean)
        CD0 = cd_clean + cd_prot

        # --- normal force ---
        cn_pot, cn_cross = self.CN_body(mach, alpha)
        cn_fin = self.CN_fins(mach, alpha)
        CN = cn_pot + cn_cross + cn_fin

        # --- centre of pressure, moment about the nose tip ---
        x_fin = self.x_cp_fins(mach)
        if abs(CN) > 1.0e-12:
            x_cp = (
                cn_pot * g.x_cp_potential + cn_cross * g.x_cp_crossflow + cn_fin * x_fin
            ) / CN
        else:
            # alpha -> 0 limit: weight by the alpha -> 0 slopes of each component.
            eps = 1.0e-4
            p0, c0 = self.CN_body(mach, eps)
            f0 = self.CN_fins(mach, eps)
            tot = p0 + c0 + f0
            x_cp = (
                (p0 * g.x_cp_potential + c0 * g.x_cp_crossflow + f0 * x_fin) / tot
                if abs(tot) > 1.0e-15
                else g.x_cp_potential
            )
        CM = -CN * x_cp / g.D

        # --- wind-axis drag and lift ---
        ca, sa = math.cos(alpha), math.sin(alpha)
        cd_induced = CN * sa
        cd_axial_proj = CD0 * (ca - 1.0)
        CD = CD0 * ca + cd_induced
        CL = CN * ca - CD0 * sa
        l_over_d = CL / CD if abs(CD) > 1.0e-12 else 0.0

        # --- CN_alpha as a secant slope, which is what test reductions report ---
        if abs(alpha) > 1.0e-6:
            CN_alpha = CN / alpha
        else:
            eps = 1.0e-4
            p0, c0 = self.CN_body(mach, eps)
            CN_alpha = (p0 + c0 + self.CN_fins(mach, eps)) / eps

        breakdown = {
            "CD_friction_body": cd_fric_body,
            "CD_wave_body": cd_wave_body,
            "CD_base": cd_base,
            "CD_boattail": cd_boattail,
            "CD_fin_friction": cd_fric_fin,
            "CD_fin_wave": cd_wave_fin,
            "CD_protuberance_GUESS": cd_prot,
            "CD_induced": cd_induced,
            "CD_axial_incidence_projection": cd_axial_proj,
            # reported, not summed
            "xcheck_CD_wave_body_sears_haack": self.CD_wave_body_crosscheck(mach),
            "xcheck_CD_fin_wave_newtonian": self.CD_fin_wave_crosscheck(mach),
            "CN_body_potential": cn_pot,
            "CN_body_crossflow": cn_cross,
            "CN_fins": cn_fin,
            "x_cp_potential_m": g.x_cp_potential,
            "x_cp_crossflow_m": g.x_cp_crossflow,
            "x_cp_fins_m": x_fin,
            "x_cp_over_D": x_cp / g.D,
            "power_on": 1.0 if power_on else 0.0,
            "n_quantities_from_ntop": float(
                sum(1 for v in self.sources_used.values() if v.startswith("nTop measured"))
            ),
            "n_guessed_quantities": float(
                sum(1 for v in self.sources_used.values() if "GUESS" in v)
            ),
            "out_of_validity_range": float(
                not (MACH_MIN_VALID <= mach <= MACH_MAX_VALID)
                or abs(alpha) > ALPHA_MAX_VALID
            ),
        }

        return AeroCoefficients(
            mach=mach,
            altitude=altitude,
            alpha=alpha,
            CD0=CD0,
            CD=CD,
            CN=CN,
            CN_alpha=CN_alpha,
            CM=CM,
            x_cp=x_cp,
            L_over_D=l_over_d,
            breakdown=breakdown,
        )

    # ---------------------------------------------------------------- trim

    def trim_alpha(
        self,
        mach: float,
        altitude: float,
        required_CN: float,
        power_on: bool = False,
        alpha_max: float = ALPHA_MAX_VALID,
    ) -> float:
        """Angle of attack, rad, that produces `required_CN`.

        CN is monotone increasing in alpha over 0 to 15 deg, so a bracketed bisection with a
        secant accelerator converges in a handful of iterations. Returns `alpha_max` (or
        `-alpha_max`) when the required CN is beyond what the configuration can produce, so the
        sizing loop sees a saturated control rather than an exception.
        """

        def cn_of(a: float) -> float:
            p, c = self.CN_body(mach, a)
            return p + c + self.CN_fins(mach, a)

        return solve_alpha_for_cn(cn_of, float(required_CN), alpha_max)
