"""Two-stage, strake-stabilised aerodynamic build-up for IV-1.

Read `SPEC_IV1.md` sections 3 and 8 first, then `aero.py`. This module does NOT reimplement the
single-body build-up. It imports the validated kernels out of `aero.py` and assembles them for a
configuration that has two body diameters, two fin sets and four body-mounted strakes.

What `stage` means
------------------
`stage` selects a FLIGHT CONFIGURATION, not a component:

    stage = 1   the full stack, before separation. The booster diameter sets `S_ref`. The nose,
                both cylindrical bodies, the interstage shoulder, BOTH fin sets and the strakes
                all contribute.
    stage = 2   the payload stage alone, after separation. Its own diameter sets `S_ref`, and
                only its nose, its body, its own fins and the strakes contribute.

The reference area therefore CHANGES at separation. A coefficient computed on the wrong area
silently corrupts the trajectory, so `S_ref(stage)` is public and every coefficient this module
returns is referenced to `S_ref(stage)` of the stage it was asked for.

Axial station convention
------------------------
Every station `x` is measured aft from the PAYLOAD-STAGE NOSE TIP, in metres, in BOTH
configurations. That is the tip of the whole stack before separation and the tip of the surviving
vehicle after it, so the strake and stage-2 fin stations are identical in the two configurations
and nothing has to be re-datumed at separation. Stack layout, front to back:

    0 .............. L_nose ....... L2 ......... L2+L_is ................ L2+L_is+L1
    |---- ogive ----|--- cyl D2 ---|-- shoulder --|------- cyl D1 -------|
                    strakes on the stage-2 mid-body       stage-1 fins near the base
                    stage-2 fins at the stage-2 aft end

What is new here, and what is inherited
---------------------------------------
Inherited unchanged from `aero.py`, and therefore still covered by the 23 Basic Finner
free-flight shots: compressible skin friction (Sommer and Short), the body form factor, the
Bonney forebody wave-drag correlation with its transonic bridge, base drag, the exact
Prandtl-Meyer boattail expansion, the Raymer surface form factor, Ackeret double-wedge thickness
drag, slender-body plus Allen-Perkins body normal force, the two-branch lifting-surface normal
force with Barrowman body upwash, and the trim solver.

New in this module:
  1. Strake normal force by the Polhamus leading-edge suction analogy, extended to the
     side-edge-dominated case. This is the important addition: the default strake has an exposed
     aspect ratio near 0.02, where linear theory gives essentially nothing and the whole load is
     vortex lift. See SOURCES["iv1_strake_polhamus"].
  2. Strake friction, thickness and blunt-trailing-edge drag, and the reason the junction
     interference is deliberately left at zero. See SOURCES["iv1_strake_drag"] and
     SOURCES["iv1_strake_junction_interference"].
  3. Strake centre of pressure, near the strake mid-chord and therefore far forward of the tail
     fins, so strakes REDUCE static margin. See SOURCES["iv1_strake_cp"].
  4. Interstage shoulder wave drag by an exact isentropic compression turn.
     See SOURCES["iv1_interstage_shoulder"].
  5. `CN_max` at an alpha limit, for requirement A11.

NOT modelled, and the direction each omission pushes:
  * Strake-shed vortex interference on the tail fins. Left at 1.0.
    See SOURCES["iv1_strake_fin_interference"].
  * A boattail on either stage. IV-1 has none in the design vector.
  * Stage-1 fin wake on the stage-2 body. There is none: stage 1 is aft of stage 2.
  * Any separation transient. `SPEC_IV1.md` section 8 makes separation instantaneous.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import AeroCoefficients, NtopMeasurements, register_sources
from ..config_iv1 import InterceptRequirements, StackDesignVector, StageSpec, StrakeSpec
from . import atmosphere as atm
from .aero import (
    M_BRIDGE_HI,
    M_BRIDGE_LO,
    NOZZLE_EXIT_AREA_FRACTION_GUESS,
    PROTUBERANCE_FRACTION_GUESS,
    X_CP_NOSE_FRACTION,
    _cp_max_stagnation,
    _frustum_planform_centroid,
    _nose_wetted_and_planform,
    _planform_from_distribution,
    barrowman_upwash,
    base_cd_on_base_area,
    body_form_factor,
    body_normal_force_terms,
    bonney_nose_wave_cd,
    cf_turbulent,
    compression_turn_cp,
    cubic_blend,
    fin_wave_cd_2d,
    lifting_surface_cn_alone,
    oblique_shock_cp,
    solve_alpha_for_cn,
    surface_form_factor,
)

# --------------------------------------------------------------------------------------
#   Sources. CLAUDE.md hard rule 3.1: no invented numbers.
#
#   Keys are prefixed `iv1_aero_` so they cannot collide with `aero.py`. Everything this
#   module inherits keeps its original `aero_*` key and is not restated here.
# --------------------------------------------------------------------------------------

SOURCES: dict[str, str] = {
    "iv1_aero_reuse": (
        "The single-body physics is imported from `rocketgen.sizing.aero`, not reimplemented: "
        "skin friction, body and fin form factors, the Bonney forebody wave-drag correlation, "
        "base drag, the Prandtl-Meyer boattail expansion, Ackeret double-wedge thickness drag, "
        "slender-body plus Allen-Perkins body normal force, the two-branch lifting-surface "
        "normal force, Barrowman body upwash and the trim solver. Those are validated against "
        "the 23 Basic Finner free-flight shots of Dupuis and Hathaway, DREV-TM-9703 (1997), "
        "Table VII; see SOURCES['aero_validation'] and tests/test_aero.py. "
        "tests/test_aero_iv1.py asserts that a single-stage, strake-free StackAero reproduces "
        "RocketAero to machine precision, which is what makes that inheritance real."
    ),
    "iv1_aero_strake_polhamus": (
        "Strake normal force by the Polhamus leading-edge suction analogy: "
        "CN = K_p sin(a) cos^2(a) + K_v sin^2(a) cos(a). "
        "E. C. Polhamus, 'A Concept of the Vortex Lift of Sharp-Edge Delta Wings Based on a "
        "Leading-Edge-Suction Analogy', NASA TN D-3767 (1966), eq. (15); and 'Predictions of "
        "Vortex-Lift Characteristics by a Leading-Edge Suction Analogy', J. Aircraft 8(4), 1971, "
        "pp. 193-199, eq. (4). K_p is the potential-flow normal-force slope per radian and K_v "
        "the vortex-lift factor; see SOURCES['iv1_aero_strake_Kp'] and "
        "SOURCES['iv1_aero_strake_Kv'] for where each number comes from. "
        "VALIDITY AND EXTENSION, stated plainly. Polhamus developed and validated the analogy "
        "for SHARP-EDGED DELTA WINGS of aspect ratio 0.5 to 4, at subsonic speed, up to the "
        "angle of vortex breakdown; TN D-3767 shows good agreement to roughly 25 deg for the "
        "slender cases and notes lift falling below the theory above about 18 deg for its "
        "aspect-ratio-2 wing. An IV-1 strake is a RECTANGULAR, UNSWEPT, BODY-MOUNTED surface of "
        "pair aspect ratio about 0.04. Applying the analogy to it is therefore an EXTENSION on "
        "three counts, each handled explicitly rather than ignored: "
        "(1) PLANFORM. The vortex comes off the streamwise side edge, not a swept leading edge, "
        "so the Lamar-Polhamus side-edge term is used instead of the delta-wing K_v curve. "
        "Reproducing the delta-wing K_v (about 3.14 at A = 0 rising to 3.45 at A = 4, "
        "TN D-3767 p. 10) would be the wrong number for this planform, although it happens to be "
        "numerically close. "
        "(2) ASPECT RATIO. Two decades below Polhamus's delta-wing range, so both coefficients "
        "come from the rectangular-wing values of NASA TN D-7921 Table III, which starts at "
        "A = 0.05 and is therefore AT the strake regime rather than extrapolated into it. "
        "(3) MOUNTING. On a body, in the body cross-flow, not isolated. The potential term "
        "carries a body-upwash factor; the vortex term does not, and the body-load enhancement "
        "is not modelled at all. See SOURCES['iv1_aero_strake_upwash'] and "
        "SOURCES['iv1_aero_strake_body_interference_on_body_load']. "
        "It OMITS vortex breakdown, compressibility of the vortex core, the strake-to-strake "
        "interaction of a cruciform set, and any Mach dependence at all: the same K_p and K_v "
        "are used at every Mach number, which measured data show is wrong subsonically."
    ),
    "iv1_aero_strake_Kp": (
        "K_p, the potential-flow term of the suction analogy, is the slender-wing lift-curve "
        "slope pi*A/2 per radian, with A the aspect ratio of the STRAKE PAIR, (2b)^2/S_pair, "
        "and the result referred to the pair exposed area. That is the same panel-pair "
        "convention `aero.py` uses for the tail fins, and it is the reflection-plane result: a "
        "body-mounted panel of span b behaves as half of a wing of span 2b. "
        "R. T. Jones, 'Properties of Low-Aspect-Ratio Pointed Wings at Speeds Below and Above "
        "the Speed of Sound', NACA Report 835 (1946); the same limit appears as K_p = pi*A/(2E) "
        "with E -> 1 in the slender limit in Polhamus, J. Aircraft 8(4), 1971, eq. (6). "
        "CONFIRMED AGAINST A PRINTED TABLE: NASA TN D-7921 (Lamar and Gloss, 1975) Table III "
        "lists K_p for rectangular wings, and pi*A/2 reproduces the tabulated value to 0.13 "
        "percent at A = 0.05 and 0.01 percent at A = 0.10. Mach-independent, which is the "
        "correct slender limit on both sides of M = 1. At the default strake pair aspect ratio "
        "of 0.043 it gives K_p = 0.067 per radian, which is why linear theory alone cannot size "
        "a strake."
    ),
    "iv1_aero_strake_Kv": (
        "K_v, the vortex-lift factor, is the SUM of a leading-edge and a SIDE-EDGE term, "
        "K_v = K_v,le + K_v,se, because a strake is rectangular and unswept: its shed vortex "
        "comes off the streamwise tip edge, not off a swept leading edge. This is the "
        "Lamar-Polhamus extension of the suction analogy to non-delta planforms: "
        "J. E. Lamar, 'Prediction of Vortex Flow Characteristics of Wings at Subsonic and "
        "Supersonic Speeds', NASA TR R-428 (1974); J. E. Lamar and B. B. Gloss, NASA TN D-7921 "
        "(1975). "
        "K_v,le = pi*A/4 = K_p/2, which TN D-7921 Table III reproduces to better than 0.1 "
        "percent for every rectangular entry from A = 0.05 to A = 0.40. "
        "K_v,se is taken directly from the TN D-7921 Table III rectangular-wing column (the m1 "
        "column, the TR R-428 continuous-loading method that the authors take as the standard), "
        "interpolated in A and anchored at A = 0 by the analytic limit K_v,se -> pi stated in "
        "TN D-7921 p. 9 and TR R-428 p. 19. See LAMAR_KV_SE_RECTANGULAR for the table. At the "
        "default strake pair aspect ratio of 0.043 that gives K_v = 3.19. "
        "NOTE, because it matters: the value is close to pi, NOT to the 2.0 that Bollay's "
        "zero-aspect-ratio nonlinear wing theory (ZAMM 19, 1939) gives and that the Newtonian "
        "'2 sin^2 a' term in SOURCES['aero_fin_normal_force'] uses. The suction analogy is "
        "about 60 percent higher. That is exactly why a strake needs its own method instead of "
        "being pushed through the inherited fin routine. "
        "APPROXIMATIONS, both of which push the strake normal force DOWN: the interpolation is "
        "linear in A, so it is C0 but not C1 with respect to the strake design variables (it is "
        "exactly constant with respect to Mach and alpha, so the trajectory integrator is "
        "unaffected); and a body-mounted strake sits in the accelerated cross-flow around the "
        "body, where the real vortex is stronger than on the isolated plate the table describes. "
        "OMITS vortex breakdown, all Mach dependence, and the strake-to-strake interaction of a "
        "cruciform set."
    ),
    "iv1_aero_strake_body_interference_on_body_load": (
        "NOT MODELLED, and it is the largest known omission of the strake normal force. A strake "
        "fixes the separation line on the body and organises the leeside vortices, which raises "
        "the load carried by the BODY as well as the load carried by the strake. This model adds "
        "only the strake's own lifting-surface load. Measured evidence, and the only "
        "configuration data found for a body with strakes: L. H. Jorgensen and E. R. Nelson, "
        "'Experimental Aerodynamic Characteristics for a Cylindrical Body of Revolution With "
        "Side Strakes and Various Noses at Angles of Attack From 0 to 58 deg and Mach Numbers "
        "From 0.6 to 2.0', NASA TM X-3130 (1975), figures 18 and 22, whose summary states that "
        "removing the strakes 'greatly decreased the lift'. Digitised strake-on minus "
        "strake-off increments from those figures are 1.4 to 3.4 times what this model's "
        "strake-alone term gives, on the three rows where both curves are sampled at the same "
        "angle of attack (M 0.6 at 30 and 58 deg, M 2.0 at 58 deg). The measured increment grows "
        "as Mach falls, while this model has no Mach dependence at all. "
        "CONSEQUENCE, with its sign: "
        "CN_max, and therefore the requirement A11 lateral acceleration, is CONSERVATIVE. "
        "See tests/test_aero_iv1.py for the digitised numbers, their uncertainty and the "
        "measured ratio, which the suite pins so the model cannot silently start overshooting."
    ),
    "iv1_aero_strake_upwash": (
        "The Barrowman body-upwash factor 1 + a/s (a = body radius, s = strake tip radius) is "
        "applied to the strake POTENTIAL term only, reusing SOURCES['aero_fin_body_upwash']. "
        "It is a linear potential-flow result, so it is not applied to the vortex-lift term: "
        "there is no published upwash factor for a separated-flow load, and the suction analogy "
        "already integrates the separated load. APPROXIMATION with a known sign: the real "
        "cross-flow acceleration around the body raises the effective incidence of the strake "
        "panel, so leaving the vortex term un-augmented makes the strake normal force a LOWER "
        "BOUND. Because the potential term is negligible at strake aspect ratios, this choice "
        "moves the answer by well under 1 percent."
    ),
    "iv1_aero_strake_cp": (
        "Strake centre of pressure at the strake mid-chord. Reasoning, not a fitted constant: "
        "(1) slender-body theory puts the potential load where the cross-section area changes, "
        "so for a CONSTANT-HEIGHT strake the potential load is a pair of equal and opposite "
        "impulses at the leading and trailing edges whose net is zero, and its station is "
        "immaterial; (2) the side-edge suction that the analogy converts into vortex lift is "
        "distributed along the streamwise edge, and for a constant-height strake that edge "
        "suction is nearly uniform in x, so the vortex load centroid falls at mid-chord. "
        "Exposed as the constructor argument `strake_cp_chord_fraction` so a DATCOM or measured "
        "value can replace it. OMITS the aft growth of the vortex, which would move the load "
        "slightly aft and therefore slightly INCREASE static margin."
    ),
    "iv1_aero_strake_drag": (
        "Strake drag reuses the fin machinery where the physics is identical and says where it "
        "is not. Friction: the same Sommer and Short compressible flat-plate law and the same "
        "Raymer thickness form factor as the fins, with the Reynolds number on the strake "
        "LENGTH, because the strake length is its streamwise chord. Identical physics, so "
        "SOURCES['aero_skin_friction'] and SOURCES['aero_fin_form_factor'] apply unchanged. "
        "Thickness wave drag: the same Ackeret double-wedge result, SOURCES "
        "['aero_fin_wave_drag']. NOT identical physics, and the difference is declared: a real "
        "strake is a constant-thickness plate with a short leading-edge bevel, not a double "
        "wedge peaking at mid-chord. Concentrating the same thickness into a bevel a few "
        "thicknesses long raises the local surface slope by more than an order of magnitude and "
        "the wave drag with it, so the double-wedge assumption UNDERSTATES strake wave drag. "
        "The opposite end is reported as `xcheck_CD_strake_le_blunt_bound`: stagnation pressure "
        "on the whole strake frontal area n*thickness*height, which is what a fully blunt "
        "unbeveled leading edge would cost. The two BRACKET the truth, and at the default "
        "geometry the bracket is wide, more than two orders of magnitude, because the "
        "double-wedge t/c of 0.0057 spread over a 1.4 m chord is an almost flat surface while a "
        "blunt edge is not. Where the real strake falls inside that bracket depends on the "
        "leading-edge bevel length, which is NOT an IV-1 design variable, so this model reports "
        "the optimistic end plus the bound rather than interpolating with an invented bevel "
        "length. Trailing-edge drag: the strake trailing edge is blunt and its base area is "
        "charged with the same base-pressure correlation as the body base, "
        "SOURCES['aero_base_drag']; that correlation was fitted to an AXISYMMETRIC body base, "
        "so applying it to a small two-dimensional trailing edge is an EXTENSION, taken because "
        "leaving the term out entirely is the larger error."
    ),
    "iv1_aero_strake_junction_interference": (
        "Strake-to-body junction interference drag is set to ZERO here, deliberately, to avoid "
        "double counting. config.SOURCES['cd0_calibration'] already attributes part of the "
        "measured 14.6 percent CD0 shortfall against the Basic Finner free-flight data to "
        "'fin-body junction interference', and that calibration is applied at the sizing-loop "
        "boundary through CalibratedAero, per CLAUDE.md section 8. Modelling the junction again "
        "inside this module would charge it twice. The consequence is that the UNCALIBRATED CD0 "
        "returned by this module is low by the same order as it is for SV-1."
    ),
    "iv1_aero_strake_fin_interference": (
        "Strake-to-fin vortex interference is NOT MODELLED. The factor "
        "`k_strake_fin_interference` is 1.0. No factor for this configuration was sourced, and "
        "CLAUDE.md hard rule 3.1 forbids inventing one. Expected sign of the error: the strakes "
        "and the tail fins are both cruciform and in line, so each tail panel sits directly in "
        "the wake of the strake ahead of it. A shed vortex inboard of and above a following "
        "panel induces DOWNWASH over that panel, which reduces its effective incidence and its "
        "normal force. This model therefore likely OVERPREDICTS tail-fin normal force at "
        "moderate to high alpha, which in turn OVERPREDICTS the aft load and so OVERPREDICTS "
        "static margin. The magnitude is unquantified. Exposed as a constructor argument so a "
        "measured or DATCOM value can be supplied without touching the model."
    ),
    "iv1_aero_interstage_shoulder": (
        "The interstage is a conical shoulder from the stage-2 diameter out to the stage-1 "
        "diameter, so it is a FLARE ON A CYLINDER, not a nose cone. Its pressure drag is the "
        "exact weak oblique shock through the shoulder half-angle, applied as +Cp on the "
        "projected annulus (S1 - S2). That is the TANGENT-WEDGE estimate; see "
        "SOURCES['aero_oblique_shock'] for the closed-form solution and its textbook "
        "verification, and SOURCES['aero_transonic_bridge'] for the blend to zero below "
        "M 0.95. THIS IS THE LARGEST SINGLE DRAG TERM OF THE STACKED CONFIGURATION at the "
        "default design point, because the projected annulus is about half the reference area, "
        "so the choice of method matters and is stated here in full. "
        "Known bias and its direction: tangent-wedge takes no credit for the three-dimensional "
        "relief of an axisymmetric surface, so it gives a HIGHER pressure than the tangent-cone "
        "(Taylor-Maccoll) estimate at the same deflection. Anderson, Modern Compressible Flow, "
        "chapter on conical flow, gives the reason. Measured flare pressures fall between the "
        "two, because the flow immediately behind the corner has not yet felt the relief. The "
        "value used here is therefore an UPPER BOUND on the shoulder drag, which is "
        "conservative for range. The magnitude of the excess is NOT quantified: this module "
        "does not integrate the Taylor-Maccoll cone equation. Two brackets are reported instead "
        "and never summed: `xcheck_CD_shoulder_isentropic`, the same turn taken isentropically, "
        "which shows the method is insensitive to the shock entropy rise above about M 2; and "
        "`xcheck_CD_shoulder_stagnation`, a hard upper bracket that no shock system can exceed. "
        "ALSO OMITS: the detached-shock overshoot below the detachment Mach number, where the "
        "value is clamped at the maximum attached deflection and is a LOWER bound (the "
        "breakdown key `shoulder_shock_attached` flags this); and pressure recovery on the "
        "booster cylinder behind the shoulder. The shoulder WETTED area is carried in the body "
        "friction term and the shoulder PLANFORM area in the body cross-flow term, so neither "
        "of those contributions is neglected."
    ),
    "iv1_aero_fin_station": (
        "MODELLING CHOICE, not a measurement: each stage's tail fins are placed with their root "
        "TRAILING EDGE at the aft end of that stage, so the stage-1 fins end at the booster "
        "base and the stage-2 fins end at the stage-2 to interstage joint. `StageSpec` carries "
        "no fin station, and SPEC_IV1.md section 3 shows tail fins at the aft end of each "
        "stage. Exposed as the constructor argument `fin_te_gap` so a real gap can be set. The "
        "consequence for stability is that the stage-2 fins are about one interstage length "
        "further forward in the stacked configuration than the booster fins, which is exactly "
        "the geometry the static-margin check needs."
    ),
    "iv1_aero_no_boattail": (
        "Neither IV-1 stage carries a boattail: `StageSpec` has no base-diameter or "
        "boattail-length field, and the base area defaults to the full body cross-section of "
        "the aft-most body in the configuration. The boattail machinery in `aero.py` is "
        "therefore inherited but evaluates to zero. OMITS: if a measurement supplies an "
        "`area_base` smaller than the body cross-section, the geometry implies a boattail whose "
        "pressure drag is NOT added, which is optimistic; the model records that in "
        "`sources_used` rather than inventing a boattail angle to go with it."
    ),
    "iv1_aero_validity": (
        "Stated validity: M 0.3 to 5.0 and |alpha| up to 25 deg. The Mach range is inherited "
        "from `aero.py`, whose validation data (Dupuis and Hathaway, DREV-TM-9703) reaches "
        "M 4.47. The alpha range is wider than the 15 deg of `aero.py` because IV-1 requirement "
        "A11 is evaluated at 20 deg: the Allen and Perkins cross-flow term is applied well "
        "beyond 25 deg in Jorgensen, NASA TR R-474 (1977), and Polhamus reports the suction "
        "analogy holding to roughly 25 to 35 deg for slender surfaces, so 25 deg is inside both "
        "constituent methods. The LIMITER is the linearised fin theory, which has no stall. "
        "Outside either range `evaluate` still returns a smooth value, so the trajectory "
        "integrator never sees a cliff, but sets breakdown['out_of_validity_range'] to 1."
    ),
}
register_sources(SOURCES)


# --------------------------------------------------------------------------------------
#   Constants
# --------------------------------------------------------------------------------------

#: Polhamus potential-flow factor: K_p = KP_SLENDER_COEFF * A. See SOURCES["iv1_aero_strake_Kp"].
KP_SLENDER_COEFF: float = math.pi / 2.0

#: Leading-edge vortex-lift factor of a rectangular surface: K_v,le = KV_LE_COEFF * A = K_p / 2.
#: See SOURCES["iv1_aero_strake_Kv"].
KV_LE_COEFF: float = math.pi / 4.0

#: Side-edge vortex-lift factor of a RECTANGULAR wing against aspect ratio, `(A, K_v,se)`.
#:
#: NASA TN D-7921 (J. E. Lamar and B. B. Gloss, 1975), Table III, rectangular-wing rows, m1
#: column, M = 0. The A = 0 entry is not from the table: it is the analytic limit
#: K_v,se -> pi stated in TN D-7921 p. 9 and NASA TR R-428 p. 19, which anchors the low end
#: where a strake actually lives. See SOURCES["iv1_aero_strake_Kv"].
#:
#: The table exists ONCE, here, because the model consumes it. Only this column is used;
#: tests/test_aero_iv1.py checks the closed forms for K_p and K_v,le against the OTHER two
#: columns of the same published table, which the model never reads.
LAMAR_KV_SE_RECTANGULAR: tuple[tuple[float, float], ...] = (
    (0.00, math.pi),
    (0.05, 3.1799),
    (0.10, 3.0188),
    (0.20, 2.7913),
    (0.30, 2.7208),
    (0.40, 2.6341),
    (1.00, 2.1255),
)

#: Strake load centroid as a fraction of strake length. See SOURCES["iv1_aero_strake_cp"].
STRAKE_CP_CHORD_FRACTION: float = 0.5

#: Chordwise station of maximum thickness for the strake double-wedge model.
STRAKE_X_T: float = 0.5

MACH_MIN_VALID: float = 0.3
MACH_MAX_VALID: float = 5.0

#: See SOURCES["iv1_aero_validity"]. Wider than `aero.ALPHA_MAX_VALID` on purpose.
ALPHA_MAX_VALID_IV1: float = math.radians(25.0)

ANALYTIC: str = "analytic (StackDesignVector closed form)"
NTOP: str = "nTop measured"


def polhamus_kp(aspect_ratio: float) -> float:
    """Potential-flow normal-force slope per radian of a low-aspect-ratio surface.

    Slender-wing limit pi*A/2. See SOURCES["iv1_aero_strake_Kp"].
    """
    return KP_SLENDER_COEFF * max(aspect_ratio, 0.0)


def polhamus_kv_le(aspect_ratio: float) -> float:
    """Leading-edge vortex-lift factor of a RECTANGULAR surface: pi*A/4, which is K_p/2.

    See SOURCES["iv1_aero_strake_Kv"].
    """
    return KV_LE_COEFF * max(aspect_ratio, 0.0)


def polhamus_kv_se(aspect_ratio: float) -> float:
    """Side-edge vortex-lift factor of a RECTANGULAR surface, from NASA TN D-7921 Table III.

    Linear interpolation in aspect ratio, held constant outside the tabulated range. The
    streamwise side edge is where a strake's vortex actually comes from, so this is the term
    that carries the strake load. See SOURCES["iv1_aero_strake_Kv"].
    """
    ar = max(aspect_ratio, 0.0)
    tab = LAMAR_KV_SE_RECTANGULAR
    if ar <= tab[0][0]:
        return tab[0][1]
    if ar >= tab[-1][0]:
        return tab[-1][1]
    for (a0, k0), (a1, k1) in zip(tab[:-1], tab[1:]):
        if ar <= a1:
            return k0 + (k1 - k0) * (ar - a0) / (a1 - a0)
    return tab[-1][1]


def polhamus_kv(aspect_ratio: float) -> float:
    """Total vortex-lift factor of a rectangular surface, K_v,le + K_v,se.

    See SOURCES["iv1_aero_strake_Kv"].
    """
    return polhamus_kv_le(aspect_ratio) + polhamus_kv_se(aspect_ratio)


def polhamus_cn(aspect_ratio: float, alpha: float) -> tuple[float, float]:
    """(potential, vortex) normal force of a surface, on ITS OWN PLANFORM AREA.

    The Polhamus leading-edge suction analogy,
    `CN = K_p sin(a) cos^2(a) + K_v sin^2(a) cos(a)`, returned as its two parts so the caller
    can report and plot the linear and the vortex contributions separately.

    `aspect_ratio` is the aspect ratio of the surface the area belongs to. For a body-mounted
    strake pair that is `(2b)^2 / S_pair` referred to `S_pair`, the same panel-pair convention
    `aero.py` uses for the tail fins. See SOURCES["iv1_aero_strake_polhamus"].
    """
    a = abs(alpha)
    sa, ca = math.sin(a), math.cos(a)
    potential = polhamus_kp(aspect_ratio) * sa * ca * ca
    vortex = polhamus_kv(aspect_ratio) * sa * sa * ca
    s = 1.0 if alpha >= 0.0 else -1.0
    return s * potential, s * vortex


# --------------------------------------------------------------------------------------
#   Resolved geometry, with provenance
# --------------------------------------------------------------------------------------


@dataclass
class _SurfaceSet:
    """One set of `n` identical panels: a stage's tail fins, or the strakes."""

    label: str
    n: int
    b: float                   # exposed semi-span (height) of one panel, m
    c_r: float                 # root chord, m
    c_t: float                 # tip chord, m
    sweep_le: float            # rad
    t_max: float               # m
    x_le: float                # root leading-edge station from the payload-stage nose tip, m
    x_t: float                 # chordwise station of maximum thickness, fraction of chord
    c_mac: float               # m
    x_mac_le: float            # from the root leading edge, m
    S_panel: float             # exposed planform of ONE panel, m^2
    area_wetted: float         # all panels, both sides, m^2
    S_pair: float              # exposed planform of ONE opposing pair, m^2
    aspect_ratio_pair: float   # (2b)^2 / S_pair
    aspect_ratio_panel: float  # b^2 / S_panel, the exposed panel aspect ratio
    k_upwash: float
    r_tip: float               # tip radius from the body axis, m
    x_cp_sub: float            # from the nose tip, m
    x_cp_sup: float            # from the nose tip, m
    is_strake: bool = False


@dataclass
class _StackGeometry:
    """Geometry of ONE flight configuration, after measurements override analytics."""

    stage: int
    D_ref: float
    S_ref: float
    L_total: float                 # of this configuration, m
    L_nose: float
    D_nose_base: float
    S_nose_base: float
    nose_shape: str
    S_base: float
    area_wetted_body: float
    area_planform_body: float
    x_cp_potential: float
    x_cp_crossflow: float

    # interstage shoulder, present only in the stacked configuration
    shoulder_delta: float          # half-angle, rad; 0 when there is no diameter change
    shoulder_annulus: float        # projected area S1 - S2, m^2
    shoulder_x: float              # dS/dx centroid of the shoulder, m

    surfaces: list[_SurfaceSet]
    area_nozzle_exit: float
    nozzle_exit_is_guess: bool
    provenance: dict[str, str] = field(default_factory=dict)

    @property
    def fin_sets(self) -> list[_SurfaceSet]:
        return [s for s in self.surfaces if not s.is_strake]

    @property
    def strake_set(self) -> _SurfaceSet | None:
        for s in self.surfaces:
            if s.is_strake:
                return s
        return None


def _mac(c_r: float, c_t: float, b: float, sweep_le: float) -> tuple[float, float]:
    """(mean aerodynamic chord, its leading-edge offset from the root leading edge)."""
    lam = c_t / c_r if c_r > 0.0 else 0.0
    c_mac = (2.0 / 3.0) * c_r * (1.0 + lam + lam * lam) / (1.0 + lam)
    y_mac = (b / 3.0) * (1.0 + 2.0 * lam) / (1.0 + lam)
    return c_mac, y_mac * math.tan(sweep_le)


def _barrowman_fin_cp(c_r: float, c_t: float, b: float, sweep_le: float) -> float:
    """Fin load centroid aft of the root leading edge, subsonic. See SOURCES["aero_fin_cp"]."""
    if c_r + c_t <= 0.0:
        return 0.0
    X_R = b * math.tan(sweep_le)
    return (X_R / 3.0) * (c_r + 2.0 * c_t) / (c_r + c_t) + (1.0 / 6.0) * (
        c_r + c_t - c_r * c_t / (c_r + c_t)
    )


def _shoulder_dsdx_centroid(r2: float, r1: float, L: float) -> float:
    """dS/dx-weighted centroid of a conical shoulder, measured from its forward station.

    S(x) = pi r(x)^2 with r linear, so dS/dx is linear in x and the centroid is the centroid of
    a linear ramp from 2 pi r2 r' to 2 pi r1 r'. Reduces to L/2 when r1 = r2.
    """
    if L <= 0.0 or (r1 + r2) <= 0.0:
        return 0.0
    return L * (r2 + 2.0 * r1) / (3.0 * (r1 + r2))


def _fin_surface(
    stage: StageSpec,
    x_stage_front: float,
    stage_length: float,
    fin_te_gap: float,
    x_t: float,
    label: str,
    area_wetted_measured: float | None,
) -> _SurfaceSet:
    """Build the tail-fin set of one stage, stationed in the stack frame."""
    n = int(stage.n_fin)
    c_r, c_t, b = stage.c_r_fin, stage.c_t_fin, stage.b_fin
    R = 0.5 * stage.D
    c_mac, x_mac_le = _mac(c_r, c_t, b, stage.sweep_fin)

    if area_wetted_measured is not None and n > 0:
        area_wetted = float(area_wetted_measured)
        S_panel = area_wetted / (2.0 * n)
    else:
        S_panel = stage.S_fin_exposed
        area_wetted = 2.0 * n * S_panel

    S_pair = 2.0 * S_panel
    # Root trailing edge at the aft end of the stage. See SOURCES["iv1_aero_fin_station"].
    x_le = x_stage_front + stage_length - fin_te_gap - c_r
    return _SurfaceSet(
        label=label,
        n=n,
        b=b,
        c_r=c_r,
        c_t=c_t,
        sweep_le=stage.sweep_fin,
        t_max=stage.t_fin,
        x_le=x_le,
        x_t=x_t,
        c_mac=c_mac,
        x_mac_le=x_mac_le,
        S_panel=S_panel,
        area_wetted=area_wetted,
        S_pair=S_pair,
        aspect_ratio_pair=(2.0 * b) ** 2 / S_pair if S_pair > 0.0 else 0.0,
        aspect_ratio_panel=b * b / S_panel if S_panel > 0.0 else 0.0,
        k_upwash=barrowman_upwash(R, R + b),
        r_tip=R + b,
        x_cp_sub=x_le + _barrowman_fin_cp(c_r, c_t, b, stage.sweep_fin),
        x_cp_sup=x_le + x_mac_le + 0.5 * c_mac,
        is_strake=False,
    )


def _strake_surface(
    st: StrakeSpec, D_stage: float, cp_fraction: float, x_t: float
) -> _SurfaceSet:
    """Build the strake set. Its chord is `length` and its span is `height`."""
    n = int(st.n)
    b = st.height
    c = st.length
    R = 0.5 * D_stage
    S_panel = st.area_one_side
    area_wetted = st.wetted_area
    S_pair = 2.0 * S_panel
    x_cp = st.x_le + cp_fraction * c
    return _SurfaceSet(
        label="strakes",
        n=n,
        b=b,
        c_r=c,
        c_t=c,                       # rectangular
        sweep_le=st.sweep_le,
        t_max=st.thickness,
        x_le=st.x_le,
        x_t=x_t,
        c_mac=c,
        x_mac_le=0.0,
        S_panel=S_panel,
        area_wetted=area_wetted,
        S_pair=S_pair,
        aspect_ratio_pair=(2.0 * b) ** 2 / S_pair if S_pair > 0.0 else 0.0,
        aspect_ratio_panel=st.aspect_ratio,
        k_upwash=barrowman_upwash(R, R + b),
        r_tip=R + b,
        x_cp_sub=x_cp,
        x_cp_sup=x_cp,
        is_strake=True,
    )


# --------------------------------------------------------------------------------------
#   The model
# --------------------------------------------------------------------------------------


class StackAero:
    """Aerodynamic build-up for the IV-1 stack, in both of its flight configurations.

    Args:
        dv: the stack design vector. Closed-form geometry from it is the fallback for
            everything.
        reqs: the intercept requirements. Only `alpha_max` is used, as the default alpha limit
            for `CN_max` and `trim_alpha`.
        meas: nTop measurements keyed by STAGE INDEX. `meas[i].area_wetted_body`,
            `area_wetted_fins`, `area_base` and `area_distribution` replace the analytic
            estimates for that stage, and `sources_used` records that they did. When absent,
            closed form is used and recorded as such.
        nose_shape: payload-stage nose shape, 'tangent_ogive' or 'cone'.
        fin_te_gap: gap from each fin root trailing edge to the aft end of its stage, m.
            See SOURCES["iv1_aero_fin_station"].
        fin_max_thickness_station: chordwise station of maximum fin thickness, fraction of
            chord, for the double-wedge wave-drag model.
        k_afterbody_carryover: fin-to-body carryover factor.
            See SOURCES["aero_fin_afterbody_carryover"].
        k_strake_fin_interference: multiplier on tail-fin normal force to account for the
            strake-shed vortex. 1.0 means NOT MODELLED.
            See SOURCES["iv1_aero_strake_fin_interference"].
        strake_cp_chord_fraction: strake load centroid as a fraction of strake length.
            See SOURCES["iv1_aero_strake_cp"].
        area_nozzle_exit: nozzle exit area by stage index, m^2, used only for powered-base drag
            relief. A missing entry falls back to a GUESSED fraction of the base area and is
            flagged.
    """

    def __init__(
        self,
        dv: StackDesignVector,
        reqs: InterceptRequirements,
        meas: dict[int, NtopMeasurements] | None = None,
        *,
        nose_shape: str = "tangent_ogive",
        fin_te_gap: float = 0.0,
        fin_max_thickness_station: float = 0.5,
        k_afterbody_carryover: float = 1.0,
        k_strake_fin_interference: float = 1.0,
        strake_cp_chord_fraction: float = STRAKE_CP_CHORD_FRACTION,
        area_nozzle_exit: dict[int, float] | None = None,
    ) -> None:
        if dv.n_stages < 1:
            raise ValueError("StackAero needs at least one stage")
        self.dv = dv
        self.reqs = reqs
        self.meas = dict(meas) if meas else {}
        self.nose_shape = nose_shape
        self.fin_te_gap = float(fin_te_gap)
        self.x_t_fin = float(fin_max_thickness_station)
        self.k_afterbody_carryover = float(k_afterbody_carryover)
        self.k_strake_fin_interference = float(k_strake_fin_interference)
        self.strake_cp_chord_fraction = float(strake_cp_chord_fraction)
        self._area_nozzle_exit = dict(area_nozzle_exit) if area_nozzle_exit else {}

        #: Provenance per configuration index. CLAUDE.md hard rule 3.3: bad news travels up.
        self.sources_used: dict[int, dict[str, str]] = {}
        self.geom: dict[int, _StackGeometry] = {}
        for stage in self.stages_available:
            self.geom[stage] = self._resolve_geometry(stage)

    # ---------------------------------------------------------------- configurations

    @property
    def stages_available(self) -> tuple[int, ...]:
        """Configuration indices this instance can evaluate: 1 for the stack, then upward."""
        return tuple(s.index for s in self.dv.stages)

    def S_ref(self, stage: int) -> float:
        """Reference area of the configuration, m^2.

        `stage=1` is the full stack and uses the BOOSTER cross-section. Higher indices are the
        surviving vehicle after separation and use their own cross-section. This is the function
        that makes the reference-area change at separation explicit.
        """
        return self._geom(stage).S_ref

    def D_ref(self, stage: int) -> float:
        """Reference diameter of the configuration, m. Moments are referenced to S_ref * D_ref."""
        return self._geom(stage).D_ref

    def _geom(self, stage: int) -> _StackGeometry:
        try:
            return self.geom[stage]
        except KeyError:
            raise KeyError(
                f"no configuration for stage {stage}; available {self.stages_available}"
            ) from None

    # ---------------------------------------------------------------- geometry

    def _resolve_geometry(self, stage: int) -> _StackGeometry:
        """Build the geometry of the configuration that flies while `stage` is the live stage.

        The configuration contains every stage from `stage` upward, plus the interstage between
        each surviving pair. Stages already burnt and jettisoned are absent.
        """
        dv = self.dv
        prov: dict[str, str] = {}
        self.sources_used[stage] = prov

        present = [s for s in dv.stages if s.index >= stage]
        if not present:
            raise KeyError(f"no stage with index >= {stage}")
        # Physical order front to back is the reverse of burn order: the payload stage leads.
        front_to_back = list(reversed(present))
        payload = front_to_back[0]
        aft = front_to_back[-1]

        n_gaps = len(front_to_back) - 1
        L_is = dv.L_interstage if n_gaps > 0 else 0.0

        # --- reference area: the widest body in this configuration, which is the aft-most ---
        D_ref = max(s.D for s in front_to_back)
        S_ref = 0.25 * math.pi * D_ref * D_ref
        prov["S_ref"] = (
            f"stage {stage} configuration: {'booster' if n_gaps else 'payload stage'} "
            f"diameter {D_ref:.4f} m"
        )

        # --- nose, on the payload stage ---
        L_nose = dv.L_nose
        D_nose = payload.D
        S_nose = 0.25 * math.pi * D_nose * D_nose
        wet_nose, plan_nose, xbar_nose = _nose_wetted_and_planform(
            self.nose_shape, L_nose, D_nose
        )
        prov["nose_shape"] = f"constructor argument: {self.nose_shape}"

        # --- march down the stack, accumulating wetted area, planform and dS/dx ---
        wet = wet_nose
        plan = plan_nose
        plan_moment = plan_nose * xbar_nose
        # dS/dx contributions as (net area change, station).
        dsdx: list[tuple[float, float]] = [
            (S_nose, X_CP_NOSE_FRACTION[self.nose_shape] * L_nose)
        ]
        shoulder_delta = 0.0
        shoulder_annulus = 0.0
        shoulder_x = 0.0

        x = 0.0
        for i, s in enumerate(front_to_back):
            R = 0.5 * s.D
            L_cyl = s.L - L_nose if i == 0 else s.L
            if L_cyl < 0.0:
                raise ValueError(
                    f"stage {s.index} length {s.L:.3f} m is shorter than the nose {L_nose:.3f} m"
                )
            x_cyl_front = x + (L_nose if i == 0 else 0.0)
            wet += math.pi * s.D * L_cyl
            plan += s.D * L_cyl
            plan_moment += s.D * L_cyl * (x_cyl_front + 0.5 * L_cyl)
            x += s.L

            if i < n_gaps:
                nxt = front_to_back[i + 1]
                r_next = 0.5 * nxt.D
                a_is, xbar_is = _frustum_planform_centroid(R, r_next, L_is)
                slant = math.hypot(L_is, r_next - R)
                wet += math.pi * (R + r_next) * slant
                plan += a_is
                plan_moment += a_is * (x + xbar_is)
                if r_next > R:
                    # An expanding shoulder: a compressive turn, and a real drag term.
                    shoulder_delta = (
                        math.atan2(r_next - R, L_is) if L_is > 0.0 else 0.5 * math.pi
                    )
                    shoulder_annulus += (
                        0.25 * math.pi * (nxt.D * nxt.D - s.D * s.D)
                    )
                    shoulder_x = x + _shoulder_dsdx_centroid(R, r_next, L_is)
                    dsdx.append(
                        (0.25 * math.pi * (nxt.D * nxt.D - s.D * s.D), shoulder_x)
                    )
                x += L_is

        L_total = x
        prov["L_total"] = ANALYTIC + f" ({L_total:.4f} m)"

        # --- base area, on the aft-most body ---
        m_aft = self.meas.get(aft.index)
        S_body_aft = 0.25 * math.pi * aft.D * aft.D
        if m_aft is not None and m_aft.area_base is not None:
            S_base = float(m_aft.area_base)
            prov["area_base"] = NTOP
            if S_base < S_body_aft - 1.0e-12:
                prov["boattail"] = (
                    "measured base area is smaller than the aft body cross-section, so the "
                    "geometry implies a boattail; its pressure drag is NOT modelled and this "
                    "is optimistic. See SOURCES['iv1_aero_no_boattail']."
                )
        else:
            S_base = S_body_aft
            prov["area_base"] = ANALYTIC
            prov["boattail"] = "none: no boattail on either IV-1 stage"

        # --- wetted body area: measured per stage where available ---
        wet_measured = 0.0
        n_measured = 0
        for s in front_to_back:
            m = self.meas.get(s.index)
            if m is not None and m.area_wetted_body is not None:
                wet_measured += float(m.area_wetted_body)
                n_measured += 1
        if n_measured == len(front_to_back):
            # The interstage is never a stage, so it is always analytic.
            wet_is = wet - wet_nose - sum(
                math.pi * s.D * (s.L - (L_nose if i == 0 else 0.0))
                for i, s in enumerate(front_to_back)
            )
            area_wetted_body = wet_measured + wet_is
            prov["area_wetted_body"] = (
                NTOP + f" for all {n_measured} stage(s), analytic interstage"
            )
        else:
            area_wetted_body = wet
            prov["area_wetted_body"] = ANALYTIC + (
                f" ({n_measured} of {len(front_to_back)} stages measured, so the analytic "
                "total is used to avoid mixing bases)"
                if n_measured
                else ""
            )

        # --- planform and the two body centres of pressure ---
        dist = self._composite_area_distribution(front_to_back, L_is, L_nose)
        if dist is not None:
            plan_meas, xcp_cross = _planform_from_distribution(dist)
            area_planform_body = plan_meas
            x_cp_crossflow = xcp_cross
            prov["area_distribution"] = NTOP + " (spliced across stages)"
            prov["area_planform_body"] = NTOP
            prov["x_cp_crossflow"] = NTOP + " (planform centroid)"
        else:
            area_planform_body = plan
            x_cp_crossflow = plan_moment / plan if plan > 0.0 else 0.0
            prov["area_distribution"] = "absent"
            prov["area_planform_body"] = ANALYTIC
            prov["x_cp_crossflow"] = ANALYTIC + " (planform centroid)"

        den = sum(d for d, _ in dsdx)
        x_cp_potential = (
            sum(d * xx for d, xx in dsdx) / den
            if abs(den) > 1.0e-12
            else X_CP_NOSE_FRACTION[self.nose_shape] * L_nose
        )
        prov["x_cp_potential"] = (
            ANALYTIC + f" (dS/dx centroid over {len(dsdx)} area change(s))"
        )

        # --- surfaces: one fin set per surviving stage, plus the strakes ---
        surfaces: list[_SurfaceSet] = []
        x_front = 0.0
        for i, s in enumerate(front_to_back):
            m = self.meas.get(s.index)
            fins_measured = m.area_wetted_fins if m is not None else None
            if fins_measured is not None:
                prov[f"area_wetted_fins_stage{s.index}"] = NTOP
            else:
                prov[f"area_wetted_fins_stage{s.index}"] = ANALYTIC
            if s.n_fin >= 2 and s.b_fin > 0.0 and s.c_r_fin > 0.0:
                surfaces.append(
                    _fin_surface(
                        s,
                        x_front,
                        s.L,
                        self.fin_te_gap,
                        self.x_t_fin,
                        f"fins_stage{s.index}",
                        fins_measured,
                    )
                )
            x_front += s.L + (L_is if i < n_gaps else 0.0)

        st = dv.strakes
        if st.n >= 2 and st.height > 0.0 and st.length > 0.0:
            surfaces.append(
                _strake_surface(
                    st, payload.D, self.strake_cp_chord_fraction, STRAKE_X_T
                )
            )
            ar_pair = surfaces[-1].aspect_ratio_pair
            prov["strakes"] = (
                f"{st.n} panels, panel aspect ratio {st.aspect_ratio:.4f}, pair aspect ratio "
                f"{ar_pair:.4f}, Polhamus K_p {polhamus_kp(ar_pair):.4f} per rad, "
                f"K_v,le {polhamus_kv_le(ar_pair):.4f}, K_v,se "
                f"{polhamus_kv_se(ar_pair):.4f}, K_v {polhamus_kv(ar_pair):.4f}"
            )
        else:
            prov["strakes"] = "none: zero height, zero length or fewer than two panels"
        prov["k_strake_fin_interference"] = (
            f"constructor argument: {self.k_strake_fin_interference}"
            + (" (1.0 means NOT MODELLED)" if self.k_strake_fin_interference == 1.0 else "")
        )
        prov["k_afterbody_carryover"] = f"constructor argument: {self.k_afterbody_carryover}"

        # --- nozzle exit area, only used for powered base drag ---
        a_e_given = self._area_nozzle_exit.get(aft.index)
        if a_e_given is not None:
            a_e = float(a_e_given)
            guessed = False
            prov["area_nozzle_exit"] = "supplied by caller"
        else:
            a_e = NOZZLE_EXIT_AREA_FRACTION_GUESS * S_base
            guessed = True
            prov["area_nozzle_exit"] = "GUESS (fraction of base area)"

        return _StackGeometry(
            stage=stage,
            D_ref=D_ref,
            S_ref=S_ref,
            L_total=L_total,
            L_nose=L_nose,
            D_nose_base=D_nose,
            S_nose_base=S_nose,
            nose_shape=self.nose_shape,
            S_base=S_base,
            area_wetted_body=area_wetted_body,
            area_planform_body=area_planform_body,
            x_cp_potential=x_cp_potential,
            x_cp_crossflow=x_cp_crossflow,
            shoulder_delta=shoulder_delta,
            shoulder_annulus=shoulder_annulus,
            shoulder_x=shoulder_x,
            surfaces=surfaces,
            area_nozzle_exit=a_e,
            nozzle_exit_is_guess=guessed,
            provenance=dict(prov),
        )

    def _composite_area_distribution(
        self, front_to_back: list[StageSpec], L_is: float, L_nose: float
    ) -> list[tuple[float, float]] | None:
        """Splice per-stage measured S(x) into one stack distribution, or None.

        Each stage's own distribution is datumed at that stage's forward end, so it is offset
        into the stack frame. The interstage between two stages is filled analytically with a
        conical shoulder because it is not part of any stage's measurement.
        """
        parts: list[tuple[float, float]] = []
        x_front = 0.0
        n_gaps = len(front_to_back) - 1
        for i, s in enumerate(front_to_back):
            m = self.meas.get(s.index)
            if m is None or len(m.area_distribution) < 3:
                return None
            pts = sorted((float(a), float(b)) for a, b in m.area_distribution)
            x0 = pts[0][0]
            parts.extend((x_front + (xx - x0), aa) for xx, aa in pts)
            x_aft = x_front + s.L
            if i < n_gaps and L_is > 0.0:
                nxt = front_to_back[i + 1]
                r0, r1 = 0.5 * s.D, 0.5 * nxt.D
                n = 16
                for k in range(1, n + 1):
                    f = k / n
                    r = r0 + f * (r1 - r0)
                    parts.append((x_aft + f * L_is, math.pi * r * r))
                x_front = x_aft + L_is
            else:
                x_front = x_aft
        del L_nose      # kept in the signature for symmetry with the analytic path
        return parts if len(parts) >= 3 else None

    # ---------------------------------------------------------------- flow

    @staticmethod
    def _flow(altitude: float, mach: float) -> tuple[float, float]:
        """(unit Reynolds number per metre, static temperature K). One atmosphere lookup."""
        stt = atm.atmo(altitude)
        v = mach * stt.speed_of_sound
        return stt.density * v / stt.dynamic_viscosity, stt.temperature

    # ---------------------------------------------------------------- drag components

    def CD_friction_body(
        self, mach: float, altitude: float, stage: int,
        flow: tuple[float, float] | None = None,
    ) -> float:
        """Body skin friction on S_ref. Reynolds number on the configuration length."""
        g = self._geom(stage)
        re_per_m, t_inf = flow if flow is not None else self._flow(altitude, mach)
        cf = cf_turbulent(re_per_m * g.L_total, mach, t_inf)
        ff = body_form_factor(g.L_total / g.D_ref)
        return ff * cf * g.area_wetted_body / g.S_ref

    def CD_wave_body(self, mach: float, stage: int) -> float:
        """Forebody wave drag on S_ref.

        The Bonney correlation is referenced to the NOSE base area, which for the stack is the
        payload-stage cross-section and NOT the reference area, so it is referred across.
        See SOURCES["aero_body_wave_drag"].
        """
        g = self._geom(stage)
        return (
            bonney_nose_wave_cd(g.L_nose / g.D_nose_base, mach) * g.S_nose_base / g.S_ref
        )

    def _shoulder_terms(self, mach: float, stage: int) -> tuple[float, float, float, bool]:
        """(tangent-wedge CD, isentropic CD, stagnation CD, shock attached) on S_ref.

        All four in one place so `evaluate` pays for the gas dynamics once instead of four
        times. Only the first is summed into CD0.
        """
        g = self._geom(stage)
        if g.shoulder_annulus <= 0.0 or g.shoulder_delta <= 0.0:
            return 0.0, 0.0, 0.0, True
        ratio = g.shoulder_annulus / g.S_ref
        bridge = 1.0 - cubic_blend(mach, M_BRIDGE_LO, M_BRIDGE_HI)
        if bridge <= 0.0:
            return 0.0, 0.0, 0.0, True
        m1 = max(mach, M_BRIDGE_HI)
        cp, attached = oblique_shock_cp(m1, g.shoulder_delta)
        cp_isen = compression_turn_cp(m1, g.shoulder_delta)
        return (
            bridge * max(cp * ratio, 0.0),
            bridge * max(cp_isen * ratio, 0.0),
            _cp_max_stagnation(mach) * ratio,
            attached,
        )

    def CD_shoulder(self, mach: float, stage: int) -> float:
        """Interstage shoulder pressure drag on S_ref.

        Exact weak oblique shock through the shoulder half-angle (the tangent-wedge estimate),
        applied as +Cp on the projected annulus and blended to zero below M 0.95. Zero when the
        configuration has no diameter step. See SOURCES["iv1_aero_interstage_shoulder"].
        """
        return self._shoulder_terms(mach, stage)[0]

    def CD_shoulder_isentropic_crosscheck(self, mach: float, stage: int) -> float:
        """Shoulder drag from an isentropic compression turn instead. Reported, never summed.

        Method sensitivity: the same annulus, the same turn angle, no shock entropy rise. See
        SOURCES["aero_isentropic_compression_turn"].
        """
        return self._shoulder_terms(mach, stage)[1]

    def CD_shoulder_stagnation_crosscheck(self, mach: float, stage: int) -> float:
        """Stagnation-pressure bound on the shoulder annulus. Reported, never summed.

        No shock system can raise the annulus pressure above the stagnation value, so this is a
        hard upper bracket on `CD_shoulder`.
        """
        return self._shoulder_terms(mach, stage)[2]

    def shoulder_shock_is_attached(self, mach: float, stage: int) -> bool:
        """True when the interstage shoulder shock is attached at this Mach number.

        False means the value is clamped at the maximum attached deflection and is a lower
        bound. See SOURCES["iv1_aero_interstage_shoulder"].
        """
        return self._shoulder_terms(mach, stage)[3]

    def CD_base(self, mach: float, stage: int, power_on: bool = False) -> float:
        """Base drag on S_ref. See SOURCES["aero_base_drag"]."""
        g = self._geom(stage)
        cd = base_cd_on_base_area(mach) * g.S_base / g.S_ref
        if power_on and g.S_base > 0.0:
            cd *= max(1.0 - g.area_nozzle_exit / g.S_base, 0.0)
        return cd

    def _surface_friction(
        self, surf: _SurfaceSet, mach: float, S_ref: float, flow: tuple[float, float]
    ) -> float:
        """Skin friction of one surface set on S_ref. Identical physics for fins and strakes."""
        if surf.area_wetted <= 0.0 or surf.c_mac <= 0.0:
            return 0.0
        re_per_m, t_inf = flow
        cf = cf_turbulent(re_per_m * surf.c_mac, mach, t_inf)
        ff = surface_form_factor(surf.t_max / surf.c_mac, surf.x_t)
        return ff * cf * surf.area_wetted / S_ref

    def _surface_wave(self, surf: _SurfaceSet, mach: float, S_ref: float) -> float:
        """Ackeret double-wedge thickness drag of one surface set on S_ref."""
        if surf.n < 1 or surf.S_panel <= 0.0 or surf.c_mac <= 0.0:
            return 0.0
        cd_2d = fin_wave_cd_2d(surf.t_max / surf.c_mac, surf.x_t, mach)
        return cd_2d * surf.n * surf.S_panel / S_ref

    def CD_fin_friction(
        self, mach: float, altitude: float, stage: int,
        flow: tuple[float, float] | None = None,
    ) -> float:
        """Skin friction on every tail fin in the configuration, on S_ref."""
        g = self._geom(stage)
        f = flow if flow is not None else self._flow(altitude, mach)
        return sum(self._surface_friction(s, mach, g.S_ref, f) for s in g.fin_sets)

    def CD_fin_wave(self, mach: float, stage: int) -> float:
        """Fin thickness (wave) drag on S_ref. See SOURCES["aero_fin_wave_drag"]."""
        g = self._geom(stage)
        return sum(self._surface_wave(s, mach, g.S_ref) for s in g.fin_sets)

    def CD_strake_friction(
        self, mach: float, altitude: float, stage: int,
        flow: tuple[float, float] | None = None,
    ) -> float:
        """Skin friction on the strake wetted area, on S_ref.

        Reynolds number on the strake LENGTH, which is its streamwise chord. Same law and same
        form factor as the fins. See SOURCES["iv1_aero_strake_drag"].
        """
        g = self._geom(stage)
        surf = g.strake_set
        if surf is None:
            return 0.0
        f = flow if flow is not None else self._flow(altitude, mach)
        return self._surface_friction(surf, mach, g.S_ref, f)

    def CD_strake_wave(self, mach: float, stage: int) -> float:
        """Strake thickness (wave) drag on S_ref.

        Double-wedge assumption, which UNDERSTATES a real slab section with a short bevel.
        See SOURCES["iv1_aero_strake_drag"].
        """
        g = self._geom(stage)
        surf = g.strake_set
        return 0.0 if surf is None else self._surface_wave(surf, mach, g.S_ref)

    def CD_strake_base(self, mach: float, stage: int) -> float:
        """Strake blunt-trailing-edge drag on S_ref.

        The body base-pressure correlation applied to the strake trailing-edge area
        `n * thickness * height`. This is an EXTENSION of an axisymmetric correlation to a
        two-dimensional trailing edge. See SOURCES["iv1_aero_strake_drag"].
        """
        g = self._geom(stage)
        surf = g.strake_set
        if surf is None:
            return 0.0
        area_te = surf.n * surf.t_max * surf.b
        return base_cd_on_base_area(mach) * area_te / g.S_ref

    def CD_strake_interference(self, mach: float, stage: int) -> float:
        """Strake-to-body junction interference drag. ZERO, deliberately.

        See SOURCES["iv1_aero_strake_junction_interference"]: the junction is already inside
        `config.CD0_CALIBRATION`, which the sizing loop applies at its boundary.
        """
        del mach, stage
        return 0.0

    def CD_strake_le_blunt_bound(self, mach: float, stage: int) -> float:
        """Upper bound on strake leading-edge drag. Reported, never summed.

        Stagnation pressure on the whole strake frontal area, `n * thickness * height`, which is
        what a fully blunt unbeveled leading edge would cost. Together with the double-wedge
        value it BRACKETS the real strake wave drag, whose position inside the bracket depends on
        the leading-edge bevel length. That length is not an IV-1 design variable, so the model
        reports the optimistic end and the bound instead of interpolating between them with an
        invented bevel. See SOURCES["iv1_aero_strake_drag"].
        """
        g = self._geom(stage)
        surf = g.strake_set
        if surf is None:
            return 0.0
        return _cp_max_stagnation(mach) * surf.n * surf.t_max * surf.b / g.S_ref

    def CD_protuberance(self, cd_clean: float) -> float:
        """Lumped protuberance allowance. THIS IS A GUESS.

        See SOURCES["aero_protuberance_GUESS"].
        """
        return PROTUBERANCE_FRACTION_GUESS * cd_clean

    # ---------------------------------------------------------------- normal force

    def CN_body(self, mach: float, alpha: float, stage: int) -> tuple[float, float]:
        """(potential, cross-flow) body normal force on S_ref.

        For the stack the potential term uses the BOOSTER base area and the cross-flow term the
        whole stack planform, including the interstage. See SOURCES["aero_body_normal_force"].
        """
        del mach
        g = self._geom(stage)
        return body_normal_force_terms(g.S_base, g.S_ref, g.area_planform_body, alpha)

    def _fin_set_cn(self, surf: _SurfaceSet, mach: float, alpha: float, S_ref: float) -> float:
        """Normal force of ONE tail-fin set on S_ref, with every interference factor applied."""
        if surf.n < 2 or surf.S_pair <= 0.0 or surf.aspect_ratio_pair <= 0.0:
            return 0.0
        return (
            lifting_surface_cn_alone(surf.aspect_ratio_pair, mach, alpha)
            * surf.S_pair
            / S_ref
            * surf.k_upwash
            * self.k_afterbody_carryover
            * self.k_strake_fin_interference
        )

    def CN_fins(self, mach: float, alpha: float, stage: int) -> float:
        """Tail-fin normal force of every fin set in the configuration, on S_ref.

        Each set carries its own aspect ratio, its own body-upwash factor from its own stage
        diameter, the afterbody carryover factor, and the strake-fin interference factor, which
        is 1.0 because it is not modelled.
        See SOURCES["aero_fin_normal_force"] and SOURCES["iv1_aero_strake_fin_interference"].
        """
        g = self._geom(stage)
        return sum(self._fin_set_cn(s, mach, alpha, g.S_ref) for s in g.fin_sets)

    def CN_strakes(self, mach: float, alpha: float, stage: int) -> tuple[float, float]:
        """(potential, vortex) strake normal force on S_ref.

        Polhamus suction analogy with the Lamar side-edge term, evaluated on the strake PAIR
        aspect ratio and the pair exposed area, which is the same convention `aero.py` uses for
        the tail fins. The Barrowman body-upwash factor multiplies the potential term only. No
        Mach dependence: the analogy has none.
        See SOURCES["iv1_aero_strake_polhamus"] and SOURCES["iv1_aero_strake_upwash"].

        A cruciform set of four strakes has two panels in the plane of alpha and two edge-on, so
        the loaded area is the PAIR area, not all `n` panels. That is why the strake wetted area
        counts every panel for drag while the normal force counts only the pair.
        """
        del mach
        g = self._geom(stage)
        surf = g.strake_set
        if surf is None or surf.S_pair <= 0.0 or surf.aspect_ratio_pair <= 0.0:
            return 0.0, 0.0
        pot, vor = polhamus_cn(surf.aspect_ratio_pair, alpha)
        ratio = surf.S_pair / g.S_ref
        return pot * ratio * surf.k_upwash, vor * ratio

    def x_cp_surface(self, surf: _SurfaceSet, mach: float) -> float:
        """Load centroid of one surface set from the nose tip, m. See SOURCES["aero_fin_cp"]."""
        if surf.is_strake:
            return surf.x_cp_sub          # mid-chord, Mach-independent
        w = cubic_blend(mach, M_BRIDGE_LO, M_BRIDGE_HI)
        return w * surf.x_cp_sub + (1.0 - w) * surf.x_cp_sup

    # ---------------------------------------------------------------- assembly

    def evaluate(
        self,
        mach: float,
        altitude: float,
        alpha: float,
        stage: int,
        power_on: bool = False,
    ) -> AeroCoefficients:
        """Full aerodynamic state of one configuration at one flight point.

        Args:
            mach: free-stream Mach number. Validated 0.3 to 5.0.
            altitude: geometric altitude, m.
            alpha: angle of attack, rad. Validated to 25 deg in magnitude.
            stage: flight configuration. 1 is the full stack, 2 the payload stage alone.
            power_on: True while the live stage is thrusting, which relieves base drag.

        Every coefficient returned is referenced to `S_ref(stage)`, and `CM` to
        `S_ref(stage) * D_ref(stage)`.
        """
        g = self._geom(stage)
        mach = float(mach)
        alpha = float(alpha)

        # --- zero-lift components ---
        flow = self._flow(altitude, mach)
        cd_fric_body = self.CD_friction_body(mach, altitude, stage, flow)
        cd_wave_body = self.CD_wave_body(mach, stage)
        (
            cd_shoulder,
            cd_shoulder_isen,
            cd_shoulder_stag,
            shoulder_attached,
        ) = self._shoulder_terms(mach, stage)
        cd_base = self.CD_base(mach, stage, power_on)
        cd_fric_fin = self.CD_fin_friction(mach, altitude, stage, flow)
        cd_wave_fin = self.CD_fin_wave(mach, stage)
        cd_fric_strake = self.CD_strake_friction(mach, altitude, stage, flow)
        cd_wave_strake = self.CD_strake_wave(mach, stage)
        cd_base_strake = self.CD_strake_base(mach, stage)
        cd_int_strake = self.CD_strake_interference(mach, stage)
        cd_clean = (
            cd_fric_body
            + cd_wave_body
            + cd_shoulder
            + cd_base
            + cd_fric_fin
            + cd_wave_fin
            + cd_fric_strake
            + cd_wave_strake
            + cd_base_strake
            + cd_int_strake
        )
        cd_prot = self.CD_protuberance(cd_clean)
        CD0 = cd_clean + cd_prot

        # --- normal force. Each surface load is computed once and reused for the moment. ---
        cn_pot, cn_cross = self.CN_body(mach, alpha, stage)
        cn_str_pot, cn_str_vor = self.CN_strakes(mach, alpha, stage)
        cn_strake = cn_str_pot + cn_str_vor
        loads: list[tuple[float, float]] = []          # (station m, CN on S_ref)
        cn_fin = 0.0
        per_set: list[tuple[str, float, float]] = []   # (label, CN on S_ref, station m)
        for s in g.fin_sets:
            c = self._fin_set_cn(s, mach, alpha, g.S_ref)
            cn_fin += c
            xs = self.x_cp_surface(s, mach)
            loads.append((xs, c))
            per_set.append((s.label, c, xs))
        strake = g.strake_set
        if strake is not None:
            loads.append((strake.x_cp_sub, cn_strake))
        CN = cn_pot + cn_cross + cn_fin + cn_strake

        # --- centre of pressure, moment about the nose tip ---
        x_cp = self._x_cp_from_loads(mach, stage, cn_pot, cn_cross, CN, loads)
        CM = -CN * x_cp / g.D_ref

        # --- wind-axis drag and lift ---
        ca, sa = math.cos(alpha), math.sin(alpha)
        cd_induced = CN * sa
        CD = CD0 * ca + cd_induced
        CL = CN * ca - CD0 * sa
        l_over_d = CL / CD if abs(CD) > 1.0e-12 else 0.0

        # --- CN_alpha as a secant slope, which is what test reductions report ---
        if abs(alpha) > 1.0e-6:
            CN_alpha = CN / alpha
        else:
            eps = 1.0e-4
            CN_alpha = self._cn_total(mach, eps, stage) / eps

        strake = g.strake_set
        breakdown = {
            "CD_friction_body": cd_fric_body,
            "CD_wave_body": cd_wave_body,
            "CD_interstage_shoulder": cd_shoulder,
            "CD_base": cd_base,
            "CD_boattail": 0.0,
            "CD_fin_friction": cd_fric_fin,
            "CD_fin_wave": cd_wave_fin,
            "CD_strake_friction": cd_fric_strake,
            "CD_strake_wave": cd_wave_strake,
            "CD_strake_base": cd_base_strake,
            "CD_strake_interference_NOT_MODELLED": cd_int_strake,
            "CD_protuberance_GUESS": cd_prot,
            "CD_induced": cd_induced,
            "CD_axial_incidence_projection": CD0 * (ca - 1.0),
            # reported, never summed
            "xcheck_CD_strake_le_blunt_bound": self.CD_strake_le_blunt_bound(mach, stage),
            "xcheck_CD_shoulder_isentropic": cd_shoulder_isen,
            "xcheck_CD_shoulder_stagnation": cd_shoulder_stag,
            "shoulder_shock_attached": float(shoulder_attached),
            "CN_body_potential": cn_pot,
            "CN_body_crossflow": cn_cross,
            "CN_fins": cn_fin,
            "CN_strakes": cn_strake,
            "CN_strakes_potential": cn_str_pot,
            "CN_strakes_vortex": cn_str_vor,
            "x_cp_potential_m": g.x_cp_potential,
            "x_cp_crossflow_m": g.x_cp_crossflow,
            "x_cp_strakes_m": strake.x_cp_sub if strake is not None else 0.0,
            "x_cp_over_D": x_cp / g.D_ref,
            "S_ref_m2": g.S_ref,
            "D_ref_m": g.D_ref,
            "L_total_m": g.L_total,
            "stage": float(stage),
            "power_on": 1.0 if power_on else 0.0,
            "strake_aspect_ratio_panel": (
                strake.aspect_ratio_panel if strake is not None else 0.0
            ),
            "strake_aspect_ratio_pair": (
                strake.aspect_ratio_pair if strake is not None else 0.0
            ),
            "strake_Kp": (
                polhamus_kp(strake.aspect_ratio_pair) if strake is not None else 0.0
            ),
            "strake_Kv_le": (
                polhamus_kv_le(strake.aspect_ratio_pair) if strake is not None else 0.0
            ),
            "strake_Kv_se": (
                polhamus_kv_se(strake.aspect_ratio_pair) if strake is not None else 0.0
            ),
            "strake_Kv": (
                polhamus_kv(strake.aspect_ratio_pair) if strake is not None else 0.0
            ),
            "n_quantities_from_ntop": float(
                sum(
                    1
                    for v in self.sources_used.get(stage, {}).values()
                    if v.startswith(NTOP)
                )
            ),
            "n_guessed_quantities": float(
                sum(
                    1
                    for v in self.sources_used.get(stage, {}).values()
                    if "GUESS" in v
                )
            ),
            "out_of_validity_range": float(
                not (MACH_MIN_VALID <= mach <= MACH_MAX_VALID)
                or abs(alpha) > ALPHA_MAX_VALID_IV1
            ),
        }
        # Per-fin-set loads, so a caller can check that the dimensional force carried by a
        # surface which SURVIVES separation is identical in both configurations.
        for label, cn_set, x_set in per_set:
            breakdown[f"CN_{label}"] = cn_set
            breakdown[f"x_cp_{label}_m"] = x_set

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

    def _x_cp_from_loads(
        self,
        mach: float,
        stage: int,
        cn_pot: float,
        cn_cross: float,
        CN: float,
        loads: list[tuple[float, float]],
    ) -> float:
        """Load-weighted centre of pressure from the nose tip, m.

        `loads` is the list of `(station, CN)` pairs already computed by `evaluate`, so no
        normal-force term is evaluated twice.
        """
        g = self._geom(stage)
        if abs(CN) <= 1.0e-12:
            # alpha -> 0 limit: weight by the alpha -> 0 slope of each component instead.
            return self.x_cp(mach, 1.0e-4, stage)
        num = cn_pot * g.x_cp_potential + cn_cross * g.x_cp_crossflow
        den = cn_pot + cn_cross
        for xx, c in loads:
            num += c * xx
            den += c
        return num / den if abs(den) > 1.0e-15 else g.x_cp_potential

    def x_cp(self, mach: float, alpha: float, stage: int) -> float:
        """Centre of pressure from the payload-stage nose tip, m, at a finite alpha."""
        g = self._geom(stage)
        cn_pot, cn_cross = self.CN_body(mach, alpha, stage)
        p, v = self.CN_strakes(mach, alpha, stage)
        loads = [(self.x_cp_surface(s, mach), self._fin_set_cn(s, mach, alpha, g.S_ref))
                 for s in g.fin_sets]
        strake = g.strake_set
        if strake is not None:
            loads.append((strake.x_cp_sub, p + v))
        num = cn_pot * g.x_cp_potential + cn_cross * g.x_cp_crossflow
        den = cn_pot + cn_cross
        for xx, c in loads:
            num += c * xx
            den += c
        return num / den if abs(den) > 1.0e-15 else g.x_cp_potential

    def _cn_total(self, mach: float, alpha: float, stage: int) -> float:
        """Total CN on S_ref, without building an `AeroCoefficients`. The trim hot path."""
        p, c = self.CN_body(mach, alpha, stage)
        sp, sv = self.CN_strakes(mach, alpha, stage)
        return p + c + self.CN_fins(mach, alpha, stage) + sp + sv

    # ---------------------------------------------------------------- trim and capability

    def trim_alpha(
        self,
        mach: float,
        altitude: float,
        required_CN: float,
        stage: int,
        power_on: bool = False,
        alpha_max: float | None = None,
    ) -> float:
        """Angle of attack, rad, that produces `required_CN` on `S_ref(stage)`.

        `required_CN` must already be referenced to `S_ref(stage)`. Saturates at the alpha limit
        rather than raising, so the sizing loop sees a clipped control. Reuses the bracketed
        solver in `aero.solve_alpha_for_cn`.
        """
        del altitude, power_on          # CN in this build-up depends on Mach and alpha only
        a_max = self.reqs.alpha_max if alpha_max is None else float(alpha_max)
        return solve_alpha_for_cn(
            lambda a: self._cn_total(mach, a, stage), float(required_CN), a_max
        )

    def CN_max(
        self, mach: float, altitude: float, stage: int, alpha_max: float
    ) -> float:
        """Normal-force coefficient available at the alpha limit, on `S_ref(stage)`.

        This is the quantity requirement A11 needs: available lateral acceleration is
        `q * S_ref * CN_max / m`, so it must be paired with `S_ref(stage)` of the SAME stage.
        The strake contribution is in it, and at the alpha limits of interest it dominates the
        strake's linear term by two orders of magnitude.
        """
        del altitude                    # CN has no Reynolds dependence in this build-up
        return self._cn_total(mach, abs(float(alpha_max)), stage)

    # ---------------------------------------------------------------- reporting helpers

    def static_margin(
        self, mach: float, altitude: float, alpha: float, stage: int, x_cg: float
    ) -> float:
        """Static margin in calibres of `D_ref(stage)`, positive when stable.

        `x_cg` is measured aft from the payload-stage nose tip, the same datum as `x_cp`.
        """
        r = self.evaluate(mach, altitude, alpha, stage)
        return (r.x_cp - x_cg) / self._geom(stage).D_ref

    def strake_summary(self, stage: int) -> dict[str, float]:
        """The strake numbers a report or a sizing loop wants, without an `evaluate` call."""
        g = self._geom(stage)
        s = g.strake_set
        if s is None:
            return {"present": 0.0}
        return {
            "present": 1.0,
            "n_panels": float(s.n),
            "height_m": s.b,
            "length_m": s.c_r,
            "thickness_m": s.t_max,
            "x_le_m": s.x_le,
            "x_cp_m": s.x_cp_sub,
            "aspect_ratio_panel": s.aspect_ratio_panel,
            "aspect_ratio_pair": s.aspect_ratio_pair,
            "K_p_per_rad": polhamus_kp(s.aspect_ratio_pair),
            "K_v_le": polhamus_kv_le(s.aspect_ratio_pair),
            "K_v_se": polhamus_kv_se(s.aspect_ratio_pair),
            "K_v": polhamus_kv(s.aspect_ratio_pair),
            "area_one_side_m2": s.S_panel,
            "area_pair_m2": s.S_pair,
            "area_wetted_m2": s.area_wetted,
            "k_upwash": s.k_upwash,
            "t_over_c": s.t_max / s.c_r if s.c_r > 0.0 else 0.0,
        }
