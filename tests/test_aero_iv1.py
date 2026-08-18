"""Tests for rocketgen.sizing.aero_iv1, the two-stage strake-stabilised build-up for IV-1.

VALIDATION REFERENCES, AND WHAT IS AND IS NOT COVERED
=====================================================
This module is an assembly of validated kernels plus three new pieces. The three new pieces are
validated separately, because no single published test case exercises all of them at once.

1. INHERITED PHYSICS: validated by REDUCTION, not by re-testing.
   `test_reduction_to_validated_single_body_*` builds a one-stage stack with the strakes set to
   zero height and asserts that `StackAero.evaluate` reproduces `aero.RocketAero.evaluate` to
   MACHINE PRECISION. `aero.RocketAero` is itself validated against 23 published Basic Finner
   free-flight shots (A. D. Dupuis and W. Hathaway, DREV-TM-9703, 1997, Table VII; see
   tests/test_aero.py). The reduction is therefore not a self-consistency check dressed up as a
   validation: it is the mechanism by which the published validation transfers. If someone
   reimplements a kernel here instead of importing it, the reduction test fails.

2. THE INTERSTAGE SHOULDER: validated against published oblique-shock table values.
   The new `aero.oblique_shock_cp` and `aero.oblique_shock_theta_max` kernels are checked against
   the standard supersonic-flow tables:

       Ames Research Staff, "Equations, Tables and Charts for Compressible Flow",
       NACA Report 1135 (1953), Charts 2 and 5 (oblique shock wave angle and maximum
       deflection angle), and the worked examples of J. D. Anderson,
       Modern Compressible Flow, chapter 4.

   Hard-coded below in OBLIQUE_SHOCK_TABLE and THETA_MAX_TABLE with the citation. This is real
   external reference data for the one genuinely new gas-dynamic kernel, and the tolerance is
   tight because the closed-form solution used here solves the same equations that generated the
   tables: it should agree to the published number of significant figures.

3. THE STRAKE NORMAL FORCE: validated against a PRINTED TABLE of rectangular-wing suction-analogy
   coefficients, and separately compared against the one measured body-with-strakes dataset that
   was found.

   3a. Coefficients, printed table:

           J. E. Lamar and B. B. Gloss, "Subsonic Aerodynamic Characteristics of Interacting
           Lifting Surfaces With Separated Flow Around Sharp Edges Predicted by a Vortex-Lattice
           Method", NASA TN D-7921 (September 1975), Table III, rectangular-wing rows, M = 0.
           https://ntrs.nasa.gov/api/citations/19750023950/downloads/19750023950.pdf

       That table lists K_p, K_v,le and K_v,se for rectangular wings from A = 0.05 to A = 1.0,
       which is AT the strake aspect ratio rather than extrapolated into it. It is transcribed
       below as LAMAR_GLOSS_TABLE_III. The model consumes only the K_v,se column (it lives in
       `aero_iv1.LAMAR_KV_SE_RECTANGULAR`); the K_p and K_v,le columns are NEVER read by the
       model, so checking the closed forms K_p = pi*A/2 and K_v,le = pi*A/4 against them is a
       real, non-circular reproduction test.

   3b. Configuration data, plot-read:

           L. H. Jorgensen and E. R. Nelson, "Experimental Aerodynamic Characteristics for a
           Cylindrical Body of Revolution With Side Strakes and Various Noses at Angles of Attack
           From 0 deg to 58 deg and Mach Numbers From 0.6 to 2.0", NASA TM X-3130 (March 1975),
           figures 18(a) and 22(a).
           https://ntrs.nasa.gov/api/citations/19750011109/downloads/19750011109.pdf

       TM X-3130 publishes computer plots and NO data tables, so the numbers in
       JORGENSEN_TMX3130 below were digitised from those figures and carry +/- 0.2 in CN and
       +/- 0.5 deg in alpha. They are used ONE WAY ONLY: to bound the sign and the size of what
       the model leaves out. They show the model is CONSERVATIVE by a factor of about 2 to 4 on
       the strake normal-force increment, and the tests pin that so the model cannot silently
       start overshooting. They are NOT used to calibrate anything.

   See STRAKE_VALIDATION_NOTE below for what all of this does and does not prove.

Everything else in this module is a limit, a monotonicity, a smoothness, a dimensional or a
bookkeeping check. No reference data is invented anywhere.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import pytest

from rocketgen.config import SOURCES, NtopMeasurements, DesignVector
from rocketgen.config_iv1 import (
    InterceptRequirements,
    StackDesignVector,
    StageSpec,
    StrakeSpec,
    default_iv1,
    lateral_g,
)
from rocketgen.sizing import atmosphere as atm
from rocketgen.sizing.aero import (
    GAMMA,
    RocketAero,
    oblique_shock_cp,
    oblique_shock_theta_max,
)
from rocketgen.sizing.aero_iv1 import (
    ALPHA_MAX_VALID_IV1,
    LAMAR_KV_SE_RECTANGULAR,
    MACH_MAX_VALID,
    StackAero,
    polhamus_cn,
    polhamus_kp,
    polhamus_kv,
    polhamus_kv_le,
    polhamus_kv_se,
)

# ======================================================================================
#   Reference data
# ======================================================================================

#: (Mach, deflection deg, shock angle deg, static pressure ratio p2/p1) for the WEAK solution.
#: Ames Research Staff, NACA Report 1135 (1953), Chart 2 and Table II; the M = 2.0 / 10 deg row
#: is also the worked example in Anderson, Modern Compressible Flow, chapter 4.
OBLIQUE_SHOCK_TABLE: tuple[tuple[float, float, float, float], ...] = (
    (1.5, 10.0, 56.68, 1.666),
    (2.0, 10.0, 39.31, 1.707),
    (2.0, 15.0, 45.34, 2.195),
    (2.0, 20.0, 53.42, 2.843),
    (3.0, 20.0, 37.76, 3.771),
    (4.0, 30.0, 45.22, 9.240),
    (5.0, 20.0, 29.80, 7.037),
)

#: (Mach, maximum attached deflection deg). NACA Report 1135, Chart 5. The M -> infinity limit
#: of 45.58 deg is the classical value.
THETA_MAX_TABLE: tuple[tuple[float, float], ...] = (
    (1.5, 12.11),
    (2.0, 22.97),
    (3.0, 34.07),
    (4.0, 38.77),
    (5.0, 41.12),
    (10.0, 44.43),
    (1.0e4, 45.58),
)

#: NASA TN D-7921 (Lamar and Gloss, 1975) Table III, RECTANGULAR-wing rows, M = 0.
#: `(A, K_p, K_v_le, K_v_se)` from the m1 column, which is the TR R-428 continuous-loading
#: method the authors take as the standard. This is a PRINTED TABLE in the report, not a figure
#: read. The model reads only the K_v,se column, so the K_p and K_v,le columns are independent
#: reference values here.
LAMAR_GLOSS_TABLE_III: tuple[tuple[float, float, float, float], ...] = (
    (0.05, 0.07844, 0.0393, 3.1799),
    (0.10, 0.15710, 0.0785, 3.0188),
    (0.20, 0.31380, 0.1571, 2.7913),
    (0.30, 0.46930, 0.2356, 2.7208),
    (0.40, 0.62270, 0.3141, 2.6341),
    (1.00, 1.46140, 0.7816, 2.1255),
)

#: NASA TM X-3130 (Jorgensen and Nelson, 1975), body N3C1S against N3C1.
#: Geometry from the printed dimensions of figure 1: body diameter d = 0.066 m, cylindrical
#: aftersection 7d long, total span across the two side strakes 1.2d so each strake protrudes
#: 0.1d, strakes run the full length of the cylinder, bevelled leading edge, reference area is
#: the body base area.
JORGENSEN_D: float = 0.066
JORGENSEN_STRAKE_HEIGHT: float = 0.1 * JORGENSEN_D
JORGENSEN_STRAKE_LENGTH: float = 7.0 * JORGENSEN_D
JORGENSEN_N_STRAKE: int = 2

#: `(Mach, alpha_on deg, CN with strakes, alpha_off deg, CN without strakes)`, DIGITISED FROM
#: FIGURES 18(a) and 22(a). Uncertainty +/- 0.2 in CN and +/- 0.5 deg in alpha.
#:
#: BOTH angles of attack are carried on purpose. The strake-on and strake-off curves in those
#: figures are not sampled at the same stations, and the M = 2.0 strake-off curve has no point at
#: 30 deg: its nearest is 32 deg. At the local slope of about 0.43 per degree that 2 deg offset is
#: worth 0.85 in CN, which is as large as the increment itself, so a row whose two angles differ
#: CANNOT be used to form an increment. `test_measured_strake_increment_*` uses matched rows only
#: and says so.
#:
#: alpha is 30 and 58 deg, FAR outside this model's declared 25 deg envelope, which is the other
#: reason these rows bound an omission rather than validate a number.
JORGENSEN_TMX3130: tuple[tuple[float, float, float, float, float], ...] = (
    (0.6, 30.0, 8.3, 30.0, 3.9),
    (0.6, 58.0, 17.6, 58.0, 12.0),
    (2.0, 30.0, 7.4, 32.0, 6.4),
    (2.0, 58.0, 17.1, 58.0, 14.0),
)

#: Rows of JORGENSEN_TMX3130 whose two angles of attack match, so an increment can be formed.
JORGENSEN_MATCHED = tuple(
    (m, a_on, on, off) for m, a_on, on, a_off, off in JORGENSEN_TMX3130 if a_on == a_off
)

STRAKE_VALIDATION_NOTE = """
What was found, and what was not.

FOUND, and used below as reference values:
  * NASA TN D-7921 Table III, a PRINTED table of K_p, K_v,le and K_v,se for RECTANGULAR wings
    from A = 0.05 to A = 1.0. This is the right planform and the right aspect-ratio range for a
    strake. The model reads only its K_v,se column, so the K_p and K_v,le columns independently
    check the closed forms pi*A/2 and pi*A/4.
  * The analytic slender limits stated in the sources themselves: K_p = pi*A/(2E) with E -> 1
    (Polhamus, J. Aircraft 8(4), 1971, eq. 6, after Stewart 1946; equivalently R. T. Jones,
    NACA Report 835, 1946), and K_v,se -> pi as beta*A -> 0 (NASA TR R-428 p. 19, NASA TN D-7921
    p. 9).
  * NASA TM X-3130 figures 18(a) and 22(a), measured CN versus alpha for a cylindrical body with
    and without side strakes, at M 0.6 and 2.0. Digitised, +/- 0.2 in CN.
  * The oblique-shock tables above, for the interstage.
  * The Basic Finner free-flight data, inherited through the reduction test.

NOT FOUND, and therefore NOT asserted:
  * Any body-with-strakes dataset in TABLE form. NASA TM X-3130, TM X-3128, TR R-474, TR R-428
    and NACA TM 798 all publish force data as plots only. A 2006 NSWC survey of the same
    literature (Moore and Rom, "Aerodynamic Analysis of Body-Strake Configurations",
    DTIC ADA460110) reached the same conclusion in one sentence: "Literature search for reliable
    data for very low aspect ratio rectangular wings did not yield result."
  * Any measured strake data inside this model's 25 deg alpha envelope that could be digitised
    with confidence. The TM X-3130 curves cross below about 25 deg and the symbols could not be
    separated reliably at the available scan resolution, so only the alpha 30 and 58 deg points
    are used, and they are used only to bound an omission.
  * Polhamus's own K_p and K_v curves for DELTA wings. They are figures, not tables, they cover
    A = 0.5 to 4, and a delta-wing K_v is the wrong coefficient for a rectangular strake anyway.

WHAT THIS ADDS UP TO. The strake COEFFICIENTS are validated against a printed table of the right
planform at the right aspect ratio. The strake INCREMENT ON A REAL CONFIGURATION is not: the one
measured dataset shows the model is low by a factor of about 2 to 4 at alpha 30 to 58 deg, and
that the real increment grows as Mach falls while this model has no Mach dependence. The gap is
attributed to strake-induced enhancement of the BODY load, which is not modelled. See
SOURCES["iv1_aero_strake_body_interference_on_body_load"]. The direction is favourable for
requirement A11: CN_max is conservative.
"""

ALPHA_CMP = math.radians(5.0)
M_CMP = 3.0
H_CMP = 15_000.0

#: Largest relative CD0 step permitted between adjacent Mach points at 0.005 spacing.
#: Same 3 percent convention as tests/test_aero.py::test_CD0_is_smooth_in_mach.
TOL_CD0_SMOOTH = 0.03
TOL_CN_SMOOTH = 0.03
TOL_XCP_SMOOTH = 0.01

#: The reduction test achieves this. It is a machine-precision bound, not a physics tolerance.
TOL_REDUCTION = 1.0e-13


# ======================================================================================
#   Fixtures
# ======================================================================================


@pytest.fixture(scope="module")
def reqs() -> InterceptRequirements:
    return InterceptRequirements()


@pytest.fixture(scope="module")
def iv1(reqs) -> StackAero:
    return StackAero(default_iv1(), reqs)


def _without_strakes(dv: StackDesignVector) -> StackDesignVector:
    """The same stack with the strake height set to zero. Everything else identical."""
    out = copy.deepcopy(dv)
    out.strakes = StrakeSpec(
        n=dv.strakes.n,
        height=0.0,
        length=dv.strakes.length,
        thickness=dv.strakes.thickness,
        x_le=dv.strakes.x_le,
        sweep_le=dv.strakes.sweep_le,
    )
    return out


@pytest.fixture(scope="module")
def iv1_no_strakes(reqs) -> StackAero:
    return StackAero(_without_strakes(default_iv1()), reqs)


def _single_stage_pair() -> tuple[StackAero, RocketAero]:
    """A one-stage strake-free stack and the equivalent validated single-body model.

    The equivalence is exact by construction: no boattail, base diameter equal to the body
    diameter, fin trailing edge at the base, same nose fineness and shape.
    """
    D, L, f_nose = 0.35, 4.00, 3.0
    fin = {
        "n_fin": 4,
        "b_fin": 0.18,
        "c_r_fin": 0.42,
        "taper_fin": 0.45,
        "sweep_fin": math.radians(45.0),
        "t_fin": 0.012,
    }
    stage = StageSpec(
        index=1, D=D, L=L, m_propellant=200.0, F_thrust=45.0e3, jettisoned=False, **fin
    )
    dv = StackDesignVector(
        stages=[stage],
        strakes=StrakeSpec(n=4, height=0.0, length=1.4, thickness=0.008, x_le=0.9),
        f_nose=f_nose,
        L_interstage=0.0,
    )
    stack = StackAero(dv, InterceptRequirements())
    ref = RocketAero(
        DesignVector(
            D=D,
            L_total=L,
            f_nose=f_nose,
            L_boattail=0.0,
            d_base=D,
            x_fin_te_gap=0.0,
            L_seeker=0.1,
            L_guidance=0.1,
            L_warhead=0.1,
            **fin,
        )
    )
    return stack, ref


def _degenerate_stack() -> StackAero:
    """A two-stage stack whose booster has zero length, zero diameter change and no fins.

    Configurations 1 and 2 then describe the SAME physical vehicle, so any dimensional force
    must be identical between them. This isolates the reference-area plumbing from every other
    difference between the two configurations.
    """
    dv = default_iv1()
    upper = dv.payload_stage
    booster = StageSpec(
        index=1,
        D=upper.D,
        L=0.0,
        m_propellant=0.0,
        F_thrust=0.0,
        n_fin=0,
        b_fin=0.0,
        c_r_fin=0.0,
    )
    deg = StackDesignVector(
        stages=[booster, upper],
        strakes=dv.strakes,
        f_nose=dv.f_nose,
        L_interstage=0.0,
    )
    return StackAero(deg, InterceptRequirements())


# ======================================================================================
#   VALIDATION 1: the interstage shoulder against published oblique-shock tables
# ======================================================================================


def _beta_and_pressure_ratio(mach: float, theta: float) -> tuple[float, float]:
    """Recover (shock angle rad, p2/p1) from the returned Cp, so the tables can be compared."""
    cp, _attached = oblique_shock_cp(mach, theta)
    msb2 = cp * (GAMMA + 1.0) / 4.0 * mach * mach + 1.0
    beta = math.asin(math.sqrt(msb2) / mach)
    p21 = 1.0 + 2.0 * GAMMA / (GAMMA + 1.0) * (msb2 - 1.0)
    return beta, p21


@pytest.mark.parametrize("mach,theta_deg,beta_deg,p21", OBLIQUE_SHOCK_TABLE)
def test_validation_oblique_shock_matches_naca_1135(mach, theta_deg, beta_deg, p21):
    """The closed-form weak solution must reproduce NACA Report 1135 to its printed precision.

    Tolerance 0.1 percent. Justified: the closed form solves the same theta-beta-Mach relation
    that generated the tables, so the only difference is the rounding of the tabulated value.
    Anything larger means the cubic root selection or the Rankine-Hugoniot step is wrong.
    """
    beta, ratio = _beta_and_pressure_ratio(mach, math.radians(theta_deg))
    assert math.degrees(beta) == pytest.approx(beta_deg, rel=1.0e-3)
    assert ratio == pytest.approx(p21, rel=1.0e-3)


@pytest.mark.parametrize("mach,theta_max_deg", THETA_MAX_TABLE)
def test_validation_max_deflection_matches_naca_1135(mach, theta_max_deg):
    """Maximum attached deflection against NACA Report 1135 Chart 5, to 0.2 percent.

    The detachment angle is what decides whether the interstage shoulder shock is attached, so
    getting it right is what makes the shoulder drag branch selection right.
    """
    assert math.degrees(oblique_shock_theta_max(mach)) == pytest.approx(
        theta_max_deg, rel=2.0e-3
    )


def test_oblique_shock_detachment_is_detected_and_clamped():
    """Past detachment there is no weak solution, so the value must clamp, not blow up."""
    delta = math.radians(30.0)
    cp_att, att = oblique_shock_cp(3.0, delta)          # M 3 detaches above 34.07 deg
    assert att is True
    cp_det, det = oblique_shock_cp(2.0, delta)          # M 2 detaches above 22.97 deg
    assert det is False
    assert cp_det > 0.0 and math.isfinite(cp_det)
    # The clamp must equal the value at the maximum attached deflection, which is what makes it
    # continuous in Mach.
    cp_max, _ = oblique_shock_cp(2.0, oblique_shock_theta_max(2.0))
    assert cp_det == pytest.approx(cp_max, rel=1.0e-6)
    assert cp_att > 0.0


def test_shoulder_drag_is_bracketed_by_the_stagnation_bound(iv1):
    """No shock system can put more than stagnation pressure on the shoulder annulus."""
    for mach in (1.3, 1.6, 2.0, 3.0, 4.0, 5.0):
        r = iv1.evaluate(mach, H_CMP, 0.0, 1)
        cd = r.breakdown["CD_interstage_shoulder"]
        assert cd > 0.0
        assert cd < r.breakdown["xcheck_CD_shoulder_stagnation"], (
            f"shoulder drag exceeds the stagnation bound at M {mach}"
        )


def test_shoulder_method_sensitivity_is_small_above_mach_2(iv1):
    """Oblique shock against isentropic compression: the two agree once the shock is weak.

    This is the reported cross-check doing its job. Below M 2 the entropy rise matters and the
    two diverge, which is why the oblique-shock value is the one that is summed.
    """
    for mach in (2.0, 3.0, 4.0, 5.0):
        r = iv1.evaluate(mach, H_CMP, 0.0, 1)
        shock = r.breakdown["CD_interstage_shoulder"]
        isen = r.breakdown["xcheck_CD_shoulder_isentropic"]
        assert abs(shock - isen) / shock < 0.05, f"methods differ by more than 5 pc at M {mach}"


# ======================================================================================
#   VALIDATION 2: the strake normal-force method against its published closed forms
# ======================================================================================


def test_strake_validation_note_states_what_was_and_was_not_found():
    """The note must exist and must say plainly which claims are and are not measured."""
    assert "NOT FOUND" in STRAKE_VALIDATION_NOTE
    assert "is not:" in STRAKE_VALIDATION_NOTE
    assert "conservative" in STRAKE_VALIDATION_NOTE


@pytest.mark.parametrize("ar,kp,kv_le,kv_se", LAMAR_GLOSS_TABLE_III)
def test_validation_Kp_reproduces_the_lamar_gloss_printed_table(ar, kp, kv_le, kv_se):
    """K_p = pi*A/2 against NASA TN D-7921 Table III, a column the model never reads.

    Tolerance is aspect-ratio dependent, and stated rather than fitted: the closed form is the
    slender limit, so it is exact as A -> 0 and drifts as A grows. Measured drift against the
    table is 0.13 percent at A = 0.05, 0.01 percent at A = 0.10, 0.9 percent at A = 0.40 and
    7.5 percent at A = 1.00. A strake pair aspect ratio lies between 0.014 and 0.20 across the
    whole SPEC_IV1.md section 4 design box, so 1 percent is the accuracy that matters.
    """
    del kv_le, kv_se
    tol = 0.02 if ar <= 0.40 else 0.08
    assert polhamus_kp(ar) == pytest.approx(kp, rel=tol)
    if ar <= 0.20:
        assert polhamus_kp(ar) == pytest.approx(kp, rel=0.005), (
            "the slender-wing limit must be good to 0.5 percent over the strake range"
        )


@pytest.mark.parametrize("ar,kp,kv_le,kv_se", LAMAR_GLOSS_TABLE_III)
def test_validation_Kv_le_reproduces_the_lamar_gloss_printed_table(ar, kp, kv_le, kv_se):
    """K_v,le = pi*A/4 = K_p/2 against the same printed table, to 0.1 percent up to A = 0.40."""
    del kp, kv_se
    tol = 0.001 if ar <= 0.40 else 0.01
    assert polhamus_kv_le(ar) == pytest.approx(kv_le, rel=tol)
    assert polhamus_kv_le(ar) == pytest.approx(0.5 * polhamus_kp(ar), rel=1.0e-15)


@pytest.mark.parametrize("ar,kp,kv_le,kv_se", LAMAR_GLOSS_TABLE_III)
def test_the_Kv_se_table_in_the_model_is_the_published_one(ar, kp, kv_le, kv_se):
    """Wiring check, NOT a validation: the model must carry the table verbatim at its nodes.

    K_v,se has no useful closed form, so the model consumes the published column directly. That
    makes this a transcription check. It is labelled as such so it is never mistaken for a
    reproduction of independent data.
    """
    del kp, kv_le
    assert polhamus_kv_se(ar) == pytest.approx(kv_se, rel=1.0e-12)
    nodes = dict(LAMAR_KV_SE_RECTANGULAR)
    assert nodes[ar] == pytest.approx(kv_se, rel=1.0e-12)


def test_validation_Kv_se_approaches_pi_at_vanishing_aspect_ratio():
    """The analytic limit stated in NASA TR R-428 p. 19 and TN D-7921 p. 9: K_v,se -> pi."""
    assert polhamus_kv_se(0.0) == pytest.approx(math.pi, rel=1.0e-15)
    assert polhamus_kv_se(1.0e-9) == pytest.approx(math.pi, rel=1.0e-6)
    assert LAMAR_KV_SE_RECTANGULAR[0] == (0.0, math.pi)
    # And the table's own A = 0.05 row must already be within a couple of percent of pi, which
    # is what makes the limit usable at a strake aspect ratio.
    assert LAMAR_GLOSS_TABLE_III[0][3] == pytest.approx(math.pi, rel=0.02)


def test_validation_polhamus_form_is_the_published_two_term_expression():
    """CN = K_p sin(a) cos^2(a) + K_v sin^2(a) cos(a), term by term, at a strake aspect ratio.

    NASA TN D-3767 eq. (15) and Polhamus, J. Aircraft 8(4), 1971, eq. (4).
    """
    ar = 0.0429
    for alpha_deg in (3.0, 8.0, 16.0, 24.0):
        a = math.radians(alpha_deg)
        pot, vor = polhamus_cn(ar, a)
        assert pot == pytest.approx(
            polhamus_kp(ar) * math.sin(a) * math.cos(a) ** 2, rel=1.0e-14
        )
        assert vor == pytest.approx(
            polhamus_kv(ar) * math.sin(a) ** 2 * math.cos(a), rel=1.0e-14
        )
        assert polhamus_kv(ar) == pytest.approx(
            polhamus_kv_le(ar) + polhamus_kv_se(ar), rel=1.0e-15
        )
    # At zero aspect ratio only the side-edge vortex term survives.
    a = math.radians(20.0)
    pot0, vor0 = polhamus_cn(0.0, a)
    assert pot0 == pytest.approx(0.0, abs=1.0e-15)
    assert vor0 == pytest.approx(math.pi * math.sin(a) ** 2 * math.cos(a), rel=1.0e-14)


def test_the_suction_analogy_is_well_above_the_newtonian_term_it_replaces():
    """K_v is near pi, not the 2.0 of the Newtonian term in the inherited fin model.

    This is the quantitative reason the strake gets its own method instead of being pushed
    through `aero.lifting_surface_cn_alone`. If this ratio ever collapsed to 1, the strake
    method would have stopped adding anything.
    """
    for ar in (0.0, 0.02, 0.05, 0.10, 0.20):
        assert polhamus_kv(ar) > 1.30 * 2.0
        assert polhamus_kv(ar) < 1.70 * 2.0


# --------------------------------------------------------------------------------------
#   VALIDATION 2b: the one measured body-with-strakes configuration found
# --------------------------------------------------------------------------------------


def _jorgensen_strake_model() -> StackAero:
    """The TM X-3130 body N3C1S strake geometry, as a StackAero.

    Only the strake normal-force increment is taken from this model, and that term depends on
    the strake geometry, the body diameter and the reference area alone. It does NOT depend on
    the nose, which is just as well: TM X-3130 calls the nose N3 and its shape is not needed
    here. The fins are absent because the test body has none.
    """
    body = StageSpec(
        index=1,
        D=JORGENSEN_D,
        L=12.0 * JORGENSEN_D,        # nose l_N/d = 5 plus the 7d cylinder
        m_propellant=0.0,
        F_thrust=0.0,
        n_fin=0,
        b_fin=0.0,
        c_r_fin=0.0,
        jettisoned=False,
    )
    dv = StackDesignVector(
        stages=[body],
        strakes=StrakeSpec(
            n=JORGENSEN_N_STRAKE,
            height=JORGENSEN_STRAKE_HEIGHT,
            length=JORGENSEN_STRAKE_LENGTH,
            thickness=0.002,
            x_le=5.0 * JORGENSEN_D,
        ),
        f_nose=5.0,
        L_interstage=0.0,
    )
    return StackAero(dv, InterceptRequirements())


def test_the_jorgensen_test_body_geometry_is_reproduced():
    """Reference area is the body base area and the strake aspect ratio is what the report says."""
    model = _jorgensen_strake_model()
    assert model.S_ref(1) == pytest.approx(
        0.25 * math.pi * JORGENSEN_D ** 2, rel=1.0e-15
    )
    # TM X-3130 gives the reference area as 34.26 cm^2. The 0.15 percent difference is the
    # report's diameter being 6.604 cm rather than the 6.6 cm used here.
    assert model.S_ref(1) == pytest.approx(34.26e-4, rel=3.0e-3)
    summary = model.strake_summary(1)
    assert summary["n_panels"] == 2.0
    # Two side strakes, total span across them 1.2 d, so the pair aspect ratio is small.
    assert 0.02 < summary["aspect_ratio_pair"] < 0.04
    assert summary["K_v"] == pytest.approx(math.pi, rel=0.02)


@pytest.mark.parametrize("mach,alpha_deg,cn_on,cn_off", JORGENSEN_MATCHED)
def test_measured_strake_increment_bounds_the_unmodelled_body_enhancement(
    mach, alpha_deg, cn_on, cn_off
):
    """The model's strake increment must be POSITIVE and BELOW the measured increment.

    What this proves, and only this: the model is conservative, and by a bounded factor. It does
    NOT prove the model is right, because the measured increment is 1.4 to 3.4 times larger on
    the three usable rows. The gap is strake-induced enhancement of the BODY load, which
    SOURCES["iv1_aero_strake_body_interference_on_body_load"] declares as not modelled.

    Matched rows only. See the note on JORGENSEN_TMX3130 for why the M 2.0, 30 deg row is
    excluded: its strake-off curve has no point at 30 deg.

    The measured increment carries +/- 0.4 in CN, from +/- 0.2 on each of the two curves, so the
    upper bound is stated with that slack. alpha is 30 and 58 deg, outside this model's declared
    25 deg envelope, which is the other reason this is a bound and not a tolerance.
    """
    model = _jorgensen_strake_model()
    a = math.radians(alpha_deg)
    pot, vor = model.CN_strakes(mach, a, 1)
    modelled = pot + vor
    measured = cn_on - cn_off

    assert modelled > 0.0
    assert measured > 0.0
    assert modelled < measured + 0.4, (
        f"the model overshoots the measured strake increment at M {mach}, alpha {alpha_deg} deg: "
        f"model {modelled:.3f} against measured {measured:.3f} +/- 0.4"
    )
    # The shortfall must also stay inside the factor recorded in SOURCES, so a change that made
    # the model wildly low would be caught too.
    ratio = measured / modelled
    assert 1.2 < ratio < 5.0, (
        f"measured/model strake increment is {ratio:.2f} at M {mach}, alpha {alpha_deg} deg, "
        "outside the factor of 1.4 to 3.4 recorded in SOURCES"
    )


def test_the_excluded_jorgensen_row_is_excluded_for_the_stated_reason():
    """Guard the exclusion itself, so it cannot become a convenient way to drop a hard point."""
    excluded = [row for row in JORGENSEN_TMX3130 if row[1] != row[3]]
    assert len(excluded) == 1
    mach, a_on, _cn_on, a_off, _cn_off = excluded[0]
    assert (mach, a_on, a_off) == (2.0, 30.0, 32.0)
    assert len(JORGENSEN_MATCHED) == 3


def test_the_measured_increment_grows_as_mach_falls_and_the_model_does_not():
    """A declared omission, asserted so it cannot be forgotten: the model has no Mach dependence.

    TM X-3130 shows a larger strake increment at M 0.6 than at M 2.0 at 58 deg, where both
    curves have a point. The suction analogy as implemented is Mach-independent, so it cannot
    reproduce that trend. The test pins both halves of the statement.
    """
    rows = {(m, a): (on, off) for m, a, on, off in JORGENSEN_MATCHED}
    subsonic = rows[(0.6, 58.0)]
    supersonic = rows[(2.0, 58.0)]
    assert (subsonic[0] - subsonic[1]) > (supersonic[0] - supersonic[1]), (
        "the digitised data no longer show the trend the note describes"
    )
    model = _jorgensen_strake_model()
    for alpha_deg in (10.0, 30.0, 58.0):
        a = math.radians(alpha_deg)
        assert model.CN_strakes(0.6, a, 1) == model.CN_strakes(2.0, a, 1)


def test_vortex_lift_dominates_a_strake_by_an_order_of_magnitude(iv1):
    """The point of using a nonlinear method at all: linear theory gives almost nothing.

    At the default strake aspect ratio of about 0.0214, K_p is 0.0337 per radian against
    K_v = 2.0, a ratio of 59. At 20 deg the two terms carry the tan(alpha) factor between them,
    and the potential term additionally gets the body-upwash factor of 1.82, so the measured
    ratio of vortex to linear normal force is about 12. Asserted above 8 so a real change in the
    method is caught while the exact factor is free to move.
    """
    a = math.radians(20.0)
    pot, vor = iv1.CN_strakes(M_CMP, a, 2)
    assert pot > 0.0
    assert vor > 8.0 * pot, (
        f"vortex term {vor:.5f} is only {vor / pot:.1f} times the linear term {pot:.5f}"
    )
    # Without the upwash boost on the potential term the raw ratio is K_v/(K_p) * tan(alpha).
    raw_pot, raw_vor = polhamus_cn(0.0214, a)
    assert raw_vor / raw_pot > 15.0
    summary = iv1.strake_summary(2)
    assert summary["aspect_ratio_panel"] < 0.05
    assert summary["K_p_per_rad"] < 0.1


def test_strake_normal_force_grows_faster_than_linearly_with_alpha(iv1):
    """The vortex-lift signature: the secant slope CN/alpha must RISE with alpha.

    A linear surface has a constant secant slope. Over 2 to 25 deg the strake secant slope must
    increase monotonically, and by a large factor, or the vortex term is not doing its job.
    """
    alphas = np.radians(np.linspace(2.0, 25.0, 24))
    secant = []
    for a in alphas:
        pot, vor = iv1.CN_strakes(M_CMP, float(a), 2)
        secant.append((pot + vor) / float(a))
    sec = np.array(secant)
    assert np.all(np.diff(sec) > 0.0), "strake secant slope is not monotone increasing in alpha"
    assert sec[-1] / sec[0] > 3.0, f"secant slope only grew by {sec[-1] / sec[0]:.2f}"


def test_strake_contribution_vanishes_as_height_goes_to_zero(reqs):
    """The limit that must hold whatever the method: no strake, no strake load or drag."""
    dv = default_iv1()
    previous = None
    for height in (0.030, 0.010, 0.003, 0.001, 1.0e-6):
        d = copy.deepcopy(dv)
        d.strakes = StrakeSpec(
            n=4, height=height, length=dv.strakes.length,
            thickness=dv.strakes.thickness, x_le=dv.strakes.x_le,
        )
        r = StackAero(d, reqs).evaluate(M_CMP, H_CMP, math.radians(15.0), 2)
        cn = r.breakdown["CN_strakes"]
        cd = (
            r.breakdown["CD_strake_friction"]
            + r.breakdown["CD_strake_wave"]
            + r.breakdown["CD_strake_base"]
        )
        assert cn > 0.0 and cd > 0.0
        if previous is not None:
            assert cn < previous[0] and cd < previous[1]
        previous = (cn, cd)
    assert previous[0] < 1.0e-4 and previous[1] < 1.0e-6

    zero = StackAero(_without_strakes(dv), reqs).evaluate(
        M_CMP, H_CMP, math.radians(15.0), 2
    )
    assert zero.breakdown["CN_strakes"] == 0.0
    assert zero.breakdown["CD_strake_friction"] == 0.0
    assert zero.breakdown["CD_strake_wave"] == 0.0
    assert zero.breakdown["CD_strake_base"] == 0.0


# ======================================================================================
#   REDUCTION to the validated single-body model
# ======================================================================================


def test_reduction_to_validated_single_body_coefficients():
    """A one-stage strake-free stack MUST equal aero.RocketAero to machine precision.

    This is what carries the Basic Finner validation into this module. The tolerance is 1e-13
    relative, and the measured worst case is about 5e-16, i.e. floating-point round-off. It is
    tight on purpose: anything looser would let a reimplemented kernel slip through.
    """
    stack, ref = _single_stage_pair()
    worst = 0.0
    worst_where = ""
    for mach in (0.3, 0.5, 0.9, 0.95, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0):
        for alpha in (0.0, 0.01, 0.05, 0.15, -0.10, 0.26):
            for power_on in (False, True):
                for h in (0.0, 12_000.0, 25_000.0):
                    a = stack.evaluate(mach, h, alpha, 1, power_on)
                    b = ref.evaluate(mach, h, alpha, power_on)
                    for key in ("CD0", "CD", "CN", "CN_alpha", "CM", "x_cp", "L_over_D"):
                        va, vb = getattr(a, key), getattr(b, key)
                        rel = abs(va - vb) / max(abs(vb), 1.0e-12)
                        if rel > worst:
                            worst, worst_where = rel, f"{key} at M {mach} alpha {alpha}"
    assert worst < TOL_REDUCTION, f"worst relative difference {worst:.3e} at {worst_where}"


def test_reduction_of_the_component_breakdown():
    """Every drag and normal-force component the two models share must also agree exactly."""
    stack, ref = _single_stage_pair()
    shared = (
        "CD_friction_body", "CD_wave_body", "CD_base", "CD_boattail",
        "CD_fin_friction", "CD_fin_wave", "CD_protuberance_GUESS", "CD_induced",
        "CN_body_potential", "CN_body_crossflow", "CN_fins",
        "x_cp_potential_m", "x_cp_crossflow_m",
    )
    for mach in (0.4, 0.97, 1.15, 2.2, 4.5):
        a = stack.evaluate(mach, 8000.0, math.radians(6.0), 1)
        b = ref.evaluate(mach, 8000.0, math.radians(6.0))
        for key in shared:
            assert a.breakdown[key] == pytest.approx(
                b.breakdown[key], rel=TOL_REDUCTION, abs=1.0e-15
            ), f"{key} differs at M {mach}"
    # And the new terms must all be identically zero in the reduced configuration.
    a = stack.evaluate(2.2, 8000.0, math.radians(6.0), 1)
    for key in (
        "CD_interstage_shoulder", "CD_strake_friction", "CD_strake_wave",
        "CD_strake_base", "CD_strake_interference_NOT_MODELLED", "CN_strakes",
    ):
        assert a.breakdown[key] == 0.0, f"{key} is not zero in the reduced configuration"


def test_reduction_of_trim_alpha():
    """The trim solver must be the shared one, so it must return identical angles."""
    stack, ref = _single_stage_pair()
    a_max = math.radians(15.0)
    for mach in (0.4, 0.9, 1.1, 2.0, 3.5, 5.0):
        for target in (0.05, 0.3, 1.0, 2.0, 1.0e6):
            got = stack.trim_alpha(mach, 12_000.0, target, 1, alpha_max=a_max)
            want = ref.trim_alpha(mach, 12_000.0, target, alpha_max=a_max)
            assert got == pytest.approx(want, rel=1.0e-14, abs=1.0e-15)


# ======================================================================================
#   REFERENCE AREA: the thing that silently corrupts a trajectory if it is wrong
# ======================================================================================


def test_S_ref_uses_the_booster_before_separation_and_the_payload_stage_after(iv1):
    """S_ref(1) is the booster cross-section, S_ref(2) the payload stage's own."""
    dv = default_iv1()
    D1, D2 = dv.booster.D, dv.payload_stage.D
    assert D2 < D1, "the fixture must have a stepped stack for this test to mean anything"
    assert iv1.S_ref(1) == pytest.approx(0.25 * math.pi * D1 * D1, rel=1.0e-15)
    assert iv1.S_ref(2) == pytest.approx(0.25 * math.pi * D2 * D2, rel=1.0e-15)
    assert iv1.D_ref(1) == pytest.approx(D1, rel=1.0e-15)
    assert iv1.D_ref(2) == pytest.approx(D2, rel=1.0e-15)
    assert iv1.S_ref(2) < iv1.S_ref(1)
    assert iv1.geom[1].L_total > iv1.geom[2].L_total


def test_unknown_stage_raises_rather_than_silently_using_the_wrong_area(iv1):
    with pytest.raises(KeyError):
        iv1.S_ref(3)
    with pytest.raises(KeyError):
        iv1.evaluate(2.0, 10_000.0, 0.1, 7)


def test_dimensional_force_is_continuous_across_separation_for_surviving_surfaces(iv1):
    """The strakes and the stage-2 fins survive separation, so their FORCE cannot jump.

    Their coefficients MUST jump, because the reference area changes. The product
    `CN * S_ref` must not. This is exactly the failure mode the reference-area plumbing has to
    prevent, and it is asserted to machine precision because nothing physical changes for these
    surfaces at separation.
    """
    S1, S2 = iv1.S_ref(1), iv1.S_ref(2)
    for mach in (0.8, 2.0, 3.5, 5.0):
        for alpha_deg in (4.0, 12.0, 20.0):
            a = math.radians(alpha_deg)
            r1 = iv1.evaluate(mach, H_CMP, a, 1)
            r2 = iv1.evaluate(mach, H_CMP, a, 2)
            for key in ("CN_strakes", "CN_fins_stage2"):
                # The coefficient must differ, by exactly the area ratio.
                assert r1.breakdown[key] != pytest.approx(r2.breakdown[key], rel=1.0e-6)
                f1 = r1.breakdown[key] * S1
                f2 = r2.breakdown[key] * S2
                assert f1 == pytest.approx(f2, rel=1.0e-13), (
                    f"{key} force jumps at separation, M {mach} alpha {alpha_deg} deg"
                )
            # The stage-2 fin station cannot move either: same datum, same geometry.
            assert r1.breakdown["x_cp_fins_stage2_m"] == pytest.approx(
                r2.breakdown["x_cp_fins_stage2_m"], rel=1.0e-15
            )


def test_dimensional_force_is_continuous_for_a_stack_with_nothing_to_jettison():
    """With a zero-length, zero-fin, same-diameter booster the two configurations are one body.

    Every dimensional force and the centre of pressure must then be identical. This isolates the
    reference-area bookkeeping: if S_ref(1) and S_ref(2) were mixed up anywhere, this fails.
    """
    deg = _degenerate_stack()
    assert deg.S_ref(1) == pytest.approx(deg.S_ref(2), rel=1.0e-15)
    q = 50_000.0
    for mach in (0.4, 0.9, 1.3, 2.0, 3.5, 5.0):
        for alpha in (0.0, 0.05, 0.20):
            a = deg.evaluate(mach, 10_000.0, alpha, 1)
            b = deg.evaluate(mach, 10_000.0, alpha, 2)
            assert q * deg.S_ref(1) * a.CD0 == pytest.approx(
                q * deg.S_ref(2) * b.CD0, rel=1.0e-14
            )
            assert q * deg.S_ref(1) * a.CN == pytest.approx(
                q * deg.S_ref(2) * b.CN, rel=1.0e-14, abs=1.0e-18
            )
            assert a.x_cp == pytest.approx(b.x_cp, rel=1.0e-14)


def test_total_force_across_separation_changes_only_by_the_jettisoned_parts(iv1):
    """The TOTAL force is not continuous, and it must not be. Account for the difference.

    Stage 1 jettisons the booster body, the booster fins and the interstage. The residual is
    therefore not zero, and this test records what it is made of rather than asserting a number
    the model cannot support:

      * the booster fin load leaves entirely,
      * the interstage shoulder drag leaves entirely,
      * the booster body wetted, planform and base areas leave, and
      * the SURVIVING body's slender-body potential lift changes, because that term is
        proportional to the base area of whatever body is now aft-most, which is correct
        physics rather than a bookkeeping error.

    The one thing that must hold is the direction: the stack is bigger, so it carries more
    axial force and more normal force at the same flight condition.
    """
    mach, alpha = 3.0, math.radians(10.0)
    S1, S2 = iv1.S_ref(1), iv1.S_ref(2)
    r1 = iv1.evaluate(mach, H_CMP, alpha, 1)
    r2 = iv1.evaluate(mach, H_CMP, alpha, 2)
    assert r1.CD0 * S1 > r2.CD0 * S2
    assert r1.CN * S1 > r2.CN * S2

    # Named accounting of the normal-force difference.
    booster_fins = r1.breakdown["CN_fins_stage1"] * S1
    body_potential_change = (
        r1.breakdown["CN_body_potential"] * S1 - r2.breakdown["CN_body_potential"] * S2
    )
    body_crossflow_change = (
        r1.breakdown["CN_body_crossflow"] * S1 - r2.breakdown["CN_body_crossflow"] * S2
    )
    residual = (r1.CN * S1 - r2.CN * S2) - (
        booster_fins + body_potential_change + body_crossflow_change
    )
    assert residual == pytest.approx(0.0, abs=1.0e-12), (
        "the normal-force jump at separation is not fully explained by the booster fins plus "
        f"the body terms; unexplained residual {residual:.3e} m^2"
    )
    assert booster_fins > 0.0
    # The surviving body's potential lift FALLS, because the base area it is referred to falls.
    assert body_potential_change > 0.0

    # And the axial-force difference must contain the whole interstage shoulder.
    assert r1.breakdown["CD_interstage_shoulder"] > 0.0
    assert r2.breakdown["CD_interstage_shoulder"] == 0.0


def test_the_shoulder_exists_only_when_there_is_a_diameter_step(reqs):
    """No step, no shoulder drag. The wetted and planform areas of the adapter stay."""
    dv = default_iv1()
    same = copy.deepcopy(dv)
    same.stages[0].D = same.stages[1].D
    flush = StackAero(same, reqs)
    assert flush.geom[1].shoulder_annulus == 0.0
    assert flush.evaluate(3.0, H_CMP, 0.0, 1).breakdown["CD_interstage_shoulder"] == 0.0
    # The interstage is still there, so it still has friction drag and planform area.
    assert flush.geom[1].area_wetted_body > flush.geom[2].area_wetted_body
    assert flush.geom[1].area_planform_body > flush.geom[2].area_planform_body


# ======================================================================================
#   Strake effect on stability: the whole reason strakes have to be in the moment balance
# ======================================================================================


def test_strakes_move_the_centre_of_pressure_forward_on_the_stack(iv1, iv1_no_strakes):
    """A load applied ahead of the existing centre of pressure moves it forward.

    On the stacked configuration the strake load centroid is well forward of the booster tail
    fins, so the strakes must reduce static margin. This is the effect SPEC_IV1.md A9 has to see.
    """
    for mach in (0.8, 2.0, 3.0, 5.0):
        on = iv1.evaluate(mach, H_CMP, math.radians(10.0), 1)
        off = iv1_no_strakes.evaluate(mach, H_CMP, math.radians(10.0), 1)
        shift_cal = (on.x_cp - off.x_cp) / iv1.D_ref(1)
        assert shift_cal < 0.0, f"strakes did not move x_cp forward at M {mach}"
        assert abs(shift_cal) > 0.05, (
            f"strake effect on x_cp is only {shift_cal:.4f} calibres at M {mach}"
        )


def test_strakes_reduce_static_margin_on_the_stack(iv1, iv1_no_strakes):
    """Stated as the sizing loop sees it: static margin in calibres about a fixed CG."""
    x_cg = 2.6
    sm_on = iv1.static_margin(3.0, H_CMP, math.radians(10.0), 1, x_cg)
    sm_off = iv1_no_strakes.static_margin(3.0, H_CMP, math.radians(10.0), 1, x_cg)
    assert sm_on < sm_off
    assert sm_off - sm_on > 0.05


def test_on_the_payload_stage_alone_the_default_strake_barely_moves_the_cp(iv1, iv1_no_strakes):
    """Recorded, not hidden: the DEFAULT strake does almost nothing to stage-2 stability.

    A strake only moves the centre of pressure if its own load centroid is away from the
    vehicle's. The default strake runs 0.95 to 2.35 m with its centroid at 1.65 m, and the
    payload stage's centre of pressure sits at 1.60 to 1.70 m over most of the Mach range, so on
    the surviving vehicle the strake load lands almost exactly ON the existing centre of
    pressure. The shift is therefore small and it CHANGES SIGN with Mach: forward subsonically,
    where the tail fins are strong and the vehicle cp is aft, and aft above about M 2.5, where
    the fin load has fallen as 1/sqrt(M^2-1) and the vehicle cp has moved ahead of the strake.

    On the stacked configuration there is no such cancellation: the booster fins hold the centre
    of pressure far aft of the strake, so the strake always moves it forward. That is the case
    `test_strakes_move_the_centre_of_pressure_forward_on_the_stack` covers.

    None of this is a defect. It is why the sizing loop has to check static margin for BOTH
    configurations rather than assuming the strake effect has one sign.
    """
    shifts = {}
    for mach in (0.5, 1.0, 2.0, 3.0, 5.0):
        on = iv1.evaluate(mach, H_CMP, math.radians(10.0), 2)
        off = iv1_no_strakes.evaluate(mach, H_CMP, math.radians(10.0), 2)
        shifts[mach] = (on.x_cp - off.x_cp) / iv1.D_ref(2)
    assert shifts[0.5] < 0.0, "expected a forward shift subsonically"
    assert shifts[5.0] > 0.0, "expected an aft shift at high Mach"
    assert max(abs(v) for v in shifts.values()) < 0.15, (
        f"the stage-2 shift is no longer small: {shifts}"
    )
    # The cancellation is geometric, so name it: the strake centroid is near the vehicle cp.
    strake = iv1.geom[2].strake_set
    assert strake is not None
    x_cp_vehicle = iv1.evaluate(2.0, H_CMP, math.radians(10.0), 2).x_cp
    assert abs(strake.x_cp_sub - x_cp_vehicle) < 0.15 * iv1.geom[2].L_total


def test_moving_a_strake_forward_moves_the_centre_of_pressure_forward(reqs):
    """Monotone in the strake station, in BOTH configurations."""
    dv = default_iv1()
    for stage in (1, 2):
        stations = []
        for x_le in (0.60, 0.80, 0.95, 1.10, 1.25):
            d = copy.deepcopy(dv)
            d.strakes = StrakeSpec(
                n=4, height=0.030, length=1.20, thickness=0.008, x_le=x_le
            )
            r = StackAero(d, reqs).evaluate(3.0, H_CMP, math.radians(10.0), stage)
            stations.append(r.x_cp)
        assert np.all(np.diff(np.array(stations)) > 0.0), (
            f"x_cp is not monotone in the strake station for stage {stage}: {stations}"
        )


def test_the_strake_load_centroid_is_forward_of_every_tail_fin(iv1):
    """The geometric statement behind the stability effect."""
    for stage in (1, 2):
        g = iv1.geom[stage]
        strake = g.strake_set
        assert strake is not None
        for fins in g.fin_sets:
            assert strake.x_cp_sub < iv1.x_cp_surface(fins, 3.0), (
                f"the strake centroid is not forward of {fins.label} in stage {stage}"
            )
        assert strake.x_cp_sub == pytest.approx(
            strake.x_le + 0.5 * strake.c_r, rel=1.0e-15
        )


# ======================================================================================
#   CN_max, requirement A11
# ======================================================================================


def test_CN_max_is_the_value_at_the_alpha_limit(iv1):
    """CN_max must be exactly what evaluate reports at that alpha, on the same S_ref."""
    for stage in (1, 2):
        for alpha_deg in (10.0, 15.0, 20.0, 25.0):
            a = math.radians(alpha_deg)
            assert iv1.CN_max(3.0, H_CMP, stage, a) == pytest.approx(
                iv1.evaluate(3.0, H_CMP, a, stage).CN, rel=1.0e-12
            )
            # Sign-insensitive: it is a capability, not a signed load.
            assert iv1.CN_max(3.0, H_CMP, stage, -a) == pytest.approx(
                iv1.CN_max(3.0, H_CMP, stage, a), rel=1.0e-15
            )


def test_CN_max_rises_monotonically_with_strake_height(reqs):
    """Taller strakes must buy more normal force. Requirement A11 is why they are there."""
    dv = default_iv1()
    a20 = math.radians(20.0)
    for stage in (1, 2):
        values = []
        for height in (0.0, 0.015, 0.030, 0.045, 0.060):
            d = copy.deepcopy(dv)
            d.strakes = StrakeSpec(
                n=4, height=height, length=1.40, thickness=0.008, x_le=0.95
            )
            values.append(StackAero(d, reqs).CN_max(3.0, H_CMP, stage, a20))
        v = np.array(values)
        assert np.all(np.diff(v) > 0.0), f"CN_max not monotone in strake height: {values}"
        assert v[-1] > v[0], "the tallest strake must beat no strake at all"


def test_CN_max_contains_the_strake_contribution(iv1, iv1_no_strakes):
    """The strakes must actually appear in the A11 number, on both configurations."""
    a20 = math.radians(20.0)
    for stage in (1, 2):
        on = iv1.CN_max(3.0, H_CMP, stage, a20)
        off = iv1_no_strakes.CN_max(3.0, H_CMP, stage, a20)
        assert on > off
        gain = (on - off) / off
        assert gain > 0.02, f"strakes add only {gain:.2%} of CN_max in stage {stage}"


def test_CN_max_rises_with_the_alpha_limit(iv1):
    for stage in (1, 2):
        values = [
            iv1.CN_max(3.0, H_CMP, stage, math.radians(d))
            for d in (5.0, 10.0, 15.0, 20.0, 25.0)
        ]
        assert np.all(np.diff(np.array(values)) > 0.0)


def test_lateral_acceleration_at_intercept_is_dimensionally_sane(iv1, reqs):
    """A11 as the loop will compute it: q * S_ref * CN_max / (m g). Units and sign only."""
    q = 30_000.0
    mass = 300.0
    g_avail = lateral_g(q, iv1.S_ref(2), iv1.CN_max(3.0, H_CMP, 2, reqs.alpha_max), mass)
    assert g_avail > 0.0
    hand = q * iv1.S_ref(2) * iv1.CN_max(3.0, H_CMP, 2, reqs.alpha_max) / (mass * 9.80665)
    assert g_avail == pytest.approx(hand, rel=1.0e-12)
    assert lateral_g(q, iv1.S_ref(2), 1.0, 0.0) == 0.0


# ======================================================================================
#   Smoothness, monotonicity, speed
# ======================================================================================


@pytest.mark.parametrize("stage", [1, 2])
def test_CD0_is_smooth_in_mach_to_mach_six(iv1, stage):
    """No CD0 step larger than 3 percent between adjacent Mach points at 0.005 spacing.

    Swept to M 6.0 even though the model is only validated to 5.0, because the trajectory
    integrator will go there and must not meet a cliff. The largest step for the stacked
    configuration sits at the interstage shoulder detachment Mach, near M 1.50, where the shock
    angle is changing fast. The value is continuous there; the step is steep slope, not a jump.
    """
    machs = np.arange(0.30, 6.0 + 1e-9, 0.005)
    cd0 = np.array([iv1.evaluate(float(m), H_CMP, 0.0, stage).CD0 for m in machs])
    assert np.all(cd0 > 0.0)
    rel = np.abs(np.diff(cd0)) / cd0[:-1]
    i = int(np.argmax(rel))
    assert rel[i] < TOL_CD0_SMOOTH, (
        f"stage {stage}: largest CD0 step {rel[i]:.3%} between M {machs[i]:.3f} and "
        f"{machs[i + 1]:.3f}"
    )


@pytest.mark.parametrize("stage", [1, 2])
def test_CN_and_xcp_are_smooth_in_mach(iv1, stage):
    machs = np.arange(0.30, 6.0 + 1e-9, 0.005)
    alpha = math.radians(10.0)
    cn = np.array([iv1.evaluate(float(m), H_CMP, alpha, stage).CN for m in machs])
    xcp = np.array([iv1.evaluate(float(m), H_CMP, alpha, stage).x_cp for m in machs])
    assert float(np.max(np.abs(np.diff(cn)) / cn[:-1])) < TOL_CN_SMOOTH
    assert float(np.max(np.abs(np.diff(xcp)) / xcp[:-1])) < TOL_XCP_SMOOTH


@pytest.mark.parametrize("stage", [1, 2])
def test_CN_is_monotone_and_antisymmetric_in_alpha(iv1, stage):
    alphas = np.linspace(0.0, ALPHA_MAX_VALID_IV1, 60)
    for mach in (0.5, 1.0, 2.0, 5.0):
        cn = np.array([iv1.evaluate(mach, H_CMP, float(a), stage).CN for a in alphas])
        assert np.all(np.diff(cn) > 0.0), f"CN not monotone in alpha at M {mach}"
    a = math.radians(12.0)
    assert iv1.evaluate(2.0, H_CMP, -a, stage).CN == pytest.approx(
        -iv1.evaluate(2.0, H_CMP, a, stage).CN, rel=1.0e-12
    )


@pytest.mark.parametrize("stage", [1, 2])
def test_ten_thousand_evaluate_calls_under_five_seconds(iv1, stage):
    atm.prime()
    machs = np.linspace(0.3, 5.0, 10_000)
    iv1.evaluate(2.0, H_CMP, 0.05, stage)
    t0 = time.perf_counter()
    for m in machs:
        iv1.evaluate(float(m), H_CMP, 0.05, stage)
    dt = time.perf_counter() - t0
    assert dt < 5.0, (
        f"stage {stage}: 10000 evaluate calls took {dt:.2f} s ({dt / 10_000 * 1e6:.1f} us each)"
    )


@pytest.mark.parametrize("stage", [1, 2])
def test_trim_alpha_round_trips(iv1, stage):
    for mach in (0.5, 1.1, 2.0, 3.5, 5.0):
        for target in (0.05, 0.4, 1.2, 2.5):
            a = iv1.trim_alpha(mach, H_CMP, target, stage)
            r = iv1.evaluate(mach, H_CMP, a, stage)
            if a >= iv1.reqs.alpha_max - 1.0e-9:
                assert r.CN <= target + 1.0e-9
            else:
                assert r.CN == pytest.approx(target, rel=1.0e-6)
    assert iv1.trim_alpha(2.0, H_CMP, 0.0, stage) == 0.0
    a_pos = iv1.trim_alpha(2.0, H_CMP, 0.5, stage)
    assert iv1.trim_alpha(2.0, H_CMP, -0.5, stage) == pytest.approx(-a_pos, rel=1.0e-9)
    assert iv1.trim_alpha(2.0, H_CMP, 1.0e6, stage) == pytest.approx(iv1.reqs.alpha_max)


# ======================================================================================
#   Internal consistency and bookkeeping
# ======================================================================================


@pytest.mark.parametrize("stage", [1, 2])
def test_breakdown_sums_to_CD0_and_CD(iv1, stage):
    r = iv1.evaluate(3.0, H_CMP, math.radians(8.0), stage)
    components = (
        "CD_friction_body", "CD_wave_body", "CD_interstage_shoulder", "CD_base",
        "CD_boattail", "CD_fin_friction", "CD_fin_wave", "CD_strake_friction",
        "CD_strake_wave", "CD_strake_base", "CD_strake_interference_NOT_MODELLED",
        "CD_protuberance_GUESS",
    )
    assert sum(r.breakdown[k] for k in components) == pytest.approx(r.CD0, rel=1.0e-12)
    assert r.CD == pytest.approx(
        r.CD0 * math.cos(r.alpha) + r.breakdown["CD_induced"], rel=1.0e-12
    )
    # Cross-checks must be reported and must never be inside CD0.
    for k in (
        "xcheck_CD_strake_le_blunt_bound",
        "xcheck_CD_shoulder_isentropic",
        "xcheck_CD_shoulder_stagnation",
    ):
        assert k in r.breakdown
        assert k not in components


@pytest.mark.parametrize("stage", [1, 2])
def test_normal_force_breakdown_sums_to_CN(iv1, stage):
    r = iv1.evaluate(3.0, H_CMP, math.radians(8.0), stage)
    total = (
        r.breakdown["CN_body_potential"]
        + r.breakdown["CN_body_crossflow"]
        + r.breakdown["CN_fins"]
        + r.breakdown["CN_strakes"]
    )
    assert total == pytest.approx(r.CN, rel=1.0e-12)
    assert r.breakdown["CN_strakes"] == pytest.approx(
        r.breakdown["CN_strakes_potential"] + r.breakdown["CN_strakes_vortex"],
        rel=1.0e-12,
    )
    fins = sum(v for k, v in r.breakdown.items() if k.startswith("CN_fins_stage"))
    assert fins == pytest.approx(r.breakdown["CN_fins"], rel=1.0e-12)


@pytest.mark.parametrize("stage", [1, 2])
def test_CM_is_referenced_to_the_configuration_diameter(iv1, stage):
    """CM = -CN x_cp / D_ref, with D_ref of THIS configuration. A wrong D_ref is a silent error."""
    r = iv1.evaluate(2.5, H_CMP, math.radians(6.0), stage)
    assert r.CM == pytest.approx(-r.CN * r.x_cp / iv1.D_ref(stage), rel=1.0e-12)
    assert r.breakdown["x_cp_over_D"] == pytest.approx(r.x_cp / iv1.D_ref(stage), rel=1.0e-12)
    assert r.breakdown["S_ref_m2"] == pytest.approx(iv1.S_ref(stage), rel=1.0e-15)
    assert r.breakdown["D_ref_m"] == pytest.approx(iv1.D_ref(stage), rel=1.0e-15)


@pytest.mark.parametrize("stage", [1, 2])
def test_x_cp_at_zero_alpha_is_the_limit_of_the_finite_alpha_value(iv1, stage):
    at_zero = iv1.evaluate(2.5, H_CMP, 0.0, stage).x_cp
    nearly = iv1.evaluate(2.5, H_CMP, 1.0e-7, stage).x_cp
    assert at_zero == pytest.approx(nearly, rel=1.0e-4)
    assert 0.0 < at_zero < iv1.geom[stage].L_total


def test_centre_of_pressure_lies_inside_the_vehicle(iv1):
    for stage in (1, 2):
        for mach in (0.3, 1.0, 2.5, 5.0):
            for alpha_deg in (0.0, 5.0, 15.0, 25.0):
                r = iv1.evaluate(mach, H_CMP, math.radians(alpha_deg), stage)
                assert 0.0 < r.x_cp < iv1.geom[stage].L_total


def test_power_on_relieves_base_drag(iv1):
    for stage in (1, 2):
        off = iv1.evaluate(3.0, H_CMP, 0.0, stage, power_on=False)
        on = iv1.evaluate(3.0, H_CMP, 0.0, stage, power_on=True)
        assert on.breakdown["CD_base"] < off.breakdown["CD_base"]
        assert on.CD0 < off.CD0


def test_validity_flag_marks_the_declared_envelope(iv1):
    assert iv1.evaluate(3.0, H_CMP, math.radians(20.0), 2).breakdown[
        "out_of_validity_range"
    ] == 0.0
    assert iv1.evaluate(6.0, H_CMP, 0.0, 2).breakdown["out_of_validity_range"] == 1.0
    assert iv1.evaluate(0.2, H_CMP, 0.0, 2).breakdown["out_of_validity_range"] == 1.0
    assert iv1.evaluate(3.0, H_CMP, math.radians(30.0), 2).breakdown[
        "out_of_validity_range"
    ] == 1.0
    # A11 is evaluated at 20 deg, so 20 deg has to be inside the envelope.
    assert InterceptRequirements().alpha_max <= ALPHA_MAX_VALID_IV1
    assert MACH_MAX_VALID == 5.0


def test_strake_fin_interference_is_declared_not_modelled_and_is_wired_up(reqs):
    """The factor must be 1.0 by default, say so in SOURCES, and actually change the answer."""
    from rocketgen.sizing.aero_iv1 import SOURCES as IV1_SOURCES

    text = IV1_SOURCES["iv1_aero_strake_fin_interference"]
    assert "NOT MODELLED" in text
    assert "OVERPREDICT" in text.upper()

    dv = default_iv1()
    base = StackAero(dv, reqs)
    assert base.k_strake_fin_interference == 1.0
    assert "NOT MODELLED" in base.sources_used[1]["k_strake_fin_interference"]

    reduced = StackAero(dv, reqs, k_strake_fin_interference=0.85)
    a = base.evaluate(3.0, H_CMP, math.radians(10.0), 2)
    b = reduced.evaluate(3.0, H_CMP, math.radians(10.0), 2)
    assert b.breakdown["CN_fins"] == pytest.approx(
        0.85 * a.breakdown["CN_fins"], rel=1.0e-12
    )
    # Losing tail load must move the centre of pressure FORWARD, the sign SOURCES claims.
    assert b.x_cp < a.x_cp


def test_strake_junction_interference_is_zero_and_says_why(iv1):
    from rocketgen.sizing.aero_iv1 import SOURCES as IV1_SOURCES

    r = iv1.evaluate(3.0, H_CMP, 0.0, 2)
    assert r.breakdown["CD_strake_interference_NOT_MODELLED"] == 0.0
    assert "cd0_calibration" in IV1_SOURCES["iv1_aero_strake_junction_interference"]
    assert "double count" in IV1_SOURCES["iv1_aero_strake_junction_interference"]


def test_strake_wave_drag_is_bracketed_and_the_bracket_is_wide(iv1):
    """The double-wedge section is the optimistic end. The blunt-edge bound is the other end.

    The bracket must be reported, and it must be wide, because that width IS the finding: at a
    strake t/c of 0.0057 the answer depends on the leading-edge bevel length, which the design
    vector does not carry. Reporting the optimistic value plus the bound is honest; inventing a
    bevel length to land in the middle would not be.
    """
    for mach in (1.5, 2.0, 3.0, 5.0):
        r = iv1.evaluate(mach, H_CMP, 0.0, 2)
        wedge = r.breakdown["CD_strake_wave"]
        blunt = r.breakdown["xcheck_CD_strake_le_blunt_bound"]
        assert wedge > 0.0 and blunt > 0.0
        assert blunt > 20.0 * wedge, (
            f"the strake wave-drag bracket has collapsed at M {mach}: "
            f"wedge {wedge:.3e}, blunt bound {blunt:.3e}"
        )
        # The bound must never be summed into CD0.
        assert blunt > r.CD0 * 0.0
    r = iv1.evaluate(3.0, H_CMP, 0.0, 2)
    total = (
        r.breakdown["CD_strake_friction"]
        + r.breakdown["CD_strake_wave"]
        + r.breakdown["CD_strake_base"]
    )
    assert r.CD0 > total, "the strake terms cannot be the whole of CD0"


# ======================================================================================
#   nTop measurement override, per stage
# ======================================================================================


def test_per_stage_ntop_measurements_override_the_analytic_areas(reqs):
    """A 20 percent larger measured wetted area on both stages must raise friction by 20 pc."""
    dv = default_iv1()
    analytic = StackAero(dv, reqs)
    meas: dict[int, NtopMeasurements] = {}
    for s in dv.stages:
        # Measure each stage on its own, so the per-stage analytic value is available to scale.
        one = StackAero(
            StackDesignVector(
                stages=[copy.deepcopy(s)], strakes=dv.strakes, f_nose=dv.f_nose,
                L_interstage=0.0,
            ),
            reqs,
        )
        alone = one.geom[s.index]
        meas[s.index] = NtopMeasurements(
            volume_total=0.3,
            volume_cavity=0.2,
            mass_structure=200.0,
            area_wetted_body=1.20 * alone.area_wetted_body,
            area_wetted_fins=1.20 * alone.fin_sets[0].area_wetted,
        )
    measured = StackAero(dv, reqs, meas=meas)
    for stage in (1, 2):
        a0 = analytic.evaluate(3.0, H_CMP, 0.0, stage)
        a1 = measured.evaluate(3.0, H_CMP, 0.0, stage)
        assert a1.breakdown["CD_fin_friction"] == pytest.approx(
            1.20 * a0.breakdown["CD_fin_friction"], rel=1.0e-9
        )
        assert a1.breakdown["CD_friction_body"] > a0.breakdown["CD_friction_body"]
        assert a1.CD0 > a0.CD0
        assert measured.sources_used[stage]["area_wetted_body"].startswith("nTop measured")
        assert analytic.sources_used[stage]["area_wetted_body"].startswith("analytic")
        assert a1.breakdown["n_quantities_from_ntop"] >= 2.0
        assert a0.breakdown["n_quantities_from_ntop"] == 0.0


def test_a_partially_measured_stack_falls_back_and_records_it(reqs):
    """One stage measured is not enough: mixing measured and analytic bases must be declared."""
    dv = default_iv1()
    analytic = StackAero(dv, reqs)
    meas = {2: NtopMeasurements(area_wetted_body=99.0)}
    partial = StackAero(dv, reqs, meas=meas)
    note = partial.sources_used[1]["area_wetted_body"]
    assert note.startswith("analytic")
    assert "1 of 2 stages measured" in note
    assert partial.geom[1].area_wetted_body == pytest.approx(
        analytic.geom[1].area_wetted_body, rel=1.0e-15
    )
    # But the stage-2 configuration IS fully measured, so it must use the measurement.
    assert partial.sources_used[2]["area_wetted_body"].startswith("nTop measured")
    assert partial.geom[2].area_wetted_body == pytest.approx(99.0, rel=1.0e-15)


def test_measured_base_area_changes_base_drag_and_declares_the_missing_boattail(reqs):
    dv = default_iv1()
    analytic = StackAero(dv, reqs)
    meas = {1: NtopMeasurements(area_base=0.5 * dv.booster.S_ref)}
    measured = StackAero(dv, reqs, meas=meas)
    assert measured.CD_base(3.0, 1) < analytic.CD_base(3.0, 1)
    assert measured.sources_used[1]["area_base"].startswith("nTop measured")
    assert "boattail" in measured.sources_used[1]["boattail"]
    assert "optimistic" in measured.sources_used[1]["boattail"]


def test_measured_area_distribution_is_spliced_across_stages(reqs):
    """A per-stage S(x) must be offset into the stack frame, with an analytic interstage."""
    dv = default_iv1()
    analytic = StackAero(dv, reqs)
    meas: dict[int, NtopMeasurements] = {}
    for i, s in enumerate(reversed(dv.stages)):
        n = 120
        dist = []
        for k in range(n + 1):
            x = s.L * k / n
            if i == 0 and x <= dv.L_nose:
                # tangent ogive, the same outer mould line the analytic path builds
                R = 0.5 * s.D
                rho = (R * R + dv.L_nose ** 2) / (2.0 * R)
                r = max(math.sqrt(max(rho * rho - (dv.L_nose - x) ** 2, 0.0)) - (rho - R), 0.0)
            else:
                r = 0.5 * s.D
            dist.append((x, math.pi * r * r))
        meas[s.index] = NtopMeasurements(area_distribution=dist)
    measured = StackAero(dv, reqs, meas=meas)
    for stage in (1, 2):
        assert measured.sources_used[stage]["area_distribution"].startswith("nTop measured")
        assert measured.geom[stage].area_planform_body == pytest.approx(
            analytic.geom[stage].area_planform_body, rel=0.02
        )
        assert measured.geom[stage].x_cp_crossflow == pytest.approx(
            analytic.geom[stage].x_cp_crossflow, rel=0.02
        )


# ======================================================================================
#   CLAUDE.md hard rule 3.1: no invented numbers
# ======================================================================================


def test_every_guess_is_labelled_as_a_guess():
    """No new guesses were introduced by this module, and the inherited ones stay visible."""
    from rocketgen.sizing.aero_iv1 import SOURCES as IV1_SOURCES

    guess_keys = [k for k in IV1_SOURCES if "GUESS" in k.upper()]
    assert guess_keys == [], f"aero_iv1 introduced a guessed constant: {guess_keys}"
    for key, text in IV1_SOURCES.items():
        if "GUESS" in text.upper():
            assert "GUESS" in key.upper() or "NOT A GUESS" in text.upper(), (
                f"{key} mentions a guess in its text but not in its key name"
            )
    # The two guesses inherited from aero.py must still be reported through the breakdown.
    r = StackAero(default_iv1(), InterceptRequirements()).evaluate(3.0, H_CMP, 0.0, 1)
    assert "CD_protuberance_GUESS" in r.breakdown
    assert r.breakdown["n_guessed_quantities"] >= 1.0


def test_sources_are_registered_globally():
    from rocketgen.sizing.aero_iv1 import SOURCES as IV1_SOURCES

    for key, value in IV1_SOURCES.items():
        assert SOURCES.get(key) == value, f"{key} not registered in config.SOURCES"
    # The new shared kernels in aero.py must also carry their sources.
    assert "aero_oblique_shock" in SOURCES
    assert "aero_isentropic_compression_turn" in SOURCES


def test_every_new_method_declares_what_it_omits():
    """Each new source string must state its approximation or what it leaves out."""
    from rocketgen.sizing.aero_iv1 import SOURCES as IV1_SOURCES

    must_declare = (
        "iv1_aero_strake_polhamus",
        "iv1_aero_strake_Kv",
        "iv1_aero_strake_upwash",
        "iv1_aero_strake_cp",
        "iv1_aero_strake_drag",
        "iv1_aero_strake_fin_interference",
        "iv1_aero_strake_body_interference_on_body_load",
        "iv1_aero_strake_junction_interference",
        "iv1_aero_interstage_shoulder",
        "iv1_aero_no_boattail",
    )
    for key in must_declare:
        text = IV1_SOURCES[key].upper()
        assert any(
            word in text
            for word in ("OMIT", "NOT MODELLED", "APPROXIMATION", "EXTENSION", "BOUND")
        ), f"{key} does not declare what it omits"


def test_provenance_is_recorded_per_configuration(iv1):
    for stage in (1, 2):
        prov = iv1.sources_used[stage]
        assert "S_ref" in prov and "strakes" in prov
        assert "area_wetted_body" in prov and "x_cp_potential" in prov
        assert "GUESS" in prov["area_nozzle_exit"]
    assert "booster" in iv1.sources_used[1]["S_ref"]
    assert "payload stage" in iv1.sources_used[2]["S_ref"]


# ======================================================================================
#   Pinned answers, so a regression is visible
# ======================================================================================


def test_iv1_design_point_is_physically_sane(iv1, reqs):
    """Pin the default-design-vector answers at a representative condition.

    M 3.0, 15 km, alpha 5 deg. These are NOT validated values; they are a regression fence, and
    the ranges are wide enough that only a real change in the model moves them.
    """
    for stage, cd0_lo, cd0_hi in ((1, 0.20, 0.50), (2, 0.15, 0.40)):
        r = iv1.evaluate(3.0, H_CMP, ALPHA_CMP, stage)
        assert cd0_lo < r.CD0 < cd0_hi, f"stage {stage} CD0 {r.CD0:.4f} out of the fence"
        assert 3.0 < r.CN_alpha < 15.0
        assert 0.3 < r.x_cp / iv1.geom[stage].L_total < 0.95
        assert r.CD > r.CD0 * math.cos(ALPHA_CMP)
    # The stacked configuration is longer, so its centre of pressure is further aft in metres.
    assert (
        iv1.evaluate(3.0, H_CMP, ALPHA_CMP, 1).x_cp
        > iv1.evaluate(3.0, H_CMP, ALPHA_CMP, 2).x_cp
    )
    # And A11 must be reachable at all: CN_max at the 20 deg limit is order unity or more.
    assert iv1.CN_max(3.0, H_CMP, 2, reqs.alpha_max) > 1.0
