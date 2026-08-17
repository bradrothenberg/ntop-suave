"""Tests for rocketgen.sizing.atmosphere and rocketgen.sizing.aero.

VALIDATION REFERENCE
--------------------
Published experimental data WAS found and is used. The reference case is the Army-Navy
"Basic Finner" reference projectile, measured in free flight:

    A. D. Dupuis and W. Hathaway, "Aeroballistic Range Tests of the Basic Finner Reference
    Projectile at Supersonic Velocities", Defence Research Establishment Valcartier,
    DREV-TM-9703, September 1997. DTIC accession AD-A636861.
    Full text: https://archive.org/details/DTIC_ADA636861

Model, from section 3.1 and Table I of that report:
    * 20 deg total-angle (10 deg half-angle) sharp cone nose on a cylindrical body
    * total length / diameter = 10.0, nominal diameter d = 30.00 mm
    * cone nose length follows from the half-angle: 0.5/tan(10 deg) = 2.836 calibres
    * four rectangular fins at the base, 1 cal x 1 cal, 0.08 cal thick at the fin root,
      leading-edge radius 0.004 cal
    * centre of gravity 16.50 cm from the nose = 5.500 calibres (Table I)
    * range conditions: 20 C at standard atmospheric pressure; Reynolds number on the
      projectile length from 7.0e6 at M 1.0 to 30e6 at M 4.5

The numbers in BASIC_FINNER_TABLE_VII below are transcribed from Table VII of that report
("Linear theory aerodynamic coefficients"), 23 shots. Column order in the report is
Shot Number, Mach Number, DBSQ, CD, CD0, CDSQ, CNa, Cma, Cmq, Cnpa, Clp(roll fit),
Clp(frequency fit). We keep Mach, DBSQ (the mean-squared total angle of attack, deg^2, which
the report uses to flag poorly determined shots), CD0, CNa and Cma. Moment coefficients are
per radian about the centre of gravity, referenced to pi d^2/4 and d.

TOLERANCES AND WHY
------------------
Comparisons are made at alpha = 2 deg because the reported CNa is a linear-theory reduction
over a finite angle-of-attack band, so the like-for-like model quantity is the secant slope
CN/alpha, not the derivative at alpha = 0.

    CD0        M >= 1.4   22 percent on Mach-band means, 25 percent on individual shots
    CN_alpha   M >= 1.4   22 percent on Mach-band means, 30 percent on individual shots
    x_cp/D     M >= 1.4    6 percent on Mach-band means, 15 percent on individual shots

Justification:
  * The brief and general practice put a conceptual component build-up within 10 to 20 percent
    on CD0. This model runs 15 percent LOW on the Basic Finner across M 1.4 to 4.5. The
    residual is understood, not random: the model has no fin trailing-edge base drag, no
    fin-body junction interference drag, and the real Basic Finner fin section ("conical, 0.08
    cal at the fin base, sharp leading edge") is not the symmetric double wedge the fin wave
    drag model assumes. Those omissions all push in the same direction.
  * CN_alpha runs 11 percent low on average, with the error growing to about 20 percent above
    M 3.5. That is the expected decay of first-order linearised supersonic fin theory, which
    falls off as 1/sqrt(M^2-1) while the measured slope flattens.
  * x_cp is the best-predicted quantity, within 4 percent on every Mach band, because it is a
    ratio of component loads and moment arms and is insensitive to the absolute level errors.
  * Individual-shot tolerances are looser than band means because the free-flight reduction
    itself scatters: two shots at nearly the same Mach (1.846 and 1.850) return CNa of 10.82
    and 16.26. The report states the scatter comes from low angle-of-attack shots, so shots
    with DBSQ < 0.5 deg^2 are excluded from the CNa and x_cp comparisons, per the report's own
    caveat. They are retained for CD0, which the report calls consistent.
  * M < 1.4 is EXCLUDED from the CD0 comparison and this is deliberate. The transonic drag rise
    in this model is an explicitly declared cubic blend, not physics, and does not reproduce the
    real drag-rise peak. It under-predicts by up to 36 percent at M 1.06. See
    `test_transonic_bridge_is_a_blend_and_underpredicts`, which asserts that the shortfall is
    there rather than pretending it is not.

Where no published data exists for a behaviour, the tests fall back to analytic limits,
monotonicity, smoothness and dimensional checks. No reference data is invented anywhere.
"""
from __future__ import annotations

import math
import statistics
import time

import numpy as np
import pytest

from rocketgen.config import DesignVector, NtopMeasurements, SOURCES
from rocketgen.sizing import atmosphere as atm
from rocketgen.sizing.aero import (
    ALPHA_MAX_VALID,
    RocketAero,
    cf_turbulent,
    cubic_blend,
)

# --------------------------------------------------------------------------------------
#   Reference data
# --------------------------------------------------------------------------------------

BASIC_FINNER_D = 0.030          # m, Table I
BASIC_FINNER_XCG_CAL = 5.500    # calibres from the nose, Table I (16.50 cm / 3.00 cm)
BASIC_FINNER_F_NOSE = 0.5 / math.tan(math.radians(10.0))   # = 2.836 cal, from the 20 deg cone

#: (Mach, DBSQ deg^2, CD0, CNa per rad, Cma per rad about the CG).
#: Dupuis and Hathaway, DREV-TM-9703, Table VII.
BASIC_FINNER_TABLE_VII: tuple[tuple[float, float, float, float, float], ...] = (
    (1.056, 0.4, 0.868, 16.391, -50.786),
    (1.057, 2.1, 0.866, 18.390, -52.022),
    (1.116, 0.1, 0.853, 22.337, -54.772),
    (1.254, 2.4, 0.756, 18.522, -51.982),
    (1.332, 0.7, 0.701, 11.422, -49.282),
    (1.380, 17.9, 0.625, 15.371, -42.073),
    (1.799, 0.5, 0.593, 12.039, -28.524),
    (1.846, 0.6, 0.596, 10.823, -27.076),
    (1.850, 0.9, 0.565, 16.263, -27.085),
    (2.348, 0.3, 0.498, 10.181, -18.764),
    (2.364, 0.4, 0.478, 7.532, -18.945),
    (2.413, 2.0, 0.471, 9.870, -18.096),
    (2.663, 3.4, 0.441, 8.804, -15.710),
    (2.741, 1.9, 0.398, 8.691, -14.888),
    (2.749, 0.1, 0.451, 1.957, -14.543),
    (2.970, 1.6, 0.376, 6.942, -13.489),
    (3.312, 2.8, 0.340, 8.300, -11.331),
    (3.337, 0.4, 0.375, 8.700, -11.036),
    (3.681, 8.0, 0.304, 8.170, -10.106),
    (3.741, 17.0, 0.297, 8.209, -10.256),
    (3.774, 4.7, 0.305, 7.694, -9.419),
    (4.127, 2.6, 0.283, 7.482, -8.243),
    (4.471, 16.3, 0.249, 7.701, -8.034),
)

MACH_BANDS = ((1.4, 2.0), (2.0, 2.6), (2.6, 3.2), (3.2, 4.0), (4.0, 4.6))

TOL_CD0_BAND = 0.22
TOL_CD0_SHOT = 0.25
TOL_CNA_BAND = 0.22
TOL_CNA_SHOT = 0.30
TOL_XCP_BAND = 0.06
TOL_XCP_SHOT = 0.15

DBSQ_MIN = 0.5      # report's own caveat: low angle-of-attack shots scatter
ALPHA_CMP = math.radians(2.0)


def basic_finner_dv() -> DesignVector:
    """The Basic Finner as a DesignVector. Geometry only; the bay lengths are irrelevant here."""
    d = BASIC_FINNER_D
    return DesignVector(
        D=d,
        L_total=10.0 * d,
        f_nose=BASIC_FINNER_F_NOSE,
        t_wall=0.0005,
        L_boattail=0.0,
        d_base=d,
        n_fin=4,
        b_fin=1.0 * d,
        c_r_fin=1.0 * d,
        taper_fin=1.0,
        sweep_fin=0.0,
        t_fin=0.08 * d,
        x_fin_te_gap=0.0,
        L_seeker=0.02,
        L_guidance=0.02,
        L_warhead=0.02,
    )


@pytest.fixture(scope="module")
def finner() -> RocketAero:
    return RocketAero(basic_finner_dv(), nose_shape="cone", fin_max_thickness_station=0.5)


@pytest.fixture(scope="module")
def sv1() -> RocketAero:
    return RocketAero(DesignVector())


def _xcp_exp_cal(cna: float, cma: float) -> float:
    """Experimental x_cp in calibres from the nose: x_cp = x_cg - Cma/CNa."""
    return BASIC_FINNER_XCG_CAL - cma / cna


# ======================================================================================
#   Atmosphere
# ======================================================================================


def test_atmosphere_table_matches_direct_suave_to_better_than_0p1_percent():
    """The cached table plus linear interpolation must not cost more than 0.1 percent."""
    atm.prime()
    from rocketgen.config import add_suave_to_path

    add_suave_to_path()
    from SUAVE.Analyses.Atmospheric import US_Standard_1976

    # Deliberately off-grid altitudes, and points straddling the 11 km and 20 km layer joins.
    h = np.array(
        [0.0, 3.7, 137.0, 999.9, 3333.3, 7777.7, 10_999.4, 11_000.6, 12_000.0,
         15_432.1, 19_999.2, 20_000.8, 23_456.7, 25_555.5, 29_999.9, 30_000.0]
    )
    ref = US_Standard_1976().compute_values(h)
    got = atm.atmo(h)

    for name in ("pressure", "temperature", "density", "speed_of_sound", "dynamic_viscosity"):
        r = np.ravel(np.asarray(ref[name], dtype=float))
        g = np.asarray(getattr(got, name), dtype=float)
        rel = np.max(np.abs(g - r) / np.abs(r))
        assert rel < 1.0e-3, f"{name} interpolation error {rel:.3e} exceeds 0.1 percent"


def test_atmosphere_scalar_and_vector_agree():
    st_v = atm.atmo(np.array([0.0, 12_000.0]))
    st_0 = atm.atmo(0.0)
    st_12 = atm.atmo(12_000.0)
    assert isinstance(st_0.pressure, float)
    assert st_v.pressure[0] == pytest.approx(st_0.pressure)
    assert st_v.pressure[1] == pytest.approx(st_12.pressure)


def test_atmosphere_sea_level_reference_values():
    """US Standard 1976 sea-level values are definitional, so check them."""
    st = atm.atmo(0.0)
    assert st.pressure == pytest.approx(101_325.0, rel=1e-4)
    assert st.temperature == pytest.approx(288.15, rel=1e-4)
    assert st.density == pytest.approx(1.225, rel=1e-3)
    assert st.speed_of_sound == pytest.approx(340.29, rel=1e-3)


def test_atmosphere_is_monotone_in_altitude():
    h = np.linspace(0.0, 30_000.0, 601)
    st = atm.atmo(h)
    assert np.all(np.diff(st.pressure) < 0.0)
    assert np.all(np.diff(st.density) < 0.0)


def test_q_dynamic_and_reynolds_per_metre_dimensional():
    h, V = 12_000.0, 590.0
    st = atm.atmo(h)
    assert atm.q_dynamic(h, V) == pytest.approx(0.5 * st.density * V * V)
    assert atm.reynolds_per_metre(h, V) == pytest.approx(st.density * V / st.dynamic_viscosity)
    # Sanity: q at 12 km and M 2 is tens of kPa, unit Re is of order 1e7 per metre.
    assert 30e3 < atm.q_dynamic(h, V) < 80e3
    assert 5e6 < atm.reynolds_per_metre(h, V) < 3e7


def test_velocity_helper():
    assert atm.velocity(2.0, 12_000.0) == pytest.approx(2.0 * atm.speed_of_sound(12_000.0))


def test_sutherland_viscosity_matches_table_at_sea_level():
    """Sutherland's law must reproduce the tabulated viscosity to a few percent."""
    st = atm.atmo(0.0)
    mu = float(atm.sutherland_viscosity(st.temperature))
    assert mu == pytest.approx(st.dynamic_viscosity, rel=0.02)


def test_atmosphere_is_fast_enough_for_the_integrator():
    atm.prime()
    n = 20_000
    t0 = time.perf_counter()
    for _ in range(n):
        atm.atmo(12_000.0)
    dt = time.perf_counter() - t0
    assert dt < 1.0, f"{n} scalar atmosphere calls took {dt:.2f} s"


def test_is_in_range():
    # The ceiling was raised from 30 km to 86 km so that a lofted two-stage intercept arc is
    # interpolated rather than clamped. This test used to assert 30_001.0 was out of range; that
    # was pinning the old ceiling, not a property worth keeping. It now checks the boundary
    # wherever the boundary actually is.
    assert atm.is_in_range(0.0)
    assert atm.is_in_range(30_000.0)
    assert atm.is_in_range(atm.H_MAX)
    assert not atm.is_in_range(-1.0)
    assert not atm.is_in_range(atm.H_MAX + 1.0)


# ======================================================================================
#   Validation against the Basic Finner free-flight data
# ======================================================================================


def test_validation_CD0_mach_band_means(finner):
    """CD0 versus Dupuis and Hathaway Table VII, averaged in Mach bands, M >= 1.4."""
    worst = 0.0
    report = []
    for lo, hi in MACH_BANDS:
        sel = [r for r in BASIC_FINNER_TABLE_VII if lo <= r[0] < hi]
        if not sel:
            continue
        exp = statistics.mean(r[2] for r in sel)
        mod = statistics.mean(finner.evaluate(r[0], 0.0, ALPHA_CMP).CD0 for r in sel)
        err = mod / exp - 1.0
        report.append(f"M {lo}-{hi}: exp {exp:.4f} model {mod:.4f} err {err:+.1%}")
        worst = max(worst, abs(err))
    assert worst <= TOL_CD0_BAND, "CD0 band means:\n  " + "\n  ".join(report)


def test_validation_CD0_individual_shots(finner):
    for mach, _dbsq, cd0_exp, _cna, _cma in BASIC_FINNER_TABLE_VII:
        if mach < 1.4:
            continue
        mod = finner.evaluate(mach, 0.0, ALPHA_CMP).CD0
        err = mod / cd0_exp - 1.0
        assert abs(err) <= TOL_CD0_SHOT, (
            f"M {mach}: CD0 exp {cd0_exp:.3f} model {mod:.3f} err {err:+.1%}"
        )


def test_validation_CD0_bias_is_the_documented_one(finner):
    """Pin the systematic bias so a future change cannot silently move it."""
    errs = [
        finner.evaluate(m, 0.0, ALPHA_CMP).CD0 / cd0 - 1.0
        for m, _d, cd0, _c, _cm in BASIC_FINNER_TABLE_VII
        if m >= 1.4
    ]
    mean_bias = statistics.mean(errs)
    assert -0.20 < mean_bias < -0.08, f"documented CD0 bias moved: {mean_bias:+.1%}"


def test_validation_CN_alpha_mach_band_means(finner):
    worst = 0.0
    report = []
    for lo, hi in MACH_BANDS:
        sel = [r for r in BASIC_FINNER_TABLE_VII if lo <= r[0] < hi and r[1] >= DBSQ_MIN]
        if not sel:
            continue
        exp = statistics.mean(r[3] for r in sel)
        mod = statistics.mean(finner.evaluate(r[0], 0.0, ALPHA_CMP).CN_alpha for r in sel)
        err = mod / exp - 1.0
        report.append(f"M {lo}-{hi}: exp {exp:.3f} model {mod:.3f} err {err:+.1%}")
        worst = max(worst, abs(err))
    assert worst <= TOL_CNA_BAND, "CN_alpha band means:\n  " + "\n  ".join(report)


def test_validation_CN_alpha_individual_shots(finner):
    for mach, dbsq, _cd0, cna_exp, _cma in BASIC_FINNER_TABLE_VII:
        if mach < 1.4 or dbsq < DBSQ_MIN:
            continue
        mod = finner.evaluate(mach, 0.0, ALPHA_CMP).CN_alpha
        err = mod / cna_exp - 1.0
        assert abs(err) <= TOL_CNA_SHOT, (
            f"M {mach}: CN_alpha exp {cna_exp:.2f} model {mod:.2f} err {err:+.1%}"
        )


def test_validation_x_cp_mach_band_means(finner):
    worst = 0.0
    report = []
    for lo, hi in MACH_BANDS:
        sel = [r for r in BASIC_FINNER_TABLE_VII if lo <= r[0] < hi and r[1] >= DBSQ_MIN]
        if not sel:
            continue
        exp = statistics.mean(_xcp_exp_cal(r[3], r[4]) for r in sel)
        mod = statistics.mean(
            finner.evaluate(r[0], 0.0, ALPHA_CMP).x_cp / BASIC_FINNER_D for r in sel
        )
        err = mod / exp - 1.0
        report.append(f"M {lo}-{hi}: exp {exp:.3f} model {mod:.3f} cal err {err:+.1%}")
        worst = max(worst, abs(err))
    assert worst <= TOL_XCP_BAND, "x_cp/D band means:\n  " + "\n  ".join(report)


def test_validation_x_cp_individual_shots(finner):
    for mach, dbsq, _cd0, cna, cma in BASIC_FINNER_TABLE_VII:
        if mach < 1.4 or dbsq < DBSQ_MIN:
            continue
        exp = _xcp_exp_cal(cna, cma)
        mod = finner.evaluate(mach, 0.0, ALPHA_CMP).x_cp / BASIC_FINNER_D
        err = mod / exp - 1.0
        assert abs(err) <= TOL_XCP_SHOT, (
            f"M {mach}: x_cp/D exp {exp:.2f} model {mod:.2f} err {err:+.1%}"
        )


def test_validation_CM_sign_and_static_stability(finner):
    """The Basic Finner is statically stable about its 5.5 cal CG at every tested Mach."""
    for mach, dbsq, _cd0, _cna, _cma in BASIC_FINNER_TABLE_VII:
        if dbsq < DBSQ_MIN:
            continue
        r = finner.evaluate(mach, 0.0, ALPHA_CMP)
        assert r.CM < 0.0, "moment about the nose tip must be nose-down for positive alpha"
        assert r.x_cp / BASIC_FINNER_D > BASIC_FINNER_XCG_CAL + 1.0, (
            f"M {mach}: static margin under one calibre"
        )


def test_transonic_bridge_is_a_blend_and_underpredicts(finner):
    """Document, by assertion, that the transonic bridge is not physics.

    The model bridges the subsonic value to the M 1.2 supersonic correlation with a cubic
    Hermite blend. The real Basic Finner drag rise peaks above the M 1.2 value, so the model
    must fall short around M 1.05 to 1.15. This test exists so nobody mistakes the blend for a
    drag-rise model.
    """
    shortfalls = [
        1.0 - finner.evaluate(m, 0.0, ALPHA_CMP).CD0 / cd0
        for m, _d, cd0, _c, _cm in BASIC_FINNER_TABLE_VII
        if m < 1.2
    ]
    assert min(shortfalls) > 0.15, "the blend unexpectedly matches the transonic peak"
    assert max(shortfalls) < 0.50, "the blend is further off than documented"


# ======================================================================================
#   Analytic limits and named-method behaviour
# ======================================================================================


def test_bare_body_normal_force_slope_is_the_slender_body_value():
    """A finless cylinder must give CN_alpha -> 2 per radian as alpha -> 0.

    Slender-body theory: N_alpha = 2 q S_base. With no boattail, S_base = S_ref, so the
    coefficient slope is exactly 2 per radian.
    """
    dv = basic_finner_dv().replace(n_fin=0)
    m = RocketAero(dv, nose_shape="cone")
    for mach in (0.5, 0.9, 2.0, 5.0):
        r = m.evaluate(mach, 5_000.0, 0.0)
        assert r.CN_alpha == pytest.approx(2.0, rel=1e-3), f"M {mach}"
        assert r.breakdown["CN_fins"] == 0.0


def test_bare_body_potential_cp_is_the_barrowman_cone_value():
    dv = basic_finner_dv().replace(n_fin=0)
    m = RocketAero(dv, nose_shape="cone")
    assert m.geom.x_cp_potential == pytest.approx((2.0 / 3.0) * dv.L_nose, rel=1e-12)
    m_og = RocketAero(dv, nose_shape="tangent_ogive")
    assert m_og.geom.x_cp_potential == pytest.approx(0.466 * dv.L_nose, rel=1e-12)


def test_crossflow_term_is_quadratic_in_alpha(sv1):
    """The Allen and Perkins term must scale as sin^2(alpha)."""
    a1, a2 = math.radians(3.0), math.radians(6.0)
    c1 = sv1.CN_body(2.0, a1)[1]
    c2 = sv1.CN_body(2.0, a2)[1]
    assert c2 / c1 == pytest.approx((math.sin(a2) / math.sin(a1)) ** 2, rel=1e-12)


def test_fin_normal_force_follows_prandtl_glauert_at_high_mach(sv1):
    """Deep supersonic, CN_fin * sqrt(M^2-1) must be nearly Mach independent at small alpha."""
    a = math.radians(0.5)
    vals = []
    for mach in (3.0, 3.5, 4.0, 4.5, 5.0):
        beta = math.sqrt(mach * mach - 1.0)
        vals.append(sv1.CN_fins(mach, a) * beta)
    spread = (max(vals) - min(vals)) / statistics.mean(vals)
    assert spread < 0.02, f"Prandtl-Glauert scaling broken, spread {spread:.3f}"


def test_fin_branch_switch_is_continuous_and_matches_at_the_branch_mach(sv1):
    """The slender-wing and linear-supersonic branches are equal at M_c by construction."""
    ar = sv1.geom.aspect_ratio_pair
    m_c = math.sqrt(1.0 + (8.0 / (math.pi * ar)) ** 2)
    assert 4.0 / math.sqrt(m_c * m_c - 1.0) == pytest.approx(math.pi * ar / 2.0, rel=1e-12)


def test_newtonian_limit_at_high_alpha(sv1):
    """At the alpha limit the Newtonian 2 sin^2 term must dominate the fin normal force."""
    a = ALPHA_MAX_VALID
    lin = 4.0 * abs(math.sin(a) * math.cos(a)) / math.sqrt(5.0 ** 2 - 1.0)
    newt = 2.0 * math.sin(a) ** 2
    assert newt > 0.4 * lin, "Newtonian term should be a substantial share at M 5, alpha 15 deg"
    # And the total normal force must stay finite and positive.
    r = sv1.evaluate(5.0, 20_000.0, a)
    assert 0.0 < r.CN < 10.0


def test_skin_friction_reduces_with_mach_and_reynolds(sv1):
    cf_lo_m = cf_turbulent(1.0e7, 0.3, 288.15)
    cf_hi_m = cf_turbulent(1.0e7, 4.0, 288.15)
    assert cf_hi_m < cf_lo_m, "compressibility must reduce skin friction"
    assert cf_turbulent(1.0e8, 2.0, 288.15) < cf_turbulent(1.0e6, 2.0, 288.15)


def test_sommer_short_reference_temperature_matches_the_published_formula():
    """Verify the T-prime transformation itself against NACA TN 3391 in closed form.

    T'/T = 1 + 0.035 M^2 + 0.45 (Tw/T - 1) with the adiabatic wall Tw/T = 1 + 0.89*0.2*M^2.
    Recovering T'/T from the returned Cf is possible because the only place the reference
    temperature enters as a multiplicative factor is rho'/rho_inf = T_inf/T'.
    """
    T = 288.15
    for mach in (0.0, 1.0, 2.0, 4.0, 5.0):
        tw_ratio = 1.0 + 0.89 * 0.2 * mach * mach
        tp_expected = 1.0 + 0.035 * mach * mach + 0.45 * (tw_ratio - 1.0)
        # Reconstruct Cf independently from the documented recipe and compare.
        rho_ratio = 1.0 / tp_expected
        mu_ratio = float(atm.sutherland_viscosity(T * tp_expected)) / float(
            atm.sutherland_viscosity(T)
        )
        re_ref = 1.0e7 * rho_ratio / mu_ratio
        cf_expected = 0.455 / (math.log10(re_ref) ** 2.58) * rho_ratio
        assert cf_turbulent(1.0e7, mach, T) == pytest.approx(cf_expected, rel=1e-12)
    # At M 2 the published reference temperature ratio is 1.4604.
    assert 1.0 + 0.035 * 4.0 + 0.45 * (1.0 + 0.178 * 4.0 - 1.0) == pytest.approx(1.4604, abs=1e-4)


def test_skin_friction_cross_checks_against_suave():
    """Independent order-of-magnitude cross-check against SUAVE's own flat-plate routine.

    SUAVE's `compressible_turbulent_flat_plate` starts from the same Sommer and Short reference
    temperature but applies a different Reynolds correction (Stanford AA241 course notes) that
    drives the friction down much harder with Mach. The two agree closely subsonically and
    diverge to a factor of about 2.1 by M 4. This test only asserts that both are the same order
    of magnitude and both fall with Mach; it is not a validation of either. The T-prime model
    here is the one this module uses, and it is verified in closed form by
    `test_sommer_short_reference_temperature_matches_the_published_formula`.
    """
    from rocketgen.config import add_suave_to_path

    add_suave_to_path()
    from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions import (
        compressible_turbulent_flat_plate,
    )

    prev_ours = prev_theirs = None
    for mach in (0.3, 1.0, 2.0, 3.0, 4.0):
        ours = cf_turbulent(1.0e7, mach, 288.15)
        theirs = float(np.ravel(compressible_turbulent_flat_plate(1.0e7, mach, 288.15)[0])[0])
        assert 0.9 < ours / theirs < 2.5, f"M {mach}: ours {ours:.5f} SUAVE {theirs:.5f}"
        if prev_ours is not None:
            assert ours < prev_ours and theirs < prev_theirs, f"M {mach}: not falling with Mach"
        prev_ours, prev_theirs = ours, theirs


def test_cubic_blend_is_c1_and_bounded():
    assert cubic_blend(0.5, 1.0, 2.0) == 1.0
    assert cubic_blend(2.5, 1.0, 2.0) == 0.0
    assert cubic_blend(1.5, 1.0, 2.0) == pytest.approx(0.5)
    # zero slope at both ends
    e = 1.0e-6
    assert abs(cubic_blend(1.0 + e, 1.0, 2.0) - 1.0) < 1.0e-11
    assert abs(cubic_blend(2.0 - e, 1.0, 2.0)) < 1.0e-11


def test_boattail_drag_present_only_with_a_boattail_and_only_supersonically():
    with_bt = RocketAero(DesignVector())
    no_bt = RocketAero(DesignVector(L_boattail=0.0, d_base=DesignVector().D))
    assert with_bt.CD_boattail(2.0) > 0.0
    assert no_bt.CD_boattail(2.0) == 0.0
    assert with_bt.CD_boattail(0.8) == 0.0
    # Prandtl-Meyer expansion weakens with Mach, so the boattail drag must fall.
    assert with_bt.CD_boattail(4.0) < with_bt.CD_boattail(2.0)


def test_base_drag_branches_are_continuous_at_mach_one(sv1):
    """Both published branches give 0.25 at M = 1, and the blend must not break that."""
    lo = sv1.CD_base(0.999)
    hi = sv1.CD_base(1.001)
    assert lo == pytest.approx(hi, rel=0.01)


def test_power_on_reduces_base_drag(sv1):
    off = sv1.evaluate(2.0, 12_000.0, 0.0, power_on=False)
    on = sv1.evaluate(2.0, 12_000.0, 0.0, power_on=True)
    assert on.breakdown["CD_base"] < off.breakdown["CD_base"]
    assert on.CD0 < off.CD0


def test_breakdown_sums_to_CD(sv1):
    """Every drag entry must account for the total. No hidden terms."""
    r = sv1.evaluate(2.0, 12_000.0, math.radians(6.0))
    keys_zero_lift = (
        "CD_friction_body", "CD_wave_body", "CD_base", "CD_boattail",
        "CD_fin_friction", "CD_fin_wave", "CD_protuberance_GUESS",
    )
    assert sum(r.breakdown[k] for k in keys_zero_lift) == pytest.approx(r.CD0, rel=1e-12)
    total = r.CD0 + r.breakdown["CD_induced"] + r.breakdown["CD_axial_incidence_projection"]
    assert total == pytest.approx(r.CD, rel=1e-12)
    assert (
        r.breakdown["CN_body_potential"]
        + r.breakdown["CN_body_crossflow"]
        + r.breakdown["CN_fins"]
    ) == pytest.approx(r.CN, rel=1e-12)


def test_cm_and_xcp_are_consistent(sv1):
    r = sv1.evaluate(2.0, 12_000.0, math.radians(4.0))
    assert r.CM == pytest.approx(-r.CN * r.x_cp / sv1.geom.D, rel=1e-12)


# ======================================================================================
#   Smoothness, monotonicity, speed
# ======================================================================================


def test_CD0_is_smooth_in_mach(sv1):
    """No jump greater than 3 percent between adjacent Mach points at 0.005 spacing.

    This is the test that catches a discontinuous transonic bridge, which would make the
    trajectory integrator step-size control thrash and would break a gradient optimiser.
    """
    machs = np.arange(0.30, 5.0 + 1e-9, 0.005)
    cd0 = np.array([sv1.evaluate(float(m), 12_000.0, 0.0).CD0 for m in machs])
    rel_jump = np.abs(np.diff(cd0)) / cd0[:-1]
    i = int(np.argmax(rel_jump))
    assert rel_jump[i] < 0.03, (
        f"largest CD0 jump {rel_jump[i]:.3%} between M {machs[i]:.3f} and {machs[i+1]:.3f}"
    )


def test_CN_is_smooth_in_mach(sv1):
    machs = np.arange(0.30, 5.0 + 1e-9, 0.005)
    cn = np.array([sv1.evaluate(float(m), 12_000.0, math.radians(5.0)).CN for m in machs])
    rel_jump = np.abs(np.diff(cn)) / cn[:-1]
    i = int(np.argmax(rel_jump))
    assert rel_jump[i] < 0.03, (
        f"largest CN jump {rel_jump[i]:.3%} between M {machs[i]:.3f} and {machs[i+1]:.3f}"
    )


def test_xcp_is_smooth_in_mach(sv1):
    machs = np.arange(0.30, 5.0 + 1e-9, 0.005)
    x = np.array([sv1.evaluate(float(m), 12_000.0, math.radians(5.0)).x_cp for m in machs])
    rel_jump = np.abs(np.diff(x)) / x[:-1]
    assert float(np.max(rel_jump)) < 0.01


def test_CD0_decreases_monotonically_above_mach_1p5(sv1):
    machs = np.arange(1.5, 5.0, 0.01)
    cd0 = np.array([sv1.evaluate(float(m), 12_000.0, 0.0).CD0 for m in machs])
    assert np.all(np.diff(cd0) < 0.0), "supersonic CD0 must fall with Mach"


def test_CN_increases_monotonically_with_alpha(sv1):
    alphas = np.linspace(0.0, ALPHA_MAX_VALID, 60)
    for mach in (0.5, 1.0, 2.0, 5.0):
        cn = np.array([sv1.evaluate(mach, 12_000.0, float(a)).CN for a in alphas])
        assert np.all(np.diff(cn) > 0.0), f"CN not monotone in alpha at M {mach}"


def test_CN_is_antisymmetric_in_alpha(sv1):
    a = math.radians(7.0)
    assert sv1.evaluate(2.0, 12_000.0, -a).CN == pytest.approx(
        -sv1.evaluate(2.0, 12_000.0, a).CN, rel=1e-12
    )


def test_ten_thousand_evaluate_calls_under_five_seconds(sv1):
    atm.prime()
    machs = np.linspace(0.3, 5.0, 10_000)
    t0 = time.perf_counter()
    for m in machs:
        sv1.evaluate(float(m), 12_000.0, 0.035)
    dt = time.perf_counter() - t0
    assert dt < 5.0, f"10000 evaluate calls took {dt:.2f} s ({dt/10_000*1e6:.1f} us each)"


def test_single_evaluate_under_one_millisecond(sv1):
    atm.prime()
    sv1.evaluate(2.0, 12_000.0, 0.035)
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        sv1.evaluate(2.0, 12_000.0, 0.035)
    per_call = (time.perf_counter() - t0) / n
    assert per_call < 1.0e-3, f"{per_call*1e6:.1f} us per evaluate"


# ======================================================================================
#   trim_alpha
# ======================================================================================


@pytest.mark.parametrize("mach", [0.4, 0.9, 1.1, 2.0, 3.5, 5.0])
@pytest.mark.parametrize("cn_target", [0.05, 0.3, 1.0, 2.0])
def test_trim_alpha_round_trips(sv1, mach, cn_target):
    a = sv1.trim_alpha(mach, 12_000.0, cn_target)
    r = sv1.evaluate(mach, 12_000.0, a)
    if a >= ALPHA_MAX_VALID - 1e-9:
        # Saturated: the configuration cannot make the requested CN inside the alpha limit.
        assert r.CN <= cn_target + 1e-9
    else:
        assert r.CN == pytest.approx(cn_target, rel=1e-6)


def test_trim_alpha_signs_and_zero(sv1):
    assert sv1.trim_alpha(2.0, 12_000.0, 0.0) == 0.0
    a_pos = sv1.trim_alpha(2.0, 12_000.0, 0.4)
    a_neg = sv1.trim_alpha(2.0, 12_000.0, -0.4)
    assert a_pos > 0.0 and a_neg == pytest.approx(-a_pos, rel=1e-9)


def test_trim_alpha_saturates_rather_than_raising(sv1):
    a = sv1.trim_alpha(2.0, 12_000.0, 1.0e6)
    assert a == pytest.approx(ALPHA_MAX_VALID)


# ======================================================================================
#   nTop measurement override
# ======================================================================================


def test_ntop_wetted_area_override_raises_friction_drag():
    """A 20 percent larger measured wetted area must raise friction drag by 20 percent."""
    dv = DesignVector()
    analytic = RocketAero(dv)
    swet = analytic.geom.area_wetted_body
    swet_fin = analytic.geom.area_wetted_fins

    meas = NtopMeasurements(
        volume_total=0.3,
        volume_cavity=0.2,
        mass_structure=200.0,
        area_wetted_body=1.20 * swet,
        area_wetted_fins=1.20 * swet_fin,
        area_base=dv.S_base,
    )
    measured = RocketAero(dv, meas)

    a0 = analytic.evaluate(2.0, 12_000.0, 0.0)
    a1 = measured.evaluate(2.0, 12_000.0, 0.0)

    assert a1.breakdown["CD_friction_body"] == pytest.approx(
        1.20 * a0.breakdown["CD_friction_body"], rel=1e-9
    )
    assert a1.breakdown["CD_fin_friction"] > a0.breakdown["CD_fin_friction"]
    assert a1.CD0 > a0.CD0

    assert measured.sources_used["area_wetted_body"].startswith("nTop measured")
    assert measured.sources_used["area_wetted_fins"].startswith("nTop measured")
    assert analytic.sources_used["area_wetted_body"].startswith("analytic")
    assert a1.breakdown["n_quantities_from_ntop"] >= 3.0
    assert a0.breakdown["n_quantities_from_ntop"] == 0.0


def test_ntop_area_distribution_drives_planform_and_cp():
    """A measured S(x) must replace both the planform area and the potential-lift cp."""
    dv = DesignVector()
    analytic = RocketAero(dv)

    # Synthesise the exact analytic outer mould line as a measured distribution: a tangent
    # ogive nose, a cylinder, then a conical boattail. If the measured path is wired up
    # correctly it must land close to the closed-form values, not identically on them.
    n = 200
    R, r_b = 0.5 * dv.D, 0.5 * dv.d_base
    rho = (R * R + dv.L_nose ** 2) / (2.0 * R)
    dist = []
    for i in range(n + 1):
        x = dv.L_total * i / n
        if x <= dv.L_nose:
            r = max(math.sqrt(max(rho * rho - (dv.L_nose - x) ** 2, 0.0)) - (rho - R), 0.0)
        elif x <= dv.L_nose + dv.L_body_cyl:
            r = R
        else:
            f = (x - dv.L_nose - dv.L_body_cyl) / dv.L_boattail
            r = R + f * (r_b - R)
        dist.append((x, math.pi * r * r))

    meas = NtopMeasurements(
        volume_total=0.3, volume_cavity=0.2, mass_structure=200.0,
        area_wetted_body=analytic.geom.area_wetted_body,
        area_distribution=dist,
    )
    measured = RocketAero(dv, meas)

    assert measured.sources_used["area_distribution"] == "nTop measured"
    assert measured.geom.area_planform_body == pytest.approx(
        analytic.geom.area_planform_body, rel=0.02
    )
    assert measured.geom.x_cp_crossflow == pytest.approx(analytic.geom.x_cp_crossflow, rel=0.02)
    # The measured dS/dx centroid is the exact slender-body result; the analytic fallback uses
    # Barrowman's tabulated ogive fraction, so they should agree to a few per cent.
    assert measured.geom.x_cp_potential == pytest.approx(
        analytic.geom.x_cp_potential, rel=0.15
    )


def test_ntop_base_area_override_changes_base_drag():
    dv = DesignVector()
    analytic = RocketAero(dv)
    meas = NtopMeasurements(
        volume_total=0.3, volume_cavity=0.2, mass_structure=200.0,
        area_wetted_body=analytic.geom.area_wetted_body,
        area_base=0.5 * dv.S_base,
    )
    measured = RocketAero(dv, meas)
    assert measured.CD_base(2.0) < analytic.CD_base(2.0)
    assert measured.sources_used["area_base"].startswith("nTop measured")


# ======================================================================================
#   Bookkeeping required by PLAN.md hard rule 2
# ======================================================================================


def test_all_guesses_are_labelled_as_guesses():
    from rocketgen.sizing.aero import SOURCES as AERO_SOURCES

    guess_keys = [k for k in AERO_SOURCES if "GUESS" in k]
    assert len(guess_keys) == 2, f"guess count changed: {guess_keys}"
    for k in guess_keys:
        assert "GUESS" in AERO_SOURCES[k].upper(), f"{k} does not say it is a guess"
    # And they must be visible in the breakdown key names.
    r = RocketAero(DesignVector()).evaluate(2.0, 12_000.0, 0.0)
    assert "CD_protuberance_GUESS" in r.breakdown


def test_sources_are_registered_globally():
    from rocketgen.sizing.aero import SOURCES as AERO_SOURCES
    from rocketgen.sizing.atmosphere import SOURCES as ATMO_SOURCES

    for k, v in {**AERO_SOURCES, **ATMO_SOURCES}.items():
        assert SOURCES.get(k) == v, f"{k} not registered in config.SOURCES"


def test_validity_flag_set_outside_the_stated_envelope(sv1):
    assert sv1.evaluate(2.0, 12_000.0, 0.0).breakdown["out_of_validity_range"] == 0.0
    assert sv1.evaluate(6.0, 12_000.0, 0.0).breakdown["out_of_validity_range"] == 1.0
    assert sv1.evaluate(2.0, 12_000.0, math.radians(20.0)).breakdown[
        "out_of_validity_range"
    ] == 1.0


def test_sv1_cruise_point_is_physically_sane(sv1):
    """SPEC.md R2: M 2.00 at 12 000 m. Pin the answer so regressions are visible.

    The default DesignVector fins are small relative to the body (exposed pair aspect ratio
    about 1.2), so CN_alpha is well below the Basic Finner's. That is a property of the
    starting design vector, not of the model; WP5 will move the fins.
    """
    r = sv1.evaluate(2.0, 12_000.0, math.radians(2.0))
    assert 0.30 < r.CD0 < 0.60
    assert 4.0 < r.CN_alpha < 20.0
    assert 0.4 < r.x_cp / sv1.geom.L_total < 0.95
    assert r.CD > r.CD0 * math.cos(math.radians(2.0))
    assert r.CM < 0.0
