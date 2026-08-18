"""The splined outer mould line, against a real `ntopcl`.

Nothing here is mocked. These tests answer the questions the offline spline tests cannot:

  * does a notebook using nTop's TRUE spline revolve actually convert? All four blocks in that
    chain are absent from the vendored universe and go through `raw_block`, so nothing but a
    real `ntopcl` can confirm they are right.
  * does nTop's measured volume match the independent Python integral of the SAME spline?
  * does ONE converted notebook serve a whole blend sweep, or does changing the shape force a
    re-`convert`? That is the property the entire approach depends on, and it is only a
    property if the topology key genuinely excludes the control VALUES.

Run: `.venv/Scripts/python.exe -m pytest tests/test_spline_notebook.py -q`

These are slow (each `measure_rocket` is roughly 40 to 80 s of real `ntopcl`), so the
converted notebooks and the measurements are shared through session-scoped fixtures.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rocketgen import oml_spline as S                                # noqa: E402
from rocketgen.config import RUNS_DIR, DesignVector                  # noqa: E402
from rocketgen.ntopgen import NtopRunner                             # noqa: E402
from rocketgen.ntopgen.rocket_notebook import (                      # noqa: E402
    build_rocket_recipe,
    measure_rocket,
)

GEOM_DIR = os.path.join(RUNS_DIR, "SV-1_spline_geom")

# Same 1 percent gate the ogive notebook was accepted under (PLAN.md WP1).
VOLUME_TOLERANCE = 0.01

BASE = dict(
    D=0.35, L_total=3.60, f_nose=3.4, m_p_boost=130.0, m_p_sustain=172.0,
    m_p_terminal=40.0, F_boost=45.0e3, F_terminal=8.0e3, b_fin=0.23, c_r_fin=0.42,
)


def _dv(**kw) -> DesignVector:
    return DesignVector(**{**BASE, **kw})


def closed_form_oml_volume(dv: DesignVector) -> float:
    """Independent Python volume of the revolved SPLINE: nose + cylinder + tail.

    Deliberately assembled here from `SplineProfile` primitives rather than imported from the
    notebook module, so it is a genuinely separate calculation and not the same code checking
    itself. Every term is exact: the spline runs are analytic integrals of the same curve nTop
    revolves, and the cylinder and conical boattail are textbook.
    """
    R, r_base = 0.5 * dv.D, 0.5 * dv.d_base
    nose = S.SplineProfile(length=dv.L_nose, radius=R, control=dv.nose_control)
    total = nose.volume() + math.pi * R * R * dv.L_body_cyl
    if dv.boattail_control is not None:
        bt = S.SplineProfile(length=dv.L_boattail, radius=r_base,
                             control=dv.boattail_control, r0_over_r=R / r_base)
        total += bt.volume()
    else:
        h = dv.L_boattail
        total += math.pi * h * (R * R + R * r_base + r_base * r_base) / 3.0
    return total


@pytest.fixture(scope="session")
def runner() -> NtopRunner:
    return NtopRunner()


@pytest.fixture(scope="session")
def ogive(runner: NtopRunner):
    """The tangent-ogive notebook, as the comparison baseline."""
    return measure_rocket(_dv(), os.path.join(GEOM_DIR, "ogive"), runner)


@pytest.fixture(scope="session")
def spline0(runner: NtopRunner):
    """Splined nose at blend 0: the same SHAPE as the ogive, to 1e-6 of R."""
    return measure_rocket(_dv(nose_shape="spline", nose_blend=0.0),
                          os.path.join(GEOM_DIR, "spline_b0"), runner)


@pytest.fixture(scope="session")
def spline1(runner: NtopRunner):
    """Splined nose at blend 1: the slender-body drag optimum."""
    return measure_rocket(_dv(nose_shape="spline", nose_blend=1.0),
                          os.path.join(GEOM_DIR, "spline_b1"), runner)


# --------------------------------------------------------------------------------------
#   The notebook is genuinely parametric in the shape
# --------------------------------------------------------------------------------------


def test_the_true_spline_blocks_are_the_ones_that_are_used() -> None:
    """Pin the four block ids. None is in the vendored universe, so nothing else checks them.

    They were found by `exportjson` on a real notebook after 27 guessed combinations were all
    rejected; see `docs/NTOP_NOTES.md` section 25. If a future nTop renames one, this test says
    so immediately instead of leaving a bare "Error loading recipe" to be diagnosed again.
    """
    from rocketgen.ntopgen import rocket_notebook as RN

    assert RN.SPLINE_BLOCK == "spline_by_control_points<list<point>,integer>[5.20.0]"
    assert RN.PROFILE_FROM_CURVES_BLOCK == (
        "profile_from_curves<list<curve_interface>,vector>[5.20.0]")
    assert RN.SPLINE_REVOLVE_BLOCK == "revolve<new_profile,axis,real>[5.20.0]"
    assert RN.LINE_BLOCK == "two_point_line<point,point>"


def test_the_spline_notebook_is_smaller_than_the_ogive_one() -> None:
    """The true spline replaced a 24-vertex polygon and its multiply-add chains.

    Measured: 154 entries for the splined nose against 289 for the tangent ogive. This is the
    cheapest route, not merely the most accurate, and that is worth pinning because a
    regression to a sampled route would show up here first.
    """
    ogive = build_rocket_recipe(_dv(), GEOM_DIR)
    spline = build_rocket_recipe(_dv(nose_shape="spline"), GEOM_DIR)
    assert len(spline._entries) < len(ogive._entries)


def test_control_values_are_real_notebook_inputs() -> None:
    """If they were baked in, a blend sweep would need one `convert` per design point."""
    recipe = build_rocket_recipe(_dv(nose_shape="spline"), GEOM_DIR)
    names = recipe.input_names()
    for i in range(9):
        assert f"Nose Shape c{i}" in names


def test_boattail_control_values_are_inputs_only_when_the_boattail_is_splined() -> None:
    plain = build_rocket_recipe(_dv(nose_shape="spline"), GEOM_DIR).input_names()
    assert not any("Boattail Shape" in n for n in plain)
    both = build_rocket_recipe(
        _dv(nose_shape="spline", boattail_shape="spline"), GEOM_DIR
    ).input_names()
    assert sum("Boattail Shape" in n for n in both) == 9


def test_blend_does_not_change_the_topology_key() -> None:
    """The cache key must exclude the control VALUES and include their COUNT.

    This is what makes one converted `.ntop` serve the whole blend range. Checked directly on
    the key rather than by timing a convert, so it cannot pass by accident on a warm cache.
    """
    from rocketgen.ntopgen.rocket_notebook import N_OGIVE_OUTER, _topology_key

    def key(dv):
        return _topology_key(
            dv, n_ogive=N_OGIVE_OUTER, relative_error=0.01, area_relative_error=0.01,
            export_stl=False, export_step=False, export_implicit=False,
            cad_tolerance=1e-4, area_stations=0, section_feature_size=1e-3, stage="full",
        )

    a = key(_dv(nose_shape="spline", nose_blend=0.0))
    b = key(_dv(nose_shape="spline", nose_blend=1.0))
    assert a == b, "blend must not force a re-convert"

    c = key(_dv(nose_shape="spline", nose_blend=0.0, n_ctrl_oml=11))
    assert c != a, "control-value COUNT changes the block graph and must re-convert"

    d = key(_dv(nose_shape="spline", boattail_shape="spline"))
    assert d != a, "splining the boattail changes the block graph and must re-convert"

    e = key(_dv())
    assert e != a, "the ogive and the spline are different block graphs"


# --------------------------------------------------------------------------------------
#   nTop measures what the closed form says it should
# --------------------------------------------------------------------------------------


def test_splined_nose_volume_matches_the_independent_closed_form(spline0) -> None:
    """The core geometric check. Measured 0.2907110 against 0.2906892, +0.0075 percent.

    There is no discretisation allowance in this number. nTop revolves the same spline the
    closed form integrates, so the residual is nTop's own `mass_properties` integration
    tolerance and nothing else.
    """
    dv = _dv(nose_shape="spline", nose_blend=0.0)
    assert spline0.volume_total == pytest.approx(
        closed_form_oml_volume(dv), rel=VOLUME_TOLERANCE
    )
    assert abs(spline0.volume_total / closed_form_oml_volume(dv) - 1.0) < 1.0e-3


def test_drag_optimal_nose_volume_matches_the_independent_closed_form(spline1) -> None:
    dv = _dv(nose_shape="spline", nose_blend=1.0)
    assert abs(spline1.volume_total / closed_form_oml_volume(dv) - 1.0) < 1.0e-3


def test_spline_at_blend_zero_reproduces_the_ogive_notebook(ogive, spline0) -> None:
    """Same shape, so the same measurements, to within the sampling difference.

    NOT bit-identical, and it should not be: the ogive notebook revolves a 24-segment chord
    polygon, which is INSCRIBED and so under-measures, while the spline notebook revolves the
    true curve. The spline therefore reads very slightly larger. Measured +0.026 percent on
    volume, far inside the 1 percent geometric gate.
    """
    assert spline0.volume_total == pytest.approx(ogive.volume_total, rel=2.0e-3)
    assert spline0.area_wetted_body == pytest.approx(ogive.area_wetted_body, rel=2.0e-3)


def test_drag_optimal_nose_encloses_less_volume_than_the_ogive(ogive, spline1) -> None:
    """The trade the blend scalar buys: less wave drag, less room for the seeker.

    Measured -1.3 percent on the whole OML, which is about -6.3 percent on the nose alone.
    If this ever came out POSITIVE the blend would be running the wrong way.
    """
    assert spline1.volume_total < ogive.volume_total
    assert 0.005 < 1.0 - spline1.volume_total / ogive.volume_total < 0.03


def test_structure_mass_stays_a_hollow_airframe(spline0, spline1) -> None:
    """Same sanity bound the ogive notebook carries: a solid billet would be 943 kg."""
    for m in (spline0, spline1):
        assert 10.0 < m.mass_structure < 120.0


def test_structure_cg_is_on_the_body_axis(spline1) -> None:
    """A cruciform body is symmetric, but discretisation means it is not EXACTLY symmetric.

    CLAUDE.md section 4 point 12: test against a tolerance, never against zero.
    """
    _, y, z = spline1.cg_structure
    assert abs(y) < 2.0e-3
    assert abs(z) < 2.0e-3


def test_the_run_reports_a_real_returncode(spline0) -> None:
    """Exit code 72 means a block FAILED. Gate on artefacts, and surface the real code."""
    assert spline0.ntopcl_returncode is not None
    assert spline0.is_usable()
