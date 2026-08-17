"""WP4 tests: the parametric SV-1 rocket notebook, against a real `ntopcl`.

Nothing here is mocked. The notebook is authored, converted once, and then run at two
different design vectors through the SAME `.ntop`, which is what proves the design variables
are real nTop inputs and that the notebook cache works.

Artefacts land in `runs/SV-1_geom/` so they can be inspected afterwards. pytest's tmpdir is
deliberately not used for them (the WP1 convention in `tests/test_ntopgen.py`).

Run: `.venv/Scripts/python.exe -m pytest tests/test_rocket_notebook.py -q`
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rocketgen.config import (                                       # noqa: E402
    MATERIALS,
    RUNS_DIR,
    DesignVector,
    NtopMeasurements,
    Requirements,
)
from rocketgen.ntopgen import NtopError, NtopRunner, Recipe          # noqa: E402
from rocketgen.ntopgen.driver import OUTPUT_NAME_MAP                 # noqa: E402
from rocketgen.ntopgen.rocket_notebook import (                     # noqa: E402
    DEFAULT_MESH_TOLERANCE,
    N_OGIVE_OUTER,
    NTOP_INPUTS,
    build_rocket_notebook,
    build_rocket_recipe,
    measure_rocket,
    tangent_ogive_rho,
    tangent_ogive_y,
)
from rocketgen.sizing.loop import GeometryFn, converge_point         # noqa: E402
from rocketgen.sizing.masses import analytic_geometry                # noqa: E402

GEOM_DIR = os.path.join(RUNS_DIR, "SV-1_geom")

# PLAN.md WP1 definition of done and the WP4 brief: the measured OML volume must match the
# independent closed form to better than 1 percent.
VOLUME_TOLERANCE = 0.01

# Wetted-area tolerance, stated and justified.
#
# The measured body area cannot equal the closed form exactly:
#   * the revolved OML is a 24-segment CHORD polygon, which under-measures the ogive lateral
#     area by 0.044 percent (measured, see SOURCES["ogive_polygon_sampling"]);
#   * `surface_area<implicit,real>` integrates over the implicit field, and nTop itself warns
#     that an implicit body's field has undefined regions (docs/NTOP_NOTES.md section 11).
#     Its relative-error input was swept from 0.002 to 0.2 and changed the answer by nothing
#     at all, so there is no knob to tighten;
#   * the flat base disc is removed inside the notebook by an analytic pi/4 d^2, so any
#     mismatch in where nTop places the base rim shows up here.
# The measured error on the default SV-1 is -0.21 percent. 1.5 percent is the gate: seven
# times the observed error, and still far tighter than the fidelity of the aero model that
# consumes the number.
AREA_TOLERANCE = 0.015

# A SOLID billet of 7075-T6 filling the default 0.3355 m^3 outer mould line would weigh
# 0.3355 * 2810 = 943 kg. The measured structure is the hollow airframe wall, three ring
# bulkheads and four fin panels ONLY - no motor case, no propellant, no warhead, no avionics,
# because `rocketgen/sizing/masses.py` charges all of those separately and double counting
# them would corrupt the whole mass statement. So the answer must be tens of kilograms.
MASS_STRUCTURE_MIN = 10.0
MASS_STRUCTURE_MAX = 120.0

# Volume closure: the cavity must hold the propellant, the warhead and the avionics.
PROPELLANT_DENSITY = MATERIALS["propellant_htpb_ap"].density          # 1800 kg/m^3
WARHEAD_PACKING_DENSITY = 1750.0     # masses.py, SOURCES["warhead_density"]
AVIONICS_DENSITY = 1200.0            # masses.py, RHO_AVIONICS


# --------------------------------------------------------------------------------------
#   Pure-Python checks: no ntopcl needed
# --------------------------------------------------------------------------------------


def test_tangent_ogive_helpers_match_the_definition() -> None:
    """rho = (R^2 + L^2)/(2R), y(0) = 0, y(L) = R, and the profile is monotone."""
    L, R = 1.05, 0.175
    rho = tangent_ogive_rho(L, R)
    assert rho == pytest.approx((R * R + L * L) / (2.0 * R))
    assert tangent_ogive_y(0.0, L, R) == pytest.approx(0.0, abs=1e-12)
    assert tangent_ogive_y(L, L, R) == pytest.approx(R, rel=1e-12)
    ys = [tangent_ogive_y(L * i / 50.0, L, R) for i in range(51)]
    assert all(b >= a for a, b in zip(ys, ys[1:])), "ogive profile must be monotone"
    # A tangent ogive is fuller than the cone of the same length and base radius.
    assert tangent_ogive_y(0.5 * L, L, R) > 0.5 * R


def test_every_required_design_variable_is_an_ntop_input() -> None:
    """The WP4 brief lists exactly which design variables must be notebook inputs."""
    required = {
        "D", "L_total", "f_nose", "t_wall", "L_boattail", "d_base", "b_fin", "c_r_fin",
        "taper_fin", "sweep_fin", "t_fin", "x_fin_te_gap", "L_seeker", "L_guidance",
        "L_warhead",
    }
    assert {attr for attr, _, _ in NTOP_INPUTS} == required
    dv = DesignVector()
    for attr, _, _ in NTOP_INPUTS:
        assert isinstance(getattr(dv, attr), float), attr


def test_recipe_declares_the_inputs_and_one_json_output() -> None:
    """Authoring is pure Python, so this runs without nTop."""
    dv = DesignVector()
    r = build_rocket_recipe(dv, GEOM_DIR, export_stl=True, export_step=True,
                             export_implicit=True)
    names = r.input_names()
    for attr, iname, _ in NTOP_INPUTS:
        assert iname in names, iname
        idx = names.index(iname)
        got = r.inputs[idx]["contents"]["value"]["val"]
        assert got == pytest.approx(float(getattr(dv, attr))), iname
    # Non-geometric inputs: cost and destinations, so a re-convert is never needed for them.
    assert "Mesh Tolerance" in names
    assert "STL Path" in names and "STEP Path" in names and "Implicit Path" in names
    # Exactly ONE output slot (docs/NTOP_NOTES.md section 1), carrying a json dictionary.
    doc = r.to_dict()
    assert isinstance(doc["output"], dict) and "id" in doc["output"]
    funcs = [e["contents"].get("func") for e in doc["body"]]
    assert "json_from_dictionary<dictionary<text,real>>[5.30.0]" in funcs
    # And it really is a single revolve, not three booleaned primitives.
    assert sum(1 for f in funcs if f and f.startswith("revolve")) == 1


def test_recipe_rejects_unsupported_topology() -> None:
    with pytest.raises(ValueError, match="n_fin"):
        build_rocket_recipe(DesignVector().replace(n_fin=3), GEOM_DIR)
    with pytest.raises(ValueError, match="nose_shape"):
        build_rocket_recipe(DesignVector().replace(nose_shape="haack"), GEOM_DIR)


def test_cone_nose_is_supported() -> None:
    r = build_rocket_recipe(DesignVector().replace(nose_shape="cone"), GEOM_DIR)
    doc = r.to_dict()
    points = [e for e in doc["body"] if e["contents"].get("func") == "point<real,real,real>"]
    ogive = build_rocket_recipe(DesignVector(), GEOM_DIR).to_dict()
    ogive_points = [e for e in ogive["body"]
                    if e["contents"].get("func") == "point<real,real,real>"]
    assert len(points) < len(ogive_points), "a cone needs far fewer profile points"


def test_output_name_map_was_extended_not_replaced() -> None:
    """WP4 owns its output names and registers them additively; config.py is untouched."""
    for name in ("volume_total", "volume_structure", "volume_cavity", "area_wetted_body",
                 "area_wetted_fins", "area_base", "mass_structure"):
        assert OUTPUT_NAME_MAP[name] == name
    # WP1's entries survive.
    assert OUTPUT_NAME_MAP["volume"] == "volume_total"
    assert OUTPUT_NAME_MAP["wetted_area"] == "area_wetted_body"


def test_measure_rocket_matches_the_geometry_fn_signature() -> None:
    """`loop.GeometryFn` is `Callable[[DesignVector, str], NtopMeasurements]`."""
    import inspect

    sig = inspect.signature(measure_rocket)
    params = list(sig.parameters.values())
    assert params[0].name == "dv" and params[0].default is inspect.Parameter.empty
    assert params[1].name == "run_dir" and params[1].default is inspect.Parameter.empty
    # everything after must be optional, so a two-argument call works
    assert all(p.default is not inspect.Parameter.empty or p.kind == p.VAR_KEYWORD
               for p in params[2:])
    assert sig.return_annotation in (NtopMeasurements, "NtopMeasurements")
    fn: GeometryFn = measure_rocket          # a type-checkable assignment
    assert fn is measure_rocket


def test_invalid_design_vector_fails_before_spending_an_ntopcl_call() -> None:
    """A geometry that cannot close must fail cleanly with diagnostics, not hang.

    `DesignVector.geometry_is_valid()` is the cheap gate. This design has a base diameter
    larger than the body and a cylindrical section that is far too short, so no notebook is
    authored and no subprocess is started.
    """
    bad = DesignVector().replace(D=0.30, L_total=3.0, f_nose=4.0, d_base=0.40)
    ok, errs = bad.geometry_is_valid()
    assert not ok and errs
    d = os.path.join(GEOM_DIR, "invalid")
    with pytest.raises(ValueError) as exc:
        measure_rocket(bad, d)
    message = str(exc.value)
    assert "invalid design vector" in message
    assert "base diameter" in message
    assert not os.path.exists(os.path.join(d, "sv1_output.json"))


# --------------------------------------------------------------------------------------
#   End to end against a real ntopcl
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def runner() -> NtopRunner:
    try:
        return NtopRunner()
    except NtopError as exc:                                          # pragma: no cover
        pytest.skip(f"no ntopcl available: {exc}")


@pytest.fixture(scope="session")
def baseline_dv() -> DesignVector:
    return DesignVector()


@pytest.fixture(scope="session")
def baseline(runner: NtopRunner, baseline_dv: DesignVector) -> NtopMeasurements:
    """Build the notebook and measure the default SV-1. This is the reference geometry.

    Deliberately uses the DEFAULT export flags - which are all OFF - so that this notebook is
    the one the sizing loop will also get and the cache-reuse tests are meaningful. Turning any
    export on adds blocks, so it is a different topology and a different cached notebook; the
    exports get their own test.
    """
    return measure_rocket(
        baseline_dv, os.path.join(GEOM_DIR, "baseline"), runner, tag="sv1",
    )


# ---- 1. volume against the independent closed form ----------------------------------


def test_volume_total_matches_closed_form(baseline: NtopMeasurements,
                                          baseline_dv: DesignVector) -> None:
    """`masses.analytic_geometry` is the independent reference and is itself tested
    (`tests/test_masses.py` proves the ogive quadrature against an exact hemisphere)."""
    ref = analytic_geometry(baseline_dv)["volume_total"]
    got = baseline.volume_total
    assert got is not None
    err = abs(got - ref) / ref
    assert err < VOLUME_TOLERANCE, (
        f"nTop OML volume {got:.6f} m^3 vs closed form {ref:.6f} m^3, "
        f"error {err * 100:.4f} percent"
    )


# ---- 2. wetted area against the closed form ----------------------------------------


def test_area_wetted_body_matches_closed_form(baseline: NtopMeasurements,
                                              baseline_dv: DesignVector) -> None:
    ref = analytic_geometry(baseline_dv)["area_wetted_body"]
    got = baseline.area_wetted_body
    assert got is not None
    err = abs(got - ref) / ref
    assert err < AREA_TOLERANCE, (
        f"nTop body wetted area {got:.6f} m^2 vs closed form {ref:.6f} m^2, "
        f"error {err * 100:.4f} percent (tolerance {AREA_TOLERANCE * 100:.1f} percent)"
    )


def test_area_base_is_exact(baseline: NtopMeasurements, baseline_dv: DesignVector) -> None:
    """The base area is pi/4 d_base^2 computed inside nTop from the input, so it is exact."""
    assert baseline.area_base is not None
    assert baseline.area_base == pytest.approx(baseline_dv.S_base, rel=1e-9)


def test_area_wetted_fins_is_close_to_the_flat_plate_value(
    baseline: NtopMeasurements, baseline_dv: DesignVector
) -> None:
    """Fin wetted area, with the reason an exact match is impossible stated.

    The closed form is a zero-thickness flat plate: 2 * n_fin * S_fin_exposed. The measured
    solid is a double wedge, so it also carries the root junction faces where the panel meets
    the body, the tip end face, and the (very slightly) longer wedge surfaces. The measured
    value must therefore be a little LARGER, not equal.
    """
    ref = analytic_geometry(baseline_dv)["area_wetted_fins"]
    got = baseline.area_wetted_fins
    assert got is not None
    assert 0.85 * ref < got < 1.35 * ref, (
        f"nTop fin wetted area {got:.6f} m^2 vs flat-plate closed form {ref:.6f} m^2"
    )


# ---- 3. structure mass excludes the motor and the payload --------------------------


def test_structure_mass_is_a_hollow_airframe_not_a_billet(
    baseline: NtopMeasurements, baseline_dv: DesignVector
) -> None:
    """The single most important check in this file.

    `mass_structure` must be the airframe wall, bulkheads and fins ONLY. The motor case comes
    from `SolidMotor` and the warhead and guidance masses come straight from `Requirements`,
    so anything else in here is double counted. A solid billet of 7075-T6 filling the
    0.3355 m^3 outer mould line would be 943 kg; a hollow 3 mm airframe is tens of kg.
    """
    m = baseline.mass_structure
    v = baseline.volume_structure
    assert m is not None and v is not None
    billet = analytic_geometry(baseline_dv)["volume_total"] * \
        MATERIALS["airframe_al7075"].density
    assert billet > 900.0, "sanity: the billet reference should be around 940 kg"
    assert MASS_STRUCTURE_MIN < m < MASS_STRUCTURE_MAX, (
        f"structure mass {m:.1f} kg is outside the physical band "
        f"({MASS_STRUCTURE_MIN} to {MASS_STRUCTURE_MAX} kg). A solid billet would be "
        f"{billet:.0f} kg, so a value near that means the body was never hollowed; a value "
        f"far above 120 kg means the motor case, propellant, warhead or avionics were "
        f"double counted into the structure."
    )
    # mass = density * volume, computed inside nTop by `mass_properties`.
    assert m == pytest.approx(v * MATERIALS["airframe_al7075"].density, rel=0.01)
    # A thin-wall shell plus fins: closed-form cross-check to a factor of well under two.
    g = analytic_geometry(baseline_dv)
    shell = g["area_wetted_body"] * baseline_dv.t_wall
    fins = baseline_dv.n_fin * 0.5 * baseline_dv.t_fin * baseline_dv.S_fin_exposed
    assert 0.6 * (shell + fins) < v < 1.8 * (shell + fins), (
        f"structure volume {v:.6f} m^3 vs closed-form shell+fins {shell + fins:.6f} m^3"
    )


def test_structure_cg_is_on_the_body_axis(baseline: NtopMeasurements,
                                          baseline_dv: DesignVector) -> None:
    """A cruciform rocket is symmetric about both the XY and XZ planes."""
    cg = baseline.cg_structure
    assert cg is not None, "cg_structure must be reported"
    x, y, z = cg
    assert 0.3 * baseline_dv.L_total < x < 0.8 * baseline_dv.L_total, (
        f"structure CG at x = {x:.3f} m is not plausibly inside the body"
    )
    # The CG is not exactly on the axis: `mass_properties` is an adaptive volume integration
    # to a relative-error target, so the first moments carry that error too. The measured
    # off-axis offset on the default SV-1 is 1.2 mm, i.e. 0.35 percent of a calibre. The gate
    # is 1 percent of D, which is still far tighter than anything the stability model needs.
    assert abs(y) < 0.01 * baseline_dv.D, f"CG y = {y:.6f} m should be on the axis"
    assert abs(z) < 0.01 * baseline_dv.D, f"CG z = {z:.6f} m should be on the axis"


def test_structure_inertia_is_reported_and_ordered(baseline: NtopMeasurements) -> None:
    """`principal moments` is a vector; its three components come back as three scalars.

    These are PRINCIPAL moments about the CG, not Ixx/Iyy/Izz in body axes. For a cruciform
    body of revolution the principal axes coincide with the body axes, so the roll moment is
    the smallest and the two transverse moments are nearly equal.
    """
    inertia = baseline.inertia_structure
    if inertia is None:
        pytest.skip("nTop did not expose principal moments on this build")
    assert all(i > 0.0 for i in inertia)
    small, mid, large = sorted(inertia)
    assert mid == pytest.approx(large, rel=0.10), (
        f"the two transverse principal moments should be nearly equal, got {inertia}"
    )
    assert small < 0.5 * mid, (
        f"the roll moment should be much smaller than the transverse ones, got {inertia}"
    )


# ---- 4. cavity volume closure ------------------------------------------------------


def test_cavity_holds_the_propellant_warhead_and_avionics(
    baseline: NtopMeasurements, baseline_dv: DesignVector
) -> None:
    v_cav = baseline.volume_cavity
    v_tot = baseline.volume_total
    assert v_cav is not None and v_tot is not None
    assert v_cav > 0.0
    assert v_cav < v_tot, "the cavity cannot be larger than the enclosed volume"

    reqs = Requirements()
    m_prop = baseline_dv.m_p_boost + baseline_dv.m_p_sustain
    v_prop = m_prop / PROPELLANT_DENSITY
    v_warhead = reqs.m_warhead / WARHEAD_PACKING_DENSITY
    v_avionics = reqs.m_guidance / AVIONICS_DENSITY
    needed = v_prop + v_warhead + v_avionics
    assert v_prop == pytest.approx(0.20, abs=0.005), (
        "the default 360 kg of propellant at 1800 kg/m^3 should be 0.20 m^3"
    )
    assert v_cav >= needed, (
        f"cavity {v_cav:.4f} m^3 cannot hold {needed:.4f} m^3 "
        f"(propellant {v_prop:.4f} + warhead {v_warhead:.4f} + avionics {v_avionics:.4f})"
    )
    # The cavity plus the structure must account for the enclosed volume, to within the
    # bulkhead volume and the integration tolerance.
    assert baseline.volume_structure is not None
    assert v_cav + baseline.volume_structure == pytest.approx(v_tot, rel=0.03)


# ---- 5. two design points through ONE notebook -------------------------------------


@pytest.fixture(scope="session")
def alternate(runner: NtopRunner, baseline: NtopMeasurements) -> NtopMeasurements:
    """A second, larger design point. Depends on `baseline` so the notebook already exists."""
    dv = DesignVector().replace(D=0.40, L_total=4.20)
    return measure_rocket(
        dv, os.path.join(GEOM_DIR, "alternate"), runner, tag="sv1_alt",
    )


def test_the_same_notebook_serves_two_design_vectors(
    baseline: NtopMeasurements, alternate: NtopMeasurements
) -> None:
    """The design variables are real nTop inputs, so one `.ntop` covers both points.

    This is the caching gate: if the dimensions had been baked in as literals, the second
    measurement would either equal the first or need its own `convert`.
    """
    assert baseline.ntop_path == alternate.ntop_path, (
        "the two design points must run through the SAME cached notebook"
    )
    assert baseline.volume_total is not None and alternate.volume_total is not None
    # D 0.35 -> 0.40 and L 4.00 -> 4.20: bigger in every way, so the volume must rise.
    assert alternate.volume_total > baseline.volume_total
    ref_base = analytic_geometry(DesignVector())["volume_total"]
    ref_alt = analytic_geometry(DesignVector().replace(D=0.40, L_total=4.20))["volume_total"]
    assert ref_alt > ref_base
    assert abs(alternate.volume_total - ref_alt) / ref_alt < VOLUME_TOLERANCE, (
        f"alternate OML volume {alternate.volume_total:.6f} vs closed form {ref_alt:.6f}"
    )
    # The measured ratio must track the closed-form ratio, not merely be larger.
    got_ratio = alternate.volume_total / baseline.volume_total
    ref_ratio = ref_alt / ref_base
    assert got_ratio == pytest.approx(ref_ratio, rel=0.02)
    # Wetted area moves the same way.
    assert alternate.area_wetted_body is not None
    assert baseline.area_wetted_body is not None
    assert alternate.area_wetted_body > baseline.area_wetted_body


def test_notebook_is_reused_not_reconverted(runner: NtopRunner,
                                            baseline: NtopMeasurements) -> None:
    """A third call must reuse the cached `.ntop` rather than convert again."""
    path = build_rocket_notebook(DesignVector().replace(D=0.30), GEOM_DIR, runner)
    assert path == baseline.ntop_path
    assert os.path.isfile(path)


# ---- exports ------------------------------------------------------------------------


def test_baseline_writes_no_exports(baseline: NtopMeasurements) -> None:
    """Every export is off by default, because they are measured to be the expensive part.

    The measurement blocks cost about 30 s per run; the STL adds about 40 s and over a gigabyte
    of resident memory, and STEP more again. The sizing loop needs none of it.
    """
    assert baseline.stl_path is None
    assert baseline.step_path is None
    assert baseline.implicit_path is None
    assert baseline.is_usable(), "the measurements still have to be complete"


@pytest.fixture(scope="session")
def exports(runner: NtopRunner, baseline_dv: DesignVector) -> NtopMeasurements:
    """The deliverable geometry: STL, STEP and `.implicit`, in `runs/SV-1_geom/exports/`.

    Turning an export on adds blocks, so this is a different topology and therefore its own
    cached `.ntop`. That is exactly why the exports are off by default.

    Both export tolerances are the measured defaults, not guesses. The STL mesh at 5.0e-3 m
    takes about 52 s and 9.2 MB; finer than 3.0e-3 m exhausts memory. The CAD tolerance for
    STEP is 1.0e-2 m, which takes about 23 s and 9.0 MB; at 2.0e-3 m the process passed 9 GB
    resident and had to be killed. See `rocket_notebook.SOURCES`.
    """
    return measure_rocket(
        baseline_dv, os.path.join(GEOM_DIR, "exports"), runner,
        export_stl=True, export_step=True, export_implicit=True,
        timeout=3600.0, tag="sv1",
    )


def test_exports_land_where_the_inputs_point(exports: NtopMeasurements,
                                             baseline_dv: DesignVector) -> None:
    """Export paths are notebook INPUTS, so artefacts follow `run_dir` with no re-convert."""
    assert exports.stl_path and os.path.getsize(exports.stl_path) > 0
    assert exports.step_path and os.path.getsize(exports.step_path) > 0, "no STEP produced"
    assert exports.implicit_path and os.path.getsize(exports.implicit_path) > 0
    assert os.path.dirname(exports.stl_path) == os.path.join(GEOM_DIR, "exports")
    # The extra blocks only export; they must not change the solid.
    ref = analytic_geometry(baseline_dv)["volume_total"]
    assert exports.volume_total is not None
    assert abs(exports.volume_total - ref) / ref < VOLUME_TOLERANCE


def test_stl_volume_is_consistent_with_the_measured_volume(
    exports: NtopMeasurements, baseline_dv: DesignVector
) -> None:
    """The STL is a coarse mesh, so it is only a consistency check, never a measurement.

    docs/NTOP_NOTES.md section 4: the block's own `mass_properties` beat the exported STL by
    16x on the WP1 smoke sphere, which is why SUAVE is fed the notebook's numbers.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from stlreader import bounding_box, enclosed_volume, read_stl

    tris = read_stl(str(exports.stl_path))
    assert len(tris) > 1000
    lo, hi = bounding_box(tris)
    # Exported in metres. The body spans L_total in x and the fin span in y and z.
    assert hi[0] - lo[0] == pytest.approx(baseline_dv.L_total, rel=0.02)
    span = baseline_dv.D + 2.0 * baseline_dv.b_fin
    assert hi[1] - lo[1] == pytest.approx(span, rel=0.06), "the fins must survive the meshing"
    v = enclosed_volume(tris)
    assert exports.volume_total is not None
    # The mesh carries the fins, which the OML volume does not, but it is also a 5 mm-tolerance
    # mesh of a 0.35 m body, so this is a consistency band and not a measurement.
    assert 0.95 * exports.volume_total < v < 1.08 * exports.volume_total


# ---- area distribution S(x) ---------------------------------------------------------


def test_area_distribution_is_empty_by_default(baseline: NtopMeasurements) -> None:
    """`area_stations = 0` by default, and the omission is declared in the warnings.

    `rocketgen/sizing/aero.py` falls back to closed-form cross-section geometry when
    `area_distribution` is empty, so the default costs nothing but a warning.
    """
    assert baseline.area_distribution == []
    assert any("area_distribution is empty" in w for w in baseline.warnings)


def test_area_distribution_reproduces_the_closed_form_section_areas(
    runner: NtopRunner, baseline_dv: DesignVector
) -> None:
    """S(x) for wave drag, measured by nTop rather than assumed.

    There is no single block (docs/NTOP_NOTES.md section 13 point 6). The route is
    `extract_section<implicit,plane,real>` then a cross-section area on the resulting 2D region.
    The reference is the closed-form ogive-cylinder-boattail radius plus the fin plate sections,
    which is entirely independent of nTop.
    """
    dv = baseline_dv
    n = 8
    m = measure_rocket(
        dv, os.path.join(GEOM_DIR, "area_distribution"), runner,
        area_stations=n, tag="sv1_sx", timeout=900.0, convert_timeout=900.0,
    )
    assert len(m.area_distribution) == n
    stations = [x for x, _ in m.area_distribution]
    assert stations == sorted(stations)
    assert all(0.0 < x < dv.L_total for x in stations)

    R = 0.5 * dv.D
    x_cyl_end = dv.L_total - dv.L_boattail
    for x, s in m.area_distribution:
        if x <= dv.L_nose:
            y = tangent_ogive_y(x, dv.L_nose, R)
        elif x <= x_cyl_end:
            y = R
        else:
            f = (x - x_cyl_end) / dv.L_boattail
            y = R + f * (0.5 * dv.d_base - R)
        ref = math.pi * y * y
        # Add the fin plate sections where the panels are present. The four plates are
        # t_fin thick and, between the root LE and the tip TE, span the local exposed chord.
        if dv.x_fin_le <= x <= dv.L_total - dv.x_fin_te_gap:
            tan_s = math.tan(dv.sweep_fin)
            steps = 400
            span = 0.0
            for k in range(steps):
                yy = y + dv.b_fin * (k + 0.5) / steps
                x_le = dv.x_fin_le + (yy - R) * tan_s
                chord = dv.c_r_fin + (yy - R) / dv.b_fin * (dv.c_t_fin - dv.c_r_fin)
                if x_le <= x <= x_le + chord:
                    span += dv.b_fin / steps
            ref += dv.n_fin * span * dv.t_fin
        assert s == pytest.approx(ref, rel=0.01), (
            f"S(x) at x = {x:.3f} m: nTop {s:.6f} m^2 vs closed form {ref:.6f} m^2"
        )
    # The maximum section must be at or after the nose shoulder and must include the fins.
    s_max = max(s for _, s in m.area_distribution)
    assert s_max >= math.pi * R * R
    assert not any("area_distribution is empty" in w for w in m.warnings)


# ---- 7. the real integration gate ---------------------------------------------------


def test_converge_point_accepts_measure_rocket_as_its_geometry_fn(
    runner: NtopRunner, baseline: NtopMeasurements
) -> None:
    """`converge_point(dv, reqs, geometry_fn=measure_rocket, run_dir=...)` must just work.

    One inner iteration only: this test is about the coupling, not about convergence. The gate
    is `PointResult.geometry_measured`, which is `NtopMeasurements.is_usable()` and therefore
    demands volume_total, volume_cavity, area_wetted_body and mass_structure all present.
    """
    d = os.path.join(GEOM_DIR, "loop")
    res = converge_point(
        DesignVector(), Requirements(),
        geometry_fn=measure_rocket, run_dir=d, max_iter=1, dt=0.1,
    )
    assert res.meas is not None, f"the loop never got measurements: {res.warnings}"
    assert res.geometry_measured is True, (
        f"geometry_measured is False; warnings: {res.warnings}"
    )
    assert res.meas.ntop_path == baseline.ntop_path, "the loop must reuse the cached notebook"
    assert res.masses is not None
    # The measured airframe really did replace the analytic estimate in the mass statement.
    names = [e.name for e in res.masses.entries]
    assert "Airframe structure and fins" in names
    entry = next(e for e in res.masses.entries if e.name == "Airframe structure and fins")
    assert entry.provenance == "ntop_measured"
    assert res.masses.measured_fraction > 0.0
    assert res.traj is not None


# ---- bookkeeping --------------------------------------------------------------------


def test_measurements_carry_their_bookkeeping(baseline: NtopMeasurements) -> None:
    assert baseline.wall_time_s is not None and baseline.wall_time_s > 0.0
    assert baseline.ntopcl_returncode in (0, 72)
    assert baseline.ntop_path and os.path.isfile(baseline.ntop_path)
    assert baseline.is_usable()
    # The measurement set is written next to the artefacts for inspection.
    assert os.path.isfile(os.path.join(GEOM_DIR, "baseline", "sv1_measurements.json"))


def test_measured_wall_time_is_reported(baseline: NtopMeasurements) -> None:
    """Not a pass/fail gate: the number is printed so the sizing loop can budget for it."""
    print(
        f"\nmeasure_rocket wall time: {baseline.wall_time_s:.1f} s at mesh tolerance "
        f"{DEFAULT_MESH_TOLERANCE:.1e} m, {N_OGIVE_OUTER} ogive segments"
    )
    assert baseline.wall_time_s is not None
