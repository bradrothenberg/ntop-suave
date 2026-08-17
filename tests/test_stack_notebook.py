"""IV-1 tests: the two-stage strake-equipped stack notebook, against a real `ntopcl`.

Nothing here is mocked. The notebook is authored, converted ONCE, and then run at three
different design vectors through the SAME `.ntop`, which is what proves the design variables are
real nTop inputs and that the topology cache works.

Artefacts land in `runs/IV-1_geom/` so they stay inspectable. pytest's tmpdir is deliberately not
used for them: an nTop artefact that vanishes when the test process exits cannot be looked at
when a number is surprising, and looking at the geometry is how most of these numbers were
debugged.

The reference for every geometric check is CLOSED FORM and independent of nTop:
`rocketgen.sizing.masses._tangent_ogive_volume` and `_tangent_ogive_surface_area`, which
`tests/test_masses.py` already validates against an exact hemisphere, plus cylinder, cone and
plate arithmetic written out in `stack_notebook.stack_geometry_closed_form` and re-derived by
hand in the comments below.

COST. One `measure_stack` call is 79 to 118 s (measured; see
`stack_notebook.SOURCES["measured_wall_time"]`). This file makes FOUR of them - the baseline, a
reduced strake height, a larger design point, and one with `area_stations = 6` - plus two
`convert`s of 63 to 96 s each, because `area_stations` changes the block graph and therefore needs
its own cached notebook. Budget roughly 10 minutes with a cold cache and 7 with a warm one.

The EXPORT run is not made here. `runs/IV-1_geom/_make_artefacts.py` produces the STL and the
render, which cost a further 220 s of run plus a 200 s convert, and `test_exported_stl_*` checks
its output when it is present and skips with instructions when it is not.

Run: `.venv/Scripts/python.exe -m pytest tests/test_stack_notebook.py -q`
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rocketgen.config import MATERIALS, RUNS_DIR, NtopMeasurements     # noqa: E402
from rocketgen.config_iv1 import (                                    # noqa: E402
    InterceptRequirements,
    StackDesignVector,
    StageSpec,
    StrakeSpec,
    default_iv1,
)
from rocketgen.ntopgen import NtopError, NtopRunner                    # noqa: E402
from rocketgen.ntopgen.driver import OUTPUT_NAME_MAP                   # noqa: E402
from rocketgen.ntopgen.stack_notebook import (                        # noqa: E402
    BODY_PREFIXES,
    DEFAULT_MESH_TOLERANCE,
    MEASURED_WALL_TIME_S,
    NTOP_GLOBAL_INPUTS,
    NTOP_STAGE_INPUTS,
    StageMeasurements,
    build_stack_notebook,
    build_stack_recipe,
    fin_solid_area,
    measure_stack,
    stack_geometry_closed_form,
    strake_solid_area,
)
from rocketgen.sizing.masses import (                                 # noqa: E402
    _tangent_ogive_surface_area,
    _tangent_ogive_volume,
)

GEOM_DIR = os.path.join(RUNS_DIR, "IV-1_geom")

# ---- tolerances, each stated and justified -------------------------------------------

# Volume. The brief's gate is 1 percent. The MEASURED errors against closed form on the default
# IV-1 are -0.008 percent (stage 2), -0.002 percent (booster) and +0.000 percent (interstage), so
# 1 percent is 125x the worst observed error. The residual is the 24-segment chord polygon of the
# ogive under-cutting the true curve, exactly as WP4 measured on SV-1.
VOLUME_TOLERANCE = 0.01

# Wetted area. Tighter than SV-1's 1.5 percent, and here is why that is justified rather than
# lucky. Three things stop an exact match:
#   * the revolved stage-2 outer mould line is a 24-segment CHORD polygon, which under-measures
#     the ogive lateral area;
#   * `surface_area<implicit,real>` integrates over the implicit field and nTop itself warns that
#     the field has undefined regions (docs/NTOP_NOTES.md section 11); its relative-error input
#     was swept from 0.002 to 0.2 and changed the answer by nothing at all, so there is no knob;
#   * the flat base disc (or, on the booster, both end discs) is removed inside the notebook by
#     an analytic pi/4 d^2, so any mismatch in where nTop places the rim shows up here.
# The MEASURED errors are -0.302 percent on the stage-2 body and +0.125 percent on the booster.
# 1.0 percent is 3.3x the worst of those, and far tighter than the fidelity of the aero model
# that consumes the number.
AREA_TOLERANCE = 0.010

# Plate wetted area, against the SOLID closed form in `strake_solid_area` / `fin_solid_area`.
# Measured: -0.531 percent (strakes), -0.183 percent (stage-2 fins), -0.184 percent (stage-1
# fins). The closed form is itself slightly approximate at the root, where it uses the arc of the
# body cylinder rather than the exact intersection with an ogive, so 1.5 percent is the gate.
PLATE_AREA_TOLERANCE = 0.015

# Structure mass, per stage. A SOLID 7075-T6 billet filling the whole default IV-1 envelope
# (0.4270 m^3 of stage-2 plus interstage plus booster outer mould line) would weigh
# 0.4270 * 2810 = 1200 kg, i.e. MORE than the entire A8 launch-mass limit of 1400 kg on its own.
# The measured structure is the hollow airframe wall, the bulkheads, the four strakes and the
# eight fin panels ONLY: no motor case, no propellant, no payload, no avionics, because
# `rocketgen/sizing/masses_iv1.py` charges every one of those separately and double counting them
# would corrupt the whole mass statement. So the answer must be TENS of kilograms per stage.
# Measured: 23.4 kg (stage 2), 32.5 kg (booster).
MASS_STRUCTURE_MIN = 8.0
MASS_STRUCTURE_MAX = 120.0
# The interstage is a 2.5 mm conical shell 0.28 m long; measured 3.4 kg.
MASS_INTERSTAGE_MIN = 0.5
MASS_INTERSTAGE_MAX = 20.0

PROPELLANT_DENSITY = MATERIALS["propellant_htpb_ap"].density          # 1800 kg/m^3
AIRFRAME_DENSITY = MATERIALS["airframe_al7075"].density               # 2810 kg/m^3


# --------------------------------------------------------------------------------------
#   Pure-Python checks: no ntopcl needed
# --------------------------------------------------------------------------------------


def test_strake_solid_area_closed_form_arithmetic() -> None:
    """The strake reference, worked by hand so the number in the report is auditable.

    A strake panel is a rectangle `height` x `length` of thickness `thickness`, standing on a
    body of radius R. Its wetted area, once the body has been subtracted, is

        two side faces        2 h L
        outboard tip face       t L
        root patch (arc)      arc L,   arc = 2 R asin(t / 2R) ~= t
        leading + trailing    2 t h

    On the default IV-1 (n=4, h=0.030 m, L=1.400 m, t=0.008 m, R=0.140 m):

        2 * 0.030 * 1.400 = 0.0840000
            0.008 * 1.400 = 0.0112000
        arc = 2*0.14*asin(0.008/0.28) = 0.00800109;  * 1.400 = 0.0112015
        2 * 0.008 * 0.030 = 0.0004800
                            ---------
                            0.1068815  per panel,  x 4 = 0.4275262 m^2

    `StrakeSpec.wetted_area` is the ZERO-THICKNESS reference, 2*4*0.030*1.400 = 0.336000 m^2.
    The solid is 1.2724x that, because an 8 mm plate only 30 mm tall is 27 percent edge.
    """
    st = StrakeSpec(n=4, height=0.030, length=1.400, thickness=0.008)
    R = 0.140
    arc = 2.0 * R * math.asin(0.008 / (2.0 * R))
    by_hand = 4.0 * (2.0 * 0.030 * 1.400 + 0.008 * 1.400 + arc * 1.400
                     + 2.0 * 0.008 * 0.030)
    assert by_hand == pytest.approx(0.4275262, abs=1e-6)
    assert strake_solid_area(st, R) == pytest.approx(by_hand, rel=1e-12)
    assert st.wetted_area == pytest.approx(0.336000, abs=1e-9)
    assert strake_solid_area(st, R) / st.wetted_area == pytest.approx(1.2724, abs=0.001)
    # Zero height or zero panels means no strakes at all, not a negative area.
    assert strake_solid_area(StrakeSpec(n=0), R) == 0.0
    assert strake_solid_area(StrakeSpec(height=0.0), R) == 0.0
    # Halving the height cuts the SIDE faces in half but leaves the tip face and the root patch
    # alone, so the total falls by less than half. That is the physics, not an error.
    half = strake_solid_area(StrakeSpec(n=4, height=0.015, length=1.400, thickness=0.008), R)
    assert 0.55 < half / by_hand < 0.65


def test_closed_form_booster_and_interstage_arithmetic() -> None:
    """The booster and interstage references, also worked by hand.

    Booster: a plain cylinder of radius R1 = D1/2 = 0.200 m and length L1 = 2.100 m.
        V = pi R^2 L         = pi * 0.04    * 2.1 = 0.2638938 m^3
        A_lateral = 2 pi R L = 2 pi * 0.2   * 2.1 = 2.6389378 m^2
        A_base = pi R^2                          = 0.1256637 m^2
    Interstage: a truncated cone from R2 = 0.140 m to R1 = 0.200 m over L = 0.280 m.
        V = pi L (R2^2 + R2 R1 + R1^2)/3
          = pi * 0.28 * (0.0196 + 0.028 + 0.04)/3 = 0.0256855 m^3
        A_lateral = pi (R2 + R1) sqrt(L^2 + (R1-R2)^2)
          = pi * 0.34 * sqrt(0.0784 + 0.0036)     = 0.3058688 m^2
    """
    dv = default_iv1()
    cf = stack_geometry_closed_form(dv)
    assert cf["s1"]["volume_total"] == pytest.approx(0.2638938, abs=1e-6)
    assert cf["s1"]["area_wetted_body"] == pytest.approx(2.6389378, abs=1e-6)
    assert cf["s1"]["area_base"] == pytest.approx(0.1256637, abs=1e-6)
    assert cf["is"]["volume_total"] == pytest.approx(0.0256855, abs=1e-6)
    assert cf["is"]["area_wetted_body"] == pytest.approx(0.3058688, abs=1e-6)

    # Stage 2 is the ogive quadrature plus a cylinder, and the quadrature is the one
    # `tests/test_masses.py` already validates against an exact hemisphere.
    R2, L_nose = 0.5 * dv.payload_stage.D, dv.L_nose
    L_cyl = dv.payload_stage.L - L_nose
    assert cf["s2"]["volume_total"] == pytest.approx(
        _tangent_ogive_volume(L_nose, R2) + math.pi * R2 * R2 * L_cyl, rel=1e-12
    )
    assert cf["s2"]["area_wetted_body"] == pytest.approx(
        _tangent_ogive_surface_area(L_nose, R2) + 2.0 * math.pi * R2 * L_cyl, rel=1e-12
    )
    # The stack totals are the sums of the three bodies, and the stack length is the design
    # vector's own derived length.
    assert cf["st"]["volume_total"] == pytest.approx(
        cf["s1"]["volume_total"] + cf["s2"]["volume_total"] + cf["is"]["volume_total"], rel=1e-12
    )
    assert cf["st"]["length"] == pytest.approx(dv.L_total, rel=1e-12)
    # And the geometry really is a stack: the booster's forward face is aft of the interstage,
    # which is aft of the payload stage.
    assert cf["s2"]["x_forward"] < cf["is"]["x_forward"] < cf["s1"]["x_forward"]


def test_fin_solid_area_exceeds_the_flat_plate_reference() -> None:
    """A solid plate is always more wetted than the zero-thickness planform it came from."""
    dv = default_iv1()
    for stage in dv.stages:
        R = 0.5 * stage.D
        flat = 2.0 * stage.n_fin * stage.S_fin_exposed
        solid = fin_solid_area(stage, R)
        assert solid > flat
        # The tail fins are much thinner relative to their span than the strakes are, so the
        # edge contribution is small: a few percent, not 27.
        assert solid / flat < 1.15
    assert fin_solid_area(dv.stages[0].__class__(index=9, D=0.3, L=1.0, m_propellant=1.0,
                                                F_thrust=1.0, n_fin=0), 0.15) == 0.0


# Fields of `StageSpec` that are NOT geometry: they never appear in the solid, so they must not
# be notebook inputs. Everything else on a `StageSpec` is a dimension of the body.
STAGE_NON_GEOMETRIC = {"index", "m_propellant", "F_thrust", "p_c", "eps_nozzle", "jettisoned",
                       "n_fin"}
# Stack-level fields that ARE geometry. The rest of `StackDesignVector` is the ascent programme.
STACK_GEOMETRIC = {"f_nose", "L_interstage", "t_interstage", "L_seeker", "L_payload_bay"}


def test_every_moving_dimension_is_an_ntop_input() -> None:
    """Convert once, run many: a dimension the sizer moves MUST be a real notebook input.

    `StackDesignVector.bounds()` is the sizer's contract, and every GEOMETRIC key in it has to be
    reachable as an nTop input. If one were not, the cached `.ntop` would silently measure the
    previous design point.

    Each bounds key is classified by the TYPE OF OBJECT THAT OWNS IT rather than by an enumerated
    list of names, so that a bounds key added by another work package cannot make this test either
    fail spuriously or pass vacuously:

    * owned by a `StageSpec`  -> geometry, unless the leaf is in `STAGE_NON_GEOMETRIC`
    * owned by a `StrakeSpec` -> always geometry
    * owned by the `StackDesignVector` itself -> geometry iff the leaf is in `STACK_GEOMETRIC`
    * owned by anything else  -> some other work package's sub-specification, and not part of the
      solid this notebook builds, so it is skipped and reported
    """
    dv0 = default_iv1()
    stage_attrs = {attr for attr, _, _ in NTOP_STAGE_INPUTS}
    global_paths = {path for path, _, _ in NTOP_GLOBAL_INPUTS}

    checked, skipped = 0, []
    for key in dv0.bounds():
        parts = key.split(".")
        owner: object = dv0
        for p in parts[:-1]:
            owner = owner[int(p)] if p.isdigit() else getattr(owner, p)  # type: ignore[index]
        leaf = parts[-1]

        if isinstance(owner, StageSpec):
            if leaf in STAGE_NON_GEOMETRIC:
                continue
            assert leaf in stage_attrs, (
                f"{key} is a StageSpec dimension the sizer moves but is not an nTop input, so "
                f"one cached .ntop cannot serve two design points"
            )
        elif isinstance(owner, StrakeSpec):
            assert f"strakes.{leaf}" in global_paths, f"{key} is not an nTop input"
        elif owner is dv0:
            if leaf not in STACK_GEOMETRIC:
                continue
            assert key in global_paths, f"{key} is not an nTop input"
        else:
            skipped.append(key)
            continue
        checked += 1

    assert checked >= 11, f"only {checked} geometric bounds keys were checked; the test is weak"
    if skipped:
        print(f"\nnon-geometry bounds keys owned by other work packages, skipped: {skipped}")

    # Topology, and therefore BAKED IN rather than an input. Named here so the list is explicit
    # and so a future author cannot make one an input without noticing this test.
    assert "n_fin" not in stage_attrs
    assert "nose_shape" not in global_paths
    assert "strakes.n" not in global_paths


def test_recipe_declares_the_inputs_and_one_json_output() -> None:
    """Authoring is pure Python, so this runs without nTop."""
    dv = default_iv1()
    r = build_stack_recipe(dv, GEOM_DIR, export_stl=True, export_step=True,
                           export_implicit=True)
    names = r.input_names()

    for stage in dv.stages:
        for attr, suffix, _ in NTOP_STAGE_INPUTS:
            iname = f"S{stage.index} {suffix}"
            assert iname in names, iname
            got = r.inputs[names.index(iname)]["contents"]["value"]["val"]
            assert got == pytest.approx(float(getattr(stage, attr))), iname
    for path, iname, _ in NTOP_GLOBAL_INPUTS:
        assert iname in names, iname
    assert "Mesh Tolerance" in names
    assert "STL Path" in names and "STEP Path" in names and "Implicit Path" in names

    doc = r.to_dict()
    # Exactly ONE output slot (docs/NTOP_NOTES.md section 1), carrying a json dictionary.
    assert isinstance(doc["output"], dict) and "id" in doc["output"]
    funcs = [e["contents"].get("func") for e in doc["body"]]
    assert "json_from_dictionary<dictionary<text,real>>[5.30.0]" in funcs
    # ONE revolve: only the payload stage is a body of revolution. The booster is a `cylinder`
    # primitive and the interstage a `cone`, which is why there is no topology switch between a
    # conical and a cylindrical interstage.
    assert sum(1 for f in funcs if f and f.startswith("revolve")) == 1
    assert sum(1 for f in funcs if f == "cone<point,point,real,real>") == 1
    # Twelve plates: four strakes and four fins on each of two stages, built as two panels plus
    # two mirrors per set, so six extrude blocks and six mirror blocks.
    assert sum(1 for f in funcs if f and f.startswith("extrude")) == 6
    assert sum(1 for f in funcs if f == "mirror_body<implicit,plane>") == 6


def test_recipe_rejects_unsupported_topology() -> None:
    dv = default_iv1()
    with pytest.raises(ValueError, match="strakes.n"):
        build_stack_recipe(dv.replace(strakes=StrakeSpec(n=3)), GEOM_DIR)
    with pytest.raises(ValueError, match="n_fin"):
        build_stack_recipe(dv.with_path("stages.1.n_fin", 3), GEOM_DIR)
    with pytest.raises(ValueError, match="nose_shape"):
        build_stack_recipe(dv.replace(nose_shape="haack"), GEOM_DIR)
    with pytest.raises(ValueError, match="build_stage"):
        build_stack_recipe(dv, GEOM_DIR, build_stage="nonsense")


def test_cone_nose_is_supported_and_cheaper() -> None:
    dv = default_iv1()
    cone = build_stack_recipe(dv.replace(nose_shape="cone"), GEOM_DIR).to_dict()
    ogive = build_stack_recipe(dv, GEOM_DIR).to_dict()

    def n_points(doc: dict) -> int:
        return sum(1 for e in doc["body"]
                   if e["contents"].get("func") == "point<real,real,real>")

    assert n_points(cone) < n_points(ogive), "a cone needs far fewer profile points"
    # The closed form follows the choice too: a cone of the same length and base radius holds
    # less than a tangent ogive, which is fuller.
    v_cone = stack_geometry_closed_form(dv.replace(nose_shape="cone"))["s2"]["volume_total"]
    v_ogive = stack_geometry_closed_form(dv)["s2"]["volume_total"]
    assert v_cone < v_ogive


def test_output_key_naming_convention_is_registered() -> None:
    """The per-body prefix convention, asserted so it cannot drift silently.

    One notebook now reports three bodies plus the stack, so every scalar is namespaced:
    `s1_` booster, `s2_` payload stage, `is_` interstage, `st_` stacked assembly.
    `driver.OUTPUT_NAME_MAP` was extended ADDITIVELY through `register_output_names`;
    `rocketgen/config.py` is untouched.
    """
    assert set(BODY_PREFIXES) == {"s1", "s2", "is", "st"}
    for prefix in BODY_PREFIXES:
        for name in ("volume_total", "volume_structure", "volume_cavity", "area_wetted_body",
                     "area_wetted_fins", "area_base", "mass_structure"):
            assert OUTPUT_NAME_MAP[f"{prefix}_{name}"] == name
    # SV-1's entries survive untouched.
    assert OUTPUT_NAME_MAP["volume"] == "volume_total"
    assert OUTPUT_NAME_MAP["wetted_area"] == "area_wetted_body"
    assert OUTPUT_NAME_MAP["volume_oml"] == "volume_total"


def test_stage_measurements_extends_ntop_measurements() -> None:
    """`StageMeasurements` is a SUBCLASS, so `config.py` did not have to be edited.

    That is what lets `dict[int, StageMeasurements]` satisfy `dict[int, NtopMeasurements]` for
    `masses_iv1.build_stack_masses` and `aero_iv1.StackAero` with no adapter, while still
    carrying `area_wetted_strakes` apart from `area_wetted_fins`.
    """
    assert issubclass(StageMeasurements, NtopMeasurements)
    m = StageMeasurements(stage_index=2, area_wetted_fins=0.25, area_wetted_strakes=0.42)
    assert isinstance(m, NtopMeasurements)
    assert m.area_wetted_surfaces == pytest.approx(0.67)
    # The strake area is NOT folded into the fin area, because aero_iv1 needs them apart.
    assert m.area_wetted_fins == 0.25
    for extra in ("area_wetted_strakes", "volume_strakes", "volume_fins", "x_forward",
                  "cg_structure_stack", "stage_index", "body"):
        assert extra in StageMeasurements.__dataclass_fields__
        assert extra not in NtopMeasurements.__dataclass_fields__


def test_measure_stack_returns_a_dict_keyed_by_stage() -> None:
    """Signature check: the sizing loop calls `measure_stack(dv, run_dir)` positionally."""
    import inspect

    sig = inspect.signature(measure_stack)
    params = list(sig.parameters.values())
    assert params[0].name == "dv" and params[0].default is inspect.Parameter.empty
    assert params[1].name == "run_dir" and params[1].default is inspect.Parameter.empty
    assert all(p.default is not inspect.Parameter.empty or p.kind == p.VAR_KEYWORD
               for p in params[2:])
    assert "dict" in str(sig.return_annotation)


def test_invalid_design_vector_fails_cleanly_before_spending_an_ntopcl_call() -> None:
    """Requirement 7: an invalid design vector fails with diagnostics, not a hang or junk.

    Three separate faults at once, so the message has to carry more than one:
    the payload stage is FATTER than the booster (SPEC_IV1 section 6 forbids it), the nose is
    long enough to swallow the bays, and the strakes run off the back of the stage.
    `StackDesignVector.geometry_is_valid()` is the cheap gate, so no notebook is authored and no
    subprocess is started.
    """
    dv = default_iv1()
    bad = (dv.with_path("stages.1.D", 0.42)
             .with_path("stages.1.L", 1.9)
             .replace(f_nose=5.0))
    ok, errs = bad.geometry_is_valid()
    assert not ok and len(errs) >= 2, errs

    d = os.path.join(GEOM_DIR, "invalid")
    with pytest.raises(ValueError) as exc:
        measure_stack(bad, d)
    message = str(exc.value)
    assert "invalid design vector" in message
    assert "exceeds booster" in message
    # Nothing was run, so nothing was written.
    assert not os.path.exists(os.path.join(d, "iv1_output.json"))

    # A stack with only one stage is not an IV-1 at all, and must also be refused.
    with pytest.raises(ValueError, match="at least two stages"):
        measure_stack(StackDesignVector(stages=[dv.booster]), d)


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
def dv() -> StackDesignVector:
    return default_iv1()


@pytest.fixture(scope="session")
def meas(runner: NtopRunner, dv: StackDesignVector) -> dict[int, StageMeasurements]:
    """Build the notebook and measure the default IV-1. The reference geometry.

    Deliberately uses the DEFAULT export flags - all OFF - so this is the same notebook the
    sizing loop will get and the cache-reuse tests mean something. Turning an export on adds
    blocks, so it is a different topology and a different cached `.ntop`.
    """
    return measure_stack(dv, os.path.join(GEOM_DIR, "baseline"), runner, tag="iv1")


@pytest.fixture(scope="session")
def closed_form(dv: StackDesignVector) -> dict[str, dict[str, float]]:
    return stack_geometry_closed_form(dv)


# ---- 8. both stage keys, both usable -------------------------------------------------


def test_measure_stack_returns_both_stages_and_each_is_usable(
    meas: dict[int, StageMeasurements]
) -> None:
    """Requirement 8. `is_usable()` demands volume_total, volume_cavity, area_wetted_body and
    mass_structure, so this is the gate the sizing loop itself applies."""
    assert set(meas) == {0, 1, 2, -1}, "keys are 1, 2, the interstage (-1) and the stack (0)"
    for key in (1, 2):
        m = meas[key]
        assert m.stage_index == key
        assert m.is_usable(), f"stage {key} is not usable; warnings: {m.warnings}"
        assert m.ntop_path and os.path.isfile(m.ntop_path)
        assert m.ntopcl_returncode in (0, 72)
    # Stage 2 is the only stage with strakes; the booster must report none.
    assert meas[2].area_wetted_strakes is not None and meas[2].area_wetted_strakes > 0.0
    assert meas[1].area_wetted_strakes is None
    # The interstage is jettisoned with stage 1, and says so.
    assert meas[-1].mass_structure is not None
    assert any("jettisoned WITH stage 1" in w for w in meas[-1].warnings)
    # It is in the stack totals too, because the vehicle carries it at launch.
    assert meas[0].mass_structure == pytest.approx(
        (meas[1].mass_structure or 0.0) + (meas[2].mass_structure or 0.0)
        + (meas[-1].mass_structure or 0.0), rel=1e-9
    )
    # The stack is measured for volume by nTop and summed in Python for everything else, and
    # that provenance is recorded rather than hidden (CLAUDE.md 3.3).
    assert meas[0].is_usable()
    assert any("PYTHON SUMS" in w for w in meas[0].warnings)


# ---- 1. per-stage volume against the independent closed form ------------------------


@pytest.mark.parametrize("key,prefix", [(2, "s2"), (1, "s1"), (-1, "is")])
def test_volume_total_matches_closed_form(
    meas: dict[int, StageMeasurements], closed_form: dict[str, dict[str, float]],
    key: int, prefix: str,
) -> None:
    """Requirement 1, per body, against closed form better than 1 percent."""
    ref = closed_form[prefix]["volume_total"]
    got = meas[key].volume_total
    assert got is not None
    err = abs(got - ref) / ref
    assert err < VOLUME_TOLERANCE, (
        f"{prefix} nTop volume {got:.6f} m^3 vs closed form {ref:.6f} m^3, "
        f"error {err * 100:.4f} percent"
    )


def test_stack_volume_accounts_for_the_three_bodies_and_the_plates(
    meas: dict[int, StageMeasurements], closed_form: dict[str, dict[str, float]]
) -> None:
    """`st_volume_total` is measured on the UNION, so it must exceed the sum of the three
    outer mould lines by exactly the plate volume that sits outside them.

    That is a real cross-check and not a tautology: it links four separate nTop measurements.
    """
    st = meas[0].volume_total
    assert st is not None
    oml_sum = sum(closed_form[p]["volume_total"] for p in ("s1", "s2", "is"))
    plates = (meas[2].volume_strakes or 0.0) + (meas[2].volume_fins or 0.0) \
        + (meas[1].volume_fins or 0.0)
    assert plates > 0.0, "no plate volume measured, so the panels are missing"
    assert st == pytest.approx(oml_sum + plates, rel=0.01), (
        f"stack union {st:.6f} m^3 vs OML sum {oml_sum:.6f} + exposed plates {plates:.6f}"
    )
    # Nothing overlaps, so the union cannot be smaller than its largest part.
    assert st > max(m.volume_total or 0.0 for m in (meas[1], meas[2], meas[-1]))


# ---- 2. per-stage wetted area against the closed form -------------------------------


@pytest.mark.parametrize("key,prefix", [(2, "s2"), (1, "s1"), (-1, "is"), (0, "st")])
def test_area_wetted_body_matches_closed_form(
    meas: dict[int, StageMeasurements], closed_form: dict[str, dict[str, float]],
    key: int, prefix: str,
) -> None:
    """Requirement 2, per stage, to the stated and justified `AREA_TOLERANCE`.

    The reported number is the LATERAL area everywhere: the notebook subtracts the flat base disc
    on stage 2, BOTH end discs on the booster (its forward disc is covered by the interstage) and
    both end discs on the interstage. That makes it directly comparable with
    `masses_iv1.stage_geometry`'s `area_wetted_body`.

    Key 0 is the Python sum of the three, so it is included here to prove the sum is complete:
    an earlier version left the interstage out of it, and only this check would have caught that.
    """
    ref = closed_form[prefix]["area_wetted_body"]
    got = meas[key].area_wetted_body
    assert got is not None
    err = abs(got - ref) / ref
    assert err < AREA_TOLERANCE, (
        f"{prefix} nTop wetted body {got:.6f} m^2 vs closed form {ref:.6f} m^2, "
        f"error {err * 100:.4f} percent (tolerance {AREA_TOLERANCE * 100:.1f} percent)"
    )


@pytest.mark.parametrize("key,prefix", [(2, "s2"), (1, "s1")])
def test_area_base_is_exact(
    meas: dict[int, StageMeasurements], dv: StackDesignVector, key: int, prefix: str
) -> None:
    """`area_base` is pi/4 d^2 computed INSIDE nTop from the diameter input, so it is exact."""
    got = meas[key].area_base
    assert got is not None
    assert got == pytest.approx(dv.stage_at(key).S_ref, rel=1e-9)


@pytest.mark.parametrize("key,prefix", [(2, "s2"), (1, "s1")])
def test_area_wetted_fins_matches_the_solid_closed_form(
    meas: dict[int, StageMeasurements], closed_form: dict[str, dict[str, float]],
    key: int, prefix: str,
) -> None:
    """The fins are solid plates, so the reference is `fin_solid_area`, not a flat plate."""
    ref = closed_form[prefix]["area_wetted_fins"]
    got = meas[key].area_wetted_fins
    assert got is not None and got > 0.0
    err = abs(got - ref) / ref
    assert err < PLATE_AREA_TOLERANCE, (
        f"{prefix} nTop fin wetted area {got:.6f} m^2 vs solid closed form {ref:.6f} m^2, "
        f"error {err * 100:.3f} percent"
    )


# ---- 3. THE STRAKES ARE ACTUALLY THERE ----------------------------------------------


def test_strakes_are_really_in_the_model(
    meas: dict[int, StageMeasurements], dv: StackDesignVector,
    closed_form: dict[str, dict[str, float]],
) -> None:
    """Requirement 3, part one. A test that passes with no strakes in the model is worthless,
    so this asserts a real number from three independent directions.

    1. The measured strake wetted area matches the SOLID closed form `strake_solid_area`.
    2. It is close to `StrakeSpec.wetted_area`, the zero-thickness aero reference, in a band
       whose centre is the geometrically necessary ratio 1.2724 (an 8 mm plate 30 mm tall is
       27 percent edge; see `test_strake_solid_area_closed_form_arithmetic`).
    3. The measured strake SOLID VOLUME matches `n * h * L * t` = 4*0.030*1.400*0.008 =
       0.001344 m^3, which no amount of body-only geometry could produce.
    """
    st = dv.strakes
    m = meas[2]
    got = m.area_wetted_strakes
    assert got is not None, "no strake area reported at all"

    ref_solid = closed_form["s2"]["area_wetted_strakes"]
    assert ref_solid > 0.0
    err = abs(got - ref_solid) / ref_solid
    assert err < PLATE_AREA_TOLERANCE, (
        f"strake wetted area {got:.6f} m^2 vs solid closed form {ref_solid:.6f} m^2, "
        f"error {err * 100:.3f} percent"
    )

    ratio = got / st.wetted_area
    assert 1.15 < ratio < 1.40, (
        f"strake wetted area {got:.6f} m^2 is {ratio:.3f}x StrakeSpec.wetted_area "
        f"{st.wetted_area:.6f} m^2. The solid plate must be MORE wetted than the "
        f"zero-thickness reference, by the edge fraction 2t(L+h)/(2hL) and no more."
    )

    v = m.volume_strakes
    assert v is not None, "no strake volume reported"
    v_ref = st.n * st.height * st.length * st.thickness
    assert v_ref == pytest.approx(0.001344, abs=1e-9)
    assert v == pytest.approx(v_ref, rel=0.02), (
        f"strake solid volume {v:.6f} m^3 vs n*h*L*t {v_ref:.6f} m^3"
    )


@pytest.fixture(scope="session")
def small_strakes(runner: NtopRunner, dv: StackDesignVector,
                  meas: dict[int, StageMeasurements]) -> dict[int, StageMeasurements]:
    """The same notebook at the minimum strake height, 15 mm. Depends on `meas` so the
    `.ntop` already exists and this costs one run and no convert."""
    small = dv.with_path("strakes.height", 0.015)
    return measure_stack(small, os.path.join(GEOM_DIR, "small_strakes"), runner,
                         tag="iv1_small_strakes")


def test_shrinking_the_strakes_drops_the_strake_area_and_nothing_else(
    meas: dict[int, StageMeasurements], small_strakes: dict[int, StageMeasurements],
    dv: StackDesignVector,
) -> None:
    """Requirement 3, part two: halving the strake height must drop the area accordingly.

    The drop is NOT a factor of two. Halving `height` halves the two side faces but leaves the
    tip face and the root patch untouched, so the predicted ratio is

        (2*0.015*1.4 + 2*0.008*1.4 + 2*0.008*0.015) / (2*0.030*1.4 + 2*0.008*1.4
                                                        + 2*0.008*0.030)
        = 0.064640 / 0.106880 = 0.6048

    Asserting the PREDICTED ratio rather than merely "it went down" is what makes this a real
    check: a model with the strakes silently deleted would report zero, and a model that ignored
    the height input would report no change.
    """
    big, small = meas[2].area_wetted_strakes, small_strakes[2].area_wetted_strakes
    assert big is not None and small is not None
    R = 0.5 * dv.payload_stage.D
    ref_ratio = (strake_solid_area(dv.strakes.__class__(**{**dv.strakes.__dict__,
                                                           "height": 0.015}), R)
                 / strake_solid_area(dv.strakes, R))
    assert ref_ratio == pytest.approx(0.6048, abs=0.005)
    assert small < big
    assert small / big == pytest.approx(ref_ratio, rel=0.02), (
        f"strake area went {big:.6f} -> {small:.6f} m^2, ratio {small / big:.4f}, "
        f"predicted {ref_ratio:.4f}"
    )
    # The body, the fins and the booster are untouched by the strake height.
    assert small_strakes[2].volume_total == pytest.approx(meas[2].volume_total, rel=1e-4)
    assert small_strakes[2].area_wetted_fins == pytest.approx(meas[2].area_wetted_fins,
                                                             rel=1e-3)
    assert small_strakes[1].mass_structure == pytest.approx(meas[1].mass_structure, rel=1e-3)
    # And the stage-2 structure got lighter, because there is less strake to build.
    assert small_strakes[2].mass_structure is not None and meas[2].mass_structure is not None
    assert small_strakes[2].mass_structure < meas[2].mass_structure


# ---- 4. structure mass excludes the motor and the payload ---------------------------


@pytest.mark.parametrize("key,prefix", [(2, "s2"), (1, "s1")])
def test_structure_mass_is_a_hollow_airframe_not_a_billet(
    meas: dict[int, StageMeasurements], closed_form: dict[str, dict[str, float]],
    key: int, prefix: str,
) -> None:
    """Requirement 4, and the single most important check in this file.

    `mass_structure` must be the airframe wall, the bulkheads, the fin panels and (on stage 2)
    the strake panels ONLY. The motor case comes from the motor model, the propellant from
    `StageSpec.m_propellant` and the payload straight from `InterceptRequirements`, so anything
    else in here is double counted, silently and largely.

    A SOLID 7075-T6 billet filling the whole default envelope would be 1200 kg, which is more
    than the entire A8 launch-mass limit. A hollow airframe is tens of kilograms.
    """
    m = meas[key].mass_structure
    v = meas[key].volume_structure
    assert m is not None and v is not None

    billet = sum(closed_form[p]["volume_total"] for p in ("s1", "s2", "is")) * AIRFRAME_DENSITY
    assert billet > 1100.0, "sanity: the whole-envelope billet should be about 1200 kg"
    assert MASS_STRUCTURE_MIN < m < MASS_STRUCTURE_MAX, (
        f"{prefix} structure mass {m:.1f} kg is outside the physical band "
        f"({MASS_STRUCTURE_MIN} to {MASS_STRUCTURE_MAX} kg). A billet of the whole stack would "
        f"be {billet:.0f} kg, so a value near that means the body was never hollowed; a value "
        f"far above {MASS_STRUCTURE_MAX} kg means the motor case, the propellant or the payload "
        f"were double counted into the structure."
    )
    # The density is applied INSIDE nTop by `mass_properties`, not multiplied in Python.
    assert m == pytest.approx(v * AIRFRAME_DENSITY, rel=0.01)

    # Independent closed-form cross-check: a thin-wall shell of the measured lateral area plus
    # the measured plate volumes. This is a factor check, not an equality: it omits the
    # bulkheads and the end plates, both of which the measurement includes.
    stage = meas[key]
    shell = closed_form[prefix]["area_wetted_body"] * _t_wall(prefix)
    plates = (stage.volume_fins or 0.0) + (stage.volume_strakes or 0.0)
    assert 0.75 * (shell + plates) < v < 1.35 * (shell + plates), (
        f"{prefix} structure volume {v:.6f} m^3 vs closed-form shell+plates "
        f"{shell + plates:.6f} m^3"
    )


def _t_wall(prefix: str) -> float:
    d = default_iv1()
    return d.stage_at(int(prefix[1:])).t_wall


def test_interstage_structure_is_a_thin_conical_shell(
    meas: dict[int, StageMeasurements], dv: StackDesignVector,
    closed_form: dict[str, dict[str, float]],
) -> None:
    """The interstage is a 2.5 mm shell, so its mass is kilograms, not tens of kilograms.

    Closed form: lateral area 0.3058688 m^2 times 0.0025 m = 7.647e-4 m^3 for the cone wall,
    plus the two annular end rings that `offset_implicit` leaves behind (it shrinks the flat
    ends as well as the lateral wall), so the measured volume must be somewhat LARGER.
    """
    m = meas[-1]
    assert m.mass_structure is not None and m.volume_structure is not None
    assert MASS_INTERSTAGE_MIN < m.mass_structure < MASS_INTERSTAGE_MAX
    assert m.mass_structure == pytest.approx(m.volume_structure * AIRFRAME_DENSITY, rel=0.01)
    wall_only = closed_form["is"]["area_wetted_body"] * dv.t_interstage
    assert wall_only == pytest.approx(7.647e-4, rel=0.01)
    assert wall_only < m.volume_structure < 2.5 * wall_only
    assert m.volume_structure < closed_form["is"]["volume_total"]


@pytest.mark.parametrize("key", [1, 2, -1])
def test_structure_cg_is_on_the_body_axis_and_inside_its_own_stage(
    meas: dict[int, StageMeasurements], dv: StackDesignVector, key: int
) -> None:
    """A cruciform stage is symmetric about both the XY and the XZ plane.

    The CG is not EXACTLY on the axis: `mass_properties` is an adaptive volume integration to a
    relative-error target, so the first moments carry that error too. WP4 measured a 1.2 mm
    off-axis offset on SV-1 (docs/NTOP_NOTES.md section 20). Test against a tolerance, never
    against zero.

    `cg_structure` is STAGE-LOCAL, because `masses_iv1.build_stack_masses` reads
    `cg_structure[0]` as a station from that stage's own forward face and adds the offset itself.
    `cg_structure_stack` keeps the stack-frame value.
    """
    m = meas[key]
    assert m.cg_structure is not None, "cg_structure must be reported"
    assert m.cg_structure_stack is not None
    x, y, z = m.cg_structure
    length = m.length
    assert length is not None and length > 0.0
    assert 0.25 * length < x < 0.85 * length, (
        f"stage {key} structure CG at x = {x:.3f} m is not plausibly inside a {length:.3f} m body"
    )
    D = dv.stage_at(key).D if key > 0 else dv.booster.D
    assert abs(y) < 0.01 * D, f"CG y = {y:.6f} m should be on the axis"
    assert abs(z) < 0.01 * D, f"CG z = {z:.6f} m should be on the axis"
    # The stack-frame value is the stage-local value plus the stage's forward face, and the
    # forward face comes from nTop rather than from Python arithmetic.
    assert m.x_forward is not None
    assert m.cg_structure_stack[0] == pytest.approx(x + m.x_forward, rel=1e-9)


@pytest.mark.parametrize("key", [1, 2])
def test_structure_inertia_is_reported_and_the_roll_moment_is_smallest(
    meas: dict[int, StageMeasurements], key: int
) -> None:
    """`principal moments` is a vector; its three components travel as three scalars.

    These are PRINCIPAL moments about the CG in the principal frame, NOT Ixx/Iyy/Izz in body
    axes, and nTop does NOT sort them (docs/NTOP_NOTES.md section 20). For a cruciform body of
    revolution the principal axes coincide with the body axes, so the roll moment is much the
    smallest and the two transverse moments are nearly equal.
    """
    inertia = meas[key].inertia_structure
    if inertia is None:                                              # pragma: no cover
        pytest.skip("nTop did not expose principal moments on this build")
    assert all(i > 0.0 for i in inertia)
    roll, mid, large = sorted(inertia)
    assert mid == pytest.approx(large, rel=0.05), (
        f"the two transverse principal moments should be nearly equal, got {inertia}"
    )
    assert roll < 0.25 * mid, (
        f"the roll moment should be much smaller than the transverse ones, got {inertia}"
    )


# ---- 5. per-stage cavity volume closure ---------------------------------------------


@pytest.mark.parametrize("key,prefix", [(2, "s2"), (1, "s1")])
def test_cavity_is_positive_smaller_than_the_body_and_holds_the_propellant(
    meas: dict[int, StageMeasurements], dv: StackDesignVector,
    closed_form: dict[str, dict[str, float]], key: int, prefix: str,
) -> None:
    """Requirement 5, per stage."""
    m = meas[key]
    v_cav, v_tot = m.volume_cavity, m.volume_total
    assert v_cav is not None and v_tot is not None
    assert v_cav > 0.0
    assert v_cav < v_tot, "the cavity cannot be larger than the volume that encloses it"

    v_prop = dv.stage_at(key).m_propellant / PROPELLANT_DENSITY
    assert v_cav >= v_prop, (
        f"{prefix} cavity {v_cav:.4f} m^3 cannot hold "
        f"{dv.stage_at(key).m_propellant:.0f} kg of propellant at "
        f"{PROPELLANT_DENSITY:.0f} kg/m^3, which needs {v_prop:.4f} m^3"
    )

    # EXACT closure, and this one is strict because it links four separate nTop measurements:
    #   cavity + structure  ==  outer mould line + the plate volume outside it.
    v_str = m.volume_structure
    assert v_str is not None
    plates = (m.volume_fins or 0.0) + (m.volume_strakes or 0.0)
    assert v_cav + v_str == pytest.approx(v_tot + plates, rel=0.01), (
        f"{prefix}: cavity {v_cav:.6f} + structure {v_str:.6f} = {v_cav + v_str:.6f} m^3, "
        f"against OML {v_tot:.6f} + exposed plates {plates:.6f} = {v_tot + plates:.6f} m^3"
    )


def test_the_default_stack_is_NOT_volume_closed_on_stage_2(
    meas: dict[int, StageMeasurements], dv: StackDesignVector
) -> None:
    """A real finding, locked into the suite so it cannot quietly disappear. CLAUDE.md 3.1, 3.3.

    `default_iv1()` says of itself "not sized; the sizer moves it", and the measured geometry now
    says exactly HOW it has to move. The stage-2 cavity holds the propellant comfortably but NOT
    the propellant plus the payload:

        propellant  150 kg / 1800 kg/m^3           = 0.083333 m^3
        payload      75 kg / 1500 kg/m^3           = 0.050000 m^3   (masses_iv1.RHO_PAYLOAD)
                                                     ----------
                                          needed     0.133333 m^3
                        nTop-measured stage-2 cavity  0.131721 m^3
                                                     ----------
                                       shortfall      0.001612 m^3, i.e. 1.2 percent

    That is the volume-closure constraint in SPEC_IV1 section 6 refusing to close, and it is a
    USEFUL output rather than a bug: it is small, so the fix is small. The stage-2 cavity has a
    cross-section of pi (R - t_wall)^2 = pi * 0.1374^2 = 0.059309 m^2, so 0.001612 m^3 is
    0.027 m of extra length. Lengthening stage 2 from 2.70 m to 2.73 m closes it, and 2.73 m is
    still inside the SPEC_IV1 bound of 3.4 m and leaves the stacked length at 5.11 m against the
    A7 limit of 5.40 m. Reducing the propellant by 3 kg would also do it.

    This test asserts the shortfall, so if a later change to the default stack, the wall thickness
    or the bay lengths closes the volume, the test fails and someone has to update the finding
    deliberately instead of the finding evaporating.
    """
    # Stated locally rather than imported, because `rocketgen/sizing/masses_iv1.py` is owned by
    # another work package. The value it uses today is 1500 kg/m^3.
    rho_payload = 1500.0
    reqs = InterceptRequirements()
    v_cav = meas[2].volume_cavity
    assert v_cav is not None

    v_prop = dv.payload_stage.m_propellant / PROPELLANT_DENSITY
    v_payload = reqs.m_payload / rho_payload
    needed = v_prop + v_payload
    assert v_prop == pytest.approx(0.083333, abs=1e-5)
    assert needed == pytest.approx(0.133333, abs=1e-5)

    assert v_cav < needed, (
        "the default stage-2 cavity now HOLDS the propellant plus the payload. Good, but the "
        "finding recorded in this test's docstring is stale: update it with the new numbers "
        f"(cavity {v_cav:.6f} m^3 against a need of {needed:.6f} m^3)."
    )
    shortfall = needed - v_cav
    assert shortfall == pytest.approx(0.001612, abs=2.0e-4), (
        f"the shortfall moved to {shortfall:.6f} m^3; update the finding"
    )
    # And the fix really is 27 mm of stage-2 length, not a redesign.
    area = math.pi * (0.5 * dv.payload_stage.D - dv.payload_stage.t_wall) ** 2
    assert area == pytest.approx(0.059309, abs=1e-5)
    dL = shortfall / area
    assert 0.015 < dL < 0.045, f"the fix is {dL * 1e3:.0f} mm of stage-2 length"
    assert dv.payload_stage.L + dL < dv.bounds()["stages.1.L"][1]


# ---- 6. one notebook, two design vectors -------------------------------------------


@pytest.fixture(scope="session")
def alternate(runner: NtopRunner, meas: dict[int, StageMeasurements]) -> dict[
        int, StageMeasurements]:
    """A second, larger design point. Depends on `meas` so the notebook already exists."""
    dv2 = (default_iv1()
           .with_path("stages.0.D", 0.42).with_path("stages.0.L", 2.40)
           .with_path("stages.1.D", 0.32).with_path("stages.1.L", 3.00))
    ok, errs = dv2.geometry_is_valid()
    assert ok, errs
    return measure_stack(dv2, os.path.join(GEOM_DIR, "alternate"), runner, tag="iv1_alt")


def test_the_same_notebook_serves_two_design_vectors(
    meas: dict[int, StageMeasurements], alternate: dict[int, StageMeasurements]
) -> None:
    """Requirement 6. The design variables are real nTop inputs, so ONE `.ntop` covers both.

    This is the caching gate: if the dimensions had been baked in as literals, the second
    measurement would either equal the first or need its own `convert`. The check is not merely
    "it got bigger": the measured RATIO has to track the closed-form ratio.
    """
    assert meas[0].ntop_path == alternate[0].ntop_path, (
        "the two design points must run through the SAME cached notebook"
    )
    dv2 = (default_iv1()
           .with_path("stages.0.D", 0.42).with_path("stages.0.L", 2.40)
           .with_path("stages.1.D", 0.32).with_path("stages.1.L", 3.00))
    cf1 = stack_geometry_closed_form(default_iv1())
    cf2 = stack_geometry_closed_form(dv2)

    for key, prefix in ((2, "s2"), (1, "s1"), (-1, "is")):
        a, b = meas[key].volume_total, alternate[key].volume_total
        assert a is not None and b is not None
        assert b > a, f"{prefix} is bigger in every dimension, so its volume must rise"
        ref_ratio = cf2[prefix]["volume_total"] / cf1[prefix]["volume_total"]
        assert b / a == pytest.approx(ref_ratio, rel=0.02), (
            f"{prefix} measured volume ratio {b / a:.4f} vs closed form {ref_ratio:.4f}"
        )
        # And the absolute value is still right at the new point, not merely proportional.
        err = abs(b - cf2[prefix]["volume_total"]) / cf2[prefix]["volume_total"]
        assert err < VOLUME_TOLERANCE, f"{prefix} at the alternate point: {err * 100:.3f} percent"

    for key, prefix in ((2, "s2"), (1, "s1")):
        a, b = meas[key].area_wetted_body, alternate[key].area_wetted_body
        assert a is not None and b is not None and b > a
        err = abs(b - cf2[prefix]["area_wetted_body"]) / cf2[prefix]["area_wetted_body"]
        assert err < AREA_TOLERANCE, f"{prefix} area at the alternate point: {err * 100:.3f} pc"

    # The strakes did not change, but the body they sit on got fatter, so the root patch grew a
    # little. The area must therefore be nearly, but not exactly, unchanged.
    s_a, s_b = meas[2].area_wetted_strakes, alternate[2].area_wetted_strakes
    assert s_a is not None and s_b is not None
    assert s_b == pytest.approx(s_a, rel=0.02)


def test_notebook_is_reused_not_reconverted(
    runner: NtopRunner, meas: dict[int, StageMeasurements]
) -> None:
    """A third topology-identical call must reuse the cached `.ntop` rather than convert again."""
    path = build_stack_notebook(
        default_iv1().with_path("strakes.length", 1.10), GEOM_DIR, runner
    )
    assert path == meas[0].ntop_path
    assert os.path.isfile(path)


# ---- area distribution S(x) ---------------------------------------------------------


def test_area_distribution_is_empty_by_default(meas: dict[int, StageMeasurements]) -> None:
    """`area_stations = 0` by default, and the omission is DECLARED in the warnings.

    `rocketgen/sizing/aero_iv1.py` can fall back to closed-form cross-section geometry, so the
    default costs nothing but a warning. CLAUDE.md 3.3: the omission is recorded, not hidden.
    """
    assert meas[0].area_distribution == []
    assert any("area_distribution is empty" in w for w in meas[0].warnings)


def test_area_distribution_reproduces_the_closed_form_and_sees_the_strakes(
    runner: NtopRunner, dv: StackDesignVector, meas: dict[int, StageMeasurements]
) -> None:
    """S(x) for wave drag, measured on the STACK rather than assumed.

    There is no single block for a cross-section area (docs/NTOP_NOTES.md section 24). The route
    is `extract_section<implicit,plane,real>` then `body_surface_area<implicit_2d,real>[1.1.0]`,
    the block the vendored universe wrongly marks deprecated.

    The reference is the closed-form radius of whichever body the station falls in: ogive nose,
    stage-2 cylinder, interstage cone, or booster cylinder. It is entirely independent of nTop.

    Six stations are enough to hit all four regions AND to prove the strakes are in the section:
    where a station lands between the strake leading and trailing edges the measured area must
    exceed the bare body by `n * height * thickness` = 4 * 0.030 * 0.008 = 0.000960 m^2.
    A model with the strakes missing would match the bare body everywhere.

    This costs its own `convert`, because `area_stations` changes the block graph. It depends on
    `meas` only to keep the nTop calls serialised.
    """
    assert meas[0].volume_total is not None                # ordering guard, see docstring
    n = 6
    got = measure_stack(dv, os.path.join(GEOM_DIR, "area_distribution"), runner,
                        tag="iv1_sx", area_stations=n, timeout=3600.0, convert_timeout=3600.0)
    sd = got[0].area_distribution
    assert len(sd) == n
    stations = [x for x, _ in sd]
    assert stations == sorted(stations)
    assert all(0.0 < x < dv.L_total for x in stations)
    assert not any("area_distribution is empty" in w for w in got[0].warnings)

    R2, R1 = 0.5 * dv.payload_stage.D, 0.5 * dv.booster.D
    L2, L_is, L_nose = dv.payload_stage.L, dv.L_interstage, dv.L_nose
    rho = (R2 * R2 + L_nose * L_nose) / (2.0 * R2)          # tangent-ogive generating radius
    st = dv.strakes
    n_strake_hits = 0
    for x, area in sd:
        if x <= L_nose:
            y = max(math.sqrt(max(rho * rho - (L_nose - x) ** 2, 0.0)) - (rho - R2), 0.0)
        elif x <= L2:
            y = R2
        elif x <= L2 + L_is:
            y = R2 + (x - L2) / L_is * (R1 - R2)            # the interstage cone
        else:
            y = R1
        ref = math.pi * y * y
        # Plates present at this station add their own rectangular slices.
        if st.x_le <= x <= st.x_le + st.length:
            ref += st.n * st.height * st.thickness
            n_strake_hits += 1
        for stage, x_aft, R in ((dv.payload_stage, L2, R2),
                                (dv.booster, dv.L_total, R1)):
            if x_aft - stage.c_r_fin <= x <= x_aft:
                ref += stage.n_fin * stage.b_fin * stage.t_fin
        assert area == pytest.approx(ref, rel=0.02), (
            f"S(x) at x = {x:.3f} m: nTop {area:.6f} m^2 vs closed form {ref:.6f} m^2"
        )
    assert n_strake_hits >= 2, "the station set must cross the strakes to prove they are there"
    # The largest section is on the booster, which is the widest body.
    assert max(a for _, a in sd) == pytest.approx(math.pi * R1 * R1, rel=0.02)


# ---- exports and the render ---------------------------------------------------------


def test_measurements_write_no_exports_by_default(meas: dict[int, StageMeasurements]) -> None:
    """Every export is off by default, because they are measured to be the expensive part.

    The measurement blocks alone cost about 118 s per run. Adding the STL means meshing a
    5.08 x 0.80 x 0.80 m box, which is 1.6x the SV-1 box, on a block whose cost goes as
    tolerance^-3. The sizing loop needs none of it, and nothing measured comes off the mesh.
    """
    for m in meas.values():
        assert m.stl_path is None
        assert m.step_path is None
        assert m.implicit_path is None


def test_exported_stl_shows_the_whole_vehicle(dv: StackDesignVector) -> None:
    """The deliverable STL, checked for the things a reader has to be able to see.

    Produced by `runs/IV-1_geom/_make_artefacts.py`, not by this test, because the export run
    costs several minutes and the sizing loop never needs it. Skipped, with instructions, when
    it has not been produced.
    """
    stl = os.path.join(GEOM_DIR, "exports", "iv1.stl")
    if not (os.path.isfile(stl) and os.path.getsize(stl) > 0):
        pytest.skip(f"{stl} not present; run runs/IV-1_geom/_make_artefacts.py")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from stlreader import bounding_box, enclosed_volume, read_stl

    tris = read_stl(stl)
    assert len(tris) > 10000
    lo, hi = bounding_box(tris)
    # Exported in metres. The stack spans L_total in x.
    assert hi[0] - lo[0] == pytest.approx(dv.L_total, rel=0.02)
    # The widest thing on the vehicle is the booster fin span, tip to tip.
    span = dv.booster.D + 2.0 * dv.booster.b_fin
    assert hi[1] - lo[1] == pytest.approx(span, rel=0.06), "the fins must survive the meshing"
    assert hi[2] - lo[2] == pytest.approx(span, rel=0.06)
    # A 5 mm-tolerance mesh of a 5 m vehicle is a consistency band, never a measurement
    # (docs/NTOP_NOTES.md section 4: the notebook beat the STL by 16x on the smoke sphere).
    cf = stack_geometry_closed_form(dv)
    assert 0.93 * cf["st"]["volume_total"] < enclosed_volume(tris) < 1.10 * \
        cf["st"]["volume_total"]


def test_render_exists_if_it_was_produced() -> None:
    png = os.path.join(GEOM_DIR, "iv1_iso.png")
    if not os.path.isfile(png):
        pytest.skip(f"{png} not present; run runs/IV-1_geom/_make_artefacts.py")
    assert os.path.getsize(png) > 20000, "the render is suspiciously small"


# ---- bookkeeping --------------------------------------------------------------------


def test_measurements_carry_their_bookkeeping(meas: dict[int, StageMeasurements]) -> None:
    for key, m in meas.items():
        assert m.wall_time_s is not None and m.wall_time_s > 0.0
        assert m.ntopcl_returncode in (0, 72)
        assert m.body, f"key {key} has no body label"
    assert os.path.isfile(os.path.join(GEOM_DIR, "baseline", "iv1_measurements.json"))
    assert os.path.isfile(os.path.join(GEOM_DIR, "baseline", "iv1_stages.json"))


def test_measured_wall_time_is_reported(meas: dict[int, StageMeasurements]) -> None:
    """Not a pass/fail gate: the number is printed so the sizing loop can budget for it.

    The band is wide on purpose. It exists only to catch an order-of-magnitude regression, for
    instance an export accidentally left on or a mesh block sneaking into the measurement path.
    """
    t = meas[0].wall_time_s
    assert t is not None
    print(
        f"\nmeasure_stack wall time: {t:.1f} s (measured reference "
        f"{MEASURED_WALL_TIME_S:.0f} s) at mesh tolerance {DEFAULT_MESH_TOLERANCE:.1e} m, "
        f"exports off. Four surface_area<implicit,real> calls dominate."
    )
    assert t < 8.0 * MEASURED_WALL_TIME_S, (
        f"one measure_stack call took {t:.0f} s against a measured reference of "
        f"{MEASURED_WALL_TIME_S:.0f} s. Something expensive was turned on."
    )
