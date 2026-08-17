"""WP1 tests: the block universe, the recipe builder, and a real end-to-end nTop run.

The end-to-end part is not mocked. It authors a notebook, runs `ntopcl convert`, `ntopcl -t`
and `ntopcl -j/-o` for real, then checks the exported STL against the analytic sphere volume.
All artefacts land in `runs/_smoke/` so they can be inspected afterwards; pytest's tmpdir is
deliberately not used for them.

Run: `.venv/Scripts/python.exe -m pytest tests/test_ntopgen.py -q`
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rocketgen.config import RUNS_DIR                              # noqa: E402
from rocketgen.ntopgen import (                                    # noqa: E402
    BLOCK_REVISION_OVERRIDES,
    ArityError,
    LiteralTypeError,
    NtopError,
    NtopRunner,
    Recipe,
    UnknownFunctionError,
    Universe,
    parse_revision,
    split_signature,
    to_ntop_path,
)
from rocketgen.ntopgen.universe import universe_from_entries       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stlreader import bounding_box, enclosed_volume, read_stl       # noqa: E402

# --------------------------------------------------------------------------------------
#   Smoke-case constants
# --------------------------------------------------------------------------------------

SMOKE_DIR = os.path.join(RUNS_DIR, "_smoke")

SPHERE_RADIUS_M = 0.025          # 25 mm, the radius the notebook is run with
SPHERE_DENSITY = 1000.0          # kg/m^3, so mass in kg equals volume in litres
VOLUME_TOLERANCE = 0.01          # 1 percent, the WP1 definition of done in PLAN.md

# `implicit_to_mesh` drives a voxel grid, so its cost scales roughly as tolerance^-3.
# Measured on this machine for a 25 mm sphere: 1.0e-3 m converts in ~3 s and gives 0.17 percent
# volume error; 1.0e-4 m had not finished after 195 s and had grown past 2.4 GB resident.
# 1.0e-3 m is therefore the working point: comfortably inside the 1 percent gate and fast.
MESH_TOLERANCE_M = 1.0e-3

ANALYTIC_VOLUME = 4.0 / 3.0 * math.pi * SPHERE_RADIUS_M ** 3


def analytic_sphere_volume(radius: float) -> float:
    return 4.0 / 3.0 * math.pi * radius ** 3


# --------------------------------------------------------------------------------------
#   1. Universe
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def uni() -> Universe:
    return Universe.load()


def test_universe_loads(uni: Universe) -> None:
    # REFERENCE.md section 4: the vendored universe holds 853 blocks.
    assert len(uni) == 853
    assert Universe.load() is uni, "Universe.load() must be a cached singleton"
    assert "sphere<point,real>" in uni
    assert uni.get("sphere<point,real>").displayname == "Sphere"


def test_universe_get_reports_near_misses(uni: Universe) -> None:
    with pytest.raises(UnknownFunctionError) as e1:
        uni.get("sphere<point,reals>")
    assert "sphere<point,real>" in str(e1.value)

    with pytest.raises(UnknownFunctionError) as e2:
        uni.get("implicit_to_mesh<implicit,real,real,bool,bool>[9.9.9]")
    assert "[2.4.0]" in str(e2.value)

    with pytest.raises(UnknownFunctionError) as e3:
        uni.get("spere<point,real>")
    assert "did you mean" in str(e3.value)


def test_parse_revision_and_split() -> None:
    assert parse_revision("export_mesh<file_path,mesh,unit_length_enum>") == (1, 0, 0)
    assert parse_revision("loft<implicit_2d,implicit_2d>[1.1.0]") == (1, 1, 0)
    assert split_signature("loft<implicit_2d,implicit_2d>[1.1.0]") == (
        "loft<implicit_2d,implicit_2d>",
        (1, 1, 0),
    )


def test_latest_on_vendored_blocks(uni: Universe) -> None:
    assert (
        uni.latest("implicit_to_mesh<implicit,real,real,bool,bool>")
        == "implicit_to_mesh<implicit,real,real,bool,bool>[2.4.0]"
    )
    # A base signature given WITH a revision must still resolve to the newest one.
    assert (
        uni.latest("implicit_to_mesh<implicit,real,real,bool,bool>[2.0.0]")
        == "implicit_to_mesh<implicit,real,real,bool,bool>[2.4.0]"
    )
    # Blocks with no bracketed revision come back unchanged.
    assert (
        uni.latest("export_mesh<file_path,mesh,unit_length_enum>")
        == "export_mesh<file_path,mesh,unit_length_enum>"
    )
    assert uni.latest("mass_properties<implicit,real_field,real>") == \
        "mass_properties<implicit,real_field,real>[1.1.0]"
    assert uni.latest("surface_area<implicit,real>") == "surface_area<implicit,real>[1.2.0]"


def test_latest_sorts_revisions_numerically_not_lexically() -> None:
    """Synthetic ids prove the sort is numeric: 2.10.0 beats 2.4.0, and 2.4.0 beats 2.10.0
    lexically, so a string sort would pick the wrong one."""
    sig = "fake_block<real>"
    entries = [
        {"function": f"{sig}[2.4.0]", "inputs": [], "output": {"type": "real"}},
        {"function": f"{sig}[2.10.0]", "inputs": [], "output": {"type": "real"}},
        {"function": f"{sig}[2.9.0]", "inputs": [], "output": {"type": "real"}},
        {"function": f"{sig}[10.0.0]", "inputs": [], "output": {"type": "real"}},
        {"function": sig, "inputs": [], "output": {"type": "real"}},
    ]
    u = universe_from_entries(entries)
    assert u.latest(sig) == f"{sig}[10.0.0]"
    # A plain lexical max would pick 2.9.0 out of the bracketed suffixes.
    assert max(f"{sig}[2.4.0]", f"{sig}[2.10.0]", f"{sig}[2.9.0]") == f"{sig}[2.9.0]"
    assert [d.func_id for d in u.revisions(sig)] == [
        sig,
        f"{sig}[2.4.0]",
        f"{sig}[2.9.0]",
        f"{sig}[2.10.0]",
        f"{sig}[10.0.0]",
    ]
    # And with the newest ones deprecated, latest() falls back to the newest live revision.
    entries2 = [dict(e) for e in entries]
    for e in entries2:
        if e["function"] in (f"{sig}[10.0.0]", f"{sig}[2.10.0]"):
            e["deprecated"] = {"message": "no"}
    u2 = universe_from_entries(entries2)
    assert u2.latest(sig) == f"{sig}[2.9.0]"
    assert u2.latest(sig, include_deprecated=True) == f"{sig}[10.0.0]"


def test_dep_resolution(uni: Universe) -> None:
    """REFERENCE.md section 4: input `dep`, integer `unitsReq`, and output `dep`."""
    # `core.if<bool,any,1>` input 2 has {"dep": 1} and unitsReq 1; its output has dep 1.
    assert uni.resolve_input_types("core.if<bool,any,1>") == ["bool", "any", "any"]
    assert uni.resolve_return_type("core.if<bool,any,1>") == "any"
    assert uni.resolve_units_req("core.if<bool,any,1>") == [{}, {}, {}]
    # `add<real,real>` operand B has unitsReq 0, i.e. "same as operand A".
    assert uni.resolve_units_req("add<real,real>") == [{}, {}]
    # A concrete dimension map survives.
    assert uni.resolve_units_req("sphere<point,real>") == [{}, {"length": 1}]
    assert uni.resolve_units_req("mass_properties<implicit,real_field,real>[1.1.0]")[1] == {
        "length": -3,
        "mass": 1,
    }


def test_input_names_and_index(uni: Universe) -> None:
    assert uni.input_index("sphere<point,real>", "Radius") == 1
    assert uni.input_index("sphere<point,real>", "radius") == 1
    desc = uni.get("implicit_to_mesh<implicit,real,real,bool,bool>[2.4.0]")
    assert desc.input_names == (
        "Body", "Tolerance", "Min. feature size", "Sharpen", "Simplify",
    )
    assert desc.input("Tolerance").units_req == {"length": 1}
    assert desc.input("Min. feature size").is_optional
    assert desc.input(0).description
    with pytest.raises(KeyError):
        desc.input_index("Nope")


def test_find(uni: Universe) -> None:
    exports = uni.find(name_fragment="export_mesh")
    assert [d.func_id for d in exports] == ["export_mesh<file_path,mesh,unit_length_enum>"]
    meshers = uni.find(returns="mesh", include_deprecated=False)
    assert any(d.func_id.startswith("implicit_to_mesh") for d in meshers)
    assert all(d.deprecated is None for d in meshers)
    assert uni.find(displayname="Sphere")


def test_type_properties(uni: Universe) -> None:
    props = uni.properties("body_mass_props")
    assert props["volume"] == "real"
    assert props["center of gravity"] == "point"
    assert props["principal moments"] == "vector"
    assert uni.property_type("mesh", "face count") == "integer"


# --------------------------------------------------------------------------------------
#   2. Recipe builder, pure Python
# --------------------------------------------------------------------------------------


def test_literal_encodings() -> None:
    r = Recipe(name="lit")
    assert r.literal_real(0.01, {"length": 1}).render() == {
        "type": "real",
        "value": {"isFinite": True, "units": {"length": 1}, "val": 0.01},
    }
    assert r.literal_integer(3).render() == {"type": "integer", "value": {"val": 3}}
    assert r.literal_bool(True).render() == {"type": "bool", "value": {"val": True}}
    assert r.literal_text("A-CAD.step").render() == {
        "type": "text", "value": {"string": "A-CAD.step"},
    }
    assert r.literal_file_path(r"C:\a\b\name").render() == {
        "type": "file_path", "value": {"val": "C:/a/b/name"},
    }
    assert r.literal_point(0.0, 1.0, 2.0).render() == {
        "type": "point",
        "value": [
            {"isFinite": True, "val": 0.0},
            {"isFinite": True, "val": 1.0},
            {"isFinite": True, "val": 2.0},
        ],
    }
    assert r.literal_real_field("(3*x + 2*y - z)/1mm").render() == {
        "type": "real_field", "value": {"expression": "(3*x + 2*y - z)/1mm", "units": {}},
    }
    assert r.literal_enum("blend_enum", 0).render() == {
        "type": "blend_enum", "value": {"enum": 0},
    }
    assert r.literal_unit_length("mm").render() == {
        "type": "unit_length_enum", "value": {"id": "mm"},
    }
    assert to_ntop_path(r"D:\x\y.stl") == "D:/x/y.stl"


def test_block_promotes_scalars_with_required_units() -> None:
    r = Recipe(name="promote")
    p = r.block("point<real,real,real>", 0.0, 0.1, 0.2, name="P")
    doc = r.to_dict()
    args = doc["body"][0]["contents"]["inputs"]
    # `point<real,real,real>` inputs all require {"length": 1}, so the literals carry it.
    assert all(a["value"]["units"] == {"length": 1} for a in args)
    assert [a["value"]["val"] for a in args] == [0.0, 0.1, 0.2]
    assert p.type == "point"


def test_block_arity_and_type_errors() -> None:
    r = Recipe(name="errs")
    p = r.block("point<real,real,real>", 0.0, 0.0, 0.0)
    with pytest.raises(ArityError, match="takes 2 input"):
        r.block("sphere<point,real>", p, 0.01, 0.02)
    with pytest.raises(ArityError, match="not filled"):
        r.block("sphere<point,real>", p)
    with pytest.raises(LiteralTypeError, match="real slot"):
        r.block("sphere<point,real>", p, "big")
    with pytest.raises(LiteralTypeError, match="integer slot"):
        r.block("text_from_scalar<real,real,integer>", 1.0, 1.0, 2.5)
    with pytest.raises(UnknownFunctionError):
        r.block("no_such_block<real>", 1.0)
    # A whole-valued float is accepted for an integer slot.
    r.block("text_from_scalar<real,real,integer>", 1.0, 1.0, 3.0)
    # A 3-sequence is accepted for a point slot.
    r.block("sphere<point,real>", (0.0, 0.0, 0.0), 0.01)
    # A type with no literal form must say so rather than emit nonsense.
    with pytest.raises(LiteralTypeError, match="no literal form"):
        r.block("mass_properties<implicit,real_field,real>[1.1.0]", 1.0, 1000.0, 0.01)


def test_block_keyword_arguments_by_input_name() -> None:
    r = Recipe(name="kw")
    p = r.block("point<real,real,real>", 0.0, 0.0, 0.0)
    a = r.block("sphere<point,real>", p, radius=0.02, name="A")
    b = r.block("sphere<point,real>", center_point=p, radius=0.02, name="B")
    doc = r.to_dict()
    ia = doc["body"][1]["contents"]["inputs"]
    ib = doc["body"][2]["contents"]["inputs"]
    assert ia == ib
    assert ia[1]["value"]["val"] == 0.02
    assert a.type == b.type == "sphere"
    with pytest.raises(ArityError, match="both positionally and by keyword"):
        r.block("sphere<point,real>", p, 0.02, radius=0.03)
    with pytest.raises(ArityError, match="no input matching keyword"):
        r.block("sphere<point,real>", p, diameter=0.02)


def test_unique_instance_ids() -> None:
    r = Recipe(name="ids")
    p = r.block("point<real,real,real>", 0.0, 0.0, 0.0)
    for _ in range(20):
        r.block("sphere<point,real>", p, 0.01)
    doc = r.to_dict()
    ids: list[str] = []
    for entry in doc["body"]:
        ids.append(entry["id"])
        contents = entry.get("contents")
        if isinstance(contents, dict) and "id" in contents:
            ids.append(contents["id"])
    assert len(ids) == len(set(ids)), "instance ids must be unique"
    assert all(i.startswith("inst") for i in ids)


def test_point_list_uses_blocks_not_literal_leaves() -> None:
    """REFERENCE.md section 3: literal `list<point>` LEAVES are dropped by `exportjson`."""
    r = Recipe(name="plist")
    pl = r.point_list([(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.1, 0.2, 0.0)], name="Pts")
    assert pl.type == "list<point>"
    doc = r.to_dict()
    lists = [e for e in doc["body"] if e["contents"].get("func", "").startswith("core.list<")]
    assert len(lists) == 1
    entries = lists[0]["contents"]["inputs"]
    assert len(entries) == 3
    for e in entries:
        assert "ref" in e, "list<point> elements must reference point blocks, not be literals"
    point_blocks = [e for e in doc["body"]
                    if e["contents"].get("func") == "point<real,real,real>"]
    assert len(point_blocks) == 3


def test_list_of_scalars() -> None:
    r = Recipe(name="lst")
    lst = r.list_of("real", [1.0, 2.0, 3.0], name="Nums")
    assert lst.type == "list<real>"
    doc = r.to_dict()
    blk = doc["body"][0]["contents"]
    assert blk["func"] == "core.list<real>"
    assert [i["value"]["val"] for i in blk["inputs"]] == [1.0, 2.0, 3.0]


def test_prop_chain_types(uni: Universe) -> None:
    r = Recipe(name="props")
    p = r.block("point<real,real,real>", 0.0, 0.0, 0.0)
    ball = r.block("sphere<point,real>", p, 0.02)
    mp = r.mass_properties(ball, density=1000.0)
    assert mp.type == "body_mass_props"
    assert mp.prop("volume").type == "real"
    assert mp.prop("center of gravity").type == "point"
    assert mp.prop("center of gravity").render() == {
        "props": ["center of gravity"], "ref": {"id": mp.block.var_id},
    }
    assert mp.prop("list", "[0]").type == "body_mass_props"


def test_notebook_input_reference_shape() -> None:
    r = Recipe(name="inp")
    a = r.add_input("Span", "real", default=3.048, dimension={"length": 1})
    b = r.add_input("Sweep", "real", default=0.8115781021773633, dimension={"angle": 1})
    assert a.render() == {"input": 0, "props": []}
    assert b.render() == {"input": 1, "props": []}
    doc = r.to_dict()
    assert doc["inputs"][0] == {
        "description": "",
        "name": "Span",
        "type": "real",
        "dimension": {"length": 1},
        "contents": {
            "type": "real",
            "value": {"isFinite": True, "units": {"length": 1}, "val": 3.048},
        },
    }


def test_recipe_top_level_keys() -> None:
    r = Recipe(name="keys", displayname="Keys", description="d")
    p = r.block("point<real,real,real>", 0.0, 0.0, 0.0)
    r.set_output(r.variable("Out", p))
    doc = r.to_dict()
    assert set(doc) == {
        "body", "cbRefs", "description", "displayname", "imports", "inputs", "name",
        "namespaces", "version", "output",
    }
    assert doc["version"] == [1, 0, 0]
    assert doc["name"].startswith("user_func_")
    assert len(doc["name"].split("_")) == 7        # user, func, 8, 4, 4, 4, 12
    assert doc["output"] == {"id": doc["body"][0]["id"]}


def test_variable_renames_block_in_place() -> None:
    r = Recipe(name="var")
    p = r.block("point<real,real,real>", 0.0, 0.0, 0.0, name="P")
    v = r.variable("Body Origin", p)
    assert v is p
    doc = r.to_dict()
    assert len(doc["body"]) == 1
    assert doc["body"][0]["name"] == "Body Origin"


def test_raw_block_escape_hatch() -> None:
    r = Recipe(name="raw")
    t = r.raw_block("table_from_columns<list<column>>", "table", [None], name="T")
    assert t.type == "table"
    doc = r.to_dict()
    assert doc["body"][0]["contents"]["func"] == "table_from_columns<list<column>>"


def test_convenience_emitters_pick_latest_revision(uni: Universe) -> None:
    r = Recipe(name="emit")
    p = r.block("point<real,real,real>", 0.0, 0.0, 0.0)
    ball = r.block("sphere<point,real>", p, 0.02)
    mesh = r.mesh_from_implicit(ball, tolerance=1.0e-3)
    r.export_stl(mesh, "C:/tmp/x", units="mm")
    r.export_implicit(ball, "C:/tmp/x")
    r.surface_area(ball)
    r.mass_properties(ball, 1000.0)
    funcs = {e["contents"].get("func") for e in r.to_dict()["body"]}
    assert r.latest("implicit_to_mesh<implicit,real,real,bool,bool>") in funcs
    assert r.latest("export_mesh<file_path,mesh,unit_length_enum>") in funcs
    assert r.latest("export_implicit_body<file_path,implicit>") in funcs
    assert r.latest("surface_area<implicit,real>") in funcs
    assert r.latest("mass_properties<implicit,real_field,real>") in funcs
    # No deprecated block may sneak in through a convenience emitter.
    for f in funcs:
        if f and f in uni:
            assert uni.get(f).deprecated is None, f


def test_revision_overrides_beat_the_vendored_universe(uni: Universe) -> None:
    """The vendored universe is stale: ntopcl has `implicit_to_mesh[2.5.0]` and calls 2.4.0
    deprecated. `Recipe.latest` must prefer the override, and `describe` must still be able to
    arity-check a revision the universe has never heard of."""
    base = "implicit_to_mesh<implicit,real,real,bool,bool>"
    r = Recipe(name="override")
    assert uni.latest(base) == f"{base}[2.4.0]"
    assert r.latest(base) == BLOCK_REVISION_OVERRIDES[base] == f"{base}[2.5.0]"
    desc = r.describe(f"{base}[2.5.0]")
    assert desc.func_id == f"{base}[2.5.0]"
    assert desc.revision == (2, 5, 0)
    assert desc.input_names == uni.get(f"{base}[2.4.0]").input_names
    # An unknown base signature must still fail, revision suffix or not.
    with pytest.raises(UnknownFunctionError):
        r.describe("not_a_block<real>[1.0.0]")
    # Blocks with no override fall through to the universe.
    assert r.latest("surface_area<implicit,real>") == uni.latest("surface_area<implicit,real>")


def test_mesh_and_text_and_table_emitters(uni: Universe) -> None:
    r = Recipe(name="more")
    p = r.block("point<real,real,real>", 0.0, 0.0, 0.0)
    ball = r.block("sphere<point,real>", p, 0.02)
    mesh = r.mesh_from_implicit(ball, tolerance=1.0e-3)
    # mass_properties and surface_area both take the mesh overload for a mesh input.
    mp_mesh = r.mass_properties(mesh, density=1000.0, name="Mesh Mass Props")
    area_mesh = r.surface_area(mesh, name="Mesh Area")
    text = r.variable("Report", r.literal_text("hello"), type="text")
    r.export_text(text, "C:/tmp/report")
    table = r.raw_block("table_from_columns<list<column>>", "table", [None], name="T")
    r.export_table(table, "C:/tmp/table")
    funcs = [e["contents"].get("func") for e in r.to_dict()["body"]]
    assert uni.latest("mass_properties<mesh,real>") in funcs
    assert uni.latest("surface_area<mesh>") in funcs
    assert uni.latest("export_text<file_path,text>") in funcs
    assert uni.latest("export_table<file_path,table>") in funcs
    assert mp_mesh.type == "body_mass_props"
    assert area_mesh.type == "real"
    paths = [e["contents"]["inputs"][0]["value"]["val"]
             for e in r.to_dict()["body"]
             if (e["contents"].get("func") or "").startswith(("export_text", "export_table"))]
    assert sorted(paths) == ["C:/tmp/report.txt", "C:/tmp/table.csv"]


def test_export_stl_meshes_an_implicit_target() -> None:
    r = Recipe(name="autoexport")
    p = r.block("point<real,real,real>", 0.0, 0.0, 0.0)
    ball = r.block("sphere<point,real>", p, 0.02)
    r.export_stl(ball, "C:/tmp/ball", units="mm", tolerance=1.0e-3)
    funcs = [e["contents"].get("func") for e in r.to_dict()["body"]]
    assert any(f and f.startswith("implicit_to_mesh") for f in funcs)
    # The path gains the .stl extension and forward slashes.
    export = [e for e in r.to_dict()["body"]
              if (e["contents"].get("func") or "").startswith("export_mesh")][0]
    assert export["contents"]["inputs"][0]["value"]["val"] == "C:/tmp/ball.stl"


# --------------------------------------------------------------------------------------
#   3. End to end against a real ntopcl
# --------------------------------------------------------------------------------------


def _build_smoke_recipe(stl_path: str) -> Recipe:
    """Sphere with the radius as a notebook input -> mass properties -> mesh -> STL."""
    r = Recipe(
        name="wp1_smoke",
        displayname="WP1 Sphere Smoke",
        description="WP1 smoke notebook: sphere of a given radius, measured and exported.",
    )
    radius = r.add_input(
        "Radius", "real", default=SPHERE_RADIUS_M, dimension={"length": 1},
        description="Sphere radius",
    )
    origin = r.block("point<real,real,real>", 0.0, 0.0, 0.0, name="Origin")
    ball = r.block("sphere<point,real>", origin, radius, name="Ball")
    props = r.mass_properties(ball, density=SPHERE_DENSITY, relative_error=0.001,
                              name="Ball Mass Properties")
    volume = r.variable("Volume", props.prop("volume"))
    r.variable("Mass", props.prop("mass"))
    mesh = r.mesh_from_implicit(ball, tolerance=MESH_TOLERANCE_M, name="Ball Mesh")
    r.export_stl(mesh, stl_path, units="m")
    r.set_output(volume)
    return r


@pytest.fixture(scope="session")
def runner() -> NtopRunner:
    try:
        return NtopRunner()
    except NtopError as exc:                                     # pragma: no cover
        pytest.skip(f"no ntopcl available: {exc}")


@pytest.fixture(scope="session")
def smoke(runner: NtopRunner) -> dict[str, object]:
    """Author, convert, template, and run the smoke notebook once for the whole session."""
    d = os.path.join(SMOKE_DIR, "e2e")
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)

    stl = to_ntop_path(os.path.join(d, "ball.stl"))
    recipe = _build_smoke_recipe(stl)
    recipe_json = os.path.join(d, "wp1_smoke_recipe.json")
    recipe.write_json(recipe_json)
    ntop = os.path.join(d, "wp1_smoke.ntop")

    convert = runner.convert(recipe_json, ntop)
    input_template, output_template = runner.templates(ntop, require_output=True)
    out_json = os.path.join(d, "output.json")
    run = runner.run(
        ntop,
        {"Radius": (SPHERE_RADIUS_M, "m")},
        out_json=out_json,
        expect=[stl],
        input_template=input_template,
        run_dir=d,
        timeout=900,
    )
    roundtrip_json = os.path.join(d, "roundtrip.json")
    runner.exportjson(ntop, roundtrip_json)

    return {
        "dir": d,
        "recipe": recipe,
        "recipe_json": recipe_json,
        "ntop": ntop,
        "stl": stl,
        "convert": convert,
        "input_template": input_template,
        "output_template": output_template,
        "run": run,
        "out_json": out_json,
        "roundtrip_json": roundtrip_json,
    }


def test_convert_succeeds(smoke: dict[str, object]) -> None:
    convert = smoke["convert"]
    # REFERENCE.md section 6: 0 and 72 both count as success for ntopcl.
    assert convert.returncode in (0, 72), convert.tail()
    assert convert.returncode_ok
    assert os.path.getsize(str(smoke["ntop"])) > 0


def test_input_template_exposes_the_radius(smoke: dict[str, object]) -> None:
    template = smoke["input_template"]
    names = [i["name"] for i in template["inputs"]]
    assert names == ["Radius"]
    entry = template["inputs"][0]
    assert entry["type"] == "real"
    assert entry["description"] == "Sphere radius"
    # nTop writes the default in DISPLAY units. For a length that is mm, so 0.025 m -> 25.
    assert entry["units"] == "mm"
    assert entry["value"] == pytest.approx(SPHERE_RADIUS_M * 1000.0, rel=1e-9)
    assert template["title"] == "WP1 Sphere Smoke"


def test_output_template_exists(smoke: dict[str, object]) -> None:
    out = smoke["output_template"]
    assert isinstance(out, list) and len(out) == 1
    assert out[0]["name"] == "Volume"
    assert out[0]["type"] == "real"


def test_run_succeeds_and_reports_its_returncode(smoke: dict[str, object]) -> None:
    run = smoke["run"]
    assert run.returncode in (0, 72), run.tail()
    assert run.returncode_ok and run.artefacts_ok
    assert run.log_path and os.path.isfile(str(run.log_path))
    assert os.path.isfile(os.path.join(str(smoke["dir"]), "input.json"))


def test_stl_exists_and_is_non_empty(smoke: dict[str, object]) -> None:
    stl = str(smoke["stl"])
    assert os.path.isfile(stl)
    assert os.path.getsize(stl) > 0


def test_stl_volume_matches_analytic_sphere(smoke: dict[str, object]) -> None:
    """Parse the exported STL and compare its enclosed volume with 4/3 pi r^3."""
    tris = read_stl(str(smoke["stl"]))
    assert len(tris) > 1000, "suspiciously coarse mesh"
    volume = enclosed_volume(tris)
    error = abs(volume - ANALYTIC_VOLUME) / ANALYTIC_VOLUME
    lo, hi = bounding_box(tris)
    # Exported with units="m", so the STL is in metres and spans one diameter.
    assert hi[0] - lo[0] == pytest.approx(2.0 * SPHERE_RADIUS_M, rel=0.01)
    assert error < VOLUME_TOLERANCE, (
        f"STL volume {volume:.6e} m^3 vs analytic {ANALYTIC_VOLUME:.6e} m^3, "
        f"error {error * 100:.3f} percent"
    )


def test_notebook_reported_volume_matches_analytic(smoke: dict[str, object]) -> None:
    parsed = NtopRunner.parse_outputs(str(smoke["out_json"]),
                                      extra_map={"Volume": "volume_total"})
    volume = parsed.measurements.volume_total
    assert volume is not None
    error = abs(volume - ANALYTIC_VOLUME) / ANALYTIC_VOLUME
    assert error < VOLUME_TOLERANCE, (
        f"notebook volume {volume:.6e} m^3 vs analytic {ANALYTIC_VOLUME:.6e} m^3, "
        f"error {error * 100:.3f} percent"
    )
    assert parsed.raw["Volume"] == pytest.approx(volume)
    assert parsed.unmapped == []

    # Passing the RunResult fills the bookkeeping fields of NtopMeasurements.
    with_run = NtopRunner.parse_outputs(str(smoke["out_json"]), run=smoke["run"])
    assert with_run.measurements.ntopcl_returncode == smoke["run"].returncode
    assert with_run.measurements.wall_time_s == pytest.approx(smoke["run"].wall_time_s)


def test_parse_outputs_default_name_map(smoke: dict[str, object]) -> None:
    """"Volume" is already in `OUTPUT_NAME_MAP`, so no extra_map is needed for it."""
    parsed = NtopRunner.parse_outputs(str(smoke["out_json"]))
    assert parsed.measurements.volume_total == pytest.approx(ANALYTIC_VOLUME, rel=0.01)
    assert parsed.unmapped == []


def test_parse_outputs_keeps_unmapped_entries() -> None:
    """Outputs the name table does not claim must survive in `raw`, and be listed."""
    d = os.path.join(SMOKE_DIR, "parse")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "synthetic_output.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {"components": [], "name": "Volume", "type": "real",
                 "value": {"isFinite": True, "units": {"length": 3}, "val": 1.25}},
                {"components": [], "name": "Some Future Quantity", "type": "real",
                 "value": {"isFinite": True, "units": {}, "val": 7.0}},
                {"components": [], "name": "CG", "type": "point",
                 "value": [{"isFinite": True, "val": 1.0},
                           {"isFinite": True, "val": 2.0},
                           {"isFinite": True, "val": 3.0}]},
                {"components": [], "name": "Report", "type": "text",
                 "value": {"string": "ok"}},
                {"components": [], "name": "Files", "type": "json",
                 "value": {"jsonObject": {"stl_path": "C:/out/a.stl"}}},
            ],
            f,
        )
    parsed = NtopRunner.parse_outputs(path)
    assert parsed.measurements.volume_total == pytest.approx(1.25)
    assert parsed.measurements.cg_structure == (1.0, 2.0, 3.0)
    # A `json` output is unpacked, so a notebook can report many values through one slot.
    assert parsed.measurements.stl_path == "C:/out/a.stl"
    assert parsed.raw["Some Future Quantity"] == 7.0
    assert parsed.raw["Report"] == "ok"
    assert "Some Future Quantity" in parsed.unmapped
    assert "Report" in parsed.unmapped
    assert "Volume" not in parsed.unmapped


def test_run_honours_a_changed_input(runner: NtopRunner) -> None:
    """Prove the notebook input drives the geometry, rather than the baked-in default.

    This builds its own notebook in its own directory. The STL path is a literal inside the
    notebook, so reusing the session notebook here would overwrite the artefact the volume
    tests inspect, and make those tests order-dependent.
    """
    d = os.path.join(SMOKE_DIR, "e2e_r30")
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    stl = to_ntop_path(os.path.join(d, "ball.stl"))
    recipe = _build_smoke_recipe(stl)
    recipe_json = os.path.join(d, "recipe.json")
    recipe.write_json(recipe_json)
    ntop = os.path.join(d, "wp1_smoke_r30.ntop")
    runner.convert(recipe_json, ntop)
    input_template, _ = runner.templates(ntop, require_output=True)

    radius = 0.030
    out_json = os.path.join(d, "output_r30.json")
    runner.run(
        ntop,
        {"Radius": (radius, "m")},
        out_json=out_json,
        expect=[stl],
        input_template=input_template,
        run_dir=d,
        timeout=900,
    )
    volume = NtopRunner.parse_outputs(out_json, extra_map={"Volume": "volume_total"}) \
        .measurements.volume_total
    assert volume is not None
    expected = analytic_sphere_volume(radius)
    assert abs(volume - expected) / expected < VOLUME_TOLERANCE
    assert volume > 1.5 * ANALYTIC_VOLUME, "input change had no effect"
    # The exported STL follows the input too.
    tris = read_stl(stl)
    stl_volume = enclosed_volume(tris)
    assert abs(stl_volume - expected) / expected < VOLUME_TOLERANCE


def test_roundtrip_preserves_authored_blocks(smoke: dict[str, object]) -> None:
    """`exportjson` the generated notebook and check our blocks survived."""
    with open(str(smoke["roundtrip_json"]), "r", encoding="utf-8") as f:
        doc = json.load(f)

    original = smoke["recipe"].to_dict()

    assert doc["displayname"] == original["displayname"]
    assert doc["description"] == original["description"]
    assert doc["version"] == [1, 0, 0]
    assert "output" in doc, "the notebook output designation must survive the round trip"

    # Notebook inputs survive verbatim, including dimension and default.
    assert len(doc["inputs"]) == 1
    got_in = doc["inputs"][0]
    want_in = original["inputs"][0]
    assert got_in["name"] == want_in["name"]
    assert got_in["type"] == want_in["type"]
    assert got_in["dimension"] == want_in["dimension"]
    assert got_in["contents"]["value"]["val"] == pytest.approx(SPHERE_RADIUS_M)

    def index(document: dict) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for entry in document["body"]:
            out[str(entry.get("name"))] = entry
        return out

    got = index(doc)
    want = index(original)
    assert set(want) <= set(got), f"blocks lost in round trip: {sorted(set(want) - set(got))}"

    for name, want_entry in want.items():
        got_entry = got[name]
        assert got_entry["type"] == want_entry["type"], name
        assert got_entry.get("variable") is True, name
        want_contents = want_entry.get("contents") or {}
        got_contents = got_entry.get("contents") or {}
        assert got_contents.get("func") == want_contents.get("func"), name
        if "inputs" in want_contents:
            assert len(got_contents.get("inputs", [])) == len(want_contents["inputs"]), name

    # The property selection on the mass-properties block survives as a props chain.
    assert (got["Volume"]["contents"]).get("props") == ["volume"]
    assert (got["Mass"]["contents"]).get("props") == ["mass"]

    # Literal arguments survive with their units. The mesh tolerance is a length in metres.
    mesh_inputs = got["Ball Mesh"]["contents"]["inputs"]
    assert mesh_inputs[1]["value"]["units"] == {"length": 1}
    assert mesh_inputs[1]["value"]["val"] == pytest.approx(MESH_TOLERANCE_M)

    # The export path stays a forward-slash file_path.
    export_path = got["Export STL"]["contents"]["inputs"][0]["value"]["val"]
    assert export_path == str(smoke["stl"])
    assert "\\" not in export_path


def test_json_output_reports_many_quantities(runner: NtopRunner) -> None:
    """A recipe has ONE output slot, so many quantities go through one `json` value.

    This is the pattern WP4 must use. It is checked end to end because every block involved is
    absent from the vendored universe.
    """
    d = os.path.join(SMOKE_DIR, "json_output")
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)

    r = Recipe(name="json_output", displayname="JSON Output")
    radius = r.add_input("Radius", "real", default=SPHERE_RADIUS_M,
                         dimension={"length": 1})
    origin = r.block("point<real,real,real>", 0.0, 0.0, 0.0, name="Origin")
    ball = r.block("sphere<point,real>", origin, radius, name="Ball")
    props = r.mass_properties(ball, density=SPHERE_DENSITY, relative_error=0.001)
    mesh = r.mesh_from_implicit(ball, tolerance=MESH_TOLERANCE_M, name="Ball Mesh")
    area = r.surface_area(mesh, name="Area")
    r.json_output(
        {
            "volume_total": (props.prop("volume"), {"length": 3}),
            "mass_structure": (props.prop("mass"), {"mass": 1}),
            "area_wetted_body": (area, {"length": 2}),
        }
    )

    ntop = os.path.join(d, "json_output.ntop")
    r.to_ntop(ntop, ntopcl=runner)
    input_template, output_template = runner.templates(ntop, require_output=True)
    assert output_template[0]["type"] == "json"

    out_json = os.path.join(d, "output.json")
    runner.run(ntop, {"Radius": (SPHERE_RADIUS_M, "m")}, out_json=out_json,
               input_template=input_template, run_dir=d, timeout=900)
    parsed = NtopRunner.parse_outputs(out_json)
    m = parsed.measurements

    assert m.volume_total is not None
    assert abs(m.volume_total - ANALYTIC_VOLUME) / ANALYTIC_VOLUME < VOLUME_TOLERANCE
    analytic_area = 4.0 * math.pi * SPHERE_RADIUS_M ** 2
    assert m.area_wetted_body is not None
    assert abs(m.area_wetted_body - analytic_area) / analytic_area < VOLUME_TOLERANCE
    analytic_mass = SPHERE_DENSITY * ANALYTIC_VOLUME
    assert m.mass_structure is not None
    assert abs(m.mass_structure - analytic_mass) / analytic_mass < VOLUME_TOLERANCE
    # The container's own name is bookkeeping, not an unmapped measurement.
    assert parsed.unmapped == []


def test_mixed_units_in_one_list_is_rejected_without_dimensionless(runner: NtopRunner) -> None:
    """`core.list<real>` demands identical units. Document that with a real failure."""
    d = os.path.join(SMOKE_DIR, "mixed_units")
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    r = Recipe(name="mixed", displayname="Mixed Units")
    origin = r.block("point<real,real,real>", 0.0, 0.0, 0.0, name="Origin")
    ball = r.block("sphere<point,real>", origin, SPHERE_RADIUS_M, name="Ball")
    props = r.mass_properties(ball, density=SPHERE_DENSITY, relative_error=0.01)
    # Volume is length^3 and mass is mass; nTop refuses to put them in one list<real>.
    r.json_output({"volume_total": props.prop("volume"),
                   "mass_structure": props.prop("mass")})
    ntop = os.path.join(d, "mixed.ntop")
    r.to_ntop(ntop, ntopcl=runner)
    with pytest.raises(NtopError) as exc:
        runner.templates(ntop, require_output=True)
    assert "Output of function not built" in str(exc.value)


def test_recipe_to_ntop_convenience(runner: NtopRunner) -> None:
    """`Recipe.to_ntop` writes the recipe and converts it in one call."""
    d = os.path.join(SMOKE_DIR, "to_ntop")
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    r = Recipe(name="to_ntop", displayname="To nTop")
    p = r.block("point<real,real,real>", 0.0, 0.0, 0.0, name="Origin")
    ball = r.block("sphere<point,real>", p, SPHERE_RADIUS_M, name="Ball")
    r.set_output(r.surface_area(ball, 0.01, name="Area"))
    ntop = os.path.join(d, "to_ntop.ntop")
    assert r.to_ntop(ntop, ntopcl=runner) == ntop
    assert os.path.getsize(ntop) > 0
    assert os.path.isfile(os.path.join(d, "to_ntop_recipe.json"))
    _, output_template = runner.templates(ntop, require_output=True)
    area = output_template[0]["value"]["val"]
    analytic = 4.0 * math.pi * SPHERE_RADIUS_M ** 2
    assert abs(area - analytic) / analytic < VOLUME_TOLERANCE


def test_exit_code_72_with_missing_artefacts_is_a_failure(
    runner: NtopRunner, smoke: dict[str, object]
) -> None:
    """Exit code 72 is NOT proof of success, so success must be gated on artefacts.

    Verified on this machine: running the smoke notebook with a negative radius makes the
    sphere block fail, no output JSON is written, and `ntopcl` exits **72** - the very code
    REFERENCE.md section 6 says means success. The driver must still raise.
    """
    d = os.path.join(SMOKE_DIR, "rc72")
    os.makedirs(d, exist_ok=True)
    out_json = os.path.join(d, "output.json")
    with pytest.raises(NtopError) as exc:
        runner.run(
            str(smoke["ntop"]),
            {"Radius": (-1.0, "m")},
            out_json=out_json,
            input_template=smoke["input_template"],
            run_dir=d,
            timeout=600,
        )
    message = str(exc.value)
    assert "ntopcl run failed" in message
    assert "missing or empty artefacts" in message
    assert "out of range" in message, "the captured nTop diagnostics must be surfaced"
    assert not os.path.exists(out_json)


def test_convert_rejects_a_bad_notebook(runner: NtopRunner) -> None:
    """A recipe that ntopcl cannot convert must raise, with the captured output attached."""
    d = os.path.join(SMOKE_DIR, "bad")
    os.makedirs(d, exist_ok=True)
    bad = os.path.join(d, "bad_recipe.json")
    with open(bad, "w", encoding="utf-8") as f:
        json.dump({"body": [{"func": "definitely_not_a_block<real>", "id": "inst101",
                             "inputs": [], "type": "real"}],
                   "cbRefs": [], "description": "", "displayname": "bad", "imports": [],
                   "inputs": [], "name": "user_func_bad", "namespaces": [],
                   "version": [1, 0, 0]}, f)
    with pytest.raises(NtopError) as exc:
        runner.convert(bad, os.path.join(d, "bad.ntop"), timeout=180)
    assert "ntopcl convert failed" in str(exc.value)
    assert "returned" in str(exc.value)


def test_runner_resolution_error() -> None:
    with pytest.raises(NtopError, match="no ntopcl executable found"):
        NtopRunner(ntopcl=os.path.join(SMOKE_DIR, "nope", "ntopcl.exe"))
