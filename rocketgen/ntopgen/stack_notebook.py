"""The parametric IV-1 two-stage notebook: booster, interstage, strake-equipped payload stage.

Read `SPEC_IV1.md` section 3 first, then `docs/NTOP_NOTES.md`. This module is to IV-1 what
`rocket_notebook.py` is to SV-1, and it deliberately does not touch it: SV-1 is the regression
baseline.

What this builds
----------------
THREE separate implicit solids, measured SEPARATELY, plus a stacked union for the picture. The
split is not cosmetic: the trajectory needs per-stage mass properties because stage 1 is
jettisoned, and the aero needs per-stage wetted areas because the reference area changes at
separation.

Everything lives on the +X axis with the STACK's nose tip at the origin:

```
    x = 0            x = L2      x = L2 + L_is                      x = L_total
    |                |           |                                  |
    | payload stage  |interstage |            booster               |
    |----------------|-----------|----------------------------------|
     ogive + strakes                              stage-1 tail fins
     + stage-2 tail fins
```

* **Payload stage (stage 2)** - one closed 2D polygon revolved 360 degrees about X: a tangent
  ogive of length `f_nose * D2` (or a cone, if `dv.nose_shape == "cone"`), then a cylinder to
  the stage-2 aft end, then a flat base. Four **strakes** and four cruciform **tail fins** are
  constant-thickness extruded plates unioned onto it. The interior is
  `offset_implicit(OML, -t_wall)` less two ring bulkheads, which makes the seeker bay, the
  payload bay and the motor bay real separated cavities.
* **Booster (stage 1)** - a `cylinder` of diameter `D1` and length `L1`, four cruciform tail
  fins, hollowed to `t_wall1`. One cavity: the motor bay.
* **Interstage** - a truncated `cone` from `D2` to `D1` over `L_interstage`, hollowed to
  `t_interstage`. ONE block covers both cases SPEC_IV1 asks for: a cone when the diameters
  differ, and the same block with equal radii IS a cylinder when they match, so there is no
  topology switch and no second cached notebook. It is jettisoned with stage 1.

WHAT THE MEASURED STRUCTURE IS, AND WHAT IT IS NOT
--------------------------------------------------
`sN_volume_structure` and `sN_mass_structure` are the **AIRFRAME AND ITS SURFACES ONLY**: the
wall, the bulkheads, the fin panels and (on stage 2) the strake panels. They deliberately do
NOT include

* the motor case or its insulation - `rocketgen/sizing/masses_iv1.py` charges those separately
  from the motor model ("Motor case", "Motor insulation");
* the propellant grain - charged from `StageSpec.m_propellant`;
* the payload - charged from `InterceptRequirements.m_payload` (SPEC_IV1 A5);
* the seeker, avionics and actuation package;
* the nozzle and igniter.

Double counting any of those would make the whole stack heavier than it is and would corrupt
the sizing loop, silently and largely. The sanity check is scale: a SOLID billet of 7075-T6
filling the default IV-1 envelope (stage 2 plus interstage plus booster outer mould lines,
0.4270 m^3) would weigh 1200 kg, which is more than the whole A8 launch-mass limit of 1400 kg
on its own. A correct hollow two-stage structure is TENS of kilograms per stage: the MEASURED
default is 23.4 kg for stage 2, 32.5 kg for the booster and 3.4 kg for the interstage.
`tests/test_stack_notebook.py` asserts that band for exactly this reason.

Output key convention
---------------------
One notebook now reports three bodies, so every scalar carries a body prefix:

| prefix | body |
|---|---|
| `s1_` | booster, stage 1 |
| `s2_` | payload stage, stage 2 |
| `is_` | interstage |
| `st_` | the stacked assembly |

so `s1_volume_total`, `s2_volume_total`, `st_volume_total` and so on. `measure_stack` splits the
raw output dictionary on that prefix and fills one `StageMeasurements` per body, so the sizing
loop sees ordinary `NtopMeasurements` fields with no prefixes at all.

CAVEAT, and it matters: because `s1_volume_total` and `s2_volume_total` both map onto the single
`NtopMeasurements.volume_total` field, calling `driver.parse_outputs` DIRECTLY on this
notebook's output JSON produces a flat `ParsedOutputs.measurements` in which each field holds
whichever body happened to be mapped last. That object is meaningless. Use `measure_stack`,
which never looks at it. The prefixed names are registered in `driver.OUTPUT_NAME_MAP` anyway so
that `ParsedOutputs.unmapped` stays honest about what the notebook emitted.

Coordinate frames for the CG
----------------------------
The geometry is built in ONE frame, the stack frame, because building each stage twice would
double the notebook. But `masses_iv1.build_stack_masses` reads `m.cg_structure[0]` as a station
from **that stage's own forward face** and adds the stage offset itself. So the notebook also
reports `sN_x_forward`, the stage's forward face in stack coordinates, and `measure_stack`
subtracts it. `StageMeasurements.cg_structure` is therefore STAGE-LOCAL, and
`StageMeasurements.cg_structure_stack` and `.x_forward` keep the stack-frame values.

Performance
-----------
`docs/NTOP_NOTES.md` sections 3, 4, 19 and 22. `ntopcl convert` EVALUATES the notebook, exports
included, and `implicit_to_mesh` costs about tolerance^-3 over the bounding box. The IV-1 stack
box is 5.08 x 0.80 x 0.80 m = 3.25 m^3 against SV-1's 2.02 m^3, so the mesh is about 1.6x more
expensive at the same tolerance. Hence: every design dimension is a real notebook INPUT, the
`.ntop` is converted ONCE and cached on a topology hash, and every export is OFF by default.

MEASURED on this machine: one `measure_stack` call at the default IV-1 costs **55 to 118 s**
(five repeats: 55.0, 78.6, 92.7, 114.7, 117.8) and the one-off `convert` that precedes it costs
**63 to 96 s**. The 2x spread on repeats of an identical job is real and is not attributed here.
Where
the time goes IS clear: four `surface_area<implicit,real>` calls are most of it, at 24.6 s on the
stage-2 body, 24.2 s on the stage-2 fins, 23.8 s on the strakes and 16.9 s on the stage-1 fins.
The fifth area call, on the booster's plain `cylinder`, costs 0.27 s, and that is the clue: the
cost is the IMPLICIT FIELD's complexity, not the surface area. A revolved ogive and a four-panel
mirrored union are expensive fields; a primitive is not. Twelve `mass_properties` calls add about
60 s more, the largest being 15.0 s on the stage-2 structure. NOTHING here is meshing. See
SOURCES["measured_wall_time"].

Units are SI: metres, radians, kilograms (CLAUDE.md hard rule 3.4).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..config import MATERIALS, RUNS_DIR, NtopMeasurements, register_sources
from ..config_iv1 import StackDesignVector, StageSpec, StrakeSpec
from .driver import (
    NtopError,
    NtopRunner,
    measurements_from_names,
    parse_outputs,
    register_output_names,
)
from .recipe import Recipe, Ref, to_ntop_path

__all__ = [
    "BODY_PREFIXES",
    "STACK_OUTPUT_NAMES",
    "NTOP_STAGE_INPUTS",
    "NTOP_GLOBAL_INPUTS",
    "StackNotebook",
    "StageMeasurements",
    "build_stack_recipe",
    "build_stack_notebook",
    "measure_stack",
    "geometry_fn",
    "stack_geometry_closed_form",
    "strake_solid_area",
    "fin_solid_area",
    "clear_stack_notebook_cache",
    "MEASURED_WALL_TIME_S",
    "MEASURED_CONVERT_TIME_S",
    "DEFAULT_MESH_TOLERANCE",
    "DEFAULT_CAD_TOLERANCE",
    "DEFAULT_RELATIVE_ERROR",
    "DEFAULT_AREA_RELATIVE_ERROR",
    "N_OGIVE",
    "SECTION_AREA_BLOCK",
    "SOURCES",
]

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
#   Empirical constants, with sources
# --------------------------------------------------------------------------------------

SOURCES: dict[str, str] = {
    "iv1_ogive_polygon_sampling": (
        "Inherited, not re-guessed. The revolved outer mould line is a chord polygon, so its "
        "volume is exactly the frustum sum that `rocketgen.sizing.masses._tangent_ogive_volume` "
        "computes with the same n. WP4 swept n on the SV-1 nose against a 20000-segment "
        "reference and measured a total outer-mould-line volume error of -0.106 percent at n=8, "
        "-0.0117 percent at n=24 and -0.0004 percent at n=128. The IV-1 stage-2 nose is "
        "SHORTER and BLUNTER than SV-1's (1.008 m on a 0.28 m body, fineness 3.6, against "
        "1.05 m on 0.35 m at fineness 3.0), and a chord polygon's error scales with the "
        "curvature it has to cut across, so n = 24 is at least as accurate here. It is kept "
        "unchanged so the two notebooks are comparable."
    ),
    "iv1_plate_sections": (
        "Measured by WP4, not chosen here. Both the strakes and the tail fins are "
        "CONSTANT-THICKNESS extruded plates, not double wedges. The double wedge was built by "
        "lofting a root diamond to a tip diamond and measured WRONG: "
        "`loft<implicit_2d,implicit_2d>[1.1.0]` is a linear MIX of two signed-distance FIELDS, "
        "not an interpolation of two boundaries, and it returned 82 percent of the exact panel "
        "volume and 73 percent of the exact area (docs/NTOP_NOTES.md section 16). A plate is "
        "exact in planform, span, sweep and wetted area. For a STRAKE the plate is also the "
        "physically right section: a strake really is a constant-thickness rib, not a "
        "double-wedge aerofoil. For the tail fins it makes the panel volume 2x a diamond of "
        "the same maximum thickness, so fin structural mass is conservative."
    ),
    "iv1_plate_root_plug": (
        "Every plate root sits at radius R - t_wall rather than exactly R, so the union with "
        "the airframe wall overlaps instead of touching tangentially. Tangential contact of two "
        "implicit bodies is numerically fragile. The leading edge and the chord are "
        "extrapolated inboard along the same straight sweep and taper lines, so the chord at "
        "the true body surface y = R is still exactly c_r_fin (or, for a strake, exactly "
        "StrakeSpec.length)."
    ),
    "iv1_single_density_structure": (
        "The nTop-measured structure is ONE solid measured at ONE density, "
        "MATERIALS['airframe_al7075'] = 2810 kg/m^3, because `mass_properties` takes a single "
        "density. `masses_iv1.py` charges fins and strakes at MATERIALS['fin_ti64'] = "
        "4430 kg/m^3 in its ANALYTIC branch, so the measured mass is lighter than the analytic "
        "estimate for the same panels by (4430-2810)/2810 = 58 percent OF THE PANEL VOLUME "
        "ONLY. The notebook therefore also reports `s2_volume_strakes`, `s2_volume_fins` and "
        "`s1_volume_fins`, the exposed panel volumes, so a consumer that wants a different "
        "panel material can re-charge them without re-running nTop. Nothing is corrected here: "
        "the notebook reports what it measured."
    ),
    "iv1_interstage_one_block": (
        "SPEC_IV1 section 3 asks for a cone when the stage diameters differ and a cylinder when "
        "they match. `cone<point,point,real,real>` with Radius 1 == Radius 2 IS a cylinder, so "
        "one block covers both and the notebook topology does not depend on the diameters. That "
        "matters: a topology that depended on a dimension would need a second cached `.ntop` "
        "and would break the convert-once-run-many pattern."
    ),
    "iv1_stage1_two_base_discs": (
        "For the booster, `area_wetted_body` is the LATERAL cylinder area only, so BOTH flat "
        "end discs are subtracted from the measured total surface area: the aft disc is the "
        "nozzle-exit base, and the forward disc is covered by the interstage and is not wetted. "
        "For the payload stage only the single aft disc is subtracted, because the nose is "
        "closed by the ogive. This matches what `masses_iv1.stage_geometry` returns as "
        "`area_wetted_body` (nose plus cylinder lateral area, no discs), so the measured and "
        "the analytic numbers are directly comparable."
    ),
    "iv1_strake_area_includes_edges": (
        "The measured strake wetted area is the area of the SOLID plate outside the body, so it "
        "includes the outboard tip face, the two edge faces and the cylindrical root patch left "
        "by the boolean. `StrakeSpec.wetted_area` is the ZERO-THICKNESS reference, "
        "2 * n * height * length. On the default IV-1 the solid is 1.27x the zero-thickness "
        "value, because an 8 mm plate 30 mm tall is 27 percent edge. That is not an error: skin "
        "friction acts on the real surface, so the measured number is the right one to hand to "
        "`aero_iv1.py`. The closed form for the solid is `strake_solid_area()` in this module, "
        "and `tests/test_stack_notebook.py` checks the measurement against it."
    ),
    "iv1_area_relative_error": (
        "Measured by WP4 on this machine, and the finding is negative. "
        "`surface_area<implicit,real>` is the most expensive block in either notebook: 13 to "
        "22 s per call. Its 'Relative error' input was swept over 0.002, 0.01, 0.05 and 0.2 and "
        "made NO difference to either the reported area (bit-identical 4.003007 m^2 at every "
        "target) or the wall time. There is no accuracy-for-speed trade to make on this build. "
        "0.01 is kept because it is the value the same block was verified accurate to "
        "0.0097 percent at on the WP1 smoke sphere (docs/NTOP_NOTES.md section 11)."
    ),
    "iv1_relative_error": (
        "`mass_properties<implicit,real_field,real>` takes a relative-error target. 0.002 is "
        "used: on the WP1 smoke sphere the block reported volume to 0.0104 percent at 0.001, so "
        "0.002 is comfortably inside the 1 percent gate while keeping the adaptive integration "
        "cheap. Inherited unchanged from `rocket_notebook.py`."
    ),
    "iv1_mesh_tolerance": (
        "Scaled from WP4's measurement, then measured again on the stack. `implicit_to_mesh` "
        "drives a DENSE voxel grid, so cost and memory go as (bounding box volume)/tolerance^3. "
        "The IV-1 stack box is 5.08 x 0.80 x 0.80 m = 3.25 m^3 against SV-1's 2.02 m^3, i.e. "
        "1.61x, so SV-1's measured 5.0e-3 m working point (51.7 s, 9.2 MB) predicts about 83 s "
        "and 15 MB here, and SV-1's 3.0e-3 m failure (past 1.4 GB, killed at 128 s) predicts a "
        "worse failure here. 5.0e-3 m is the default: it resolves the 8 mm strake thickness "
        "with about 1.6 cells and the 10 mm fin thickness with 2. Anything finer must be run "
        "deliberately with the resident set watched. See MEASURED_MESH_COST for what actually "
        "happened. The mesh is used ONLY for the STL export, never for a measurement."
    ),
    "iv1_cad_tolerance": (
        "Inherited from WP4's measurement on a part of the same class. "
        "`cad_body_from_implicit_body` scales with the bounding box: on the 4 m SV-1, 2.0e-2 m "
        "took 11.6 s and 1.0e-2 m took 22.8 s, while 2.0e-3 m passed 9 GB resident and had to "
        "be killed (docs/NTOP_NOTES.md section 23). The IV-1 box is 1.6x larger again, so "
        "1.0e-2 m is the default and 2.0e-2 m is the safe fallback."
    ),
    "iv1_fin_te_at_stage_aft_end": (
        "MODELLING CHOICE, and it is visible in the render. SPEC_IV1 gives `StageSpec` a fin "
        "semi-span, root chord, taper, sweep and thickness but NO longitudinal station, so the "
        "trailing edge is placed flush with each stage's own aft end, which is what 'tail fin' "
        "means. On stage 2 that puts the panels immediately forward of the interstage, and "
        "because the stage-2 tip-to-tip span (0.56 m on the default) exceeds the booster "
        "diameter (0.40 m) the panels stick out past the interstage flare. A real vehicle would "
        "move them forward or use a wrap-around skirt. Nothing in the aero or mass model depends "
        "on the station, so this is a picture-level simplification, not a numerical one."
    ),
    "iv1_two_stage2_bulkheads": (
        "SPEC_IV1 section 3 asks for a seeker bay of `L_seeker`, a payload bay of "
        "`L_payload_bay` and a motor bay filling the rest of the payload stage. Three bays need "
        "TWO ring bulkheads, at `L_seeker` and at `L_seeker + L_payload_bay`. The booster gets "
        "none: its cavity is one motor bay, which is what SPEC_IV1 asks for. Bulkheads are "
        "oversized discs that are only ever SUBTRACTED from the interior void, so nothing has "
        "to trim them (docs/NTOP_NOTES.md section 18)."
    ),
    "measured_wall_time": (
        "MEASURED on this machine with `ntopcl -v 2` per-block timings, not estimated. Five runs "
        "of the full notebook with every export off took 55.0, 78.6, 92.7, 114.7 and 117.8 s "
        "wall, of which about 8 s each is fixed ntopcl startup. `convert` of the same notebook "
        "took 62.6, 95.4 and 96.5 s. The 2x SPREAD on repeats of an identical job is real and "
        "is not "
        "explained here: it is not the block graph, because the 62.6 s convert was of the LARGER "
        "422-block version, and no attempt is made to attribute it. Budget the upper end. Where "
        "the time goes is clear from the per-block timings: the four expensive "
        "`surface_area<implicit,real>` calls are 24.6 s (stage-2 body), 24.2 s (stage-2 fins), "
        "23.8 s (strakes) and 16.9 s (stage-1 fins), while the fifth, on the booster's plain "
        "`cylinder`, is 0.27 s. `mass_properties` on the stage-2 structure is 15.0 s and on the "
        "stack union 8.6 s. Nothing here is meshing. The build ladder measured 14.7 s at "
        "build_stage 's2_oml', 42.2 s at 's2_plates', 50.5 s at 's2_hollow', 76.3 s at 'booster' "
        "and 117.8 s at 'full', so each added body costs what its own area and mass blocks cost "
        "and nothing more. Probe: `runs/IV-1_geom/_probe.py`."
    ),
}

# Measured wall time of one `measure_stack` call at the default IV-1, seconds, exports off. The
# mean of five measured repeats (55.0, 78.6, 92.7, 114.7, 117.8 s); the spread on an identical
# job is about 2x and is unexplained. See SOURCES["measured_wall_time"]. Reported, not enforced:
# the sizing loop uses it to budget, and the test only gates an order-of-magnitude regression.
MEASURED_WALL_TIME_S = 92.0
MEASURED_CONVERT_TIME_S = 85.0

# Every key is namespaced `iv1_`. `config.register_sources` REFUSES a key that another module
# already registered with different text, which is a real guard rail and not a nuisance:
# `rocket_notebook.py` owns `relative_error` and `area_relative_error` with SV-1 timings in
# them, and a full-suite run imports both modules. Sharing a key would have silently attached
# SV-1 numbers to an IV-1 constant in the report.
register_sources(SOURCES)


# `implicit_to_mesh` tolerance for the exported STL, metres. See SOURCES["iv1_mesh_tolerance"].
# NOT a measurement tolerance: every measured quantity comes off the implicit body directly.
DEFAULT_MESH_TOLERANCE = 5.0e-3

# `cad_body_from_implicit_body` tolerance for STEP export, metres.
DEFAULT_CAD_TOLERANCE = 1.0e-2

# Relative-error target handed to `mass_properties`. See SOURCES["iv1_relative_error"].
DEFAULT_RELATIVE_ERROR = 0.002

# Relative-error target handed to `surface_area<implicit,real>`. It gets its own knob because it
# is by far the most expensive block. See SOURCES["iv1_area_relative_error"].
DEFAULT_AREA_RELATIVE_ERROR = 0.01

# Chord-polygon sample count for the stage-2 ogive. See SOURCES["iv1_ogive_polygon_sampling"].
N_OGIVE = 24

# nTop enum encodings. `blend_enum` 0 is the no-blend option, which is what a sharp-edged
# airframe wants. REFERENCE.md section 5 documents the {"enum": N} encoding.
BLEND_NONE = 0

# Cross-section area of an `implicit_2d`. The vendored universe lists
# `surface_area<implicit_2d,real>[1.2.0]` as current and `body_surface_area<implicit_2d,real>`
# as deprecated. On both installed builds that is BACKWARDS: every revision of
# `surface_area<implicit_2d,real>` is rejected by `ntopcl convert`, and only the "deprecated"
# one loads. See `docs/NTOP_NOTES.md` section 24.
SECTION_AREA_BLOCK = "body_surface_area<implicit_2d,real>[1.1.0]"

# `extract_section`'s optional "Min. Feature Size", metres. A section through an 8 mm strake
# plate is a thin sliver, and the default feature size is a plausible way to lose it.
SECTION_FEATURE_SIZE = 1.0e-3

STRUCTURE_DENSITY = MATERIALS["airframe_al7075"].density         # 2810 kg/m^3

# Recipe schema version, and it is part of the topology cache key. BUMP IT whenever the authored
# block graph changes, so a cached `.ntop` built by an older version of this module is not reused
# and silently measured as if it were the current one.
#   1 -> 2: the interstage lateral wetted area is measured, so the stack total is complete.
RECIPE_VERSION = 2


# --------------------------------------------------------------------------------------
#   The nTop input contract
# --------------------------------------------------------------------------------------

# Per-stage inputs. (StageSpec attribute, nTop input-name suffix, dimension map).
# The nTop input name is "S<index> <suffix>", e.g. "S1 Diameter".
NTOP_STAGE_INPUTS: tuple[tuple[str, str, dict[str, int]], ...] = (
    ("D",         "Diameter",       {"length": 1}),
    ("L",         "Length",         {"length": 1}),
    ("t_wall",    "Wall Thickness", {"length": 1}),
    ("b_fin",     "Fin Semi Span",  {"length": 1}),
    ("c_r_fin",   "Fin Root Chord", {"length": 1}),
    ("taper_fin", "Fin Taper",      {}),
    ("sweep_fin", "Fin Sweep",      {"angle": 1}),
    ("t_fin",     "Fin Thickness",  {"length": 1}),
)

# Stack-level inputs. (dotted attribute path on StackDesignVector, nTop input name, dimension).
NTOP_GLOBAL_INPUTS: tuple[tuple[str, str, dict[str, int]], ...] = (
    ("f_nose",             "Nose Fineness",      {}),
    ("L_interstage",       "Interstage Length",  {"length": 1}),
    ("t_interstage",       "Interstage Wall",    {"length": 1}),
    ("L_seeker",           "Seeker Bay Length",  {"length": 1}),
    ("L_payload_bay",      "Payload Bay Length", {"length": 1}),
    ("strakes.height",     "Strake Height",      {"length": 1}),
    ("strakes.length",     "Strake Length",      {"length": 1}),
    ("strakes.thickness",  "Strake Thickness",   {"length": 1}),
    ("strakes.x_le",       "Strake LE Station",  {"length": 1}),
    ("strakes.sweep_le",   "Strake Sweep",       {"angle": 1}),
)

# Non-geometric inputs: they steer cost and where artefacts land, never the shape.
MESH_TOLERANCE_INPUT = "Mesh Tolerance"
STL_PATH_INPUT = "STL Path"
STEP_PATH_INPUT = "STEP Path"
IMPLICIT_PATH_INPUT = "Implicit Path"

# Display units used when writing the `-j` input JSON. NTOP_NOTES.md section 2: an explicit
# `units` string IS honoured, and omitting it falls back to the template's display unit, so the
# driver always writes one. Everything here is SI.
_UNIT_FOR_DIMENSION = {(("length", 1),): "m", (("angle", 1),): "rad", (): ""}


def _display_unit(dimension: Mapping[str, int]) -> str | None:
    key = tuple(sorted(dimension.items()))
    return _UNIT_FOR_DIMENSION.get(key) or None


def _stage_input_name(index: int, suffix: str) -> str:
    return f"S{int(index)} {suffix}"


def _dotted(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


# --------------------------------------------------------------------------------------
#   Output names
# --------------------------------------------------------------------------------------

# Body prefixes. See the module docstring for the convention.
BODY_PREFIXES: dict[str, str] = {
    "s1": "booster, stage 1",
    "s2": "payload stage, stage 2",
    "is": "interstage",
    "st": "stacked assembly",
}

# Scalars emitted per body, unprefixed. Not every body emits every name: the interstage has no
# cavity and no aerodynamic surfaces, and the stacked assembly is only measured for volume.
STACK_OUTPUT_NAMES: tuple[str, ...] = (
    "volume_total",
    "volume_structure",
    "volume_cavity",
    "volume_strakes",
    "volume_fins",
    "area_wetted_body",
    "area_wetted_strakes",
    "area_wetted_fins",
    "area_base",
    "mass_structure",
    "x_forward",
    "length",
)

# Vector-valued measurements travel as three named scalars, because `core.list<real>` (and
# therefore the json output) only carries scalars. NTOP_NOTES.md section 13 point 2.
CG_COMPONENTS = ("cg_x", "cg_y", "cg_z")
INERTIA_COMPONENTS = ("inertia_1", "inertia_2", "inertia_3")

# Names that land on a real `NtopMeasurements` field, unprefixed. `volume_strakes`,
# `volume_fins`, `area_wetted_strakes`, `x_forward` and `length` do NOT: they are extra fields on
# `StageMeasurements`, which is why `config.py` does not have to be edited for them.
_TO_MEASUREMENT_FIELD: dict[str, str] = {
    "volume_total": "volume_total",
    "volume_structure": "volume_structure",
    "volume_cavity": "volume_cavity",
    "area_wetted_body": "area_wetted_body",
    "area_wetted_fins": "area_wetted_fins",
    "area_base": "area_base",
    "mass_structure": "mass_structure",
}

# Additively extend `driver.OUTPUT_NAME_MAP`, so the prefixed names this notebook emits are not
# reported as unmapped. `config.py` is never edited (NTOP_NOTES.md section 13 point 3).
#
# READ THE CAVEAT IN THE MODULE DOCSTRING. Registering `s1_volume_total` and `s2_volume_total`
# onto the same `volume_total` field means the FLAT `ParsedOutputs.measurements` from this
# notebook is meaningless. `measure_stack` never uses it; it splits `ParsedOutputs.raw` by
# prefix instead.
register_output_names(
    {
        f"{prefix}_{name}": target
        for prefix in BODY_PREFIXES
        for name, target in _TO_MEASUREMENT_FIELD.items()
    }
)


# --------------------------------------------------------------------------------------
#   The measurement record
# --------------------------------------------------------------------------------------


@dataclass
class StageMeasurements(NtopMeasurements):
    """`NtopMeasurements` plus the IV-1-specific quantities.

    A dataclass SUBCLASS, not an edit to `config.py`: SV-1 is the regression baseline and
    `config.NtopMeasurements` is its contract. Everything the sizing loop and `masses_iv1.py`
    already read is inherited unchanged, so `dict[int, StageMeasurements]` satisfies
    `dict[int, NtopMeasurements]` and `aero_iv1.StackAero(meas=...)` needs no adapter.

    The added fields:

    * `area_wetted_strakes` - the strake panels' wetted area, kept APART from
      `area_wetted_fins` because `aero_iv1.py` needs them separately: strakes are
      vortex-dominated at low aspect ratio and the fins are not.
    * `volume_strakes`, `volume_fins` - the exposed panel volumes, so a consumer that charges
      panels at a different density than the airframe can do so without re-running nTop. See
      SOURCES["iv1_single_density_structure"].
    * `x_forward` - this body's forward face in STACK coordinates.
    * `cg_structure_stack` - the structure CG in STACK coordinates. `cg_structure` itself is
      stage-local, which is what `masses_iv1.build_stack_masses` expects.
    """

    stage_index: int = 0
    body: str = ""
    area_wetted_strakes: float | None = None
    volume_strakes: float | None = None
    volume_fins: float | None = None
    x_forward: float | None = None
    length: float | None = None
    cg_structure_stack: tuple[float, float, float] | None = None

    @property
    def area_wetted_surfaces(self) -> float:
        """Fins plus strakes, m^2. Convenience for a drag build-up that wants the total."""
        return (self.area_wetted_fins or 0.0) + (self.area_wetted_strakes or 0.0)


# --------------------------------------------------------------------------------------
#   Closed-form helpers, for the cross-checks and for the callers that want them
# --------------------------------------------------------------------------------------


def strake_solid_area(st: StrakeSpec, R: float) -> float:
    """Wetted area of the SOLID strake panels outside a body of radius `R`, m^2.

    `StrakeSpec.wetted_area` is the zero-thickness reference, `2 n h L`. A real plate of
    thickness `t` also has

    * an outboard tip face, `t L` per panel;
    * a leading and a trailing edge face, `t h` each;
    * a root patch where the boolean cut it against the body, an arc of `2 R asin(t / 2R)`
      by `L`, which is `t L` to better than 0.01 percent for `t << R`.

    So per panel `2 h L + 2 t L + 2 t h`, times `n`. On the default IV-1
    (`n=4, h=0.030, L=1.400, t=0.008, R=0.140`):

        2*0.030*1.400 = 0.084000      two side faces
        2*0.008*1.400 = 0.022400      tip face plus root patch
        2*0.008*0.030 = 0.000480      leading and trailing edge faces
                        --------
                        0.106880  per panel,  x 4 = 0.427520 m^2

    against `StrakeSpec.wetted_area` = 2*4*0.030*1.400 = 0.336000 m^2, i.e. 1.2724x. See
    SOURCES["iv1_strake_area_includes_edges"].
    """
    if st.n <= 0 or st.height <= 0.0 or st.length <= 0.0:
        return 0.0
    h, L, t = st.height, st.length, st.thickness
    arc = 2.0 * R * math.asin(min(1.0, 0.5 * t / R)) if R > 0.0 else t
    per_panel = 2.0 * h * L + t * L + arc * L + 2.0 * t * h
    return st.n * per_panel


def fin_solid_area(stage: StageSpec, R: float) -> float:
    """Wetted area of the SOLID tail-fin panels outside a body of radius `R`, m^2.

    Same construction as `strake_solid_area`, but the planform is a swept tapered trapezium:

    * two side faces, `2 S_exposed` per panel;
    * a tip end face, `t c_t`;
    * a leading edge face `t * b / cos(sweep_le)` and a trailing edge face
      `t * b / cos(sweep_te)`, where the trailing-edge sweep follows from the taper;
    * a root patch, `arc(t) * c_root` where `c_root` is the chord at the body surface.
    """
    n = stage.n_fin
    if n <= 0 or stage.b_fin <= 0.0:
        return 0.0
    b, c_r, c_t, t = stage.b_fin, stage.c_r_fin, stage.c_t_fin, stage.t_fin
    arc = 2.0 * R * math.asin(min(1.0, 0.5 * t / R)) if R > 0.0 else t
    dx_le = b * math.tan(stage.sweep_fin)
    dx_te = dx_le + (c_t - c_r)
    le = math.hypot(b, dx_le)
    te = math.hypot(b, dx_te)
    per_panel = 2.0 * stage.S_fin_exposed + t * c_t + t * (le + te) + arc * c_r
    return n * per_panel


def stack_geometry_closed_form(dv: StackDesignVector) -> dict[str, dict[str, float]]:
    """Independent closed-form outer-mould-line geometry, per body. The test reference.

    Uses `rocketgen.sizing.masses._tangent_ogive_volume` and `_tangent_ogive_surface_area`,
    which `tests/test_masses.py` already validates against an exact hemisphere. Nothing here
    touches nTop, so it is a genuinely independent check.

    Keys are the body prefixes of `BODY_PREFIXES`. Per body: `volume_total`,
    `area_wetted_body` (LATERAL area only, no end discs), `area_base`, `area_wetted_fins`,
    `area_wetted_strakes`, `x_forward`, `length`.
    """
    from ..sizing.masses import _tangent_ogive_surface_area, _tangent_ogive_volume

    s2, s1 = dv.payload_stage, dv.booster
    R2, R1 = 0.5 * s2.D, 0.5 * s1.D
    L_nose = dv.L_nose
    L_cyl2 = s2.L - L_nose

    # stage 2: tangent ogive (or cone) plus a cylinder to the stage aft end
    if dv.nose_shape == "cone":
        v_nose = math.pi * R2 * R2 * L_nose / 3.0
        a_nose = math.pi * R2 * math.hypot(L_nose, R2)
    else:
        v_nose = _tangent_ogive_volume(L_nose, R2)
        a_nose = _tangent_ogive_surface_area(L_nose, R2)

    out: dict[str, dict[str, float]] = {}
    out["s2"] = {
        "volume_total": v_nose + math.pi * R2 * R2 * L_cyl2,
        "area_wetted_body": a_nose + 2.0 * math.pi * R2 * L_cyl2,
        "area_base": math.pi * R2 * R2,
        "area_wetted_fins": fin_solid_area(s2, R2),
        "area_wetted_strakes": strake_solid_area(dv.strakes, R2),
        "x_forward": 0.0,
        "length": s2.L,
    }
    # booster: a plain cylinder. volume = pi R^2 L, lateral area = 2 pi R L.
    out["s1"] = {
        "volume_total": math.pi * R1 * R1 * s1.L,
        "area_wetted_body": 2.0 * math.pi * R1 * s1.L,
        "area_base": math.pi * R1 * R1,
        "area_wetted_fins": fin_solid_area(s1, R1),
        "area_wetted_strakes": 0.0,
        "x_forward": s2.L + dv.L_interstage,
        "length": s1.L,
    }
    # interstage: a truncated cone from R2 to R1 over L_interstage.
    #   V = pi L (R2^2 + R2 R1 + R1^2) / 3,  lateral A = pi (R2 + R1) sqrt(L^2 + (R1-R2)^2)
    L_is = dv.L_interstage
    out["is"] = {
        "volume_total": math.pi * L_is * (R2 * R2 + R2 * R1 + R1 * R1) / 3.0,
        "area_wetted_body": math.pi * (R2 + R1) * math.hypot(L_is, R1 - R2),
        "area_base": 0.0,
        "area_wetted_fins": 0.0,
        "area_wetted_strakes": 0.0,
        "x_forward": s2.L,
        "length": L_is,
    }
    out["st"] = {
        "volume_total": (out["s2"]["volume_total"] + out["s1"]["volume_total"]
                         + out["is"]["volume_total"]),
        "area_wetted_body": (out["s2"]["area_wetted_body"] + out["s1"]["area_wetted_body"]
                             + out["is"]["area_wetted_body"]),
        "area_base": out["s1"]["area_base"],
        "area_wetted_fins": out["s2"]["area_wetted_fins"] + out["s1"]["area_wetted_fins"],
        "area_wetted_strakes": out["s2"]["area_wetted_strakes"],
        "x_forward": 0.0,
        "length": dv.L_total,
    }
    return out


# --------------------------------------------------------------------------------------
#   Recipe construction
# --------------------------------------------------------------------------------------

# How far the build goes. A debugging ladder, exactly as `rocket_notebook.stage` is: build the
# bare stage-2 body first, then add the plates, then hollow it, then the booster, then the
# interstage. CLAUDE.md section 3.5.
BUILD_STAGES: tuple[str, ...] = ("s2_oml", "s2_plates", "s2_hollow", "booster", "full")


class _Builder:
    """Assembles the IV-1 recipe. One instance per recipe; not reusable."""

    def __init__(
        self,
        dv: StackDesignVector,
        *,
        n_ogive: int,
        relative_error: float,
        area_relative_error: float,
        mesh_tolerance: float,
        cad_tolerance: float,
        export_stl: bool,
        export_step: bool,
        export_implicit: bool,
        default_dir: str,
        area_stations: int,
        section_feature_size: float,
        build_stage: str,
    ) -> None:
        if build_stage not in BUILD_STAGES:
            raise ValueError(
                f"build_stage {build_stage!r} is not one of {BUILD_STAGES}"
            )
        self.dv = dv
        self.n_ogive = int(n_ogive)
        self.rel = float(relative_error)
        self.area_rel = float(area_relative_error)
        self.mesh_tolerance = float(mesh_tolerance)
        self.cad_tolerance = float(cad_tolerance)
        self.export_stl = bool(export_stl)
        self.export_step = bool(export_step)
        self.export_implicit = bool(export_implicit)
        self.default_dir = to_ntop_path(default_dir)
        self.area_stations = int(area_stations)
        self.section_feature_size = float(section_feature_size)
        self.build_stage = build_stage
        self.nose_shape = str(dv.nose_shape)
        self.n_strake = int(dv.strakes.n)

        self.r = Recipe(
            name="iv1_stack",
            displayname="IV-1 Two-Stage Stack",
            description=(
                "IV-1 parametric two-stage interceptor-class stack. Ogive-cylinder payload "
                "stage with four strakes and four cruciform tail fins, a conical interstage, "
                "and a cylindrical booster with four cruciform tail fins. Each of the three "
                "bodies is hollowed to its own wall thickness and measured separately. Outputs "
                "volumes, wetted areas and structural mass properties as one JSON value, with "
                "s1_/s2_/is_/st_ prefixes naming the body."
            ),
        )
        self.inp: dict[str, Ref] = {}
        self.g: dict[str, Ref] = {}
        # measured values, keyed "<prefix>_<name>" -> (ref, dimension map)
        self.values: dict[str, tuple[Ref, dict[str, int]]] = {}

    # ---- small arithmetic conveniences ------------------------------------------------

    def _mul(self, a: Any, b: Any, name: str | None = None) -> Ref:
        return self.r.block("multiply<real,real>", a, b, name=name)

    def _add(self, a: Any, b: Any, name: str | None = None) -> Ref:
        return self.r.block("add<real,real>", a, b, name=name)

    def _sub(self, a: Any, b: Any, name: str | None = None) -> Ref:
        return self.r.block("subtract<real,real>", a, b, name=name)

    def _div(self, a: Any, b: Any, name: str | None = None) -> Ref:
        return self.r.block("divide<real,real>", a, b, name=name)

    def _scale(self, ref: Ref, factor: float, name: str | None = None) -> Ref:
        """Multiply a dimensioned ref by a dimensionless Python constant."""
        return self._mul(ref, self.r.literal_real(factor, {}), name=name)

    def _point(self, x: Any, y: Any, z: Any, name: str | None = None) -> Ref:
        return self.r.block("point<real,real,real>", x, y, z, name=name)

    def _zero(self) -> Ref:
        return self.r.literal_real(0.0, {"length": 1})

    def _union(self, bodies: Sequence[Ref], name: str) -> Ref:
        r = self.r
        func = r.latest("boolean_union<blend_enum,real_field,list<implicit>>")
        return r.block(
            func,
            r.literal_enum("blend_enum", BLEND_NONE),
            self._zero(),
            r.list_of("implicit", list(bodies), name=f"{name} Bodies"),
            name=name,
        )

    def _subtract(self, primary: Ref, bodies: Sequence[Ref], name: str) -> Ref:
        r = self.r
        func = r.latest("boolean_subtract<blend_enum,real_field,implicit,list<implicit>>")
        return r.block(
            func,
            r.literal_enum("blend_enum", BLEND_NONE),
            self._zero(),
            primary,
            r.list_of("implicit", list(bodies), name=f"{name} Subtractions"),
            name=name,
        )

    def _mass_props(self, body: Ref, density: float, name: str) -> Ref:
        return self.r.mass_properties(body, density=density, relative_error=self.rel, name=name)

    def _area(self, body: Ref, name: str) -> Ref:
        return self.r.surface_area(body, relative_error=self.area_rel, name=name)

    # ---- inputs -----------------------------------------------------------------------

    def declare_inputs(self) -> None:
        """Every dimension the sizer moves becomes a real nTop notebook input.

        That is what makes convert-once-run-many possible (NTOP_NOTES.md section 13 point 5).
        Only the topology choices - the nose shape, the panel counts, the ogive sample count and
        which exports exist - are baked in, and they form the cache key.
        """
        r, dv = self.r, self.dv
        for stage in dv.stages:
            for attr, suffix, dim in NTOP_STAGE_INPUTS:
                key = f"s{stage.index}.{attr}"
                self.inp[key] = r.add_input(
                    _stage_input_name(stage.index, suffix), "real",
                    default=float(getattr(stage, attr)), dimension=dim,
                    description=f"StageSpec(index={stage.index}).{attr}",
                )
        for path, iname, dim in NTOP_GLOBAL_INPUTS:
            self.inp[path] = r.add_input(
                iname, "real", default=float(_dotted(dv, path)), dimension=dim,
                description=f"StackDesignVector.{path}",
            )
        self.inp[MESH_TOLERANCE_INPUT] = r.add_input(
            MESH_TOLERANCE_INPUT, "real", default=self.mesh_tolerance,
            dimension={"length": 1},
            description="implicit_to_mesh tolerance for the exported STL",
        )
        for flag, iname, stem in (
            (self.export_stl, STL_PATH_INPUT, "iv1.stl"),
            (self.export_step, STEP_PATH_INPUT, "iv1.step"),
            (self.export_implicit, IMPLICIT_PATH_INPUT, "iv1.implicit"),
        ):
            if flag:
                self.inp[iname] = r.add_input(
                    iname, "file_path",
                    default=to_ntop_path(os.path.join(self.default_dir, stem)),
                    description=f"{stem} export path",
                )

    # ---- derived stations, all computed INSIDE nTop ------------------------------------

    def derive(self) -> None:
        """Every derived dimension, computed from the notebook inputs inside nTop.

        Nothing here may be a Python number taken from `self.dv`, or the cached `.ntop` would
        be silently wrong at any other design point.
        """
        i = self.inp
        s2 = self.dv.payload_stage.index
        s1 = self.dv.booster.index

        R2 = self._scale(i[f"s{s2}.D"], 0.5, name="S2 Radius")
        R1 = self._scale(i[f"s{s1}.D"], 0.5, name="S1 Radius")
        L_nose = self._mul(i["f_nose"], i[f"s{s2}.D"], name="Nose Length")

        # stack stations
        x_s2_aft = i[f"s{s2}.L"]
        x_is_aft = self._add(x_s2_aft, i["L_interstage"], name="Interstage Aft Station")
        x_s1_aft = self._add(x_is_aft, i[f"s{s1}.L"], name="Stack Aft Station")

        # stage-2 bulkhead stations, from its own (and the stack's) nose tip
        x_bh1 = i["L_seeker"]
        x_bh2 = self._add(i["L_seeker"], i["L_payload_bay"], name="Payload Bay Aft Station")

        self.g.update(
            R2=R2, R1=R1, L_nose=L_nose,
            x_s2_fwd=self._zero(), x_s2_aft=x_s2_aft,
            x_is_fwd=x_s2_aft, x_is_aft=x_is_aft,
            x_s1_fwd=x_is_aft, x_s1_aft=x_s1_aft,
            x_bh1=x_bh1, x_bh2=x_bh2,
            neg_t_wall_2=self._scale(i[f"s{s2}.t_wall"], -1.0, name="S2 Inward Offset"),
            neg_t_wall_1=self._scale(i[f"s{s1}.t_wall"], -1.0, name="S1 Inward Offset"),
            neg_t_is=self._scale(i["t_interstage"], -1.0, name="Interstage Inward Offset"),
        )

    # ---- profiles ---------------------------------------------------------------------

    def _ogive_points(self, length: Ref, radius: Ref, n: int) -> list[Ref]:
        """Chord-polygon points of a tangent ogive from the tip, all arithmetic done in nTop.

        The profile is `y(x) = sqrt(rho^2 - (L - x)^2) - (rho - R)` with
        `rho = (R^2 + L^2) / (2R)`. Written with `u = x / L` and `k = L / R` that becomes

            y / R = sqrt(c^2 - k^2 (1 - u)^2) - (c - 1),    c = (1 + k^2) / 2,

        which is dimensionless apart from the final multiply by R, so `sqrt<real>` never has to
        take the root of a length squared. `k`, `c^2` and `c - 1` are shared across all samples.
        """
        r = self.r
        k = self._div(length, radius, name="Ogive k")
        k2 = self._mul(k, k, name="Ogive k^2")
        c = self._add(self._scale(k2, 0.5), r.literal_real(0.5, {}), name="Ogive c")
        c2 = self._mul(c, c, name="Ogive c^2")
        cm1 = self._sub(c, r.literal_real(1.0, {}), name="Ogive c-1")

        pts: list[Ref] = [self._point(self._zero(), self._zero(), self._zero(),
                                      name="Nose Tip")]
        if n < 2:
            raise ValueError(f"need at least 2 ogive segments, got {n}")
        for idx in range(1, n):
            u = idx / float(n)
            d = (1.0 - u) ** 2
            s = self._sub(c2, self._mul(k2, r.literal_real(d, {})))
            yr = self._sub(r.block("sqrt<real>", s), cm1)
            pts.append(self._point(self._mul(length, r.literal_real(u, {})),
                                   self._mul(radius, yr), self._zero()))
        pts.append(self._point(length, radius, self._zero(), name="Nose Shoulder"))
        return pts

    def _cone_nose_points(self, length: Ref, radius: Ref) -> list[Ref]:
        return [
            self._point(self._zero(), self._zero(), self._zero(), name="Nose Tip"),
            self._point(length, radius, self._zero(), name="Nose Shoulder"),
        ]

    def build_stage2_oml(self) -> Ref:
        """The payload stage's outer mould line: ONE closed profile revolved about X.

        `docs/NTOP_NOTES.md` section 15: one closed polygon revolved 360 degrees is the cheapest
        exact body of revolution. No booleans, one implicit body, and the measured volume is
        exactly the frustum sum of the chord polygon.
        """
        r, g = self.r, self.g

        if self.nose_shape == "cone":
            pts = self._cone_nose_points(g["L_nose"], g["R2"])
        elif self.nose_shape == "tangent_ogive":
            pts = self._ogive_points(g["L_nose"], g["R2"], self.n_ogive)
        else:
            raise ValueError(
                f"unsupported nose_shape {self.nose_shape!r}; use 'tangent_ogive' or 'cone'"
            )
        # cylinder to the stage-2 aft end, then a flat base, then close along the axis
        pts.append(self._point(g["x_s2_aft"], g["R2"], self._zero(), name="S2 Base Rim"))
        pts.append(self._point(g["x_s2_aft"], self._zero(), self._zero(),
                               name="S2 Base Centre"))

        poly = r.block("profile_from_points<list<point>>", r.point_list(pts,
                       name="S2 OML Points"), name="S2 OML Polygon")
        # `profile_from_points` returns a `profile` ("Polygon"); `revolve` wants an
        # `implicit_2d` ("Profile"). types.json gives `profile` a `profile: implicit_2d`
        # property, so the props chain is the only bridge. NTOP_NOTES.md section 14.
        profile_2d = r.variable("S2 OML Profile", poly.prop("profile"))
        axis = r.block(
            "axis<point,vector>",
            self._point(self._zero(), self._zero(), self._zero(), name="Axis Origin"),
            r.literal_vector(1.0, 0.0, 0.0),
            name="Body Axis",
        )
        body = r.block(
            "revolve<implicit_2d,axis,real>", profile_2d, axis,
            r.literal_real(2.0 * math.pi, {"angle": 1}), name="S2 Body OML",
        )
        self.g["axis"] = axis
        self.g["oml2"] = body
        return body

    # ---- plates: strakes and fins ------------------------------------------------------

    def _plate(
        self,
        span_axis: str,
        x_le_root: Ref, x_te_root: Ref, x_le_tip: Ref, x_te_tip: Ref,
        y_root: Ref, y_tip: Ref, thickness: Ref, half_neg: Ref,
        name: str,
    ) -> Ref:
        """One panel: a planform quadrilateral extruded to a constant thickness.

        Both the strakes and the tail fins are built by this one routine, because both are
        CONSTANT-THICKNESS PLATES. See SOURCES["iv1_plate_sections"] for why a double wedge is
        not reachable with these blocks and why the plate is the right choice for a strake
        anyway.

        `extrude<implicit_2d,real,vector>` extrudes ONE-SIDED from the profile's own plane, so
        placing the polygon at `-t/2` and extruding `+t` centres the plate on the panel plane.
        The traversal LE-root, LE-tip, TE-tip, TE-root gives a consistent polygon normal on both
        span axes (NTOP_NOTES.md section 17).

        `span_axis` is "y" for the panel spanning +Y (thickness along Z) or "z" for the panel
        spanning +Z (thickness along Y).
        """
        r = self.r
        off = half_neg
        if span_axis == "y":
            quad = [(x_le_root, y_root, off), (x_le_tip, y_tip, off),
                    (x_te_tip, y_tip, off), (x_te_root, y_root, off)]
            direction = r.literal_vector(0.0, 0.0, 1.0)
        elif span_axis == "z":
            quad = [(x_le_root, off, y_root), (x_le_tip, off, y_tip),
                    (x_te_tip, off, y_tip), (x_te_root, off, y_root)]
            direction = r.literal_vector(0.0, 1.0, 0.0)
        else:
            raise ValueError(f"span_axis must be 'y' or 'z', got {span_axis!r}")

        pts = [self._point(*q) for q in quad]
        poly = r.block("profile_from_points<list<point>>",
                       r.point_list(pts, name=f"{name} Planform Points"),
                       name=f"{name} Planform Polygon")
        profile_2d = r.variable(f"{name} Planform", poly.prop("profile"))
        return r.block("extrude<implicit_2d,real,vector>", profile_2d, thickness, direction,
                       name=name)

    def _cruciform(self, make_panel: Any, label: str) -> Ref:
        """Four panels at 0, 90, 180 and 270 degrees, as two panels and two mirrors.

        `mirror_body<implicit,plane>` returns a real `implicit`, whereas
        `rotate<spatial3d,point,vector,real>` returns `any` and would then have to be forced
        into a `list<implicit>`. A cruciform is exactly two mirror pairs, so no rotation block
        is needed at all. NTOP_NOTES.md section 17.
        """
        r = self.r
        zero = self._zero()
        origin = self._point(zero, zero, zero, name=f"{label} Mirror Origin")
        plane_xz = r.block("plane<point,vector,vector>", origin,
                           r.literal_vector(1.0, 0.0, 0.0), r.literal_vector(0.0, 0.0, 1.0),
                           name=f"{label} XZ Plane")
        plane_xy = r.block("plane<point,vector,vector>", origin,
                           r.literal_vector(1.0, 0.0, 0.0), r.literal_vector(0.0, 1.0, 0.0),
                           name=f"{label} XY Plane")
        panel_y = make_panel("y", f"{label} +Y")
        panel_z = make_panel("z", f"{label} +Z")
        panel_ny = r.block("mirror_body<implicit,plane>", panel_y, plane_xz, name=f"{label} -Y")
        panel_nz = r.block("mirror_body<implicit,plane>", panel_z, plane_xy, name=f"{label} -Z")
        return self._union([panel_y, panel_ny, panel_z, panel_nz], name=label)

    def build_strakes(self) -> Ref | None:
        """Four strakes on the payload stage mid-body.

        A strake is a rectangle in planform (a parallelogram if `sweep_le` is non-zero), of
        chordwise length `StrakeSpec.length` and radial height `StrakeSpec.height` above the
        body surface, its leading edge at `StrakeSpec.x_le` from the stage-2 nose tip. There is
        no taper: `StrakeSpec` does not define one.

        The root sits at `R - t_wall` so it plugs into the wall rather than touching it
        tangentially, and the leading edge is extrapolated inboard along the same sweep line so
        that the chord at the true body surface is still exactly `length`.
        SOURCES["iv1_plate_root_plug"].
        """
        if self.n_strake == 0:
            return None
        if self.n_strake != 4:
            raise ValueError(
                f"strakes.n = {self.n_strake} is not supported. The notebook bakes the "
                f"cruciform in as two panels and two mirrors, because the panel count changes "
                f"the block graph and therefore the cached notebook. Use 4 or 0."
            )
        i, g = self.inp, self.g
        s2 = self.dv.payload_stage.index
        h = i["strakes.height"]
        Ls = i["strakes.length"]
        t = i["strakes.thickness"]
        half_neg = self._scale(t, -0.5, name="Strake Half Thickness Negative")
        tan_s = self.r.block("tan<real>", i["strakes.sweep_le"], name="Tan Strake Sweep")

        y_root = self._sub(g["R2"], i[f"s{s2}.t_wall"], name="Strake Root Radius")
        y_tip = self._add(g["R2"], h, name="Strake Tip Radius")
        x_le_root = self._sub(i["strakes.x_le"],
                              self._mul(i[f"s{s2}.t_wall"], tan_s),
                              name="Strake Root LE Station")
        x_te_root = self._add(x_le_root, Ls, name="Strake Root TE Station")
        x_le_tip = self._add(i["strakes.x_le"], self._mul(h, tan_s),
                             name="Strake Tip LE Station")
        x_te_tip = self._add(x_le_tip, Ls, name="Strake Tip TE Station")

        def make(span_axis: str, name: str) -> Ref:
            return self._plate(span_axis, x_le_root, x_te_root, x_le_tip, x_te_tip,
                               y_root, y_tip, t, half_neg, name)

        strakes = self._cruciform(make, "Strakes")
        self.g["strakes"] = strakes
        return strakes

    def build_fins(self, stage: StageSpec, x_aft: Ref, radius: Ref, label: str) -> Ref | None:
        """Four cruciform tail fins with their trailing edge flush with `x_aft`.

        The station is a modelling choice, because `StageSpec` gives no longitudinal position.
        See SOURCES["iv1_fin_te_at_stage_aft_end"].
        """
        n = int(stage.n_fin)
        if n == 0:
            return None
        if n != 4:
            raise ValueError(
                f"stage {stage.index}: n_fin = {n} is not supported. The notebook bakes the "
                f"cruciform in as two panels and two mirrors, because the panel count changes "
                f"the block graph and therefore the cached notebook. Use 4 or 0."
            )
        i = self.inp
        k = f"s{stage.index}"
        b = i[f"{k}.b_fin"]
        c_r = i[f"{k}.c_r_fin"]
        t_wall = i[f"{k}.t_wall"]
        t = i[f"{k}.t_fin"]
        half_neg = self._scale(t, -0.5, name=f"{label} Half Thickness Negative")
        c_t = self._mul(i[f"{k}.taper_fin"], c_r, name=f"{label} Tip Chord")
        tan_s = self.r.block("tan<real>", i[f"{k}.sweep_fin"], name=f"Tan {label} Sweep")

        x_le = self._sub(x_aft, c_r, name=f"{label} Root LE Station")
        y_root = self._sub(radius, t_wall, name=f"{label} Root Radius")
        y_tip = self._add(radius, b, name=f"{label} Tip Radius")

        # inboard extrapolation along the same sweep and taper lines, so the chord at the true
        # body surface y = R is exactly c_r even though the plate root is at R - t_wall
        x_le_root = self._sub(x_le, self._mul(t_wall, tan_s))
        c_root = self._add(c_r, self._mul(self._div(t_wall, b), self._sub(c_r, c_t)))
        x_te_root = self._add(x_le_root, c_root)
        x_le_tip = self._add(x_le, self._mul(b, tan_s))
        x_te_tip = self._add(x_le_tip, c_t)

        def make(span_axis: str, name: str) -> Ref:
            return self._plate(span_axis, x_le_root, x_te_root, x_le_tip, x_te_tip,
                               y_root, y_tip, t, half_neg, name)

        fins = self._cruciform(make, label)
        self.g[f"fins{stage.index}"] = fins
        return fins

    # ---- the booster and the interstage -------------------------------------------------

    def build_booster_oml(self) -> Ref:
        """The booster outer mould line: a plain cylinder.

        `cylinder<point,point,real>` is one exact block. There is no ogive to revolve, so the
        polygon route would only add arithmetic.
        """
        g = self.g
        zero = self._zero()
        body = self.r.block(
            "cylinder<point,point,real>",
            self._point(g["x_s1_fwd"], zero, zero, name="S1 Forward Centre"),
            self._point(g["x_s1_aft"], zero, zero, name="S1 Aft Centre"),
            g["R1"], name="S1 Body OML",
        )
        self.g["oml1"] = body
        return body

    def build_interstage(self) -> tuple[Ref, Ref]:
        """The interstage: a truncated cone from the stage-2 radius to the booster radius.

        One block covers both of SPEC_IV1's cases: with equal radii a cone IS a cylinder, so the
        notebook topology does not depend on the diameters.
        SOURCES["iv1_interstage_one_block"].

        Returns `(solid, structure)`, where the structure is the conical shell of thickness
        `t_interstage`. It is jettisoned with stage 1.
        """
        g = self.g
        zero = self._zero()
        solid = self.r.block(
            "cone<point,point,real,real>",
            self._point(g["x_is_fwd"], zero, zero, name="Interstage Forward Centre"),
            self._point(g["x_is_aft"], zero, zero, name="Interstage Aft Centre"),
            g["R2"], g["R1"], name="Interstage OML",
        )
        void = self.r.block(
            "offset_implicit<implicit,real_field>", solid, g["neg_t_is"],
            name="Interstage Void",
        )
        structure = self._subtract(solid, [void], name="Interstage Structure")
        self.g["is_solid"] = solid
        self.g["is_structure"] = structure
        return solid, structure

    # ---- hollowing ----------------------------------------------------------------------

    def _bulkhead(self, x_station: Ref, t_wall: Ref, radius: Ref, name: str) -> Ref:
        """A ring bulkhead: a disc of thickness `t_wall` centred on `x_station`.

        The radius is the full body radius, oversize on purpose: the disc is only ever
        SUBTRACTED from the interior void, so anything outside the void is discarded and nothing
        has to be trimmed. NTOP_NOTES.md section 18.
        """
        zero = self._zero()
        half = self._scale(t_wall, 0.5)
        return self.r.block(
            "cylinder<point,point,real>",
            self._point(self._sub(x_station, half), zero, zero),
            self._point(self._add(x_station, half), zero, zero),
            radius, name=name,
        )

    def build_stage2_cavity(self) -> Ref:
        """Stage-2 usable internal volume: the inward offset of the OML, less two bulkheads.

        `offset_implicit(OML, -t_wall)` is ONE block and gives a true normal offset of the
        implicit field, which is exactly the inner surface of a constant-thickness shell. The
        alternative - a second revolved profile with every radius reduced by t_wall - needs
        about a hundred more arithmetic blocks and is only approximate near the nose.
        NTOP_NOTES.md section 18.

        Two bulkheads make three real separated bays: seeker, payload, motor.
        SOURCES["iv1_two_stage2_bulkheads"].
        """
        g, i = self.g, self.inp
        s2 = self.dv.payload_stage.index
        t_wall = i[f"s{s2}.t_wall"]
        void = self.r.block(
            "offset_implicit<implicit,real_field>", g["oml2"], g["neg_t_wall_2"],
            name="S2 Interior Void",
        )
        bulkheads = [
            self._bulkhead(g["x_bh1"], t_wall, g["R2"], "S2 Bulkhead Seeker Aft"),
            self._bulkhead(g["x_bh2"], t_wall, g["R2"], "S2 Bulkhead Payload Aft"),
        ]
        cavity = self._subtract(void, bulkheads, name="S2 Internal Cavity")
        self.g["cavity2"] = cavity
        return cavity

    def build_booster_cavity(self) -> Ref:
        """Booster internal volume: one motor bay, the inward offset of the cylinder.

        No bulkheads. SPEC_IV1 section 3 gives the booster a motor and nothing else, so its
        cavity is one bay. `offset_implicit` shrinks the two flat end discs as well as the
        lateral wall, which is exactly a closed case with flat end plates of thickness `t_wall`.
        """
        g = self.g
        cavity = self.r.block(
            "offset_implicit<implicit,real_field>", g["oml1"], g["neg_t_wall_1"],
            name="S1 Motor Bay",
        )
        self.g["cavity1"] = cavity
        return cavity

    # ---- measurements -------------------------------------------------------------------

    def _emit(self, key: str, ref: Ref, dim: dict[str, int]) -> None:
        self.values[key] = (ref, dim)

    def _measure_structure(self, prefix: str, structure: Ref, label: str) -> None:
        """Volume, mass, CG and principal moments of one body's structure.

        THE DENSITY IS APPLIED INSIDE NTOP, by `mass_properties<implicit,real_field,real>`, so
        `mass_structure` is nTop's own number and not a Python multiplication.
        """
        mp = self._mass_props(structure, STRUCTURE_DENSITY, f"{label} Mass Properties")
        cg = mp.prop("center of gravity")
        pm = mp.prop("principal moments")
        self._emit(f"{prefix}_volume_structure", mp.prop("volume"), {"length": 3})
        self._emit(f"{prefix}_mass_structure", mp.prop("mass"), {"mass": 1})
        for key, comp in zip(CG_COMPONENTS, ("x", "y", "z")):
            self._emit(f"{prefix}_{key}", cg.prop(comp), {"length": 1})
        for key, comp in zip(INERTIA_COMPONENTS, ("x", "y", "z")):
            self._emit(f"{prefix}_{key}", pm.prop(comp), {"length": 2, "mass": 1})

    def _base_area(self, diameter: Ref, name: str) -> Ref:
        """pi/4 d^2, computed in nTop so it tracks the input exactly."""
        d2 = self._mul(diameter, diameter)
        return self._mul(d2, self.r.literal_real(0.25 * math.pi, {}), name=name)

    def measure_stage2(self) -> None:
        g, i = self.g, self.inp
        s2 = self.dv.payload_stage.index

        mp_oml = self._mass_props(g["oml2"], 1.0, "S2 OML Mass Properties")
        self._emit("s2_volume_total", mp_oml.prop("volume"), {"length": 3})
        self._emit("s2_x_forward", g["x_s2_fwd"], {"length": 1})
        self._emit("s2_length", i[f"s{s2}.L"], {"length": 1})

        area_base = self._base_area(i[f"s{s2}.D"], "S2 Base Area")
        self._emit("s2_area_base", area_base, {"length": 2})
        # The flat base disc is removed so the number is the LATERAL area, matching what
        # `masses_iv1.stage_geometry` calls `area_wetted_body`. The nose is closed by the ogive,
        # so there is no forward disc to remove. SOURCES["iv1_stage1_two_base_discs"].
        area_oml = self._area(g["oml2"], "S2 OML Surface Area")
        self._emit("s2_area_wetted_body", self._sub(area_oml, area_base, name="S2 Wetted Body"),
                   {"length": 2})

        # Exposed panels: the part of each plate set OUTSIDE the body. The fins also have the
        # strakes subtracted, so that a design point where a strake and a fin overlap cannot
        # double count the shared volume. The strakes are subtracted from the body only.
        if "strakes" in g:
            exposed = self._subtract(g["strakes"], [g["oml2"]], name="Strakes Exposed")
            self.g["strakes_exposed"] = exposed
            self._emit("s2_area_wetted_strakes", self._area(exposed, "Strake Wetted Area"),
                       {"length": 2})
            mp = self._mass_props(exposed, 1.0, "Strake Mass Properties")
            self._emit("s2_volume_strakes", mp.prop("volume"), {"length": 3})
        if f"fins{s2}" in g:
            others = [g["oml2"]] + ([g["strakes"]] if "strakes" in g else [])
            exposed = self._subtract(g[f"fins{s2}"], others, name="S2 Fins Exposed")
            self._emit("s2_area_wetted_fins", self._area(exposed, "S2 Fin Wetted Area"),
                       {"length": 2})
            mp = self._mass_props(exposed, 1.0, "S2 Fin Mass Properties")
            self._emit("s2_volume_fins", mp.prop("volume"), {"length": 3})
        if "cavity2" in g:
            mp = self._mass_props(g["cavity2"], 1.0, "S2 Cavity Mass Properties")
            self._emit("s2_volume_cavity", mp.prop("volume"), {"length": 3})
        if "structure2" in g:
            self._measure_structure("s2", g["structure2"], "S2 Structure")

    def measure_booster(self) -> None:
        g, i = self.g, self.inp
        s1 = self.dv.booster.index

        mp_oml = self._mass_props(g["oml1"], 1.0, "S1 OML Mass Properties")
        self._emit("s1_volume_total", mp_oml.prop("volume"), {"length": 3})
        self._emit("s1_x_forward", g["x_s1_fwd"], {"length": 1})
        self._emit("s1_length", i[f"s{s1}.L"], {"length": 1})

        area_base = self._base_area(i[f"s{s1}.D"], "S1 Base Area")
        self._emit("s1_area_base", area_base, {"length": 2})
        # BOTH end discs come off: the aft one is the nozzle-exit base and the forward one is
        # covered by the interstage. SOURCES["iv1_stage1_two_base_discs"].
        area_oml = self._area(g["oml1"], "S1 OML Surface Area")
        lateral = self._sub(area_oml, self._scale(area_base, 2.0), name="S1 Wetted Body")
        self._emit("s1_area_wetted_body", lateral, {"length": 2})

        if f"fins{s1}" in g:
            exposed = self._subtract(g[f"fins{s1}"], [g["oml1"]], name="S1 Fins Exposed")
            self._emit("s1_area_wetted_fins", self._area(exposed, "S1 Fin Wetted Area"),
                       {"length": 2})
            mp = self._mass_props(exposed, 1.0, "S1 Fin Mass Properties")
            self._emit("s1_volume_fins", mp.prop("volume"), {"length": 3})
        if "cavity1" in g:
            mp = self._mass_props(g["cavity1"], 1.0, "S1 Cavity Mass Properties")
            self._emit("s1_volume_cavity", mp.prop("volume"), {"length": 3})
        if "structure1" in g:
            self._measure_structure("s1", g["structure1"], "S1 Structure")

    def measure_interstage(self) -> None:
        g, i = self.g, self.inp
        s2, s1 = self.dv.payload_stage.index, self.dv.booster.index
        mp_oml = self._mass_props(g["is_solid"], 1.0, "Interstage OML Mass Properties")
        self._emit("is_volume_total", mp_oml.prop("volume"), {"length": 3})
        self._emit("is_x_forward", g["x_is_fwd"], {"length": 1})
        self._emit("is_length", i["L_interstage"], {"length": 1})
        # Lateral area only: both flat end discs are covered by the stages they join. This costs
        # almost nothing - `surface_area<implicit,real>` on the booster's `cylinder` primitive
        # took 0.27 s against 24.6 s on the revolved ogive - so the stack's total wetted area can
        # be complete rather than missing the interstage.
        area = self._area(g["is_solid"], "Interstage Surface Area")
        ends = self._add(self._base_area(i[f"s{s2}.D"], "Interstage Forward Disc"),
                         self._base_area(i[f"s{s1}.D"], "Interstage Aft Disc"))
        self._emit("is_area_wetted_body", self._sub(area, ends, name="Interstage Wetted Body"),
                   {"length": 2})
        self._measure_structure("is", g["is_structure"], "Interstage Structure")

    def measure_stack(self, stacked: Ref) -> None:
        """The stacked assembly. Volume only: the expensive area blocks are per stage.

        `st_volume_total` is a genuine cross-check rather than a Python sum, because the union
        also carries the plate volume that sits outside the three outer mould lines.
        """
        g = self.g
        mp = self._mass_props(stacked, 1.0, "Stack OML Mass Properties")
        self._emit("st_volume_total", mp.prop("volume"), {"length": 3})
        self._emit("st_length", g["x_s1_aft"], {"length": 1})
        self._emit("st_x_forward", g["x_s2_fwd"], {"length": 1})

    def build_area_distribution(self, target: Ref) -> None:
        """Cross-section area S(x) of the STACK, for the wave-drag model, if asked for.

        There is no single block. The route is `extract_section<implicit,plane,real>` at each
        station, then a cross-section area on the resulting 2D region with `SECTION_AREA_BLOCK`.
        WP4 measured it accurate to 0.13 percent on SV-1 (NTOP_NOTES.md section 24).

        OFF by default, on cost: about 0.9 s per station on SV-1, and the sizing loop calls this
        tens of times. Stations sit at `(j + 0.5) / n` of the stack length so they follow the
        inputs and never land on the degenerate sections at either end.
        """
        n = self.area_stations
        if n <= 0:
            return
        r, g = self.r, self.g
        zero = self._zero()
        for j in range(n):
            frac = (j + 0.5) / n
            x = self._mul(g["x_s1_aft"], r.literal_real(frac, {}))
            plane = r.block(
                "plane<point,vector,vector>", self._point(x, zero, zero),
                r.literal_vector(0.0, 1.0, 0.0), r.literal_vector(0.0, 0.0, 1.0),
                name=f"Section Plane {j:02d}",
            )
            func = r.latest("extract_section<implicit,plane,real>")
            section = r.block(func, target, plane, self.section_feature_size,
                              name=f"Section {j:02d}")
            # SECTION_AREA_BLOCK, not `surface_area<implicit_2d,real>`: the vendored universe's
            # deprecation flags are backwards for the implicit_2d overload and every revision of
            # `surface_area<implicit_2d,real>` is rejected at load time. NTOP_NOTES.md sec. 24.
            area = r.raw_block(SECTION_AREA_BLOCK, "real",
                               [section, r.literal_real(self.area_rel, {})],
                               name=f"Section Area {j:02d}")
            self._emit(f"st_area_section_{j:02d}", area, {"length": 2})
            self._emit(f"st_station_{j:02d}", x, {"length": 1})

    # ---- exports -------------------------------------------------------------------------

    def build_exports(self, stacked: Ref) -> None:
        """Export the STACKED assembly, so a reader sees the whole vehicle in one file."""
        r, i = self.r, self.inp
        if self.export_stl:
            mesh = r.mesh_from_implicit(stacked, tolerance=i[MESH_TOLERANCE_INPUT],
                                        name="Export Mesh")
            r.block(r.latest("export_mesh<file_path,mesh,unit_length_enum>"),
                    i[STL_PATH_INPUT], mesh, r.literal_unit_length("m"), name="Export STL")
        if self.export_step:
            # NTOP_NOTES.md section 9: there is no implicit -> part block. The working chain is
            # cad_body_from_implicit_body -> brep, then .prop("part") -> part, then export_part.
            brep = r.block("cad_body_from_implicit_body<implicit,real,list<brep>>",
                           stacked, self.cad_tolerance, None, name="CAD Body")
            part = r.variable("CAD Part", brep.prop("part"))
            r.block(r.latest("export_part<file_path,part>"), i[STEP_PATH_INPUT], part,
                    name="Export STEP")
        if self.export_implicit:
            r.block(r.latest("export_implicit_body<file_path,implicit>"),
                    i[IMPLICIT_PATH_INPUT], stacked, name="Export Implicit")

    # ---- assembly -------------------------------------------------------------------------

    def build(self) -> Recipe:
        dv = self.dv
        s2, s1 = dv.payload_stage, dv.booster
        want = self.build_stage

        self.declare_inputs()
        self.derive()
        self.build_stage2_oml()

        outer2 = [self.g["oml2"]]
        if want in ("s2_plates", "s2_hollow", "booster", "full"):
            strakes = self.build_strakes()
            if strakes is not None:
                outer2.append(strakes)
            fins2 = self.build_fins(s2, self.g["x_s2_aft"], self.g["R2"], "S2 Fins")
            if fins2 is not None:
                outer2.append(fins2)
        solid2 = (self.g["oml2"] if len(outer2) == 1
                  else self._union(outer2, name="S2 Outer Solid"))
        self.g["solid2"] = solid2

        if want in ("s2_hollow", "booster", "full"):
            cavity2 = self.build_stage2_cavity()
            # THIS IS THE BODY WHOSE MASS IS REPORTED for stage 2: the wall, the two bulkheads,
            # the strake panels and the fin panels. No motor case, no propellant, no payload,
            # no avionics. See the module docstring.
            self.g["structure2"] = self._subtract(solid2, [cavity2], name="S2 Structure")

        stacked_parts = list(outer2)
        if want in ("booster", "full"):
            self.build_booster_oml()
            outer1 = [self.g["oml1"]]
            fins1 = self.build_fins(s1, self.g["x_s1_aft"], self.g["R1"], "S1 Fins")
            if fins1 is not None:
                outer1.append(fins1)
            solid1 = (self.g["oml1"] if len(outer1) == 1
                      else self._union(outer1, name="S1 Outer Solid"))
            self.g["solid1"] = solid1
            cavity1 = self.build_booster_cavity()
            self.g["structure1"] = self._subtract(solid1, [cavity1], name="S1 Structure")
            stacked_parts.extend(outer1)

        if want == "full":
            solid_is, _ = self.build_interstage()
            stacked_parts.append(solid_is)

        stacked = (stacked_parts[0] if len(stacked_parts) == 1
                   else self._union(stacked_parts, name="Stacked Assembly"))
        self.g["stacked"] = stacked

        self.measure_stage2()
        if want in ("booster", "full"):
            self.measure_booster()
        if want == "full":
            self.measure_interstage()
        self.measure_stack(stacked)
        self.build_area_distribution(stacked)
        self.build_exports(stacked)
        self.r.json_output(self.values, name="Measurements")
        return self.r


def build_stack_recipe(
    dv: StackDesignVector,
    out_dir: str,
    *,
    mesh_tolerance: float = DEFAULT_MESH_TOLERANCE,
    export_stl: bool = False,
    export_step: bool = False,
    export_implicit: bool = False,
    area_stations: int = 0,
    n_ogive: int = N_OGIVE,
    relative_error: float = DEFAULT_RELATIVE_ERROR,
    area_relative_error: float = DEFAULT_AREA_RELATIVE_ERROR,
    cad_tolerance: float = DEFAULT_CAD_TOLERANCE,
    section_feature_size: float = SECTION_FEATURE_SIZE,
    build_stage: str = "full",
) -> Recipe:
    """Author the IV-1 recipe for `dv`, with every dimension exposed as an nTop input.

    `dv` supplies the DEFAULT value of every notebook input, plus the topology choices that
    cannot be inputs. Those are `nose_shape`, `StageSpec.n_fin`, `StrakeSpec.n`, `n_ogive`,
    `area_stations` and which exports exist; everything else in `NTOP_STAGE_INPUTS` and
    `NTOP_GLOBAL_INPUTS` is a real notebook input, so the same converted `.ntop` measures any
    other design point.

    `out_dir` is where the export defaults point, so a bare `ntopcl convert` - which EVALUATES
    the notebook, NTOP_NOTES.md section 3 - writes its artefacts somewhere sensible.

    EXPORTS ARE OFF BY DEFAULT, unlike `build_rocket_recipe`. The IV-1 bounding box is 1.6x
    SV-1's, so the mesh costs 1.6x more at the same tolerance, and this entry point is also the
    one `measure_stack` uses: two entry points disagreeing about the default would convert the
    same geometry twice.

    `build_stage` truncates the build for debugging: "s2_oml" stops after the revolved payload
    stage, "s2_plates" adds the strakes and stage-2 fins, "s2_hollow" adds the cavity and the
    measured structure, "booster" adds the whole booster, "full" adds the interstage. Use
    "full".
    """
    b = _Builder(
        dv,
        n_ogive=n_ogive,
        relative_error=relative_error,
        area_relative_error=area_relative_error,
        mesh_tolerance=mesh_tolerance,
        cad_tolerance=cad_tolerance,
        export_stl=export_stl,
        export_step=export_step,
        export_implicit=export_implicit,
        default_dir=out_dir,
        area_stations=area_stations,
        section_feature_size=section_feature_size,
        build_stage=build_stage,
    )
    return b.build()


# --------------------------------------------------------------------------------------
#   The notebook cache
# --------------------------------------------------------------------------------------


@dataclass
class StackNotebook:
    """A converted `.ntop` plus the input template needed to drive it."""

    path: str
    input_template: dict[str, Any]
    key: str
    recipe_json: str
    convert_wall_time_s: float = 0.0
    reused: bool = False

    def input_names(self) -> list[str]:
        return [str(d.get("name")) for d in self.input_template.get("inputs", [])]


DEFAULT_CACHE_DIR = os.path.join(RUNS_DIR, "_ntop_cache")

# Process-local cache: key -> StackNotebook. Saves the `-t` call as well as the convert.
_MEMO: dict[str, StackNotebook] = {}


def _topology_key(
    dv: StackDesignVector,
    *,
    n_ogive: int,
    relative_error: float,
    area_relative_error: float,
    export_stl: bool,
    export_step: bool,
    export_implicit: bool,
    cad_tolerance: float,
    area_stations: int,
    section_feature_size: float,
    build_stage: str,
) -> str:
    """Hash of everything that changes the BLOCK GRAPH, and nothing that does not.

    Dimensions are absent on purpose: they are notebook inputs, so a new design point reuses
    the same `.ntop`. `mesh_tolerance` and the export PATHS are absent for the same reason.
    Whether an export EXISTS is present, because that really does add blocks.
    """
    payload = {
        "version": RECIPE_VERSION,
        "n_stages": dv.n_stages,
        "stage_indices": [s.index for s in dv.stages],
        "n_fin": [int(s.n_fin) for s in dv.stages],
        "n_strake": int(dv.strakes.n),
        "nose_shape": str(dv.nose_shape),
        "n_ogive": int(n_ogive),
        "relative_error": float(relative_error),
        "area_relative_error": float(area_relative_error),
        "export_stl": bool(export_stl),
        "export_step": bool(export_step),
        "export_implicit": bool(export_implicit),
        "cad_tolerance": float(cad_tolerance),
        "area_stations": int(area_stations),
        "section_feature_size": float(section_feature_size),
        "build_stage": str(build_stage),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def clear_stack_notebook_cache(cache_dir: str | None = None) -> None:
    """Forget every cached IV-1 notebook. Tests use this to force a re-`convert`."""
    _MEMO.clear()
    if cache_dir and os.path.isdir(cache_dir):
        for f in os.listdir(cache_dir):
            if f.startswith("iv1_") and f.endswith((".ntop", ".json", ".log")):
                try:
                    os.remove(os.path.join(cache_dir, f))
                except OSError:                                     # pragma: no cover
                    pass


def build_stack_notebook(
    dv: StackDesignVector,
    run_dir: str,
    runner: NtopRunner | None = None,
    *,
    cache_dir: str | None = None,
    force: bool = False,
    convert_timeout: float = 3600.0,
    **kw: Any,
) -> str:
    """Convert (or reuse) the IV-1 `.ntop` for the topology `dv` implies. Returns its path.

    Cached under `cache_dir` (default `runs/_ntop_cache`) on a topology key, because
    `ntopcl convert` evaluates the whole notebook and is therefore as expensive as a run:
    convert once, run many times (NTOP_NOTES.md section 13 point 5).

    The export defaults match `measure_stack`'s - all OFF. They have to: an export changes the
    block graph, so it is part of the topology key.
    """
    return _notebook(dv, run_dir, runner, cache_dir=cache_dir, force=force,
                     convert_timeout=convert_timeout, **kw).path


def _notebook(
    dv: StackDesignVector,
    run_dir: str,
    runner: NtopRunner | None = None,
    *,
    cache_dir: str | None = None,
    force: bool = False,
    convert_timeout: float = 3600.0,
    **kw: Any,
) -> StackNotebook:
    opts = dict(
        n_ogive=kw.pop("n_ogive", N_OGIVE),
        relative_error=kw.pop("relative_error", DEFAULT_RELATIVE_ERROR),
        area_relative_error=kw.pop("area_relative_error", DEFAULT_AREA_RELATIVE_ERROR),
        # Defaults MUST match `measure_stack`, or the two would key different notebooks and
        # convert twice for the same geometry.
        export_stl=kw.pop("export_stl", False),
        export_step=kw.pop("export_step", False),
        export_implicit=kw.pop("export_implicit", False),
        cad_tolerance=kw.pop("cad_tolerance", DEFAULT_CAD_TOLERANCE),
        area_stations=kw.pop("area_stations", 0),
        section_feature_size=kw.pop("section_feature_size", SECTION_FEATURE_SIZE),
        build_stage=kw.pop("build_stage", "full"),
    )
    mesh_tolerance = kw.pop("mesh_tolerance", DEFAULT_MESH_TOLERANCE)
    if kw:
        raise TypeError(f"unexpected keyword arguments: {sorted(kw)}")

    key = _topology_key(dv, **opts)
    cdir = os.path.abspath(cache_dir or DEFAULT_CACHE_DIR)
    ntop_path = os.path.join(cdir, f"iv1_{key}.ntop")
    recipe_json = os.path.join(cdir, f"iv1_{key}_recipe.json")

    memo = _MEMO.get(key)
    if memo is not None and not force and os.path.isfile(memo.path):
        log.info("reusing cached notebook %s", memo.path)
        return StackNotebook(memo.path, memo.input_template, key, memo.recipe_json,
                             memo.convert_wall_time_s, reused=True)

    os.makedirs(cdir, exist_ok=True)
    run = runner if runner is not None else NtopRunner()

    if force or not os.path.isfile(ntop_path) or os.path.getsize(ntop_path) == 0:
        recipe = build_stack_recipe(dv, cdir, mesh_tolerance=mesh_tolerance, **opts)
        recipe.write_json(recipe_json)
        t0 = time.perf_counter()
        run.convert(recipe_json, ntop_path, timeout=convert_timeout)
        dt = time.perf_counter() - t0
        log.info("converted %s in %.1f s", ntop_path, dt)
        reused = False
    else:
        dt = 0.0
        reused = True

    input_template, output_template = run.templates(ntop_path, out_dir=cdir,
                                                    require_output=True)
    if output_template is None:                                    # pragma: no cover
        raise NtopError(f"{ntop_path} designates no Automate output")
    nb = StackNotebook(ntop_path, input_template, key, recipe_json, dt, reused=reused)
    _MEMO[key] = nb
    return nb


# --------------------------------------------------------------------------------------
#   The measurement entry point
# --------------------------------------------------------------------------------------


def _input_payload(
    dv: StackDesignVector,
    template: Mapping[str, Any],
    *,
    mesh_tolerance: float,
    stl_path: str | None,
    step_path: str | None,
    implicit_path: str | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """The `-j` values and their explicit display units, for one design point."""
    accepted = {str(d.get("name")) for d in template.get("inputs", [])}
    values: dict[str, Any] = {}
    units: dict[str, str] = {}

    def put(iname: str, value: float, dim: Mapping[str, int]) -> None:
        if iname not in accepted:                                  # pragma: no cover
            return
        values[iname] = float(value)
        u = _display_unit(dim)
        if u:
            units[iname] = u

    for stage in dv.stages:
        for attr, suffix, dim in NTOP_STAGE_INPUTS:
            put(_stage_input_name(stage.index, suffix), float(getattr(stage, attr)), dim)
    for path, iname, dim in NTOP_GLOBAL_INPUTS:
        put(iname, float(_dotted(dv, path)), dim)
    put(MESH_TOLERANCE_INPUT, float(mesh_tolerance), {"length": 1})

    for iname, path in ((STL_PATH_INPUT, stl_path), (STEP_PATH_INPUT, step_path),
                        (IMPLICIT_PATH_INPUT, implicit_path)):
        if iname in accepted and path:
            values[iname] = to_ntop_path(path)
    return values, units


def _split_by_prefix(raw: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """Every `<prefix>_<name>` entry, with the prefix stripped."""
    head = prefix + "_"
    return {k[len(head):]: v for k, v in raw.items() if k.startswith(head)}


def _triple(values: Mapping[str, Any], names: Iterable[str]) -> tuple[float, float, float] | None:
    out: list[float] = []
    for n in names:
        v = values.get(n)
        if v is None:
            return None
        out.append(float(v))
    return (out[0], out[1], out[2])


def _body_measurements(
    raw: Mapping[str, Any], prefix: str, stage_index: int
) -> StageMeasurements:
    """Assemble one body's `StageMeasurements` from the prefixed output names."""
    vals = _split_by_prefix(raw, prefix)
    m = StageMeasurements(stage_index=stage_index, body=BODY_PREFIXES.get(prefix, prefix))
    # `driver.measurements_from_names` does the OUTPUT_NAME_MAP lookup and the field casting,
    # so this module never has to duplicate that table.
    measurements_from_names(vals, target=m)

    # Fields that are NOT on `NtopMeasurements` and therefore cannot be registered.
    for name, attr in (("area_wetted_strakes", "area_wetted_strakes"),
                       ("volume_strakes", "volume_strakes"),
                       ("volume_fins", "volume_fins"),
                       ("x_forward", "x_forward"),
                       ("length", "length")):
        v = vals.get(name)
        if v is not None:
            setattr(m, attr, float(v))

    # Vectors travel as three named scalars (NTOP_NOTES.md section 13 point 2).
    cg = _triple(vals, CG_COMPONENTS)
    if cg is not None:
        m.cg_structure_stack = cg
        # `masses_iv1.build_stack_masses` reads cg_structure[0] as a station from THIS STAGE's
        # forward face and adds the stage offset itself, so the reported CG is stage-local.
        x0 = float(vals.get("x_forward") or 0.0)
        m.cg_structure = (cg[0] - x0, cg[1], cg[2])
    inertia = _triple(vals, INERTIA_COMPONENTS)
    if inertia is not None:
        m.inertia_structure = inertia

    rows: list[tuple[float, float]] = []
    for j in range(1000):
        x = vals.get(f"station_{j:02d}")
        a = vals.get(f"area_section_{j:02d}")
        if x is None or a is None:
            break
        rows.append((float(x), float(a)))
    if rows:
        m.area_distribution = sorted(rows)
    return m


def _sum_or_none(parts: Sequence[float | None]) -> float | None:
    vals = [p for p in parts if p is not None]
    return sum(vals) if vals else None


def _stacked_measurements(
    raw: Mapping[str, Any], bodies: Sequence[StageMeasurements]
) -> StageMeasurements:
    """Key 0: the whole stack, INCLUDING the interstage.

    `volume_total` and the area distribution are nTop's own numbers, measured on the union.
    Everything else is a SUM over the three bodies, done in Python, and a warning says so:
    CLAUDE.md 3.3 - approximations are recorded, never swallowed. Summing is exact for volumes
    and areas of disjoint bodies, which these are, so nothing is lost but the provenance, and the
    warning carries that.
    """
    m = _body_measurements(raw, "st", 0)
    m.body = BODY_PREFIXES["st"]
    bodies = list(bodies)
    m.volume_structure = _sum_or_none([b.volume_structure for b in bodies])
    m.volume_cavity = _sum_or_none([b.volume_cavity for b in bodies])
    m.mass_structure = _sum_or_none([b.mass_structure for b in bodies])
    m.area_wetted_body = _sum_or_none([b.area_wetted_body for b in bodies])
    m.area_wetted_fins = _sum_or_none([b.area_wetted_fins for b in bodies])
    m.area_wetted_strakes = _sum_or_none([b.area_wetted_strakes for b in bodies])
    m.volume_strakes = _sum_or_none([b.volume_strakes for b in bodies])
    m.volume_fins = _sum_or_none([b.volume_fins for b in bodies])
    # The base of the STACK is the booster's aft face.
    aft = max(bodies, key=lambda b: b.x_forward or 0.0) if bodies else None
    if aft is not None:
        m.area_base = aft.area_base
    m.warnings.append(
        "stack (key 0): volume_total and the area distribution are measured by nTop on the "
        "union; volume_structure, volume_cavity, mass_structure and the wetted areas are "
        "PYTHON SUMS over the per-body measurements, not separate nTop measurements"
    )
    return m


def measure_stack(
    dv: StackDesignVector,
    run_dir: str,
    runner: NtopRunner | None = None,
    *,
    mesh_tolerance: float = DEFAULT_MESH_TOLERANCE,
    export_stl: bool = False,
    export_step: bool = False,
    export_implicit: bool = False,
    cache_dir: str | None = None,
    force_convert: bool = False,
    timeout: float = 3600.0,
    verbose: int = 2,
    tag: str | None = None,
    **kw: Any,
) -> dict[int, StageMeasurements]:
    """Build (or reuse) the IV-1 notebook, run it at `dv`, and return the measurements.

    The return value is keyed by STAGE INDEX:

    * `1` - the booster, jettisoned at separation
    * `2` - the payload stage that reaches intercept
    * `0` - the stacked assembly, including the interstage
    * `-1` - the interstage alone, which is jettisoned WITH stage 1 and whose mass therefore
      has to be added to the stage-1 jettisoned mass by the caller

    `StageMeasurements` subclasses `config.NtopMeasurements`, so this satisfies
    `dict[int, NtopMeasurements]` and feeds `masses_iv1.build_stack_masses(meas=...)` and
    `aero_iv1.StackAero(meas=...)` with no adapter. The extra IV-1 fields are documented on
    `StageMeasurements`; the important one is `area_wetted_strakes`, kept apart from
    `area_wetted_fins`.

    A design vector that `StackDesignVector.geometry_is_valid()` rejects raises `ValueError`
    immediately, before an `ntopcl` subprocess is spent on it. A geometry that nTop itself
    cannot build raises `NtopError` carrying the captured diagnostics.

    EXPORTS ARE ALL OFF BY DEFAULT, which is measured and not lazy. See
    SOURCES["iv1_mesh_tolerance"]. Turn them on for the converged design point:

        measure_stack(dv, run_dir, export_stl=True, export_step=True)
    """
    ok, errs = dv.geometry_is_valid()
    if not ok:
        raise ValueError(
            "invalid design vector, refusing to spend an ntopcl call: " + "; ".join(errs)
        )

    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    stem = tag or "iv1"

    nb = _notebook(
        dv, run_dir, runner,
        cache_dir=cache_dir, force=force_convert,
        mesh_tolerance=mesh_tolerance,
        export_stl=export_stl, export_step=export_step, export_implicit=export_implicit,
        **kw,
    )
    run = runner if runner is not None else NtopRunner()

    stl_path = os.path.join(run_dir, f"{stem}.stl") if export_stl else None
    step_path = os.path.join(run_dir, f"{stem}.step") if export_step else None
    implicit_path = os.path.join(run_dir, f"{stem}.implicit") if export_implicit else None

    values, units = _input_payload(
        dv, nb.input_template, mesh_tolerance=mesh_tolerance,
        stl_path=stl_path, step_path=step_path, implicit_path=implicit_path,
    )
    expect = [p for p in (stl_path, step_path, implicit_path) if p]
    out_json = os.path.join(run_dir, f"{stem}_output.json")

    result = run.run(
        nb.path, values,
        out_json=out_json, expect=expect,
        input_template=nb.input_template, units=units,
        input_json=os.path.join(run_dir, f"{stem}_input.json"),
        run_dir=run_dir, timeout=timeout, verbose=verbose,
    )

    # NOT `parsed.measurements`: see the CAVEAT in the module docstring. The flat object mixes
    # the three bodies onto one set of fields, so only `parsed.raw` is used.
    parsed = parse_outputs(out_json, run=result)
    raw = parsed.raw

    out: dict[int, StageMeasurements] = {}
    for stage in dv.stages:
        prefix = f"s{stage.index}"
        if prefix not in BODY_PREFIXES:                            # pragma: no cover
            raise NtopError(
                f"stage index {stage.index} has no output prefix; this notebook supports "
                f"stage indices {sorted(int(p[1:]) for p in BODY_PREFIXES if p[0] == 's')}"
            )
        out[stage.index] = _body_measurements(raw, prefix, stage.index)

    interstage = _body_measurements(raw, "is", -1)
    interstage.body = BODY_PREFIXES["is"]
    interstage.warnings.append(
        "the interstage is jettisoned WITH stage 1; add its mass_structure to the stage-1 "
        "jettisoned mass"
    )
    out[-1] = interstage
    # The interstage IS part of the stack at launch, so it is in the key-0 sums. It leaves with
    # stage 1 at separation, which is why it also has its own key.
    out[0] = _stacked_measurements(raw, [out[s.index] for s in dv.stages] + [interstage])

    for key, m in out.items():
        m.ntop_path = nb.path
        m.wall_time_s = result.wall_time_s
        m.ntopcl_returncode = result.returncode
        if key == 0:
            m.stl_path = stl_path
            m.step_path = step_path
            m.implicit_path = implicit_path
        if not m.area_distribution and key == 0:
            m.warnings.append(
                "area_distribution is empty: the notebook was built with area_stations = 0. "
                "The aero model falls back to closed-form cross-section geometry."
            )
        if key > 0 and not m.is_usable():
            missing = [
                f for f in ("volume_total", "volume_cavity", "area_wetted_body",
                            "mass_structure")
                if getattr(m, f) is None
            ]
            m.warnings.append("nTop did not report: " + ", ".join(missing))

    parsed.to_json(os.path.join(run_dir, f"{stem}_measurements.json"))
    with open(os.path.join(run_dir, f"{stem}_stages.json"), "w", encoding="utf-8") as f:
        from dataclasses import asdict as _asdict
        json.dump({str(k): _asdict(v) for k, v in sorted(out.items())}, f, indent=2,
                  default=str)
    return out


def geometry_fn(**kw: Any) -> Any:
    """A `measure_stack` with its options pinned, for the IV-1 sizing loop.

        from rocketgen.ntopgen.stack_notebook import geometry_fn
        size_stack(dv0, reqs, geometry_fn=geometry_fn(export_stl=False), run_dir=...)

    MEASURED COST, so the choice is informed rather than guessed: see
    SOURCES["measured_wall_time"] and the report in `docs/NTOP_NOTES.md`. The five
    `surface_area<implicit,real>` calls dominate. A sizing loop that must be faster should drop
    `area_wetted_fins`, whose planform the aero model can get exactly from the design vector,
    rather than coarsen anything.
    """
    from functools import partial

    return partial(measure_stack, **kw)
