"""WP4: the parametric SV-1 rocket notebook, authored programmatically for nTop Automate.

What this builds
---------------
An implicit solid of the SV-1 concept described in `SPEC.md` section 2:

* **Outer mould line (OML)** - ONE closed 2D polygon in the XY plane, revolved 360 degrees
  about the X axis. The polygon walks the surface from the nose tip aft (tangent-ogive nose,
  cylindrical mid-body, conical boattail, flat base) and returns along the axis. A single
  revolve is cheaper and topologically cleaner than booleaning three primitives, and it makes
  the OML exactly one implicit body whose volume and area can be measured directly.
* **Fins** - `n_fin` = 4 planar panels in a cruciform at 0, 90, 180 and 270 degrees about the
  X axis. Each panel is a **CONSTANT-THICKNESS TAPERED PLATE**: the trapezoidal planform
  (swept leading edge, tapered chord) extruded to thickness `t_fin`. It is NOT a double wedge.
  The double wedge was built first and measured wrong - see `_Builder._panel` for the numbers
  and the reason. `rocketgen/sizing/aero.py` still assumes a double-wedge section for its
  supersonic fin wave drag, which is a stated fidelity mismatch: the plate's planform, span,
  sweep, taper and wetted area are exact, and only the section volume differs (a plate holds
  twice a diamond of the same maximum thickness, so fin structural mass is conservative).
  Panels 3 and 4 are mirrors of panels 1 and 2, which avoids needing a rotation block that
  returns `any`.
* **Hollow airframe** - the interior is `offset_implicit(OML, -t_wall)`, i.e. the OML offset
  inward by the wall thickness, so the airframe is a real thin-wall structure and not a solid
  billet. Three ring bulkheads (discs of thickness `t_wall`) are subtracted from that interior
  at the seeker/guidance, guidance/warhead and warhead/motor stations, so the four bays
  (seeker, guidance, warhead, motor) are real separated cavities.

WHAT THE MEASURED STRUCTURE IS, AND WHAT IT IS NOT
--------------------------------------------------
`volume_structure` and `mass_structure` are the **AIRFRAME AND FINS ONLY**: the wall, the
bulkheads and the fin panels. They deliberately do NOT include

* the motor case or its insulation - `rocketgen/sizing/masses.py` charges those separately,
  from the `SolidMotor` model (see `MassBuildup` entries "Motor case", "Motor insulation");
* the propellant grain - charged from `dv.m_p_boost + dv.m_p_sustain`;
* the warhead - charged from `Requirements.m_warhead` (SPEC R4);
* the guidance, seeker and actuation package - charged from `Requirements.m_guidance` (R5);
* the nozzle and igniter.

Double counting any of those would make the whole rocket heavier than it is and would corrupt
the sizing loop. The sanity check is scale: a SOLID billet of 7075-T6 filling the default
0.3355 m^3 OML would weigh 943 kg. A correct hollow airframe plus fins is tens of kilograms.
`tests/test_rocket_notebook.py` asserts 10 kg < mass_structure < 120 kg for exactly this
reason.

Performance
-----------
`docs/NTOP_NOTES.md` sections 3 and 4: `ntopcl convert` EVALUATES the notebook, and
`implicit_to_mesh` costs roughly tolerance^-3. So the design variables are real nTop notebook
**inputs**, the `.ntop` is built ONCE and cached by a topology key, and each design point is a
plain `-j` run against the cached notebook. Export paths and the mesh tolerance are inputs too,
so changing where the STL goes does not force a re-`convert`.

None of the measurements need a mesh: `mass_properties<implicit,...>` and
`surface_area<implicit,real>` work straight off the implicit and are far more accurate than
anything measured off an exported STL (0.0104 percent versus 0.169 percent on the WP1 smoke
sphere). The mesh exists only for the STL export.

Units are SI: metres, radians, kilograms (PLAN.md hard rule 4).
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

from ..config import (
    MATERIALS,
    RUNS_DIR,
    DesignVector,
    NtopMeasurements,
    register_sources,
)
from .driver import NtopError, NtopRunner, parse_outputs, register_output_names
from .recipe import Recipe, Ref, to_ntop_path

__all__ = [
    "NTOP_INPUTS",
    "WP4_OUTPUT_NAMES",
    "RocketNotebook",
    "build_rocket_recipe",
    "build_rocket_notebook",
    "measure_rocket",
    "geometry_fn",
    "clear_notebook_cache",
    "DEFAULT_MESH_TOLERANCE",
    "DEFAULT_CAD_TOLERANCE",
    "DEFAULT_RELATIVE_ERROR",
    "DEFAULT_AREA_RELATIVE_ERROR",
    "N_OGIVE_OUTER",
    "N_OGIVE_INNER",
    "SECTION_AREA_BLOCK",
    "SOURCES",
]

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
#   Empirical constants, with sources
# --------------------------------------------------------------------------------------

SOURCES: dict[str, str] = {
    "ogive_polygon_sampling": (
        "Measured, not guessed. The revolved OML is a chord polygon, so its volume is the "
        "frustum sum that `rocketgen.sizing.masses._tangent_ogive_volume` computes with the "
        "same n. Sweeping n against a 20000-segment reference on the default SV-1 gives a "
        "TOTAL OML volume error of -0.106 percent at n=8, -0.026 percent at n=16, "
        "-0.0117 percent at n=24, -0.0066 percent at n=32 and -0.0004 percent at n=128. "
        "n=24 is used for the outer profile: 85x inside the 1 percent gate in PLAN.md, and "
        "small enough to keep the recipe near 300 blocks."
    ),
    "ogive_polygon_sampling_inner": (
        "The inner (cavity) surface is an offset of the outer one, so it inherits the outer "
        "sampling. No separate inner polygon is built; see `offset_implicit` below."
    ),
    "area_relative_error": (
        "Measured on this machine, and the finding is negative. `surface_area<implicit,real>` "
        "is the most expensive block in the notebook: 19.9 s on the SV-1 body and 22.4 s on "
        "the fin set, against 2.1 s for `mass_properties` on the same body (per-block timings "
        "from ntopcl -v 2). Its 'Relative error' input was swept over 0.002, 0.01, 0.05 and "
        "0.2 and made NO difference to either the reported area (bit-identical 4.003007 m^2 "
        "at every target) or the wall time (about 15 s at every target). The input therefore "
        "does not buy speed on this build, and the whole-run cost of about 30 s is a floor set "
        "by the two area blocks, not by the mesh. 0.01 is kept because it is the value the "
        "same block was verified accurate to 0.0097 percent at on the WP1 smoke sphere "
        "(NTOP_NOTES.md section 11)."
    ),
    "mesh_tolerance": (
        "Measured on this machine. `implicit_to_mesh` drives a DENSE voxel grid, so cost and "
        "memory go as (bounding box volume)/tolerance^3. The SV-1 box is 4.0 x 0.71 x 0.71 m = "
        "2.02 m^3, which is 16000 times the WP1 smoke sphere's box, so the sphere's 1.0e-3 m "
        "working point does NOT carry over: at 1.5e-3 m the process passed 4.8 GB resident and "
        "had to be killed. Measured on the whole rocket (OML plus fins, convert wall time, "
        "which includes about 10 s of fixed overhead): 8.0e-3 m -> 28.2 s and a 3.6 MB STL; "
        "5.0e-3 m -> 51.7 s and a 9.2 MB STL; 3.0e-3 m had not finished at 128 s and was past "
        "1.4 GB. 5.0e-3 m is the default: it resolves the 12 mm fin thickness with "
        "about 2.4 cells and is the coarsest tolerance that still gives a presentable STL. The "
        "mesh is used ONLY for the STL export, never for a measurement."
    ),
    "cad_tolerance": (
        "Measured on this machine. `cad_body_from_implicit_body` is the STEP route "
        "(NTOP_NOTES.md section 9), where a 25 mm sphere at a 5.0e-4 m tolerance took 11.6 s. "
        "That does not scale to a 4 m part: at 2.0e-3 m on the whole SV-1 the process passed "
        "9 GB resident and had to be killed. Measured on the SV-1 body plus fins, convert wall "
        "time: 2.0e-2 m -> 11.6 s and a 1.15 MB STEP; 1.0e-2 m -> 22.8 s and a 8.96 MB STEP. "
        "1.0e-2 m is the default. It is coarse against a 3 mm wall, but the STEP carries only "
        "the OUTER solid, whose smallest feature is the 12 mm fin thickness, and no measurement "
        "is taken from it."
    ),
    "relative_error": (
        "`mass_properties<implicit,real_field,real>` and `surface_area<implicit,real>` take a "
        "relative-error target. 0.002 is used: on the WP1 smoke sphere the block reported "
        "volume to 0.0104 percent at 0.001, so 0.002 is comfortably inside the 1 percent gate "
        "while keeping the adaptive integration cheap."
    ),
    "fin_section_plate": (
        "Measured, not chosen. The fin section is a constant-thickness tapered plate, not the "
        "double wedge the aero model assumes. A double wedge WAS built, by lofting a root "
        "diamond profile to a tip diamond profile, and measured against the exact value: "
        "loft<implicit_2d,implicit_2d>[1.1.0] returned 2.747e-4 m^3 for one panel against an "
        "exact 3.343e-4 m^3, i.e. 82 percent, and 73 percent of the exact wetted area. The "
        "block is a linear MIX of two extruded signed-distance fields, not a linear "
        "interpolation of two boundaries, and nTop's own deprecation message says so. The "
        "plate is exact in planform and wetted area; its section volume is 2x a diamond of "
        "the same maximum thickness, so fin mass is conservative by about 4 kg on the SV-1."
    ),
    "fin_root_plug": (
        "The panel root sits at radius R - t_wall rather than exactly R, so the union with the "
        "airframe wall overlaps instead of touching tangentially. Tangential contact of two "
        "implicit bodies is numerically fragile. The leading edge and chord are extrapolated "
        "inboard along the same straight sweep and taper lines, so the chord at the true body "
        "surface y = R is still exactly c_r_fin."
    ),
}


# PLAN.md hard rule 2: every empirical constant is declared where the report can print it.
register_sources(SOURCES)


# `implicit_to_mesh` tolerance for the exported STL, metres. See SOURCES["mesh_tolerance"].
# NOT a measurement tolerance: every measured quantity comes off the implicit body directly.
DEFAULT_MESH_TOLERANCE = 5.0e-3

# `cad_body_from_implicit_body` tolerance for STEP export, metres.
# See SOURCES["cad_tolerance"]: 5.0e-4 m is the sphere-scale value from NTOP_NOTES.md section 9
# and does NOT scale to a 4 m part.
DEFAULT_CAD_TOLERANCE = 1.0e-2

# Relative-error target handed to `mass_properties`. See SOURCES["relative_error"].
DEFAULT_RELATIVE_ERROR = 0.002

# Relative-error target handed to `surface_area<implicit,real>`. It gets its own knob because
# it is by far the most expensive block in the notebook: at 0.002 on the SV-1 body it took
# 19.9 s, against 2.1 s for `mass_properties` on the same body (measured from the per-block
# timings in `runs/SV-1_geom/_probe/fins/ntopcl_run.log`). See SOURCES["area_relative_error"].
DEFAULT_AREA_RELATIVE_ERROR = 0.01

# Chord-polygon sample counts. See SOURCES["ogive_polygon_sampling"].
N_OGIVE_OUTER = 24
N_OGIVE_INNER = 12          # kept for the explicit-inner-profile fallback route

# nTop enum encodings. `blend_enum` 0 is the no-blend option; a boolean with no blend is what a
# sharp-edged airframe wants. REFERENCE.md section 5 documents the {"enum": N} encoding.
BLEND_NONE = 0

# Cross-section area of an `implicit_2d`. The vendored universe lists
# `surface_area<implicit_2d,real>[1.2.0]` as the current, non-deprecated block and
# `body_surface_area<implicit_2d,real>[1.1.0]` as deprecated. On both installed builds that is
# backwards: every revision of `surface_area<implicit_2d,real>` is REJECTED by
# `ntopcl convert` with "[E]: Error loading recipe:", and only the "deprecated" one loads.
# See `docs/NTOP_NOTES.md` section 24. Emitted with `Recipe.raw_block` so the universe's wrong
# deprecation flag does not get in the way.
SECTION_AREA_BLOCK = "body_surface_area<implicit_2d,real>[1.1.0]"

# `extract_section`'s optional "Min. Feature Size", metres. A section of a thin fin plate needs
# a feature size below the plate thickness or the section can come back empty.
SECTION_FEATURE_SIZE = 1.0e-3

STRUCTURE_DENSITY = MATERIALS["airframe_al7075"].density     # 2810 kg/m^3

# Recipe schema version. Bump when the authored topology changes, so cached notebooks from an
# older build are not reused.
RECIPE_VERSION = 4


# --------------------------------------------------------------------------------------
#   The nTop input contract
# --------------------------------------------------------------------------------------

# (DesignVector attribute, nTop notebook input name, dimension map).
#
# These are the design variables the sizing loop moves that change the SOLID's dimensions but
# not its topology, so one converted `.ntop` serves every design point.
#
# `n_fin` is NOT here: changing the number of panels changes the block graph, so it is baked
# into the notebook and forms part of the cache key. Only the cruciform n_fin == 4 (and the
# degenerate n_fin == 0, body only) are supported. `nose_shape` is baked in for the same
# reason. `m_p_boost`, `m_p_sustain`, `F_boost`, `eps_nozzle` and `p_c` are propulsion
# variables that do not appear in the geometry at all.
NTOP_INPUTS: tuple[tuple[str, str, dict[str, int]], ...] = (
    ("D",            "Body Diameter",       {"length": 1}),
    ("L_total",      "Overall Length",      {"length": 1}),
    ("f_nose",       "Nose Fineness",       {}),
    ("t_wall",       "Wall Thickness",      {"length": 1}),
    ("L_boattail",   "Boattail Length",     {"length": 1}),
    ("d_base",       "Base Diameter",       {"length": 1}),
    ("b_fin",        "Fin Semi Span",       {"length": 1}),
    ("c_r_fin",      "Fin Root Chord",      {"length": 1}),
    ("taper_fin",    "Fin Taper Ratio",     {}),
    ("sweep_fin",    "Fin Sweep",           {"angle": 1}),
    ("t_fin",        "Fin Thickness",       {"length": 1}),
    ("x_fin_te_gap", "Fin TE Gap",          {"length": 1}),
    ("L_seeker",     "Seeker Bay Length",   {"length": 1}),
    ("L_guidance",   "Guidance Bay Length", {"length": 1}),
    ("L_warhead",    "Warhead Bay Length",  {"length": 1}),
)

INPUT_NAME_BY_ATTR: dict[str, str] = {a: n for a, n, _ in NTOP_INPUTS}
INPUT_DIMENSION_BY_NAME: dict[str, dict[str, int]] = {n: d for _, n, d in NTOP_INPUTS}

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


# --------------------------------------------------------------------------------------
#   Output names the notebook emits
# --------------------------------------------------------------------------------------

# Scalar outputs, packed into the notebook's single `json` output slot (NTOP_NOTES.md
# section 1 and 13). `driver.parse_outputs` maps these onto `NtopMeasurements`.
WP4_OUTPUT_NAMES: tuple[str, ...] = (
    "volume_total",
    "volume_structure",
    "volume_cavity",
    "area_wetted_body",
    "area_wetted_fins",
    "area_base",
    "mass_structure",
)

# Vector-valued measurements have to be split into components, because `core.list<real>` (and
# therefore the json output) only carries scalars. NTOP_NOTES.md section 13 point 2.
CG_COMPONENT_NAMES = ("cg_structure_x", "cg_structure_y", "cg_structure_z")
INERTIA_COMPONENT_NAMES = (
    "inertia_structure_1",
    "inertia_structure_2",
    "inertia_structure_3",
)

# Additively extend `driver.OUTPUT_NAME_MAP`. WP4 owns the notebook and therefore owns these
# names; `config.py` is not touched (NTOP_NOTES.md section 13 point 3). The component names
# above are NOT registered, because `NtopMeasurements.cg_structure` is a 3-tuple field and has
# to be reassembled from three scalars; `_collect_vectors` below does that.
register_output_names(
    {
        # names this notebook actually emits
        "volume_total": "volume_total",
        "volume_structure": "volume_structure",
        "volume_cavity": "volume_cavity",
        "area_wetted_body": "area_wetted_body",
        "area_wetted_fins": "area_wetted_fins",
        "area_base": "area_base",
        "mass_structure": "mass_structure",
        # readable aliases, in case the notebook is inspected or edited in the GUI
        "volume_oml": "volume_total",
        "volume_airframe": "volume_structure",
        "volume_internal": "volume_cavity",
        "mass_airframe": "mass_structure",
        "area_fins": "area_wetted_fins",
    }
)


# --------------------------------------------------------------------------------------
#   Geometry helpers (pure Python, used for the closed-form cross-checks)
# --------------------------------------------------------------------------------------


def tangent_ogive_rho(length: float, radius: float) -> float:
    """Generating-circle radius of a tangent ogive: rho = (R^2 + L^2) / (2R)."""
    return (radius * radius + length * length) / (2.0 * radius)


def tangent_ogive_y(x: float, length: float, radius: float) -> float:
    """Tangent-ogive profile radius at station `x` from the tip.

    y(x) = sqrt(rho^2 - (L - x)^2) - (rho - R).
    """
    rho = tangent_ogive_rho(length, radius)
    inner = rho * rho - (length - x) ** 2
    return max(math.sqrt(max(inner, 0.0)) - (rho - radius), 0.0)


def ogive_sample_fractions(n: int) -> tuple[float, ...]:
    """The `u = x / L_nose` sample fractions of the chord polygon, excluding the ends.

    Uniform in `u`. A cosine clustering toward the tip was tried and is not needed: the
    uniform polygon already lands 85x inside the 1 percent volume gate at n = 24 (see
    SOURCES["ogive_polygon_sampling"]).
    """
    if n < 2:
        raise ValueError(f"need at least 2 ogive segments, got {n}")
    return tuple(i / float(n) for i in range(1, n))


# --------------------------------------------------------------------------------------
#   Recipe construction
# --------------------------------------------------------------------------------------


class _Builder:
    """Assembles the rocket recipe. One instance per recipe; not reusable."""

    def __init__(
        self,
        dv: DesignVector,
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
        stage: str,
    ) -> None:
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
        self.stage = stage
        self.n_fin = int(dv.n_fin)
        self.nose_shape = str(dv.nose_shape)

        self.r = Recipe(
            name="sv1_rocket",
            displayname="SV-1 Rocket",
            description=(
                "SV-1 parametric rocket vehicle (WP4). Ogive-cylinder-boattail body revolved "
                "from one closed profile, cruciform constant-thickness tapered fins, hollow "
                "airframe with ring bulkheads. Outputs measured volumes, wetted areas and "
                "structural mass properties as a single JSON value."
            ),
        )
        self.inp: dict[str, Ref] = {}
        self.refs: dict[str, Ref] = {}

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

    def _shift(self, ref: Ref, offset: Ref | float, name: str | None = None) -> Ref:
        if isinstance(offset, Ref):
            return self._add(ref, offset, name=name)
        return self._add(ref, self.r.literal_real(offset, {"length": 1}), name=name)

    def _point(self, x: Any, y: Any, z: Any, name: str | None = None) -> Ref:
        return self.r.block("point<real,real,real>", x, y, z, name=name)

    def _zero_length(self) -> Ref:
        return self.r.literal_real(0.0, {"length": 1})

    # ---- inputs -----------------------------------------------------------------------

    def declare_inputs(self) -> None:
        r, dv = self.r, self.dv
        for attr, iname, dim in NTOP_INPUTS:
            self.inp[attr] = r.add_input(
                iname, "real", default=float(getattr(dv, attr)), dimension=dim,
                description=f"DesignVector.{attr}",
            )
        self.inp[MESH_TOLERANCE_INPUT] = r.add_input(
            MESH_TOLERANCE_INPUT, "real", default=self.mesh_tolerance,
            dimension={"length": 1},
            description="implicit_to_mesh tolerance for the exported STL",
        )
        if self.export_stl:
            self.inp[STL_PATH_INPUT] = r.add_input(
                STL_PATH_INPUT, "file_path",
                default=to_ntop_path(os.path.join(self.default_dir, "sv1.stl")),
                description="STL export path",
            )
        if self.export_step:
            self.inp[STEP_PATH_INPUT] = r.add_input(
                STEP_PATH_INPUT, "file_path",
                default=to_ntop_path(os.path.join(self.default_dir, "sv1.step")),
                description="STEP export path",
            )
        if self.export_implicit:
            self.inp[IMPLICIT_PATH_INPUT] = r.add_input(
                IMPLICIT_PATH_INPUT, "file_path",
                default=to_ntop_path(os.path.join(self.default_dir, "sv1.implicit")),
                description=".implicit export path (readable by nTopCore)",
            )

    # ---- derived stations -------------------------------------------------------------

    def derive(self) -> None:
        """Every derived dimension, computed INSIDE nTop from the notebook inputs.

        Nothing here may be a Python number derived from `self.dv`, or the notebook would stop
        being parametric and the cached `.ntop` would silently be wrong for other design
        points.
        """
        i = self.inp
        R = self._scale(i["D"], 0.5, name="Body Radius")
        L_nose = self._mul(i["f_nose"], i["D"], name="Nose Length")
        x_cyl_end = self._sub(i["L_total"], i["L_boattail"], name="Boattail Start Station")
        r_base = self._scale(i["d_base"], 0.5, name="Base Radius")

        # fins
        c_t = self._mul(i["taper_fin"], i["c_r_fin"], name="Fin Tip Chord")
        x_fin_te = self._sub(i["L_total"], i["x_fin_te_gap"], name="Fin TE Station")
        x_fin_le = self._sub(x_fin_te, i["c_r_fin"], name="Fin Root LE Station")
        tan_sweep = self.r.block("tan<real>", i["sweep_fin"], name="Tan Fin Sweep")
        dx_sweep = self._mul(i["b_fin"], tan_sweep, name="Fin LE Sweep Offset")
        y_root = self._sub(R, i["t_wall"], name="Fin Root Radius")
        y_tip = self._add(R, i["b_fin"], name="Fin Tip Radius")
        half_t = self._scale(i["t_fin"], 0.5, name="Fin Half Thickness")
        neg_half_t = self._scale(i["t_fin"], -0.5, name="Fin Half Thickness Negative")

        # bulkhead stations
        x_bh1 = i["L_seeker"]
        x_bh2 = self._add(i["L_seeker"], i["L_guidance"], name="Guidance Bay Aft Station")
        x_bh3 = self._add(x_bh2, i["L_warhead"], name="Warhead Bay Aft Station")

        neg_t_wall = self._scale(i["t_wall"], -1.0, name="Inward Offset")

        self.refs.update(
            R=R, L_nose=L_nose, x_cyl_end=x_cyl_end, r_base=r_base, tan_sweep=tan_sweep,
            c_t=c_t, x_fin_te=x_fin_te, x_fin_le=x_fin_le, dx_sweep=dx_sweep,
            y_root=y_root, y_tip=y_tip, half_t=half_t, neg_half_t=neg_half_t,
            x_bh1=x_bh1, x_bh2=x_bh2, x_bh3=x_bh3, neg_t_wall=neg_t_wall,
        )

    # ---- the OML profile --------------------------------------------------------------

    def _ogive_points(self, length: Ref, radius: Ref, x0: Ref | None, n: int) -> list[Ref]:
        """Chord-polygon points of a tangent ogive, all arithmetic done in nTop.

        The profile is
            y(x) = sqrt(rho^2 - (L - x)^2) - (rho - R),  rho = (R^2 + L^2) / (2R).
        Written with u = x / L and k = L / R this becomes
            y / R = sqrt(c^2 - k^2 (1 - u)^2) - (c - 1),   c = (1 + k^2) / 2,
        which is entirely dimensionless apart from the final multiply by R. Keeping the square
        root dimensionless avoids relying on nTop taking the square root of a length^2.

        `k`, `c^2` and `c - 1` are shared across all samples, so each interior sample costs
        one multiply, one subtract, one sqrt, one subtract, one multiply for y, one multiply
        for x and the point block itself.
        """
        r = self.r
        k = self._div(length, radius, name="Ogive k")
        k2 = self._mul(k, k, name="Ogive k^2")
        c = self._add(self._scale(k2, 0.5), r.literal_real(0.5, {}), name="Ogive c")
        c2 = self._mul(c, c, name="Ogive c^2")
        cm1 = self._sub(c, r.literal_real(1.0, {}), name="Ogive c-1")

        pts: list[Ref] = []
        # tip: y = 0 exactly
        pts.append(self._point(x0 if x0 is not None else self._zero_length(),
                               self._zero_length(), self._zero_length(), name="Nose Tip"))
        for u in ogive_sample_fractions(n):
            d = (1.0 - u) ** 2
            b = self._mul(k2, r.literal_real(d, {}))
            s = self._sub(c2, b)
            q = r.block("sqrt<real>", s)
            yr = self._sub(q, cm1)
            y = self._mul(radius, yr)
            x = self._mul(length, r.literal_real(u, {}))
            if x0 is not None:
                x = self._add(x0, x)
            pts.append(self._point(x, y, self._zero_length()))
        # shoulder: exactly (x0 + L, R)
        x_end = length if x0 is None else self._add(x0, length)
        pts.append(self._point(x_end, radius, self._zero_length(),
                               name="Nose Shoulder"))
        return pts

    def _cone_points(self, length: Ref, radius: Ref, x0: Ref | None) -> list[Ref]:
        """A conical nose: two points. `dv.nose_shape == "cone"` selects this."""
        x_end = length if x0 is None else self._add(x0, length)
        return [
            self._point(x0 if x0 is not None else self._zero_length(),
                        self._zero_length(), self._zero_length(), name="Nose Tip"),
            self._point(x_end, radius, self._zero_length(), name="Nose Shoulder"),
        ]

    def build_oml(self) -> Ref:
        """The outer mould line: one closed profile revolved 360 degrees about the X axis."""
        r, i, g = self.r, self.inp, self.refs

        if self.nose_shape == "cone":
            pts = self._cone_points(g["L_nose"], g["R"], None)
        elif self.nose_shape == "tangent_ogive":
            pts = self._ogive_points(g["L_nose"], g["R"], None, self.n_ogive)
        else:
            raise ValueError(
                f"unsupported nose_shape {self.nose_shape!r}; use 'tangent_ogive' or 'cone'"
            )

        # cylinder shoulder -> boattail start -> base rim -> base centre, then the polygon
        # closes back along the axis to the nose tip.
        pts.append(self._point(g["x_cyl_end"], g["R"], self._zero_length(),
                               name="Boattail Start"))
        pts.append(self._point(i["L_total"], g["r_base"], self._zero_length(),
                               name="Base Rim"))
        pts.append(self._point(i["L_total"], self._zero_length(), self._zero_length(),
                               name="Base Centre"))

        poly = r.point_list(pts, name="OML Profile Points")
        profile = r.block("profile_from_points<list<point>>", poly, name="OML Polygon")
        # `profile_from_points` returns a `profile` (display name "Polygon"); `revolve` wants
        # an `implicit_2d`. types.json says `profile` exposes `profile: implicit_2d`, so the
        # props chain is the bridge. NTOP_NOTES.md section 6.
        profile_2d = r.variable("OML Profile", profile.prop("profile"))
        axis = r.block(
            "axis<point,vector>",
            self._point(self._zero_length(), self._zero_length(), self._zero_length(),
                        name="Axis Origin"),
            r.literal_vector(1.0, 0.0, 0.0),
            name="Body Axis",
        )
        body = r.block(
            "revolve<implicit_2d,axis,real>",
            profile_2d, axis, r.literal_real(2.0 * math.pi, {"angle": 1}),
            name="Body OML",
        )
        self.refs["axis"] = axis
        self.refs["body_oml"] = body
        return body

    # ---- fins -------------------------------------------------------------------------

    def _panel(self, span_axis: str, name: str) -> Ref:
        """One fin panel: the trapezoidal planform extruded to a constant thickness `t_fin`.

        WHAT SECTION THIS ACTUALLY IS. A **constant-thickness tapered plate**, not a double
        wedge. The double wedge was built first, by lofting a root diamond to a tip diamond,
        and it was measured to be WRONG: `loft<implicit_2d,implicit_2d>[1.1.0]` is a linear
        MIX of the two extruded signed-distance fields, not a linear interpolation of the two
        boundaries. nTop says so itself - the block logs "Loft between Profiles 1.1.0 is
        deprecated. Please use two Extrude Profile blocks, a Mix block, and a Ramp block to
        achieve similar results." Field mixing rounds the diamond's corners off, and the
        measured panel came out at 82 percent of the exact double-wedge volume
        (2.747e-4 m^3 against 3.343e-4 m^3) and 73 percent of its area. That is a 27 percent
        error on a measured quantity, which is not acceptable in a measurement notebook.

        The tapered plate is exact instead. Its planform is a four-point polygon extruded
        perpendicular to its own plane, so the enclosed planform area and the wetted area are
        right to machine precision. The price is the section: a plate of thickness `t_fin`
        holds twice the volume of a double wedge of the same maximum thickness, because a
        diamond section has area 0.5*c*t and a rectangle has c*t. Fin structural mass is
        therefore CONSERVATIVE (heavier) by a factor of two on the panels alone, which is
        about 4 kg out of a 40 kg airframe on the default SV-1.

        An exact double wedge is not reachable with the available blocks: for a swept, tapered
        panel the leading edge, the mid-chord ridge and the trailing edge are three mutually
        skew straight lines, so the wedge faces are hyperbolic paraboloids and cannot be built
        from planes, extrusions or a rotation. See `docs/NTOP_NOTES.md`.

        `span_axis` is "y" for the panel spanning +Y (thickness along Z) or "z" for the panel
        spanning +Z (thickness along Y).
        """
        r, i, g = self.r, self.inp, self.refs
        # Inboard extrapolation, so that the chord at the true body surface y = R is exactly
        # c_r_fin even though the plate root plugs into the wall at y = R - t_wall.
        # SOURCES["fin_root_plug"].
        dx_in = self._mul(i["t_wall"], g["tan_sweep"])
        x_le_root = self._sub(g["x_fin_le"], dx_in)
        c_root = self._add(
            i["c_r_fin"],
            self._mul(self._div(i["t_wall"], i["b_fin"]),
                      self._sub(i["c_r_fin"], g["c_t"])),
        )
        x_te_root = self._add(x_le_root, c_root)
        x_le_tip = self._add(g["x_fin_le"], g["dx_sweep"])
        x_te_tip = self._add(x_le_tip, g["c_t"])

        # The planform polygon sits at the -half-thickness offset, so extruding by the full
        # thickness along the positive normal centres the plate on the panel plane.
        off = g["neg_half_t"]
        if span_axis == "y":
            quad = [
                (x_le_root, g["y_root"], off),
                (x_le_tip, g["y_tip"], off),
                (x_te_tip, g["y_tip"], off),
                (x_te_root, g["y_root"], off),
            ]
            direction = r.literal_vector(0.0, 0.0, 1.0)
        elif span_axis == "z":
            quad = [
                (x_le_root, off, g["y_root"]),
                (x_le_tip, off, g["y_tip"]),
                (x_te_tip, off, g["y_tip"]),
                (x_te_root, off, g["y_root"]),
            ]
            direction = r.literal_vector(0.0, 1.0, 0.0)
        else:
            raise ValueError(f"span_axis must be 'y' or 'z', got {span_axis!r}")

        pts = [self._point(*q) for q in quad]
        plist = r.point_list(pts, name=f"{name} Planform Points")
        poly = r.block("profile_from_points<list<point>>", plist,
                       name=f"{name} Planform Polygon")
        profile_2d = r.variable(f"{name} Planform", poly.prop("profile"))
        return r.block(
            "extrude<implicit_2d,real,vector>", profile_2d, self.inp["t_fin"], direction,
            name=name,
        )

    def build_fins(self) -> Ref | None:
        """The cruciform fin set: two lofted panels plus their two mirrors.

        Mirroring rather than rotating is deliberate. `rotate<spatial3d,point,vector,real>`
        returns `any`, which then has to be fed into a `list<implicit>`; `mirror_body` returns
        a real `implicit`, so the block graph stays typed. A cruciform is exactly two mirror
        pairs, so no rotation is needed.
        """
        if self.n_fin == 0:
            return None
        if self.n_fin != 4:
            raise ValueError(
                f"n_fin = {self.n_fin} is not supported. The notebook bakes the cruciform in "
                f"as two lofted panels and two mirrors, because the panel count changes the "
                f"block graph and therefore the cached notebook. Use n_fin = 4 (cruciform) "
                f"or n_fin = 0 (body only)."
            )
        r = self.r
        zero = self._zero_length()
        origin = self._point(zero, zero, zero, name="Mirror Origin")
        plane_xz = r.block(
            "plane<point,vector,vector>", origin,
            r.literal_vector(1.0, 0.0, 0.0), r.literal_vector(0.0, 0.0, 1.0),
            name="XZ Plane",
        )
        plane_xy = r.block(
            "plane<point,vector,vector>", origin,
            r.literal_vector(1.0, 0.0, 0.0), r.literal_vector(0.0, 1.0, 0.0),
            name="XY Plane",
        )
        fin_y = self._panel("y", "Fin +Y")
        fin_z = self._panel("z", "Fin +Z")
        fin_ny = r.block("mirror_body<implicit,plane>", fin_y, plane_xz, name="Fin -Y")
        fin_nz = r.block("mirror_body<implicit,plane>", fin_z, plane_xy, name="Fin -Z")
        fins = self._union([fin_y, fin_ny, fin_z, fin_nz], name="Fins")
        self.refs["fins"] = fins
        return fins

    # ---- booleans ---------------------------------------------------------------------

    def _union(self, bodies: Sequence[Ref], name: str) -> Ref:
        r = self.r
        func = r.latest("boolean_union<blend_enum,real_field,list<implicit>>")
        return r.block(
            func,
            r.literal_enum("blend_enum", BLEND_NONE),
            self._zero_length(),
            r.list_of("implicit", list(bodies), name=f"{name} Bodies"),
            name=name,
        )

    def _subtract(self, primary: Ref, bodies: Sequence[Ref], name: str) -> Ref:
        r = self.r
        func = r.latest("boolean_subtract<blend_enum,real_field,implicit,list<implicit>>")
        return r.block(
            func,
            r.literal_enum("blend_enum", BLEND_NONE),
            self._zero_length(),
            primary,
            r.list_of("implicit", list(bodies), name=f"{name} Subtractions"),
            name=name,
        )

    # ---- hollow structure --------------------------------------------------------------

    def _bulkhead(self, x_station: Ref, name: str) -> Ref:
        """A ring bulkhead: a disc of thickness `t_wall` centred on `x_station`.

        The radius is the full body radius, which is oversize on purpose: the disc is only ever
        SUBTRACTED from the interior void, so anything beyond the void is discarded.
        """
        i, g = self.inp, self.refs
        zero = self._zero_length()
        half = self._scale(i["t_wall"], 0.5)
        x0 = self._sub(x_station, half)
        x1 = self._add(x_station, half)
        return self.r.block(
            "cylinder<point,point,real>",
            self._point(x0, zero, zero),
            self._point(x1, zero, zero),
            g["R"],
            name=name,
        )

    def build_cavity(self) -> Ref:
        """The usable internal volume: the inward offset of the OML, less the bulkheads.

        `offset_implicit(OML, -t_wall)` is one block and gives a true normal offset of the
        implicit field, which is exactly the inner surface of a constant-thickness shell. The
        alternative - building a second revolved profile with radii reduced by t_wall - needs
        another ~100 arithmetic blocks and is only an approximate offset near the nose.

        Subtracting three discs makes the seeker, guidance, warhead and motor bays real
        separated cavities rather than one continuous tube.
        """
        g = self.refs
        void = self.r.block(
            "offset_implicit<implicit,real_field>", g["body_oml"], g["neg_t_wall"],
            name="Interior Void",
        )
        self.refs["interior_void"] = void
        bulkheads = [
            self._bulkhead(g["x_bh1"], "Bulkhead Seeker Aft"),
            self._bulkhead(g["x_bh2"], "Bulkhead Guidance Aft"),
            self._bulkhead(g["x_bh3"], "Bulkhead Warhead Aft"),
        ]
        cavity = self._subtract(void, bulkheads, name="Internal Cavity")
        self.refs["cavity"] = cavity
        return cavity

    def build_structure(self, outer_all: Ref, cavity: Ref) -> Ref:
        """The airframe and fins: everything solid once the cavity is removed.

        THIS IS THE BODY WHOSE MASS IS REPORTED. It is the wall, the bulkheads and the fin
        panels. It contains no motor case, no propellant, no warhead and no avionics; those
        are charged elsewhere in `rocketgen/sizing/masses.py`. See the module docstring.
        """
        structure = self._subtract(outer_all, [cavity], name="Airframe Structure")
        self.refs["structure"] = structure
        return structure

    # ---- measurements -----------------------------------------------------------------

    def _mass_props(self, body: Ref, density: float, name: str) -> Ref:
        return self.r.mass_properties(body, density=density, relative_error=self.rel,
                                      name=name)

    def _area(self, body: Ref, name: str) -> Ref:
        return self.r.surface_area(body, relative_error=self.area_rel, name=name)

    def build_measurements(self) -> dict[str, tuple[Ref, dict[str, int]]]:
        r, i, g = self.r, self.inp, self.refs

        mp_oml = self._mass_props(g["body_oml"], 1.0, "OML Mass Properties")
        area_oml = self._area(g["body_oml"], "OML Surface Area")

        # area_base = pi/4 * d_base^2, computed in nTop so it tracks the input.
        d2 = self._mul(i["d_base"], i["d_base"])
        area_base = self._mul(d2, r.literal_real(0.25 * math.pi, {}), name="Base Area")
        # `area_wetted_body` in `masses.analytic_geometry` is the nose plus cylinder plus
        # boattail lateral area and EXCLUDES the flat base disc, so the disc is removed here
        # for an apples-to-apples comparison.
        area_body = self._sub(area_oml, area_base, name="Body Wetted Area")

        values: dict[str, tuple[Ref, dict[str, int]]] = {
            "volume_total": (mp_oml.prop("volume"), {"length": 3}),
            "area_wetted_body": (area_body, {"length": 2}),
            "area_base": (area_base, {"length": 2}),
        }

        if "fins" in g:
            # Exposed panels only: the part of the fin set outside the body. This is what the
            # aero model wets. It also picks up the small root-junction and tip edge faces,
            # which is a real over-count of about 5 percent against the flat-plate analytic
            # value on the default SV-1.
            exposed = self._subtract(g["fins"], [g["body_oml"]], name="Fins Exposed")
            values["area_wetted_fins"] = (self._area(exposed, "Fin Wetted Area"),
                                          {"length": 2})
        if "cavity" in g:
            mp_cav = self._mass_props(g["cavity"], 1.0, "Cavity Mass Properties")
            values["volume_cavity"] = (mp_cav.prop("volume"), {"length": 3})
        if "structure" in g:
            mp_str = self._mass_props(g["structure"], STRUCTURE_DENSITY,
                                      "Structure Mass Properties")
            cg = mp_str.prop("center of gravity")
            pm = mp_str.prop("principal moments")
            values["volume_structure"] = (mp_str.prop("volume"), {"length": 3})
            values["mass_structure"] = (mp_str.prop("mass"), {"mass": 1})
            for key, comp in zip(CG_COMPONENT_NAMES, ("x", "y", "z")):
                values[key] = (cg.prop(comp), {"length": 1})
            for key, comp in zip(INERTIA_COMPONENT_NAMES, ("x", "y", "z")):
                values[key] = (pm.prop(comp), {"length": 2, "mass": 1})

        values.update(self.build_area_distribution())
        return values

    def build_area_distribution(self) -> dict[str, tuple[Ref, dict[str, int]]]:
        """Cross-section area S(x) for the wave-drag model, if it is asked for.

        There is no single block for this (NTOP_NOTES.md section 13 point 6). The route is
        `extract_section<implicit,plane,real>` at each station, then a cross-section area on the
        resulting 2D region with `SECTION_AREA_BLOCK`.

        IT WORKS, and it is accurate. Measured against the closed-form ogive-cylinder-boattail
        section plus the fin plate sections, at 8 stations on the default SV-1, the errors were
        -0.13, -0.04, +0.01, +0.01, +0.01, +0.01, +0.01 and +0.01 percent. The fins ARE picked
        up: at x = 3.75 m the measured section exceeded the bare body by 0.00864 m^2, which is
        exactly 4 * b_fin * t_fin.

        It is nevertheless OFF by default, on cost. Each station adds about 0.9 s to the run
        (measured: 28.2 s at 0 stations, 35.3 s at 8), and the sizing loop calls this tens of
        times. `rocketgen/sizing/aero.py` already falls back to closed-form cross-section
        geometry when `area_distribution` is empty, and `measure_rocket` says so in its
        warnings, so nothing is silently lost. Turn it on for a fidelity run:
        `measure_rocket(dv, run_dir, area_stations=24)`.

        Stations are placed at (j + 0.5) / n of `L_total`, so they follow the input and never
        land on the degenerate sections at x = 0 or x = L_total.
        """
        n = self.area_stations
        if n <= 0:
            return {}
        r, i, g = self.r, self.inp, self.refs
        target = g.get("outer_all", g["body_oml"])
        zero = self._zero_length()
        out: dict[str, tuple[Ref, dict[str, int]]] = {}
        for j in range(n):
            frac = (j + 0.5) / n
            x = self._mul(i["L_total"], r.literal_real(frac, {}))
            origin = self._point(x, zero, zero)
            plane = r.block(
                "plane<point,vector,vector>", origin,
                r.literal_vector(0.0, 1.0, 0.0), r.literal_vector(0.0, 0.0, 1.0),
                name=f"Section Plane {j:02d}",
            )
            func = r.latest("extract_section<implicit,plane,real>")
            section = r.block(func, target, plane, self.section_feature_size,
                              name=f"Section {j:02d}")
            # SECTION_AREA_BLOCK, not `surface_area<implicit_2d,real>`. Both builds REJECT
            # every revision of `surface_area<implicit_2d,real>` at load time with
            # "[E]: Error loading recipe:", even though the vendored universe lists [1.2.0] as
            # current and non-deprecated. The block that `convert` actually accepts is the one
            # the universe marks deprecated. Verified on a sphere mid-plane section: it
            # reported 0.00785414 m^2 against an exact pi*r^2 = 0.00785398, i.e. +0.002 %.
            area = r.raw_block(
                SECTION_AREA_BLOCK, "real",
                [section, r.literal_real(self.area_rel, {})],
                name=f"Section Area {j:02d}",
            )
            out[f"area_section_{j:02d}"] = (area, {"length": 2})
            # The station itself is reported so Python does not have to re-derive it.
            out[f"station_{j:02d}"] = (x, {"length": 1})
        return out

    # ---- exports ----------------------------------------------------------------------

    def build_exports(self, outer_all: Ref) -> None:
        r, i = self.r, self.inp
        if self.export_stl:
            mesh = r.mesh_from_implicit(
                outer_all, tolerance=i[MESH_TOLERANCE_INPUT], name="Export Mesh"
            )
            r.block(
                r.latest("export_mesh<file_path,mesh,unit_length_enum>"),
                i[STL_PATH_INPUT], mesh, r.literal_unit_length("m"), name="Export STL",
            )
        if self.export_step:
            # NTOP_NOTES.md section 9: there is no implicit -> part block. The working chain is
            # cad_body_from_implicit_body -> brep, then .prop("part") -> part, then export_part.
            brep = r.block(
                "cad_body_from_implicit_body<implicit,real,list<brep>>",
                outer_all, self.cad_tolerance, None, name="CAD Body",
            )
            part = r.variable("CAD Part", brep.prop("part"))
            r.block(
                r.latest("export_part<file_path,part>"),
                i[STEP_PATH_INPUT], part, name="Export STEP",
            )
        if self.export_implicit:
            r.block(
                r.latest("export_implicit_body<file_path,implicit>"),
                i[IMPLICIT_PATH_INPUT], outer_all, name="Export Implicit",
            )

    # ---- assembly ---------------------------------------------------------------------

    def build(self) -> Recipe:
        self.declare_inputs()
        self.derive()
        body = self.build_oml()
        stage = self.stage

        fins = None
        if stage in ("fins", "hollow", "full"):
            fins = self.build_fins()

        outer_all = body if fins is None else self._union([body, fins], name="Outer Solid")
        self.refs["outer_all"] = outer_all

        if stage in ("hollow", "full"):
            cavity = self.build_cavity()
            self.build_structure(outer_all, cavity)

        values = self.build_measurements()
        self.build_exports(outer_all)
        self.r.json_output(values, name="Measurements")
        return self.r


def build_rocket_recipe(
    dv: DesignVector,
    out_dir: str,
    *,
    mesh_tolerance: float = DEFAULT_MESH_TOLERANCE,
    export_stl: bool = True,
    export_step: bool = False,
    export_implicit: bool = False,
    n_ogive: int = N_OGIVE_OUTER,
    relative_error: float = DEFAULT_RELATIVE_ERROR,
    area_relative_error: float = DEFAULT_AREA_RELATIVE_ERROR,
    cad_tolerance: float = DEFAULT_CAD_TOLERANCE,
    area_stations: int = 0,
    section_feature_size: float = SECTION_FEATURE_SIZE,
    stage: str = "full",
) -> Recipe:
    """Author the SV-1 recipe for `dv`, with `dv` exposed as nTop notebook inputs.

    `dv` supplies the DEFAULT value of every notebook input, and the topology choices
    (`n_fin`, `nose_shape`) that cannot be inputs. Every dimension in `NTOP_INPUTS` is a real
    notebook input, so the same converted `.ntop` measures any other design point.

    `out_dir` is where the export defaults point, so a bare `ntopcl convert` (which evaluates
    the notebook, NTOP_NOTES.md section 3) writes its artefacts somewhere sensible.

    `stage` is a debugging aid that truncates the build: "oml" stops after the revolved outer
    mould line, "fins" adds the cruciform, "hollow" and "full" add the cavity and the measured
    structure. Use "full".
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
        stage=stage,
    )
    return b.build()


# --------------------------------------------------------------------------------------
#   The notebook cache
# --------------------------------------------------------------------------------------


@dataclass
class RocketNotebook:
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

# Process-local cache: key -> RocketNotebook. Saves the `-t` call as well as the convert.
_MEMO: dict[str, RocketNotebook] = {}


def _topology_key(
    dv: DesignVector,
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
    stage: str,
) -> str:
    """Hash of everything that changes the BLOCK GRAPH, and nothing that does not.

    Dimensions are absent on purpose: they are notebook inputs, so a new design point reuses
    the same `.ntop`. `mesh_tolerance` and the export paths are absent for the same reason -
    they are inputs too.
    """
    payload = {
        "version": RECIPE_VERSION,
        "n_fin": int(dv.n_fin),
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
        "stage": str(stage),
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def clear_notebook_cache(cache_dir: str | None = None) -> None:
    """Forget every cached notebook. Tests use this to force a re-`convert`."""
    _MEMO.clear()
    if cache_dir and os.path.isdir(cache_dir):
        for f in os.listdir(cache_dir):
            if f.startswith("sv1_") and f.endswith((".ntop", ".json", ".log")):
                try:
                    os.remove(os.path.join(cache_dir, f))
                except OSError:                                     # pragma: no cover
                    pass


def build_rocket_notebook(
    dv: DesignVector,
    run_dir: str,
    runner: NtopRunner | None = None,
    *,
    cache_dir: str | None = None,
    force: bool = False,
    convert_timeout: float = 1800.0,
    **kw: Any,
) -> str:
    """Convert (or reuse) the SV-1 `.ntop` for the topology `dv` implies. Returns its path.

    The notebook is cached under `cache_dir` (default `runs/_ntop_cache`) by a topology key,
    because `ntopcl convert` evaluates the whole notebook and is therefore as expensive as a
    run. NTOP_NOTES.md section 13 point 5: convert once, run many times.

    The export defaults match `measure_rocket`'s - all OFF. They have to: an export changes
    the block graph, so it is part of the topology key, and two entry points disagreeing about
    the default would convert the same geometry twice.
    """
    return _notebook(dv, run_dir, runner, cache_dir=cache_dir, force=force,
                     convert_timeout=convert_timeout, **kw).path


def _notebook(
    dv: DesignVector,
    run_dir: str,
    runner: NtopRunner | None = None,
    *,
    cache_dir: str | None = None,
    force: bool = False,
    convert_timeout: float = 1800.0,
    **kw: Any,
) -> RocketNotebook:
    opts = dict(
        n_ogive=kw.pop("n_ogive", N_OGIVE_OUTER),
        relative_error=kw.pop("relative_error", DEFAULT_RELATIVE_ERROR),
        area_relative_error=kw.pop("area_relative_error", DEFAULT_AREA_RELATIVE_ERROR),
        # Defaults MUST match `measure_rocket`, or the two would key different notebooks and
        # convert twice for the same geometry.
        export_stl=kw.pop("export_stl", False),
        export_step=kw.pop("export_step", False),
        export_implicit=kw.pop("export_implicit", False),
        cad_tolerance=kw.pop("cad_tolerance", DEFAULT_CAD_TOLERANCE),
        area_stations=kw.pop("area_stations", 0),
        section_feature_size=kw.pop("section_feature_size", SECTION_FEATURE_SIZE),
        stage=kw.pop("stage", "full"),
    )
    mesh_tolerance = kw.pop("mesh_tolerance", DEFAULT_MESH_TOLERANCE)
    if kw:
        raise TypeError(f"unexpected keyword arguments: {sorted(kw)}")

    key = _topology_key(dv, **opts)
    cdir = os.path.abspath(cache_dir or DEFAULT_CACHE_DIR)
    ntop_path = os.path.join(cdir, f"sv1_{key}.ntop")
    recipe_json = os.path.join(cdir, f"sv1_{key}_recipe.json")

    memo = _MEMO.get(key)
    if memo is not None and not force and os.path.isfile(memo.path):
        log.info("reusing cached notebook %s", memo.path)
        return RocketNotebook(memo.path, memo.input_template, key, memo.recipe_json,
                               memo.convert_wall_time_s, reused=True)

    os.makedirs(cdir, exist_ok=True)
    run = runner if runner is not None else NtopRunner()

    if force or not os.path.isfile(ntop_path) or os.path.getsize(ntop_path) == 0:
        recipe = build_rocket_recipe(dv, cdir, mesh_tolerance=mesh_tolerance, **opts)
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
    nb = RocketNotebook(ntop_path, input_template, key, recipe_json, dt, reused=reused)
    _MEMO[key] = nb
    return nb


# --------------------------------------------------------------------------------------
#   The geometry function the sizing loop calls
# --------------------------------------------------------------------------------------


def _input_payload(
    dv: DesignVector,
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
    for attr, iname, dim in NTOP_INPUTS:
        if iname not in accepted:                                  # pragma: no cover
            continue
        values[iname] = float(getattr(dv, attr))
        u = _display_unit(dim)
        if u:
            units[iname] = u
    if MESH_TOLERANCE_INPUT in accepted:
        values[MESH_TOLERANCE_INPUT] = float(mesh_tolerance)
        units[MESH_TOLERANCE_INPUT] = "m"
    for iname, path in (
        (STL_PATH_INPUT, stl_path),
        (STEP_PATH_INPUT, step_path),
        (IMPLICIT_PATH_INPUT, implicit_path),
    ):
        if iname in accepted and path:
            values[iname] = to_ntop_path(path)
    return values, units


def _collect_vectors(raw: Mapping[str, Any], m: NtopMeasurements) -> None:
    """Reassemble `cg_structure` and `inertia_structure` from their scalar components.

    `core.list<real>` only carries scalars, so a point or a vector has to travel as three
    named reals (NTOP_NOTES.md section 13 point 2). `NtopMeasurements` wants 3-tuples.
    """
    def triple(names: Iterable[str]) -> tuple[float, float, float] | None:
        vals: list[float] = []
        for n in names:
            v = raw.get(n)
            if v is None:
                return None
            vals.append(float(v))
        return (vals[0], vals[1], vals[2])

    cg = triple(CG_COMPONENT_NAMES)
    if cg is not None:
        m.cg_structure = cg
    inertia = triple(INERTIA_COMPONENT_NAMES)
    if inertia is not None:
        m.inertia_structure = inertia


def _collect_area_distribution(raw: Mapping[str, Any], m: NtopMeasurements) -> None:
    """Build S(x) from the `station_NN` / `area_section_NN` pairs, if the notebook made them."""
    rows: list[tuple[float, float]] = []
    for j in range(1000):
        sx = raw.get(f"station_{j:02d}")
        ax = raw.get(f"area_section_{j:02d}")
        if sx is None or ax is None:
            break
        rows.append((float(sx), float(ax)))
    if rows:
        m.area_distribution = sorted(rows)


def measure_rocket(
    dv: DesignVector,
    run_dir: str,
    runner: NtopRunner | None = None,
    *,
    mesh_tolerance: float = DEFAULT_MESH_TOLERANCE,
    export_stl: bool = False,
    export_step: bool = False,
    export_implicit: bool = False,
    cache_dir: str | None = None,
    force_convert: bool = False,
    timeout: float = 1800.0,
    verbose: int = 2,
    tag: str | None = None,
    **kw: Any,
) -> NtopMeasurements:
    """Build (or reuse) the SV-1 notebook, run it at `dv`, and return the measurements.

    This is THE `geometry_fn` for `rocketgen.sizing.loop`. Its signature matches
    `loop.GeometryFn = Callable[[DesignVector, str], NtopMeasurements]`, so

        converge_point(dv, reqs, geometry_fn=measure_rocket, run_dir=...)

    works with no adapter.

    A design vector that `DesignVector.geometry_is_valid()` rejects raises `ValueError`
    immediately, before an `ntopcl` subprocess is spent on it. A geometry that nTop itself
    cannot build raises `NtopError` carrying the captured nTop diagnostics.

    EXPORTS ARE ALL OFF BY DEFAULT, which is a measured decision and not laziness. The
    measurement blocks alone cost about 30 s per run, of which about 8 s is fixed `ntopcl`
    startup. Adding the STL costs a further 40 s and over a gigabyte of resident memory,
    because `implicit_to_mesh` drives a dense voxel grid over the whole 4.0 x 0.71 x 0.71 m
    bounding box; adding STEP costs more again. The sizing loop calls this function tens of
    times and needs none of it. Turn the exports on for the converged design point:

        measure_rocket(dv, run_dir, export_stl=True, export_step=True,
                        export_implicit=True)

    `build_rocket_recipe` keeps `export_stl=True` as its own default, because a recipe built by
    hand is normally being built to look at.
    """
    ok, errs = dv.geometry_is_valid()
    if not ok:
        raise ValueError(
            "invalid design vector, refusing to spend an ntopcl call: " + "; ".join(errs)
        )

    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    stem = tag or "sv1"

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
        nb.path,
        values,
        out_json=out_json,
        expect=expect,
        input_template=nb.input_template,
        units=units,
        input_json=os.path.join(run_dir, f"{stem}_input.json"),
        run_dir=run_dir,
        timeout=timeout,
        verbose=verbose,
    )

    parsed = parse_outputs(out_json, run=result)
    m = parsed.measurements
    _collect_vectors(parsed.raw, m)
    _collect_area_distribution(parsed.raw, m)

    m.ntop_path = nb.path
    m.stl_path = stl_path
    m.step_path = step_path
    m.implicit_path = implicit_path
    m.wall_time_s = result.wall_time_s
    m.ntopcl_returncode = result.returncode

    if not m.area_distribution:
        m.warnings.append(
            "area_distribution is empty: the notebook was built with area_stations = 0. "
            "The aero model falls back to closed-form cross-section geometry."
        )
    if m.inertia_structure is None:
        m.warnings.append("inertia_structure not reported by the notebook")
    if not m.is_usable():
        missing = [
            f for f in ("volume_total", "volume_cavity", "area_wetted_body",
                        "mass_structure")
            if getattr(m, f) is None
        ]
        m.warnings.append("nTop did not report: " + ", ".join(missing))

    parsed.to_json(os.path.join(run_dir, f"{stem}_measurements.json"))
    return m


def geometry_fn(**kw: Any) -> Any:
    """A `loop.GeometryFn` with `measure_rocket`'s options pinned.

    `converge_point` and `size` call `geometry_fn(dv, run_dir)` with no keywords, so this is
    how WP5 chooses the cost/artefact trade for a whole sizing run:

        from rocketgen.ntopgen.rocket_notebook import geometry_fn
        size(dv0, reqs, geometry_fn=geometry_fn(export_stl=False), run_dir=...)

    MEASURED COST SPLIT, so the choice is informed rather than guessed. One run of the full
    notebook is about 30 s, of which about 8 s is fixed `ntopcl` startup and most of the rest is
    the two `surface_area<implicit,real>` blocks (section 19 of `docs/NTOP_NOTES.md`). Adding the
    STL export costs a further 15 to 25 s and several hundred megabytes, because
    `implicit_to_mesh` drives a DENSE voxel grid over the whole 4.0 x 0.71 x 0.71 m bounding
    box. A sizing loop wants `export_stl=False`; the final converged design point wants it, and
    STEP, turned on.
    """
    from functools import partial

    return partial(measure_rocket, **kw)
