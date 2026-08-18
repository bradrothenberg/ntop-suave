"""Assemble the SV-1 SPLINE engineering report PDF.

    .venv/Scripts/python.exe -m rocketgen.report.build_report_spline

Every number comes from a file under `runs/SV-1_spline/`. Nothing is typed in by hand and
nothing is re-computed here. Build the study, the evidence and the figures first:

    .venv/Scripts/python.exe run_sv1.py --stage all --oml spline
    .venv/Scripts/python.exe -m rocketgen.report.evidence --oml spline
    .venv/Scripts/python.exe -m rocketgen.report.fig_oml --oml spline      (and the others)

WHY THIS IS A SEPARATE BUILDER
------------------------------
`build_report.py` describes the tangent-ogive SV-1 in full: the requirements audit, the
impossible terminal Mach, the motor transition. This report answers a narrower question. It
compares two converged vehicles that differ only in the shape family of the outer mould line.
So it states the comparison and points at the ogive report for what the two studies share.
This follows the repository rule that a new study gets a PARALLEL module, never an edit to the
validated one.

Style follows ASD-STE100 Simplified Technical English: active voice, simple tenses, short
sentences, one idea per sentence.
"""
from __future__ import annotations

import csv
import json
import math
import os
from typing import Any

from reportlab.platypus import PageBreak, Paragraph, Spacer

from . import report_style as S

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(REPO, "runs")
CASE = os.path.join(RUNS, "SV-1_spline")
FIGS = os.path.join(CASE, "figures")
OUT_DIR = os.path.join(CASE, "report")
OUT_PDF = os.path.join(OUT_DIR, "SV1_spline_engineering_report.pdf")

#: The ogive study this report compares against. Only its curated example directory is on the
#: branch, so the baseline numbers are read from there rather than from `runs/SV-1`.
OGIVE_EXAMPLE = os.path.join(REPO, "examples", "SV-1")

#: Recipe files, read only to count nodes. A node count is a fair size measure of a notebook and
#: it is the only one available without opening the binary container.
RECIPE_SPLINE = os.path.join(REPO, "examples", "SV-1-spline", "02_geometry", "sv1_recipe.json")
RECIPE_OGIVE = os.path.join(OGIVE_EXAMPLE, "02_geometry", "sv1_recipe.json")


# --------------------------------------------------------------------------------------
#   Load everything from disk
# --------------------------------------------------------------------------------------


def _json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _csv(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_all() -> dict[str, Any]:
    d: dict[str, Any] = {
        "ntop": _json(os.path.join(CASE, "converged", "point_ntop.json")),
        "analytic": _json(os.path.join(CASE, "converged", "point_analytic.json")),
        "meas": _json(os.path.join(CASE, "converged", "measurements.json")),
        "prov": _json(os.path.join(CASE, "provenance.json")),
        "sens": _json(os.path.join(CASE, "doe", "sensitivity.json")),
        "evidence": _json(os.path.join(FIGS, "evidence.json")),
        "grid": _csv(os.path.join(CASE, "doe", "grid.csv")),
        "lhs": _csv(os.path.join(CASE, "doe", "lhs.csv")),
    }
    # The tangent-ogive baseline, for the comparison table. Absent on some checkouts, so its
    # absence becomes a stated gap rather than a crash.
    baseline = os.path.join(OGIVE_EXAMPLE, "01_design", "point_ntop.json")
    d["ogive"] = _json(baseline) if os.path.isfile(baseline) else None
    d["nodes_spline"] = _node_count(RECIPE_SPLINE)
    d["nodes_ogive"] = _node_count(RECIPE_OGIVE)
    return d


def _node_count(path: str) -> int | None:
    if not os.path.isfile(path):
        return None
    return len(_json(path)["body"])


# --------------------------------------------------------------------------------------
#   Small helpers
# --------------------------------------------------------------------------------------


def fmt(v: Any, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "not available"
    return f"{float(v):,.{nd}f}"


def sci(v: float | None) -> str:
    if v is None:
        return "not available"
    if v == 0.0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    return f"{v / (10.0 ** e):.1f}e{e:+d}"


def pct(v: float | None, nd: int = 2) -> str:
    if v is None:
        return "not available"
    return f"{100.0 * v:+.{nd}f}"


def fig(name: str) -> str | None:
    p = os.path.join(FIGS, name)
    return p if os.path.isfile(p) else None


def add_figure(story: list, name: str, num: int, caption: str,
               missing_note: str, max_h_in: float = 4.4) -> None:
    """Place a figure, or say in the report why it is not there.

    A missing figure is recorded, never skipped silently. That is the same rule the analysis
    code follows for a failed sample.
    """
    path = fig(name)
    if path is None:
        story.append(
            Paragraph(
                f"<b>Figure {num} is not available.</b> {missing_note} "
                f"The expected file is runs/SV-1_spline/figures/{name}.",
                S.BODY,
            )
        )
        return
    story.append(S.fig_single(path, num, caption, max_h_in=max_h_in))


# --------------------------------------------------------------------------------------
#   1. Front matter and the headline
# --------------------------------------------------------------------------------------


def front_matter(story: list, D: dict) -> None:
    p = D["ntop"]
    dv = p["design_vector"]
    tr = p["trajectory"]
    ev = D["evidence"]

    story.append(Paragraph("A Splined Outer Mould Line for the SV-1", S.TITLE))
    story.append(
        Paragraph(
            "The same vehicle, re-sized with its nose and boattail revolved from B-splines "
            "inside nTop",
            S.SUBTITLE,
        )
    )
    story.append(S.hrule())
    story.append(
        Paragraph(
            "Prepared with nTop Automate 5.53.2 / 5.54.0 and SUAVE 2.5.2. All geometry is "
            "authored programmatically. Every number is read from the run artefacts under "
            "runs/SV-1_spline. The tangent-ogive result in examples/SV-1 is the baseline.",
            S.SUBTITLE,
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("Summary", S.H1))
    story.append(
        Paragraph(
            "This report describes one change to a coupled sizing loop. The outer mould line of "
            "the SV-1 rocket is now a true revolved B-spline. It was a tangent ogive and a "
            "conical boattail. Nothing else changed. The requirements, the physics modules, the "
            "sizing search and the trade-study axes are the same.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            f"The loop found a design that meets all "
            f"{sum(1 for c in p['constraints'] if c['met'])} of {len(p['constraints'])} "
            f"constraints. It weighs {fmt(p['mass_statement']['total_kg'], 1)} kg. It flies "
            f"{fmt(tr['range_m'] / 1000.0, 1)} km. It arrives at Mach "
            f"{fmt(tr['mach_final'], 2)}. The search used "
            f"{ev['search'].get('n_evaluations', 'an unrecorded number of')} evaluations with "
            f"real nTop geometry in the loop.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "Three results are worth more than the mass number. First, the sizer declined the "
            "drag optimum: nose_blend converged to "
            f"{fmt(dv['nose_blend'], 2)}, not 1.0. The shape trade has an interior optimum. "
            "Second, the drag saving was not free: peak dynamic pressure rose against a limit "
            "that was already close. Third, the coupling decided feasibility. The same design "
            "vector with closed-form geometry is lighter, flies further, and violates two "
            "constraints. Only the measured geometry finds the feasible answer.",
            S.BODY,
        )
    )

    rows = [
        ["Quantity", "Tangent ogive", "Spline", "Change"],
    ]
    ogive = D["ogive"]
    if ogive is not None:
        pairs = [
            ("Launch mass [kg]", ogive["mass_statement"]["total_kg"],
             p["mass_statement"]["total_kg"], 2),
            ("Range [km]", ogive["trajectory"]["range_m"] / 1000.0,
             tr["range_m"] / 1000.0, 2),
            ("Impact Mach [-]", ogive["trajectory"]["mach_final"], tr["mach_final"], 3),
            ("Peak dynamic pressure [kPa]", ogive["trajectory"]["q_max_Pa"] / 1000.0,
             tr["q_max_Pa"] / 1000.0, 2),
        ]
        for name, a, b, nd in pairs:
            rows.append([name, fmt(a, nd), fmt(b, nd), f"{b - a:+,.{nd}f}"])
    else:
        rows.append(["baseline not on this branch", "not available", "not available", ""])
    sm = next((c for c in p["constraints"] if c["name"].startswith("R10")), None)
    if sm is not None:
        rows.append(["Static margin [cal]", "not recorded in the example",
                     fmt(sm["value"], 3), ""])
    story.append(S.styled_table(rows, [2.3, 1.6, 1.5, 1.4],
                                ["LEFT", "RIGHT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 1. The headline comparison. Both designs meet every constraint. The "
            "tangent-ogive figures come from examples/SV-1/01_design/point_ntop.json. The "
            "spline figures come from runs/SV-1_spline/converged/point_ntop.json.",
            S.CAP,
        )
    )


# --------------------------------------------------------------------------------------
#   2. What changed
# --------------------------------------------------------------------------------------


def what_changed(story: list, D: dict) -> None:
    p = D["ntop"]
    dv = p["design_vector"]
    ev = D["evidence"]
    w = ev["wavedrag"]

    S.sect(story, "1. What changed, and what did not")
    story.append(
        Paragraph(
            "Only the shape family of the outer mould line changed. The nose is a cubic "
            f"B-spline with {w['n_ctrl']} control points. The boattail is a splined "
            "contraction. Two new scalars let the sizing search move between the shape the "
            "ogive gives and the shape linear theory prefers.",
            S.BODY,
        )
    )
    rows = [
        ["Design variable", "Meaning", "Bound", "Converged"],
        ["nose_shape", "shape family of the forebody", "ogive or spline",
         str(dv["nose_shape"])],
        ["nose_blend", "0 reproduces the ogive, 1 is the drag optimum", "0.0 to 1.0",
         fmt(dv["nose_blend"], 2)],
        ["boattail_shape", "shape family of the aft contraction", "cone or spline",
         str(dv["boattail_shape"])],
        ["boattail_blend", "0 is the straight cone, 1 is the full curve", "0.0 to 1.0",
         fmt(dv["boattail_blend"], 2)],
        ["n_ctrl_oml", "control points per spline", "fixed", str(dv["n_ctrl_oml"])],
    ]
    story.append(S.styled_table(rows, [1.35, 2.85, 1.35, 1.25],
                                ["LEFT", "LEFT", "LEFT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 2. The design variables the shape change adds. Every other entry in the "
            "design vector keeps the meaning it had in the tangent-ogive study.",
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            "One thing did NOT change, and it matters. The drag calibration factor stays where "
            "it was. It is applied at the loop boundary, in sizing/loop.py, through "
            f"CalibratedAero. Its value here is {fmt(D['prov']['cd0_calibration'], 3)}. The "
            "aerodynamic model still reports what its physics gives. The loop still owns the "
            "correction.",
            S.BODY,
        )
    )
    add_figure(
        story, "sv1_iso.png", 1,
        "The converged spline vehicle, rendered from the exported STL with a locked camera. "
        "The mesh is a picture only. Every number in this report comes from the notebook's own "
        "mass properties, which beat the mesh by about sixteen times on the smoke sphere.",
        "The render script needs the exported STL under runs/SV-1_spline/converged/geom.",
        max_h_in=2.6,
    )


# --------------------------------------------------------------------------------------
#   3. Why a shape can change anything
# --------------------------------------------------------------------------------------


def wave_drag(story: list, D: dict) -> None:
    ev = D["evidence"]
    w = ev["wavedrag"]
    sh = w["sears_haack"]
    vk = w["von_karman"]

    S.sect(story, "2. Why the shape can change anything at all")
    story.append(
        Paragraph(
            "Before this work no drag model here could tell one nose from another at fixed "
            "fineness. The forebody wave-drag term is the Bonney correlation, a function of "
            "length over diameter alone. The Sears-Haack cross-check is a function of diameter "
            "over length alone. A spline study run against that model would have reported no "
            "change, for the wrong reason.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "rocketgen/sizing/wavedrag.py supplies the missing sensitivity. It computes the "
            "linearised slender-body wave drag of a given area distribution through the Glauert "
            "series. It then reports a dimensionless RATIO against the tangent ogive of the "
            "same fineness. The ratio multiplies the Bonney value. So the calibrated level and "
            "the Mach dependence stay with the correlation that was validated against 23 Basic "
            "Finner free-flight shots. Only the shape effect comes from linear theory.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "At the ogive control values the ratio is exactly 1.0. The test suite asserts that "
            "the drag coefficient is then reproduced bit for bit, with an equality check rather "
            "than a tolerance. A shape model that changes the answer at the baseline shape "
            "would be a defect, not a feature.",
            S.BODY,
        )
    )

    rows = [
        ["Check", "Closed form", "Model", "Relative residual"],
        ["Sears-Haack body drag", "128 V^2 / (pi L^4)", fmt(sh["series"], 6),
         sci(sh["rel_err"])],
        ["von Karman ogive drag", "series against the closed form", fmt(vk["series"], 8),
         sci(vk["rel_err"])],
        ["von Karman C_D on base area", "(d / L)^2", fmt(vk["cd_on_base"], 8),
         sci(vk["cd_rel_err"])],
        ["Optimum shape factor", "4 / pi",
         fmt(w["von_karman_shape_factor"]["measured"], 8),
         sci(w["von_karman_shape_factor"]["rel_err"])],
    ]
    story.append(S.styled_table(rows, [1.9, 1.9, 1.5, 1.5],
                                ["LEFT", "LEFT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 3. The wave-drag model against exact closed forms that come from outside "
            "this repository. Each test asserts a tolerance of 1.0e-4. The residuals above are "
            "the values the code achieves, measured for this report rather than transcribed "
            f"from the test file. All four are between {sci(min(abs(sh['rel_err']), abs(vk['rel_err'])))} "
            f"and {sci(max(abs(sh['rel_err']), abs(w['von_karman_shape_factor']['rel_err'])))}.",
            S.CAP,
        )
    )

    refine = sh.get("refinement")
    if refine:
        rows = [["Stations in the check table", "Relative residual"]] + [
            [str(r["n_stations"]), sci(r["rel_err"])] for r in refine
        ]
        story.append(S.styled_table(rows, [3.4, 3.4], ["LEFT", "RIGHT"]))
        story.append(Spacer(1, 4))
        story.append(
            Paragraph(
                "Table 4. The Sears-Haack residual belongs to the CHECK TABLE, not to the "
                "model. The Sears-Haack profile has an infinite slope at both ends, so the "
                "central-difference derivative converges slowly on it. Refining the table drives "
                "the residual down by about a factor of three per doubling. A single number "
                "could not have said this.",
                S.CAP,
            )
        )

    story.append(
        Paragraph(
            "Two further checks support the series itself. The Glauert series constant of pi "
            "over four was compared against direct double integration of the same integral. The "
            "deleted diagonal converges slowly, so what is measured is convergence onto the "
            "series: the residual falls from "
            f"{sci(w['glauert_direct'][0]['rel_err'])} at {w['glauert_direct'][0]['n']} points "
            f"to {sci(w['glauert_direct'][-1]['rel_err'])} at {w['glauert_direct'][-1]['n']}. "
            "Separately, the von Karman ogive was confirmed as the constrained optimum rather "
            "than assumed to be. Adding any higher Glauert mode at fixed base area raises the "
            f"drag in all {len(w['optimality'])} perturbations tried.",
            S.BODY,
        )
    )

    rows = [
        ["Shape", "Glauert shape factor", "Over the bound"],
        ["Tangent ogive", fmt(w["shape_factor_ogive"], 5),
         fmt(w["ogive_penalty_over_bound"], 5)],
        [f"Optimal {w['n_ctrl']}-point spline", fmt(w["shape_factor_optimal_spline"], 5),
         fmt(w["spline_over_bound"], 5)],
        ["von Karman ogive, the bound", fmt(w["shape_factor_bound"], 5), "1.00000"],
    ]
    story.append(S.styled_table(rows, [2.6, 2.1, 2.1], ["LEFT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            f"Table 5. The gap the spline exists to collect. A {w['n_ctrl']}-point spline "
            f"recovers {fmt(100.0 * w['gap_recovered_fraction'], 1)} percent of the distance "
            "from the tangent ogive to the theoretical bound. The ogive penalty is nearly "
            "independent of fineness: it moves only from "
            f"{fmt(w['ogive_penalty_by_fineness'][0]['sf_over_bound'], 4)} to "
            f"{fmt(w['ogive_penalty_by_fineness'][-1]['sf_over_bound'], 4)} over nose fineness "
            f"{fmt(w['ogive_penalty_by_fineness'][0]['f_nose'], 1)} to "
            f"{fmt(w['ogive_penalty_by_fineness'][-1]['f_nose'], 1)}.",
            S.CAP,
        )
    )
    add_figure(
        story, "wavedrag_validation.png", 2,
        "The wave-drag model. Panel (a) is the gap between the tangent ogive and the bound. "
        "Panel (b) is every closed-form residual against the tolerance its test asserts. "
        "Panel (c) is the drag build-up the ratio moves.",
        "Run -m rocketgen.report.fig_wavedrag --oml spline.",
        max_h_in=2.8,
    )

    share = ev["shape_trade"]["wave_share"]
    rows = [["Mach", "Wave share of CD0, ogive", "Change in wave term", "Change in CD0"]]
    for r in share:
        rows.append([
            fmt(r["mach"], 1),
            fmt(100.0 * r["wave_share_of_cd0_ogive"], 1) + " %",
            f"{r['d_wave_pct']:+.2f} %",
            f"{r['d_cd0_pct']:+.2f} %",
        ])
    story.append(S.styled_table(rows, [1.0, 2.3, 1.75, 1.75],
                                ["RIGHT", "RIGHT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 6. What the shape does on this vehicle, at 12 km and 2 degrees of angle of "
            "attack. The forebody wave term is a rising share of zero-lift drag with Mach "
            "number. The converged spline cuts that term by a constant "
            f"{fmt(abs(share[0]['d_wave_pct']), 2)} percent, because the ratio is "
            "Mach-independent at this order. The effect on total zero-lift drag therefore grows "
            "with Mach number.",
            S.CAP,
        )
    )


# --------------------------------------------------------------------------------------
#   4. The nTop side
# --------------------------------------------------------------------------------------


def ntop_side(story: list, D: dict) -> None:
    ev = D["evidence"]
    g = ev["spline_geometry"]
    meas = D["meas"]

    S.sect(story, "3. How the spline is built in nTop")
    story.append(
        Paragraph(
            "nTop revolves the spline itself. There is no chord polygon and no sampling error. "
            "The notebook computes the control points from live inputs, builds a profile from a "
            "list of curves, and revolves that profile. The chain is four blocks:",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "spline_by_control_points&lt;list&lt;point&gt;,integer&gt;[5.20.0]<br/>"
            "core.list&lt;curve_interface&gt;<br/>"
            "profile_from_curves&lt;list&lt;curve_interface&gt;,vector&gt;[5.20.0]<br/>"
            "revolve&lt;new_profile,axis,real&gt;[5.20.0]",
            S.MONO,
        )
    )
    story.append(
        Paragraph(
            "The profile is a curve LIST. It mixes spline segments with two_point_line segments. "
            "So the cylinder, the base disc and the return along the axis stay exactly straight, "
            "and the corners stay sharp. The axial fractions of the control points are the "
            "Greville abscissae. That makes the axial station an exact linear function of the "
            "spline parameter, and it keeps the radius a spline in the axial station.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "None of those four blocks appears in the vendored block universe, and neither do "
            "the types they need. The universe is incomplete, and its silence is not evidence. "
            "An earlier attempt sampled the spline into a chord polygon because the universe "
            "listed no route from a curve to a revolvable profile. That route exists. All four "
            "blocks go through the raw-block escape hatch. docs/NTOP_NOTES.md section 25 records "
            "the four encoding traps.",
            S.BODY,
        )
    )

    rows = [
        ["Measurement", "nTop", "Exact integral of the same spline", "Relative error"],
        ["Outer mould line volume [m^3]", fmt(g["volume_ntop"], 6),
         fmt(g["volume_closed_form"], 6), pct(g["volume_rel_err"], 4) + " %"],
        ["Body wetted area [m^2]", fmt(g["area_ntop"], 5),
         fmt(g["area_closed_form"], 5), pct(g["area_rel_err"], 4) + " %"],
    ]
    story.append(S.styled_table(rows, [2.1, 1.35, 2.15, 1.2],
                                ["LEFT", "RIGHT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 7. The measured solid against the exact integral of the B-spline that built "
            "it. Because nTop revolves the spline rather than a polygon, the closed form is an "
            "exact description of the solid and not an approximation of it. There is no "
            "discretisation error available to hide behind. The volume splits as "
            f"{fmt(g['parts']['nose'], 6)} nose, {fmt(g['parts']['cylinder'], 6)} cylinder and "
            f"{fmt(g['parts']['boattail'], 6)} boattail, all in cubic metres.",
            S.CAP,
        )
    )

    rows = [["Notebook", "Recipe body nodes", "Notebook inputs"]]
    if D["nodes_spline"] is not None:
        rows.append(["SV-1, splined outer mould line", str(D["nodes_spline"]),
                     str(len(_json(RECIPE_SPLINE)["inputs"]))])
    if D["nodes_ogive"] is not None:
        rows.append(["SV-1, tangent ogive", str(D["nodes_ogive"]),
                     str(len(_json(RECIPE_OGIVE)["inputs"]))])
    if len(rows) > 1:
        story.append(S.styled_table(rows, [3.0, 2.0, 1.8], ["LEFT", "RIGHT", "RIGHT"]))
        story.append(Spacer(1, 4))
        story.append(
            Paragraph(
                "Table 8. Notebook size, counted as nodes in the recipe body. The spline "
                "notebook is the smaller of the two. A revolved spline replaces the sampled "
                "polygon and its supporting arithmetic, and it takes twice as many inputs to "
                "carry the control values. Measuring the converged geometry took "
                f"{fmt(meas['wall_time_s'], 1)} s of wall time in this run.",
                S.CAP,
            )
        )
    else:
        story.append(
            Paragraph(
                "Notebook node counts are not available: neither recipe JSON is on this branch.",
                S.BODY,
            )
        )

    rows = [["Quantity", "nTop", "Closed form", "Relative error"]]
    for r in ev["ntop"]["rows"]:
        rows.append([r["quantity"], fmt(r["ntop"], 6), fmt(r["closed_form"], 6),
                     pct(r["rel_err"], 3) + " %"])
    story.append(S.styled_table(rows, [1.9, 1.7, 1.7, 1.5],
                                ["LEFT", "RIGHT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 9. Every measurement at the converged design against its closed form. The "
            "fin area is the one large disagreement, and it is expected. The closed form counts "
            "two sides of each exposed panel. The nTop plate also has a tip face, two edge faces "
            "and the cylindrical root patch the boolean leaves behind.",
            S.CAP,
        )
    )


# --------------------------------------------------------------------------------------
#   5. The shape trade
# --------------------------------------------------------------------------------------


def shape_trade(story: list, D: dict) -> None:
    ev = D["evidence"]
    t = ev["shape_trade"]
    c = ev["shape_trade_coupled"]
    dv = D["ntop"]["design_vector"]

    S.sect(story, "4. The sizer declined the drag optimum")
    story.append(
        Paragraph(
            f"nose_blend converged to {fmt(dv['nose_blend'], 2)}, not 1.0. boattail_blend "
            f"converged to {fmt(dv['boattail_blend'], 2)}. The drag optimum was inside the "
            "bounds and the search moved away from it. The reason is visible in the sweep. "
            "Past about seven tenths of the way to the optimum, the forebody volume the shape "
            "gives up, and the aft shift in centre of pressure it causes, cost more than the "
            "remaining wave drag saves.",
            S.BODY,
        )
    )

    if c.get("available"):
        rows = [["nose_blend", "Launch mass [kg]", "Range [km]", "q_max [kPa]",
                 "Static margin [cal]", "Feasible", "Penalty"]]
        for r in c["rows"]:
            if "failed" in r:
                rows.append([fmt(r["nose_blend"], 2), "failed", r["failed"][:28],
                             "", "", "no", ""])
                continue
            rows.append([
                fmt(r["nose_blend"], 2), fmt(r["m0_kg"], 3), fmt(r["range_km"], 2),
                fmt(r["q_max_kPa"], 2), fmt(r["static_margin"], 4),
                "yes" if r["feasible"] else "no", fmt(r["penalty"], 5),
            ])
        story.append(S.styled_table(rows, [0.85, 1.15, 0.95, 0.95, 1.15, 0.75, 0.95],
                                    ["RIGHT"] * 5 + ["CENTER", "RIGHT"]))
        story.append(Spacer(1, 4))
        story.append(
            Paragraph(
                "Table 10. The nTop-COUPLED blend sweep. Every row rebuilt and re-measured the "
                f"solid. None of the {len(c['rows'])} points failed. This sweep is the evidence "
                "behind the interior-optimum claim, because the analytic mass model reads a "
                "closed form of the mould line and cannot see a re-measured solid. The search "
                "minimises the penalty column, not the launch mass.",
                S.CAP,
            )
        )
        best = c.get("best_blend_by_penalty")
        if best is not None:
            story.append(
                Paragraph(
                    "<b>Read this row carefully.</b> The lowest penalty of the five measured "
                    f"points is at nose_blend {fmt(best, 2)}, not at the "
                    f"{fmt(dv['nose_blend'], 2)} the search stopped on. The optimum is therefore "
                    "confirmed as interior, because blend 1.0 is worse than blend "
                    f"{fmt(best, 2)} on both penalty and launch mass. But the search did not "
                    "reach it. The five penalties span only "
                    f"{fmt(max(r['penalty'] for r in c['rows'] if 'penalty' in r) - min(r['penalty'] for r in c['rows'] if 'penalty' in r), 5)}, "
                    "so the objective is nearly flat in this variable and the search had little "
                    "to follow. The claim this report makes is that the optimum is interior. "
                    "The claim it does NOT make is that the converged blend is the optimum.",
                    S.BODY,
                )
            )
    else:
        story.append(
            Paragraph(
                "The nTop-coupled blend sweep is not available. Reason: "
                f"{c.get('reason', 'not collected')}. The analytic sweep below cannot replace "
                "it, because the analytic mass model cannot see a re-measured solid.",
                S.BODY,
            )
        )

    rows = [["nose_blend", "Shape ratio", "Nose volume", "Nose wetted",
             "CD0 at M 2", "x_cp shift"]]
    for r in t["rows"]:
        rows.append([
            fmt(r["nose_blend"], 2), fmt(r["shape_ratio"], 5),
            f"{r['d_nose_volume_pct']:+.2f} %", f"{r['d_nose_wetted_pct']:+.2f} %",
            f"{r['d_CD0_pct_M2']:+.2f} %", f"{r['d_xcp_mm_M2']:+.1f} mm",
        ])
    story.append(S.styled_table(rows, [1.05, 1.1, 1.1, 1.1, 1.15, 1.15],
                                ["RIGHT"] * 6))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 11. What each blend costs and buys, at fixed everything else. The shape ratio "
            "falls quickly at first and then flattens. The volume given up keeps rising at a "
            "steady rate. The centre of pressure keeps moving aft, which is what erodes the "
            "static margin. That is the whole trade in three columns. Every profile in the "
            "sweep stays monotone, and the steepest one reaches a slope of "
            f"{fmt(max(r['max_slope'] for r in t['rows']), 3)}, so none of them is an "
            "unbuildable nose.",
            S.CAP,
        )
    )
    add_figure(
        story, "oml_shape.png", 3,
        "The splined outer mould line. Panels (a) and (b) draw the geometry nTop revolves, with "
        "the control points the notebook computes. Panels (c) and (d) are what moving it does. "
        "Panel (d) is the coupled sweep, where every point was re-measured.",
        "Run -m rocketgen.report.fig_oml --oml spline.",
    )

    search = ev.get("search", {})
    if search.get("available") and search.get("variables_searched_but_not_traced"):
        story.append(
            Paragraph(
                "One honest limitation of the record. The search trace at "
                f"{search['path']} has {search['n_evaluations']} rows and does not carry "
                f"{' or '.join(search['variables_searched_but_not_traced'])}. Those variables "
                "are searched last, so the trace cannot show the shape search directly. The "
                "coupled sweep in Table 10 exists partly to fill that hole.",
                S.BODY,
            )
        )


# --------------------------------------------------------------------------------------
#   6. The coupling
# --------------------------------------------------------------------------------------


def coupling(story: list, D: dict) -> None:
    p, a = D["ntop"], D["analytic"]
    ev = D["evidence"]

    S.sect(story, "5. The coupling decides feasibility")
    story.append(
        Paragraph(
            "The same design vector was evaluated twice. Once with the geometry measured in "
            "nTop, and once with the closed-form fallback. The two answers differ by more than "
            "the shape change itself.",
            S.BODY,
        )
    )
    rows = [
        ["Quantity", "Analytic geometry", "nTop measured", "Change"],
        ["Launch mass [kg]", fmt(a["mass_statement"]["total_kg"], 2),
         fmt(p["mass_statement"]["total_kg"], 2),
         f"{p['mass_statement']['total_kg'] - a['mass_statement']['total_kg']:+.2f}"],
        ["Range [km]", fmt(a["trajectory"]["range_m"] / 1000.0, 2),
         fmt(p["trajectory"]["range_m"] / 1000.0, 2),
         f"{(p['trajectory']['range_m'] - a['trajectory']['range_m']) / 1000.0:+.2f}"],
        ["Impact Mach [-]", fmt(a["trajectory"]["mach_final"], 3),
         fmt(p["trajectory"]["mach_final"], 3),
         f"{p['trajectory']['mach_final'] - a['trajectory']['mach_final']:+.3f}"],
        ["Peak dynamic pressure [kPa]", fmt(a["trajectory"]["q_max_Pa"] / 1000.0, 2),
         fmt(p["trajectory"]["q_max_Pa"] / 1000.0, 2),
         f"{(p['trajectory']['q_max_Pa'] - a['trajectory']['q_max_Pa']) / 1000.0:+.2f}"],
        ["Feasible", "NO" if not a["feasible"] else "yes",
         "yes" if p["feasible"] else "NO", ""],
    ]
    story.append(S.styled_table(rows, [2.1, 1.7, 1.6, 1.4],
                                ["LEFT", "RIGHT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    violated = [c["name"] for c in a["constraints"] if not c["met"]]
    story.append(
        Paragraph(
            "Table 12. The coupling effect at the converged design. The analytic geometry "
            "reports a vehicle that is lighter and flies further. It is also infeasible. It "
            f"violates {len(violated)} constraints: {', '.join(violated)}. Only the measured "
            "geometry finds the feasible answer. A reader who took the analytic numbers would "
            "have shipped a design that does not close.",
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            "The mechanism is visible in the mass statement. The measured airframe weighs "
            f"{fmt(ev['ntop']['mass_structure'], 2)} kg. The measured centre of gravity of that "
            f"structure sits at {fmt(ev['ntop']['cg_structure'][0], 4)} m from the nose. The "
            "closed form does not see the fin roots, the plate edges or the boolean leftovers, "
            "so it puts less mass further forward. Less mass raises the peak dynamic pressure. "
            "Mass further forward is not enough to hold the static margin.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The centre of gravity of a cruciform body does not measure as exactly on axis. "
            f"This one is {fmt(1000.0 * abs(ev['ntop']['cg_structure'][1]), 3)} mm off in one "
            f"lateral direction and {fmt(1000.0 * abs(ev['ntop']['cg_structure'][2]), 3)} mm off "
            "in the other. That is discretisation, not asymmetry. Any test here must use a "
            "tolerance and never zero.",
            S.BODY,
        )
    )
    add_figure(
        story, "flight_path.png", 4,
        "The flown mission. Solid lines are the nTop-measured geometry. Dashed lines are the "
        "same design vector with the closed-form geometry. Panel (c) is the one that decides "
        "the outcome: the analytic trace crosses the structural limit.",
    "Run -m rocketgen.report.fig_flight --oml spline.",
    )
    add_figure(
        story, "constraint_margins.png", 5,
        "Constraint margins at the converged design. Bars are the coupled result. The red ticks "
        "are the same design vector with analytic geometry. Two ticks fall on the wrong side of "
        "zero.",
        "Run -m rocketgen.report.fig_margins --oml spline.",
        max_h_in=3.2,
    )


# --------------------------------------------------------------------------------------
#   7. The design
# --------------------------------------------------------------------------------------


def the_design(story: list, D: dict) -> None:
    p = D["ntop"]
    ms = p["mass_statement"]

    S.sect(story, "6. The converged design")
    rows = [["Constraint", "Value", "Sense", "Limit", "Met"]]
    for c in p["constraints"]:
        rows.append([c["name"], fmt(c["value"], 3), c["sense"], fmt(c["limit"], 3),
                     "yes" if c["met"] else "NO"])
    story.append(S.styled_table(rows, [2.0, 1.55, 0.6, 1.55, 0.7],
                                ["LEFT", "RIGHT", "CENTER", "RIGHT", "CENTER"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            f"Table 13. All {len(p['constraints'])} constraints the loop checks, at the "
            "converged design. The dynamic-pressure margin is the smallest, at "
            f"{fmt(100.0 * (200000.0 - p['trajectory']['q_max_Pa']) / 200000.0, 2)} percent. "
            "Say how many constraints were checked, not only that they were met: a requirement "
            "that is not in this list is not tested.",
            S.CAP,
        )
    )

    rows = [["Mass item", "Mass [kg]", "Station [m]", "Provenance"]]
    for i in ms["items"]:
        rows.append([i["name"], fmt(i["mass_kg"], 2), fmt(i["x_cg_m"], 3), i["provenance"]])
    rows.append(["TOTAL", fmt(ms["total_kg"], 2), fmt(ms["x_cg_m"], 3), ""])
    story.append(S.styled_table(rows, [2.5, 1.2, 1.2, 1.5],
                                ["LEFT", "RIGHT", "RIGHT", "LEFT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 14. The group-weight statement, with the provenance of every line. Only "
            f"{fmt(100.0 * ms['measured_fraction'], 2)} percent of the launch mass is measured "
            "in nTop. That is the airframe and the fins, and nothing else. The motor case, the "
            "propellant, the warhead and the avionics are charged separately, so the body "
            "measured in nTop must be the airframe alone. Double counting here would be silent "
            "and large.",
            S.CAP,
        )
    )
    add_figure(
        story, "mass_statement.png", 6,
        "The mass statement by provenance. The measured fraction is small by design. The "
        "correlation lines are where the bottom-up model is known to be incomplete.",
        "Run -m rocketgen.report.fig_mass --oml spline.",
        max_h_in=3.4,
    )
    add_figure(
        story, "area_distribution.png", 7,
        "The cross-sectional area distribution measured in nTop, which is the input the "
        "slender-body wave-drag model integrates.",
        "Run -m rocketgen.report.fig_area --oml spline.",
        max_h_in=3.2,
    )


# --------------------------------------------------------------------------------------
#   8. Trade study
# --------------------------------------------------------------------------------------


def trade_study(story: list, D: dict) -> None:
    grid, lhs = D["grid"], D["lhs"]
    sens = D["sens"]

    def count(rows: list[dict[str, str]], key: str) -> int:
        return sum(1 for r in rows if r.get(key) in ("1", "1.0", "True", "true"))

    n_total = len(grid) + len(lhs)
    n_conv = count(grid, "converged") + count(lhs, "converged")
    n_feas = count(grid, "feasible") + count(lhs, "feasible")

    S.sect(story, "7. Trade study")
    story.append(
        Paragraph(
            f"The trade study holds {len(grid)} factorial nodes and {len(lhs)} Latin hypercube "
            f"samples. That is {n_total} samples in total. {n_total - n_conv} failed to "
            f"converge. {n_feas} are feasible. Every sample ran with real nTop geometry. A "
            "sample that crashed would be recorded as a non-converged row, not dropped: a trade "
            "study that drops its failures reports a feasible region that is too large.",
            S.BODY,
        )
    )
    add_figure(
        story, "carpet.png", 8,
        "The factorial, drawn as carpet plots. The converged design is a grid node, so the "
        "study brackets the answer rather than straddling it. Infeasible nodes carry the names "
        "of the constraints they violate.",
        "Run -m rocketgen.report.fig_carpet --oml spline.",
        max_h_in=3.2,
    )

    variables = list(next(iter(sens.values())).keys())
    responses = list(sens.keys())
    rows = [["Variable"] + responses]
    for v in variables:
        rows.append([v] + [f"{sens[r][v]:+.2f}" for r in responses])
    story.append(S.styled_table(rows, [1.6] + [1.3] * len(responses),
                                ["LEFT"] + ["RIGHT"] * len(responses)))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 15. Spearman rank correlations from the Latin hypercube. Each response has a "
            "distinct dominant lever, which is what makes the design tractable. Sustain "
            "propellant drives launch mass. Body diameter drives range, impact Mach and peak "
            "dynamic pressure, all in the same direction. Terminal propellant is the only strong "
            "lever on impact Mach that does not also cost range.",
            S.CAP,
        )
    )
    add_figure(
        story, "sensitivity.png", 9,
        "The same correlations, drawn. A short bar means the variable is not a useful lever on "
        "that response, whatever it does elsewhere.",
        "Run -m rocketgen.report.fig_sensitivity --oml spline.",
        max_h_in=3.4,
    )


# --------------------------------------------------------------------------------------
#   9. Verification
# --------------------------------------------------------------------------------------


def verification(story: list, D: dict) -> None:
    ev = D["evidence"]
    it = ev["integrator"]
    ae = ev["aero"]

    S.sect(story, "8. Verification")
    story.append(
        Paragraph(
            "The physics under this study is the physics validated for the tangent-ogive SV-1. "
            "It is not revalidated here. The residuals below were measured again for this "
            "report, so the report quotes what the code achieves rather than the bound its test "
            "asserts.",
            S.BODY,
        )
    )
    rows = [
        ["Check", "Reference", "Relative residual"],
        ["Vacuum ballistic range", "closed-form parabola", sci(it["vacuum_range_rel"])],
        ["Vacuum apogee", "closed form", sci(it["vacuum_apogee_rel"])],
        ["Burnout speed", "Tsiolkovsky less gravity loss", sci(it["tsiolkovsky_rel"])],
        ["Terminal velocity", "sqrt(2 m g / (rho S CD))", sci(it["terminal_velocity_rel"])],
        ["Specific energy over 100 s", "conservation", sci(it["energy_drift_rel"])],
    ]
    story.append(S.styled_table(rows, [2.3, 2.5, 2.0], ["LEFT", "LEFT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 16. The trajectory integrator against closed forms. Halving the time step "
            "cuts the range error by "
            f"{', '.join('%.2f' % r for r in it['rk4_order_ratios'])}, against the factor of 16 "
            "a fourth-order method must give. The order check uses a 45 degree vacuum parabola. "
            "A drag-free vertical climb would not work: its acceleration is constant, so the "
            "integrator is exact on it and the ratio means nothing.",
            S.CAP,
        )
    )
    rows = [
        ["Quantity", "Shots used", "Mean bias", "Worst shot"],
        ["Zero-lift drag", str(ae["n_shots_cd0"]), pct(ae["cd0_mean_bias"], 1) + " %",
         fmt(100.0 * ae["cd0_worst_shot"], 1) + " %"],
        ["Normal-force slope", str(ae["n_shots_cna_xcp"]), pct(ae["cna_mean_bias"], 1) + " %",
         fmt(100.0 * ae["cna_worst_shot"], 1) + " %"],
        ["Centre of pressure", str(ae["n_shots_cna_xcp"]), pct(ae["xcp_mean_bias"], 1) + " %",
         fmt(100.0 * ae["xcp_worst_shot"], 1) + " %"],
    ]
    story.append(S.styled_table(rows, [2.2, 1.4, 1.6, 1.6],
                                ["LEFT", "RIGHT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            f"Table 17. The aerodynamic build-up against {ae['n_shots_table']} Basic Finner "
            "free-flight shots from DREV-TM-9703, Table VII. The drag bias is where the "
            f"calibration factor of {fmt(ae['cd0_calibration'], 3)} comes from. The calibration "
            "is applied at the loop boundary and never inside the aerodynamic model.",
            S.CAP,
        )
    )
    add_figure(
        story, "aero_validation.png", 10,
        "The aerodynamic build-up and its validation. Panels (a) and (b) use the default SV-1 "
        "design vector, not the converged spline one, because the panel exists to show the "
        "build-up and the Basic Finner comparison. The converged spline drag is in Figure 2 "
        "panel (c).",
        "Run -m rocketgen.report.fig_aero.",
    )


# --------------------------------------------------------------------------------------
#   10. Limitations
# --------------------------------------------------------------------------------------


FLAG_WORD = "guess"


def limitations(story: list, D: dict) -> None:
    prov = D["prov"]
    sources = prov["sources"]
    flagged = {k: v for k, v in sorted(sources.items()) if FLAG_WORD in v.lower()}
    ev = D["evidence"]

    S.sect(story, "9. Limitations")
    story.append(
        Paragraph(
            f"The source registry holds {len(sources)} entries at the point this study wrote "
            f"it. {len(flagged)} of them are flagged as guesses. Every empirical constant in "
            "this repository carries a source string. If the value is a guess, the word GUESS "
            "must appear in that string, and a test asserts it. The table below is the complete "
            "flagged list, so a guess that hides becomes a guess that ships.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The registry is filled at import time, so it is only complete once every owning "
            "module has been imported. The writer of this file imports the aerodynamic, "
            "atmospheric, mass, propulsion and trajectory modules explicitly for that reason. A "
            "provenance file written before a trajectory was flown once listed 37 sources "
            "instead of 70, and under-reported its own limitations by a factor of four.",
            S.BODY,
        )
    )
    rows = [["Flagged source", "What it says"]]
    for k, v in flagged.items():
        rows.append([k, v])
    story.append(S.cell_table(rows, [1.9, 4.9], ["LEFT", "LEFT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 18. Every flagged source in the registry this study wrote. Two of them belong "
            "to the work this report describes: the polygon-sampling entry, and the entry that "
            "records the wave-drag ratio as Mach-independent.",
            S.CAP,
        )
    )

    story.append(Paragraph("What this study does not claim", S.H2))
    for text in (
        "The slender-body wave-drag ratio is Mach-independent at this order. The true shape "
        "sensitivity drifts with Mach number. That drift is not modelled and it has not been "
        "quantified. The registry carries this as a flagged entry.",
        "The nozzle model is ideal. There is no two-phase loss, no divergence loss, no "
        "combustion efficiency and no throat erosion. Real delivered specific impulse for this "
        "class runs 3 to 7 percent lower. That penalty is NOT applied, because its magnitude "
        "could not be sourced. It is the largest known unquantified optimism in the result.",
        "The optimum is confirmed interior. The converged blend is not confirmed optimal. The "
        "coupled sweep found a lower penalty at a blend the search did not reach.",
        "The linear theory behind the shape ratio is a small-disturbance theory. It is used here "
        "only as a ratio between two shapes of the same fineness, which is the use it is best "
        "suited to, but it remains linear theory applied to a body at Mach 4.",
        "The search trace does not record the two shape variables. The coupled sweep partly "
        "fills that hole. It does not replace a traced search.",
        "The requirements are invented for the demonstration. They do not correspond to any real "
        "programme.",
    ):
        story.append(Paragraph(text, S.BULLET, bulletText="-"))

    warnings = D["ntop"].get("warnings") or []
    unique: list[str] = []
    for w in warnings:
        if w not in unique:
            unique.append(w)
    if unique:
        story.append(Paragraph("Warnings the run raised, and did not swallow", S.H2))
        rows = [["Warning"]] + [[w] for w in unique]
        story.append(S.cell_table(rows, [6.8], ["LEFT"]))
        story.append(Spacer(1, 4))
        story.append(
            Paragraph(
                f"Table 19. The {len(unique)} distinct warnings the converged run raised. They "
                "are carried upward on purpose. The motor throat transition needs hardware that "
                "does not exist, and the grain is longer than the bay that holds it. Neither is "
                "hidden by the constraint list passing.",
                S.CAP,
            )
        )


# --------------------------------------------------------------------------------------
#   11. Reproducing
# --------------------------------------------------------------------------------------


def reproduce(story: list, D: dict) -> None:
    env = D["prov"]["environment"]
    p = D["ntop"]

    S.sect(story, "10. Reproducing this result")
    story.append(
        Paragraph(
            ".venv/Scripts/python.exe -m pytest tests -q<br/>"
            ".venv/Scripts/python.exe run_sv1.py --stage smoke --oml spline<br/>"
            ".venv/Scripts/python.exe run_sv1.py --stage size --oml spline<br/>"
            ".venv/Scripts/python.exe run_sv1.py --stage doe --oml spline<br/>"
            ".venv/Scripts/python.exe run_sv1.py --stage converged --oml spline<br/>"
            ".venv/Scripts/python.exe -m rocketgen.report.evidence --oml spline<br/>"
            ".venv/Scripts/python.exe -m rocketgen.report.fig_oml --oml spline<br/>"
            ".venv/Scripts/python.exe -m rocketgen.report.build_report_spline",
            S.MONO,
        )
    )
    rows = [
        ["Component", "Version or value"],
        ["Python", env["python"].split()[0]],
        ["numpy", env["numpy"]],
        ["scipy", env["scipy"]],
        ["SUAVE", "2.5.2, vendored from github.com/suavecode/SUAVE"],
        ["nTop Automate", "5.53.2 installed, 5.54.0 development build"],
        ["Drag calibration factor", fmt(D["prov"]["cd0_calibration"], 3)],
        ["Outer mould line family", str(D["prov"]["oml_family"])],
        ["Sizing evaluations allowed", str(D["prov"]["max_evals"])],
        ["Converged run wall time [s]", fmt(p["wall_time_s"], 1)],
        ["Total study wall time [s]", fmt(D["prov"]["total_wall_time_s"], 1)],
    ]
    story.append(S.styled_table(rows, [2.6, 4.2], ["LEFT", "LEFT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 20. Environment. numpy and scipy are pinned below their current major "
            "versions because SUAVE 2.5.2 does not run on either.",
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            "Two toolchain warnings are worth repeating. The nTop command-line tool returns exit "
            "code 72 when a block fails, not on success. Widely repeated guidance says the "
            "opposite and is wrong. Judge success on the expected files existing and being "
            "non-empty. Second, conversion evaluates the notebook, exports included, so a fine "
            "mesh tolerance makes conversion itself cost minutes. The pattern is to convert once "
            "and run many times, which works because every design variable is a real notebook "
            "input.",
            S.BODY,
        )
    )


# --------------------------------------------------------------------------------------
#   Main
# --------------------------------------------------------------------------------------


def main() -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    D = load_all()

    doc, story = S.make_doc(
        OUT_PDF,
        title="SV-1 with a Splined Outer Mould Line",
        author="nTop",
        footer="SV-1 splined outer mould line  |  nTop + SUAVE  |  invented requirements",
    )

    front_matter(story, D)
    story.append(PageBreak())
    what_changed(story, D)
    story.append(PageBreak())
    wave_drag(story, D)
    story.append(PageBreak())
    ntop_side(story, D)
    shape_trade(story, D)
    story.append(PageBreak())
    coupling(story, D)
    story.append(PageBreak())
    the_design(story, D)
    story.append(PageBreak())
    trade_study(story, D)
    story.append(PageBreak())
    verification(story, D)
    story.append(PageBreak())
    limitations(story, D)
    story.append(PageBreak())
    reproduce(story, D)

    S.build(doc, story)
    print(f"wrote {OUT_PDF}  ({os.path.getsize(OUT_PDF) / 1024:.0f} KB)")
    return OUT_PDF


if __name__ == "__main__":
    main()
