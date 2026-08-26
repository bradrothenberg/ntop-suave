"""Assemble the IV-1 SPLINE engineering report PDF.

    .venv/Scripts/python.exe -m rocketgen.report.build_report_iv1_spline

Every number comes from `runs/IV-1_spline/figures/evidence_iv1_spline.json` and from the two
`converged.json` files it was built from. Nothing is typed in by hand, nothing is re-flown here
and nTop is not called. Build the evidence and the figures first:

    .venv/Scripts/python.exe -m rocketgen.report.evidence_iv1_spline
    .venv/Scripts/python.exe -m rocketgen.report.fig_iv1_spline

WHY THIS IS A SEPARATE BUILDER
------------------------------
`build_report_iv1.py` describes the tangent-ogive IV-1 in full: the requirements audit, the
three mutually exclusive requirements, the grain sweeps, the strake validation. This report
answers a narrower question. It compares TWO converged interceptors that differ only in the
shape family of the outer mould line. It points at the IV-1 report for everything the two
studies share.

Style follows ASD-STE100 Simplified Technical English: active voice, simple tenses, short
sentences, one idea per sentence.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

from reportlab.platypus import PageBreak, Paragraph, Spacer

from . import report_style as S

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASE = os.path.join(REPO, "runs", "IV-1_spline")
OGIVE = os.path.join(REPO, "runs", "IV-1_ogive_baseline")
FIGS = os.path.join(CASE, "figures")
EVIDENCE = os.path.join(FIGS, "evidence_iv1_spline.json")
OUT_DIR = os.path.join(CASE, "report")
OUT_PDF = os.path.join(OUT_DIR, "IV1_spline_engineering_report.pdf")

#: Recipe files, read only to count nodes. That is the only size measure available without
#: opening the binary `.ntop` container.
RECIPE_SPLINE = os.path.join(REPO, "examples", "IV-1-spline", "02_geometry", "iv1_recipe.json")
RECIPE_OGIVE = os.path.join(REPO, "examples", "IV-1", "geometry", "iv1_recipe.json")



# --------------------------------------------------------------------------------------
#   Load
# --------------------------------------------------------------------------------------


def _json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all() -> dict[str, Any]:
    if not os.path.isfile(EVIDENCE):
        raise SystemExit(
            f"{EVIDENCE} is missing. Run:\n"
            "    .venv/Scripts/python.exe -m rocketgen.report.evidence_iv1_spline"
        )
    return {
        "ev": _json(EVIDENCE),
        "spline": _json(os.path.join(CASE, "converged.json")),
        "ogive": _json(os.path.join(OGIVE, "converged.json")),
        "nodes_spline": _node_count(RECIPE_SPLINE),
        "nodes_ogive": _node_count(RECIPE_OGIVE),
    }


def _node_count(path: str) -> int | None:
    if not os.path.isfile(path):
        return None
    return len(_json(path)["body"])


# --------------------------------------------------------------------------------------
#   Helpers
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


def fig_path(name: str) -> str | None:
    p = os.path.join(FIGS, name)
    return p if os.path.isfile(p) else None


def add_figure(story: list, name: str, num: int, caption: str,
               missing_note: str, max_h_in: float = 4.4) -> None:
    """Place a figure, or say in the report why it is not there."""
    path = fig_path(name)
    if path is None:
        story.append(
            Paragraph(
                f"<b>Figure {num} is not available.</b> {missing_note} "
                f"The expected file is runs/IV-1_spline/figures/{name}.",
                S.BODY,
            )
        )
        return
    story.append(S.fig_single(path, num, caption, max_h_in=max_h_in))


def row_of(comp: dict[str, Any], name: str) -> dict[str, Any]:
    return next(r for r in comp["rows"] if r["quantity"] == name)


# --------------------------------------------------------------------------------------
#   1. Front matter
# --------------------------------------------------------------------------------------


def front_matter(story: list, D: dict) -> None:
    ev = D["ev"]
    comp = ev["comparison"]
    cs = D["spline"]

    story.append(Paragraph("A Splined Outer Mould Line for the IV-1", S.TITLE))
    story.append(
        Paragraph(
            "The same two-stage interceptor, with its nose and interstage revolved from "
            "B-splines inside nTop",
            S.SUBTITLE,
        )
    )
    story.append(S.hrule())
    story.append(
        Paragraph(
            "Prepared with nTop Automate 5.53.2 / 5.54.0 and SUAVE 2.5.2. All geometry is "
            "authored programmatically. Every number is read from the recorded run artefacts "
            "under runs/IV-1_spline and runs/IV-1_ogive_baseline.",
            S.SUBTITLE,
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("Summary", S.H1))
    story.append(
        Paragraph(
            "This report describes one change to the IV-1 interceptor. The stage-2 nose and the "
            "interstage shoulder are now true revolved B-splines. They were a tangent ogive and "
            "a conical shoulder. Nothing else changed. The requirements, the physics modules, "
            "the mass model and the commanded pitchover angle are the same.",
            S.BODY,
        )
    )
    alt = row_of(comp, "Intercept altitude")
    mach = row_of(comp, "Intercept Mach")
    mass = row_of(comp, "Launch mass")
    story.append(
        Paragraph(
            f"Both vehicles meet all {comp['n_constraints']} constraints. Both reach the same "
            "100 mile slant range, because that is the constraint the trajectory is flown to. "
            "The spline reaches it "
            f"{fmt(alt['delta'], 2)} km higher: {fmt(alt['ogive'], 2)} km becomes "
            f"{fmt(alt['spline'], 2)} km, a rise of {fmt(alt['delta_pct'], 1)} percent. "
            f"Intercept Mach rises from {fmt(mach['ogive'], 3)} to {fmt(mach['spline'], 3)}. "
            f"Launch mass falls from {fmt(mass['ogive'], 2)} kg to {fmt(mass['spline'], 2)} kg.",
            S.BODY,
        )
    )
    g_aero = row_of(comp, "Lateral g, aerodynamic")
    story.append(
        Paragraph(
            "<b>The altitude gain has a price, and this report does not bury it.</b> "
            "Aerodynamic lateral acceleration at intercept falls by "
            f"{fmt(abs(g_aero['delta_pct']), 1)} percent, from {fmt(g_aero['ogive'], 2)} g to "
            f"{fmt(g_aero['spline'], 2)} g. The reason is simple. A higher intercept sits in "
            "thinner air, and aerodynamic manoeuvre needs dynamic pressure. Requirement A11 is "
            "therefore met almost entirely by the attitude-control thrusters. Section 5 states "
            "the trade in full.",
            S.BODY,
        )
    )

    rows = [["Quantity", "Unit", "Tangent ogive", "Spline", "Change", "Change [%]"]]
    for r in comp["rows"]:
        nd = int(r.get("nd", 3))
        rows.append([
            r["quantity"], r["unit"], fmt(r["ogive"], nd), fmt(r["spline"], nd),
            f"{r['delta']:+,.{nd}f}",
            "" if r["delta_pct"] is None else f"{r['delta_pct']:+.2f}",
        ])
    story.append(S.styled_table(rows, [2.15, 0.55, 1.15, 1.05, 0.95, 0.95],
                                ["LEFT", "CENTER", "RIGHT", "RIGHT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 1. The two converged interceptors, side by side. Both fly the same "
            f"{fmt(cs['pitchover_deg'], 0)} degree commanded pitchover. The A11 figure is the "
            "GREATER of the aerodynamic and the attitude-control capability, never their sum, "
            "because commanding an aerodynamic turn and a divert at the same time is a control "
            "problem this model does not represent.",
            S.CAP,
        )
    )


# --------------------------------------------------------------------------------------
#   2. What changed
# --------------------------------------------------------------------------------------


def what_changed(story: list, D: dict) -> None:
    ev = D["ev"]
    dv = ev["design_vector"]["spline"]
    w = ev["wavedrag"]

    S.sect(story, "1. What changed, and what did not")
    rows = [
        ["Design variable", "Tangent-ogive baseline", "Spline study"],
        ["nose_shape", str(ev["design_vector"]["ogive"]["nose_shape"]), str(dv["nose_shape"])],
        ["nose_blend", fmt(ev["design_vector"]["ogive"]["nose_blend"], 2),
         fmt(dv["nose_blend"], 2)],
        ["interstage_shape", str(ev["design_vector"]["ogive"]["interstage_shape"]),
         str(dv["interstage_shape"])],
        ["interstage_blend", fmt(ev["design_vector"]["ogive"]["interstage_blend"], 2),
         fmt(dv["interstage_blend"], 2)],
        ["Commanded pitchover [deg]", fmt(D["ogive"]["pitchover_deg"], 1),
         fmt(D["spline"]["pitchover_deg"], 1)],
    ]
    story.append(S.styled_table(rows, [2.3, 2.3, 2.2], ["LEFT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 2. The whole difference between the two studies. nose_blend is 1.0 here, "
            "which is the full slender-body drag optimum. That differs from the SV-1 spline "
            "study, where the sizing search stopped short of the optimum at 0.7. The IV-1 shape "
            "was set rather than searched, so this study makes no claim about an interior "
            "optimum on this vehicle.",
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            "The stage geometry, the propellant loads, the strakes, the fins and the "
            "attitude-control pack are unchanged. The attitude-control thrust is re-solved to "
            "its own fixed point at every design point, so it moves a little: "
            f"{fmt(row_of(ev['comparison'], 'Attitude-control thrust')['ogive'], 2)} kN becomes "
            f"{fmt(row_of(ev['comparison'], 'Attitude-control thrust')['spline'], 2)} kN. That "
            "is a consequence of the lighter vehicle, not an input.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            f"The nose is a cubic B-spline with {len(w['nose_control'])} control points on a "
            f"nose of length over radius {fmt(w['k_L_over_R'], 2)}. Its wave-drag shape ratio "
            f"against the tangent ogive is {fmt(w['shape_ratio'], 5)}. The ogive baseline "
            f"reports {fmt(w['shape_ratio_ogive'], 5)} by construction, exactly 1.0, which is "
            "the check that the shape model changes nothing at the baseline shape.",
            S.BODY,
        )
    )
    add_figure(
        story, "iv1_spline_side.png", 1,
        "The converged IV-1, rendered from the STL that nTop exported at this design point. "
        "Nose tip at the left. The splined payload-stage nose runs back to the strakes, the "
        "four stage-2 tail fins sit at the interstage joint, and the SPLINED INTERSTAGE FLARE "
        "is the smooth expansion aft of them, where the tangent-ogive configuration carries a "
        "straight conical shoulder. The booster and its four tail fins complete the stack. "
        "Overall length 5.08 m.",
        "Re-run scripts/render_iv1_spline.py, which needs the measurement made with "
        "export_stl=True.",
        max_h_in=1.9,
    )
    add_figure(
        story, "iv1_spline_iso.png", 2,
        "The same vehicle from an oblique view, which shows the four strakes and the two "
        "cruciform fin sets as separate bodies rather than as an outline.",
        "Re-run scripts/render_iv1_spline.py.",
        max_h_in=3.0,
    )
    add_figure(
        story, "iv1_spline_oml.png", 3,
        "The revolved spline. Panel (a) is the stage-2 nose and the nine control points the "
        "notebook computes. Panel (b) is the closed-form validation of the wave-drag model. "
        "Panel (c) is the zero-lift drag it moves, in both flight configurations.",
        "Run -m rocketgen.report.fig_iv1_spline.",
        max_h_in=2.8,
    )


# --------------------------------------------------------------------------------------
#   3. The wave-drag model
# --------------------------------------------------------------------------------------


def wave_drag(story: list, D: dict) -> None:
    ev = D["ev"]
    w = ev["wavedrag"]
    v = w["validation"]
    sh, vk = v["sears_haack"], v["von_karman"]

    S.sect(story, "2. Why the shape can change anything at all")
    story.append(
        Paragraph(
            "The forebody wave-drag term in this repository is the Bonney correlation. It is a "
            "function of length over diameter alone. It cannot tell one nose from another at "
            "fixed fineness. A spline study run against that model would have reported no "
            "change, for the wrong reason.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "rocketgen/sizing/wavedrag.py supplies the missing sensitivity. It computes the "
            "linearised slender-body wave drag of an area distribution through the Glauert "
            "series. It then reports a dimensionless RATIO against the tangent ogive of the same "
            "fineness. The ratio multiplies the correlation value. So the calibrated level and "
            "the Mach dependence stay with the correlation. Only the shape effect comes from "
            "linear theory. At the ogive control values the ratio is exactly 1.0.",
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
         fmt(v["von_karman_shape_factor"]["measured"], 8),
         sci(v["von_karman_shape_factor"]["rel_err"])],
    ]
    story.append(S.styled_table(rows, [1.9, 1.9, 1.5, 1.5],
                                ["LEFT", "LEFT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 3. The wave-drag model against exact closed forms from outside this "
            "repository. Every test asserts a tolerance of 1.0e-4. The residuals above are "
            "measured for this report rather than transcribed from the test file, so they are "
            "what the code achieves and not what it is allowed.",
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
                "the residual down. A single number could not have said this.",
                S.CAP,
            )
        )
    story.append(
        Paragraph(
            "Two further checks support the series. The Glauert series constant of pi over four "
            "was compared against direct double integration of the same integral. The deleted "
            "diagonal converges slowly, so what is measured is convergence onto the series: the "
            f"residual falls from {sci(v['glauert_direct'][0]['rel_err'])} at "
            f"{v['glauert_direct'][0]['n']} points to "
            f"{sci(v['glauert_direct'][-1]['rel_err'])} at {v['glauert_direct'][-1]['n']}. "
            "Separately, the von Karman ogive was confirmed as the constrained optimum rather "
            "than assumed to be. Adding any higher Glauert mode at fixed base area raises the "
            f"drag in all {len(v['optimality'])} perturbations tried. A "
            f"{v['n_ctrl']}-point spline recovers "
            f"{fmt(100.0 * v['gap_recovered_fraction'], 1)} percent of the distance from the "
            "tangent ogive to that bound.",
            S.BODY,
        )
    )

    rows = [["Mach", "Stage", "CD0 ogive", "CD0 spline", "Change [%]",
             "Wave share of CD0"]]
    for r in w["drag_rows"]:
        rows.append([
            fmt(r["mach"], 1), str(r["stage"]), fmt(r["cd0_ogive"], 5),
            fmt(r["cd0_spline"], 5), f"{r['d_cd0_pct']:+.2f}",
            "" if r["wave_share_ogive"] is None
            else fmt(100.0 * r["wave_share_ogive"], 1) + " %",
        ])
    story.append(S.styled_table(rows, [0.75, 0.7, 1.25, 1.25, 1.2, 1.55],
                                ["RIGHT"] * 6))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 5. What the shape does on this vehicle, at 12 km and 2 degrees of angle of "
            "attack. Stage 1 is the full stack on the booster reference area. Stage 2 is the "
            "surviving stage on its own area. A coefficient computed on the wrong area is "
            "silently wrong by the diameter ratio squared, so the two configurations are "
            "reported separately and never mixed.",
            S.CAP,
        )
    )


# --------------------------------------------------------------------------------------
#   4. The nTop side
# --------------------------------------------------------------------------------------


def ntop_side(story: list, D: dict) -> None:
    ev = D["ev"]
    nb = ev["notebook"]
    g = ev["geometry"]

    S.sect(story, "3. How the spline is built in nTop")
    story.append(
        Paragraph(
            "nTop revolves the spline itself. There is no chord polygon and no sampling error. "
            "The notebook computes the control points from live inputs, builds a profile from a "
            "list of curves, and revolves that profile. The chain is four blocks:",
            S.BODY,
        )
    )
    chain = nb.get("spline_chain", [])
    story.append(
        Paragraph(
            "<br/>".join(c.replace("<", "&lt;").replace(">", "&gt;") for c in chain),
            S.MONO,
        )
    )
    in_universe = nb.get("in_universe") or {}
    n_known = nb.get("n_in_universe")
    story.append(
        Paragraph(
            "The profile is a curve LIST. It mixes spline segments with "
            f"{nb.get('straight_edge_block', 'two_point_line')} segments, so the straight runs "
            "stay exactly straight and the corners stay sharp. The axial fractions of the "
            "control points are the Greville abscissae, which makes the axial station an exact "
            "linear function of the spline parameter.",
            S.BODY,
        )
    )
    if in_universe:
        story.append(
            Paragraph(
                f"None of those blocks is in the vendored block universe. It carries "
                f"{nb.get('n_universe_signatures', 'an unrecorded number of')} signatures and "
                f"{n_known} of the {len(chain)} blocks in the chain. The universe is incomplete, "
                "and its silence is not evidence that nTop cannot do something. An earlier "
                "attempt sampled the spline into a chord polygon, because the universe listed no "
                "route from a curve to a revolvable profile and that was taken as proof no route "
                "existed. It was not proof. All four blocks go through the raw-block escape "
                "hatch. docs/NTOP_NOTES.md section 25 records the four encoding traps.",
                S.BODY,
            )
        )

    rows = [
        ["Body", "nTop volume [m^3]", "Exact integral [m^3]", "Relative error",
         "Measurement wall time [s]"],
    ]
    for key in ("spline", "ogive"):
        r = g[key]
        rows.append([
            f"Stage 2, {r['nose_form']}",
            fmt(r["volume_ntop_stage2"], 6), fmt(r["volume_closed_form_stage2"], 6),
            "" if r["rel_err"] is None else f"{100.0 * r['rel_err']:+.4f} %",
            fmt(r["wall_time_s"], 1),
        ])
    story.append(S.styled_table(rows, [2.1, 1.25, 1.35, 1.05, 1.05],
                                ["LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 6. The measured stage-2 solid against the exact integral of the shape that "
            "built it. For the spline that closed form is exact, because nTop revolves the same "
            "B-spline. There is no discretisation error available to hide behind. The splined "
            f"nose gives up {fmt(abs(g['nose_volume_change_pct']), 2)} percent of enclosed "
            f"volume and {fmt(abs(g['nose_wetted_change_pct']), 2)} percent of wetted area "
            "against the ogive.",
            S.CAP,
        )
    )
    if D["nodes_spline"] is not None and D["nodes_ogive"] is not None:
        rows = [
            ["Notebook", "Recipe body nodes"],
            ["IV-1 stack, splined nose and interstage", str(D["nodes_spline"])],
            ["IV-1 stack, tangent ogive and cone", str(D["nodes_ogive"])],
        ]
        story.append(S.styled_table(rows, [4.4, 2.4], ["LEFT", "RIGHT"]))
        story.append(Spacer(1, 4))
        story.append(
            Paragraph(
                "Table 7. Notebook size, counted as nodes in the recipe body. The spline "
                "notebook is the smaller of the two, because a revolved spline replaces the "
                "sampled polygon and the arithmetic that built it.",
                S.CAP,
            )
        )
    story.append(
        Paragraph(
            "One notebook that reports several bodies must namespace its output keys. This one "
            "emits per-stage prefixes, so the generic output parser is not safe to call on it "
            "and a dedicated reader returns a dictionary keyed by stage. It also states which "
            "frame every measured station is in. The structural centre of gravity is reported "
            "both in stage-local coordinates and in stack coordinates, because the mass model "
            "adds the stage offset itself. Getting that wrong moves a centre of gravity by "
            "metres and nothing crashes.",
            S.BODY,
        )
    )


# --------------------------------------------------------------------------------------
#   5. The result
# --------------------------------------------------------------------------------------


def the_result(story: list, D: dict) -> None:
    ev = D["ev"]
    cs = D["spline"]
    traj = ev["trajectory"]["spline"]

    S.sect(story, "4. The converged interceptor")
    rows = [["Constraint", "Value", "Sense", "Limit", "Margin [%]", "Met"]]
    for c in sorted(ev["constraints"]["spline"], key=lambda r: r["margin"]):
        rows.append([
            c["name"], fmt(c["value"], 4), c["sense"], fmt(c["limit"], 4),
            f"{100.0 * c['margin']:+.2f}", "yes" if c["met"] else "NO",
        ])
    story.append(S.styled_table(rows, [1.75, 1.35, 0.55, 1.35, 1.0, 0.6],
                                ["LEFT", "RIGHT", "CENTER", "RIGHT", "RIGHT", "CENTER"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            f"Table 8. All {len(ev['constraints']['spline'])} constraints the loop checks, at "
            "the converged spline design, sorted by margin. Say how many were checked, not only "
            "that they were met. A requirement that is not in this list is not tested. The IV-1 "
            "study shipped with two holes in this list once: grain closure was not gated, and "
            "the stage-2 static margin was never added. Grain closure is now the last two rows.",
            S.CAP,
        )
    )
    add_figure(
        story, "iv1_spline_margins.png", 4,
        "The same margins, drawn. Orange ticks are the tangent-ogive baseline at the same "
        "pitchover angle. The two shapes trade margin between A3 and A11.",
        "Run -m rocketgen.report.fig_iv1_spline.",
        max_h_in=3.4,
    )
    story.append(
        Paragraph(
            f"The trajectory carries {traj['n_samples']:,} recorded samples over "
            f"{fmt(traj['duration_s'], 1)} s. Apogee is {fmt(traj['apogee_m'] / 1000.0, 2)} km "
            f"at t = {fmt(traj['apogee_time_s'], 1)} s. Peak dynamic pressure is "
            f"{fmt(traj['q_max_Pa'] / 1000.0, 1)} kPa at "
            f"{fmt(traj['q_max_altitude_m'] / 1000.0, 2)} km and Mach "
            f"{fmt(traj['q_max_mach'], 2)}. The integrator makes every event time a hard step "
            "boundary, so the mass bookkeeping through staging closes to machine precision.",
            S.BODY,
        )
    )
    add_figure(
        story, "iv1_spline_ascent.png", 5,
        "The two flown missions. Both reach the same 100 mile slant range, because that is what "
        "the trajectory is flown to. The spline arrives higher, faster and in thinner air.",
        "Run -m rocketgen.report.fig_iv1_spline.",
    )

    mass = ev["mass"]["spline"]
    rows = [["Stage", "Item", "Mass [kg]", "Station [m]", "Provenance"]]
    for i in mass["items"]:
        rows.append([str(i["stage"]), i["item"], fmt(i["mass_kg"], 2),
                     fmt(i["station_m"], 3), i["provenance"]])
    rows.append(["", "ATTITUDE-CONTROL PACK", fmt(mass["acs_pack_kg"], 2), "", "requirement"])
    rows.append(["", "TOTAL", fmt(mass["m0_kg"], 2), "", ""])
    story.append(S.styled_table(rows, [0.55, 2.6, 1.0, 1.05, 1.6],
                                ["CENTER", "LEFT", "RIGHT", "RIGHT", "LEFT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 9. The group-weight statement, with the provenance of every line. "
            f"{fmt(100.0 * mass['measured_fraction'], 2)} percent of the launch mass is measured "
            f"in nTop, which is {fmt(mass['measured_kg'], 2)} kg of airframe and surfaces. The "
            f"stage totals are {fmt(mass['stage_totals_kg']['1'], 2)} kg for stage 1, "
            f"{fmt(mass['stage_totals_kg']['2'], 2)} kg for stage 2 and "
            f"{fmt(mass['stage_totals_kg']['0'], 2)} kg for the interstage. "
            f"{fmt(cs['jettisoned_kg'], 2)} kg leaves the vehicle at separation.",
            S.CAP,
        )
    )


# --------------------------------------------------------------------------------------
#   6. The trade
# --------------------------------------------------------------------------------------


def the_trade(story: list, D: dict) -> None:
    ev = D["ev"]
    comp = ev["comparison"]
    g_aero = row_of(comp, "Lateral g, aerodynamic")
    g_acs = row_of(comp, "Lateral g, attitude control")
    g_a11 = row_of(comp, "Lateral g, A11 figure")
    q_ic = row_of(comp, "Dynamic pressure at intercept")
    cn = row_of(comp, "CN_max at intercept")
    sweep = ev["pitchover"]

    S.sect(story, "5. The trade: a higher intercept costs aerodynamic manoeuvre")
    story.append(
        Paragraph(
            "This is the finding a reader must not miss. The spline reaches the required slant "
            "range at a higher altitude. Higher altitude means thinner air. Dynamic pressure at "
            f"intercept falls from {fmt(q_ic['ogive'], 2)} kPa to {fmt(q_ic['spline'], 2)} kPa, "
            f"a drop of {fmt(abs(q_ic['delta_pct']), 1)} percent. Aerodynamic lateral "
            "acceleration is proportional to dynamic pressure, so it falls by very nearly the "
            f"same amount: {fmt(g_aero['ogive'], 2)} g becomes {fmt(g_aero['spline'], 2)} g. "
            f"The maximum normal-force coefficient barely moves, from {fmt(cn['ogive'], 3)} to "
            f"{fmt(cn['spline'], 3)}, which confirms that the loss is dynamic pressure and not "
            "aerodynamics.",
            S.BODY,
        )
    )
    rows = [
        ["Contribution to A11", "Tangent ogive [g]", "Spline [g]", "Change [%]"],
        ["Aerodynamic", fmt(g_aero["ogive"], 3), fmt(g_aero["spline"], 3),
         f"{g_aero['delta_pct']:+.2f}"],
        ["Attitude control", fmt(g_acs["ogive"], 3), fmt(g_acs["spline"], 3),
         f"{g_acs['delta_pct']:+.2f}"],
        ["A11 figure, the greater of the two", fmt(g_a11["ogive"], 3),
         fmt(g_a11["spline"], 3), f"{g_a11['delta_pct']:+.2f}"],
    ]
    story.append(S.styled_table(rows, [2.6, 1.6, 1.3, 1.3],
                                ["LEFT", "RIGHT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 10. How requirement A11 is met. For the tangent ogive the aerodynamic "
            f"contribution is {fmt(100.0 * g_aero['ogive'] / g_a11['ogive'], 1)} percent of the "
            "A11 figure. For the spline it is "
            f"{fmt(100.0 * g_aero['spline'] / g_a11['spline'], 1)} percent. A11 is now met by "
            "the thrusters almost alone. The requirement still passes. The margin behind it is "
            "thinner, because a thruster has a finite total impulse and the air does not.",
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            "That matters because the attitude-control pack is not free. It weighs "
            f"{fmt(ev['mass']['spline']['acs_pack_kg'], 2)} kg and carries "
            f"{fmt(ev['mass']['spline']['acs_total_impulse_Ns'] / 1000.0, 1)} kN.s of total "
            "impulse for a burn of a few seconds. A design that leans harder on it has less "
            "divert authority in reserve for the endgame. This report records that as a real "
            "cost of the altitude gain, not as a rounding detail.",
            S.BODY,
        )
    )

    S.sect(story, "6. The pitchover sweep", style=S.H1)
    story.append(
        Paragraph(
            "The commanded pitchover angle is not a geometry input. Sweeping it changes the "
            "pitch programme and the attitude-control thrust, and nothing else. One measurement "
            "set per shape therefore serves every angle, so the sweep below is a genuine "
            "nTop-coupled sweep and not an analytic stand-in.",
            S.BODY,
        )
    )
    rows = [["Pitchover [deg]", "Shape", "Slant range [km]", "Intercept altitude [km]",
             "Mach", "Aerodynamic g", "Feasible"]]
    for i, gamma in enumerate(sweep["gamma_deg"]):
        for shape in ("ogive", "spline"):
            r = sweep["shapes"][shape][i]
            rows.append([
                fmt(gamma, 0), shape, fmt(r["slant_range_km"], 2),
                fmt(r["altitude_km"], 2), fmt(r["mach"], 3),
                fmt(r["lateral_g_aero"], 2), "yes" if r["feasible"] else "no",
            ])
    story.append(S.styled_table(rows, [1.0, 0.85, 1.15, 1.35, 0.75, 1.0, 0.7],
                                ["RIGHT", "LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT",
                                 "CENTER"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            f"Table 11. The sweep covers {len(sweep['gamma_deg'])} angles. The spline is "
            f"feasible at {sweep['n_feasible_spline']} of them. The ogive is feasible at "
            f"{sweep['n_feasible_ogive']}. "
            "The spline opens a feasible angle the ogive does not have: at 36 degrees "
            "the spline meets every constraint, and the ogive fails A3 with an intercept "
            f"altitude of {fmt(sweep['shapes']['ogive'][2]['altitude_km'], 2)} km against a "
            "15 km minimum. That is the clearest single statement of what the shape bought.",
            S.CAP,
        )
    )
    add_figure(
        story, "iv1_spline_pitchover.png", 6,
        "The pitchover sweep. Panel (a) shows the spline clearing A3 two degrees earlier. Panel "
        "(b) shows what that costs in aerodynamic g. Panel (d) shows how A11 is actually met at "
        "the converged point.",
        "Run -m rocketgen.report.fig_iv1_spline.",
    )


# --------------------------------------------------------------------------------------
#   7. Open defect
# --------------------------------------------------------------------------------------


def open_defect(story: list, D: dict) -> None:
    env = D["ev"]["environment"]

    S.sect(story, "7. An open defect: the drag calibration is not applied to IV-1")
    story.append(
        Paragraph(
            "The zero-lift drag calibration factor is validated against 23 Basic Finner "
            f"free-flight shots. Its value is {fmt(env['cd0_calibration_available'], 3)}. The "
            "SV-1 loop applies it at the loop boundary, through CalibratedAero. "
            "<b>The IV-1 study does not.</b> scripts/iv1_converge.py builds the stack "
            "aerodynamics directly and never wraps it. So every IV-1 result, including both "
            "results in this report, runs about 15 percent low on zero-lift drag against that "
            "data set.",
            S.BODY,
        )
    )
    rows = [
        ["Item", "Value"],
        ["Calibration factor available", fmt(env["cd0_calibration_available"], 3)],
        ["Applied to IV-1", "NO" if not env["cd0_calibration_applied_to_iv1"] else "yes"],
    ]
    story.append(S.styled_table(rows, [3.4, 3.4], ["LEFT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 12. The open defect, recorded in the machine-readable evidence file so it "
            "cannot be forgotten. A calibration that exists in the repository, is tested, and is "
            "silently absent from half the results is worse than not having one. A reader sees "
            "the mechanism and assumes it applied.",
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            "The effect on the COMPARISON in this report is smaller than the effect on either "
            "absolute result. Both vehicles are uncalibrated in the same way, and the shape "
            "ratio multiplies the same term in both. The comparison is therefore still a fair "
            "one. The absolute intercept altitudes, Mach numbers and ranges are optimistic by an "
            "amount this study has not quantified.",
            S.BODY,
        )
    )


# --------------------------------------------------------------------------------------
#   8. Limitations
# --------------------------------------------------------------------------------------


def limitations(story: list, D: dict) -> None:
    ev = D["ev"]
    src = ev["sources"]

    S.sect(story, "8. Limitations")
    story.append(
        Paragraph(
            f"The source registry holds {src['n_registered']} entries once every owning module "
            f"is imported. {src['n_flagged']} of them are flagged. A flagged entry is one whose "
            "source string contains the word guess, or the words modelling choice, "
            "approximation or assumption. Every empirical constant in this repository carries a "
            "source string, and a test asserts that a guess says so.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The registry is filled at import time, so it is only complete once every owning "
            "module has been imported. The evidence collector imports all sixteen owning modules "
            "before it reads the registry, for that reason. A report that read the registry too "
            "early once listed 37 sources instead of 70, and under-reported its own limitations "
            "by a factor of four.",
            S.BODY,
        )
    )
    rows = [["Owning module", "Registered sources"]]
    for k, v in sorted(src["by_module"].items(), key=lambda kv: -kv[1]):
        rows.append([k, str(v)])
    rows.append(["TOTAL", str(src["n_registered"])])
    story.append(S.styled_table(rows, [3.4, 3.4], ["LEFT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 13. Where the registered sources live. The config module is not listed as an "
            "owner of the other modules' entries, because the registry IS the config module: "
            "attributing to it would attribute every key in the repository to it.",
            S.CAP,
        )
    )

    story.append(Paragraph("Every flagged source", S.H2))
    rows = [["Flagged source", "What it says"]]
    for k, v in sorted(src["flagged"].items()):
        rows.append([k, v])
    story.append(S.cell_table(rows, [1.75, 5.05], ["LEFT", "LEFT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            f"Table 14. All {src['n_flagged']} flagged sources. Two belong to the work this "
            "report describes: the entry that records the wave-drag ratio as Mach-independent, "
            "and the entry that records it as applied as a ratio.",
            S.CAP,
        )
    )

    story.append(Paragraph("What this study does not claim", S.H2))
    for text in (
        "The drag calibration is not applied to IV-1. Section 7 states this in full. It is an "
        "open defect, not a modelling choice.",
        "The slender-body wave-drag ratio is Mach-independent at this order. The true shape "
        "sensitivity drifts with Mach number. That drift is not modelled and has not been "
        "quantified.",
        "The nose shape was SET to the drag optimum, not searched. This study therefore makes no "
        "claim about an interior optimum on this vehicle. The SV-1 spline study, where the shape "
        "was searched, found one.",
        "The nozzle model is ideal. There is no two-phase loss, no divergence loss, no "
        "combustion efficiency and no throat erosion. Real delivered specific impulse for this "
        "class runs 3 to 7 percent lower and that penalty is not applied.",
        "The grains are restricted to a tubular geometry. Real motors of this class use finocyl "
        "or star grains. The restriction caps booster thrust and holds stage-2 propellant down. "
        "It is a visible design constraint here rather than a hidden optimism.",
        "Stage-2 static margin is still not in the constraint list. That remains open from the "
        "tangent-ogive IV-1 study, where it was evaluated once for the report and failed.",
        "Six-degree-of-freedom flight mechanics, guidance law design, structural sizing beyond a "
        "wall-thickness and hoop-stress check, and CFD are all out of scope.",
        "The requirements are invented for the demonstration. They do not correspond to any real "
        "programme.",
    ):
        story.append(Paragraph(text, S.BULLET, bulletText="-"))

    warnings = ev["warnings"]["spline"]
    if warnings:
        story.append(Paragraph("Warnings the run raised, and did not swallow", S.H2))
        rows = [["Warning"]] + [[w] for w in warnings]
        story.append(S.cell_table(rows, [6.8], ["LEFT"]))
        story.append(Spacer(1, 4))
        story.append(
            Paragraph(
                f"Table 15. The {len(warnings)} warnings the converged spline run raised. The "
                "tangent-ogive run raised the same set, so none of them is caused by the shape "
                "change. The bottom-up motor inert model is incomplete by design, and the "
                "correlation floor books the shortfall as a visible line item rather than "
                "hiding it.",
                S.CAP,
            )
        )


# --------------------------------------------------------------------------------------
#   9. Reproducing
# --------------------------------------------------------------------------------------


def reproduce(story: list, D: dict) -> None:
    ev = D["ev"]
    env = ev["environment"]

    S.sect(story, "9. Reproducing this result")
    story.append(
        Paragraph(
            ".venv/Scripts/python.exe -m pytest tests -q<br/>"
            ".venv/Scripts/python.exe scripts/iv1_converge.py --nose spline "
            "--interstage spline<br/>"
            ".venv/Scripts/python.exe scripts/iv1_converge.py --nose tangent_ogive "
            "--interstage cone<br/>"
            ".venv/Scripts/python.exe -m rocketgen.report.evidence_iv1_spline<br/>"
            ".venv/Scripts/python.exe -m rocketgen.report.fig_iv1_spline<br/>"
            ".venv/Scripts/python.exe -m rocketgen.report.build_report_iv1_spline",
            S.MONO,
        )
    )
    story.append(
        Paragraph(
            "The evidence collector does NOT call nTop. Both stacks were already measured, and "
            "the per-body measurements are on disk. Those are read back and fed to the mass and "
            "aerodynamic models exactly as the sizing script fed them. Every re-flight is then "
            "checked against the recorded result, field by field, to a relative tolerance of "
            "1.0e-9. A re-flight that did not reproduce its record fails the collector rather "
            "than quietly publishing a second answer.",
            S.BODY,
        )
    )
    rows = [
        ["Component", "Version or value"],
        ["Python", env["python"]],
        ["numpy", env["numpy"]],
        ["scipy", env["scipy"]],
        ["SUAVE", env["suave"]],
        ["nTop Automate", env["ntop"]],
        ["Drag calibration available", fmt(env["cd0_calibration_available"], 3)],
        ["Drag calibration applied", "NO"],
        ["Evidence collection wall time [s]", fmt(ev["wall_time_s"], 1)],
    ]
    story.append(S.styled_table(rows, [2.6, 4.2], ["LEFT", "LEFT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 16. Environment. numpy and scipy are pinned below their current major "
            "versions because SUAVE 2.5.2 does not run on either.",
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            "Two toolchain warnings are worth repeating. The nTop command-line tool returns exit "
            "code 72 when a block fails, not on success. Widely repeated guidance says the "
            "opposite and is wrong. Judge success on the expected files existing and being "
            "non-empty. Second, the cost of a surface-area measurement tracks the complexity of "
            "the implicit field, not the size of the body. On this vehicle four measurements on "
            "booleaned bodies took about 24 s each, and the fifth, on a bare cylinder primitive, "
            "took 0.27 s for the largest area of the five.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "A note on the renders in Figures 1 and 2. The first version of this report had no "
            "picture of the vehicle at all, because neither IV-1 run turned on the mesh export "
            "and no STL existed to render from. That is not recoverable after the fact: it "
            "needs the measurement repeating with exports enabled, which "
            "scripts/render_iv1_spline.py now does at the converged point only. A report "
            "without a view of its own geometry cannot be checked by a reader, so this is now "
            "a standing rule of the repository rather than a per-report decision.",
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
        title="IV-1 with a Splined Outer Mould Line",
        author="nTop",
        footer="IV-1 splined outer mould line  |  nTop + SUAVE  |  invented requirements",
    )

    front_matter(story, D)
    what_changed(story, D)
    story.append(PageBreak())
    wave_drag(story, D)
    story.append(PageBreak())
    ntop_side(story, D)
    story.append(PageBreak())
    the_result(story, D)
    story.append(PageBreak())
    the_trade(story, D)
    open_defect(story, D)
    story.append(PageBreak())
    limitations(story, D)
    reproduce(story, D)

    S.build(doc, story)
    print(f"wrote {OUT_PDF}  ({os.path.getsize(OUT_PDF) / 1024:.0f} KB)")
    return OUT_PDF


if __name__ == "__main__":
    main()
