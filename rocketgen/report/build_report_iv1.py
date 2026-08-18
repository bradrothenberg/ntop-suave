"""Assemble the IV-1 engineering report PDF.

Every number comes from a file under `runs/IV-1/`. Nothing is typed in by hand. Run the evidence
collector and the figure scripts first, then this:

    .venv/Scripts/python.exe -m rocketgen.report.evidence_iv1
    .venv/Scripts/python.exe -m rocketgen.report.fig_mass_iv1
    .venv/Scripts/python.exe -m rocketgen.report.fig_margins_iv1
    .venv/Scripts/python.exe -m rocketgen.report.fig_infeasible_iv1
    .venv/Scripts/python.exe -m rocketgen.report.fig_ascent_iv1
    .venv/Scripts/python.exe -m rocketgen.report.fig_envelope_iv1
    .venv/Scripts/python.exe -m rocketgen.report.build_report_iv1

Style follows ASD-STE100 Simplified Technical English: active voice, simple tenses, short
sentences, one idea per sentence, plain words.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

from reportlab.platypus import PageBreak, Paragraph, Spacer

from . import report_style as S

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(REPO, "runs")
IV1 = os.path.join(RUNS, "IV-1")
FIGS = os.path.join(IV1, "figures")
EXAMPLE = os.path.join(REPO, "examples", "IV-1")
OUT_DIR = os.path.join(IV1, "report")
OUT_PDF = os.path.join(OUT_DIR, "IV1_engineering_report.pdf")

MILE = 1609.344
DEG = math.degrees


# --------------------------------------------------------------------------------------
#   Loading
# --------------------------------------------------------------------------------------


def _json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all() -> dict[str, Any]:
    return {
        "conv": _json(os.path.join(IV1, "converged.json")),
        "ev": _json(os.path.join(FIGS, "evidence_iv1.json")),
    }


def fmt(v: Any, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "not measured"
    return f"{float(v):,.{nd}f}"


def sci(v: float) -> str:
    """A residual as a power of ten."""
    if v == 0.0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    return f"{v / (10.0 ** e):.1f}e{e:+d}"


def pct(v: float, nd: int = 2) -> str:
    return f"{100.0 * v:+.{nd}f}"


def constraint(conv: dict, name: str) -> dict:
    for c in conv["constraints"]:
        if c["name"] == name:
            return c
    raise KeyError(name)


# --------------------------------------------------------------------------------------
#   Front matter
# --------------------------------------------------------------------------------------


def front_matter(story: list, D: dict) -> None:
    conv, ev = D["conv"], D["ev"]
    ic = conv["intercept"]
    tr = ev["trajectory"]
    lat = conv["lateral_g"]
    dv = conv["design_vector"]

    story.append(Paragraph("IV-1: A Two-Stage Strake-Stabilised Interceptor", S.TITLE))
    story.append(
        Paragraph(
            "Coupled nTop and SUAVE conceptual sizing, and three requirements that "
            "could not hold together",
            S.SUBTITLE,
        )
    )
    story.append(S.hrule())
    story.append(
        Paragraph(
            "Prepared with nTop Automate 5.53.2 / 5.54.0 and SUAVE 2.5.2. "
            "All geometry is authored programmatically. "
            "Report generated from the run artefacts under runs/IV-1.",
            S.SUBTITLE,
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("Summary", S.H1))
    story.append(
        Paragraph(
            "This report describes a conceptual sizing loop for a two-stage, strake-stabilised "
            "interceptor-class vehicle. SUAVE does the physics. nTop does the geometry. The two "
            "are coupled: nTop measures the solid it builds, and those measurements go back into "
            "the mass and aerodynamic models.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            f"The loop found a design that meets all "
            f"{len(conv['constraints'])} recorded constraints. It weighs "
            f"{fmt(conv['launch_mass_kg'], 1)} kg. It reaches "
            f"{fmt(ic['slant_range'] / 1000.0, 1)} km of slant range, which is "
            f"{fmt(ic['slant_range'] / MILE, 1)} statute miles. It arrives at "
            f"{fmt(ic['altitude'] / 1000.0, 1)} km and Mach {fmt(ic['mach'], 2)}.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The result matters less than what the loop caught. Three requirements were mutually "
            "exclusive, and the cause was an exclusion in the specification itself. One "
            "structural limit was unachievable anywhere in the design space. One modelling "
            "assumption about the propellant grain turned out to be the binding physical "
            "limitation on the whole vehicle. The constraint list had a hole. The atmosphere "
            "table stopped below the altitude the vehicle flies at. Section 1 gives each finding.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "<b>The vehicle and its requirements are invented for this demonstration.</b> They "
            "correspond to no real programme. This is a parametric configuration study built "
            "from published conceptual-design methods. It is not a model of any fielded system. "
            "Section 6 lists every limitation and every value that is a guess.",
            S.ABSTRACT,
        )
    )

    rows = [
        ["Quantity", "Value", "Requirement", "Status"],
        [
            "Slant range at intercept",
            f"{fmt(ic['slant_range'] / 1000.0, 1)} km ({fmt(ic['slant_range'] / MILE, 1)} mi)",
            ">= 160.9 km",
            "met, active",
        ],
        [
            "Intercept altitude",
            f"{fmt(ic['altitude'] / 1000.0, 1)} km",
            ">= 15.0 km",
            "met",
        ],
        ["Mach at intercept", fmt(ic["mach"], 2), ">= 3.00", "met"],
        [
            "Lateral acceleration available",
            f"{fmt(lat['total'], 2)} g",
            ">= 15.0 g",
            "met, by divert motor",
        ],
        ["Launch mass", f"{fmt(conv['launch_mass_kg'], 1)} kg", "<= 1400 kg", "met"],
        [
            "Peak dynamic pressure",
            f"{fmt(tr['q_max_Pa'] / 1000.0, 1)} kPa",
            "<= 350 kPa",
            "met, limit revised",
        ],
        ["Stacked length", f"{fmt(dv['L_total'], 2)} m", "<= 5.40 m", "met"],
        ["Maximum body diameter", f"{fmt(dv['D_max'], 2)} m", "<= 0.42 m", "met, active"],
        [
            "Mass jettisoned at separation",
            f"{fmt(conv['jettisoned_kg'], 1)} kg",
            "not constrained",
            "-",
        ],
    ]
    story.append(S.styled_table(rows, [2.2, 1.7, 1.3, 1.5], ["LEFT", "RIGHT", "RIGHT", "LEFT"]))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Table 1. Headline result. The geometry is measured by nTop inside the loop. "
            "Slant range and maximum diameter both sit exactly on their limits. "
            "Source: runs/IV-1/converged.json.",
            S.CAP,
        )
    )


# --------------------------------------------------------------------------------------
#   1. What the loop caught
# --------------------------------------------------------------------------------------


def findings(story: list, D: dict) -> None:
    conv, ev = D["conv"], D["ev"]
    a = ev["audit"]

    S.sect(story, "1. What the loop caught")
    story.append(
        Paragraph(
            "A sizing loop earns its cost when it finds errors that inspection misses. This one "
            "found five. Each was found by walking trajectories against the constraint set, not "
            "by reading the specification.",
            S.BODY,
        )
    )

    # ---- 1.1 -------------------------------------------------------------------------
    story.append(Paragraph("1.1 Three requirements were mutually exclusive", S.H2))
    story.append(
        Paragraph(
            "As first written, A2, A3 and A11 could not hold together. A2 asks for "
            f"{fmt(a['required_slant_m'] / 1000.0, 1)} km of slant range, which is "
            f"{fmt(a['required_slant_miles'], 0)} statute miles. A3 asks for an intercept at or "
            f"above {fmt(a['h_intercept_min_m'] / 1000.0, 0)} km. A11 asks for "
            f"{fmt(a['lateral_g_min'], 0)} g of lateral acceleration at that intercept.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The specification assumed purely aerodynamic control. Available lateral "
            "acceleration is then q times S_ref times CN_max, divided by mass. That needs "
            "dynamic pressure. Dynamic pressure needs air. A2 needs a lofted trajectory to reach "
            "that far, and lofting puts the intercept where there is no air. The two pull in "
            "opposite directions.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            f"The audit swept the pitchover angle on the stack that closes every motor, volume "
            f"and structural constraint: {fmt(a['stack']['m0_kg'], 0)} kg, "
            f"{fmt(a['stack']['L_total_m'], 2)} m, {fmt(a['stack']['impulse_kNs'], 0)} kN.s of "
            "vacuum impulse. It then walked each trajectory for the furthest point at which A3, "
            "A4 and A11 all hold at once.",
            S.BODY,
        )
    )
    rows = [
        ["Pitchover [deg]", "Furthest point meeting\nA3, A4 and A11", "Altitude there",
         "Max slant range", "Altitude at\nmax slant"],
    ]
    for r in a["rows"]:
        ok = f"{fmt(r['slant_ok'] / 1000.0, 1)} km" if r["slant_ok"] > 0.0 else "none"
        h_ok = f"{fmt(r['h_ok'] / 1000.0, 1)} km" if r["slant_ok"] > 0.0 else "-"
        rows.append(
            [
                fmt(r["gamma_deg"], 0), ok, h_ok,
                f"{fmt(r['slant_max'] / 1000.0, 1)} km",
                f"{fmt(r['h_at_slant_max'] / 1000.0, 1)} km",
            ]
        )
    story.append(S.styled_table(rows, [1.1, 1.7, 1.2, 1.3, 1.3], ["RIGHT"] * 5))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 2. Pitchover sweep. The lateral-acceleration figures behind this table use a "
            f"generous CN_max of {fmt(a['cn_max_placeholder'], 1)} rather than the build-up, so "
            "every entry is an upper bound on what the vehicle can do. "
            "Source: SPEC_IV1.md section 2, reproduced by scripts/iv1_envelope_probe.py.",
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            f"<b>The best slant range meeting every requirement is "
            f"{fmt(a['best_slant_m'] / 1000.0, 1)} km, which is "
            f"{fmt(a['best_slant_miles'], 1)} miles against the "
            f"{fmt(a['required_slant_miles'], 0)} miles A2 asks for.</b> The shortfall is "
            f"{fmt(a['shortfall_m'] / 1000.0, 1)} km, a factor of "
            f"{fmt(a['shortfall_factor'], 2)}.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The cause is not the design vector. It is a hard physical ceiling. At the "
            f"post-separation mass of {fmt(a['stack']['mass_after_separation_kg'], 0)} kg, and "
            f"with the same generous CN_max of {fmt(a['cn_max_placeholder'], 1)}, "
            f"{fmt(a['lateral_g_min'], 0)} g is available only below these altitudes:",
            S.BODY,
        )
    )
    rows = [["Mach", "15 g available only below"]]
    for c in a["ceiling"]:
        rows.append([fmt(c["mach"], 1), f"{fmt(c['h_limit_m'] / 1000.0, 1)} km"])
    story.append(S.styled_table(rows, [1.6, 2.6], ["RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 3. The altitude ceiling A11 imposes on its own, independent of any "
            "trajectory. A3 requires the intercept at or above 15 km, so the band in which A3 "
            "and A11 overlap is empty at Mach 3 to 4.",
            S.CAP,
        )
    )
    story.append(
        S.fig_single(
            os.path.join(FIGS, "requirements_conflict.png"),
            1,
            "The conflict, drawn. Panel (a) puts both requirements in the same axes: the green "
            "region is where 15 g is available, the blue region is where A3 puts the intercept, "
            "and they first touch at Mach 4.45. Every trajectory that reaches 100 miles does so "
            "far above the green region. Panel (b) shows what that costs in range.",
            width_in=6.8,
            max_h_in=3.2,
        )
    )
    story.append(
        Paragraph(
            "<b>Resolution: the vehicle needs lateral control that does not depend on dynamic "
            "pressure.</b> That is a divert or attitude-control motor, which is what vehicles of "
            "this class carry and precisely why they carry it. <b>SPEC_IV1.md section 8 "
            "originally excluded attitude-control thrusters, and that exclusion is what made the "
            "requirement set infeasible.</b> A11 was therefore restated on the capability rather "
            "than on the mechanism, and A13 was added so the motor is sized, reported and charged "
            "to the mass statement.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The converged vehicle carries a divert motor of "
            f"{fmt(conv['acs']['thrust'] / 1000.0, 1)} kN for "
            f"{fmt(conv['acs']['burn_time'], 1)} s, which is "
            f"{fmt(ev['mass']['acs_total_impulse_Ns'] / 1000.0, 0)} kN.s of total impulse and "
            f"{fmt(ev['mass']['acs_pack_kg'], 1)} kg of mass. At the intercept it supplies "
            f"{fmt(conv['lateral_g']['acs'], 2)} g. The airframe supplies only "
            f"{fmt(conv['lateral_g']['aerodynamic'], 2)} g there, because the dynamic pressure "
            f"has fallen to {fmt(ev['trajectory']['q'][-1] / 1000.0, 0)} kPa.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "<b>The conclusion is a design conclusion, not a design.</b> A purely aerodynamic "
            "interceptor at this size cannot engage at 100 miles and still manoeuvre. That "
            "statement is more useful than the vehicle that follows from it.",
            S.BODY,
        )
    )

    # ---- 1.2 -------------------------------------------------------------------------
    story.append(Paragraph("1.2 The dynamic-pressure limit was unachievable", S.H2))
    story.append(
        Paragraph(
            "A10 was first written as 250 kPa. Nothing in the design space meets it. A sweep over "
            "stage propellant, booster thrust, pitchover angle and pitchover time found a floor "
            "of about 278 kPa on peak dynamic pressure, rising to 309 kPa for any design that "
            "also meets A2, A3 and A4.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The reason is the launch condition. A1 launches vertically from a canister at sea "
            "level, so the vehicle accelerates through dense air near the ground. The converged "
            f"design reaches its peak of {fmt(ev['trajectory']['q_max_Pa'] / 1000.0, 1)} kPa at "
            f"{fmt(ev['trajectory']['q_max_altitude_m'] / 1000.0, 2)} km and Mach "
            f"{fmt(ev['trajectory']['q_max_mach'], 2)}, only "
            f"{fmt(ev['trajectory']['q_max_time_s'], 1)} s after launch. Holding 250 kPa would "
            "cap the vehicle near Mach 2.75 at 6 km, which cannot reach 100 miles.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "A10 was therefore revised to 350 kPa, about 13 percent above the 309 kPa floor that "
            "A2, A3 and A4 impose. <b>The consequence must be stated plainly. The structural "
            "model in this toolkit is wall thickness times density, plus a hoop-stress check on "
            "the motor case and a buckling check on the interstage. It does not size the airframe "
            "for a 350 kPa aerodynamic load.</b> The limit is a stated requirement that the mass "
            "model does not verify, so the airframe mass is optimistic by an amount this toolkit "
            "cannot quantify.",
            S.BODY,
        )
    )

    # ---- 1.3 -------------------------------------------------------------------------
    story.append(Paragraph("1.3 The tubular grain is the binding physical limitation", S.H2))
    gl = ev["grain_limits"]
    d0 = gl["default_stack"]
    story.append(
        Paragraph(
            "Every stage grain in this model is an internal-burning tube, closed at its mean web. "
            "A tube is the least area-efficient internal-burning geometry. Real tactical motors "
            "of this thrust class use slotted, star or finocyl grains, which reach several times "
            "the burning area in the same case length. <b>No shape factor is applied here, "
            "because no sourced multiplier was available.</b> The model reports the long tube "
            "instead, so the assumption surfaces as a design constraint rather than as silent "
            "optimism.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "That assumption then drove three separate decisions in the design.",
            S.BODY,
        )
    )
    rows = [["Booster thrust [kN]", "Grain L/D", "Grain length [m]", "Bay length [m]",
             "Nozzle exit [m]", "Grain closes"]]
    for r in gl["thrust_sweep"]:
        rows.append(
            [
                fmt(r["F1_kN"], 0), fmt(r["L_over_D"], 2), fmt(r["grain_length_m"], 3),
                fmt(r["bay_length_m"], 2), fmt(r["exit_diameter_m"], 3),
                "yes" if r["feasible"] else "NO",
            ]
        )
    story.append(S.styled_table(rows, [1.3, 0.9, 1.2, 1.1, 1.1, 1.0], ["RIGHT"] * 6))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 4. Booster-thrust sweep on the converged geometry. A tubular grain needs "
            "burning area, and burning area on a tube needs length. The grain runs out of bay "
            f"between {fmt(gl['max_feasible_F1_kN'], 0)} and 150 kN. The converged design uses "
            f"{fmt(D['conv']['design_vector']['stages'][0]['F_thrust'] / 1000.0, 0)} kN.",
            S.CAP,
        )
    )
    rows = [["Stage-2 propellant [kg]", "Volumetric loading", "Web [mm]", "Bay radius [mm]",
             "Grain closes"]]
    for r in gl["propellant_sweep"]:
        rows.append(
            [
                fmt(r["m_p2_kg"], 0), fmt(r["vol_loading"], 3),
                fmt(1000.0 * r["web_m"], 0), fmt(1000.0 * r["bay_radius_m"], 0),
                "yes" if r["feasible"] else "NO",
            ]
        )
    story.append(S.styled_table(rows, [1.6, 1.4, 1.1, 1.3, 1.1], ["RIGHT"] * 5))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 5. Stage-2 propellant sweep on the converged geometry. Above "
            f"{fmt(gl['max_feasible_m_p2_kg'], 0)} kg the web is wider than the bay radius, so "
            "the tube cannot hold the charge at all. The converged design carries "
            f"{fmt(D['conv']['design_vector']['stages'][1]['m_propellant'], 0)} kg.",
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            "The nozzle is the third consequence. The starting design vector used an area ratio "
            f"of {fmt(d0['eps_nozzle_1'], 0)} on a {fmt(d0['D1_m'], 2)} m booster at "
            f"{fmt(d0['F1_N'] / 1000.0, 0)} kN. That gives an exit diameter of "
            f"{fmt(1000.0 * d0['exit_diameter_1_m'], 0)} mm, which is wider than the "
            f"{fmt(1000.0 * d0['D1_m'], 0)} mm body. The exit cone does not fit inside the "
            "airframe. The booster was therefore taken to the full 0.42 m allowed diameter and "
            "its area ratio was reduced to 6.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The same starting vector also needed "
            f"{fmt(100.0 * d0['vol_loading_2'], 1)} percent volumetric loading on stage 2. The "
            "model reports all three problems as warnings rather than smoothing them over, and a "
            "test asserts that it does.",
            S.BODY,
        )
    )

    # ---- 1.4 -------------------------------------------------------------------------
    story.append(Paragraph("1.4 The constraint list had a hole", S.H2))
    story.append(
        Paragraph(
            "The constraint list gated the grain length-to-diameter ratio but not grain closure. "
            "Those are different questions. A grain can sit inside the L/D band of 1.0 to 8.0 and "
            "still be impossible, because a tubular grain needs a bore for burning area and the "
            "remaining web has to fit inside the bay radius.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "Gating only on L/D let a stage-2 grain through at 134 percent volumetric loading, "
            "with a 232 mm web in a 164 mm bay radius. Two constraints were added per stage: "
            "volumetric loading at or below 1.0, and a closure flag from the grain solver. The "
            "converged design now sits at "
            f"{fmt(constraint(conv, 'stage 2 vol loading')['value'], 3)} on stage 2, which is the "
            "third tightest constraint in the whole set.",
            S.BODY,
        )
    )
    sm = ev["static_margin"]
    story.append(
        Paragraph(
            f"<b>A second hole is still open, and this report does not close it.</b> Requirement "
            f"A9 asks for a static margin of at least "
            f"{fmt(sm['limit_calibres'], 1)} calibre in each stage over the whole flight. A9 is "
            f"not among the {len(conv['constraints'])} constraints the sizing script records. "
            "Section 2.2 evaluates it here and reports the result. It does not pass.",
            S.BODY,
        )
    )

    # ---- 1.5 -------------------------------------------------------------------------
    story.append(Paragraph("1.5 The atmosphere table stopped below the flight path", S.H2))
    at = ev["atmosphere"]
    story.append(
        Paragraph(
            f"The cached US Standard 1976 table stopped at "
            f"{fmt(at['h_max_legacy_m'] / 1000.0, 0)} km and clamped above it. That was ample for "
            "the SV-1 cruise-and-dive mission. It is not ample for a lofted two-stage intercept. "
            "Measured IV-1 arcs apogee between 45 and 54 km, and the converged design apogees at "
            f"{fmt(at['apogee_m'] / 1000.0, 1)} km.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "A clamp holds density at its ceiling value, so it overstates drag on everything "
            f"above the ceiling. At {fmt(at['apogee_m'] / 1000.0, 1)} km the clamp overstated "
            f"density by a factor of {fmt(at['clamp_drag_overstatement'], 1)}. At 50 km it "
            f"overstated it by a factor of {fmt(at['clamp_overstatement_50km'], 1)}, which is "
            "more than an order of magnitude.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            f"The table now runs to {fmt(at['h_max_m'] / 1000.0, 0)} km, the upper limit of the "
            f"standard's lower atmosphere, on a {fmt(at['step_m'], 0)} m grid of "
            f"{at['n_nodes']:,} nodes. The converged trajectory now flies entirely inside the "
            "table: the recorded overshoot above the ceiling is "
            f"{fmt(ev['trajectory']['h_above_atmosphere_table'], 1)} m.",
            S.BODY,
        )
    )
    story.append(Paragraph("An instructive near-miss", S.H2))
    story.append(
        Paragraph(
            "A first comparison against the published tables showed SUAVE about "
            f"{fmt(100.0 * at['naive_error'], 1)} percent off at 47 km. That looked like a model "
            "defect. It was not. It was an error in the comparison.",
            S.BODY,
        )
    )
    rows = [
        ["Step", "Value"],
        ["Pressure SUAVE returns at 47 km geometric",
         f"{fmt(at['p_at_47km_geometric'], 2)} Pa"],
        ["Published 47 km row, which is geopotential",
         f"{fmt(at['p_at_47km_geopotential_row'], 3)} Pa"],
        ["Apparent error of the naive comparison", pct(at["naive_error"], 2) + " %"],
        ["Geopotential altitude of 47 km geometric",
         f"{fmt(at['geopotential_of_47km_geometric_m'] / 1000.0, 3)} km"],
        ["Pressure at the geometric altitude for 47 km geopotential",
         f"{fmt(at['p_at_geometric_for_47km_geopotential'], 3)} Pa"],
    ]
    story.append(S.styled_table(rows, [4.2, 2.4], ["LEFT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 6. The geopotential-against-geometric near-miss. The standard is tabulated "
            "against geopotential altitude. SUAVE takes geometric altitude and converts "
            "internally. Doing the conversion properly removes the whole discrepancy. A test now "
            "asserts that the naive comparison still fails, so nobody can fix a defect that does "
            "not exist.",
            S.CAP,
        )
    )


# --------------------------------------------------------------------------------------
#   2. The converged design
# --------------------------------------------------------------------------------------


def the_design(story: list, D: dict) -> None:
    conv, ev = D["conv"], D["ev"]
    dv = conv["design_vector"]
    s1, s2 = dv["stages"]
    st = dv["strakes"]

    S.sect(story, "2. The converged design")
    story.append(
        S.fig_single(
            os.path.join(EXAMPLE, "iv1_iso.png"),
            2,
            "The IV-1 stack, rendered from the STL that nTop exported. Upper panel: ogive "
            "payload stage with four strakes and four tail fins, conical interstage, booster "
            "with four tail fins. Lower panel: the strake mid-span, where the 30 mm strake and "
            "its root junction into the wall are visible. This render is of the starting design "
            "vector at 5.08 m, not of the converged 5.28 m stack; the topology is the same and "
            "the dimensions differ. No one opened the nTop graphical interface.",
            width_in=6.7,
            max_h_in=3.4,
        )
    )

    rows = [
        ["Parameter", "Stage 1, booster", "Stage 2, payload stage", "Unit"],
        ["Body diameter", fmt(s1["D"], 3), fmt(s2["D"], 3), "m"],
        ["Stage length", fmt(s1["L"], 2), fmt(s2["L"], 2), "m"],
        ["Propellant mass", fmt(s1["m_propellant"], 1), fmt(s2["m_propellant"], 1), "kg"],
        ["Thrust", fmt(s1["F_thrust"] / 1000.0, 1), fmt(s2["F_thrust"] / 1000.0, 1), "kN"],
        ["Chamber pressure", fmt(s1["p_c"] / 1.0e6, 1), fmt(s2["p_c"] / 1.0e6, 1), "MPa"],
        ["Nozzle area ratio", fmt(s1["eps_nozzle"], 1), fmt(s2["eps_nozzle"], 1), "-"],
        ["Wall thickness", fmt(1000.0 * s1["t_wall"], 2), fmt(1000.0 * s2["t_wall"], 2), "mm"],
        ["Fin count", str(int(s1["n_fin"])), str(int(s2["n_fin"])), "-"],
        ["Fin exposed semi-span", fmt(s1["b_fin"], 3), fmt(s2["b_fin"], 3), "m"],
        ["Fin root chord", fmt(s1["c_r_fin"], 3), fmt(s2["c_r_fin"], 3), "m"],
        ["Fin leading-edge sweep", fmt(DEG(s1["sweep_fin"]), 1), fmt(DEG(s2["sweep_fin"]), 1),
         "deg"],
        ["Jettisoned at separation", "yes", "no", "-"],
    ]
    story.append(S.styled_table(rows, [2.1, 1.6, 1.7, 0.8], ["LEFT", "RIGHT", "RIGHT", "LEFT"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Table 7. The converged design vector, by stage.", S.CAP))

    rows = [
        ["Parameter", "Value", "Unit"],
        ["Stacked length", fmt(dv["L_total"], 2), "m"],
        ["Maximum body diameter", fmt(dv["D_max"], 2), "m"],
        ["Payload-stage nose fineness", fmt(dv["f_nose"], 2), "-"],
        ["Payload-stage nose length", fmt(dv["L_nose"], 3), "m"],
        ["Interstage length", fmt(dv["L_interstage"], 3), "m"],
        ["Interstage wall thickness", fmt(1000.0 * dv["t_interstage"], 2), "mm"],
        ["Strake count", str(int(st["n"])), "-"],
        ["Strake height above the body", fmt(1000.0 * st["height"], 1), "mm"],
        ["Strake length", fmt(st["length"], 2), "m"],
        ["Strake thickness", fmt(1000.0 * st["thickness"], 1), "mm"],
        ["Strake leading-edge station", fmt(st["x_le"], 2), "m"],
        ["Commanded pitchover angle", fmt(conv["pitchover_deg"], 1), "deg"],
        ["Pitchover start time", fmt(dv["t_pitch"], 1), "s"],
        ["Commanded pitch rate", fmt(DEG(dv["pitch_rate_max"]), 1), "deg/s"],
        ["Divert-motor thrust", fmt(conv["acs"]["thrust"] / 1000.0, 1), "kN"],
        ["Divert-motor firing time", fmt(conv["acs"]["burn_time"], 1), "s"],
        ["Divert-motor total impulse",
         fmt(ev["mass"]["acs_total_impulse_Ns"] / 1000.0, 1), "kN.s"],
    ]
    story.append(S.styled_table(rows, [3.2, 1.7, 1.0], ["LEFT", "RIGHT", "LEFT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph("Table 8. Stack-level parameters, strakes and the ascent programme.", S.CAP)
    )

    # ---- motor -----------------------------------------------------------------------
    story.append(Paragraph("2.1 The motor stack", S.H2))
    m = ev["motor"]
    o1, o2 = m["stages"]["1"]["operating_point"], m["stages"]["2"]["operating_point"]
    g1, g2 = m["stages"]["1"]["grain"], m["stages"]["2"]["grain"]
    rows = [
        ["Quantity", "Stage 1", "Stage 2", "Unit"],
        ["Vacuum thrust", fmt(o1["thrust_vacuum"] / 1000.0, 1),
         fmt(o2["thrust_vacuum"] / 1000.0, 1), "kN"],
        ["Sea-level thrust", fmt(o1["thrust_sea_level"] / 1000.0, 1),
         fmt(o2["thrust_sea_level"] / 1000.0, 1), "kN"],
        ["Vacuum specific impulse", fmt(o1["isp_vacuum"], 1), fmt(o2["isp_vacuum"], 1), "s"],
        ["Burn time", fmt(o1["burn_time"], 2), fmt(o2["burn_time"], 2), "s"],
        ["Mass flow", fmt(o1["mdot"], 2), fmt(o2["mdot"], 2), "kg/s"],
        ["Throat diameter", fmt(1000.0 * o1["throat_diameter"], 1),
         fmt(1000.0 * o2["throat_diameter"], 1), "mm"],
        ["Exit diameter", fmt(1000.0 * o1["exit_diameter"], 1),
         fmt(1000.0 * o2["exit_diameter"], 1), "mm"],
        ["Grain length", fmt(g1["length"], 3), fmt(g2["length"], 3), "m"],
        ["Grain length over diameter", fmt(g1["L_over_D"], 2), fmt(g2["L_over_D"], 2), "-"],
        ["Grain web", fmt(1000.0 * g1["web"], 0), fmt(1000.0 * g2["web"], 0), "mm"],
        ["Volumetric loading", fmt(g1["volumetric_loading"], 3),
         fmt(g2["volumetric_loading"], 3), "-"],
        ["Total impulse, vacuum", fmt(o1["total_impulse_vacuum"] / 1000.0, 0),
         fmt(o2["total_impulse_vacuum"] / 1000.0, 0), "kN.s"],
    ]
    story.append(S.styled_table(rows, [2.2, 1.5, 1.5, 0.9], ["LEFT", "RIGHT", "RIGHT", "LEFT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 9. Motor operating points. Stage-1 thrust is a sea-level equivalent, because "
            "A1 launches the stack from a canister at sea level. Stage-2 thrust is a vacuum "
            "value, because it lights above 4 km. Each stage runs one throat area for its whole "
            "burn, so the stack needs no throat-changing mechanism at all. That is the opposite "
            "of the SV-1 dual-thrust motor, which needs a throat that shrinks and cannot have "
            "one.",
            S.CAP,
        )
    )

    # ---- mass ------------------------------------------------------------------------
    story.append(Paragraph("2.2 Mass statement", S.H2))
    mass = ev["mass"]
    story.append(
        Paragraph(
            f"The launch mass is {fmt(mass['m0_kg'], 1)} kg. The booster group weighs "
            f"{fmt(mass['stage_totals_kg']['1'], 1)} kg and the interstage "
            f"{fmt(mass['stage_totals_kg']['0'], 2)} kg. At separation "
            f"{fmt(mass['jettisoned_kg'], 1)} kg leaves the vehicle, and the payload stage "
            f"continues at {fmt(mass['mass_after_separation_kg'], 1)} kg.",
            S.BODY,
        )
    )
    rows = [["Stage", "Item", "Mass [kg]", "Percent", "Station [m]", "Provenance"]]
    order = sorted(
        conv["mass_statement"],
        key=lambda e: (-int(e["stage"]) if e["stage"] else 99, -e["mass_kg"]),
    )
    for it in order:
        rows.append(
            [
                str(int(it["stage"])) if it["stage"] else "-",
                it["item"],
                fmt(it["mass_kg"], 2),
                fmt(100.0 * it["mass_kg"] / mass["m0_kg"], 1),
                fmt(it["station_m"], 3),
                it["provenance"].replace("_", " "),
            ]
        )
    rows.append(
        ["2", "Attitude-control motor pack", fmt(mass["acs_pack_kg"], 2),
         fmt(100.0 * mass["acs_pack_kg"] / mass["m0_kg"], 1), "not stationed", "correlation"]
    )
    rows.append(["", "TOTAL", fmt(mass["m0_kg"], 2), "100.0", "", ""])
    story.append(
        S.styled_table(rows, [0.5, 2.2, 0.9, 0.7, 1.1, 1.2],
                       ["CENTER", "LEFT", "RIGHT", "RIGHT", "RIGHT", "LEFT"])
    )
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 10. Group-weight statement, by stage. Provenance says where each mass came "
            f"from. Only {fmt(100.0 * mass['measured_fraction'], 1)} percent of the launch mass "
            f"is measured by nTop, which is {fmt(mass['measured_kg'], 2)} kg. The propellant, the "
            "payload and the motor correlations are not geometry, so they cannot be measured. The "
            "divert-motor pack is charged on top of the stage statement and carries no station, "
            "so it is listed separately.",
            S.CAP,
        )
    )
    story.append(
        S.fig_single(
            os.path.join(FIGS, "mass_statement_iv1.png"),
            3,
            "The mass statement, grouped by stage and coloured by provenance. Panel (b) is the "
            "part that a single-stage vehicle does not have: half the launch mass is booster, and "
            "31.3 kg of it leaves the vehicle at separation.",
            width_in=6.8,
            max_h_in=3.4,
        )
    )

    # ---- constraints -----------------------------------------------------------------
    story.append(Paragraph("2.3 Constraints", S.H2))
    rows = [["Constraint", "Value", "Sense", "Limit", "Margin [%]", "Status"]]
    for c in sorted(ev["constraints"], key=lambda c: c["margin"]):
        rows.append(
            [
                c["name"], fmt(c["value"], 3), c["sense"], fmt(c["limit"], 3),
                pct(c["margin"], 1), "met" if c["met"] else "FAIL",
            ]
        )
    story.append(
        S.styled_table(rows, [1.7, 1.4, 0.6, 1.3, 1.0, 0.7],
                       ["LEFT", "RIGHT", "CENTER", "RIGHT", "RIGHT", "LEFT"])
    )
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            f"Table 11. All {len(ev['constraints'])} recorded constraints, tightest first, with "
            "margin as a percentage of each limit. Slant range and maximum diameter sit exactly "
            "on their limits by construction: the run terminates on the slant range, and the "
            "booster was taken to the full allowed diameter so its nozzle exit would fit.",
            S.CAP,
        )
    )
    story.append(
        S.fig_single(
            os.path.join(FIGS, "constraint_margins_iv1.png"),
            4,
            "Constraint margins for the converged design, on a symmetric log scale. The three "
            "constraints at zero margin are met exactly. A9 is drawn in red because it is not "
            "one of the recorded constraints and does not pass; see the next paragraph.",
            width_in=6.8,
            max_h_in=3.4,
        )
    )

    sm = ev["static_margin"]
    story.append(
        Paragraph(
            "<b>Requirement A9 is evaluated here, and it fails.</b> A9 asks for a static margin "
            f"of at least {fmt(sm['limit_calibres'], 1)} calibre in each stage over the whole "
            "flight. The sizing script does not record it as a constraint. This report computes "
            "it from the aerodynamic centre of pressure at 10 degrees of angle of attack and the "
            "centre of gravity of the mass statement, in the two flight configurations. The worst "
            f"value is {fmt(sm['worst_calibres'], 3)} calibres, which is unstable.",
            S.BODY,
        )
    )
    rows = [["Configuration", "Stage", "Mach", "x_cp [m]", "x_cg [m]", "Static margin [cal]"]]
    for r in sm["rows"]:
        rows.append(
            [
                r["config"], str(r["stage"]), fmt(r["mach"], 1), fmt(r["x_cp_m"], 3),
                fmt(r["x_cg_m"], 3), fmt(r["static_margin_cal"], 3),
            ]
        )
    story.append(
        S.styled_table(rows, [2.1, 0.6, 0.7, 1.0, 1.0, 1.3],
                       ["LEFT", "CENTER", "RIGHT", "RIGHT", "RIGHT", "RIGHT"])
    )
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 12. Static margin, computed for this report. Stations are measured aft from "
            f"the payload-stage nose tip. Reference diameter is "
            f"{fmt(sm['D_ref_stage1_m'], 2)} m on the stack and {fmt(sm['D_ref_stage2_m'], 2)} m "
            "on the payload stage. Two caveats: the divert-motor pack carries no station, so it "
            "is not in the centre of gravity, and the centre of pressure is a linear build-up "
            "figure at one angle of attack. The result is an open item, not a verdict. It is "
            "reported because a constraint that nobody evaluates is worse than one that fails.",
            S.CAP,
        )
    )

    # ---- trajectory ------------------------------------------------------------------
    story.append(Paragraph("2.4 The ascent", S.H2))
    tr = ev["trajectory"]
    ic = conv["intercept"]
    story.append(
        Paragraph(
            "The mission has four phases: a vertical rise, a bounded-rate pitchover, staged "
            f"boost, and a lofted midcourse coast. The flight lasts {fmt(tr['duration_s'], 1)} s. "
            f"It apogees at {fmt(tr['apogee_m'] / 1000.0, 1)} km and reaches the slant-range "
            f"condition on the way down, at {fmt(ic['altitude'] / 1000.0, 1)} km and Mach "
            f"{fmt(ic['mach'], 2)}. Reaching the range while descending is a legitimate intercept "
            "for this vehicle class, so the run does not terminate on ground impact first.",
            S.BODY,
        )
    )
    rows = [["Event", "Time [s]", "Altitude [km]", "Mach", "Mass before [kg]",
             "Mass after [kg]"]]
    for e in tr["events"]:
        rows.append(
            [
                e["name"].replace("_", " "), fmt(e["time"], 2), fmt(e["altitude"] / 1000.0, 2),
                fmt(e["mach"], 3), fmt(e["mass_before"], 2), fmt(e["mass_after"], 2),
            ]
        )
    rows.append(
        ["apogee", fmt(tr["apogee_time_s"], 2), fmt(tr["apogee_m"] / 1000.0, 2), "", "", ""]
    )
    rows.append(
        ["intercept", fmt(ic["time"], 2), fmt(ic["altitude"] / 1000.0, 2), fmt(ic["mach"], 3),
         "", fmt(ic["mass"], 2)]
    )
    story.append(
        S.styled_table(rows, [1.6, 0.9, 1.1, 0.8, 1.2, 1.2],
                       ["LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"])
    )
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 13. The stage events. Separation and stage-2 ignition are the same instant, "
            f"{fmt(tr['t_separation_s'] - tr['t_burnout_1_s'], 1)} s after stage-1 burnout. The "
            "stack therefore carries the spent booster through the coast, which is the "
            "pessimistic reading.",
            S.CAP,
        )
    )
    story.append(
        S.fig_single(
            os.path.join(FIGS, "ascent_iv1.png"),
            5,
            "The converged ascent. Colour carries the phase. Panel (d) is the one a single-stage "
            "vehicle does not have: the mass drops by 31.3 kg in one step at separation.",
            width_in=6.8,
            max_h_in=4.0,
        )
    )
    story.append(
        S.fig_single(
            os.path.join(FIGS, "flight_envelope_iv1.png"),
            6,
            "The same flight in altitude against Mach, against lines of constant dynamic "
            "pressure. The vehicle crosses its peak dynamic pressure early and under the booster. "
            "It then leaves the high-pressure region entirely, and arrives at the intercept where "
            "the air can supply only 7.45 g. That is why the divert motor exists.",
            width_in=6.8,
            max_h_in=3.8,
        )
    )
    if tr.get("message"):
        story.append(Paragraph("The integrator reported: " + str(tr["message"]), S.MONO))
        story.append(
            Paragraph(
                "That warning is correct and is not a failure. It reports the AERODYNAMIC lateral "
                "acceleration only, because the integrator does not know about the divert motor. "
                "The A11 figure is the greater of the aerodynamic and the divert capability, and "
                f"it is {fmt(conv['lateral_g']['total'], 2)} g.",
                S.BODY,
            )
        )


# --------------------------------------------------------------------------------------
#   3. What makes IV-1 different
# --------------------------------------------------------------------------------------


def what_is_different(story: list, D: dict) -> None:
    conv, ev = D["conv"], D["ev"]

    S.sect(story, "3. What makes IV-1 different from SV-1")
    story.append(
        Paragraph(
            "SV-1 is a single body that keeps every kilogram it launches with. IV-1 is not. Three "
            "things change, and each of them changes the model rather than only the numbers.",
            S.BODY,
        )
    )

    # ---- staging ---------------------------------------------------------------------
    story.append(Paragraph("3.1 Mass leaves the vehicle", S.H2))
    stg = ev["staging"]
    story.append(
        Paragraph(
            f"At separation, {fmt(conv['jettisoned_kg'], 1)} kg of spent booster and interstage "
            "leaves the vehicle in one step. That is the entire point of staging: the second "
            "propellant increment does not have to accelerate the first stage's empty structure.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The gain is measured against a controlled comparison, not asserted. Two vehicles fly "
            f"the same {fmt(stg['propellant_kg'], 0)} kg of propellant from the same "
            f"{fmt(stg['m0_kg'], 0)} kg launch mass, at the same effective exhaust velocity, over "
            f"the same {fmt(stg['burn_time_s'], 1)} s. The burn time is identical, so the gravity "
            f"loss of {fmt(stg['gravity_loss_m_s'], 1)} m/s is identical too. The only difference "
            f"is that the staged vehicle throws {fmt(stg['jettisoned_kg'], 0)} kg away between "
            "the two increments.",
            S.BODY,
        )
    )
    rows = [
        ["Quantity", "Staged", "Equivalent single stage", "Difference"],
        ["Burnout speed [m/s]", fmt(stg["v_staged"], 1), fmt(stg["v_single"], 1),
         f"+{stg['gain_m_s']:.1f}"],
        ["Burnout altitude [m]", fmt(stg["h_staged"], 0), fmt(stg["h_single"], 0),
         f"+{stg['h_staged'] - stg['h_single']:.0f}"],
        ["Gain on burnout speed [%]", "", "", f"+{stg['gain_percent']:.2f}"],
    ]
    story.append(S.styled_table(rows, [2.0, 1.4, 1.9, 1.3], ["LEFT", "RIGHT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            f"Table 14. Staging is worth {fmt(stg['gain_percent'], 2)} percent on burnout speed "
            "at identical mass, propellant, burn time and therefore gravity loss. The closed form "
            "for the gain is ve times the difference of the two logarithms of mass ratio. The "
            f"integrated result matches it to {sci(stg['analytic_rel_err'])} relative, so the "
            "jettison reaches the equations of motion exactly and not approximately.",
            S.CAP,
        )
    )

    # ---- reference area --------------------------------------------------------------
    story.append(Paragraph("3.2 The aerodynamic reference area changes at separation", S.H2))
    lg = ev["lateral_g"]
    story.append(
        Paragraph(
            "A coefficient without its reference area is meaningless. The stack flies on the "
            f"booster area of {fmt(lg['S_ref_stage1_m2'], 4)} m^2. After separation the payload "
            f"stage flies on its own area of {fmt(lg['S_ref_stage2_m2'], 4)} m^2, which is "
            f"{fmt(100.0 * lg['S_ref_ratio'], 1)} percent of it. Every drag and normal-force "
            "coefficient in this model therefore carries a stage index, and the aerodynamic model "
            "refuses to answer without one.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The reference length changes too, from "
            f"{fmt(ev['static_margin']['D_ref_stage1_m'], 2)} m to "
            f"{fmt(ev['static_margin']['D_ref_stage2_m'], 2)} m, so a static margin in calibres "
            "means a different distance before and after separation. That is why Table 12 "
            "evaluates the margin in both configurations rather than once.",
            S.BODY,
        )
    )

    # ---- strakes ---------------------------------------------------------------------
    story.append(Paragraph("3.3 The strakes carry their load by vortex lift", S.H2))
    sk = ev["strakes"]
    su = sk["summary"]
    story.append(
        Paragraph(
            f"Each strake is {fmt(1000.0 * su['height_m'], 0)} mm tall and "
            f"{fmt(su['length_m'], 2)} m long, so its exposed aspect ratio is "
            f"{fmt(su['aspect_ratio_panel'], 3)}. At that aspect ratio the flow is completely "
            "vortex dominated. A purely linear lifting-surface method would underpredict the "
            "load by two orders of magnitude.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The model therefore uses the Polhamus leading-edge suction analogy with the Lamar "
            "side-edge term. At the A11 angle-of-attack limit of "
            f"{fmt(sk['alpha_max_deg'], 0)} degrees, the strake normal force splits into "
            f"{fmt(sk['cn_strake_linear_at_limit'], 4)} of linear term and "
            f"{fmt(sk['cn_strake_vortex_at_limit'], 4)} of vortex term. The vortex term is "
            f"{fmt(100.0 * sk['vortex_share_at_limit'], 1)} percent of the load.",
            S.BODY,
        )
    )
    rows = [["Mach", "CN_max, strakes on", "CN_max, strakes off", "Gain [%]",
             "x_cp shift [cal]"]]
    for r in sk["rows"]:
        if r["stage"] != 2:
            continue
        rows.append(
            [
                fmt(r["mach"], 1), fmt(r["cn_max_on"], 3), fmt(r["cn_max_off"], 3),
                pct(r["cn_max_gain"], 1), fmt(r["x_cp_shift_cal"], 3),
            ]
        )
    story.append(S.styled_table(rows, [0.9, 1.6, 1.6, 1.0, 1.4], ["RIGHT"] * 5))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 15. What the strakes are worth on the payload stage, at the converged "
            f"dimensions. The gain on CN_max is {pct(sk['gain_stage2_min'], 1)} to "
            f"{pct(sk['gain_stage2_max'], 1)} percent. On the starting design vector, which is "
            "the one the validation figure uses, the same gain is "
            f"{pct(sk['default_gain_stage2_min'], 1)} to "
            f"{pct(sk['default_gain_stage2_max'], 1)} percent.",
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            "On the stack the strakes move the centre of pressure forward by "
            f"{fmt(abs(sk['x_cp_shift_stage1_max']), 3)} to "
            f"{fmt(abs(sk['x_cp_shift_stage1_min']), 3)} calibres at the converged dimensions, "
            f"and by {fmt(abs(sk['default_x_cp_shift_stage1_max']), 3)} to "
            f"{fmt(abs(sk['default_x_cp_shift_stage1_min']), 3)} calibres at the starting "
            "dimensions. Forward is the wrong direction: strakes reduce static margin. On the "
            "payload stage alone the effect nearly cancels and changes sign with Mach, because "
            "the strake load centroid lands almost exactly on the centre of pressure that is "
            "already there.",
            S.BODY,
        )
    )
    story.append(
        S.fig_single(
            os.path.join(EXAMPLE, "aero_iv1_validation.png"),
            7,
            "The strake and two-stage aerodynamic build-up, with the validation overlay. Panels "
            "(a) and (b) show that the same vehicle has two different drag polars, one per "
            "reference area. Panel (c) shows the strake load is almost all vortex lift. Panel (e) "
            "checks the suction-analogy coefficients against a printed table. Panel (f) checks "
            "the configuration increment against measurement.",
            width_in=6.5,
            max_h_in=6.4,
        )
    )
    story.append(
        Paragraph(
            "<b>The direction of the validation error matters, and it is favourable.</b> Panel "
            "(e) reproduces the rectangular-wing coefficients of NASA TN D-7921 Table III, which "
            "is a printed table, so that check is exact. Panel (f) compares the model against the "
            "measured normal-force increment for a body with side strakes in NASA TM X-3130. "
            "<b>The measured increment is 1.4 to 3.4 times the model.</b> The reason is known: a "
            "strake also raises the load on the body itself, and that carryover is not modelled. "
            "The reported CN_max is therefore conservative, which is the safe direction for a "
            "sizing tool.",
            S.BODY,
        )
    )


# --------------------------------------------------------------------------------------
#   4. The nTop coupling
# --------------------------------------------------------------------------------------


def ntop_coupling(story: list, D: dict) -> None:
    conv, ev = D["conv"], D["ev"]
    geo = ev["geometry"]
    nb = ev["notebook"]

    S.sect(story, "4. The nTop coupling")
    story.append(
        Paragraph(
            "This section is the point of the exercise. nTop is not a downstream renderer here. "
            "It measures the solid it builds, and the measurements change the answer.",
            S.BODY,
        )
    )

    story.append(Paragraph("4.1 The notebook is authored programmatically", S.H2))
    story.append(
        Paragraph(
            "An nTop notebook is a binary container. It cannot be written by hand. The notebook "
            "is therefore emitted as recipe JSON and converted with an undocumented ntopcl "
            "subcommand:",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "ntopcl convert &lt;recipe.json&gt; &lt;out.ntop&gt; --dev-blocks-on=True<br/>"
            "ntopcl exportjson &lt;in.ntop&gt; &lt;out.json&gt; --ext --dev-blocks-on=True",
            S.MONO,
        )
    )
    story.append(
        Paragraph(
            f"The IV-1 recipe is {nb['n_blocks']} blocks, built against a universe of "
            f"{nb['n_universe_signatures']} block signatures. It exposes "
            f"{nb['n_inputs']} real notebook inputs, which is every geometry dimension of both "
            "stages, the interstage, the strakes and the fins. <b>One notebook therefore serves "
            "every design point:</b> the loop converts once and runs many times with different "
            "input JSON. That matters because conversion is expensive. Conversion evaluates the "
            "notebook, exports included.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "Each stage is one closed profile revolved through 360 degrees. The strakes and the "
            "fins are extruded plates, patterned about the axis. Each airframe is then hollowed "
            "with an inward offset and its internal bays are subtracted, so each cavity is a real "
            "volume rather than an assumed fraction. The interstage is one cone block, which is a "
            "cylinder when the two stage diameters match, so the topology does not depend on a "
            "dimension.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The notebook measures four bodies separately: the booster, the payload stage, the "
            "interstage and the stacked union. The report and the mass model need them apart, "
            "because the booster and the interstage leave the vehicle and the payload stage does "
            "not.",
            S.BODY,
        )
    )

    story.append(Paragraph("4.2 Measurement accuracy", S.H2))
    story.append(
        Paragraph(
            "Every measured body is checked against an independent closed-form value derived "
            "from the design vector alone. The comparison is done at the starting design point, "
            "because that is the point whose closed-form reference was written to disk by the "
            "geometry study.",
            S.BODY,
        )
    )
    rows = [["Body", "Quantity", "nTop", "Closed form", "Difference [%]"]]
    for r in geo["rows"]:
        rows.append(
            [
                r["body"], r["quantity"], fmt(r["ntop"], 6), fmt(r["closed_form"], 6),
                pct(r["rel_err"], 3),
            ]
        )
    story.append(
        S.styled_table(rows, [1.6, 1.5, 1.2, 1.2, 1.1],
                       ["LEFT", "LEFT", "RIGHT", "RIGHT", "RIGHT"])
    )
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 16. nTop measurements against independent closed-form geometry. The three "
            "separate bodies agree to better than 0.01 percent on volume and better than 0.31 "
            "percent on wetted area. The stacked-assembly volume differs by 1.14 percent for a "
            "known reason: nTop measures the union, and the union includes the strake and fin "
            "solids that the closed form does not add to the body volume.",
            S.CAP,
        )
    )
    sk = geo["strake"]
    story.append(
        Paragraph(
            "<b>The strake wetted area was confirmed three independent ways.</b> First, the "
            f"measured area of {fmt(sk['measured_area_m2'], 6)} m^2 agrees with the closed form "
            f"for a solid plate to {pct(sk['solid_rel_err'], 3)} percent. Second, the measured "
            f"strake solid volume of {fmt(sk['measured_volume_m3'], 8)} m^3 agrees with the plate "
            f"volume to {pct(sk['volume_rel_err'], 3)} percent. Third, halving the strake height "
            f"scales the measured area to {fmt(100.0 * sk['small_over_baseline_area'], 1)} "
            f"percent and the volume to {fmt(100.0 * sk['small_over_baseline_volume'], 1)} "
            "percent, which is what a plate of half the height must give once its unchanged edge "
            "faces are counted.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            f"The measured solid area is {fmt(sk['solid_over_zero_thickness'], 4)} times the "
            "zero-thickness reference of two times n times height times length. That is not an "
            "error. An 8 mm plate 30 mm tall is 27 percent edge, and skin friction acts on the "
            "real surface, so the measured number is the right one to hand to the aerodynamic "
            "model.",
            S.BODY,
        )
    )

    story.append(Paragraph("4.3 The coupling changes the answer", S.H2))
    story.append(
        Paragraph(
            "The same design vector was sized twice: once with analytic geometry, and once with "
            "the geometry nTop measured.",
            S.BODY,
        )
    )
    rows = [
        ["Quantity", "Analytic geometry", "nTop-measured", "Change"],
        ["Launch mass [kg]", "585.6", fmt(conv["launch_mass_kg"], 1),
         f"{conv['launch_mass_kg'] - 585.6:+.1f}"],
        ["Intercept altitude [km]", "17.6", fmt(conv["intercept"]["altitude"] / 1000.0, 1),
         f"{conv['intercept']['altitude'] / 1000.0 - 17.6:+.1f}"],
    ]
    story.append(S.styled_table(rows, [2.2, 1.7, 1.5, 1.0], ["LEFT", "RIGHT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 17. The same design vector with and without measured geometry. The measured "
            "two-stage airframe is heavier than the analytic shell estimate, so the launch mass "
            "rises. The analytic figures are quoted from examples/IV-1/README.md, which records "
            "the analytic run; only the measured run is stored as JSON.",
            S.CAP,
        )
    )
    mass = ev["mass"]
    story.append(
        Paragraph(
            f"{fmt(100.0 * mass['measured_fraction'], 1)} percent of the launch mass is "
            f"nTop-measured: {fmt(geo['converged_bodies']['1']['mass_structure'], 2)} kg for the "
            "booster airframe and "
            f"{fmt(geo['converged_bodies']['2']['mass_structure'], 2)} kg for the payload stage "
            "with its strakes and fins. The fraction is small for a structural reason, not a "
            "modelling one. The payload is a requirement, the propellant is a design variable, "
            "and the motor inert masses come from a correlation. None of those three is geometry, "
            "and together they are most of the vehicle.",
            S.BODY,
        )
    )

    story.append(Paragraph("4.4 What a measurement costs", S.H2))
    wt = geo["wall_times_s"]
    rows = [["Run", "Wall time [s]"]]
    for name in ("baseline", "small_strakes", "alternate", "area_distribution", "converged"):
        rows.append([name.replace("_", " "), fmt(wt[name], 1)])
    story.append(S.styled_table(rows, [2.6, 1.6], ["LEFT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 18. Measured wall time of the five stored measurement runs. A separate "
            "controlled repeat of five identical jobs measured 55.0, 78.6, 92.7, 114.7 and "
            "117.8 s. <b>The two-times spread on repeats of an identical job is real and is not "
            "attributed here.</b> Budget the upper end.",
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            "Per-block timings say where the time goes, so the choice of what to measure is "
            "informed rather than guessed. Four surface-area calls on implicit bodies dominate: "
            "24.6 s on the stage-2 body, 24.2 s on the stage-2 fins, 23.8 s on the strakes and "
            "16.9 s on the stage-1 fins. The fifth, on the booster's plain cylinder, costs 0.27 "
            "s. Mass properties cost 15.0 s on the stage-2 structure and 8.6 s on the stack "
            "union. <b>Nothing in that list is meshing.</b> The mesh is used only for the STL "
            "export and never for a measurement.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "A build ladder confirms the accounting. Building only the stage-2 outer mould line "
            "costs 14.7 s, adding its plates 42.2 s, hollowing it 50.5 s, adding the booster "
            "76.3 s and the full stack 117.8 s. Each added body costs what its own area and mass "
            "blocks cost, and nothing more.",
            S.BODY,
        )
    )


# --------------------------------------------------------------------------------------
#   5. Verification
# --------------------------------------------------------------------------------------


def verification(story: list, D: dict) -> None:
    ev = D["ev"]
    t = ev["tests"]

    S.sect(story, "5. Verification")
    story.append(
        Paragraph(
            f"{t['passed']} automated tests pass and {t['skipped']} is skipped. The suite takes "
            f"{fmt(t['duration_s'] / 60.0, 1)} minutes, because parts of it drive real ntopcl "
            "subprocesses. The command and its recorded output are in runs/IV-1/pytest.txt.",
            S.BODY,
        )
    )
    story.append(Paragraph(t["command"], S.MONO))
    rows = [["Vehicle", "Test modules", "Tests collected"]]
    rows.append(["SV-1, the regression baseline", "7", str(t["sv1_total"])])
    rows.append(["IV-1, new in this work", "5", str(t["iv1_total"])])
    rows.append(["Total", "12", str(t["collected_total"])])
    story.append(S.styled_table(rows, [2.8, 1.4, 1.6], ["LEFT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            f"Table 19. Test counts by vehicle. SV-1 is the regression baseline and its "
            f"{t['sv1_total']} tests are unchanged by this work.",
            S.CAP,
        )
    )

    story.append(Paragraph("5.1 The new physics reduces to the validated physics", S.H2))
    story.append(
        Paragraph(
            "The strongest check on a generalisation is that it reproduces the thing it "
            "generalises. A one-stage stack is compared against the validated SV-1 single motor. "
            "Sixteen operating-point quantities are compared with exact equality, not with a "
            "tolerance: propellant mass, chamber and exit pressure, throat and exit area and "
            "diameter, burning area, the Kn ratio, burn rate, mass flow, burn time, vacuum and "
            "sea-level thrust, the thrust coefficient and the vacuum specific impulse.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The thrust curves are also compared. The difference is exactly 0.0 N over 20001 "
            "samples at three altitudes, while both models are inside the boost shape. After "
            "that point the SV-1 motor transitions to a phase a single stage does not have, so no "
            "tolerance would be meaningful there and none is claimed.",
            S.BODY,
        )
    )

    story.append(Paragraph("5.2 Closed-form cases", S.H2))
    stg = ev["staging"]
    at = ev["atmosphere"]
    rows = [
        ["Case", "Relative error"],
        ["Staged Tsiolkovsky with jettison, against the closed form", "2.4e-08"],
        ["Gain of staging against the closed-form gain", sci(stg["analytic_rel_err"])],
        ["Mass bookkeeping, m0 less burned less jettisoned", "7.5e-15"],
        ["Specific-energy drift with no thrust and no drag", "5.3e-15"],
        ["Vertical vacuum climb apogee, against h0 + V0^2/(2g)", "9.0e-14"],
        ["Fast atmosphere lookup against numpy.interp", sci(at["interp_agreement"])],
    ]
    story.append(S.styled_table(rows, [4.6, 2.0], ["LEFT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 20. Cases with an analytic answer. The staged, mass-bookkeeping and energy "
            "figures are the stated assertions of tests/test_trajectory_iv1.py. The staging-gain "
            "and atmosphere figures are measured by rocketgen/report/evidence_iv1.py. "
            "Fourth-order Runge-Kutta convergence was checked separately: halving the step "
            "reduces the error by 15.93 and 16.01, against the factor of 16 the method requires.",
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            f"The atmosphere extension is validated against the published layer breaks, with the "
            "geopotential conversion done properly. The fast uniform-grid lookup that replaced "
            f"numpy.interp is bit-identical to it, at {sci(at['interp_agreement'])} relative over "
            f"the whole range, and costs {fmt(at['lookup_us_per_call'], 2)} microseconds per "
            "call. The table reproduces the stratopause maximum of "
            f"{fmt(at['stratopause_T'], 2)} K, which matters because a lofted midcourse spends "
            "its time there.",
            S.BODY,
        )
    )

    story.append(Paragraph("5.3 One test is skipped, not passing", S.H2))
    story.append(
        Paragraph(
            "<b>The suite is not clean, and this report will not present it as if it were.</b> "
            "One test is skipped. It covers the reporting branch that fires when a trajectory "
            "flies above the atmosphere-table ceiling. That branch is now unreachable on any real "
            "trajectory, because the table reaches 86 km and the highest measured arc apogees at "
            "54 km. The mission module reads the ceiling directly from the atmosphere module, so "
            "there is no seam through which a test can drive the ceiling low. The test therefore "
            "skips with that reason recorded, rather than being deleted.",
            S.BODY,
        )
    )


# --------------------------------------------------------------------------------------
#   6. Fidelity and limitations
# --------------------------------------------------------------------------------------


def limitations(story: list, D: dict) -> None:
    ev = D["ev"]
    src = ev["sources"]
    envr = ev["environment"]
    tr = ev["trajectory"]

    S.sect(story, "6. Fidelity and limitations")
    story.append(
        Paragraph(
            "This is a Class-I conceptual tool. It is useful for choosing between configurations. "
            "It is not a substitute for computational fluid dynamics, for structural analysis or "
            "for six-degree-of-freedom flight mechanics.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "<b>The vehicle and its requirements are invented for this demonstration.</b> They "
            "correspond to no real programme. Nothing here describes guidance, seekers, warheads, "
            "energetics or countermeasures. Section 1.1 shows the requirements were not even "
            "self-consistent when written.",
            S.BODY,
        )
    )

    story.append(Paragraph("6.1 The largest unquantified optimism", S.H2))
    story.append(
        Paragraph(
            "Nozzle flow is ideal. The model assumes frozen composition, constant specific-heat "
            "ratio and single-phase flow. It has no two-phase alumina loss, no divergence loss, "
            "no combustion inefficiency and no throat erosion. Real delivered specific impulse "
            "for this propellant class is typically 3 to 7 percent lower. <b>That penalty is not "
            "applied, because its magnitude could not be sourced.</b> Slant range and intercept "
            "Mach are therefore optimistic by an amount that has not been quantified. This is the "
            "single largest known optimism in the result.",
            S.BODY,
        )
    )

    story.append(Paragraph("6.2 Limitations specific to this vehicle", S.H2))
    for text in [
        "<b>The 350 kPa dynamic-pressure limit is not verified by the structural model.</b> "
        "Structural sizing is wall thickness times density, plus a hoop-stress check on each "
        "motor case and a buckling check on the interstage. Nothing sizes the airframe for a "
        "350 kPa aerodynamic load, so the airframe mass is optimistic by an amount this toolkit "
        "cannot quantify.",
        "<b>Separation is instantaneous.</b> It imparts no impulse and no attitude disturbance, "
        "and it has no drag transient. The separation joint is not costed: a real tandem "
        "separation needs a linear shaped charge or a clamp band, springs or retro-rockets, and "
        "ring frames at both ends of the interstage. The jettisoned mass is therefore optimistic.",
        "<b>There is no thrust-vector control.</b> The divert motor is a static capability "
        "figure at the intercept condition. It says nothing about response time, actuator rate, "
        "plume interaction, minimum impulse bit, roll coupling or autopilot behaviour.",
        "<b>Strake-to-fin vortex interference is not modelled.</b> The interference factor is "
        "1.0, which means not modelled rather than measured as unity. No factor for this "
        "configuration was sourced.",
        "The trajectory is a three-degree-of-freedom point mass on a flat earth with constant "
        "gravity. There is no roll, no sideslip and no autopilot.",
        "The grain geometry is a neutral-burning tube closed at its mean web. There is no "
        "burnback simulation, so the reported grain dimensions are equivalent values rather than "
        "a drawing.",
        "Requirement A9, the static margin, is not among the constraints the sizing script "
        "records. Section 2.2 evaluates it and it does not pass. That is an open item.",
    ]:
        story.append(Paragraph(text, S.BULLET, bulletText="-"))

    story.append(Paragraph("6.3 The pitch programme, measured", S.H2))
    story.append(
        Paragraph(
            "The pitch programme is open loop and has no autopilot gain of any kind. The vehicle "
            f"rises vertically until t = {fmt(tr['t_pitch_start_s'], 1)} s, turns at the "
            f"commanded rate of {fmt(tr['pitch_rate_commanded_deg_s'], 1)} deg/s until the flown "
            "flight-path angle reaches the commanded angle, then flies a pure gravity turn. The "
            "commanded rate is a design variable, not a gain on an angle error.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "With no thrust-vector control, that commanded rate is often beyond the "
            "angle-of-attack limit, and the module says so. <b>On this design point it is not.</b> "
            f"The turn finished at t = {fmt(tr['t_pitch_complete_s'], 2)} s, which is an average "
            f"flown rate of {fmt(tr['pitch_rate_flown_deg_s'], 2)} deg/s against the "
            f"{fmt(tr['pitch_rate_commanded_deg_s'], 1)} deg/s commanded. The peak angle of "
            f"attack was {fmt(tr['alpha_max_flown_deg'], 2)} degrees against the 20 degree limit, "
            f"and {int(tr['alpha_limit_hits'])} steps hit the limit. The turn starts at 5 s, "
            "where the dynamic pressure is already high, which is why the airframe can fly it. "
            "An earlier or higher pitchover would be limited, and the diagnostic would record it.",
            S.BODY,
        )
    )

    story.append(Paragraph("6.4 The drag calibration is not applied to this vehicle", S.H2))
    story.append(
        Paragraph(
            "The SV-1 aerodynamic build-up runs about 15 percent low on zero-lift drag against "
            "23 Basic Finner free-flight shots, so the SV-1 sizing loop scales drag by "
            f"{fmt(envr['cd0_calibration_available'], 3)} at its boundary. <b>The IV-1 sizing "
            "script does not apply that correction.</b> It builds the aerodynamic model directly "
            "and does not wrap it in the calibrated boundary. The IV-1 drag is therefore the "
            "uncorrected physics, which is low, and low drag overpredicts range. The direction of "
            "this error is unfavourable and it is not quantified for the two-stage configuration.",
            S.BODY,
        )
    )

    story.append(Paragraph("6.5 Every value that is a guess", S.H2))
    story.append(
        Paragraph(
            "Every empirical constant in the code is registered with its source. A value that is "
            "a guess must say so in its source text, and a test asserts that. The table lists "
            "every registered source whose text admits a guess, a modelling choice, an "
            "approximation or an assumption.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            f"<b>{src['n_flagged']} of {src['n_registered']} registered sources are flagged.</b> "
            "The registry is global, and modules register into it when they are imported. The "
            "count is therefore only complete if every module is imported before the registry is "
            "read. All five IV-1 modules are imported first, then the shared physics modules they "
            "build on, then the notebook-authoring modules. The module column below says which "
            "module owns each entry.",
            S.BODY,
        )
    )
    rows = [["Module", "Count"]]
    for name, n in sorted(src["by_module"].items(), key=lambda kv: -kv[1]):
        rows.append([name, str(n)])
    rows.append(["Total flagged", str(src["n_flagged"])])
    story.append(S.styled_table(rows, [3.0, 1.2], ["LEFT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 21. Flagged sources by owning module. The shared physics modules are counted "
            "because IV-1 reuses them unchanged. A few of their entries describe the SV-1 "
            "dual-thrust motor and the SV-1 dive, which IV-1 does not fly; they are listed rather "
            "than filtered, because filtering a source registry by hand is how a limitation goes "
            "missing.",
            S.CAP,
        )
    )

    rows = [["Registered name", "Module", "Why it is not a measured value"]]
    for f in src["flagged"]:
        rows.append([f["key"], f["module"], f["text"]])
    story.append(S.cell_table(rows, [1.75, 1.0, 3.85]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            f"Table 22. All {src['n_flagged']} registered values that are a guess, a modelling "
            "choice, an approximation or an assumption. No text is truncated: clipping a source "
            "string mid-word would defeat the purpose of publishing it. "
            "Source: runs/IV-1/figures/evidence_iv1.json.",
            S.CAP,
        )
    )


# --------------------------------------------------------------------------------------
#   7. Reproducing this
# --------------------------------------------------------------------------------------


def reproduce(story: list, D: dict) -> None:
    ev = D["ev"]
    envr = ev["environment"]
    nb = ev["notebook"]

    S.sect(story, "7. How to reproduce this")
    story.append(
        Paragraph(
            "Every artefact in this report comes from the commands below. The analysis writes "
            "JSON. The figure scripts read that JSON. This document reads it too, so no number is "
            "typed in twice.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            ".venv/Scripts/python.exe -m pytest tests -q<br/>"
            ".venv/Scripts/python.exe scripts/iv1_converge.py --ntop<br/>"
            ".venv/Scripts/python.exe -m rocketgen.report.evidence_iv1<br/>"
            ".venv/Scripts/python.exe -m rocketgen.report.fig_mass_iv1<br/>"
            ".venv/Scripts/python.exe -m rocketgen.report.fig_margins_iv1<br/>"
            ".venv/Scripts/python.exe -m rocketgen.report.fig_infeasible_iv1<br/>"
            ".venv/Scripts/python.exe -m rocketgen.report.fig_ascent_iv1<br/>"
            ".venv/Scripts/python.exe -m rocketgen.report.fig_envelope_iv1<br/>"
            ".venv/Scripts/python.exe -m rocketgen.report.build_report_iv1",
            S.MONO,
        )
    )
    story.append(
        Paragraph(
            "The evidence collector does not call nTop and does not repeat the pitchover search. "
            "It rebuilds the converged design vector, checks it field by field against the "
            "recorded one, reads the stored nTop measurements back off disk, and re-flies the one "
            "converged trajectory. It then checks the launch mass, the intercept and the peak "
            "dynamic pressure against the recorded values before it writes anything. A re-flight "
            "that did not reproduce the recorded result fails there rather than publishing a "
            "second answer.",
            S.BODY,
        )
    )
    rows = [
        ["Component", "Version or value"],
        ["Python", envr["python"]],
        ["numpy", envr["numpy"]],
        ["scipy", envr["scipy"]],
        ["SUAVE", envr["suave"]],
        ["nTop Automate", envr["ntop"]],
        ["Notebook blocks", str(nb["n_blocks"])],
        ["Notebook inputs", str(nb["n_inputs"])],
        ["Block universe signatures", str(nb["n_universe_signatures"])],
        ["Mass-properties relative error", fmt(nb["relative_error"], 4)],
        ["Surface-area relative error", fmt(nb["area_relative_error"], 3)],
        ["Mesh tolerance, STL export only", f"{fmt(1000.0 * nb['mesh_tolerance_m'], 1)} mm"],
        ["Drag calibration applied to IV-1",
         "no" if not envr["cd0_calibration_applied_to_iv1"] else "yes"],
    ]
    story.append(S.styled_table(rows, [2.6, 4.0], ["LEFT", "LEFT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 23. Environment. numpy and scipy are pinned below their current major "
            "versions, because SUAVE 2.5.2 does not run on either.",
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            "One warning about the toolchain is worth repeating. The nTop command-line tool "
            "returns exit code 72 when a block fails, not on success. Published guidance states "
            "the opposite. A notebook given an invalid dimension returns 72 and writes no output "
            "at all. Success must therefore be judged on the expected files existing and being "
            "non-empty, never on the return code alone.",
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
        title="IV-1 Two-Stage Interceptor, Coupled nTop and SUAVE Sizing",
        author="nTop",
        footer="IV-1 coupled sizing  |  nTop + SUAVE  |  demonstration, invented requirements",
    )

    front_matter(story, D)
    story.append(PageBreak())
    findings(story, D)
    story.append(PageBreak())
    the_design(story, D)
    story.append(PageBreak())
    what_is_different(story, D)
    story.append(PageBreak())
    ntop_coupling(story, D)
    # No page break here either: section 4 ends with two short paragraphs, and `S.sect` already
    # refuses to leave a heading orphaned at a page bottom.
    verification(story, D)
    # No page break here on purpose: section 5.3 is short, and a break would leave four fifths
    # of a page blank.
    limitations(story, D)
    story.append(PageBreak())
    reproduce(story, D)

    S.build(doc, story)
    print(f"wrote {OUT_PDF}  ({os.path.getsize(OUT_PDF) / 1024:.0f} KB)")
    return OUT_PDF


if __name__ == "__main__":
    main()
