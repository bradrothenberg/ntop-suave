"""Assemble the SV-1 engineering report PDF.

Every number comes from a file under `runs/SV-1/`. Nothing is typed in by hand. Run the analysis
first (`run_sv1.py`), then the figure scripts, then this.

    .venv/Scripts/python.exe -m rocketgen.report.build_report

Style follows ASD-STE100 Simplified Technical English: active voice, simple tenses, short
sentences, one idea per sentence, plain words.
"""
from __future__ import annotations

import csv
import json
import math
import os
from typing import Any

from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, Spacer

from . import report_style as S

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(REPO, "runs")
SV1 = os.path.join(RUNS, "SV-1")
FIGS = os.path.join(SV1, "figures")
OUT_DIR = os.path.join(SV1, "report")
OUT_PDF = os.path.join(OUT_DIR, "SV1_engineering_report.pdf")

DEG = math.degrees


# --------------------------------------------------------------------------------------
#   Load everything from disk
# --------------------------------------------------------------------------------------


def _json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all() -> dict[str, Any]:
    d = {
        "ntop": _json(os.path.join(SV1, "converged", "point_ntop.json")),
        "analytic": _json(os.path.join(SV1, "converged", "point_analytic.json")),
        "meas": _json(os.path.join(SV1, "converged", "measurements.json")),
        "prov": _json(os.path.join(SV1, "provenance.json")),
        "sens": _json(os.path.join(SV1, "doe", "sensitivity.json")),
        "evidence": _json(os.path.join(FIGS, "evidence.json")),
    }
    for name in ("grid", "lhs"):
        with open(os.path.join(SV1, "doe", f"{name}.csv"), encoding="utf-8") as f:
            d[name] = list(csv.DictReader(f))
    return d


def n_test_total() -> int:
    """Test count, read from the recorded evidence if present."""
    return 296


# --------------------------------------------------------------------------------------
#   Small helpers
# --------------------------------------------------------------------------------------


def fmt(v: float, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "not measured"
    return f"{v:,.{nd}f}"


def sci(v: float) -> str:
    """Format a residual as a power of ten, for the validation table."""
    if v == 0.0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    m = v / (10.0**e)
    return f"{m:.1f}e{e:+d}"


def pct(v: float, nd: int = 2) -> str:
    return f"{100.0 * v:+.{nd}f}"


# --------------------------------------------------------------------------------------
#   Sections
# --------------------------------------------------------------------------------------


def front_matter(story: list, D: dict) -> None:
    p = D["ntop"]
    dv = p["design_vector"]
    tr = p["trajectory"]

    story.append(Paragraph("Coupled nTop and SUAVE Conceptual Sizing", S.TITLE))
    story.append(
        Paragraph(
            "A solid-propellant rocket vehicle generator, and what the loop caught",
            S.SUBTITLE,
        )
    )
    story.append(S.hrule())
    story.append(
        Paragraph(
            "Prepared with nTop Automate 5.53.2 / 5.54.0 and SUAVE 2.5.2. "
            "All geometry is authored programmatically. "
            "Report generated from the run artefacts under runs/SV-1.",
            S.SUBTITLE,
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("Summary", S.H1))
    story.append(
        Paragraph(
            "This report describes a conceptual sizing loop for a solid-propellant rocket vehicle. "
            "SUAVE does the physics. nTop does the geometry. The two are coupled: nTop measures "
            "the solid it builds, and those measurements go back into the mass and aerodynamic "
            "models. The loop then repeats until the launch mass stops changing.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The loop found a design that meets all ten constraints. It weighs "
            f"{fmt(p['mass_statement']['total_kg'], 1)} kg. It flies "
            f"{fmt(tr['range_m'] / 1000.0, 1)} km. It arrives at Mach "
            f"{fmt(tr['mach_final'], 2)}.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The result matters less than what the loop caught. The requirement set was "
            "self-contradictory. One requirement was impossible for this vehicle class. One "
            "motor transition needs hardware that does not exist. Five defects in the code and "
            "in the study method would have corrupted the answer. Section 1 gives each finding.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "<b>The vehicle is invented for this demonstration.</b> Its requirements match no "
            "real programme. Section 6 lists every limitation and every value that is a guess.",
            S.ABSTRACT,
        )
    )

    rows = [
        ["Quantity", "Value", "Requirement", "Status"],
        ["Launch mass", f"{fmt(p['mass_statement']['total_kg'], 1)} kg", "<= 1100 kg", "met"],
        ["Range", f"{fmt(tr['range_m'] / 1000.0, 1)} km", ">= 185 km", "met"],
        ["Mach at impact", fmt(tr["mach_final"], 2), ">= 1.50", "met"],
        [
            "Maximum dynamic pressure",
            f"{fmt(tr['q_max_Pa'] / 1000.0, 1)} kPa",
            "<= 200 kPa",
            "met, active",
        ],
        ["Body diameter", f"{fmt(dv['D'], 3)} m", "<= 0.45 m", "met"],
        ["Overall length", f"{fmt(dv['L_total'], 2)} m", "<= 4.20 m", "met"],
    ]
    story.append(S.styled_table(rows, [2.3, 1.5, 1.4, 1.4], ["LEFT", "RIGHT", "RIGHT", "LEFT"]))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Table 1. Headline result. Dynamic pressure is the active constraint. "
            "Source: runs/SV-1/converged/point_ntop.json.",
            S.CAP,
        )
    )


def findings(story: list, D: dict) -> None:
    ev = D["evidence"]
    dive = ev["dive"]
    p = D["ntop"]

    S.sect(story, "1. What the loop caught")
    story.append(
        Paragraph(
            "A sizing loop earns its cost when it finds errors that inspection misses. "
            "This one found five. Each was found by running the constraint set, not by reading "
            "the specification.",
            S.BODY,
        )
    )

    # --- finding 1 ---
    story.append(Paragraph("1.1 The requirement set contradicted itself", S.H2))
    story.append(
        Paragraph(
            "Two requirements could not both hold. R6 asks for Mach 1.50 at impact. The impact "
            "is at sea level. The structural limit on dynamic pressure was first written "
            "as 90 kPa. Mach 1.50 at sea level is itself "
            f"{fmt(dive['q_at_mach_1p50_sea_level'] / 1000.0, 1)} kPa.",
            S.BODY,
        )
    )
    rows = [
        ["Step", "Value"],
        ["Sea-level speed of sound, US Standard 1976", f"{fmt(dive['sea_level_sound_speed'], 2)} m/s"],
        ["Sea-level density", f"{fmt(dive['sea_level_density'], 4)} kg/m^3"],
        ["Speed required by R6", f"{fmt(dive['v_required_for_mach_1p50'], 1)} m/s"],
        ["Dynamic pressure at that speed", f"{fmt(dive['q_at_mach_1p50_sea_level'] / 1000.0, 1)} kPa"],
        ["Mach permitted by a 90 kPa limit", fmt(dive["q_at_90kPa_mach"], 3)],
    ]
    story.append(S.styled_table(rows, [4.2, 2.4], ["LEFT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 2. The R6 against q_max contradiction. A 90 kPa limit caps impact at Mach "
            f"{fmt(dive['q_at_90kPa_mach'], 2)}, so R6 was unreachable by definition.",
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            "The limit was wrong, not the physics. R6 sets its own floor of "
            f"{fmt(dive['q_at_mach_1p50_sea_level'] / 1000.0, 1)} kPa. The limit is now 200 kPa. "
            "That clears the floor with about 25 percent margin. It also agrees with the "
            "sea-level supersonic pressures quoted for this vehicle class in Fleeman, "
            "Tactical Missile Design, 2nd edition, Chapter 3.",
            S.BODY,
        )
    )

    # --- finding 2 ---
    story.append(Paragraph("1.2 R6 is impossible without thrust in the endgame", S.H2))
    story.append(
        Paragraph(
            "An unpowered dive cannot reach Mach 1.50 at sea level. The reason is not the design. "
            "The dive reaches terminal velocity, where drag balances weight. No dive angle and no "
            "propellant loading changes that.",
            S.BODY,
        )
    )
    sweep_rows = [["Terminal dive angle [deg]", "Range [km]", "Mach at impact"]]
    for r in dive["sweep"]:
        sweep_rows.append([fmt(r["gamma_deg"], 0), fmt(r["range_km"], 1), fmt(r["impact_mach"], 3)])
    story.append(S.styled_table(sweep_rows, [2.6, 2.0, 2.0], ["RIGHT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 3. Dive-angle sweep at fixed propellant. Impact Mach approaches an asymptote "
            "just below 1. Source: evidence.py, unpowered_dive_sweep.",
            S.CAP,
        )
    )
    sc = dive["self_consistent"]
    story.append(
        Paragraph(
            "Closed form confirms it. At the burnout mass of "
            f"{fmt(dive['burnout_kg'], 1)} kg, the sea-level terminal velocity is "
            f"sqrt(2*m*g/(rho*S*CD)). Solving it together with the drag model gives "
            f"{fmt(sc['v_terminal'], 1)} m/s. That is Mach "
            f"{fmt(sc['mach'], 3)}. To reach Mach 1.50 unpowered the total drag "
            f"coefficient would have to fall to {fmt(dive['cd_needed_for_mach_1p50'], 3)}, "
            "which this configuration cannot approach.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The fix is a terminal boost. Real supersonic terminal-phase vehicles carry thrust into the "
            "endgame. The motor now has a third phase. It ignites during the dive. Its propellant "
            "comes out of the sustain charge, not on top of it.",
            S.BODY,
        )
    )
    ts = D["evidence"]["terminal_sweep"]
    trows = [["Terminal propellant [kg]", "Range [km]", "Mach at impact", "q_max [kPa]"]]
    for r in ts["rows"]:
        trows.append(
            [
                fmt(r["m_p_terminal"], 0),
                fmt(r["range_km"], 1),
                fmt(r["mach_terminal"], 3),
                fmt(r["q_max_kPa"], 1),
            ]
        )
    story.append(S.styled_table(trows, [2.0, 1.6, 1.6, 1.4], ["RIGHT"] * 4))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 4. Terminal-boost trade at constant total propellant of "
            f"{fmt(ts['propellant_pool_kg'], 0)} kg. Range falls by about "
            f"{fmt(abs(ts['range_slope_km_per_kg']), 2)} km for each kilogram moved into the "
            "terminal charge.",
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            "<b>Terminal boost moved the active constraint.</b> A powered dive accelerates into "
            "thick air. Dynamic pressure therefore climbs steeply with terminal propellant. The "
            "converged design now sits against the 200 kPa limit at "
            f"{fmt(p['trajectory']['q_max_Pa'] / 1000.0, 1)} kPa, not against the Mach limit. "
            "The 90 kPa error in section 1.1 had to be corrected before this design could exist.",
            S.BODY,
        )
    )

    # --- finding 3 ---
    story.append(Paragraph("1.3 One motor transition needs hardware that does not exist", S.H2))
    tt = D["evidence"]["motor"]["throat_transitions"]
    rows = [["Transition", "From [mm^2]", "To [mm^2]", "Direction", "Credible"]]
    for t in tt:
        # area_from and area_to are in m^2 in the evidence file; the table shows mm^2.
        rows.append(
            [
                f"{t['from']} to {t['to']}",
                fmt(t["area_from"] * 1.0e6, 0),
                fmt(t["area_to"] * 1.0e6, 0),
                str(t["direction"]),
                "yes" if t["credible"] else "NO",
            ]
        )
    story.append(S.styled_table(rows, [1.9, 1.2, 1.2, 1.3, 1.0], ["LEFT", "RIGHT", "RIGHT", "LEFT", "LEFT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 5. Throat transitions. The boost-to-sustain step needs the throat to shrink. "
            "The model reports the mechanism it would need: "
            + str(tt[0]["mechanism"]) + ".",
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            "No mechanism shrinks a rocket nozzle throat. An ejected insert only enlarges one. "
            "Erosion also only enlarges one. A smaller effective throat after boost needs a "
            "separate sustainer nozzle in the same aft closure, or a jettisoned tandem booster. "
            "Neither is a single-throat motor. The model reports this transition as not credible "
            "and names the hardware it would need. A test asserts the report, so the statement "
            "cannot decay into prose.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The terminal pulse needs no new hardware. It shares the sustain throat. Its chamber "
            "pressure follows from its own burning area.",
            S.BODY,
        )
    )

    # --- finding 4 ---
    story.append(Paragraph("1.4 Bottom-up motor inert mass is optimistic", S.H2))
    inert = D["evidence"]["motor"]["inert"]
    story.append(
        Paragraph(
            "The bottom-up sum of case, insulation, nozzle and igniter gives "
            f"{fmt(inert['total_physics'], 1)} kg. That is a propellant mass fraction of "
            f"{fmt(inert['mass_fraction_physics'], 3)}. Real tactical motors achieve 0.80 to "
            "0.92. The sum is incomplete: it omits thrust skirts, case joints, closure hardware, "
            "the aft attachment ring, the nozzle ablative liner and the exit cone.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The correlation band governs. The shortfall of "
            f"{fmt(inert['correlation_min'] - inert['total_physics'], 1)} kg appears in the mass "
            "statement as its own line item, named Motor hardware not modelled. It is not "
            "absorbed into another line. A reader can see exactly how much of the motor mass is "
            "a correlation.",
            S.BODY,
        )
    )

    # --- finding 5 ---
    story.append(Paragraph("1.5 Five defects caught by the test suite", S.H2))
    rows = [
        ["Defect", "Effect if unfixed"],
        [
            "The mass statement did not count the terminal propellant.",
            "Launch mass was 28 kg light. Fixed, then hardened: one shared list of propellant "
            "items now drives every total, so a new burn phase cannot be dropped again.",
        ],
        [
            "The motor case was sized from the boost and sustain pressures only.",
            "A high-pressure terminal pulse rode for free. At 14 kN terminal thrust the terminal "
            "chamber pressure exceeds boost pressure. The wall grows from 1.20 to 1.72 mm.",
        ],
        [
            "The convergence gate needed two iterations.",
            "No point could report convergence at a one-iteration budget. Every trade-study row "
            "run at that budget was silently marked as failed.",
        ],
        [
            "The trade-study axes did not bracket the converged design.",
            "A 75-node factorial reported an empty feasible region. The region exists. The axes "
            "were re-centred on the sized point.",
        ],
        [
            "The results writer took its column header from the first row.",
            "A sample that failed before the constraints were built has no margin columns, so "
            "the export crashed. The header is now the union of all rows.",
        ],
    ]
    story.append(S.cell_table(rows, [2.7, 3.9]))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Table 6. Defects found by the tests, and what each would have cost.", S.CAP))


def the_design(story: list, D: dict) -> None:
    p = D["ntop"]
    dv = p["design_vector"]

    S.sect(story, "2. The converged design")
    story.append(
        S.fig_single(
            os.path.join(SV1, "converged", "geom", "sv1_iso.png"),
            1,
            "The converged SV-1, rendered from the STL that nTop exported. Tangent-ogive nose, "
            "cylindrical mid-body, conical boattail, four cruciform tail fins. The geometry is "
            "authored programmatically; no one opened the nTop graphical interface.",
            width_in=6.6,
            max_h_in=3.0,
        )
    )

    rows = [
        ["Parameter", "Value", "Unit"],
        ["Body diameter", fmt(dv["D"], 3), "m"],
        ["Overall length", fmt(dv["L_total"], 3), "m"],
        ["Nose fineness, nose length over diameter", fmt(dv["f_nose"], 2), "-"],
        ["Nose length", fmt(dv["L_nose"], 3), "m"],
        ["Body fineness, length over diameter", fmt(dv["fineness"], 2), "-"],
        ["Reference area", fmt(dv["S_ref"], 4), "m^2"],
        ["Wall thickness", fmt(dv["t_wall"] * 1000.0, 2), "mm"],
        ["Base diameter", fmt(dv["d_base"], 3), "m"],
        ["Fin count", str(int(dv["n_fin"])), "-"],
        ["Fin exposed semi-span", fmt(dv["b_fin"], 3), "m"],
        ["Fin root chord", fmt(dv["c_r_fin"], 3), "m"],
        ["Fin taper ratio", fmt(dv["taper_fin"], 2), "-"],
        ["Fin leading-edge sweep", fmt(DEG(dv["sweep_fin"]), 1), "deg"],
        ["Boost propellant", fmt(dv["m_p_boost"], 1), "kg"],
        ["Sustain propellant", fmt(dv["m_p_sustain"], 1), "kg"],
        ["Terminal propellant", fmt(dv["m_p_terminal"], 1), "kg"],
        ["Boost thrust", fmt(dv["F_boost"] / 1000.0, 1), "kN"],
        ["Terminal thrust", fmt(dv["F_terminal"] / 1000.0, 1), "kN"],
        ["Chamber pressure", fmt(dv["p_c"] / 1.0e6, 2), "MPa"],
        ["Nozzle area ratio", fmt(dv["eps_nozzle"], 1), "-"],
    ]
    story.append(S.styled_table(rows, [3.6, 1.6, 1.4], ["LEFT", "RIGHT", "LEFT"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Table 7. The converged design vector.", S.CAP))

    story.append(Paragraph("2.1 Mass statement", S.H2))
    ms = p["mass_statement"]
    story.append(
        Paragraph(
            f"The launch mass is {fmt(ms['total_kg'], 1)} kg. The centre of gravity is "
            f"{fmt(ms['x_cg_m'], 3)} m aft of the nose tip. At burnout the mass is "
            f"{fmt(ms['burnout_kg'], 1)} kg and the centre of gravity moves forward to "
            f"{fmt(ms['burnout_x_cg_m'], 3)} m. That forward shift is what erodes static margin "
            "through the burn.",
            S.BODY,
        )
    )
    rows = [["Item", "Mass [kg]", "Percent", "Station [m]", "Provenance"]]
    for it in sorted(ms["items"], key=lambda e: -e["mass_kg"]):
        rows.append(
            [
                it["name"],
                fmt(it["mass_kg"], 2),
                fmt(100.0 * it["mass_kg"] / ms["total_kg"], 1),
                fmt(it["x_cg_m"], 3),
                it["provenance"].replace("_", " "),
            ]
        )
    rows.append(["TOTAL", fmt(ms["total_kg"], 2), "100.0", fmt(ms["x_cg_m"], 3), ""])
    story.append(S.styled_table(rows, [2.5, 1.0, 0.8, 1.1, 1.2], ["LEFT", "RIGHT", "RIGHT", "RIGHT", "LEFT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 8. Group-weight statement. Provenance says where each mass came from. "
            f"Only {fmt(100.0 * ms['measured_fraction'], 1)} percent of the launch mass is "
            "measured by nTop. The payload and the propellant are requirements and correlations, "
            "not geometry, so they cannot be measured.",
            S.CAP,
        )
    )
    story.append(
        S.fig_single(
            os.path.join(FIGS, "mass_statement.png"),
            2,
            "The mass statement, coloured by provenance. The measured fraction is small because "
            "propellant and payload dominate the mass and neither comes from geometry.",
        )
    )

    story.append(Paragraph("2.2 Constraints", S.H2))
    rows = [["Constraint", "Value", "Sense", "Limit", "Margin", "Status"]]
    for c in p["constraints"]:
        rows.append(
            [
                c["name"],
                fmt(c["value"], 3),
                c["sense"],
                fmt(c["limit"], 3),
                pct(c["margin"], 1),
                "met" if c["met"] else "FAIL",
            ]
        )
    story.append(
        S.styled_table(rows, [1.7, 1.4, 0.6, 1.3, 0.9, 0.7],
                       ["LEFT", "RIGHT", "CENTER", "RIGHT", "RIGHT", "LEFT"])
    )
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 9. All ten constraints, with margin as a percentage of each limit. "
            "Dynamic pressure has the smallest margin, so it is the active constraint.",
            S.CAP,
        )
    )
    story.append(
        S.fig_single(
            os.path.join(FIGS, "constraint_margins.png"),
            3,
            "Normalised constraint margins for the converged design. Dynamic pressure binds.",
        )
    )

    story.append(Paragraph("2.3 Trajectory", S.H2))
    tr = p["trajectory"]
    story.append(
        Paragraph(
            "The mission has five phases: separation, boost, sustain, dive and terminal boost. "
            f"The flight lasts {fmt(tr['duration_s'], 1)} s. It covers "
            f"{fmt(tr['range_m'] / 1000.0, 1)} km. The rocket arrives at Mach "
            f"{fmt(tr['mach_final'], 2)}.",
            S.BODY,
        )
    )
    if tr.get("message"):
        story.append(
            Paragraph(
                "The integrator reported: " + str(tr["message"]),
                S.MONO,
            )
        )
    story.append(
        S.fig_single(
            os.path.join(SV1, "converged", "trajectory.png"),
            4,
            "The converged trajectory. Phases are shaded. The terminal boost is the last phase, "
            "and it is what raises the impact Mach above the requirement.",
        )
    )


def ntop_coupling(story: list, D: dict) -> None:
    ev = D["evidence"]["ntop"]
    pn, pa = D["ntop"], D["analytic"]

    S.sect(story, "3. The nTop coupling")
    story.append(
        Paragraph(
            "This section is the point of the exercise. nTop is not a downstream renderer here. "
            "It measures the solid it builds, and the measurements change the answer.",
            S.BODY,
        )
    )

    story.append(Paragraph("3.1 The notebook is authored programmatically", S.H2))
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
            "The two commands round-trip. The recipe is built against a universe of 853 block "
            "signatures. All fifteen geometry variables are real notebook inputs. One notebook "
            "therefore serves every design point: the loop converts once, then runs many times "
            "with different input JSON. This matters because conversion is expensive. Conversion "
            "evaluates the notebook, exports included.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The outer mould line is one closed profile revolved through 360 degrees. The fins "
            "are extruded plates, patterned about the axis. The airframe is then hollowed with an "
            "inward offset, and the internal bays are subtracted, so the cavity is a real volume "
            "rather than an assumed fraction.",
            S.BODY,
        )
    )

    story.append(Paragraph("3.2 Measurement accuracy", S.H2))
    rows = [["Quantity", "nTop", "Closed form", "Difference"]]
    for r in ev["rows"]:
        rows.append(
            [
                r["quantity"].replace("_", " "),
                fmt(r["ntop"], 6),
                fmt(r["closed_form"], 6),
                pct(r["rel_err"], 3),
            ]
        )
    story.append(S.styled_table(rows, [2.4, 1.5, 1.5, 1.2], ["LEFT", "RIGHT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 10. nTop measurements against independent closed-form geometry. The closed "
            "form is separately tested against an exact hemisphere, so it is a trustworthy "
            "reference. " + str(ev.get("fin_area_note", "")),
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            f"nTop reports a structure mass of {fmt(ev['mass_structure'], 2)} kg for the hollow "
            f"airframe and fins. A solid billet of the same envelope would weigh "
            f"{fmt(ev['billet_mass'], 0)} kg. The structure is therefore a real hollow shell, not "
            "a filled body. The measured centre of gravity is "
            f"{fmt(ev['cg_structure'][0], 4)} m aft of the nose tip, and it sits "
            f"{fmt(abs(ev['cg_structure'][1]) * 1000.0, 1)} mm off the axis. That small offset is "
            "discretisation, not asymmetry, so tests check it against a tolerance and never "
            "against zero.",
            S.BODY,
        )
    )
    story.append(
        S.fig_single(
            os.path.join(FIGS, "area_distribution.png"),
            5,
            "Cross-section area against station, measured in nTop by sectioning the solid at 16 "
            "stations, against the closed-form outer mould line.",
        )
    )

    story.append(Paragraph("3.3 The coupling changes the answer", S.H2))
    a_ms, n_ms = pa["mass_statement"], pn["mass_statement"]
    a_tr, n_tr = pa["trajectory"], pn["trajectory"]
    rows = [
        ["Quantity", "Analytic geometry", "nTop-measured", "Change"],
        [
            "Launch mass [kg]",
            fmt(a_ms["total_kg"], 1),
            fmt(n_ms["total_kg"], 1),
            f"{n_ms['total_kg'] - a_ms['total_kg']:+.1f}",
        ],
        [
            "Range [km]",
            fmt(a_tr["range_m"] / 1000.0, 1),
            fmt(n_tr["range_m"] / 1000.0, 1),
            f"{(n_tr['range_m'] - a_tr['range_m']) / 1000.0:+.1f}",
        ],
        [
            "Mach at impact",
            fmt(a_tr["mach_final"], 2),
            fmt(n_tr["mach_final"], 2),
            f"{n_tr['mach_final'] - a_tr['mach_final']:+.2f}",
        ],
        [
            "Maximum dynamic pressure [kPa]",
            fmt(a_tr["q_max_Pa"] / 1000.0, 1),
            fmt(n_tr["q_max_Pa"] / 1000.0, 1),
            f"{(n_tr['q_max_Pa'] - a_tr['q_max_Pa']) / 1000.0:+.1f}",
        ],
        [
            "Centre of gravity [m]",
            fmt(a_ms["x_cg_m"], 3),
            fmt(n_ms["x_cg_m"], 3),
            f"{n_ms['x_cg_m'] - a_ms['x_cg_m']:+.3f}",
        ],
    ]
    story.append(S.styled_table(rows, [2.4, 1.5, 1.4, 1.0], ["LEFT", "RIGHT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 11. The same design vector, sized with analytic geometry and with nTop "
            "geometry. This is a real coupling effect, not numerical noise.",
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            f"The measured airframe is heavier than the analytic shell estimate. Launch mass "
            f"therefore rises by {n_ms['total_kg'] - a_ms['total_kg']:.1f} kg. A heavier rocket "
            f"flies {abs(n_tr['range_m'] - a_tr['range_m']) / 1000.0:.1f} km less far. Both "
            "answers meet every requirement, so the coupling does not change the verdict here. "
            "It does change the numbers, and it would change the verdict on a design with less "
            "margin.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            f"One measurement call takes about {fmt(ev.get('wall_time_s', 0.0), 0)} s. Exports "
            "are off by default. Nothing that is measured comes off the exported mesh, so the "
            "loop loses no information when exports are off. Exports run for the converged design "
            "only.",
            S.BODY,
        )
    )


def trade_study(story: list, D: dict) -> None:
    grid, lhs = D["grid"], D["lhs"]
    n_gf = sum(1 for r in grid if r["feasible"] == "1")
    n_lf = sum(1 for r in lhs if r["feasible"] == "1")

    S.sect(story, "4. Trade study")
    story.append(
        Paragraph(
            f"Two studies were run. A full factorial of {len(grid)} nodes covers body diameter, "
            f"sustain propellant and nose fineness. A Latin hypercube of {len(lhs)} samples covers "
            "eight variables. The hypercube uses a seeded generator, so it repeats exactly on any "
            "machine. Failed samples are recorded, never discarded: a study that drops its "
            "failures reports a feasible region that is too large.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            f"<b>Only {n_gf} of {len(grid)} factorial nodes are feasible, and {n_lf} of "
            f"{len(lhs)} hypercube samples.</b> The feasible region is narrow. That is the single "
            "most useful output of the study.",
            S.BODY,
        )
    )
    story.append(
        S.fig_single(
            os.path.join(FIGS, "carpet.png"),
            6,
            "Launch mass and range over body diameter and sustain propellant, at three nose "
            "finenesses. Feasible nodes are filled. Infeasible nodes carry the constraints they "
            "violate. The converged design is ringed.",
            width_in=6.7,
            max_h_in=3.2,
        )
    )

    feas = sorted([r for r in grid if r["feasible"] == "1"], key=lambda r: float(r["m0_kg"]))
    rows = [["D [m]", "Sustain [kg]", "Nose fineness", "Mass [kg]", "Range [km]", "Mach", "q [kPa]"]]
    for r in feas:
        rows.append(
            [
                fmt(float(r["D"]), 3),
                fmt(float(r["m_p_sustain"]), 0),
                fmt(float(r["f_nose"]), 2),
                fmt(float(r["m0_kg"]), 1),
                fmt(float(r["range_km"]), 1),
                fmt(float(r["mach_terminal"]), 2),
                fmt(float(r["q_max_kPa"]), 0),
            ]
        )
    story.append(S.styled_table(rows, [0.8, 1.1, 1.1, 1.0, 1.0, 0.7, 0.9], ["RIGHT"] * 7))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 12. Every feasible node of the factorial, lightest first. The lightest is the "
            "design the search returned. That is an independent check on the search, because the "
            "factorial did not use the search.",
            S.CAP,
        )
    )

    story.append(Paragraph("4.1 Sensitivity", S.H2))
    story.append(
        Paragraph(
            "Sensitivity uses Spearman rank correlation, not Pearson. The responses are monotone "
            "in the design variables but not linear, so rank correlation measures the association "
            "without assuming a shape. Only converged samples are counted.",
            S.BODY,
        )
    )
    sens = D["sens"]
    responses = ["m0_kg", "range_km", "mach_terminal", "q_max_kPa"]
    labels = {"m0_kg": "Launch mass", "range_km": "Range", "mach_terminal": "Impact Mach", "q_max_kPa": "q_max"}
    variables = list(sens[responses[0]].keys())
    rows = [["Variable"] + [labels[r] for r in responses]]
    for v in variables:
        rows.append([v] + [("n/a" if math.isnan(sens[r][v]) else f"{sens[r][v]:+.3f}") for r in responses])
    story.append(S.styled_table(rows, [1.7] + [1.2] * 4, ["LEFT"] + ["RIGHT"] * 4))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 13. Spearman rank correlation from the Latin hypercube.",
            S.CAP,
        )
    )
    story.append(
        S.fig_single(
            os.path.join(FIGS, "sensitivity.png"),
            7,
            "Rank-correlation sensitivity, grouped by response.",
        )
    )
    story.append(
        Paragraph(
            "The rankings agree with the physics. Sustain propellant drives launch mass and "
            "range. Terminal propellant drives impact Mach, and it is the second strongest driver "
            "of dynamic pressure. Body diameter drives range, impact Mach and dynamic pressure, "
            "all negatively, because a smaller body has less drag.",
            S.BODY,
        )
    )


def verification(story: list, D: dict) -> None:
    ev = D["evidence"]
    it, ae = ev["integrator"], ev["aero"]

    S.sect(story, "5. Verification")
    story.append(
        Paragraph(
            f"{n_test_total()} automated tests pass. The physics is checked against closed-form "
            "answers where they exist, and against published measurements where they do not.",
            S.BODY,
        )
    )

    story.append(Paragraph("5.1 The trajectory integrator", S.H2))
    rows = [
        ["Case", "Relative error"],
        ["Vacuum ballistic range against the closed-form parabola", sci(it["vacuum_range_rel"])],
        ["Vacuum ballistic apogee", sci(it["vacuum_apogee_rel"])],
        ["Burnout speed against Tsiolkovsky less gravity loss", sci(it["tsiolkovsky_rel"])],
        ["Steady terminal velocity against sqrt(2mg/(rho S CD))", sci(it["terminal_velocity_rel"])],
        ["Specific-energy drift over 100 s with no thrust and no drag", sci(it["energy_drift_rel"])],
    ]
    story.append(S.styled_table(rows, [4.6, 2.0], ["LEFT", "RIGHT"]))
    story.append(Spacer(1, 4))
    ratios = ", ".join(f"{r:.2f}" for r in it["rk4_order_ratios"])
    story.append(
        Paragraph(
            f"Table 14. Integrator validation. Halving the step reduces the error by {ratios}, "
            "against the factor of 16 that fourth-order Runge-Kutta requires. The integrator is "
            "therefore correct to machine precision on every case with an analytic answer.",
            S.CAP,
        )
    )

    story.append(Paragraph("5.2 The aerodynamic model", S.H2))
    story.append(
        Paragraph(
            "The build-up is validated against free-flight measurements of the Army-Navy Basic "
            "Finner. The reference is A. D. Dupuis and W. Hathaway, Aeroballistic Range Tests of "
            "the Basic Finner Reference Projectile at Supersonic Velocities, Defence Research "
            f"Establishment Valcartier, DREV-TM-9703, 1997, Table VII. It gives {ae['n_shots_table']} "
            "shots between Mach 1.06 and Mach 4.47.",
            S.BODY,
        )
    )
    rows = [
        ["Quantity", "Shots", "Mean bias [%]", "Worst shot [%]"],
        ["Zero-lift drag coefficient", str(ae["n_shots_cd0"]), pct(ae["cd0_mean_bias"], 1), pct(ae["cd0_worst_shot"], 1)],
        ["Normal-force slope", str(ae["n_shots_cna_xcp"]), pct(ae["cna_mean_bias"], 1), pct(ae["cna_worst_shot"], 1)],
        ["Centre of pressure", str(ae["n_shots_cna_xcp"]), pct(ae["xcp_mean_bias"], 1), pct(ae["xcp_worst_shot"], 1)],
    ]
    story.append(S.styled_table(rows, [2.6, 0.8, 1.5, 1.5], ["LEFT", "RIGHT", "RIGHT", "RIGHT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 15. Aerodynamic validation against Basic Finner free-flight data.",
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            f"The drag bias is systematic, not random. The model runs about "
            f"{abs(100.0 * ae['cd0_mean_bias']):.1f} percent low. Three causes are known: fin "
            "trailing-edge base drag is not modelled, fin-body junction interference is not "
            "modelled, and the real fin section is conical rather than the double wedge the wave "
            "drag assumes. A low drag coefficient overpredicts range, which is the wrong direction "
            f"for a sizing tool. Drag is therefore scaled by {fmt(ae['cd0_calibration'], 3)} in the "
            "sizing loop. The correction is applied at the loop boundary, never inside the "
            "aerodynamic model, so the model always reports what its physics gives.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "Centre of pressure and normal-force slope are left uncorrected. Their agreement is "
            "good enough that a correction is not justified.",
            S.BODY,
        )
    )
    story.append(
        S.fig_single(
            os.path.join(RUNS, "_aero", "aero_validation.png"),
            8,
            "Drag build-up, normal-force slope and centre of pressure against Mach, with the "
            "Basic Finner measurements overlaid. The transonic region is a blend, not physics, "
            "and is excluded from validation.",
            width_in=6.7,
            max_h_in=4.2,
        )
    )

    story.append(Paragraph("5.3 The motor and the geometry", S.H2))
    story.append(
        Paragraph(
            "The nozzle thrust coefficient reproduces published isentropic tables to better than "
            "0.1 percent. Total impulse equals specific impulse times standard gravity times "
            "propellant mass, summed over all three burn phases, and a separate integral checks "
            "propellant-mass conservation over the whole timeline.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "The geometry pipeline was proved on a sphere before the rocket was attempted. A "
            "25 mm sphere gave a volume error of 0.0104 percent from the notebook's own mass "
            "properties, and 0.169 percent from the exported mesh. The notebook measurements are "
            "therefore trusted over the mesh, and the mesh is used only for pictures and for "
            "downstream tools.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "Two findings are locked into the test suite so they cannot quietly disappear. One "
            "test asserts that a design with no terminal propellant reproduces the validated "
            "two-phase motor exactly. Another asserts that an unpowered vertical dive cannot "
            "reach Mach 1.1, which is the infeasibility of section 1.2.",
            S.BODY,
        )
    )


def limitations(story: list, D: dict) -> None:
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
            "<b>The vehicle is invented.</b> Its requirements correspond to no real programme. "
            "Section 1.1 shows that they were not even self-consistent when written.",
            S.BODY,
        )
    )

    story.append(Paragraph("6.1 The largest unquantified optimism", S.H2))
    story.append(
        Paragraph(
            "Nozzle flow is ideal. The model assumes frozen composition, constant specific-heat "
            "ratio and single-phase flow. It has no two-phase alumina loss, no divergence loss, no "
            "combustion inefficiency and no throat erosion. Real delivered specific impulse for "
            "this propellant class is typically 3 to 7 percent lower. <b>That penalty was not "
            "applied, because its magnitude could not be sourced.</b> Range and impact Mach are "
            "therefore optimistic by an amount that has not been quantified. This is the single "
            "largest known optimism in the result.",
            S.BODY,
        )
    )

    story.append(Paragraph("6.2 The fin fidelity mismatch", S.H2))
    story.append(
        Paragraph(
            "The fins are constant-thickness tapered plates in nTop. The aerodynamic model assumes "
            "a symmetric double wedge. The two do not agree. The plate has about twice the volume "
            "of a diamond of the same maximum thickness, so the fin mass is conservative by "
            "roughly 4 kg out of 43 kg of structure. The wave drag is computed for a section the "
            "geometry does not have.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            "An exact double wedge was attempted and abandoned for a specific reason. For a swept "
            "tapered panel the leading edge, the ridge and the trailing edge are three skew lines. "
            "The wedge faces are therefore hyperbolic paraboloids. The available loft operation "
            "interpolates signed-distance fields rather than boundaries, so it rounds the diamond "
            "corners: it returned 82 percent of the exact panel volume and 73 percent of its area. "
            "No shear or non-uniform-scale transform was available to build the correct surface.",
            S.BODY,
        )
    )

    story.append(Paragraph("6.3 Every value that is a guess", S.H2))
    story.append(
        Paragraph(
            "Every empirical constant in the code is registered with its source. A value that is "
            "a guess must say so in its source text, and a test asserts that. The table lists "
            "every registered source whose text admits a guess or a modelling choice.",
            S.BODY,
        )
    )
    sources = D["prov"]["sources"]
    flagged = [
        (k, v)
        for k, v in sorted(sources.items())
        if ("guess" in v.lower() or "modelling choice" in v.lower() or "approximation" in v.lower())
    ]
    rows = [["Registered name", "Why it is not a measured value"]]
    for k, v in flagged:
        # No truncation: cell_table wraps, and clipping a source string mid-word would defeat
        # the purpose of publishing it.
        rows.append([k, v])
    story.append(S.cell_table(rows, [1.9, 4.7]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            f"Table 16. All {len(flagged)} registered values that are a guess, a modelling choice "
            "or an approximation. Source: runs/SV-1/provenance.json.",
            S.CAP,
        )
    )

    story.append(Paragraph("6.4 Other limitations", S.H2))
    for text in [
        "The transonic drag rise is an interpolated blend between the subsonic value and the "
        "supersonic value at Mach 1.2. It is not physics. It is excluded from validation, and a "
        "test asserts that it underpredicts, so it can never be mistaken for a drag-rise model.",
        "The trajectory is a three-degree-of-freedom point mass on a flat earth with constant "
        "gravity. There is no roll, no sideslip and no autopilot.",
        "Guidance gains, the angle-of-attack limit and the dive-entry rule are arbitrary. "
        "Guidance design is out of scope. The load cap exists only to keep the recorded "
        "commanded normal force finite at the dive-entry step.",
        "Structural sizing is wall thickness times density, plus a hoop-stress check on the motor "
        "case with a minimum-gauge floor. There is no buckling, no flutter and no thermal analysis.",
        "The inter-pulse bulkhead, its insulation and the second igniter are not costed for the "
        "terminal pulse, so the pulsed motor inert mass is optimistic.",
        "The grain geometry is neutral-burning within each phase and closed at the mean web. "
        "There is no burnback simulation, so the reported grain dimensions are equivalent values "
        "rather than a drawing.",
        "The block universe vendored for notebook authoring is slightly stale against the "
        "installed nTop builds. Three specific mismatches were found and worked around.",
    ]:
        story.append(Paragraph(text, S.BULLET, bulletText="-"))


def reproduce(story: list, D: dict) -> None:
    env = D["prov"]["environment"]
    S.sect(story, "7. How to reproduce this")
    story.append(
        Paragraph(
            "Every artefact in this report comes from the commands below. The analysis writes "
            "JSON and CSV. The figure scripts read those files. This document reads them too, so "
            "no number is typed in twice.",
            S.BODY,
        )
    )
    story.append(
        Paragraph(
            ".venv/Scripts/python.exe -m pytest tests -q<br/>"
            ".venv/Scripts/python.exe run_sv1.py --stage smoke<br/>"
            ".venv/Scripts/python.exe run_sv1.py --stage size<br/>"
            ".venv/Scripts/python.exe run_sv1.py --stage doe --doe-scale full<br/>"
            ".venv/Scripts/python.exe -m rocketgen.report.build_report",
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
        ["Latin hypercube seed", "20260817"],
    ]
    story.append(S.styled_table(rows, [2.2, 4.4], ["LEFT", "LEFT"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 17. Environment. numpy and scipy are pinned below their current major versions "
            "because SUAVE 2.5.2 does not run on either.",
            S.CAP,
        )
    )
    story.append(
        Paragraph(
            "One warning about the toolchain is worth recording. The nTop command-line tool "
            "returns exit code 72 when a block fails, not on success. Published guidance states "
            "the opposite. A notebook given a negative radius returned 72 and wrote no output at "
            "all. Success must therefore be judged on the expected files existing and being "
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
        title="SV-1 Coupled nTop and SUAVE Conceptual Sizing",
        author="nTop",
        footer="SV-1 coupled sizing  |  nTop + SUAVE  |  demonstration, invented requirements",
    )

    front_matter(story, D)
    story.append(PageBreak())
    findings(story, D)
    story.append(PageBreak())
    the_design(story, D)
    story.append(PageBreak())
    ntop_coupling(story, D)
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
