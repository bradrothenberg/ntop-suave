"""Figures for the IV-1 SPLINE report: the same interceptor with two outer mould lines.

    .venv/Scripts/python.exe -m rocketgen.report.fig_iv1_spline

Writes four PNGs into `runs/IV-1_spline/figures/`:

    iv1_spline_ascent.png     the two flown missions, side by side
    iv1_spline_pitchover.png  the pitchover sweep, which is where the shape earns its keep
    iv1_spline_oml.png        the revolved spline, its validation, and the drag it moves
    iv1_spline_margins.png    the fifteen constraints, both shapes

Every number is read from `runs/IV-1_spline/figures/evidence_iv1_spline.json`. Nothing is
re-flown and nTop is not called. Build that file first with:

    .venv/Scripts/python.exe -m rocketgen.report.evidence_iv1_spline
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt                                        # noqa: E402
import numpy as np                                                     # noqa: E402

from ..config import RUNS_DIR                                          # noqa: E402
from .figstyle import ACCENT, BAD, COOL, GOOD, GREY, INK, STYLE, WARN  # noqa: E402

CASE_DIR = os.path.join(RUNS_DIR, "IV-1_spline")
FIG_DIR = os.path.join(CASE_DIR, "figures")
EVIDENCE = os.path.join(FIG_DIR, "evidence_iv1_spline.json")

#: One colour per outer-mould-line family, used by every panel in this module.
SHAPE_COLOUR = {"ogive": INK, "spline": ACCENT}
SHAPE_LABEL = {"ogive": "tangent ogive", "spline": "revolved spline"}



def evidence() -> dict[str, Any]:
    if not os.path.isfile(EVIDENCE):
        raise SystemExit(
            f"{EVIDENCE} is missing. Run:\n"
            "    .venv/Scripts/python.exe -m rocketgen.report.evidence_iv1_spline"
        )
    with open(EVIDENCE, encoding="utf-8") as f:
        return json.load(f)


def out_path(name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, name)


def _requirement(ev: dict[str, Any], name: str) -> float | None:
    """A constraint limit, read from the recorded constraint list rather than typed in."""
    for c in ev["constraints"]["spline"]:
        if c["name"] == name:
            return float(c["limit"])
    return None


# --------------------------------------------------------------------------------------
#   1. The two flown missions
# --------------------------------------------------------------------------------------


def ascent(path: str | None = None) -> str:
    ev = evidence()
    traj = ev["trajectory"]
    h_min = _requirement(ev, "A3 intercept alt [m]")
    m_min = _requirement(ev, "A4 intercept Mach")
    q_lim = _requirement(ev, "A10 q_max [Pa]")

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.4))
        ax_a, ax_b = axes[0]
        ax_c, ax_d = axes[1]

        for shape in ("ogive", "spline"):
            t = traj[shape]
            colour = SHAPE_COLOUR[shape]
            ic = t["intercept"]
            ax_a.plot([v / 1000.0 for v in t["x"]], [v / 1000.0 for v in t["h"]],
                      color=colour, lw=1.5, label=SHAPE_LABEL[shape])
            ax_a.plot([ic["slant_range"] / 1000.0], [ic["altitude"] / 1000.0], "o",
                      ms=6.5, mfc="none", mec=colour, mew=1.6)
            ax_a.text(0.985, 0.50 if shape == "spline" else 0.27,
                      "%s: intercept at %.2f km" % (SHAPE_LABEL[shape],
                                                    ic["altitude"] / 1000.0),
                      transform=ax_a.transAxes, ha="right", fontsize=6.5, color=colour)
            ax_b.plot(t["time"], t["mach"], color=colour, lw=1.4, label=SHAPE_LABEL[shape])
            ax_c.plot(t["time"], [v / 1000.0 for v in t["q"]], color=colour, lw=1.4,
                      label=SHAPE_LABEL[shape])
            ax_d.plot(t["time"], t["mass"], color=colour, lw=1.4, label=SHAPE_LABEL[shape])

        if h_min is not None:
            ax_a.axhline(h_min / 1000.0, color=BAD, lw=0.9, ls=":",
                         label="A3 minimum intercept altitude %.0f km" % (h_min / 1000.0))
        ax_a.set_xlabel("ground range [km]")
        ax_a.set_ylabel("altitude [km]")
        ax_a.set_title("(a) the two missions, both to the same 100 mile slant range",
                       loc="left", fontsize=8.4)
        ax_a.legend(loc="upper left", fontsize=6.4, handlelength=1.8)

        if m_min is not None:
            ax_b.axhline(m_min, color=BAD, lw=0.9, ls=":",
                         label="A4 minimum intercept Mach %.1f" % m_min)
        ax_b.set_xlabel("time [s]")
        ax_b.set_ylabel("Mach")
        ax_b.set_title("(b) Mach", loc="left", fontsize=8.4)
        ax_b.legend(loc="lower right", fontsize=6.4, handlelength=1.8)

        if q_lim is not None:
            ax_c.axhline(q_lim / 1000.0, color=BAD, lw=0.9, ls=":",
                         label="A10 limit %.0f kPa" % (q_lim / 1000.0))
        ax_c.set_xlabel("time [s]")
        ax_c.set_ylabel("dynamic pressure [kPa]")
        ax_c.set_title("(c) dynamic pressure", loc="left", fontsize=8.4)
        ax_c.legend(loc="upper right", fontsize=6.4, handlelength=1.8)

        ax_d.set_xlabel("time [s]")
        ax_d.set_ylabel("stack mass [kg]")
        ax_d.set_title("(d) mass, with the stage-1 jettison visible", loc="left", fontsize=8.4)
        ax_d.legend(loc="upper right", fontsize=6.4, handlelength=1.8)

        fig.text(
            0.012, 0.985,
            "The same interceptor flown twice. Only the nose and interstage shape family "
            "differs. Both fly the same 38 degree pitchover.\n"
            "Source: runs/IV-1_spline/figures/evidence_iv1_spline.json.",
            fontsize=7.0, color=GREY, va="top", ha="left",
        )
        fig.subplots_adjust(left=0.075, right=0.985, top=0.885, bottom=0.105,
                            hspace=0.52, wspace=0.24)
        path = path or out_path("iv1_spline_ascent.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
#   2. The pitchover sweep
# --------------------------------------------------------------------------------------


def pitchover(path: str | None = None) -> str:
    ev = evidence()
    sweep = ev["pitchover"]
    gammas = sweep["gamma_deg"]
    h_min = _requirement(ev, "A3 intercept alt [m]")
    g_min = _requirement(ev, "A11 lateral g")
    range_min = _requirement(ev, "A2 slant range [m]")
    comp = {r["quantity"]: r for r in ev["comparison"]["rows"]}

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.4))
        ax_a, ax_b = axes[0]
        ax_c, ax_d = axes[1]

        for shape in ("ogive", "spline"):
            rows = sweep["shapes"][shape]
            colour = SHAPE_COLOUR[shape]
            ax_a.plot(gammas, [r["altitude_km"] for r in rows], "-o", color=colour, ms=4.0,
                      label=SHAPE_LABEL[shape])
            ax_b.plot(gammas, [r["lateral_g_aero"] for r in rows], "-o", color=colour, ms=4.0,
                      label=SHAPE_LABEL[shape])
            ax_c.plot(gammas, [r["slant_range_km"] for r in rows], "-o", color=colour, ms=4.0,
                      label=SHAPE_LABEL[shape])
            # A filled marker marks a feasible angle, so the reader can count them.
            for r in rows:
                if r["feasible"]:
                    ax_a.plot([r["gamma_deg"]], [r["altitude_km"]], "o", ms=8.5, mfc="none",
                              mec=GOOD, mew=1.6)

        if h_min is not None:
            ax_a.axhline(h_min / 1000.0, color=BAD, lw=1.0, ls=":",
                         label="A3 limit %.0f km" % (h_min / 1000.0))
        ax_a.set_xlabel("commanded pitchover angle [deg]")
        ax_a.set_ylabel("intercept altitude [km]")
        ax_a.set_title("(a) the spline clears A3 two degrees earlier", loc="left", fontsize=8.4)
        ax_a.legend(loc="upper left", fontsize=6.4, handlelength=1.8)
        ax_a.annotate("green ring: every constraint met", (0.97, 0.06),
                      xycoords="axes fraction", ha="right", fontsize=6.4, color=GOOD)

        if g_min is not None:
            ax_b.axhline(g_min, color=BAD, lw=1.0, ls=":",
                         label="A11 requirement %.0f g" % g_min)
        ax_b.set_xlabel("commanded pitchover angle [deg]")
        ax_b.set_ylabel("aerodynamic lateral acceleration [g]")
        ax_b.set_title("(b) a higher intercept costs aerodynamic g", loc="left", fontsize=8.4)
        ax_b.legend(loc="upper right", fontsize=6.4, handlelength=1.8)

        if range_min is not None:
            ax_c.axhline(range_min / 1000.0, color=BAD, lw=1.0, ls=":",
                         label="A2 limit %.1f km" % (range_min / 1000.0))
        ax_c.set_xlabel("commanded pitchover angle [deg]")
        ax_c.set_ylabel("slant range at intercept [km]")
        ax_c.set_title("(c) slant range, held on the requirement", loc="left", fontsize=8.4)
        ax_c.legend(loc="lower right", fontsize=6.4, handlelength=1.8)

        # ---- (d) how A11 is actually met, at the converged point ---------------------
        # Grouped, never stacked. `config_iv1.lateral_g_total` takes the GREATER of the two,
        # not their sum, because commanding an aerodynamic turn and a divert at once is a
        # control problem this model does not represent.
        aero = [comp["Lateral g, aerodynamic"]["ogive"], comp["Lateral g, aerodynamic"]["spline"]]
        acs = [comp["Lateral g, attitude control"]["ogive"],
               comp["Lateral g, attitude control"]["spline"]]
        y = np.arange(2)
        ax_d.barh(y + 0.19, aero, height=0.34, color=COOL, label="aerodynamic")
        ax_d.barh(y - 0.19, acs, height=0.34, color=WARN, label="attitude control")
        if g_min is not None:
            ax_d.axvline(g_min, color=BAD, lw=1.0, ls=":",
                         label="A11 requirement %.0f g" % g_min)
        for yi, (a, s) in enumerate(zip(aero, acs)):
            ax_d.text(a + 0.35, yi + 0.19, "%.2f g" % a, va="center", fontsize=6.6, color=INK)
            ax_d.text(s + 0.35, yi - 0.19, "%.2f g" % s, va="center", fontsize=6.6, color=INK)
        ax_d.set_yticks(y)
        ax_d.set_yticklabels([SHAPE_LABEL["ogive"], SHAPE_LABEL["spline"]], fontsize=7.0)
        ax_d.set_xlabel("lateral acceleration at intercept [g]")
        ax_d.set_xlim(0.0, max(max(aero), max(acs)) * 1.30)
        ax_d.grid(axis="y", visible=False)
        ax_d.set_title("(d) the trade: A11 is now met by the thruster alone",
                       loc="left", fontsize=8.4)
        ax_d.legend(loc="center right", fontsize=6.4, handlelength=1.4)

        fig.text(
            0.012, 0.985,
            "The pitchover sweep, with the geometry measured. The pitchover angle is not a "
            "geometry input, so one measurement set serves every angle.\n"
            "Source: runs/IV-1_spline/figures/evidence_iv1_spline.json, section pitchover.",
            fontsize=7.0, color=GREY, va="top", ha="left",
        )
        fig.subplots_adjust(left=0.105, right=0.985, top=0.885, bottom=0.105,
                            hspace=0.52, wspace=0.30)
        path = path or out_path("iv1_spline_pitchover.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
#   3. The outer mould line, its validation and its effect
# --------------------------------------------------------------------------------------


def _tangent_ogive(length: float, radius: float, n: int = 401):
    rho = (radius * radius + length * length) / (2.0 * radius)
    xs = np.linspace(0.0, length, n)
    rs = np.array([
        max(math.sqrt(max(rho * rho - (length - x) ** 2, 0.0)) - (rho - radius), 0.0)
        for x in xs
    ])
    return xs, rs


def oml(path: str | None = None) -> str:
    from ..oml_spline import SplineProfile

    ev = evidence()
    w = ev["wavedrag"]
    v = w["validation"]
    geom = ev["geometry"]

    # The stage-2 nose, taken from the recorded design vector so the drawing cannot drift
    # from the geometry that was measured.
    dv = ev["design_vector"]["spline"]
    length = float(dv["L_nose"])
    radius = length / float(w["k_L_over_R"])
    profile = SplineProfile(length=length, radius=radius, control=list(w["nose_control"]))

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(9.4, 3.6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.05])
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])

        # ---- (a) the profile nTop revolves --------------------------------------------
        xo, ro = _tangent_ogive(length, radius)
        ax_a.plot(xo, ro, color=INK, lw=1.5, ls="--", label="tangent ogive baseline")
        pts = np.array([profile.point_at(float(u)) for u in np.linspace(0.0, 1.0, 401)])
        ctrl = np.array(profile.control_points())
        ax_a.plot(pts[:, 0], pts[:, 1], color=ACCENT, lw=1.5, label="revolved B-spline")
        ax_a.plot(ctrl[:, 0], ctrl[:, 1], "o", ms=3.4, mfc="none", mec=ACCENT, mew=1.0,
                  label="%d control points" % len(ctrl))
        ax_a.set_xlabel("station from the nose tip [m]")
        ax_a.set_ylabel("radius [m]")
        ax_a.set_title("(a) the stage-2 nose", loc="left", fontsize=8.4)
        ax_a.legend(loc="lower right", fontsize=6.3, handlelength=1.8)
        ax_a.set_xlim(0.0, length * 1.02)
        ax_a.set_ylim(0.0, radius * 1.25)
        ax_a.annotate(
            "enclosed volume %+.2f %%\nwetted area %+.2f %%"
            % (geom["nose_volume_change_pct"], geom["nose_wetted_change_pct"]),
            (0.03, 0.96), xycoords="axes fraction", va="top", fontsize=6.3, color=GREY,
        )

        # ---- (b) the closed-form validation residuals ---------------------------------
        checks = [
            ("Sears-Haack D/q", abs(v["sears_haack"]["rel_err"]), 1.0e-4),
            ("von Karman D/q", abs(v["von_karman"]["rel_err"]), 1.0e-4),
            ("von Karman C_D on base", abs(v["von_karman"]["cd_rel_err"]), 1.0e-4),
            ("shape factor = 4/pi", abs(v["von_karman_shape_factor"]["rel_err"]), 1.0e-4),
        ]
        y = np.arange(len(checks))
        vals = [max(c[1], 1.0e-12) for c in checks]
        ax_b.barh(y, vals, height=0.5, color=COOL, edgecolor="none")
        ax_b.plot([c[2] for c in checks], y, "|", ms=12.0, mew=1.6, color=BAD,
                  label="tolerance asserted")
        for yi, val in zip(y, vals):
            ax_b.text(val * 1.35, yi, "%.1e" % val, va="center", fontsize=6.2, color="#3a3a3a")
        ax_b.set_yticks(y)
        ax_b.set_yticklabels([c[0] for c in checks], fontsize=6.8)
        ax_b.set_xscale("log")
        ax_b.set_xlim(1.0e-6, 3.0e-3)
        ax_b.set_xlabel("relative residual")
        ax_b.grid(axis="y", visible=False)
        ax_b.set_title("(b) against exact closed forms", loc="left", fontsize=8.4)
        ax_b.legend(loc="lower right", fontsize=6.3, handlelength=1.0)

        # ---- (c) the drag it moves ----------------------------------------------------
        for stage, marker in ((1, "-o"), (2, "-s")):
            rows = [r for r in w["drag_rows"] if r["stage"] == stage]
            mach = [r["mach"] for r in rows]
            ax_c.plot(mach, [r["cd0_ogive"] for r in rows], marker, color=INK, ms=3.4,
                      label="CD0 ogive, stage %d" % stage)
            ax_c.plot(mach, [r["cd0_spline"] for r in rows], marker, color=ACCENT, ms=3.4,
                      label="CD0 spline, stage %d" % stage)
        ax_c.set_xlabel("Mach")
        ax_c.set_ylabel("CD0 on the stage reference area")
        ax_c.set_title("(c) zero-lift drag, both stages", loc="left", fontsize=8.4)
        ax_c.legend(loc="upper right", fontsize=6.0, handlelength=1.8, ncol=2)
        ax_c.set_ylim(0.0, max(r["cd0_ogive"] for r in w["drag_rows"]) * 1.55)
        ax_c.annotate("forebody wave-term\nshape ratio %.4f" % w["shape_ratio"],
                      (0.97, 0.06), xycoords="axes fraction", ha="right", fontsize=6.3,
                      color=GREY)

        fig.text(
            0.012, 0.982,
            "nTop revolves the spline itself: no chord polygon, no discretisation error. "
            "The shape ratio multiplies the calibrated wave-drag term only.\n"
            "Source: runs/IV-1_spline/figures/evidence_iv1_spline.json, section wavedrag.",
            fontsize=7.0, color=GREY, va="top", ha="left",
        )
        fig.subplots_adjust(left=0.072, right=0.985, top=0.845, bottom=0.135, wspace=0.58)
        path = path or out_path("iv1_spline_oml.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
#   4. The constraint margins
# --------------------------------------------------------------------------------------


def margins(path: str | None = None) -> str:
    ev = evidence()
    spline = ev["constraints"]["spline"]
    ogive = {c["name"]: c for c in ev["constraints"]["ogive"]}

    rows = sorted(spline, key=lambda c: c["margin"])
    names = [c["name"] for c in rows]
    y = np.arange(len(rows))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9.4, 4.4))
        for i, c in enumerate(rows):
            colour = GOOD if c["met"] else BAD
            ax.barh(i, c["margin"], height=0.55, color=colour, edgecolor="none")
            o = ogive.get(c["name"])
            if o is not None:
                ax.plot([o["margin"]], [i], "|", ms=11.0, mew=1.6, color=WARN)
            ax.text(max(c["margin"], 0.0) + 0.03, i,
                    "%+.1f %%   (%.4g %s %.4g)"
                    % (100.0 * c["margin"], c["value"], c["sense"], c["limit"]),
                    va="center", fontsize=6.6, color="#3a3a3a")
        ax.plot([], [], "|", ms=11.0, mew=1.6, color=WARN, label="tangent ogive baseline")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7.0)
        ax.set_xlabel("normalised margin, (value - limit) / |limit| in the meeting direction")
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_xlim(-0.05, 60.0)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", fontsize=7.0, handlelength=1.2)
        ax.set_title(
            "Constraint margins, converged IV-1 spline: %d of %d constraints met"
            % (ev["comparison"]["n_met_spline"], ev["comparison"]["n_constraints"]),
            loc="left", fontsize=9.5,
        )
        fig.text(
            0.012, 0.985,
            "Bars are the spline result. Orange ticks are the tangent-ogive baseline at the "
            "same pitchover angle. Both meet all fifteen.\n"
            "Source: runs/IV-1_spline/figures/evidence_iv1_spline.json, section constraints.",
            fontsize=7.0, color=GREY, va="top", ha="left",
        )
        fig.subplots_adjust(left=0.185, right=0.985, top=0.845, bottom=0.115)
        path = path or out_path("iv1_spline_margins.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


def make_all() -> list[str]:
    return [ascent(), pitchover(), oml(), margins()]


if __name__ == "__main__":
    for _p in make_all():
        print(_p)
