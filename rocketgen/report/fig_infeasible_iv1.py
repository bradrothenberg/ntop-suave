"""IV-1 figure: why A2, A3 and A11 could not hold together.

This is the centrepiece of the report. It draws the requirements audit of SPEC_IV1.md section 2.

Source data: `runs/IV-1/figures/evidence_iv1.json`, key `audit`. That record is produced by the
same walk `scripts/iv1_envelope_probe.py` does, and the evidence script asserts the two agree
field by field before writing anything.

The two panels answer two halves of one question.

(a) Where the two requirements live, in altitude against Mach. The lower region is where 15 g of
    AERODYNAMIC lateral acceleration is available. The upper region is where A3 puts the
    intercept. Their overlap is empty below Mach 4.45 and a few hundred metres tall above it. The
    markers show where each lofted trajectory actually reaches 100 miles of slant range: every
    one of them is far above the region where the vehicle could manoeuvre.

(b) What that costs in range. For each pitchover angle, the furthest point at which A3, A4 and
    A11 all hold at once, against the 160.9 km A2 asks for. The best is 58.3 km.

Run:
    .venv/Scripts/python.exe -m rocketgen.report.fig_infeasible_iv1
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .figstyle import BAD, COOL, GOOD, GREY, INK, STYLE
from .figstyle_iv1 import evidence, out_path

MILE = 1609.344


def make_figure(path: str | None = None) -> str:
    ev = evidence()
    a = ev["audit"]
    fine = a["ceiling_fine"]
    mach = np.array([r["mach"] for r in fine])
    ceiling_km = np.array([r["h_limit_m"] for r in fine]) / 1000.0
    h_min_km = a["h_intercept_min_m"] / 1000.0
    top_km = 30.0

    # Where the two regions first overlap: the lowest Mach at which the 15 g ceiling reaches A3.
    over = [r for r in fine if r["h_limit_m"] >= a["h_intercept_min_m"]]
    m_overlap = over[0]["mach"] if over else None

    with plt.rc_context(STYLE):
        fig, (ax, ax2) = plt.subplots(
            1, 2, figsize=(11.2, 4.9), gridspec_kw={"width_ratios": [1.15, 1.0]}
        )

        # ---------------- (a) altitude against Mach ----------------
        ax.fill_between(mach, 0.0, ceiling_km, color=GOOD, alpha=0.20, lw=0)
        ax.plot(mach, ceiling_km, color=GOOD, lw=1.5)
        ax.axhspan(h_min_km, top_km, color=COOL, alpha=0.15, lw=0)
        ax.axhline(h_min_km, color=COOL, lw=1.4)
        ax.text(
            6.55, ceiling_km[-1] - 1.6, "ceiling of 15 g\naerodynamic availability",
            fontsize=6.4, color="#2e5b2e", ha="right", va="top",
        )
        ax.text(
            2.06, h_min_km + 0.35, "A3: intercept at or above 15 km", fontsize=6.4,
            color="#2b4a6b", ha="left", va="bottom",
        )
        ax.axvline(a["mach_intercept_min"], color=GREY, lw=0.9, ls=":")
        ax.text(
            a["mach_intercept_min"] + 0.05, 1.0, "A4: Mach 3 minimum", rotation=90,
            fontsize=6.4, color=GREY, ha="left", va="bottom",
        )

        ax.text(
            5.9, 3.4, "15 g AVAILABLE\nbut too low for A3", fontsize=7.2, color="#2e5b2e",
            ha="center", va="center",
        )
        ax.text(
            3.1, 23.5, "A3 SATISFIED\nbut no dynamic pressure,\nso no manoeuvre",
            fontsize=7.2, color="#2b4a6b", ha="center", va="center",
        )

        # Where each lofted trajectory actually reaches 100 miles of slant range.
        reached = [
            r for r in a["rows"]
            if r["slant_max"] >= a["required_slant_m"] - 1.0 and r["h_at_slant_max"] > 100.0
        ]
        ax.plot(
            [r["mach_at_slant_max"] for r in reached],
            [r["h_at_slant_max"] / 1000.0 for r in reached],
            "o", ms=6.0, mfc="none", mec=BAD, mew=1.4,
        )
        # Most of the lofted arcs reach 100 miles above the top of this frame. Say where, rather
        # than stretching the axis until the ceiling curve is unreadable.
        off_frame = [r for r in reached if r["h_at_slant_max"] / 1000.0 > top_km]
        if off_frame:
            ax.annotate(
                "%d more lofted arcs reach 100 miles at %.1f to %.1f km,\n"
                "above the top of this frame, where %.2f g is available"
                % (len(off_frame),
                   min(r["h_at_slant_max"] for r in off_frame) / 1000.0,
                   max(r["h_at_slant_max"] for r in off_frame) / 1000.0,
                   max(r["g_at_slant_max"] for r in off_frame)),
                xy=(5.9, top_km), xytext=(2.06, 29.4), fontsize=6.4, color=BAD,
                ha="left", va="top",
                arrowprops={"arrowstyle": "->", "color": BAD, "lw": 0.8},
            )
        for r in reached:
            if r["h_at_slant_max"] / 1000.0 > top_km:
                continue
            ax.annotate(
                "100 miles reached here:\npitchover %.0f deg, %.1f km, M %.1f, %.1f g"
                % (r["gamma_deg"], r["h_at_slant_max"] / 1000.0, r["mach_at_slant_max"],
                   r["g_at_slant_max"]),
                xy=(r["mach_at_slant_max"], r["h_at_slant_max"] / 1000.0),
                xytext=(r["mach_at_slant_max"] + 0.25, r["h_at_slant_max"] / 1000.0 + 1.6),
                fontsize=6.2, color="#8a3a3a", ha="left", va="bottom",
                arrowprops={"arrowstyle": "-", "color": "#8a3a3a", "lw": 0.6},
            )
        if m_overlap is not None:
            ax.annotate(
                "the two regions first overlap\nat Mach %.2f, and the overlap\nis %.1f km tall "
                "at Mach 5" % (m_overlap,
                               a["ceiling"][-1]["h_limit_m"] / 1000.0 - h_min_km),
                xy=(m_overlap, h_min_km), xytext=(4.7, 8.4),
                fontsize=6.4, color=INK, ha="left", va="top",
                arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.7},
            )

        ax.set_xlim(mach[0], mach[-1])
        ax.set_ylim(0.0, top_km)
        ax.set_xlabel("Mach")
        ax.set_ylabel("altitude [km]")
        ax.set_title(
            "(a) A3 and A11 do not overlap where the vehicle flies.\n"
            "Post-separation mass %.0f kg, generous CN_max %.1f"
            % (a["stack"]["mass_after_separation_kg"], a["cn_max_placeholder"]),
            loc="left",
        )

        # ---------------- (b) slant range against the requirement ----------------
        rows = sorted(a["rows"], key=lambda r: r["gamma_deg"])
        ypos = list(range(len(rows)))
        ok_km = [r["slant_ok"] / 1000.0 for r in rows]
        max_km = [r["slant_max"] / 1000.0 for r in rows]
        req_km = a["required_slant_m"] / 1000.0

        ax2.barh(ypos, ok_km, color=GOOD, height=0.58, edgecolor="none")
        ax2.plot(max_km, ypos, "d", ms=5.5, mfc="none", mec=BAD, mew=1.3)
        ax2.axvline(req_km, color=BAD, lw=1.5)
        ax2.set_ylim(-0.7, len(rows) - 0.1)
        ax2.text(
            req_km, len(rows) - 0.22, "A2: %.1f km = %.0f miles" % (req_km, a["required_slant_miles"]),
            fontsize=6.6, color=BAD, ha="right", va="bottom",
        )
        ax2.set_yticks(ypos)
        ax2.set_yticklabels(["%.0f deg" % r["gamma_deg"] for r in rows], fontsize=7.2)
        ax2.set_ylabel("commanded pitchover angle")
        ax2.set_xlabel("slant range from the launch point [km]")
        ax2.set_xlim(0.0, 185.0)
        ax2.grid(axis="y", visible=False)
        for y, r in zip(ypos, rows):
            if r["slant_ok"] > 0.0:
                ax2.text(
                    r["slant_ok"] / 1000.0 + 2.0, y,
                    "%.1f km, h %.1f km" % (r["slant_ok"] / 1000.0, r["h_ok"] / 1000.0),
                    va="center", ha="left", fontsize=6.2, color="#2e5b2e",
                )
            else:
                ax2.text(2.0, y, "nothing on this arc meets all three",
                         va="center", ha="left", fontsize=6.2, color=GREY)
        best_y = [i for i, r in enumerate(rows)
                  if abs(r["slant_ok"] - a["best_slant_m"]) < 1.0][0]
        ax2.annotate(
            "best point anywhere: %.1f km = %.1f miles.\nA2 asks for %.0f miles, so the "
            "shortfall\nis %.1f km, a factor of %.2f."
            % (a["best_slant_m"] / 1000.0, a["best_slant_miles"], a["required_slant_miles"],
               a["shortfall_m"] / 1000.0, a["shortfall_factor"]),
            xy=(a["best_slant_m"] / 1000.0, best_y + 0.3),
            xytext=(74.0, len(rows) - 1.15), fontsize=6.8, color="#8a3a3a",
            ha="left", va="top",
            arrowprops={"arrowstyle": "->", "color": "#8a3a3a", "lw": 0.8},
        )
        ax2.set_title(
            "(b) bars: furthest point meeting A3, A4 and A11 at once.\n"
            "Diamonds: furthest point the arc reaches",
            loc="left",
        )

        fig.text(
            0.008, 0.985,
            "Stack: %.0f kg, %.2f m, %.0f kN.s of vacuum impulse, %.0f kg jettisoned. "
            "Every lateral-acceleration figure here is an UPPER bound, because the audit uses a "
            "generous CN_max of %.1f\n   in place of the build-up. "
            "Source: SPEC_IV1.md section 2, reproduced in runs/IV-1/figures/evidence_iv1.json."
            % (a["stack"]["m0_kg"], a["stack"]["L_total_m"], a["stack"]["impulse_kNs"],
               a["stack"]["jettisoned_kg"], a["cn_max_placeholder"]),
            fontsize=6.8, color=GREY, va="top", ha="left",
        )
        fig.subplots_adjust(left=0.062, right=0.988, top=0.845, bottom=0.105, wspace=0.30)
        path = path or out_path("requirements_conflict.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


if __name__ == "__main__":
    print(make_figure())
