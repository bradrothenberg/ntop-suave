"""IV-1 figure: normalised constraint margin for the converged design.

Source data: `runs/IV-1/converged.json` through `runs/IV-1/figures/evidence_iv1.json`. The margin
is computed there, not here, as `(value - limit)/limit` for a lower bound and
`(limit - value)/limit` for an upper bound, so a value of 0.10 means ten percent to spare.

The figure exists to show which constraints bind. Three do: the slant range sits exactly on its
limit because the run terminates on it, the maximum diameter sits exactly on its limit, and the
stage-2 grain closes with 11 percent of its bay left.

Requirement A9, the static margin, is drawn separately and in a different colour. It is NOT one
of the fifteen constraints the sizing script records, and the value shown is computed for this
report. See `evidence_iv1.static_margin_record`.

Run:
    .venv/Scripts/python.exe -m rocketgen.report.fig_margins_iv1
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from .figstyle import BAD, GOOD, GREY, STYLE, WARN
from .figstyle_iv1 import evidence, out_path

#: Margins at or below this fraction are drawn as tight and shaded.
TIGHT = 0.05

#: Display scale, unit and format per constraint, so the labels read in engineering units.
DISPLAY: dict[str, tuple[float, str, str]] = {
    "A2 slant range [m]": (1.0e-3, "km", "%.1f"),
    "A3 intercept alt [m]": (1.0e-3, "km", "%.1f"),
    "A4 intercept Mach": (1.0, "Mach", "%.2f"),
    "A6 max diameter [m]": (1.0, "m", "%.2f"),
    "A7 stacked length [m]": (1.0, "m", "%.2f"),
    "A8 launch mass [kg]": (1.0, "kg", "%.0f"),
    "A10 q_max [Pa]": (1.0e-3, "kPa", "%.0f"),
    "A11 lateral g": (1.0, "g", "%.2f"),
    "A12 s1 burnout alt [m]": (1.0e-3, "km", "%.1f"),
    "grain L/D stage 1": (1.0, "", "%.2f"),
    "grain L/D stage 2": (1.0, "", "%.2f"),
    "stage 1 vol loading": (1.0, "", "%.3f"),
    "stage 2 vol loading": (1.0, "", "%.3f"),
    "stage 1 grain closes": (1.0, "", "%.0f"),
    "stage 2 grain closes": (1.0, "", "%.0f"),
}


def _label(c: dict) -> str:
    scale, unit, fmt = DISPLAY.get(c["name"], (1.0, "", "%.3g"))
    value = fmt % (c["value"] * scale)
    limit = fmt % (c["limit"] * scale)
    tail = (" " + unit) if unit else ""
    return "%+.1f %%   (%s %s %s%s)" % (100.0 * c["margin"], value, c["sense"], limit, tail)


def make_figure(path: str | None = None) -> str:
    ev = evidence()
    cons = sorted(ev["constraints"], key=lambda c: -c["margin"])
    sm = ev["static_margin"]

    a9 = {
        "name": "A9 static margin *",
        "value": sm["worst_calibres"],
        "limit": sm["limit_calibres"],
        "sense": ">=",
        "margin": (sm["worst_calibres"] - sm["limit_calibres"]) / sm["limit_calibres"],
        "met": sm["all_met"],
    }
    rows = cons + [a9]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9.8, 4.6))
        ypos = list(range(len(rows)))
        margins = [c["margin"] for c in rows]
        colours = []
        for c in rows:
            if not c["met"]:
                colours.append(BAD)
            elif c["margin"] <= TIGHT:
                colours.append(WARN)
            else:
                colours.append(GOOD)

        ax.barh(ypos, margins, color=colours, height=0.62, edgecolor="none")
        ax.axvline(0.0, color="#4d4d4d", lw=0.9)
        ax.axvspan(-0.02, TIGHT, color=WARN, alpha=0.08, lw=0)
        ax.set_yticks(ypos)
        ax.set_yticklabels([c["name"] for c in rows], fontsize=7.2)
        ax.set_xscale("symlog", linthresh=0.02, linscale=0.9)
        ax.set_xlim(-2.2, 30.0)
        ticks = [-2.0, -1.0, -0.5, 0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["-2", "-1", "-0.5", "0", "0.02", "0.05", "0.1", "0.25",
                            "0.5", "1", "2", "5", "10"])
        ax.set_xlabel(
            "normalised margin against the limit, symmetric log scale. Positive is met."
        )
        ax.grid(axis="y", visible=False)
        ax.set_title(
            "Constraint margins, converged IV-1. All fifteen recorded constraints are met.",
            loc="left",
        )
        for y, c in zip(ypos, rows):
            x = max(c["margin"], 0.0)
            ax.text(
                x * 1.14 + 0.004, y, _label(c), va="center", ha="left",
                fontsize=6.4, color="#3a3a3a",
            )

        handles = [
            plt.Rectangle((0, 0), 1, 1, fc=GOOD, ec="none", label="met"),
            plt.Rectangle((0, 0), 1, 1, fc=WARN, ec="none",
                          label="met, margin below %d %%" % int(100 * TIGHT)),
            plt.Rectangle((0, 0), 1, 1, fc=BAD, ec="none", label="not met"),
        ]
        # Bottom left: the only part of the frame with no bar in it.
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.015, 0.02),
                  fontsize=6.8, handlelength=1.4)

        fig.text(
            0.010, 0.978,
            "* A9 is NOT one of the fifteen constraints the sizing script records. The value "
            "shown is the worst static margin computed for this report over four flight "
            "configurations\n   and three Mach numbers, about the mass-statement centre of "
            "gravity. It is an open item, not a verdict: see section 6. "
            "Source: runs/IV-1/converged.json.",
            fontsize=6.8, color=GREY, va="top", ha="left",
        )
        fig.subplots_adjust(left=0.175, right=0.985, top=0.845, bottom=0.115)
        path = path or out_path("constraint_margins_iv1.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


if __name__ == "__main__":
    print(make_figure())
