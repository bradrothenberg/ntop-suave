"""WP7 figure: normalised constraint margin for the converged SV-1.

Source data: runs/SV-1/converged/point_ntop.json. The `margin` field is already normalised by
the limit, so a value of 0.10 means the constraint is met with 10 percent to spare. Nothing is
recomputed here.

The figure exists to make the active constraint obvious. Dynamic pressure and range sit within
2.5 percent of their limits; every other constraint has an order of magnitude more room.

Run:
    .venv/Scripts/python.exe -m rocketgen.report.fig_margins
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from .figstyle import BAD, GOOD, GREY, STYLE, WARN, out_path, point_analytic, point_ntop

#: Margins at or below this fraction are drawn as "tight" and labelled.
TIGHT = 0.05

#: Display scale and unit per constraint, so the labels read in engineering units instead of
#: raw SI exponents. Key is the constraint name as the loop writes it.
DISPLAY: dict[str, tuple[float, str, str]] = {
    "R3 range": (1.0e-3, "km", "%.1f"),
    "R6 terminal Mach": (1.0, "Mach", "%.2f"),
    "R7 diameter": (1.0, "m", "%.2f"),
    "R8 length": (1.0, "m", "%.2f"),
    "R9 launch mass": (1.0, "kg", "%.0f"),
    "R10 static margin": (1.0, "cal", "%.2f"),
    "R11 fin span": (1.0, "m", "%.2f"),
    "q_max": (1.0e-3, "kPa", "%.1f"),
    "grain L/D lower": (1.0, "", "%.2f"),
    "grain L/D upper": (1.0, "", "%.2f"),
}


def _label(c: dict) -> str:
    scale, unit, fmt = DISPLAY.get(c["name"], (1.0, c["units"], "%.3g"))
    value = fmt % (c["value"] * scale)
    limit = fmt % (c["limit"] * scale)
    tail = (" " + unit) if unit else ""
    return "%+.1f %%   (%s %s %s%s)" % (100.0 * c["margin"], value, c["sense"], limit, tail)


def make_figure(path: str | None = None) -> str:
    point = point_ntop()
    analytic = {c["name"]: c for c in point_analytic()["constraints"]}
    cons = sorted(point["constraints"], key=lambda c: -c["margin"])

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9.4, 3.9))
        names = [c["name"] for c in cons]
        margins = [c["margin"] for c in cons]
        colours = [WARN if m <= TIGHT else GOOD for m in margins]
        ypos = list(range(len(cons)))

        ax.barh(ypos, margins, color=colours, height=0.6, edgecolor="none")
        ax.plot(
            [analytic[n]["margin"] for n in names], ypos, "|", ms=9.0, mew=1.4, color=BAD,
            label="same design, analytic geometry",
        )
        ax.axvline(0.0, color="#4d4d4d", lw=0.9)
        ax.axvspan(-0.02, TIGHT, color=WARN, alpha=0.08, lw=0)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=7.4)
        ax.set_xscale("symlog", linthresh=0.05, linscale=0.9)
        ax.set_xlim(-0.02, 40.0)
        ax.set_xticks([0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
        ax.set_xticklabels(["0", "0.02", "0.05", "0.1", "0.25", "0.5", "1", "2", "5", "10"])
        ax.set_xlabel("normalised margin, (limit - value) / limit, symmetric log scale")
        ax.grid(axis="y", visible=False)
        ax.set_title("Constraint margins, converged SV-1, all ten constraints met", loc="left")

        for y, c in zip(ypos, cons):
            ax.text(
                max(c["margin"], 0.0) * 1.12 + 0.004, y, _label(c), va="center", ha="left",
                fontsize=6.3, color="#3a3a3a",
            )
        ax.legend(loc="center right", handlelength=1.2)
        fig.text(
            0.012, 0.975,
            "Bars are the nTop-coupled result; the red ticks are the same design vector with "
            "analytic geometry.\nThe shaded band marks margins below %d percent. Source: "
            "runs/SV-1/converged/point_ntop.json." % int(100 * TIGHT),
            fontsize=7.0, color=GREY, va="top", ha="left",
        )
        fig.subplots_adjust(left=0.155, right=0.985, top=0.825, bottom=0.125)
        path = path or out_path("constraint_margins.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


if __name__ == "__main__":
    print(make_figure())
