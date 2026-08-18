"""WP7 figure: the SV-1 mass statement, coloured by provenance.

Source data: runs/SV-1/converged/point_ntop.json, the converged design with real nTop geometry
in the loop. Nothing is recomputed here.

The colour is the message. Only one line item is nTop-measured, and the reader must be able to
see at a glance how much of the launch mass is measured geometry and how much is a requirement,
a correlation or an analytic estimate.

Run:
    .venv/Scripts/python.exe -m rocketgen.report.fig_mass
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from .figstyle import (
    GREY,
    PROVENANCE_COLOUR,
    PROVENANCE_LABEL,
    STYLE,
    out_path,
    point_ntop,
    source_label,
)


def make_figure(path: str | None = None) -> str:
    point = point_ntop()
    ms = point["mass_statement"]
    items = sorted(ms["items"], key=lambda e: e["mass_kg"])
    total = ms["total_kg"]

    totals: dict[str, float] = {}
    for e in ms["items"]:
        totals[e["provenance"]] = totals.get(e["provenance"], 0.0) + e["mass_kg"]
    order = sorted(totals, key=lambda k: -totals[k])

    with plt.rc_context(STYLE):
        fig, (ax, ax2) = plt.subplots(
            1, 2, figsize=(9.4, 4.4), gridspec_kw={"width_ratios": [3.05, 1.0]}
        )

        names = [e["name"] for e in items]
        masses = [e["mass_kg"] for e in items]
        colours = [PROVENANCE_COLOUR.get(e["provenance"], GREY) for e in items]
        ypos = list(range(len(items)))
        ax.barh(ypos, masses, color=colours, height=0.66, edgecolor="none")
        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=7.0)
        ax.set_xlabel("mass [kg]")
        ax.set_xlim(0.0, max(masses) * 1.26)
        ax.grid(axis="y", visible=False)
        ax.set_title("(a) group-weight statement, converged SV-1", loc="left")
        for y, m in zip(ypos, masses):
            ax.text(
                m + 0.008 * max(masses), y, "%.1f kg  %.1f%%" % (m, 100.0 * m / total),
                va="center", ha="left", fontsize=6.3, color="#3a3a3a",
            )

        handles = [
            plt.Rectangle((0, 0), 1, 1, fc=PROVENANCE_COLOUR[k], ec="none",
                          label=PROVENANCE_LABEL[k])
            for k in order
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=6.8, handlelength=1.5)

        # ---- (b) provenance shares as one stacked bar ----
        bottom = 0.0
        for key in order:
            share = 100.0 * totals[key] / total
            ax2.bar(
                0.0, share, bottom=bottom, width=0.62, color=PROVENANCE_COLOUR[key],
                edgecolor="white", linewidth=0.8,
            )
            ax2.text(
                0.0, bottom + 0.5 * share,
                "%s\n%.1f kg  %.1f%%" % (PROVENANCE_LABEL[key], totals[key], share),
                ha="center", va="center", fontsize=6.4,
                color="white" if key in ("requirement", "analytic") else "#1c1c1c",
            )
            bottom += share
        ax2.set_xlim(-0.55, 0.55)
        ax2.set_ylim(0.0, 106.0)
        ax2.set_yticks([0, 20, 40, 60, 80, 100])
        ax2.set_xticks([])
        ax2.set_ylabel("share of launch mass [%]")
        ax2.grid(axis="x", visible=False)
        ax2.set_title("(b) by provenance", loc="left")

        fig.text(
            0.012, 0.982,
            "Launch mass %.1f kg, CG %.3f m from the nose tip, burnout mass %.1f kg. "
            "nTop-measured share %.1f %%.\nSource: %s."
            % (total, ms["x_cg_m"], ms["burnout_kg"], 100.0 * ms["measured_fraction"],
               source_label("converged/point_ntop.json")),
            fontsize=7.0, color=GREY, va="top", ha="left",
        )
        fig.subplots_adjust(left=0.185, right=0.985, top=0.875, bottom=0.095, wspace=0.42)
        path = path or out_path("mass_statement.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


if __name__ == "__main__":
    import argparse

    from .figstyle import select_study

    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--oml", default="ogive", choices=["ogive", "spline"],
                     help="which study to draw; spline reads runs/SV-1_spline")
    select_study(_ap.parse_args().oml)
    print(make_figure())
