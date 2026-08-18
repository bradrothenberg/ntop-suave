"""IV-1 figure: the per-stage mass statement, coloured by provenance.

Source data: `runs/IV-1/converged.json` through `runs/IV-1/figures/evidence_iv1.json`. Nothing is
recomputed here.

The message is what leaves the vehicle. IV-1 is a two-stage vehicle, so the mass statement is not
one list: it is a booster that is thrown away, a payload stage that flies on, and an interstage
that goes with the booster. The right-hand panel shows that split, and the jettisoned mass is
called out on it.

Run:
    .venv/Scripts/python.exe -m rocketgen.report.fig_mass_iv1
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from .figstyle import GREY, PROVENANCE_COLOUR, PROVENANCE_LABEL, STYLE
from .figstyle_iv1 import evidence, out_path

STAGE_TITLE = {
    1: "stage 1, booster: jettisoned at separation",
    2: "stage 2, payload stage: flies to intercept",
    0: "interstage",
}


def make_figure(path: str | None = None) -> str:
    ev = evidence()
    mass = ev["mass"]
    rows = mass["rows"]
    total = mass["m0_kg"]

    # The attitude-control pack is charged on top of the mass statement by the sizing script, so
    # it is added here as a stage-2 line item with its own provenance. Leaving it out would make
    # the bars miss 13 percent of the launch mass.
    items = list(rows) + [
        {
            "stage": 2, "item": "Attitude-control motor pack",
            "mass_kg": mass["acs_pack_kg"], "station_m": float("nan"),
            "provenance": "correlation",
        }
    ]

    order = [1, 0, 2]
    grouped: list[dict] = []
    for stage in order:
        part = sorted(
            [r for r in items if int(r["stage"]) == stage], key=lambda e: e["mass_kg"]
        )
        grouped.extend(part)

    with plt.rc_context(STYLE):
        fig, (ax, ax2) = plt.subplots(
            1, 2, figsize=(10.2, 5.0), gridspec_kw={"width_ratios": [3.0, 1.15]}
        )

        ypos = list(range(len(grouped)))
        masses = [r["mass_kg"] for r in grouped]
        colours = [PROVENANCE_COLOUR.get(r["provenance"], GREY) for r in grouped]
        labels = [f"s{int(r['stage'])}  {r['item']}" if r["stage"] else f"--  {r['item']}"
                  for r in grouped]
        ax.barh(ypos, masses, color=colours, height=0.68, edgecolor="none")
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=6.8)
        ax.set_xlabel("mass [kg]")
        ax.set_xlim(0.0, max(masses) * 1.32)
        ax.grid(axis="y", visible=False)
        ax.set_title("(a) group-weight statement by stage, converged IV-1", loc="left")
        for y, m in zip(ypos, masses):
            ax.text(
                m + 0.008 * max(masses), y, "%.2f kg  %.1f%%" % (m, 100.0 * m / total),
                va="center", ha="left", fontsize=6.2, color="#3a3a3a",
            )

        # Rules between the stage blocks, so the three groups read as three groups. The group
        # name goes on the top row of its own block, horizontally: a rotated label long enough to
        # carry the text overlaps its neighbours.
        n1 = sum(1 for r in grouped if int(r["stage"]) == 1)
        n0 = sum(1 for r in grouped if int(r["stage"]) == 0)
        for edge in (n1 - 0.5, n1 + n0 - 0.5):
            ax.axhline(edge, color="#4d4d4d", lw=0.8, ls="--")
        # The label sits on the SMALLEST bar of its group, which is the first row of the group
        # because each group is sorted by mass. That is the only row guaranteed to have space.
        for stage, y in ((1, 0), (0, n1), (2, n1 + n0)):
            ax.text(
                max(masses) * 1.31, y, STAGE_TITLE[stage], fontsize=6.4,
                color="#8a3a3a", ha="right", va="center", style="italic",
            )

        prov_order = sorted(mass["by_provenance"], key=lambda k: -mass["by_provenance"][k])
        handles = [
            plt.Rectangle((0, 0), 1, 1, fc=PROVENANCE_COLOUR[k], ec="none",
                          label=PROVENANCE_LABEL[k])
            for k in prov_order
        ]
        ax.legend(handles=handles, loc="center right", bbox_to_anchor=(0.995, 0.24),
                  fontsize=6.8, handlelength=1.5)

        # ---- (b) what stays and what goes ----
        stage_totals = mass["stage_totals_kg"]
        s1 = stage_totals["1"]
        s0 = stage_totals["0"]
        s2 = stage_totals["2"] + mass["acs_pack_kg"]
        blocks = [("stage 1", s1, "#c1121f"), ("interstage", s0, "#7a7a7a"),
                  ("stage 2", s2, "#3d5a80")]
        bottom = 0.0
        for name, value, colour in blocks:
            ax2.bar(0.0, value, bottom=bottom, width=0.5, color=colour,
                    edgecolor="white", linewidth=0.9)
            ax2.text(
                0.28, bottom + 0.5 * value,
                "%s\n%.1f kg, %.1f%%" % (name, value, 100.0 * value / total),
                ha="left", va="center", fontsize=6.6, color="#1c1c1c",
            )
            bottom += value

        jett = mass["jettisoned_kg"]
        ax2.annotate(
            "%.1f kg leaves\nthe vehicle at\nseparation" % jett,
            xy=(0.0, jett), xytext=(0.28, 0.16 * total),
            fontsize=6.6, color="#8a3a3a", ha="left", va="center",
            arrowprops={"arrowstyle": "->", "color": "#8a3a3a", "lw": 0.8},
        )
        ax2.plot([-0.25, 0.25], [jett, jett], color="#8a3a3a", lw=1.1, ls="--")
        ax2.set_xlim(-0.42, 1.60)
        ax2.set_ylim(0.0, total * 1.04)
        ax2.set_xticks([])
        ax2.set_ylabel("mass [kg]")
        ax2.grid(axis="x", visible=False)
        ax2.set_title("(b) what stays, what goes", loc="left")

        fig.text(
            0.010, 0.984,
            "Launch mass %.1f kg. nTop-measured share %.1f percent (%.2f kg). "
            "Jettisoned at separation %.1f kg, so the payload stage weighs %.1f kg "
            "after separation.\nThe propellant, the payload and the motor correlations do not "
            "come from geometry, which is why the measured share is small. "
            "Source: runs/IV-1/converged.json."
            % (total, 100.0 * mass["measured_fraction"], mass["measured_kg"],
               jett, mass["mass_after_separation_kg"]),
            fontsize=6.8, color=GREY, va="top", ha="left",
        )
        fig.subplots_adjust(left=0.235, right=0.975, top=0.885, bottom=0.085, wspace=0.30)
        path = path or out_path("mass_statement_iv1.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


if __name__ == "__main__":
    print(make_figure())
