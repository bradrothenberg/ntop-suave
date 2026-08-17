"""WP7 figure: the trade-study carpet over D and m_p_sustain, at each f_nose.

Source data: runs/SV-1/doe/grid.csv, the 45-node full factorial written by `run_sv1.py
--stage doe`. Nothing is recomputed here.

Only 3 of the 45 nodes are feasible, so the feasible island is drawn hard: a filled green
marker with a heavy ring, against small grey crosses for everything else. The converged design
is ringed in red and labelled.

Run:
    .venv/Scripts/python.exe -m rocketgen.report.fig_carpet
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .figstyle import (
    ACCENT,
    BAD,
    GOOD,
    GREY,
    INK,
    STYLE,
    grid_rows,
    out_path,
    point_ntop,
)

#: Short labels for the violated-constraint annotation, so the panels stay readable.
SHORT: dict[str, str] = {
    "R3 range": "range",
    "R6 terminal Mach": "M_imp",
    "R10 static margin": "SM",
    "R11 fin span": "span",
    "q_max": "q",
    "grain L/D upper": "L/D",
}


def _panel_data(rows: list[dict[str, str]], f_nose: float):
    """Return (D axis, m_p_sustain axis, m0 grid, range grid, feasible mask, violations)."""
    sel = [r for r in rows if float(r["f_nose"]) == f_nose]
    d_axis = sorted({float(r["D"]) for r in sel})
    s_axis = sorted({float(r["m_p_sustain"]) for r in sel})
    m0 = np.full((len(s_axis), len(d_axis)), np.nan)
    rng = np.full_like(m0, np.nan)
    feas = np.zeros_like(m0, dtype=bool)
    viol: dict[tuple[int, int], str] = {}
    for r in sel:
        i = s_axis.index(float(r["m_p_sustain"]))
        j = d_axis.index(float(r["D"]))
        m0[i, j] = float(r["m0_kg"])
        rng[i, j] = float(r["range_km"])
        feas[i, j] = r["feasible"] == "1"
        names = [SHORT.get(v, v) for v in r["violations"].split("|") if v]
        viol[(i, j)] = ",".join(names)
    return np.array(d_axis), np.array(s_axis), m0, rng, feas, viol


def make_figure(path: str | None = None) -> str:
    rows = grid_rows()
    point = point_ntop()
    dv = point["design_vector"]
    d_star, s_star, f_star = dv["D"], dv["m_p_sustain"], dv["f_nose"]

    f_values = sorted({float(r["f_nose"]) for r in rows})
    all_m0 = np.array([float(r["m0_kg"]) for r in rows])
    n_feasible = sum(1 for r in rows if r["feasible"] == "1")

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(f_values), figsize=(9.4, 3.8), sharey=True)
        levels = np.linspace(all_m0.min(), all_m0.max(), 13)
        mesh = None
        for ax, f_nose in zip(axes, f_values):
            d_axis, s_axis, m0, rng, feas, viol = _panel_data(rows, f_nose)
            xx, yy = np.meshgrid(d_axis, s_axis)
            mesh = ax.contourf(xx, yy, m0, levels=levels, cmap="YlGnBu", alpha=0.85)
            cs = ax.contour(xx, yy, rng, levels=6, colors=INK, linewidths=0.7)
            ax.clabel(cs, inline=True, fontsize=6.0, fmt="%.0f km")

            for i in range(len(s_axis)):
                for j in range(len(d_axis)):
                    if feas[i, j]:
                        ax.plot(
                            d_axis[j], s_axis[i], "o", ms=9.0, mfc=GOOD, mec="white", mew=1.4,
                            zorder=5,
                        )
                        ax.plot(
                            d_axis[j], s_axis[i], "o", ms=15.0, mfc="none", mec=GOOD, mew=1.6,
                            zorder=5,
                        )
                    else:
                        ax.plot(d_axis[j], s_axis[i], "x", ms=4.0, color=GREY, mew=1.0, zorder=4)
                        ax.annotate(
                            viol[(i, j)], (d_axis[j], s_axis[i]), textcoords="offset points",
                            xytext=(0, 5), ha="center", fontsize=5.0, color=GREY,
                        )
            if abs(f_nose - f_star) < 1e-9:
                ax.plot(
                    d_star, s_star, "o", ms=22.0, mfc="none", mec=BAD, mew=2.0, zorder=6,
                )
                ax.annotate(
                    "converged SV-1", (d_star, s_star), textcoords="offset points",
                    xytext=(0, 17), ha="center", va="bottom", fontsize=6.5, color=BAD,
                )
            ax.set_xlim(d_axis.min() - 0.012, d_axis.max() + 0.012)
            ax.set_ylim(s_axis.min() - 14.0, s_axis.max() + 14.0)
            ax.set_xticks(d_axis)
            ax.set_yticks(s_axis)
            ax.set_xlabel("body diameter D [m]")
            ax.set_title("f_nose = %.1f" % f_nose, loc="left")
        axes[0].set_ylabel("sustain propellant [kg]")

        cbar = fig.colorbar(mesh, ax=axes, fraction=0.030, pad=0.015)
        cbar.set_label("launch mass m0 [kg]", fontsize=7.5)
        cbar.ax.tick_params(labelsize=7.0)

        handles = [
            plt.Line2D([], [], marker="o", ls="none", ms=7, mfc=GOOD, mec="white",
                       label="feasible node"),
            plt.Line2D([], [], marker="x", ls="none", ms=5, color=GREY,
                       label="infeasible, violated constraints labelled"),
            plt.Line2D([], [], marker="o", ls="none", ms=9, mfc="none", mec=BAD,
                       label="converged design"),
            plt.Line2D([], [], color=INK, lw=0.8, label="range contour"),
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=4, fontsize=6.8, handlelength=1.6,
            bbox_to_anchor=(0.45, 0.005),
        )
        fig.text(
            0.012, 0.965,
            "SV-1 trade study: 45-node full factorial over D, sustain propellant and nose "
            "fineness.\n%d of %d nodes are feasible. Source: runs/SV-1/doe/grid.csv."
            % (n_feasible, len(rows)),
            fontsize=7.0, color=GREY, va="top", ha="left",
        )
        fig.subplots_adjust(left=0.075, right=0.885, top=0.855, bottom=0.185, wspace=0.09)
        path = path or out_path("carpet.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


if __name__ == "__main__":
    print(make_figure())
