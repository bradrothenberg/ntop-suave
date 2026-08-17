"""WP2 validation figure: runs/_aero/aero_validation.png.

Four panels:
  (a) SV-1 CD0 versus Mach with the component drag breakdown stacked.
  (b) CN_alpha versus Mach, model curves for SV-1 and the Basic Finner, with the Basic Finner
      free-flight data overlaid.
  (c) x_cp/D versus Mach, same two configurations, same overlay.
  (d) CD0 validation for the Basic Finner: model against free-flight data, with the stated
      tolerance band and the transonic-blend region marked.

Reference data is Dupuis and Hathaway, DREV-TM-9703 (1997), Table VII. It is imported from
tests/test_aero.py so there is exactly one copy of it in the repository.

Run:
    .venv/Scripts/python.exe -m rocketgen.report.fig_aero
"""
from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import DesignVector, RUNS_DIR  # noqa: E402
from ..sizing.aero import RocketAero  # noqa: E402

# --------------------------------------------------------------------------------------
#   Plot style: clean, monospaced, no chartjunk.
# --------------------------------------------------------------------------------------

STYLE: dict[str, object] = {
    "font.family": "monospace",
    "font.monospace": ["DejaVu Sans Mono", "Consolas", "Courier New"],
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.0,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#4d4d4d",
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#dcdcdc",
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "legend.fontsize": 7.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "lines.linewidth": 1.3,
}

INK = "#1c1c1c"
ACCENT = "#8a9a00"          # nTop accent, muted for print
GREY = "#7a7a7a"

# Ordered so the largest, most physical components sit at the bottom of the stack.
STACK = (
    ("CD_friction_body", "body friction", "#3d5a80"),
    ("CD_wave_body", "body wave", "#98c1d9"),
    ("CD_base", "base", "#ee6c4d"),
    ("CD_boattail", "boattail", "#f0c05a"),
    ("CD_fin_friction", "fin friction", "#5c8001"),
    ("CD_fin_wave", "fin wave", "#a3c644"),
    ("CD_protuberance_GUESS", "protuberance (GUESS)", "#b0b0b0"),
)


def _reference_data():
    """Basic Finner free-flight data and geometry, from the single copy in the test module.

    Importing the reference data from the test module is deliberate: the digitised Table VII
    must exist exactly once in the repository, and the tests are its home.
    """
    import sys

    from ..config import REPO_ROOT

    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from tests.test_aero import (
        BASIC_FINNER_D,
        BASIC_FINNER_TABLE_VII,
        BASIC_FINNER_XCG_CAL,
        DBSQ_MIN,
        basic_finner_dv,
    )

    return BASIC_FINNER_TABLE_VII, BASIC_FINNER_D, BASIC_FINNER_XCG_CAL, DBSQ_MIN, basic_finner_dv


def make_figure(out_path: str | None = None, alpha_deg: float = 2.0) -> str:
    """Build the figure and return the path written."""
    table, d_bf, xcg_cal, dbsq_min, bf_dv = _reference_data()

    alpha = math.radians(alpha_deg)
    sv1 = RocketAero(DesignVector())
    bf = RocketAero(bf_dv(), nose_shape="cone")

    mach = np.arange(0.30, 5.0 + 1e-9, 0.01)
    sv1_r = [sv1.evaluate(float(m), 12_000.0, alpha) for m in mach]
    bf_r = [bf.evaluate(float(m), 0.0, alpha) for m in mach]

    exp_m = np.array([r[0] for r in table])
    exp_dbsq = np.array([r[1] for r in table])
    exp_cd0 = np.array([r[2] for r in table])
    exp_cna = np.array([r[3] for r in table])
    exp_cma = np.array([r[4] for r in table])
    exp_xcp = xcg_cal - exp_cma / exp_cna
    good = exp_dbsq >= dbsq_min

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.6))
        ax_a, ax_b = axes[0]
        ax_c, ax_d = axes[1]

        # ---------------- (a) SV-1 CD0 with the breakdown stacked ----------------
        comps = np.array([[r.breakdown[k] for r in sv1_r] for k, _lab, _c in STACK])
        labels = [lab for _k, lab, _c in STACK]
        colors = [c for _k, _lab, c in STACK]
        ax_a.stackplot(mach, comps, labels=labels, colors=colors, edgecolor="none")
        sv1_cd0 = np.array([r.CD0 for r in sv1_r])
        ax_a.plot(mach, sv1_cd0, color=INK, lw=1.4, label="CD0 total")
        top_a = 1.35 * float(sv1_cd0.max())
        ax_a.plot([0.95, 1.20], [top_a * 0.90] * 2, color="#8a3a3a", lw=1.0, solid_capstyle="butt")
        ax_a.text(
            1.075, top_a * 0.885, "transonic blend", ha="center", va="top",
            fontsize=6.5, color="#8a3a3a",
        )
        ax_a.set_xlim(0.3, 5.0)
        ax_a.set_ylim(0.0, top_a)
        ax_a.set_xlabel("Mach")
        ax_a.set_ylabel("CD0 on S_ref")
        ax_a.set_title("(a) SV-1 zero-lift drag build-up, h = 12 km", loc="left")
        ax_a.legend(loc="upper right", ncol=2, handlelength=1.4, columnspacing=1.0)

        # ---------------- (b) CN_alpha ----------------
        ax_b.plot(mach, [r.CN_alpha for r in bf_r], color=INK, label="model, Basic Finner")
        ax_b.plot(mach, [r.CN_alpha for r in sv1_r], color=ACCENT, ls="--", label="model, SV-1")
        ax_b.plot(
            exp_m[good], exp_cna[good], "o", ms=4.0, mfc="none", mec="#c1121f", mew=1.0,
            label="Dupuis 1997, free flight",
        )
        ax_b.plot(
            exp_m[~good], exp_cna[~good], "x", ms=4.0, color="#d0a0a0",
            label="low-alpha shots (excluded)",
        )
        ax_b.set_xlim(0.3, 5.0)
        ax_b.set_ylim(0.0, 26.0)
        ax_b.set_xlabel("Mach")
        ax_b.set_ylabel("CN_alpha, per radian")
        ax_b.set_title(f"(b) normal-force slope at alpha = {alpha_deg:g} deg", loc="left")
        ax_b.legend(loc="upper right", handlelength=1.6)

        # ---------------- (c) x_cp/D ----------------
        ax_c.plot(mach, [r.x_cp / d_bf for r in bf_r], color=INK, label="model, Basic Finner")
        ax_c.plot(
            mach, [r.x_cp / sv1.geom.D for r in sv1_r], color=ACCENT, ls="--",
            label="model, SV-1",
        )
        ax_c.plot(
            exp_m[good], exp_xcp[good], "o", ms=4.0, mfc="none", mec="#c1121f", mew=1.0,
            label="Dupuis 1997, x_cg - CMa/CNa",
        )
        ax_c.axhline(xcg_cal, color=GREY, lw=0.8, ls=":")
        ax_c.text(
            4.9, xcg_cal + 0.08, "Basic Finner CG, 5.50 D", ha="right", va="bottom",
            fontsize=6.5, color=GREY,
        )
        ax_c.set_xlim(0.3, 5.0)
        ax_c.set_ylim(2.0, 12.0)
        ax_c.set_xlabel("Mach")
        ax_c.set_ylabel("x_cp / D from the nose tip")
        ax_c.set_title("(c) centre of pressure", loc="left")
        ax_c.legend(loc="lower left", handlelength=1.6)

        # ---------------- (d) CD0 validation overlay ----------------
        bf_cd0 = np.array([r.CD0 for r in bf_r])
        ax_d.fill_between(
            mach, 0.75 * bf_cd0, 1.25 * bf_cd0, color=ACCENT, alpha=0.16, lw=0,
            label="model +/- 25 %",
        )
        ax_d.plot(mach, bf_cd0, color=INK, label="model, Basic Finner")
        ax_d.plot(
            exp_m, exp_cd0, "o", ms=4.0, mfc="none", mec="#c1121f", mew=1.0,
            label="Dupuis 1997, 23 shots",
        )
        ax_d.axvspan(0.95, 1.40, color="#c1121f", alpha=0.06, lw=0)
        ax_d.annotate(
            "transonic blend,\nnot validated",
            xy=(1.17, 0.75), xytext=(2.05, 0.90), fontsize=6.5, color="#8a3a3a",
            ha="left", va="top",
            arrowprops={"arrowstyle": "-", "color": "#8a3a3a", "lw": 0.7},
        )
        ax_d.set_xlim(0.3, 5.0)
        ax_d.set_ylim(0.0, 1.02)
        ax_d.set_xlabel("Mach")
        ax_d.set_ylabel("CD0 on S_ref")
        ax_d.set_title("(d) validation: Basic Finner CD0", loc="left")
        ax_d.legend(loc="upper right", handlelength=1.6)

        fig.suptitle(
            "WP2 aerodynamic build-up. Reference: A. D. Dupuis and W. Hathaway, "
            "DREV-TM-9703 (1997), Table VII, Basic Finner free-flight data.",
            fontsize=7.5, color=GREY, y=0.985,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))

        if out_path is None:
            out_dir = os.path.join(RUNS_DIR, "_aero")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "aero_validation.png")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    return out_path


if __name__ == "__main__":
    print(make_figure())
