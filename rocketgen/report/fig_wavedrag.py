"""Figure: the slender-body wave-drag model, and its validation against exact closed forms.

    .venv/Scripts/python.exe -m rocketgen.report.fig_wavedrag --oml spline

Three panels:

  (a) the Glauert shape factor against nose fineness, for the tangent ogive, the 9-point
      optimal spline and the von Karman ogive. The von Karman line is the theoretical bound,
      and the vertical gap between the ogive and the bound is what the spline exists to collect.
  (b) the residual of every closed-form check, on a log scale, against the tolerance each test
      asserts. Bars far below their tolerance mean the model reproduces the closed form rather
      than merely passing.
  (c) the drag build-up against Mach for the two shapes, with the forebody wave-drag term
      drawn separately. The shape ratio multiplies that term and nothing else.

Source data: runs/SV-1_spline/figures/evidence.json, section `wavedrag`.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from .figstyle import (
    ACCENT,
    BAD,
    COOL,
    GOOD,
    GREY,
    INK,
    STYLE,
    evidence,
    out_path,
    select_study,
    source_label,
)


def make_figure(path: str | None = None) -> str:
    ev = evidence()
    w = ev["wavedrag"]
    share = ev["shape_trade"]["wave_share"]

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(9.4, 3.6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.05])
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])

        # ---- (a) shape factor against fineness ---------------------------------------
        bound = w["shape_factor_bound"]
        fs = [r["f_nose"] for r in w["ogive_penalty_by_fineness"]]
        og = [r["sf_over_bound"] * bound for r in w["ogive_penalty_by_fineness"]]
        ax_a.plot(fs, og, "-o", color=BAD, ms=4.0, label="tangent ogive")
        ax_a.axhline(w["shape_factor_optimal_spline"], color=ACCENT, lw=1.5,
                     label="optimal %d-point spline" % w["n_ctrl"])
        ax_a.axhline(bound, color=GOOD, lw=1.5, ls="--",
                     label="von Karman bound, 4/pi")
        ax_a.set_xlabel("nose fineness, L_nose / D")
        ax_a.set_ylabel("Glauert shape factor")
        ax_a.set_title("(a) the gap the spline collects", loc="left", fontsize=8.4)
        ax_a.legend(loc="center right", fontsize=6.3, handlelength=1.8)
        ax_a.set_ylim(bound * 0.985, max(og) * 1.02)
        ax_a.annotate(
            "%.0f %% of the gap\nrecovered" % (100.0 * w["gap_recovered_fraction"]),
            (fs[1], 0.5 * (w["shape_factor_optimal_spline"] + og[1])),
            fontsize=6.4, color=GREY, ha="center",
        )

        # ---- (b) validation residuals ------------------------------------------------
        checks = [
            ("Sears-Haack D/q", abs(w["sears_haack"]["rel_err"]), 1.0e-4),
            ("von Karman D/q", abs(w["von_karman"]["rel_err"]), 1.0e-4),
            ("von Karman C_D on base", abs(w["von_karman"]["cd_rel_err"]), 1.0e-4),
            ("shape factor = 4/pi", abs(w["von_karman_shape_factor"]["rel_err"]), 1.0e-4),
        ]
        names = [c[0] for c in checks]
        vals = [max(c[1], 1.0e-12) for c in checks]
        tols = [c[2] for c in checks]
        y = np.arange(len(checks))
        ax_b.barh(y, vals, height=0.5, color=COOL, edgecolor="none")
        ax_b.plot(tols, y, "|", ms=12.0, mew=1.6, color=BAD, label="tolerance asserted")
        for yi, v in zip(y, vals):
            ax_b.text(v * 1.35, yi, "%.1e" % v, va="center", fontsize=6.2, color="#3a3a3a")
        ax_b.set_yticks(y)
        ax_b.set_yticklabels(names, fontsize=6.8)
        ax_b.set_xscale("log")
        ax_b.set_xlim(1.0e-6, 3.0e-3)
        ax_b.set_xlabel("relative residual")
        ax_b.grid(axis="y", visible=False)
        ax_b.set_title("(b) against exact closed forms", loc="left", fontsize=8.4)
        ax_b.legend(loc="lower right", fontsize=6.3, handlelength=1.0)

        # ---- (c) what it does to the drag build-up -----------------------------------
        mach = [r["mach"] for r in share]
        cd0_o = [r["cd0_ogive"] for r in share]
        cd0_s = [r["cd0_spline"] for r in share]
        wv_o = [r["cd_wave_body_ogive"] for r in share]
        wv_s = [r["cd_wave_body_spline"] for r in share]
        ax_c.plot(mach, cd0_o, "-o", color=INK, ms=3.6, label="CD0, tangent ogive")
        ax_c.plot(mach, cd0_s, "-o", color=ACCENT, ms=3.6, label="CD0, converged spline")
        ax_c.plot(mach, wv_o, "--s", color=GREY, ms=3.2, label="forebody wave term, ogive")
        ax_c.plot(mach, wv_s, "--s", color=BAD, ms=3.2, label="forebody wave term, spline")
        ax_c.set_xlabel("Mach")
        ax_c.set_ylabel("coefficient on S_ref")
        ax_c.set_title("(c) the term the shape ratio moves", loc="left", fontsize=8.4)
        ax_c.legend(loc="upper left", fontsize=6.2, handlelength=1.8)
        ax_c.set_ylim(0.0, max(cd0_o) * 1.28)
        for m, a, b in zip(mach, cd0_o, cd0_s):
            ax_c.annotate("%.1f %%" % (100.0 * (b / a - 1.0)), (m, b),
                          textcoords="offset points", xytext=(0, -12), ha="center",
                          fontsize=6.0, color=ACCENT)

        fig.text(
            0.012, 0.982,
            "The slender-body wave-drag ratio. It multiplies the calibrated Bonney correlation, "
            "so only the SHAPE effect comes from linear theory.\nSource: %s, section wavedrag."
            % source_label("figures/evidence.json"),
            fontsize=7.0, color=GREY, va="top", ha="left",
        )
        fig.subplots_adjust(left=0.075, right=0.988, top=0.845, bottom=0.135, wspace=0.62)
        path = path or out_path("wavedrag_validation.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--oml", default="spline", choices=["ogive", "spline"],
                     help="which study to draw; only the spline study collects the wave-drag "
                          "evidence")
    select_study(_ap.parse_args().oml)
    print(make_figure())
