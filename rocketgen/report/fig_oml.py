"""Figure: the splined outer mould line, and what moving it costs and buys.

    .venv/Scripts/python.exe -m rocketgen.report.fig_oml --oml spline

Four panels:

  (a) the nose profile at three blends, against the tangent ogive it replaces, with the nine
      B-spline control points drawn. These are the points the nTop notebook computes and
      revolves, so the panel shows the actual notebook input rather than a sampled curve.
  (b) the boattail run, same idea.
  (c) the shape ratio and the forebody volume against the blend. The two curves are the whole
      trade: less wave drag, less room inside.
  (d) the flown result against the blend, from the nTop-COUPLED sweep. This panel is the one
      that answers whether the trade has an interior optimum.

Source data: runs/SV-1_spline/figures/evidence.json plus the design vector.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..config import DesignVector
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
    point_ntop,
    select_study,
    source_label,
)

#: Blends drawn as profiles in panels (a) and (b), and their colours.
DRAWN = ((0.0, COOL), (0.7, ACCENT), (1.0, BAD))


def _profile(dv: DesignVector, blend: float):
    """(x, r) of the nose profile, plus its control points, at one blend."""
    from ..oml_spline import SplineProfile

    d = dv.replace(nose_blend=float(blend))
    p = SplineProfile(length=d.L_nose, radius=0.5 * d.D, control=d.nose_control)
    xs = np.linspace(0.0, 1.0, 401)
    pts = np.array([p.point_at(float(u)) for u in xs])
    return pts[:, 0], pts[:, 1], np.array(p.control_points())


def _boattail(dv: DesignVector, blend: float):
    from ..oml_spline import SplineProfile

    R, r_base = 0.5 * dv.D, 0.5 * dv.d_base
    d = dv.replace(boattail_blend=float(blend))
    p = SplineProfile(length=dv.L_boattail, radius=r_base, control=d.boattail_control,
                      r0_over_r=R / r_base)
    xs = np.linspace(0.0, 1.0, 201)
    pts = np.array([p.point_at(float(u)) for u in xs])
    return pts[:, 0], pts[:, 1], np.array(p.control_points())


def _tangent_ogive(dv: DesignVector):
    import math

    R, L = 0.5 * dv.D, dv.L_nose
    rho = (R * R + L * L) / (2.0 * R)
    xs = np.linspace(0.0, L, 401)
    rs = np.array([
        max(math.sqrt(max(rho * rho - (L - x) ** 2, 0.0)) - (rho - R), 0.0) for x in xs
    ])
    return xs, rs


def make_figure(path: str | None = None) -> str:
    point = point_ntop()
    fields = DesignVector.__dataclass_fields__
    dv = DesignVector(**{k: v for k, v in point["design_vector"].items() if k in fields})
    ev = evidence()
    trade = ev["shape_trade"]
    coupled = ev.get("shape_trade_coupled", {})

    b_star = float(dv.nose_blend)
    bt_star = float(dv.boattail_blend)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.6))
        ax_a, ax_b = axes[0]
        ax_c, ax_d = axes[1]

        # ---- (a) the nose ------------------------------------------------------------
        xo, ro = _tangent_ogive(dv)
        ax_a.plot(xo, ro, color=INK, lw=1.6, ls="--", label="tangent ogive, f_nose %.1f"
                  % dv.f_nose)
        for blend, colour in DRAWN:
            x, r, ctrl = _profile(dv, blend)
            label = "spline, blend %.2f" % blend
            if abs(blend - b_star) < 1e-9:
                label += "  (converged)"
            ax_a.plot(x, r, color=colour, lw=1.4, label=label)
            ax_a.plot(ctrl[:, 0], ctrl[:, 1], "o", ms=3.0, mfc="none", mec=colour, mew=0.9)
        ax_a.set_xlabel("station from the nose tip [m]")
        ax_a.set_ylabel("outer mould line radius [m]")
        ax_a.set_title("(a) nose profile and its 9 control points", loc="left", fontsize=8.4)
        ax_a.legend(loc="lower right", fontsize=6.4, handlelength=1.8)
        ax_a.set_xlim(0.0, dv.L_nose * 1.02)
        ax_a.set_ylim(0.0, 0.5 * dv.D * 1.12)

        # ---- (b) the boattail --------------------------------------------------------
        R, r_base = 0.5 * dv.D, 0.5 * dv.d_base
        ax_b.plot([0.0, dv.L_boattail], [R, r_base], color=INK, lw=1.6, ls="--",
                  label="straight cone")
        for blend, colour in ((0.0, COOL), (bt_star, ACCENT), (1.0, BAD)):
            x, r, ctrl = _boattail(dv, blend)
            label = "spline, blend %.2f" % blend
            if abs(blend - bt_star) < 1e-9:
                label += "  (converged)"
            ax_b.plot(x, r, color=colour, lw=1.4, label=label)
            ax_b.plot(ctrl[:, 0], ctrl[:, 1], "o", ms=3.0, mfc="none", mec=colour, mew=0.9)
        ax_b.set_xlabel("station from the boattail start [m]")
        ax_b.set_ylabel("radius [m]")
        ax_b.set_title("(b) boattail contraction", loc="left", fontsize=8.4)
        ax_b.legend(loc="lower left", fontsize=6.4, handlelength=1.8)
        ax_b.set_xlim(0.0, dv.L_boattail)

        # ---- (c) the shape trade -----------------------------------------------------
        blends = [r["nose_blend"] for r in trade["rows"]]
        ratio = [r["shape_ratio"] for r in trade["rows"]]
        d_vol = [r["d_nose_volume_pct"] for r in trade["rows"]]
        ax_c.plot(blends, ratio, "-o", color=BAD, ms=3.5, label="nose wave-drag shape ratio")
        ax_c.set_xlabel("nose_blend")
        ax_c.set_ylabel("shape ratio against the tangent ogive")
        ax_c.set_title("(c) less drag costs forebody volume", loc="left", fontsize=8.4)
        ax_c.axvline(b_star, color=GREY, lw=0.8, ls=":")
        ax_c2 = ax_c.twinx()
        ax_c2.plot(blends, d_vol, "-s", color=COOL, ms=3.2, label="nose enclosed volume")
        ax_c2.set_ylabel("change in nose enclosed volume [%]")
        ax_c2.grid(visible=False)
        h1, l1 = ax_c.get_legend_handles_labels()
        h2, l2 = ax_c2.get_legend_handles_labels()
        ax_c.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=6.5, handlelength=1.8)
        ax_c.text(b_star, ratio[0], "  converged", fontsize=6.3, color=GREY, va="top")

        # ---- (d) the coupled result --------------------------------------------------
        rows = [r for r in coupled.get("rows", []) if "failed" not in r]
        if rows:
            bs = [r["nose_blend"] for r in rows]
            m0 = [r["m0_kg"] for r in rows]
            rng = [r["range_km"] for r in rows]
            best = coupled.get("best_blend_by_penalty")
            ax_d.plot(bs, m0, "-o", color=INK, ms=4.0, label="launch mass, nTop-measured")
            if best is not None:
                i = bs.index(best)
                ax_d.plot([best], [m0[i]], "o", ms=11.0, mfc="none", mec=GOOD, mew=1.8)
                ax_d.annotate(
                    "lowest penalty of the five\nmeasured, blend %.2f" % best, (best, m0[i]),
                    textcoords="offset points", xytext=(-16, 30), ha="right",
                    fontsize=6.4, color=GOOD,
                    arrowprops={"arrowstyle": "-", "color": GOOD, "lw": 0.7},
                )
            ax_d.plot([b_star], [m0[bs.index(b_star)]], "o", ms=11.0, mfc="none", mec=BAD,
                      mew=1.8)
            ax_d.annotate(
                "where the search stopped,\nblend %.2f" % b_star, (b_star, m0[bs.index(b_star)]),
                textcoords="offset points", xytext=(-18, -26), ha="right", fontsize=6.4,
                color=BAD, arrowprops={"arrowstyle": "-", "color": BAD, "lw": 0.7},
            )
            ax_d.set_xlabel("nose_blend")
            ax_d.set_ylabel("launch mass [kg]")
            ax_d2 = ax_d.twinx()
            ax_d2.plot(bs, rng, "-s", color=COOL, ms=3.2, label="range")
            ax_d2.set_ylabel("range [km]")
            ax_d2.grid(visible=False)
            h1, l1 = ax_d.get_legend_handles_labels()
            h2, l2 = ax_d2.get_legend_handles_labels()
            ax_d.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=6.5, handlelength=1.8)
            ax_d.set_title("(d) flown result, geometry measured at every blend",
                           loc="left", fontsize=8.4)
        else:
            ax_d.set_axis_off()
            ax_d.text(
                0.5, 0.5,
                "the nTop-coupled blend sweep is not available\n"
                "(reason: %s)" % coupled.get("reason", "not collected"),
                ha="center", va="center", fontsize=7.5, color=BAD,
            )

        fig.text(
            0.012, 0.985,
            "The splined outer mould line. Panels (a) and (b) are the geometry nTop revolves; "
            "(c) and (d) are what moving it does.\nSource: %s and the design vector."
            % source_label("figures/evidence.json"),
            fontsize=7.0, color=GREY, va="top", ha="left",
        )
        fig.subplots_adjust(left=0.075, right=0.918, top=0.895, bottom=0.105,
                            hspace=0.50, wspace=0.48)
        path = path or out_path("oml_shape.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--oml", default="spline", choices=["ogive", "spline"],
                     help="which study to draw; only the spline study has a shape trade")
    select_study(_ap.parse_args().oml)
    print(make_figure())
