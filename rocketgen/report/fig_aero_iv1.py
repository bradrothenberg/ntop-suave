"""IV-1 aerodynamic validation figure: runs/_aero_iv1/aero_iv1_validation.png.

Six panels:
  (a) stacked-configuration CD0 versus Mach with the component drag breakdown stacked.
  (b) payload-stage CD0 versus Mach, same breakdown, on ITS OWN reference area.
  (c) CN versus alpha at a representative Mach, both configurations, with the strake linear and
      vortex-lift parts drawn separately, and the requirement A11 alpha limit marked.
  (d) x_cp/D versus Mach for both configurations, strakes on and strakes off.
  (e) validation: the Polhamus and Lamar suction-analogy coefficients against the printed
      rectangular-wing table of NASA TN D-7921.
  (f) validation: measured strake normal-force increment for a body with side strakes,
      NASA TM X-3130, against what this model's strake term gives.

Reference data is imported from tests/test_aero_iv1.py so there is exactly one copy of it in the
repository, the same arrangement fig_aero.py uses for the Basic Finner table.

Run:
    .venv/Scripts/python.exe -m rocketgen.report.fig_aero_iv1
"""
from __future__ import annotations

import copy
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import RUNS_DIR  # noqa: E402
from ..config_iv1 import InterceptRequirements, StrakeSpec, default_iv1  # noqa: E402
from ..sizing.aero_iv1 import (  # noqa: E402
    StackAero,
    polhamus_kp,
    polhamus_kv,
    polhamus_kv_le,
    polhamus_kv_se,
)
from .fig_aero import ACCENT, GREY, INK, STYLE  # noqa: E402

# Ordered so the largest, most physical components sit at the bottom of the stack.
STACK = (
    ("CD_friction_body", "body friction", "#3d5a80"),
    ("CD_wave_body", "body wave", "#98c1d9"),
    ("CD_interstage_shoulder", "interstage shoulder", "#8a3a3a"),
    ("CD_base", "base", "#ee6c4d"),
    ("CD_fin_friction", "fin friction", "#5c8001"),
    ("CD_fin_wave", "fin wave", "#a3c644"),
    ("CD_strake_friction", "strake friction", "#6a4c93"),
    ("CD_strake_wave", "strake wave", "#b39ddb"),
    ("CD_strake_base", "strake base", "#d7c9ec"),
    ("CD_protuberance_GUESS", "protuberance (GUESS)", "#b0b0b0"),
)

STAGE_LABEL = {
    1: "stage 1: full stack, S_ref on booster D",
    2: "stage 2: payload stage alone, S_ref on its own D",
}

MACH_MIN = 0.3
MACH_MAX = 6.0
H_REF = 15_000.0        # m, the SPEC_IV1.md A3 intercept altitude
M_REF = 3.0             # SPEC_IV1.md A4 intercept Mach


def _reference_data():
    """Reference tables, from the single copy of them in the test module."""
    import sys

    from ..config import REPO_ROOT

    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from tests.test_aero_iv1 import (
        JORGENSEN_MATCHED,
        LAMAR_GLOSS_TABLE_III,
        _jorgensen_strake_model,
    )

    return LAMAR_GLOSS_TABLE_III, JORGENSEN_MATCHED, _jorgensen_strake_model


def _without_strakes(dv):
    out = copy.deepcopy(dv)
    out.strakes = StrakeSpec(
        n=dv.strakes.n,
        height=0.0,
        length=dv.strakes.length,
        thickness=dv.strakes.thickness,
        x_le=dv.strakes.x_le,
        sweep_le=dv.strakes.sweep_le,
    )
    return out


def _panel_cd0_breakdown(ax, aero: StackAero, stage: int, mach, letter: str) -> None:
    """Stacked component breakdown of CD0 against Mach for one configuration."""
    results = [aero.evaluate(float(m), H_REF, 0.0, stage) for m in mach]
    comps = np.array([[r.breakdown[k] for r in results] for k, _l, _c in STACK])
    ax.stackplot(
        mach,
        comps,
        labels=[lab for _k, lab, _c in STACK],
        colors=[c for _k, _lab, c in STACK],
        edgecolor="none",
    )
    cd0 = np.array([r.CD0 for r in results])
    ax.plot(mach, cd0, color=INK, lw=1.4, label="CD0 total")
    top = 1.45 * float(cd0.max())

    ax.plot([0.95, 1.20], [top * 0.94] * 2, color="#8a3a3a", lw=1.0, solid_capstyle="butt")
    ax.text(
        1.075, top * 0.925, "transonic blend", ha="center", va="top",
        fontsize=6.5, color="#8a3a3a",
    )
    # The model is validated to M 5.0 only; mark the extrapolated tail.
    ax.axvspan(5.0, MACH_MAX, color=GREY, alpha=0.12, lw=0)
    ax.text(
        5.5, top * 0.55, "extrapolated\nabove M 5", ha="center", va="center",
        fontsize=6.0, color=GREY,
    )
    ax.set_xlim(MACH_MIN, MACH_MAX)
    ax.set_ylim(0.0, top)
    ax.set_xlabel("Mach")
    ax.set_ylabel("CD0 on S_ref of this stage")
    ax.set_title(
        f"({letter}) {STAGE_LABEL[stage]} = {aero.S_ref(stage):.4f} m2, h = 15 km",
        loc="left",
    )
    ax.legend(loc="upper right", ncol=2, handlelength=1.3, columnspacing=0.9)


def _panel_cn_alpha(ax, aero: StackAero, reqs: InterceptRequirements) -> None:
    """CN against alpha, with the strake linear and vortex terms drawn separately."""
    alpha_deg = np.linspace(0.0, 25.0, 120)
    alpha = np.radians(alpha_deg)

    for stage, colour, style in ((1, INK, "-"), (2, ACCENT, "--")):
        cn = np.array([aero.evaluate(M_REF, H_REF, float(a), stage).CN for a in alpha])
        ax.plot(alpha_deg, cn, color=colour, ls=style, label=f"total CN, stage {stage}")

    # Strake terms, on the payload-stage reference area, magnified so they are readable.
    pot = np.array([aero.CN_strakes(M_REF, float(a), 2)[0] for a in alpha])
    vor = np.array([aero.CN_strakes(M_REF, float(a), 2)[1] for a in alpha])
    scale = 4.0
    ax.plot(
        alpha_deg, scale * (pot + vor), color="#6a4c93", lw=1.2,
        label=f"strake CN, stage 2 (x{scale:g})",
    )
    ax.plot(
        alpha_deg, scale * vor, color="#6a4c93", lw=1.0, ls=":",
        label=f"  vortex-lift part (x{scale:g})",
    )
    ax.plot(
        alpha_deg, scale * pot, color="#6a4c93", lw=1.0, ls="-.",
        label=f"  linear part (x{scale:g})",
    )

    a11 = math.degrees(reqs.alpha_max)
    ax.axvline(a11, color="#c1121f", lw=0.8, ls=":")
    cn_max_2 = aero.CN_max(M_REF, H_REF, 2, reqs.alpha_max)
    cn_max_1 = aero.CN_max(M_REF, H_REF, 1, reqs.alpha_max)
    ax.plot([a11], [cn_max_2], "o", ms=4.0, mfc="none", mec="#c1121f", mew=1.1)
    ax.plot([a11], [cn_max_1], "o", ms=4.0, mfc="none", mec="#c1121f", mew=1.1)
    ax.annotate(
        f"A11 limit {a11:g} deg\nCN_max = {cn_max_1:.2f} (stage 1)\n"
        f"             {cn_max_2:.2f} (stage 2)",
        xy=(a11, cn_max_1), xytext=(9.0, 0.92 * cn_max_1),
        fontsize=6.5, color="#8a3a3a", ha="left", va="top",
        arrowprops={"arrowstyle": "-", "color": "#8a3a3a", "lw": 0.7},
    )
    top = 1.15 * float(
        max(aero.evaluate(M_REF, H_REF, math.radians(25.0), s).CN for s in (1, 2))
    )
    ax.set_xlim(0.0, 25.0)
    ax.set_ylim(0.0, top)
    ax.set_xlabel("alpha, deg")
    ax.set_ylabel("CN on S_ref of that stage")
    ax.set_title(
        f"(c) normal force at M {M_REF:g}, h = 15 km.\n"
        "The strake load is almost all vortex lift",
        loc="left",
    )
    ax.legend(loc="upper left", handlelength=1.8)


def _panel_xcp(ax, aero: StackAero, aero_off: StackAero, mach) -> None:
    """x_cp in calibres against Mach, both configurations, strakes on and off."""
    for stage, colour in ((1, INK), (2, ACCENT)):
        d = aero.D_ref(stage)
        on = np.array(
            [aero.evaluate(float(m), H_REF, math.radians(10.0), stage).x_cp / d for m in mach]
        )
        off = np.array(
            [
                aero_off.evaluate(float(m), H_REF, math.radians(10.0), stage).x_cp / d
                for m in mach
            ]
        )
        ax.plot(mach, on, color=colour, lw=1.4, label=f"stage {stage}, strakes on")
        ax.plot(mach, off, color=colour, lw=1.0, ls="--", label=f"stage {stage}, strakes off")
        ax.fill_between(mach, on, off, color=colour, alpha=0.12, lw=0)

    x_on = aero.evaluate(M_REF, H_REF, math.radians(10.0), 1).x_cp / aero.D_ref(1)
    shift = x_on - aero_off.evaluate(M_REF, H_REF, math.radians(10.0), 1).x_cp / aero.D_ref(1)
    ax.annotate(
        f"strakes move x_cp FORWARD by\n{abs(shift):.2f} calibres on the stack, so\n"
        "they REDUCE static margin",
        xy=(M_REF, x_on), xytext=(3.5, 7.3), fontsize=6.5, color="#8a3a3a",
        ha="left", va="top",
        arrowprops={"arrowstyle": "-", "color": "#8a3a3a", "lw": 0.7},
    )
    ax.set_xlim(MACH_MIN, MACH_MAX)
    ax.set_ylim(5.0, 8.8)
    ax.set_xlabel("Mach")
    ax.set_ylabel("x_cp / D_ref, from the nose tip")
    ax.set_title(
        "(d) centre of pressure at alpha = 10 deg.\nNote D_ref itself changes at separation",
        loc="left",
    )
    ax.legend(loc="lower left", ncol=2, handlelength=1.6, columnspacing=0.9)


def _panel_coefficient_validation(ax, table) -> None:
    """Suction-analogy coefficients against the printed NASA TN D-7921 Table III."""
    ar = np.linspace(0.0, 1.05, 400)
    ax.plot(
        ar, [polhamus_kv(a) for a in ar], color=INK, lw=1.4,
        label="model K_v = K_v,le + K_v,se",
    )
    ax.plot(
        ar, [polhamus_kv_se(a) for a in ar], color="#6a4c93", lw=1.1, ls="--",
        label="model K_v,se",
    )
    ax.plot(
        ar, [polhamus_kv_le(a) for a in ar], color="#5c8001", lw=1.1, ls="-.",
        label="model K_v,le = pi A / 4",
    )
    ax.plot(ar, [polhamus_kp(a) for a in ar], color=ACCENT, lw=1.1, label="model K_p = pi A / 2")

    t_ar = np.array([r[0] for r in table])
    ax.plot(
        t_ar, [r[3] for r in table], "s", ms=4.5, mfc="none", mec="#6a4c93", mew=1.1,
        label="TN D-7921 Table III, K_v,se",
    )
    ax.plot(
        t_ar, [r[2] for r in table], "^", ms=4.5, mfc="none", mec="#5c8001", mew=1.1,
        label="TN D-7921 Table III, K_v,le",
    )
    ax.plot(
        t_ar, [r[1] for r in table], "o", ms=4.5, mfc="none", mec="#c1121f", mew=1.1,
        label="TN D-7921 Table III, K_p",
    )
    ax.axhline(math.pi, color=GREY, lw=0.8, ls=":")
    ax.text(0.62, math.pi + 0.04, "pi", ha="left", va="bottom", fontsize=6.5, color=GREY)

    # The band a strake actually occupies, from the SPEC_IV1.md section 4 bounds.
    ax.axvspan(0.0136, 0.20, color=ACCENT, alpha=0.14, lw=0)
    ax.text(
        0.105, 0.35, "strake pair\naspect ratio,\nwhole design box", ha="center", va="bottom",
        fontsize=6.0, color="#5c6b00",
    )
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 4.15)
    ax.set_xlabel("aspect ratio A of the surface")
    ax.set_ylabel("suction-analogy coefficient")
    ax.set_title(
        "(e) validation: rectangular-wing K_p, K_v,le, K_v,se\nagainst a printed table",
        loc="left",
    )
    ax.legend(loc="upper right", handlelength=1.6, fontsize=6.2, ncol=2, columnspacing=0.9)


def _panel_configuration_validation(ax, matched, jorgensen_model) -> None:
    """Measured strake increment for a body with side strakes against the model's strake term."""
    model = jorgensen_model()
    alpha_deg = np.linspace(0.0, 60.0, 200)
    modelled = np.array(
        [sum(model.CN_strakes(2.0, math.radians(float(a)), 1)) for a in alpha_deg]
    )
    ax.plot(alpha_deg, modelled, color=INK, lw=1.4, label="this model, strake term only")

    marks = {0.6: ("o", "#c1121f"), 2.0: ("s", "#3d5a80")}
    seen = set()
    for mach, a_deg, cn_on, cn_off in matched:
        marker, colour = marks[mach]
        label = None if mach in seen else f"TM X-3130 measured, M {mach:g}"
        seen.add(mach)
        ax.errorbar(
            a_deg, cn_on - cn_off, yerr=0.4, fmt=marker, ms=5.0, mfc="none", mec=colour,
            mew=1.1, ecolor=colour, elinewidth=0.8, capsize=2.0, label=label,
        )

    ax.axvspan(25.0, 60.0, color=GREY, alpha=0.12, lw=0)
    ax.text(
        42.0, 0.35, "outside the declared 25 deg envelope", ha="center", va="bottom",
        fontsize=6.0, color=GREY,
    )
    ax.annotate(
        "measured increment is 1.4 to 3.4x the model,\n"
        "because a strake also raises the BODY load.\n"
        "Not modelled, so CN_max is CONSERVATIVE.",
        xy=(58.0, 5.6), xytext=(2.0, 4.1), fontsize=6.5, color="#8a3a3a", ha="left", va="top",
        arrowprops={"arrowstyle": "-", "color": "#8a3a3a", "lw": 0.7},
    )
    ax.set_xlim(0.0, 60.0)
    ax.set_ylim(0.0, 6.8)
    ax.set_xlabel("alpha, deg")
    ax.set_ylabel("strake CN increment, on the body base area")
    ax.set_title(
        "(f) validation: body-with-strakes increment,\nNASA TM X-3130 figs 18(a) and 22(a)",
        loc="left",
    )
    ax.legend(loc="upper left", handlelength=1.6)


def make_figure(out_path: str | None = None) -> str:
    """Build the figure and return the path written."""
    table, matched, jorgensen_model = _reference_data()

    reqs = InterceptRequirements()
    dv = default_iv1()
    aero = StackAero(dv, reqs)
    aero_off = StackAero(_without_strakes(dv), reqs)
    mach = np.arange(MACH_MIN, MACH_MAX + 1e-9, 0.01)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 2, figsize=(12.6, 12.4))
        _panel_cd0_breakdown(axes[0][0], aero, 1, mach, "a")
        _panel_cd0_breakdown(axes[0][1], aero, 2, mach, "b")
        _panel_cn_alpha(axes[1][0], aero, reqs)
        _panel_xcp(axes[1][1], aero, aero_off, mach)
        _panel_coefficient_validation(axes[2][0], table)
        _panel_configuration_validation(axes[2][1], matched, jorgensen_model)

        fig.suptitle(
            "IV-1 two-stage strake-stabilised aerodynamic build-up.\n"
            "Strake normal force by the Polhamus suction analogy with the Lamar side-edge term. "
            "References: NASA TN D-7921 Table III (Lamar and Gloss, 1975),\n"
            "a printed table; and NASA TM X-3130 (Jorgensen and Nelson, 1975), figures "
            "digitised. The inherited single-body physics is validated in fig_aero.py.",
            fontsize=7.5, color=GREY, y=0.995,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.966))

        if out_path is None:
            out_dir = os.path.join(RUNS_DIR, "_aero_iv1")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "aero_iv1_validation.png")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    return out_path


if __name__ == "__main__":
    print(make_figure())
