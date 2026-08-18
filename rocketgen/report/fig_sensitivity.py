"""WP7 figure: Spearman rank correlations from the Latin-hypercube trade study.

Source data: runs/SV-1/doe/sensitivity.json, written by `run_sv1.py --stage doe` from the
40-sample Latin hypercube over eight design variables. Nothing is recomputed here.

One panel per response. Bars are sorted by absolute correlation, so each panel reads as a
tornado. Positive and negative signs get different colours because the sign is the engineering
content: a positive correlation on q_max means the variable pushes the design towards the
dynamic-pressure limit.

Run:
    .venv/Scripts/python.exe -m rocketgen.report.fig_sensitivity
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from .figstyle import (
    BAD,
    COOL,
    GREY,
    STYLE,
    lhs_meta,
    out_path,
    sensitivity,
    source_label,
)

RESPONSE_LABEL: dict[str, str] = {
    "m0_kg": "(a) launch mass m0",
    "range_km": "(b) range",
    "mach_terminal": "(c) impact Mach",
    "q_max_kPa": "(d) maximum dynamic pressure",
}


def make_figure(path: str | None = None) -> str:
    sens = sensitivity()
    meta = lhs_meta()
    n_samples = meta.get("n_total")
    n_failed = meta.get("n_failed")
    responses = [r for r in RESPONSE_LABEL if r in sens]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.0))
        for ax, response in zip(axes.ravel(), responses):
            items = sorted(sens[response].items(), key=lambda kv: abs(kv[1]))
            names = [k for k, _v in items]
            values = [v for _k, v in items]
            colours = [COOL if v >= 0.0 else BAD for v in values]
            ypos = range(len(names))
            ax.barh(list(ypos), values, color=colours, height=0.62, edgecolor="none")
            ax.axvline(0.0, color="#4d4d4d", lw=0.8)
            ax.set_yticks(list(ypos))
            ax.set_yticklabels(names, fontsize=7.0)
            ax.set_xlim(-1.0, 1.0)
            ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax.set_xlabel("Spearman rank correlation")
            ax.set_title(RESPONSE_LABEL[response], loc="left")
            ax.grid(axis="y", visible=False)
            for y, v in zip(ypos, values):
                offset = 0.03 if v >= 0.0 else -0.03
                ax.text(
                    v + offset, y, "%+.2f" % v, va="center",
                    ha="left" if v >= 0.0 else "right", fontsize=6.2, color="#3a3a3a",
                )

        handles = [
            plt.Rectangle((0, 0), 1, 1, fc=COOL, ec="none", label="positive correlation"),
            plt.Rectangle((0, 0), 1, 1, fc=BAD, ec="none", label="negative correlation"),
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=2, fontsize=7.0,
            bbox_to_anchor=(0.5, 0.002),
        )
        fig.text(
            0.012, 0.982,
            "Sensitivity from the %s-sample Latin hypercube over eight variables "
            "(%s failed samples). Source: %s."
            % (n_samples, n_failed, source_label("doe/sensitivity.json")),
            fontsize=7.0, color=GREY, va="top", ha="left",
        )
        fig.subplots_adjust(left=0.115, right=0.985, top=0.905, bottom=0.115, hspace=0.55,
                            wspace=0.30)
        path = path or out_path("sensitivity.png")
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
