"""Figure: the flown mission of the converged SV-1, and what the coupling did to it.

    .venv/Scripts/python.exe -m rocketgen.report.fig_flight --oml spline

Four panels, all read from the recorded trajectory history. Nothing is re-flown here.

  (a) altitude against ground range, coloured by mission phase.
  (b) Mach against time, with the terminal requirement drawn.
  (c) dynamic pressure against time, with the structural limit drawn. This is the panel that
      shows how little margin the spline design keeps on q_max.
  (d) mass against time, so the propellant leaving the vehicle is visible.

Every panel draws the nTop-MEASURED point as a solid line and the same design vector flown with
ANALYTIC geometry as a dashed line. The gap between them is the coupling, and on this vehicle it
decides feasibility.

Source data: runs/SV-1_spline/converged/point_ntop.json and point_analytic.json.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from .figstyle import (
    BAD,
    GREY,
    INK,
    STYLE,
    out_path,
    point_analytic,
    point_ntop,
    select_study,
    source_label,
)

#: One colour per mission phase. Same idea as `figstyle_iv1.PHASE_COLOUR`, so the two reports
#: read as one document.
PHASE_COLOUR: dict[str, str] = {
    "separation": "#7a7a7a",
    "boost": "#c1121f",
    "sustain": "#3d5a80",
    "terminal": "#e07b00",
    "terminal_boost": "#8a9a00",
}
PHASE_LABEL: dict[str, str] = {
    "separation": "separation",
    "boost": "boost",
    "sustain": "sustain cruise",
    "terminal": "terminal dive",
    "terminal_boost": "terminal boost",
}


def _series(point: dict, key: str) -> list[float]:
    return [float(r[key]) for r in point["trajectory"]["history"]]


def _phase_runs(point: dict) -> list[tuple[str, int, int]]:
    """Contiguous index runs of one phase, as (phase, start, stop_inclusive)."""
    phases = [r["phase"] for r in point["trajectory"]["history"]]
    runs: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i] != phases[start]:
            runs.append((phases[start], start, i - 1))
            start = i
    return runs


def make_figure(path: str | None = None) -> str:
    ntop = point_ntop()
    analytic = point_analytic()

    q_limit_kPa = next(
        (c["limit"] for c in ntop["constraints"] if c["name"] == "q_max"), None
    )
    m_terminal_min = next(
        (c["limit"] for c in ntop["constraints"] if c["name"] == "R6 terminal Mach"), None
    )

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.4))
        ax_a, ax_b = axes[0]
        ax_c, ax_d = axes[1]

        # ---- (a) the flight path, coloured by phase ---------------------------------
        x = [v / 1000.0 for v in _series(ntop, "x")]
        h = [v / 1000.0 for v in _series(ntop, "h")]
        drawn: set[str] = set()
        for phase, i0, i1 in _phase_runs(ntop):
            colour = PHASE_COLOUR.get(phase, GREY)
            label = PHASE_LABEL.get(phase, phase) if phase not in drawn else None
            drawn.add(phase)
            ax_a.plot(x[i0:i1 + 2], h[i0:i1 + 2], color=colour, lw=1.6, label=label)
        xa = [v / 1000.0 for v in _series(analytic, "x")]
        ha = [v / 1000.0 for v in _series(analytic, "h")]
        ax_a.plot(xa, ha, color=GREY, lw=0.9, ls="--", label="analytic geometry")
        ax_a.set_xlabel("ground range [km]")
        ax_a.set_ylabel("altitude [km]")
        ax_a.set_title("(a) flight path, by mission phase", loc="left", fontsize=8.4)
        ax_a.legend(loc="lower left", fontsize=6.3, handlelength=1.8, ncol=2)

        # ---- (b) Mach ----------------------------------------------------------------
        ax_b.plot(_series(ntop, "t"), _series(ntop, "mach"), color=INK, lw=1.4,
                  label="nTop-measured geometry")
        ax_b.plot(_series(analytic, "t"), _series(analytic, "mach"), color=GREY, lw=0.9,
                  ls="--", label="analytic geometry")
        if m_terminal_min is not None:
            ax_b.axhline(m_terminal_min, color=BAD, lw=0.9, ls=":",
                         label="R6 terminal Mach %.2f" % m_terminal_min)
        ax_b.set_xlabel("time [s]")
        ax_b.set_ylabel("Mach")
        ax_b.set_title("(b) Mach", loc="left", fontsize=8.4)
        ax_b.legend(loc="lower left", fontsize=6.3, handlelength=1.8)

        # ---- (c) dynamic pressure ----------------------------------------------------
        q = [v / 1000.0 for v in _series(ntop, "q")]
        qa = [v / 1000.0 for v in _series(analytic, "q")]
        ax_c.plot(_series(ntop, "t"), q, color=INK, lw=1.4, label="nTop-measured geometry")
        ax_c.plot(_series(analytic, "t"), qa, color=GREY, lw=0.9, ls="--",
                  label="analytic geometry")
        if q_limit_kPa is not None:
            ax_c.axhline(q_limit_kPa / 1000.0, color=BAD, lw=0.9, ls=":",
                         label="structural limit %.0f kPa" % (q_limit_kPa / 1000.0))
        ax_c.set_xlabel("time [s]")
        ax_c.set_ylabel("dynamic pressure [kPa]")
        ax_c.set_title("(c) dynamic pressure, the active constraint", loc="left", fontsize=8.4)
        ax_c.legend(loc="upper left", fontsize=6.3, handlelength=1.8)
        # The recorded scalar, not the maximum of the plotted trace. The history is
        # downsampled from 15825 integration steps, so its maximum is lower than the peak the
        # integrator saw and the constraint was tested against.
        ax_c.annotate(
            "recorded peak %.1f kPa, measured\nrecorded peak %.1f kPa, analytic: over the limit"
            % (ntop["trajectory"]["q_max_Pa"] / 1000.0,
               analytic["trajectory"]["q_max_Pa"] / 1000.0),
            (0.97, 0.42), xycoords="axes fraction", ha="right", fontsize=6.3, color=GREY,
        )

        # ---- (d) mass ----------------------------------------------------------------
        ax_d.plot(_series(ntop, "t"), _series(ntop, "mass"), color=INK, lw=1.4,
                  label="nTop-measured geometry")
        ax_d.plot(_series(analytic, "t"), _series(analytic, "mass"), color=GREY, lw=0.9,
                  ls="--", label="analytic geometry")
        ax_d.set_xlabel("time [s]")
        ax_d.set_ylabel("vehicle mass [kg]")
        ax_d.set_title("(d) mass, propellant leaving the vehicle", loc="left", fontsize=8.4)
        ax_d.legend(loc="upper right", fontsize=6.3, handlelength=1.8)

        fig.text(
            0.012, 0.985,
            "The flown mission of the converged design. Solid: geometry measured in nTop. "
            "Dashed: the same design vector with the closed-form geometry.\nSource: %s and "
            "point_analytic.json." % source_label("converged/point_ntop.json"),
            fontsize=7.0, color=GREY, va="top", ha="left",
        )
        fig.subplots_adjust(left=0.075, right=0.985, top=0.885, bottom=0.105,
                            hspace=0.52, wspace=0.26)
        path = path or out_path("flight_path.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--oml", default="spline", choices=["ogive", "spline"],
                     help="which study to draw")
    select_study(_ap.parse_args().oml)
    print(make_figure())
