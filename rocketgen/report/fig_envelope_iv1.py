"""IV-1 figure: the flight envelope in altitude against Mach.

Source data: `runs/IV-1/figures/evidence_iv1.json`, key `trajectory`. That record is the converged
trajectory, re-flown from `runs/IV-1/converged.json` and the nTop measurements already on disk,
and checked against the recorded intercept before it was written.

The iso-dynamic-pressure curves are evaluated here from `rocketgen.sizing.atmosphere`, the same
US Standard 1976 table the trajectory used. A curve of constant q is the altitude at which
0.5 * rho(h) * (Mach * a(h))^2 equals that q, so it is a property of the atmosphere alone and
carries no vehicle assumption.

What to look at: the ascent runs up the left of the frame at high dynamic pressure, peaks under
the booster and then leaves the high-q region entirely. The intercept sits high and fast, where
the dynamic pressure has collapsed. That is the whole reason the vehicle needs a divert motor.

Run:
    .venv/Scripts/python.exe -m rocketgen.report.fig_envelope_iv1
"""
from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt

from ..sizing.atmosphere import atmo
from .figstyle import BAD, GREY, INK, STYLE
from .figstyle_iv1 import PHASE_COLOUR, PHASE_LABEL, evidence, out_path

#: Iso-q curves to draw, Pa. The last one is the A10 limit and is drawn in red.
Q_LEVELS = (50.0e3, 100.0e3, 200.0e3, 300.0e3)

#: Events to mark, and where to put their label relative to the point.
EVENT_STYLE = {
    "pitchover": ("pitchover starts", (8, 16)),
    "pitchover_complete": ("pitchover complete", (-118, 30)),
    "stage_1_burnout": ("stage-1 burnout", (4, -22)),
    "separation": ("separation", (30, 30)),
    "stage_2_burnout": ("stage-2 burnout", (-84, 18)),
}


def iso_q(q: float, mach: np.ndarray) -> np.ndarray:
    """Altitude at which the dynamic pressure equals `q`, for each Mach number, m.

    Solved by bisection on altitude. q falls monotonically with altitude at fixed Mach, because
    density falls faster than the speed of sound changes, so the bisection is safe.
    """
    out = np.empty_like(mach)
    for i, m in enumerate(mach):
        lo, hi = 0.0, 60_000.0

        def q_at(h: float, m: float = float(m)) -> float:
            st = atmo(h)
            v = m * st.speed_of_sound
            return 0.5 * st.density * v * v

        if q_at(lo) < q:
            out[i] = np.nan
            continue
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if q_at(mid) > q:
                lo = mid
            else:
                hi = mid
        out[i] = 0.5 * (lo + hi)
    return out


def make_figure(path: str | None = None) -> str:
    ev = evidence()
    tr = ev["trajectory"]
    mach = np.array(tr["mach"])
    h_km = np.array(tr["h"]) / 1000.0
    phase = tr["phase"]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9.6, 5.2))

        grid = np.linspace(0.2, 5.6, 220)
        for k, q in enumerate(Q_LEVELS):
            hq = iso_q(q, grid) / 1000.0
            ax.plot(grid, hq, color=GREY, lw=0.7, ls=":")
            # Stagger the labels along the curves, or the 200 and 300 kPa labels land on top
            # of each other and on the stage-2 burnout marker.
            j = int((0.93 - 0.13 * k) * (len(grid) - 1))
            if np.isfinite(hq[j]):
                ax.text(
                    grid[j], hq[j] + 0.4, "%.0f kPa" % (q / 1000.0), fontsize=6.0,
                    color=GREY, ha="center", va="bottom", rotation=14,
                )
        q_limit = next(
            c["limit"] for c in ev["constraints"] if c["name"] == "A10 q_max [Pa]"
        )
        h_limit = iso_q(q_limit, grid) / 1000.0
        # Black, not red: the stage-1 boost trace is red, and two red curves in one frame read
        # as one curve.
        ax.plot(grid, h_limit, color=INK, lw=1.2, ls="--")
        j = int(0.95 * (len(grid) - 1))
        ax.text(
            grid[j], h_limit[j] - 0.5, "A10 limit, %.0f kPa" % (q_limit / 1000.0),
            fontsize=6.6, color=INK, ha="center", va="top", rotation=13,
        )

        # The flown path, one line segment set per phase so the colour carries the phase.
        for name in ("stage_1_boost", "separation_coast", "stage_2_boost", "midcourse_coast"):
            idx = [i for i, p in enumerate(phase) if p == name]
            if not idx:
                continue
            lo, hi = min(idx), max(idx) + 1
            ax.plot(mach[lo:hi], h_km[lo:hi], color=PHASE_COLOUR[name], lw=2.0,
                    label=PHASE_LABEL[name], solid_capstyle="round")

        # Peak dynamic pressure.
        ax.plot([tr["q_max_mach"]], [tr["q_max_altitude_m"] / 1000.0], "s", ms=6.0,
                mfc="none", mec=BAD, mew=1.5)
        ax.annotate(
            "peak q %.0f kPa\nat %.2f km, Mach %.2f, t = %.1f s"
            % (tr["q_max_Pa"] / 1000.0, tr["q_max_altitude_m"] / 1000.0, tr["q_max_mach"],
               tr["q_max_time_s"]),
            xy=(tr["q_max_mach"], tr["q_max_altitude_m"] / 1000.0),
            xytext=(0.35, 14.5), fontsize=6.6, color=BAD, ha="left", va="center",
            arrowprops={"arrowstyle": "->", "color": BAD, "lw": 0.8},
        )

        # Discrete events.
        for e in tr["events"]:
            style = EVENT_STYLE.get(e["name"])
            if style is None:
                continue
            label, (dx, dy) = style
            if e["name"] == "separation":
                label = "separation, -%.1f kg" % (e["mass_before"] - e["mass_after"])
            ax.plot([e["mach"]], [e["altitude"] / 1000.0], "o", ms=4.5, color=INK)
            ax.annotate(
                "%s\nt = %.1f s" % (label, e["time"]),
                xy=(e["mach"], e["altitude"] / 1000.0), textcoords="offset points",
                xytext=(dx, dy), fontsize=6.2, color=INK, ha="left", va="center",
            )

        # Apogee and intercept.
        i_apo = int(np.argmax(h_km))
        ax.plot([mach[i_apo]], [h_km[i_apo]], "^", ms=6.0, mfc="none", mec=INK, mew=1.3)
        ax.annotate(
            "apogee %.1f km at t = %.0f s" % (h_km[i_apo], tr["apogee_time_s"]),
            xy=(mach[i_apo], h_km[i_apo]), textcoords="offset points", xytext=(8, 6),
            fontsize=6.4, color=INK, ha="left", va="bottom",
        )

        ic = ev["lateral_g"]
        m_end, h_end = mach[-1], h_km[-1]
        ax.plot([m_end], [h_end], "*", ms=13.0, color="#8a3a3a")
        ax.annotate(
            "INTERCEPT at %.1f km, Mach %.2f, t = %.1f s\n"
            "q = %.0f kPa, so only %.2f g aerodynamically.\n"
            "The divert motor supplies %.2f g."
            % (h_end, m_end, tr["duration_s"], tr["q"][-1] / 1000.0,
               ic["aerodynamic_g"], ic["acs_g"]),
            xy=(m_end, h_end), xytext=(1.55, 28.5), fontsize=6.8, color="#8a3a3a",
            ha="left", va="top",
            arrowprops={"arrowstyle": "->", "color": "#8a3a3a", "lw": 0.9},
        )

        ax.set_xlim(0.0, 5.6)
        ax.set_ylim(0.0, 42.0)
        ax.set_xlabel("Mach")
        ax.set_ylabel("altitude [km]")
        ax.set_title(
            "IV-1 flight envelope: the flown path against lines of constant dynamic pressure",
            loc="left",
        )
        ax.legend(loc="lower right", fontsize=7.0, handlelength=1.8)
        fig.text(
            0.008, 0.982,
            "Dotted lines are constant dynamic pressure, evaluated from the same US Standard "
            "1976 table the trajectory used. The vehicle crosses its peak dynamic pressure "
            "early,\n   under the booster, and arrives at the intercept with almost none. "
            "Source: runs/IV-1/figures/evidence_iv1.json.",
            fontsize=6.8, color=GREY, va="top", ha="left",
        )
        fig.subplots_adjust(left=0.070, right=0.985, top=0.865, bottom=0.095)
        path = path or out_path("flight_envelope_iv1.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


if __name__ == "__main__":
    print(make_figure())
