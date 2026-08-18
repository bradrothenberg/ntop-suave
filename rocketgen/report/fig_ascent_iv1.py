"""IV-1 figure: the converged ascent, with every staging event marked.

Source data: `runs/IV-1/figures/evidence_iv1.json`, key `trajectory`.

This is the IV-1 equivalent of the SV-1 trajectory figure. It is a separate script rather than a
call into `fig_trajectory.plot_trajectory`, because that script labels the terminal-Mach line
"R6", which is the SV-1 requirement. The IV-1 requirement is A4, and a report must not print the
wrong requirement identifier.

Four panels: the flight profile, Mach against time, dynamic pressure against time, and the mass
programme. The mass panel is the one that is new for a two-stage vehicle: mass leaves the vehicle
in one step at separation.

Run:
    .venv/Scripts/python.exe -m rocketgen.report.fig_ascent_iv1
"""
from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt

from .figstyle import BAD, GOOD, GREY, INK, STYLE
from .figstyle_iv1 import PHASE_COLOUR, PHASE_LABEL, evidence, out_path

PHASES = ("stage_1_boost", "separation_coast", "stage_2_boost", "midcourse_coast")

#: Events worth a vertical rule on the time-history panels.
RULES = ("pitchover", "stage_1_burnout", "separation", "stage_2_burnout")


def _phase_slices(phase: list[str]) -> list[tuple[str, int, int]]:
    """Contiguous runs of one phase name, as (name, start, stop) index pairs."""
    out: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(phase) + 1):
        if i == len(phase) or phase[i] != phase[start]:
            out.append((phase[start], start, min(i + 1, len(phase))))
            start = i
    return out


def make_figure(path: str | None = None) -> str:
    ev = evidence()
    tr = ev["trajectory"]
    t = np.array(tr["time"])
    x_km = np.array(tr["x"]) / 1000.0
    h_km = np.array(tr["h"]) / 1000.0
    mach = np.array(tr["mach"])
    q_kPa = np.array(tr["q"]) / 1000.0
    mass = np.array(tr["mass"])
    runs = _phase_slices(tr["phase"])

    q_limit = next(c["limit"] for c in ev["constraints"] if c["name"] == "A10 q_max [Pa]")
    m_limit = next(c["limit"] for c in ev["constraints"] if c["name"] == "A4 intercept Mach")
    events = {e["name"]: e for e in tr["events"]}

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.2))
        (ax_p, ax_m), (ax_q, ax_w) = axes

        # ---- (a) flight profile ----
        seen = set()
        for name, lo, hi in runs:
            ax_p.plot(
                x_km[lo:hi], h_km[lo:hi], color=PHASE_COLOUR[name], lw=2.0,
                label=None if name in seen else PHASE_LABEL[name],
            )
            seen.add(name)
        for name in RULES:
            e = events.get(name)
            if e is None:
                continue
            i = int(np.argmin(np.abs(t - e["time"])))
            ax_p.plot([x_km[i]], [h_km[i]], "o", ms=4.0, color=INK)
        sep = events["separation"]
        i_sep = int(np.argmin(np.abs(t - sep["time"])))
        ax_p.annotate(
            "separation at t = %.1f s\n%.1f km, Mach %.2f, -%.1f kg"
            % (sep["time"], sep["altitude"] / 1000.0, sep["mach"],
               sep["mass_before"] - sep["mass_after"]),
            xy=(x_km[i_sep], h_km[i_sep]), xytext=(18.0, 12.0), fontsize=6.4, color=INK,
            ha="left", va="center",
            arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.8},
        )
        ax_p.plot([x_km[-1]], [h_km[-1]], "*", ms=12.0, color="#8a3a3a")
        ax_p.annotate(
            "intercept: slant %.1f km, h %.1f km"
            % (np.hypot(x_km[-1], h_km[-1]), h_km[-1]),
            xy=(x_km[-1], h_km[-1]), xytext=(-14.0, 16.0), textcoords="offset points",
            fontsize=6.4, color="#8a3a3a", ha="right", va="bottom",
        )
        ax_p.axhline(0.0, color=GREY, lw=0.7)
        ax_p.set_xlabel("ground range [km]")
        ax_p.set_ylabel("altitude [km]")
        ax_p.set_ylim(-2.0, 46.0)
        ax_p.set_title("(a) flight profile", loc="left")
        ax_p.legend(loc="lower right", fontsize=6.6, handlelength=1.6)

        # ---- (b) Mach ----
        for name, lo, hi in runs:
            ax_m.plot(t[lo:hi], mach[lo:hi], color=PHASE_COLOUR[name], lw=1.6)
        ax_m.axhline(m_limit, color=GOOD, lw=1.0, ls="--")
        ax_m.text(
            t[-1], m_limit + 0.08, "A4 minimum, Mach %.2f" % m_limit, fontsize=6.4,
            color=GOOD, ha="right", va="bottom",
        )
        ax_m.plot([t[-1]], [mach[-1]], "*", ms=11.0, color="#8a3a3a")
        ax_m.text(
            t[-1] - 3.0, mach[-1] + 0.15, "Mach %.2f at intercept" % mach[-1],
            fontsize=6.4, color="#8a3a3a", ha="right", va="bottom",
        )
        ax_m.set_xlabel("time [s]")
        ax_m.set_ylabel("Mach")
        ax_m.set_ylim(0.0, 5.6)
        ax_m.set_title("(b) Mach number", loc="left")

        # ---- (c) dynamic pressure ----
        for name, lo, hi in runs:
            ax_q.plot(t[lo:hi], q_kPa[lo:hi], color=PHASE_COLOUR[name], lw=1.6)
        ax_q.axhline(q_limit / 1000.0, color=BAD, lw=1.0, ls="--")
        ax_q.text(
            t[-1], q_limit / 1000.0 - 8.0, "A10 limit, %.0f kPa" % (q_limit / 1000.0),
            fontsize=6.4, color=BAD, ha="right", va="top",
        )
        ax_q.plot([tr["q_max_time_s"]], [tr["q_max_Pa"] / 1000.0], "s", ms=5.0,
                  mfc="none", mec=BAD, mew=1.3)
        ax_q.annotate(
            "peak %.0f kPa at t = %.1f s, h = %.2f km"
            % (tr["q_max_Pa"] / 1000.0, tr["q_max_time_s"], tr["q_max_altitude_m"] / 1000.0),
            xy=(tr["q_max_time_s"], tr["q_max_Pa"] / 1000.0), xytext=(38.0, 250.0),
            fontsize=6.4, color=BAD, ha="left", va="center",
            arrowprops={"arrowstyle": "->", "color": BAD, "lw": 0.8},
        )
        ax_q.set_xlabel("time [s]")
        ax_q.set_ylabel("dynamic pressure [kPa]")
        ax_q.set_ylim(0.0, 380.0)
        ax_q.set_title("(c) dynamic pressure", loc="left")

        # ---- (d) mass programme ----
        for name, lo, hi in runs:
            ax_w.plot(t[lo:hi], mass[lo:hi], color=PHASE_COLOUR[name], lw=1.6)
        ax_w.annotate(
            "%.1f kg leaves the vehicle\nin one step at separation"
            % (sep["mass_before"] - sep["mass_after"]),
            xy=(sep["time"], 0.5 * (sep["mass_before"] + sep["mass_after"])),
            xytext=(42.0, 470.0), fontsize=6.4, color=INK, ha="left", va="center",
            arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.8},
        )
        ax_w.set_xlabel("time [s]")
        ax_w.set_ylabel("vehicle mass [kg]")
        ax_w.set_ylim(200.0, 640.0)
        ax_w.set_title("(d) mass programme", loc="left")

        for ax in (ax_m, ax_q, ax_w):
            for name in RULES:
                e = events.get(name)
                if e is not None:
                    ax.axvline(e["time"], color=GREY, lw=0.6, ls=":")

        fig.text(
            0.008, 0.985,
            "Converged IV-1. Launch mass %.1f kg, slant range %.1f km at t = %.1f s, "
            "intercept at %.2f km and Mach %.2f, peak dynamic pressure %.0f kPa.\n   "
            "Dotted vertical rules mark pitchover, stage-1 burnout, separation and stage-2 "
            "burnout. The separation coast lasts %.1f s, so it is a short grey segment. "
            "Source: runs/IV-1/converged.json."
            % (ev["mass"]["m0_kg"], np.hypot(x_km[-1], h_km[-1]), t[-1], h_km[-1], mach[-1],
               tr["q_max_Pa"] / 1000.0, tr["t_separation_s"] - tr["t_burnout_1_s"]),
            fontsize=6.8, color=GREY, va="top", ha="left",
        )
        fig.subplots_adjust(left=0.070, right=0.985, top=0.885, bottom=0.085,
                            hspace=0.36, wspace=0.22)
        path = path or out_path("ascent_iv1.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


if __name__ == "__main__":
    print(make_figure())
