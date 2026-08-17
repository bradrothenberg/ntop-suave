"""WP3 figure - trajectory summary plate.

`plot_trajectory` takes a `config.TrajectoryResult` and nothing else, so the coupled
sizing loop (WP5) and the report (WP7) can reuse it directly on a converged run.

Six panels:
    altitude vs ground range      the mission profile
    Mach vs time                  with the cruise and terminal requirements marked
    thrust and drag vs time       shows the boost, sustain and coast phases
    dynamic pressure vs time      with the SPEC.md section 4 structural limit marked
    mass vs time                  propellant expenditure
    angle of attack vs time       trim history, for fin authority

Style: monospaced, thin lines, no grid fill, no legend boxes, no chartjunk.

Run this module directly to write `runs/_traj/trajectory.png` from a demonstration
trajectory. The demonstration uses a PLACEHOLDER aerodynamic model when WP2's
`rocketgen.sizing.aero.RocketAero` is not importable; that placeholder is not a physics
model and exists only to exercise the plotting code.
"""
from __future__ import annotations

import os
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt   # noqa: E402

from ..config import RUNS_DIR, TrajectoryResult   # noqa: E402

DEFAULT_PATH = os.path.join(RUNS_DIR, "_traj", "trajectory.png")

_STYLE = {
    "font.family": "monospace",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "axes.titlesize": 8,
    "axes.titleweight": "bold",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}

_PHASE_COLOURS = {
    "separation": "#9aa0a6",
    "boost": "#c0392b",
    "sustain": "#d68910",
    "coast": "#2874a6",
    "terminal": "#1e8449",
    "terminal_boost": "#7d3c98",
    "ballistic": "#34495e",
    "fall": "#34495e",
}


def _tidy(axis: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.margins(x=0.02, y=0.06)


def _phase_spans(
    time: Sequence[float], phase: Sequence[str]
) -> list[tuple[str, float, float]]:
    """Contiguous runs of the same phase label, as (label, t_start, t_end)."""
    spans: list[tuple[str, float, float]] = []
    if not phase:
        return spans
    current = phase[0]
    start = time[0]
    for t, p in zip(time, phase):
        if p != current:
            spans.append((current, start, t))
            current = p
            start = t
    spans.append((current, start, time[-1]))
    return spans


def _shade_phases(axis: plt.Axes, result: TrajectoryResult) -> None:
    for label, t0, t1 in _phase_spans(result.time, result.phase):
        colour = _PHASE_COLOURS.get(label)
        if colour is None or t1 <= t0:
            continue
        axis.axvspan(t0, t1, color=colour, alpha=0.08, linewidth=0.0)


def plot_trajectory(
    result: TrajectoryResult,
    path: str = DEFAULT_PATH,
    title: str = "SV-1 trajectory",
    q_limit: float | None = 200_000.0,
    mach_cruise: float | None = 2.00,
    mach_terminal_min: float | None = 1.50,
) -> str:
    """Write the six-panel trajectory plate and return the path written.

    `q_limit`, `mach_cruise` and `mach_terminal_min` are drawn as reference lines. Pass
    None to omit any of them. Their default values are the SPEC.md section 2 and 4
    requirements R2, R6 and `Requirements.q_max`, which the coordinator raised to
    200 kPa after the 90 kPa figure was found to be mutually exclusive with R6.
    """
    if not result.time:
        raise ValueError("TrajectoryResult is empty, nothing to plot")

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    time = result.time
    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(3, 2, figsize=(9.0, 8.4))
        (ax_profile, ax_mach), (ax_force, ax_q), (ax_mass, ax_alpha) = axes

        # --- altitude vs range, coloured by phase ---
        for label, t0, t1 in _phase_spans(time, result.phase):
            indices = [i for i, t in enumerate(time) if t0 <= t <= t1]
            if len(indices) < 2:
                continue
            ax_profile.plot(
                [result.x[i] / 1000.0 for i in indices],
                [result.h[i] / 1000.0 for i in indices],
                color=_PHASE_COLOURS.get(label, "#333333"),
                label=label,
            )
        ax_profile.axhline(0.0, color="#666666", linewidth=0.5)
        _tidy(ax_profile, "ground range [km]", "altitude [km]", "flight profile")
        ax_profile.legend(loc="upper right", ncol=2)

        # --- Mach vs time ---
        _shade_phases(ax_mach, result)
        ax_mach.plot(time, result.mach, color="#154360")
        if mach_cruise is not None:
            ax_mach.axhline(mach_cruise, color="#c0392b", linewidth=0.6, linestyle="--")
            ax_mach.annotate(
                f"R2 cruise M {mach_cruise:.2f}",
                (time[-1], mach_cruise),
                textcoords="offset points",
                xytext=(-4, 3),
                ha="right",
                fontsize=6,
                color="#c0392b",
            )
        if mach_terminal_min is not None:
            ax_mach.axhline(
                mach_terminal_min, color="#1e8449", linewidth=0.6, linestyle="--"
            )
            ax_mach.annotate(
                f"R6 terminal M {mach_terminal_min:.2f}",
                (time[-1], mach_terminal_min),
                textcoords="offset points",
                xytext=(-4, -9),
                ha="right",
                fontsize=6,
                color="#1e8449",
            )
        _tidy(ax_mach, "time [s]", "Mach", "Mach number")

        # --- thrust and drag ---
        _shade_phases(ax_force, result)
        ax_force.plot(
            time, [f / 1000.0 for f in result.thrust], color="#c0392b", label="thrust"
        )
        ax_force.plot(
            time, [f / 1000.0 for f in result.drag], color="#2874a6", label="drag"
        )
        ax_force.set_yscale("symlog", linthresh=1.0)
        _tidy(ax_force, "time [s]", "force [kN]", "thrust and drag")
        ax_force.legend(loc="upper right")

        # --- dynamic pressure ---
        _shade_phases(ax_q, result)
        ax_q.plot(time, [q / 1000.0 for q in result.q], color="#6c3483")
        if q_limit is not None:
            ax_q.axhline(q_limit / 1000.0, color="#c0392b", linewidth=0.6, linestyle="--")
            ax_q.annotate(
                f"limit {q_limit / 1000.0:.0f} kPa",
                (time[-1], q_limit / 1000.0),
                textcoords="offset points",
                xytext=(-4, 3),
                ha="right",
                fontsize=6,
                color="#c0392b",
            )
        _tidy(ax_q, "time [s]", "q [kPa]", "dynamic pressure")

        # --- mass ---
        _shade_phases(ax_mass, result)
        ax_mass.plot(time, result.mass, color="#117a65")
        _tidy(ax_mass, "time [s]", "mass [kg]", "vehicle mass")

        # --- angle of attack ---
        _shade_phases(ax_alpha, result)
        ax_alpha.plot(
            time, [a * 57.29577951308232 for a in result.alpha], color="#884ea0"
        )
        ax_alpha.axhline(0.0, color="#666666", linewidth=0.5)
        _tidy(ax_alpha, "time [s]", "alpha [deg]", "trim angle of attack")

        subtitle = (
            f"range {result.range_final / 1000.0:.1f} km   "
            f"terminal M {result.mach_final:.2f}   "
            f"q_max {result.q_max / 1000.0:.1f} kPa   "
            f"converged {result.converged}"
        )
        figure.suptitle(f"{title}\n{subtitle}", fontsize=9, x=0.02, ha="left")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        figure.savefig(path)
        plt.close(figure)
    return path


# --------------------------------------------------------------------------------------
#   Demonstration entry point
# --------------------------------------------------------------------------------------


class _PlaceholderAero:
    """NOT A PHYSICS MODEL. Constant CD0, linear CN. Demonstration plotting only.

    WP2 owns the real build-up (`rocketgen.sizing.aero.RocketAero`). This class is used
    only when that module is not importable, so that running this file standalone still
    produces a figure.
    """

    CD0 = 0.40
    CN_ALPHA = 12.0
    INDUCED = 0.35

    def evaluate(self, mach, altitude, alpha, power_on=False):   # type: ignore[no-untyped-def]
        from ..config import AeroCoefficients

        cn = self.CN_ALPHA * alpha
        cd0 = self.CD0 - (0.06 if power_on else 0.0)
        cd = cd0 + self.INDUCED * cn * cn
        return AeroCoefficients(
            mach=mach,
            altitude=altitude,
            alpha=alpha,
            CD0=cd0,
            CD=cd,
            CN=cn,
            CN_alpha=self.CN_ALPHA,
            CM=0.0,
            x_cp=2.4,
            L_over_D=cn / cd if cd > 0 else 0.0,
        )

    def trim_alpha(self, mach, altitude, required_CN):   # type: ignore[no-untyped-def]
        return required_CN / self.CN_ALPHA


def _demo(path: str = DEFAULT_PATH) -> str:
    """Fly a demonstration SV-1 mission and plot it."""
    from ..config import DesignVector, MassStatement, Requirements
    from ..sizing.propulsion import SolidMotor
    from ..sizing.trajectory import Mission

    # A terminal-boost design point, because that is the configuration that can meet
    # SPEC R6. The terminal charge is traded out of the sustain charge.
    dv = DesignVector().replace(
        m_p_terminal=32.0, F_terminal=8000.0, m_p_sustain=228.0
    )
    reqs = Requirements()
    motor = SolidMotor(dv)
    motor.size_sustain_for_thrust(2600.0)

    try:
        from ..sizing.aero import RocketAero   # type: ignore

        aero: object = RocketAero(dv)
        aero_note = "WP2 RocketAero"
    except Exception:
        aero = _PlaceholderAero()
        aero_note = "PLACEHOLDER aero, not a physics model"

    mass = MassStatement()
    mass.add("seeker", 8.0, 0.5 * dv.L_seeker)
    mass.add("guidance", reqs.m_guidance, dv.L_seeker + 0.5 * dv.L_guidance)
    mass.add("warhead", reqs.m_warhead, dv.L_seeker + dv.L_guidance + 0.5 * dv.L_warhead)
    mass.add("airframe", 90.0, 0.5 * dv.L_total)
    mass.add("motor_inert", motor.inert_mass_breakdown()["recommended"], 0.75 * dv.L_total)
    mass.add("propellant", motor.propellant_mass, 0.70 * dv.L_total)

    mission = Mission(dv, reqs, motor, aero, mass, dive_rule="terminal_boost")   # type: ignore[arg-type]
    result = mission.fly(dt=0.02)
    written = plot_trajectory(result, path=path, title=f"SV-1 trajectory ({aero_note})")
    print(f"wrote {written}")
    print(f"message: {result.message}")
    return written


if __name__ == "__main__":
    _demo()
