"""WP7 figure: the nTop cross-section area distribution S(x) against the closed form.

Source data: runs/SV-1/converged/measurements.json, the 16-station S(x) table that the nTop
notebook measured with the `extract_section` plus `body_surface_area<implicit_2d,real>` chain
(see docs/NTOP_NOTES.md section 24).

The closed form is built here from the design vector, in three parts:
  * tangent-ogive nose, cylindrical mid-body and conical boattail, as a circle of the local
    outer-mould-line radius;
  * the four constant-thickness plate fins, as `n_fin * t_fin` times the radial extent of the
    fin at that station.

The fin term is what makes the comparison a real test. It is not a constant: the fin leading
edge is swept, so between the root leading edge and the station where the full semi-span is
present the radial extent grows linearly.

Run:
    .venv/Scripts/python.exe -m rocketgen.report.fig_area
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from ..config import DesignVector
from .figstyle import (
    ACCENT,
    BAD,
    COOL,
    GREY,
    INK,
    STYLE,
    measurements,
    out_path,
    point_ntop,
    source_label,
)


def _spline_radius(length: float, radius: float, control, r0_over_r: float, s: float) -> float:
    """Radius of a splined run at fraction `s` of its length, m.

    The control points sit at the GREVILLE abscissae of the clamped knot vector, so the axial
    coordinate satisfies `x(u) = length * u` exactly. The curve parameter is therefore the
    length fraction, and no inverse solve is needed. `oml_spline.station_fractions` is where
    that choice is made; see `docs/NTOP_NOTES.md` section 25.
    """
    from ..oml_spline import SplineProfile

    p = SplineProfile(length=length, radius=radius, control=tuple(control),
                      r0_over_r=r0_over_r)
    return p.point_at(min(max(s, 0.0), 1.0))[1]


def oml_radius(dv: DesignVector, x: float) -> float:
    """Outer-mould-line radius at station x from the nose tip, m.

    Handles both shape families. A splined run is evaluated as the SAME B-spline that nTop
    revolves, read from `DesignVector.nose_control` and `DesignVector.boattail_control`, so
    this closed form and the measured solid describe one body rather than two.
    """
    R = 0.5 * dv.D
    if x <= 0.0:
        return 0.0
    if x < dv.L_nose:
        nose_control = getattr(dv, "nose_control", None)
        if nose_control is not None:
            return _spline_radius(dv.L_nose, R, nose_control, 0.0, x / dv.L_nose)
        rho = (R * R + dv.L_nose ** 2) / (2.0 * R)
        return max(math.sqrt(max(rho * rho - (dv.L_nose - x) ** 2, 0.0)) - (rho - R), 0.0)
    x_boat = dv.L_nose + dv.L_body_cyl
    if x <= x_boat:
        return R
    if x <= dv.L_total:
        r_base = 0.5 * dv.d_base
        boattail_control = getattr(dv, "boattail_control", None)
        if boattail_control is not None:
            # The boattail run is expressed on the CONTRACTION. `rocket_notebook` places its
            # control points at `R + (r_base - R) * c_i`, so in SplineProfile terms the run
            # END radius is r_base and it STARTS at R, i.e. r0_over_r = R / r_base > 1.
            return _spline_radius(dv.L_boattail, r_base, boattail_control, R / r_base,
                                  (x - x_boat) / dv.L_boattail)
        return R + (r_base - R) * (x - x_boat) / dv.L_boattail
    return 0.0


def fin_section_area(dv: DesignVector, x: float) -> float:
    """Cross-section area of the four plate fins at station x, m^2.

    The fin panel is a swept tapered trapezoid. At radial position r from the root the leading
    edge sits at `x_fin_le + r * tan(sweep)` and the chord is `c_r + (c_t - c_r) * r / b_fin`.
    The plate has constant thickness, so the section area is the thickness times the radial
    extent that the station cuts, times the number of panels.
    """
    if dv.b_fin <= 0.0:
        return 0.0
    samples = 4001
    inside = 0
    for i in range(samples):
        r = dv.b_fin * i / (samples - 1)
        x_le = dv.x_fin_le + r * math.tan(dv.sweep_fin)
        chord = dv.c_r_fin + (dv.c_t_fin - dv.c_r_fin) * r / dv.b_fin
        if x_le <= x <= x_le + chord:
            inside += 1
    extent = dv.b_fin * inside / samples
    return dv.n_fin * dv.t_fin * extent


def closed_form_area(dv: DesignVector, x: float) -> tuple[float, float]:
    """Return (body section area, fin section area) at station x, m^2."""
    r = oml_radius(dv, x)
    return math.pi * r * r, fin_section_area(dv, x)


def make_figure(path: str | None = None) -> str:
    point = point_ntop()
    fields = DesignVector.__dataclass_fields__
    dv = DesignVector(**{k: v for k, v in point["design_vector"].items() if k in fields})
    table = measurements()["area_distribution"]

    x_meas = np.array([row[0] for row in table])
    s_meas = np.array([row[1] for row in table])
    body_meas = np.array([closed_form_area(dv, float(x))[0] for x in x_meas])
    fin_meas = np.array([closed_form_area(dv, float(x))[1] for x in x_meas])
    s_closed = body_meas + fin_meas
    err = 100.0 * (s_meas / s_closed - 1.0)

    x_fine = np.linspace(0.0, dv.L_total, 601)
    body_fine = np.array([closed_form_area(dv, float(x))[0] for x in x_fine])
    fin_fine = np.array([closed_form_area(dv, float(x))[1] for x in x_fine])

    with plt.rc_context(STYLE):
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(9.4, 5.0), sharex=True, gridspec_kw={"height_ratios": [2.4, 1.0]}
        )
        ax.fill_between(
            x_fine, 0.0, body_fine, color=COOL, alpha=0.16, lw=0, label="closed form, body"
        )
        ax.fill_between(
            x_fine, body_fine, body_fine + fin_fine, color=ACCENT, alpha=0.35, lw=0,
            label="closed form, fin plates",
        )
        ax.plot(x_fine, body_fine + fin_fine, color=INK, lw=1.1, label="closed form, total")
        ax.plot(
            x_meas, s_meas, "o", ms=5.0, mfc="none", mec=BAD, mew=1.3,
            label="nTop measured, %d stations" % len(table),
        )
        ax.axvline(dv.L_nose, color=GREY, lw=0.7, ls=":")
        ax.axvline(dv.L_nose + dv.L_body_cyl, color=GREY, lw=0.7, ls=":")
        ax.axvline(dv.x_fin_le, color=GREY, lw=0.7, ls="--")
        ax.text(dv.L_nose, 0.004, " nose end", fontsize=6.3, color=GREY, va="bottom")
        ax.text(
            dv.L_nose + dv.L_body_cyl, 0.004, " boattail", fontsize=6.3, color=GREY, va="bottom"
        )
        ax.text(dv.x_fin_le, 0.062, " fin root LE", fontsize=6.3, color=GREY, va="bottom")
        ax.set_ylabel("cross-section area S(x) [m^2]")
        ax.set_ylim(0.0, 0.132)
        ax.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
        ax.set_title("(a) area distribution, converged SV-1", loc="left")
        ax.legend(loc="upper left", ncol=2, handlelength=1.6, columnspacing=1.2)

        ax2.axhline(0.0, color="#4d4d4d", lw=0.8)
        ax2.bar(x_meas, err, width=0.10, color=COOL, edgecolor="none")
        for x, e in zip(x_meas, err):
            ax2.text(
                x, e + (0.012 if e >= 0.0 else -0.012), "%+.2f" % e, ha="center",
                va="bottom" if e >= 0.0 else "top", fontsize=5.6, color="#3a3a3a",
            )
        ax2.set_xlabel("station from the nose tip [m]")
        ax2.set_ylabel("error [%]")
        ax2.set_xlim(0.0, dv.L_total)
        span = max(0.35, 1.5 * float(np.abs(err).max()))
        ax2.set_ylim(-span, span)
        ax2.set_title(
            "(b) nTop minus closed form, worst station %+.2f %%, mean %+.2f %%"
            % (err[np.argmax(np.abs(err))], err.mean()),
            loc="left",
        )

        fig.text(
            0.012, 0.982,
            "S(x) from the nTop notebook against the closed-form outer mould line plus plate "
            "fins.\nSource: %s." % source_label("converged/measurements.json"),
            fontsize=7.0, color=GREY, va="top", ha="left",
        )
        fig.subplots_adjust(left=0.085, right=0.985, top=0.875, bottom=0.095, hspace=0.30)
        path = path or out_path("area_distribution.png")
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
