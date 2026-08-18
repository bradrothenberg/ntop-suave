"""Shared matplotlib style and paths for the WP7 report figures.

Every figure script imports from here so the report has one visual language. The style matches
`rocketgen/report/fig_aero.py` and `fig_trajectory.py`, which were written first.

Run any figure script as a module, for example:
    .venv/Scripts/python.exe -m rocketgen.report.fig_carpet
"""
from __future__ import annotations

import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")

from ..config import RUNS_DIR  # noqa: E402

#: Which SV-1 study the figure scripts read. `select_study("spline")` re-points every path at
#: `runs/SV-1_spline`, so the same figure code serves both outer-mould-line families and the
#: ogive figures are never overwritten. Same pattern as `scripts/build_example.py::select_study`.
OML = "ogive"

CASE_DIR = os.path.join(RUNS_DIR, "SV-1")
CONVERGED_DIR = os.path.join(CASE_DIR, "converged")
DOE_DIR = os.path.join(CASE_DIR, "doe")
FIG_DIR = os.path.join(CASE_DIR, "figures")


def select_study(oml: str) -> None:
    """Point every figure path at the ogive study or the spline study.

    Call this BEFORE any figure module reads a path. Modules must therefore call the accessor
    functions below rather than importing `DOE_DIR` and friends by value: a `from .figstyle
    import DOE_DIR` binds the string at import time and would not see this rebinding.
    """
    global OML, CASE_DIR, CONVERGED_DIR, DOE_DIR, FIG_DIR
    if oml not in ("ogive", "spline"):
        raise ValueError(f"unknown oml family {oml!r}")
    OML = oml
    CASE_DIR = os.path.join(RUNS_DIR, "SV-1_spline" if oml == "spline" else "SV-1")
    CONVERGED_DIR = os.path.join(CASE_DIR, "converged")
    DOE_DIR = os.path.join(CASE_DIR, "doe")
    FIG_DIR = os.path.join(CASE_DIR, "figures")


def case_dir() -> str:
    return CASE_DIR


def converged_dir() -> str:
    return CONVERGED_DIR


def doe_dir() -> str:
    return DOE_DIR


def fig_dir() -> str:
    return FIG_DIR


def source_label(name: str) -> str:
    """Path of a run artefact as it should be QUOTED in a figure footer, relative to the repo."""
    return f"runs/{os.path.basename(CASE_DIR)}/{name}"


#: Same rcParams as fig_aero.py, so the whole report shares one look.
STYLE: dict[str, Any] = {
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
ACCENT = "#8a9a00"            # nTop accent, muted for print
GREY = "#7a7a7a"
GOOD = "#2e7d32"
BAD = "#c1121f"
WARN = "#e07b00"
COOL = "#3d5a80"

#: Mass-statement provenance colours. One colour per provenance, used in the mass figure and
#: quoted in the report text.
PROVENANCE_COLOUR: dict[str, str] = {
    "ntop_measured": "#8a9a00",
    "analytic": "#3d5a80",
    "requirement": "#1c1c1c",
    "correlation": "#c98b2e",
}
PROVENANCE_LABEL: dict[str, str] = {
    "ntop_measured": "nTop measured",
    "analytic": "analytic",
    "requirement": "requirement",
    "correlation": "correlation",
}


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def point_ntop() -> dict[str, Any]:
    return load_json(os.path.join(CONVERGED_DIR, "point_ntop.json"))


def point_analytic() -> dict[str, Any]:
    return load_json(os.path.join(CONVERGED_DIR, "point_analytic.json"))


def measurements() -> dict[str, Any]:
    return load_json(os.path.join(CONVERGED_DIR, "measurements.json"))


def sensitivity() -> dict[str, Any]:
    return load_json(os.path.join(DOE_DIR, "sensitivity.json"))


def lhs_meta() -> dict[str, Any]:
    return load_json(os.path.join(DOE_DIR, "lhs.json"))


def evidence() -> dict[str, Any]:
    return load_json(os.path.join(FIG_DIR, "evidence.json"))


def grid_rows() -> list[dict[str, str]]:
    import csv

    with open(os.path.join(DOE_DIR, "grid.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def out_path(name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, name)
