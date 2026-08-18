"""Paths and shared loaders for the IV-1 report figures.

The visual language is the SV-1 one: `figstyle.STYLE` and the same palette. Only the paths and
the data loaders are new, so the two reports look like one document.

Run any figure script as a module, for example:
    .venv/Scripts/python.exe -m rocketgen.report.fig_envelope_iv1
"""
from __future__ import annotations

import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")

from ..config import RUNS_DIR  # noqa: E402

CASE_DIR = os.path.join(RUNS_DIR, "IV-1")
GEOM_DIR = os.path.join(CASE_DIR, "geom")
FIG_DIR = os.path.join(CASE_DIR, "figures")
EVIDENCE = os.path.join(FIG_DIR, "evidence_iv1.json")
CONVERGED = os.path.join(CASE_DIR, "converged.json")

#: One colour per mission phase, used by every IV-1 figure that draws the trajectory.
PHASE_COLOUR: dict[str, str] = {
    "stage_1_boost": "#c1121f",
    "separation_coast": "#7a7a7a",
    "stage_2_boost": "#e07b00",
    "midcourse_coast": "#3d5a80",
}
PHASE_LABEL: dict[str, str] = {
    "stage_1_boost": "stage-1 boost",
    "separation_coast": "separation coast",
    "stage_2_boost": "stage-2 boost",
    "midcourse_coast": "midcourse coast",
}

#: One colour per stage, for the mass figure.
STAGE_COLOUR: dict[int, str] = {1: "#c1121f", 2: "#3d5a80", 0: "#7a7a7a"}


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evidence() -> dict[str, Any]:
    """The evidence file. Build it first with `-m rocketgen.report.evidence_iv1`."""
    if not os.path.isfile(EVIDENCE):
        raise SystemExit(
            f"{EVIDENCE} is missing. Run:\n"
            "    .venv/Scripts/python.exe -m rocketgen.report.evidence_iv1"
        )
    return load_json(EVIDENCE)


def converged() -> dict[str, Any]:
    return load_json(CONVERGED)


def out_path(name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, name)
