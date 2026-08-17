"""Produce the SV-1 result set: converged design, nTop geometry, and the trade study.

Usage:

    .venv/Scripts/python.exe run_sv1.py --stage smoke      # sub-minute, proves the pipeline
    .venv/Scripts/python.exe run_sv1.py --stage size       # the sizing search
    .venv/Scripts/python.exe run_sv1.py --stage doe        # the trade study
    .venv/Scripts/python.exe run_sv1.py --stage all

PLAN.md hard rule 5: validate at small scale before scaling up. The `smoke` stage runs the whole
pipeline including one real nTop call, in well under a minute. Only the scale parameters change
between stages.

Everything lands under `runs/SV-1/`, and every artefact carries the inputs that produced it so the
report can be regenerated from disk without re-running the analysis.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from typing import Any

from rocketgen.config import (
    CD0_CALIBRATION,
    SOURCES,
    DesignVector,
    Requirements,
)
from rocketgen.doe import grid_samples, lhs_samples, run_doe, sensitivity
from rocketgen.sizing.loop import converge_point, size
from rocketgen.sizing.masses import PROPELLANT_ITEMS

OUT = os.path.join("runs", "SV-1")


# --------------------------------------------------------------------------------------
#   nTop geometry hook
# --------------------------------------------------------------------------------------


def get_geometry_fn(enabled: bool):
    """Return the nTop `measure_rocket` callable, or None.

    Imported lazily and defensively: the notebook module is the newest part of the project, and a
    failure to import it must degrade the run to analytics with a loud message, not abort it.
    """
    if not enabled:
        print("nTop geometry: DISABLED by flag, analytic geometry only")
        return None
    try:
        from rocketgen.ntopgen.rocket_notebook import measure_rocket
    except Exception as exc:                              # noqa: BLE001
        print(f"nTop geometry: UNAVAILABLE ({type(exc).__name__}: {exc})")
        print("              falling back to analytic geometry; results are NOT nTop-measured")
        return None
    print("nTop geometry: enabled")
    return measure_rocket


# --------------------------------------------------------------------------------------
#   Serialisation
# --------------------------------------------------------------------------------------


def dump_point(p, path: str) -> None:
    """Write one design point to JSON, including provenance."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: dict[str, Any] = {
        "design_vector": p.dv.as_dict(),
        "converged": p.converged,
        "feasible": p.feasible,
        "geometry_measured": p.geometry_measured,
        "iterations": p.iterations,
        "message": p.message,
        "wall_time_s": p.wall_time_s,
        "warnings": p.warnings,
        "summary": p.summary(),
        "constraints": [asdict(c) for c in p.constraints],
        "history": p.history,
    }
    if p.masses is not None:
        payload["mass_statement"] = {
            "total_kg": p.masses.total,
            "x_cg_m": p.masses.x_cg,
            "measured_fraction": p.masses.measured_fraction,
            "burnout_kg": p.masses.excluding(*PROPELLANT_ITEMS)[0],
            "burnout_x_cg_m": p.masses.excluding(*PROPELLANT_ITEMS)[1],
            "items": [
                {
                    "name": e.name,
                    "mass_kg": e.mass,
                    "x_cg_m": e.x_cg,
                    "provenance": e.provenance,
                    "note": e.note,
                }
                for e in p.masses.entries
            ],
        }
    if p.meas is not None:
        payload["ntop_measurements"] = asdict(p.meas)
    if p.traj is not None:
        t = p.traj
        payload["trajectory"] = {
            "range_m": t.range_final,
            "mach_final": t.mach_final,
            "q_max_Pa": t.q_max,
            "duration_s": t.time[-1] if t.time else 0.0,
            "n_steps": len(t.time),
            "message": t.message,
            "diagnostics": t.diagnostics,
            # decimated history, so the JSON stays readable
            "history": _decimate(t, 400),
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"wrote {path}")


def _decimate(t, n: int) -> list[dict[str, float]]:
    if not t.time:
        return []
    step = max(1, len(t.time) // n)
    out = []
    for i in range(0, len(t.time), step):
        out.append(
            {
                "t": t.time[i],
                "x": t.x[i],
                "h": t.h[i],
                "V": t.V[i],
                "mach": t.mach[i],
                "mass": t.mass[i],
                "gamma": t.gamma[i],
                "thrust": t.thrust[i],
                "drag": t.drag[i],
                "q": t.q[i],
                "alpha": t.alpha[i],
                "phase": t.phase[i] if i < len(t.phase) else "",
            }
        )
    return out


def dump_provenance(path: str, extra: dict[str, Any]) -> None:
    """Write the full source registry and environment, so the report can cite everything.

    Every module registers its sources at import time, so the registry is only complete once every
    module has been imported. `sizing.loop` imports `propulsion` and `trajectory` LAZILY, inside
    `converge_point`, so calling this before a trajectory has been flown silently omitted their
    entries. That dropped `prop.ideal_nozzle` (the largest declared optimism in the result) from
    the report's limitations table. Import them explicitly here so the registry is always whole.
    """
    import numpy
    import scipy

    from rocketgen.sizing import aero, atmosphere, masses, propulsion, trajectory  # noqa: F401

    n_sources = len(SOURCES)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "environment": {
            "python": sys.version,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
        "cd0_calibration": CD0_CALIBRATION,
        "requirements": asdict(Requirements()),
        "sources": dict(sorted(SOURCES.items())),
        **extra,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"wrote {path} ({n_sources} registered sources)")


# --------------------------------------------------------------------------------------
#   Stages
# --------------------------------------------------------------------------------------


def stage_smoke(reqs: Requirements, geometry_fn) -> None:
    """Sub-minute end-to-end proof. One design point, one nTop call, coarse integration."""
    print("\n=== SMOKE: one point, coarse ===")
    t0 = time.perf_counter()
    p = converge_point(
        DesignVector(),
        reqs,
        geometry_fn=geometry_fn,
        run_dir=os.path.join(OUT, "smoke", "ntop"),
        max_iter=2,
        dt=0.10,
        adaptive=True,
    )
    print(p.summary())
    print(f"geometry measured: {p.geometry_measured}")
    for w in p.warnings:
        print("  ! " + w)
    dump_point(p, os.path.join(OUT, "smoke", "point.json"))
    print(f"smoke stage took {time.perf_counter() - t0:.1f} s")


# Starting vector for the sizing search.
#
# It is NOT the bare `DesignVector()` default. That default has `m_p_terminal = 0`, and a design
# with no terminal boost cannot meet R6 at any dive angle: the unpowered dive is terminal-velocity
# limited to about Mach 0.93. Starting the search there wastes evaluations walking out of a region
# that is provably infeasible. `DesignVector()` keeps `m_p_terminal = 0` so the two-phase motor
# regression baseline stays exact, so the terminal charge is introduced here instead, taken out of
# the sustain charge rather than added to the total.
SIZING_START = DesignVector(
    m_p_sustain=232.0,
    m_p_terminal=28.0,
    F_terminal=8.0e3,
)

# The converged SV-1, as found by `--stage size` (55 evaluations) and re-verified with real nTop
# geometry in the loop: 554.3 kg, 189.3 km, impact Mach 1.65, q_max 193.5 kPa, all ten constraints
# met. Recorded here so `--stage doe` can be run on its own and still centre the trade study on the
# sized design.
#
# The trade study MUST be centred here, not on SIZING_START. Centred on SIZING_START the fins are
# the default 0.18 m semi-span, which fails R10 static margin at essentially every sample, so a
# 75-point factorial returned 0 feasible points and looked like a modelling failure when it was
# only a badly chosen base point.
SV1_CONVERGED = DesignVector(
    D=0.35,
    L_total=3.60,
    f_nose=3.4,
    m_p_boost=130.0,
    m_p_sustain=172.0,
    m_p_terminal=40.0,
    F_boost=45.0e3,
    F_terminal=8.0e3,
    b_fin=0.23,
    c_r_fin=0.42,
)


def stage_size(reqs: Requirements, geometry_fn, max_evals: int) -> Any:
    """The sizing search."""
    print(f"\n=== SIZE: pattern search, budget {max_evals} evaluations ===")
    print(f"start: m_p_terminal = {SIZING_START.m_p_terminal:.1f} kg (R6 needs terminal thrust)")
    res = size(
        SIZING_START,
        reqs,
        geometry_fn=geometry_fn,
        run_dir=os.path.join(OUT, "size", "ntop"),
        max_evals=max_evals,
        inner_iter=2,
        dt=0.05,
        adaptive=True,
        verbose=True,
    )
    if res.best is not None:
        dump_point(res.best, os.path.join(OUT, "size", "best_point.json"))
        # re-run the winner at the fine fixed step, for the report figures
        print("\nre-running the best point at the fine fixed step for the report figures")
        fine = converge_point(
            res.best.dv,
            reqs,
            geometry_fn=geometry_fn,
            run_dir=os.path.join(OUT, "size", "ntop_fine"),
            max_iter=3,
            dt=0.02,
            adaptive=False,
        )
        print("fine: " + fine.summary())
        dump_point(fine, os.path.join(OUT, "size", "best_point_fine.json"))
        _write_rows(res.trace(), os.path.join(OUT, "size", "search_trace.csv"))
        try:
            from rocketgen.report.fig_trajectory import make_figure  # noqa: F401
        except Exception:                                  # noqa: BLE001
            pass
        return fine
    print("sizing produced no best point")
    return None


def stage_doe(reqs: Requirements, geometry_fn, base: DesignVector, scale: str) -> None:
    """The trade study: a factorial for the contour charts, plus an LHS for sensitivity."""
    print(f"\n=== DOE: {scale} ===")

    if scale == "smoke":
        axes = {"D": [0.30, 0.35], "m_p_sustain": [200.0, 280.0]}
        n_lhs = 6
    else:
        # SPEC.md section 7 item 4 requires at least D, m_p_sustain and f_nose
        # Axes are centred ON the converged design, so the sized point is itself a grid node.
        # An earlier version swept D in {0.28..0.40} and m_p_sustain in {160..400}; neither
        # contained the converged 0.35 m / 172 kg point, so all 75 nodes violated something and
        # the study showed an empty feasible region that does not exist. A trade study has to
        # bracket the answer, not straddle it.
        axes = {
            "D": [0.32, 0.35, 0.38],
            "m_p_sustain": [150.0, 172.0, 200.0, 235.0, 275.0],
            "f_nose": [3.0, 3.4, 3.8],
        }
        n_lhs = 40

    grid = grid_samples(axes)
    print(f"factorial: {len(grid)} samples over {list(axes)}")
    res_grid = run_doe(
        base,
        reqs,
        grid,
        geometry_fn=geometry_fn,
        run_dir=os.path.join(OUT, "doe", "grid_ntop") if geometry_fn else None,
        inner_iter=2,
        dt=0.06,
        adaptive=True,
        verbose=True,
        meta={"kind": "full_factorial", "axes": {k: list(v) for k, v in axes.items()}},
    )
    res_grid.to_csv(os.path.join(OUT, "doe", "grid.csv"))
    res_grid.to_json(os.path.join(OUT, "doe", "grid.json"))
    best = res_grid.best()
    if best is not None:
        print(f"lightest feasible in the factorial: {best.summary()}")
        dump_point(best, os.path.join(OUT, "doe", "grid_best.json"))
    else:
        print("no feasible point in the factorial")

    # The LHS brackets the converged design on every axis and, unlike the factorial, sweeps the
    # terminal charge as well. Leaving m_p_terminal fixed made the sensitivity study blind to the
    # variable that decides whether R6 is met at all.
    ranges = {
        "D": (0.30, 0.40),
        "L_total": (3.4, 4.2),
        "f_nose": (2.8, 3.8),
        "m_p_sustain": (140.0, 260.0),
        "m_p_boost": (90.0, 190.0),
        "m_p_terminal": (20.0, 55.0),
        "b_fin": (0.18, 0.28),
        "c_r_fin": (0.32, 0.52),
    }
    print(f"\nLHS: {n_lhs} samples over {list(ranges)}")
    res_lhs = run_doe(
        base,
        reqs,
        lhs_samples(ranges, n_lhs, seed=20260817),
        geometry_fn=geometry_fn,
        run_dir=os.path.join(OUT, "doe", "lhs_ntop") if geometry_fn else None,
        inner_iter=2,
        dt=0.06,
        adaptive=True,
        verbose=True,
        meta={"kind": "lhs", "ranges": {k: list(v) for k, v in ranges.items()}, "seed": 20260817},
    )
    res_lhs.to_csv(os.path.join(OUT, "doe", "lhs.csv"))
    res_lhs.to_json(os.path.join(OUT, "doe", "lhs.json"))

    responses = ["m0_kg", "range_km", "mach_terminal", "q_max_kPa"]
    sens = sensitivity(res_lhs, list(ranges), responses)
    with open(os.path.join(OUT, "doe", "sensitivity.json"), "w", encoding="utf-8") as f:
        json.dump(sens, f, indent=2)
    print("\nSpearman rank correlation (LHS, converged points only)")
    hdr = "  %-14s" % "variable" + "".join("%14s" % r for r in responses)
    print(hdr)
    for var in ranges:
        line = "  %-14s" % var
        for r in responses:
            v = sens[r][var]
            line += "%14s" % ("nan" if math.isnan(v) else f"{v:+.3f}")
        print(line)


def _write_rows(rows: list[dict[str, Any]], path: str) -> None:
    import csv

    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


# --------------------------------------------------------------------------------------
#   Main
# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="smoke", choices=["smoke", "size", "doe", "all"])
    ap.add_argument("--no-ntop", action="store_true", help="skip nTop, analytic geometry only")
    ap.add_argument("--max-evals", type=int, default=60, help="sizing evaluation budget")
    ap.add_argument("--doe-scale", default=None, choices=["smoke", "full"])
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    reqs = Requirements()
    geometry_fn = get_geometry_fn(not args.no_ntop)
    doe_scale = args.doe_scale or ("smoke" if args.stage == "smoke" else "full")

    t0 = time.perf_counter()
    # The DOE and sizing stages both start from SIZING_START, not the bare default. A base vector
    # with m_p_terminal = 0 fails R6 at every sample, so a standalone `--stage doe` run against the
    # default returned 0 feasible points out of 75 and looked like a modelling failure rather than
    # a badly chosen base point.
    best_dv = SV1_CONVERGED

    if args.stage in ("smoke", "all"):
        stage_smoke(reqs, geometry_fn)
    if args.stage in ("size", "all"):
        fine = stage_size(reqs, geometry_fn, args.max_evals)
        if fine is not None:
            best_dv = fine.dv
    if args.stage in ("doe", "all"):
        stage_doe(reqs, geometry_fn, best_dv, doe_scale)

    dump_provenance(
        os.path.join(OUT, "provenance.json"),
        {
            "stage": args.stage,
            "ntop_enabled": geometry_fn is not None,
            "doe_scale": doe_scale,
            "max_evals": args.max_evals,
            "best_design_vector": best_dv.as_dict(),
            "total_wall_time_s": time.perf_counter() - t0,
        },
    )
    print(f"\ndone in {time.perf_counter() - t0:.1f} s. Artefacts under {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
