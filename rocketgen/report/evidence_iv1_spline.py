"""Collect every number the IV-1 SPLINE report needs into one JSON file.

    .venv/Scripts/python.exe -m rocketgen.report.evidence_iv1_spline

Writes `runs/IV-1_spline/figures/evidence_iv1_spline.json`.

WHY THIS IS A SEPARATE COLLECTOR
--------------------------------
`evidence_iv1.py` describes ONE converged vehicle in full: the requirements audit, the grain
sweeps, the atmosphere near-miss, the strake validation. The spline study is a different
question. It compares TWO converged vehicles that differ only in the shape of the outer mould
line. So this module collects the comparison, and the report it feeds points at the IV-1
report for everything the two studies share.

It also cannot reuse `evidence_iv1.py` as it stands, because that module reads
`runs/IV-1_geom/`, a geometry probe study that this branch does not carry.

WHAT THIS RE-RUNS, AND WHAT IT DOES NOT
---------------------------------------
It does **not** call nTop. Both stacks were already measured, and the per-body measurements are
on disk under `<run>/geom/iv1_stages.json`. Those are read back and fed to the mass and
aerodynamic models exactly as `scripts/iv1_converge.py` fed them.

It **does** re-fly every trajectory, including the whole pitchover sweep. That is affordable at
zero nTop cost for a specific reason: **the pitchover angle is not a geometry input.** Sweeping
it changes `gamma_pitch` and the attitude-control thrust and nothing else, so one measurement
set is valid for every angle of the same shape. The sweep is therefore the real nTop-coupled
sweep, not an analytic stand-in.

Every re-flight of a converged point is checked against the recorded `converged.json`. A
re-flight that did not reproduce the recorded result fails here rather than quietly publishing a
second answer.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if os.path.join(REPO, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "scripts"))

from rocketgen.config import SOURCES as _SOURCES_REGISTRY  # noqa: E402
from rocketgen.config_iv1 import InterceptRequirements  # noqa: E402

SPLINE_DIR = os.path.join(REPO, "runs", "IV-1_spline")
OGIVE_DIR = os.path.join(REPO, "runs", "IV-1_ogive_baseline")
FIGS = os.path.join(SPLINE_DIR, "figures")
OUT_JSON = os.path.join(FIGS, "evidence_iv1_spline.json")

MILE = 1609.344

#: Commanded pitchover angles, deg. The same grid `scripts/iv1_converge.py` searches.
PITCHOVER_DEG: tuple[float, ...] = (32.0, 34.0, 36.0, 38.0)

#: Words that mark a registered source as something other than a measured or cited value.
FLAG_WORDS = ("guess", "modelling choice", "approximation", "assumption")

#: The modules the IV-1 spline result is built from. Imported before the registry is read, so
#: the limitations table cannot under-report. CLAUDE.md section 3.7.
#:
#: `rocketgen.config` is NOT in this list and must not be: `config.SOURCES` IS the registry, so
#: listing it would attribute every key in the repository to it.
SOURCE_MODULES = (
    ("config_iv1", "rocketgen.config_iv1"),
    ("oml_spline", "rocketgen.oml_spline"),
    ("wavedrag", "rocketgen.sizing.wavedrag"),
    ("atmosphere", "rocketgen.sizing.atmosphere"),
    ("aero", "rocketgen.sizing.aero"),
    ("masses", "rocketgen.sizing.masses"),
    ("propulsion", "rocketgen.sizing.propulsion"),
    ("trajectory", "rocketgen.sizing.trajectory"),
    ("aero_iv1", "rocketgen.sizing.aero_iv1"),
    ("masses_iv1", "rocketgen.sizing.masses_iv1"),
    ("propulsion_iv1", "rocketgen.sizing.propulsion_iv1"),
    ("trajectory_iv1", "rocketgen.sizing.trajectory_iv1"),
    ("ntopgen.stack_notebook", "rocketgen.ntopgen.stack_notebook"),
    ("ntopgen.driver", "rocketgen.ntopgen.driver"),
    ("ntopgen.recipe", "rocketgen.ntopgen.recipe"),
    ("ntopgen.rocket_notebook", "rocketgen.ntopgen.rocket_notebook"),
)


def _json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
#   Rebuilding a converged point off disk
# --------------------------------------------------------------------------------------


def load_measurements(path: str) -> dict[int, Any]:
    """Rebuild the `StageMeasurements` objects from a `*_stages.json` file.

    Field by field, not by `**kwargs`, so a field the file does not carry keeps its dataclass
    default instead of raising.
    """
    from rocketgen.ntopgen.stack_notebook import StageMeasurements

    raw = _json(path)
    out: dict[int, Any] = {}
    for key, d in raw.items():
        m = StageMeasurements(stage_index=int(d.get("stage_index", int(key))))
        for name, value in d.items():
            if not hasattr(m, name):
                continue
            if isinstance(value, list) and name in (
                "cg_structure", "inertia_structure", "cg_structure_stack"
            ):
                setattr(m, name, tuple(float(v) for v in value))
            elif name == "area_distribution":
                setattr(m, name, [(float(a), float(b)) for a, b in value])
            else:
                setattr(m, name, value)
        out[int(key)] = m
    return out


def shaped(dv, conv: dict[str, Any]):
    """Apply the RECORDED outer-mould-line shape to a freshly built stack.

    Data driven on purpose. `scripts/iv1_converge.py` applies the shape from its own command
    line; reading it back from `converged.json` means this module cannot describe a different
    shape from the one that was flown.
    """
    want = conv["design_vector"]
    return dv.replace(
        nose_shape=want["nose_shape"],
        nose_blend=float(want["nose_blend"]),
        interstage_shape=want["interstage_shape"],
        interstage_blend=float(want["interstage_blend"]),
    )


def rebuild_design_vector(conv: dict[str, Any]):
    """The converged design vector, rebuilt from `base_stack` and then checked field by field."""
    from iv1_converge import base_stack  # noqa: PLC0415

    dv = shaped(base_stack(), conv)
    dv = dv.replace(gamma_pitch=math.radians(float(conv["pitchover_deg"])))
    dv.stages[1].F_thrust = float(conv["design_vector"]["stages"][1]["F_thrust"])
    dv.acs.thrust = float(conv["acs"]["thrust"])

    got, want = dv.as_dict(), conv["design_vector"]
    bad: list[str] = []

    def cmp(label: str, have: Any, wanted: Any) -> None:
        if isinstance(wanted, (int, float)) and not isinstance(wanted, bool):
            if not math.isclose(float(have), float(wanted), rel_tol=1e-12, abs_tol=1e-12):
                bad.append(f"{label}: rebuilt {have!r} against recorded {wanted!r}")
        elif have != wanted:
            bad.append(f"{label}: rebuilt {have!r} against recorded {wanted!r}")

    for key, wanted in want.items():
        have = got.get(key)
        if isinstance(wanted, list):
            for i, w in enumerate(wanted):
                for k2, v2 in w.items():
                    cmp(f"{key}[{i}].{k2}", have[i][k2], v2)
        elif isinstance(wanted, dict):
            for k2, v2 in wanted.items():
                cmp(f"{key}.{k2}", have[k2], v2)
        else:
            cmp(key, have, wanted)
    if bad:
        raise SystemExit(
            "the rebuilt design vector does not match converged.json:\n  " + "\n  ".join(bad)
        )
    return dv


def refly(case_dir: str, conv: dict[str, Any]) -> dict[str, Any]:
    """Re-fly one converged point with the geometry read back from disk."""
    from iv1_converge import evaluate  # noqa: PLC0415

    reqs = InterceptRequirements()
    dv = rebuild_design_vector(conv)
    meas = load_measurements(os.path.join(case_dir, "geom", "iv1_stages.json"))

    t0 = time.perf_counter()
    r = evaluate(dv, reqs, geometry_fn=lambda _dv, _rd: meas, dt=0.02, adaptive=False)
    wall = time.perf_counter() - t0

    ic, rec = r["intercept"], conv["intercept"]
    for name, got, wanted in (
        ("launch_mass_kg", r["m0"], conv["launch_mass_kg"]),
        ("slant_range", ic.slant_range, rec["slant_range"]),
        ("altitude", ic.altitude, rec["altitude"]),
        ("mach", ic.mach, rec["mach"]),
        ("time", ic.time, rec["time"]),
        ("mass", ic.mass, rec["mass"]),
        ("lateral_g_aero", r["g_aero"], conv["lateral_g"]["aerodynamic"]),
    ):
        if not math.isclose(float(got), float(wanted), rel_tol=1e-9, abs_tol=1e-9):
            raise SystemExit(
                f"re-flight of {os.path.basename(case_dir)} disagrees with converged.json "
                f"on {name}: {got!r} against {wanted!r}"
            )
    r["wall_time_s"] = wall
    r["meas_loaded"] = meas
    return r


# --------------------------------------------------------------------------------------
#   The comparison
# --------------------------------------------------------------------------------------


def trajectory_record(r: dict[str, Any]) -> dict[str, Any]:
    """The arrays the figures need, plus the scalar summary."""
    t = r["traj"]
    ic = r["intercept"]
    apogee = max(t.h) if t.h else 0.0
    i_ap = t.h.index(apogee) if t.h else 0
    i_q = max(range(len(t.q)), key=lambda i: t.q[i]) if t.q else 0
    return {
        "time": list(t.time),
        "x": list(t.x),
        "h": list(t.h),
        "mach": list(t.mach),
        "q": list(t.q),
        "mass": list(t.mass),
        "phase": list(getattr(t, "phase", []) or []),
        "n_samples": len(t.time),
        "duration_s": t.time[-1] if t.time else 0.0,
        "apogee_m": apogee,
        "apogee_time_s": t.time[i_ap] if t.time else 0.0,
        "q_max_Pa": t.q_max,
        "q_max_altitude_m": t.h[i_q] if t.h else 0.0,
        "q_max_mach": t.mach[i_q] if t.mach else 0.0,
        "q_max_time_s": t.time[i_q] if t.time else 0.0,
        "message": t.message,
        "intercept": {
            "time": ic.time, "altitude": ic.altitude, "mach": ic.mach,
            "slant_range": ic.slant_range, "slant_range_miles": ic.slant_range / MILE,
            "q": ic.q, "mass": ic.mass,
        },
    }


def comparison_record(rs: dict[str, Any], ro: dict[str, Any],
                      cs: dict[str, Any], co: dict[str, Any]) -> dict[str, Any]:
    """The headline table: the same vehicle, two outer mould lines."""
    ics, ico = rs["intercept"], ro["intercept"]

    def row(name: str, unit: str, ogive: float, spline: float, nd: int = 3) -> dict[str, Any]:
        return {
            "quantity": name, "unit": unit, "ogive": ogive, "spline": spline,
            "delta": spline - ogive,
            "delta_pct": (100.0 * (spline / ogive - 1.0)) if ogive else None,
            "nd": nd,
        }

    return {
        "rows": [
            row("Launch mass", "kg", co["launch_mass_kg"], cs["launch_mass_kg"], 2),
            row("Slant range at intercept", "km",
                ico.slant_range / 1000.0, ics.slant_range / 1000.0, 2),
            row("Intercept altitude", "km", ico.altitude / 1000.0, ics.altitude / 1000.0, 2),
            row("Intercept Mach", "-", ico.mach, ics.mach, 3),
            row("Time to intercept", "s", ico.time, ics.time, 1),
            row("Dynamic pressure at intercept", "kPa", ico.q / 1000.0, ics.q / 1000.0, 2),
            row("Mass at intercept", "kg", ico.mass, ics.mass, 2),
            row("Lateral g, aerodynamic", "g",
                co["lateral_g"]["aerodynamic"], cs["lateral_g"]["aerodynamic"], 3),
            row("Lateral g, attitude control", "g",
                co["lateral_g"]["acs"], cs["lateral_g"]["acs"], 3),
            row("Lateral g, A11 figure", "g",
                co["lateral_g"]["total"], cs["lateral_g"]["total"], 3),
            row("CN_max at intercept", "-",
                co["lateral_g"]["cn_max"], cs["lateral_g"]["cn_max"], 3),
            row("Peak dynamic pressure", "kPa",
                ro["traj"].q_max / 1000.0, rs["traj"].q_max / 1000.0, 2),
            row("Apogee", "km",
                max(ro["traj"].h) / 1000.0, max(rs["traj"].h) / 1000.0, 2),
            row("Attitude-control thrust", "kN",
                co["acs"]["thrust"] / 1000.0, cs["acs"]["thrust"] / 1000.0, 2),
            row("Mass jettisoned at separation", "kg",
                co["jettisoned_kg"], cs["jettisoned_kg"], 2),
        ],
        "n_constraints": len(cs["constraints"]),
        "ogive_feasible": bool(co["feasible"]),
        "spline_feasible": bool(cs["feasible"]),
        "n_met_ogive": sum(1 for c in co["constraints"] if c["met"]),
        "n_met_spline": sum(1 for c in cs["constraints"] if c["met"]),
    }


def pitchover_sweep(conv_by_shape: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Sweep the commanded pitchover angle for both shapes, with measured geometry.

    Cheap and still nTop-coupled: the pitchover angle is not a geometry input, so one
    measurement set per shape serves every angle. The attitude-control fixed point is re-solved
    at every angle, exactly as `scripts/iv1_converge.py` does.
    """
    from iv1_converge import size_acs, base_stack  # noqa: PLC0415

    reqs = InterceptRequirements()
    out: dict[str, Any] = {"gamma_deg": list(PITCHOVER_DEG), "shapes": {}}
    for shape, (case_dir, conv) in conv_by_shape.items():
        meas = load_measurements(os.path.join(case_dir, "geom", "iv1_stages.json"))
        rows = []
        for gamma_deg in PITCHOVER_DEG:
            dv = shaped(base_stack(), conv).replace(gamma_pitch=math.radians(gamma_deg))
            dv.stages[1].F_thrust = float(conv["design_vector"]["stages"][1]["F_thrust"])
            dv, r = size_acs(dv, reqs, geometry_fn=lambda _dv, _rd: meas)
            ic = r["intercept"]
            rows.append({
                "gamma_deg": gamma_deg,
                "slant_range_km": ic.slant_range / 1000.0,
                "altitude_km": ic.altitude / 1000.0,
                "mach": ic.mach,
                "m0_kg": r["m0"],
                "q_at_intercept_kPa": ic.q / 1000.0,
                "lateral_g_aero": r["g_aero"],
                "feasible": bool(r["feasible"]),
                "violations": [n for n, *_rest, ok in r["constraints"] if not ok],
            })
        out["shapes"][shape] = rows
    for shape, rows in out["shapes"].items():
        out[f"n_feasible_{shape}"] = sum(1 for r in rows if r["feasible"])
    return out


def wave_drag_record(rs: dict[str, Any], ro: dict[str, Any]) -> dict[str, Any]:
    """The shape ratio on THIS vehicle, and the drag build-up it moves.

    The closed-form validation of `sizing/wavedrag.py` itself is collected by
    `rocketgen.report.evidence.wave_drag_validation`, which is shared with the SV-1 spline
    report. It is called here so both reports quote one measurement.
    """
    from rocketgen.report.evidence import wave_drag_validation  # noqa: PLC0415

    payload = {"validation": wave_drag_validation()}

    dv_s, dv_o = rs["dv"], ro["dv"]
    aero_s, aero_o = rs["aero"], ro["aero"]
    payload["nose_control"] = list(dv_s.nose_control or [])
    payload["k_L_over_R"] = dv_s.L_nose / (0.5 * dv_s.payload_stage.D)
    payload["shape_ratio"] = aero_s._resolve_nose_shape_factor()
    payload["shape_ratio_ogive"] = aero_o._resolve_nose_shape_factor()

    rows = []
    for mach in (1.2, 2.0, 3.0, 4.0, 5.0):
        for stage in (1, 2):
            a_s = aero_s.evaluate(mach, 12_000.0, math.radians(2.0), stage)
            a_o = aero_o.evaluate(mach, 12_000.0, math.radians(2.0), stage)
            rows.append({
                "mach": mach, "stage": stage,
                "cd0_ogive": a_o.CD0, "cd0_spline": a_s.CD0,
                "d_cd0_pct": 100.0 * (a_s.CD0 / a_o.CD0 - 1.0),
                "cd_wave_ogive": a_o.breakdown.get("CD_wave_body"),
                "cd_wave_spline": a_s.breakdown.get("CD_wave_body"),
                "wave_share_ogive": (
                    a_o.breakdown.get("CD_wave_body", 0.0) / a_o.CD0 if a_o.CD0 else None
                ),
                "x_cp_ogive": a_o.x_cp, "x_cp_spline": a_s.x_cp,
            })
    payload["drag_rows"] = rows
    return payload


def geometry_record(rs: dict[str, Any], ro: dict[str, Any]) -> dict[str, Any]:
    """Measured stage volumes against the exact integral of the same B-spline.

    nTop revolves the spline itself, so the closed form is an exact description of the solid
    rather than an approximation of it, and this is a real check of the notebook.
    """
    from rocketgen.oml_spline import SplineProfile

    def one(r: dict[str, Any], label: str) -> dict[str, Any]:
        dv = r["dv"]
        meas = r["meas_loaded"]
        s2 = dv.payload_stage
        R = 0.5 * s2.D
        control = getattr(dv, "nose_control", None)
        if control is not None:
            nose = SplineProfile(length=dv.L_nose, radius=R, control=control)
            v_nose, a_nose = nose.volume(), nose.lateral_area()
            nose_form = "revolved cubic B-spline, exact integral"
        else:
            rho = (R * R + dv.L_nose ** 2) / (2.0 * R)
            # Tangent-ogive volume and lateral area, by the same quadrature masses.py uses.
            n = 20001
            v_nose = a_nose = 0.0
            prev_x = prev_y = 0.0
            for i in range(1, n):
                x = dv.L_nose * i / (n - 1)
                y = max(math.sqrt(max(rho * rho - (dv.L_nose - x) ** 2, 0.0)) - (rho - R), 0.0)
                v_nose += 0.5 * math.pi * (prev_y ** 2 + y ** 2) * (x - prev_x)
                a_nose += math.pi * (prev_y + y) * math.hypot(x - prev_x, y - prev_y)
                prev_x, prev_y = x, y
            nose_form = "tangent ogive, quadrature"
        v_cyl = math.pi * R * R * (s2.L - dv.L_nose)
        v_closed = v_nose + v_cyl
        m2 = meas.get(2)
        v_meas = getattr(m2, "volume_total", None) if m2 is not None else None
        return {
            "label": label,
            "nose_form": nose_form,
            "volume_ntop_stage2": v_meas,
            "volume_closed_form_stage2": v_closed,
            "rel_err": (v_meas / v_closed - 1.0) if v_meas else None,
            "nose_volume_m3": v_nose,
            "nose_wetted_m2": a_nose,
            "wall_time_s": getattr(m2, "wall_time_s", None) if m2 is not None else None,
        }

    out = {"spline": one(rs, "spline"), "ogive": one(ro, "ogive")}
    ns, no = out["spline"]["nose_volume_m3"], out["ogive"]["nose_volume_m3"]
    out["nose_volume_change_pct"] = 100.0 * (ns / no - 1.0)
    out["nose_wetted_change_pct"] = 100.0 * (
        out["spline"]["nose_wetted_m2"] / out["ogive"]["nose_wetted_m2"] - 1.0
    )
    return out


def constraint_record(conv: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for c in conv["constraints"]:
        limit = float(c["limit"])
        value = float(c["value"])
        if limit == 0.0:
            m = 0.0
        else:
            m = (value - limit) / abs(limit) if c["sense"] == ">=" else (limit - value) / abs(limit)
        rows.append({**c, "margin": m})
    return rows


def mass_record(r: dict[str, Any], conv: dict[str, Any]) -> dict[str, Any]:
    sm = r["masses"]
    return {
        "m0_kg": r["m0"],
        "measured_fraction": sm.measured_fraction,
        "measured_kg": sm.measured_fraction * sm.m0,
        "jettisoned_kg": conv["jettisoned_kg"],
        "acs_pack_kg": r["dv"].acs.total_mass,
        "acs_total_impulse_Ns": r["dv"].acs.total_impulse,
        "items": conv["mass_statement"],
        "stage_totals_kg": {
            str(i): sum(e["mass_kg"] for e in conv["mass_statement"] if e["stage"] == i)
            for i in (0, 1, 2)
        },
    }


def source_provenance() -> dict[str, Any]:
    """The registry, read only AFTER every owning module is imported. CLAUDE.md section 3.7."""
    import importlib

    owned: dict[str, str] = {}
    for label, module in SOURCE_MODULES:
        try:
            m = importlib.import_module(module)
        except Exception:                                    # noqa: BLE001
            continue
        for key in getattr(m, "SOURCES", {}):
            owned.setdefault(key, label)

    sources = dict(_SOURCES_REGISTRY)
    flagged = {
        k: v for k, v in sorted(sources.items())
        if any(w in v.lower() for w in FLAG_WORDS)
    }
    by_module: dict[str, int] = {}
    for key in sources:
        by_module[owned.get(key, "config")] = by_module.get(owned.get(key, "config"), 0) + 1
    return {
        "n_registered": len(sources),
        "n_flagged": len(flagged),
        "flagged": flagged,
        "owner": {k: owned.get(k, "config") for k in sources},
        "by_module": by_module,
    }


def environment_record() -> dict[str, Any]:
    import platform

    import numpy
    import scipy

    from rocketgen.config import CD0_CALIBRATION  # noqa: PLC0415

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "suave": "2.5.2, vendored from github.com/suavecode/SUAVE",
        "ntop": "5.53.2 installed, 5.54.0 development build",
        "cd0_calibration_available": float(CD0_CALIBRATION),
        # `scripts/iv1_converge.py` builds `StackAero` directly and does NOT wrap it in the
        # `loop.CalibratedAero` boundary the SV-1 loop uses. The IV-1 result is therefore
        # uncalibrated. CLAUDE.md section 3.8. Recorded so the report must say so.
        "cd0_calibration_applied_to_iv1": False,
    }


def notebook_record() -> dict[str, Any]:
    """Block counts of the two stack notebooks, and which blocks are outside the universe."""
    try:
        from rocketgen.ntopgen.universe import Universe  # noqa: PLC0415
    except Exception as exc:                                 # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    out: dict[str, Any] = {"available": True}
    universe = None
    try:
        universe = Universe.load()
        out["n_universe_signatures"] = len(universe._by_id)
    except Exception as exc:                                 # noqa: BLE001
        out["n_universe_signatures"] = None
        out["universe_error"] = f"{type(exc).__name__}: {exc}"

    # The four blocks the spline chain needs. None is in the vendored universe, so each one is
    # emitted through `Recipe.raw_block`. docs/NTOP_NOTES.md section 25.
    out["spline_chain"] = [
        "spline_by_control_points<list<point>,integer>[5.20.0]",
        "core.list<curve_interface>",
        "profile_from_curves<list<curve_interface>,vector>[5.20.0]",
        "revolve<new_profile,axis,real>[5.20.0]",
    ]
    out["straight_edge_block"] = "two_point_line<point,point>"
    if universe is not None:
        known = set(universe._by_id)
        out["in_universe"] = {sig: (sig in known) for sig in out["spline_chain"]}
        out["n_in_universe"] = sum(1 for v in out["in_universe"].values() if v)
    else:
        out["in_universe"] = None
    return out


# --------------------------------------------------------------------------------------
#   Main
# --------------------------------------------------------------------------------------


def main() -> str:
    os.makedirs(FIGS, exist_ok=True)
    for d in (SPLINE_DIR, OGIVE_DIR):
        if not os.path.isfile(os.path.join(d, "converged.json")):
            raise SystemExit(f"missing {os.path.join(d, 'converged.json')}")

    cs = _json(os.path.join(SPLINE_DIR, "converged.json"))
    co = _json(os.path.join(OGIVE_DIR, "converged.json"))
    t0 = time.perf_counter()
    rs = refly(SPLINE_DIR, cs)
    ro = refly(OGIVE_DIR, co)

    payload: dict[str, Any] = {
        "generated_from": [
            "runs/IV-1_spline/converged.json + runs/IV-1_spline/geom/iv1_stages.json",
            "runs/IV-1_ogive_baseline/converged.json + "
            "runs/IV-1_ogive_baseline/geom/iv1_stages.json",
        ],
        "comparison": comparison_record(rs, ro, cs, co),
        "trajectory": {"spline": trajectory_record(rs), "ogive": trajectory_record(ro)},
        "constraints": {"spline": constraint_record(cs), "ogive": constraint_record(co)},
        "mass": {"spline": mass_record(rs, cs), "ogive": mass_record(ro, co)},
        "pitchover": pitchover_sweep({
            "spline": (SPLINE_DIR, cs), "ogive": (OGIVE_DIR, co),
        }),
        "wavedrag": wave_drag_record(rs, ro),
        "geometry": geometry_record(rs, ro),
        "sources": source_provenance(),
        "environment": environment_record(),
        "notebook": notebook_record(),
        "warnings": {"spline": cs["warnings"], "ogive": co["warnings"]},
        "design_vector": {"spline": cs["design_vector"], "ogive": co["design_vector"]},
        "requirements": cs["requirements"],
    }
    payload["wall_time_s"] = time.perf_counter() - t0

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=str)
    print(f"wrote {os.path.relpath(OUT_JSON, REPO)} "
          f"({os.path.getsize(OUT_JSON) / 1024:.0f} KB)")
    print("  both re-flights reproduced their converged.json")
    print(f"  flagged sources: {payload['sources']['n_flagged']} of "
          f"{payload['sources']['n_registered']} registered")
    print(f"  pitchover: {payload['pitchover']['n_feasible_spline']} feasible for the spline, "
          f"{payload['pitchover']['n_feasible_ogive']} for the ogive")
    return OUT_JSON


if __name__ == "__main__":
    main()
