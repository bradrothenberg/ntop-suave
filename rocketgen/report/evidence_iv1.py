"""Collect every number the IV-1 report needs into one JSON file.

    .venv/Scripts/python.exe -m rocketgen.report.evidence_iv1

Writes `runs/IV-1/figures/evidence_iv1.json`.

WHAT THIS DOES AND DOES NOT RE-RUN
----------------------------------
It does **not** call nTop. One `measure_stack` call costs 55 to 118 s, and the geometry has
already been measured: `runs/IV-1/geom/iv1_stages.json` holds the per-body measurements of the
converged design point. Those are read back and fed to the mass and aerodynamic models exactly as
the sizing script fed them.

It does **not** repeat the pitchover search either. The converged pitchover angle is read from
`runs/IV-1/converged.json` and the design vector is rebuilt from `scripts/iv1_converge.base_stack`
plus that angle and the converged attitude-control thrust. The rebuilt vector is then checked
field by field against the design vector recorded in `converged.json`, so this module cannot drift
away from the result it describes.

It **does** re-fly the one converged trajectory, because the trajectory arrays were never written
to disk and the report needs them for the flight-envelope figure. That is a single integration of
a 3-DOF point mass and it costs a few seconds. The intercept conditions it produces are asserted
against `converged.json`, so a re-flight that did not reproduce the recorded result would fail
here rather than quietly publish a second answer.

Everything else is a cheap diagnostic on the converged point: strake sensitivity, static margin in
both flight configurations, the closed-form geometry cross-check, the staging gain against the
equivalent single stage, and the atmosphere checks.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if os.path.join(REPO, "tests") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "tests"))

from rocketgen.config import SOURCES as _SOURCES_REGISTRY  # noqa: E402
from rocketgen.config_iv1 import (  # noqa: E402
    InterceptRequirements,
    StrakeSpec,
    lateral_g,
    lateral_g_acs,
    slant_range,
)
from rocketgen.sizing.aero_iv1 import StackAero  # noqa: E402
from rocketgen.sizing.atmosphere import H_MAX, H_MAX_LEGACY, atmo, table  # noqa: E402
from rocketgen.sizing.masses_iv1 import build_stack_masses, static_margin_stage  # noqa: E402
from rocketgen.sizing.propulsion_iv1 import MultiStageMotor  # noqa: E402
from rocketgen.sizing.trajectory_iv1 import AscentMission  # noqa: E402

IV1 = os.path.join(REPO, "runs", "IV-1")
GEOM = os.path.join(IV1, "geom")
GEOM_STUDY = os.path.join(REPO, "runs", "IV-1_geom")
FIGS = os.path.join(IV1, "figures")
OUT_JSON = os.path.join(FIGS, "evidence_iv1.json")

#: Words that mark a registered source as something other than a measured or cited value.
FLAG_WORDS = ("guess", "modelling choice", "approximation", "assumption")

#: Which module registered which source key. Filled by `source_provenance`.
#:
#: `rocketgen.config` is NOT in this list, and must not be: `config.SOURCES` IS the global
#: registry that `register_sources` writes into, so iterating it would attribute every key in the
#: repository to config. Keys that no listed module claims are config's own.
SOURCE_MODULES = (
    ("config_iv1", "rocketgen.config_iv1"),
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

#: The five modules the IV-1 result is built from. Imported at module load, so the registry is
#: complete before `source_provenance` reads it.
IV1_MODULES = (
    "rocketgen.config_iv1",
    "rocketgen.sizing.masses_iv1",
    "rocketgen.sizing.propulsion_iv1",
    "rocketgen.sizing.trajectory_iv1",
    "rocketgen.sizing.aero_iv1",
)


# --------------------------------------------------------------------------------------
#   Loading the converged point back off disk
# --------------------------------------------------------------------------------------


def _json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_measurements(path: str) -> dict[int, Any]:
    """Rebuild the `StageMeasurements` objects from a `*_stages.json` file.

    The dataclass is reconstructed field by field rather than by `**kwargs`, so a field that the
    file does not carry keeps its dataclass default instead of raising.
    """
    from rocketgen.ntopgen.stack_notebook import StageMeasurements

    raw = _json(path)
    out: dict[int, Any] = {}
    for key, d in raw.items():
        m = StageMeasurements(stage_index=int(d.get("stage_index", int(key))))
        for name, value in d.items():
            if not hasattr(m, name):
                continue
            if isinstance(value, list) and name in ("cg_structure", "inertia_structure",
                                                    "cg_structure_stack"):
                setattr(m, name, tuple(float(v) for v in value))
            elif name == "area_distribution":
                setattr(m, name, [(float(a), float(b)) for a, b in value])
            else:
                setattr(m, name, value)
        out[int(key)] = m
    return out


def rebuild_design_vector(conv: dict[str, Any]):
    """The converged design vector, rebuilt and then checked against the recorded one."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from iv1_converge import base_stack  # noqa: PLC0415

    dv = base_stack().replace(gamma_pitch=math.radians(float(conv["pitchover_deg"])))
    dv.stages[1].F_thrust = float(conv["design_vector"]["stages"][1]["F_thrust"])
    dv.acs.thrust = float(conv["acs"]["thrust"])

    got, want = dv.as_dict(), conv["design_vector"]
    bad = []
    for key, wanted in want.items():
        have = got.get(key)
        if isinstance(wanted, (int, float)) and not isinstance(wanted, bool):
            if not math.isclose(float(have), float(wanted), rel_tol=1e-12, abs_tol=1e-12):
                bad.append(f"{key}: rebuilt {have!r} against recorded {wanted!r}")
        elif isinstance(wanted, list):
            for i, w in enumerate(wanted):
                for k2, v2 in w.items():
                    h2 = have[i][k2]
                    if isinstance(v2, (int, float)) and not isinstance(v2, bool):
                        if not math.isclose(float(h2), float(v2), rel_tol=1e-12, abs_tol=1e-12):
                            bad.append(f"{key}[{i}].{k2}: {h2!r} against {v2!r}")
                    elif h2 != v2:
                        bad.append(f"{key}[{i}].{k2}: {h2!r} against {v2!r}")
        elif isinstance(wanted, dict):
            for k2, v2 in wanted.items():
                h2 = have[k2]
                if isinstance(v2, (int, float)) and not isinstance(v2, bool):
                    if not math.isclose(float(h2), float(v2), rel_tol=1e-12, abs_tol=1e-12):
                        bad.append(f"{key}.{k2}: {h2!r} against {v2!r}")
                elif h2 != v2:
                    bad.append(f"{key}.{k2}: {h2!r} against {v2!r}")
        elif have != wanted:
            bad.append(f"{key}: {have!r} against {wanted!r}")
    if bad:
        raise SystemExit(
            "the rebuilt design vector does not match runs/IV-1/converged.json:\n  "
            + "\n  ".join(bad)
        )
    return dv


# --------------------------------------------------------------------------------------
#   The one re-flown trajectory
# --------------------------------------------------------------------------------------


def refly(conv: dict[str, Any]) -> dict[str, Any]:
    """Re-fly the converged point with the geometry read back from disk."""
    reqs = InterceptRequirements()
    dv = rebuild_design_vector(conv)
    meas = load_measurements(os.path.join(GEOM, "iv1_stages.json"))

    motor = MultiStageMotor(dv, reqs)
    sm = build_stack_masses(dv, reqs, meas=meas, motor=motor)
    m0 = sm.m0 + dv.acs.total_mass
    aero = StackAero(dv, reqs, meas=meas)

    t0 = time.perf_counter()
    mission = AscentMission(dv, reqs, motor, aero, m0)
    traj = mission.fly(dt=0.02, adaptive=False, t_max=600.0)
    wall = time.perf_counter() - t0
    ic = mission.intercept

    # The re-flight must reproduce the recorded result, or this module is describing a
    # different vehicle from the one in converged.json.
    rec = conv["intercept"]
    for name, got, want in (
        ("launch_mass_kg", m0, conv["launch_mass_kg"]),
        ("slant_range", ic.slant_range, rec["slant_range"]),
        ("altitude", ic.altitude, rec["altitude"]),
        ("mach", ic.mach, rec["mach"]),
        ("time", ic.time, rec["time"]),
        ("mass", ic.mass, rec["mass"]),
        ("q_max", traj.q_max, _constraint(conv, "A10 q_max [Pa]")["value"]),
    ):
        if not math.isclose(float(got), float(want), rel_tol=1e-9, abs_tol=1e-9):
            raise SystemExit(
                f"re-flight disagrees with converged.json on {name}: {got!r} against {want!r}"
            )

    return {
        "dv": dv, "reqs": reqs, "meas": meas, "motor": motor, "masses": sm,
        "aero": aero, "traj": traj, "intercept": ic, "m0": m0, "wall_time_s": wall,
    }


def _constraint(conv: dict[str, Any], name: str) -> dict[str, Any]:
    for c in conv["constraints"]:
        if c["name"] == name:
            return c
    raise KeyError(name)


def margin(value: float, limit: float, sense: str) -> float:
    """Normalised margin: positive means the constraint is met with room to spare."""
    if limit == 0.0:
        return 0.0
    return (value - limit) / abs(limit) if sense == ">=" else (limit - value) / abs(limit)


# --------------------------------------------------------------------------------------
#   Pieces of evidence
# --------------------------------------------------------------------------------------


def trajectory_record(R: dict[str, Any]) -> dict[str, Any]:
    """The trajectory, its events, and the quantities the envelope figure needs."""
    traj, motor = R["traj"], R["motor"]
    h = np.asarray(traj.h, dtype=float)
    apogee = int(np.argmax(h))

    events = []
    for e in traj.diagnostics.get("events", []):
        events.append(
            {
                "name": e["name"], "time": e["time"], "altitude": e["altitude"],
                "mach": e["mach"], "mass_before": e["mass_before"],
                "mass_after": e["mass_after"], "note": e.get("note", ""),
            }
        )

    q = np.asarray(traj.q, dtype=float)
    iq = int(np.argmax(q))
    return {
        "n_samples": len(traj.time),
        "dt": 0.02,
        "time": [round(v, 6) for v in traj.time],
        "x": [round(v, 3) for v in traj.x],
        "h": [round(v, 3) for v in traj.h],
        "mach": [round(v, 6) for v in traj.mach],
        "q": [round(v, 3) for v in traj.q],
        "mass": [round(v, 6) for v in traj.mass],
        "phase": list(traj.phase),
        "events": events,
        "apogee_m": float(h[apogee]),
        "apogee_time_s": float(traj.time[apogee]),
        "q_max_Pa": float(q[iq]),
        "q_max_altitude_m": float(h[iq]),
        "q_max_time_s": float(traj.time[iq]),
        "q_max_mach": float(traj.mach[iq]),
        "duration_s": float(traj.time[-1]),
        "message": traj.message,
        "converged": bool(traj.converged),
        "t_separation_s": float(motor.t_separation),
        "t_burnout_1_s": float(motor.t_burnout(1)),
        "t_ignition_2_s": float(motor.t_ignition(2)),
        "t_burnout_2_s": float(motor.t_burnout(2)),
        "alpha": [round(v, 8) for v in traj.alpha],
        "gamma": [round(v, 8) for v in traj.gamma],
        "CN_required": [round(v, 6) for v in traj.CN_required],
        "alpha_limited_steps": int(sum(1 for f in traj.alpha_limited if f)),
        "alpha_limit_hits": float(traj.diagnostics.get("alpha_limit_hits", 0.0)),
        "alpha_limit_fraction": float(traj.diagnostics.get("alpha_limit_fraction", 0.0)),
        "alpha_max_flown_deg": math.degrees(max(abs(a) for a in traj.alpha)),
        "t_pitch_start_s": float(traj.diagnostics.get("t_pitch_start", float("nan"))),
        "t_pitch_complete_s": float(traj.diagnostics.get("t_pitch_complete", float("nan"))),
        "pitch_rate_commanded_deg_s": math.degrees(float(R["dv"].pitch_rate_max)),
        "pitch_rate_flown_deg_s": (
            math.degrees(abs(float(R["reqs"].gamma_launch) - float(R["dv"].gamma_pitch)))
            / (float(traj.diagnostics["t_pitch_complete"]) - float(R["dv"].t_pitch))
            if traj.diagnostics.get("t_pitch_complete") else float("nan")
        ),
        "guidance_segments": traj.diagnostics.get("guidance_segments", []),
        "atmosphere_table_ceiling_m": float(
            traj.diagnostics.get("atmosphere_table_ceiling", 0.0)
        ),
        "force_calls": float(traj.diagnostics.get("force_calls", 0.0)),
        "h_above_atmosphere_table": float(
            traj.diagnostics.get("h_above_atmosphere_table", 0.0)
        ),
        "q_floor_hit": bool(traj.diagnostics.get("q_floor_hit", False)),
    }


def mass_record(R: dict[str, Any], conv: dict[str, Any]) -> dict[str, Any]:
    """Per-stage mass roll-up, and what leaves the vehicle."""
    sm, motor, dv = R["masses"], R["motor"], R["dv"]
    rows = conv["mass_statement"]

    by_stage: dict[str, dict[str, float]] = {}
    by_prov: dict[str, float] = {}
    for r in rows:
        s = str(int(r["stage"]))
        by_stage.setdefault(s, {})[r["provenance"]] = (
            by_stage.setdefault(s, {}).get(r["provenance"], 0.0) + r["mass_kg"]
        )
        by_prov[r["provenance"]] = by_prov.get(r["provenance"], 0.0) + r["mass_kg"]

    return {
        "rows": rows,
        "by_stage_provenance": by_stage,
        "by_provenance": by_prov,
        "acs_pack_kg": float(dv.acs.total_mass),
        "acs_propellant_kg": float(dv.acs.propellant_mass),
        "acs_inert_kg": float(dv.acs.inert_mass),
        "acs_total_impulse_Ns": float(dv.acs.total_impulse),
        "m0_kg": float(R["m0"]),
        "measured_fraction": float(sm.measured_fraction),
        "measured_kg": float(sum(r["mass_kg"] for r in rows
                                 if r["provenance"] == "ntop_measured")),
        "jettisoned_kg": float(motor.jettisoned_mass()),
        "mass_after_separation_kg": float(sm.mass_after_separation()),
        "stage_totals_kg": {
            str(k): float(sum(r["mass_kg"] for r in rows if int(r["stage"]) == k))
            for k in sorted({int(r["stage"]) for r in rows})
        },
        "warnings": list(conv["warnings"]),
    }


def constraint_record(conv: dict[str, Any]) -> list[dict[str, Any]]:
    """The recorded constraints with a normalised margin added."""
    out = []
    for c in conv["constraints"]:
        out.append(
            {
                **c,
                "margin": margin(float(c["value"]), float(c["limit"]), c["sense"]),
            }
        )
    return out


def static_margin_record(R: dict[str, Any], conv: dict[str, Any]) -> dict[str, Any]:
    """Requirement A9, computed here because the recorded constraint list does not carry it.

    Two flight configurations, because the reference length and the aerodynamic surfaces both
    change at separation. `x_cg` comes from the mass statement, aft of the payload-stage nose tip,
    which is the same datum `aero.x_cp` uses.
    """
    aero, dv, reqs = R["aero"], R["dv"], R["reqs"]
    rows = conv["mass_statement"]

    def cg(items: list[dict[str, Any]]) -> tuple[float, float]:
        m = sum(r["mass_kg"] for r in items)
        return m, sum(r["mass_kg"] * r["station_m"] for r in items) / m

    launch_m, launch_cg = cg(rows)
    # At stage-1 burnout the booster charge is gone but the booster is still attached.
    burnout1 = [r for r in rows
                if not (int(r["stage"]) == 1 and r["item"] == "Propellant")]
    b1_m, b1_cg = cg(burnout1)
    stage2 = [r for r in rows if int(r["stage"]) == 2]
    stage2_dry = [r for r in stage2 if r["item"] != "Propellant"]
    s2_m, s2_cg = cg(stage2)
    s2d_m, s2d_cg = cg(stage2_dry)

    alpha = math.radians(10.0)
    out: dict[str, Any] = {
        "alpha_deg": 10.0,
        "limit_calibres": float(reqs.static_margin_min),
        "x_cg_launch_m": launch_cg,
        "mass_launch_kg": launch_m,
        "x_cg_stage1_burnout_m": b1_cg,
        "mass_stage1_burnout_kg": b1_m,
        "x_cg_stage2_full_m": s2_cg,
        "mass_stage2_full_kg": s2_m,
        "x_cg_stage2_dry_m": s2d_cg,
        "mass_stage2_dry_kg": s2d_m,
        "D_ref_stage1_m": float(aero.D_ref(1)),
        "D_ref_stage2_m": float(aero.D_ref(2)),
        "rows": [],
    }
    for label, stage, x_cg, alt in (
        ("stack at launch", 1, launch_cg, 500.0),
        ("stack at stage-1 burnout", 1, b1_cg, 4664.0),
        ("payload stage after separation", 2, s2_cg, 6000.0),
        ("payload stage at intercept", 2, s2d_cg, 19948.0),
    ):
        for mach in (1.5, 3.0, 4.0):
            x_cp = aero.x_cp(mach, alpha, stage)
            sm = static_margin_stage(x_cp, x_cg, aero.D_ref(stage))
            out["rows"].append(
                {
                    "config": label, "stage": stage, "mach": mach, "altitude": alt,
                    "x_cp_m": float(x_cp), "x_cg_m": float(x_cg),
                    "static_margin_cal": float(sm),
                }
            )
    out["worst_calibres"] = min(r["static_margin_cal"] for r in out["rows"])
    out["all_met"] = bool(out["worst_calibres"] >= float(reqs.static_margin_min))
    return out


def strake_record(R: dict[str, Any]) -> dict[str, Any]:
    """What the strakes are worth, and where they move the centre of pressure."""
    import copy

    from rocketgen.config_iv1 import default_iv1  # noqa: PLC0415

    dv, reqs, aero = R["dv"], R["reqs"], R["aero"]

    def strakes_off(vector):
        out = copy.deepcopy(vector)
        out.strakes = StrakeSpec(
            n=vector.strakes.n, height=0.0, length=vector.strakes.length,
            thickness=vector.strakes.thickness, x_le=vector.strakes.x_le,
            sweep_le=vector.strakes.sweep_le,
        )
        return out

    alpha = math.radians(10.0)

    def compare(vector) -> list[dict[str, Any]]:
        """CN_max and x_cp with and without the strakes, closed-form geometry on both sides."""
        on_aero = StackAero(vector, reqs)
        off_aero = StackAero(strakes_off(vector), reqs)
        out = []
        for mach in (1.5, 2.0, 3.0, 4.0, 5.0):
            for stage in (1, 2):
                on = on_aero.CN_max(mach, 15_000.0, stage, reqs.alpha_max)
                off = off_aero.CN_max(mach, 15_000.0, stage, reqs.alpha_max)
                d = on_aero.D_ref(stage)
                xon = on_aero.x_cp(mach, alpha, stage) / d
                xoff = off_aero.x_cp(mach, alpha, stage) / d
                out.append(
                    {
                        "mach": mach, "stage": stage,
                        "cn_max_on": float(on), "cn_max_off": float(off),
                        "cn_max_gain": float(on / off - 1.0),
                        "x_cp_on_cal": float(xon), "x_cp_off_cal": float(xoff),
                        "x_cp_shift_cal": float(xon - xoff),
                    }
                )
        return out

    rows = compare(dv)
    default_rows = compare(default_iv1())

    summary = aero.strake_summary(2)
    gains2 = [r["cn_max_gain"] for r in rows if r["stage"] == 2]
    gains1 = [r["cn_max_gain"] for r in rows if r["stage"] == 1]
    shifts1 = [r["x_cp_shift_cal"] for r in rows if r["stage"] == 1]
    shifts2 = [r["x_cp_shift_cal"] for r in rows if r["stage"] == 2]

    # Vortex against linear share of the strake load, at the alpha limit.
    pot, vor = aero.CN_strakes(3.0, reqs.alpha_max, 2)
    d_shifts1 = [r["x_cp_shift_cal"] for r in default_rows if r["stage"] == 1]
    d_gains2 = [r["cn_max_gain"] for r in default_rows if r["stage"] == 2]
    return {
        "summary": {k: float(v) for k, v in summary.items()},
        "summary_default": {
            k: float(v)
            for k, v in StackAero(default_iv1(), reqs).strake_summary(2).items()
        },
        "rows": rows,
        "default_rows": default_rows,
        "default_x_cp_shift_stage1_min": min(d_shifts1),
        "default_x_cp_shift_stage1_max": max(d_shifts1),
        "default_gain_stage2_min": min(d_gains2),
        "default_gain_stage2_max": max(d_gains2),
        "gain_stage2_min": min(gains2), "gain_stage2_max": max(gains2),
        "gain_stage1_min": min(gains1), "gain_stage1_max": max(gains1),
        "x_cp_shift_stage1_min": min(shifts1), "x_cp_shift_stage1_max": max(shifts1),
        "x_cp_shift_stage2_min": min(shifts2), "x_cp_shift_stage2_max": max(shifts2),
        "cn_strake_linear_at_limit": float(pot),
        "cn_strake_vortex_at_limit": float(vor),
        "vortex_share_at_limit": float(vor / (pot + vor)),
        "alpha_max_deg": math.degrees(reqs.alpha_max),
    }


def lateral_g_record(R: dict[str, Any], conv: dict[str, Any]) -> dict[str, Any]:
    """The A11 figure, and the altitude ceiling that a purely aerodynamic vehicle would face."""
    aero, dv, reqs = R["aero"], R["dv"], R["reqs"]
    ic = R["intercept"]
    S2 = aero.S_ref(2)
    cn = aero.CN_max(max(ic.mach, 0.3), ic.altitude, 2, reqs.alpha_max)
    mass = R["masses"].mass_after_separation()

    # The ceiling, with the converged CN_max rather than the audit's placeholder.
    ceiling = []
    for mach in (3.0, 4.0, 5.0):
        cnm = aero.CN_max(mach, 15_000.0, 2, reqs.alpha_max)
        h_lim = None
        for h in range(0, 40_000, 100):
            st = atmo(float(h))
            v = mach * st.speed_of_sound
            q = 0.5 * st.density * v * v
            if lateral_g(q, S2, cnm, mass) < reqs.lateral_g_min:
                h_lim = float(h)
                break
        ceiling.append({"mach": mach, "cn_max": float(cnm),
                        "h_limit_m": h_lim if h_lim is not None else 40_000.0})
    return {
        "aerodynamic_g": float(conv["lateral_g"]["aerodynamic"]),
        "acs_g": float(conv["lateral_g"]["acs"]),
        "total_g": float(conv["lateral_g"]["total"]),
        "cn_max_at_intercept": float(cn),
        "S_ref_stage1_m2": float(aero.S_ref(1)),
        "S_ref_stage2_m2": float(S2),
        "S_ref_ratio": float(aero.S_ref(2) / aero.S_ref(1)),
        "acs_thrust_N": float(dv.acs.thrust),
        "acs_g_check": float(lateral_g_acs(dv.acs.thrust, ic.mass)),
        "mass_after_separation_kg": float(mass),
        "ceiling": ceiling,
        "limit_g": float(reqs.lateral_g_min),
    }


def audit_record() -> dict[str, Any]:
    """Reproduce the SPEC_IV1.md section 2 requirements audit, so the figure has its own data.

    The stub aerodynamic model and the generous placeholder CN_max of 2.5 are the ones the audit
    used, so every lateral-acceleration number here is an UPPER bound on what the vehicle can do.
    """
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from iv1_envelope_probe import (  # noqa: PLC0415
        CN_MAX_PLACEHOLDER,
        closing_stack,
        envelope,
    )
    from test_trajectory_iv1 import StubStackAero  # noqa: PLC0415

    reqs = InterceptRequirements()
    dv = closing_stack()
    motor = MultiStageMotor(dv, reqs)
    sm = build_stack_masses(dv, reqs, motor=motor)

    def walk(gamma_deg: float) -> dict[str, Any]:
        """The same walk `iv1_envelope_probe.envelope` does, with the extra fields a figure
        needs. The shared fields are asserted against the probe below, so this cannot drift."""
        d = dv.replace(gamma_pitch=math.radians(gamma_deg))
        mission = AscentMission(d, reqs, motor, StubStackAero(), sm.m0)
        res = mission.fly(dt=0.05, adaptive=True, t_max=600.0)
        sep = res.diagnostics.get("separation_index", 0)
        S2 = dv.payload_stage.S_ref
        track = {"slant_km": [], "h_km": [], "mach": [], "g": []}
        best = {"slant_ok": 0.0, "h_ok": 0.0, "mach_ok": 0.0, "g_ok": 0.0,
                "slant_max": 0.0, "h_at_slant_max": 0.0, "g_at_slant_max": 0.0,
                "mach_at_slant_max": 0.0}
        for i in range(len(res.time)):
            h, x, V, M, mass = res.h[i], res.x[i], res.V[i], res.mach[i], res.mass[i]
            if h < 0.0:
                continue
            st = atmo(h)
            q = 0.5 * st.density * V * V
            S = S2 if i >= sep else dv.booster.S_ref
            g_av = lateral_g(q, S, CN_MAX_PLACEHOLDER, mass)
            sr = slant_range(x, h)
            if i % 8 == 0:
                track["slant_km"].append(round(sr / 1e3, 3))
                track["h_km"].append(round(h / 1e3, 4))
                track["mach"].append(round(M, 4))
                track["g"].append(round(g_av, 4))
            if sr > best["slant_max"]:
                best.update(slant_max=sr, h_at_slant_max=h, g_at_slant_max=g_av,
                            mach_at_slant_max=M)
            if (h >= reqs.h_intercept_min and M >= reqs.mach_intercept_min
                    and g_av >= reqs.lateral_g_min and sr > best["slant_ok"]):
                best.update(slant_ok=sr, h_ok=h, mach_ok=M, g_ok=g_av)
        best["gamma_deg"] = gamma_deg
        best["track"] = track
        return best

    rows = []
    for gamma in (12, 16, 20, 24, 28, 32, 40, 50):
        mine = walk(float(gamma))
        ref = envelope(dv, reqs, motor, sm.m0, float(gamma))
        for key, want in ref.items():
            got = mine[key]
            if not math.isclose(float(got), float(want), rel_tol=1e-12, abs_tol=1e-9):
                raise SystemExit(
                    f"the audit walk disagrees with scripts/iv1_envelope_probe.envelope at "
                    f"gamma {gamma}: {key} {got!r} against {want!r}"
                )
        rows.append(mine)

    best = max(rows, key=lambda r: r["slant_ok"])
    mass = sm.mass_after_separation()
    S2 = dv.payload_stage.S_ref

    def ceiling_at(mach: float) -> float:
        """Highest altitude at which 15 g is available aerodynamically, m."""
        for h in range(0, 40_000, 100):
            st = atmo(float(h))
            v = mach * st.speed_of_sound
            q = 0.5 * st.density * v * v
            if lateral_g(q, S2, CN_MAX_PLACEHOLDER, mass) < reqs.lateral_g_min:
                return float(h)
        return 40_000.0

    ceiling = [{"mach": m, "h_limit_m": ceiling_at(m)} for m in (3.0, 4.0, 5.0)]
    ceiling_fine = [
        {"mach": round(2.0 + 0.05 * i, 3), "h_limit_m": ceiling_at(2.0 + 0.05 * i)}
        for i in range(int((7.0 - 2.0) / 0.05) + 1)
    ]

    # Where 100 miles of slant range puts the intercept, on the same trajectories.
    return {
        "stack": {
            "m0_kg": float(sm.m0), "L_total_m": float(dv.L_total),
            "impulse_kNs": float(motor.total_impulse_vacuum() / 1e3),
            "jettisoned_kg": float(motor.jettisoned_mass()),
            "mass_after_separation_kg": float(mass),
        },
        "cn_max_placeholder": float(CN_MAX_PLACEHOLDER),
        "rows": rows,
        "best_slant_m": best["slant_ok"],
        "best_slant_miles": best["slant_ok"] / 1609.344,
        "best_gamma_deg": best["gamma_deg"],
        "best_altitude_m": best["h_ok"],
        "required_slant_m": float(reqs.slant_range_min),
        "required_slant_miles": float(reqs.slant_range_min_miles),
        "shortfall_m": float(reqs.slant_range_min) - best["slant_ok"],
        "shortfall_factor": float(reqs.slant_range_min) / best["slant_ok"],
        "ceiling": ceiling,
        "ceiling_fine": ceiling_fine,
        "h_intercept_min_m": float(reqs.h_intercept_min),
        "mach_intercept_min": float(reqs.mach_intercept_min),
        "lateral_g_min": float(reqs.lateral_g_min),
    }


def geometry_record(R: dict[str, Any]) -> dict[str, Any]:
    """nTop measurements against independent closed-form geometry.

    The comparison is done at the BASELINE design point, because that is the point whose
    closed-form reference was written to `runs/IV-1_geom/closed_form.json` by the geometry study.
    The converged point is reported alongside it, without a closed-form column, so no number is
    invented for it.
    """
    from rocketgen.config_iv1 import default_iv1  # noqa: PLC0415
    from rocketgen.ntopgen.stack_notebook import (  # noqa: PLC0415
        stack_geometry_closed_form,
        strake_solid_area,
    )

    cf = _json(os.path.join(GEOM_STUDY, "closed_form.json"))
    base = _json(os.path.join(GEOM_STUDY, "baseline", "iv1_stages.json"))
    small = _json(os.path.join(GEOM_STUDY, "small_strakes", "iv1_small_strakes_stages.json"))
    alt = _json(os.path.join(GEOM_STUDY, "alternate", "iv1_alt_stages.json"))
    sx = _json(os.path.join(GEOM_STUDY, "area_distribution", "iv1_sx_stages.json"))
    conv_meas = _json(os.path.join(GEOM, "iv1_stages.json"))

    # Re-derive the closed form for the baseline stack, and check it against the stored file, so
    # a change in the closed-form geometry cannot silently invalidate the comparison.
    fresh = stack_geometry_closed_form(default_iv1())
    for body, fields in cf.items():
        for name, want in fields.items():
            got = fresh[body][name]
            if not math.isclose(float(got), float(want), rel_tol=1e-9, abs_tol=1e-12):
                raise SystemExit(
                    f"closed_form.json disagrees with stack_geometry_closed_form on "
                    f"{body}.{name}: {got!r} against {want!r}"
                )

    key_for = {"s2": "2", "s1": "1", "is": "-1", "st": "0"}
    label = {"s2": "payload stage", "s1": "booster", "is": "interstage",
             "st": "stacked assembly"}
    rows = []
    for body in ("s2", "s1", "is", "st"):
        m = base[key_for[body]]
        for field, name in (("volume_total", "volume"),
                            ("area_wetted_body", "wetted area"),
                            ("area_wetted_strakes", "strake wetted area")):
            got, want = m.get(field), cf[body].get(field)
            if got is None or not want:
                continue
            rows.append(
                {
                    "body": label[body], "quantity": name,
                    "ntop": float(got), "closed_form": float(want),
                    "rel_err": float(got) / float(want) - 1.0,
                }
            )

    # The curated copy under examples/IV-1 and the working copy under runs/IV-1_geom must agree
    # on every measured value. Only the wall time may differ, because the test suite reruns the
    # baseline job. If a measured value ever differs, the two artefacts are not the same run.
    curated = _json(
        os.path.join(REPO, "examples", "IV-1", "measurements_baseline.json")
    )["raw"]
    for name, value in curated.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        prefix, _, field = name.partition("_")
        key = key_for.get(prefix)
        if key is None or field not in base[key]:
            continue
        got = base[key][field]
        if got is not None and not math.isclose(float(got), float(value), rel_tol=1e-12):
            raise SystemExit(
                f"examples/IV-1/measurements_baseline.json and "
                f"runs/IV-1_geom/baseline/iv1_stages.json disagree on {name}: "
                f"{value!r} against {got!r}"
            )

    dv0 = default_iv1()
    st = dv0.strakes
    solid = strake_solid_area(st, 0.5 * dv0.payload_stage.D)
    zero_thickness = st.wetted_area
    measured_strake_area = float(base["2"]["area_wetted_strakes"])
    measured_strake_vol = float(base["2"]["volume_strakes"])
    plate_vol = float(st.n) * st.height * st.length * st.thickness

    # Strake height sensitivity: the small-strake run halves the height.
    small_h = _json(os.path.join(GEOM_STUDY, "small_strakes", "iv1_small_strakes_input.json"))

    return {
        "rows": rows,
        "worst_volume_err": max(abs(r["rel_err"]) for r in rows if r["quantity"] == "volume"),
        "worst_area_err": max(abs(r["rel_err"]) for r in rows if r["quantity"] == "wetted area"),
        "strake": {
            "measured_area_m2": measured_strake_area,
            "solid_closed_form_m2": float(solid),
            "solid_rel_err": measured_strake_area / float(solid) - 1.0,
            "zero_thickness_m2": float(zero_thickness),
            "solid_over_zero_thickness": float(solid) / float(zero_thickness),
            "measured_volume_m3": measured_strake_vol,
            "plate_volume_m3": float(plate_vol),
            "volume_rel_err": measured_strake_vol / float(plate_vol) - 1.0,
            "small_area_m2": float(small["2"]["area_wetted_strakes"]),
            "small_volume_m3": float(small["2"]["volume_strakes"]),
            "small_over_baseline_area": float(small["2"]["area_wetted_strakes"])
            / measured_strake_area,
            "small_over_baseline_volume": float(small["2"]["volume_strakes"])
            / measured_strake_vol,
            "small_input": {k: v for k, v in small_h.items() if "strake" in k.lower()},
        },
        "area_distribution": [
            [float(a), float(b)] for a, b in sx["0"]["area_distribution"]
        ],
        # Wall times come from the CURATED copies under examples/IV-1, not from the working
        # directories under runs/IV-1_geom. The test suite drives real `ntopcl` and rewrites
        # `runs/IV-1_geom/baseline`, so the wall time there is whoever ran last. The measured
        # values themselves are deterministic and agree between the two, which is checked below.
        "wall_times_s": {
            name: float(
                _json(os.path.join(REPO, "examples", "IV-1", f"measurements_{name}.json"))
                ["measurements"]["wall_time_s"]
            )
            for name in ("baseline", "small_strakes", "alternate", "area_distribution")
        }
        | {"converged": float(conv_meas["0"]["wall_time_s"])},
        "converged_bodies": {
            k: {
                "body": v["body"],
                "volume_total": v["volume_total"],
                "volume_structure": v["volume_structure"],
                "volume_cavity": v["volume_cavity"],
                "area_wetted_body": v["area_wetted_body"],
                "area_wetted_fins": v["area_wetted_fins"],
                "area_wetted_strakes": v["area_wetted_strakes"],
                "mass_structure": v["mass_structure"],
                "length": v["length"],
                "x_forward": v["x_forward"],
                "cg_structure": v["cg_structure"],
                "inertia_structure": v["inertia_structure"],
            }
            for k, v in conv_meas.items()
        },
    }


def staging_record() -> dict[str, Any]:
    """The point of staging: same mass, same propellant, same clock, same gravity loss."""
    from test_trajectory_iv1 import (  # noqa: PLC0415
        _M0,
        _M_JETTISON,
        _MP1,
        _MP2,
        _T_TOTAL,
        _VE,
        _single_stage_motor,
        _staged_motor,
        _vertical_vacuum_mission,
    )

    sm = _staged_motor()
    staged = _vertical_vacuum_mission(sm, _M0).fly(
        dt=0.01, t_max=sm.t_all_burnout, adaptive=False
    )
    ss = _single_stage_motor()
    single = _vertical_vacuum_mission(ss, _M0, n_stages=1).fly(
        dt=0.01, t_max=ss.t_all_burnout, adaptive=False
    )

    m1 = _M0 - _MP1
    analytic = _VE * (
        math.log((m1 - _M_JETTISON) / (m1 - _M_JETTISON - _MP2))
        - math.log(m1 / (m1 - _MP2))
    )
    gain = staged.V[-1] - single.V[-1]
    return {
        "m0_kg": float(_M0), "propellant_kg": float(_MP1 + _MP2),
        "jettisoned_kg": float(_M_JETTISON), "burn_time_s": float(_T_TOTAL),
        "ve_m_s": float(_VE),
        "v_staged": float(staged.V[-1]), "v_single": float(single.V[-1]),
        "gain_m_s": float(gain), "gain_percent": float(100.0 * gain / single.V[-1]),
        "analytic_gain_m_s": float(analytic),
        "analytic_rel_err": float(abs(gain - analytic) / analytic),
        "h_staged": float(staged.h[-1]), "h_single": float(single.h[-1]),
        "gravity_loss_m_s": float(9.80665 * _T_TOTAL),
    }


def atmosphere_record(R: dict[str, Any]) -> dict[str, Any]:
    """The atmosphere-table extension, and the geopotential near-miss."""
    R0 = 6_356_766.0        # the effective earth radius US Standard 1976 uses, m

    def geopotential(z: float) -> float:
        return R0 * z / (R0 + z)

    def geometric(hh: float) -> float:
        return R0 * hh / (R0 - hh)

    tb = table()
    grid = tb["altitude"]
    worst = 0.0
    probe = np.linspace(-5_000.0, H_MAX + 5_000.0, 4001)
    for name in ("pressure", "temperature", "density", "speed_of_sound",
                 "dynamic_viscosity"):
        got = np.array([getattr(atmo(float(h)), name) for h in probe])
        ref = np.interp(np.clip(probe, 0.0, H_MAX), grid, tb[name])
        worst = max(worst, float(np.max(np.abs(got - ref) / np.abs(ref))))

    t0 = time.perf_counter()
    for h in probe:
        atmo(float(h))
    t_fast = time.perf_counter() - t0

    apogee = max(R["traj"].h)
    return {
        "h_max_m": float(H_MAX),
        "h_max_legacy_m": float(H_MAX_LEGACY),
        "n_nodes": int(grid.size),
        "step_m": float(grid[1] - grid[0]),
        "apogee_m": float(apogee),
        "rho_at_apogee": float(atmo(apogee).density),
        "rho_at_legacy_ceiling": float(atmo(H_MAX_LEGACY).density),
        "clamp_drag_overstatement": float(
            atmo(H_MAX_LEGACY).density / atmo(apogee).density
        ),
        "rho_50km": float(atmo(50_000.0).density),
        "clamp_overstatement_50km": float(
            atmo(H_MAX_LEGACY).density / atmo(50_000.0).density
        ),
        "p_at_47km_geometric": float(atmo(47_000.0).pressure),
        "geopotential_of_47km_geometric_m": geopotential(47_000.0),
        "p_at_47km_geopotential_row": 110.906,
        "p_at_geometric_for_47km_geopotential": float(
            atmo(geometric(47_000.0)).pressure
        ),
        "naive_error": float(atmo(47_000.0).pressure / 110.906 - 1.0),
        "interp_agreement": worst,
        "lookup_us_per_call": 1.0e6 * t_fast / probe.size,
        "stratopause_T": float(atmo(geometric(48_000.0)).temperature),
    }


def motor_record(R: dict[str, Any]) -> dict[str, Any]:
    """Per-stage motor operating points, and the grain that binds."""
    motor, dv = R["motor"], R["dv"]
    out: dict[str, Any] = {"stages": {}, "warnings": list(motor.warnings)}
    for index in motor.stage_indices:
        op = motor.operating_point(index)
        g = motor.grain_geometry(index)
        inert = motor.inert_mass_breakdown(index)
        out["stages"][str(index)] = {
            "operating_point": {k: (float(v) if isinstance(v, (int, float)) else v)
                                for k, v in op.items()},
            "grain": {
                "L_over_D": float(g.L_over_D),
                "volumetric_loading": float(g.volumetric_loading),
                "feasible": bool(g.feasible),
                "d_outer": float(g.d_outer),
                "d_inner": float(g.d_inner),
                "length": float(g.length),
                "web": float(g.web),
                "bay_diameter": float(g.bay_diameter),
                "bay_length_available": float(g.bay_length_available),
                "burning_area": float(g.burning_area),
                "volume_total": float(g.volume_total),
            },
            "inert": {k: (float(v) if isinstance(v, (int, float)) else v)
                      for k, v in inert.items()},
        }
    out["total_impulse_vacuum_Ns"] = float(motor.total_impulse_vacuum())
    out["jettisoned_kg"] = float(motor.jettisoned_mass())
    out["throat_credibility"] = motor.throat_credibility_report()
    out["nozzle_exit_diameter_1_m"] = float(motor.operating_point(1)["exit_diameter"])
    out["booster_diameter_m"] = float(dv.booster.D)
    return out


def grain_limits_record(R: dict[str, Any]) -> dict[str, Any]:
    """What the tubular-grain assumption costs, measured rather than asserted.

    Three separate measurements, all cheap because none of them flies a trajectory:

    1. The default stack, which does not close. The model reports it instead of smoothing it.
    2. A booster-thrust sweep on the converged geometry, which finds the thrust above which a
       tubular grain stops fitting its bay.
    3. A stage-2 propellant sweep, which finds the charge above which the web is wider than the
       bay radius.
    """
    from rocketgen.config_iv1 import default_iv1  # noqa: PLC0415

    reqs = R["reqs"]
    d0 = default_iv1()
    m0 = MultiStageMotor(d0, reqs)
    default = {
        "D1_m": float(d0.booster.D),
        "eps_nozzle_1": float(d0.booster.eps_nozzle),
        "F1_N": float(d0.booster.F_thrust),
        "exit_diameter_1_m": float(m0.operating_point(1)["exit_diameter"]),
        "exit_fits_in_body": bool(
            m0.operating_point(1)["exit_diameter"] <= d0.booster.D
        ),
        "vol_loading_1": float(m0.grain_geometry(1).volumetric_loading),
        "vol_loading_2": float(m0.grain_geometry(2).volumetric_loading),
        "feasible_1": bool(m0.grain_geometry(1).feasible),
        "feasible_2": bool(m0.grain_geometry(2).feasible),
        "warnings": list(m0.warnings),
    }

    dv = R["dv"]
    thrust_rows = []
    for f_kN in (45.0, 60.0, 80.0, 100.0, 110.0, 120.0, 150.0, 200.0):
        d = dv.with_path("stages.0.F_thrust", f_kN * 1e3)
        m = MultiStageMotor(d, reqs)
        g = m.grain_geometry(1)
        op = m.operating_point(1)
        thrust_rows.append(
            {
                "F1_kN": f_kN, "L_over_D": float(g.L_over_D),
                "grain_length_m": float(g.length),
                "bay_length_m": float(g.bay_length_available),
                "vol_loading": float(g.volumetric_loading),
                "web_m": float(g.web),
                "bay_radius_m": 0.5 * float(g.bay_diameter),
                "exit_diameter_m": float(op["exit_diameter"]),
                "feasible": bool(g.feasible),
                "warnings": list(g.warnings),
            }
        )

    prop_rows = []
    for mp in (70.0, 80.0, 90.0, 95.0, 100.0, 110.0, 130.0):
        d = dv.with_path("stages.1.m_propellant", mp)
        m = MultiStageMotor(d, reqs)
        g = m.grain_geometry(2)
        prop_rows.append(
            {
                "m_p2_kg": mp, "L_over_D": float(g.L_over_D),
                "vol_loading": float(g.volumetric_loading),
                "web_m": float(g.web),
                "bay_radius_m": 0.5 * float(g.bay_diameter),
                "feasible": bool(g.feasible),
                "warnings": list(g.warnings),
            }
        )

    ok_thrust = [r["F1_kN"] for r in thrust_rows if r["feasible"]]
    ok_prop = [r["m_p2_kg"] for r in prop_rows if r["feasible"]]
    return {
        "default_stack": default,
        "thrust_sweep": thrust_rows,
        "propellant_sweep": prop_rows,
        "max_feasible_F1_kN": max(ok_thrust) if ok_thrust else None,
        "max_feasible_m_p2_kg": max(ok_prop) if ok_prop else None,
    }


def source_provenance() -> dict[str, Any]:
    """Every registered source that is a guess, a modelling choice, an approximation.

    ALL FIVE IV-1 modules are imported at module load above, so the registry is complete before
    it is read. The SV-1 report under-reported its own limitations by a factor of four because it
    read the registry before the lazily imported modules had registered into it.
    """
    import importlib

    for dotted in IV1_MODULES:
        importlib.import_module(dotted)

    owner: dict[str, str] = {}
    for short, dotted in SOURCE_MODULES:
        mod = importlib.import_module(dotted)
        for key in getattr(mod, "SOURCES", {}):
            owner.setdefault(key, short)

    flagged = []
    for key, text in sorted(_SOURCES_REGISTRY.items()):
        low = text.lower()
        hits = [w for w in FLAG_WORDS if w in low]
        if hits:
            flagged.append(
                {"key": key, "module": owner.get(key, "config"), "text": text,
                 "flags": hits}
            )
    modules = sorted({f["module"] for f in flagged})
    return {
        "n_registered": len(_SOURCES_REGISTRY),
        "n_flagged": len(flagged),
        "iv1_modules_imported": list(IV1_MODULES),
        "modules_attributed": [d for _s, d in SOURCE_MODULES],
        "flag_words": list(FLAG_WORDS),
        "flagged": flagged,
        "by_module": {m: sum(1 for f in flagged if f["module"] == m) for m in modules},
    }


def notebook_record() -> dict[str, Any]:
    """The nTop authoring facts, read from the module that owns them."""
    from rocketgen.ntopgen import stack_notebook as SN  # noqa: PLC0415
    from rocketgen.ntopgen.universe import Universe  # noqa: PLC0415

    n_blocks = None
    n_inputs = None
    input_names: list[str] = []
    recipe = None
    cache = os.path.join(REPO, "runs", "_ntop_cache")
    if os.path.isdir(cache):
        cands = [f for f in os.listdir(cache) if f.startswith("iv1_")
                 and f.endswith("_recipe.json")]
        if cands:
            recipe = os.path.join(cache, sorted(cands)[0])
            r = _json(recipe)
            n_blocks = len(r.get("body", []))
            input_names = [str(i.get("name")) for i in r.get("inputs", [])]
            n_inputs = len(input_names)

    n_signatures = None
    try:
        n_signatures = len(Universe.load())
    except Exception:                                        # pragma: no cover
        pass

    return {
        "n_blocks": n_blocks,
        "n_inputs": n_inputs,
        "input_names": input_names,
        "recipe_example": os.path.relpath(recipe, REPO) if recipe else None,
        "n_universe_signatures": n_signatures,
        "mesh_tolerance_m": float(SN.DEFAULT_MESH_TOLERANCE),
        "relative_error": float(SN.DEFAULT_RELATIVE_ERROR),
        "area_relative_error": float(SN.DEFAULT_AREA_RELATIVE_ERROR),
        "cad_tolerance_m": float(SN.DEFAULT_CAD_TOLERANCE),
        "measured_wall_time_source": SN.SOURCES["measured_wall_time"],
    }


#: Which test module belongs to which vehicle. SV-1 is the regression baseline, so its count has
#: to be reported separately and must not move.
SV1_TESTS = ("test_aero", "test_masses", "test_propulsion", "test_trajectory", "test_doe",
             "test_ntopgen", "test_rocket_notebook")
IV1_TESTS = ("test_aero_iv1", "test_propulsion_iv1", "test_trajectory_iv1",
             "test_stack_notebook", "test_atmosphere_high")


def tests_record() -> dict[str, Any]:
    """Test counts. The suite result is READ from the recorded run, never claimed.

    `runs/IV-1/pytest.txt` is the captured output of `pytest tests -q`. The per-module split is
    recollected here, because collection is fast and the full suite is not: it drives real
    `ntopcl` subprocesses and takes about ten minutes.
    """
    import re
    import subprocess

    log = os.path.join(IV1, "pytest.txt")
    passed = skipped = failed = None
    duration = None
    if os.path.isfile(log):
        text = open(log, encoding="utf-8").read()
        m = re.search(r"(\d+) passed(?:, (\d+) skipped)?(?:, (\d+) failed)?"
                      r" in ([\d.]+)s", text)
        if m:
            passed = int(m.group(1))
            skipped = int(m.group(2) or 0)
            failed = int(m.group(3) or 0)
            duration = float(m.group(4))

    python = os.path.join(REPO, ".venv", "Scripts", "python.exe")
    per_module: dict[str, int] = {}
    for name in SV1_TESTS + IV1_TESTS:
        path = os.path.join(REPO, "tests", f"{name}.py")
        if not os.path.isfile(path):
            continue
        try:
            out = subprocess.run(
                [python, "-m", "pytest", path, "--collect-only", "-q"],
                cwd=REPO, capture_output=True, text=True, timeout=300,
            ).stdout
        except Exception:                                    # pragma: no cover
            continue
        per_module[name] = sum(1 for line in out.splitlines() if "::" in line)

    return {
        "log": "runs/IV-1/pytest.txt",
        "command": ".venv/Scripts/python.exe -m pytest tests -q",
        "passed": passed, "skipped": skipped, "failed": failed,
        "duration_s": duration,
        "per_module": per_module,
        "sv1_total": sum(per_module.get(n, 0) for n in SV1_TESTS),
        "iv1_total": sum(per_module.get(n, 0) for n in IV1_TESTS),
        "collected_total": sum(per_module.values()),
    }


def environment_record() -> dict[str, Any]:
    """Versions, and the calibration state of the IV-1 aerodynamic model."""
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
        # scripts/iv1_converge.py builds `StackAero` directly and does NOT wrap it in the
        # `loop.CalibratedAero` boundary that the SV-1 sizing loop uses. The IV-1 result is
        # therefore uncalibrated. Recorded here so the report can say so.
        "cd0_calibration_applied_to_iv1": False,
    }


# --------------------------------------------------------------------------------------
#   Main
# --------------------------------------------------------------------------------------


def main() -> str:
    os.makedirs(FIGS, exist_ok=True)
    conv = _json(os.path.join(IV1, "converged.json"))
    R = refly(conv)

    payload = {
        "generated_from": "runs/IV-1/converged.json + runs/IV-1/geom/iv1_stages.json",
        "refly_wall_time_s": R["wall_time_s"],
        "trajectory": trajectory_record(R),
        "mass": mass_record(R, conv),
        "constraints": constraint_record(conv),
        "static_margin": static_margin_record(R, conv),
        "strakes": strake_record(R),
        "lateral_g": lateral_g_record(R, conv),
        "audit": audit_record(),
        "geometry": geometry_record(R),
        "staging": staging_record(),
        "atmosphere": atmosphere_record(R),
        "motor": motor_record(R),
        "grain_limits": grain_limits_record(R),
        "sources": source_provenance(),
        "notebook": notebook_record(),
        "environment": environment_record(),
        "tests": tests_record(),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=str)
    print(f"wrote {os.path.relpath(OUT_JSON, REPO)} "
          f"({os.path.getsize(OUT_JSON) / 1024:.0f} KB)")
    print(f"  re-flight reproduced converged.json; {payload['trajectory']['n_samples']} samples")
    print(f"  flagged sources: {payload['sources']['n_flagged']} of "
          f"{payload['sources']['n_registered']} registered")
    return OUT_JSON


if __name__ == "__main__":
    main()
