"""WP7 evidence collector: every number the report needs that is not already on disk.

The converged run, the DOE and the provenance registry are all written by `run_sv1.py`. This
module adds the numbers that support the FINDINGS, and writes them to one JSON file so the
report and the figure scripts read from disk and never recompute:

    runs/SV-1/figures/evidence.json

What is measured here, and why it cannot come from the existing artefacts:

  1. Integrator validation residuals. The tests assert loose bounds; the report quotes the
     residual actually achieved, so it must be measured.
  2. Aero validation bias and worst-case error against Dupuis and Hathaway, Table VII.
  3. nTop measurements against the closed form, AT THE CONVERGED DESIGN. `docs/NTOP_NOTES.md`
     section 15 gives the same comparison for the DEFAULT design vector, which is a different
     shape.
  4. The unpowered-dive sweep. `tests/test_trajectory.py` proves impact Mach stays below 1.1;
     the report quotes the numbers, so they are measured here.
  5. The terminal-propellant sweep at the converged design. This is the only measurement that
     shows the q_max active-constraint shift and the range cost per kilogram.
  6. The flown motor operating point, built exactly as `loop.converge_point` builds it.

Run:
    .venv/Scripts/python.exe -m rocketgen.report.evidence
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from typing import Any

from ..config import (
    CD0_CALIBRATION,
    MATERIALS,
    RUNS_DIR,
    REPO_ROOT,
    DesignVector,
    Requirements,
)

CONVERGED_DIR = os.path.join(RUNS_DIR, "SV-1", "converged")
FIG_DIR = os.path.join(RUNS_DIR, "SV-1", "figures")
EVIDENCE_JSON = os.path.join(FIG_DIR, "evidence.json")

#: Terminal-propellant loadings for the sweep, kg. 0 is the unpowered dive; 40 is the converged
#: design; the bounds come from DesignVector.bounds()["m_p_terminal"].
TERMINAL_SWEEP_KG: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0)

#: Dive angles for the unpowered sweep, deg. Same set as tests/test_trajectory.py.
DIVE_ANGLES_DEG: tuple[float, ...] = (-25.0, -35.0, -50.0, -70.0, -89.0)


# --------------------------------------------------------------------------------------
#   Helpers
# --------------------------------------------------------------------------------------


def load_point(name: str) -> dict[str, Any]:
    with open(os.path.join(CONVERGED_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def design_vector_from(payload: dict[str, Any]) -> DesignVector:
    """Rebuild a DesignVector from a dumped design_vector dict, ignoring derived keys."""
    fields = DesignVector.__dataclass_fields__
    return DesignVector(**{k: v for k, v in payload.items() if k in fields})


# --------------------------------------------------------------------------------------
#   1. Trajectory integrator against closed forms
# --------------------------------------------------------------------------------------


def integrator_residuals() -> dict[str, Any]:
    """Relative residual of each analytic trajectory case, plus the RK4 order ratios."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    import tests.test_trajectory as tt

    from ..sizing import trajectory as T

    g = T.G0
    out: dict[str, Any] = {}

    # (a) vacuum ballistic range and apogee
    analytic_range = tt.BALLISTIC_V0 ** 2 * math.sin(2.0 * tt.BALLISTIC_GAMMA) / g
    result = tt._fly_ballistic(0.02)
    out["vacuum_range_rel"] = abs(result.range_final / analytic_range - 1.0)
    analytic_apogee = (tt.BALLISTIC_V0 * math.sin(tt.BALLISTIC_GAMMA)) ** 2 / (2.0 * g)
    out["vacuum_apogee_rel"] = abs(T.apogee(result) / analytic_apogee - 1.0)

    # (b) RK4 order: halving dt must cut the error by about 16
    errors = [abs(tt._fly_ballistic(dt).range_final - analytic_range) for dt in (0.8, 0.4, 0.2, 0.1)]
    out["rk4_dt"] = [0.8, 0.4, 0.2, 0.1]
    out["rk4_order_ratios"] = [a / b for a, b in zip(errors, errors[1:])]

    # (c) Tsiolkovsky, vertical drag-free burn
    m0, mdot, thrust, burn_time, v0 = 600.0, 12.0, 60_000.0, 20.0, 100.0
    integrator = T.PointMass3DOF(tt._constant_thrust(thrust, mdot))
    state0 = T.FlightState(t=0.0, V=v0, gamma=0.5 * math.pi, x=0.0, h=0.0, m=m0)
    burn = integrator.integrate(
        state0, dt=0.005, t_max=burn_time, velocity_floor=0.0, stop_on_ground=False
    )
    mass_final = m0 - mdot * burn_time
    analytic_v = v0 + (thrust / mdot) * math.log(m0 / mass_final) - g * burn_time
    out["tsiolkovsky_rel"] = abs(burn.V[-1] / analytic_v - 1.0)

    # (d) terminal velocity
    mass, rho, area, cd = 400.0, 1.225, 0.0962, 0.50
    analytic_vt = math.sqrt(2.0 * mass * g / (rho * area * cd))
    integrator = T.PointMass3DOF(tt._constant_density_drag(rho, area, cd))
    state0 = T.FlightState(t=0.0, V=10.0, gamma=-0.5 * math.pi, x=0.0, h=50_000.0, m=mass)
    fall = integrator.integrate(
        state0, dt=0.01, t_max=400.0, velocity_floor=0.0, stop_on_ground=False
    )
    out["terminal_velocity_rel"] = abs(fall.V[-1] / analytic_vt - 1.0)

    # (e) specific-energy conservation over 100 s
    integrator = T.PointMass3DOF(tt._no_forces)
    state0 = T.FlightState(t=0.0, V=600.0, gamma=math.radians(30.0), x=0.0, h=12_000.0, m=500.0)
    glide = integrator.integrate(
        state0, dt=0.02, t_max=100.0, velocity_floor=0.0, stop_on_ground=False
    )
    energy = T.specific_energy(glide)
    out["energy_drift_rel"] = max(abs(e - energy[0]) for e in energy) / abs(energy[0])
    return out


# --------------------------------------------------------------------------------------
#   2. Aero validation against the Basic Finner free-flight data
# --------------------------------------------------------------------------------------


def aero_validation() -> dict[str, Any]:
    """Mean bias and worst error on CD0, CN_alpha and x_cp against DREV-TM-9703 Table VII."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from tests.test_aero import (
        ALPHA_CMP,
        BASIC_FINNER_D,
        BASIC_FINNER_TABLE_VII,
        BASIC_FINNER_XCG_CAL,
        DBSQ_MIN,
        MACH_BANDS,
        TOL_CD0_BAND,
        TOL_CD0_SHOT,
        TOL_CNA_BAND,
        TOL_CNA_SHOT,
        TOL_XCP_BAND,
        TOL_XCP_SHOT,
        basic_finner_dv,
    )

    from ..sizing.aero import RocketAero

    finner = RocketAero(basic_finner_dv(), nose_shape="cone")
    table = BASIC_FINNER_TABLE_VII

    cd0_err = [
        finner.evaluate(m, 0.0, ALPHA_CMP).CD0 / cd0 - 1.0
        for m, _d, cd0, _c, _cm in table
        if m >= 1.4
    ]
    cna_err: list[float] = []
    xcp_err: list[float] = []
    for mach, dbsq, _cd0, cna, cma in table:
        if mach < 1.4 or dbsq < DBSQ_MIN:
            continue
        r = finner.evaluate(mach, 0.0, ALPHA_CMP)
        cna_err.append(r.CN_alpha / cna - 1.0)
        xcp_err.append((r.x_cp / BASIC_FINNER_D) / (BASIC_FINNER_XCG_CAL - cma / cna) - 1.0)

    bands: list[dict[str, Any]] = []
    for lo, hi in MACH_BANDS:
        all_in = [r for r in table if lo <= r[0] < hi]
        good = [r for r in all_in if r[1] >= DBSQ_MIN]
        row: dict[str, Any] = {"lo": lo, "hi": hi, "n_all": len(all_in), "n_good": len(good)}
        if all_in:
            exp = statistics.mean(r[2] for r in all_in)
            mod = statistics.mean(finner.evaluate(r[0], 0.0, ALPHA_CMP).CD0 for r in all_in)
            row["cd0_err"] = mod / exp - 1.0
        if good:
            exp = statistics.mean(r[3] for r in good)
            mod = statistics.mean(finner.evaluate(r[0], 0.0, ALPHA_CMP).CN_alpha for r in good)
            row["cna_err"] = mod / exp - 1.0
            exp = statistics.mean(BASIC_FINNER_XCG_CAL - r[4] / r[3] for r in good)
            mod = statistics.mean(
                finner.evaluate(r[0], 0.0, ALPHA_CMP).x_cp / BASIC_FINNER_D for r in good
            )
            row["xcp_err"] = mod / exp - 1.0
        bands.append(row)

    return {
        "n_shots_table": len(table),
        "n_shots_cd0": len(cd0_err),
        "n_shots_cna_xcp": len(cna_err),
        "cd0_mean_bias": statistics.mean(cd0_err),
        "cd0_worst_shot": max(abs(e) for e in cd0_err),
        "cna_mean_bias": statistics.mean(cna_err),
        "cna_worst_shot": max(abs(e) for e in cna_err),
        "xcp_mean_bias": statistics.mean(xcp_err),
        "xcp_worst_shot": max(abs(e) for e in xcp_err),
        "cd0_worst_band": max(abs(b["cd0_err"]) for b in bands if "cd0_err" in b),
        "cna_worst_band": max(abs(b["cna_err"]) for b in bands if "cna_err" in b),
        "xcp_worst_band": max(abs(b["xcp_err"]) for b in bands if "xcp_err" in b),
        "tolerances": {
            "cd0_band": TOL_CD0_BAND, "cd0_shot": TOL_CD0_SHOT,
            "cna_band": TOL_CNA_BAND, "cna_shot": TOL_CNA_SHOT,
            "xcp_band": TOL_XCP_BAND, "xcp_shot": TOL_XCP_SHOT,
        },
        "bands": bands,
        "cd0_calibration": CD0_CALIBRATION,
    }


# --------------------------------------------------------------------------------------
#   3. nTop measurements against the closed form
# --------------------------------------------------------------------------------------


def ntop_versus_closed_form() -> dict[str, Any]:
    """Compare every nTop measurement at the converged design with the closed form."""
    from ..sizing.masses import analytic_geometry

    point = load_point("point_ntop.json")
    dv = design_vector_from(point["design_vector"])
    measured = point["ntop_measurements"]
    closed = analytic_geometry(dv)

    rows = []
    for key in ("volume_total", "area_wetted_body", "area_wetted_fins", "area_base"):
        me, cf = measured.get(key), closed.get(key)
        if me is None or not cf:
            continue
        rows.append({"quantity": key, "ntop": me, "closed_form": cf, "rel_err": me / cf - 1.0})

    rho_airframe = MATERIALS["airframe_al7075"].density
    return {
        "rows": rows,
        "volume_structure": measured["volume_structure"],
        "volume_cavity": measured["volume_cavity"],
        "mass_structure": measured["mass_structure"],
        "billet_mass": measured["volume_total"] * rho_airframe,
        "airframe_density": rho_airframe,
        "cg_structure": measured["cg_structure"],
        "inertia_structure": measured["inertia_structure"],
        "wall_time_s": measured["wall_time_s"],
        "fin_area_note": (
            "The closed form counts two sides of each exposed panel only. The nTop plate has "
            "edges as well, so a positive error is expected."
        ),
    }


# --------------------------------------------------------------------------------------
#   4. The unpowered terminal dive
# --------------------------------------------------------------------------------------


def unpowered_dive_sweep() -> dict[str, Any]:
    """Impact Mach against dive angle for an UNPOWERED dive, plus the closed-form asymptote.

    The sweep runs the full loop at the converged design with the terminal charge removed, so
    the aerodynamics, the motor and the mass statement are the real ones. `tests/simple_aero.py`
    is deliberately not used: its drag coefficient is a made-up constant.

    The closed-form asymptote is solved self-consistently, because the calibrated drag
    coefficient itself depends on Mach: V = sqrt(2*m*g/(rho*S*CD(V/a))).
    """
    from ..sizing import trajectory as T
    from ..sizing.aero import RocketAero
    from ..sizing.loop import CalibratedAero, converge_point

    point = load_point("point_ntop.json")
    base = design_vector_from(point["design_vector"])
    # Terminal charge removed, and returned to the sustain charge so the total is unchanged.
    dv = base.replace(
        m_p_terminal=0.0, m_p_sustain=base.m_p_sustain + base.m_p_terminal
    )

    sweep = []
    for gamma_deg in DIVE_ANGLES_DEG:
        reqs = Requirements()
        reqs.gamma_terminal = math.radians(gamma_deg)
        res = converge_point(dv, reqs, geometry_fn=None, max_iter=2, dt=0.06, adaptive=True)
        sweep.append(
            {
                "gamma_deg": gamma_deg,
                "impact_mach": res.traj.mach_final if res.traj is not None else None,
                "range_km": res.traj.range_final / 1000.0 if res.traj is not None else None,
                "q_max_kPa": res.traj.q_max / 1000.0 if res.traj is not None else None,
                "burnout_kg": (
                    res.masses.excluding(*_propellant_items())[0]
                    if res.masses is not None
                    else None
                ),
                "converged": res.converged,
            }
        )

    burnout_kg = next(r["burnout_kg"] for r in sweep if r["burnout_kg"] is not None)
    rho, _p, _t, sound = T.atmosphere_properties(0.0)
    aero = CalibratedAero(RocketAero(dv, nose_shape=dv.nose_shape), factor=CD0_CALIBRATION)

    # Fixed-point solve of the vertical-fall asymptote with the Mach-dependent drag.
    v = 300.0
    for _ in range(200):
        cd = aero.evaluate(v / sound, 0.0, math.radians(0.5)).CD
        v_next = math.sqrt(2.0 * burnout_kg * T.G0 / (rho * dv.S_ref * cd))
        if abs(v_next - v) < 1.0e-9:
            v = v_next
            break
        v = 0.5 * (v + v_next)
    cd_at_solution = aero.evaluate(v / sound, 0.0, math.radians(0.5)).CD

    fixed_cd = [
        {
            "CD": cd,
            "v_terminal": math.sqrt(2.0 * burnout_kg * T.G0 / (rho * dv.S_ref * cd)),
        }
        for cd in (0.338, 0.45, 0.60)
    ]
    for row in fixed_cd:
        row["mach"] = row["v_terminal"] / sound

    v_required = 1.50 * sound
    return {
        "sweep": sweep,
        "burnout_kg": burnout_kg,
        "self_consistent": {
            "v_terminal": v,
            "mach": v / sound,
            "CD": cd_at_solution,
        },
        "fixed_cd": fixed_cd,
        "cd_needed_for_mach_1p50": (
            2.0 * burnout_kg * T.G0 / (rho * dv.S_ref * v_required ** 2)
        ),
        "v_required_for_mach_1p50": v_required,
        "q_at_mach_1p50_sea_level": 0.5 * rho * v_required ** 2,
        "q_at_90kPa_mach": math.sqrt(2.0 * 90_000.0 / rho) / sound,
        "sea_level_density": rho,
        "sea_level_sound_speed": sound,
        "S_ref": dv.S_ref,
    }


def _propellant_items() -> tuple[str, ...]:
    from ..sizing.masses import PROPELLANT_ITEMS

    return PROPELLANT_ITEMS


# --------------------------------------------------------------------------------------
#   5. Terminal-propellant sweep at the converged design
# --------------------------------------------------------------------------------------


def terminal_propellant_sweep() -> dict[str, Any]:
    """Sweep m_p_terminal at the converged design, analytic geometry, DOE integration step.

    The terminal charge is taken OUT of the sustain charge, exactly as `run_sv1.py` does, so the
    total propellant mass is held constant and the sweep isolates the terminal phase.
    """
    from ..sizing.loop import converge_point

    point = load_point("point_ntop.json")
    base = design_vector_from(point["design_vector"])
    propellant_pool = base.m_p_sustain + base.m_p_terminal
    reqs = Requirements()

    rows = []
    for m_terminal in TERMINAL_SWEEP_KG:
        dv = base.replace(
            m_p_terminal=m_terminal, m_p_sustain=propellant_pool - m_terminal
        )
        res = converge_point(dv, reqs, geometry_fn=None, max_iter=2, dt=0.06, adaptive=True)
        rows.append(
            {
                "m_p_terminal": m_terminal,
                "m_p_sustain": dv.m_p_sustain,
                "m0_kg": res.masses.total if res.masses is not None else None,
                "range_km": res.traj.range_final / 1000.0 if res.traj is not None else None,
                "mach_terminal": res.traj.mach_final if res.traj is not None else None,
                "q_max_kPa": res.traj.q_max / 1000.0 if res.traj is not None else None,
                "feasible": res.feasible,
                "violations": [c.name for c in res.constraints if not c.met],
            }
        )

    # Range cost per kilogram of terminal propellant: least-squares slope over the sweep.
    xs = [r["m_p_terminal"] for r in rows if r["range_km"] is not None]
    ys = [r["range_km"] for r in rows if r["range_km"] is not None]
    slope = None
    if len(xs) > 1:
        mx, my = statistics.mean(xs), statistics.mean(ys)
        den = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None
    return {"rows": rows, "range_slope_km_per_kg": slope, "propellant_pool_kg": propellant_pool}


# --------------------------------------------------------------------------------------
#   6. The flown motor
# --------------------------------------------------------------------------------------


def motor_operating_point() -> dict[str, Any]:
    """The motor as `loop.converge_point` builds it: sustain thrust matched to cruise drag."""
    from ..sizing.propulsion import SolidMotor

    point = load_point("point_ntop.json")
    dv = design_vector_from(point["design_vector"])
    drag_cruise = point["history"][0]["cruise_drag"]

    motor = SolidMotor(dv)
    motor.size_sustain_for_thrust(drag_cruise)
    grain = motor.grain_geometry()
    inert = motor.inert_mass_breakdown()
    return {
        "cruise_drag_N": drag_cruise,
        "operating_point": motor.operating_point(),
        "throat_transitions": motor.throat_transition_report(),
        "separation_check": motor.separation_check(),
        "inert": inert,
        "grain": {
            "length_total": grain.length_total,
            "L_over_D": grain.L_over_D,
            "bay_length_available": grain.bay_length_available,
            "volumetric_loading": grain.volumetric_loading,
            "feasible": grain.feasible,
            "warnings": list(grain.warnings),
        },
        "propellant_mass": motor.propellant_mass,
        "warnings": list(motor.warnings),
    }


# --------------------------------------------------------------------------------------
#   Assembly
# --------------------------------------------------------------------------------------


def collect() -> dict[str, Any]:
    t0 = time.perf_counter()
    payload: dict[str, Any] = {
        "integrator": integrator_residuals(),
        "aero": aero_validation(),
        "ntop": ntop_versus_closed_form(),
        "dive": unpowered_dive_sweep(),
        "terminal_sweep": terminal_propellant_sweep(),
        "motor": motor_operating_point(),
    }
    payload["wall_time_s"] = time.perf_counter() - t0
    return payload


def write(path: str | None = None) -> str:
    path = path or EVIDENCE_JSON
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(collect(), f, indent=2, default=str)
    return path


def load(path: str | None = None) -> dict[str, Any]:
    with open(path or EVIDENCE_JSON, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    print(write())
