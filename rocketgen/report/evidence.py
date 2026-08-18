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

#: Which SV-1 study is being described. `select_study("spline")` re-points every path at
#: `runs/SV-1_spline`. Same pattern as `figstyle.select_study` and `build_example.select_study`.
OML = "ogive"
CASE_DIR = os.path.join(RUNS_DIR, "SV-1")
CONVERGED_DIR = os.path.join(CASE_DIR, "converged")
FIG_DIR = os.path.join(CASE_DIR, "figures")
EVIDENCE_JSON = os.path.join(FIG_DIR, "evidence.json")


def select_study(oml: str) -> None:
    """Point the evidence collector at the ogive study or the spline study."""
    global OML, CASE_DIR, CONVERGED_DIR, FIG_DIR, EVIDENCE_JSON
    if oml not in ("ogive", "spline"):
        raise ValueError(f"unknown oml family {oml!r}")
    OML = oml
    CASE_DIR = os.path.join(RUNS_DIR, "SV-1_spline" if oml == "spline" else "SV-1")
    CONVERGED_DIR = os.path.join(CASE_DIR, "converged")
    FIG_DIR = os.path.join(CASE_DIR, "figures")
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


# --------------------------------------------------------------------------------------
#   7. The splined outer mould line
#
#   Only collected when the study under description is the spline study. None of it is
#   meaningful for the tangent-ogive study, where every shape ratio is 1.0 by construction.
# --------------------------------------------------------------------------------------


#: `nose_blend` values for the shape trade. 0.0 is the ogive-equivalent spline, 1.0 is the
#: slender-body drag optimum, and 0.7 is where the sizing search stopped.
NOSE_BLEND_SWEEP: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0)

#: Mach numbers at which the drag build-up is decomposed. 1.2 is the top of the transonic
#: blend, so it is the lowest Mach at which the wave-drag term is physics.
SHAPE_MACH: tuple[float, ...] = (1.2, 2.0, 3.0, 4.0)


def wave_drag_validation() -> dict[str, Any]:
    """Residuals of `sizing/wavedrag.py` against closed forms that are outside this repo.

    Every row is MEASURED here, not transcribed from the test file. The tests assert bounds;
    the report quotes what the code actually achieves.
    """
    from ..oml_spline import (
        SplineProfile,
        ogive_control_values,
        tangent_ogive_radius,
        von_karman_radius,
    )
    from ..sizing import wavedrag as W

    # 4001 stations, the same grid `tests/test_wavedrag.py::_area_table` uses, so the residual
    # quoted in the report is the residual the test asserts against and not a different number.
    def area_table(radius_of_t, length: float, radius: float, n: int = 4001):
        xs = [length * i / (n - 1) for i in range(n)]
        areas = [math.pi * (radius * radius_of_t(x / length)) ** 2 for x in xs]
        return xs, areas

    # (a) Sears-Haack: D/q = 128 V^2 / (pi L^4), an exact published result.
    L, R = 3.0, 0.5
    sears_haack_radius = lambda u: (4.0 * u * (1.0 - u)) ** 0.75      # noqa: E731
    xs, areas = area_table(sears_haack_radius, L, R)
    volume = 3.0 * math.pi ** 2 * R * R * L / 16.0
    sh_series = W.wave_drag_over_q(xs, areas, L)
    sh_closed = W.sears_haack_drag_over_q(volume, L)

    # The Sears-Haack profile has an infinite slope at both ends, so the central-difference
    # S'(x) in `glauert_coefficients` converges slowly on it. The refinement study says the
    # residual belongs to the CHECK TABLE, not to the model, which a single number cannot.
    sh_refine = []
    for n in (501, 1001, 2001, 4001, 8001):
        x_n, a_n = area_table(sears_haack_radius, L, R, n=n)
        sh_refine.append({
            "n_stations": n,
            "rel_err": W.wave_drag_over_q(x_n, a_n, L) / sh_closed - 1.0,
        })

    # (b) von Karman ogive: C_D on base area = (d/L)^2.
    L2, R2 = 2.1, 0.175
    xs, areas = area_table(von_karman_radius, L2, R2)
    vk_series = W.wave_drag_over_q(xs, areas, L2)
    vk_closed = W.von_karman_ogive_drag_over_q(2.0 * R2, L2)
    vk_cd_on_base = vk_series / (math.pi * R2 * R2)

    # (c) the fineness-free statement of the same result: shape factor of the optimum = 4/pi.
    sf_vk = W.shape_factor(von_karman_radius)

    # (d) the Glauert series constant pi/4, against direct double integration. The deleted
    #     diagonal converges slowly, so what is measured is CONVERGENCE ONTO the series.
    coeffs = [0.4, 0.3, -0.15, 0.05]
    series = (math.pi / 4.0) * sum((n + 1) * a * a for n, a in enumerate(coeffs))

    def drag_direct(n: int) -> float:
        th = [(i + 0.5) * math.pi / n for i in range(n)]
        dth = math.pi / n
        g = [sum((m + 1) * a * math.cos((m + 1) * t) for m, a in enumerate(coeffs)) for t in th]
        c = [math.cos(t) for t in th]
        total = 0.0
        for i in range(n):
            gi, ci = g[i], c[i]
            for j in range(n):
                if i == j:
                    continue
                total += gi * g[j] * math.log(abs(ci - c[j]) * 3.0 / 2.0)
        return -total * dth * dth / (2.0 * math.pi)

    direct = [{"n": n, "rel_err": abs(drag_direct(n) / series - 1.0)} for n in (250, 500, 1000)]

    # (e) the optimum really is an optimum: adding any higher Glauert mode at fixed base area
    #     can only raise the drag, because base area is set by A_1 alone.
    def drag_of(a: list[float]) -> float:
        return (math.pi / 4.0) * sum((n + 1) * v * v for n, v in enumerate(a))

    optimality = [
        {
            "modes": str(extra),
            "drag": drag_of(extra),
            "raises": drag_of(extra) > drag_of([0.4]),
        }
        for extra in ([0.4, 0.02], [0.4, 0.0, 0.02], [0.4, -0.05], [0.4, 0.01, 0.01, 0.01])
    ]

    # (f) what the tangent ogive costs, and how much of it a 9-point spline gets back.
    n_ctrl = 9
    sf_ogive = W.shape_factor_of_control(ogive_control_values(6.0, n_ctrl))
    sf_opt = W.shape_factor_of_control(W.optimal_control_values(n_ctrl))
    penalty = [
        {
            "f_nose": f,
            "sf_over_bound": W.shape_factor(
                lambda t, f=f: tangent_ogive_radius(t, 2.0 * f)
            ) / sf_vk,
        }
        for f in (2.0, 3.0, 4.0, 5.0)
    ]

    # (g) the optimal shape is still a usable nose, not merely a low number.
    p_opt = SplineProfile(length=1.19, radius=0.175, control=W.optimal_control_values(n_ctrl))

    return {
        "sears_haack": {"series": sh_series, "closed_form": sh_closed,
                        "rel_err": sh_series / sh_closed - 1.0, "L": L, "R": R,
                        "n_stations": 4001, "refinement": sh_refine},
        "von_karman": {"series": vk_series, "closed_form": vk_closed,
                       "rel_err": vk_series / vk_closed - 1.0,
                       "cd_on_base": vk_cd_on_base,
                       "cd_on_base_closed": (2.0 * R2 / L2) ** 2,
                       "cd_rel_err": vk_cd_on_base / ((2.0 * R2 / L2) ** 2) - 1.0,
                       "L": L2, "R": R2},
        "von_karman_shape_factor": {"measured": sf_vk, "closed_form": 4.0 / math.pi,
                                    "rel_err": sf_vk / (4.0 / math.pi) - 1.0},
        "glauert_direct": direct,
        "optimality": optimality,
        "n_ctrl": n_ctrl,
        "shape_factor_ogive": sf_ogive,
        "shape_factor_optimal_spline": sf_opt,
        "shape_factor_bound": sf_vk,
        "ogive_penalty_over_bound": sf_ogive / sf_vk,
        "spline_over_bound": sf_opt / sf_vk,
        "gap_recovered_fraction": (sf_ogive - sf_opt) / (sf_ogive - sf_vk),
        "ogive_penalty_by_fineness": penalty,
        "optimal_is_monotone": p_opt.is_monotone(),
        "optimal_max_slope": p_opt.max_slope(),
    }


def nose_shape_trade() -> dict[str, Any]:
    """Sweep `nose_blend` at the converged design and record what each blend costs and buys.

    This is the evidence behind the headline result that the sizing search STOPPED SHORT of
    the drag optimum. Each row carries the drag saving, the forebody volume given up, the
    centre-of-pressure shift, and the flown result of the whole loop.

    The loop rows run with analytic geometry, because 8 nTop measurements would cost about
    4 minutes and the question here is the trend, not the absolute level. The converged row
    of the real, nTop-coupled result is reported beside it so the offset is visible.
    """
    from ..oml_spline import SplineProfile
    from ..sizing.aero import RocketAero
    from ..sizing.loop import CalibratedAero, converge_point, penalty
    from ..sizing.wavedrag import nose_wave_shape_ratio

    point = load_point("point_ntop.json")
    base = design_vector_from(point["design_vector"])
    if getattr(base, "nose_control", None) is None:
        return {"applicable": False,
                "reason": "the converged design has no splined nose, so there is no trade"}
    reqs = Requirements()
    R = 0.5 * base.D
    k = base.L_nose / R

    rows: list[dict[str, Any]] = []
    for blend in NOSE_BLEND_SWEEP:
        dv = base.replace(nose_blend=float(blend))
        control = dv.nose_control
        profile = SplineProfile(length=dv.L_nose, radius=R, control=control)
        aero = CalibratedAero(RocketAero(dv, nose_shape=dv.nose_shape),
                              factor=CD0_CALIBRATION)
        res = converge_point(dv, reqs, geometry_fn=None, max_iter=2, dt=0.06, adaptive=True)
        row: dict[str, Any] = {
            "nose_blend": float(blend),
            "shape_ratio": nose_wave_shape_ratio(control, k),
            "nose_volume_m3": profile.volume(),
            "nose_wetted_m2": profile.lateral_area(),
            "max_slope": profile.max_slope(),
            "monotone": profile.is_monotone(),
            "m0_kg": res.masses.total if res.masses is not None else None,
            "range_km": res.traj.range_final / 1000.0 if res.traj is not None else None,
            "mach_terminal": res.traj.mach_final if res.traj is not None else None,
            "q_max_kPa": res.traj.q_max / 1000.0 if res.traj is not None else None,
            "feasible": res.feasible,
            # The search minimises THIS, not the launch mass. Recording it is what lets the
            # report say whether the shape trade has an interior optimum or not.
            "penalty": penalty(res, reqs),
            "violations": [c.name for c in res.constraints if not c.met],
        }
        for mach in SHAPE_MACH:
            r = aero.evaluate(mach, 12_000.0, math.radians(2.0))
            row[f"CD0_M{mach:g}"] = r.CD0
            row[f"xcp_M{mach:g}"] = r.x_cp
        rows.append(row)

    ref = next(r for r in rows if r["nose_blend"] == 0.0)
    for r in rows:
        r["d_nose_volume_pct"] = 100.0 * (r["nose_volume_m3"] / ref["nose_volume_m3"] - 1.0)
        r["d_nose_wetted_pct"] = 100.0 * (r["nose_wetted_m2"] / ref["nose_wetted_m2"] - 1.0)
        for mach in SHAPE_MACH:
            r[f"d_CD0_pct_M{mach:g}"] = 100.0 * (
                r[f"CD0_M{mach:g}"] / ref[f"CD0_M{mach:g}"] - 1.0
            )
            r[f"d_xcp_mm_M{mach:g}"] = 1000.0 * (r[f"xcp_M{mach:g}"] - ref[f"xcp_M{mach:g}"])

    # The nose wave-drag share of CD0, which is what the shape ratio multiplies.
    ogive = RocketAero(base.replace(nose_shape="tangent_ogive", nose_blend=0.0),
                       nose_shape="tangent_ogive")
    spline = RocketAero(base, nose_shape=base.nose_shape)
    share = []
    for mach in SHAPE_MACH:
        ro = ogive.evaluate(mach, 12_000.0, math.radians(2.0))
        rs = spline.evaluate(mach, 12_000.0, math.radians(2.0))
        bo, bs = ro.breakdown, rs.breakdown
        cd0_o, cd0_s = ro.CD0, rs.CD0
        share.append({
            "mach": mach,
            "cd_wave_body_ogive": bo["CD_wave_body"],
            "cd_wave_body_spline": bs["CD_wave_body"],
            "wave_share_of_cd0_ogive": bo["CD_wave_body"] / cd0_o,
            "cd0_ogive": cd0_o,
            "cd0_spline": cd0_s,
            "d_cd0_pct": 100.0 * (cd0_s / cd0_o - 1.0),
            "d_wave_pct": 100.0 * (bs["CD_wave_body"] / bo["CD_wave_body"] - 1.0),
        })

    return {
        "applicable": True,
        "k_L_over_R": k,
        "converged_nose_blend": float(base.nose_blend),
        "converged_boattail_blend": float(base.boattail_blend),
        "rows": rows,
        "wave_share": share,
        # The real, nTop-coupled converged numbers, for the offset against the analytic rows.
        "coupled": {
            "m0_kg": point["mass_statement"]["total_kg"],
            "range_km": point["trajectory"]["range_m"] / 1000.0,
            "mach_terminal": point["trajectory"]["mach_final"],
            "q_max_kPa": point["trajectory"]["q_max_Pa"] / 1000.0,
        },
    }


#: Blends for the nTop-COUPLED sweep. Fewer points than the analytic sweep, because each one
#: costs two `measure_rocket` calls of about 25 to 35 s. The set brackets the converged 0.7 and
#: reaches the drag optimum at 1.0, which is the only comparison the interior-optimum claim needs.
NOSE_BLEND_COUPLED: tuple[float, ...] = (0.0, 0.5, 0.7, 0.85, 1.0)


def nose_shape_trade_coupled() -> dict[str, Any]:
    """The same blend sweep, with nTop measuring every shape.

    This exists because the analytic sweep CANNOT answer the question the result turns on.
    The analytic mass model reads a closed form of the outer mould line, so a blend that gives
    up forebody volume changes the airframe mass only through that closed form. The measured
    model rebuilds and re-measures the solid. If the shape trade has an interior optimum, this
    sweep is where it appears.

    Returns `available: False`, with the reason, when nTop cannot be reached. The report then
    says the sweep is missing rather than quoting the analytic sweep as if it were this one.
    """
    from ..sizing.loop import converge_point, penalty

    point = load_point("point_ntop.json")
    base = design_vector_from(point["design_vector"])
    if getattr(base, "nose_control", None) is None:
        return {"available": False, "reason": "the converged design has no splined nose"}

    try:
        from ..ntopgen.rocket_notebook import measure_rocket
    except Exception as exc:                                   # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    run_dir = os.path.join(CASE_DIR, "shape_trade")
    reqs = Requirements()
    rows: list[dict[str, Any]] = []
    for blend in NOSE_BLEND_COUPLED:
        dv = base.replace(nose_blend=float(blend))
        t0 = time.perf_counter()
        try:
            res = converge_point(dv, reqs, geometry_fn=measure_rocket, run_dir=run_dir,
                                 max_iter=2, dt=0.05, adaptive=True)
        except Exception as exc:                               # noqa: BLE001
            # Never swallow: a blend that fails becomes a visible row, not a missing one.
            rows.append({"nose_blend": float(blend), "failed": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append({
            "nose_blend": float(blend),
            "geometry_measured": res.geometry_measured,
            "m0_kg": res.masses.total if res.masses is not None else None,
            "x_cg_m": res.masses.x_cg if res.masses is not None else None,
            "volume_total": res.meas.volume_total if res.meas is not None else None,
            "volume_cavity": res.meas.volume_cavity if res.meas is not None else None,
            "mass_structure": res.meas.mass_structure if res.meas is not None else None,
            "area_wetted_body": res.meas.area_wetted_body if res.meas is not None else None,
            "range_km": res.traj.range_final / 1000.0 if res.traj is not None else None,
            "mach_terminal": res.traj.mach_final if res.traj is not None else None,
            "q_max_kPa": res.traj.q_max / 1000.0 if res.traj is not None else None,
            "static_margin": next(
                (c.value for c in res.constraints if c.name == "R10 static margin"), None
            ),
            "feasible": res.feasible,
            "penalty": penalty(res, reqs),
            "violations": [c.name for c in res.constraints if not c.met],
            "wall_time_s": time.perf_counter() - t0,
        })

    good = [r for r in rows if r.get("penalty") is not None]
    best = min(good, key=lambda r: r["penalty"]) if good else None
    return {
        "available": True,
        "rows": rows,
        "n_failed": sum(1 for r in rows if "failed" in r),
        "best_blend_by_penalty": best["nose_blend"] if best else None,
        "interior_optimum": (
            best is not None and 0.0 < best["nose_blend"] < 1.0
        ),
    }


def search_record() -> dict[str, Any]:
    """What the sizing search actually did, read back from its own trace.

    The trace is the only record of the search, and it does NOT carry every searched variable.
    That is recorded here rather than assumed, because the shape blends are searched last and
    a reader who does not know they are missing from the trace will draw the wrong conclusion.
    """
    import csv as _csv

    path = os.path.join(CASE_DIR, "size", "search_trace.csv")
    if not os.path.isfile(path):
        return {"available": False, "reason": f"no search trace at {path}"}
    with open(path, encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    columns = list(rows[0].keys()) if rows else []
    dv_fields = set(DesignVector.__dataclass_fields__)
    searched_but_untraced = [
        n for n in ("nose_blend", "boattail_blend")
        if n in dv_fields and n not in columns
    ]
    best = None
    for r in rows:
        if r.get("feasible") == "1.0" or r.get("feasible") == "1":
            v = float(r["m0"])
            if best is None or v < best[0]:
                best = (v, r)
    return {
        "available": True,
        "path": f"runs/{os.path.basename(CASE_DIR)}/size/search_trace.csv",
        "n_evaluations": len(rows),
        "columns": columns,
        "variables_searched_but_not_traced": searched_but_untraced,
        "n_feasible_in_trace": sum(
            1 for r in rows if r.get("feasible") in ("1", "1.0")
        ),
        "best_traced_m0_kg": best[0] if best else None,
    }


def spline_geometry_check() -> dict[str, Any]:
    """The measured nTop outer-mould-line volume against the exact integral of the B-spline.

    nTop revolves the spline itself, so `SplineProfile.volume` is an EXACT description of the
    solid rather than an approximation of it. That makes this a real check of the notebook,
    not a check of a discretisation.
    """
    from ..oml_spline import SplineProfile

    point = load_point("point_ntop.json")
    dv = design_vector_from(point["design_vector"])
    measured = point["ntop_measurements"]
    R = 0.5 * dv.D
    r_base = 0.5 * dv.d_base

    nose_control = getattr(dv, "nose_control", None)
    boattail_control = getattr(dv, "boattail_control", None)
    if nose_control is None:
        return {"applicable": False, "reason": "no splined nose in this design"}

    nose = SplineProfile(length=dv.L_nose, radius=R, control=nose_control)
    v_nose = nose.volume()
    a_nose = nose.lateral_area()
    v_cyl = math.pi * R * R * dv.L_body_cyl
    a_cyl = 2.0 * math.pi * R * dv.L_body_cyl
    if boattail_control is not None:
        # Run expressed on the contraction: control points are R + (r_base - R) * c_i, so the
        # SplineProfile end radius is r_base and it starts at R.
        boat = SplineProfile(length=dv.L_boattail, radius=r_base,
                             control=boattail_control, r0_over_r=R / r_base)
        v_boat, a_boat = boat.volume(), boat.lateral_area()
        boat_form = "splined contraction, exact integral"
    else:
        slant = math.hypot(dv.L_boattail, R - r_base)
        a_boat = math.pi * (R + r_base) * slant
        v_boat = math.pi * dv.L_boattail * (R * R + R * r_base + r_base * r_base) / 3.0
        boat_form = "straight cone, closed form"

    v_closed = v_nose + v_cyl + v_boat
    a_closed = a_nose + a_cyl + a_boat
    return {
        "applicable": True,
        "volume_ntop": measured["volume_total"],
        "volume_closed_form": v_closed,
        "volume_rel_err": measured["volume_total"] / v_closed - 1.0,
        "area_ntop": measured["area_wetted_body"],
        "area_closed_form": a_closed,
        "area_rel_err": measured["area_wetted_body"] / a_closed - 1.0,
        "parts": {"nose": v_nose, "cylinder": v_cyl, "boattail": v_boat},
        "boattail_form": boat_form,
        "wall_time_s": measured.get("wall_time_s"),
        "note": (
            "The closed form is the exact integral of the same B-spline nTop revolves, so a "
            "discretisation error is not available to hide behind."
        ),
    }


def collect() -> dict[str, Any]:
    t0 = time.perf_counter()
    payload: dict[str, Any] = {
        "oml_family": OML,
        "case_dir": os.path.basename(CASE_DIR),
        "integrator": integrator_residuals(),
        "aero": aero_validation(),
        "ntop": ntop_versus_closed_form(),
        "dive": unpowered_dive_sweep(),
        "terminal_sweep": terminal_propellant_sweep(),
        "motor": motor_operating_point(),
    }
    if OML == "spline":
        payload["wavedrag"] = wave_drag_validation()
        payload["shape_trade"] = nose_shape_trade()
        payload["shape_trade_coupled"] = nose_shape_trade_coupled()
        payload["spline_geometry"] = spline_geometry_check()
        payload["search"] = search_record()
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
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oml", default="ogive", choices=["ogive", "spline"],
                    help="which study to describe; spline reads runs/SV-1_spline")
    _args = ap.parse_args()
    select_study(_args.oml)
    print(write())
