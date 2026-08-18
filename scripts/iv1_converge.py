"""Converge IV-1 against the 100 mile requirement, with the real strake aerodynamics.

    .venv/Scripts/python.exe scripts/iv1_converge.py [--ntop]

Two nested fixed points:

- The attitude-control motor has to accelerate its own mass, so its thrust is solved for rather
  than chosen: `F_acs = lateral_g_min * g0 * mass_at_intercept`, iterated.
- The pitchover angle is searched, because range is not monotone in it. There is an interior
  optimum, so a bound search would find the wrong answer.

With `--ntop` the geometry is measured by nTop and the measured structure mass replaces the
analytic estimate, closing the SPEC.md section 6 loop for the two-stage vehicle.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rocketgen.config_iv1 import (  # noqa: E402
    InterceptRequirements,
    default_iv1,
    lateral_g,
    lateral_g_acs,
    lateral_g_total,
)
from rocketgen.sizing.aero_iv1 import StackAero  # noqa: E402
from rocketgen.sizing.masses_iv1 import build_stack_masses  # noqa: E402
from rocketgen.sizing.propulsion_iv1 import MultiStageMotor  # noqa: E402
from rocketgen.sizing.trajectory_iv1 import AscentMission  # noqa: E402

OUT = os.path.join(REPO, "runs", "IV-1")


def base_stack():
    """The stack that closes every motor, volume, nozzle-fit and structural constraint.

    Found by sweeping: a tubular grain cannot deliver more than about 110 kN in this envelope, and
    the stage-1 nozzle exit must fit inside the stage-1 body, which forced the booster to the full
    0.42 m allowed diameter and its area ratio down to 6.
    """
    dv = default_iv1()
    s1, s2 = dv.stages
    # Every number here is the result of a sweep, not a choice:
    #   D1 = 0.42  the full allowed diameter, forced by the stage-1 nozzle exit having to fit
    #              inside the stage-1 body
    #   F1 = 45 kN a tubular grain cannot deliver more than about 110 kN in this envelope, and
    #              lower thrust was needed again to bring q_max under the limit
    #   mp1 = 220  traded down from 390: the vehicle arrived with Mach to spare, and the excess
    #              energy was being paid for in peak dynamic pressure
    #   mp2 = 90   the largest stage-2 charge whose tubular grain actually closes; above about
    #              95 kg the web is wider than the bay radius
    s1.D, s1.L, s1.m_propellant, s1.F_thrust, s1.eps_nozzle = 0.42, 2.6, 220.0, 45.0e3, 6.0
    s2.D, s2.L, s2.m_propellant, s2.F_thrust, s2.eps_nozzle = 0.34, 2.4, 90.0, 18.0e3, 18.0
    dv.f_nose = 3.0
    dv.t_pitch = 5.0
    return dv


def evaluate(dv, reqs, geometry_fn=None, run_dir=None, dt=0.02, adaptive=False):
    """One design point: motor, masses, aero, trajectory, constraints."""
    motor = MultiStageMotor(dv, reqs)
    meas = None
    if geometry_fn is not None:
        try:
            meas = geometry_fn(dv, run_dir or os.path.join(OUT, "geom"))
        except Exception as exc:  # noqa: BLE001 - degrade loudly, never silently
            print(f"  nTop geometry FAILED ({type(exc).__name__}: {exc}); using analytic geometry")
            meas = None
    sm = build_stack_masses(dv, reqs, meas=meas, motor=motor)
    m0 = sm.m0 + dv.acs.total_mass

    aero = StackAero(dv, reqs, meas=meas)
    mission = AscentMission(dv, reqs, motor, aero, m0)
    traj = mission.fly(dt=dt, adaptive=adaptive, t_max=600.0)
    ic = mission.intercept

    S2 = aero.S_ref(2)
    cn_max = aero.CN_max(max(ic.mach, 0.3), ic.altitude, 2, reqs.alpha_max)
    g_aero = lateral_g(ic.q, S2, cn_max, ic.mass)
    g_acs = lateral_g_acs(dv.acs.thrust, ic.mass)
    g_tot = lateral_g_total(ic.q, S2, cn_max, ic.mass, dv.acs.thrust)

    g1, g2 = motor.grain_geometry(1), motor.grain_geometry(2)

    # A2 is checked against the mission's own `reached_slant_range` flag, not by re-comparing the
    # float. The terminating step is refined by bisection ONTO the slant-range target, so the
    # terminal value lands within about 1e-9 m of it and can fall either side by construction.
    # Comparing a bisection-refined value against the very target it was refined to, with a strict
    # inequality, measures the bisection tolerance rather than the vehicle: it was reporting A2 as
    # failed by 2e-10 m. The flag records whether the condition was actually met.
    slant_ok = bool(ic.reached_slant_range) or ic.slant_range >= reqs.slant_range_min

    cons = [
        ("A2 slant range [m]", ic.slant_range, reqs.slant_range_min, ">="),
        ("A3 intercept alt [m]", ic.altitude, reqs.h_intercept_min, ">="),
        ("A4 intercept Mach", ic.mach, reqs.mach_intercept_min, ">="),
        ("A6 max diameter [m]", dv.D_max, reqs.D_max, "<="),
        ("A7 stacked length [m]", dv.L_total, reqs.L_max, "<="),
        ("A8 launch mass [kg]", m0, reqs.m0_max, "<="),
        ("A10 q_max [Pa]", traj.q_max, reqs.q_max, "<="),
        ("A11 lateral g", g_tot, reqs.lateral_g_min, ">="),
        ("A12 s1 burnout alt [m]", _burnout_alt(traj, motor), reqs.h_stage1_burnout_max, "<="),
        ("grain L/D stage 1", g1.L_over_D, 8.0, "<="),
        ("grain L/D stage 2", g2.L_over_D, 8.0, "<="),
        # Grain closure, per stage. L/D alone is NOT enough: a grain can sit inside the L/D band
        # and still be impossible, because a tubular grain needs a bore for burning area and the
        # remaining web has to fit inside the bay radius. Gating only on L/D let a stage-2 grain
        # through at 134 percent volumetric loading with a 232 mm web in a 164 mm bay radius.
        ("stage 1 vol loading", g1.volumetric_loading, 1.0, "<="),
        ("stage 2 vol loading", g2.volumetric_loading, 1.0, "<="),
        ("stage 1 grain closes", 1.0 if g1.feasible else 0.0, 1.0, ">="),
        ("stage 2 grain closes", 1.0 if g2.feasible else 0.0, 1.0, ">="),
    ]
    met = [(n, v, lim, s, (v >= lim if s == ">=" else v <= lim)) for n, v, lim, s in cons]
    met[0] = (met[0][0], met[0][1], met[0][2], met[0][3], slant_ok)
    return {
        "dv": dv, "motor": motor, "masses": sm, "aero": aero, "traj": traj, "intercept": ic,
        "m0": m0, "cn_max": cn_max, "g_aero": g_aero, "g_acs": g_acs, "g_total": g_tot,
        "constraints": met, "feasible": all(m[4] for m in met),
        "meas": meas, "geometry_measured": meas is not None,
    }


def _burnout_alt(traj, motor) -> float:
    tb = motor.t_burnout(1)
    for t, h in zip(traj.time, traj.h):
        if t >= tb:
            return h
    return traj.h[-1] if traj.h else 0.0


def size_acs(dv, reqs, geometry_fn=None, run_dir=None, margin: float = 1.06, iters: int = 8):
    """Fixed point on attitude-control thrust: the pack must accelerate its own mass."""
    for i in range(iters):
        r = evaluate(dv, reqs, geometry_fn, run_dir, dt=0.05, adaptive=True)
        need = margin * reqs.lateral_g_min * 9.80665 * r["intercept"].mass
        if r["g_acs"] > 0 and abs(need - dv.acs.thrust) / dv.acs.thrust < 0.004:
            return dv, r
        dv = dv.with_path("acs.thrust", need)
    return dv, r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ntop", action="store_true", help="measure the geometry with nTop")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    reqs = InterceptRequirements()

    geometry_fn = None
    if args.ntop:
        try:
            from rocketgen.ntopgen.stack_notebook import measure_stack

            geometry_fn = measure_stack
            print("nTop geometry: enabled")
        except Exception as exc:  # noqa: BLE001
            print(f"nTop geometry: UNAVAILABLE ({type(exc).__name__}: {exc}); analytic only")
    else:
        print("nTop geometry: disabled, analytic geometry only")

    print("\nsearching the pitchover angle (range is not monotone in it)")
    best = None
    grid = [(g, 18.0e3) for g in (32.0, 34.0, 36.0, 38.0)]
    for gamma_deg, F2 in grid:
        dv = base_stack().replace(gamma_pitch=math.radians(gamma_deg))
        dv.stages[1].F_thrust = F2
        dv, r = size_acs(dv, reqs, geometry_fn, os.path.join(OUT, "geom"))
        ic = r["intercept"]
        tag = "FEASIBLE" if r["feasible"] else ",".join(
            n.replace(" [m]", "").replace(" [kg]", "").replace(" [Pa]", "")
            for n, *_rest, ok in r["constraints"] if not ok
        )
        print(
            f"  gamma {gamma_deg:4.1f} F2 {F2/1e3:4.0f}kN: slant {ic.slant_range/1e3:6.1f} km, "
            f"h {ic.altitude/1e3:5.1f} km, M {ic.mach:4.2f}, m0 {r['m0']:6.1f} kg -> {tag}"
        )
        if r["feasible"] and (best is None or r["m0"] < best[1]["m0"]):
            best = (dv, r, gamma_deg)

    if best is None:
        print("\nno feasible pitchover angle found")
        return 1

    dv, _, gamma_deg = best
    print(f"\nre-running the best point (gamma {gamma_deg:.1f} deg) at the fine fixed step")
    r = evaluate(dv, reqs, geometry_fn, os.path.join(OUT, "geom"), dt=0.02, adaptive=False)
    ic, sm = r["intercept"], r["masses"]

    print(f"\n{'='*74}\nIV-1 CONVERGED"
          f"{'  (nTop-measured geometry)' if r['geometry_measured'] else '  (analytic geometry)'}")
    print(f"{'='*74}")
    print(f"  launch mass {r['m0']:.1f} kg   stacked length {dv.L_total:.2f} m   "
          f"D1 {dv.booster.D:.2f} m  D2 {dv.payload_stage.D:.2f} m")
    print(f"  propellant {dv.booster.m_propellant:.0f} + {dv.payload_stage.m_propellant:.0f} kg   "
          f"jettisoned at separation {r['motor'].jettisoned_mass():.1f} kg")
    print(f"  ACS {dv.acs.thrust/1e3:.1f} kN for {dv.acs.burn_time:.1f} s "
          f"({dv.acs.total_impulse/1e3:.0f} kN.s, {dv.acs.total_mass:.1f} kg)")
    print(f"  intercept: slant {ic.slant_range/1e3:.1f} km ({ic.slant_range_miles:.1f} mi), "
          f"h {ic.altitude/1e3:.1f} km, Mach {ic.mach:.2f}, t {ic.time:.1f} s")
    print(f"  lateral g at intercept: aerodynamic {r['g_aero']:.2f}, "
          f"attitude-control {r['g_acs']:.2f}, A11 figure {r['g_total']:.2f} "
          f"(CN_max {r['cn_max']:.2f})")
    print(f"\n  {'requirement':<24}{'value':>13}{'limit':>13}  status")
    for n, v, lim, s, ok in r["constraints"]:
        print(f"  {n:<24}{v:>13.4g}{lim:>13.4g}  {'OK' if ok else 'FAIL'}")
    print(f"\n  {'FEASIBLE' if r['feasible'] else 'INFEASIBLE'}")

    print(f"\n  mass statement, {sm.measured_fraction*100:.1f} percent nTop-measured")
    for idx, name, mass, x, prov in sm.table_rows():
        print(f"    s{idx}  {name:<34}{mass:>9.2f} kg  x {x:>6.3f} m  {prov}")
    print(f"    {'ACS pack':<38}{dv.acs.total_mass:>9.2f} kg")
    print(f"    {'TOTAL':<38}{r['m0']:>9.2f} kg")

    if sm.warnings or r["motor"].warnings:
        print("\n  warnings")
        for w in list(dict.fromkeys(list(r["motor"].warnings) + list(sm.warnings))):
            print(f"    ! {w}")

    payload = {
        "design_vector": dv.as_dict(),
        "requirements": asdict(reqs),
        "geometry_measured": r["geometry_measured"],
        "launch_mass_kg": r["m0"],
        "pitchover_deg": gamma_deg,
        "intercept": asdict(ic),
        "lateral_g": {"aerodynamic": r["g_aero"], "acs": r["g_acs"], "total": r["g_total"],
                      "cn_max": r["cn_max"]},
        "constraints": [
            {"name": n, "value": v, "limit": lim, "sense": s, "met": ok}
            for n, v, lim, s, ok in r["constraints"]
        ],
        "feasible": r["feasible"],
        "mass_statement": [
            {"stage": i, "item": n, "mass_kg": m, "station_m": x, "provenance": p}
            for i, n, m, x, p in sm.table_rows()
        ],
        "jettisoned_kg": r["motor"].jettisoned_mass(),
        "acs": asdict(dv.acs),
        "warnings": list(dict.fromkeys(list(r["motor"].warnings) + list(sm.warnings))),
    }
    path = os.path.join(OUT, "converged.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nwrote {os.path.relpath(path, REPO)}")

    try:
        from rocketgen.report.fig_trajectory import plot_trajectory

        fig = os.path.join(OUT, "trajectory.png")
        plot_trajectory(r["traj"], fig, title="IV-1 two-stage ascent to intercept",
                        q_limit=reqs.q_max, mach_cruise=None,
                        mach_terminal_min=reqs.mach_intercept_min)
        print(f"wrote {os.path.relpath(fig, REPO)}")
    except Exception as exc:  # noqa: BLE001
        print(f"trajectory figure skipped: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
