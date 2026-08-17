"""The coupled nTop + SUAVE sizing loop. SPEC.md section 6.

Two nested levels:

- `converge_point(dv)` runs the geometry-mass-aero-trajectory fixed point for ONE design vector.
  This is where nTop enters: the notebook builds the solid, measures it, and the measured wetted
  area, volumes and structural mass replace the analytic estimates on the next pass.
- `size(...)` moves the design vector to satisfy the requirements at minimum launch mass.

The loop is honest about two things:

1. `CD0_CALIBRATION` is applied here, at the boundary, not inside the aero model. The aero model
   reports what its physics gives; the loop corrects the known systematic bias measured against
   the Basic Finner free-flight data. See `config.SOURCES["cd0_calibration"]`.
2. When nTop is unavailable or a geometry fails to build, the loop does not silently fall back to
   analytics and carry on as if nothing happened. It records the failure in the result and marks
   the point as unmeasured.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable

from ..config import (
    CD0_CALIBRATION,
    AeroCoefficients,
    DesignVector,
    NtopMeasurements,
    Requirements,
    TrajectoryResult,
)
from .aero import RocketAero
from .masses import PROPELLANT_ITEMS, MassBuildup, build_masses, static_margin

# --------------------------------------------------------------------------------------
#   CD0 calibration wrapper
# --------------------------------------------------------------------------------------


class CalibratedAero:
    """Wraps `RocketAero` and applies the Basic Finner CD0 calibration.

    The wrapper scales CD0 and the zero-lift part of CD. It leaves CN, CN_alpha, CM and x_cp
    untouched, because those were validated to much tighter tolerance (mean bias +2.0 percent on
    x_cp, -10.7 percent on CN_alpha) and no calibration is justified.

    Set `factor=1.0` to see the raw physics.
    """

    def __init__(self, inner: RocketAero, factor: float = CD0_CALIBRATION) -> None:
        self.inner = inner
        self.factor = factor

    def evaluate(
        self, mach: float, altitude: float, alpha: float, power_on: bool = False
    ) -> AeroCoefficients:
        c = self.inner.evaluate(mach, altitude, alpha, power_on=power_on)
        if self.factor == 1.0:
            return c
        cd0_new = c.CD0 * self.factor
        cd_induced = c.CD - c.CD0
        breakdown = {k: v * self.factor for k, v in c.breakdown.items()}
        breakdown["CD0_calibration_factor"] = self.factor
        return AeroCoefficients(
            mach=c.mach,
            altitude=c.altitude,
            alpha=c.alpha,
            CD0=cd0_new,
            CD=cd0_new + cd_induced,
            CN=c.CN,
            CN_alpha=c.CN_alpha,
            CM=c.CM,
            x_cp=c.x_cp,
            L_over_D=(c.CN / (cd0_new + cd_induced)) if (cd0_new + cd_induced) > 0 else 0.0,
            breakdown=breakdown,
        )

    def trim_alpha(
        self, mach: float, altitude: float, required_CN: float, power_on: bool = False
    ) -> float:
        # CN is uncalibrated, so trim is unaffected.
        return self.inner.trim_alpha(mach, altitude, required_CN, power_on=power_on)


# --------------------------------------------------------------------------------------
#   Results
# --------------------------------------------------------------------------------------


@dataclass
class ConstraintReport:
    """One requirement, its value, its limit, and whether it is met."""

    name: str
    value: float
    limit: float
    sense: str          # ">=" or "<="
    units: str
    met: bool
    margin: float       # positive means satisfied, normalised by the limit

    @staticmethod
    def check(name: str, value: float, limit: float, sense: str, units: str) -> "ConstraintReport":
        if sense == ">=":
            met = value >= limit
            margin = (value - limit) / abs(limit) if limit else (value - limit)
        elif sense == "<=":
            met = value <= limit
            margin = (limit - value) / abs(limit) if limit else (limit - value)
        else:
            raise ValueError(f"bad sense {sense!r}")
        return ConstraintReport(name, value, limit, sense, units, met, margin)


@dataclass
class PointResult:
    """Everything known about one converged (or failed) design point."""

    dv: DesignVector
    masses: MassBuildup | None = None
    meas: NtopMeasurements | None = None
    traj: TrajectoryResult | None = None
    constraints: list[ConstraintReport] = field(default_factory=list)

    iterations: int = 0
    converged: bool = False
    geometry_measured: bool = False
    history: list[dict[str, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    message: str = ""
    wall_time_s: float = 0.0

    @property
    def m0(self) -> float:
        return self.masses.total if self.masses else float("nan")

    @property
    def range_km(self) -> float:
        return self.traj.range_final / 1000.0 if self.traj else float("nan")

    @property
    def feasible(self) -> bool:
        return self.converged and bool(self.constraints) and all(c.met for c in self.constraints)

    def failed_constraints(self) -> list[str]:
        return [c.name for c in self.constraints if not c.met]

    def summary(self) -> str:
        if not self.converged:
            return f"NOT CONVERGED after {self.iterations} iterations: {self.message}"
        tag = "FEASIBLE" if self.feasible else "INFEASIBLE (" + ", ".join(self.failed_constraints()) + ")"
        return (
            f"m0 = {self.m0:.1f} kg, range = {self.range_km:.1f} km, "
            f"Mach_terminal = {self.traj.mach_final:.2f}, "
            f"q_max = {self.traj.q_max/1000.0:.1f} kPa, "
            f"{self.iterations} iterations, {tag}"
        )


# --------------------------------------------------------------------------------------
#   Static margin over the flight
# --------------------------------------------------------------------------------------


def static_margin_history(
    dv: DesignVector,
    mb: MassBuildup,
    aero: CalibratedAero,
    traj: TrajectoryResult,
) -> tuple[float, float]:
    """Minimum static margin over the trajectory, and the Mach at which it occurs.

    The CG moves forward as propellant burns, and x_cp moves forward with Mach, so the
    worst case is not at either end of the flight and has to be swept.
    """
    m0 = mb.total
    m_burnout, x_burnout = mb.excluding(*PROPELLANT_ITEMS)
    x0 = mb.x_cg
    m_prop = m0 - m_burnout

    worst = float("inf")
    worst_mach = 0.0
    for mach, alt, mass, alpha in zip(traj.mach, traj.h, traj.mass, traj.alpha):
        if mach < 0.3:
            continue
        # linear CG interpolation on burnt fraction
        burnt = (m0 - mass) / m_prop if m_prop > 0.0 else 0.0
        burnt = min(max(burnt, 0.0), 1.0)
        x_cg = x0 + burnt * (x_burnout - x0)
        c = aero.evaluate(mach, alt, alpha)
        sm = static_margin(c.x_cp, x_cg, dv.D)
        if sm < worst:
            worst, worst_mach = sm, mach
    if worst == float("inf"):
        return float("nan"), 0.0
    return worst, worst_mach


# --------------------------------------------------------------------------------------
#   The fixed point for one design vector
# --------------------------------------------------------------------------------------

GeometryFn = Callable[[DesignVector, str], NtopMeasurements]
"""Signature of the nTop geometry call: (design_vector, run_directory) -> measurements."""


def converge_point(
    dv: DesignVector,
    reqs: Requirements,
    geometry_fn: GeometryFn | None = None,
    run_dir: str | None = None,
    max_iter: int = 8,
    tol: float = 0.002,
    cd0_factor: float = CD0_CALIBRATION,
    dt: float = 0.02,
    adaptive: bool = True,
    tolerance: float = 1.0e-7,
) -> PointResult:
    """Run the SPEC.md section 6 fixed point for one design vector.

    `geometry_fn` is the nTop call. When it is None the loop runs analytics only and says so.
    """
    from .propulsion import SolidMotor           # imported here so a missing WP3 is a clear error
    from .trajectory import Mission

    t_start = time.perf_counter()
    res = PointResult(dv=dv)

    ok, errs = dv.geometry_is_valid()
    if not ok:
        res.message = "invalid geometry: " + "; ".join(errs)
        res.wall_time_s = time.perf_counter() - t_start
        return res

    meas: NtopMeasurements | None = None
    m0_prev = float("nan")
    range_prev = float("nan")

    for it in range(1, max_iter + 1):
        res.iterations = it

        # --- [1] mass build-up, using whatever nTop has measured so far ---
        # The motor is built first so its closed grain and nozzle geometry is the authority for
        # the motor inert masses, rather than the independent estimate in masses.py.
        motor = SolidMotor(dv)
        try:
            mb = build_masses(dv, reqs, meas=meas, motor=motor)
        except ValueError as exc:
            res.message = f"mass build-up failed: {exc}"
            break
        for w in getattr(motor, "warnings", []) or []:
            if w not in res.warnings:
                res.warnings.append(f"motor: {w}")
        grain = motor.grain_geometry()
        if not getattr(grain, "feasible", True):
            for w in getattr(grain, "warnings", []) or []:
                if w not in res.warnings:
                    res.warnings.append(f"grain: {w}")

        # --- [2] nTop geometry, on the first pass and whenever masses moved materially ---
        if geometry_fn is not None and meas is None:
            try:
                meas = geometry_fn(dv, run_dir or os.path.join("runs", "_point"))
                res.meas = meas
                res.geometry_measured = meas.is_usable()
                if not res.geometry_measured:
                    res.warnings.append(
                        "nTop ran but did not report every needed measurement; "
                        "analytic fallbacks are in use for the missing ones"
                    )
                res.warnings.extend(meas.warnings)
                # rebuild masses now that geometry exists
                mb = build_masses(dv, reqs, meas=meas)
            except Exception as exc:                      # noqa: BLE001 - report, do not hide
                res.warnings.append(f"nTop geometry failed: {type(exc).__name__}: {exc}")
                res.geometry_measured = False
            else:
                mb = build_masses(dv, reqs, meas=meas, motor=motor)
        elif geometry_fn is None and it == 1:
            res.warnings.append("no nTop geometry supplied; analytic geometry only")

        res.masses = mb
        res.warnings.extend(w for w in mb.warnings if w not in res.warnings)

        # --- [3] aero, using nTop's measured areas when available ---
        motor = SolidMotor(dv)
        area_exit = getattr(motor, "area_nozzle_exit", None)
        inner = RocketAero(
            dv,
            meas=meas,
            nose_shape=dv.nose_shape,
            area_nozzle_exit=area_exit,
        )
        aero = CalibratedAero(inner, factor=cd0_factor)

        # --- [3b] size the sustain thrust to the cruise drag at the cruise point ---
        c_cruise = aero.evaluate(reqs.M_cruise, reqs.h_cruise, math.radians(2.0), power_on=True)
        from .atmosphere import atmo

        st = atmo(reqs.h_cruise)
        V_cruise = reqs.M_cruise * st.speed_of_sound
        q_cruise = 0.5 * st.density * V_cruise**2
        drag_cruise = q_cruise * dv.S_ref * c_cruise.CD
        if hasattr(motor, "size_sustain_for_thrust"):
            motor.size_sustain_for_thrust(drag_cruise)

        # --- [4] trajectory ---
        # dive_rule="terminal_boost" dives the instant the sustain phase ends, which removes the
        # 120 s dead coast the earlier "max_range" rule left between sustain burnout and dive
        # entry. terminal_ignition_margin 1.2 is the tuned value: the physically-derived default
        # of 1.0 lights slightly late, because the pulse itself accelerates the rocket, and
        # leaves propellant still burning at impact. Measured by WP3 at 32 kg terminal
        # propellant: margin 1.0 gives Mach 1.523 with 1.9 s wasted, 1.2 gives Mach 1.649 with
        # 0.4 s wasted, 1.5 overshoots and burns out 2.1 s above the ground back at Mach 1.627.
        if dv.m_p_terminal > 0.0:
            mission = Mission(
                dv, reqs, motor, aero, mb.to_statement(),
                dive_rule="terminal_boost",
                terminal_ignition_margin=1.2,
            )
        else:
            mission = Mission(dv, reqs, motor, aero, mb.to_statement())
        # Adaptive stepping is roughly 6x faster than the fixed 0.02 s step for the same range to
        # within 0.1 percent (measured by WP3), which matters because the sizer needs tens of
        # trajectory evaluations.
        traj = mission.fly(dt=dt, adaptive=adaptive, tolerance=tolerance)
        res.traj = traj
        if traj.message:
            res.warnings.append(f"trajectory: {traj.message}")

        # --- [5] convergence test on launch mass and range ---
        m0 = mb.total
        rng = traj.range_final
        res.history.append(
            {
                "iteration": float(it),
                "m0": m0,
                "range": rng,
                "mach_terminal": traj.mach_final,
                "q_max": traj.q_max,
                "cruise_drag": drag_cruise,
            }
        )
        if it > 1:
            d_m0 = abs(m0 - m0_prev) / max(m0_prev, 1e-9)
            d_rng = abs(rng - range_prev) / max(range_prev, 1e-9)
            if d_m0 < tol and d_rng < tol:
                res.converged = True
                res.message = f"converged: d_m0 {d_m0:.2e}, d_range {d_rng:.2e}"
                break
        m0_prev, range_prev = m0, rng
    else:
        res.message = f"hit the {max_iter}-iteration limit without meeting tol {tol}"

    # A run with no nTop feedback is a legitimate converged answer after ONE iteration: the fixed
    # point in SPEC.md section 6 is driven entirely by the geometry measurements coming back, so
    # with nothing coming back there is nothing to iterate on. The earlier form of this test
    # required two iterations, which made `max_iter=1` incapable of ever reporting convergence and
    # silently poisoned every DOE row run at that budget.
    #
    # The same applies when the geometry call was attempted and failed: the point is degraded and
    # says so through `geometry_measured` and the warnings, but the analytic answer it fell back
    # to is still a converged analytic answer, not a numerical non-convergence.
    if not res.converged and res.traj is not None and not res.geometry_measured:
        res.converged = True
        res.message = (
            "converged (analytic geometry only, no nTop measurements to iterate on)"
            if geometry_fn is None
            else "converged (analytic fallback; the nTop geometry call did not return usable "
            "measurements, see warnings)"
        )

    # --- constraints ---
    if res.masses is not None and res.traj is not None:
        mb, traj = res.masses, res.traj
        inner = RocketAero(dv, meas=meas, nose_shape=dv.nose_shape)
        aero = CalibratedAero(inner, factor=cd0_factor)
        sm_min, sm_mach = static_margin_history(dv, mb, aero, traj)

        res.constraints = [
            ConstraintReport.check("R3 range", traj.range_final, reqs.range_min, ">=", "m"),
            ConstraintReport.check("R6 terminal Mach", traj.mach_final, reqs.M_terminal_min, ">=", "-"),
            ConstraintReport.check("R7 diameter", dv.D, reqs.D_max, "<=", "m"),
            ConstraintReport.check("R8 length", dv.L_total, reqs.L_max, "<=", "m"),
            ConstraintReport.check("R9 launch mass", mb.total, reqs.m0_max, "<=", "kg"),
            ConstraintReport.check(
                "R10 static margin", sm_min, reqs.static_margin_min, ">=", "calibres"
            ),
            ConstraintReport.check(
                "R11 fin span", dv.D + 2.0 * dv.b_fin, reqs.b_fin_span_max, "<=", "m"
            ),
            ConstraintReport.check("q_max", traj.q_max, reqs.q_max, "<=", "Pa"),
        ]
        grain_ld = _grain_ld(dv)
        if grain_ld is not None:
            res.constraints.append(
                ConstraintReport.check("grain L/D lower", grain_ld, 1.0, ">=", "-")
            )
            res.constraints.append(
                ConstraintReport.check("grain L/D upper", grain_ld, 8.0, "<=", "-")
            )
        if not math.isnan(sm_min):
            res.history.append({"static_margin_min": sm_min, "static_margin_mach": sm_mach})

    res.wall_time_s = time.perf_counter() - t_start
    return res


def _grain_ld(dv: DesignVector) -> float | None:
    """Grain length-to-diameter, from the motor model if it exposes it."""
    try:
        from .propulsion import SolidMotor

        g = SolidMotor(dv).grain_geometry()
        for attr in ("L_over_D", "length_over_diameter", "ld"):
            if hasattr(g, attr):
                return float(getattr(g, attr))
        if hasattr(g, "length") and hasattr(g, "outer_diameter"):
            return float(g.length) / float(g.outer_diameter)
    except Exception:                                     # noqa: BLE001
        return None
    return None


# --------------------------------------------------------------------------------------
#   The sizer
# --------------------------------------------------------------------------------------


@dataclass
class SizingResult:
    """Outcome of the design-variable search."""

    best: PointResult | None = None
    evaluations: list[PointResult] = field(default_factory=list)
    message: str = ""
    wall_time_s: float = 0.0

    @property
    def feasible_points(self) -> list[PointResult]:
        return [p for p in self.evaluations if p.feasible]

    def trace(self) -> list[dict[str, float]]:
        rows = []
        for i, p in enumerate(self.evaluations):
            rows.append(
                {
                    "eval": float(i),
                    "D": p.dv.D,
                    "L_total": p.dv.L_total,
                    "f_nose": p.dv.f_nose,
                    "m_p_boost": p.dv.m_p_boost,
                    "m_p_sustain": p.dv.m_p_sustain,
                    "F_boost": p.dv.F_boost,
                    "b_fin": p.dv.b_fin,
                    "c_r_fin": p.dv.c_r_fin,
                    "m0": p.m0,
                    "range_km": p.range_km,
                    "mach_terminal": p.traj.mach_final if p.traj else float("nan"),
                    "q_max_kPa": (p.traj.q_max / 1000.0) if p.traj else float("nan"),
                    "feasible": 1.0 if p.feasible else 0.0,
                    "n_violations": float(len(p.failed_constraints())),
                }
            )
        return rows


def penalty(p: PointResult, reqs: Requirements) -> float:
    """Objective for the search: launch mass plus a penalty for every violated constraint.

    The penalty is normalised by each constraint's own limit, so a 10 percent range shortfall and
    a 10 percent mass overrun cost the same. The mass term is scaled by `reqs.m0_max` so it is
    dimensionless too.

    A non-converged point gets a large finite value rather than infinity, so the search can still
    tell a nearly-working design from a hopeless one.
    """
    if not p.converged or p.masses is None or p.traj is None:
        return 1.0e3
    obj = p.m0 / reqs.m0_max
    viol = 0.0
    for c in p.constraints:
        if not c.met:
            viol += abs(c.margin)
    return obj + 25.0 * viol


def size(
    dv0: DesignVector,
    reqs: Requirements,
    geometry_fn: GeometryFn | None = None,
    run_dir: str | None = None,
    max_evals: int = 60,
    dt: float = 0.05,
    inner_iter: int = 3,
    adaptive: bool = True,
    tolerance: float = 1.0e-7,
    verbose: bool = True,
) -> SizingResult:
    """Move the design vector to minimise launch mass subject to the requirements.

    Deliberately a coordinate-descent pattern search, not a gradient method or a full optimiser:

    - The objective calls a trajectory integration and, when nTop is wired in, an `ntopcl`
      subprocess. Evaluations cost seconds, so the budget is tens of calls, not thousands.
    - The constraint boundaries (grain length, alpha limiting, motor thrust shortfall) put kinks
      in the objective that finite-difference gradients handle badly.
    - Pattern search needs no derivatives and makes monotone progress on the penalty, which is
      what matters for a demo that has to be reproducible.

    The variables are searched in the order that matters most for this configuration: propellant
    first (it drives range and mass), then the fins (static margin), then the body.
    """
    t0 = time.perf_counter()
    res = SizingResult()

    def evaluate(dv: DesignVector) -> PointResult:
        p = converge_point(
            dv, reqs, geometry_fn=geometry_fn, run_dir=run_dir, max_iter=inner_iter, dt=dt,
            adaptive=adaptive, tolerance=tolerance,
        )
        res.evaluations.append(p)
        return p

    current = evaluate(dv0)
    best_pen = penalty(current, reqs)
    if verbose:
        print(f"start:  {current.summary()}  penalty {best_pen:.4f}")

    # variable, initial step, minimum step
    schedule: list[tuple[str, float, float]] = [
        ("m_p_sustain", 60.0, 5.0),
        ("b_fin", 0.05, 0.005),
        ("c_r_fin", 0.08, 0.01),
        ("m_p_terminal", 12.0, 2.0),
        ("F_terminal", 4.0e3, 1.0e3),
        ("m_p_boost", 30.0, 5.0),
        ("F_boost", 12.0e3, 2.0e3),
        ("L_total", 0.20, 0.02),
        ("D", 0.03, 0.005),
        ("f_nose", 0.4, 0.05),
        ("x_fin_te_gap", 0.03, 0.01),
    ]
    bounds = dv0.bounds()
    bounds.setdefault("x_fin_te_gap", (0.02, 0.30))

    steps = {name: step for name, step, _ in schedule}
    mins = {name: mn for name, _, mn in schedule}

    while len(res.evaluations) < max_evals:
        improved = False
        for name, _, _ in schedule:
            if len(res.evaluations) >= max_evals:
                break
            step = steps[name]
            if step < mins[name]:
                continue
            lo, hi = bounds.get(name, (-math.inf, math.inf))
            for sign in (+1.0, -1.0):
                trial_val = getattr(current.dv, name) + sign * step
                if not (lo <= trial_val <= hi):
                    continue
                trial_dv = current.dv.replace(**{name: trial_val})
                ok, _ = trial_dv.geometry_is_valid()
                if not ok:
                    continue
                if len(res.evaluations) >= max_evals:
                    break
                trial = evaluate(trial_dv)
                pen = penalty(trial, reqs)
                if pen < best_pen - 1e-9:
                    current, best_pen, improved = trial, pen, True
                    if verbose:
                        print(
                            f"eval {len(res.evaluations):3d}: {name} -> {trial_val:.4g}  "
                            f"penalty {pen:.4f}  {trial.summary()}"
                        )
                    break
        if not improved:
            # nothing helped at the current step sizes, so refine them
            if all(steps[n] / 2.0 < mins[n] for n, _, _ in schedule):
                res.message = "pattern search converged: every step below its minimum"
                break
            for n, _, _ in schedule:
                steps[n] = steps[n] / 2.0
    else:
        res.message = f"stopped at the {max_evals}-evaluation budget"

    res.best = current
    res.wall_time_s = time.perf_counter() - t0
    if verbose:
        print(f"\nbest: {current.summary()}")
        print(f"{len(res.evaluations)} evaluations in {res.wall_time_s:.1f} s. {res.message}")
    return res
