"""Trade study over the design variables. SPEC.md section 7 item 4.

Two sampling modes:

- `grid`: a full factorial over named variables. Use it for the two- and three-variable trade
  charts that go in the report, because a regular grid contours cleanly.
- `lhs`: a Latin hypercube over any number of variables, for sensitivity ranking. Reproducible
  across platforms because the RNG is seeded explicitly and the permutation is done with
  `numpy.random.Generator`, not the global RNG.

Each sample is one `converge_point` call. When `geometry_fn` is supplied that includes a real
`ntopcl` run, so budget accordingly and validate at small scale first, per PLAN.md hard rule 5.

Failures are recorded, never dropped. A DOE that silently discards the points that crashed
reports a feasible region that is too large.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .config import DesignVector, Requirements
from .sizing.loop import GeometryFn, PointResult, converge_point
from .sizing.masses import PROPELLANT_ITEMS

# --------------------------------------------------------------------------------------
#   Samplers
# --------------------------------------------------------------------------------------


def grid_samples(axes: dict[str, Sequence[float]]) -> list[dict[str, float]]:
    """Full factorial. `axes` maps a `DesignVector` field name to the values it takes."""
    names = list(axes.keys())
    return [dict(zip(names, combo)) for combo in itertools.product(*(axes[n] for n in names))]


def lhs_samples(
    ranges: dict[str, tuple[float, float]], n: int, seed: int = 20260817
) -> list[dict[str, float]]:
    """Latin hypercube, centred in each stratum with a seeded permutation.

    Reproducible on any platform and any numpy 1.x: `default_rng(seed)` is specified to give the
    same stream everywhere, unlike the legacy global RNG.
    """
    rng = np.random.default_rng(seed)
    names = list(ranges.keys())
    k = len(names)
    # one independent permutation of the n strata per variable
    strata = np.empty((n, k))
    for j in range(k):
        perm = rng.permutation(n)
        strata[:, j] = (perm + rng.random(n)) / n
    out: list[dict[str, float]] = []
    for i in range(n):
        row = {}
        for j, name in enumerate(names):
            lo, hi = ranges[name]
            row[name] = lo + strata[i, j] * (hi - lo)
        out.append(row)
    return out


# --------------------------------------------------------------------------------------
#   Runner
# --------------------------------------------------------------------------------------


@dataclass
class DoeResult:
    """Every sample, feasible or not, plus the sampling metadata for reproducibility."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    points: list[PointResult] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    wall_time_s: float = 0.0

    @property
    def n_total(self) -> int:
        return len(self.rows)

    @property
    def n_feasible(self) -> int:
        return sum(1 for r in self.rows if r["feasible"] == 1)

    @property
    def n_failed(self) -> int:
        """Points that did not even converge, as opposed to converging but violating a limit."""
        return sum(1 for r in self.rows if r["converged"] == 0)

    def best(self) -> PointResult | None:
        """Lightest feasible point, or None when nothing is feasible."""
        feas = [p for p in self.points if p.feasible]
        if not feas:
            return None
        return min(feas, key=lambda p: p.m0)

    def to_csv(self, path: str) -> None:
        """Write every row, using the UNION of all row keys as the header.

        Taking the header from `rows[0]` alone is wrong: a sample that failed before the
        constraint set was built carries no `margin_*` columns, so if it happens to sort first
        every later row raises. Rows are padded with an empty string for keys they lack.
        """
        if not self.rows:
            raise ValueError("no DOE rows to write")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        keys: list[str] = []
        for r in self.rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(self.rows)

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta": self.meta,
                    "n_total": self.n_total,
                    "n_feasible": self.n_feasible,
                    "n_failed": self.n_failed,
                    "wall_time_s": self.wall_time_s,
                    "rows": self.rows,
                },
                f,
                indent=2,
            )

    def summary(self) -> str:
        return (
            f"{self.n_total} samples, {self.n_feasible} feasible, {self.n_failed} failed to "
            f"converge, {self.wall_time_s:.1f} s"
        )


def _row_for(dv: DesignVector, p: PointResult, sample: dict[str, float]) -> dict[str, Any]:
    """One flat CSV row. Swept variables first, then responses, then diagnostics."""
    row: dict[str, Any] = {name: sample[name] for name in sample}
    row.update(
        {
            "D": dv.D,
            "L_total": dv.L_total,
            "f_nose": dv.f_nose,
            "m_p_boost": dv.m_p_boost,
            "m_p_sustain": dv.m_p_sustain,
            "m_p_terminal": getattr(dv, "m_p_terminal", 0.0),
            "F_boost": dv.F_boost,
            "b_fin": dv.b_fin,
            "c_r_fin": dv.c_r_fin,
            "fineness": dv.fineness,
            "S_ref": dv.S_ref,
            # responses
            "m0_kg": p.m0,
            "range_km": p.range_km,
            "mach_terminal": p.traj.mach_final if p.traj else float("nan"),
            "q_max_kPa": (p.traj.q_max / 1000.0) if p.traj else float("nan"),
            "burnout_kg": (
                p.masses.excluding(*PROPELLANT_ITEMS)[0]
                if p.masses
                else float("nan")
            ),
            "x_cg_m": p.masses.x_cg if p.masses else float("nan"),
            # diagnostics
            "converged": 1 if p.converged else 0,
            "feasible": 1 if p.feasible else 0,
            "geometry_measured": 1 if p.geometry_measured else 0,
            "n_violations": len(p.failed_constraints()),
            "violations": "|".join(p.failed_constraints()),
            "iterations": p.iterations,
            "wall_time_s": round(p.wall_time_s, 3),
            "message": p.message,
        }
    )
    # per-constraint margins, so the report can plot how close each limit was
    for c in p.constraints:
        key = "margin_" + c.name.split()[0].replace("/", "_")
        row[key] = round(c.margin, 5)
    return row


def run_doe(
    base: DesignVector,
    reqs: Requirements,
    samples: Iterable[dict[str, float]],
    geometry_fn: GeometryFn | None = None,
    run_dir: str | None = None,
    inner_iter: int = 2,
    dt: float = 0.05,
    adaptive: bool = True,
    verbose: bool = True,
    meta: dict[str, Any] | None = None,
) -> DoeResult:
    """Evaluate every sample. Records failures rather than dropping them."""
    t0 = time.perf_counter()
    res = DoeResult(meta=dict(meta or {}))
    samples = list(samples)
    res.meta.setdefault("n_samples_requested", len(samples))
    res.meta.setdefault("geometry_fn", "nTop" if geometry_fn is not None else "analytic only")
    res.meta.setdefault("inner_iter", inner_iter)
    res.meta.setdefault("dt", dt)

    for i, sample in enumerate(samples, start=1):
        dv = base.replace(**sample)
        ok, errs = dv.geometry_is_valid()
        if not ok:
            # Recorded as a non-converged sample so the feasible-region count stays honest.
            p = PointResult(dv=dv, message="invalid geometry: " + "; ".join(errs))
            res.points.append(p)
            res.rows.append(_row_for(dv, p, sample))
            if verbose:
                print(f"[{i}/{len(samples)}] SKIP invalid: {'; '.join(errs)}")
            continue

        point_dir = None
        if run_dir is not None:
            point_dir = os.path.join(run_dir, f"pt_{i:04d}")
        try:
            p = converge_point(
                dv,
                reqs,
                geometry_fn=geometry_fn,
                run_dir=point_dir,
                max_iter=inner_iter,
                dt=dt,
                adaptive=adaptive,
            )
        except Exception as exc:                      # noqa: BLE001 - record, never hide
            p = PointResult(dv=dv, message=f"{type(exc).__name__}: {exc}")
        res.points.append(p)
        res.rows.append(_row_for(dv, p, sample))
        if verbose:
            tag = "FEAS" if p.feasible else ("conv" if p.converged else "FAIL")
            print(
                f"[{i}/{len(samples)}] {tag} "
                + ", ".join(f"{k}={v:.4g}" for k, v in sample.items())
                + f" -> m0 {p.m0:.1f} kg, range {p.range_km:.1f} km, "
                f"{len(p.failed_constraints())} violations"
            )

    res.wall_time_s = time.perf_counter() - t0
    if verbose:
        print("\n" + res.summary())
    return res


# --------------------------------------------------------------------------------------
#   Sensitivity ranking
# --------------------------------------------------------------------------------------


def sensitivity(res: DoeResult, variables: Sequence[str], responses: Sequence[str]) -> dict[str, dict[str, float]]:
    """Rank-correlation sensitivity of each response to each swept variable.

    Spearman rank correlation, not Pearson: the responses are monotone but not linear in the
    design variables (range versus propellant mass is a log-like curve), and rank correlation
    measures monotone association without assuming a functional form. Computed only over samples
    that converged.

    Returns response -> variable -> rho in [-1, 1]. `nan` when there is not enough spread.
    """

    def rankdata(a: np.ndarray) -> np.ndarray:
        """Average ranks, ties shared. Avoids a scipy.stats dependency."""
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(len(a), dtype=float)
        ranks[order] = np.arange(1, len(a) + 1, dtype=float)
        # average tied ranks
        sorted_a = a[order]
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
                j += 1
            if j > i:
                mean_rank = 0.5 * (i + j) + 1.0
                ranks[order[i : j + 1]] = mean_rank
            i = j + 1
        return ranks

    good = [r for r in res.rows if r["converged"] == 1]
    out: dict[str, dict[str, float]] = {}
    for resp in responses:
        out[resp] = {}
        y = np.array([float(r[resp]) for r in good], dtype=float)
        for var in variables:
            x = np.array([float(r[var]) for r in good], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 4 or np.ptp(x[mask]) == 0.0 or np.ptp(y[mask]) == 0.0:
                out[resp][var] = float("nan")
                continue
            rx, ry = rankdata(x[mask]), rankdata(y[mask])
            rx -= rx.mean()
            ry -= ry.mean()
            denom = math.sqrt(float((rx**2).sum()) * float((ry**2).sum()))
            out[resp][var] = float((rx * ry).sum() / denom) if denom > 0 else float("nan")
    return out
