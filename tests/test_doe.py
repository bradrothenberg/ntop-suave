"""Tests for the DOE sampler, runner bookkeeping and sensitivity ranking.

The sampling and statistics are checked against exact properties (stratification, reproducibility,
rank correlation on known monotone relations). The runner is checked on bookkeeping: a DOE that
drops failed points would report a feasible region that is too large, so the tests assert that
failures survive into the output.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from rocketgen.config import DesignVector, Requirements
from rocketgen.doe import (
    DoeResult,
    grid_samples,
    lhs_samples,
    run_doe,
    sensitivity,
)
from rocketgen.sizing.loop import PointResult


# --------------------------------------------------------------------------------------
#   Samplers
# --------------------------------------------------------------------------------------


def test_grid_is_a_full_factorial_in_declared_order():
    s = grid_samples({"D": [0.30, 0.35], "f_nose": [2.5, 3.0, 3.5]})
    assert len(s) == 6
    assert list(s[0].keys()) == ["D", "f_nose"]
    assert {(r["D"], r["f_nose"]) for r in s} == {
        (d, f) for d in (0.30, 0.35) for f in (2.5, 3.0, 3.5)
    }


def test_lhs_is_reproducible_for_a_given_seed():
    ranges = {"D": (0.25, 0.45), "m_p_sustain": (100.0, 500.0)}
    assert lhs_samples(ranges, 16, seed=7) == lhs_samples(ranges, 16, seed=7)


def test_lhs_seeds_differ():
    ranges = {"D": (0.25, 0.45)}
    assert lhs_samples(ranges, 16, seed=7) != lhs_samples(ranges, 16, seed=8)


@pytest.mark.parametrize("n", [4, 8, 32])
def test_lhs_puts_exactly_one_sample_in_every_stratum(n):
    """The defining property of a Latin hypercube, checked per variable."""
    lo, hi = 0.25, 0.45
    s = lhs_samples({"D": (lo, hi)}, n, seed=3)
    idx = np.floor((np.array([r["D"] for r in s]) - lo) / (hi - lo) * n).astype(int)
    assert sorted(idx.tolist()) == list(range(n))


def test_lhs_stays_inside_the_requested_ranges():
    ranges = {"D": (0.25, 0.45), "b_fin": (0.10, 0.30)}
    for r in lhs_samples(ranges, 64, seed=11):
        for k, (lo, hi) in ranges.items():
            assert lo <= r[k] <= hi


def test_lhs_covers_multiple_variables_independently():
    """Independent permutations must not leave the variables perfectly correlated."""
    s = lhs_samples({"a": (0.0, 1.0), "b": (0.0, 1.0)}, 64, seed=5)
    a = np.array([r["a"] for r in s])
    b = np.array([r["b"] for r in s])
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.4


# --------------------------------------------------------------------------------------
#   Sensitivity
# --------------------------------------------------------------------------------------


def _rows(xs, ys):
    return [{"converged": 1, "x": float(x), "y": float(y)} for x, y in zip(xs, ys)]


def test_rank_correlation_is_exactly_one_on_a_monotone_nonlinear_relation():
    """Spearman, not Pearson: a cubic is monotone but not linear, so rho must be exactly 1."""
    xs = list(range(12))
    res = DoeResult(rows=_rows(xs, [x**3 for x in xs]))
    assert sensitivity(res, ["x"], ["y"])["y"]["x"] == pytest.approx(1.0, abs=1e-12)


def test_rank_correlation_is_minus_one_when_decreasing():
    xs = list(range(12))
    res = DoeResult(rows=_rows(xs, [-(x**3) for x in xs]))
    assert sensitivity(res, ["x"], ["y"])["y"]["x"] == pytest.approx(-1.0, abs=1e-12)


def test_rank_correlation_handles_ties():
    """Tied values share an averaged rank, so a constant response gives nan, not a crash."""
    res = DoeResult(rows=_rows(range(10), [5.0] * 10))
    assert math.isnan(sensitivity(res, ["x"], ["y"])["y"]["x"])


def test_rank_correlation_needs_enough_samples():
    res = DoeResult(rows=_rows([1, 2], [1, 2]))
    assert math.isnan(sensitivity(res, ["x"], ["y"])["y"]["x"])


def test_sensitivity_ignores_points_that_did_not_converge():
    """A non-converged point carries a meaningless response and must not enter the statistic."""
    xs = list(range(12))
    rows = _rows(xs, [x**3 for x in xs])
    rows.append({"converged": 0, "x": 99.0, "y": -1.0e9})    # would destroy rho if counted
    assert sensitivity(DoeResult(rows=rows), ["x"], ["y"])["y"]["x"] == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------------------
#   Runner bookkeeping
# --------------------------------------------------------------------------------------


def test_invalid_geometry_is_recorded_not_dropped():
    """A geometrically impossible sample must still appear as a non-converged row.

    Dropping it would make the feasible fraction look better than it is.
    """
    base = DesignVector()
    # L_total 3.0 with the default bays leaves no cylindrical section
    samples = [{"L_total": 4.0}, {"L_total": 1.2}]
    res = run_doe(base, Requirements(), samples, inner_iter=1, dt=0.2, verbose=False)
    assert res.n_total == 2
    assert res.n_failed >= 1
    bad = [r for r in res.rows if r["converged"] == 0]
    assert bad and "invalid geometry" in bad[0]["message"]


def test_runner_records_every_sample_and_writes_csv(tmp_path):
    base = DesignVector()
    samples = [{"m_p_sustain": v} for v in (200.0, 240.0)]
    res = run_doe(base, Requirements(), samples, inner_iter=1, dt=0.2, verbose=False)
    assert res.n_total == len(samples) == len(res.points) == len(res.rows)

    csv_path = os.path.join(str(tmp_path), "doe.csv")
    res.to_csv(csv_path)
    with open(csv_path, encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == len(samples) + 1                 # header plus one row per sample
    assert "m0_kg" in lines[0] and "violations" in lines[0]

    json_path = os.path.join(str(tmp_path), "doe.json")
    res.to_json(json_path)
    assert os.path.getsize(json_path) > 0


def test_swept_variable_actually_reaches_the_design_vector():
    base = DesignVector()
    res = run_doe(
        base, Requirements(), [{"m_p_sustain": 180.0}], inner_iter=1, dt=0.2, verbose=False
    )
    assert res.points[0].dv.m_p_sustain == pytest.approx(180.0)
    assert res.rows[0]["m_p_sustain"] == pytest.approx(180.0)


def test_more_sustain_propellant_gives_more_range():
    """Physical monotonicity, and a check that the sweep is actually doing something."""
    base = DesignVector()
    res = run_doe(
        base,
        Requirements(),
        [{"m_p_sustain": 160.0}, {"m_p_sustain": 300.0}],
        inner_iter=1,
        dt=0.1,
        verbose=False,
    )
    lo, hi = res.rows[0], res.rows[1]
    assert lo["converged"] == 1 and hi["converged"] == 1
    assert hi["range_km"] > lo["range_km"]
    assert hi["m0_kg"] > lo["m0_kg"]


def test_a_raising_geometry_fn_is_recorded_not_propagated():
    """An nTop failure must degrade the point, not abort the whole study."""

    def broken(dv, run_dir):
        raise RuntimeError("ntopcl exploded")

    res = run_doe(
        DesignVector(),
        Requirements(),
        [{"m_p_sustain": 240.0}],
        geometry_fn=broken,
        inner_iter=1,
        dt=0.2,
        verbose=False,
    )
    assert res.n_total == 1
    p = res.points[0]
    assert p.geometry_measured is False
    assert any("ntopcl exploded" in w for w in p.warnings)
    assert res.rows[0]["geometry_measured"] == 0


def test_best_returns_none_when_nothing_is_feasible():
    impossible = Requirements(range_min=1.0e9)          # nothing can fly a million km
    res = run_doe(
        DesignVector(), impossible, [{"m_p_sustain": 240.0}], inner_iter=1, dt=0.2, verbose=False
    )
    assert res.n_feasible == 0
    assert res.best() is None


def test_empty_result_refuses_to_write_csv(tmp_path):
    with pytest.raises(ValueError):
        DoeResult().to_csv(os.path.join(str(tmp_path), "empty.csv"))
