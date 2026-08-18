# IV-1 with a splined outer mould line

The IV-1 two-stage interceptor, re-sized with its payload-stage nose and its interstage flare
built as **true revolved B-splines in nTop** instead of a tangent ogive and a straight cone.
The tangent-ogive result is included alongside it, produced by the same code on the same day,
because a comparison whose baseline is not in the repository cannot be checked.

## What changed, and what did not

Only the outer mould line shape. Same requirements, same propellant loads, same stage
diameters and lengths, same attitude-control pack sizing loop, same pitchover search.

| | tangent ogive | spline |
|---|---|---|
| Payload-stage nose | tangent ogive | cubic B-spline, `nose_blend = 1.0` |
| Interstage flare | straight truncated cone | cubic B-spline, `interstage_blend = 1.0` |

## Result

Both designs are FEASIBLE against all 15 constraints at a 38 degree pitchover.

| quantity | ogive | spline | delta |
|---|---|---|---|
| Launch mass | 592.24 kg | 591.05 kg | -1.19 kg |
| Slant range | 160.93 km (100.0 mi) | 160.93 km | held at the requirement |
| **Intercept altitude** | 19.95 km | **24.39 km** | **+4.44 km, +22.3 percent** |
| Intercept Mach | 4.004 | 4.057 | +1.3 percent |
| Time to intercept | 157.8 s | 153.7 s | -4.1 s |
| Lateral g, aerodynamic | 7.45 | 3.84 | -48.5 percent |

The headline is the intercept altitude. Slant range is pinned at the 100 mile requirement, so
the drag saving does not show up as extra range; it shows up as reaching that range **higher and
faster**, which is what A3 and A4 actually ask for.

`ogive_vs_spline.csv` carries the same table in machine-readable form.

### The result is a trade, not a free win

Aerodynamic lateral acceleration at intercept **halves**, from 7.45 g to 3.84 g. That is not a
modelling artefact: the intercept now happens 4.4 km higher, where dynamic pressure is lower, and
aerodynamic manoeuvre authority scales with dynamic pressure. The A11 lateral-g requirement is
still met, but it is met almost entirely by the attitude-control pack rather than aerodynamically.

A vehicle sized this way is more dependent on its ACS than the ogive vehicle was. That is a real
consequence of the shape change and it is recorded here rather than left for a reader to notice.

### The spline opens a pitchover angle the ogive could not reach

Sweeping the commanded pitchover angle, with everything else held:

| gamma | ogive | spline |
|---|---|---|
| 32 deg | 151.7 km, 0.0 km, M 1.76 | 159.3 km, 0.0 km, M 1.88 |
| 34 deg | 160.9 km, 2.9 km, M 2.37 | 160.9 km, 8.7 km, M 3.34 |
| 36 deg | 160.9 km, 12.0 km, M 3.70, **fails A3** | 160.9 km, 16.9 km, M 3.98, **FEASIBLE** |
| 38 deg | 160.9 km, 19.9 km, M 4.00, FEASIBLE | 160.9 km, 24.4 km, M 4.06, FEASIBLE |

The ogive design had exactly one feasible pitchover angle, at the edge of the swept range. The
spline design has two, and clears the 15 km minimum intercept altitude a full two degrees
earlier. The feasible region is wider, not merely better at one point.

## Why the shape can change anything at all

Before this work, no drag model in the repository could distinguish one nose from another at
fixed fineness: `aero_iv1.CD_wave_body` is the Bonney correlation, a function of `L/D` alone.
A spline study run against that model would have reported "no change" for the wrong reason.

`rocketgen/sizing/wavedrag.py` supplies the missing sensitivity from linearised slender-body
theory, as a dimensionless RATIO against the tangent ogive that multiplies the Bonney value. So
the calibrated drag level and its Mach dependence stay with the correlation that was validated
against real data, and only the part linear theory is good at - how much worse one shape is than
another - is taken from linear theory. At the ogive shape the ratio is exactly 1.0, so every
pre-spline result is reproduced.

That model is validated against two exact closed forms, neither of which is a curve fit:

| check | achieved |
|---|---|
| Sears-Haack body, `D/q = 128 V^2 / (pi L^4)` | machine precision |
| Von Karman ogive, `C_D = (d/L)^2` on base area | 2.1e-5 |
| Glauert series against direct double integration | converges onto it |
| Von Karman ogive IS the constrained optimum | asserted, not assumed |

The tangent ogive carries 1.17 times the minimum-possible wave drag at every fineness in the
design range. A 9-control-point spline recovers 86 percent of that gap.

## How the spline is built in nTop

nTop revolves the spline itself. There is no chord polygon and no discretisation error.

```
spline_by_control_points<list<point>,integer>[5.20.0]      -> spline
core.list<curve_interface>
profile_from_curves<list<curve_interface>,vector>[5.20.0]  -> new_profile
revolve<new_profile,axis,real>[5.20.0]                     -> implicit
```

The profile is a curve LIST mixing that spline with `two_point_line<point,point>` segments, so
the cylinder, the base and the return along the axis stay exactly straight and the corners stay
sharp. One spline through the whole outline would round them off.

None of those blocks is in the vendored block universe, so all of them are emitted through
`Recipe.raw_block`. `docs/NTOP_NOTES.md` section 25 records the four encoding traps that make
`ntopcl convert` fail with a bare `Error loading recipe`.

The measured stage-2 volume agrees with an independent analytic integral of the same B-spline to
**+0.011 percent**, and the whole stack measures in 72 s against 127 s for the ogive notebook.

## Files

| file | what it is |
|---|---|
| `IV1_spline_engineering_report.pdf` | the write-up: 20 pages, 4 figures. Read this first. |
| `ogive_vs_spline.csv` | the comparison table above, machine readable |
| `01_design/converged.json` | the full converged spline design: design vector, constraints, mass statement, intercept state |
| `01_design/constraints.csv` | all 15 requirements with value, limit and pass/fail |
| `01_design/trajectory.png` | the ascent and intercept trajectory |
| `02_geometry/iv1_stages.json` | per-stage nTop measurements: volume, wetted area, structural mass, CG, inertia |
| `02_geometry/iv1_measurements.json` | the raw measurement set |
| `03_ogive_baseline/` | the tangent-ogive result, same code, same day |
| `04_validation/` | `validation_summary.csv` and the machine-readable `evidence.json` behind it |
| `figures/` | the four report figures as PNG |

## Reproducing

```
.venv/Scripts/python.exe scripts/iv1_converge.py --ntop --oml spline --nose-blend 1.0 --interstage-blend 1.0
.venv/Scripts/python.exe scripts/iv1_converge.py --ntop            # the ogive baseline
.venv/Scripts/python.exe scripts/build_example_iv1_spline.py
```

## What is NOT in this result

- `config.CD0_CALIBRATION`, the Basic Finner calibration, is still not wired into IV-1. That was
  an open defect before this work and it remains one; see CLAUDE.md section 3.8. The spline
  comparison is unaffected, because both sides omit it equally, but the absolute drag level is
  known to run low.
- The nozzle model is still ideal. Real delivered specific impulse for this class runs 3 to 7
  percent lower, and that penalty is not applied.
- No DOE was run for IV-1. The comparison is two converged points and a four-point pitchover
  sweep, not a trade study.
- **No render of the vehicle.** Neither IV-1 run enabled mesh export, so there is no STL under
  `runs/IV-1_spline/geom` to render from. Section 9 of the report says so rather than quietly
  omitting the figure.
- The wave-drag residuals quoted in the report are measured on a 4001-station check table and
  are recorded in `04_validation/`. They are NOT "machine precision": Sears-Haack comes out at
  4.5e-5 and von Karman at 1.3e-5. The residual belongs to the check table rather than the
  model, because the Sears-Haack profile has infinite end slopes that a central difference
  resolves slowly; refining 501 to 8001 stations drives it from 1.7e-2 to 1.7e-5.
