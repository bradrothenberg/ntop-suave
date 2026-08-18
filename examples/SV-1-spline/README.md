# SV-1 with a splined outer mould line

The SV-1 rocket, re-sized with its nose and boattail built as **true revolved B-splines in
nTop** instead of a tangent ogive and a conical boattail. The tangent-ogive result is in
`examples/SV-1/` and is the comparison baseline throughout.

## What changed, and what did not

Only the outer mould line shape family, plus the two blend scalars the sizer is now allowed to
move. Same requirements, same physics modules, same sizing search, same DOE axes.

## Result

| quantity | ogive | spline | delta |
|---|---|---|---|
| Launch mass | 554.3 kg | **551.3 kg** | -3.0 kg |
| Range | 189.3 km | **194.0 km** | +4.7 km |
| Impact Mach | 1.65 | 1.68 | |
| q_max | 193.5 kPa | 199.1 kPa | limit 200 |
| Static margin | | 1.59 calibres | limit 1.0 |

All ten constraints met. 60 sizing evaluations in 2115 s with real nTop geometry in the loop.

### The sizer declined the drag optimum

`nose_blend` converged to **0.7, not 1.0**, and `boattail_blend` to **0.35**. The slender-body
drag optimum was available and the search moved away from it. Past roughly 0.7 the forebody
volume the optimal shape gives up, and the aft shift in centre of pressure it causes, cost more
than the remaining wave drag saves. The shape trade has a genuine interior optimum, which is
only visible because the drag model can now see shape at all.

Read the nTop-coupled blend sweep in `05_validation/evidence.json` carefully before quoting this.
Of the five blends re-measured in nTop, the lowest penalty is at **0.85**, not at the 0.70 the
search stopped on, and 1.00 is worse than 0.85 on both penalty and launch mass. So the optimum is
confirmed INTERIOR. The converged blend is NOT confirmed optimal. The five penalties span only
0.00049, so the objective is nearly flat in this variable and the search had little to follow.

### The drag saving was not free

`q_max` rose from 193.5 to 199.1 kPa against a 200 kPa structural limit, a margin of 0.44
percent. Lower drag buys speed and speed is paid for in dynamic pressure, so the spline design
sits harder against the structural constraint than the ogive design did. A reader taking the
mass and range numbers without this one would be misreading the result.

## The coupling still earns its keep

`01_design/ntop_coupling_effect.csv`, the same design vector evaluated with measured nTop
geometry and with the closed-form fallback:

| quantity | analytic | nTop-measured | change |
|---|---|---|---|
| Launch mass | 542.57 kg | 551.27 kg | +8.71 kg |
| Range | 196.26 km | 194.00 km | -2.25 km |
| q_max | 201.01 kPa | 199.11 kPa | -1.90 kPa |

The analytic geometry reports a design that is 8.7 kg lighter, flies 2.3 km further, and **is
infeasible**: it violates both the static margin and the q_max limit. Only the measured geometry
finds the feasible answer. That is the coupling doing its job on this vehicle, at this design
point, and it is a larger effect than the shape change itself.

## Why the shape can change anything at all

Before this work, no drag model here could distinguish one nose from another at fixed fineness:
`aero.CD_wave_body` is the Bonney correlation, a function of `L/D` alone, and the Sears-Haack
cross-check is a function of `(d/L)^2` alone. A spline study run against that would have
reported "no change" for the wrong reason.

`rocketgen/sizing/wavedrag.py` supplies the missing sensitivity from linearised slender-body
theory as a dimensionless RATIO against the tangent ogive, multiplying the Bonney value. The
calibrated level and Mach dependence stay with the correlation validated against 23 Basic Finner
free-flight shots; only the shape ratio comes from linear theory. At the ogive shape the ratio is
exactly 1.0 and CD0 is reproduced bit for bit, asserted with `==` rather than a tolerance.

Validated against two exact closed forms. Every residual below is measured, on a 4001-station
check table, and recorded in `05_validation/validation_summary.csv`:

| check | achieved |
|---|---|
| Sears-Haack body, `D/q = 128 V^2 / (pi L^4)` | 4.5e-5 |
| Von Karman ogive, `D/q` against the closed form | 1.3e-5 |
| Von Karman ogive, `C_D = (d/L)^2` on base area | 1.3e-5 |
| Optimum Glauert shape factor against `4/pi` | 6.8e-5 |
| Glauert series against direct double integration | converges onto it |
| Von Karman ogive IS the constrained optimum | asserted, not assumed |

The Sears-Haack residual belongs to the CHECK TABLE, not to the model: that profile has an
infinite slope at both ends, so the central-difference derivative converges slowly on it.
Refining the table from 501 to 8001 stations drives the residual from 1.7e-2 to 1.7e-5.

Measured effect on this vehicle: nose wave drag is 22 percent of CD0 at Mach 1.2 rising to 36
percent at Mach 4, and the optimal spline nose cuts it 12.5 percent, worth 2.9 to 4.8 percent of
total CD0.

## How the spline is built in nTop

nTop revolves the spline itself. No chord polygon, no discretisation error.

```
spline_by_control_points<list<point>,integer>[5.20.0]      -> spline
core.list<curve_interface>
profile_from_curves<list<curve_interface>,vector>[5.20.0]  -> new_profile
revolve<new_profile,axis,real>[5.20.0]                     -> implicit
```

The profile is a curve LIST mixing splines with `two_point_line<point,point>` segments, so the
cylinder, the base disc and the return along the axis stay exactly straight and the corners stay
sharp. The control points are computed inside nTop from live inputs, and their axial fractions
are the **Greville abscissae**, which makes `x(u) = L u` exactly and keeps the radius a spline in
the axial station.

None of those blocks is in the vendored universe; all go through `Recipe.raw_block`.
`docs/NTOP_NOTES.md` section 25 records the four encoding traps.

Measured OML volume agrees with an independent analytic integral of the same B-spline to
**-0.0064 percent**, and the body wetted area to +0.0148 percent. See
`05_validation/evidence.json`, section `spline_geometry`. The spline recipe carries 317 body
nodes against 405 for the ogive recipe.

## Trade study

`03_trade_study/` holds a 45-point factorial over `D`, `m_p_sustain`, `f_nose` and a 40-sample
Latin hypercube over eight variables. 85 samples, **0 failed to converge**, 3 feasible. The
factorial is centred on the converged design, which is itself a grid node.

Spearman rank correlations from the LHS:

| variable | m0 | range | impact Mach | q_max |
|---|---|---|---|---|
| `m_p_sustain` | **+0.72** | +0.58 | -0.03 | -0.19 |
| `m_p_boost` | +0.56 | +0.12 | -0.28 | -0.05 |
| `D` | +0.13 | **-0.79** | -0.75 | -0.66 |
| `m_p_terminal` | +0.01 | -0.13 | **+0.63** | +0.40 |
| `c_r_fin` | -0.35 | -0.20 | -0.00 | +0.08 |

Each requirement has a distinct dominant lever, which is what makes the design tractable.

## Files

Numbered so the reading order is obvious.

| directory | what is in it |
|---|---|
| `SV1_spline_engineering_report.pdf` | the write-up: 19 pages, 10 figures, 20 tables. Read this first. |
| `01_design/` | converged design vector, constraint table, mass statement, the analytic-vs-measured coupling table, source provenance |
| `02_geometry/` | the `.ntop` notebook regenerated with neutral export paths, plus STL, STEP, `.implicit` and the measurement JSON |
| `03_trade_study/` | factorial and LHS results as CSV, plus the sensitivity table |
| `04_figures/` | the ten report figures as PNG |
| `05_validation/` | `validation_summary.csv` and the machine-readable `evidence.json` behind it |

## Reproducing

```
.venv/Scripts/python.exe run_sv1.py --stage smoke --oml spline
.venv/Scripts/python.exe run_sv1.py --stage size --oml spline
.venv/Scripts/python.exe run_sv1.py --stage doe --oml spline
.venv/Scripts/python.exe run_sv1.py --stage converged --oml spline
.venv/Scripts/python.exe -m rocketgen.report.evidence --oml spline
.venv/Scripts/python.exe -m rocketgen.report.fig_oml --oml spline
.venv/Scripts/python.exe -m rocketgen.report.fig_wavedrag --oml spline
.venv/Scripts/python.exe -m rocketgen.report.fig_flight --oml spline
.venv/Scripts/python.exe -m rocketgen.report.build_report_spline
.venv/Scripts/python.exe scripts/build_example.py --oml spline
```

## What is NOT in this result

- The nozzle model is still ideal: no two-phase, divergence or combustion loss. Real delivered
  specific impulse for this class runs 3 to 7 percent lower and that penalty is not applied. It
  remains the largest declared optimism in the result.
- The slender-body wave-drag ratio is Mach-independent at this order. The true shape sensitivity
  does drift with Mach number; that drift is not modelled and has not been quantified.
