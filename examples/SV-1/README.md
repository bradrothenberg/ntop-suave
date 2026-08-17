# SV-1 reference example

The converged design, its geometry, the trade study and the evidence behind them. Produced by
`run_sv1.py` and curated by `scripts/build_example.py`. Every number here came out of the run;
nothing is typed in by hand.

**The requirements for this vehicle are invented for the demonstration.** They correspond to no
real programme. See `SPEC.md` at the repo root, and section 6 of the report.

## Start here

| | |
|---|---|
| **[SV1_engineering_report.pdf](SV1_engineering_report.pdf)** | The write-up: 20 pages, 8 figures, 17 tables. Read this first. |
| [01_design/design_summary.csv](01_design/design_summary.csv) | The headline numbers on one page. |
| [04_figures/sv1_iso.png](04_figures/sv1_iso.png) | What it looks like. |
| [02_geometry/sv1.ntop](02_geometry/sv1.ntop) | The parametric nTop notebook. Open it in nTop. |

## Headline result

Converged with real nTop geometry inside the sizing loop. All ten constraints met.

| Quantity | Value | Requirement |
|---|---|---|
| Launch mass | 554.3 kg | <= 1100 kg |
| Range | 189.5 km | >= 185 km |
| Mach at impact | 1.66 | >= 1.50 |
| Maximum dynamic pressure | 195.1 kPa | <= 200 kPa, **active constraint** |
| Body diameter, overall length | 0.35 m, 3.60 m | <= 0.45 m, <= 4.20 m |

## What is in each folder

### 01_design
The sized vehicle.

| File | What it is |
|---|---|
| `design_summary.csv` | Headline quantities against their requirements. Open this first. |
| `design_vector.csv` | Every geometry and propulsion parameter, with units and a description. |
| `mass_statement.csv` | Group-weight statement. The `provenance` column says whether each line was measured by nTop, computed analytically, taken from a requirement, or taken from a correlation. |
| `constraints.csv` | All ten constraints with their margins. |
| `ntop_coupling_effect.csv` | The same design sized with analytic geometry and with nTop geometry. This is the point of the project. |
| `trajectory_history.csv` | The flown trajectory, decimated, with the phase labelled per row. |
| `sources.csv` | Every registered constant, and whether it is sourced or a guess. The `confidence` column is the one to read. |
| `point_ntop.json`, `point_analytic.json` | The complete machine-readable records the CSVs were flattened from. |
| `provenance.json` | Environment, requirements and the full source registry. |

### 02_geometry
What nTop built and measured.

| File | What it is |
|---|---|
| `sv1.ntop` | **The parametric nTop notebook.** Every design variable is a real notebook input, so you can open it and change the rocket. |
| `sv1_recipe.json` | The recipe JSON the notebook was converted from. This is the human-readable source; `ntopcl convert` turns it into the `.ntop`. |
| `sv1_input.json` | The `ntopcl` input JSON for this design point. |
| `sv1_output.json` | What the notebook returned. |
| `sv1.stl` | Surface mesh. GitHub renders this in the browser. |
| `sv1.step` | CAD interchange, for import into anything else. |
| `sv1.implicit` | nTop implicit body, for field queries through nTop Core. |
| `sv1_measurements.json`, `measurements_flat.csv` | Everything nTop measured. |
| `area_distribution.csv` | Cross-section area against station, 16 stations, used for wave drag. |

nTop measured the enclosed volume to within 0.013 percent of independent closed-form geometry, the
body wetted area to within 0.224 percent, and the area distribution to within 0.16 percent at the
worst station.

### 03_trade_study
| File | What it is |
|---|---|
| `grid.csv` | 45-node full factorial over diameter, sustain propellant and nose fineness. Only 3 nodes are feasible. |
| `lhs.csv` | 40-sample Latin hypercube over 8 variables, seeded and reproducible. |
| `sensitivity.csv` | Spearman rank correlation of each response against each variable. |

Both files record every sample, including the ones that failed. A study that drops its failures
reports a feasible region that is too large.

### 04_figures
Every figure in the report, as PNG. All are produced by scripts under `rocketgen/report/`.

### 05_validation
| File | What it is |
|---|---|
| `validation_summary.csv` | Every validation check and its measured result. |
| `evidence.json` | The machine-readable evidence, recomputed from live code rather than transcribed. |

## Reproducing it

```
.venv/Scripts/python.exe run_sv1.py --stage size
.venv/Scripts/python.exe run_sv1.py --stage doe --doe-scale full
.venv/Scripts/python.exe -m rocketgen.report.build_report
.venv/Scripts/python.exe scripts/build_example.py
```
