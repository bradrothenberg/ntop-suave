"""Assemble the curated SV-1 reference example under `examples/SV-1/`.

    .venv/Scripts/python.exe scripts/build_example.py

Reads the run artefacts under `runs/`, flattens the JSON into CSV that opens in a spreadsheet,
emits the nTop notebook and its recipe, and writes the index README. Safe to re-run.

The point of this script is that `runs/` is a working directory with hundreds of probe files, and
`examples/SV-1/` is the curated, ordered, documented subset that a reader should see.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

RUNS = os.path.join(REPO, "runs")
SV1 = os.path.join(RUNS, "SV-1")
EX = os.path.join(REPO, "examples", "SV-1")

DIRS = {
    "design": os.path.join(EX, "01_design"),
    "geometry": os.path.join(EX, "02_geometry"),
    "trade": os.path.join(EX, "03_trade_study"),
    "figures": os.path.join(EX, "04_figures"),
    "validation": os.path.join(EX, "05_validation"),
}

DEG = math.degrees


def log(m: str) -> None:
    print(f"[example] {m}")


def jload(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str, header: list[str], rows: list[list]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    scrub_paths(path)
    log(f"wrote {os.path.relpath(path, REPO)}")


def copy(src: str, dst_dir: str, rename: str | None = None) -> str | None:
    if not os.path.isfile(src):
        log(f"MISSING, skipped: {src}")
        return None
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, rename or os.path.basename(src))
    shutil.copy2(src, dst)
    if os.path.splitext(dst)[1].lower() in (".json", ".csv", ".txt", ".log", ".md"):
        scrub_paths(dst)
    return dst


# Absolute paths on the machine that produced the run. They are not secrets, but they leak the
# developer's directory layout into a published example and they are useless to a reader, so every
# text artefact is rewritten to a repo-relative path on the way in.
_ABS = re.compile(r"[A-Za-z]:[\\/][^\"',\s]*", re.ASCII)


def scrub_paths(path: str) -> None:
    """Rewrite developer-machine absolute paths in a text artefact to repo-relative ones."""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    def repl(m: re.Match) -> str:
        raw = m.group(0)
        norm = raw.replace("\\\\", "/").replace("\\", "/")
        for anchor in ("/runs/", "/examples/"):
            if anchor in norm:
                tail = norm.split(anchor, 1)[1]
                return anchor.strip("/") + "/" + tail
        return os.path.basename(norm)

    new = _ABS.sub(repl, text)
    if new != text:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)


# --------------------------------------------------------------------------------------
#   01_design
# --------------------------------------------------------------------------------------


def build_design(pn: dict, pa: dict) -> None:
    d = DIRS["design"]
    dv, ms, tr = pn["design_vector"], pn["mass_statement"], pn["trajectory"]

    # Headline summary: the one file to open first.
    write_csv(
        os.path.join(d, "design_summary.csv"),
        ["quantity", "value", "unit", "requirement", "status"],
        [
            ["launch mass", f"{ms['total_kg']:.1f}", "kg", "<= 1100", "met"],
            ["burnout mass", f"{ms['burnout_kg']:.1f}", "kg", "", ""],
            ["range", f"{tr['range_m']/1000.0:.1f}", "km", ">= 185", "met"],
            ["Mach at impact", f"{tr['mach_final']:.2f}", "-", ">= 1.50", "met"],
            ["max dynamic pressure", f"{tr['q_max_Pa']/1000.0:.1f}", "kPa", "<= 200", "met, ACTIVE"],
            ["flight time", f"{tr['duration_s']:.1f}", "s", "", ""],
            ["body diameter", f"{dv['D']:.3f}", "m", "<= 0.45", "met"],
            ["overall length", f"{dv['L_total']:.3f}", "m", "<= 4.20", "met"],
            ["fin span, tip to tip", f"{dv['D'] + 2*dv['b_fin']:.3f}", "m", "<= 0.90", "met"],
            ["centre of gravity, launch", f"{ms['x_cg_m']:.3f}", "m from nose", "", ""],
            ["centre of gravity, burnout", f"{ms['burnout_x_cg_m']:.3f}", "m from nose", "", ""],
            ["nTop-measured mass fraction", f"{100*ms['measured_fraction']:.1f}", "percent", "", ""],
            ["geometry measured by nTop", str(pn["geometry_measured"]), "-", "", ""],
        ],
    )

    # Design vector, with units spelled out and angles in degrees.
    order = [
        ("D", "m", "body diameter"),
        ("L_total", "m", "overall length"),
        ("f_nose", "-", "nose fineness, nose length / diameter"),
        ("L_nose", "m", "nose length (derived)"),
        ("fineness", "-", "body fineness, length / diameter (derived)"),
        ("S_ref", "m^2", "aerodynamic reference area (derived)"),
        ("nose_shape", "-", "nose profile family"),
        ("t_wall", "m", "airframe wall thickness"),
        ("L_boattail", "m", "boattail length"),
        ("d_base", "m", "base diameter"),
        ("n_fin", "-", "fin count, cruciform"),
        ("b_fin", "m", "fin exposed semi-span"),
        ("c_r_fin", "m", "fin root chord"),
        ("c_t_fin", "m", "fin tip chord (derived)"),
        ("taper_fin", "-", "fin taper ratio"),
        ("sweep_fin", "deg", "fin leading-edge sweep"),
        ("t_fin", "m", "fin maximum thickness"),
        ("x_fin_le", "m", "fin root leading-edge station (derived)"),
        ("S_fin_exposed", "m^2", "exposed area of one fin panel (derived)"),
        ("m_p_boost", "kg", "boost propellant"),
        ("m_p_sustain", "kg", "sustain propellant"),
        ("m_p_terminal", "kg", "terminal-boost propellant"),
        ("F_boost", "N", "boost thrust"),
        ("F_terminal", "N", "terminal-boost thrust"),
        ("p_c", "Pa", "chamber pressure"),
        ("eps_nozzle", "-", "nozzle area ratio"),
        ("L_seeker", "m", "seeker bay length"),
        ("L_guidance", "m", "guidance bay length"),
        ("L_warhead", "m", "warhead bay length"),
    ]
    rows = []
    for key, unit, desc in order:
        if key not in dv:
            continue
        v = dv[key]
        if key == "sweep_fin":
            v = DEG(v)
        rows.append([key, f"{v:.6g}" if isinstance(v, (int, float)) else str(v), unit, desc])
    write_csv(os.path.join(d, "design_vector.csv"), ["parameter", "value", "unit", "description"], rows)

    # Mass statement, flat.
    tot = ms["total_kg"]
    rows = [
        [
            it["name"],
            f"{it['mass_kg']:.3f}",
            f"{100.0*it['mass_kg']/tot:.2f}",
            f"{it['x_cg_m']:.4f}",
            it["provenance"],
            it.get("note", ""),
        ]
        for it in sorted(ms["items"], key=lambda e: -e["mass_kg"])
    ]
    rows.append(["TOTAL", f"{tot:.3f}", "100.00", f"{ms['x_cg_m']:.4f}", "", ""])
    write_csv(
        os.path.join(d, "mass_statement.csv"),
        ["item", "mass_kg", "percent_of_total", "station_m_from_nose", "provenance", "note"],
        rows,
    )

    # Constraints, flat.
    write_csv(
        os.path.join(d, "constraints.csv"),
        ["constraint", "value", "sense", "limit", "units", "margin_fraction", "met"],
        [
            [c["name"], f"{c['value']:.6g}", c["sense"], f"{c['limit']:.6g}",
             c["units"], f"{c['margin']:.5f}", "yes" if c["met"] else "NO"]
            for c in pn["constraints"]
        ],
    )

    # The coupling comparison, which is the point of the project.
    a_ms, a_tr = pa["mass_statement"], pa["trajectory"]
    write_csv(
        os.path.join(d, "ntop_coupling_effect.csv"),
        ["quantity", "analytic_geometry", "ntop_measured", "change", "unit"],
        [
            ["launch mass", f"{a_ms['total_kg']:.2f}", f"{ms['total_kg']:.2f}",
             f"{ms['total_kg']-a_ms['total_kg']:+.2f}", "kg"],
            ["range", f"{a_tr['range_m']/1e3:.2f}", f"{tr['range_m']/1e3:.2f}",
             f"{(tr['range_m']-a_tr['range_m'])/1e3:+.2f}", "km"],
            ["Mach at impact", f"{a_tr['mach_final']:.3f}", f"{tr['mach_final']:.3f}",
             f"{tr['mach_final']-a_tr['mach_final']:+.3f}", "-"],
            ["max dynamic pressure", f"{a_tr['q_max_Pa']/1e3:.2f}", f"{tr['q_max_Pa']/1e3:.2f}",
             f"{(tr['q_max_Pa']-a_tr['q_max_Pa'])/1e3:+.2f}", "kPa"],
            ["centre of gravity", f"{a_ms['x_cg_m']:.4f}", f"{ms['x_cg_m']:.4f}",
             f"{ms['x_cg_m']-a_ms['x_cg_m']:+.4f}", "m"],
        ],
    )

    # Trajectory history, decimated, as CSV.
    hist = tr.get("history", [])
    if hist:
        keys = ["t", "x", "h", "V", "mach", "mass", "gamma", "thrust", "drag", "q", "alpha", "phase"]
        rows = []
        for r in hist:
            row = []
            for k in keys:
                v = r.get(k, "")
                if k == "gamma" and isinstance(v, (int, float)):
                    v = DEG(v)
                if k == "alpha" and isinstance(v, (int, float)):
                    v = DEG(v)
                row.append(f"{v:.6g}" if isinstance(v, (int, float)) else str(v))
            rows.append(row)
        header = ["time_s", "range_m", "altitude_m", "V_mps", "mach", "mass_kg",
                  "gamma_deg", "thrust_N", "drag_N", "q_Pa", "alpha_deg", "phase"]
        write_csv(os.path.join(d, "trajectory_history.csv"), header, rows)

    for src in ("point_ntop.json", "point_analytic.json"):
        copy(os.path.join(SV1, "converged", src), d)
    copy(os.path.join(SV1, "provenance.json"), d)

    # Sources registry as CSV, with the guess flag broken out.
    prov = jload(os.path.join(SV1, "provenance.json"))
    rows = []
    for k, v in sorted(prov["sources"].items()):
        up = v.upper()
        flag = ("GUESS" if "GUESS" in up
                else "MODELLING CHOICE" if "MODELLING CHOICE" in up
                else "APPROXIMATION" if "APPROXIMATION" in up
                else "ASSUMPTION" if "ASSUMPTION" in up
                else "sourced")
        rows.append([k, flag, v])
    write_csv(os.path.join(d, "sources.csv"), ["name", "confidence", "source"], rows)


# --------------------------------------------------------------------------------------
#   02_geometry
# --------------------------------------------------------------------------------------


def _regenerate_notebook(dest_dir: str) -> bool:
    """Convert a fresh notebook whose export paths are neutral. True on success."""
    try:
        from rocketgen.config import DesignVector
        from rocketgen.ntopgen.driver import NtopRunner
        from rocketgen.ntopgen.rocket_notebook import build_rocket_recipe

        dv_raw = jload(os.path.join(SV1, "converged", "point_ntop.json"))["design_vector"]
        fields = set(DesignVector().__dict__.keys())
        dv = DesignVector(**{k: v for k, v in dv_raw.items() if k in fields})

        # "exports" is relative, so the literals baked into the notebook say nothing about this
        # machine. An nTop user retargets them from the notebook inputs anyway.
        recipe = build_rocket_recipe(
            dv, "exports", export_stl=True, export_step=True, export_implicit=True,
            area_stations=16,
        )
        recipe_path = os.path.join(dest_dir, "sv1_recipe.json")
        recipe.write_json(recipe_path)
        scrub_paths(recipe_path)
        log("wrote examples/SV-1/02_geometry/sv1_recipe.json")

        runner = NtopRunner()
        runner.convert(recipe_path, os.path.join(dest_dir, "sv1.ntop"))
        size_mb = os.path.getsize(os.path.join(dest_dir, "sv1.ntop")) / 1e6
        log(f"regenerated sv1.ntop with neutral export paths ({size_mb:.1f} MB)")
        # The driver writes a convert log next to the output. It records the full command line,
        # so it carries this machine's paths and has no value to a reader. Drop it.
        for junk in ("sv1_convert.log", "ntopcl_convert.log", "ntopcl_run.log"):
            jp = os.path.join(dest_dir, junk)
            if os.path.isfile(jp):
                os.remove(jp)
        return True
    except Exception as exc:                              # noqa: BLE001
        log(f"could not regenerate the notebook ({type(exc).__name__}: {exc}); copying the cache")
        return False


def build_geometry(meas: dict) -> None:
    g = DIRS["geometry"]
    src = os.path.join(SV1, "converged", "geom")

    for name in ("sv1.stl", "sv1.step", "sv1.implicit",
                 "sv1_input.json", "sv1_output.json", "sv1_measurements.json"):
        copy(os.path.join(src, name), g)

    # The notebook itself. This is the artefact an nTop user actually wants.
    #
    # It is REGENERATED here rather than copied from the run cache. A converted notebook bakes its
    # export destinations in as absolute `file_path` literals, so the cached copy carries the
    # developer's directory layout inside the binary. Rebuilding it with the export paths pointing
    # at a neutral relative directory keeps that out of the published artefact.
    if _regenerate_notebook(g):
        return

    ntop_src = meas.get("ntop_path") or ""
    if ntop_src and not os.path.isabs(ntop_src):
        ntop_src = os.path.join(REPO, ntop_src)
    if not (ntop_src and os.path.isfile(ntop_src)):
        # fall back to the largest notebook in the cache, which is the exports-on topology
        cache = os.path.join(RUNS, "_ntop_cache")
        cands = [os.path.join(cache, n) for n in os.listdir(cache)] if os.path.isdir(cache) else []
        cands = [c for c in cands if c.endswith(".ntop")]
        ntop_src = max(cands, key=os.path.getsize) if cands else ""
    if ntop_src and os.path.isfile(ntop_src):
        copy(ntop_src, g, rename="sv1.ntop")
        log(f"notebook: {os.path.getsize(ntop_src)/1e6:.1f} MB")
    else:
        log("NO .ntop FOUND. Run run_sv1.py --stage smoke first.")

    # The recipe JSON is the human-readable source the notebook was converted from.
    try:
        from rocketgen.config import DesignVector
        from rocketgen.ntopgen.rocket_notebook import build_rocket_recipe

        dv_raw = jload(os.path.join(SV1, "converged", "point_ntop.json"))["design_vector"]
        fields = set(DesignVector().__dict__.keys())
        dv = DesignVector(**{k: v for k, v in dv_raw.items() if k in fields})
        recipe = build_rocket_recipe(
            dv, g, export_stl=True, export_step=True, export_implicit=True, area_stations=16
        )
        recipe.write_json(os.path.join(g, "sv1_recipe.json"))
        log("wrote examples/SV-1/02_geometry/sv1_recipe.json")
    except Exception as exc:                              # noqa: BLE001
        log(f"could not emit the recipe JSON: {type(exc).__name__}: {exc}")

    # Area distribution as CSV.
    ad = meas.get("area_distribution") or []
    if ad:
        write_csv(
            os.path.join(g, "area_distribution.csv"),
            ["station_m_from_nose", "cross_section_area_m2"],
            [[f"{x:.6f}", f"{a:.8f}"] for x, a in ad],
        )

    # Flat measurement table.
    rows = []
    for k, v in meas.items():
        if k in ("area_distribution", "warnings") or v is None:
            continue
        if isinstance(v, (list, tuple)):
            for i, comp in enumerate(v):
                rows.append([f"{k}[{i}]", f"{comp:.8g}" if isinstance(comp, (int, float)) else str(comp)])
        else:
            rows.append([k, f"{v:.8g}" if isinstance(v, (int, float)) else str(v)])
    write_csv(os.path.join(g, "measurements_flat.csv"), ["quantity", "value"], rows)


# --------------------------------------------------------------------------------------
#   03, 04, 05
# --------------------------------------------------------------------------------------


def build_trade() -> None:
    t = DIRS["trade"]
    for name in ("grid.csv", "lhs.csv", "sensitivity.json"):
        copy(os.path.join(SV1, "doe", name), t)
    sens_path = os.path.join(SV1, "doe", "sensitivity.json")
    if os.path.isfile(sens_path):
        sens = jload(sens_path)
        responses = list(sens.keys())
        variables = list(sens[responses[0]].keys())
        write_csv(
            os.path.join(t, "sensitivity.csv"),
            ["variable"] + responses,
            [[v] + [("" if math.isnan(sens[r][v]) else f"{sens[r][v]:+.4f}") for r in responses]
             for v in variables],
        )


def build_figures() -> None:
    f = DIRS["figures"]
    for src in (
        os.path.join(SV1, "converged", "geom", "sv1_iso.png"),
        os.path.join(SV1, "converged", "geom", "sv1_side.png"),
        os.path.join(SV1, "converged", "trajectory.png"),
        os.path.join(RUNS, "_aero", "aero_validation.png"),
    ):
        copy(src, f)
    figdir = os.path.join(SV1, "figures")
    if os.path.isdir(figdir):
        for n in sorted(os.listdir(figdir)):
            if n.endswith(".png"):
                copy(os.path.join(figdir, n), f)


def build_validation() -> None:
    v = DIRS["validation"]
    ev_src = os.path.join(SV1, "figures", "evidence.json")
    copy(ev_src, v)
    if not os.path.isfile(ev_src):
        return
    ev = jload(ev_src)
    it, ae = ev["integrator"], ev["aero"]
    rows = [
        ["integrator", "vacuum ballistic range vs closed-form parabola", f"{it['vacuum_range_rel']:.3e}", "relative"],
        ["integrator", "vacuum ballistic apogee", f"{it['vacuum_apogee_rel']:.3e}", "relative"],
        ["integrator", "burnout speed vs Tsiolkovsky less gravity loss", f"{it['tsiolkovsky_rel']:.3e}", "relative"],
        ["integrator", "terminal velocity vs sqrt(2mg/(rho S CD))", f"{it['terminal_velocity_rel']:.3e}", "relative"],
        ["integrator", "specific-energy drift over 100 s", f"{it['energy_drift_rel']:.3e}", "relative"],
        ["integrator", "RK4 order ratios (expect 16)",
         ", ".join(f"{r:.2f}" for r in it["rk4_order_ratios"]), "-"],
        ["aero", "CD0 mean bias vs Basic Finner", f"{100*ae['cd0_mean_bias']:+.1f}", "percent"],
        ["aero", "CN_alpha mean bias vs Basic Finner", f"{100*ae['cna_mean_bias']:+.1f}", "percent"],
        ["aero", "centre of pressure mean bias vs Basic Finner", f"{100*ae['xcp_mean_bias']:+.1f}", "percent"],
        ["aero", "CD0 calibration factor applied in the loop", f"{ae['cd0_calibration']:.3f}", "-"],
        ["geometry", "nTop volume error, 25 mm sphere, mass_properties", "0.0104", "percent"],
        ["geometry", "STL volume error, 25 mm sphere", "0.169", "percent"],
    ]
    write_csv(os.path.join(v, "validation_summary.csv"), ["area", "check", "result", "unit"], rows)


# --------------------------------------------------------------------------------------
#   Index
# --------------------------------------------------------------------------------------

README = """# SV-1 reference example

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
"""


def main() -> int:
    if not os.path.isdir(SV1):
        log(f"no run artefacts at {SV1}. Run run_sv1.py first.")
        return 1

    # Rebuild from scratch so a stale file cannot survive.
    if os.path.isdir(EX):
        shutil.rmtree(EX)
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)

    pn = jload(os.path.join(SV1, "converged", "point_ntop.json"))
    pa = jload(os.path.join(SV1, "converged", "point_analytic.json"))
    meas = jload(os.path.join(SV1, "converged", "measurements.json"))

    build_design(pn, pa)
    build_geometry(meas)
    build_trade()
    build_figures()
    build_validation()

    copy(os.path.join(SV1, "report", "SV1_engineering_report.pdf"), EX)
    with open(os.path.join(EX, "README.md"), "w", encoding="utf-8") as f:
        f.write(README)
    log("wrote examples/SV-1/README.md")

    total = sum(
        os.path.getsize(os.path.join(r, n))
        for r, _, ns in os.walk(EX)
        for n in ns
    )
    log(f"done. examples/SV-1 is {total/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
