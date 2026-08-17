# PLAN

Read `SPEC.md` for what we are building and `docs/REFERENCE.md` for the verified toolchain facts.

## Work packages

| WP | Owner | Depends on | Deliverable |
|---|---|---|---|
| WP0 | done | - | Feasibility proven, venv built, SUAVE vendored, block universe vendored, `docs/REFERENCE.md` |
| WP1 | agent | WP0 | `rocketgen/ntopgen/` - recipe-JSON builder + `ntopcl` driver + smoke test |
| WP2 | agent | WP0 | `rocketgen/sizing/aero.py` - rocket aero build-up, subsonic to M5, validated |
| WP3 | agent | WP0 | `rocketgen/sizing/propulsion.py` + `trajectory.py` - solid motor + 3-DOF mission |
| WP4 | agent | WP1 | `rocketgen/ntopgen/rocket_notebook.py` - the parametric rocket notebook |
| WP5 | me | WP1-4 | `rocketgen/sizing/loop.py` - the coupled convergence loop + mass build-up |
| WP6 | me | WP5 | Converged SV-1 run + DOE trade study in `runs/` |
| WP7 | agent | WP6 | Engineering report PDF |

WP1, WP2 and WP3 are independent and run in parallel.

## Module layout

```
rocketgen/
  __init__.py
  config.py                 # DesignVector, Requirements, Materials dataclasses; YAML load
  ntopgen/
    __init__.py
    universe.py             # load vendor/functions.json; signature lookup; type resolution
    recipe.py               # Recipe: blocks, variables, inputs, refs, literals -> JSON
    driver.py               # ntopcl convert / template / run; exit-code-72 handling
    rocket_notebook.py     # WP4: build the SV-1 notebook from a DesignVector
  sizing/
    __init__.py
    atmosphere.py           # thin wrapper on SUAVE US_Standard_1976
    aero.py                 # WP2
    propulsion.py           # WP3
    trajectory.py           # WP3
    masses.py               # WP5
    loop.py                 # WP5
  report/
    figures.py
    report.py
runs/<case>/                # generated artefacts
tests/                      # pytest
```

## Hard rules for every work package

1. **Python**: `uv run --python .venv/Scripts/python.exe`, or call the interpreter directly.
   Add `vendor/` to `sys.path` to get SUAVE. Never `pip install suave`.
2. **No invented numbers.** Every empirical constant, coefficient or material property must
   carry a source in a comment, and must be declared in a module-level `SOURCES` dict so the
   report can print it. If a value is a guess, name it a guess in the code and in the dict.
3. **Validate against something.** Each physics module ships a test that reproduces a published
   or analytically-known case to a stated tolerance. If no reference case is available, say so
   explicitly in the test file and assert only on self-consistency (e.g. limits, monotonicity,
   dimensional checks).
4. **SI internally.** Convert at the boundary. nTop literals are metres and radians.
5. **Small-scale first.** Prove the pipeline on a sub-second case before running anything big.
6. **No emojis, no em dashes** in any output, code, comment or document.
7. Tests live in `tests/`, run with `.venv/Scripts/python.exe -m pytest tests -q`.

## Definition of done, per work package

- WP1: `pytest tests/test_ntopgen.py` passes, and a smoke notebook (sphere -> mesh -> STL +
  mass properties) round-trips through `convert`, runs under `ntopcl`, and produces a non-empty
  STL whose measured volume matches the analytic sphere volume to 1 %.
- WP2: `CD0` and `CN_alpha` curves versus Mach for a reference ogive-cylinder-fin body, with a
  validation test and a plot.
- WP3: motor gives the right total impulse for a known grain; trajectory reproduces a vacuum
  ballistic case analytically and a drag-free constant-thrust case analytically.
- WP4: notebook accepts the full design vector as nTop inputs, builds the solid, and outputs
  volume, wetted area, CG, inertia, internal cavity volume, S(x) table, STL, STEP.
- WP5: loop converges on the SV-1 case, all constraints reported.
- WP6: `runs/SV-1/` complete; DOE CSV + plots.
- WP7: PDF in `runs/report/`.
