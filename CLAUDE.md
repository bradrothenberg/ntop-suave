# CLAUDE.md - agent guide for ntop-suave

You are working in a **coupled conceptual-design toolkit**: SUAVE does the physics, nTop does the
geometry, and the two are wired into a fixed point. The SV-1 rocket under `examples/` is a
reference example, not the purpose of the repo. The purpose is the coupling.

Read this file fully before writing code. Then read `docs/REFERENCE.md`. Both exist so you do not
have to rediscover things that cost hours to find.

---

## 1. Orientation, in one minute

```
rocketgen/
  config.py          THE CONTRACT. Every module imports its dataclasses from here.
  ntopgen/           nTop side: author a notebook, run it, parse what it measured
    universe.py        load the block/type universe, resolve signatures and revisions
    recipe.py          typed builder that emits nTop recipe JSON
    driver.py          ntopcl process driver: convert, templates, run, parse outputs
    rocket_notebook.py the SV-1 parametric notebook (the reference example)
  sizing/            SUAVE side
    atmosphere.py      cached US Standard 1976
    aero.py            component drag and normal-force build-up
    propulsion.py      three-phase solid motor
    trajectory.py      3-DOF RK4 mission integrator
    masses.py          group-weight statement with provenance per line
    loop.py            THE COUPLING. converge_point() and size()
  doe.py             factorial and Latin hypercube trade studies
  report/            scripted figures and the reportlab assembly
run_sv1.py           staged driver: --stage smoke | size | doe | all
scripts/bootstrap.py fetches SUAVE, locates the nTop block universe
```

The data flow is a loop, not a pipeline:

```
  design vector
      |
      v
  [1] masses.build_masses  ---------------------------+
      |                                              |
      v                                              |
  [2] ntopgen.rocket_notebook.measure_rocket        | measured volume, wetted area,
      |   (ntopcl builds the solid and measures it)   | cavity volume, CG, inertia, S(x)
      v                                              |
  [3] sizing.aero.RocketAero  <---------------------+
      |
      v
  [4] sizing.trajectory.Mission.fly
      |
      v
  [5] constraint residuals -> new design vector, back to [1]
```

`sizing/loop.py::converge_point` is that loop. Start there when you need to understand anything.

---

## 2. Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
uv run --python .venv/Scripts/python.exe scripts/bootstrap.py
```

`bootstrap.py` fetches SUAVE and locates the nTop block universe. Neither is committed. Run
`scripts/bootstrap.py --check` any time to see what is and is not working.

**The dependency pins are not negotiable.** SUAVE 2.5.2 breaks on all three current majors:

| Pin | Why |
|---|---|
| `numpy<2` | SUAVE fails on numpy 2.x |
| `scipy<1.14` | SUAVE imports `scipy.integrate.cumtrapz`, removed in 1.14 |
| `setuptools<81` | SUAVE's bundled `pint` imports `pkg_resources`, removed in 81 |

If you "fix" a dependency warning by upgrading one of these, the whole repo stops importing.

Run tests with:

```bash
.venv/Scripts/python.exe -m pytest tests -q
```

The full suite takes about 5 minutes because parts of it drive real `ntopcl` subprocesses.

---

## 3. Hard rules

These are the rules the repo was built under. Keep them.

### 3.1 No invented numbers

Every empirical constant, coefficient and material property carries a source comment **and** an
entry in a module-level `SOURCES` dict passed to `config.register_sources`.

If a value is a guess, the word **`GUESS`** must appear in its source string. There are tests that
assert this. The engineering report prints every flagged entry in a table, so a guess that hides
becomes a guess that ships.

```python
SOURCES = {
    "aero_base_drag": "Fleeman, Tactical Missile Design, 2nd ed., Chapter 2, Figure 2.16",
    "my_new_factor": "GUESS: no source found for this; 0.85 chosen to match the trend",
}
register_sources(SOURCES)
```

Never quietly widen a tolerance or nudge a constant to make a test pass. Fix the model, or record
the discrepancy.

### 3.2 Validate against something outside the repo

Each physics module ships a test that reproduces a published or analytically-known case to a
stated tolerance. Existing precedents:

- `tests/test_trajectory.py` checks the integrator against a closed-form vacuum parabola, the
  Tsiolkovsky equation, and analytic terminal velocity, all to machine precision, plus an RK4
  order check.
- `tests/test_aero.py` checks the drag and stability build-up against 23 published Basic Finner
  free-flight shots.
- `tests/test_masses.py` checks the ogive quadrature against an exact hemisphere.

If no reference case exists, **say so explicitly at the top of the test file** and assert only
self-consistency: limits, monotonicity, dimensional correctness. Do not fabricate reference data.

### 3.3 Failures are recorded, never swallowed

A DOE that drops the samples that crashed reports a feasible region that is too large. A loop that
silently falls back to analytic geometry reports a measured result that was never measured.

So: `PointResult.geometry_measured`, `MassBuildup.measured_fraction`, `NtopMeasurements.warnings`
and `TrajectoryResult.message` all exist to carry bad news upward. Populate them. `run_doe` records
a failed sample as a non-converged row rather than skipping it.

### 3.4 SI internally

Metre, kilogram, second, radian, newton, pascal, kelvin. Convert at the boundary only. nTop
literals are already metres and radians, so no conversion is needed there. Report tables convert
for display.

### 3.5 Validate at small scale first

`run_sv1.py --stage smoke` runs the entire pipeline, including one real nTop call, in under a
minute. Use it before anything long. When you scale up, **only the scale parameter changes.**

### 3.6 Style

No emojis. No em dashes: use hyphens, double hyphens or colons. Type hints on public functions.
Match the surrounding comment density, which is high, because the comments carry the engineering
rationale.

Written prose in reports follows ASD-STE100 Simplified Technical English: active voice, simple
tenses, sentences of 20 to 25 words maximum, one idea per sentence.

---

## 4. The nTop side, and its traps

`docs/REFERENCE.md` and `docs/NTOP_NOTES.md` are the accumulated empirical record. Read them before
touching `ntopgen/`. The headline traps:

1. **`.ntop` is a binary container.** It cannot be edited as text. Notebooks are emitted as recipe
   JSON and converted by `ntopcl`. `driver.py` wraps this. `docs/REFERENCE.md` section 5 documents
   the recipe schema and the literal encoding for every type.

2. **Exit code 72 means a block failed. It is NOT success.** Widely-repeated guidance says the
   opposite; it is wrong. A notebook given a negative radius returns 72 and writes nothing. Gate
   success on the expected artefacts existing and being non-empty, and always surface the real
   return code. `driver.py` already does this. Do not "simplify" it.

3. **Conversion evaluates the notebook, exports included.** So a fine mesh tolerance makes
   *conversion itself* cost minutes and gigabytes. `implicit_to_mesh` cost scales as
   `tolerance^-3`. Hence the pattern: **convert once, run many times.** Every design variable is a
   real notebook input, so one `.ntop` serves every design point. `rocket_notebook.py` caches on a
   topology hash and only re-converts when the topology actually changes.

4. **A notebook has exactly one output slot.** Use `Recipe.json_output({...})`, which builds the
   verified list-to-dictionary-to-JSON chain. Then extend `driver.OUTPUT_NAME_MAP` to land the
   values on `NtopMeasurements`. Do not edit `config.py` to add an output.

5. **Input templates report display units, output JSON reports SI.** `-t` returns a 0.025 m default
   as `{"units": "mm", "value": 25.0}`. The driver always writes units explicitly to avoid this.

6. **The block universe drifts from the installed nTop version.** Signatures carry a trailing
   `[maj.min.patch]`; `Universe.latest()` sorts them numerically, not lexically. Some blocks are
   accepted by `ntopcl` but absent from the universe file, and at least one is marked current but
   rejected. `recipe.BLOCK_REVISION_OVERRIDES` and `Recipe.raw_block` are the escape hatches.

7. **Trust the notebook's own measurements over the exported mesh.** On a 25 mm sphere the
   notebook's `mass_properties` was accurate to 0.0104 percent and the STL to 0.169 percent.
   Meshes are for pictures and downstream tools.

When `ntopcl` rejects your JSON, do not guess. Dissect a real notebook:

```bash
ntopcl exportjson some_real_notebook.ntop out.json --ext --dev-blocks-on=True
```

---

## 5. Adding a new vehicle

The SV-1 is one example. To add another:

1. Extend or subclass `config.DesignVector` with the parameters your geometry needs. Add bounds to
   `bounds()`. Add derived quantities as `@property` so there is one source of truth.
2. Write `ntopgen/<vehicle>_notebook.py` exposing `measure_<vehicle>(dv, run_dir) -> NtopMeasurements`.
   That signature is `sizing.loop.GeometryFn`, so the loop takes it with no adapter. Copy
   `rocket_notebook.py`: build the outer mould line as one revolved profile where you can, hollow
   it, subtract the bays, and measure inside nTop rather than in Python.
3. Decide what `NtopMeasurements` fields your geometry can fill, and make `aero.py` prefer them
   over its closed-form fallbacks. Record which values came from nTop.
4. Write the requirements as a `Requirements` instance. **Then check them against each other**
   before trusting any result. See section 7.
5. Run `--stage smoke`, then `--stage size`, then `--stage doe`.

**Critical, learned the hard way:** the structure body you measure in nTop must be the airframe
only. It must not include the motor case, the propellant, the warhead or the avionics, because
`masses.py` charges those separately. Double counting here is silent and large.

---

## 6. Things that will bite you

| Symptom | Cause |
|---|---|
| Everything fails to import | Someone upgraded numpy, scipy or setuptools. See section 2. |
| `ModuleNotFoundError: SUAVE` | `scripts/bootstrap.py` has not run, or `add_suave_to_path()` was not called. |
| Geometry silently analytic | `measure_rocket` raised. Check `PointResult.warnings` and `geometry_measured`. The loop degrades on purpose and says so. |
| `convert` hangs or eats RAM | Mesh or CAD tolerance too fine. Conversion evaluates exports. |
| Trade study reports nothing feasible | Your axes probably do not bracket the converged design. This exact mistake produced an empty feasible region that did not exist. Bracket the answer, do not straddle it. |
| A DOE row is marked not converged for no reason | Check the iteration budget. With analytic geometry only, one iteration IS the fixed point. |
| Launch mass looks light | Some mass group is not in the totals. Use `masses.PROPELLANT_ITEMS` and friends rather than listing item names inline. |
| Motor mass fraction above 0.92 | The bottom-up inert model is incomplete by design. The correlation floor governs and books the shortfall as a visible line item. |

---

## 7. Audit the requirements before trusting a result

This repo exists partly because a sizing loop finds contradictions that inspection misses. In the
reference example, two requirements were mutually exclusive and one was physically impossible:

- Mach 1.50 at sea level **is** 159.6 kPa of dynamic pressure. A 90 kPa structural limit therefore
  capped impact at Mach 1.13. No design could satisfy both.
- An unpowered terminal dive is terminal-velocity limited. Sweeping the dive angle from -25 to -89
  degrees moved impact Mach only from 0.66 to 0.97, and closed form gives 0.935. Mach 1.50 at
  impact was unreachable without thrust in the endgame, for **any** design vector.

When a constraint refuses to close, ask whether it *can* close before you tune the design vector.
Compute the physical bound in closed form. If the requirement is impossible, say so and derive the
bound: that is a more valuable output than a design.

Then lock the finding into the suite. `tests/test_trajectory.py` asserts the unpowered-dive
infeasibility, so it cannot quietly disappear.

---

## 8. Calibration belongs at the boundary

The aero build-up runs about 15 percent low on zero-lift drag against the Basic Finner data. That
is corrected by `config.CD0_CALIBRATION`, applied in `sizing/loop.py` through `CalibratedAero`.

It is applied **at the loop boundary, never inside the aero model.** The model always reports what
its physics gives; the loop owns the correction. Keep that separation. If you add a calibration,
put it in the same place, give it a `SOURCES` entry that states the data it came from, and leave
the quantities you did not validate uncorrected.

---

## 9. What is deliberately out of scope

CFD. Six-degree-of-freedom flight mechanics. Guidance law design. Structural sizing beyond
wall-thickness-times-density plus a hoop-stress check. Energetics and propellant chemistry.
Real-world programme correspondence: the SV-1 requirements are invented for the demonstration.

The nozzle model is ideal: no two-phase loss, no divergence loss, no combustion efficiency, no
throat erosion. Real delivered specific impulse for this class runs 3 to 7 percent lower and that
penalty is **not** applied, because its magnitude could not be sourced. It is the largest known
unquantified optimism in the reference result, and it is documented as such. Do not quietly
"improve" this with a made-up efficiency factor; either source it properly or leave it declared.
