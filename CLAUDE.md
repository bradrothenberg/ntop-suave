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

There are now TWO reference examples, and the second one is where most of the hard-won knowledge
lives:

| | SV-1 | IV-1 |
|---|---|---|
| Configuration | single body, ogive-cylinder-boattail, cruciform fins | two stages, strakes, interstage, divert pack |
| Mission | air launch, cruise, terminal dive | vertical canister launch, pitchover, staging, lofted intercept |
| Exercises | the coupling itself | mass leaving mid-flight, reference area changing at separation, vortex lift |
| Modules | `config.py`, `sizing/*.py`, `ntopgen/rocket_notebook.py` | `config_iv1.py`, `sizing/*_iv1.py`, `ntopgen/stack_notebook.py` |
| Driver | `run_sv1.py` | `scripts/iv1_converge.py` |

**IV-1 is built as PARALLEL modules, not as changes to the SV-1 ones.** That is deliberate. SV-1 is
the regression baseline: its 296 tests are the contract that says the shared physics still works.
When you add a third vehicle, do the same. Share code by importing or by lifting a function to
module level in the original, never by editing the original's behaviour.

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
.venv/Scripts/python.exe -m pytest -m fast
```

**Use the tiers.** They are applied by module in `tests/conftest.py`, and measured:

| tier | tests | wall clock | needs |
|---|---|---|---|
| `fast` | 297 | about 11 s | SUAVE only |
| `medium` | 267 | about 85 s | SUAVE only |
| `slow` | 126 | about 11 min | SUAVE, a licensed `ntopcl`, the nTop block universe |

`-m fast` is the inner loop. `-m "fast or medium"` is everything CI can run. `-m slow` is the nTop
subprocess tests, which is where the 11 minutes lives.

Two things to know before you touch the tiering:

- **A new test module must be added to `TIERS`** in `tests/conftest.py`, or it is excluded from
  every fast and medium run and CI goes green without executing it. `tests/test_tiers.py` fails if
  you forget.
- **A typo in `-m` used to exit 0** having deselected everything, which in CI is
  indistinguishable from success. `conftest` now rejects an unknown marker and fails a tier that
  selects zero tests. `--strict-markers` does NOT do this; it only checks markers used in code.

CI is `.github/workflows/ci.yml`. It runs fast then medium on Ubuntu, and states in the run summary
that the geometry path was not exercised, because the slow tier cannot run on a hosted runner.

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

### 3.7 The source registry is only complete once everything is imported

`config.SOURCES` is a single global registry. Every module fills it at IMPORT time by calling
`register_sources`. So anything that reads the registry gets whatever happens to have been imported
by then.

This caused a real defect. `sizing/loop.py` imports `propulsion` and `trajectory` LAZILY, inside
`converge_point`. The SV-1 report wrote its provenance file before any trajectory had been flown,
so those two modules had never registered, and the report's limitations table listed **37 sources
with 6 flagged as guesses instead of 70 with 25.** It under-reported its own limitations by a factor
of four, including `prop.ideal_nozzle`, the single largest declared optimism in the result.

So: **import every module that owns sources before you read the registry.** `run_sv1.py::dump_provenance`
and `report/evidence_iv1.py` both do this explicitly, with a comment saying why. Do the same
anywhere you serialise the registry.

A related trap when attributing ownership: because `config.SOURCES` IS the registry, listing
`rocketgen.config` as one of the owning modules attributes EVERY entry to it. Build ownership from
the other modules and let config take the remainder.

### 3.8 A calibration that is not wired in is worse than none

`config.CD0_CALIBRATION` is validated against 23 published free-flight shots and applied through
`loop.CalibratedAero`. SV-1 uses it. **IV-1 does not**, because `scripts/iv1_converge.py` builds
`StackAero` directly and never wraps it. The factor exists, is tested, and is silently absent from
half the results, which is worse than not having it: a reader sees the mechanism in the repo and
assumes it applied.

If you add a vehicle, wire the calibration in and assert it. If you deliberately leave it out, say
so in the result, not only in a comment. This is currently an OPEN defect for IV-1, recorded in
`examples/IV-1/README.md` and section 6 of its report.

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

6. **The block universe drifts from the installed nTop version, and it is missing whole BLOCKS
   and whole TYPES, not merely stale revisions.** Signatures carry a trailing
   `[maj.min.patch]`; `Universe.latest()` sorts them numerically, not lexically. Some blocks are
   accepted by `ntopcl` but absent from the universe file, and at least one is marked current but
   rejected. `recipe.BLOCK_REVISION_OVERRIDES` and `Recipe.raw_block` are the escape hatches.

   **"Not in `functions.json`" is NOT evidence that nTop cannot do something.** This cost real
   work. The spline OML was first built by sampling the spline into a chord polygon, because the
   universe lists no route from a curve to a revolvable profile and that was taken as proof no
   such route existed. It was not proof: nTop revolves splines directly, through four blocks and
   four types that are all absent from the vendored export. See `docs/NTOP_NOTES.md` section 25
   for the chain and the four encoding traps. Check the GUI, or `exportjson` a notebook that
   uses the block, before concluding a capability is missing.

   And do not guess block ids when `convert` rejects one: 27 plausible combinations were tried
   and all failed. `exportjson` a real notebook and read the `func` strings out of it, then
   build the chain up one block at a time so `convert` isolates the offending one.

7. **Trust the notebook's own measurements over the exported mesh.** On a 25 mm sphere the
   notebook's `mass_properties` was accurate to 0.0104 percent and the STL to 0.169 percent.
   Meshes are for pictures and downstream tools.

8. **The cost of `surface_area<implicit,real>` tracks the implicit field's complexity, not the
   body's size.** On IV-1 the four calls on booleaned bodies took 24.6, 24.2, 23.8 and 16.9 s. The
   fifth, on the booster's bare `cylinder` primitive, took **0.27 s for the largest area of the
   five.** Meshing is not what makes a measurement slow. Build primitives as primitives, and if you
   need a measurement cheaper, simplify the field or drop the measurement, not the mesh tolerance.

9. **A plate's real wetted area is not `n * height * length`.** The idealised formula ignores the tip
   face, the two edge faces and the cylindrical root patch the boolean leaves behind. For an 8 mm
   plate 30 mm tall that is 27 percent of the total, so measured area came out at 1.2656 times
   `StrakeSpec.wetted_area`. A test demanding a few percent against the idealised value FAILS on a
   correct model. Write the solid closed form out with the arithmetic shown, and validate against
   that instead.

10. **One notebook reporting several bodies must namespace its output keys**, and then
    `driver.parse_outputs` is no longer safe to call on it. `stack_notebook` emits `s1_`, `s2_`,
    `is_` and `st_` prefixes; `s1_volume_total` and `s2_volume_total` would both map onto
    `volume_total` and the flat result would be meaningless. Use its `measure_stack`, which returns
    a dict keyed by stage.

11. **State which frame a measured station is in.** `stack_notebook` reports `cg_structure` in
    STAGE-LOCAL coordinates, because `masses_iv1` adds the stage offset itself, and
    `cg_structure_stack` in stack coordinates. Getting this wrong moves a centre of gravity by
    metres and nothing crashes.

12. **A symmetric body does not measure as exactly symmetric.** A cruciform vehicle came back with
    its centre of gravity 1.2 mm off axis. That is discretisation. Test against a tolerance, never
    against zero.

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

### 5.1 If the vehicle stages

Three things change, and each one is a place to get it wrong quietly:

1. **Mass leaves.** The jettisoned total must be exact, because the integrator subtracts it in one
   step and any error becomes a velocity error forever after. Gate it with a mass-bookkeeping test
   that asserts `m(t) == m0 - burned(t) - jettisoned(t)` at every step to machine precision. Make
   every event time a hard step boundary, or the step containing a burnout books only part of the
   propellant burned in it and machine precision is unreachable.
2. **The reference area changes.** A coefficient computed on the wrong area is silently wrong by the
   diameter ratio squared. `StackAero.evaluate` takes a `stage` argument that selects the FLIGHT
   CONFIGURATION, not a component: `stage=1` is the full stack on the booster area, `stage=2` is the
   surviving stage on its own.
3. **Stability must be checked twice**, once per configuration, because the centre of gravity and
   the centre of pressure both move at separation.

Prove code reuse with an **exact degeneracy test**. `MultiStageMotor` with one stage reproduces
`SolidMotor` bit for bit, asserted with `==` rather than `approx`, thrust identical to 0.0 N over
20001 samples. That is the strongest available evidence that you generalised the validated physics
instead of writing it again, and it is cheap. Where the two legitimately differ, say exactly why and
claim no tolerance there.

### 5.2 Enumerate the requirements against the constraint list

Write the requirements down, then assert programmatically that every one of them appears in the
constraint list the loop actually checks. IV-1 shipped with two holes in that list:

- Grain closure was not gated at all, so a stage passed at 134 percent volumetric loading with a web
  wider than the bay radius. Found during sizing, fixed.
- A9 static margin was never added, and when it was finally evaluated for the report it **failed at
  -0.947 calibres.** Still open.

A constraint that is not in the list is not checked, and "all constraints met" then means something
much weaker than it sounds. Say how many were checked.

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
| The limitations table looks short | The registry was read before every module was imported. See section 3.7. |
| A result you produced is not in the repo | `runs/` is gitignored. Nothing under it ships. Curate into `examples/` or it does not exist to anyone else. See section 11. |
| A test asserts a specific altitude ceiling, table size or tolerance value | It is pinning an implementation detail. When the detail legitimately changes, fix the test to read the module, and say in the test why. Two tests pinned the old 30 km atmosphere ceiling. |
| Atmospheric values disagree with the published tables by a few percent | US Standard 1976 is tabulated against GEOPOTENTIAL altitude; SUAVE takes GEOMETRIC. Convert before comparing: `H = r0*z/(r0+z)` with `r0 = 6356766` m. 47 km geometric is 46.655 km geopotential. This looked like a 4 percent model defect and was a comparison error. |
| A cached-table lookup is unexpectedly slow | `np.interp` binary-searches per field per call. If the grid is uniform, use index arithmetic: it is exact, and it took the atmosphere lookup from 95 us back to 3.5 us. |
| A path-scrubbing or text-rewriting script reports success but the strings are still there | Shell escaping mangled the pattern. Verify with `grep`, not with the script's own message. This defeated two attempts in a row. |
| `ntopcl convert` says only `Error loading recipe:` with no detail | A block id or a literal encoding is wrong, and it will not tell you which. Build the graph up one block at a time, converting after each; the first failure names the culprit. |
| A geometry approach looks impossible because the block does not exist | The vendored universe is incomplete. See section 4 point 6. |
| A wall-clock timing test fails intermittently | Concurrent agents saturate the CPU. Confirm in isolation before believing it. |

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

IV-1 then produced a third and more interesting kind of contradiction. Three requirements were
mutually exclusive: 100 miles of slant range, a 15 km minimum intercept altitude, and 15 g of
lateral acceleration. Aerodynamic lateral acceleration needs dynamic pressure, so 15 g is
unavailable above about 14 km at Mach 4, while 100 miles of range forces the intercept above 20 km.
The best point satisfying all three was 36.2 miles, short by a factor of 2.8.

**The cause was an exclusion in the specification, not a bad design vector.** SPEC section 8 had
ruled out attitude-control thrusters. Vehicles of this class carry them for exactly this reason.
So when a requirement will not close, also ask whether something the spec forbade is the thing that
would close it. That is a different question from "is this requirement physically possible", and it
found the answer here.

A fourth pattern worth knowing: **a modelling choice can become the dominant design driver.**
Restricting IV-1's grains to a tubular geometry capped booster thrust, pushed a nozzle exit outside
the body, and held stage-2 propellant to 90 kg. Real motors use finocyl or star grains. Because the
model refused to paper over it with an unsourced shape factor, the restriction surfaced as a visible
design constraint instead of quiet optimism. Keep that behaviour.

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

---

## 10. Physics validation that is already banked

Do not redo these, and do not weaken them. If a change makes one fail, the change is wrong until
proven otherwise.

| What | Against | Achieved |
|---|---|---|
| Trajectory integrator | closed-form vacuum parabola, apogee and range | 2e-14 relative |
| Burnout speed | Tsiolkovsky less gravity loss | 1.7e-13 |
| Two-stage burnout speed | staged Tsiolkovsky across the jettison | 6.7e-13 |
| Terminal velocity | `sqrt(2mg/(rho S CD))` | 8.7e-10 |
| Specific energy, 100 s, no thrust or drag | conservation | 5.3e-15 |
| Mass bookkeeping through staging | `m0 - burned - jettisoned` | 7.5e-15 |
| RK4 order, step halved | factor of 16 | 15.93, 16.01, 15.76 |
| Nozzle thrust coefficient | published isentropic tables, gamma 1.2 and 1.4 | better than 0.1 percent |
| Drag, normal force, centre of pressure | 23 Basic Finner free-flight shots, DREV-TM-9703 | -14.6, -10.7, +2.0 percent mean bias |
| Strake vortex lift | NASA TN D-7921 and TM X-3130 | model is conservative by 1.4 to 3.4x |
| Atmosphere to 86 km | US Standard 1976 layer breaks | 0.5 percent |
| nTop volume, 25 mm sphere | analytic | 0.0104 percent from `mass_properties`, 0.169 from the STL |
| nTop volume, per stage | independent closed form | -0.008 and -0.002 percent |

Two are worth reading before you touch the aero or the atmosphere. The Basic Finner comparison is
where `CD0_CALIBRATION` comes from. The atmosphere comparison is where the geopotential trap in
section 6 was found.

An RK4 order check needs a case whose exact solution is NOT a low-order polynomial. A drag-free
vertical climb has constant acceleration, so RK4 integrates it exactly and the order ratio is
meaningless. Use the 45 degree vacuum parabola, where the `gamma_dot` term is active.

---

## 11. Results only ship if they are curated

`runs/` is gitignored. Everything the analysis writes lands there, so **nothing the analysis
produces is in the repository unless you copy it into `examples/`.** This bit twice: the IV-1
notebook and the IV-1 report were both finished, on disk, and absent from the branch.

`scripts/build_example.py` is the pattern for SV-1. What it does is worth copying:

- Flatten the JSON into CSV as well, because a reader opens a spreadsheet before they open a JSON.
- Number the directories so the reading order is obvious.
- Write a README that says what every file is.
- **Regenerate the `.ntop` rather than copying the cached one.** A converted notebook bakes its
  export destinations in as absolute `file_path` literals, so the cached copy carries the developer's
  directory layout inside the binary. Rebuild it with a relative export directory.
- Scrub developer absolute paths out of every text artefact on the way in, and delete the `ntopcl`
  convert log, which records the full command line.
- **Verify the scrub with `grep`, not with the script's own success message.** See section 6.

Keep the committed set small and deliberate. The full `runs/` tree for two vehicles is over 100 MB
of probe output; the curated examples are 20 to 30 MB and carry the notebook, the CAD, the figures,
the report and the machine-readable record.
