# nTop notes - added by WP1

Everything here was verified by running `ntopcl` on this machine on 2026-08-17. It extends
`docs/REFERENCE.md`; it does not replace it. `REFERENCE.md` is the WP0 baseline and must not be
edited.

Builds used:

| Path | Version |
|---|---|
| `$NTOPCL` (a locally built nTop, if you have one) | nTop 5.54.0 (dev, default) |
| `C:/Program Files/nTopology/nTopology/ntopcl.exe` | nTop 5.53.2 (installed) |

## 1. The notebook OUTPUT is a single top-level key

A recipe declares its Automate output with a top-level key, not a per-block flag:

```json
"output": { "id": "inst107" }
```

`id` is the id of a **root entry** (a variable is what a GUI notebook uses). Findings:

- **Exactly one output.** `"output": [{"id": ...}, {"id": ...}]` is rejected by
  `ntopcl convert` with `[E]: Error loading recipe:` and exit code 1. A recipe therefore has
  one output slot. To report many quantities, make that one output a composite value.
- Without `output`, `ntopcl -t` prints
  `[E]: Error generating output template : Output of function not set`, exits **0**, and
  writes `input_template.json` but no `output_template.json`. Neither
  `a real 360-block reference notebook` nor `voxel_grid_from_mesh.ntop` designates an output, which
  is why `-t` on them produces no output template.
- `output` survives `exportjson` round-trips (with the ids renumbered).

Production notebooks that DO have an output use it as a single scalar
(`ntop-ai-codev-demo/Model 1/surrogate/CD surrogate.ntop` -> `real` named "CD") or as a single
`json` value carrying a dictionary (`D:/nTop/DOERunner/gcp_10K_v4/nTopGrp3_v3.ntop` -> `json`
named "JSON Out").

## 2. Where `ntopcl -t` writes the templates

`ntopcl -t <notebook.ntop>` writes `input_template.json` and `output_template.json` into the
**process working directory**, with those exact fixed names. NOT next to the notebook, and
there is no flag to choose the location. `NtopRunner.templates()` therefore sets `cwd`.

### Input template schema

```json
{
  "description": "<notebook description>",
  "inputs": [ {"description": "", "name": "Radius", "type": "real",
               "units": "mm", "value": 25.0} ],
  "title": "<notebook displayname>"
}
```

`units` are **display** units and `value` is converted into them. A recipe input declared with
`dimension: {"length": 1}` and a default of 0.025 (metres) comes back as `"units": "mm"` and
`"value": 25.0`. An angle comes back as `"deg"`.

### Input JSON for `-j`

The same shape, minus `title`/`description`:

```json
{"inputs": [{"name": "Radius", "type": "real", "units": "m", "value": 0.025}]}
```

An explicit `"units"` string IS honoured. Verified: `{"units": "m", "value": 0.025}` and
`{"units": "mm", "value": 25.0}` both produce a 25 mm sphere. Omitting `units` falls back to
the template's display unit, so the driver always writes `units` explicitly.

### Output JSON

A top-level **list**:

```json
[ {"components": [], "name": "Volume", "type": "real",
   "value": {"isFinite": true, "units": {"length": 3}, "val": 6.544303600204088e-05}} ]
```

`value` uses the same literal encodings as a recipe. A `json`-typed output appears as
`{"jsonObject": { ... }}`. Values are in SI base units (metres, m^3, radians), not display
units, which is the opposite of the input side.

## 3. `convert` EVALUATES the notebook

`ntopcl convert` does not just serialise: it builds and runs every block, including
side-effecting export blocks. A recipe that exports an STL leaves the STL on disk after
`convert` alone. Consequences:

- A slow or divergent block makes `convert` slow or hang. A `convert` of a sphere meshed at a
  1.0e-4 m tolerance had not finished after 195 s and had grown past 2.4 GB resident, with no
  diagnostic output at all on stdout or stderr.
- Always put a timeout on `convert`, and treat "no output plus exit code 1" as a real failure.
- `convert` failures print `[E]: Error loading recipe:` and nothing else useful. Bisect by
  building the notebook one block at a time (see `runs/_smoke/_bisect.py`).

## 4. `implicit_to_mesh` cost scales as tolerance^-3

It drives a voxel grid, so halving the tolerance multiplies cost by about 8. Measured for a
25 mm-radius sphere on this machine:

| Tolerance | `convert` wall time | Triangles | STL volume error |
|---|---|---|---|
| 1.0e-3 m | 3 s | 9428 | 0.169 % |
| 1.0e-4 m | did not finish in 195 s, > 2.4 GB | - | - |

1.0e-3 m is the working point for the WP1 smoke test. WP4 should size the tolerance from the
part's thinnest feature, not from a habit, and should expect an ogive-cylinder rocket at
0.5 mm tolerance to be far more expensive than a sphere.

The block's own `mass_properties` is much more accurate than the exported mesh: it reported the
sphere volume to 0.0104 %, versus 0.169 % for the STL at the same tolerance. Feed SUAVE the
notebook's measured values, not values re-derived from the STL.

## 5. The vendored block universe is stale

`vendor/functions.json` is dated Aug 8 2026, and both installed builds already have blocks it
does not list. Two separate problems:

1. **Missing revisions.** `implicit_to_mesh<implicit,real,real,bool,bool>[2.5.0]` and
   `implicit_to_mesh<implicit,real,real,integer,implicit,bool>[2.5.0]` both exist in 5.53.2 and
   5.54.0. The vendored file stops at `[2.4.0]` and marks it current, but both builds log:
   `[W]: Mesh from Implicit Body: Mesh from Implicit Body 2.4.0 is deprecated due to a bug that
   removed features larger than Min. Feature Size.` `recipe.BLOCK_REVISION_OVERRIDES` records
   the newer ids; `Recipe.latest()` prefers them and `Recipe.describe()` arity-checks an unknown
   revision against the newest known one of the same base signature.
   `[3.0.0]` is rejected, so `[2.5.0]` really is the newest.
   `runs/_smoke/_probe_revisions.py` re-runs this survey; it found no newer revision for
   `mass_properties`, `surface_area`, `export_mesh`, `export_part`, `export_implicit_body`,
   `export_table`, `export_text`, `cad_body_from_implicit_body`, the boolean blocks, `revolve`,
   `profile_from_points`, `shell` or `thicken_implicit`.
2. **Missing blocks entirely.** Several blocks used by production notebooks are absent from
   the vendored file but accepted by `ntopcl convert`. Verified accepted:

   | Block id | Returns |
   |---|---|
   | `core.list<T>` | `list<T>` |
   | `json_from_dictionary<dictionary<text,real>>[5.30.0]` | `json` |
   | `table_from_columns<list<column>>` | `table` |
   | `column_from_text<list<text>,text>` | `column` |
   | `await<any,any>` | follows input |

   Verified **rejected**, so the revision suffix and exact overload matter:
   `json_from_dictionary<dictionary<text,real>>` without `[5.30.0]`, and
   `column_from_scalars<list<real>,text>`.

   `Recipe.raw_block()` is the way to emit any of these. `Recipe.list_of` and
   `Recipe.point_list` already special-case `core.list<T>`.

Treat `Universe` as a checking aid, not as the authority. `ntopcl convert` is the authority.

## 6. Property access: the `props` chain

A ref selects a property of the value it points at:

```json
{"props": ["volume"], "ref": {"id": "inst106"}}
{"props": ["components", "[0]", "instance", "bodies", "[0]"], "ref": {"id": "inst1820"}}
{"input": 0, "props": ["x"]}
```

Property names come from `vendor/types.json` (`Universe.properties(type_name)`) and are the
GUI's display names, spaces and all: `"center of gravity"`, `"principal moments"`,
`"face count"`. A list index is the string `"[N]"`.

`body_mass_props` exposes `center of gravity` (point), `mass` (real), `principal frame`
(frame), `principal moments` (vector) and `volume` (real). There is no block that extracts
them, so props are the only way.

A variable's `contents` may itself be a props ref, which is how a scalar is pulled out into its
own named notebook value:

```json
{"contents": {"props": ["volume"], "ref": {"id": "inst106"}},
 "id": "inst107", "name": "Volume", "type": "real", "variable": true}
```

An inline literal cannot carry props. Wrap it in a variable first.

## 7. Exit codes actually observed - and why 72 does NOT mean success

REFERENCE.md section 6 says `ntopcl` returns 72 on success in some configurations. That is
half the story. **72 was observed on a FAILING run.** Feeding the smoke notebook a negative
radius makes the sphere block error out, no output JSON is written, and `ntopcl` exits 72:

```
[E]: Ball: The value of input 'Radius' is out of range.  Please provide a value that is
     greater than 0.
[E]: Fatal error: JSON Out: Output block is unbuilt
[I]: nTop exited with errors.
-> exit 72, no output.json
```

So the return code alone cannot decide success in either direction. Gate on the expected
artefacts existing and being non-empty, always. `NtopRunner` does exactly that, and
`RunResult.returncode` always carries the real code.

Full table of what was seen on this machine:

| Invocation | Build | Code | Notes |
|---|---|---|---|
| `convert` success | 5.54.0 | 0 | |
| `convert` failure: unknown block, bad recipe, multi-output | 5.54.0 | 1 | `[E]: Error loading recipe:` |
| `convert` killed mid-run | 5.54.0 | 1 | no diagnostics at all |
| `-t`, output designated and buildable | 5.54.0 | 0 | both templates written |
| `-t`, no output designated | 5.54.0 | 0 | `Error generating output template : Output of function not set`, no output template |
| `-t`, output designated but unbuildable | 5.54.0 | 1 | `Error generating output template : Output of function not built` |
| `-j -o` run success | 5.54.0 | 0 | |
| `-j -o` run success | 5.53.2 | 0 | |
| `-j -o` run, a block errored | 5.54.0 | **72** | no output JSON written |
| `-j -o` run, input JSON has no `inputs` | 5.54.0 | 1 | `JSON file has no inputs` |
| `exportjson` success | 5.54.0 | 0 | |

## 8. Notebooks are portable between builds

A notebook produced by `convert` on the 5.54.0 dev build runs unchanged on the installed
5.53.2 build, exit code 0, same numbers. Useful if the dev build is unavailable.

## 9. STEP export works headless, via `brep.part`

There is no `implicit -> part` block and `export_part` will not take a `brep`. The working
chain is:

```
cad_body_from_implicit_body<implicit,real,list<brep>>   ->  brep
   .prop("part")                                       ->  part
export_part<file_path,part>[2.0.0]
```

Verified: a 25 mm sphere at a 5.0e-4 m CAD tolerance converted in 11.6 s and produced a
450 KB `.step`. No extra environment setup was needed on Windows (on Linux, STEP export needs
the CAD-interop environment; see the `run-ntop-automate` skill).

`Recipe.export_step` handles the `export_part<file_path,part>` versus
`export_part<file_path,list<part>>` choice from the ref's type.

## 10. `core.list<real>` demands identical units across its elements

Putting a volume and a mass in one `list<real>` fails at run time:

```
[E]: Measurement Values: The units of inputs '0' (units of length^3) and '1' (units of mass)
     do not match. Please ensure that both input units match to run this block.
[E]: Fatal error: JSON Out: Output block is unbuilt
```

The fix is to divide each quantity by a literal 1 of its own unit first, making every element
dimensionless. Because nTop stores SI internally, the resulting numbers are already SI.
`Recipe.dimensionless(ref, units)` does this, and `Recipe.json_output` applies it
automatically when an entry is given as a `(ref, units)` pair.

## 11. `surface_area<implicit,real>` warns on implicit input

Running it on a plain sphere logs:

```
[W]: Area: Input Body's field contains undefined regions with NaN values. The surface area may
     not be accurate. Convert the body to a mesh with the Mesh from Implicit Body block then
     use the Surface Area from Mesh block for a more accurate result.
```

It is still accurate on a sphere (0.0097 % against the analytic area), but nTop's own advice is
to mesh first and use `surface_area<mesh>`. `Recipe.surface_area` already dispatches to
`surface_area<mesh>` when handed a mesh ref, so pass it a mesh for wetted-area work.

## 12. Smaller things

- Root entries in a GUI-authored notebook are overwhelmingly `variable` wrappers (345 of 360 in
  `a real 360-block reference notebook`). Only a `variable` (or any root entry) can be the target of
  a `ref`. `recipe.py` therefore wraps every authored block in a root variable, which also
  makes the generated notebook readable in the GUI.
- Block ids are renumbered by `convert`, so never compare ids across a round trip. Compare by
  `name`.
- `sphere<point,real>` returns type `sphere`, not `implicit`. Likewise `box`, `cube`. These are
  accepted anywhere an `implicit` is wanted. `Recipe` dispatches on `Ref.type`, so an emitter
  that switches on `== "implicit"` would be wrong; check for `"mesh"` instead and let
  everything else take the implicit path.
- An input's required units come from `unitsReq`, which may be an int (meaning "same as input
  N"), a dimension map, or a full unit spec with a `dimension` sub-key (seen on
  `modal_simulation<...>`).
- `outputBaseUnits` and a property's `baseUnits` are usually a dimension map but are sometimes
  a LIST of maps (e.g. `core.dictionary<...>`, `bind<...>`). Do not assume a dict.
- `export_mesh`'s `unit_length_enum` is the unit the STL is WRITTEN in. `{"id": "m"}` gives an
  STL in metres, `{"id": "mm"}` the same mesh in millimetres. The enum uses the `{"id": ...}`
  encoding, not `{"enum": N}`.
- `real` literals are accepted in `real_field` slots; nTop auto-converts. That is how a density
  is passed to `mass_properties<implicit,real_field,real>`.
- A `file_path` literal wants forward slashes on Windows. `recipe.to_ntop_path` normalises.

## 13. What WP4 must know

1. **One output.** Use `Recipe.json_output({...})`. It builds this chain, which was verified
   end to end (convert, `-t`, run, parse):

   ```
   core.list<text>                                     -> list<text>   (the names)
   core.list<real>                                     -> list<real>   (dimensionless values)
   core.dictionary<list<text>,list<real>>              -> dictionary<text,real>
   json_from_dictionary<dictionary<text,real>>[5.30.0] -> json         <- the output
   ```

   None of those four blocks are in `vendor/functions.json`. `json_output` emits them with
   `raw_block` and divides each value by 1 of its own unit (section 10).
   `driver.parse_outputs` unpacks the resulting `jsonObject` keys as if they were separate
   outputs, so `{"volume_total": ..., "area_wetted_body": ...}` maps straight onto
   `NtopMeasurements`. Usage:

   ```python
   r.json_output({
       "volume_total":     (props.prop("volume"), {"length": 3}),
       "mass_structure":   (props.prop("mass"),   {"mass": 1}),
       "area_wetted_body": (area,                 {"length": 2}),
   })
   ```

   An alternative for tabular data is a `table` written to CSV with `export_table`, with a
   scalar as the declared output.
2. **Vectors do not fit the json output.** `center of gravity` is a `point` and
   `principal moments` is a `vector`. Split them into components first. Both `point` and
   `vector` expose `x`, `y`, `z` as `real` properties, so a chained props ref works and was
   verified to run:

   ```python
   cg = props.prop("center of gravity")
   r.json_output({"cg_x": (cg.prop("x"), {"length": 1}), ...})
   # -> {"cg_x": 0.10000078341947423, ...} for a sphere centred at (0.1, 0.2, 0.3)
   ```

   `principal moments` components carry `{"length": 2, "mass": 1}`.
3. **Extend `driver.OUTPUT_NAME_MAP`**, do not edit `rocketgen/config.py`. Names are matched
   case-insensitively with spaces, dots and dashes folded to underscores.
4. Mesh tolerance drives the cost (section 4). Budget it. Prefer the notebook's own
   `mass_properties` over anything measured off the STL: 0.0104 % versus 0.169 % on the smoke
   sphere.
5. `convert` runs the notebook (section 3), so a heavy notebook costs its full run time twice:
   once at convert, once per run. Convert once, then run many times with different inputs.
6. The area-distribution S(x) that SPEC.md section 6 needs has no single block. Candidates in
   the universe: `surface_area<implicit_2d,real>[1.2.0]` on a section profile, or a
   `ray_cast<implicit,point,vector,real,real,integer>[5.30.0]` sweep. Needs its own probe.
7. STEP export needs the `brep.part` chain in section 9; `Recipe.export_step` expects a `part`.

---

# nTop notes - added by WP4

Everything below was measured by running `ntopcl` (nTop 5.54.0 dev build, this machine) while
building `rocketgen/ntopgen/rocket_notebook.py`. It extends the WP1 sections above and does
not replace them. The probes that produced these numbers are kept in
`runs/SV-1_geom/_probe.py`, `_probe_fin.py`, `_probe_area_cost.py` and `_probe_sx.py`.

## 14. `profile_from_points` returns a `profile`, not an `implicit_2d`

`vendor/functions.json` says `profile_from_points<list<point>>` returns `profile`, whose
display name is "Polygon". `revolve`, `extrude` and `loft` all want an `implicit_2d`, whose
display name is "Profile". They are different types and there is no conversion block. The
bridge is a props chain: `types.json` gives `profile` a property `profile: implicit_2d`, so

```python
poly = r.block("profile_from_points<list<point>>", point_list)
profile_2d = r.variable("OML Profile", poly.prop("profile"))     # -> implicit_2d
```

Verified end to end. Both blocks log a deprecation warning naming a `5.20.0` revision
("Polygon from Points 1.0.0 is deprecated. Use Profile from Points 5.20.0 instead",
"Revolve Profile 1.0.0 is deprecated. Use Revolve Profile 5.20.0 instead"). The vendored
universe has neither, and the 1.0.0 blocks measure correctly, so WP4 stayed on them. A later
work package could try `profile_from_points<list<point>>[5.20.0]`.

## 15. One closed polygon revolved 360 degrees is the cheapest exact body of revolution

The whole SV-1 outer mould line - tangent-ogive nose, cylindrical mid-body, conical boattail,
flat base - is ONE closed polygon in the XY plane revolved about the X axis. No booleans, one
implicit body, and the measured volume matches the chord-polygon prediction exactly:

| Ogive chord segments | Predicted OML volume error | Measured `mass_properties` volume |
|---|---|---|
| 24 | -0.0117 % | 0.3354646 m^3 against a closed-form 0.3355094 m^3, i.e. -0.013 % |

The agreement confirms that the revolved solid really is the frustum sum of the chord polygon,
and that nTop adds nothing measurable of its own. Wetted area came out -0.224 % against the
closed form.

Doing the profile arithmetic INSIDE nTop is what keeps the notebook parametric. Writing the
tangent ogive as `y/R = sqrt(c^2 - k^2 (1-u)^2) - (c-1)` with `k = L/R` and `c = (1+k^2)/2`
makes everything except the final multiply by `R` dimensionless, so `sqrt<real>` never has to
take the root of a length squared, and `k`, `c^2` and `c-1` are shared across all samples. That
is 7 blocks per sample point. A 24-segment ogive plus fins, cavity, bulkheads, measurements and
exports is 287 root entries and 182 KB of recipe JSON, which `convert` handles in about 30 s.

## 16. `loft<implicit_2d,implicit_2d>` is a FIELD MIX, not a boundary loft. Never measure it.

This one cost real time. The fins were first built as proper symmetric double wedges, by
lofting a root diamond profile to a tip diamond profile. It converts, it runs, and it is
**wrong**:

| Quantity | Lofted double wedge | Exact | Ratio |
|---|---|---|---|
| One panel volume | 2.747e-4 m^3 | 3.343e-4 m^3 | 0.82 |
| Four-panel volume | 1.0978e-3 m^3 | 1.3374e-3 m^3 | 0.82 |
| Exposed wetted area | 0.3116 m^2 | 0.4385 m^2 | 0.73 |

nTop says why itself, in a warning that is easy to skim past:

```
[W]: Loft between Profiles: Loft between Profiles 1.1.0 is deprecated. Please use two
     Extrude Profile blocks, a Mix block, and a Ramp block to achieve similar results.
```

The block linearly interpolates the two profiles' signed-distance FIELDS, not their boundaries.
Interpolating SDFs rounds convex corners off, and a diamond is all corners, so the lofted panel
is systematically small. The error does not shrink with a finer input; it is what the block
does.

The replacement is exact: extrude the trapezoidal planform polygon.
`extrude<implicit_2d,real,vector>` extrudes ONE-SIDED from the profile's own plane along the
given direction, so placing the polygon at `z = -t/2` and extruding `+t` along `+Z` gives a slab
centred on `z = 0`. Planform area and wetted area are then exact. The price is the section: a
constant-thickness plate holds twice the volume of a diamond of the same maximum thickness.

An exact double wedge is not reachable with these blocks. For a swept tapered panel the leading
edge, the mid-chord ridge and the trailing edge are three mutually skew straight lines, so the
wedge faces are hyperbolic paraboloids: not planes, not extrusions, not a revolve. There is no
shear or non-uniform-scale transform either - `combine_transforms<list<transformation>>` only
takes `rotation_transform` and `translation_transform` - so the shape cannot be sheared out of a
prism.

## 17. Cruciform patterns: use `mirror_body`, not `rotate`

`rotate<spatial3d,point,vector,real>[1.1.0]` and `transform_object<...>[1.1.0]` both return
`any`, which then has to be fed into a `list<implicit>`. `mirror_body<implicit,plane>` returns a
real `implicit`, so the block graph stays typed. A cruciform is exactly two mirror pairs, so no
rotation is needed at all: build the +Y panel and the +Z panel from the same arithmetic refs
with the Y and Z coordinates swapped, then mirror each about `plane<point,vector,vector>`. The
XZ plane (which mirrors Y) is `plane(origin, (1,0,0), (0,0,1))`; the XY plane (which mirrors Z)
is `plane(origin, (1,0,0), (0,1,0))`.

Polygon winding matters: both profiles of a loft must come out with the same normal or the loft
twists. The traversal LE-root, LE-tip, TE-tip, TE-root gives a consistent normal on both span
axes.

## 18. `offset_implicit` with a NEGATIVE distance is the cheap way to hollow a body

`offset_implicit<implicit,real_field>(Body, Distance)` accepts a negative distance and shrinks
the body. One block replaces the roughly 120 arithmetic blocks that a second, inward-offset
revolved profile would need, and it is a true normal offset rather than an approximation.
Measured on the SV-1 at `t_wall = 3 mm`:

| Quantity | Measured |
|---|---|
| OML volume | 0.335465 m^3 |
| Interior cavity, less three ring bulkheads | 0.322744 m^3 |
| Structure = (OML + fins) - cavity | 0.015350 m^3 |
| Structure mass at 2810 kg/m^3 | 43.13 kg |

`cavity + structure` came to 0.338094 m^3 against an OML of 0.335465 m^3. The 0.78 percent
excess is not an error: the structure includes the fin volume OUTSIDE the OML, and 0.00267 m^3
of exposed plate fins is exactly that difference.

Bulkheads are `cylinder<point,point,real>` discs of thickness `t_wall`, deliberately oversized
in radius and only ever SUBTRACTED from the interior void, so nothing has to trim them.

`blend_enum` value `{"enum": 0}` with a zero blend radius is the no-blend boolean and works on
`boolean_union[1.1.0]`, `boolean_subtract[1.1.0]`. Both log
"version 1.1.0 is now deprecated for improved quality when the Blend Type is set to
Continuous", which does not apply to a no-blend boolean.

## 19. `surface_area<implicit,real>` ignores its Relative error input, and it sets the cost floor

Per-block timings from `ntopcl -v 2` on the full SV-1 notebook:

| Block | Wall time |
|---|---|
| `surface_area<implicit,real>` on the body | 15.5 to 19.9 s |
| `surface_area<implicit,real>` on the fin set | 13.1 to 22.4 s |
| `mass_properties<implicit,real_field,real>` on the structure | 6.1 s |
| `mass_properties` on the cavity | 2.6 s |
| `mass_properties` on the OML | 2.5 s |
| every one of the other 280-odd root entries | under 3.3 s |

Sweeping the area block's "Relative error" input over 0.002, 0.01, 0.05 and 0.2 changed
**nothing**: the reported area was bit-identical (4.003007274647162 m^2) at every target and the
wall time stayed near 15 s. There is no accuracy-for-speed trade to make on this build.

Consequence for WP5: one `measure_rocket` call costs about 30 s, and the two area blocks are
most of it. That is NOT the mesh - `implicit_to_mesh` at 1.5e-3 m over the whole rocket is a
minor cost next to the area integration. A sizing loop that needs to be faster should drop
`area_wetted_fins` (the aero model can get the fin planform exactly from the design vector)
rather than coarsen the mesh.

Fixed overhead per run is about 8 s: 1 s to log in, then about 7 s between "Notebook started"
and the first block completing.

## 20. `mass_properties` on a symmetric body puts the CG about 1 mm off axis

At a 0.002 relative-error target the SV-1 structure (43 kg, symmetric about both the XY and the
XZ plane) reported a centre of gravity of (2.409358, 0.001224, -0.0000173) m. The 1.2 mm
off-axis offset is the volume integration's error showing up in the first moments, and it is
0.35 percent of a calibre. Test against a tolerance, never against zero.

`principal moments` works through a chained props ref and came back as
(60.3648, 60.3549, 1.4458) kg.m^2. Note the ORDER: the two transverse moments first and the
roll moment last, i.e. NOT sorted ascending. Do not assume an ordering; sort them, or identify
the roll moment as the smallest. They are principal moments about the CG in the principal frame,
not `Ixx`, `Iyy`, `Izz` in body axes. For this configuration the principal axes coincide with
the body axes so the distinction does not bite, but it would for an asymmetric part.

## 21. A `file_path` notebook INPUT works, which makes export destinations free

`Recipe.add_input(name, "file_path", default=...)` declares an input that `-t` reports as
`"type": "file_path"`, and `NtopRunner.build_input_json` already normalises the value with
`to_ntop_path`. Feeding that input ref straight into `export_mesh`, `export_part` or
`export_implicit_body` makes the STL, STEP and `.implicit` destinations run-time values.
Combined with a mesh-tolerance input, the notebook's topology then depends only on `n_fin`,
`nose_shape`, the ogive sample count and WHICH exports exist, so one `convert` serves every
design point of the sizing loop. WP4 caches the `.ntop` at
`runs/_ntop_cache/sv1_<topology hash>.ntop` and memoises the `-t` templates in the process, so
a repeat call spends nothing at all before the run.

Turning an export ON or OFF does change the block graph, so it is part of the cache key. That is
why `measure_rocket` keeps STEP off by default.

## 22. The WP1 mesh tolerance does NOT scale to a 4 m part

Section 4 measured `implicit_to_mesh` on a 25 mm sphere and found 1.0e-3 m to be a good working
point. That number is worthless for the rocket, because the block drives a DENSE voxel grid:
cost and memory go as (bounding box volume) / tolerance^3, and the SV-1 box is
4.0 x 0.71 x 0.71 m = 2.02 m^3, about 16000 times the sphere's box.

Measured on the whole rocket (OML plus the four fin panels), `convert` wall time, which
includes about 10 s of fixed overhead:

| Tolerance | `convert` wall time | STL size | Peak resident |
|---|---|---|---|
| 8.0e-3 m | 28.2 s | 3.6 MB | about 0.4 GB |
| 5.0e-3 m | 51.7 s | 9.2 MB | about 0.65 GB |
| 3.0e-3 m | not finished at 128 s | - | past 1.4 GB, killed |
| 1.5e-3 m | not finished | - | past 4.8 GB, killed |

`(8/5)^3 = 4.1` predicts a 4x rise from 8 to 5 mm; the measured rise was 2.7x, so the scaling is
a little better than cubic in this range but not much.

WP4 defaults to 5.0e-3 m, which resolves the 12 mm fin thickness with about 2.4 cells. Anything
finer needs to be run deliberately, with the memory watched.

The practical consequence is that the exports, not the measurements, dominate a full run, so
`measure_rocket` has **every export off by default**. One measurement-only run is about 30 s;
adding the STL takes it to about 70 s. Nothing measured comes off the mesh, so the sizing loop
loses no information by leaving it off.

## 23. STEP export works on the rocket, but only at a coarse CAD tolerance

Section 9's `brep.part` chain is correct and still works. Its 5.0e-4 m tolerance does not:

| CAD tolerance | `convert` wall time | STEP size | Result |
|---|---|---|---|
| 2.0e-2 m | 11.6 s | 1.15 MB | fine |
| 1.0e-2 m | 22.8 s | 8.96 MB | fine, and the WP4 default |
| 2.0e-3 m | killed | - | passed 9 GB resident with no diagnostic |

Same lesson as section 22: `cad_body_from_implicit_body` scales with the part's bounding box,
and a 25 mm sphere is not a 4 m rocket. Watch the resident set, not the clock: the process
gives no progress output at all and looks identical to a slow-but-healthy run right up to the
point where it exhausts memory.

Practical rule for a part of this size: start at 2.0e-2 m, halve once, and stop.

## 24. S(x) for wave drag DOES work, but only through the block the universe calls deprecated

Section 13 point 6 asked for a probe of the cross-section area distribution. Result: it works,
and it is accurate to 0.13 percent.

The chain is

```
plane<point,vector,vector>                        (normal along X, at station x)
extract_section<implicit,plane,real>[1.1.0]  ->   implicit_2d
body_surface_area<implicit_2d,real>[1.1.0]   ->   real   (the section AREA)
```

The trap is the last block. `vendor/functions.json` lists
`surface_area<implicit_2d,real>[1.2.0]` as the current, non-deprecated block and
`body_surface_area<implicit_2d,real>[1.1.0]` as deprecated. On these builds that is exactly
backwards. Every one of

```
surface_area<implicit_2d,real>
surface_area<implicit_2d,real>[1.1.0]
surface_area<implicit_2d,real>[1.2.0]
surface_area<implicit_2d,real>[2.0.0]
surface_area<implicit_2d,real>[5.20.0]
```

is REJECTED by `ntopcl convert` at load time with a bare `[E]: Error loading recipe:` and
nothing else. Both `body_surface_area<implicit_2d,real>` and
`body_surface_area<implicit_2d,real>[1.1.0]` load and run. On the mid-plane section of a 50 mm
sphere they returned 0.00785414 m^2 and 0.00785414 m^2 against an exact pi r^2 = 0.00785398,
i.e. +0.002 percent.

This is a THIRD failure mode of the vendored universe, on top of the two in section 5: not just
missing revisions and missing blocks, but a WRONG deprecation flag that points at a block the
shipped builds do not have. `Recipe.raw_block` is the way past it.

The 3D `surface_area<implicit,real>[1.2.0]` is fine, so the problem is specific to the
`implicit_2d` overload.

Measured S(x) against the closed-form ogive-cylinder-boattail section plus the fin plate
sections, 8 stations on the default SV-1:

| x (m) | nTop S(x) (m^2) | closed form | error |
|---|---|---|---|
| 0.250 | 0.017461 | 0.017484 | -0.13 % |
| 0.750 | 0.081475 | 0.081504 | -0.04 % |
| 1.250 | 0.096219 | 0.096211 | +0.01 % |
| 2.750 | 0.096219 | 0.096211 | +0.01 % |
| 3.750 | 0.104861 | 0.104851 | +0.01 % |

The fins are picked up: at x = 3.75 m the measured section exceeds the bare cylinder by
0.00864 m^2, which is exactly `n_fin * b_fin * t_fin`.

Cost is about 0.9 s per station: the run went from 28.2 s at 0 stations to 35.3 s at 8. WP4
therefore leaves `area_stations = 0` by default and lets the caller ask for it.

`extract_section`'s optional "Min. Feature Size" is worth filling in (1.0e-3 m here). A section
through a 12 mm fin plate is a thin sliver, and the default feature size is a plausible way to
lose it.

---

# nTop notes - added by the IV-1 two-stage work package

Everything below was measured by running `ntopcl` (nTop 5.53.2, installed build, this machine)
while building `rocketgen/ntopgen/stack_notebook.py`, the two-stage IV-1 notebook. It extends the
sections above and does not replace them. The probe that produced these numbers is kept at
`runs/IV-1_geom/_probe.py`, which converts and runs ONE rung of the build ladder at a time and
prints every measured value next to its closed form.

## 25. `surface_area<implicit,real>` cost is set by the FIELD, not by the area

Section 19 recorded 13 to 22 s per call and concluded that the block sets the cost floor. The
two-stage notebook makes five such calls and shows WHY, because one of them is nearly free:

| Body measured | Area | `surface_area` wall time |
|---|---|---|
| stage-2 revolved ogive-cylinder | 2.080 m^2 | 24.6 s |
| stage-2 four-panel fin union | 0.283 m^2 | 24.2 s |
| four-panel strake union | 0.425 m^2 | 23.8 s |
| stage-1 four-panel fin union | 0.522 m^2 | 16.9 s |
| **stage-1 `cylinder` primitive** | **2.642 m^2** | **0.27 s** |

The booster has the LARGEST area of the five and costs ninety times less than the smallest. The
difference is not the area and not the bounding box: it is what the signed-distance field is made
of. A `cylinder<point,point,real>` primitive has a closed-form field. A `revolve` of a 27-point
polygon, or a `boolean_union` of four `extrude`d profiles two of which are `mirror_body` results,
does not, and the area integration pays for every field evaluation.

Practical consequence, and it is actionable: **use a primitive whenever the shape is a
primitive.** Building the booster as a revolved four-point polygon, for consistency with the
payload stage, would have cost 25 s per run for nothing. It would also have been exact, so no
accuracy test would ever have caught it. The same effect appears in `mass_properties`: 0.23 s on
the booster cylinder against 3.8 s on the revolved stage-2 body.

## 26. `cone<point,point,real,real>` removes a topology branch

SPEC_IV1 asks for a conical interstage when the stage diameters differ and a cylindrical one when
they match. Written as a branch, that would put a DIMENSION into the topology cache key, so the
sizer would need a second `.ntop` the moment it equalised the diameters, and the
convert-once-run-many pattern would quietly break.

`cone` with `Radius 1 == Radius 2` IS a cylinder. One block covers both cases and the block graph
no longer depends on any dimension. Measured on the default IV-1 interstage (R2 = 0.140 m,
R1 = 0.200 m, L = 0.280 m): volume 0.02568578 m^3 against a closed-form
`pi L (R2^2 + R2 R1 + R1^2) / 3 = 0.02568550` m^3, i.e. **+0.001 percent**. A `cone` is exact.

Generalisation worth keeping: before writing a topology branch, check whether one block's
degenerate case already covers both sides of it. A branch in the graph costs a whole extra
`convert`.

## 27. `offset_implicit` hollows a `cone` and a `cylinder` as well as a `revolve`

Section 18 established the negative-offset trick on a revolved body. It works unchanged on the
primitives, and it shrinks the FLAT ENDS too, which is what you want and is easy to forget:

* on the booster `cylinder`, `offset_implicit(body, -t_wall)` gives a closed case with flat end
  plates of thickness `t_wall` at BOTH ends. Measured cavity 0.2547456 m^3 against a closed-form
  `pi (R - t)^2 (L - 2t) = 0.2547430` m^3, +0.001 percent;
* on the interstage `cone`, `-t_interstage` gives a conical shell that keeps two annular end
  rings. The measured shell volume is 1.59x the bare lateral-area-times-thickness value
  (0.0012163 m^3 against 7.647e-4 m^3), and the excess is exactly those rings. A test that
  asserted equality with the lateral estimate would FAIL on a correct model.

## 28. Volume closure across four separate measurements is the strongest available check

For each hollow stage the notebook measures four things with four different blocks: the outer
mould line volume, the cavity volume, the structure volume, and the exposed plate volume. They
must satisfy

```
cavity + structure  ==  outer mould line + (plate volume outside the outer mould line)
```

Measured on the default IV-1:

| Stage | cavity + structure | OML + exposed plates | agreement |
|---|---|---|---|
| payload stage | 0.1317206 + 0.0083268 = 0.1400474 | 0.1374584 + 0.0026040 = 0.1400624 | 1.1e-4 |
| booster | 0.2547456 + 0.0115681 = 0.2663137 | 0.2638892 + 0.0024007 = 0.2662899 | 8.9e-5 |

Both close to about one part in ten thousand. This is a far better test than comparing any single
number to closed form, because it cannot be satisfied by a body of the wrong shape: it links a
boolean subtract, a negative offset, two unions and four independent adaptive integrations. It is
the check that catches a bulkhead placed outside the void, a plate that failed to plug into the
wall, or a cavity that leaked out through the base.

## 29. A plate root must plug into the wall, and the reported area then includes the root patch

Section 16's constant-thickness plate is right for a fin and even better for a strake, which
really is a constant-thickness rib rather than an aerofoil. Two measured consequences:

1. **Plug, do not touch.** Every plate root sits at radius `R - t_wall`, not at `R`. Tangential
   contact of two implicit bodies is numerically fragile. The leading edge and chord are
   extrapolated inboard along the same straight sweep and taper lines, so the chord at the true
   surface `y = R` is still exact.
2. **`boolean_subtract(plates, body)` leaves a root patch.** The exposed-plate area therefore
   includes a curved strip of the BODY surface, `2 R asin(t / 2R)` wide by the chord long, which
   is `t` times the chord to better than 0.01 percent. For a STRAKE that matters a lot, because a
   strake is thick relative to its height. Measured against the closed form that includes it:

   | Panel set | measured | solid closed form | error | zero-thickness reference | ratio |
   |---|---|---|---|---|---|
   | 4 strakes, 30 mm tall, 8 mm thick | 0.425257 m^2 | 0.427526 m^2 | -0.53 % | 0.336000 m^2 | 1.266 |
   | 4 stage-2 fins | 0.282702 m^2 | 0.283219 m^2 | -0.18 % | 0.252000 m^2 | 1.122 |
   | 4 stage-1 fins | 0.521842 m^2 | 0.522806 m^2 | -0.18 % | 0.480000 m^2 | 1.087 |

   So a test that compared the measured strake area with `StrakeSpec.wetted_area` and demanded a
   few percent would FAIL on a correct model, by 27 percent. The right reference is the solid, and
   `stack_notebook.strake_solid_area` / `fin_solid_area` write it out. The measured number is also
   the right one to hand to a skin-friction model, because friction acts on the real surface.

   The measured plate VOLUMES confirm the plates independently: strakes 0.00134475 m^3 against
   `n h L t = 0.00134400`, +0.06 percent.

## 30. Namespacing one notebook's outputs across several bodies

A notebook has exactly one output slot (section 1) and `driver.parse_outputs` flattens the JSON
dictionary onto one `NtopMeasurements`. A notebook that measures THREE bodies cannot use that flat
object: `s1_volume_total` and `s2_volume_total` both map onto `volume_total` and the second
overwrites the first.

What works, and is now in the repo:

1. Prefix every emitted name by body: `s1_`, `s2_`, `is_`, `st_`. Register the prefixed names in
   `driver.OUTPUT_NAME_MAP` anyway, so `ParsedOutputs.unmapped` stays honest about what the
   notebook emitted.
2. IGNORE `ParsedOutputs.measurements` entirely and split `ParsedOutputs.raw` by prefix.
3. Fill one record per body with the new additive helper
   `driver.measurements_from_names(values, target=...)`, which applies the same name table and the
   same field casting as `parse_outputs`, so the two cannot drift apart.
4. Carry the extra per-body quantities on a `NtopMeasurements` SUBCLASS
   (`stack_notebook.StageMeasurements`), not by editing `config.py`. `register_output_names`
   refuses a target that is not a field of `NtopMeasurements`, which turns out to be a useful
   guard rail: it forces the subclass fields to be collected explicitly rather than
   half-registered.

Do not skip step 2. The flat object looks plausible and is wrong.

## 31. `register_sources` catches a real defect between two notebooks

The same guard rail exists on the SOURCES side, and it fired. `rocket_notebook.py` registers
`relative_error` and `area_relative_error` with SV-1 timings in their text. Reusing those two key
names in `stack_notebook.py` with IV-1 timings made a full-suite run abort at import with
`ValueError: conflicting source for 'area_relative_error'`, because a full run imports both
modules while a single-file run imports only one.

That is the right behaviour and worth stating plainly: source keys are GLOBAL. Namespace them per
vehicle (`iv1_relative_error`, `iv1_area_relative_error`). Had the check not existed, the
engineering report would have printed SV-1's 19.9 s timing as the provenance of an IV-1 constant,
and nobody would have noticed.

## 32. Report a per-body reference station, and let Python do the frame change

The geometry is built in ONE frame - the stack, nose tip at the origin - because building each
stage in its own frame would double the notebook. But `masses_iv1.build_stack_masses` reads
`cg_structure[0]` as a station from that stage's OWN forward face and adds the stage offset
itself, so a stack-frame CG would be silently wrong by 2.98 m on the booster.

The fix that keeps nTop as the authority: emit `sN_x_forward`, the stage's forward face in stack
coordinates, as an ordinary `real` output alongside the CG, and subtract it in Python. It costs
nothing, because the station is already a `real` in the graph, and the subtraction then uses
nTop's own number rather than Python arithmetic on the design vector. Both frames are reported:
`StageMeasurements.cg_structure` is stage-local and `.cg_structure_stack` is stack-frame.

Measured on the default IV-1: booster structure CG at 4.213071 m in stack coordinates, forward
face at 2.980000 m, so 1.233071 m from its own nose, which is 0.587 of its 2.100 m length. The
off-axis components were 0.32 mm and 0.02 mm, consistent with the 1.2 mm section 20 measured on
SV-1. Test against a tolerance, never against zero.

## 33. Measured cost of the two-stage notebook, and how it grows

Build ladder, one rung at a time, every export off. `convert` and one `-j` run, wall seconds:

| `build_stage` | bodies present | convert | run |
|---|---|---|---|
| `s2_oml` | payload stage outer mould line | 18.3 | 14.7 |
| `s2_plates` | + 4 strakes, 4 stage-2 fins | 42.2 | 42.2 |
| `s2_hollow` | + cavity, 2 bulkheads, measured structure | 56.0 | 50.5 |
| `booster` | + booster, 4 fins, cavity, structure | 73.9 | 76.3 |
| `full` | + interstage shell | 98.5 | 117.8 |

422 root entries, 287 KB of recipe JSON, 27 notebook inputs. The growth is not mysterious: each
rung costs what its own `surface_area` and `mass_properties` blocks cost and nothing more, so the
ladder is also a cost model. `convert` costs about the same as a run, as section 3 says it must.

**There is a 2x SPREAD on repeats of an identical job, and it is not explained.** Five runs of the
final notebook took 55.0, 78.6, 92.7, 114.7 and 117.8 s, and three converts of it took 62.6, 95.4
and 96.5 s. It is not the block graph: the FASTEST convert was of the LARGER 422-block version,
and the two 95-96 s converts were of the 414-block one. It is not the measured numbers, which are
bit-stable across repeats. No cause is claimed here; the honest statement is that a single timing
of this notebook is worth about a factor of two, so budget the upper end and never read a change
of tens of percent between two runs as a regression. `tests/test_stack_notebook.py` therefore
gates the wall time only at 8x the reference, which catches an export left on and nothing subtler.

Adding an STL of the whole stack at a 5.0e-3 m mesh tolerance took `convert` from 98.5 s to about
250 s and a run to 392 s, and produced a 12.4 MB, 247682-triangle STL. The stack bounding box is
5.08 x 0.80 x 0.80 m = 3.25 m^3 against SV-1's 2.02 m^3, so the section 22 scaling holds:
exports, not measurements, dominate a full run. That is why every export is off by default in
`measure_stack`.

## 34. Check the price of a measurement before assuming it is expensive

Because `surface_area` on a primitive costs 0.27 s (section 25), it is worth measuring things you
would otherwise leave out. The interstage's lateral wetted area was originally omitted from the
stack total on cost grounds; on a `cone` it turned out to cost almost nothing, so the stack's
`area_wetted_body` is now the complete sum of three MEASURED bodies instead of two measured plus
one missing. On this block the price varies by two orders of magnitude with the field, not with
the geometry's size.

## 35. The build ladder found nothing wrong, which is itself the finding

CLAUDE.md section 3.5 asks for small-scale validation first, so the notebook was built in five
rungs and each rung was converted and run before the next was added. Every rung was correct on the
first attempt, and the errors against closed form did not move as bodies were added: the stage-2
volume error stayed at -0.008 percent from `s2_oml` through `full`, and its wetted-area error
stayed at -0.302 percent.

That is worth recording because it says where the risk actually is. The single-body work package
lost hours to `loft` (section 16) and to the `implicit_2d` area block (section 24), both of which
are BLOCK SEMANTICS problems. Adding more bodies to a notebook whose blocks are already understood
is cheap and safe. The expensive unknowns are new block types, not new geometry.

## 36. S(x) works across a four-body stack, and it sees the strakes

Section 24's `extract_section` plus `body_surface_area<implicit_2d,real>[1.1.0]` chain carries over
to the stacked union with no change, and it correctly crosses all four regions of the stack.
Measured at 6 stations on the default IV-1, with `SECTION_FEATURE_SIZE = 1.0e-3` m:

| x (m) | region | nTop S(x) (m^2) | closed-form bare body | excess |
|---|---|---|---|---|
| 0.423 | ogive nose | 0.027463 | 0.027464 | -0.000001 |
| 1.270 | stage-2 cylinder + strakes | 0.062545 | 0.061575 | +0.000970 |
| 2.117 | stage-2 cylinder + strakes | 0.062545 | 0.061575 | +0.000970 |
| 2.963 | interstage cone | 0.121226 | 0.121216 | +0.000010 |
| 3.810 | booster cylinder | 0.125674 | 0.125664 | +0.000010 |
| 4.657 | booster cylinder | 0.125674 | 0.125664 | +0.000010 |

The bare-body regions agree to 0.01 percent or better, INCLUDING the conical interstage, where the
station has to hit the cone at the right local radius. And the two stations that land between the
strake leading and trailing edges show an excess of 0.000970 m^2 against a predicted
`n * height * thickness = 4 * 0.030 * 0.008 = 0.000960` m^2, i.e. +1.0 percent. So a 30 mm by 8 mm
strake, which is a 0.24 cm^2 sliver, is resolved by the section block. That is a second,
independent confirmation that the strakes are really in the solid.

Cost: 6 stations gave a run of 75 s against a 55 to 118 s spread on the same notebook without
them, so the added cost is well inside the timing noise of section 33 and
the marginal cost per station is at most about 1 s, matching SV-1. It is still
off by default, because the sizing loop calls the notebook tens of times and does not need it.

## 37. The measured geometry found a real infeasibility in the default IV-1

Not an nTop note as such, but it came out of the measurement and belongs with it. The default
`config_iv1.default_iv1()` stack does NOT close on stage-2 internal volume:

```
propellant  150 kg / 1800 kg/m^3  = 0.083333 m^3
payload      75 kg / 1500 kg/m^3  = 0.050000 m^3   (masses_iv1.RHO_PAYLOAD)
                                    ----------
                         needed     0.133333 m^3
   nTop-measured stage-2 cavity     0.131721 m^3
                                    ----------
                      shortfall     0.001612 m^3, i.e. 1.2 percent
```

The stage-2 cavity cross-section is `pi (R - t_wall)^2 = pi * 0.1374^2 = 0.059309 m^2`, so the
shortfall is 27 mm of stage-2 length. Lengthening stage 2 from 2.70 m to 2.73 m closes it and
leaves the stacked length at 5.11 m against the A7 limit of 5.40 m; dropping 3 kg of propellant
would also do it. `default_iv1()` says of itself "not sized; the sizer moves it", and this is
exactly the constraint that tells it which way to move.

`tests/test_stack_notebook.py::test_the_default_stack_is_NOT_volume_closed_on_stage_2` asserts the
shortfall, so it cannot quietly disappear if the defaults change (CLAUDE.md sections 3.1 and 7).

---

## 25. Revolving a real spline, and why `functions.json` cannot be trusted to say no

**nTop can revolve a genuine spline.** No chord polygon is needed. Verified end to end through
`ntopcl` on 5.53.2: converts, runs, measures.

```
spline_by_control_points<list<point>,integer>[5.20.0]       -> spline
    (core.list<point>, integer degree)

core.list<curve_interface>                                  -> list<curve_interface>

profile_from_curves<list<curve_interface>,vector>[5.20.0]   -> new_profile
    (curve list, NORMAL vector)          the Normal belongs to THIS block

revolve<new_profile,axis,real>[5.20.0]                      -> implicit
    (new_profile, axis, angle)           NO normal input here
```

Four traps, each of which alone makes `convert` fail with a bare `Error loading recipe:`:

1. the curve type is **`curve_interface`**, not `curve`, so the list is
   `core.list<curve_interface>`;
2. `profile_from_curves` returns **`new_profile`**, not `profile`, and there is no props bridge
   from it to `implicit_2d` - you must use the `new_profile` revolve overload;
3. the Normal vector is **dimensionless** (`"units": {}`), not a length;
4. the degree argument is a plain integer literal, `{"type": "integer", "value": {"val": 3}}`.

All four blocks need `Recipe.raw_block`, because **none of them are in the vendored
`functions.json`**, and `types.json` is missing `spline`, `curve_interface`, `polycurve` and
`new_profile` as well.

### The lesson that generalises

Section 6 of `docs/REFERENCE.md` and CLAUDE.md section 4 point 6 already say the universe drifts
from the installed build. This is the sharper statement: **the universe is missing whole BLOCKS
and whole TYPES, not merely stale revisions.** So a block being absent from `functions.json` is
NOT evidence that nTop cannot do the thing.

That mistake was made here. `rocketgen/oml_spline.py` was written asserting "a `curve` CANNOT be
revolved, verified against the block universe". It was wrong, and the whole polygon-sampling
approach in that module exists because of it. The polygon route is validated and correct, but it
is not the cheapest or the most accurate option.

### How to find block ids, since guessing does not work

27 plausible id combinations were tried and all were rejected. Two methods do work:

- `ntopcl exportjson <notebook>.ntop out.json --ext --dev-blocks-on=True` on a notebook that
  already uses the block, then read the real `func` strings out of the JSON;
- build the chain up ONE block at a time, converting after each. `convert` either loads the
  recipe or says `Error loading recipe`, which isolates the offending block immediately. That is
  how the revolve overload was found once the first three blocks were known.

### Cost and accuracy

Converted in 24 s, ran in 29 s. `mass_properties` volume agreed with a Python analytic
integration of the same clamped B-spline to **-0.0255 percent**, inside the 0.05 percent relative
error requested of the block. The GUI log reports the revolve itself at 616,600 ns.
