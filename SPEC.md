# SPEC - nTop + SUAVE Rocket Generator (demo)

## 1. What this is

A closed-loop conceptual-design generator for a solid-propellant rocket vehicle.

- **SUAVE** does the physics: atmosphere, aerodynamic build-up, solid-motor performance,
  trajectory integration, mass build-up, and the sizing solve.
- **nTop** does the geometry: a parametric notebook, authored programmatically, that turns the
  sized parameters into a real implicit solid, then measures it and exports it.
- The two are coupled. nTop is not a downstream renderer - its measured mass properties,
  wetted area and internal volumes are fed **back** into SUAVE and the loop is iterated to
  convergence.

Deliverables: a converged design, STL + STEP geometry, a parameter trade study, and an
engineering report.

## 2. Reference concept: "SV-1"

Air-launched supersonic rocket vehicle. Ogive-cylinder body, cruciform tail fins,
boost-sustain solid motor with a fixed aft nozzle, nose radome over the seeker.

### Top-level requirements (demo TLRs)

These are **assumed** for the purpose of the demo. They are not derived from any real program.

| ID | Requirement | Value |
|---|---|---|
| R1 | Launch condition | M 0.85 at 10,000 m, level |
| R2 | Cruise (sustain) condition | M 2.00 at 12,000 m |
| R3 | Range, launch point to impact | >= 185 km (100 nmi) |
| R4 | Warhead mass | 90 kg |
| R5 | Guidance + seeker + actuation mass | 15 kg |
| R6 | Terminal Mach at impact | >= 1.50 |
| R7 | Max body diameter | <= 0.45 m |
| R8 | Max overall length | <= 4.20 m |
| R9 | Max launch mass (pylon limit) | <= 1,100 kg |
| R10 | Static margin, whole flight | >= 1.0 calibre |
| R11 | Max fin span (folded not modelled) | <= 0.90 m |
| R12 | Max dynamic pressure | <= 200 kPa (revised, see below) |

### Requirements audit: R6 versus R12

R12 was first written as 90 kPa. It is mutually exclusive with R6.

R6 demands Mach 1.50 at impact. At sea level, using US Standard 1976, that is 510.4 m/s at
1.225 kg/m^3, so `q = 159.6 kPa`. A 90 kPa limit caps sea-level impact speed at 383.3 m/s,
which is Mach 1.13. No design can satisfy both requirements at once.

R12 is therefore revised to 200 kPa. That clears the 159.6 kPa floor which R6 itself sets, with
25 percent margin, and it is consistent with the sea-level supersonic manoeuvre pressures quoted
for this missile class in Fleeman, *Tactical Missile Design*, 2nd ed., Chapter 3.

The point of recording this is that the conflict was found by running the constraint set, not by
inspection. It is the kind of defect a sizing loop exists to catch.

### Mission profile

1. **Separation** - 1.5 s unpowered, launch condition held.
2. **Boost** - high thrust, accelerate from M 0.85 to M 2.00 and climb 10,000 -> 12,000 m.
3. **Sustain** - constant-Mach, constant-altitude cruise at R2, thrust = drag.
4. **Coast** - motor burnout to terminal dive entry.
5. **Terminal dive** - constant-flight-path-angle dive to sea level, unpowered.

## 3. Design variables (what the sizer moves)

| Symbol | Meaning | Bounds | Units |
|---|---|---|---|
| `D` | Body diameter | 0.25 - 0.45 | m |
| `L_total` | Overall length | 3.0 - 4.2 | m |
| `f_nose` | Nose fineness, `L_nose / D` | 2.0 - 4.0 | - |
| `m_p_boost` | Boost propellant mass | 40 - 250 | kg |
| `m_p_sustain` | Sustain propellant mass | 100 - 500 | kg |
| `F_boost` | Boost thrust (sea-level equivalent) | 20 - 90 | kN |
| `b_fin` | Fin exposed semi-span (per fin) | 0.10 - 0.30 | m |
| `c_r_fin` | Fin root chord | 0.25 - 0.60 | m |
| `taper_fin` | Fin taper ratio | 0.2 - 0.8 | - |
| `sweep_fin` | Fin leading-edge sweep | 20 - 60 | deg |
| `x_fin` | Fin LE station from nose tip | derived (aft-set) | m |

## 4. Objective and constraints

Minimise launch mass `m_0` subject to R3, R6, R7, R8, R9, R10, R11, plus:

- Internal volume closure: warhead + guidance + motor grain + case + structure must fit inside
  the nTop-measured internal cavity, with 5 % packing margin.
- Motor grain L/D between 1.0 and 8.0.
- Burnout dynamic pressure <= 90 kPa (structural).

## 5. Fidelity, stated plainly

This is a **conceptual** (Class-I / "sizing") tool. Named methods only, no invented numbers.

- Atmosphere: US Standard 1976 (SUAVE `Analyses.Atmospheric.US_Standard_1976`).
- Body wave drag: Van Driest / slender-body ogive-cylinder wave drag; cross-checked against a
  second correlation.
- Skin friction: Van Driest II compressible flat-plate, with form factor for a body of revolution.
- Base drag: correlation on base area and Mach, with a powered-base reduction while thrusting.
- Fin lift and drag: linear supersonic thin-airfoil / Busemann for CN_alpha, plus fin skin
  friction and LE wave drag; subsonic fallback via lifting-line.
- Body normal force: slender-body plus Allen-Perkins viscous cross-flow.
- Solid motor: constant-`c*` grain regression, nozzle expansion from area ratio, altitude
  thrust correction. `Isp` from published solid-propellant ranges, cited.
- Trajectory: 3-DOF point mass, explicit integration, SUAVE mission segments where they fit.
- Masses: mixture of nTop-measured (structure, case, fins, from real volumes x material density)
  and correlation-based (motor inerts, avionics).

Any quantity that cannot be computed from a named method must be reported as an input
assumption in a table, never buried in code.

## 6. Coupling loop

```
  guess design vector x
    |
    v
  [1] SUAVE mass build-up        -> preliminary masses, stations
    |
    v
  [2] nTop notebook (Automate)   -> solid geometry
        emits: volume, wetted area, CG, inertia tensor, internal cavity volume,
               cross-section area distribution S(x), fin planform areas,
               STL + STEP + .implicit
    |
    v
  [3] SUAVE aero build-up        <- uses nTop S(x), wetted area, fin areas
    |
    v
  [4] SUAVE trajectory           -> range, terminal Mach, q history
    |
    v
  [5] constraint / objective residuals -> update x, back to [1]
```

Convergence: relative change in `m_0` and range below 0.2 % on successive iterations.

## 7. Deliverables

1. `rocketgen/` - the generator package.
2. A converged `SV-1` design: parameter table, mass statement, trajectory plots.
3. `runs/<name>/` with the generated `.ntop`, input/output JSON, STL, STEP, and measurements.
4. A trade study (DOE) over at least `D`, `m_p_sustain`, and `f_nose`, showing the feasible
   region and the sensitivity of range and launch mass.
5. An engineering report (PDF) following the `engineering-report` skill, in ASD-STE100
   Simplified Technical English.

## 8. Non-goals

- 6-DOF flight mechanics, autopilot, or guidance law design.
- CFD. (The AFLR3/FUN3D path exists on this machine but is out of scope here.)
- Energetics or propellant chemistry.
- Structural sizing beyond a wall-thickness-and-density mass estimate.
- Any real-world program correspondence. The TLRs in section 2 are invented for the demo.
