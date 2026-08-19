# SPEC - IV-1, a two-stage interceptor-class vehicle

Second reference example for the toolkit. Read `SPEC.md` first for the SV-1 example and the
fidelity statement; this document only records what is different.

## 1. What this is

A configuration study of a **canister-launched, two-stage, strake-stabilised interceptor-class
vehicle**, sized to reach 100 statute miles of slant range.

The configuration features requested are the ones that distinguish it from SV-1:

- **Multi-stage.** A jettisoned first-stage booster and a second stage that carries the payload.
  This is a real architecture change: mass leaves the vehicle mid-flight, the aerodynamic
  reference area changes, and the stability analysis has to be done twice.
- **Strakes.** Long, low-aspect-ratio surfaces running along the second-stage mid-body, plus
  cruciform tail fins. Strakes are the reason this class of vehicle can hold a large angle of
  attack at altitude without the fins stalling.

**The vehicle and its requirements are invented for the demonstration.** They correspond to no
real programme. This is a parametric configuration study built from published conceptual-design
methods, not a model of any fielded system, and nothing here describes guidance, seekers,
warheads, energetics or countermeasures. Those are all out of scope, exactly as in `SPEC.md`
section 8.

## 2. Top-level requirements (invented)

| ID | Requirement | Value |
|---|---|---|
| A1 | Launch | vertical, sea level, from a canister |
| A2 | Slant range to intercept | >= 160,934 m (100 statute miles) |
| A3 | Intercept altitude | >= 15,000 m |
| A4 | Velocity at intercept | >= Mach 3.0 |
| A5 | Payload carried to intercept | 75 kg |
| A6 | Maximum body diameter | <= 0.42 m |
| A7 | Maximum overall length, stacked | <= 5.40 m |
| A8 | Maximum launch mass | <= 1400 kg |
| A9 | Static margin, each stage, whole flight | >= 1.0 calibre |
| A10 | Maximum dynamic pressure | <= 250 kPa |
| A11 | Lateral acceleration available at intercept | >= 15 g |
| A12 | First-stage burnout altitude | <= 20,000 m |

### Notes on the requirements

**A2 is a slant range, not a ground range.** `sqrt(x^2 + h^2)` from the launch point to the
intercept point. An interceptor flies up, so ground range alone is the wrong measure.

**A10 is higher than the 200 kPa used for SV-1.** A vertical launch accelerates through dense air
near the ground, which is the opposite of SV-1's launch at 10 km. Sizing SV-1 showed that a limit
must be checked against the mission that has to meet it.

**A11 is a capability, not a manoeuvre.** It is the lateral acceleration the vehicle could pull at
the intercept condition, with `CN_max` taken at the alpha limit. It is what makes A3 and A4 matter:
an intercept needs energy and control authority, not just arrival.

### Requirements audit: A2 against A3 against A11

**As first written, A2, A3 and A11 could not hold together.** Purely aerodynamic control was
assumed, so the available lateral acceleration was `q * S_ref * CN_max / m`. That needs dynamic
pressure, which needs air.

Measured on the stack that closes every motor, volume and structural constraint (732 kg, 5.28 m,
1404 kN.s), sweeping the pitchover angle and walking each trajectory for the furthest point at
which A3, A4 and A11 all hold at once:

| Pitchover [deg] | Furthest point meeting A3, A4 and A11 | Altitude there | Max slant range | Altitude at max slant |
|---|---|---|---|---|
| 12 | none | - | 80.7 km | 0.0 km |
| 16 | none | - | 102.4 km | 0.0 km |
| 20 | **58.3 km** | 18.5 km | 140.0 km | 0.0 km |
| 24 | 52.6 km | 20.9 km | 160.9 km | 21.9 km |
| 32 | 43.8 km | 23.4 km | 160.9 km | 56.5 km |
| 50 | 23.3 km | 18.4 km | 160.9 km | 107.7 km |

The best slant range meeting every requirement is **58.3 km, which is 36.2 miles against the 100
miles A2 asks for**. The shortfall is 102.7 km, a factor of 2.8.

The cause is not the design vector. It is a hard physical ceiling. At the post-separation mass, with
a generous `CN_max` of 2.5:

| Mach | 15 g available only below |
|---|---|
| 3.0 | 10.0 km |
| 4.0 | 13.7 km |
| 5.0 | 16.6 km |

A3 requires the intercept at or above 15 km, so the band in which A3 and A11 overlap is empty at
Mach 3 to 4 and a few hundred metres wide at Mach 5. Reaching 100 miles of slant range, meanwhile,
needs a lofted trajectory, and lofting puts the intercept above 20 km where the dynamic pressure has
collapsed. A flat trajectory keeps the air but runs out of range: it reaches the ground at 80 to
140 km.

**Resolution: the vehicle needs lateral control that does not depend on dynamic pressure.** That is
a divert or attitude-control motor, which is exactly what vehicles of this class carry and precisely
why they carry it. Section 8 originally excluded attitude-control thrusters, and that exclusion is
what made the requirement set infeasible. A12 is therefore added, and A11 is restated so that the
requirement is on the capability rather than on the mechanism:

| ID | Requirement | Value |
|---|---|---|
| A11 (restated) | Lateral acceleration available at intercept, from aerodynamic surfaces OR an attitude-control motor, whichever is greater | >= 15 g |
| A13 | Attitude-control motor total impulse | sized, reported, and charged to the mass statement |

The point of recording this is that the conflict was found by walking the trajectories against the
constraint set, not by inspection. A purely aerodynamic interceptor at this size cannot engage at
100 miles and still manoeuvre. That is a design conclusion, and it is more useful than a design.

## 3. Configuration

```
   nose        stage 2 (payload, sustainer)          stage 1 (booster)
  |----|------------------------------------|--------------------------|
   ogive      strakes along mid-body            interstage    tail fins
              + stage-2 tail fins
```

- **Stage 2** carries the payload, its own motor, four strakes and four tail fins. Diameter
  `D2 <= D1`.
- **Stage 1** is a booster with four tail fins and its own motor. It is jettisoned at separation,
  and its inert mass leaves the vehicle.
- **Interstage** is a conical or cylindrical adapter between the two, jettisoned with stage 1.
- **Strakes** are on stage 2 and therefore survive separation. They are long, thin, low-aspect
  ratio, and body-mounted.

## 4. Design variables

| Symbol | Meaning | Bounds | Units |
|---|---|---|---|
| `D1` | Stage-1 body diameter | 0.28 - 0.42 | m |
| `D2` | Stage-2 body diameter | 0.20 - 0.36 | m |
| `L1` | Stage-1 length, including interstage | 1.2 - 2.8 | m |
| `L2` | Stage-2 length, including nose | 1.8 - 3.4 | m |
| `f_nose2` | Stage-2 nose fineness | 2.5 - 5.0 | - |
| `m_p1` | Stage-1 propellant mass | 150 - 600 | kg |
| `m_p2` | Stage-2 propellant mass | 60 - 300 | kg |
| `F1` | Stage-1 thrust | 80 - 300 | kN |
| `F2` | Stage-2 thrust | 20 - 90 | kN |
| `b_strake` | Strake height above the body | 0.015 - 0.060 | m |
| `L_strake` | Strake length | 0.6 - 2.2 | m |
| `x_strake` | Strake leading-edge station on stage 2 | derived | m |
| `t_strake` | Strake thickness | 0.004 - 0.014 | m |
| `b_fin1`, `c_r_fin1` | Stage-1 fin semi-span, root chord | as SV-1 | m |
| `b_fin2`, `c_r_fin2` | Stage-2 fin semi-span, root chord | as SV-1 | m |
| `gamma_pitch` | Programmed flight-path angle after pitchover | 35 - 75 | deg |
| `t_pitch` | Time at which pitchover starts | 1.5 - 8.0 | s |

## 5. Mission profile

1. **Vertical rise.** Ignition at sea level, flight-path angle 90 degrees, held for `t_pitch`.
   This clears the canister before any manoeuvre.
2. **Pitchover.** Turn to `gamma_pitch` at a bounded rate. A gravity turn is acceptable and is
   the preferred model if the turn rate would otherwise need a made-up gain.
3. **Stage-1 boost** to burnout.
4. **Separation.** Stage-1 inert mass and the interstage leave the vehicle. The aerodynamic
   reference area becomes the stage-2 area. A short unpowered coast is allowed.
5. **Stage-2 boost** to burnout.
6. **Midcourse coast** along a lofted arc.
7. **Intercept.** The run ends when slant range reaches A2, or when the vehicle falls back to the
   ground, whichever happens first. Report altitude, Mach and available lateral acceleration at
   that point.

The run must **not** terminate on ground impact if the slant-range condition is reached first.
Reaching A2 while descending is legitimate for an interceptor.

## 6. Objective and constraints

Minimise launch mass subject to A2 through A12, plus:

- Grain length-to-diameter between 1.0 and 8.0, for each stage separately.
- Internal volume closure for each stage, from the nTop-measured cavity.
- `D2 <= D1`.
- Stage-2 nose fineness must leave room for the payload.

## 7. What must be added to the toolkit

| Module | Addition |
|---|---|
| `config.py` | `StageSpec`, a multi-stage `DesignVector`, `InterceptRequirements`, strake fields |
| `sizing/propulsion.py` | `MultiStageMotor`: a stack of stages with independent grains, separation |
| `sizing/trajectory.py` | `AscentMission`: vertical rise, pitchover, staging with mass jettison, slant-range termination |
| `sizing/aero.py` | strake normal force and drag; a reference area that changes at separation |
| `sizing/masses.py` | per-stage mass statements, and the jettisoned mass |
| `ntopgen/` | a two-body notebook with an interstage and strakes |

Everything already in the repo for SV-1 must keep working unchanged. SV-1 is the regression
baseline: its tests must still pass.

## 8. Fidelity, additional to `SPEC.md` section 5

- Strake aerodynamics use a low-aspect-ratio lifting-surface method with a body-interference
  factor. Strakes at low aspect ratio are dominated by vortex lift, so a purely linear method
  will underpredict them. State which method is used and what it omits.
- Stage separation is instantaneous and imparts no impulse or attitude disturbance.
- The interstage is jettisoned with stage 1.
- No thrust-vector control and no attitude-control thrusters. Pitchover is aerodynamic or a
  gravity turn. Real vehicles in this class use thrust-vector control or side thrusters, so the
  pitchover model here is a modelling choice and must say so.
- Available lateral acceleration is a static calculation at the intercept condition. It is not a
  dynamic manoeuvre and says nothing about response time or autopilot behaviour.
