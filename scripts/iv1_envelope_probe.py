"""Map the IV-1 engagement envelope, and test whether A2, A3 and A11 can hold together.

Not a deliverable. A diagnostic, kept because the answer it produced changed the specification.

The question: SPEC_IV1.md asks for 160.934 km of slant range (A2), an intercept at or above
15 km (A3), and 15 g of lateral acceleration available at intercept (A11). A11 is aerodynamic, so
it needs dynamic pressure, so it needs air. A2 needs a lofted trajectory to reach that far, and
lofting puts the intercept where there is no air. Those pull in opposite directions.

This walks each trajectory and finds the furthest point at which EVERY requirement holds at once.
"""
from __future__ import annotations

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

from test_trajectory_iv1 import StubStackAero  # noqa: E402

from rocketgen.config_iv1 import (  # noqa: E402
    InterceptRequirements,
    default_iv1,
    lateral_g,
    slant_range,
)
from rocketgen.sizing.atmosphere import atmo  # noqa: E402
from rocketgen.sizing.masses_iv1 import build_stack_masses  # noqa: E402
from rocketgen.sizing.propulsion_iv1 import MultiStageMotor  # noqa: E402
from rocketgen.sizing.trajectory_iv1 import AscentMission  # noqa: E402

# CN_max used for the capability figure. The real value comes from aero_iv1.StackAero; 2.5 is a
# placeholder that is generous for a strake-plus-fin configuration at 20 degrees alpha, so the
# envelope reported here is an UPPER bound on what the vehicle can do.
CN_MAX_PLACEHOLDER = 2.5


def closing_stack():
    """The stack that closes every motor, volume and structural constraint."""
    dv = default_iv1()
    s1, s2 = dv.stages
    s1.D, s1.L, s1.m_propellant, s1.F_thrust, s1.eps_nozzle = 0.42, 2.6, 380.0, 90.0e3, 6.0
    s2.D, s2.L, s2.m_propellant, s2.F_thrust, s2.eps_nozzle = 0.34, 2.4, 130.0, 18.0e3, 18.0
    dv.f_nose = 3.0
    return dv


def envelope(dv, reqs, motor, m0, gamma_deg: float) -> dict[str, float]:
    """Furthest point on one trajectory at which every requirement holds simultaneously."""
    d = dv.replace(gamma_pitch=math.radians(gamma_deg))
    mission = AscentMission(d, reqs, motor, StubStackAero(), m0)
    res = mission.fly(dt=0.05, adaptive=True, t_max=600.0)

    S2 = dv.payload_stage.S_ref
    best = {
        "gamma_deg": gamma_deg,
        "slant_ok": 0.0,
        "h_ok": 0.0,
        "mach_ok": 0.0,
        "g_ok": 0.0,
        "slant_max": 0.0,
        "h_at_slant_max": 0.0,
        "g_at_slant_max": 0.0,
    }
    sep = res.diagnostics.get("separation_index", 0)

    for i in range(len(res.time)):
        h, x, V, M, mass = res.h[i], res.x[i], res.V[i], res.mach[i], res.mass[i]
        if h < 0.0:
            continue
        st = atmo(h)
        q = 0.5 * st.density * V * V
        S = S2 if i >= sep else dv.booster.S_ref
        g_av = lateral_g(q, S, CN_MAX_PLACEHOLDER, mass)
        sr = slant_range(x, h)

        if sr > best["slant_max"]:
            best["slant_max"] = sr
            best["h_at_slant_max"] = h
            best["g_at_slant_max"] = g_av

        # every requirement at once
        if (
            h >= reqs.h_intercept_min
            and M >= reqs.mach_intercept_min
            and g_av >= reqs.lateral_g_min
            and sr > best["slant_ok"]
        ):
            best["slant_ok"] = sr
            best["h_ok"] = h
            best["mach_ok"] = M
            best["g_ok"] = g_av
    return best


def main() -> int:
    reqs = InterceptRequirements()
    dv = closing_stack()
    motor = MultiStageMotor(dv, reqs)
    sm = build_stack_masses(dv, reqs, motor=motor)

    print(f"stack: m0 {sm.m0:.1f} kg, L {dv.L_total:.2f} m, "
          f"impulse {motor.total_impulse_vacuum()/1e3:.0f} kN.s, "
          f"jettison {motor.jettisoned_mass():.1f} kg")
    print(f"CN_max placeholder {CN_MAX_PLACEHOLDER} makes every g figure an UPPER bound\n")
    print("Furthest point at which A3 (h >= 15 km), A4 (M >= 3) and A11 (15 g) ALL hold:")
    print(f"{'gamma':>6} {'all-req slant':>14} {'h':>8} {'M':>6} {'g avail':>8} "
          f"| {'max slant':>10} {'h there':>8} {'g there':>8}")

    rows = []
    for gamma in (12, 16, 20, 24, 28, 32, 40, 50):
        b = envelope(dv, reqs, motor, sm.m0, float(gamma))
        rows.append(b)
        ok = b["slant_ok"]
        print(
            f"{gamma:>6} {ok/1e3:>11.1f} km {b['h_ok']/1e3:>6.1f} km {b['mach_ok']:>6.2f} "
            f"{b['g_ok']:>8.1f} | {b['slant_max']/1e3:>7.1f} km {b['h_at_slant_max']/1e3:>6.1f} km "
            f"{b['g_at_slant_max']:>8.2f}"
        )

    best = max(rows, key=lambda r: r["slant_ok"])
    print()
    print(f"BEST slant range meeting every requirement: {best['slant_ok']/1e3:.1f} km "
          f"({best['slant_ok']/1609.344:.1f} miles) at gamma {best['gamma_deg']:.0f} deg, "
          f"h {best['h_ok']/1e3:.1f} km")
    print(f"A2 asks for {reqs.slant_range_min/1e3:.1f} km ({reqs.slant_range_min_miles:.0f} miles)")
    shortfall = reqs.slant_range_min - best["slant_ok"]
    if shortfall > 0:
        print(f"SHORTFALL {shortfall/1e3:.1f} km. A2, A3 and A11 cannot hold together.")
    else:
        print("A2, A3 and A11 hold together.")

    # the altitude ceiling A11 imposes, independent of any trajectory
    print()
    print("Altitude ceiling that A11 imposes on its own, at the post-separation mass:")
    mass = sm.mass_after_separation()
    S = dv.payload_stage.S_ref
    for M in (3.0, 4.0, 5.0):
        h_lim = None
        for h in range(0, 40_000, 100):
            st = atmo(float(h))
            V = M * st.speed_of_sound
            q = 0.5 * st.density * V * V
            if lateral_g(q, S, CN_MAX_PLACEHOLDER, mass) < reqs.lateral_g_min:
                h_lim = h
                break
        print(f"   Mach {M:.1f}: 15 g available only below {h_lim/1000.0 if h_lim else 40.0:.1f} km")
    print(f"   A3 requires the intercept at or above {reqs.h_intercept_min/1e3:.0f} km, so the "
          f"usable band is narrow or empty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
