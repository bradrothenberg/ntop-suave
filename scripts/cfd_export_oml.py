"""Export body-only SV-1 outer mould lines for CFD validation: ogive against spline.

WHAT THIS EXPERIMENT IS FOR
---------------------------
The spline study rests on ONE number that no measurement in this repository checks: the
slender-body wave-drag shape ratio, 0.87521, which says the splined nose carries 12.5 percent
less wave drag than the tangent ogive of the same length and base area. That ratio comes from
linearised theory. It is applied to a Bonney correlation that cannot see shape at all, so
nothing downstream of it can falsify it. FUN3D can.

WHY BODY ONLY, AND WHY INVISCID
-------------------------------
The comparison has to isolate the thing being tested.

* **No fins.** Fin wave drag, fin-body interference and the thin-plate meshing they demand are
  all identical between the two configurations and would add noise to a difference of a few
  percent. `n_fin = 0` is supported by the notebook for exactly this kind of use.
* **Inviscid.** Skin friction is roughly 25 percent of this vehicle's CD0 and is unchanged by
  the nose shape to within the wetted-area difference, which is 0.4 percent. Solving Euler
  removes it from the comparison entirely, so what CFD reports IS pressure drag: wave plus
  base. That is what the model claims to predict.
* **Same everything else.** Both configurations use the converged spline design's dimensions.
  ONLY `nose_shape` differs. A comparison in which two things changed would answer nothing.

The prediction under test, from `rocketgen/sizing/aero.py` at 10 km:

    Mach 1.5   nose wave CD 0.09303 (ogive) -> 0.08142 (spline)
    Mach 2.5               0.07288          -> 0.06379
    Mach 3.5               0.06733          -> 0.05893

Run:
    .venv/Scripts/python.exe scripts/cfd_export_oml.py
"""
from __future__ import annotations

import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

OUT = os.path.join(REPO, "runs", "CFD_spline", "geometry")

# The converged spline SV-1, minus the fins. Every dimension is shared by both cases.
BASE = dict(
    D=0.35, L_total=3.60, f_nose=3.4, d_base=0.30, L_boattail=0.20,
    m_p_boost=130.0, m_p_sustain=172.0, m_p_terminal=40.0,
    F_boost=45.0e3, F_terminal=8.0e3,
    n_fin=0,
)

# Mesh tolerance for the exported STL, and the two-step build that keeps it affordable.
#
# CLAUDE.md section 4 point 3: `ntopcl convert` EVALUATES the notebook, exports included, and
# `implicit_to_mesh` costs about tolerance^-3. A first attempt at this script set the tolerance
# to 1.0e-4 directly in `measure_rocket`, which made it the notebook input's DEFAULT, which made
# the CONVERT mesh at that tolerance: 15.6 times the default cost, plus 32 section measurements,
# and it timed out at 1800 s having produced nothing.
#
# The repository already solves this. `mesh_tolerance` is a notebook INPUT and is deliberately
# excluded from the topology key, so the notebook can be converted once at a cheap default and
# then RUN at whatever tolerance the job needs, with no re-convert. That is what the two
# constants below are for.
CONVERT_TOLERANCE = 5.0e-4     # coarse, and only ever used by the convert
MESH_TOLERANCE = 1.5e-4        # what the STL is actually written at

# Section-area measurements are expensive and this experiment does not use them: the wave-drag
# comparison is done by CFD, not by the Glauert series. 0 keeps them out of every evaluation.
AREA_STATIONS = 0


def log(m: str) -> None:
    print(f"[cfd-geom] {m}")


def main() -> int:
    from rocketgen.config import DesignVector
    from rocketgen.ntopgen.rocket_notebook import build_rocket_notebook, measure_rocket

    os.makedirs(OUT, exist_ok=True)
    cases = {
        "ogive": DesignVector(**BASE),
        "spline": DesignVector(**BASE, nose_shape="spline", nose_blend=1.0),
    }

    summary = {}
    for name, dv in cases.items():
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        log(f"{name}: nose_shape={dv.nose_shape} blend={getattr(dv, 'nose_blend', None)}")

        # Step 1: convert at the COARSE tolerance. This is the only step the tolerance makes
        # expensive, and the resulting notebook is cached on its topology key.
        t0 = time.perf_counter()
        nb = build_rocket_notebook(
            dv, d, export_stl=True, mesh_tolerance=CONVERT_TOLERANCE,
            area_stations=AREA_STATIONS, convert_timeout=1800.0,
        )
        log(f"  notebook ready in {time.perf_counter() - t0:.0f} s "
            f"({os.path.getsize(nb) / 1e6:.1f} MB)")

        # Step 2: run at the FINE tolerance. Same topology key, so this reuses the notebook
        # above rather than converting again.
        t0 = time.perf_counter()
        m = measure_rocket(dv, d, export_stl=True, mesh_tolerance=MESH_TOLERANCE,
                           area_stations=AREA_STATIONS, timeout=3600.0)
        dt = time.perf_counter() - t0
        stl = m.stl_path if m.stl_path and os.path.isfile(m.stl_path) else None
        log(f"  volume {m.volume_total:.7f} m^3, wetted {m.area_wetted_body:.6f} m^2 "
            f"({dt:.0f} s)")
        if stl:
            log(f"  STL {os.path.basename(stl)} ({os.path.getsize(stl)/1e6:.1f} MB)")
        else:
            log("  NO STL WAS WRITTEN; the CFD cannot proceed from this case")
        summary[name] = dict(
            nose_shape=dv.nose_shape,
            nose_blend=float(getattr(dv, "nose_blend", 0.0) or 0.0),
            volume_total=m.volume_total, area_wetted_body=m.area_wetted_body,
            area_base=m.area_base, stl=stl,
            S_ref=0.25 * math.pi * dv.D ** 2, L_nose=dv.L_nose, D=dv.D,
        )

    # The geometric difference between the two cases, which the CFD result has to be read
    # against: if the volumes differ by more than the nose shape can explain, the two meshes
    # are not the same experiment.
    og, sp = summary["ogive"], summary["spline"]
    if og["volume_total"] and sp["volume_total"]:
        log(f"volume  ogive {og['volume_total']:.7f}  spline {sp['volume_total']:.7f}  "
            f"({(sp['volume_total']/og['volume_total'] - 1) * 100:+.3f} percent)")
    if og["area_wetted_body"] and sp["area_wetted_body"]:
        log(f"wetted  ogive {og['area_wetted_body']:.6f}  spline {sp['area_wetted_body']:.6f}  "
            f"({(sp['area_wetted_body']/og['area_wetted_body'] - 1) * 100:+.3f} percent)")
    log(f"S_ref {og['S_ref']:.6f} m^2 (both), used to normalise every CFD force")

    import json
    with open(os.path.join(OUT, "cases.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {os.path.relpath(os.path.join(OUT, 'cases.json'), REPO)}")
    return 0 if all(v["stl"] for v in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
