"""Export and render the converged IV-1 spline stack.

CLAUDE.md section 10.1: a report without a picture of the thing it is about is not finished. The
first IV-1 spline report shipped without a vehicle render because neither IV-1 run had enabled
mesh export, and by the time anyone looked the runs had finished. This closes that.

    .venv/Scripts/python.exe scripts/render_iv1_spline.py

Exports are ON here and only here. `measure_stack` keeps them off by default because they cost
minutes, so they are paid for once, at the converged point, which is the only geometry that ships.
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

OUT = os.path.join(REPO, "runs", "IV-1_spline", "render")
FIGS = os.path.join(REPO, "examples", "IV-1-spline", "figures")


def log(m: str) -> None:
    print(f"[render] {m}")


def converged_design():
    """The design the report describes, read back from its own result file."""
    from rocketgen.config_iv1 import default_iv1

    path = os.path.join(REPO, "runs", "IV-1_spline", "converged.json")
    raw = json.load(open(path, encoding="utf-8")).get("design_vector") or {}
    dv = default_iv1().replace(
        nose_shape=raw.get("nose_shape", "spline"),
        nose_blend=float(raw.get("nose_blend", 1.0)),
        interstage_shape=raw.get("interstage_shape", "spline"),
        interstage_blend=float(raw.get("interstage_blend", 1.0)),
    )
    if raw.get("gamma_pitch") is not None:
        dv = dv.replace(gamma_pitch=float(raw["gamma_pitch"]))
    log(f"nose {dv.nose_shape} blend {dv.nose_blend}, "
        f"interstage {dv.interstage_shape} blend {dv.interstage_blend}")
    return dv


def export_stl(dv, reuse: bool = True) -> str | None:
    from rocketgen.ntopgen.stack_notebook import measure_stack

    os.makedirs(OUT, exist_ok=True)
    if reuse:
        # The export costs minutes of nTop. Reuse it when only the CAMERA is being adjusted,
        # which is the common case: getting a view framed right takes several attempts and none
        # of them need new geometry. Pass --re-export to force a fresh measurement.
        existing = [os.path.join(OUT, n) for n in sorted(os.listdir(OUT))
                    if n.endswith(".stl")]
        if existing:
            biggest = max(existing, key=os.path.getsize)
            log(f"reusing {os.path.relpath(biggest, REPO)} "
                f"({os.path.getsize(biggest)/1e6:.1f} MB); pass --re-export to rebuild it")
            return biggest
    log("measuring with exports ON; this converts a new topology and takes minutes")
    got = measure_stack(dv, OUT, export_stl=True, mesh_tolerance=1.5e-3, timeout=3600.0)
    for key in sorted(got):
        m = got[key]
        if m.stl_path and os.path.isfile(m.stl_path):
            log(f"stage {key} STL: {os.path.basename(m.stl_path)} "
                f"({os.path.getsize(m.stl_path)/1e6:.1f} MB)")
    # the stack union is key 0
    stl = None
    for key in (0, -1, 1, 2):
        m = got.get(key)
        if m is not None and m.stl_path and os.path.isfile(m.stl_path):
            stl = m.stl_path
            break
    if stl is None:
        for n in sorted(os.listdir(OUT)):
            if n.endswith(".stl"):
                stl = os.path.join(OUT, n)
                break
    return stl


def render(stl_path: str) -> list[str]:
    """Isometric and side views. Returns the PNGs written."""
    import pyvista as pv

    pv.global_theme.allow_empty_mesh = True
    mesh = pv.read(stl_path)
    log(f"mesh: {mesh.n_cells} cells, bounds "
        f"{[round(b, 3) for b in mesh.bounds]}")
    os.makedirs(FIGS, exist_ok=True)

    written = []
    views = (
        ("iv1_spline_iso", (1.0, -1.6, 0.75), (1500, 560), "isometric"),
        ("iv1_spline_side", (0.0, -1.0, 0.0), (1800, 330), "side"),
    )
    for name, direction, size, label in views:
        pl = pv.Plotter(off_screen=True, window_size=size)
        pl.add_mesh(mesh, color="#b9c2cc", specular=0.35, specular_power=18,
                    smooth_shading=True)
        pl.set_background("white")
        pl.enable_parallel_projection()
        span = max(mesh.length, 1.0)
        pl.camera_position = [
            tuple(c + d * span for c, d in zip(mesh.center, direction)),
            mesh.center,
            (0.0, 0.0, 1.0),
        ]
        pl.camera.parallel_scale = _parallel_scale(mesh, pl.camera, size, pad=1.06)
        out = os.path.join(FIGS, f"{name}.png")
        pl.screenshot(out)
        pl.close()
        log(f"wrote {os.path.relpath(out, REPO)} ({label}, {size[0]}x{size[1]})")
        written.append(out)
    return written


def _parallel_scale(mesh, camera, size, pad: float = 1.06) -> float:
    """The half-height that just fits the mesh in this window, under parallel projection.

    Computed from the eight bounding-box corners PROJECTED onto the camera's own right and up
    axes, so it is correct for an oblique view as well as an axis-aligned one, and it does not
    depend on the vehicle's size.

    This replaced a hand-tuned `zoom()`. On a 5.08 m body a fixed zoom of 1.5 produced a
    close-up of the middle of the booster, which passed a file-size check and was useless as a
    figure. Framing has to be derived from the geometry, not guessed.
    """
    import numpy as np

    pos = np.array(camera.position, float)
    foc = np.array(camera.focal_point, float)
    up = np.array(camera.up, float)

    forward = foc - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    corners = np.array([[x, y, z] for x in (xmin, xmax)
                        for y in (ymin, ymax) for z in (zmin, zmax)], float)
    rel = corners - foc
    half_w = np.abs(rel @ right).max()
    half_h = np.abs(rel @ true_up).max()

    aspect = size[0] / size[1]
    return pad * max(half_h, half_w / aspect)


def main() -> int:
    reuse = "--re-export" not in sys.argv
    dv = converged_design()
    stl = export_stl(dv, reuse=reuse)
    if stl is None:
        log("no STL was produced; cannot render. Say so in the report rather than omitting it.")
        return 1
    log(f"rendering from {os.path.relpath(stl, REPO)}")
    pngs = render(stl)

    # Verify by counting INK, not bytes. A file-size threshold cannot tell an empty render from
    # a correct one of a slender body on white: the first side view here was geometrically
    # perfect and compressed to 18 KB, and the size check called it empty. Measure the thing
    # that actually matters, which is how much of the frame the vehicle covers.
    bad = []
    for p in pngs:
        frac = _ink_fraction(p)
        log(f"  {os.path.basename(p)}: {frac * 100:.1f} percent of the frame is vehicle")
        if frac < 0.02:
            bad.append(f"{os.path.basename(p)} (nearly blank)")
        elif frac > 0.85:
            bad.append(f"{os.path.basename(p)} (clipped, fills the frame)")
    if bad:
        log(f"renders are not usable: {bad}")
        return 1
    log("done; both renders written, framed and verified by ink coverage")
    return 0


def _ink_fraction(path: str) -> float:
    """Fraction of pixels that are not the white background."""
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(path).convert("L"), dtype=np.int16)
    return float((a < 245).mean())


if __name__ == "__main__":
    raise SystemExit(main())
