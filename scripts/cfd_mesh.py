"""Build the AFLR3 volume meshes for the ogive-versus-spline CFD validation.

    .venv/Scripts/python.exe scripts/cfd_mesh.py [--case ogive|spline|both]

INVISCID, ON PURPOSE
--------------------
`--num-layers 0`: no boundary-layer prisms. The experiment compares PRESSURE drag between two
nose shapes. Skin friction is roughly a quarter of this vehicle's CD0 and the two shapes differ
in wetted area by 0.67 percent, so resolving a boundary layer would add cost and a turbulence
model's uncertainty to a comparison that does not need either. What Euler reports is wave plus
base drag, which is exactly what `sizing/wavedrag.py` claims to predict.

FARFIELD
--------
A sphere of radius 20 m, about 5.6 body lengths, centred on the body mid-point. Large enough
that the bow shock reaches it well aft of the vehicle at the Mach numbers of interest, and small
enough to keep the tet count affordable for six solves.

Both cases use the SAME farfield and the SAME AFLR3 settings. Only the body surface differs.
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

GEOM = os.path.join(REPO, "runs", "CFD_spline", "geometry")
MESHDIR = os.path.join(REPO, "runs", "CFD_spline", "mesh")
AFLR3 = os.path.join("D:", os.sep, "cplusplus", "ntop", "AFLR3", "aflr3_cfd_mesh.py")

FARFIELD_RADIUS = 20.0
BODY_LENGTH = 3.60


def log(m: str) -> None:
    print(f"[cfd-mesh] {m}")


def write_sphere(path: str, radius: float, centre, n_lat: int = 48, n_lon: int = 96) -> int:
    """A closed UV sphere as binary STL, normals pointing OUTWARD from the centre.

    The farfield is a boundary of the fluid domain, so AFLR3 only needs it closed and
    consistently oriented; the tessellation is coarse because nothing is resolved on it.
    """
    cx, cy, cz = centre

    def v(i, j):
        phi = math.pi * i / n_lat
        th = 2.0 * math.pi * j / n_lon
        return (cx + radius * math.cos(phi),
                cy + radius * math.sin(phi) * math.cos(th),
                cz + radius * math.sin(phi) * math.sin(th))

    tris = []
    for i in range(n_lat):
        for j in range(n_lon):
            k = (j + 1) % n_lon
            a, b, c, d = v(i, j), v(i, k), v(i + 1, k), v(i + 1, j)
            if i == 0:
                tris.append((a, c, d))
            elif i == n_lat - 1:
                tris.append((a, b, c))
            else:
                tris.append((a, b, c))
                tris.append((a, c, d))

    def unit(a, b, c):
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        m = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        return nx / m, ny / m, nz / m

    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            f.write(struct.pack("<3f", *unit(a, b, c)))
            for p in (a, b, c):
                f.write(struct.pack("<3f", *p))
            f.write(struct.pack("<H", 0))
    return len(tris)


def mesh_case(name: str, timeout: float) -> bool:
    body = os.path.join(GEOM, f"sv1_{name}.stl")
    if not os.path.isfile(body):
        log(f"{name}: no body STL at {body}; run scripts/cfd_surface driver first")
        return False
    ff = os.path.join(MESHDIR, "farfield.stl")
    out = os.path.join(MESHDIR, name, f"sv1_{name}")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    cmd = [
        sys.executable, AFLR3, body, ff, "-o", out,
        "--num-layers", "0",            # inviscid: no BL prisms
        "--angqbf", "180", "--angqbfmin", "0",
        "--format", "b8.ugrid",
        "--no-nml",                     # the namelist is written by cfd_solve.py instead
    ]
    log(f"{name}: {' '.join(os.path.basename(c) for c in cmd[:4])} ...")
    t0 = time.perf_counter()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"{name}: AFLR3 TIMED OUT after {timeout:.0f} s")
        return False
    dt = time.perf_counter() - t0
    grid = out + ".b8.ugrid"
    ok = os.path.isfile(grid) and os.path.getsize(grid) > 0
    log(f"{name}: rc={res.returncode} in {dt:.0f} s, grid {'OK' if ok else 'MISSING'}"
        + (f" ({os.path.getsize(grid)/1e6:.0f} MB)" if ok else ""))
    if not ok:
        tail = (res.stdout or "").strip().splitlines()[-12:]
        for line in tail:
            log(f"    {line}")
        err = (res.stderr or "").strip().splitlines()[-5:]
        for line in err:
            log(f"  ! {line}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="both", choices=["ogive", "spline", "both"])
    ap.add_argument("--timeout", type=float, default=2400.0)
    args = ap.parse_args()

    os.makedirs(MESHDIR, exist_ok=True)
    ff = os.path.join(MESHDIR, "farfield.stl")
    n = write_sphere(ff, FARFIELD_RADIUS, (0.5 * BODY_LENGTH, 0.0, 0.0))
    log(f"farfield sphere r={FARFIELD_RADIUS} m, {n} triangles, "
        f"{os.path.getsize(ff)/1e6:.1f} MB")

    cases = ["ogive", "spline"] if args.case == "both" else [args.case]
    ok = True
    for c in cases:
        ok &= mesh_case(c, args.timeout)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
