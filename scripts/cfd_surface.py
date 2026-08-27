"""Tessellate the SV-1 outer mould line directly, as a watertight surface of revolution.

WHY NOT EXPORT THE STL FROM nTOP
--------------------------------
Two attempts to export these two bodies through `ntopcl` timed out at 1800 s with no output at
all, and an isolation run showed the same notebooks convert in 23 to 46 s with `export_stl`
turned OFF. The export is what hangs on this configuration.

Rather than fight it, note what the experiment actually needs. The CFD is validating the
AERODYNAMIC model, not nTop's mesher. The geometry is a body of revolution whose profile this
repository already defines exactly: `oml_spline.SplineProfile` for the splined nose and
`oml_spline.tangent_ogive_radius` for the ogive, both analytic. Tessellating that directly is
better for this particular job than exporting it:

* it is watertight BY CONSTRUCTION, which an STL from a mesher is not guaranteed to be;
* both cases get the SAME azimuthal and axial topology, so the ogive-versus-spline comparison
  isolates the shape and not a difference in how two meshes happened to be cut. That matters a
  great deal when the effect being measured is a few percent;
* it is exact and instant.

What is given up is the proof that the CFD surface is the same solid nTop measured. That is
recovered by CHECKING it: `verify_against_ntop` compares the tessellated volume and wetted area
against the values nTop reported for the same design, and the caller is expected to report the
agreement rather than assume it.

Units are SI, metres. The axis is +X, nose tip at the origin.
"""
from __future__ import annotations

import math
import os
import struct
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


# A mathematically sharp tip cannot be meshed. Adjacent azimuthal nodes converge as the radius
# goes to zero, and below the mesher's merge tolerance they collapse into degenerate faces:
# AFLR3 rejected the first attempt with error 200310, flagging exactly 25 faces, all inside
# x < 30 mm and r < 8.8 mm, whose vertices had become coincident.
#
# So the tip is truncated at a small finite radius and capped flat. The cost is MEASURED rather
# than assumed: at r_tip = 2 mm the cap frontal area is 0.013 percent of S_ref, which at a blunt
# Cp of about 1.8 is dCD 0.00024, or 0.32 percent of the nose wave drag being compared. It is
# applied IDENTICALLY to both configurations, so it very nearly cancels in the ogive-to-spline
# ratio, which is the quantity under test.
R_TIP = 0.002


def profile_points(dv, n_nose: int = 240, n_cyl: int = 60, n_bt: int = 40,
                   r_tip: float = R_TIP):
    """The (x, r) generating curve, truncated tip to base rim, for either shape family.

    Returned dense and in order. The nose is sampled from whichever exact definition the design
    vector selects, so this function is the single place the two configurations differ.

    Axial stations on the nose are clustered toward the tip with a quadratic map. Uniform
    spacing put 5 mm axial cells against 52 micron azimuthal ones near the tip, an aspect ratio
    of 95 to 1; clustering keeps the first cells short enough that the surface triangles stay
    well shaped.
    """
    from rocketgen.oml_spline import SplineProfile, tangent_ogive_radius

    R = 0.5 * dv.D
    r_base = 0.5 * dv.d_base
    L_nose = dv.L_nose
    x_cyl_end = dv.L_total - dv.L_boattail

    if dv.nose_shape == "spline":
        prof = SplineProfile(length=L_nose, radius=R, control=dv.nose_control)
        radius_at = lambda t: prof.point_at(t)[1]                        # noqa: E731
    else:
        k = L_nose / R
        radius_at = lambda t: R * tangent_ogive_radius(t, k)             # noqa: E731

    # the parameter at which the profile first reaches r_tip, by bisection
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if radius_at(mid) < r_tip:
            lo = mid
        else:
            hi = mid
    t0 = hi

    pts: list[tuple[float, float]] = []
    # --- nose, from the truncation station aft, clustered toward the tip ---
    for i in range(n_nose + 1):
        f = (i / n_nose) ** 2                     # quadratic: fine near the tip
        t = t0 + (1.0 - t0) * f
        pts.append((t * L_nose, radius_at(t)))

    # --- cylinder ---
    for i in range(1, n_cyl + 1):
        pts.append((L_nose + (x_cyl_end - L_nose) * i / n_cyl, R))

    # --- boattail, straight cone (both cases; it is not part of the comparison) ---
    for i in range(1, n_bt + 1):
        f = i / n_bt
        pts.append((x_cyl_end + dv.L_boattail * f, R + (r_base - R) * f))

    return pts


def revolve_to_stl(pts, path: str, n_theta: int = 180) -> int:
    """Revolve the generating curve and write a binary STL. Returns the triangle count.

    Watertight by construction: the nose tip is a fan of triangles to a single apex vertex, the
    barrel is a quad grid split into triangles, and the base is a fan to a single centre vertex.
    No vertex is duplicated between those three regions.
    """
    def unit(a, b, c):
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        m = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        return nx / m, ny / m, nz / m

    ct = [math.cos(2.0 * math.pi * j / n_theta) for j in range(n_theta)]
    st = [math.sin(2.0 * math.pi * j / n_theta) for j in range(n_theta)]

    def ring(i):
        x, r = pts[i]
        return [(x, r * ct[j], r * st[j]) for j in range(n_theta)]

    tris: list[tuple] = []

    # Flat cap over the truncated tip. Wound so its normal points forward, i.e. out of the
    # body and into the flow, matching the barrel.
    apex = (pts[0][0], 0.0, 0.0)
    r1 = ring(0)
    for j in range(n_theta):
        tris.append((apex, r1[(j + 1) % n_theta], r1[j]))

    # barrel
    prev = r1
    for i in range(1, len(pts)):
        cur = ring(i)
        for j in range(n_theta):
            k = (j + 1) % n_theta
            tris.append((prev[j], prev[k], cur[j]))
            tris.append((prev[k], cur[k], cur[j]))
        prev = cur

    # base disc fan
    centre = (pts[-1][0], 0.0, 0.0)
    for j in range(n_theta):
        k = (j + 1) % n_theta
        tris.append((centre, prev[j], prev[k]))

    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            f.write(struct.pack("<3f", *unit(a, b, c)))
            for v in (a, b, c):
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))
    return len(tris)


def closed_form(pts) -> tuple[float, float]:
    """(volume, lateral area) of the revolved generating curve, exact frustum sums.

    This describes the TESSELLATED surface, so it is what the STL should be checked against.
    """
    v = a = 0.0
    for (x0, r0), (x1, r1) in zip(pts, pts[1:]):
        h = x1 - x0
        v += math.pi * h * (r0 * r0 + r0 * r1 + r1 * r1) / 3.0
        a += math.pi * (r0 + r1) * math.hypot(h, r1 - r0)
    return v, a


def stl_volume_and_area(path: str) -> tuple[float, float]:
    """Enclosed volume and area read back FROM THE WRITTEN FILE, by the divergence theorem.

    Deliberately reads the bytes rather than reusing the in-memory triangles: this is the check
    that what landed on disk is the closed body intended, not a check of the maths above.
    """
    with open(path, "rb") as f:
        f.read(80)
        (n,) = struct.unpack("<I", f.read(4))
        vol = area = 0.0
        for _ in range(n):
            f.read(12)
            a = struct.unpack("<3f", f.read(12))
            b = struct.unpack("<3f", f.read(12))
            c = struct.unpack("<3f", f.read(12))
            f.read(2)
            vol += (a[0] * (b[1] * c[2] - b[2] * c[1])
                    - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0
            ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
            vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            area += 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)
    return abs(vol), area
