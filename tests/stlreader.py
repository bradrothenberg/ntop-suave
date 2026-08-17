"""Minimal STL reader plus mesh volume, for validating nTop exports without extra deps.

Binary and ASCII STL. `enclosed_volume` uses the signed-tetrahedron sum, which is exact for a
closed oriented triangle mesh:

    V = (1/6) * sum over triangles of  a . (b x c)
"""
from __future__ import annotations

import os
import struct
from typing import Iterator

__all__ = ["read_stl", "enclosed_volume", "surface_area", "bounding_box"]

Triangle = tuple[
    tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
]


def read_stl(path: str | os.PathLike[str]) -> list[Triangle]:
    """Read an STL file. Detects binary versus ASCII from the content, not the extension."""
    with open(path, "rb") as f:
        raw = f.read()
    if not raw:
        raise ValueError(f"empty STL: {path}")
    if _looks_binary(raw):
        return list(_read_binary(raw))
    return list(_read_ascii(raw.decode("utf-8", "replace")))


def _looks_binary(raw: bytes) -> bool:
    """A binary STL is exactly 84 + 50*N bytes and its header does not start with 'solid'."""
    if len(raw) < 84:
        return False
    head = raw[:5].lower()
    (count,) = struct.unpack_from("<I", raw, 80)
    exact = len(raw) == 84 + 50 * count
    if head == b"solid" and not exact:
        return False
    return exact


def _read_binary(raw: bytes) -> Iterator[Triangle]:
    (count,) = struct.unpack_from("<I", raw, 80)
    off = 84
    for _ in range(count):
        vals = struct.unpack_from("<12fH", raw, off)
        off += 50
        yield (
            (vals[3], vals[4], vals[5]),
            (vals[6], vals[7], vals[8]),
            (vals[9], vals[10], vals[11]),
        )


def _read_ascii(text: str) -> Iterator[Triangle]:
    verts: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] != "vertex" or len(parts) < 4:
            continue
        verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        if len(verts) == 3:
            yield (verts[0], verts[1], verts[2])
            verts = []


def enclosed_volume(tris: list[Triangle]) -> float:
    """Signed volume enclosed by a closed oriented mesh, in the mesh's own length units cubed."""
    total = 0.0
    for a, b, c in tris:
        total += (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
    return abs(total) / 6.0


def surface_area(tris: list[Triangle]) -> float:
    total = 0.0
    for a, b, c in tris:
        u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        cx = u[1] * v[2] - u[2] * v[1]
        cy = u[2] * v[0] - u[0] * v[2]
        cz = u[0] * v[1] - u[1] * v[0]
        total += 0.5 * (cx * cx + cy * cy + cz * cz) ** 0.5
    return total


def bounding_box(tris: list[Triangle]) -> tuple[tuple[float, float, float],
                                                tuple[float, float, float]]:
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for tri in tris:
        for v in tri:
            for i in range(3):
                lo[i] = min(lo[i], v[i])
                hi[i] = max(hi[i], v[i])
    return (lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])
