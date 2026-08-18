"""Curate the IV-1 spline study into `examples/IV-1-spline/`.

`runs/` is gitignored (CLAUDE.md section 11), so nothing the analysis produces ships unless it
is copied here. This is the IV-1 counterpart of `scripts/build_example.py`, and it curates BOTH
the spline result and the ogive baseline it is compared against, because a comparison whose
baseline is not in the repository cannot be checked by a reader.

    .venv/Scripts/python.exe scripts/build_example_iv1_spline.py

Every path written into a text artefact is scrubbed of developer absolute paths, and the scrub
is verified with a search over the written files rather than trusted from a success message.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

RUNS = os.path.join(REPO, "runs")
SPLINE = os.path.join(RUNS, "IV-1_spline")
OGIVE = os.path.join(RUNS, "IV-1_ogive_baseline")
EX = os.path.join(REPO, "examples", "IV-1-spline")

# Anything that looks like a developer path must not ship.
#
# The JSON-ESCAPED form matters and is easy to miss: a path inside a JSON string carries
# DOUBLED backslashes, so a pattern built from the real path does not match it. That exact miss
# happened here and was caught by `verify_scrub`, which is precisely why the verification
# searches the written files instead of trusting the copy step (CLAUDE.md section 6).
#
# One pattern covers both forms by treating a separator as "one or two backslashes, or a
# forward slash".
_SEP = r"(?:\\\\|\\|/)"
_WINDOWS_PATH = r"[A-Za-z]:" + _SEP + r"(?:[^\"'\s\\/]+" + _SEP + r")*[^\"'\s\\/]*"
SCRUB = [
    (re.compile(_WINDOWS_PATH), "<path>"),
]


def log(m: str) -> None:
    print(f"[iv1-example] {m}")


def scrub_text(s: str) -> str:
    for pat, rep in SCRUB:
        s = pat.sub(rep, s)
    return s


def copy_scrubbed(src: str, dst_dir: str, name: str | None = None) -> str | None:
    if not os.path.isfile(src):
        log(f"MISSING, skipped: {os.path.relpath(src, REPO)}")
        return None
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, name or os.path.basename(src))
    if os.path.splitext(src)[1].lower() in (".json", ".csv", ".md", ".txt"):
        with open(src, "r", encoding="utf-8") as f:
            body = f.read()
        with open(dst, "w", encoding="utf-8") as f:
            f.write(scrub_text(body))
    else:
        shutil.copy2(src, dst)
    log(f"wrote {os.path.relpath(dst, REPO)}")
    return dst


def flatten_constraints(conv: dict, path: str) -> None:
    """A reader opens a spreadsheet before they open a JSON."""
    rows = conv.get("constraints") or []
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["requirement", "value", "limit", "met"])
        for r in rows:
            if isinstance(r, dict):
                w.writerow([r.get("name"), r.get("value"), r.get("limit"), r.get("met")])
            elif isinstance(r, (list, tuple)) and len(r) >= 4:
                w.writerow([r[0], r[1], r[2], r[-1]])
    log(f"wrote {os.path.relpath(path, REPO)}")


def comparison_csv(og: dict, sp: dict, path: str) -> None:
    """The whole point of the example: what the shape change did, side by side."""
    def ic(d, k):
        return (d.get("intercept") or {}).get(k)

    rows = [
        ("launch mass", "kg", og.get("launch_mass_kg"), sp.get("launch_mass_kg")),
        ("slant range", "m", ic(og, "slant_range"), ic(sp, "slant_range")),
        ("intercept altitude", "m", ic(og, "altitude"), ic(sp, "altitude")),
        ("intercept Mach", "-", ic(og, "mach"), ic(sp, "mach")),
        ("time to intercept", "s", ic(og, "time"), ic(sp, "time")),
        ("pitchover angle", "deg", og.get("pitchover_deg"), sp.get("pitchover_deg")),
        ("lateral g, aerodynamic", "g",
         (og.get("lateral_g") or {}).get("aerodynamic"),
         (sp.get("lateral_g") or {}).get("aerodynamic")),
        ("ACS thrust", "N",
         (og.get("acs") or {}).get("thrust"), (sp.get("acs") or {}).get("thrust")),
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "unit", "tangent ogive", "spline", "delta", "delta percent"])
        for name, unit, a, b in rows:
            if a is None or b is None:
                continue
            d = b - a
            pc = (d / a * 100.0) if a else float("nan")
            w.writerow([name, unit, f"{a:.6g}", f"{b:.6g}", f"{d:+.6g}", f"{pc:+.3f}"])
    log(f"wrote {os.path.relpath(path, REPO)}")


def verify_scrub(root: str) -> int:
    """Search the written files. NEVER trust the copy step's own success message."""
    bad = 0
    needles = ("nTop_Suave", "bradrothenberg", "C:\\Users", "D:\\cplusplus")
    for r, _, names in os.walk(root):
        for n in names:
            if os.path.splitext(n)[1].lower() not in (".json", ".csv", ".md", ".txt"):
                continue
            p = os.path.join(r, n)
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                body = f.read()
            for needle in needles:
                if needle in body:
                    log(f"SCRUB FAILED: {needle!r} still in {os.path.relpath(p, REPO)}")
                    bad += 1
    return bad


def main() -> int:
    if not os.path.isdir(SPLINE):
        log(f"no spline run at {SPLINE}")
        return 1
    if os.path.isdir(EX):
        shutil.rmtree(EX)
    d_design = os.path.join(EX, "01_design")
    d_geom = os.path.join(EX, "02_geometry")
    d_base = os.path.join(EX, "03_ogive_baseline")
    for d in (d_design, d_geom, d_base):
        os.makedirs(d, exist_ok=True)

    copy_scrubbed(os.path.join(SPLINE, "converged.json"), d_design)
    copy_scrubbed(os.path.join(SPLINE, "trajectory.png"), d_design)
    copy_scrubbed(os.path.join(OGIVE, "converged.json"), d_base)
    copy_scrubbed(os.path.join(OGIVE, "trajectory.png"), d_base)

    with open(os.path.join(SPLINE, "converged.json"), encoding="utf-8") as f:
        sp = json.load(f)
    og = None
    if os.path.isfile(os.path.join(OGIVE, "converged.json")):
        with open(os.path.join(OGIVE, "converged.json"), encoding="utf-8") as f:
            og = json.load(f)

    flatten_constraints(sp, os.path.join(d_design, "constraints.csv"))
    if og is not None:
        comparison_csv(og, sp, os.path.join(EX, "ogive_vs_spline.csv"))

    # geometry: whatever the last stack measurement wrote
    geom = os.path.join(SPLINE, "geom")
    if os.path.isdir(geom):
        for n in sorted(os.listdir(geom)):
            if n.endswith((".stl", ".ntop", ".json")) and not n.startswith("input"):
                copy_scrubbed(os.path.join(geom, n), d_geom)

    bad = verify_scrub(EX)
    total = sum(os.path.getsize(os.path.join(r, n))
                for r, _, ns in os.walk(EX) for n in ns)
    log(f"done. {os.path.relpath(EX, REPO)} is {total/1e6:.1f} MB")
    if bad:
        log(f"{bad} scrub failures: NOT safe to commit")
        return 1
    log("scrub verified by search over the written files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
