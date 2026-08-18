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


def regenerate_notebook(dest_dir: str) -> bool:
    """Convert a fresh IV-1 notebook whose export paths are neutral. True on success.

    REGENERATED, never copied from `runs/_ntop_cache/`. CLAUDE.md section 11: a converted
    notebook bakes its export destinations in as absolute `file_path` literals, so the cached
    copy carries the developer's directory layout INSIDE THE BINARY, where no text scrub will
    ever find it. Rebuilding with a relative export directory keeps that out of the artefact.

    This is also the most valuable single file in the example: it is the parametric model, and
    a reader can open it in nTop and move the design variables. Shipping the measurements
    without it would be shipping the answer without the question.
    """
    try:
        from rocketgen.config_iv1 import default_iv1
        from rocketgen.ntopgen.driver import NtopRunner
        from rocketgen.ntopgen.stack_notebook import build_stack_recipe

        conv = json.load(open(os.path.join(SPLINE, "converged.json"), encoding="utf-8"))
        raw = conv.get("design_vector") or {}
        dv = default_iv1().replace(
            nose_shape=raw.get("nose_shape", "spline"),
            nose_blend=float(raw.get("nose_blend", 1.0)),
            interstage_shape=raw.get("interstage_shape", "spline"),
            interstage_blend=float(raw.get("interstage_blend", 1.0)),
        )

        # "exports" is RELATIVE, so the literals baked into the binary say nothing about this
        # machine. An nTop user retargets them from the notebook inputs anyway.
        recipe = build_stack_recipe(dv, "exports", export_stl=True, area_stations=16)
        recipe_path = os.path.join(dest_dir, "iv1_recipe.json")
        recipe.write_json(recipe_path)
        with open(recipe_path, "r", encoding="utf-8") as f:
            body = f.read()
        with open(recipe_path, "w", encoding="utf-8") as f:
            f.write(scrub_text(body))
        log(f"wrote {os.path.relpath(recipe_path, REPO)}")

        runner = NtopRunner()
        out = os.path.join(dest_dir, "iv1.ntop")
        runner.convert(recipe_path, out, timeout=1800)
        log(f"regenerated iv1.ntop with neutral export paths "
            f"({os.path.getsize(out)/1e6:.1f} MB)")

        # The driver writes a convert log beside the output. It records the full command line,
        # so it carries this machine's paths and is of no value to a reader.
        for junk in ("iv1_convert.log", "ntopcl_convert.log", "ntopcl_run.log"):
            jp = os.path.join(dest_dir, junk)
            if os.path.isfile(jp):
                os.remove(jp)
        return True
    except Exception as exc:  # noqa: BLE001
        log(f"could not regenerate the notebook: {type(exc).__name__}: {exc}")
        return False


def verify_scrub(root: str) -> int:
    """Search EVERY written file, text and binary alike.

    Binaries are the point. A `.ntop` is a binary container that bakes absolute export paths
    into itself, so it is exactly the file most likely to carry a developer path and the one a
    text-only check would skip. An earlier version of this function skipped every extension it
    did not recognise, which meant the notebook was never checked at all.

    The needles are MACHINE-IDENTIFYING TOKENS, not path punctuation. A drive-letter pattern
    like "D:/" fires constantly on packed float data - "Y@L:/>Q" and "6@c:/>" are real hits
    from a real .ntop - and a scanner that cries wolf on every binary is a scanner nobody
    reads. These tokens cannot occur by chance.

    A relative "runs/_ntop_cache/..." is deliberately NOT a needle: it points inside the repo
    and names no machine.

    NEVER trust the copy step's own success message (CLAUDE.md section 6).
    """
    bad = 0
    needles = (b"nTop_Suave", b"bradrothenberg", b"cplusplus", b"worktrees",
               b"AppData", b"Users", b"Program Files")
    for r, _, names in os.walk(root):
        for n in names:
            p = os.path.join(r, n)
            with open(p, "rb") as f:
                body = f.read()
            for needle in needles:
                if needle in body:
                    log(f"SCRUB FAILED: {needle!r} still in {os.path.relpath(p, REPO)}")
                    bad += 1
                # a .ntop stores some literals as UTF-16, which an ASCII byte search misses
                if needle.decode("ascii").encode("utf-16-le") in body:
                    log(f"SCRUB FAILED (utf-16): {needle!r} in {os.path.relpath(p, REPO)}")
                    bad += 1
    return bad


def main() -> int:
    if not os.path.isdir(SPLINE):
        log(f"no spline run at {SPLINE}")
        return 1
    # Rebuild from scratch so a stale artefact cannot survive, but PRESERVE files this script
    # does not itself write. The README is hand-authored and lives here; an earlier version of
    # this function deleted the whole tree and silently destroyed it, which was only noticed
    # because someone went looking for it. A build step may replace what it produces. It has no
    # business deleting what it does not.
    keep = {}
    for name in ("README.md",):
        p = os.path.join(EX, name)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                keep[name] = f.read()
    if os.path.isdir(EX):
        shutil.rmtree(EX)
    os.makedirs(EX, exist_ok=True)
    for name, body in keep.items():
        with open(os.path.join(EX, name), "wb") as f:
            f.write(body)
        log(f"preserved {name}")
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

    # geometry: the measurement JSON the stack notebook wrote...
    geom = os.path.join(SPLINE, "geom")
    if os.path.isdir(geom):
        for n in sorted(os.listdir(geom)):
            if n.endswith((".stl", ".json")) and not n.startswith("input"):
                copy_scrubbed(os.path.join(geom, n), d_geom)
    # ...and the notebook itself, REGENERATED. It lives in `runs/_ntop_cache/`, not beside the
    # measurements, so a glob over the run directory silently ships an example with no
    # parametric model in it. That is what happened on the first pass here.
    regenerate_notebook(d_geom)

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
