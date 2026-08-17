"""One-time setup: fetch SUAVE and locate the nTop block universe.

    uv run --python .venv/Scripts/python.exe scripts/bootstrap.py

Neither of the two external dependencies is committed to this repository, for different reasons:

- **SUAVE** is LGPL 2.1. It is fetched from its own repository so this repo does not redistribute
  it and does not inherit its licence obligations.
- **The nTop block universe** (`functions.json`, `types.json`, `type_defaults.json`) is a bulk
  export of nTop's block and type API surface. It belongs to nTop, it is version-specific, and it
  contains blocks that are not publicly released. This script finds it on your machine. It does
  not download it from here, because it is not here.

Run with `--check` to verify an existing setup without changing anything.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(REPO, "vendor")

SUAVE_REPO = "https://github.com/suavecode/SUAVE.git"
SUAVE_TAG = "master"          # 2.5.2 is the head of master as of Mar 2022
UNIVERSE_FILES = ("functions.json", "types.json", "type_defaults.json")

# Where to look for the block universe.
#
# Set NTOP_UNIVERSE_DIR to the directory holding the three JSON files. The fallbacks are an
# unpacked nTopDocsIntermediates archive from an nTop build, and the repo's own vendor directory
# if you copied them there by hand. An nTop source checkout also keeps a copy under
# scripts/notebook_author/tests, if you have one.
UNIVERSE_SEARCH = [
    os.environ.get("NTOP_UNIVERSE_DIR", ""),
    os.path.join(os.path.expanduser("~"), "Downloads", "nTopDocsIntermediates"),
    os.path.join(os.path.expanduser("~"), "nTopDocsIntermediates"),
]


def log(msg: str) -> None:
    print(f"[bootstrap] {msg}")


# --------------------------------------------------------------------------------------
#   SUAVE
# --------------------------------------------------------------------------------------


def suave_present() -> bool:
    return os.path.isfile(os.path.join(VENDOR, "SUAVE", "__init__.py"))


def fetch_suave() -> None:
    """Shallow-clone SUAVE and move its package directory into vendor/."""
    if suave_present():
        log("SUAVE already present, skipping")
        return
    os.makedirs(VENDOR, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        log(f"cloning {SUAVE_REPO} (shallow)")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", SUAVE_TAG, SUAVE_REPO, tmp],
            check=True,
            capture_output=True,
        )
        src = os.path.join(tmp, "trunk", "SUAVE")
        if not os.path.isdir(src):
            raise RuntimeError(f"SUAVE layout changed: expected {src}")
        shutil.copytree(src, os.path.join(VENDOR, "SUAVE"))
        # Keep the licence next to the code we vendor locally, as LGPL 2.1 requires.
        for name in ("LICENSE", "README.md"):
            p = os.path.join(tmp, name)
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(VENDOR, f"SUAVE_{name}"))
    log("SUAVE fetched into vendor/SUAVE")

    # SUAVE/version.py is normally written by setup.py. We do not run setup.py, so write it.
    vp = os.path.join(VENDOR, "SUAVE", "version.py")
    if not os.path.isfile(vp):
        with open(vp, "w", encoding="utf-8") as f:
            f.write("\n# Written by scripts/bootstrap.py; setup.py normally generates this.\nversion = '2.5.2'\n")
        log("wrote vendor/SUAVE/version.py")


# --------------------------------------------------------------------------------------
#   nTop block universe
# --------------------------------------------------------------------------------------


def universe_present() -> bool:
    return all(os.path.isfile(os.path.join(VENDOR, n)) for n in UNIVERSE_FILES)


def find_universe() -> str | None:
    for base in UNIVERSE_SEARCH:
        if not base:
            continue
        if all(os.path.isfile(os.path.join(base, n)) for n in UNIVERSE_FILES):
            return base
    # last resort: a recursive search under any nTop checkout on the same drive is too slow,
    # so try a narrow glob instead
    for pat in (
        r"D:\**\notebook_author\tests\functions.json",
        r"C:\**\notebook_author\tests\functions.json",
    ):
        hits = glob.glob(pat, recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    return None


def link_universe() -> bool:
    if universe_present():
        log("block universe already present, skipping")
        return True
    src = find_universe()
    if src is None:
        log("BLOCK UNIVERSE NOT FOUND. The nTop notebook authoring path will not work.")
        log("  Provide it in one of these ways:")
        log("    1. set NTOP_UNIVERSE_DIR to a directory holding the three JSON files, or")
        log("    2. copy functions.json, types.json and type_defaults.json into vendor/")
        log("  They come from an nTop source checkout at")
        log("  scripts/notebook_author/tests, or from the nTopDocsIntermediates build archive.")
        log("  Everything else in this repo (aero, propulsion, trajectory, sizing) works without")
        log("  them; only geometry generation needs them.")
        return False
    os.makedirs(VENDOR, exist_ok=True)
    for n in UNIVERSE_FILES:
        shutil.copy2(os.path.join(src, n), os.path.join(VENDOR, n))
    log(f"copied the block universe from {src}")
    return True


# --------------------------------------------------------------------------------------
#   Verification
# --------------------------------------------------------------------------------------


def check() -> int:
    ok = True

    if suave_present():
        sys.path.insert(0, VENDOR)
        try:
            import SUAVE  # noqa: F401

            log(f"SUAVE imports: OK ({SUAVE.__version__})")
        except Exception as exc:                          # noqa: BLE001
            log(f"SUAVE present but does NOT import: {type(exc).__name__}: {exc}")
            log("  numpy must be < 2, scipy < 1.14, setuptools < 81. See pyproject.toml.")
            ok = False
    else:
        log("SUAVE missing. Run without --check to fetch it.")
        ok = False

    if universe_present():
        with open(os.path.join(VENDOR, "functions.json"), encoding="utf-8") as f:
            n = len(json.load(f))
        log(f"block universe: OK ({n} blocks)")
    else:
        log("block universe missing. Geometry generation will not work.")
        ok = False

    ntopcl = os.environ.get("NTOPCL", r"C:/Program Files/nTopology/nTopology/ntopcl.exe")
    if os.path.isfile(ntopcl):
        try:
            out = subprocess.run([ntopcl, "--version"], capture_output=True, timeout=60)
            log(f"ntopcl: OK ({out.stdout.decode(errors='replace').strip().splitlines()[0]})")
        except Exception as exc:                          # noqa: BLE001
            log(f"ntopcl found but would not run: {exc}")
            ok = False
    else:
        log(f"ntopcl not found at {ntopcl}. Set the NTOPCL environment variable.")
        log("  Geometry generation needs it. The physics modules do not.")
        ok = False

    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only, change nothing")
    args = ap.parse_args()

    if args.check:
        return check()

    fetch_suave()
    link_universe()
    print()
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
