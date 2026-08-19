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
import traceback
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(REPO, "vendor")

SUAVE_REPO = "https://github.com/suavecode/SUAVE.git"

# Pinned, not tracked. `master` is a moving target, and cloning it means two machines can get
# different code from the same documented setup step. This is the head of master as of
# 2026-08-19, which is still the 2.5.2 tree from March 2022.
SUAVE_COMMIT = "0f5a2bc21bd97913aee43beca16b1ea53fb75f10"
SUAVE_BRANCH = "master"

UNIVERSE_FILES = ("functions.json", "types.json", "type_defaults.json")

# SUAVE 2.5.2 does NOT import on Python 3.10 or newer as published.
#
# Its bundled copy of `pint` imports the abstract base classes from `collections`, which moved to
# `collections.abc` in Python 3.3 and were finally removed from `collections` in 3.10. Two files
# are affected, and both fail at import time, so nothing in SUAVE is usable until they are fixed.
#
# These patches are applied here, after the clone, rather than by hand. That matters: they were
# originally applied by hand to a gitignored `vendor/SUAVE`, which meant the working environment
# existed on exactly one machine and the documented setup produced a broken one everywhere else.
# CI is what exposed it.
#
# Each patch is (file, needle, replacement). `apply_patches` fails loudly if a needle is missing,
# so an upstream change cannot leave a patch silently unapplied.
SUAVE_PATCHES: tuple[tuple[str, str, str], ...] = (
    (
        os.path.join("Plugins", "pint", "compat.py"),
        "from collections import MutableMapping\n",
        "try:\n"
        "    from collections import MutableMapping\n"
        "except ImportError:  # Python 3.10 removed the ABCs from collections\n"
        "    from collections.abc import MutableMapping\n",
    ),
    (
        os.path.join("Plugins", "pint", "quantity.py"),
        "from collections import Iterable\n",
        "try:\n"
        "    from collections import Iterable\n"
        "except ImportError:  # Python 3.10 removed the ABCs from collections\n"
        "    from collections.abc import Iterable\n",
    ),
)

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


def patches_applied() -> tuple[bool, list[str]]:
    """Whether every SUAVE_PATCHES replacement is present. Returns (all_applied, missing)."""
    root = os.path.join(VENDOR, "SUAVE")
    missing: list[str] = []
    for rel, _needle, replacement in SUAVE_PATCHES:
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            missing.append(f"{rel} (file absent)")
            continue
        with open(path, encoding="utf-8") as f:
            if replacement not in f.read():
                missing.append(rel)
    return (not missing), missing


def apply_patches() -> None:
    """Make the vendored SUAVE importable on Python 3.10 and newer.

    Idempotent: a patch already present is left alone. A needle that is absent AND unpatched is a
    hard error, because that means upstream changed and the fix no longer fits, which must not pass
    silently.
    """
    root = os.path.join(VENDOR, "SUAVE")
    for rel, needle, replacement in SUAVE_PATCHES:
        path = os.path.join(root, rel)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if replacement in text:
            log(f"patch already applied: {rel}")
            continue
        if needle not in text:
            raise RuntimeError(
                f"cannot patch {rel}: expected to find {needle!r}. Upstream SUAVE changed; "
                f"update SUAVE_PATCHES in scripts/bootstrap.py."
            )
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace(needle, replacement, 1))
        log(f"patched {rel} for Python 3.10+ collections.abc")


def fetch_suave() -> None:
    """Shallow-clone SUAVE at the pinned commit and move its package directory into vendor/."""
    if suave_present():
        log("SUAVE already present, skipping the clone")
        apply_patches()
        return
    os.makedirs(VENDOR, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        log(f"cloning {SUAVE_REPO} at {SUAVE_COMMIT[:12]}")
        # --depth 1 cannot fetch an arbitrary commit, so clone the branch shallow then deepen
        # only if the pinned commit is not the tip. Pinning is what makes two machines agree.
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", SUAVE_BRANCH, SUAVE_REPO, tmp],
            check=True,
            capture_output=True,
        )
        head = subprocess.run(
            ["git", "-C", tmp, "rev-parse", "HEAD"], check=True, capture_output=True
        ).stdout.decode().strip()
        if head != SUAVE_COMMIT:
            log(f"tip is {head[:12]}, fetching the pinned {SUAVE_COMMIT[:12]}")
            subprocess.run(
                ["git", "-C", tmp, "fetch", "--depth", "1", "origin", SUAVE_COMMIT],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", tmp, "checkout", "--detach", SUAVE_COMMIT],
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

    # Without these SUAVE does not import at all on Python 3.10 or newer.
    apply_patches()


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


def check(require_ntop: bool = False) -> int:
    """Report what resolved. Exit code reflects only what is REQUIRED.

    SUAVE is always required: nothing in the repo runs without it. nTop is optional, because the
    aerodynamics, propulsion, trajectory, mass and trade-study modules have no nTop dependency at
    all, and a hosted CI runner can never have it: `ntopcl` is licensed and the block universe is
    not redistributable. Failing the exit code on a missing nTop made a normal hosted CI state look
    like a broken setup, so `require_ntop` has to be asked for explicitly. The self-hosted runner
    that does run the geometry tier passes it.
    """
    suave_ok = True
    ntop_ok = True

    if suave_present():
        applied, missing = patches_applied()
        if applied:
            log(f"SUAVE Python 3.10+ patches: OK ({len(SUAVE_PATCHES)} applied)")
        else:
            log(f"SUAVE is NOT patched for Python 3.10+: {missing}")
            log("  Run scripts/bootstrap.py (without --check) to apply them.")
        sys.path.insert(0, VENDOR)
        try:
            import SUAVE  # noqa: F401

            log(f"SUAVE imports: OK ({SUAVE.__version__})")
        except Exception as exc:                          # noqa: BLE001
            log(f"SUAVE present but does NOT import: {type(exc).__name__}: {exc}")
            log("  numpy must be < 2, scipy < 1.14, setuptools < 81. See pyproject.toml.")
            # A one-line message is not enough to diagnose an import failure that only happens on
            # one platform. Print the whole traceback: a setup script that cannot say WHY is of
            # little use to whoever has to fix it.
            log("  full traceback follows:")
            traceback.print_exc()
            suave_ok = False
    else:
        log("SUAVE missing. Run without --check to fetch it.")
        suave_ok = False

    if universe_present():
        with open(os.path.join(VENDOR, "functions.json"), encoding="utf-8") as f:
            n = len(json.load(f))
        log(f"block universe: OK ({n} blocks)")
    else:
        log("block universe missing. Geometry generation will not work.")
        ntop_ok = False

    ntopcl = os.environ.get("NTOPCL", r"C:/Program Files/nTopology/nTopology/ntopcl.exe")
    if os.path.isfile(ntopcl):
        try:
            out = subprocess.run([ntopcl, "--version"], capture_output=True, timeout=60)
            log(f"ntopcl: OK ({out.stdout.decode(errors='replace').strip().splitlines()[0]})")
        except Exception as exc:                          # noqa: BLE001
            log(f"ntopcl found but would not run: {exc}")
            ntop_ok = False
    else:
        log(f"ntopcl not found at {ntopcl}. Set the NTOPCL environment variable.")
        log("  Geometry generation needs it. The physics modules do not.")
        ntop_ok = False

    if not ntop_ok:
        log(
            "nTop is unavailable, so the slow test tier will skip. That is expected on a hosted "
            "runner and is not a setup failure."
            + ("  It IS a failure here, because --require-ntop was given." if require_ntop else "")
        )

    failed = (not suave_ok) or (require_ntop and not ntop_ok)
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only, change nothing")
    ap.add_argument(
        "--require-ntop",
        action="store_true",
        help="treat a missing ntopcl or block universe as a failure. Off by default, because a "
        "hosted CI runner can never have them and the physics modules do not need them.",
    )
    args = ap.parse_args()

    if args.check:
        return check(require_ntop=args.require_ntop)

    fetch_suave()
    link_universe()
    print()
    return check(require_ntop=args.require_ntop)


if __name__ == "__main__":
    raise SystemExit(main())
