"""The `ntopcl` process driver: convert, template, run, and parse outputs.

Verified `ntopcl` behaviour (nTop 5.54.0 dev build, this machine, 2026-08-17). Everything
recorded in `docs/NTOP_NOTES.md`; the short version:

* `ntopcl convert <recipe.json> <out.ntop> --dev-blocks-on=True` writes a real notebook.
* `ntopcl -t <notebook.ntop>` writes `input_template.json` and `output_template.json` into the
  PROCESS WORKING DIRECTORY, not next to the notebook. This driver therefore runs it with cwd
  set to the directory it wants the templates in.
* `ntopcl -j in.json -o out.json <notebook.ntop> -v 2` runs the notebook.
* Exit code 72 means success in some configurations (REFERENCE.md section 6). Both 0 and 72
  are treated as non-failure, and real success is gated on the expected artefacts existing and
  being non-empty.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..config import NTOPCL_FALLBACK, NTOPCL_PATH, NtopMeasurements
from .recipe import to_ntop_path

__all__ = [
    "NtopRunner",
    "NtopError",
    "RunResult",
    "ParsedOutputs",
    "OUTPUT_NAME_MAP",
    "SUCCESS_RETURNCODES",
    "INPUT_TEMPLATE_NAME",
    "OUTPUT_TEMPLATE_NAME",
    "register_output_names",
    "measurements_from_names",
]

log = logging.getLogger(__name__)

# REFERENCE.md section 6: ntopcl returns 72 on success in some configurations. Verified on this
# machine: `convert` returns 0, and `-j/-o` runs return 72 with correct artefacts written.
SUCCESS_RETURNCODES: frozenset[int] = frozenset({0, 72})

# `ntopcl -t` writes these two fixed filenames into the process working directory.
INPUT_TEMPLATE_NAME = "input_template.json"
OUTPUT_TEMPLATE_NAME = "output_template.json"

DEFAULT_TIMEOUT_S = 900.0


class NtopError(RuntimeError):
    """An `ntopcl` invocation failed, or produced no usable artefacts."""


# --------------------------------------------------------------------------------------
#   Results
# --------------------------------------------------------------------------------------


@dataclass
class RunResult:
    """Outcome of one `ntopcl` invocation."""

    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    wall_time_s: float
    cwd: str
    log_path: str | None = None
    artefacts: dict[str, bool] = field(default_factory=dict)   # path -> exists and non-empty
    output_json: str | None = None

    @property
    def returncode_ok(self) -> bool:
        """True when the return code is one of the codes nTop uses for success."""
        return self.returncode in SUCCESS_RETURNCODES

    @property
    def artefacts_ok(self) -> bool:
        """True when every expected artefact exists and is non-empty."""
        return all(self.artefacts.values())

    @property
    def ok(self) -> bool:
        """Success.

        Artefacts are the primary evidence, because neither a good return code nor a bad one is
        reliable: 72 was observed on a run that produced nothing (see `docs/NTOP_NOTES.md`
        section 7). When artefacts were expected, their presence decides. When none were
        expected, there is nothing to check but the return code.
        """
        if self.artefacts:
            return self.artefacts_ok
        return self.returncode_ok

    def missing_artefacts(self) -> list[str]:
        return [p for p, good in self.artefacts.items() if not good]

    def tail(self, n: int = 40) -> str:
        lines = (self.stdout + "\n" + self.stderr).splitlines()
        return "\n".join(lines[-n:])


@dataclass
class ParsedOutputs:
    """`NtopMeasurements` plus everything the mapping table did not claim.

    `raw` holds every output entry by name, decoded to plain Python. WP4 extends
    `OUTPUT_NAME_MAP` rather than editing `rocketgen/config.py`.
    """

    measurements: NtopMeasurements
    raw: dict[str, Any] = field(default_factory=dict)
    unmapped: list[str] = field(default_factory=list)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "measurements": asdict(self.measurements),
                    "raw": self.raw,
                    "unmapped": self.unmapped,
                },
                f,
                indent=2,
                default=str,
            )


# --------------------------------------------------------------------------------------
#   Output name mapping
# --------------------------------------------------------------------------------------

# Notebook output NAME -> field of `config.NtopMeasurements`.
#
# WP4 owns the notebook and therefore owns these names. Extend this dict (or pass
# `extra_map=` to `parse_outputs`) instead of changing `config.py`. Names are matched
# case-insensitively after stripping, and also with spaces/dots/dashes folded to underscores,
# so "Volume Total", "volume_total" and "VOLUME TOTAL" all hit the same entry.
OUTPUT_NAME_MAP: dict[str, str] = {
    # solid volumes, m^3
    "volume_total": "volume_total",
    "volume": "volume_total",
    "volume_structure": "volume_structure",
    "volume_cavity": "volume_cavity",
    "volume_grain": "volume_grain",
    # areas, m^2
    "area_wetted_body": "area_wetted_body",
    "wetted_area": "area_wetted_body",
    "surface_area": "area_wetted_body",
    "area_wetted_fins": "area_wetted_fins",
    "area_base": "area_base",
    # mass properties
    "mass_structure": "mass_structure",
    "mass": "mass_structure",
    "cg_structure": "cg_structure",
    "center_of_gravity": "cg_structure",
    "cg": "cg_structure",
    "inertia_structure": "inertia_structure",
    "principal_moments": "inertia_structure",
    # artefact paths
    "stl_path": "stl_path",
    "step_path": "step_path",
    "implicit_path": "implicit_path",
    # cross-section area distribution
    "area_distribution": "area_distribution",
    # --- WP4 additions, owned by ntopgen/rocket_notebook.py -------------------------
    # The SV-1 notebook emits these names through its single `json` output slot. They are
    # listed here so the table is self-documenting; `rocket_notebook` also registers them
    # at import through `register_output_names`, which is the mechanism a later work package
    # should use rather than editing this dict.
    #
    # Vector-valued measurements (`cg_structure`, `inertia_structure`) travel as three named
    # scalars, because `core.list<real>` only carries scalars. They are deliberately NOT
    # mapped here: `rocket_notebook._collect_vectors` reassembles them into 3-tuples.
    "volume_oml": "volume_total",
    "volume_airframe": "volume_structure",
    "volume_internal": "volume_cavity",
    "mass_airframe": "mass_structure",
    "area_fins": "area_wetted_fins",
}


def register_output_names(mapping: Mapping[str, str]) -> None:
    """Additively extend `OUTPUT_NAME_MAP`. The way a notebook author declares its names.

    A work package that owns a notebook calls this at import with `{output name: field of
    NtopMeasurements}`. Re-registering the same pair is a no-op, so importing twice is safe.
    A conflicting redefinition raises, because two notebooks quietly disagreeing about what
    "volume" means is a defect. `rocketgen/config.py` is never edited for this.
    """
    fields = set(NtopMeasurements.__dataclass_fields__)
    for name, target in mapping.items():
        if target not in fields:
            raise ValueError(
                f"output name {name!r} maps to {target!r}, which is not a field of "
                f"NtopMeasurements. Fields are: {sorted(fields)}"
            )
        existing = OUTPUT_NAME_MAP.get(name)
        if existing is not None and existing != target:
            raise ValueError(
                f"output name {name!r} is already mapped to {existing!r}; refusing to "
                f"remap it to {target!r}"
            )
        OUTPUT_NAME_MAP[name] = target


def measurements_from_names(
    values: Mapping[str, Any],
    target: NtopMeasurements | None = None,
    extra_map: Mapping[str, str] | None = None,
) -> NtopMeasurements:
    """Map an ALREADY-DECODED `{output name: value}` dictionary onto an `NtopMeasurements`.

    `parse_outputs` does this for a whole output-JSON file and returns one flat object. That is
    the right shape for a notebook that measures one body. A notebook that measures SEVERAL
    bodies has to namespace its output names per body (for example `s1_volume_total` and
    `s2_volume_total`), which means a single flat object cannot hold the result: both names map
    onto the same `volume_total` field and the second would overwrite the first.

    So this helper exists to fill ONE body's record from a sub-dictionary the caller has already
    separated out. It uses the same `OUTPUT_NAME_MAP` table and the same field casting as
    `parse_outputs`, so the two cannot drift apart.

    `target` may be an instance of a SUBCLASS of `NtopMeasurements` carrying extra fields; it is
    filled in place and returned. Names that do not map to a field of `NtopMeasurements` are
    ignored here, because the subclass owns them and knows their types.

    Used by `ntopgen/stack_notebook.py`. `rocketgen/config.py` is never edited for this.
    """
    m = target if target is not None else NtopMeasurements()
    table = {_fold(k): v for k, v in OUTPUT_NAME_MAP.items()}
    table.update({_fold(k): v for k, v in (extra_map or {}).items()})
    fields = set(NtopMeasurements.__dataclass_fields__)
    for name, value in values.items():
        field_name = table.get(_fold(str(name)))
        if field_name is None or field_name not in fields:
            continue
        setattr(m, field_name, _cast_for_field(field_name, value))
    return m


def _fold(name: str) -> str:
    out = []
    for ch in name.strip().lower():
        out.append(ch if ch.isalnum() else "_")
    folded = "".join(out)
    while "__" in folded:
        folded = folded.replace("__", "_")
    return folded.strip("_")


# --------------------------------------------------------------------------------------
#   Value decoding
# --------------------------------------------------------------------------------------


def decode_value(entry: Mapping[str, Any]) -> Any:
    """Decode one output-JSON entry's `value` into a plain Python object.

    Output JSON is a top-level LIST of `{"components": [...], "name", "type", "value"}` and
    the `value` uses the same literal encodings as a recipe (REFERENCE.md section 5).
    Verified against real Automate output on this machine.
    """
    t = str(entry.get("type", ""))
    v = entry.get("value")

    if v is None:
        return None
    if isinstance(v, list):
        # `point` and `vector` come back as a list of {isFinite, val}.
        return tuple(_num(c) for c in v)
    if not isinstance(v, dict):
        return v

    if "val" in v:
        return v["val"]
    if "string" in v:
        return v["string"]
    if "jsonObject" in v:
        return v["jsonObject"]
    if "value" in v and isinstance(v["value"], list):
        return tuple(_num(c) for c in v["value"])
    if "enum" in v:
        return int(v["enum"])
    if "id" in v:
        return str(v["id"])
    if "expression" in v:
        return str(v["expression"])
    if "choices" in v:
        sel = int(v.get("selected", 0))
        choices = list(v.get("choices", []))
        return choices[sel] if 0 <= sel < len(choices) else None
    log.debug("unrecognised output value encoding for type %r: %r", t, v)
    return v


def _num(component: Any) -> float | None:
    if isinstance(component, Mapping):
        val = component.get("val")
        return None if val is None else float(val)
    try:
        return float(component)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------------------
#   The runner
# --------------------------------------------------------------------------------------


class NtopRunner:
    """Resolve and drive the `ntopcl` executable."""

    def __init__(self, ntopcl: str | os.PathLike[str] | None = None,
                 dev_blocks: bool = True) -> None:
        self.exe = self._resolve(ntopcl)
        self.dev_blocks = dev_blocks
        self.version = self._probe_version()
        log.info("ntopcl: %s (%s)", self.exe, self.version)

    # ---- resolution --------------------------------------------------------------------

    @staticmethod
    def _resolve(ntopcl: str | os.PathLike[str] | None) -> str:
        candidates: list[str] = []
        if ntopcl:
            candidates.append(str(ntopcl))
        else:
            # config.NTOPCL_PATH honours the NTOPCL environment variable already.
            candidates.extend([NTOPCL_PATH, NTOPCL_FALLBACK])
        for c in candidates:
            if c and os.path.isfile(c):
                return os.path.abspath(c)
        raise NtopError(
            "no ntopcl executable found. Tried:\n  "
            + "\n  ".join(repr(c) for c in candidates)
            + "\nSet the NTOPCL environment variable or pass ntopcl=<path>."
        )

    def _probe_version(self) -> str:
        try:
            p = subprocess.run([self.exe, "--version"], capture_output=True, text=True,
                               timeout=120)
            return (p.stdout or p.stderr).strip().splitlines()[0] if (p.stdout or p.stderr) \
                else "unknown"
        except (OSError, subprocess.SubprocessError, IndexError) as exc:  # pragma: no cover
            log.warning("could not read ntopcl version: %s", exc)
            return "unknown"

    # ---- low-level invocation ----------------------------------------------------------

    def _invoke(
        self,
        args: Sequence[str],
        cwd: str,
        timeout: float,
        artefacts: Iterable[str] = (),
        log_path: str | None = None,
    ) -> RunResult:
        argv = [self.exe, *[str(a) for a in args]]
        os.makedirs(cwd, exist_ok=True)
        log.info("run: %s (cwd=%s)", " ".join(argv), cwd)
        t0 = time.perf_counter()
        try:
            p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout)
            rc, out, err = p.returncode, p.stdout or "", p.stderr or ""
        except subprocess.TimeoutExpired as exc:
            rc = -1
            out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) \
                else (exc.stdout or "")
            err = (exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes)
                   else (exc.stderr or "")) + f"\nTIMEOUT after {timeout} s"
        dt = time.perf_counter() - t0

        res = RunResult(
            argv=argv,
            returncode=rc,
            stdout=out,
            stderr=err,
            wall_time_s=dt,
            cwd=cwd,
            artefacts={str(a): _nonempty(a) for a in artefacts},
        )
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(" ".join(argv) + "\n")
                f.write(f"cwd: {cwd}\nreturncode: {rc}\nwall_time_s: {dt:.3f}\n")
                f.write("--- stdout ---\n" + out + "\n--- stderr ---\n" + err + "\n")
            res.log_path = log_path
        log.info("  -> rc=%s in %.2f s", rc, dt)
        return res

    def _check(self, res: RunResult, what: str) -> RunResult:
        """Raise unless the invocation looks like a success.

        An unexpected return code with every artefact present is a warning, not an error,
        because nTop is known to return odd codes on success. A missing artefact is always an
        error, even when the return code is 0 or 72.
        """
        if res.ok:
            if not res.returncode_ok:
                log.warning("%s: unexpected return code %s but artefacts are present",
                            what, res.returncode)
            return res
        detail = [f"{what} failed: ntopcl returned {res.returncode}."]
        missing = res.missing_artefacts()
        if missing:
            detail.append("missing or empty artefacts: " + ", ".join(missing))
        if res.log_path:
            detail.append(f"log: {res.log_path}")
        detail.append("--- stderr ---\n" + (res.stderr.strip() or "(empty)"))
        detail.append("--- stdout tail ---\n" + res.tail(30))
        raise NtopError("\n".join(detail))

    # ---- convert -----------------------------------------------------------------------

    def convert(
        self,
        recipe_json: str | os.PathLike[str],
        out_ntop: str | os.PathLike[str],
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> RunResult:
        """`ntopcl convert <recipe.json> <out.ntop> --dev-blocks-on=True`.

        `--dev-blocks-on=True` matters: without it, blocks that the dev build exposes are
        rejected (REFERENCE.md section 3). Returns the RunResult; raises `NtopError` on
        failure.
        """
        recipe_json = os.path.abspath(recipe_json)
        out_ntop = os.path.abspath(out_ntop)
        if not os.path.isfile(recipe_json):
            raise NtopError(f"recipe JSON does not exist: {recipe_json}")
        if os.path.splitext(out_ntop)[1].lower() != ".ntop":
            raise NtopError(f"output must have a .ntop extension: {out_ntop}")
        os.makedirs(os.path.dirname(out_ntop) or ".", exist_ok=True)
        if os.path.exists(out_ntop):
            os.remove(out_ntop)

        args = ["convert", recipe_json, out_ntop]
        if self.dev_blocks:
            args.append("--dev-blocks-on=True")
        res = self._invoke(
            args,
            cwd=os.path.dirname(out_ntop),
            timeout=timeout,
            artefacts=[out_ntop],
            log_path=os.path.splitext(out_ntop)[0] + "_convert.log",
        )
        return self._check(res, "ntopcl convert")

    def exportjson(
        self,
        ntop: str | os.PathLike[str],
        out_json: str | os.PathLike[str],
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> RunResult:
        """`ntopcl exportjson <in.ntop> <out.json> --ext --dev-blocks-on=True`. The inverse
        of `convert`, for round-trip checks."""
        ntop = os.path.abspath(ntop)
        out_json = os.path.abspath(out_json)
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        if os.path.exists(out_json):
            os.remove(out_json)
        args = ["exportjson", ntop, out_json, "--ext"]
        if self.dev_blocks:
            args.append("--dev-blocks-on=True")
        res = self._invoke(
            args,
            cwd=os.path.dirname(out_json),
            timeout=timeout,
            artefacts=[out_json],
            log_path=os.path.splitext(out_json)[0] + "_exportjson.log",
        )
        return self._check(res, "ntopcl exportjson")

    # ---- templates ---------------------------------------------------------------------

    def templates(
        self,
        ntop: str | os.PathLike[str],
        out_dir: str | os.PathLike[str] | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        require_output: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """`ntopcl -t <ntop>`; returns the parsed (input template, output template).

        `-t` writes `input_template.json` and `output_template.json` into the process working
        directory (verified empirically: they appear in the cwd, NOT beside the notebook), so
        `out_dir` selects where they land. It defaults to the notebook's directory.

        The output template is `None` when the notebook designates no Automate output. In that
        case `ntopcl` logs "Error generating output template : Output of function not set" and
        still exits 0. Pass `require_output=True` to turn that into an error.
        """
        ntop = os.path.abspath(ntop)
        cwd = os.path.abspath(out_dir) if out_dir else os.path.dirname(ntop)
        os.makedirs(cwd, exist_ok=True)
        in_path = os.path.join(cwd, INPUT_TEMPLATE_NAME)
        out_path = os.path.join(cwd, OUTPUT_TEMPLATE_NAME)
        for p in (in_path, out_path):
            if os.path.exists(p):
                os.remove(p)

        res = self._invoke(
            ["-t", ntop],
            cwd=cwd,
            timeout=timeout,
            artefacts=[in_path],
            log_path=os.path.join(cwd, "template.log"),
        )
        self._check(res, "ntopcl -t")

        with open(in_path, "r", encoding="utf-8") as f:
            input_template = json.load(f)
        output_template: dict[str, Any] | None = None
        if _nonempty(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                output_template = json.load(f)
        elif require_output:
            raise NtopError(
                f"ntopcl -t wrote no {OUTPUT_TEMPLATE_NAME}: the notebook designates no "
                f"Automate output. Call Recipe.set_output(...) before converting.\n"
                + res.tail(10)
            )
        return input_template, output_template

    # ---- run ---------------------------------------------------------------------------

    def build_input_json(
        self,
        inputs: Mapping[str, Any],
        input_template: Mapping[str, Any] | None = None,
        units: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Build the `-j` input JSON.

        The schema `-t` reveals is `{"inputs": [{"name", "type", "units", "value"}], ...}`.
        `units` in the template are DISPLAY units (nTop writes "mm" and "deg", with the value
        converted into them), so this driver always writes an explicit `units` string when it
        knows one, and never relies on the default.

        A value may be given as a plain scalar, or as a `(value, unit)` tuple to state the
        unit inline. Explicit `units=` entries win over the tuple form, which wins over the
        template's unit.
        """
        by_name: dict[str, dict[str, Any]] = {}
        for decl in (input_template or {}).get("inputs", []) or []:
            by_name[str(decl.get("name"))] = dict(decl)

        entries: list[dict[str, Any]] = []
        unknown = [k for k in inputs if by_name and k not in by_name]
        if unknown:
            raise NtopError(
                f"inputs {unknown} are not notebook inputs. The notebook accepts: "
                f"{sorted(by_name)}"
            )
        for key, value in inputs.items():
            unit: str | None = None
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
                value, unit = value
            if units and key in units:
                unit = units[key]
            decl = by_name.get(key, {})
            entry: dict[str, Any] = {"name": key, "type": str(decl.get("type", "real"))}
            if unit is None:
                unit = decl.get("units")
            if unit:
                entry["units"] = unit
            if isinstance(value, (str, os.PathLike)) and entry["type"] == "file_path":
                entry["value"] = to_ntop_path(value)
            else:
                entry["value"] = value
            entries.append(entry)
        return {"inputs": entries}

    def run(
        self,
        ntop: str | os.PathLike[str],
        inputs: Mapping[str, Any] | None = None,
        out_json: str | os.PathLike[str] | None = None,
        expect: Iterable[str | os.PathLike[str]] = (),
        timeout: float = DEFAULT_TIMEOUT_S,
        verbose: int = 2,
        input_template: Mapping[str, Any] | None = None,
        units: Mapping[str, str] | None = None,
        input_json: str | os.PathLike[str] | None = None,
        run_dir: str | os.PathLike[str] | None = None,
        save: bool = False,
    ) -> RunResult:
        """`ntopcl -j <in.json> -o <out.json> <ntop> -v <verbose>`.

        `expect` lists artefacts (STL, STEP, ...) that must exist and be non-empty for the run
        to count as a success. `out_json` is always in that list when given. stdout and stderr
        go to `<run_dir>/ntopcl_run.log`.
        """
        ntop = os.path.abspath(ntop)
        rd = os.path.abspath(run_dir) if run_dir else os.path.dirname(ntop)
        os.makedirs(rd, exist_ok=True)

        in_path = os.path.abspath(input_json or os.path.join(rd, "input.json"))
        payload = self.build_input_json(inputs or {}, input_template, units)
        with open(in_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        artefacts = [str(os.path.abspath(a)) for a in expect]
        args: list[str] = ["-j", in_path]
        if out_json is not None:
            out_json = os.path.abspath(out_json)
            if os.path.exists(out_json):
                os.remove(out_json)
            args += ["-o", out_json]
            artefacts.insert(0, out_json)
        args.append(ntop)
        args += ["-v", str(int(verbose))]
        if save:
            args.append("-s")

        for a in artefacts:
            if os.path.exists(a):
                os.remove(a)

        res = self._invoke(
            args,
            cwd=rd,
            timeout=timeout,
            artefacts=artefacts,
            log_path=os.path.join(rd, "ntopcl_run.log"),
        )
        res.output_json = str(out_json) if out_json else None
        return self._check(res, "ntopcl run")

    # ---- output parsing ----------------------------------------------------------------

    @staticmethod
    def parse_outputs(
        out_json: str | os.PathLike[str],
        extra_map: Mapping[str, str] | None = None,
        run: RunResult | None = None,
    ) -> ParsedOutputs:
        """Map a notebook's output JSON onto `config.NtopMeasurements`.

        See `OUTPUT_NAME_MAP` for the name table. Unmapped outputs are kept in
        `ParsedOutputs.raw` and listed in `ParsedOutputs.unmapped`; nothing is dropped.
        Pass `run` to fill in the bookkeeping fields from the invocation that produced it.
        """
        return parse_outputs(out_json, extra_map, run)


def parse_outputs(
    out_json: str | os.PathLike[str],
    extra_map: Mapping[str, str] | None = None,
    run: RunResult | None = None,
) -> ParsedOutputs:
    """Module-level form of `NtopRunner.parse_outputs`."""
    with open(out_json, "r", encoding="utf-8") as f:
        doc = json.load(f)

    entries: list[Mapping[str, Any]]
    if isinstance(doc, list):
        entries = [e for e in doc if isinstance(e, Mapping)]
    elif isinstance(doc, Mapping) and isinstance(doc.get("outputs"), list):
        entries = [e for e in doc["outputs"] if isinstance(e, Mapping)]
    else:
        raise NtopError(
            f"unexpected output JSON shape in {out_json}: expected a list of "
            f"{{name, type, value}} entries, got {type(doc).__name__}"
        )

    table = {_fold(k): v for k, v in OUTPUT_NAME_MAP.items()}
    table.update({_fold(k): v for k, v in (extra_map or {}).items()})

    m = NtopMeasurements()
    raw: dict[str, Any] = {}
    unmapped: list[str] = []
    fields = set(m.__dataclass_fields__)

    flat: list[tuple[str, Any]] = []
    containers: set[str] = set()
    for e in entries:
        name = str(e.get("name", ""))
        value = decode_value(e)
        raw[name] = value
        flat.append((name, value))
        # A `json` output carries a whole dictionary; treat its keys as outputs too, so a
        # notebook can report many quantities through nTop's single-output slot.
        if isinstance(value, Mapping):
            containers.add(name)
            for k, v in value.items():
                flat.append((str(k), v))
                raw.setdefault(str(k), v)

    for name, value in flat:
        target = table.get(_fold(name))
        if target is None or target not in fields:
            # A container's own name is bookkeeping, not a measurement, so do not report it as
            # an unmapped quantity; its unpacked keys are reported instead.
            if target is None and name not in containers:
                unmapped.append(name)
            continue
        setattr(m, target, _cast_for_field(target, value))

    if run is not None:
        m.wall_time_s = run.wall_time_s
        m.ntopcl_returncode = run.returncode
        if not run.returncode_ok:
            m.warnings.append(f"ntopcl returned {run.returncode}")

    return ParsedOutputs(measurements=m, raw=raw, unmapped=sorted(set(unmapped)))


def _cast_for_field(field_name: str, value: Any) -> Any:
    """Coerce a decoded output value into the shape `NtopMeasurements` declares."""
    if field_name in ("cg_structure", "inertia_structure"):
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return tuple(float(v) if v is not None else 0.0 for v in value)
        return None
    if field_name == "area_distribution":
        out: list[tuple[float, float]] = []
        if isinstance(value, (list, tuple)):
            for row in value:
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    out.append((float(row[0]), float(row[1])))
        return out
    if field_name.endswith("_path"):
        return None if value is None else str(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _nonempty(path: str | os.PathLike[str]) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False
