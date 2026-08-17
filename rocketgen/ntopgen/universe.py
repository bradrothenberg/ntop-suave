"""The nTop block universe: signature lookup, revision sorting, type resolution.

Loads `vendor/functions.json` (the function universe) and `vendor/types.json` (the type
universe) that were vendored in WP0. See `docs/REFERENCE.md` section 4 for the shape of a
function entry and for the `dep` indirections resolved here.

The universe is a *description* of the blocks a matching `ntopcl` build knows about. It is
used to check arity, resolve input types and required units, and to pick the newest revision
of a block. It is not exhaustive: some blocks that `ntopcl` accepts are absent from the
vendored file (see `docs/NTOP_NOTES.md`), so `recipe.Recipe.raw_block` exists as an escape
hatch.

Everything here is read-only and cached. `Universe.load()` returns a process-wide singleton.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Sequence

from ..config import FUNCTIONS_JSON, TYPE_DEFAULTS_JSON, TYPES_JSON

__all__ = [
    "InputDesc",
    "FunctionDesc",
    "TypeDesc",
    "Universe",
    "UnknownFunctionError",
    "split_signature",
    "parse_revision",
]


# A block id looks like `implicit_to_mesh<implicit,real,real,bool,bool>[2.4.0]`.
# The trailing bracketed group is the revision. Blocks without one are revision 1.0.0
# (verified: `body_surface_area<implicit,real>` and `body_surface_area<implicit,real>[1.1.0]`
# are the 1.0.0 and 1.1.0 revisions of the same block, and the deprecation message on the
# unversioned entry literally says "Version 1.0.0 ... is deprecated").
_REVISION_RE = re.compile(r"^(?P<base>.*?)\[(?P<maj>\d+)\.(?P<min>\d+)\.(?P<patch>\d+)\]$")

IMPLICIT_REVISION: tuple[int, int, int] = (1, 0, 0)


class UnknownFunctionError(KeyError):
    """Raised when a block id is not in the vendored function universe."""


def parse_revision(func_id: str) -> tuple[int, int, int]:
    """Return the `[maj.min.patch]` revision of a block id, numerically.

    Blocks with no bracketed suffix are revision 1.0.0.

    >>> parse_revision("implicit_to_mesh<implicit,real,real,bool,bool>[2.10.0]")
    (2, 10, 0)
    >>> parse_revision("export_mesh<file_path,mesh,unit_length_enum>")
    (1, 0, 0)
    """
    m = _REVISION_RE.match(func_id)
    if m is None:
        return IMPLICIT_REVISION
    return (int(m.group("maj")), int(m.group("min")), int(m.group("patch")))


def split_signature(func_id: str) -> tuple[str, tuple[int, int, int]]:
    """Split a block id into (base signature, revision).

    >>> split_signature("loft<implicit_2d,implicit_2d>[1.1.0]")
    ('loft<implicit_2d,implicit_2d>', (1, 1, 0))
    """
    m = _REVISION_RE.match(func_id)
    if m is None:
        return func_id, IMPLICIT_REVISION
    return m.group("base"), (int(m.group("maj")), int(m.group("min")), int(m.group("patch")))


def base_name(func_id: str) -> str:
    """The bare function name, without the angle-bracket overload or the revision.

    >>> base_name("sphere<point,real>")
    'sphere'
    """
    base, _ = split_signature(func_id)
    return base.split("<", 1)[0]


# --------------------------------------------------------------------------------------
#   Descriptions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class InputDesc:
    """One input slot of a block, with `dep` indirections already resolved."""

    index: int
    name: str
    type: str
    units_req: dict[str, int]
    description: str
    cardinality: str            # "one", "optional", "many", ...
    extensions: Any = None      # file_path slots carry allowed extensions

    @property
    def is_optional(self) -> bool:
        return self.cardinality == "optional"

    @property
    def is_variadic(self) -> bool:
        return self.cardinality == "many"


@dataclass(frozen=True)
class FunctionDesc:
    """A single entry of the function universe."""

    func_id: str
    base_signature: str
    revision: tuple[int, int, int]
    displayname: str
    description: str
    inputs: tuple[InputDesc, ...]
    return_type: str
    # Usually a dimension map. A handful of blocks (e.g. `core.dictionary<...>`) carry a LIST
    # of maps here, one per output component; kept verbatim in that case.
    output_base_units: dict[str, int] | list[Any]
    deprecated: str | None
    release_stage: int
    side_effects: bool
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def name(self) -> str:
        return base_name(self.func_id)

    @property
    def input_types(self) -> tuple[str, ...]:
        return tuple(i.type for i in self.inputs)

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(i.name for i in self.inputs)

    def input_index(self, input_name: str) -> int:
        """Index of the input slot with this name. Case-insensitive, exact otherwise."""
        wanted = input_name.strip().lower()
        for i in self.inputs:
            if i.name.strip().lower() == wanted:
                return i.index
        raise KeyError(
            f"{self.func_id!r} has no input named {input_name!r}. "
            f"Inputs are: {list(self.input_names)}"
        )

    def input(self, key: int | str) -> InputDesc:
        if isinstance(key, int):
            return self.inputs[key]
        return self.inputs[self.input_index(key)]

    def summary(self) -> str:
        args = ", ".join(f"{i.name}:{i.type}" for i in self.inputs)
        return f"{self.func_id}({args}) -> {self.return_type}"


@dataclass(frozen=True)
class TypeDesc:
    """A single entry of the type universe. `properties` are the `props` a ref may select."""

    name: str
    display_name: str
    description: str
    properties: dict[str, str]              # property name -> property type
    property_units: dict[str, dict[str, int]]   # property name -> dimension map
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


# --------------------------------------------------------------------------------------
#   The universe
# --------------------------------------------------------------------------------------


class Universe:
    """Indexed, read-only view of `vendor/functions.json` and `vendor/types.json`."""

    def __init__(
        self,
        functions: Sequence[dict[str, Any]],
        types: Sequence[dict[str, Any]],
        type_defaults: dict[str, Any] | None = None,
    ) -> None:
        self._by_id: dict[str, FunctionDesc] = {}
        self._by_base: dict[str, list[FunctionDesc]] = {}
        self._by_name: dict[str, list[FunctionDesc]] = {}
        for raw in functions:
            desc = _build_function_desc(raw)
            # Later duplicates would silently shadow. The vendored file has none; assert it.
            if desc.func_id in self._by_id:
                raise ValueError(f"duplicate function id in universe: {desc.func_id!r}")
            self._by_id[desc.func_id] = desc
            self._by_base.setdefault(desc.base_signature, []).append(desc)
            self._by_name.setdefault(desc.name, []).append(desc)

        self._types: dict[str, TypeDesc] = {}
        for raw in types:
            td = _build_type_desc(raw)
            self._types[td.name] = td

        self.type_defaults: dict[str, Any] = dict(type_defaults or {})

    # ---- construction -----------------------------------------------------------------

    @classmethod
    def load(
        cls,
        functions_json: str = FUNCTIONS_JSON,
        types_json: str = TYPES_JSON,
        type_defaults_json: str = TYPE_DEFAULTS_JSON,
    ) -> "Universe":
        """Cached singleton for the vendored universe (or any other triple of paths)."""
        return _load_cached(functions_json, types_json, type_defaults_json)

    # ---- lookup -----------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, func_id: object) -> bool:
        return func_id in self._by_id

    def get(self, func_id: str) -> FunctionDesc:
        """Exact block-id lookup. Raises `UnknownFunctionError` listing near misses."""
        try:
            return self._by_id[func_id]
        except KeyError:
            raise UnknownFunctionError(self._not_found_message(func_id)) from None

    def _not_found_message(self, func_id: str) -> str:
        base, _ = split_signature(func_id)
        lines = [f"block id {func_id!r} is not in the vendored function universe."]

        same_base = sorted(d.func_id for d in self._by_base.get(base, ()))
        if same_base:
            lines.append("  same signature, other revisions: " + ", ".join(same_base))

        same_name = sorted(d.func_id for d in self._by_name.get(base_name(func_id), ()))
        if same_name and same_name != same_base:
            shown = same_name[:12]
            more = "" if len(same_name) <= 12 else f" (+{len(same_name) - 12} more)"
            lines.append("  same function name, other overloads: " + ", ".join(shown) + more)

        if not same_base and not same_name:
            close = difflib.get_close_matches(func_id, self._by_id.keys(), n=8, cutoff=0.55)
            if not close:
                close = difflib.get_close_matches(
                    base_name(func_id), sorted(self._by_name.keys()), n=8, cutoff=0.55
                )
                close = [c for name in close for c in
                         sorted(d.func_id for d in self._by_name[name])][:8]
            if close:
                lines.append("  did you mean: " + ", ".join(close))
        lines.append(
            "  the vendored universe is not exhaustive; use Recipe.raw_block() for a block "
            "ntopcl knows but this file does not."
        )
        return "\n".join(lines)

    def find(
        self,
        name_fragment: str | None = None,
        displayname: str | None = None,
        returns: str | None = None,
        include_deprecated: bool = True,
    ) -> list[FunctionDesc]:
        """Discovery helper. All given filters must match. Results are id-sorted."""
        out: list[FunctionDesc] = []
        frag = name_fragment.lower() if name_fragment else None
        disp = displayname.lower() if displayname else None
        for desc in self._by_id.values():
            if not include_deprecated and desc.deprecated:
                continue
            if frag is not None and frag not in desc.func_id.lower():
                continue
            if disp is not None and disp not in desc.displayname.lower():
                continue
            if returns is not None and desc.return_type != returns:
                continue
            out.append(desc)
        out.sort(key=lambda d: d.func_id)
        return out

    def revisions(self, base_signature: str) -> list[FunctionDesc]:
        """Every revision of one base signature, oldest first (numeric sort)."""
        base, _ = split_signature(base_signature)
        descs = list(self._by_base.get(base, ()))
        descs.sort(key=lambda d: d.revision)
        return descs

    def latest(self, base_signature: str, include_deprecated: bool = False) -> str:
        """The highest-revision block id for a base signature.

        `base_signature` may be given with or without a `[maj.min.patch]` suffix; the suffix
        is ignored. Revisions sort numerically, so `[2.10.0]` beats `[2.4.0]`, which plain
        string sorting would get wrong.

        Deprecated revisions are skipped unless every revision is deprecated (in which case
        the newest deprecated one is returned, because there is nothing better).
        """
        descs = self.revisions(base_signature)
        if not descs:
            raise UnknownFunctionError(self._not_found_message(base_signature))
        if not include_deprecated:
            live = [d for d in descs if not d.deprecated]
            if live:
                descs = live
        return descs[-1].func_id

    # ---- type / units resolution ------------------------------------------------------

    def resolve_input_types(self, func_id: str) -> list[str]:
        """Input types of a block, with `{"dep": N}` indirections resolved."""
        return [i.type for i in self.get(func_id).inputs]

    def resolve_return_type(self, func_id: str) -> str:
        """Return type of a block, with `output.dep` resolved to the input it follows."""
        return self.get(func_id).return_type

    def resolve_units_req(self, func_id: str) -> list[dict[str, int]]:
        """Required units per input slot, with integer `unitsReq` indirections resolved."""
        return [dict(i.units_req) for i in self.get(func_id).inputs]

    def input_index(self, func_id: str, input_name: str) -> int:
        """Index of a named input slot, so callers can build blocks by input NAME."""
        return self.get(func_id).input_index(input_name)

    # ---- types ------------------------------------------------------------------------

    def type(self, type_name: str) -> TypeDesc:
        try:
            return self._types[type_name]
        except KeyError:
            close = difflib.get_close_matches(type_name, self._types.keys(), n=6, cutoff=0.5)
            hint = (" did you mean: " + ", ".join(close)) if close else ""
            raise KeyError(f"type {type_name!r} is not in the type universe.{hint}") from None

    def has_type(self, type_name: str) -> bool:
        return type_name in self._types

    def type_names(self) -> list[str]:
        return sorted(self._types)

    def properties(self, type_name: str) -> dict[str, str]:
        """Property name -> property type, i.e. the legal `props` entries for a ref."""
        return dict(self.type(type_name).properties)

    def property_type(self, type_name: str, prop: str) -> str:
        props = self.type(type_name).properties
        try:
            return props[prop]
        except KeyError:
            raise KeyError(
                f"type {type_name!r} has no property {prop!r}. Properties: {sorted(props)}"
            ) from None


# --------------------------------------------------------------------------------------
#   Builders (module-private)
# --------------------------------------------------------------------------------------


def _resolve_input_type(raw_inputs: Sequence[dict[str, Any]], index: int, seen: frozenset[int]
                        ) -> str:
    """Follow `{"dep": N}` on an input type. REFERENCE.md section 4."""
    t = raw_inputs[index].get("type", {}) or {}
    if "type" in t:
        return str(t["type"])
    if "dep" in t:
        dep = int(t["dep"])
        if dep in seen or dep < 0 or dep >= len(raw_inputs):
            raise ValueError(f"bad type dep chain at input {index}: dep={dep}")
        return _resolve_input_type(raw_inputs, dep, seen | {index})
    raise ValueError(f"input {index} has neither 'type' nor 'dep': {t!r}")


def _resolve_units_req(raw_inputs: Sequence[dict[str, Any]], index: int, seen: frozenset[int]
                       ) -> dict[str, int]:
    """Follow an integer `unitsReq` on an input. REFERENCE.md section 4.

    An int means "same required units as input N". A dict is the required dimension map.
    Absent or empty means dimensionless / unconstrained.
    """
    req = raw_inputs[index].get("unitsReq")
    if isinstance(req, bool):        # guard: bool is an int subclass
        return {}
    if isinstance(req, int):
        dep = int(req)
        if dep in seen or dep < 0 or dep >= len(raw_inputs):
            raise ValueError(f"bad unitsReq dep chain at input {index}: dep={dep}")
        return _resolve_units_req(raw_inputs, dep, seen | {index})
    if isinstance(req, dict):
        # Most entries are a bare dimension map, e.g. {"length": 1}. A few (verified on
        # `modal_simulation<...>` in vendor/functions.json) are a full unit spec with
        # {"dimension": ..., "display": ..., "scale": ...}; take the dimension out of those.
        if "dimension" in req and isinstance(req["dimension"], dict):
            req = req["dimension"]
        return {str(k): int(v) for k, v in req.items() if isinstance(v, (int, float))}
    return {}


def _units_map(value: Any) -> dict[str, int] | list[Any]:
    """Normalise a dimension map. A few blocks carry a list of maps; pass those through."""
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items()}
    if isinstance(value, list):
        return value
    return {}


def _build_function_desc(raw: dict[str, Any]) -> FunctionDesc:
    func_id = str(raw["function"])
    base, rev = split_signature(func_id)
    raw_inputs: list[dict[str, Any]] = list(raw.get("inputs") or [])

    inputs: list[InputDesc] = []
    for i, ri in enumerate(raw_inputs):
        inputs.append(
            InputDesc(
                index=i,
                name=str(ri.get("name", f"input {i}")),
                type=_resolve_input_type(raw_inputs, i, frozenset()),
                units_req=_resolve_units_req(raw_inputs, i, frozenset()),
                description=str(ri.get("description", "")),
                cardinality=str(ri.get("cardinality", "one")),
                extensions=ri.get("extensions"),
            )
        )

    out = raw.get("output") or {}
    if "dep" in out:
        # The return type follows an input. REFERENCE.md section 4.
        return_type = inputs[int(out["dep"])].type
    else:
        return_type = str(out.get("type", ""))

    dep_msg = None
    if raw.get("deprecated"):
        d = raw["deprecated"]
        dep_msg = str(d.get("message", "deprecated")) if isinstance(d, dict) else str(d)

    return FunctionDesc(
        func_id=func_id,
        base_signature=base,
        revision=rev,
        displayname=str(raw.get("displayname", "")),
        description=str(raw.get("description", "")),
        inputs=tuple(inputs),
        return_type=return_type,
        output_base_units=_units_map(raw.get("outputBaseUnits")),
        deprecated=dep_msg,
        release_stage=int(raw.get("releasestage", 0)),
        side_effects=bool(raw.get("runaspects.sideeffects", False)),
        raw=raw,
    )


def _build_type_desc(raw: dict[str, Any]) -> TypeDesc:
    props: dict[str, str] = {}
    units: dict[str, dict[str, int]] = {}
    for p in raw.get("properties") or []:
        props[str(p["name"])] = str(p.get("type", ""))
        pu = _units_map(p.get("baseUnits"))
        units[str(p["name"])] = pu if isinstance(pu, dict) else {}
    return TypeDesc(
        name=str(raw["name"]),
        display_name=str(raw.get("displayName", "")),
        description=str(raw.get("description", "")),
        properties=props,
        property_units=units,
        raw=raw,
    )


@lru_cache(maxsize=4)
def _load_cached(functions_json: str, types_json: str, type_defaults_json: str) -> Universe:
    with open(functions_json, "r", encoding="utf-8") as f:
        functions = json.load(f)
    with open(types_json, "r", encoding="utf-8") as f:
        types = json.load(f)
    type_defaults: dict[str, Any] = {}
    if type_defaults_json:
        try:
            with open(type_defaults_json, "r", encoding="utf-8") as f:
                type_defaults = json.load(f)
        except FileNotFoundError:
            type_defaults = {}
    return Universe(functions, types, type_defaults)


def universe_from_entries(
    functions: Iterable[dict[str, Any]],
    types: Iterable[dict[str, Any]] = (),
) -> Universe:
    """Build a Universe from in-memory entries. Used by tests with synthetic block ids."""
    return Universe(list(functions), list(types), {})
