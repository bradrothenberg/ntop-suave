"""A typed builder for nTop notebook-recipe JSON.

`docs/REFERENCE.md` section 5 documents the schema this module emits; `docs/NTOP_NOTES.md`
records everything learned since. Read those before changing an encoding here.

Design notes
------------
* Every block the caller creates becomes a root **variable** wrapping the block as its
  `contents`, and every reference to it is the `{"props": [...], "ref": {"id": ...}}` form.
  That is exactly what `ntopcl exportjson` emits for GUI-authored notebooks (345 of 360 root
  entries in `a real 360-block reference notebook`), so it is the best-tested shape, it lets one
  value feed several consumers, and it makes the generated notebook readable in the GUI as a
  flat list of named values.
* Literal *arguments* are inlined, never wrapped, because a literal has no id.
* `Ref` is the single currency: it wraps a block, a notebook input, or an inline literal, and
  knows how to render itself into an input slot.
* `block()` looks the signature up in the `Universe`, checks arity, and promotes bare Python
  scalars to the literal encoding required by that specific input slot, including the slot's
  required units. Positional-argument mistakes therefore fail in Python, not in nTop.

Units: all lengths are metres, all angles radians (REFERENCE.md section 5).
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import string
from dataclasses import dataclass, replace as dc_replace
from typing import Any, Iterable, Mapping, Sequence

from .universe import FunctionDesc, InputDesc, UnknownFunctionError, Universe, split_signature

__all__ = [
    "Recipe",
    "Ref",
    "RecipeError",
    "ArityError",
    "LiteralTypeError",
    "BLOCK_REVISION_OVERRIDES",
    "FIRST_INSTANCE_ID",
    "to_ntop_path",
]

log = logging.getLogger(__name__)

# `notebook_author` starts instance ids at 101 (REFERENCE.md section 5). Anything unique works;
# matching the prior art keeps generated JSON diffable against its test fixtures.
FIRST_INSTANCE_ID = 101

# The vendored block universe is dated Aug 8 2026 and is already behind the shipped ntopcl
# builds. Where a newer revision exists in ntopcl but not in `vendor/functions.json`, list it
# here; `Recipe.latest()` prefers it. Every entry below was proven by running
# `runs/_smoke/_probe_revisions.py`, which converts a one-block notebook for each candidate
# revision and keeps the newest one `ntopcl convert` accepts.
#
#   implicit_to_mesh ... [2.5.0]: both nTop 5.53.2 and the 5.54.0 dev build accept 2.5.0 and
#   both log "Mesh from Implicit Body 2.4.0 is deprecated due to a bug that removed features
#   larger than Min. Feature Size" when 2.4.0 is used. The vendored file still marks 2.4.0 as
#   current. Verified 2026-08-17 on this machine.
#
# The same probe found NO newer revision for mass_properties, surface_area, export_mesh,
# export_part, export_implicit_body, export_table, export_text, cad_body_from_implicit_body,
# the boolean blocks, revolve, profile_from_points, shell or thicken_implicit.
BLOCK_REVISION_OVERRIDES: dict[str, str] = {
    "implicit_to_mesh<implicit,real,real,bool,bool>":
        "implicit_to_mesh<implicit,real,real,bool,bool>[2.5.0]",
    "implicit_to_mesh<implicit,real,real,integer,implicit,bool>":
        "implicit_to_mesh<implicit,real,real,integer,implicit,bool>[2.5.0]",
}

# Types whose literal encoding this module knows. Anything else must be built as a block.
_SCALAR_LITERAL_TYPES = frozenset(
    {"real", "integer", "bool", "text", "file_path", "point", "vector", "real_field"}
)


class RecipeError(RuntimeError):
    """Base class for builder errors."""


class ArityError(RecipeError):
    """Wrong number of arguments for a block, or an unfilled required slot."""


class LiteralTypeError(RecipeError):
    """A Python value cannot be promoted to the literal type an input slot needs."""


def to_ntop_path(path: str | os.PathLike[str]) -> str:
    """nTop `file_path` literals want forward slashes, even on Windows.

    REFERENCE.md section 5: `{"type":"file_path","value":{"val":"C:/path/with/forward/slashes"}}`
    """
    return str(path).replace("\\", "/")


# --------------------------------------------------------------------------------------
#   Literals
# --------------------------------------------------------------------------------------


def literal_real(val: float, units: Mapping[str, int] | None = None) -> dict[str, Any]:
    """`real` literal. Value in SI base units for `units` (metres, radians, kg)."""
    return {
        "type": "real",
        "value": {
            "isFinite": True,
            "units": {str(k): int(v) for k, v in (units or {}).items()},
            "val": float(val),
        },
    }


def literal_integer(val: int) -> dict[str, Any]:
    return {"type": "integer", "value": {"val": int(val)}}


def literal_bool(val: bool) -> dict[str, Any]:
    return {"type": "bool", "value": {"val": bool(val)}}


def literal_text(val: str) -> dict[str, Any]:
    return {"type": "text", "value": {"string": str(val)}}


def literal_file_path(path: str | os.PathLike[str]) -> dict[str, Any]:
    return {"type": "file_path", "value": {"val": to_ntop_path(path)}}


def literal_point(x: float, y: float, z: float) -> dict[str, Any]:
    """`point` literal. Components are metres and carry no units key of their own."""
    return {
        "type": "point",
        "value": [
            {"isFinite": True, "val": float(x)},
            {"isFinite": True, "val": float(y)},
            {"isFinite": True, "val": float(z)},
        ],
    }


def literal_vector(
    x: float, y: float, z: float, units: Mapping[str, int] | None = None
) -> dict[str, Any]:
    """`vector` literal. Shape verified against `exportjson` of a real notebook."""
    return {
        "type": "vector",
        "value": {
            "units": {str(k): int(v) for k, v in (units or {}).items()},
            "value": [
                {"isFinite": True, "val": float(x)},
                {"isFinite": True, "val": float(y)},
                {"isFinite": True, "val": float(z)},
            ],
        },
    }


def literal_real_field(expr: str, units: Mapping[str, int] | None = None) -> dict[str, Any]:
    """`real_field` literal: an nTop field expression, e.g. `"(3*x + 2*y - z)/1mm"`."""
    return {
        "type": "real_field",
        "value": {
            "expression": str(expr),
            "units": {str(k): int(v) for k, v in (units or {}).items()},
        },
    }


def literal_enum(type_name: str, index_or_id: int | str) -> dict[str, Any]:
    """Enum literal. Two encodings exist (REFERENCE.md section 5).

    An int emits `{"enum": N}`, the generic form. A str emits `{"id": "..."}`, which is what
    `unit_length_enum` uses.
    """
    if isinstance(index_or_id, bool):
        raise LiteralTypeError(f"enum {type_name!r} needs an int index or a str id, got a bool")
    if isinstance(index_or_id, int):
        return {"type": type_name, "value": {"enum": int(index_or_id)}}
    return {"type": type_name, "value": {"id": str(index_or_id)}}


def literal_unit_length(unit_id: str = "mm") -> dict[str, Any]:
    """`unit_length_enum` literal, e.g. `"mm"`, `"m"`, `"in"`."""
    return literal_enum("unit_length_enum", str(unit_id))


# --------------------------------------------------------------------------------------
#   Blocks and refs
# --------------------------------------------------------------------------------------


@dataclass
class _Block:
    """One authored block, plus the root variable that will hold it."""

    func: str
    block_id: str
    var_id: str
    return_type: str
    name: str | None
    args: list[Any]                 # each entry is a Ref, an inline literal dict, or None
    desc: FunctionDesc | None = None

    def render_block(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "func": self.func,
            "id": self.block_id,
            "inputs": [_render_arg(a) for a in self.args],
            "type": self.return_type,
        }
        if self.name:
            # nTop autogenerates a name when absent; giving both the block and its variable
            # the same name keeps the GUI readable.
            out["name"] = self.name
        return out

    def render_variable(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "contents": self.render_block(),
            "id": self.var_id,
            "type": self.return_type,
            "variable": True,
        }
        if self.name:
            out["name"] = self.name
        return out


@dataclass
class _ValueVariable:
    """A root variable whose contents is a literal or another reference, not a block."""

    var_id: str
    return_type: str
    name: str | None
    contents: Any                   # Ref, inline literal dict, or None
    description: str = ""

    def render_variable(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "contents": _render_arg(self.contents),
            "id": self.var_id,
            "type": self.return_type,
            "variable": True,
        }
        if self.name:
            out["name"] = self.name
        if self.description:
            out["metadata"] = {"description": self.description}
        return out


@dataclass(frozen=True)
class Ref:
    """A value that can fill a block input slot.

    Exactly one of `block`, `variable`, `input_index` and `literal` is set. `props` selects a
    property chain on the referenced value, e.g. `("volume",)` on a `body_mass_props`, or
    `("bodies", "[0]")` to index a list-valued property.
    """

    type: str
    block: _Block | None = None
    variable: _ValueVariable | None = None
    input_index: int | None = None
    literal: dict[str, Any] | None = None
    props: tuple[str, ...] = ()
    label: str = ""

    # ---- construction -----------------------------------------------------------------

    @classmethod
    def from_literal(cls, literal: dict[str, Any]) -> "Ref":
        return cls(type=str(literal.get("type", "")), literal=literal)

    # ---- property access --------------------------------------------------------------

    def prop(self, *names: str, type: str | None = None,
             universe: Universe | None = None) -> "Ref":
        """Select a property of this value, e.g. `mp.prop("volume")`.

        Use `"[N]"` to index a list-valued property, matching what `exportjson` emits:
        `{"props": ["bodies", "[0]"], "ref": {...}}`.
        """
        if self.literal is not None:
            raise RecipeError("cannot take a property of an inline literal; wrap it in a "
                              "variable first with Recipe.variable()")
        new_type = type
        if new_type is None:
            u = universe or Universe.load()
            cur = self.type
            for n in names:
                if re.fullmatch(r"\[\d+\]", n):
                    m = re.fullmatch(r"list<(.+)>", cur)
                    cur = m.group(1) if m else "any"
                    continue
                if u.has_type(cur):
                    cur = u.property_type(cur, n)
                else:
                    cur = "any"
            new_type = cur
        return Ref(
            type=new_type,
            block=self.block,
            variable=self.variable,
            input_index=self.input_index,
            props=self.props + tuple(names),
            label=self.label,
        )

    # ---- rendering --------------------------------------------------------------------

    def render(self) -> dict[str, Any]:
        props = list(self.props)
        if self.literal is not None:
            if props:
                raise RecipeError("an inline literal cannot carry props")
            return dict(self.literal)
        if self.input_index is not None:
            return {"input": int(self.input_index), "props": props}
        target = self.block.var_id if self.block is not None else self.variable.var_id  # type: ignore[union-attr]
        return {"props": props, "ref": {"id": target}}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        what = self.label or (self.block.func if self.block else "input")
        p = ("." + ".".join(self.props)) if self.props else ""
        return f"<Ref {what}{p}: {self.type}>"


def _render_arg(arg: Any) -> Any:
    if arg is None:
        return None
    if isinstance(arg, Ref):
        return arg.render()
    if isinstance(arg, dict):
        return dict(arg)
    raise RecipeError(f"cannot render argument of type {type(arg).__name__}: {arg!r}")


# --------------------------------------------------------------------------------------
#   The recipe
# --------------------------------------------------------------------------------------


class Recipe:
    """Build a notebook recipe, then write it as JSON or convert it to `.ntop`.

    Example
    -------
    >>> r = Recipe(name="smoke")
    >>> R = r.add_input("Radius", "real", default=0.025, dimension={"length": 1})
    >>> p0 = r.block("point<real,real,real>", 0.0, 0.0, 0.0, name="Origin")
    >>> ball = r.block("sphere<point,real>", p0, R, name="Ball")
    >>> mesh = r.mesh_from_implicit(ball, tolerance=2.5e-4)
    >>> _ = r.export_stl(mesh, "C:/tmp/ball", units="mm")
    """

    def __init__(
        self,
        name: str | None = None,
        displayname: str = "",
        description: str = "",
        universe: Universe | None = None,
        first_id: int = FIRST_INSTANCE_ID,
        rng: random.Random | None = None,
    ) -> None:
        self.universe = universe if universe is not None else Universe.load()
        self.displayname = displayname or (name or "")
        self.description = description
        self._next_id = int(first_id)
        self._entries: list[_Block | _ValueVariable] = []
        self._inputs: list[dict[str, Any]] = []
        self._output_ref: Ref | None = None
        self._used_names: set[str] = set()
        self.internal_name = _make_internal_name(rng or random.Random())

    # ---- ids ---------------------------------------------------------------------------

    def _new_id(self) -> str:
        """Allocate a unique `inst<N>` id."""
        i = self._next_id
        self._next_id += 1
        return f"inst{i}"

    def _unique_name(self, name: str | None) -> str | None:
        """nTop tolerates duplicate names, but they make a notebook unreadable. Suffix them."""
        if not name:
            return None
        if name not in self._used_names:
            self._used_names.add(name)
            return name
        n = 1
        while f"{name} {n}" in self._used_names:
            n += 1
        out = f"{name} {n}"
        self._used_names.add(out)
        return out

    # ---- notebook inputs ---------------------------------------------------------------

    def add_input(
        self,
        name: str,
        type: str,
        default: Any = None,
        dimension: Mapping[str, int] | None = None,
        description: str = "",
    ) -> Ref:
        """Declare a notebook input and return a `Ref` to it.

        `dimension` is the physical dimension map, e.g. `{"length": 1}` for a length or
        `{"angle": 1}` for an angle. `default` is in SI base units for that dimension.
        """
        index = len(self._inputs)
        decl: dict[str, Any] = {"description": description, "name": name, "type": type}
        if dimension:
            decl["dimension"] = {str(k): int(v) for k, v in dimension.items()}
        if default is not None:
            decl["contents"] = self._promote(default, type, dict(dimension or {}),
                                             where=f"default of input {name!r}")
        self._inputs.append(decl)
        return Ref(type=type, input_index=index, label=f"input:{name}")

    @property
    def inputs(self) -> list[dict[str, Any]]:
        """The input declarations, in declaration order. Read-only view."""
        return [dict(d) for d in self._inputs]

    def input_names(self) -> list[str]:
        return [str(d["name"]) for d in self._inputs]

    # ---- literal helpers ---------------------------------------------------------------

    def literal_real(self, val: float, units: Mapping[str, int] | None = None) -> Ref:
        return Ref.from_literal(literal_real(val, units))

    def literal_integer(self, val: int) -> Ref:
        return Ref.from_literal(literal_integer(val))

    def literal_bool(self, val: bool) -> Ref:
        return Ref.from_literal(literal_bool(val))

    def literal_text(self, val: str) -> Ref:
        return Ref.from_literal(literal_text(val))

    def literal_file_path(self, path: str | os.PathLike[str]) -> Ref:
        return Ref.from_literal(literal_file_path(path))

    def literal_point(self, x: float, y: float, z: float) -> Ref:
        return Ref.from_literal(literal_point(x, y, z))

    def literal_vector(self, x: float, y: float, z: float,
                       units: Mapping[str, int] | None = None) -> Ref:
        return Ref.from_literal(literal_vector(x, y, z, units))

    def literal_real_field(self, expr: str, units: Mapping[str, int] | None = None) -> Ref:
        return Ref.from_literal(literal_real_field(expr, units))

    def literal_enum(self, type_name: str, index_or_id: int | str) -> Ref:
        return Ref.from_literal(literal_enum(type_name, index_or_id))

    def literal_unit_length(self, unit_id: str = "mm") -> Ref:
        return Ref.from_literal(literal_unit_length(unit_id))

    # ---- block lookup -------------------------------------------------------------------

    def latest(self, base_signature: str) -> str:
        """Newest usable revision of a base signature.

        Prefers `BLOCK_REVISION_OVERRIDES` (revisions proven to exist in `ntopcl` but missing
        from the vendored universe), otherwise defers to `Universe.latest`.
        """
        base, _ = split_signature(base_signature)
        override = BLOCK_REVISION_OVERRIDES.get(base)
        if override:
            return override
        return self.universe.latest(base)

    def describe(self, func_id: str) -> FunctionDesc:
        """Descriptor for a block id, tolerating a revision the vendored universe lacks.

        When `func_id` names a revision the universe does not know but its base signature IS
        known, the newest known revision's input list is used for arity and type checking and
        the requested id is emitted verbatim. A revision bump does not usually change the input
        list, and `ntopcl convert` is the final arbiter. A completely unknown base signature
        still raises.
        """
        try:
            return self.universe.get(func_id)
        except UnknownFunctionError:
            base, rev = split_signature(func_id)
            if base == func_id:
                raise
            known = self.universe.revisions(base)
            if not known:
                raise
            newest = known[-1]
            log.debug(
                "block %s is not in the vendored universe; checking arity against %s",
                func_id, newest.func_id,
            )
            return dc_replace(newest, func_id=func_id, revision=rev)

    # ---- blocks ------------------------------------------------------------------------

    def block(
        self,
        func_id: str,
        *args: Any,
        name: str | None = None,
        allow_unfilled: bool = False,
        **kwargs: Any,
    ) -> Ref:
        """Create a block and return a `Ref` to it.

        `func_id` is looked up in the `Universe`, so a typo or a stale revision fails here.
        Positional args fill input slots in order; keyword args fill them by input NAME (the
        `name` field in `functions.json`, matched case-insensitively with spaces intact, or
        the same name lowercased with spaces replaced by underscores).

        Each argument may be a `Ref`, an already-built literal dict, a bare Python scalar
        (promoted to the literal type and units that slot requires), a 3-sequence for a
        `point`/`vector` slot, or `None` for an unfilled slot.
        """
        desc = self.describe(func_id)
        args_list = self._bind_arguments(desc, args, kwargs, allow_unfilled)
        blk = _Block(
            func=desc.func_id,
            block_id=self._new_id(),
            var_id=self._new_id(),
            return_type=desc.return_type,
            name=self._unique_name(name or desc.displayname or desc.name),
            args=args_list,
            desc=desc,
        )
        self._entries.append(blk)
        return Ref(type=blk.return_type, block=blk, label=desc.func_id)

    def raw_block(
        self,
        func_id: str,
        return_type: str,
        args: Sequence[Any],
        name: str | None = None,
    ) -> Ref:
        """Escape hatch for a block that `ntopcl` knows but the vendored universe does not.

        No arity or type checking happens: `args` are rendered as given. Prefer `block()`.
        """
        blk = _Block(
            func=func_id,
            block_id=self._new_id(),
            var_id=self._new_id(),
            return_type=return_type,
            name=self._unique_name(name),
            args=list(args),
            desc=None,
        )
        self._entries.append(blk)
        return Ref(type=return_type, block=blk, label=func_id)

    def variable(
        self,
        name: str,
        value: Any = None,
        type: str | None = None,
        description: str = "",
    ) -> Ref:
        """Wrap a value in a named root variable and return a `Ref` to it.

        If `value` is a `Ref` to a block with no property chain, the block's own variable is
        renamed in place instead of adding a second variable, so `r.variable("Body", sph)`
        does the obvious thing without extra indirection.
        """
        if isinstance(value, Ref) and value.block is not None and not value.props:
            value.block.name = self._unique_name(name)
            return value

        if type is None:
            if isinstance(value, Ref):
                type = value.type
            elif isinstance(value, dict) and "type" in value:
                type = str(value["type"])
            else:
                raise RecipeError(
                    f"variable {name!r}: give an explicit type= for value {value!r}"
                )
        contents: Any
        if value is None or isinstance(value, (Ref, dict)):
            contents = value
        else:
            contents = self._promote(value, type, {}, where=f"variable {name!r}")
        var = _ValueVariable(
            var_id=self._new_id(),
            return_type=type,
            name=self._unique_name(name),
            contents=contents,
            description=description,
        )
        self._entries.append(var)
        return Ref(type=type, variable=var, label=f"var:{name}")

    # ---- lists -------------------------------------------------------------------------

    def list_of(self, type_name: str, items: Iterable[Any], name: str | None = None) -> Ref:
        """Build a `core.list<T>` block.

        `core.list<T>` is not in the vendored function universe, so it is emitted directly
        (this is what `notebook_author` does too, and it matches `exportjson` output).

        REFERENCE.md warns that a literal `list<point>` LEAF is dropped by `exportjson`, so
        never build a list as a literal; always use this block form. `point_list()` goes one
        step further and wraps each point in a `point<real,real,real>` block.
        """
        rendered: list[Any] = []
        for it in items:
            if it is None or isinstance(it, (Ref, dict)):
                rendered.append(it)
            else:
                rendered.append(self._promote(it, type_name, {},
                                              where=f"element of list<{type_name}>"))
        blk = _Block(
            func=f"core.list<{type_name}>",
            block_id=self._new_id(),
            var_id=self._new_id(),
            return_type=f"list<{type_name}>",
            name=self._unique_name(name),
            args=rendered,
            desc=None,
        )
        self._entries.append(blk)
        return Ref(type=blk.return_type, block=blk, label=blk.func)

    def point_list(
        self,
        points: Iterable[Sequence[float] | Ref],
        name: str | None = None,
    ) -> Ref:
        """Build a `core.list<point>` of `point<real,real,real>` BLOCKS, not point literals.

        REFERENCE.md section 3: `exportjson` drops literal `list<point>` leaves, so a list of
        point literals cannot round-trip. Wrapping each point in a `point<real,real,real>`
        block avoids that.
        """
        items: list[Ref] = []
        for p in points:
            if isinstance(p, Ref):
                items.append(p)
                continue
            x, y, z = (float(v) for v in p)
            items.append(self.block("point<real,real,real>", x, y, z, name=None))
        return self.list_of("point", items, name=name)

    # ---- convenience emitters ----------------------------------------------------------

    def mesh_from_implicit(
        self,
        body: Ref,
        tolerance: float,
        min_feature_size: float | None = None,
        sharpen: bool = False,
        simplify: bool = True,
        name: str = "Mesh",
    ) -> Ref:
        """`implicit_to_mesh` at its newest revision. `tolerance` is metres."""
        func = self.latest("implicit_to_mesh<implicit,real,real,bool,bool>")
        return self.block(
            func, body, tolerance, min_feature_size, sharpen, simplify, name=name
        )

    def export_stl(
        self,
        target: Ref,
        path: str | os.PathLike[str],
        units: str = "mm",
        tolerance: float = 1.0e-4,
    ) -> Ref:
        """Export a mesh to STL. An `implicit` target is meshed first at `tolerance` metres.

        `path` may omit the extension; `.stl` is appended when there is none.
        """
        mesh = target
        if target.type != "mesh":
            mesh = self.mesh_from_implicit(target, tolerance=tolerance,
                                           name="Mesh for STL")
        p = _with_extension(path, ".stl")
        func = self.latest("export_mesh<file_path,mesh,unit_length_enum>")
        return self.block(func, literal_file_path(p), mesh,
                          literal_unit_length(units), name="Export STL")

    def export_step(self, part: Ref, path: str | os.PathLike[str]) -> Ref:
        """Export a `part` (or `list<part>`) to STEP. Needs nTop CAD interop at run time."""
        sig = ("export_part<file_path,list<part>>" if part.type.startswith("list<")
               else "export_part<file_path,part>")
        func = self.latest(sig)
        p = _with_extension(path, ".step")
        return self.block(func, literal_file_path(p), part, name="Export STEP")

    def export_implicit(self, body: Ref, path: str | os.PathLike[str]) -> Ref:
        """Export an implicit body to `.implicit` (readable by nTopCore)."""
        func = self.latest("export_implicit_body<file_path,implicit>")
        p = _with_extension(path, ".implicit")
        return self.block(func, literal_file_path(p), body, name="Export Implicit")

    def export_table(self, table: Ref, path: str | os.PathLike[str]) -> Ref:
        func = self.latest("export_table<file_path,table>")
        p = _with_extension(path, ".csv")
        return self.block(func, literal_file_path(p), table, name="Export Table")

    def export_text(self, text: Ref, path: str | os.PathLike[str]) -> Ref:
        sig = ("export_text<file_path,list<text>>" if text.type.startswith("list<")
               else "export_text<file_path,text>")
        func = self.latest(sig)
        p = _with_extension(path, ".txt")
        return self.block(func, literal_file_path(p), text, name="Export Text")

    def mass_properties(
        self,
        body: Ref,
        density: float | Ref = 1.0,
        relative_error: float = 0.01,
        name: str = "Mass Properties",
    ) -> Ref:
        """`mass_properties` on an implicit or a mesh, newest revision.

        `density` is kg/m^3. The result is a `body_mass_props`; use `.prop("volume")`,
        `.prop("mass")`, `.prop("center of gravity")` or `.prop("principal moments")` to pull
        scalars out of it.
        """
        if body.type == "mesh":
            func = self.latest("mass_properties<mesh,real>")
            return self.block(func, body, density, name=name)
        func = self.latest("mass_properties<implicit,real_field,real>")
        return self.block(func, body, density, relative_error, name=name)

    def surface_area(self, body: Ref, relative_error: float = 0.01,
                     name: str = "Surface Area") -> Ref:
        """Surface area in m^2. Uses `surface_area<...>`, not the deprecated
        `body_surface_area<...>`."""
        if body.type == "mesh":
            func = self.latest("surface_area<mesh>")
            return self.block(func, body, name=name)
        sig = ("surface_area<implicit_2d,real>" if body.type == "implicit_2d"
               else "surface_area<implicit,real>")
        func = self.latest(sig)
        return self.block(func, body, relative_error, name=name)

    def dimensionless(self, value: Ref, units: Mapping[str, int],
                      name: str | None = None) -> Ref:
        """Divide a dimensioned `real` by a literal 1 of the same unit, giving a pure number.

        `core.list<real>` rejects elements whose units differ, so a volume (m^3), a mass (kg)
        and an area (m^2) cannot share one list. Dividing each by 1 of its own unit fixes that,
        and the resulting numbers are in SI because nTop stores SI internally.
        Verified error message when this step is skipped:
        "The units of inputs '0' (units of length^3) and '1' (units of mass) do not match."
        """
        one = literal_real(1.0, units)
        return self.block("divide<real,real>", value, one, name=name)

    def json_output(
        self,
        values: Mapping[str, tuple[Ref, Mapping[str, int]] | Ref],
        name: str = "Measurements",
    ) -> Ref:
        """Pack many named scalars into ONE `json` value and make it the notebook output.

        A recipe has exactly one output slot (see `set_output`), so this is how a notebook
        reports a whole measurement set. `driver.parse_outputs` unpacks the resulting
        `{"jsonObject": {...}}` and maps its keys onto `NtopMeasurements`.

        Each entry is either a bare `Ref` (already dimensionless) or a
        `(ref, units)` pair, in which case the value is divided by 1 of `units` first.

        The three blocks used here are absent from the vendored universe but accepted by
        `ntopcl convert`, so they are emitted with `raw_block`. Verified end to end on this
        machine: convert, `-t`, and a run all succeed and the output JSON carries the
        dictionary (see `runs/_smoke/_probe_json_output.py`).
        """
        keys: list[str] = []
        refs: list[Ref] = []
        for key, entry in values.items():
            keys.append(str(key))
            if isinstance(entry, Ref):
                refs.append(entry)
            else:
                ref, units = entry
                refs.append(self.dimensionless(ref, units, name=str(key)))
        names = self.list_of("text", keys, name=f"{name} Names")
        vals = self.list_of("real", refs, name=f"{name} Values")
        table = self.raw_block(
            "core.dictionary<list<text>,list<real>>",
            "dictionary<text,real>",
            [names, vals],
            name=name,
        )
        out = self.raw_block(
            # The [5.30.0] suffix is required; the unversioned id is rejected by convert.
            "json_from_dictionary<dictionary<text,real>>[5.30.0]",
            "json",
            [table],
            name=f"{name} JSON",
        )
        self.set_output(out)
        return out

    # ---- the notebook output -----------------------------------------------------------

    def set_output(self, value: Ref, name: str | None = None) -> Ref:
        """Designate the notebook's single Automate output.

        A recipe carries at most ONE output, encoded as a top-level `{"output": {"id": ...}}`
        pointing at a root entry (verified by `exportjson` on two production notebooks; see
        `docs/NTOP_NOTES.md`). Without it, `ntopcl -t` reports
        "Error generating output template : Output of function not set" and `-o` writes
        nothing. To report many quantities, make the output a composite value.
        """
        ref = value
        if ref.props or ref.literal is not None or ref.input_index is not None:
            ref = self.variable(name or "Output", value)
        elif name:
            ref = self.variable(name, value)
        self._output_ref = ref
        return ref

    @property
    def output_ref(self) -> Ref | None:
        return self._output_ref

    # ---- rendering ---------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Render the whole recipe. REFERENCE.md section 5 lists the top-level keys."""
        body = [e.render_variable() for e in self._entries]
        doc: dict[str, Any] = {
            "body": body,
            "cbRefs": [],
            "description": self.description,
            "displayname": self.displayname,
            "imports": [],
            "inputs": [dict(d) for d in self._inputs],
            "name": self.internal_name,
            "namespaces": [],
            "version": [1, 0, 0],
        }
        if self._output_ref is not None:
            target = self._output_ref
            var_id = target.block.var_id if target.block else target.variable.var_id  # type: ignore[union-attr]
            doc["output"] = {"id": var_id}
        return doc

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def write_json(self, path: str | os.PathLike[str]) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return str(path)

    def to_ntop(
        self,
        path: str | os.PathLike[str],
        ntopcl: Any = None,
        recipe_json: str | os.PathLike[str] | None = None,
    ) -> str:
        """Write the recipe and run `ntopcl convert` on it. Returns the `.ntop` path.

        `ntopcl` may be a path, an existing `driver.NtopRunner`, or None to resolve one.
        """
        from .driver import NtopRunner    # local import: driver imports nothing from here

        runner = ntopcl if isinstance(ntopcl, NtopRunner) else NtopRunner(ntopcl)
        json_path = recipe_json or (os.path.splitext(str(path))[0] + "_recipe.json")
        self.write_json(json_path)
        runner.convert(json_path, path)
        return str(path)

    # ---- argument binding and literal promotion -----------------------------------------

    def _bind_arguments(
        self,
        desc: FunctionDesc,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        allow_unfilled: bool,
    ) -> list[Any]:
        n = len(desc.inputs)
        variadic = bool(desc.inputs) and desc.inputs[-1].is_variadic

        if len(args) > n and not variadic:
            raise ArityError(
                f"{desc.func_id} takes {n} input(s) but {len(args)} positional argument(s) "
                f"were given.\n  inputs: {desc.summary()}"
            )

        slots: list[Any] = [_UNSET] * max(n, len(args))
        for i, a in enumerate(args):
            slots[i] = a

        for key, val in kwargs.items():
            idx = _kwarg_index(desc, key)
            if idx < len(slots) and slots[idx] is not _UNSET:
                raise ArityError(
                    f"{desc.func_id}: input {desc.inputs[idx].name!r} (index {idx}) was given "
                    f"both positionally and by keyword {key!r}"
                )
            slots[idx] = val

        out: list[Any] = []
        for i, raw in enumerate(slots):
            slot = desc.inputs[min(i, n - 1)] if n else None
            if raw is _UNSET or raw is None:
                if raw is _UNSET and slot is not None and not slot.is_optional \
                        and not allow_unfilled:
                    raise ArityError(
                        f"{desc.func_id}: required input {i} {slot.name!r} ({slot.type}) is "
                        f"not filled. Pass None explicitly, or allow_unfilled=True, if that "
                        f"is intended.\n  inputs: {desc.summary()}"
                    )
                out.append(None)
                continue
            out.append(self._coerce(raw, slot, desc, i))
        return out

    def _coerce(self, raw: Any, slot: InputDesc | None, desc: FunctionDesc, i: int) -> Any:
        if isinstance(raw, Ref):
            return raw
        if isinstance(raw, dict) and "type" in raw:
            return raw
        if slot is None:
            raise LiteralTypeError(f"{desc.func_id}: no input slot {i}")
        where = f"{desc.func_id} input {i} {slot.name!r}"
        return self._promote(raw, slot.type, slot.units_req, where=where)

    def _promote(
        self,
        value: Any,
        type_name: str,
        units: Mapping[str, int],
        where: str,
    ) -> dict[str, Any]:
        """Promote a bare Python value to the literal encoding for `type_name`."""
        if isinstance(value, Ref):
            raise LiteralTypeError(f"{where}: internal error, Ref reached _promote")
        if isinstance(value, dict) and "type" in value:
            return dict(value)

        t = type_name
        if t == "bool":
            if not isinstance(value, bool):
                raise LiteralTypeError(f"{where} is a bool slot; got {value!r}")
            return literal_bool(value)
        if t == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                if isinstance(value, float) and float(value).is_integer():
                    return literal_integer(int(value))
                raise LiteralTypeError(f"{where} is an integer slot; got {value!r}")
            return literal_integer(value)
        if t in ("real", "temperature"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise LiteralTypeError(f"{where} is a {t} slot; got {value!r}")
            return literal_real(value, units)
        if t == "real_field":
            if isinstance(value, str):
                return literal_real_field(value, units)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise LiteralTypeError(
                    f"{where} is a real_field slot; give a number or an expression string, "
                    f"got {value!r}"
                )
            # A `real` literal is accepted in a `real_field` slot; nTop auto-converts. This is
            # what a GUI-authored notebook contains (verified in that reference notebook exportjson,
            # variable "Radius": type real_field, contents a real literal).
            return literal_real(value, units)
        if t == "text":
            if not isinstance(value, str):
                raise LiteralTypeError(f"{where} is a text slot; got {value!r}")
            return literal_text(value)
        if t == "file_path":
            if not isinstance(value, (str, os.PathLike)):
                raise LiteralTypeError(f"{where} is a file_path slot; got {value!r}")
            return literal_file_path(value)
        if t == "point":
            xyz = _as_triple(value, where, "point")
            return literal_point(*xyz)
        if t == "vector":
            xyz = _as_triple(value, where, "vector")
            return literal_vector(*xyz, units=units)
        if t.endswith("_enum") or t == "unit_length_enum":
            if isinstance(value, (int, str)) and not isinstance(value, bool):
                return literal_enum(t, value)
            raise LiteralTypeError(f"{where} is a {t} slot; give an int index or a str id")
        raise LiteralTypeError(
            f"{where} needs a {t}, which has no literal form here. Build it as a block and "
            f"pass the Ref. Types with a literal form: {sorted(_SCALAR_LITERAL_TYPES)} plus "
            f"any *_enum."
        )


class _Unset:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()


def _kwarg_index(desc: FunctionDesc, key: str) -> int:
    """Resolve a keyword argument to an input index.

    Accepts the exact input name, the name case-insensitively, and a pythonised form where
    spaces, dots and dashes become underscores (so `min__feature_size` is not needed:
    `min_feature_size` matches "Min. feature size").
    """
    try:
        return desc.input_index(key)
    except KeyError:
        pass
    want = _pythonise(key)
    for i in desc.inputs:
        if _pythonise(i.name) == want:
            return i.index
    raise ArityError(
        f"{desc.func_id} has no input matching keyword {key!r}. "
        f"Inputs are: {list(desc.input_names)}"
    )


def _pythonise(s: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.strip().lower())).strip("_")


def _as_triple(value: Any, where: str, kind: str) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            pass
    raise LiteralTypeError(
        f"{where} is a {kind} slot; give a 3-sequence of numbers or a Ref, got {value!r}"
    )


def _with_extension(path: str | os.PathLike[str], ext: str) -> str:
    p = to_ntop_path(path)
    root, cur = os.path.splitext(p)
    return p if cur else root + ext


def _make_internal_name(rng: random.Random) -> str:
    """`user_func_<8>_<4>_<4>_<4>_<12>` of random alphanumerics. REFERENCE.md section 5."""
    alphabet = string.ascii_letters + string.digits
    s = "".join(rng.choices(alphabet, k=32))
    return f"user_func_{s[:8]}_{s[8:12]}_{s[12:16]}_{s[16:20]}_{s[20:32]}"
