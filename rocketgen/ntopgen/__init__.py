"""nTop notebook authoring and execution (WP1).

`universe` describes the block universe, `recipe` builds the notebook-recipe JSON, `driver`
drives `ntopcl`. See `docs/REFERENCE.md` for the verified toolchain facts and
`docs/NTOP_NOTES.md` for everything learned while building this.
"""
from .driver import (
    OUTPUT_NAME_MAP,
    SUCCESS_RETURNCODES,
    NtopError,
    NtopRunner,
    ParsedOutputs,
    RunResult,
    parse_outputs,
)
from .recipe import (
    BLOCK_REVISION_OVERRIDES,
    ArityError,
    LiteralTypeError,
    Recipe,
    RecipeError,
    Ref,
    to_ntop_path,
)
from .universe import (
    FunctionDesc,
    InputDesc,
    TypeDesc,
    UnknownFunctionError,
    Universe,
    parse_revision,
    split_signature,
)

__all__ = [
    "BLOCK_REVISION_OVERRIDES",
    "ArityError",
    "FunctionDesc",
    "InputDesc",
    "LiteralTypeError",
    "NtopError",
    "NtopRunner",
    "OUTPUT_NAME_MAP",
    "ParsedOutputs",
    "Recipe",
    "RecipeError",
    "Ref",
    "RunResult",
    "SUCCESS_RETURNCODES",
    "TypeDesc",
    "UnknownFunctionError",
    "Universe",
    "parse_outputs",
    "parse_revision",
    "split_signature",
    "to_ntop_path",
]
