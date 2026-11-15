"""
GPAW calculation input file generation (``doped.io.gpaw.inputs``) and output
parsing (``doped.io.gpaw.outputs``) for ``doped``.

GPAW support is currently **experimental**, but implements the core of the
``doped.io`` backend protocol, so GPAW calculations can be parsed with
``DefectsParser(..., calculator="gpaw")``. The submodule docstrings list what
is still missing; those entry points raise ``NotImplementedError``. See the
GPAW tracking issue.

Submodule attributes can be accessed directly from this package (e.g.
``from doped.io.gpaw import DefectsSet``); they are imported lazily to
avoid unnecessary import costs.
"""

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    for submodule in ("inputs", "outputs"):
        module = import_module(f"doped.io.gpaw.{submodule}")
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
