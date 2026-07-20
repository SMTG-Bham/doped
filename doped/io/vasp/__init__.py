"""
VASP calculation input file generation (``doped.io.vasp.inputs``) and output
parsing (``doped.io.vasp.outputs``) for ``doped``.

Submodule attributes can be accessed directly from this package (e.g.
``from doped.io.vasp import DefectsSet``); they are imported lazily to avoid
unnecessary import costs and circular imports.
"""

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    for submodule in ("inputs", "outputs"):
        module = import_module(f"doped.io.vasp.{submodule}")
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
