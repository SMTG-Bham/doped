"""
GPAW calculation input file generation (``doped.io.gpaw.inputs``) and output
parsing (``doped.io.gpaw.outputs``) for ``doped``.

GPAW support is currently **experimental**. The submodule docstrings list what
this means in practice, and the protocol entry points which ``doped`` requires
raise ``NotImplementedError``. See the GPAW tracking issue.

Submodule attributes can be accessed directly from this package (e.g.
``from doped.io.gpaw import GPAWDefectsParser``); they are imported lazily to
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
