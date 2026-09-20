"""
GPAW calculation input file generation (``doped.io.gpaw.inputs``) and output
parsing (``doped.io.gpaw.outputs``) for ``doped``.

GPAW support is **experimental**. Unlike the reference ``doped.io.vasp``
backend, it is not yet wired into ``doped``'s calculator-agnostic backend
protocol (see the "Adding Support for a New Calculator" docs page), so GPAW
calculations are generated and parsed with the GPAW-specific classes here
rather than with :class:`~doped.parsing.DefectsParser` /
:class:`~doped.io.vasp.inputs.DefectsSet`. The submodule docstrings list what
this means in practice, and the protocol entry points raise
``NotImplementedError`` rather than failing obscurely. See the GPAW tracking
issue.

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
