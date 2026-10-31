"""
Deprecated alias for ``doped.parsing``.

``doped.analysis`` has been renamed to ``doped.parsing``, matching its contents
(defect calculation parsing & defect identification), while
``shallow_dopant_binding_energy`` has moved to ``doped.thermodynamics``. This
shim forwards the old names with deprecation warnings, and will be removed in
a future release.
"""

import warnings
from typing import Any


def __getattr__(name: str) -> Any:
    from doped import parsing, thermodynamics

    for module in (parsing, thermodynamics):
        if hasattr(module, name):
            warnings.warn(
                f"doped.analysis has been renamed to doped.parsing; import {name} from "
                f"{module.__name__} instead. This deprecated alias will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(module, name)
    raise AttributeError(f"module 'doped.analysis' has no attribute {name!r}")
