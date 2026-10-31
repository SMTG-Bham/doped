"""
Deprecated alias for ``doped.io.vasp.inputs``.

``doped.vasp`` has moved to ``doped.io.vasp.inputs`` as part of the
``doped.io`` calculator input/output framework. This shim will be removed in
a future release.
"""

import warnings
from typing import Any


def __getattr__(name: str) -> Any:
    from doped.io.vasp import inputs

    if hasattr(inputs, name):
        warnings.warn(
            f"doped.vasp has moved to doped.io.vasp.inputs; import {name} from there instead. "
            "This deprecated alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(inputs, name)
    raise AttributeError(f"module 'doped.vasp' has no attribute {name!r}")
