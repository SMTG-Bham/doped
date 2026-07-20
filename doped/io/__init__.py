"""
Calculator-specific input/output code for ``doped``.

This subpackage separates all calculator-specific code (input file
generation, output parsing) from the calculator-agnostic core of ``doped``.
Each supported calculator has a ``doped.io.<calculator>`` subpackage with:

- ``inputs``: calculation input file generation
- ``outputs``: output parsing, providing a ``get_calculation_outputs()``
  function which returns a calculator-agnostic
  :class:`~doped.io.outputs.CalculationOutputs` object

See the "Adding Support for a New Calculator" docs page for details on
implementing support for additional calculators.
"""

from importlib import import_module
from types import ModuleType

from pymatgen.util.typing import PathLike

from doped.io.outputs import CalculationOutputs


def get_backend(calculator: str = "vasp") -> ModuleType:
    """
    Get the output-parsing backend module (``doped.io.<calculator>.outputs``)
    for the given calculator.

    Backend modules provide a ``get_calculation_outputs()`` function
    (returning a calculator-agnostic
    :class:`~doped.io.outputs.CalculationOutputs` object) and a
    ``CALC_OUTPUT_MASK`` constant (filename patterns identifying calculation
    output files, for calculation folder discovery), plus optional further
    constants/functions used by ``doped``'s calculator-agnostic parsing
    machinery (``SUBFOLDER_PRIORITY``, ``FILE_PARSING_ACTIONS``,
    ``get_planar_averaged_potentials()``, ``get_site_potentials()`` etc.) --
    see the "Adding Support for a New Calculator" docs page for details.

    Args:
        calculator (str):
            Name of the calculator (matching a ``doped.io.<calculator>``
            subpackage). Default: "vasp".

    Returns:
        ModuleType: The ``doped.io.<calculator>.outputs`` backend module.
    """
    backend_name = f"doped.io.{calculator.lower()}.outputs"
    try:
        return import_module(backend_name)
    except ModuleNotFoundError as exc:
        if exc.name and backend_name.startswith(exc.name):  # backend itself missing, not its dependencies
            raise ValueError(
                f"Unrecognised calculator {calculator!r}: no `{backend_name}` backend module found. See "
                f"the 'Adding Support for a New Calculator' docs page for implementing support for new "
                f"calculators."
            ) from exc
        raise


def get_calculation_outputs(path: PathLike, calculator: str = "vasp", **kwargs) -> CalculationOutputs:
    """
    Parse the outputs of a supercell calculation in ``path`` to a (calculator-
    agnostic) :class:`~doped.io.outputs.CalculationOutputs` object, using the
    parser for the given ``calculator``.

    Args:
        path (PathLike): Path to the calculation directory.
        calculator (str): Name of the calculator used (matching a
            ``doped.io.<calculator>`` subpackage). Default: "vasp".
        **kwargs: Additional keyword arguments to pass to the calculator's
            ``get_calculation_outputs()`` function.

    Returns:
        CalculationOutputs: The parsed calculation outputs.
    """
    return get_backend(calculator).get_calculation_outputs(path, **kwargs)
