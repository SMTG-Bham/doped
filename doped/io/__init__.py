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

from pymatgen.util.typing import PathLike

from doped.io.outputs import CalculationOutputs


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
    return import_module(f"doped.io.{calculator.lower()}.outputs").get_calculation_outputs(path, **kwargs)
