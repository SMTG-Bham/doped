"""
Escape-hatch ``doped.io`` backend which reads pre-serialised
:class:`~doped.io.outputs.CalculationOutputs` JSON files, rather than raw
calculator output files.

This allows the ``doped`` parsing & analysis workflow to be used with `any`
calculator, without a dedicated ``doped.io.<calculator>`` backend: build the
:class:`~doped.io.outputs.CalculationOutputs` objects however you like
(populating the optional attributes needed for your desired analyses -- see
its docstring), save each to its calculation directory with e.g.
``dumpfn(outputs, "<calculation directory>/calculation_outputs.json.gz")``,
and then parse as usual with ``DefectsParser(..., calculator="serialized")``
or ``DefectParser.from_paths(..., calculator="serialized")``.
"""

from monty.serialization import loadfn
from pymatgen.util.typing import PathLike

from doped.io.outputs import CalculationOutputs
from doped.io.utils import _get_output_files_and_check_if_multiple, _multiple_files_warning

CALC_OUTPUT_MASK = ("calculation_outputs.json", "calculation_outputs.json.gz")
"""
Filename patterns identifying serialised calculation output files, used for
calculation folder discovery.

Part of the ``doped.io`` backend protocol.
"""

FILE_PARSING_ACTIONS = {
    "calculation_outputs.json": "parse the serialised calculation outputs.",
}
"""
The calculation output file types parsed by this backend, and what they are
used for (for informative warning messages).

Part of the ``doped.io`` backend
protocol.
"""


def get_calculation_outputs(path: PathLike, label: str = "calculation", **kwargs) -> CalculationOutputs:
    """
    Load the pre-serialised :class:`~doped.io.outputs.CalculationOutputs` JSON
    file (``calculation_outputs.json(.gz)``) from the calculation directory
    ``path``.

    Part of the ``doped.io`` backend protocol.

    Args:
        path (PathLike):
            Path to the calculation directory, containing a
            ``calculation_outputs.json(.gz)`` file.
        label (str):
            Label for the type of calculation being parsed (e.g. ``"bulk"``,
            ``"defect"``), for informative warnings. Default is
            ``"calculation"``.
        **kwargs:
            Ignored (accepted for compatibility with the generic backend
            calling convention; all data comes pre-parsed from the serialised
            file).

    Returns:
        CalculationOutputs: The loaded calculation outputs.
    """
    fname, multiple = _get_output_files_and_check_if_multiple("calculation_outputs.json", path)
    if multiple:
        _multiple_files_warning(
            "calculation_outputs.json",
            path,
            fname,
            action=FILE_PARSING_ACTIONS["calculation_outputs.json"],
            dir_type=label,
        )
    try:
        outputs = loadfn(fname)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"calculation_outputs.json(.gz) not found at {path}. Needed for parsing calculation outputs "
            f"with the 'serialized' backend!"
        ) from exc

    if not isinstance(outputs, CalculationOutputs):  # plain-dict JSON (no MSON @class info)
        outputs = CalculationOutputs.from_dict(outputs)
    outputs.calculator = outputs.calculator or "serialized"
    outputs.directory = path
    return outputs
