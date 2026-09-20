"""
Parsing of GPAW defect / bulk supercell calculation outputs.

GPAW support is experimental, but implements the core of the ``doped.io``
backend protocol, so GPAW calculations are parsed with ``doped``'s generic
machinery, as for any other calculator:
``DefectsParser(..., calculator="gpaw")`` and
``DefectParser.from_paths(..., calculator="gpaw")``.

Not implemented, and so unavailable with GPAW:

- ``projected_eigenvalues``, and thus ``load_eigenvalue_outputs()`` and
  eigenvalue analysis of band-edge & in-gap states
  (``DefectEntry.get_eigenvalue_analysis()``). GPAW's PAW projections are in
  the ``.gpw`` files, but are not yet mapped to ``pymatgen``'s format.
- ``get_fermi_dos()``: bulk DOS parsing, for Fermi level / carrier
  concentration analysis (``FermiSolver``).
- ``get_competing_phase_entry()``: competing phase parsing, for chemical
  potential analysis (``CompetingPhasesAnalyzer``).
- Occupation-based band edges, see
  :func:`_get_eigenvalue_properties_from_calc`.

See the GPAW tracking issue.
"""

import contextlib
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.util.typing import PathLike

from doped.io.outputs import CalculationOutputs

# ``doped`` accesses these two remaining unimplemented protocol names directly (rather than probing
# with ``getattr(backend, name, default)``), so they are intercepted here to fail informatively rather
# than with an obscure ``AttributeError``. The optional, ``getattr``-probed protocol names
# (``SUBFOLDER_PRIORITY``, ``MISMATCH_WARNING_SPECS``, ``check_entry_compatibility``,
# ``load_eigenvalue_outputs``, ``PLANAR_POTENTIALS_FILE``, ``SITE_POTENTIALS_FILE``) must keep raising
# ``AttributeError``, so that those features degrade gracefully as intended:
_UNIMPLEMENTED_BACKEND_ATTRS = (
    "get_competing_phase_entry",
    "get_fermi_dos",
)


def __getattr__(name: str) -> Any:
    """
    Raise an informative error for the ``doped.io`` backend protocol
    functions/constants which are not implemented for GPAW (see the module
    docstring).
    """
    if name in _UNIMPLEMENTED_BACKEND_ATTRS:
        raise NotImplementedError(
            f"`{__name__}.{name}` is not implemented; GPAW support in `doped` is experimental, and "
            f"does not yet cover competing phase or bulk DOS parsing. Defect & bulk supercells are "
            f'parsed as usual, with `DefectsParser(..., calculator="gpaw")`. For competing phases or '
            f"a bulk DOS, build the `pymatgen` `ComputedStructureEntry` / `FermiDos` objects yourself "
            f'(e.g. from `get_calculation_outputs(..., calculator="gpaw")`) and pass them to '
            f"`CompetingPhasesAnalyzer` / `DefectThermodynamics` directly. See the GPAW tracking issue."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_GPAW_OUTPUT_PRIORITY = (
    "relaxed.gpw",
    "singlepoint.gpw",
    "final.gpw",
)


def _find_gpaw_output(
    output_path: str | os.PathLike,
    subfolder: str | os.PathLike | None = None,
) -> str:
    """
    Find a GPAW restart file in a calculation directory.

    If multiple restart files are present, standard relaxation, single-point,
    and final-state filenames are preferred in that order.
    """
    calc_path = Path(output_path)
    if calc_path.is_file():
        if calc_path.name.lower().endswith(".gpw"):
            return str(calc_path)
        raise ValueError(f"GPAW output must be a '.gpw' file: {calc_path}")

    if subfolder is not None and subfolder != ".":
        calc_path /= Path(subfolder)

    if not calc_path.is_dir():
        raise FileNotFoundError(f"GPAW calculation directory not found: {calc_path}")

    gpw_files = [
        path for path in calc_path.iterdir() if path.is_file() and path.name.lower().endswith(".gpw")
    ]
    files_by_name = {path.name.lower(): path for path in gpw_files}
    for preferred_name in _GPAW_OUTPUT_PRIORITY:
        if preferred_name in files_by_name:
            return str(files_by_name[preferred_name])

    if len(gpw_files) == 1:
        return str(gpw_files[0])
    if not gpw_files:
        raise FileNotFoundError(f"No '.gpw' file found in: {calc_path}")

    filenames = ", ".join(sorted(path.name for path in gpw_files))
    raise ValueError(
        f"Multiple GPAW output files found in {calc_path}, with no preferred filename: {filenames}"
    )


def _get_site_potentials_from_calc(calc) -> np.ndarray:
    """
    Get the atom-centred electrostatic site potentials from a ``GPAW``
    calculator.

    ``GPAW``'s ``get_atomic_electrostatic_potentials()`` integrates the pseudo
    Hartree potential against each atom's L=0 compensation-charge shape
    function, which is the direct analogue of the average electrostatic
    potential at the core reported by ``VASP`` in the ``OUTCAR``. It is negated
    here to match the sign convention used by ``doped``/``pydefect`` for eFNV
    (Kumagai-Oba) corrections.

    Args:
        calc (GPAW): ``GPAW`` calculator object.

    Returns:
        np.ndarray: Atomic site potentials (in eV), one per atom.
    """
    return -1.0 * np.array(calc.get_atomic_electrostatic_potentials())


def _structure_from_calc(calc) -> Structure:
    """
    Get the calculation structure from a ``GPAW`` calculator, as a ``pymatgen``
    ``Structure``.

    The ``ase`` ``Atoms`` returned by ``GPAW`` still hold the live calculator,
    and a ``Structure`` converted from them keeps a reference to it -- making
    the ``Structure`` (and any ``DefectEntry`` built from it) unpicklable
    (breaking multiprocessed parsing and ``copy.deepcopy``). The calculator is
    therefore detached before converting, and the final magnetic moments
    re-attached as a site property.

    Args:
        calc (GPAW): ``GPAW`` calculator object.

    Returns:
        Structure: The calculation structure.
    """
    atoms = calc.get_atoms().copy()  # ``copy()`` drops the attached calculator
    atoms.calc = None
    structure = AseAtomsAdaptor.get_structure(atoms)
    with contextlib.suppress(Exception):  # not available for non-spin-polarised calculations
        structure.add_site_property("final_magmom", list(calc.get_magnetic_moments()))

    return structure


def _get_eigenvalue_properties_from_calc(calc) -> tuple[float, float, float, float]:
    """
    Get ``(band_gap, cbm, vbm, efermi)`` (eV) from a ``GPAW`` calculator.
    """
    # TODO: Band edges are taken purely from the Fermi level here (VBM = highest eigenvalue at or below
    # E_F, CBM = lowest above it), with no reference to occupations. That is only safe for a gapped
    # bulk with E_F in the gap: it gives meaningless edges for a metallic or heavily-smeared bulk, and
    # silently returns ``efermi`` for both edges (i.e. a zero gap) if no eigenvalue falls on one side.
    # It is also outright wrong for charged defect supercells, where a partially-occupied in-gap state
    # would be reported as a band edge -- so ``get_calculation_outputs`` only takes the band edges from
    # the bulk. Replace with an occupation-based determination, as done for VASP in
    # ``doped.utils.eigenvalues.band_edge_properties_from_outputs``; the occupations are available from
    # ``calc.get_occupation_numbers()``. See the GPAW tracking issue.
    efermi = calc.get_fermi_level()
    energies: list[float] = []
    for spin in range(calc.get_number_of_spins()):
        for kpt in range(len(calc.get_ibz_k_points())):
            energies.extend(calc.get_eigenvalues(kpt=kpt, spin=spin))

    below = [energy for energy in energies if energy <= efermi]
    above = [energy for energy in energies if energy > efermi]
    vbm = max(below) if below else efermi
    cbm = min(above) if above else efermi

    return cbm - vbm, cbm, vbm, efermi


def _get_planar_averaged_potential_from_calc(calc) -> dict[str, np.ndarray]:
    """
    Helper to extract planar-averaged potentials from a GPAW calculator.
    """
    # note that the FNV correction requires the bulk and defect potentials to be on the same grid
    v_ext = calc.get_electrostatic_potential()
    planar_averages = {}
    for i in range(3):
        axes = [0, 1, 2]
        axes.remove(i)
        planar_averages[str(i)] = v_ext.mean(axis=tuple(axes))

    return planar_averages


def get_gpaw_site_potentials(
    gpw_file: str | os.PathLike,
) -> np.ndarray:
    """
    Extracts atomic site potentials from a ``GPAW`` ``.gpw`` file.
    """
    from gpaw import GPAW

    gpw_file = _find_gpaw_output(gpw_file)
    calc = GPAW(gpw_file)
    site_potentials = _get_site_potentials_from_calc(calc)

    if hasattr(calc, "close"):
        calc.close()

    if hasattr(calc, "atoms") and calc.atoms:
        calc.atoms.calc = None

    return site_potentials


def get_gpaw_planar_averaged_potential(
    gpw_file: str | os.PathLike,
) -> dict[str, np.ndarray]:
    """
    Extracts planar-averaged potential from a ``GPAW`` ``.gpw`` file.
    """
    from gpaw import GPAW

    gpw_file = _find_gpaw_output(gpw_file)
    calc = GPAW(gpw_file)
    planar_averages = _get_planar_averaged_potential_from_calc(calc)

    if hasattr(calc, "close"):
        calc.close()

    return planar_averages


def get_potentials_from_input(
    potentials_input: str | os.PathLike | dict | list | np.ndarray,
    potential_type: str = "planar",
    dir_type: str = "",
    entry_energy: float | None = None,
    run_metadata: dict | None = None,
):
    """
    Get planar-averaged (``potential_type="planar"``) or atomic-site
    (``"site"``) electrostatic potentials from calculator-native inputs, for
    finite-size charge corrections.

    Accepts a path to a ``.gpw`` file (or to a calculation directory containing
    one), or already-parsed potentials (dict/list/array), which are returned
    as-is. :func:`get_calculation_outputs` parses both potential types into
    ``DefectEntry.calculation_metadata``, so the already-parsed case is the
    usual one here.

    Args:
        potentials_input (PathLike | dict | list | np.ndarray):
            The calculator-native potentials input (see above).
        potential_type (str):
            ``"planar"`` for planar-averaged potentials (Freysoldt/FNV
            correction) or ``"site"`` for atomic-site potentials
            (Kumagai/eFNV correction). Default is ``"planar"``.
        dir_type (str):
            The type of directory being parsed (e.g. ``"bulk"`` or
            ``"defect"``), for informative errors.
        entry_energy (float):
            Accepted for backend-protocol compatibility and otherwise
            unused; ``GPAW`` writes the potentials and the total energy to
            the same ``.gpw`` file, so there is no separate calculation to
            cross-check against (unlike ``VASP``'s ``OUTCAR``). Default is
            ``None``.
        run_metadata (dict):
            Accepted for backend-protocol compatibility and otherwise
            unused, as for ``entry_energy``. Default is ``None``.

    Returns:
        The planar-averaged potentials (dict of ``{axis: potentials}``) or
        the atomic-site potentials (array), depending on ``potential_type``.
    """
    if isinstance(potentials_input, str | os.PathLike):
        if potential_type == "planar":
            return get_gpaw_planar_averaged_potential(potentials_input)
        return get_gpaw_site_potentials(potentials_input)

    if not isinstance(potentials_input, dict | list | np.ndarray):
        raise TypeError(
            f"{dir_type or 'GPAW'} potentials input must be either a path to a '.gpw' file (or a "
            f"directory containing one), or already-parsed potentials, but got "
            f"{type(potentials_input)} instead."
        )

    return potentials_input


CALC_OUTPUT_MASK = (".gpw",)
"""
Filename patterns identifying ``GPAW`` calculation output files, used for
calculation folder discovery.
"""

FILE_PARSING_ACTIONS = {
    ".gpw": (
        "parse the calculation energy, metadata and electrostatic potentials (planar-averaged and "
        "atomic-site), and compute the charged-defect finite-size corrections."
    ),
}
"""
The ``GPAW`` calculation output file types parsed by ``doped``, and what they
are used for (for informative warning messages).
"""


def get_calculation_outputs(
    path: PathLike,
    label: str = "calculation",
    parse_projected_eigen: bool | None = None,
    subfolder: PathLike | None = None,
    **kwargs,
) -> CalculationOutputs:
    """
    Parse the outputs of a ``GPAW`` supercell calculation in ``path`` to a
    (calculator-agnostic) :class:`~doped.io.outputs.CalculationOutputs` object.

    Args:
        path (PathLike):
            Path to the calculation directory, or directly to a ``.gpw`` file.
        label (str):
            Label for the type of calculation being parsed (e.g. ``"bulk"``,
            ``"defect"``), for informative warnings. Default is
            ``"calculation"``.
        parse_projected_eigen (bool):
            Accepted for backend-protocol compatibility and otherwise
            **unused**: mapping ``GPAW``'s PAW projections to ``pymatgen``'s
            ``projected_eigenvalues`` format is not yet implemented, so
            eigenvalue analysis is unavailable with ``GPAW``. Default is
            ``None``.
        subfolder (PathLike):
            Optional subfolder within ``path`` containing the ``.gpw`` file.
            Default is ``None``.
        **kwargs:
            Ignored (accepted for compatibility with the generic backend
            calling convention).

    Returns:
        CalculationOutputs: The parsed calculation outputs.
    """
    from gpaw import GPAW

    gpw_file = _find_gpaw_output(path, subfolder)
    calc = GPAW(gpw_file, txt=None)
    try:
        charge = calc.parameters.get("charge") or 0
        band_gap, cbm, vbm, efermi = _get_eigenvalue_properties_from_calc(calc)
        eigenvalues = {}
        for spin_index, spin in enumerate((Spin.up, Spin.down)[: calc.get_number_of_spins()]):
            eigenvalues[spin] = np.array(
                [  # (n_kpoints, n_bands, 2); energy and occupancy, as ``pymatgen`` expects
                    np.stack(
                        [
                            calc.get_eigenvalues(kpt=kpt, spin=spin_index),
                            calc.get_occupation_numbers(kpt=kpt, spin=spin_index),
                        ],
                        axis=-1,
                    )
                    for kpt in range(len(calc.get_ibz_k_points()))
                ]
            )

        return CalculationOutputs(
            structure=_structure_from_calc(calc),
            energy=calc.get_potential_energy(),
            calculator="gpaw",
            directory=str(path),
            converged_electronic=bool(getattr(calc.scf, "converged", True)),
            # a ``.gpw`` restart file holds the final state only, with no ionic-step history to check:
            converged_ionic=None,
            efermi=efermi,
            eigenvalues=eigenvalues,
            kpoint_coords=calc.get_ibz_k_points(),
            kpoint_weights=calc.get_k_point_weights(),
            nelect=calc.get_number_of_electrons(),
            charge=charge,
            magnetization=calc.get_magnetic_moment(),
            # the Fermi-level band edges are only meaningful for a gapped, neutral bulk:
            vbm=vbm if not charge else None,
            cbm=cbm if not charge else None,
            band_gap=band_gap if not charge else None,
            planar_averaged_potentials={
                int(axis): potentials
                for axis, potentials in _get_planar_averaged_potential_from_calc(calc).items()
            },
            site_potentials=_get_site_potentials_from_calc(calc),
            run_metadata={"gpaw_parameters": dict(calc.parameters)},
        )
    finally:
        if hasattr(calc, "close"):
            calc.close()
        if getattr(calc, "atoms", None) is not None:
            calc.atoms.calc = None


_COMPATIBILITY_PARAMETERS = ("mode", "xc", "kpts", "setups", "spinpol", "convergence")
"""
The ``GPAW`` calculation parameters which must match between the bulk and
defect supercell calculations for their energies to be comparable.
"""


def check_run_compatibility(
    defect_outputs: CalculationOutputs,
    bulk_outputs: CalculationOutputs,
    warn: bool = True,
) -> dict:
    """
    Check that the defect and bulk ``GPAW`` calculations used compatible
    settings, and collect their parameters for
    ``DefectEntry.calculation_metadata``.

    Only the parameters in :data:`_COMPATIBILITY_PARAMETERS` are compared;
    ``GPAW``'s full parameter dictionaries are returned either way.

    Args:
        defect_outputs (CalculationOutputs): The parsed defect supercell outputs.
        bulk_outputs (CalculationOutputs): The parsed bulk supercell outputs.
        warn (bool):
            Whether to warn about mismatched parameters. Default is ``True``.

    Returns:
        dict: ``"run_metadata"`` (both parameter sets) and
        ``"mismatching_gpaw_parameters"`` (a ``{parameter: (defect, bulk)}``
        dict, or ``False`` if they match).
    """
    run_metadata = {
        f"{label}_gpaw_parameters": (outputs.run_metadata or {}).get("gpaw_parameters", {})
        for label, outputs in (("defect", defect_outputs), ("bulk", bulk_outputs))
    }
    defect_parameters = run_metadata["defect_gpaw_parameters"]
    bulk_parameters = run_metadata["bulk_gpaw_parameters"]
    mismatches = {
        parameter: (defect_parameters.get(parameter), bulk_parameters.get(parameter))
        for parameter in _COMPATIBILITY_PARAMETERS
        if defect_parameters.get(parameter) != bulk_parameters.get(parameter)
    }
    if mismatches and warn:
        mismatch_info = "\n".join(
            f"{parameter}: {defect!r} (defect) vs {bulk!r} (bulk)"
            for parameter, (defect, bulk) in mismatches.items()
        )
        warnings.warn(
            f"There are mismatching GPAW parameters between your bulk and defect calculations, which "
            f"may mean your energies are not comparable:\n{mismatch_info}"
        )

    return {"mismatching_gpaw_parameters": mismatches or False, "run_metadata": run_metadata}


def get_planar_averaged_potentials(
    path: PathLike, dir_type: str = "bulk", quiet: bool = False
) -> dict[str, np.ndarray]:
    """
    Get the planar-averaged electrostatic potentials from the ``GPAW``
    calculation in ``path``, for the Freysoldt (FNV) charge correction.

    Note that :func:`get_calculation_outputs` already parses these, so
    ``doped`` only calls this when they were not parsed up-front.

    Args:
        path (PathLike): Path to the calculation directory or ``.gpw`` file.
        dir_type (str): The type of directory being parsed (``"bulk"`` or
            ``"defect"``), for informative errors. Default is ``"bulk"``.
        quiet (bool): Accepted for backend-protocol compatibility and
            otherwise unused (nothing is printed here). Default is ``False``.

    Returns:
        dict[str, np.ndarray]: The planar-averaged potentials, keyed by axis.
    """
    return get_gpaw_planar_averaged_potential(path)


def get_site_potentials(
    path: PathLike,
    dir_type: str = "bulk",
    quiet: bool = False,
    outputs: CalculationOutputs | None = None,
    total_energy: list | float | None = None,
) -> np.ndarray:
    """
    Get the atomic-site electrostatic potentials from the ``GPAW`` calculation
    in ``path``, for the Kumagai (eFNV) charge correction.

    Note that :func:`get_calculation_outputs` already parses these, so
    ``doped`` only calls this when they were not parsed up-front.

    Args:
        path (PathLike): Path to the calculation directory or ``.gpw`` file.
        dir_type (str): The type of directory being parsed (``"bulk"`` or
            ``"defect"``), for informative errors. Default is ``"bulk"``.
        quiet (bool): Accepted for backend-protocol compatibility and
            otherwise unused. Default is ``False``.
        outputs (CalculationOutputs): Already-parsed outputs, whose site
            potentials are used if present. Default is ``None``.
        total_energy (list | float): Accepted for backend-protocol
            compatibility and otherwise **unused**; ``GPAW`` writes the
            potentials and the total energy to the same ``.gpw`` file, so
            there is no separate calculation to cross-check against (unlike
            ``VASP``'s ``OUTCAR``). Default is ``None``.

    Returns:
        np.ndarray: The atomic-site potentials (eV), one per site.
    """
    if outputs is not None and outputs.site_potentials is not None:
        return np.asarray(outputs.site_potentials)
    return get_gpaw_site_potentials(path)
