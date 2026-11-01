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

See the GPAW tracking issue.
"""

import contextlib
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core.entries import ComputedStructureEntry
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.util.typing import PathLike

from doped.io import utils as _io_utils
from doped.io.outputs import CalculationOutputs

# ``doped`` accesses the remaining unimplemented protocol name directly (rather than probing with
# ``getattr(backend, name, default)``), so it is intercepted here to fail informatively rather than with an
# obscure ``AttributeError``. The optional, ``getattr``-probed protocol names (``SUBFOLDER_PRIORITY``,
# ``load_eigenvalue_outputs``, ``PLANAR_POTENTIALS_FILE``, ``SITE_POTENTIALS_FILE``) must keep raising
# ``AttributeError``, so that those features degrade gracefully as intended:
_UNIMPLEMENTED_BACKEND_ATTRS = ("get_fermi_dos",)


def __getattr__(name: str) -> Any:
    """
    Raise an informative error for the ``doped.io`` backend protocol
    functions/constants which are not implemented for GPAW (see the module
    docstring).
    """
    if name in _UNIMPLEMENTED_BACKEND_ATTRS:
        raise NotImplementedError(
            f"`{__name__}.{name}` is not implemented; GPAW support in `doped` is experimental, and "
            f"does not yet cover bulk DOS parsing. For a bulk DOS, build a `pymatgen` `FermiDos` "
            f"object yourself and pass it to `DefectThermodynamics` directly. See the GPAW tracking issue."
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


@contextlib.contextmanager
def _open_gpaw_restart(gpw_file: PathLike, **kwargs):
    """
    Load a ``GPAW`` calculator from the ``gpw_file`` restart file, as a context
    manager which closes it and detaches it from its ``Atoms`` on exit.

    Raises a ``ValueError`` if the file holds no stored results (e.g. from an
    unconverged calculation).

    Args:
        gpw_file (PathLike):
            Path to the ``GPAW`` ``.gpw`` restart file.
        **kwargs:
            Passed to ``GPAW()`` (e.g. ``txt=None``).

    Yields:
        GPAW:
            The ``GPAW`` calculator.
    """
    from gpaw import GPAW

    calc = GPAW(gpw_file, **kwargs)
    try:
        if calc.calculation_required(calc.atoms, ["energy"]):
            raise ValueError(
                f"{gpw_file} holds no stored results (e.g. from an unconverged or unfinished calculation)"
            )
        yield calc
    finally:
        if hasattr(calc, "close"):  # removed in GPAW >= 26
            calc.close()
        if getattr(calc, "atoms", None) is not None:  # break the ``Atoms`` <-> calculator reference
            calc.atoms.calc = None


def _get_gpaw_parameters(calc) -> dict:
    """
    Get the parameters of a ``GPAW`` calculator as a ``dict``.

    With GPAW >= 26, these are only the explicitly-set values, while with older
    versions they also include the defaults.
    """
    return calc.parameters.todict() if hasattr(calc.parameters, "todict") else dict(calc.parameters)


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
        calc (Calculator):
            ``GPAW`` calculator object, or ``ase`` single-point calculator with
            the results read from a ``.gpw`` file.

    Returns:
        Structure: The calculation structure.
    """
    atoms = calc.get_atoms().copy()  # ``copy()`` drops the attached calculator
    atoms.calc = None
    structure = AseAtomsAdaptor.get_structure(atoms)
    with contextlib.suppress(Exception):  # not available for non-spin-polarised calculations
        structure.add_site_property("final_magmom", list(calc.get_magnetic_moments()))

    return structure


def _get_eigenvalue_properties_from_calc(
    calc, occupation_tolerance: float = 0.1
) -> tuple[float, float, float, float]:
    """
    Get ``(band_gap, cbm, vbm, efermi)`` (eV) using band occupations.

    GPAW's raw occupations exclude spin degeneracy and k-point weights, so
    fully occupied states have occupation one in either spin channel. States
    between ``occupation_tolerance`` and ``1 - occupation_tolerance`` are
    partially occupied and give a zero gap, as do bands whose occupation
    changes across k-points or overlapping occupied and empty states. For
    these metallic cases both edges are set to the Fermi level. Missing
    occupied or empty states raise an informative error rather than
    substituting the Fermi level for an uncomputed band edge.

    These are the edges of the supplied calculation, not host band edges
    inferred from defect states. Charged supercell outputs omit band edges.
    """
    if not 0 < occupation_tolerance < 0.5:
        raise ValueError("occupation_tolerance must be between zero and 0.5")
    efermi = calc.get_fermi_level()
    occupied: list[float] = []
    empty: list[float] = []
    partial = False
    crosses_fermi_surface = False
    for spin in range(calc.get_number_of_spins()):
        filled_counts = []
        for kpt in range(len(calc.get_ibz_k_points())):
            energies = np.asarray(calc.get_eigenvalues(kpt=kpt, spin=spin))
            occupations = np.asarray(calc.get_occupation_numbers(kpt=kpt, spin=spin, raw=True))
            filled = occupations >= 1 - occupation_tolerance
            vacant = occupations <= occupation_tolerance
            filled_counts.append(np.count_nonzero(filled))
            partial |= bool(np.any(~(filled | vacant)))
            occupied.extend(energies[filled])
            empty.extend(energies[vacant])
        crosses_fermi_surface |= bool(np.ptp(filled_counts))

    if partial or crosses_fermi_surface:
        return 0.0, efermi, efermi, efermi
    if not occupied or not empty:
        raise ValueError(
            "Cannot determine GPAW band edges: no fully occupied or empty states were found. "
            "Include both occupied and unoccupied bands (increase nbands if necessary)."
        )
    vbm, cbm = max(occupied), min(empty)
    if cbm <= vbm:
        return 0.0, efermi, efermi, efermi
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
    with _open_gpaw_restart(_find_gpaw_output(gpw_file)) as calc:
        return _get_site_potentials_from_calc(calc)


def get_gpaw_planar_averaged_potential(
    gpw_file: str | os.PathLike,
) -> dict[str, np.ndarray]:
    """
    Extracts planar-averaged potential from a ``GPAW`` ``.gpw`` file.
    """
    with _open_gpaw_restart(_find_gpaw_output(gpw_file)) as calc:
        return _get_planar_averaged_potential_from_calc(calc)


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
            Accepted for backend-protocol compatibility but **unused**;
            ``GPAW`` writes the potentials and total energy to the same
            ``.gpw`` file, so there is no separate calculation to cross-check.
        run_metadata (dict):
            Accepted for backend-protocol compatibility but unused.

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


find_calc_output = _find_gpaw_output  # used by CompetingPhasesAnalyzer for file selection
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
    gpw_file = _find_gpaw_output(path, subfolder)
    with _open_gpaw_restart(gpw_file, txt=None) as calc:
        parameters = _get_gpaw_parameters(calc)
        charge = parameters.get("charge") or 0  # parameters only holds explicitly-set, non-default values
        if charge:
            band_gap = cbm = vbm = None
            efermi = calc.get_fermi_level()
        else:
            band_gap, cbm, vbm, efermi = _get_eigenvalue_properties_from_calc(calc)
        eigenvalues = {}
        for spin_index, spin in enumerate((Spin.up, Spin.down)[: calc.get_number_of_spins()]):
            eigenvalues[spin] = np.array(
                [  # (n_kpoints, n_bands, 2); energy and occupancy, as ``pymatgen`` expects
                    np.stack(
                        [
                            calc.get_eigenvalues(kpt=kpt, spin=spin_index),
                            # ``raw`` = in [0, 1], without k-point weights or spin degeneracy:
                            calc.get_occupation_numbers(kpt=kpt, spin=spin_index, raw=True),
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
            converged_electronic=True,  # (as results-less restarts are rejected by _open_gpaw_restart)
            # a ``.gpw`` restart file holds the final state only, with no ionic-step history to check:
            converged_ionic=None,
            efermi=efermi,
            eigenvalues=eigenvalues,
            kpoint_coords=calc.get_ibz_k_points(),
            kpoint_weights=calc.get_k_point_weights(),
            nelect=calc.get_number_of_electrons(),
            charge=charge,
            magnetization=calc.get_magnetic_moment(),
            # Defect states cannot establish the host edges of a charged supercell.
            vbm=vbm,
            cbm=cbm,
            band_gap=band_gap,
            planar_averaged_potentials={
                int(axis): potentials
                for axis, potentials in _get_planar_averaged_potential_from_calc(calc).items()
            },
            site_potentials=_get_site_potentials_from_calc(calc),
            run_metadata={"gpaw_parameters": parameters},
        )


def get_competing_phase_entry(path: PathLike) -> ComputedStructureEntry:
    """
    Parse a competing-phase calculation for ``CompetingPhasesAnalyzer``,
    reading the stored structure, energy and calculation parameters, and
    creating a |ComputedStructureEntry|.

    Args:
        path (PathLike):
            Calculation directory or a GPAW ``.gpw`` restart file.

    Returns:
        ComputedStructureEntry:
            The uncorrected total energy and structure, with GPAW parameters
            (as set for the calculation), formula, volume, atom count, source
            folder, k-point mesh and magnetization in ``entry.data``. Ionic
            convergence is not reported: restart files have no ionic-step
            history. Band gaps are omitted pending occupation-based band edges
            (TODO).
    """
    from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
    from ase.io.gpw import read_gpw
    from ase.io.ulm import open as ulm_open

    gpw_file = _find_gpaw_output(path)
    with ulm_open(gpw_file) as reader:  # first, so that non-ULM (e.g. empty) files give a clear error
        parameters = reader.parameters.asdict()
    atoms = read_gpw(gpw_file)  # with a ``SinglePointDFTCalculator`` holding the stored results
    if "energy" not in atoms.calc.results:
        raise ValueError(
            f"{gpw_file} holds no stored results (e.g. from an unconverged or unfinished calculation)"
        )

    structure = _structure_from_calc(atoms.calc)
    entry = ComputedStructureEntry(structure, atoms.get_potential_energy())  # parameters in ``entry.data``
    elements = sorted(el.symbol for el in entry.composition.elements)
    entry.data.update(
        {
            "calculator": "gpaw",
            "gpaw_parameters": parameters,
            "formula_pretty": entry.composition.reduced_formula,
            "nsites": len(structure),
            "volume": structure.volume,
            "energy_per_atom": entry.energy_per_atom,
            "elements": elements,
            "nelements": len(elements),
            "summary": {"total_magnetization": atoms.calc.get_magnetic_moment()},
            "folder": str(Path(gpw_file).parent),
            "converged_electronic": True,  # (as results-less restarts are rejected above)
        }
    )
    with contextlib.suppress(ValueError):  # non-Monkhorst-Pack sampling has no mesh dimensions
        entry.data["kpoints"] = [
            get_monkhorst_pack_size_and_offset(atoms.calc.get_bz_k_points())[0].tolist()
        ]
    return entry


_COMPATIBILITY_PARAMETERS = (
    "mode",  # PW/LCAO/FD and the plane-wave cutoff
    "xc",
    "kpts",
    "setups",  # PAW datasets
    "basis",  # LCAO basis set
    "h",  # real-space grid spacing...
    "gpts",  # ...or grid points
    "external",  # external potential (e.g. an electric field)
    "background_charge",  # explicit jellium (not standard uniform background for charged supercells)
)
"""
The ``GPAW`` calculation parameters which must match between calculations for
their energies to be comparable; functional, basis/grid, k-points, PAW datasets
and external fields (with k-point and real-space meshes only compared for
defect & bulk supercells, not competing phases).
"""


def _comparable(value: Any) -> Any:
    """
    Convert a ``GPAW`` parameter value to a form which compares by value.

    ``GPAW`` parameters can hold ``numpy`` arrays (e.g. an explicit k-point
    list), for which ``!=`` gives an array rather than a ``bool``, and
    equivalent lists/tuples; all are converted to (nested) lists.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _comparable(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_comparable(val) for val in value]
    return value


MISMATCH_WARNING_SPECS = {
    "mismatching_gpaw_parameters": {
        "object_name": "GPAW parameters",
        "data_key": "gpaw_parameters",
        "per_defect_warning_prefix": "There are mismatching GPAW parameters",
        "format": "{{parameter: ({name_1} value, {name_2} value)}}):",
        "closing": (
            "In general, the same GPAW settings should be used for all final calculations for accurate "
            "results!"
        ),
    },
}
"""
Specifications for collectively warning about mismatching ``GPAW`` calculation
parameters (see :func:`compare_calculation_settings`), for defect & bulk
supercell calculations and competing phases.
"""


def compare_calculation_settings(
    settings: dict,
    reference_settings: dict,
    competing_phases: bool = False,
    warn: bool = False,
) -> dict[str, Any]:
    """
    Compare the energy-affecting ``GPAW`` parameters (``"gpaw_parameters"``) of
    a calculation to those of a reference calculation.

    The parameters in :data:`_COMPATIBILITY_PARAMETERS` are compared, except
    for the k-point and real-space meshes (``kpts``, ``h``, ``gpts``) of
    competing phases, which can differ between phases.

    Used for both defect & bulk supercell calculations (by
    :func:`check_run_compatibility`, with ``CalculationOutputs.run_metadata``
    settings) and competing phases (by |CompetingPhasesAnalyzer| when
    ``check_compatibility=True`` (default), with ``entry.data`` settings).

    Args:
        settings (dict):
            The calculation settings, with the ``"gpaw_parameters"`` dict.
        reference_settings (dict):
            The settings of the reference calculation (e.g. the bulk
            supercell).
        competing_phases (bool):
            Whether these are competing phase calculations (for which meshes
            are not compared). Default is ``False``.
        warn (bool):
            Whether to warn about any mismatches found. Default is ``False``.

    Returns:
        dict:
            ``"mismatching_gpaw_parameters"``: a
            ``{parameter: (value, reference value)}`` dict, or ``False`` if
            they match.
    """
    parameters, reference = (s.get("gpaw_parameters") or {} for s in [settings, reference_settings])
    mismatches = {
        parameter: (parameters.get(parameter), reference.get(parameter))
        for parameter in _COMPATIBILITY_PARAMETERS
        if not (competing_phases and parameter in {"kpts", "h", "gpts"})
        and _comparable(parameters.get(parameter)) != _comparable(reference.get(parameter))
    }
    if mismatches and warn:
        labels = _io_utils._MISMATCH_WARNING_CONTEXTS[competing_phases]  # e.g. "defect" & "bulk"
        mismatch_info = "\n".join(
            f"{parameter}: {value!r} ({labels['name_1']}) vs {reference_value!r} ({labels['name_2']})"
            for parameter, (value, reference_value) in mismatches.items()
        )
        warnings.warn(
            f"There are mismatching GPAW parameters between your {labels['name_2']} and "
            f"{labels['name_1']} calculations, which may mean your energies are not comparable:\n"
            f"{mismatch_info}"
        )

    return {"mismatching_gpaw_parameters": mismatches or False}


def check_run_compatibility(
    defect_outputs: CalculationOutputs,
    bulk_outputs: CalculationOutputs,
    warn: bool = True,
) -> dict:
    """
    Check that the defect and bulk ``GPAW`` calculations used compatible
    settings (with :func:`compare_calculation_settings`), and collect their
    parameters for ``DefectEntry.calculation_metadata``.

    ``GPAW``'s full parameter dictionaries are returned either way.

    Args:
        defect_outputs (CalculationOutputs):
            The parsed defect supercell outputs.
        bulk_outputs (CalculationOutputs):
            The parsed bulk supercell outputs.
        warn (bool):
            Whether to warn about mismatched parameters. Default is ``True``.

    Returns:
        dict:
            ``"run_metadata"`` (both parameter sets) and
            ``"mismatching_gpaw_parameters"`` (a
            ``{parameter: (defect, bulk)}`` dict, or ``False`` if they match).
    """
    run_metadata = {
        f"{label}_gpaw_parameters": (outputs.run_metadata or {}).get("gpaw_parameters", {})
        for label, outputs in (("defect", defect_outputs), ("bulk", bulk_outputs))
    }
    return {
        **compare_calculation_settings(
            defect_outputs.run_metadata or {}, bulk_outputs.run_metadata or {}, warn=warn
        ),
        "run_metadata": run_metadata,
    }


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
        dir_type (str):
            The type of directory being parsed (``"bulk"`` or ``"defect"``),
            for informative errors. Default is ``"bulk"``.
        quiet (bool):
            Accepted for backend-protocol compatibility but unused.
        outputs (CalculationOutputs):
            Already-parsed outputs, whose site potentials are used if present.
            Default is ``None``.
        total_energy (list | float):
            Accepted for backend-protocol compatibility but **unused**;
            ``GPAW`` writes the potentials and total energy to the same
            ``.gpw`` file, so there is no separate calculation to cross-check.

    Returns:
        np.ndarray: The atomic-site potentials (eV), one per site.
    """
    if outputs is not None and outputs.site_potentials is not None:
        return np.asarray(outputs.site_potentials)
    return get_gpaw_site_potentials(path)
