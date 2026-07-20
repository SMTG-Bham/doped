"""
Parsing of VASP defect / bulk supercell calculation outputs.

These functions load and process VASP output files (``vasprun.xml(.gz)``,
``OUTCAR``, ``LOCPOT``, ``PROCAR``), and can provide the parsed outputs in
calculator-agnostic form (:class:`~doped.io.outputs.CalculationOutputs`) via
:func:`get_calculation_outputs`.
"""

import contextlib
import itertools
import re
import warnings
from collections.abc import Iterable
from functools import lru_cache, partialmethod
from pathlib import Path
from xml.etree.ElementTree import Element as XML_Element

import numpy as np
import pandas as pd
from monty.io import reverse_readfile
from monty.serialization import loadfn
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.inputs import POTCAR_STATS_PATH, UnknownPotcarWarning
from pymatgen.io.vasp.outputs import Locpot, Outcar, Procar, Vasprun, _parse_vasp_array
from pymatgen.util.typing import PathLike

from doped.io import utils as _io_utils
from doped.io.outputs import CalculationOutputs
from doped.io.utils import _dataframe_of_files, find_archived_fname  # noqa: F401 (re-exported)


@lru_cache(maxsize=1000)  # cache POTCAR generation to speed up generation and writing
def _get_potcar_summary_stats() -> dict:
    return loadfn(POTCAR_STATS_PATH)


# has to be defined as staticmethod to be consistent with usage in pymatgen, alternatively could make
# fake custom class:
@staticmethod  # type: ignore[misc]
def parse_projected_eigen(
    elem: XML_Element, parse_mag: bool = True
) -> tuple[dict[Spin, np.ndarray], np.ndarray | None]:
    """
    Parse the projected eigenvalues from a |Vasprun| object (used during
    initialisation), but excluding the projected magnetization for efficiency.

    Note that following SK's PRs to ``pymatgen`` (#4359, #4360), parsing of
    projected eigenvalues adds minimal additional cost to |Vasprun| parsing
    (~1-5%), while parsing of projected magnetization can add ~30% cost.

    This is a modified version of ``_parse_projected_eigen`` from
    |Vasprun|, which allows skipping of projected magnetization parsing in
    order to expedite parsing in ``doped``, as well as some small adjustments
    to maximise efficiency.

    Args:
        elem (Element):
            The XML element to parse, with projected eigenvalues/magnetization.
        parse_mag (bool):
            Whether to parse the projected magnetization. Default is ``True``.

    Returns:
        tuple[dict[Spin, np.ndarray], np.ndarray | None]:
            A dictionary of projected eigenvalues for each spin channel
            (up/down), and the projected magnetization (if parsed).
    """
    root = elem.find("array/set")
    assert root is not None  # projected eigenvalue array always present when this is called
    proj_eigen = {}
    sets = root.findall("set")

    for s in sets:
        spin_match = re.match(r"spin(\d+)", s.attrib["comment"])
        assert spin_match is not None
        spin = int(spin_match[1])
        if spin == 1 or (spin == 2 and len(sets) == 2):
            spin_key = Spin.up if spin == 1 else Spin.down
        elif parse_mag:  # parse projected magnetization
            spin_key = spin  # {2:"x", 3:"y", 4:"z"}
        else:
            continue

        proj_eigen[spin_key] = np.array(
            [[_parse_vasp_array(sss) for sss in ss.findall("set")] for ss in s.findall("set")]
        )

    if len(proj_eigen) > 2:
        # non-collinear magnetism (spin-orbit coupling) enabled, last three "spin channels" are the
        # projected magnetization of the orbitals in the x, y, and z Cartesian coordinates:
        proj_mag = np.stack([proj_eigen.pop(i) for i in range(2, 5)], axis=-1)
        proj_eigen = {Spin.up: proj_eigen[Spin.up]}
    else:
        proj_mag = None

    # here we _could_ round to 3 decimal places (and ensure rounding 0.0005 up to 0.001) to be _mostly_
    # consistent with PROCAR values (still not 100% the same as e.g. 0.00047 will be rounded to 0.0005
    # in vasprun, but 0.000 in PROCAR), but this is _reducing_ the accuracy so better not to do this,
    # and accept that PROCAR results may not be as numerically robust
    # proj_eigen = {k: np.round(v+0.00001, 3) for k, v in proj_eigen.items()}
    elem.clear()
    return proj_eigen, proj_mag


def get_vasprun(vasprun_path: PathLike, parse_mag: bool = True, **kwargs):
    """
    Read the ``vasprun.xml(.gz)`` file as a ``pymatgen`` |Vasprun| object.
    """
    vasprun_path = str(vasprun_path)  # convert to string if Path object
    warnings.filterwarnings(
        "ignore", category=UnknownPotcarWarning
    )  # Ignore unknown POTCAR warnings when loading vasprun.xml
    # pymatgen assumes the default PBE with no way of changing this within get_vasprun())
    warnings.filterwarnings(
        "ignore", message="No POTCAR file with matching TITEL fields"
    )  # `message` only needs to match start of message
    default_kwargs = {"parse_dos": False, "exception_on_bad_xml": False}
    default_kwargs.update(kwargs)

    Vasprun._parse_projected_eigen = partialmethod(parse_projected_eigen, parse_mag=parse_mag)
    try:
        with warnings.catch_warnings(record=True) as w:
            vasprun = Vasprun(find_archived_fname(vasprun_path), **default_kwargs)
        for warning in w:
            if "XML is malformed" in str(warning.message):
                warnings.warn(
                    f"vasprun.xml file at {vasprun_path} is corrupted/incomplete. Attempting to "
                    f"continue parsing but may fail!"
                )
            else:  # show warning, preserving original category:
                warnings.warn(warning.message, category=warning.category)

    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"vasprun.xml not found at {vasprun_path}(.gz/.xz/.bz/.lzma). Needed for parsing calculation "
            f"output!"
        ) from exc
    return vasprun


def get_locpot(locpot_path: PathLike):
    """
    Read the ``LOCPOT(.gz)`` file as a ``pymatgen`` ``Locpot`` object.
    """
    locpot_path = str(locpot_path)  # convert to string if Path object
    try:
        locpot = Locpot.from_file(find_archived_fname(locpot_path))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"LOCPOT file not found at {locpot_path}(.gz/.xz/.bz/.lzma). Needed for calculating the "
            f"Freysoldt (FNV) image charge correction!"
        ) from None
    return locpot


def _get_outcar_path(outcar_path: PathLike, raise_error=True):
    outcar_path = str(outcar_path)  # convert to string if Path object
    try:
        return find_archived_fname(outcar_path)
    except FileNotFoundError:
        if raise_error:
            raise FileNotFoundError(
                f"OUTCAR file not found at {outcar_path}(.gz/.xz/.bz/.lzma). Needed for calculating the "
                f"Kumagai (eFNV) image charge correction."
            ) from None


def get_outcar(outcar_path: PathLike):
    """
    Read the ``OUTCAR(.gz)`` file as a ``pymatgen`` |Outcar| object.
    """
    outcar_path = _get_outcar_path(outcar_path)
    return Outcar(outcar_path)


def get_core_potentials_from_outcar(
    outcar_path: PathLike, dir_type: str = "", total_energy: list | float | None = None
):
    """
    Get the core potentials from the ``OUTCAR`` file, which are needed for the
    Kumagai-Oba (eFNV) finite-size correction.

    This parser skips the full ``pymatgen`` |Outcar| initialisation/parsing,
    to expedite parsing and make it more robust (doesn't fail if ``OUTCAR`` is
    incomplete, as long as it has the core potentials information).

    Args:
        outcar_path (PathLike):
            The path to the ``OUTCAR`` file.
        dir_type (str):
            The type of directory the ``OUTCAR`` is in (e.g. ``bulk`` or
            ``defect``) for informative error messages.
        total_energy (list | float | None):
            The already-parsed total energy for the structure. If provided,
            will check that the total energy of the ``OUTCAR`` matches this
            value / one of these values, and throw a warning if not.

    Returns:
        np.ndarray:
            The core potentials from the last ionic step in the ``OUTCAR``.
    """
    # initialise Outcar class without running __init__ method:
    outcar = Outcar.__new__(Outcar)
    outcar.filename = _get_outcar_path(outcar_path)
    core_pots_list = outcar.read_avg_core_poten()
    if not core_pots_list:
        _raise_incomplete_outcar_error(outcar_path, dir_type=dir_type)

    _check_outcar_energy(outcar_path, total_energy=total_energy)

    return -1 * np.array(core_pots_list[-1])  # core potentials from last step


def _get_final_energy_from_outcar(outcar_path):
    """
    Get the final total energy from an ``OUTCAR`` file, even if the calculation
    was not completed.

    Templated on the ``OUTCAR`` parsing code from ``pymatgen``, but works even
    if the ``OUTCAR`` is incomplete.
    """
    e0_pattern = re.compile(r"energy\(sigma->0\)\s*=\s+([\d\-\.]+)")
    e0 = None
    for line in reverse_readfile(outcar_path):
        clean = line.strip()
        if e0 is None and (match := e0_pattern.search(clean)):
            e0 = float(match[1])

    return e0


def _get_core_potentials_from_outcar_obj(
    outcar: Outcar, dir_type: str = "", total_energy: list | float | None = None
):
    if outcar.electrostatic_potential is None and not outcar.read_avg_core_poten():
        _raise_incomplete_outcar_error(outcar, dir_type=dir_type)
    _check_outcar_energy(outcar, total_energy=total_energy)

    return -1 * np.array(outcar.electrostatic_potential) or -1 * np.array(outcar.read_avg_core_poten()[-1])


def _check_outcar_energy(outcar: Outcar | PathLike, total_energy: list | float | None = None):
    if total_energy:
        outcar_energy = (
            outcar.final_energy if isinstance(outcar, Outcar) else _get_final_energy_from_outcar(outcar)
        )
        total_energy = total_energy if isinstance(total_energy, list) else [total_energy]
        total_energies = set(np.round(total_energy, 3))
        formatted_total_energy = "eV, ".join(f"{energy:.3f}" for energy in total_energies) + " eV"
        if len(total_energies) == 2:  # most cases, final energy and last electronic step energy
            formatted_total_energy += "; final energy & last electronic step energy"
        if not any(np.isclose(outcar_energy, energy, atol=0.025) for energy in total_energy):
            # 0.025 eV tolerance
            warnings.warn(
                f"The total energies of the provided (bulk) `OUTCAR` ({outcar_energy:.3f} eV), "
                f"used to obtain the atomic core potentials for the eFNV correction, and the "
                f"`vasprun.xml` ({formatted_total_energy}), used for energies and structures, do not "
                f"match. Please make sure the correct file combination is being used!"
            )


def _raise_incomplete_outcar_error(outcar: PathLike | Outcar, dir_type: str = ""):
    """
    Raise error about supplied ``OUTCAR`` not having atomic core potential
    info.

    Input outcar is either a path or a ``pymatgen`` |Outcar| object
    """
    outcar_info = f"`OUTCAR` at {outcar}" if isinstance(outcar, PathLike) else "`OUTCAR` object"
    dir_type = f"{dir_type} " if dir_type else ""
    raise ValueError(
        f"Unable to parse atomic core potentials from {dir_type}{outcar_info}. This can happen if "
        f"`ICORELEVEL` was not set to 0 (= default) in the `INCAR`, the calculation was finished "
        f"prematurely with a `STOPCAR`, or the calculation crashed. The Kumagai (eFNV) charge correction "
        f"cannot be computed without this data!"
    )


def get_procar(procar_path: PathLike) -> Procar:
    """
    Read the ``PROCAR(.gz)`` file as a ``pymatgen`` |Procar| object.

    Previously, ``pymatgen`` |Procar| parsing did not support SOC calculations,
    however this was updated in
    https://github.com/materialsproject/pymatgen/pull/3890 to use code from
    ``easyunfold`` (https://smtg-bham.github.io/easyunfold -- a package for
    unfolding electronic band structures for symmetry-broken / defect /
    dopant systems, with many plotting & analysis tools).
    """
    try:
        procar_path = find_archived_fname(str(procar_path))  # convert to string if Path object
    except FileNotFoundError:
        raise FileNotFoundError(f"PROCAR file not found at {procar_path}(.gz/.xz/.bz/.lzma)!") from None

    return Procar(procar_path)


def _parse_procar(procar: PathLike | Procar | None = None):
    """
    Parse the input path or ``pymatgen`` |Procar| to a |Procar| object in the
    correct format, for eigenvalue analysis.

    Args:
        procar (PathLike, |Procar|):
            Either a path to the ``VASP`` ``PROCAR``` output file (with
            ``LORBIT > 10`` in the ``INCAR``) or a``pymatgen`` |Procar|.

    Returns:
        Procar: The parsed |Procar| object in ``pymatgen`` format.
    """
    from pymatgen.electronic_structure.core import Spin

    if not hasattr(procar, "data"):  # not a parsed Procar object
        if procar and hasattr(procar, "proj_data") and not isinstance(procar, PathLike | Procar):
            if procar._is_soc:
                procar.data = {Spin.up: procar.proj_data[0]}
            else:
                procar.data = {Spin.up: procar.proj_data[0], Spin.down: procar.proj_data[1]}
            del procar.proj_data

        elif isinstance(procar, PathLike):  # path to PROCAR file
            procar = get_procar(procar)

    return procar


def _get_output_files_and_check_if_multiple(
    output_file: PathLike = "vasprun.xml", path: PathLike = "."
) -> tuple[PathLike, bool]:
    """
    Search for all files with filenames matching ``output_file``, case-
    insensitive; VASP-defaulted wrapper of
    :func:`doped.io.utils._get_output_files_and_check_if_multiple`.

    Args:
        output_file (PathLike):
            The filename to search for (case-insensitive). Should be either
            ``vasprun.xml``, ``OUTCAR``, ``LOCPOT`` or ``PROCAR``.
        path (PathLike):
            The path to the directory to search in.

    Returns:
        Tuple[PathLike, bool]:
            The path to the identified file, and a boolean indicating whether
            multiple files were found.
    """
    search_patterns = ["vasprun", ".xml"] if str(output_file).lower() == "vasprun.xml" else None
    return _io_utils._get_output_files_and_check_if_multiple(output_file, path, search_patterns)


_VASP_CALC_OUTPUT_MASK = ("vasprun.xml", "vasprun.xml.gz")
"""
Filename patterns that identify calculation output files.
"""

_VASP_SUBFOLDER_PRIORITY = [
    "vasp_ncl",
    "singlepoint",
    "final",
    "relax",
    "vasp_std",
    "vasp_nkred_std",
    "vasp_gam",
]
"""
Priority order when auto-detecting calculation subfolders.
"""


def _get_calc_files_df(
    root: Path, calc_output_mask: Iterable[str] = _VASP_CALC_OUTPUT_MASK
) -> pd.DataFrame:
    """
    Get a DataFrame of calculation output files (matching ``calc_output_mask``,
    defaulting to VASP ``vasprun.xml(.gz)`` files) found recursively under
    ``root``; VASP-defaulted wrapper of
    :func:`doped.io.utils._get_calc_files_df`.
    """
    return _io_utils._get_calc_files_df(root, calc_output_mask)


def _determine_subfolder(
    files_df: pd.DataFrame,
    candidate_folders: list[str] | None = None,
    subfolder_priority: list[str] = _VASP_SUBFOLDER_PRIORITY,
) -> str:
    """
    Pick the highest-priority calculation subfolder name present in
    ``files_df`` (defaulting to the VASP subfolder priority order,
    ``_VASP_SUBFOLDER_PRIORITY``); VASP-defaulted wrapper of
    :func:`doped.io.utils._determine_subfolder`.
    """
    return _io_utils._determine_subfolder(
        files_df, candidate_folders, subfolder_priority=subfolder_priority
    )


def _find_calc_outputs(
    output_path: PathLike,
    subfolder: PathLike | None = None,
) -> tuple[pd.DataFrame, list[str], str]:
    """
    Recursively find VASP calculation output files (``vasprun.xml(.gz)``) under
    ``output_path`` and auto-detect the calculation subfolder (from
    ``_VASP_SUBFOLDER_PRIORITY``) when ``subfolder`` is ``None``; VASP-
    defaulted wrapper of :func:`doped.io.utils._find_calc_outputs`.

    Shared discovery logic used by both :func:`~doped.analysis.DefectsParser`
    and :meth:`~doped.chemical_potentials.CompetingPhasesAnalyzer`.
    """
    return _io_utils._find_calc_outputs(
        output_path,
        subfolder,
        calc_output_mask=_VASP_CALC_OUTPUT_MASK,
        subfolder_priority=_VASP_SUBFOLDER_PRIORITY,
    )


def _compare_potcar_symbols(
    defect_potcar_symbols,
    bulk_potcar_symbols,
    defect_name="defect",
    bulk_name="bulk",
    warn=True,
    only_matching_elements=False,
):
    """
    Check all POTCAR symbols in the bulk are the same in the defect
    calculation.

    Returns True if the symbols match, otherwise returns a list of the symbols
    for the bulk and defect calculations.
    """
    if only_matching_elements:
        defect_elements = [symbol["titel"].split()[1].split("_")[0] for symbol in defect_potcar_symbols]
        symbols_to_check = [
            symbol
            for symbol in bulk_potcar_symbols
            if symbol["titel"].split()[1].split("_")[0] in defect_elements
        ]
    else:
        symbols_to_check = bulk_potcar_symbols

    bulk_mismatch_list = []
    defect_mismatch_list = []
    for symbol in symbols_to_check:
        if symbol["titel"] not in [symbol["titel"] for symbol in defect_potcar_symbols]:
            if warn:
                warnings.warn(
                    f"The POTCAR symbols for your {defect_name} and {bulk_name} calculations do not "
                    f"match, which is likely to cause severe errors in the parsed results. Found the "
                    f"following symbol in the {bulk_name} calculation:"
                    f"\n{symbol['titel']}\n"
                    f"but not in the {defect_name} calculation:"
                    f"\n{[symbol['titel'] for symbol in defect_potcar_symbols]}\n"
                    f"The same POTCAR settings should be used for all calculations for accurate results!"
                )
            if not only_matching_elements:
                return [defect_potcar_symbols, bulk_potcar_symbols]
            bulk_mismatch_list.append(symbol)
            defect_mismatch_list.append(
                next(
                    def_symbol
                    for def_symbol in defect_potcar_symbols
                    if def_symbol["titel"].split()[1].split("_")[0]
                    == symbol["titel"].split()[1].split("_")[0]
                )
            )

    if bulk_mismatch_list:
        return [defect_mismatch_list, bulk_mismatch_list]

    return True


def _compare_kpoints(
    defect_actual_kpoints,
    bulk_actual_kpoints,
    defect_kpoints=None,
    bulk_kpoints=None,
    defect_name="defect",
    bulk_name="bulk",
    warn=True,
):
    """
    Check bulk and defect KPOINTS are the same, using the
    ``Vasprun.actual_kpoints`` lists (i.e. the VASP IBZKPTs essentially).

    Returns ``True`` if the KPOINTS match, otherwise returns a list of the
    KPOINTS for the bulk and defect calculations.
    """
    # sort kpoints, in case same KPOINTS just different ordering:
    sorted_bulk_kpoints = sorted(np.array(bulk_actual_kpoints), key=tuple)
    sorted_defect_kpoints = sorted(np.array(defect_actual_kpoints), key=tuple)

    actual_kpoints_eq = len(sorted_bulk_kpoints) == len(sorted_defect_kpoints) and np.allclose(
        sorted_bulk_kpoints, sorted_defect_kpoints
    )
    # if different symmetry settings used (e.g. for bulk), actual_kpoints can differ but are the same
    # input kpoints, which we assume is fine:
    kpoints_eq = (
        (
            bulk_kpoints.kpts == defect_kpoints.kpts
            and np.allclose(bulk_kpoints.kpts_shift, defect_kpoints.kpts_shift)
        )
        if bulk_kpoints and defect_kpoints
        else False
    )

    if not (actual_kpoints_eq or kpoints_eq):
        if warn:
            formatted_defect_kpts = [[float(kpt) for kpt in kpoints] for kpoints in sorted_defect_kpoints]
            formatted_bulk_kpts = [[float(kpt) for kpt in kpoints] for kpoints in sorted_bulk_kpoints]
            warnings.warn(  # list form is more readable
                f"The KPOINTS for your {defect_name} and {bulk_name} calculations do not match, which is "
                f"likely to cause errors in the parsed results. Found the following KPOINTS in the "
                f"{defect_name} calculation:"
                f"\n{formatted_defect_kpts}\n"
                f"and in the {bulk_name} calculation:"
                f"\n{formatted_bulk_kpts}\n"
                f"In general, the same KPOINTS settings should be used for all final calculations for "
                f"accurate results!"
            )
        return [
            [list(kpoints) for kpoints in sorted_defect_kpoints],
            [list(kpoints) for kpoints in sorted_bulk_kpoints],
        ]

    return True


def _compare_incar_tags(
    defect_incar_dict: dict[str, str | int | float],
    bulk_incar_dict: dict[str, str | int | float],
    fatal_incar_mismatch_tags: dict[str, str | int | float] | None = None,
    ignore_tags: set[str] | None = None,
    defect_name: str = "defect",
    bulk_name: str = "bulk",
    warn: bool = True,
):
    """
    Check bulk and defect INCAR tags (that can affect energies) are the same.

    Returns True if no mismatching tags are found, otherwise returns a list of
    the mismatching tags.
    """
    if fatal_incar_mismatch_tags is None:
        fatal_incar_mismatch_tags = {  # dict of tags that can affect energies and their defaults in VASP
            "AEXX": 0.25,  # default 0.25
            "ENCUT": 0,
            "LREAL": False,  # default False
            "HFSCREEN": 0,  # default 0 (None)
            "GGA": "PE",  # default PE
            "LHFCALC": False,  # default False
            "ADDGRID": False,  # default False
            "ISIF": 2,
            "LASPH": False,  # default False
            "PREC": "Normal",  # default Normal
            "PRECFOCK": "Normal",  # default Normal
            "LDAU": False,  # default False
            "NKRED": 1,  # default 1
            "LSORBIT": False,  # default False
        }
    if ignore_tags is not None:
        fatal_incar_mismatch_tags = {
            key: val for key, val in fatal_incar_mismatch_tags.items() if key not in ignore_tags
        }

    def _compare_incar_vals(val1, val2):
        if isinstance(val1, str):
            return val1.split()[0].lower() == str(val2).split()[0].lower()
        if isinstance(val1, int | float) and isinstance(val2, int | float):
            return np.isclose(val1, val2, rtol=1e-3)

        return val1 == val2

    mismatch_list = []
    for key, val in bulk_incar_dict.items():
        if key in fatal_incar_mismatch_tags:
            defect_val = defect_incar_dict.get(key, fatal_incar_mismatch_tags[key])
            if not _compare_incar_vals(val, defect_val):
                mismatch_list.append((key, defect_val, val))

    # get any missing keys:
    defect_incar_keys_not_in_bulk = set(defect_incar_dict.keys()) - set(bulk_incar_dict.keys())

    for key in defect_incar_keys_not_in_bulk:
        if key in fatal_incar_mismatch_tags and not _compare_incar_vals(
            defect_incar_dict[key], fatal_incar_mismatch_tags[key]
        ):
            mismatch_list.append((key, defect_incar_dict[key], fatal_incar_mismatch_tags[key]))

    if mismatch_list:
        if warn:
            warnings.warn(
                f"There are mismatching INCAR tags for your {defect_name} and {bulk_name} calculations "
                f"which are likely to cause errors in the parsed results (energies). Found the following "
                f"differences:\n"
                f"(in the format: (INCAR tag, value in {defect_name} calculation, value in {bulk_name} "
                f"calculation)):"
                f"\n{mismatch_list}\n"
                f"In general, the same INCAR settings should be used in all final calculations for these "
                f"tags which can affect energies!"
            )
        return mismatch_list
    return True


def _format_mismatching_incar_warning(mismatching_INCAR_warnings: list[tuple[str, set]]) -> str:
    """
    Convenience function to generate a formatted warning string listing
    mismatching INCAR tags and their values in a clean output.

    Used in ``doped.analysis`` and ``doped.chemical_potentials`` when checking
    calculation compatibilities.

    Args:
        mismatching_INCAR_warnings (list[tuple[str, set]]):
            A list of tuples containing the INCAR tag and the set of
            mismatching values for that tag.

    Returns:
        str:
            A formatted string listing the mismatching INCAR tags and their
            values.
    """
    # group by the mismatching tags, so we can print them together:
    mismatching_tags_name_list_dict = {
        tuple(sorted(mismatching_set)): sorted(
            [
                name
                for name, other_mismatching_set in mismatching_INCAR_warnings
                if other_mismatching_set == mismatching_set
            ]
        )  # sort for consistency
        for mismatching_set in [mismatching for name, mismatching in mismatching_INCAR_warnings]
    }
    return "\n".join(
        [
            f"{entry_list}:\n{list(mismatching)}"
            for mismatching, entry_list in mismatching_tags_name_list_dict.items()
        ]
    )


def get_magnetization_from_vasprun(vasprun: Vasprun) -> int | float | np.ndarray:
    """
    Determine the total magnetization from a |Vasprun| object.

    For spin-polarised calculations, this is the difference between the number
    of spin-up vs spin-down electrons. For non-spin-polarised calculations,
    there is no magnetization. For non-collinear (NCL) magnetization (e.g.
    spin-orbit coupling (SOC) calculations), the magnetization becomes a vector
    (spinor), in which case we take the vector norm as the total magnetization.

    VASP does not write the total magnetization to ``vasprun.xml`` file (but
    does to the ``OUTCAR`` file), and so here we have to reverse-engineer it
    from the eigenvalues (for normal spin-polarised calculations) or the
    projected magnetization & eigenvalues (for NCL calculations). For NCL
    calculations, we sum the projected orbital magnetizations for all occupied
    states, weighted by the `k`-point weights and normalised by the total
    orbital projections for each band and `k`-point. This gives the best
    estimate of the total magnetization from the projected magnetization array,
    but due to incomplete orbital projections and orbital-dependent non-uniform
    scaling factors (i.e. completeness of orbital projects for `s` vs `p` vs
    `d` orbitals etc.), there can be inaccuracies up to ~30% in the estimated
    total magnetization for tricky cases.

    Args:
        vasprun (|Vasprun|):
            The |Vasprun| object from which to extract the total
            magnetization.

    Returns:
        int or float or np.ndarray:
            The total magnetization of the system.
    """
    # in theory should be able to use vasprun.idos (integrated dos), but this doesn't show
    # spin-polarisation / account for NELECT changes from neutral apparently
    eigenvalues_and_occs = vasprun.eigenvalues
    kweights = np.array(vasprun.actual_kpoints_weights)

    # first check if it's a spin-polarised calculation:
    if len(eigenvalues_and_occs) == 1 or not vasprun.is_spin:
        # non-spin-polarised or NCL calculation:
        if not vasprun.parameters.get("LNONCOLLINEAR", False):
            return 0  # non-spin polarised calculation
        if getattr(vasprun, "projected_magnetization", None) is None:
            raise RuntimeError(
                "Cannot determine magnetization from non-collinear Vasprun calculation, as this requires "
                "the `Vasprun.projected_magnetization` attribute, which is parsed with "
                "`Vasprun(parse_projected_eigen=True)` (default in `doped`)."
            )

        # else NCL calculation:
        # need to scale by the summed orbital projections for each band (which should be 1):
        # vasprun.projected_eigenvalues[Spin.up].shape -> (nkpoints, nbands, natoms, norbitals)
        summed_orbital_projections = vasprun.projected_eigenvalues[Spin.up].sum(axis=(-2, -1))
        summed_orbital_projections = np.where(
            summed_orbital_projections == 0, 1, summed_orbital_projections
        )  # avoid division by zero, by setting any zero values to 1
        normalisation_factors = 1 / summed_orbital_projections

        # vasprun.projected_magnetization.shape -> (nkpoints, nbands, natoms, norbitals, 3 -- x/y/z)
        # sum the projected magnetization over atoms and orbitals, then multiply by per-band/kpoint
        # normalisation factors:
        normalised_proj_mag_per_kpoint_band_direction = (
            vasprun.projected_magnetization.sum(axis=(-3, -2)) * normalisation_factors[..., None]
        )  # [..., None] adds new axis, which allows broadcasting (i.e.
        # (nkpoints, nbands, 3) * (nkpoints, nbands, 1) -- adding the "(...,1 )" dimension)

        # then multiply by occupancies, sum over bands, multiply by k-point weights, sum over k-points:
        return (
            (
                normalised_proj_mag_per_kpoint_band_direction
                * eigenvalues_and_occs[Spin.up][:, :, 1][..., None]
            ).sum(axis=1)
            * kweights[..., None]
        ).sum(axis=0)

    # product of the sum of occupations over all bands, times the k-point weights:
    n_spin_up = np.sum(eigenvalues_and_occs[Spin.up][:, :, 1].sum(axis=1) * kweights)
    n_spin_down = np.sum(eigenvalues_and_occs[Spin.down][:, :, 1].sum(axis=1) * kweights)

    return n_spin_up - n_spin_down


def get_nelect_from_vasprun(vasprun: Vasprun) -> int | float:
    """
    Determine the number of electrons (``NELECT``) from a |Vasprun| object.

    Args:
        vasprun (|Vasprun|):
            The |Vasprun| object from which to extract ``NELECT``.

    Returns:
        int or float: The number of electrons in the system.
    """
    # can also obtain this (NELECT), charge and magnetization from Outcar objects, worth keeping in mind
    # but not needed atm
    # in theory should be able to use vasprun.idos (integrated dos), but this doesn't show
    # spin-polarisation / account for NELECT changes from neutral apparently

    eigenvalues_and_occs = vasprun.eigenvalues
    kweights = vasprun.actual_kpoints_weights

    # product of the sum of occupations over all bands, times the k-point weights:
    nelect = np.sum(eigenvalues_and_occs[Spin.up][:, :, 1].sum(axis=1) * kweights)
    if len(eigenvalues_and_occs) > 1:
        nelect += np.sum(eigenvalues_and_occs[Spin.down][:, :, 1].sum(axis=1) * kweights)
    elif not vasprun.parameters.get("LNONCOLLINEAR", False):
        nelect *= 2  # non-spin-polarised or SOC calc

    return round(nelect, 2)


def get_neutral_nelect_from_vasprun(vasprun: Vasprun, skip_potcar_init: bool = False) -> int:
    """
    Determine the number of electrons (``NELECT``) from a |Vasprun| object,
    corresponding to a neutral charge state for the structure.

    Args:
        vasprun (|Vasprun|):
            The |Vasprun| object from which to extract ``NELECT``.
        skip_potcar_init (bool):
            Whether to skip the initialisation of the ``POTCAR`` statistics
            (i.e. the auto-charge determination) and instead try to reverse
            engineer ``NELECT`` using the ``DefectDictSet``.

    Returns:
        int:
            The number of electrons in the system for a neutral charge state.
    """
    nelect = None
    if not skip_potcar_init:
        with contextlib.suppress(Exception):  # try determine charge without POTCARs first:
            grouped_symbols = [list(group) for key, group in itertools.groupby(vasprun.atomic_symbols)]
            potcar_summary_stats = _get_potcar_summary_stats()

            for trial_functional in ["PBE_64", "PBE_54", "PBE_52", "PBE", potcar_summary_stats.keys()]:
                if all(
                    potcar_summary_stats[trial_functional].get(
                        vasprun.potcar_spec[i]["titel"].replace(" ", ""), False
                    )
                    for i in range(len(grouped_symbols))
                ):
                    break

            nelect = sum(  # this is always the NELECT for the bulk
                np.array([len(i) for i in grouped_symbols])
                * np.array(
                    [
                        potcar_summary_stats[trial_functional][
                            vasprun.potcar_spec[i]["titel"].replace(" ", "")
                        ][0]["ZVAL"]
                        for i in range(len(grouped_symbols))
                    ]
                )
            )

    if nelect is not None:
        return int(nelect)

    # else try reverse engineer NELECT using DefectDictSet
    from doped.io.vasp.inputs import DefectDictSet

    potcar_symbols = [titel.split()[1] for titel in vasprun.potcar_symbols]
    potcar_settings = {symbol.split("_")[0]: symbol for symbol in potcar_symbols}
    with warnings.catch_warnings():  # ignore POTCAR warnings if not available
        warnings.simplefilter("ignore", UserWarning)
        return int(
            DefectDictSet(
                vasprun.structures[-1],
                charge_state=0,
                user_potcar_settings=potcar_settings,
            ).nelect
        )


def spin_degeneracy_from_vasprun(vasprun: Vasprun, charge_state: int | None = None) -> int:
    """
    Get the spin degeneracy (multiplicity) of a system from a ``VASP`` vasprun
    output.

    Spin degeneracy is determined by first getting the total magnetization and
    thus electron spin (S = N_μB/2 -- where N_μB is the magnetization in Bohr
    magnetons (i.e. electronic units, as used in VASP), and using the spin
    multiplicity equation: ``g_spin = 2S + 1``. The total magnetization
    ``N_μB`` is determined using ``get_magnetization_from_vasprun`` (see
    docstring for details), and if this fails, then simple spin behaviour is
    assumed with singlet (S = 0) behaviour for even-electron systems and
    doublet behaviour (S = 1/2) for odd-electron systems.

    For non-collinear (NCL) magnetization (e.g. spin-orbit coupling (SOC)
    calculations), the magnetization ``N_μB`` becomes a vector (spinor), in
    which case we take the vector norm as the total magnetization. This can be
    non-integer in these cases (e.g. due to SOC mixing of spin states, as
    **_S_** is no longer a good quantum number). As an approximation for these
    cases, we round ``N_μB`` to the nearest integer which would be allowed
    under collinear magnetism (i.e. even numbers for even-electron systems, odd
    numbers for odd-electron systems).

    Args:
        vasprun (|Vasprun|):
            ``pymatgen`` |Vasprun| for which to determine spin degeneracy.
        charge_state (int):
            The charge state of the system, which can be used to determine the
            number of electrons. If ``None`` (default), automatically
            determines the number of electrons using
            ``get_nelect_from_vasprun(vasprun)``.

    Returns:
        int: Spin degeneracy of the system.
    """
    from doped.utils.symmetry import (  # avoid circular import (symmetry imports doped.core)
        _num_electrons_from_charge_state,
        _spin_degeneracy_from_num_electrons_and_magnetization,
    )

    if charge_state is None:
        num_electrons = get_nelect_from_vasprun(vasprun)
    else:
        num_electrons = _num_electrons_from_charge_state(vasprun.final_structure, charge_state)

    try:
        magnetization: float | np.ndarray | None = get_magnetization_from_vasprun(vasprun)
    except (RuntimeError, TypeError):  # NCL calculation without parsed projected magnetization:
        magnetization = None  # guess from electron count

    return _spin_degeneracy_from_num_electrons_and_magnetization(int(num_electrons), magnetization)


def total_charge_from_vasprun(vasprun: Vasprun) -> int | None:
    """
    Determine the total charge state of a system from the vasprun, and compare
    to the expected charge state if provided.

    Note that if the system is charged, then this function relies on access to
    ``POTCAR`` data, which can be setup with ``pymatgen`` as detailed on the
    :ref:`installation page <setup_potcars_mp_api>`.

    Args:
        vasprun (|Vasprun|):
            ``pymatgen`` |Vasprun| object for which to determine the total
            charge.

    Returns:
        int or None:
            The total charge state, or ``None`` if it cannot be determined.
    """
    if (nelect := vasprun.incar.get("NELECT")) is None:
        return 0  # neutral if NELECT not specified

    auto_charge = None
    with contextlib.suppress(Exception):  # otherwise determine neutral NELECT from vasprun & POTCARs:
        neutral_nelect = get_neutral_nelect_from_vasprun(vasprun)
        auto_charge = -1 * (nelect - neutral_nelect)

        if abs(auto_charge) >= 10:
            neutral_nelect = get_neutral_nelect_from_vasprun(vasprun, skip_potcar_init=True)
            auto_charge = -1 * (nelect - neutral_nelect)

    return auto_charge


def _get_bulk_locpot_dict(bulk_path, quiet=False):
    bulk_locpot_path, multiple = _get_output_files_and_check_if_multiple("LOCPOT", bulk_path)
    if multiple and not quiet:
        _multiple_files_warning(
            "LOCPOT",
            bulk_path,
            bulk_locpot_path,
            dir_type="bulk",
        )
    bulk_locpot = get_locpot(bulk_locpot_path)
    return {str(k): bulk_locpot.get_average_along_axis(k) for k in [0, 1, 2]}


def _get_bulk_site_potentials(
    bulk_path: PathLike, quiet: bool = False, total_energy: list | float | None = None
):
    bulk_outcar_path, multiple = _get_output_files_and_check_if_multiple("OUTCAR", bulk_path)
    if multiple and not quiet:
        _multiple_files_warning(
            "OUTCAR",
            bulk_path,
            bulk_outcar_path,
            dir_type="bulk",
        )
    return get_core_potentials_from_outcar(bulk_outcar_path, dir_type="bulk", total_energy=total_energy)


_vasp_file_parsing_action_dict = {
    "vasprun.xml": "parse the calculation energy and metadata.",
    "OUTCAR": "parse core levels and compute the Kumagai (eFNV) image charge correction.",
    "LOCPOT": "parse the electrostatic potential and compute the Freysoldt (FNV) charge correction.",
}


def _multiple_files_warning(file_type, directory, chosen_filepath, action=None, dir_type="bulk"):
    """
    Warn that multiple files matching ``file_type`` were found in
    ``directory``, with ``action`` defaulting to the VASP file parsing actions
    (``_vasp_file_parsing_action_dict``); VASP-defaulted wrapper of
    :func:`doped.io.utils._multiple_files_warning`.
    """
    if action is None:
        action = _vasp_file_parsing_action_dict[file_type]
    _io_utils._multiple_files_warning(file_type, directory, chosen_filepath, action, dir_type)


def _parse_vr_and_poss_procar(
    output_path: PathLike,
    parse_projected_eigen: bool | None = None,
    label: str = "bulk",
    parse_procar: bool = True,
):
    """
    Parse the ``vasprun.xml(.gz)`` file at ``output_path``, and possibly a
    ``PROCAR`` file if both ``parse_procar`` and ``parse_projected_eigen`` are
    ``True`` and  projected eigenvalues cannot be parsed from the
    ``vasprun.xml(.gz)`` file.
    """
    procar = None
    failed_eig_parsing_warning_message = (
        f"Could not parse eigenvalue data from vasprun.xml.gz files in {label} folder at {output_path}"
    )

    vr_path, multiple = _get_output_files_and_check_if_multiple("vasprun.xml", output_path)
    if multiple:
        _multiple_files_warning("vasprun.xml", output_path, vr_path, dir_type=label)

    try:
        vr = get_vasprun(
            vr_path,
            parse_projected_eigen=parse_projected_eigen is not False,
            parse_eigen=(parse_projected_eigen is not False or label == "bulk"),
        )  # vr.eigenvalues not needed for defects except for vr-only eigenvalue analysis
    except Exception as vr_exc:
        vr = get_vasprun(vr_path, parse_projected_eigen=False, parse_eigen=label == "bulk")
        failed_eig_parsing_warning_message += f", got error:\n{vr_exc}"

        if parse_procar:
            procar_path, multiple = _get_output_files_and_check_if_multiple("PROCAR", output_path)
            if multiple:
                _multiple_files_warning("PROCAR", output_path, procar_path, dir_type=label)
            if "PROCAR" in procar_path and parse_projected_eigen is not False:
                try:
                    procar = get_procar(procar_path)

                except Exception as procar_exc:
                    failed_eig_parsing_warning_message += (
                        f"\nThen got the following error when attempting to parse projected eigenvalues "
                        f"from the defect PROCAR(.gz):\n{procar_exc}"
                    )

    if vr.projected_eigenvalues is None and procar is None and parse_projected_eigen is True:
        # only warn if parse_projected_eigen is set to True (not None)
        warnings.warn(failed_eig_parsing_warning_message)

    return vr, procar if parse_procar else vr


def get_calculation_outputs(
    path: PathLike,
    load_planar_averaged_potentials: bool = False,
    load_site_potentials: bool = False,
    **kwargs,
) -> CalculationOutputs:
    """
    Parse the outputs of a VASP supercell calculation in ``path`` to a
    (calculator-agnostic) :class:`~doped.io.outputs.CalculationOutputs` object.

    The ``vasprun.xml(.gz)`` file in ``path`` is parsed for energies,
    structures, eigenvalue data and calculation metadata, with optional
    parsing of the ``LOCPOT`` (planar-averaged electrostatic potentials, for
    Freysoldt (FNV) charge corrections) and ``OUTCAR`` (atomic-site core
    potentials, for Kumagai (eFNV) charge corrections) files.

    Args:
        path (PathLike):
            Path to the VASP calculation directory, containing the
            ``vasprun.xml(.gz)`` file to parse.
        load_planar_averaged_potentials (bool):
            Whether to also parse the planar-averaged electrostatic potentials
            from the ``LOCPOT(.gz)`` file in ``path``. Default is ``False``.
        load_site_potentials (bool):
            Whether to also parse the atomic-site core potentials from the
            ``OUTCAR(.gz)`` file in ``path``. Default is ``False``.
        **kwargs:
            Additional keyword arguments to pass to :func:`get_vasprun` (e.g.
            ``parse_projected_eigen``, ``parse_mag``).

    Returns:
        CalculationOutputs: The parsed calculation outputs.
    """
    vr_path, multiple = _get_output_files_and_check_if_multiple("vasprun.xml", path)
    if multiple:
        _multiple_files_warning("vasprun.xml", path, vr_path, dir_type="calculation")
    vr = get_vasprun(vr_path, **kwargs)

    planar_averaged_potentials = None
    if load_planar_averaged_potentials:
        locpot_path, multiple = _get_output_files_and_check_if_multiple("LOCPOT", path)
        if multiple:
            _multiple_files_warning("LOCPOT", path, locpot_path, dir_type="calculation")
        locpot = get_locpot(locpot_path)
        planar_averaged_potentials = {axis: locpot.get_average_along_axis(axis) for axis in [0, 1, 2]}

    site_potentials = None
    if load_site_potentials:
        outcar_path, multiple = _get_output_files_and_check_if_multiple("OUTCAR", path)
        if multiple:
            _multiple_files_warning("OUTCAR", path, outcar_path, dir_type="calculation")
        site_potentials = get_core_potentials_from_outcar(
            outcar_path, dir_type="calculation", total_energy=vr.final_energy
        )

    return calculation_outputs_from_vasprun(
        vr,
        path=path,
        planar_averaged_potentials=planar_averaged_potentials,
        site_potentials=site_potentials,
    )


def calculation_outputs_from_vasprun(
    vasprun: Vasprun,
    procar: PathLike | Procar | None = None,
    path: PathLike | None = None,
    planar_averaged_potentials: dict[int, np.ndarray] | None = None,
    site_potentials: list | np.ndarray | None = None,
) -> CalculationOutputs:
    """
    Build a (calculator-agnostic) :class:`~doped.io.outputs.CalculationOutputs`
    object from already-parsed VASP output objects.

    The input ``vasprun`` (and ``procar``, if provided) objects are kept in
    the (non-serialised) ``CalculationOutputs.raw`` dict (as ``"vasprun"`` /
    ``"procar"``, alongside a ``"computed_entry"``
    ``ComputedStructureEntry``), for reuse by VASP-specific code without
    re-parsing.

    Args:
        vasprun (|Vasprun|):
            ``pymatgen`` |Vasprun| object for the calculation.
        procar (PathLike | |Procar|):
            Path to a ``PROCAR(.gz)`` file or a ``pymatgen`` |Procar| object
            for the calculation, if parsed (stored in ``raw["procar"]`` for
            eigenvalue analyses when the ``vasprun`` lacks orbital
            projections). Default is ``None``.
        path (PathLike):
            Directory from which the outputs were parsed, if applicable.
        planar_averaged_potentials (dict[int, np.ndarray]):
            Planar-averaged electrostatic potentials (from ``LOCPOT``), if
            already parsed. Default is ``None``.
        site_potentials (list | np.ndarray):
            Atomic-site core potentials (from ``OUTCAR``), if already parsed.
            Default is ``None``.

    Returns:
        CalculationOutputs: The calculation outputs.
    """
    try:
        magnetization = get_magnetization_from_vasprun(vasprun)
    except (RuntimeError, TypeError):  # NCL calculation without parsed projected magnetization
        magnetization = None

    band_gap, cbm, vbm, _direct = (
        vasprun.eigenvalue_band_properties if vasprun.eigenvalues else (None,) * 4
    )

    charge = None
    with contextlib.suppress(Exception):
        charge = total_charge_from_vasprun(vasprun)

    return CalculationOutputs(
        structure=vasprun.final_structure,
        energy=vasprun.final_energy,
        calculator="vasp",
        directory=path,
        converged_electronic=vasprun.converged_electronic,
        converged_ionic=vasprun.converged_ionic,
        efermi=vasprun.efermi,
        eigenvalues=vasprun.eigenvalues,
        projected_eigenvalues=vasprun.projected_eigenvalues,
        projected_magnetisation=getattr(vasprun, "projected_magnetization", None),
        kpoint_coords=np.array(vasprun.actual_kpoints),
        kpoint_weights=np.array(vasprun.actual_kpoints_weights),
        nelect=vasprun.parameters.get("NELECT"),
        charge=charge,
        magnetization=magnetization,
        noncollinear=vasprun.parameters.get("LNONCOLLINEAR"),
        vbm=vbm,
        cbm=cbm,
        band_gap=band_gap,
        planar_averaged_potentials=planar_averaged_potentials,
        site_potentials=site_potentials,
        run_metadata={
            "incar": vasprun.incar,
            "kpoints": vasprun.kpoints,
            "actual_kpoints": vasprun.actual_kpoints,
            "potcar_symbols": vasprun.potcar_spec,
        },
        raw={
            "vasprun": vasprun,
            "procar": _parse_procar(procar) if procar is not None else None,
            "computed_entry": vasprun.get_computed_entry(),
        },
    )
