"""
Parsing of VASP defect / bulk supercell calculation outputs.

These functions load and process VASP output files (``vasprun.xml(.gz)``,
``OUTCAR``, ``LOCPOT``, ``PROCAR``), and can provide the parsed outputs in
calculator-agnostic form (:class:`~doped.io.outputs.CalculationOutputs`) via
:func:`get_calculation_outputs`.
"""

import contextlib
import itertools
import os
import re
import warnings
from collections.abc import Iterable
from functools import lru_cache, partialmethod
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element as XML_Element

import numpy as np
import pandas as pd
from monty.io import reverse_readfile
from monty.serialization import loadfn
from pymatgen.core.entries import ComputedStructureEntry
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.dos import FermiDos
from pymatgen.io.vasp.inputs import POTCAR_STATS_PATH, UnknownPotcarWarning
from pymatgen.io.vasp.outputs import Locpot, Outcar, Procar, Vasprun, _parse_vasp_array
from pymatgen.util.typing import PathLike

from doped.io import utils as _io_utils
from doped.io.outputs import CalculationOutputs, nelect_from_eigenvalues
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


CALC_OUTPUT_MASK = ("vasprun.xml", "vasprun.xml.gz")
"""
Filename patterns identifying (VASP) calculation output files, used for
calculation folder discovery.

Part of the ``doped.io`` backend protocol.
"""

SUBFOLDER_PRIORITY = [
    "vasp_ncl",
    "singlepoint",
    "final",
    "relax",
    "vasp_std",
    "vasp_nkred_std",
    "vasp_gam",
]
"""
Priority order when auto-detecting (VASP) calculation subfolders.

Part of the
``doped.io`` backend protocol.
"""


def _get_calc_files_df(root: Path, calc_output_mask: Iterable[str] = CALC_OUTPUT_MASK) -> pd.DataFrame:
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
    subfolder_priority: list[str] = SUBFOLDER_PRIORITY,
) -> str:
    """
    Pick the highest-priority calculation subfolder name present in
    ``files_df`` (defaulting to the VASP subfolder priority order,
    ``SUBFOLDER_PRIORITY``); VASP-defaulted wrapper of
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
    ``SUBFOLDER_PRIORITY``) when ``subfolder`` is ``None``; VASP-defaulted
    wrapper of :func:`doped.io.utils._find_calc_outputs`.

    Shared discovery logic used by both :func:`~doped.parsing.DefectsParser`
    and :meth:`~doped.chemical_potentials.CompetingPhasesAnalyzer`.
    """
    return _io_utils._find_calc_outputs(
        output_path,
        subfolder,
        calc_output_mask=CALC_OUTPUT_MASK,
        subfolder_priority=SUBFOLDER_PRIORITY,
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

    Used in ``doped.parsing`` and ``doped.chemical_potentials`` when checking
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
    # ``NELECT`` is written near the start of ``vasprun.xml`` (so is also available for truncated files
    # from crashed/killed runs, unlike the eigenvalues & occupancies at the end); can also be obtained
    # from ``OUTCAR`` files, along with the charge and magnetization, if ever needed
    if (nelect := vasprun.parameters.get("NELECT")) is not None:
        return nelect

    return nelect_from_eigenvalues(  # else reverse-engineer from the band occupancies:
        vasprun.eigenvalues,
        vasprun.actual_kpoints_weights,
        noncollinear=bool(vasprun.parameters.get("LNONCOLLINEAR", False)),
    )


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

    Convenience (VASP) wrapper for
    :meth:`~doped.io.outputs.CalculationOutputs.spin_degeneracy`, determining
    the spin degeneracy from the electron count (``NELECT``, or
    ``charge_state`` if provided) and the total magnetization -- see
    :func:`get_magnetization_from_vasprun` and
    :func:`~doped.utils.symmetry._spin_degeneracy_from_num_electrons_and_magnetization`
    for details (including the handling of non-collinear (NCL) magnetization).

    Args:
        vasprun (|Vasprun|):
            ``pymatgen`` |Vasprun| for which to determine spin degeneracy.
        charge_state (int):
            The charge state of the system, which can be used to determine the
            number of electrons. If ``None`` (default), the number of electrons
            is taken from the calculation ``NELECT``.

    Returns:
        int: Spin degeneracy of the system.
    """
    return calculation_outputs_from_vasprun(vasprun).spin_degeneracy(charge_state)


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


def _total_energies_for_potential_check(
    entry_energy: float | None, run_metadata: dict | None, label: str = ""
) -> list[float]:
    """
    Build the list of accepted total energies for cross-checking a supplied
    ``OUTCAR`` against the parsed calculation (see
    :func:`_check_outcar_energy`).

    VASP reports both the free energy and energy(sigma->0), and the ``OUTCAR``
    final energy may match either, so the latter is added (from the serialised
    ``vasprun`` dict in ``run_metadata``) when available.

    Note that an empty list is returned if ``entry_energy`` is ``None`` (i.e.
    the entry energy could not be determined), skipping the cross-check
    entirely -- this is deliberate, matching the previous behaviour where both
    energies were fetched under a single ``contextlib.suppress``.

    Args:
        entry_energy (float):
            The final energy of the corresponding parsed ``ComputedEntry``,
            if available.
        run_metadata (dict):
            ``DefectEntry.calculation_metadata["run_metadata"]``, from which
            the energy(sigma->0) value is taken if present.
        label (str):
            ``"defect"`` or ``"bulk"``, selecting the ``run_metadata`` entry.

    Returns:
        list[float]: The accepted total energies (in eV).
    """
    if entry_energy is None:
        return []

    energies = [entry_energy]
    with contextlib.suppress(Exception):
        energies.append(
            (run_metadata or {})[f"{label}_vasprun_dict"]["output"]["ionic_steps"][-1]["electronic_steps"][
                -1
            ]["e_0_energy"]
        )
    return energies


def get_potentials_from_input(
    potentials_input: PathLike | Locpot | Outcar | dict | list,
    potential_type: str = "planar",
    dir_type: str = "",
    entry_energy: float | None = None,
    run_metadata: dict | None = None,
):
    """
    Get planar-averaged (``potential_type="planar"``) or atomic-site
    (``"site"``) electrostatic potentials from calculator-native inputs, for
    finite-size charge corrections.

    Accepts a path to a ``LOCPOT``/``OUTCAR`` file, an already-loaded
    ``Locpot``/``Outcar`` object, or already-parsed potentials
    (dict/list), which are returned as-is. When an ``OUTCAR`` is supplied, its
    final energy is cross-checked against the parsed calculation energies (see
    :func:`_total_energies_for_potential_check`) to catch mismatched
    calculations.

    Note that ``Locpot`` objects are deliberately `not` converted to
    planar-averaged potential dictionaries here, as ``pymatgen``'s FNV
    correction accepts either (and uses the ``Locpot`` lattice directly).

    Part of the ``doped.io`` backend protocol.

    Args:
        potentials_input (PathLike | |Locpot| | |Outcar| | dict | list):
            The calculator-native potentials input (see above).
        potential_type (str):
            ``"planar"`` for planar-averaged potentials (Freysoldt/FNV
            correction, from ``LOCPOT``) or ``"site"`` for atomic-site
            potentials (Kumagai/eFNV correction, from ``OUTCAR``). Default is
            ``"planar"``.
        dir_type (str):
            The type of directory being parsed (e.g. ``"bulk"`` or
            ``"defect"``), for informative warnings/errors.
        entry_energy (float):
            The final energy of the corresponding parsed ``ComputedEntry``,
            for the ``OUTCAR`` energy cross-check.
        run_metadata (dict):
            ``DefectEntry.calculation_metadata["run_metadata"]``, for the
            ``OUTCAR`` energy cross-check.

    Returns:
        The planar-averaged potentials (``Locpot`` object or dict) or
        atomic-site potentials (list), depending on ``potential_type``.
    """
    total_energy = _total_energies_for_potential_check(entry_energy, run_metadata, dir_type)

    if isinstance(potentials_input, PathLike):
        if potential_type == "planar":
            return get_locpot(potentials_input)
        return get_core_potentials_from_outcar(  # otherwise OUTCAR
            potentials_input, dir_type=dir_type, total_energy=total_energy
        )

    if isinstance(potentials_input, Outcar):
        return _get_core_potentials_from_outcar_obj(
            potentials_input, dir_type=dir_type, total_energy=total_energy
        )

    if not isinstance(potentials_input, Locpot | dict | list):
        obj_type = "LOCPOT" if potential_type == "planar" else "OUTCAR"
        raise TypeError(
            f"`{obj_type.lower()}` input must be either a path to a {obj_type} file or a pymatgen "
            f"{obj_type[0] + obj_type[1:].lower()} object, but got {type(potentials_input)} instead."
        )

    return potentials_input


def get_fermi_dos(dos_path: PathLike | Vasprun) -> tuple[FermiDos, float, float]:
    """
    Create a ``pymatgen`` ``FermiDos`` object from the outputs of a bulk DOS
    calculation (``vasprun.xml(.gz)`` file, parsed with ``parse_dos = True``,
    with VASP), for calculating Fermi level positions and defect/carrier
    concentrations.

    Part of the ``doped.io`` backend protocol.

    Args:
        dos_path (PathLike | |Vasprun|):
            Path to a ``vasprun.xml(.gz)`` file from a bulk DOS calculation,
            or an already-parsed |Vasprun| object.

    Returns:
        tuple[FermiDos, float, float]:
            The ``FermiDos`` object, along with the VBM eigenvalue and band
            gap (in eV) from the DOS calculation -- used by ``doped`` to check
            consistency with the bulk supercell calculation.
    """
    if not isinstance(dos_path, Vasprun):
        dos_path = get_vasprun(dos_path, parse_dos=True)

    band_gap, _cbm, vbm, _ = dos_path.eigenvalue_band_properties
    return FermiDos(dos_path.complete_dos, nelecs=get_nelect_from_vasprun(dos_path)), vbm, band_gap


def get_competing_phase_entry(path: PathLike, **kwargs) -> ComputedStructureEntry:
    """
    Parse the outputs of a competing phase (bulk crystal) calculation in
    ``path`` to a ``pymatgen`` ``ComputedStructureEntry``, with the calculation
    summary info & settings in ``entry.data`` -- for chemical potential
    analysis with :class:`~doped.chemical_potentials.CompetingPhasesAnalyzer`.

    Part of the ``doped.io`` backend protocol.

    Args:
        path (PathLike):
            Path to the calculation directory, or directly to a
            ``vasprun.xml(.gz)`` file.
        **kwargs:
            Additional keyword arguments to pass to :func:`get_vasprun`.

    Returns:
        ComputedStructureEntry:
            The parsed entry, with ``entry.data`` containing the calculation
            summary info (``"summary"``; band gap & total magnetization),
            settings (``"incar"``, ``"kpoints"``, ``"potcar_symbols"``),
            convergence (``"converged_electronic"``, ``"converged_ionic"``)
            and the folder it was parsed from (``"folder"``).
    """
    vasprun = get_vasprun(path, **kwargs)
    entry = vasprun.get_computed_entry()
    unique_symbols = sorted(set(vasprun.atomic_symbols))
    summary_dict = {}
    with contextlib.suppress(Exception):  # non-essential properties, can fail with incomplete vasprun
        summary_dict["band_gap"] = vasprun.eigenvalue_band_properties[0]
        summary_dict["total_magnetization"] = get_magnetization_from_vasprun(vasprun)

    entry.data.update(
        {
            "formula_pretty": entry.composition.reduced_formula,
            "nsites": len(entry.structure),
            "volume": entry.structure.volume,
            "energy_per_atom": entry.energy_per_atom,
            "elements": unique_symbols,
            "nelements": len(unique_symbols),
            "kpoints": vasprun.kpoints.kpts,
            "incar": {k: v for k, v in vasprun.incar.as_dict().items() if "@" not in k},
            "potcar_symbols": vasprun.potcar_spec,
            "summary": summary_dict,
            "converged_electronic": vasprun.converged_electronic,
            "converged_ionic": vasprun.converged_ionic,
            "folder": str(path).removesuffix(".gz").removesuffix("vasprun.xml"),
        }
    )
    return entry


def get_planar_averaged_potentials(
    path: PathLike, dir_type: str = "bulk", quiet: bool = False
) -> dict[str, np.ndarray]:
    """
    Get the planar-averaged electrostatic potentials along each lattice vector
    for the calculation in ``path`` (from the ``LOCPOT(.gz)`` file with VASP),
    needed for Freysoldt (FNV) finite-size charge corrections.

    Part of the ``doped.io`` backend protocol.

    Args:
        path (PathLike):
            Path to the calculation directory.
        dir_type (str):
            The type of directory being parsed (e.g. ``"bulk"`` or
            ``"defect"``), for informative warnings/errors. Default is
            ``"bulk"``.
        quiet (bool):
            Whether to skip the multiple-files warning if several matching
            files are found. Default is ``False``.

    Returns:
        dict[str, np.ndarray]:
            The planar-averaged potentials, as ``{axis index (str): 1D
            array}``.
    """
    locpot_path, multiple = _get_output_files_and_check_if_multiple("LOCPOT", path)
    if multiple and not quiet:
        _multiple_files_warning("LOCPOT", path, locpot_path, dir_type=dir_type)
    locpot = get_locpot(locpot_path)
    return {str(k): locpot.get_average_along_axis(k) for k in [0, 1, 2]}


def get_site_potentials(
    path: PathLike,
    dir_type: str = "bulk",
    quiet: bool = False,
    outputs: CalculationOutputs | None = None,
    total_energy: list | float | None = None,
) -> np.ndarray:
    """
    Get the atomic-site electrostatic potentials for the calculation in
    ``path`` (from the core potentials in the ``OUTCAR(.gz)`` file with VASP),
    needed for Kumagai (eFNV) finite-size charge corrections.

    Part of the ``doped.io`` backend protocol.

    Args:
        path (PathLike):
            Path to the calculation directory.
        dir_type (str):
            The type of directory being parsed (e.g. ``"bulk"`` or
            ``"defect"``), for informative warnings/errors. Default is
            ``"bulk"``.
        quiet (bool):
            Whether to skip the multiple-files warning if several matching
            files are found. Default is ``False``.
        outputs (CalculationOutputs):
            Already-parsed :class:`~doped.io.outputs.CalculationOutputs` for
            `this` calculation, if available, used to cross-check the total
            energy of the parsed ``OUTCAR`` (warning if mismatching, i.e. if
            an inconsistent file combination is being used). Default is
            ``None``.
        total_energy (list | float):
            Total energy / energies to cross-check the parsed ``OUTCAR``
            energy against (alternative to ``outputs``). Default is ``None``.

    Returns:
        np.ndarray: The atomic-site electrostatic potentials.
    """
    outcar_path, multiple = _get_output_files_and_check_if_multiple("OUTCAR", path)
    if multiple and not quiet:
        _multiple_files_warning("OUTCAR", path, outcar_path, dir_type=dir_type)
    if total_energy is None and outputs is not None:
        total_energy = _total_energies_from_outputs(outputs)
    return get_core_potentials_from_outcar(outcar_path, dir_type=dir_type, total_energy=total_energy)


def _total_energies_from_outputs(outputs: CalculationOutputs) -> list[float]:
    """
    Get the total energy / energies of a parsed calculation (final energy, plus
    the last electronic step energy from the raw ``Vasprun`` if available), for
    energy cross-checks.
    """
    energies = [outputs.energy]
    if (vr := outputs.raw.get("vasprun")) is not None:
        with contextlib.suppress(Exception):
            energies.append(vr.ionic_steps[-1]["electronic_steps"][-1]["e_0_energy"])
    return [energy for energy in energies if energy is not None]


SITE_POTENTIALS_FILE = "OUTCAR"
"""
Name of the (VASP) output file providing atomic-site electrostatic potentials
(for Kumagai (eFNV) charge corrections).

Part of the ``doped.io`` backend
protocol.
"""

PLANAR_POTENTIALS_FILE = "LOCPOT"
"""
Name of the (VASP) output file providing planar-averaged electrostatic
potentials (for Freysoldt (FNV) charge corrections).

Part of the ``doped.io``
backend protocol.
"""

FILE_PARSING_ACTIONS = {
    "vasprun.xml": "parse the calculation energy and metadata.",
    "OUTCAR": "parse core levels and compute the Kumagai (eFNV) image charge correction.",
    "LOCPOT": "parse the electrostatic potential and compute the Freysoldt (FNV) charge correction.",
    "PROCAR": "parse orbital projections for eigenvalue analysis.",
}
"""
The (VASP) calculation output file types parsed by ``doped``, and what they are
used for (for informative warning messages).

Part of the ``doped.io``
backend protocol.
"""

MISMATCH_WARNING_SPECS = {
    "mismatching_INCAR_tags": {
        "object_name": "INCAR tags",
        "per_defect_warning_prefix": "There are mismatching",
        "transform": set,
        "message": lambda lst: (
            "'Defects: (INCAR tag, value in defect calculation, value in bulk calculation))':\n"
            f"{_format_mismatching_incar_warning(lst)}\n"
            "In general, the same INCAR settings should be used in all final calculations for these "
            "tags which can affect energies!"
        ),
    },
    "mismatching_KPOINTS": {
        "object_name": "KPOINTS",
        "per_defect_warning_prefix": "The KPOINTS",
        "transform": lambda defect_and_bulk_kpoints_lists: [
            [[float(kpt) for kpt in kpoints] for kpoints in kpoints_list]
            for kpoints_list in defect_and_bulk_kpoints_lists
        ],
        "message": lambda lst: (
            "(defect kpoints, bulk kpoints)):\n" + "\n".join(f"{n}: {m}" for n, m in lst) + "\n"
            "In general, the same KPOINTS settings should be used for all final calculations for "
            "accurate results!"
        ),
    },
    "mismatching_POTCAR_symbols": {
        "object_name": "POTCAR symbols",
        "per_defect_warning_prefix": "The POTCAR",
        "transform": lambda v: v,
        "message": lambda lst: (
            "(defect POTCARs, bulk POTCARs)):\n" + "\n".join(f"{n}: {m}" for n, m in lst) + "\n"
            "In general, the same POTCAR settings should be used for all calculations for accurate "
            "results!"
        ),
    },
}
"""
Specifications for collectively warning about mismatching (VASP) defect/bulk
calculation settings, as stored in ``DefectEntry.calculation_metadata`` by
``check_run_compatibility()``: for each metadata key, the human-readable
setting name, the per-defect warning prefix (for identifying warnings to
aggregate), and the value transform & message formatting functions.

Part of the ``doped.io`` backend protocol.
"""


def _multiple_files_warning(file_type, directory, chosen_filepath, action=None, dir_type="bulk"):
    """
    Warn that multiple files matching ``file_type`` were found in
    ``directory``, with ``action`` defaulting to the VASP file parsing actions
    (``FILE_PARSING_ACTIONS``); VASP-defaulted wrapper of
    :func:`doped.io.utils._multiple_files_warning`.
    """
    if action is None:
        action = FILE_PARSING_ACTIONS[file_type]
    _io_utils._multiple_files_warning(file_type, directory, chosen_filepath, action, dir_type)


def _parse_vr_and_poss_procar(
    output_path: PathLike,
    parse_projected_eigen: bool | None = None,
    label: str = "bulk",
    **kwargs,
):
    """
    Parse the ``vasprun.xml(.gz)`` file at ``output_path``, and possibly a
    ``PROCAR`` file if ``parse_projected_eigen`` is not ``False`` and projected
    eigenvalues cannot be parsed from the ``vasprun.xml(.gz)`` file.

    ``kwargs`` are passed to :func:`get_vasprun` (e.g. ``parse_mag``).
    """
    procar = None
    failed_eig_parsing_warning_message = (
        f"Could not parse eigenvalue data from vasprun.xml.gz files in {label} folder at {output_path}"
    )

    if not os.path.isdir(output_path):  # direct path to a ``vasprun.xml(.gz)`` file
        vr_path, multiple = output_path, False  # ``get_vasprun`` handles archived (e.g. ``.gz``) suffixes
        output_path = os.path.dirname(output_path) or "."  # for possible ``PROCAR`` fallback searches
    else:
        vr_path, multiple = _get_output_files_and_check_if_multiple("vasprun.xml", output_path)
    if multiple:
        _multiple_files_warning("vasprun.xml", output_path, vr_path, dir_type=label)

    # vr.eigenvalues not needed for defect supercells, except for vr-only eigenvalue analysis:
    parse_eigen = kwargs.pop("parse_eigen", parse_projected_eigen is not False or label != "defect")
    try:
        vr = get_vasprun(
            vr_path,
            parse_projected_eigen=parse_projected_eigen is not False,
            parse_eigen=parse_eigen,
            **kwargs,
        )
    except Exception as vr_exc:
        vr = get_vasprun(vr_path, parse_projected_eigen=False, parse_eigen=label != "defect", **kwargs)
        failed_eig_parsing_warning_message += f", got error:\n{vr_exc}"

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

    return vr, procar


def get_calculation_outputs(
    path: PathLike,
    load_planar_averaged_potentials: bool = False,
    load_site_potentials: bool = False,
    parse_projected_eigen: bool | None = None,
    label: str = "calculation",
    **kwargs,
) -> CalculationOutputs:
    """
    Parse the outputs of a VASP supercell calculation in ``path`` to a
    (calculator-agnostic) :class:`~doped.io.outputs.CalculationOutputs` object.

    The ``vasprun.xml(.gz)`` file in ``path`` is parsed for energies,
    structures, eigenvalue data and calculation metadata, with optional
    parsing of the ``LOCPOT`` (planar-averaged electrostatic potentials, for
    Freysoldt (FNV) charge corrections) and ``OUTCAR`` (atomic-site core
    potentials, for Kumagai (eFNV) charge corrections) files. The parsed
    ``Vasprun`` (and possible ``Procar``) objects are retained in the
    (non-serialised) ``CalculationOutputs.raw`` dict.

    Part of the ``doped.io`` backend protocol.

    Args:
        path (PathLike):
            Path to the VASP calculation directory (containing the
            ``vasprun.xml(.gz)`` file to parse), or directly to a
            ``vasprun.xml(.gz)`` file.
        load_planar_averaged_potentials (bool):
            Whether to also parse the planar-averaged electrostatic potentials
            from the ``LOCPOT(.gz)`` file in ``path``. Default is ``False``.
        load_site_potentials (bool):
            Whether to also parse the atomic-site core potentials from the
            ``OUTCAR(.gz)`` file in ``path``. Default is ``False``.
        parse_projected_eigen (bool):
            Whether to parse orbital projections (from the
            ``vasprun.xml(.gz)`` file, or failing that, a ``PROCAR(.gz)``
            file if present). If ``None`` (default), tries to parse
            projections but with no warning if this fails; if ``True``, warns
            on failure; if ``False``, skips projection parsing (and
            eigenvalue parsing for defect supercells, i.e. if
            ``label="defect"``) for expedited parsing.
        label (str):
            Label for the type of calculation being parsed (e.g. ``"bulk"``,
            ``"defect"``), for informative warnings and parsing efficiency
            choices. Default is ``"calculation"``.
        **kwargs:
            Additional keyword arguments to pass to :func:`get_vasprun` (e.g.
            ``parse_mag``).

    Returns:
        CalculationOutputs: The parsed calculation outputs.
    """
    vr, procar = _parse_vr_and_poss_procar(
        path, parse_projected_eigen=parse_projected_eigen, label=label, **kwargs
    )

    planar_averaged_potentials = None
    if load_planar_averaged_potentials:
        locpot_path, multiple = _get_output_files_and_check_if_multiple("LOCPOT", path)
        if multiple:
            _multiple_files_warning("LOCPOT", path, locpot_path, dir_type=label)
        locpot = get_locpot(locpot_path)
        planar_averaged_potentials = {axis: locpot.get_average_along_axis(axis) for axis in [0, 1, 2]}

    site_potentials = None
    if load_site_potentials:
        outcar_path, multiple = _get_output_files_and_check_if_multiple("OUTCAR", path)
        if multiple:
            _multiple_files_warning("OUTCAR", path, outcar_path, dir_type=label)
        site_potentials = get_core_potentials_from_outcar(
            outcar_path, dir_type=label, total_energy=vr.final_energy
        )

    return calculation_outputs_from_vasprun(
        vr,
        procar=procar,
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

    procar = _parse_procar(procar) if procar is not None else None
    projected_eigenvalues = vasprun.projected_eigenvalues
    if projected_eigenvalues is None and procar is not None:
        projected_eigenvalues = procar.data  # fall back to PROCAR orbital projections

    return CalculationOutputs(
        structure=vasprun.final_structure,
        energy=vasprun.final_energy,
        calculator="vasp",
        directory=path,
        converged_electronic=vasprun.converged_electronic,
        converged_ionic=vasprun.converged_ionic,
        efermi=vasprun.efermi,
        eigenvalues=vasprun.eigenvalues,
        projected_eigenvalues=projected_eigenvalues,
        kpoint_coords=np.array(vasprun.actual_kpoints),
        kpoint_weights=np.array(vasprun.actual_kpoints_weights),
        nelect=vasprun.parameters.get("NELECT"),
        charge=charge,
        magnetization=magnetization,
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
            "procar": procar,
            "computed_entry": vasprun.get_computed_entry(),
        },
    )


def load_eigenvalue_outputs(
    path: PathLike | None = None,
    outputs: PathLike | Vasprun | None = None,
    projections: PathLike | Procar | None = None,
    label: str = "bulk",
    run_metadata: dict | None = None,
) -> CalculationOutputs:
    """
    Load VASP calculation outputs `with orbital projections` (from
    ``vasprun.xml(.gz)``, or failing that ``PROCAR(.gz)``, files) for
    eigenvalue / band-edge analysis, raising an informative error if no
    projection data can be parsed.

    Part of the ``doped.io`` backend protocol (optional hook used by
    ``DefectEntry.get_eigenvalue_analysis()`` when eigenvalue data was not
    parsed up-front).

    Args:
        path (PathLike):
            Path to the calculation directory (e.g.
            ``DefectEntry.calculation_metadata["bulk_path"]``), to load
            output files from if ``outputs`` is not provided / lacks orbital
            projections.
        outputs (PathLike | |Vasprun|):
            Path to a ``vasprun.xml(.gz)`` file, or a ``pymatgen`` |Vasprun|
            object, if already loaded. Default is ``None``.
        projections (PathLike | |Procar|):
            Path to a ``PROCAR(.gz)`` file, or a ``pymatgen`` |Procar|
            object, if already loaded. Default is ``None``.
        label (str):
            Label for the type of calculation being parsed (e.g. ``"bulk"``,
            ``"defect"``), for informative warnings/errors. Default is
            ``"bulk"``.
        run_metadata (dict):
            The ``DefectEntry.calculation_metadata["run_metadata"]`` dict, to
            re-hydrate the |Vasprun| from its serialised
            ``{label}_vasprun_dict`` as a last resort. Default is ``None``.

    Returns:
        CalculationOutputs:
            The parsed calculation outputs, with ``projected_eigenvalues``.
    """
    vr, procar = outputs, projections
    if vr is not None and not isinstance(vr, Vasprun):  # just try loading from vasprun first
        with contextlib.suppress(Exception):
            vr = get_vasprun(vr, parse_projected_eigen=True)
        if not isinstance(vr, Vasprun):  # e.g. a directory path; fall back to searching ``path`` below
            vr = None

    if path is not None and (vr is None or vr.projected_eigenvalues is None):
        vr_path, multiple = _get_output_files_and_check_if_multiple("vasprun.xml", path)  # try from path
        if multiple:
            _multiple_files_warning("vasprun.xml", path, vr_path, dir_type=label)
        with contextlib.suppress(Exception):
            vr = get_vasprun(vr_path, parse_projected_eigen=True)

    if vr is None and procar is not None:  # then try take from serialised vasprun dict:
        with contextlib.suppress(Exception):
            vr = Vasprun.from_dict((run_metadata or {})[f"{label}_vasprun_dict"])

    if not isinstance(vr, Vasprun):
        raise FileNotFoundError(
            f"No {label} 'vasprun.xml(.gz)' file found (and successfully parsed) in path: {path}. "
            f"Required for eigenvalue analysis!"
        )

    # try load procar data, to see if projected eigenvalues are available:
    if procar is not None and vr.projected_eigenvalues is None:
        procar = _parse_procar(procar)

    if procar is None and path is not None and vr.projected_eigenvalues is None:
        # no procar, try parse from directory:
        try:
            procar_path, multiple = _get_output_files_and_check_if_multiple("PROCAR", path)
            if multiple:
                _multiple_files_warning("PROCAR", path, procar_path, dir_type=label)
            procar = get_procar(procar_path)

        except (FileNotFoundError, IsADirectoryError):
            procar = None

    if procar is None and vr.projected_eigenvalues is None:
        raise FileNotFoundError(
            f"No {label} 'PROCAR' or 'vasprun.xml(.gz)' file found (and successfully parsed) with "
            f"projected orbitals in path: {path}. Required for eigenvalue analysis!"
        )

    return calculation_outputs_from_vasprun(vr, procar=procar, path=path)


def _get_vr_dict_without_proj_eigenvalues(vr: Vasprun) -> dict:
    """
    Get the ``Vasprun.as_dict()`` representation, with the (large) projected
    eigenvalues / magnetization data excluded (as these are not needed in later
    stages of ``doped`` analysis workflows).
    """
    attributes_to_cut = ["projected_eigenvalues", "projected_magnetization"]
    orig_values = {}
    for attribute in attributes_to_cut:
        orig_values[attribute] = getattr(vr, attribute)
        setattr(vr, attribute, None)

    vr_dict = vr.as_dict()  # only call once
    vr_dict_wout_proj = {  # projected eigenvalue data might be present, but not needed (v slow
        # and data-heavy)
        **{k: v for k, v in vr_dict.items() if k != "output"},
        "output": {k: v for k, v in vr_dict["output"].items() if k not in attributes_to_cut},
    }
    for attribute in attributes_to_cut:
        vr_dict_wout_proj["output"][attribute] = None
        setattr(vr, attribute, orig_values[attribute])  # reset to original value

    return vr_dict_wout_proj


def check_entry_compatibility(entries, template_candidates=None) -> None:
    r"""
    Check the compatibility of parsed competing phase entries, by comparing
    their ``INCAR`` tags and ``POTCAR`` symbols against those of a reference
    (template) entry, and warning about any mismatches (which can cause errors
    in the parsed energies, and thus the chemical potential limits).

    Mismatches are also recorded in ``entry.data`` under
    ``"mismatching_INCAR_tags"`` / ``"mismatching_POTCAR_symbols"``, matching
    the corresponding keys set by :func:`check_run_compatibility` for defect
    & bulk supercell calculations. Entries without ``"incar"`` /
    ``"potcar_symbols"`` data (e.g. ``ComputedEntry``\s supplied directly by
    the user, rather than parsed from calculation outputs) are simply skipped
    for the corresponding check.

    Part of the ``doped.io`` backend protocol (optional; used by
    :class:`~doped.chemical_potentials.CompetingPhasesAnalyzer` when
    ``check_compatibility=True``).

    Args:
        entries (list[|ComputedEntry|]):
            The competing phase entries to check.
        template_candidates (list[|ComputedEntry|]):
            Priority-ordered candidate entries from which to take the
            reference (template) calculation settings; the first with the
            relevant data is used. Default is ``None``, in which case
            ``entries`` is used.
    """
    if template_candidates is None:
        template_candidates = entries

    sorted_entries_with_incar_data = [entry for entry in template_candidates if entry.data.get("incar")]
    sorted_entries_with_potcar_data = [
        entry for entry in template_candidates if entry.data.get("potcar_symbols")
    ]
    if sorted_entries_with_incar_data:
        incar_template_entry = sorted_entries_with_incar_data[0]
        for entry in entries:
            if not entry.data.get("incar"):  # no settings data for this entry (e.g. a user-supplied
                continue  # ``ComputedEntry``); skip rather than compare against nothing
            incar_mismatches = _compare_incar_tags(
                entry.data["incar"],
                incar_template_entry.data["incar"],
                ignore_tags={"NKRED"},  # no NKRED mismatch warnings for competing phases
                warn=False,
            )  # warned collectively below if any mismatches
            # ignore ISIF warnings in cases of supercell calculations (i.e. either gas calculations
            # or bulk supercell -- assumed to be the correct volume):
            if not isinstance(incar_mismatches, bool):
                incar_mismatches = [
                    i
                    for i in incar_mismatches
                    if i[0] != "ISIF"
                    or all(ent.structure.volume < 800 for ent in [incar_template_entry, entry])
                ]
            incar_mismatches = incar_mismatches if incar_mismatches else False
            entry.data["mismatching_INCAR_tags"] = (
                incar_mismatches if not (isinstance(incar_mismatches, bool)) else False
            )

        mismatching_INCAR_warnings = sorted(
            [
                (entry.name, set(entry.data.get("mismatching_INCAR_tags")))
                for entry in entries
                if entry.data.get("mismatching_INCAR_tags")
            ],
            key=lambda x: (len(x[1]), x[0]),
            reverse=True,
        )  # sort by number of mismatches, reversed
        if mismatching_INCAR_warnings:
            warnings.warn(
                f"There are mismatching INCAR tags for (some of) your competing phases "
                f"calculations which are likely to cause errors in the parsed results (energies "
                f"& thus chemical potential limits). Found the following differences:\n"
                f"(in the format: 'Entries: (INCAR tag, value in entry calculation, "
                f"value in reference calculation))':"
                f"\n{_format_mismatching_incar_warning(mismatching_INCAR_warnings)}\n"
                f"Where {incar_template_entry.name} was used as the reference entry calculation.\n"
                f"In general, the same INCAR settings should be used in all final calculations "
                f"for these tags which can affect energies!"
            )

    if sorted_entries_with_potcar_data:
        potcar_template_entry = sorted_entries_with_potcar_data[0]
        for entry in entries:
            if not entry.data.get("potcar_symbols"):  # no settings data for this entry; skip
                continue
            potcar_mismatches = _compare_potcar_symbols(
                entry.data["potcar_symbols"],
                potcar_template_entry.data["potcar_symbols"],
                warn=False,
                only_matching_elements=True,
            )  # warned collectively below if any mismatches
            entry.data["mismatching_POTCAR_symbols"] = (
                potcar_mismatches if not (isinstance(potcar_mismatches, bool)) else False
            )

        mismatching_potcars_warnings = sorted(
            [
                (entry.name, entry.data.get("mismatching_POTCAR_symbols"))
                for entry in entries
                if entry.data.get("mismatching_POTCAR_symbols")
            ],
            key=lambda x: (len(x[1]), x[0]),
            reverse=True,
        )  # sort by number of mismatches, reversed
        if mismatching_potcars_warnings:
            joined_info_string = "\n".join(
                [f"{name}: {mismatching}" for name, mismatching in mismatching_potcars_warnings]
            )
            warnings.warn(
                f"There are mismatching POTCAR symbols for (some of) your competing phases "
                f"calculations which are likely to cause errors in the parsed results (energies & "
                f"thus chemical potential limits). Found the following differences:\n"
                f"(in the format: (entry POTCARs, reference POTCARs)):"
                f"\n{joined_info_string}\n"
                f"Where {potcar_template_entry.name} was used as the reference entry "
                f"calculation.\n"
                f"In general, the same POTCAR settings should be used in all final calculations "
                f"for these tags which can affect energies!"
            )


def check_run_compatibility(
    defect_outputs: CalculationOutputs,
    bulk_outputs: CalculationOutputs,
    warn: bool = True,
) -> dict:
    """
    Check the compatibility of the calculation settings of a defect & bulk
    supercell calculation pair (INCAR tags, KPOINTS and POTCARs with VASP),
    returning the run metadata and any mismatches for storage in
    ``DefectEntry.calculation_metadata``.

    Part of the ``doped.io`` backend protocol.

    Args:
        defect_outputs (CalculationOutputs):
            Parsed outputs of the defect supercell calculation.
        bulk_outputs (CalculationOutputs):
            Parsed outputs of the reference bulk supercell calculation.
        warn (bool):
            Whether to warn about any found mismatches. Default is ``True``.

    Returns:
        dict:
            ``calculation_metadata`` updates: the ``"run_metadata"`` dict
            (INCAR/KPOINTS/POTCAR data, plus serialised ``Vasprun`` dicts
            when available), and ``"mismatching_INCAR_tags"``,
            ``"mismatching_POTCAR_symbols"`` & ``"mismatching_KPOINTS"``
            entries (``False``, or the mismatching values).
    """

    def _incar_dict(outputs: CalculationOutputs) -> dict:
        incar = outputs.run_metadata.get("incar", {})
        incar_dict = incar.as_dict() if hasattr(incar, "as_dict") else dict(incar)
        return {k: v for k, v in incar_dict.items() if "@" not in k}  # not JSONable with module keys

    run_metadata: dict[str, Any] = {
        "defect_incar": _incar_dict(defect_outputs),
        "bulk_incar": _incar_dict(bulk_outputs),
        "defect_kpoints": defect_outputs.run_metadata.get("kpoints"),
        "bulk_kpoints": bulk_outputs.run_metadata.get("kpoints"),
        "defect_actual_kpoints": defect_outputs.run_metadata.get("actual_kpoints"),
        "bulk_actual_kpoints": bulk_outputs.run_metadata.get("actual_kpoints"),
        "defect_potcar_symbols": defect_outputs.run_metadata.get("potcar_symbols"),
        "bulk_potcar_symbols": bulk_outputs.run_metadata.get("potcar_symbols"),
    }
    for label, outputs in [("defect", defect_outputs), ("bulk", bulk_outputs)]:
        if (vr := outputs.raw.get("vasprun")) is not None:
            run_metadata[f"{label}_vasprun_dict"] = _get_vr_dict_without_proj_eigenvalues(vr)

    incar_mismatches = _compare_incar_tags(
        run_metadata["defect_incar"], run_metadata["bulk_incar"], warn=warn
    )
    potcar_mismatches = _compare_potcar_symbols(
        run_metadata["defect_potcar_symbols"], run_metadata["bulk_potcar_symbols"], warn=warn
    )
    kpoint_mismatches = _compare_kpoints(
        run_metadata["defect_actual_kpoints"],
        run_metadata["bulk_actual_kpoints"],
        run_metadata["defect_kpoints"],
        run_metadata["bulk_kpoints"],
        warn=warn,
    )
    return {
        "mismatching_INCAR_tags": incar_mismatches if not isinstance(incar_mismatches, bool) else False,
        "mismatching_POTCAR_symbols": (
            potcar_mismatches if not isinstance(potcar_mismatches, bool) else False
        ),
        "mismatching_KPOINTS": kpoint_mismatches if not isinstance(kpoint_mismatches, bool) else False,
        "run_metadata": run_metadata,
    }
