"""
Helper functions for parsing defect supercell calculations.
"""

import contextlib
import itertools
import os
import re
import warnings
from collections.abc import Iterable
from copy import deepcopy
from functools import lru_cache, partialmethod
from pathlib import Path
from typing import Literal, Any
from xml.etree.ElementTree import Element as XML_Element

import numpy as np
import pandas as pd
from monty.io import reverse_readfile
from monty.serialization import loadfn
from pymatgen.analysis.defects.core import DefectType
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Composition, Lattice, PeriodicSite, Structure
from pymatgen.core.structure_matcher import get_linear_assignment_solution, pbc_shortest_vectors
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.inputs import POTCAR_STATS_PATH, UnknownPotcarWarning
from pymatgen.io.vasp.outputs import Locpot, Outcar, Procar, Vasprun, _parse_vasp_array
from pymatgen.util.coord import all_distances
from pymatgen.util.typing import PathLike, SpeciesLike
from scipy.interpolate import RegularGridInterpolator

from doped import _warn_parameter_order
from doped.core import DefectEntry, remove_site_oxi_state


@lru_cache(maxsize=1000)  # cache POTCAR generation to speed up generation and writing
def _get_potcar_summary_stats() -> dict:
    return loadfn(POTCAR_STATS_PATH)


def find_archived_fname(fname, raise_error=True):
    """
    Find a suitable filename, taking account of possible use of compression
    software.
    """
    if os.path.exists(fname):
        return fname
    # Check for archive files
    for ext in [".gz", ".xz", ".bz", ".lzma"]:
        if os.path.exists(fname + ext):
            return fname + ext
    if raise_error:
        raise FileNotFoundError
    return None


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


def _get_output_files_and_check_if_multiple(
    output_file: PathLike = "vasprun.xml",
    path: PathLike = ".",
    dir_type: str | None = None,
    quiet: bool = False,
) -> tuple[PathLike, bool]:
    """
    Search for all files with filenames matching ``output_file``, case-
    insensitive.

    Args:
        output_file (PathLike):
            The filename to search for (case-insensitive). Should be either
            ``vasprun.xml``, ``OUTCAR``, ``LOCPOT``, ``PROCAR``, ``.cube`` or
            ``.xml`` (matching any file with that extension).
        path (PathLike):
            The path to the directory to search in.
        dir_type (str | None):
            Optional label (e.g. ``"bulk"`` / ``"defect"``) for the directory.
            When provided and multiple matching files are found, a
            ``_multiple_files_warning`` is emitted internally (unless
            ``quiet``). When ``None`` (default), no warning is emitted and the
            caller is responsible for warning, preserving the original
            two-argument behaviour.
        quiet (bool):
            If ``True``, suppress the multiple-files warning even when
            ``dir_type`` is provided.

    Returns:
        Tuple[PathLike, bool]:
            The path to the identified file, and a boolean indicating whether
            multiple files were found.
    """
    if output_file.lower() == "vasprun.xml":
        search_patterns = ["vasprun", ".xml"]
    else:
        search_patterns = [output_file.lower()]

    files = os.listdir(path)
    output_files = [
        filename
        for filename in files
        if all(i in filename.lower() for i in search_patterns) and not filename.startswith(".")
    ]
    # sort by direct match to {output_file}, direct match to {output_file}.gz, then alphabetically:
    if output_files := sorted(
        output_files,
        key=lambda x: (x == output_file, x == f"{output_file}.gz", x),
        reverse=True,
    ):
        output_path = os.path.join(path, output_files[0])
        multiple = len(output_files) > 1
        if multiple and dir_type is not None and not quiet:
            _multiple_files_warning(output_file, path, output_path, dir_type=dir_type)
        return output_path, multiple
    return (
        os.path.join(path, output_file),
        False,
    )  # so `get_X()` will raise an informative FileNotFoundError


_CALC_OUTPUT_MASK = ("vasprun.xml", "vasprun.xml.gz")
"""
Filename patterns that identify calculation output files.
"""

_SUBFOLDER_PRIORITY = [
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


def _dataframe_of_files(root: Path) -> pd.DataFrame:
    """
    Get a dataframe with one row per file under ``root``, found recursively.

    Hidden files/directories (names starting with ``"."``) and files
    sitting directly in ``root`` (i.e. with only one path component
    relative to ``root``) are excluded.

    Columns: ``filename``, ``full_path``, ``folder_path``,
    ``folder_in_root`` (the first path component relative to ``root``).

    Args:
        root (Path):
            Path to the root directory.

    Returns:
        pd.DataFrame:
            One row per discovered file.
    """
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for f in root.rglob("*"):  # recursively find all files under root, ignoring hidden folders/files
        if f.is_file():
            relative_parts = f.relative_to(root).parts
            if any(part.startswith(".") for part in relative_parts) or len(relative_parts) < 2:
                continue  # ignore hidden files and folders, and files in root directory itself
            rows.append(
                {
                    "filename": f.name,
                    "full_path": f,
                    "folder_path": f.parent,
                    "folder_in_root": relative_parts[0],
                }
            )
    return pd.DataFrame(rows)


def _get_calc_files_df(root: Path, calc_output_mask: Iterable[str] = _CALC_OUTPUT_MASK) -> pd.DataFrame:
    """
    Get a DataFrame of calculation output files (matching ``calc_output_mask``)
    found recursively under ``root``, excluding hidden files/directories and
    files sitting directly in ``root``.

    This is a filtered view of :func:`_dataframe_of_files`.

    Args:
        root (Path):
            Path to the root directory.
        calc_output_mask (Iterable[str]):
            Iterable of filename patterns to match.  Defaults to
            ``_CALC_OUTPUT_MASK`` (``("vasprun.xml", "vasprun.xml.gz")``).
            Matching is case-insensitive.

    Returns:
        pd.DataFrame:
            One row per matching calculation output file.
    """
    files_df = _dataframe_of_files(root)
    if files_df.empty:
        return pd.DataFrame()
    pattern = "|".join(map(re.escape, calc_output_mask))
    return files_df[files_df["filename"].str.contains(pattern, regex=True, na=False)]


def _determine_subfolder(
    files_df: pd.DataFrame,
    candidate_folders: list[str] | None = None,
    subfolder_priority: list[str] = _SUBFOLDER_PRIORITY,
) -> str:
    """
    Pick the highest-priority calculation subfolder name present in
    ``files_df`` (restricted to rows whose ``folder_in_root`` is in
    ``candidate_folders``), or ``"."`` if none of the priority names are found.

    Args:
        files_df (pd.DataFrame):
            DataFrame produced by :func:`_dataframe_of_files`, filtered
            to calculation output files.
        candidate_folders (list[str]):
            Top-level folder names to consider. If ``None`` (default),
            considers all top-level folder names.
        subfolder_priority (list[str]):
            Priority order for subfolder names.  Defaults to
            ``_SUBFOLDER_PRIORITY``
            (``["vasp_ncl", "singlepoint", "final", "relax", "vasp_std", "vasp_nkred_std", "vasp_gam"]``)
            where folder names are compared case-insensitively.

    Returns:
        str:
            The detected subfolder name, or ``"."``.
    """
    candidate_folder_df = (
        files_df[files_df["folder_in_root"].isin(candidate_folders)]
        if candidate_folders is not None
        else files_df
    )
    for subfolder in subfolder_priority:
        if any(subfolder in p.name.lower() for p in candidate_folder_df["folder_path"].unique()):
            return subfolder
    return "."


def _find_calc_outputs(
    output_path: PathLike,
    subfolder: PathLike | None = None,
) -> tuple[pd.DataFrame, list[str], str]:
    """
    Recursively find calculation output files under ``output_path`` and auto-
    detect the calculation subfolder when ``subfolder`` is ``None``.

    Shared discovery logic used by both :func:`~doped.analysis.DefectsParser`
    and :meth:`~doped.chemical_potentials.CompetingPhasesAnalyzer`.

    Args:
        output_path (PathLike):
            Root directory to search.
        subfolder (PathLike | None):
            Explicit subfolder name (e.g. ``"vasp_std"``).  If
            ``None``, auto-detected using ``_SUBFOLDER_PRIORITY``
            (``["vasp_ncl", "singlepoint", "final", "relax", "vasp_std", "vasp_nkred_std", "vasp_gam"]``)
            where folder names are compared case-insensitively.

    Returns:
        tuple[pd.DataFrame, list[str], str]:
            ``(calc_files_df, candidate_folders, resolved_subfolder)``
            where *resolved_subfolder* is ``"."`` when no priority
            subfolder is found.
    """
    calc_files_df = _get_calc_files_df(Path(output_path))
    if calc_files_df.empty:
        return pd.DataFrame(), [], "."

    candidate_folders = calc_files_df["folder_in_root"].unique().tolist()
    resolved_subfolder = (
        _determine_subfolder(calc_files_df, candidate_folders) if subfolder is None else str(subfolder)
    )
    return calc_files_df, candidate_folders, resolved_subfolder


def get_defect_type_and_composition_diff(
    defect: Structure | Composition,
    bulk: Structure | Composition,
    _parameter_order_warn: bool = True,
) -> tuple[str, dict]:
    """
    Get the difference in composition between a bulk structure and a defect
    structure.

    Contributed by Dr. Alex Ganose (@ Imperial Chemistry) and refactored for
    extrinsic species and code efficiency/robustness improvements.

    Args:
        defect (|Structure| | |Composition|):
            The defect structure or composition.
        bulk (|Structure| | |Composition|):
            The bulk structure or composition.

    Returns:
        tuple[str, dict[str, int]]:
            The defect type (``interstitial``, ``vacancy`` or ``substitution``)
            and the composition difference between the bulk and defect
            structures as a dictionary.
    """
    if _parameter_order_warn:
        _warn_parameter_order("get_defect_type_and_composition_diff")  # TODO: Remove in doped v4.1
    bulk_comp = bulk.composition if isinstance(bulk, Structure) else bulk
    defect_comp = defect.composition if isinstance(defect, Structure) else defect

    bulk_comp_dict = bulk_comp.get_el_amt_dict()
    defect_comp_dict = defect_comp.get_el_amt_dict()

    composition_diff = {
        element: int(defect_amount - bulk_comp_dict.get(element, 0))
        for element, defect_amount in defect_comp_dict.items()
        if int(defect_amount - bulk_comp_dict.get(element, 0)) != 0
    }

    if len(composition_diff) == 1 and next(iter(composition_diff.values())) == 1:
        defect_type = "interstitial"
    elif len(composition_diff) == 1 and next(iter(composition_diff.values())) == -1:
        defect_type = "vacancy"
    elif len(composition_diff) == 2:
        defect_type = "substitution"
    else:
        raise RuntimeError(
            f"Could not determine defect type from composition difference of bulk ({bulk_comp_dict}) and "
            f"defect ({defect_comp_dict})."
        )

    return defect_type, composition_diff


def get_defect_type_site_idxs_and_unrelaxed_structure(
    defect_supercell: Structure,
    bulk_supercell: Structure,
    _parameter_order_warn: bool = True,
) -> tuple[str, int | None, int | None, Structure]:
    """
    Get the defect type, site (indices in the bulk and defect supercells) and
    unrelaxed structure, where 'unrelaxed structure' corresponds to the
    pristine defect supercell structure for vacancies / substitutions (with no
    relaxation), and the pristine bulk structure with the `final` relaxed
    interstitial site for interstitials.

    Note that this assumes consistent cell definitions (lattice vectors and
    bases) for the input defect and bulk supercells, and does not perform any
    structural re-orientations.

    Initial draft contributed by Dr. Alex Ganose (@ Imperial Chemistry) and
    refactored for extrinsic species and several code efficiency/robustness
    improvements.

    Args:
        defect_supercell (|Structure|):
            The defect supercell structure.
        bulk_supercell (|Structure|):
            The bulk supercell structure.

    Returns:
        defect_type (str):
            The type of defect as a string (``interstitial``, ``vacancy`` or
            ``substitution``).
        bulk_site_idx (int):
            Index of the site in the bulk structure that corresponds to the
            defect site in the defect structure.
        defect_site_idx (int):
            Index of the defect site in the defect structure.
        unrelaxed_defect_structure (|Structure|):
            Pristine defect supercell structure for vacancies/substitutions
            (i.e. pristine bulk with unrelaxed vacancy/substitution), or the
            pristine bulk structure with the `final` relaxed interstitial site
            for interstitials.
    """
    if _parameter_order_warn:
        _warn_parameter_order(  # TODO: Remove in doped v4.1
            "get_defect_type_site_idxs_and_unrelaxed_structure"
        )

    def process_substitution(defect_supercell, bulk_supercell, composition_diff):
        old_species = _get_species_from_composition_diff(composition_diff, -1)
        new_species = _get_species_from_composition_diff(composition_diff, 1)

        bulk_new_species_coords, _idx = get_coords_and_idx_of_species(bulk_supercell, new_species)
        defect_new_species_coords, defect_new_species_idx = get_coords_and_idx_of_species(
            defect_supercell, new_species
        )

        if bulk_new_species_coords.size > 0:  # intrinsic substitution
            # find coords of new species in defect structure, taking into account periodic boundaries
            defect_site_arg_idx = find_missing_idx(
                bulk_new_species_coords,
                defect_new_species_coords,
                bulk_supercell.lattice,
            )

        else:  # extrinsic substitution
            defect_site_arg_idx = 0

        # Get the coords and site index of the defect that was used in the calculation
        defect_site_idx = defect_new_species_idx[defect_site_arg_idx]

        # now find the closest old_species site in the bulk structure to the defect site
        # again, make sure to use periodic boundaries
        bulk_old_species_coords, bulk_old_species_idx = get_coords_and_idx_of_species(
            bulk_supercell, old_species
        )
        _bulk_coords, bulk_site_arg_idx = find_nearest_coords(
            bulk_old_species_coords,
            defect_new_species_coords[defect_site_arg_idx],  # defect coords
            bulk_supercell.lattice,
            return_idx=True,
        )
        bulk_site_idx = bulk_old_species_idx[bulk_site_arg_idx]
        unrelaxed_defect_structure = _create_unrelaxed_defect_structure(
            bulk_supercell,
            new_species=new_species,
            bulk_site_idx=bulk_site_idx,
            defect_site_idx=defect_site_idx,
        )
        return bulk_site_idx, defect_site_idx, unrelaxed_defect_structure

    def process_vacancy(defect_supercell, bulk_supercell, composition_diff):
        old_species = _get_species_from_composition_diff(composition_diff, -1)
        bulk_old_species_coords, bulk_old_species_idx = get_coords_and_idx_of_species(
            bulk_supercell, old_species
        )
        defect_old_species_coords, _idx = get_coords_and_idx_of_species(defect_supercell, old_species)

        bulk_site_arg_idx = find_missing_idx(
            bulk_old_species_coords,
            defect_old_species_coords,
            bulk_supercell.lattice,
        )
        bulk_site_idx = bulk_old_species_idx[bulk_site_arg_idx]
        defect_site_idx = None
        unrelaxed_defect_structure = _create_unrelaxed_defect_structure(
            bulk_supercell,
            bulk_site_idx=bulk_site_idx,
        )
        return bulk_site_idx, defect_site_idx, unrelaxed_defect_structure

    def process_interstitial(defect_supercell, bulk_supercell, composition_diff):
        new_species = _get_species_from_composition_diff(composition_diff, 1)

        bulk_new_species_coords, _bulk_new_species_idx = get_coords_and_idx_of_species(
            bulk_supercell, new_species
        )
        defect_new_species_coords, defect_new_species_idx = get_coords_and_idx_of_species(
            defect_supercell, new_species
        )

        if bulk_new_species_coords.size > 0:  # intrinsic interstitial
            defect_site_arg_idx = find_missing_idx(
                bulk_new_species_coords,
                defect_new_species_coords,
                bulk_supercell.lattice,
            )

        else:  # extrinsic interstitial
            defect_site_arg_idx = 0

        # Get the coords and site index of the defect that was used in the calculation
        defect_site_coords = defect_new_species_coords[defect_site_arg_idx]  # frac coords of defect site
        defect_site_idx = defect_new_species_idx[defect_site_arg_idx]

        bulk_site_idx = None
        unrelaxed_defect_structure = _create_unrelaxed_defect_structure(
            bulk_supercell,
            frac_coords=defect_site_coords,
            new_species=new_species,
            defect_site_idx=defect_site_idx,
        )
        return bulk_site_idx, defect_site_idx, unrelaxed_defect_structure

    handlers = {
        "substitution": process_substitution,
        "vacancy": process_vacancy,
        "interstitial": process_interstitial,
    }

    try:
        defect_type, comp_diff = get_defect_type_and_composition_diff(
            defect_supercell, bulk_supercell, _parameter_order_warn=False
        )
    except RuntimeError as exc:
        raise ValueError(
            "Could not identify defect type from number of sites in structure: "
            f"{len(bulk_supercell)} in bulk vs. {len(defect_supercell)} in defect?"
        ) from exc

    return (defect_type, *handlers[defect_type](defect_supercell, bulk_supercell, comp_diff))


def _get_species_from_composition_diff(composition_diff, el_change):
    """
    Get the species corresponding to the given change in composition.
    """
    return next(el for el, amt in composition_diff.items() if amt == el_change)


def get_coords_and_idx_of_species(structure_or_sites, species_name, frac_coords=True):
    """
    Get arrays of the coordinates and indices of the given species in the
    structure/list of sites.
    """
    from doped.utils.efficiency import _parse_site_species_str

    coords = []
    idx = []
    for i, site in enumerate(structure_or_sites):
        if _parse_site_species_str(site, wout_charge=True) == species_name:
            coords.append(site.frac_coords if frac_coords else site.coords)
            idx.append(i)

    return np.array(coords), np.array(idx)


def get_matching_site(
    site: PeriodicSite | np.ndarray, structure: Structure, anonymous: bool = False, tol: float = 0.5
) -> PeriodicSite:
    """
    Get the (closest) matching |PeriodicSite| in ``structure`` for the input
    ``site``, which can be a |PeriodicSite| or fractional coordinates.

    If the closest matching site in ``structure`` is > ``tol`` Å (0.5 Å by
    default) away from the input ``site`` coordinates, an error is raised.

    Automatically accounts for possible differences in assigned oxidation
    states, site property dicts etc.

    Args:
        site (|PeriodicSite| | np.ndarray):
            The site for which to find the closest matching site in
            ``structure``, either as a |PeriodicSite| or fractional
            coordinates array. If fractional coordinates, then ``anonymous``
            is set to ``True``.
        structure (|Structure|):
            The structure in which to search for matching sites to ``site``.
        anonymous (bool):
            Whether to use anonymous matching, allowing different
            species/elements to match each other (i.e. just matching based on
            coordinates). Default is ``False`` if ``site`` is a
            |PeriodicSite|, and ``True`` if ``site`` is fractional
            coordinates.
        tol (float):
            A distance tolerance (in Å), where an error will be thrown if the
            closest matching site is > ``tol`` Å away from the input ``site``.
            Default is 0.5 Å.

    Returns:
        PeriodicSite:
            The closest matching site in ``structure`` to the input ``site``.
    """
    if (
        isinstance(site, PeriodicSite) and not anonymous
    ):  # try directly match first         if site in structure:
        if site in structure:
            return site

        site_w_no_ox_state = deepcopy(site)
        remove_site_oxi_state(site_w_no_ox_state)
        site_w_no_ox_state.properties = {}

        bulk_sites_w_no_ox_state = structure.copy().sites
        for bulk_site in bulk_sites_w_no_ox_state:
            remove_site_oxi_state(bulk_site)
            bulk_site.properties = {}

        if site_w_no_ox_state in bulk_sites_w_no_ox_state:
            return structure.sites[bulk_sites_w_no_ox_state.index(site_w_no_ox_state)]

    site_frac_coords = (
        site.frac_coords if hasattr(site, "frac_coords") else np.array(site, dtype=float)
    )  # ensure site is in fractional coords

    # else get closest site in structure, raising error if not within tol Å:
    if isinstance(site, PeriodicSite) and not anonymous:  # reduce to only matching species
        candidate_frac_coords, candidate_indices = get_coords_and_idx_of_species(
            structure, site.specie.symbol, frac_coords=True
        )
    else:
        candidate_frac_coords = structure.frac_coords
        candidate_indices = np.arange(len(structure))

    closest_site_idx = candidate_indices[
        np.argmin(structure.lattice.get_all_distances(site_frac_coords, candidate_frac_coords).ravel())
    ]
    closest_site = structure.sites[closest_site_idx]

    closest_site_dist = closest_site.distance_and_image_from_frac_coords(site_frac_coords)[0]
    if closest_site_dist > tol:
        raise ValueError(
            f"Closest site to input defect site ({site}) in bulk supercell is {closest_site} "
            f"with distance {closest_site_dist:.2f} Å (greater than {tol} Å and suggesting a likely "
            f"mismatch in sites/structures here!)."
        )

    if (
        not anonymous
        and isinstance(site, PeriodicSite)
        and site.specie.symbol != closest_site.specie.symbol
    ):
        raise ValueError(
            f"Closest site to input defect site ({site}) in bulk supercell is {closest_site} "
            f"with distance {closest_site_dist:.2f} Å which is a different element! Set `anonymous=True` "
            f"to allow matching of different elements/species if this is desired."
        )

    return closest_site


def find_nearest_coords(
    candidate_frac_coords: list | np.ndarray,
    target_frac_coords: list | np.ndarray,
    lattice: Lattice,
    return_idx: bool = False,
) -> tuple[list | np.ndarray, int] | list | np.ndarray:
    """
    Find the nearest coords in ``candidate_frac_coords`` to
    ``target_frac_coords``.

    If ``return_idx`` is ``True``, also returns the index of the nearest coords
    in ``candidate_frac_coords`` to ``target_frac_coords``.

    Args:
        candidate_frac_coords (list | np.ndarray):
            Fractional coordinates (typically from a bulk supercell), to find
            the nearest coordinates to ``target_frac_coords``.
        target_frac_coords (list | np.ndarray):
            The target coordinates to find the nearest coordinates to in
            ``candidate_frac_coords``.
        lattice (|Lattice|):
            The lattice object to use with the fractional coordinates.
        return_idx (bool):
            Whether to also return the index of the nearest coordinates in
            ``candidate_frac_coords`` to ``target_frac_coords``.
    """
    if len(np.array(target_frac_coords).shape) > 1:
        raise ValueError("`target_frac_coords` should be a 1D array of fractional coordinates!")

    distance_matrix = lattice.get_all_distances(candidate_frac_coords, target_frac_coords).ravel()
    match = distance_matrix.argmin()

    return candidate_frac_coords[match], match if return_idx else candidate_frac_coords[match]


def find_missing_idx(
    frac_coords1: list | np.ndarray,
    frac_coords2: list | np.ndarray,
    lattice: Lattice,
):
    """
    Find the missing/outlier index between two sets of fractional coordinates
    (differing in size by 1), by grouping the coordinates based on the minimum
    distances between coordinates or, if that doesn't give a unique match, the
    site combination that gives the minimum summed squared distances between
    paired sites.

    The index returned is the index of the missing/outlier coordinate in the
    larger set of coordinates.

    Args:
        frac_coords1 (list | np.ndarray):
            First set of fractional coordinates.
        frac_coords2 (list | np.ndarray):
            Second set of fractional coordinates.
        lattice (|Lattice|):
            The lattice object to use with the fractional coordinates.
    """
    subset, superset = (  # supa-set
        (frac_coords1, frac_coords2)
        if len(frac_coords1) < len(frac_coords2)
        else (frac_coords2, frac_coords1)
    )
    # in theory this could be made even faster using ``lll_frac_tol`` as in ``_cart_dists()`` in
    # ``pymatgen``, with smart choice of initial ``lll_frac_tol`` and scanning upwards if the match is
    # below the threshold tolerance (as in ``StructureMatcher_scan_stol()``), but in practice this
    # function seems to be incredibly fast as is. Can revisit if it ever becomes a bottleneck
    _vecs, d_2 = pbc_shortest_vectors(lattice, subset, superset, return_d2=True)
    site_matches, _ = get_linear_assignment_solution(d_2)  # matching superset indices, of len(subset)

    return next(iter(set(np.arange(len(superset), dtype=int)) - set(site_matches)))


def _create_unrelaxed_defect_structure(
    bulk_supercell: Structure,
    frac_coords: list | np.ndarray | None = None,
    new_species: str | None = None,
    bulk_site_idx: int | None = None,
    defect_site_idx: int | None = None,
) -> Structure:
    """
    Create the unrelaxed defect structure, which corresponds to the bulk
    supercell with the unrelaxed defect site.

    The unrelaxed defect site corresponds to the vacancy/substitution site in
    the pristine (bulk) supercell for vacancies/substitutions, and the `final`
    relaxed interstitial site for interstitials (as the assignment of their
    initial site is ambiguous).

    Args:
        bulk_supercell (|Structure|):
            The bulk supercell structure.
        frac_coords (list | np.ndarray):
            The fractional coordinates of the defect site. Unnecessary if
            ``bulk_site_idx`` is provided.
        new_species (str):
            The species of the defect site. Unnecessary for vacancies.
        bulk_site_idx (int):
            The index of the site in the bulk structure that corresponds to the
            defect site in the defect structure.
        defect_site_idx (int):
            The index of the defect site to use in the unreleaxed defect
            structure. Just for consistency with the relaxed defect structure.

    Returns:
        Structure:
            The unrelaxed defect structure.
    """
    unrelaxed_defect_structure = bulk_supercell.copy()  # create unrelaxed defect structure

    if bulk_site_idx is not None:
        unrelaxed_defect_structure.remove_sites([bulk_site_idx])
        defect_coords = bulk_supercell[bulk_site_idx].frac_coords

    else:
        defect_coords = frac_coords

    if new_species is not None:  # not a vacancy
        # Place defect in same location as output from calculation
        defect_site_idx = (
            defect_site_idx if defect_site_idx is not None else len(unrelaxed_defect_structure)
        )  # use "is not None" to allow 0 index
        unrelaxed_defect_structure.insert(defect_site_idx, new_species, defect_coords)

    return unrelaxed_defect_structure


def get_wigner_seitz_radius(lattice: Structure | Lattice) -> float:
    """
    Calculates the Wigner-Seitz radius of the structure, which corresponds to
    the maximum radius of a sphere fitting inside the cell.

    Templated on the ``calc_max_sphere_radius`` function from ``pydefect``,
    but rewritten to avoid calling ``vise`` which causes hanging on Windows.
    (https://github.com/SMTG-Bham/doped/issues/147).

    Args:
        lattice (|Structure| | |Lattice|):
            The lattice of the structure (either a ``pymatgen`` |Structure|
            or |Lattice| object).

    Returns:
        float:
            The Wigner-Seitz radius of the structure.
    """
    lattice_matrix = lattice.matrix if isinstance(lattice, Lattice) else lattice.lattice.matrix
    distances = np.zeros(3, dtype=float)  # copied over from pydefect v0.9.4; avoid vise issues
    for i in range(3):
        a_i_a_j = np.cross(lattice_matrix[i - 2], lattice_matrix[i - 1])
        a_k = lattice_matrix[i]
        distances[i] = abs(np.dot(a_i_a_j, a_k)) / np.linalg.norm(a_i_a_j)
    return max(distances) / 2.0


def check_atom_mapping_far_from_defect(
    defect_supercell: Structure,
    bulk_supercell: Structure,
    defect_coords: np.ndarray,
    coords_are_cartesian: bool = False,
    displacement_tol: float = 0.5,
    warning: bool | str = "verbose",
) -> bool:
    """
    Check the displacement of atoms far from the determined defect site, and
    warn the user if they are large (often indicates a mismatch between the
    bulk and defect supercell definitions).

    The threshold for identifying 'large' displacements is if the mean
    displacement of any species is greater than ``displacement_tol`` Ångströms
    for sites of that species outside the Wigner-Seitz radius of the defect in
    the defect supercell. The Wigner-Seitz radius corresponds to the radius of
    the largest sphere which can fit in the cell.

    Args:
        defect_supercell (|Structure|):
            The defect structure.
        bulk_supercell (|Structure|):
            The bulk structure.
        defect_coords (np.ndarray):
            The coordinates of the defect site.
        coords_are_cartesian (bool):
            Whether the defect coordinates are in Cartesian or fractional
            coordinates. Default is ``False`` (fractional).
        displacement_tol (float):
            The tolerance for the displacement of atoms far from the defect
            site, in Ångströms. Default is 0.5 Å.
        warning (bool, str):
            Whether to throw a warning if a mismatch is detected. If
            ``warning = "verbose"`` (default), the individual atomic
            displacements are included in the warning message.

    Returns:
        bool:
            Returns ``False`` if a mismatch is detected, else ``True``.
    """
    far_from_defect_disps: dict[str, list[float]] = {site.specie.symbol: [] for site in bulk_supercell}
    wigner_seitz_radius = get_wigner_seitz_radius(bulk_supercell.lattice)
    defect_frac_coords = (
        defect_coords
        if not coords_are_cartesian
        else bulk_supercell.lattice.get_fractional_coords(defect_coords)
    )

    bulk_sites_outside_or_at_ws_radius = [  # vectorised for fast computation
        bulk_supercell[i]
        for i in np.where(
            bulk_supercell.lattice.get_all_distances(
                bulk_supercell.frac_coords, defect_frac_coords
            ).ravel()
            > np.max((wigner_seitz_radius - 1, 1))
        )[0]
    ]
    defect_sites_outside_wigner_radius = [  # vectorised for fast computation
        defect_supercell[i]
        for i in np.where(
            defect_supercell.lattice.get_all_distances(
                defect_supercell.frac_coords, defect_frac_coords
            ).ravel()
            > wigner_seitz_radius
        )[0]
    ]

    for species in bulk_supercell.composition.elements:  # divide and vectorise calc for efficiency
        bulk_species_outside_near_ws_coords = get_coords_and_idx_of_species(
            bulk_sites_outside_or_at_ws_radius, species.name
        )[0]
        defect_species_outside_ws_coords = get_coords_and_idx_of_species(
            defect_sites_outside_wigner_radius, species.name
        )[0]
        if (
            min(
                len(bulk_species_outside_near_ws_coords),
                len(defect_species_outside_ws_coords),
            )
            == 0
        ):
            continue  # if no sites of this species outside the WS radius, skip

        subset, superset = (  # supa-set
            (defect_species_outside_ws_coords, bulk_species_outside_near_ws_coords)
            if len(defect_species_outside_ws_coords) < len(bulk_species_outside_near_ws_coords)
            else (bulk_species_outside_near_ws_coords, defect_species_outside_ws_coords)
        )
        vecs, d_2 = pbc_shortest_vectors(bulk_supercell.lattice, subset, superset, return_d2=True)
        site_matches, _ = get_linear_assignment_solution(d_2)  # matching superset indices, of len(subset)
        matching_vecs = vecs[np.arange(len(site_matches)), site_matches]
        displacements = np.linalg.norm(matching_vecs, axis=1)
        far_from_defect_disps[species.name].extend(
            np.round(displacements[displacements > displacement_tol], 2)
        )

    if far_from_defect_large_disps := {
        specie: list
        for specie, list in far_from_defect_disps.items()
        if list and np.mean(list) > displacement_tol
    }:
        message = (
            f"Detected atoms far from the defect site (>{wigner_seitz_radius:.2f} Å) with major "
            f"displacements (>{displacement_tol} Å) in the defect supercell. This likely indicates a "
            f"mismatch between the bulk and defect supercell definitions (-> see troubleshooting docs) or "
            f"an unconverged supercell size, both of which could cause errors in parsing. The mean "
            f"displacement of the following species, at sites far from the determined defect position, "
            f"is >{displacement_tol} Å: {list(far_from_defect_large_disps.keys())}"
        )
        if warning == "verbose":
            message += f", with displacements (Å): {far_from_defect_large_disps}"
        if warning:
            warnings.warn(message)

        return False

    return True


def get_site_mapping_indices(
    struct1: Structure,
    struct2: Structure,
    species: SpeciesLike | None = None,
    allow_duplicates: bool = False,
    threshold: float = 2.0,
    dists_only: bool = False,
    anonymous: bool = False,
    ignored_species: list[str] | None = None,
    frac_coords: bool = True,
):
    """
    Get the site mapping indices between two structures (from ``struct1`` to
    ``struct2``), based on the fractional coordinates of the sites.

    The template structure may have a different species ordering to the
    ``input_structure``.

    NOTE: if ``frac_coords = True`` (default), this assumes that both
    structures have the same lattice definitions (i.e. that they match, and
    aren't rigidly translated/rotated with respect to each other), which is
    mostly the case unless we have a mismatching defect/bulk supercell (in
    which case the ``check_atom_mapping_far_from_defect`` warning should be
    thrown anyway during parsing).

    Args:
        struct1 (|Structure|):
            The input structure.
        struct2 (|Structure|):
            The template structure.
        species (str):
            If provided, only sites of this species will be considered when
            matching sites. Default is ``None`` (all species).
        allow_duplicates (bool):
            If ``True``, allow multiple sites in ``struct1`` to be matched to
            the same site in ``struct2``. Default is ``False``.
        threshold (float):
            If the distance between a pair of matched sites is larger than
            this, then a warning will be thrown. Default is 2.0 Å.
        dists_only (bool):
            Whether to return only the distances between matched sites, rather
            than a list of lists containing the distance, index in ``struct1``
            and index in ``struct2``. Default is ``False``.
        anonymous (bool):
            If ``True``, the species of the sites will not be considered when
            matching sites. Default is ``False`` (only matching species can be
            matched together).
        ignored_species (list[str]):
            A list of species to ignore when matching sites. Default is no
            species ignored.
        frac_coords (bool):
            Whether to match sites based on their fractional coordinate
            distances (i.e. assuming PBC with matching lattice definitions,
            using the lattice of ``struct1``)(default). If ``False``, instead
            matches sites based on distances between their Cartesian
            coordinates, with no consideration of PBC.

    Returns:
        list:
            A list of lists containing the distance, index in ``struct1`` and
            index in ``struct2`` for each matched site. If ``dists_only`` is
            ``True``, then only the distances between matched sites are
            returned.
    """

    def get_coords(site: PeriodicSite):
        return list(site.frac_coords) if frac_coords else list(site.coords)

    def get_distances(
        coords1: np.ndarray | list, coords2: np.ndarray | list, lattice: Lattice | None = None
    ):
        if frac_coords:
            assert lattice is not None, "Lattice needs to be given if frac_coords is True!"
            return lattice.get_all_distances(coords1, coords2)
        return all_distances(coords1, coords2)

    ## Generate a site matching table between the input and the template
    min_dist_with_index: list[tuple] = []
    s1_species_symbols = (
        [
            species.symbol
            for species in struct1.composition.elements
            if species.symbol not in (ignored_species or [])
        ]
        if not anonymous
        else [None]
    )

    for s1_species_symbol in s1_species_symbols:
        if species is not None and s1_species_symbol != species:
            continue
        # Build (struct1_index, coords) pairs for this species, preserving ``struct1`` order:
        species_input = [
            (i, get_coords(site))
            for i, site in enumerate(struct1)
            if (site.specie.symbol == s1_species_symbol or anonymous)
        ]
        input_coords = [coords for _, coords in species_input]
        species_s2_indices = [
            i for i, site in enumerate(struct2) if (site.specie.symbol == s1_species_symbol or anonymous)
        ]
        template_coords = [get_coords(struct2[i]) for i in species_s2_indices]

        dmat = (
            get_distances(input_coords, template_coords, lattice=struct1.lattice)
            if template_coords
            else None
        )

        if not allow_duplicates and dmat is not None:
            # Use linear assignment for order-independent optimal matching.
            # get_linear_assignment_solution returns (col_ind, total_cost), where col_ind[i] is the
            # template index assigned to input row i (requires n_rows <= n_cols). For n > m, transpose
            # the problem (assign each template to one input) and invert the mapping:
            if len(input_coords) <= len(template_coords):
                tmpl_col_indices, _ = get_linear_assignment_solution(dmat)
                input_to_template = dict(enumerate(tmpl_col_indices.tolist()))
            else:
                # dmat.T is (n_templates, n_inputs): each template row j is assigned input column
                # input_col_indices[j]. We need input_idx -> tmpl_idx for the loop below:
                input_col_indices, _ = get_linear_assignment_solution(dmat.T)
                input_to_template = {int(input_col_indices[j]): j for j in range(len(template_coords))}
        else:
            input_to_template = None

        for input_idx, (index, _) in enumerate(species_input):
            if dmat is None:
                min_dist_with_index.append((None, index) if dists_only else (None, index, None))
                continue

            if input_to_template is not None:
                if input_idx not in input_to_template:
                    # No unique template available (more inputs than templates for this species)
                    min_dist_with_index.append((None, index) if dists_only else (None, index, None))
                    continue
                tmpl_idx = input_to_template[input_idx]

            else:  # allow_duplicates=True: each input independently picks its closest template
                dists = dmat[input_idx]
                tmpl_idx = dists.argmin()

            current_dist = float(dmat[input_idx, tmpl_idx])
            # Map species-local template index (tmpl_idx) to global struct2 index (species_s2_indices):
            template_index = species_s2_indices[tmpl_idx]

            if current_dist > threshold:
                warnings.warn(
                    f"Large site displacement {current_dist:.2f} Å detected when matching atomic sites: "
                    f"{struct1[index]} -> {struct2[template_index]}."
                )

            min_dist_with_index.append(
                (current_dist, index) if dists_only else (current_dist, index, template_index)
            )

    if not min_dist_with_index:
        raise RuntimeError(
            f"No matching sites for species {species} found between the two structures!\n"
            f"Struct1 composition: {struct1.composition}, Struct2 composition: {struct2.composition}"
        )

    if dists_only and min_dist_with_index:  # sort by index in struct1:
        return [
            x[0]
            for x in sorted(min_dist_with_index, key=lambda x: x[1] if x[1] is not None else float("inf"))
        ]

    return min_dist_with_index


def reorder_s1_like_s2(s1_structure: Structure, s2_structure: Structure, threshold=5.0) -> Structure:
    """
    Reorder the atoms of a (relaxed) structure, s1, to match the ordering of
    the atoms in s2_structure.

    s1/s2 structures may have a different species orderings.

    Previously used to ensure correct site matching when pulling site
    potentials for the eFNV Kumagai correction, though no longer used for this
    purpose. If threshold is set to a low value, it will raise a warning if
    there is a large site displacement detected.

    NOTE: This assumes that both structures have the same lattice definitions
    (i.e. that they match, and aren't rigidly translated/rotated with respect
    to each other), which is mostly the case unless we have a mismatching
    defect/bulk supercell (in which case the
    ``check_atom_mapping_far_from_defect`` warning should be thrown anyway
    during parsing). Currently, this function is no longer used, but if it is
    reintroduced at any point, this point should be noted!

    Args:
        s1_structure (|Structure|):
            The input structure.
        s2_structure (|Structure|):
            The template structure.
        threshold (float):
            If the distance between a pair of matched sites is larger than
            this, then a warning will be thrown. Default is 5.0 Å.

    Returns:
        Structure:
            The reordered structure.
    """
    # Obtain site mapping between the initial_relax_structure and the unrelaxed structure
    mapping = get_site_mapping_indices(s2_structure, s1_structure, threshold=threshold)

    # Reorder s1_structure so that it matches the ordering of s2_structure
    reordered_sites = [s1_structure[tmp[2]] for tmp in mapping]

    # avoid warning about selective_dynamics properties (can happen if user explicitly set "T T T" (or
    # otherwise) for the bulk):
    warnings.filterwarnings("ignore", message="Not all sites have property")

    new_structure = Structure.from_sites(reordered_sites)

    if len(new_structure) != len(s1_structure):
        raise ValueError("Structure reordering failed: structures have different number of sites?")

    return new_structure


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
    kweights = np.array(vasprun.actual_kpoints_weights)
    if kweights.sum() != 1:
        kweights /= kweights.sum()

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
    from doped.vasp import DefectDictSet

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

        return nelect


def _get_bulk_supercell(defect_entry: DefectEntry):
    if hasattr(defect_entry, "bulk_supercell") and defect_entry.bulk_supercell:
        return defect_entry.bulk_supercell

    if (
        hasattr(defect_entry, "bulk_entry")
        and defect_entry.bulk_entry
        and hasattr(defect_entry.bulk_entry, "structure")
        and defect_entry.bulk_entry.structure
    ):
        return defect_entry.bulk_entry.structure

    return None


def _get_defect_supercell(defect_entry: DefectEntry):
    if hasattr(defect_entry, "defect_supercell") and defect_entry.defect_supercell:
        return defect_entry.defect_supercell

    if (
        hasattr(defect_entry, "sc_entry")
        and defect_entry.sc_entry
        and hasattr(defect_entry.sc_entry, "structure")
        and defect_entry.sc_entry.structure
    ):
        return defect_entry.sc_entry.structure

    return None


def _get_unrelaxed_defect_structure(defect_entry: DefectEntry, **kwargs) -> Structure | None:
    if (
        hasattr(defect_entry, "calculation_metadata")
        and defect_entry.calculation_metadata
        and "unrelaxed_defect_structure" in defect_entry.calculation_metadata
    ):
        return defect_entry.calculation_metadata["unrelaxed_defect_structure"]

    bulk_supercell = _get_bulk_supercell(defect_entry)

    if bulk_supercell is not None:  # reparse info:
        _update_defect_entry_structure_metadata(defect_entry, **kwargs)

    return defect_entry.calculation_metadata.get("unrelaxed_defect_structure")


def _get_defect_supercell_frac_coords(
    defect_entry: DefectEntry, relaxed=True
) -> np.ndarray | tuple[float, float, float] | None:
    sc_defect_frac_coords: np.ndarray | tuple[float, float, float] | None = (
        defect_entry.sc_defect_frac_coords
    )
    site = None

    if not relaxed:
        site = _get_defect_supercell_site(defect_entry, relaxed=False)
    if sc_defect_frac_coords is None and site is None:
        site = _get_defect_supercell_site(defect_entry)
    if site is not None:
        sc_defect_frac_coords = site.frac_coords

    return sc_defect_frac_coords


def _get_defect_supercell_site(defect_entry: DefectEntry, relaxed=True, **kwargs) -> PeriodicSite | None:
    def _return_defect_supercell_site(defect_entry: DefectEntry, relaxed=True):
        if relaxed or defect_entry.defect.defect_type == DefectType.Interstitial:
            # always final relaxed site for interstitials (note that "bulk_site" may be guessed initial
            # site if it is close enough to the final relaxed site):
            if site := getattr(defect_entry, "defect_supercell_site", None):
                return site

            if defect_entry.sc_defect_frac_coords is not None:
                return PeriodicSite(
                    defect_entry.defect.site.species,
                    defect_entry.sc_defect_frac_coords,
                    _get_defect_supercell(defect_entry).lattice,
                )

        # otherwise we use ``bulk_site``, for relaxed = False (vacancies & substitutions)
        if (
            hasattr(defect_entry, "calculation_metadata")
            and defect_entry.calculation_metadata
            and defect_entry.calculation_metadata.get("bulk_site")
        ):
            return defect_entry.calculation_metadata.get("bulk_site")

        return None

    if defect_supercell_site := _return_defect_supercell_site(defect_entry, relaxed=relaxed):
        return defect_supercell_site

    # otherwise need to reparse info:
    _update_defect_entry_structure_metadata(defect_entry, **kwargs)

    return _return_defect_supercell_site(defect_entry, relaxed=relaxed)


def _update_defect_entry_structure_metadata(defect_entry: DefectEntry, overwrite: bool = False, **kwargs):
    """
    Helper function to reparse the defect site information for a given
    |DefectEntry|, updating the relevant attributes and calculation metadata.

    Args:
        defect_entry (|DefectEntry|):
            The |DefectEntry| object for which to update the defect site
            information.
        overwrite (bool):
            Whether to overwrite existing |DefectEntry| attributes with the
            newly parsed values. Default is ``False`` (i.e. only update if the
            attributes are not already set).
        **kwargs:
            Keyword arguments to pass to ``get_equiv_frac_coords_in_primitive``
            (such as ``symprec``, ``dist_tol_factor``,
            ``fixed_symprec_and_dist_tol_factor``, ``verbose``) and/or
            |Defect| initialization (such as ``oxi_state``, ``multiplicity``,
            ``symprec``, ``dist_tol_factor``) in the
            ``defect_and_info_from_structures`` function.
    """
    from doped.analysis import defect_and_info_from_structures

    bulk_supercell = _get_bulk_supercell(defect_entry)
    defect_supercell = _get_defect_supercell(defect_entry)
    (
        defect,
        defect_site,
        defect_structure_metadata,
    ) = defect_and_info_from_structures(
        defect_supercell,
        bulk_supercell,
        _parameter_order_warn=False,
        **kwargs,  # pass any additional kwargs (e.g. oxidation state, multiplicity, etc.)
    )
    if not getattr(defect_entry, "calculation_metadata", None):
        defect_entry.calculation_metadata = {}

    # update any missing calculation_metadata:
    for k, v in defect_structure_metadata.items():
        if not defect_entry.calculation_metadata.get(k) or overwrite:
            defect_entry.calculation_metadata[k] = v

    for attr_name, value in {
        "defect": defect,
        "sc_defect_frac_coords": defect_site.frac_coords,  # _relaxed_ defect site
        "defect_supercell_site": defect_site,
        "defect_supercell": defect_supercell,
        "bulk_supercell": bulk_supercell,
    }.items():
        if getattr(defect_entry, attr_name, None) is None or overwrite:
            setattr(defect_entry, attr_name, value)


def _num_electrons_from_charge_state(structure: Structure, charge_state: int = 0) -> int:
    """
    Get the total number of electrons (including core electrons! -- so
    different to ``NELECT`` in VASP in most cases) for a given structure and
    charge state.

    Args:
        structure (|Structure|):
            The structure for which to get the total number of electrons.
        charge_state (int):
            The charge state of the system. Default is 0.

    Returns:
        int:
            The total number of electrons in the system, including core
            electrons.
    """
    total_Z = int(
        sum(Element(elt).Z * num for elt, num in structure.composition.get_el_amt_dict().items())
    )
    return int(total_Z + charge_state)


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
    if charge_state is None:
        num_electrons = get_nelect_from_vasprun(vasprun)
    else:
        num_electrons = _num_electrons_from_charge_state(vasprun.final_structure, charge_state)

    try:
        raw_magnetization = get_magnetization_from_vasprun(vasprun)
        # take the vector norm as the total magnetization (for NCL (SOC) / vector magnetization):
        magnetization = float(np.linalg.norm(raw_magnetization))

        # round to nearest possible value (even numbers for even-electron systems, odd for odd-electron):
        if num_electrons % 2 == 0:  # even-electron system, spin degeneracy = 1, 3, 5, ...
            magnetization = round(magnetization / 2) * 2  # nearest even number
        else:
            magnetization = round((magnetization - 1) / 2) * 2 + 1  # nearest odd number

        # spin multiplicity = 2S + 1 = 2(mag/2) + 1 = mag + 1 (where mag is in Bohr magnetons
        # i.e. number of electrons, as in VASP):
        return abs(magnetization) + 1

    except (RuntimeError, TypeError):  # NCL calculation without parsed projected magnetization:
        return _simple_spin_degeneracy_from_num_electrons(int(num_electrons))  # guess from charge


def _simple_spin_degeneracy_from_num_electrons(num_electrons: int = 0) -> int:
    """
    Get the spin degeneracy of a system from the total number of electrons,
    assuming simple singlet (S=0) behaviour for even-electron systems or
    doublet (S=1/2) behaviour for odd-electron systems.

    Spin multiplicity is equal to ``2S + 1``, so 1 for singlets (S = 0), 2 for
    doublets (S = 1/2), 3 for triplets (S = 1) etc.

    Args:
        num_electrons (int): The total number of electrons.

    Returns:
        int:
            The spin multiplicity assuming singlet or doublet behaviour.
    """
    return int(num_electrons % 2 + 1)


def total_charge_from_vasprun(vasprun: Vasprun) -> int | None:
    """
    Determine the total charge state of a system from the vasprun.

    This is VASP-specific; for Quantum ESPRESSO the charge is read directly
    from the ``PWxml`` object (``PWxml.total_charge``, parsed from the QE
    ``tot_charge`` XML field).

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
    with contextlib.suppress(Exception):  # determine neutral NELECT from vasprun & POTCARs:
        nelect = get_nelect_from_vasprun(vasprun)
        neutral_nelect = get_neutral_nelect_from_vasprun(vasprun)
        auto_charge = -1 * (nelect - neutral_nelect)

        if abs(auto_charge) >= 10:
            neutral_nelect = get_neutral_nelect_from_vasprun(vasprun, skip_potcar_init=True)
            auto_charge = -1 * (nelect - neutral_nelect)

    return auto_charge


def _get_bulk_locpot_dict(bulk_path, quiet=False, filename="LOCPOT"):
    bulk_locpot_path, multiple = _get_output_files_and_check_if_multiple(
        filename, bulk_path, dir_type="bulk", quiet=quiet
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


def _update_defect_entry_charge_corrections(defect_entry, charge_correction_type):
    meta = defect_entry.calculation_metadata[f"{charge_correction_type}_meta"]
    corr = (
        meta[f"{charge_correction_type}_electrostatic"]
        + meta[f"{charge_correction_type}_potential_alignment_correction"]
    )
    defect_entry.corrections.update({f"{charge_correction_type}_charge_correction": corr})


_vasp_file_parsing_action_dict = {
    "vasprun.xml": "parse the calculation energy and metadata.",
    "OUTCAR": "parse core levels and compute the Kumagai (eFNV) image charge correction.",
    "LOCPOT": "parse the electrostatic potential and compute the Freysoldt (FNV) charge correction.",
    ".cube": "parse electrostatic potentials (average/core) to compute charged defect corrections in QE",
}


def _multiple_files_warning(file_type, directory, chosen_filepath, action=None, dir_type="bulk"):
    filename = os.path.basename(chosen_filepath)
    if action is None:
        action = _vasp_file_parsing_action_dict[file_type]
    warnings.warn(
        f"Multiple `{file_type}` files found in {dir_type} directory: {directory}. Using {filename} to "
        f"{action}"
    )


def get_dimer_bonds(structure: Structure, rtol: float = 1.05) -> dict[str, list[float]]:
    """
    Get a dictionary of all homoionic (dimer) bonds in the structure.

    This function uses the ``get_homoionic_bonds`` and
    ``get_dimer_bond_length`` functions from ``shakenbreak`` to identify dimer
    bonds in the structure (where any pair of atoms of the same element with
    distance < ``rtol * get_dimer_bond_length(elt, elt)`` are considered a
    dimer bond), returning a dictionary of the site names and the dimer bond
    length.

    Args:
        structure (|Structure|): The structure to get the dimer bond lengths for.
        rtol (float):
            The relative tolerance to use for classifying bonds as dimer bonds,
            where distances < ``rtol * get_dimer_bond_length(elt, elt)`` are
            considered dimer bonds. Default is 1.05.

    Returns:
        dict[str, list[float]]:
            A dictionary of element names with values being sub-dictionaries of
            site names and their homoionic neighbours and distances (in Å)
            which are classified as dimer bonds.
            (e.g. {'O': {'O(1)': {'O(3)': '1.44 Å'}}})
    """
    from shakenbreak.analysis import get_homoionic_bonds
    from shakenbreak.distortions import get_dimer_bond_length

    dimer_bond_dict = {
        str(elt): get_homoionic_bonds(
            structure=structure,
            elements=str(elt),
            radius=rtol * get_dimer_bond_length(elt, elt),
            verbose=False,
        )
        for elt in structure.composition.elements
    }
    return {k: v for k, v in dimer_bond_dict.items() if v}


from ase.io.cube import read_cube_data
from pymatgen.core.units import Ry_to_eV
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.io.espresso.outputs.pwxml import PWxml
from pymatgen.io.vasp import VolumetricData
from scipy.ndimage import map_coordinates

BOHR_TO_ANGSTROM = 0.529177


class RunParser:
    def __new__(cls, code: Literal["vasp", "espresso"], **kwargs):
        code = code.lower()
        if code == "vasp":
            return RunParserVasp  # (**kwargs) #NOT IMPLEMENTED
        elif code == "espresso":
            return RunParserEspresso  # (**kwargs)
        else:
            raise ValueError(f"Unsupported code: {code}")


class RunParserEspresso:
    @classmethod
    def get_run(cls, espressorun_path: PathLike, parse_mag: bool = False, standardize=True, **kwargs):
        """
        Similar to get_vasprun but for espresso.

        if parse_projected_eigen = True: must provide filproj (for pwxml). (Use filproj = 'filproj' for projwfc.x
        if parse_dos: Must give fildos.
        """
        espressorun_path = str(espressorun_path)  # convert to string if Path object
        warnings.filterwarnings(
            "ignore", category=UnknownPotcarWarning
        )  # Ignore unknown POTCAR warnings when loading vasprun.xml
        # pymatgen assumes the default PBE with no way of changing this within get_vasprun())
        warnings.filterwarnings(
            "ignore", message="No POTCAR file with matching TITEL fields"
        )  # `message` only needs to match start of message
        default_kwargs = {"parse_dos": False, "exception_on_bad_xml": False}
        default_kwargs.update(kwargs)
        #TODO: Devise a test for working with projected eigenvalues: Currently untested with doped examples.
        #PWxml._parse_projected_eigen = partialmethod(parse_projected_eigen, parse_mag=parse_mag) #??? Never called in doped? PWxml already has a _parse_projected_eigen though it only accepts filproj.

        try:
            with warnings.catch_warnings(record=True) as w:

                # if standardize:
                #     vasprun = cls.standardized_computed_entry(find_archived_fname(espressorun_path),
                #                                             **default_kwargs)
                # else:
                vasprun = PWxml(find_archived_fname(espressorun_path), **default_kwargs)

                # PWxml does not initialize atomic states and kpoints_opt_props
                # see https://github.com/Griffin-Group/pymatgen-io-espresso/issues/27
                vasprun.atomic_states = None

                # if isinstance(vasprun.potcar_spec, list):
                #     vasprun.potcar_spec = cls.potcar_spec_fix(vasprun)
                # -----------------------------------
            for warning in w:
                if "XML is malformed" in str(warning.message):
                    warnings.warn(
                        f"espresso.xml file at {espressorun_path} is corrupted/incomplete. Attempting to "
                        f"continue parsing but may fail!"
                    )
                else:  # show warning, preserving original category:
                    warnings.warn(warning.message, category=warning.category)

        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"espresso.xml not found at {espressorun_path}. Needed for parsing calculation output!"
            ) from exc
        return vasprun

    @classmethod
    def _parse_run_and_poss_projwfc(
        cls,
        vr_path: PathLike,
        parse_projected_eigen: bool | None = None,
        output_path: PathLike | None = None,
        label: str = "bulk",
        parse_procar: bool = True,
    ):
        procar = None

        failed_eig_parsing_warning_message = (
            f"Could not parse eigenvalue data from vasprun.xml.gz files in {label} folder at {output_path}"
        )

        try:
            # Get run, parse_proj_eigen (if demanded), parse_eigen (if demanded) but definitely if bulk
            vr = cls.get_run(
                vr_path,
                parse_projected_eigen=bool(parse_projected_eigen),
                parse_eigen=(bool(parse_projected_eigen) or label == "bulk"),
            )  # vr.eigenvalues not needed for defects except for vr-only eigenvalue analysis

        except Exception as vr_exc:
            # Get run, don't parse_proj_eigen, parse_eigen if bulkrun.
            vr = cls.get_run(vr_path, parse_projected_eigen=False, parse_eigen=label == "bulk")
            failed_eig_parsing_warning_message += f", got error:\n{vr_exc}"

            # Parse from PROCAR if needed -> Goes to projwfc for espresso.
            if parse_procar:
                # But there might be multiple, so check.
                procar_path, multiple = _get_output_files_and_check_if_multiple("PROCAR", output_path)
                # Have the PROCAR? Now parse_projected_eigen if needed
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

    @classmethod
    def ensure_band_edges(cls, vasprun_obj, occu_tol=1e-8, backend="doped"):
        """
        Ensure that the Vasprun object has VBM, CBM, and band_gap set.
        """
        if backend == "pymatgen":
            vasprun_obj.occu_tol = occu_tol
            band_gap, cbm, vbm, _ = vasprun_obj.eigenvalue_band_properties
            vasprun_obj.vbm = vbm
            vasprun_obj.cbm = cbm
            vasprun_obj.band_gap = band_gap

        elif backend == "doped":
            from doped.utils.eigenvalues import band_edge_properties_from_vasprun

            if (
                not hasattr(vasprun_obj, "vbm")
                or vasprun_obj.vbm is None
                or not hasattr(vasprun_obj, "cbm")
                or vasprun_obj.cbm is None
                or not hasattr(vasprun_obj, "band_gap")
                or vasprun_obj.band_gap is None
            ):

                band_edge_prop = band_edge_properties_from_vasprun(vasprun_obj)

                if not band_edge_prop.is_metal:
                    vasprun_obj.vbm = band_edge_prop.vbm_info.as_dict()["energy"]
                    vasprun_obj.cbm = band_edge_prop.cbm_info.as_dict()["energy"]
                    vasprun_obj.band_gap = vasprun_obj.cbm - vasprun_obj.vbm
        else:
            raise ValueError("Use doped or pymatgen for finding band_gap")
        return vasprun_obj

    @classmethod
    def _get_cube_dict(cls, bulk_path, quiet=False):

        bulk_cube_path, multiple = _get_output_files_and_check_if_multiple(".cube", bulk_path, dir_type="bulk")

        bulk_cube = cls.get_cube(bulk_cube_path)
        return {str(k): bulk_cube.get_average_along_axis(k) for k in [0, 1, 2]}

    @classmethod
    def _get_bulk_site_potentials(
        cls, bulk_path: PathLike, quiet: bool = False, total_energy: list | float | None = None, beta: float = 0.5
    ):
        # try QE .cube first, then fall back to VASP LOCPOT:
        bulk_vol_data_path, multiple = _get_output_files_and_check_if_multiple(".cube", bulk_path)
        output_file = ".cube"
        if not os.path.exists(bulk_vol_data_path):
            bulk_vol_data_path, multiple = _get_output_files_and_check_if_multiple("LOCPOT", bulk_path)
            output_file = "LOCPOT"
        if multiple and not quiet:
            _multiple_files_warning(
                output_file,
                bulk_path,
                bulk_vol_data_path,
                dir_type="bulk",
            )
        return get_atomic_site_potentials(bulk_vol_data_path, beta=beta)


    @classmethod
    def get_cube(cls, cube_path: PathLike):
        """
        Read the ``LOCPOT(.gz)`` file as a ``pymatgen`` ``Locpot`` object.
        """
        cube_path = str(cube_path)  # convert to string if Path object

        try:
            cube = VolumetricData.from_cube(cube_path)

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Cube file not found at {cube_path}(.gz/.xz/.bz/.lzma). Needed for calculating the "
                f"Freysoldt (FNV) image charge correction!"
            ) from None
        return cube

    def _get_bulk_cube_dict(bulk_path, quiet=False, filename=".cube"):
        bulk_cube_path, multiple = _get_output_files_and_check_if_multiple(
            filename, bulk_path, dir_type="bulk", quiet=quiet
        )

        bulk_cube = RunParser("espresso").get_cube(bulk_cube_path)
        return {str(k): bulk_cube.get_average_along_axis(k) for k in [0, 1, 2]}

    @classmethod
    def _get_core_site_potentials(
        cls,
        cube_file=None,
        data=None,
        atoms=None,
        radius_bohr=1.5,
        n_points=500000,
        verbose=False,
    ):
        """
        Calculate spherical average potential at atomic sites from a .cube file
        or preloaded data.
        """
        BOHR_TO_ANGSTROM = 0.529177

        def spherical_average(pos, radius, data, cell, n_points=500000):
            """
            Compute spherical average of potential field around a point.
            """
            rand_dirs = np.random.normal(size=(n_points, 3))
            rand_dirs /= np.linalg.norm(rand_dirs, axis=1)[:, None]
            rand_radii = np.random.rand(n_points) ** (1 / 3) * radius
            sample_points = pos + rand_dirs * rand_radii[:, None]

            # Convert to fractional coordinates and then grid indices
            frac = np.linalg.solve(cell.T, sample_points.T).T
            frac %= 1.0
            grid_points = frac * (np.array(data.shape) - 1)

            # Interpolate potential values
            values = map_coordinates(data, grid_points.T, order=1, mode="wrap")
            return np.mean(values)

        # === Load data if a file path is provided ===
        if cube_file:
            data, atoms = read_cube_data(cube_file)
        elif data is None or atoms is None:
            raise ValueError("You must provide either `cube_file` or both `data` and `atoms`.")

        # === Prepare variables ===
        cell = atoms.get_cell()
        positions = atoms.get_positions()
        radius_ang = radius_bohr * BOHR_TO_ANGSTROM

        core_potentials = []

        for i, pos in enumerate(positions):
            avg_pot = spherical_average(pos, radius_ang, data, cell, n_points=n_points)
            core_potentials.append(avg_pot)
            if verbose:
                print(f"{atoms[i].symbol:<6}{i + 1:>6}{avg_pot:>30.6f}")

        core_dict = {
            "site_potentials": np.array(core_potentials) * Ry_to_eV,
            "atoms": atoms,
            "positions": positions,
        }

        return core_dict

    @classmethod
    # @fileread
    def standardized_computed_entry(
        cls, xml_file: PathLike = None, computed_entry: ComputedEntry = None, **kwargs
    ):
        """
        Return a computed entry with the standard formation enthalpy as total
        energy.
        """
        if xml_file:
            # print(xml_file, "\n")
            calc = PWxml(xml_file)
            computed_entry = calc.get_computed_entry(entry_id="")

        # print("COMPUTEDENTRY: ", dir(computed_entry))
        d_ = {
            "energy": cls._standardize_total_energy(computed_entry),
            "composition": computed_entry.composition,
            "entry_id": "",
            "correction": 0,  # pristine_calc.get_computed_entry(entry_id = "").correction
            # "structure": computed_entry.structure
        }

        # print(computed_entry.structure)
        ent = ComputedEntry.from_dict(d_)  # Computed entries list. Why twice?
        ent.structure = computed_entry.structure

        return ent

    @classmethod
    def _standardize_total_energy(cls, struct):
        """
        Hack for PWxml.

        PWxml puts energy as the formation energy. Might need to be changed if
        PWxml updates.
        """
        e_bulk = struct.energy
        composition = struct.composition

        comp_dict = composition.as_data_dict()["unit_cell_composition"]

        elements = [k.name for k in struct.elements]
        n_i = np.array(list(comp_dict.values()))
        u_i = np.array([cls._get_element_formation_energy(elem) for elem in elements])

        std_form_energy = (e_bulk - np.sum(n_i * u_i)) / np.sum(n_i)

        return std_form_energy

    @classmethod
    def _get_element_formation_energy(cls, elem, pseudo="pbe", root=Path(".")):

        elem_file = root / elem / f"{elem}_{pseudo}.xml"

        comp_entry = PWxml(elem_file).get_computed_entry(entry_id="")
        n_atoms = comp_entry.composition.as_data_dict()["unit_cell_composition"][elem]

        energy = comp_entry.energy

        en_per_atom = energy / n_atoms
        return en_per_atom

def get_atomic_site_potentials(volumetric_data_path: PathLike | VolumetricData, beta: float = 0.5):
    """
            Calculates atomic gaussian average site potential.

            cube_path:  cube path for the potential

            beta : Gaussian broadening factor at atomic sites (in bohr)

            Returns:
                 dict with keys:
                     atomic sites
                     Positions
                     site_potential
            """
    if isinstance(volumetric_data_path, VolumetricData):
        volumetric_data = volumetric_data_path
        is_cube = False
    elif str(volumetric_data_path).endswith('.cube'):
        volumetric_data = VolumetricData.from_cube(volumetric_data_path)
        is_cube = True
    else:
        volumetric_data = get_locpot(volumetric_data_path)
        is_cube = False

    nx, ny, nz = volumetric_data.dim
    lattice = volumetric_data.structure.lattice

    reci_latt = lattice.reciprocal_lattice
    # integer Miller indices along each reciprocal axis, FFT-ordered:
    nx_idx = np.roll(np.arange(-nx // 2, nx // 2, 1, dtype=int), int(nx // 2))
    ny_idx = np.roll(np.arange(-ny // 2, ny // 2, 1, dtype=int), int(ny // 2))
    nz_idx = np.roll(np.arange(-nz // 2, nz // 2, 1, dtype=int), int(nz // 2))

    Nx, Ny, Nz = np.meshgrid(nx_idx, ny_idx, nz_idx, indexing="ij")
    # G = n1*b1 + n2*b2 + n3*b3; compute |G|^2 using the reciprocal metric tensor to correctly handle
    # non-orthorhombic cells:
    recip_matrix = reci_latt.matrix  # rows are b1, b2, b3
    metric = recip_matrix @ recip_matrix.T  # G_ij = bi . bj
    g2 = (
            Nx ** 2 * metric[0, 0]
            + Ny ** 2 * metric[1, 1]
            + Nz ** 2 * metric[2, 2]
            + 2 * Nx * Ny * metric[0, 1]
            + 2 * Nx * Nz * metric[0, 2]
            + 2 * Ny * Nz * metric[1, 2]
    )
    if is_cube:  # QE cube: potential in Ry, beta given in bohr (atomic units)
        pot = volumetric_data.data["total"] * -Ry_to_eV
        beta_angstrom = beta * BOHR_TO_ANGSTROM
    else:  # VASP LOCPOT: potential already in eV, and beta given directly in angstroms
        pot = -volumetric_data.data["total"]
        beta_angstrom = beta

    v_G = np.fft.fftn(pot)
    v_G *= np.exp(-0.5 * (beta_angstrom ** 2) * g2)  # Gaussian broadening in reciprocal space
    v_R = np.real(np.fft.ifftn(v_G))

    v_R_atomic_sites = interpolate_potentials_at_atomic_sites(v_R, volumetric_data)

    sites = volumetric_data.structure.sites
    coords = np.array([site.coords for site in sites])

    efnv_plot_data_dict = {"positions": [], "site_potentials": [], "atoms": []}

    efnv_plot_data_dict["site_potentials"].extend(v_R_atomic_sites)
    efnv_plot_data_dict["positions"].extend(coords)
    efnv_plot_data_dict["atoms"].extend(site.specie.symbol for site in sites)

    return efnv_plot_data_dict




def interpolate_potentials_at_atomic_sites(
    smoothed_potential: np.ndarray,
    volumetric_data: VolumetricData,
):
    nx, ny, nz = volumetric_data.dim

    xpoints = np.linspace(0.0, 1.0, nx, endpoint=False)
    ypoints = np.linspace(0.0, 1.0, ny, endpoint=False)
    zpoints = np.linspace(0.0, 1.0, nz, endpoint=False)

    # pad the grid with periodic images so (cubic) interpolation works at cell boundaries:
    xpoints_padded = np.concatenate([xpoints[-1:] - 1.0, xpoints, xpoints[:1] + 1.0])
    ypoints_padded = np.concatenate([ypoints[-1:] - 1.0, ypoints, ypoints[:1] + 1.0])
    zpoints_padded = np.concatenate([zpoints[-1:] - 1.0, zpoints, zpoints[:1] + 1.0])

    padded = np.concatenate(
        [smoothed_potential[-1:, :, :], smoothed_potential, smoothed_potential[:1, :, :]], axis=0
    )
    padded = np.concatenate([padded[:, -1:, :], padded, padded[:, :1, :]], axis=1)
    padded = np.concatenate([padded[:, :, -1:], padded, padded[:, :, :1]], axis=2)

    # TODO: Will want to revisit this implementation; currently quite slow with cubic interpolation, but
    # dropping to linear significantly worsens accuracy...
    interpolator = RegularGridInterpolator(
        (xpoints_padded, ypoints_padded, zpoints_padded),
        padded,
        method="cubic",  # 'linear' is faster, but 'cubic' is more accurate for interpolation
        bounds_error=True,
    )
    frac_coords = np.mod(volumetric_data.structure.frac_coords, 1.0)

    return interpolator(frac_coords)


# ──────────────────────────────────────────────────────────────────────────
# QE convergence + relaxation input generation from a bare ``Structure``
# ──────────────────────────────────────────────────────────────────────────

import copy as _copy

from pymatgen.io.espresso.inputs.pwin import (
    AtomicPositionsCard,
    AtomicSpeciesCard,
    CellNamelist,
    CellParametersCard,
    ControlNamelist,
    ElectronsNamelist,
    IonsNamelist,
    KPointsCard,
    PWin,
    SystemNamelist,
)
from pymatgen.io.vasp.inputs import Kpoints as _Kpoints

# ``doped`` package directory — equivalent to ``doped.vasp.MODULE_DIR``, but
# computed locally to avoid a circular import (``doped.vasp`` imports from
# ``doped.generation``, which imports this module).
_DOPED_MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_QE_SSSP_CONVERGENCE_DEFAULTS: dict = loadfn(
    os.path.join(_DOPED_MODULE_DIR, "QE_sets/SSSP_Convergence_set.yaml")
)
_QE_SSSP_PSEUDO_FILENAMES: dict = {
    element: metadata["filename"]
    for element, metadata in loadfn(
        os.path.join(_DOPED_MODULE_DIR, "QE_sets/SSSP_1.3.0_PBE_efficiency.json")
    ).items()
}


class _GammaKPointsCard(KPointsCard):
    """``KPointsCard`` that emits an empty body for ``gamma`` (Γ-only) sampling."""

    def get_body(self, indent: str) -> str:
        if self.option == "gamma":
            return ""
        return super().get_body(indent)


def _kpoints_grid_from_reciprocal_density(structure: Structure, reciprocal_density: int) -> list[int]:
    """``[kx, ky, kz]`` Monkhorst-Pack grid at the given k-points-per-Å^-3 density."""
    kpoints_obj = _Kpoints.automatic_density_by_vol(structure, kppvol=reciprocal_density)
    return [int(k) for k in kpoints_obj.kpts[0]]


def _write_qe_pw_input(
    filepath: str,
    structure: Structure,
    namelist_settings: dict[str, dict],
    kpoints: list[int] | None,
    pseudo_map: dict[str, str] | None = None,
) -> None:
    """
    Write a QE ``pw.in`` for ``structure``.

    Args:
        filepath: Destination path for ``pw.in`` (parent dirs are created).
        structure: Structure to write.
        namelist_settings: ``{namelist: {key: value, ...}}`` for the QE
            ``control``/``system``/``electrons``/``ions``/``cell`` namelists.
        kpoints: ``[kx, ky, kz]`` Monkhorst-Pack grid, or ``None`` for
            Γ-only sampling.
        pseudo_map: ``{element: UPF filename}`` overrides on top of the SSSP
            defaults; missing elements fall back to ``"{element}.upf"``.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    unique_species = sorted(
        {str(el) for el in structure.species}, key=lambda s: Element(s).Z
    )
    resolved_pseudos = {
        sp: _QE_SSSP_PSEUDO_FILENAMES.get(sp, f"{sp}.upf") for sp in unique_species
    }
    resolved_pseudos.update(pseudo_map or {})

    fixed_cell_calcs = {"relax", "scf", "nscf", "bands"}
    calc_type = namelist_settings.get("control", {}).get("calculation", "vc-relax")

    namelist_classes = {
        "control": ControlNamelist,
        "system": SystemNamelist,
        "electrons": ElectronsNamelist,
        "ions": IonsNamelist,
        "cell": CellNamelist,
    }
    namelists: dict = {}
    for nl_name, nl_cls in namelist_classes.items():
        if nl_name not in namelist_settings:
            continue
        if nl_name == "cell" and calc_type in fixed_cell_calcs:
            continue
        namelists[nl_name] = nl_cls(namelist_settings[nl_name])

    if kpoints is None:
        k_points_card: KPointsCard = _GammaKPointsCard("gamma", [], [], [], [], [])
    else:
        kx, ky, kz = kpoints
        k_points_card = KPointsCard("automatic", [kx, ky, kz], [0, 0, 0], [], [], [])

    cards = {
        "atomic_species": AtomicSpeciesCard(
            None,
            unique_species,
            [float(Element(sp).atomic_mass) for sp in unique_species],
            [resolved_pseudos[sp] for sp in unique_species],
        ),
        "atomic_positions": AtomicPositionsCard(
            "angstrom",
            [site.species_string for site in structure],
            np.array([site.coords for site in structure]),
            None,
        ),
        "k_points": k_points_card,
        "cell_parameters": CellParametersCard(
            "angstrom",
            structure.lattice.matrix[0],
            structure.lattice.matrix[1],
            structure.lattice.matrix[2],
        ),
    }

    PWin(namelists, cards).to_file(filepath)


def _build_qe_base_settings(
    structure: Structure,
    pseudo_dir: str,
    is_metal: bool,
    user_control_settings: dict | None,
    user_system_settings: dict | None,
    user_electron_settings: dict | None,
) -> dict:
    """
    Build the per-structure base namelist dict from the SSSP convergence
    YAML defaults: sets ``ibrav=0``, ``nat``, ``ntyp``, ``pseudo_dir``,
    optional metallic smearing, and merges any user overrides.
    """
    base = _copy.deepcopy(_QE_SSSP_CONVERGENCE_DEFAULTS)
    base["control"]["pseudo_dir"] = pseudo_dir
    base["control"].update(user_control_settings or {})
    base["system"].update(user_system_settings or {})
    base["electrons"].update(user_electron_settings or {})

    base["system"]["ibrav"] = 0
    base["system"]["nat"] = len(structure)
    base["system"]["ntyp"] = len(set(structure.species))

    if is_metal:
        base["system"].setdefault("occupations", "smearing")
        base["system"].setdefault("smearing", "gaussian")
        base["system"].setdefault("degauss", 0.005)

    return base


def qe_convergence_setup_from_structure(
    structure: Structure,
    output_dir: PathLike = "QE_convergence",
    kpoint_density_range: tuple = (20, 200, 20),
    kpoint_sweep_ecutwfc: int | None = None,
    ecut_range: tuple = (20, 90, 10),
    ecut_sweep_kpoint_density: int = 100,
    is_metal: bool = False,
    pseudo_dir: str = "./pseudo_folder_name/",
    pseudo_map: dict | None = None,
    user_control_settings: dict | None = None,
    user_system_settings: dict | None = None,
    user_electron_settings: dict | None = None,
) -> dict[str, list[str]]:
    """
    Generate QE ``pw.in`` files for k-point and plane-wave cutoff
    (``ecutwfc``) convergence testing from a single ``pymatgen``
    ``Structure``.

    Elements are read from ``structure.species`` and matched against the
    bundled SSSP 1.3.0 PBE Efficiency library (``doped/QE_sets``); the
    resolved UPF filenames are written into the ``ATOMIC_SPECIES`` card of
    every ``pw.in``. ``pseudo_map`` can override individual entries or
    supply pseudos for elements absent from SSSP.

    Two sub-trees are written under ``output_dir``:

    - ``kpoint_converge/k<kx>_<ky>_<kz>/pw.in`` — ``ecutwfc`` held at
      ``kpoint_sweep_ecutwfc`` (or the set default if ``None``), k-grid
      swept over ``kpoint_density_range``. Duplicate grids produced by
      nearby densities are skipped.
    - ``ecut_convergence/ecutwfc_<N>/pw.in`` — k-grid held at
      ``ecut_sweep_kpoint_density``, ``ecutwfc`` swept over ``ecut_range``.
      ``ecutrho`` is left at the set default.

    After running these and choosing converged values, call
    :func:`qe_relax_setup_from_structure` (with those converged values
    and, ideally, the relaxed structure) to write the final ``vc-relax``.

    Args:
        structure: Input structure (no MP lookup is performed).
        output_dir: Root folder for the two sub-trees. Default
            ``"QE_convergence"``.
        kpoint_density_range: ``(min, max, step)`` reciprocal k-point
            density (Å^-3) sweep for the k-grid test (``max`` exclusive,
            matching ``range``). Default ``(20, 200, 20)``.
        kpoint_sweep_ecutwfc: ``ecutwfc`` (Ry) held fixed while the
            k-grid is swept. ``None`` (default) keeps the YAML set default
            (60 Ry).
        ecut_range: ``(min, max, step)`` ``ecutwfc`` (Ry) sweep for the
            cutoff test (``max`` inclusive). Default ``(20, 90, 10)``.
        ecut_sweep_kpoint_density: k-point density (Å^-3) held fixed
            while ``ecutwfc`` is swept. Default 100.
        is_metal: If ``True``, set ``occupations='smearing'``,
            ``smearing='gaussian'``, ``degauss=0.005`` in ``&SYSTEM``.
        pseudo_dir: Path written to ``&CONTROL.pseudo_dir``.
        pseudo_map: ``{element: UPF filename}`` overrides on top of SSSP.
        user_control_settings, user_system_settings, user_electron_settings:
            Per-namelist overrides merged on top of the YAML defaults.

    Returns:
        ``{"kpoint_converge": [...], "ecut_convergence": [...]}`` listing
        every ``pw.in`` path written.
    """
    base = _build_qe_base_settings(
        structure,
        pseudo_dir,
        is_metal,
        user_control_settings,
        user_system_settings,
        user_electron_settings,
    )

    written: dict[str, list[str]] = {"kpoint_converge": [], "ecut_convergence": []}

    # ── k-point convergence: vary k-grid at a fixed ecutwfc ──
    kp_min, kp_max, kp_step = kpoint_density_range
    kpoint_scf = _copy.deepcopy(base)
    kpoint_scf["control"]["calculation"] = "scf"
    if kpoint_sweep_ecutwfc is not None:
        kpoint_scf["system"]["ecutwfc"] = kpoint_sweep_ecutwfc
    seen_kgrids: set[tuple[int, int, int]] = set()
    for density in range(kp_min, kp_max, kp_step):
        kgrid = _kpoints_grid_from_reciprocal_density(structure, density)
        kgrid_tuple = (kgrid[0], kgrid[1], kgrid[2])
        if kgrid_tuple in seen_kgrids:
            continue
        seen_kgrids.add(kgrid_tuple)
        kname = "k" + ("_" * (kgrid[0] // 10)) + ",".join(str(k) for k in kgrid)
        filepath = os.path.join(str(output_dir), "kpoint_converge", kname, "pw.in")
        _write_qe_pw_input(
            filepath=filepath,
            structure=structure,
            namelist_settings=kpoint_scf,
            kpoints=kgrid,
            pseudo_map=pseudo_map,
        )
        written["kpoint_converge"].append(filepath)

    # ── ecutwfc convergence: vary ecutwfc at a fixed k-grid ──
    ecut_kgrid = _kpoints_grid_from_reciprocal_density(structure, ecut_sweep_kpoint_density)
    ecut_min, ecut_max, ecut_step = ecut_range
    for ecut in range(ecut_min, ecut_max + 1, ecut_step):
        ecut_scf = _copy.deepcopy(base)
        ecut_scf["control"]["calculation"] = "scf"
        ecut_scf["system"]["ecutwfc"] = ecut
        filepath = os.path.join(str(output_dir), "ecut_convergence", f"ecutwfc_{ecut}", "pw.in")
        _write_qe_pw_input(
            filepath=filepath,
            structure=structure,
            namelist_settings=ecut_scf,
            kpoints=ecut_kgrid,
            pseudo_map=pseudo_map,
        )
        written["ecut_convergence"].append(filepath)

    return written


def qe_relax_setup_from_structure(
    structure: Structure,
    ecutwfc: int,
    kpoint_density: int,
    output_dir: PathLike = "QE_relax",
    ecutrho: int | None = None,
    calculation: str = "vc-relax",
    is_metal: bool = False,
    pseudo_dir: str = "./pseudo_folder_name/",
    pseudo_map: dict | None = None,
    user_control_settings: dict | None = None,
    user_system_settings: dict | None = None,
    user_electron_settings: dict | None = None,
) -> str:
    """
    Generate a QE ``pw.in`` for a final relaxation, using the converged
    ``ecutwfc`` and k-point density obtained from
    :func:`qe_convergence_setup_from_structure`.

    Pass the *relaxed* (or best-current) structure here once convergence
    is done — element identification and pseudopotential lookup use the
    same SSSP 1.3.0 PBE Efficiency map as the convergence helper.

    Args:
        structure: Structure to relax (ideally the one returned by the
            converged SCF/relax workflow).
        ecutwfc: Converged plane-wave cutoff (Ry) from the ``ecutwfc``
            sweep. Required.
        kpoint_density: Converged reciprocal k-point density (Å^-3) from
            the k-grid sweep. Required.
        output_dir: Folder to write ``pw.in`` into. Default ``"QE_relax"``.
        ecutrho: Optional ``ecutrho`` (Ry); ``None`` keeps the set default.
        calculation: ``"vc-relax"`` (default, full cell+ions) or
            ``"relax"`` (ions only, fixed cell).
        is_metal: If ``True``, set ``occupations='smearing'``,
            ``smearing='gaussian'``, ``degauss=0.005`` in ``&SYSTEM``.
        pseudo_dir: Path written to ``&CONTROL.pseudo_dir``.
        pseudo_map: ``{element: UPF filename}`` overrides on top of SSSP.
        user_control_settings, user_system_settings, user_electron_settings:
            Per-namelist overrides merged on top of the YAML defaults.

    Returns:
        Path of the written ``pw.in``.
    """
    base = _build_qe_base_settings(
        structure,
        pseudo_dir,
        is_metal,
        user_control_settings,
        user_system_settings,
        user_electron_settings,
    )
    base["control"]["calculation"] = calculation
    base["system"]["ecutwfc"] = ecutwfc
    if ecutrho is not None:
        base["system"]["ecutrho"] = ecutrho

    kgrid = _kpoints_grid_from_reciprocal_density(structure, kpoint_density)
    filepath = os.path.join(str(output_dir), "pw.in")
    _write_qe_pw_input(
        filepath=filepath,
        structure=structure,
        namelist_settings=base,
        kpoints=kgrid,
        pseudo_map=pseudo_map,
    )
    return filepath

