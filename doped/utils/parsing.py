"""
Helper functions for parsing defect supercell calculations.
"""

import contextlib
import itertools
import os
import re
import warnings
from copy import deepcopy
from functools import lru_cache, partialmethod
from xml.etree.ElementTree import Element as XML_Element

import numpy as np
from monty.io import reverse_readfile
from monty.serialization import loadfn
from pymatgen.analysis.defects.core import DefectType
from pymatgen.analysis.structure_matcher import get_linear_assignment_solution, pbc_shortest_vectors
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Composition, Lattice, PeriodicSite, Structure
from pymatgen.electronic_structure.core import Spin
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.inputs import POTCAR_STATS_PATH, UnknownPotcarWarning
from pymatgen.io.vasp.outputs import Locpot, Outcar, Procar, Vasprun, _parse_vasp_array
from pymatgen.util.typing import PathLike, SpeciesLike

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
    Parse the projected eigenvalues from a ``Vasprun`` object (used during
    initialisation), but excluding the projected magnetization for efficiency.

    Note that following SK's PRs to ``pymatgen`` (#4359, #4360), parsing of
    projected eigenvalues adds minimal additional cost to Vasprun parsing
    (~1-5%), while parsing of projected magnetization can add ~30% cost.

    This is a modified version of ``_parse_projected_eigen`` from
    ``pymatgen.io.vasp.outputs.Vasprun``, which allows skipping of projected
    magnetization parsing in order to expedite parsing in ``doped``, as well as
    some small adjustments to maximise efficiency.

    Args:
        elem (Element):
            The XML element to parse, with projected eigenvalues/magnetization.
        parse_mag (bool):
            Whether to parse the projected magnetization. Default is ``True``.

    Returns:
        Tuple[Dict[Spin, np.ndarray], Optional[np.ndarray]]:
            A dictionary of projected eigenvalues for each spin channel
            (up/down), and the projected magnetization (if parsed).
    """
    root = elem.find("array/set")
    proj_eigen = {}
    sets = root.findall("set")  # type: ignore[union-attr]

    for s in sets:
        spin = int(re.match(r"spin(\d+)", s.attrib["comment"])[1])  # type: ignore[index]
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
    Read the ``vasprun.xml(.gz)`` file as a ``pymatgen`` ``Vasprun`` object.
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
    Read the ``OUTCAR(.gz)`` file as a ``pymatgen`` ``Outcar`` object.
    """
    outcar_path = _get_outcar_path(outcar_path)
    return Outcar(outcar_path)


def get_core_potentials_from_outcar(
    outcar_path: PathLike, dir_type: str = "", total_energy: list | float | None = None
):
    """
    Get the core potentials from the OUTCAR file, which are needed for the
    Kumagai-Oba (eFNV) finite-size correction.

    This parser skips the full ``pymatgen`` ``Outcar`` initialisation/parsing,
    to expedite parsing and make it more robust (doesn't fail if ``OUTCAR`` is
    incomplete, as long as it has the core potentials information).

    Args:
        outcar_path (PathLike):
            The path to the OUTCAR file.
        dir_type (str):
            The type of directory the OUTCAR is in (e.g. ``bulk`` or
            ``defect``) for informative error messages.
        total_energy (Optional[Union[list, float]]):
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

    Input outcar is either a path or a ``pymatgen`` ``Outcar`` object
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
    Read the ``PROCAR(.gz)`` file as a ``pymatgen`` ``Procar`` object.

    Previously, ``pymatgen`` ``Procar`` parsing did not support SOC
    calculations, however this was updated in
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
    output_file: PathLike = "vasprun.xml", path: PathLike = "."
) -> tuple[PathLike, bool]:
    """
    Search for all files with filenames matching ``output_file``, case-
    insensitive.

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
        return (output_path, True) if len(output_files) > 1 else (output_path, False)
    return (
        os.path.join(path, output_file),
        False,
    )  # so `get_X()` will raise an informative FileNotFoundError


def get_defect_type_and_composition_diff(
    bulk: Structure | Composition, defect: Structure | Composition
) -> tuple[str, dict]:
    """
    Get the difference in composition between a bulk structure and a defect
    structure.

    Args:
        bulk (Union[Structure, Composition]):
            The bulk structure or composition.
        defect (Union[Structure, Composition]):
            The defect structure or composition.

    Returns:
        tuple[str, Dict[str, int]]:
            The defect type (``interstitial``, ``vacancy``, ``substitution`` or
            ``complex``) and the composition difference between the bulk and
            defect structures as a dictionary.
    """
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
        defect_type = "complex"

    if len(composition_diff) > 5:  # likely a mistake, warn user:
        warnings.warn(
            f"The composition difference between the bulk ({bulk_comp_dict}) and defect "
            f"({defect_comp_dict}) structures is quite large, suggesting either a large complex defect "
            f"or a mistake in the inputs. Beware!"
        )

    return defect_type, composition_diff


def get_defect_type_site_idxs_and_unrelaxed_structure(
    bulk_supercell: Structure,
    defect_supercell: Structure,
    site_tol: float | None = None,  # TODO: Change to 0.5 and add complex defect handling
    abs_tol: bool = False,
) -> tuple[str, int | None, int | None, Structure]:
    """
    Get the defect type, site (indices in the bulk and defect supercells) and
    unrelaxed structure, where 'unrelaxed structure' corresponds to the
    pristine defect supercell structure for vacancies / substitutions (with no
    relaxation), and the pristine bulk structure with the `final` relaxed
    interstitial site for interstitials.

    Args:
        bulk_supercell (Structure):
            The bulk supercell structure.
        defect_supercell (Structure):
            The defect supercell structure.
        site_tol (float | None):
            The (fractional) tolerance for matching sites between the defect
            and bulk structures. If ``abs_tol`` is ``False`` (default), then
            this value multiplied by the shortest bond length in the bulk
            structure will be used as the distance threshold for matching,
            otherwise the value is used directly (as a length in Å).
            If set to ``None``, the defect is assumed to be a point defect, and
            the largest site mismatch is assigned as the defect site.
            Default is 0.5 (i.e. half the shortest bond length in the bulk
            structure).
        abs_tol (bool):
            Whether to use ``site_tol`` as an absolute distance tolerance (in
            Å) instead of a fractional tolerance (in terms of the shortest bond
            length in the structure). Default is ``False``.

    Returns:
        defect_type (str):
            The type of defect as a string (``interstitial``, ``vacancy`` or
            ``substitution``).
        bulk_site_idx (int):
            Index of the site in the bulk structure that corresponds to the
            defect site in the defect structure.
        defect_site_idx (int):
            Index of the defect site in the defect structure.
        unrelaxed_defect_structure (Structure):
            Pristine defect supercell structure for vacancies/substitutions
            (i.e. pristine bulk with unrelaxed vacancy/substitution), or the
            pristine bulk structure with the `final` relaxed interstitial site
            for interstitials.
    """

    def process_substitution(bulk_supercell, defect_supercell, composition_diff, site_dist_tol):
        old_species = _get_species_from_composition_diff(composition_diff, -1)
        new_species = _get_species_from_composition_diff(composition_diff, 1)

        bulk_new_species_coords, bulk_new_species_indices = get_coords_and_idx_of_species(
            bulk_supercell, new_species
        )
        defect_new_species_coords, defect_new_species_indices = get_coords_and_idx_of_species(
            defect_supercell, new_species
        )

        # Get the coords and site index of the defect that was used in the calculation
        if bulk_new_species_coords.size > 0:  # intrinsic substitution
            site_mapping = _get_site_mapping_from_coords_and_indices(
                bulk_new_species_coords,
                defect_new_species_coords,
                lattice=bulk_supercell.lattice,
                s1_indices=bulk_new_species_indices,
                s2_indices=defect_new_species_indices,
            )
            defect_site_mappings = [
                mapping
                for mapping in site_mapping
                if mapping[0] is None or (site_dist_tol is not None and mapping[0] > site_dist_tol)
            ]  # TODO: Handle multiple matches for complexes...
            defect_site_idx = defect_site_mappings[0][-1]

        else:  # extrinsic substitution
            defect_site_idx = next(iter(defect_new_species_indices))

        # now find the closest old_species site in the bulk structure to the defect site
        bulk_old_species_coords, bulk_old_species_idx = get_coords_and_idx_of_species(
            bulk_supercell, old_species
        )
        defect_site_coords = defect_supercell[defect_site_idx].frac_coords  # frac coords of defect site
        _bulk_coords, bulk_site_arg_idx = find_nearest_coords(
            bulk_old_species_coords,
            defect_site_coords,
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

    def process_vacancy(bulk_supercell, defect_supercell, composition_diff, site_dist_tol):
        old_species = _get_species_from_composition_diff(composition_diff, -1)
        bulk_old_species_coords, bulk_old_species_indices = get_coords_and_idx_of_species(
            bulk_supercell, old_species
        )
        defect_old_species_coords, defect_old_species_indices = get_coords_and_idx_of_species(
            defect_supercell, old_species
        )

        site_mapping = _get_site_mapping_from_coords_and_indices(
            bulk_old_species_coords,
            defect_old_species_coords,
            lattice=bulk_supercell.lattice,
            s1_indices=bulk_old_species_indices,
            s2_indices=defect_old_species_indices,
        )
        defect_site_mappings = [
            mapping
            for mapping in site_mapping
            if mapping[0] is None or (site_dist_tol is not None and mapping[0] > site_dist_tol)
        ]  # TODO: Handle multiple matches for complexes...
        bulk_site_idx, defect_site_idx = defect_site_mappings[0][1], None
        unrelaxed_defect_structure = _create_unrelaxed_defect_structure(
            bulk_supercell,
            bulk_site_idx=bulk_site_idx,
        )
        return bulk_site_idx, defect_site_idx, unrelaxed_defect_structure

    def process_interstitial(bulk_supercell, defect_supercell, composition_diff, site_dist_tol):
        new_species = _get_species_from_composition_diff(composition_diff, 1)

        bulk_new_species_coords, bulk_new_species_indices = get_coords_and_idx_of_species(
            bulk_supercell, new_species
        )
        defect_new_species_coords, defect_new_species_indices = get_coords_and_idx_of_species(
            defect_supercell, new_species
        )

        if bulk_new_species_coords.size > 0:  # intrinsic interstitial
            site_mapping = _get_site_mapping_from_coords_and_indices(
                bulk_new_species_coords,
                defect_new_species_coords,
                lattice=bulk_supercell.lattice,
                s1_indices=bulk_new_species_indices,
                s2_indices=defect_new_species_indices,
            )
            defect_site_mappings = [
                mapping
                for mapping in site_mapping
                if mapping[0] is None or (site_dist_tol is not None and mapping[0] > site_dist_tol)
            ]  # TODO: Handle multiple matches for complexes...
            defect_site_idx = defect_site_mappings[0][-1]

        else:  # extrinsic interstitial
            defect_site_idx = next(iter(defect_new_species_indices))

        defect_site_coords = defect_supercell[defect_site_idx].frac_coords  # frac coords of defect site
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

    from doped.utils.supercells import min_dist

    bulk_bond_length = max(min_dist(bulk_supercell), 1)
    site_dist_tol = site_tol if site_tol is None or abs_tol else site_tol * bulk_bond_length

    try:
        defect_type, comp_diff = get_defect_type_and_composition_diff(bulk_supercell, defect_supercell)
    except RuntimeError as exc:
        raise ValueError(
            "Could not identify defect type from number of sites in structure: "
            f"{len(bulk_supercell)} in bulk vs. {len(defect_supercell)} in defect?"
        ) from exc

    if site_tol is None and defect_type == "complex":
        raise ValueError(
            f"Based on the composition difference between defect and bulk structures ({comp_diff}), "
            f"the defect is a complex defect, but ``site_tol`` is set to ``None`` which enforces the "
            f"assumption of a point defect. Please set ``site_tol`` to allow parsing of complex defect "
            f"sites."
        )

    return (
        defect_type,
        *handlers[defect_type](bulk_supercell, defect_supercell, comp_diff, site_dist_tol),
    )


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
    site: PeriodicSite | np.ndarray[float], structure: Structure, anonymous: bool = False, tol: float = 0.5
) -> PeriodicSite:
    """
    Get the (closest) matching ``PeriodicSite`` in ``structure`` for the input
    ``site``, which can be a ``PeriodicSite`` or fractional coordinates.

    If the closest matching site in ``structure`` is > ``tol`` Å (0.5 Å by
    default) away from the input ``site`` coordinates, an error is raised.

    Automatically accounts for possible differences in assigned oxidation
    states, site property dicts etc.

    Args:
        site (PeriodicSite | np.ndarray[float]):
            The site for which to find the closest matching site in
            ``structure``, either as a ``PeriodicSite`` or fractional
            coordinates array. If fractional coordinates, then ``anonymous``
            is set to ``True``.
        structure (Structure):
            The structure in which to search for matching sites to ``site``.
        anonymous (bool):
            Whether to use anonymous matching, allowing different
            species/elements to match each other (i.e. just matching based on
            coordinates). Default is ``False`` if ``site`` is a
            ``PeriodicSite``, and ``True`` if ``site`` is fractional
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
        candidate_frac_coords (Union[list, np.ndarray]):
            Fractional coordinates (typically from a bulk supercell), to find
            the nearest coordinates to ``target_frac_coords``.
        target_frac_coords (Union[list, np.ndarray]):
            The target coordinates to find the nearest coordinates to in
            ``candidate_frac_coords``.
        lattice (Lattice):
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
        frac_coords1 (Union[list, np.ndarray]):
            First set of fractional coordinates.
        frac_coords2 (Union[list, np.ndarray]):
            Second set of fractional coordinates.
        lattice (Lattice):
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
        bulk_supercell (Structure):
            The bulk supercell structure.
        frac_coords (Union[list, np.ndarray]):
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
        lattice (Union[Structure,Lattice]):
            The lattice of the structure (either a ``pymatgen`` ``Structure``
            or ``Lattice`` object).

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
    bulk_supercell: Structure,
    defect_supercell: Structure,
    defect_coords: np.ndarray[float],
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
        bulk_supercell (Structure):
            The bulk structure.
        defect_supercell (Structure):
            The defect structure.
        defect_coords (np.ndarray[float]):
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

        displacement_dists = np.min(
            bulk_supercell.lattice.get_all_distances(
                defect_species_outside_ws_coords, bulk_species_outside_near_ws_coords
            ),
            axis=1,
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


def _get_site_mapping_from_coords_and_indices(
    s1_coords: np.ndarray[float],
    s2_coords: np.ndarray[float],
    s1_indices: np.ndarray[int] | None = None,
    s2_indices: np.ndarray[int] | None = None,
    lattice: Lattice | None = None,
) -> list[tuple[float | None, int, int]]:
    """
    Get the site mapping between two sets of coordinates and indices, based on
    the shortest distances between sites.

    The coordinates are treated as fractional coordinates if ``lattice`` is
    provided, otherwise they are treated as Cartesian coordinates.

    Args:
        s1_coords (np.ndarray[float]):
            The fractional coordinates of the first set of sites.
        s2_coords (np.ndarray[float]):
            The fractional coordinates of the second set of sites.
        s1_indices (np.ndarray[int] | None):
            The indices of the first set of sites. If ``None``, the indices are
            assumed to be the range of the number of sites in ``s1_coords``.
        s2_indices (np.ndarray[int] | None):
            The indices of the second set of sites. If ``None``, the indices are
            assumed to be the range of the number of sites in ``s2_coords``.
        lattice (Lattice | None):
            The lattice of the structures. If ``None``, the identity matrix is
            used (corresponding to the assumption that the input coordinates
            are Cartesian).

    Returns:
        list:
            A list of lists containing the distance, index from ``s1_indices``
            and index from ``s2_indices`` for each matched site.
    """
    lattice = lattice or Lattice(np.eye(3))
    if s1_indices is None:
        s1_indices = np.arange(len(s1_coords))
    if s2_indices is None:
        s2_indices = np.arange(len(s2_coords))

    s1_is_subset = len(s1_coords) < len(s2_coords)
    subset_fcoords, subset_indices = (s1_coords, s1_indices) if s1_is_subset else (s2_coords, s2_indices)
    superset_fcoords, superset_indices = (
        (s2_coords, s2_indices) if s1_is_subset else (s1_coords, s1_indices)
    )
    _vecs, d_2 = pbc_shortest_vectors(lattice, subset_fcoords, superset_fcoords, return_d2=True)
    site_matches = LinearAssignment(d_2).solution  # matching superset indices, of len(subset)
    min_dists = np.min(np.sqrt(d_2), axis=1)
    superset_site_indices = [superset_indices[i] for i in site_matches]
    site_mapping = [
        (min_dists[i], subset_indices[i], superset_site_indices[i]) for i in range(len(min_dists))
    ]
    for missing_index in set(range(len(superset_fcoords))) - set(site_matches):
        site_mapping.append((None, None, superset_indices[missing_index]))  # unmatched sites

    if not s1_is_subset:  # swap tuple order, to match (dist, s1_index, s2_index)
        site_mapping = [(dist, index2, index1) for dist, index1, index2 in site_mapping]

    return site_mapping


def get_site_mappings(
    struct1: Structure,
    struct2: Structure,
    species: SpeciesLike | None = None,
    allow_duplicates: bool = False,
    threshold: float = 2.0,
    anonymous: bool = False,
    ignored_species: list[str] | None = None,
) -> list[tuple[float | None, int, int]]:
    """
    Get the site mappings between two structures (from ``struct1`` to
    ``struct2``), based on the shortest distances between sites.

    The two structures may have different species orderings.

    NOTE: This assumes that both structures have the same lattice definitions
    (i.e. that they match, and aren't rigidly translated/rotated with respect
    to each other), which is mostly the case unless we have a mismatching
    defect/bulk supercell (in which case the
    ``check_atom_mapping_far_from_defect`` warning should be thrown anyway
    during parsing).

    Args:
        struct1 (Structure):
            The input structure.
        struct2 (Structure):
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
        anonymous (bool):
            If ``True``, the species of the sites will not be considered when
            matching sites. Default is ``False`` (only matching species can be
            matched together).
        ignored_species (list[str]):
            A list of species to ignore when matching sites. Default is no
            species ignored.

    Returns:
        list:
            A list of lists containing the distance, index in ``struct1`` and
            index in ``struct2`` for each matched site.
    """
    # Generate a site matching table between the input and the template
    min_dist_with_index = []
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
        if species is not None and s1_species_symbol != species and not anonymous:
            continue

        if anonymous:
            s1_fcoords, s1_indices = struct1.frac_coords, np.arange(len(struct1))
            s2_fcoords, s2_indices = struct2.frac_coords, np.arange(len(struct2))
        else:
            s1_fcoords, s1_indices = get_coords_and_idx_of_species(struct1, s1_species_symbol)
            s2_fcoords, s2_indices = get_coords_and_idx_of_species(struct2, s1_species_symbol)

        if not allow_duplicates:
            min_dist_with_index.extend(
                _get_site_mapping_from_coords_and_indices(
                    s1_fcoords,
                    s2_fcoords,
                    lattice=struct1.lattice,
                    s1_indices=s1_indices,
                    s2_indices=s2_indices,
                )
            )
            continue

        dmat = struct1.lattice.get_all_distances(s1_fcoords, s2_fcoords)
        for i, index in enumerate(s1_indices):
            dists = dmat[i]
            min_dist_idx = dists.argmin()
            current_dist = dists[min_dist_idx]
            s2_index = s2_indices[min_dist_idx]
            min_dist_with_index.append((current_dist, index, s2_index))

            if current_dist > threshold:
                warnings.warn(
                    f"Large site displacement {current_dist:.2f} Å detected when matching atomic sites: "
                    f"{struct1[index]} -> {struct2[s2_index]}."
                )

    if not min_dist_with_index:
        raise RuntimeError(
            f"No matching sites for species {species} found between the two structures!\n"
            f"Struct1 composition: {struct1.composition}, Struct2 composition: {struct2.composition}"
        )

    return min_dist_with_index


def reorder_s2_like_s1(s1_structure: Structure, s2_structure: Structure, threshold=5.0) -> Structure:
    """
    Reorder the atoms of a (relaxed) structure, ``s2_structure``, to match the
    ordering of the atoms in ``s1_structure``.

    s1/s2 structures may have a different species orderings.

    NOTE: This assumes that both structures have the same lattice definitions
    (i.e. that they match, and aren't rigidly translated/rotated with respect
    to each other), which is mostly the case unless we have a mismatching
    defect/bulk supercell (in which case the
    ``check_atom_mapping_far_from_defect`` warning should be thrown anyway
    during parsing).

    Args:
        s1_structure (Structure):
            The template structure.
        s2_structure (Structure):
            The structure to reorder, to match ``s1_structure``.
        threshold (float):
            If the distance between a pair of matched sites is larger than
            this value in Å, then a warning will be thrown. Default is 5.0 Å.

    Returns:
        Structure:
            ``s2_structure`` reordered to match ``s1_structure``.
    """
    # This function was previously used to ensure correct site matching when pulling site potentials for
    # the eFNV Kumagai correction, though no longer used for this purpose. If threshold is set to a low
    # value, it will raise a warning if there is a large site displacement detected.
    if len(s2_structure) != len(s1_structure):
        raise ValueError("Structure reordering not possible, structures have different number of sites.")

    # Obtain site mapping between the initial_relax_structure and the unrelaxed structure
    mapping = get_site_mappings(s1_structure, s2_structure, threshold=threshold)
    mapping = sorted(mapping, key=lambda x: x[1])  # sort by s1 index (to match s1 ordering)

    # Reorder s2_structure so that it matches the ordering of s1_structure
    reordered_sites = [s2_structure[mapping_tuple[-1]] for mapping_tuple in mapping]

    # Avoid warning about selective_dynamics properties (can happen if user explicitly set "T T T" (or
    # otherwise) for the bulk):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Not all sites have property")
        return Structure.from_sites(reordered_sites)


def _compare_potcar_symbols(
    bulk_potcar_symbols,
    defect_potcar_symbols,
    bulk_name="bulk",
    defect_name="defect",
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
    bulk_actual_kpoints,
    defect_actual_kpoints,
    bulk_kpoints=None,
    defect_kpoints=None,
    bulk_name="bulk",
    defect_name="defect",
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
    bulk_incar_dict: dict[str, str | int | float],
    defect_incar_dict: dict[str, str | int | float],
    fatal_incar_mismatch_tags: dict[str, str | int | float] | None = None,
    ignore_tags: set[str] | None = None,
    bulk_name: str = "bulk",
    defect_name: str = "defect",
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


def get_magnetization_from_vasprun(vasprun: Vasprun) -> int | float | np.ndarray[float]:
    """
    Determine the total magnetization from a ``Vasprun`` object.

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
        vasprun (Vasprun):
            The ``Vasprun`` object from which to extract the total
            magnetization.

    Returns:
        int or float or np.ndarray[float]:
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
    Determine the number of electrons (``NELECT``) from a ``Vasprun`` object.

    Args:
        vasprun (Vasprun):
            The ``Vasprun`` object from which to extract ``NELECT``.

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
    Determine the number of electrons (``NELECT``) from a ``Vasprun`` object,
    corresponding to a neutral charge state for the structure.

    Args:
        vasprun (Vasprun):
            The ``Vasprun`` object from which to extract ``NELECT``.
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


def _get_defect_supercell_frac_coords(defect_entry: DefectEntry, relaxed=True) -> np.ndarray[float] | None:
    sc_defect_frac_coords = defect_entry.sc_defect_frac_coords
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
    ``DefectEntry``, updating the relevant attributes and calculation metadata.

    Args:
        defect_entry (DefectEntry):
            The ``DefectEntry`` object for which to update the defect site
            information.
        overwrite (bool):
            Whether to overwrite existing ``DefectEntry`` attributes with the
            newly parsed values. Default is ``False`` (i.e. only update if the
            attributes are not already set).
        **kwargs:
            Keyword arguments to pass to ``get_equiv_frac_coords_in_primitive``
            (such as ``symprec``, ``dist_tol_factor``,
            ``fixed_symprec_and_dist_tol_factor``, ``verbose``) and/or
            ``Defect`` initialization (such as ``oxi_state``, ``multiplicity``,
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
        bulk_supercell,
        defect_supercell,
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


def _partial_defect_entry_from_structures(
    bulk_supercell: Structure, defect_supercell: Structure, **kwargs
) -> DefectEntry:
    """
    Helper function to create a partial ``DefectEntry`` from the input bulk and
    defect supercells.

    Uses ``defect_and_info_from_structures`` to extract the defect structural
    information, and creates a corresponding ``DefectEntry`` object (which has
    no ``bulk_entry`` and a fake zero-energy ``sc_entry``, and so cannot be
    used for energy analyses). Primarily intended for internal usage in
    ``doped`` parsing/analysis functions.

    Args:
        bulk_supercell (Structure):
            The bulk supercell structure.
        defect_supercell (Structure):
            The defect supercell structure.
        **kwargs:
            Keyword arguments to pass to ``get_equiv_frac_coords_in_primitive``
            (such as ``symprec``, ``dist_tol_factor``,
            ``fixed_symprec_and_dist_tol_factor``, ``verbose``) and/or
            ``Defect`` initialization (such as ``oxi_state``, ``multiplicity``,
            ``symprec``, ``dist_tol_factor``) in the
            ``defect_and_info_from_structures`` function.

    Returns:
        DefectEntry:
            A partial ``DefectEntry`` object containing the defect and defect
            site information, but no ``bulk_entry`` and a zero-energy
            ``sc_entry``.
    """
    from doped.analysis import defect_and_info_from_structures

    (
        defect,
        defect_site,
        defect_structure_metadata,
    ) = defect_and_info_from_structures(
        bulk_supercell,
        defect_supercell,
        **kwargs,  # pass any additional kwargs (e.g. oxidation state, multiplicity, etc.)
    )

    return DefectEntry(
        # pmg attributes:
        defect=defect,  # this corresponds to _unrelaxed_ defect
        charge_state=0,
        sc_entry=ComputedStructureEntry(
            structure=bulk_supercell,
            energy=0.0,  # needs to be set, so set to 0.0
        ),
        sc_defect_frac_coords=defect_site.frac_coords,  # _relaxed_ defect site
        bulk_entry=None,
        # doped attributes:
        name="Partial Defect Entry",
        defect_supercell_site=defect_site,  # _relaxed_ defect site
        defect_supercell=defect_supercell,
        bulk_supercell=bulk_supercell,
        calculation_metadata=defect_structure_metadata,  # only structural metadata here
    )


def _num_electrons_from_charge_state(structure: Structure, charge_state: int = 0) -> int:
    """
    Get the total number of electrons (including core electrons! -- so
    different to ``NELECT`` in VASP in most cases) for a given structure and
    charge state.

    Args:
        structure (Structure):
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
        vasprun (Vasprun):
            ``pymatgen`` ``Vasprun`` for which to determine spin degeneracy.
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
        magnetization = get_magnetization_from_vasprun(vasprun)
        if isinstance(magnetization, np.ndarray):
            # take the vector norm as the total magnetization
            magnetization = np.linalg.norm(magnetization)

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
    Determine the total charge state of a system from the vasprun, and compare
    to the expected charge state if provided.

    Note that if the system is charged, then this function relies on access to
    ``POTCAR`` data, which can be setup with ``pymatgen`` as detailed on the
    installation page here:
    https://doped.readthedocs.io/en/latest/Installation.html#setup-potcars-and-materials-project-api

    Args:
        vasprun (Vasprun):
            ``pymatgen`` ``Vasprun`` object for which to determine the total
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
        structure (Structure): The structure to get the dimer bond lengths for.
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
