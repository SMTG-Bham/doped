"""
Code to generate Defect objects and supercell structures for ab-initio
calculations.
"""

import contextlib
import copy
import logging
import operator
import warnings
from collections import defaultdict
from functools import partial, reduce
from itertools import chain
from typing import TYPE_CHECKING, Any, Union, cast
from unittest.mock import MagicMock

import numpy as np
from monty.json import MontyDecoder, MSONable
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.defects import core, thermo
from pymatgen.analysis.defects.generators import (
    AntiSiteGenerator,
    InterstitialGenerator,
    SubstitutionGenerator,
    VacancyGenerator,
)
from pymatgen.analysis.defects.utils import remove_collisions
from pymatgen.core import IStructure, Structure
from pymatgen.core.composition import Composition, Element
from pymatgen.core.periodic_table import DummySpecies
from pymatgen.core.structure import PeriodicSite
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation
from pymatgen.util.typing import PathLike
from tabulate import tabulate
from tqdm import tqdm

from doped import pool_manager
from doped.core import (
    Defect,
    DefectEntry,
    Interstitial,
    Substitution,
    Vacancy,
    doped_defect_from_pmg_defect,
    guess_and_set_oxi_states_with_timeout,
)
from doped.utils import parsing, supercells, symmetry
from doped.utils.efficiency import Composition as doped_Composition
from doped.utils.efficiency import DopedTopographyAnalyzer, _doped_cluster_frac_coords
from doped.utils.efficiency import IStructure as doped_IStructure
from doped.utils.efficiency import PeriodicSite as doped_PeriodicSite
from doped.utils.parsing import reorder_s1_like_s2
from doped.utils.plotting import format_defect_name

if TYPE_CHECKING:
    from ase.atoms import Atoms

_dummy_species = DummySpecies("X")  # Dummy species used to keep track of defect coords in the supercell

core._logger.setLevel(logging.CRITICAL)  # avoid unnecessary pymatgen-analysis-defects warnings about
# oxi states (already handled within doped)


def _custom_formatwarning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    line: str | None = None,
) -> str:
    """
    Reformat warnings to just print the warning message.
    """
    return f"{message}\n"


warnings.formatwarning = _custom_formatwarning


def _list_index_or_val(lst, item, val=100):
    """
    Returns the index of the item in the lst, or val if not found.
    """
    try:
        return lst.index(item)
    except ValueError:
        return val


def get_defect_entry_from_defect(
    defect: Defect,
    defect_supercell: Structure,
    charge_state: int,
    dummy_species: DummySpecies = _dummy_species,
    sc_defect_frac_coords: np.ndarray | None = None,
):
    """
    Generate a ``doped`` ``DefectEntry`` object from a ``doped`` ``Defect``
    object.

    This is used to describe a ``Defect`` with a specified simulation cell.
    To set the ``sc_defect_frac_coords`` attribute for ``DefectEntry``
    (fractional coordinates of the defect in the ``defect_supercell``),
    either ``dummy_species`` must be present in the ``defect_supercell``
    (which is taken as the defect site), or ``sc_defect_frac_coords``
    must be set.

    Args:
        defect (Defect): ``doped``/``pymatgen`` ``Defect`` object.
        defect_supercell (Structure): Defect supercell structure.
        charge_state (int): Charge state of the defect.
        dummy_species (DummySpecies):
            Dummy species present in the ``defect_supercell`` structure,
            used to determine ``sc_defect_frac_coords``. If not found
            and ``sc_defect_frac_coords`` is not set,
            ``DefectEntry.sc_defect_frac_coords`` is set to ``None``.
            Default is ``DummySpecies("X")``.
        sc_defect_frac_coords (np.ndarray):
            Fractional coordinates of the defect in the defect supercell.
            If not set and ``dummy_species`` is not found in the
            ``defect_supercell``, ``DefectEntry.sc_defect_frac_coords``
            is set to ``None``.
            Default is None.

    Returns:
        DefectEntry: ``doped`` ``DefectEntry`` object.
    """
    defect_entry_structure = (
        defect_supercell.copy()
    )  # duplicate the structure so we don't edit the input Structure

    if sc_defect_frac_coords is None:
        # Dummy species (used to keep track of the defect coords in the supercell)
        # Find its fractional coordinates & remove it from the supercell
        dummy_sites = [
            site for site in defect_entry_structure if site.specie.symbol == dummy_species.symbol
        ]
        if dummy_sites:
            dummy_site = next(iter(dummy_sites))
            sc_defect_frac_coords = dummy_site.frac_coords
            defect_entry_structure.remove(dummy_site)

    computed_structure_entry = ComputedStructureEntry(
        structure=defect_entry_structure,
        energy=0,  # needs to be set, so set to 0
    )
    return DefectEntry(
        defect=defect,
        charge_state=charge_state,
        sc_entry=computed_structure_entry,
        sc_defect_frac_coords=sc_defect_frac_coords,
    )


def _defect_dict_key_from_pmg_type(defect_type: core.DefectType) -> str:
    """
    Get the corresponding defect dictionary key from the ``pymatgen``
    ``DefectType``.

    Args:
        defect_type (core.DefectType): ``pymatgen`` ``DefectType``.

    Returns:
        str: Defect dictionary key.
    """
    if defect_type == core.DefectType.Vacancy:
        return "vacancies"
    if defect_type == core.DefectType.Substitution:
        return "substitutions"
    if defect_type == core.DefectType.Interstitial:
        return "interstitials"
    if defect_type == core.DefectType.Other:
        return "others"

    raise ValueError(
        f"Defect type {defect_type} not recognised. Must be one of {core.DefectType.Vacancy}, "
        f"{core.DefectType.Substitution}, {core.DefectType.Interstitial}, {core.DefectType.Other}."
    )


def _defect_type_key_from_pmg_type(defect_type: core.DefectType) -> str:
    """
    Get the corresponding defect type name from the ``pymatgen``
    ``DefectType``.

    i.e. "vacancy", "substitution", "interstitial", "other"

    Args:
        defect_type (core.DefectType): ``pymatgen`` ``DefectType``.

    Returns:
        str: Defect type key.
    """
    if defect_type == core.DefectType.Vacancy:
        return "vacancy"
    if defect_type == core.DefectType.Substitution:
        return "substitution"
    if defect_type == core.DefectType.Interstitial:
        return "interstitial"
    if defect_type == core.DefectType.Other:
        return "other"

    raise ValueError(
        f"Defect type {defect_type} not recognised. Must be one of {core.DefectType.Vacancy}, "
        f"{core.DefectType.Substitution}, {core.DefectType.Interstitial}, {core.DefectType.Other}."
    )


def get_neighbour_distances_and_symbols(
    site: PeriodicSite,
    structure: Structure,
    n: int = 1,
    element_list: list[str] | None = None,
    dist_tol_prefactor: float = 3.0,
    min_dist: float = 0.02,
):
    r"""
    Get a list of sorted tuples of (distance, element) for the ``n``\th closest
    sites to the input ``site`` in ``structure``, where each consecutive
    neighbour in the list must be at least 0.02 Å further away than the
    previous neighbour.

    If there are multiple elements with the same distance (to within ~0.01 Å),
    then the preference for which element to return is controlled by
    ``element_list``. If ``element_list`` is not provided, then it is set to
    match the order of appearance in the structure composition.

    For efficiency, this function dynamically increases the radius when
    searching for nearest neighbours as necessary, using the input
    ``dist_tol_prefactor`` and ``n`` to estimate a reasonable starting range.

    Args:
        site (PeriodicSite):
            Site to get neighbour info.
        structure (Structure):
            Structure containing the site and neighbours.
        n (int):
            Return the element symbol and distance tuples for the ``n``\th
            closest neighbours. Default is 1, corresponding to only the
            nearest neighbour.
        element_list (list):
            Sorted list of elements in the host structure to govern the
            preference of elemental symbols to return, when the distance to
            multiple neighbours with different elements is the same.
            Default is to use the order of appearance of elements in the
            ``Structure`` composition.
        dist_tol_prefactor (float):
            Initial distance tolerance prefactor to use when searching for
            neighbours, where the initial search radius is set to
            ``dist_tol_prefactor * sqrt(n)`` Å. This value is dynamically
            updated as needed, and so should only be adjusted if providing
            odd systems with very small/large bond lengths for which
            efficiency improvements are possible. Default is 3.0.
        min_dist (float):
            Minimum distance in Å of neighbours from ``site``. Intended as a
            distance tolerance to exclude the site itself from the neighbour
            list, for which the default value (0.02 Å) should be perfectly
            fine in most cases. Set to 0 to include the site itself in the
            output list (in which case it counts toward ``n``).

    Returns:
        list: Sorted list of tuples of (distance, element) for the ``n``\th
        closest neighbours to the input ``site``.
    """
    if element_list is None:
        element_list = [el.symbol for el in structure.composition.elements]

    neighbour_tuples: list[tuple[float, str]] = []
    while len(neighbour_tuples) < n:
        # for efficiency, ignore sites further than dist_tol*sqrt(n) Å away
        # dynamic upscaling of the distance tolerance is required for weird structures (e.g. mp-674158,
        # mp-1208561):
        neighbours_w_site_itself = structure.get_sites_in_sphere(
            site.coords, dist_tol_prefactor * np.sqrt(n)
        )
        neighbours = sorted(neighbours_w_site_itself, key=lambda x: x.nn_distance)

        if not neighbours:
            continue

        dist_tol_prefactor += 0.5  # increase the distance tolerance if no other sites are found
        if dist_tol_prefactor > 40:
            warnings.warn(
                "No other sites found within 40*sqrt(n) Å of the defect site, indicating a very "
                "weird structure..."
            )
            break

        neighbour_tuples = sorted(  # Could make this faster using caching if it was becoming a bottleneck
            [
                (
                    neigh.nn_distance,
                    neigh.specie.symbol,
                )
                for neigh in neighbours
            ],
            key=lambda x: (symmetry._custom_round(x[0], 2), _list_index_or_val(element_list, x[1]), x[1]),
        )
        neighbour_tuples = [  # prune site_distances to remove any with distances within 0.02 Å of the
            # previous n:
            neighbour_tuples[i]
            for i in range(len(neighbour_tuples))
            if neighbour_tuples[i][0] > min_dist
            and (
                i == 0
                or abs(neighbour_tuples[i][0] - neighbour_tuples[i - 1][0]) > 0.02
                or neighbour_tuples[i][1] != neighbour_tuples[i - 1][1]
            )
        ][:n]

    return neighbour_tuples


def closest_site_info(
    defect_entry_or_defect: DefectEntry | Defect,
    n: int = 1,
    element_list: list[str] | None = None,
):
    r"""
    Return the element and distance (rounded to 2 decimal places) of the nth
    closest site to the defect site in the input ``DefectEntry`` or ``Defect``
    object.

    If there are multiple elements with the same distance (to within ~0.01 Å),
    then the preference for which element to return is controlled by
    ``element_list``. If ``element_list`` is not provided, then it is set to
    match the order of appearance in the structure composition.

    If ``n`` > 1, then it returns the nth closest site, where the nth site
    must be at least 0.02 Å further away than the (n-1)th site.

    Args:
        defect_entry_or_defect (Union[DefectEntry, Defect]):
            ``DefectEntry`` or ``Defect`` object, to get neighbour info.
        n (int):
            Return the element symbol and distance for the ``n``\th closest
            site. Default is 1, corresponding to the (1st) closest neighbour.
        element_list (list):
            Sorted list of elements in the host structure to govern the
            preference of elemental symbols to return, when the distance to
            multiple neighbours with different elements is the same.
            Default is to use ``_get_element_list()``, which follows the
            order of appearance of elements in the ``Structure`` composition.

    Returns:
        str: Element symbol and distance (rounded to 2 decimal places) of the
        nth closest site to the defect site.
    """
    if isinstance(defect_entry_or_defect, DefectEntry | thermo.DefectEntry):
        site = None
        with contextlib.suppress(Exception):
            defect = defect_entry_or_defect.defect
            site = defect.site
            structure = defect.structure
        if site is None:
            # use defect_supercell_site if attribute exists, otherwise use sc_defect_frac_coords:
            site = parsing._get_defect_supercell_site(defect_entry_or_defect)
            structure = parsing._get_bulk_supercell(defect_entry_or_defect)

    elif isinstance(defect_entry_or_defect, Defect | core.Defect):
        if isinstance(defect_entry_or_defect, core.Defect):
            defect = doped_defect_from_pmg_defect(defect_entry_or_defect)  # convert to doped Defect
        else:
            defect = defect_entry_or_defect

        site = defect.site
        structure = defect.structure

    else:
        raise TypeError(
            f"defect_entry_or_defect must be a DefectEntry or Defect object, not "
            f"{type(defect_entry_or_defect)}"
        )

    if element_list is None:
        element_list = _get_element_list(defect)

    site_distances = get_neighbour_distances_and_symbols(site, structure, n, element_list)
    if not site_distances:
        return ""  # min dist > 40, already warned in ``get_neighbour_distances_and_symbols``

    min_distance, closest_site = site_distances[n - 1]
    return f"{closest_site}{symmetry._custom_round(min_distance, 2):.2f}"


def get_defect_name_from_defect(
    defect: Defect,
    element_list: list[str] | None = None,
    symm_ops: list | None = None,
    symprec: float = 0.01,
):
    """
    Get the doped/SnB defect name from Defect object.

    Args:
        defect (Defect): Defect object.
        element_list (list):
            Sorted list of elements in the host structure, so that
            closest_site_info returns deterministic results (in case two
            different elements located at the same distance from defect site).
            Default is None.
        symm_ops (list):
            List of symmetry operations of ``defect.structure``, to avoid
            re-calculating. Default is None (recalculates).
        symprec (float):
            Symmetry tolerance for ``spglib``. Default is 0.01.

    Returns:
        str: Defect name.
    """
    point_group_symbol = symmetry.point_symmetry_from_defect(defect, symm_ops=symm_ops, symprec=symprec)

    return f"{defect.name}_{point_group_symbol}_{closest_site_info(defect, element_list=element_list)}"


def get_defect_name_from_entry(
    defect_entry: DefectEntry,
    element_list: list | None = None,
    symm_ops: list | None = None,
    symprec: float | None = None,
    relaxed: bool = True,
):
    r"""
    Get the doped/SnB defect name from a DefectEntry object.

    Note: If relaxed = True (default), then this tries to use the
    defect_entry.defect_supercell to determine the site symmetry. This will
    thus give the `relaxed` defect point symmetry if this is a DefectEntry
    created from parsed defect calculations. However, it should be noted
    that this is not guaranteed to work in all cases; namely for non-diagonal
    supercell expansions, or sometimes for non-scalar supercell expansion
    matrices (e.g. a 2x1x2 expansion)(particularly with high-symmetry materials)
    which can mess up the periodicity of the cell. doped tries to automatically
    check if this is the case, and will warn you if so.

    This can also be checked by using this function on your doped `generated` defects:

    .. code-block:: python

        from doped.generation import get_defect_name_from_entry
        for defect_name, defect_entry in defect_gen.items():
            print(defect_name, get_defect_name_from_entry(defect_entry, relaxed=False),
                  get_defect_name_from_entry(defect_entry), "\n")

    And if the point symmetries match in each case, then using this function on your
    parsed `relaxed` DefectEntry objects should correctly determine the final relaxed
    defect symmetry (and closest site info) - otherwise periodicity-breaking prevents this.

    Args:
        defect_entry (DefectEntry): ``DefectEntry`` object.
        element_list (list):
            Sorted list of elements in the host structure, so that
            closest_site_info returns deterministic results (in case two
            different elements located at the same distance from defect site).
            Default is None.
        symm_ops (list):
            List of symmetry operations of either the defect_entry.bulk_supercell
            structure (if relaxed=False) or defect_entry.defect_supercell (if
            relaxed=True), to avoid re-calculating. Default is None (recalculates).
        symprec (float):
            Symmetry tolerance for ``spglib``. Default is 0.01 for unrelaxed structures,
            0.2 for relaxed (to account for residual structural noise). You may
            want to adjust for your system (e.g. if there are very slight
            octahedral distortions etc).
        relaxed (bool):
            If False, determines the site symmetry using the defect site `in the
            unrelaxed bulk supercell`, otherwise tries to determine the point
            symmetry of the relaxed defect in the defect supercell).
            Default is True.

    Returns:
        str: Defect name.
    """
    point_group_symbol = symmetry.point_symmetry_from_defect_entry(
        defect_entry, symm_ops=symm_ops, symprec=symprec, relaxed=relaxed
    )

    return (
        f"{defect_entry.defect.name}_{point_group_symbol}"
        f"_{closest_site_info(defect_entry, element_list=element_list)}"
    )


def _get_neutral_defect_entry(
    defect,
    supercell_matrix,
    target_frac_coords,
    bulk_supercell,
    conventional_structure,
    _BilbaoCS_conv_cell_vector_mapping,
    wyckoff_label_dict,
    conv_symm_ops,
):
    (
        dummy_defect_supercell,
        defect_supercell_site,
        equivalent_supercell_sites,
    ) = defect.get_supercell_structure(
        sc_mat=supercell_matrix,
        dummy_species=_dummy_species.symbol,  # keep track of the defect frac coords in the supercell
        target_frac_coords=target_frac_coords,
        return_sites=True,
    )
    dummy_sites = [site for site in dummy_defect_supercell if site.specie.symbol == _dummy_species.symbol]
    if dummy_sites:  # set defect_supercell_site to exactly match coordinates,
        # as can have very small differences in generation due to rounding
        dummy_site = next(iter(dummy_sites))
        defect_supercell_site._frac_coords = dummy_site.frac_coords

    neutral_defect_entry = get_defect_entry_from_defect(
        defect,
        dummy_defect_supercell,
        0,
        dummy_species=_dummy_species,
    )
    neutral_defect_entry.defect_supercell = neutral_defect_entry.sc_entry.structure
    neutral_defect_entry.defect_supercell_site = defect_supercell_site
    neutral_defect_entry.equivalent_supercell_sites = equivalent_supercell_sites
    neutral_defect_entry.bulk_supercell = bulk_supercell

    neutral_defect_entry.conventional_structure = neutral_defect_entry.defect.conventional_structure = (
        conventional_structure
    )

    try:
        wyckoff_label, conv_cell_sites = symmetry.get_wyckoff(
            symmetry.get_conv_cell_site(neutral_defect_entry).frac_coords,
            conventional_structure,
            conv_symm_ops,
            equiv_sites=True,
        )
        conv_cell_coord_list = [
            np.mod(symmetry._vectorized_custom_round(site.frac_coords), 1) for site in conv_cell_sites
        ]

    except Exception as e:  # (slightly) less efficient algebraic matching:
        try:
            wyckoff_label, conv_cell_coord_list = symmetry.get_wyckoff_label_and_equiv_coord_list(
                defect_entry=neutral_defect_entry,
                wyckoff_dict=wyckoff_label_dict,
            )
            conv_cell_coord_list = np.mod(
                symmetry._vectorized_custom_round(conv_cell_coord_list), 1
            ).tolist()
        except Exception as e2:
            warnings.warn(
                f"Conventional cell site (and thus Wyckoff label) could not be determined! Got "
                f"errors: {e!r}\nand: {e2!r}"
            )
            wyckoff_label = "N/A"
            conv_cell_coord_list = []

    # sort array with symmetry._frac_coords_sort_func:
    conv_cell_coord_list.sort(key=symmetry._frac_coords_sort_func)

    neutral_defect_entry.wyckoff = neutral_defect_entry.defect.wyckoff = wyckoff_label
    neutral_defect_entry.conv_cell_frac_coords = neutral_defect_entry.defect.conv_cell_frac_coords = (
        None if not conv_cell_coord_list else conv_cell_coord_list[0]
    )  # ideal/cleanest coords
    neutral_defect_entry.equiv_conv_cell_frac_coords = (
        neutral_defect_entry.defect.equiv_conv_cell_frac_coords
    ) = conv_cell_coord_list
    neutral_defect_entry._BilbaoCS_conv_cell_vector_mapping = (
        neutral_defect_entry.defect._BilbaoCS_conv_cell_vector_mapping
    ) = _BilbaoCS_conv_cell_vector_mapping

    return neutral_defect_entry


def _check_if_name_subset(long_name: str, poss_subset_name: str):
    """
    Check if the longer name is a superset of the shorter name, where
    underscores are used as a name delimiter as in ``doped``.

    Previously used ``startswith`` to compare, but caused issues
    with e.g. ``v_Cl`` and ``v_C`` defects in the same material.
    """
    subset_num_underscores = poss_subset_name.count("_")
    for i in range(subset_num_underscores + 1):
        long_name_part = long_name.split("_")[i]
        if "." in long_name_part and any(char.isdigit() for char in long_name_part):
            # closest site info, use startswith for this as we no longer use delimiters
            if not long_name_part.startswith(poss_subset_name.split("_")[i]):
                return False

        elif long_name_part != poss_subset_name.split("_")[i]:
            return False

    return True


def name_defect_entries(
    defect_entries: list[DefectEntry | Defect],
    element_list: list[str] | None = None,
    symm_ops: list | None = None,
):
    """
    Create a dictionary of ``{name: DefectEntry}`` from a list of
    ``DefectEntry`` objects, where the names are set according to the default
    doped algorithm; which is to use the pymatgen defect name (e.g. v_Cd, Cd_Te
    etc.) for vacancies/antisites/substitutions, unless there are multiple
    inequivalent sites for the defect, in which case the point group of the
    defect site is appended (e.g. v_Cd_Td, Cd_Te_Td etc.), and if this is still
    not unique, then element identity and distance to the nearest neighbour of
    the defect site is appended (e.g. v_Cd_Td_Te2.83, Cd_Te_Td_Cd2.83 etc.).
    Names do not yet have charge states included.

    For interstitials, the same naming scheme is used, but the point group is
    always appended to the pymatgen defect name.

    If still not unique after the 3rd nearest neighbour info, then "a, b, c"
    etc is appended to the name of different defects to distinguish.

    Args:
        defect_entries (list): List of ``DefectEntry`` or ``Defect`` objects to name.
        element_list (list):
            Sorted list of elements in the host structure, so that
            ``closest_site_info`` returns deterministic results (in case two
            different elements located at the same distance from defect site).
            Default is None.
        symm_ops (list):
            List of symmetry operations of defect.structure (i.e. the primitive
            structure), to avoid re-calculating. Default is None (recalculates).

    Returns:
        dict: Dictionary of ``{name: DefectEntry}`` objects.
    """

    def get_shorter_name(full_defect_name, split_number):
        if split_number < 1:  # if split number is less than 1, return full name
            return full_defect_name
        return full_defect_name.rsplit("_", split_number)[0]

    def get_matching_names(defect_naming_dict, defect_name):
        return [name for name in defect_naming_dict if _check_if_name_subset(name, defect_name)]

    def handle_unique_match(defect_naming_dict, matching_names, split_number):
        if len(matching_names) == 1:
            previous_entry = defect_naming_dict.pop(matching_names[0])
            prev_defect = (
                previous_entry.defect
                if isinstance(previous_entry, DefectEntry | thermo.DefectEntry)
                else previous_entry
            )
            previous_entry_full_name = get_defect_name_from_defect(prev_defect, element_list, symm_ops)
            previous_entry_name = get_shorter_name(previous_entry_full_name, split_number - 1)
            defect_naming_dict[previous_entry_name] = previous_entry

        return defect_naming_dict

    def append_closest_site_info(name, entry, n):
        return name + closest_site_info(entry, n=n, element_list=element_list)

    def handle_multiple_matches(defect_naming_dict, full_defect_name, defect_entry, element_list=None):
        n = 2
        while True:
            for name in list(defect_naming_dict.keys()):
                if full_defect_name == name:
                    try:
                        prev_defect_entry_full_name = append_closest_site_info(
                            name, defect_naming_dict[name], n
                        )
                        prev_defect_entry = defect_naming_dict.pop(name)
                        defect_naming_dict[prev_defect_entry_full_name] = prev_defect_entry

                    except IndexError:
                        return handle_repeated_name(defect_naming_dict, full_defect_name)

            try:
                full_defect_name = append_closest_site_info(full_defect_name, defect_entry, n)
            except IndexError:
                return handle_repeated_name(defect_naming_dict, full_defect_name)

            if not any(_check_if_name_subset(name, full_defect_name) for name in defect_naming_dict):
                return defect_naming_dict, full_defect_name

            if n == 3:  # if still not unique after 3rd nearest neighbour, just use alphabetical indexing
                return handle_repeated_name(defect_naming_dict, full_defect_name)
            n += 1

    def handle_repeated_name(defect_naming_dict, full_defect_name):
        defect_name = None
        for name in list(defect_naming_dict.keys()):
            if full_defect_name == name:
                prev_defect_entry = defect_naming_dict.pop(name)
                defect_naming_dict[f"{name}a"] = prev_defect_entry
                defect_name = f"{full_defect_name}b"
                break
            if full_defect_name == name[:-1] and not Element("H").is_valid_symbol(name.split("_")[-1]):
                # if name is a subset barring the last letter, and last underscore split is not an Element
                # (i.e. ``v_Cl`` not being matched with ``v_C``)
                last_letters = [name[-1] for name in defect_naming_dict if name[:-1] == full_defect_name]
                last_letters.sort()
                new_letter = chr(ord(last_letters[-1]) + 1)
                defect_name = full_defect_name + new_letter
                break

        if defect_name is None:
            raise ValueError(
                f"Multiple defect names found for {full_defect_name}, and couldn't be "
                f"renamed properly. Please report this issue to the developers."
            )

        return defect_naming_dict, defect_name

    defect_naming_dict: dict[str, DefectEntry] = {}
    for defect_entry in defect_entries:
        defect = (
            defect_entry.defect
            if isinstance(defect_entry, DefectEntry | thermo.DefectEntry)
            else defect_entry
        )
        full_defect_name = get_defect_name_from_defect(defect, element_list, symm_ops)
        split_number = 1 if defect.defect_type == core.DefectType.Interstitial else 2
        shorter_defect_name = get_shorter_name(full_defect_name, split_number)
        if not any(_check_if_name_subset(name, shorter_defect_name) for name in defect_naming_dict):
            defect_naming_dict[shorter_defect_name] = defect_entry
            continue

        matching_shorter_names = get_matching_names(defect_naming_dict, shorter_defect_name)
        defect_naming_dict = handle_unique_match(defect_naming_dict, matching_shorter_names, split_number)
        shorter_defect_name = get_shorter_name(full_defect_name, split_number - 1)
        if not any(_check_if_name_subset(name, shorter_defect_name) for name in defect_naming_dict):
            defect_naming_dict[shorter_defect_name] = defect_entry
            continue

        matching_shorter_names = get_matching_names(defect_naming_dict, shorter_defect_name)
        defect_naming_dict = handle_unique_match(
            defect_naming_dict, matching_shorter_names, split_number - 1
        )
        shorter_defect_name = get_shorter_name(full_defect_name, split_number - 2)
        if not any(_check_if_name_subset(name, shorter_defect_name) for name in defect_naming_dict):
            defect_naming_dict[shorter_defect_name] = defect_entry
            continue

        defect_naming_dict, full_defect_name = handle_multiple_matches(
            defect_naming_dict, full_defect_name, defect_entry
        )
        defect_naming_dict[full_defect_name] = defect_entry

    if len(defect_entries) != len(defect_naming_dict):
        raise ValueError(
            f"Number of defect entries ({len(defect_entries)}) does not match "
            f"number of unique defect names ({len(defect_naming_dict)}). "
            f"Please report this issue to the developers."
        )
    return defect_naming_dict


def get_oxi_probabilities(element_symbol: str) -> dict:
    """
    Get a dictionary of oxidation states and their probabilities for an
    element.

    Tries to get the probabilities from the ``pymatgen`` tabulated ICSD oxidation
    state probabilities, and if not available, uses the common oxidation states
    of the element.

    Args:
        element_symbol (str): Element symbol.

    Returns:
        dict: Dictionary of oxidation states (ints) and their probabilities (floats).
    """
    comp_obj = Composition(element_symbol)
    comp_obj.add_charges_from_oxi_state_guesses()  # add oxidation states to Composition object
    if oxi_probabilities := {
        k.oxi_state: v
        for k, v in comp_obj.oxi_prob.items()
        if k.element.symbol == element_symbol and k.oxi_state != 0
    }:  # not empty
        return {
            int(k): round(v / sum(oxi_probabilities.values()), 3) for k, v in oxi_probabilities.items()
        }

    element_obj = Element(element_symbol)
    if element_obj.common_oxidation_states:
        return {
            int(k): 1 / len(element_obj.common_oxidation_states)
            for k in element_obj.common_oxidation_states
        }  # known common oxidation states

    # no known _common_ oxidation state, make guess and warn user
    if element_obj.oxidation_states:
        oxi_states = {
            int(k): 1 / len(element_obj.oxidation_states) for k in element_obj.oxidation_states
        }  # known oxidation states
    else:
        oxi_states = {0: 1}  # no known oxidation states, return 0 with 100% probability

    warnings.warn(
        f"No known common oxidation states in pymatgen/ICSD dataset for element "
        f"{element_obj.name}. If this results in unreasonable charge states, you "
        f"should manually edit the defect charge states."
    )

    return oxi_states


def charge_state_probability(
    charge_state: int,
    defect_el_oxi_state: int,
    defect_el_oxi_probability: float,
    max_host_oxi_magnitude: int,
    return_log: bool = False,
) -> float | dict:
    """
    Function to estimate the probability of a given defect charge state, using
    the probability of the corresponding defect element oxidation state, the
    magnitude of the charge state, and the maximum magnitude of the host
    oxidation states (i.e. how 'charged' the host is).

    Disfavours large (absolute) charge states, low probability oxidation
    states and greater charge/oxidation state magnitudes than that of the host.
    This charge state probability function is primarily intended for substitutions
    and interstitials, while the ``get_vacancy_charge_states()`` function is used
    for vacancies.

    Specifically, the overall probability is given by the product of these
    probability factors:

    - The probability of the corresponding oxidation state of the defect element
      (e.g. Na_i^+1 has Na in the +1 oxidation state), as given by its prevalence
      in the ICSD.
    - The magnitude of the charge state; with a probability function:
      ``1/|charge_state|^(2/3)``
    - The magnitude of the charge state relative to the max host oxidation state
      (i.e. how 'charged' the host is); with a probability function:
      ``1/(2*|charge_state - max_host_oxi_magnitude|)^(2/3)`` if
      ``charge_state > max_host_oxi_magnitude``, otherwise 1.
    - The magnitude of the defect element oxidation state relative to the max host
      oxidation state; with a probability function:
      ``1/(2*|defect_el_oxi_state - max_host_oxi_magnitude|)^(2/3)`` if
      ``defect_el_oxi_state > max_host_oxi_magnitude``, otherwise 1.

    Note that neutral charge states are always included.

    This probability function was found to give optimal performance in terms of
    efficiency and completeness when tested against other approaches (see the
    ``doped`` JOSS paper: https://doi.org/10.21105/joss.06433) but of course may
    not be perfect in all cases, so make sure to critically consider the estimated
    charge states for your system!

    Args:
        charge_state (int): Charge state of defect.
        defect_el_oxi_state (int): Oxidation state of defect element.
        defect_el_oxi_probability (float):
            Probability of oxidation state of defect element.
        max_host_oxi_magnitude (int): Maximum host oxidation state magnitude.
        return_log (bool):
            If true, returns a dictionary of input & computed values
            used to determine charge state probability. Default is False.

    Returns:
        Probability of the defect charge state (between 0 and 1) if return_log
        is False, otherwise a dictionary of input & computed values used to
        determine charge state probability.
    """
    # for defect charge states; 0 to +/-2 likely, 3-4 less likely, 5-6 v unlikely, >=7 unheard of

    # incorporates oxidation state probabilities and overall defect site charge
    # (and thus by proxy the cationic/anionic identity of the substituting site)
    # Thought about incorporating the oxi state probabilities (i.e. reducibility/oxidisability)
    # of the neighbouring elements to this (particularly for vacancies), but no clear-cut
    # cases where this would actually improve performance

    def _defect_vs_host_charge(charge_state: int, host_charge: int) -> float:
        if abs(charge_state) <= abs(host_charge):
            return 1

        return 1 / (2 * (abs(charge_state) - abs(host_charge)))

    charge_state_guessing_log = {
        "input_parameters": {
            "charge_state": int(charge_state),
            "oxi_state": int(defect_el_oxi_state),
            "oxi_probability": defect_el_oxi_probability,
            "max_host_oxi_magnitude": int(max_host_oxi_magnitude),
        },
        "probability_factors": {
            "oxi_probability": defect_el_oxi_probability,
            "charge_state_magnitude": (1 / abs(charge_state)) ** (2 / 3) if charge_state != 0 else 1,
            "charge_state_vs_max_host_charge": _defect_vs_host_charge(charge_state, max_host_oxi_magnitude)
            ** (2 / 3),
            "oxi_state_vs_max_host_charge": _defect_vs_host_charge(
                defect_el_oxi_state, max_host_oxi_magnitude
            )
            ** (2 / 3),
        },
    }
    # product of charge_state_guessing_log["probability_factors"].values()
    charge_state_guessing_log["probability"] = (
        np.prod(list(charge_state_guessing_log["probability_factors"].values()))
        if charge_state != 0
        else 1
    )  # always include neutral charge state

    if return_log:
        return charge_state_guessing_log

    return charge_state_guessing_log["probability"]


def get_vacancy_charge_states(vacancy: Vacancy, padding: int = 1) -> list[int]:
    """
    Get the estimated charge states for a vacancy defect, which is from
    +/-``padding`` to the fully-ionised vacancy charge state (a.k.a. the
    vacancy oxidation state).

    e.g. for vacancies in Sb2O5 (https://doi.org/10.1021/acs.chemmater.3c03257),
    the fully-ionised charge states for ``V_Sb`` and ``V_O`` are -5 and +2
    respectively (i.e. the negative of the elemental oxidation states in Sb2O5),
    so the estimated charge states would be from +1 to -5 for ``V_Sb`` and from
    +2 to -1 for ``V_O`` for the default ``padding`` of 1.

    This probability function was found to give optimal performance in terms of
    efficiency and completeness when tested against other approaches (see the
    ``doped`` JOSS paper: https://doi.org/10.21105/joss.06433) but of course may
    not be perfect in all cases, so make sure to critically consider the estimated
    charge states for your system!

    Args:
        vacancy (Defect): A ``doped`` ``Vacancy`` object.
        padding (int):
            Padding for vacancy charge states, such that the vacancy
            charge states are set to range(vacancy oxi state, padding),
            if vacancy oxidation state is negative, or to
            range(-padding, vacancy oxi state), if positive.
            Default is 1.

    Returns:
        list[int]: A list of estimated charge states for the defect.
    """
    if not isinstance(vacancy.oxi_state, int | float):
        raise ValueError(
            f"Vacancy oxidation state (= {vacancy.oxi_state}) is not an integer or float (needed for "
            f"charge state guessing)! Please manually set the vacancy oxidation state."
        )
    if vacancy.oxi_state > 0:
        return list(range(-padding, int(vacancy.oxi_state) + 1))  # from -1 to oxi_state
    if vacancy.oxi_state < 0:
        return list(range(int(vacancy.oxi_state), padding + 1))  # from oxi_state to +1

    # oxi_state is 0
    return list(range(-padding, padding + 1))  # from -1 to +1 for default


def _get_possible_oxi_states(defect: Defect) -> dict:
    """
    Get the possible oxidation states and probabilities for a defect.

    Args:
        defect (Defect): A doped Defect object.

    Returns:
        dict: A dictionary with possible oxidation states as
            keys and their probabilities as values.
    """
    return {
        int(k): prob
        for k, prob in get_oxi_probabilities(defect.site.specie.symbol).items()
        if prob > 0.001  # at least 0.1% occurrence
    }


def _get_charge_states(
    possible_oxi_states: dict,
    orig_oxi: int,
    max_host_oxi_magnitude: int,
    return_log: bool = False,
) -> dict:
    return {
        int(oxi - orig_oxi): charge_state_probability(
            oxi - orig_oxi, oxi, oxi_prob, max_host_oxi_magnitude, return_log=return_log
        )
        for oxi, oxi_prob in possible_oxi_states.items()
    }


def guess_defect_charge_states(
    defect: Defect, probability_threshold: float = 0.0075, padding: int = 1, return_log: bool = False
) -> list[int] | tuple[list[int], list[dict]]:
    """
    Guess the possible stable charge states of a defect.

    This function estimates the probabilities of the charge states of a defect,
    using the probability of the corresponding defect element oxidation states,
    the magnitudes of the charge states, and the maximum magnitude of the host
    oxidation states (i.e. how 'charged' the host is), and returns a list of
    charge states that have an estimated probability greater than the
    ``probability_threshold``.

    Disfavours large (absolute) charge states, low probability oxidation
    states and greater charge/oxidation state magnitudes than that of the host.
    Note that neutral charge states are always included.

    For specific details on the probability functions employed, see the
    ``charge_state_probability`` (for substitutions and interstitials) and
    ``get_vacancy_charge_states()`` (for vacancies) functions.

    These probability functions were found to give optimal performance in terms of
    efficiency and completeness when tested against other approaches (see the
    ``doped`` JOSS paper: https://doi.org/10.21105/joss.06433) but of course may
    not be perfect in all cases, so make sure to critically consider the estimated
    charge states for your system!

    Args:
        defect (Defect): doped Defect object.
        probability_threshold (float):
            Probability threshold for including defect charge states
            (for substitutions and interstitials). Default is 0.0075.
        padding (int):
            Padding for vacancy charge states, such that the vacancy
            charge states are set to range(vacancy oxi state, padding),
            if vacancy oxidation state is negative, or to
            range(-padding, vacancy oxi state), if positive.
            Default is 1.
        return_log (bool):
            If true, returns a tuple of the defect charge states and
            a list of dictionaries of input & computed values
            used to determine charge state probability. Default is False.

    Returns:
        List of defect charge states (int) or a tuple of the defect
        charge states (list) and a list of dictionaries of input &
        computed values used to determine charge state probability.
    """
    # Could consider bandgap magnitude as well, by pulling from Materials Project. Smaller gaps mean
    # extreme charge states less likely. Would rather avoid having to query the database here though,
    # as could give inconsistent results depending on whether the user generated defects with internet
    # access or not (i.e. MP access or not). Will keep in mind.
    if defect.defect_type == core.DefectType.Vacancy:
        # Set defect charge state: from +/-1 to defect oxi state
        vacancy_charge_states = get_vacancy_charge_states(defect, padding=padding)
        if return_log:
            charge_state_guessing_log = [
                {
                    "input_parameters": {
                        "charge_state": int(charge_state),
                    },
                    "probability_factors": {"oxi_probability": 1},
                    "probability": 1,
                    "probability_threshold": probability_threshold,
                    "padding": padding,
                }
                for charge_state in vacancy_charge_states
            ]
            return (vacancy_charge_states, charge_state_guessing_log)

        return vacancy_charge_states

    possible_oxi_states = _get_possible_oxi_states(defect)
    max_host_oxi_magnitude = int(max(abs(site.specie.oxi_state) for site in defect.structure))
    if defect.defect_type == core.DefectType.Substitution:
        orig_oxi = int(defect.structure[defect.defect_site_index].specie.oxi_state)
    else:  # interstitial
        orig_oxi = 0
    possible_charge_states = _get_charge_states(
        possible_oxi_states, orig_oxi, max_host_oxi_magnitude, return_log=True
    )

    if charge_state_list := [
        k for k, v in possible_charge_states.items() if v["probability"] > probability_threshold
    ]:
        charge_state_range = (int(min(charge_state_list)), int(max(charge_state_list)))
    else:
        charge_state_range = (0, 0)

    # check if defect element (interstitial/substitution) is present in structure (i.e. intrinsic
    # interstitial or antisite):
    defect_el_sites_in_struct = [
        site for site in defect.structure if site.specie.symbol == defect.site.specie.symbol
    ]
    defect_el_oxi_in_struct = (
        int(np.mean([site.specie.oxi_state for site in defect_el_sites_in_struct]))
        if defect_el_sites_in_struct
        else None
    )

    if (
        defect.defect_type == core.DefectType.Substitution
        and defect_el_oxi_in_struct is not None
        and defect_el_oxi_in_struct - orig_oxi
        not in range(charge_state_range[0], charge_state_range[1] + 1)
    ):
        # if simple antisite oxidation state difference not included, recheck with bumped up oxi_state
        # probability for the oxi_state of the substitution atom in the structure
        # should really be included unless it gives an absolute charge state >= 5, so set oxi_state
        # probability to 100%
        possible_charge_states[defect_el_oxi_in_struct - orig_oxi] = charge_state_probability(
            defect_el_oxi_in_struct - orig_oxi,
            defect_el_oxi_in_struct,
            1,
            max_host_oxi_magnitude,
            return_log=True,
        )

    if (
        defect.defect_type == core.DefectType.Interstitial
        and defect_el_oxi_in_struct is not None
        and defect_el_oxi_in_struct not in range(charge_state_range[0], charge_state_range[1] + 1)
    ):
        # if oxidation state of interstitial element in the host structure is not included, include it!
        possible_charge_states[defect_el_oxi_in_struct] = charge_state_probability(
            defect_el_oxi_in_struct - orig_oxi,
            defect_el_oxi_in_struct,
            1,
            max_host_oxi_magnitude,
            return_log=True,
        )

    sorted_charge_state_dict = dict(
        sorted(possible_charge_states.items(), key=lambda x: x[1]["probability"], reverse=True)
    )

    if charge_state_list := [
        k for k, v in sorted_charge_state_dict.items() if v["probability"] > probability_threshold
    ]:
        charge_state_range = (int(min(charge_state_list)), int(max(charge_state_list)))
    else:
        # if no charge states are included, take most probable (if probability > 0.1*threshold)
        charge_state_list = [
            k
            for k, v in sorted_charge_state_dict.items()
            if v["probability"] > 0.1 * probability_threshold
        ]

        most_likely_charge_state = charge_state_list[0] if charge_state_list else 0

        charge_state_range = (most_likely_charge_state, most_likely_charge_state)

    if (
        defect.defect_type == core.DefectType.Substitution
        and defect_el_oxi_in_struct is not None
        and defect_el_oxi_in_struct - orig_oxi == 0
        and (charge_state_range[0] >= 0 or charge_state_range[1] <= 0)
    ) or (charge_state_range[0] == 0 and charge_state_range[1] == 0):
        # if defect is an antisite of two equal oxi state elements, or if range is 0, ensure at least
        # (-1, 0, +1) included
        charge_state_range = (min(charge_state_range[0], -1), max(charge_state_range[1], 1))
        for charge_state, probability_dict in sorted_charge_state_dict.items():
            if charge_state in (-1, 0, 1):
                probability_dict["probability"] = 1

    # set charge_state_range to min/max of range, ensuring range is extended to 0:
    charge_state_range = (min(charge_state_range[0], 0), max(charge_state_range[1], 0))

    guessed_charge_states = list(range(charge_state_range[0], charge_state_range[1] + 1))

    for probability_dict in sorted_charge_state_dict.values():
        probability_dict["probability_threshold"] = probability_threshold

    if return_log:
        return guessed_charge_states, list(sorted_charge_state_dict.values())

    return guessed_charge_states


def get_ideal_supercell_matrix(
    structure: Structure,
    min_image_distance: float = 10.0,
    min_atoms: int = 50,
    force_cubic: bool = False,
    force_diagonal: bool = False,
    ideal_threshold: float = 0.1,
    pbar: tqdm | None = None,
) -> np.ndarray | None:
    """
    Determine the ideal supercell matrix for a given structure, based on the
    minimum image distance, minimum number of atoms and ``ideal_threshold`` for
    further expanding if a diagonal expansion of the primitive/conventional
    cell is possible.

    The ideal supercell is the smallest possible supercell which has
    a minimum image distance (i.e. minimum distance between periodic
    images of atoms/sites in a lattice) greater than
    ``min_image_distance`` (default = 10 Å - which is a typical threshold
    value used in DFT defect supercell calculations) and a number of atoms
    greater than ``min_atoms`` (default = 50). Once these criteria have
    been reached, ``doped`` will then continue searching up to supercell
    sizes (numbers of atoms) ``1 + ideal_threshold`` times larger
    (rounded up) to see if they return a diagonal expansion of the
    primitive/conventional cell (which can make later visualisation and
    analysis much easier) - if so, this larger supercell will be returned.

    This search for the ideal supercell transformation matrix is performed using
    the ``find_ideal_supercell`` function from ``doped.utils.supercells`` (see
    its docstring for more details), which efficiently scans over possible
    supercell matrices and identifies that with the minimum image distance and
    most cubic-like supercell shape. The advantage of this over that in
    ``pymatgen-analysis-defects`` is that it avoids the ``find_optimal_cell_shape``
    function from ``ASE`` (which currently does not work for rotated matrices,
    is inefficient, and optimises based on cubic-like shape rather than minimum
    image distance), giving greatly reduced supercell sizes for a given minimum
    image distance.

    If ``force_cubic`` or ``force_diagonal`` are ``True``, then the
    ``CubicSupercellTransformation`` from ``pymatgen`` is used to identify any
    simple near-cubic supercell transformations which satisfy the minimum image
    distance and atom number criteria.

    Args:
        structure (Structure):
            Primitive unit cell structure to generate supercell for.
        min_image_distance (float):
            Minimum image distance in Å of the supercell (i.e. minimum
            distance between periodic images of atoms/sites in the lattice).
            (Default = 10.0)
        min_atoms (int):
            Minimum number of atoms allowed in the supercell.
            (Default = 50)
        force_cubic (bool):
            Enforce usage of ``CubicSupercellTransformation`` from
            ``pymatgen`` for supercell generation.
            (Default = False)
        force_diagonal (bool):
            If True, return a transformation with a diagonal
            transformation matrix.
            (Default = False)
        ideal_threshold (float):
            Threshold for increasing supercell size (beyond that which satisfies
            ``min_image_distance`` and `min_atoms``) to achieve an ideal
            supercell matrix (i.e. a diagonal expansion of the primitive or
            conventional cell). Supercells up to ``1 + perfect_cell_threshold``
            times larger (rounded up) are trialled, and will instead be
            returned if they yield an ideal transformation matrix.
            (Default = 0.1; i.e. 10% larger than the minimum size)
        pbar (tqdm):
            tqdm progress bar object to update (for internal ``doped``
            usage). Default is None.

    Returns:
        Ideal supercell matrix (np.ndarray) or None if no suitable
        supercell could be found.
    """
    if force_cubic or force_diagonal:
        cst = CubicSupercellTransformation(
            min_atoms=min_atoms,
            min_length=min_image_distance,
            force_diagonal=force_diagonal,
        )

        try:
            cst.apply_transformation(structure)
            return cst.transformation_matrix

        except Exception:  # cubic supercell generation failed, used doped algorithm
            print("Could not find a suitable cubic supercell within the limits: ")
            print(f"min_atoms = {min_atoms}, min_image_distance = {min_image_distance}")
            print("Attempting doped supercell generation algorithm...")

    # get min (hypothetical) target_size from min_atoms and min_image_distance:
    min_target_size_from_atoms = int(np.ceil(min_atoms / len(structure)))
    # most efficient min_dist from volume is FCC with min_dist = lattice vector lengths = 2**(1/6) times
    # the effective cubic length (i.e. cube root of the volume), so use this to get min target_size:
    min_target_size_from_min_dist = int(
        np.ceil((min_image_distance / (2 ** (1 / 6))) ** 3 / structure.volume)
    )
    target_size = max(min_target_size_from_atoms, min_target_size_from_min_dist)
    optimal_P, best_min_dist = supercells.find_ideal_supercell(
        structure.lattice.matrix,
        target_size=target_size,
        return_min_dist=True,
    )

    while best_min_dist < min_image_distance:
        target_size += 1
        if pbar is not None:
            pbar.set_description(
                f"Best min distance: {best_min_dist:.2f} Å, trialling size = {target_size} unit cells..."
            )
        optimal_P, best_min_dist = supercells.find_ideal_supercell(
            structure.lattice.matrix,
            target_size=target_size,
            return_min_dist=True,
        )

    # check if supercell matrix is ideal (diagonal expansion of primitive or conventional cells), otherwise
    # extend search by threshold amount:
    if round(supercells._min_sum_off_diagonals(structure, optimal_P)) != 0:
        max_target_size = int(np.ceil(target_size * (1 + ideal_threshold)))
        for alt_target_size in range(target_size + 1, max_target_size + 1):
            if pbar is not None:
                pbar.set_description(
                    f"Best min distance: {best_min_dist:.2f} Å, trialling size = {alt_target_size} unit "
                    f"cells..."
                )
            alt_optimal_P, alt_best_min_dist = supercells.find_ideal_supercell(
                structure.lattice.matrix,
                target_size=alt_target_size,
                return_min_dist=True,
            )
            alt_optimal_P = supercells._check_and_return_scalar_matrix(
                alt_optimal_P, structure.lattice.matrix
            )
            if (
                (
                    alt_optimal_P[0, 0] != 0
                    and np.allclose(np.abs(alt_optimal_P / alt_optimal_P[0, 0]), np.eye(3))
                )
                or round(supercells._min_sum_off_diagonals(structure, alt_optimal_P)) == 0
            ) and alt_best_min_dist > min_image_distance:
                optimal_P = alt_optimal_P
                best_min_dist = alt_best_min_dist
                target_size = alt_target_size

    if pbar is not None:
        pbar.set_description(
            f"Best min distance: {best_min_dist:.2f} Å, with size = {target_size} unit cells"
        )

    return optimal_P


class DefectsGenerator(MSONable):
    """
    Class for generating doped DefectEntry objects.
    """

    def __init__(
        self,
        structure: Union[Structure, "Atoms", PathLike],
        extrinsic: str | list | dict | None = None,
        interstitial_coords: list | None = None,
        generate_supercell: bool = True,
        charge_state_gen_kwargs: dict | None = None,
        supercell_gen_kwargs: dict[str, int | float | bool] | None = None,
        interstitial_gen_kwargs: dict | bool | None = None,
        target_frac_coords: list | None = None,
        processes: int | None = None,
        **kwargs,
    ):
        """
        Generates ``doped`` ``DefectEntry`` objects for defects in the input
        host structure. By default, generates all intrinsic defects, but
        extrinsic defects (impurities) can also be created using the
        ``extrinsic`` argument.

        Interstitial sites are generated using Voronoi tessellation by default (found
        to be the most reliable), which can be controlled using the
        ``interstitial_gen_kwargs`` argument. Alternatively, a list of interstitial
        sites (or single interstitial site) can be manually specified using the
        ``interstitial_coords`` argument.

        By default, supercells are generated for each defect using the doped
        ``get_ideal_supercell_matrix()`` function (see docstring), with default settings
        of ``min_image_distance = 10`` (minimum distance between periodic images of 10 Å),
        ``min_atoms = 50`` (minimum 50 atoms in the supercell) and ``ideal_threshold = 0.1``
        (allow up to 10% larger supercell if it is a diagonal expansion of the primitive
        or conventional cell). This uses a custom algorithm in ``doped`` to efficiently
        search over possible supercell transformations and identify that with the minimum
        number of atoms (hence computational cost) that satisfies the minimum image distance,
        number of atoms and ``ideal_threshold`` constraints. These settings can be controlled
        by specifying keyword arguments with ``supercell_gen_kwargs``, which are passed to
        ``get_ideal_supercell_matrix()`` (e.g. for a minimum image distance of 15 Å with at
        least 100 atoms, use:
        ``supercell_gen_kwargs = {'min_image_distance': 15, 'min_atoms': 100}``). If the
        input structure already satisfies these constraints (for the same number of atoms as
        the ``doped``-generated supercell), then it will be used.
        Alternatively if ``generate_supercell = False``, then no supercell is generated
        and the input structure is used as the defect & bulk supercell. (Note this
        may give a slightly different (but fully equivalent) set of coordinates).

        The algorithm for determining defect entry names is to use the pymatgen defect
        name (e.g. ``v_Cd``, ``Cd_Te`` etc.) for vacancies/antisites/substitutions, unless
        there are multiple inequivalent sites for the defect, in which case the point
        group of the defect site is appended (e.g. ``v_Cd_Td``, ``Cd_Te_Td`` etc.), and if
        this is still not unique, then element identity and distance to the nearest
        neighbour of the defect site is appended (e.g. ``v_Cd_Td_Te2.83``, ``Cd_Te_Td_Cd2.83``
        etc.). For interstitials, the same naming scheme is used, but the point group
        is always appended to the pymatgen defect name.

        Possible charge states for the defects are estimated using the probability of
        the corresponding defect element oxidation state, the magnitude of the charge
        state, and the maximum magnitude of the host oxidation states (i.e. how
        'charged' the host is). Large (absolute) charge states, low probability
        oxidation states and/or greater charge/oxidation state magnitudes than that of
        the host are disfavoured. This can be controlled using the
        ``probability_threshold`` (default = 0.0075) or ``padding`` (default = 1) keys in
        the ``charge_state_gen_kwargs`` parameter, which are passed to the
        ``guess_defect_charge_states()`` function. The input and computed values used to
        guess charge state probabilities are provided in the
        ``DefectEntry.charge_state_guessing_log`` attributes. See docs for examples of
        modifying the generated charge states, and the docstrings of
        ``charge_state_probability()`` & ``get_vacancy_charge_states()`` for more details
        on the charge state guessing algorithm. The ``doped`` algorithm was found to give
        optimal performance in terms of efficiency and completeness (see JOSS paper), but
        of course may not be perfect in all cases, so make sure to critically consider the
        estimated charge states for your system!

        Note that Wyckoff letters can depend on the ordering of elements in the conventional
        standard structure, for which doped uses the ``spglib`` convention.

        Args:
            structure (Structure):
                Structure of the host material, either as a ``pymatgen`` ``Structure``,
                ``ASE`` ``Atoms`` or path to a structure file (e.g. ``CONTCAR``).
                If this is not the primitive unit cell, it will be reduced to the
                primitive cell for defect generation, before supercell generation.
            extrinsic (Union[str, list, dict]):
                List or dict of elements (or string for single element) to be used
                for extrinsic defect generation (i.e. dopants/impurities). If a
                list is provided, all possible substitutional defects for each
                extrinsic element will be generated. If a dict is provided, the keys
                should be the host elements to be substituted, and the values the
                extrinsic element(s) to substitute in; as a string or list.
                In both cases, all possible extrinsic interstitials are generated.
            interstitial_coords (list):
                List of fractional coordinates (corresponding to the input structure),
                or a single set of fractional coordinates, to use as interstitial
                defect site(s). Default (when interstitial_coords not specified) is
                to automatically generate interstitial sites using Voronoi tessellation.
                The input interstitial_coords are converted to
                ``DefectsGenerator.prim_interstitial_coords``, which are the corresponding
                fractional coordinates in ``DefectsGenerator.primitive_structure`` (which
                is used for defect generation), along with the multiplicity and
                equivalent coordinates, sorted according to the doped convention.
            generate_supercell (bool):
                Whether to generate a supercell for the output defect entries
                (using the custom algorithm in ``doped`` which efficiently searches over
                possible supercell transformations and identifies that with the minimum
                number of atoms (hence computational cost) that satisfies the minimum
                image distance, number of atoms and ``ideal_threshold`` constraints
                - which can be controlled with ``supercell_gen_kwargs``).
                If False, then the input structure is used as the defect & bulk supercell.
                (Note this may give a slightly different (but fully equivalent) set of coordinates).
            charge_state_gen_kwargs (dict):
                Keyword arguments to be passed to the ``guess_defect_charge_states``
                function (such as ``probability_threshold`` (default = 0.0075, used for
                substitutions and interstitials) and ``padding`` (default = 1, used for
                vacancies)) to control defect charge state generation.
            supercell_gen_kwargs (dict):
                Keyword arguments to be passed to the ``get_ideal_supercell_matrix``
                function (such as ``min_image_distance`` (default = 10), ``min_atoms``
                (default = 50), ``ideal_threshold`` (default = 0.1), ``force_cubic``
                - which enforces a (near-)cubic supercell output (default = False),
                or ``force_diagonal`` (default = False)).
            interstitial_gen_kwargs (dict, bool):
                Keyword arguments to be passed to ``get_Voronoi_interstitial_sites``
                (such as ``min_dist`` (0.9 Å), ``clustering_tol`` (0.8 Å),
                ``symmetry_preference`` (0.1 Å), ``stol`` (0.32), ``tight_stol`` (0.02)
                and ``symprec`` (0.01)  -- see its docstring, parentheses indicate
                default values), or ``InterstitialGenerator`` if ``interstitial_coords``
                is specified. If set to ``False``, interstitial generation will be
                skipped entirely.
            target_frac_coords (list):
                Defects are placed at the closest equivalent site to these fractional
                coordinates in the generated supercells. Default is [0.5, 0.5, 0.5]
                if not set (i.e. the supercell centre, to aid visualisation).
            processes (int):
                Number of processes to use for multiprocessing. If not set, defaults to
                one less than the number of CPUs available.
            **kwargs:
                Additional keyword arguments for defect generation. Options:
                ``{defect}_elements`` where ``{defect}`` is ``vacancy``, ``substitution``,
                or ``interstitial``, in which cases only those defects of the specified
                elements will be generated (where ``{defect}_elements`` is a list of
                element symbol strings). Setting ``{defect}_elements`` to an empty list
                will skip defect generation for that defect type entirely.
                ``{defect}_charge_states`` to specify the charge states to use for all
                defects of that type (as a list of integers).
                ``neutral_only`` to only generate neutral charge states.

        Attributes:
            defect_entries (dict): Dictionary of {defect_species: DefectEntry} for all
                defect entries (with charge state and supercell properties) generated.
            defects (dict): Dictionary of {defect_type: [Defect, ...]} for all defect
                objects generated.
            primitive_structure (Structure): Primitive cell structure of the host
                used to generate defects.
            supercell_matrix (Matrix): Matrix to generate defect/bulk supercells from
                the primitive cell structure.
            bulk_supercell (Structure): Supercell structure of the host
                (equal to primitive_structure * supercell_matrix).
            conventional_structure (Structure): Conventional cell structure of the
                host according to the Bilbao Crystallographic Server (BCS) definition,
                used to determine defect site Wyckoff labels and multiplicities.
            prim_interstitial_coords (list):
                List of interstitial coordinates in the primitive cell structure.

            ``DefectsGenerator`` input parameters are also set as attributes.
        """
        self.defects: dict[str, list[Defect]] = {}  # {defect_type: [Defect, ...]}
        self.defect_entries: dict[str, DefectEntry] = {}  # {defect_species: DefectEntry}
        if isinstance(structure, str | PathLike):
            structure = Structure.from_file(structure)
        elif not isinstance(structure, Structure):
            structure = Structure.from_ase_atoms(structure)

        self.structure = structure
        self.extrinsic = extrinsic if extrinsic is not None else []
        self.kwargs = kwargs
        if isinstance(self.kwargs, dict):
            for kwarg, val in self.kwargs.items():
                if isinstance(val, set):
                    self.kwargs[kwarg] = list(val)  # convert sets to lists for JSON serialisation

        if interstitial_coords is not None:
            # if a single list or array, convert to list of lists
            self.interstitial_coords = (
                interstitial_coords
                if isinstance(interstitial_coords[0], list | tuple | np.ndarray)
                else [interstitial_coords]  # ensure list of lists
            )
        else:
            self.interstitial_coords = []

        self.prim_interstitial_coords = None
        self.generate_supercell = generate_supercell
        self.charge_state_gen_kwargs = (
            charge_state_gen_kwargs if charge_state_gen_kwargs is not None else {}
        )
        self.supercell_gen_kwargs: dict[str, int | float | bool] = {
            "min_image_distance": 10.0,  # same as current pymatgen-analysis-defects `min_length` ( = 10)
            "min_atoms": 50,  # different from current pymatgen-analysis-defects `min_atoms` ( = 80)
            "ideal_threshold": 0.1,
            "force_cubic": False,
            "force_diagonal": False,
        }
        self.supercell_gen_kwargs.update(supercell_gen_kwargs if supercell_gen_kwargs is not None else {})
        self.interstitial_gen_kwargs: dict[str, int | float | bool] | bool = (
            interstitial_gen_kwargs if interstitial_gen_kwargs is not None else {}
        )
        self.target_frac_coords = target_frac_coords if target_frac_coords is not None else [0.5, 0.5, 0.5]
        specified_min_image_distance = self.supercell_gen_kwargs["min_image_distance"]

        if len(self.structure) == 1 and not self.generate_supercell:
            # raise error if only one atom in primitive cell and no supercell generated, as vacancy will
            # give empty structure
            raise ValueError(
                "Input structure has only one site, so cannot generate defects without supercell (i.e. "
                "with generate_supercell=False)! Vacancy defect will give empty cell!"
            )

        # use lru_cache for Composition and PeriodicSite comparisons (speeds up structure matching
        # dramatically), and for Structure as well as fast ``doped`` ``__eq__`` function
        Composition.__instances__ = {}
        Composition.__eq__ = doped_Composition.__eq__
        Composition.__hash__ = doped_Composition.__hash__
        PeriodicSite.__eq__ = doped_PeriodicSite.__eq__
        PeriodicSite.__hash__ = doped_PeriodicSite.__hash__
        IStructure.__instances__ = {}
        IStructure.__eq__ = doped_IStructure.__eq__

        pbar = tqdm(
            total=100, bar_format="{desc}{percentage:.1f}%|{bar}| [{elapsed},  {rate_fmt}{postfix}]"
        )  # tqdm progress
        # bar. 100% is completion
        pbar.set_description("Getting primitive structure")

        try:  # put code in try/except block so progress bar always closed if interrupted
            # Reduce structure to primitive cell for efficient defect generation
            # same symprec as defect generators in pymatgen-analysis-defects:
            sga, symprec = symmetry.get_sga(self.structure, return_symprec=True)
            if sga.get_space_group_number() == 1:  # print sanity check message
                print(
                    "Note that the detected symmetry of the input structure is P1 (i.e. only "
                    "translational symmetry). If this is not expected (i.e. host system is not "
                    "disordered/defective), then you should check your input structure!"
                )
            if symprec != 0.01:  # default
                warnings.warn(
                    f"\nSymmetry determination failed for the default symprec value of 0.01, "
                    f"but succeeded with symprec = {symprec}, which will be used for symmetry "
                    f"determination functions here."
                )

            prim_struct = symmetry.get_primitive_structure(self.structure, symprec=symprec)
            primitive_structure = symmetry._round_struct_coords(
                prim_struct if prim_struct.num_sites < self.structure.num_sites else self.structure,
                to_unit_cell=True,  # wrap to unit cell
            )  # if primitive cell is the same as input structure, use input structure to avoid rotations

            pbar.update(5)  # 5% of progress bar

            # check if input structure is already greater than ``min_image_distance`` Å in each direction:
            input_min_image_distance = supercells.get_min_image_distance(self.structure)

            if self.generate_supercell:
                # Generate supercell once, so this isn't redundantly rerun for each defect, and ensures the
                # same supercell is used for each defect and bulk calculation
                pbar.set_description("Generating simulation supercell")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="The 'warn' method is deprecated")
                    supercell_matrix = get_ideal_supercell_matrix(
                        structure=primitive_structure,
                        pbar=pbar,
                        **self.supercell_gen_kwargs,  # type: ignore
                    )

            if not self.generate_supercell or (
                input_min_image_distance >= specified_min_image_distance
                and (primitive_structure * supercell_matrix).num_sites
                >= self.structure.num_sites
                >= self.supercell_gen_kwargs["min_atoms"]
            ):
                if input_min_image_distance < 10:
                    # input structure is <10 Å in at least one direction, and generate_supercell=False,
                    # so use input structure but warn user:
                    warnings.warn(
                        f"\nInput structure is <10 Å in at least one direction (minimum image distance = "
                        f"{input_min_image_distance:.2f} Å, which is usually too "
                        f"small for accurate defect calculations, but generate_supercell = False, so "
                        f"using input structure as defect & bulk supercells. Caution advised!"
                    )

                # ``generate_supercell=False`` or input structure has fewer or same number of atoms as
                # doped supercell, so use input structure:
                (
                    self.primitive_structure,
                    self.supercell_matrix,
                ) = symmetry._get_supercell_matrix_and_possibly_redefine_prim(
                    primitive_structure, self.structure, sga=sga
                )

                self.primitive_structure, self._T = symmetry.get_clean_structure(
                    self.primitive_structure, return_T=True
                )  # T maps orig prim struct to new prim struct; T * Orig = New -> Orig = T^-1 * New
                # supercell matrix P was: P * Orig = Super -> P * T^-1 * New = Super -> P' = P * T^-1

                self.supercell_matrix = np.matmul(self.supercell_matrix, np.linalg.inv(self._T))

            else:
                self.primitive_structure = primitive_structure
                self.supercell_matrix = supercell_matrix

            self.supercell_matrix = np.rint(self.supercell_matrix).astype(int)  # round to nearest integer
            self.primitive_structure = Structure.from_sites(
                [site.to_unit_cell() for site in self.primitive_structure]
            )

            # get oxidation states:
            self._bulk_oxi_states: Structure | Composition | dict | bool = False
            # if input structure was oxi-state-decorated, use these oxi states for defect generation:
            if all(hasattr(site.specie, "oxi_state") for site in self.structure.sites) and all(
                isinstance(site.specie.oxi_state, int | float) for site in self.structure.sites
            ):
                self._bulk_oxi_states = self.primitive_structure

            else:  # guess & set oxidation states now, to speed up oxi state handling in defect generation
                pbar.set_description("Guessing oxidation states")
                if prim_struct_w_oxi := guess_and_set_oxi_states_with_timeout(self.primitive_structure):
                    self.primitive_structure = self._bulk_oxi_states = prim_struct_w_oxi
                else:
                    warnings.warn(
                        "\nOxidation states could not be guessed for the input structure. This is "
                        "required for charge state guessing, so defects will still be generated but "
                        "all charge states will be set to -1, 0, +1. You can manually edit these "
                        "with the add/remove_charge_states methods (see tutorials), or you can set "
                        "the oxidation states of the input structure (e.g. using "
                        "structure.add_oxidation_state_by_element()) and re-initialize "
                        "DefectsGenerator()."
                    )

            self.bulk_supercell = symmetry._round_struct_coords(
                (self.primitive_structure * self.supercell_matrix).get_sorted_structure(),
                to_unit_cell=True,
            )
            if not generate_supercell:  # re-order bulk supercell to match that of input supercell
                self.bulk_supercell = reorder_s1_like_s2(self.bulk_supercell, self.structure)

            # get and round (to avoid tiny mismatches, due to rounding in search functions,
            # flagging issues) min image distance of supercell:
            self.min_image_distance = np.round(supercells.get_min_image_distance(self.bulk_supercell), 3)

            # check that generated supercell is greater than ``min_image_distance``` Å in each direction:
            if self.min_image_distance < specified_min_image_distance and self.generate_supercell:
                raise ValueError(
                    f"Error in supercell generation! Auto-generated supercell is less than chosen minimum "
                    f"image distance ({specified_min_image_distance:.2f} Å) in at least one direction ("
                    f"minimum image distance = {self.min_image_distance:.2f} Å), which should not happen. "
                    f"If you used force_cubic or force_diagonal, you may need to relax these constraints "
                    f"to find an appropriate supercell - otherwise please report this to the developers!"
                )

            pbar.update(10)  # 15% of progress bar

            # Generate defects
            # Vacancies:
            pbar.set_description("Generating vacancies")
            vac_generator_obj = VacancyGenerator()
            vac_generator = vac_generator_obj.generate(
                self.primitive_structure, oxi_state=0, rm_species=self.kwargs.get("vacancy_elements", None)
            )  # set oxi_state using doped functions; more robust and efficient
            self.defects["vacancies"] = [
                Vacancy._from_pmg_defect(vac, bulk_oxi_states=self._bulk_oxi_states)
                for vac in vac_generator
            ]
            pbar.update(5)  # 20% of progress bar

            # determine which, if any, extrinsic elements are present:
            if isinstance(self.extrinsic, str):
                extrinsic_elements = [self.extrinsic]
            elif isinstance(self.extrinsic, list):
                extrinsic_elements = self.extrinsic
            elif isinstance(self.extrinsic, dict):  # dict of host: extrinsic elements, as lists or strings
                # convert to flattened list of extrinsic elements:
                extrinsic_elements = list(
                    chain(*[i if isinstance(i, list) else [i] for i in self.extrinsic.values()])
                )
                extrinsic_elements = list(set(extrinsic_elements))  # get only unique elements
            else:
                extrinsic_elements = []

            host_element_list = [el.symbol for el in self.primitive_structure.composition.elements]
            # if any "extrinsic" elements are actually host elements, remove them and warn user:
            if any(el in host_element_list for el in extrinsic_elements) and not isinstance(
                self.extrinsic, dict  # don't warn if ``extrinsic`` was supplied as a dict
            ):
                warnings.warn(
                    f"\nSpecified 'extrinsic' elements "
                    f"{[el for el in extrinsic_elements if el in host_element_list]} are present in "
                    f"the host structure, so do not need to be specified as 'extrinsic' in "
                    f"DefectsGenerator(). These will be ignored."
                )

            # sort extrinsic elements by periodic group and atomic number for deterministic ordering:
            extrinsic_elements = sorted(
                [el for el in extrinsic_elements if el not in host_element_list],
                key=_element_sort_func,
            )

            # Antisites:
            if self.kwargs.get("substitution_elements", False) == []:  # skip substitutions
                pbar.update(10)  # 30% of progress bar
            else:
                pbar.set_description("Generating substitutions")
                antisite_generator_obj = AntiSiteGenerator()
                as_generator = antisite_generator_obj.generate(self.primitive_structure, oxi_state=0)
                self.defects["substitutions"] = [
                    Substitution._from_pmg_defect(anti, bulk_oxi_states=self._bulk_oxi_states)
                    for anti in as_generator
                ]
                pbar.update(5)  # 25% of progress bar

                # Substitutions:
                substitution_generator_obj = SubstitutionGenerator()
                if isinstance(self.extrinsic, str | list):  # substitute all host elements:
                    substitutions = {
                        el.symbol: extrinsic_elements
                        for el in self.primitive_structure.composition.elements
                    }
                elif isinstance(self.extrinsic, dict):  # substitute only specified host elements
                    substitutions = self.extrinsic
                else:
                    warnings.warn(
                        f"Invalid `extrinsic` defect input. Got type {type(self.extrinsic)}, but string "
                        f"or list or dict required. No extrinsic defects will be generated."
                    )
                    substitutions = {}

                if substitutions:
                    sub_generator = substitution_generator_obj.generate(
                        self.primitive_structure, substitution=substitutions, oxi_state=0
                    )
                    sub_defects = [
                        Substitution._from_pmg_defect(sub, bulk_oxi_states=self._bulk_oxi_states)
                        for sub in sub_generator
                    ]
                    if "substitutions" in self.defects:
                        self.defects["substitutions"].extend(sub_defects)
                    else:
                        self.defects["substitutions"] = sub_defects

                if sub_elts := self.kwargs.get("substitution_elements", False):
                    # filter out substitutions for elements not in ``substitution_elements``:
                    self.defects["substitutions"] = [
                        sub
                        for sub in self.defects["substitutions"]
                        if any(sub.name.startswith(sub_elt) for sub_elt in sub_elts)
                    ]

                if not self.defects[
                    "substitutions"
                ]:  # no substitutions, single-element system, no extrinsic
                    del self.defects["substitutions"]  # remove empty list
                pbar.update(5)  # 30% of progress bar

            # Interstitials:
            self._element_list = host_element_list + extrinsic_elements  # all elements in system
            if (
                self.interstitial_gen_kwargs is not False
                and self.kwargs.get("interstitial_elements", True) != []
            ):  # skip interstitials
                self.interstitial_gen_kwargs = (
                    self.interstitial_gen_kwargs if isinstance(self.interstitial_gen_kwargs, dict) else {}
                )
                pbar.set_description("Generating interstitials")
                if self.interstitial_coords:
                    # map interstitial coords to primitive structure, and get multiplicities
                    symm_ops = sga.get_symmetry_operations(cartesian=False)
                    self.prim_interstitial_coords = []

                    for interstitial_frac_coords in self.interstitial_coords:
                        equiv_prim_coords = symmetry.get_equiv_frac_coords_in_primitive(
                            interstitial_frac_coords,
                            self.structure,
                            self.primitive_structure,
                            symm_ops,
                            equiv_coords=True,
                        )
                        self.prim_interstitial_coords.append(
                            (equiv_prim_coords[0], len(equiv_prim_coords), equiv_prim_coords)
                        )

                    sorted_sites_mul_and_equiv_fpos = self.prim_interstitial_coords

                else:
                    # Generate interstitial sites using Voronoi tessellation
                    sorted_sites_mul_and_equiv_fpos = get_Voronoi_interstitial_sites(
                        host_structure=self.primitive_structure,
                        interstitial_gen_kwargs=self.interstitial_gen_kwargs,
                    )

                self.defects["interstitials"] = []
                ig = InterstitialGenerator(self.interstitial_gen_kwargs.get("min_dist", 0.9))
                cand_sites, multiplicity, equiv_fpos = zip(*sorted_sites_mul_and_equiv_fpos, strict=False)
                for el in self.kwargs.get("interstitial_elements", self._element_list):
                    inter_generator = ig.generate(
                        self.primitive_structure,
                        insertions={el: cand_sites},
                        multiplicities={el: multiplicity},
                        equivalent_positions={el: equiv_fpos},
                        oxi_state=0,
                    )
                    self.defects["interstitials"].extend(
                        [
                            Interstitial._from_pmg_defect(inter, bulk_oxi_states=self._bulk_oxi_states)
                            for inter in inter_generator
                        ]
                    )

                    # check if any manually-specified interstitials were skipped due to min_dist and
                    # warn user:
                    if self.interstitial_coords and len(self.interstitial_coords) > len(
                        self.defects["interstitials"]
                    ):
                        warnings.warn(
                            f"\nNote that some manually-specified interstitial sites were skipped due to "
                            f"being too close to host lattice sites (minimum distance = `min_dist` = "
                            f"{self.interstitial_gen_kwargs.get('min_dist', 0.9):.2f} Å). If for some "
                            f"reason you still want to include these sites, you can adjust `min_dist` ("
                            f"default = 0.9 Å), or just use the default Voronoi tessellation algorithm "
                            f"for generating interstitials (by not setting the `interstitial_coords` "
                            f"argument)."
                        )

            pbar.update(15)  # 45% of progress bar, generating interstitials typically takes the longest

            # Generate DefectEntry objects:
            pbar.set_description("Determining Wyckoff sites")
            defect_list: list[Defect] = reduce(operator.iconcat, self.defects.values(), [])
            num_defects = len(defect_list)

            # get BCS conventional structure and lattice vector swap array:
            (
                self.conventional_structure,
                self._BilbaoCS_conv_cell_vector_mapping,
                wyckoff_label_dict,
            ) = symmetry.get_BCS_conventional_structure(
                self.primitive_structure, pbar=pbar, return_wyckoff_dict=True
            )

            conv_sga = symmetry.get_sga(self.conventional_structure)
            conv_symm_ops = conv_sga.get_symmetry_operations(cartesian=False)

            # process defects into defect entries:
            partial_func = partial(
                _get_neutral_defect_entry,
                supercell_matrix=self.supercell_matrix,
                target_frac_coords=self.target_frac_coords,
                bulk_supercell=self.bulk_supercell,
                conventional_structure=self.conventional_structure,
                _BilbaoCS_conv_cell_vector_mapping=self._BilbaoCS_conv_cell_vector_mapping,
                wyckoff_label_dict=wyckoff_label_dict,
                conv_symm_ops=conv_symm_ops,
            )

            if not isinstance(pbar, MagicMock):  # to allow tqdm to be mocked for testing
                _pbar_increment_per_defect = max(
                    0, min((1 / num_defects) * ((pbar.total * 0.9) - pbar.n), pbar.total - pbar.n)
                )  # up to 90% of progress bar
            else:
                _pbar_increment_per_defect = 0

            defect_entry_list = []
            if len(self.primitive_structure) > 8 and processes != 1:
                # skip for small systems as comms overhead / process init outweighs speedup
                with pool_manager(processes) as pool:
                    for result in pool.imap_unordered(partial_func, defect_list):
                        defect_entry_list.append(result)
                        pbar.update(_pbar_increment_per_defect)  # 90% of progress bar

            else:
                for defect in defect_list:
                    defect_entry_list.append(partial_func(defect))
                    pbar.update(_pbar_increment_per_defect)  # 90% of progress bar

            pbar.set_description("Generating DefectEntry objects")
            # sort defect_entry_list by _frac_coords_sort_func applied to the conv_cell_frac_coords,
            # in order for naming and defect generation output info to be deterministic
            defect_entry_list.sort(key=lambda x: symmetry._frac_coords_sort_func(x.conv_cell_frac_coords))

            # redefine defects dict with DefectEntry.defect objects (as attributes have been updated in
            # _get_neutral_defect_entry):
            self.defects = {
                "vacancies": [
                    defect_entry.defect
                    for defect_entry in defect_entry_list
                    if _defect_dict_key_from_pmg_type(defect_entry.defect.defect_type) == "vacancies"
                ],
                "substitutions": [
                    defect_entry.defect
                    for defect_entry in defect_entry_list
                    if _defect_dict_key_from_pmg_type(defect_entry.defect.defect_type) == "substitutions"
                ],
                "interstitials": [
                    defect_entry.defect
                    for defect_entry in defect_entry_list
                    if _defect_dict_key_from_pmg_type(defect_entry.defect.defect_type) == "interstitials"
                ],
            }
            # remove empty defect lists: (e.g. single-element systems with no antisite substitutions)
            self.defects = {k: v for k, v in self.defects.items() if v}

            prim_sga = symmetry.get_sga(self.primitive_structure)
            prim_symm_ops = prim_sga.get_symmetry_operations(cartesian=False)
            named_defect_dict = name_defect_entries(
                defect_entry_list, element_list=self._element_list, symm_ops=prim_symm_ops
            )
            pbar.update(5)  # 95% of progress bar
            if not isinstance(pbar, MagicMock):
                _pbar_increment_per_defect = max(
                    0, min((1 / num_defects) * (pbar.total - pbar.n) * 0.999, pbar.total - pbar.n)
                )  # multiply by 0.999 to avoid rounding errors, overshooting the 100% limit and getting
                # warnings from tqdm

            Structure.__deepcopy__ = lambda x, y: x.copy()  # faster deepcopying, shallow copy fine
            for defect_name_wout_charge, neutral_defect_entry in named_defect_dict.items():
                type_name = _defect_type_key_from_pmg_type(neutral_defect_entry.defect.defect_type)
                if self.kwargs.get("neutral_only", False):
                    charge_states = [
                        0,
                    ]  # only neutral
                    neutral_defect_entry.charge_state_guessing_log = {}

                elif charge_states := self.kwargs.get(f"{type_name}_charge_states", []):
                    neutral_defect_entry.charge_state_guessing_log = {}

                elif self._bulk_oxi_states is not False:
                    charge_state_guessing_output = guess_defect_charge_states(
                        neutral_defect_entry.defect, return_log=True, **self.charge_state_gen_kwargs
                    )
                    charge_state_guessing_output = cast(
                        tuple[list[int], list[dict]], charge_state_guessing_output
                    )  # for correct type checking; guess_defect_charge_states can return different types
                    # depending on return_log
                    charge_states, charge_state_guessing_log = charge_state_guessing_output
                    neutral_defect_entry.charge_state_guessing_log = charge_state_guessing_log

                else:
                    charge_states = [-1, 0, 1]  # no oxi states, so can't guess charge states
                    neutral_defect_entry.charge_state_guessing_log = {}

                for charge in charge_states:
                    defect_entry = (
                        copy.deepcopy(neutral_defect_entry) if charge != 0 else neutral_defect_entry
                    )
                    defect_entry.charge_state = charge
                    # set name attribute:
                    defect_entry.name = f"{defect_name_wout_charge}_{'+' if charge > 0 else ''}{charge}"
                    self.defect_entries[defect_entry.name] = defect_entry

                pbar.update(_pbar_increment_per_defect)  # 100% of progress bar

            # sort defects and defect entries for deterministic behaviour:
            self.defects = _sort_defects(self.defects, element_list=self._element_list)
            self.defect_entries = sort_defect_entries(
                self.defect_entries, element_list=self._element_list
            )  # type:ignore

            if not isinstance(pbar, MagicMock) and pbar.total - pbar.n > 0:
                pbar.update(pbar.total - pbar.n)  # 100%

        except Exception as e:
            pbar.close()
            raise e

        finally:
            pbar.close()

        self.defect_generator_info()

    def defect_generator_info(self):
        """
        Prints information about the defects that have been generated.
        """
        return print(self._defect_generator_info())

    def _defect_generator_info(self):
        """
        Returns a string with information about the defects that have been
        generated by the DefectsGenerator.
        """
        info_string = ""
        for defect_class, defect_list in self.defects.items():
            if len(defect_list) > 0:
                table = []
                header = [
                    defect_class.capitalize(),
                    "Guessed Charges",
                    "Conv. Cell Coords",
                    "Wyckoff",
                ]
                defect_type = defect_list[0].defect_type
                matching_defect_types = {
                    defect_entry_name: defect_entry
                    for defect_entry_name, defect_entry in self.defect_entries.items()
                    if defect_entry.defect.defect_type == defect_type
                }
                seen = set()
                matching_type_names_wout_charge = [
                    defect_entry_name.rsplit("_", 1)[0]
                    for defect_entry_name in matching_defect_types
                    if defect_entry_name.rsplit("_", 1)[0] not in seen
                    and not seen.add(defect_entry_name.rsplit("_", 1)[0])  # track unique defect names
                    # w/out charge
                ]
                for defect_name in matching_type_names_wout_charge:
                    charges = [
                        name.rsplit("_", 1)[1]
                        for name in self.defect_entries
                        if _check_if_name_subset(name, defect_name)
                    ]  # so e.g. Te_i_m1 doesn't match with Te_i_m1b
                    # convert list of strings to one string with comma-separated charges
                    charges = "[" + ",".join(charges) + "]"
                    defect_entry = next(
                        entry
                        for name, entry in self.defect_entries.items()
                        if _check_if_name_subset(name, defect_name)
                    )
                    frac_coords_string = (
                        "N/A"
                        if defect_entry.conv_cell_frac_coords is None
                        else ",".join(f"{x:.3f}" for x in defect_entry.conv_cell_frac_coords)
                    )
                    row = [
                        defect_name,
                        charges,
                        f"[{frac_coords_string}]",
                        defect_entry.wyckoff,
                    ]
                    table.append(row)
                info_string += (
                    tabulate(
                        table,
                        headers=header,
                        stralign="left",
                        numalign="left",
                    )
                    + "\n\n"
                )
        conventional_cell_comp = self.conventional_structure.composition
        formula, fu = conventional_cell_comp.get_reduced_formula_and_factor(iupac_ordering=True)
        info_string += (
            "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in the "
            f"conventional ('conv.') unit cell, which comprises {fu} formula unit(s) of {formula}.\n"
        )

        return info_string

    def _process_name_and_charge_states_and_get_matching_entries(
        self,
        defect_entry_name: str,
        charge_states: list | int,
        match_charge_states: bool = True,
    ) -> tuple[list, list]:
        if defect_entry_name[-1].isdigit():  # if defect entry name ends with number:
            defect_entry_name = defect_entry_name.rsplit("_", 1)[0]  # name without charge

        if isinstance(charge_states, int | float):
            charge_states = [round(charge_states)]

        matching_entry_names_wout_charge = [
            name for name in self.defect_entries if _check_if_name_subset(name, defect_entry_name)
        ]
        if not match_charge_states:
            return charge_states, matching_entry_names_wout_charge

        return charge_states, [
            name
            for charge in charge_states
            for name in matching_entry_names_wout_charge
            if name.endswith(f"_{'+' if charge > 0 else ''}{charge}")
        ]

    def add_charge_states(self, defect_entry_name: str, charge_states: list | int):
        r"""
        Add additional ``DefectEntry``\s with the specified charge states to
        ``self.defect_entries``.

        Args:
            defect_entry_name (str):
                Name of defect entry to add charge states to.
                Doesn't need to include the charge state.
            charge_states (list):
                List of charge states to add to defect entry
                (e.g. [-2, -3]), or a single charge state (e.g. -2).
        """
        charge_states, matching_entry_names_wout_charge = (
            self._process_name_and_charge_states_and_get_matching_entries(
                defect_entry_name, charge_states, match_charge_states=False
            )
        )
        # get unique defect entry names without charge state:
        unique_matching_entry_names_wout_charge = {
            i.rsplit("_", 1)[0] for i in matching_entry_names_wout_charge
        }

        Structure.__deepcopy__ = lambda x, y: x.copy()  # faster deepcopying, shallow copy fine
        for defect_entry_name_wout_charge in unique_matching_entry_names_wout_charge:
            previous_defect_entry = next(
                entry
                for name, entry in self.defect_entries.items()
                if _check_if_name_subset(name, defect_entry_name_wout_charge)
            )
            for charge in charge_states:
                defect_entry = copy.deepcopy(previous_defect_entry)
                defect_entry.charge_state = charge
                defect_entry.name = (
                    f"{defect_entry.name.rsplit('_', 1)[0]}_{'+' if charge > 0 else ''}{charge}"
                )
                self.defect_entries[defect_entry.name] = defect_entry

        # sort defects and defect entries for deterministic behaviour:
        self.defects = _sort_defects(self.defects, element_list=self._element_list)
        self.defect_entries = sort_defect_entries(
            self.defect_entries, element_list=self._element_list
        )  # type:ignore

    def remove_charge_states(self, defect_entry_name: str, charge_states: list | int):
        r"""
        Remove ``DefectEntry``\s with the specified charge states from
        ``self.defect_entries``.

        Args:
            defect_entry_name (str):
                Name of defect entry to remove charge states from.
                Doesn't need to include the charge state.
            charge_states (list):
                List of charge states to add to defect entry
                (e.g. [-2, -3]), or a single charge state (e.g. -2).
        """
        charge_states, matching_entry_names = (
            self._process_name_and_charge_states_and_get_matching_entries(defect_entry_name, charge_states)
        )
        for defect_entry_name_to_remove in matching_entry_names:
            del self.defect_entries[defect_entry_name_to_remove]

        # sort defects and defect entries for deterministic behaviour:
        self.defects = _sort_defects(self.defects, element_list=self._element_list)
        self.defect_entries = sort_defect_entries(
            self.defect_entries, element_list=self._element_list
        )  # type:ignore

    def as_dict(self):
        """
        JSON-serializable dict representation of DefectsGenerator.
        """
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            **self.__dict__,
        }

    @classmethod
    def from_dict(cls, d):
        """
        Reconstructs ``DefectsGenerator`` object from a dict representation
        created using ``DefectsGenerator.as_dict()``.

        Args:
            d (dict): dict representation of ``DefectsGenerator``.

        Returns:
            ``DefectsGenerator`` object
        """

        def process_attributes(attributes, iterable):
            result = {}
            for attr in attributes:
                result[attr] = MontyDecoder().process_decoded(iterable.pop(attr))
            return result

        def decode_dict(iterable):
            if isinstance(iterable, dict) and "@module" in iterable:
                class_name = iterable["@class"]

                defect_additional_attributes = [
                    "conventional_structure",
                    "conv_cell_frac_coords",
                    "equiv_conv_cell_frac_coords",
                    "_BilbaoCS_conv_cell_vector_mapping",
                    "wyckoff",
                ]
                attribute_groups = {
                    "DefectEntry": [
                        "conventional_structure",
                        "conv_cell_frac_coords",
                        "equiv_conv_cell_frac_coords",
                        "_BilbaoCS_conv_cell_vector_mapping",
                        "wyckoff",
                        "charge_state_guessing_log",
                        "defect_supercell",
                        "defect_supercell_site",
                        "equivalent_supercell_sites",
                        "bulk_supercell",
                        "name",
                    ],
                    **{
                        k: defect_additional_attributes
                        for k in [
                            "Interstitial",
                            "Substitution",
                            "Vacancy",
                            "Defect",
                            "DefectComplex",
                            "Adsorbate",
                        ]
                    },
                }

                if class_name in attribute_groups:
                    # pull attributes not in __init__ signature and define after object creation
                    attributes = process_attributes(attribute_groups[class_name], iterable)
                    if class_name == "DefectEntry":
                        attributes["defect"] = decode_dict(iterable["defect"])
                    decoded_obj = MontyDecoder().process_decoded(iterable)
                    for attr, value in attributes.items():
                        setattr(decoded_obj, attr, value)

                    return decoded_obj

                return MontyDecoder().process_decoded(iterable)

            if isinstance(iterable, dict):
                return {k: decode_dict(v) for k, v in iterable.items()}

            if isinstance(iterable, list):
                return [decode_dict(v) for v in iterable]

            return iterable

        # recursively decode nested dicts (in dicts or lists) with @module key
        d_decoded = {k: decode_dict(v) for k, v in d.items()}
        defects_generator = cls.__new__(
            cls
        )  # Create new DefectsGenerator object without invoking __init__

        # set the instance variables directly from the dictionary
        for key, value in d_decoded.items():
            if key not in ["@module", "@class", "@version"]:
                setattr(defects_generator, key, value)

        return defects_generator

    def to_json(self, filename: PathLike | None = None):
        """
        Save the ``DefectsGenerator`` object as a json file, which can be
        reloaded with the ``DefectsGenerator.from_json()`` class method.

        Note that file extensions with ".gz" will be automatically compressed
        (recommended to save space)!

        Args:
            filename (PathLike):
                Filename to save json file as. If None, the filename will be
                set as ``{Chemical Formula}_defects_generator.json.gz`` where
                {Chemical Formula} is the chemical formula of the host material.
        """
        if filename is None:
            formula = self.primitive_structure.composition.get_reduced_formula_and_factor(
                iupac_ordering=True
            )[0]
            filename = f"{formula}_defects_generator.json.gz"

        dumpfn(self, filename)

    @classmethod
    def from_json(cls, filename: PathLike):
        """
        Load a ``DefectsGenerator`` object from a json(.gz) file.

        Note that ``.json.gz`` files can be loaded directly.

        Args:
            filename (PathLike):
                Filename of json file to load ``DefectsGenerator``
                object from.

        Returns:
            ``DefectsGenerator`` object
        """
        return loadfn(filename)

    def __getattr__(self, attr):
        """
        Redirects an unknown attribute/method call to the ``defect_entries``
        dictionary attribute, if the attribute doesn't exist in
        ``DefectsGenerator``.
        """
        try:
            super().__getattribute__(attr)
        except AttributeError as exc:
            if attr == "defect_entries":
                raise exc
            return getattr(self.defect_entries, attr)

    def __getitem__(self, key):
        """
        Makes ``DefectsGenerator`` object subscriptable, so that it can be
        indexed like a dictionary, using the ``defect_entries`` dictionary
        attribute.
        """
        return self.defect_entries[key]

    def __setitem__(self, key, value):
        """
        Set the value of a specific key (defect name) in the ``defect_entries``
        dictionary.

        Also adds the corresponding defect to the self.defects dictionary, if
        it doesn't already exist.
        """
        # check the input, must be a DefectEntry object, with same supercell and primitive structure
        if not isinstance(value, DefectEntry | thermo.DefectEntry):
            raise TypeError(f"Value must be a DefectEntry object, not {type(value).__name__}")

        # compare structures without oxidation states:
        defect_struc_wout_oxi = value.defect.structure.copy()
        defect_struc_wout_oxi.remove_oxidation_states()
        prim_struc_wout_oxi = self.primitive_structure.copy()
        prim_struc_wout_oxi.remove_oxidation_states()

        if defect_struc_wout_oxi != prim_struc_wout_oxi:
            raise ValueError(
                f"Value must have the same primitive structure as the DefectsGenerator object, "
                f"instead has: {value.defect.structure} while DefectsGenerator has: "
                f"{prim_struc_wout_oxi}"
            )

        # check supercell
        defect_supercell = value.defect.get_supercell_structure(
            sc_mat=self.supercell_matrix,
            dummy_species="X",  # keep track of the defect frac coords in the supercell
            target_frac_coords=self.target_frac_coords,
        )
        defect_entry = get_defect_entry_from_defect(
            value.defect,
            defect_supercell,
            charge_state=0,  # just checking supercell structure here
            dummy_species=_dummy_species,
        )
        if defect_entry.sc_entry != value.sc_entry:
            raise ValueError(
                f"Value must have the same supercell as the DefectsGenerator object, instead has: "
                f"{defect_entry.sc_entry} while DefectsGenerator has: {value.sc_entry}"
            )

        self.defect_entries[key] = value

        # add to self.defects if not already there
        defects_key = _defect_dict_key_from_pmg_type(value.defect.defect_type)
        if defects_key not in self.defects:
            self.defects[defects_key] = []
        try:
            if value.defect not in self.defects[defects_key]:
                self.defects[defects_key].append(value.defect)
        except ValueError as value_err:
            if "You need at least" not in value_err.args[0]:
                raise value_err

            # just test based on names instead
            if value.defect.name not in [defect.name for defect in self.defects[defects_key]]:
                self.defects[defects_key].append(value.defect)

        # sort defects and defect entries for deterministic behaviour:
        self.defects = _sort_defects(self.defects, element_list=self._element_list)
        self.defect_entries = sort_defect_entries(
            self.defect_entries, element_list=self._element_list
        )  # type:ignore

    def __delitem__(self, key):
        """
        Deletes the specified defect entry from the ``defect_entries``
        dictionary.

        Doesn't remove the defect from the defects dictionary attribute, as
        there may be other charge states of the same defect still present.
        """
        del self.defect_entries[key]

    def __contains__(self, key):
        """
        Returns ``True`` if the ``defect_entries`` dictionary contains the
        specified defect name.
        """
        return key in self.defect_entries

    def __len__(self):
        """
        Returns the number of entries in the ``defect_entries`` dictionary.
        """
        return len(self.defect_entries)

    def __iter__(self):
        """
        Returns an iterator over the ``defect_entries`` dictionary.
        """
        return iter(self.defect_entries)

    def __str__(self):
        """
        Returns a string representation of the ``DefectsGenerator`` object.
        """
        formula = self.primitive_structure.composition.get_reduced_formula_and_factor(iupac_ordering=True)[
            0
        ]

        return (
            f"DefectsGenerator for input composition {formula}, space group "
            f"{self.primitive_structure.get_space_group_info()[0]} with {len(self)} defect entries "
            f"created."
        )

    def __repr__(self):
        """
        Returns a string representation of the ``DefectsGenerator`` object, and
        prints the ``DefectsGenerator`` info.

        Note that Wyckoff letters can depend on the ordering of elements in
        the conventional standard structure, for which doped uses the ``spglib``
        convention.
        """
        return (
            self.__str__()
            + "\n---------------------------------------------------------\n"
            + self._defect_generator_info()
        )


def _get_element_list(defect: Defect | DefectEntry | dict | list) -> list[str]:
    """
    Given an input ``Defect`` or ``DefectEntry``, or dictionary/list of these,
    return a (non-duplicated) list of elements present in the defect
    structures, following the order of appearance in the composition.

    Extrinsic elements are sorted according to ``_element_sort_func``,
    which sorts based on periodic group and atomic number.
    """

    def _get_single_defect_element_list(single_defect):
        element_list = [el.symbol for el in single_defect.structure.composition.elements]
        defect_element = single_defect.defect_site.specie.symbol  # possibly extrinsic
        if defect_element not in element_list:
            element_list.append(defect_element)

        return element_list

    if isinstance(defect, Defect | DefectEntry | core.Defect | thermo.DefectEntry):
        return _get_single_defect_element_list(
            defect if isinstance(defect, Defect | core.Defect) else defect.defect
        )

    # else is dict/list
    defect_list = defect if isinstance(defect, list) else list(defect.values())
    defect_list = [
        (
            entry_or_defect.defect
            if isinstance(entry_or_defect, DefectEntry | thermo.DefectEntry)
            else entry_or_defect
        )
        for entry_or_defect in defect_list
    ]
    host_element_list = [el.symbol for el in next(iter(defect_list)).structure.composition.elements]
    extrinsic_element_list: list[str] = []
    for single_defect in defect_list:
        extrinsic_element_list.append(single_defect.defect_site.specie.symbol)  # possibly extrinsic
    extrinsic_element_list = list(set(extrinsic_element_list) - set(host_element_list))

    # sort extrinsic elements by periodic group and atomic number for deterministic ordering:
    extrinsic_element_list.sort(key=_element_sort_func)
    return host_element_list + extrinsic_element_list


def _first_and_second_element(defect_name: str) -> tuple[str, str]:
    """
    Return a tuple of the first and second element in the defect name.

    For sorting purposes.

    Args:
        defect_name (str): Defect name.

    Returns:
        tuple: Tuple of the first and second element in the defect name.
    """
    # by using ``format_defect_name``, we can simultaneously handle (amalgamated) old and new ``doped``
    # defect names:
    formatted_defect_name = format_defect_name(
        defect_name, include_site_info_in_name=False, wout_charge=not defect_name.split("_")[-1].isdigit()
    )
    if formatted_defect_name:
        if not formatted_defect_name.startswith("$"):  # substitution or interstitial
            first_element = formatted_defect_name.split("$")[0]

            if "$_i" in formatted_defect_name:  # interstitial
                return (first_element, first_element)

            return (first_element, formatted_defect_name.split("$_{")[1].split("}")[0])  # substitution

        vacancy_elt = formatted_defect_name.split("$_{")[1].split("}")[0]  # else vacancy
        return (vacancy_elt, vacancy_elt)

    return (defect_name.split("_")[0], defect_name.split("_")[1])  # return name split if formatting fails


def _element_sort_func(element_str: str) -> tuple[int, int]:
    """
    Return a tuple of the group (+16 if it's a transition metal, to move them
    after main group elements) and atomic number of the element, for sorting
    purposes.

    Args:
        element_str (str): Element symbol.

    Returns:
        tuple: Tuple of the group and atomic number of the
            element.
    """
    elt = Element(element_str)
    group = elt.group + 16 if 3 <= elt.group <= 18 else elt.group
    return (group, elt.Z)


def sort_defect_entries(defect_entries: dict | list, element_list: list | None = None) -> dict | list:
    """
    Sort defect entries for deterministic behaviour (for output and when
    reloading ``DefectsGenerator`` objects, with ``DefectThermodynamics``
    entries (particularly for deterministic plotting behaviour), and with
    ``DefectsParser`` objects.

    Sorts defect entries by defect type (vacancies, substitutions,
    interstitials), then by order of appearance of elements in the host
    composition, then by periodic group (main groups 1, 2, 13-18 first,
    then TMs), then by atomic number, then (for defect entries of the same
    type) sort by charge state (from positive to negative).

    Args:
        defect_entries (dict or list):
            Dictionary (in the format: ``{defect_entry_name: defect_entry}``)
            or list of defect entries to sort.
        element_list (list, optional):
            List of elements present, used to determine preferential
            ordering. If ``None``, determined by ``_get_element_list()``,
            which orders by appearance of elements in the host composition,
            then by periodic group (main groups 1, 2, 13-18 first, then TMs),
            then by atomic number.

    Returns:
        Sorted dictionary or list of defect entries.
    """
    if element_list is None:
        element_list = _get_element_list(defect_entries)

    defect_entries_dict = (
        defect_entries
        if isinstance(defect_entries, dict)
        else name_defect_entries(defect_entries, element_list)
    )

    try:
        sorted_defect_entries_dict = dict(
            sorted(
                defect_entries_dict.items(),
                key=lambda s: (
                    s[1].defect.defect_type.value,
                    _list_index_or_val(element_list, _first_and_second_element(s[0])[0]),
                    _list_index_or_val(element_list, _first_and_second_element(s[0])[1]),
                    s[0].rsplit("_", 1)[0],  # name without charge
                    -s[1].charge_state,  # charge state
                ),
            )
        )
    except ValueError as value_err:
        # possibly defect entries with names not in doped format, try sorting without using name:
        try:

            def _defect_entry_sort_func(defect_entry):
                unrelaxed_defect_name_w_charge = defect_entry.calculation_metadata.get(
                    "full_unrelaxed_defect_name"
                )
                if unrelaxed_defect_name_w_charge is not None:
                    name_from_defect = unrelaxed_defect_name_w_charge.rsplit("_", 1)[0]  # without charge
                else:
                    name_from_defect = get_defect_name_from_defect(
                        defect_entry.defect,
                        element_list=element_list,
                    )
                return (
                    defect_entry.defect.defect_type.value,
                    _list_index_or_val(
                        element_list, _first_and_second_element(defect_entry.defect.name)[0]
                    ),
                    _list_index_or_val(
                        element_list, _first_and_second_element(defect_entry.defect.name)[1]
                    ),
                    defect_entry.name.rsplit("_", 1)[0],  # name without charge
                    name_from_defect,  # doped name without charge
                    -defect_entry.charge_state,  # charge state
                )

            sorted_defect_entries_dict = dict(
                sorted(
                    defect_entries_dict.items(),
                    key=lambda s: _defect_entry_sort_func(s[1]),  # sort by defect entry object
                )
            )
        except ValueError as value_err_2:
            raise value_err_2 from value_err

    if isinstance(defect_entries, list):  # then return as a list
        return list(sorted_defect_entries_dict.values())

    return sorted_defect_entries_dict  # else dict


def _sort_defects(defects_dict: dict, element_list: list[str] | None = None):
    """
    Sort defect objects for deterministic behaviour (for output and when
    reloading ``DefectsGenerator`` objects.

    Sorts defects by defect type (vacancies, substitutions, interstitials),
    then by order of appearance of elements in the composition, then
    by periodic group (main groups 1, 2, 13-18 first, then TMs), then by
    atomic number, then according to ``symmetry._frac_coords_sort_func``.
    """
    if element_list is None:
        element_list = _get_element_list(defects_dict)

    return {
        defect_type: sorted(
            defect_list,
            key=lambda d: (
                _list_index_or_val(element_list, _first_and_second_element(d.name)[0]),
                _list_index_or_val(element_list, _first_and_second_element(d.name)[1]),
                d.name,  # bare name without charge
                symmetry._frac_coords_sort_func(d.conv_cell_frac_coords),
            ),
        )
        for defect_type, defect_list in defects_dict.items()
    }


def get_stol_equiv_dist(stol: float, structure: Structure) -> float:
    """
    Get the equivalent Cartesian distance of a given ``stol`` value for a given
    ``Structure``.

    ``stol`` is a site tolerance parameter used in ``pymatgen``
    ``StructureMatcher`` functions, defined as the fraction of the average
    free length per atom := ( V / Nsites ) ** (1/3).

    Args:
        stol (float): Site tolerance parameter.
        structure (Structure): Structure to get equivalent distance for.

    Returns:
        float: Equivalent Cartesian distance for the given ``stol`` value.
    """
    return stol * (structure.volume / len(structure)) ** (1 / 3)


def get_Voronoi_interstitial_sites(
    host_structure: Structure, interstitial_gen_kwargs: dict[str, Any] | None = None
) -> list:
    """
    Generate candidate interstitial sites using Voronoi analysis.

    This function uses a similar approach to that in
    ``VoronoiInterstitialGenerator`` from ``pymatgen-analysis-defects``,
    but with modifications to make interstitial generation much faster,
    fix bugs with interstitial grouping (which could lead to undesired
    dropping of candidate sites) and achieve better control over site
    placement in order to favour sites which are higher-symmetry and
    furthest from the host lattice atoms (typically the most favourable
    interstitial sites).

    The logic for picking interstitial sites is as follows:

    - Generate all candidate sites using (efficient) Voronoi analysis
    - Remove any sites which are within ``min_dist`` of any host atoms
    - Cluster the remaining sites using a tolerance of ``clustering_tol``
      and symmetry-preference of ``symmetry_preference``
      (see ``_doped_cluster_frac_coords``)
    - Determine the multiplicities and symmetry-equivalent coordinates of
      the clustered sites using ``doped`` symmetry functions.
    - Group the clustered sites by symmetry using (looser) site matching
      as controlled by ``stol``.
    - From each group, pick the site with the highest symmetry and furthest
      distance from the host atoms, if its ``min_dist`` is no more than
      ``symmetry_preference`` (0.1 Å by default) smaller than the site with
      the largest ``min_dist`` (to the host atoms).

    (Parameters mentioned here can be supplied via ``interstitial_gen_kwargs``
    as noted in the args section below.)

    The motivation for favouring high symmetry interstitial sites and then
    distance to host atoms is because higher symmetry interstitial sites
    are typically the more intuitive sites for placement, cleaner, easier
    for analysis etc, and interstitials are often lowest-energy when
    furthest from host atoms (i.e. in the largest interstitial voids --
    particularly for fully-ionised charge states), and so this approach
    tries to strike a balance between these two goals.

    One caveat to this preference for high symmetry interstitial sites, is
    that they can also be slightly more prone to being stuck in local minima
    on the PES, and so as always it is **highly recommended** to use
    ``ShakeNBreak`` or another structure-searching technique to account for
    symmetry-breaking when performing defect relaxations!

    You can see what Cartesian distance the chosen ``stol`` corresponds to
    using the ``get_stol_equiv_dist`` function.

    Args:
        host_structure (Structure): Host structure.
        interstitial_gen_kwargs (dict):
            Keyword arguments for interstitial generation. Supported kwargs are:

            - min_dist (float):
                Minimum distance from host atoms for interstitial sites.
                Defaults to 0.9 Å.
            - clustering_tol (float):
                Tolerance for clustering interstitial sites. Defaults to 0.8 Å.
            - symmetry_preference (float):
                Symmetry preference for interstitial site selection. Defaults to 0.1 Å.
            - stol (float):
                Structure matcher tolerance for looser site matching. Defaults to 0.32.
            - tight_stol (float):
                Structure matcher tolerance for tighter site matching. Defaults to 0.02.
            - symprec (float):
                Symmetry precision for (symmetry-)equivalent site determination. Defaults
                to 0.01.

    Returns:
        list: List of interstitial sites as fractional coordinates
    """
    # so we want to pick the higher symmetry sites because it's cleaner, more intuitive etc
    # but, this is slightly more likely to be stuck in local minima, compared to the (nearby)
    # lower symmetry interstitial sites... avoided by using ShakeNBreak, other structure-searching
    # approaches, or rattling the output structures (default in ``doped.vasp``)

    interstitial_gen_kwargs = interstitial_gen_kwargs or {}
    supported_interstitial_gen_kwargs = {
        "min_dist",
        "clustering_tol",
        "symmetry_preference",
        "stol",
        "tight_stol",
        "symprec",
    }
    if any(  # check interstitial_gen_kwargs and warn if any missing:
        i not in supported_interstitial_gen_kwargs for i in interstitial_gen_kwargs
    ):
        raise TypeError(
            f"Invalid interstitial_gen_kwargs supplied!\nGot: {interstitial_gen_kwargs}\nbut "
            f"only the following keys are supported: {supported_interstitial_gen_kwargs}"
        )
    top = DopedTopographyAnalyzer(host_structure)
    if not top.vnodes:
        warnings.warn("No interstitial sites found in host structure!")
        return []

    sites_list = [v.frac_coords for v in top.vnodes]
    min_dist = interstitial_gen_kwargs.get("min_dist", 0.9)
    sites_array = remove_collisions(sites_list, structure=host_structure, min_dist=min_dist)
    if sites_array.size == 0:
        warnings.warn(
            f"No interstitial sites found after removing those within {min_dist} Å of host atoms!"
        )
        return []

    site_frac_coords_array: np.ndarray = _doped_cluster_frac_coords(
        sites_array,
        host_structure,
        tol=interstitial_gen_kwargs.get("clustering_tol", 0.8),
        symmetry_preference=interstitial_gen_kwargs.get("symmetry_preference", 0.1),
    )

    label_equiv_fpos_dict: dict[int, list[np.ndarray[float]]] = {}
    sga = symmetry.get_sga(host_structure, symprec=interstitial_gen_kwargs.get("symprec", 0.01))
    symm_ops = sga.get_symmetry_operations()
    tight_dist = get_stol_equiv_dist(
        interstitial_gen_kwargs.get("tight_stol", 0.02), host_structure
    )  # 0.06 Å for CdTe, Sb2Si2Te6

    # this now depends on symprec in `get_all_equiv_sites` (doesn't matter in most cases,
    # but e.g. changes results in Ag2Se where we have some slight differences in site coordinations)
    for i, frac_coords in enumerate(site_frac_coords_array.tolist()):
        match_found = False
        for equiv_fpos in list(label_equiv_fpos_dict.values()):
            if np.min(host_structure.lattice.get_all_distances(equiv_fpos, frac_coords)) < 0.01:
                match_found = True
                break

        if not match_found:  # try equiv sites:
            this_equiv_fpos = [
                site.frac_coords
                for site in symmetry.get_all_equiv_sites(
                    frac_coords,
                    host_structure,
                    symm_ops=symm_ops,
                    symprec=interstitial_gen_kwargs.get("symprec", 0.01),
                )
            ]
            for label, equiv_fpos in list(label_equiv_fpos_dict.items()):
                if np.min(host_structure.lattice.get_all_distances(equiv_fpos, frac_coords)) < tight_dist:
                    label_equiv_fpos_dict[label].extend(this_equiv_fpos)
                    match_found = True

        if not match_found:
            label_equiv_fpos_dict[i] = this_equiv_fpos

    tight_cand_site_mul_and_equiv_fpos_list = [
        (equiv_fpos[0], len(equiv_fpos), equiv_fpos) for equiv_fpos in label_equiv_fpos_dict.values()
    ]

    loose_dist = get_stol_equiv_dist(
        interstitial_gen_kwargs.get("stol", 0.32),  # matches pymatgen-analysis-defects default
        host_structure,  # ~1 Å for CdTe, Sb2Si2Te6
    )
    looser_site_matched_dict: dict[int, list] = defaultdict(list)
    for i, tight_cand_site_mul_and_equiv_fpos in enumerate(tight_cand_site_mul_and_equiv_fpos_list):
        # should only need to compare equiv_fpos[0] with all equiv sites of other groups
        match_found = False
        for label, sublist in list(looser_site_matched_dict.items()):
            if (
                np.min(
                    host_structure.lattice.get_all_distances(
                        # only match to first equiv_fpos in list of matched sites, so we don't get a
                        # chaining effect (e.g. site 1 -> 1 Å from site 2 -> 1 Å from site 3 (but 2 Å
                        # from site 1) etc.)
                        sublist[0][2],
                        tight_cand_site_mul_and_equiv_fpos[0],
                    )
                )
                < loose_dist  # loose tol here
            ):
                match_found = True
                looser_site_matched_dict[label].append(tight_cand_site_mul_and_equiv_fpos)
                break

        if not match_found:
            looser_site_matched_dict[i].append(tight_cand_site_mul_and_equiv_fpos)

    cand_site_mul_and_equiv_fpos_list = []
    symmetry_preference = interstitial_gen_kwargs.get("symmetry_preference", 0.1)
    for tight_cand_site_mul_and_equiv_fpos_sublist in looser_site_matched_dict.values():
        if len(tight_cand_site_mul_and_equiv_fpos_sublist) == 1:
            cand_site_mul_and_equiv_fpos_list.append(tight_cand_site_mul_and_equiv_fpos_sublist[0])

        else:  # pick site with the highest symmetry and furthest distance from host atoms:
            site_scores = [
                (
                    cand_site_mul_and_equiv_fpos[1],  # multiplicity (lower is higher symmetry)
                    -np.min(  # distance to nearest host atom (minus; so max -> min for sorting)
                        host_structure.lattice.get_all_distances(
                            cand_site_mul_and_equiv_fpos[0], host_structure.frac_coords
                        ),
                        axis=1,
                    ),
                    *symmetry._frac_coords_sort_func(
                        sorted(cand_site_mul_and_equiv_fpos[2], key=symmetry._frac_coords_sort_func)[0]
                    ),
                )
                for cand_site_mul_and_equiv_fpos in tight_cand_site_mul_and_equiv_fpos_sublist
            ]
            symmetry_favoured_site_mul_and_equiv_fpos = tight_cand_site_mul_and_equiv_fpos_sublist[
                site_scores.index(min(site_scores))
            ]
            dist_favoured_reordered_score = min(
                [(score[1], score[0], *score[2:]) for score in site_scores]
            )
            dist_favoured_site_mul_and_equiv_fpos = tight_cand_site_mul_and_equiv_fpos_sublist[
                site_scores.index(
                    (
                        dist_favoured_reordered_score[1],
                        dist_favoured_reordered_score[0],
                        *dist_favoured_reordered_score[2:],
                    )
                )
            ]

            cand_site_mul_and_equiv_fpos_list.append(
                dist_favoured_site_mul_and_equiv_fpos
                if (
                    np.min(
                        host_structure.lattice.get_all_distances(
                            dist_favoured_site_mul_and_equiv_fpos[0], host_structure.frac_coords
                        ),
                        axis=1,
                    )
                    < np.min(
                        host_structure.lattice.get_all_distances(
                            symmetry_favoured_site_mul_and_equiv_fpos[0], host_structure.frac_coords
                        ),
                        axis=1,
                    )
                    - symmetry_preference
                )
                else symmetry_favoured_site_mul_and_equiv_fpos
            )

    sorted_sites_mul_and_equiv_fpos = []
    for _cand_site, multiplicity, equiv_fpos in cand_site_mul_and_equiv_fpos_list:  # type: ignore
        # take site with equiv_fpos sorted by symmetry._frac_coords_sort_func:
        sorted_equiv_fpos = sorted(equiv_fpos, key=symmetry._frac_coords_sort_func)
        ideal_cand_site = sorted_equiv_fpos[0]
        sorted_sites_mul_and_equiv_fpos.append((ideal_cand_site, multiplicity, sorted_equiv_fpos))

    return sorted_sites_mul_and_equiv_fpos
