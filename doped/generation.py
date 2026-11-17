"""
Code to generate |Defect| objects and supercell structures for ab-initio
calculations.
"""

import contextlib
import logging
import operator
import re
import warnings
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from copy import deepcopy
from functools import lru_cache, partial, reduce
from itertools import chain, count, product
from string import ascii_lowercase
from typing import TYPE_CHECKING, Any, cast
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
from pymatgen.core.adsorption import AdsorbateSiteFinder
from pymatgen.core.entries import ComputedStructureEntry
from pymatgen.core.periodic_table import DummySpecies, Species, get_el_sp
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation
from pymatgen.util.typing import PathLike
from tabulate import tabulate  # type: ignore[import-untyped]
from tqdm import tqdm

from doped.core import (
    Defect,
    DefectComplex,
    DefectEntry,
    Interstitial,
    Substitution,
    Vacancy,
    doped_defect_from_pmg_defect,
    get_oxi_probabilities,
    guess_and_set_oxi_states_with_timeout,
)
from doped.utils import _signed_charge, parsing, pool_manager, supercells, symmetry
from doped.utils.efficiency import Composition, DopedTopographyAnalyzer, Element, PeriodicSite, Structure
from doped.utils.parsing import reorder_s2_like_s1
from doped.utils.plotting import format_defect_name

if TYPE_CHECKING:
    from ase.atoms import Atoms

_dummy_species = DummySpecies("X")  # Dummy species used to keep track of defect coords in the supercell
_H_INTERSTITIAL_GEN_KWARGS = {"min_dist": 0.5, "include_bond_centres": True}  # H defaults

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
    Returns the index of the item in the list, or val if not found.
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
    Generate a ``doped`` |DefectEntry| object from a ``doped`` |Defect| object.

    This is used to describe a |Defect| with a specified simulation cell. To
    set the ``sc_defect_frac_coords`` attribute for |DefectEntry| (fractional
    coordinates of the defect in the ``defect_supercell``), either
    ``dummy_species`` must be present in the ``defect_supercell`` (which is
    taken as the defect site), or ``sc_defect_frac_coords`` must be set.

    Args:
        defect (|Defect|): ``doped``/``pymatgen`` |Defect| object.
        defect_supercell (|Structure|): Defect supercell structure.
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
        DefectEntry: ``doped`` |DefectEntry| object.
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
        sc_defect_frac_coords=tuple(sc_defect_frac_coords) if sc_defect_frac_coords is not None else None,
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
    neighbour in the list must be at least 0.02 Å further away than the
    previous neighbour.

    If there are multiple elements with the same distance (to within ~0.01 Å),
    then the preference for which element to return is controlled by
    ``element_list``. If ``element_list`` is not provided, then it is set to
    match the order of appearance in the structure composition.

    For efficiency, this function dynamically increases the radius when
    searching for nearest neighbours as necessary, using the input
    ``dist_tol_prefactor`` and ``n`` to estimate a reasonable starting range.

    Args:
        site (|PeriodicSite|):
            Site to get neighbour info.
        structure (|Structure|):
            |Structure| containing the site and neighbours.
        n (int):
            Return the element symbol and distance tuples for the ``n``\th
            closest neighbours. Default is 1, corresponding to only the
            nearest neighbour.
        element_list (list):
            Sorted list of elements in the host structure to govern the
            preference of elemental symbols to return, when the distance to
            multiple neighbours with different elements is the same.
            Default is to use the order of appearance of elements in the
            |Structure| composition.
        dist_tol_prefactor (float):
            Initial distance tolerance prefactor to use when searching for
            neighbours, where the initial search radius is set to
            ``dist_tol_prefactor * sqrt(n)`` Å. This value is dynamically
            updated as needed, and so should only be adjusted if providing
            odd systems with very small/large bond lengths for which
            efficiency improvements are possible. Default is 3.0.
        min_dist (float):
            Minimum distance in Å of neighbours from ``site``. Intended as a
            distance tolerance to exclude the site itself from the neighbour
            list, for which the default value (0.02 Å) should be perfectly
            fine in most cases. Set to 0 to include the site itself in the
            output list (in which case it counts toward ``n``).

    Returns:
        list: Sorted list of tuples of (distance, element) for the ``n``\th
        closest neighbours to the input ``site``.
    """
    if element_list is None:
        element_list = list(dict.fromkeys(el.symbol for el in structure.composition.elements))

    neighbour_tuples: list[tuple[float, str]] = []
    while len(neighbour_tuples) < n:
        # for efficiency, ignore sites further than dist_tol*sqrt(n) Å away
        # dynamic upscaling of the distance tolerance is required for weird structures (e.g. mp-674158,
        # mp-1208561):
        neighbours_w_site_itself = structure.get_sites_in_sphere(
            site.coords, dist_tol_prefactor * np.sqrt(n)
        )
        neighbours = sorted(neighbours_w_site_itself, key=lambda x: x.nn_distance)

        dist_tol_prefactor += 0.5  # increase the distance tolerance if no other sites are found
        if dist_tol_prefactor > 40:
            warnings.warn(
                "No other sites found within 40*sqrt(n) Å of the defect site, indicating a very "
                "weird structure..."
            )
            break

        if not neighbours:
            continue

        neighbour_tuples = sorted(  # Could make this faster using caching if it was becoming a bottleneck
            [
                (
                    neigh.nn_distance,
                    neigh.specie.symbol,
                )
                for neigh in neighbours
            ],
            key=lambda x: (round(x[0], 2), _list_index_or_val(element_list, x[1]), x[1]),
        )
        neighbour_tuples = [  # prune site_distances to remove any with distances within 0.02 Å of the
            # previous n (and the same element as the previous n):
            neighbour_tuples[i]
            for i in range(len(neighbour_tuples))
            if (neighbour_tuples[i][0] > min_dist and i == 0)
            or (neighbour_tuples[i][0] - neighbour_tuples[i - 1][0] > 0.02)
            or (
                neighbour_tuples[i][1] != neighbour_tuples[i - 1][1]
                and (
                    neighbour_tuples[i][0]
                    - max(
                        [
                            neighbour_tuple[0]
                            for neighbour_tuple in neighbour_tuples[:i]
                            if neighbour_tuple[1] == neighbour_tuples[i][1]
                        ]
                        or [0]
                    )
                    > 0.02
                )
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
    closest site to the defect site in the input |DefectEntry| or |Defect|
    object.

    If there are multiple elements with the same distance (to within ~0.01 Å),
    then the preference for which element to return is controlled by
    ``element_list``. If ``element_list`` is not provided, then it is set to
    match the order of appearance in the structure composition.

    If ``n`` > 1, then it returns the nth closest site, where the nth site must
    be at least 0.02 Å further away than the (n-1)th site.

    Args:
        defect_entry_or_defect (|DefectEntry| | |Defect|):
            |DefectEntry| or |Defect| object, to get neighbour info.
        n (int):
            Return the element symbol and distance for the ``n``\th closest
            site. Default is 1, corresponding to the (1st) closest neighbour.
        element_list (list):
            Sorted list of elements in the host structure to govern the
            preference of elemental symbols to return, when the distance to
            multiple neighbours with different elements is the same.
            Default is to use ``_get_element_list()``, which follows the
            order of appearance of elements in the |Structure| composition.

    Returns:
        str:
            Element symbol and distance (rounded to 2 decimal places) of the
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
    return f"{closest_site}{min_distance:.2f}"


def get_defect_name_from_defect(
    defect: Defect,
    element_list: list[str] | None = None,
    symprec: float = 0.01,
):
    """
    Get the doped/SnB defect name from a |Defect| object.

    Args:
        defect (|Defect|): |Defect| object.
        element_list (list):
            Sorted list of elements in the host structure, so that
            ``closest_site_info()`` returns deterministic results (in case two
            different elements located at the same distance from defect site).
            Default is None.
        symprec (float):
            Symmetry tolerance for ``spglib``. Default is 0.01.

    Returns:
        str: Defect name.
    """
    if isinstance(defect, DefectComplex):  # constituents, separations and point group, as in generation
        from doped.complexes import get_defect_complex_name  # avoid a circular import

        return get_defect_complex_name(defect, element_list=element_list)

    point_group_symbol = symmetry.point_symmetry_from_defect(defect, symprec=symprec)

    return f"{defect.name}_{point_group_symbol}_{closest_site_info(defect, element_list=element_list)}"


def get_defect_name_from_entry(
    defect_entry: DefectEntry,
    element_list: list | None = None,
    symprec: float | None = None,
    relaxed: bool = True,
    **kwargs,
):
    r"""
    Get the doped/SnB defect name from a |DefectEntry| object.

    Note: If relaxed = True (default), then this uses the
    ``defect_entry.defect_supercell`` to determine the point symmetry of the
    (relaxed) defect structure, via direct isometry analysis of the local
    defect environment (see |point_symmetry_from_defect_entry|; insensitive to
    periodicity-breaking supercell shapes, unlike global space-group analysis).

    Args:
        defect_entry (|DefectEntry|): |DefectEntry| object.
        element_list (list):
            Sorted list of elements in the host structure, so that
            ``closest_site_info()`` returns deterministic results (in case two
            different elements located at the same distance from defect site).
            Default is None.
        symprec (float):
            Distance tolerance (in Å) for symmetry determination (see
            |point_symmetry_from_defect_entry|). Default is 0.01 Å for
            unrelaxed structures, 0.1 Å for relaxed (to account for residual
            structural noise). You may want to adjust for your system (e.g.
            if there are very slight octahedral distortions etc).
        relaxed (bool):
            If ``False``, determines the site symmetry using the defect site
            `in the unrelaxed bulk supercell`, otherwise determines the point
            symmetry of the relaxed defect in the defect supercell. Default is
            ``True``.
        **kwargs:
            Additional keyword arguments to pass to
            |point_symmetry_from_defect_entry|, such as ``dist_tol_factor``
            or ``verbose``.

    Returns:
        str: Defect name.
    """
    point_group_symbol = symmetry.point_symmetry_from_defect_entry(
        defect_entry, symprec=symprec, relaxed=relaxed, **kwargs
    )
    if isinstance(defect_entry.defect, DefectComplex):  # constituents and separations, then point group
        from doped.complexes import get_defect_complex_name  # avoid a circular import

        complex_name = get_defect_complex_name(
            defect_entry.defect, element_list=element_list, include_point_group=False
        )
        return f"{complex_name}-{point_group_symbol}"

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
    symprec=0.01,
):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Not all sites")
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

        # add complex specific attributes - any single sites get the centroid
        defect_supercell_sites = equivalent_supercell_complexes = None  # ``None`` for point defects
        if isinstance(defect, DefectComplex):
            defect_supercell_sites, equivalent_supercell_complexes = (
                defect_supercell_site,
                equivalent_supercell_sites,
            )
            defect_supercell_site, *equivalent_supercell_sites = [
                PeriodicSite(
                    defect.site.species,
                    np.mean([site.frac_coords for site in sites], axis=0),
                    dummy_defect_supercell.lattice,
                )
                for sites in [defect_supercell_sites, *equivalent_supercell_complexes]
            ]

        dummy_sites = [
            site for site in dummy_defect_supercell if site.specie.symbol == _dummy_species.symbol
        ]
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
        neutral_defect_entry.defect_supercell_sites = defect_supercell_sites  # complexes only
        neutral_defect_entry.equivalent_supercell_complexes = equivalent_supercell_complexes
        neutral_defect_entry.bulk_supercell = bulk_supercell

        neutral_defect_entry.conventional_structure = (
            neutral_defect_entry.defect.conventional_structure
        ) = conventional_structure

        try:
            wyckoff_label, conv_cell_sites = symmetry.get_wyckoff(
                symmetry.get_conv_cell_site(neutral_defect_entry).frac_coords,
                conventional_structure,
                equiv_sites=True,
                symprec=symprec,
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

        if isinstance(defect, DefectComplex):
            # TODO ?
            wyckoff_label = "N/A"

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


def _name_wout_charge_state(name: str) -> str:
    """
    Strip the charge state from a ``doped`` defect name, if present: this is
    the last _ split if the final substring is +int or -int.
    """
    charge = name.rsplit("_", 1)[-1]
    return name.rsplit("_", 1)[0] if "_" in name and charge.lstrip("+-").isdigit() else name


def _match_defect_names(name: str, available: Iterable[str], label: str = "Defect") -> list[str]:
    """
    Match a given defect name against ``available`` names, ignoring any charge
    states and allowing partial matches (see ``_check_if_name_subset``).
    """
    if not (matches := [n for n in available if _check_if_name_subset(n, _name_wout_charge_state(name))]):
        raise ValueError(
            f"{label} name {name!r} does not match any of the generated {label.lower()} names: "
            f"{list(available)}"
        )
    return matches


def _check_if_name_subset(long_name: str, poss_subset_name: str):
    """
    Check if the longer name is a superset of the shorter name, where
    underscores are used as a name delimiter as in ``doped``.

    Previously used ``startswith`` to compare, but caused issues with e.g.
    ``v_Cl`` and ``v_C`` defects in the same material.

    Defect complex names use hyphens as their outer delimiter (constituent
    names and their separations, then the complex point group), so these are
    compared term-by-term, of which only a trailing complex point group may be
    omitted from ``poss_subset_name``. Point defect names contain no hyphens
    and so are a single term, meaning e.g. ``v_Cd`` does not match the complex
    ``v_Cd-2.83-v_Te-C3v``, while ``v_Cd-2.83-v_Te`` does.
    """
    # a hyphen directly following an underscore is the sign of a negative charge state
    # (``v_Cd_-2``), not a complex name delimiter:
    long_terms, subset_terms = (re.split(r"(?<!_)-", name) for name in (long_name, poss_subset_name))
    if len(long_terms) != len(subset_terms) and not (
        len(subset_terms) == len(long_terms) - 1 and len(long_terms) % 2 == 0
    ):  # an even number of terms means the last is a complex point group, which may be omitted
        # because a complex name looks like (defect-sep)*n-defect-pointgroup
        return False

    return all(
        _check_if_underscore_name_subset(long_term, subset_term)
        for long_term, subset_term in zip(long_terms, subset_terms, strict=False)
    )


def _check_if_underscore_name_subset(long_name: str, poss_subset_name: str):
    """
    As ``_check_if_name_subset``, but for a single hyphen-delimited term (i.e.
    an individual point defect name, or a complex point group).
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
):
    """
    Create a dictionary of ``{name: DefectEntry}`` from a list of |DefectEntry|
    objects, where the names are set according to the default doped algorithm;
    which is to use the pymatgen defect name (e.g. v_Cd, Cd_Te etc.) for
    vacancies/antisites/substitutions, unless there are multiple inequivalent
    sites for the defect, in which case the point group of the defect site is
    appended (e.g. v_Cd_Td, Cd_Te_Td etc.), and if this is still not unique,
    then element identity and distance to the nearest neighbour of the defect
    site is appended (e.g. v_Cd_Td_Te2.83, Cd_Te_Td_Cd2.83 etc.). Names do not
    yet have charge states included.

    For interstitials, the same naming scheme is used, but the point group is
    always appended to the pymatgen defect name.

    If still not unique after the 3rd nearest neighbour info, then "a, b, c"
    etc. is appended to the name of different defects to distinguish.

    Args:
        defect_entries (list):
            List of |DefectEntry| or |Defect| objects to name.
        element_list (list):
            Sorted list of elements in the host structure, so that
            ``closest_site_info()`` returns deterministic results (in case two
            different elements located at the same distance from defect site).
            Default is None.

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
            previous_entry_full_name = get_defect_name_from_defect(prev_defect, element_list)
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

    def string_until_last_number(string: str) -> str:
        """
        Return the input string, truncated at the last numeric character.

        e.g. "Ga_i_C3i_O2.83aa" -> "Ga_i_C3i_O2.83"
        """
        match = re.search(r"\d(?!.*\d)", string)
        if match:
            return string[: match.end()]
        return string

    def handle_repeated_name(defect_naming_dict, full_defect_name):
        defect_name = None
        for name in list(defect_naming_dict.keys()):
            if full_defect_name == name:
                prev_defect_entry = defect_naming_dict.pop(name)
                defect_naming_dict[f"{name}a"] = prev_defect_entry
                defect_name = f"{full_defect_name}b"
                break
            name_until_last_number = string_until_last_number(name)
            if full_defect_name == name_until_last_number:
                last_letters = sorted(
                    [
                        name.removeprefix(name_until_last_number)
                        for name in defect_naming_dict
                        if string_until_last_number(name) == full_defect_name
                    ],
                    key=lambda x: (len(x), x),
                )
                if last_letters[-1][-1] == "z":  # we go again
                    defect_name = full_defect_name + ("a" * (len(last_letters[-1]) + 1))
                    break

                new_letter = chr(ord(last_letters[-1][-1]) + 1)
                defect_name = full_defect_name + last_letters[-1][:-1] + new_letter
                break

        if defect_name is None:
            raise ValueError(
                f"Multiple defect names found for {full_defect_name}, and couldn't be "
                f"renamed properly. Please report this issue to the developers."
            )

        return defect_naming_dict, defect_name

    def _get_defect(defect_entry):
        return (
            defect_entry.defect
            if isinstance(defect_entry, DefectEntry | thermo.DefectEntry)
            else defect_entry
        )

    complex_entries = [entry for entry in defect_entries if isinstance(_get_defect(entry), DefectComplex)]
    defect_naming_dict: dict[str, DefectEntry] = {}
    for defect_entry in defect_entries:
        defect = _get_defect(defect_entry)
        if isinstance(defect, DefectComplex):
            continue  # named below
        full_defect_name = get_defect_name_from_defect(defect, element_list)
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

    # defect complexes are named as in |DefectsGenerator| (constituents, separations and point group; e.g.
    # ``v_Cd-1.87-Cd_i-1.87-v_Cd-C3v``), with ``_a``, ``_b``... suffixes for any shared names:
    complex_names, _renames = _disambiguate_names(
        [get_defect_name_from_defect(_get_defect(entry), element_list) for entry in complex_entries],
        existing_names=defect_naming_dict,
    )
    defect_naming_dict.update(zip(complex_names, complex_entries, strict=True))

    if len(defect_entries) != len(defect_naming_dict):
        raise ValueError(
            f"Number of defect entries ({len(defect_entries)}) does not match "
            f"number of unique defect names ({len(defect_naming_dict)}). "
            f"Please report this issue to the developers."
        )
    return defect_naming_dict


def _letter_suffixes() -> Iterator[str]:
    """
    Yield a, b, ..., z, aa, ab, ... as an iterator.

    Note this differs from the scheme in ``name_defect_entries``, which goes
    ``"z"`` -> ``"aa"`` -> ``"aaa"`` (ie repeated letters only): here a full
    lexicographic sequence is used for sorting and due to the frequency of collisions.
    """
    for length in count(1):
        for letters in product(ascii_lowercase, repeat=length):
            yield "".join(letters)


def _disambiguate_names(
    new_names: Sequence[str], existing_names: Iterable[str] = ()
) -> tuple[list[str], dict[str, str]]:
    """
    Append ``_a``, ``_b``, ... suffixes to make ``new_names`` unique, both
    among themselves and against ``existing_names``.

    Used for defect complexes, whose names are necessarily not unique. Any existing
    holder of a unique name in ``existing_names`` that is now shared is appended
    with ``"_a"``, instead of remaining bare.

    Args:
        new_names (Sequence[str]):
            Names to be added, in the order they should be assigned suffixes.
        existing_names (Iterable[str]):
            Names already in use, which ``new_names`` must not collide with.
            (Default: ``()``)

    Returns:
        tuple[list[str], dict[str, str]]:
            The names to use for ``new_names`` (in the same order), and any
            ``{old_name: new_name}`` renames the caller must apply to
            ``existing_names``.
    """
    taken = set(existing_names)
    renames: dict[str, str] = {}
    demoted: set[str] = set()  # bare names which have been superseded by a suffixed series
    names: list[str] = []

    def _next_free(name: str) -> str:
        return next(f"{name}_{suffix}" for suffix in _letter_suffixes() if f"{name}_{suffix}" not in taken)

    def _series_base(name: str) -> str | None:
        """
        The bare name of ``name`` if it is a member of a suffixed series, else
        ``None``; suffixes from ``_letter_suffixes`` are lowercase ASCII
        letters, unlike any point group symbol.
        """
        base, _, suffix = name.rpartition("_")
        return base if base and suffix.isascii() and suffix.isalpha() and suffix.islower() else None

    # bare names already superseded by a suffixed series, detected from *any* surviving member of
    # that series -- not just ``_a`` -- so that deleting members cannot resurrect the bare name:
    series = {base for name in taken if (base := _series_base(name)) is not None}

    for name in new_names:
        # a bare name which has already been superseded by a series must not be reused, even if the
        # series was assigned in an earlier call (or before reloading):
        if name not in taken and name not in series:
            taken.add(name)  # no collision, use the bare name
            names.append(name)
            continue

        # TODO: this renames a complex - which the user may already possibly have generated
        # files/calculations for? alternative is to leave bare, but then not obvious that this
        # is a name in common with other complexes - and code would be simpler i think...
        if name in taken and name not in demoted:  # demote whoever holds the bare name
            demoted.add(name)  # (and keep the bare name reserved, so it is never reused)
            suffixed = _next_free(name)
            taken.add(suffixed)
            if name in names:  # an earlier new name, so can be corrected directly
                names[names.index(name)] = suffixed
            else:  # an existing entry, so the caller must rename it
                renames[name] = suffixed

        taken.add(new_name := _next_free(name))
        names.append(new_name)

    return names, renames


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

    Disfavours large (absolute) charge states, low probability oxidation states
    and greater charge/oxidation state magnitudes than that of the host.
    Specifically, the overall probability is given by the product of these
    probability factors:

    - The probability of the corresponding oxidation state of the defect
      element (e.g. Na_i^+1 has Na in the +1 oxidation state), typically given
      by its prevalence in the ICSD (for substitutions & interstitials), or the
      vacancy oxidation state probability function (described in the
      :func:`guess_defect_charge_states` docstring; implemented in
      ``_get_vacancy_oxi_states``).
    - The magnitude of the charge state; with a probability function:
      ``1/|charge_state|^(2/3)``
    - The magnitude of the charge state relative to the max host oxidation
      state (i.e. how 'charged' the host is); with a probability function:
      ``1/(2*|charge_state - max_host_oxi_magnitude|)^(2/3)`` if
      ``charge_state > max_host_oxi_magnitude``, otherwise 1.
    - The magnitude of the defect element oxidation state relative to the max
      host oxidation state; with a probability function:
      ``1/(2*|defect_el_oxi_state - max_host_oxi_magnitude|)^(2/3)`` if
      ``defect_el_oxi_state > max_host_oxi_magnitude``, otherwise 1.

    Note that neutral charge states are always included.

    This probability function was found to give optimal performance in terms of
    efficiency and completeness when tested against other approaches (see the
    ``doped`` JOSS paper: https://doi.org/10.21105/joss.06433) but of course
    may not be perfect in all cases, so make sure to critically consider the
    estimated charge states for your system!

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
        Probability of the defect charge state (between 0 and 1) if
        ``return_log`` is ``False``, otherwise a dictionary of input & computed
        values used to determine charge state probability.
    """
    # Incorporates oxidation state probabilities and overall defect site charge
    # (and thus by proxy the cationic/anionic identity of the substituting site)
    # Incorporating the oxi state probabilities (i.e. reducibility/oxidisability) of the neighbouring
    # elements could potentially help (particularly for vacancies), but no clear-cut cases where this would
    # actually improve performance, and adds significant complexity
    # In future, one could consider using an electrostatic penalty function, particularly for defect
    # complexes, but would introduce some complexity... For now, onus is on the user to screen candidate
    # defect complexes appropriately

    def _defect_vs_host_charge(charge_state: int, host_charge: int) -> float:
        if abs(charge_state) <= abs(host_charge):
            return 1

        return 1 / (2 * (abs(charge_state) - abs(host_charge)))

    charge_state_guessing_log: dict[str, Any] = {
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
    charge_state_guessing_log["probability"] = (  # product of probability_factors
        float(np.prod(list(charge_state_guessing_log["probability_factors"].values())))
        if charge_state != 0
        else 1
    )  # always include neutral charge state

    if return_log:
        return charge_state_guessing_log

    return charge_state_guessing_log["probability"]


def get_defect_site_orig_oxi_state(defect: Defect) -> int:
    """
    Get the oxidation state of the original (bulk) site at the defect position
    in the host structure.

    This is the oxidation state of the original bulk site for vacancies and
    substitutions -- for vacancies, taken as the negative of
    ``Vacancy.oxi_state`` (the fully-ionised vacancy charge state), to account
    for possible user-specified vacancy oxidation states -- or 0 for
    interstitials (no original bulk site occupant).

    Note that this requires oxidation states to be assigned to
    ``defect.structure`` for substitutions, and ``defect.oxi_state`` for
    vacancies).

    Args:
        defect (|Defect|): A ``doped`` |Defect| object.

    Returns:
        int:
            Oxidation state of the original (bulk) site at the defect position,
            rounded to the nearest integer (for possible mixed-valence cases).
    """
    if defect.defect_type == core.DefectType.Vacancy:
        if not isinstance(defect.oxi_state, int | float):
            raise ValueError(
                f"Vacancy oxidation state (= {defect.oxi_state}) is not an integer or float (needed for "
                f"charge state guessing)! Please manually set the vacancy oxidation state."
            )
        return -round(defect.oxi_state)  # Vacancy.oxi_state is the negative of the site oxi state
    if defect.defect_type == core.DefectType.Substitution:  # bulk site oxi state:
        return round(defect.structure[defect.defect_site_index].specie.oxi_state)
    return 0  # interstitial; no original site occupant; neutral


def _get_possible_oxi_states(defect: Defect) -> dict:
    """
    Get the possible oxidation states and corresponding probabilities for a
    |Defect|.

    For substitutions and interstitials, these are the oxidation states of the
    defect element and their probabilities as given by their prevalence in the
    ICSD, while for vacancies, these are the effective 'oxidation states' of
    the vacant site and corresponding probabilities (see
    ``_get_vacancy_oxi_states()``).

    Args:
        defect (|Defect|): A ``doped`` |Defect| object.

    Returns:
        dict:
            A dictionary with possible oxidation states as keys and their
            probabilities as values.
    """
    if defect.defect_type == core.DefectType.Vacancy:  # throws informative error if invalid oxi_state:
        return _get_vacancy_oxi_states(get_defect_site_orig_oxi_state(defect))
    return {
        int(k): prob
        for k, prob in get_oxi_probabilities(defect.site.specie.symbol).items()
        if prob > 0.001  # at least 0.1% occurrence
    }


def _get_vacancy_oxi_states(orig_oxi: int) -> dict:
    """
    Get the effective 'oxidation states' and probabilities for a vacant site.

    Vacant site oxidation states between the original site oxidation state
    (i.e. the neutral vacancy) and zero (i.e. the fully-ionised vacancy) are
    assigned 100% probability, while those outside this range are assigned
    Gaussian-decaying probabilities (corresponding to Boltzmann-weighted
    quadratic energy cost of additional charge; ∝exp(-Δq²)) of ``(1/4)^(Δ²)``
    (``= exp(-Δ²·ln4)``) where ``Δ`` is the difference to the closest oxidation
    state in the 100% probability range, +1 if sign-reversed with respect to
    the original site oxidation state (i.e. effectively anionic on a cation
    site or vice versa; beyond 'fully-ionised').

    Args:
        orig_oxi (int): Oxidation state of the original (now vacant) site.

    Returns:
        dict:
            A dictionary with possible vacant site oxidation states as keys
            and their probabilities as values.
    """
    lower, upper = min(orig_oxi, 0), max(orig_oxi, 0)  # 100% probability range of vacant site oxi states
    # oxi state probability = exp(-Δ²·ln4) = (1/4)^(Δ²); Δ = distance from 100% range, +1 if sign-reversed:
    return {
        oxi: 0.25 ** ((max(lower - oxi, oxi - upper, 0) + (oxi * orig_oxi < 0)) ** 2)  # (1/4)^(Δ²)
        for oxi in range(lower - 10, upper + 11)
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


def _guess_complex_charge_states(
    defect_complex: DefectComplex, probability_threshold: float = 0.0075
) -> tuple[list[int], list[dict]]:
    r"""
    Guess the possible stable charge states of a defect `complex`, from those
    of its constituent point defects; see ``guess_defect_charge_states``.

    The charge state probabilities of the constituents (from
    ``charge_state_probability``) are treated as independent distributions and
    convolved such that the probability of a charge state of a complex is the
    sum over probabilities of each decomposition of the charge over its constituents.

    The result is rescaled so that the neutral
    state has a probability of 1 (as for point defects), and the same
    ``probability_threshold`` is applied.

    TODO this disfavours high charge states more strongly, not less?

    Args:
        defect_complex (DefectComplex):
            |DefectComplex| object to guess the charge states of.
        probability_threshold (float):
            Probability threshold for including charge states, applied both to
            the constituents and to the combined probabilities. (Default: 0.0075)

    Returns:
        tuple[list[int], list[dict]]:
            The guessed charge states, and a list of dictionaries of the input
            & computed values used to determine each charge state probability
            (ordered by decreasing probability).
    """
    constituent_probabilities = []  # {charge state: probability} for each constituent
    for constituent in defect_complex.defects:
        _charge_states, log = cast(
            "tuple[list[int], list[dict]]",
            guess_defect_charge_states(
                constituent,
                probability_threshold=probability_threshold,
                return_log=True,
            ),
        )
        # charge state probability is only defined where there is a corresponding common oxi state
        # of the element, plus the neutral state (and for vacancy charge states, all probabilities
        # in the range are set to 1). the intermediate states filled in between such charge states
        # are not included (ie probability 0) here.
        probabilities = {entry["input_parameters"]["charge_state"]: entry["probability"] for entry in log}
        probabilities[0] = 1.0
        total_probability = sum(probabilities.values())  # normalise
        constituent_probabilities.append({q: p / total_probability for q, p in probabilities.items()})

    # convolution of the constituent distributions, accumulating one constituent at a time
    # for efficiency. also keep most probable distribution for log
    # {total charge state: (probability, most probable decomposition, its probability)}
    distribution: dict[int, tuple[float, tuple[int, ...], float]] = {0: (1.0, (), 1.0)}
    for probabilities in constituent_probabilities:
        combined: dict[int, tuple[float, tuple[int, ...], float]] = {}
        for total, (probability, combo, combo_probability) in distribution.items():
            for charge_state, factor in probabilities.items():
                prev_probability, prev_combo, prev_combo_probability = combined.get(
                    total + charge_state, (0.0, (), 0.0)
                )
                joint = combo_probability * factor
                combined[total + charge_state] = (
                    prev_probability + probability * factor,  # sum over decompositions
                    *(
                        ((*combo, charge_state), joint)
                        if joint > prev_combo_probability
                        else (prev_combo, prev_combo_probability)
                    ),
                )
        distribution = combined

    # rescale so that the neutral state has a probability of 1, to match point defects
    neutral_probability = distribution[0][0]
    distribution = {
        q: (p / neutral_probability, combo, combo_probability)
        for q, (p, combo, combo_probability) in distribution.items()
    }

    charge_state_list = [q for q, (p, _, _) in distribution.items() if p > probability_threshold]
    if not charge_state_list:
        charge_state_list = [max(distribution, key=lambda q: distribution[q][0])]  # most probable
    # set charge_state_range to min/max of range, ensuring range is extended to 0, as for point
    # defects (so that any intermediate charge states are included):
    charge_state_range = (min(*charge_state_list, 0), max(*charge_state_list, 0))

    charge_state_guessing_log = [
        {
            "input_parameters": {
                "charge_state": int(charge_state),
                "most_probable_constituent_charge_states": [int(q) for q in combo],
            },
            "probability_factors": {
                "constituent_probabilities": [
                    probabilities[q]
                    for probabilities, q in zip(constituent_probabilities, combo, strict=True)
                ]
            },
            "probability": probability,
            "probability_threshold": probability_threshold,
        }
        for charge_state, (probability, combo, _combo_probability) in sorted(
            distribution.items(), key=lambda item: item[1][0], reverse=True
        )
    ]

    return list(range(charge_state_range[0], charge_state_range[1] + 1)), charge_state_guessing_log


def guess_defect_charge_states(
    defect: Defect, probability_threshold: float = 0.0075, return_log: bool = False
) -> list[int] | tuple[list[int], list[dict]]:
    """
    Guess the possible stable charge states of a defect.

    This function estimates the probabilities of the charge states of a defect,
    using the probability of the corresponding defect element oxidation states,
    the magnitudes of the charge states, and the maximum magnitude of the host
    oxidation states (i.e. how 'charged' the host is), and returns a list of
    charge states that have an estimated probability greater than the
    ``probability_threshold``.

    Disfavours large (absolute) charge states, low probability oxidation states
    and greater charge/oxidation state magnitudes than that of the host. The
    neutral charge state is always included, as well as the fully-ionised
    charge state (a.k.a. the natural charge state / 'defect oxidation state',
    i.e. the simple oxidation state difference) for intrinsic defects. If no
    charge states have probabilities above ``probability_threshold``, the most
    probable charge state is included, and if the resulting charge state range
    is only {0} (e.g. for elemental systems) or the defect is a homovalent
    substitution then (-1, 0, +1) at minimum is included.

    For substitutions and interstitials, the oxidation state probabilities are
    those of the corresponding defect element (as given by their prevalence in
    the ICSD), while for vacancies, effective 'oxidation states' of the vacant
    site are used, with 100% probability for oxidation states between that of
    the original site (i.e. the neutral vacancy) and zero (i.e. the
    fully-ionised vacancy), and Gaussian-decaying probabilities (corresponding
    to Boltzmann-weighted quadratic energy cost of additional charge;
    ∝exp(-Δq²)) of ``(1/4)^(Δ²)`` (``= exp(-Δ²·ln4)``) where ``Δ`` is the
    difference to the closest oxidation state in the 100% probability range, +1
    if sign-reversed with respect to the original site oxidation state (i.e.
    effectively anionic on a cation site or vice versa; beyond
    'fully-ionised'). For specific details on the probability functions
    employed, see the :func:`charge_state_probability()` function.

    For a defect `complex`, the charge states are instead combined from those of
    its constituent point defects; see ``_guess_complex_charge_states``.

    These probability functions were found to give optimal performance in terms
    of efficiency and completeness when tested against other approaches (see
    the ``doped`` JOSS paper: https://doi.org/10.21105/joss.06433) but of
    course may not be perfect in all cases, so make sure to critically consider
    the estimated charge states for your system!

    Args:
        defect (|Defect|):
            ``doped`` |Defect| object. Note that ``defect.structure`` must have
            oxidation states assigned (e.g. with
            ``add_oxidation_state_by_element()``) for charge state guessing;
            this is handled automatically when generating defects with
            |DefectsGenerator|.
        probability_threshold (float):
            Probability threshold for including defect charge states. Default
            is 0.0075.
        return_log (bool):
            If ``True``, returns a tuple of the defect charge states and a list
            of dictionaries of input & computed values used to determine charge
            state probabilities, including near-miss charge states (with
            probabilities above 10% of ``probability_threshold``). Default is
            ``False``.

    Returns:
        List of defect charge states (int) or a tuple of the defect charge
        states (list) and a list of dictionaries of input & computed values
        used to determine charge state probability.
    """
    # Could consider band gap magnitude as well, by pulling from Materials Project. Smaller gaps mean
    # extreme charge states less likely. Would mean querying the MP database here though; not ideal.
    if isinstance(defect, DefectComplex):  # combine the constituent point defect charge states
        charge_states, charge_state_guessing_log = _guess_complex_charge_states(
            defect, probability_threshold=probability_threshold
        )
        return (charge_states, charge_state_guessing_log) if return_log else charge_states

    orig_oxi = get_defect_site_orig_oxi_state(defect)  # checks vacancy oxi state validity first
    max_host_oxi_magnitude = round(max(abs(site.specie.oxi_state) for site in defect.structure))
    possible_oxi_states = _get_possible_oxi_states(defect)
    possible_charge_states = _get_charge_states(
        possible_oxi_states, orig_oxi, max_host_oxi_magnitude, return_log=True
    )  # compute charge state probabilities based on doped heuristic scoring functions
    possible_charge_states = {  # prune negligible entries (e.g. vacancy oxi state tails), for a tidy log
        k: v for k, v in possible_charge_states.items() if v["probability"] > 0.1 * probability_threshold
    }

    # checks; ensure fully-ionised included, and...
    ensure_pm1 = False  # ...ensure at least (-1, 0, +1)? (homovalent substitutions, {0} charge ranges)
    if defect.defect_type in (core.DefectType.Substitution, core.DefectType.Interstitial):
        # if defect element is present in host structure (i.e. intrinsic), ensure fully-ionised included:
        defect_el_oxi_states = [
            site.specie.oxi_state
            for site in defect.structure
            if site.specie.symbol == defect.site.specie.symbol
        ]
        if defect_el_oxi_states:
            defect_el_oxi_in_struct = round(float(np.mean(defect_el_oxi_states)))
            defect_oxi_state = defect_el_oxi_in_struct - orig_oxi  # fully-ionised / natural charge state
            charge_state_list = [
                k for k, v in possible_charge_states.items() if v["probability"] > probability_threshold
            ]
            lower, upper = (
                (min(charge_state_list), max(charge_state_list)) if charge_state_list else (0, 0)
            )
            if not lower <= defect_oxi_state <= upper:
                possible_charge_states[defect_oxi_state] = charge_state_probability(
                    defect_oxi_state,
                    defect_el_oxi_in_struct,
                    1,  # 100% oxidation state probability
                    max_host_oxi_magnitude,
                    return_log=True,
                )
            ensure_pm1 = (  # antisite of two equal oxi state elements, with one-sided charge range?
                defect.defect_type == core.DefectType.Substitution
                and defect_oxi_state == 0
                and (lower >= 0 or upper <= 0)
            )

    # determine charge state range from probability dict and chosen threshold:
    if charge_state_list := [
        k for k, v in possible_charge_states.items() if v["probability"] > probability_threshold
    ]:
        charge_state_range = (min(charge_state_list), max(charge_state_list))
    else:  # no charge states over threshold; take most probable with probability > 0.1*threshold, or 0
        over_log_threshold = [
            k for k, v in possible_charge_states.items() if v["probability"] > 0.1 * probability_threshold
        ]  # (only the injected fully-ionised entry can be below this, having skipped the earlier pruning)
        most_likely_charge_state = max(
            over_log_threshold, key=lambda k: possible_charge_states[k]["probability"], default=0
        )
        charge_state_range = (most_likely_charge_state, most_likely_charge_state)

    if ensure_pm1 or charge_state_range == (0, 0):  # ensure at least (-1, 0, +1) included for these cases:
        charge_state_range = (min(charge_state_range[0], -1), max(charge_state_range[1], 1))
        for charge_state in (-1, 0, 1):
            if probability_dict := possible_charge_states.get(charge_state):
                probability_dict["probability"] = 1

    # set charge_state_range to min/max of range, ensuring range is extended to 0:
    charge_state_range = (min(charge_state_range[0], 0), max(charge_state_range[1], 0))
    guessed_charge_states = list(range(charge_state_range[0], charge_state_range[1] + 1))

    for probability_dict in possible_charge_states.values():
        probability_dict["probability_threshold"] = probability_threshold

    if return_log:  # sorted by (final) probability, so forced-inclusion (probability = 1) entries at top:
        return guessed_charge_states, sorted(
            possible_charge_states.values(), key=lambda d: d["probability"], reverse=True
        )
    return guessed_charge_states


def get_ideal_supercell_matrix(
    structure: Structure,
    min_image_distance: float = 10.0,
    min_atoms: int = 50,
    force_cubic: bool = False,
    force_diagonal: bool = False,
    ideal_threshold: float = 0.1,
    verbose: bool = False,
    pbar: tqdm | None = None,
) -> np.ndarray:
    """
    Determine the ideal supercell matrix for a given structure, based on the
    minimum image distance, minimum number of atoms and ``ideal_threshold`` for
    further expanding if a diagonal expansion of the primitive/conventional
    cell is possible.

    The ideal supercell is the smallest possible supercell which has a minimum
    image distance (i.e. minimum distance between periodic images of
    atoms/sites in a lattice) greater than ``min_image_distance`` (default = 10
    Å -- which is a typical threshold value used in QM/DFT defect supercell
    calculations) and a number of atoms greater than ``min_atoms`` (default =
    50). Once these criteria have been reached, ``doped`` will then continue
    searching up to supercell sizes (numbers of atoms) ``1 + ideal_threshold``
    times larger (rounded up) to see if they return a diagonal expansion of the
    primitive/conventional cell (which can make later visualisation and
    analysis much easier) -- if so, this larger supercell will be returned.

    This search for the ideal supercell transformation matrix is performed
    using the ``find_ideal_supercell`` function from ``doped.utils.supercells``
    (see its docstring for more details), which efficiently scans over possible
    supercell matrices and identifies that with the minimum image distance and
    most cubic-like supercell shape. The advantage of this over that in
    ``pymatgen-analysis-defects`` is that it avoids the
    ``find_optimal_cell_shape`` function from ``ASE`` (which currently does not
    work for rotated matrices, is inefficient, and optimises based on cubic-
    like shape rather than minimum image distance), giving greatly reduced
    supercell sizes for a given minimum image distance.

    If ``force_cubic`` or ``force_diagonal`` are ``True``, then the
    ``CubicSupercellTransformation`` from ``pymatgen`` is used to identify any
    simple near-cubic supercell transformations which satisfy the minimum image
    distance and atom number criteria.

    Args:
        structure (|Structure|):
            Primitive unit cell structure to generate supercell for.
        min_image_distance (float):
            Minimum image distance in Å of the supercell (i.e. minimum distance
            between periodic images of atoms/sites in the lattice).
            (Default = 10.0)
        min_atoms (int):
            Minimum number of atoms allowed in the supercell.
            (Default = 50)
        force_cubic (bool):
            Enforce usage of ``CubicSupercellTransformation`` from ``pymatgen``
            for supercell generation. (Default = False)
        force_diagonal (bool):
            If True, return a transformation with a diagonal transformation
            matrix. (Default = False)
        ideal_threshold (float):
            Threshold for increasing supercell size (beyond that which
            satisfies ``min_image_distance`` and `min_atoms``) to achieve an
            ideal supercell matrix (i.e. a diagonal expansion of the primitive
            or conventional cell). Supercells up to
            ``1 + perfect_cell_threshold`` times larger (rounded up) are
            trialled, and will instead be returned if they yield an ideal
            transformation matrix.
            (Default = 0.1; i.e. 10% larger than the minimum size)
        verbose (bool):
            Whether to print out extra information about the supercell search.
            (Default = False)
        pbar (tqdm):
            ``tqdm`` progress bar object to update (for internal ``doped``
            usage). Default is ``None``.

    Returns:
        Ideal supercell matrix (``np.ndarray``).
    """
    if force_cubic or force_diagonal:
        cst = CubicSupercellTransformation(
            min_atoms=min_atoms,
            min_length=min_image_distance,
            force_diagonal=force_diagonal,
        )

        try:
            cst.apply_transformation(structure)
            if cst.transformation_matrix is None:
                raise RuntimeError(
                    "CubicSupercellTransformation.apply_transformation() did not return a transformation "
                    "matrix."
                )
            return cst.transformation_matrix

        except Exception as e:  # cubic supercell generation failed, used doped algorithm
            print(
                f"Could not find a suitable cubic supercell within the limits:\nmin_atoms = {min_atoms}, "
                f"min_image_distance = {min_image_distance}.\nGot exception: {e!r}\n"
                f"Attempting doped supercell generation algorithm..."
            )

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
        verbose=verbose,
    )

    while best_min_dist < min_image_distance:
        target_size += 1
        if pbar is not None:
            pbar.set_description(
                f"Best min distance: {best_min_dist:.2f} Å, trialling size = {target_size} unit cells..."
            )
        optimal_P, best_min_dist = supercells.find_ideal_supercell(
            structure.lattice.matrix,
            target_size=target_size,
            return_min_dist=True,
            verbose=verbose,
        )

    # check if supercell matrix is ideal (diagonal expansion of primitive or conventional cells), otherwise
    # extend search by threshold amount:
    if round(supercells._min_sum_off_diagonals(structure, optimal_P)) != 0:
        max_target_size = int(np.ceil(target_size * (1 + ideal_threshold)))
        for alt_target_size in range(target_size + 1, max_target_size + 1):
            if pbar is not None:
                pbar.set_description(
                    f"Best min distance: {best_min_dist:.2f} Å, trialling size = {alt_target_size} unit "
                    f"cells..."
                )
            alt_optimal_P, alt_best_min_dist = supercells.find_ideal_supercell(
                structure.lattice.matrix,
                target_size=alt_target_size,
                return_min_dist=True,
                verbose=verbose,
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
            f"Best min distance: {best_min_dist:.2f} Å, with size = {target_size} unit cells"
        )

    return optimal_P


def get_ideal_complex_supercell_matrix(
    structure: Structure,
    cart_coords: np.ndarray,
    min_image_distance: float | None = None,
    min_atoms: int = 50,
    force_cubic: bool = False,
    force_diagonal: bool = False,
    ideal_threshold: float = 0.1,
    verbose: bool = False,
    pbar: tqdm | None = None,
) -> np.ndarray:
    """
    Determine the ideal supercell matrix for a defect complex in the given
    structure, based on the complex minimum image distance (i.e. the minimum
    distance between any constituent point defect and a constituent point
    defect of a periodic image), minimum number of atoms and
    ``ideal_threshold``.

    The equivalent of ``get_ideal_supercell_matrix`` for defect complexes.
    In addition to the different metric, the supercell search is currently
    exhaustive, instead of based on a scan around a guessed FCC/SC shape,
    which may not be good candidates for a complex defect.

    Sizes are preferred which give a diagonal expansion of the
    primitive/conventional cell, then which give the largest minimum image distance,
    then which are smallest.

    Args:
        structure (|Structure|): Unit cell structure to generate a supercell of.
        cart_coords (np.ndarray):
            ``(n, 3)`` array of Cartesian coordinates of the constituent
            point defect sites of the complex, unwrapped.
        min_image_distance (float):
            Minimum complex minimum image distance (in Å) for the supercell.
            If ``None`` (default), ``max(10, 2 * span)`` is used, where
            ``span`` is the largest separation between any two constituent
            sites of the complex -- scaling the target with the size of the
            complex, so that extended complexes are still well-separated from
            their periodic images, while reducing to the usual point defect
            default of 10 Å for compact complexes (spans up to 5 Å).
        min_atoms (int):
            Minimum number of atoms in the supercell. (Default = 50)
        force_cubic (bool):
            Whether to return the most cubic supercell satisfying the above
            criteria, rather than that with the largest complex minimum image
            distance. Note that, unlike for point defects, this is done by
            doped using the same exhaustive search rather than using
            the ``pymatgen`` ``CubicSupercellTransformation``, which optimises
            the single-site minimum image distance. (Default = False)
        force_diagonal (bool):
            As ``force_cubic``, but additionally requiring a diagonal supercell
            transformation matrix. (Default = False)
        ideal_threshold (float):
            Threshold for scanning larger supercell sizes, as a fraction of the
            smallest satisfactory size. Not used if ``force_cubic`` or
            ``force_diagonal`` is set, as the supercell shape is then already
            constrained. (Default = 0.1; i.e. 10% larger)
        verbose (bool):
            Whether to print out extra information about the supercell search.
            (Default = False)
        pbar (tqdm):
            ``tqdm`` progress bar object to update (for internal ``doped``
            usage). Default is ``None``.

    Returns:
        Ideal supercell matrix (``np.ndarray``).
    """
    span = np.max(np.linalg.norm(cart_coords[:, None] - cart_coords[None, :], axis=-1))
    if min_image_distance is None:  # scale the target with the size of the complex; span = 0 for a
        # single site, giving the same 10 Å default as for point defects
        min_image_distance = max(10.0, 2 * span)
    elif min_image_distance < 1.1 * span:  # can only be triggered by an explicit request, as the
        # default above is always at least 2 * span
        warnings.warn(
            f"The requested complex minimum image distance ({min_image_distance:.2f} Å) is not "
            f"appreciably larger than the span of the complex ({span:.2f} Å; the largest separation "
            f"between its constituent sites). The returned supercell will then have constituent sites "
            f"which are as close (or closer) to those of neighbouring complex images as they are to "
            f"each other, so the complex is not well-defined in this supercell. A minimum image "
            f"distance of at least ~2x the complex span is recommended.",
            UserWarning,
            stacklevel=2,
        )

    # smallest size which could possibly satisfy both criteria
    target_size = max(
        int(np.ceil(min_atoms / len(structure))),
        supercells._get_min_complex_target_size(structure.lattice.matrix, cart_coords, min_image_distance),
    )

    def _trial(size: int) -> tuple[np.ndarray, float]:
        if pbar is not None:
            pbar.set_description(f"Trialling size = {size} unit cells...")
        return supercells.find_ideal_complex_supercell(  # type: ignore[return-value]
            structure.lattice.matrix,
            cart_coords,
            target_size=size,
            return_min_dist=True,
            force_cubic=force_cubic,
            force_diagonal=force_diagonal,
            verbose=verbose,
        )

    # trial target sizes until min image dist hit
    optimal_P, best_min_dist = _trial(target_size)
    while best_min_dist < min_image_distance:
        target_size += 1
        optimal_P, best_min_dist = _trial(target_size)

    if force_cubic or force_diagonal:  # constrained - no gain by searching higher?
        return optimal_P

    # test up to threshold
    candidates = [(target_size, optimal_P, best_min_dist)]
    for alt_target_size in range(target_size + 1, int(np.ceil(target_size * (1 + ideal_threshold))) + 1):
        alt_optimal_P, alt_min_dist = _trial(alt_target_size)
        if alt_min_dist >= min_image_distance:
            candidates.append(
                (
                    alt_target_size,
                    supercells._check_and_return_scalar_matrix(alt_optimal_P, structure.lattice.matrix),
                    alt_min_dist,
                )
            )

    target_size, optimal_P, best_min_dist = min(  # diagonal, then max distance, then smallest size
        candidates,
        key=lambda x: (round(supercells._min_sum_off_diagonals(structure, x[1])) != 0, -x[2], x[0]),
    )

    if pbar is not None:
        pbar.set_description(
            f"Best min distance: {best_min_dist:.2f} Å, with size = {target_size} unit cells"
        )

    return optimal_P


class DefectsGenerator(MSONable):
    """
    Class for generating ``doped`` |DefectEntry| objects.
    """

    def __init__(
        self,
        structure: "Structure | Atoms | PathLike",
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
        Generates ``doped`` |DefectEntry| objects for defects in the input host
        structure. By default, generates all intrinsic defects, but extrinsic
        defects (impurities) can also be created using the ``extrinsic``
        argument.

        Interstitial sites are generated using Voronoi tessellation by default
        (found to be the most reliable) using the ``get_interstitial_sites``
        function, which also generates adsorbate sites if the structure is
        determined to be a slab/layer/rod. This can be controlled using the
        ``interstitial_gen_kwargs`` argument. Alternatively, a list of
        interstitial sites (or single interstitial site) can be manually
        specified using the ``interstitial_coords`` argument.

        By default, supercells are generated for each defect using the
        ``doped`` ``get_ideal_supercell_matrix()`` function (see docstring),
        with default settings of ``min_image_distance = 10`` (minimum distance
        between periodic images of 10 Å), ``min_atoms = 50`` (minimum 50 atoms
        in the supercell) and ``ideal_threshold = 0.1`` (allow up to 10% larger
        supercell if it is a diagonal expansion of the primitive or
        conventional cell). This uses a custom algorithm in ``doped`` to
        efficiently search over possible supercell transformations and identify
        that with the minimum number of atoms (hence computational cost) that
        satisfies the minimum image distance, number of atoms and
        ``ideal_threshold`` constraints. These settings can be controlled by
        specifying keyword arguments with ``supercell_gen_kwargs``, which are
        passed to ``get_ideal_supercell_matrix()`` (e.g. for a minimum image
        distance of 15 Å with at least 100 atoms, ``supercell_gen_kwargs``
        would be: ``{'min_image_distance': 15, 'min_atoms': 100}``). If the
        input structure already satisfies these constraints (for the same
        number of atoms as the ``doped``-generated supercell), then it will be
        used. Alternatively if ``generate_supercell = False``, then no
        supercell is generated and the input structure is used as the defect &
        bulk supercell. (Note this may give a slightly different (but fully
        equivalent) set of coordinates).

        The algorithm for determining defect entry names is to use the
        ``pymatgen`` defect name (e.g. ``v_Cd``, ``Cd_Te`` etc.) for vacancies
        / antisites / substitutions, unless there are multiple inequivalent
        sites for the defect, in which case the point group of the defect site
        is appended (e.g. ``v_Cd_Td``, ``Cd_Te_Td`` etc.), and if this is still
        not unique, then element identity and distance to the nearest neighbour
        of the defect site is appended (e.g. ``v_Cd_Td_Te2.83``,
        ``Cd_Te_Td_Cd2.83`` etc.). For interstitials, the same naming scheme is
        used, but the point group is always appended to the ``pymatgen`` name.

        Possible charge states for the defects are estimated using the
        probability of the corresponding defect element oxidation state, the
        magnitude of the charge state, and the maximum magnitude of the host
        oxidation states (i.e. how 'charged' the host is). Large (absolute)
        charge states, low probability oxidation states and/or greater
        charge/oxidation state magnitudes than that of the host are
        disfavoured. This can be controlled using the ``probability_threshold``
        (default = 0.0075) key in the ``charge_state_gen_kwargs`` parameter,
        which is passed to the :func:`guess_defect_charge_states()` function.
        The input and computed values used to guess charge state probabilities
        are provided in the ``DefectEntry.charge_state_guessing_log``
        attributes. See docs for examples of modifying the generated charge
        states, and the :func:`charge_state_probability()` docstring for more
        details on the charge state guessing algorithm. The ``doped`` algorithm
        was found to give optimal performance in terms of efficiency and
        completeness (see JOSS paper), but of course may not be perfect in all
        cases, so make sure to critically consider the estimated charge states
        for your system!

        Note that Wyckoff letters can depend on the ordering of elements in the
        conventional standard structure, for which doped uses the ``spglib``
        convention.

        Args:
            structure (|Structure|):
                Structure of the host material, either as a ``pymatgen``
                |Structure|, ``ASE`` |Atoms| or path to a structure file
                (e.g. ``CONTCAR``). If this is not the primitive unit cell, it
                will be reduced to the primitive cell for defect generation,
                before supercell generation.
            extrinsic (str | list | dict):
                List or dict of elements (or string for single element) to be
                used for extrinsic defect generation (i.e. dopants/impurities).
                If a list is provided, all possible substitutional defects for
                each extrinsic element will be generated. If a dict is
                provided, the keys should be the host elements to be
                substituted, and the values the extrinsic element(s) to
                substitute in; as a string or list. In both cases, all possible
                extrinsic interstitials are generated.
            interstitial_coords (list):
                List of fractional coordinates (corresponding to the input
                structure), or a single set of fractional coordinates, to use
                as interstitial defect site(s). Default (when
                ``interstitial_coords`` not specified) is to automatically
                generate interstitial sites using Voronoi tessellation.
                The input ``interstitial_coords`` are converted to
                ``DefectsGenerator.prim_interstitial_coords_mult_and_equiv_coords``,
                which are the corresponding fractional coordinates in
                ``DefectsGenerator.primitive_structure`` (which is used for
                defect generation), along with the multiplicity and equivalent
                coordinates, sorted according to the ``doped`` convention.
            generate_supercell (bool):
                Whether to generate a supercell for the output defect entries
                (using the custom algorithm in ``doped`` which efficiently
                searches over possible supercell transformations and identifies
                that with the minimum number of atoms (hence computational
                cost) that satisfies the minimum image distance, number of
                atoms and ``ideal_threshold`` constraints -- which can be
                controlled with ``supercell_gen_kwargs``).
                If ``False``, then the input structure is used as the defect &
                bulk supercell (note this may give a slightly different (but
                fully equivalent) set of coordinates).
                Default is ``True``.
            charge_state_gen_kwargs (dict):
                Keyword arguments to pass to the ``guess_defect_charge_states``
                function (such as ``probability_threshold``; default = 0.0075)
                to control defect charge state generation.
            supercell_gen_kwargs (dict):
                Keyword arguments to pass to the ``get_ideal_supercell_matrix``
                function (such as ``min_image_distance`` (default = 10),
                ``min_atoms`` (default = 50), ``ideal_threshold`` (default =
                0.1), ``force_cubic`` -- which enforces a (near-)cubic
                supercell output (default = False), or ``force_diagonal``
                (default = False)).
            interstitial_gen_kwargs (dict, bool):
                Keyword arguments for :func:`get_interstitial_sites()` (such as
                ``min_dist`` (0.9 Å, or 0.5 Å for Hydrogen), ``clustering_tol``
                (0.8 Å), ``symm_pref_dist_factor`` (0.85), ``stol`` (0.32),
                ``tight_stol`` (0.02), ``symprec`` (0.01), ``vacuum_radius``
                (1.5 * bulk bond length), ``include_unique_wyckoffs`` (False),
                ``include_bond_centres`` (False, or True for Hydrogen),
                ``bond_tol`` (0.2) -- see its docstring, parentheses indicate
                default values), or ``InterstitialGenerator`` (where only
                ``min_dist`` is used) if ``interstitial_coords`` is specified.
                If set to ``False``, interstitial generation will be skipped
                entirely.
            target_frac_coords (list):
                Defects are placed at the closest equivalent site to these
                fractional coordinates in the generated supercells. Default is
                ``[0.5, 0.5, 0.5]`` (i.e. the supercell centre, to aid
                visualisation).
            processes (int):
                Number of processes to use for multiprocessing. If not set,
                defaults to one less than the number of CPUs available.
            **kwargs:
                Additional keyword arguments for defect generation. Options:

                - ``{defect}_elements`` where ``{defect}`` is ``vacancy``,
                  ``substitution``, or ``interstitial``, in which cases only
                  those defects of the specified elements will be generated
                  (where ``{defect}_elements`` is a list of element symbol
                  strings). Setting ``{defect}_elements`` to an empty list will
                  skip defect generation for that defect type entirely.
                - ``{defect}_charge_states`` to specify the charge states to use
                  for all defects of that type (as a list of integers).
                - ``neutral_only`` to only generate neutral charge states.
                - ``symprec`` for ``get_sga_and_symprec()``,
                  ``get_primitive_structure``, defect object initialisation
                  etc. Default is ``0.01``.
                - Keyword arguments for ``get_all_equiv_sites``, such as
                  ``dist_tol_factor``.

        Key attributes:
            defect_entries (dict):
                Dictionary of ``{defect_species: DefectEntry}`` for all defect
                entries (with charge state and supercell properties) generated.
            defects (dict):
                Dictionary of ``{defect_type: [Defect, ...]}`` for all defect
                objects generated.
            primitive_structure (|Structure|):
                Primitive cell structure of the host used to generate defects.
            supercell_matrix (Matrix):
                Matrix to generate defect/bulk supercells from the primitive
                cell structure.
            bulk_supercell (|Structure|):
                Supercell structure of the host (equal to
                ``self.primitive_structure * self.supercell_matrix``).
            conventional_structure (|Structure|):
                Conventional cell structure of the host according to the Bilbao
                Crystallographic Server (BCS) definition, used to determine
                defect site Wyckoff labels and multiplicities.
            prim_interstitial_coords_mult_and_equiv_coords (list):
                List of interstitial coordinates in the primitive cell, their
                multiplicity and the equivalent fractional coordinates, as a
                list of tuples.

            |DefectsGenerator| input parameters are also set as attributes.
        """
        # attribute setup:
        self.defects: dict[str, list[Defect]] = {}  # {defect_type: [Defect, ...]}
        self.defect_entries: dict[str, DefectEntry] = {}  # {defect_species: DefectEntry}
        if isinstance(structure, str | PathLike):
            structure = Structure.from_file(structure)
        elif not isinstance(structure, Structure):  # assume ASE Atoms
            structure = Structure.from_ase_atoms(structure)

        if extra_site_props := set(structure.site_properties) - {"magmom"}:
            # e.g. ``velocities``/``momenta``, ``selective_dynamics`` or ``tags`` from (ASE) reads of VASP
            # CONTCARs / MD frames; irrelevant for defect generation and causes various downstream issues:
            structure = structure.copy()  # don't modify input structure
            for prop in extra_site_props:
                structure.remove_site_property(prop)

        if isotopes := [sp for sp in structure.composition if getattr(sp, "name", sp.symbol) != sp.symbol]:
            # pymatgen only partly keeps host D/T (giving mixes like v_D & H_Li), so use parent elements:
            warnings.warn(
                f"Named isotopes {[sp.name for sp in isotopes]} in the host structure are treated as "
                "their parent element (e.g. D as H), as in written VASP inputs."
            )
            structure = structure.copy()  # don't modify input structure
            structure.replace_species(
                {
                    sp: Species(sp.symbol, sp.oxi_state) if isinstance(sp, Species) else sp.symbol
                    for sp in isotopes
                }
            )

        self.structure = structure
        self.extrinsic = extrinsic if extrinsic is not None else []
        self.kwargs = kwargs
        if isinstance(self.kwargs, dict):
            for kwarg, val in self.kwargs.items():
                if isinstance(val, set):
                    self.kwargs[kwarg] = list(val)  # convert sets to lists for JSON serialisation

        self.interstitial_coords = (  # list of lists, also for a single set of coords (``[]`` if ``None``)
            np.reshape(interstitial_coords, (-1, 3)).tolist() if interstitial_coords is not None else []
        )

        self.prim_interstitial_coords_mult_and_equiv_coords: list[tuple] = []
        self.generate_supercell = generate_supercell
        self.charge_state_gen_kwargs = (
            dict(charge_state_gen_kwargs) if charge_state_gen_kwargs is not None else {}
        )
        if "padding" in self.charge_state_gen_kwargs:  # TODO: remove in v4.1
            warnings.warn(
                "The `padding` option (for vacancy charge state guessing) was removed in doped v4.0, with "
                "vacancy charge states now determined by the same probability-based scheme as other "
                "defects (see the `guess_defect_charge_states()` docstring) -- ignoring `padding`; use "
                "`probability_threshold` to control the guessed charge state ranges.",
                DeprecationWarning,
                stacklevel=2,
            )
            del self.charge_state_gen_kwargs["padding"]

        self.supercell_gen_kwargs: dict[str, Any] = {
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

        # setup progress bar:
        pbar = tqdm(
            total=100, bar_format="{desc}{percentage:.1f}%|{bar}| [{elapsed},  {rate_fmt}{postfix}]"
        )  # tqdm progress bar. 100% is completion

        try:  # put code in try/except block so progress bar always closed if interrupted
            primitive_structure, sga = self._sanity_checks_and_symmetry_determination(
                pbar
            )  # sets self.symprec
            pbar.update(5)  # 5% of progress bar

            self._supercell_generation_and_checks(
                primitive_structure=primitive_structure,
                specified_min_image_distance=specified_min_image_distance,
                pbar=pbar,
                sga=sga,
            )  # sets self.primitive_structure and self.supercell_matrix

            self._get_and_set_bulk_oxi_states(
                pbar
            )  # sets self._bulk_oxi_states, may overwrite self.primitive_structure

            self._define_bulk_supercell_and_check_min_image_distance(
                specified_min_image_distance=specified_min_image_distance
            )  # sets self.bulk_supercell and updates self.min_image_distance
            pbar.update(10)  # 15% of progress bar

            # Generate defects
            self._generate_vacancies(pbar)  # sets self.defects["vacancies"]
            pbar.update(5)  # 20% of progress bar

            self.host_element_list, self.extrinsic_elements = self._get_host_and_extrinsic_elements()
            self._generate_substitutions(pbar)  # sets self.defects["substitutions"]

            self._element_list = self.host_element_list + self.extrinsic_elements  # all elements in system
            self._generate_interstitials(
                pbar
            )  # sets self.defects["interstitials"], self.prim_interstitial_coords_mult_and_equiv_coords
            pbar.update(15)  # 45% of progress bar, generating interstitials typically takes the longest

            # Generate DefectEntry objects:
            pbar.set_description("Determining conventional structure and Wyckoff labels")
            wyckoff_dict = self._get_conventional_structure_and_wyckoff_label_dict(
                pbar
            )  # sets self.conventional_structure, self._BilbaoCS_conv_cell_vector_mapping
            pbar.set_description("Generating DefectEntry objects and symmetry information")
            defect_entry_dict = self._generate_neutral_defect_entries(pbar, processes, wyckoff_dict)
            self._set_defects_attr_dict(defect_entry_dict)  # sets self.defects
            pbar.update(5)  # 95% of progress bar

            self._guess_and_set_charge_states(defect_entry_dict, pbar)  # sets self.defect_entries
            # pbar now 100%

            # sort defects and defect entries for deterministic behaviour:
            self.defects = _sort_defects(self.defects, element_list=self._element_list)
            self.defect_entries = sort_defect_entries(self.defect_entries, element_list=self._element_list)

            # clean up progress bar:
            if not isinstance(pbar, MagicMock) and pbar.total - pbar.n > 0:
                pbar.update(pbar.total - pbar.n)  # 100%

        finally:
            pbar.close()

        self.defect_generator_info()

    def _sanity_checks_and_symmetry_determination(
        self, pbar: tqdm
    ) -> tuple[Structure, symmetry.SpacegroupAnalyzer]:
        if len(self.structure) == 1 and not self.generate_supercell:
            # raise error if only one atom in primitive cell and no supercell being generated:
            raise ValueError(
                "Input structure has only one site, so cannot generate defects without supercell (i.e. "
                "with generate_supercell=False)! Vacancy defect will give empty cell!"
            )

        # Reduce structure to primitive cell for efficient defect generation
        # same symprec as defect generators in pymatgen-analysis-defects:
        pbar.set_description("Getting primitive structure")
        sga, self.symprec = symmetry.get_sga_and_symprec(
            self.structure, symprec=self.kwargs.get("symprec", 0.01)
        )
        if sga.get_space_group_number() == 1:  # print sanity check message
            print(
                "Note that the detected symmetry of the input structure is P1 (i.e. only "
                "translational symmetry). If this is not expected (i.e. host system is not "
                "disordered/defective), then you should check your input structure!"
            )
        if self.symprec != self.kwargs.get("symprec", 0.01):  # default
            input_symprec = self.kwargs.get("symprec", 0.01)
            warnings.warn(
                f"\nSymmetry determination failed for the input symprec value of {input_symprec}, "
                f"but succeeded with symprec = {self.symprec}, which will be used for symmetry "
                f"determination functions here."
            )

        prim_struct = symmetry.get_primitive_structure(self.structure, symprec=self.symprec)
        primitive_structure = symmetry._round_struct_coords(
            prim_struct if prim_struct.num_sites < self.structure.num_sites else self.structure,
            to_unit_cell=True,  # wrap to unit cell
        )  # if primitive cell is the same as input structure, use input structure to avoid rotations

        return primitive_structure, sga

    def _supercell_generation_and_checks(
        self,
        primitive_structure: Structure,
        specified_min_image_distance: float,
        pbar: tqdm,
        sga: symmetry.SpacegroupAnalyzer | None = None,
    ):
        # check if input structure is already greater than ``min_image_distance`` Å in each direction:
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
                    **self.supercell_gen_kwargs,
                )

        if not self.generate_supercell or (
            input_min_image_distance >= specified_min_image_distance
            and (primitive_structure * supercell_matrix).num_sites
            >= self.structure.num_sites
            >= self.supercell_gen_kwargs["min_atoms"]
        ):
            if input_min_image_distance < 10:
                # input structure is <10 Å in at least one direction, and generate_supercell=False,
                # so use input structure but warn user:
                warnings.warn(
                    f"\nInput structure is <10 Å in at least one direction (minimum image distance = "
                    f"{input_min_image_distance:.2f} Å, which is usually too "
                    f"small for accurate defect calculations, but generate_supercell = False, so "
                    f"using input structure as defect & bulk supercells. Caution advised!"
                )

            # ``generate_supercell=False`` or input structure has fewer or same number of atoms as
            # doped supercell, so use input structure:
            (
                self.primitive_structure,
                self.supercell_matrix,
            ) = symmetry._get_supercell_matrix_and_possibly_redefine_prim(
                primitive_structure, self.structure, sga=sga, symprec=self.symprec
            )

            self.primitive_structure, self._T = cast(
                "tuple[Structure, np.ndarray]",  # typing; return_T -> (Structure, np.ndarray)
                symmetry.get_clean_structure(self.primitive_structure, return_T=True),
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

    def _get_and_set_bulk_oxi_states(self, pbar: tqdm):
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
                    "\nOxidation states could not be guessed for the input structure. This is required "
                    "for charge state guessing, so defects will still be generated but all charge states "
                    "will be set to -1, 0, +1. You can manually edit these with the "
                    "add/remove_charge_states methods (see tutorials), or you can set the oxidation "
                    "states of the input structure (e.g. using structure.add_oxidation_state_by_element("
                    ")) and re-initialize DefectsGenerator()."
                )

    def _define_bulk_supercell_and_check_min_image_distance(self, specified_min_image_distance: float):
        self.bulk_supercell = symmetry._round_struct_coords(
            (self.primitive_structure * self.supercell_matrix).get_sorted_structure(),
            to_unit_cell=True,
        )
        if not self.generate_supercell:  # re-order bulk supercell to match that of input supercell
            self.bulk_supercell = reorder_s2_like_s1(self.structure, self.bulk_supercell)

        # get and round (to avoid tiny mismatches, due to rounding in search functions,
        # flagging issues) min image distance of supercell:
        self.min_image_distance = np.round(supercells.get_min_image_distance(self.bulk_supercell), 3)

        # check that generated supercell is greater than ``min_image_distance``` Å in each direction:
        if self.min_image_distance < specified_min_image_distance and self.generate_supercell:
            raise ValueError(
                f"Error in supercell generation! Auto-generated supercell is less than chosen minimum "
                f"image distance ({specified_min_image_distance:.2f} Å) in at least one direction ("
                f"minimum image distance = {self.min_image_distance:.2f} Å), which should not happen. "
                f"If you used force_cubic or force_diagonal, you may need to relax these constraints "
                f"to find an appropriate supercell -- otherwise please report this to the developers!"
            )

    def _generate_vacancies(self, pbar: tqdm):
        pbar.set_description("Generating vacancies")
        vac_generator_obj = VacancyGenerator()
        vac_generator = vac_generator_obj.generate(
            self.primitive_structure,
            oxi_state=0,
            rm_species=self.kwargs.get("vacancy_elements", None),
            symprec=self.symprec,
        )  # set oxi_state using doped functions; more robust and efficient
        self.defects["vacancies"] = [
            Vacancy._from_pmg_defect(vac, bulk_oxi_states=self._bulk_oxi_states) for vac in vac_generator
        ]

    def _get_host_and_extrinsic_elements(self) -> tuple[list[str], list[str]]:
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

        host_element_list = list(
            dict.fromkeys(el.symbol for el in self.primitive_structure.composition.elements)
        )
        # if any "extrinsic" elements are actually host elements, remove them and warn user:
        if any(el in host_element_list for el in extrinsic_elements) and not isinstance(
            self.extrinsic,
            dict,  # don't warn if ``extrinsic`` was supplied as a dict
        ):
            warnings.warn(
                f"\nSpecified 'extrinsic' elements "
                f"{[el for el in extrinsic_elements if el in host_element_list]} are present in "
                f"the host structure, so should not need to be specified as 'extrinsic' in "
                f"DefectsGenerator(). These will be ignored. You can input `extrinsic` as a "
                f"dictionary e.g. {{'Cd': ['Zn', 'Mg']}} to override this behaviour and force "
                f"treatment as 'extrinsic species'."
            )

        # sort extrinsic elements by periodic group and atomic number for deterministic ordering:
        extrinsic_elements = sorted(
            [el for el in extrinsic_elements if el not in host_element_list],
            key=_element_sort_func,
        )
        return host_element_list, extrinsic_elements

    def _generate_substitutions(self, pbar: tqdm):
        """
        Generate substitutions, including antisites.
        """
        if self.kwargs.get("substitution_elements", False) == []:  # skip substitutions
            pbar.update(10)  # 30% of progress bar
            return

        pbar.set_description("Generating substitutions")
        antisite_generator_obj = AntiSiteGenerator()
        as_generator = antisite_generator_obj.generate(
            self.primitive_structure, oxi_state=0, symprec=self.symprec
        )
        self.defects["substitutions"] = [
            Substitution._from_pmg_defect(anti, bulk_oxi_states=self._bulk_oxi_states)
            for anti in as_generator
        ]
        pbar.update(5)  # 25% of progress bar

        # Substitutions:
        substitution_generator_obj = SubstitutionGenerator()
        if isinstance(self.extrinsic, str | list):  # substitute all host elements:
            substitutions = {
                el.symbol: self.extrinsic_elements for el in self.primitive_structure.composition.elements
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
                self.primitive_structure,
                substitution=substitutions,
                oxi_state=0,
                symprec=self.symprec,
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
                if sub.name.split("_", 1)[0] in sub_elts  # exact match, not prefix (e.g. T vs Te)
            ]

        if not self.defects["substitutions"]:  # no substitutions, single-element system, no extrinsic
            del self.defects["substitutions"]  # remove empty list
        pbar.update(5)  # 30% of progress bar

    def _generate_interstitials(self, pbar: tqdm):
        if self.interstitial_gen_kwargs is False or self.kwargs.get("interstitial_elements", True) == []:
            return  # skip interstitials

        self.interstitial_gen_kwargs = interstitial_gen_kwargs = {
            **(self.interstitial_gen_kwargs if isinstance(self.interstitial_gen_kwargs, dict) else {}),
            "symprec": self.symprec,
        }

        pbar.set_description("Generating interstitials")
        if self.interstitial_coords:
            # map interstitial coords to primitive structure, and get multiplicities
            for interstitial_frac_coords in self.interstitial_coords:
                equiv_frac_coords_in_prim = symmetry.get_equiv_frac_coords_in_primitive(
                    frac_coords=interstitial_frac_coords,
                    primitive=self.primitive_structure,
                    supercell=self.structure,
                    equiv_coords=True,
                    symprec=self.symprec,
                    dist_tol_factor=self.kwargs.get("dist_tol_factor", 1.0),
                )  # equiv_coords=True
                assert isinstance(equiv_frac_coords_in_prim, list | np.ndarray)
                self.prim_interstitial_coords_mult_and_equiv_coords.append(
                    (
                        equiv_frac_coords_in_prim[0],
                        len(equiv_frac_coords_in_prim),
                        equiv_frac_coords_in_prim,
                    )
                )

            sorted_sites_mul_and_equiv_fpos = self.prim_interstitial_coords_mult_and_equiv_coords

        else:
            # Generate interstitial sites using Voronoi tessellation
            sorted_sites_mul_and_equiv_fpos = get_interstitial_sites(
                host_structure=self.primitive_structure,
                **interstitial_gen_kwargs,
            )

        self.defects["interstitials"] = []
        for el in self.kwargs.get("interstitial_elements", self._element_list):
            el_gen_kwargs, el_sites = interstitial_gen_kwargs, sorted_sites_mul_and_equiv_fpos
            if (
                get_el_sp(el).symbol == "H"  # also D/T (named isotopes, with symbol "H")
                and not self.interstitial_coords
                and (H_gen_kwargs := {**_H_INTERSTITIAL_GEN_KWARGS, **interstitial_gen_kwargs})
                != interstitial_gen_kwargs
            ):
                # Hydrogen present, H-specific generation defaults not overridden, and no manually
                # specified interstitial sites; so re-generate interstitial sites for H with these settings
                el_gen_kwargs = H_gen_kwargs
                el_sites = get_interstitial_sites(self.primitive_structure, **H_gen_kwargs)

            cand_sites, multiplicity, equiv_fpos = zip(*el_sites, strict=False)
            inter_generator = InterstitialGenerator(el_gen_kwargs.get("min_dist", 0.9)).generate(
                self.primitive_structure,
                insertions={el: cand_sites},
                multiplicities={el: multiplicity},
                equivalent_positions={el: equiv_fpos},
                oxi_state=0,
                symprec=self.symprec,
            )
            self.defects["interstitials"].extend(
                Interstitial._from_pmg_defect(inter, bulk_oxi_states=self._bulk_oxi_states)
                for inter in inter_generator
            )

            # check if any manually-specified interstitials were skipped due to min_dist and
            # warn user:
            if self.interstitial_coords and len(self.interstitial_coords) > len(
                self.defects["interstitials"]
            ):
                warnings.warn(
                    f"\nNote that some manually-specified interstitial sites were skipped due to "
                    f"being too close to host lattice sites (minimum distance = `min_dist` = "
                    f"{interstitial_gen_kwargs.get('min_dist', 0.9):.2f} Å). If for some "
                    f"reason you still want to include these sites, you can adjust `min_dist` ("
                    f"default = 0.9 Å), or just use the default Voronoi tessellation algorithm "
                    f"for generating interstitials (by not setting the `interstitial_coords` "
                    f"argument)."
                )

    def _get_conventional_structure_and_wyckoff_label_dict(self, pbar: tqdm) -> dict:
        # get BCS conventional structure, lattice vector swap array, and Wyckoff label dict:
        conv_struct_results = symmetry.get_BCS_conventional_structure(
            self.primitive_structure, pbar=pbar, return_wyckoff_dict=True
        )
        assert len(conv_struct_results) == 3  # with return_wyckoff_dict=True, for typing
        (
            self.conventional_structure,
            self._BilbaoCS_conv_cell_vector_mapping,
            wyckoff_dict,
        ) = conv_struct_results

        return wyckoff_dict

    def _generate_neutral_defect_entries(
        self, pbar: tqdm, processes: int | None, wyckoff_dict: dict | None
    ) -> dict[str, DefectEntry]:
        # process defects into defect entries:
        partial_func = partial(
            _get_neutral_defect_entry,
            supercell_matrix=self.supercell_matrix,
            target_frac_coords=self.target_frac_coords,
            bulk_supercell=self.bulk_supercell,
            conventional_structure=self.conventional_structure,
            _BilbaoCS_conv_cell_vector_mapping=self._BilbaoCS_conv_cell_vector_mapping,
            wyckoff_label_dict=wyckoff_dict or {},
            symprec=self.symprec,
        )

        defect_list: list[Defect] = reduce(operator.iconcat, self.defects.values(), [])
        num_defects = len(defect_list)
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

        # sort defect_entry_list by _frac_coords_sort_func applied to the conv_cell_frac_coords,
        # in order for naming and defect generation output info to be deterministic
        defect_entry_list.sort(key=lambda x: symmetry._frac_coords_sort_func(x.conv_cell_frac_coords))
        return name_defect_entries(  # named defect entry dict
            defect_entry_list,
            element_list=self._element_list,
        )

    def _set_defects_attr_dict(self, defect_entry_dict: dict[str, DefectEntry]):
        # redefine defects dict with DefectEntry.defect objects (as attributes have been updated in
        # _get_neutral_defect_entry):
        self.defects = {
            "vacancies": [
                defect_entry.defect
                for defect_entry in defect_entry_dict.values()
                if _defect_dict_key_from_pmg_type(defect_entry.defect.defect_type) == "vacancies"
            ],
            "substitutions": [
                defect_entry.defect
                for defect_entry in defect_entry_dict.values()
                if _defect_dict_key_from_pmg_type(defect_entry.defect.defect_type) == "substitutions"
            ],
            "interstitials": [
                defect_entry.defect
                for defect_entry in defect_entry_dict.values()
                if _defect_dict_key_from_pmg_type(defect_entry.defect.defect_type) == "interstitials"
            ],
            "others": [  # defect complexes, only present after calling a complex generator method
                defect_entry.defect
                for defect_entry in defect_entry_dict.values()
                if _defect_dict_key_from_pmg_type(defect_entry.defect.defect_type) == "others"
            ],
        }
        # remove empty defect lists: (e.g. single-element systems with no antisite substitutions)
        self.defects = {k: v for k, v in self.defects.items() if v}

    def _guess_and_set_charge_states(self, defect_entry_dict: dict[str, DefectEntry], pbar: tqdm):
        pbar.set_description("Guessing charge states")
        _pbar_increment_per_defect = 1
        if not isinstance(pbar, MagicMock):
            _pbar_increment_per_defect = max(
                0, min((1 / len(defect_entry_dict)) * (pbar.total - pbar.n) * 0.999, pbar.total - pbar.n)
            )  # multiply by 0.999 to avoid rounding errors, overshooting the 100% limit and getting
            # warnings from tqdm

        for defect_name_wout_charge, neutral_defect_entry in defect_entry_dict.items():
            type_name = _defect_type_key_from_pmg_type(neutral_defect_entry.defect.defect_type)
            if self.kwargs.get("neutral_only", False):
                charge_states = [
                    0,
                ]  # only neutral
                neutral_defect_entry.charge_state_guessing_log = cast("list[dict] | None", None)

            elif charge_states := self.kwargs.get(f"{type_name}_charge_states", []):
                neutral_defect_entry.charge_state_guessing_log = cast("list[dict] | None", None)

            elif self._bulk_oxi_states is not False:
                charge_state_guessing_output = guess_defect_charge_states(
                    neutral_defect_entry.defect, return_log=True, **self.charge_state_gen_kwargs
                )
                charge_state_guessing_output = cast(
                    "tuple[list[int], list[dict]]", charge_state_guessing_output
                )  # type checking; guess_defect_charge_states return type depends on ``return_log``
                charge_states, charge_state_guessing_log = charge_state_guessing_output
                neutral_defect_entry.charge_state_guessing_log = cast(
                    "list[dict] | None", charge_state_guessing_log
                )

            else:
                charge_states = [-1, 0, 1]  # no oxi states, so can't guess charge states
                neutral_defect_entry.charge_state_guessing_log = cast("list[dict] | None", None)

            for charge in charge_states:
                defect_entry = deepcopy(neutral_defect_entry) if charge != 0 else neutral_defect_entry
                defect_entry.charge_state = charge
                defect_entry.name = f"{defect_name_wout_charge}_{_signed_charge(charge)}"
                self.defect_entries[defect_entry.name] = defect_entry

            pbar.update(_pbar_increment_per_defect)  # 100% of progress bar

    # TODO check signature
    def generate_complex_chains(
        self,
        chain: Iterable[str] | Iterable[Iterable[str]] | str | None = None,
        chain_length: int = 2,
        max_separation: float = 2.5,
        min_separation: float = 0.0,
        symprec: float | None = None,
        dist_tol_factor: float = 1.0,
        additional_defects: dict[str, Defect] | None = None,
        **kwargs,
    ) -> None:
        r"""
        Generate a set of symmetry-inequivalent defect complexes as ordered
        chains of point defects. The set of possible point defects at each
        position in the chain can be specified, as well as the maximum and
        minimum separation between consecutive point defects in the chain.

        ``chain`` can set both the length of the chains and the point defects allowed
        at each position along them, so e.g. ``[["v_Cd"], ["Cd_i"], ["v_Cd"]]``
        gives split vacancies (vacancy-interstitial-vacancy complexes), or
        ``[["v_Cd", "v_Te"], ["v_Cd", "v_Te"]]`` gives all possible divacancies.
        Alternatively, a single list of names allows chains composed of these defects
        in any order, with ``chain_length`` setting the length.

        For an alternative method of defect complex generation, see
        ``generate_complex_clusters``.

        Args:
            chain (Iterable[str] | Iterable[Iterable[str]] | str | None):
                Names of the point defects (matching those in
                ``self.defect_entries``, with or without their charge states)
                to use as candidate constituents of the generated chains.
                Partial names match all defects of which they are a subset (as
                in ``add_charge_states`` etc.), so e.g. ``"Cd_i"`` matches every
                generated Cd interstitial site. If ``None`` (default), all
                generated point defects are used.

                May also be given as one iterable of names `per position` along
                the chain, which then sets both the chain length (overriding
                ``chain_length``) and the constituents allowed at each position;
                e.g. ``[["v_Cd"], ["Cd_i"], ["v_Cd"]]`` for a split vacancy.
            chain_length (int):
                Number of constituent point defects in the generated chains.
                Ignored if ``defects`` is given per position. (Default: 2)
            max_separation (float):
                Maximum distance (in Å) between consecutive constituent point
                defects in a chain. (Default: 2.5)
            min_separation (float):
                Minimum distance (in Å) between consecutive constituent point
                defects in a chain. (Default: 0.0)
            symprec (float):
                Symmetry precision to use for identifying and pruning
                symmetry-equivalent complexes. If ``None`` (default), uses
                ``self.symprec``.
            dist_tol_factor (float):
                Factor by which ``symprec`` is multiplied to give the distance
                tolerance for matching symmetry-equivalent complexes. (Default:
                1.0)
            additional_defects (dict[str, Defect] | None):
                Additional |Defect| objects to include as candidate
                constituents, as a ``{name: Defect}`` dictionary, for point
                defects not generated by this |DefectsGenerator|. The given
                names must not collide with those of the generated defects.
                (Default: ``None``)
            **kwargs:
                Additional keyword arguments, e.g. for the complex generation
                and symmetry analysis functions in ``doped.complexes``.
        """
        from doped.complexes import get_complex_chains  # here, to avoid a circular import

        pool, names, nested = self._complex_constituent_pool(chain, additional_defects)
        positions = (
            cast("list[list[str]]", names)  # positions given explicitly, so they set the chain
            if nested  # length and the allowed constituents
            else [[name for group in names for name in group]] * chain_length  # any, at each position
        )
        show_pbar = kwargs.pop("pbar", True)
        complexes = get_complex_chains(
            pool,
            positions,
            max_separation=max_separation,
            min_separation=min_separation,
            symprec=self.symprec if symprec is None else symprec,
            dist_tol_factor=dist_tol_factor,
            pbar=show_pbar,
            **kwargs,
        )

        self._add_defect_complexes(complexes, pbar=show_pbar)

    def generate_complex_clusters(
        self,
        defects: Iterable[str] | str | None = None,
        cluster_size: int | tuple[int, int] = 2,
        max_diameter: float = 3.0,
        min_separation: float = 0.0,
        allow_repeats: bool = True,
        symprec: float | None = None,
        dist_tol_factor: float = 1.0,
        additional_defects: dict[str, Defect] | None = None,
        **kwargs,
    ) -> None:
        r"""
        Generate a set of symmetry-inequivalent defect complexes as unordered
        clusters of point defects. The set of possible point defects, the
        maximum and minimum size of the cluster, and the maximum diameter of
        the complex can be specified.

        Constituents are drawn from ``defects`` in any combination, so e.g.
        ``[v_Cd, v_Te]`` with ``size = 2`` gives all divacancies, and with
        ``size = (2, 3)`` all di- and tri-vacancies. With ``allow_repeats = False``,
        complexes with an exact composition can be specified, e.g.
        ``[v_Cd, v_Cd, Cd_i, Cd_Te]`` with ``size = 4`` will only give clusters
        with this exact composition of point defects.

        For an alternative method of defect complex generation, see
        ``generate_complex_chains``.

        Args:
            defects (Iterable[str] | str | None):
                Names of the point defects (matching those in
                ``self.defect_entries``, with or without their charge states)
                to use as candidate constituents of the generated clusters.
                Partial names match all defects of which they are a subset (as
                in ``add_charge_states`` etc.), so e.g. ``"Cd_i"`` matches every
                generated Cd interstitial site. If ``None`` (default), all
                generated point defects are used.

                A name may be given more than once, setting the number of times
                that point defect may appear in a cluster when ``allow_repeats``
                is ``False``; e.g. ``["v_Cd", "v_Cd", "Cd_i"]`` with
                ``cluster_size=3, allow_repeats=False`` gives only
                ``v_Cd-v_Cd-Cd_i`` type complexes. An entry may itself be a
                group of names, which then count as one constituent between
                them; e.g. ``["v_Cd", ["Cd_i", "Te_i"]]`` to permit a single
                Cd interstitial OR Te interstitial in the cluster. Note groups
                are assumed disjoint or identical, partially overlapping
                groups are not supported.
            cluster_size (int | tuple[int, int]):
                Number of constituent point defects in the generated
                clusters, or a ``(min, max)`` range thereof. (Default: 2)
            max_diameter (float):
                Maximum span (diameter) of the generated clusters in Å; i.e.
                the maximum allowed distance between any two constituent
                point defects. (Default: 3.0)
            min_separation (float):
                Minimum distance (in Å) between any two constituent point
                defects in a cluster. (Default: 0.0)
            allow_repeats (bool):
                Whether the same point defect may appear more than once in a
                cluster. If ``False``, the multiplicity of each constituent is
                set by how many times it is named in ``defects``. (Default: True)
            symprec (float):
                Symmetry precision to use for identifying and pruning
                symmetry-equivalent complexes. If ``None`` (default), uses
                ``self.symprec``.
            dist_tol_factor (float):
                Factor by which ``symprec`` is multiplied to give the distance
                tolerance for matching symmetry-equivalent complexes. (Default:
                1.0)
            additional_defects (dict[str, Defect] | None):
                Additional |Defect| objects to include as candidate
                constituents, as a ``{name: Defect}`` dictionary, for point
                defects not generated by this |DefectsGenerator|. The given
                names must not collide with those of the generated defects.
                (Default: ``None``)
            **kwargs:
                Additional keyword arguments, e.g. for the complex generation
                and symmetry analysis functions in ``doped.complexes``.
        """
        from doped.complexes import get_complex_clusters  # avoid a circular import

        pool, names, _nested = self._complex_constituent_pool(defects, additional_defects)
        show_pbar = kwargs.pop("pbar", True)
        complexes = get_complex_clusters(
            pool,
            names=names,  # one group per given name - repeats set constituent multiplicity
            size=cluster_size,
            max_diameter=max_diameter,
            min_separation=min_separation,
            allow_repeats=allow_repeats,
            symprec=self.symprec if symprec is None else symprec,
            dist_tol_factor=dist_tol_factor,
            pbar=show_pbar,
            **kwargs,
        )

        self._add_defect_complexes(complexes, pbar=show_pbar)

    def _defects_by_name(self, complexes: bool = False) -> dict[str, Defect]:
        """
        The generated point defects (or defect complexes, if ``complexes``),
        keyed by name without charge state, taking the neutral |DefectEntry|'s
        |Defect| if present.
        """
        available: dict[str, Defect] = {}
        for name, defect_entry in self.defect_entries.items():
            if isinstance(defect_entry.defect, DefectComplex) is not complexes:
                continue
            name_wout_charge = _name_wout_charge_state(name)
            if name_wout_charge not in available or defect_entry.charge_state == 0:
                available[name_wout_charge] = defect_entry.defect

        return available

    def _complex_constituent_pool(
        self,
        defects: Iterable[str] | Iterable[Iterable[str]] | str | None = None,
        additional_defects: dict[str, Defect] | None = None,
    ) -> tuple[dict[str, Defect], list[list[str]], bool]:
        """
        Construct a labelled dictionary of point defects for a pool of
        constituent point defects for complexes. May take names with or without
        charge state, and the names are matched to the self.defect_entries
        dict. Also take partial matches e.g. Cd_i will match all Cd
        interstitials if multiple.

        Returns the ``{name: Defect}`` pool, the given names resolved against
        it as one group per given name (or per position, for a nested input),
        and whether the input was nested (ie to distinguish the two uses for
        the chains/clusters applications).
        """
        available = self._defects_by_name()  # point defects only, no complexes of complexes

        def _resolve(names: Iterable[str] | str) -> list[str]:
            """
            Expand the given names against ``available``, preserving order and
            repeats.
            """
            return [
                match
                for name in ([names] if isinstance(names, str) else names)
                for match in _match_defect_names(name, available, "Point defect")
            ]

        nested = (
            defects is not None
            and not isinstance(defects, str)
            and not all(isinstance(entry, str) for entry in defects)
        )
        if defects is None:
            names = [[name] for name in available]  # each generated point defect, once
        elif nested:
            names = [_resolve(position) for position in defects]  # type: ignore[arg-type]
        else:
            names = [
                _resolve(name)
                for name in ([defects] if isinstance(defects, str) else cast("Iterable[str]", defects))
            ]

        pool = {  # any position may draw on any of the resolved names
            name: available[name] for group in names for name in group
        }

        taken = {name.rsplit("_", 1)[0] for name in self.defect_entries}
        for name, defect in (additional_defects or {}).items():
            if (generated := available.get(name)) is not None and generated == defect:
                # user gave a defect already in self.defects - use generated to be safe
                pool[name] = generated
            elif name in taken:
                raise ValueError(
                    f"The name {name!r} given in ``additional_defects`` is already used by a "
                    f"different generated defect! Please use a different name to avoid ambiguity."
                )
            else:
                pool[name] = defect
            # also available at each position, unless the caller named the positions explicitly
            if nested:
                continue
            names.append([name])

        return pool, names, nested

    def _add_defect_complexes(
        self, complexes: list[tuple[DefectComplex, list[str]]], pbar: tqdm | bool = True
    ) -> None:
        """
        From DefectComplexes, generate DefectEntries and add to self.defects
        and self.defect_entries. Coordinates pbar, deduplication, warnings,
        name disambiguation, charge state guessing, defect entry generation,
        sorting.

        Args:
            complexes (list[tuple[DefectComplex, list[str]]]):
                The generated complexes and the ``doped`` names of their
                constituent point defects, as returned by the generation
                functions in ``doped.complexes``.
            pbar (tqdm | bool):
                An existing progress bar to report into, or ``True``/``False``
                to create a new bar which is shown/hidden. One step is taken
                per complex. (Default: True)
        """
        # avoid a circular import:
        from doped.complexes import _complex_generation_pbar, get_defect_complex_name

        if not complexes:
            warnings.warn(
                "No defect complexes were generated with the given parameters! Try increasing "
                "``max_separation``/``max_diameter``, or check the given ``defects``.",
                UserWarning,
                stacklevel=2,
            )
            return

        # match against complexes already generated
        generated: dict[str, DefectComplex] = {  # {name (wout charge): complex} already generated
            name.rsplit("_", 1)[0]: cast("DefectComplex", defect_entry.defect)
            for name, defect_entry in self.defect_entries.items()
            if isinstance(defect_entry.defect, DefectComplex)
        }
        new_complexes = []
        duplicate_names = []  # names of the already generated complexes regenerated here
        for defect_complex, constituent_names in complexes:
            name = next((n for n, other in generated.items() if other == defect_complex), None)
            if name is None:
                new_complexes.append((defect_complex, constituent_names))
            else:
                duplicate_names.append(name)

        # sort new complexes before labelling to give deterministic _a, _b etc suffixes up to
        # within a single generation method run - but cannot guarantee deterministic if collided
        # names are generated in separate generation calls
        new_complexes.sort(
            key=lambda complex_and_names: _defect_sort_key(
                complex_and_names[0], self._element_list, include_frac_coords=True
            )
        )

        # label collided names
        names, renames = _disambiguate_names(
            [
                get_defect_complex_name(defect_complex, names=constituent_names)
                for defect_complex, constituent_names in new_complexes
            ],
            existing_names=[name.rsplit("_", 1)[0] for name in self.defect_entries],
        )
        for old_name, new_name in renames.items():  # rename all charge states of any demoted complex
            for entry_name in [n for n in self.defect_entries if n.rsplit("_", 1)[0] == old_name]:
                defect_entry = self.defect_entries.pop(entry_name)
                defect_entry.name = f"{new_name}_{entry_name.rsplit('_', 1)[1]}"
                self.defect_entries[defect_entry.name] = defect_entry
        duplicate_names = [renames.get(name, name) for name in duplicate_names]  # if demoted just above

        partial_func = partial(
            _get_neutral_defect_entry,
            supercell_matrix=self.supercell_matrix,
            target_frac_coords=self.target_frac_coords,
            bulk_supercell=self.bulk_supercell,
            conventional_structure=self.conventional_structure,
            _BilbaoCS_conv_cell_vector_mapping=self._BilbaoCS_conv_cell_vector_mapping,
            wyckoff_label_dict={},
            symprec=self.symprec,
        )
        with (
            warnings.catch_warnings(),
            # one step per complex for each of the two stages below:
            # DefectEntry copying and charge state guessing take roughly comparable time
            _complex_generation_pbar(2 * len(new_complexes), pbar) as entry_pbar,
        ):
            # aggregate small supercell warnings
            warnings.filterwarnings("ignore", message="The complex minimum image distance")
            entry_pbar.set_description(
                f"Generating DefectEntry objects for {len(new_complexes)} complex(es)"
            )
            neutral_defect_entries = {}
            for name, (defect_complex, _constituent_names) in zip(names, new_complexes, strict=True):
                neutral_defect_entries[name] = partial_func(defect_complex)
                entry_pbar.update(1)

            if neutral_defect_entries:  # all complexes may have already been generated
                self._guess_and_set_charge_states(neutral_defect_entries, entry_pbar)
            if (remaining := entry_pbar.total - entry_pbar.n) > 0:
                entry_pbar.update(remaining)

        if neutral_defect_entries:
            self._warn_if_small_supercell(neutral_defect_entries)

        added = [defect_entry.defect for defect_entry in neutral_defect_entries.values()]
        self.defects.setdefault("others", []).extend(added)
        # re sort both dicts
        self.defects = _sort_defects(self.defects, element_list=self._element_list)
        self.defect_entries = sort_defect_entries(self.defect_entries, element_list=self._element_list)

        # print the complexes info table, as for point defects on |DefectsGenerator| initialisation:
        # showing all complexes matching the given parameters, including any skipped as already
        # generated, so the printed set shows all complexes matching the query
        shown = set(names) | set(duplicate_names)
        self.defect_generator_info(
            defect_classes=["others"],
            defect_entries={
                name: defect_entry
                for name, defect_entry in self.defect_entries.items()
                if name.rsplit("_", 1)[0] in shown
            },
        )

    def _warn_if_small_supercell(
        self, defect_entries: dict[str, DefectEntry], min_span_factor: float = 1.5
    ):
        """
        Small supercell warnings for complexes.
        """
        ill_defined = {}
        for name, defect_entry in defect_entries.items():
            cart_coords = np.array([site.coords for site in defect_entry.defect_supercell_sites or []])
            min_image_distance = supercells._get_complex_min_image_distance_from_matrix(
                self.bulk_supercell.lattice.matrix, cart_coords
            )
            span = float(np.linalg.norm(cart_coords[:, None] - cart_coords[None, :], axis=-1).max())
            if min_image_distance < span * min_span_factor:
                ill_defined[name] = (min_image_distance, span)

        if not ill_defined:
            return

        # split by severity: at or below the span, each constituent is at least as close to those of
        # a neighbouring image as to the rest of its own complex, and complex is likely not well defined
        # above it, just results may be physically inaccurate
        severe = {name: v for name, v in ill_defined.items() if round(v[0], 2) <= round(v[1], 2)}
        marginal = {name: v for name, v in ill_defined.items() if round(v[0], 2) > round(v[1], 2)}

        def _listing(entries: dict[str, tuple[float, float]], max_shown: int = 10) -> str:
            shown = list(entries.items())[:max_shown]
            return "\n".join(
                f"  {name}: complex minimum image distance {dist:.2f} Å vs span {span:.2f} Å"
                for name, (dist, span) in shown
            ) + (f"\n  ... and {len(entries) - len(shown)} more" if len(entries) > len(shown) else "")

        message = (
            f"\n!!! {len(ill_defined)} of the {len(defect_entries)} generated defect complexes are not "
            f"well-defined in this supercell !!!\nThe complex minimum image distance (the minimum "
            f"distance between any constituent and a constituent of a periodic image) is compared "
            f"against the complex span (the largest separation between its own constituents):\n"
        )
        if severe:
            message += (
                f"\n{len(severe)} with a minimum image distance less than or equal to their span, so each "
                f"constituent is as close (or closer) to those of neighbouring complex images as it is "
                f"to the rest of its own complex; these complexes are not physically meaningful in "
                f"this supercell and their formation energies may be unreliable:\n" + _listing(severe)
            )
        if marginal:
            message += (
                f"\n{len(marginal)} with a minimum image distance larger than, but close to "
                f"({min_span_factor}x), their span, so complex-image interactions will be "
                f"significant:\n" + _listing(marginal)
            )
        warnings.warn(
            f"{message}\nA larger and/or complex-optimised supercell is strongly recommended; see "
            f"``optimise_supercell``.",
            UserWarning,
            stacklevel=3,
        )

    def defect_generator_info(
        self,
        defect_classes: Iterable[str] | None = None,
        defect_entries: dict[str, DefectEntry] | None = None,
    ):
        """
        Prints information about the defects that have been generated.

        Args:
            defect_classes (Iterable[str]):
                The classes of defect to print, as keys of ``self.defects``
                (i.e. ``"vacancies"``, ``"substitutions"``, ``"interstitials"``
                and/or ``"others"``; the last being defect complexes). If
                ``None`` (default), all generated classes are printed.
            defect_entries (dict[str, DefectEntry]):
                The |DefectEntry| objects to tabulate, allowing a subset of
                the generated defects to be printed. If ``None`` (default),
                ``self.defect_entries`` is used.
        """
        return print(
            self._defect_generator_info(defect_classes=defect_classes, defect_entries=defect_entries)
        )

    def _defect_generator_info(
        self,
        defect_classes: Iterable[str] | None = None,
        defect_entries: dict[str, DefectEntry] | None = None,
    ):
        """
        Returns a string with information about the defects that have been
        generated by the |DefectsGenerator|.

        Args:
            defect_classes (Iterable[str]):
                The classes of defect to include, as keys of ``self.defects``
                (i.e. ``"vacancies"``, ``"substitutions"``, ``"interstitials"``
                and/or ``"others"``; the last being defect complexes). If
                ``None`` (default), all generated classes are included.
            defect_entries (dict[str, DefectEntry]):
                The |DefectEntry| objects to tabulate, allowing a subset of
                the generated defects to be printed. If ``None`` (default),
                ``self.defect_entries`` is used.
        """
        from doped.complexes import get_defect_complex_name  # avoid a circular import

        if defect_entries is None:
            defect_entries = self.defect_entries
        info_string = ""
        shown_wyckoff = shown_complexes = False
        for defect_class in self.defects if defect_classes is None else defect_classes:
            defect_list = self.defects.get(defect_class, [])
            if len(defect_list) > 0:
                table = []
                complexes = defect_class == "others"  # complexes; named differently, and no Wyckoff label
                header = [
                    "Complexes" if complexes else defect_class.capitalize(),
                    "Guessed Charges",
                    # the full name already gives the point group, so is shown in place of "Wyckoff":
                    *(["Name", "Conv. Cell Coords"] if complexes else ["Conv. Cell Coords", "Wyckoff"]),
                ]
                defect_type = defect_list[0].defect_type
                matching_defect_types = {
                    defect_entry_name: defect_entry
                    for defect_entry_name, defect_entry in defect_entries.items()
                    if defect_entry.defect.defect_type == defect_type
                }
                if not matching_defect_types:  # none of this class in the given ``defect_entries``
                    continue
                shown_wyckoff = shown_wyckoff or not complexes
                shown_complexes = shown_complexes or complexes
                matching_type_names_wout_charge = []
                for defect_entry_name in matching_defect_types:
                    defect_name_wout_charge = defect_entry_name.rsplit("_", 1)[0]
                    if defect_name_wout_charge not in matching_type_names_wout_charge:
                        matching_type_names_wout_charge.append(defect_name_wout_charge)

                for defect_name_wout_charge in matching_type_names_wout_charge:
                    charges = [
                        defect_entry_name.rsplit("_", 1)[1]
                        for defect_entry_name in matching_defect_types
                        if defect_entry_name.rsplit("_", 1)[0] == defect_name_wout_charge
                    ]
                    # convert list of strings to one string with comma-separated charges
                    charges_string = "[" + ",".join(charges) + "]"
                    defect_entry = next(
                        entry
                        for name, entry in matching_defect_types.items()
                        if name.rsplit("_", 1)[0] == defect_name_wout_charge
                    )
                    frac_coords_string = (
                        "N/A"
                        if defect_entry.conv_cell_frac_coords is None
                        else ",".join(f"{x:.3f}" for x in defect_entry.conv_cell_frac_coords)
                    )
                    row = (
                        [  # composition at a glance first, then the full name:
                            get_defect_complex_name(
                                cast("DefectComplex", defect_entry.defect),
                                element_list=self._element_list,
                                include_point_group=False,
                                include_separations=False,
                            ),
                            charges_string,
                            defect_name_wout_charge,
                            f"[{frac_coords_string}]",
                        ]
                        if complexes
                        else [
                            defect_name_wout_charge,
                            charges_string,
                            f"[{frac_coords_string}]",
                            defect_entry.wyckoff,
                        ]
                    )
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
        if shown_wyckoff:
            info_string += (
                "The number in the Wyckoff label is the site multiplicity/degeneracy of that defect in "
                f"the conventional ('conv.') unit cell, which comprises {fu} formula unit(s) of "
                f"{formula}.\n"
            )
        if shown_complexes:
            info_string += (
                "Complex names give the constituent point defects in order along the shortest chain "
                "through the complex, with their separations (in Å) interleaved and the point group of "
                "the complex appended, while the conv. cell coordinates are those of its centroid.\n"
            )

        return info_string

    def _process_name_and_charge_states_and_get_matching_entries(
        self,
        defect_entry_name: str,
        charge_states: list | int,
        match_charge_states: bool = True,
    ) -> tuple[list, list]:
        defect_entry_name = _name_wout_charge_state(defect_entry_name)

        charge_states = [round(charge) for charge in np.atleast_1d(charge_states).tolist()]  # int list

        matching_entry_names_wout_charge = [
            name for name in self.defect_entries if _check_if_name_subset(name, defect_entry_name)
        ]
        if not match_charge_states:
            return charge_states, matching_entry_names_wout_charge

        return charge_states, [
            name
            for charge in charge_states
            for name in matching_entry_names_wout_charge
            if name.endswith(f"_{_signed_charge(charge)}")
        ]

    def add_charge_states(self, defect_entry_name: str, charge_states: list | int):
        r"""
        Add additional |DefectEntry|\s with the specified charge states to
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

        for defect_entry_name_wout_charge in unique_matching_entry_names_wout_charge:
            previous_defect_entry = next(
                entry
                for name, entry in self.defect_entries.items()
                if _check_if_name_subset(name, defect_entry_name_wout_charge)
            )
            for charge in charge_states:
                defect_entry = deepcopy(previous_defect_entry)
                defect_entry.charge_state = charge
                defect_entry.name = f"{defect_entry.name.rsplit('_', 1)[0]}_{_signed_charge(charge)}"
                self.defect_entries[defect_entry.name] = defect_entry

        # sort defects and defect entries for deterministic behaviour:
        self.defects = _sort_defects(self.defects, element_list=self._element_list)
        self.defect_entries = sort_defect_entries(self.defect_entries, element_list=self._element_list)

    def remove_charge_states(self, defect_entry_name: str, charge_states: list | int):
        r"""
        Remove |DefectEntry|\s with the specified charge states from
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
        self.defect_entries = sort_defect_entries(self.defect_entries, element_list=self._element_list)

    def as_dict(self):
        """
        JSON-serializable dict representation of |DefectsGenerator|.
        """
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            **self.__dict__,
        }

    @classmethod
    def from_dict(cls, d):
        """
        Reconstructs |DefectsGenerator| object from a dict representation
        created using ``DefectsGenerator.as_dict()``.

        Args:
            d (dict): dict representation of |DefectsGenerator|.

        Returns:
            |DefectsGenerator| object
        """
        defects_generator = cls.__new__(cls)  # create new object without invoking __init__
        for key, value in d.items():  # set instance variables directly, decoding nested MSON dicts
            if key not in {"@module", "@class", "@version"}:
                setattr(defects_generator, key, MontyDecoder().process_decoded(value))

        # ``padding`` (removed in v4.0; see ``__init__``) may be saved in generators from older versions:
        getattr(defects_generator, "charge_state_gen_kwargs", {}).pop("padding", None)  # TODO: remove v4.1

        return defects_generator

    def to_json(self, filename: PathLike | None = None):
        """
        Save the |DefectsGenerator| object as a json file, which can be
        reloaded with the ``DefectsGenerator.from_json()`` class method.

        Note that file extensions with ".gz" will be automatically compressed
        (recommended to save space)!

        Args:
            filename (PathLike):
                Filename to save ``json`` file as. If ``None``, the filename
                will be set as ``{Chemical Formula}_defects_generator.json.gz``
                where {Chemical Formula} is the chemical formula of the host
                material.
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
        Load a |DefectsGenerator| object from a json(.gz) file.

        Note that ``.json.gz`` files can be loaded directly.

        Args:
            filename (PathLike):
                Filename of json file to load |DefectsGenerator| object from.

        Returns:
            |DefectsGenerator| object
        """
        return loadfn(filename)

    def __getattr__(self, attr):
        """
        Redirects an unknown attribute/method call to the ``defect_entries``
        dictionary attribute, if the attribute doesn't exist in
        |DefectsGenerator|.
        """
        if attr == "defect_entries":  # not yet set (e.g. partially-initialised object), avoid recursion:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
        return getattr(self.defect_entries, attr)

    def __getitem__(self, key):
        """
        Makes |DefectsGenerator| object subscriptable, so that it can be
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
        # check the input, must be a |DefectEntry| object, with same supercell and primitive structure
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
        self.defect_entries = sort_defect_entries(self.defect_entries, element_list=self._element_list)

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
        Returns a string representation of the |DefectsGenerator| object.
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
        Returns a string representation of the |DefectsGenerator| object, and
        prints the |DefectsGenerator| info.

        Note that Wyckoff letters can depend on the ordering of elements in the
        conventional standard structure, for which doped uses the ``spglib``
        convention.
        """
        return (
            self.__str__()
            + "\n---------------------------------------------------------\n"
            + self._defect_generator_info()
        )


def _get_element_list(defect: Defect | DefectEntry | dict | list) -> list[str]:
    """
    Given an input |Defect| or |DefectEntry|, or dictionary/list of these,
    return a (non-duplicated) list of elements present in the defect
    structures, following the order of appearance in the composition.

    Extrinsic elements are sorted according to ``_element_sort_func``, which
    sorts based on periodic group and atomic number.
    """

    def _get_single_defect_element_list(single_defect):
        element_list = list(
            dict.fromkeys(el.symbol for el in single_defect.structure.composition.elements)
        )
        defect_element = single_defect.site.specie.symbol  # possibly extrinsic
        # for substitutions; defect.site is substitution element, defect.defect_site is the original site..
        if defect_element not in element_list:
            element_list.append(defect_element)

        return element_list

    if isinstance(defect, Defect | DefectEntry | core.Defect | thermo.DefectEntry):
        single_defect = defect if isinstance(defect, Defect | core.Defect) else defect.defect
        if isinstance(single_defect, core.DefectComplex):  # constituent species, not dummy centroid
            return _get_element_list(single_defect.defects)
        return _get_single_defect_element_list(single_defect)

    # else is dict/list
    defect_list = defect if isinstance(defect, list) else list(defect.values())
    defect_list = [  # flatten nested lists
        entry_or_defect
        for entry_or_defect_or_list in defect_list
        for entry_or_defect in (
            entry_or_defect_or_list
            if isinstance(entry_or_defect_or_list, list)
            else [entry_or_defect_or_list]
        )
    ]
    defect_list = [
        (
            entry_or_defect.defect
            if isinstance(entry_or_defect, DefectEntry | thermo.DefectEntry)
            else entry_or_defect
        )
        for entry_or_defect in defect_list
    ]
    defect_list = [  # constituent point defects of any complexes, not their dummy centroid sites:
        constituent
        for single_defect in defect_list
        for constituent in (
            single_defect.defects if isinstance(single_defect, core.DefectComplex) else [single_defect]
        )
    ]
    host_element_list = list(
        dict.fromkeys(el.symbol for el in next(iter(defect_list)).structure.composition.elements)
    )
    extrinsic_element_list: list[str] = []
    for single_defect in defect_list:
        extrinsic_element_list.append(single_defect.site.specie.symbol)  # possibly extrinsic
        # for substitutions; defect.site is substitution element, defect.defect_site is the original site..
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
        defect_name,
        include_site_info=False,
        include_charge=defect_name.rsplit("_", maxsplit=1)[-1].isdigit(),
    )
    if formatted_defect_name:
        if not formatted_defect_name.startswith("$"):  # substitution or interstitial
            first_element = formatted_defect_name.split("$")[0]

            if "$_i" in formatted_defect_name:  # interstitial
                return (first_element, first_element)

            return (first_element, formatted_defect_name.split("$_{")[1].split("}")[0])  # substitution

        vacancy_elt = formatted_defect_name.split("$_{")[1].split("}")[0]  # else vacancy
        return (vacancy_elt, vacancy_elt)

    return (
        defect_name.split("_", maxsplit=1)[0],
        defect_name.split("_")[1],
    )  # return name split if formatting fails


def _element_sort_func(element_str: str) -> tuple[int, int, float]:
    """
    Return a tuple of the group (+16 if it's a transition metal, to move them
    after main group elements), atomic number and atomic mass (to order named
    isotopes, i.e. H, D, T) of the element, for sorting purposes.

    Args:
        element_str (str): Element symbol.

    Returns:
        tuple: Tuple of the group, atomic number and atomic mass of the
            element.
    """
    elt = Element(element_str)
    group = elt.group + 16 if 3 <= elt.group <= 18 else elt.group
    return (group, elt.Z, elt.atomic_mass)


def sort_defect_entries(defect_entries: dict | list, element_list: list | None = None):
    """
    Sort defect entries for deterministic behaviour; for output and when
    reloading |DefectsGenerator| objects, with |DefectThermodynamics| entries
    (particularly for deterministic plotting behaviour), and with
    |DefectsParser| objects.

    Sorts defect entries by defect type (vacancies, substitutions,
    interstitials, complexes; where complexes with the net composition change
    of a single point defect, e.g. split vacancies, are sorted with that point
    defect type), then by order of appearance of elements in the host
    composition, then by periodic group (main groups 1, 2, 13-18 first, then
    TMs), then by atomic number, then (for defect entries of the same type)
    sort by name and charge state (from positive to negative).

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

    def _type_and_name_elements(name: str, defect: Defect) -> tuple[int, tuple]:
        """
        Complex-aware equivalent of the defect type and first and second
        element sort keys.

        Defect complexes with the net composition change of a single point
        defect (such as split vacancies; see ``DefectComplex.net_name``) are
        sorted with that point defect type, while for other complexes this
        takes only symmetry-invariant, always available (generation and
        parsing) keys - number of defects, constituent point defects
        composition, and interdefect distances.
        """
        if isinstance(defect, core.DefectComplex):
            if defect.net_name == defect.name:  # not a single net point defect
                return defect.defect_type.value, _defect_sort_key(defect, element_list)[1:-1]
            name = defect.net_name  # sort with the corresponding point defect type
            defect_type = (
                core.DefectType.Vacancy
                if name.startswith("v_")
                else core.DefectType.Interstitial
                if name.endswith("_i")
                else core.DefectType.Substitution
            )
        else:
            defect_type = defect.defect_type
        return defect_type.value, tuple(
            _list_index_or_val(element_list, element) for element in _first_and_second_element(name)
        )

    try:
        sorted_defect_entries_dict = dict(
            sorted(
                defect_entries_dict.items(),
                key=lambda s: (
                    *_type_and_name_elements(s[0], s[1].defect),
                    s[0].rsplit("_", 1)[0],  # name without charge
                    -s[1].charge_state,  # charge state
                ),
            )
        )
    except (ValueError, TypeError) as sort_err:
        # possibly defect entries with names not in doped format, try sorting without using name:
        from doped.complexes import get_defect_complex_name  # avoid circular import

        try:

            def _defect_entry_sort_func(defect_entry):
                unrelaxed_defect_name_w_charge = defect_entry.calculation_metadata.get(
                    "full_unrelaxed_defect_name"
                )
                if unrelaxed_defect_name_w_charge is not None:
                    name_from_defect = unrelaxed_defect_name_w_charge.rsplit("_", 1)[0]  # without charge
                elif isinstance(defect_entry.defect, core.DefectComplex):
                    name_from_defect = get_defect_complex_name(
                        defect_entry.defect, element_list=element_list
                    )
                else:
                    name_from_defect = get_defect_name_from_defect(
                        defect_entry.defect,
                        element_list=element_list,
                    )
                return (
                    *_type_and_name_elements(defect_entry.defect.name, defect_entry.defect),
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
        except (ValueError, TypeError) as sort_err_2:
            raise sort_err_2 from sort_err

    if isinstance(defect_entries, list):  # then return as a list
        return list(sorted_defect_entries_dict.values())

    return sorted_defect_entries_dict  # else dict


def _defect_sort_key(defect: Defect, element_list: list[str], include_frac_coords: bool = False) -> tuple:
    """
    Deterministic sort key for |Defect| objects; see ``_sort_defects``.

    Defect `complexes` sort after all point defects (by ``defect_type``), then
    by number of constituents, then by their constituent point defects, then by
    interdefect distances, then by point group, then by conventional cell
    fractional coordinates of the constituents.

    Args:
        defect (Defect):
            |Defect| (or |DefectComplex|) object to generate the sort key for.
        element_list (list[str]):
            Ordered list of elements, used to sort by order of appearance in
            the host composition.
        include_frac_coords (bool):
            If ``True``, the fractional coordinates of the defect site (and of
            each constituent, for a complex) are appended to the key, for stronger
            discrimination - for a complex, the invariant terms
            above do not always distinguish inequivalent complexes, while these
            coordinates always do.

            Note that these coordinates are not symmetry invariant, so if the
            sort reaches this term then symmetry-equivalent complexes (which
            differ only in their choice of configuration) may be sorted further
            apart than inequivalent ones. Only appropriate therefore where a
            total order matters more than symmetry invariance, and where all
            objects being sorted come from the same pathway (and duplicate
            equivalent defects are not present?).
            (Default: False)

    Returns:
        tuple: Sort key for the defect.
    """
    if isinstance(defect, core.DefectComplex):
        cart_coords = defect.structure.lattice.get_cartesian_coords(
            [defect.site.frac_coords for defect in defect.defects]
        )
        seps = np.linalg.norm(cart_coords[:, None, :] - cart_coords[None, :, :], axis=-1)
        complex_seps = tuple(sorted(np.round(seps[np.triu_indices(len(cart_coords), k=1)], 2).tolist()))
        point_sort_keys = sorted(
            _defect_sort_key(constituent, element_list, include_frac_coords=include_frac_coords)
            for constituent in defect.defects
        )
        key_terms = list(zip(*point_sort_keys, strict=True))
        return (
            defect.defect_type.value,
            len(defect.defects),
            tuple(key_terms[:4]),  # symmetry invariant: type, elements and name of each constituent
            complex_seps,
            defect.point_group or "",  # ``None`` if not yet determined
            # conv_cell_frac_coords is actually symmetry invariant but we sort on this last
            # to maintain maximum parity between generation (attribute is present) and parsing
            # (attribute is not present)
            (
                symmetry._frac_coords_sort_func(getattr(defect, "conv_cell_frac_coords", None)),
                # centroid conv cell frac coords
                *key_terms[4:],
                # constituent conv cell frac coords, then optionally raw constituent frac coords
            ),
        )

    return (
        defect.defect_type.value,
        _list_index_or_val(element_list, _first_and_second_element(defect.name)[0]),
        _list_index_or_val(element_list, _first_and_second_element(defect.name)[1]),
        defect.name,  # bare name without charge
        symmetry._frac_coords_sort_func(getattr(defect, "conv_cell_frac_coords", None)),
        *(  # not symmetry invariant, so only ever an optional tiebreak; see docstring
            [symmetry._frac_coords_sort_func(defect.site.frac_coords)] if include_frac_coords else []
        ),
    )


def _sort_defects(
    defects_dict: dict,
    element_list: list[str] | None = None,
    include_frac_coords: bool = False,
):
    """
    Sort defect objects for deterministic behaviour (for output and when
    reloading |DefectsGenerator| objects.

    Sorts defects by defect type (vacancies, substitutions, interstitials),
    then by order of appearance of elements in the composition, then by
    periodic group (main groups 1, 2, 13-18 first, then TMs), then by atomic
    number, then according to ``symmetry._frac_coords_sort_func``.

    Defect complexes sort after all point defects, then by number of constituents,
    then by their constituent point defects (except coords), then by interdefect
    separations, then by constituent conventional cell fractional coords.

    If ``include_frac_coords`` is used, raw fractional coordinates are included
    to fully discriminate between defects - however note these are not symmetry
    invariant. Conventional cell fractional coords (symmetry invariant) are
    not available parsing side, and do not distinguish fully between different
    inequivalent complexes, so this is required for a fully deterministic ordering
    of complex constituents, for example.
    """
    if element_list is None:
        element_list = _get_element_list(defects_dict)

    return {
        defect_type: sorted(
            defect_list,
            key=lambda d: _defect_sort_key(d, element_list, include_frac_coords=include_frac_coords),
        )
        for defect_type, defect_list in defects_dict.items()
    }


def get_stol_equiv_dist(stol: float, structure: Structure) -> float:
    """
    Get the equivalent Cartesian distance of a given ``stol`` value for a given
    |Structure|.

    ``stol`` is a site tolerance parameter used in ``pymatgen``
    |StructureMatcher| functions, defined as the fraction of the average free
    length per atom := ( V / Nsites ) ** (1/3).

    Args:
        stol (float): Site tolerance parameter.
        structure (|Structure|): |Structure| to get equivalent distance for.

    Returns:
        float: Equivalent Cartesian distance for the given ``stol`` value.
    """
    return stol * (structure.volume / len(structure)) ** (1 / 3)


def get_bond_centre_sites(
    host_structure: Structure, bond_tol: float = 0.2, symprec: float = 0.01
) -> list[tuple[np.ndarray, int, list[np.ndarray]]]:
    """
    Get the symmetry-inequivalent bond-centre sites (i.e. midpoints of nearest-
    neighbour bonds) in ``host_structure``.

    Bond-centre sites are often favoured by hydrogen interstitials (e.g. in
    covalent semiconductors), but are excluded by the Voronoi approach in
    ``get_interstitial_sites`` (as they lie close to host atoms and are
    unlikely for most interstitial species, other than Hydrogen).

    Bonds are taken as pairs of sites separated by no more than
    ``(1 + bond_tol)`` times the shortest bond length of either site (i.e. its
    distance to its nearest neighbour, of any species). Bonds to hydrogen are
    excluded.

    Args:
        host_structure (|Structure|): Host structure.
        bond_tol (float):
            Fractional tolerance for bond lengths, relative to the shortest
            bond length of each site, to include for bond centre sites.
            Default is 0.2.
        symprec (float):
            Symmetry precision for (symmetry-)equivalent site determination.
            Default is 0.01.

    Returns:
        list[tuple[np.ndarray, int, list[np.ndarray]]]:
            List of ``(frac_coords, multiplicity, equiv_frac_coords)`` for each
            symmetry-inequivalent bond-centre site, as in the output of
            ``get_interstitial_sites``.
    """
    r = 2 * (host_structure.volume / len(host_structure)) ** (1 / 3)  # ~2x mean site spacing
    while not all(all_neighbours := host_structure.get_all_neighbors(r)):  # need a neighbour for each site
        r *= 1.5
    shortest_bonds = [min(nbr.nn_distance for nbr in neighbours) for neighbours in all_neighbours]
    if (bond_cutoff := (1 + bond_tol) * max(shortest_bonds)) > r:
        all_neighbours = host_structure.get_all_neighbors(bond_cutoff)

    bond_centre_sites: list[tuple[np.ndarray, int, list[np.ndarray]]] = []
    for i, (site, neighbours) in enumerate(zip(host_structure, all_neighbours, strict=True)):
        for neighbour in neighbours:
            if "H" in {site.specie.symbol, neighbour.specie.symbol} or neighbour.nn_distance > (
                1 + bond_tol
            ) * max(shortest_bonds[i], shortest_bonds[neighbour.index]):
                continue  # not bonded, or bonded to H (bond centres of bonds to H are not meaningful)
            bond_centre = host_structure.lattice.get_fractional_coords(
                (site.coords + neighbour.coords) / 2
            )
            if any(  # each bond is found from both ends, and symmetry-equivalent bonds already included
                np.min(host_structure.lattice.get_all_distances(equiv_fpos, bond_centre)) < 0.1
                for *_, equiv_fpos in bond_centre_sites
            ):
                continue
            equiv_fpos = symmetry.get_all_equiv_sites(
                bond_centre, host_structure, symprec=symprec, just_frac_coords=True
            )
            equiv_fpos = sorted(cast("list[np.ndarray]", equiv_fpos), key=symmetry._frac_coords_sort_func)
            bond_centre_sites.append((equiv_fpos[0], len(equiv_fpos), equiv_fpos))

    return bond_centre_sites


@lru_cache(maxsize=int(1e4))
def get_interstitial_sites(
    host_structure: Structure,
    min_dist: float = 0.9,
    clustering_tol: float = 0.8,
    symm_pref_dist_factor: float = 0.85,
    stol: float = 0.32,
    tight_stol: float = 0.02,
    symprec: float = 0.01,
    vacuum_radius: float | None = None,
    include_unique_wyckoffs: bool = False,
    include_bond_centres: bool = False,
    bond_tol: float = 0.2,
) -> list:
    """
    Generate candidate interstitial sites using Voronoi analysis.

    This function uses a similar approach to that in
    ``VoronoiInterstitialGenerator`` from ``pymatgen-analysis-defects``, but
    with modifications to make interstitial generation much faster, fix bugs
    with interstitial grouping (which could lead to undesired dropping of
    candidate sites) and achieve better control over site placement in order to
    favour sites which are higher-symmetry and furthest from the host lattice
    atoms (typically the most favourable interstitial sites).

    If the structure is detected to have a significant vacuum volume
    (determined using ``vacuum_radius``), thus corresponding to a
    slab/layer/line structure, then the ``pymatgen`` adsorbate finder will be
    used to generate further candidate interstitial sites.

    The logic for picking interstitial sites is as follows:

    - Generate all candidate sites using (efficient) Voronoi analysis
    - If vacuum volume present, include adsorbate sites as candidate sites
    - Remove any sites which are within ``min_dist`` of any host atoms
    - Cluster the remaining sites using a tolerance of ``clustering_tol`` and
      symmetry-preference of ``symm_pref_dist_factor``
      (see ``doped_cluster_frac_coords``)
    - Determine the multiplicities and symmetry-equivalent coordinates of the
      clustered sites using ``doped`` symmetry functions.
    - Group the clustered sites by symmetry using (looser) site matching as
      controlled by ``stol``.
    - From each group, pick the site with the highest symmetry and furthest
      distance from the host atoms, if its ``min_dist`` is no more than
      no more than ``symm_pref_dist_factor`` (0.85 by default) times the largest
      possible ``min_dist`` (distanceto the host atoms).
    - If ``include_bond_centres`` is ``True``, add bond-centre sites (see
      ``get_bond_centre_sites``), excluding any within ``min_dist`` of host
      atoms or coinciding with the sites above (i.e. within the distance
      equivalent of ``tight_stol``; see ``get_stol_equiv_dist``).

    (Parameters mentioned here can be supplied via ``interstitial_gen_kwargs``
    as noted in the args section below.)

    The motivation for favouring high symmetry interstitial sites and then
    distance to host atoms is because higher symmetry interstitial sites are
    typically the more intuitive sites for placement, cleaner, easier for
    analysis etc., and interstitials are often lowest-energy when furthest from
    host atoms (i.e. in the largest interstitial voids -- particularly for
    fully-ionised charge states), and so this approach tries to strike a
    balance between these two goals.

    One caveat to this preference for high symmetry interstitial sites, is that
    they can also be slightly more prone to being stuck in local minima on the
    PES, and so as always it is **highly recommended** to use |ShakeNBreak|
    or another structure-searching technique to account for symmetry-breaking
    when performing defect relaxations!

    You can see what Cartesian distance the chosen ``stol`` corresponds to
    using the ``get_stol_equiv_dist`` function. Note that you will likely want
    to reduce ``min_dist`` and include bond-centre sites for hydrogen
    interstitials! (This is done by default with |DefectsGenerator|, reducing
    ``min_dist`` to 0.5 Å and setting ``include_bond_centres = True`` for
    Hydrogen.)

    Args:
        host_structure (|Structure|): Host structure.
        min_dist (float):
            Minimum distance from host atoms for interstitial sites.
            Defaults to 0.9 Å.
        clustering_tol (float):
            Tolerance for clustering interstitial sites. Defaults to 0.8 Å.
        symm_pref_dist_factor (float):
            Minimum acceptable ratio of distance to host atoms for
            symmetry-favoured sites vs distance-to-host-favoured sites, for
            which to prefer symmetry-favoured sites. Defaults to 0.85.
        stol (float):
            Structure matcher tolerance for looser site matching. Defaults
            to 0.32.
        tight_stol (float):
            Structure matcher tolerance for tighter site matching. Defaults
            to 0.02.
        symprec (float):
            Symmetry precision for (symmetry-)equivalent site
            determination. Defaults to 0.01.
        vacuum_radius (float):
            Tolerance radius for determining if a significant vacuum volume
            is present in the structure (if any Voronoi node has a minimum
            distance to a host atom greater than this value), in which case
            adsorbate sites will also be included for interstitial
            generation. Defaults to ``None``, in which case it is set to 1.5
            times the minimum bulk bond length.
        include_unique_wyckoffs (bool):
            Include unique Wyckoff positions (those with no free variables)
            in the set of candidate interstitial sites. This can help
            ensure that expected canonical interstitial sites (e.g.
            distorted tetrahedral and octahedral in BCC crystals) are included,
            but will increase the number of candidate interstitials and is
            not expected to locate lower-energy sites compared to the default
            Voronoi approach in most cases. Defaults to ``False``.
        include_bond_centres (bool):
            Include bond-centre sites (i.e. midpoints of nearest-neighbour
            bonds; see ``get_bond_centre_sites``) in the set of candidate
            interstitial sites. These are often favoured by hydrogen
            interstitials, but are excluded by the Voronoi generation approach.
            Defaults to ``False`` (but set to ``True`` for Hydrogen by default
            in |DefectsGenerator|).
        bond_tol (float):
            Fractional bond length tolerance for determining bond-centre sites,
            if ``include_bond_centres`` is ``True`` (see
            ``get_bond_centre_sites``). Default is 0.2.


    Returns:
        list: List of interstitial sites as fractional coordinates
    """
    # so we want to pick the higher symmetry sites because it's cleaner, more intuitive etc
    # but, this is slightly more likely to be stuck in local minima, compared to the (nearby)
    # lower symmetry interstitial sites... avoided by using ShakeNBreak, other structure-searching
    # approaches, or rattling the output structures (default in ``doped.vasp``)
    sites_list = [v.frac_coords for v in DopedTopographyAnalyzer(host_structure).vnodes]
    if not sites_list:  # empty list
        warnings.warn("No interstitial sites found in host structure!")
        return sites_list

    bulk_min_bond_length = supercells.min_dist(host_structure)
    if vacuum_radius is None:
        vacuum_radius = max(bulk_min_bond_length * 1.5, 2.5)
    vacuum_sites = remove_collisions(
        sites_list,
        structure=host_structure,
        min_dist=vacuum_radius,
    )
    if vacuum_sites.size > 0:  # low-dimensional structure with significant vacuum, include adsorbate sites
        asf = AdsorbateSiteFinder(host_structure)
        adsorption_sites = asf.find_adsorption_sites()["all"]  # in cartesian coordinates
        sites_list.extend(  # convert to frac coords and add to list
            [host_structure.lattice.get_fractional_coords(i) for i in adsorption_sites]
        )

    if include_unique_wyckoffs:
        sga = symmetry.get_sga(host_structure, symprec=symprec)
        wyckoff_label_dict = symmetry.get_wyckoff_dict_from_sgn(sga.get_space_group_number())
        wyckoff_candidate_frac_coords = np.unique(
            [
                np.mod(np.array([float(f) for f in coords]), 1)
                for label, coords_list in wyckoff_label_dict.items()
                if np.all([coord.is_constant() for coord in coords_list[0]])
                for coords in coords_list
            ],
            axis=0,
        )
        sites_list.extend(wyckoff_candidate_frac_coords)

    sites_array = remove_collisions(sites_list, structure=host_structure, min_dist=min_dist)
    site_frac_coords_array = symmetry.doped_cluster_frac_coords(
        sites_array,
        host_structure,
        tol=clustering_tol,
        symm_pref_dist_factor=symm_pref_dist_factor,
    )

    label_equiv_fpos_dict: dict[int, list[np.ndarray]] = {}
    tight_dist = get_stol_equiv_dist(tight_stol, host_structure)  # 0.06 Å for CdTe, Sb2Si2Te6

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
                for site in cast(  # just_frac_coords=False -> PeriodicSite objects
                    "list[PeriodicSite]",
                    symmetry.get_all_equiv_sites(
                        frac_coords,
                        host_structure,
                        symprec=symprec,
                    ),
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
        stol,  # matches pymatgen-analysis-defects default
        host_structure,  # ~1 Å for CdTe, Sb2Si2Te6
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
                        # chaining effect (e.g. site 1 -> 1 Å from site 2 -> 1 Å from site 3 (but 2 Å
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

    cand_site_mul_and_equiv_fpos_list: list[tuple] = []
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
                        min(cand_site_mul_and_equiv_fpos[2], key=symmetry._frac_coords_sort_func)
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
                    (
                        np.min(
                            host_structure.lattice.get_all_distances(
                                symmetry_favoured_site_mul_and_equiv_fpos[0], host_structure.frac_coords
                            ),
                            axis=1,
                        )
                        / np.min(
                            host_structure.lattice.get_all_distances(
                                dist_favoured_site_mul_and_equiv_fpos[0], host_structure.frac_coords
                            ),
                            axis=1,
                        )
                    )
                    < symm_pref_dist_factor
                )
                else symmetry_favoured_site_mul_and_equiv_fpos
            )

    # bond-centre sites are added after grouping, as they could otherwise be dropped in favour of nearby
    # (higher-symmetry, further from host atoms) candidate sites by the looser site grouping above:
    if include_bond_centres:
        cand_fpos = [fpos for *_, equiv_fpos in cand_site_mul_and_equiv_fpos_list for fpos in equiv_fpos]
        for bond_centre_site_mul_and_equiv_fpos in get_bond_centre_sites(
            host_structure, bond_tol=bond_tol, symprec=symprec
        ):
            bond_centre_fpos = bond_centre_site_mul_and_equiv_fpos[0]
            cand_dists = host_structure.lattice.get_all_distances(cand_fpos, bond_centre_fpos)
            if (
                remove_collisions([bond_centre_fpos], structure=host_structure, min_dist=min_dist).size
                and np.min(cand_dists, initial=np.inf) > tight_dist
            ):  # not coinciding with other candidates or within min_dist of host atoms
                cand_site_mul_and_equiv_fpos_list.append(bond_centre_site_mul_and_equiv_fpos)

    sorted_sites_mul_and_equiv_fpos = []
    for _cand_site, multiplicity, equiv_fpos in cand_site_mul_and_equiv_fpos_list:
        # take site with equiv_fpos sorted by symmetry._frac_coords_sort_func:
        sorted_equiv_fpos = sorted(equiv_fpos, key=symmetry._frac_coords_sort_func)
        ideal_cand_site = sorted_equiv_fpos[0]
        sorted_sites_mul_and_equiv_fpos.append((ideal_cand_site, multiplicity, sorted_equiv_fpos))

    if not sorted_sites_mul_and_equiv_fpos:
        warnings.warn(
            f"No interstitial sites found after removing those within {min_dist} Å of host atoms!"
        )

    return sorted_sites_mul_and_equiv_fpos
