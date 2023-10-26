"""
Code to generate Defect objects and supercell structures for ab-initio
calculations.
"""
import copy
import warnings
from functools import partial
from itertools import chain
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple, Type, Union, cast
from unittest.mock import MagicMock

import numpy as np
from monty.json import MontyDecoder, MSONable
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.defects import core
from pymatgen.analysis.defects.generators import (
    AntiSiteGenerator,
    InterstitialGenerator,
    SubstitutionGenerator,
    VacancyGenerator,
    VoronoiInterstitialGenerator,
)
from pymatgen.analysis.defects.supercells import get_sc_fromstruct
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.composition import Composition, Element
from pymatgen.core.periodic_table import DummySpecies
from pymatgen.core.structure import PeriodicSite, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.transformations.advanced_transformations import _proj
from tabulate import tabulate
from tqdm import tqdm

from doped.core import Defect, DefectEntry, Interstitial, Substitution, Vacancy
from doped.utils.wyckoff import (
    _custom_round,
    _frac_coords_sort_func,
    _get_equiv_frac_coords_in_primitive,
    _get_sga,
    _rotate_and_get_supercell_matrix,
    _round_floats,
    _vectorized_custom_round,
    get_BCS_conventional_structure,
    get_conv_cell_site,
    get_primitive_structure,
    get_wyckoff,
    get_wyckoff_label_and_equiv_coord_list,
)

_dummy_species = DummySpecies("X")  # Dummy species used to keep track of defect coords in the supercell


def _custom_formatwarning(
    message: Union[Warning, str],
    category: Type[Warning],
    filename: str,
    lineno: int,
    line: Optional[str] = None,
) -> str:
    """
    Reformat warnings to just print the warning message.
    """
    return f"{message}\n"


warnings.formatwarning = _custom_formatwarning


def get_defect_entry_from_defect(
    defect: Defect,
    defect_supercell: Structure,
    charge_state: int,
    dummy_species: DummySpecies = _dummy_species,
):
    """
    Generate doped DefectEntry object from a doped Defect object.

    This is used to describe a Defect with a specified simulation cell.
    """
    defect_entry_structure = (
        defect_supercell.copy()
    )  # duplicate the structure so we don't edit the input Structure

    # Dummy species (used to keep track of the defect coords in the supercell)
    # Find its fractional coordinates & remove it from the supercell
    dummy_site = [
        site for site in defect_entry_structure if site.species.elements[0].symbol == dummy_species.symbol
    ][0]
    sc_defect_frac_coords = dummy_site.frac_coords
    defect_entry_structure.remove(dummy_site)

    computed_structure_entry = ComputedStructureEntry(
        structure=defect_entry_structure,
        energy=0.0,  # needs to be set, so set to 0.0
    )
    return DefectEntry(
        defect=defect,
        charge_state=charge_state,
        sc_entry=computed_structure_entry,
        sc_defect_frac_coords=sc_defect_frac_coords,
    )


def _defect_dict_key_from_pmg_type(defect_type: core.DefectType) -> str:
    """
    Returns the corresponding defect dictionary key for a Defect object.
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


def closest_site_info(defect_entry, n=1, element_list=None):
    """
    Return the element and distance (rounded to 2 decimal places) of the
    closest site to defect_entry.sc_defect_frac_coords in
    defect_entry.sc_entry.structure, with distance > 0.01 (i.e. so not the site
    itself), and if there are multiple elements with the same distance, sort by
    order of appearance of elements in the composition, then alphabetically and
    return the first one.

    If n is set, then it returns the nth closest site, where the nth site must be at least
    0.02 Å further away than the n-1th site.
    """
    if element_list is None:
        element_list = [  # host elements
            el.symbol for el in defect_entry.defect.structure.composition.elements
        ]
        element_list += sorted(
            [  # extrinsic elements, sorted alphabetically for deterministic ordering in output:
                el.symbol
                for el in defect_entry.defect.defect_structure.composition.elements
                if el.symbol not in element_list
            ]
        )

    # use defect_supercell_site if attribute exists, otherwise use sc_defect_frac_coords:
    defect_site = (
        defect_entry.defect_supercell_site
        if hasattr(defect_entry, "defect_supercell_site")
        else PeriodicSite("X", defect_entry.sc_defect_frac_coords, defect_entry.sc_entry.structure.lattice)
    )
    site_distances = sorted(
        [
            (
                site.distance(defect_site),
                site.specie.symbol,
            )
            for site in defect_entry.sc_entry.structure.sites
            if site.distance(defect_site) > 0.01
        ],
        key=lambda x: (_custom_round(x[0]), element_list.index(x[1]), x[1]),
    )

    # prune site_distances to remove any tuples with distances within 0.02 Å of the previous entry:
    site_distances = [
        site_distances[i]
        for i in range(len(site_distances))
        if i == 0
        or abs(site_distances[i][0] - site_distances[i - 1][0]) > 0.02
        or site_distances[i][1] != site_distances[i - 1][1]
    ]

    min_distance, closest_site = site_distances[n - 1]

    return f"{closest_site}{_custom_round(min_distance, 2):.2f}"


def get_defect_name_from_entry(defect_entry, element_list=None):
    """
    Get the doped/SnB defect name from DefectEntry object.
    """
    defect_diagonal_supercell = defect_entry.defect.get_supercell_structure(
        sc_mat=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        dummy_species=_dummy_species,
    )  # create defect supercell, which is a diagonal expansion of the unit cell so that the defect
    # periodic image retains the unit cell symmetry, in order not to affect the point group symmetry
    sga = _get_sga(defect_diagonal_supercell)
    return (
        f"{defect_entry.defect.name}_{herm2sch(sga.get_point_group_symbol())}"
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
    symm_ops,
):
    (
        dummy_defect_supercell,
        defect_supercell_site,
        equivalent_supercell_sites,
    ) = defect.get_supercell_structure(
        sc_mat=supercell_matrix,
        dummy_species=_dummy_species,  # keep track of the defect frac coords in the supercell
        target_frac_coords=target_frac_coords,
        return_sites=True,
    )
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

    neutral_defect_entry.conventional_structure = (
        neutral_defect_entry.defect.conventional_structure
    ) = conventional_structure

    try:
        wyckoff_label, conv_cell_sites = get_wyckoff(
            get_conv_cell_site(neutral_defect_entry).frac_coords,
            conventional_structure,
            symm_ops,
            equiv_sites=True,
        )
        conv_cell_coord_list = [
            _vectorized_custom_round(np.mod(_vectorized_custom_round(site.to_unit_cell().frac_coords), 1))
            for site in conv_cell_sites
        ]

    except Exception as e:  # (slightly) less efficient algebraic matching:
        try:
            wyckoff_label, conv_cell_coord_list = get_wyckoff_label_and_equiv_coord_list(
                defect_entry=neutral_defect_entry,
                wyckoff_dict=wyckoff_label_dict,
            )
            conv_cell_coord_list = _vectorized_custom_round(
                np.mod(_vectorized_custom_round(conv_cell_coord_list), 1)
            ).tolist()
        except Exception as e2:
            raise e2 from e

    # sort array with _frac_coords_sort_func:
    conv_cell_coord_list.sort(key=_frac_coords_sort_func)

    neutral_defect_entry.wyckoff = neutral_defect_entry.defect.wyckoff = wyckoff_label
    neutral_defect_entry.conv_cell_frac_coords = (
        neutral_defect_entry.defect.conv_cell_frac_coords
    ) = conv_cell_coord_list[
        0
    ]  # ideal/cleanest coords
    neutral_defect_entry.equiv_conv_cell_frac_coords = (
        neutral_defect_entry.defect.equiv_conv_cell_frac_coords
    ) = conv_cell_coord_list
    neutral_defect_entry._BilbaoCS_conv_cell_vector_mapping = (
        neutral_defect_entry.defect._BilbaoCS_conv_cell_vector_mapping
    ) = _BilbaoCS_conv_cell_vector_mapping

    return neutral_defect_entry


def name_defect_entries(defect_entries, element_list=None):
    """
    Create a dictionary of {Name: DefectEntry} from a list of DefectEntry
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
    etc is appended to the name of different defects to distinguish.
    """

    def get_shorter_name(full_defect_name, split_number):
        if split_number < 1:  # if split number is less than 1, return full name
            return full_defect_name
        return full_defect_name.rsplit("_", split_number)[0]

    def get_matching_names(defect_naming_dict, defect_name):
        return [name for name in defect_naming_dict if defect_name.startswith(name)]

    def handle_unique_match(defect_naming_dict, matching_names, split_number):
        if len(matching_names) == 1:
            previous_entry = defect_naming_dict.pop(matching_names[0])
            previous_entry_full_name = get_defect_name_from_entry(previous_entry, element_list)
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

            if not any(name.startswith(full_defect_name) for name in defect_naming_dict):
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
            if full_defect_name == name[:-1]:
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

    defect_naming_dict = {}
    for defect_entry in defect_entries:
        full_defect_name = get_defect_name_from_entry(defect_entry, element_list)
        split_number = 1 if defect_entry.defect.defect_type == core.DefectType.Interstitial else 2
        shorter_defect_name = get_shorter_name(full_defect_name, split_number)
        if not any(name.startswith(shorter_defect_name) for name in defect_naming_dict):
            defect_naming_dict[shorter_defect_name] = defect_entry
            continue

        matching_shorter_names = get_matching_names(defect_naming_dict, shorter_defect_name)
        defect_naming_dict = handle_unique_match(defect_naming_dict, matching_shorter_names, split_number)
        shorter_defect_name = get_shorter_name(full_defect_name, split_number - 1)
        if not any(name.startswith(shorter_defect_name) for name in defect_naming_dict):
            defect_naming_dict[shorter_defect_name] = defect_entry
            continue

        matching_shorter_names = get_matching_names(defect_naming_dict, shorter_defect_name)
        defect_naming_dict = handle_unique_match(
            defect_naming_dict, matching_shorter_names, split_number - 1
        )
        shorter_defect_name = get_shorter_name(full_defect_name, split_number - 2)
        if not any(name.startswith(shorter_defect_name) for name in defect_naming_dict):
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


def herm2sch(herm_symbol):
    """
    Convert from Hermann-Mauguin to Schoenflies.
    """
    return _HERM2SCH.get(herm_symbol, None)


def get_oxi_probabilities(element_symbol: str) -> dict:
    """
    Get a dictionary of oxidation states and their probabilities for an
    element.

    Tries to get the probabilities from the `pymatgen` tabulated ICSD oxidation
    state probabilities, and if not available, uses the common oxidation states
    of the element.

    Args:
        element_symbol (str): Element symbol.

    Returns:
        dict: Dictionary of oxidation states and their probabilities.
    """
    comp_obj = Composition(element_symbol)
    oxi_probabilities = {
        k: v
        for k, v in comp_obj.oxi_prob.items()
        if k.element.symbol == element_symbol and k.oxi_state != 0
    }
    if oxi_probabilities:  # not empty
        return {k: round(v / sum(oxi_probabilities.values()), 3) for k, v in oxi_probabilities.items()}

    element_obj = Element(element_symbol)
    if element_obj.common_oxidation_states:
        return {
            k: 1 / len(element_obj.common_oxidation_states) for k in element_obj.common_oxidation_states
        }  # known common oxidation states

    # no known _common_ oxidation state, make guess and warn user
    if element_obj.oxidation_states:
        oxi_states = {
            k: 1 / len(element_obj.oxidation_states) for k in element_obj.oxidation_states
        }  # known oxidation states
    else:
        oxi_states = {0: 1}  # no known oxidation states, return 0 with 100% probability

    warnings.warn(
        f"No known common oxidation states in pymatgen/ICSD dataset for element "
        f"{element_obj.name}. If this results in unreasonable charge states, you "
        f"should manually edit the defect charge states."
    )

    return oxi_states


def _charge_state_probability(
    charge_state: int,
    defect_elt_oxi_state: int,
    defect_elt_oxi_probability: float,
    max_host_oxi_magnitude: int,
    return_log: bool = False,
) -> Union[float, dict]:
    """
    Function to estimate the probability of a given defect charge state, using
    the probability of the corresponding defect element oxidation state, the
    magnitude of the charge state, and the maximum magnitude of the host
    oxidation states (i.e. how 'charged' the host is).

    Disfavours large (absolute) charge states, low probability oxidation
    states and greater charge/oxidation state magnitudes than that of the host.

    Args:
        charge_state (int): Charge state of defect.
        defect_elt_oxi_state (int): Oxidation state of defect element.
        defect_elt_oxi_probability (float):
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
            "charge_state": charge_state,
            "oxi_state": defect_elt_oxi_state,
            "oxi_probability": defect_elt_oxi_probability,
            "max_host_oxi_magnitude": max_host_oxi_magnitude,
        },
        "probability_factors": {
            "oxi_probability": defect_elt_oxi_probability,
            "charge_state_magnitude": (1 / abs(charge_state)) ** (2 / 3) if charge_state != 0 else 1,
            "charge_state_vs_max_host_charge": _defect_vs_host_charge(charge_state, max_host_oxi_magnitude)
            ** (2 / 3),
            "oxi_state_vs_max_host_charge": _defect_vs_host_charge(
                defect_elt_oxi_state, max_host_oxi_magnitude
            )
            ** (2 / 3),
        },
    }
    # product of charge_state_guessing_log["probability_factors"].values()
    charge_state_guessing_log["probability"] = (
        np.product(list(charge_state_guessing_log["probability_factors"].values()))
        if charge_state != 0
        else 1
    )  # always include neutral charge state

    if return_log:
        return charge_state_guessing_log

    return charge_state_guessing_log["probability"]


def _get_vacancy_charge_states(defect: Vacancy, padding: int = 1) -> List[int]:
    """
    Get the possible charge states for a vacancy defect, which is from +/-1 to
    the vacancy oxidation state.

    Args:
        defect (Defect): A doped Vacancy object.
        padding (int):
            Padding for vacancy charge states, such that the vacancy
            charge states are set to range(vacancy oxi state, padding),
            if vacancy oxidation state is negative, or to
            range(-padding, vacancy oxi state), if positive.
            Default is 1.

    Returns:
        List[int]: A list of possible charge states for the defect.
    """
    if defect.oxi_state > 0:
        return list(range(-padding, int(defect.oxi_state) + 1))  # from -1 to oxi_state
    if defect.oxi_state < 0:
        return list(range(int(defect.oxi_state), padding + 1))  # from oxi_state to +1

    # oxi_state is 0
    return list(range(-padding, padding + 1))  # from -1 to +1 for default


def _get_possible_oxi_states(defect: Defect) -> Dict:
    """
    Get the possible oxidation states and probabilities for a defect.

    Args:
        defect (Defect): A doped Defect object.

    Returns:
        Dict: A dictionary with possible oxidation states as
            keys and their probabilities as values.
    """
    return {
        k.oxi_state: prob
        for k, prob in get_oxi_probabilities(defect.site.specie.symbol).items()
        if prob > 0.01  # at least 1% occurrence
    }


def _get_charge_states(
    possible_oxi_states: Dict,
    orig_oxi: int,
    max_host_oxi_magnitude: int,
    return_log: bool = False,
) -> Dict:
    return {
        oxi
        - orig_oxi: _charge_state_probability(
            oxi - orig_oxi, oxi, oxi_prob, max_host_oxi_magnitude, return_log=return_log
        )
        for oxi, oxi_prob in possible_oxi_states.items()
    }


def guess_defect_charge_states(
    defect: Defect, probability_threshold: float = 0.01, padding: int = 1, return_log: bool = False
) -> Union[List[int], Tuple[List[int], List[Dict]]]:
    """
    Guess the possible charge states of a defect.

    Args:
        defect (Defect): doped Defect object.
        probability_threshold (float):
            Probability threshold for including defect charge states
            (for substitutions and interstitials). Default is 0.01.
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
        vacancy_charge_states = _get_vacancy_charge_states(defect, padding=padding)
        if return_log:
            charge_state_guessing_log = [
                {
                    "input_parameters": {
                        "charge_state": charge_state,
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
    max_host_oxi_magnitude = max(abs(site.specie.oxi_state) for site in defect.structure)
    if defect.defect_type == core.DefectType.Substitution:
        orig_oxi = defect.structure[defect.defect_site_index].specie.oxi_state
    else:  # interstitial
        orig_oxi = 0
    possible_charge_states = _get_charge_states(
        possible_oxi_states, orig_oxi, max_host_oxi_magnitude, return_log=True
    )

    charge_state_list = [
        k for k, v in possible_charge_states.items() if v["probability"] > probability_threshold
    ]
    if charge_state_list:
        charge_state_range = (int(min(charge_state_list)), int(max(charge_state_list)))
    else:
        charge_state_range = (0, 0)

    # check if defect element (interstitial/substitution) is present in structure (i.e. intrinsic
    # interstitial or antisite):
    defect_elt_sites_in_struct = [
        site for site in defect.structure if site.specie.symbol == defect.site.specie.symbol
    ]
    defect_elt_oxi_in_struct = (
        int(np.mean([site.specie.oxi_state for site in defect_elt_sites_in_struct]))
        if defect_elt_sites_in_struct
        else None
    )

    if (
        defect.defect_type == core.DefectType.Substitution
        and defect_elt_oxi_in_struct is not None
        and defect_elt_oxi_in_struct - orig_oxi
        not in range(charge_state_range[0], charge_state_range[1] + 1)
    ):
        # if simple antisite oxidation state difference not included, recheck with bumped up oxi_state
        # probability for the oxi_state of the substitution atom in the structure
        # should really be included unless it gives an absolute charge state >= 5, so set oxi_state
        # probability to 100%
        possible_charge_states[defect_elt_oxi_in_struct - orig_oxi] = _charge_state_probability(
            defect_elt_oxi_in_struct - orig_oxi,
            defect_elt_oxi_in_struct,
            1,
            max_host_oxi_magnitude,
            return_log=True,
        )

    if (
        defect.defect_type == core.DefectType.Interstitial
        and defect_elt_oxi_in_struct is not None
        and defect_elt_oxi_in_struct not in range(charge_state_range[0], charge_state_range[1] + 1)
    ):
        # if oxidation state of interstitial element in the host structure is not included, include it!
        possible_charge_states[defect_elt_oxi_in_struct] = _charge_state_probability(
            defect_elt_oxi_in_struct - orig_oxi,
            defect_elt_oxi_in_struct,
            1,
            max_host_oxi_magnitude,
            return_log=True,
        )

    sorted_charge_state_dict = dict(
        sorted(possible_charge_states.items(), key=lambda x: x[1]["probability"], reverse=True)
    )

    charge_state_list = [
        k for k, v in sorted_charge_state_dict.items() if v["probability"] > probability_threshold
    ]
    if charge_state_list:
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
        and defect_elt_oxi_in_struct is not None
        and defect_elt_oxi_in_struct - orig_oxi == 0
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


class DefectsGenerator(MSONable):
    """
    Class for generating doped DefectEntry objects.
    """

    def __init__(
        self,
        structure: Structure,
        extrinsic: Optional[Union[str, List, Dict]] = None,
        interstitial_coords: Optional[List] = None,
        generate_supercell: bool = True,
        charge_state_gen_kwargs: Optional[Dict] = None,
        supercell_gen_kwargs: Optional[Dict] = None,
        interstitial_gen_kwargs: Optional[Dict] = None,
        target_frac_coords: Optional[List] = None,
        processes: Optional[int] = None,
    ):
        """
        Generates doped DefectEntry objects for defects in the input host
        structure. By default, generates all intrinsic defects, but extrinsic
        defects (impurities) can also be created using the `extrinsic`
        argument.

        Interstitial sites are generated using Voronoi tessellation by default (found
        to be the most reliable), which can be controlled using the
        `interstitial_gen_kwargs` argument (passed as keyword arguments to the
        `VoronoiInterstitialGenerator` class). Alternatively, a list of interstitial
        sites (or single interstitial site) can be manually specified using the
        `interstitial_coords` argument.

        By default, supercells are generated for each defect using the pymatgen
        `get_supercell_structure()` method, with `doped` default settings of
        `min_length = 10` (minimum supercell length of 10 Å) and `min_atoms = 50`
        (minimum 50 atoms in supercell). If a different supercell is desired, this
        can be controlled by specifying keyword arguments with `supercell_gen_kwargs`,
        which are passed to the `get_sc_fromstruct()` function.
        Alternatively if `generate_supercell = False`, then no supercell is generated
        and the input structure is used as the defect & bulk supercell.

        The algorithm for determining defect entry names is to use the pymatgen defect
        name (e.g. v_Cd, Cd_Te etc.) for vacancies/antisites/substitutions, unless
        there are multiple inequivalent sites for the defect, in which case the point
        group of the defect site is appended (e.g. v_Cd_Td, Cd_Te_Td etc.), and if
        this is still not unique, then element identity and distance to the nearest
        neighbour of the defect site is appended (e.g. v_Cd_Td_Te2.83, Cd_Te_Td_Cd2.83
        etc.). For interstitials, the same naming scheme is used, but the point group
        is always appended to the pymatgen defect name.

        Possible charge states for the defects are estimated using the probability of
        the corresponding defect element oxidation state, the magnitude of the charge
        state, and the maximum magnitude of the host oxidation states (i.e. how
        'charged' the host is), with large (absolute) charge states, low probability
        oxidation states and/or greater charge/oxidation state magnitudes than that of
        the host being disfavoured. This can be controlled using the
        `probability_threshold` or `padding` keys in the `charge_state_gen_kwargs`
        parameter, which are passed to the `_charge_state_probability()` function.
        The input and computed values used to guess charge state probabilities are
        provided in the `DefectEntry.charge_state_guessing_log` attributes.
        See docs for examples of modifying the generated charge states.

        Args:
            structure (Structure):
                Structure of the host material (as a pymatgen Structure object).
                If this is not the primitive unit cell, it will be reduced to the
                primitive cell for defect generation, before supercell generation.
            extrinsic (Union[str, List, Dict]):
                List or dict of elements (or string for single element) to be used
                for extrinsic defect generation (i.e. dopants/impurities). If a
                list is provided, all possible substitutional defects for each
                extrinsic element will be generated. If a dict is provided, the keys
                should be the host elements to be substituted, and the values the
                extrinsic element(s) to substitute in; as a string or list.
                In both cases, all possible extrinsic interstitials are generated.
            interstitial_coords (List):
                List of fractional coordinates (corresponding to the input structure),
                or a single set of fractional coordinates, to use as interstitial
                defect site(s). Default (when interstitial_coords not specified) is
                to automatically generate interstitial sites using Voronoi tessellation.
                The input interstitial_coords are converted to
                DefectsGenerator.prim_interstitial_coords, which are the corresponding
                fractional coordinates in DefectsGenerator.primitive_structure (which
                is used for defect generation), sorted according to the doped
                _frac_coords_sort_func.
            generate_supercell (bool):
                Whether to generate a supercell for the output defect entries
                (using pymatgen's `CubicSupercellTransformation` and ASE's
                `find_optimal_cell_shape()` functions) or not. If False, then the
                input structure is used as the defect & bulk supercell.
            charge_state_gen_kwargs (Dict):
                Keyword arguments to be passed to the `_charge_state_probability`
                function (such as `probability_threshold` (default = 0.01, used for
                substitutions and interstitials) and `padding` (default = 1, used for
                vacancies)) to control defect charge state generation.
            supercell_gen_kwargs (Dict):
                Keyword arguments to be passed to the `get_sc_fromstruct` function
                in `pymatgen.analysis.defects.supercells` (such as `min_atoms`
                (default = 50), `max_atoms` (default = 500), `min_length` (default
                = 10), and `force_diagonal` (default = False)).
            interstitial_gen_kwargs (Dict):
                Keyword arguments to be passed to the `VoronoiInterstitialGenerator`
                class (such as `clustering_tol`, `stol`, `min_dist` etc), or to
                `InterstitialGenerator` if `interstitial_coords` is specified.
            target_frac_coords (List):
                Defects are placed at the closest equivalent site to these fractional
                coordinates in the generated supercells. Default is [0.5, 0.5, 0.5]
                if not set (i.e. the supercell centre, to aid visualisation).
            processes (int):
                Number of processes to use for multiprocessing. If not set, defaults to
                one less than the number of CPUs available.

        Attributes:
            defect_entries (Dict): Dictionary of {defect_species: DefectEntry} for all
                defect entries (with charge state and supercell properties) generated.
            defects (Dict): Dictionary of {defect_type: [Defect, ...]} for all defect
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

            `DefectsGenerator` input parameters are also set as attributes.
        """
        self.defects: Dict[str, List[Defect]] = {}  # {defect_type: [Defect, ...]}
        self.defect_entries: Dict[str, DefectEntry] = {}  # {defect_species: DefectEntry}
        self.structure = structure
        self.extrinsic = extrinsic if extrinsic is not None else []
        if interstitial_coords is not None:
            # if a single list or array, convert to list of lists
            self.interstitial_coords = (
                [interstitial_coords]
                if not isinstance(interstitial_coords[0], (list, tuple, np.ndarray))
                else interstitial_coords  # ensure list of lists
            )
        else:
            self.interstitial_coords = []

        self.prim_interstitial_coords = None
        self.generate_supercell = generate_supercell
        self.charge_state_gen_kwargs = (
            charge_state_gen_kwargs if charge_state_gen_kwargs is not None else {}
        )
        self.supercell_gen_kwargs = supercell_gen_kwargs if supercell_gen_kwargs is not None else {}
        self.interstitial_gen_kwargs = (
            interstitial_gen_kwargs if interstitial_gen_kwargs is not None else {}
        )
        self.target_frac_coords = target_frac_coords if target_frac_coords is not None else [0.5, 0.5, 0.5]

        if len(self.structure) == 1 and not self.generate_supercell:
            # raise error if only one atom in primitive cell and no supercell generated, as vacancy will
            # give empty structure
            raise ValueError(
                "Input structure has only one site, so cannot generate defects without supercell (i.e. "
                "with generate_supercell=False)! Vacancy defect will give empty cell!"
            )

        pbar = tqdm(
            total=100, bar_format="{desc}{percentage:.1f}%|{bar}| [{elapsed},  {rate_fmt}{postfix}]"
        )  # tqdm progress
        # bar. 100% is completion
        pbar.set_description("Getting primitive structure")

        try:  # put code in try/except block so progress bar always closed if interrupted
            # Reduce structure to primitive cell for efficient defect generation
            # same symprec as defect generators in pymatgen-analysis-defects:
            sga = _get_sga(self.structure)

            prim_struct = get_primitive_structure(sga)
            if prim_struct.num_sites < self.structure.num_sites:
                primitive_structure = Structure.from_dict(_round_floats(prim_struct.as_dict()))

            else:  # primitive cell is the same as input structure, use input structure to avoid rotations
                # wrap to unit cell:
                primitive_structure = Structure.from_sites(
                    [site.to_unit_cell() for site in self.structure]
                )
                primitive_structure = Structure.from_dict(_round_floats(primitive_structure.as_dict()))

            pbar.update(5)  # 5% of progress bar

            # Generate supercell once, so this isn't redundantly rerun for each defect, and ensures the
            # same supercell is used for each defect and bulk calculation
            pbar.set_description("Generating simulation supercell")
            pmg_supercell_matrix = get_sc_fromstruct(
                primitive_structure,
                min_atoms=self.supercell_gen_kwargs.get("min_atoms", 50),  # different to current
                # pymatgen default (80)
                max_atoms=self.supercell_gen_kwargs.get(
                    "max_atoms", 500
                ),  # different to current pymatgen default (240)
                min_length=self.supercell_gen_kwargs.get(
                    "min_length", 10
                ),  # same as current pymatgen default
                force_diagonal=self.supercell_gen_kwargs.get(
                    "force_diagonal", False
                ),  # same as current pymatgen default
            )

            # check if input structure is already >10 Å in each direction:
            a = self.structure.lattice.matrix[0]
            b = self.structure.lattice.matrix[1]
            c = self.structure.lattice.matrix[2]

            length_vecs = np.array(
                [
                    c - _proj(c, a),  # a-c plane
                    a - _proj(a, c),
                    b - _proj(b, a),  # b-a plane
                    a - _proj(a, b),
                    c - _proj(c, b),  # b-c plane
                    b - _proj(b, c),
                ]
            )

            if np.min(np.linalg.norm(length_vecs, axis=1)) >= self.supercell_gen_kwargs.get(
                "min_length", 10
            ):
                # input structure is >10 Å in each direction
                if (
                    not self.generate_supercell
                    or self.structure.num_sites <= (primitive_structure * pmg_supercell_matrix).num_sites
                ):
                    # input structure has fewer or same number of atoms as pmg supercell or
                    # generate_supercell=False, so use input structure:
                    self.primitive_structure, self.supercell_matrix = _rotate_and_get_supercell_matrix(
                        primitive_structure, self.structure
                    )
                else:
                    self.primitive_structure = primitive_structure
                    self.supercell_matrix = pmg_supercell_matrix

            elif not self.generate_supercell:
                # input structure is <10 Å in at least one direction, and generate_supercell=False,
                # so use input structure but warn user:
                warnings.warn(
                    f"\nInput structure is <10 Å in at least one direction (minimum image distance = "
                    f"{np.min(np.linalg.norm(length_vecs, axis=1)):.2f} Å, which is usually too "
                    f"small for accurate defect calculations, but generate_supercell = False, so "
                    f"using input structure as defect & bulk supercells. Caution advised!"
                )
                self.primitive_structure, self.supercell_matrix = _rotate_and_get_supercell_matrix(
                    primitive_structure, self.structure
                )

            else:
                self.primitive_structure = primitive_structure
                self.supercell_matrix = pmg_supercell_matrix

            self.bulk_supercell = self.primitive_structure * self.supercell_matrix
            # check that generated supercell is >10 Å in each direction:
            if (
                np.min(np.linalg.norm(self.bulk_supercell.lattice.matrix, axis=1))
                < self.supercell_gen_kwargs.get("min_length", 10)
                and self.generate_supercell
            ):
                warnings.warn(
                    f"\nAuto-generated supercell is <10 Å in at least one direction (minimum image "
                    f"distance = "
                    f"{np.min(np.linalg.norm(self.bulk_supercell.lattice.matrix, axis=1)):.2f} Å, "
                    f"which is usually too small for accurate defect calculations. You can try increasing "
                    f"max_atoms (default = 500), or manually identifying a suitable supercell matrix "
                    f"(which can then be specified with the `supercell_matrix` argument)."
                )

            pbar.update(10)  # 15% of progress bar

            # Generate defects
            # Vacancies:
            pbar.set_description("Generating vacancies")
            vac_generator_obj = VacancyGenerator()
            vac_generator = vac_generator_obj.generate(self.primitive_structure)
            self.defects["vacancies"] = [Vacancy._from_pmg_defect(vac) for vac in vac_generator]
            pbar.update(5)  # 20% of progress bar

            # Antisites:
            pbar.set_description("Generating substitutions")
            antisite_generator_obj = AntiSiteGenerator()
            as_generator = antisite_generator_obj.generate(self.primitive_structure)
            self.defects["substitutions"] = [Substitution._from_pmg_defect(anti) for anti in as_generator]
            pbar.update(5)  # 25% of progress bar

            # Substitutions:
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
            # if any "extrinsic" elements are actually host elements, remove them from the list and warn
            # user:
            if any(el in host_element_list for el in extrinsic_elements):
                warnings.warn(
                    f"\nSpecified 'extrinsic' elements "
                    f"{[el for el in extrinsic_elements if el in host_element_list]} are present in "
                    f"the host structure, so do not need to be specified as 'extrinsic' in "
                    f"DefectsGenerator(). These will be ignored."
                )
            # sort extrinsic elements alphabetically for deterministic ordering in output:
            extrinsic_elements = sorted([el for el in extrinsic_elements if el not in host_element_list])

            substitution_generator_obj = SubstitutionGenerator()
            if isinstance(self.extrinsic, (str, list)):  # substitute all host elements:
                substitutions = {
                    el.symbol: extrinsic_elements for el in self.primitive_structure.composition.elements
                }
            elif isinstance(self.extrinsic, dict):  # substitute only specified host elements
                substitutions = self.extrinsic
            else:
                warnings.warn(
                    f"Invalid `extrinsic` defect input. Got type {type(self.extrinsic)}, but string or "
                    f"list or dict required. No extrinsic defects will be generated."
                )
                substitutions = {}

            if substitutions:
                sub_generator = substitution_generator_obj.generate(
                    self.primitive_structure, substitution=substitutions
                )
                sub_defects = [Substitution._from_pmg_defect(sub) for sub in sub_generator]
                if "substitutions" in self.defects:
                    self.defects["substitutions"].extend(sub_defects)
                else:
                    self.defects["substitutions"] = sub_defects
            if not self.defects["substitutions"]:  # no substitutions, single-elt system, no extrinsic
                del self.defects["substitutions"]  # remove empty list
            pbar.update(5)  # 30% of progress bar

            # Interstitials:
            pbar.set_description("Generating interstitials")
            self._element_list = host_element_list + extrinsic_elements  # all elements in system
            if self.interstitial_coords:
                # map interstitial coords to primitive structure, and get multiplicities
                sga = _get_sga(self.structure)
                symm_ops = sga.get_symmetry_operations(cartesian=False)
                self.prim_interstitial_coords = []

                for interstitial_frac_coords in self.interstitial_coords:
                    prim_inter_coords, equiv_coords = _get_equiv_frac_coords_in_primitive(
                        interstitial_frac_coords,
                        self.structure,
                        self.primitive_structure,
                        symm_ops,
                        equiv_coords=True,
                    )
                    self.prim_interstitial_coords.append(
                        (prim_inter_coords, len(equiv_coords), equiv_coords)
                    )

                sorted_sites_mul_and_equiv_fpos = self.prim_interstitial_coords

            else:
                # Generate interstitial sites using Voronoi tessellation
                vig = VoronoiInterstitialGenerator(**self.interstitial_gen_kwargs)
                tight_vig = VoronoiInterstitialGenerator(stol=0.01)  # for determining multiplicities of
                # any merged/grouped interstitial sites from Voronoi tessellation + structure-matching

                # parallelize Voronoi interstitial site generation:
                if cpu_count() >= 2 and len(self.primitive_structure) > 8:  # skip for small systems as
                    # communication overhead / process initialisation outweighs speedup
                    with Pool(2) as p:
                        interstitial_gen_mp_results = p.map(
                            _get_interstitial_candidate_sites,
                            [(vig, self.primitive_structure), (tight_vig, self.primitive_structure)],
                        )

                    cand_sites_mul_and_equiv_fpos = interstitial_gen_mp_results[0]
                    tight_cand_sites_mul_and_equiv_fpos = interstitial_gen_mp_results[1]

                else:
                    cand_sites_mul_and_equiv_fpos = [*vig._get_candidate_sites(self.primitive_structure)]
                    tight_cand_sites_mul_and_equiv_fpos = [
                        *tight_vig._get_candidate_sites(self.primitive_structure)
                    ]

                structure_matcher = StructureMatcher(
                    self.interstitial_gen_kwargs.get("ltol", 0.2),
                    self.interstitial_gen_kwargs.get("stol", 0.3),
                    self.interstitial_gen_kwargs.get("angle_tol", 5),
                )  # pymatgen-analysis-defects default
                unique_tight_cand_sites_mul_and_equiv_fpos = [
                    cand_site_mul_and_equiv_fpos
                    for cand_site_mul_and_equiv_fpos in tight_cand_sites_mul_and_equiv_fpos
                    if cand_site_mul_and_equiv_fpos not in cand_sites_mul_and_equiv_fpos
                ]

                # structure-match the non-matching site & multiplicity tuples, and return the site &
                # multiplicity of the tuple with the lower multiplicity (i.e. higher symmetry site)
                output_sites_mul_and_equiv_fpos = []
                for cand_site_mul_and_equiv_fpos in cand_sites_mul_and_equiv_fpos:
                    matching_sites_mul_and_equiv_fpos = []
                    if cand_site_mul_and_equiv_fpos not in tight_cand_sites_mul_and_equiv_fpos:
                        for (
                            tight_cand_site_mul_and_equiv_fpos
                        ) in unique_tight_cand_sites_mul_and_equiv_fpos:
                            interstitial_struct = self.primitive_structure.copy()
                            interstitial_struct.insert(
                                0, "H", cand_site_mul_and_equiv_fpos[0], coords_are_cartesian=False
                            )
                            tight_interstitial_struct = self.primitive_structure.copy()
                            tight_interstitial_struct.insert(
                                0, "H", tight_cand_site_mul_and_equiv_fpos[0], coords_are_cartesian=False
                            )
                            if structure_matcher.fit(interstitial_struct, tight_interstitial_struct):
                                matching_sites_mul_and_equiv_fpos += [tight_cand_site_mul_and_equiv_fpos]

                    # take the site with the lower multiplicity (higher symmetry), and most ideal site
                    # according to _frac_coords_sort_func if multiplicities equal:
                    output_sites_mul_and_equiv_fpos.append(
                        min(
                            [cand_site_mul_and_equiv_fpos, *matching_sites_mul_and_equiv_fpos],
                            key=lambda cand_site_mul_and_equiv_fpos: (
                                cand_site_mul_and_equiv_fpos[1],
                                # return the minimum _frac_coords_sort_func for all equiv fpos:
                                *_frac_coords_sort_func(
                                    sorted(cand_site_mul_and_equiv_fpos[2], key=_frac_coords_sort_func)[0]
                                ),
                            ),
                        )
                    )

                sorted_sites_mul_and_equiv_fpos = []
                for _cand_site, multiplicity, equiv_fpos in output_sites_mul_and_equiv_fpos:
                    # take site with equiv_fpos sorted by _frac_coords_sort_func:
                    sorted_equiv_fpos = sorted(equiv_fpos, key=_frac_coords_sort_func)
                    ideal_cand_site = sorted_equiv_fpos[0]
                    sorted_sites_mul_and_equiv_fpos.append(
                        (ideal_cand_site, multiplicity, sorted_equiv_fpos)
                    )

            self.defects["interstitials"] = []
            ig = InterstitialGenerator(self.interstitial_gen_kwargs.get("min_dist", 0.9))
            cand_sites, multiplicity, equiv_fpos = zip(*sorted_sites_mul_and_equiv_fpos)
            for el in self._element_list:
                inter_generator = ig.generate(
                    self.primitive_structure,
                    insertions={el: cand_sites},
                    multiplicities={el: multiplicity},
                    equivalent_positions={el: equiv_fpos},
                )
                self.defects["interstitials"].extend(
                    [Interstitial._from_pmg_defect(inter) for inter in inter_generator]
                )

                # check if any manually-specified interstitials were skipped due to min_dist and warn user:
                if self.interstitial_coords and len(self.interstitial_coords) > len(
                    self.defects["interstitials"]
                ):
                    warnings.warn(
                        f"\nNote that some manually-specified interstitial sites were skipped due to "
                        f"being too close to host lattice sites (minimum distance = `min_dist` = "
                        f"{self.interstitial_gen_kwargs.get('min_dist', 0.9):.2f} Å). If for some reason "
                        f"you still want to include these sites, you can adjust `min_dist` (default = 0.9 "
                        f"Å), or just use the default Voronoi tessellation algorithm for generating "
                        f"interstials (by not setting the `interstitial_coords` argument)."
                    )

            pbar.update(15)  # 45% of progress bar, generating interstitials typically takes the longest

            # Generate DefectEntry objects:
            pbar.set_description("Determining Wyckoff sites")
            defect_list: List[Defect] = sum(self.defects.values(), [])
            num_defects = len(defect_list)

            # get BCS conventional structure and lattice vector swap array:
            (
                self.conventional_structure,
                self._BilbaoCS_conv_cell_vector_mapping,
                wyckoff_label_dict,
            ) = get_BCS_conventional_structure(
                self.primitive_structure, pbar=pbar, return_wyckoff_dict=True
            )

            sga = _get_sga(self.conventional_structure)
            symm_ops = sga.get_symmetry_operations(cartesian=False)

            # process defects into defect entries:
            partial_func = partial(
                _get_neutral_defect_entry,
                supercell_matrix=self.supercell_matrix,
                target_frac_coords=self.target_frac_coords,
                bulk_supercell=self.bulk_supercell,
                conventional_structure=self.conventional_structure,
                _BilbaoCS_conv_cell_vector_mapping=self._BilbaoCS_conv_cell_vector_mapping,
                wyckoff_label_dict=wyckoff_label_dict,
                symm_ops=symm_ops,
            )

            if not isinstance(pbar, MagicMock):  # to allow tqdm to be mocked for testing
                _pbar_increment_per_defect = max(
                    0, min((1 / num_defects) * ((pbar.total * 0.9) - pbar.n), pbar.total - pbar.n)
                )  # up to 90% of progress bar
            else:
                _pbar_increment_per_defect = 0

            defect_entry_list = []
            if len(self.primitive_structure) > 8:  # skip for small systems as communication overhead /
                # process initialisation outweighs speedup
                with Pool(processes=processes or cpu_count() - 1) as pool:
                    results = pool.imap_unordered(partial_func, defect_list)
                    for result in results:
                        defect_entry_list.append(result)
                        pbar.update(_pbar_increment_per_defect)  # 90% of progress bar

            else:
                for defect in defect_list:
                    defect_entry_list.append(partial_func(defect))
                    pbar.update(_pbar_increment_per_defect)  # 90% of progress bar

            pbar.set_description("Generating DefectEntry objects")
            # sort defect_entry_list by _frac_coords_sort_func applied to the conv_cell_frac_coords,
            # in order for naming and defect generation output info to be deterministic
            defect_entry_list.sort(key=lambda x: _frac_coords_sort_func(x.conv_cell_frac_coords))

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

            named_defect_dict = name_defect_entries(defect_entry_list, element_list=self._element_list)
            pbar.update(5)  # 95% of progress bar
            if not isinstance(pbar, MagicMock):
                _pbar_increment_per_defect = max(
                    0, min((1 / num_defects) * (pbar.total - pbar.n) * 0.999, pbar.total - pbar.n)
                )  # multiply by 0.999 to avoid rounding errors, overshooting the 100% limit and getting
                # warnings from tqdm

            for defect_name_wout_charge, neutral_defect_entry in named_defect_dict.items():
                charge_state_guessing_output = guess_defect_charge_states(
                    neutral_defect_entry.defect, return_log=True, **self.charge_state_gen_kwargs
                )
                charge_state_guessing_output = cast(
                    Tuple[List[int], List[Dict]], charge_state_guessing_output
                )  # for correct type checking; guess_defect_charge_states can return different types
                # depending on return_log
                charge_states, charge_state_guessing_log = charge_state_guessing_output
                neutral_defect_entry.charge_state_guessing_log = charge_state_guessing_log

                for charge in charge_states:
                    defect_entry = copy.deepcopy(neutral_defect_entry)
                    defect_entry.charge_state = charge
                    # set name attribute:
                    defect_entry.name = f"{defect_name_wout_charge}_{'+' if charge > 0 else ''}{charge}"
                    self.defect_entries[defect_entry.name] = defect_entry

                pbar.update(_pbar_increment_per_defect)  # 100% of progress bar

            # sort defects and defect entries for deterministic behaviour:
            self.defects = _sort_defects(self.defects, element_list=self._element_list)
            self.defect_entries = _sort_defect_entries(
                self.defect_entries, element_list=self._element_list
            )

            # remove oxidation states from structures (causes deprecation warnings and issues with
            # comparison tests, also only added from oxi state guessing in defect generation so no extra
            # info provided)
            self.primitive_structure.remove_oxidation_states()

            for defect_list in self.defects.values():
                for defect_obj in defect_list:
                    defect_obj.structure.remove_oxidation_states()

            for defect_entry in self.defect_entries.values():
                defect_entry.defect.structure.remove_oxidation_states()

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
                    "Charge States",
                    "Conv. Cell Coords",
                    "Wyckoff",
                ]
                defect_type = defect_list[0].defect_type
                matching_defect_types = {
                    defect_entry_name: defect_entry
                    for defect_entry_name, defect_entry in self.defect_entries.items()
                    if defect_entry.defect.defect_type == defect_type
                }
                matching_type_names_wout_charge = [
                    defect_entry_name.rsplit("_", 1)[0]
                    for defect_entry_name in matching_defect_types
                    if defect_entry_name.endswith("_0")  # one neutral defect entry per defect
                ]
                for defect_name in matching_type_names_wout_charge:
                    charges = [
                        name.rsplit("_", 1)[1]
                        for name in self.defect_entries
                        if name.startswith(f"{defect_name}_")
                    ]  # so e.g. Te_i_m1 doesn't match with Te_i_m1b
                    # convert list of strings to one string with comma-separated charges
                    charges = "[" + ",".join(charges) + "]"
                    neutral_defect_entry = self.defect_entries[f"{defect_name}_0"]  # neutral, no +/- sign
                    frac_coords_string = ",".join(
                        f"{x:.3f}" for x in neutral_defect_entry.conv_cell_frac_coords
                    )
                    row = [
                        defect_name,
                        charges,
                        f"[{frac_coords_string}]",
                        neutral_defect_entry.wyckoff,
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
            "Note that Wyckoff letters can depend on the ordering of elements in the conventional "
            "standard structure, for which doped uses the spglib convention."
        )

        return info_string

    def add_charge_states(self, defect_entry_name: str, charge_states: list):
        """
        Add additional `DefectEntry`s with the specified charge states to
        `self.defect_entries`.

        Args:
            defect_entry_name (str):
                Name of defect entry to add charge states to.
                Doesn't need to include the charge state.
            charge_states (list): List of charge states to add to defect entry (e.g. [-2, -3]).
        """
        previous_defect_entry = [
            entry for name, entry in self.defect_entries.items() if name.startswith(defect_entry_name)
        ][0]
        for charge in charge_states:
            defect_entry = copy.deepcopy(previous_defect_entry)
            defect_entry.charge_state = charge
            defect_entry.name = (
                f"{defect_entry.name.rsplit('_', 1)[0]}_{'+' if charge > 0 else ''}{charge}"
            )
            self.defect_entries[defect_entry.name] = defect_entry

        # sort defects and defect entries for deterministic behaviour:
        self.defects = _sort_defects(self.defects, element_list=self._element_list)
        self.defect_entries = _sort_defect_entries(self.defect_entries, element_list=self._element_list)

    def remove_charge_states(self, defect_entry_name: str, charge_states: list):
        """
        Remove `DefectEntry`s with the specified charge states from
        `self.defect_entries`.

        Args:
            defect_entry_name (str):
                Name of defect entry to remove charge states from.
                Doesn't need to include the charge state.
            charge_states (list): List of charge states to add to defect entry (e.g. [-2, -3]).
        """
        # if defect entry name ends with number:
        if defect_entry_name[-1].isdigit():
            defect_entry_name = defect_entry_name.rsplit("_", 1)[0]  # name without charge

        for charge in charge_states:
            # remove defect entries with defect_entry_name in name:
            for defect_entry_name_to_remove in [
                name
                for name in self.defect_entries
                if name.startswith(defect_entry_name)
                and name.endswith(f"_{'+' if charge > 0 else ''}{charge}")
            ]:
                del self.defect_entries[defect_entry_name_to_remove]

        # sort defects and defect entries for deterministic behaviour:
        self.defects = _sort_defects(self.defects, element_list=self._element_list)
        self.defect_entries = _sort_defect_entries(self.defect_entries, element_list=self._element_list)

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
        Reconstructs DefectsGenerator object from a dict representation created
        using DefectsGenerator.as_dict().

        Args:
            d (dict): dict representation of DefectsGenerator.

        Returns:
            DefectsGenerator object
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
                        attributes["defect"] = copy.deepcopy(decode_dict(iterable["defect"]))
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

    def to_json(self, filename: Optional[str] = None):
        """
        Save the DefectsGenerator object as a json file, which can be reloaded
        with the DefectsGenerator.from_json() class method.

        Args:
            filename (str): Filename to save json file as. If None, the filename will be
                set as "{Chemical Formula}_defects_generator.json" where {Chemical Formula}
                is the chemical formula of the host material.
        """
        if filename is None:
            formula = self.primitive_structure.composition.get_reduced_formula_and_factor(
                iupac_ordering=True
            )[0]
            filename = f"{formula}_defects_generator.json"

        dumpfn(self, filename)

    @classmethod
    def from_json(cls, filename: str):
        """
        Load a DefectsGenerator object from a json file.

        Args:
            filename (str): Filename of json file to load DefectsGenerator
            object from.

        Returns:
            DefectsGenerator object
        """
        return loadfn(filename)

    def __getattr__(self, attr):
        """
        Redirects an unknown attribute/method call to the defect_entries
        dictionary attribute, if the attribute doesn't exist in
        DefectsGenerator.
        """
        # Return the attribute if it exists in self.__dict__
        if attr in self.__dict__:
            return self.__dict__[attr]

        # If trying to access defect_entries and it doesn't exist, raise an error
        if attr == "defect_entries" or "defect_entries" not in self.__dict__:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

        # Check if the attribute exists in defect_entries
        if hasattr(self.defect_entries, attr):
            return getattr(self.defect_entries, attr)

        # If all else fails, raise an AttributeError
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __getitem__(self, key):
        """
        Makes DefectsGenerator object subscriptable, so that it can be indexed
        like a dictionary, using the defect_entries dictionary attribute.
        """
        return self.defect_entries[key]

    def __setitem__(self, key, value):
        """
        Set the value of a specific key (defect name) in the defect_entries
        dictionary.

        Also adds the corresponding defect to the self.defects dictionary, if
        it doesn't already exist.
        """
        # check the input, must be a DefectEntry object, with same supercell and primitive structure
        if not isinstance(value, DefectEntry):
            raise TypeError(f"Value must be a DefectEntry object, not {type(value).__name__}")

        defect_struc_wout_oxi = value.defect.structure.copy()
        defect_struc_wout_oxi.remove_oxidation_states()

        if defect_struc_wout_oxi != self.primitive_structure:
            raise ValueError(
                f"Value must have the same primitive structure as the DefectsGenerator object, "
                f"instead has: {value.defect.structure} while DefectsGenerator has: "
                f"{self.primitive_structure}"
            )

        # check supercell
        defect_supercell = value.defect.get_supercell_structure(
            sc_mat=self.supercell_matrix,
            dummy_species=_dummy_species,  # keep track of the defect frac coords in the supercell
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
        self.defect_entries = _sort_defect_entries(self.defect_entries, element_list=self._element_list)

    def __delitem__(self, key):
        """
        Deletes the specified defect entry from the defect_entries dictionary.

        Doesn't remove the defect from the defects dictionary attribute, as
        there may be other charge states of the same defect still present.
        """
        del self.defect_entries[key]

    def __contains__(self, key):
        """
        Returns True if the defect_entries dictionary contains the specified
        defect name.
        """
        return key in self.defect_entries

    def __len__(self):
        """
        Returns the number of entries in the defect_entries dictionary.
        """
        return len(self.defect_entries)

    def __iter__(self):
        """
        Returns an iterator over the defect_entries dictionary.
        """
        return iter(self.defect_entries)

    def __str__(self):
        """
        Returns a string representation of the DefectsGenerator object.
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
        Returns a string representation of the DefectsGenerator object, and
        prints the DefectsGenerator info.
        """
        return (
            self.__str__()
            + "\n---------------------------------------------------------\n"
            + self._defect_generator_info()
        )


def _first_and_second_element(defect_name):
    """
    Return a tuple of the first and second element in the defect name.

    For sorting purposes.
    """
    if defect_name.startswith("v"):
        return (defect_name.split("_")[1], defect_name.split("_")[1])
    if defect_name.split("_")[1] == "i":
        return (defect_name.split("_")[0], defect_name.split("_")[0])

    return (
        defect_name.split("_")[0],
        defect_name.split("_")[1],
    )


def _sort_defect_entries(defect_entries_dict, element_list=None):
    """
    Sort defect entries for deterministic behaviour (for output and when
    reloading DefectsGenerator objects, and with DefectPhaseDiagram entries
    (particularly for deterministic plotting behaviour)).

    Sorts defect entries by defect type (vacancies, substitutions,
    interstitials), then by order of appearance of elements in the composition,
    then alphabetically, then (for defect entries of the same type) sort by
    charge state.
    """
    if element_list is None:
        host_element_list = None
        extrinsic_element_list = []
        for defect_entry in defect_entries_dict.values():
            if host_element_list is None:  # first iteration
                host_element_list = [
                    el.symbol for el in defect_entry.defect.structure.composition.elements
                ]
            extrinsic_element_list.extend(
                el.symbol
                for el in defect_entry.defect.defect_structure.composition.elements
                if el.symbol not in host_element_list
            )

        # sort extrinsic elements alphabetically for deterministic ordering in output:
        extrinsic_element_list = sorted(
            [el for el in extrinsic_element_list if el not in host_element_list]
        )
        element_list = host_element_list + extrinsic_element_list

    try:
        return dict(
            sorted(
                defect_entries_dict.items(),
                key=lambda s: (
                    s[1].defect.defect_type.value,
                    element_list.index(_first_and_second_element(s[0])[0]),
                    element_list.index(_first_and_second_element(s[0])[1]),
                    s[0].rsplit("_", 1)[0],  # name without charge
                    s[1].charge_state,  # charge state
                ),
            )
        )
    except ValueError as value_err:
        # possibly defect entries with names not in doped format, try sorting without using name:
        try:
            return dict(
                sorted(
                    defect_entries_dict.items(),
                    key=lambda s: (
                        s[1].defect.defect_type.value,
                        element_list.index(
                            _first_and_second_element(
                                get_defect_name_from_entry(
                                    s[1],
                                    element_list=element_list,
                                )
                            )[0]
                        ),
                        element_list.index(
                            _first_and_second_element(
                                get_defect_name_from_entry(
                                    s[1],
                                    element_list=element_list,
                                )
                            )[1]
                        ),
                        get_defect_name_from_entry(s[1], element_list=element_list),  # name without charge
                        s[1].charge_state,  # charge state
                    ),
                )
            )
        except ValueError:
            raise value_err


def _sort_defects(defects_dict, element_list=None):
    """
    Sort defect objects for deterministic behaviour (for output and when
    reloading DefectsGenerator objects.

    Sorts defects by defect type (vacancies, substitutions, interstitials),
    then by order of appearance of elements in the composition, then
    alphabetically, then according to _frac_coords_sort_func.
    """
    if element_list is None:
        all_elements = []
        host_element_list = None

        for _defect_type, defect_list in defects_dict.items():
            for defect in defect_list:
                if host_element_list is None:  # first iteration
                    host_element_list = [el.symbol for el in defect.structure.composition.elements]
                all_elements.extend(el.symbol for el in defect.defect_structure.composition.elements)
        extrinsic_element_list = list(set(all_elements) - set(host_element_list))

        # sort extrinsic elements alphabetically for deterministic ordering in output:
        extrinsic_element_list = sorted(
            [el for el in extrinsic_element_list if el not in host_element_list]
        )
        element_list = host_element_list + extrinsic_element_list

    return {
        defect_type: sorted(
            defect_list,
            key=lambda d: (
                element_list.index(_first_and_second_element(d.name)[0]),
                element_list.index(_first_and_second_element(d.name)[1]),
                d.name,  # bare name without charge
                _frac_coords_sort_func(d.conv_cell_frac_coords),
            ),
        )
        for defect_type, defect_list in defects_dict.items()
    }


def _get_interstitial_candidate_sites(args):
    """
    Return a list of cand_sites_mul_and_equiv_fpos for interstitials in the
    structure. Defined separately here to allow for multiprocessing.

    Args:
        args: tuple of arguments (to work with multiprocessing.pool)
            to be passed to the function, in the form:
                interstitial_generator: InterstitialGenerator object
                structure: Structure object
    """
    interstitial_generator, structure = args
    return [*interstitial_generator._get_candidate_sites(structure)]


# Schoenflies, Hermann-Mauguin, spgid dict: (Taken from the excellent Abipy with GNU GPL License)
_PTG_IDS = [
    ("C1", "1", 1),
    ("Ci", "-1", 2),
    ("C2", "2", 3),
    ("Cs", "m", 6),
    ("C2h", "2/m", 10),
    ("D2", "222", 16),
    ("C2v", "mm2", 25),
    ("D2h", "mmm", 47),
    ("C4", "4", 75),
    ("S4", "-4", 81),
    ("C4h", "4/m", 83),
    ("D4", "422", 89),
    ("C4v", "4mm", 99),
    ("D2d", "-42m", 111),
    ("D4h", "4/mmm", 123),
    ("C3", "3", 143),
    ("C3i", "-3", 147),
    ("D3", "32", 149),
    ("C3v", "3m", 156),
    ("D3d", "-3m", 162),
    ("C6", "6", 168),
    ("C3h", "-6", 174),
    ("C6h", "6/m", 175),
    ("D6", "622", 177),
    ("C6v", "6mm", 183),
    ("D3h", "-6m2", 189),
    ("D6h", "6/mmm", 191),
    ("T", "23", 195),
    ("Th", "m-3", 200),
    ("O", "432", 207),
    ("Td", "-43m", 215),
    ("Oh", "m-3m", 221),
]

_SCH2HERM = {t[0]: t[1] for t in _PTG_IDS}
_HERM2SCH = {t[1]: t[0] for t in _PTG_IDS}
_SPGID2SCH = {t[2]: t[0] for t in _PTG_IDS}
_SCH2SPGID = {t[0]: t[2] for t in _PTG_IDS}

sch_symbols = list(_SCH2HERM.keys())
