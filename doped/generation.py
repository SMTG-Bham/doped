"""
Code to generate Defect objects and supercell structures for ab-initio
calculations.
"""
import copy
import warnings
from itertools import chain
from typing import Dict, List, Optional, Type, Union

import numpy as np
from ase.spacegroup.wyckoff import Wyckoff
from monty.json import MontyDecoder
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.defects.core import Defect, DefectType
from pymatgen.analysis.defects.generators import (
    AntiSiteGenerator,
    InterstitialGenerator,
    SubstitutionGenerator,
    VacancyGenerator,
    VoronoiInterstitialGenerator,
)
from pymatgen.analysis.defects.supercells import get_sc_fromstruct
from pymatgen.analysis.defects.thermo import DefectEntry
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core.composition import Composition, Element
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import DummySpecies
from pymatgen.core.structure import PeriodicSite, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.advanced_transformations import _proj
from pymatgen.transformations.standard_transformations import SupercellTransformation
from sympy import Eq, simplify, solve, symbols
from tabulate import tabulate
from tqdm import tqdm

# TODO: For specifying interstitial sites, will want to be able to specify as either primitive or
#  supercell coords in this case, so will need functions for transforming between primitive and
#  supercell defect structures (will want this for defect parsing as well). Defectivator has functions
#  that do some of this. This will be tricky (SSX trickay you might say) for relaxed interstitials ->
#  get symmetry-equivalent positions of relaxed interstitial position in unrelaxed bulk (easy pal,
#  tf you mean 'tricky'??)
# TODO: Add multiprocessing for Wyckoff determination and interstitial split generation

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
    Generate DefectEntry object from a Defect object.

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


def _round_floats(obj):
    """
    Recursively round floats in a dictionary to 5 decimal places.
    """
    if isinstance(obj, float):
        return round(obj, 5) + 0.0
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(x) for x in obj]
    return obj


def _defect_dict_key_from_pmg_type(defect_type: DefectType) -> str:
    """
    Returns the corresponding defect dictionary key for a pymatgen Defect
    object.
    """
    if defect_type == DefectType.Vacancy:
        return "vacancies"
    if defect_type == DefectType.Substitution:
        return "substitutions"
    if defect_type == DefectType.Interstitial:
        return "interstitials"
    if defect_type == DefectType.Other:
        return "others"

    raise ValueError(
        f"Defect type {defect_type} not recognised. Must be one of {DefectType.Vacancy}, "
        f"{DefectType.Substitution}, {DefectType.Interstitial}, {DefectType.Other}."
    )


def closest_site_info(defect_entry, n=1, r=5):
    """
    Return the element and distance (rounded to 2 decimal places) of the
    closest site to defect_entry.sc_defect_frac_coords in
    defect_entry.sc_entry.structure, with distance > 0.01 (i.e. so not the site
    itself), and if there are multiple elements with the same distance, sort
    alphabetically and return the first one.

    If n is set, then it returns the nth closest site, where the nth site must be at least
    0.02 Å further away than the n-1th site.
    """
    site_distances = sorted(
        [
            (
                site.distance_and_image_from_frac_coords(defect_entry.sc_defect_frac_coords)[0],
                site,
            )
            for site in defect_entry.sc_entry.structure.sites
            if site.distance_and_image_from_frac_coords(defect_entry.sc_defect_frac_coords)[0] > 0.01
        ],
        key=lambda x: (round(x[0], 2), x[1].specie.symbol),
    )

    # prune site_distances to remove any tuples with distances <0.02 Å greater than the previous
    # entry:
    site_distances = [
        site_distances[i]
        for i in range(len(site_distances))
        if i == 0
        or site_distances[i][0] - site_distances[i - 1][0] > 0.02
        or site_distances[i][1].specie.symbol != site_distances[i - 1][1].specie.symbol
    ]

    min_distance, closest_site = site_distances[n - 1]

    return closest_site.specie.symbol + f"{min_distance:.2f}"


def get_defect_name_from_entry(defect_entry):
    """
    Get the doped/SnB defect name from DefectEntry object.
    """
    defect_diagonal_supercell = defect_entry.defect.get_supercell_structure(
        sc_mat=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        dummy_species="X",
    )  # create defect supercell, which is a diagonal expansion of the unit cell so that the defect
    # periodic image retains the unit cell symmetry, in order not to affect the point group symmetry
    sga = SpacegroupAnalyzer(defect_diagonal_supercell, symprec=1e-2)
    return (
        f"{defect_entry.defect.name}_{herm2sch(sga.get_point_group_symbol())}"
        f"_{closest_site_info(defect_entry)}"
    )


def name_defect_entries(defect_entries):
    """
    Create a dictionary of {Name: DefectEntry} from a list of DefectEntry
    objects, where the names are set according to the default doped algorithm;
    which is to use the pymatgen defect name (e.g. v_Cd, Cd_Te etc.) for
    vacancies/antisites/substitutions, unless there are multiple inequivalent
    sites for the defect, in which case the point group of the defect site is
    appended (e.g. v_Cd_Td, Cd_Te_Td etc.), and if this is still not unique,
    then element identity and distance to the nearest neighbour of the defect
    site is appended (e.g. v_Cd_Td_Te2.83, Cd_Te_Td_Cd2.83 etc.).

    For interstitials, the same naming scheme is used, but the point group is
    always appended to the pymatgen defect name.
    """

    def get_shorter_name(full_defect_name, split_number):
        if split_number < 1:  # if split number is less than 1, return full name
            return full_defect_name
        return full_defect_name.rsplit("_", split_number)[0]

    def get_matching_names(defect_naming_dict, defect_name):
        return [name for name in defect_naming_dict if defect_name in name]

    def handle_unique_match(defect_naming_dict, matching_names, split_number):
        if len(matching_names) == 1:
            previous_entry = defect_naming_dict.pop(matching_names[0])
            previous_entry_full_name = get_defect_name_from_entry(previous_entry)
            previous_entry_name = get_shorter_name(previous_entry_full_name, split_number - 1)
            defect_naming_dict[previous_entry_name] = previous_entry

        return defect_naming_dict

    def append_closest_site_info(name, entry, n):
        return name + closest_site_info(entry, n=n)

    def handle_multiple_matches(defect_naming_dict, full_defect_name, defect_entry):
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

            if not any(name for name in defect_naming_dict if full_defect_name in name):
                return defect_naming_dict, full_defect_name

            if n == 3:  # if still not unique after 3rd nearest neighbour, just use alphabetical indexing
                return handle_repeated_name(defect_naming_dict, full_defect_name)
            n += 1

    def handle_repeated_name(defect_naming_dict, full_defect_name):
        for name in list(defect_naming_dict.keys()):
            if full_defect_name == name:
                prev_defect_entry = defect_naming_dict.pop(name)
                defect_naming_dict[name + "a"] = prev_defect_entry
                defect_name = full_defect_name + "b"
                break
            if full_defect_name == name[:-1]:
                last_letters = [name[-1] for name in defect_naming_dict if name[:-1] == full_defect_name]
                last_letters.sort()
                new_letter = chr(ord(last_letters[-1]) + 1)
                defect_name = full_defect_name + new_letter
                break
        return defect_naming_dict, defect_name

    defect_naming_dict = {}
    for defect_entry in defect_entries:
        full_defect_name = get_defect_name_from_entry(defect_entry)
        split_number = 1 if defect_entry.defect.defect_type == DefectType.Interstitial else 2
        shorter_defect_name = get_shorter_name(full_defect_name, split_number)
        if not any(name for name in defect_naming_dict if shorter_defect_name in name):
            defect_naming_dict[shorter_defect_name] = defect_entry
            continue

        matching_shorter_names = get_matching_names(defect_naming_dict, shorter_defect_name)
        defect_naming_dict = handle_unique_match(defect_naming_dict, matching_shorter_names, split_number)
        shorter_defect_name = get_shorter_name(full_defect_name, split_number - 1)
        if not any(name for name in defect_naming_dict if shorter_defect_name in name):
            defect_naming_dict[shorter_defect_name] = defect_entry
            continue

        matching_shorter_names = get_matching_names(defect_naming_dict, shorter_defect_name)
        defect_naming_dict = handle_unique_match(
            defect_naming_dict, matching_shorter_names, split_number - 1
        )
        shorter_defect_name = get_shorter_name(full_defect_name, split_number - 2)
        if not any(name for name in defect_naming_dict if shorter_defect_name in name):
            defect_naming_dict[shorter_defect_name] = defect_entry
            continue

        defect_naming_dict, full_defect_name = handle_multiple_matches(
            defect_naming_dict, full_defect_name, defect_entry
        )
        defect_naming_dict[full_defect_name] = defect_entry
    return defect_naming_dict


def herm2sch(herm_symbol):
    """
    Convert from Hermann-Mauguin to Schoenflies.
    """
    return _HERM2SCH.get(herm_symbol, None)


def get_wyckoff_dict_from_sgn(sgn):
    """
    Get dictionary of {Wyckoff label: coordinates} for a given space group
    number.
    """
    wyckoff = Wyckoff(sgn).wyckoff
    wyckoff_label_coords_dict = {}

    def _coord_string_to_array(coord_string):
        # Split string into substrings, parse each as a sympy expression,
        # then convert to list of sympy expressions
        return [simplify(x.replace("2x", "2*x")) for x in coord_string.split(",")]

    for element in wyckoff["letters"]:
        label = wyckoff[element]["multiplicity"] + element  # e.g. 4d
        wyckoff_coords = [_coord_string_to_array(coords) for coords in wyckoff[element]["coordinates"]]
        wyckoff_label_coords_dict[label] = wyckoff_coords

        equivalent_sites = [
            _coord_string_to_array(coords) for coords in wyckoff.get("equivalent_sites", [])
        ]

        new_coords = []  # new list for equivalent coordinates

        for coord_array in wyckoff_coords:
            for equivalent_site in equivalent_sites:
                # add coord_array and equivalent_site element-wise
                equiv_coord_array = coord_array.copy()
                equiv_coord_array = equiv_coord_array + np.array(equivalent_site)
                new_coords.append(equiv_coord_array)

        # add new_coords to wyckoff_label_coords:
        wyckoff_label_coords_dict[label].extend(new_coords)
    return wyckoff_label_coords_dict


def _frac_coords_sort_func(coords):
    """
    Sorting function to apply on an iterable of fractional coordinates, where
    entries are sorted by the number of x, y, z that are (almost) equal (i.e.
    between 0 and 3), then by the magnitude of x+y+z, then by the magnitudes of
    x, y and z.
    """
    num_equals = sum(
        np.isclose(coords[i], coords[j], atol=1e-3)
        for i in range(len(coords))
        for j in range(i + 1, len(coords))
    )
    magnitude = round(sum(np.abs(coords)), 4)
    return (-num_equals, magnitude, *np.abs(np.round(coords, 4)))


def get_primitive_structure(sga):
    """
    Get a consistent/deterministic primitive structure from a
    SpacegroupAnalyzer object.

    For some materials (e.g. zinc blende), there are multiple equivalent
    primitive cells, so for reproducibility and in line with most structure
    conventions/definitions, take the one with the lowest summed norm of the
    fractional coordinates of the sites (i.e. favour Cd (0,0,0) and Te
    (0.25,0.25,0.25) over Cd (0,0,0) and Te (0.75,0.75,0.75) for F-43m CdTe).
    """
    possible_prim_structs = []
    for _i in range(10):
        struct = sga.get_primitive_standard_structure()
        possible_prim_structs.append(struct)
        sga = SpacegroupAnalyzer(struct, symprec=1e-2)

    possible_prim_structs = sorted(possible_prim_structs, key=lambda x: np.sum(x.frac_coords))
    return Structure.from_dict(_round_floats(possible_prim_structs[0].as_dict()))


def get_conv_cell_site(defect_entry, lattice_vec_swap_array=None):
    """
    Get an equivalent site of the defect entry in the conventional structure of
    the host material.
    """
    if lattice_vec_swap_array is None:
        lattice_vec_swap_array = [0, 1, 2]

    sga = SpacegroupAnalyzer(defect_entry.defect.structure)
    conv_to_prim_transf_matrix = sga.get_conventional_to_primitive_transformation_matrix()
    prim_struct_with_X = defect_entry.defect.structure.copy()
    prim_struct_with_X.remove_oxidation_states()
    prim_struct_with_X.append("X", defect_entry.defect.site.frac_coords, coords_are_cartesian=False)

    # convert to sga prim structure:
    sm = StructureMatcher(primitive_cell=False, ignored_species=["X"], comparator=ElementComparator())
    s2_like_s1 = sm.get_s2_like_s1(get_primitive_structure(sga), prim_struct_with_X)
    # sometimes this get_s2_like_s1 doesn't work properly due to different (but equivalent) lattice vectors
    # (e.g. a=(010) instead of (100) etc.), so do this to be sure:
    s2_really_like_s1 = Structure.from_sites(
        [
            PeriodicSite(
                site.specie,
                site.frac_coords,
                sga.get_primitive_standard_structure().lattice,
                to_unit_cell=True,
            )
            for site in s2_like_s1.sites
        ]
    )
    regenerated_conv_structure = s2_really_like_s1 * np.linalg.inv(conv_to_prim_transf_matrix)
    reorientated_regenerated_conv_structure = swap_axes(regenerated_conv_structure, lattice_vec_swap_array)

    defect_conv_cell_sites = [
        site for site in reorientated_regenerated_conv_structure.sites if site.specie.symbol == "X"
    ]

    defect_conv_cell_sites.sort(key=lambda site: _frac_coords_sort_func(site.frac_coords))
    conv_cell_site = defect_conv_cell_sites[0].to_unit_cell()
    conv_cell_site.frac_coords = np.round(conv_cell_site.frac_coords, 5)

    return conv_cell_site


def get_wyckoff_label_and_equiv_coord_list(
    defect_entry=None, conv_cell_site=None, wyckoff_dict=None, lattice_vec_swap_array=None
):
    """
    Return the Wyckoff label and list of equivalent fractional coordinates
    within the conventional cell for the input defect_entry or conv_cell_site
    (whichever is provided, defaults to defect_entry if both), given a
    dictionary of Wyckoff labels and coordinates (`wyckoff_dict`).

    If `wyckoff_dict` is not provided, the spacegroup of the bulk structure is
    determined and used to generate it with `get_wyckoff_dict_from_sgn()`.
    """
    if lattice_vec_swap_array is None:
        lattice_vec_swap_array = [0, 1, 2]
    if wyckoff_dict is None:
        if defect_entry is None:
            raise ValueError(
                "Must provide a `defect_entry` (not `conv_cell_site`) if `wyckoff_dict` is not provided."
            )
        sga = SpacegroupAnalyzer(defect_entry.defect.structure)  # primitive unit cell
        wyckoff_dict = get_wyckoff_dict_from_sgn(sga.get_space_group_number())

    def _compare_arrays(coord_list, coord_array):
        """
        Compare a list of arrays of sympy expressions (`coord_list`) with an
        array of coordinates (`coord_array`).

        Returns the matching array from the list.
        """
        x, y, z = symbols("x y z")
        variable_dicts = [{}]  # list of dicts for x,y,z

        for sympy_array in coord_list:
            match, variable_dict = evaluate_expression_and_update_dict(
                sympy_array, coord_array, variable_dicts
            )

            if match:
                # return coord list with sympy expressions subbed with variable_dict:
                return [
                    np.array(
                        [
                            np.mod(float(simplify(sympy_expr).subs(variable_dict)), 1)
                            for sympy_expr in sympy_array
                        ]
                    )
                    for sympy_array in coord_list
                ]

        return None  # No match found

    # get match of coords in wyckoff_label_coords to defect site coords:
    def find_closest_match(defect_site, wyckoff_label_coords_dict):
        for label, coord_list in wyckoff_label_coords_dict.items():
            subbed_coord_list = _compare_arrays(coord_list, np.array(defect_site.frac_coords))
            if subbed_coord_list is not None:
                return label, subbed_coord_list

        return None  # No match found

    def evaluate_expression(sympy_expr, coord, variable_dict):
        equation = Eq(sympy_expr, coord)
        variable = list(sympy_expr.free_symbols)[0]
        variable_dict[variable] = solve(equation, variable)[0]

        return simplify(sympy_expr).subs(variable_dict)

    def add_new_variable_dict(
        sympy_expr_prepend, sympy_expr, coord, current_variable_dict, variable_dicts
    ):
        new_sympy_expr = simplify(sympy_expr_prepend + str(sympy_expr))
        new_dict = current_variable_dict.copy()
        evaluate_expression(new_sympy_expr, coord, new_dict)  # solve for new variable
        if new_dict not in variable_dicts:
            variable_dicts.append(new_dict)

    def evaluate_expression_and_update_dict(sympy_array, coord_array, variable_dicts):
        match = False

        for variable_dict in variable_dicts:
            temp_dict = variable_dict.copy()
            match = True

            # sort zipped arrays by number of variables in sympy expression:
            coord_array, sympy_array = zip(
                *sorted(zip(coord_array, sympy_array), key=lambda x: len(x[1].free_symbols))
            )

            for coord, sympy_expr in zip(coord_array, sympy_array):
                # Evaluate the expression with the current variable_dict
                expr_value = simplify(sympy_expr).subs(temp_dict)

                # If the expression cannot be evaluated to a float
                # it means that there is a new variable in the expression
                try:
                    expr_value = np.mod(float(expr_value), 1)  # wrap to 0-1 (i.e. to unit cell)

                except TypeError:
                    # Assign the expression the value of the corresponding coordinate, and solve
                    # for the new variable
                    # first, special cases with two possible solutions due to PBC:
                    if sympy_expr == simplify("-2*x"):
                        add_new_variable_dict("1+", sympy_expr, coord, temp_dict, variable_dicts)
                    elif sympy_expr == simplify("2*x"):
                        add_new_variable_dict("-1+", sympy_expr, coord, temp_dict, variable_dicts)

                    expr_value = evaluate_expression(
                        sympy_expr, coord, temp_dict
                    )  # solve for new variable

                # Check if the evaluated expression matches the corresponding coordinate
                if not np.isclose(
                    np.mod(float(coord), 1),  # wrap to 0-1 (i.e. to unit cell)
                    np.mod(float(expr_value), 1),
                    atol=0.003,
                ) and not np.isclose(
                    np.mod(float(coord), 1) - 1,  # wrap to 0-1 (i.e. to unit cell)
                    np.mod(float(expr_value), 1),
                    atol=0.003,
                ):
                    match = False
                    break

            if match:
                break

        return match, temp_dict

    if defect_entry is not None:
        defect_entry.defect.site.to_unit_cell()  # ensure wrapped to unit cell

        # convert defect site to conventional unit cell for Wyckoff label matching:
        conv_cell_site = get_conv_cell_site(defect_entry, lattice_vec_swap_array)

    label, equiv_coord_list = find_closest_match(conv_cell_site, wyckoff_dict)

    if defect_entry is not None:
        # need to transform subbed_coord_list back to original (unswapped) cell:
        reoriented_coord_list = []
        for coords in equiv_coord_list:
            reoriented_coord_list.append([coords[i] for i in lattice_vec_swap_array])

    else:
        reoriented_coord_list = equiv_coord_list

    return label, reoriented_coord_list


def _compare_wyckoffs(wyckoff_symbols, conv_struct, wyckoff_dict, lattice_vec_swap_array):
    """
    Compare the Wyckoff labels of a conventional structure to a list of Wyckoff
    labels.
    """

    def _multiply_wyckoffs(wyckoff_labels, n=2):
        return [str(n * int(wyckoff[:-1])) + wyckoff[-1] for wyckoff in wyckoff_labels]

    wyckoff_symbol_lists = [_multiply_wyckoffs(wyckoff_symbols, n=n) for n in range(1, 5)]  # up to 4x
    doped_wyckoffs = []

    for site in conv_struct:
        wyckoff_label, equiv_coords = get_wyckoff_label_and_equiv_coord_list(
            conv_cell_site=site, wyckoff_dict=wyckoff_dict, lattice_vec_swap_array=lattice_vec_swap_array
        )
        if not any(
            # allow for sga conventional cell (and thus wyckoffs) being a multiple of BCS conventional cell
            wyckoff_label in wyckoff_symbol_list
            for wyckoff_symbol_list in wyckoff_symbol_lists
        ) and not any(
            # allow for BCS conv cell (and thus wyckoffs) being a multiple of sga conv cell (allow it fam)
            multiplied_wyckoff_symbol in wyckoff_symbols
            for multiplied_wyckoff_symbol in [
                _multiply_wyckoffs([wyckoff_label], n=n)[0] for n in range(1, 5)  # up to 4x
            ]
        ):
            return False  # break on first non-match
        doped_wyckoffs.append(wyckoff_label)

    if not any(
        # allow for sga conventional cell (and thus wyckoffs) being a multiple of BCS conventional cell
        set(i) == set(doped_wyckoffs)
        for i in wyckoff_symbol_lists
    ) and not any(
        set(i) == set(wyckoff_symbols)
        for i in [
            # allow for BCS conv cell (and thus wyckoffs) being a multiple of sga conv cell (allow it fam)
            _multiply_wyckoffs(doped_wyckoffs, n=n)
            for n in range(1, 5)  # up to 4x
        ]
    ):
        return False

    return True


def swap_axes(structure, axes):
    """
    Swap axes of the given structure.

    The new order of the axes is given by the axes parameter. For example,
    axes=(2, 1, 0) will swap the first and third axes.
    """
    transformation_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for i, axis in enumerate(axes):
        transformation_matrix[i][axis] = 1

    transformation = SupercellTransformation(transformation_matrix)

    return transformation.apply_transformation(structure)


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
    charge_state: int, oxi_state: int, oxi_probability: float, max_host_oxi_magnitude: int
) -> float:
    """
    Function to estimate the probability of a given defect charge state, using
    the probability of the corresponding defect element oxidation state, the
    magnitude of the charge state, and the maximum magnitude of the host
    oxidation states (i.e. how 'charged' the host is).

    Disfavours large (absolute) charge states, low probability oxidation
    states and greater charge/oxidation state magnitudes than that of the host.

    Args:
        charge_state (int): Charge state of defect.
        oxi_state (int): Oxidation state of defect element.
        oxi_probability (float): Probability of oxidation state of defect element.
        max_host_oxi_magnitude (int): Maximum host oxidation state magnitude.

    Returns:
        float: Probability of the defect charge state (between 0 and 1).
    """
    # for defect charge states; 0 to +/-2 likely, 3-4 less likely, 5-6 v unlikely, >=7 unheard of

    # incorporates oxidation state probabilities and overall defect site charge
    # (and thus by proxy the cationic/anionic identity of the substituting site)
    # Thought about incorporating the oxi state probabilities (i.e. reducibility/oxidisability)
    # of the neighbouring elements to this (particularly for vacancies), but no clear-cut
    # cases where this would actually improve performance

    if charge_state == 0:
        return 1

    def _defect_vs_host_charge(charge_state: int, host_charge: int) -> float:
        if abs(charge_state) <= abs(host_charge):
            return 1

        return 1 / (2 * (abs(charge_state) - abs(host_charge)))

    return oxi_probability * (
        1
        / abs(charge_state)
        * _defect_vs_host_charge(charge_state, max_host_oxi_magnitude)
        * _defect_vs_host_charge(oxi_state, max_host_oxi_magnitude)
    ) ** (2 / 3)


def _get_vacancy_charge_states(defect: Defect, padding: int = 1) -> List[int]:
    """
    Get the possible charge states for a vacancy defect, which is from +/-1 to
    the vacancy oxidation state.

    Args:
        defect (Defect): A `pymatgen` vacancy Defect object.
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
        defect (Defect): A `pymatgen` Defect object.

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
) -> Dict:
    return {
        oxi - orig_oxi: _charge_state_probability(oxi - orig_oxi, oxi, oxi_prob, max_host_oxi_magnitude)
        for oxi, oxi_prob in possible_oxi_states.items()
    }


def guess_defect_charge_states(
    defect: Defect, probability_threshold: float = 0.01, padding: int = 1
) -> list:
    """
    Guess the possible charge states of a defect.

    Args:
        defect (Defect): `pymatgen` Defect object.
        probability_threshold (float):
            Probability threshold for including defect charge states
            (for substitutions and interstitials). Default is 0.01.
        padding (int):
            Padding for vacancy charge states, such that the vacancy
            charge states are set to range(vacancy oxi state, padding),
            if vacancy oxidation state is negative, or to
            range(-padding, vacancy oxi state), if positive.
            Default is 1.

    Returns:
        list: List of defect charge states.
    """
    if defect.defect_type == DefectType.Vacancy:
        # Set defect charge state: from +/-1 to defect oxi state
        return _get_vacancy_charge_states(defect, padding=padding)

    possible_oxi_states = _get_possible_oxi_states(defect)
    max_host_oxi_magnitude = max([abs(site.specie.oxi_state) for site in defect.structure])
    if defect.defect_type == DefectType.Substitution:
        orig_oxi = defect.structure[defect.defect_site_index].specie.oxi_state
        possible_charge_states = _get_charge_states(possible_oxi_states, orig_oxi, max_host_oxi_magnitude)
    else:  # interstitial
        orig_oxi = 0
        possible_charge_states = _get_charge_states(possible_oxi_states, orig_oxi, max_host_oxi_magnitude)

    charge_state_list = [k for k, v in possible_charge_states.items() if v > probability_threshold]
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
        defect.defect_type == DefectType.Substitution
        and defect_elt_oxi_in_struct is not None
        and defect_elt_oxi_in_struct - orig_oxi
        not in range(charge_state_range[0], charge_state_range[1] + 1)
    ):
        # if simple antisite oxidation state difference not included, recheck with bumped up oxi_state
        # probability for the oxi_state of the substitution atom in the structure
        # should really be included unless it gives an absolute charge state >= 5, so set oxi_state
        # probability to 100%
        possible_charge_states[defect_elt_oxi_in_struct - orig_oxi] = _charge_state_probability(
            defect_elt_oxi_in_struct - orig_oxi, defect_elt_oxi_in_struct, 1, max_host_oxi_magnitude
        )

    if (
        defect.defect_type == DefectType.Interstitial
        and defect_elt_oxi_in_struct is not None
        and defect_elt_oxi_in_struct not in range(charge_state_range[0], charge_state_range[1] + 1)
    ):
        # if oxidation state of interstitial element in the host structure is not included, include it!
        possible_charge_states[defect_elt_oxi_in_struct] = _charge_state_probability(
            defect_elt_oxi_in_struct - orig_oxi, defect_elt_oxi_in_struct, 1, max_host_oxi_magnitude
        )

    sorted_charge_state_dict = dict(
        sorted(possible_charge_states.items(), key=lambda x: x[1], reverse=True)
    )

    charge_state_list = [k for k, v in sorted_charge_state_dict.items() if v > probability_threshold]
    if charge_state_list:
        charge_state_range = (int(min(charge_state_list)), int(max(charge_state_list)))
    else:
        # if no charge states are included, take most probable (if probability > 0.1*threshold)
        charge_state_list = [
            k for k, v in sorted_charge_state_dict.items() if v > 0.1 * probability_threshold
        ]

        most_likely_charge_state = charge_state_list[0] if charge_state_list else 0

        charge_state_range = (most_likely_charge_state, most_likely_charge_state)

    if (
        defect.defect_type == DefectType.Substitution
        and defect_elt_oxi_in_struct is not None
        and defect_elt_oxi_in_struct - orig_oxi == 0
        and (charge_state_range[0] >= 0 or charge_state_range[1] <= 0)
    ):
        # if defect is an antisite of two equal oxi state elements, ensure at least (-1, 0, +1) included
        charge_state_range = (min(charge_state_range[0], -1), max(charge_state_range[1], 1))

    # set charge_state_range to min/max of range, ensuring range is extended to 0:
    if charge_state_range[0] == 0 and charge_state_range[1] == 0:  # if range is 0
        charge_state_range = (-1, 1)

    charge_state_range = (min(charge_state_range[0], 0), max(charge_state_range[1], 0))
    return list(range(charge_state_range[0], charge_state_range[1] + 1))


class DefectsGenerator:
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
    ):
        """
        Generates pymatgen DefectEntry objects for defects in the input host
        structure. By default, generates all intrinsic defects, but extrinsic
        defects (impurities) can also be created using the `extrinsic`
        argument.

        Interstitial sites are generated using Voronoi tessellation by default (found
        to be the most reliable), which can be controlled using the
        `interstitial_gen_kwargs` argument (passed as keyword arguments to the
        `VoronoiInterstitialGenerator` class). Alternatively, a list of interstitial
        sites can be manually specified using the `interstitial_coords` argument.

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
                List of fractional coordinates (in the primitive cell) to use as
                interstitial defect sites. Default (when interstitial_coords not
                specified) is to automatically generate interstitial sites using
                Voronoi tessellation.
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

        Attributes:
            defects (Dict): Dictionary of {defect_type: [Defect, ...]} for all defect
                objects generated.
            defect_entries (Dict): Dictionary of {defect_name: DefectEntry} for all
                defect entries (with charge state and supercell properties) generated.
            primitive_structure (Structure): Primitive cell structure of the host
                used to generate defects.
            supercell_matrix (Matrix): Matrix to generate defect/bulk supercells from
                the primitive cell structure.
            bulk_supercell (Structure): Supercell structure of the host
                (equal to primitive_structure * supercell_matrix).

            `DefectsGenerator` input parameters are also set as attributes.
        """
        self.defects = {}  # {defect_type: [Defect, ...]}
        self.defect_entries = {}  # {defect_name: DefectEntry}
        self.structure = structure
        self.extrinsic = extrinsic if extrinsic is not None else []
        self.interstitial_coords = interstitial_coords if interstitial_coords is not None else []
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

        pbar = tqdm(total=100)  # tqdm progress bar. 100% is completion
        pbar.set_description("Getting primitive structure")

        try:  # put code in try/except block so progress bar always closed if interrupted
            # Reduce structure to primitive cell for efficient defect generation
            # same symprec as defect generators in pymatgen-analysis-defects:
            sga = SpacegroupAnalyzer(self.structure, symprec=1e-2)

            prim_struct = get_primitive_structure(sga)
            self.conventional_structure = Structure.from_dict(
                _round_floats(sga.get_conventional_standard_structure().as_dict())
            )
            if prim_struct.num_sites < self.structure.num_sites:
                primitive_structure = Structure.from_dict(_round_floats(prim_struct.as_dict()))

            else:  # primitive cell is the same as input structure, so use input structure to avoid
                # rotations
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
                min_atoms=self.supercell_gen_kwargs.get("min_atoms", 50),
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

            def _rotate_and_get_supercell_matrix(prim_struct, target_struct):
                """
                Rotates the input prim_struct to match the target_struct
                orientation, and returns the supercell matrix to convert from
                the rotated prim_struct to the target_struct.
                """
                # first rotate primitive structure to match target structure:
                mapping = prim_struct.lattice.find_mapping(target_struct.lattice)
                rotation_matrix = mapping[1]
                if np.allclose(rotation_matrix, -1 * np.eye(3)):
                    # pymatgen sometimes gives a rotation matrix of -1 * identity matrix, which is
                    # equivalent to no rotation. Just use the identity matrix instead.
                    rotation_matrix = np.eye(3)
                    supercell_matrix = -1 * mapping[2]
                else:
                    supercell_matrix = mapping[2]
                rotation_symmop = SymmOp.from_rotation_and_translation(
                    rotation_matrix=rotation_matrix.T
                )  # Transpose = inverse of rotation matrices (orthogonal matrices), better numerical
                # stability
                output_prim_struct = prim_struct.copy()
                output_prim_struct.apply_operation(rotation_symmop)
                clean_prim_struct_dict = _round_floats(output_prim_struct.as_dict())
                return Structure.from_dict(clean_prim_struct_dict), supercell_matrix

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
            self.defects["vacancies"] = list(vac_generator)
            pbar.update(5)  # 20% of progress bar

            # Antisites:
            pbar.set_description("Generating substitutions")
            antisite_generator_obj = AntiSiteGenerator()
            as_generator = antisite_generator_obj.generate(self.primitive_structure)
            self.defects["substitutions"] = list(as_generator)
            pbar.update(5)  # 25% of progress bar

            # Substitutions:
            substitution_generator_obj = SubstitutionGenerator()
            if isinstance(self.extrinsic, str):  # substitute all host elements
                substitutions = {
                    el.symbol: [self.extrinsic] for el in self.primitive_structure.composition.elements
                }
            elif isinstance(self.extrinsic, list):  # substitute all host elements
                substitutions = {
                    el.symbol: self.extrinsic for el in self.primitive_structure.composition.elements
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
                if "substitutions" in self.defects:
                    self.defects["substitutions"].extend(list(sub_generator))
                else:
                    self.defects["substitutions"] = list(sub_generator)
            if self.defects["substitutions"] == []:  # no substitutions, single-elt system, no extrinsic
                del self.defects["substitutions"]  # remove empty list
            pbar.update(5)  # 30% of progress bar

            # Interstitials:
            pbar.set_description("Generating interstitials")
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

            if self.interstitial_coords:
                # For the moment, this assumes interstitial_sites
                insertions = {
                    el.symbol: self.interstitial_coords
                    for el in self.primitive_structure.composition.elements
                }
                insertions.update({el: self.interstitial_coords for el in extrinsic_elements})
                interstitial_generator_obj = InterstitialGenerator(**self.interstitial_gen_kwargs)
                interstitial_generator = interstitial_generator_obj.generate(
                    self.primitive_structure, insertions=insertions
                )
                self.defects["interstitials"] = list(interstitial_generator)

            else:
                # Generate interstitial sites using Voronoi tessellation
                vig = VoronoiInterstitialGenerator(**self.interstitial_gen_kwargs)
                tight_vig = VoronoiInterstitialGenerator(stol=0.01)  # for determining multiplicities of
                # any merged/grouped interstitial sites from Voronoi tessellation + structure-matching
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

                    # take the site and multiplicity with the lower multiplicity
                    output_sites_mul_and_equiv_fpos.append(
                        min(
                            [cand_site_mul_and_equiv_fpos, *matching_sites_mul_and_equiv_fpos],
                            key=lambda cand_site_mul_and_equiv_fpos: cand_site_mul_and_equiv_fpos[1],
                        )
                    )

                self.defects["interstitials"] = []
                ig = InterstitialGenerator(
                    self.interstitial_gen_kwargs.get("min_dist", 0.9),
                )  # pmg defects default
                for species in [
                    el.symbol for el in self.primitive_structure.composition.elements
                ] + extrinsic_elements:
                    cand_sites, multiplicity, equiv_fpos = zip(*output_sites_mul_and_equiv_fpos)

                    self.defects["interstitials"].extend(
                        ig.generate(
                            self.primitive_structure,
                            insertions={species: cand_sites},
                            multiplicities={species: multiplicity},
                            equivalent_positions={species: equiv_fpos},
                        )
                    )

            pbar.update(15)  # 45% of progress bar, generating interstitials typically takes the longest

            # Generate DefectEntry objects:
            pbar.set_description("Determining Wyckoff sites")
            num_defects = sum([len(defect_list) for defect_list in self.defects.values()])

            defect_entry_list = []
            wyckoff_label_dict = get_wyckoff_dict_from_sgn(sga.get_space_group_number())
            # determine cell orientation for Wyckoff site determination (needs to match the Bilbao
            # Crystallographic Server's convention, which can differ from spglib (pymatgen) in some cases)

            sga_wyckoffs = sga.get_symmetrized_structure().wyckoff_symbols

            for trial_lattice_vec_swap_array in [  # 3C2 -> 6 possible combinations
                # ordered according to frequency of occurrence in the Materials Project
                [0, 1, 2],  # abc, ~95% of cases
                [0, 2, 1],  # acb
                [2, 1, 0],  # cba
                [1, 0, 2],  # bac
                [2, 0, 1],  # cab
                [1, 2, 0],  # bca
                None,  # no perfect match, default to original orientation
            ]:
                if trial_lattice_vec_swap_array is None:
                    lattice_vec_swap_array = [0, 1, 2]
                    break

                reoriented_conv_structure = swap_axes(
                    self.conventional_structure, trial_lattice_vec_swap_array
                )
                if _compare_wyckoffs(
                    sga_wyckoffs,
                    reoriented_conv_structure,
                    wyckoff_label_dict,
                    trial_lattice_vec_swap_array,
                ):
                    lattice_vec_swap_array = trial_lattice_vec_swap_array
                    break

            self._BilbaoCS_conv_cell_vector_mapping = lattice_vec_swap_array

            for _defect_type, defect_list in self.defects.items():
                for defect in defect_list:
                    defect_supercell, defect_supercell_site = defect.get_supercell_structure(
                        sc_mat=self.supercell_matrix,
                        dummy_species="X",  # keep track of the defect frac coords in the supercell
                        target_frac_coords=self.target_frac_coords,
                        return_site=True,
                    )
                    neutral_defect_entry = get_defect_entry_from_defect(
                        defect,
                        defect_supercell,
                        0,
                        dummy_species=_dummy_species,
                    )
                    neutral_defect_entry.defect_supercell_site = defect_supercell_site
                    neutral_defect_entry.bulk_supercell = self.bulk_supercell
                    neutral_defect_entry.conventional_structure = self.conventional_structure
                    neutral_defect_entry.defect.conventional_structure = self.conventional_structure
                    wyckoff_label, conv_cell_coord_list = get_wyckoff_label_and_equiv_coord_list(
                        defect_entry=neutral_defect_entry,
                        wyckoff_dict=wyckoff_label_dict,
                        lattice_vec_swap_array=self._BilbaoCS_conv_cell_vector_mapping,
                    )
                    conv_cell_coord_list.sort(key=_frac_coords_sort_func)
                    conv_cell_coord_list = np.round(conv_cell_coord_list, 5)
                    neutral_defect_entry.wyckoff = wyckoff_label
                    neutral_defect_entry.defect.wyckoff = neutral_defect_entry.wyckoff
                    neutral_defect_entry.conv_cell_frac_coords = conv_cell_coord_list[
                        0
                    ]  # ideal/cleanest coords
                    neutral_defect_entry.defect.conv_cell_frac_coords = (
                        neutral_defect_entry.conv_cell_frac_coords
                    )
                    neutral_defect_entry.equiv_conv_cell_frac_coords = conv_cell_coord_list
                    neutral_defect_entry.defect.equiv_conv_cell_frac_coords = (
                        neutral_defect_entry.equiv_conv_cell_frac_coords
                    )
                    neutral_defect_entry._BilbaoCS_conv_cell_vector_mapping = (
                        self._BilbaoCS_conv_cell_vector_mapping
                    )
                    defect_entry_list.append(neutral_defect_entry)
                    pbar.update((1 / num_defects) * ((pbar.total * 0.9) - pbar.n))  # 90% of progress bar

            pbar.set_description("Generating DefectEntry objects")
            named_defect_dict = name_defect_entries(defect_entry_list)
            pbar.update(5)  # 95% of progress bar

            for defect_name_wout_charge, neutral_defect_entry in named_defect_dict.items():
                charge_states = guess_defect_charge_states(
                    neutral_defect_entry.defect, **self.charge_state_gen_kwargs
                )

                for charge in charge_states:
                    defect_entry = copy.deepcopy(neutral_defect_entry)
                    defect_entry.charge_state = charge
                    defect_name = defect_name_wout_charge + f"_{'+' if charge > 0 else ''}{charge}"
                    defect_entry.name = defect_name  # set name attribute
                    self.defect_entries[defect_name] = defect_entry
                    pbar.update((1 / num_defects) * (pbar.total - pbar.n))  # 100% of progress bar

        except Exception as e:
            pbar.close()
            raise e

        finally:
            pbar.close()

        print(self._defect_generator_info())

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
        # TODO: Saving and reloading removes the name attribute from defect entries (and wyckoff and
        #  others?), need to fix this!
        # All other attributes correctly regenerated here, or need to be modified? Check!!
        d_decoded = MontyDecoder().process_decoded(d)  # decode dict
        defects_generator = cls.__new__(
            cls
        )  # Create new DefectsGenerator object without invoking __init__

        # Manually set object attributes
        for key, value in d_decoded.items():
            if hasattr(defects_generator, key):
                setattr(defects_generator, key, value)
            else:
                warnings.warn(
                    f"Attribute {key} found in dict but not a valid attribute for "
                    f"DefectsGenerator object!"
                )

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
            filename = f"{self.primitive_structure.composition.to_pretty_string()}_defects_generator.json"

        dumpfn(self, filename)

    @classmethod
    def from_json(cls, filename: str):
        """
        Load a DefectsGenerator object from a json file.

        Args:
            filename (str): Filename of json file to load DefectsGenerator object from.

        Returns:
            DefectsGenerator object
        """
        return loadfn(filename)

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
                matching_type_names_wout_charge = list(
                    {defect_entry_name.rsplit("_", 1)[0] for defect_entry_name in matching_defect_types}
                )

                def _first_and_second_element(defect_name):  # for sorting purposes
                    if defect_name.startswith("v"):
                        return (defect_name.split("_")[1], defect_name.split("_")[1])
                    if defect_name.split("_")[1] == "i":
                        return (defect_name.split("_")[0], defect_name.split("_")[0])

                    return (
                        defect_name.split("_")[0],
                        defect_name.split("_")[1],
                    )

                # sort to match order of appearance of elements in the primitive structure
                # composition, then alphabetically
                element_list = [el.symbol for el in self.primitive_structure.composition.elements]
                sorted_defect_names = sorted(
                    matching_type_names_wout_charge,
                    key=lambda s: (
                        element_list.index(_first_and_second_element(s)[0]),
                        element_list.index(_first_and_second_element(s)[1]),
                        s,
                    ),
                )
                for defect_name in sorted_defect_names:
                    charges = [
                        name.rsplit("_", 1)[1]
                        for name in self.defect_entries
                        if name.startswith(defect_name + "_")
                    ]  # so e.g. Te_i_m1 doesn't match with Te_i_m1b
                    # convert list of strings to one string with comma-separated charges
                    charges = "[" + ",".join(charges) + "]"
                    neutral_defect_entry = self.defect_entries[
                        defect_name + "_0"
                    ]  # neutral has no +/- sign
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

    def __getattr__(self, attr):
        """
        Redirects an unknown attribute/method call to the defect_entries
        dictionary attribute, if the attribute doesn't exist in
        DefectsGenerator.
        """
        if hasattr(self.defect_entries, attr):
            return getattr(self.defect_entries, attr)

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

        if value.defect.structure != self.primitive_structure:
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
        if value.defect not in self.defects[defects_key]:
            self.defects[defects_key].append(value.defect)

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
        return (
            f"DefectsGenerator for input composition "
            f"{self.primitive_structure.composition.to_pretty_string()}, space group "
            f"{self.primitive_structure.get_space_group_info()[0]} with {len(self)} defect entries "
            f"created."
        )

    def __repr__(self):
        """
        Returns a string representation of the DefectsGenerator object, and
        prints the DefectsGenerator info.
        """
        return self.__str__() + "\n" + self._defect_generator_info()


# Schoenflies, Hermann-Mauguin, spgid dict: (Taken from the excellent Abipy with GNU GPL License) 🙌
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
