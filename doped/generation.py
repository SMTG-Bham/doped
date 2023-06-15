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
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import DummySpecies
from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.advanced_transformations import _proj
from sympy import Eq, simplify, solve, symbols
from tabulate import tabulate
from tqdm import tqdm

# TODO: Use new doped naming functions in SnB
# TODO: Should have option to provide the bulk supercell to use, and generate from this, in case ppl
#  want to directly compare to calculations with the same supercell before etc
# TODO: For specifying interstitial sites, will want to be able to specify as either primitive or
#  supercell coords in this case, so will need functions for transforming between primitive and
#  supercell defect structures (will want this for defect parsing as well). Defectivator has functions
#  that do some of this. This will be tricky (SSX trickay you might say) for relaxed interstitials ->
#  get symmetry-equivalent positions of relaxed interstitial position in unrelaxed bulk (easy pal,
#  tf you mean 'tricky'??)

dummy_species = DummySpecies("X")  # Dummy species used to keep track of defect coords in the supercell


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
    dummy_species: DummySpecies = dummy_species,
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


def closest_site_info(defect_entry, n=1):
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
        if i == 0 or site_distances[i][0] - site_distances[i - 1][0] > 0.02
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
    defect_naming_dict = {}
    for defect_entry in defect_entries:
        full_defect_name = get_defect_name_from_entry(defect_entry)
        if defect_entry.defect.defect_type == DefectType.Interstitial:
            # append point group to pmg name for interstitials
            # need to determine matching key, update it, then recheck if matching until unique
            shorter_defect_name = full_defect_name.rsplit("_", 1)[0]  # pmg name + point group
            matching_previous_defect_name = [
                name for name in defect_naming_dict if shorter_defect_name in name
            ]
            if not matching_previous_defect_name:
                defect_naming_dict[shorter_defect_name] = defect_entry
                continue

            matching_previous_shorter_defect_names = [
                name for name in defect_naming_dict if name == shorter_defect_name
            ]
            if len(matching_previous_shorter_defect_names) > 1:
                raise ValueError(
                    f"Multiple defect entries with same name: {matching_previous_shorter_defect_names}"
                )

            if len(matching_previous_shorter_defect_names) == 1:
                # match for shortest_defect_name, need to rename previous entry
                prev_defect_entry = defect_naming_dict.pop(matching_previous_shorter_defect_names[0])
                prev_defect_entry_full_name = get_defect_name_from_entry(
                    prev_defect_entry
                )  # w/closest site info
                defect_naming_dict[
                    prev_defect_entry_full_name
                ] = prev_defect_entry  # update previous entry

            if not any(name for name in defect_naming_dict if full_defect_name in name):
                # renaming previous entry no means no match with full defect name
                defect_naming_dict[full_defect_name] = defect_entry
                continue

            # if still no match, need to add closest site info to name
            matching_names = True
            n = 2
            while matching_names:
                # append 2nd, 3rd, 4th etc closest site info to name until unique:
                for name in list(defect_naming_dict.keys()):
                    if full_defect_name == name:
                        prev_defect_entry_full_name = name + closest_site_info(
                            defect_naming_dict[name], n=n
                        )
                        prev_defect_entry = defect_naming_dict.pop(name)
                        defect_naming_dict[prev_defect_entry_full_name] = prev_defect_entry

                full_defect_name += closest_site_info(defect_entry, n=n)

                if not any(name for name in defect_naming_dict if full_defect_name in name):
                    # no match, can add to dict
                    defect_naming_dict[full_defect_name] = defect_entry
                    matching_names = False

                elif n > 4:  # revert to a,b,c... naming if still not unique at this point
                    # (extremely rare, essentially only if generating defects using a defective
                    # supercell as the 'bulk base')
                    for name in list(defect_naming_dict.keys()):
                        if full_defect_name == name:
                            prev_defect_entry = defect_naming_dict.pop(name)
                            defect_naming_dict[name + "a"] = prev_defect_entry
                            defect_naming_dict[full_defect_name + "b"] = defect_entry
                            break

                        if full_defect_name == name[:-1]:
                            # rename to {defect_name}{iterated letter}
                            last_letters = [
                                name[-1] for name in defect_naming_dict if name[:-1] == full_defect_name
                            ]
                            last_letters.sort()
                            last_letter = last_letters[-1]
                            new_letter = chr(ord(last_letter) + 1)
                            full_defect_name = f"{full_defect_name}{new_letter}"
                            defect_naming_dict[full_defect_name] = defect_entry
                            break

                    matching_names = False

                n += 1

        else:  # vacancies and substitutions, start with pmg name
            shortest_defect_name = full_defect_name.rsplit("_", 2)[0]  # pmg name
            matching_previous_defect_names = [
                name for name in defect_naming_dict if shortest_defect_name in name
            ]
            if not matching_previous_defect_names:
                defect_naming_dict[shortest_defect_name] = defect_entry
                continue

            matching_previous_shortest_defect_names = [
                name for name in defect_naming_dict if name == shortest_defect_name
            ]
            if len(matching_previous_shortest_defect_names) > 1:
                raise ValueError(
                    f"Multiple defect entries with same name: "
                    f"{matching_previous_shortest_defect_names}"
                )

            if len(matching_previous_shortest_defect_names) == 1:
                # match for shortest_defect_name, need to rename previous entry
                prev_defect_entry = defect_naming_dict.pop(matching_previous_shortest_defect_names[0])
                prev_defect_entry_full_name = get_defect_name_from_entry(prev_defect_entry)
                prev_defect_entry_shorter_name = prev_defect_entry_full_name.rsplit("_", 1)[
                    0
                ]  # pmg name + point group
                defect_naming_dict[prev_defect_entry_shorter_name] = prev_defect_entry

            shorter_defect_name = full_defect_name.rsplit("_", 1)[0]  # pmg name + point group
            if not any(name for name in defect_naming_dict if shorter_defect_name in name):
                # renaming previous entry no means no match with shorter defect name
                defect_naming_dict[shorter_defect_name] = defect_entry
                continue

            if any(name for name in defect_naming_dict if shorter_defect_name == name):
                # match for shorter_defect_name, need to rename previous entry
                prev_defect_entry = defect_naming_dict.pop(
                    [name for name in defect_naming_dict if shorter_defect_name == name][0]
                )
                prev_defect_entry_full_name = get_defect_name_from_entry(
                    prev_defect_entry
                )  # w/closest site info
                defect_naming_dict[prev_defect_entry_full_name] = prev_defect_entry

            if not any(name for name in defect_naming_dict if full_defect_name in name):
                # renaming previous entry no means no match with full defect name
                defect_naming_dict[full_defect_name] = defect_entry
                continue

            # if still no match, need to add closest site info to name
            matching_names = True
            n = 2
            while matching_names:
                # append 2nd, 3rd, 4th etc closest site info to name until unique:
                for name in list(defect_naming_dict.keys()):
                    if full_defect_name == name:
                        prev_defect_entry_full_name = name + closest_site_info(
                            defect_naming_dict[name], n=n
                        )
                        prev_defect_entry = defect_naming_dict.pop(name)
                        defect_naming_dict[prev_defect_entry_full_name] = prev_defect_entry

                full_defect_name += closest_site_info(defect_entry, n=n)

                if not any(name for name in defect_naming_dict if full_defect_name in name):
                    # no match, can add to dict
                    defect_naming_dict[full_defect_name] = defect_entry
                    matching_names = False

                elif n > 4:  # revert to a,b,c... naming if still not unique at this point
                    # (extremely rare, essentially only if generating defects using a defective
                    # supercell as the 'bulk base')
                    for name in list(defect_naming_dict.keys()):
                        if full_defect_name == name:
                            prev_defect_entry = defect_naming_dict.pop(name)
                            defect_naming_dict[name + "a"] = prev_defect_entry
                            defect_naming_dict[full_defect_name + "b"] = defect_entry
                            break

                        if full_defect_name == name[:-1]:
                            # rename to {defect_name}{iterated letter}
                            last_letters = [
                                name[-1] for name in defect_naming_dict if name[:-1] == full_defect_name
                            ]
                            last_letters.sort()
                            last_letter = last_letters[-1]
                            new_letter = chr(ord(last_letter) + 1)
                            full_defect_name = f"{full_defect_name}{new_letter}"
                            defect_naming_dict[full_defect_name] = defect_entry
                            break

                    matching_names = False

                n += 1

    return defect_naming_dict


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


def get_wyckoff_label(defect_entry, wyckoff_dict=None):
    """
    Return the Wyckoff label for a defect entry's site, given a dictionary of
    Wyckoff labels and coordinates (`wyckoff_dict`).

    If `wyckoff_dict` is not provided, the spacegroup of the bulk structure is
    determined and used to generate it with `get_wyckoff_dict_from_sgn()`.
    """
    if wyckoff_dict is None:
        sga = SpacegroupAnalyzer(defect_entry.defect.structure)
        wyckoff_dict = get_wyckoff_dict_from_sgn(sga.get_space_group_number())

    def _compare_arrays(coord_list, coord_array):
        """
        Compare a list of arrays of sympy expressions (`coord_list`) with an
        array of coordinates (`coord_array`).

        Returns the matching array from the list.
        """
        x, y, z = symbols("x y z")
        variable_dict = {}  # dict for x,y,z

        for sympy_array in coord_list:
            variable_dict.clear()
            match = True

            for coord, sympy_expr in zip(coord_array, sympy_array):
                # Evaluate the expression with the current variable_dict
                expr_value = simplify(sympy_expr).subs(variable_dict)

                # If the expression cannot be evaluated to a float
                # it means that there is a new variable in the expression
                try:
                    expr_value = np.mod(float(expr_value), 1)  # wrap to 0-1 (i.e. to unit cell)

                except TypeError:
                    # Assign the expression the value of the corresponding coordinate, and solve
                    # for the new variable
                    equation = Eq(sympy_expr, coord)
                    variable = list(sympy_expr.free_symbols)[0]
                    variable_dict[variable] = solve(equation, variable)[0]
                    expr_value = simplify(sympy_expr).subs(variable_dict)

                # Check if the evaluated expression matches the corresponding coordinate
                if not np.isclose(float(coord), float(expr_value), rtol=1e-2):
                    match = False
                    break

            if match:
                return True  # This is the matching array

        return False  # No match found

    # get match of coords in wyckoff_label_coords to defect site coords:
    def find_closest_match(defect_site, wyckoff_label_coords_dict):
        for label, coord_list in wyckoff_label_coords_dict.items():
            if _compare_arrays(coord_list, np.array(defect_site.frac_coords)):
                return label

        return None  # No match found

    unit_cell_defect_site = defect_entry.defect.site.to_unit_cell()  # ensure wrapped to unit cell
    return find_closest_match(unit_cell_defect_site, wyckoff_dict)


class DefectsGenerator:
    def __init__(
        self,
        structure: Structure,
        extrinsic: Optional[Union[str, List, Dict]] = None,
        interstitial_coords: Optional[List] = None,
        generate_supercell: bool = True,
        **kwargs,
    ):
        """
        Generates pymatgen DefectEntry objects for defects in the input host
        structure. By default, generates all intrinsic defects, but extrinsic
        defects (impurities) can also be created using the `extrinsic`
        argument.

        Interstitial sites are generated using Voronoi tessellation by default (found
        to be the most reliable), however these can also be manually specified using
        the `interstitial_coords` argument.

        By default, supercells are generated for each defect using the pymatgen
        `get_supercell_structure()` method, with `doped` default settings of
        `min_length = 10` (minimum supercell length of 10 Å) and `min_atoms = 50`
        (minimum 50 atoms in supercell). If a different supercell is desired, this
        can be controlled by specifying keyword arguments in DefectsGenerator(),
        which are passed to the `get_supercell_structure()` method.
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

        # TODO: Mention how charge states are generated, and how to modify, as shown in the example
        # notebook
        # Also show how to remove certain defects from the dictionary? Mightn't be worth the space for
        # this though

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
            **kwargs:
                Keyword arguments to be passed to the `get_supercell_structure()` method.

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
        """
        self.defects = {}  # {defect_type: [Defect, ...]}
        self.defect_entries = {}  # {defect_name: DefectEntry}
        if extrinsic is None:
            extrinsic = []
        if interstitial_coords is None:
            interstitial_coords = []

        pbar = tqdm(total=100)  # tqdm progress bar. 100% is completion
        pbar.set_description("Getting primitive structure")

        try:  # put code in try/except block so progress bar always closed if interrupted
            # Reduce structure to primitive cell for efficient defect generation
            # same symprec as defect generators in pymatgen-analysis-defects:
            sga = SpacegroupAnalyzer(structure, symprec=1e-2)
            prim_struct = sga.get_primitive_standard_structure()
            if prim_struct.num_sites < structure.num_sites:
                clean_prim_struct_dict = _round_floats(prim_struct.as_dict())
                primitive_structure = Structure.from_dict(clean_prim_struct_dict)
            else:  # primitive cell is the same as input structure, so use input structure to avoid
                # rotations
                # wrap to unit cell:
                primitive_structure = Structure.from_sites([site.to_unit_cell() for site in structure])
            pbar.update(5)  # 5% of progress bar

            # Generate supercell once, so this isn't redundantly rerun for each defect, and ensures the
            # same supercell is used for each defect and bulk calculation
            pbar.set_description("Generating simulation supercell")
            pmg_supercell_matrix = get_sc_fromstruct(
                primitive_structure,
                min_atoms=kwargs.get("min_atoms", 50),
                max_atoms=kwargs.get("max_atoms", 500),  # different to current pymatgen default (240)
                min_length=kwargs.get("min_length", 10),  # same as current pymatgen default
                force_diagonal=kwargs.get("force_diagonal", False),  # same as current pymatgen default
            )

            # check if input structure is already >10 Å in each direction:
            a = structure.lattice.matrix[0]
            b = structure.lattice.matrix[1]
            c = structure.lattice.matrix[2]

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
                # first rotate primitive structure to match target structure:
                mapping = prim_struct.lattice.find_mapping(target_struct.lattice)
                rotation_matrix = mapping[1]
                supercell_matrix = mapping[2]
                rotation_symmop = SymmOp.from_rotation_and_translation(
                    rotation_matrix=rotation_matrix.T
                )  # Transpose = inverse of rotation matrices (orthogonal matrices), better numerical
                # stability
                output_prim_struct = prim_struct.copy()
                output_prim_struct.apply_operation(rotation_symmop)
                clean_prim_struct_dict = _round_floats(output_prim_struct.as_dict())
                return Structure.from_dict(clean_prim_struct_dict), supercell_matrix

            if np.min(np.linalg.norm(length_vecs, axis=1)) >= kwargs.get("min_length", 10):
                # input structure is >10 Å in each direction
                if (
                    not generate_supercell
                    or structure.num_sites <= (primitive_structure * pmg_supercell_matrix).num_sites
                ):
                    # input structure has fewer or same number of atoms as pmg supercell or
                    # generate_supercell=False, so use input structure:
                    self.primitive_structure, self.supercell_matrix = _rotate_and_get_supercell_matrix(
                        primitive_structure, structure
                    )
                else:
                    self.primitive_structure = primitive_structure
                    self.supercell_matrix = pmg_supercell_matrix

            elif not generate_supercell:
                # input structure is <10 Å in at least one direction, and generate_supercell=False,
                # so use input structure but warn user:
                warnings.warn(
                    f"\nInput structure is <10 Å in at least one direction (minimum image distance = "
                    f"{np.min(np.linalg.norm(length_vecs, axis=1)):.2f} Å, which is usually too "
                    f"small for accurate defect calculations, but generate_supercell=False, so "
                    f"using input structure as defect & bulk supercells. Caution advised!"
                )
                self.primitive_structure, self.supercell_matrix = _rotate_and_get_supercell_matrix(
                    primitive_structure, structure
                )

            else:
                self.primitive_structure = primitive_structure
                self.supercell_matrix = pmg_supercell_matrix

            self.bulk_supercell = self.primitive_structure * self.supercell_matrix
            # check that generated supercell is >10 Å in each direction:
            if (
                np.min(np.linalg.norm(self.bulk_supercell.lattice.matrix, axis=1))
                < kwargs.get("min_length", 10)
                and generate_supercell
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
            if isinstance(extrinsic, str):  # substitute all host elements
                substitutions = {
                    el.symbol: [extrinsic] for el in self.primitive_structure.composition.elements
                }
            elif isinstance(extrinsic, list):  # substitute all host elements
                substitutions = {
                    el.symbol: extrinsic for el in self.primitive_structure.composition.elements
                }
            elif isinstance(extrinsic, dict):  # substitute only specified host elements
                substitutions = extrinsic
            else:
                warnings.warn(
                    f"Invalid `extrinsic` defect input. Got type {type(extrinsic)}, but string or list or "
                    f"dict required. No extrinsic defects will be generated."
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
            pbar.update(5)  # 30% of progress bar

            # Interstitials:
            # determine which, if any, extrinsic elements are present:
            pbar.set_description("Generating interstitials")
            # previous generators add oxidation states, but messes with interstitial generators, so
            # remove oxi states:
            self.primitive_structure.remove_oxidation_states()
            if isinstance(extrinsic, str):
                extrinsic_elements = [extrinsic]
            elif isinstance(extrinsic, list):
                extrinsic_elements = extrinsic
            elif isinstance(extrinsic, dict):  # dict of host: extrinsic elements, as lists or strings
                # convert to flattened list of extrinsic elements:
                extrinsic_elements = list(
                    chain(*[i if isinstance(i, list) else [i] for i in extrinsic.values()])
                )
                extrinsic_elements = list(set(extrinsic_elements))  # get only unique elements
            else:
                extrinsic_elements = []

            if interstitial_coords:
                # For the moment, this assumes interstitial_sites
                insertions = {
                    el.symbol: interstitial_coords for el in self.primitive_structure.composition.elements
                }
                insertions.update({el: interstitial_coords for el in extrinsic_elements})
                interstitial_generator_obj = InterstitialGenerator()
                interstitial_generator = interstitial_generator_obj.generate(
                    self.primitive_structure, insertions=insertions
                )
                self.defects["interstitials"] = list(interstitial_generator)

            else:
                # Generate interstitial sites using Voronoi tessellation
                voronoi_interstitial_generator_obj = VoronoiInterstitialGenerator()
                voronoi_interstitial_generator = voronoi_interstitial_generator_obj.generate(
                    self.primitive_structure,
                    insert_species=[el.symbol for el in self.primitive_structure.composition.elements]
                    + extrinsic_elements,
                )
                self.defects["interstitials"] = list(voronoi_interstitial_generator)

            pbar.update(15)  # 45% of progress bar, generating interstitials typically takes the longest

            # Generate DefectEntry objects:
            pbar.set_description("Determining Wyckoff sites")
            num_defects = sum([len(defect_list) for defect_list in self.defects.values()])

            defect_entry_list = []
            wyckoff_label_dict = get_wyckoff_dict_from_sgn(sga.get_space_group_number())
            for _defect_type, defect_list in self.defects.items():
                for defect in defect_list:
                    defect_supercell = defect.get_supercell_structure(
                        sc_mat=self.supercell_matrix,
                        dummy_species="X",  # keep track of the defect frac coords in the supercell
                    )
                    neutral_defect_entry = get_defect_entry_from_defect(
                        defect,
                        defect_supercell,
                        0,
                        dummy_species=DummySpecies("X"),
                    )
                    wyckoff_label = get_wyckoff_label(neutral_defect_entry, wyckoff_label_dict)
                    neutral_defect_entry.wyckoff = wyckoff_label
                    defect_entry_list.append(neutral_defect_entry)
                    pbar.update((1 / num_defects) * ((pbar.total * 0.9) - pbar.n))  # 90% of progress bar

            pbar.set_description("Generating DefectEntry objects")
            named_defect_dict = name_defect_entries(defect_entry_list)
            pbar.update(5)  # 95% of progress bar

            for defect_name_wout_charge, neutral_defect_entry in named_defect_dict.items():
                defect = neutral_defect_entry.defect
                # set defect charge states: currently from +/-1 to defect oxi state
                if defect.oxi_state > 0:
                    charge_states = [*range(-1, int(defect.oxi_state) + 1)]  # from -1 to oxi_state
                elif defect.oxi_state < 0:
                    charge_states = [*range(int(defect.oxi_state), 2)]  # from oxi_state to +1
                else:  # oxi_state is 0
                    charge_states = [-1, 0, 1]

                for charge in charge_states:
                    # TODO: Will be updated to our chosen charge generation algorithm!
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
            "defects": self.defects,
            "defect_entries": self.defect_entries,
            "primitive_structure": self.primitive_structure,
            "supercell_matrix": self.supercell_matrix,
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
        d_decoded = MontyDecoder().process_decoded(d)  # decode dict
        defects_generator = cls.__new__(
            cls
        )  # Create new DefectsGenerator object without invoking __init__

        # Manually set object attributes
        defects_generator.defects = d_decoded["defects"]
        defects_generator.defect_entries = d_decoded[
            "defect_entries"
        ]  # TODO: Saving and reloading removes the name attribute from defect entries, need to fix this!
        defects_generator.primitive_structure = d_decoded["primitive_structure"]
        defects_generator.supercell_matrix = d_decoded["supercell_matrix"]

        return defects_generator

    def _defect_generator_info(self):
        """
        Returns a string with information about the defects that have been
        generated by the DefectsGenerator.
        """
        info_string = ""
        for defect_class, defect_list in self.defects.items():
            table = []
            header = [
                defect_class.capitalize(),
                "Charge States",
                "Unit Cell Coords",
                "\x1B[3mg\x1B[0m_site",
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
                neutral_defect_entry = self.defect_entries[defect_name + "_0"]  # neutral has no +/- sign
                frac_coords_string = ",".join(
                    f"{x:.2f}" for x in neutral_defect_entry.defect.site.frac_coords
                )
                row = [
                    defect_name,
                    charges,
                    f"[{frac_coords_string}]",
                    neutral_defect_entry.defect.multiplicity,
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
        info_string += (
            "\x1B[3mg\x1B[0m_site = Site Multiplicity (in Primitive Unit Cell)\n"
            "Note that Wyckoff letters can depend on the ordering of elements in the primitive standard "
            "structure (returned by spglib)\n"
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
            dummy_species="X",  # keep track of the defect frac coords in the supercell
        )
        defect_entry = get_defect_entry_from_defect(
            value.defect,
            defect_supercell,
            charge_state=0,  # just checking supercell structure here
            dummy_species=DummySpecies("X"),
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
