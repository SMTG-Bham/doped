"""
Code to generate Defect objects and supercell structures for ab-initio calculations.
"""
import copy
import warnings
from tabulate import tabulate
from typing import Optional, List, Dict, Union
from itertools import chain
from tqdm import tqdm

import numpy as np
from monty.json import MontyDecoder
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import DummySpecies
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.defects.core import Defect, DefectType
from pymatgen.analysis.defects.supercells import get_sc_fromstruct
from pymatgen.analysis.defects.thermo import DefectEntry
from pymatgen.analysis.defects.generators import (
    VacancyGenerator,
    AntiSiteGenerator,
    SubstitutionGenerator,
    InterstitialGenerator,
    VoronoiInterstitialGenerator,
)
from ase.spacegroup.wyckoff import Wyckoff  # TODO: PR this to ASE if not already merged

# TODO: Use new doped naming functions in SnB


# TODO: Should have option to provide the bulk supercell to use, and generate from this, in case ppl want to directly
#  compare to calculations with the same supercell before etc
# TODO: For specifying interstitial sites, will want to be able to specify as either primitive or supercell coords in
#  this case, so will need functions for transforming between primitive and supercell defect structures (will want this
#  for defect parsing as well). Defectivator has functions that do some of this. This will be tricky (SSX trickay you
#  might say) for relaxed interstitials -> get symmetry-equivalent positions of relaxed interstitial position in
#  unrelaxed bulk (easy pal, tf you mean 'tricky'??)


def get_defect_entry_from_defect(
    defect: Defect,
    defect_supercell: Structure,
    charge_state: int,
    dummy_species: DummySpecies = DummySpecies("X"),
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
        site
        for site in defect_entry_structure
        if site.species.elements[0].symbol == dummy_species.symbol
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
    elif isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_round_floats(x) for x in obj]
    return obj


def _defect_dict_key_from_pmg_type(defect_type: DefectType) -> str:
    """
    Returns the corresponding defect dictionary key for a pymatgen Defect object.
    """
    if defect_type == DefectType.Vacancy:
        return "vacancies"
    elif defect_type == DefectType.Substitution:
        return "substitutions"
    elif defect_type == DefectType.Interstitial:
        return "interstitials"
    elif defect_type == DefectType.Other:
        return "others"


def closest_site_info(defect_entry, n=1):
    """
    Return the element and distance (rounded to 2 decimal places) of the closest site to defect_entry.sc_defect_frac_coords in defect_entry.sc_entry.structure, with distance > 0.01 (i.e. so not the site itself), and if there are multiple elements with the same distance, sort alphabetically and return the first one.
    If n is set, then it returns the nth closest site, where the nth site must be at least 0.02 Å further away than the n-1th site.
    """

    site_distances = [
        (
            site.distance_and_image_from_frac_coords(
                defect_entry.sc_defect_frac_coords
            )[0],
            site,
        )
        for site in defect_entry.sc_entry.structure.sites
        if site.distance_and_image_from_frac_coords(defect_entry.sc_defect_frac_coords)[
            0
        ]
        > 0.01
    ]

    min_distance, closest_site = min(
        site_distances, key=lambda x: (round(x[0], 2), x[1].specie.symbol)
    )

    if n > 1:
        for i in range(n - 1):
            site_distances = [
                (
                    site.distance_and_image_from_frac_coords(
                        defect_entry.sc_defect_frac_coords
                    )[0],
                    site,
                )
                for site in defect_entry.sc_entry.structure.sites
                if site.distance_and_image_from_frac_coords(
                    defect_entry.sc_defect_frac_coords
                )[0]
                > min_distance + 0.02
            ]
            min_distance, closest_site = min(
                site_distances, key=lambda x: (round(x[0], 2), x[1].specie.symbol)
            )

    return closest_site.specie.symbol + f"{min_distance:.2f}"


def get_defect_name_from_entry(defect_entry):
    """Get the doped/SnB defect name from DefectEntry object"""
    sga = SpacegroupAnalyzer(defect_entry.sc_entry.structure)
    defect_name = (
        f"{defect_entry.defect.name}_{herm2sch(sga.get_point_group_symbol())}"
        f"_{closest_site_info(defect_entry)}"
    )

    return defect_name


def name_defect_entries(defect_entries):
    """
    Create a dictionary of {Name: DefectEntry} from a list of DefectEntry objects, where the
    names are set according to the default doped algorithm; which is to use the pymatgen defect
    name (e.g. v_Cd, Cd_Te etc.) for vacancies/antisites/substitutions, unless there are multiple
    inequivalent sites for the defect, in which case the point group of the defect site is appended
    (e.g. v_Cd_Td, Cd_Te_Td etc.), and if this is still not unique, then element identity and distance
    to the nearest neighbour of the defect site is appended (e.g. v_Cd_Td_Te2.83, Cd_Te_Td_Cd2.83
    etc.).
    For interstitials, the same naming scheme is used, but the point group is always appended
    to the pymatgen defect name.
    """
    defect_naming_dict = {}
    for defect_entry in defect_entries:
        full_defect_name = get_defect_name_from_entry(defect_entry)
        if defect_entry.defect.defect_type == DefectType.Interstitial:
            shortest_defect_name = full_defect_name.rsplit("_", 1)[
                0
            ]  # pmg name + point group
            matching_previous_defect_name = [
                name for name in defect_naming_dict if shortest_defect_name in name
            ]
            if not matching_previous_defect_name:
                defect_naming_dict[shortest_defect_name] = defect_entry

            else:
                if len(matching_previous_defect_name) > 1:
                    raise ValueError(
                        f"Multiple previous defect names match "
                        f"{shortest_defect_name}."
                    )
                else:
                    prev_defect_entry = defect_naming_dict.pop(
                        matching_previous_defect_name[0]
                    )
                    prev_defect_entry_full_name = get_defect_name_from_entry(
                        prev_defect_entry
                    )
                    if prev_defect_entry_full_name != full_defect_name:
                        defect_naming_dict[
                            prev_defect_entry_full_name
                        ] = prev_defect_entry
                        defect_naming_dict[full_defect_name] = defect_entry
                    else:
                        matching_names = True
                        n = 2
                        while matching_names:
                            # append 2nd,3rd,4th etc closest site info to name until unique:
                            prev_defect_entry_full_name += closest_site_info(
                                prev_defect_entry, n=n
                            )
                            full_defect_name += closest_site_info(defect_entry, n=n)
                            if prev_defect_entry_full_name != full_defect_name:
                                defect_naming_dict[
                                    prev_defect_entry_full_name
                                ] = prev_defect_entry
                                defect_naming_dict[full_defect_name] = defect_entry
                                matching_names = False

                            n += 1

        else:  # vacancies and substitutions
            shortest_defect_name = full_defect_name.rsplit("_", 2)[0]  # pmg name
            matching_previous_defect_name = [
                name for name in defect_naming_dict if shortest_defect_name in name
            ]
            if not matching_previous_defect_name:
                defect_naming_dict[shortest_defect_name] = defect_entry

            else:
                if len(matching_previous_defect_name) > 1:
                    raise ValueError(
                        f"Multiple previous defect names match {shortest_defect_name}."
                    )
                else:
                    prev_defect_entry = defect_naming_dict.pop(
                        matching_previous_defect_name[0]
                    )
                    prev_defect_entry_full_name = get_defect_name_from_entry(
                        prev_defect_entry
                    )
                    prev_defect_entry_shorter_name = prev_defect_entry_full_name.rsplit(
                        "_", 1
                    )[
                        0
                    ]  # pmg name + point group
                    shorter_defect_name = full_defect_name.rsplit("_", 1)[
                        0
                    ]  # pmg name + point group
                    if prev_defect_entry_shorter_name != shorter_defect_name:
                        defect_naming_dict[
                            prev_defect_entry_shorter_name
                        ] = prev_defect_entry
                        defect_naming_dict[shorter_defect_name] = defect_entry

                    else:
                        if (
                            prev_defect_entry_full_name != full_defect_name
                        ):  # w/closest site info
                            defect_naming_dict[
                                prev_defect_entry_full_name
                            ] = prev_defect_entry
                            defect_naming_dict[full_defect_name] = defect_entry

                        else:
                            matching_names = True
                            n = 2
                            while matching_names:
                                # append 2nd,3rd,4th etc closest site info to name until unique:
                                prev_defect_entry_full_name += closest_site_info(
                                    prev_defect_entry, n=n
                                )
                                full_defect_name += closest_site_info(defect_entry, n=n)
                                if prev_defect_entry_full_name != full_defect_name:
                                    defect_naming_dict[
                                        prev_defect_entry_full_name
                                    ] = prev_defect_entry
                                    defect_naming_dict[full_defect_name] = defect_entry
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
    """Convert from Hermann-Mauguin to Schoenflies."""
    return _HERM2SCH.get(herm_symbol, None)


def get_wyckoff_dict_from_sgn(sgn):
    """Get dictionary of {Wyckoff label: coordinates} for a given space group number."""
    wyckoff = Wyckoff(sgn).wyckoff
    wyckoff_label_coords_dict = {}

    def _coord_string_to_array(coord_string):
        # Split string into substrings, evaluate each as a Python expression,
        # then convert to numpy array
        return np.array(
            [
                float(eval(x)) if not any(i in x for i in ["x", "y", "z"]) else x
                for x in coord_string.split(",")
            ]
        )

    for element in wyckoff["letters"]:
        label = wyckoff[element]["multiplicity"] + element  # e.g. 4d
        wyckoff_coords = [
            _coord_string_to_array(coords) for coords in wyckoff[element]["coordinates"]
        ]
        wyckoff_label_coords_dict[label] = wyckoff_coords

        equivalent_sites = [
            _coord_string_to_array(coords) for coords in wyckoff["equivalent_sites"]
        ]

        new_coords = []  # new list for equivalent coordinates

        for coord_array in wyckoff_coords:
            if not all(isinstance(i, str) for i in coord_array):
                for equivalent_site in equivalent_sites:
                    # add coord_array and equivalent_site element-wise,
                    # for each element when neither is a string:
                    equiv_coord_array = coord_array.copy()
                    for idx in range(len(equiv_coord_array)):
                        if not isinstance(
                            equiv_coord_array[idx], str
                        ) and not isinstance(equivalent_site[idx], str):
                            equiv_coord_array[idx] += equivalent_site[idx]
                            equiv_coord_array[idx] = np.mod(
                                equiv_coord_array[idx], 1
                            )  # wrap to 0-1 (i.e. to unit cell)
                    new_coords.append(equiv_coord_array)

        # add new_coords to wyckoff_label_coords:
        wyckoff_label_coords_dict[label].extend(new_coords)
    return wyckoff_label_coords_dict


def get_wyckoff_label(defect_entry, wyckoff_dict=None):
    """
    Return the Wyckoff label for a defect entry's site, given a dictionary of Wyckoff labels and
    coordinates (`wyckoff_dict`). If `wyckoff_dict` is not provided, the spacegroup of the bulk
    structure is determined and used to generate it with `get_wyckoff_dict_from_sgn()`.
    """
    if wyckoff_dict is None:
        sga = SpacegroupAnalyzer(defect_entry.defect.structure)
        wyckoff_dict = get_wyckoff_dict_from_sgn(sga.get_space_group_number())

    # compare array to coord_string_to_array("x,x,x") element-wise. If an element is x, y or z, and z/y/z not set, set x/y/z to the corresponding element in the array:
    def _compare_arrays(array1, array2):
        variable_dict = {}  # dict for x,y,z
        for i, j in zip(array1, array2):
            if j in ["x", "y", "z", "-x", "-y", "-z"]:
                if j in [
                    "x",
                    "-x",
                ]:  # get x from variable dict if present, otherwise set to i:
                    x = variable_dict.get("x", i)
                    if j == "-x":
                        x = -x
                    if not np.isclose(x, i, rtol=1e-2):
                        return False
                    else:
                        variable_dict["x"] = i
                elif j in ["y", "-y"]:
                    y = variable_dict.get("y", i)
                    if j == "-y":
                        y = -y
                    if not np.isclose(y, i, rtol=1e-2):
                        return False
                    else:
                        variable_dict["y"] = i
                elif j in ["z", "-z"]:
                    z = variable_dict.get("z", i)
                    if j == "-z":
                        z = -z
                    if not np.isclose(z, i, rtol=1e-2):
                        return False
                    else:
                        variable_dict["z"] = i

            else:
                if np.isclose(i, float(j), rtol=1e-2):
                    continue
                else:
                    return False
        return True

    # get closest match of any value (coords) in wyckoff_label_coords to defect site coords:
    def find_closest_match(defect_site, wyckoff_label_coords_dict):
        # try with non-variable coordinates
        for label, coord_list in wyckoff_label_coords_dict.items():
            for coords in coord_list:
                if not any(
                    i in coords.tolist() for i in ["x", "y", "z", "-x", "-y", "-z"]
                ):
                    if np.allclose(defect_site.frac_coords, coords, rtol=1e-2):
                        return label

        # no direct match with non-variable coordinates, try with variable
        for label, coord_list in wyckoff_label_coords_dict.items():
            for coords in coord_list:
                if any(i in coords.tolist() for i in ["x", "y", "z", "-x", "-y", "-z"]):
                    if _compare_arrays(defect_site.frac_coords, coords):
                        return label

        return None  # No match found

    # Loop over the names
    defect_site = defect_entry.defect.site
    return find_closest_match(defect_site)


class DefectsGenerator:
    def __init__(
        self,
        structure: Structure,
        extrinsic: Union[str, List, Dict] = {},
        interstitial_coords: List = [],
        **kwargs,
    ):
        """
        Generates pymatgen DefectEntry objects for defects in the input host structure.
        By default, generates all intrinsic defects, but extrinsic defects (impurities)
        can also be created using the `extrinsic` argument.

        Interstitial sites are generated using Voronoi tesselation by default (found
        to be the most reliable), however these can also be manually specified using
        the `interstitial_coords` argument.

        Supercells are generated for each defect using the pymatgen `get_supercell_structure()`
        method, with `doped` default settings of `min_length = 10` (minimum supercell length
        of 10 Å) and `min_atoms = 50` (minimum 50 atoms in supercell). If a different supercell
        is desired, this can be controlled by specifying keyword arguments in DefectsGenerator(),
        which are passed to the `get_supercell_structure()` method.

        # TODO: Describe naming scheme here

        # TODO: Mention how charge states are generated, and how to modify, as shown in the example notebook.
        # Also show how to remove certain defects from the dictionary? Mightn't be worth the space for this though
        # TODO: Add option to not reduce the structure to the primitive cell, and just use the input structure as
        # the bulk supercell, in case the user doesn't want to generate the supercell with `pymatgen`. In this case,
        # warn the user that this might take a while, especially for interstitials in low-symmetry systems. Add note
        # to docs that if for some reason this is the case, the user could use a modified version of the pymatgen
        # interstitial finding tools directly along with multi-processing

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
                Voronoi tesselation.
            **kwargs:
                Keyword arguments to be passed to the `get_supercell_structure()` method.
        """
        self.defects = {}  # {defect_type: [Defect, ...]}
        self.defect_entries = {}  # {defect_name: DefectEntry}
        pbar = tqdm(total=100)  # tqdm progress bar. 100% is completion
        pbar.set_description(f"Getting primitive structure")

        # Reduce structure to primitive cell for efficient defect generation
        # same symprec as defect generators in pymatgen-analysis-defects:
        sga = SpacegroupAnalyzer(structure, symprec=1e-2)
        prim_struct = sga.get_primitive_standard_structure()
        clean_prim_struct_dict = _round_floats(prim_struct.as_dict())
        self.primitive_structure = Structure.from_dict(clean_prim_struct_dict)
        pbar.update(5)  # 5% of progress bar

        # Generate defects
        # Vacancies:
        pbar.set_description(f"Generating vacancies")
        vac_generator_obj = VacancyGenerator()
        vac_generator = vac_generator_obj.generate(self.primitive_structure)
        self.defects["vacancies"] = [vacancy for vacancy in vac_generator]
        pbar.update(10)  # 15% of progress bar

        # Antisites:
        pbar.set_description(f"Generating substitutions")
        antisite_generator_obj = AntiSiteGenerator()
        as_generator = antisite_generator_obj.generate(self.primitive_structure)
        self.defects["substitutions"] = [antisite for antisite in as_generator]
        pbar.update(10)  # 25% of progress bar

        # Substitutions:
        substitution_generator_obj = SubstitutionGenerator()
        if isinstance(extrinsic, str):  # substitute all host elements
            substitutions = {
                el.symbol: [extrinsic]
                for el in self.primitive_structure.composition.elements
            }
        elif isinstance(extrinsic, list):  # substitute all host elements
            substitutions = {
                el.symbol: extrinsic
                for el in self.primitive_structure.composition.elements
            }
        elif isinstance(extrinsic, dict):  # substitute only specified host elements
            substitutions = extrinsic
        else:
            warnings.warn(
                f"Invalid `extrinsic` defect input. Got type {type(extrinsic)}, but string or list or dict required. "
                f"No extrinsic defects will be generated."
            )
            substitutions = {}

        if substitutions:
            sub_generator = substitution_generator_obj.generate(
                self.primitive_structure, substitution=substitutions
            )
            if "substitutions" in self.defects:
                self.defects["substitutions"].extend(
                    [substitution for substitution in sub_generator]
                )
            else:
                self.defects["substitutions"] = [
                    substitution for substitution in sub_generator
                ]
        pbar.update(10)  # 35% of progress bar

        # Interstitials:
        # determine which, if any, extrinsic elements are present:
        pbar.set_description(f"Generating interstitials")
        # previous generators add oxidation states, but messes with interstitial generators, so remove oxi states:
        self.primitive_structure.remove_oxidation_states()
        if isinstance(extrinsic, str):
            extrinsic_elements = [extrinsic]
        elif isinstance(extrinsic, list):
            extrinsic_elements = extrinsic
        elif isinstance(
            extrinsic, dict
        ):  # dict of host: extrinsic elements, as lists or strings
            # convert to flattened list of extrinsic elements:
            extrinsic_elements = list(
                chain(*[i if isinstance(i, list) else [i] for i in extrinsic.values()])
            )
            extrinsic_elements = list(
                set(extrinsic_elements)
            )  # get only unique elements
        else:
            extrinsic_elements = []

        if interstitial_coords:
            # For the moment, this assumes interstitial_sites
            insertions = {
                el.symbol: interstitial_coords
                for el in self.primitive_structure.composition.elements
            }
            insertions.update({el: interstitial_coords for el in extrinsic_elements})
            interstitial_generator_obj = InterstitialGenerator()
            interstitial_generator = interstitial_generator_obj.generate(
                self.primitive_structure, insertions=insertions
            )
            self.defects["interstitials"] = [
                interstitial for interstitial in interstitial_generator
            ]

        else:
            # Generate interstitial sites using Voronoi tesselation
            voronoi_interstitial_generator_obj = VoronoiInterstitialGenerator()
            voronoi_interstitial_generator = (
                voronoi_interstitial_generator_obj.generate(
                    self.primitive_structure,
                    insert_species=[
                        el.symbol
                        for el in self.primitive_structure.composition.elements
                    ]
                    + extrinsic_elements,
                )
            )
            self.defects["interstitials"] = [
                interstitial for interstitial in voronoi_interstitial_generator
            ]

        pbar.update(
            30
        )  # 65% of progress bar, generating interstitials typically takes the longest

        # Generate supercell once, so this isn't redundantly rerun for each defect, and ensures the same supercell is
        # used for each defect and bulk calculation
        pbar.set_description(f"Generating simulation supercell")
        self.supercell_matrix = get_sc_fromstruct(
            self.primitive_structure,
            min_atoms=kwargs.get("min_atoms", 50),
            max_atoms=kwargs.get("max_atoms", 240),  # same as current pymatgen default
            min_length=kwargs.get("min_length", 10),  # same as current pymatgen default
            force_diagonal=kwargs.get(
                "force_diagonal", False
            ),  # same as current pymatgen default
        )
        pbar.update(20)  # 85% of progress bar

        # Generate DefectEntry objects:
        pbar.set_description(f"Generating DefectEntry objects")
        num_defects = sum([len(defect_list) for defect_list in self.defects.values()])

        defect_entry_list = []
        for defect_type, defect_list in self.defects.items():
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
                defect_entry_list.append(neutral_defect_entry)

        named_defect_dict = name_defect_entries(defect_entry_list)
        pbar.update(5)  # 90% of progress bar

        for defect_name_wout_charge, neutral_defect_entry in named_defect_dict.items():
            defect = neutral_defect_entry.defect
            # set defect charge states: currently from +/-1 to defect oxi state
            if defect.oxi_state > 0:
                charge_states = [
                    *range(-1, int(defect.oxi_state) + 1)
                ]  # from -1 to oxi_state
            elif defect.oxi_state < 0:
                charge_states = [
                    *range(int(defect.oxi_state), 2)
                ]  # from oxi_state to +1
            else:  # oxi_state is 0
                charge_states = [-1, 0, 1]

            for charge in charge_states:
                # TODO: Will be updated to our chosen charge generation algorithm!
                defect_entry = copy.deepcopy(neutral_defect_entry)
                defect_entry.charge_state = charge
                defect_name = (
                    defect_name_wout_charge + f"_{'+' if charge > 0 else ''}{charge}"
                )
                defect_entry.name = defect_name  # set name attribute
                self.defect_entries[defect_name] = defect_entry
                pbar.update(
                    (1 / num_defects) * (pbar.total - pbar.n)
                )  # 100% of progress bar
        pbar.close()

        print(self._defect_generator_info())

    def as_dict(self):
        """
        JSON-serializable dict representation of DefectsGenerator
        """
        json_dict = {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "defects": self.defects,
            "defect_entries": self.defect_entries,
            "primitive_structure": self.primitive_structure,
            "supercell_matrix": self.supercell_matrix,
        }

        return json_dict

    @classmethod
    def from_dict(cls, d):
        """
        Reconstructs DefectsGenerator object from a dict representation
        created using DefectsGenerator.as_dict().

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
        Returns a string with information about the defects that have been generated by the DefectsGenerator.
        """
        info_string = ""
        for defect_class, defect_list in self.defects.items():
            table = []
            header = [
                defect_class.capitalize(),
                "Charge States",
                "Unit Cell Coords",
                "\x1B[3mg\x1B[0m_site",
            ]
            defect_type = defect_list[0].defect_type
            matching_defect_types = {
                defect_entry_name: defect_entry
                for defect_entry_name, defect_entry in self.defect_entries.items()
                if defect_entry.defect.defect_type == defect_type
            }
            matching_type_names_wout_charge = list(
                set(
                    [
                        defect_entry_name.rsplit("_", 1)[0]
                        for defect_entry_name in matching_defect_types
                    ]
                )
            )
            # sort to match order of appearance of elements in the primitive structure composition:
            sorted_defect_names = sorted(
                matching_type_names_wout_charge,
                key=lambda s: [
                    s.find(sub) if s.find(sub) != -1 else float("inf")
                    for sub in [
                        el.symbol
                        for el in self.primitive_structure.composition.elements
                    ]
                ],
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
                    f"{x:.2f}" for x in neutral_defect_entry.defect.site.frac_coords
                )
                row = [
                    defect_name,
                    charges,
                    f"[{frac_coords_string}]",
                    neutral_defect_entry.defect.multiplicity,
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
            "\x1B[3mg\x1B[0m_site = Site Multiplicity (in Primitive Unit Cell)"
        )

        return info_string

    def __getattr__(self, attr):
        """
        Redirects an unknown attribute/method call to the defect_entries dictionary attribute,
        if the attribute doesn't exist in DefectsGenerator.
        """
        if hasattr(self.defect_entries, attr):
            return getattr(self.defect_entries, attr)
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'"
            )

    def __getitem__(self, key):
        """
        Makes DefectsGenerator object subscriptable, so that it can be indexed like a dictionary,
        using the defect_entries dictionary attribute.
        """
        return self.defect_entries[key]

    def __setitem__(self, key, value):
        """
        Set the value of a specific key (defect name) in the defect_entries dictionary.
        Also adds the corresponding defect to the self.defects dictionary, if it doesn't already exist.
        """
        # check the input, must be a DefectEntry object, with same supercell and primitive structure
        if not isinstance(value, DefectEntry):
            raise TypeError(
                f"Value must be a DefectEntry object, not {type(value).__name__}"
            )

        if value.defect.structure != self.primitive_structure:
            raise ValueError(
                f"Value must have the same primitive structure as the DefectsGenerator object, instead has:"
                f"{value.defect.structure}"
                f"while DefectsGenerator has:"
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
                f"Value must have the same supercell as the DefectsGenerator object, instead has:"
                f"{defect_entry.sc_entry}"
                f"while DefectsGenerator has:"
                f"{value.sc_entry}"
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
        Doesn't remove the defect from the defects dictionary attribute, as there
        may be other charge states of the same defect still present.
        """
        del self.defect_entries[key]

    def __contains__(self, key):
        """
        Returns True if the defect_entries dictionary contains the specified defect name.
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
            f"DefectsGenerator for input {repr(self.primitive_structure.composition)}, space group "
            f"{self.primitive_structure.get_space_group_info()[0]} with {len(self)} defect entries created."
        )

    def __repr__(self):
        """
        Returns a string representation of the DefectsGenerator object, and prints the DefectsGenerator info.
        """
        return self.__str__() + "\n" + self._defect_generator_info()
