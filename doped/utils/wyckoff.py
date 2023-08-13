"""
Code to analyse the Wyckoff positions of defects.

The database for Wyckoff analysis (`wyckpos.dat`) was obtained from code written by JaeHwan Shim
@schinavro (ORCID: 0000-0001-7575-4788)(https://gitlab.com/ase/ase/-/merge_requests/1035) based on the
tabulated datasets in https://github.com/xtalopt/randSpg (also found at
https://github.com/spglib/spglib/blob/develop/database/Wyckoff.csv).
"""

import os

import numpy as np
from ase.utils import basestring
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core.structure import PeriodicSite, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import SupercellTransformation
from sympy import Eq, simplify, solve, symbols


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


def _struc_sorting_func(struct):
    """
    Sort by the sum of the fractional coordinates, then by the magnitudes of
    high-symmetry coordinates (x=y=z, then 2 equal coordinates), then by the
    summed magnitude of all x coordinates, then y coordinates, then z
    coordinates.
    """
    # get summed magnitudes of x=y=z coords:
    matching_coords = struct.frac_coords[  # Find the coordinates where x = y = z:
        (struct.frac_coords[:, 0] == struct.frac_coords[:, 1])
        & (struct.frac_coords[:, 1] == struct.frac_coords[:, 2])
    ]
    xyz_sum_magnitudes = np.sum(np.linalg.norm(matching_coords, axis=1))

    # get summed magnitudes of x=y / y=z / x=z coords:
    matching_coords = struct.frac_coords[
        (struct.frac_coords[:, 0] == struct.frac_coords[:, 1])
        | (struct.frac_coords[:, 1] == struct.frac_coords[:, 2])
        | (struct.frac_coords[:, 0] == struct.frac_coords[:, 2])
    ]
    xy_sum_magnitudes = np.sum(np.linalg.norm(matching_coords, axis=1))

    return (
        np.sum(struct.frac_coords),
        xyz_sum_magnitudes,
        xy_sum_magnitudes,
        np.sum(struct.frac_coords[:, 0]),
        np.sum(struct.frac_coords[:, 1]),
        np.sum(struct.frac_coords[:, 2]),
    )


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
    for _i in range(4):
        struct = sga.get_primitive_standard_structure()
        possible_prim_structs.append(struct)
        sga = SpacegroupAnalyzer(struct, symprec=1e-2)

    possible_prim_structs = sorted(possible_prim_structs, key=_struc_sorting_func)
    return Structure.from_dict(_round_floats(possible_prim_structs[0].as_dict()))


def get_spglib_conv_structure(sga):
    """
    Get a consistent/deterministic conventional structure from a
    SpacegroupAnalyzer object. Also returns the corresponding
    SpacegroupAnalyzer (for getting Wyckoff symbols corresponding to this
    conventional structure definition).

    For some materials (e.g. zinc blende), there are multiple equivalent
    primitive/conventional cells, so for reproducibility and in line with most
    structure conventions/definitions, take the one with the lowest summed norm
    of the fractional coordinates of the sites (i.e. favour Cd (0,0,0) and Te
    (0.25,0.25,0.25) over Cd (0,0,0) and Te (0.75,0.75,0.75) for F-43m CdTe;
    SGN 216).
    """
    possible_conv_structs_and_sgas = []
    for _i in range(4):
        struct = sga.get_conventional_standard_structure()
        possible_conv_structs_and_sgas.append((struct, sga))
        sga = SpacegroupAnalyzer(sga.get_primitive_standard_structure(), symprec=1e-2)

    possible_conv_structs_and_sgas = sorted(
        possible_conv_structs_and_sgas, key=lambda x: _struc_sorting_func(x[0])
    )
    return (
        Structure.from_dict(_round_floats(possible_conv_structs_and_sgas[0][0].as_dict())),
        possible_conv_structs_and_sgas[0][1],
    )


def get_BCS_conventional_structure(structure, pbar=None, return_wyckoff_dict=False):
    """
    Get the conventional crystal structure of the input structure, according to
    the Bilbao Crystallographic Server (BCS) definition. Also returns the
    transformation matrix from the spglib (SpaceGroupAnalyzer) conventional
    structure definition to the BCS definition.

    Args:
        structure (Structure): pymatgen Structure object for this to
            get the corresponding BCS conventional crystal structure
        pbar (ProgressBar): tqdm progress bar object, to update progress.
        return_wyckoff_dict (bool): whether to return the Wyckoff label
            dict ({Wyckoff label: coordinates})
    number.

    Returns:
        pymatgen Structure object and spglib -> BCS conv cell transformation matrix.
    """
    struc_wout_oxi = structure.copy()
    struc_wout_oxi.remove_oxidation_states()
    sga = SpacegroupAnalyzer(struc_wout_oxi, symprec=1e-2)
    conventional_structure, conv_sga = get_spglib_conv_structure(sga)

    wyckoff_label_dict = get_wyckoff_dict_from_sgn(conv_sga.get_space_group_number())
    # determine cell orientation for Wyckoff site determination (needs to match the Bilbao
    # Crystallographic Server's convention, which can differ from spglib (pymatgen) in some cases)

    sga_wyckoffs = conv_sga.get_symmetrized_structure().wyckoff_symbols

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

        reoriented_conv_structure = swap_axes(conventional_structure, trial_lattice_vec_swap_array)
        if _compare_wyckoffs(
            sga_wyckoffs,
            reoriented_conv_structure,
            wyckoff_label_dict,
        ):
            lattice_vec_swap_array = trial_lattice_vec_swap_array
            break

        if pbar is not None:
            pbar.update(1 / 6 * 10)  # 45 up to 55% of progress bar in DefectsGenerator. This part can
            # take a little while for low-symmetry structures

    if return_wyckoff_dict:
        return (
            swap_axes(conventional_structure, lattice_vec_swap_array),
            lattice_vec_swap_array,
            wyckoff_label_dict,
        )

    return (
        swap_axes(conventional_structure, lattice_vec_swap_array),
        lattice_vec_swap_array,
    )


def get_conv_cell_site(defect_entry):
    """
    Gets an equivalent site of the defect entry in the conventional structure
    of the host material. If the conventional_structure attribute is not
    defined for defect_entry, then it is generated using SpaceGroupAnalyzer and
    then reoriented to match the Bilbao Crystallographic Server's conventional
    structure definition.

    Args:
        defect_entry: DefectEntry object.
    """
    bulk_prim_structure = defect_entry.defect.structure.copy()
    bulk_prim_structure.remove_oxidation_states()  # adding oxidation states adds the
    # # deprecated 'properties' attribute with -> {"spin": None}, giving a deprecation warning

    prim_struct_with_X = defect_entry.defect.structure.copy()
    prim_struct_with_X.remove_oxidation_states()
    prim_struct_with_X.append("X", defect_entry.defect.site.frac_coords, coords_are_cartesian=False)

    sga = SpacegroupAnalyzer(bulk_prim_structure)
    # convert to match sga primitive structure first:
    sm = StructureMatcher(primitive_cell=False, ignored_species=["X"], comparator=ElementComparator())
    sga_prim_struct = sga.get_primitive_standard_structure()
    s2_like_s1 = sm.get_s2_like_s1(sga_prim_struct, prim_struct_with_X)
    s2_really_like_s1 = Structure.from_sites(
        [  # sometimes this get_s2_like_s1 doesn't work properly due to different (but equivalent) lattice
            PeriodicSite(  # vectors (e.g. a=(010) instead of (100) etc.), so do this to be sure
                site.specie,
                site.frac_coords,
                sga_prim_struct.lattice,
                to_unit_cell=True,
            )
            for site in s2_like_s1.sites
        ]
    )

    conv_struct_with_X = s2_really_like_s1 * np.linalg.inv(
        sga.get_conventional_to_primitive_transformation_matrix()
    )

    # convert to match defect_entry conventional structure definition
    s2_like_s1 = sm.get_s2_like_s1(defect_entry.conventional_structure, conv_struct_with_X)
    s2_really_like_s1 = Structure.from_sites(
        [  # sometimes this get_s2_like_s1 doesn't work properly due to different (but equivalent) lattice
            PeriodicSite(  # vectors (e.g. a=(010) instead of (100) etc.), so do this to be sure
                site.specie,
                site.frac_coords,
                defect_entry.conventional_structure.lattice,
                to_unit_cell=True,
            )
            for site in s2_like_s1.sites
        ]
    )

    conv_cell_site = [site for site in s2_really_like_s1.sites if site.specie.symbol == "X"][0]
    # site choice doesn't matter so much here, as we later get the equivalent coordinates using the
    # Wyckoff dict and choose the conventional site based on that anyway (in the DefectsGenerator
    # initialisation)
    conv_cell_site.to_unit_cell()
    conv_cell_site.frac_coords = np.round(conv_cell_site.frac_coords, 5)

    return conv_cell_site


def swap_axes(structure, axes):
    """
    Swap axes of the given structure.

    The new order of the axes is given by the axes parameter. For example,
    axes=(2, 1, 0) will swap the first and third axes.
    """
    transformation_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    if axes != [0, 1, 2]:
        raise ValueError(f"Caught one!: {structure}")  # TODO: Remove!

    for i, axis in enumerate(axes):
        transformation_matrix[i][axis] = 1

    transformation = SupercellTransformation(transformation_matrix)

    return transformation.apply_transformation(structure)


def get_wyckoff_dict_from_sgn(sgn):
    """
    Get dictionary of {Wyckoff label: coordinates} for a given space group
    number.
    """
    datafile = _get_wyckoff_datafile()
    with open(datafile) as f:
        wyckoff = _read_wyckoff_datafile(sgn, f)

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


def get_wyckoff_label_and_equiv_coord_list(
    defect_entry=None, conv_cell_site=None, sgn=None, wyckoff_dict=None
):
    """
    Return the Wyckoff label and list of equivalent fractional coordinates
    within the conventional cell for the input defect_entry or conv_cell_site
    (whichever is provided, defaults to defect_entry if both), given a
    dictionary of Wyckoff labels and coordinates (`wyckoff_dict`).

    If `wyckoff_dict` is not provided, it is generated from the spacegroup
    number (sgn) using `get_wyckoff_dict_from_sgn(sgn)`. If `sgn` is not
    provided, it is obtained from the bulk structure of the `defect_entry` if
    provided.
    """
    if wyckoff_dict is None:
        if sgn is None:
            if defect_entry is None:
                raise ValueError(
                    "If inputting `conv_cell_site` and not `defect_entry`, either `sgn` or `wyckoff_dict` "
                    "must be provided."
                )
            # get sgn from primitive unit cell of bulk structure:
            sgn = SpacegroupAnalyzer(defect_entry.defect.structure).get_space_group_number()

        wyckoff_dict = get_wyckoff_dict_from_sgn(sgn)

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
        temp_dict = {}
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
        conv_cell_site = get_conv_cell_site(defect_entry)

    return find_closest_match(conv_cell_site, wyckoff_dict)


def _compare_wyckoffs(wyckoff_symbols, conv_struct, wyckoff_dict):
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
            conv_cell_site=site, wyckoff_dict=wyckoff_dict
        )
        if all(
            # allow for sga conventional cell (and thus wyckoffs) being a multiple of BCS conventional cell
            wyckoff_label not in wyckoff_symbol_list
            for wyckoff_symbol_list in wyckoff_symbol_lists
        ) and all(
            # allow for BCS conv cell (and thus wyckoffs) being a multiple of sga conv cell (allow it fam)
            multiplied_wyckoff_symbol not in wyckoff_symbols
            for multiplied_wyckoff_symbol in [
                _multiply_wyckoffs([wyckoff_label], n=n)[0] for n in range(1, 5)  # up to 4x
            ]
        ):
            return False  # break on first non-match
        doped_wyckoffs.append(wyckoff_label)

    return any(
        # allow for sga conventional cell (and thus wyckoffs) being a multiple of BCS conventional cell
        set(i) == set(doped_wyckoffs)
        for i in wyckoff_symbol_lists
    ) or any(
        set(i) == set(wyckoff_symbols)
        for i in [
            # allow for BCS conv cell (and thus wyckoffs) being a multiple of sga conv cell (allow it fam)
            _multiply_wyckoffs(doped_wyckoffs, n=n)
            for n in range(1, 5)  # up to 4x
        ]
    )  # False if no complete match, True otherwise


def _read_wyckoff_datafile(spacegroup, f, setting=None):
    """
    Read the `wyckpos.dat` file of specific spacegroup and returns a dictionary
    with this information.
    """
    if isinstance(spacegroup, int):
        pass
    elif isinstance(spacegroup, basestring):
        spacegroup = " ".join(spacegroup.strip().split())
    else:
        raise ValueError("`spacegroup` must be of type int or str")

    line = _skip_to_spacegroup(f, spacegroup, setting)
    wyckoff_dict = {"letters": [], "multiplicity": [], "number_of_letters": 0}
    line_list = line.split()
    if line_list[0].isdigit():
        wyckoff_dict["spacegroup"] = int(line_list[0])
    else:
        spacegroup, wyckoff_dict["setting"] = line_list[0].split("-")
        wyckoff_dict["spacegroup"] = int(spacegroup)
    if len(line.split()) > 1:
        eq_sites = line.split("(")[1:]
        wyckoff_dict["equivalent_sites"] = ([eq[:-1] for eq in eq_sites])[1:]
        wyckoff_dict["equivalent_sites"][-1] = wyckoff_dict["equivalent_sites"][-1][:-1]

    while True:
        line = f.readline()
        if line == "\n":
            break
        letter, multiplicity = line.split()[:2]
        coordinates_raw = line.split()[-1].split("(")[1:]
        site_symmetry = "".join(line.split()[2:-1])
        wyckoff_dict["letters"].append(letter)
        wyckoff_dict["number_of_letters"] += 1
        wyckoff_dict["multiplicity"].append(int(multiplicity))
        coordinates = [coord[:-1] for coord in coordinates_raw]
        wyckoff_dict[letter] = {
            "multiplicity": multiplicity,
            "site_symmetry": site_symmetry,
            "coordinates": coordinates,
        }

    return wyckoff_dict


def _get_wyckoff_datafile():
    """
    Return default path to Wyckoff datafile.
    """
    return os.path.join(os.path.dirname(__file__), "wyckpos.dat")


def _skip_to_spacegroup(f, spacegroup, setting=None):
    """
    Read lines from f until a blank line is encountered.
    """
    name = str(spacegroup) if setting is None else f"{spacegroup!s}-{setting}"
    while True:
        line = f.readline()
        if not line:
            raise ValueError(
                f"Invalid spacegroup {spacegroup} with setting: {setting}. Not found in the Wyckoff "
                f"database!"
            )
        if line.startswith(name):
            break
    return line
