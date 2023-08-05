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


def get_conv_cell_site(defect_entry, lattice_vec_swap_array=None):
    """
    Get an equivalent site of the defect entry in the conventional structure of
    the host material.
    """
    from doped.generation import _frac_coords_sort_func, get_primitive_structure

    if lattice_vec_swap_array is None:
        lattice_vec_swap_array = [0, 1, 2]

    bulk_prim_structure = defect_entry.defect.structure.copy()
    bulk_prim_structure.remove_oxidation_states()  # adding oxidation states adds the
    # deprecated 'properties' attribute with -> {"spin": None}, giving a deprecation warning
    sga = SpacegroupAnalyzer(bulk_prim_structure)
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
    defect_entry=None, conv_cell_site=None, sgn=None, wyckoff_dict=None, lattice_vec_swap_array=None
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
    if lattice_vec_swap_array is None:
        lattice_vec_swap_array = [0, 1, 2]
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
