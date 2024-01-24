"""
Code to analyse the Wyckoff positions of defects.

The database for Wyckoff analysis (`wyckpos.dat`) was obtained from code written by JaeHwan Shim
@schinavro (ORCID: 0000-0001-7575-4788)(https://gitlab.com/ase/ase/-/merge_requests/1035) based on the
tabulated datasets in https://github.com/xtalopt/randSpg (also found at
https://github.com/spglib/spglib/blob/develop/database/Wyckoff.csv).
"""

import os
import warnings
from typing import Any, Optional, Tuple, Union

import numpy as np
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import PeriodicSite, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import SupercellTransformation
from pymatgen.util.coord import pbc_diff
from sympy import Eq, simplify, solve, symbols

from doped.core import DefectEntry


def _round_floats(obj, places=5):
    """
    Recursively round floats in a dictionary to `places` decimal places.
    """
    if isinstance(obj, float):
        return _custom_round(obj, places) + 0.0
    if isinstance(obj, dict):
        return {k: _round_floats(v, places) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(x, places) for x in obj]
    return obj


def _custom_round(number: float, decimals: int = 3):
    """
    Custom rounding function that rounds numbers to a specified number of
    decimals, if that rounded number is within 0.15*10^(-decimals) of the
    original number, else rounds to [decimals+1] decimals.

    Primarily because float rounding with pymatgen/numpy
    can give cell coordinates of 0.5001 instead of 0.5
    etc, but also can have coordinates of e.g. 0.6125
    that should not be rounded to 0.613.

    Args:
        number (float): The number to round
        decimals (int):
            The number of decimals to round to (default: 3)

    Returns:
        float: The rounded number
    """
    rounded_number = round(number, decimals)
    if abs(rounded_number - number) < 0.15 * float(10) ** (-decimals):
        return rounded_number

    return round(number, decimals + 1)


_vectorized_custom_round = np.vectorize(_custom_round)


def _frac_coords_sort_func(coords):
    """
    Sorting function to apply on an iterable of fractional coordinates, where
    entries are sorted by the number of x, y, z that are (almost) equal (i.e.
    between 0 and 3), then by the magnitude of x+y+z, then by the magnitudes of
    x, y and z.
    """
    coords_for_sorting = _vectorized_custom_round(
        np.mod(_vectorized_custom_round(coords), 1)
    )  # to unit cell
    num_equals = sum(
        np.isclose(coords_for_sorting[i], coords_for_sorting[j], atol=1e-3)
        for i in range(len(coords_for_sorting))
        for j in range(i + 1, len(coords_for_sorting))
    )
    magnitude = _custom_round(np.linalg.norm(coords_for_sorting))
    return (-num_equals, magnitude, *np.abs(coords_for_sorting))


def _get_sga(struct, symprec=0.01):
    """
    Get a SpacegroupAnalyzer object of the input structure, dynamically
    adjusting symprec if needs be.
    """
    sga = SpacegroupAnalyzer(struct, symprec)  # default symprec of 0.01
    if sga.get_symmetry_dataset() is not None:
        return sga

    for trial_symprec in [0.1, 0.001, 1, 0.0001]:  # go one up first, then down, then criss-cross (cha cha)
        sga = SpacegroupAnalyzer(struct, symprec=trial_symprec)  # go one up first
        if sga.get_symmetry_dataset() is not None:
            return sga

    raise ValueError("Could not get SpacegroupAnalyzer object of input structure!")  # well shiiii...


def _get_all_equiv_sites(frac_coords, struct, symm_ops=None, symprec=0.01, dist_tol=0.01):
    """
    Get all equivalent sites of the input fractional coordinates in struct.
    """
    if symm_ops is None:
        sga = _get_sga(struct, symprec=symprec)
        symm_ops = sga.get_symmetry_operations()

    dummy_site = PeriodicSite("X", frac_coords, struct.lattice)
    struct_with_x = struct.copy()
    struct_with_x.sites += [dummy_site]

    x_sites = []
    for symm_op in symm_ops:
        transformed_struct = struct_with_x.copy()
        transformed_struct.apply_operation(symm_op, fractional=True)
        x_site = transformed_struct[-1].to_unit_cell()
        # if distance is >dist_tol for all other sites in x_sites, add x_site to x_sites:
        if (
            not x_sites
            or min(
                np.linalg.norm(
                    np.dot(
                        pbc_diff(np.array([site.frac_coords for site in x_sites]), x_site.frac_coords),
                        struct.lattice.matrix,
                    ),
                    axis=-1,
                )
            )
            > dist_tol
        ):
            x_sites.append(x_site)

    return x_sites


def _get_symm_dataset_of_struc_with_all_equiv_sites(
    frac_coords, struct, symm_ops=None, symprec=0.01, dist_tol=0.01
):
    unique_sites = _get_all_equiv_sites(frac_coords, struct, symm_ops, dist_tol=dist_tol)
    sga_with_all_X = _get_sga_with_all_X(struct, unique_sites, symprec=symprec)
    return sga_with_all_X.get_symmetry_dataset(), unique_sites


def _get_sga_with_all_X(struct, unique_sites, symprec=0.01):
    """
    Add all sites in unique_sites to a _copy_ of struct and return
    SpacegroupAnalyzer of this new structure.
    """
    struct_with_all_X = struct.copy()
    struct_with_all_X.sites += unique_sites
    return _get_sga(struct_with_all_X, symprec=symprec)


def _get_equiv_frac_coords_in_primitive(
    frac_coords, supercell, primitive, symm_ops=None, equiv_coords=True, dist_tol=0.01
):
    """
    Get an equivalent fractional coordinates of frac_coords in supercell, in
    the primitive cell.

    Also returns a list of equivalent fractional coords in the primitive cell
    if equiv_coords is True.
    """
    unique_sites = _get_all_equiv_sites(frac_coords, supercell, symm_ops, dist_tol=dist_tol)
    sga_with_all_X = _get_sga_with_all_X(supercell, unique_sites)

    prim_with_all_X = get_primitive_structure(sga_with_all_X, ignored_species=["X"])

    # ensure matched to primitive structure:
    rotated_struct, matrix = _rotate_and_get_supercell_matrix(prim_with_all_X, primitive)
    primitive_with_all_X = rotated_struct * matrix

    sm = StructureMatcher(primitive_cell=False, ignored_species=["X"], comparator=ElementComparator())
    s2_like_s1 = sm.get_s2_like_s1(primitive, primitive_with_all_X)
    s2_really_like_s1 = Structure.from_sites(
        [  # sometimes this get_s2_like_s1 doesn't work properly due to different (but equivalent) lattice
            PeriodicSite(  # vectors (e.g. a=(010) instead of (100) etc.), so do this to be sure
                site.specie,
                site.frac_coords,
                primitive.lattice,
                to_unit_cell=True,
            )
            for site in s2_like_s1.sites
        ]
    )

    prim_coord_list = [
        _vectorized_custom_round(np.mod(_vectorized_custom_round(site.frac_coords), 1))
        for site in s2_really_like_s1.sites
        if site.specie.symbol == "X"
    ]

    if equiv_coords:
        return (  # sort with _frac_coords_sort_func
            sorted(prim_coord_list, key=_frac_coords_sort_func)[0],
            prim_coord_list,
        )

    return sorted(prim_coord_list, key=_frac_coords_sort_func)[0]


def _rotate_and_get_supercell_matrix(prim_struct, target_struct):
    """
    Rotates the input prim_struct to match the target_struct orientation, and
    returns the supercell matrix to convert from the rotated prim_struct to the
    target_struct.
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
    )  # Transpose = inverse of rotation matrices (orthogonal matrices), better numerical stability
    output_prim_struct = prim_struct.copy()
    output_prim_struct.apply_operation(rotation_symmop)
    clean_prim_struct_dict = _round_floats(output_prim_struct.as_dict())
    return Structure.from_dict(clean_prim_struct_dict), supercell_matrix


def _get_supercell_matrix_and_possibly_rotate_prim(prim_struct, target_struct):
    """
    Determines the supercell transformation matrix to convert from the
    primitive structure to the target structure. The supercell matrix is
    defined to be T in `T*P = S` where P and S.

    are the primitive and supercell lattice matrices respectively.
    Equivalently, multiplying `prim_struct * T` will give the target_struct.

    First tries to determine a simple (integer) transformation matrix with no
    basis set rotation required. If that fails, then defaults to using
    _rotate_and_get_supercell_matrix.

    Args:
        prim_struct: pymatgen Structure object of the primitive cell.
        target_struct: pymatgen Structure object of the target cell.

    Returns:
        prim_struct: rotated primitive structure, if needed.
        supercell_matrix: supercell transformation matrix to convert from the
            primitive structure to the target structure.
    """
    try:
        # supercell transform matrix is T in `T*P = S` (P = prim, S = super), so `T = S*P^-1`:
        transformation_matrix = np.rint(
            target_struct.lattice.matrix @ np.linalg.inv(prim_struct.lattice.matrix)
        )
        if not np.allclose(
            (prim_struct * transformation_matrix).lattice.matrix,
            target_struct.lattice.matrix,
            rtol=5e-3,
        ):
            raise ValueError  # if non-integer transformation matrix

        return prim_struct, transformation_matrix

    except ValueError:  # if non-integer transformation matrix
        prim_struct, transformation_matrix = _rotate_and_get_supercell_matrix(prim_struct, target_struct)

    return prim_struct, transformation_matrix


def get_wyckoff(frac_coords, struct, symm_ops: Optional[list] = None, equiv_sites=False, symprec=0.01):
    """
    Get the Wyckoff label of the input fractional coordinates in the input
    structure. If the symmetry operations of the structure have already been
    computed, these can be input as a list to speed up the calculation.

    Args:
        frac_coords:
            Fractional coordinates of the site to get the Wyckoff label of.
        struct:
            pymatgen Structure object for which frac_coords corresponds to.
        symm_ops:
            List of pymatgen SymmOps of the structure. If None (default),
            will recompute these from the input struct.
        equiv_sites:
            If True, also returns a list of equivalent sites in struct.
        symprec:
            Symmetry precision for SpacegroupAnalyzer.
    """
    symm_dataset, unique_sites = _get_symm_dataset_of_struc_with_all_equiv_sites(
        frac_coords, struct, symm_ops, symprec=symprec
    )
    conv_cell_factor = len(symm_dataset["std_positions"]) / len(symm_dataset["wyckoffs"])
    multiplicity = int(conv_cell_factor * len(unique_sites))
    wyckoff_label = f"{multiplicity}{symm_dataset['wyckoffs'][-1]}"

    return (wyckoff_label, unique_sites) if equiv_sites else wyckoff_label


def _struc_sorting_func(struct):
    """
    Sort by the sum of the fractional coordinates, then by the magnitudes of
    high-symmetry coordinates (x=y=z, then 2 equal coordinates), then by the
    summed magnitude of all x coordinates, then y coordinates, then z
    coordinates.
    """
    struct_for_sorting = Structure.from_dict(_round_floats(struct.as_dict(), 3))

    # get summed magnitudes of x=y=z coords:
    matching_coords = struct_for_sorting.frac_coords[  # Find the coordinates where x = y = z:
        (struct_for_sorting.frac_coords[:, 0] == struct_for_sorting.frac_coords[:, 1])
        & (struct_for_sorting.frac_coords[:, 1] == struct_for_sorting.frac_coords[:, 2])
    ]
    xyz_sum_magnitudes = np.sum(np.linalg.norm(matching_coords, axis=1))

    # get summed magnitudes of x=y / y=z / x=z coords:
    matching_coords = struct_for_sorting.frac_coords[
        (struct_for_sorting.frac_coords[:, 0] == struct_for_sorting.frac_coords[:, 1])
        | (struct_for_sorting.frac_coords[:, 1] == struct_for_sorting.frac_coords[:, 2])
        | (struct_for_sorting.frac_coords[:, 0] == struct_for_sorting.frac_coords[:, 2])
    ]
    xy_sum_magnitudes = np.sum(np.linalg.norm(matching_coords, axis=1))

    return (
        np.sum(struct_for_sorting.frac_coords),
        xyz_sum_magnitudes,
        xy_sum_magnitudes,
        np.sum(struct_for_sorting.frac_coords[:, 0]),
        np.sum(struct_for_sorting.frac_coords[:, 1]),
        np.sum(struct_for_sorting.frac_coords[:, 2]),
    )


def get_primitive_structure(sga, ignored_species: Optional[list] = None):
    """
    Get a consistent/deterministic primitive structure from a
    SpacegroupAnalyzer object.

    For some materials (e.g. zinc blende), there are multiple equivalent
    primitive cells, so for reproducibility and in line with most structure
    conventions/definitions, take the one with the lowest summed norm of the
    fractional coordinates of the sites (i.e. favour Cd (0,0,0) and Te
    (0.25,0.25,0.25) over Cd (0,0,0) and Te (0.75,0.75,0.75) for F-43m CdTe).

    If ignored_species is set, then the sorting function used to determine the
    ideal primitive structure will ignore sites with species in
    ignored_species.
    """
    possible_prim_structs = []
    for _i in range(4):
        struct = sga.get_primitive_standard_structure()
        possible_prim_structs.append(struct)
        sga = _get_sga(struct, sga._symprec)  # use same symprec

    if ignored_species is not None:
        pruned_possible_prim_structs = [
            Structure.from_sites([site for site in struct if site.specie.symbol not in ignored_species])
            for struct in possible_prim_structs
        ]
    else:
        pruned_possible_prim_structs = possible_prim_structs

    # sort and return indices:
    sorted_indices = sorted(
        range(len(pruned_possible_prim_structs)),
        key=lambda i: _struc_sorting_func(pruned_possible_prim_structs[i]),
    )

    return Structure.from_dict(_round_floats(possible_prim_structs[sorted_indices[0]].as_dict()))


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
        sga = _get_sga(sga.get_primitive_standard_structure(), symprec=sga._symprec)

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
    sga = _get_sga(struc_wout_oxi)
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

    sga = _get_sga(bulk_prim_structure)
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
    conv_cell_site.frac_coords = _vectorized_custom_round(conv_cell_site.frac_coords)

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
    with open(datafile, encoding="utf-8") as f:
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
            sgn = _get_sga(defect_entry.defect.structure).get_space_group_number()

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
                # convert coords in subbed_coord_list to unit cell, by rounding to 5 decimal places and
                # then modding by 1:
                subbed_coord_list = [
                    _vectorized_custom_round(np.mod(_vectorized_custom_round(coord_array, 5), 1))
                    for coord_array in subbed_coord_list
                ]
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
    elif isinstance(spacegroup, str):
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


def point_symmetry_from_defect(defect, symm_ops=None, symprec=0.01):
    """
    Get the defect site point symmetry from a Defect object.

    Note that this is intended only to be used for unrelaxed,
    as-generated Defect objects (rather than parsed defects).

    Args:
        defect (Defect): Defect object.
        symm_ops (list):
            List of symmetry operations of defect.structure, to avoid
            re-calculating. Default is None (recalculates).
        symprec (float):
            Symmetry tolerance for spglib. Default is 0.01.

    Returns:
        str: Defect point symmetry.
    """
    symm_dataset, _unique_sites = _get_symm_dataset_of_struc_with_all_equiv_sites(
        defect.site.frac_coords, defect.structure, symm_ops=symm_ops, symprec=symprec
    )
    spglib_point_group_symbol = schoenflies_from_hermann(symm_dataset["site_symmetry_symbols"][-1])
    if spglib_point_group_symbol is not None:
        return spglib_point_group_symbol

    # symm_ops approach failed, just use diagonal defect supercell approach:
    defect_diagonal_supercell = defect.get_supercell_structure(
        sc_mat=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        dummy_species="X",
    )  # create defect supercell, which is a diagonal expansion of the unit cell so that the defect
    # periodic image retains the unit cell symmetry, in order not to affect the point group symmetry
    sga = _get_sga(defect_diagonal_supercell, symprec=symprec)
    return schoenflies_from_hermann(sga.get_point_group_symbol())


def point_symmetry_from_defect_entry(
    defect_entry: DefectEntry,
    symm_ops: Optional[list] = None,
    symprec: Optional[float] = None,
    relaxed: bool = True,
):
    r"""
    Get the defect site point symmetry from a DefectEntry object.

    Note: If relaxed = True (default), then this tries to use the
    defect_entry.defect_supercell to determine the site symmetry. This will
    thus give the _relaxed_ defect point symmetry if this is a DefectEntry
    created from parsed defect calculations. However, it should be noted
    that this is not guaranteed to work in all cases; namely for non-diagonal
    supercell expansions, or sometimes for non-scalar supercell expansion
    matrices (e.g. a 2x1x2 expansion)(particularly with high-symmetry materials)
    which can mess up the periodicity of the cell. doped tries to automatically
    check if this is the case, and will warn you if so.

    This can also be checked by using this function on your doped _generated_ defects:

    from doped.generation import get_defect_name_from_entry
    for defect_name, defect_entry in defect_gen.items():
        print(defect_name, get_defect_name_from_entry(defect_entry, relaxed=False),
              get_defect_name_from_entry(defect_entry), "\n")

    And if the point symmetries match in each case, then using this function on your
    parsed _relaxed_ DefectEntry objects should correctly determine the final relaxed
    defect symmetry - otherwise periodicity-breaking prevents this.

    If periodicity-breaking prevents auto-symmetry determination, you can manually
    determine the relaxed and unrelaxed point symmetries and/or orientational degeneracy
    from visualising the structures (e.g. using VESTA)(can use
    `get_orientational_degeneracy` to obtain the corresponding orientational degeneracy
    factor for given initial/relaxed point symmetries) and setting the corresponding
    values in the calculation_metadata['relaxed point symmetry']/['unrelaxed point
    symmetry'] and/or degeneracy_factors['orientational degeneracy'] attributes.
    The degeneracy factor is used in the calculation of defect/carrier concentrations
    and Fermi level behaviour (see e.g. doi.org/10.1039/D2FD00043A &
    doi.org/10.1039/D3CS00432E).

    Args:
        defect_entry (DefectEntry): DefectEntry object.
        symm_ops (list):
            List of symmetry operations of either the defect_entry.bulk_supercell
            structure (if relaxed=False) or defect_entry.defect_supercell (if
            relaxed=True), to avoid re-calculating. Default is None (recalculates).
        symprec (float):
            Symmetry tolerance for spglib. Default is 0.01 for unrelaxed structures,
            0.2 for relaxed (to account for residual structural noise). You may
            want to adjust for your system (e.g. if there are very slight
            octahedral distortions etc).
        relaxed (bool):
            If False, determines the site symmetry using the defect site _in the
            unrelaxed bulk supercell_, otherwise uses the defect supercell to
            determine the site symmetry (i.e. try determine the point symmetry
            of a relaxed defect in the defect supercell). Default is True.

    Returns:
        str: Defect point symmetry.
    """
    supercell = defect_entry.defect_supercell if relaxed else defect_entry.bulk_supercell
    if symprec is None:
        symprec = 0.2 if relaxed else 0.01  # relaxed structures likely have structural noise
        # May need to adjust symprec (e.g. for Ag2Se, symprec of 0.2 is acc too large as we have very
        # slight distortions present in the unrelaxed material).

    # For relaxed = True, often only works for relaxed defect structures if it is a scalar matrix
    # supercell expansion of the primitive/conventional cell (otherwise can mess up the periodicity).
    # Sometimes works even if not a scalar matrix for _low-symmetry_ systems (counter-intuitively),
    # because then the breakdown in periodicity affects the defect site symmetry less. This can be
    # checked by seeing if the site symmetry of the defect site in the unrelaxed structure is correctly
    # guessed (in which case the supercell site symmetry is not messing up the symmetry detection),
    # as shown in the docstrings. Here we test using the 'unrelaxed_defect_structure' in the DefectEntry
    # calculation_metadata if present, which, if it gives the same result as relaxed=False, means that
    # for this defect at least, there is no periodicity-breaking which is affecting the symmetry
    # determination.
    if relaxed:
        if not hasattr(defect_entry, "calculation_metadata") or not defect_entry.calculation_metadata:
            warnings.warn(
                "`relaxed` was set to True (i.e. get _relaxed_ defect symmetry), but the "
                "`calculation_metadata` attribute is not set for `DefectEntry`, suggesting that this "
                "DefectEntry was not parsed from calculations using doped. This means doped cannot "
                "automatically check if the supercell shape is breaking the cell periodicity here or not "
                "(see docstring) - the point symmetry groups may not be correct here!"
            )
        elif defect_entry.calculation_metadata.get("unrelaxed_defect_structure"):
            _matching = _check_relaxed_defect_symmetry_determination(
                defect_entry, symm_ops=symm_ops, symprec=symprec, verbose=True
            )

    _failed = False
    if defect_entry.defect_supercell_site is not None:
        try:
            symm_dataset, _unique_sites = _get_symm_dataset_of_struc_with_all_equiv_sites(
                defect_entry.defect_supercell_site.frac_coords,
                supercell,
                symm_ops=symm_ops,  # defect symm_ops needed for relaxed=True, bulk for relaxed=False
                symprec=symprec,
                dist_tol=symprec,
            )

            # Note:
            # This code works to get the site symmetry of a defect site in a periodicity-breaking
            # supercell, but only when the defect has not yet been relaxed. Still has the issue that once
            # we have relaxation around the defect site in a periodicity-breaking supercell, then the
            # (local) point symmetry cannot be easily determined as the whole supercell symmetry is broken.
            # Future work could try a local structure analysis to determine the local point symmetry to
            # counteract this.
            # unique_sites = _get_all_equiv_sites(  # defect site but bulk supercell & symm_ops
            #     defect_entry.defect_supercell_site.frac_coords, defect_entry.bulk_supercell, symm_ops
            # )
            # sga_with_all_X = _get_sga_with_all_X(  # defect unique sites but bulk supercell
            #     defect_entry.bulk_supercell, unique_sites, symprec=symprec
            # )
            # symm_dataset = sga_with_all_X.get_symmetry_dataset()
        except AttributeError:
            _failed = True

    if defect_entry.defect_supercell_site is None or _failed:
        # possibly pymatgen DefectEntry object without defect_supercell_site set
        if relaxed:
            warnings.warn(
                "Symmetry determination failed with the standard approach (likely due to this being a "
                "DefectEntry which has not been generated/parsed with doped?). Thus the _relaxed_ point "
                "group symmetry cannot be reliably automatically determined."
            )

        return point_symmetry_from_defect(defect_entry.defect, symm_ops=symm_ops, symprec=symprec)

    if not relaxed:
        # site_symmetry_symbols[-1] works better for unrelaxed defects (as sometimes with the equivalent
        # sites population it can change the overall point group symbol (but site symmetry symbol is
        # still correct))
        spglib_point_group_symbol = schoenflies_from_hermann(symm_dataset["site_symmetry_symbols"][-1])

    else:
        # For relaxed defects the "defect supercell site" is not necessarily the true centre of mass of
        # the defect (e.g. for split-interstitials, split-vacancies, swapped vacancies etc),
        # so use 'pointgroup' output (in this case the reduced symmetry avoids the symmetry-upgrade
        # possibility with the equivalent sites, as when relaxed=False)
        spglib_point_group_symbol = schoenflies_from_hermann(symm_dataset["pointgroup"])

    if spglib_point_group_symbol is not None:
        return spglib_point_group_symbol

    # symm_ops approach failed, just use diagonal defect supercell approach:
    if relaxed:
        raise RuntimeError(
            "Site symmetry could not be determined using the defect supercell, and so the relaxed site "
            "symmetry cannot be automatically determined (set relaxed=False to obtain the unrelaxed site "
            "symmetry)."
        )

    defect_diagonal_supercell = defect_entry.defect.get_supercell_structure(
        sc_mat=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        dummy_species="X",
    )  # create defect supercell, which is a diagonal expansion of the unit cell so that the defect
    # periodic image retains the unit cell symmetry, in order not to affect the point group symmetry
    sga = _get_sga(defect_diagonal_supercell, symprec=symprec)
    return schoenflies_from_hermann(sga.get_point_group_symbol())


def _check_relaxed_defect_symmetry_determination(
    defect_entry: DefectEntry,
    symm_ops: Optional[list] = None,
    symprec: float = 0.2,
    verbose: bool = False,
):
    if defect_entry.defect_supercell_site is None:
        raise AttributeError(
            "`defect_entry.defect_supercell_site` not defined! Needed to check defect supercell "
            "periodicity (for symmetry determination)"
        )
    unrelaxed_defect_structure = defect_entry.calculation_metadata.get("unrelaxed_defect_structure")
    if unrelaxed_defect_structure is not None:
        symm_dataset, _unique_sites = _get_symm_dataset_of_struc_with_all_equiv_sites(
            defect_entry.defect_supercell_site.frac_coords,
            unrelaxed_defect_structure,
            symm_ops=symm_ops,
            symprec=symprec,
            dist_tol=symprec,
        )
        unrelaxed_spglib_point_group_symbol = schoenflies_from_hermann(
            symm_dataset["site_symmetry_symbols"][-1]
        )

        symm_dataset, _unique_sites = _get_symm_dataset_of_struc_with_all_equiv_sites(
            defect_entry.defect_supercell_site.frac_coords,
            defect_entry.bulk_supercell,
            symm_ops=symm_ops,
            symprec=symprec,
            dist_tol=symprec,
        )
        bulk_spglib_point_group_symbol = schoenflies_from_hermann(
            symm_dataset["site_symmetry_symbols"][-1]
        )

        if bulk_spglib_point_group_symbol != unrelaxed_spglib_point_group_symbol:
            if verbose:
                warnings.warn(
                    "`relaxed` is set to True (i.e. get _relaxed_ defect symmetry), but doped has "
                    "detected that the defect supercell is _possibly_ a non-scalar matrix expansion which "
                    "could be breaking the cell periodicity and possibly preventing the correct _relaxed_ "
                    "point group symmetry from being automatically determined. You can set relaxed=False "
                    "to instead get the unrelaxed/initial point group symmetry, and/or manually "
                    "check/set/edit the point symmetries and corresponding orientational degeneracy "
                    "factors by inspecting/editing the "
                    "calculation_metadata['relaxed point symmetry']/['unrelaxed point symmetry'] and "
                    "degeneracy_factors['orientational degeneracy'] attributes."
                )
            return False

        return True

    return False  # return False if symmetry couldn't be checked


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

_SCH_to_HERM = {t[0]: t[1] for t in _PTG_IDS}
_HERM_to_SCH = {t[1]: t[0] for t in _PTG_IDS}
_SPGID_to_SCH = {t[2]: t[0] for t in _PTG_IDS}
_SCH_to_SPGID = {t[0]: t[2] for t in _PTG_IDS}

sch_symbols = list(_SCH_to_HERM.keys())


def schoenflies_from_hermann(herm_symbol):
    """
    Convert from Hermann-Mauguin to Schoenflies.
    """
    herm_symbol = herm_symbol.replace(".", "")
    schoenflies = _HERM_to_SCH.get(herm_symbol)
    if schoenflies is None:
        # try rearranging, symbols in spglib can be rearranged vs _HERM_to_SCH dict
        # get _HERM_to_SCH key that has the same characters as herm_symbol
        # (i.e. same characters, but possibly in a different order)
        from collections import Counter

        def find_matching_key(input_str, input_dict):
            input_str_counter = Counter(input_str)
            for key in input_dict:
                if Counter(key) == input_str_counter:
                    return key
            return None

        herm_key = find_matching_key(herm_symbol, _HERM_to_SCH)
        if herm_key is not None:
            schoenflies = _HERM_to_SCH[herm_key]

    return schoenflies


_point_group_order = {
    "C1": 1,
    "Ci": 2,  # aka. S2, -1 in Hermann-Mauguin
    "C2": 2,
    "Cs": 2,  # aka. C1h (m in Hermann-Mauguin)
    "C3": 3,
    "C4": 4,
    "S4": 4,  # C4 with improper rotation
    "C2h": 4,  # 2/m in Hermann-Mauguin
    "D2": 4,  # 222 in Hermann-Mauguin
    "C2v": 4,  # mm2 in Hermann-Mauguin
    "C3i": 6,  # aka. S6, -3 in Hermann-Mauguin
    "C6": 6,
    "C3h": 6,
    "D3": 6,  # 32 in Hermann-Mauguin
    "C3v": 6,  # 3m in Hermann-Mauguin
    "D2h": 8,  # mmm in Hermann-Mauguin
    "C4h": 8,  # 4/m in Hermann-Mauguin
    "D4": 8,  # 422 in Hermann-Mauguin
    "C4v": 8,  # 4mm in Hermann-Mauguin
    "D2d": 8,  # 42m in Hermann-Mauguin
    "C6h": 12,  # 6/m in Hermann-Mauguin
    "T": 12,  # 23 in Hermann-Mauguin
    "D3d": 12,  # 3m1 in Hermann-Mauguin
    "D6": 12,  # 622 in Hermann-Mauguin
    "C6v": 12,  # 6mm in Hermann-Mauguin
    "D3h": 12,  # 6m2 in Hermann-Mauguin
    "D4h": 16,  # 4/mmm in Hermann-Mauguin
    "D6h": 24,  # 6/mmm in Hermann-Mauguin
    "Th": 24,  # m3 in Hermann-Mauguin
    "O": 24,  # 432 in Hermann-Mauguin
    "Td": 24,  # 43m in Hermann-Mauguin
    "Oh": 48,  # m3m in Hermann-Mauguin
}


def group_order_from_schoenflies(sch_symbol):
    """
    Return the order of the point group from the Schoenflies symbol.

    Useful for symmetry and orientational degeneracy analysis.
    """
    return _point_group_order[sch_symbol]


def get_min_image_distance(structure: Structure) -> float:
    """
    Get the minimum image distance (i.e. minimum distance between periodic
    images of sites in a lattice) for the input structure.

    This is also known as the Shortest Vector Problem (SVP), and has
    no known analytical solution, requiring enumeration type approaches.
    (https://wikipedia.org/wiki/Lattice_problem#Shortest_vector_problem_(SVP))

    Args:
        structure (Structure): Structure object.

    Returns:
        float: Minimum image distance.
    """
    return _get_min_image_distance_from_matrix(structure.lattice.matrix)


def _proj(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Returns the vector projection of vector b onto vector a.

    Based on the _proj() function in
    pymatgen.transformations.advanced_transformations, but
    made significantly more efficient for looping over many
    times in optimisation functions.

    Args:
        b (np.ndarray): Vector to project.
        a (np.ndarray): Vector to project onto.

    Returns:
        np.ndarray: Vector projection of b onto a.
    """
    normalised_a = a / np.linalg.norm(a)
    return np.dot(b, normalised_a) * normalised_a


def _get_min_image_distance_from_matrix(matrix: np.ndarray) -> float:
    """
    Get the minimum image distance (i.e. minimum distance between periodic
    images of sites in a lattice) for the input lattice matrix, using the
    pymatgen get_points_in_sphere() Lattice method.

    This is also known as the Shortest Vector Problem (SVP), and has
    no known analytical solution, requiring enumeration type approaches.
    (https://wikipedia.org/wiki/Lattice_problem#Shortest_vector_problem_(SVP))

    Args:
        matrix (np.ndarray): Lattice matrix.
    """
    # Note that the max hypothetical min image distance in a 3D lattice is sixth root of 2 times the
    # effective cubic lattice parameter (i.e. the cube root of the volume), which is for HCP/FCC systems
    # while of course the minimum possible min image distance is the minimum cell vector length

    lattice = Lattice(matrix)
    eff_cubic_length = lattice.volume ** (1 / 3)
    max_min_dist = eff_cubic_length * (2 ** (1 / 6))  # max hypothetical min image distance in 3D lattice

    zipped_fcoords_dist_idx_image = lattice.get_points_in_sphere(
        [[0, 0, 0]], [0, 0, 0], r=max_min_dist * 1.01
    )

    # sort zipped list by dist:
    zipped_fcoords_dist_idx_image.sort(key=lambda x: x[1])
    min_dist = zipped_fcoords_dist_idx_image[1][1]  # second in list is min image (first is itself, zero)
    if min_dist <= 0:
        raise ValueError(
            "Minimum image distance less than or equal to zero! This is possibly due to a co-planar / "
            "non-orthogonal lattice. Please check your inputs!"
        )
    return round(  # round to 4 decimal places to avoid tiny numerical differences messing with sorting
        min_dist, 4
    )


def _get_min_image_distance_from_matrix_raw(matrix: np.ndarray, max_ijk: int = 10):
    """
    Get the minimum image distance (i.e. minimum distance between periodic
    images of sites in a lattice) for the input lattice matrix, using brute
    force numpy enumeration.

    This is also known as the Shortest Vector Problem (SVP), and has
    no known analytical solution, requiring enumeration type approaches.
    (https://wikipedia.org/wiki/Lattice_problem#Shortest_vector_problem_(SVP))

    As the cell angles deviate more from cubic (90°), the required
    max_ijk to get the correct converged result increases. For near-cubic
    systems, a max_ijk of 2 or 3 is usually sufficient.

    Args:
        matrix (np.ndarray): Lattice matrix.
        max_ijk (int):
            Maximum absolute i/j/k coefficient to allow in the search
            for the shortest (minimum image) vector: [i*a, j*b, k*c].
            (Default = 10)
    """
    # Note that the max hypothetical min image distance in a 3D lattice is sixth root of 2 times the
    # effective cubic lattice parameter (i.e. the cube root of the volume), which is for HCP/FCC systems
    # while of course the minimum possible min image distance is the minimum cell vector length
    ijk_range = np.array(range(-max_ijk, max_ijk + 1))
    i, j, k = np.meshgrid(ijk_range, ijk_range, ijk_range, indexing="ij")
    vectors = (
        i[..., np.newaxis] * matrix[0] + j[..., np.newaxis] * matrix[1] + k[..., np.newaxis] * matrix[2]
    )

    distances = np.linalg.norm(vectors, axis=-1).flatten()
    return round(  # round to 4 decimal places to avoid tiny numerical differences messing with sorting
        np.min(distances[distances > 0]), 4
    )


def _get_largest_cube_from_matrix(matrix: np.ndarray, max_ijk: int = 10):
    """
    Gets the side length of the largest possible cube that can fit in the cell
    defined by the input lattice matrix.

    As the cell angles deviate more from cubic (90°), the required
    max_ijk to get the correct converged result increases. For near-cubic
    systems, a max_ijk of 2 or 3 is usually sufficient.

    Similar to the implementation in pymatgen's CubicSupercellTransformation,
    but generalised to work for all cell shapes (e.g. needly thin cells etc),
    as the pymatgen one relies on the input cell being nearly cubic.
    E.g. gives incorrect cube size for: [[-1, -2, 0], [1, -1, 2], [1, -2, 3]]

    Args:
        matrix (np.ndarray): Lattice matrix.
        max_ijk (int):
            Maximum absolute i/j/k coefficient to allow in the search
            for the shortest cube length, using the projections along:
            [i*a, j*b, k*c].
            (Default = 10)
    """
    a = matrix[0]
    b = matrix[1]
    c = matrix[2]

    proj_ca = _proj(c, a)  # a-c plane
    proj_ac = _proj(a, c)
    proj_ba = _proj(b, a)  # b-a plane
    proj_ab = _proj(a, b)
    proj_cb = _proj(c, b)  # b-c plane
    proj_bc = _proj(b, c)

    ijk_range = np.array(range(-max_ijk, max_ijk + 1))

    # Create a grid of i, j indices
    I_vals, J_vals = np.meshgrid(ijk_range, ijk_range, indexing="ij")

    # Flatten I and J for vectorized computation
    I_flat = I_vals.flatten()
    J_flat = J_vals.flatten()

    # Include k in the vectorized computation
    K = ijk_range[ijk_range != 0][:, None, None]  # exclude cases with k=0

    # Vectorized computation for each of the three terms
    term1 = c * K - I_flat[:, None] * proj_ca - J_flat[:, None] * proj_cb
    term2 = a * K - I_flat[:, None] * proj_ac - J_flat[:, None] * proj_ab
    term3 = b * K - I_flat[:, None] * proj_ba - J_flat[:, None] * proj_bc

    # Concatenate the results and reshape
    length_vecs = np.concatenate((term1, term2, term3), axis=1).reshape(-1, 3)

    return np.min(np.linalg.norm(length_vecs, axis=1))


def cubic_cell_metric(cell_matrix: np.ndarray) -> float:
    """
    Calculates the deviation of the given cell matrix from an ideal simple
    cubic matrix, by evaluating the root mean square (RMS) difference of the
    vector lengths from that of the effective cubic structure (i.e. the cube
    root of the volume).

    This is a fixed version of the simple cubic cell metric
    in ASE (get_deviation_from_optimal_cell_shape(shape='sc')),
    described in https://wiki.fysik.dtu.dk/ase/tutorials/defects/defects.html
    which currently does not account for rotated matrices
    (e.g. a cubic cell, which should have a perfect score of 0,
    will have a bad score if its lattice vectors are rotated away
    from x, y and z, or even if they are just swapped as z, x, y).

    e.g. with ASE, [[1, 0, 0], [0, 1, 0], [0, 0, 1]] and
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]] give scores of 0 and 1,
    but with this function they both give perfect scores of 0 as
    desired.

    Args:
        cell_matrix (np.ndarray):
            Cell matrix for which to calculate the cubic cell metric.

    Returns:
        float: Cubic cell metric (0 is perfectly cubic)
    """
    eff_cubic_length = float(abs(np.linalg.det(cell_matrix)) ** (1 / 3))
    norms = np.linalg.norm(cell_matrix, axis=0)
    return round(
        np.sqrt(  # get rms difference to eff cubic
            np.sum(((norms - eff_cubic_length) / eff_cubic_length) ** 2)
        ),
        4,
    )  # round to 4 decimal places to avoid tiny numerical differences messing with sorting


def _lengths_and_angles_from_matrix(matrix: np.ndarray) -> Tuple[Any, ...]:
    lengths = tuple(np.sqrt(np.sum(matrix**2, axis=1)).tolist())
    angles = np.zeros(3)
    for dim in range(3):
        j = (dim + 1) % 3
        k = (dim + 2) % 3
        angles[dim] = np.clip(np.dot(matrix[j], matrix[k]) / (lengths[j] * lengths[k]), -1, 1)
    angles = np.arccos(angles) * 180.0 / np.pi
    angles = tuple(angles.tolist())
    return (*lengths, *angles)


def _vectorized_lengths_and_angles_from_matrices(matrices: np.ndarray) -> np.ndarray:
    """
    Vectorized version of _lengths_and_angles_from_matrix().

    Matrices is a numpy array of shape (n, 3, 3), where n is the number of
    matrices.
    """
    lengths = np.linalg.norm(matrices, axis=2)  # Compute lengths (norms of row vectors)

    angles = np.zeros((matrices.shape[0], 3))
    for dim in range(3):  # compute angles
        j = (dim + 1) % 3
        k = (dim + 2) % 3
        dot_products = np.sum(matrices[:, j, :] * matrices[:, k, :], axis=1)
        angle = np.arccos(np.clip(dot_products / (lengths[:, j] * lengths[:, k]), -1, 1))
        angles[:, dim] = np.degrees(angle)

    # Return lengths and angles, as shape matrices.shape[0] x 6
    return np.concatenate((lengths, angles), axis=1)


def _P_matrix_sorting_func(P: np.ndarray, cell: np.ndarray = None) -> tuple:
    """
    Sorting function to apply on an iterable of transformation matrices,.

    where matrices are sorted by:
    - minimum ASE style cubic-like metric
      (using the fixed, efficient doped version)
    - minimum absolute sum of elements
    - minimum number of negative elements
    - minimum largest (absolute) element
    - maximum number of x, y, z that are equal
    - maximum sum of diagonal elements.

    Args:
        P (np.ndarray): Transformation matrix.
        cell (np.ndarray): Cell matrix (on which to apply P).

    Returns:
        tuple: Tuple of sorting criteria values
    """
    cubic_metric = cubic_cell_metric(np.dot(P, cell)) if cell is not None else cubic_cell_metric(P)

    abs_P = np.abs(P)
    abs_sum = np.sum(abs_P)
    num_negs = np.sum(P < 0)
    max_abs = np.max(abs_P)
    diag_sum = np.sum(np.diag(P))
    P_flat = P.flatten()
    num_equals = sum(
        P_flat[i] == P_flat[j] for i in range(len(P_flat)) for j in range(i, len(P_flat))
    )  # double (square) counting, but doesn't matter (sorting behaviour the same)

    # Note: Initial idea was also to use cell symmetry operations to sort, but this is far too slow, and
    #  in theory should be accounted for with the other (min dist, cubic cell metric) criteria anyway.
    # struct = Structure(Lattice(P), ["H"], [[0, 0, 0]])
    # sga = _get_sga(struct)
    # symm_ops = len(sga.get_symmetry_operations())

    return (cubic_metric, abs_sum, num_negs, max_abs, -num_equals, -diag_sum)


def _lean_sort_func(P):
    abs_P = np.abs(P)
    abs_sum = np.sum(abs_P)
    num_negs = np.sum(P < 0)
    max_abs = np.max(abs_P)
    diag_sum = np.sum(np.diag(P))
    return (abs_sum, num_negs, max_abs, -diag_sum)


def _vectorized_lean_sort_func(P_batch):
    abs_P = np.abs(P_batch)
    abs_sum = np.sum(abs_P, axis=(1, 2))
    num_negs = np.sum(P_batch < 0, axis=(1, 2))
    max_abs = np.max(abs_P, axis=(1, 2))
    diag_sum = np.sum(np.diagonal(P_batch, axis1=1, axis2=2), axis=1)
    return np.stack((abs_sum, num_negs, max_abs, -diag_sum), axis=1)


def find_ideal_supercell(
    cell: np.ndarray,
    target_size: int,
    limit: int = 2,
    return_min_dist: bool = False,
    verbose: bool = False,
) -> Union[np.ndarray, tuple]:
    """
    Given an input cell matrix (e.g. Structure.lattice.matrix or Atoms.cell)
    and chosen target_size (size of supercell in number of `cell`s), finds an
    ideal supercell matrix (P) that yields the largest minimum image distance
    (i.e. minimum distance between periodic images of sites in a lattice),
    while also being as close to cubic as possible.

    Supercell matrices are searched for by first identifying the ideal
    (fractional) transformation matrix (P) that would yield a perfectly cubic
    supercell with volume equal to target_size, and then scanning over all
    matrices where the elements are within +/-`limit` of the ideal P matrix
    elements (rounded to the nearest integer).
    For relatively small target_sizes (<100) and/or cells with mostly similar
    lattice vector lengths, the default `limit` of +/-2 performs well. For
    larger `target_size`s, `cell`s with very different lattice vector lengths,
    and/or cases where small differences in minimum image distance are
    important, a larger `limit` may be required (though typically only improves
    the minimum image distance by 1-10%).

    This is also known as the Shortest Vector Problem (SVP), and has
    no known analytical solution, requiring enumeration type approaches.
    (https://wikipedia.org/wiki/Lattice_problem#Shortest_vector_problem_(SVP)),
    so can be slow for certain cases.

    Args:
        cell (np.ndarray): Unit cell matrix for which to find a supercell.
        target_size (int): Target supercell size (in number of `cell`s).
        limit (int):
            Supercell matrices are searched for by first identifying the
            ideal (fractional) transformation matrix (P) that would yield
            a perfectly cubic supercell with volume equal to target_size,
            and then scanning over all matrices where the elements are
            within +/-`limit` of the ideal P matrix elements (rounded to the
            nearest integer).
            (Default = 2)
        return_min_dist (bool):
            Whether to return the minimum image distance (in Å) as a second
            return value.
            (Default = False)
        verbose (bool): Whether to print out extra information.
            (Default = False)

    Returns:
        np.ndarray: Supercell matrix (P).
        float: Minimum image distance (in Å) if `return_min_dist` is True.
    """
    # Initial code here is based off that in ASE's find_optimal_cell_shape() function, but with significant
    # efficiency improvements, and then re-based on the minimum image distance rather than cubic cell
    # metric, then secondarily sorted by the (fixed) cubic cell metric (in doped), and then by some other
    # criteria to give the cleanest output
    target_metric = np.eye(3)  # simple cubic type target

    # Normalize cell metric to reduce computation time during looping
    norm = (target_size * np.linalg.det(cell) / np.linalg.det(target_metric)) ** (-1.0 / 3)
    norm_cell = norm * cell

    if verbose:
        print(f"Normalization factor (Q): {norm}")

    if target_size == 1:  # just identity innit
        return np.eye(3, dtype=int), _get_min_image_distance_from_matrix(
            cell
        ) if return_min_dist else np.eye(3, dtype=int)

    ideal_P = np.dot(target_metric, np.linalg.inv(norm_cell))  # Approximate initial P matrix

    if verbose:
        print("Idealized transformation matrix (ideal_P):")
        print(ideal_P)

    starting_P = np.array(np.around(ideal_P, 0), dtype=int)
    if verbose:
        print("Closest integer transformation matrix (P_0, starting_P):")
        print(starting_P)

    indices = np.indices([2 * limit + 1] * 9).reshape(9, -1).T - limit
    dP_array = indices.reshape(-1, 3, 3)
    P_array = starting_P[None, :, :] + dP_array

    # Compute determinants and filter to only those with the correct size:
    dets = np.abs(np.linalg.det(P_array))
    rounded_dets = np.around(dets, 0).astype(int)
    valid_P = P_array[rounded_dets == target_size]

    # any P in valid_P that are all negative, flip the sign of the matrix:
    valid_P[np.all(valid_P <= 0, axis=(1, 2))] *= -1

    # get unique lattices before computing metrics:
    cell_matrices = np.einsum("ijk,kl->ijl", valid_P, norm_cell)
    lengths_angles = _vectorized_lengths_and_angles_from_matrices(cell_matrices)
    # for each row in lengths_angles, get the product multiplied by the sum, as a hash:
    lengths_angles_hash = np.around(np.prod(lengths_angles, axis=1) / np.sum(lengths_angles, axis=1), 4)
    unique_hashes, indices = np.unique(lengths_angles_hash, return_index=True)
    unique_cell_matrices = cell_matrices[indices]

    if verbose:
        print(f"Searched matrices (P_array): {len(P_array)}")
        print(f"Valid matrices (matching target_size; valid_P): {len(valid_P)}")
        print(f"Unique valid matrices (unique_cell_matrices): {len(unique_cell_matrices)}")

    min_image_dists = np.array(
        [_get_min_image_distance_from_matrix(cell_matrix) for cell_matrix in unique_cell_matrices]
    )  # for near cubic systems, the min image distance in most cases is just the minimum cell vector,
    # so if the efficiency of this function was the bottleneck we could rank first with the fixed
    # cubic-cell metric, then subselect and apply this function, but at present this is not the limiting
    # factor in this function so not worth it.
    if len(min_image_dists) == 0:
        raise ValueError("No valid P matrices found with given settings")

    # get indices of min_image_dists that are equal to the minimum
    best_min_dist = np.max(min_image_dists)  # in terms of supercell effective cubic length
    if verbose:
        print(f"Best minimum image distance (best_min_dist): {best_min_dist}")

    min_dist_indices = np.where(min_image_dists == best_min_dist)[0]

    poss_P = []
    for idx in min_dist_indices:
        hash_value = unique_hashes[idx]
        matching_indices = np.where(lengths_angles_hash == hash_value)[0]
        poss_P.extend(valid_P[matching_indices])

    poss_P.sort(key=lambda x: _P_matrix_sorting_func(x, norm_cell))
    if verbose:
        print(f"Number of possible P matrices with best_min_dist (poss_P): {len(poss_P)}")

    optimal_P = poss_P[0]

    # Finalize.
    if verbose:
        print("Optimal transformation matrix (P_opt):")
        print(optimal_P)
        print("Supercell size:")
        print(np.round(np.dot(optimal_P, cell), 4))
        print(f"Minimum image distance (Å): {(best_min_dist / norm)}")

    return (optimal_P, best_min_dist / norm) if return_min_dist else optimal_P
