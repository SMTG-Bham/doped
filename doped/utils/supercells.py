"""
Utility code and functions for generating defect supercells.
"""

from typing import Any, Optional, Tuple, Union

import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation
from tqdm import tqdm


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


def cell_metric(cell_matrix: np.ndarray, target: str = "SC") -> float:
    """
    Calculates the deviation of the given cell matrix from an ideal simple
    cubic (if target = "SC") or face-centred cubic (if target = "FCC") matrix,
    by evaluating the root mean square (RMS) difference of the vector lengths
    from that of the idealised values (i.e. the corresponding SC/FCC lattice
    vector lengths for the given cell volume).

    For target = "SC", the idealised lattice vector length is the effective
    cubic length (i.e. the cube root of the volume), while for "FCC" it is
    2^(1/6) (~1.12) times the effective cubic length.
    This is a fixed version of the cell metric function in ASE
    (``get_deviation_from_optimal_cell_shape``),
    described in https://wiki.fysik.dtu.dk/ase/tutorials/defects/defects.html
    which currently does not account for rotated matrices
    (e.g. a cubic cell with target = "SC", which should have a perfect score of 0,
    will have a bad score if its lattice vectors are rotated away
    from x, y and z, or even if they are just swapped as z, x, y).
    e.g. with ASE, [[1, 0, 0], [0, 1, 0], [0, 0, 1]] and
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]] give scores of 0 and 1,
    but with this function they both give perfect scores of 0 as
    desired.

    Args:
        cell_matrix (np.ndarray):
            Cell matrix for which to calculate the cell metric.
        target (str):
            Target cell shape, for which to calculate the normalised
            deviation score from. Either "SC" for simple cubic or
            "FCC" for face-centred cubic.
            Default = "SC"

    Returns:
        float: Cell metric (0 is perfect score)
    """
    eff_cubic_length = float(abs(np.linalg.det(cell_matrix)) ** (1 / 3))
    norms = np.linalg.norm(cell_matrix, axis=0)

    if target.upper() == "SC":
        return round(
            np.sqrt(  # get rms difference to eff cubic
                np.sum(((norms - eff_cubic_length) / eff_cubic_length) ** 2)
            ),
            4,
        )  # round to 4 decimal places to avoid tiny numerical differences messing with sorting

    if target.upper() != "FCC":
        raise ValueError(f"Allowed values for `target` are 'SC' or 'FCC'. Got {target}")

    # FCC is characterised by 60 degree angles & lattice vectors = 2**(1/6) times the eff cubic length
    eff_fcc_length = eff_cubic_length * 2 ** (1 / 6)
    return round(
        np.sqrt(  # get rms difference to eff cubic
            np.sum(((norms - eff_fcc_length) / eff_fcc_length) ** 2)
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
    cubic_metric = cell_metric(np.dot(P, cell)) if cell is not None else cell_metric(P)

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


def _get_candidate_P_arrays(
    cell: np.ndarray,
    target_size: int,
    limit: int = 2,
    verbose: bool = False,
    target_metric: Optional[np.ndarray] = None,
    label="SC",
) -> tuple:
    """
    Get the possible supercell transformation (P) matrices for the given cell,
    target_size, limit and target_metric, and also determine the unique
    matrices based on the transformed cell lengths and angles.
    """
    if target_metric is None:
        target_metric = np.eye(3)  # SC by default

    # Normalize cell metric to reduce computation time during looping
    norm = (target_size * np.linalg.det(cell) / np.linalg.det(target_metric)) ** (-1.0 / 3)
    norm_cell = norm * cell

    if verbose:
        print(f"{label} normalization factor (Q): {norm}")

    ideal_P = np.dot(target_metric, np.linalg.inv(norm_cell))  # Approximate initial P matrix

    if verbose:
        print(f"{label} idealized transformation matrix (ideal_P):")
        print(ideal_P)

    starting_P = np.array(np.around(ideal_P, 0), dtype=int)
    if verbose:
        print(f"{label} closest integer transformation matrix (P_0, starting_P):")
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
        print(f"{label} searched matrices (P_array): {len(P_array)}")
        print(f"{label} valid matrices (matching target_size; valid_P): {len(valid_P)}")
        print(f"{label} unique valid matrices (unique_cell_matrices): {len(unique_cell_matrices)}")

    return valid_P, norm, norm_cell, unique_cell_matrices, unique_hashes, lengths_angles_hash


def _get_optimal_P(
    valid_P, selected_indices, unique_hashes, lengths_angles_hash, norm_cell, verbose, label, cell
):
    """
    Get the optimal/cleanest P matrix from the given valid_P array (with
    provided set of grouped unique matrices), according to the
    _P_matrix_sorting_func.
    """
    poss_P = []
    for idx in selected_indices:
        hash_value = unique_hashes[idx]
        matching_indices = np.where(lengths_angles_hash == hash_value)[0]
        poss_P.extend(valid_P[matching_indices])

    poss_P.sort(key=lambda x: _P_matrix_sorting_func(x, norm_cell))
    if verbose:
        print(f"{label} number of possible P matrices with best score (poss_P): {len(poss_P)}")

    optimal_P = poss_P[0]

    # Finalize.
    if verbose:
        print(f"{label} optimal transformation matrix (P_opt):")
        print(optimal_P)
        print(f"{label} supercell size:")
        print(np.round(np.dot(optimal_P, cell), 4))

    return optimal_P


def find_ideal_supercell(
    cell: np.ndarray,
    target_size: int,
    limit: int = 2,
    return_min_dist: bool = False,
    verbose: bool = False,
) -> Union[np.ndarray, tuple]:
    r"""
    Given an input cell matrix (e.g. Structure.lattice.matrix or Atoms.cell)
    and chosen target_size (size of supercell in number of ``cell``\s), finds an
    ideal supercell matrix (P) that yields the largest minimum image distance
    (i.e. minimum distance between periodic images of sites in a lattice),
    while also being as close to cubic as possible.

    Supercell matrices are searched for by first identifying the ideal
    (fractional) transformation matrix (P) that would yield a perfectly cubic
    supercell with volume equal to target_size, and then scanning over all
    matrices where the elements are within +/-``limit`` of the ideal P matrix
    elements (rounded to the nearest integer).
    For relatively small target_sizes (<100) and/or cells with mostly similar
    lattice vector lengths, the default ``limit`` of +/-2 performs very well. For
    larger ``target_size``\s, ``cell``\s with very different lattice vector lengths,
    and/or cases where small differences in minimum image distance are very
    important, a larger ``limit`` may be required (though typically only improves
    the minimum image distance by 1-6%).

    This is also known as the Shortest Vector Problem (SVP), and has
    no known analytical solution, requiring enumeration type approaches.
    (https://wikipedia.org/wiki/Lattice_problem#Shortest_vector_problem_(SVP)),
    so can be slow for certain cases.

    Args:
        cell (np.ndarray): Unit cell matrix for which to find a supercell.
        target_size (int): Target supercell size (in number of ``cell``\s).
        limit (int):
            Supercell matrices are searched for by first identifying the
            ideal (fractional) transformation matrix (P) that would yield
            a perfectly SC/FCC supercell with volume equal to target_size,
            and then scanning over all matrices where the elements are
            within +/-``limit`` of the ideal P matrix elements (rounded to the
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
        float: Minimum image distance (in Å) if ``return_min_dist`` is True.
    """
    if target_size == 1:  # just identity innit
        return np.eye(3, dtype=int), _get_min_image_distance_from_matrix(
            cell
        ) if return_min_dist else np.eye(3, dtype=int)

    # Initial code here is based off that in ASE's find_optimal_cell_shape() function, but with significant
    # efficiency improvements, and then re-based on the minimum image distance rather than cubic cell
    # metric, then secondarily sorted by the (fixed) cubic cell metric (in doped), and then by some other
    # criteria to give the cleanest output
    sc_target_metric = np.eye(3)  # simple cubic type target
    fcc_target_metric = 0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)

    def _find_ideal_supercell_for_target_metric(
        cell: np.ndarray,
        target_size: int,
        limit: int = 2,
        verbose: bool = False,
        target_metric: np.ndarray = sc_target_metric,
        label="SC",
    ):
        (
            valid_P,
            norm,
            norm_cell,
            unique_cell_matrices,
            unique_hashes,
            lengths_angles_hash,
        ) = _get_candidate_P_arrays(
            cell=cell,
            target_size=target_size,
            limit=limit,
            verbose=verbose,
            target_metric=target_metric,
            label=label,
        )

        min_image_dists = np.array(
            [_get_min_image_distance_from_matrix(cell_matrix) for cell_matrix in unique_cell_matrices]
        )  # for near cubic systems, the min image distance in most cases is just the minimum cell vector,
        # so if the efficiency of this function was the bottleneck we could rank first with the fixed
        # cubic-cell metric, then subselect and apply this function, but at present this is not the
        # limiting factor in this function so not worth it.
        if len(min_image_dists) == 0:
            raise ValueError("No valid P matrices found with given settings")

        # get indices of min_image_dists that are equal to the minimum
        best_min_dist = np.max(min_image_dists)  # in terms of supercell effective cubic length
        if verbose:
            print(f"{label} best minimum image distance (best_min_dist): {best_min_dist}")

        min_dist_indices = np.where(min_image_dists == best_min_dist)[0]

        optimal_P = _get_optimal_P(
            valid_P=valid_P,
            selected_indices=min_dist_indices,
            unique_hashes=unique_hashes,
            lengths_angles_hash=lengths_angles_hash,
            norm_cell=norm_cell,
            verbose=verbose,
            label=label,
            cell=cell,
        )

        if verbose:
            print(f"{label} minimum image distance (Å): {(best_min_dist / norm)}")

        return (optimal_P, best_min_dist / norm)

    sc_optimal_P, sc_min_dist = _find_ideal_supercell_for_target_metric(
        cell=cell,
        target_size=target_size,
        limit=limit,
        verbose=verbose,
        target_metric=sc_target_metric,
        label="SC",
    )  # tested and found that amalgamating SC/FCC target matrices earlier leads to massive slowdown,
    # so more efficient to just generate both this way and compare
    fcc_optimal_P, fcc_min_dist = _find_ideal_supercell_for_target_metric(
        cell=cell,
        target_size=target_size,
        limit=limit,
        verbose=verbose,
        target_metric=fcc_target_metric,
        label="FCC",
    )

    if sc_min_dist > fcc_min_dist:
        return (sc_optimal_P, sc_min_dist) if return_min_dist else sc_optimal_P

    return (fcc_optimal_P, fcc_min_dist) if return_min_dist else fcc_optimal_P


def get_pmg_cubic_supercell_dict(struct: Structure, uc_range: tuple = (1, 200)) -> dict:
    """
    Get a dictionary of (near-)cubic supercell matrices for the given structure
    and range of numbers of unit cells (in the supercell).

    Returns a dictionary of format:

    .. code-block:: python

        {Number of Unit Cells:
            {"P": transformation matrix,
             "min_dist": minimum image distance}
        }

    for (near-)cubic supercells generated by the pymatgen
    CubicSupercellTransformation class. If a (near-)cubic
    supercell cannot be found for a given number of unit
    cells, then the corresponding dict value will be set
    to an empty dict.

    Args:
        struct (Structure):
            pymatgen Structure object to generate supercells for
        uc_range (tuple):
            Range of numbers of unit cells to search over

    Returns:
        dict of:
        ``{Number of Unit Cells: {"P": transformation matrix, "min_dist": minimum image distance}}``
    """
    pmg_supercell_dict = {}
    prim_min_dist = get_min_image_distance(struct)

    for i in tqdm(range(*uc_range)):
        cst = CubicSupercellTransformation(
            min_atoms=i * len(struct),
            max_atoms=i * len(struct),
            min_length=prim_min_dist,
            force_diagonal=False,
        )
        try:
            supercell = cst.apply_transformation(struct)
            pmg_supercell_dict[i] = {
                "P": cst.transformation_matrix,
                "min_dist": get_min_image_distance(supercell),
            }
        except Exception:
            pmg_supercell_dict[i] = {}

    return pmg_supercell_dict


def find_optimal_cell_shape(
    cell: np.ndarray,
    target_size: int,
    target_shape: str = "SC",
    limit: int = 2,
    return_score: bool = False,
    verbose: bool = False,
) -> Union[np.ndarray, tuple]:
    r"""
    Find the transformation matrix that produces a supercell corresponding to
    *target_size* unit cells that most closely approximates the shape defined
    by *target_shape*.

    This is an updated version of ASE's find_optimal_cell_shape() function, but
    fixed to be rotationally-invariant (explained below), with significant
    efficiency improvements, and then secondarily sorted by the (fixed) cell
    metric (in doped), and then by some other criteria to give the cleanest
    output.

    Finds the optimal supercell transformation matrix by calculating the deviation
    of the possible supercell matrices from an ideal simple cubic (if target = "SC")
    or face-centred cubic (if target = "FCC") matrix - and then taking that with the
    best (lowest) score, by evaluating the root mean square (RMS) difference of the
    vector lengths from that of the idealised values (i.e. the corresponding SC/FCC
    lattice vector lengths for the given cell volume).

    For target = "SC", the idealised lattice vector length is the effective
    cubic length (i.e. the cube root of the volume), while for "FCC" it is
    2^(1/6) (~1.12) times the effective cubic length.
    The ``get_deviation_from_optimal_cell_shape`` function in ASE -
    described in https://wiki.fysik.dtu.dk/ase/tutorials/defects/defects.html -
    currently does not account for rotated matrices
    (e.g. a cubic cell with target = "SC", which should have a perfect score of 0,
    will have a bad score if its lattice vectors are rotated away
    from x, y and z, or even if they are just swapped as z, x, y).
    e.g. with ASE, [[1, 0, 0], [0, 1, 0], [0, 0, 1]] and
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]] give scores of 0 and 1,
    but with this function they both give perfect scores of 0 as
    desired.


    Args:
        cell (np.ndarray):
            Unit cell matrix for which to find a supercell transformation.
        target_size (int): Target supercell size (in number of ``cell``\s).
        target_shape (str):
            Target cell shape, for which to calculate the normalised
            deviation score from. Either "SC" for simple cubic or
            "FCC" for face-centred cubic.
            Default = "SC"
        limit (int):
            Supercell matrices are searched for by first identifying the
            ideal (fractional) transformation matrix (P) that would yield
            a perfectly SC/FCC supercell with volume equal to target_size,
            and then scanning over all matrices where the elements are
            within +/-``limit`` of the ideal P matrix elements (rounded to the
            nearest integer).
            (Default = 2)
        return_score (bool):
            Whether to return the cell metric score as a second return value.
            (Default = False)
        verbose (bool): Whether to print out extra information.
            (Default = False)

    Returns:
        np.ndarray: Supercell matrix (P).
        float: Cell metric (0 is perfect score) if ``return_score`` is True.
    """
    # Set up target metric
    if target_shape.lower() in {"sc", "simple-cubic"}:
        target_shape = "SC"
        target_metric = np.eye(3)
    elif target_shape.lower() in {"fcc", "face-centered cubic"}:
        target_shape = "FCC"
        target_metric = 0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)

    (
        valid_P,
        norm,
        norm_cell,
        unique_cell_matrices,
        unique_hashes,
        lengths_angles_hash,
    ) = _get_candidate_P_arrays(
        cell=cell,
        target_size=target_size,
        limit=limit,
        verbose=verbose,
        target_metric=target_metric,
        label=target_shape,
    )

    score_list = [cell_metric(cell_matrix, target=target_shape) for cell_matrix in unique_cell_matrices]
    best_score = np.min(score_list)
    if verbose:
        print(f"Best score: {best_score}")

    best_score_indices = np.where(np.array(score_list) == best_score)[0]

    optimal_P = _get_optimal_P(
        valid_P=valid_P,
        selected_indices=best_score_indices,
        unique_hashes=unique_hashes,
        lengths_angles_hash=lengths_angles_hash,
        norm_cell=norm_cell,
        verbose=verbose,
        label=target_shape,
        cell=cell,
    )

    if verbose:
        print(f"Score: {best_score}")

    return (optimal_P, best_score) if return_score else optimal_P
