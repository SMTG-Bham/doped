"""
Utility code and functions for generating defect supercells.
"""

from itertools import permutations
from typing import Any, Optional, Union

import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation
from tqdm import tqdm

from doped.utils.symmetry import get_clean_structure, get_sga


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
    return _get_min_image_distance_from_matrix(structure.lattice.matrix)  # type: ignore


def min_dist(structure: Structure, ignored_species: Optional[list[str]] = None) -> float:
    """
    Return the minimum interatomic distance in a structure.

    Uses numpy vectorisation for fast computation.

    Args:
        structure (Structure):
            The structure to check.
        ignored_species (list[str]):
            A list of species symbols to ignore when calculating
            the minimum interatomic distance. Default is ``None``.

    Returns:
        float:
            The minimum interatomic distance in the structure.
    """
    if ignored_species is not None:
        structure = structure.copy()
        structure.remove_species(ignored_species)

    distances = structure.distance_matrix.flatten()
    return (  # fast vectorised evaluation of minimum distance
        0
        if len(np.nonzero(distances)[0]) < (len(distances) - structure.num_sites)
        else np.min(distances[np.nonzero(distances)])
    )


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


def _get_min_image_distance_from_matrix(
    matrix: np.ndarray, normalised: bool = False, break_if_less_than: Optional[float] = None
) -> Union[float, tuple[float, float]]:
    """
    Get the minimum image distance (i.e. minimum distance between periodic
    images of sites in a lattice) for the input lattice matrix, using the
    ``pymatgen`` ``get_points_in_sphere()`` ``Lattice`` method.

    This is also known as the Shortest Vector Problem (SVP), and has
    no known analytical solution, requiring enumeration type approaches.
    (https://wikipedia.org/wiki/Lattice_problem#Shortest_vector_problem_(SVP))

    Args:
        matrix (np.ndarray): Lattice matrix.
        normalised (bool):
            If the cell matrix volume is normalised (to 1). This is done in
            the ``doped`` supercell generation functions, and boosts
            efficiency by skipping volume calculation.
            Default = False.
        break_if_less_than (Optional[float]):
            If the minimum image distance is definitely less than this value
            (based on the minimum cell vector length), then return early
            with the minimum cell length and this value. Mainly for internal
            use in ``doped`` to speed up supercell generation.
            Default = None.

    Returns:
        Union[float, tuple[float, float]]: Minimum image distance, or tuple
        of minimum image distance and the break value if ``break_if_less_than``
        is not None.
    """
    # Note that the max hypothetical min image distance in a 3D lattice is sixth root of 2 times the
    # effective cubic lattice parameter (i.e. the cube root of the volume), which is for HCP/FCC systems
    # while of course the minimum possible min image distance is the minimum cell vector length
    lattice = Lattice(matrix)
    if break_if_less_than is not None:
        min_cell_length = np.min(lattice.abc)
        if min_cell_length < break_if_less_than:
            return min_cell_length, break_if_less_than

    volume = 1 if normalised else lattice.volume
    eff_cubic_length = volume ** (1 / 3)
    max_min_dist = eff_cubic_length * (2 ** (1 / 6))  # max hypothetical min image distance in 3D lattice

    _fcoords, dists, _idxs, _images = lattice.get_points_in_sphere(
        np.array([[0, 0, 0]]), [0, 0, 0], r=max_min_dist * 1.01, zip_results=False
    )
    dists.sort()
    min_dist = dists[1]  # second in list is min image (first is itself, zero)
    if min_dist <= 0:
        raise ValueError(
            "Minimum image distance less than or equal to zero! This is possibly due to a co-planar / "
            "non-orthogonal lattice. Please check your inputs!"
        )
    # round to 4 decimal places to avoid tiny numerical differences messing with sorting:
    min_dist = round(min_dist, 4)
    if break_if_less_than is not None:
        return min_dist, max(min_dist, break_if_less_than)

    return min_dist


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


def _largest_cube_length_from_matrix(matrix: np.ndarray, max_ijk: int = 10) -> float:
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

    Returns:
        float: Side length of the largest possible cube that can fit in the cell.
    """
    # Note: Not sure if this function works perfectly with odd-shaped cells...
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


def cell_metric(
    cell_matrix: np.ndarray, target: str = "SC", rms: bool = True, eff_cubic_length: Optional[float] = None
) -> float:
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
        rms (bool):
            Whether to return the `root` mean square (RMS) difference of
            the vector lengths from that of the idealised values (default),
            or just the mean square difference (to reduce computation time
            when scanning over many possible matrices).
            Default = True
        eff_cubic_length (float):
            Effective cubic length of the cell matrix (to reduce
            computation time during looping).
            Default = None

    Returns:
        float: Cell metric (0 is perfect score)
    """
    if eff_cubic_length is None:
        eff_cubic_length = np.abs(np.linalg.det(cell_matrix)) ** (1 / 3)
    norms = np.linalg.norm(cell_matrix, axis=1)

    if target.upper() == "SC":  # get rms/msd difference to eff cubic
        deviations = (norms - eff_cubic_length) / eff_cubic_length

    elif target.upper() == "FCC":
        # FCC is characterised by 60 degree angles & lattice vectors = 2**(1/6) times the eff cubic length
        eff_fcc_length = eff_cubic_length * 2 ** (1 / 6)
        deviations = (norms - eff_fcc_length) / eff_fcc_length

    else:
        raise ValueError(f"Allowed values for `target` are 'SC' or 'FCC'. Got {target}")

    msd = np.sum(deviations**2)
    # round to 4 decimal places to avoid tiny numerical differences messing with sorting:
    return round(np.sqrt(msd), 4) if rms else round(msd, 4)


def _lengths_and_angles_from_matrix(matrix: np.ndarray) -> tuple[Any, ...]:
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


def _P_matrix_sort_func(
    P: np.ndarray, cell: np.ndarray = None, eff_norm_cubic_length: Optional[float] = None
) -> tuple:
    """
    Sorting function to apply on an iterable of transformation matrices.

    Matrices are sorted by:

    - minimum ASE style cubic-like metric
      (using the fixed, efficient doped version)
    - P is diagonal?
    - lattice matrix is diagonal?
    - lattice matrix is symmetric?
    - matrix symmetry (around diagonal)
    - minimum absolute sum of elements
    - minimum absolute sum of off-diagonal elements
    - minimum number of negative elements
    - minimum largest (absolute) element
    - maximum number of x, y, z that are equal
    - maximum absolute sum of diagonal elements.
    - maximum sum of diagonal elements.

    Args:
        P (np.ndarray): Transformation matrix.
        cell (np.ndarray): Cell matrix (on which to apply P).
        eff_norm_cubic_length (float):
            Effective cubic length of the cell matrix (to reduce
            computation time during looping).

    Returns:
        tuple: Tuple of sorting criteria values
    """
    # Note: Lazy-loading _could_ make this quicker (screening out bad matrices early), if efficiency was
    # an issue for supercell generation
    transformed_cell = np.matmul(P, cell) if cell is not None else P
    cubic_metric = cell_metric(transformed_cell, rms=False, eff_cubic_length=eff_norm_cubic_length)
    abs_P = np.abs(P)
    diag_P = np.diag(P)
    abs_diag_P = np.abs(diag_P)

    abs_sum_off_diag = np.sum(abs_P - np.diag(abs_diag_P))
    abs_sum = np.sum(abs_P)
    num_negs = np.sum(P < 0)
    max_abs = np.max(abs_P)
    abs_diag_sum = np.sum(abs_diag_P)
    diag_sum = np.sum(diag_P)
    P_flat = P.flatten()
    P_flat_sorted = np.sort(P_flat)
    diffs = np.diff(P_flat_sorted)
    num_equals = np.sum(diffs == 0)
    if num_equals >= 3:  # integer matrices so can use direct comparison instead of allclose
        symmetric = P[0, 1] == P[1, 0] and P[0, 2] == P[2, 0] and P[1, 2] == P[2, 1]
        is_diagonal = False if not symmetric else P[0, 1] == 0 and P[0, 2] == 0 and P[1, 2] == 0
    else:
        symmetric = is_diagonal = False

    # Note: Initial idea was also to use cell symmetry operations to sort, but this is far too slow, and
    #  in theory should be accounted for with the other (min dist, cubic cell metric) criteria anyway.
    # struct = Structure(Lattice(P), ["H"], [[0, 0, 0]])
    # sga = get_sga(struct)
    # symm_ops = len(sga.get_symmetry_operations())
    lattice_matrix_is_symmetric = (
        np.isclose(transformed_cell[0, 1], transformed_cell[1, 0])
        and np.isclose(transformed_cell[0, 2], transformed_cell[2, 0])
        and np.isclose(transformed_cell[1, 2], transformed_cell[2, 1])
    )
    lattice_matrix_is_diagonal = (
        False
        if not lattice_matrix_is_symmetric
        else np.isclose(transformed_cell[0, 1], 0)
        and np.isclose(transformed_cell[0, 2], 0)
        and np.isclose(transformed_cell[1, 2], 0)
    )

    return (
        not is_diagonal,
        cubic_metric,
        not lattice_matrix_is_diagonal,
        not lattice_matrix_is_symmetric,
        not symmetric,
        abs_sum_off_diag,
        abs_sum,
        num_negs,
        max_abs,
        -num_equals,
        -abs_diag_sum,
        -diag_sum,
    )


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


def _fast_3x3_determinant_vectorized(matrices):
    # Apply the determinant formula for each matrix (Nx3x3)
    return (
        matrices[:, 0, 0] * (matrices[:, 1, 1] * matrices[:, 2, 2] - matrices[:, 1, 2] * matrices[:, 2, 1])
        - matrices[:, 0, 1]
        * (matrices[:, 1, 0] * matrices[:, 2, 2] - matrices[:, 1, 2] * matrices[:, 2, 0])
        + matrices[:, 0, 2]
        * (matrices[:, 1, 0] * matrices[:, 2, 1] - matrices[:, 1, 1] * matrices[:, 2, 0])
    )


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
    norm = (target_size * abs(np.linalg.det(cell)) / abs(np.linalg.det(target_metric))) ** (-1.0 / 3)
    norm_cell = norm * cell

    if verbose:
        print(f"{label} normalization factor (Q): {norm}")

    ideal_P = np.matmul(target_metric, np.linalg.inv(norm_cell))  # Approximate initial P matrix

    if verbose:
        print(f"{label} idealized transformation matrix (ideal_P):")
        print(ideal_P)

    starting_P = np.array(np.around(ideal_P, 0), dtype=int)
    if verbose:
        print(f"{label} closest integer transformation matrix (P_0, starting_P):")
        print(starting_P)

    P_array = starting_P[None, :, :] + (np.indices([2 * limit + 1] * 9).reshape(9, -1).T - limit).reshape(
        -1, 3, 3
    )  # combined transformation functions to reduce memory demand, only having one big P array

    # Compute determinants and filter to only those with the correct size:
    dets = np.abs(_fast_3x3_determinant_vectorized(P_array))
    valid_P = P_array[np.around(dets, 0).astype(int) == target_size]

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


def _check_and_return_scalar_matrix(P, cell=None):
    """
    Check if the input transformation matrix (``P``) is equivalent to a scalar
    matrix (multiple of the identity matrix), and return the scalar matrix if
    so.
    """
    eigenvalues = np.abs(np.linalg.eigvals(P))
    if np.allclose(eigenvalues, eigenvalues[0], atol=1e-4):
        scalar_P = np.eye(3) * eigenvalues[0]
        if cell is None:
            return scalar_P

        # otherwise check if the min image distance is the same
        if np.isclose(
            _get_min_image_distance_from_matrix(np.matmul(P, cell)),
            _get_min_image_distance_from_matrix(np.matmul(scalar_P, cell)),
        ):
            P = scalar_P

    return P


def _get_optimal_P(
    valid_P, selected_indices, unique_hashes, lengths_angles_hash, norm_cell, verbose, label, cell
):
    """
    Get the optimal/cleanest P matrix from the given valid_P array (with
    provided set of grouped unique matrices), according to the
    ``_P_matrix_sort_func``.
    """
    poss_P = []
    for idx in selected_indices:
        hash_value = unique_hashes[idx]
        matching_indices = np.where(lengths_angles_hash == hash_value)[0]
        poss_P.extend(valid_P[matching_indices])

    eff_norm_cubic_length = Lattice(np.matmul(next(iter(poss_P)), norm_cell)).volume ** (1 / 3)
    poss_P.sort(key=lambda x: _P_matrix_sort_func(x, norm_cell, eff_norm_cubic_length))
    if verbose:
        print(f"{label} number of possible P matrices with best score (poss_P): {len(poss_P)}")

    optimal_P = poss_P[0]
    # check if P is equivalent to a scalar multiple of the identity matrix
    optimal_P = _check_and_return_scalar_matrix(optimal_P, cell)

    # Finalize.
    if verbose:
        print(f"{label} optimal transformation matrix (P_opt):")
        print(optimal_P)
        print(f"{label} supercell size:")
        print(np.round(np.matmul(optimal_P, cell), 4))

    return optimal_P


def _min_sum_off_diagonals(prim_struct: Structure, supercell_matrix: np.ndarray):
    """
    Get the minimum absolute sum of off-diagonal elements in the given
    supercell matrix (for the primitive structure), or the corresponding
    supercell matrix for the conventional structure (of ``prim_struct``).

    Used to determine if we have an ideal supercell matrix (i.e. a diagonal
    transformation matrix of either the primitive or conventional cells).

    Args:
        prim_struct (Structure): Primitive structure.
        supercell_matrix (np.ndarray): Supercell matrix to check.

    Returns:
        int: Minimum absolute sum of off-diagonal elements, for the
             primitive or conventional supercell matrix.
    """
    num_off_diagonals_prim = np.sum(np.abs(supercell_matrix - np.diag(np.diag(supercell_matrix))))

    sga = get_sga(prim_struct)
    conv_supercell_matrix = np.matmul(
        supercell_matrix, sga.get_conventional_to_primitive_transformation_matrix()
    )
    num_off_diagonals_conv = np.sum(
        np.abs(conv_supercell_matrix - np.diag(np.diag(conv_supercell_matrix)))
    )

    return min(num_off_diagonals_prim, num_off_diagonals_conv)


def find_ideal_supercell(
    cell: np.ndarray,
    target_size: int,
    limit: int = 2,
    clean: bool = True,
    return_min_dist: bool = False,
    verbose: bool = False,
) -> Union[np.ndarray, tuple]:
    r"""
    Given an input cell matrix (e.g. Structure.lattice.matrix or Atoms.cell)
    and chosen target_size (size of supercell in number of ``cell``\s), finds
    an ideal supercell matrix (P) that yields the largest minimum image
    distance (i.e. minimum distance between periodic images of sites in a
    lattice), while also being as close to cubic as possible.

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

    Note that this function is used by default to generate defect supercells with
    the ``doped`` ``DefectsGenerator`` class, unless specific supercell settings
    are used.

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
        clean (bool):
            Whether to return the supercell matrix which gives the 'cleanest'
            supercell (according to `_lattice_matrix_sort_func`; most
            symmetric, with mostly positive diagonals and c >= b >= a).
            (Default = True)
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
        return np.eye(3, dtype=int), (
            _get_min_image_distance_from_matrix(cell) if return_min_dist else np.eye(3, dtype=int)
        )

    # Initial code here is based off that in ASE's find_optimal_cell_shape() function, but with significant
    # efficiency improvements, and then re-based on the minimum image distance rather than cubic cell
    # metric, then secondarily sorted by the (fixed) cubic cell metric (in doped), and then by some other
    # criteria to give the cleanest output
    sc_target_metric = np.eye(3)  # simple cubic type target

    a = [0, 1, 1]
    b = [1, 0, 1]
    c = [1, 1, 0]  # get FCC metric which aligns best with input cell:
    fcc_target_metrics = [0.5 * np.array(perm, dtype=float) for perm in permutations([a, b, c])]
    fcc_target_metric = sorted(fcc_target_metrics, key=lambda x: -np.abs(np.linalg.norm(x * cell)))[0]

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

        current_best_min_image_distance = 0.001
        min_dists = []
        # for near cubic systems, the min image distance in most cases is just the minimum cell vector,
        # so if the efficiency of this function was the bottleneck we could rank first with the fixed
        # cubic-cell metric, then subselect and apply this function, but at present this is not the
        # limiting factor in this function so not worth it
        for cell_matrix in unique_cell_matrices:
            min_dist, current_best_min_image_distance = _get_min_image_distance_from_matrix(
                cell_matrix, normalised=True, break_if_less_than=current_best_min_image_distance
            )  # type: ignore
            min_dists.append(min_dist)

        min_image_dists = np.array(min_dists)
        if len(min_image_dists) == 0:
            raise ValueError("No valid P matrices found with given settings")

        # get indices of min_image_dists that are equal to the minimum
        best_min_dist = np.max(min_image_dists)  # in terms of supercell effective cubic length
        if verbose:
            print(f"{label} best minimum image distance (best_min_dist): {best_min_dist}")

        min_dist_indices = np.where(min_image_dists == best_min_dist)[0]

        return _get_optimal_P(
            valid_P=valid_P,
            selected_indices=min_dist_indices,
            unique_hashes=unique_hashes,
            lengths_angles_hash=lengths_angles_hash,
            norm_cell=norm_cell,
            verbose=verbose,
            label=label,
            cell=cell,
        )

    sc_optimal_P = _find_ideal_supercell_for_target_metric(
        cell=cell,
        target_size=target_size,
        limit=limit,
        verbose=verbose,
        target_metric=sc_target_metric,
        label="SC",
    )  # tested and found that amalgamating SC/FCC target matrices earlier leads to massive slowdown,
    # so more efficient to just generate both this way and compare
    fcc_optimal_P = _find_ideal_supercell_for_target_metric(
        cell=cell,
        target_size=target_size,
        limit=limit,
        verbose=verbose,
        target_metric=fcc_target_metric,
        label="FCC",
    )
    # recalculate min dists (reduces numerical errors inherited from transformations)
    sc_min_dist = round(
        _get_min_image_distance_from_matrix(np.matmul(sc_optimal_P, cell)), 3  # type: ignore
    )
    fcc_min_dist = round(
        _get_min_image_distance_from_matrix(np.matmul(fcc_optimal_P, cell)), 3  # type: ignore
    )

    sc_fcc_P_and_min_dists = [
        (sc_optimal_P, sc_min_dist),
        (fcc_optimal_P, fcc_min_dist),
    ]
    sc_fcc_P_and_min_dists.sort(
        key=lambda x: (-x[1], _P_matrix_sort_func(x[0], cell))
    )  # sort by max min dist, then by sorting func

    optimal_P, min_dist = sc_fcc_P_and_min_dists[0]

    if clean and not (
        optimal_P[0, 0] != 0 and np.allclose(np.abs(optimal_P / optimal_P[0, 0]), np.eye(3))
    ):
        # only try cleaning if it's not a perfect scalar expansion
        supercell = Structure(Lattice(cell), ["H"], [[0, 0, 0]]) * optimal_P
        clean_supercell, T = get_clean_structure(supercell, return_T=True)  # T maps orig to clean_super
        # T*orig = clean -> orig = T^-1*clean
        # optimal_P was: P*cell = orig -> T*P*cell = clean -> P' = T*P

        optimal_P = np.matmul(T, optimal_P)

        # if negative cell determinant, swap lattice vectors to get a positive determinant (as this can
        # cause issues with VASP, and results in POSCAR lattice matrix changes), picking that with the best
        # score according to the sorting function:
        if np.linalg.det(clean_supercell.lattice.matrix) < 0:
            swap_combo_score_dict = {}
            for swap_combo in permutations([0, 1, 2], 2):
                swapped_P = np.copy(optimal_P)
                swapped_P[swap_combo[0]], swapped_P[swap_combo[1]] = (
                    swapped_P[swap_combo[1]],
                    swapped_P[swap_combo[0]].copy(),
                )
                swap_combo_score_dict[swap_combo] = _P_matrix_sort_func(swapped_P, cell)
            best_swap_combo = min(swap_combo_score_dict, key=lambda x: swap_combo_score_dict[x])
            optimal_P[best_swap_combo[0]], optimal_P[best_swap_combo[1]] = (
                optimal_P[best_swap_combo[1]],
                optimal_P[best_swap_combo[0]].copy(),
            )

    return (optimal_P, min_dist) if return_min_dist else optimal_P


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

    score_list = [
        cell_metric(cell_matrix, target=target_shape, rms=False) for cell_matrix in unique_cell_matrices
    ]
    best_msd = np.min(score_list)
    if verbose:
        print(f"Best score: {np.sqrt(best_msd)}")

    best_score_indices = np.where(np.array(score_list) == best_msd)[0]

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

    return (optimal_P, np.sqrt(best_msd)) if return_score else optimal_P
