"""
Utility code and functions for generating & analysing defect supercells.
"""

from functools import lru_cache
from itertools import permutations
from typing import Any

import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation
from tqdm import tqdm


def get_min_image_distance(structure: Structure) -> float:
    """
    Get the minimum image distance (i.e. minimum distance between periodic
    images of sites in a lattice) for the input structure.

    This is also known as the Shortest Vector Problem (SVP), and has no known
    analytical solution, requiring enumeration type approaches.
    https://wikipedia.org/wiki/Lattice_problem#Shortest_vector_problem_%28SVP%29

    Args:
        structure (|Structure|): |Structure| object.

    Returns:
        float: Minimum image distance.
    """
    return _get_min_image_distance_from_matrix(structure.lattice.matrix)


def min_dist(structure: Structure, ignored_species: list[str] | None = None) -> float:
    """
    Return the minimum interatomic distance in a structure (ignoring any zero
    distances).

    Uses ``numpy`` vectorisation for fast computation.

    Args:
        structure (|Structure|):
            The structure to check.
        ignored_species (list[str]):
            A list of species symbols to ignore when calculating the minimum
            interatomic distance. Default is ``None`` (don't ignore any
            species).

    Returns:
        float:
            The minimum interatomic distance in the structure.
    """
    if ignored_species is not None:
        structure = structure.copy()
        structure.remove_species(ignored_species)

    distances = structure.distance_matrix.flatten()
    nonzero_dists = np.nonzero(distances)[0]

    if len(nonzero_dists) == 0:  # likely single-site structure
        return get_min_image_distance(structure)

    return (  # fast vectorised evaluation of minimum distance
        0
        if len(nonzero_dists) < (len(distances) - structure.num_sites)
        else np.min(distances[nonzero_dists])
    )


def _proj(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Returns the vector projection of vector b onto vector a.

    Based on the ``_proj()`` function in
    ``pymatgen.transformations.advanced_transformations``, but made
    significantly more efficient for looping over many times in optimisation
    functions.

    Args:
        b (np.ndarray): Vector to project.
        a (np.ndarray): Vector to project onto.

    Returns:
        np.ndarray: Vector projection of b onto a.
    """
    normalised_a = a / np.linalg.norm(a)
    return np.dot(b, normalised_a) * normalised_a


def _get_min_image_distance_from_matrix(
    matrix: np.ndarray,
    normalised: bool = False,
) -> float:
    """
    Get the minimum image distance (i.e. minimum distance between periodic
    images of sites in a lattice) for the input lattice matrix, using the
    ``pymatgen`` ``get_points_in_sphere()`` |Lattice| method.

    This is also known as the Shortest Vector Problem (SVP), and has no known
    analytical solution, requiring enumeration type approaches.
    https://wikipedia.org/wiki/Lattice_problem#Shortest_vector_problem_%28SVP%29

    Args:
        matrix (np.ndarray): Lattice matrix.
        normalised (bool):
            If the cell matrix volume is normalised (to 1). This is done in the
            ``doped`` supercell generation functions, and boosts efficiency by
            skipping volume calculation. Default = False.

    Returns:
        float: Minimum image distance.
    """
    # Note that the max hypothetical min image distance in a 3D lattice is sixth root of 2 times the
    # effective cubic lattice parameter (i.e. the cube root of the volume), which is for HCP/FCC systems,
    # which is also the cell vector length. In near-cubic cells, the minimum image distance is typically
    # approximately equal to the minimum cell vector length. So, the max possible min image distance is
    # typically in the range: ``(~0.8*min_cell_length, min_cell_length]``, for near-cubic cells
    # (see Figure 1; doped JOSS)
    lattice = Lattice(matrix)
    if normalised:
        max_min_dist = 2 ** (1 / 6)
    else:
        volume = lattice.volume
        eff_cubic_length = volume ** (1 / 3)
        max_min_dist = eff_cubic_length * 2 ** (1 / 6)  # max hypothetical min image distance in 3D lattice

    _fcoords, dists, _idxs, _images = lattice.get_points_in_sphere(
        np.array([[0, 0, 0]]), [0, 0, 0], r=max_min_dist * 1.01, zip_results=False
    )
    dists = np.array(dists)
    min_dist = np.min(dists[dists > 0])  # second in list is min image (first is itself, zero)
    if min_dist <= 0:
        raise ValueError(
            "Minimum image distance less than or equal to zero! This is possibly due to a co-planar / "
            "non-orthogonal lattice. Please check your inputs!"
        )

    return round(min_dist, 4)  # round to 4 decimal places to avoid issues with tiny numerical differences


def _get_min_image_distance_from_matrix_raw(matrix: np.ndarray, max_ijk: int = 10) -> float:
    """
    Get the minimum image distance (i.e. minimum distance between periodic
    images of sites in a lattice) for the input lattice matrix, using brute
    force numpy enumeration.

    This is also known as the Shortest Vector Problem (SVP), and has no known
    analytical solution, requiring enumeration type approaches.
    https://wikipedia.org/wiki/Lattice_problem#Shortest_vector_problem_%28SVP%29

    As the cell angles deviate more from cubic (90°), the required
    ``max_ijk`` to get the correct converged result increases. For near-cubic
    systems, a ``max_ijk`` of 2 or 3 is usually sufficient.

    Args:
        matrix (np.ndarray): Lattice matrix.
        max_ijk (int):
            Maximum absolute i/j/k coefficient to allow in the search for the
            shortest (minimum image) vector: ``[i*a, j*b, k*c]``. Default = 10.

    Returns:
        float: Minimum image distance.
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
    r"""
    Gets the side length of the largest possible cube that can fit in the cell
    defined by the input lattice matrix.

    As the cell angles deviate more from cubic (90°), the required ``max_ijk``
    to get the correct converged result increases. For near-cubic systems, a
    ``max_ijk`` of 2 or 3 is usually sufficient.

    Similar to the implementation in ``pymatgen``\'s
    ``CubicSupercellTransformation``, but generalised to work for all cell
    shapes (e.g. needly thin cells etc), as the ``pymatgen`` one relies on the
    input cell being nearly cubic. E.g. gives incorrect cube size for:
    ``[[-1, -2, 0], [1, -1, 2], [1, -2, 3]]``.

    Args:
        matrix (np.ndarray): Lattice matrix.
        max_ijk (int):
            Maximum absolute i/j/k coefficient to allow in the search for the
            shortest cube length, using the projections along:
            ``[i*a, j*b, k*c]``. Default = 10.

    Returns:
        float:
            Side length of the largest possible cube that can fit in the cell.
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
    cell_matrix: np.ndarray, target: str = "SC", rms: bool = True, eff_cubic_length: float | None = None
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

    This is an expanded version of the cell metric function in ASE
    (``get_deviation_from_optimal_cell_shape``), described in
    https://ase-lib.org/examples_generated/tutorials/defects.html
    which previously did not account for rotational invariance (now fixed;
    https://gitlab.com/ase/ase/-/merge_requests/3404,
    https://gitlab.com/ase/ase/-/merge_requests/3616).


    Args:
        cell_matrix (np.ndarray):
            Cell matrix for which to calculate the cell metric.
        target (str):
            Target cell shape, for which to calculate the normalised deviation
            score from. Either "SC" for simple cubic or "FCC" for face-centred
            cubic. Default = "SC"
        rms (bool):
            Whether to return the `root` mean square (RMS) difference of the
            vector lengths from that of the idealised values (default), or just
            the mean square difference (to reduce computation time when
            scanning over many possible matrices). Default = True
        eff_cubic_length (float):
            Effective cubic length of the cell matrix (to reduce computation
            time during looping). Default = None

    Returns:
        float: Cell metric (0 is perfect score).
    """
    # Note that ``eval_length_deviation`` and ``eval_shape_deviation`` from ASE >=3.25 also now implement
    # this functionality
    if eff_cubic_length is None:
        eff_cubic_length = np.abs(np.linalg.det(cell_matrix)) ** (1 / 3)
    norms = np.linalg.norm(cell_matrix, axis=1)

    if eff_cubic_length == 0:
        raise ValueError("Effective cubic length is zero; cannot compute cell metric.")

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
    return (*lengths, *tuple(angles.tolist()))


def _vectorized_lengths_and_angles_from_matrices(matrices: np.ndarray) -> np.ndarray:
    """
    Vectorized version of _lengths_and_angles_from_matrix().

    Matrices is a numpy array of shape (n, 3, 3), where n is the number of
    matrices.

    No longer used, superseded by better Gram matrix based approach, for
    determining rotationally-invariant unique cell matrix descriptors.
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
    P: np.ndarray,
    cell: np.ndarray | None = None,
    eff_norm_cubic_length: float | None = None,
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
            Effective cubic length of the cell matrix (to reduce computation
            time during looping).

    Returns:
        tuple: Tuple of sorting criteria values.
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


def _argmin_p_matrix_sort(P_batch: np.ndarray, cell: np.ndarray, eff: float) -> int:
    """
    Index of the best ``P`` in ``P_batch`` under the same ordering as
    ``_P_matrix_sort_func(P, cell, eff)``, without a Python loop (vectorised).
    """
    P_batch = np.asarray(P_batch)
    transformed = P_batch @ cell
    norms = np.linalg.norm(transformed, axis=2)
    d = norms / eff - 1.0
    cubic_metric = np.round(np.sum(d * d, axis=1), 4)

    abs_P = np.abs(P_batch)
    abs_sum = np.sum(abs_P, axis=(1, 2))
    diag_P = np.diagonal(P_batch, axis1=1, axis2=2)
    abs_diag_sum = np.sum(np.abs(diag_P), axis=1)
    abs_sum_off_diag = abs_sum - abs_diag_sum
    diag_sum = np.sum(diag_P, axis=1)
    num_negs = np.sum(P_batch < 0, axis=(1, 2))
    max_abs = np.max(abs_P, axis=(1, 2))

    P_sorted = np.sort(P_batch.reshape(-1, 9), axis=1)
    num_equals = np.sum(np.diff(P_sorted, axis=1) == 0, axis=1)

    sym_m = (
        (P_batch[:, 0, 1] == P_batch[:, 1, 0])
        & (P_batch[:, 0, 2] == P_batch[:, 2, 0])
        & (P_batch[:, 1, 2] == P_batch[:, 2, 1])
    )
    diag_m = sym_m & (P_batch[:, 0, 1] == 0) & (P_batch[:, 0, 2] == 0) & (P_batch[:, 1, 2] == 0)
    ge3 = num_equals >= 3
    symmetric = np.where(ge3, sym_m, False)
    is_diagonal = np.where(ge3, diag_m, False)

    t = transformed
    lat_sym = (
        np.isclose(t[:, 0, 1], t[:, 1, 0])
        & np.isclose(t[:, 0, 2], t[:, 2, 0])
        & np.isclose(t[:, 1, 2], t[:, 2, 1])
    )
    lat_diag = lat_sym & np.isclose(t[:, 0, 1], 0) & np.isclose(t[:, 0, 2], 0) & np.isclose(t[:, 1, 2], 0)

    not_is_diag = (~is_diagonal).astype(np.int8)
    not_lat_diag = (~lat_diag).astype(np.int8)
    not_lat_sym = (~lat_sym).astype(np.int8)
    not_sym = (~symmetric).astype(np.int8)

    order = np.lexsort(
        (
            -diag_sum.astype(np.float64),
            -abs_diag_sum.astype(np.float64),
            -num_equals.astype(np.float64),
            max_abs.astype(np.float64),
            num_negs.astype(np.float64),
            abs_sum.astype(np.float64),
            abs_sum_off_diag.astype(np.float64),
            not_sym,
            not_lat_sym,
            not_lat_diag,
            cubic_metric.astype(np.float64),
            not_is_diag,
        )
    )
    return int(order[0])


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
    target_metric: np.ndarray | None = None,
    target_shape="SC",
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
        print(f"{target_shape} normalization factor (Q): {norm}")

    ideal_P = np.matmul(target_metric, np.linalg.inv(norm_cell))  # Approximate initial P matrix

    if verbose:
        print(f"{target_shape} idealized transformation matrix (ideal_P):")
        print(ideal_P)

    starting_P = np.array(np.around(ideal_P, 0), dtype=int)
    if verbose:
        print(f"{target_shape} closest integer transformation matrix (P_0, starting_P):")
        print(starting_P)

    P_array = starting_P[None, :, :] + _p_matrix_offsets_grid(limit)
    # combined transformation functions to reduce memory demand, only having one big P array

    # Compute determinants and filter to only those with the correct size:
    dets = np.abs(_fast_3x3_determinant_vectorized(P_array))
    valid_P = P_array[np.around(dets, 0).astype(int) == target_size]

    # any P in valid_P that are all negative, flip the sign of the matrix:
    valid_P[np.all(valid_P <= 0, axis=(1, 2))] *= -1

    # get unique lattices before computing metrics (batched matmul, uses BLAS rather than ``np.einsum``):
    cell_matrices = valid_P @ norm_cell

    lengths_angles = _vectorized_lengths_and_angles_from_matrices(cell_matrices)
    # for each row in lengths_angles, get the product multiplied by the sum, as a hash:
    lengths_angles_hash = np.around(np.prod(lengths_angles, axis=1) / np.sum(lengths_angles, axis=1), 4)
    unique_hashes, indices = np.unique(lengths_angles_hash, return_index=True)
    unique_cell_matrices = cell_matrices[indices]

    if verbose:
        print(f"{target_shape} searched matrices (P_array): {len(P_array)}")
        print(f"{target_shape} valid matrices (matching target_size; valid_P): {len(valid_P)}")
        print(f"{target_shape} unique valid matrices (unique_cell_matrices): {len(unique_cell_matrices)}")

    return valid_P, norm_cell, unique_cell_matrices, unique_hashes, lengths_angles_hash


@lru_cache(maxsize=4)
def _p_matrix_offsets_grid(limit: int) -> np.ndarray:
    """
    All integer offset matrices with elements in ``[-limit, +limit]``, as a
    ``((2*limit+1)^9, 3, 3)`` array; cached as it is constant for a given
    ``limit`` (and somewhat expensive to construct; ~4M element array).
    """
    return ((np.indices([2 * limit + 1] * 9).reshape(9, -1).T - limit).reshape(-1, 3, 3)).astype(
        np.int8
    )  # int8 to reduce cached memory footprint (elements are small); upcast on addition


def _check_and_return_scalar_matrix(P, cell=None):
    """
    Check if the input transformation matrix (``P``) is equivalent to a scalar
    matrix (multiple of the identity matrix), and return the scalar matrix if
    so.
    """
    scalar_P = np.eye(3) * P[0, 0]
    if np.allclose(P, scalar_P, atol=1e-4):
        if cell is None:
            return scalar_P

        # otherwise check if the min image distance is the same
        if np.isclose(
            _get_min_image_distance_from_matrix(np.matmul(P, cell)),
            _get_min_image_distance_from_matrix(np.matmul(scalar_P, cell)),
            atol=1e-4,
        ):
            P = scalar_P

    return P


def _get_optimal_P(
    valid_P, selected_indices, unique_hashes, lengths_angles_hash, norm_cell, verbose, target_shape, cell
):
    """
    Get the optimal/cleanest P matrix from the given valid_P array (with
    provided set of grouped unique matrices), according to the
    ``_P_matrix_sort_func``.
    """
    # collect all valid P matrices whose cell shape (lengths and angles) matches any of the selected unique
    # shapes (based on their minimum image distances):
    selected_hashes = unique_hashes[selected_indices]
    poss_P = valid_P[np.isin(lengths_angles_hash, selected_hashes)]

    eff_norm_cubic_length = Lattice(np.matmul(next(iter(poss_P)), norm_cell)).volume ** (1 / 3)
    if verbose:
        print(f"{target_shape} number of possible P matrices with best score (poss_P): {len(poss_P)}")

    optimal_P = poss_P[_argmin_p_matrix_sort(poss_P, norm_cell, eff_norm_cubic_length)]

    # check if P is equivalent to a scalar multiple of the identity matrix
    optimal_P = _check_and_return_scalar_matrix(optimal_P, cell)

    # Finalize.
    if verbose:
        print(f"{target_shape} optimal transformation matrix (P_opt):")
        print(optimal_P)
        print(f"{target_shape} supercell size:")
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
        prim_struct (|Structure|): Primitive structure.
        supercell_matrix (np.ndarray): Supercell matrix to check.

    Returns:
        int:
            Minimum absolute sum of off-diagonal elements, for the primitive or
            conventional supercell matrix.
    """
    num_off_diagonals_prim = np.sum(np.abs(supercell_matrix - np.diag(np.diag(supercell_matrix))))

    from doped.utils.symmetry import get_sga  # avoid circular import

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
) -> np.ndarray | tuple[np.ndarray, float]:
    r"""
    Given an input cell matrix (e.g. ``Structure.lattice.matrix`` or
    ``Atoms.cell``) and chosen ``target_size`` (size of supercell in number of
    ``cell``\s), finds an ideal supercell matrix (P) that yields the largest
    minimum image distance (i.e. minimum distance between periodic images of
    sites in a lattice), while also being as close to cubic as possible.

    Supercell matrices are searched for by first identifying the ideal
    (fractional) transformation matrix (P) that would yield a perfectly cubic
    supercell with volume equal to ``target_size``, and then scanning over all
    matrices where the elements are within +/-``limit`` of the ideal P matrix
    elements (rounded to the nearest integer). For relatively small
    ``target_size``\s (<100) and/or cells with mostly similar lattice vector
    lengths, the default ``limit`` of +/-2 performs very well. For larger
    ``target_size``\s, ``cell``\s with very different lattice vector lengths,
    and/or cases where small differences in minimum image distance are very
    important, a larger ``limit`` may be required (though typically only
    improves the minimum image distance by 1-6%).

    This is also known as the Shortest Vector Problem (SVP), and has no known
    analytical solution, requiring enumeration type approaches.
    https://wikipedia.org/wiki/Lattice_problem#Shortest_vector_problem_%28SVP%29

    Note that this function is used by default to generate defect supercells
    with the ``doped`` |DefectsGenerator| class, unless specific supercell
    settings are used.

    Args:
        cell (np.ndarray): Unit cell matrix for which to find a supercell.
        target_size (int): Target supercell size (in number of ``cell``\s).
        limit (int):
            Supercell matrices are searched for by first identifying the ideal
            (fractional) transformation matrix (P) that would yield a perfectly
            SC/FCC supercell with volume equal to ``target_size``, and then
            scanning over all matrices where the elements are within
            +/-``limit`` of the ideal P matrix elements (rounded to the nearest
            integer). (Default = 2)
        clean (bool):
            Whether to return the supercell matrix which gives the 'cleanest'
            supercell (according to `_lattice_matrix_sort_func`; most
            symmetric, with mostly positive diagonals and c >= b >= a).
            (Default = True)
        return_min_dist (bool):
            Whether to return the minimum image distance (in Å) as a second
            return value. (Default = False)
        verbose (bool):
            Whether to print out extra information about the supercell search.
            (Default = False)

    Returns:
        np.ndarray | tuple[np.ndarray, float]:
            The supercell transformation matrix (P), and if ``return_min_dist``
            is ``True``, the minimum image distance (in Å).
    """
    if target_size == 1:  # just identity innit
        identity = np.eye(3, dtype=int)
        return (identity, _get_min_image_distance_from_matrix(cell)) if return_min_dist else identity

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

    sc_optimal_P = _find_ideal_supercell_for_target_metric(
        cell=cell,
        target_size=target_size,
        limit=limit,
        verbose=verbose,
        target_metric=sc_target_metric,
        target_shape="SC",
    )  # tested and found that amalgamating SC/FCC target matrices earlier leads to massive slowdown,
    # so more efficient to just generate both this way and compare
    fcc_optimal_P = _find_ideal_supercell_for_target_metric(
        cell=cell,
        target_size=target_size,
        limit=limit,
        verbose=verbose,
        target_metric=fcc_target_metric,
        target_shape="FCC",
    )
    # recalculate min dists (reduces numerical errors inherited from transformations)
    sc_min_dist = round(_get_min_image_distance_from_matrix(np.matmul(sc_optimal_P, cell)), 3)
    fcc_min_dist = round(_get_min_image_distance_from_matrix(np.matmul(fcc_optimal_P, cell)), 3)

    sc_fcc_P_and_min_dists = [
        (sc_optimal_P, sc_min_dist),
        (fcc_optimal_P, fcc_min_dist),
    ]
    sc_fcc_P_and_min_dists.sort(
        key=lambda x: (-x[1], _P_matrix_sort_func(x[0], cell))
    )  # sort by max min dist, then by sorting func

    optimal_P, min_dist = sc_fcc_P_and_min_dists[0]

    from doped.utils.symmetry import get_clean_structure  # avoid circular import

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


@lru_cache(maxsize=int(1e3))
def _nonzero_coeffs_in_box(ni: int, nj: int, nk: int) -> np.ndarray:
    """
    All non-zero integer coefficient vectors ``[i, j, k]`` with ``|i| <= ni``,
    ``|j| <= nj``, ``|k| <= nk``, as an ``(N, 3)`` array; cached as the same
    (small) ranges recur constantly in supercell searches.
    """
    grid = np.mgrid[-ni : ni + 1, -nj : nj + 1, -nk : nk + 1].reshape(3, -1).T
    return grid[np.any(grid != 0, axis=1)]


def _get_min_image_distances_from_matrices(matrices: np.ndarray) -> np.ndarray:
    """
    Get the minimum image distances for a batch of lattice matrices at once,
    with fully vectorised ``numpy`` enumeration.

    Exact equivalent of
    ``_get_min_image_distance_from_matrix(..., normalised=True)`` for each
    matrix (but orders of magnitude faster than looping over ``pymatgen``'s
    ``get_points_in_sphere``): all lattice vectors within the max possible min
    image distance (``2^(1/6)`` for unit volume) are enumerated, using the
    standard reciprocal-lattice bounding box to determine the required integer
    coefficient ranges, and grouping matrices by required range for batched
    computation.

    Args:
        matrices (np.ndarray):
            ``(N, 3, 3)`` array of lattice matrices (any volumes; per-matrix
            search radii are used).

    Returns:
        np.ndarray: ``(N,)`` array of min image distances, rounded to 4 d.p.
    """
    # max possible min image distance is 2^(1/6) * volume^(1/3) (for FCC/HCP packing), per matrix, plus a
    # 1% buffer. Note candidate cell volumes equal |det(target_metric)| (= 1 for SC, but e.g. 0.25 for the
    # FCC target metric), as |det(P)| = target_size and det(norm_cell) = det(target_metric)/target_size --
    # using per-matrix radii (rather than assuming volume 1) tightens the search ranges below:
    max_rs = 2 ** (1 / 6) * np.cbrt(np.abs(_fast_3x3_determinant_vectorized(matrices))) * 1.01  # (N,)
    # a lattice point v = i*a + j*b + k*c within |v| <= r has |i| <= r*|b_i*| for reciprocal basis vectors
    # b_i* (columns of the inverse matrix), bounding the required (per-axis) search ranges:
    recip_lens = np.linalg.norm(np.linalg.inv(matrices), axis=1)  # (N, 3) reciprocal vector norms
    naxes = np.ceil(max_rs[:, None] * recip_lens + 1e-9).astype(int)  # (N, 3) per-axis integer ranges

    min_image_dists = np.empty(len(matrices))
    unique_triples, inverse = np.unique(naxes, axis=0, return_inverse=True)
    for triple_idx, (ni, nj, nk) in enumerate(unique_triples):  # group matrices by required ranges
        coeffs = _nonzero_coeffs_in_box(int(ni), int(nj), int(nk))
        group_indices = np.flatnonzero(inverse == triple_idx)  # indices of matrices with these ranges;
        # ``inverse[i]`` is the index of ``naxes[i]``'s match in ``unique_triples`` (``np.unique`` output)
        for chunk in np.array_split(group_indices, max(1, len(group_indices) * len(coeffs) // int(4e6))):
            # chunked to bound peak memory usage (~100 MB) for large candidate sets / ranges
            vectors = coeffs @ matrices[chunk]  # (M, C, 3) possible lattice vectors, batched matmul
            sq_dists = np.einsum("kij,kij->ki", vectors, vectors)  # (M, C) squared vector lengths
            min_image_dists[chunk] = np.sqrt(sq_dists.min(axis=1))

    return min_image_dists.round(4)  # round to 4 d.p. as in _get_min_image_distance_from_matrix


def _find_ideal_supercell_for_target_metric(
    cell: np.ndarray,
    target_size: int,
    limit: int = 2,
    verbose: bool = False,
    target_metric: np.ndarray | None = None,
    target_shape="SC",
):
    """
    Find the optimal supercell transformation matrix for the given ``cell``,
    ``target_size``, transformation matrix search ``limit`` and
    ``target_metric``, and returns the optimal P matrix.

    First identifies unique transformation matrices of the given
    ``target_size`` with integer P matrices that have element values within
    +/-``limit`` of the ideal (fractional) P matrix, then identifies those
    which maximise the minimum image distance, then of those returns the most
    preferred (cleanest) P matrix choice as given by ``_get_optimal_P``.
    """
    target_metric = np.eye(3) if target_metric is None else target_metric
    (
        valid_P,
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
        target_shape=target_shape,
    )

    if len(unique_cell_matrices) == 0:
        raise ValueError("No valid P matrices found with given settings")

    min_image_dists = _get_min_image_distances_from_matrices(unique_cell_matrices)

    # get indices of min_image_dists that are equal to the minimum
    best_min_dist = np.max(min_image_dists)  # in terms of supercell effective cubic length
    if verbose:
        print(f"{target_shape} best minimum image distance (best_min_dist): {best_min_dist}")

    min_dist_indices = np.where(min_image_dists == best_min_dist)[0]

    return _get_optimal_P(
        valid_P=valid_P,
        selected_indices=min_dist_indices,
        unique_hashes=unique_hashes,
        lengths_angles_hash=lengths_angles_hash,
        norm_cell=norm_cell,
        verbose=verbose,
        target_shape=target_shape,
        cell=cell,
    )


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

    for (near-)cubic supercells generated by the ``pymatgen``
    ``CubicSupercellTransformation`` class. If a (near-)cubic supercell cannot
    be found for a given number of unit cells, then the corresponding dict
    value will be set to an empty dict.

    Args:
        struct (|Structure|):
            |Structure| to generate supercells for.
        uc_range (tuple):
            Range of numbers of unit cells to search over.

    Returns:
        dict:
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
