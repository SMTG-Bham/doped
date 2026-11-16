"""
Utility code and functions for generating & analysing defect supercells.
"""

import warnings
from collections.abc import Sequence
from functools import lru_cache
from itertools import combinations, permutations, product
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation
from pymatgen.util.coord import pbc_shortest_vectors
from tqdm import tqdm

if TYPE_CHECKING:
    from doped.core import DefectComplex, DefectEntry


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
            "linearly dependent lattice. Please check your inputs!"
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


def get_complex_min_image_distance(
    defect_complex: "DefectComplex | DefectEntry | Sequence[PeriodicSite] | np.ndarray",
    structure: Structure | Lattice | np.ndarray | None = None,
    coords_are_cartesian: bool = True,
    unwrap: bool = True,
    all_configurations: bool = False,
) -> float:
    r"""
    Get the minimum image distance for a defect complex, defined as the minimum
    distance between any constituent point defect and a constituent point
    defect of a periodic image of the complex.

    This is the defect complex analogue of ``get_min_image_distance`` (to which
    it reduces for a single constituent). Note that, unlike the point defect
    minimum image distance, it depends on the size and orientation of the complex
    within the lattice.

    Args:
        defect_complex (|DefectComplex| | |DefectEntry| | Sequence[|PeriodicSite|] | np.ndarray):
            The defect complex, as a |DefectComplex|, a |DefectEntry|, the
            constituent point defect |PeriodicSite|\s, or an ``(n, 3)``
            array of their coordinates.
        structure (|Structure| | |Lattice| | np.ndarray):
            The cell in which the complex sits, as a |Structure|, |Lattice| or
            ``(3, 3)`` lattice matrix. Optional if ``defect_complex`` carries
            its own cell (i.e. not an array), which this will override.
        coords_are_cartesian (bool):
            Whether coordinates given, if as an array, are Cartesian rather than
            fractional. (Default: True)
        unwrap (bool):
            If only sites or coordinates are provided, whether to unwrap the complex
            before computing the distance. Ignored for |DefectComplex| and
            |DefectEntry| inputs, whose sites are already unwrapped for
            instances constructed by ``doped``. (Default: True)
        all_configurations (bool):
            Whether to return the largest complex minimum image distance over
            all symmetry-equivalent configurations  of the complex in the cell,
            rather than that of the given configuration alone. Requires a
            |DefectComplex| or |DefectEntry| input, from which the configurations
            can be determined. (Default: False)

    Returns:
        float: Complex minimum image distance, in Å.
    """
    from doped.core import DefectComplex, DefectEntry  # avoid circular imports

    configs: list[Any]  # per configuration, either constituent sites or their coordinates
    if isinstance(defect_complex, DefectEntry):
        from doped.utils.parsing import (  # avoid circular imports
            _get_bulk_supercell,
            _get_defect_supercell,
            _get_defect_supercell_sites,
        )

        if not isinstance(defect_complex.defect, DefectComplex):
            raise ValueError(
                f"The given |DefectEntry| ({defect_complex.name}) is not a defect complex: use "
                f"``get_min_image_distance`` for point defects."
            )
        if (sites := _get_defect_supercell_sites(defect_complex, unwrapped=True)) is None:
            raise ValueError(
                f"The constituent point defect sites of the given |DefectEntry| "
                f"({defect_complex.name}) could not be determined: please provide them directly."
            )
        configs = [sites]
        if all_configurations:  # all symmetry-equivalent placements in the supercell
            configs += defect_complex.equivalent_supercell_complexes or []
        cell = _get_bulk_supercell(defect_complex) or _get_defect_supercell(defect_complex)
    elif isinstance(defect_complex, DefectComplex):
        configs = (
            defect_complex._orbit_cart_coords()  # all symmetry-equivalent configurations
            if all_configurations
            else [[constituent.site for constituent in defect_complex.defects]]
        )
        cell = defect_complex.structure
    else:
        if all_configurations:
            raise ValueError(
                "``all_configurations`` requires a |DefectComplex| or |DefectEntry| input, from which "
                "the symmetry-equivalent configurations of the complex can be determined."
            )  # TODO allow Structure and PeriodicSite
        configs = [list(defect_complex)]  # |PeriodicSite|s or coordinates
        cell = getattr(configs[0][0], "lattice", None)

    # assume DefectComplex | DefectEntry objects already unwrapped
    unwrap = unwrap and not isinstance(defect_complex, DefectComplex | DefectEntry)

    if structure is not None:
        cell = structure
    if cell is None:
        raise ValueError(
            "No lattice given or found on the input coordinates: please provide ``structure``."
        )
    lattice = (
        cell
        if isinstance(cell, Lattice)
        else (cell.lattice if isinstance(cell, Structure) else Lattice(np.asarray(cell)))
    )

    dists = []
    for config in configs:
        if isinstance(config[0], PeriodicSite):  # cartesian
            cart_coords = np.array([site.coords for site in config])
        else:
            coords = np.asarray(config, dtype=float)
            cart_coords = coords if coords_are_cartesian else lattice.get_cartesian_coords(coords)

        if unwrap:
            cart_coords = lattice.get_cartesian_coords(
                _get_unwrapped_complex_frac_coords(lattice, lattice.get_fractional_coords(cart_coords))
            )
        dists.append(_get_complex_min_image_distance_from_matrix(lattice.matrix, cart_coords))

    return max(dists)


def _get_complex_min_image_distance_from_matrix(matrix: np.ndarray, cart_coords: np.ndarray) -> float:
    """
    Get the minimum image distance for a defect complex given a lattice matrix,
    defined as the minimum distance between any constituent point defect to a
    constituent point defect of a periodic image.

    The same algorithm as for point defects, but lattice points R are here
    bounded by ||R|| <= d_0 + complex span, where d_0 is the close-packing
    bound as above.

    Note not independent of lattice orientation and can't be normalised.

    Args:
        matrix (np.ndarray): Lattice matrix.
        cart_coords (np.ndarray):
            ``(n, 3)`` array of Cartesian coordinates of the constituent
            point defect sites of the complex, unwrapped.

    Returns:
        float: Complex minimum image distance.
    """
    # all separations r_b - r_a between constituent sites, including a == b (the zero vector)
    intra_vecs = (cart_coords[:, None, :] - cart_coords[None, :, :]).reshape(-1, 3)  # (n^2, 3)
    complex_span = np.linalg.norm(intra_vecs, axis=1).max()  # largest separation within complex

    # evaluate min image distance here instead for better bound?
    # -> not really faster
    lattice = Lattice(matrix)
    max_min_dist = lattice.volume ** (1 / 3) * 2 ** (1 / 6)  # d_min <= 2^(1/6) * V^(1/3) for any lattice
    _fcoords, _dists, _idxs, images = lattice.get_points_in_sphere(  # 1.01 factor for rounding issues
        np.array([[0, 0, 0]]), [0, 0, 0], r=(max_min_dist + complex_span) * 1.01, zip_results=False
    )  # any longer R cannot contribute, as |v + R| >= |R| - |v| >= |R| - complex_span
    images = np.array(images)  # (n_R, 3) integer coefficients of the lattice vectors R found
    lattice_vecs = images[np.any(images != 0, axis=1)] @ matrix  # no R = 0 (intra-complex)

    # (n_R, n^2) distances |v + R| between each site and the images of each site (including itself):
    dists = np.linalg.norm(lattice_vecs[:, None, :] + intra_vecs[None, :, :], axis=-1)
    min_dist = float(np.min(dists))
    if min_dist <= 0:
        raise ValueError(
            "Complex minimum image distance less than or equal to zero! This is possibly due to "
            "a co-planar / linearly dependent lattice. Please check your inputs! Or the lattice may be "
            "small relative to the complex?"
        )

    return round(min_dist, 4)  # round to 4 decimal places to avoid issues with tiny numerical differences


def _get_complex_ws_radius(
    matrix: np.ndarray, cart_coords: np.ndarray, point: np.ndarray | None = None
) -> float:
    r"""
    Defining the complex WS cell as the set of points closer to a constituent
    point defect than any image point defect, ie the union of Voronoi cells of
    the constituent point defects, returns the closest distance to the complex
    WS cell boundary.

    Measured from the complex centroid by default, but from any given ``point``
    of the cell if provided.

    Note that the complex WS cell is a union of Voronoi cells so is not necessarily
    convex, and so this is not necessarily equal to the perpendicular distance to
    some bisecting plane.

    Args:
        matrix (np.ndarray): Lattice matrix.
        cart_coords (np.ndarray):
            ``(n, 3)`` array of Cartesian coordinates of the constituent
            point defect sites of the complex, unwrapped.
        point (np.ndarray):
            Cartesian coordinates of the point to measure from. If ``None``
            (default), the complex centroid is used.

    Returns:
        float:
            Complex Wigner-Seitz radius; zero if ``point`` lies outside the
            complex WS cell (i.e. is closer to an image constituent than to
            any constituent of the provided complex).
    """
    cart_coords = np.asarray(cart_coords)
    origin = cart_coords.mean(axis=0) if point is None else np.asarray(point, dtype=float)
    constituent_vecs = cart_coords - origin  # (n, 3), from the origin (centroid by default)
    c_max = float(np.linalg.norm(constituent_vecs, axis=1).max())  # largest distance to constituent

    # closest bisector of an image constituent at v is (|v|-c_max)/2
    # upper bound on the closest image constituent is |u| <= d_min + c_max
    # so nearest boundary <= d_min/2 + c_max
    # therefore only test lattice points R for which (|v|-c_max)/2 <= d_min/2 + c_max
    # ie R <= d_min + 4*c_max
    lattice = Lattice(matrix)
    max_min_dist = lattice.volume ** (1 / 3) * 2 ** (1 / 6)
    _fcoords, _dists, _idxs, images = lattice.get_points_in_sphere(  # while the cell itself reaches no
        np.array([[0, 0, 0]]), [0, 0, 0], r=(max_min_dist + 4 * c_max) * 1.01, zip_results=False
    )
    images = np.array(images)
    lattice_vecs = images[np.any(images != 0, axis=1)] @ matrix  # no R = 0 (this complex itself)
    image_vecs = (constituent_vecs[:, None, :] + lattice_vecs[None, :, :]).reshape(-1, 3)
    # (n_point*n_R, 3) = (m, 3)

    # bisecting planes are 2x.(q_j - c_i) = q_j^2 - c_i^2 for image constituent q_j, complex
    # constituents c_i
    normals = image_vecs[:, None, :] - constituent_vecs[None, :, :]  # (m, n, 3)
    offsets = (
        (image_vecs**2).sum(axis=1)[:, None] - (constituent_vecs**2).sum(axis=1)[None, :]
    ) / 2  # (m, n)
    plane_dists = (offsets / np.linalg.norm(normals, axis=-1)).max(axis=1)  # (m,) nearest bounding plane
    if plane_dists.min() <= 0:  # the origin lies outside complex WS cell
        if point is not None:
            return 0.0
        raise ValueError(
            "Defect complex is closer to its own periodic images than to itself! This is possibly due to "
            "a co-planar / linearly dependent lattice, or a lattice small relative to the complex. "
            "Please check your inputs!"
        )

    # the complex WS cell is not necessarily convex - we have to check for nearest vertices, edges
    # not just perpendicular plane distances. only intersections over image's planes: for a constituent
    # we are inside a convex voronoi cell
    actives = [  # faces, edges, vertices
        list(subset) for size in (1, 2, 3) for subset in combinations(range(len(constituent_vecs)), size)
    ]
    radius = np.inf
    # over set of p <= 3 planes {i} from an image constituent:
    # under constraint n_i.x = d_i, we solve for min |x| ie L = x.T*x - 2a.T*(N*x-d)
    # where N_ji = (n_i)_j then x = N.T @ a and N @ x = d -> x = N.T @ inv(N @ N.T) @ d
    for i in np.argsort(plane_dists):  # loop over image constituents, nearest bounding plane first
        if plane_dists[i] >= radius:
            break  # cannot be closer than perpendicular distance so none remaining
        for active in actives:
            try:
                proj = normals[i][active].T @ np.linalg.solve(
                    normals[i][active] @ normals[i][active].T, offsets[i][active]
                )
            except np.linalg.LinAlgError:  # parallel etc taken care of at smaller p
                continue
            if np.all(normals[i] @ proj >= offsets[i] - 1e-9):  # inside the rest so in the region
                radius = min(radius, float(np.linalg.norm(proj)))

    return round(radius, 4)  # round to 4 decimal places to avoid tiny numerical differences


def _get_unwrapped_complex_frac_coords(
    bulk_supercell: Structure | Lattice,
    sites: Sequence[PeriodicSite] | Sequence[np.ndarray] | np.ndarray,
) -> list[np.ndarray]:
    """
    Unwrap the sites (or fractional coordinates) of a defect complex in a
    periodic supercell to a single compact cluster, by choosing the periodic
    image of each constituent such that the complex is one contiguous
    configuration.

    This is unambiguous if every pair of constituents can be separated by its
    minimum image vector, i.e. every constituent lies within the Wigner-Seitz
    cell centred on every other constituent. If not (typically when the
    supercell is too small for the complex), a warning is emitted, and the
    unwrapping which maximises the complex minimum image distance (the shortest
    distance from any constituent to any constituent of a periodic image of the
    complex) is chosen, with ties broken by minimising the variance of the
    constituent positions.

    Args:
        bulk_supercell (|Structure| | |Lattice|):
            Bulk supercell host structure (or its |Lattice|), in which
            ``sites`` are defined.
        sites (Sequence[|PeriodicSite|] | np.ndarray):
            Constituent periodic sites to unwrap, defined in the
            ``bulk_supercell`` frame; either as |PeriodicSite| objects or
            fractional coordinates.

    Returns:
        list[np.ndarray]:
            The unwrapped fractional coordinates of ``sites`` (ordering
            unchanged), in the ``bulk_supercell`` frame.
    """
    lattice = bulk_supercell if isinstance(bulk_supercell, Lattice) else bulk_supercell.lattice
    if len(sites) and isinstance(sites[0], PeriodicSite):
        sites = [site.frac_coords for site in cast("Sequence[PeriodicSite]", sites)]
    frac_coords = np.array(sites, dtype=float)
    n_sites = len(frac_coords)
    if n_sites < 2:
        return list(frac_coords)

    # u_ij is the shortest vector from constituent i to any periodic image of constituent j (the 'minimum
    # image' of j, with u_ij in the Wigner-Seitz cell). The complex minimum image distance (CMID) is the
    # shortest distance from any constituent to a constituent of a neighbouring periodic copy of the
    # complex (or the host minimum image distance d_min, if smaller); i.e. the minimum over all pairs of
    # constituents (i, j) of the distance from i to the nearest copy of j outside the complex, which is:
    # - if j is unwrapped to its minimum image relative to i (a 'reduced' pair): the next-nearest image
    #   distance min_{l != 0} |u_ij - l| (l: lattice vectors), as the nearest image is in the complex
    # - otherwise: |u_ij|, as the nearest image of j to i is then in a neighbouring copy of the complex
    #   (only needed if no all-reduced unwrapping exists; i.e. 3+ constituents in a too-small supercell;
    #   warning given about ambiguous/ill-defined complex definition in such cases)
    u = np.asarray(pbc_shortest_vectors(lattice, frac_coords, frac_coords))  # (n, n, 3)
    min_image_dists = np.linalg.norm(u, axis=-1)
    lower = np.tril_indices(n_sites, -1)
    u[lower] = -u[lower[::-1]]  # ensure u_ij = -u_ji, for separations on the WS cell boundary
    # integer image translation of j for its minimum image relative to i:
    w = np.rint(lattice.get_fractional_coords(u) - (frac_coords - frac_coords[:, None])).astype(int)

    # the minimising l for the next-nearest image distance has |l| <= |u| + |u - l| <= 2|u| + d_min:
    d_min = _get_min_image_distance_from_matrix(lattice.matrix)
    *_, lattice_ints = lattice.get_points_in_sphere(  # 1.01 factor for rounding issues
        np.array([[0, 0, 0]]), [0, 0, 0], r=(d_min + 2 * min_image_dists.max()) * 1.01, zip_results=False
    )
    lattice_ints = np.array(lattice_ints)
    lattice_vecs = lattice_ints[np.any(lattice_ints != 0, axis=1)] @ lattice.matrix
    next_image_dists = np.linalg.norm(u[:, :, None, :] - lattice_vecs, axis=-1).min(axis=2)

    # Reduce pairs in order of increasing |u_ij| (then decreasing next-nearest image distance), grouping
    # constituents as their relative images become fixed (Kruskal-like), which gives a CMID-optimal
    # unwrapping (not necessarily unique). Note that simply unwrapping relative to one 'anchor' constituent
    # is not CMID-optimal in general.
    translations = np.zeros((n_sites, 3), dtype=int)
    group = np.arange(n_sites)
    reduced = np.zeros((n_sites, n_sites), dtype=bool)
    for i, j in sorted(
        combinations(range(n_sites), 2), key=lambda ij: (min_image_dists[ij], -next_image_dists[ij], ij)
    ):
        shift = w[i, j] - (translations[j] - translations[i])
        if group[i] != group[j]:  # relative image not yet fixed; merge j's group into i's
            merging = group == group[j]
            translations[merging] += shift
            group[merging] = group[i]
        elif shift.any():  # relative image already fixed, and not the minimum image
            continue
        reduced[i, j] = reduced[j, i] = True

    intersite = ~np.eye(n_sites, dtype=bool)
    if not reduced[intersite].all():
        # CMID-optimal unwrapping is then generally not unique; pairs separated by less than the CMID must
        # be reduced, so group constituents connected by these, and choose the relative images of these
        # groups which minimise the variance of the constituent positions:
        cmid = min(d_min, next_image_dists[reduced].min(), min_image_dists[intersite & ~reduced].min())
        block = np.arange(n_sites)
        for i, j in zip(*np.nonzero(np.triu(min_image_dists < cmid - 1e-8, 1)), strict=True):
            block[block == block[j]] = block[i]
        groups = [block == label for label in np.unique(block)]

        if len(groups) > 1:
            # spread (n * variance) bounds pair separations, |r_i - r_j|^2 <= 2 * spread, so for the group
            # centroids c_m & c_n, a lowering offset l has |c_m + l - c_n| <= sqrt(2 * spread) & thus
            # |l| <= 2 * sqrt(2 * spread):
            cart = lattice.get_cartesian_coords(frac_coords + translations)
            spread = float(((cart - cart.mean(axis=0)) ** 2).sum())
            *_, shift_ints = lattice.get_points_in_sphere(  # 1.01 factor for rounding issues
                np.array([[0, 0, 0]]), [0, 0, 0], r=2 * (2 * spread) ** 0.5 * 1.01, zip_results=False
            )
            shifts = np.array(shift_ints, dtype=int)  # (M, 3)
            _MAX_COMPLEX_ARRANGEMENTS = 200000  # max group arrangements to search when minimising variance
            if len(shifts) ** (len(groups) - 1) <= _MAX_COMPLEX_ARRANGEMENTS:  # else keep above unwrapping
                arrangements = np.array(list(product(range(len(shifts)), repeat=len(groups) - 1)))
                sizes = np.array([grp.sum() for grp in groups])
                centroids = np.array([cart[grp].mean(axis=0) for grp in groups])
                block_centroids = np.repeat(centroids[None], len(arrangements), axis=0)  # (N, K, 3)
                block_centroids[:, 1:] += (shifts @ lattice.matrix)[arrangements]
                # spread = sum over groups k of (spread(k) + n_k * |c_k - c|^2), with spread(k) fixed:
                centres = np.average(block_centroids, axis=1, weights=sizes)  # (N, 3)
                spreads = (sizes * ((block_centroids - centres[:, None]) ** 2).sum(axis=-1)).sum(axis=1)
                for grp, index in zip(groups[1:], arrangements[spreads.argmin()], strict=True):
                    translations[grp] += shifts[index]
                reduced = np.all(translations - translations[:, None] == w, axis=-1) & intersite

    unwrapped_fc = list(frac_coords + translations)
    if not reduced[intersite].all():  # no unambiguous unwrapping exists
        cmid = min(d_min, next_image_dists[reduced].min(), min_image_dists[intersite & ~reduced].min())
        cart_coords = lattice.get_cartesian_coords(unwrapped_fc)
        span = np.linalg.norm(cart_coords[:, None] - cart_coords, axis=-1).max()
        warnings.warn(
            f"A defect complex separation is not a minimum image of the supercell (complex span: "
            f"{span:.2f} Å, complex minimum image distance: {cmid:.2f} Å), so the unwrapping of this "
            f"complex may be ambiguous - the supercell is likely too small for this complex. "
            f"The unwrapping which maximises the complex minimum image distance, and then minimises the "
            f"variance of the constituent positions, has been chosen - although this may not be unique."
        )

    return unwrapped_fc


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


def _get_hnf_P_arrays(target_size: int) -> np.ndarray:
    """
    Get all Hermite Normal Forms with a given determinant, corresponding to all
    possible sublattices of a given order. Given as upper triangular matrices
    with off diagonals around zero (rather than strictly positive). HNF is an
    integer matrix.

    [a u v]
    [0 b w]
    [0 0 c]

    where u is defined mod b, v and w mod c.

    Note n ~ O(target_size^2) so memory usage might be large for big target_size.
    Would hit 1GB around target_size = 4000. May need to split this array and
    process in chunks TODO?

    Args:
        target_size (int): Target supercell size (in number of unit cells).

    Returns:
        np.ndarray: ``(n, 3, 3)`` array of HNF transformation matrices.
    """
    factorisations = [  # all a*b*c = target_size, each giving b*c^2 matrices
        (a, b, target_size // (a * b))
        for a in range(1, target_size + 1)
        if target_size % a == 0
        for b in range(1, target_size // a + 1)
        if (target_size // a) % b == 0
    ]

    # preallocate array and use int16 to save memory
    P_arrays = np.zeros((sum(b * c * c for _, b, c in factorisations), 3, 3), dtype=np.int16)

    start_idx = 0
    for a, b, c in factorisations:
        # zero-centred residues, mod b for u, and mod c for v/w: this gives slightly more
        # orthogonal starting basis before reduction, vs strictly positive u,v,w
        res_b = np.arange(-((b - 1) // 2), b // 2 + 1, dtype=np.int16)  # (b,)
        res_c = np.arange(-((c - 1) // 2), c // 2 + 1, dtype=np.int16)  # (c,)
        off_diagonals = np.stack(np.meshgrid(res_b, res_c, res_c, indexing="ij"), axis=-1)
        # (b, c, c, 3)

        block = P_arrays[start_idx : start_idx + b * c * c]
        block[:, 0, 0], block[:, 1, 1], block[:, 2, 2] = a, b, c
        block[:, 0, 1], block[:, 0, 2], block[:, 1, 2] = off_diagonals.reshape(-1, 3).T
        start_idx += b * c * c

    return P_arrays


def _reduce_lattice_matrices(
    matrices: np.ndarray, sweeps: int = 4, one_directional: bool = False
) -> np.ndarray:
    """
    Fast rough lattice reduction vectorized. A finite number of iterations are
    performed, and no check for any reduced conditions. Just reduces each
    vector against every other vector.

    Args:
        matrices (np.ndarray): ``(N, 3, 3)`` array of lattice matrices.
        sweeps (int):
            Number of reduction iterations to perform. More sweeps -> slower,
            but more reduced. (Default = 4)
        one_directional (bool):
            If ``True``, only reduce later vectors against earlier ones.
            Useful for the first sweep if there are a large number of lattices
            from HNF generation. See comment below for explanation.
            (Default = False)

    Returns:
        np.ndarray: ``(N, 3, 3)`` array of reduced lattice matrices.
    """
    pairs = list(permutations(range(3), 2))
    pairs = [(i, j) for (i, j) in pairs if i > j] if one_directional else pairs
    # why to use one-directional reduction from an HNF basis lattice:
    # true: l_1 and l_2 are already reduced against l_3, so drop (0,2) and
    # (1,2) reductions for free.
    # conjecture: because of multiplicity of the factorisation abc=N is bc^2,
    # HNFs are dominated by large c, small a lattices, so l_0 and l_3 are often
    # parallel, and so dropping (0,1) results in a very short l_2

    matrices = np.array(matrices, dtype=float)  # copy for working in place
    for _ in range(sweeps):
        for i, j in pairs:
            # b_i -> b_i - round(c_ij) * b_j, where c_ij = (b_i . b_j)/|b_j|^2
            projections = np.einsum("ni,ni->n", matrices[:, i], matrices[:, j]) / np.einsum(
                "ni,ni->n", matrices[:, j], matrices[:, j]
            )  # (N,) values of c_ij
            matrices[:, i] -= np.round(projections)[:, None] * matrices[:, j]

    return matrices


def _get_best_complex_candidates(
    cell_matrices: np.ndarray,
    cart_coords: np.ndarray,
    block_size: int = 512,
    best_dist: float = -np.inf,
) -> tuple[float, np.ndarray]:
    """
    Get the largest complex minimum image distance over the given candidate
    supercell lattices, and the indices of all candidates which achieve it.

    The complex min image distance <= point min image distance <= shortest
    lattice vector, so there is an upper bound for each lattice. This bound
    is evaluated for all lattices, the lattices are sorted by the bound, then
    the complex min image distance is evaluated in batches until it's no
    longer possible for a better candidate to be found.

    Args:
        cell_matrices (np.ndarray):
            ``(N, 3, 3)`` array of candidate supercell lattice matrices.
        cart_coords (np.ndarray):
            ``(n, 3)`` array of Cartesian coordinates of the constituent
            point defect sites of the complex, unwrapped.
        block_size (int):
            Number of candidates to evaluate per block. (Default = 512)
        best_dist (float):
            Best complex minimum image distance already found (e.g. from a
            previous chunk of candidates), used to screen these candidates
            at the start. (Default: ``-inf``.)

    Returns:
        tuple[float, np.ndarray]:
            The largest complex minimum image distance, and the indices of the
            candidates which give it (empty if none beat ``best_dist``).
    """
    # complex minimum image distance D <= d_min <= min_i|L_i| for any basis {L_i}
    # bounding done in squared distance for speed (other than -inf)
    sq_bounds = np.einsum("nij,nij->ni", cell_matrices, cell_matrices).min(axis=1)  # (N,)
    order = np.argsort(-sq_bounds)  # (N,) candidate indices, by decreasing bound

    dists = np.full(len(cell_matrices), -np.inf)  # (N,)
    for start_idx in range(0, len(order), block_size):
        block = order[start_idx : start_idx + block_size]
        block = block[sq_bounds[block] >= max(best_dist - 1e-4, 0) ** 2]
        # retain only those candidates which can beat best_dist
        if not len(block):  # if none left in block, done as already sorted by bound
            break

        dists[block] = _get_complex_min_image_distances_from_matrices(cell_matrices[block], cart_coords)
        best_dist = max(best_dist, dists[block].max())

    # note unevaluated candidates cannot beat best_dist
    return best_dist, np.flatnonzero(dists >= best_dist)


def _get_best_complex_P_arrays(
    cell: np.ndarray,
    cart_coords: np.ndarray,
    target_size: int,
) -> tuple[float, np.ndarray]:
    """
    Gets the largest possible complex minimum image distance for a supercell of
    target_size.

    A roughly optimised algorithm for target_size up to around 500 (and
    reasonably sized complexes):
    - generate HNFs
    - estimate the min image distance from a small sample
    - reduce HNFs one sweep at a time, pruning by the shortest lattice vector
      bound after each sweep
    - sort by shortest lattice vector bound and evaluate complex min image
      distance until bound below best one found

    HNFs are poorly reduced bases so they must be reduced for good
    pruning and for faster complex min image distance search. Chunking here
    is mostly just for memory considerations, batching for search is done
    in _get_best_complex_candidates. Note that the number of
    supercell lattices scales as ``O(target_size^2)``, taking ~2 s for
    ``target_size = 1000``, so this is not well suited for large supercells.

    Args:
        cell (np.ndarray): Unit cell matrix, to generate supercells of.
        cart_coords (np.ndarray):
            ``(n, 3)`` array of Cartesian coordinates of the constituent
            point defect sites of the complex, unwrapped.
        target_size (int): Target supercell size (in number of unit cells).

    Returns:
        tuple[float, np.ndarray]:
            The largest complex minimum image distance, and an ``(n, 3, 3)``
            array of the supercell matrices which give it.
    """
    P_arrays = _get_hnf_P_arrays(target_size)  # (a_N, 3, 3), where a_N = sum(b*c^2) ~ 2*target_size^2
    best_P_arrays = []

    # estimate a best distance from a coarse sample
    sample = P_arrays[:: max(1, len(P_arrays) // 1000)]
    best_dist = _get_best_complex_candidates(_reduce_lattice_matrices(sample @ cell), cart_coords)[0]

    # chunking for memory usage (~50 MB)
    for start in range(0, len(P_arrays), int(5e4)):
        P_chunk = P_arrays[start : start + int(5e4)]  # (M, 3, 3), M <= 5e4
        cell_matrices = P_chunk @ cell  # (M, 3, 3), real cells

        # reduce/prune cycle
        sq_best_dist = max(best_dist - 1e-4, 0) ** 2
        for one_directional in (True, False, False, False):
            # the first sweep is one-directional as the HNF form is basically already reduced
            # upwards
            cell_matrices = _reduce_lattice_matrices(
                cell_matrices,
                sweeps=1,
                one_directional=one_directional,
            )
            can_win = np.einsum("nij,nij->ni", cell_matrices, cell_matrices).min(axis=1) >= sq_best_dist
            P_chunk, cell_matrices = P_chunk[can_win], cell_matrices[can_win]
        if not len(P_chunk):
            continue

        # min image search
        chunk_dist, indices = _get_best_complex_candidates(cell_matrices, cart_coords, best_dist=best_dist)

        if chunk_dist > best_dist:  # new best
            best_dist, best_P_arrays = chunk_dist, [P_chunk[indices]]
        elif len(indices):  # tied best
            best_P_arrays.append(P_chunk[indices])

    return best_dist, np.concatenate(best_P_arrays)


def _get_min_complex_target_size(
    cell: np.ndarray, cart_coords: np.ndarray, min_image_distance: float
) -> int:
    r"""
    Get the smallest possible supercell size (in number of ``cell``\ s) which
    could give a complex minimum image distance D of ``min_image_distance``.

    Gives the maximum of:
    - The point defect bound.
    - Rough complex bound: take spheres D/2 around the furthest separated
    constituents, and use the total volume of the two cut spheres.

    Tighter bounds could be found?

    Args:
        cell (np.ndarray): Unit cell matrix, to generate supercells of.
        cart_coords (np.ndarray):
            ``(n, 3)`` array of Cartesian coordinates of the constituent
            point defect sites of the complex, unwrapped.
        min_image_distance (float):
            Target complex minimum image distance (in Å).

    Returns:
        int: Minimum possible supercell size (in number of unit cells).
    """
    pair_distances = [np.linalg.norm(j - i) for i, j in combinations(cart_coords, 2)]
    span = min(max(pair_distances, default=0.0), min_image_distance)
    # span clamped to D ie up to two disjoint spheres

    # close-packed point defect bound
    min_dist_volume = min_image_distance**3 / np.sqrt(2)

    # two cut spheres bound
    # 2*(4/3)*pi*(D/2)^3 - (pi/12)*(D - span)^2 * (2D + span)
    two_sphere_volume = np.pi / 3 * min_image_distance**3 - np.pi / 12 * (
        min_image_distance - span
    ) ** 2 * (2 * min_image_distance + span)

    return int(np.ceil(max(min_dist_volume, two_sphere_volume) / abs(np.linalg.det(cell))))


def _get_optimal_complex_P(P_arrays: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """
    Get a clean, best supercell transformation for a complex, from a set which
    have the same complex minimum image distance. Candidates are ranked by
    cleanness then by point minimum image distance. TODO sort order?

    Args:
        P_arrays (np.ndarray):
            ``(n, 3, 3)`` array of candidate supercell matrices.
        cell (np.ndarray): Unit cell matrix which these transform.

    Returns:
        np.ndarray: The best supercell transformation matrix.
    """
    # reduce (HNFs) to sensible basis
    reduced_P_arrays = np.round(_reduce_lattice_matrices(P_arrays @ cell) @ np.linalg.inv(cell))

    best_P = min(  # get best
        reduced_P_arrays.astype(int),
        key=lambda P: (
            _P_matrix_sort_func(P, cell),
            -_get_min_image_distance_from_matrix(np.matmul(P, cell)),
        ),
    )

    return _clean_P_matrix(best_P, cell)  # clean


def _get_cubic_complex_P(
    cell: np.ndarray, cart_coords: np.ndarray, target_size: int, force_diagonal: bool = False
) -> tuple[np.ndarray, float]:
    """
    Get the most cubic supercell transformation (P) matrix of the given
    ``target_size``, breaking ties by the complex minimum image distance.

    Currently the cubic metric is the primary criterion and the complex
    minimum image distance is secondary, matching pymatgen
    CubicSupercellTransformation - subject to change?

    Args:
        cell (np.ndarray): Unit cell matrix, to generate a supercell of.
        cart_coords (np.ndarray):
            ``(n, 3)`` array of Cartesian coordinates of the constituent
            point defect sites of the complex, unwrapped.
        target_size (int): Target supercell size (in number of unit cells).
        force_diagonal (bool):
            Whether to only consider diagonal transformation matrices.
            (Default = False)

    Returns:
        tuple[np.ndarray, float]:
            The supercell transformation matrix, and its complex minimum image
            distance.
    """

    def _cubic_cell_metrics(matrices: np.ndarray) -> np.ndarray:
        lengths = np.linalg.norm(matrices, axis=2)  # (n, 3)
        deviations = lengths / np.cbrt(np.abs(np.linalg.det(matrices)))[:, None] - 1  # (n, 3)
        return np.round(np.sum(deviations**2, axis=1), 4)

    if force_diagonal:
        P_arrays = np.array(
            [
                np.diag((a, b, target_size // (a * b)))
                for a in range(1, target_size + 1)
                if target_size % a == 0
                for b in range(1, target_size // a + 1)
                if (target_size // a) % b == 0
            ]
        )
        metrics = _cubic_cell_metrics(P_arrays @ cell)  # not reduced to keep diagonal

    else:  # reduce - cubic metric is basis independent
        # There are O(target_size^2) of these, so only the (n,) metrics are kept, chunk by chunk:
        P_arrays = _get_hnf_P_arrays(target_size)
        metrics = np.concatenate(
            [
                _cubic_cell_metrics(_reduce_lattice_matrices(P_arrays[start : start + int(5e4)] @ cell))
                for start in range(0, len(P_arrays), int(5e4))
            ]  # chunked for memory consideration (~50MB)
        )

    # only the most cubic candidates need complex min image dist evaluated
    # as cubicness is currently primary criterion
    P_arrays = P_arrays[metrics == metrics.min()]
    cell_matrices = P_arrays @ cell if force_diagonal else _reduce_lattice_matrices(P_arrays @ cell)
    dists = _get_complex_min_image_distances_from_matrices(cell_matrices, cart_coords)

    optimal_P = np.round(cell_matrices[dists.argmax()] @ np.linalg.inv(cell)).astype(int)

    return (optimal_P if force_diagonal else _clean_P_matrix(optimal_P, cell)), float(dists.max())


def find_ideal_complex_supercell(
    cell: np.ndarray,
    cart_coords: np.ndarray,
    target_size: int,
    return_min_dist: bool = False,
    force_cubic: bool = False,
    force_diagonal: bool = False,
    verbose: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    r"""
    Given an input cell matrix and the constituent point defect sites of a
    defect complex, find the supercell matrix (P) of ``target_size`` ``cell``\
    s which maximises the complex minimum image distance (i.e. the minimum
    distance between any constituent point defect and a constituent point
    defect of a periodic image).

    Currently the search exhaustively scans every possible sublattice
    with determinant N=``target_size`` via their Hermite Normal Forms, so the
    cost scales as O(N^2), and becomes expensive for large N (i.e. slower
    than the default box scan for point defects around N=500).

    Args:
        cell (np.ndarray): Unit cell matrix, to generate a supercell of.
        cart_coords (np.ndarray):
            ``(n, 3)`` array of Cartesian coordinates of the constituent
            point defect sites of the complex, unwrapped.
        target_size (int): Target supercell size (in number of unit cells).
        return_min_dist (bool):
            Whether to return the complex minimum image distance (in Å) as a
            second return value. (Default = False)
        force_cubic (bool):
            Whether to return the most cubic supercell of this size, rather
            than that with the largest complex minimum image distance (see
            ``_get_cubic_complex_P``). (Default = False)
        force_diagonal (bool):
            As ``force_cubic``, but additionally only considering diagonal
            transformation matrices. (Default = False)
        verbose (bool):
            Whether to print out extra information about the supercell search.
            (Default = False)

    Returns:
        np.ndarray | tuple[np.ndarray, float]:
            The supercell transformation matrix (P), and if ``return_min_dist``
            is ``True``, the complex minimum image distance (in Å).
    """
    num_best = 1
    if force_cubic or force_diagonal:
        optimal_P, best_dist = _get_cubic_complex_P(cell, cart_coords, target_size, force_diagonal)
    else:
        best_dist, P_arrays = _get_best_complex_P_arrays(cell, cart_coords, target_size)
        optimal_P = _get_optimal_complex_P(P_arrays, cell)
        num_best = len(P_arrays)

    if verbose:
        print(f"Best complex minimum image distance: {best_dist:.4f} Å")
        print(f"Supercell matrices which give it: {num_best}")
        print(f"Optimal transformation matrix (P_opt):\n{optimal_P}")

    return (optimal_P, best_dist) if return_min_dist else optimal_P


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
    fcc_target_metric = max(fcc_target_metrics, key=lambda x: np.abs(np.linalg.norm(x * cell)))

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

    if clean:
        optimal_P = _clean_P_matrix(optimal_P, cell)

    return (optimal_P, min_dist) if return_min_dist else optimal_P


def _clean_P_matrix(P: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """
    Get the cleanest P matrix.

    Args:
        P (np.ndarray): Supercell transformation matrix.
        cell (np.ndarray): Unit cell matrix which ``P`` acts on.

    Returns:
        np.ndarray: Cleaned ``P``.
    """
    from doped.utils.symmetry import get_clean_structure  # avoid circular import

    if P[0, 0] != 0 and np.allclose(np.abs(P / P[0, 0]), np.eye(3)):
        return P  # only try cleaning if it's not a perfect scalar expansion

    supercell = Structure(Lattice(cell), ["H"], [[0, 0, 0]]) * P
    clean_supercell, T = get_clean_structure(supercell, return_T=True)  # T maps orig to clean_super
    # T*orig = clean -> orig = T^-1*clean
    # P was: P*cell = orig -> T*P*cell = clean -> P' = T*P

    P = np.matmul(T, P)

    # if negative cell determinant, swap lattice vectors to get a positive determinant (as this can
    # cause issues with VASP, and results in POSCAR lattice matrix changes), picking that with the best
    # score according to the sorting function:
    if np.linalg.det(clean_supercell.lattice.matrix) < 0:
        swap_combo_score_dict = {}
        for swap_combo in permutations([0, 1, 2], 2):
            swapped_P = np.copy(P)
            swapped_P[swap_combo[0]], swapped_P[swap_combo[1]] = (
                swapped_P[swap_combo[1]],
                swapped_P[swap_combo[0]].copy(),
            )
            swap_combo_score_dict[swap_combo] = _P_matrix_sort_func(swapped_P, cell)
        best_swap_combo = min(swap_combo_score_dict, key=lambda x: swap_combo_score_dict[x])
        P[best_swap_combo[0]], P[best_swap_combo[1]] = (
            P[best_swap_combo[1]],
            P[best_swap_combo[0]].copy(),
        )

    return P


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


def _get_complex_min_image_distances_from_matrices(
    matrices: np.ndarray,
    cart_coords: np.ndarray,
    min_image_dists: np.ndarray | None = None,
) -> np.ndarray:
    """
    Batched _get_complex_min_image_distance_from_matrix. Can take point min
    image distances for a tighter bound than close-packed if you already have
    them. Note not orientation independent and not normalised.

    Args:
        matrices (np.ndarray):
            ``(N, 3, 3)`` array of lattice matrices.
        cart_coords (np.ndarray):
            ``(n, 3)`` array of Cartesian coordinates of the constituent
            point defect sites of the complex, unwrapped.
        min_image_dists (np.ndarray):
            Optional ``(N,)`` array of pre-computed (single-site) minimum
            image distances for ``matrices``, which upper bound the complex
            minimum image distances and so tighten the search radii. If
            ``None`` (default), the ``2**(1/6)`` close-packing bound is used
            instead (looser, but requiring no extra computation).

    Returns:
        np.ndarray: ``(N,)`` array of complex min image distances, to 4 d.p.
    """
    # intra complex vectors plus self vector - only i<j needed because (-R,r_ji) -> (R,r_ij)
    # (1+n(n-1)/2, 3)
    intra_vecs = np.array([np.zeros(3), *(j - i for i, j in combinations(cart_coords, 2))])
    complex_span = np.linalg.norm(intra_vecs, axis=1).max()

    upper_bounds = (
        2 ** (1 / 6) * np.cbrt(np.abs(_fast_3x3_determinant_vectorized(matrices)))
        if min_image_dists is None
        else np.asarray(min_image_dists, dtype=float)
    )  # (N,)
    max_rs = (upper_bounds + complex_span) * 1.01  # (N,)
    recip_lens = np.linalg.norm(np.linalg.inv(matrices), axis=1)  # (N, 3) reciprocal vector norms
    naxes = np.ceil(max_rs[:, None] * recip_lens + 1e-9).astype(int)  # (N, 3) per-axis integer ranges

    # TODO outer loop over intra_vecs instead and use tighter bound instead of span?
    complex_min_image_dists = np.empty(len(matrices))
    unique_triples, inverse = np.unique(naxes, axis=0, return_inverse=True)
    for triple_idx, (ni, nj, nk) in enumerate(unique_triples):  # group matrices by required ranges
        coeffs = _nonzero_coeffs_in_box(int(ni), int(nj), int(nk))
        group_indices = np.flatnonzero(inverse == triple_idx)  # indices of matrices with these ranges
        for chunk in np.array_split(group_indices, max(1, len(group_indices) * len(coeffs) // int(4e6))):
            # TODO ceil not floor?
            # NOTE peak memory usage is now ~175 MB
            vectors = coeffs @ matrices[chunk]  # (M, C, 3) possible lattice vectors, batched matmul
            sq_lengths = np.einsum("kij,kij->ki", vectors, vectors)  # (M, C) squared vector lengths
            best_sq_dists = sq_lengths.min(axis=1)  # (M,); the point min image distances
            for intra_vec in intra_vecs[1:]:  # loop over O(n^2) intra complex vectors
                # |a+b|^2 = |a|^2+|b|^2+2a.b in place - instead of extra (M,C,3) to add directly...
                sq_dists = vectors @ intra_vec  # (M, C) products
                sq_dists *= 2
                sq_dists += sq_lengths
                sq_dists += intra_vec @ intra_vec
                np.minimum(best_sq_dists, sq_dists.min(axis=1), out=best_sq_dists)
            complex_min_image_dists[chunk] = np.sqrt(best_sq_dists)

    return complex_min_image_dists.round(4)  # as in _get_complex_min_image_distance_from_matrix


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
