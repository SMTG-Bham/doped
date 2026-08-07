"""
Utility code and functions for symmetry analysis of structures and defects.
"""

import contextlib
import math
import os
import warnings
from collections.abc import Iterable, Sequence
from functools import lru_cache, partial
from itertools import combinations, permutations, product
from typing import cast

import numpy as np
import pandas as pd
import spglib
from numpy.typing import ArrayLike
from pymatgen.analysis.defects.core import DefectType
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Lattice
from pymatgen.core.structure_matcher import ElementComparator
from pymatgen.symmetry.analyzer import SymmetryUndeterminedError
from pymatgen.symmetry.groups import PointGroup
from pymatgen.transformations.standard_transformations import SupercellTransformation
from pymatgen.util.coord import is_coord_subset_pbc, lattice_points_in_supercell
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import KDTree
from scipy.spatial.distance import squareform
from sympy import Eq, Expr, simplify, solve
from tqdm import tqdm

from doped.core import Defect, DefectEntry, template_defect_entry_from_structures
from doped.utils.configurations import orient_s2_like_s1
from doped.utils.efficiency import PeriodicSite, SpacegroupAnalyzer, Structure
from doped.utils.parsing import (
    _get_bulk_supercell,
    _get_defect_supercell,
    _get_defect_supercell_frac_coords,
    _get_site_mapping_from_coords_and_indices,
    get_site_mappings,
)
from doped.utils.supercells import get_min_image_distance, min_dist


@lru_cache(maxsize=int(1e5))
def cached_simplify(eq):
    """
    Cached simplification function for ``sympy`` equations, for efficiency.
    """
    return simplify(eq)


@lru_cache(maxsize=int(1e5))
def cached_solve(equation, variable):
    """
    Cached solve function for ``sympy`` equations, for efficiency.

    ``rational=False`` keeps float coefficients as floats, avoiding expensive
    ``Float`` -> ``Rational`` conversions in ``sympy``; fine here as solutions
    are only used numerically (with tolerance-based comparisons).
    """
    return solve(equation, variable, rational=False)


def _set_spglib_warnings_error_handling_env_var():
    """
    Set the SPGLIB environment variable to use new error handling.
    """
    os.environ["SPGLIB_OLD_ERROR_HANDLING"] = "False"  # can be removed with spglib >=2.8
    os.environ["SPGLIB_WARNING"] = "OFF"


def _check_spglib_version():
    """
    Check the versions of spglib and its C libraries, and raise a warning if
    the correct installation instructions have not been followed.
    """
    python_version = spglib.__version__
    c_version = spglib.spg_get_version_full()

    if python_version != c_version:
        warnings.warn(  # think this issue is avoided with latest spglib versions, but not sure
            f"Your spglib Python version (spglib.__version__ = {python_version}) does not match its C "
            f"library version (spglib.spg_get_version_full() = {c_version}). This can lead to unnecessary "
            f"spglib warning messages, but can be avoided by upgrading spglib with `pip install --upgrade "
            f"spglib`."
        )  # previously also had to do conda or special pip install settings, with spglib <2.5


_set_spglib_warnings_error_handling_env_var()
_check_spglib_version()


def _round_floats(obj, places: int = 5):
    """
    Recursively round floats in a dictionary to ``places`` decimal places,
    using the ``_custom_round`` function.
    """
    if isinstance(obj, float):
        return _custom_round(obj, places) + 0.0
    if isinstance(obj, dict):
        return {k: _round_floats(v, places) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_round_floats(x, places) for x in obj]
    if isinstance(obj, np.ndarray):
        return _vectorized_custom_round(obj, places) + 0.0
    if isinstance(obj, pd.DataFrame):  # if dataframe, convert to dict and round floats
        return pd.DataFrame(_round_floats(obj.to_dict(), places))

    return obj


def _custom_round(number: float, decimals: int = 3):
    """
    Custom rounding function that rounds numbers to a specified number of
    decimals, if that rounded number is within 0.15*10^(-decimals) of the
    original number, else rounds to [decimals+1] decimals.

    Primarily because float rounding with ``pymatgen``/``numpy`` can give cell
    coordinates of 0.5001 instead of 0.5 etc, but also can have coordinates of
    e.g. ``0.6125`` that should not be rounded to ``0.613``.

    Args:
        number (float):
            The number to round.
        decimals (int):
            The number of decimals to round to (default: 3).

    Returns:
        float: The rounded number.
    """
    rounded_number = round(number, decimals)
    if abs(rounded_number - number) < 0.15 * float(10) ** (-decimals):
        return rounded_number

    return round(number, decimals + 1)


_vectorized_custom_round = np.vectorize(_custom_round)


def _get_num_places_for_dist_precision(
    structure: Structure | Lattice, dist_precision: float = 0.001
) -> int:
    """
    Given a structure or lattice, get the number of decimal places that we need
    to keep / can round to for `fractional coordinates` (``frac_coords``), to
    maintain a distance precision of ``dist_precision`` in Å.

    Intended for use with the ``_round_floats()`` function, to achieve cleanly
    formatted structure outputs while ensuring no significant rounding errors
    are introduced in site positions (e.g. for very large supercells, small
    differences in fraction coordinates become significant).

    Args:
        structure (|Structure| | |Lattice|):
            The input structure or lattice.
        dist_precision (float):
            The desired distance precision in Å (default: 0.001).

    Returns:
        int:
            The number of decimal places to keep for fractional coordinates to
            maintain the desired distance precision.
    """
    lattice = structure if isinstance(structure, Lattice) else structure.lattice
    frac_precision = dist_precision / max(lattice.abc)

    # get corresponding number of decimal places for this precision:
    return -1 * min(math.floor(math.log(frac_precision, 10)), -8)  # use 8 dp as max precision


def _round_struct_coords(structure: Structure, dist_precision: float = 0.001, to_unit_cell=False):
    """
    Convenience method to round the lattice parameters and fractional
    coordinates of a structure to a given distance precision, for cleanly
    formatted structure outputs.

    Does not apply this operation in-place!

    Args:
        structure:
            The input structure.
        dist_precision:
            The desired distance precision in Å (default: 0.001).
        to_unit_cell:
            Whether to round the fractional coordinates to the unit cell
            (default: False).

    Returns:
        Structure:
            The structure with rounded lattice parameters and fractional
            coordinates.
    """
    rounded_struct = structure.copy()
    req_places = _get_num_places_for_dist_precision(rounded_struct, dist_precision)
    frac_coords = _round_floats(rounded_struct.frac_coords, places=req_places)
    lattice = Lattice(_round_floats(rounded_struct.lattice.matrix, places=req_places))

    for idx in range(len(rounded_struct)):
        orig_site = structure[idx]
        rounded_struct._sites[idx] = PeriodicSite(
            orig_site.species,
            frac_coords[idx],
            lattice,
            properties=orig_site.properties,
            label=orig_site._label,
            skip_checks=True,
            to_unit_cell=to_unit_cell,
        )

    return rounded_struct


def _frac_coords_sort_func(coords):
    """
    Sorting function to apply on an iterable of fractional coordinates, where
    entries are sorted by the number of x, y, z that are (almost) equal (i.e.
    between 0 and 3), then by the magnitude of x+y+z, then by the magnitudes of
    x, y and z.
    """
    if coords is None:
        return (1e10, 1e10, 1e10, 1e10, 1e10)
    coords_for_sorting = _vectorized_custom_round(
        np.mod(_vectorized_custom_round(coords), 1)
    )  # to unit cell
    num_equals = sum(  # scalar comparisons, matching ``np.isclose(..., atol=1e-3)`` but avoiding overhead
        abs(coords_for_sorting[i] - coords_for_sorting[j]) <= 1e-3
        for i in range(len(coords_for_sorting))
        for j in range(i + 1, len(coords_for_sorting))
    )
    magnitude = _custom_round(np.linalg.norm(coords))
    return (-num_equals, magnitude, *np.abs(coords_for_sorting))


def get_sga(struct: Structure, symprec: float = 0.01) -> SpacegroupAnalyzer:
    """
    Get a ``SpacegroupAnalyzer`` object of the input structure, dynamically
    adjusting ``symprec`` if needs be.

    Note that by default, magnetic symmetry (i.e. MAGMOMs) are not used in
    symmetry analysis in ``doped``, as noise in these values (particularly in
    structures from the Materials Project) often leads to incorrect symmetry
    determinations. To use magnetic moments in symmetry analyses, set the
    environment variable ``USE_MAGNETIC_SYMMETRY=1`` (i.e.
    ``os.environ["USE_MAGNETIC_SYMMETRY"] = "1"`` in Python).

    Args:
        struct (|Structure|):
            The input structure.
        symprec (float):
            The symmetry precision to use (default: 0.01).

    Returns:
        SpacegroupAnalyzer: The symmetry analyzer object.
    """
    return _get_sga(struct, symprec=symprec, return_symprec=False)


def get_sga_and_symprec(struct: Structure, symprec: float = 0.01) -> tuple[SpacegroupAnalyzer, float]:
    """
    Get a ``SpacegroupAnalyzer`` object of the input structure, dynamically
    adjusting ``symprec`` if needs be, and the final successful ``symprec``
    used for ``SpacegroupAnalyzer`` initialisation.

    Note that by default, magnetic symmetry (i.e. MAGMOMs) are not used in
    symmetry analysis in ``doped``, as noise in these values (particularly in
    structures from the Materials Project) often leads to incorrect symmetry
    determinations. To use magnetic moments in symmetry analyses, set the
    environment variable ``USE_MAGNETIC_SYMMETRY=1`` (i.e.
    ``os.environ["USE_MAGNETIC_SYMMETRY"] = "1"`` in Python).

    Args:
        struct (|Structure|):
            The input structure.
        symprec (float):
            The symmetry precision to use (default: 0.01).

    Returns:
        tuple[SpacegroupAnalyzer, float]:
            Tuple of the ``SpacegroupAnalyzer`` object and the final
            ``symprec`` used.
    """
    return _get_sga(struct, symprec=symprec, return_symprec=True)


def _get_sga(
    struct: Structure, symprec: float = 0.01, return_symprec: bool = False
) -> SpacegroupAnalyzer | tuple[SpacegroupAnalyzer, float]:
    return _cache_ready_get_sga(
        struct,
        symprec=symprec,
        return_symprec=return_symprec,
        use_magnetic_symmetry=(os.environ.get("USE_MAGNETIC_SYMMETRY", "0") == "1"),  # default no mag symm
    )


@lru_cache(maxsize=int(1e3))
def _cache_ready_get_sga(
    struct: Structure,
    symprec: float = 0.01,
    return_symprec: bool = False,
    use_magnetic_symmetry: bool = False,
) -> SpacegroupAnalyzer | tuple[SpacegroupAnalyzer, float]:
    """
    ``get_sga`` code, with hashable input arguments for caching (using
    |Structure| hash function from ``doped.utils.efficiency``).
    """
    if not use_magnetic_symmetry:  # don't use magnetic symmetry by default
        struct = struct.copy()
        for site in struct:
            site.properties = {}

    sga = None
    trial_symprecs = [symprec, 0.1, 0.001, 1, 0.0001]
    spg_2pt7 = False
    symm_error_types: tuple[type[Exception], ...] = (SymmetryUndeterminedError, ValueError)
    with contextlib.suppress(AttributeError):  # introduced with spglib 2.7.0, can remove once spglib
        symm_error_types += (spglib.SpglibError,)  # (indirect) requirement is >= 2.7
        spg_2pt7 = True

    for trial_symprec in trial_symprecs:
        try:  # if symmetry determination fails, increase symprec first, then decrease, then criss-cross
            sga = SpacegroupAnalyzer(struct, symprec=trial_symprec)
            # check symmetry determination, sometimes SpacegroupAnalyzer initialises but methods fail:
            _detected_symmetry = sga._get_symmetry()
            return (sga, trial_symprec) if return_symprec else sga
        except symm_error_types as latest_symm_error:
            symm_error = latest_symm_error  # save before auto-deleted at end of except block
            continue

    raise SymmetryUndeterminedError(
        "Could not determine symmetry of input structure!"
        + (f"Got spglib error: {spglib.get_error_message()}" if not spg_2pt7 else "")
    ) from symm_error


def apply_symm_op_to_site(
    symm_op: SymmOp,
    site: PeriodicSite,
    fractional: bool = False,
    rotate_lattice: Lattice | bool = True,
    just_unit_cell_frac_coords: bool = False,
) -> PeriodicSite:
    """
    Apply the given symmetry operation to the input site (**not in place**) and
    return the new site.

    By default, also rotates the lattice accordingly. If you want to apply the
    symmetry operation but keep the same lattice definition, set
    ``rotate_lattice=False``.

    Args:
        symm_op (SymmOp):
            ``pymatgen`` ``SymmOp`` object.
        site (|PeriodicSite|):
            ``pymatgen`` |PeriodicSite| object.
        fractional (bool):
            If the ``SymmOp`` is in fractional or Cartesian (default)
            coordinates (i.e. to apply to ``site.frac_coords`` or
            ``site.coords``). Default: False
        rotate_lattice (|Lattice| | bool):
            Either a ``pymatgen`` |Lattice| object (to use as the new lattice
            basis of the transformed site, which can be provided to reduce
            computation time when looping) or ``True/False``. If ``True``
            (default), the ``SymmOp`` rotation matrix will be applied to the
            input site lattice, or if ``False``, the original lattice will be
            retained.
        just_unit_cell_frac_coords (bool):
            If ``True``, just returns the `fractional coordinates` of the
            transformed site (rather than the site itself), within the unit
            cell. Default: False

    Returns:
        PeriodicSite:
            Site with the symmetry operation applied.
    """
    if isinstance(rotate_lattice, Lattice):
        rotated_lattice = rotate_lattice
    else:
        if rotate_lattice:
            if fractional:
                rotated_lattice = Lattice(np.dot(symm_op.rotation_matrix, site.lattice.matrix))
            else:
                rotated_lattice = Lattice(
                    [symm_op.apply_rotation_only(row) for row in site.lattice.matrix]
                )
        else:
            rotated_lattice = site.lattice

    if fractional:  # operate in **original** lattice, then convert to new lattice
        frac_coords = symm_op.operate(site.frac_coords)
        new_coords = site.lattice.get_cartesian_coords(frac_coords)
    else:
        new_coords = symm_op.operate(site.coords)

    if just_unit_cell_frac_coords:
        rotated_frac_coords = rotated_lattice.get_fractional_coords(new_coords)
        return np.array(
            [
                np.mod(f, 1) if p else f
                for p, f in zip(rotated_lattice.pbc, rotated_frac_coords, strict=False)
            ]
        )

    return PeriodicSite(
        site.species,
        new_coords,
        rotated_lattice,
        coords_are_cartesian=True,
        properties=site.properties,
        skip_checks=True,
        label=site._label,
    )


def apply_symm_op_to_struct(
    symm_op: SymmOp, struct: Structure, fractional: bool = False, rotate_lattice: bool = True
) -> Structure:
    """
    Apply a symmetry operation to a structure and return the new structure.

    This differs from pymatgen's ``apply_operation`` method in that it **does
    not apply the operation in place as well (i.e. does not modify the input
    structure)**, which avoids the use of unnecessary and slow
    ``Structure.copy()`` calls, making the structure manipulation / symmetry
    analysis functions more efficient. Also fixes an issue when applying
    fractional symmetry operations.

    By default, also rotates the lattice accordingly. If you want to apply the
    symmetry operation to the sites but keep the same lattice definition, set
    ``rotate_lattice=False``.

    Args:
        symm_op:
            ``pymatgen`` ``SymmOp`` object.
        struct:
            ``pymatgen`` |Structure| object.
        fractional:
            If the ``SymmOp`` is in fractional or Cartesian (default)
            coordinates (i.e. to apply to ``site.frac_coords`` or
            ``site.coords``). Default: False
        rotate_lattice:
            If the lattice of the input structure should be rotated according
            to the symmetry operation. Default: True.

    Returns:
        Structure:
            |Structure| with the symmetry operation applied.
    """
    # using modified version of ``pymatgen``\'s ``apply_operation`` method:
    if rotate_lattice:
        if not fractional:
            rotated_lattice = Lattice([symm_op.apply_rotation_only(row) for row in struct._lattice.matrix])
        else:
            rotated_lattice = Lattice(np.dot(symm_op.rotation_matrix, struct._lattice.matrix))
    else:
        rotated_lattice = struct._lattice

    # note could also use ``SymmOp.operate_multi`` for speedup if ever necessary, but requires some more
    # accounting of species ordering etc, and this isn't an efficiency bottleneck currently
    return Structure.from_sites(
        [
            apply_symm_op_to_site(symm_op, site, fractional=fractional, rotate_lattice=rotated_lattice)
            for site in struct
        ]
    )


def summed_dist(
    struct_a: Structure, struct_b: Structure, ignored_species: list[str] | None = None
) -> float:
    """
    Get the summed distance between closest-matched sites of two structures, in
    Å.

    Note that this assumes the lattices of the two structures are equal!

    Args:
        struct_a: ``pymatgen`` |Structure| object.
        struct_b: ``pymatgen`` |Structure| object.
        ignored_species:
            List of species to ignore when calculating the summed distance
            (default: None).

    Returns:
        float:
            The summed distance between the sites of the two structures, in Å,
            or ``inf`` if any site could not be matched (i.e. the structures
            have differing compositions).
    """
    # This is orders of magnitude faster than StructureMatcher.get_rms_dist() from pymatgen (though this
    # assumes lattices are equal). Threshold set to a large number to avoid possible site-matching warnings
    site_mappings = get_site_mappings(struct_a, struct_b, threshold=1e10, ignored_species=ignored_species)
    return float(sum(dist if dist is not None else float("inf") for dist, _i, _j in site_mappings))


def get_distance_matrix(fcoords: ArrayLike, lattice: Lattice) -> np.ndarray:
    """
    Get a matrix of the distances between the input fractional coordinates in
    the input lattice.

    Args:
        fcoords (ArrayLike):
            Fractional coordinates to get distances between.
        lattice (|Lattice|):
            Lattice for the fractional coordinates.

    Returns:
        np.ndarray:
            Matrix of distances between the input fractional coordinates in the
            input lattice.
    """
    # tuple-ify for caching:
    return _get_distance_matrix(tuple(tuple(row) for row in np.asarray(fcoords)), lattice)


@lru_cache(maxsize=int(1e2))
def _get_distance_matrix(fcoords: tuple[tuple, ...], lattice: Lattice):
    """
    Get a matrix of the distances between the input fractional coordinates in
    the input lattice.

    This function requires the input fcoords to be given as tuples, to allow
    hashing and caching for efficiency.
    """
    dist_matrix = np.array(lattice.get_all_distances(fcoords, fcoords))
    dist_matrix = (dist_matrix + dist_matrix.T) / 2  # ensure ij symmetry
    dist_matrix.flags.writeable = False  # cached array shared across callers; prevent mutation
    return dist_matrix


def cluster_coords(
    fcoords: ArrayLike,
    structure: Structure | Lattice,
    dist_tol: float = 0.01,
    method: str = "single",
    criterion: str = "distance",
) -> np.ndarray:
    """
    Cluster fractional coordinates based on their distances (using ``scipy``
    functions) and return the cluster numbers (as an array matching the shape
    and order of ``fcoords``).

    ``method`` chooses the clustering algorithm to use with ``linkage()``
    (``"single"`` by default, matching the ``scipy`` default), along with a
    ``dist_tol`` distance tolerance in Å. ``"single"`` corresponds to the
    Nearest Point algorithm and is the recommended choice for ``method`` when
    ``dist_tol`` is small, but can be sensitive to how many fractional
    coordinates are included in ``fcoords`` (allowing for daisy-chaining of
    sites to give large spaced-out clusters), while ``"average"`` or
    ``"complete"`` (furthest point algorithm) are good choices to avoid this
    issue. ``"centroid"``/``"median"``/``"ward"`` should not be used for
    ``method`` as they assume a flat Euclidean space, which is violated with
    PBC distances.

    See the ``scipy`` API docs for more info.

    Args:
        fcoords (ArrayLike):
            Fractional coordinates to cluster.
        structure (|Structure| | |Lattice|):
            |Structure| or |Lattice| to which the fractional coordinates
            correspond.
        dist_tol (float):
            Distance tolerance for clustering, in Å (default: 0.01). For the
            most part, fractional coordinates with distances less than this
            tolerance will be clustered together (when ``method = "single"``,
            giving the Nearest Point algorithm, as is the default).
        method (str):
            Clustering algorithm to use with ``linkage()``. Default is
            ``"single"`` (recommended for small ``dist_tol``), while
            ``"average"`` or ``"complete"`` are recommended with medium/large
            ``dist_tol`` (e.g. for candidate interstitial site clustering or
            defect site clustering (for determining defect site competition)).
            ``"centroid"``/``"median"``/``"ward"`` should not be used for
            as they assume a flat Euclidean space, which is violated with PBC
            distances.
        criterion (str):
            Criterion to use for flattening hierarchical clusters from the
            linkage matrix, used with ``fcluster()``. Default: ``"distance"``.

    Returns:
        np.ndarray:
            Array of cluster numbers, matching the shape and order of
            ``fcoords`` (i.e. corresponding to the index/number of the cluster
            to which that fractional coordinate belongs).
    """
    fcoords = np.asarray(fcoords)
    if len(fcoords) == 1:  # only one input coordinate
        return np.array([0])

    lattice = structure if isinstance(structure, Lattice) else structure.lattice
    condensed_m = squareform(get_distance_matrix(fcoords, lattice), checks=False)
    z = linkage(condensed_m, method=method)
    # Note: with method = "single", the z distances are the minimum pairwise distance between any point in
    # one and any point in the other cluster (so two clusters should merge when any site in one is within
    # ``dist_tol`` of any site in the other, and kept separate when all points in one are >``dist_tol``
    # away from all points in the other), which of course can easily lead to daisy-chaining (for medium /
    # large dist_tol values).
    # With method = "complete", the z distances are instead the maximum pairwise distance between any point
    # in one and any point in the other cluster (so two clusters should merge only when _every_ site in one
    # is within ``dist_tol`` of _every_ site in the other cluster). Clusters are thus compact and bounded
    # -- the diameter of any cluster is guaranteed <= ``dist_tol``, with every pair of sites within a
    # cluster within ``dist_tol`` of each other.
    # With method = "average", the z distances are the mean pairwise distance across all point pairs (one
    # from each cluster), so two clusters should merge only when the average distance drops below
    # ``dist_tol``. This is a compromise; less chain-prone than "single", less outlier-sensitive than
    # "complete". Mean within-cluster distance is controlled, but individual pairs can exceed ``dist_tol``.
    return fcluster(z, dist_tol, criterion=criterion)


def doped_cluster_frac_coords(
    fcoords: np.typing.ArrayLike,
    structure: Structure,
    tol: float = 0.55,
    symm_pref_dist_factor: float = 0.85,
    method: str = "average",
    criterion: str = "distance",
) -> np.ndarray:
    """
    Cluster fractional coordinates that are within a certain distance tolerance
    of each other, and return the cluster site.

    Modified from the ``pymatgen-analysis-defects``` function as follows:
    For each site cluster, the possible sites to choose from are the sites
    in the cluster `and` the cluster midpoint (average position). Of these
    sites, the site with the highest symmetry, and then largest ``min_dist``
    (distance to any host lattice site), is chosen -- if its ``min_dist`` is
    no more than ``symm_pref_dist_factor`` (0.85 by default) times the largest
    possible ``min_dist``. This is because we want to favour the higher
    symmetry interstitial sites (as these are typically the more intuitive
    sites for placement, cleaner, easier for analysis etc, and work well when
    combined with |ShakeNBreak| or other structure-searching techniques to
    account for symmetry-breaking), but also interstitials are often
    lowest-energy when furthest from host atoms (i.e. in the largest
    interstitial voids -- particularly for fully-ionised charge states), and so
    this approach tries to strike a balance between these two goals.

    In ``pymatgen-analysis-defects``, the average cluster position is used,
    which breaks symmetries and is less easy to manipulate in the following
    interstitial generation functions. ``pymatgen-analysis-defects`` also uses
    the default ``"single"`` method for site clustering, which can lead to
    large unwanted daisy-chaining effects, unintentionally grouping
    interstitials with distances far larger than ``tol``.

    Args:
        fcoords (ArrayLike):
            Fractional coordinates of points to cluster.
        structure (|Structure|):
            The host structure.
        tol (float):
            Distance tolerance for clustering Voronoi nodes. Default is 0.55 Å.
        symm_pref_dist_factor (float):
            Minimum acceptable ratio of distance to host atoms for
            symmetry-favoured sites vs distance-to-host-favoured sites, for
            which to prefer symmetry-favoured sites. Default is 0.85.
        method (str):
            Clustering algorithm to use with ``linkage()``. Default is
            ``"average"``, which is typically better than the ``scipy`` default
            of ``"single`` for interstitial generation, as it avoids
            unintentional daisy-chaining effects. Another reasonable choice is
            ``"complete"``, which ensures that no two sites in a given cluster
            are more than ``tol`` apart. See the docstrings and source code of
            :func:`~doped.utils.symmetry.cluster_coords` for more details.
            ``"centroid"``/``"median"``/``"ward"`` should not be used for
            as they assume a flat Euclidean space, which is violated with PBC
            distances.
        criterion (str):
            Criterion to use for flattening hierarchical clusters from the
            linkage matrix, used with ``fcluster()`` Default is ``"distance"``.

    Returns:
        np.ndarray: Clustered fractional coordinates.
    """
    fcoords = np.asarray(fcoords)
    if len(fcoords) == 0:
        return np.array([])
    if len(fcoords) == 1:
        return _vectorized_custom_round(np.mod(_vectorized_custom_round(fcoords, 5), 1), 4)  # to unit cell

    lattice = structure.lattice
    cn = cluster_coords(fcoords, structure, dist_tol=tol, method=method, criterion=criterion)
    unique_fcoords = []

    # cn is an array of cluster numbers, of length ``len(fcoords)``, so we take the set of cluster numbers
    # ``n``, use ``np.where(cn == n)[0]`` to get the indices of ``cn`` / ``fcoords`` which are in cluster
    # ``n``, and then decide which coordinates to take as the cluster site based on symmetry and distance:
    for n in set(cn):
        frac_coords = []
        for i, j in enumerate(np.where(cn == n)[0]):
            if i == 0:
                frac_coords.append(fcoords[j])
            else:
                fcoord = fcoords[j]  # We need the image to combine the frac_coords properly:
                _d, image = lattice.get_distance_and_image(frac_coords[0], fcoord)
                frac_coords.append(fcoord + image)

        frac_coords.append(np.average(frac_coords, axis=0))  # midpoint of cluster
        frac_coords_scores = {
            tuple(x): (
                -group_order_from_schoenflies(
                    point_symmetry_from_site(x, structure)
                ),  # higher order = higher symmetry
                -np.min(lattice.get_all_distances(x, structure.frac_coords), axis=1),
                *_frac_coords_sort_func(x),
            )
            for x in frac_coords
        }
        symmetry_favoured_site = sorted(frac_coords_scores.items(), key=lambda x: x[1])[0][0]
        dist_favoured_site = sorted(
            frac_coords_scores.items(), key=lambda x: (x[1][1], x[1][0], *x[1][2:])
        )[0][0]

        if (
            np.min(lattice.get_all_distances(symmetry_favoured_site, structure.frac_coords), axis=1)
            / np.min(lattice.get_all_distances(dist_favoured_site, structure.frac_coords), axis=1)
        ) < symm_pref_dist_factor:
            unique_fcoords.append(dist_favoured_site)
        else:  # prefer symmetry over distance if difference is sufficiently small
            unique_fcoords.append(symmetry_favoured_site)

    return _vectorized_custom_round(
        np.mod(_vectorized_custom_round(unique_fcoords, 5), 1), 4
    )  # to unit cell


def get_all_equiv_sites(
    frac_coords: ArrayLike,
    structure: Structure,
    symprec: float = 0.01,
    dist_tol_factor: float = 1.0,
    species: str = "X",
    just_frac_coords: bool = False,
    return_symprec_and_dist_tol_factor: bool = False,
    fixed_symprec_and_dist_tol_factor: bool = False,
    verbose: bool = False,
    fold_to_primitive: bool = True,
) -> list[PeriodicSite | np.ndarray] | tuple[list[PeriodicSite | np.ndarray], float, float]:
    """
    Get a list of all equivalent sites of the input fractional coordinates in
    ``structure``.

    If ``fold_to_primitive`` is ``True`` (default) and ``structure`` is a
    supercell of a smaller primitive cell, the site orbit is generated in the
    (orientation-preserving) primitive cell and then expanded back to
    ``structure`` -- giving the complete orbit even in periodicity-breaking
    supercells, where direct supercell symmetry analysis undercounts orbits (as
    ``spglib`` can only use symmetry operations with integer rotation matrices
    in the given cell basis).

    Tries to use hashing and caching to accelerate if possible.

    Args:
        frac_coords (ArrayLike):
            Fractional coordinates to get equivalent sites of.
        structure (|Structure|):
            |Structure| to use for the lattice, to which the fractional
            coordinates correspond, and for determining symmetry operations
            if not provided.
        symprec (float):
            Symmetry precision to use for determining symmetry operations.
            Default is 0.01. If ``fixed_symprec_and_dist_tol_factor`` is
            ``False`` (default), this value will be automatically adjusted (up
            to 10x, down to 0.1x) until the identified equivalent sites from
            ``spglib`` have consistent point group symmetries. Setting
            ``verbose`` to ``True`` will print information on the trialled
            ``symprec`` (and ``dist_tol_factor`` values), and setting
            ``return_symprec_and_dist_tol_factor`` to ``True`` will return the
            final ``symprec`` (and ``dist_tol_factor``) used for the equivalent
            site generation.
        dist_tol_factor (float):
            Distance tolerance for clustering generated sites (to ensure they
            are truly distinct), as a multiplicative factor of ``symprec``.
            Default is 1.0 (i.e. ``dist_tol = symprec``, in Å). If
            ``fixed_symprec_and_dist_tol_factor`` is ``False`` (default), this
            value will also be automatically adjusted if necessary (up to 10x,
            down to 0.1x)(after ``symprec`` adjustments) until the identified
            equivalent sites from ``spglib`` have consistent point group
            symmetries. Setting ``verbose`` to ``True`` will print information
            on the trialled ``dist_tol_factor`` (and ``symprec``) values, and
            setting ``return_symprec_and_dist_tol_factor`` to ``True`` will
            return the final ``symprec`` (and ``dist_tol_factor``) used for
            the equivalent site generation.
        species (str):
            Species to use for the equivalent sites (default: "X").
        just_frac_coords (bool):
            If ``True``, just returns the fractional coordinates of the
            equivalent sites (rather than ``pymatgen`` |PeriodicSite|
            objects). Default: False.
        return_symprec_and_dist_tol_factor (bool):
            If ``True``, returns the final symmetry precision and distance
            tolerance factor used for the equivalent site generation (see
            ``symprec`` and ``dist_tol_factor`` argument descriptions). Default
            is ``False``.
        fixed_symprec_and_dist_tol_factor (bool):
            If ``True``, uses the provided ``symprec`` and ``dist_tol_factor``
            values without any automatic adjustments (see ``symprec`` and
            ``dist_tol_factor`` argument descriptions). Default is ``False``.
        verbose (bool):
            If ``True``, prints information on the trialled ``symprec`` and
            ``dist_tol_factor`` values, and the identified equivalent sites.
            Default is ``False``.
        fold_to_primitive (bool):
            If ``True`` (default) and ``structure`` is a supercell of a smaller
            primitive cell, generate the site orbit in the
            (orientation-preserving) primitive cell and expand back to
            ``structure``, giving the complete orbit even in
            periodicity-breaking supercells (see docstring above). If
            ``False``, uses direct symmetry analysis of ``structure``.

    Returns:
        list[PeriodicSite | np.ndarray]:
            List of equivalent sites of the input fractional coordinates in
            ``structure``, either as ``pymatgen`` |PeriodicSite| objects or
            as fractional coordinates (depending on the value of
            ``just_frac_coords``).

        If ``return_symprec_and_dist_tol_factor`` is ``True`` (default is
        ``False``), also returns the final ``symprec`` and ``dist_tol_factor``
        values used for the equivalent site generation.
    """
    args = (
        structure,
        symprec,
        dist_tol_factor,
        species,
        just_frac_coords,
        return_symprec_and_dist_tol_factor,
        fixed_symprec_and_dist_tol_factor,
        verbose,
        fold_to_primitive,
    )
    try:  # check hashability upfront, to avoid catching unrelated ``TypeError``s from the function body
        key = (tuple(cast("Sequence", frac_coords)), *args)
        hash(key)
    except TypeError:  # issue with hashing (possibly due to ``species`` choice), use raw function
        return _raw_get_all_equiv_sites(frac_coords, *args)
    output = _cache_ready_get_all_equiv_sites(*key)
    if return_symprec_and_dist_tol_factor:
        return (list(output[0]), *output[1:])
    return list(output)  # fresh list (incl. cache hits) so caller mutation can't corrupt the cache


@lru_cache(maxsize=int(1e3))
def _cache_ready_get_all_equiv_sites(
    frac_coords: tuple,
    structure: Structure,
    symprec: float = 0.01,
    dist_tol_factor: float = 1.0,
    species: str = "X",
    just_frac_coords: bool = False,
    return_symprec_and_dist_tol_factor: bool = False,
    fixed_symprec_and_dist_tol_factor: bool = False,
    verbose: bool = False,
    fold_to_primitive: bool = True,
) -> list[PeriodicSite | np.ndarray] | tuple[list[PeriodicSite | np.ndarray], float, float]:
    return _raw_get_all_equiv_sites(
        frac_coords,
        structure,
        symprec,
        dist_tol_factor,
        species,
        just_frac_coords,
        return_symprec_and_dist_tol_factor,
        fixed_symprec_and_dist_tol_factor,
        verbose,
        fold_to_primitive,
    )


def _get_orientation_preserving_primitive(
    structure: Structure, symprec: float = 0.01
) -> tuple[Structure, np.ndarray] | None:
    """
    Get the orientation-preserving primitive cell of ``structure`` (via
    ``spglib`` with ``no_idealize=True``, keeping the original Cartesian
    orientation and origin), along with the integer primitive-to-supercell.

    transformation matrix ``M`` (such that:
    ``structure.lattice.matrix = M @ primitive.lattice.matrix``).

    Returns ``None`` if ``structure`` is already primitive or ``spglib``
    primitive cell determination fails.
    """
    cell = (structure.lattice.matrix, structure.frac_coords, [site.specie.Z for site in structure])
    prim_cell = spglib.standardize_cell(cell, to_primitive=True, no_idealize=True, symprec=symprec)
    if prim_cell is None or len(prim_cell[2]) >= len(structure):
        return None

    prim_lattice = Lattice(prim_cell[0])
    supercell_matrix = structure.lattice.matrix @ np.linalg.inv(prim_lattice.matrix)
    int_supercell_matrix = np.rint(supercell_matrix)
    if not np.allclose(supercell_matrix, int_supercell_matrix, atol=0.01):
        raise ValueError(  # shouldn't happen for a true orientation-preserving primitive
            f"Non-integer supercell matrix ({supercell_matrix}) between the input structure and its "
            f"orientation-preserving primitive cell!"
        )
    return Structure(prim_lattice, list(prim_cell[2]), prim_cell[1]), int_supercell_matrix.astype(int)


_TRIAL_SYMPREC_DIST_TOL_FACTORS = np.array([1, 1.05, 0.95, 1.1, 0.9, 1.2, 0.8, 1.5, 0.75, 2, 0.5, 10, 0.1])


def _orbit_site_symmetry_consistent(
    structure: Structure,
    n_equiv_sites: int,
    site_symmetry_symbol: str,
    symprec: float = 0.01,
) -> bool:
    """
    Check the Wyckoff orbit-stabilizer relation for a generated site orbit: the
    per-primitive-cell orbit multiplicity times the site point group order must
    equal the host crystal point group order.

    This catches undercounted orbits and (more commonly) under-certified site
    symmetries from slightly-noisy site coordinates -- e.g. a parsed (relaxed)
    interstitial site sitting ~``symprec`` off its ideal position, where site
    symmetry analysis gives a spurious subgroup `consistently` for all orbit
    sites (so uniformity alone cannot catch it).

    Returns ``True`` if consistent, or if the relation cannot be evaluated
    (e.g. ``spglib`` failure) -- only a definite violation returns ``False``.
    """
    try:
        site_pg_order = group_order_from_schoenflies(schoenflies_from_hermann(site_symmetry_symbol))
        prim_and_matrix = _get_orientation_preserving_primitive(structure, symprec=symprec)
        host = structure if prim_and_matrix is None else prim_and_matrix[0]
        n_prim = round(len(structure) / len(host))
        host_pg_order = group_order_from_schoenflies(
            schoenflies_from_hermann(get_sga(host, symprec=symprec).get_point_group_symbol())
        )
    except Exception:  # can't evaluate (unrecognised symbol, spglib failure...); don't block acceptance
        return True
    return n_equiv_sites * site_pg_order == host_pg_order * n_prim


def _raw_get_all_equiv_sites(
    frac_coords: ArrayLike,
    structure: Structure,
    symprec: float = 0.01,
    dist_tol_factor: float = 1.0,
    species: str = "X",
    just_frac_coords: bool = False,
    return_symprec_and_dist_tol_factor: bool = False,
    fixed_symprec_and_dist_tol_factor: bool = False,
    verbose: bool = False,
    fold_to_primitive: bool = True,
) -> list[PeriodicSite | np.ndarray] | tuple[list[PeriodicSite | np.ndarray], float, float]:
    # ensure sites have the same property keys, otherwise can cause issues with pymatgen primitive
    # structure determination:
    if (
        "magmom" in structure.site_properties
    ):  # if species matches those in structure, and all the same, then use the
        # same magmom, otherwise remove magmom from properties
        matching_sites = [site for site in structure if site.species_string == str(species)]
        if matching_sites and np.std([site.properties["magmom"] for site in matching_sites]) < 0.1:
            properties = {"magmom": next(site.properties.get("magmom", 0) for site in matching_sites)}
        else:
            properties = {"magmom": 0}
    else:
        properties = {}

    def _clustered_orbit(structure: Structure, coords: ArrayLike, symprec: float, dist_tol: float):
        """
        Generate the orbit of ``coords`` under the symmetry operations of
        ``structure`` (i.e. the set of symmetry-equivalent positions), as
        deduplicated `unit-cell` fractional coordinates.
        """
        sga = get_sga_and_symprec(structure, symprec=symprec)[0]
        return cluster_sites_by_dist_tol(
            [
                symm_op.operate(coords) % 1  # apply symm_op and move to unit cell
                for symm_op in sga.get_symmetry_operations()  # fractional symm_ops by default
            ],
            structure,
            dist_tol=dist_tol,
        )

    def _fold_to_primitive_equiv_sites(symprec: float, dist_tol: float):
        """
        Generate the complete orbit of ``frac_coords`` in ``structure`` by
        folding to the orientation-preserving primitive cell, generating the
        orbit there with the primitive symmetry operations, and expanding back
        to ``structure`` with the primitive lattice translations.

        This gives the complete orbit even in periodicity-breaking supercells,
        where direct supercell symmetry analysis undercounts orbits (as
        ``spglib`` can only use symmetry operations with integer rotation
        matrices in the given cell basis). Returns the orbit as unit-cell
        fractional coordinates, or ``None`` if ``structure`` is already
        primitive (no folding possible/needed).
        """
        if (prim_and_matrix := _get_orientation_preserving_primitive(structure, symprec=symprec)) is None:
            return None  # already primitive (or spglib failure); use direct supercell analysis
        prim, int_supercell_matrix = prim_and_matrix
        prim_frac_coords = np.asarray(frac_coords) @ int_supercell_matrix  # f_prim = f_super @ M
        prim_orbit = _clustered_orbit(prim, prim_frac_coords, symprec, dist_tol)

        # expand back up to ``structure``; supercell frac coords of each orbit member, plus all primitive
        # lattice translations within the supercell (i.e. the coset representatives):
        coset_translations = lattice_points_in_supercell(int_supercell_matrix)
        all_frac_coords = (
            np.array(prim_orbit) @ np.linalg.inv(int_supercell_matrix) + coset_translations[:, None]
        ).reshape(-1, 3) % 1
        return cluster_sites_by_dist_tol(list(all_frac_coords), structure, dist_tol=dist_tol)

    def _get_equiv_sites_with_given_symprec(
        symprec: float,
        dist_tol_factor: float,
        just_frac_coords: bool = False,
    ):
        dist_tol = dist_tol_factor * symprec  # distance tolerance for clustering sites
        orbit = None
        if fold_to_primitive:
            try:
                orbit = _fold_to_primitive_equiv_sites(symprec, dist_tol)
            except Exception as exc:
                warnings.warn(
                    f"Equivalent-site generation via primitive-cell folding failed with error: {exc!r}. "
                    f"Falling back to direct symmetry analysis of the input structure, which can miss "
                    f"equivalent sites in periodicity-breaking supercells."
                )
        if orbit is None:  # already primitive, folding disabled, or folding failed
            orbit = _clustered_orbit(structure, frac_coords, symprec, dist_tol)

        return (
            orbit
            if just_frac_coords
            else [
                PeriodicSite(species, site_frac_coords, structure.lattice, properties=properties)
                for site_frac_coords in orbit
            ]
        )

    if fixed_symprec_and_dist_tol_factor:
        equiv_sites = _get_equiv_sites_with_given_symprec(
            symprec, dist_tol_factor, just_frac_coords=just_frac_coords
        )
        return (
            (equiv_sites, symprec, dist_tol_factor) if return_symprec_and_dist_tol_factor else equiv_sites
        )

    # the choice of equivalent sites should give consistent site symmetries for each equivalent site (using
    # the same ``symprec`` as for generation), however this is sometimes not the case (due to small
    # numerical noise / ``dist_tol`` choices etc), so check that the site symmetries (according to
    # ``symprec``) are self-consistent, and adjust ``symprec`` if not:
    trial_symprecs = _TRIAL_SYMPREC_DIST_TOL_FACTORS * symprec
    trial_dist_tol_factors = _TRIAL_SYMPREC_DIST_TOL_FACTORS * dist_tol_factor
    fallback = None  # first uniform-site-symmetry result failing the orbit-stabilizer check, as fallback
    for trial_dist_tol_factor, trial_symprec in product(trial_dist_tol_factors, trial_symprecs):
        equiv_sites = _get_equiv_sites_with_given_symprec(
            trial_symprec, trial_dist_tol_factor, just_frac_coords=False
        )
        struct_with_all_X = _get_struct_with_all_X(structure, equiv_sites)
        sga_with_all_X = get_sga(struct_with_all_X, symprec=trial_symprec)
        site_sym_symbols = sga_with_all_X.get_symmetry_dataset().site_symmetry_symbols[-len(equiv_sites) :]
        if len(set(site_sym_symbols)) == 1:
            if _orbit_site_symmetry_consistent(
                structure, len(equiv_sites), site_sym_symbols[0], trial_symprec
            ):
                symprec = trial_symprec
                dist_tol_factor = trial_dist_tol_factor
                equiv_sites = [s.frac_coords for s in equiv_sites] if just_frac_coords else equiv_sites
                if verbose:
                    print(
                        f"Equivalent site generation succeeded (with consistent site symmetries) with "
                        f"symprec = {symprec} & dist_tol_factor = {dist_tol_factor}, giving "
                        f"{len(equiv_sites)} equivalent sites in the input structure."
                    )
                break
            if fallback is None:  # uniform site symmetries, but violating the orbit-stabilizer relation;
                # keep as fallback in case no trial satisfies both criteria:
                fallback = (equiv_sites, trial_symprec, trial_dist_tol_factor)
            if verbose:
                print(
                    f"Equivalent site generation gave uniform site symmetries but violated the "
                    f"orbit-stabilizer relation with symprec = {trial_symprec} & dist_tol_factor = "
                    f"{trial_dist_tol_factor}, giving {len(equiv_sites)} equivalent sites in the input "
                    f"structure."
                )
            continue

        if verbose:
            print(
                f"Equivalent site generation failed with symprec = {trial_symprec} & dist_tol_factor "
                f"= {trial_dist_tol_factor}, giving {len(equiv_sites)} equivalent sites in the input "
                f"structure."
            )
    else:  # no trial passed both checks; fall back to the first uniform-site-symmetry result if any
        if fallback is not None:
            equiv_sites, symprec, dist_tol_factor = fallback
            equiv_sites = [s.frac_coords for s in equiv_sites] if just_frac_coords else equiv_sites

    return (equiv_sites, symprec, dist_tol_factor) if return_symprec_and_dist_tol_factor else equiv_sites


def cluster_sites_by_dist_tol(
    sites: Iterable[PeriodicSite | np.ndarray],
    structure: Structure | Lattice,
    dist_tol: float = 0.01,
    method: str = "single",
    criterion: str = "distance",
) -> list[PeriodicSite | np.ndarray]:
    r"""
    Cluster sites based on their distances (using ``cluster_coords``).

    Args:
        sites (Iterable[|PeriodicSite| | np.ndarray]):
            Sites to cluster, as an iterable of |PeriodicSite| objects or
            fractional coordinates.
        structure (|Structure| | |Lattice|):
            |Structure| or |Lattice| to which the sites correspond.
        dist_tol (float):
            Distance tolerance for clustering, in Å (default: 0.01).
        method (str):
            Clustering algorithm to use with ``scipy``\'s ``linkage()``
            clustering function in ``cluster_coords``. Default is ``"single"``,
            which is the ``scipy`` default and is typically recommended when
            ``dist_tol`` is small. See the docstrings and source code of
            :func:`~doped.utils.symmetry.cluster_coords` for more details.
        criterion (str):
            Criterion to use for flattening hierarchical clusters from the
            linkage matrix, used with ``fcluster()``. Default: ``"distance"``.

    Returns:
        list[PeriodicSite | np.ndarray]:
            List of clustered sites, as |PeriodicSite| objects or fractional
            coordinates depending on the input ``sites`` type.
    """
    dist_precision_num_places = _get_num_places_for_dist_precision(structure, dist_tol)
    just_frac_coords = not hasattr(next(iter(sites)), "frac_coords")
    sites = list(sites)  # needs to be indexable for reducing to unique sites below
    all_frac_coords = [
        tuple(np.round(i, dist_precision_num_places))
        for i in (
            sites if just_frac_coords else [cast("PeriodicSite", site).frac_coords for site in sites]
        )
    ]
    unique_frac_coords, unique_indices = np.unique(all_frac_coords, axis=0, return_index=True)
    unique_sites = [sites[i] for i in unique_indices]

    cn = cluster_coords(
        unique_frac_coords, structure, dist_tol=dist_tol, method=method, criterion=criterion
    )
    # cn is an array of cluster numbers, of length ``len(unique_frac_coords)``, so we take the set of
    # cluster numbers ``n``, use ``np.where(cn == n)[0]`` to get the indices of ``cn`` /
    # ``unique_frac_coords`` which are in cluster ``n``, and then take the first of each cluster
    # (because here these should be basically the same sites just with possibly small numerical
    # differences due to symmetry operations, unlike when ``cluster_coords`` is used for Voronoi
    # interstitial generation, where we choose the cluster site based on symmetry/distance to host)
    return [unique_sites[np.where(cn == n)[0][0]] for n in set(cn)]  # take 1st of each cluster


def get_min_dist_between_equiv_sites(
    site_1: PeriodicSite | Sequence[float] | Defect | DefectEntry,
    site_2: PeriodicSite | Sequence[float] | Defect | DefectEntry,
    structure: Structure | None = None,
    structure_2: Structure | None = None,
    strip_oxi_states: bool | None = None,
    symprec: float = 0.01,
    dist_tol_factor: float = 1.0,
    return_symprec_and_dist_tol_factor: bool = False,
    fixed_symprec_and_dist_tol_factor: bool = False,
    verbose: bool = False,
) -> float | tuple[float, float, float]:
    """
    Get the minimum distance (in Å) between equivalent sites of two input
    site/|Defect|/|DefectEntry| objects.

    Args:
        site_1 (|PeriodicSite| | Sequence[float, float, float] | |Defect| | |DefectEntry|):
            First site to get equivalent sites of, to determine minimum
            distance to equivalent sites of ``site_2``. Can be a
            |PeriodicSite| object, a sequence of fractional coordinates, or a
            |Defect|/|DefectEntry| object.
        site_2 (|PeriodicSite| | Sequence[float, float, float] | |Defect| | |DefectEntry|):
            Second site to get equivalent sites of, to determine minimum
            distance to equivalent sites of ``site_1``. Can be a
            |PeriodicSite| object, a sequence of fractional coordinates, or a
            |Defect|/|DefectEntry| object.
        structure (|Structure|):
            |Structure| to use for determining symmetry-equivalent sites of
            ``site_1`` (and ``site_2``, if ``structure_2`` is not set).
            Required if ``site_1`` and ``site_2`` are not |Defect| or
            |DefectEntry| objects. Default: None.
        structure_2 (|Structure|):
            Separate host |Structure| for ``site_2``, if the two sites are
            potentially defined in different (but equivalent) host frames --
            e.g. differently-oriented/-defined cells, primitive vs supercell
            definitions, or differently oxi-state-decorated hosts. Each site
            is then folded via its own host into a shared canonical primitive
            cell (from ``get_primitive_structure``) for comparison, returning
            ``np.inf`` if the two hosts do not correspond to matching primitive
            structures. If ``None`` (default), taken from ``site_2`` if it is a
            |Defect|/|DefectEntry| object, otherwise assumed to match
            ``structure``.
        strip_oxi_states (bool | None):
            Whether to strip oxidation states from the host structure(s)
            before symmetry analysis / host matching. If ``None`` (default),
            oxidation states are only stripped when the two host structures
            (``structure``/``structure_2``) have mismatching oxi-state
            decorations (which can otherwise hinder host matching) -- so
            consistently-decorated hosts retain any decoration-dependent
            symmetry (e.g. inequivalent sites in mixed-valence hosts). Set to
            ``True``/``False`` to always/never strip oxidation states.
        symprec (float):
            Symmetry precision to use for determining symmetry operations.
            Default is 0.01. If ``fixed_symprec_and_dist_tol_factor`` is
            ``False`` (default), this value will be automatically adjusted (up
            to 10x, down to 0.1x) until the identified equivalent sites from
            ``spglib`` have consistent point group symmetries. Setting
            ``verbose`` to ``True`` will print information on the trialled
            ``symprec`` (and ``dist_tol_factor`` values), and setting
            ``return_symprec_and_dist_tol_factor`` to ``True`` will return the
            final ``symprec`` (and ``dist_tol_factor``) used for the equivalent
            site generation.
        dist_tol_factor (float):
            Distance tolerance for clustering generated sites (to ensure they
            are truly distinct), as a multiplicative factor of ``symprec``.
            Default is 1.0 (i.e. ``dist_tol = symprec``, in Å). If
            ``fixed_symprec_and_dist_tol_factor`` is ``False`` (default), this
            value will also be automatically adjusted if necessary (up to 10x,
            down to 0.1x)(after ``symprec`` adjustments) until the identified
            equivalent sites from ``spglib`` have consistent point group
            symmetries. Setting ``verbose`` to ``True`` will print information
            on the trialled ``dist_tol_factor`` (and ``symprec``) values, and
            setting ``return_symprec_and_dist_tol_factor`` to ``True`` will
            return the final ``symprec`` (and ``dist_tol_factor``) used for
            the equivalent site generation.
        return_symprec_and_dist_tol_factor (bool):
            If ``True``, returns the final symmetry precision and distance
            tolerance factor used for the equivalent site generation (see
            ``symprec`` and ``dist_tol_factor`` argument descriptions). Default
            is ``False``.
        fixed_symprec_and_dist_tol_factor (bool):
            If ``True``, uses the provided ``symprec`` and ``dist_tol_factor``
            values without any automatic adjustments (see ``symprec`` and
            ``dist_tol_factor`` argument descriptions). Default is ``False``.
        verbose (bool):
            If ``True``, prints information on the trialled ``symprec`` and
            ``dist_tol_factor`` values, and the identified equivalent sites.
            Default is ``False``.

    Returns:
        float | tuple[float, float, float]:
            Minimum distance (in Å) between equivalent sites of ``site_1``
            and ``site_2``, or a tuple of  (minimum distance, ``symprec``,
            ``dist_tol_factor``) if ``return_symprec_and_dist_tol_factor`` is
            ``True``.
    """
    if structure is None:
        for site in [site_2, site_1]:  # if both ``DefectEntry``s/``Defect``s, take structure from site_1
            if isinstance(site, DefectEntry):
                structure = site.defect.structure
            elif isinstance(site, Defect):
                structure = site.structure
    if structure is None:
        raise ValueError(
            "Structure must be provided if site_1 and site_2 are not DefectEntry or Defect objects."
        )
    if structure_2 is None:  # take ``site_2`` host if provided as a ``Defect``/``DefectEntry``:
        if isinstance(site_2, DefectEntry):
            structure_2 = site_2.defect.structure
        elif isinstance(site_2, Defect):
            structure_2 = site_2.structure

    def _parse_site_to_PeriodicSite(site):
        if isinstance(site, DefectEntry):
            return site.defect.site
        if isinstance(site, Defect):
            return site.site
        if isinstance(site, PeriodicSite):
            return site
        return None  # frac coords provided, not site

    def _parse_site_to_frac_coords(site):
        if periodic_site := _parse_site_to_PeriodicSite(site):
            return periodic_site.frac_coords
        return site  # otherwise ``site`` should be frac coords

    primitive = get_primitive_structure(structure)

    if strip_oxi_states is None:  # default: strip only when mismatching decorations
        strip_oxi_states = structure_2 is not None and (
            {str(sp) for sp in structure.composition} != {str(sp) for sp in structure_2.composition}
        )  # compare based on species string sets; Composition equality is oxi-state-insensitive

    if strip_oxi_states:
        structure = structure.copy()
        structure.remove_oxidation_states()
        if structure_2 is not None:
            structure_2 = structure_2.copy()
            structure_2.remove_oxidation_states()
        primitive = get_primitive_structure(structure)

    if different_structures := structure_2 is not None and structure_2 != structure:
        assert structure_2 is not None  # given ``different_structures``; for ``mypy``
        # fold each site via its own host into a shared canonical primitive:
        prim_2 = get_primitive_structure(structure_2)
        if (  # fast-fail for clearly-different host crystals, before matching structures below
            len(prim_2) != len(primitive)
            or prim_2.composition.reduced_formula != primitive.composition.reduced_formula
        ):
            return (np.inf, symprec, dist_tol_factor) if return_symprec_and_dist_tol_factor else np.inf
    else:
        structure_2 = structure

    def _get_equiv_fcoords_symprec_and_dist_tol(
        site, host_structure, symprec=symprec, dist_tol_factor=dist_tol_factor
    ):
        frac_coords = _parse_site_to_frac_coords(site)
        return get_equiv_frac_coords_in_primitive(  # returns ``None`` if no mapping found
            frac_coords,
            primitive,
            host_structure,
            symprec=symprec,
            dist_tol_factor=dist_tol_factor,
            return_symprec_and_dist_tol_factor=True,
            fixed_symprec_and_dist_tol_factor=fixed_symprec_and_dist_tol_factor,
            verbose=verbose,
        )

    with warnings.catch_warnings():
        if different_structures:  # host equivalence not guaranteed; map failure -> ``inf`` (not an error):
            warnings.filterwarnings("ignore", message="Could not find a mapping")
        try:
            output_1 = _get_equiv_fcoords_symprec_and_dist_tol(site_1, structure)
            output_2 = _get_equiv_fcoords_symprec_and_dist_tol(site_2, structure_2) if output_1 else None
        except RuntimeError:  # e.g. ``StructureMatcher.get_transformation()`` failure for similar but
            if not different_structures:  # non-equivalent different host lattices
                raise
            output_1 = output_2 = None

    if output_1 is None or output_2 is None:  # no mapping found between host structure(s) and primitive
        min_dist = np.inf
    else:
        equiv_fcoords_1, symprec, dist_tol_factor = output_1
        equiv_fcoords_2, symprec, dist_tol_factor = output_2
        min_dist = np.min(primitive.lattice.get_all_distances(equiv_fcoords_1, equiv_fcoords_2))

    return (min_dist, symprec, dist_tol_factor) if return_symprec_and_dist_tol_factor else min_dist


def _get_symm_dataset_of_struct_with_all_equiv_sites(
    frac_coords: ArrayLike,
    struct: Structure,
    symprec: float = 0.01,
    dist_tol_factor: float = 1.0,
    species: str = "X",
    return_symprec_and_dist_tol_factor: bool = False,
    fixed_symprec_and_dist_tol_factor: bool = False,
    verbose: bool = False,
    fold_to_primitive: bool = True,
):
    """
    Get the symmetry dataset of a ``SpacegroupAnalyzer`` object of a structure
    with all equivalent sites of the input fractional coordinates added to
    ``struct``, and also returning the list of unique equivalent sites.

    Tries to use hashing and caching to accelerate if possible.

    Returns:
        tuple[SpacegroupDataset, list[PeriodicSite], float, float]:
            Symmetry dataset of the structure with all equivalent sites of
            ``frac_coords`` added, the list of unique equivalent sites, and
            if ``return_symprec_and_dist_tol_factor`` is ``True``, the final
            ``symprec`` and ``dist_tol_factor`` used for the equivalent site
            generation.
    """
    args = (
        struct,
        symprec,
        dist_tol_factor,
        species,
        return_symprec_and_dist_tol_factor,
        fixed_symprec_and_dist_tol_factor,
        verbose,
        fold_to_primitive,
    )
    try:  # check hashability upfront, to avoid catching unrelated ``TypeError``s from the function body
        key = (tuple(cast("Sequence", frac_coords)), *args)
        hash(key)
    except TypeError:  # issue with hashing (possibly due to ``species`` choice), use raw function
        return _raw_get_symm_dataset_of_struct_with_all_equiv_sites(frac_coords, *args)
    output = _cache_ready_get_symm_dataset_of_struct_with_all_equiv_sites(*key)
    # fresh unique-sites list on every call (incl. cache hits) so caller mutation can't corrupt the
    # cache; the symmetry dataset is shared and should be treated as read-only:
    return (output[0], list(output[1]), *output[2:])


def _raw_get_symm_dataset_of_struct_with_all_equiv_sites(
    frac_coords: ArrayLike,
    struct: Structure,
    symprec: float = 0.01,
    dist_tol_factor: float = 1.0,
    species: str = "X",
    return_symprec_and_dist_tol_factor: bool = False,
    fixed_symprec_and_dist_tol_factor: bool = False,
    verbose: bool = False,
    fold_to_primitive: bool = True,
):
    equiv_sites_output = get_all_equiv_sites(
        frac_coords,
        struct,
        symprec=symprec,
        dist_tol_factor=dist_tol_factor,
        species=species,
        return_symprec_and_dist_tol_factor=True,
        fixed_symprec_and_dist_tol_factor=fixed_symprec_and_dist_tol_factor,
        verbose=verbose,
        fold_to_primitive=fold_to_primitive,
    )
    assert isinstance(equiv_sites_output, tuple)  # return_symprec_and_dist_tol_factor = True
    unique_sites, symprec, dist_tol_factor = equiv_sites_output
    struct_with_all_X = _get_struct_with_all_X(struct, unique_sites)
    sga_with_all_X, symprec = get_sga_and_symprec(struct_with_all_X, symprec=symprec)
    return_tuple = (sga_with_all_X.get_symmetry_dataset(), unique_sites)
    return (
        (*return_tuple, symprec, dist_tol_factor) if return_symprec_and_dist_tol_factor else return_tuple
    )


_cache_ready_get_symm_dataset_of_struct_with_all_equiv_sites = lru_cache(maxsize=int(1e3))(
    _raw_get_symm_dataset_of_struct_with_all_equiv_sites
)


def _get_struct_with_all_X(struct, unique_sites):
    """
    Add all sites in unique_sites to a ``copy`` of ``struct``, and return this
    new |Structure|.
    """
    struct_with_all_X = struct.copy()
    struct_with_all_X.sites += unique_sites
    return struct_with_all_X


def get_equiv_frac_coords_in_primitive(
    frac_coords: ArrayLike,
    primitive: Structure,
    supercell: Structure,
    symprec: float = 0.01,
    dist_tol_factor: float = 1.0,
    equiv_coords: bool = True,
    return_symprec_and_dist_tol_factor: bool = False,
    fixed_symprec_and_dist_tol_factor: bool = False,
    verbose: bool = False,
) -> list[np.ndarray] | np.ndarray | tuple[list[np.ndarray] | np.ndarray, float, float] | None:
    """
    Get equivalent fractional coordinates of ``frac_coords`` (in ``supercell``)
    in the given ``primitive`` cell.

    Returns a list of equivalent fractional coords in the primitive cell if
    ``equiv_coords`` is ``True`` (default).

    Note that there may be multiple possible symmetry-equivalent sites, all of
    which are returned if ``equiv_coords`` is ``True``, otherwise the first
    site in the list (sorted using ``_frac_coords_sort_func``) is returned.

    Args:
        frac_coords (ArrayLike):
            Fractional coordinates in the supercell, for which to get
            equivalent coordinates in the primitive cell.
        primitive (|Structure|):
            Primitive cell structure.
        supercell (|Structure|):
            Supercell structure.
        symprec (float):
            Symmetry precision to use for determining symmetry operations.
            Default is 0.01. If ``fixed_symprec_and_dist_tol_factor`` is
            ``False`` (default), this value will be automatically adjusted (up
            to 10x, down to 0.1x) until the identified equivalent sites from
            ``spglib`` have consistent point group symmetries. Setting
            ``verbose`` to ``True`` will print information on the trialled
            ``symprec`` (and ``dist_tol_factor`` values), and setting
            ``return_symprec_and_dist_tol_factor`` to ``True`` will return the
            final ``symprec`` (and ``dist_tol_factor``) used for the equivalent
            site generation.
        dist_tol_factor (float):
            Distance tolerance for clustering generated sites (to ensure they
            are truly distinct), as a multiplicative factor of ``symprec``.
            Default is 1.0 (i.e. ``dist_tol = symprec``, in Å). If
            ``fixed_symprec_and_dist_tol_factor`` is ``False`` (default), this
            value will also be automatically adjusted if necessary (up to 10x,
            down to 0.1x)(after ``symprec`` adjustments) until the identified
            equivalent sites from ``spglib`` have consistent point group
            symmetries. Setting ``verbose`` to ``True`` will print information
            on the trialled ``dist_tol_factor`` (and ``symprec``) values, and
            setting ``return_symprec_and_dist_tol_factor`` to ``True`` will
            return the final ``symprec`` (and ``dist_tol_factor``) used for
            the equivalent site generation.
        equiv_coords (bool):
            If ``True``, returns a list of equivalent fractional coords in the
            primitive cell. If ``False``, returns the first equivalent
            fractional coordinates in the list, sorted using
            ``_frac_coords_sort_func``. Default: ``True``.
        return_symprec_and_dist_tol_factor (bool):
            If ``True``, returns the final symmetry precision and distance
            tolerance factor used for the equivalent site generation (see
            ``symprec`` and ``dist_tol_factor`` argument descriptions). Default
            is ``False``.
        fixed_symprec_and_dist_tol_factor (bool):
            If ``True``, uses the provided ``symprec`` and ``dist_tol_factor``
            values without any automatic adjustments (see ``symprec`` and
            ``dist_tol_factor`` argument descriptions). Default is ``False``.
        verbose (bool):
            If ``True``, prints information on the trialled ``symprec`` and
            ``dist_tol_factor`` values, and the identified equivalent sites.
            Default is ``False``.

    Returns:
        list[np.ndarray] | np.ndarray | tuple[list[np.ndarray] | np.ndarray, float, float]:
            List of equivalent fractional coordinates in the primitive cell, or
            the first equivalent fractional coordinate in the list (sorted
            using ``_frac_coords_sort_func``), depending on the value of
            ``equiv_coords``. If ``return_symprec_and_dist_tol_factor`` is
            ``True``, also returns the final ``symprec`` and
            ``dist_tol_factor`` used for the equivalent site generation.
    """
    from doped.utils.configurations import get_transformation_from_s2_to_s1  # avoid circular import

    # get the affine map ``f -> (f @ M + t) % 1`` taking supercell frac coords to primitive frac coords,
    # using strict ``StructureMatcher`` tolerances so that a successful match certifies a rigid isometry
    # mapping every supercell atom onto a same-element primitive atom within ``symprec * dist_tol_factor``
    # Å (``stol`` is this distance tolerance in SM's normalised units: ``dist * (n/V)^(1/3)``). Any two
    # such maps differ only by a symmetry operation of ``primitive``, so the generated equivalent sites are
    # independent of the choice made here. Cached (in ``get_transformation_from_s2_to_s1``), as the map
    # depends only on the host pair and tolerances:
    stol = symprec * dist_tol_factor * (len(supercell) / supercell.volume) ** (1 / 3)
    transformation = get_transformation_from_s2_to_s1(
        supercell,
        primitive,
        min_stol=stol,
        max_stol=stol,  # min = max -> single strict trial, no upward stol scanning
        ltol=1e-4,
        angle_tol=0.01,
        scale=False,  # don't rescale volumes; hydrostatic strain must fail the match
        attempt_supercell=True,
    )
    if transformation is not None:
        # affine fold map found, so just generate the equivalent supercell sites and fold directly
        # (much faster than the structure-based folding below):
        M, t, _mapping = transformation  # M: integer supercell matrix relating the lattices
        translation = (-t @ M) % 1  # SM convention: prim_supercell_frac + t = supercell_frac
        equiv_sites_output = get_all_equiv_sites(
            frac_coords,
            supercell,
            symprec=symprec,
            dist_tol_factor=dist_tol_factor,
            return_symprec_and_dist_tol_factor=True,
            fixed_symprec_and_dist_tol_factor=fixed_symprec_and_dist_tol_factor,
            fold_to_primitive=False,  # this function performs its own folding, just needs seed sites
            verbose=verbose,
        )
        assert isinstance(equiv_sites_output, tuple)  # return_symprec_and_dist_tol_factor = True
        unique_sites, symprec, dist_tol_factor = equiv_sites_output
        dist_tol = symprec * dist_tol_factor
        folded_frac_coords = (
            np.array([cast("PeriodicSite", site).frac_coords for site in unique_sites]) @ M + translation
        ) % 1
        # collapse primitive-translation-equivalent folded images to unique sites before re-expansion:
        prim_X_frac_coords = cluster_sites_by_dist_tol(folded_frac_coords, primitive.lattice, dist_tol)
        # symmetrize folded coords by averaging over their site-symmetry images (as effectively done by
        # ``spglib`` standardization in the structure-based folding path below), so slightly-off-symmetry
        # input ``frac_coords`` fold to their ideal (symmetrized) primitive cell sites:
        dataset = get_sga(primitive, symprec=symprec).get_symmetry_dataset()
        for i, frac_coords_i in enumerate(prim_X_frac_coords):
            images = np.einsum("nij,j->ni", dataset.rotations, frac_coords_i) + dataset.translations
            diffs = (images - frac_coords_i + 0.5) % 1 - 0.5  # min-image fractional differences
            site_symm_images = np.linalg.norm(diffs @ primitive.lattice.matrix, axis=1) < dist_tol
            shift = diffs[site_symm_images].mean(axis=0)  # shift to mean position of site-symmetry images
            if np.linalg.norm(shift @ primitive.lattice.matrix) > 1e-6:  # keep exact coords (and thus
                prim_X_frac_coords[i] = frac_coords_i + shift  # cache keys) unchanged for clean inputs

    else:  # no strict affine map match (e.g. distorted/noisy cells); fold via X-decorated structures
        trial_symprecs = _TRIAL_SYMPREC_DIST_TOL_FACTORS * symprec
        trial_dist_tol_factors = _TRIAL_SYMPREC_DIST_TOL_FACTORS * dist_tol_factor
        for trial_dist_tol_factor, trial_symprec in product(trial_dist_tol_factors, trial_symprecs):
            # sometimes we can have edge cases where slight numerical differences cause issues with
            # dist_tol/symprec choices, and then primitive cell determination as a result, so scan over
            # some values if necessary. Here we scan over symprec values first (following the approach in
            # ``get_all_equiv_sites``), then dist_tol values -- this approach was found best from testing
            equiv_sites_output = get_all_equiv_sites(
                frac_coords,
                supercell,
                symprec=trial_symprec,
                dist_tol_factor=trial_dist_tol_factor,
                return_symprec_and_dist_tol_factor=True,
                fixed_symprec_and_dist_tol_factor=fixed_symprec_and_dist_tol_factor,
                fold_to_primitive=False,  # this function performs its own folding, just needs seed sites
                verbose=verbose,
            )
            assert isinstance(equiv_sites_output, tuple)  # return_symprec_and_dist_tol_factor = True
            unique_sites, adjusted_trial_symprec, adjusted_trial_dist_tol_factor = equiv_sites_output
            supercell_with_all_X = _get_struct_with_all_X(supercell, unique_sites)
            prim_with_all_X = get_primitive_structure(
                supercell_with_all_X, ignored_species=["X"], symprec=adjusted_trial_symprec
            )

            # NOTE: If "No mapping between the primitive and supercell structures" difficulties ever prove
            # recurrent here, this could be restructured to fold via the orientation-preserving primitive
            # (``_get_orientation_preserving_primitive``, as in ``_raw_get_all_equiv_sites``), leaving only
            # a single primitive <-> reference-primitive match rather than this supercell -> primitive
            # matching:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="No mapping")
                rotated_struct, matrix = _rotate_and_get_supercell_matrix(
                    prim_with_all_X,
                    primitive,
                    ltol=adjusted_trial_symprec,
                    atol=100 * adjusted_trial_symprec,  # default is 1
                )
            if fixed_symprec_and_dist_tol_factor:
                break  # just take first attempt

            if rotated_struct is not None:
                symprec = adjusted_trial_symprec
                dist_tol_factor = adjusted_trial_dist_tol_factor
                if verbose:
                    print(
                        f"Succeeded folding to primitive cell of equivalent supercell sites, with symprec "
                        f"= {symprec}, dist_tol_factor = {dist_tol_factor}."
                    )
                break

            if verbose:
                print(
                    f"Failed folding to primitive cell of equivalent supercell sites, with symprec = "
                    f"{symprec}, dist_tol_factor = {dist_tol_factor}."
                )

        if rotated_struct is None:
            warnings.warn(
                "Could not find a mapping between the primitive and supercell structures! You may need to "
                "tune the symprec/dist_tol parameters for this system."
            )
            return None

        dist_tol = symprec * dist_tol_factor
        primitive_with_all_X = rotated_struct * matrix
        orig_summed_dist = summed_dist(primitive, primitive_with_all_X, ignored_species=["X"])
        if orig_summed_dist != 0:
            # may have different primitive cell definitions, try re-orienting
            orig_min_dist = min_dist(primitive_with_all_X, ignored_species=["X"])
            reoriented_primitive_with_all_X = orient_s2_like_s1(
                primitive,
                primitive_with_all_X,
                primitive_cell=False,
                ignored_species=["X"],
                comparator=ElementComparator(),
            )
            new_min_dist = min_dist(reoriented_primitive_with_all_X, ignored_species=["X"])
            new_summed_dist = summed_dist(
                primitive, reoriented_primitive_with_all_X, ignored_species=["X"]
            )
            if (
                abs(new_summed_dist - orig_summed_dist) > abs(orig_min_dist - new_min_dist)
                and abs(orig_min_dist - new_min_dist) < dist_tol * 2
            ):  # only take re-oriented cell if it improves RMS diff & doesn't much change min_dist
                primitive_with_all_X = reoriented_primitive_with_all_X
                dist_tol = max(dist_tol, abs(orig_min_dist - new_min_dist))
                dist_tol_factor = dist_tol / symprec

        prim_X_frac_coords = [
            site.frac_coords for site in primitive_with_all_X.sites if site.specie.symbol == "X"
        ]

    # now re-apply ``get_all_equiv_sites`` to each folded primitive cell site, to account for possible
    # periodicity-breaking in the supercell, which would then only give a subset of the actual equivalent
    # sites in the primitive cell:
    if verbose:
        print("Regenerating equivalent sites in primitive cell...")
    all_equiv_prim_frac_coords = cluster_sites_by_dist_tol(
        [
            equiv_frac_coords
            for prim_frac_coords in prim_X_frac_coords
            for equiv_frac_coords in get_all_equiv_sites(
                prim_frac_coords,
                primitive,
                just_frac_coords=True,
                symprec=symprec,
                dist_tol_factor=dist_tol_factor,
                fixed_symprec_and_dist_tol_factor=fixed_symprec_and_dist_tol_factor,
                verbose=verbose,
            )
        ],
        primitive.lattice,
        dist_tol=dist_tol,
    )

    prim_coord_list = sorted(
        [
            _vectorized_custom_round(np.mod(_vectorized_custom_round(frac_coords), 1))
            for frac_coords in all_equiv_prim_frac_coords
        ],
        key=_frac_coords_sort_func,
    )

    if return_symprec_and_dist_tol_factor:
        return (prim_coord_list if equiv_coords else prim_coord_list[0]), symprec, dist_tol_factor
    return prim_coord_list if equiv_coords else prim_coord_list[0]


def are_equivalent_lattices(
    lattice_1: Lattice | Structure,
    lattice_2: Lattice | Structure,
    ltol: float = 5e-3,
    atol: float = 1,
) -> bool:
    """
    Check if two lattices are (symmetry-)equivalent, allowing for different
    cell sizes.

    Args:
        lattice_1 (|Lattice| | |Structure|):
            The first lattice to check for equivalence.
        lattice_2 (|Lattice| | |Structure|):
            The second lattice to check for equivalence.
        ltol (float):
            Fractional tolerance for matching lattice vector lengths.
            Defaults to 5e-3 (i.e. 0.5% tolerance).
        atol (float):
            Tolerance for matching angles. Defaults to 1 degree.

    Returns:
        bool:
            ``True`` if the two lattices are (symmetry-)equivalent, ``False``
            otherwise.
    """
    lattice_1 = lattice_1 if isinstance(lattice_1, Lattice) else lattice_1.lattice
    lattice_2 = lattice_2 if isinstance(lattice_2, Lattice) else lattice_2.lattice
    return lattice_1.find_mapping(lattice_2, ltol=ltol, atol=atol, skip_rotation_matrix=True) is not None


def _rotate_and_get_supercell_matrix(
    prim_struct: Structure, target_struct: Structure, ltol: float = 1e-5, atol: float = 1
) -> tuple[Structure, np.ndarray] | tuple[None, None]:
    """
    Rotates the input ``prim_struct`` to match the ``target_struct``
    orientation, and returns the supercell matrix to convert from the rotated
    ``prim_struct`` to the ``target_struct``.

    Returns ``(None, None)`` if no mapping is found.

    Args:
        prim_struct (|Structure|):
            The primitive structure.
        target_struct (|Structure|):
            The target structure to match.
        ltol (float):
            Length tolerance for matching the lattice vectors (default: 1e-5).
        atol (float):
            Angle tolerance for matching the angles between the lattice vectors
            (default: 1).

    Returns:
        tuple[Structure, np.ndarray]:
            The rotated primitive structure and the supercell matrix to convert
            from the rotated primitive structure to the target structure.
    """
    possible_mappings = list(
        prim_struct.lattice.find_all_mappings(target_struct.lattice, ltol=ltol, atol=atol)
    )
    if not possible_mappings:
        warnings.warn("No mapping between the primitive and target structures found!")
        return None, None

    mapping = next(
        iter(  # get possible mappings, then sort by R*S, S, R, then return first
            sorted(
                possible_mappings,
                key=lambda x: (
                    _lattice_matrix_sort_func(np.dot(x[1].T, x[2])),
                    _lattice_matrix_sort_func(x[2]),
                    _lattice_matrix_sort_func(x[1]),
                ),
            )
        )
    )

    rotation_matrix = mapping[1]
    if np.allclose(rotation_matrix, -1 * np.eye(3)):
        # pymatgen sometimes gives a rotation matrix of -1 * identity matrix, which is
        # equivalent to no rotation. Just use the identity matrix instead.
        rotation_matrix = np.eye(3)
        supercell_matrix = -1 * mapping[2]
    else:
        supercell_matrix = mapping[2]

    rotation_symm_op = SymmOp.from_rotation_and_translation(
        rotation_matrix=rotation_matrix.T
    )  # Transpose = inverse of rotation matrices (orthogonal matrices), better numerical stability
    output_prim_struct = apply_symm_op_to_struct(rotation_symm_op, prim_struct, rotate_lattice=True)
    return _round_struct_coords(output_prim_struct), supercell_matrix


def translate_structure(
    structure: Structure, vector: np.ndarray, frac_coords: bool = True, to_unit_cell: bool = True
) -> Structure:
    """
    Translate a structure and its sites by a given vector (**not in place**).

    Args:
        structure: ``pymatgen`` |Structure| object.
        vector: Translation vector, fractional or Cartesian.
        frac_coords:
            Whether the input vector is in fractional coordinates.
            (Default: True)
        to_unit_cell:
            Whether to translate the sites to the unit cell.
            (Default: True)

    Returns:
        ``pymatgen`` |Structure| object with translated sites.
    """
    translated_structure = structure.copy()
    return translated_structure.translate_sites(
        indices=list(range(len(translated_structure))),
        vector=vector,
        to_unit_cell=to_unit_cell,
        frac_coords=frac_coords,
    )


def _get_supercell_matrix_and_possibly_redefine_prim(
    prim_struct, target_struct, sga: SpacegroupAnalyzer | None = None, symprec=0.01
):
    """
    Determines the supercell transformation matrix to convert from the
    primitive structure to the target structure.

    The supercell matrix is defined to be T in ``T*P = S`` where P and S are
    the primitive and supercell lattice matrices respectively. Equivalently,
    multiplying ``prim_struct * T`` will give the target_struct. In
    ``pymatgen``, this requires the output transformation matrix to be integer.

    First tries to determine a simple (integer) transformation matrix with no
    basis set rotation required. If that fails, then defaults to using
    ``_rotate_and_get_supercell_matrix``. Searches over various possible
    primitive cell definitions from spglib.

    Args:
        prim_struct: ``pymatgen`` |Structure| object of the primitive cell.
        target_struct: ``pymatgen`` |Structure| object of the target cell.
        sga:
            ``SpacegroupAnalyzer`` object of the primitive cell. If ``None``,
            will be computed from ``prim_struct``.
        symprec:
            Symmetry precision for ``SpacegroupAnalyzer``, if being generated.

    Returns:
        prim_struct:
            Primitive structure, possibly rotated/redefined.
        supercell_matrix:
            Supercell transformation matrix to convert from the primitive
            structure to the target structure.
    """

    def _get_supercell_matrix_and_possibly_rotate_prim(prim_struct, target_struct):
        try:
            # supercell transform matrix is T in `T*P = S` (P = prim, S = super), so `T = S*P^-1`:
            transformation_matrix = np.dot(
                target_struct.lattice.matrix, np.linalg.inv(prim_struct.lattice.matrix)
            )
            if not np.allclose(np.rint(transformation_matrix), transformation_matrix, atol=1e-3):
                raise ValueError  # if non-integer transformation matrix

            return prim_struct, np.rint(transformation_matrix)

        except ValueError:  # if non-integer transformation matrix
            attempt_prim_struct, attempt_transformation_matrix = _rotate_and_get_supercell_matrix(
                prim_struct,
                target_struct,
                ltol=symprec,
                atol=100 * symprec,
            )
            if attempt_prim_struct:  # otherwise failed, stick with original T matrix
                prim_struct = attempt_prim_struct
                transformation_matrix = attempt_transformation_matrix

        if np.allclose(np.rint(transformation_matrix), transformation_matrix, atol=1e-3):
            return prim_struct, np.rint(transformation_matrix)

        return prim_struct, transformation_matrix

    summed_dists_w_candidate_prim_structs_and_T_matrices = []
    # Could also apply possible origin shifts to other structs (refined, find_primitive) as well,
    # if we find any structures for which this code still fails
    candidate_prim_structs = [
        *_get_candidate_prim_structs(prim_struct, symprec=symprec),
        *_get_candidate_prim_structs(target_struct, symprec=symprec),
    ]

    for possible_prim_struct in candidate_prim_structs:
        new_prim_struct, transformation_matrix = _get_supercell_matrix_and_possibly_rotate_prim(
            possible_prim_struct, target_struct
        )
        if not np.allclose(
            np.rint(transformation_matrix), transformation_matrix, atol=1e-3
        ) or not np.allclose(
            (new_prim_struct * transformation_matrix).lattice.matrix,
            target_struct.lattice.matrix,
            atol=1e-3,
        ):
            # not integer or doesn't exactly match bulk supercell, so bad transformation matrix, skip
            continue
        new_prim_struct = Structure.from_sites([site.to_unit_cell() for site in new_prim_struct])
        summed_dist_to_target = summed_dist(
            Structure.from_sites(
                [
                    site.to_unit_cell()
                    for site in (new_prim_struct * transformation_matrix).get_sorted_structure()
                ]
            ),
            target_struct,
        )
        summed_dists_w_candidate_prim_structs_and_T_matrices.append(
            (summed_dist_to_target, new_prim_struct, transformation_matrix)
        )

    closest_match = sorted(  # sort to get ideal primitive cell definition
        summed_dists_w_candidate_prim_structs_and_T_matrices,
        key=lambda x: (
            round(x[0], 3),
            _lattice_matrix_sort_func(x[1].lattice.matrix),
            _lattice_matrix_sort_func(x[2]),
            _struct_sort_func(x[1]),
        ),
    )[0]
    if closest_match[0] > 0.1:  # no perfect match has been found. Warn user and return the closest:
        warnings.warn(
            f"Found the transformation matrix from the primitive cell lattice to the supplied supercell, "
            f"but could not determine the transformation to directly match the atomic coordinates ("
            f"infinite possible symmetry-equivalent coordinate definitions). Closest match has RMS "
            f"distance of {closest_match[0]:.3f} Å.\n"
            f"The bulk and defect supercells generated will be equivalent to the input supercell, "
            f"but with a different choice of atomic coordinates (e.g. [0.1, 0.1, 0.1] instead of [0.9, "
            f"0.9, 0.9]). You should make sure to do the bulk supercell calculation with this "
            f"doped-generated supercell (DefectsGenerator.bulk_supercell, which is output to the `Bulk` "
            f"folders with the file generation functions), so that the coordinates match those of the "
            f"defect supercells (this matters when computing finite-size corrections)."
        )
    # Note: Could always just get the transformation of the generated supercell to the input supercell, and
    # then apply this transformation to each generated bulk/defect supercell at the end of defect
    # generation, but means that self.primitive_structure * self.supercell_matrix is no longer
    # guaranteed to match self.bulk_supercell... Not the biggest deal though
    # Likely way more work than worth
    return closest_match[1:]


def _get_candidate_prim_structs(structure, **kwargs):
    sga = get_sga(structure, **kwargs)

    pmg_prim_struct = structure.get_primitive_structure(tolerance=kwargs.get("symprec", 0.01))
    candidate_prim_structs = (
        [structure, pmg_prim_struct] if len(structure) == len(pmg_prim_struct) else [pmg_prim_struct]
    )

    prev_struct = None
    for _i in range(4):
        struct = sga.get_primitive_standard_structure()
        if prev_struct is not None and struct == prev_struct:
            break  # standardisation converged; further iterations just duplicate these candidates
        candidate_prim_structs.append(struct)

        spglib_dataset = sga.get_symmetry_dataset()
        if not np.allclose(spglib_dataset.origin_shift, 0):
            candidate_prim_structs.append(translate_structure(struct, spglib_dataset.origin_shift))

        sga = get_sga(struct, sga._symprec)  # use same symprec
        prev_struct = struct

    candidate_prim_structs.append(sga.find_primitive())
    for candidate_conv_struct in [sga.get_refined_structure(), sga.get_conventional_standard_structure()]:
        if len(candidate_conv_struct) == len(pmg_prim_struct):
            # only also try conventional if equivalent to the primitive cell
            candidate_prim_structs = [candidate_conv_struct, *candidate_prim_structs]

    # sometimes Structure.get_primitive_structure() can fail to identify the primitive structure, returning
    # the same input structure, so if the number of atoms differs in different candidate primitive
    # structures, then just take those with the minimum number of atoms:
    return [
        candidate_prim_struct
        for candidate_prim_struct in candidate_prim_structs
        if len(candidate_prim_struct) == min(len(i) for i in candidate_prim_structs)
    ]


def get_wyckoff(
    frac_coords: ArrayLike,
    struct: Structure,
    equiv_sites: bool = False,
    symprec: float = 0.01,
    **kwargs,
) -> str | tuple:
    r"""
    Get the Wyckoff label of the input fractional coordinates in the input
    structure. If the symmetry operations of the structure have already been
    computed, these can be input as a list to speed up the calculation.

    Args:
        frac_coords (ArrayLike):
            Fractional coordinates of the site to get the Wyckoff label of.
        struct (|Structure|):
            |Structure| for which ``frac_coords`` corresponds to.
        equiv_sites (bool):
            If ``True``, returns a tuple of (Wyckoff label, list of equivalent
            sites). Default is ``False``.
        symprec (float):
            Symmetry precision to use for determining symmetry operations.
            Default is 0.01. If ``fixed_symprec_and_dist_tol_factor`` is
            ``False`` (default), this value will be automatically adjusted (up
            to 10x, down to 0.1x) until the identified equivalent sites from
            ``spglib`` have consistent point group symmetries. Setting
            ``verbose`` to ``True`` will print information on the trialled
            ``symprec`` (and ``dist_tol_factor`` values).
        **kwargs:
            Additional keyword arguments to pass to ``get_all_equiv_sites``,
            such as ``dist_tol_factor``, ``fixed_symprec_and_dist_tol_factor``,
            and ``verbose``.

    Returns:
        str | tuple:
            The Wyckoff label of the input fractional coordinates in the
            structure. If ``equiv_sites`` is ``True``, also returns a list of
            equivalent sites in the structure.
    """
    symm_dataset, unique_sites = _get_symm_dataset_of_struct_with_all_equiv_sites(
        frac_coords,
        struct,
        symprec=symprec,
        return_symprec_and_dist_tol_factor=False,
        **kwargs,
    )
    conv_cell_factor = len(symm_dataset.std_positions) / len(symm_dataset.wyckoffs)
    multiplicity = int(conv_cell_factor * len(unique_sites))
    wyckoff_label = f"{multiplicity}{symm_dataset.wyckoffs[-1]}"

    return (wyckoff_label, unique_sites) if equiv_sites else wyckoff_label


def _struct_sort_func(struct: Structure | np.ndarray) -> tuple:
    """
    Sort by the lattice matrix sorting function, then by (minus) the number of
    high-symmetry coordinates (x=y=z, then 2 equal coordinates), then by the
    sum of all fractional coordinates, then by the magnitudes of high-symmetry
    coordinates (x=y=z, then 2 equal coordinates), then by the summed magnitude
    of all x coordinates, then y coordinates, then z coordinates.

    Args:
        struct:
            ``pymatgen`` |Structure| object, or an array of fractional
            coordinates of sites in the structure (in which case the lattice
            matrix metric is skipped).

    Returns:
        tuple: Tuple of sorting criteria values.
    """
    if isinstance(struct, Structure):
        struct_for_sorting = _round_struct_coords(struct, to_unit_cell=True)
        lattice_metric = _lattice_matrix_sort_func(struct_for_sorting.lattice.matrix)
        frac_coords = struct_for_sorting.frac_coords
    else:
        lattice_metric = (False,)
        frac_coords = struct

    # get summed magnitudes of x=y=z coords:
    xyz_matching_coords = frac_coords[  # Find the coordinates where x = y = z:
        (frac_coords[:, 0] == frac_coords[:, 1]) & (frac_coords[:, 1] == frac_coords[:, 2])
    ]
    xyz_sum_magnitudes = np.sum(np.linalg.norm(xyz_matching_coords, axis=1))

    # get summed magnitudes of x=y / y=z / x=z coords:
    xy_matching_coords = frac_coords[
        (frac_coords[:, 0] == frac_coords[:, 1])
        | (frac_coords[:, 1] == frac_coords[:, 2])
        | (frac_coords[:, 0] == frac_coords[:, 2])
    ]
    xy_sum_magnitudes = np.sum(np.linalg.norm(xy_matching_coords, axis=1))

    return (
        *lattice_metric,
        -len(xyz_matching_coords),
        -len(xy_matching_coords),
        round(np.sum(frac_coords), 2),
        round(xyz_sum_magnitudes, 2),
        round(xy_sum_magnitudes, 2),
        round(np.sum(frac_coords[:, 0]), 2),
        round(np.sum(frac_coords[:, 1]), 2),
        round(np.sum(frac_coords[:, 2]), 2),
    )


def _lattice_matrix_sort_func(lattice_matrix: np.ndarray) -> tuple:
    """
    Sorting function to apply on an iterable of lattice matrices.

    Matrices are sorted by:

    - lattice_matrix is diagonal
    - matrix symmetry (around diagonal)
    - maximum sum of diagonal element magnitudes.
    - minimum number of negative elements
    - maximum number of x, y, z that are equal
    - maximum number of abs(x), abs(y), abs(z) that are equal
    - a, b, c magnitudes (favouring c >= b >= a)

    Args:
        lattice_matrix (np.ndarray): Lattice matrix to sort.

    Returns:
        tuple: Tuple of sorting criteria values.
    """

    def is_symmetric(matrix: np.ndarray, tol: float = 1e-3) -> bool:
        iu = np.triu_indices_from(matrix, k=1)  # indices of upper triangle of matrix
        return bool(np.all(np.abs(matrix[iu] - matrix.T[iu]) <= tol))

    is_diagonal = np.all(np.abs(lattice_matrix[~np.eye(3, dtype=bool)]) < 1e-3)
    symmetric = is_diagonal or is_symmetric(lattice_matrix)
    num_negs = np.sum(lattice_matrix < 0)
    diag_sum = np.round(np.sum(np.abs(np.diag(lattice_matrix))), 1)
    flat_matrix = lattice_matrix.ravel()
    _unique_vals, counts = np.unique(flat_matrix, return_counts=True)
    num_equals = np.sum(counts * (counts + 1) // 2)
    _abs_vals, abs_counts = np.unique(np.abs(flat_matrix), return_counts=True)
    num_abs_equals = np.sum(abs_counts * (abs_counts + 1) // 2)
    a, b, c = np.linalg.norm(lattice_matrix, axis=1)

    return (
        not is_diagonal,
        not symmetric,
        -diag_sum,
        num_negs,
        -num_equals,
        -num_abs_equals,
        -round(c, 2),
        -round(b, 2),
        -round(a, 2),
    )


def get_clean_structure(
    structure: Structure, return_T: bool = False, dist_precision: float = 0.001, niggli_reduce: bool = True
) -> Structure | tuple[Structure, np.ndarray]:
    """
    Get a 'clean' version of the input `structure` by searching over equivalent
    cells, and finding the most optimal according to
    ``_lattice_matrix_sort_func`` (most symmetric, with mostly positive
    diagonals and c >= b >= a).

    Args:
        structure (|Structure|): |Structure| object.
        return_T (bool):
            Whether to return the transformation matrix from the original
            structure lattice to the new structure lattice (T * Orig = New).
            (Default = False)
        dist_precision (float):
            The desired distance precision in Å for rounding of lattice
            parameters and fractional coordinates. (Default: 0.001)
        niggli_reduce (bool):
            Whether to Niggli reduce the lattice before searching for the
            optimal lattice matrix. If this is set to ``False``, we also skip
            the search for the best positive determinant lattice matrix.
            (Default: True)

    Returns:
        Structure | tuple[Structure, np.ndarray]:
            The 'clean' version of the input structure, or a tuple of the
            'clean' structure and the transformation matrix from the original
            structure lattice to the new structure lattice (T * Orig = New).
    """
    lattice = structure.lattice
    if np.all(lattice.matrix <= 0):
        lattice = Lattice(lattice.matrix * -1)
    possible_lattice_matrices = [
        lattice.matrix,
    ]

    for _ in range(4):
        lattice = lattice.get_niggli_reduced_lattice() if niggli_reduce else lattice

        # want to maximise the number of non-negative diagonals, and also have a positive determinant
        # can multiply two rows by -1 to get a positive determinant:
        possible_lattice_matrices.append(lattice.matrix)
        for i in range(3):
            for j in range(i + 1, 3):
                new_lattice_matrix = lattice.matrix.copy()
                new_lattice_matrix[i] = new_lattice_matrix[i] * -1
                new_lattice_matrix[j] = new_lattice_matrix[j] * -1
                possible_lattice_matrices.append(new_lattice_matrix)

    possible_lattice_matrices.sort(key=_lattice_matrix_sort_func)
    new_lattice_matrix = possible_lattice_matrices[0]
    if np.all(new_lattice_matrix <= 0):
        new_lattice_matrix = new_lattice_matrix * -1

    new_structure = Structure(
        new_lattice_matrix,
        structure.species_and_occu,
        structure.cart_coords,
        coords_are_cartesian=True,
        to_unit_cell=True,
        site_properties=structure.site_properties,
        labels=structure.labels,
        charge=structure._charge,
    )
    new_structure = _round_struct_coords(new_structure, dist_precision=dist_precision, to_unit_cell=True)

    # sort structure to match a desired, deterministic format:
    new_structure = new_structure.get_sorted_structure(
        key=lambda x: (
            x.species.average_electroneg,
            x.species_string,
            _frac_coords_sort_func(x.frac_coords),
        )
    )
    if niggli_reduce:
        new_structure = _get_best_pos_det_structure(new_structure)  # ensure positive determinant

    if return_T:
        # T * Orig = New; T = New * Orig^-1; Orig = T^-1 * New
        transformation_matrix = np.matmul(
            new_structure.lattice.matrix, np.linalg.inv(structure.lattice.matrix)
        )
        if not np.allclose(transformation_matrix, np.rint(transformation_matrix), atol=1e-5):
            raise ValueError(
                "Transformation matrix for clean/reduced structure could not be found! If you are seeing "
                "this bug, please notify the `doped` developers"
            )

        return (new_structure, np.rint(transformation_matrix))

    return new_structure


def _get_best_pos_det_structure(structure: Structure):
    """
    If the input structure has a negative determinant (corresponding to a left-
    hand coordinate system), then find the best possible re-definition of the
    lattice vectors which gives a positive determinant, according to
    ``_struct_sort_func``.

    This is to avoid an apparent VASP bug with negative triple products of the
    lattice vectors -- not sure if this is only in old versions?
    """
    if np.linalg.det(structure.lattice.matrix) < 0:
        swap_combo_score_dict = {}
        for swap_combo in permutations([0, 1, 2]):
            candidate_structure = swap_axes(structure, swap_combo)
            if np.linalg.det(candidate_structure.lattice.matrix) > 0:
                swap_combo_score_dict[swap_combo] = _struct_sort_func(candidate_structure)

        best_swap_combo = min(swap_combo_score_dict, key=lambda x: swap_combo_score_dict[x])
        structure = swap_axes(structure, best_swap_combo)

    return structure


def get_primitive_structure(
    structure: Structure,
    ignored_species: list | None = None,
    clean: bool = True,
    return_all: bool = False,
    **kwargs,
):
    """
    Get a consistent/deterministic primitive structure from a ``pymatgen``
    |Structure|.

    For some materials (e.g. zinc blende), there are multiple equivalent
    primitive cells (e.g. Cd (0,0,0) & Te (0.25,0.25,0.25); Cd (0,0,0) & Te
    (0.75,0.75,0.75) for F-43m CdTe), so for reproducibility and in line with
    most structure conventions/definitions, take the one with the cleanest
    lattice and structure definition, according to ``_struct_sort_func``.

    If ``ignored_species`` is set, then the sorting function used to determine
    the ideal primitive structure will ignore sites with species in
    ``ignored_species``.

    Args:
        structure (|Structure|):
            |Structure| to get the corresponding primitive structure of.
        ignored_species (list | None):
            List of species to ignore when determining the ideal primitive
            structure. (Default: None)
        clean (bool):
            Whether to return a 'clean' version of the primitive structure,
            with the lattice matrix in a standardised form. (Default: True)
        return_all (bool):
            Whether to return all possible primitive structures tested, sorted
            by the sorting function. (Default: False)
        **kwargs:
            Additional keyword arguments to pass to the ``get_sga`` function
            (e.g. ``symprec`` etc).

    Returns:
        Structure | list[Structure]:
            The primitive structure of the input structure, or a list of all
            possible primitive structures tested, sorted by the sorting
            function.
    """
    # make inputs hashable, then call ``_cache_ready_get_primitive_structure``:
    cache_ready_ignored_species = tuple(ignored_species) if ignored_species is not None else None
    cache_ready_kwargs = tuple(kwargs.items()) if kwargs else None

    output = _cache_ready_get_primitive_structure(
        structure,
        ignored_species=cache_ready_ignored_species,
        clean=clean,
        return_all=return_all,
        kwargs=cache_ready_kwargs,
    )
    # copy on every call (incl. cache hits) so caller mutation can't corrupt the cached structure(s):
    return [struct.copy() for struct in output] if return_all else output.copy()


@lru_cache(maxsize=int(1e3))
def _cache_ready_get_primitive_structure(
    structure: Structure,
    ignored_species: tuple | None = None,
    clean: bool = True,
    return_all: bool = False,
    kwargs: tuple | None = None,
):
    """
    ``get_primitive_structure`` code, with hashable input arguments for caching
    (using |Structure| hash function from ``doped.utils.efficiency``).
    """
    # clean structure site_properties (if mismatching ``None`` values present, can mess with primitive
    # structure determination) -- this can happen if e.g. a slab structure is input with "bulk_wyckoff"
    # etc site properties. Done on a copy, so that neither the caller's structure nor this function's
    # (already-captured) ``lru_cache`` key is mutated:
    mismatching_props = [
        key
        for key, val in structure.site_properties.items()
        if any(i is not None for i in val) and any(i is None for i in val)
    ]
    if mismatching_props:
        structure = structure.copy()
        for site in structure:
            for key in mismatching_props:
                site.properties.pop(key, None)

    kwargs_dict = dict(kwargs) if kwargs is not None else {}
    candidate_prim_structs = _get_candidate_prim_structs(structure, **kwargs_dict)

    if ignored_species is not None:
        pruned_possible_prim_structs = [
            Structure.from_sites([site for site in struct if site.specie.symbol not in ignored_species])
            for struct in candidate_prim_structs
        ]
    else:
        pruned_possible_prim_structs = candidate_prim_structs

    # sort and return indices:
    sorted_indices = sorted(
        range(len(pruned_possible_prim_structs)),
        key=lambda i: _struct_sort_func(pruned_possible_prim_structs[i]),
    )

    prim_structs = [
        _get_best_pos_det_structure(_round_struct_coords(candidate_prim_structs[i], to_unit_cell=True))
        for i in sorted_indices
    ]
    if clean:
        prim_structs = [get_clean_structure(struct) for struct in prim_structs]

    return prim_structs if return_all else _get_best_pos_det_structure(prim_structs[0])


def get_spglib_conv_structure(sga: SpacegroupAnalyzer) -> tuple[Structure, SpacegroupAnalyzer]:
    """
    Get a consistent/deterministic conventional structure from a
    ``SpacegroupAnalyzer`` object. Also returns the corresponding
    ``SpacegroupAnalyzer`` (for getting Wyckoff symbols corresponding to this
    conventional structure definition).

    For some materials (e.g. zinc blende), there are multiple equivalent
    primitive/conventional cells, so for reproducibility and in line with most
    structure conventions/definitions, take the one with the lowest summed norm
    of the fractional coordinates of the sites (i.e. favour Cd (0,0,0) and Te
    (0.25,0.25,0.25) over Cd (0,0,0) and Te (0.75,0.75,0.75) for F-43m CdTe;
    SGN 216).
    """
    possible_conv_structs_and_sgas = []
    for _i in range(3):
        struct = sga.get_conventional_standard_structure()
        possible_conv_structs_and_sgas.append((struct, sga))
        sga = get_sga(sga.get_primitive_standard_structure(), symprec=sga._symprec)

    possible_conv_structs_and_sgas = sorted(
        possible_conv_structs_and_sgas, key=lambda x: _struct_sort_func(x[0])
    )
    return (
        _round_struct_coords(possible_conv_structs_and_sgas[0][0], to_unit_cell=True),
        possible_conv_structs_and_sgas[0][1],
    )


def get_BCS_conventional_structure(
    structure: Structure, pbar: tqdm | None = None, return_wyckoff_dict: bool = False
) -> tuple[Structure, list[int]] | tuple[Structure, list[int], dict[str, list[list[Expr]]]]:
    """
    Get the conventional crystal structure of the input structure, according to
    the Bilbao Crystallographic Server (BCS) definition.

    Also returns an array of the lattice vector swaps (used with ``swap_axes``)
    to convert from the ``spglib`` (``SpaceGroupAnalyzer``) conventional
    structure definition to the BCS definition.

    Args:
        structure (|Structure|):
            |Structure| for which to get the corresponding BCS conventional
            crystal structure.
        pbar (ProgressBar):
            ``tqdm`` progress bar object, to update progress. Default is
            ``None``.
        return_wyckoff_dict (bool):
            Whether to return the Wyckoff label dict (as
            ``{Wyckoff label: coordinates}``).

    Returns:
        tuple[Structure, np.ndarray] | tuple[Structure, np.ndarray, dict[str, np.ndarray]]:
            A tuple of the BCS conventional structure of the input structure,
            the lattice vector swapping array and, if ``return_wyckoff_dict``
            is ``True``, the Wyckoff label dict.
    """
    struc_wout_oxi = structure.copy()
    struc_wout_oxi.remove_oxidation_states()
    sga = get_sga(struc_wout_oxi)
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

    bcs_conv_structure = get_clean_structure(
        swap_axes(conventional_structure, lattice_vec_swap_array), niggli_reduce=False
    )
    assert isinstance(bcs_conv_structure, Structure)  # return_T = False

    if return_wyckoff_dict:
        return bcs_conv_structure, lattice_vec_swap_array, wyckoff_label_dict

    return bcs_conv_structure, lattice_vec_swap_array


def get_conv_cell_site(defect_entry: DefectEntry) -> PeriodicSite | None:
    """
    Gets an equivalent site of the defect entry in the conventional structure
    of the host material. If the ``conventional_structure`` attribute is not
    defined for defect_entry, then it is generated using ``SpacegroupAnalyzer``
    and then reoriented to match the Bilbao Crystallographic Server's
    conventional structure definition.

    Args:
        defect_entry: |DefectEntry| object.

    Returns:
        PeriodicSite | None:
            The equivalent site of the defect entry in the conventional
            structure of the host material, or ``None`` if not found.
    """
    bulk_prim_structure = defect_entry.defect.structure.copy()
    bulk_prim_structure.remove_oxidation_states()  # adding oxidation states adds the
    # # deprecated 'properties' attribute with -> {"spin": None}, giving a deprecation warning

    prim_struct_with_X = bulk_prim_structure.copy()
    prim_struct_with_X.append("X", defect_entry.defect.site.frac_coords, coords_are_cartesian=False)

    sga = get_sga(bulk_prim_structure)
    # convert to match sga primitive structure first:
    sga_prim_struct = sga.get_primitive_standard_structure()
    prim_struct_with_X_like_sga_prim = orient_s2_like_s1(
        sga_prim_struct,
        prim_struct_with_X,
        primitive_cell=False,
        ignored_species=["X"],
        comparator=ElementComparator(),
    )
    if not prim_struct_with_X_like_sga_prim:
        warnings.warn(
            "The transformation from the DefectEntry primitive cell to the spglib primitive cell could "
            "not be determined, and so the corresponding conventional cell site cannot be identified."
        )
        return None

    conv_struct_with_X = prim_struct_with_X_like_sga_prim * np.linalg.inv(
        sga.get_conventional_to_primitive_transformation_matrix()
    )

    # convert to match defect_entry conventional structure definition
    assert defect_entry.conventional_structure is not None
    conv_struct_with_X_like_defect_entry_conv = orient_s2_like_s1(
        defect_entry.conventional_structure,
        conv_struct_with_X,
        primitive_cell=False,
        ignored_species=["X"],
        comparator=ElementComparator(),
    )

    conv_cell_site = next(
        site for site in conv_struct_with_X_like_defect_entry_conv.sites if site.specie.symbol == "X"
    )
    # site choice doesn't matter so much here, as we later get the equivalent coordinates using the
    # Wyckoff dict and choose the conventional site based on that anyway (in the ``DefectsGenerator``
    # initialisation)
    conv_cell_site.to_unit_cell()
    conv_cell_site.frac_coords = _vectorized_custom_round(conv_cell_site.frac_coords)

    return conv_cell_site


def swap_axes(structure: Structure, axes: list[int] | tuple[int, ...]) -> Structure:
    """
    Swap axes of the given structure.

    The new order of the axes is given by the axes parameter. For example,
    ``axes=(2, 1, 0)`` will swap the first and third axes.
    """
    transformation_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for i, axis in enumerate(axes):
        transformation_matrix[i][axis] = 1

    transformation = SupercellTransformation(transformation_matrix)

    return transformation.apply_transformation(structure)


def get_wyckoff_dict_from_sgn(sgn: int) -> dict[str, list[list[Expr]]]:
    """
    Get dictionary of ``{Wyckoff label: coordinates}`` for a given space group
    number.

    The database used here for Wyckoff analysis (``wyckpos.dat``) was obtained
    from code written by JaeHwan Shim @schinavro (ORCID: 0000-0001-7575-4788)
    (https://gitlab.com/ase/ase/-/merge_requests/1035) based on the tabulated
    datasets in https://github.com/xtalopt/randSpg (also found at
    https://github.com/spglib/spglib/blob/develop/database/Wyckoff.csv).
    By default, doped uses the Wyckoff functionality of ``spglib`` (along with
    symmetry operations in pymatgen) when possible, however.

    Args:
        sgn (int):
            Space group number.

    Returns:
        dict[str, list[list[float]]]:
            Dictionary of Wyckoff labels and their corresponding coordinates.
    """
    datafile = _get_wyckoff_datafile()
    with open(datafile, encoding="utf-8") as f:
        wyckoff = _read_wyckoff_datafile(sgn, f)

    wyckoff_label_coords_dict = {}

    def _coord_string_to_array(coord_string):
        # Split string into substrings, parse each as a sympy expression,
        # then convert to list of sympy expressions
        return np.array([cached_simplify(x.replace("2x", "2*x")) for x in coord_string.split(",")])

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
    defect_entry: DefectEntry | None = None,
    conv_cell_site: PeriodicSite | None = None,
    sgn: int | None = None,
    wyckoff_dict: dict | None = None,
) -> tuple[str, list[list[float]]]:
    """
    Return the Wyckoff label and list of equivalent fractional coordinates
    within the conventional cell for the input defect_entry or conv_cell_site
    (whichever is provided, defaults to defect_entry if both), given a
    dictionary of Wyckoff labels and coordinates (``wyckoff_dict``).

    If ``wyckoff_dict`` is not provided, it is generated from the spacegroup
    number (sgn) using ``get_wyckoff_dict_from_sgn(sgn)``. If ``sgn`` is not
    provided, it is obtained from the bulk structure of the ``defect_entry`` if
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
            sgn = get_sga(defect_entry.defect.structure).get_space_group_number()

        wyckoff_dict = get_wyckoff_dict_from_sgn(sgn)

    def _compare_arrays(coord_list, coord_array):
        """
        Compare a list of arrays of sympy expressions (``coord_list``) with an
        array of coordinates (``coord_array``).

        Returns the matching array from the list.
        """
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
                            np.mod(float(cached_simplify(sympy_expr).subs(variable_dict)), 1)
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
        variable = next(iter(sympy_expr.free_symbols))
        variable_dict[variable] = cached_solve(equation, variable)[0]

        return cached_simplify(sympy_expr).subs(variable_dict)

    def add_new_variable_dict(
        sympy_expr_prepend, sympy_expr, coord, current_variable_dict, variable_dicts
    ):
        new_sympy_expr = cached_simplify(sympy_expr_prepend + str(sympy_expr))
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
                *sorted(zip(coord_array, sympy_array, strict=False), key=lambda x: len(x[1].free_symbols)),
                strict=False,
            )

            for coord, sympy_expr in zip(coord_array, sympy_array, strict=False):
                # Evaluate the expression with the current variable_dict
                expr_value = cached_simplify(sympy_expr).subs(temp_dict)

                # If the expression cannot be evaluated to a float
                # it means that there is a new variable in the expression
                try:
                    expr_value = np.mod(float(expr_value), 1)  # wrap to 0-1 (i.e. to unit cell)

                except TypeError:
                    # Assign the expression the value of the corresponding coordinate, and solve for the
                    # new variable first, special cases with two possible solutions due to PBC:
                    if sympy_expr == cached_simplify("-2*x"):
                        add_new_variable_dict("1+", sympy_expr, coord, temp_dict, variable_dicts)
                    elif sympy_expr == cached_simplify("2*x"):
                        add_new_variable_dict("-1+", sympy_expr, coord, temp_dict, variable_dicts)

                    expr_value = evaluate_expression(
                        sympy_expr, coord, temp_dict
                    )  # solve for new variable

                # Check if the evaluated expression matches the corresponding coordinate, under periodic
                # wrapping (i.e. modulo 1, to the unit cell); scalar comparisons matching
                # ``np.isclose(..., atol=3e-3)``, but avoiding per-call overhead:
                diff = float(coord) - float(expr_value)
                if abs(diff - round(diff)) > 3e-3:
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
    labels, allowing for either conventional cell definition (``spglib`` /
    Bilbao) -- and thus Wyckoff multiplicities -- being an integer multiple
    (up to 4x) of the other.
    """

    def _multiply_wyckoff(wyckoff, n):
        return f"{n * int(wyckoff[:-1])}{wyckoff[-1]}"

    symbol_set = set(wyckoff_symbols)
    multiplied_symbols = [{_multiply_wyckoff(w, n) for w in wyckoff_symbols} for n in range(1, 5)]  # <=4x
    doped_wyckoffs = []

    for site in conv_struct:
        wyckoff_label, _equiv_coords = get_wyckoff_label_and_equiv_coord_list(
            conv_cell_site=site, wyckoff_dict=wyckoff_dict
        )
        if not any(wyckoff_label in symbols for symbols in multiplied_symbols) and not any(
            _multiply_wyckoff(wyckoff_label, n) in symbol_set for n in range(1, 5)
        ):
            return False  # break on first non-match
        doped_wyckoffs.append(wyckoff_label)

    return any(symbols == set(doped_wyckoffs) for symbols in multiplied_symbols) or any(
        {_multiply_wyckoff(w, n) for w in doped_wyckoffs} == symbol_set for n in range(1, 5)
    )  # False if no complete match, True otherwise


def _read_wyckoff_datafile(spacegroup, f, setting=None):
    """
    Read the ``wyckpos.dat`` file of specific spacegroup and returns a
    dictionary with this information.
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
    Read lines from ``f`` until a blank line is encountered.
    """
    name = str(spacegroup) if setting is None else f"{spacegroup!s}-{setting}"
    while True:
        line = f.readline()
        if not line:
            raise ValueError(
                f"Invalid spacegroup {spacegroup}, setting: {setting}. Not found in the Wyckoff database!"
            )
        if line.startswith(name):
            break
    return line


def point_symmetry_from_defect(
    defect: Defect,
    symprec: float = 0.01,
    **kwargs,
) -> str:
    """
    Get the defect site point symmetry from a |Defect| object.

    Note that this is intended only to be used for unrelaxed, as-generated
    |Defect| objects (rather than parsed defects).

    Args:
        defect (|Defect|): |Defect| object.
        symprec (float):
            Symmetry precision to use for determining symmetry operations and
            thus point symmetries. Default is 0.01. If
            ``fixed_symprec_and_dist_tol_factor`` is ``False`` (default), this
            value will be automatically adjusted (up to 10x, down to 0.1x)
            until the identified equivalent sites from ``spglib`` have
            consistent point group symmetries. Setting ``verbose`` to ``True``
            will print information on the trialled ``symprec`` (and
            ``dist_tol_factor`` values).
        **kwargs:
            Additional keyword arguments to pass to ``get_all_equiv_sites``,
            such as ``dist_tol_factor``, ``fixed_symprec_and_dist_tol_factor``,
            and ``verbose``.

    Returns:
        str: Defect point symmetry.
    """
    try:
        return point_symmetry_from_site(
            defect.site,
            defect.structure,
            symprec=symprec,
            **kwargs,
        )
    except (ValueError, KeyError):
        # symm_ops approach failed (e.g. spglib symmetry determination failure (``ValueError``), or
        # unrecognised Hermann-Mauguin symbol (``KeyError``)); use diagonal defect supercell approach:
        warnings.warn(
            "Defect point symmetry could not be determined from the standard approach. Falling back "
            "to supercell generation approach (which can be less efficient)."
        )
        defect_diagonal_supercell = defect.get_supercell_structure(
            sc_mat=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
            dummy_species="X",
        )  # create defect supercell, which is a diagonal expansion of the unit cell so that the defect
        # periodic image retains the unit cell symmetry, in order not to affect the point group symmetry
        sga = get_sga(defect_diagonal_supercell, symprec=symprec)
        return schoenflies_from_hermann(sga.get_point_group_symbol())


def _extract_defect_cluster(
    structure: Structure, centre_cart: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the atoms within ``radius`` of ``centre_cart`` in ``structure``
    (PBC-aware), returning their Cartesian coordinates `relative` to
    ``centre_cart``, and their element symbols.

    Args:
        structure (|Structure|):
            The structure to extract the local atomic cluster from.
        centre_cart (np.ndarray):
            Cartesian coordinates of the extraction sphere centre.
        radius (float):
            Radius (in Å) of the extraction sphere.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            ``(N, 3)`` array of atomic Cartesian coordinates relative to
            ``centre_cart``, and ``(N,)`` array of their element symbols.
    """
    # Note: We could instead work with ``pymatgen`` ``Site``/``Molecule`` objects here and in the local
    # symmetry functions, to simplify some of the API, but this would incur significant overhead from many
    # property accesses / array unpacking (e.g. Site.specie -> Composition init), so avoided for now.
    sites = structure.get_sites_in_sphere(centre_cart, radius)
    coords = np.array([site.coords for site in sites]).reshape(-1, 3) - centre_cart
    species = np.array([site.specie.symbol for site in sites])
    return coords, species


def _matching_rot_index(
    rotations: Sequence[np.ndarray], rotation: np.ndarray, rot_tol: float = 0.3
) -> int | None:
    """
    Index of the first rotation matrix in ``rotations`` matching ``rotation``
    within ``rot_tol`` (Frobenius norm), else ``None``.

    Distinct crystallographic point operations differ by >= ~1.41 in Frobenius
    norm (rotation angles differ by >= 60°), so tolerance-based matching is
    robust for both exact and noisy (refined) operations.
    """
    if not len(rotations):
        return None
    diffs = np.asarray(rotations) - rotation  # vectorised over all rotations at once
    sq_dists = np.einsum("ijk,ijk->i", diffs, diffs)  # squared Frobenius norms
    matches = np.flatnonzero(sq_dists < rot_tol**2)
    return int(matches[0]) if len(matches) else None


def _bulk_cartesian_rotations(bulk_structure: Structure, symprec: float = 0.01) -> list[np.ndarray]:
    """
    Returns the unique rotation operations of the bulk crystal in the Cartesian
    frame, determined from its primitive cell with ``spglib`` (which preserves
    the Cartesian orientation).
    """
    cell = (
        bulk_structure.lattice.matrix,
        bulk_structure.frac_coords,
        [site.specie.Z for site in bulk_structure],
    )
    # get primitive cell, but with no reorientation of the Cartesian frame (``no_idealize=True`` is crucial
    # here (keeps original Cartesian orientation), and so we can't use ``get_sga`` directly:
    primitive_cell = spglib.standardize_cell(cell, to_primitive=True, no_idealize=True, symprec=symprec)
    if primitive_cell is not None:  # otherwise spglib failure; use the input cell directly
        bulk_structure = Structure(
            Lattice(primitive_cell[0]),
            species=list(primitive_cell[2]),
            coords=primitive_cell[1],
        )

    cart_rotations: list[np.ndarray] = [np.eye(3)]
    try:  # get Cartesian rotations (symmetry operations) from the bulk structure:
        ops = get_sga(bulk_structure, symprec=symprec).get_symmetry_operations(cartesian=True)
    except SymmetryUndeterminedError:  # spglib failure; no candidate cart_rotations beyond identity
        return cart_rotations

    for op in ops:
        if _matching_rot_index(cart_rotations, op.rotation_matrix) is None:
            cart_rotations.append(op.rotation_matrix)
    return cart_rotations


def _candidate_rotations_from_cluster(
    coords: np.ndarray,
    species: np.ndarray,
    dists: np.ndarray | None = None,
    symprec: float = 0.1,
    t_max: float = 3.0,
    min_leg: float = 1.5,
    min_sin: float = 0.4,
) -> list[np.ndarray]:
    """
    Candidate (im)proper symmetry-preserving rotations generated directly from
    the local geometry of an input atomic cluster.

    Any true local isometry must map a chosen "anchor triple" of atoms onto
    some same-species, distance-preserving image triple. Enumerating those
    correspondences therefore yields the maximal candidate set -- inferred
    from the cluster geometry itself (in its given Cartesian frame, relative
    to the centre), rather than from a fixed menu of assumed crystallographic
    axes / orientations.

    Algorithm:

    1. Choose the anchor triple: the atom nearest the centre, plus the two
       next-nearest atoms that form a well-conditioned (non-degenerate)
       triangle with it.
    2. Build a right-handed orthonormal frame ``F_a`` from the anchor triple.
    3. For each anchor atom, collect same-species image candidates whose
       distance from the centre is within ``t_max + symprec`` of the anchor's
       (rotations preserve ``|x|``, so partners cannot lie much farther out
       once a translation of size ``<= t_max`` is allowed).
    4. Enumerate image triples ``(j0, j1, j2)``, pruning any whose pairwise
       separations do not match the anchor's (within ``2 * symprec``).
    5. For each surviving triple, build another right-handed orthonormal frame
       ``F_b`` and take the proper rotation ``R = F_b @ F_a^{-1}``, plus its
       improper (reflected) counterpart; deduplicate within a loose rotation
       tolerance.

    False candidates are rejected downstream by the residual test in
    ``local_point_symmetry``, and noisy true candidates are refined by
    orthogonal Procrustes fitting (``_refine_symm_op``).

    Args:
        coords (np.ndarray):
            ``(N, 3)`` Cartesian coordinates of the local atomic cluster,
            relative to the (rough) defect centre.
        species (np.ndarray):
            ``(N,)`` element symbols of the local atomic cluster.
        dists (np.ndarray | None):
            ``(N,)`` distances of each atom from the (rough) defect centre.
            If ``None`` or empty (default), recomputed from ``coords``.
        symprec (float):
            Distance tolerance (in Å), as in ``local_point_symmetry``.
            Default is 0.1 Å.
        t_max (float):
            Maximum allowed translation magnitude (in Å) for a local isometry
            ``x -> R @ x + t``. Needed because ``coords`` are relative to a
            *rough* defect centre that may be offset from the true point-group
            centre; that offset appears as nonzero ``t``, so image partners can
            sit up to ``~t_max`` (-> ``2 * error in centre position``) farther
            from the origin than their anchors. Used to prune same-species
            image candidates; ``dists[j] <= dists[anchor] + t_max + symprec``.
            If the origin were exact, ``t_max`` could be ~0 (noise covered by
            ``symprec``). Default is 3.0 Å (matching ``t_max`` from
            ``local_point_symmetry`` with ``centre_error_range`` of 1.5 Å).
        min_leg (float):
            Minimum anchor-triangle leg length (in Å), ensuring a
            well-conditioned (non-degenerate) anchor triple. Default is 1.5 Å.
        min_sin (float):
            Minimum sine of the anchor-triangle apex angle, ensuring a
            well-conditioned (non-degenerate) anchor triple. Default is 0.4.

    Returns:
        list[np.ndarray]:
            Candidate (im)proper rotation matrices (always includes the
            identity).
    """
    dists = np.linalg.norm(coords, axis=1) if dists is None or len(dists) == 0 else dists

    # anchor-triple selection: nearest atom to the centre, plus two more forming a well-conditioned
    # (non-degenerate) triangle:
    order = np.argsort(dists)  # sort by distance from the centre
    anchor_idxs = [int(order[0])]  # anchor triple: atom nearest the centre first
    for idx in order[1:]:
        vec = coords[idx] - coords[anchor_idxs[0]]  # vector from anchor to current site
        candidate_leg_length = np.linalg.norm(vec)
        if candidate_leg_length < min_leg:
            continue  # too short -> not a valid leg

        if len(anchor_idxs) == 1:  # first leg -> add to anchor triple
            anchor_idxs.append(int(idx))
        else:  # check against other two sites; non-degenerate triangle matching constraints?
            leg_1 = coords[anchor_idxs[1]] - coords[anchor_idxs[0]]
            # |a x b|/(|a||b|) = sin(θ); reject near-collinear triples:
            sin_theta = np.linalg.norm(np.cross(leg_1 / np.linalg.norm(leg_1), vec / candidate_leg_length))
            if sin_theta > min_sin:
                anchor_idxs.append(int(idx))
                break  # valid anchor triple completed

    if len(anchor_idxs) < 3:  # degenerate cluster geometry; no rotations determinable
        return [np.eye(3)]

    def orthonormal_frame(p0, p1, p2):  # orthonormal frame from non-degenerate triple
        e1 = p1 - p0
        e1 = e1 / np.linalg.norm(e1)  # unit vector along leg 1
        e2 = p2 - p0
        # Gram-Schmidt: subtract leg 2's component along e1, leaving its perpendicular part:
        e2 = e2 - (e2 @ e1) * e1
        e2 = e2 / np.linalg.norm(e2)  # unit vector perpendicular to e1, maximally-aligned with leg 2:
        # cross product gives third vector for right-handed orthonormal frame:
        return np.array([e1, e2, np.cross(e1, e2)]).T

    anchor_coords = coords[anchor_idxs]
    anchor_pair_dists = {
        (i, j): np.linalg.norm(anchor_coords[i] - anchor_coords[j]) for i, j in combinations(range(3), 2)
    }  # all pairs of the anchor triple
    pair_tol = 2 * symprec  # atoms each matched within symprec -> pair distances within 2*symprec
    frame_a_inv = orthonormal_frame(*anchor_coords).T  # orthogonal, so inverse = transpose
    reflection = np.diag([1.0, 1.0, -1.0])
    # candidate image atoms for each anchor: same species, and within |t| + noise of its distance from the
    # centre (as |x| = |R @ x + t| for any rotation R (and translation t) that preserves distances):
    candidate_idxs = [
        [
            int(j)
            for j in np.where(species == species[anchor_idxs[k]])[0]  # matching species
            if dists[j] <= dists[anchor_idxs[k]] + t_max + symprec  # distance within tolerance range
        ]
        for k in range(3)  # for each atom in triple
    ]
    rotations: list[np.ndarray] = [np.eye(3)]
    for j0 in candidate_idxs[0]:
        for j1 in candidate_idxs[1]:
            # pair-distance pruning: image pair must reproduce the anchor pair separation:
            if abs(np.linalg.norm(coords[j1] - coords[j0]) - anchor_pair_dists[(0, 1)]) > pair_tol:
                continue
            for j2 in candidate_idxs[2]:
                if (  # distances match tolerance range?
                    abs(np.linalg.norm(coords[j2] - coords[j0]) - anchor_pair_dists[(0, 2)]) > pair_tol
                    or abs(np.linalg.norm(coords[j2] - coords[j1]) - anchor_pair_dists[(1, 2)]) > pair_tol
                ):
                    continue
                # surviving triple -> frame-matched rotation, plus its improper (reflected) variant:
                frame_b = orthonormal_frame(coords[j0], coords[j1], coords[j2])
                # F_b = R @ F_a; R = F_b @ F_a^-1:
                # both frames are built right-handed (via the cross product), so frame matching alone
                # always yields a proper rotation; composing with a reflection generates the improper
                # counterpart (mirror/S_n/inversion) that maps the same triple correspondence:
                for rotation in (frame_b @ frame_a_inv, frame_b @ reflection @ frame_a_inv):
                    if _matching_rot_index(rotations, rotation, rot_tol=0.5) is None:  # don't duplicate
                        rotations.append(rotation)
    return rotations


def _map_residual(
    coords: np.ndarray,
    species: np.ndarray,
    trees: dict | None = None,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    radius: float | None = None,
    symprec: float = 0.1,
    dists: np.ndarray | None = None,
) -> tuple[bool, float, int]:
    """
    Maximum nearest-neighbour residual (displacement) mapping test atoms with
    ``x -> rotation @ x + translation``.

    An atom is tested if and only if its predicted image lands within
    ``radius - symprec`` of the centre, where its true partner (within
    ``symprec``, if the operation is genuine) is guaranteed to be inside the
    extracted local sphere -- so boundary truncation can never falsely reject
    a true operation, while every observable atom image is still checked.

    Args:
        coords (np.ndarray):
            ``(N, 3)`` Cartesian coordinates of the local atomic cluster,
            relative to the (rough) defect centre.
        species (np.ndarray):
            ``(N,)`` element symbols of the local atomic cluster.
        trees (dict | None):
            Dict of per-species ``scipy`` ``KDTree`` objects, built on the
            local cluster coordinates. If ``None`` (default), rebuilt from
            ``coords``/``species``.
        rotation (np.ndarray | None):
            Candidate operation rotation matrix (Cartesian). Defaults to
            the identity.
        translation (np.ndarray | None):
            Candidate operation translation vector (Cartesian). Defaults to
            the zero vector.
        radius (float | None):
            Radius (in Å) of the local environment extraction sphere. If
            ``None`` (default), set to ``max(dists) + symprec`` so the provided
            cluster is fully observable.
        symprec (float):
            Distance tolerance (in Å), as in ``local_point_symmetry``.
            Default is 0.1 Å.
        dists (np.ndarray | None):
            ``(N,)`` precomputed distances of each atom from the centre
            (norms of ``coords``). If ``None`` or empty (default),
            recomputed from ``coords``.

    Returns:
        tuple[bool, float, int]:
            ``(accepted, max_residual, n_tested)``: whether the operation
            was accepted, the maximum nearest-neighbour mapping residual
            (displacement) in Å, and the number of atoms tested.
    """
    dists = np.linalg.norm(coords, axis=1) if dists is None or len(dists) == 0 else dists
    trees = trees or {sp: KDTree(coords[species == sp]) for sp in set(species)}
    rotation = np.eye(3) if rotation is None else rotation
    translation = np.zeros(3) if translation is None else translation
    if radius is None:
        radius = float(dists.max()) + symprec

    mapped = coords @ rotation.T + translation  # apply the candidate operation to all atoms

    # only test atoms whose images land observably inside the sphere; ``coords`` are relative to centre:
    mask = np.linalg.norm(mapped, axis=1) < radius - symprec  # mask for sites within radius - symprec
    n_test, max_residual = int(mask.sum()), 0.0

    # enforce minimum test-set size to prevent vacuous certification (a spurious operation trivially
    # "passing" against a near-empty test region); the bar is set per-operation as half the number of atoms
    # whose images are guaranteed testable (i.e. atoms within ``radius - symprec - |translation|``).
    # Some near-boundary atoms may be untestable due to noise/truncation effects, but upstream ``t_max``
    # handling should ensure this is never a majority of the test set (preventing any true ops from
    # breaking here):
    n_guaranteed = (dists < radius - symprec - np.linalg.norm(translation)).sum()  # guaranteed testable
    if n_test == 0 or n_test < round(0.5 * n_guaranteed):  # vacuous test region -> cannot certify anything
        return False, np.inf, n_test

    # per-species nearest-neighbour residual, early exit on failure:
    for _sp, _sp_mask, nn_dists, _nn_idxs in _mapped_species_matches(species, trees, mapped, mask):
        max_residual = max(max_residual, nn_dists.max())
        if max_residual > symprec:  # break early when maximum residual exceeds tolerance
            return False, max_residual, n_test
    return True, max_residual, n_test


def _mapped_species_matches(
    species: np.ndarray, trees: dict, mapped: np.ndarray, mask: np.ndarray
) -> Iterable[tuple]:
    """
    Per-species nearest-neighbour queries for masked atom images, shared by
    ``_map_residual`` and ``_refine_symm_op``.

    A generator, so that ``_map_residual``'s early exit on residual failure
    (important, as many candidate operations are tested) skips querying any
    remaining species.

    Args:
        species (np.ndarray):
            ``(N,)`` element symbols of the local atomic cluster.
        trees (dict):
            Dict of per-species ``scipy`` ``KDTree`` objects, built on cluster
            coordinates (for nearest-neighbour queries).
        mapped (np.ndarray):
            ``(N, 3)`` Cartesian coordinates of cluster atoms under the
            candidate operation (``coords @ rotation.T + translation``).
        mask (np.ndarray):
            ``(N,)`` boolean mask of atoms to test (e.g. those whose image
            lands within the observable local sphere).

    Yields:
        tuple[str, np.ndarray, np.ndarray, np.ndarray]:
            ``(species_symbol, species_mask, nn_dists, nn_idxs)`` for each
            species with at least one tested atom; ``nn_dists``/``nn_idxs``
            are the nearest same-species neighbour distances/indices for
            the ``mapped[species_mask]`` atoms.
    """
    for species_symbol in sorted(set(species)):
        species_mask = mask & (species == species_symbol)
        if species_mask.any():  # at least one atom of this species to test
            nn_dists, nn_idxs = trees[species_symbol].query(mapped[species_mask])  # NN queries, for mapped
            yield species_symbol, species_mask, nn_dists, nn_idxs


def _refine_symm_op(
    coords: np.ndarray,
    species: np.ndarray,
    trees: dict,
    species_coords: dict,
    rotation: np.ndarray,
    translation: np.ndarray,
    radius: float,
    symprec: float,
    match_tol: float,
    refine_rotation: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine a candidate ``(rotation, translation)`` operation against matched
    atom pairs: mean-offset update of the translation, and optionally a full
    Procrustes refit of the rotation (iterated; needed for noisy
    geometry-derived candidate rotations from
    ``_candidate_rotations_from_cluster``).

    Algorithm:

    1. Apply the current ``(R, t)`` to the cluster: ``x -> R @ x + t``.
    2. Keep only images that land inside the observable local sphere
       (outside it, partners may be missing from the truncated cluster).
    3. Match each kept image to its nearest same-species atom (within
       ``match_tol``); collect the corresponding source -> destination pairs.
    4. If ``refine_rotation`` and >=3 pairs: solve the orthogonal Procrustes
       problem for the best ``R`` aligning source -> destination pairs, then
       set ``t`` from the centroids. Else: shift ``t`` by the mean residual of
       the matched pairs (with ``R`` fixed).
    5. If ``refine_rotation``, rematch and repeat (up to 3x): better ``R`` can
       change which atoms pair, so one shot is often insufficient.
    """
    for _ in range(3 if refine_rotation else 1):  # rematch & refit when refining R; one-shot for t-only:
        mapped = coords @ rotation.T + translation  # map atoms with the current candidate operation
        mask = np.linalg.norm(mapped, axis=1) < radius - symprec  # only test images within local sphere
        matched_src, matched_dst = [], []
        # match mapped atoms to their nearest same-species partner:
        for species_symbol, species_mask, nn_dists, nn_idxs in _mapped_species_matches(
            species, trees, mapped, mask
        ):
            close = nn_dists < match_tol
            if close.any():
                matched_src.append(coords[species_mask][close])  # original positions
                matched_dst.append(species_coords[species_symbol][nn_idxs[close]])  # matched partners
        if not matched_src:
            return rotation, translation  # no matches found, return current operation as-is

        src, dst = np.concatenate(matched_src), np.concatenate(matched_dst)
        if refine_rotation and len(src) >= 3:  # >=3 pairs needed for a 3D fit
            # orthogonal Procrustes refinement: R = argmin_Q∈O(3) Σ‖Q @ (src_i - src̄) - (dst_i - dst̄)‖²
            # via SVD of the cross-covariance; determinant also left unconstrained so mirrors / inversion
            # refine naturally (-1 determinant) as well as proper rotations.
            src_centroid, dst_centroid = src.mean(axis=0), dst.mean(axis=0)
            U, _S, Vt = np.linalg.svd((src - src_centroid).T @ (dst - dst_centroid))
            rotation = (U @ Vt).T  # orthogonal Procrustes solution (determinant unconstrained)
            translation = dst_centroid - rotation @ src_centroid  # t that maps src̄ → dst̄ under R
        else:  # translation-only update: mean matched-pair residual (R held fixed):
            translation = translation + (dst - (src @ rotation.T + translation)).mean(axis=0)
    return rotation, translation


def _ops_profile(rotations: Iterable[np.ndarray]) -> tuple:
    """
    Basis-independent profile of a set of point operations: the sorted counts
    of their ``(determinant, trace)`` values (which classify each operation
    type: E, C2, C3, C4, C6, inversion, mirror, S3, S4, S6).

    Args:
        rotations (Iterable[np.ndarray]):
            Iterable of (Cartesian) rotation matrices.

    Returns:
        tuple:
            Sorted tuple of ``((det, trace), count)`` pairs. E.g. for the
            Td point group (``-43m``, order 24):
            ``(((-1, -1), 6), ((-1, 1), 6), ((1, -1), 3), ((1, 0), 8),
            ((1, 3), 1))`` (6 S4, 6 sigma_d, 3 C2, 8 C3, 1 E operations).
    """
    counts: dict[tuple[int, int], int] = {}
    for rotation in rotations:
        key = (int(np.round(np.linalg.det(rotation))), int(np.round(np.trace(rotation))))
        counts[key] = counts.get(key, 0) + 1
    return tuple(sorted(counts.items()))


@lru_cache(maxsize=1)
def _pointgroup_profile_table() -> dict[tuple, str]:
    """
    Map from operation-type profile (``_ops_profile``) to Schoenflies point
    group symbol, for the 32 crystallographic point groups.

    The profiles are unique across all 32 point groups, so this provides basis-
    independent point group identification from any closed set of (Cartesian)
    point operation matrices.

    Cached so lazily-generated only once.
    """
    return {
        _ops_profile([op.rotation_matrix for op in PointGroup(herm_symbol).symmetry_ops]): sch_symbol
        for sch_symbol, herm_symbol in _SCH_to_HERM.items()
    }


def _schoenflies_from_cartesian_ops(rotations: Sequence[np.ndarray]) -> str:
    """
    Schoenflies point group symbol for a closed set of (possibly noisy)
    Cartesian point operation matrices.
    """
    symbol = _pointgroup_profile_table().get(_ops_profile(rotations))
    if symbol is None:  # cannot occur for a closed group of crystallographic operations
        warnings.warn(
            "The determined local symmetry operations do not correspond to a crystallographic point group "
            "(possibly due to residual structural noise); returning C1 (no symmetry)."
        )
        return "C1"
    return symbol


def _defect_coords_from_structures(defect_supercell: Structure, bulk_supercell: Structure) -> np.ndarray:
    """
    Cartesian defect-site coordinates from bulk vs defect structure comparison.
    """
    from doped.analysis import defect_site_from_structures  # avoid circular import

    site = defect_site_from_structures(defect_supercell, bulk_supercell, _parameter_order_warn=False)
    assert isinstance(site, PeriodicSite)
    return site.coords


def local_point_symmetry(
    defect_supercell: Structure,
    bulk_supercell: Structure | None = None,
    defect_frac_coords: ArrayLike | None = None,
    symprec: float = 0.1,
    centre_error_range: float | None = None,
    bulk_symprec: float = 0.01,
    radius: float | None = None,
    verbose: bool = False,
    _first_pass: bool = True,
) -> tuple[str, list[tuple[np.ndarray, np.ndarray]], dict]:
    r"""
    Determine the point symmetry of the local environment around a defect (or
    other local perturbation) in a (supercell) structure, by direct isometry
    analysis of the `local` defect environment -- rather than global symmetry
    analysis with ``spglib``.

    Symmetry analysis of relaxed defect supercells with global space-group
    tools (e.g. ``spglib``) fails for `periodicity-breaking` supercells (i.e.
    supercells whose shape breaks translational symmetries of the host crystal,
    making sites which are equivalent in the host crystal inequivalent under
    the supercell's reduced translational symmetry). This function instead
    analyses only the `local` defect environment (the atoms within the minimum
    periodic image distance of the defect centre), following the algorithmic
    structure of ``spglib`` point symmetry analysis but adapted to a finite
    cluster size:

    1. Place a rough cluster centre at the defect site (``defect_frac_coords``;
       else taken from ``defect_site_from_structures`` if ``bulk_supercell``
       provided, or ``guess_defect_position`` without), and extract the local
       atomic cluster with this centre point and a radius equal to half the
       minimum periodic image distance.
    2. Candidate rotations are taken as the host crystal's point operations
       in the Cartesian frame (from the bulk primitive cell; independent of
       supercell shape), or generated directly from the local atomic geometry
       if no bulk reference is available.
    3. For each candidate rotation ``R``, candidate translations are
       enumerated from anchor-atom correspondences ``t = x_b - R @ x_a`` in the
       local cluster -- as ``spglib`` does for space-group translations.
    4. ``(R, t)`` is accepted if it maps every atom whose predicted image
       lies within the local cluster onto a matching same-species atom within
       ``symprec``, accounting for boundary truncation.
    5. Group closure (i.e. combining any two operations gives another operation
       in the group, required for any valid set of symmetry operations) is
       enforced on the accepted operation set (dropping the worst-residual
       operations until closed). The defect symmetry centre is then `derived`
       from the accepted operations as their common fixed point (least-squares
       fit), and point group identified from the fitted symmetry operations.

       If this derived centre differs appreciably (> ``symprec``) from the
       cluster centre used, the analysis is re-run once, recentred on the
       `derived` centre, keeping the result which certifies the most operations
       -- extending the tolerance for imperfect defect/perturbation (cluster)
       centre positions. `In the noise-free limit`, off-centre cluster
       placement can only spuriously `lower` the certified symmetry; for
       borderline distortions of magnitude ~``symprec``, however, a shifted
       placement can alter cluster membership (test region) and flip the
       symmetry assignment in either direction.

    Args:
        defect_supercell (|Structure|):
            The defect (supercell) structure.
        bulk_supercell (|Structure| | None):
            The bulk (pristine, reference) supercell structure, if available.
            If provided, candidate rotations are taken from the bulk crystal
            symmetry (recommended; most robust) and the defect position (if not
            provided) is determined from bulk vs defect structure comparison
            (``defect_site_from_structures``). Otherwise, candidate operations
            are generated directly from the atomic geometry about the (guessed)
            defect position. Default is ``None``.
        defect_frac_coords (ArrayLike | None):
            Approximate fractional coordinates of the defect position in
            ``defect_supercell``. Only used to place the cluster sphere for
            symmetry analysis; the symmetry centre itself is derived from the
            identified symmetry operations, with the analysis re-run recentred
            on the derived centre when it differs appreciably from the input
            (see step 5 above). The true centre must lie within
            ~``centre_error_range`` of the input (tightening in small
            supercells; see ``centre_error_range``) to be recovered, with the
            recentring re-run typically extending this somewhat further. If
            ``None`` (default), the defect position is taken from
            ``defect_site_from_structures`` when ``bulk_supercell`` is
            provided, or ``guess_defect_position`` otherwise.
        symprec (float):
            Distance tolerance (in Å) for symmetry determination; an operation
            is accepted if it maps each (locally observable) atomic position
            onto a matching position within ``symprec`` (matching the role of
            ``symprec`` in ``spglib`` and other ``doped`` symmetry functions;
            here a strict Cartesian distance tolerance on the atom-mapping
            residuals). Default is 0.1 Å (appropriate for relaxed structures
            with residual structural noise; while ~0.01 Å is typically more
            appropriate for unrelaxed/idealised/noise-free geometries).
        centre_error_range (float | None):
            Maximum expected error (in Å) in the rough defect centre placement,
            setting the (upper) bound ``t_max`` on candidate operation
            translations. A centre error ``d`` needs a fixing translation ``t``
            of up to ~``2*d``, so the true centre must lie within ~``t_max/2``
            of the input to be recovered; ``t_max`` is capped both by
            ``2*centre_error_range`` and the internal cluster test radius, so
            the effective tolerance is ~``centre_error_range`` in typical
            (min-image >= 10 Å) supercells but tightens in small supercells. If
            ``None`` (default), uses 1.5 Å, or 3.0 Å when the defect position
            is taken from ``guess_defect_position`` (no bulk reference and no
            ``defect_frac_coords``) to allow for a larger potential error.
        bulk_symprec (float):
            Distance tolerance (in Å) for ``spglib`` symmetry analysis of the
            (pristine) ``bulk_supercell``, used to generate the candidate
            rotations for the local symmetry analysis. Unused if
            ``bulk_supercell`` is ``None``. Default is 0.01 Å (the ``pymatgen``
            / ``spglib`` default, appropriate for noise-free unrelaxed /
            idealised structures).
        radius (float | None):
            Radius (in Å) of the local atomic cluster extracted around the
            defect centre for symmetry analysis. If ``None`` (default), uses
            half the minimum periodic image distance of ``defect_supercell``
            (capped at 12 Å) -- the maximum radius free of periodic-image
            artefacts. Smaller values can be used to restrict the analysis to
            a more local environment, e.g. to obtain the local point symmetry
            of an individual point defect within a (separated) defect complex.
        verbose (bool):
            If ``True``, prints diagnostic information on the local symmetry
            analysis. Default is ``False``.

    Returns:
        tuple[str, list, dict]:
            The Schoenflies point group symbol; the fitted symmetry operations
            as ``(rotation_matrix, translation_vector)`` pairs (Cartesian, in
            the frame of the extracted local cluster); and an info dict with
            diagnostics:

            - ``"centre_cart"``: the `derived` defect symmetry centre in
              Cartesian coordinates;
            - ``"cluster_centre_cart"``: the local cluster centre used for the
              (final) analysis pass -- the input/determined defect position,
              or the ops-derived centre of the first pass if recentred;
            - ``"fixed_point_consistency"``: max deviation (Å) of the `derived`
              centre from being a true fixed point of each fitted operation;
            - ``"closed"``: whether the accepted operations formed a closed
              group (before enforcement);
            - ``"residuals"``: best mapping residual (displacement) per
              candidate operation, allowing quantification of the separation
              between accepted and rejected operations.
            - ``"degenerate_cluster"``: whether the local cluster contained
              too few atoms (< 4) to certify any symmetry operations (e.g. very
              small supercells or sparse/vacuum-spaced structures), in which
              case ``C1`` is returned here, and global ``spglib`` analysis of
              the defect supercell is used instead by
              |point_symmetry_from_defect_entry| (where determinable).
            - ``"empty_cluster"``: whether the local cluster contained no atoms
              besides the defect itself (e.g. adsorbates/defects in
              vacuum-spaced low-dimensional structures), in which case ``C1``
              is returned here, but nothing local can have relaxed/distorted
              and so the (unrelaxed) bulk site symmetry is the appropriate
              relaxed point symmetry (used automatically by
              |point_symmetry_from_defect_entry|).
    """
    if radius is None:
        radius = min(get_min_image_distance(defect_supercell) / 2, 12)  # cap at 12 Å for large supercells

    # determine cluster centre:
    # only needs to be accurate to ~centre_error_range, as it is just used to place the local cluster
    # sphere; while the symmetry centre itself is then derived from the fitted symmetry operations (with
    # a recentred re-run if it differs appreciably from the input)
    if defect_frac_coords is not None:  # use the provided defect position
        centre_cart = defect_supercell.lattice.get_cartesian_coords(defect_frac_coords)
    elif bulk_supercell is not None:  # determine from bulk vs defect structure comparison
        centre_cart = _defect_coords_from_structures(defect_supercell, bulk_supercell)
    else:  # no bulk reference either; guess the defect position
        from doped.analysis import guess_defect_position  # avoid circular import

        centre_cart = guess_defect_position(defect_supercell)
        if centre_error_range is None:
            centre_error_range = 3.0  # default = 3.0 Å w/guessed position (larger error)

    if centre_error_range is None:
        centre_error_range = 1.5  # default = 1.5 Å, except w/``guess_defect_position``

    coords, species = _extract_defect_cluster(defect_supercell, centre_cart, radius)
    point_symmetry_info: dict = {
        "closed": True,
        "centre_cart": centre_cart,
        "cluster_centre_cart": centre_cart,
        "fixed_point_consistency": 0.0,
        "residuals": [],
        "degenerate_cluster": False,
        "empty_cluster": False,
    }
    if len(coords) < 4:  # degenerate cluster; too few atoms to certify any symmetry -> C1
        point_symmetry_info["degenerate_cluster"] = True
        if len(coords) == 0 or not np.any(np.linalg.norm(coords, axis=1) > symprec):
            # no atoms in the local environment (besides any defect atom at the centre itself; e.g.
            # adsorbates/defects in vacuum-spaced low-dimensional structures), so nothing local can have
            # relaxed/distorted, and the (relaxed) defect point symmetry is the unrelaxed (bulk) site
            # symmetry -- flagged here for the ``relaxed=False`` fallback in
            # ``point_symmetry_from_defect_entry``:
            point_symmetry_info["empty_cluster"] = True
            warnings.warn(
                f"No atoms within the local symmetry analysis radius ({radius:.2f} Å) of the defect site "
                f"(besides the defect itself), so the point symmetry cannot be determined from the local "
                f"environment; returning C1. As nothing local can have relaxed/distorted in this case, "
                f"the (unrelaxed) bulk site symmetry (``relaxed=False``) is the appropriate relaxed "
                f"point symmetry -- used automatically when a bulk reference is available (e.g. in "
                f"``point_symmetry_from_defect_entry`` / ``doped`` parsing)."
            )
        else:  # too few atoms to certify any symmetry operations, but atoms present may have
            # relaxed/distorted (so the bulk site symmetry cannot just be assumed); flagged for global
            # ``spglib`` fallback in ``point_symmetry_from_defect_entry``:
            warnings.warn(
                f"Only {len(coords)} atom(s) within the local symmetry analysis radius ({radius:.2f} Å) "
                f"of the defect site; too few to certify any symmetry operations, so the point symmetry "
                f"cannot be determined from the local environment; returning C1. Global symmetry analysis "
                f"of the defect supercell (``spglib``) may be more appropriate here -- used automatically "
                f"in ``point_symmetry_from_defect_entry`` / ``doped`` parsing (though note this can be "
                f"affected by periodicity-breaking supercell shapes)."
            )
        return "C1", [(np.eye(3), np.zeros(3))], point_symmetry_info

    dists = np.linalg.norm(coords, axis=1)  # distances from the cluster centre
    unique_species = sorted(set(species))

    # per-species coordinates and KD-trees, for the nearest-neighbour residual queries below:
    species_coords = {sp: coords[species == sp] for sp in unique_species}  # cluster coordinates by species
    trees = {sp: KDTree(species_coords[sp]) for sp in unique_species}

    # translation |t| bound: symmetry operation fixed point(s) must stay local (near the defect / cluster
    # centre), and the test region ``(radius - |t| - 2*symprec)`` must cover the defect's first
    # coordination shell:
    non_centre_dists = dists[dists > 0.75]  # distances beyond the defect/cluster centre (site) itself
    min_coordination_shell_distance = (
        float(non_centre_dists.min()) if non_centre_dists.size else float(dists.min())
    ) + 0.5  # just past the 1st coordination shell
    t_max = max(min(2 * centre_error_range, radius - 2 * symprec - min_coordination_shell_distance), 1e-3)
    match_tol = max(4 * symprec, 0.5)  # generous pair-matching radius for iterative refinement

    # get the candidate rotations for local (defect/point) symmetry analysis, independent of the
    # (supercell) shape, and thus immune to periodicity-breaking:
    rotations = (  # either from triplet transformations in the cluster, or from the bulk primitive cell:
        _candidate_rotations_from_cluster(coords, species, dists, symprec, t_max)
        if bulk_supercell is None
        else _bulk_cartesian_rotations(bulk_supercell, symprec=bulk_symprec)
    )

    # now determine candidate translations (t = x_b - R @ x_a); for this we use 'anchor atoms': the atom
    # nearest the rough symmetry centre, of each species; a true symmetry operation must map each anchor
    # onto an orbit partner (same-species atom in the cluster), so we enumerate anchor-partner
    # correspondences before refining:
    anchors = [(sp, species_coords[sp][np.argmin(dists[species == sp])]) for sp in unique_species]

    kept: list[list] = []  # accepted operations, as [rotation, translation, residual]
    best_residuals: list[float] = []  # best residual per candidate rotation (diagnostics)
    # shared cluster/tolerance args for repeated residual/refinement calls below:
    refine_op = partial(
        _refine_symm_op,
        coords,
        species,
        trees,
        species_coords,
        radius=radius,
        symprec=symprec,
        match_tol=match_tol,
    )
    map_residual = partial(
        _map_residual, coords, species, trees, radius=radius, symprec=symprec, dists=dists
    )
    for candidate_rotation in rotations:
        # get candidate translations from anchor -> orbit-partner correspondences, deduped within 0.05 Å:
        candidate_translations: list[np.ndarray] = []
        for anchor_species, species_anchor in anchors:
            for orbit_partner in species_coords[anchor_species]:
                translation = orbit_partner - candidate_rotation @ species_anchor
                if np.linalg.norm(translation) <= t_max and not any(
                    np.linalg.norm(translation - prev) < 0.05 for prev in candidate_translations
                ):  # within t_max and not already in the list (to within 0.05 Å)
                    candidate_translations.append(translation)

        # determine translation with the best (minimum) residual (displacement):
        best_residual = np.inf
        for candidate_translation in candidate_translations:
            rotation, translation = candidate_rotation, candidate_translation
            if bulk_supercell is None:  # noisy geometry-derived rotation; refine before testing
                rotation, translation = refine_op(rotation, translation, refine_rotation=True)

            accepted, residual, _n_test = map_residual(rotation, translation)
            best_residual = min(best_residual, residual)
            if accepted:  # keep the best-residual op per distinct rotation -- Procrustes refinement may
                # result in duplicate rotations, so check and de-dup (taking that with the best residual):
                i_match = _matching_rot_index([op[0] for op in kept], rotation)
                if i_match is None:  # new rotation, so add it to the list
                    kept.append([rotation, translation, residual])
                elif residual < kept[i_match][2]:  # new `best`` residual for this rotation, so overwrite
                    kept[i_match] = [rotation, translation, residual]
        best_residuals.append(best_residual)

    # refine each kept operation's translation by the mean matched-pair offset (removes anchor noise):
    for op in kept:
        rotation, translation = refine_op(op[0], op[1])
        accepted, residual, _n_test = map_residual(rotation, translation)
        if accepted and residual < op[2]:  # improved residual after refinement; overwrite list entries
            op[:] = [rotation, translation, residual]

    if _matching_rot_index([op[0] for op in kept], np.eye(3)) is None:
        # identity not certifiable (degenerate local sphere); include it explicitly
        kept.insert(0, [np.eye(3), np.zeros(3), 0.0])

    # enforce group closure on the accepted (R, t) operations (spglib's analogue: shrink symprec and retry
    # until the operations form a (closed) group); drop worst-residual ops until closed:
    def _is_closed(ops: list[list], t_tol: float) -> bool:
        """
        Check that every pairwise product of ``ops`` matches a kept operation
        (rotation within tolerance, translation within ``t_tol`` Å).
        """
        rotations_kept = [op[0] for op in ops]
        return all(
            (i_product := _matching_rot_index(rotations_kept, op1[0] @ op2[0])) is not None  # rotation
            and np.linalg.norm(op1[0] @ op2[1] + op1[1] - ops[i_product][1]) < t_tol  # and translation
            for op1 in ops
            for op2 in ops
        )

    t_tol = max(0.3, 2 * symprec)
    point_symmetry_info["closed"] = _is_closed(kept, t_tol)
    while not _is_closed(kept, t_tol):
        # ensure identity operation is not considered for removal:
        removable = [i for i, op in enumerate(kept) if np.linalg.norm(op[0] - np.eye(3)) > 0.1]
        if not removable:  # only (pseudo-)identity ops left (e.g. an uncertified pure translation in a
            break  # translation-symmetric cluster); accept as-is rather than crash on ``max([])``
        del kept[max(removable, key=lambda i: kept[i][2])]  # remove the operation with the worst residual

    # the defect symmetry centre is _derived_ from the accepted operations, as their common fixed point
    # (least squares fit); free directions (e.g. along rotation axes, within mirror planes) are pinned to
    # the cluster centre by the small regularisation term:
    # each operation x -> R @ x + t fixes a point c where c = R @ c + t = I @ c;
    # R @ c - I @ c = -t; (R - I) @ c = -t
    lhs, rhs = 1e-6 * np.eye(3), np.zeros(3)  # regularisation term
    for rotation, translation, _residual in kept:  # least squares fit:
        displacement_matrix = rotation - np.eye(3)  # (R - I)
        lhs += displacement_matrix.T @ displacement_matrix  # (R - I)^T @ (R - I)
        rhs += displacement_matrix.T @ -translation  # (R - I)^T @ -t

    centre_offset = np.linalg.solve(lhs, rhs)  # solve for c
    point_symmetry_info["centre_cart"] = centre_cart + centre_offset
    point_symmetry_info["fixed_point_consistency"] = max(
        np.linalg.norm((rotation - np.eye(3)) @ centre_offset + translation)
        for rotation, translation, _residual in kept
    )  # max deviation from fixed point
    point_symmetry_info["residuals"] = sorted(best_residuals)

    symbol = _schoenflies_from_cartesian_ops([op[0] for op in kept])
    if verbose:
        print(
            f"Local symmetry analysis: radius {radius:.2f} Å, {len(coords)} atoms, {len(kept)} operations "
            f"kept (initial group closure: {point_symmetry_info['closed']}), point group {symbol}, "
            f"derived centre {np.round(point_symmetry_info['centre_cart'], 3)} (fixed-point consistency: "
            f"{point_symmetry_info['fixed_point_consistency']:.3f} Å)."
        )

    # in the noise-free limit, an off-centre cluster placement can only spuriously _lower_ the certified
    # symmetry, so if not all candidate rotations were certified, re-run once recentred on the ops-derived
    # symmetry centre (if it differs appreciably from the cluster centre used), keeping the
    # highest-symmetry result. Note that for knife-edge cases (distortions of magnitude ``~symprec``), a
    # shifted placement can instead certify _more_ ops than the true centre (borderline ops slipping under
    # tolerance), but expected to be rare in practice (and they are cases at the borderline of ``symprec``
    # anyway):
    if _first_pass and len(kept) < len(rotations):
        derived_centre = point_symmetry_info["centre_cart"]
        if np.linalg.norm(derived_centre - centre_cart) > symprec:  # differs appreciably; recentre
            retry_result = local_point_symmetry(
                defect_supercell,
                bulk_supercell,
                defect_frac_coords=defect_supercell.lattice.get_fractional_coords(derived_centre),
                symprec=symprec,
                centre_error_range=centre_error_range,
                bulk_symprec=bulk_symprec,
                radius=radius,
                verbose=verbose,
                _first_pass=False,
            )
            if len(retry_result[1]) > len(kept):  # more certified ops (higher symmetry)
                return retry_result

    return symbol, [(op[0], op[1]) for op in kept], point_symmetry_info


def point_symmetry_from_defect_entry(
    defect_entry: DefectEntry,
    symprec: float | None = None,
    relaxed: bool = True,
    verbose: bool | None = None,
    **kwargs,
) -> str:
    r"""
    Get the defect site point symmetry from a |DefectEntry| object.

    If ``relaxed = True`` (default), the point symmetry of the `relaxed` defect
    structure (``defect_entry.defect_supercell``) is determined by local point
    symmetry analysis of the defect environment; see ``local_point_symmetry``
    for algorithm details. Unlike global space-group analysis of the defect
    supercell (with e.g. ``spglib``), this local approach is insensitive to
    periodicity-breaking supercell shapes (i.e. supercells whose shape breaks
    translational symmetries of the host crystal -- as can occur with
    non-diagonal supercell expansions), which otherwise prevent relaxed
    defect point symmetry determination.

    When ``relaxed=True``, the local analysis cluster centre (see
    ``local_point_symmetry``) and the derived defect point symmetry centre
    (the common fixed point of the identified symmetry operations) are also
    stored in ``defect_entry.calculation_metadata``, under the
    ``"symmetry cluster centre"`` and ``"defect symmetry centre"`` keys, as
    fractional coordinates of the defect supercell.

    If ``relaxed = False``, determines the site symmetry of the defect site
    `in the unrelaxed bulk supercell` (i.e. the bulk site symmetry). This
    corresponds to the point symmetry of ``DefectEntry.defect``, or
    equivalently ``calculation_metadata["bulk_site"]``, which for
    vacancies/substitutions is the symmetry of the corresponding bulk site,
    while for interstitials it is the point symmetry of the `relaxed`
    interstitial site when placed in the (unrelaxed) bulk structure.

    The bulk site and relaxed defect point symmetries can be used to compute
    orientational degeneracy factors (with |get_orientational_degeneracy|),
    which are used in the calculation of defect/carrier concentrations and
    Fermi level behaviour (discussion in https://doi.org/10.1039/D2FD00043A,
    https://doi.org/10.1039/D3CS00432E,
    https://doi.org/10.1038/s41578-025-00879-y ...). The computed point
    symmetries and corresponding orientational degeneracy factors can be
    manually checked/edited via the
    ``calculation_metadata['relaxed point symmetry']/['bulk site symmetry']``
    and ``degeneracy_factors['orientational degeneracy']`` attributes.

    Args:
        defect_entry (|DefectEntry|): |DefectEntry| object.
        symprec (float):
            Distance tolerance (in Å) for symmetry determination. As in
            ``spglib``, an operation is considered a symmetry of the (local)
            structure if it maps each (locally observable) atomic position
            onto a matching position within ``symprec``. Default is 0.01 for
            unrelaxed structures (``relaxed=False``; matching the
            ``pymatgen``/``spglib`` default), and 0.1 for relaxed structures
            (to account for residual structural noise, matching that used by
            the ``Materials Project``). You may want to adjust for your
            system (e.g. if there are very slight octahedral distortions
            etc.). For ``relaxed=False``, if
            ``fixed_symprec_and_dist_tol_factor`` is ``False`` (default),
            this value will be automatically adjusted (up to 10x, down to
            0.1x) until the identified equivalent sites from ``spglib`` have
            consistent point group symmetries.
        relaxed (bool):
            If ``False``, determines the site symmetry using the defect site
            `in the unrelaxed bulk supercell` (i.e. the bulk site symmetry),
            otherwise determines the point symmetry of the relaxed defect in
            the defect supercell. Default is ``True``.
        verbose (bool):
            If ``True``, prints diagnostic information on the local symmetry
            analysis (when ``relaxed=True``), or on the trialled ``symprec``
            (and ``dist_tol_factor``) values in equivalent site generation
            (when ``relaxed=False``). Default is ``None`` (no diagnostic
            output).
        **kwargs:
            Additional keyword arguments to pass to ``local_point_symmetry``
            when ``relaxed=True`` (``centre_error_range``, ``bulk_symprec``),
            or ``get_all_equiv_sites`` when ``relaxed=False`` (such as
            ``dist_tol_factor`` and ``fixed_symprec_and_dist_tol_factor``);
            kwargs not applicable to the chosen mode are ignored.

    Returns:
        str: Defect point symmetry (Schoenflies symbol).
    """
    if symprec is None:
        symprec = 0.1 if relaxed else 0.01  # relaxed structures likely have structural noise
        # May need to adjust symprec (e.g. for Ag2Se, symprec of 0.2 is too large as we have very
        # slight distortions present in the unrelaxed material).

    # split off ``local_point_symmetry`` (relaxed) kwargs from ``get_all_equiv_sites`` (unrelaxed)
    # kwargs, so a shared kwargs dict can be used for both (e.g. from ``get_orientational_degeneracy``):
    local_kwargs = {
        k: kwargs.pop(k) for k in ("centre_error_range", "bulk_symprec", "radius") if k in kwargs
    }

    if relaxed:
        defect_supercell = _get_defect_supercell(defect_entry)
        symbol, _ops, info = local_point_symmetry(
            defect_supercell,
            bulk_supercell=_get_bulk_supercell(defect_entry),
            defect_frac_coords=_get_defect_supercell_frac_coords(defect_entry, relaxed=True),
            symprec=symprec,
            verbose=bool(verbose),
            **local_kwargs,
        )
        for key, cart_coords in [
            ("symmetry cluster centre", info["cluster_centre_cart"]),
            ("defect symmetry centre", info["centre_cart"]),
        ]:
            defect_entry.calculation_metadata[key] = (
                defect_supercell.lattice.get_fractional_coords(cart_coords) % 1
            )
        if info.get("empty_cluster"):  # no local environment to have relaxed/distorted (warned in
            # ``local_point_symmetry``), so the relaxed point symmetry is the unrelaxed bulk site symmetry:
            return point_symmetry_from_defect_entry(
                defect_entry,
                symprec=local_kwargs.get("bulk_symprec"),  # ``None`` -> 0.01 Å default
                relaxed=False,
                verbose=verbose,
                **kwargs,
            )
        if info.get("degenerate_cluster"):  # too few local atoms to certify any symmetry (warned in
            # ``local_point_symmetry``); fall back to global ``spglib`` analysis:
            with contextlib.suppress(SymmetryUndeterminedError):
                symbol = schoenflies_from_hermann(
                    get_sga(defect_supercell, symprec=symprec).get_point_group_symbol()
                )
        return symbol

    if defect_entry.defect.defect_type != DefectType.Interstitial:  # take from symmetry dataset of bulk:
        symm_dataset = get_sga(defect_entry.defect.structure, symprec=symprec).get_symmetry_dataset()
        return schoenflies_from_hermann(
            symm_dataset.site_symmetry_symbols[defect_entry.defect.defect_site_index]
        )

    # otherwise, we have an unrelaxed interstitial -> determine via equivalent sites analysis:
    # NOTE: ``local_point_symmetry`` on the unrelaxed interstitial structure gives the same result ~10x
    # faster for ideal interstitial sites, but interstitial sites can sit slightly off their ideal
    # positions, which the ``symprec``/``dist_tol_factor`` auto-adjustment in the equiv-sites machinery
    # handles, and the reported bulk site symmetry should remain consistent with ``defect.multiplicity``
    # / ``equivalent_sites`` (generated by this same machinery), so the equiv-sites approach is retained:
    defect_supercell_bulk_site_coords = _get_defect_supercell_frac_coords(defect_entry, relaxed=False)
    if defect_supercell_bulk_site_coords is not None:
        try:
            symm_dataset, unique_sites = _get_symm_dataset_of_struct_with_all_equiv_sites(
                defect_supercell_bulk_site_coords,
                _get_bulk_supercell(defect_entry),
                symprec=symprec,
                species=defect_entry.defect.site.species_string,
                verbose=verbose is True,
                **kwargs,
            )
            # ``site_symmetry_symbols`` should be used (within this equiv sites approach) for unrelaxed
            # defects (rather than ``pointgroup``), as the site symmetry can be lower than the crystal
            # point group, but not vice versa; so when populating all equivalent sites (of the defect site,
            # in the bulk supercell) the overall point group should be retained and is not necessarily the
            # defect site symmetry. e.g. consider populating all equivalent sites of a C1 interstitial site
            # in a structure (such as CdTe), then the overall point group is still the bulk point group,
            # but the site symmetry is in fact C1
            spglib_point_group_symbols = [
                schoenflies_from_hermann(hermann_symbol)
                for hermann_symbol in symm_dataset.site_symmetry_symbols[-len(unique_sites) :]
            ]  # get point group symbols for all unique sites, and take highest symmetry symbol:
            return max(spglib_point_group_symbols, key=group_order_from_schoenflies)

        except AttributeError:  # fall back to direct determination from the Defect object below
            pass

    # otherwise fall back to ``point_symmetry_from_defect``, for unrelaxed case:
    return point_symmetry_from_defect(
        defect_entry.defect, symprec=symprec, verbose=verbose is True, **kwargs
    )


def point_symmetry_from_structure(
    structure: Structure,
    bulk_structure: Structure | None = None,
    defect_position: ArrayLike | None = None,
    coords_are_cartesian: bool = False,
    symprec: float | None = None,
    relaxed: bool = True,
    verbose: bool | None = None,
    **kwargs,
) -> str:
    r"""
    Get the point symmetry of a defect (or other local perturbation) in a given
    (supercell) structure.

    The point symmetry is determined by direct isometry analysis of the local
    defect environment (see |point_symmetry_from_defect_entry| and
    ``local_point_symmetry`` for algorithm details), which is insensitive to
    periodicity-breaking supercell shapes (i.e. supercells whose shape breaks
    translational symmetries of the host crystal -- as can occur with
    non-diagonal supercell expansions), unlike global space-group analysis of
    the supercell (with e.g. ``spglib``).

    If the bulk (pristine, reference) structure is provided
    (``bulk_structure``; recommended), it is used to determine the defect /
    local perturbation position and the candidate symmetry operations of the
    host crystal. Otherwise, the defect / local perturbation position is taken
    from ``defect_position`` if provided, or guessed using SOAP-based local
    environment analysis (``guess_defect_position``; requires ``dscribe``), and
    candidate symmetry operations are generated directly from the local atomic
    geometry. This position is only used to place the cluster sphere for local
    symmetry analysis; the symmetry centre itself is derived from the
    identified symmetry operations, with the analysis re-run recentred on the
    derived centre when it differs appreciably from the initial (guessed)
    position. Without a bulk reference cell, the (guessed) defect position
    (i.e. cluster centre) needs to be accurate to ~1 - 2 Å (depending on the
    supercell size).

    To sanity-check results, the derived symmetry centre and the cluster
    centre used can be obtained by calling ``local_point_symmetry`` directly
    (returned info dict, ``"centre_cart"`` and ``"cluster_centre_cart"`` keys),
    or printed with ``verbose=True``; when working from a |DefectEntry|, they
    are stored in ``calculation_metadata`` (under ``"defect symmetry centre"``
    and ``"symmetry cluster centre"``) by |point_symmetry_from_defect_entry|.

    In the bulk-reference-free case, the local isometry result is also
    cross-checked against global space-group analysis of the defect supercell
    (whose point group matches the defect site symmetry, for supercells
    containing a single defect and without spurious periodicity-breaking),
    taking the higher-symmetry result -- as each approach can only spuriously
    `lower` the true symmetry (periodicity-breaking supercell shapes for the
    global analysis; imperfect cluster centring for the local analysis).

    If ``bulk_structure`` is supplied and ``relaxed`` is set to ``False``, then
    returns the bulk site symmetry of the defect / local perturbation, which
    for vacancies/substitutions is the symmetry of the corresponding bulk site,
    while for interstitials it is the point symmetry of the `relaxed`
    interstitial site when placed in the (unrelaxed) bulk structure.

    Note: this function determines the point symmetry of the local defect /
    perturbation environment. For the global point group of a (perfect) crystal
    structure, use e.g. :func:`~doped.utils.symmetry.get_sga` /
    :meth:`~pymatgen.symmetry.analyzer.SpacegroupAnalyzer.get_point_group_symbol`.

    Args:
        structure (|Structure|):
            Defect (supercell) structure for which to determine the defect
            point symmetry.
        bulk_structure (|Structure|):
            |Structure| object of the bulk (pristine, reference) structure,
            if available. Default is ``None``.
        defect_position (ArrayLike):
            Approximate position of the defect in ``structure`` (fractional
            coordinates by default, or Cartesian if
            ``coords_are_cartesian = True``); only needs to be accurate to a
            few Å (if a bulk reference structure is provided, ~1-3 Å otherwise,
            depending on the supercell size). If ``None`` (default), the defect
            position is determined from bulk vs defect structure comparison if
            ``bulk_structure`` is provided, or guessed using SOAP-based local
            environment analysis (``guess_defect_position``; requires
            ``dscribe``) otherwise.
        coords_are_cartesian (bool):
            If ``True``, ``defect_position`` is interpreted as Cartesian
            coordinates. Default is ``False`` (fractional coordinates).
        symprec (float):
            Distance tolerance (in Å) for symmetry determination. As in
            ``spglib``, an operation is considered a symmetry of the (local)
            structure if it maps each (locally observable) atomic position
            onto a matching position within ``symprec``. Default is 0.01 Å for
            unrelaxed structures (``relaxed=False``; matching the
            ``pymatgen``/``spglib`` default), and 0.1 Å for relaxed structures
            (to account for residual structural noise, matching that used by
            the ``Materials Project``). You may want to adjust for your
            system (e.g. if there are very slight octahedral distortions etc.).
        relaxed (bool):
            If ``False``, determines the site symmetry using the defect site
            `in the unrelaxed bulk supercell` (i.e. the bulk site symmetry;
            requires ``bulk_structure``), otherwise determines the point
            symmetry of the (relaxed) defect / local perturbation in
            ``structure``. Default is ``True``.
        verbose (bool):
            If ``True``, prints diagnostic information about the local symmetry
            analysis. Default is ``None`` (no diagnostic output).
        **kwargs:
            Additional keyword arguments to pass to ``local_point_symmetry``
            when ``relaxed=True`` (``centre_error_range``, ``bulk_symprec``),
            or ``get_all_equiv_sites`` when ``relaxed=False`` (such as
            ``dist_tol_factor`` and ``fixed_symprec_and_dist_tol_factor``);
            kwargs not applicable to the chosen mode are ignored.

    Returns:
        str: Defect point symmetry (Schoenflies symbol).
    """
    if symprec is None:  # expanded symprec for relaxed structures to account for structural noise
        symprec = 0.1 if relaxed else 0.01

    if bulk_structure is not None:  # create a defect entry and use ``point_symmetry_from_defect_entry``:
        defect_entry = template_defect_entry_from_structures(
            structure,
            bulk_structure,
            oxi_state="Undetermined",
            multiplicity=1,
        )

        return point_symmetry_from_defect_entry(
            defect_entry,
            symprec=symprec,
            relaxed=relaxed,
            verbose=verbose,
            **kwargs,
        )

    if not relaxed:
        raise RuntimeError(
            "The bulk site symmetry (`relaxed=False`) cannot be determined without a bulk reference "
            "structure. Please also supply the unrelaxed bulk structure (`bulk_structure`)."
        )

    defect_frac_coords = None
    if defect_position is not None:
        defect_frac_coords = (
            structure.lattice.get_fractional_coords(defect_position)
            if coords_are_cartesian
            else np.asarray(defect_position)
        )

    symbol, _ops, _info = local_point_symmetry(
        structure,
        bulk_supercell=None,
        defect_frac_coords=defect_frac_coords,  # if still None, guessed in ``local_point_symmetry``
        symprec=symprec,
        verbose=bool(verbose),
        **{k: kwargs[k] for k in ("centre_error_range", "radius") if k in kwargs},  # no bulk_symprec
    )

    # cross-check against global space-group analysis of the defect supercell, whose point group matches
    # the defect site symmetry for supercells containing a single defect and no periodicity-breaking. Each
    # approach can only spuriously _lower_ the true symmetry (global analysis: periodicity-breaking
    # supercell shapes; local analysis: imperfect cluster sphere centring, beyond what the ops-derived
    # recentring re-run can recover), so the higher-symmetry result is taken:
    spglib_symbol = None
    with contextlib.suppress(SymmetryUndeterminedError):
        spglib_symbol = schoenflies_from_hermann(
            get_sga(structure, symprec=symprec).get_point_group_symbol()
        )
    if spglib_symbol is not None and group_order_from_schoenflies(
        spglib_symbol
    ) > group_order_from_schoenflies(symbol):
        if verbose:
            print(
                f"Global spglib analysis of the defect supercell gives a higher point symmetry "
                f"({spglib_symbol}) than reference-free local isometry analysis ({symbol}); taking the "
                f"higher-symmetry result (see ``point_symmetry_from_structure`` docstring)."
            )
        return spglib_symbol
    return symbol


def point_symmetry_from_site(
    site: PeriodicSite | np.ndarray | list,
    structure: Structure,
    coords_are_cartesian: bool = False,
    symprec: float = 0.01,
    **kwargs,
) -> str:
    r"""
    Get the point symmetry of a site in a structure.

    Args:
        site (|PeriodicSite| | np.ndarray | list):
            Site for which to determine the point symmetry. Can be a
            |PeriodicSite| object, or a list or numpy array of the
            coordinates of the site (fractional coordinates by default, or
            Cartesian if ``coords_are_cartesian = True``).
        structure (|Structure|):
            |Structure| object for which to determine the point symmetry of
            the site.
        coords_are_cartesian (bool):
            If ``True``, the site coordinates are assumed to be in Cartesian
            coordinates. Default is False.
        symprec (float):
            Symmetry precision to use for determining symmetry operations and
            thus point symmetries with ``spglib``. Default is 0.01. You may
            want to adjust for your system (e.g. if there are very slight
            octahedral distortions etc.). If
            ``fixed_symprec_and_dist_tol_factor`` is ``False`` (default), this
            value will be automatically adjusted (up to 10x, down to 0.1x)
            until the identified equivalent sites from ``spglib`` have
            consistent point group symmetries. Setting ``verbose`` to ``True``
            will print information on the trialled ``symprec`` (and
            ``dist_tol_factor`` values).
        **kwargs:
            Additional keyword arguments to pass to ``get_all_equiv_sites``,
            such as ``dist_tol_factor``, ``fixed_symprec_and_dist_tol_factor``,
            and ``verbose``.

    Returns:
        str: Site point symmetry.
    """
    if isinstance(site, np.ndarray | list):
        site = PeriodicSite(
            species="X", coords=site, lattice=structure.lattice, coords_are_cartesian=coords_are_cartesian
        )

    try:
        symm_dataset, unique_sites = _get_symm_dataset_of_struct_with_all_equiv_sites(
            site.frac_coords,
            structure,
            symprec=symprec,
            species=site.species_string,
            **kwargs,
        )
    except SymmetryUndeterminedError:
        symm_dataset, unique_sites = _get_symm_dataset_of_struct_with_all_equiv_sites(
            site.frac_coords, structure, symprec=symprec, species="X", **kwargs
        )

    spglib_point_group_symbols = [
        schoenflies_from_hermann(hermann_symbol)
        for hermann_symbol in symm_dataset.site_symmetry_symbols[-len(unique_sites) :]
    ]  # get point group symbols for all unique sites, and use the highest symmetry point group symbol:
    return max(spglib_point_group_symbols, key=group_order_from_schoenflies)


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


def get_orientational_degeneracy(
    defect_entry: DefectEntry | None = None,
    relaxed_point_group: str | None = None,
    bulk_site_point_group: str | None = None,
    symprec: float = 0.1,
    bulk_symprec: float = 0.01,
    **kwargs,
) -> float:
    r"""
    Get the orientational degeneracy factor for a given `relaxed`
    |DefectEntry|, by supplying either the |DefectEntry| object or the bulk-
    site & relaxed defect point group symbols (e.g. "Td", "C3v" etc.).

    If a |DefectEntry| is supplied (and the point group symbols are not),
    this is computed by determining the `relaxed` defect point symmetry and the
    (unrelaxed) bulk site symmetry, and then getting the ratio of their point
    group orders (equivalent to the ratio of partition functions or number of
    symmetry operations (i.e. degeneracy)).

    For interstitials, the bulk site symmetry corresponds to the point symmetry
    of the interstitial site with `no relaxation of the host structure`, while
    for vacancies/substitutions it is simply the symmetry of their
    corresponding bulk site. This corresponds to the point symmetry of
    ``DefectEntry.defect``, or
    ``calculation_metadata["bulk_site"]/["unrelaxed_defect_structure"]``.

    The relaxed defect point symmetry is determined by direct isometry analysis
    of the local defect environment (see |point_symmetry_from_defect_entry|
    and ``local_point_symmetry`` for algorithm details), which is insensitive
    to periodicity-breaking supercell shapes (unlike global space-group
    analysis).

    You can also manually determine the relaxed defect and bulk site point
    symmetries, and/or orientational degeneracy, from visualising the
    structures (e.g. using VESTA)(can use |get_orientational_degeneracy| to
    obtain the corresponding orientational degeneracy factor for given
    defect/bulk site point symmetries) and setting the corresponding values in
    ``calculation_metadata['relaxed point symmetry']/['bulk site symmetry']``
    and/or ``degeneracy_factors['orientational degeneracy']`` attributes. Note
    that the bulk site point symmetry corresponds to that of
    ``DefectEntry.defect``, or equivalently
    ``calculation_metadata["bulk_site"]/["unrelaxed_defect_structure"]``, which
    for vacancies/substitutions is the symmetry of the corresponding bulk site,
    while for interstitials it is the point symmetry of the `relaxed`
    interstitial site when placed in the (unrelaxed) bulk structure. The
    degeneracy factor is used in the calculation of defect/carrier
    concentrations and Fermi level behaviour (discussion in
    https://doi.org/10.1039/D2FD00043A, https://doi.org/10.1039/D3CS00432E,
    https://doi.org/10.1038/s41578-025-00879-y...).

    Args:
        defect_entry (|DefectEntry|):
            |DefectEntry| object. (Default = None)
        relaxed_point_group (str | None):
            Point group symmetry (e.g. "Td", "C3v" etc.) of the `relaxed`
            defect structure, if already calculated / manually determined.
            Default is ``None`` (automatically calculated by ``doped``).
        bulk_site_point_group (str | None):
            Point group symmetry (e.g. "Td", "C3v" etc.) of the defect site in
            the bulk, if already calculated / manually determined. For
            vacancies/substitutions, this should match the site symmetry label
            from ``doped`` when generating the defect, while for interstitials
            it should be the point symmetry of the `relaxed` interstitial site,
            when placed in the bulk structure.
            Default is ``None`` (automatically calculated by ``doped``).
        symprec (float):
            Distance tolerance (in Å) for `relaxed` defect point symmetry
            determination (see |point_symmetry_from_defect_entry|). Default
            is ``0.1`` which matches that used by the ``Materials Project``
            and is larger than the ``pymatgen`` default of ``0.01`` to
            account for residual structural noise in relaxed defect
            supercells. You may want to adjust for your system (e.g. if there
            are very slight octahedral distortions etc.).
        bulk_symprec (float):
            Symmetry precision to use for determining symmetry operations and
            thus point symmetries with ``spglib``, for the `unrelaxed` (bulk
            site) point symmetry -- also used for generating the candidate
            rotations (from the bulk structure) in the `relaxed` local symmetry
            analysis. Default is ``0.01`` which matches the ``pymatgen``
            default. You may want to adjust for your system (e.g. if there are
            very slight octahedral distortions etc.).
            If ``fixed_symprec_and_dist_tol_factor`` is ``False`` (default),
            this value will be automatically adjusted (up to 10x, down to 0.1x)
            until the identified equivalent sites from ``spglib`` have
            consistent point group symmetries. Setting ``verbose`` to ``True``
            will print information on the trialled ``symprec`` (and
            ``dist_tol_factor`` values).
        **kwargs:
            Additional keyword arguments to pass to
            |point_symmetry_from_defect_entry|, such as ``dist_tol_factor``,
            ``fixed_symprec_and_dist_tol_factor`` and ``verbose`` (for
            ``get_all_equiv_sites`` in the unrelaxed bulk-site analysis), or
            ``centre_error_range`` (for ``local_point_symmetry`` in the
            relaxed analysis).

    Returns:
        float: Orientational degeneracy factor for the defect.
    """
    if defect_entry is None:
        if relaxed_point_group is None or bulk_site_point_group is None:
            raise ValueError(
                "Either the DefectEntry or both defect and bulk site point group symbols must be "
                "provided for doped to determine the orientational degeneracy! "
            )

    else:
        if relaxed_point_group is None:
            relaxed_point_group = point_symmetry_from_defect_entry(
                defect_entry,
                symprec=symprec,
                bulk_symprec=bulk_symprec,  # also used for bulk candidate rotations in local analysis
                relaxed=True,  # relaxed
                **kwargs,
            )

        if bulk_site_point_group is None:
            bulk_site_point_group = point_symmetry_from_defect_entry(
                defect_entry,
                symprec=bulk_symprec,  # same default as equiv_sites (-> multiplicity) for consistency
                relaxed=False,  # unrelaxed
                **kwargs,
            )

    return group_order_from_schoenflies(bulk_site_point_group) / group_order_from_schoenflies(
        relaxed_point_group
    )


def is_periodic_image(
    sites_1: Iterable[PeriodicSite | np.ndarray],
    sites_2: Iterable[PeriodicSite | np.ndarray],
    frac_tol: float = 0.01,
    same_image: bool = False,
) -> bool:
    r"""
    Determine if the |PeriodicSite|/``frac_coords`` in ``sites_1`` are a
    periodic image of those in ``sites_2``.

    This function determines if the set of fractional coordinates in
    ``sites_1`` are periodic images of those in ``sites_2``, with only unique
    site matches permitted (i.e. no repeat matches; each site can only have
    one match).

    If ``same_image`` is ``True``, then the sites must all be of the same
    periodic image translation (i.e. the same rigid translation vector), such
    that ``sites_1`` can be `rigidly` translated by any combination of lattice
    vectors to match the set of fractional coordinates in ``sites_2``.

    Note that the this function tests if the `full` set of sites is a periodic
    image of the other, and not just that `each` site in ``sites_1`` is
    (individually) a periodic image of a site in ``sites_2`` (for which the
    ``PeriodicSite.is_periodic_image`` method could be used).

    Args:
        sites_1 (list): List of |PeriodicSite|\s or ``frac_coords`` arrays.
        sites_2 (list): List of |PeriodicSite|\s or ``frac_coords`` arrays.
        frac_tol (float): Fractional coordinate tolerance for comparing sites.
        same_image (bool):
            If ``True``, also check that the sites are the `same` periodic
            image translation (i.e. the same rigid translation vector).
            Default is ``False``.

    Returns:
        bool:
            ``True`` if ``sites_1`` is a periodic image of ``sites_2``,
            ``False`` otherwise.
    """
    sites_1_frac_coords = [site.frac_coords if hasattr(site, "frac_coords") else site for site in sites_1]
    sites_2_frac_coords = [site.frac_coords if hasattr(site, "frac_coords") else site for site in sites_2]

    if len(sites_1_frac_coords) != len(sites_2_frac_coords):
        raise ValueError("``is_periodic_image`` requires the same number of sites in both lists!")

    if not same_image:
        return len(sites_1_frac_coords) == len(sites_2_frac_coords) and is_coord_subset_pbc(
            sites_1_frac_coords, sites_2_frac_coords
        )

    lattice = Lattice(np.eye(3))  # if fractional coords
    for site in [next(iter(sites_1)), next(iter(sites_2))]:
        if isinstance(site, PeriodicSite):
            lattice = site.lattice

    # first need to match sites with their closest (individual) periodic images, to account for order /
    # permutation invariance:
    site_mapping = _get_site_mapping_from_coords_and_indices(
        sites_1_frac_coords, sites_2_frac_coords, lattice=lattice
    )  # list of tuples of (dist, s1_index, s2_index)
    reordered_sites_1_frac_coords = [
        sites_1_frac_coords[s1_idx] for _dist, s1_idx, _s2_idx in site_mapping if s1_idx is not None
    ]

    pbc_frac_dist = np.subtract(reordered_sites_1_frac_coords, sites_2_frac_coords)
    pbc_frac_diff = pbc_frac_dist - np.round(pbc_frac_dist)
    return np.allclose(  # all sites are periodic images
        pbc_frac_diff, np.zeros(pbc_frac_diff.shape), atol=frac_tol
    ) and (  # all sites are _the same_ translation (periodic image)
        np.allclose(pbc_frac_dist, pbc_frac_dist[0], atol=frac_tol)
    )
