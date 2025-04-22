"""
Utility code and functions for symmetry analysis of structures and defects.
"""

import contextlib
import math
import os
import warnings
from collections.abc import Sequence
from functools import lru_cache
from itertools import permutations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pymatgen.analysis.defects.core import DefectType
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Composition, IStructure, Lattice, PeriodicSite, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.symmetry.analyzer import SymmetryUndeterminedError
from pymatgen.transformations.standard_transformations import SupercellTransformation
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sympy import Eq, simplify, solve, symbols
from tqdm import tqdm

from doped.core import Defect, DefectEntry
from doped.utils.efficiency import IStructure as doped_IStructure
from doped.utils.efficiency import SpacegroupAnalyzer
from doped.utils.parsing import (
    _get_bulk_supercell,
    _get_defect_supercell,
    _get_defect_supercell_frac_coords,
    _get_defect_supercell_site,
    _get_unrelaxed_defect_structure,
    get_site_mapping_indices,
)


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
    """
    return solve(equation, variable)


def _set_spglib_warnings_env_var():
    """
    Set the SPGLIB environment variable to suppress spglib warnings.
    """
    os.environ["SPGLIB_WARNING"] = "OFF"


def _check_spglib_version():
    """
    Check the versions of spglib and its C libraries, and raise a warning if
    the correct installation instructions have not been followed.
    """
    import spglib

    python_version = spglib.__version__
    c_version = spglib.spg_get_version_full()

    if python_version != c_version:
        warnings.warn(
            f"Your spglib Python version (spglib.__version__ = {python_version}) does not match its C "
            f"library version (spglib.spg_get_version_full() = {c_version}). This can lead to unnecessary "
            f"spglib warning messages, but can be avoided by upgrading spglib with `pip install --upgrade "
            f"spglib`."
            # No longer required as of spglib v2.5:
            # f"- First uninstalling spglib with both `conda uninstall spglib` and `pip uninstall spglib` "
            # f"(to ensure no duplicate installations).\n"
            # f"- Then, install spglib with `conda install -c conda-forge spglib` or "
            # f"`pip install git+https://github.com/spglib/spglib "
            # f"--config-settings=cmake.define.SPGLIB_SHARED_LIBS=OFF` as detailed in the doped "
            # f"installation instructions: https://doped.readthedocs.io/en/latest/Installation.html"
        )


_check_spglib_version()
_set_spglib_warnings_env_var()


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


def _get_num_places_for_dist_precision(structure: Structure | Lattice, dist_precision: float = 0.001):
    """
    Given a structure or lattice, get the number of decimal places that we need
    to keep / can round to for `fractional coordinates` (``frac_coords``), to
    maintain a distance precision of ``dist_precision`` in Å.

    Intended for use with the ``_round_floats()`` function, to achieve cleanly
    formatted structure outputs while ensuring no significant rounding errors
    are introduced in site positions (e.g. for very large supercells, small
    differences in fraction coordinates become significant).

    Args:
        structure:
            The input structure or lattice.
        dist_precision:
            The desired distance precision in Å (default: 0.001).

    Returns:
        int:
            The number of decimal places to keep for fractional coordinates to
            maintain the desired distance precision.
    """
    lattice = structure if isinstance(structure, Lattice) else structure.lattice
    frac_precision = dist_precision / max(lattice.abc)

    # get corresponding number of decimal places for this precision:
    return -1 * min(math.floor(math.log(frac_precision, 10)), 0)


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
            The desired distance precision in Å (default: 0.001).
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
    num_equals = sum(
        np.isclose(coords_for_sorting[i], coords_for_sorting[j], atol=1e-3)
        for i in range(len(coords_for_sorting))
        for j in range(i + 1, len(coords_for_sorting))
    )
    magnitude = _custom_round(np.linalg.norm(coords_for_sorting))
    return (-num_equals, magnitude, *np.abs(coords_for_sorting))


def get_sga(struct: Structure, symprec: float = 0.01, return_symprec: bool = False):
    """
    Get a ``SpacegroupAnalyzer`` object of the input structure, dynamically
    adjusting ``symprec`` if needs be.

    Args:
        struct (Structure):
            The input structure.
        symprec (float):
            The symmetry precision to use (default: 0.01).
        return_symprec (bool):
            Whether to return the fianl ``symprec`` used (default: False).

    Returns:
        SpacegroupAnalyzer: The symmetry analyzer object. If ``return_symprec``
        is ``True``, returns a tuple of the symmetry analyzer object and the
        final ``symprec`` used.
    """
    IStructure.__hash__ = doped_IStructure.__hash__

    return _cache_ready_get_sga(struct, symprec=symprec, return_symprec=return_symprec)


@lru_cache(maxsize=int(1e3))
def _cache_ready_get_sga(struct: Structure, symprec: float = 0.01, return_symprec: bool = False):
    """
    ``get_sga`` code, with hashable input arguments for caching (using
    ``Structure`` hash function from ``doped.utils.efficiency``).
    """
    sga = None
    for trial_symprec in [symprec, 0.1, 0.001, 1, 0.0001]:
        # if symmetry determination fails, increase symprec first, then decrease, then criss-cross
        with contextlib.suppress(SymmetryUndeterminedError):
            sga = SpacegroupAnalyzer(struct, symprec=trial_symprec)
        if sga:
            try:
                _detected_symmetry = sga._get_symmetry()
            except ValueError:  # symmetry determination failed
                continue
            return (sga, trial_symprec) if return_symprec else sga
    import spglib

    raise SymmetryUndeterminedError(
        f"Could not determine symmetry of input structure! Got spglib error: {spglib.get_error_message()}"
    )


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
        site (PeriodicSite):
            ``pymatgen`` ``PeriodicSite`` object.
        fractional (bool):
            If the ``SymmOp`` is in fractional or Cartesian (default)
            coordinates (i.e. to apply to ``site.frac_coords`` or
            ``site.coords``). Default: False
        rotate_lattice (Union[Lattice, bool]):
            Either a ``pymatgen`` ``Lattice`` object (to use as the new lattice
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
            ``pymatgen`` ``Structure`` object.
        fractional:
            If the ``SymmOp`` is in fractional or Cartesian (default)
            coordinates (i.e. to apply to ``site.frac_coords`` or
            ``site.coords``). Default: False
        rotate_lattice:
            If the lattice of the input structure should be rotated according
            to the symmetry operation. Default: True.

    Returns:
        Structure:
            Structure with the symmetry operation applied.
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


def summed_rms_dist(struct_a: Structure, struct_b: Structure) -> float:
    """
    Get the summed root-mean-square (RMS) distance between the sites of two
    structures, in Å.

    Note that this assumes the lattices of the two structures are equal!

    Args:
        struct_a: ``pymatgen`` ``Structure`` object.
        struct_b: ``pymatgen`` ``Structure`` object.

    Returns:
        float:
            The summed RMS distance between the sites of the two structures, in
            Å.
    """
    # orders of magnitude faster than StructureMatcher.get_rms_dist() from pymatgen
    # (though this assumes lattices are equal)
    # set threshold to a large number to avoid possible site-matching warnings
    return sum(get_site_mapping_indices(struct_a, struct_b, threshold=1e10, dists_only=True))


def get_distance_matrix(fcoords: ArrayLike, lattice: Lattice) -> np.ndarray:
    """
    Get a matrix of the distances between the input fractional coordinates in
    the input lattice.

    Args:
        fcoords (ArrayLike):
            Fractional coordinates to get distances between.
        lattice (Lattice):
            Lattice for the fractional coordinates.

    Returns:
        np.ndarray:
            Matrix of distances between the input fractional coordinates in the
            input lattice.
    """
    return _get_distance_matrix(tuple(tuple(i) for i in fcoords), lattice)  # tuple-ify for caching


@lru_cache(maxsize=int(1e2))
def _get_distance_matrix(fcoords: tuple[tuple, ...], lattice: Lattice):
    """
    Get a matrix of the distances between the input fractional coordinates in
    the input lattice.

    This function requires the input fcoords to be given as tuples, to allow
    hashing and caching for efficiency.
    """
    dist_matrix = np.array(lattice.get_all_distances(fcoords, fcoords))
    return (dist_matrix + dist_matrix.T) / 2


def cluster_coords(
    fcoords: ArrayLike,
    struct: Structure,
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
    ``dist_tol`` distance tolerance in Å. ``"single"`` corresponds to the
    Nearest Point algorithm and is the recommended choice for ``method`` when
    ``dist_tol`` is small, but can be sensitive to how many fractional
    coordinates are included in ``fcoords`` (allowing for daisy-chaining of
    sites to give large spaced-out clusters), while ``"centroid"`` or
    ``"ward"`` are good choices to avoid this issue.

    See the ``scipy`` API docs for more info.

    Args:
        fcoords (ArrayLike):
            Fractional coordinates to cluster.
        struct (Structure):
            Structure to use for the lattice, to which the fractional
            coordinates correspond.
        dist_tol (float):
            Distance tolerance for clustering, in Å (default: 0.01). For the
            most part, fractional coordinates with distances less than this
            tolerance will be clustered together (when ``method = "single"``,
            giving the Nearest Point algorithm, as is the default).
        method (str):
            Clustering algorithm to use with ``linkage()`` (default:
            ``"single"``).
        criterion (str):
            Criterion to use for flattening hierarchical clusters from the
            linkage matrix, used with ``fcluster()`` (default: ``"distance"``).

    Returns:
        np.ndarray:
            Array of cluster numbers, matching the shape and order of
            ``fcoords`` (i.e. corresponding to the index/number of the cluster
            to which that fractional coordinate belongs).
    """
    if len(fcoords) == 1:  # only one input coordinates
        return np.array([0])

    condensed_m = squareform(get_distance_matrix(fcoords, struct.lattice), checks=False)
    z = linkage(condensed_m, method=method)
    return fcluster(z, dist_tol, criterion=criterion)


def get_all_equiv_sites(
    frac_coords: ArrayLike,
    structure: Structure,
    symm_ops: list[SymmOp] | None = None,
    symprec: float = 0.01,
    dist_tol: float = 0.01,
    species: str = "X",
    just_frac_coords: bool = False,
) -> list[PeriodicSite | np.ndarray]:
    """
    Get a list of all equivalent sites of the input fractional coordinates in
    ``structure``.

    Tries to use hashing and caching to accelerate if possible.

    Args:
        frac_coords (ArrayLike):
            Fractional coordinates to get equivalent sites of.
        structure (Structure):
            Structure to use for the lattice, to which the fractional
            coordinates correspond.
        symm_ops (list[SymmOp]):
            List of symmetry operations to use for the equivalent site
            generation (can be provided to avoid re-calculation). If not
            provided, the symmetry operations will be determined using
            ``get_sga`` with the chosen ``symprec``.
        symprec (float):
            Symmetry precision to use for determining symmetry operations
            (default: 0.01).
        dist_tol (float):
            Distance tolerance for clustering generated sites (to ensure they
            are truly distinct), in Å (default: 0.01).
        species (str):
            Species to use for the equivalent sites (default: "X").
        just_frac_coords (bool):
            If ``True``, just returns the fractional coordinates of the
            equivalent sites (rather than ``pymatgen`` ``PeriodicSite``
            objects). Default: False.

    Returns:
        list[PeriodicSite | np.ndarray]:
            List of equivalent sites of the input fractional coordinates in
            ``structure``, either as ``pymatgen`` ``PeriodicSite`` objects or
            as fractional coordinates (depending on the value of
            ``just_frac_coords``).
    """
    IStructure.__hash__ = doped_IStructure.__hash__
    try:
        return _cache_ready_get_all_equiv_sites(
            tuple(frac_coords),
            structure,
            tuple(symm_ops) if symm_ops else None,
            symprec,
            dist_tol,
            species,
            just_frac_coords,
        )
    except TypeError:  # issue with hashing (possibly due to ``species`` choice), use raw function
        return _raw_get_all_equiv_sites(
            frac_coords, structure, symm_ops, symprec, dist_tol, species, just_frac_coords
        )


@lru_cache(maxsize=int(1e3))
def _cache_ready_get_all_equiv_sites(
    frac_coords: tuple,
    structure: Structure,
    symm_ops: tuple[SymmOp] | None = None,
    symprec: float = 0.01,
    dist_tol: float = 0.01,
    species: str = "X",
    just_frac_coords: bool = False,
):
    return _raw_get_all_equiv_sites(
        frac_coords, structure, symm_ops, symprec, dist_tol, species, just_frac_coords
    )


def _raw_get_all_equiv_sites(
    frac_coords: ArrayLike,
    structure: Structure,
    symm_ops: Sequence[SymmOp] | None = None,
    symprec: float = 0.01,
    dist_tol: float = 0.01,
    species: str = "X",
    just_frac_coords: bool = False,
):
    if symm_ops is None:
        sga = get_sga(structure, symprec=symprec)
        symm_ops = sga.get_symmetry_operations()  # fractional symm_ops by default

    dummy_site = PeriodicSite(species, frac_coords, structure.lattice)
    x_sites = [
        apply_symm_op_to_site(
            symm_op,
            dummy_site,
            fractional=True,
            rotate_lattice=structure.lattice,  # enforce same lattice, just want transformed frac coords
            just_unit_cell_frac_coords=just_frac_coords,
        )
        for symm_op in symm_ops
    ]
    if not just_frac_coords:
        for site in x_sites:
            site.to_unit_cell(in_place=True)  # faster with in_place

    dist_precision_num_places = _get_num_places_for_dist_precision(structure, dist_tol)
    all_frac_coords = [
        tuple(i.round(dist_precision_num_places) if dist_tol != 0 else i)
        for i in (x_sites if just_frac_coords else [site.frac_coords for site in x_sites])
    ]
    unique_frac_coords, unique_indices = np.unique(all_frac_coords, axis=0, return_index=True)
    unique_x_sites = [x_sites[i] for i in unique_indices]

    cn = cluster_coords(unique_frac_coords, structure, dist_tol=dist_tol, method="single")
    # cn is an array of cluster numbers, of length ``len(unique_frac_coords)``, so we take the set of
    # cluster numbers ``n``, use ``np.where(cn == n)[0]`` to get the indices of ``cn`` /
    # ``unique_frac_coords`` which are in cluster ``n``, and then take the first of each cluster
    # (because here these should be basically the same sites just with possibly small numerical
    # differences due to symmetry operations, unlike when ``cluster_coords`` is used for Voronoi
    # interstitial generation, where we choose the cluster site based on symmetry/distance to host)
    return [unique_x_sites[np.where(cn == n)[0][0]] for n in set(cn)]  # take 1st of each cluster


def get_min_dist_between_equiv_sites(
    site_1: PeriodicSite | Sequence[float] | Defect | DefectEntry,
    site_2: PeriodicSite | Sequence[float] | Defect | DefectEntry,
    structure: Structure | None = None,
    symprec: float = 0.01,
) -> float:
    """
    Convenience function to get the minimum distance (in Å) between equivalent
    sites of two input site/``Defect``/``DefectEntry`` objects in a structure.

    Args:
        site_1 (PeriodicSite | Sequence[float, float, float] | Defect | DefectEntry):
            First site to get equivalent sites of, to determine minimum
            distance to equivalent sites of ``site_2``. Can be a
            ``PeriodicSite`` object, a sequence of fractional coordinates, or a
            ``Defect``/``DefectEntry`` object.
        site_2 (PeriodicSite | Sequence[float, float, float] | Defect | DefectEntry):
            Second site to get equivalent sites of, to determine minimum
            distance to equivalent sites of ``site_1``. Can be a
            ``PeriodicSite`` object, a sequence of fractional coordinates, or a
            ``Defect``/``DefectEntry`` object.
        structure (Structure):
            Structure to use for determining symmetry-equivalent sites of
            ``site_1`` and ``site_2``. Required if ``site_1`` and ``site_2``
            are not ``Defect`` or ``DefectEntry`` objects. Default: None.
        symprec (float):
            Symmetry precision to use for determining symmetry-equivalent
            sites. Default: 0.01.

    Returns:
        float:
            The minimum distance (in Å) between equivalent sites of ``site_1``
            and ``site_2``.
    """

    def _parse_site_to_frac_coords(site):
        if isinstance(site, PeriodicSite):
            return site.frac_coords
        if isinstance(site, DefectEntry):
            return _get_defect_supercell_frac_coords(site)
        if isinstance(site, Defect):
            return site.site.frac_coords
        return site

    frac_coords_1 = _parse_site_to_frac_coords(site_1)
    frac_coords_2 = _parse_site_to_frac_coords(site_2)
    if structure is None:
        for site in [site_1, site_2]:
            if isinstance(site, DefectEntry):
                structure = _get_bulk_supercell(site)
                break
            if isinstance(site, Defect):
                structure = site.structure
                break
    if structure is None:
        raise ValueError(
            "Structure must be provided if site_1 and site_2 are not DefectEntry or Defect objects."
        )

    bulk_lattice = structure.lattice
    bulk_supercell_sga = get_sga(structure, symprec=symprec)
    symm_bulk_struct = bulk_supercell_sga.get_symmetrized_structure()
    bulk_symm_ops = bulk_supercell_sga.get_symmetry_operations()

    def _get_equiv_frac_coords(frac_coords, site):
        try:
            bulk_site = site.calculation_metadata.get("bulk_site") or _get_defect_supercell_site(site)
        except AttributeError:  # not a DefectEntry
            try:
                bulk_site = site.site
            except AttributeError:  # not a Defect
                bulk_site = None

        equiv_sites = []
        if bulk_site is not None:
            with contextlib.suppress(ValueError):  # faster, but will fail for interstitials
                equiv_sites = [i.frac_coords for i in symm_bulk_struct.find_equivalent_sites(bulk_site)]

        if not equiv_sites:
            equiv_sites = get_all_equiv_sites(
                frac_coords,
                symm_bulk_struct,
                bulk_symm_ops,
                symprec=symprec,
                just_frac_coords=True,
            )

        return equiv_sites

    equiv_fcoords_1 = _get_equiv_frac_coords(frac_coords_1, site_1)
    equiv_fcoords_2 = _get_equiv_frac_coords(frac_coords_2, site_2)

    return np.min(bulk_lattice.get_all_distances(equiv_fcoords_1, equiv_fcoords_2))


def _get_symm_dataset_of_struc_with_all_equiv_sites(
    frac_coords, struct, symm_ops=None, symprec=0.01, dist_tol=0.01, species="X"
):
    """
    Get the symmetry dataset of a ``SpacegroupAnalyzer`` object of a structure
    with all equivalent sites of the input fractional coordinates added to
    ``struct``, and also returning the list of unique equivalent sites.

    Tries to use hashing and caching to accelerate if possible.
    """
    IStructure.__hash__ = doped_IStructure.__hash__
    try:
        return _cache_ready_get_symm_dataset_of_struc_with_all_equiv_sites(
            tuple(frac_coords), struct, tuple(symm_ops) if symm_ops else None, symprec, dist_tol, species
        )
    except TypeError:  # issue with hashing (possibly due to ``species`` choice), use raw function
        return _raw_get_symm_dataset_of_struc_with_all_equiv_sites(
            frac_coords, struct, symm_ops, symprec, dist_tol, species
        )


@lru_cache(maxsize=int(1e3))
def _cache_ready_get_symm_dataset_of_struc_with_all_equiv_sites(
    frac_coords, struct, symm_ops=None, symprec=0.01, dist_tol=0.01, species="X"
):
    return _raw_get_symm_dataset_of_struc_with_all_equiv_sites(
        frac_coords, struct, symm_ops, symprec, dist_tol, species
    )


def _raw_get_symm_dataset_of_struc_with_all_equiv_sites(
    frac_coords, struct, symm_ops=None, symprec=0.01, dist_tol=0.01, species="X"
):
    unique_sites = get_all_equiv_sites(
        list(frac_coords), struct, symm_ops, dist_tol=dist_tol, species=species
    )
    struct_with_all_X = _get_struct_with_all_X(struct, unique_sites)
    sga_with_all_X = get_sga(struct_with_all_X, symprec=symprec)
    return sga_with_all_X.get_symmetry_dataset(), unique_sites


def _get_struct_with_all_X(struct, unique_sites):
    """
    Add all sites in unique_sites to a ``copy`` of ``struct``, and return this
    new ``Structure``.
    """
    struct_with_all_X = struct.copy()
    struct_with_all_X.sites += unique_sites
    return struct_with_all_X


def get_equiv_frac_coords_in_primitive(
    frac_coords: ArrayLike,
    primitive: Structure,
    supercell: Structure,
    symm_ops: list[SymmOp] | None = None,
    symprec: float = 0.01,
    dist_tol: float = 0.01,
    equiv_coords: bool = True,
) -> list[np.ndarray] | np.ndarray:
    """
    Get an equivalent fractional coordinates of ``frac_coords`` from a
    supercell, in the primitive cell.

    Returns a list of equivalent fractional coords in the primitive cell if
    ``equiv_coords`` is ``True``.

    Note that there may be multiple possible symmetry-equivalent sites, all of
    which are returned if ``equiv_coords`` is ``True``, otherwise the first
    site in the list (sorted using ``_frac_coords_sort_func``) is returned.

    Args:
        frac_coords (ArrayLike):
            Fractional coordinates in the supercell, for which to get
            equivalent coordinates in the primitive cell.
        primitive (Structure):
            Primitive cell structure.
        supercell (Structure):
            Supercell structure.
        symm_ops (list[SymmOp]):
            List of symmetry operations to use for the equivalent site
            generation (can be provided to avoid re-calculation). If not
            provided, the symmetry operations will be determined using
            ``get_sga`` with the chosen ``symprec``.
        symprec (float):
            Symmetry precision to use for determining symmetry operations
            (default: 0.01).
        dist_tol (float):
            Distance tolerance for clustering generated sites (to ensure they
            are truly distinct), in Å (default: 0.01).
        equiv_coords (bool):
            If ``True``, returns a list of equivalent fractional coords in the
            primitive cell. If ``False``, returns the first equivalent
            fractional coordinates in the list, sorted using
            ``_frac_coords_sort_func``.

    Returns:
        list[np.ndarray] | np.ndarray:
            List of equivalent fractional coordinates in the primitive cell, or
            the first equivalent fractional coordinate in the list (sorted
            using ``_frac_coords_sort_func``), depending on the value of
            ``equiv_coords``.
    """
    unique_sites = get_all_equiv_sites(
        frac_coords, supercell, symm_ops, dist_tol=dist_tol, symprec=symprec
    )
    supercell_with_all_X = _get_struct_with_all_X(supercell, unique_sites)
    prim_with_all_X = get_primitive_structure(supercell_with_all_X, ignored_species=["X"])

    # ensure matched to primitive structure:
    rotated_struct, matrix = _rotate_and_get_supercell_matrix(prim_with_all_X, primitive)
    if rotated_struct is None:
        warnings.warn("Could not find a mapping between the primitive and supercell structures!")
        return None

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

    prim_coord_list = sorted(
        [
            _vectorized_custom_round(np.mod(_vectorized_custom_round(site.frac_coords), 1))
            for site in s2_really_like_s1.sites
            if site.specie.symbol == "X"
        ],
        key=_frac_coords_sort_func,
    )

    return prim_coord_list if equiv_coords else prim_coord_list[0]


def _rotate_and_get_supercell_matrix(
    prim_struct: Structure, target_struct: Structure, ltol: float = 1e-5, atol: float = 1
) -> tuple[Structure, np.ndarray]:
    """
    Rotates the input ``prim_struct`` to match the ``target_struct``
    orientation, and returns the supercell matrix to convert from the rotated
    ``prim_struct`` to the ``target_struct``.

    Returns ``(None, None)`` if no mapping is found.

    Args:
        prim_struct (Structure):
            The primitive structure.
        target_struct (Structure):
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
    possible_mappings = list(prim_struct.lattice.find_all_mappings(target_struct.lattice))
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
        structure: ``pymatgen`` ``Structure`` object.
        vector: Translation vector, fractional or Cartesian.
        frac_coords:
            Whether the input vector is in fractional coordinates.
            (Default: True)
        to_unit_cell:
            Whether to translate the sites to the unit cell.
            (Default: True)

    Returns:
        ``pymatgen`` ``Structure`` object with translated sites.
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
        prim_struct: ``pymatgen`` ``Structure`` object of the primitive cell.
        target_struct: ``pymatgen`` ``Structure`` object of the target cell.
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
                prim_struct, target_struct
            )
            if attempt_prim_struct:  # otherwise failed, stick with original T matrix
                prim_struct = attempt_prim_struct
                transformation_matrix = attempt_transformation_matrix

        if np.allclose(np.rint(transformation_matrix), transformation_matrix, atol=1e-3):
            return prim_struct, np.rint(transformation_matrix)

        return prim_struct, transformation_matrix

    rms_dists_w_candidate_prim_structs_and_T_matrices = []
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
        rms_dist_to_target = summed_rms_dist(
            Structure.from_sites(
                [
                    site.to_unit_cell()
                    for site in (new_prim_struct * transformation_matrix).get_sorted_structure()
                ]
            ),
            target_struct,
        )
        rms_dists_w_candidate_prim_structs_and_T_matrices.append(
            (rms_dist_to_target, new_prim_struct, transformation_matrix)
        )

    closest_match = sorted(  # sort to get ideal primitive cell definition
        rms_dists_w_candidate_prim_structs_and_T_matrices,
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

    pmg_prim_struct = structure.get_primitive_structure()
    candidate_prim_structs = (
        [structure, pmg_prim_struct] if len(structure) == len(pmg_prim_struct) else [pmg_prim_struct]
    )

    for _i in range(4):
        struct = sga.get_primitive_standard_structure()
        candidate_prim_structs.append(struct)

        spglib_dataset = sga.get_symmetry_dataset()
        if not np.allclose(spglib_dataset.origin_shift, 0):
            candidate_prim_structs.append(translate_structure(struct, spglib_dataset.origin_shift))

        sga = get_sga(struct, sga._symprec)  # use same symprec

    candidate_prim_structs.append(sga.find_primitive())
    for candidate_conv_struct in [sga.get_refined_structure(), sga.get_conventional_standard_structure()]:
        if len(candidate_conv_struct) == len(pmg_prim_struct):
            # only also try conventional if equivalent to the primitive cell
            candidate_prim_structs = [candidate_conv_struct, *candidate_prim_structs]

    return candidate_prim_structs


def get_wyckoff(
    frac_coords: ArrayLike,
    struct: Structure,
    symm_ops: list | None = None,
    equiv_sites: bool = False,
    symprec: float = 0.01,
) -> str | tuple:
    r"""
    Get the Wyckoff label of the input fractional coordinates in the input
    structure. If the symmetry operations of the structure have already been
    computed, these can be input as a list to speed up the calculation.

    Args:
        frac_coords (ArrayLike):
            Fractional coordinates of the site to get the Wyckoff label of.
        struct (Structure):
            Structure for which ``frac_coords`` corresponds to.
        symm_ops (list[SymmOp] | None):
            List of ``pymatgen`` ``SymmOp``\s of the structure. If ``None``
            (default), will recompute these from the input structure.
        equiv_sites (bool):
            If ``True``, also returns a list of equivalent sites in ``struct``.
        symprec (float):
            Symmetry precision for ``SpacegroupAnalyzer``.
            Default is 0.01.

    Returns:
        str | tuple:
            The Wyckoff label of the input fractional coordinates in the
            structure. If ``equiv_sites`` is ``True``, also returns a list of
            equivalent sites in the structure.
    """
    symm_dataset, unique_sites = _get_symm_dataset_of_struc_with_all_equiv_sites(
        frac_coords, struct, symm_ops, symprec=symprec
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
            ``pymatgen`` ``Structure`` object, or an array of fractional
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
    is_diagonal = np.allclose(lattice_matrix, np.diag(np.diag(lattice_matrix)))
    symmetric = np.allclose(lattice_matrix, lattice_matrix.T)
    num_negs = np.sum(lattice_matrix < 0)
    diag_sum = np.round(np.sum(np.abs(np.diag(lattice_matrix))), 1)
    flat_matrix = lattice_matrix.flatten()
    num_equals = sum(
        flat_matrix[i] == flat_matrix[j]
        for i in range(len(flat_matrix))
        for j in range(i, len(flat_matrix))
    )  # double (square) counting, but doesn't matter (sorting behaviour the same)
    num_abs_equals = sum(
        abs(flat_matrix[i]) == abs(flat_matrix[j])
        for i in range(len(flat_matrix))
        for j in range(i, len(flat_matrix))
    )  # double (square) counting, but doesn't matter (sorting behaviour the same)
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
        structure (Structure): Structure object.
        return_T (bool):
            Whether to return the transformation matrix from the original
            structure lattice to the new structure lattice (T * Orig = New).
            (Default = False)
        dist_precision (float):
            The desired distance precision in Å for rounding of lattice
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
        structure.cart_coords,  # type: ignore
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
    ``Structure``.

    For some materials (e.g. zinc blende), there are multiple equivalent
    primitive cells (e.g. Cd (0,0,0) & Te (0.25,0.25,0.25); Cd (0,0,0) & Te
    (0.75,0.75,0.75) for F-43m CdTe), so for reproducibility and in line with
    most structure conventions/definitions, take the one with the cleanest
    lattice and structure definition, according to ``struct_sort_func``.

    If ``ignored_species`` is set, then the sorting function used to determine
    the ideal primitive structure will ignore sites with species in
    ``ignored_species``.

    Args:
        structure (Structure):
            Structure to get the corresponding primitive structure of.
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
    IStructure.__hash__ = doped_IStructure.__hash__
    cache_ready_ignored_species = tuple(ignored_species) if ignored_species is not None else None
    cache_ready_kwargs = tuple(kwargs.items()) if kwargs else None

    return _cache_ready_get_primitive_structure(
        structure,
        ignored_species=cache_ready_ignored_species,
        clean=clean,
        return_all=return_all,
        kwargs=cache_ready_kwargs,
    )


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
    (using ``Structure`` hash function from ``doped.utils.efficiency``).
    """
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
    for _i in range(4):
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
) -> tuple[Structure, np.ndarray] | tuple[Structure, np.ndarray, dict[str, np.ndarray]]:
    """
    Get the conventional crystal structure of the input structure, according to
    the Bilbao Crystallographic Server (BCS) definition.

    Also returns an array of the lattice vector swaps (used with ``swap_axes``)
    to convert from the ``spglib`` (``SpaceGroupAnalyzer``) conventional
    structure definition to the BCS definition.

    Args:
        structure (Structure):
            Structure for which to get the corresponding BCS conventional
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
        defect_entry: ``DefectEntry`` object.

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
    sm = StructureMatcher(primitive_cell=False, ignored_species=["X"], comparator=ElementComparator())
    sga_prim_struct = sga.get_primitive_standard_structure()
    s2_like_s1 = sm.get_s2_like_s1(sga_prim_struct, prim_struct_with_X)
    if not s2_like_s1:
        warnings.warn(
            "The transformation from the DefectEntry primitive cell to the spglib primitive "
            "cell could not be determined, and so the corresponding conventional cell site "
            "cannot be identified."
        )
        return None
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
    assert defect_entry.conventional_structure is not None
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

    conv_cell_site = next(site for site in s2_really_like_s1.sites if site.specie.symbol == "X")
    # site choice doesn't matter so much here, as we later get the equivalent coordinates using the
    # Wyckoff dict and choose the conventional site based on that anyway (in the DefectsGenerator
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


def get_wyckoff_dict_from_sgn(sgn: int) -> dict[str, list[list[float]]]:
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
        return [cached_simplify(x.replace("2x", "2*x")) for x in coord_string.split(",")]

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
                    # Assign the expression the value of the corresponding coordinate, and solve
                    # for the new variable
                    # first, special cases with two possible solutions due to PBC:
                    if sympy_expr == cached_simplify("-2*x"):
                        add_new_variable_dict("1+", sympy_expr, coord, temp_dict, variable_dicts)
                    elif sympy_expr == cached_simplify("2*x"):
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


def point_symmetry_from_defect(defect: Defect, symm_ops: list | None = None, symprec: float = 0.01) -> str:
    """
    Get the defect site point symmetry from a `Defect` object.

    Note that this is intended only to be used for unrelaxed, as-generated
    ``Defect`` objects (rather than parsed defects).

    Args:
        defect (Defect): ``Defect`` object.
        symm_ops (list):
            List of symmetry operations of ``defect.structure``, to avoid
            re-calculating. Default is ``None`` (recalculates).
        symprec (float):
            Symmetry tolerance for ``spglib``. Default is 0.01.

    Returns:
        str: Defect point symmetry.
    """
    symm_dataset, _unique_sites = _get_symm_dataset_of_struc_with_all_equiv_sites(
        defect.site.frac_coords,
        defect.structure,
        symm_ops=symm_ops,
        symprec=symprec,
        species=defect.site.species_string if defect.defect_type == DefectType.Interstitial else "X",
    )
    spglib_point_group_symbol = schoenflies_from_hermann(symm_dataset.site_symmetry_symbols[-1])
    if spglib_point_group_symbol is not None:
        return spglib_point_group_symbol

    # symm_ops approach failed, just use diagonal defect supercell approach:
    defect_diagonal_supercell = defect.get_supercell_structure(
        sc_mat=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        dummy_species="X",
    )  # create defect supercell, which is a diagonal expansion of the unit cell so that the defect
    # periodic image retains the unit cell symmetry, in order not to affect the point group symmetry
    sga = get_sga(defect_diagonal_supercell, symprec=symprec)
    return schoenflies_from_hermann(sga.get_point_group_symbol())


def point_symmetry_from_defect_entry(
    defect_entry: DefectEntry,
    symm_ops: list | None = None,
    symprec: float | None = None,
    relaxed: bool = True,
    verbose: bool = True,
    return_periodicity_breaking: bool = False,
) -> str | tuple[str, bool]:
    r"""
    Get the defect site point symmetry from a ``DefectEntry`` object.

    Note: If ``relaxed = True`` (default), then this tries to use the
    ``defect_entry.defect_supercell`` to determine the site symmetry. This will
    thus give the `relaxed` defect point symmetry if this is a ``DefectEntry``
    created from parsed defect calculations. However, it should be noted that
    this is not guaranteed to work in all cases; namely for non-diagonal
    supercell expansions, or sometimes for non-scalar supercell expansion
    matrices (e.g. a 2x1x2 expansion)(particularly with high-symmetry
    materials) which can mess up the periodicity of the cell. ``doped`` tries
    to automatically check if this is the case, and will warn you if so.

    This can also be checked by using this function on your doped `generated`
    defects:

    .. code-block:: python

        from doped.generation import get_defect_name_from_entry
        for defect_name, defect_entry in defect_gen.items():
            print(defect_name,
                  get_defect_name_from_entry(defect_entry, relaxed=False),
                  get_defect_name_from_entry(defect_entry), "\n")

    And if the point symmetries match in each case, then using this function on
    your parsed `relaxed` ``DefectEntry`` objects should correctly determine
    the final relaxed defect symmetry -- otherwise periodicity-breaking
    prevents this.

    If periodicity-breaking prevents auto-symmetry determination, you can
    manually determine the relaxed defect and bulk-site point symmetries,
    and/or orientational degeneracy, from visualising the structures (e.g.
    using VESTA)(can use ``get_orientational_degeneracy`` to obtain the
    corresponding orientational degeneracy factor for given defect/bulk-site
    point symmetries) and setting the corresponding values in the
    ``calculation_metadata['relaxed point symmetry']/['bulk site symmetry']``
    and/or ``degeneracy_factors['orientational degeneracy']`` attributes. Note
    that the bulk-site point symmetry corresponds to that of
    ``DefectEntry.defect``, or equivalently
    ``calculation_metadata["bulk_site"]/["unrelaxed_defect_structure"]``, which
    for vacancies/substitutions is the symmetry of the corresponding bulk site,
    while for interstitials it is the point symmetry of the `final relaxed`
    interstitial site when placed in the (unrelaxed) bulk structure. The
    degeneracy factor is used in the calculation of defect/carrier
    concentrations and Fermi level behaviour (see e.g.
    https://doi.org/10.1039/D2FD00043A & https://doi.org/10.1039/D3CS00432E).

    Args:
        defect_entry (DefectEntry): ``DefectEntry`` object.
        symm_ops (list):
            List of symmetry operations of either the
            ``defect_entry.bulk_supercell`` structure (if ``relaxed=False``) or
            ``defect_entry.defect_supercell`` (if ``relaxed=True``), to avoid
            re-calculating. Default is ``None`` (recalculates).
        symprec (float):
            Symmetry tolerance for ``spglib``. Default is 0.01 for unrelaxed
            structures, 0.1 for relaxed (to account for residual structural
            noise, matching that used by the ``Materials Project``). You may
            want to adjust for your system (e.g. if there are very slight
            octahedral distortions etc.).
        relaxed (bool):
            If ``False``, determines the site symmetry using the defect site
            `in the unrelaxed bulk supercell` (i.e. the bulk site symmetry),
            otherwise tries to determine the point symmetry of the relaxed
            defect in the defect supercell. Default is True.
        verbose (bool):
            If True, prints a warning if the supercell is detected to break the
            crystal periodicity (and hence not be able to return a reliable
            `relaxed` point symmetry). Default is ``True``.
        return_periodicity_breaking (bool):
            If ``True``, also returns a boolean specifying if the supercell has
            been detected to break the crystal periodicity (and hence not be
            able to return a reliable `relaxed` point symmetry) or not. Mainly
            for internal ``doped`` usage. Default is ``False``.

    Returns:
        str:
            Defect point symmetry (and if
            ``return_periodicity_breaking = True``, a boolean specifying if the
            supercell has been detected to break the crystal periodicity).
    """
    if symprec is None:
        symprec = 0.1 if relaxed else 0.01  # relaxed structures likely have structural noise
        # May need to adjust symprec (e.g. for Ag2Se, symprec of 0.2 is too large as we have very
        # slight distortions present in the unrelaxed material).

    # from spglib docs: For atomic positions, roughly speaking, two position vectors x and x' in
    # Cartesian coordinates are considered to be the same if |x' - x| < symprec. The angle distortion
    # between basis vectors is converted to a length and compared with this distance tolerance.
    # we _could_ do bulk-bond length dependent symprec, which seems like it would be physically
    # reasonable (basically being a proxy accounting for larger structural/positional noise for same force
    # noise in DFT supercell calcs), but from testing this didn't seem to really improve accuracy in
    # general (e.g. for Sb2O5 split-interstitial seemed like >0.1 best, while <0.12 required for SrTiO3
    # despite smaller bond length)

    if not relaxed and defect_entry.defect.defect_type != DefectType.Interstitial:
        # then easy, can just be taken from symmetry dataset of defect structure
        symm_dataset = get_sga(defect_entry.defect.structure, symprec=symprec).get_symmetry_dataset()
        return schoenflies_from_hermann(
            symm_dataset.site_symmetry_symbols[defect_entry.defect.defect_site_index]
        )

    supercell = _get_defect_supercell(defect_entry) if relaxed else _get_bulk_supercell(defect_entry)
    supercell_sga = get_sga(supercell, symprec=symprec)

    if symm_ops is None:
        symm_ops = supercell_sga.get_symmetry_operations()

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
    matching = True
    if relaxed:
        if unrelaxed_defect_structure := _get_unrelaxed_defect_structure(defect_entry):
            matching = _check_relaxed_defect_symmetry_determination(
                defect_entry,
                unrelaxed_defect_structure=unrelaxed_defect_structure,
                symprec=symprec,
                verbose=verbose,
            )
        else:
            warnings.warn(
                "`relaxed` was set to True (i.e. get _relaxed_ defect symmetry), but the "
                "`calculation_metadata`/`bulk_entry.structure` attributes are not set for `DefectEntry`, "
                "suggesting that this DefectEntry was not parsed from calculations using doped. This "
                "means doped cannot automatically check if the supercell shape is breaking the cell "
                "periodicity here or not (see docstring) -- the point symmetry groups are not guaranteed "
                "to be correct here!"
            )

    _failed = False

    spglib_point_group_symbol = None
    if relaxed:
        with contextlib.suppress(Exception):
            spglib_point_group_symbol = schoenflies_from_hermann(supercell_sga.get_point_group_symbol())

    if not relaxed or spglib_point_group_symbol is None:
        defect_supercell_bulk_site_coords = _get_defect_supercell_frac_coords(
            defect_entry, relaxed=relaxed
        )
        if defect_supercell_bulk_site_coords is not None:
            try:
                symm_dataset, _unique_sites = _get_symm_dataset_of_struc_with_all_equiv_sites(
                    defect_supercell_bulk_site_coords,
                    supercell,
                    symm_ops=symm_ops,  # defect symm_ops needed for relaxed=True, bulk for relaxed=False
                    symprec=symprec,
                    dist_tol=symprec,
                    species=(
                        defect_entry.defect.site.species_string
                        if defect_entry.defect.defect_type == DefectType.Interstitial
                        else "X"
                    ),
                )

                # Note:
                # This code works to get the site symmetry of a defect site in a periodicity-breaking
                # supercell, but only when the defect has not yet been relaxed. Still has the issue that
                # once we have relaxation around the defect site in a periodicity-breaking supercell,
                # then the (local) point symmetry cannot be easily determined as the whole supercell
                # symmetry is broken.
                # Future work could try a local structure analysis to determine the local point symmetry to
                # counteract this.
                # unique_sites = get_all_equiv_sites(  # defect site but bulk supercell & symm_ops
                #     site.frac_coords, bulk_supercell, symm_ops
                # )
                # sga_with_all_X = _get_sga_with_all_X(  # defect unique sites but bulk supercell
                #     bulk_supercell, unique_sites, symprec=symprec
                # )
                # symm_dataset = sga_with_all_X.get_symmetry_dataset()
            except AttributeError:
                _failed = True

        if defect_supercell_bulk_site_coords is None or _failed:
            point_group = point_symmetry_from_defect(
                defect_entry.defect, symm_ops=symm_ops, symprec=symprec
            )
            # possibly pymatgen DefectEntry object without defect_supercell_site set
            if relaxed:
                warnings.warn(
                    "Symmetry determination failed with the standard approach (likely due to this being a "
                    "DefectEntry which has not been generated/parsed with doped?). Thus the _relaxed_ "
                    "point group symmetry cannot be reliably automatically determined."
                )
                return (point_group, not matching) if return_periodicity_breaking else point_group

            return point_group

        if not relaxed:
            # `site_symmetry_symbols[-1]` should be used (within this equiv sites approach) for unrelaxed
            # defects (rather than `pointgroup`), as the site symmetry can be lower than the crystal point
            # group, but not vice versa; so when populating all equivalent sites (of the defect site,
            # in the bulk supercell) the overall point group is retained and is not necessarily the defect
            # site symmetry. e.g. consider populating all equivalent sites of a C1 interstitial site in a
            # structure (such as CdTe), then the overall point group is still the bulk point group,
            # but the site symmetry is in fact C1.
            # This issue is avoided for relaxed defect supercells as we take the symm_ops of our reduced
            # symmetry cell rather than that of the bulk (so no chance of spurious symmetry upgrade from
            # equivalent sites), and hence the max point symmetry is the point symmetry of the defect
            spglib_point_group_symbol = schoenflies_from_hermann(symm_dataset.site_symmetry_symbols[-1])

            # Note that, if the supercell is non-periodicity-breaking, then the site symmetry can be simply
            # determined using the point group of the unrelaxed defect structure:
            # unrelaxed_defect_supercell = defect_entry.calculation_metadata.get(
            #     "unrelaxed_defect_structure", defect_supercell
            # )
            # return schoenflies_from_hermann(
            #     get_sga(unrelaxed_defect_supercell, symprec).get_symmetry_dataset().pointgroup,
            # )
            # But current approach works for all cases with unrelaxed defect structures

        else:
            # For relaxed defects the "defect supercell site" is not necessarily the true centre of mass of
            # the defect (e.g. for split-interstitials, split-vacancies, swapped vacancies etc),
            # so use 'pointgroup' output (in this case the reduced symmetry avoids the symmetry-upgrade
            # possibility with the equivalent sites, as when relaxed=False)
            spglib_point_group_symbol = schoenflies_from_hermann(symm_dataset.pointgroup)

            # This also works (at least for non-periodicity-breaking supercells) for relaxed defects in
            # most cases, but is slightly less robust (more sensitive to ``symprec`` choice) than the
            # approach above:
            # schoenflies_from_hermann(
            #     get_sga(
            #         defect_supercell, symprec=symprec
            #     ).get_symmetry_dataset().pointgroup
            # )

    if spglib_point_group_symbol is not None:
        return (
            (spglib_point_group_symbol, not matching)
            if return_periodicity_breaking
            else (spglib_point_group_symbol)
        )

    # symm_ops approach failed, just use diagonal defect supercell approach:
    if relaxed:
        raise RuntimeError(
            "Site symmetry could not be determined using the defect supercell, and so the relaxed site "
            "symmetry cannot be automatically determined (set relaxed=False to obtain the (unrelaxed) "
            "bulk site symmetry)."
        )

    defect_diagonal_supercell = defect_entry.defect.get_supercell_structure(
        sc_mat=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        dummy_species="X",
    )  # create defect supercell, which is a diagonal expansion of the unit cell so that the defect
    # periodic image retains the unit cell symmetry, in order not to affect the point group symmetry
    sga = get_sga(defect_diagonal_supercell, symprec=symprec)
    point_group = schoenflies_from_hermann(sga.get_point_group_symbol())
    return (point_group, not matching) if return_periodicity_breaking else point_group


def _check_relaxed_defect_symmetry_determination(
    defect_entry: DefectEntry,
    unrelaxed_defect_structure: Structure = None,
    symprec: float = 0.1,
    verbose: bool = False,
):
    defect_supercell_bulk_site_coords = _get_defect_supercell_frac_coords(defect_entry, relaxed=False)

    if defect_supercell_bulk_site_coords is None:
        raise AttributeError(
            "`defect_entry.defect_supercell_site` or `defect_entry.sc_defect_frac_coords` are not "
            "defined! Needed to check defect supercell periodicity (for symmetry determination)"
        )

    if unrelaxed_defect_structure is None:
        unrelaxed_defect_structure = _get_unrelaxed_defect_structure(defect_entry)

    if unrelaxed_defect_structure is not None:
        unrelaxed_spglib_point_group_symbol = schoenflies_from_hermann(
            get_sga(unrelaxed_defect_structure, symprec=symprec).get_symmetry_dataset().pointgroup,
        )

        bulk_supercell = _get_bulk_supercell(defect_entry)
        if symprec not in [0.1, None]:  # only pre-calculate bulk_symm_ops if not default symprec
            bulk_symm_ops = get_sga(bulk_supercell).get_symmetry_operations()
        else:
            bulk_symm_ops = None

        match = False
        # allow some variation in dist_tol as this can be a little sensitive and not always perfectly
        # mappable to symprec:
        for trial_dist_tol in [symprec, symprec * 0.85, symprec * 1.15]:
            symm_dataset, _unique_sites = _get_symm_dataset_of_struc_with_all_equiv_sites(
                defect_supercell_bulk_site_coords,
                bulk_supercell,
                symm_ops=bulk_symm_ops,
                symprec=symprec,
                dist_tol=trial_dist_tol,
                species=(
                    defect_entry.defect.site.species_string
                    if defect_entry.defect.defect_type == DefectType.Interstitial
                    else "X"
                ),
            )
            bulk_spglib_point_group_symbol = schoenflies_from_hermann(
                symm_dataset.site_symmetry_symbols[-1]
            )

            if bulk_spglib_point_group_symbol == unrelaxed_spglib_point_group_symbol:
                match = True
                break

        if not match:
            if verbose:
                warnings.warn(
                    "`relaxed` is set to True (i.e. get _relaxed_ defect symmetry), but doped has "
                    "detected that the defect supercell is likely a non-scalar matrix expansion which "
                    "could be breaking the cell periodicity and possibly preventing the correct _relaxed_ "
                    "point group symmetry from being automatically determined. You can set relaxed=False "
                    "to instead get the (unrelaxed) bulk site symmetry, and/or manually "
                    "check/set/edit the point symmetries and corresponding orientational degeneracy "
                    "factors by inspecting/editing the "
                    "calculation_metadata['relaxed point symmetry']/['bulk site symmetry'] and "
                    "degeneracy_factors['orientational degeneracy'] attributes."
                )
            return False

        return True

    return False  # return False if symmetry couldn't be checked


def point_symmetry_from_structure(
    structure: Structure,
    bulk_structure: Structure | None = None,
    symm_ops: list | None = None,
    symprec: float | None = None,
    relaxed: bool = True,
    verbose: bool = True,
    return_periodicity_breaking: bool = False,
) -> str | tuple[str, bool]:
    r"""
    Get the point symmetry of a given structure.

    Note: For certain non-trivial supercell expansions, the broken cell
    periodicity can break the site symmetry and lead to incorrect point
    symmetry determination (particularly if using non-scalar supercell matrices
    with high symmetry materials). If the unrelaxed bulk structure
    (``bulk_structure``) is also supplied, then ``doped`` will determine the
    defect site and then automatically check if this is the case, and warn you
    if so.

    This can also be checked by using this function on your doped `generated`
    defects:

    .. code-block:: python

        from doped.generation import get_defect_name_from_entry
        for defect_name, defect_entry in defect_gen.items():
            print(defect_name,
                  get_defect_name_from_entry(defect_entry, relaxed=False),
                  get_defect_name_from_entry(defect_entry), "\n")

    And if the point symmetries match in each case, then using this function on
    your parsed `relaxed` ``DefectEntry`` objects should correctly determine
    the final relaxed defect symmetry -- otherwise periodicity-breaking
    prevents this.

    If ``bulk_structure`` is supplied and ``relaxed`` is set to ``False``, then
    returns the bulk site symmetry of the defect, which for
    vacancies/substitutions is the symmetry of the corresponding bulk site,
    while for interstitials it is the point symmetry of the `final relaxed`
    interstitial site when placed in the (unrelaxed) bulk structure.

    Args:
        structure (Structure):
            ``Structure`` object for which to determine the point symmetry.
        bulk_structure (Structure):
            ``Structure`` object of the bulk structure, if known. Default is
            ``None``. If provided and ``relaxed = True``, will be used to check
            if the supercell is breaking the crystal periodicity (and thus
            preventing accurate determination of the relaxed defect point
            symmetry) and warn you if so.
        symm_ops (list):
            List of symmetry operations of either the ``bulk_structure``
            structure (if ``relaxed=False``) or ``structure`` (if
            ``relaxed=True``), to avoid re-calculating. Default is ``None``
            (recalculates).
        symprec (float):
            Symmetry tolerance for ``spglib``. Default is 0.01 for unrelaxed
            structures, 0.1 for relaxed (to account for residual structural
            noise, matching that used by the ``Materials Project``). You may
            want to adjust for your system (e.g. if there are very slight
            octahedral distortions etc.).
        relaxed (bool):
            If ``False``, determines the site symmetry using the defect site
            `in the unrelaxed bulk supercell` (i.e. the bulk site symmetry),
            otherwise tries to determine the point symmetry of the relaxed
            defect in the defect supercell. Default is ``True``.
        verbose (bool):
            If ``True``, prints a warning if the supercell is detected to break
            the crystal periodicity (and hence not be able to return a reliable
            `relaxed` point symmetry). Default is ``True``.
        return_periodicity_breaking (bool):
            If ``True``, also returns a boolean specifying if the supercell has
            been detected to break the crystal periodicity (and hence not be
            able to return a reliable `relaxed` point symmetry) or not. Default
            is False.

    Returns:
        str:
            Structure point symmetry (and if
            ``return_periodicity_breaking = True``, a boolean specifying if the
            supercell has been detected to break the crystal periodicity).
    """
    if symprec is None:
        symprec = 0.1 if relaxed else 0.01  # relaxed structures likely have structural noise

    spglib_point_group_symbol = None
    if relaxed and bulk_structure is None:
        with contextlib.suppress(Exception):
            spglib_point_group_symbol = schoenflies_from_hermann(
                get_sga(structure, symprec=symprec).get_point_group_symbol()
            )
        if spglib_point_group_symbol is not None:
            return (
                (spglib_point_group_symbol, False)
                if return_periodicity_breaking
                else spglib_point_group_symbol
            )

    if bulk_structure is not None:
        from doped.analysis import defect_from_structures

        (
            defect,
            defect_site,
            defect_site_in_bulk,  # bulk site for vac/sub, relaxed defect site w/interstitials
            defect_site_index,  # in this initial_defect_structure
            bulk_site_index,
            guessed_initial_defect_structure,
            unrelaxed_defect_structure,
            _bulk_voronoi_node_dict,
        ) = defect_from_structures(
            bulk_structure,
            structure,
            oxi_state="Undetermined",
            return_all_info=True,
        )
        defect_entry = DefectEntry(
            # pmg attributes:
            defect=defect,  # this corresponds to _unrelaxed_ defect
            charge_state=0,
            sc_entry=ComputedStructureEntry(
                structure=structure,
                energy=0.0,  # needs to be set, so set to 0.0
            ),
            sc_defect_frac_coords=defect_site.frac_coords,  # _relaxed_ defect site
            bulk_entry=None,
            # doped attributes:
            defect_supercell_site=defect_site,  # _relaxed_ defect site
            defect_supercell=structure,
            bulk_supercell=bulk_structure,
        )

        return point_symmetry_from_defect_entry(
            defect_entry,
            symm_ops=symm_ops,
            symprec=symprec,
            relaxed=relaxed,
            verbose=verbose,
            return_periodicity_breaking=return_periodicity_breaking,
        )

    # else bulk structure is None and normal relaxed structure symmetry determination failed
    raise RuntimeError(
        "Target site symmetry could not be determined using just the input structure. Please also supply "
        "the unrelaxed bulk structure (`bulk_structure`)."
    )


def point_symmetry_from_site(
    site: PeriodicSite | np.ndarray | list,
    structure: Structure,
    coords_are_cartesian: bool = False,
    symm_ops: list | None = None,
    symprec: float = 0.01,
) -> str:
    r"""
    Get the point symmetry of a site in a structure.

    Args:
        site (Union[PeriodicSite, np.ndarray, list]):
            Site for which to determine the point symmetry. Can be a
            ``PeriodicSite`` object, or a list or numpy array of the
            coordinates of the site (fractional coordinates by default, or
            Cartesian if ``coords_are_cartesian = True``).
        structure (Structure):
            ``Structure`` object for which to determine the point symmetry of
            the site.
        coords_are_cartesian (bool):
            If ``True``, the site coordinates are assumed to be in Cartesian
            coordinates. Default is False.
        symm_ops (list):
            List of symmetry operations of the ``structure`` to avoid
            re-calculating. Default is ``None`` (recalculates).
        symprec (float):
            Symmetry tolerance for ``spglib``. Default is 0.01. You may want to
            adjust for your system (e.g. if there are very slight octahedral
            distortions etc.).

    Returns:
        str: Site point symmetry.
    """
    if isinstance(site, np.ndarray | list):
        site = PeriodicSite(
            species="X", coords=site, lattice=structure.lattice, coords_are_cartesian=coords_are_cartesian
        )

    try:
        symm_dataset, _unique_sites = _get_symm_dataset_of_struc_with_all_equiv_sites(
            site.frac_coords, structure, symm_ops=symm_ops, symprec=symprec, species=site.species_string
        )
    except SymmetryUndeterminedError:
        symm_dataset, _unique_sites = _get_symm_dataset_of_struc_with_all_equiv_sites(
            site.frac_coords, structure, symm_ops=symm_ops, symprec=symprec, species="X"
        )

    return schoenflies_from_hermann(symm_dataset.site_symmetry_symbols[-1])


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


def get_interstitial_site_and_orientational_degeneracy(
    interstitial_defect_entry: DefectEntry, dist_tol: float = 0.15
) -> int:
    """
    Get the combined site and orientational degeneracy of an interstitial
    defect entry.

    The standard approach of using ``_get_equiv_sites()`` for interstitial site
    multiplicity and then ``point_symmetry_from_defect_entry()`` &
    ``get_orientational_degeneracy`` for symmetry/orientational degeneracy is
    preferred (as used in the ``DefectParser`` code), but alternatively this
    function can be used to compute the product of the site and orientational
    degeneracies.

    This is done by determining the number of equivalent sites in the bulk
    supercell for the given interstitial site (from defect_supercell_site),
    which gives the combined site and orientational degeneracy `if` there was
    no relaxation of the bulk lattice atoms. This matches the true combined
    degeneracy in most cases, except for split-interstitial type defects etc.,
    where this would give an artificially high degeneracy (as, for example, the
    interstitial site is automatically assigned to one of the split-
    interstitial atoms and not the midpoint, giving a doubled degeneracy as it
    considers the two split-interstitial sites as two separate (degenerate)
    interstitial sites, instead of one). This is counteracted by dividing by
    the number of sites which are present in the defect supercell (within a
    distance tolerance of dist_tol in Å) with the same species, ensuring none
    of the predicted `different` equivalent sites are actually `included` in
    the defect structure.

    Args:
        interstitial_defect_entry:
            ``DefectEntry`` object for the interstitial defect.
        dist_tol:
            Distance tolerance in Å for determining equivalent sites.

    Returns:
        int:
            Combined site and orientational degeneracy of the interstitial
            defect entry.
    """
    if interstitial_defect_entry.bulk_entry is None:
        raise ValueError(
            "bulk_entry must be set for interstitial_defect_entry to determine the site and orientational "
            "degeneracies! (i.e. must be a parsed DefectEntry)"
        )
    equiv_sites = get_all_equiv_sites(
        _get_defect_supercell_frac_coords(interstitial_defect_entry),
        _get_bulk_supercell(interstitial_defect_entry),
    )
    equiv_sites_array = np.array([site.frac_coords for site in equiv_sites])
    defect_supercell_sites_of_same_species_array = np.array(
        [
            site.frac_coords
            for site in _get_defect_supercell(interstitial_defect_entry)
            if site.specie.symbol == interstitial_defect_entry.defect.site.specie.symbol
        ]
    )

    distance_matrix = _get_bulk_supercell(interstitial_defect_entry).lattice.get_all_distances(
        defect_supercell_sites_of_same_species_array[:, None], equiv_sites_array
    )[:, 0]

    return len(equiv_sites) // len(distance_matrix[distance_matrix < dist_tol])


def get_orientational_degeneracy(
    defect_entry: DefectEntry | None = None,
    relaxed_point_group: str | None = None,
    bulk_site_point_group: str | None = None,
    bulk_symm_ops: list | None = None,
    defect_symm_ops: list | None = None,
    symprec: float = 0.1,
) -> float:
    r"""
    Get the orientational degeneracy factor for a given `relaxed`
    ``DefectEntry``, by supplying either the ``DefectEntry`` object or the
    bulk-site & relaxed defect point group symbols (e.g. "Td", "C3v" etc.).

    If a ``DefectEntry`` is supplied (and the point group symbols are not),
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

    Note: This tries to use the ``defect_entry.defect_supercell`` to determine
    the `relaxed` site symmetry. However, it should be noted that this is not
    guaranteed to work in all cases; namely for non-diagonal supercell
    expansions, or sometimes for non-scalar supercell expansion matrices (e.g.
    a 2x1x2 expansion)(particularly with high-symmetry materials) which can
    mess up the periodicity of the cell. ``doped`` tries to automatically check
    if this is the case, and will warn you if so.

    This can also be checked by using this function on your doped `generated`
    defects:

    .. code-block:: python

        from doped.generation import get_defect_name_from_entry
        for defect_name, defect_entry in defect_gen.items():
            print(defect_name,
                  get_defect_name_from_entry(defect_entry, relaxed=False),
                  get_defect_name_from_entry(defect_entry), "\n")

    And if the point symmetries match in each case, then using this function on
    your parsed `relaxed` ``DefectEntry`` objects should correctly determine
    the final relaxed defect symmetry (and orientational degeneracy) --
    otherwise periodicity-breaking prevents this.

    If periodicity-breaking prevents auto-symmetry determination, you can
    manually determine the relaxed defect and bulk-site point symmetries,
    and/or orientational degeneracy, from visualising the structures (e.g.
    using VESTA)(can use ``get_orientational_degeneracy`` to obtain the
    corresponding orientational degeneracy factor for given defect/bulk-site
    point symmetries) and setting the corresponding values in the
    ``calculation_metadata['relaxed point symmetry']/['bulk site symmetry']``
    and/or ``degeneracy_factors['orientational degeneracy']`` attributes. Note
    that the bulk-site point symmetry corresponds to that of
    ``DefectEntry.defect``, or equivalently
    ``calculation_metadata["bulk_site"]/["unrelaxed_defect_structure"]``, which
    for vacancies/substitutions is the symmetry of the corresponding bulk site,
    while for interstitials it is the point symmetry of the `final relaxed`
    interstitial site when placed in the (unrelaxed) bulk structure. The
    degeneracy factor is used in the calculation of defect/carrier
    concentrations and Fermi level behaviour (see e.g.
    https://doi.org/10.1039/D2FD00043A & https://doi.org/10.1039/D3CS00432E).

    Args:
        defect_entry (DefectEntry):
            ``DefectEntry`` object. (Default = None)
        relaxed_point_group (str):
            Point group symmetry (e.g. "Td", "C3v" etc.) of the `relaxed`
            defect structure, if already calculated / manually determined.
            Default is ``None`` (automatically calculated by ``doped``).
        bulk_site_point_group (str):
            Point group symmetry (e.g. "Td", "C3v" etc.) of the defect site in
            the bulk, if already calculated / manually determined. For
            vacancies/substitutions, this should match the site symmetry label
            from ``doped`` when generating the defect, while for interstitials
            it should be the point symmetry of the `final relaxed` interstitial
            site, when placed in the bulk structure.
            Default is ``None`` (automatically calculated by ``doped``).
        bulk_symm_ops (list):
            List of symmetry operations of the ``defect_entry.bulk_supercell``
            structure (used in determining the `unrelaxed` bulk site symmetry),
            to avoid re-calculating. Default is ``None`` (recalculates).
        defect_symm_ops (list):
            List of symmetry operations of ``defect_entry.defect_supercell``
            (used in determining the `relaxed` point symmetry), to avoid
            re-calculating. Default is ``None`` (recalculates).
        symprec (float):
            Symmetry tolerance for ``spglib`` to use when determining point
            symmetries and thus orientational degeneracies. Default is ``0.1``
            which matches that used by the ``Materials Project`` and is larger
            than the ``pymatgen`` default of ``0.01`` to account for residual
            structural noise in relaxed defect supercells. You may want to
            adjust for your system (e.g. if there are very slight octahedral
            distortions etc.).

    Returns:
        float: Orientational degeneracy factor for the defect.
    """
    if defect_entry is None:
        if relaxed_point_group is None or bulk_site_point_group is None:
            raise ValueError(
                "Either the DefectEntry or both defect and bulk site point group symbols must be "
                "provided for doped to determine the orientational degeneracy! "
            )

    elif defect_entry.bulk_entry is None:
        raise ValueError(
            "bulk_entry must be set for defect_entry to determine the (relaxed) orientational degeneracy! "
            "(i.e. must be a parsed DefectEntry)"
        )

    if relaxed_point_group is None:
        # this will throw warning if auto-detected that supercell breaks trans symmetry
        relaxed_point_group = point_symmetry_from_defect_entry(
            defect_entry,  # type: ignore
            symm_ops=defect_symm_ops,  # defect not bulk symm_ops
            symprec=symprec,
            relaxed=True,  # relaxed
        )

    if bulk_site_point_group is None:
        bulk_site_point_group = point_symmetry_from_defect_entry(
            defect_entry,  # type: ignore
            symm_ops=bulk_symm_ops,  # bulk not defect symm_ops
            symprec=symprec,  # same symprec as relaxed_point_group for consistency
            relaxed=False,  # unrelaxed
        )

    # actually fine for split-vacancies (e.g. Ke's V_Sb in Sb2O5), or antisite-swaps etc:
    # (so avoid warning for now; user will be warned anyway if symmetry determination failing)
    # if orientational_degeneracy < 1 and not (
    #     defect_type == DefectType.Interstitial
    #     or (isinstance(defect_type, str) and defect_type.lower() == "interstitial")
    # ):
    #     raise ValueError(
    #         f"From the input/determined point symmetries, an orientational degeneracy factor of "
    #         f"{orientational_degeneracy} is predicted, which is less than 1, which is not reasonable "
    #         f"for vacancies/substitutions, indicating an error in the symmetry determination!"
    #     )

    return group_order_from_schoenflies(bulk_site_point_group) / group_order_from_schoenflies(
        relaxed_point_group
    )


def are_equivalent_sites(
    s1: PeriodicSite, s2: PeriodicSite, structure: Structure, symprec: float = 0.01
) -> bool:
    """
    Check if two periodic sites are symmetry-equivalent in the given
    ``structure``.

    Symmetry-equivalency is checked using the ``SymmetrizedStructure`` class
    for vacancies and substitutions, or using the equivalent sites for
    interstitials, along with ``doped`` efficiency functions for fast
    comparisons.

    This matches the approach taken for determining defect site multiplicities
    (used in concentrations) and in ``group_defects_by_distance`` used in
    ``doped.thermodynamics``.

    Args:
        s1 (PeriodicSite):
            First periodic site.
        s2 (PeriodicSite):
            Second periodic site.
        structure (Structure):
            Structure in which to check for symmetry-equivalency between the
            two input sites.
        symprec (float):
            Symmetry tolerance for ``spglib``. Default is 0.01.

    Returns:
        bool:
            ``True`` if the sites are symmetry-equivalent, ``False`` otherwise.
    """
    from doped.utils.efficiency import Composition as doped_Composition
    from doped.utils.efficiency import PeriodicSite as doped_PeriodicSite

    # use doped efficiency tools to make structure-matching as fast as possible:
    Composition.__instances__ = {}
    Composition.__eq__ = doped_Composition.__eq__
    Composition.__hash__ = doped_Composition.__hash__
    PeriodicSite.__eq__ = doped_PeriodicSite.__eq__
    PeriodicSite.__hash__ = doped_PeriodicSite.__hash__

    host_struct = structure.copy()
    if isinstance(s1.specie, Element) or isinstance(s2.specie, Element):
        s1._specie = Element(s1.specie)
        s2._specie = Element(s2.specie)
        host_struct.remove_oxidation_states()

    bulk_sga = get_sga(host_struct, symprec=symprec)
    try:
        symm_struct = bulk_sga.get_symmetrized_structure()  # could cache this for speed if needed?
        return s2 in symm_struct.find_equivalent_sites(s1)

    except ValueError:  # s1 not in symm_struct, possibly interstitial
        if s2 in symm_struct:  # s2 not an interstitial though, return False
            return False

        s1_equiv_sites = get_all_equiv_sites(
            s1.frac_coords,
            host_struct,
            symm_ops=bulk_sga.get_symmetry_operations(),
            symprec=symprec,
            dist_tol=symprec,
            species=s1.species_string,
        )
        return s2 in s1_equiv_sites
